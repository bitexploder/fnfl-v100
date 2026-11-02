#include "fn/runtime/ple_host.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/embed/embed.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"

#include <cuda_runtime_api.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

const char * const arch_prefix = "qwen4exp.ple.";

// PN1: the page range ple_pin holds pinned, [first, end); 0, 0 when none. Published by ple_pin before its owner hands
// the table to a gather, cleared before it unregisters; read by ple_prefetch on every round, so a pinned table costs
// two loads and no system call. Not a resource: the pin is owned by the model's ple_pin.
std::atomic<uintptr_t> pinned_first{0};
std::atomic<uintptr_t> pinned_end{0};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool read_array(const std::string & path, const char * key, size_t n, std::vector<int64_t> & out, std::string & err) {
    if (!fnfl_gguf_read_ints(path, std::string(arch_prefix) + key, out) || out.size() != n) {
        err = std::string("ple.") + key + ": missing, not integers or not " + std::to_string(n) + " values";
        return false;
    }
    return true;
}

bool read_scalar(const std::string & path, const char * key, int64_t want, int64_t & got, std::string & err) {
    got = fnfl_gguf_read_int(path, std::string(arch_prefix) + key);
    if (want >= 0 && got != want) {
        err = std::string("ple.") + key + " is " + std::to_string(got) + ", FNFL is built for " + std::to_string(want);
        return false;
    }
    return true;
}

} // namespace

bool read_ple_params(const std::string & first_shard, ple_params & p, std::string & err) {
    p = ple_params();
    std::vector<int64_t> layers, mult, offset, vocab;
    int64_t ngram = 0, heads = 0, kern = 0, eos = 0, image = 0;
    if (!read_array(first_shard, "layers", 1, layers, err) ||
        !read_scalar(first_shard, "ngram_size", ple_ngram, ngram, err) ||
        !read_scalar(first_shard, "heads_per_ngram", ple_heads_per_ngram, heads, err) ||
        !read_scalar(first_shard, "conv_kernel", ple_conv_kernel, kern, err) ||
        !read_scalar(first_shard, "eos_token_id", ple_eos, eos, err) ||
        !read_scalar(first_shard, "image_token_id", -1, image, err) ||
        !read_array(first_shard, "layer_multipliers", ple_ngram, mult, err) ||
        !read_array(first_shard, "head_offsets", ple_heads, offset, err) ||
        !read_array(first_shard, "head_vocab_sizes", ple_heads, vocab, err)) {
        return false;
    }
    if (layers[0] != ple_layer) {
        err = "ple.layers is [" + std::to_string(layers[0]) + "], FNFL is built for [1]";
        return false;
    }
    if (image < 0 || image >= fn::embed::n_vocab) {
        err = "ple.image_token_id " + std::to_string(image) + " is missing or outside the vocabulary";
        return false;
    }
    // the table's shape, from whichever shard holds it
    std::vector<std::string> paths;
    if (!fnfl_gguf_shard_paths(first_shard, paths, err)) {
        return false;
    }
    for (const std::string & path : paths) {
        fnfl_gguf_dir dir;
        if (!fnfl_gguf_read_dir(path, dir, err)) {
            return false;
        }
        for (const fnfl_gguf_tensor & t : dir.tensors) {
            if (t.name == "per_layer_token_embd.weight") {
                if (t.type_name != "iq4_nl" || t.ne[0] != ple_head_dim || t.ne[2] != 1 ||
                    t.bytes != uint64_t(t.ne[1]) * ple_row_bytes) {
                    err = "per_layer_token_embd.weight is not IQ4_NL with 160 values a row";
                    return false;
                }
                p.table_rows = t.ne[1];
            }
        }
    }
    if (p.table_rows == 0) {
        err = "per_layer_token_embd.weight is in no shard";
        return false;
    }
    p.eos = static_cast<int32_t>(eos);
    p.image = static_cast<int32_t>(image);
    for (int32_t j = 0; j < ple_ngram; ++j) {
        p.mult[j] = static_cast<uint64_t>(mult[size_t(j)]); // the reader widened UINT64 to int64 bit for bit
    }
    for (int32_t h = 0; h < ple_heads; ++h) {
        const int64_t o = offset[size_t(h)], v = vocab[size_t(h)];
        if (o < 0 || v < 1 || o > UINT32_MAX || v > UINT32_MAX || o + v > p.table_rows) {
            err = "ple head " + std::to_string(h) + ": offset " + std::to_string(o) + " and vocabulary " +
                  std::to_string(v) + " leave the table's " + std::to_string(p.table_rows) + " rows";
            return false;
        }
        p.offset[h] = static_cast<uint32_t>(o);
        p.vocab[h] = static_cast<uint32_t>(v);
    }
    return true;
}

bool ple_residency_parse(const std::string & s, ple_residency & r) {
    if (s == "lazy" || s == "full") {
        r = s == "lazy" ? ple_residency::lazy : ple_residency::full;
        return true;
    }
    return false;
}

const char * ple_residency_name(ple_residency r) {
    return r == ple_residency::lazy ? "lazy" : "full";
}

int64_t ple_foll_pinned_pages() {
    std::ifstream f("/proc/vmstat");
    std::string k;
    int64_t v = 0, acquired = -1, released = -1;
    while (f >> k >> v) {
        acquired = k == "nr_foll_pin_acquired" ? v : acquired;
        released = k == "nr_foll_pin_released" ? v : released;
    }
    return acquired < 0 || released < 0 ? -1 : acquired - released;
}

void ple_pinned_range(uintptr_t & first, uintptr_t & end) {
    first = pinned_first.load(std::memory_order_acquire);
    end = pinned_end.load(std::memory_order_acquire);
}

ple_pin::~ple_pin() {
    release();
}

bool ple_pin::populate(const uint8_t * table, uint64_t bytes, std::string & err) {
    release();
    const auto t0 = std::chrono::steady_clock::now();
    const long page_l = sysconf(_SC_PAGESIZE);
    if (table == nullptr || bytes == 0 || page_l <= 0 || (page_l & (page_l - 1)) != 0) {
        err = "PLE table pin: no table, or page size " + std::to_string(page_l) +
              " (--fnfl-ple-table lazy keeps the "
              "table unpinned)";
        return false;
    }
    const uintptr_t page = static_cast<uintptr_t>(page_l);
    // reason: page arithmetic on the mapping's addresses
    const uintptr_t first = reinterpret_cast<uintptr_t>(table) & ~(page - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(table) + bytes + page - 1) & ~(page - 1);
    // reason: madvise and cudaHostRegister take the range as a mutable pointer; neither writes it
    a_ = reinterpret_cast<uint8_t *>(first);
    n_ = uint64_t(end - first);
    if (madvise(a_, n_, MADV_POPULATE_READ) != 0) {
        err = std::string("PLE table pin: madvise(MADV_POPULATE_READ) of ") + std::to_string(n_) +
              " bytes: " + std::strerror(errno) + " (--fnfl-ple-table lazy keeps the table unpinned)";
        a_ = nullptr;
        n_ = 0;
        return false;
    }
    populated_ = true;
    populate_s = seconds_since(t0);
    return true;
}

bool ple_pin::pin(std::string & err) {
    if (!populated_ || pinned_) {
        err = "PLE table pin: cudaHostRegister before populate, or twice";
        return false;
    }
    foll_pin_before = ple_foll_pinned_pages();
    const auto t0 = std::chrono::steady_clock::now();
    const cudaError_t e = cudaHostRegister(a_, n_, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    register_s = seconds_since(t0);
    foll_pin_after = ple_foll_pinned_pages();
    if (e != cudaSuccess) {
        cudaGetLastError(); // a refused registration is not sticky; clear it for the caller's next check
        err = std::string("PLE table pin: cudaHostRegister(Portable | ReadOnly) of ") + std::to_string(n_) +
              " bytes: " + cudaGetErrorString(e) + " (--fnfl-ple-table lazy keeps the table unpinned)";
        return false;
    }
    pinned_ = true;
    // reason: the published range is compared as addresses
    pinned_first.store(reinterpret_cast<uintptr_t>(a_), std::memory_order_release);
    pinned_end.store(reinterpret_cast<uintptr_t>(a_) + uintptr_t(n_), std::memory_order_release);
    return true;
}

void ple_pin::release() {
    if (pinned_) {
        pinned_first.store(0, std::memory_order_release);
        pinned_end.store(0, std::memory_order_release);
        cudaHostUnregister(a_); // at teardown; a failure (the runtime already unloading at exit) leaves nothing to do
    }
    a_ = nullptr;
    n_ = 0;
    populated_ = false;
    pinned_ = false;
}

int64_t ple_resident_pages(const uint8_t * p, uint64_t n) {
    const long page_l = sysconf(_SC_PAGESIZE);
    if (p == nullptr || n == 0 || page_l <= 0) {
        return -1;
    }
    const uintptr_t page = static_cast<uintptr_t>(page_l);
    // reason: page arithmetic on the mapping's addresses
    const uintptr_t first = reinterpret_cast<uintptr_t>(p) & ~(page - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(p) + n + page - 1) & ~(page - 1);
    std::vector<unsigned char> v((end - first) / page);
    // reason: mincore takes the range as a mutable pointer; it writes nothing there
    if (mincore(reinterpret_cast<void *>(first), size_t(end - first), v.data()) != 0) {
        return -1;
    }
    int64_t r = 0;
    for (const unsigned char c : v) {
        r += c & 1;
    }
    return r;
}

namespace {

// a "Key:   N kB" line of /proc/meminfo, in bytes; -1 when absent
int64_t meminfo_bytes(const char * key) {
    std::ifstream f("/proc/meminfo");
    std::string k;
    int64_t v = 0;
    std::string unit;
    const std::string want = std::string(key) + ":";
    while (f >> k >> v) {
        std::getline(f, unit);
        if (k == want) {
            return v * 1024;
        }
    }
    return -1;
}

double gib(int64_t b) {
    return double(b) / 1073741824.0;
}

} // namespace

ple_warm::~ple_warm() {
    stop();
    if (r_.locked) {
        munlock(a_, n_);
    }
}

bool ple_warm::start(const uint8_t * table, uint64_t bytes, bool log, bool lock) {
    const long page_l = sysconf(_SC_PAGESIZE);
    if (th_.joinable() || done() || table == nullptr || bytes == 0 || page_l <= 0) {
        return false;
    }
    const uintptr_t page = static_cast<uintptr_t>(page_l);
    // reason: page arithmetic on the mapping's addresses
    const uintptr_t first = reinterpret_cast<uintptr_t>(table) & ~(page - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(table) + bytes + page - 1) & ~(page - 1);
    // reason: madvise, mlock and mincore take the range as a mutable pointer; none writes it
    a_ = reinterpret_cast<uint8_t *>(first);
    n_ = uint64_t(end - first);
    log_ = log;
    lock_ = lock;
    r_ = report();
    r_.bytes = n_;
    r_.pages = int64_t(n_ / page);
    if (log_) {
        fprintf(stderr,
                "FNFL: PLE warm: reading the lazy table's %.2f GiB into the page cache on a background thread (STALL; "
                "--fnfl-ple-warm off disables it); the server does not wait\n",
                gib(int64_t(n_)));
        fflush(stderr);
    }
    th_ = std::thread([this] { run(); });
    return true;
}

void ple_warm::stop() {
    stop_.store(true, std::memory_order_release);
    if (th_.joinable()) {
        th_.join();
    }
}

void ple_warm::run() {
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = true;
    // the next piece's readahead is queued before this piece is mapped: POPULATE_READ alone faults with the small
    // fault-path readahead, WILLNEED reads a piece in large requests
    madvise(a_, std::min(piece_bytes, n_), MADV_WILLNEED);
    for (uint64_t off = 0; ok && off < n_; off += piece_bytes) {
        if (stop_.load(std::memory_order_acquire)) {
            r_.stopped = true;
            break;
        }
        const uint64_t len = std::min(piece_bytes, n_ - off);
        if (off + len < n_) {
            madvise(a_ + off + len, std::min(piece_bytes, n_ - off - len), MADV_WILLNEED);
        }
        if (madvise(a_ + off, len, MADV_POPULATE_READ) != 0) {
            r_.err = std::string("madvise(MADV_POPULATE_READ) at ") + std::to_string(off) + ": " + std::strerror(errno);
            ok = false;
        }
    }
    r_.populated = ok && !r_.stopped;
    r_.populate_s = seconds_since(t0);
    struct rlimit rl = {};
    getrlimit(RLIMIT_MEMLOCK, &rl);
    r_.memlock_limit = uint64_t(rl.rlim_cur);
    r_.mem_total = meminfo_bytes("MemTotal");
    r_.mem_available = meminfo_bytes("MemAvailable");
    // the table's own cached pages count in MemAvailable; locking takes them out of it
    const bool room = r_.mem_available >= 0 && uint64_t(r_.mem_available) >= n_ + lock_margin;
    if (r_.populated && lock_ && room) {
        const auto t1 = std::chrono::steady_clock::now();
        if (mlock(a_, n_) == 0) {
            r_.locked = true;
        } else {
            r_.err = std::string("mlock: ") + std::strerror(errno);
        }
        r_.lock_s = seconds_since(t1);
    }
    r_.resident = ple_resident_pages(a_, n_);
    r_.total_s = seconds_since(t0);
    done_.store(true, std::memory_order_release);
    if (log_) {
        fprintf(stderr, "FNFL: PLE warm: %s\n", line().c_str());
        fflush(stderr);
    }
}

std::string ple_warm::line() const {
    if (!done()) {
        return "in progress";
    }
    const report & r = r_;
    const bool full = r.resident == r.pages;
    const std::string lim = r.memlock_limit == uint64_t(RLIM_INFINITY) ? std::string("unlimited") :
                                                                          std::to_string(r.memlock_limit) + " bytes";
    const bool room = r.mem_available >= 0 && uint64_t(r.mem_available) >= r.bytes + lock_margin;
    const char * lock_why = r.locked    ? "locked (mlock)" :
                            !lock_      ? "not locked (lock disabled)" :
                            !r.populated ? "not locked (populate incomplete)" :
                            !room       ? "not locked (MemAvailable less the table is under the 16 GiB margin)" :
                                          "not locked (mlock refused)";
    char b[640];
    snprintf(b, sizeof b,
             "the table is %s: %lld of %lld pages (%.1f%%) of %.2f GiB after %.1f s (populate %.1f s, lock %.1f s); "
             "%s; MemTotal %.1f GiB, MemAvailable %.1f GiB before the lock, RLIMIT_MEMLOCK %s%s%s%s",
             full ? "resident 100%" : "partly resident", static_cast<long long>(r.resident),
             static_cast<long long>(r.pages), r.pages > 0 ? 100.0 * double(r.resident) / double(r.pages) : 0.0,
             gib(int64_t(r.bytes)), r.total_s, r.populate_s, r.lock_s, lock_why, gib(r.mem_total),
             gib(r.mem_available), lim.c_str(), r.stopped ? "; stopped before the end" : "",
             r.err.empty() ? "" : "; ", r.err.c_str());
    return b;
}

// set_input's loop body for one row (qwen4exp.cpp 1934-1957), with the predecessors passed in (-1: missing)
void ple_row_indices(const ple_params & p, int32_t t0, int32_t t1, int32_t t2, int32_t (&idx)[ple_heads]) {
    const int32_t pred[ple_ngram] = {t0, t1, t2};
    int64_t ctx[ple_ngram];
    ctx[0] = t0;
    bool cut = false;
    for (int32_t s = 1; s < ple_ngram; ++s) {
        const int32_t t = cut ? -1 : pred[s];
        cut = cut || t < 0 || t == p.eos;
        ctx[s] = cut ? p.eos : t;
    }
    for (int32_t n = 2; n <= ple_ngram; ++n) {
        uint64_t mixed = static_cast<uint64_t>(ctx[0]) * p.mult[0];
        for (int32_t j = 1; j < n; ++j) {
            mixed ^= static_cast<uint64_t>(ctx[j]) * p.mult[j];
        }
        const int32_t base = (n - 2) * ple_heads_per_ngram;
        for (int32_t g = 0; g < ple_heads_per_ngram; ++g) {
            const int32_t h = base + g;
            idx[h] = static_cast<int32_t>(mixed % p.vocab[h] + p.offset[h]); // below table_rows < 2^31 (checked)
        }
    }
}

bool ple_indices(const ple_params & p, const int32_t * ledger, int64_t n_before, const int32_t * tok, int32_t rows,
                 int32_t * idx) {
    if (tok == nullptr || idx == nullptr || n_before < 0 || (n_before > 0 && ledger == nullptr) ||
        !fn::embed::check_tokens(tok, rows)) {
        return false;
    }
    // the token at position n_before + d (d < 0: before row 0), -1 before position 0
    auto at = [&](int64_t d) -> int32_t { return d >= 0 ? tok[d] : n_before + d >= 0 ? ledger[n_before + d] : -1; };
    for (int32_t t = 0; t < rows; ++t) {
        int32_t row[ple_heads];
        ple_row_indices(p, tok[t], at(int64_t(t) - 1), at(int64_t(t) - 2), row);
        for (int32_t h = 0; h < ple_heads; ++h) {
            idx[t * ple_heads + h] = row[h];
        }
    }
    return true;
}

bool ple_gather(const ple_params & p, const uint8_t * table, const int32_t * idx, int32_t rows, float * e) {
    if (table == nullptr || idx == nullptr || e == nullptr || rows < 1 || rows > ple_max_rows) {
        return false;
    }
    for (int32_t t = 0; t < rows; ++t) {
        for (int32_t h = 0; h < ple_heads; ++h) {
            const int32_t r = idx[t * ple_heads + h];
            if (r < 0 || r >= p.table_rows ||
                !fn::quant::decode_row(fn::quant::wfmt::iq4_nl, table + int64_t(r) * ple_row_bytes,
                                       e + int64_t(t) * ple_input + int64_t(h) * ple_head_dim, ple_head_dim)) {
                return false;
            }
        }
    }
    return true;
}

bool ple_prefetch(const ple_params & p, const uint8_t * table, const int32_t * idx, int32_t rows, int32_t & n_ranges,
                  std::string & err) {
    n_ranges = 0;
    if (table == nullptr || idx == nullptr || rows < 1 || rows > ple_max_rows) {
        err = "PLE prefetch: null pointer or rows outside 1..5";
        return false;
    }
    // reason: the table's address against the pinned range (PN1: pinned pages need no hint)
    const uintptr_t t = reinterpret_cast<uintptr_t>(table);
    if (t >= pinned_first.load(std::memory_order_acquire) && t < pinned_end.load(std::memory_order_acquire)) {
        return true;
    }
    static const long page_l = sysconf(_SC_PAGESIZE);
    if (page_l <= 0 || (page_l & (page_l - 1)) != 0 || page_l < ple_row_bytes) {
        err = "PLE prefetch: page size " + std::to_string(page_l);
        return false;
    }
    const uintptr_t page = static_cast<uintptr_t>(page_l);
    // reason: page arithmetic on the mapping's addresses
    const uintptr_t base = reinterpret_cast<uintptr_t>(table);
    std::array<uintptr_t, size_t(ple_max_rows) * ple_heads * 2> pages; // page numbers; a row spans at most two
    size_t n = 0;
    for (int32_t i = 0; i < rows * ple_heads; ++i) {
        const int32_t r = idx[i];
        if (r < 0 || r >= p.table_rows) {
            err = "PLE prefetch: row " + std::to_string(r) + " outside the table";
            return false;
        }
        const uintptr_t first = base + uintptr_t(r) * uintptr_t(ple_row_bytes);
        const uintptr_t last = first + uintptr_t(ple_row_bytes) - 1;
        for (uintptr_t pg = first / page; pg <= last / page; ++pg) {
            pages[n++] = pg;
        }
    }
    std::sort(pages.begin(), pages.begin() + std::ptrdiff_t(n));
    for (size_t i = 0; i < n;) {
        size_t j = i + 1;
        while (j < n && pages[j] <= pages[j - 1] + 1) { // equal or adjacent
            ++j;
        }
        // reason: posix_madvise takes the range's address as a mutable pointer; the advice writes nothing
        void * a = reinterpret_cast<void *>(pages[i] * page);
        const int rc = posix_madvise(a, size_t(pages[j - 1] - pages[i] + 1) * page, POSIX_MADV_WILLNEED);
        if (rc != 0) {
            err = std::string("PLE prefetch: posix_madvise(WILLNEED): ") + std::strerror(rc);
            return false;
        }
        ++n_ranges;
        i = j;
    }
    return true;
}

} // namespace fn::runtime
