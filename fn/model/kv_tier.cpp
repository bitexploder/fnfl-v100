#include "fn/model/kv_tier.h"

#include "fn/kernels/qsa/kv_tier.h"
#include "fn/model/model.h"
#include "fn/model/shapes.h"
#include "fn/model/state.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace fn::model {
namespace {

namespace tier = fn::qsa::tier;

constexpr int64_t cell_bytes = int64_t(tier::cell_values) * 2;
constexpr int64_t key_cell_bytes = int64_t(fn::qsa::head_dim) * 2; // ME6: a cell's raw index key, BF16
static_assert(kv_tier::prompt_key_cells == prefill_stage_rows + fn::qsa::ratio, "ME6: a chunk and its first block");
static_assert(kv_tier::key_slots == n_index_owned + 1, "ME6: the owned target layers and the draft");

bool fail(std::string & err, const std::string & what) {
    err = "KV tier: " + what;
    return false;
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string("KV tier: ") + what + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// MEKV: restore the caller's current device on both success and failure.
template <typename F> bool on_device(int32_t device, std::string & err, F fn) {
    int previous = -1;
    if (!cuda_ok(cudaGetDevice(&previous), "current device", err) ||
        !cuda_ok(cudaSetDevice(device), "shard device", err)) {
        return false;
    }
    const bool ok = fn();
    const bool restored = cuda_ok(cudaSetDevice(previous), "restore device", err);
    return ok && restored;
}

uint8_t * class_ptr(const model & m, const char * cls, const std::string & name, int32_t card, int64_t & bytes,
               std::string & err) {
    const reserved_class * rc = m.state_class(cls);
    for (int32_t i = 0; rc != nullptr && i < int32_t(rc->plan.parts.size()); ++i) {
        if (rc->plan.parts[size_t(i)].name == name) {
            bytes = rc->plan.parts[size_t(i)].bytes[card];
            if (uint8_t * p = rc->part_ptr(i, card)) {
                return p;
            }
        }
    }
    err = std::string("KV tier: ") + cls + " / " + name + ": not reserved on card " + std::to_string(card);
    return nullptr;
}

constexpr int64_t al(int64_t n) {
    return (n + 255) / 256 * 256;
}

// the tables part: table, list, count and total, the two bodies; ME3: the window's first cell (graph mode)
constexpr int64_t table_at = 0;
constexpr int64_t list_at = al(int64_t(fn::qsa::max_blocks) * 4);
constexpr int64_t count_at = list_at + al(int64_t(tier::max_gather) * 4);
constexpr int64_t total_at = count_at + 256;
constexpr int64_t body_at = total_at + 256;
constexpr int64_t lo_at = body_at + 2 * al(int64_t(tier::max_body_words) * 4);
constexpr int64_t tables_end = lo_at + 256;

uint16_t * cells_at(uint16_t * base, int64_t cell) {
    return reinterpret_cast<uint16_t *>(reinterpret_cast<uintptr_t>(base) + uintptr_t(cell * cell_bytes));
}

uint16_t * key_cells_at(uint16_t * base, int64_t cell) {
    return reinterpret_cast<uint16_t *>(reinterpret_cast<uintptr_t>(base) + uintptr_t(cell * key_cell_bytes));
}

// the launches read the history by the host pointer (UVA: a mapped pinned allocation has one address)
bool mapped_same(void * h, std::string & err) {
    void * d = nullptr;
    if (!cuda_ok(cudaHostGetDevicePointer(&d, h, 0), "host history (mapped)", err)) {
        return false;
    }
    return d == h ? true : fail(err, "the host history's device address differs from its host address");
}

} // namespace

int64_t kv_tier::arena_cells(int32_t window, int64_t n_cells) {
    return std::max(int64_t(n_attn) * window + tier::gather_cells, n_cells);
}

int64_t kv_tier::table_bytes() {
    return tables_end;
}

int64_t kv_tier::key_arena_cells(int32_t window) {
    return std::max(int64_t(key_slots) * window, int64_t(prompt_key_cells));
}

int32_t kv_tier::key_slot(int32_t c, int32_t a) {
    if (a == draft_attn) {
        return key_draft;
    }
    return a < 0 || a >= draft_attn || a % 2 != c ? -1 : a / 2;
}

bool kv_tier::init(const model & m, int32_t window, std::string & err, void * const * streams) {
    if (!m.kv_host) {
        return fail(err, "the host-backed KV (--kv-host) must be enabled");
    }
    if (window < 64 || window % 4 != 0) {
        return fail(err, "a window of " + std::to_string(window) + " cells");
    }
    w_ = window;
    for (int32_t c = 0; c < 2; ++c) {
        device_[c] = m.device[c];
        streams_[c] = streams == nullptr ? nullptr : streams[c];
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!on_device(device_[c], err, [&] {
            card & k = cards_[c];
            int64_t bytes = 0;
            for (int32_t a = 0; a < n_attn; ++a) {
                const char * cls = a == draft_attn ? "draft KV (F16)" : "target KV (K0)";
                const std::string L = a == draft_attn ? "L48" : "L" + std::to_string(
                    a * fn::full_attn_interval + fn::full_attn_interval - 1);
                k.host_k[a] = reinterpret_cast<uint16_t *>(class_ptr(m, cls, L + " K", c, bytes, err));
                k.host_v[a] = reinterpret_cast<uint16_t *>(class_ptr(m, cls, L + " V", c, bytes, err));
                if (k.host_k[a] == nullptr || k.host_v[a] == nullptr) {
                    return false;
                }
                const int64_t cells = bytes / cell_bytes;
                if (a == 0) {
                    n_cells_ = cells;
                } else if (cells != n_cells_) {
                    return fail(err, L + ": " + std::to_string(cells) + " cells, not the slot's " +
                                         std::to_string(n_cells_));
                }
                if (!mapped_same(k.host_k[a], err) || !mapped_same(k.host_v[a], err)) {
                    return false;
                }
                // ME6: the layer's index keys, where this card holds them
                const int32_t j = key_slot(c, a);
                if (j < 0) {
                    continue;
                }
                const char * kcls = a == draft_attn ? "draft index keys (ME6)" : "target index keys (ME6)";
                k.host_keys[j] = reinterpret_cast<uint16_t *>(class_ptr(m, kcls, L + " keys", c, bytes, err));
                if (k.host_keys[j] == nullptr) {
                    return false;
                }
                if (bytes / key_cell_bytes != n_cells_) {
                    return fail(err, L + " keys: " + std::to_string(bytes / key_cell_bytes) +
                                     " cells, not the slot's " + std::to_string(n_cells_));
                }
                if (!mapped_same(k.host_keys[j], err)) {
                    return false;
                }
            }
            k.key_arena = reinterpret_cast<uint16_t *>(
                class_ptr(m, "index key windows (ME6)", "key arena", c, bytes, err));
            if (k.key_arena == nullptr || bytes < key_arena_cells(w_) * key_cell_bytes) {
                return k.key_arena == nullptr ? false : fail(err, "the key arena is smaller than its cells");
            }
            k.k_arena = reinterpret_cast<uint16_t *>(class_ptr(m, "KV tier (ME1)", "K arena", c, bytes, err));
            if (k.k_arena == nullptr || bytes < arena_cells(w_, n_cells_) * cell_bytes) {
                return k.k_arena == nullptr ? false : fail(err, "the K arena is smaller than its cells");
            }
            k.v_arena = reinterpret_cast<uint16_t *>(class_ptr(m, "KV tier (ME1)", "V arena", c, bytes, err));
            if (k.v_arena == nullptr || bytes < arena_cells(w_, n_cells_) * cell_bytes) {
                return k.v_arena == nullptr ? false : fail(err, "the V arena is smaller than its cells");
            }
            uint8_t * t = class_ptr(m, "KV tier (ME1)", "tables", c, bytes, err);
            if (t == nullptr || bytes < tables_end) {
                return t == nullptr ? false : fail(err, "the tables part is smaller than its layout");
            }
            k.table = reinterpret_cast<int32_t *>(t + table_at);
            k.list = reinterpret_cast<int32_t *>(t + list_at);
            k.count = reinterpret_cast<int32_t *>(t + count_at);
            k.total = reinterpret_cast<unsigned long long *>(t + total_at);
            k.body[0] = reinterpret_cast<uint32_t *>(t + body_at);
            k.body[1] = reinterpret_cast<uint32_t *>(t + body_at + al(int64_t(tier::max_body_words) * 4));
            k.lo_word = reinterpret_cast<int64_t *>(t + lo_at);

            return true;
        })) {
            return false;
        }
    }
    zeroed();
    return true;
}

uint16_t * kv_tier::k(int32_t c, int32_t a) const {
    return cells_at(cards_[c].k_arena, int64_t(a) * w_ - (graph_ ? 0 : lo_));
}

uint16_t * kv_tier::v(int32_t c, int32_t a) const {
    return cells_at(cards_[c].v_arena, int64_t(a) * w_ - (graph_ ? 0 : lo_));
}

uint16_t * kv_tier::keys(int32_t c, int32_t a) const {
    const int32_t j = key_slot(c, a);
    return j < 0 ? nullptr : key_cells_at(cards_[c].key_arena, int64_t(j) * w_ - (graph_ ? 0 : lo_));
}

uint16_t * kv_tier::host_keys(int32_t c, int32_t a) const {
    const int32_t j = key_slot(c, a);
    return j < 0 ? nullptr : cards_[c].host_keys[j];
}

bool kv_tier::copy_window(bool to_host, int64_t first, int64_t last, void * stream, std::string & err) {
    last = std::min(last, std::min(lo_ + w_, n_cells_));
    first = std::max(first, lo_);
    if (last <= first) {
        return true;
    }
    const size_t n = size_t((last - first) * cell_bytes);
    for (int32_t c = 0; c < 2; ++c) {
        if (!on_device(device_[c], err, [&] {
            cudaStream_t s = static_cast<cudaStream_t>(streams_[c] != nullptr ? streams_[c] : stream);
            for (int32_t a = 0; a < n_attn; ++a) {
                uint16_t * h[2] = {cells_at(cards_[c].host_k[a], first), cells_at(cards_[c].host_v[a], first)};
                uint16_t * d[2] = {cells_at(cards_[c].k_arena, int64_t(a) * w_ + first - lo_),
                                   cells_at(cards_[c].v_arena, int64_t(a) * w_ + first - lo_)};
                for (int32_t i = 0; i < 2; ++i) {
                    if (!cuda_ok(to_host ? cudaMemcpyAsync(h[i], d[i], n, cudaMemcpyDeviceToHost, s)
                                         : cudaMemcpyAsync(d[i], h[i], n, cudaMemcpyHostToDevice, s),
                                 to_host ? "window to host" : "host to window", err)) {
                        return false;
                    }
                }
            }
            // ME6: the card's key slots, in the same window
            for (int32_t j = 0; j < key_slots; ++j) {
                uint16_t * h = key_cells_at(cards_[c].host_keys[j], first);
                uint16_t * d = key_cells_at(cards_[c].key_arena, int64_t(j) * w_ + first - lo_);
                const size_t nk = size_t((last - first) * key_cell_bytes);
                if (!cuda_ok(to_host ? cudaMemcpyAsync(h, d, nk, cudaMemcpyDeviceToHost, s)
                                     : cudaMemcpyAsync(d, h, nk, cudaMemcpyHostToDevice, s),
                             to_host ? "key window to host" : "host to key window", err)) {
                    return false;
                }
            }

            return true;
        })) {
            return false;
        }
    }
    (to_host ? n_.flush_bytes : n_.load_bytes) += int64_t(n) * 2 * 2 * n_attn;
    (to_host ? n_.key_flush_bytes : n_.key_load_bytes) += (last - first) * key_cell_bytes * key_slots * 2;
    return true;
}

bool kv_tier::decode(int64_t first, int64_t end, int64_t valid, void * stream, std::string & err) {
    if (first < 0 || end <= first || end > n_cells_ || end - first > w_ / 2) {
        return fail(err, "a decode round over cells " + std::to_string(first) + " .. " + std::to_string(end));
    }
    if (decode_ && first >= lo_ && end <= lo_ + w_) {
        hw_ = std::max(hw_, end);
        return true;
    }
    // the window's written cells to host memory, then the new window from it
    if (decode_ && !copy_window(true, lo_, hw_, stream, err)) {
        return false;
    }
    const int64_t lo = std::min(std::max(int64_t(0), end - w_ / 2), first) / 4 * 4;
    lo_ = lo;
    // ME3: the window's first cell for the graphs' launches (both executors write it; eager launches ignore it)
    for (int32_t c = 0; c < 2; ++c) {
        if (!on_device(device_[c], err, [&] {
            cudaStream_t s = static_cast<cudaStream_t>(streams_[c] != nullptr ? streams_[c] : stream);
            if (!cuda_ok(cudaMemcpyAsync(cards_[c].lo_word, &lo_, sizeof(lo_), cudaMemcpyHostToDevice,
                                         s),
                         "window start", err)) {
                return false;
            }

            return true;
        })) {
            return false;
        }
    }
    const int64_t top = std::min(std::min(valid, n_cells_), lo + w_);
    if (!copy_window(false, lo, top, stream, err)) {
        return false;
    }
    decode_ = true;
    hw_ = std::max(top, end);
    ++n_.rebuilds;
    return true;
}

bool kv_tier::select(int32_t c, int32_t a, int32_t rows, const uint32_t * body, const int32_t * offsets,
                     uint32_t * err_word, void * stream, const uint32_t *& out, std::string & err) {
    return on_device(device_[c], err, [&] {
        card & k = cards_[c];
        if (!decode_ && !graph_) { // ME3: a capture precedes every window; decode builds one before each replay
            return fail(err, "a decode launch without a decode window");
        }
        if (lo_ == 0 && !graph_) { // every cell in the window: the body as it is
            out = body;
            if (a == draft_attn) {
                draft_out_[c] = body;
            }
            return true;
        }
        tier::remap_args r;
        r.body = body;
        r.offsets = offsets;
        r.rows = rows;
        r.lo = graph_ ? 0 : static_cast<int32_t>(lo_);
        r.base = static_cast<int32_t>(((graph_ ? 0 : lo_) + int64_t(n_attn - a) * w_) / 4);
        r.lo_d = graph_ ? k.lo_word : nullptr; // ME3: graph mode's lo, and base's lo / 4, from the device word
        r.out_body = k.body[a == draft_attn ? 1 : 0];
        r.table = k.table;
        r.list = k.list;
        r.count = k.count;
        r.total = k.total;
        r.err = err_word;
        r.err_bit = fn::xchg::err_kv_tier;
        tier::gather_args g;
        g.list = k.list;
        g.count = k.count;
        g.host_k = k.host_k[a];
        g.host_v = k.host_v[a];
        g.area_k = cells_at(k.k_arena, int64_t(n_attn) * w_);
        g.area_v = cells_at(k.v_arena, int64_t(n_attn) * w_);
        if (!tier::remap(r, stream) || !tier::gather(g, stream)) {
            return fail(err, "remap or gather launch");
        }
        out = r.out_body;
        if (a == draft_attn) {
            draft_out_[c] = out;
        }
        return true;
    });
}

bool kv_tier::prompt(void * stream, std::string & err) {
    if (decode_ && !copy_window(true, lo_, hw_, stream, err)) {
        return false;
    }
    if (decode_) {
        ++n_.flushes;
    }
    decode_ = false;
    for (card & k : cards_) { // ME6: no keys staged yet in this prompt
        k.key_s0 = -1;
        k.key_a = -1;
    }
    return true;
}

bool kv_tier::stage(int32_t c, int32_t a, int64_t cells, void * stream, std::string & err) {
    return on_device(device_[c], err, [&] {
        if (decode_ || cells < 0 || cells > n_cells_) {
            return fail(err, "staging outside a prompt");
        }
        if (cells == 0) {
            return true;
        }
        cudaStream_t s = static_cast<cudaStream_t>(stream);
        const size_t n = size_t(cells * cell_bytes);
        if (!cuda_ok(cudaMemcpyAsync(cards_[c].k_arena, cards_[c].host_k[a], n, cudaMemcpyHostToDevice, s), "stage K",
                     err) ||
            !cuda_ok(cudaMemcpyAsync(cards_[c].v_arena, cards_[c].host_v[a], n, cudaMemcpyHostToDevice, s), "stage V",
                     err)) {
            return false;
        }
        n_.stage_bytes += 2 * int64_t(n);
        return true;
    });
}

bool kv_tier::write_back(int32_t c, int32_t a, int64_t first, int64_t n, void * stream, std::string & err) {
    return on_device(device_[c], err, [&] {
        if (decode_ || first < 0 || n < 0 || first + n > n_cells_) {
            return fail(err, "a write-back outside a prompt");
        }
        cudaStream_t s = static_cast<cudaStream_t>(stream);
        const size_t b = size_t(n * cell_bytes);
        if (!cuda_ok(cudaMemcpyAsync(cells_at(cards_[c].host_k[a], first), cells_at(cards_[c].k_arena, first), b,
                                     cudaMemcpyDeviceToHost, s),
                     "write back K", err) ||
            !cuda_ok(cudaMemcpyAsync(cells_at(cards_[c].host_v[a], first), cells_at(cards_[c].v_arena, first), b,
                                     cudaMemcpyDeviceToHost, s),
                     "write back V", err)) {
            return false;
        }
        n_.write_back_bytes += 2 * int64_t(b);
        return true;
    });
}

bool kv_tier::stage_keys(int32_t c, int32_t a, int64_t start, int64_t rows, void * stream, std::string & err) {
    return on_device(device_[c], err, [&] {
        const int32_t j = key_slot(c, a);
        const int64_t s0 = start / fn::qsa::ratio * fn::qsa::ratio;
        if (decode_ || j < 0 || start < 0 || rows < 1 || start + rows > n_cells_ ||
            start - s0 + rows > key_arena_cells(w_)) {
            return fail(err, "key staging outside a prompt or beyond the key arena");
        }
        card & k = cards_[c];
        k.key_s0 = s0;
        k.key_a = a;
        k.key_end = start + rows;
        if (start == s0) {
            return true;
        }
        const size_t n = size_t((start - s0) * key_cell_bytes);
        if (!cuda_ok(cudaMemcpyAsync(k.key_arena, key_cells_at(k.host_keys[j], s0), n, cudaMemcpyHostToDevice,
                                     static_cast<cudaStream_t>(stream)),
                     "stage keys", err)) {
            return false;
        }
        n_.key_stage_bytes += int64_t(n);
        return true;
    });
}

uint16_t * kv_tier::stage_keys_ptr(int32_t c) const {
    return cards_[c].key_s0 < 0 ? nullptr : key_cells_at(cards_[c].key_arena, -cards_[c].key_s0);
}

bool kv_tier::write_back_keys(int32_t c, int32_t a, int64_t first, int64_t n, void * stream, std::string & err) {
    return on_device(device_[c], err, [&] {
        const card & k = cards_[c];
        const int32_t j = key_slot(c, a);
        if (decode_ || j < 0 || k.key_a != a || first < k.key_s0 || n < 0 || first + n > k.key_end) {
            return fail(err, "a key write-back outside the staged chunk");
        }
        const size_t b = size_t(n * key_cell_bytes);
        if (!cuda_ok(cudaMemcpyAsync(key_cells_at(k.host_keys[j], first),
                                     key_cells_at(k.key_arena, first - k.key_s0), b,
                                     cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)),
                     "write back keys", err)) {
            return false;
        }
        n_.key_write_back_bytes += int64_t(b);
        return true;
    });
}

void kv_tier::zeroed() {
    decode_ = false;
    lo_ = 0;
    hw_ = 0;
}

bool kv_tier::flush(void * stream, std::string & err) {
    if (decode_ && !copy_window(true, lo_, hw_, stream, err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!on_device(device_[c], err, [&] {
            return cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(
                               streams_[c] != nullptr ? streams_[c] : stream)), "flush", err);
        })) {
            return false;
        }
    }
    return true;
}

bool kv_tier::reset_counters(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!on_device(device_[c], err, [&] {
            if (!cuda_ok(cudaMemset(cards_[c].total, 0, sizeof(unsigned long long)), "counters", err)) {
                return false;
            }

            return true;
        })) {
            return false;
        }
    }
    n_ = counters();
    return true;
}

std::string kv_tier::counters_text() const {
    const double mib = 1048576.0;
    const int64_t block = int64_t(fn::qsa::ratio) * cell_bytes * 2; // a gathered block's K and V
    char b[768];
    std::snprintf(b, sizeof(b),
                  "KV tier: window %d cells at %lld; gathered blocks %lld + %lld (cards 0, 1: %.1f MiB K and V from "
                  "host memory); window rebuilds %lld (to host %.1f MiB, from host %.1f MiB); prompt flushes %lld, "
                  "staged %.1f MiB, written back %.1f MiB; index keys (ME6): window to host %.1f MiB, from host "
                  "%.1f MiB, staged %.3f MiB, written back %.1f MiB",
                  w_, (long long)lo_, (long long)n_.gathered_blocks[0], (long long)n_.gathered_blocks[1],
                  double((n_.gathered_blocks[0] + n_.gathered_blocks[1]) * block) / mib, (long long)n_.rebuilds,
                  double(n_.flush_bytes) / mib, double(n_.load_bytes) / mib, (long long)n_.flushes,
                  double(n_.stage_bytes) / mib, double(n_.write_back_bytes) / mib, double(n_.key_flush_bytes) / mib,
                  double(n_.key_load_bytes) / mib, double(n_.key_stage_bytes) / mib,
                  double(n_.key_write_back_bytes) / mib);
    return b;
}

bool kv_tier::read_counters(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!on_device(device_[c], err, [&] {
            unsigned long long t = 0;
            if (!cuda_ok(cudaMemcpy(&t, cards_[c].total, sizeof(t), cudaMemcpyDeviceToHost), "counters", err)) {
                return false;
            }
            n_.gathered_blocks[c] = int64_t(t);

            return true;
        })) {
            return false;
        }
    }
    return true;
}

} // namespace fn::model
