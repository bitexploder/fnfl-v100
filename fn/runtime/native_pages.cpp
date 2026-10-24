#include "fn/runtime/native_pages.h"

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

namespace fn::runtime {

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

std::vector<page_range> merge_pages(std::vector<page_range> spans, uintptr_t page, uintptr_t ex_first,
                                    uintptr_t ex_end, uint64_t & out_excluded) {
    out_excluded = 0;
    std::vector<page_range> m;
    for (page_range & s : spans) {
        s.first &= ~(page - 1);
        s.end = (s.end + page - 1) & ~(page - 1);
    }
    std::sort(spans.begin(), spans.end(), [](const page_range & a, const page_range & b) { return a.first < b.first; });
    for (const page_range & s : spans) {
        if (s.end <= s.first) {
            continue;
        }
        if (!m.empty() && s.first <= m.back().end) {
            m.back().end = std::max(m.back().end, s.end);
        } else {
            m.push_back(s);
        }
    }
    if (ex_end <= ex_first) {
        return m;
    }
    std::vector<page_range> out;
    for (const page_range & r : m) {
        const uintptr_t a = std::max(r.first, ex_first), b = std::min(r.end, ex_end);
        if (a >= b) {
            out.push_back(r);
            continue;
        }
        out_excluded += uint64_t(b - a);
        if (r.first < a) {
            out.push_back({r.first, a});
        }
        if (b < r.end) {
            out.push_back({b, r.end});
        }
    }
    return out;
}

status_bytes read_status_bytes() {
    status_bytes s;
    std::ifstream f("/proc/self/status");
    std::string k;
    while (f >> k) {
        int64_t v = 0;
        if (k == "VmRSS:" || k == "RssFile:" || k == "VmLck:") {
            if (!(f >> v)) {
                break;
            }
            (k == "VmRSS:" ? s.rss : k == "RssFile:" ? s.rss_file : s.locked) = v * 1024; // kB
        }
        f.ignore(1 << 20, '\n');
    }
    return s;
}

native_pages::~native_pages() {
    release();
}

bool native_pages::hold(const std::vector<page_range> & spans, int32_t threads, std::string & err) {
    release();
    const long page_l = sysconf(_SC_PAGESIZE);
    if (spans.empty() || page_l <= 0 || (page_l & (page_l - 1)) != 0) {
        err = "native expert pages: no spans, or page size " + std::to_string(page_l);
        return false;
    }
    span_bytes = 0;
    for (const page_range & s : spans) {
        span_bytes += uint64_t(s.end - s.first);
    }
    uint64_t none = 0;
    const std::vector<page_range> r = merge_pages(spans, uintptr_t(page_l), 0, 0, none);
    held_bytes = 0;
    for (const page_range & x : r) {
        held_bytes += uint64_t(x.end - x.first);
    }
    n_ranges = int32_t(r.size());
    before = read_status_bytes();
    // populate: 64 MiB pieces of the ranges, by threads
    constexpr uintptr_t piece = uintptr_t(64) << 20;
    std::vector<page_range> pieces;
    for (const page_range & x : r) {
        for (uintptr_t a = x.first; a < x.end; a += piece) {
            pieces.push_back({a, std::min(x.end, a + piece)});
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    std::atomic<int> bad_errno{0};
    std::atomic<size_t> bad_at{0};
    std::vector<std::thread> th;
    for (int32_t t = 0; t < std::max(1, threads); ++t) {
        th.emplace_back([&]() {
            for (size_t i = next++; i < pieces.size() && bad_errno.load() == 0; i = next++) {
                // reason: madvise takes the range as a mutable pointer; POPULATE_READ writes nothing
                if (madvise(reinterpret_cast<void *>(pieces[i].first), size_t(pieces[i].end - pieces[i].first),
                            MADV_POPULATE_READ) != 0) {
                    bad_at.store(i);
                    bad_errno.store(errno == 0 ? -1 : errno);
                }
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
    populate_s = seconds_since(t0);
    if (bad_errno.load() != 0) {
        const page_range & p = pieces[bad_at.load()];
        err = "native expert pages: madvise(MADV_POPULATE_READ) of " + std::to_string(p.end - p.first) +
              " bytes at piece " + std::to_string(bad_at.load()) + ": " +
              (bad_errno.load() > 0 ? std::strerror(bad_errno.load()) : "unknown error");
        return false;
    }
    ranges_ = r;
    // lock when the memlock limit covers the pages (and what this process has locked already)
    struct rlimit rl = {};
    getrlimit(RLIMIT_MEMLOCK, &rl);
    limit_soft = rl.rlim_cur == RLIM_INFINITY ? -1 : int64_t(rl.rlim_cur);
    limit_hard = rl.rlim_max == RLIM_INFINITY ? -1 : int64_t(rl.rlim_max);
    const int64_t have = std::max<int64_t>(0, before.locked);
    const uint64_t need = held_bytes + uint64_t(have);
    const auto covers = [&](rlim_t v) { return v == RLIM_INFINITY || uint64_t(v) >= need; };
    if (!covers(rl.rlim_cur) && covers(rl.rlim_max)) {
        rl.rlim_cur = rl.rlim_max;
        if (setrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
            limit_soft = limit_hard;
        }
    }
    if (covers(rl.rlim_cur)) {
        const auto t1 = std::chrono::steady_clock::now();
        size_t k = 0;
        for (; k < ranges_.size(); ++k) {
            // reason: mlock takes the range's address; it writes nothing
            if (mlock(reinterpret_cast<void *>(ranges_[k].first), size_t(ranges_[k].end - ranges_[k].first)) != 0) {
                break;
            }
        }
        if (k < ranges_.size()) {
            const int e = errno;
            for (size_t j = 0; j < k; ++j) {
                munlock(reinterpret_cast<void *>(ranges_[j].first), size_t(ranges_[j].end - ranges_[j].first));
            }
            ranges_.clear();
            err = "native expert pages: mlock of range " + std::to_string(k) + " of " + std::to_string(r.size()) +
                  " failed under a memlock limit that covers them (" + std::to_string(limit_soft) + " bytes): " +
                  std::strerror(e);
            return false;
        }
        locked_ = true;
        lock_s = seconds_since(t1);
    }
    after = read_status_bytes();
    at_hold = snapshot();
    return true;
}

void native_pages::release() {
    if (locked_) {
        for (const page_range & x : ranges_) {
            // at teardown: nothing to do on a failure
            munlock(reinterpret_cast<void *>(x.first), size_t(x.end - x.first));
        }
    }
    locked_ = false;
    ranges_.clear();
}

page_snapshot native_pages::snapshot() const {
    page_snapshot s;
    const long page_l = sysconf(_SC_PAGESIZE);
    std::vector<unsigned char> v;
    s.resident = 0;
    for (const page_range & x : ranges_) {
        const size_t n = size_t(x.end - x.first) / size_t(page_l);
        v.resize(n);
        s.pages += int64_t(n);
        // reason: mincore takes the range's address; it reads the page tables only
        if (mincore(reinterpret_cast<void *>(x.first), size_t(x.end - x.first), v.data()) != 0) {
            s.resident = -1;
            break;
        }
        for (const unsigned char c : v) {
            s.resident += c & 1;
        }
    }
    s.rss_file = read_status_bytes().rss_file;
    struct rusage ru = {};
    s.majflt = getrusage(RUSAGE_SELF, &ru) == 0 ? int64_t(ru.ru_majflt) : -1;
    return s;
}

std::string native_pages::json(const page_snapshot & s) const {
    char b[256];
    std::snprintf(b, sizeof(b),
                  "{\"locked\": %s, \"resident_pages\": %lld, \"pages\": %lld, \"rss_file\": %lld, \"majflt\": %lld}",
                  locked_ ? "true" : "false", (long long)s.resident, (long long)s.pages, (long long)s.rss_file,
                  (long long)s.majflt);
    return b;
}

std::string native_pages::line() const {
    const double gib = double(uint64_t(1) << 30);
    char how[200];
    if (locked_) {
        std::snprintf(how, sizeof(how), "populated and LOCKED (mlock, %.2f s; RLIMIT_MEMLOCK %lld)", lock_s,
                      (long long)limit_soft);
    } else {
        std::snprintf(how, sizeof(how),
                      "populated, NOT locked (RLIMIT_MEMLOCK soft %lld, hard %lld bytes, below %.3f GiB): evictable; "
                      "residency and major faults reported at each request's end",
                      (long long)limit_soft, (long long)limit_hard, double(held_bytes) / gib);
    }
    char b[768];
    std::snprintf(b, sizeof(b),
                  "native expert pages: %s; no CUDA registration (no device memory); tensors %.3f GiB, held %.3f GiB "
                  "in %d ranges (pages shared once); populate %.2f s; resident %lld of %lld pages; VmRSS %.3f -> "
                  "%.3f GiB, RssFile %.3f -> %.3f GiB, VmLck %.3f GiB; major faults %lld",
                  how, double(span_bytes) / gib, double(held_bytes) / gib, n_ranges, populate_s,
                  (long long)at_hold.resident, (long long)at_hold.pages, double(before.rss) / gib,
                  double(after.rss) / gib, double(before.rss_file) / gib, double(after.rss_file) / gib,
                  double(after.locked) / gib, (long long)at_hold.majflt);
    return b;
}

} // namespace fn::runtime
