#pragma once

// ME4 (expires at ME4's disposition): the residency of the routed experts' native GGUF pages, which the CPU products
// read in place (experts_cpu.h native_layer). The packed backing stays for the GPU lane, so these pages are a second,
// file-backed copy, read by the CPU only. Fable's R4-15 ruling: they cost no device memory, so no CUDA registration
// (that maps them for the devices too: device page tables of about 1/512 of their bytes, 80 MiB of a 16 GB card's
// pool for 40 GiB).
//
// The hold: madvise(MADV_POPULATE_READ) reads every page of the read-only shared file mapping in (an I/O error fails
// the hold), then mlock locks them when the process's RLIMIT_MEMLOCK allows (the soft limit raised to the hard one when
// that covers them); otherwise they stay populated and not locked, which the report says, with the process's RssFile
// and major page faults (getrusage ru_majflt) at the hold, at each request's end and at release, and the pages
// resident (mincore). A major fault after the hold is shown, never hidden. A failed lock the limit allowed fails the
// hold. The tensors' byte spans are rounded out to whole pages and merged where they overlap or touch, so a page two
// tensors share is held and counted once.

#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime {

struct page_range {
    uintptr_t first = 0, end = 0; // [first, end)
};

// spans rounded out to pages of size page (a power of two), sorted, merged where they overlap or touch, then with
// [ex_first, ex_end) (page-aligned; empty when equal) cut out. out_excluded: the bytes cut
std::vector<page_range> merge_pages(std::vector<page_range> spans, uintptr_t page, uintptr_t ex_first,
                                    uintptr_t ex_end, uint64_t & out_excluded);

// this process's VmRSS, RssFile and VmLck from /proc/self/status, in bytes; -1 unreadable
struct status_bytes {
    int64_t rss = -1, rss_file = -1, locked = -1;
};
status_bytes read_status_bytes();

// the residency at one moment: the held pages resident (mincore; -1 unreadable), the process's RssFile and major faults
struct page_snapshot {
    int64_t resident = -1, pages = 0;
    int64_t rss_file = -1;
    int64_t majflt = -1;
};

class native_pages {
  public:
    native_pages() = default;
    ~native_pages();
    native_pages(const native_pages &) = delete;
    native_pages & operator=(const native_pages &) = delete;

    // spans: the tensors' byte ranges inside the model's file mappings (the model outlives this hold). threads: the
    // populate's. False with err naming the call; nothing stays locked then
    bool hold(const std::vector<page_range> & spans, int32_t threads, std::string & err);
    void release(); // the CPU pool idle: no job reads a view after this
    page_snapshot snapshot() const;

    bool held() const { return !ranges_.empty(); }
    bool locked() const { return locked_; }
    uint64_t span_bytes = 0; // the tensors' bytes (the sum of the spans)
    uint64_t held_bytes = 0; // the merged page ranges' bytes, each page once
    int32_t n_ranges = 0;
    double populate_s = 0, lock_s = 0;
    int64_t limit_soft = -1, limit_hard = -1; // RLIMIT_MEMLOCK at the hold (bytes; -1 unlimited)
    status_bytes before, after;               // around the hold
    page_snapshot at_hold;                    // after it
    std::string line() const;                 // the report's line at init
    std::string json(const page_snapshot & s) const; // a request's or the release's: {"locked", "resident_pages", ...}

  private:
    std::vector<page_range> ranges_; // populated (and locked when locked_)
    bool locked_ = false;
};

} // namespace fn::runtime
