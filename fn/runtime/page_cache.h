#pragma once

// PL1, measurement only: the page-cache state of a range of a read-only shared file mapping (the PLE table), and its
// eviction for a cold-table run. Not used by a round.
//
// Eviction, in this order (PL1's protocol): madvise(MADV_DONTNEED) on the range of this process's mapping drops its
// page-table entries (the pages stay cached; posix_madvise's DONTNEED is a no-op in glibc, so madvise is called), then
// posix_fadvise(POSIX_FADV_DONTNEED) on the file's page-aligned range frees the clean cached pages no process maps. It
// touches no other file and needs no privilege. A page another process maps stays cached; the counts show it.

#include <cstdint>
#include <string>

namespace fn::runtime {

struct page_counts {
    int64_t pages = 0;       // pages of the range, rounded out to whole pages
    int64_t mapping_res = 0; // mincore(2) over this process's mapping of the range: pages resident
    int64_t file_cached = 0; // cachestat(2) over the file range: pages in the page cache
};

// the counts of [p, p + bytes) of the mapping, whose bytes are the file's [offset, offset + bytes)
bool page_cache_counts(const uint8_t * p, uint64_t bytes, const std::string & path, uint64_t offset, page_counts & c,
                       std::string & err);

// the two eviction steps above on [p, p + bytes) and the file's [offset, offset + bytes), rounded out to whole pages
bool page_cache_evict(const uint8_t * p, uint64_t bytes, const std::string & path, uint64_t offset, std::string & err);

// only the first step: this process's page-table entries for the range dropped, the pages left in the page cache (a
// page-cached first touch follows)
bool page_cache_unmap(const uint8_t * p, uint64_t bytes, std::string & err);

// the bytes this process has caused to be read from storage (/proc/self/io read_bytes); -1 when unreadable
int64_t storage_read_bytes();

} // namespace fn::runtime
