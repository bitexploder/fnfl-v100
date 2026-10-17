#include "fn/runtime/page_cache.h"

#include <fcntl.h>
#include <linux/mman.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

bool fail_errno(std::string & err, const std::string & what) {
    err = what + ": " + std::strerror(errno);
    return false;
}

uint64_t page_size() {
    const long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? static_cast<uint64_t>(p) : 4096;
}

// the range rounded out to whole pages: [*a, *a + *n)
void round_out(uint64_t first, uint64_t bytes, uint64_t & a, uint64_t & n) {
    const uint64_t pg = page_size();
    a = first / pg * pg;
    n = (first + bytes + pg - 1) / pg * pg - a;
}

} // namespace

bool page_cache_counts(const uint8_t * p, uint64_t bytes, const std::string & path, uint64_t offset, page_counts & c,
                       std::string & err) {
    c = page_counts();
    if (p == nullptr || bytes == 0) {
        err = "page counts: empty range";
        return false;
    }
    uint64_t a = 0, n = 0;
    // reason: page arithmetic on the mapping's addresses
    round_out(reinterpret_cast<uintptr_t>(p), bytes, a, n);
    c.pages = static_cast<int64_t>(n / page_size());
    std::vector<unsigned char> vec(size_t(c.pages));
    // reason: mincore takes the address as a mutable pointer; it reads no memory
    if (mincore(reinterpret_cast<void *>(a), size_t(n), vec.data()) != 0) {
        return fail_errno(err, "mincore");
    }
    for (const unsigned char v : vec) {
        c.mapping_res += v & 1;
    }
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return fail_errno(err, "open '" + path + "'");
    }
    uint64_t fa = 0, fn_ = 0;
    round_out(offset, bytes, fa, fn_);
    struct cachestat_range r = {fa, fn_};
    struct cachestat cs = {};
    const long rc = syscall(SYS_cachestat, fd, &r, &cs, 0);
    const int e = errno;
    close(fd);
    if (rc != 0) {
        errno = e;
        return fail_errno(err, "cachestat");
    }
    c.file_cached = static_cast<int64_t>(cs.nr_cache);
    return true;
}

bool page_cache_unmap(const uint8_t * p, uint64_t bytes, std::string & err) {
    uint64_t a = 0, n = 0;
    // reason: page arithmetic on the mapping's addresses
    round_out(reinterpret_cast<uintptr_t>(p), bytes, a, n);
    // reason: madvise takes the address as a mutable pointer; DONTNEED on a shared file mapping drops page-table
    // entries only (the file is unchanged)
    if (p == nullptr || madvise(reinterpret_cast<void *>(a), size_t(n), MADV_DONTNEED) != 0) {
        return fail_errno(err, "madvise(MADV_DONTNEED)");
    }
    return true;
}

bool page_cache_evict(const uint8_t * p, uint64_t bytes, const std::string & path, uint64_t offset, std::string & err) {
    if (!page_cache_unmap(p, bytes, err)) {
        return false;
    }
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return fail_errno(err, "open '" + path + "'");
    }
    uint64_t fa = 0, fn_ = 0;
    round_out(offset, bytes, fa, fn_);
    const int rc = posix_fadvise(fd, off_t(fa), off_t(fn_), POSIX_FADV_DONTNEED);
    close(fd);
    if (rc != 0) {
        err = std::string("posix_fadvise(POSIX_FADV_DONTNEED): ") + std::strerror(rc);
        return false;
    }
    return true;
}

int64_t storage_read_bytes() {
    FILE * f = fopen("/proc/self/io", "r");
    if (f == nullptr) {
        return -1;
    }
    char line[128];
    long long v = -1;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (std::sscanf(line, "read_bytes: %lld", &v) == 1) {
            break;
        }
    }
    fclose(f);
    return static_cast<int64_t>(v);
}

} // namespace fn::runtime
