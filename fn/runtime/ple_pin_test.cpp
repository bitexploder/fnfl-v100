#include "fn/runtime/ple_pin_test.h"

#include "fn/runtime/ple_host.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr uint64_t probe_bytes = uint64_t(64) << 20;

// a "Key:   N kB" line of /proc/self/status, in kB; -1 when absent
int64_t status_kb(const char * key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    const size_t n = std::strlen(key);
    while (std::getline(f, line)) {
        if (line.compare(0, n, key) == 0 && line.size() > n && line[n] == ':') {
            return std::stoll(line.substr(n + 1));
        }
    }
    return -1;
}

// a "key N" line of /proc/vmstat; -1 when absent
int64_t vmstat(const char * key) {
    std::ifstream f("/proc/vmstat");
    std::string k;
    int64_t v = 0;
    while (f >> k >> v) {
        if (k == key) {
            return v;
        }
    }
    return -1;
}

// pages the kernel pins with FOLL_PIN and has not released (pin_user_pages)
int64_t foll_pinned() {
    const int64_t a = vmstat("nr_foll_pin_acquired"), r = vmstat("nr_foll_pin_released");
    return a < 0 || r < 0 ? -1 : a - r;
}

// the same after the kernel's vmstat worker has folded every CPU's pending counts into /proc/vmstat: it runs every
// vm.stat_interval seconds, and until then a CPU's change below its threshold is not counted
int64_t foll_pinned_settled() {
    std::ifstream f("/proc/sys/vm/stat_interval");
    int64_t interval = 1;
    f >> interval;
    sleep(static_cast<unsigned>(2 * std::max<int64_t>(interval, 1) + 1));
    return foll_pinned();
}

// resident pages of [p, p + n) by mincore; -1 on error
int64_t resident_pages(const uint8_t * p, uint64_t n) {
    const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    std::vector<unsigned char> v((n + page - 1) / page);
    // reason: mincore takes the mapping's address as a mutable pointer; it writes nothing there
    if (mincore(const_cast<uint8_t *>(p), n, v.data()) != 0) {
        return -1;
    }
    int64_t r = 0;
    for (unsigned char c : v) {
        r += c & 1;
    }
    return r;
}

// a file of n bytes (a counting pattern) in dir, written, synced and dropped from the page cache; its path in path
bool make_file(const char * dir, uint64_t n, std::string & path) {
    path = std::string(dir) + "/ple-pin-test." + std::to_string(getpid()) + ".bin";
    const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) {
        fprintf(stderr, "ple-pin-test: cannot create '%s': %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    std::vector<uint32_t> buf(size_t(1) << 18);
    bool ok = true;
    for (uint64_t off = 0; ok && off < n; off += buf.size() * 4) {
        for (size_t i = 0; i < buf.size(); ++i) {
            buf[i] = static_cast<uint32_t>(off / 4 + i) * 2654435761u;
        }
        const size_t w = size_t(std::min<uint64_t>(n - off, buf.size() * 4));
        ok = ::write(fd, buf.data(), w) == ssize_t(w);
    }
    ok = ok && fsync(fd) == 0 && posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0;
    close(fd);
    if (!ok) {
        fprintf(stderr, "ple-pin-test: cannot write '%s'\n", path.c_str());
        unlink(path.c_str());
    }
    return ok;
}

const uint8_t * map_file(const std::string & path, uint64_t n) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return nullptr;
    }
    void * p = mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return p == MAP_FAILED ? nullptr : static_cast<const uint8_t *>(p);
}

void counters(const char * when, const uint8_t * p, uint64_t n) {
    const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    printf("probe: %-34s resident %lld / %llu pages; VmLck %lld kB, VmPin %lld kB; FOLL_PIN pages held %lld\n", when,
           p != nullptr ? static_cast<long long>(resident_pages(p, n)) : -1LL,
           static_cast<unsigned long long>((n + page - 1) / page), static_cast<long long>(status_kb("VmLck")),
           static_cast<long long>(status_kb("VmPin")), static_cast<long long>(foll_pinned()));
}

int probe(const char * dir, int device) {
    struct rlimit rl = {};
    getrlimit(RLIMIT_MEMLOCK, &rl);
    int drv = 0, rtv = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rtv);
    printf("probe: RLIMIT_MEMLOCK soft %llu, hard %llu bytes; page %ld bytes; CUDA driver %d, runtime %d\n",
           static_cast<unsigned long long>(rl.rlim_cur), static_cast<unsigned long long>(rl.rlim_max),
           sysconf(_SC_PAGESIZE), drv, rtv);
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess) {
        fprintf(stderr, "probe: no device\n");
        return 1;
    }
    for (int d = 0; d < n_dev; ++d) {
        int ro = -1, reg = -1;
        const cudaError_t e1 = cudaDeviceGetAttribute(&ro, cudaDevAttrHostRegisterReadOnlySupported, d);
        const cudaError_t e2 = cudaDeviceGetAttribute(&reg, cudaDevAttrHostRegisterSupported, d);
        printf("probe: device %d: HostRegisterReadOnlySupported %d (%s), HostRegisterSupported %d (%s)\n", d, ro,
               cudaGetErrorString(e1), reg, cudaGetErrorString(e2));
    }
    if (cudaSetDevice(device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        fprintf(stderr, "probe: device %d\n", device);
        return 1;
    }
    std::string path;
    if (!make_file(dir, probe_bytes, path)) {
        return 1;
    }
    const uint8_t * p = map_file(path, probe_bytes);
    if (p == nullptr) {
        fprintf(stderr, "probe: cannot map '%s'\n", path.c_str());
        unlink(path.c_str());
        return 1;
    }
    // reason: madvise, cudaHostRegister and munmap take the mapping's address as a mutable pointer
    void * a = const_cast<uint8_t *>(p);
    counters("mapped, cold", p, probe_bytes);
    const int rc = madvise(a, probe_bytes, MADV_POPULATE_READ);
    printf("probe: madvise(MADV_POPULATE_READ): %s\n", rc == 0 ? "ok" : std::strerror(errno));
    counters("populated", p, probe_bytes);

    cudaError_t e = cudaHostRegister(a, probe_bytes, cudaHostRegisterPortable);
    printf("probe: cudaHostRegister(Portable) on the read-only mapping: %s\n", cudaGetErrorString(e));
    if (e == cudaSuccess) {
        cudaHostUnregister(a);
    }
    cudaGetLastError();
    e = cudaHostRegister(a, probe_bytes, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    printf("probe: cudaHostRegister(Portable | ReadOnly) on the read-only mapping: %s\n", cudaGetErrorString(e));
    cudaGetLastError();
    if (e == cudaSuccess) {
        counters("registered", p, probe_bytes);
        // the cache's copy of the file is not dropped while registered
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
        }
        counters("registered, after FADV_DONTNEED", p, probe_bytes);
        bool same = true;
        for (uint64_t i = 0; i < probe_bytes / 4; i += 4099) {
            uint32_t v = 0;
            std::memcpy(&v, p + i * 4, 4);
            same = same && v == static_cast<uint32_t>(i) * 2654435761u;
        }
        printf("probe: bytes read through the registered mapping: %s\n", same ? "equal to the file's" : "DIFFER");
        e = cudaHostUnregister(a);
        printf("probe: cudaHostUnregister: %s\n", cudaGetErrorString(e));
        counters("unregistered", p, probe_bytes);
    }
    munmap(a, probe_bytes);
    unlink(path.c_str());

    void * h = nullptr;
    counters("before cudaHostAlloc", nullptr, probe_bytes);
    e = cudaHostAlloc(&h, probe_bytes, cudaHostAllocPortable);
    printf("probe: cudaHostAlloc(Portable) of 64 MiB: %s\n", cudaGetErrorString(e));
    if (e == cudaSuccess) {
        std::memset(h, 0, probe_bytes);
        counters("allocated (64 MiB, written)", static_cast<const uint8_t *>(h), probe_bytes);
        cudaFreeHost(h);
        counters("freed", nullptr, probe_bytes);
    }
    printf("probe: done\n");
    return 0;
}

// the counting pattern make_file writes, compared over the whole mapping
bool bytes_equal(const uint8_t * p, uint64_t n) {
    for (uint64_t i = 0; i < n / 4; ++i) {
        uint32_t v = 0;
        std::memcpy(&v, p + i * 4, 4);
        if (v != static_cast<uint32_t>(i) * 2654435761u) {
            return false;
        }
    }
    return true;
}

void drop_cache(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
}

// ple_prefetch over one round row of 16 rows spread across a table of table_rows rows
bool hint(const uint8_t * table, int64_t table_rows, int32_t & n_ranges, std::string & err) {
    fn::runtime::ple_params pp;
    pp.table_rows = table_rows;
    int32_t idx[fn::runtime::ple_heads];
    for (int32_t h = 0; h < fn::runtime::ple_heads; ++h) {
        idx[h] = static_cast<int32_t>(table_rows / fn::runtime::ple_heads * h);
    }
    return fn::runtime::ple_prefetch(pp, table, idx, 1, n_ranges, err);
}

struct checker {
    int failed = 0;
    void operator()(bool ok, const char * what, const std::string & detail = std::string()) {
        printf("ple-pin-test: %s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : ": ", detail.c_str());
        failed += ok ? 0 : 1;
    }
};

// the two FOLL_PIN counters' changes since a sample: "+A acquired, +R released"
struct foll_sample {
    int64_t acquired = vmstat("nr_foll_pin_acquired");
    int64_t released = vmstat("nr_foll_pin_released");
    std::string since() const {
        return "+" + std::to_string(vmstat("nr_foll_pin_acquired") - acquired) + " acquired, +" +
               std::to_string(vmstat("nr_foll_pin_released") - released) + " released";
    }
};

bool names(const std::string & err, const char * call) {
    return err.find(call) != std::string::npos && err.find("--fnfl-ple-table lazy") != std::string::npos;
}

int unit(const char * dir, int device) {
    namespace rt = fn::runtime;
    if (cudaSetDevice(device) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
        fprintf(stderr, "ple-pin-test: device %d\n", device);
        return 1;
    }
    struct rlimit rl = {};
    getrlimit(RLIMIT_MEMLOCK, &rl);
    const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    const uint64_t n = probe_bytes; // 64 MiB, eight times RLIMIT_MEMLOCK's 8 MiB
    const int64_t pages = static_cast<int64_t>(n / page);
    // the table's start is off the page boundary, as the real table's is inside shard 2
    const uint64_t skew = 1000;
    const int64_t rows = static_cast<int64_t>((n - skew) / rt::ple_row_bytes);
    printf("ple-pin-test: unit: %llu-byte file (%lld pages), RLIMIT_MEMLOCK soft %llu bytes, device %d\n",
           static_cast<unsigned long long>(n), static_cast<long long>(pages),
           static_cast<unsigned long long>(rl.rlim_cur), device);
    checker check;
    std::string path;
    if (!make_file(dir, n, path)) {
        return 1;
    }
    const uint8_t * p = map_file(path, n);
    if (p == nullptr) {
        fprintf(stderr, "ple-pin-test: cannot map '%s'\n", path.c_str());
        unlink(path.c_str());
        return 1;
    }
    std::string err;
    int32_t n_ranges = -1;

    // lazy: no ple_pin, so nothing pinned and the hint still runs
    {
        check(resident_pages(p, n) == 0, "lazy: the mapping starts cold",
              std::to_string(resident_pages(p, n)) + " resident");
        const int64_t f0 = foll_pinned();
        const bool ok = hint(p + skew, rows, n_ranges, err);
        const int64_t f1 = foll_pinned();
        check(ok && n_ranges > 0, "lazy: ple_prefetch hints", std::to_string(n_ranges) + " ranges " + err);
        check(f1 - f0 < pages, "lazy: nothing pinned",
              "FOLL_PIN net " + std::to_string(f0) + " -> " + std::to_string(f1));
    }
    drop_cache(path);

    // full: populate and pin, every page resident, the bytes unchanged, no hint
    {
        rt::ple_pin pin;
        const int64_t f0 = foll_pinned_settled();
        const foll_sample s0;
        const bool pop = pin.populate(p + skew, n - skew, err);
        check(pop, "full: populate", err);
        check(resident_pages(p, n) == pages, "full: every page resident after populate",
              std::to_string(resident_pages(p, n)) + " / " + std::to_string(pages));
        const foll_sample s1;
        const bool pinned = pop && pin.pin(err);
        const std::string reg = s1.since();
        check(pinned && pin.pinned() && pin.pinned_bytes() == n, "full: pin",
              std::to_string(pin.pinned_bytes()) + " bytes pinned " + err);
        const int64_t f1 = foll_pinned_settled();
        // ple_pin's own samples are taken at once, before the vmstat worker folds the CPUs' counts: shown, not checked
        check(f1 - f0 >= pages, "full: the driver pins the range",
              "FOLL_PIN net settled " + std::to_string(f0) + " -> " + std::to_string(f1) + "; ple_pin's samples " +
                  std::to_string(pin.foll_pin_before) + " -> " + std::to_string(pin.foll_pin_after) + "; register " +
                  reg);
        drop_cache(path);
        check(resident_pages(p, n) == pages, "full: every page resident after FADV_DONTNEED",
              std::to_string(resident_pages(p, n)) + " / " + std::to_string(pages));
        check(bytes_equal(p, n), "full: the bytes equal the file's");
        n_ranges = -1;
        const bool ok = hint(p + skew, rows, n_ranges, err);
        check(ok && n_ranges == 0, "full: ple_prefetch reports 0 ranges", std::to_string(n_ranges) + " " + err);
        const foll_sample s2;
        pin.release();
        const int64_t f2 = foll_pinned_settled();
        check(f1 - f2 >= pages && !pin.pinned(), "full: release unpins",
              "FOLL_PIN net settled " + std::to_string(f1) + " -> " + std::to_string(f2) + "; release " + s2.since() +
                  "; populate to release " + s0.since());
        n_ranges = -1;
        const bool ok2 = hint(p + skew, rows, n_ranges, err);
        check(ok2 && n_ranges > 0, "full: after release ple_prefetch hints again", std::to_string(n_ranges));
    }

    // forced failures: each names its call and suggests lazy, and nothing stays pinned
    {
        rt::ple_pin pin;
        err.clear();
        check(!pin.populate(nullptr, 0, err) && names(err, "PLE table pin"), "failure: no table", err);
        err.clear();
        check(!pin.pin(err) && err.find("before populate") != std::string::npos, "failure: pin before populate", err);
        // an unmapped range: reserved, then released
        void * hole = mmap(nullptr, n, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (hole != MAP_FAILED) {
            munmap(hole, n);
            err.clear();
            check(!pin.populate(static_cast<const uint8_t *>(hole), n, err) &&
                      names(err, "madvise(MADV_POPULATE_READ)"),
                  "failure: populate of an unmapped range", err);
        }
        // a populated range unmapped before the registration
        const uint8_t * q = map_file(path, n);
        if (q != nullptr) {
            const bool pop = pin.populate(q, n, err);
            // reason: munmap takes the mapping's address as a mutable pointer
            munmap(const_cast<uint8_t *>(q), n);
            err.clear();
            const int64_t f0 = foll_pinned();
            check(pop && !pin.pin(err) && names(err, "cudaHostRegister") && !pin.pinned(),
                  "failure: registration of an unmapped range", err);
            check(foll_pinned() - f0 < pages, "failure: nothing pinned");
            check(cudaGetLastError() == cudaSuccess, "failure: no sticky CUDA error left");
        }
    }

    // reason: munmap takes the mapping's address as a mutable pointer
    munmap(const_cast<uint8_t *>(p), n);
    unlink(path.c_str());
    printf("ple-pin-test: %s (%d failed)\n", check.failed == 0 ? "PASS" : "FAIL", check.failed);
    return check.failed == 0 ? 0 : 1;
}

} // namespace

int fnfl_ple_pin_test(const char * mode, const char * dir, int device) {
    if (std::strcmp(mode, "probe") == 0) {
        return probe(dir, device);
    }
    if (std::strcmp(mode, "unit") == 0) {
        return unit(dir, device);
    }
    fprintf(stderr, "ple-pin-test: mode is probe or unit\n");
    return 2;
}
