#include "fn/runtime/cache_util.h"

#include <cuda.h>

#include <cstdio>
#include <cstring>

namespace fn::runtime::cache {

namespace {

using value_op = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
value_op g_wait = nullptr, g_write = nullptr;

bool cu_ok(CUresult r, const char * what, std::string & err) {
    if (r == CUDA_SUCCESS) {
        return true;
    }
    return fail(err, std::string(what) + ": driver error " + std::to_string(int32_t(r)));
}

} // namespace

bool fail(std::string & err, const std::string & what) {
    err = "expert cache: " + what;
    return false;
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e == cudaSuccess) {
        return true;
    }
    return fail(err, std::string(what) + ": " + cudaGetErrorString(e));
}

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

std::string gib(int64_t b) {
    char s[32];
    std::snprintf(s, sizeof(s), "%.3f GiB", double(b) / double(int64_t(1) << 30));
    return s;
}

bool memops(std::string & err) {
    if (g_wait != nullptr && g_write != nullptr) {
        return true;
    }
    void * w = nullptr;
    void * v = nullptr;
    cudaDriverEntryPointQueryResult q1 = cudaDriverEntryPointSymbolNotFound, q2 = cudaDriverEntryPointSymbolNotFound;
    if (!cuda_ok(cudaGetDriverEntryPointByVersion("cuStreamWaitValue32", &w, 12000, cudaEnableDefault, &q1),
                 "cuStreamWaitValue32", err) ||
        !cuda_ok(cudaGetDriverEntryPointByVersion("cuStreamWriteValue32", &v, 12000, cudaEnableDefault, &q2),
                 "cuStreamWriteValue32", err)) {
        return false;
    }
    if (q1 != cudaDriverEntryPointSuccess || q2 != cudaDriverEntryPointSuccess || w == nullptr || v == nullptr) {
        return fail(err, "the driver has no stream memory operations");
    }
    // a kernel's first load under lazy loading waits for the device, and a stream of the device may be waiting on a
    // flag the service has yet to write: every module loaded at the context's creation, or no cache
    void * lm = nullptr;
    cudaDriverEntryPointQueryResult q3 = cudaDriverEntryPointSymbolNotFound;
    int mode = 0; // CUmoduleLoadingMode: 1 eager, 2 lazy
    if (!cuda_ok(cudaGetDriverEntryPointByVersion("cuModuleGetLoadingMode", &lm, 12000, cudaEnableDefault, &q3),
                 "cuModuleGetLoadingMode", err)) {
        return false;
    }
    // the entry point's type (a driver function pointer returned as void *)
    if (q3 != cudaDriverEntryPointSuccess || lm == nullptr ||
        reinterpret_cast<CUresult (*)(CUmoduleLoadingMode *)>(lm)(reinterpret_cast<CUmoduleLoadingMode *>(&mode)) !=
            CUDA_SUCCESS) {
        return fail(err, "the driver's module loading mode cannot be read");
    }
    if (mode != CU_MODULE_EAGER_LOADING) {
        return fail(err, "lazy module loading: the cache needs CUDA_MODULE_LOADING=EAGER set before CUDA starts");
    }
    g_wait = reinterpret_cast<value_op>(w);
    g_write = reinterpret_cast<value_op>(v);
    return true;
}

// the runtime's stream and device pointer as the driver's (the same handles)
bool stream_write(void * stream, uint32_t * word, uint32_t value, const char * what, std::string & err) {
    return cu_ok(g_write(reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(word), value, 0), what, err);
}

bool stream_wait(void * stream, const uint32_t * word, uint32_t value, const char * what, std::string & err) {
    return cu_ok(g_wait(reinterpret_cast<CUstream>(stream), reinterpret_cast<CUdeviceptr>(word), value,
                        CU_STREAM_WAIT_VALUE_EQ),
                 what, err);
}

uint64_t pin_thread(pthread_t th, int32_t cpu) {
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(th, sizeof(set), &set);
    }
    cpu_set_t got;
    CPU_ZERO(&got);
    uint64_t mask = 0;
    if (pthread_getaffinity_np(th, sizeof(got), &got) == 0) {
        for (int32_t c = 0; c < 64; ++c) {
            mask |= CPU_ISSET(c, &got) ? uint64_t(1) << c : 0;
        }
    }
    return mask;
}

bool owned::dalloc(const devices & dv, int32_t d, size_t bytes, void ** p, bool zero, std::string & err) {
    if (!dv.set(d, err) || !cuda_ok(cudaMalloc(p, bytes), "cudaMalloc", err)) {
        return false;
    }
    dev_allocs[d].push_back(*p);
    dev_bytes[d] += int64_t(bytes);
    return !zero || cuda_ok(cudaMemset(*p, 0, bytes), "cudaMemset", err);
}

bool owned::halloc(size_t bytes, void ** p, bool mapped, std::string & err) {
    const unsigned int fl = cudaHostAllocPortable | (mapped ? cudaHostAllocMapped : 0u);
    if (!cuda_ok(cudaHostAlloc(p, bytes, fl), "cudaHostAlloc", err)) {
        return false;
    }
    host_allocs.push_back(*p);
    host_bytes += int64_t(bytes);
    std::memset(*p, 0, bytes);
    return true;
}

bool owned::event(const devices & dv, int32_t d, void ** e, bool timing, bool block, std::string & err) {
    cudaEvent_t x = nullptr;
    const unsigned int fl =
        (timing ? cudaEventDefault : cudaEventDisableTiming) | (block && dv.quiet ? cudaEventBlockingSync : 0u);
    if (!dv.set(d, err) || !cuda_ok(cudaEventCreateWithFlags(&x, fl), "event", err)) {
        return false;
    }
    *e = x;
    events[d].push_back(x);
    return true;
}

void owned::release(const devices & dv) {
    for (int32_t d = 0; d < 2; ++d) {
        if (dv.dev[d] < 0) {
            continue;
        }
        cudaSetDevice(dv.dev[d]);
        for (void * p : dev_allocs[d]) {
            cudaFree(p);
        }
        for (void * e : events[d]) {
            cudaEventDestroy(static_cast<cudaEvent_t>(e));
        }
        dev_allocs[d].clear();
        events[d].clear();
        dev_bytes[d] = 0;
    }
    for (void * p : host_allocs) {
        cudaFreeHost(p);
    }
    host_allocs.clear();
    host_bytes = 0;
}

} // namespace fn::runtime::cache
