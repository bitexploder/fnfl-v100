#pragma once

// ME3 (the cleanup): what the expert cache's units (expert_cache.h) share and nothing else: error and clock helpers,
// the caller's device kept, the driver's stream memory operations, thread pinning, the devices fixed at init, and one
// unit's allocations. Implementation only: included by the units' sources and headers, never outside fn/runtime.

#include <cuda_runtime_api.h>
#include <pthread.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime::cache {

constexpr int32_t n_ex = 512;
constexpr int32_t D = 2560;
constexpr int32_t max_prompt_rows = 4096; // a half of a prompt chunk
using clk = std::chrono::steady_clock;

bool fail(std::string & err, const std::string & what); // err = "expert cache: " + what; false
bool cuda_ok(cudaError_t e, const char * what, std::string & err);
double since(clk::time_point t0);
std::string gib(int64_t b);

// the caller's current device back at the end of a public call (cached TP sets each device in turn)
struct device_guard {
    int dev = -1;
    device_guard() { cudaGetDevice(&dev); }
    ~device_guard() {
        if (dev >= 0) {
            cudaSetDevice(dev);
        }
    }
    device_guard(const device_guard &) = delete;
    device_guard & operator=(const device_guard &) = delete;
};

// the driver's stream memory operations (CUDA 12's forms, through the runtime's entry points: no driver link), checked
// once with the module loading mode (eager: a kernel's first load would otherwise wait for the device). The eager
// windows' flag waits, the service's flag writes and the swap passes' counts use them; a replay holds none
bool memops(std::string & err);
bool stream_write(void * stream, uint32_t * word, uint32_t value, const char * what, std::string & err);
bool stream_wait(void * stream, const uint32_t * word, uint32_t value, const char * what, std::string & err);

// ME2b: a thread pinned to one CPU (-1: left as it is), and its affinity mask read back (CPUs 0..63)
uint64_t pin_thread(pthread_t th, int32_t cpu);

// the devices, fixed at init and read by every unit: one device (both cards) or two (cached TP), each device's copy
// stream (the service's lane copies, flags and CPU products; the loan's staging; the residency's fills), its flags
struct devices {
    int32_t nd = 1;    // devices: 1 (both cards on one) or 2 (cached TP)
    int32_t parts = 4; // table parts a device: 4 or 2
    int32_t dev[2] = {-1, -1};
    bool quiet = false; // one card: the waits block or back off, the threads pinned (two cards as ME2)
    unsigned int flags[2] = {0, 0};
    void * copy[2] = {nullptr, nullptr};
    int32_t dev_of(int32_t card) const { return nd == 1 ? 0 : card; }
    // card's gate/up (or down) part index in a device's tables
    int32_t part_of(int32_t card, bool down) const { return (nd == 1 ? 2 * card : 0) + (down ? 1 : 0); }
    bool set(int32_t d, std::string & err) const { return cuda_ok(cudaSetDevice(dev[d]), "cudaSetDevice", err); }
};

// one unit's allocations and events, freed by release once every device is idle (the owner's lifetime covers every
// queued consumer: the facade synchronizes the devices first)
struct owned {
    std::vector<void *> dev_allocs[2];
    int64_t dev_bytes[2] = {0, 0};
    std::vector<void *> host_allocs;
    int64_t host_bytes = 0;
    std::vector<void *> events[2];
    bool dalloc(const devices & dv, int32_t d, size_t bytes, void ** p, bool zero, std::string & err);
    bool halloc(size_t bytes, void ** p, bool mapped, std::string & err); // zeroed, portable
    // ME2b item 3: block, an event a host thread synchronizes on (cudaEventBlockingSync on one card: the wait sleeps)
    bool event(const devices & dv, int32_t d, void ** e, bool timing, bool block, std::string & err);
    void release(const devices & dv);
};

} // namespace fn::runtime::cache
