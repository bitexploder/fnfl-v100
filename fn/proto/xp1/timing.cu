#include "fn/proto/xp1/timing.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace fn::xp1 {
namespace {

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void round_kernel(uint64_t * stamp, int32_t * counter, int32_t * decode, int32_t n_decode) {
    stamp[0] = globaltimer();
    const int32_t r = *counter;
    *decode = r % n_decode;
    *counter = r + 1;
}

// 64 threads: each reads the counter before thread 0 advances it, then copies one of the 2 x 30 captured IDs
__global__ void round_routes_kernel(uint64_t * stamp, int32_t * counter, int32_t * decode, int32_t n_decode,
                                    const int32_t * routes0, const int32_t * routes1, int32_t * ids0, int32_t * ids1,
                                    int32_t n) {
    const int32_t r = *counter;
    __syncthreads();
    const int32_t d = r % n_decode;
    if (threadIdx.x == 0) {
        stamp[0] = globaltimer();
        *decode = d;
        *counter = r + 1;
    }
    const int32_t t = static_cast<int32_t>(threadIdx.x);
    if (t < n) {
        ids0[t] = routes0[d * 30 + t];
    } else if (t >= 32 && t - 32 < n) {
        ids1[t - 32] = routes1[d * 30 + t - 32];
    }
}

__global__ void stamp_kernel(uint64_t * at) {
    *at = globaltimer();
}

// reads the buffer, so L2 ends full of clean lines: a write flush would leave dirty lines whose write-back the timed
// kernel pays; a store under a condition on the loaded words keeps the reads
__global__ void __launch_bounds__(512) flush_kernel(uint4 * p, int64_t n) {
    uint32_t x = 0u;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const uint4 v = p[i];
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x9e3779b9u) {
        p[blockIdx.x].x = x; // practically never taken; keeps the loads
    }
}

__global__ void clock_probe_kernel(uint64_t * out) {
    const uint64_t t0 = globaltimer();
    const uint64_t c0 = clock64();
    uint64_t t1 = t0;
    while (t1 - t0 < 200000u) {
        t1 = globaltimer();
    }
    out[0] = clock64() - c0;
    out[1] = t1 - t0;
}

constexpr uint32_t wait_budget = 1u << 24;

__device__ __forceinline__ bool wait_flag(const uint32_t * f, uint32_t v) {
    for (uint32_t n = 0; n < wait_budget; ++n) {
        uint32_t x;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(x) : "l"(f) : "memory");
        if (x == v) {
            return true;
        }
    }
    return false;
}

// card 0: t0, flag the peer, wait for its reply, t2
__global__ void ping_kernel(const uint32_t * own, volatile uint32_t * peer, uint64_t * out, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        const uint64_t t0 = globaltimer();
        peer[0] = uint32_t(i + 1);
        __threadfence_system();
        const bool ok = wait_flag(own, uint32_t(i + 1));
        out[2 * i] = t0;
        out[2 * i + 1] = ok ? globaltimer() : 0u;
    }
}

// card 1: wait for the flag, t1, reply
__global__ void pong_kernel(const uint32_t * own, volatile uint32_t * peer, uint64_t * out, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        const bool ok = wait_flag(own, uint32_t(i + 1));
        out[i] = ok ? globaltimer() : 0u;
        peer[0] = uint32_t(i + 1);
        __threadfence_system();
    }
}

bool launched() {
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

bool round_start(uint64_t * stamp, int32_t * counter, int32_t * decode, int32_t n_decode, void * stream) {
    if (stamp == nullptr || counter == nullptr || decode == nullptr || n_decode < 1 || stream == nullptr) {
        return false;
    }
    round_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(stamp, counter, decode, n_decode);
    return launched();
}

bool round_start_routes(uint64_t * stamp, int32_t * counter, int32_t * decode, int32_t n_decode,
                        const int32_t * const routes[2], int32_t * const ids[2], int32_t rows, void * stream) {
    if (stamp == nullptr || counter == nullptr || decode == nullptr || n_decode < 1 || routes[0] == nullptr ||
        routes[1] == nullptr || ids[0] == nullptr || ids[1] == nullptr || rows < 1 || rows > 3 || stream == nullptr) {
        return false;
    }
    round_routes_kernel<<<1, 64, 0, static_cast<cudaStream_t>(stream)>>>(stamp, counter, decode, n_decode, routes[0],
                                                                         routes[1], ids[0], ids[1], 10 * rows);
    return launched();
}

bool stamp(uint64_t * at, void * stream) {
    if (at == nullptr || stream == nullptr) {
        return false;
    }
    stamp_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(at);
    return launched();
}

bool flush_l2(void * scratch, int64_t bytes, void * stream) {
    if (scratch == nullptr || bytes % 16 != 0 || stream == nullptr) {
        return false;
    }
    flush_kernel<<<144, 512, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<uint4 *>(scratch), bytes / 16);
    return launched();
}

bool clock_probe(uint64_t * out, void * stream) {
    if (out == nullptr || stream == nullptr) {
        return false;
    }
    clock_probe_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(out);
    return launched();
}

bool clock_offset(int32_t dev0, int32_t dev1, uint32_t * flags0, uint32_t * flags1, uint64_t * out0, uint64_t * out1,
                  int32_t n, double & offset_ns, double & rtt_ns) {
    cudaStream_t s[2] = {nullptr, nullptr};
    const int32_t dev[2] = {dev0, dev1};
    bool ok = true;
    for (int32_t c = 0; c < 2 && ok; ++c) {
        ok = cudaSetDevice(dev[c]) == cudaSuccess &&
             cudaStreamCreateWithFlags(&s[c], cudaStreamNonBlocking) == cudaSuccess &&
             cudaMemset(c == 0 ? flags0 : flags1, 0, 16) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }
    if (ok) {
        cudaSetDevice(dev1);
        pong_kernel<<<1, 1, 0, s[1]>>>(flags1, flags0, out1, n);
        ok = launched();
        cudaSetDevice(dev0);
        ping_kernel<<<1, 1, 0, s[0]>>>(flags0, flags1, out0, n);
        ok = ok && launched();
    }
    std::vector<uint64_t> h0(size_t(2) * n), h1(static_cast<size_t>(n));
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(dev[c]);
        ok = ok && cudaStreamSynchronize(s[c]) == cudaSuccess;
    }
    cudaSetDevice(dev0);
    ok = ok && cudaMemcpy(h0.data(), out0, h0.size() * 8, cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaSetDevice(dev1);
    ok = ok && cudaMemcpy(h1.data(), out1, h1.size() * 8, cudaMemcpyDeviceToHost) == cudaSuccess;
    for (int32_t c = 0; c < 2; ++c) {
        if (s[c] != nullptr) {
            cudaSetDevice(dev[c]);
            cudaStreamDestroy(s[c]);
        }
    }
    std::vector<double> off, rtt;
    for (int32_t i = 1; ok && i < n; ++i) { // the first trip warms the path
        if (h0[2 * i + 1] == 0u || h1[i] == 0u) {
            return false;
        }
        const double t0 = double(h0[2 * i]), t2 = double(h0[2 * i + 1]), t1 = double(h1[i]);
        off.push_back(t1 - 0.5 * (t0 + t2));
        rtt.push_back(t2 - t0);
    }
    if (!ok || off.empty()) {
        return false;
    }
    std::sort(off.begin(), off.end());
    std::sort(rtt.begin(), rtt.end());
    offset_ns = off[off.size() / 2];
    rtt_ns = rtt[rtt.size() / 2];
    return true;
}

} // namespace fn::xp1
