#include "fn/kernels/dense/dense_cold.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::dense {
namespace {

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

// about 12,288 distinct FFMA instructions in a row (16 bytes each: about 192 KB of code), run by one warp on every SM,
// so each SM's instruction caches hold this code and not the previous kernel's
template <int N> __device__ __forceinline__ float chain(float a) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        a = fmaf(a, 1.0f + float(i % 97) * 1e-7f, float(i) * 1e-9f);
    }
    return a;
}

// CR1: the same reads as flush_kernel, through ld.global.cs (__ldcs): L2 allocates the lines evict-first (PTX ISA
// 9.7.9.1), so they go before normal-priority lines when L2 evicts
__global__ void __launch_bounds__(512) flush_ef_kernel(uint4 * p, int64_t n) {
    uint32_t x = 0u;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcs(p + i);
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x9e3779b9u) {
        p[blockIdx.x].x = x; // practically never taken; keeps the loads
    }
}

// CR1's retention arm: n_ef 16-byte words of ef read evict-first and n_nm of nm read with normal loads, interleaved
// evenly in one pass (word j of the pass is normal when floor((j + 1) n_nm / n) > floor(j n_nm / n))
__global__ void __launch_bounds__(512) stream_mix_kernel(uint4 * ef, int64_t n_ef, uint4 * nm, int64_t n_nm) {
    const int64_t n = n_ef + n_nm;
    uint32_t x = 0u;
    for (int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; j < n; j += int64_t(gridDim.x) * blockDim.x) {
        const int64_t a = j * n_nm / n, b = (j + 1) * n_nm / n;
        const uint4 v = b > a ? nm[a] : __ldcs(ef + (j - a));
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x9e3779b9u) {
        ef[blockIdx.x].x = x; // practically never taken; keeps the loads
    }
}

// CR1's retention probe: one thread a block walks n_steps 32-byte sectors of the keep buffer, each load's address taken
// from the previous load (word 0 of a sector holds the next sector's index), and times every load with clock64 from
// before the load to after its value is stored to shared memory. ld.global.cg: L2 only, so no L1 hit hides a miss.
__global__ void __launch_bounds__(32) keep_walk_kernel(const uint32_t * keep, int32_t n_steps, uint32_t * cycles,
                                                       uint32_t * walk_cycles) {
    __shared__ uint32_t seen[32];
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t s = blockIdx.x;
    const long long w0 = clock64();
    for (int32_t k = 0; k < n_steps; ++k) {
        const long long t0 = clock64();
        s = __ldcg(keep + size_t(s) * 8);
        seen[k & 31] = s; // waits for the load before the second clock read
        const long long t1 = clock64();
        cycles[size_t(blockIdx.x) * n_steps + k] = static_cast<uint32_t>(t1 - t0);
    }
    const long long w1 = clock64();
    walk_cycles[blockIdx.x] = static_cast<uint32_t>(w1 - w0) + (seen[0] == 0xffffffffu ? 1u : 0u);
}

__global__ void __launch_bounds__(32) icache_kernel(float * out) {
    const float a = chain<12288>(float(threadIdx.x));
    if (a == 123.0f) {
        out[0] = a; // practically never taken; keeps the chain
    }
}

bool launched() {
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

bool cold_flush(void * scratch, int64_t bytes, void * stream) {
    if (scratch == nullptr || bytes % 16 != 0 || stream == nullptr) {
        return false;
    }
    flush_kernel<<<144, 512, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<uint4 *>(scratch), bytes / 16);
    return launched();
}

bool cold_flush_ef(void * scratch, int64_t bytes, void * stream) {
    if (scratch == nullptr || bytes % 16 != 0 || stream == nullptr) {
        return false;
    }
    flush_ef_kernel<<<144, 512, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<uint4 *>(scratch), bytes / 16);
    return launched();
}

bool cold_r4(int32_t variant, const r4_bufs & b, void * stream, const std::function<bool()> & pre) {
    const bool flush0 = cold_flush(b.flush, b.bytes, stream) && cold_icache(b.evict, stream);
    switch (variant) {
        case 0:
            return flush0;
        case 1:
            return flush0 && pre() && cold_flush_ef(b.flush_ef, b.bytes, stream) && cold_icache(b.evict, stream);
        case 2:
            return flush0 && cold_flush_ef(b.flush_ef, b.bytes, stream) && cold_icache(b.evict, stream);
        case 3:
            return flush0 && pre() && cold_flush(b.flush, b.bytes, stream) && cold_icache(b.evict, stream);
        default:
            return false;
    }
}

bool l2keep_stream(void * ef, int64_t ef_bytes, void * normal, int64_t normal_bytes, void * stream) {
    if (ef == nullptr || ef_bytes % 16 != 0 || ef_bytes <= 0 || normal_bytes % 16 != 0 || normal_bytes < 0 ||
        (normal_bytes > 0 && normal == nullptr) || stream == nullptr) {
        return false;
    }
    stream_mix_kernel<<<144, 512, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint4 *>(ef), ef_bytes / 16, static_cast<uint4 *>(normal), normal_bytes / 16);
    return launched();
}

bool l2keep_walk(const uint32_t * keep, int32_t n_walks, int32_t n_steps, uint32_t * cycles, uint32_t * walk_cycles,
                 void * stream) {
    if (keep == nullptr || cycles == nullptr || walk_cycles == nullptr || n_walks <= 0 || n_steps <= 0 ||
        stream == nullptr) {
        return false;
    }
    keep_walk_kernel<<<n_walks, 32, 0, static_cast<cudaStream_t>(stream)>>>(keep, n_steps, cycles, walk_cycles);
    return launched();
}

bool cold_icache(float * out, void * stream) {
    if (out == nullptr || stream == nullptr) {
        return false;
    }
    icache_kernel<<<160, 32, 0, static_cast<cudaStream_t>(stream)>>>(out);
    return launched();
}

} // namespace fn::dense
