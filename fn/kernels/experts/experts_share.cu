#include "fn/kernels/experts/experts_share.h"
#include "fn/model/shapes.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::experts {
namespace {

constexpr int32_t D = fn::n_embd; // 2560
constexpr int32_t n_expert = fn::n_expert;
constexpr int32_t row_threads = D / 8; // a row of F16 as 16-byte words, one a thread
constexpr int32_t drop_threads = 1024;
constexpr int32_t drop_warps = drop_threads / 32;

static_assert(D == 2560 && n_expert == 512 && ep_max_tasks <= 2 * drop_threads, "the shapes this file is written for");

__global__ void __launch_bounds__(row_threads) es_gather_kernel(const uint4 * x, const int32_t * rows, uint4 * out) {
    const int32_t i = static_cast<int32_t>(blockIdx.x), c = static_cast<int32_t>(threadIdx.x);
    const int32_t r = __ldg(rows + i);
    out[int64_t(i) * row_threads + c] = __ldg(x + int64_t(r) * row_threads + c);
}

// the tasks [0, counters[0]) whose expert skip leaves, in order, at [0, kept): blocks of 1,024 tasks, each read
// whole before it writes (a task moves only down, never into a later block's unread place); counters[0] = kept
__global__ void __launch_bounds__(drop_threads, 1)
    es_drop_kernel(int4 * tasks, int32_t * counters, const uint8_t * skip) {
    __shared__ int32_t wsum[drop_warps];
    __shared__ int32_t base;
    const int32_t tid = static_cast<int32_t>(threadIdx.x), lane = tid & 31, w = tid >> 5;
    if (counters[1] != 0) {
        return;
    }
    const int32_t n = counters[0];
    if (tid == 0) {
        base = 0;
    }
    __syncthreads();
    for (int32_t b0 = 0; b0 < n; b0 += drop_threads) {
        const int32_t i = b0 + tid;
        const int4 t = i < n ? tasks[i] : make_int4(0, 0, 0, 0);
        const bool keep = i < n && skip[t.x] == 0;
        const uint32_t m = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) {
            wsum[w] = __popc(m);
        }
        __syncthreads();
        int32_t before = base;
        for (int32_t k = 0; k < w; ++k) {
            before += wsum[k];
        }
        if (keep) {
            tasks[before + __popc(m & ((1u << lane) - 1u))] = t;
        }
        __syncthreads();
        if (tid == 0) {
            for (int32_t k = 0; k < drop_warps; ++k) {
                base += wsum[k];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        counters[0] = base;
    }
}

__global__ void __launch_bounds__(row_threads) es_add_kernel(const int32_t * pairs, const uint4 * rows16,
                                                              const int32_t * pos, uint4 * part16, int32_t bad,
                                                              uint32_t * err, uint32_t err_bit) {
    const int32_t i = static_cast<int32_t>(blockIdx.x), c = static_cast<int32_t>(threadIdx.x);
    if (bad != 0 && i == 0 && c == 0) {
        atomicOr(err, err_bit);
    }
    const int32_t q = __ldg(pos + __ldg(pairs + i));
    part16[int64_t(q) * row_threads + c] =
        rows16 != nullptr ? __ldg(rows16 + int64_t(i) * row_threads + c) : make_uint4(0u, 0u, 0u, 0u);
}

bool aligned(const void * p, uintptr_t a) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & (a - 1u)) == 0;
}

} // namespace

bool es_gather(const uint16_t * x, const int32_t * rows, int32_t n, uint16_t * out, void * stream) {
    if (!aligned(x, 16) || !aligned(rows, 4) || !aligned(out, 16) || n < 1 || n > ep_max_rows || stream == nullptr) {
        return false;
    }
    es_gather_kernel<<<n, row_threads, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const uint4 *>(x), rows, reinterpret_cast<uint4 *>(out));
    return cudaPeekAtLastError() == cudaSuccess;
}

bool es_drop(const ep_route & r, const uint8_t * skip, void * stream) {
    if (!aligned(r.tasks, 16) || !aligned(r.counters, 4) || skip == nullptr || stream == nullptr) {
        return false;
    }
    es_drop_kernel<<<1, drop_threads, 0, static_cast<cudaStream_t>(stream)>>>(reinterpret_cast<int4 *>(r.tasks),
                                                                              r.counters, skip);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool es_add(const int32_t * pairs, const uint16_t * rows16, int32_t n, const int32_t * pos, uint16_t * part16,
            int32_t bad, uint32_t * err, uint32_t err_bit, void * stream) {
    if (!aligned(pairs, 4) || (rows16 != nullptr && !aligned(rows16, 16)) || !aligned(pos, 4) ||
        !aligned(part16, 16) || err == nullptr || n < 1 || n > ep_max_pairs || stream == nullptr) {
        return false;
    }
    es_add_kernel<<<n, row_threads, 0, static_cast<cudaStream_t>(stream)>>>(
        pairs, reinterpret_cast<const uint4 *>(rows16), pos, reinterpret_cast<uint4 *>(part16), bad, err, err_bit);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::experts
