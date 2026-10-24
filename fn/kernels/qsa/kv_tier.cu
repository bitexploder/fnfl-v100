#include "fn/kernels/qsa/kv_tier.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::qsa::tier {
namespace {

constexpr int32_t remap_threads = 1024;
constexpr int32_t bitmap_words = max_blocks / 32;            // 2,048: one bit a block
constexpr int32_t words_a_thread = bitmap_words / remap_threads; // 2
constexpr int32_t gather_ctas = 144;                        // 2 an SM
constexpr int32_t gather_threads = 256;                     // a slot's K and V: 2 x 128 16-byte words
constexpr int32_t block_words = ratio * cell_values * 2 / 16; // 128: a block of K (or V) in 16-byte words

static_assert(words_a_thread * remap_threads == bitmap_words, "every bitmap word has one scanning thread");
static_assert(gather_threads == 2 * block_words, "a thread a 16-byte word of the slot's K and V");

__global__ void __launch_bounds__(remap_threads) remap_kernel(remap_args a) {
    __shared__ uint32_t bits[bitmap_words];
    __shared__ int32_t warp_sum[remap_threads / 32];
    const int32_t tid = static_cast<int32_t>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int32_t lo = a.lo_d != nullptr ? static_cast<int32_t>(*a.lo_d) : a.lo;
    const int32_t base = a.lo_d != nullptr ? a.base + lo / ratio : a.base;
    const int32_t lo_block = lo / ratio;
    const int32_t n_words = (lo_block + 31) / 32;
    for (int32_t i = tid; i < n_words; i += remap_threads) {
        bits[i] = 0u;
    }
    __syncthreads();
    // the blocks below the window, over every row; a tail cell below it is an error
    int32_t below = 0;
    for (int32_t t = 0; t < a.rows; ++t) {
        const int32_t w0 = a.offsets[t] / 4, w1 = a.offsets[t + 1] / 4;
        const int32_t K = static_cast<int32_t>(a.body[w0]);
        for (int32_t i = w0 + 2 + tid; i < w1; i += remap_threads) {
            const uint32_t v = a.body[i];
            if (i < w0 + 2 + K) {
                if (v < static_cast<uint32_t>(lo_block)) {
                    atomicOr(&bits[v / 32], 1u << (v % 32));
                }
            } else if (v < static_cast<uint32_t>(lo)) {
                below = 1;
            }
        }
    }
    const int32_t bad = __syncthreads_or(below);
    // each thread's two words: their set bits' slots, by a scan of the counts over the CTA
    uint32_t w[words_a_thread];
    int32_t n = 0;
#pragma unroll
    for (int32_t j = 0; j < words_a_thread; ++j) {
        const int32_t i = tid * words_a_thread + j;
        w[j] = i < n_words ? bits[i] : 0u;
        n += __popc(w[j]);
    }
    int32_t incl = n;
#pragma unroll
    for (int32_t d = 1; d < 32; d *= 2) {
        const int32_t v = __shfl_up_sync(0xffffffffu, incl, d);
        incl += lane >= d ? v : 0;
    }
    if (lane == 31) {
        warp_sum[warp] = incl;
    }
    __syncthreads();
    if (warp == 0) {
        int32_t s = warp_sum[lane];
#pragma unroll
        for (int32_t d = 1; d < 32; d *= 2) {
            const int32_t v = __shfl_up_sync(0xffffffffu, s, d);
            s += lane >= d ? v : 0;
        }
        warp_sum[lane] = s; // inclusive over the warps
    }
    __syncthreads();
    const int32_t total = warp_sum[remap_threads / 32 - 1];
    const bool ok = bad == 0 && total <= max_gather;
    int32_t slot = (warp > 0 ? warp_sum[warp - 1] : 0) + incl - n;
#pragma unroll
    for (int32_t j = 0; j < words_a_thread && ok; ++j) {
        uint32_t m = w[j];
        while (m != 0u) {
            const int32_t b = (tid * words_a_thread + j) * 32 + (__ffs(m) - 1);
            m &= m - 1u;
            a.table[b] = slot;
            a.list[slot] = b;
            ++slot;
        }
    }
    __syncthreads(); // the table's writes are visible to the CTA
    for (int32_t t = 0; t < a.rows; ++t) {
        const int32_t w0 = a.offsets[t] / 4, w1 = a.offsets[t + 1] / 4;
        const int32_t K = static_cast<int32_t>(a.body[w0]);
        for (int32_t i = w0 + tid; i < w1; i += remap_threads) {
            uint32_t v = a.body[i];
            // refused (not ok): every cell below the window is pointed at its first block, so the attention reads
            // only allocated cells of a round that fails
            if (i >= w0 + 2 && i < w0 + 2 + K && v < static_cast<uint32_t>(lo_block)) {
                v = ok ? static_cast<uint32_t>(base + a.table[v]) : static_cast<uint32_t>(lo_block);
            } else if (i >= w0 + 2 + K && v < static_cast<uint32_t>(lo)) {
                v = static_cast<uint32_t>(lo);
            }
            a.out_body[i] = v;
        }
    }
    if (tid == 0) {
        *a.count = ok ? total : 0;
        if (!ok) {
            atomicOr(a.err, a.err_bit);
        } else if (a.total != nullptr) {
            *a.total += static_cast<unsigned long long>(total);
        }
    }
}

__global__ void __launch_bounds__(gather_threads) gather_kernel(gather_args a) {
    const int32_t n = *a.count;
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t j = tid % block_words;
    const uint4 * src = reinterpret_cast<const uint4 *>(tid < block_words ? a.host_k : a.host_v);
    uint4 * dst = reinterpret_cast<uint4 *>(tid < block_words ? a.area_k : a.area_v);
    for (int32_t s = static_cast<int32_t>(blockIdx.x); s < n; s += static_cast<int32_t>(gridDim.x)) {
        const int64_t b = a.list[s];
        dst[int64_t(s) * block_words + j] = src[b * block_words + j];
    }
}

bool aligned16(const void * p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

} // namespace

bool remap(const remap_args & a, void * stream) {
    if (a.body == nullptr || a.offsets == nullptr || a.rows < 1 || a.rows > max_rows || a.lo < 0 ||
        a.lo > max_cells || a.lo % ratio != 0 || a.base < 0 || a.out_body == nullptr || a.table == nullptr ||
        a.list == nullptr || a.count == nullptr || a.err == nullptr || stream == nullptr) {
        return false;
    }
    remap_kernel<<<1, remap_threads, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool gather(const gather_args & a, void * stream) {
    if (a.list == nullptr || a.count == nullptr || !aligned16(a.host_k) || !aligned16(a.host_v) ||
        !aligned16(a.area_k) || !aligned16(a.area_v) || a.host_k == nullptr || a.area_k == nullptr ||
        stream == nullptr) {
        return false;
    }
    gather_kernel<<<gather_ctas, gather_threads, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::qsa::tier
