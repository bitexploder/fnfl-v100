#include "fn/kernels/dense/iq3s.cuh"
#include "fn/kernels/embed/embed.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace fn::embed {
namespace {

constexpr int32_t n_full      = 2 * n_own;
constexpr int32_t grid_words  = 512;
constexpr int32_t row_threads = n_super * 4;  // a thread a superblock quarter (64 values)

__device__ __forceinline__ uint32_t ld16(const uint8_t * p) {
    return __ldg(reinterpret_cast<const unsigned short *>(p));
}

__device__ __forceinline__ uint32_t ld32u(const uint8_t * p) {
    return ld16(p) | (ld16(p + 2) << 16);
}

// block b: row b; thread t: superblock t / 4, quarter t % 4 (embed_kernel's decode, unchanged)
__global__ void __launch_bounds__(row_threads) embed_rows_kernel(const rows_args a) {
    __shared__ uint32_t grid[grid_words];
    const int32_t       b = static_cast<int32_t>(blockIdx.x);
    const int32_t       t = static_cast<int32_t>(threadIdx.x);
    for (int32_t i = t; i < grid_words; i += row_threads) {
        grid[i] = __ldg(reinterpret_cast<const uint32_t *>(a.grids) + i);
    }
    const int32_t id = __ldg(a.tok + b);
    __syncthreads();
    if (id < 0 || id >= n_vocab) {
        if (t == 0) {
            atomicOr(a.err, err_token);
        }
        return;
    }
    const int32_t           sb = t / 4, q = t % 4;
    const uint8_t *         blk = a.table + int64_t(id) * row_bytes + sb * super_bytes;
    fn::dense::iq3s_quarter v;
    v.qs = make_uint4(ld32u(blk + 2 + 16 * q), ld32u(blk + 6 + 16 * q), ld32u(blk + 10 + 16 * q),
                      ld32u(blk + 14 + 16 * q));
    v.sg = make_uint2(ld32u(blk + 74 + 8 * q), ld32u(blk + 78 + 8 * q));
    v.qh = ld16(blk + 66 + 2 * q);
    v.sc = __ldg(blk + 106 + q);
    v.d  = ld16(blk);
    uint32_t w[16];
    float    f0 = 0.0f, f1 = 0.0f;
    fn::dense::iq3s_decode(v, grid, w, f0, f1);
    const int32_t c0 = sb * 256 + 64 * q;  // in the card's half
    const int32_t cf = a.card * n_own + c0;
#pragma unroll
    for (int32_t k = 0; k < 16; ++k) {
        const float    f = k < 8 ? f0 : f1;
        const uint32_t x = w[k];
        float4         o;
        o.x = f * static_cast<float>(static_cast<int8_t>(x & 0xffu));
        o.y = f * static_cast<float>(static_cast<int8_t>((x >> 8) & 0xffu));
        o.z = f * static_cast<float>(static_cast<int8_t>((x >> 16) & 0xffu));
        o.w = f * static_cast<float>(static_cast<int8_t>(x >> 24));
        reinterpret_cast<float4 *>(a.own + int64_t(b) * n_own + c0)[k]   = o;
        reinterpret_cast<float4 *>(a.full + int64_t(b) * n_full + cf)[k] = o;
#pragma unroll
        for (int32_t s = 0; s < n_stream; ++s) {
            reinterpret_cast<float4 *>(a.streams + (int64_t(b) * n_stream + s) * n_full + cf)[k] = o;
        }
    }
}

// a thread a float4 of the peer's half
__global__ void embed_join_kernel(const rows_args a, const float * peer) {
    const int64_t     i  = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr int32_t q4 = n_own / 4;
    if (i >= int64_t(a.rows) * q4) {
        return;
    }
    const int64_t b = i / q4, c4 = i % q4;
    const int32_t id = __ldg(a.tok + b);
    if (id < 0 || id >= n_vocab) {
        return;  // embed_rows flagged it
    }
    const float4  o  = __ldg(reinterpret_cast<const float4 *>(peer + b * n_own) + c4);
    const int64_t cf = int64_t(1 - a.card) * n_own + 4 * c4;
    *reinterpret_cast<float4 *>(a.full + b * n_full + cf) = o;
#pragma unroll
    for (int32_t s = 0; s < n_stream; ++s) {
        *reinterpret_cast<float4 *>(a.streams + (b * n_stream + s) * n_full + cf) = o;
    }
}

bool al16(const void * p) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

bool rows_ok(const rows_args & a, void * stream) {
    if (a.rows < 1 || a.rows > prompt_max_rows || a.table == nullptr || !al16(a.grids) || a.tok == nullptr ||
        !al16(a.own) || !al16(a.full) || !al16(a.streams) || a.err == nullptr || (a.card != 0 && a.card != 1) ||
        stream == nullptr) {
        fprintf(stderr, "embed_rows: rejected (rows %d, card %d, or a null or misaligned pointer)\n", a.rows, a.card);
        return false;
    }
    return true;
}

}  // namespace

bool embed_rows(const rows_args & a, void * stream) {
    if (!rows_ok(a, stream)) {
        return false;
    }
    embed_rows_kernel<<<a.rows, row_threads, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool embed_join(const rows_args & a, const float * peer, void * stream) {
    if (!rows_ok(a, stream) || !al16(peer)) {
        return false;
    }
    const int64_t n = int64_t(a.rows) * (n_own / 4);
    embed_join_kernel<<<static_cast<uint32_t>((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(a, peer);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace fn::embed
