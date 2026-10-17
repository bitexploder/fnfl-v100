#include "fn/kernels/embed/embed.h"

#include "fn/kernels/dense/iq3s.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace fn::embed {
namespace {

constexpr int32_t n_quarter = 4;                              // 64-weight quarters of a superblock
constexpr int32_t n_threads = n_stream * n_super * n_quarter; // 80
constexpr int32_t grid_words = 512;                           // iq3s_grid
static_assert(n_super * 256 == n_own, "a card's row is whole superblocks");

bool al16(const void * p) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

// a 2-byte aligned 16-bit load through the read-only path (the GGUF superblocks are only 2-byte aligned)
__device__ __forceinline__ uint32_t ld16(const uint8_t * p) {
    return __ldg(reinterpret_cast<const unsigned short *>(p));
}
__device__ __forceinline__ uint32_t ld32u(const uint8_t * p) { // two 16-bit loads, little-endian
    return ld16(p) | (ld16(p + 2) << 16);
}

// block b: row b. thread t: stream t / 20, superblock (t % 20) / 4, quarter t % 4, so consecutive lanes write
// consecutive 256-byte runs of a stream
__global__ void __launch_bounds__(n_threads)
embed_kernel(const uint8_t * __restrict__ table, const uint32_t * __restrict__ grids, const int32_t * __restrict__ tok,
             float * __restrict__ hres, float * __restrict__ hpost, float * __restrict__ hx, uint32_t * err) {
    __shared__ uint32_t grid[grid_words];
    const int32_t b = blockIdx.x;
    const int32_t t = threadIdx.x;
    for (int32_t i = t; i < grid_words; i += n_threads) {
        grid[i] = __ldg(grids + i);
    }
    const int32_t id = __ldg(tok + b);
    __syncthreads();
    if (id < 0 || id >= n_vocab) {
        if (t == 0) {
            atomicOr(err, err_token);
        }
        return;
    }
    const int32_t s = t / (n_super * n_quarter);
    const int32_t sb = (t % (n_super * n_quarter)) / n_quarter;
    const int32_t q = t % n_quarter;
    const uint8_t * blk = table + int64_t(id) * row_bytes + sb * super_bytes;

    fn::dense::iq3s_quarter v;
    v.qs = make_uint4(ld32u(blk + 2 + 16 * q), ld32u(blk + 6 + 16 * q), ld32u(blk + 10 + 16 * q),
                      ld32u(blk + 14 + 16 * q));
    v.sg = make_uint2(ld32u(blk + 74 + 8 * q), ld32u(blk + 78 + 8 * q));
    v.qh = ld16(blk + 66 + 2 * q);
    v.sc = __ldg(blk + 106 + q);
    v.d = ld16(blk);
    uint32_t w[16];
    float f0 = 0.0f, f1 = 0.0f;
    fn::dense::iq3s_decode(v, grid, w, f0, f1);

    float * out = hres + (int64_t(b) * n_stream + s) * n_own + sb * 256 + 64 * q;
#pragma unroll
    for (int32_t k = 0; k < 16; ++k) {
        const float f = k < 8 ? f0 : f1; // w[4c + i]: c < 2 is sub-block 2q
        const uint32_t x = w[k];
        float4 o;
        o.x = f * static_cast<float>(static_cast<int8_t>(x & 0xffu));
        o.y = f * static_cast<float>(static_cast<int8_t>((x >> 8) & 0xffu));
        o.z = f * static_cast<float>(static_cast<int8_t>((x >> 16) & 0xffu));
        o.w = f * static_cast<float>(static_cast<int8_t>(x >> 24));
        reinterpret_cast<float4 *>(out)[k] = o;
    }
    // the zero terms of the boundary: hx 320 float4 a row (4 a thread), hpost one float4
    const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int32_t k = 0; k < n_own / 4 / n_threads; ++k) {
        reinterpret_cast<float4 *>(hx + int64_t(b) * n_own)[t + k * n_threads] = z;
    }
    if (t == 0) {
        reinterpret_cast<float4 *>(hpost)[b] = z;
    }
}

} // namespace

bool check_tokens(const int32_t * tok, int32_t rows) {
    if (tok == nullptr || rows < 1 || rows > max_rows) {
        fprintf(stderr, "embed: %d rows (1..%d) or no tokens\n", rows, max_rows);
        return false;
    }
    for (int32_t b = 0; b < rows; ++b) {
        if (tok[b] < 0 || tok[b] >= n_vocab) {
            fprintf(stderr, "embed: row %d: token ID %d outside [0, %d)\n", b, tok[b], n_vocab);
            return false;
        }
    }
    return true;
}

bool embed(int32_t rows, const args & a, void * stream) {
    if (rows < 1 || rows > max_rows || a.table == nullptr || !al16(a.grids) || a.tok == nullptr || !al16(a.hres) ||
        !al16(a.hpost) || !al16(a.hx) || a.err == nullptr || stream == nullptr) {
        fprintf(stderr, "embed: rejected (rows %d, or a null or misaligned pointer)\n", rows);
        return false;
    }
    embed_kernel<<<rows, n_threads, 0, static_cast<cudaStream_t>(stream)>>>(
        a.table, reinterpret_cast<const uint32_t *>(a.grids), a.tok, a.hres, a.hpost, a.hx, a.err);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::embed
