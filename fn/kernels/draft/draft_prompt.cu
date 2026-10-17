#include "fn/kernels/draft/draft_prompt.h"
#include "fn/model/shapes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace fn::draft {
namespace {

constexpr int32_t D  = fn::n_embd;
constexpr int32_t HD = 256;
constexpr int32_t nt = 320;  // a block a (row, segment): 8 values a thread

__device__ __forceinline__ uint32_t h2(float a, float b) {
    return static_cast<uint32_t>(__half_as_ushort(__float2half_rn(a))) |
           (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(b))) << 16);
}

// block (t, segment g): g = 0 the embedding, 1 + s hidden stream s. 320 threads x 8 values; the sum of squares a
// thread's 8 by fmaf chain, the xor tree, the 10 warps in order
__global__ void __launch_bounds__(nt) norm_kernel(const norm_args a) {
    __shared__ float ws[nt / 32];
    const int32_t    t = static_cast<int32_t>(blockIdx.x), g = static_cast<int32_t>(blockIdx.y);
    const int32_t    tid  = static_cast<int32_t>(threadIdx.x);
    const float *    src  = g == 0 ? a.emb + int64_t(t) * D :
                            t == 0 ? a.pred + int64_t(g - 1) * D :
                                     a.hidden + (int64_t(t - 1) * 4 + (g - 1)) * D;
    const float *    gw   = g == 0 ? a.enorm : a.hnorm + int64_t(g - 1) * D;
    const float4     x0   = __ldg(reinterpret_cast<const float4 *>(src) + 2 * tid);
    const float4     x1   = __ldg(reinterpret_cast<const float4 *>(src) + 2 * tid + 1);
    const float      x[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
    float            ss   = x[0] * x[0];
#pragma unroll
    for (int32_t k = 1; k < 8; ++k) {
        ss = fmaf(x[k], x[k], ss);
    }
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if ((tid & 31) == 0) {
        ws[tid >> 5] = ss;
    }
    __syncthreads();
    float tot = ws[0];
#pragma unroll
    for (int32_t w = 1; w < nt / 32; ++w) {
        tot += ws[w];
    }
    const float rs = 1.0f / sqrtf(tot / float(D) + 1e-6f);
    float       y[8];
    bool        bad = false;
#pragma unroll
    for (int32_t k = 0; k < 8; ++k) {
        y[k] = (x[k] * __ldg(gw + 8 * tid + k)) * rs;
        bad  = bad || !(fabsf(y[k]) <= 65504.0f);
    }
    if (bad && a.err != nullptr) {
        atomicOr(a.err, 1u);
    }
    uint16_t * dst                      = g == 0 ? a.e16 + int64_t(t) * D : a.h16 + (int64_t(t) * 4 + (g - 1)) * D;
    reinterpret_cast<uint4 *>(dst)[tid] = make_uint4(h2(y[0], y[1]), h2(y[2], y[3]), h2(y[4], y[5]), h2(y[6], y[7]));
}

__global__ void mixed_kernel(int32_t       rows,
                             int32_t       card,
                             const float * own,
                             const float * peer,
                             uint16_t *    mixed16,
                             uint32_t *    err) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // a pair of values
    if (i >= int64_t(rows) * (D / 2)) {
        return;
    }
    const int64_t r = i / (D / 2);
    const int32_t c = static_cast<int32_t>(i % (D / 2)) * 2;
    const int32_t h = c / 1280, cc = c % 1280;
    const float * src = (h == card ? own : peer) + r * 1280 + cc;
    const float   a = src[0], b = src[1];
    if ((!(fabsf(a) <= 65504.0f) || !(fabsf(b) <= 65504.0f)) && err != nullptr) {
        atomicOr(err, 1u);
    }
    reinterpret_cast<uint32_t *>(mixed16 + r * D)[c / 2] = h2(a, b);
}

// QA2's norm_rope (attend.cu; DA-TC's prepare copied it) with every rope plane the position
__device__ void norm_rope_pos(float (&y)[8],
                              const float * src,
                              const float * w,
                              int32_t       pos,
                              float         theta_scale,
                              int32_t       lane) {
    const float4 a    = __ldg(reinterpret_cast<const float4 *>(src) + 2 * lane);
    const float4 b    = __ldg(reinterpret_cast<const float4 *>(src) + 2 * lane + 1);
    const float  x[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
    float        ss   = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        ss = fmaf(x[e], x[e], ss);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    const float r = 1.0f / sqrtf(ss * (1.0f / HD) + fn::rms_eps);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        y[e] = x[e] * r * __ldg(w + 8 * lane + e);
    }
    float theta = static_cast<float>(pos);
#pragma unroll
    for (int i = 0; i < 31; ++i) {
        theta = i < lane ? theta * theta_scale : theta;
    }
    float s = 0.0f, c = 0.0f;
    sincosf(theta, &s, &c);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float ce    = __shfl_sync(0xffffffffu, c, 8 * (lane & 3) + e);
        const float se    = __shfl_sync(0xffffffffu, s, 8 * (lane & 3) + e);
        const float other = __shfl_xor_sync(0xffffffffu, y[e], 4);
        if (lane < 4) {
            y[e] = y[e] * ce - other * se;
        } else if (lane < 8) {
            y[e] = other * se + y[e] * ce;
        }
    }
}

__global__ void __launch_bounds__(256) prepare_kernel(const kvp_args a) {
    const int32_t lane = threadIdx.x & 31;
    const int32_t t    = static_cast<int32_t>(blockIdx.x) * 8 + static_cast<int32_t>(threadIdx.x >> 5);
    if (t >= a.rows) {
        return;
    }
    const int64_t cell = int64_t(a.pos0) + t;
    float         y[8];
    norm_rope_pos(y, a.kv_k + int64_t(t) * HD, a.k_norm, a.pos0 + t, a.theta_scale, lane);
    reinterpret_cast<uint4 *>(a.k_cache + cell * HD)[lane] =
        make_uint4(h2(y[0], y[1]), h2(y[2], y[3]), h2(y[4], y[5]), h2(y[6], y[7]));
    const float4 v0 = __ldg(reinterpret_cast<const float4 *>(a.kv_v + int64_t(t) * HD) + 2 * lane);
    const float4 v1 = __ldg(reinterpret_cast<const float4 *>(a.kv_v + int64_t(t) * HD) + 2 * lane + 1);
    reinterpret_cast<uint4 *>(a.v_cache + cell * HD)[lane] =
        make_uint4(h2(v0.x, v0.y), h2(v0.z, v0.w), h2(v1.x, v1.y), h2(v1.z, v1.w));
}

bool al16(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

}  // namespace

bool dcp_norm(const norm_args & a, void * stream) {
    if (a.rows < 1 || a.rows > prompt_max_rows || !al16(a.emb) || !al16(a.pred) || (a.rows > 1 && !al16(a.hidden)) ||
        !al16(a.enorm) || !al16(a.hnorm) || !al16(a.e16) || !al16(a.h16) || stream == nullptr) {
        fprintf(stderr, "dcp_norm: rejected\n");
        return false;
    }
    norm_kernel<<<dim3(a.rows, 5), nt, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool dcp_mixed(int32_t       rows,
               int32_t       card,
               const float * own,
               const float * peer,
               uint16_t *    mixed16,
               uint32_t *    err,
               void *        stream) {
    if (rows < 1 || (card != 0 && card != 1) || own == nullptr || peer == nullptr || mixed16 == nullptr ||
        stream == nullptr) {
        return false;
    }
    const int64_t n = int64_t(rows) * (D / 2);
    mixed_kernel<<<static_cast<uint32_t>((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        rows, card, own, peer, mixed16, err);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool dcp_prepare(const kvp_args & a, void * stream) {
    if (a.rows < 1 || a.rows > prompt_max_rows || a.pos0 < 0 || a.pos0 + a.rows > a.max_cells || !al16(a.kv_k) ||
        !al16(a.kv_v) || a.k_norm == nullptr || !al16(a.k_cache) || !al16(a.v_cache) || stream == nullptr) {
        fprintf(stderr, "dcp_prepare: rejected\n");
        return false;
    }
    prepare_kernel<<<(a.rows + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace fn::draft
