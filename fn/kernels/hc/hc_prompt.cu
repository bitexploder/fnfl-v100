#include "fn/kernels/hc/hc.h"
#include "fn/kernels/hc/hc_prompt.h"
#include "fn/kernels/tc/tc_gemm.cuh"
#include "fn/kernels/tc/tc_ladder.cuh"
#include "fn/model/shapes.h"
#include "fn/xchg/prompt_xchg.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace fn::hc {
namespace {

constexpr int32_t D       = fn::n_embd;   // 2560
constexpr int32_t F       = n_own;        // 1280
constexpr int32_t J       = n_rows_down;  // 324
constexpr int32_t tile    = 64;
constexpr int32_t kstep   = 32;
constexpr int32_t down_jt = (J + tile - 1) / tile;  // 6 j-tiles (the last holds the 4 injection rows)
constexpr int32_t up_c    = 32;                     // output features a hcp_up block

__device__ __forceinline__ float bf16f(uint32_t h) {
    return __uint_as_float(h << 16);
}

__device__ __forceinline__ float sigm(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// s += t with Kahan compensation c (no multiply, so no contraction applies)
__device__ __forceinline__ void kadd(float & s, float & c, float t) {
    const float y = t - c;
    const float u = s + y;
    c             = (u - s) - y;
    s             = u;
}

// ---- weights: 8 consecutive K values of one row of the down or up matrix ------------------------------------------
// down row (s, j) over the summed features i: replicated i in 0..2559 (half i / 1280); split i in 0..1279 (own)
// up row (c, s) over j in 0..319: replicated c in 0..2559 (half c / 1280); split c own
template <hcp_form FORM>
__device__ __forceinline__ void down8(const hcp_weights & w, int32_t s, int32_t j, int32_t i, float (&o)[8]) {
    if constexpr (FORM == hcp_form::replicated) {
        const int32_t  h = i / F, ii = i - h * F;
        const uint4    v    = __ldg(reinterpret_cast<const uint4 *>(w.down[h] + (int64_t(s) * J + j) * F + ii));
        const uint32_t x[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int32_t k = 0; k < 4; ++k) {
            o[2 * k]     = bf16f(x[k] & 0xffffu);
            o[2 * k + 1] = bf16f(x[k] >> 16);
        }
    } else {
        const int64_t  row  = int64_t(s) * J + j;
        const uint2    q    = __ldg(reinterpret_cast<const uint2 *>(w.q8_down + row * F + i));
        const float    d    = __half2float(__ushort_as_half(__ldg(
            reinterpret_cast<const unsigned short *>(w.q8_down + int64_t(4) * J * F + (row * (F / 32) + i / 32) * 2))));
        const uint32_t x[2] = { q.x, q.y };
#pragma unroll
        for (int32_t k = 0; k < 8; ++k) {
            o[k] = d * static_cast<float>(static_cast<int8_t>((x[k >> 2] >> (8 * (k & 3))) & 0xffu));
        }
    }
}

template <hcp_form FORM>
__device__ __forceinline__ void up8(const hcp_weights & w, int32_t c, int32_t s, int32_t j, float (&o)[8]) {
    if constexpr (FORM == hcp_form::replicated) {
        const int32_t  h = c / F, cc = c - h * F;
        const uint4    v    = __ldg(reinterpret_cast<const uint4 *>(w.up[h] + (int64_t(cc) * 4 + s) * n_down + j));
        const uint32_t x[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int32_t k = 0; k < 4; ++k) {
            o[2 * k]     = bf16f(x[k] & 0xffffu);
            o[2 * k + 1] = bf16f(x[k] >> 16);
        }
    } else {
        const int64_t  row  = int64_t(c) * 4 + s;
        const uint2    q    = __ldg(reinterpret_cast<const uint2 *>(w.q8_up + row * n_down + j));
        const float    d    = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short *>(
            w.q8_up + int64_t(F) * 4 * n_down + (row * (n_down / 32) + j / 32) * 2))));
        const uint32_t x[2] = { q.x, q.y };
#pragma unroll
        for (int32_t k = 0; k < 8; ++k) {
            o[k] = d * static_cast<float>(static_cast<int8_t>((x[k >> 2] >> (8 * (k & 3))) & 0xffu));
        }
    }
}

template <hcp_form FORM> __device__ __forceinline__ float gamma_of(const hcp_weights & w, int32_t s, int32_t i) {
    if constexpr (FORM == hcp_form::replicated) {
        const int32_t h = i / F;
        return __ldg(w.gamma[h] + s * F + (i - h * F));
    } else {
        return __ldg(w.gamma_own + s * F + i);
    }
}

// ---- hcp_down: grid (6 j-tiles, 4 streams, row tiles), 256 threads, 64 rows x 64 j a block ----
template <hcp_form FORM> __global__ void __launch_bounds__(256) hcp_down_kernel(const hcp_args a) {
    constexpr int32_t NF = FORM == hcp_form::replicated ? D : F;  // features summed
    __shared__ float  xs[kstep][tile + 4];
    __shared__ float  ws[kstep][tile + 4];
    __shared__ float  vpart[tile][4];
    const int32_t     tid = static_cast<int32_t>(threadIdx.x);
    const int32_t     ty = tid >> 4, tx = tid & 15;
    const int32_t     jt = static_cast<int32_t>(blockIdx.x), s = static_cast<int32_t>(blockIdx.y);
    const int32_t     r0 = static_cast<int32_t>(blockIdx.z) * tile, j0 = jt * tile;
    const int32_t     lr = tid >> 2, lk = (tid & 3) * 8;
    const bool        xr = r0 + lr < a.rows, wj = j0 + lr < J;
    const float *     xp = a.x + (int64_t(xr ? r0 + lr : 0) * 4 + s) * NF + lk;
    float             acc[4][4], cmp[4][4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
            cmp[i][j] = 0.0f;
        }
    }
    float vs = 0.0f, vc = 0.0f;  // the loader row's sum of squares (the last j-tile's blocks write v)
    for (int32_t k0 = 0; k0 < NF; k0 += kstep) {
        {
            float xv[8], wv[8];
            if (xr) {
                const float4 x0 = __ldg(reinterpret_cast<const float4 *>(xp + k0));
                const float4 x1 = __ldg(reinterpret_cast<const float4 *>(xp + k0) + 1);
                xv[0] = x0.x, xv[1] = x0.y, xv[2] = x0.z, xv[3] = x0.w;
                xv[4] = x1.x, xv[5] = x1.y, xv[6] = x1.z, xv[7] = x1.w;
            } else {
#pragma unroll
                for (int32_t k = 0; k < 8; ++k) {
                    xv[k] = 0.0f;
                }
            }
            float sq = xv[0] * xv[0];
#pragma unroll
            for (int32_t k = 1; k < 8; ++k) {
                sq = fmaf(xv[k], xv[k], sq);
            }
            kadd(vs, vc, sq);
            if (wj) {
                down8<FORM>(a.w, s, j0 + lr, k0 + lk, wv);
            } else {
#pragma unroll
                for (int32_t k = 0; k < 8; ++k) {
                    wv[k] = 0.0f;
                }
            }
#pragma unroll
            for (int32_t k = 0; k < 8; ++k) {
                xs[lk + k][lr] = gamma_of<FORM>(a.w, s, k0 + lk + k) * xv[k];
                ws[lk + k][lr] = wv[k];
            }
        }
        __syncthreads();
        float tmp[4][4];
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (int32_t j = 0; j < 4; ++j) {
                tmp[i][j] = 0.0f;
            }
        }
#pragma unroll 8
        for (int32_t kk = 0; kk < kstep; ++kk) {
            const float4 xa    = *reinterpret_cast<const float4 *>(&xs[kk][4 * ty]);
            const float4 wa    = *reinterpret_cast<const float4 *>(&ws[kk][4 * tx]);
            const float  xq[4] = { xa.x, xa.y, xa.z, xa.w }, wq[4] = { wa.x, wa.y, wa.z, wa.w };
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    tmp[i][j] = fmaf(xq[i], wq[j], tmp[i][j]);
                }
            }
        }
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (int32_t j = 0; j < 4; ++j) {
                kadd(acc[i][j], cmp[i][j], tmp[i][j]);
            }
        }
        __syncthreads();
    }
    if (jt == down_jt - 1) {
        vpart[lr][tid & 3] = vs;
        __syncthreads();
        if (tid < tile && r0 + tid < a.rows) {
            const float v                          = ((vpart[tid][0] + vpart[tid][1]) + vpart[tid][2]) + vpart[tid][3];
            a.mom[int64_t(r0 + tid) * hcp_mom + s] = v;
        }
    }
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        const int32_t row = r0 + 4 * ty + i;
        if (row >= a.rows) {
            continue;
        }
#pragma unroll
        for (int32_t jj = 0; jj < 4; ++jj) {
            const int32_t j = j0 + 4 * tx + jj;
            if (j < n_down) {
                a.mom[int64_t(row) * hcp_mom + 4 + n_down * s + j] = acc[i][jj];
            } else if (j < J) {
                a.mom[int64_t(row) * hcp_mom + 4 + 4 * n_down + 4 * s + (j - n_down)] = acc[i][jj];
            }
        }
    }
}

// ---- hcp_mid: a block of 128 threads a row ----
__global__ void __launch_bounds__(128) hcp_mid_kernel(const hcp_args a) {
    const int32_t r   = static_cast<int32_t>(blockIdx.x);
    const int32_t t   = static_cast<int32_t>(threadIdx.x);
    const float * mo  = a.mom + int64_t(r) * hcp_mom;
    const float * mp  = a.mom_peer != nullptr ? a.mom_peer + int64_t(r) * hcp_mom : nullptr;
    // card order: card 0's moments first (the sum is commutative, so both cards hold the same bits)
    const auto    tot = [&](int32_t i) {
        if (mp == nullptr) {
            return mo[i];
        }
        return a.card == 0 ? mo[i] + mp[i] : mp[i] + mo[i];
    };
    float ri[4];
#pragma unroll
    for (int32_t s = 0; s < 4; ++s) {
        ri[s] = 1.0f / sqrtf(tot(s) / float(D) + fn::rms_eps);
    }
    if (t < 4) {
        a.rinv[int64_t(r) * 4 + t] = ri[t];
    }
    for (int32_t j = t; j < J; j += 128) {
        const int32_t base    = j < n_down ? 4 + j : 4 + 4 * n_down + (j - n_down);
        const int32_t sstride = j < n_down ? n_down : 4;
        float         z       = tot(base) * ri[0];
#pragma unroll
        for (int32_t s = 1; s < 4; ++s) {
            z = fmaf(tot(base + s * sstride), ri[s], z);
        }
        z = 0.25f * z;
        if (j < n_down) {
            a.low[int64_t(r) * n_down + j] = z * sigm(z);
        } else {
            a.inject[int64_t(r) * 4 + (j - n_down)] = 2.0f * sigm(z);
        }
    }
}

// ---- hcp_up: grid (output feature tiles of 32, row tiles of 64), 256 threads ----
// thread (ty, tx): rows 4 ty .. + 3, features 2 tx, 2 tx + 1 of the block's 32, all four streams
template <hcp_form FORM> __global__ void __launch_bounds__(256) hcp_up_kernel(const hcp_args a) {
    constexpr int32_t NC = FORM == hcp_form::replicated ? D : F;  // output features (and the stream width)
    __shared__ float  ls[kstep][tile + 4];
    __shared__ float  wsm[kstep][4 * up_c + 4];
    const int32_t     tid = static_cast<int32_t>(threadIdx.x);
    const int32_t     ty = tid >> 4, tx = tid & 15;
    const int32_t     c0 = static_cast<int32_t>(blockIdx.x) * up_c, r0 = static_cast<int32_t>(blockIdx.y) * tile;
    float             acc[4][2][4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
        for (int32_t c = 0; c < 2; ++c) {
#pragma unroll
            for (int32_t s = 0; s < 4; ++s) {
                acc[i][c][s] = 0.0f;
            }
        }
    }
    // loaders: low row lr, k lk .. + 7; weights output o = tid / 2 (feature o / 4, stream o % 4), k 16 (tid % 2) .. + 15
    const int32_t lr = tid >> 2, lk = (tid & 3) * 8;
    const bool    lrow = r0 + lr < a.rows;
    const int32_t wo = tid >> 1, wk = (tid & 1) * 16;
    const int32_t wc = c0 + wo / 4, wsi = wo % 4;
    for (int32_t k0 = 0; k0 < n_down; k0 += kstep) {
        {
            float lv[8];
            if (lrow) {
                const float4 l0 = __ldg(reinterpret_cast<const float4 *>(a.low + int64_t(r0 + lr) * n_down + k0 + lk));
                const float4 l1 =
                    __ldg(reinterpret_cast<const float4 *>(a.low + int64_t(r0 + lr) * n_down + k0 + lk) + 1);
                lv[0] = l0.x, lv[1] = l0.y, lv[2] = l0.z, lv[3] = l0.w;
                lv[4] = l1.x, lv[5] = l1.y, lv[6] = l1.z, lv[7] = l1.w;
            } else {
#pragma unroll
                for (int32_t k = 0; k < 8; ++k) {
                    lv[k] = 0.0f;
                }
            }
#pragma unroll
            for (int32_t k = 0; k < 8; ++k) {
                ls[lk + k][lr] = lv[k];
            }
            float w0[8], w1[8];
            up8<FORM>(a.w, wc, wsi, k0 + wk, w0);
            up8<FORM>(a.w, wc, wsi, k0 + wk + 8, w1);
#pragma unroll
            for (int32_t k = 0; k < 8; ++k) {
                wsm[wk + k][wo]     = w0[k];
                wsm[wk + 8 + k][wo] = w1[k];
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int32_t kk = 0; kk < kstep; ++kk) {
            const float4 la       = *reinterpret_cast<const float4 *>(&ls[kk][4 * ty]);
            const float  lq[4]    = { la.x, la.y, la.z, la.w };
            // outputs 8 tx .. 8 tx + 7: features 2 tx, 2 tx + 1, streams 0..3
            const float4 wa       = *reinterpret_cast<const float4 *>(&wsm[kk][8 * tx]);
            const float4 wb       = *reinterpret_cast<const float4 *>(&wsm[kk][8 * tx + 4]);
            const float  wq[2][4] = {
                { wa.x, wa.y, wa.z, wa.w },
                { wb.x, wb.y, wb.z, wb.w }
            };
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
                for (int32_t c = 0; c < 2; ++c) {
#pragma unroll
                    for (int32_t s = 0; s < 4; ++s) {
                        acc[i][c][s] = fmaf(lq[i], wq[c][s], acc[i][c][s]);
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        const int32_t row = r0 + 4 * ty + i;
        if (row >= a.rows) {
            continue;
        }
        float ri[4];
#pragma unroll
        for (int32_t s = 0; s < 4; ++s) {
            ri[s] = a.rinv[int64_t(row) * 4 + s];
        }
#pragma unroll
        for (int32_t c = 0; c < 2; ++c) {
            const int32_t cf    = c0 + 2 * tx + c;
            float         mixed = 0.0f;
#pragma unroll
            for (int32_t s = 0; s < 4; ++s) {
                const float gx = gamma_of<FORM>(a.w, s, cf) * a.x[(int64_t(row) * 4 + s) * NC + cf];
                mixed          = fmaf(gx * ri[s], sigm(acc[i][c][s]), mixed);
            }
            mixed                                   = 0.25f * mixed;
            a.mixed[int64_t(row) * a.ld_mixed + cf] = mixed;
            if (a.mixed16 != nullptr) {
                if (!(fabsf(mixed) <= 65504.0f) && a.err != nullptr) {
                    atomicOr(a.err, 1u);
                }
                a.mixed16[int64_t(row) * a.ld_mixed16 + cf] = __half_as_ushort(__float2half_rn(mixed));
            }
        }
    }
}

// ---- PB3: the replicated form on tensor cores ----------------------------------------------------------------------
constexpr int32_t NB = 4 * D;  // up rows (feature, stream)

// the up row b's feature and stream (hc_prompt.h: a lane's columns 8 g + {0, 1, 4, 5} or + {2, 3, 6, 7} are one
// feature's streams 0..3)
__device__ __forceinline__ int32_t up_feature(int32_t b) {
    return 2 * (b >> 3) + ((b >> 1) & 1);
}

__device__ __forceinline__ int32_t up_stream(int32_t b) {
    return (b & 1) | (((b >> 2) & 1) << 1);
}

using fn::tc::pack2;
using fn::tc::pow2;
using fn::tc::range_exp;
using fn::tc::warp_max;

// hct_convert: a warp a row; rows 0 .. 1,295 down (s, j), then the 10,240 up rows
constexpr int32_t cv_down_rows = 4 * J;
constexpr int32_t cv_rows      = cv_down_rows + NB;

// APF1 Part 4: VALUES false writes the row scales only (hct_scales: the fused route's prepared scales, bitwise these)
template <bool VALUES = true>
__global__ void __launch_bounds__(256) hct_convert_kernel(const hcp_weights w, const hct_weights o) {
    const int32_t row  = static_cast<int32_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (row >= cv_rows) {
        return;
    }
    if (row < cv_down_rows) {
        // K 2,560: lane's 8 values at 8 lane + 256 t
        const int32_t s = row / J, j = row % J;
        uint4         v[10];
        float         m = 0.0f;
#pragma unroll
        for (int32_t t = 0; t < 10; ++t) {
            const int32_t i = 8 * lane + 256 * t, h = i / F;
            v[t]            = __ldg(reinterpret_cast<const uint4 *>(w.down[h] + (int64_t(s) * J + j) * F + (i - h * F)));
            const uint32_t x[4] = { v[t].x, v[t].y, v[t].z, v[t].w };
#pragma unroll
            for (int32_t k = 0; k < 4; ++k) {
                m = fmaxf(m, fmaxf(fabsf(bf16f(x[k] & 0xffffu)), fabsf(bf16f(x[k] >> 16))));
            }
        }
        const int32_t e  = range_exp(warp_max(m));
        const float   sc = pow2(e);
#pragma unroll
        for (int32_t t = 0; t < 10 && VALUES; ++t) {
            const uint32_t x[4] = { v[t].x, v[t].y, v[t].z, v[t].w };
            uint4          out;
            out.x = pack2(bf16f(x[0] & 0xffffu) * sc, bf16f(x[0] >> 16) * sc);
            out.y = pack2(bf16f(x[1] & 0xffffu) * sc, bf16f(x[1] >> 16) * sc);
            out.z = pack2(bf16f(x[2] & 0xffffu) * sc, bf16f(x[2] >> 16) * sc);
            out.w = pack2(bf16f(x[3] & 0xffffu) * sc, bf16f(x[3] >> 16) * sc);
            *reinterpret_cast<uint4 *>(o.down + int64_t(row) * D + 8 * lane + 256 * t) = out;
        }
        if (lane == 0) {
            o.down_s[row] = pow2(-e);
        }
        return;
    }
    // K 320: 40 groups of 8, lane's groups lane and lane + 32
    const int32_t b = row - cv_down_rows, f = up_feature(b), s = up_stream(b);
    const int32_t h = f / F, cc = f - h * F;
    const uint16_t * src = w.up[h] + (int64_t(cc) * 4 + s) * n_down;
    uint4            v[2];
    float            m = 0.0f;
#pragma unroll
    for (int32_t t = 0; t < 2; ++t) {
        const int32_t g = lane + 32 * t;
        v[t]            = g < n_down / 8 ? __ldg(reinterpret_cast<const uint4 *>(src) + g) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t x[4] = { v[t].x, v[t].y, v[t].z, v[t].w };
#pragma unroll
        for (int32_t k = 0; k < 4; ++k) {
            m = fmaxf(m, fmaxf(fabsf(bf16f(x[k] & 0xffffu)), fabsf(bf16f(x[k] >> 16))));
        }
    }
    const int32_t e  = range_exp(warp_max(m));
    const float   sc = pow2(e);
#pragma unroll
    for (int32_t t = 0; t < 2 && VALUES; ++t) {
        const int32_t g = lane + 32 * t;
        if (g < n_down / 8) {
            const uint32_t x[4] = { v[t].x, v[t].y, v[t].z, v[t].w };
            uint4          out;
            out.x = pack2(bf16f(x[0] & 0xffffu) * sc, bf16f(x[0] >> 16) * sc);
            out.y = pack2(bf16f(x[1] & 0xffffu) * sc, bf16f(x[1] >> 16) * sc);
            out.z = pack2(bf16f(x[2] & 0xffffu) * sc, bf16f(x[2] >> 16) * sc);
            out.w = pack2(bf16f(x[3] & 0xffffu) * sc, bf16f(x[3] >> 16) * sc);
            *reinterpret_cast<uint4 *>(o.up + int64_t(b) * n_down + 8 * g) = out;
        }
    }
    if (lane == 0) {
        o.up_s[b] = pow2(-e);
    }
}

// TTTS2: hct_convert from the Q8_0 halves (hcp_weights::q8). A Q8_0 value of 8 at i (a multiple of 8) of a block row:
// d x q exactly; v[8] out
__device__ __forceinline__ void q8_vals(const uint8_t * codes, const uint8_t * scales, int64_t i, float (&v)[8]) {
    const uint2    q = __ldg(reinterpret_cast<const uint2 *>(codes + i));
    const float    d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short *>(scales) + i / 32)));
    const uint32_t w[2] = { q.x, q.y };
#pragma unroll
    for (int32_t k = 0; k < 8; ++k) {
        v[k] = d * static_cast<float>(static_cast<int8_t>((w[k / 4] >> (8 * (k % 4))) & 0xffu));
    }
}

template <bool VALUES = true>
__global__ void __launch_bounds__(256) hct_convert_q8_kernel(const hcp_weights w, const hct_weights o) {
    const int32_t row  = static_cast<int32_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (row >= cv_rows) {
        return;
    }
    if (row < cv_down_rows) {
        const int32_t s = row / J, j = row % J;
        float         v[10][8];
        float         m = 0.0f;
#pragma unroll
        for (int32_t t = 0; t < 10; ++t) {
            const int32_t   i = 8 * lane + 256 * t, h = i / F, ii = i - h * F;
            const uint8_t * b = reinterpret_cast<const uint8_t *>(w.down[h]);
            if (j < n_down) {
                const int64_t r = int64_t(s) * n_down + j;
                q8_vals(b + r * F, b + int64_t(4) * n_down * F + r * (F / 32) * 2, ii, v[t]);
            } else {
                const uint4    x     = __ldg(reinterpret_cast<const uint4 *>(
                    b + w_down_head_q8_bytes + ((int64_t(s) * n_inject + (j - n_down)) * F + ii) * 2));
                const uint32_t xs[4] = { x.x, x.y, x.z, x.w };
#pragma unroll
                for (int32_t k = 0; k < 4; ++k) {
                    v[t][2 * k]     = bf16f(xs[k] & 0xffffu);
                    v[t][2 * k + 1] = bf16f(xs[k] >> 16);
                }
            }
#pragma unroll
            for (int32_t k = 0; k < 8; ++k) {
                m = fmaxf(m, fabsf(v[t][k]));
            }
        }
        const int32_t e  = range_exp(warp_max(m));
        const float   sc = pow2(e);
#pragma unroll
        for (int32_t t = 0; t < 10 && VALUES; ++t) {
            uint4 out;
            out.x = pack2(v[t][0] * sc, v[t][1] * sc);
            out.y = pack2(v[t][2] * sc, v[t][3] * sc);
            out.z = pack2(v[t][4] * sc, v[t][5] * sc);
            out.w = pack2(v[t][6] * sc, v[t][7] * sc);
            *reinterpret_cast<uint4 *>(o.down + int64_t(row) * D + 8 * lane + 256 * t) = out;
        }
        if (lane == 0) {
            o.down_s[row] = pow2(-e);
        }
        return;
    }
    const int32_t   b = row - cv_down_rows, f = up_feature(b), s = up_stream(b);
    const int32_t   h = f / F, cc = f - h * F;
    const uint8_t * u = reinterpret_cast<const uint8_t *>(w.up[h]);
    const int64_t   r = int64_t(cc) * 4 + s;
    float           v[2][8];
    float           m = 0.0f;
#pragma unroll
    for (int32_t t = 0; t < 2; ++t) {
        const int32_t g = lane + 32 * t;
        if (g < n_down / 8) {
            q8_vals(u + r * n_down, u + int64_t(F) * 4 * n_down + r * (n_down / 32) * 2, 8 * g, v[t]);
        } else {
#pragma unroll
            for (int32_t k = 0; k < 8; ++k) {
                v[t][k] = 0.0f;
            }
        }
#pragma unroll
        for (int32_t k = 0; k < 8; ++k) {
            m = fmaxf(m, fabsf(v[t][k]));
        }
    }
    const int32_t e  = range_exp(warp_max(m));
    const float   sc = pow2(e);
#pragma unroll
    for (int32_t t = 0; t < 2 && VALUES; ++t) {
        const int32_t g = lane + 32 * t;
        if (g < n_down / 8) {
            uint4 out;
            out.x                                                          = pack2(v[t][0] * sc, v[t][1] * sc);
            out.y                                                          = pack2(v[t][2] * sc, v[t][3] * sc);
            out.z                                                          = pack2(v[t][4] * sc, v[t][5] * sc);
            out.w                                                          = pack2(v[t][6] * sc, v[t][7] * sc);
            *reinterpret_cast<uint4 *>(o.up + int64_t(b) * n_down + 8 * g) = out;
        }
    }
    if (lane == 0) {
        o.up_s[b] = pow2(-e);
    }
}

// ---- APF1 Part 4: the fused route (hct_mix_fused) --------------------------------------------------------------------
// The products' B operands formed in the tensor core's loader from the halves themselves, value for value as
// hct_convert forms them (BF16 widened or Q8_0 d x q, times the row's 2^e from the prepared scale, rounded to FP16),
// so every output is bitwise the converted route's; hct_convert's pass over the 9.8 MB of FP16 coefficients and its
// writes are gone. 2^e from the prepared 2^-e exactly (both normal powers of two: the exponent field negated)
__device__ __forceinline__ float inv_pow2(float s) {
    return __int_as_float((254 << 23) - __float_as_int(s));
}

// half h's pointer by selection, not a dynamic index (which puts the kernel parameter's array in local memory)
__device__ const __forceinline__ uint16_t * half_of(const uint16_t * const (&p)[2], int32_t h) {
    return h == 0 ? p[0] : p[1];
}

__device__ __forceinline__ void bf16x8(const uint4 & q, float (&v)[8]) {
    const uint32_t x[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
    for (int32_t k = 0; k < 4; ++k) {
        v[2 * k]     = bf16f(x[k] & 0xffffu);
        v[2 * k + 1] = bf16f(x[k] >> 16);
    }
}

__device__ __forceinline__ uint4 pack8(const float (&v)[8], float sc) {
    uint4 out;
    out.x = pack2(v[0] * sc, v[1] * sc);
    out.y = pack2(v[2] * sc, v[3] * sc);
    out.z = pack2(v[4] * sc, v[5] * sc);
    out.w = pack2(v[6] * sc, v[7] * sc);
    return out;
}

// down row (s, j) (j < 324), the 8 values at feature i (a multiple of 8) of the full width
template <bool Q8>
__device__ __forceinline__ void down_vals(const hcp_weights & w, int32_t s, int32_t j, int32_t i, float (&v)[8]) {
    const int32_t h = i / F, ii = i - h * F;
    if constexpr (!Q8) {
        bf16x8(__ldg(reinterpret_cast<const uint4 *>(half_of(w.down, h) + (int64_t(s) * J + j) * F + ii)), v);
    } else {
        const uint8_t * b = reinterpret_cast<const uint8_t *>(half_of(w.down, h));
        if (j < n_down) {
            const int64_t r = int64_t(s) * n_down + j;
            q8_vals(b + r * F, b + int64_t(4) * n_down * F + r * (F / 32) * 2, ii, v);
        } else {
            bf16x8(__ldg(reinterpret_cast<const uint4 *>(b + w_down_head_q8_bytes +
                                                         ((int64_t(s) * n_inject + (j - n_down)) * F + ii) * 2)),
                   v);
        }
    }
}

// up row b (hct_convert's order), the 8 values at j (a multiple of 8, < 320)
template <bool Q8> __device__ __forceinline__ void up_vals(const hcp_weights & w, int32_t b, int32_t j, float (&v)[8]) {
    const int32_t f = up_feature(b), s = up_stream(b), h = f / F, cc = f - h * F;
    const int64_t r = int64_t(cc) * 4 + s;
    if constexpr (!Q8) {
        bf16x8(__ldg(reinterpret_cast<const uint4 *>(half_of(w.up, h) + r * n_down + j)), v);
    } else {
        const uint8_t * u = reinterpret_cast<const uint8_t *>(half_of(w.up, h));
        q8_vals(u + r * n_down, u + int64_t(F) * 4 * n_down + r * (n_down / 32) * 2, j, v);
    }
}

// gemm_cta_b's loaders: the down of stream s (columns the 320 low rows), the up (columns the 10,240 rows)
template <bool Q8> struct down_loader {
    hcp_weights   w;
    const float * ds = nullptr;
    int32_t       s  = 0;

    __device__ __forceinline__ uint4 operator()(int32_t col, int32_t k) const {
        if (col >= n_down) {
            return make_uint4(0u, 0u, 0u, 0u);
        }
        float v[8];
        down_vals<Q8>(w, s, col, k, v);
        return pack8(v, inv_pow2(__ldg(ds + s * J + col)));
    }
};

template <bool Q8> struct up_loader {
    hcp_weights   w;
    const float * us = nullptr;

    __device__ __forceinline__ uint4 operator()(int32_t col, int32_t k) const {
        float v[8];
        up_vals<Q8>(w, col, k, v);
        return pack8(v, inv_pow2(__ldg(us + col)));
    }
};

// the scratch of hct_mix
struct hct_scr {
    uint16_t * x16;  // [rows][4][2560] y scaled by 2^e
    float *    x_s;  // [rows][4] 2^-e
    uint16_t * l16;  // [rows][320]
    float *    l_s;  // [rows]
};

int64_t al256(int64_t n) {
    return (n + 255) / 256 * 256;
}

hct_scr carve_scr(uint8_t * p, int32_t rows) {
    hct_scr o;
    int64_t off = 0;
    const auto take = [&](int64_t bytes) {
        uint8_t * r = p + off;
        off += al256(bytes);
        return r;
    };
    o.x16 = reinterpret_cast<uint16_t *>(take(int64_t(rows) * 4 * D * 2));
    o.x_s = reinterpret_cast<float *>(take(int64_t(rows) * 4 * 4));
    o.l16 = reinterpret_cast<uint16_t *>(take(int64_t(rows) * n_down * 2));
    o.l_s = reinterpret_cast<float *>(take(int64_t(rows) * 4));
    return o;
}

// a message header against gen (prompt_xchg.h's layout: U32 gen_lo, gen_hi first)
__device__ __forceinline__ bool msg_ok(const uint8_t * slot, uint64_t gen) {
    const volatile uint32_t * h = reinterpret_cast<const volatile uint32_t *>(slot);
    return h[0] == static_cast<uint32_t>(gen) && h[1] == static_cast<uint32_t>(gen >> 32);
}

// f32 of the four FP16 values at p (8 bytes)
__device__ __forceinline__ float4 f16x4(const uint8_t * p) {
    const uint2 u = __ldg(reinterpret_cast<const uint2 *>(p));
    return make_float4(__half2float(__ushort_as_half(static_cast<unsigned short>(u.x & 0xffffu))),
                       __half2float(__ushort_as_half(static_cast<unsigned short>(u.x >> 16))),
                       __half2float(__ushort_as_half(static_cast<unsigned short>(u.y & 0xffffu))),
                       __half2float(__ushort_as_half(static_cast<unsigned short>(u.y >> 16))));
}

// the FP16 values (widened) of injection row k of stream s at i (a multiple of 4): converted (MODE 0), or formed
// from the BF16 (1) or Q8_0 layout's BF16 (2) inject rows as hct_convert forms them
template <int32_t MODE>
__device__ __forceinline__ float4
inject4(const hcp_weights & hw, const hct_weights & w, int32_t s, int32_t k, int32_t i) {
    if constexpr (MODE == 0) {
        return f16x4(reinterpret_cast<const uint8_t *>(w.down + (int64_t(s) * J + n_down + k) * D + i));
    } else {
        const int32_t   h = i / F, ii = i - h * F;
        const uint8_t * src =
            MODE == 1 ?
                reinterpret_cast<const uint8_t *>(half_of(hw.down, h) + ((int64_t(s) * J + n_down + k) * F + ii)) :
                reinterpret_cast<const uint8_t *>(half_of(hw.down, h)) + w_down_head_q8_bytes +
                    ((int64_t(s) * n_inject + k) * F + ii) * 2;
        const uint2 u  = __ldg(reinterpret_cast<const uint2 *>(src));
        const float sc = inv_pow2(__ldg(w.down_s + s * J + n_down + k));
        const auto  r  = [&](uint32_t x) {
            return __half2float(__float2half_rn(bf16f(x) * sc));
        };
        return make_float4(r(u.x & 0xffffu), r(u.x >> 16), r(u.y & 0xffffu), r(u.y >> 16));
    }
}

// prescale: a block of 128 threads per (row, stream); thread t's float4s t + 128 u (u < 5). PB3: with cb.own, the
// exchange's combine first (pcombine's arithmetic on the same values), the streams written back and used from registers.
// PB3: the four injection rows of the down (J's rows 320 .. 323, a whole 64-column tile of the down for 4 columns) are
// summed here over the same FP16 operands: each FP16 product exact in F32, summed over the thread's values in order,
// the warp's xor tree, then the four warps as (0 + 1) + (2 + 3); scaled as the down's epilogue scales its columns
// APF1 Part 4: MODE 0 reads the converted injection rows; 1 (BF16 halves) and 2 (Q8_0 halves) form the same FP16
// values from the halves and the prepared scales (w.down null)
template <int32_t MODE = 0>
__global__ void __launch_bounds__(128) hct_split_kernel(const hcp_args    a,
                                                        const hct_scr     sc,
                                                        const hct_combine cb,
                                                        const hct_weights w) {
    __shared__ double vs[4];
    __shared__ float  ms[4];
    __shared__ float qs[4][n_inject];
    const int32_t     r = static_cast<int32_t>(blockIdx.x), s = static_cast<int32_t>(blockIdx.y);
    const int32_t     tid = static_cast<int32_t>(threadIdx.x), lane = tid & 31, wp = tid >> 5;
    const float *     xr   = a.x + (int64_t(r) * 4 + s) * D;
    const bool        comb = cb.own != nullptr;
    if (comb && !(msg_ok(cb.own, cb.gen) && msg_ok(cb.peer, cb.gen))) {
        if (tid == 0) {
            atomicOr(a.err, fn::xchg::err_pmsg);
        }
        return;
    }
    const float iv = comb ? __ldg(cb.inject + int64_t(r) * 4 + s) : 0.0f;
    float4      y[5];
    double      v = 0.0;
    float       m = 0.0f;
#pragma unroll
    for (int32_t u = 0; u < 5; ++u) {
        const int32_t i = 4 * (tid + 128 * u), h = i / F;
        float4        x;
        if (comb) {
            float4 *      sp = reinterpret_cast<float4 *>(cb.streams + (int64_t(r) * 4 + s) * D + i);
            const int64_t at = fn::xchg::pmsg_payload_at + (int64_t(r) * D + i) * 2;
            const float4  o  = f16x4(cb.own + at), p = f16x4(cb.peer + at);
            const float4  c  = cb.card == 0 ? make_float4(o.x + p.x, o.y + p.y, o.z + p.z, o.w + p.w) :
                                              make_float4(p.x + o.x, p.y + o.y, p.z + o.z, p.w + o.w);
            x                = *sp;
            x                = make_float4(fmaf(iv, c.x, x.x), fmaf(iv, c.y, x.y), fmaf(iv, c.z, x.z), fmaf(iv, c.w, x.w));
            *sp              = x;
        } else {
            x = __ldg(reinterpret_cast<const float4 *>(xr + i));
        }
        const float4  g  = __ldg(reinterpret_cast<const float4 *>(a.w.gamma[h] + s * F + (i - h * F)));
        y[u]             = make_float4(g.x * x.x, g.y * x.y, g.z * x.z, g.w * x.w);
        v                = fma(double(x.x), double(x.x), v);
        v                = fma(double(x.y), double(x.y), v);
        v                = fma(double(x.z), double(x.z), v);
        v                = fma(double(x.w), double(x.w), v);
        m = fmaxf(m, fmaxf(fmaxf(fabsf(y[u].x), fabsf(y[u].y)), fmaxf(fabsf(y[u].z), fabsf(y[u].w))));
    }
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    m = warp_max(m);
    if (lane == 0) {
        vs[wp] = v;
        ms[wp] = m;
    }
    __syncthreads();
    const float   mx = fmaxf(fmaxf(ms[0], ms[1]), fmaxf(ms[2], ms[3]));
    const int32_t e  = range_exp(mx);
    const float   f  = pow2(e);
    if (tid == 0) {
        a.mom[int64_t(r) * hcp_mom + s] = static_cast<float>((vs[0] + vs[1]) + (vs[2] + vs[3]));
        sc.x_s[int64_t(r) * 4 + s]      = pow2(-e);
    }
    const int64_t o           = (int64_t(r) * 4 + s) * D;
    float q[n_inject] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int32_t u = 0; u < 5; ++u) {
        const int32_t i = 4 * (tid + 128 * u);
        const uint2 h = make_uint2(pack2(y[u].x * f, y[u].y * f), pack2(y[u].z * f, y[u].w * f));
        *reinterpret_cast<uint2 *>(sc.x16 + o + i) = h;
        const float4 yh = make_float4(__half2float(__ushort_as_half(static_cast<unsigned short>(h.x & 0xffffu))),
                                      __half2float(__ushort_as_half(static_cast<unsigned short>(h.x >> 16))),
                                      __half2float(__ushort_as_half(static_cast<unsigned short>(h.y & 0xffffu))),
                                      __half2float(__ushort_as_half(static_cast<unsigned short>(h.y >> 16))));
#pragma unroll
        for (int32_t k = 0; k < n_inject; ++k) {
            const float4 wk = inject4<MODE>(a.w, w, s, k, i);
            q[k] = fmaf(yh.w, wk.w, fmaf(yh.z, wk.z, fmaf(yh.y, wk.y, fmaf(yh.x, wk.x, q[k]))));
        }
    }
#pragma unroll
    for (int32_t k = 0; k < n_inject; ++k) {
#pragma unroll
        for (int32_t d = 16; d > 0; d >>= 1) {
            q[k] += __shfl_xor_sync(0xffffffffu, q[k], d);
        }
        if (lane == 0) {
            qs[wp][k] = q[k];
        }
    }
    __syncthreads();
    if (tid < n_inject) {
        const float t = (qs[0][tid] + qs[1][tid]) + (qs[2][tid] + qs[3][tid]);
        a.mom[int64_t(r) * hcp_mom + 4 + 4 * n_down + n_inject * s + tid] =
            t * pow2(-e) * __ldg(w.down_s + s * J + n_down + tid);
    }
}

// mid (tensor path): hcp_mid_kernel's arithmetic, then low (j < 320) scaled by its row's power of two and rounded to
// FP16; thread t's values j = t + 128 i
__global__ void __launch_bounds__(128) hct_mid_kernel(const hcp_args a, const hct_scr sc) {
    __shared__ float ms[4];
    const int32_t    r   = static_cast<int32_t>(blockIdx.x);
    const int32_t    t   = static_cast<int32_t>(threadIdx.x);
    const float *    mo  = a.mom + int64_t(r) * hcp_mom;
    float            ri[4];
#pragma unroll
    for (int32_t s = 0; s < 4; ++s) {
        ri[s] = 1.0f / sqrtf(mo[s] / float(D) + fn::rms_eps);
    }
    if (t < 4) {
        a.rinv[int64_t(r) * 4 + t] = ri[t];
    }
    float lv[3] = { 0.0f, 0.0f, 0.0f };
    float m     = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < 3; ++i) {
        const int32_t j = t + 128 * i;
        if (j < J) {
            const int32_t base    = j < n_down ? 4 + j : 4 + 4 * n_down + (j - n_down);
            const int32_t sstride = j < n_down ? n_down : 4;
            float         z       = mo[base] * ri[0];
#pragma unroll
            for (int32_t s = 1; s < 4; ++s) {
                z = fmaf(mo[base + s * sstride], ri[s], z);
            }
            z = 0.25f * z;
            if (j < n_down) {
                lv[i]                          = z * sigm(z);
                a.low[int64_t(r) * n_down + j] = lv[i];
                m                              = fmaxf(m, fabsf(lv[i]));
            } else {
                a.inject[int64_t(r) * 4 + (j - n_down)] = 2.0f * sigm(z);
            }
        }
    }
    m = warp_max(m);
    if ((t & 31) == 0) {
        ms[t >> 5] = m;
    }
    __syncthreads();
    const int32_t e = range_exp(fmaxf(fmaxf(ms[0], ms[1]), fmaxf(ms[2], ms[3])));
    const float   f = pow2(e);
    if (t == 0) {
        sc.l_s[r] = pow2(-e);
    }
#pragma unroll
    for (int32_t i = 0; i < 3; ++i) {
        const int32_t j = t + 128 * i;
        if (j < n_down) {
            sc.l16[int64_t(r) * n_down + j] = __half_as_ushort(__float2half_rn(lv[i] * f));
        }
    }
}

// the down's epilogue (the ladder's; hct_down_kernel keeps its own lambda, the same arithmetic, so its compiled code
// is unchanged): the moments of stream s's low rows, scaled by the row's 2^-e and the coefficient row's
struct down_epi {
    float *       mom;
    const float * x_s;
    const float * down_s;
    int32_t       s;

    __device__ __forceinline__ void operator()(int32_t row, int32_t c0, const float (&v)[4]) const {
        const float   xs    = x_s[int64_t(row) * 4 + s];
        float *       mo    = mom + int64_t(row) * hcp_mom + 4 + n_down * s;
        const int32_t jj[4] = { c0, c0 + 1, c0 + 4, c0 + 5 };
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            mo[jj[i]] = v[i] * xs * __ldg(down_s + s * J + jj[i]);
        }
    }
};

// the up's epilogue: a lane's columns c0, c0 + 1, c0 + 4, c0 + 5 are one feature's streams 0..3, mixed into the
// feature's output
struct up_epi {
    const hcp_args &    a;
    const hct_weights & w;
    const hct_scr &     sc;

    __device__ __forceinline__ void operator()(int32_t row, int32_t c0, const float (&v)[4]) const {
        const int32_t f      = up_feature(c0);
        const float   ls     = sc.l_s[row];
        const int32_t col[4] = { c0, c0 + 1, c0 + 4, c0 + 5 };  // streams 0..3
        float         mixed  = 0.0f;
    #pragma unroll
        for (int32_t s = 0; s < 4; ++s) {
            const float g  = v[s] * ls * __ldg(w.up_s + col[s]);
            const float ri = a.rinv[int64_t(row) * 4 + s];
            const float gx = gamma_of<hcp_form::replicated>(a.w, s, f) * a.x[(int64_t(row) * 4 + s) * D + f];
            mixed          = fmaf(gx * ri, sigm(g), mixed);
        }
        mixed = 0.25f * mixed;
        if (a.mixed != nullptr) {
            a.mixed[int64_t(row) * a.ld_mixed + f] = mixed;
        }
        if (a.mixed16 != nullptr) {
            if (!(fabsf(mixed) <= 65504.0f) && a.err != nullptr) {
                atomicOr(a.err, 1u);
            }
            a.mixed16[int64_t(row) * a.ld_mixed16 + f] = __half_as_ushort(__float2half_rn(mixed));
        }
    }
};

// down: grid (j-tiles of 64 over the 320 low rows, row tiles of 128, streams); the injection rows in hct_split.
// PB3: a CTA 128 rows x 64 columns, a warp 64 x 32 (the 32 x 32 warp tile read 0.25 shared bytes a MAC, the shared
// memory's whole bandwidth at the tensor rate; 64 x 32 reads 0.1875)
// APF1 Part 4: MODE 0 the converted coefficients (w.down), 1 and 2 the fused loaders (BF16, Q8_0 halves); MI 2 the
// small tile (64 rows a CTA) for chunks of at most 64 rows. The same products in the same K order: bitwise
template <int32_t MODE = 0, int32_t MI = 4>
__global__ void __launch_bounds__(fn::tc::shape<64, MI>::threads, MI == 4 ? 3 : 4) hct_down_kernel(const hcp_args    a,
                                                                                                   const hct_weights w,
                                                                                                   const hct_scr sc) {
    extern __shared__ uint4 dn_sm[];
    const int32_t           s = static_cast<int32_t>(blockIdx.z);
    fn::tc::problem         p;
    p.a.a0         = sc.x16 + int64_t(s) * D;
    p.a.ld0        = 4 * D;
    p.a.k_split    = D;
    if constexpr (MODE == 0) {
        p.b = w.down + int64_t(s) * J * D;
    }
    p.ldb          = D;
    p.rows         = a.rows;
    p.n = n_down;
    p.k            = D;
    const auto epi = [&](int32_t row, int32_t c0, const float (&v)[4]) {
        const float xs   = sc.x_s[int64_t(row) * 4 + s];
        float * mo = a.mom + int64_t(row) * hcp_mom + 4 + n_down * s;
        const int32_t jj[4] = { c0, c0 + 1, c0 + 4, c0 + 5 };
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            mo[jj[i]] = v[i] * xs * __ldg(w.down_s + s * J + jj[i]);
        }
    };
    static_assert(n_down % 64 == 0, "the low rows tile exactly");
    if constexpr (MODE == 0) {
        fn::tc::gemm_cta<64, MI>(p, static_cast<int32_t>(blockIdx.y) * fn::tc::shape<64, MI>::rows,
                                 static_cast<int32_t>(blockIdx.x) * 64, reinterpret_cast<uint8_t *>(dn_sm), epi);
    } else {
        down_loader<MODE == 2> bl;
        bl.w  = a.w;
        bl.ds = w.down_s;
        bl.s  = s;
        fn::tc::gemm_cta_b<64, MI>(p, static_cast<int32_t>(blockIdx.y) * fn::tc::shape<64, MI>::rows,
                                   static_cast<int32_t>(blockIdx.x) * 64, reinterpret_cast<uint8_t *>(dn_sm), epi, bl);
    }
}

// up: grid (row tiles of 128, 10,240 / 128 column tiles). PB3: a CTA 128 x 128 of four 64 x 64 warps (as pt_kernel),
// and the row tiles adjacent in launch order, so a wave of CTAs shares a few 80 KB column slices of the 6.5 MB of
// weights (row-tile-major order streamed all of them from memory for every two row tiles)
// APF1 Part 4: MODE and MI as hct_down_kernel's
template <int32_t MODE = 0, int32_t MI = 4>
__global__ void __launch_bounds__(fn::tc::shape<128, MI, 4>::threads, MI == 4 ? 2 : 3)
hct_up_kernel(const hcp_args a, const hct_weights w, const hct_scr sc) {
    extern __shared__ uint4 up_sm[];
    fn::tc::problem         p;
    p.a.a0         = sc.l16;
    p.a.ld0        = n_down;
    p.a.k_split    = n_down;
    p.b            = w.up;
    p.ldb          = n_down;
    p.rows         = a.rows;
    p.n            = NB;
    p.k            = n_down;
    const up_epi epi{ a, w, sc };
    const int32_t row0 = static_cast<int32_t>(blockIdx.x) * fn::tc::shape<128, MI, 4>::rows;
    const int32_t col0 = static_cast<int32_t>(blockIdx.y) * 128;
    if constexpr (MODE == 0) {
        fn::tc::gemm_cta<128, MI, 4>(p, row0, col0, reinterpret_cast<uint8_t *>(up_sm), epi);
    } else {
        up_loader<MODE == 2> bl;
        bl.w  = a.w;
        bl.us = w.up_s;
        fn::tc::gemm_cta_b<128, MI, 4>(p, row0, col0, reinterpret_cast<uint8_t *>(up_sm), epi, bl);
    }
}

// APF2: the ladder's tiles for the fused route's two products (fn/kernels/tc/tc_ladder.cuh), MODE 1 (BF16 halves)
// or 2 (Q8_0 halves), the same loaders, operands and epilogues as hct_down_kernel / hct_up_kernel. The down's four
// streams are one launch: grid y = stream x the column tiles over its 320 low rows (the last tile's columns past 320
// read zero and are not written); K 2,560. The up: grid y the column tiles of its 10,240; K 320. Split K over grid z.
template <int32_t MODE, int32_t BM, int32_t BN>
__global__ void __launch_bounds__(fn::tc::ltile<BM, BN>::threads) hcl_down_kernel(const hcp_args          a,
                                                                                  const hct_weights       w,
                                                                                  const hct_scr           sc,
                                                                                  const fn::tc::lsplit    ws,
                                                                                  const int32_t           splits) {
    static_assert(MODE == 1 || MODE == 2, "the fused route's loaders");
    extern __shared__ uint4 hl_sm[];
    constexpr int32_t       ct = (n_down + BN - 1) / BN;
    const int32_t           s  = static_cast<int32_t>(blockIdx.y) / ct;
    fn::tc::a_plain         al;
    al.a.a0      = sc.x16 + int64_t(s) * D;
    al.a.ld0     = 4 * D;
    al.a.k_split = D;
    al.rows      = a.rows;
    fn::tc::b_chunks<BN, fn::tc::ltile<BM, BN>::threads, down_loader<MODE == 2>> bl;
    bl.ch.w              = a.w;
    bl.ch.ds             = w.down_s;
    bl.ch.s              = s;
    const down_epi  e{ a.mom, sc.x_s, w.down_s, s };
    const auto      epi = [&](int32_t row, int32_t c0, const float (&v)[4]) {
        if (c0 < n_down) {  // n_down a multiple of 8: a lane's four columns are in or out together
            e(row, c0, v);
        }
    };
    fn::tc::lgemm<BM, BN>(a.rows, D, splits, ws, reinterpret_cast<uint8_t *>(hl_sm), al, bl, epi, ct);
}

template <int32_t MODE, int32_t BM, int32_t BN>
__global__ void __launch_bounds__(fn::tc::ltile<BM, BN>::threads) hcl_up_kernel(const hcp_args       a,
                                                                                const hct_weights    w,
                                                                                const hct_scr        sc,
                                                                                const fn::tc::lsplit ws,
                                                                                const int32_t        splits) {
    static_assert(MODE == 1 || MODE == 2, "the fused route's loaders");
    static_assert(NB % BN == 0, "the up's columns tile exactly");
    extern __shared__ uint4 hl_sm[];
    fn::tc::a_plain         al;
    al.a.a0      = sc.l16;
    al.a.ld0     = n_down;
    al.a.k_split = n_down;
    al.rows      = a.rows;
    fn::tc::b_chunks<BN, fn::tc::ltile<BM, BN>::threads, up_loader<MODE == 2>> bl;
    bl.ch.w            = a.w;
    bl.ch.us           = w.up_s;
    const up_epi epi{ a, w, sc };
    fn::tc::lgemm<BM, BN>(a.rows, n_down, splits, ws, reinterpret_cast<uint8_t *>(hl_sm), al, bl, epi);
}

bool al16(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

bool weights_ok(const hcp_weights & w) {
    if (w.form == hcp_form::replicated) {
        return al16(w.down[0]) && al16(w.down[1]) && al16(w.up[0]) && al16(w.up[1]) && w.gamma[0] != nullptr &&
               w.gamma[1] != nullptr;
    }
    return al16(w.q8_down) && al16(w.q8_up) && w.gamma_own != nullptr;
}

bool common_ok(const hcp_args & a, void * stream) {
    if (a.rows < 1 || a.rows > hcp_max_rows || !weights_ok(a.w) || !al16(a.x) || a.mom == nullptr ||
        stream == nullptr) {
        fprintf(stderr, "hcp: rejected (rows %d, or a null or misaligned pointer)\n", a.rows);
        return false;
    }
    return true;
}

}  // namespace

bool hcp_down(const hcp_args & a, void * stream) {
    if (!common_ok(a, stream)) {
        return false;
    }
    const dim3         grid(down_jt, 4, (a.rows + tile - 1) / tile);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (a.w.form == hcp_form::replicated) {
        hcp_down_kernel<hcp_form::replicated><<<grid, 256, 0, s>>>(a);
    } else {
        hcp_down_kernel<hcp_form::split><<<grid, 256, 0, s>>>(a);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool hcp_mid(const hcp_args & a, void * stream) {
    if (!common_ok(a, stream) || a.low == nullptr || a.rinv == nullptr || a.inject == nullptr ||
        (a.w.form == hcp_form::split && a.mom_peer == nullptr) || (a.card != 0 && a.card != 1)) {
        fprintf(stderr, "hcp_mid: rejected\n");
        return false;
    }
    hcp_mid_kernel<<<a.rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool hcp_up(const hcp_args & a, void * stream) {
    if (!common_ok(a, stream) || !al16(a.low) || a.rinv == nullptr || a.mixed == nullptr) {
        fprintf(stderr, "hcp_up: rejected\n");
        return false;
    }
    const int32_t      nc = a.w.form == hcp_form::replicated ? D : F;
    const dim3         grid(nc / up_c, (a.rows + tile - 1) / tile);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (a.w.form == hcp_form::replicated) {
        hcp_up_kernel<hcp_form::replicated><<<grid, 256, 0, s>>>(a);
    } else {
        hcp_up_kernel<hcp_form::split><<<grid, 256, 0, s>>>(a);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

int64_t hct_weight_bytes() {
    return al256(hct_down_elems * 2) + al256(int64_t(4) * J * 4) + al256(hct_up_elems * 2) + al256(int64_t(NB) * 4);
}

hct_weights hct_carve(uint8_t * region) {
    hct_weights w;
    int64_t     off = 0;
    w.down          = reinterpret_cast<uint16_t *>(region + off);
    off += al256(hct_down_elems * 2);
    w.down_s = reinterpret_cast<float *>(region + off);
    off += al256(int64_t(4) * J * 4);
    w.up = reinterpret_cast<uint16_t *>(region + off);
    off += al256(hct_up_elems * 2);
    w.up_s = reinterpret_cast<float *>(region + off);
    return w;
}

int64_t hct_scratch_bytes(int32_t rows) {
    return al256(int64_t(rows) * 4 * D * 2) + al256(int64_t(rows) * 4 * 4) + al256(int64_t(rows) * n_down * 2) +
           al256(int64_t(rows) * 4);
}

bool hct_convert(const hcp_weights & a, const hct_weights & w, void * stream) {
    if (a.form != hcp_form::replicated || !weights_ok(a) || !al16(w.down) || !al16(w.up) || w.down_s == nullptr ||
        w.up_s == nullptr || stream == nullptr) {
        fprintf(stderr, "hct_convert: rejected\n");
        return false;
    }
    if (a.q8) {
        hct_convert_q8_kernel<<<(cv_rows * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(a, w);
        return cudaPeekAtLastError() == cudaSuccess;
    }
    hct_convert_kernel<<<(cv_rows * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(a, w);
    return cudaPeekAtLastError() == cudaSuccess;
}

int64_t hct_scale_bytes() {
    return al256(int64_t(4) * J * 4) + al256(int64_t(NB) * 4);
}

hct_weights hct_carve_scales(uint8_t * region) {
    hct_weights w;
    w.down_s = reinterpret_cast<float *>(region);
    w.up_s   = reinterpret_cast<float *>(region + al256(int64_t(4) * J * 4));
    return w;
}

bool hct_scales(const hcp_weights & a, const hct_weights & w, void * stream) {
    if (a.form != hcp_form::replicated || !weights_ok(a) || w.down_s == nullptr || w.up_s == nullptr ||
        stream == nullptr) {
        fprintf(stderr, "hct_scales: rejected\n");
        return false;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    hct_weights        o;
    o.down_s = w.down_s;
    o.up_s   = w.up_s;
    if (a.q8) {
        hct_convert_q8_kernel<false><<<(cv_rows * 32 + 255) / 256, 256, 0, s>>>(a, o);
    } else {
        hct_convert_kernel<false><<<(cv_rows * 32 + 255) / 256, 256, 0, s>>>(a, o);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

namespace {

template <int32_t MODE, int32_t MI>
void launch_down(const hcp_args & a, const hct_weights & w, const hct_scr & sc, cudaStream_t s) {
    using dn            = fn::tc::shape<64, MI>;
    const uint32_t rows = static_cast<uint32_t>(a.rows);
    hct_down_kernel<MODE, MI>
        <<<dim3(n_down / 64, (rows + dn::rows - 1) / dn::rows, 4), dn::threads, dn::smem, s>>>(a, w, sc);
}

template <int32_t MODE, int32_t MI>
void launch_up(const hcp_args & a, const hct_weights & w, const hct_scr & sc, cudaStream_t s) {
    using up            = fn::tc::shape<128, MI, 4>;
    const uint32_t rows = static_cast<uint32_t>(a.rows);
    hct_up_kernel<MODE, MI><<<dim3((rows + up::rows - 1) / up::rows, NB / 128), up::threads, up::smem, s>>>(a, w, sc);
}

template <int32_t MODE, int32_t MI>
void launch_mix(const hcp_args & a, const hct_combine & cb, const hct_weights & w, const hct_scr & sc, cudaStream_t s) {
    const uint32_t rows = static_cast<uint32_t>(a.rows);
    hct_split_kernel<MODE><<<dim3(rows, 4), 128, 0, s>>>(a, sc, cb, w);
    launch_down<MODE, MI>(a, w, sc, s);
    hct_mid_kernel<<<rows, 128, 0, s>>>(a, sc);
    launch_up<MODE, MI>(a, w, sc, s);
}

// APF2: the ladder's products. hcl_with(t, product, run): run(kernel, ltile, column tiles of the launch's grid y)
template <int32_t MODE, typename Fn> bool hcl_with(int32_t t, int32_t product, Fn && run) {
    return fn::tc::lad_with_tile(t, [&](auto sh) {
        using T = decltype(sh);
        if (product == 0) {
            return run(hcl_down_kernel<MODE, T::bm, T::bn>, sh, 4 * ((n_down + T::bn - 1) / T::bn));
        }
        return run(hcl_up_kernel<MODE, T::bm, T::bn>, sh, NB / T::bn);
    });
}

int32_t hcl_k(int32_t product) {
    return product == 0 ? D : n_down;
}

// whether product's tile t and splits launch at a's rows in ws
bool hcl_fits(int32_t product, int32_t rows, int32_t t, int32_t splits, const fn::tc::lad_ws & ws) {
    if (t < 0 || t >= fn::tc::lad_n_tiles || splits < 1 || splits > fn::tc::lad_max_splits ||
        splits > hcl_k(product) / fn::tc::bk) {
        return false;
    }
    const fn::tc::lad_tile & tl = fn::tc::lad_tiles[t];
    return fn::tc::lad_ws_fits(ws, hcl_tile_count(product, rows, t), splits, tl.bm, tl.bn);
}

template <int32_t MODE>
bool hcl_launch(const hcp_args &       a,
                const hct_weights &    w,
                const hct_scr &        sc,
                int32_t                product,
                int32_t                t,
                int32_t                splits,
                const fn::tc::lad_ws & ws,
                cudaStream_t           s) {
    return hcl_with<MODE>(t, product, [&](auto kern, auto sh, int32_t ny) {
        using T = decltype(sh);
        const dim3 grid((a.rows + T::bm - 1) / T::bm, ny, splits);
        kern<<<grid, T::threads, T::smem, s>>>(a, w, sc, fn::tc::lsplit{ ws.part, ws.count }, splits);
        return true;
    });
}

template <int32_t MODE>
void launch_mix_ladder(const hcp_args &       a,
                       const hct_combine &    cb,
                       const hct_weights &    w,
                       const hct_scr &        sc,
                       const hct_route &      r,
                       const fn::tc::lad_ws & ws,
                       cudaStream_t           s) {
    const uint32_t rows = static_cast<uint32_t>(a.rows);
    const bool     tiny = a.rows <= hct_small_rows;
    hct_split_kernel<MODE><<<dim3(rows, 4), 128, 0, s>>>(a, sc, cb, w);
    if (r.down_tile < 0) {
        tiny ? launch_down<MODE, 2>(a, w, sc, s) : launch_down<MODE, 4>(a, w, sc, s);
    } else {
        hcl_launch<MODE>(a, w, sc, 0, r.down_tile, r.down_splits, ws, s);
    }
    hct_mid_kernel<<<rows, 128, 0, s>>>(a, sc);
    if (r.up_tile < 0) {
        tiny ? launch_up<MODE, 2>(a, w, sc, s) : launch_up<MODE, 4>(a, w, sc, s);
    } else {
        hcl_launch<MODE>(a, w, sc, 1, r.up_tile, r.up_splits, ws, s);
    }
}

bool mix_ok(const hcp_args & a, const hct_combine & cb, uint8_t * scratch, void * stream) {
    return common_ok(a, stream) && a.w.form == hcp_form::replicated && scratch != nullptr &&
           (reinterpret_cast<uintptr_t>(scratch) & 255u) == 0 && al16(a.low) && a.rinv != nullptr &&
           a.inject != nullptr && (a.mixed != nullptr || a.mixed16 != nullptr) &&
           (cb.own == nullptr || ((cb.card == 0 || cb.card == 1) && al16(cb.own) && al16(cb.peer) &&
                                  cb.inject != nullptr && cb.streams == a.x && a.err != nullptr));
}

}  // namespace

bool hct_mix(const hcp_args & a, const hct_combine & cb, const hct_weights & w, uint8_t * scratch, void * stream) {
    if (!mix_ok(a, cb, scratch, stream) || !al16(w.down) || !al16(w.up)) {
        fprintf(stderr, "hct_mix: rejected\n");
        return false;
    }
    launch_mix<0, 4>(a, cb, w, carve_scr(scratch, a.rows), static_cast<cudaStream_t>(stream));
    return cudaPeekAtLastError() == cudaSuccess;
}

bool hct_mix_fused(const hcp_args &    a,
                   const hct_combine & cb,
                   const hct_weights & scales,
                   uint8_t *           scratch,
                   void *              stream) {
    if (!mix_ok(a, cb, scratch, stream) || scales.down_s == nullptr || scales.up_s == nullptr) {
        fprintf(stderr, "hct_mix_fused: rejected\n");
        return false;
    }
    hct_weights w;
    w.down_s                = scales.down_s;
    w.up_s                  = scales.up_s;
    const hct_scr      sc   = carve_scr(scratch, a.rows);
    const cudaStream_t s    = static_cast<cudaStream_t>(stream);
    const bool         tiny = a.rows <= hct_small_rows;
    if (a.w.q8) {
        tiny ? launch_mix<2, 2>(a, cb, w, sc, s) : launch_mix<2, 4>(a, cb, w, sc, s);
    } else {
        tiny ? launch_mix<1, 2>(a, cb, w, sc, s) : launch_mix<1, 4>(a, cb, w, sc, s);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

int32_t hcl_tile_count(int32_t product, int32_t rows, int32_t t) {
    if (t < 0 || t >= fn::tc::lad_n_tiles || rows < 1 || (product != 0 && product != 1)) {
        return 0;
    }
    const fn::tc::lad_tile & tl = fn::tc::lad_tiles[t];
    const int32_t            ny = product == 0 ? 4 * ((n_down + tl.bn - 1) / tl.bn) : NB / tl.bn;
    return (rows + tl.bm - 1) / tl.bm * ny;
}

bool hct_mix_ladder(const hcp_args &       a,
                    const hct_combine &    cb,
                    const hct_weights &    scales,
                    uint8_t *              scratch,
                    const hct_route &      r,
                    const fn::tc::lad_ws & ws,
                    void *                 stream) {
    const bool dn_ok = r.down_tile < 0 ? r.down_splits == 1 : hcl_fits(0, a.rows, r.down_tile, r.down_splits, ws);
    const bool up_ok = r.up_tile < 0 ? r.up_splits == 1 : hcl_fits(1, a.rows, r.up_tile, r.up_splits, ws);
    if (!mix_ok(a, cb, scratch, stream) || scales.down_s == nullptr || scales.up_s == nullptr || !dn_ok || !up_ok) {
        fprintf(stderr, "hct_mix_ladder: rejected (hct_mix_fused's checks, a tile, a split count or the workspace)\n");
        return false;
    }
    hct_weights w;
    w.down_s                = scales.down_s;
    w.up_s                  = scales.up_s;
    const hct_scr      sc   = carve_scr(scratch, a.rows);
    const cudaStream_t s    = static_cast<cudaStream_t>(stream);
    if (a.w.q8) {
        launch_mix_ladder<2>(a, cb, w, sc, r, ws, s);
    } else {
        launch_mix_ladder<1>(a, cb, w, sc, r, ws, s);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

namespace {

template <int32_t MODE> bool hcl_attr_mode() {
    for (int32_t t = 0; t < fn::tc::lad_n_tiles; ++t) {
        for (int32_t product = 0; product < 2; ++product) {
            const bool ok = hcl_with<MODE>(t, product, [](auto kern, auto sh, int32_t) {
                return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            decltype(sh)::smem) == cudaSuccess;
            });
            if (!ok) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool hcl_attributes() {
    return hcl_attr_mode<1>() && hcl_attr_mode<2>();
}

bool hcl_resources(bool q8, int32_t product, int32_t t, fn::tc::lad_resources & r) {
    const auto get = [&](auto kern, auto sh, int32_t) {
        using T               = decltype(sh);
        cudaFuncAttributes at = {};
        int32_t            n  = 0;
        if (cudaFuncGetAttributes(&at, kern) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, T::threads, T::smem) != cudaSuccess) {
            return false;
        }
        r.regs         = at.numRegs;
        r.local_bytes  = static_cast<int32_t>(at.localSizeBytes);
        r.shared_bytes = static_cast<int32_t>(at.sharedSizeBytes) + T::smem;
        r.ctas_per_sm  = n;
        return true;
    };
    if (product != 0 && product != 1) {
        return false;
    }
    return q8 ? hcl_with<2>(t, product, get) : hcl_with<1>(t, product, get);
}

bool hcp_kernel_resources(int32_t which, hcp_resources & r) {
    cudaFuncAttributes at = {};
    cudaError_t        e  = cudaErrorInvalidValue;
    switch (which) {
        case 0:
            e = cudaFuncGetAttributes(&at, hcp_down_kernel<hcp_form::replicated>);
            break;
        case 1:
            e = cudaFuncGetAttributes(&at, hcp_down_kernel<hcp_form::split>);
            break;
        case 2:
            e = cudaFuncGetAttributes(&at, hcp_mid_kernel);
            break;
        case 3:
            e = cudaFuncGetAttributes(&at, hcp_up_kernel<hcp_form::replicated>);
            break;
        case 4:
            e = cudaFuncGetAttributes(&at, hcp_up_kernel<hcp_form::split>);
            break;
        case 5:
            e = cudaFuncGetAttributes(&at, hct_split_kernel<0>);
            break;
        case 6:
            e = cudaFuncGetAttributes(&at, hct_down_kernel<0, 4>);
            break;
        case 7:
            e = cudaFuncGetAttributes(&at, hct_mid_kernel);
            break;
        case 8:
            e = cudaFuncGetAttributes(&at, hct_up_kernel<0, 4>);
            break;
        case 9:
            e = cudaFuncGetAttributes(&at, hct_convert_kernel<true>);
            break;
        case 10:
            e = cudaFuncGetAttributes(&at, hct_split_kernel<1>);
            break;
        case 11:
            e = cudaFuncGetAttributes(&at, hct_split_kernel<2>);
            break;
        case 12:
            e = cudaFuncGetAttributes(&at, hct_down_kernel<1, 4>);
            break;
        case 13:
            e = cudaFuncGetAttributes(&at, hct_down_kernel<2, 4>);
            break;
        case 14:
            e = cudaFuncGetAttributes(&at, hct_down_kernel<2, 2>);
            break;
        case 15:
            e = cudaFuncGetAttributes(&at, hct_up_kernel<1, 4>);
            break;
        case 16:
            e = cudaFuncGetAttributes(&at, hct_up_kernel<2, 4>);
            break;
        case 17:
            e = cudaFuncGetAttributes(&at, hct_up_kernel<2, 2>);
            break;
        default:
            break;
    }
    if (e != cudaSuccess) {
        return false;
    }
    r.regs         = at.numRegs;
    r.local_bytes  = static_cast<int32_t>(at.localSizeBytes);
    r.shared_bytes = static_cast<int32_t>(at.sharedSizeBytes);
    return true;
}

}  // namespace fn::hc
