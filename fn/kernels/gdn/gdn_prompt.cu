#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/gdn/gdn_prompt.h"
#include "fn/kernels/round/version.h"
#include "fn/kernels/tc/tc_gemm.cuh"
#include "fn/model/shapes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace fn::gdn {
namespace {

constexpr int32_t C       = prompt_block;
constexpr int32_t threads = 256;
constexpr int32_t ks      = 132;            // a K, Q or V row's shared stride (16-byte aligned, 4 banks apart)
constexpr int32_t db      = 16;             // the solve's diagonal block
constexpr int32_t nbd     = C / db;
constexpr int32_t bs      = db * (db + 1);  // a stored 16 x 16 block, rows padded to 17
// prep's shared memory: K and Q (then V) [64][132], A and T as the 10 lower 16 x 16 blocks each, the off-diagonal
// stage's products [3][16][17], G, beta and beta exp(G) [64] each
constexpr int32_t pm_k = 0, pm_q = pm_k + C * ks, pm_a = pm_q + C * ks, pm_t = pm_a + 10 * bs, pm_x = pm_t + 10 * bs,
                  pm_g = pm_x + 3 * bs, pm_b = pm_g + C, pm_sb = pm_b + C, pm_floats = pm_sb + C;
constexpr int32_t prep_smem = pm_floats * 4;
static_assert(prep_smem <= 96 * 1024, "prep's shared memory fits one SM's opt-in 96 KiB");
// the WY item's layout (prompt_wy_item_floats)
constexpr int32_t wy_w = 0, wy_u = wy_w + C * head_dim, wy_p = wy_u + C * head_dim, wy_eg = wy_p + C * C,
                  wy_wd = wy_eg + C;
static_assert(wy_wd + C == prompt_wy_item_floats, "the WY item");
// PB3: W and P as the scan's FP16 operands: each row scaled by the power of two that brings its largest magnitude into
// [2^14, 2^15) (fn::tc::range_exp) and rounded once, in the first half of their regions; the rows' 2^-e after them
constexpr int32_t wy_wsc = wy_w + C * head_dim / 2, wy_psc = wy_p + C * C / 2;
static_assert(wy_wsc + C <= wy_u && wy_psc + C <= wy_eg, "the FP16 rows and their scales fit");
// PB3: the group region, after the run's n_blocks x n_head items: per (block, Q/K group) the scan's Q operand (FP16
// rows [64][128], each scaled as sc_row128 makes it) with the rows' 2^-e, then its K^T operand (FP16 [128][64], K
// rounded unscaled), written once by the group's prep CTA and copied by its 12 scan CTAs
constexpr int32_t wg_q = 0, wg_qsc = wg_q + C * head_dim / 2, wg_kt = wg_qsc + C, wg_floats = wg_kt + head_dim * C / 2;
static_assert(wg_qsc % 4 == 0 && wg_kt % 4 == 0 && wg_floats % 4 == 0, "16-byte aligned group pieces");

// the scan: 4 slices of 32 value rows a head (its shared memory below scan_kernel's comment)
constexpr int32_t slices = 4, srow = head_dim / slices, n_units = n_head * slices;

__device__ __forceinline__ float sigmoid_fast(float x) {  // gdn.cu's form (beta)
    float e = 0.0f, r = 0.0f;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(x * -1.44269504088896341f));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.0f + e));
    return r;
}

__device__ __forceinline__ float rsqrt_fast(float x) {  // gdn.cu's form (the q and k norms)
    float r = 0.0f;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

// ---- convolution: channel c of row t (SiLU of the 4-tap sum); the run's last row also writes c's new history ----
__device__ __forceinline__ float conv_one(const prompt_args & a, uint32_t cur, int64_t t, int32_t c) {
    const float * h = a.hist + fn::round::version_slot(cur, a.v_in) * hist_version_floats + int64_t(c) * n_hist;
    const float4 w = __ldg(reinterpret_cast<const float4 *>(a.conv_w) + c);
    // x at row t - k: the run's own input, or the history (oldest first: h[3 + (t - k)])
    // PB3: the input F32, or the product's FP16 output (qkv16, row stride ld16)
    const auto xr = [&](int64_t r) {
        return a.qkv != nullptr ? a.qkv[r * n_channel + c] : __half2float(__ushort_as_half(a.qkv16[r * a.ld16 + c]));
    };
    const auto xa = [&](int64_t r) { return r >= 0 ? xr(r) : h[n_hist + r]; };
    float acc     = xa(t - 3) * w.x;
    acc           = fmaf(xa(t - 2), w.y, acc);
    acc           = fmaf(xa(t - 1), w.z, acc);
    acc = fmaf(xr(t), w.w, acc);
    // the run's last 3 inputs; with fewer than 3 rows the older history fills the front (the last row writes every
    // tap, so no tap depends on another thread's write)
    if (t == a.rows - 1) {
        float * ho = a.hist + fn::round::version_slot(cur, a.v_out) * hist_version_floats + int64_t(c) * n_hist;
#pragma unroll
        for (int32_t j = 0; j < n_hist; ++j) {
            ho[j] = xa(t - (n_hist - 1) + j);
        }
    }
    return acc / (1.0f + expf(-acc));
}

// a thread per (row, channel)
__global__ void conv_kernel(const prompt_args a) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(a.rows) * n_channel) {
        return;
    }
    a.conv_out[i] = conv_one(a, *a.cur, i / n_channel, static_cast<int32_t>(i % n_channel));
}

// PB3 (fuse_norm): a warp per (row, 128-channel vector), lane l channels 4 l .. + 3 of it, each as conv_kernel; the
// Q and K vectors (0 .. 15) then L2-normalized as norm_kernel (the same lane order, so the same values bitwise), and
// the conv output written once
__global__ void conv_norm_kernel(const prompt_args a) {
    const int64_t wv = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (wv >= int64_t(a.rows) * (n_channel / head_dim)) {
        return;
    }
    const int64_t t = wv / (n_channel / head_dim);
    const int32_t v = static_cast<int32_t>(wv % (n_channel / head_dim));
    const int32_t c = v * head_dim + 4 * lane;
    const uint32_t cur = *a.cur;
    float4 x = make_float4(conv_one(a, cur, t, c), conv_one(a, cur, t, c + 1), conv_one(a, cur, t, c + 2),
                           conv_one(a, cur, t, c + 3));
    if (v < 2 * n_group) {
        float ss = x.x * x.x;
        ss = fmaf(x.y, x.y, ss);
        ss = fmaf(x.z, x.z, ss);
        ss = fmaf(x.w, x.w, ss);
        const float r = rsqrt_fast(warp_sum(ss) + fn::rms_eps);
        x.x *= r;
        x.y *= r;
        x.z *= r;
        x.w *= r;
    }
    *reinterpret_cast<float4 *>(a.conv_out + t * n_channel + c) = x;
}

// ---- prep: warps over (row, vector): vectors 0..7 the Q groups, 8..15 the K groups; then gates ----
__global__ void norm_kernel(const prompt_args a) {
    const int64_t wv   = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (wv >= int64_t(a.rows) * 2 * n_group) {
        return;
    }
    const int64_t t   = wv / (2 * n_group);
    const int32_t v   = static_cast<int32_t>(wv % (2 * n_group));
    const int32_t off = v < n_group ? q_first + v * head_dim : k_first + (v - n_group) * head_dim;
    float4 *      vec = reinterpret_cast<float4 *>(a.conv_out + t * n_channel + off) + lane;
    float4        x   = *vec;
    float         ss  = x.x * x.x;
    ss                = fmaf(x.y, x.y, ss);
    ss                = fmaf(x.z, x.z, ss);
    ss                = fmaf(x.w, x.w, ss);
    const float r     = rsqrt_fast(warp_sum(ss) + fn::rms_eps);
    x.x *= r;
    x.y *= r;
    x.z *= r;
    x.w *= r;
    *vec = x;
}

__global__ void gate_in_kernel(const prompt_args a) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(a.rows) * n_head) {
        return;
    }
    const int32_t h  = static_cast<int32_t>(i % n_head);
    const float   xg = a.alpha[i] + a.dt[h];
    const float   lp = logf(1.0f + expf(xg));
    const float   sp = xg > 20.0f ? xg : lp;
    a.g[i]           = sp * a.a[h];
    a.beta[i]        = sigmoid_fast(a.beta_raw[i]);
}

// one 128-wide F32 row (a lane's 4 values, columns 4 lane .. + 3) to FP16 at dst, scaled; returns 2^-e
__device__ __forceinline__ float sc_row128(float4 v, uint16_t * dst, int32_t lane) {
    const float m = fn::tc::warp_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    const int32_t e = fn::tc::range_exp(m);
    const float sc = fn::tc::pow2(e);
    *reinterpret_cast<uint2 *>(dst + 4 * lane) =
        make_uint2(fn::tc::pack2(v.x * sc, v.y * sc), fn::tc::pack2(v.z * sc, v.w * sc));
    return fn::tc::pow2(-e);
}

// the group region of block bi, group grp
__device__ __forceinline__ float * group_item(const prompt_args & a, int32_t bi, int32_t grp) {
    const int64_t nb = (a.rows + C - 1) / C;
    return a.wy + nb * n_head * prompt_wy_item_floats + (int64_t(bi) * n_group + grp) * wg_floats;
}

// ---- the recurrence, prep: a CTA per (64-row block, Q/K group) ----
// PB3: the group's raw Grams Q K^T and K K^T computed once and held in registers for its three value heads
// (h = grp + n_group r), whose decay, beta, masks, solves and W/U are each their own (the same products and order as a
// CTA a head, so every output is unchanged).
// the lower 16 x 16 block (i, j), i >= j, of A or T
__device__ __forceinline__ float * blk(float * base, int32_t i, int32_t j) {
    return base + (i * (i + 1) / 2 + j) * bs;
}

__global__ void __launch_bounds__(threads, 1) wy_kernel(const prompt_args a) {
    extern __shared__ float4 smem4[];
    float *                  sm  = reinterpret_cast<float *>(smem4);
    float *                  Ks  = sm + pm_k;
    float *                  QVs = sm + pm_q;
    float *                  As  = sm + pm_a;
    float *                  Ts  = sm + pm_t;
    float *                  Xs  = sm + pm_x;
    float *                  Gs  = sm + pm_g;
    float *                  Bs  = sm + pm_b;
    float *                  SBs = sm + pm_sb;
    const int32_t bi = static_cast<int32_t>(blockIdx.x), grp = static_cast<int32_t>(blockIdx.y);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t            t0 = bi * C, nt = min(C, a.rows - t0);

    // 1. K and Q of the group; rows past the run zero. PB3: a thread's loads issued together, then stored
    constexpr int32_t per = C * head_dim / 4 / threads; // float4s a thread stages from a [64][128] tile
    static_assert(per * threads == C * head_dim / 4, "the tile's float4s split evenly over the threads");
    {
        float4 kv[per], qv[per];
#pragma unroll
        for (int32_t q = 0; q < per; ++q) {
            const int32_t e = tid + q * threads, t = e / (head_dim / 4), c4 = e % (head_dim / 4);
            kv[q] = qv[q] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t < nt) {
                const float * row = a.conv_out + int64_t(t0 + t) * n_channel;
                kv[q] = __ldg(reinterpret_cast<const float4 *>(row + k_first + grp * head_dim) + c4);
                qv[q] = __ldg(reinterpret_cast<const float4 *>(row + q_first + grp * head_dim) + c4);
            }
        }
#pragma unroll
        for (int32_t q = 0; q < per; ++q) {
            const int32_t e = tid + q * threads, t = e / (head_dim / 4), c4 = e % (head_dim / 4);
            reinterpret_cast<float4 *>(Ks + t * ks)[c4] = kv[q];
            reinterpret_cast<float4 *>(QVs + t * ks)[c4] = qv[q];
        }
    }
    __syncthreads();

    // 2. the raw Grams Q K^T and K K^T: thread (ty, tx) rows ty + 16 i, columns tx + 16 j
    const int32_t ty = tid >> 4, tx = tid & 15;
    float qk[4][4] = {}, kk[4][4] = {};
#pragma unroll 4
    for (int32_t k4 = 0; k4 < head_dim / 4; ++k4) {
        float4 qr[4], kr[4], kc[4];
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            qr[i] = reinterpret_cast<const float4 *>(QVs + (ty + 16 * i) * ks)[k4];
            kr[i] = reinterpret_cast<const float4 *>(Ks + (ty + 16 * i) * ks)[k4];
            kc[i] = reinterpret_cast<const float4 *>(Ks + (tx + 16 * i) * ks)[k4];
        }
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (int32_t j = 0; j < 4; ++j) {
                qk[i][j] = fmaf(qr[i].x, kc[j].x, qk[i][j]);
                qk[i][j] = fmaf(qr[i].y, kc[j].y, qk[i][j]);
                qk[i][j] = fmaf(qr[i].z, kc[j].z, qk[i][j]);
                qk[i][j] = fmaf(qr[i].w, kc[j].w, qk[i][j]);
                kk[i][j] = fmaf(kr[i].x, kc[j].x, kk[i][j]);
                kk[i][j] = fmaf(kr[i].y, kc[j].y, kk[i][j]);
                kk[i][j] = fmaf(kr[i].z, kc[j].z, kk[i][j]);
                kk[i][j] = fmaf(kr[i].w, kc[j].w, kk[i][j]);
            }
        }
    }
    // PB3: the scan's Q and K^T operands of the group (the shared rows are read only, so no barrier)
    {
        float * gi = group_item(a, bi, grp);
        const int32_t w = tid >> 5, lane = tid & 31;
        uint16_t * q16 = reinterpret_cast<uint16_t *>(gi + wg_q);
#pragma unroll 1
        for (int32_t r = 0; r < 8; ++r) {
            const int32_t t = 8 * w + r;
            const float qs = sc_row128(reinterpret_cast<const float4 *>(QVs + t * ks)[lane], q16 + t * head_dim, lane);
            if (lane == 0) {
                gi[wg_qsc + t] = qs;
            }
        }
        uint32_t * kt = reinterpret_cast<uint32_t *>(gi + wg_kt);
        for (int32_t e = tid; e < head_dim * C / 2; e += threads) {
            const int32_t c = e / (C / 2), sp = e % (C / 2);
            kt[e] = fn::tc::pack2(Ks[2 * sp * ks + c], Ks[(2 * sp + 1) * ks + c]);
        }
    }

#pragma unroll 1
    for (int32_t hr = 0; hr < n_head / n_group; ++hr) {
        const int32_t h = grp + n_group * hr;
        float * item = a.wy + (int64_t(bi) * n_head + h) * prompt_wy_item_floats;
        // the previous head's readers of G, beta, V and T are done (the first head: the Grams' reads of Q)
        __syncthreads();
        // beta and g of head h (G stays constant past nt)
        if (tid < 32) {
            // the cumulative log decay: each lane's two rows, two inclusive warp scans, the first half's total carried
            const int32_t lane = tid;
            float g0 = lane < nt ? a.g[int64_t(t0 + lane) * n_head + h] : 0.0f;
            float g1 = lane + 32 < nt ? a.g[int64_t(t0 + lane + 32) * n_head + h] : 0.0f;
#pragma unroll
            for (int32_t o = 1; o < 32; o <<= 1) {
                const float u0 = __shfl_up_sync(0xffffffffu, g0, o), u1 = __shfl_up_sync(0xffffffffu, g1, o);
                if (lane >= o) {
                    g0 += u0;
                    g1 += u1;
                }
            }
            g1 += __shfl_sync(0xffffffffu, g0, 31);
            const float b0 = lane < nt ? a.beta[int64_t(t0 + lane) * n_head + h] : 0.0f;
            const float b1 = lane + 32 < nt ? a.beta[int64_t(t0 + lane + 32) * n_head + h] : 0.0f;
            const float e0 = expf(g0), e1 = expf(g1);
            Gs[lane] = g0;
            Gs[lane + 32] = g1;
            Bs[lane] = b0;
            Bs[lane + 32] = b1;
            SBs[lane] = b0 * e0;
            SBs[lane + 32] = b1 * e1;
            item[wy_eg + lane] = e0;
            item[wy_eg + lane + 32] = e1;
        }
        __syncthreads();
        if (tid < C) {
            item[wy_wd + tid] = tid < nt ? expf(Gs[C - 1] - Gs[tid]) : 0.0f;
        }

        // P from the decay-masked Q K^T into the item, A from K K^T into its lower blocks (zeros on and above the
        // diagonal, and in rows past nt)
        {
            float pv[4][4];
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                const int32_t t = ty + 16 * i;
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    const int32_t s = tx + 16 * j;
                    const bool live = t < nt;
                    const float dec = live && s <= t ? expf(Gs[t] - Gs[s]) : 0.0f;
                    pv[i][j] = s <= t ? dec * qk[i][j] : 0.0f; // the scan's causal mask
                    if (j <= i) {
                        blk(As, i, j)[ty * (db + 1) + tx] = s < t ? Bs[t] * dec * kk[i][j] : 0.0f;
                    }
                }
            }
            // P's rows as the scan's FP16 operand (the 16 threads of a row are lanes tx of one half-warp)
            uint16_t * p16 = reinterpret_cast<uint16_t *>(item + wy_p);
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                const int32_t t = ty + 16 * i;
                float m = fmaxf(fmaxf(fabsf(pv[i][0]), fabsf(pv[i][1])), fmaxf(fabsf(pv[i][2]), fabsf(pv[i][3])));
#pragma unroll
                for (int32_t o = 1; o < 16; o <<= 1) {
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                }
                const int32_t e = fn::tc::range_exp(m);
                const float sc = fn::tc::pow2(e);
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    p16[t * C + tx + 16 * j] = __half_as_ushort(__float2half_rn(pv[i][j] * sc));
                }
                if (tx == 0) {
                    item[wy_psc + t] = fn::tc::pow2(-e);
                }
            }
        }
        __syncthreads();

        // 3. V of the head over Q or the previous head's V (rows past nt zero); the diagonal blocks' inverses, a warp
        // each, lane c (< 16) column c by forward substitution
        {
            float4 vv[per]; // PB3: a thread's loads issued together, then stored
#pragma unroll
            for (int32_t q = 0; q < per; ++q) {
                const int32_t e = tid + q * threads, t = e / (head_dim / 4), c4 = e % (head_dim / 4);
                vv[q] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (t < nt) {
                    vv[q] = __ldg(reinterpret_cast<const float4 *>(a.conv_out + int64_t(t0 + t) * n_channel + v_first +
                                                                   h * head_dim) +
                                  c4);
                }
            }
#pragma unroll
            for (int32_t q = 0; q < per; ++q) {
                const int32_t e = tid + q * threads, t = e / (head_dim / 4), c4 = e % (head_dim / 4);
                reinterpret_cast<float4 *>(QVs + t * ks)[c4] = vv[q];
            }
        }
        {
            const int32_t w = tid >> 5, c = tid & 31;
            if (w < nbd && c < db) {
                const float * ab = blk(As, w, w);
                float x[db];
#pragma unroll
                for (int32_t r = 0; r < db; ++r) {
                    float v = r == c ? 1.0f : 0.0f;
#pragma unroll
                    for (int32_t q = 0; q < r; ++q) {
                        v = fmaf(-ab[r * (db + 1) + q], x[q], v);
                    }
                    x[r] = v;
                }
                float * tb = blk(Ts, w, w);
#pragma unroll
                for (int32_t r = 0; r < db; ++r) {
                    tb[r * (db + 1) + c] = x[r];
                }
            }
        }
        __syncthreads();

        // 4. the off-diagonal blocks: T_ij = -T_ii sum_{k = j}^{i - 1} A_ik T_kj, by distance i - j = 1, 2, 3
#pragma unroll 1
        for (int32_t dist = 1; dist < nbd; ++dist) {
            const int32_t pairs = nbd - dist;
            for (int32_t e = tid; e < pairs * db * db; e += threads) {
                const int32_t pr = e / (db * db), r = (e / db) % db, c = e % db;
                const int32_t j = pr, i = pr + dist;
                float v = 0.0f;
                for (int32_t k = j; k < i; ++k) {
                    const float * ab = blk(As, i, k);
                    const float * tb = blk(Ts, k, j);
#pragma unroll
                    for (int32_t q = 0; q < db; ++q) {
                        v = fmaf(ab[r * (db + 1) + q], tb[q * (db + 1) + c], v);
                    }
                }
                Xs[pr * bs + r * (db + 1) + c] = v;
            }
            __syncthreads();
            for (int32_t e = tid; e < pairs * db * db; e += threads) {
                const int32_t pr = e / (db * db), r = (e / db) % db, c = e % db;
                const int32_t j = pr, i = pr + dist;
                const float * tb = blk(Ts, i, i);
                float v = 0.0f;
                for (int32_t q = 0; q <= r; ++q) {
                    v = fmaf(tb[r * (db + 1) + q], Xs[pr * bs + q * (db + 1) + c], v);
                }
                blk(Ts, i, j)[r * (db + 1) + c] = -v;
            }
            __syncthreads();
        }

        // 5. W = T diag(beta exp G) K and U = T diag(beta) V: thread (ty, tx) rows ty + 16 i, columns 4 tx + 64 jj
        {
            float aw[4][8] = {}, au[4][8] = {};
#pragma unroll
            for (int32_t sb = 0; sb < nbd; ++sb) {
#pragma unroll 4
                for (int32_t q = 0; q < db; ++q) {
                    const int32_t s = db * sb + q;
                    const float4 k0 = reinterpret_cast<const float4 *>(Ks + s * ks)[tx];
                    const float4 k1 = reinterpret_cast<const float4 *>(Ks + s * ks)[tx + 16];
                    const float4 v0 = reinterpret_cast<const float4 *>(QVs + s * ks)[tx];
                    const float4 v1 = reinterpret_cast<const float4 *>(QVs + s * ks)[tx + 16];
                    const float kv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
                    const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
                    const float fw = SBs[s], fu = Bs[s];
#pragma unroll
                    for (int32_t i = 0; i < 4; ++i) {
                        if (i >= sb) {
                            const float tv = blk(Ts, i, sb)[ty * (db + 1) + q];
                            const float cw = tv * fw, cu = tv * fu;
#pragma unroll
                            for (int32_t c = 0; c < 8; ++c) {
                                aw[i][c] = fmaf(cw, kv[c], aw[i][c]);
                                au[i][c] = fmaf(cu, vv[c], au[i][c]);
                            }
                        }
                    }
                }
            }
            // W's rows as the scan's FP16 operand (PB3), U F32
            uint16_t * w16 = reinterpret_cast<uint16_t *>(item + wy_w);
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                const int32_t t = ty + 16 * i;
                float m = 0.0f;
#pragma unroll
                for (int32_t c = 0; c < 8; ++c) {
                    m = fmaxf(m, fabsf(aw[i][c]));
                }
#pragma unroll
                for (int32_t o = 1; o < 16; o <<= 1) {
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                }
                const int32_t e = fn::tc::range_exp(m);
                const float sc = fn::tc::pow2(e);
                if (tx == 0) {
                    item[wy_wsc + t] = fn::tc::pow2(-e);
                }
#pragma unroll
                for (int32_t jj = 0; jj < 2; ++jj) {
                    const int32_t c = 4 * tx + 64 * jj;
                    *reinterpret_cast<uint2 *>(w16 + t * head_dim + c) =
                        make_uint2(fn::tc::pack2(aw[i][4 * jj] * sc, aw[i][4 * jj + 1] * sc),
                                   fn::tc::pack2(aw[i][4 * jj + 2] * sc, aw[i][4 * jj + 3] * sc));
                    *reinterpret_cast<float4 *>(item + wy_u + t * head_dim + c) =
                        make_float4(au[i][4 * jj], au[i][4 * jj + 1], au[i][4 * jj + 2], au[i][4 * jj + 3]);
                }
            }
        }
    }
}

// ---- the recurrence, scan: a CTA per (head, slice of 32 value rows), the blocks in order (PB3: on tensor cores) ----
// The slice's state M [32 value rows][128 key channels] stays F32 in the products' accumulators; every product is
// HMMA.884 (FP16 operands, F32 accumulators; Amendment 2), each FP16 operand row scaled by the power of two that brings
// its largest magnitude into [2^14, 2^15) (fn::tc::range_exp), the scales undone in the F32 epilogues. A block:
//   A: W and Q rows to FP16 (W16, Q16); with M's FP16 copy (Ms16) V' = U - W M^T and Y = exp(G) Q M^T, a warp a
//      16 x 16 tile of the [64 rows][32 value rows] outputs (K = 128);
//   B: V' (F32, VpF), P (causal: s <= t; P16) and K transposed (K16T, |k| <= 1 unscaled) staged; V' to FP16 per value
//      row as V'^T (Vp16T) and (wd V')^T (VW16);
//   C: o = (Y + P V') / sqrt(128) (K = 64), the same tiles; M <- exp(G_last) M + (wd V')^T K, a warp a 16 x 32 tile
//      of M (K = 64); M's FP16 copy rebuilt for the next block.
// PB3: W, P, Q and K^T arrive as FP16 operands from the prep (W and P per head item, Q and K^T per group region); the
// scan copies them into shared memory.
// Shared memory (bytes): phase A W16 [64][132], Q16 [64][132], Ms16 [32][132]; phases B and C alias W16 and Q16 with
// P16 [64][68], K16T [128][68], VpF [64][33], Vp16T and VW16 [32][68] (Ms16 is rebuilt after C); then the scales.
constexpr int32_t sa_ld = 132, sb_ld = 68, vp_ld = 33;
constexpr int32_t sc_w = 0, sc_q = sc_w + C * sa_ld * 2, sc_m = sc_q + C * sa_ld * 2, sc_p = 0, sc_kt = sc_p + C * sb_ld * 2,
                  sc_vf = sc_kt + head_dim * sb_ld * 2, sc_vt = sc_vf + C * vp_ld * 4, sc_vw = sc_vt + srow * sb_ld * 2,
                  sc_end = sc_vw + srow * sb_ld * 2;
constexpr int32_t sc_small = (sc_m + srow * sa_ld * 2 > sc_end ? sc_m + srow * sa_ld * 2 : sc_end);
// F32 [64] each: wsc, qsc, psc, eg, wd; [32] each: msc, vsc, vmul; rmax [4][32]
constexpr int32_t sc_wsc = sc_small, sc_qsc = sc_wsc + C * 4, sc_psc = sc_qsc + C * 4, sc_eg = sc_psc + C * 4,
                  sc_wd = sc_eg + C * 4, sc_msc = sc_wd + C * 4, sc_vsc = sc_msc + srow * 4, sc_vmul = sc_vsc + srow * 4,
                  sc_rmax = sc_vmul + srow * 4;
constexpr int32_t scan_smem = sc_rmax + 4 * srow * 4;
static_assert(2 * scan_smem <= 96 * 1024, "two scan CTAs an SM");

__device__ __forceinline__ uint2 lds64(const uint16_t * p) {
    return *reinterpret_cast<const uint2 *>(p);
}

__global__ void __launch_bounds__(threads, 2) scan_kernel(const prompt_args a) {
    extern __shared__ float4 smem4[];
    uint8_t *     sm    = reinterpret_cast<uint8_t *>(smem4);
    uint16_t *    W16   = reinterpret_cast<uint16_t *>(sm + sc_w);
    uint16_t *    Q16   = reinterpret_cast<uint16_t *>(sm + sc_q);
    uint16_t *    Ms16  = reinterpret_cast<uint16_t *>(sm + sc_m);
    uint16_t *    P16   = reinterpret_cast<uint16_t *>(sm + sc_p);
    uint16_t *    K16T  = reinterpret_cast<uint16_t *>(sm + sc_kt);
    float *       VpF   = reinterpret_cast<float *>(sm + sc_vf);
    uint16_t *    Vp16T = reinterpret_cast<uint16_t *>(sm + sc_vt);
    uint16_t *    VW16  = reinterpret_cast<uint16_t *>(sm + sc_vw);
    float *       wsc   = reinterpret_cast<float *>(sm + sc_wsc);
    float *       qsc   = reinterpret_cast<float *>(sm + sc_qsc);
    float *       psc   = reinterpret_cast<float *>(sm + sc_psc);
    float *       egs   = reinterpret_cast<float *>(sm + sc_eg);
    float *       wds   = reinterpret_cast<float *>(sm + sc_wd);
    float *       msc   = reinterpret_cast<float *>(sm + sc_msc);
    float *       vsc   = reinterpret_cast<float *>(sm + sc_vsc);
    float *       vmul  = reinterpret_cast<float *>(sm + sc_vmul);
    float *       rmax  = reinterpret_cast<float *>(sm + sc_rmax);
    const int32_t h = static_cast<int32_t>(blockIdx.x) / slices, p = static_cast<int32_t>(blockIdx.x) % slices;
    const int32_t grp = h % n_group, tid = static_cast<int32_t>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int32_t qp = (lane >> 2) & 3, ra = (lane & 3) | ((lane >> 2) & 4);
    const int32_t lr = ((lane & 16) ? 4 : 0) | (lane & 1), cc = 2 * ((lane >> 1) & 1);
    const uint32_t    cur = *a.cur;
    constexpr int64_t sv  = state_version_floats;
    const float * st_in = a.state + fn::round::version_slot(cur, a.v_in) * sv + (int64_t(h) * head_dim + p * srow) * head_dim;
    float * st_out = a.state + fn::round::version_slot(cur, a.v_out) * sv + (int64_t(h) * head_dim + p * srow) * head_dim;
    // M's tile: value rows 16 wj + 8 (qp & 1) + lr + (i & 2), key channels 32 wk + 16 ni + 8 (qp >> 1) + cc + (i & 5)
    const int32_t wj = warp >> 2, wk = warp & 3;
    float         m[2][8];
#pragma unroll
    for (int32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t j = 16 * wj + 8 * (qp & 1) + lr + (i & 2), k = 32 * wk + 16 * ni + 8 * (qp >> 1) + cc + (i & 5);
            m[ni][i]        = st_in[j * head_dim + k];
        }
    }
    // M's FP16 copy and its row scales: each lane's two rows' maxima, over the 8 lanes and the 4 warps of a row
    const auto rebuild = [&]() {
        float mx[2] = { 0.0f, 0.0f };
#pragma unroll
        for (int32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                mx[(i >> 1) & 1] = fmaxf(mx[(i >> 1) & 1], fabsf(m[ni][i]));
            }
        }
#pragma unroll
        for (int32_t rh = 0; rh < 2; ++rh) {
            mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 2));
            mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 8));
            if ((lane & 10) == 0) {
                rmax[wk * srow + 16 * wj + 8 * (qp & 1) + lr + 2 * rh] = mx[rh];
            }
        }
        __syncthreads();  // also: phase C's reads of the aliased VpF, Vp16T and VW16 are done
#pragma unroll
        for (int32_t rh = 0; rh < 2; ++rh) {
            const int32_t j  = 16 * wj + 8 * (qp & 1) + lr + 2 * rh;
            const float   rm = fmaxf(fmaxf(rmax[j], rmax[srow + j]), fmaxf(rmax[2 * srow + j], rmax[3 * srow + j]));
            const int32_t e  = fn::tc::range_exp(rm);
            const float   sc = fn::tc::pow2(e);
#pragma unroll
            for (int32_t ni = 0; ni < 2; ++ni) {
                const int32_t k = 32 * wk + 16 * ni + 8 * (qp >> 1) + cc;
                *reinterpret_cast<uint32_t *>(Ms16 + j * sa_ld + k) = fn::tc::pack2(m[ni][2 * rh] * sc, m[ni][2 * rh + 1] * sc);
                *reinterpret_cast<uint32_t *>(Ms16 + j * sa_ld + k + 4) =
                    fn::tc::pack2(m[ni][2 * rh + 4] * sc, m[ni][2 * rh + 5] * sc);
            }
            if (wk == 0 && (lane & 10) == 0) {
                msc[j] = fn::tc::pow2(-e);
            }
        }
        __syncthreads();
    };
    rebuild();
    const float   scale    = 1.0f / sqrtf(float(head_dim));
    const int32_t n_blocks = (a.rows + C - 1) / C;
    // the output tiles of A and C: rows 16 mt + 8 (qp & 1) + lr + (i & 2), value rows 16 nv + 8 (qp >> 1) + cc + (i & 5)
    const int32_t mt = warp >> 1, nv = warp & 1;
#pragma unroll 1
    for (int32_t bi = 0; bi < n_blocks; ++bi) {
        const int32_t t0 = bi * C, nt = min(C, a.rows - t0);
        const float * item = a.wy + (int64_t(bi) * n_head + h) * prompt_wy_item_floats;
        const float * gi = group_item(a, bi, grp);
        // A: W and Q rows (warp w: rows 8 w .. 8 w + 7), exp(G) and the update weights. PB3: W's rows as the prep
        // wrote them (FP16, scaled as sc_row128 would), Q's as the group's prep wrote them (sc_row128's rows; zero past
        // nt); a warp's 16 row loads issued together, then stored
        uint2 wv[8], qv[8];
#pragma unroll
        for (int32_t r = 0; r < 8; ++r) {
            const int32_t t = 8 * warp + r;
            wv[r] = __ldg(reinterpret_cast<const uint2 *>(item + wy_w) + t * (head_dim / 4) + lane);
            qv[r] = __ldg(reinterpret_cast<const uint2 *>(gi + wg_q) + t * (head_dim / 4) + lane);
        }
        if (lane < 8) {
            wsc[8 * warp + lane] = __ldg(item + wy_wsc + 8 * warp + lane);
            qsc[8 * warp + lane] = __ldg(gi + wg_qsc + 8 * warp + lane);
        }
        if (tid < C) {
            egs[tid] = __ldg(item + wy_eg + tid);
            wds[tid] = __ldg(item + wy_wd + tid);
        }
#pragma unroll
        for (int32_t r = 0; r < 8; ++r) {
            const int32_t t = 8 * warp + r;
            *reinterpret_cast<uint2 *>(W16 + t * sa_ld + 4 * lane) = wv[r];
            *reinterpret_cast<uint2 *>(Q16 + t * sa_ld + 4 * lane) = qv[r];
        }
        // PB3: the lane's U values, loaded ahead of the barrier and the products
        float uv[8];
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t t = 16 * mt + 8 * (qp & 1) + lr + (i & 2), j = 16 * nv + 8 * (qp >> 1) + cc + (i & 5);
            uv[i] = __ldg(item + wy_u + t * head_dim + p * srow + j);
        }
        __syncthreads();
        float aw[8] = {}, aq[8] = {};
        {
            const uint16_t * wr = W16 + (16 * mt + 8 * (qp & 1) + ra) * sa_ld;
            const uint16_t * qr = Q16 + (16 * mt + 8 * (qp & 1) + ra) * sa_ld;
            const uint16_t * mr = Ms16 + (16 * nv + 8 * (qp >> 1) + ra) * sa_ld;
#pragma unroll 8
            for (int32_t ks = 0; ks < head_dim / 4; ++ks) {
                const uint2 bm = lds64(mr + 4 * ks), w2 = lds64(wr + 4 * ks), q2 = lds64(qr + 4 * ks);
                fn::tc::detail::mma884(aw, w2.x, w2.y, bm.x, bm.y);
                fn::tc::detail::mma884(aq, q2.x, q2.y, bm.x, bm.y);
            }
        }
        float vp[8], yq[8];
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t t = 16 * mt + 8 * (qp & 1) + lr + (i & 2), j = 16 * nv + 8 * (qp >> 1) + cc + (i & 5);
            vp[i] = uv[i] - (aw[i] * wsc[t]) * msc[j];
            yq[i] = egs[t] * ((aq[i] * qsc[t]) * msc[j]);
        }
        // B: V' (F32; zero past nt, so the padding rows touch neither the scales nor M), P (causal) and K^T staged.
        // PB3: P's rows as the prep wrote them (masked, FP16, scaled) and K^T as the group's prep wrote it (FP16,
        // unscaled; zero columns past nt), all of a thread's loads issued together before the barrier
        constexpr int32_t kt_per = head_dim * C / 2 / threads;
        static_assert(kt_per * threads == head_dim * C / 2, "K^T's words split evenly over the threads");
        uint32_t pw[8], kw[kt_per];
#pragma unroll
        for (int32_t r = 0; r < 8; ++r) {
            pw[r] = __ldg(reinterpret_cast<const uint32_t *>(item + wy_p) + (8 * warp + r) * (C / 2) + lane);
        }
#pragma unroll
        for (int32_t q = 0; q < kt_per; ++q) {
            kw[q] = __ldg(reinterpret_cast<const uint32_t *>(gi + wg_kt) + tid + q * threads);
        }
        const float ps = lane < 8 ? __ldg(item + wy_psc + 8 * warp + lane) : 0.0f;
        __syncthreads();  // W16 and Q16 are free
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t t = 16 * mt + 8 * (qp & 1) + lr + (i & 2), j = 16 * nv + 8 * (qp >> 1) + cc + (i & 5);
            VpF[t * vp_ld + j] = t < nt ? vp[i] : 0.0f;
        }
#pragma unroll
        for (int32_t r = 0; r < 8; ++r) {
            *reinterpret_cast<uint32_t *>(P16 + (8 * warp + r) * sb_ld + 2 * lane) = pw[r];
        }
        if (lane < 8) {
            psc[8 * warp + lane] = ps;
        }
#pragma unroll
        for (int32_t q = 0; q < kt_per; ++q) {
            const int32_t idx = tid + q * threads, c = idx / (C / 2), sp = idx % (C / 2);
            reinterpret_cast<uint32_t *>(K16T + c * sb_ld)[sp] = kw[q];
        }
        __syncthreads();
        if (tid < srow) {
            float mx = 0.0f;
            for (int32_t t = 0; t < C; ++t) {
                mx = fmaxf(mx, fabsf(VpF[t * vp_ld + tid]));
            }
            const int32_t e = fn::tc::range_exp(mx);
            vmul[tid]       = fn::tc::pow2(e);
            vsc[tid]        = fn::tc::pow2(-e);
        }
        __syncthreads();
        for (int32_t idx = tid; idx < srow * C; idx += threads) {
            const int32_t s = idx & (C - 1), j = idx / C;
            const float   v = VpF[s * vp_ld + j] * vmul[j];
            Vp16T[j * sb_ld + s] = __half_as_ushort(__float2half_rn(v));
            VW16[j * sb_ld + s]  = __half_as_ushort(__float2half_rn(wds[s] * v));
        }
        __syncthreads();
        // C: the output, then the state
        {
            float            ao[8] = {};
            const uint16_t * pr    = P16 + (16 * mt + 8 * (qp & 1) + ra) * sb_ld;
            const uint16_t * vr    = Vp16T + (16 * nv + 8 * (qp >> 1) + ra) * sb_ld;
#pragma unroll
            for (int32_t ks = 0; ks < C / 4; ++ks) {
                const uint2 a2 = lds64(pr + 4 * ks), b2 = lds64(vr + 4 * ks);
                fn::tc::detail::mma884(ao, a2.x, a2.y, b2.x, b2.y);
            }
            // PB3: F32 o, or FP16 o16: the lane's columns j, j + 1 (elements i, i + 1) as one word
            bool bad = false;
#pragma unroll
            for (int32_t i = 0; i < 8; i += 2) {
                const int32_t t = 16 * mt + 8 * (qp & 1) + lr + (i & 2), j = 16 * nv + 8 * (qp >> 1) + cc + (i & 5);
                if (t < nt) {
                    const float v0 = (yq[i] + (ao[i] * psc[t]) * vsc[j]) * scale;
                    const float v1 = (yq[i + 1] + (ao[i + 1] * psc[t]) * vsc[j + 1]) * scale;
                    const int64_t off = int64_t(t0 + t) * n_value + h * head_dim + p * srow + j;
                    if (a.o != nullptr) {
                        a.o[off] = v0;
                        a.o[off + 1] = v1;
                    } else {
                        bad = bad || !(fabsf(v0) <= 65504.0f) || !(fabsf(v1) <= 65504.0f);
                        *reinterpret_cast<uint32_t *>(a.o16 + off) = fn::tc::pack2(v0, v1);
                    }
                }
            }
            if (bad) {
                atomicOr(a.err, 1u);
            }
        }
        {
            float            am[2][8] = {};
            const uint16_t * vr       = VW16 + (16 * wj + 8 * (qp & 1) + ra) * sb_ld;
#pragma unroll
            for (int32_t ks = 0; ks < C / 4; ++ks) {
                const uint2 a2 = lds64(vr + 4 * ks);
#pragma unroll
                for (int32_t ni = 0; ni < 2; ++ni) {
                    const uint2 b2 = lds64(K16T + (32 * wk + 16 * ni + 8 * (qp >> 1) + ra) * sb_ld + 4 * ks);
                    fn::tc::detail::mma884(am[ni], a2.x, a2.y, b2.x, b2.y);
                }
            }
            const float dc = egs[C - 1];
#pragma unroll
            for (int32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
                for (int32_t i = 0; i < 8; ++i) {
                    const int32_t j = 16 * wj + 8 * (qp & 1) + lr + (i & 2);
                    m[ni][i]        = fmaf(dc, m[ni][i], am[ni][i] * vsc[j]);
                }
            }
        }
        rebuild();
    }
#pragma unroll
    for (int32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t j = 16 * wj + 8 * (qp & 1) + lr + (i & 2), k = 32 * wk + 16 * ni + 8 * (qp >> 1) + cc + (i & 5);
            st_out[j * head_dim + k] = m[ni][i];
        }
    }
}

// ---- the gated norm: a warp per (row, head) ----
__global__ void gate_kernel(const prompt_args a) {
    const int64_t wv   = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (wv >= int64_t(a.rows) * n_head) {
        return;
    }
    const int64_t t    = wv / n_head;
    const int32_t h    = static_cast<int32_t>(wv % n_head);
    const int64_t off  = t * n_value + h * head_dim + 4 * lane;
    float4 o;
    if (a.o != nullptr) {
        o = *reinterpret_cast<const float4 *>(a.o + off);
    } else { // PB3: o16
        const uint2 oh = *reinterpret_cast<const uint2 *>(a.o16 + off);
        const float2 o01 = __half22float2(*reinterpret_cast<const __half2 *>(&oh.x));
        const float2 o23 = __half22float2(*reinterpret_cast<const __half2 *>(&oh.y));
        o = make_float4(o01.x, o01.y, o23.x, o23.y);
    }
    float         ss   = o.x * o.x;
    ss                 = fmaf(o.y, o.y, ss);
    ss                 = fmaf(o.z, o.z, ss);
    ss                 = fmaf(o.w, o.w, ss);
    const float  r     = 1.0f / sqrtf(warp_sum(ss) / float(head_dim) + fn::rms_eps);
    // PB3: z F32, or the product's FP16 output (z16, row stride ld16)
    float4 z;
    if (a.z != nullptr) {
        z = *reinterpret_cast<const float4 *>(a.z + off);
    } else {
        const uint2 zh = *reinterpret_cast<const uint2 *>(a.z16 + t * a.ld16 + h * head_dim + 4 * lane);
        const float2 z01 = __half22float2(*reinterpret_cast<const __half2 *>(&zh.x));
        const float2 z23 = __half22float2(*reinterpret_cast<const __half2 *>(&zh.y));
        z = make_float4(z01.x, z01.y, z23.x, z23.y);
    }
    const float4 w     = __ldg(reinterpret_cast<const float4 *>(a.norm_w) + lane);
    const float  ov[4] = { o.x, o.y, o.z, o.w }, zv[4] = { z.x, z.y, z.z, z.w }, wvv[4] = { w.x, w.y, w.z, w.w };
    float        out[4];
    bool         bad = false;
#pragma unroll
    for (int32_t k = 0; k < 4; ++k) {
        out[k] = ov[k] * r * wvv[k] * (1.0f / (1.0f + expf(-zv[k])));
        bad    = bad || !(fabsf(out[k]) <= 65504.0f);
    }
    if (bad && a.err != nullptr) {
        atomicOr(a.err, 1u);
    }
    if (a.gated != nullptr) {
        *reinterpret_cast<float4 *>(a.gated + off) = make_float4(out[0], out[1], out[2], out[3]);
    }
    const uint32_t h0 = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(out[0]))) |
                        (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(out[1]))) << 16);
    const uint32_t h1 = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(out[2]))) |
                        (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(out[3]))) << 16);
    *reinterpret_cast<uint2 *>(a.gated16 + off) = make_uint2(h0, h1);
}

uint32_t blocks(int64_t n, int32_t per) {
    return static_cast<uint32_t>((n + per - 1) / per);
}

bool common_ok(const prompt_args & a, void * stream) {
    return a.rows >= 1 && a.rows <= prompt_max_rows && a.cur != nullptr && stream != nullptr;
}

bool fail(const char * what) {
    fprintf(stderr, "gdn_prompt: %s rejected\n", what);
    return false;
}

// cudaFuncSetAttribute is per device; the kernels run on both cards
bool smem_once(std::atomic<uint32_t> & done) {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        return false;
    }
    const uint32_t bit = 1u << dev;
    if ((done.load() & bit) != 0u) {
        return true;
    }
    if (cudaFuncSetAttribute(wy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, prep_smem) != cudaSuccess) {
        return false;
    }
    done.fetch_or(bit);
    return true;
}

std::atomic<uint32_t> wy_attr{ 0 };

bool resources(const void * f, int32_t dyn, prompt_resources & r) {
    cudaFuncAttributes at = {};
    if (cudaFuncGetAttributes(&at, f) != cudaSuccess) {
        return false;
    }
    r.regs             = at.numRegs;
    r.local_bytes      = static_cast<int32_t>(at.localSizeBytes);
    r.shared_bytes     = static_cast<int32_t>(at.sharedSizeBytes);
    r.dyn_shared_bytes = dyn;
    return true;
}

}  // namespace

bool gdn_prompt_conv(const prompt_args & a, void * stream) {
    const bool in16 = a.qkv16 != nullptr && a.ld16 >= n_channel;
    if (!common_ok(a, stream) || (a.qkv == nullptr && !in16) || a.conv_out == nullptr || a.conv_w == nullptr ||
        a.hist == nullptr || a.v_in < 0 || a.v_out < 1 || a.v_in >= a.v_out || a.v_out > 4) {
        return fail("conv");
    }
    if (a.fuse_norm) {
        conv_norm_kernel<<<blocks(int64_t(a.rows) * (n_channel / head_dim) * 32, 256), 256, 0,
                           static_cast<cudaStream_t>(stream)>>>(a);
    } else {
        conv_kernel<<<blocks(int64_t(a.rows) * n_channel, 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool gdn_prompt_prep(const prompt_args & a, void * stream) {
    if (!common_ok(a, stream) || a.conv_out == nullptr || a.alpha == nullptr || a.beta_raw == nullptr ||
        a.dt == nullptr || a.a == nullptr || a.g == nullptr || a.beta == nullptr) {
        return fail("prep");
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (!a.fuse_norm) {
        norm_kernel<<<blocks(int64_t(a.rows) * 2 * n_group * 32, 256), 256, 0, s>>>(a);
    }
    gate_in_kernel<<<blocks(int64_t(a.rows) * n_head, 256), 256, 0, s>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool gdn_prompt_recur(const prompt_args & a, void * stream) {
    if (!common_ok(a, stream) || a.conv_out == nullptr || a.g == nullptr || a.beta == nullptr ||
        (a.o == nullptr && (a.o16 == nullptr || a.err == nullptr)) || a.state == nullptr || a.wy == nullptr ||
        a.v_in < 0 || a.v_out < 1 || a.v_in >= a.v_out || a.v_out > 4 || !smem_once(wy_attr)) {
        return fail("recur");
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    wy_kernel<<<dim3(static_cast<uint32_t>((a.rows + C - 1) / C), n_group), threads, prep_smem, s>>>(a);
    scan_kernel<<<n_units, threads, scan_smem, s>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool gdn_prompt_gate(const prompt_args & a, void * stream) {
    const bool z16 =
        a.z16 != nullptr && a.ld16 >= n_value && a.ld16 % 4 == 0 && (reinterpret_cast<uintptr_t>(a.z16) & 7u) == 0;
    if (!common_ok(a, stream) || (a.o == nullptr && a.o16 == nullptr) || (a.z == nullptr && !z16) ||
        a.norm_w == nullptr || a.gated16 == nullptr) {
        return fail("gate");
    }
    gate_kernel<<<blocks(int64_t(a.rows) * n_head * 32, 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

int64_t prompt_wy_bytes(int32_t rows) {
    return int64_t((rows + C - 1) / C) * (n_head * prompt_wy_item_floats + n_group * wg_floats) * 4;
}

bool gdn_prompt_recur_resources(prompt_resources & prep, prompt_resources & scan) {
    return resources(reinterpret_cast<const void *>(wy_kernel), prep_smem, prep) &&
           resources(reinterpret_cast<const void *>(scan_kernel), scan_smem, scan);
}

}  // namespace fn::gdn
