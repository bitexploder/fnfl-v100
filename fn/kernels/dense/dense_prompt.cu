#include "fn/kernels/dense/dense_dec.cuh"
#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/dense/dense_ladder.cuh"
#include "fn/kernels/dense/dense_prompt_dec.cuh"
#include "fn/kernels/tc/tc_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace fn::dense {
namespace {

using namespace pdc;  // APF2: the decode helpers and pd_tc's epilogue, shared with the ladder (dense_prompt_dec.cuh)

constexpr int32_t pd_threads = 256;
constexpr int32_t pd_cols    = 256;  // columns a block: 8 warps x 32
constexpr int32_t pd_rows =
    64;  // rows a panel (8 tiles of 8; one CTA an SM: the unpack and the tiles need about 150-200 registers)
constexpr int32_t pd_nt     = pd_rows / 8;

// ---- the product kernel -------------------------------------------------------------------------------------------

struct pd_kargs {
    pd_args a;
    int32_t steps;       // stages over K
    int32_t row_bytes8;  // DP2: a tile's bytes a step (8 rows x 256 weights)
};

// stage s of the panel's A rows into shared memory (the kernel's K order, swizzled), and Q4_K/Q5_K's group sums.
// Thread t < 4 pd_rows: row t / 4, the stage's 32-K group t % 4 (DP2) or 16 values (EX1), original K order in, the
// kernel's order out.
template <fmt F, pd_layout L>
__device__ __forceinline__ void stage_a(const pd_kargs & k, int32_t row0, int32_t s, uint8_t * at, float * gs) {
    const int32_t t   = static_cast<int32_t>(threadIdx.x);
    const int32_t row = t >> 2, g = t & 3;
    const int32_t r = row0 + row;
    if (row >= pd_rows) {
        return;  // threads 4 pd_rows .. stage nothing
    }
    constexpr int32_t SK   = L == pd_layout::dp2 ? dp2_stage : ex1_stage;
    constexpr int32_t VALS = SK / 4;  // values a thread stages: 32 (DP2) or 16 (EX1)
    const int32_t     k0   = s * SK + g * VALS;
    uint4             v[VALS / 8];
    if (r < k.a.rows) {
        const pd_act &   x   = k.a.x;
        const uint16_t * src = k0 < x.k_split ? x.a0 + (int64_t(r) >> x.shift0) * x.ld0 + k0 :
                                                x.a1 + (int64_t(r) >> x.shift1) * x.ld1 + (k0 - x.k_split);
#pragma unroll
        for (int32_t c = 0; c < VALS / 8; ++c) {
            v[c] = __ldg(reinterpret_cast<const uint4 *>(src) + c);
        }
    } else {
#pragma unroll
        for (int32_t c = 0; c < VALS / 8; ++c) {
            v[c] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    uint8_t * rowp = at + row * (SK * 2);
    if constexpr (L == pd_layout::dp2) {
        // chunks of 8 keep together in every DP2 format: chunk oc (of the stage's 16) goes to chunk sc, swizzled
#pragma unroll
        for (int32_t c = 0; c < 4; ++c) {
            const int32_t oc = 4 * g + c;
            int32_t       sc = oc;
            if constexpr (F == fmt::q6_k) {
                const int32_t i = oc >> 2, rem = oc & 3;
                sc = 8 * (rem >> 1) + 2 * i + (rem & 1);
            }
            *reinterpret_cast<uint4 *>(rowp + 16 * (sc ^ (row & 7))) = v[c];
        }
        if constexpr (has_min<F>()) {
            float sum = 0.0f;
#pragma unroll
            for (int32_t c = 0; c < 4; ++c) {
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    const uint32_t h2 = word(v[c], j);
                    sum += __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xffffu)));
                    sum += __half2float(__ushort_as_half(static_cast<unsigned short>(h2 >> 16)));
                }
            }
            gs[row * 4 + g] = sum;
        }
    } else {
        // element by element: original K k0 + e to the kernel's position (inverse of k_of), swizzled by 16-byte chunk
        const uint16_t * h = reinterpret_cast<const uint16_t *>(v);
#pragma unroll
        for (int32_t e = 0; e < VALS; ++e) {
            const int32_t ko  = g * VALS + e;  // original K in the stage
            int32_t       pos = ko;
            if constexpr (F == fmt::iq4_nl) {
                const int32_t bb = ko >> 5, kk = ko & 31, lo = kk >= 16 ? 1 : 0, k1 = kk & 15;
                const int32_t c = k1 >> 2, rem = k1 & 3;
                pos = 32 * bb + 4 * (2 * c + (rem >> 1)) + 2 * (rem & 1) + lo;
            }
            const int32_t ch                                                    = pos >> 3;
            reinterpret_cast<uint16_t *>(rowp + 16 * (ch ^ (row & 7)))[pos & 7] = h[e];
        }
    }
}

template <fmt F, pd_layout L> __global__ void __launch_bounds__(pd_threads, 1) pd_kernel(const pd_kargs k) {
    constexpr int32_t SK  = L == pd_layout::dp2 ? dp2_stage : ex1_stage;
    constexpr int32_t QPS = L == pd_layout::dp2 ? 2 : 1;  // code blocks of 16 words a stage
    constexpr int32_t GW  = group_words<F>();
    constexpr int32_t NG  = 16 / GW;                      // groups a code block
    __shared__ __align__(16) uint8_t at[2][pd_rows * SK * 2];
    __shared__ float                 gs[2][pd_rows * 4];
    __shared__ uint32_t              grid_s[F == fmt::iq3_s ? 512 : 1];
    const int32_t                    tid = static_cast<int32_t>(threadIdx.x);
    const int32_t                    w = tid >> 5, lane = tid & 31;
    const int32_t                    qp = (lane >> 2) & 3, r = (lane & 3) | ((lane >> 2) & 4);
    const int32_t n       = static_cast<int32_t>(blockIdx.x) * pd_cols + 32 * w + 8 * qp + r;  // the lane's column
    const bool    warp_on = static_cast<int32_t>(blockIdx.x) * pd_cols + 32 * w < k.a.n;
    const int32_t row0    = static_cast<int32_t>(blockIdx.y) * pd_rows;
    if constexpr (F == fmt::iq3_s) {
        for (int32_t i = tid; i < 512; i += pd_threads) {
            grid_s[i] = __ldg(reinterpret_cast<const uint32_t *>(k.a.grids) + i);
        }
    }
    uint4 kv = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (F == fmt::iq4_xs || F == fmt::iq4_nl) {
        kv = __ldg(reinterpret_cast<const uint4 *>(k.a.grids + 2048));
    }
    float acc[pd_nt][8];
#pragma unroll
    for (int32_t t = 0; t < pd_nt; ++t) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            acc[t][i] = 0.0f;
        }
    }
    const int32_t lr = ((lane & 16) ? 4 : 0) | (lane & 1);  // the lane's C rows: lr and lr + 2 of each tile
    stage_a<F, L>(k, row0, 0, at[0], gs[0]);
    __syncthreads();
#pragma unroll 1
    for (int32_t s = 0; s < k.steps; ++s) {
        const int32_t b = s & 1;
        if (s + 1 < k.steps) {
            stage_a<F, L>(k, row0, s + 1, at[b ^ 1], gs[b ^ 1]);
        }
        if (warp_on) {
#pragma unroll 1
            for (int32_t qq = 0; qq < QPS; ++qq) {
                quarter_codes qc;
                if constexpr (L == pd_layout::dp2) {
                    const int32_t   step = s >> 1, q = 2 * (s & 1) + qq;
                    const uint8_t * p =
                        k.a.w + int64_t(n >> 3) * k.row_bytes8 * (k.a.k / 256) + int64_t(step) * k.row_bytes8;
                    dp2_quarter<F>(p, r, q, kv, grid_s, qc);
                } else {
                    ex1_stage_codes<F>(k.a.w, n, s, kv, qc);
                }
#pragma unroll
                for (int32_t g = 0; g < NG; ++g) {
                    // this group's scales and mins of the lane's four C columns
                    float fc[4], mc[4];
#pragma unroll
                    for (int32_t j = 0; j < 4; ++j) {
                        const int32_t col = (((lane >> 1) & 1) << 1) + (j & 1) + 4 * (j >> 1);
                        fc[j]             = __shfl_sync(0xffffffffu, qc.f[g], col_lane(lane, col));
                        mc[j] = has_min<F>() ? __shfl_sync(0xffffffffu, qc.m[g], col_lane(lane, col)) : 0.0f;
                    }
                    uint32_t bf[GW][2];
#pragma unroll
                    for (int32_t x = 0; x < GW; ++x) {
                        codes16(qc.w[g * GW + x], is_signed_code<F>() ? bias_s : bias_u, bf[x][0], bf[x][1]);
                    }
#pragma unroll
                    for (int32_t t = 0; t < pd_nt; ++t) {
                        float           p[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                        const int32_t   arow = 8 * t + r;
                        const uint8_t * ar   = at[b] + arow * (SK * 2);
#pragma unroll
                        for (int32_t x = 0; x < GW; ++x) {
                            // stage_a put word x's four values together at 64 qq + 4 x (EX1: 4 x)
                            const int32_t pos = 64 * qq + 4 * (g * GW + x);
                            const uint2   av  = lds64(ar + 16 * ((pos >> 3) ^ (arow & 7)) + 2 * (pos & 7));
                            mma884(p, av.x, av.y, bf[x][0], bf[x][1]);
                        }
#pragma unroll
                        for (int32_t i = 0; i < 8; ++i) {
                            const int32_t j = (i & 1) + 2 * (i >> 2);
                            acc[t][i]       = fmaf(fc[j], p[i], acc[t][i]);
                            if constexpr (has_min<F>()) {
                                const int32_t grow = 8 * t + ((i & 2) | lr);
                                acc[t][i]          = fmaf(-mc[j], gs[b][grow * 4 + 2 * qq + g], acc[t][i]);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!warp_on) {
        return;
    }
    const int32_t nb = static_cast<int32_t>(blockIdx.x) * pd_cols + 32 * w + 8 * qp;
#pragma unroll
    for (int32_t t = 0; t < pd_nt; ++t) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t row = row0 + 8 * t + c_row(lane, i);
            if (row < k.a.rows) {
                const float sc = k.a.row_scale != nullptr ? __ldg(k.a.row_scale + row) : 1.0f;
                k.a.out[int64_t(row) * k.a.ld_out + nb + c_col(lane, i)] = acc[t][i] * sc;
            }
        }
    }
}

// pd-test: each weight's code, scale and min at its K position, from the same device code the product runs
template <fmt F, pd_layout L>
__global__ void __launch_bounds__(pd_threads) pd_codes_kernel(pd_kargs k, float * code, float * scale, float * mins) {
    constexpr int32_t   SK  = L == pd_layout::dp2 ? dp2_stage : ex1_stage;
    constexpr int32_t   QPS = L == pd_layout::dp2 ? 2 : 1;
    constexpr int32_t   GW  = group_words<F>();
    __shared__ uint32_t grid_s[F == fmt::iq3_s ? 512 : 1];
    const int32_t       tid = static_cast<int32_t>(threadIdx.x);
    const int32_t       w = tid >> 5, lane = tid & 31;
    const int32_t       qp = (lane >> 2) & 3, r = (lane & 3) | ((lane >> 2) & 4);
    const int32_t       n = static_cast<int32_t>(blockIdx.x) * pd_cols + 32 * w + 8 * qp + r;
    if constexpr (F == fmt::iq3_s) {
        for (int32_t i = tid; i < 512; i += pd_threads) {
            grid_s[i] = __ldg(reinterpret_cast<const uint32_t *>(k.a.grids) + i);
        }
        __syncthreads();
    }
    uint4 kv = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (F == fmt::iq4_xs || F == fmt::iq4_nl) {
        kv = __ldg(reinterpret_cast<const uint4 *>(k.a.grids + 2048));
    }
    if (static_cast<int32_t>(blockIdx.x) * pd_cols + 32 * w >= k.a.n) {
        return;
    }
    for (int32_t s = 0; s < k.steps; ++s) {
        for (int32_t qq = 0; qq < QPS; ++qq) {
            quarter_codes qc;
            if constexpr (L == pd_layout::dp2) {
                const int32_t   step = s >> 1, q = 2 * (s & 1) + qq;
                const uint8_t * p =
                    k.a.w + int64_t(n >> 3) * k.row_bytes8 * (k.a.k / 256) + int64_t(step) * k.row_bytes8;
                dp2_quarter<F>(p, r, q, kv, grid_s, qc);
            } else {
                ex1_stage_codes<F>(k.a.w, n, s, kv, qc);
            }
            for (int32_t x = 0; x < 16; ++x) {
                uint32_t lo, hi;
                codes16(qc.w[x], is_signed_code<F>() ? bias_s : bias_u, lo, hi);
                const uint32_t h[2] = { lo, hi };
                for (int32_t i = 0; i < 4; ++i) {
                    const int32_t        kk = s * SK + k_of<F, L>(qq, x, i);
                    const unsigned short hb = static_cast<unsigned short>((h[i >> 1] >> (16 * (i & 1))) & 0xffffu);
                    const int64_t        o  = int64_t(n) * k.a.k + kk;
                    code[o]                 = __half2float(__ushort_as_half(hb));
                    scale[o]                = qc.f[x / GW];
                    mins[o]                 = qc.m[x / GW];
                }
            }
        }
    }
}

// ---- PB3: convert once, then the tensor product ---------------------------------------------------------------------


// MUP1 piece 2: a quarter's 16 eight-byte words of each of the warp's 32 columns in its staging buffer (4 KiB a warp,
// so 32 KiB a CTA and three CTAs an SM), word x of column c at 16 c + (x ^ (c & 15)): the warp's 32 columns storing
// one word, or 16 lanes reading one column's words, spread over the banks
__device__ __forceinline__ int32_t cv_at(int32_t c, int32_t x) {
    return 16 * c + (x ^ (c & 15));
}

// a CTA per (256 columns, stage): lane (qp, r) decodes column n's stage (pd_codes_kernel's walk) into its weights in
// K order, 8 bytes a word (a word's 4 values are consecutive in K except IQ4_NL's, whose word pair 2c, 2c + 1 holds
// K 4c .. 4c + 3 and 4c + 16 .. 4c + 19 of its block). MUP1 piece 2: each warp stages its 32 columns' words of a
// quarter in shared memory, then stores them row by row (two rows of wt an instruction, 16 lanes a row: DP2 and EX1's
// Q2_0 and Q8_0 write 128 contiguous bytes a row, Q6_K four 32-byte runs), not one word in each of 32 rows a store;
// the same values at the same places. EX1's IQ4_NL keeps its own stores.
template <fmt F, pd_layout L>
__global__ void __launch_bounds__(pd_threads) pd_convert_kernel(const pd_kargs k, uint16_t * wt) {
    constexpr int32_t   SK  = L == pd_layout::dp2 ? dp2_stage : ex1_stage;
    constexpr int32_t   QPS = L == pd_layout::dp2 ? 2 : 1;
    constexpr int32_t   GW  = group_words<F>();
    __shared__ uint32_t grid_s[F == fmt::iq3_s ? 512 : 1];
    const int32_t       tid = static_cast<int32_t>(threadIdx.x);
    const int32_t       w = tid >> 5, lane = tid & 31;
    const int32_t       qp = (lane >> 2) & 3, r = (lane & 3) | ((lane >> 2) & 4);
    const int32_t       n = static_cast<int32_t>(blockIdx.x) * pd_cols + 32 * w + 8 * qp + r;
    const int32_t       s = static_cast<int32_t>(blockIdx.y);
    if constexpr (F == fmt::iq3_s) {
        for (int32_t i = tid; i < 512; i += pd_threads) {
            grid_s[i] = __ldg(reinterpret_cast<const uint32_t *>(k.a.grids) + i);
        }
        __syncthreads();
    }
    uint4 kv = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (F == fmt::iq4_xs || F == fmt::iq4_nl) {
        kv = __ldg(reinterpret_cast<const uint4 *>(k.a.grids + 2048));
    }
    if (static_cast<int32_t>(blockIdx.x) * pd_cols + 32 * w >= k.a.n) {
        return;
    }
    constexpr bool   own_stores = L == pd_layout::ex1_down && F == fmt::iq4_nl;
    __shared__ uint2 cv_s[pd_threads / 32][own_stores ? 1 : 32 * 16];
#pragma unroll
    for (int32_t qq = 0; qq < QPS; ++qq) {
        quarter_codes qc;
        if constexpr (L == pd_layout::dp2) {
            const int32_t   step = s >> 1, q = 2 * (s & 1) + qq;
            const uint8_t * p    = k.a.w + int64_t(n >> 3) * k.row_bytes8 * (k.a.k / 256) + int64_t(step) * k.row_bytes8;
            dp2_quarter<F>(p, r, q, kv, grid_s, qc);
        } else {
            ex1_stage_codes<F>(k.a.w, n, s, kv, qc);
        }
        constexpr uint32_t bias = is_signed_code<F>() ? bias_s : bias_u;
        if constexpr (own_stores) {
            uint16_t * const row = wt + int64_t(n) * k.a.k + int64_t(s) * SK;
#pragma unroll
            for (int32_t x = 0; x < 16; x += 2) {
                uint32_t lo0, hi0, lo1, hi1;
                codes16(qc.w[x], bias, lo0, hi0);
                codes16(qc.w[x + 1], bias, lo1, hi1);
                const uint2 a = weights16(lo0, hi0, qc.f[x / GW], qc.m[x / GW]);  // K +0, +16, +1, +17
                const uint2 b = weights16(lo1, hi1, qc.f[x / GW], qc.m[x / GW]);  // K +2, +18, +3, +19
                const int32_t k0 = k_of<F, L>(qq, x, 0);
                const uint2   near = make_uint2(__byte_perm(a.x, a.y, 0x5410), __byte_perm(b.x, b.y, 0x5410));
                const uint2   far  = make_uint2(__byte_perm(a.x, a.y, 0x7632), __byte_perm(b.x, b.y, 0x7632));
                *reinterpret_cast<uint2 *>(row + k0)      = near;
                *reinterpret_cast<uint2 *>(row + k0 + 16) = far;
            }
        } else {
            uint2 * const    buf   = cv_s[w];
            const int32_t    c_own = 8 * qp + r;  // the lane's column within the warp's 32
            uint16_t * const base  = wt + (int64_t(blockIdx.x) * pd_cols + 32 * w) * k.a.k + int64_t(s) * SK;
#pragma unroll
            for (int32_t x = 0; x < 16; ++x) {
                uint32_t lo, hi;
                codes16(qc.w[x], bias, lo, hi);
                buf[cv_at(c_own, x)] = weights16(lo, hi, qc.f[x / GW], qc.m[x / GW]);
            }
            __syncwarp();
            const int32_t x = lane & 15;
#pragma unroll 4
            for (int32_t p = 0; p < 16; ++p) {
                const int32_t c = 2 * p + (lane >> 4);
                *reinterpret_cast<uint2 *>(base + int64_t(c) * k.a.k + k_of<F, L>(qq, x, 0)) = buf[cv_at(c, x)];
            }
            __syncwarp();  // the buffer is rewritten by the next quarter
        }
    }
}

// PB3: a CTA 128 x 128 of four warps, each 64 x 64 (0.125 shared operand bytes a MAC against the 64 x 32 warp
// tile's 0.1875; the K order and so every output unchanged)
constexpr int32_t pt_bn      = 128;
using pt_shape = fn::tc::shape<pt_bn, 4, 4>;
constexpr int32_t pt_threads = pt_shape::threads;
constexpr int32_t pt_smem    = pt_shape::smem;

__global__ void __launch_bounds__(pt_threads, 2) pt_kernel(const pt_kargs k) {
    extern __shared__ uint4 pt_sm[];
    const auto epi = [&](int32_t row, int32_t c0, const float (&v)[4]) { pt_epi(k, row, c0, v); };
    fn::tc::gemm_cta<pt_bn, 4, 4>(k.p, static_cast<int32_t>(blockIdx.y) * pt_shape::rows,
                                  static_cast<int32_t>(blockIdx.x) * pt_bn, reinterpret_cast<uint8_t *>(pt_sm), epi);
}

// MUP1 piece 2: a small read's tile, 64 x 64 of four warps (each 32 x 32), for at most 64 rows: a 64-row read's
// 128 x 128 tile left three quarters of its rows empty and its grids 1-64 CTAs (Load 1). Every output's K-order sum
// is the same (the m8n8k4 products of a k4 step in K order whatever the tile), so the outputs equal pt_kernel's
constexpr int32_t pt_small_bn      = 64;
using pt_small_shape = fn::tc::shape<pt_small_bn, 2, 2>;
constexpr int32_t pt_small_threads = pt_small_shape::threads;
constexpr int32_t pt_small_smem    = pt_small_shape::smem;

__global__ void __launch_bounds__(pt_small_threads, 4) pt_small_kernel(const pt_kargs k) {
    extern __shared__ uint4 pt_sm[];
    const auto epi = [&](int32_t row, int32_t c0, const float (&v)[4]) { pt_epi(k, row, c0, v); };
    fn::tc::gemm_cta<pt_small_bn, 2, 2>(k.p, static_cast<int32_t>(blockIdx.y) * pt_small_shape::rows,
                                        static_cast<int32_t>(blockIdx.x) * pt_small_bn,
                                        reinterpret_cast<uint8_t *>(pt_sm), epi);
}

// ---- APF1 Part 4: the fused route (pq_tc) -------------------------------------------------------------------------
// A CTA 64 rows x 32 columns, four warps (2 down x 2 across) of 32 x 16; quadpair q of a warp holds rows
// 8 (q & 1) .. + 7 and columns 8 (q >> 1) .. + 7 of each 16 x 16 tile, as fn/kernels/tc's core does. A stage of SK
// K values in shared memory: activations [64][SK], weights [32][SK], rows of SK halves with the 16-byte chunk c of
// row x at c ^ (x & 7), so the 8 rows (columns) of a quadpair's fragment read fall in distinct bank groups.
constexpr int32_t pq_bn      = 32;
constexpr int32_t pq_bm      = 64;
constexpr int32_t pq_threads = 128;

__host__ __device__ constexpr int32_t pq_smem(int32_t sk) {
    return 2 * (pq_bm + pq_bn) * sk * 2;
}

static_assert(pq_smem(dp2_stage) <= 48 * 1024, "two stages in the default shared memory");

struct pq_kargs {
    pt_kargs        e;        // the activation operand, rows, n, k and pd_tc's epilogue
    const uint8_t * w[3];
    int32_t         col0[4];  // source s's first column; col0[n_src] = n
    int32_t         n_src;
    int32_t         rb8;      // DP2: a tile's bytes a step
    int32_t         steps;    // stages over K
    const uint8_t * grids;
    const uint16_t *b0, *b1;  // BF16 rows (n_src 0)
    int32_t         n0, n1;
    const float *   b_scale;
};

// the byte offset of (row, k) in a stage of rows of SK halves (k a multiple of 4 for 8-byte stores, of 8 for 16)
template <int32_t SK> __device__ __forceinline__ uint32_t pq_at(int32_t row, int32_t k) {
    return static_cast<uint32_t>(row * SK * 2 + 16 * ((k >> 3) ^ (row & 7)) + 2 * (k & 7));
}

__device__ __forceinline__ void pq_sts64(uint32_t addr, const uint2 & v) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(addr), "r"(v.x), "r"(v.y));
}

// the stage's activation chunks a thread loads (all 128 threads), then stores
template <int32_t SK> struct pq_a_regs {
    static constexpr int32_t CH = SK / 8;
    static constexpr int32_t IT = pq_bm * CH / pq_threads;
    uint4                    v[IT];
};

template <int32_t SK>
__device__ __forceinline__ void pq_load_a(const pq_kargs & k, int32_t row0, int32_t s, pq_a_regs<SK> & r) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t i = 0; i < pq_a_regs<SK>::IT; ++i) {
        const int32_t ci = tid + i * pq_threads, row = ci / pq_a_regs<SK>::CH, c = ci % pq_a_regs<SK>::CH;
        r.v[i] = fn::tc::detail::load_a(k.e.p.a, k.e.p.rows, row0 + row, s * SK + 8 * c);
    }
}

template <int32_t SK> __device__ __forceinline__ void pq_store_a(uint32_t abase, const pq_a_regs<SK> & r) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t i = 0; i < pq_a_regs<SK>::IT; ++i) {
        const int32_t ci = tid + i * pq_threads, row = ci / pq_a_regs<SK>::CH, c = ci % pq_a_regs<SK>::CH;
        fn::tc::detail::sts128(abase + pq_at<SK>(row, 8 * c), r.v[i]);
    }
}

// stage s's weights of the CTA's 32 columns into bbase, each exactly pd_convert_kernel's FP16 value at its K place.
// DP2: threads 0..63, column t & 31, quarter t >> 5 of the stage (dp2_quarter); EX1 down: threads 0..31, a column's
// 64 weights (ex1_stage_codes)
template <fmt F, pd_layout L>
__device__ __forceinline__ void pq_decode(const pq_kargs & k,
                                          int32_t          colt0,
                                          int32_t          s,
                                          uint32_t         bbase,
                                          const uint4 &    kv,
                                          const uint32_t * grid_s) {
    constexpr int32_t  SK   = L == pd_layout::dp2 ? dp2_stage : ex1_stage;
    constexpr int32_t  GW   = group_words<F>();
    constexpr uint32_t bias = is_signed_code<F>() ? bias_s : bias_u;
    const int32_t      tid  = static_cast<int32_t>(threadIdx.x);
    if (tid >= (L == pd_layout::dp2 ? 2 * pq_bn : pq_bn)) {
        return;
    }
    const int32_t col = tid & (pq_bn - 1), qq = L == pd_layout::dp2 ? tid >> 5 : 0;
    const int32_t n = colt0 + col;
    quarter_codes qc;
    if constexpr (L == pd_layout::dp2) {
        const int32_t   src = (k.n_src > 1 && n >= k.col0[1]) + (k.n_src > 2 && n >= k.col0[2]);
        const int32_t   nl  = n - k.col0[src];
        const uint8_t * p   = k.w[src] + int64_t(nl >> 3) * k.rb8 * (k.e.p.k / 256) + int64_t(s >> 1) * k.rb8;
        dp2_quarter<F>(p, nl & 7, 2 * (s & 1) + qq, kv, grid_s, qc);
    } else {
        ex1_stage_codes<F>(k.w[0], n, s, kv, qc);
    }
    if constexpr (L == pd_layout::ex1_down && F == fmt::iq4_nl) {
        // pd_convert_kernel's own stores: word pair x, x + 1 holds K 4c .. 4c + 3 (near) and 4c + 16 .. 4c + 19 (far)
#pragma unroll
        for (int32_t x = 0; x < 16; x += 2) {
            uint32_t lo0, hi0, lo1, hi1;
            codes16(qc.w[x], bias, lo0, hi0);
            codes16(qc.w[x + 1], bias, lo1, hi1);
            const uint2   a    = weights16(lo0, hi0, qc.f[x / GW], qc.m[x / GW]);
            const uint2   b    = weights16(lo1, hi1, qc.f[x / GW], qc.m[x / GW]);
            const int32_t k0   = k_of<F, L>(qq, x, 0);
            const uint2   near = make_uint2(__byte_perm(a.x, a.y, 0x5410), __byte_perm(b.x, b.y, 0x5410));
            const uint2   far  = make_uint2(__byte_perm(a.x, a.y, 0x7632), __byte_perm(b.x, b.y, 0x7632));
            pq_sts64(bbase + pq_at<SK>(col, k0), near);
            pq_sts64(bbase + pq_at<SK>(col, k0 + 16), far);
        }
    } else {
#pragma unroll
        for (int32_t x = 0; x < 16; ++x) {
            uint32_t lo, hi;
            codes16(qc.w[x], bias, lo, hi);
            pq_sts64(bbase + pq_at<SK>(col, k_of<F, L>(qq, x, 0)), weights16(lo, hi, qc.f[x / GW], qc.m[x / GW]));
        }
    }
}

// the BF16 rows' stage: 32 columns x 16 chunks of 8, four a thread; loaded before the stage's products, converted
// (pd_convert_bf16's value: BF16 times the row's 2^e, rounded once) and stored after them
struct pq_b16_regs {
    uint4 v[4];
    float sc[4];
};

__device__ __forceinline__ void pq_load_b16(const pq_kargs & k, int32_t colt0, int32_t s, pq_b16_regs & r) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        const int32_t    ci = tid + i * pq_threads, col = ci >> 4, c = ci & 15;
        const int32_t    n   = colt0 + col;
        const uint16_t * row = n < k.n0        ? k.b0 + int64_t(n) * k.e.p.k :
                               n < k.n0 + k.n1 ? k.b1 + int64_t(n - k.n0) * k.e.p.k :
                                                 nullptr;
        r.v[i]               = row != nullptr ? __ldg(reinterpret_cast<const uint4 *>(row + s * dp2_stage + 8 * c)) :
                                                make_uint4(0u, 0u, 0u, 0u);
        // the prepared 2^-e, inverted exactly (a power of two in range)
        r.sc[i]              = row != nullptr ? 1.0f / __ldg(k.b_scale + n) : 0.0f;
    }
}

__device__ __forceinline__ void pq_store_b16(uint32_t bbase, const pq_b16_regs & r) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const auto    f   = [](uint32_t h) {
        return __uint_as_float(h << 16);
    };
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        const int32_t ci = tid + i * pq_threads, col = ci >> 4, c = ci & 15;
        const float   sc = r.sc[i];
        const uint4 & v  = r.v[i];
        uint4         o;
        o.x = fn::tc::pack2(f(v.x & 0xffffu) * sc, f(v.x >> 16) * sc);
        o.y = fn::tc::pack2(f(v.y & 0xffffu) * sc, f(v.y >> 16) * sc);
        o.z = fn::tc::pack2(f(v.z & 0xffffu) * sc, f(v.z >> 16) * sc);
        o.w = fn::tc::pack2(f(v.w & 0xffffu) * sc, f(v.w >> 16) * sc);
        fn::tc::detail::sts128(bbase + pq_at<dp2_stage>(col, 8 * c), o);
    }
}

// one stage's products: chunk c is k4 steps 2c (x, y) and 2c + 1 (z, w), in K order for every accumulator
template <int32_t SK> __device__ __forceinline__ void pq_mma(uint32_t abase, uint32_t bbase, float (&acc)[2][8]) {
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31, warp = static_cast<int32_t>(threadIdx.x) >> 5;
    const int32_t q = (lane >> 2) & 3, ra = (lane & 3) | ((lane >> 2) & 4);
    const int32_t ar0 = 32 * (warp & 1) + 8 * (q & 1) + ra, ar1 = ar0 + 16;
    const int32_t bc = 16 * (warp >> 1) + 8 * (q >> 1) + ra;
#pragma unroll
    for (int32_t c = 0; c < SK / 8; ++c) {
        const uint4 bf = fn::tc::detail::lds128(bbase + pq_at<SK>(bc, 8 * c));
        const uint4 a0 = fn::tc::detail::lds128(abase + pq_at<SK>(ar0, 8 * c));
        const uint4 a1 = fn::tc::detail::lds128(abase + pq_at<SK>(ar1, 8 * c));
        fn::tc::detail::mma884(acc[0], a0.x, a0.y, bf.x, bf.y);
        fn::tc::detail::mma884(acc[1], a1.x, a1.y, bf.x, bf.y);
        fn::tc::detail::mma884(acc[0], a0.z, a0.w, bf.z, bf.w);
        fn::tc::detail::mma884(acc[1], a1.z, a1.w, bf.z, bf.w);
    }
}

__device__ __forceinline__ void pq_epilogue(const pq_kargs & k, int32_t row0, int32_t colt0, const float (&acc)[2][8]) {
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31, warp = static_cast<int32_t>(threadIdx.x) >> 5;
    const int32_t q  = (lane >> 2) & 3;
    const int32_t lr = ((lane & 16) ? 4 : 0) | (lane & 1), cc = 2 * ((lane >> 1) & 1);
#pragma unroll
    for (int32_t mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int32_t rh = 0; rh < 2; ++rh) {
            const int32_t row = row0 + 32 * (warp & 1) + 16 * mi + 8 * (q & 1) + lr + 2 * rh;
            if (row < k.e.p.rows) {
                const float v[4] = { acc[mi][2 * rh], acc[mi][2 * rh + 1], acc[mi][2 * rh + 4], acc[mi][2 * rh + 5] };
                pt_epi(k.e, row, colt0 + 16 * (warp >> 1) + 8 * (q >> 1) + cc, v);
            }
        }
    }
}

template <fmt F, pd_layout L> __global__ void __launch_bounds__(pq_threads) pq_kernel(const pq_kargs k) {
    constexpr int32_t       SK = L == pd_layout::dp2 ? dp2_stage : ex1_stage;
    extern __shared__ uint4 pq_sm[];
    __shared__ uint32_t     grid_s[F == fmt::iq3_s ? 512 : 1];
    const int32_t           tid   = static_cast<int32_t>(threadIdx.x);
    const int32_t           row0  = static_cast<int32_t>(blockIdx.x) * pq_bm;
    const int32_t           colt0 = static_cast<int32_t>(blockIdx.y) * pq_bn;
    const uint32_t          s0    = static_cast<uint32_t>(__cvta_generic_to_shared(pq_sm));
    constexpr uint32_t      AB = pq_bm * SK * 2, BB = pq_bn * SK * 2;  // a stage's activation and weight bytes
    const auto              abuf = [&](int32_t b) {
        return s0 + static_cast<uint32_t>(b) * (AB + BB);
    };
    const auto bbuf = [&](int32_t b) {
        return abuf(b) + AB;
    };
    if constexpr (F == fmt::iq3_s) {
        for (int32_t i = tid; i < 512; i += pq_threads) {
            grid_s[i] = __ldg(reinterpret_cast<const uint32_t *>(k.grids) + i);
        }
        __syncthreads();
    }
    uint4 kv = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (F == fmt::iq4_xs || F == fmt::iq4_nl) {
        kv = __ldg(reinterpret_cast<const uint4 *>(k.grids + 2048));
    }
    float acc[2][8];
#pragma unroll
    for (int32_t mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            acc[mi][i] = 0.0f;
        }
    }
    pq_a_regs<SK> ar;
    pq_load_a<SK>(k, row0, 0, ar);
    pq_store_a<SK>(abuf(0), ar);
    pq_decode<F, L>(k, colt0, 0, bbuf(0), kv, grid_s);
    __syncthreads();
#pragma unroll 1
    for (int32_t s = 0; s < k.steps; ++s) {
        const int32_t b    = s & 1;
        const bool    more = s + 1 < k.steps;
        if (more) {
            pq_load_a<SK>(k, row0, s + 1, ar);
        }
        pq_mma<SK>(abuf(b), bbuf(b), acc);
        if (more) {
            pq_store_a<SK>(abuf(b ^ 1), ar);
            pq_decode<F, L>(k, colt0, s + 1, bbuf(b ^ 1), kv, grid_s);
        }
        __syncthreads();
    }
    pq_epilogue(k, row0, colt0, acc);
}

__global__ void __launch_bounds__(pq_threads) pqb_kernel(const pq_kargs k) {
    constexpr int32_t       SK = dp2_stage;
    extern __shared__ uint4 pq_sm[];
    const int32_t           row0  = static_cast<int32_t>(blockIdx.x) * pq_bm;
    const int32_t           colt0 = static_cast<int32_t>(blockIdx.y) * pq_bn;
    const uint32_t          s0    = static_cast<uint32_t>(__cvta_generic_to_shared(pq_sm));
    constexpr uint32_t      AB = pq_bm * SK * 2, BB = pq_bn * SK * 2;
    const auto              abuf = [&](int32_t b) {
        return s0 + static_cast<uint32_t>(b) * (AB + BB);
    };
    const auto bbuf = [&](int32_t b) {
        return abuf(b) + AB;
    };
    float acc[2][8];
#pragma unroll
    for (int32_t mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            acc[mi][i] = 0.0f;
        }
    }
    pq_a_regs<SK> ar;
    pq_b16_regs   br;
    pq_load_a<SK>(k, row0, 0, ar);
    pq_load_b16(k, colt0, 0, br);
    pq_store_a<SK>(abuf(0), ar);
    pq_store_b16(bbuf(0), br);
    __syncthreads();
#pragma unroll 1
    for (int32_t s = 0; s < k.steps; ++s) {
        const int32_t b    = s & 1;
        const bool    more = s + 1 < k.steps;
        if (more) {
            pq_load_a<SK>(k, row0, s + 1, ar);
            pq_load_b16(k, colt0, s + 1, br);
        }
        pq_mma<SK>(abuf(b), bbuf(b), acc);
        if (more) {
            pq_store_a<SK>(abuf(b ^ 1), ar);
            pq_store_b16(bbuf(b ^ 1), br);
        }
        __syncthreads();
    }
    pq_epilogue(k, row0, colt0, acc);
}

// pd_bf16_scales: pt_bf16_kernel's first pass alone (its maximum and exponent), a warp a row
__global__ void __launch_bounds__(256) pq_bf16_scale_kernel(const uint16_t * w, int32_t n, int32_t k, float * col_s) {
    const int32_t row  = static_cast<int32_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (row >= n) {
        return;
    }
    const uint4 * src = reinterpret_cast<const uint4 *>(w + int64_t(row) * k);
    const auto    f   = [](uint32_t h) {
        return __uint_as_float(h << 16);
    };
    float m = 0.0f;
    for (int32_t g = lane; g < k / 8; g += 32) {
        const uint4    v    = __ldg(src + g);
        const uint32_t x[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            m = fmaxf(m, fmaxf(fabsf(f(x[i] & 0xffffu)), fabsf(f(x[i] >> 16))));
        }
    }
    const int32_t e = fn::tc::range_exp(fn::tc::warp_max(m));
    if (lane == 0) {
        col_s[row] = fn::tc::pow2(-e);
    }
}

// pd_convert_bf16: a warp a row, the row read twice (its maximum, then the scaled conversion; the second read an L2 hit)
__global__ void __launch_bounds__(256) pt_bf16_kernel(const uint16_t * w, int32_t n, int32_t k, uint16_t * wt,
                                                      float * col_s) {
    const int32_t row  = static_cast<int32_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (row >= n) {
        return;
    }
    const uint4 * src = reinterpret_cast<const uint4 *>(w + int64_t(row) * k);
    uint4 *       dst = reinterpret_cast<uint4 *>(wt + int64_t(row) * k);
    const auto    f   = [](uint32_t h) {
        return __uint_as_float(h << 16);
    };
    float m = 0.0f;
    for (int32_t g = lane; g < k / 8; g += 32) {
        const uint4    v    = __ldg(src + g);
        const uint32_t x[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            m = fmaxf(m, fmaxf(fabsf(f(x[i] & 0xffffu)), fabsf(f(x[i] >> 16))));
        }
    }
    const int32_t e  = fn::tc::range_exp(fn::tc::warp_max(m));
    const float   sc = fn::tc::pow2(e);
    for (int32_t g = lane; g < k / 8; g += 32) {
        const uint4 v = __ldg(src + g);
        uint4       o;
        o.x    = fn::tc::pack2(f(v.x & 0xffffu) * sc, f(v.x >> 16) * sc);
        o.y    = fn::tc::pack2(f(v.y & 0xffffu) * sc, f(v.y >> 16) * sc);
        o.z    = fn::tc::pack2(f(v.z & 0xffffu) * sc, f(v.z >> 16) * sc);
        o.w    = fn::tc::pack2(f(v.w & 0xffffu) * sc, f(v.w >> 16) * sc);
        dst[g] = o;
    }
    if (lane == 0) {
        col_s[row] = fn::tc::pow2(-e);
    }
}

// ---- the BF16 product ----------------------------------------------------------------------------------------------

constexpr int32_t pb_tile = 64;
constexpr int32_t pb_k    = 32;

__device__ __forceinline__ auto pb_row(const pb_args & a, int32_t n) -> const uint16_t * {
    if (a.period <= 0) {
        return a.w_lo + int64_t(n) * a.k;
    }
    const int32_t m = n % a.period, h = m / a.half;
    const int64_t row = int64_t(n / a.period) * a.half + (m % a.half);
    return (h == 0 || a.w_hi == nullptr ? a.w_lo : a.w_hi) + row * a.k;
}

// TTTS2: pb_row for F32 rows (pb_args::f32)
__device__ const __forceinline__ float * pb_row32(const pb_args & a, int32_t n) {
    int64_t          row  = n;
    const uint16_t * base = a.w_lo;
    if (a.period > 0) {
        const int32_t m = n % a.period, h = m / a.half;
        row  = int64_t(n / a.period) * a.half + (m % a.half);
        base = h == 0 || a.w_hi == nullptr ? a.w_lo : a.w_hi;
    }
    return reinterpret_cast<const float *>(base) + row * a.k;
}

// TTTS3: pb_row for Q8_0 rows (pb_args::q8): the row's first block
__device__ const __forceinline__ uint8_t * pb_row_q8(const pb_args & a, int32_t n) {
    int64_t          row  = n;
    const uint16_t * base = a.w_lo;
    if (a.period > 0) {
        const int32_t m = n % a.period, h = m / a.half;
        row  = int64_t(n / a.period) * a.half + (m % a.half);
        base = h == 0 || a.w_hi == nullptr ? a.w_lo : a.w_hi;
    }
    return reinterpret_cast<const uint8_t *>(base) + row * (a.k / 32) * 34;
}

// 64 rows x 64 outputs a block, 256 threads, 4 x 4 outputs a thread; each 32-K step's products summed fresh, then
// added to the running sum (two levels: the K = 2,560 sums keep a short chain). W32: the weights widened to F32 before
// the products, F32 rows (TTTS2) or Q8_0 rows (TTTS3, a.q8: each weight d x q, exactly the F32 rows' value)
template <bool W32 = false> __global__ void __launch_bounds__(256) pb_kernel(const pb_args a) {
    __shared__ float xs[pb_k][pb_tile + 4];
    __shared__ float ws[pb_k][pb_tile + 4];
    const int32_t    tid = static_cast<int32_t>(threadIdx.x);
    const int32_t    ty = tid >> 4, tx = tid & 15;
    const int32_t    r0 = static_cast<int32_t>(blockIdx.y) * pb_tile, n0 = static_cast<int32_t>(blockIdx.x) * pb_tile;
    float            acc[4][4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    // loaders: element (row lr, k lk .. lk + 7) of the x tile and (n lr, k lk ..) of the w tile
    const int32_t    lr = tid >> 2, lk = (tid & 3) * 8;
    const bool       xr = r0 + lr < a.rows, wr = n0 + lr < a.n;
    const float *    xp = a.x + int64_t(xr ? r0 + lr : 0) * a.ldx + lk;
    const uint16_t * wp = W32 ? nullptr : pb_row(a, wr ? n0 + lr : 0) + lk;
    const float *    wq = W32 && !a.q8 ? pb_row32(a, wr ? n0 + lr : 0) + lk : nullptr;
    for (int32_t k0 = 0; k0 < a.k; k0 += pb_k) {
        {
            const float4 x0    = xr ? __ldg(reinterpret_cast<const float4 *>(xp + k0)) : make_float4(0, 0, 0, 0);
            const float4 x1    = xr ? __ldg(reinterpret_cast<const float4 *>(xp + k0) + 1) : make_float4(0, 0, 0, 0);
            const float  xv[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
            if constexpr (W32) {
                float wf[8];
                if (a.q8) {  // block k0 / 32: its d, then codes lk .. lk + 7 (2-byte aligned: four 16-bit loads)
                    const uint8_t *        blk = pb_row_q8(a, wr ? n0 + lr : 0) + (k0 / 32) * 34;
                    const unsigned short * cp  = reinterpret_cast<const unsigned short *>(blk + 2 + lk);
                    const unsigned short   dh  = wr ? __ldg(reinterpret_cast<const unsigned short *>(blk)) : 0;
                    const float            d   = __half2float(__ushort_as_half(dh));
#pragma unroll
                    for (int32_t j = 0; j < 8; ++j) {
                        const unsigned short c2 = wr ? __ldg(cp + j / 2) : static_cast<unsigned short>(0);
                        wf[j]                   = static_cast<float>(static_cast<int8_t>(c2 >> (8 * (j & 1)))) * d;
                    }
                } else {
                    const float4 * wv = reinterpret_cast<const float4 *>(wq + k0);
                    const float4   w0 = wr ? __ldg(wv) : make_float4(0, 0, 0, 0);
                    const float4   w1 = wr ? __ldg(wv + 1) : make_float4(0, 0, 0, 0);
                    wf[0] = w0.x, wf[1] = w0.y, wf[2] = w0.z, wf[3] = w0.w;
                    wf[4] = w1.x, wf[5] = w1.y, wf[6] = w1.z, wf[7] = w1.w;
                }
#pragma unroll
                for (int32_t j = 0; j < 8; ++j) {
                    xs[lk + j][lr] = xv[j];
                    ws[lk + j][lr] = wf[j];
                }
            } else {
                const uint4 wv = wr ? __ldg(reinterpret_cast<const uint4 *>(wp + k0)) : make_uint4(0, 0, 0, 0);
#pragma unroll
                for (int32_t j = 0; j < 8; ++j) {
                    xs[lk + j][lr]    = xv[j];
                    const uint32_t h2 = word(wv, j >> 1);
                    ws[lk + j][lr]    = __uint_as_float((j & 1) ? (h2 & 0xffff0000u) : (h2 << 16));
                }
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
        for (int32_t kk = 0; kk < pb_k; ++kk) {
            const float4 xa    = *reinterpret_cast<const float4 *>(&xs[kk][4 * ty]);
            const float4 wa    = *reinterpret_cast<const float4 *>(&ws[kk][4 * tx]);
            const float  xv[4] = { xa.x, xa.y, xa.z, xa.w }, wv[4] = { wa.x, wa.y, wa.z, wa.w };
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    tmp[i][j] = fmaf(xv[i], wv[j], tmp[i][j]);
                }
            }
        }
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (int32_t j = 0; j < 4; ++j) {
                acc[i][j] += tmp[i][j];
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
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            const int32_t col = n0 + 4 * tx + j;
            if (col < a.n) {
                a.out[int64_t(row) * a.ld_out + col] = acc[i][j];
            }
        }
    }
}

__global__ void pd_glu_kernel(int32_t       rows,
                              int32_t       n,
                              const float * gate,
                              const float * up,
                              int64_t       ld,
                              uint16_t *    glu,
                              int64_t       ld_glu) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(rows) * n) {
        return;
    }
    const int64_t r = i / n, c = i % n;
    const float   g = gate[r * ld + c], u = up[r * ld + c];
    glu[r * ld_glu + c] = __half_as_ushort(__float2half_rn(g / (1.0f + expf(-g)) * u));
}

// a warp a row: lane l holds experts l + 32 i; ten rounds of the warp's best (p descending, then the lower ID)
__global__ void pd_route_kernel(int32_t rows, const float * logits, const float * gate_logit, int32_t * ids,
                                float * wts, float * sg, int32_t gate_ld) {
    const int32_t w    = static_cast<int32_t>((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (w >= rows) {
        return;
    }
    constexpr int32_t per = route_experts / 32;
    float             e[per];
    float             mx = -INFINITY;
#pragma unroll
    for (int32_t i = 0; i < per; ++i) {
        e[i] = logits[int64_t(w) * route_experts + lane + 32 * i];
        mx   = fmaxf(mx, e[i]);
    }
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    float sum = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < per; ++i) {
        e[i] = expf(e[i] - mx);
        sum += e[i];
    }
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float inv = 1.0f / sum;
#pragma unroll
    for (int32_t i = 0; i < per; ++i) {
        e[i] = fmaxf(e[i] * inv, -FLT_MAX);  // a NaN sanitized (DP2's rule)
    }
    // lane k keeps the k-th pick (no indexed array: registers only)
    float   mine_p = 0.0f;
    int32_t mine_i = 0;
    float   tot    = 0.0f;
    for (int32_t k = 0; k < route_used; ++k) {
        // the lane's best, then the warp's: p descending, ties to the lower ID
        float   bp = -INFINITY;
        int32_t bi = 1 << 30;
#pragma unroll
        for (int32_t i = 0; i < per; ++i) {
            const int32_t id = lane + 32 * i;
            if (e[i] > bp || (e[i] == bp && id < bi)) {
                bp = e[i];
                bi = id;
            }
        }
#pragma unroll
        for (int32_t o = 16; o > 0; o >>= 1) {
            const float   op = __shfl_xor_sync(0xffffffffu, bp, o);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (op > bp || (op == bp && oi < bi)) {
                bp = op;
                bi = oi;
            }
        }
        if (lane == k) {
            mine_p = bp;
            mine_i = bi;
        }
        tot += bp;
#pragma unroll
        for (int32_t i = 0; i < per; ++i) {
            if (lane + 32 * i == bi) {
                e[i] = -INFINITY;  // taken
            }
        }
    }
    if (lane < route_used) {
        const float r                       = 1.0f / fmaxf(tot, 6.103515625e-5f);
        ids[int64_t(w) * route_used + lane] = mine_i;
        wts[int64_t(w) * route_used + lane] = mine_p * r;
    }
    if (lane == 0) {
        sg[w] = 1.0f / (1.0f + expf(-gate_logit[int64_t(w) * gate_ld]));
    }
}

bool al16(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

bool reject(const char * what) {
    fprintf(stderr, "pd: rejected: %s\n", what);
    return false;
}

bool pd_check(const pd_args & a, int32_t & steps, int32_t & rb8) {
    if (a.rows < 1 || a.rows > pd_max_rows || a.n < 32 || a.n % 32 != 0 || !al16(a.w) || a.out == nullptr ||
        a.x.a0 == nullptr || a.x.k_split <= 0 || a.x.k_split % 8 != 0 || a.x.ld0 % 8 != 0 ||
        (a.x.k_split < a.k && (a.x.a1 == nullptr || a.x.ld1 % 8 != 0))) {
        return reject("rows, N, pointers or the activation operand");
    }
    if (a.lay == pd_layout::dp2) {
        const bool ok = a.f == fmt::q4_k || a.f == fmt::q5_k || a.f == fmt::q6_k || a.f == fmt::iq4_xs ||
                        a.f == fmt::iq3_s || a.f == fmt::q8_0;
        if (!ok || a.k % 256 != 0 || a.k > max_k || (a.x.k_split < a.k && a.x.k_split % dp2_stage != 0)) {
            return reject("a DP2 format with K % 256 == 0");
        }
        steps = a.k / dp2_stage;
        rb8   = step_bytes(a.f, 4);
    } else {
        if ((a.f != fmt::iq4_nl && a.f != fmt::q2_0 && a.f != fmt::q8_0) || a.k != ex1_k || a.n != ex1_n ||
            a.x.k_split != a.k) {
            return reject("EX1's down layout: IQ4_NL or Q2_0, K 320, N 2560");
        }
        steps = a.k / ex1_stage;
        rb8   = 0;
    }
    if ((a.f == fmt::iq3_s || a.f == fmt::iq4_xs || a.f == fmt::iq4_nl) && !al16(a.grids)) {
        return reject("the grid table");
    }
    return true;
}

template <fmt F, pd_layout L> bool pd_launch(const pd_kargs & k, cudaStream_t s) {
    const dim3 grid((k.a.n + pd_cols - 1) / pd_cols, (k.a.rows + pd_rows - 1) / pd_rows);
    pd_kernel<F, L><<<grid, pd_threads, 0, s>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <fmt F, pd_layout L>
bool pd_codes_launch(const pd_kargs & k, float * c, float * sc, float * m, cudaStream_t s) {
    pd_codes_kernel<F, L><<<(k.a.n + pd_cols - 1) / pd_cols, pd_threads, 0, s>>>(k, c, sc, m);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <fmt F, pd_layout L> bool pd_convert_launch(const pd_kargs & k, uint16_t * wt, cudaStream_t s) {
    const dim3 grid((k.a.n + pd_cols - 1) / pd_cols, k.steps);
    pd_convert_kernel<F, L><<<grid, pd_threads, 0, s>>>(k, wt);
    return cudaPeekAtLastError() == cudaSuccess;
}

// the instantiations: one switch for every dispatch, so each (format, layout) is compiled once
template <typename Fn> bool dispatch(fmt f, pd_layout lay, Fn && fn) {
    if (lay == pd_layout::ex1_down) {
        switch (f) {
            case fmt::iq4_nl:
                return fn(std::integral_constant<fmt, fmt::iq4_nl>{},
                          std::integral_constant<pd_layout, pd_layout::ex1_down>{});
            case fmt::q2_0:
                return fn(std::integral_constant<fmt, fmt::q2_0>{},
                          std::integral_constant<pd_layout, pd_layout::ex1_down>{});
            case fmt::q8_0:
                return fn(std::integral_constant<fmt, fmt::q8_0>{},
                          std::integral_constant<pd_layout, pd_layout::ex1_down>{});
            default:
                return false;
        }
    }
    using dl = std::integral_constant<pd_layout, pd_layout::dp2>;
    switch (f) {
        case fmt::q4_k:
            return fn(std::integral_constant<fmt, fmt::q4_k>{}, dl{});
        case fmt::q5_k:
            return fn(std::integral_constant<fmt, fmt::q5_k>{}, dl{});
        case fmt::q6_k:
            return fn(std::integral_constant<fmt, fmt::q6_k>{}, dl{});
        case fmt::iq4_xs:
            return fn(std::integral_constant<fmt, fmt::iq4_xs>{}, dl{});
        case fmt::iq3_s:
            return fn(std::integral_constant<fmt, fmt::iq3_s>{}, dl{});
        case fmt::q8_0:
            return fn(std::integral_constant<fmt, fmt::q8_0>{}, dl{});
        default:
            return false;
    }
}

}  // namespace

bool pd_product(const pd_args & a, void * stream) {
    pd_kargs k = { a, 0, 0 };
    if (!pd_check(a, k.steps, k.row_bytes8) || stream == nullptr) {
        return false;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    return dispatch(a.f, a.lay,
                    [&](auto F, auto L) { return pd_launch<decltype(F)::value, decltype(L)::value>(k, s); });
}

bool pd_convert(const pd_args & a, uint16_t * wt, void * stream) {
    pd_args c = a;
    c.rows    = 1;
    c.out     = reinterpret_cast<float *>(wt);  // the check wants an output and an operand
    if (c.x.a0 == nullptr) {
        c.x.a0      = reinterpret_cast<const uint16_t *>(a.w);
        c.x.k_split = a.k;
    }
    pd_kargs k = { c, 0, 0 };
    if (!pd_check(c, k.steps, k.row_bytes8) || !al16(wt) || stream == nullptr) {
        return reject("pd_convert");
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    return dispatch(a.f, a.lay, [&](auto F, auto L) {
        return pd_convert_launch<decltype(F)::value, decltype(L)::value>(k, wt, s);
    });
}

namespace {

// pd_tc's argument checks and kernel arguments (APF1: shared with pq_tc, which reads no FP16 w)
bool pt_make(const pt_args & a, bool need_w, void * stream, pt_kargs & k) {
    if (a.rows < 1 || a.rows > pd_max_rows || a.n < 8 || a.n % 8 != 0 || a.k < fn::tc::bk || a.k % fn::tc::bk != 0 ||
        (need_w && !al16(a.w)) || !al16(a.x.a0) || a.x.ld0 % 8 != 0 || a.x.k_split <= 0 || a.x.k_split > a.k ||
        (a.x.k_split < a.k && (a.x.k_split % fn::tc::bk != 0 || !al16(a.x.a1) || a.x.ld1 % 8 != 0)) || a.n_seg < 1 ||
        a.n_seg > 3 || (a.col_scale != nullptr && (reinterpret_cast<uintptr_t>(a.col_scale) & 7u) != 0) ||
        (a.out16 != nullptr && (a.n_seg != 1 || a.col_scale != nullptr || !al16(a.out16) || a.ld16 % 8 != 0 ||
                                a.ld16 < a.n || a.err == nullptr)) ||
        stream == nullptr) {
        return reject("pd_tc: shapes, pointers or the activation operand");
    }
    int32_t start = 0;
    for (int32_t i = 0; i < a.n_seg; ++i) {
        const pt_seg & g = a.seg[i];
        if ((g.out == nullptr && a.out16 == nullptr) || (reinterpret_cast<uintptr_t>(g.out) & 7u) != 0 ||
            g.ld % 2 != 0 || g.end <= start || g.end % 8 != 0 || g.ld < g.end - start) {
            return reject("pd_tc: an output segment");
        }
        start = g.end;
    }
    if (start != a.n) {
        return reject("pd_tc: the segments must end at N");
    }
    k                = {};
    k.p.a.a0      = a.x.a0;
    k.p.a.ld0     = a.x.ld0;
    k.p.a.shift0  = a.x.shift0;
    k.p.a.a1      = a.x.a1;
    k.p.a.ld1     = a.x.ld1;
    k.p.a.shift1  = a.x.shift1;
    k.p.a.k_split = a.x.k_split;
    k.p.b            = a.w;
    k.p.ldb          = a.k;
    k.p.rows         = a.rows;
    k.p.n            = a.n;
    k.p.k            = a.k;
    for (int32_t i = 0; i < 3; ++i) {
        k.seg[i] = a.seg[i];
    }
    k.n_seg     = a.n_seg;
    k.row_scale = a.row_scale;
    k.col_scale = a.col_scale;
    k.out16 = a.out16;
    k.ld16 = a.ld16;
    k.err = a.err;
    k.err_bit = a.err_bit;
    return true;
}

}  // namespace

bool pd_tc(const pt_args & a, void * stream) {
    pt_kargs k = {};
    if (!pt_make(a, true, stream, k)) {
        return false;
    }
    if (a.rows <= pt_small_shape::rows && !a.big_tile) {  // MUP1 piece 2: a small read's tile (the same outputs)
        const dim3 grid((a.n + pt_small_bn - 1) / pt_small_bn, 1);
        pt_small_kernel<<<grid, pt_small_threads, pt_small_smem, static_cast<cudaStream_t>(stream)>>>(k);
        return cudaPeekAtLastError() == cudaSuccess;
    }
    const dim3 grid((a.n + pt_bn - 1) / pt_bn, (a.rows + pt_shape::rows - 1) / pt_shape::rows);
    pt_kernel<<<grid, pt_threads, pt_smem, static_cast<cudaStream_t>(stream)>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pd_convert_bf16(const uint16_t * w, int32_t n, int32_t k, uint16_t * wt, float * col_s, void * stream) {
    if (n < 1 || k < 8 || k % 8 != 0 || !al16(w) || !al16(wt) || col_s == nullptr || stream == nullptr) {
        return reject("pd_convert_bf16: shapes or pointers");
    }
    pt_bf16_kernel<<<(n + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(w, n, k, wt, col_s);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pd_bf16_scales(const uint16_t * w, int32_t n, int32_t k, float * col_s, void * stream) {
    if (n < 1 || k < 8 || k % 8 != 0 || !al16(w) || col_s == nullptr || stream == nullptr) {
        return reject("pd_bf16_scales: shapes or pointers");
    }
    pq_bf16_scale_kernel<<<(n + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(w, n, k, col_s);
    return cudaPeekAtLastError() == cudaSuccess;
}

namespace {

// the dynamic shared memory opt-in of an instance on the current device (IQ3_S's 2 KiB grid table takes it past
// 48 KiB): cudaFuncSetAttribute is per device and the prompt path runs on both cards, so one bit a device
template <fmt F, pd_layout L> bool pq_attr() {
    constexpr int32_t            SK = L == pd_layout::dp2 ? dp2_stage : ex1_stage;
    static std::atomic<uint32_t> done{ 0u };
    int                          dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        return false;
    }
    const uint32_t bit = 1u << dev;
    if ((done.load(std::memory_order_acquire) & bit) != 0u) {
        return true;
    }
    if (cudaFuncSetAttribute(pq_kernel<F, L>, cudaFuncAttributeMaxDynamicSharedMemorySize, pq_smem(SK)) !=
        cudaSuccess) {
        return false;
    }
    done.fetch_or(bit, std::memory_order_release);
    return true;
}

template <fmt F, pd_layout L> bool pq_launch(const pq_kargs & k, cudaStream_t s) {
    constexpr int32_t SK = L == pd_layout::dp2 ? dp2_stage : ex1_stage;
    if (!pq_attr<F, L>()) {
        return reject("pq_tc: the shared memory attribute");
    }
    const dim3 grid((k.e.p.rows + pq_bm - 1) / pq_bm, k.e.p.n / pq_bn);
    pq_kernel<F, L><<<grid, pq_threads, pq_smem(SK), s>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace

bool pq_tc_attributes() {
    bool ok = true;
    for (const pd_layout lay : { pd_layout::dp2, pd_layout::ex1_down }) {
        for (const fmt f :
             { fmt::q4_k, fmt::q5_k, fmt::q6_k, fmt::iq4_xs, fmt::iq3_s, fmt::q8_0, fmt::iq4_nl, fmt::q2_0 }) {
            bool       known = false;
            const bool set   = dispatch(f, lay, [&](auto F, auto L) {
                known = true;
                return pq_attr<decltype(F)::value, decltype(L)::value>();
            });
            ok               = ok && (!known || set);
        }
    }
    return ok;
}

bool pq_tc(const pq_args & a, void * stream) {
    pq_kargs k = {};
    // packed sources: N a multiple of 32 (whole column tiles); BF16 rows: of 8 (columns past N read zero rows and
    // pd_tc's epilogue writes none)
    if (!pt_make(a.t, false, stream, k.e) || (a.n_src > 0 && a.t.n % pq_bn != 0)) {
        return reject("pq_tc: the product (pd_tc's checks; packed sources: N a multiple of 32)");
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (a.n_src == 0) {
        if (!al16(a.b0) || a.n0 < 1 || a.n1 < 0 || (a.n1 > 0 && !al16(a.b1)) || a.n0 + a.n1 > a.t.n ||
            a.b_scale == nullptr || a.t.k % dp2_stage != 0) {
            return reject("pq_tc: the BF16 rows (K a multiple of 128) or their scales");
        }
        k.b0      = a.b0;
        k.b1      = a.b1;
        k.n0      = a.n0;
        k.n1      = a.n1;
        k.b_scale = a.b_scale;
        k.steps   = a.t.k / dp2_stage;
        const dim3 grid((a.t.rows + pq_bm - 1) / pq_bm, (a.t.n + pq_bn - 1) / pq_bn);
        pqb_kernel<<<grid, pq_threads, pq_smem(dp2_stage), s>>>(k);
        return cudaPeekAtLastError() == cudaSuccess;
    }
    if (a.n_src < 1 || a.n_src > 3 || (a.lay == pd_layout::ex1_down && a.n_src != 1)) {
        return reject("pq_tc: one to three packed sources (EX1 down: one)");
    }
    int32_t n = 0;
    for (int32_t i = 0; i < a.n_src; ++i) {
        if (a.src[i].n < pq_bn || a.src[i].n % pq_bn != 0 || !al16(a.src[i].w)) {
            return reject("pq_tc: a source's columns (a multiple of 32) or its pointer");
        }
        k.w[i]    = a.src[i].w;
        k.col0[i] = n;
        n += a.src[i].n;
    }
    k.col0[a.n_src] = n;
    k.n_src         = a.n_src;
    k.grids         = a.grids;
    // pd_check on the whole product's shape: the format, layout, K and the grid table
    pd_args c;
    c.f           = a.f;
    c.lay         = a.lay;
    c.k           = a.t.k;
    c.n           = n;
    c.rows        = a.t.rows;
    c.w           = a.src[0].w;
    c.grids       = a.grids;
    c.x           = a.t.x;
    c.out         = reinterpret_cast<float *>(1);  // the check wants an output; pq_tc writes pd_tc's
    int32_t steps = 0, rb8 = 0;
    if (n != a.t.n || !pd_check(c, steps, rb8)) {
        return reject("pq_tc: the sources' columns against N, or pd_convert's format checks");
    }
    k.rb8   = rb8;
    k.steps = steps;
    return dispatch(a.f, a.lay,
                    [&](auto F, auto L) { return pq_launch<decltype(F)::value, decltype(L)::value>(k, s); });
}

// ---- APF2: the ladder's registry (dense_ladder.h) ---------------------------------------------------------------

namespace {

// the ladder kernel source of a's weights, or count when the registry does not take them
pk::pl_src pl_source_of(const pq_args & a) {
    using pk::pl_src;
    if (a.n_src == 0) {
        return pl_src::bf16;
    }
    if (a.lay == pd_layout::ex1_down) {
        return a.f == fmt::q8_0 ? pl_src::q8_0_ex1 :
               a.f == fmt::iq4_nl ? pl_src::iq4_nl_ex1 :
               a.f == fmt::q2_0   ? pl_src::q2_0_ex1 :
                                    pl_src::count;
    }
    switch (a.f) {
        case fmt::q8_0:
            return pl_src::q8_0;
        case fmt::q4_k:
            return pl_src::q4_k;
        case fmt::q5_k:
            return pl_src::q5_k;
        case fmt::q6_k:
            return pl_src::q6_k;
        case fmt::iq4_xs:
            return pl_src::iq4_xs;
        case fmt::iq3_s:
            return pl_src::iq3_s;
        default:
            return pl_src::count;
    }
}

bool pl_q4_source(pk::pl_src s) {
    return s == pk::pl_src::bf16 || s == pk::pl_src::q8_0 || s == pk::pl_src::q8_0_ex1;
}

}  // namespace

bool pl_takes(const pq_args & a) {
    return pl_source_of(a) != pk::pl_src::count;
}

bool pl_tc(const pq_args & a, int32_t t, int32_t splits, const pl_ws & ws, void * stream) {
    pk::pl_kargs k = {};
    const pk::pl_src src = pl_source_of(a);
    if (src == pk::pl_src::count || t < 0 || t >= pl_n_tiles || !pt_make(a.t, false, stream, k.e) ||
        (a.n_src > 0 && a.t.n % pq_bn != 0)) {
        return reject("pl_tc: the sources, the tile or the product (pd_tc's checks; packed sources: N a multiple of 32)");
    }
    const int32_t steps = a.t.k / fn::tc::bk;
    if (splits < 1 || splits > pl_max_splits || splits > steps) {
        return reject("pl_tc: the split count (1 .. min(16, K / 32))");
    }
    if (splits > 1 && (ws.part == nullptr || ws.count == nullptr ||
                       lad_split_floats(a.t.rows, a.t.n, t, splits) > ws.part_floats ||
                       lad_tile_count(a.t.rows, a.t.n, t) > ws.count_len)) {
        return reject("pl_tc: the split workspace");
    }
    k.splits     = splits;
    k.ws.part    = ws.part;
    k.ws.count   = ws.count;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (a.n_src == 0) {
        if (!al16(a.b0) || a.n0 < 1 || a.n1 < 0 || (a.n1 > 0 && !al16(a.b1)) || a.n0 + a.n1 > a.t.n ||
            a.b_scale == nullptr) {
            return reject("pl_tc: the BF16 rows or their scales");
        }
        k.b0      = a.b0;
        k.b1      = a.b1;
        k.n0      = a.n0;
        k.n1      = a.n1;
        k.b_scale = a.b_scale;
        return pk::pl_launch_q4(k, src, t, s);
    }
    if (a.n_src < 1 || a.n_src > 3 || (a.lay == pd_layout::ex1_down && a.n_src != 1)) {
        return reject("pl_tc: one to three packed sources (EX1 down: one)");
    }
    int32_t n = 0;
    for (int32_t i = 0; i < a.n_src; ++i) {
        if (a.src[i].n < pq_bn || a.src[i].n % pq_bn != 0 || !al16(a.src[i].w)) {
            return reject("pl_tc: a source's columns (a multiple of 32) or its pointer");
        }
        k.w[i]    = a.src[i].w;
        k.col0[i] = n;
        n += a.src[i].n;
    }
    k.col0[a.n_src] = n;
    k.n_src         = a.n_src;
    k.grids         = a.grids;
    pd_args c;
    c.f            = a.f;
    c.lay          = a.lay;
    c.k            = a.t.k;
    c.n            = n;
    c.rows         = a.t.rows;
    c.w            = a.src[0].w;
    c.grids        = a.grids;
    c.x            = a.t.x;
    c.out          = reinterpret_cast<float *>(1);  // the check wants an output; pl_tc writes pd_tc's
    int32_t dsteps = 0, rb8 = 0;
    if (n != a.t.n || !pd_check(c, dsteps, rb8)) {
        return reject("pl_tc: the sources' columns against N, or pd_convert's format checks");
    }
    k.rb8 = rb8;
    k.kb  = a.t.k / 256;
    return pl_q4_source(src) ? pk::pl_launch_q4(k, src, t, s) : pk::pl_launch_iq(k, src, t, s);
}

bool pl_attributes() {
    return pk::pl_attr_q4() && pk::pl_attr_iq();
}

bool pl_resources(const pq_args & a, int32_t t, pd_resources & r) {
    const pk::pl_src src = pl_source_of(a);
    if (src == pk::pl_src::count || t < 0 || t >= pl_n_tiles) {
        return false;
    }
    return pl_q4_source(src) ? pk::pl_res_q4(src, t, r) : pk::pl_res_iq(src, t, r);
}

bool pd_codes(fmt             f,
              pd_layout       lay,
              int32_t         k_,
              int32_t         n,
              const uint8_t * w,
              const uint8_t * grids,
              float *         code,
              float *         scale,
              float *         mins,
              void *          stream) {
    pd_args a;
    a.f         = f;
    a.lay       = lay;
    a.k         = k_;
    a.n         = n;
    a.rows      = 1;
    a.w         = w;
    a.grids     = grids;
    a.x.a0      = reinterpret_cast<const uint16_t *>(w);  // unused: the check wants an operand
    a.x.k_split = k_;
    a.out       = code;
    pd_kargs k  = { a, 0, 0 };
    if (!pd_check(a, k.steps, k.row_bytes8) || code == nullptr || scale == nullptr || mins == nullptr ||
        stream == nullptr) {
        return false;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    return dispatch(f, lay, [&](auto F, auto L) {
        return pd_codes_launch<decltype(F)::value, decltype(L)::value>(k, code, scale, mins, s);
    });
}

bool pd_bf16(const pb_args & a, void * stream) {
    if (a.rows < 1 || a.n < 1 || a.k < pb_k || a.k % pb_k != 0 || !al16(a.w_lo) ||
        (a.w_hi != nullptr && !al16(a.w_hi)) || !al16(a.x) || a.ldx % 4 != 0 || a.out == nullptr || stream == nullptr ||
        (a.period > 0 && (a.half <= 0 || a.period % a.half != 0)) || (a.q8 && a.f32)) {
        return reject("pd_bf16: shapes or pointers");
    }
    const dim3 grid((a.n + pb_tile - 1) / pb_tile, (a.rows + pb_tile - 1) / pb_tile);
    if (a.f32 || a.q8) {  // TTTS3: the widened instance reads the Q8_0 rows too
        pb_kernel<true><<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
    } else {
        pb_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pd_glu(int32_t       rows,
            int32_t       n,
            const float * gate,
            const float * up,
            int64_t       ld,
            uint16_t *    glu,
            int64_t       ld_glu,
            void *        stream) {
    if (rows < 1 || n < 1 || gate == nullptr || up == nullptr || glu == nullptr || stream == nullptr) {
        return reject("pd_glu: shapes or pointers");
    }
    const int64_t total = int64_t(rows) * n;
    pd_glu_kernel<<<static_cast<uint32_t>((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        rows, n, gate, up, ld, glu, ld_glu);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pd_route(int32_t rows, const float * logits, const float * gate_logit, int32_t * ids, float * wts, float * sg,
              void * stream, int32_t gate_ld) {
    if (rows < 1 || logits == nullptr || gate_logit == nullptr || ids == nullptr || wts == nullptr || sg == nullptr ||
        gate_ld < 1 || stream == nullptr) {
        return reject("pd_route: shapes or pointers");
    }
    pd_route_kernel<<<static_cast<uint32_t>((int64_t(rows) * 32 + 255) / 256), 256, 0,
                      static_cast<cudaStream_t>(stream)>>>(rows, logits, gate_logit, ids, wts, sg, gate_ld);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pd_kernel_resources(fmt f, pd_layout lay, pd_resources & r) {
    cudaFuncAttributes at   = {};
    int32_t            ctas = 0;
    if (f == fmt::bf16) {
        if (cudaFuncGetAttributes(&at, pb_kernel<>) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, pb_kernel<>, 256, 0) != cudaSuccess) {
            return false;
        }
    } else {
        const bool ok = dispatch(f, lay, [&](auto F, auto L) {
            void (*kern)(pd_kargs) = pd_kernel<decltype(F)::value, decltype(L)::value>;
            return cudaFuncGetAttributes(&at, kern) == cudaSuccess &&
                   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kern, pd_threads, 0) == cudaSuccess;
        });
        if (!ok) {
            return false;
        }
    }
    r.regs         = at.numRegs;
    r.local_bytes  = static_cast<int32_t>(at.localSizeBytes);
    r.shared_bytes = static_cast<int32_t>(at.sharedSizeBytes);
    r.ctas_per_sm  = ctas;
    return true;
}

bool pd_tc_resources(pd_resources & r) {
    cudaFuncAttributes at   = {};
    int32_t            ctas = 0;
    if (cudaFuncGetAttributes(&at, pt_kernel) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, pt_kernel, pt_threads, pt_smem) != cudaSuccess) {
        return false;
    }
    r.regs         = at.numRegs;
    r.local_bytes  = static_cast<int32_t>(at.localSizeBytes);
    r.shared_bytes = static_cast<int32_t>(at.sharedSizeBytes) + pt_smem;
    r.ctas_per_sm  = ctas;
    return true;
}

bool pq_tc_resources(bool bf16, fmt f, pd_layout lay, pd_resources & r) {
    cudaFuncAttributes at   = {};
    int32_t            ctas = 0;
    if (bf16) {
        if (cudaFuncGetAttributes(&at, pqb_kernel) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, pqb_kernel, pq_threads, pq_smem(dp2_stage)) !=
                cudaSuccess) {
            return false;
        }
    } else {
        const bool ok = dispatch(f, lay, [&](auto F, auto L) {
            void (*kern)(pq_kargs) = pq_kernel<decltype(F)::value, decltype(L)::value>;
            const int32_t sk       = decltype(L)::value == pd_layout::dp2 ? dp2_stage : ex1_stage;
            return cudaFuncGetAttributes(&at, kern) == cudaSuccess &&
                   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kern, pq_threads, pq_smem(sk)) == cudaSuccess;
        });
        if (!ok) {
            return false;
        }
    }
    r.regs        = at.numRegs;
    r.local_bytes = static_cast<int32_t>(at.localSizeBytes);
    r.shared_bytes =
        static_cast<int32_t>(at.sharedSizeBytes) + pq_smem(lay == pd_layout::dp2 || bf16 ? dp2_stage : ex1_stage);
    r.ctas_per_sm = ctas;
    return true;
}

bool pd_convert_resources(fmt f, pd_layout lay, pd_resources & r) {
    cudaFuncAttributes at   = {};
    int32_t            ctas = 0;
    const bool         ok   = dispatch(f, lay, [&](auto F, auto L) {
        void (*kern)(pd_kargs, uint16_t *) = pd_convert_kernel<decltype(F)::value, decltype(L)::value>;
        return cudaFuncGetAttributes(&at, kern) == cudaSuccess &&
               cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kern, pd_threads, 0) == cudaSuccess;
    });
    if (!ok) {
        return false;
    }
    r.regs         = at.numRegs;
    r.local_bytes  = static_cast<int32_t>(at.localSizeBytes);
    r.shared_bytes = static_cast<int32_t>(at.sharedSizeBytes);
    r.ctas_per_sm  = ctas;
    return true;
}

}  // namespace fn::dense
