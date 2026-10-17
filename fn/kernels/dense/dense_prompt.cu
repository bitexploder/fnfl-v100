#include "fn/kernels/dense/dense_dec.cuh"
#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/tc/tc_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace fn::dense {
namespace {

constexpr int32_t pd_threads = 256;
constexpr int32_t pd_cols    = 256;  // columns a block: 8 warps x 32
constexpr int32_t pd_rows =
    64;  // rows a panel (8 tiles of 8; one CTA an SM: the unpack and the tiles need about 150-200 registers)
constexpr int32_t pd_nt     = pd_rows / 8;
constexpr int32_t dp2_stage = 128;   // K a stage (two quarters of a 256-weight step)
constexpr int32_t ex1_stage = 64;    // K a stage of EX1's down layout (one Q2_0 block, two IQ4_NL blocks)
constexpr int32_t ex1_n     = 2560;  // the shared down's outputs a card
constexpr int32_t ex1_k     = 320;   // its inputs (the card's channels)

// ---- small helpers -----------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t hsub2(uint32_t a, uint32_t b) {
    const half2 r = __hsub2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r);  // reinterpret: half2 is the word's two halves
}

// D += A B: one m8n8k4 product per quadpair, FP16 inputs, F32 accumulators (HMMA.884.F32)
__device__ __forceinline__ void mma884(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// an A fragment (8 bytes) from shared memory, asm volatile so it stays beside its product: ptxas otherwise reads a
// group's fragments of every tile ahead of the products and runs out of registers
__device__ __forceinline__ uint2 lds64(const uint8_t * p) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
    return v;
}

// the C element i of a lane: its row and column in the quadpair's 8 x 8 block (m8n8k4, F32 accumulators)
__device__ __forceinline__ int32_t c_row(int32_t lane, int32_t i) {
    return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}

__device__ __forceinline__ int32_t c_col(int32_t lane, int32_t i) {
    return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}

// the lane holding column col's B fragment in the lane's quadpair (row r of m8n8k4's B is lane (r & 3) | (r & 4) << 2)
__device__ __forceinline__ int32_t col_lane(int32_t lane, int32_t col) {
    return (col & 3) | (lane & 12) | ((col & 4) << 2);
}

// A code word (4 codes as bytes) as the two FP16 pairs of one k4 B fragment: bitcast(0x6400 | byte) - bias, exact
__device__ __forceinline__ void codes16(uint32_t w, uint32_t bias2, uint32_t & lo, uint32_t & hi) {
    lo = hsub2(__byte_perm(w, 0x64646464u, 0x5140u), bias2);
    hi = hsub2(__byte_perm(w, 0x64646464u, 0x5342u), bias2);
}

constexpr uint32_t bias_u = 0x64006400u;  // 1024: unsigned codes
constexpr uint32_t bias_s = 0x64806480u;  // 1152: signed codes biased by 128

// ---- the formats ----------------------------------------------------------------------------------------------------
// fmt_dec<F, L>: per lane, its column's codes for one stage as 32-bit words (4 codes each, in the kernel's K order),
// the groups' scales and mins. DP2: a stage is quarters 2 h, 2 h + 1 of a 256-weight step (16 words each), decoded
// by DP2's own device code (dense_dec.cuh). EX1 down: a stage is 64 weights.

// a stage's decoded words for one quarter (DP2) and its groups
struct quarter_codes {
    uint32_t w[16];
    float    f[4], m[4];  // per group: DP2 non-Q6_K groups 0, 1 (words 8 j ..); Q6_K groups 0..3 (words 4 i ..)
};

template <fmt F> __host__ __device__ constexpr bool is_signed_code() {
    return F == fmt::iq4_xs || F == fmt::iq3_s || F == fmt::q8_0 || F == fmt::q6_k || F == fmt::iq4_nl ||
           F == fmt::q2_0;
}

template <fmt F> __host__ __device__ constexpr bool has_min() {
    return F == fmt::q4_k || F == fmt::q5_k;
}

// words a group (Q6_K: 16 weights)
template <fmt F> __host__ __device__ constexpr int32_t group_words() {
    return F == fmt::q6_k ? 4 : 8;
}

// DP2: decode quarter q of the lane's column (r: its row in the 8-row tile) at step base p
template <fmt F>
__device__ __forceinline__ void dp2_quarter(const uint8_t *  p,
                                            int32_t          r,
                                            int32_t          q,
                                            const uint4 &    kv,
                                            const uint32_t * grid,
                                            quarter_codes &  o) {
    raw<F>       v;
    const cursor c = make_cursor<F>(p, r, q);
    load_full<F, 0>(v, c);
    dec64 d;
    decode<F>(v, q, kv, grid, d);
    if constexpr (F == fmt::q6_k) {
        // codes q (0..63) as the signed q - 32: byte + 96 then the signed bias (no byte carries: 63 + 96 < 256)
#pragma unroll
        for (int32_t i = 0; i < 16; ++i) {
            o.w[i] = d.w[i] + 0x60606060u;
        }
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            o.f[i] = d.d * static_cast<float>(d.sc[i]);
            o.m[i] = 0.0f;
        }
    } else {
#pragma unroll
        for (int32_t i = 0; i < 16; ++i) {
            o.w[i] = is_signed_code<F>() ? d.w[i] ^ 0x80808080u : d.w[i];
        }
        o.f[0] = d.f0;
        o.f[1] = d.f1;
        o.m[0] = has_min<F>() ? d.m0 : 0.0f;
        o.m[1] = has_min<F>() ? d.m1 : 0.0f;
        o.f[2] = o.f[3] = o.m[2] = o.m[3] = 0.0f;
    }
}

// EX1 down: the lane's column n at stage s (64 weights): 16 words, groups 0 and 1 (32 weights each)
template <fmt F>
__device__ __forceinline__ void ex1_stage_codes(const uint8_t * w,
                                                int32_t         n,
                                                int32_t         s,
                                                const uint4 &   kv,
                                                quarter_codes & o) {
    if constexpr (F == fmt::iq4_nl) {
        constexpr int64_t d_plane = int64_t(10) * ex1_n * 16;
        const uint32_t    dd = __ldg(reinterpret_cast<const uint32_t *>(w + d_plane + (int64_t(s) * ex1_n + n) * 4));
#pragma unroll
        for (int32_t bb = 0; bb < 2; ++bb) {
            const uint4 qs = __ldg(reinterpret_cast<const uint4 *>(w + (int64_t(2 * s + bb) * ex1_n + n) * 16));
#pragma unroll
            for (int32_t c = 0; c < 4; ++c) {
                uint32_t lo, hi;
                iq4_lookup(word(qs, c), kv, lo, hi);
                o.w[8 * bb + 2 * c]     = lo ^ 0x80808080u;
                o.w[8 * bb + 2 * c + 1] = hi ^ 0x80808080u;
            }
            o.f[bb] = __half2float(__ushort_as_half(static_cast<unsigned short>((dd >> (16 * bb)) & 0xffffu)));
            o.m[bb] = 0.0f;
        }
    } else {
        static_assert(F == fmt::q2_0, "EX1 down: IQ4_NL or Q2_0");
        constexpr int64_t d_plane = int64_t(5) * ex1_n * 16;
        const uint4       qs      = __ldg(reinterpret_cast<const uint4 *>(w + (int64_t(s) * ex1_n + n) * 16));
        const float       d       = __half2float(__ushort_as_half(
            __ldg(reinterpret_cast<const unsigned short *>(w + d_plane + (int64_t(s) * ex1_n + n) * 2))));
#pragma unroll
        for (int32_t x = 0; x < 16; ++x) {
            // byte x holds weights 4x .. 4x + 3 in its 2-bit fields: each to its own byte, then q - 1 biased by 128
            const uint32_t b  = (word(qs, x >> 2) >> (8 * (x & 3))) & 0xffu;
            const uint32_t sp = (b & 3u) | ((b & 0xcu) << 6) | ((b & 0x30u) << 12) | ((b & 0xc0u) << 18);
            o.w[x]            = sp + 0x7f7f7f7fu;
        }
        o.f[0] = o.f[1] = d;
        o.m[0] = o.m[1] = 0.0f;
    }
    o.f[2] = o.f[3] = o.m[2] = o.m[3] = 0.0f;
}

// The original K (within its stage) of byte i of word x: DP2 non-Q6_K words are in K order (quarter qq's word x at
// 64 qq + 4 x); Q6_K's lane q = 2 n + h covers 16h + 32i + l' of half n (its word 4i + m: l' = 4m + byte); EX1's
// IQ4_NL word 2c (lo) / 2c + 1 (hi) of block bb holds nibbles in GGUF order (byte j: weight j low, 16 + j high).
// stage_a places that value at 64 qq + 4 x + i (EX1: 4 x + i), so a word's four values are one 8-byte A fragment.
template <fmt F, pd_layout L> __device__ __forceinline__ int32_t k_of(int32_t qq, int32_t x, int32_t i) {
    if constexpr (L == pd_layout::ex1_down) {
        if constexpr (F == fmt::iq4_nl) {
            const int32_t bb = x >> 3, xx = x & 7, c = xx >> 1, h = xx & 1;
            return 32 * bb + 4 * c + 2 * h + (i >> 1) + 16 * (i & 1);
        } else {
            return 4 * x + i;
        }
    } else if constexpr (F == fmt::q6_k) {
        const int32_t ii = x >> 2, m = x & 3;
        return 16 * qq + 32 * ii + 4 * m + i;
    } else {
        return 64 * qq + 4 * x + i;
    }
}

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

// four FP16 codes (lo, hi: k4 B fragment halves) to the decoder's F32 weights f x code - m, each rounded to FP16
__device__ __forceinline__ uint2 weights16(uint32_t lo, uint32_t hi, float f, float m) {
    const uint32_t h[2] = { lo, hi };
    uint32_t       o[2];
#pragma unroll
    for (int32_t j = 0; j < 2; ++j) {
        const float c0 = __half2float(__ushort_as_half(static_cast<unsigned short>(h[j] & 0xffffu)));
        const float c1 = __half2float(__ushort_as_half(static_cast<unsigned short>(h[j] >> 16)));
        const float v0 = __fsub_rn(__fmul_rn(f, c0), m), v1 = __fsub_rn(__fmul_rn(f, c1), m);
        o[j]           = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(v0))) |
               (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(v1))) << 16);
    }
    return make_uint2(o[0], o[1]);
}

// a CTA per (256 columns, stage): lane (qp, r) decodes column n's stage (pd_codes_kernel's walk) and writes its
// weights in K order, 8 bytes a store (a word's 4 values are consecutive in K except IQ4_NL's, whose word pair
// 2c, 2c + 1 holds K 4c .. 4c + 3 and 4c + 16 .. 4c + 19 of its block)
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
    uint16_t * row = wt + int64_t(n) * k.a.k + int64_t(s) * SK;
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
        if constexpr (L == pd_layout::ex1_down && F == fmt::iq4_nl) {
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
#pragma unroll
            for (int32_t x = 0; x < 16; ++x) {
                uint32_t lo, hi;
                codes16(qc.w[x], bias, lo, hi);
                *reinterpret_cast<uint2 *>(row + k_of<F, L>(qq, x, 0)) = weights16(lo, hi, qc.f[x / GW], qc.m[x / GW]);
            }
        }
    }
}

struct pt_kargs {
    fn::tc::problem    p;
    pt_seg             seg[3];
    int32_t            n_seg;
    const float *      row_scale;
    const float *      col_scale;
    uint16_t * out16;
    int64_t ld16;
    uint32_t * err;
    uint32_t err_bit;
};

// PB3: a CTA 128 x 128 of four warps, each 64 x 64 (0.125 shared operand bytes a MAC against the 64 x 32 warp
// tile's 0.1875; the K order and so every output unchanged)
constexpr int32_t pt_bn      = 128;
using pt_shape = fn::tc::shape<pt_bn, 4, 4>;
constexpr int32_t pt_threads = pt_shape::threads;
constexpr int32_t pt_smem    = pt_shape::smem;

__global__ void __launch_bounds__(pt_threads, 2) pt_kernel(const pt_kargs k) {
    extern __shared__ uint4 pt_sm[];
    const auto epi = [&](int32_t row, int32_t c0, const float (&v)[4]) {
        if (c0 >= k.p.n) {
            return;
        }
        // the segment by selects (a runtime index into the parameter array would copy it to the stack)
        const bool    s1    = k.n_seg > 1 && c0 >= k.seg[0].end;
        const bool    s2    = k.n_seg > 2 && c0 >= k.seg[1].end;
        float *       base  = s2 ? k.seg[2].out : s1 ? k.seg[1].out : k.seg[0].out;
        const int64_t ld    = s2 ? k.seg[2].ld : s1 ? k.seg[1].ld : k.seg[0].ld;
        const int32_t start = s2 ? k.seg[1].end : s1 ? k.seg[0].end : 0;
        const float   sc    = k.row_scale != nullptr ? __ldg(k.row_scale + row) : 1.0f;
        if (k.out16 != nullptr) {
            const float f[4] = {v[0] * sc, v[1] * sc, v[2] * sc, v[3] * sc};
            if (!(fabsf(f[0]) <= 65504.0f) || !(fabsf(f[1]) <= 65504.0f) || !(fabsf(f[2]) <= 65504.0f) ||
                !(fabsf(f[3]) <= 65504.0f)) {
                atomicOr(k.err, k.err_bit);
            }
            uint16_t * o16 = k.out16 + int64_t(row) * k.ld16 + c0;
            *reinterpret_cast<uint32_t *>(o16) = fn::tc::pack2(f[0], f[1]);
            *reinterpret_cast<uint32_t *>(o16 + 4) = fn::tc::pack2(f[2], f[3]);
            return;
        }
        float *       out   = base + int64_t(row) * ld + (c0 - start);
        if (k.col_scale != nullptr) {
            const float2 c01 = __ldg(reinterpret_cast<const float2 *>(k.col_scale + c0));
            const float2 c45 = __ldg(reinterpret_cast<const float2 *>(k.col_scale + c0 + 4));
            *reinterpret_cast<float2 *>(out)     = make_float2(v[0] * c01.x * sc, v[1] * c01.y * sc);
            *reinterpret_cast<float2 *>(out + 4) = make_float2(v[2] * c45.x * sc, v[3] * c45.y * sc);
            return;
        }
        *reinterpret_cast<float2 *>(out)     = make_float2(v[0] * sc, v[1] * sc);
        *reinterpret_cast<float2 *>(out + 4) = make_float2(v[2] * sc, v[3] * sc);
    };
    fn::tc::gemm_cta<pt_bn, 4, 4>(k.p, static_cast<int32_t>(blockIdx.y) * pt_shape::rows,
                                  static_cast<int32_t>(blockIdx.x) * pt_bn, reinterpret_cast<uint8_t *>(pt_sm), epi);
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

// 64 rows x 64 outputs a block, 256 threads, 4 x 4 outputs a thread; each 32-K step's products summed fresh, then
// added to the running sum (two levels: the K = 2,560 sums keep a short chain)
__global__ void __launch_bounds__(256) pb_kernel(const pb_args a) {
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
    const uint16_t * wp = pb_row(a, wr ? n0 + lr : 0) + lk;
    for (int32_t k0 = 0; k0 < a.k; k0 += pb_k) {
        {
            const float4 x0    = xr ? __ldg(reinterpret_cast<const float4 *>(xp + k0)) : make_float4(0, 0, 0, 0);
            const float4 x1    = xr ? __ldg(reinterpret_cast<const float4 *>(xp + k0) + 1) : make_float4(0, 0, 0, 0);
            const uint4  wv    = wr ? __ldg(reinterpret_cast<const uint4 *>(wp + k0)) : make_uint4(0, 0, 0, 0);
            const float  xv[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
#pragma unroll
            for (int32_t j = 0; j < 8; ++j) {
                xs[lk + j][lr]    = xv[j];
                const uint32_t h2 = word(wv, j >> 1);
                ws[lk + j][lr]    = __uint_as_float((j & 1) ? (h2 & 0xffff0000u) : (h2 << 16));
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
        if ((a.f != fmt::iq4_nl && a.f != fmt::q2_0) || a.k != ex1_k || a.n != ex1_n || a.x.k_split != a.k) {
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

bool pd_tc(const pt_args & a, void * stream) {
    if (a.rows < 1 || a.rows > pd_max_rows || a.n < 8 || a.n % 8 != 0 || a.k < fn::tc::bk || a.k % fn::tc::bk != 0 ||
        !al16(a.w) || !al16(a.x.a0) || a.x.ld0 % 8 != 0 || a.x.k_split <= 0 || a.x.k_split > a.k ||
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
    pt_kargs k = {};
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
        (a.period > 0 && (a.half <= 0 || a.period % a.half != 0))) {
        return reject("pd_bf16: shapes or pointers");
    }
    const dim3 grid((a.n + pb_tile - 1) / pb_tile, (a.rows + pb_tile - 1) / pb_tile);
    pb_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
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
        if (cudaFuncGetAttributes(&at, pb_kernel) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, pb_kernel, 256, 0) != cudaSuccess) {
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
