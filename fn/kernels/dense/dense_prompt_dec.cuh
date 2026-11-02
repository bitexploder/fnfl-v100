#pragma once

// The prompt path's dense decode helpers (DP2's and EX1 down's per-lane decodes into FP16 code words, their scales,
// each weight's FP16 value) and pd_tc's kernel arguments and epilogue (device only): moved unchanged from
// dense_prompt.cu by APF2, so the ladder's kernels (dense_ladder.cuh) form the same FP16 weights and run the same
// epilogue as pd_convert + pd_tc and pq_tc.

#include "fn/kernels/dense/dense_dec.cuh"
#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/tc/tc_gemm.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace fn::dense::pdc {

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
    } else if constexpr (F == fmt::q8_0) {
        // pack_down's Q8_0 (DR1): stage s's four 16-weight chunks at ((4 s + c) D + n) 16 in K order, the two blocks'
        // FP16 scales at the d plane's (s D + n) 4; signed codes biased by 128
        constexpr int64_t d_plane = int64_t(20) * ex1_n * 16;
        const uint32_t    dd = __ldg(reinterpret_cast<const uint32_t *>(w + d_plane + (int64_t(s) * ex1_n + n) * 4));
#pragma unroll
        for (int32_t c = 0; c < 4; ++c) {
            const uint4 qs = __ldg(reinterpret_cast<const uint4 *>(w + (int64_t(4 * s + c) * ex1_n + n) * 16));
#pragma unroll
            for (int32_t x = 0; x < 4; ++x) {
                o.w[4 * c + x] = word(qs, x) ^ 0x80808080u;
            }
        }
#pragma unroll
        for (int32_t bb = 0; bb < 2; ++bb) {
            o.f[bb] = __half2float(__ushort_as_half(static_cast<unsigned short>((dd >> (16 * bb)) & 0xffffu)));
            o.m[bb] = 0.0f;
        }
    } else {
        static_assert(F == fmt::q2_0, "EX1 down: IQ4_NL, Q8_0 or Q2_0");
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

// a tile's outputs at (row, c0 .. c0 + 7 step 4: v[0], v[1] at c0, c0 + 1; v[2], v[3] at c0 + 4, c0 + 5): the segment,
// the optional row and column scales, F32 or (out16) FP16. MUP1 piece 2: shared by both tiles
__device__ __forceinline__ void pt_epi(const pt_kargs & k, int32_t row, int32_t c0, const float (&v)[4]) {
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
}

}  // namespace fn::dense::pdc
