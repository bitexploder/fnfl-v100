#pragma once

// DP2's per-format weight loads, decode and dot products of one lane's 64 weights (a quarter of a 256-weight step of
// an 8-row tile), and its staged activations: moved here from dense.cu unchanged so the experts' launch (EX2: the
// shared expert's units) runs the same device code on the same packed layout (fn::dense::pack), as iq3s.cuh does for
// the token embedding. dense.h describes the layout and the numerical contract.

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/iq3s.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace fn::dense {

constexpr int32_t tile_q = 8;   // rows of a quantized tile (4 lanes a row)
constexpr int32_t step_k = 256; // weights of a row per step

// ---- packed layout ----------------------------------------------------------------------------------------------
// A step of a tile (8 rows x 256 weights) holds the GGUF bytes of the 8 rows' superblocks (Q8_0, IQ4_NL: 8 blocks of
// 32) in segments. A lane segment of z bytes is [r][q] (lane li = r nq + q owns bytes li z); a row segment is [r]. The
// four lanes of a row cover its 256 weights in quarters q of 64 (two 32-weight sub-blocks, 2q and 2q + 1), except Q6_K,
// where lane q = 2n + h covers l = 16h .. 16h + 15 of half n of the superblock and so weights 128n + 16h + 32i + l'.
//   Q4_K   (144): qs[32q, +16) L16 | qs[32q + 16, +16) L16 | {d, dmin, scales[12]} R16                  = 1152
//   Q5_K   (176): qs halves L16 L16 | qh[0, 16) R16 | qh[16, 32) R16 | {d, dmin, scales} R16              = 1408
//   Q6_K   (210): ql[64n + 16h, +16) L16 | ql[64n + 32 + 16h, +16) L16 | qh[32n + 16h, +16) L16 | scales R16 |
//                 d R2                                                                                 = 1680
//   IQ4_XS (136): qs[32q, +16) L16 | qs[32q + 16, +16) L16 | {d, scales_h, scales_l} R8                 = 1088
//                 (IQ4_XS, IQ4_NL: each 16-byte qs chunk's nibbles reordered to K order, iq4_chunk)
//   IQ3_S  (110): qs[16q, +16) L16 | signs[8q, +8) L8 | qh[2q, +2) L2 | scales[q] L1 | d R2             = 880
//   Q8_0   (8 x 34): qs of blocks 2q, 2q + 1 as four L16 | {d 2q, d 2q + 1} L4                          = 544 nq
//   IQ4_NL (8 x 18): qs of blocks 2q, 2q + 1 as two L16 | {d 2q, d 2q + 1} L4                           = 288 nq
//   Q2_0   (4 x 18): qs of block q as L16 (2-bit fields transposed, q2_chunk) | d of block q L2         = 144 nq
// Q8_0, IQ4_NL and Q2_0 take K % 256 == 64 j: the last step has nq = j lanes a row (K = 320: one full step, then
// nq = 1).

__host__ __device__ constexpr int32_t sb_bytes(fmt f) { // GGUF bytes per 256 weights
    return f == fmt::q4_k     ? 144
           : f == fmt::q5_k   ? 176
           : f == fmt::q6_k   ? 210
           : f == fmt::iq4_xs ? 136
           : f == fmt::iq3_s  ? 110
           : f == fmt::q8_0   ? 272
           : f == fmt::iq4_nl ? 144
           : f == fmt::q2_0   ? 72
                              : 512;
}
__host__ __device__ constexpr bool has_tail(fmt f) {
    return f == fmt::q8_0 || f == fmt::iq4_nl || f == fmt::q2_0;
}
__host__ __device__ constexpr int32_t lane_bytes(fmt f) { // a lane's bytes per step, the formats with a tail step
    return f == fmt::q8_0 ? 68 : f == fmt::q2_0 ? 18 : 36;
}
__host__ __device__ constexpr int32_t step_bytes(fmt f, int32_t nq) {
    return nq == 4 ? tile_q * sb_bytes(f) : tile_q * nq * lane_bytes(f);
}
static_assert(step_bytes(fmt::q4_k, 4) == 2 * 512 + 128, "Q4_K step");
static_assert(step_bytes(fmt::q5_k, 4) == 2 * 512 + 3 * 128, "Q5_K step");
static_assert(step_bytes(fmt::q6_k, 4) == 3 * 512 + 128 + 16, "Q6_K step");
static_assert(step_bytes(fmt::iq4_xs, 4) == 2 * 512 + 64, "IQ4_XS step");
static_assert(step_bytes(fmt::iq3_s, 4) == 512 + 256 + 64 + 32 + 16, "IQ3_S step");
static_assert(step_bytes(fmt::q8_0, 4) == 4 * 512 + 128 && step_bytes(fmt::iq4_nl, 4) == 2 * 512 + 128, "32-blocks");
static_assert(step_bytes(fmt::q2_0, 4) == 512 + 64 && step_bytes(fmt::q2_0, 1) == 144, "Q2_0 step");

// ---- device helpers ---------------------------------------------------------------------------------------------
// Weight loads read each byte once a round: ld.global.cs allocates them evict-first in L1 and L2 (PTX ISA 9.7.9.1), so
// the round's code and small state stay in L2 between launches (XB1, from CR3; CR1's probe measured the L2 behaviour of
// this form; XB1's probe 3 kept it against ld.global.cs.nc: equal at a dense point, 0.36 us faster cold at the shared
// expert's). sm_70 rejects .cs with .L1::no_allocate.
__device__ __forceinline__ uint4 ldw4(const uint8_t * p) {
    uint4 v;
    asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint2 ldw2(const uint8_t * p) {
    uint2 v;
    asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ldw1(const uint8_t * p) {
    uint32_t v;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ldh1(const uint8_t * p) {
    unsigned short v;
    asm("ld.global.cs.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ldb1(const uint8_t * p) {
    unsigned short v;
    asm("ld.global.cs.u8 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float half_hi(uint32_t h2) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(h2 >> 16)));
}
__device__ __forceinline__ int32_t dp4a(uint32_t a, uint32_t b, int32_t c) {
    return __dp4a(static_cast<int32_t>(a), static_cast<int32_t>(b), c);
}

// The 16-entry IQ4 table (t: kvalues_iq4nl's 16 bytes) at the 8 nibbles of v, in nibble order: lo gets the values of
// v's low 4 nibbles, hi of its high 4 (pack stores IQ4 weights in K order nibble by nibble). Three bits select within a
// table half, the fourth picks the half (ggml's get_int_from_table_16, with the selectors masked so PRMT's sign mode
// never applies).
__device__ __forceinline__ void iq4_lookup(uint32_t v, const uint4 & t, uint32_t & lo, uint32_t & hi) {
    const uint32_t sel = v & 0x77777777u;
    const uint32_t pick = ((v >> 1) & 0x44444444u) | 0x32103210u;
    lo = prmt(prmt(t.x, t.y, sel), prmt(t.z, t.w, sel), pick);
    hi = prmt(prmt(t.x, t.y, sel >> 16), prmt(t.z, t.w, sel >> 16), pick >> 16);
}

// ---- staged activations ---------------------------------------------------------------------------------------
// Row b's codes at xs + b k (16-byte chunks swizzled: chunk i at i ^ ((i >> 2) & 2), so the four lanes of a row read
// four bank groups), and per run of 32 the meta {d, s, sum of q[0..15], sum of q[16..31]} at meta[b][run].
__device__ __forceinline__ int32_t swz(int32_t i) {
    return i ^ ((i >> 2) & 2);
}

struct act {
    const uint8_t * xs;
    const float4 * meta;
    int32_t k;

    __device__ __forceinline__ uint4 chunk(int32_t b, int32_t i) const {
        return *reinterpret_cast<const uint4 *>(xs + b * k + 16 * swz(i));
    }
    __device__ __forceinline__ float2 ds(int32_t b, int32_t j) const { // {d, s}
        return reinterpret_cast<const float2 *>(meta)[2 * (b * (k / 32) + j)];
    }
    __device__ __forceinline__ float d(int32_t b, int32_t j) const { // the block's d
        return reinterpret_cast<const float *>(meta)[4 * (b * (k / 32) + j)];
    }
    __device__ __forceinline__ int32_t sum16(int32_t b, int32_t j, int32_t h) const { // sum of q[16h .. 16h + 15]
        return reinterpret_cast<const int32_t *>(meta)[4 * (b * (k / 32) + j) + 2 + h];
    }
    __device__ __forceinline__ int32_t sum32(int32_t b, int32_t j) const { // sum of the run's 32 codes, exact
        const int2 t = reinterpret_cast<const int2 *>(meta)[2 * (b * (k / 32) + j) + 1];
        return t.x + t.y;
    }
};

// ---- per-format raw loads and step math -------------------------------------------------------------------------
template <fmt F> struct raw;
template <> struct raw<fmt::q4_k> {
    uint4 a, b, h;
};
template <> struct raw<fmt::q5_k> {
    uint4 a, b, h0, h1, hd;
};
template <> struct raw<fmt::q6_k> {
    uint4 a, b, c, sc;
    uint32_t d;
};
template <> struct raw<fmt::iq4_xs> {
    uint4 a, b;
    uint2 h;
};
template <> struct raw<fmt::iq3_s> : iq3s_quarter {};
template <> struct raw<fmt::q8_0> {
    uint4 c[4];
    uint32_t d;
};
template <> struct raw<fmt::iq4_nl> {
    uint4 a, b; // blocks 2q and 2q + 1
    uint32_t d;
};
template <> struct raw<fmt::q2_0> {
    uint4 qs; // block q
    uint32_t d;
};

// The lane's segment pointers at a full step (nq = 4): its lane segments and its row's row segments. Loads add the
// step's offset as an immediate, so a step costs no address arithmetic beyond the pointers' advance.
struct cursor {
    const uint8_t * p[5];

    __device__ __forceinline__ void advance(int32_t n) {
#pragma unroll
        for (int32_t i = 0; i < 5; ++i) {
            p[i] += n;
        }
    }
};

template <fmt F> __device__ __forceinline__ cursor make_cursor(const uint8_t * step, int32_t r, int32_t q) {
    const int32_t li = 4 * r + q;
    cursor c = {{step, step, step, step, step}};
    if constexpr (F == fmt::q4_k || F == fmt::q5_k || F == fmt::iq4_xs || F == fmt::iq4_nl || F == fmt::q8_0) {
        c.p[0] = step + 16 * li;
        c.p[1] = F == fmt::iq4_xs   ? step + 1024 + 8 * r
                 : F == fmt::q8_0   ? step + 2048 + 4 * li
                 : F == fmt::iq4_nl ? step + 1024 + 4 * li
                                    : step + 1024 + 16 * r;
    } else if constexpr (F == fmt::q2_0) {
        c.p[0] = step + 16 * li;
        c.p[1] = step + 512 + 2 * li;
    } else if constexpr (F == fmt::q6_k) {
        c.p[0] = step + 16 * li;
        c.p[1] = step + 1536 + 16 * r;
        c.p[2] = step + 1664 + 2 * r;
    } else {
        static_assert(F == fmt::iq3_s, "a quantized format");
        c.p[0] = step + 16 * li;
        c.p[1] = step + 512 + 8 * li;
        c.p[2] = step + 768 + 2 * li;
        c.p[3] = step + 832 + li;
        c.p[4] = step + 864 + 2 * r;
    }
    return c;
}

// the loads of a full step at OFF bytes past the cursor
template <fmt F, int32_t OFF> __device__ __forceinline__ void load_full(raw<F> & v, const cursor & c) {
    if constexpr (F == fmt::q4_k) {
        v.a = ldw4(c.p[0] + OFF);
        v.b = ldw4(c.p[0] + OFF + 512);
        v.h = ldw4(c.p[1] + OFF);
    } else if constexpr (F == fmt::q5_k) {
        v.a = ldw4(c.p[0] + OFF);
        v.b = ldw4(c.p[0] + OFF + 512);
        v.h0 = ldw4(c.p[1] + OFF);
        v.h1 = ldw4(c.p[1] + OFF + 128);
        v.hd = ldw4(c.p[1] + OFF + 256);
    } else if constexpr (F == fmt::q6_k) {
        v.a = ldw4(c.p[0] + OFF);
        v.b = ldw4(c.p[0] + OFF + 512);
        v.c = ldw4(c.p[0] + OFF + 1024);
        v.sc = ldw4(c.p[1] + OFF);
        v.d = ldh1(c.p[2] + OFF);
    } else if constexpr (F == fmt::iq4_xs) {
        v.a = ldw4(c.p[0] + OFF);
        v.b = ldw4(c.p[0] + OFF + 512);
        v.h = ldw2(c.p[1] + OFF);
    } else if constexpr (F == fmt::iq3_s) {
        v.qs = ldw4(c.p[0] + OFF);
        v.sg = ldw2(c.p[1] + OFF);
        v.qh = ldh1(c.p[2] + OFF);
        v.sc = ldb1(c.p[3] + OFF);
        v.d = ldh1(c.p[4] + OFF);
    } else if constexpr (F == fmt::q8_0) {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            v.c[j] = ldw4(c.p[0] + OFF + 512 * j);
        }
        v.d = ldw1(c.p[1] + OFF);
    } else if constexpr (F == fmt::q2_0) {
        v.qs = ldw4(c.p[0] + OFF);
        v.d = ldh1(c.p[1] + OFF);
    } else {
        static_assert(F == fmt::iq4_nl, "a quantized format");
        v.a = ldw4(c.p[0] + OFF);
        v.b = ldw4(c.p[0] + OFF + 512);
        v.d = ldw1(c.p[1] + OFF);
    }
}

// the loads of lane (r, q) at a tail step p of Q8_0, IQ4_NL or Q2_0 (nq lanes a row, q < nq)
template <fmt F>
__device__ __forceinline__ void load_tail(raw<F> & v, const uint8_t * p, int32_t r, int32_t q, int32_t nq) {
    const int32_t li = r * nq + q;
    if constexpr (F == fmt::q8_0) {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            v.c[j] = ldw4(p + j * 128 * nq + 16 * li);
        }
        v.d = ldw1(p + 512 * nq + 4 * li);
    } else if constexpr (F == fmt::iq4_nl) {
        v.a = ldw4(p + 16 * li);
        v.b = ldw4(p + 128 * nq + 16 * li);
        v.d = ldw1(p + 256 * nq + 4 * li);
    } else if constexpr (F == fmt::q2_0) {
        v.qs = ldw4(p + 16 * li);
        v.d = ldh1(p + 128 * nq + 2 * li);
    }
}

// The lane's 64 weights decoded: 16 int8 words in K order (w[4c + i]: weights 64q + 16c + 4i .. + 3; Q6_K: group i
// in w[4i .. 4i + 3]), and the scales: sub-block j's value = f_j * q - m_j (m only for Q4_K and Q5_K); Q6_K: d and
// the four int8 group scales.
struct dec64 {
    uint32_t w[16];
    float f0, f1, m0, m1, d;
    int32_t sc[4];
};

// Q4_K and Q5_K scale/min of sub-blocks 2q, 2q + 1 from the 12 scale bytes (get_scale_min_k4, branch-free over q)
__device__ __forceinline__ void kq_scales(const uint4 & h, int32_t q, dec64 & o) {
    const float d = half_lo(h.x), dmin = half_hi(h.x);
    const uint32_t sc_lo = h.y & 0x3f3f3f3fu, mn_lo = h.z & 0x3f3f3f3fu;
    const uint32_t sc_hi = (h.w & 0x0f0f0f0fu) | ((h.y >> 2) & 0x30303030u);
    const uint32_t mn_hi = ((h.w >> 4) & 0x0f0f0f0fu) | ((h.z >> 2) & 0x30303030u);
    const uint32_t sc = q < 2 ? sc_lo : sc_hi, mn = q < 2 ? mn_lo : mn_hi;
    const int32_t sh = 16 * (q & 1);
    o.f0 = d * static_cast<float>((sc >> sh) & 0xffu);
    o.f1 = d * static_cast<float>((sc >> (sh + 8)) & 0xffu);
    o.m0 = dmin * static_cast<float>((mn >> sh) & 0xffu);
    o.m1 = dmin * static_cast<float>((mn >> (sh + 8)) & 0xffu);
}

template <fmt F>
__device__ __forceinline__ void decode(const raw<F> & v, int32_t q, const uint4 & kv, const uint32_t * grid,
                                       dec64 & o) {
    if constexpr (F == fmt::q4_k || F == fmt::q5_k) {
#pragma unroll
        for (int32_t k = 0; k < 4; ++k) {
            const uint32_t a = word(v.a, k), b = word(v.b, k);
            o.w[k] = a & 0x0f0f0f0fu;
            o.w[4 + k] = b & 0x0f0f0f0fu;
            o.w[8 + k] = (a >> 4) & 0x0f0f0f0fu;
            o.w[12 + k] = (b >> 4) & 0x0f0f0f0fu;
            if constexpr (F == fmt::q5_k) {
                // qh byte l: bit 2q for weight 64q + l, bit 2q + 1 for 64q + 32 + l
                const uint32_t h0 = word(v.h0, k) >> (2 * q), h1 = word(v.h1, k) >> (2 * q);
                o.w[k] |= (h0 << 4) & 0x10101010u;
                o.w[4 + k] |= (h1 << 4) & 0x10101010u;
                o.w[8 + k] |= (h0 << 3) & 0x10101010u;
                o.w[12 + k] |= (h1 << 3) & 0x10101010u;
            }
        }
        if constexpr (F == fmt::q4_k) {
            kq_scales(v.h, q, o);
        } else {
            kq_scales(v.hd, q, o);
        }
    } else if constexpr (F == fmt::iq4_xs || F == fmt::iq4_nl) {
#pragma unroll
        for (int32_t k = 0; k < 4; ++k) {
            iq4_lookup(word(v.a, k), kv, o.w[2 * k], o.w[2 * k + 1]);
            iq4_lookup(word(v.b, k), kv, o.w[8 + 2 * k], o.w[9 + 2 * k]);
        }
        if constexpr (F == fmt::iq4_xs) {
            const float d = half_lo(v.h.x);
            const uint32_t sl = v.h.y >> (8 * q), sh = v.h.x >> (16 + 4 * q);
            const int32_t ls0 = static_cast<int32_t>((sl & 0xfu) | ((sh & 3u) << 4));
            const int32_t ls1 = static_cast<int32_t>(((sl >> 4) & 0xfu) | (((sh >> 2) & 3u) << 4));
            o.f0 = d * static_cast<float>(ls0 - 32);
            o.f1 = d * static_cast<float>(ls1 - 32);
        } else {
            o.f0 = half_lo(v.d);
            o.f1 = half_hi(v.d);
        }
    } else if constexpr (F == fmt::q8_0) {
#pragma unroll
        for (int32_t c = 0; c < 4; ++c) {
#pragma unroll
            for (int32_t k = 0; k < 4; ++k) {
                o.w[4 * c + k] = word(v.c[c], k);
            }
        }
        o.f0 = half_lo(v.d);
        o.f1 = half_hi(v.d);
    } else if constexpr (F == fmt::q2_0) {
        // pack put weight 16c + 4i + j of the block in bits 2i of byte j of word c; the codes are q + 1, and the -1 is
        // the run's code sum (dots)
#pragma unroll
        for (int32_t c = 0; c < 4; ++c) {
            const uint32_t wd = word(v.qs, c);
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                o.w[4 * c + i] = (wd >> (2 * i)) & 0x03030303u;
            }
        }
        o.f0 = half_lo(v.d);
        o.f1 = o.f0;
    } else if constexpr (F == fmt::iq3_s) {
        iq3s_decode(v, grid, o.w, o.f0, o.f1);
    } else {
        // Q6_K: lane q = 2n + h covers l = 16h .. 16h + 15 of half n; group i (scale sc[8n + h + 2i]) holds weights
        // 128n + 16h + 32i + l'
        static_assert(F == fmt::q6_k, "a quantized format");
        const int32_t n = q >> 1, h = q & 1;
#pragma unroll
        for (int32_t k = 0; k < 4; ++k) {
            const uint32_t a = word(v.a, k), b = word(v.b, k), c = word(v.c, k);
            o.w[k] = (a & 0x0f0f0f0fu) | ((c << 4) & 0x30303030u);
            o.w[4 + k] = (b & 0x0f0f0f0fu) | ((c << 2) & 0x30303030u);
            o.w[8 + k] = ((a >> 4) & 0x0f0f0f0fu) | (c & 0x30303030u);
            o.w[12 + k] = ((b >> 4) & 0x0f0f0f0fu) | ((c >> 2) & 0x30303030u);
        }
        const uint32_t s0 = n ? v.sc.z : v.sc.x, s1 = n ? v.sc.w : v.sc.y;
        o.sc[0] = static_cast<int32_t>(s0 << (24 - 8 * h)) >> 24;
        o.sc[1] = static_cast<int32_t>(s0 << (8 - 8 * h)) >> 24;
        o.sc[2] = static_cast<int32_t>(s1 << (24 - 8 * h)) >> 24;
        o.sc[3] = static_cast<int32_t>(s1 << (8 - 8 * h)) >> 24;
        o.d = half_lo(v.d);
    }
}

// The decoded lane unit dotted with every row's activations at step s, one 32-weight sub-block (Q6_K: one 16-weight
// group) at a time, so only that part's activations are live
template <fmt F, int B>
__device__ __forceinline__ void dots(const dec64 & o, int32_t s, int32_t q, const act & x, float (&acc)[B]) {
    if constexpr (F == fmt::q6_k) {
        // q - 32 is folded into the activations' 16-element sums: sc (sum q a - 32 sum a)
        const int32_t n = q >> 1, h = q & 1;
        const int32_t c0 = 16 * s + 8 * n + h; // group i at chunk c0 + 2i
        const int32_t j0 = 8 * s + 4 * n;      // group i in activation run j0 + i
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
                const uint4 a = x.chunk(b, c0 + 2 * i);
                int32_t is = dp4a(o.w[4 * i + 0], a.x, 0);
                is = dp4a(o.w[4 * i + 1], a.y, is);
                is = dp4a(o.w[4 * i + 2], a.z, is);
                is = dp4a(o.w[4 * i + 3], a.w, is);
                const int32_t t = o.sc[i] * (is - 32 * x.sum16(b, j0 + i, h));
                acc[b] = fmaf(o.d * x.d(b, j0 + i), static_cast<float>(t), acc[b]);
            }
        }
    } else {
        const int32_t c0 = 16 * s + 4 * q; // the lane's first 16-byte activation chunk
        const int32_t j0 = 8 * s + 2 * q;  // its first activation run
#pragma unroll
        for (int32_t j = 0; j < 2; ++j) {
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
                int32_t is = 0;
#pragma unroll
                for (int32_t c = 0; c < 2; ++c) {
                    const uint4 a = x.chunk(b, c0 + 2 * j + c);
                    is = dp4a(o.w[8 * j + 4 * c + 0], a.x, is);
                    is = dp4a(o.w[8 * j + 4 * c + 1], a.y, is);
                    is = dp4a(o.w[8 * j + 4 * c + 2], a.z, is);
                    is = dp4a(o.w[8 * j + 4 * c + 3], a.w, is);
                }
                if constexpr (F == fmt::q2_0) {
                    is -= x.sum32(b, j0 + j);
                }
                const float2 m = x.ds(b, j0 + j);
                acc[b] = fmaf(m.x * (j ? o.f1 : o.f0), static_cast<float>(is), acc[b]);
                if constexpr (F == fmt::q4_k || F == fmt::q5_k) {
                    acc[b] = fmaf(-(j ? o.m1 : o.m0), m.y, acc[b]);
                }
            }
        }
    }
}

} // namespace fn::dense
