// ME2: the i-quant expert rows in 256-bit lanes, several window rows at once. Copied from Strata-V100 3d4ce4e0,
// src/kernels/cpu/iq_avx2.cpp and src/kernels/cpu/iq_avx2_rows.inl (the plain AVX2 copy, namespace plain), and
// q8k_quant_avx2 from the same iq_avx2.cpp. Strata's notice:
//
// MIT License
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Contract (cpu_kernels.h): each 32-value chunk is decoded once (grid lookups, one sign vector, one scale vector) and
// every row of the job applies it with a load, a sign, a maddubs, a madd and an add; the integer sums of a 256-value
// block are scaled into one F32 accumulator vector per row (an FMA a block), reduced once at the row's end. IQ3_XXS,
// IQ3_S and IQ2_S reduce in ggml-cpu's hsum_float_8 order and apply their K after it (Strata: ggml_hsum); IQ2_XXS and
// IQ2_XS fold K into d. Weight bytes are prefetched 2,048 B ahead (T0). IQ2_XS takes ggml's parity-sign decode
// (Strata row_dot_iq2xs). The grids are FNFL's (grids.h, ggml-common.h's tables at 66e11c3).
//
// Edits against the source (ME2 Result lists them): FNFL names and style; the prefetch distance is the constant
// 2,048 (Strata's STRATA_IQ_PREFETCH default; no environment read); the AVX-VNNI copy, the gathered decodes (118, 121,
// 122: Intel-only on Strata's dispatch), IQ4_XS and the i-quant down rows are not taken (no FNFL format needs them);
// iq4nl_nt takes up to eight rows in one pass (cases 5..8, MX7's 6..8; Strata ran 4 + 1, decoding the block twice);
// every row count runs the multi-row kernel (Strata's STRATA_IQ_MT_MIN=1 rule), so a lone row does not fall back to a
// per-row dot.

#include "fn/kernels/experts/cpu_kernels.h"
#include "fn/kernels/experts/grids.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace fn::experts::cpu {

namespace {

constexpr int qk_k = 256;
constexpr int qk4_nl = 32;
constexpr int iq4_nl_block = 18; // block_iq4_nl: FP16 d, 16 nibble bytes

inline float h2f(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(h))));
}
inline uint32_t u32(const uint8_t * p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint16_t u16(const uint8_t * p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline uint64_t u64(const uint8_t * p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

// The same sign vector read in place from the half's four sign bytes (IQ3_S, IQ2_S): one broadcast load and one
// pshufb expand them.
inline __m256i sgn_vec_at(const uint8_t * p) {
    const __m256i bits =
        _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(u32(p))),
                            _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                                             3, 3, 3, 3, 3, 3, 3));
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, static_cast<char>(0x80), 1, 2, 4, 8, 16, 32, 64,
                                         static_cast<char>(0x80), 1, 2, 4, 8, 16, 32, 64, static_cast<char>(0x80), 1, 2,
                                         4, 8, 16, 32, 64, static_cast<char>(0x80));
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// One scale for the whole 32-value half (IQ2_XXS, IQ3_XXS).
inline __m256i sc32(int s) {
    return _mm256_set1_epi16(static_cast<short>(s));
}

// keven_signs_q2xs: one u64 per 7-bit sign index, byte k = 0xFF when bit k of ksigns_iq2xs[i] is set, 0x01 otherwise
// (ksigns_iq2xs[i] is i's 7 bits plus an even-parity bit 7). Computed at compile time.
struct even_signs_table {
    uint64_t v[128];
    constexpr even_signs_table() : v{} {
        for (int i = 0; i < 128; ++i) {
            int par = 0;
            for (int k = 0; k < 7; ++k) {
                par ^= (i >> k) & 1;
            }
            const int s = i | (par << 7); // == ksigns_iq2xs[i]
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) {
                r |= static_cast<uint64_t>(((s >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            }
            v[i] = r;
        }
    }
};
constexpr even_signs_table even_signs{};
static_assert(even_signs.v[0] == 0x0101010101010101ull && even_signs.v[1] == 0xFF010101010101FFull,
              "keven_signs_q2xs: byte k = 0xFF when bit k of ksigns_iq2xs[i] is set");

// IQ3_S / IQ2_S grid indices: the high index bits of a 32-value half (one qh byte) spread one index per byte.
// IQ3_S: byte k = bit k (the 9th bit of index k). IQ2_S: byte k = bits 2k, 2k+1 (bits 8-9 of index k), k < 4.
struct hi_spread_table {
    uint64_t iq3s[256];
    uint64_t iq2s[256];
    constexpr hi_spread_table() : iq3s{}, iq2s{} {
        for (int h = 0; h < 256; ++h) {
            uint64_t r3 = 0, r2 = 0;
            for (int k = 0; k < 8; ++k) {
                r3 |= static_cast<uint64_t>((h >> k) & 1) << (8 * k);
            }
            for (int k = 0; k < 4; ++k) {
                r2 |= static_cast<uint64_t>((h >> (2 * k)) & 3) << (8 * k);
            }
            iq3s[h] = r3;
            iq2s[h] = r2;
        }
    }
};
constexpr hi_spread_table hi_spread{};
static_assert(hi_spread.iq3s[0x81] == 0x0100000000000001ull && hi_spread.iq2s[0xE4] == 0x0000000003020100ull,
              "hi_spread: one high-bit group per index byte");

// The (2s+1) scale vectors as tables (IQ3_S: one 4-bit scale per 32-value half; IQ2_S: one scale byte holding the two
// 16-value scales of the half).
struct scale_vecs_table {
    int16_t s32[16][16];  // all 16 lanes = 2s+1
    int16_t s16[256][16]; // lanes 0-7 = 2*(b&15)+1, lanes 8-15 = 2*(b>>4)+1
    constexpr scale_vecs_table() : s32{}, s16{} {
        for (int s = 0; s < 16; ++s) {
            for (int l = 0; l < 16; ++l) {
                s32[s][l] = static_cast<int16_t>(2 * s + 1);
            }
        }
        for (int b = 0; b < 256; ++b) {
            for (int l = 0; l < 8; ++l) {
                s16[b][l] = static_cast<int16_t>(2 * (b & 15) + 1);
                s16[b][l + 8] = static_cast<int16_t>(2 * (b >> 4) + 1);
            }
        }
    }
};
constexpr scale_vecs_table scale_vecs{};
static_assert(scale_vecs.s32[15][0] == 31 && scale_vecs.s16[0x2F][7] == 31 && scale_vecs.s16[0x2F][8] == 5,
              "scale_vecs: 2s+1, low nibble in lanes 0-7");

inline __m256i scale_vec(const int16_t * lanes) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lanes));
}

// The low index bytes q[0..7] interleaved with the spread high bits: eight 16-bit grid indices in sp[].
inline void grid_indices(const uint8_t * q, uint64_t hi, uint32_t * sp) {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(sp),
                     _mm_unpacklo_epi8(_mm_cvtsi64_si128(static_cast<long long>(u64(q))),
                                       _mm_cvtsi64_si128(static_cast<long long>(hi))));
}
// One IQ2_S half's four index bytes: the half's four 16-bit indices in the low 8 bytes.
inline void grid_indices4(const uint8_t * q, uint32_t hi, uint32_t * sp) {
    _mm_storel_epi64(reinterpret_cast<__m128i *>(sp), _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(u32(q))),
                                                                        _mm_cvtsi32_si128(static_cast<int>(hi))));
}
// The two 16-bit indices of one 32-bit word of sp[]: (idx_lo, idx_hi) = (index 2k, index 2k+1).
inline uint32_t idx_lo(uint32_t w) {
    return w & 0xffffu;
}
inline uint32_t idx_hi(uint32_t w) {
    return w >> 16;
}

inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ggml-cpu's own hsum_float_8 order (arch/x86/quants.c, __AVX2__): ((r0+r4)+(r2+r6)) + ((r1+r5)+(r3+r7)).
inline float hsum8_ggml(__m256 v) {
    __m128 res = _mm256_extractf128_ps(v, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(v));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// The formats that reduce in ggml's order and scale the result by K: IQ3_XXS, IQ3_S and IQ2_S.
template <int TY> constexpr bool ggml_hsum() {
    return TY == 18 || TY == 21 || TY == 22;
}

// Strata's STRATA_IQ_PREFETCH default: 2048 B ahead, T0 (ME2: a constant)
constexpr int prefetch_distance = 2048;

inline void rows_ahead(const uint8_t * blk, int pf) {
    if (pf <= 0) {
        return;
    }
    _mm_prefetch(reinterpret_cast<const char *>(blk) + pf, _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char *>(blk) + pf + 64, _MM_HINT_T0);
}

// ---- per format: one 32-value half (values 64*j + 32*half .. +31) -> grid magnitudes, sign vector, scales
template <int TY> struct fmt32;

template <> struct fmt32<16> { // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        const uint8_t * q = b + 2 + 16 * j + 8 * half;
        const uint32_t w0 = u32(q), w1 = u32(q + 4);
        g = _mm256_set_epi64x(static_cast<long long>(iq2xxs_grid[w0 >> 24]),
                              static_cast<long long>(iq2xxs_grid[(w0 >> 16) & 255]),
                              static_cast<long long>(iq2xxs_grid[(w0 >> 8) & 255]),
                              static_cast<long long>(iq2xxs_grid[w0 & 255]));
        sgn = _mm256_set_epi64x(static_cast<long long>(even_signs.v[(w1 >> 21) & 127]),
                                static_cast<long long>(even_signs.v[(w1 >> 14) & 127]),
                                static_cast<long long>(even_signs.v[(w1 >> 7) & 127]),
                                static_cast<long long>(even_signs.v[w1 & 127]));
        sc = sc32(2 * static_cast<int>(w1 >> 28) + 1);
    }
};

template <> struct fmt32<22> { // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        alignas(16) uint32_t sp[2];
        grid_indices4(b + 2 + 8 * j + 4 * half, static_cast<uint32_t>(hi_spread.iq2s[b[66 + 2 * j + half]]), sp);
        g = _mm256_set_epi64x(
            static_cast<long long>(iq2s_grid[idx_hi(sp[1])]), static_cast<long long>(iq2s_grid[idx_lo(sp[1])]),
            static_cast<long long>(iq2s_grid[idx_hi(sp[0])]), static_cast<long long>(iq2s_grid[idx_lo(sp[0])]));
        sgn = sgn_vec_at(b + 2 + 32 + 8 * j + 4 * half);
        sc = scale_vec(scale_vecs.s16[b[74 + 2 * j + half]]);
    }
};

template <> struct fmt32<18> { // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        const uint8_t * q = b + 2 + 16 * j + 8 * half;
        g = _mm256_set_epi32(static_cast<int>(iq3xxs_grid[q[7]]), static_cast<int>(iq3xxs_grid[q[6]]),
                             static_cast<int>(iq3xxs_grid[q[5]]), static_cast<int>(iq3xxs_grid[q[4]]),
                             static_cast<int>(iq3xxs_grid[q[3]]), static_cast<int>(iq3xxs_grid[q[2]]),
                             static_cast<int>(iq3xxs_grid[q[1]]), static_cast<int>(iq3xxs_grid[q[0]]));
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = _mm256_set_epi64x(static_cast<long long>(even_signs.v[(w >> 21) & 127]),
                                static_cast<long long>(even_signs.v[(w >> 14) & 127]),
                                static_cast<long long>(even_signs.v[(w >> 7) & 127]),
                                static_cast<long long>(even_signs.v[w & 127]));
        sc = sc32(2 * static_cast<int>(w >> 28) + 1);
    }
};

template <> struct fmt32<21> { // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t * b, int j, int half, __m256i & g, __m256i & sgn, __m256i & sc) {
        const uint8_t * q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
        alignas(16) uint32_t sp[4];
        grid_indices(q, hi_spread.iq3s[h], sp);
        const auto g3 = [](uint32_t k) { return static_cast<int>(iq3s_grid[k]); };
        g = _mm256_set_epi32(g3(idx_hi(sp[3])), g3(idx_lo(sp[3])), g3(idx_hi(sp[2])), g3(idx_lo(sp[2])),
                             g3(idx_hi(sp[1])), g3(idx_lo(sp[1])), g3(idx_hi(sp[0])), g3(idx_lo(sp[0])));
        sgn = sgn_vec_at(b + 74 + 8 * j + 4 * half);
        const uint8_t s = b[106 + j];
        sc = scale_vec(scale_vecs.s32[half ? s >> 4 : s & 15]);
    }
};

// ---- the row kernels (iq_avx2_rows.inl, STRATA_ROWS_VNNI 0)

// acc + madd(p16, sc)
inline __m256i madd_add(__m256i acc, __m256i p16, __m256i sc) {
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p16, sc));
}

// Four u8 x s8 products per int32 lane (maddubs cannot saturate: the u8 side is an IQ4_NL |w| <= 127)
inline __m256i dot4u(__m256i u8, __m256i s8) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u8, s8), _mm256_set1_epi16(1));
}

template <int TY, int NT> inline void row_dot(const uint8_t * row, int nblocks, const block_q8_k * const * y, float * res) {
    const int pf = prefetch_distance;
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) {
        accf[t] = _mm256_setzero_ps();
    }
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t * blk = row + static_cast<size_t>(i) * fmt32<TY>::bytes;
        rows_ahead(blk, pf);
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t) {
            acci[t] = _mm256_setzero_si256();
        }
        for (int j = 0; j < 4; ++j) {
            for (int half = 0; half < 2; ++half) {
                __m256i g, sgn, sc;
                fmt32<TY>::decode(blk, j, half, g, sgn, sc);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = madd_add(acci[t], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        // ggml-cpu's reference for these formats scales the reduced sum by K, not the per-block d, so K is applied at
        // the end below. The other formats keep their K in d.
        const float dx = ggml_hsum<TY>() ? h2f(u16(blk)) : h2f(u16(blk)) * fmt32<TY>::K;
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        res[t] = ggml_hsum<TY>() ? fmt32<TY>::K * hsum8_ggml(accf[t]) : hsum8(accf[t]);
    }
}

// ---- IQ2_XS (17) with ggml's vectorized sign decode (arch/x86/quants.c, ggml_vec_dot_iq2_xs_q8_K): the 8th sign bit
// is reconstructed by parity and the per-half sign vectors come out of shuffles of one 32-byte load. The 8 scale bytes
// become all 16 half-scales with ggml's unpack trick. The grid lookups stay scalar loads into set_epi64x.
template <int NT> inline void row_dot_iq2xs(const uint8_t * row, int nblocks, const block_q8_k * const * y, float * res) {
    static const uint8_t bit_sel[32] = {1, 2, 4, 8, 16, 32, 64, 0x80, 1, 2, 4, 8, 16, 32, 64, 0x80,
                                        1, 2, 4, 8, 16, 32, 64, 0x80, 1, 2, 4, 8, 16, 32, 64, 0x80};
    static const char shuf_even[32] = {0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
                                       4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6};
    static const char shuf_odd[32] = {8,  8,  8,  8,  8,  8,  8,  8,  10, 10, 10, 10, 10, 10, 10, 10,
                                      12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14};
    static const uint8_t bit_help[32] = {0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00,
                                         0x80, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x80, 0x00,
                                         0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00};
    static const uint8_t k_sc_shuffle[128] = {
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
        3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
        6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
        9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
        13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    const __m256i bsel = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bit_sel));
    const __m256i shev = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(shuf_even));
    const __m256i shod = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(shuf_odd));
    const __m256i bhelp = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bit_help));
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
    const __m256i one8 = _mm256_set1_epi8(1);
    const int pf = prefetch_distance;

    __m256 accf[NT];
    for (int t = 0; t < NT; ++t) {
        accf[t] = _mm256_setzero_ps();
    }
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t * blk = row + static_cast<size_t>(i) * 74;
        rows_ahead(blk, pf);
        // the 8 scale bytes -> 16 half-scales of 2*s+1, interleaved [a0, b0, a1, b1, ...] (ggml's unpack)
        __m128i st = _mm_set1_epi64x(static_cast<long long>(u64(blk + 66)));
        st = _mm_unpacklo_epi8(_mm_and_si128(st, m4), _mm_and_si128(_mm_srli_epi16(st, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(st, 1), m1);
        __m256i acc[NT][2];
        for (int t = 0; t < NT; ++t) {
            acc[t][0] = acc[t][1] = _mm256_setzero_si256();
        }
        for (int jj = 0; jj < 2; ++jj) { // 128 values (16 u16) per step
            const __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 2 + 32 * jj));
            const __m256i p7 = _mm256_srli_epi16(q2, 9);  // 7 sign bits per u16, even bytes
            const __m256i p3 = _mm256_srli_epi16(q2, 13); // sign index bits 4-6
            const __m256i fsb =
                _mm256_or_si256(p7, _mm256_shuffle_epi8(bhelp, _mm256_xor_si256(p7, p3))); // + the parity sign bit 7
            const __m256i s01 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(fsb));      // u16 0-7
            const __m256i s23 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(fsb, 1)); // u16 8-15
            alignas(32) uint16_t gi[16];
            _mm256_store_si256(reinterpret_cast<__m256i *>(gi), _mm256_and_si256(q2, m511));
            for (int h = 0; h < 4; ++h) { // the four 32-value halves of this group
                const int H = 4 * jj + h; // global half index -> scales byte pair
                const __m256i g = _mm256_set_epi64x(
                    static_cast<long long>(iq2xs_grid[gi[4 * h + 3]]), static_cast<long long>(iq2xs_grid[gi[4 * h + 2]]),
                    static_cast<long long>(iq2xs_grid[gi[4 * h + 1]]), static_cast<long long>(iq2xs_grid[gi[4 * h]]));
                const __m256i sb = _mm256_shuffle_epi8(h < 2 ? s01 : s23, (h & 1) ? shod : shev);
                const __m256i sgn = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sb, bsel), bsel), one8);
                const __m256i sc = _mm256_cvtepi8_epi16(
                    _mm_shuffle_epi8(scales, _mm_loadu_si128(reinterpret_cast<const __m128i *>(k_sc_shuffle) + H)));
                const int off = 128 * jj + 32 * h;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acc[t][h & 1] = madd_add(acc[t][h & 1], _mm256_maddubs_epi16(g, ys), sc);
                }
            }
        }
        const float dx = h2f(u16(blk)) * 0.125f;
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d),
                                      _mm256_cvtepi32_ps(_mm256_add_epi32(acc[t][0], acc[t][1])), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        res[t] = hsum8(accf[t]);
    }
}

template <int TY, int NT>
inline void row_dot_any(const uint8_t * row, int nblocks, const block_q8_k * const * y, float * res) {
    if constexpr (TY == 17) {
        row_dot_iq2xs<NT>(row, nblocks, y, res);
    } else {
        row_dot<TY, NT>(row, nblocks, y, res);
    }
}

template <int TY, int NT>
void gu_rows(const uint8_t * gate, const uint8_t * up, size_t gu_row, int n, const void * const * act,
             float * const * ff, int r0, int r1) {
    const block_q8_k * y[NT];
    for (int t = 0; t < NT; ++t) {
        y[t] = static_cast<const block_q8_k *>(act[t]);
    }
    const int nb = n / qk_k;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_any<TY, NT>(gate + static_cast<size_t>(r) * gu_row, nb, y, g);
        row_dot_any<TY, NT>(up + static_cast<size_t>(r) * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) {
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t * gate, const uint8_t * up, size_t gu_row, int n, const void * const * act,
                float * const * ff, int r0, int r1) {
    switch (nt) {
        case 1:
            gu_rows<TY, 1>(gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case 2:
            gu_rows<TY, 2>(gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case 3:
            gu_rows<TY, 3>(gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case 4:
            gu_rows<TY, 4>(gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case 5:
            gu_rows<TY, 5>(gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case 6:
            gu_rows<TY, 6>(gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case 7:
            gu_rows<TY, 7>(gate, up, gu_row, n, act, ff, r0, r1);
            break;
        default:
            gu_rows<TY, 8>(gate, up, gu_row, n, act, ff, r0, r1);
            break;
    }
}

// ---- IQ4_NL: 32-value blocks of 18 bytes (FP16 d + 16 nibble bytes) against Q8_0 activations. The nibble decode
// and |w| are computed once per block; every row then costs a sign, a maddubs, a madd and an FMA.
template <int NT>
void iq4nl_rows(const uint8_t * w, size_t row_bytes, int n, const block_q8_0 * const * y, float * const * out, int r0,
                int r1) {
    const __m128i values = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_iq4nl));
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const int nb = n / qk4_nl;
    const int pf = prefetch_distance;
    for (int r = r0; r < r1; ++r) {
        const uint8_t * row = w + static_cast<size_t>(r) * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_setzero_ps();
        }
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t * blk = row + static_cast<size_t>(ib) * iq4_nl_block;
            rows_ahead(blk, pf);
            const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2));
            const __m128i lo = _mm_and_si128(bits, m4b);                    // values 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b); // values 16..31
            const __m256i q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)),
                                                       _mm_shuffle_epi8(values, hi), 1);
            const __m256i aq = _mm256_sign_epi8(q4, q4); // |w|: the unsigned operand of maddubs
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                const block_q8_0 & b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b.qs));
                const __m256i p = dot4u(aq, _mm256_sign_epi8(q8, q4));
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * h2f(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            out[t][r] = hsum8(accf[t]);
        }
    }
}

void iq4nl_nt(const uint8_t * w, size_t row_bytes, int n, const block_q8_0 * const * y, int nt, float * const * out,
              int r0, int r1) {
    switch (nt) {
        case 1:
            iq4nl_rows<1>(w, row_bytes, n, y, out, r0, r1);
            break;
        case 2:
            iq4nl_rows<2>(w, row_bytes, n, y, out, r0, r1);
            break;
        case 3:
            iq4nl_rows<3>(w, row_bytes, n, y, out, r0, r1);
            break;
        case 4:
            iq4nl_rows<4>(w, row_bytes, n, y, out, r0, r1);
            break;
        case 5: // ME2: five rows in one pass (decode once for all of them)
            iq4nl_rows<5>(w, row_bytes, n, y, out, r0, r1);
            break;
        case 6: // MX7: a prompt job's rows up to max_nt (Strata's MAXT)
            iq4nl_rows<6>(w, row_bytes, n, y, out, r0, r1);
            break;
        case 7:
            iq4nl_rows<7>(w, row_bytes, n, y, out, r0, r1);
            break;
        default:
            iq4nl_rows<8>(w, row_bytes, n, y, out, r0, r1);
            break;
    }
}

} // namespace

void iq_gu_rows(fmt f, const uint8_t * gate, const uint8_t * up, size_t gu_row, int n, const void * const * act, int nt,
                float * const * ff, int r0, int r1) {
    switch (f) {
        case fmt::iq2_xxs:
            gu_rows_nt<16>(nt, gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case fmt::iq2_xs:
            gu_rows_nt<17>(nt, gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case fmt::iq3_xxs:
            gu_rows_nt<18>(nt, gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case fmt::iq3_s:
            gu_rows_nt<21>(nt, gate, up, gu_row, n, act, ff, r0, r1);
            break;
        case fmt::iq2_s:
            gu_rows_nt<22>(nt, gate, up, gu_row, n, act, ff, r0, r1);
            break;
        default:
            break;
    }
}

void iq4nl_down_rows(const uint8_t * w, size_t row_bytes, int n, const void * const * act, int nt, float * const * out,
                     int r0, int r1) {
    const block_q8_0 * y[max_nt];
    for (int t = 0; t < nt; ++t) {
        y[t] = static_cast<const block_q8_0 *>(act[t]);
    }
    iq4nl_nt(w, row_bytes, n, y, nt, out, r0, r1);
}

// ggml's quantize_row_q8_K_ref (ggml-quants.c) in AVX2 (Strata q8k_quant_avx2): the same per-value arithmetic, one
// IEEE multiply by the same iscale, the same 1.5*2^23 rounding add, the same clamp, so the bytes are identical. The
// block's sign-carrying max is the FIRST value of the largest magnitude, as in the scalar loop.
void quantize_q8_k(const float * x, void * vy, int64_t k) {
    block_q8_k * y = static_cast<block_q8_k *>(vy);
    const int64_t nb = k / qk_k;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 magic = _mm256_set1_ps(12582912.f);
    const __m256i mant = _mm256_set1_epi32(0x007fffff), off = _mm256_set1_epi32(0x00400000);
    const __m256i c127 = _mm256_set1_epi32(127);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t i = 0; i < nb; ++i, x += qk_k) {
        __m256 m = _mm256_setzero_ps();
        for (int j = 0; j < qk_k; j += 8) {
            m = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), m);
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, qk_k);
            continue;
        }
        float mx = 0.f;
        const __m256 va = _mm256_set1_ps(amax);
        for (int j = 0; j < qk_k; j += 8) {
            const int msk =
                _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), va, _CMP_EQ_OQ));
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1)) {
                    ++b;
                }
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m256 vis = _mm256_set1_ps(iscale);
        for (int j = 0; j < qk_k; j += 32) {
            __m256i v[4];
            for (int q = 0; q < 4; ++q) {
                __m256 p = _mm256_mul_ps(vis, _mm256_loadu_ps(x + j + 8 * q));
                // GCC fuses a multiply and an add into an FMA by default (-ffp-contract=fast); the reference rounds
                // the product before the add, so the product must stay a separate value
                __asm__("" : "+x"(p));
                const __m256 t = _mm256_add_ps(p, magic);
                v[q] = _mm256_min_epi32(_mm256_sub_epi32(_mm256_and_si256(_mm256_castps_si256(t), mant), off), c127);
                // the reference stores MIN(127, v) into an int8 (keeps the low byte), and its bsums add those bytes
                v[q] = _mm256_srai_epi32(_mm256_slli_epi32(v[q], 24), 24);
            }
            for (int q = 0; q < 2; ++q) { // two 16-value sums
                const __m256i s = _mm256_add_epi32(v[2 * q], v[2 * q + 1]);
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
                y[i].bsums[j / 16 + q] = static_cast<int16_t>(_mm_cvtsi128_si32(s4));
            }
            const __m256i p16a = _mm256_packs_epi32(v[0], v[1]), p16b = _mm256_packs_epi32(v[2], v[3]);
            const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), perm);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(y[i].qs + j), p8);
        }
        y[i].d = 1 / iscale;
    }
}

} // namespace fn::experts::cpu
