// ME2: the K-quant and legacy-quant expert rows (ME1's Q4 pairs: gate/up Q4_K, Q5_K; down Q5_1, Q8_0) for several
// window rows at once, and their activation quantizers. Copied from Strata-V100 3d4ce4e0, src/kernels/cpu/kq_avx2.cpp
// (q4k_dot, q5_1_dot, q8_0_dot, kq256_gu_rows, kq256_rows), whose functions are ggml-cpu's __AVX2__ dot products
// (ggml/src/ggml-cpu/arch/x86/quants.c) with the loop over rows moved inside the loop over blocks. Strata's notice:
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
// q5k_dot (ggml_vec_dot_q5_K_q8_K, __AVX2__), quantize_q8_0 (quantize_row_q8_0) and quantize_q8_1 (quantize_row_q8_1)
// are ggml-cpu's arch/x86/quants.c at ggml-vfast 66e11c3, q5k_dot reshaped like Strata's q4k_dot (the weight side once
// a block, the rows inside). ggml's notice:
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
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
// Contract (cpu_kernels.h): everything that depends only on the weights (the 4-bit unpacking, the 6-bit scales and
// mins, the fifth bits, |w| of Q8_0, the FP16 scales) is done once a block; every row then runs ggml's integer chain
// and float operations in ggml's order, one F32 accumulator vector a row, reduced once at the end (hsum_float_8).
// Up to 8 rows a pass (MAXT). No prefetch (Strata's kq rows have none).
//
// Edits against the sources (ME2 Result): FNFL names, block structs and style; Q5_K added (Strata's kq256 covers Q4_K,
// Q5_1, Q8_0); the quantizers keep ggml's AVX2 branch only, with ggml's FP16 conversion as F16C _cvtss_sh(x, 0).

#include "fn/kernels/experts/cpu_kernels.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace fn::experts::cpu {

namespace {

constexpr int maxt = max_nt; // Strata's MAXT: the window's rows per group
constexpr int qk_k = 256;

// GGUF's weight blocks (ggml-common.h)
struct block_q4_k {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct block_q5_k {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct block_q5_1 {
    uint16_t d, m;
    uint8_t qh[4];
    uint8_t qs[16];
};
static_assert(sizeof(block_q4_k) == 144 && sizeof(block_q5_k) == 176 && sizeof(block_q5_1) == 24, "GGUF blocks");

inline float h2f(uint16_t u) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(u))));
}
inline uint16_t f2h(float f) {
    return static_cast<uint16_t>(_cvtss_sh(f, 0));
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 / non-VNNI forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask =
        _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(x32)), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rsi));
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i low_mask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(low_mask, bytes);
}
inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}
inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}
alignas(32) const uint8_t k_shuffle[256] = {
    0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,
    1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
    2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,
    5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,
    6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,
    9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
    13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15,
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
inline __m256i get_scale_shuffle_k4(int i) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(k_shuffle) + i);
}

// the K-quants' 12 scale bytes -> 8 scales then 8 mins as 16 x u16 (ggml's utmp unpack)
inline __m256i k_mins_and_scales(const uint8_t * scales12) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    std::memcpy(utmp, scales12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    return _mm256_cvtepu8_epi16(_mm_set_epi32(static_cast<int>(utmp[3]), static_cast<int>(utmp[2]),
                                              static_cast<int>(utmp[1]), static_cast<int>(utmp[0])));
}

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
void q4k_dot(const block_q4_k * x, int nb, const void * const * act, int nt, float * res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[maxt];
    __m128 acc_m[maxt];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d), xdmin = h2f(x[i].dmin);
        const __m256i mins_and_scales = k_mins_and_scales(x[i].scales);
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t * q4 = x[i].qs;
        for (int j = 0; j < qk_k / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q4));
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_k * y = static_cast<const block_q8_k *>(act[t]) + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t * q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y->bsums));
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < qk_k / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_K x Q8_K (ggml_vec_dot_q5_K_q8_K, __AVX2__): the weight side (scales, mins, the 5-bit values) once a block
void q5k_dot(const block_q5_k * x, int nb, const void * const * act, int nt, float * res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc[maxt];
    float summs[maxt];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        summs[t] = 0.f;
    }
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d), xdmin = h2f(x[i].dmin);
        const __m256i mins_and_scales = k_mins_and_scales(x[i].scales);
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const __m256i hbits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x[i].qh));
        __m256i hmask = mone;
        int bit = 0;
        __m256i scale_0[4], scale_1[4], q5_0[4], q5_1[4];
        const uint8_t * q5 = x[i].qs;
        for (int j = 0; j < qk_k / 64; ++j) {
            scale_0[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_1[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q5));
            q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_0[j] = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_1[j] = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_k * y = static_cast<const block_q8_k *>(act[t]) + i;
            const int8_t * q8 = y->qs;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y->bsums));
            const __m128i q8s =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[t] += dmin * _mm_extract_epi32(hsum, 0);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < qk_k / 64; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
                q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8));
                q8 += 32;
                __m256i p16_0 = _mm256_maddubs_epi16(q5_0[j], q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q5_1[j], q8_1);
                p16_0 = _mm256_madd_epi16(scale_0[j], p16_0);
                p16_1 = _mm256_madd_epi16(scale_1[j], p16_1);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        res[t] = hsum_float_8(acc[t]) + summs[t];
    }
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, __AVX2__)
void q5_1_dot(const block_q5_1 * x, int nb, const void * const * act, int nt, float * res) {
    __m256 acc[maxt];
    float summs[maxt];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        summs[t] = 0.0f;
    }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(h2f(x[ib].d));
        const float xm = h2f(x[ib].m);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        for (int t = 0; t < nt; ++t) {
            const block_q8_1 * y = static_cast<const block_q8_1 *>(act[t]) + ib;
            summs[t] += xm * h2f(y->s);
            const __m256 dy = _mm256_set1_ps(h2f(y->d));
            const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y->qs));
            const __m256 q = mul_sum_us8_pairs_float(qx, qy);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        res[t] = hsum_float_8(acc[t]) + summs[t];
    }
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, __AVX2__; mul_sum_i8_pairs_float without AVX-VNNI-INT8)
void q8_0_dot(const block_q8_0 * x, int nb, const void * const * act, int nt, float * res) {
    __m256 acc[maxt];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
    }
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x[ib].qs));
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0 * y = static_cast<const block_q8_0 *>(act[t]) + ib;
            const __m256 d = _mm256_set1_ps(xd * h2f(y->d));
            const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y->qs));
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy);
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        res[t] = hsum_float_8(acc[t]);
    }
}

void dot_rows(fmt f, const uint8_t * row, int n, const void * const * act, int nt, float * res) {
    switch (f) {
        case fmt::q4_k:
            q4k_dot(reinterpret_cast<const block_q4_k *>(row), n / qk_k, act, nt, res);
            break;
        case fmt::q5_k:
            q5k_dot(reinterpret_cast<const block_q5_k *>(row), n / qk_k, act, nt, res);
            break;
        case fmt::q5_1:
            q5_1_dot(reinterpret_cast<const block_q5_1 *>(row), n / 32, act, nt, res);
            break;
        case fmt::q8_0:
            q8_0_dot(reinterpret_cast<const block_q8_0 *>(row), n / 32, act, nt, res);
            break;
        default:
            break;
    }
}

} // namespace

void kq_gu_rows(fmt f, const uint8_t * gate, const uint8_t * up, size_t gu_row, int n, const void * const * act, int nt,
                float * const * ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += maxt) {
        const int m = nt - t0 < maxt ? nt - t0 : maxt;
        float g[maxt], u[maxt];
        for (int r = r0; r < r1; ++r) {
            dot_rows(f, gate + static_cast<size_t>(r) * gu_row, n, act + t0, m, g);
            dot_rows(f, up + static_cast<size_t>(r) * gu_row, n, act + t0, m, u);
            for (int t = 0; t < m; ++t) {
                ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
            }
        }
    }
}

void kq_down_rows(fmt f, const uint8_t * w, size_t row_bytes, int n, const void * const * act, int nt,
                  float * const * out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += maxt) {
        const int m = nt - t0 < maxt ? nt - t0 : maxt;
        float s[maxt];
        for (int r = r0; r < r1; ++r) {
            dot_rows(f, w + static_cast<size_t>(r) * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) {
                out[t0 + t][r] = s[t];
            }
        }
    }
}

// ggml's quantize_row_q8_0 (arch/x86/quants.c, __AVX2__): d = amax / 127 to FP16, codes round(x * 127 / amax)
// (nearest even), packed and permuted into order
void quantize_q8_0(const float * x, void * vy, int64_t k) {
    const int64_t nb = k / 32;
    block_q8_0 * y = static_cast<block_q8_0 *>(vy);
    for (int64_t i = 0; i < nb; i++) {
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;
        const __m256 sign_bit = _mm256_set1_ps(-0.0f);
        __m256 max_abs = _mm256_andnot_ps(sign_bit, v0);
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max_abs, 1), _mm256_castps256_ps128(max_abs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float max_scalar = _mm_cvtss_f32(max4);
        const float d = max_scalar / 127.f;
        y[i].d = f2h(d);
        const float id = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);
        v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_ROUND_NEAREST);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(y[i].qs), i0);
    }
}

// ggml's quantize_row_q8_1 (arch/x86/quants.c, __AVX2__): quantize_q8_0's codes, and s = FP16(d x the code sum)
void quantize_q8_1(const float * x, void * vy, int64_t k) {
    const int64_t nb = k / 32;
    block_q8_1 * y = static_cast<block_q8_1 *>(vy);
    for (int64_t i = 0; i < nb; i++) {
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;
        const __m256 sign_bit = _mm256_set1_ps(-0.0f);
        __m256 max_abs = _mm256_andnot_ps(sign_bit, v0);
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max_abs, 1), _mm256_castps256_ps128(max_abs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float max_scalar = _mm_cvtss_f32(max4);
        const float d = max_scalar / 127.f;
        y[i].d = f2h(d);
        const float id = (max_scalar != 0.0f) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);
        v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_ROUND_NEAREST);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        y[i].s = f2h(d * static_cast<float>(hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1),
                                                                         _mm256_add_epi32(i2, i3)))));
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(y[i].qs), i0);
    }
}

} // namespace fn::experts::cpu
