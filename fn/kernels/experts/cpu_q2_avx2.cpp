// ME2: the Q2_0 expert rows (the installed down format) and Strata's planar activation quantizer, for AVX2 CPUs.
// Copied from Strata-V100: the quantizer from 3d4ce4e0, src/kernels/cpu/q2_avx2.cpp (act_quant_q8_1_avx2); the rows
// and the activation's bit-plane image from 69713b61, the same file (:65-145 row_bp_acc, reduce1, rows_bp; :195-226
// bitplane_image; ME6 rider). Strata's notice:
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
// Contract (cpu_kernels.h): a 64-weight Q2_0 block (FP16 d, 16 bytes of 2-bit codes 0..3, value d (c - 1)) spans two
// 32-value activation chunks with their own scales. Code byte i holds values 4i..4i+3, value 4i+k at bits 2k, so
// (bytes >> 2k) & 3 is plane k; the activation is permuted once per row into that plane order (act_q::qp), and two
// blocks (36 contiguous bytes) share one 256-bit vector, a block a 128-bit lane. Per pair and proposal row: four
// maddubs summed in int16 (|four planes| <= 4 x 2 x 3 x 127 = 3048, no saturation), one madd, minus the exact sum of
// the 16 activations each int32 lane met (act_q::psum: the -1 offset in the integers, so sum((c - 1) q) is exact),
// one convert, times the lane's chunk scale (act_q::pscale), one FMA by d into one F32 accumulator a row. Each output
// is reduced alone with one tree, ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)), and a row's operations do not
// depend on the other rows in its call, so its bits are the same at any nt and any cut of the outputs.
// The quantizer: s = amax / 127, codes x / s rounded half away from zero and clamped to +-127, sum the exact code sum,
// hx = s x sum; then the bit-plane image.
//
// Edits against the source: FNFL names and style; act_q holds the 640-value down width and always carries the image
// (Strata's STRATA_Q2_BITPLANE switch and its legacy fallback are not taken; FNFL's contract is the FP64 bound, not
// Strata's bits); rows_bp does one output at a time for up to five proposal rows in one pass (Strata ran 4 + 1 and
// batched four outputs, acc[4][NT]: twenty YMM accumulators at NT = 5 against AVX2's sixteen), so each loaded weight
// pair meets all five rows; reduce4 (the same tree for four outputs) is not taken with it.

#include "fn/kernels/experts/cpu_kernels.h"

#include <immintrin.h>

#include <cstring>

namespace fn::experts::cpu {

namespace {

constexpr int qka = 32; // activation values per chunk
static_assert(act_q_n % 128 == 0, "the bit-plane image takes whole pairs of 64-value blocks");

// One output's weight row against NT proposal rows: acc[t] row t's eight lane sums. At NT = 5 the five accumulators,
// four planes, d, the constants and a row's temporaries fit AVX2's sixteen YMM: g++-14 -O3's pair loop has no stack
// traffic (ME6 rider)
template <int NT> inline void row_bp(const uint8_t * row, const act_q * const * a, int npairs, __m256 * acc) {
    for (int t = 0; t < NT; ++t) {
        acc[t] = _mm256_setzero_ps();
    }
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i didx = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    for (int p = 0; p < npairs; ++p) {
        const uint8_t * blk = row + static_cast<size_t>(p) * 36;
        uint16_t d0, d1;
        std::memcpy(&d0, blk, 2);
        std::memcpy(&d1, blk + 18, 2);
        // the two FP16 bit patterns as one int lane (cvtph reads its low 32 bits)
        const __m128 d01 = _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(uint32_t(d0) | (uint32_t(d1) << 16))));
        const __m256 dp = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(d01), didx); // d0 x4 | d1 x4
        const __m256i c = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2))),
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 20)), 1);
        const __m256i c0 = _mm256_and_si256(c, m3);
        const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(c, 2), m3);
        const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(c, 4), m3);
        const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(c, 6), m3);
        for (int t = 0; t < NT; ++t) {
            // the pair's four 32-byte planes (act_q's alignment)
            const __m256i * q = reinterpret_cast<const __m256i *>(a[t]->qp + static_cast<size_t>(p) * 128);
            __m256i s = _mm256_maddubs_epi16(c0, _mm256_load_si256(q));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c1, _mm256_load_si256(q + 1)));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c2, _mm256_load_si256(q + 2)));
            s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c3, _mm256_load_si256(q + 3)));
            const __m256i i32 = _mm256_sub_epi32(
                _mm256_madd_epi16(s, ones), _mm256_load_si256(reinterpret_cast<const __m256i *>(a[t]->psum + p * 8)));
            const __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), _mm256_load_ps(a[t]->pscale + p * 8));
            acc[t] = _mm256_fmadd_ps(dp, v, acc[t]);
        }
    }
}

inline float reduce1(__m256 v) {
    __m256 h = _mm256_hadd_ps(v, v);
    h = _mm256_hadd_ps(h, h);
    return _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1)));
}

template <int NT>
void rows_bp(const uint8_t * w, size_t row_bytes, int npairs, const act_q * const * a, float * const * out, int r0,
             int r1) {
    for (int r = r0; r < r1; ++r) {
        __m256 acc[NT];
        row_bp<NT>(w + static_cast<size_t>(r) * row_bytes, a, npairs, acc);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = reduce1(acc[t]);
        }
    }
}

void rows_nt(const uint8_t * w, size_t row_bytes, int npairs, const act_q * const * a, int nt, float * const * out,
             int r0, int r1) {
    switch (nt) {
        case 1:
            rows_bp<1>(w, row_bytes, npairs, a, out, r0, r1);
            break;
        case 2:
            rows_bp<2>(w, row_bytes, npairs, a, out, r0, r1);
            break;
        case 3:
            rows_bp<3>(w, row_bytes, npairs, a, out, r0, r1);
            break;
        case 4:
            rows_bp<4>(w, row_bytes, npairs, a, out, r0, r1);
            break;
        case 5: // ME2: five rows in one pass (each weight pair loaded once for all of them)
            rows_bp<5>(w, row_bytes, npairs, a, out, r0, r1);
            break;
        case 6: // MX7: a prompt job's rows up to max_nt (Strata's MAXT)
            rows_bp<6>(w, row_bytes, npairs, a, out, r0, r1);
            break;
        case 7:
            rows_bp<7>(w, row_bytes, npairs, a, out, r0, r1);
            break;
        default:
            rows_bp<8>(w, row_bytes, npairs, a, out, r0, r1);
            break;
    }
}

// act_q::qp / psum / pscale from q and scale: a 64-value block's four 16-byte groups, each shuffled to plane-major
// (4 bytes a plane), then a 4 x 4 dword transpose gives its four 16-byte planes
void bitplane_image(act_q & a) {
    const int nblk = a.nchunks / 2;
    const __m128i shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m128i bias = _mm_set1_epi8(static_cast<char>(0x80));
    for (int b = 0; b < nblk; ++b) {
        const int8_t * q = a.q + b * 64;
        __m128i s[4];
        for (int g = 0; g < 4; ++g) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + 16 * g));
            s[g] = _mm_shuffle_epi8(v, shuf);
            // the sum of 16 signed bytes: sad of q ^ 0x80 is the sum of q + 128, in two 8-byte halves
            const __m128i sad = _mm_sad_epu8(_mm_xor_si128(v, bias), _mm_setzero_si128());
            a.psum[(b >> 1) * 8 + (b & 1) * 4 + g] = _mm_cvtsi128_si32(sad) + _mm_extract_epi16(sad, 4) - 16 * 128;
        }
        const __m128i t0 = _mm_unpacklo_epi32(s[0], s[1]), t1 = _mm_unpacklo_epi32(s[2], s[3]);
        const __m128i t2 = _mm_unpackhi_epi32(s[0], s[1]), t3 = _mm_unpackhi_epi32(s[2], s[3]);
        int8_t * dst = a.qp + (b >> 1) * 128 + (b & 1) * 16;
        _mm_store_si128(reinterpret_cast<__m128i *>(dst), _mm_unpacklo_epi64(t0, t1));
        _mm_store_si128(reinterpret_cast<__m128i *>(dst + 32), _mm_unpackhi_epi64(t0, t1));
        _mm_store_si128(reinterpret_cast<__m128i *>(dst + 64), _mm_unpacklo_epi64(t2, t3));
        _mm_store_si128(reinterpret_cast<__m128i *>(dst + 96), _mm_unpackhi_epi64(t2, t3));
    }
    for (int c = 0; c < a.nchunks; ++c) {
        a.pscale[2 * c] = a.scale[c];
        a.pscale[2 * c + 1] = a.scale[c];
    }
    a.bp_pairs = nblk / 2;
}

} // namespace

void q2_down_rows(const uint8_t * w, size_t row_bytes, int nblocks, const act_q * const * a, int nt,
                  float * const * out, int r0, int r1) {
    rows_nt(w, row_bytes, nblocks / 2, a, nt, out, r0, r1);
}

void quantize_act_q(const float * x, int n, act_q & a) {
    a.nchunks = n / qka;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 half = _mm256_set1_ps(0.5f), mhalf = _mm256_set1_ps(-0.5f), zero = _mm256_setzero_ps();
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float * xb = x + k * qka;
        __m256 v[4];
        __m256 m = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_loadu_ps(xb + 8 * i);
            m = _mm256_max_ps(m, _mm256_and_ps(v[i], absmask));
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        const float amax = _mm_cvtss_f32(_mm_max_ss(h, _mm_movehdup_ps(h)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m256 vinv = _mm256_set1_ps(inv);
        __m256i sum = _mm256_setzero_si256();
        alignas(32) int32_t qi[qka];
        for (int i = 0; i < 4; ++i) {
            const __m256 t = _mm256_mul_ps(v[i], vinv);
            const __m256 r = _mm256_add_ps(t, _mm256_blendv_ps(mhalf, half, _mm256_cmp_ps(t, zero, _CMP_GE_OQ)));
            __m256i q = _mm256_cvttps_epi32(r);
            q = _mm256_min_epi32(_mm256_max_epi32(q, lo), hi);
            sum = _mm256_add_epi32(sum, q);
            _mm256_store_si256(reinterpret_cast<__m256i *>(qi + 8 * i), q);
        }
        for (int j = 0; j < qka; ++j) {
            a.q[k * qka + j] = static_cast<int8_t>(qi[j]);
        }
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        const int32_t total = _mm_cvtsi128_si32(s4);
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * static_cast<float>(total);
    }
    bitplane_image(a);
}

} // namespace fn::experts::cpu
