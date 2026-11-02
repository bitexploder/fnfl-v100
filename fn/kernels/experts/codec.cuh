#pragma once

// TTTS3 item 2: the experts' weight formats, one codec a format (Amendment 1: plain structs of static constexpr
// constants and static __forceinline__ device functions, one explicit specialization a format, written out in full).
//
// gu_codec<F>, a routed gate/up format: its packed tile (experts_dev.cuh's layout comment: main chunks, aux8, sc8 or
// the K-quants' scale plane, then d and dmin), its chunk sequence (chunk 0 the record's d, then the aux chunk(s), then
// the main chunks), where each chunk of a record lies (chunk_off) and how it is loaded (load), and take: the step that
// takes chunk q of a record and decodes it, calling out(row, r) for each finished row r (row_dec for the IQ formats,
// kq_row for the K-quants), or, for Q8_0 and Q2_0, finishing its rows' integer sums itself (row_out_q8). The row math
// of a decoded row is the codec's too: row_sums with the split and signed forms for IQ, part (kq_part) for the
// K-quants. A kernel streams the chunks and owns the rows' loop, reduction and GLU; the codec owns the format.
//
// dn_codec<F>, a routed down format, is codec_down.cuh's: its item's chunks and d (dn_slot), the expert's bytes, load,
// the item math (item: one output row's dot with 64 columns, for the rows routed to it) and its walk (the fixed walk
// at 544 threads, or the 512-thread strided walk) with its depth.
//
// Internal to fn/kernels/experts. The kernels take a codec as their format's compile-time code: a format choice stays
// compile-time or block-uniform (the families' with_f), never a branch in an inner loop.

#include "fn/kernels/experts/experts_dev.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace fn::experts {
namespace {

// ---- the decoded rows and their math -------------------------------------------------------------------------------
// One row's sub-block, decoded: 8 grid words, their 8 sign masks and the sub-block scale(s) (two for the formats
// with a 4-bit scale per 16 weights). dw1 is unused when split is false. FMS4: for the signed formats (signed_words)
// g holds the signed words themselves and m is unused.
struct row_dec {
    uint32_t g[8];
    uint32_t m[8];
    float    dw0, dw1;
};

// FMS4: the formats whose decoders write signed grid words into row_dec::g: IQ3_S (item 1, its grid bytes are odd)
// and the three IQ2 formats (item 2, their selector tables)
__host__ __device__ constexpr bool signed_words(fmt f) {
    return f == fmt::iq3_s || f == fmt::iq2_xxs || f == fmt::iq2_xs || f == fmt::iq2_s;
}

// A row's integer sums over its sub-block: lo over words 0..3 and hi over 4..7 (SPLIT), or lo over all 8. SGN: rd.g
// holds signed words, one dp4a a word; else dot_word's mask form, p - 2n. Both are the same integers.
template <bool SPLIT, bool SGN>
__device__ __forceinline__ void row_sums(const row_dec & rd, const uint32_t (&act)[8], int32_t & lo, int32_t & hi) {
    if constexpr (SGN) {
        lo = 0;
        hi = 0;
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            int32_t & s = i < 4 || !SPLIT ? lo : hi;
            s           = __dp4a(static_cast<int32_t>(rd.g[i]), static_cast<int32_t>(act[i]), s);
        }
    } else {
        dot2 l = { 0, 0 }, h = { 0, 0 };
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            dot_word(i < 4 || !SPLIT ? l : h, rd.g[i], rd.m[i], act[i]);
        }
        lo = l.p - 2 * l.n;
        hi = h.p - 2 * h.n;
    }
}

// Q8_0 (DR1): row r's sub-block j is 32 int8 weights against row b's 32 codes, an exact integer sum is[b] (IDP4A on
// signed bytes, PTX ISA dp4a), then (dh * da_b) * is[b] in F32 as the other formats' row_out; the warp reduction is
// row_out's.
template <int B>
__device__ __forceinline__ void row_out_q8(const int32_t (&is)[B],
                                           float dh,
                                           const float (&da)[B],
                                           uint32_t mask,
                                           float *  red_buf,
                                           int32_t  r,
                                           int32_t  w,
                                           int32_t  lane) {
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        if (((mask >> b) & 1u) == 0) {
            continue;
        }
        float part = (dh * da[b]) * static_cast<float>(is[b]);
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        part += __shfl_xor_sync(0xffffffffu, part, 16);
        if (lane < 8) {
            red_buf[(b * 64 + r) * red_stride + w] = part;
        }
    }
}

// ME7, Q2_0: code byte k of word w (weights 4k .. 4k + 3, two bits each from bit 0) as four bytes 0..3, weight 4k + i in
// byte i: the codes against one activation word in __dp4a. The byte broadcast to all four lanes, then lane i keeps its
// bits 2i, 2i + 1
__device__ __forceinline__ uint32_t q2_codes(uint32_t w, int32_t k) {
    const uint32_t y = __byte_perm(w, 0u, 0x1111u * static_cast<uint32_t>(k));
    return (y & 0x00000003u) | ((y >> 2) & 0x00000300u) | ((y >> 4) & 0x00030000u) | ((y >> 6) & 0x03000000u);
}

// ME1, the K-quants (Q4_K, Q5_K): row r's sub-block j as 32 weights 0..31 in the codes' order (word i: weights 4 i ..
// 4 i + 3), its scale f = d sc and min mn = dmin m (each exact in F32: an FP16 times a 6-bit integer). Row b's partial
// is (f da_b) is - mn s_b, is the exact integer sum against row b's codes and s_b the run's staged sum (the meta's y,
// d_b times the codes' sum): one product, then one fused rounding. The warp reduction is row_out's.
struct kq_row {
    uint32_t q[8];
    float    f, mn;
};

__device__ __forceinline__ float kq_part(const kq_row & kr, const uint32_t (&act)[8], float da, float xs) {
    int32_t is = 0;
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
        is = __dp4a(static_cast<int32_t>(kr.q[i]), static_cast<int32_t>(act[i]), is);
    }
    return __fmaf_rn(-kr.mn, xs, __fmul_rn(__fmul_rn(kr.f, da), static_cast<float>(is)));
}

__device__ __forceinline__ void kq_store(float part, float * at, int32_t lane) {
    part += __shfl_xor_sync(0xffffffffu, part, 8);
    part += __shfl_xor_sync(0xffffffffu, part, 16);
    const uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(at));
    asm volatile("{\n\t.reg .pred p;\n\tsetp.lt.s32 p, %0, 8;\n\t@p st.shared.f32 [%1], %2;\n\t}" ::"r"(lane), "r"(sa),
                 "f"(part));
}

// a main chunk's row: byte t's low nibble is weight t, its high nibble weight t + 16; qh's bit t (Q5_K) is bit 4 of
// weight t
template <bool Q5> __device__ __forceinline__ void dec_kq(kq_row & kr, const uint4 & v, uint32_t qh) {
#pragma unroll
    for (int32_t k = 0; k < 4; ++k) {
        const uint32_t x = word_of(v, k);
        kr.q[k]          = x & 0x0f0f0f0fu;
        kr.q[4 + k]      = (x >> 4) & 0x0f0f0f0fu;
        if (Q5) {
            kr.q[k] |= spread4(qh >> (4 * k)) << 4;
            kr.q[4 + k] |= spread4(qh >> (16 + 4 * k)) << 4;
        }
    }
}

// Decoders, one row's sub-block each. dh: the row's FP16 d as F32.
__device__ __forceinline__ void dec_iq3_xxs(row_dec & rd, uint2 q, uint32_t sg, float dh, const uint32_t * grid) {
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
        rd.g[i] = grid[__byte_perm(i < 4 ? q.x : q.y, 0, 0x4440 + (i & 3))];
    }
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        sign_masks7((sg >> (7 * l)) & 0x7fu, rd.m[2 * l], rd.m[2 * l + 1]);
    }
    rd.dw0 = dh * (0.5f + static_cast<float>(sg >> 28)) * 0.5f;
}

// FMS4 item 1: the signed words, g ^ (m & 0xfefefefe) with m the sign mask (0xff in each negative byte). Every IQ3_S
// grid byte u is odd, so u ^ 0xfe is -u exactly (fnfl/evidence/fms4-decode-recipes/verify.py): one LOP3 a word.
__device__ __forceinline__ void
dec_iq3_s(row_dec & rd, uint2 q, uint32_t sg, uint32_t qh, uint32_t sc, float dh, const uint32_t * grid) {
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
        const uint32_t lo = __byte_perm(i < 4 ? q.x : q.y, 0, 0x4440 + (i & 3));
        rd.g[i]           = grid[lo | ((qh << (8 - i)) & 0x100u)];
    }
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        const uint32_t s = (sg >> (8 * l)) & 0xffu;
        rd.g[2 * l] ^= sign_lo(s) & 0xfefefefeu;
        rd.g[2 * l + 1] ^= sign_hi(s) & 0xfefefefeu;
    }
    rd.dw0 = dh * static_cast<float>(1 + 2 * static_cast<int32_t>(sc));
}

// FMS4 item 2: the IQ2 decoders give the signed words (m unused), from the selector tables (grid: 4 bytes an entry)
__device__ __forceinline__ void dec_iq2_xxs(row_dec & rd, uint2 q, float dh, const uint32_t * grid) {
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        iq2_signed(grid[__byte_perm(q.x, 0, 0x4440 + l)], sign8_of7((q.y >> (7 * l)) & 0x7fu), rd.g[2 * l],
                   rd.g[2 * l + 1]);
    }
    rd.dw0 = dh * (0.5f + static_cast<float>(q.y >> 28)) * 0.25f;
}

__device__ __forceinline__ void dec_iq2_xs(row_dec & rd, uint2 q, uint32_t sc, float dh, const uint32_t * grid) {
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        const uint32_t v = ((l < 2 ? q.x : q.y) >> (16 * (l & 1))) & 0xffffu;
        iq2_signed(grid[v & 511u], sign8_of7(v >> 9), rd.g[2 * l], rd.g[2 * l + 1]);
    }
    rd.dw0 = dh * (0.5f + static_cast<float>(sc & 0xfu)) * 0.25f;
    rd.dw1 = dh * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
}

__device__ __forceinline__ void
dec_iq2_s(row_dec & rd, uint32_t qs, uint32_t sg, uint32_t qh, uint32_t sc, float dh, const uint32_t * grid) {
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        const uint32_t idx = __byte_perm(qs, 0, 0x4440 + l) | ((qh << (8 - 2 * l)) & 0x300u);
        iq2_signed(grid[idx], sg >> (8 * l), rd.g[2 * l], rd.g[2 * l + 1]);  // the sign byte's 8 explicit bits
    }
    rd.dw0 = dh * (0.5f + static_cast<float>(sc & 0xfu)) * 0.25f;
    rd.dw1 = dh * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
}

// XB2 (c): a taken chunk's byte pair (x, y), one byte a row (row m < 4: byte m of x; else byte m - 4 of y), taken by
// two byte permutes into lo = [x0 y0 x1 y1], hi = [x2 y2 x3 y3]: real consumers where the chunk is taken. A plain
// copy of the pair (what holding the words compiled to) was placed by the compiler right after the chunk's load, so
// the warp waited for the load there (ring_copies.py), long before the data was used.
struct bytes8 {
    uint32_t lo, hi;
};

__device__ __forceinline__ bytes8 take8(uint32_t x, uint32_t y) {
    return { __byte_perm(x, y, 0x5140), __byte_perm(x, y, 0x7362) };
}

// row m's byte (m < 8, a constant in the unrolled bodies)
__device__ __forceinline__ uint32_t byte8(const bytes8 & b, int32_t m) {
    const int32_t k = m & 3;
    return ((k < 2 ? b.lo : b.hi) >> (8 * (2 * (k & 1) + (m >> 2)))) & 0xffu;
}

// ---- gate/up codecs -------------------------------------------------------------------------------------------------

// the d chunk: rows 0..7's FP16 d as F32, converted where the chunk is taken (XB2 (c): held as words, it compiled to a
// copy right after the next unit's load, a wait long before the data was used)
__device__ __forceinline__ void take_d(const uint4 & v, float (&dh8)[8]) {
#pragma unroll
    for (int32_t mr = 0; mr < 8; ++mr) {
        dh8[mr] = (mr & 1) ? half_hi(word_of(v, mr >> 1)) : half_lo(word_of(v, mr >> 1));
    }
}

template <fmt F> struct gu_codec;

// IQ2_XXS (EX1b, FMS4 item 2): chunk c = [row 2c: qs 8 B][row 2c + 1: qs 8 B] (4 index bytes, then the aux word)
template <> struct gu_codec<fmt::iq2_xxs> {
    static constexpr fmt     f           = fmt::iq2_xxs;
    static constexpr int32_t block_bytes = 66, chunks = 4, grid_off = 17424, grid_len = 1024;
    static constexpr bool    aux8 = false, sc8 = false, kq = false;
    static constexpr int32_t NA = 0, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off, d_off = sc8_off;
    static constexpr bool    split = false, sgn = true, int_rows = false, code_sums = false, b4_two_copies = false;
    static constexpr int32_t max_b  = 5;
    static constexpr int32_t body_b = 1;  // the family body's smallest B (gate_up_kernel_u)
    using row                       = row_dec;

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return q == 0 ? d_off + ((j >> 3) * 8 + rs) * 16 : (int64_t(q - 1 - NA) * gu_threads + rec) * 16;
    }

    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    template <typename O>
    static __device__ __forceinline__ void take(int32_t       q,
                                                const uint4 & v,
                                                float (&dh8)[8],
                                                float (&mn8)[8],
                                                bytes8 &         ax8,
                                                bytes8 &         sc8,
                                                uint4 &          held,
                                                row_dec &        rd,
                                                const uint32_t * grid,
                                                int32_t          j,
                                                int32_t          rs,
                                                O &&             out) {
        const int32_t c = q - 1 - NA;
#pragma unroll
        for (int32_t h = 0; h < 2; ++h) {
            const int32_t mrow = 2 * c + h;
            const uint2   qq   = h == 0 ? make_uint2(v.x, v.y) : make_uint2(v.z, v.w);
            const float   dh   = dh8[mrow];
            dec_iq2_xxs(rd, qq, dh, grid);
            out(rd, rs + 8 * mrow);
        }
    }
};

// IQ2_XS: chunk c = [row 2c: qs 8 B][row 2c + 1: qs 8 B] (4 x (9-bit index, 7 signs)); aux8 = the scales byte
template <> struct gu_codec<fmt::iq2_xs> {
    static constexpr fmt     f           = fmt::iq2_xs;
    static constexpr int32_t block_bytes = 74, chunks = 4, grid_off = 18448, grid_len = 2048;
    static constexpr bool    aux8 = true, sc8 = false, kq = false;
    static constexpr int32_t NA = 1, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off + gu_threads * 8, d_off = sc8_off;
    static constexpr bool    split = true, sgn = true, int_rows = false, code_sums = false, b4_two_copies = false;
    static constexpr int32_t max_b  = 5;
    static constexpr int32_t body_b = 1;  // the family body's smallest B (gate_up_kernel_u)
    using row                       = row_dec;

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return q == 0 ? d_off + ((j >> 3) * 8 + rs) * 16 : (int64_t(q - 1 - NA) * gu_threads + rec) * 16;
    }

    // the aux chunk is one 8-byte load (aux8 in x, y), every other chunk one 16-byte load
    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        if (q == 1) {
            uint4       v = { 0, 0, 0, 0 };
            const uint2 x = __ldcs(reinterpret_cast<const uint2 *>(tl + aux8_off + rec * 8 + z));
            v.x           = x.x;
            v.y           = x.y;
            return v;
        }
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    template <typename O>
    static __device__ __forceinline__ void take(int32_t       q,
                                                const uint4 & v,
                                                float (&dh8)[8],
                                                float (&mn8)[8],
                                                bytes8 &         ax8,
                                                bytes8 &         sc8,
                                                uint4 &          held,
                                                row_dec &        rd,
                                                const uint32_t * grid,
                                                int32_t          j,
                                                int32_t          rs,
                                                O &&             out) {
        const int32_t c = q - 1 - NA;
#pragma unroll
        for (int32_t h = 0; h < 2; ++h) {
            const int32_t  mrow = 2 * c + h;
            const uint2    qq   = h == 0 ? make_uint2(v.x, v.y) : make_uint2(v.z, v.w);
            const float    dh   = dh8[mrow];
            const uint32_t sc   = byte8(ax8, mrow);
            dec_iq2_xs(rd, qq, sc, dh, grid);
            out(rd, rs + 8 * mrow);
        }
    }
};

// IQ2_S: chunk 0 = [qh of rows 0..7][scales of rows 0..7]; chunk 1 + c = [row 2c: q 4, sg 4][row 2c + 1: q, sg]
template <> struct gu_codec<fmt::iq2_s> {
    static constexpr fmt     f           = fmt::iq2_s;
    static constexpr int32_t block_bytes = 82, chunks = 5, grid_off = 20496, grid_len = 4096;
    static constexpr bool    aux8 = false, sc8 = false, kq = false;
    static constexpr int32_t NA = 0, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off, d_off = sc8_off;
    // b4_two_copies: the per-triple kernel at B = 4 keeps two copies of the unit body (its single copy's run-time
    // buffer index spilled 8 bytes at the 96-register cap)
    static constexpr bool    split = true, sgn = true, int_rows = false, code_sums = false, b4_two_copies = true;
    static constexpr int32_t max_b  = 5;
    static constexpr int32_t body_b = 1;  // the family body's smallest B (gate_up_kernel_u)
    using row                       = row_dec;

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return q == 0 ? d_off + ((j >> 3) * 8 + rs) * 16 : (int64_t(q - 1 - NA) * gu_threads + rec) * 16;
    }

    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    template <typename O>
    static __device__ __forceinline__ void take(int32_t       q,
                                                const uint4 & v,
                                                float (&dh8)[8],
                                                float (&mn8)[8],
                                                bytes8 &         ax8,
                                                bytes8 &         sc8,
                                                uint4 &          held,
                                                row_dec &        rd,
                                                const uint32_t * grid,
                                                int32_t          j,
                                                int32_t          rs,
                                                O &&             out) {
        const int32_t c = q - 1 - NA;
        if (c == 0) {
            ax8 = take8(v.x, v.y);  // qh of rows 0..7
            sc8 = take8(v.z, v.w);  // scales of rows 0..7
            return;
        }
#pragma unroll
        for (int32_t h = 0; h < 2; ++h) {
            const int32_t  mrow = 2 * (c - 1) + h;
            const float    dh   = dh8[mrow];
            const uint32_t qh   = byte8(ax8, mrow);
            const uint32_t sc   = byte8(sc8, mrow);
            dec_iq2_s(rd, h == 0 ? v.x : v.z, h == 0 ? v.y : v.w, qh, sc, dh, grid);
            out(rd, rs + 8 * mrow);
        }
    }
};

// IQ3_XXS: rows 4h .. 4h + 3 in three chunks: [q r0][sg r0][sg r1], [q r1][sg r2][sg r3], [q r2][q r3] (q 8 B, sg
// 4 B), so each chunk completes a row and at most three sign words wait in registers
template <> struct gu_codec<fmt::iq3_xxs> {
    static constexpr fmt     f           = fmt::iq3_xxs;
    static constexpr int32_t block_bytes = 98, chunks = 6, grid_off = 14336, grid_len = 1024;
    static constexpr bool    aux8 = false, sc8 = false, kq = false;
    static constexpr int32_t NA = 0, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off, d_off = sc8_off;
    static constexpr bool    split = false, sgn = false, int_rows = false, code_sums = false, b4_two_copies = false;
    static constexpr int32_t max_b  = 5;
    static constexpr int32_t body_b = 4;  // the family body's smallest B (the IQ3 body: 4 and 5)
    using row                       = row_dec;

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return q == 0 ? d_off + (sb * 8 + rs) * 16 : (int64_t(q - 1 - NA) * gu_threads + rec) * 16;
    }

    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    template <typename O>
    static __device__ __forceinline__ void take(int32_t       q,
                                                const uint4 & v,
                                                float (&dh8)[8],
                                                float (&mn8)[8],
                                                bytes8 &         ax8,
                                                bytes8 &         sc8,
                                                uint4 &          held,
                                                row_dec &        rd,
                                                const uint32_t * grid,
                                                int32_t          j,
                                                int32_t          rs,
                                                O &&             out) {
        const int32_t c   = q - 1 - NA;
        const int32_t hq  = c / 3;
        const int32_t k   = c - 3 * hq;
        const auto    row = [&](int32_t mrow, uint2 qq, uint32_t sg) {
            const float dh = dh8[mrow];
            dec_iq3_xxs(rd, qq, sg, dh, grid);
            out(rd, rs + 8 * mrow);
        };
        if (k == 0) {
            held.x = v.w;
            row(4 * hq + 0, make_uint2(v.x, v.y), v.z);
        } else if (k == 1) {
            held.y = v.z;
            held.z = v.w;
            row(4 * hq + 1, make_uint2(v.x, v.y), held.x);
        } else {
            row(4 * hq + 2, make_uint2(v.x, v.y), held.y);
            row(4 * hq + 3, make_uint2(v.z, v.w), held.z);
        }
    }
};

// IQ3_S: IQ3_XXS's chunks with sg = the 4 sign bytes; aux8 = the qh byte, sc8 = scales byte j / 2 of rows 0..7
template <> struct gu_codec<fmt::iq3_s> {
    static constexpr fmt     f           = fmt::iq3_s;
    static constexpr int32_t block_bytes = 110, chunks = 6, grid_off = 15360, grid_len = 2048;
    static constexpr bool    aux8 = true, sc8 = true, kq = false;
    static constexpr int32_t NA = 1, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off + gu_threads * 8, d_off = sc8_off + (n_sub / 2) * 8 * 8;
    static constexpr bool    split = false, sgn = true, int_rows = false, code_sums = false, b4_two_copies = false;
    static constexpr int32_t max_b  = 5;
    static constexpr int32_t body_b = 4;  // the family body's smallest B (the IQ3 body: 4 and 5)
    using row                       = row_dec;

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return q == 0 ? d_off + (sb * 8 + rs) * 16 : (int64_t(q - 1 - NA) * gu_threads + rec) * 16;
    }

    // the aux chunk is two 8-byte loads (aux8 in x, y and sc8 in z, w), every other chunk one 16-byte load
    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        if (q == 1) {
            uint4 v = { 0, 0, 0, 0 };
            {
                const uint2 x = __ldcs(reinterpret_cast<const uint2 *>(tl + aux8_off + rec * 8 + z));
                v.x           = x.x;
                v.y           = x.y;
            }
            {
                const uint2 x = __ldcs(reinterpret_cast<const uint2 *>(tl + sc8_off + ((j >> 1) * 8 + rs) * 8 + z));
                v.z           = x.x;
                v.w           = x.y;
            }
            return v;
        }
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    template <typename O>
    static __device__ __forceinline__ void take(int32_t       q,
                                                const uint4 & v,
                                                float (&dh8)[8],
                                                float (&mn8)[8],
                                                bytes8 &         ax8,
                                                bytes8 &         sc8,
                                                uint4 &          held,
                                                row_dec &        rd,
                                                const uint32_t * grid,
                                                int32_t          j,
                                                int32_t          rs,
                                                O &&             out) {
        const int32_t c   = q - 1 - NA;
        const int32_t hq  = c / 3;
        const int32_t k   = c - 3 * hq;
        const auto    row = [&](int32_t mrow, uint2 qq, uint32_t sg) {
            const float    dh  = dh8[mrow];
            const uint32_t qh  = byte8(ax8, mrow);
            const uint32_t scb = byte8(sc8, mrow);
            dec_iq3_s(rd, qq, sg, qh, (j & 1) ? scb >> 4 : scb & 0xfu, dh, grid);
            out(rd, rs + 8 * mrow);
        };
        if (k == 0) {
            held.x = v.w;
            row(4 * hq + 0, make_uint2(v.x, v.y), v.z);
        } else if (k == 1) {
            held.y = v.z;
            held.z = v.w;
            row(4 * hq + 1, make_uint2(v.x, v.y), held.x);
        } else {
            row(4 * hq + 2, make_uint2(v.x, v.y), held.y);
            row(4 * hq + 3, make_uint2(v.z, v.w), held.z);
        }
    }
};

// Q8_0 (DR1, the draft's): chunk c = row c / 2's int8 weights 16 (c & 1) .. 16 (c & 1) + 15 of sub-block j; each
// 32-weight block has its own d, so the d plane is [rec] of 16 B (the 8 rows' FP16 d of sub-block j). Its rows finish
// as exact integer sums (int_rows): the codec runs them against the routed rows' codes itself
template <> struct gu_codec<fmt::q8_0> {
    static constexpr fmt     f           = fmt::q8_0;
    static constexpr int32_t block_bytes = 8 * 34, chunks = 16, grid_off = 0, grid_len = 0;
    static constexpr bool    aux8 = false, sc8 = false, kq = false;
    static constexpr int32_t NA = 0, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off, d_off = sc8_off;
    static constexpr bool    split = false, sgn = false, int_rows = true, code_sums = false, b4_two_copies = false;
    static constexpr int32_t max_b  = 4;        // the draft's FFN runs B <= 4 (catch-up has none)
    static constexpr int32_t body_b = 0;        // no family body (0): no lanes
    using row                       = row_dec;  // unused: int_rows

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t, int32_t, int32_t) {
        return q == 0 ? d_off + rec * 16 : (int64_t(q - 1 - NA) * gu_threads + rec) * 16;
    }

    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    // int_rows: a chunk taken against every routed row's codes, the row's integer sums finished by row_out_q8 (the
    // kernel's own arrays and values, passed as they are: through a unit struct the compiler placed the partials'
    // addresses differently, TTTS3 sass/b1..b4)
    template <int B>
    static __device__ __forceinline__ void take_rows(int32_t       q,
                                                     const uint4 & v,
                                                     float (&dh8)[8],
                                                     int32_t (&is8)[B],
                                                     const uint32_t (&act)[B][8],
                                                     const float (&da)[B],
                                                     [[maybe_unused]] const int32_t (&sq)[B],
                                                     uint32_t mask,
                                                     float *  red_buf,
                                                     int32_t  rs,
                                                     int32_t  w,
                                                     int32_t  lane) {
        if (q == 0) {
            take_d(v, dh8);
            return;
        }
        // chunk c: half c & 1 of row c / 2's sub-block, against codes 16 (c & 1) .. of each routed row
        const int32_t c    = q - 1 - NA;
        const int32_t mrow = c >> 1, hh = c & 1;
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            if (((mask >> b) & 1u) == 0) {
                continue;
            }
            int32_t acc = hh == 0 ? 0 : is8[b];
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                acc = __dp4a(static_cast<int32_t>(word_of(v, i)), static_cast<int32_t>(act[b][4 * hh + i]), acc);
            }
            is8[b] = acc;
        }
        if (hh == 1) {
            row_out_q8<B>(is8, dh8[mrow], da, mask, red_buf, rs + 8 * mrow, w, lane);
        }
    }
};

// Q2_0 (ME7, the Q2 draft's): chunk c = [row 2c: its 8 code bytes of sub-block j][row 2c + 1's]; a GGUF Q2_0 block
// (FP16 d, 16 code bytes, 64 weights) is sub-blocks 2i, 2i + 1, so the d plane is [j / 2][rs] of 16 B. A row's weights
// are (code - 1) d, so its sub-block integer sum is sum(code x) - sum(x) (code_sums: the staged rows' sums), exact,
// finished as Q8_0's: (d da) is
template <> struct gu_codec<fmt::q2_0> {
    static constexpr fmt     f           = fmt::q2_0;
    static constexpr int32_t block_bytes = 72, chunks = 4, grid_off = 0, grid_len = 0;
    static constexpr bool    aux8 = false, sc8 = false, kq = false;
    static constexpr int32_t NA = 0, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off, d_off = sc8_off;
    static constexpr bool    split = false, sgn = false, int_rows = true, code_sums = true, b4_two_copies = false;
    static constexpr int32_t max_b  = 4;        // the Q2 draft runs B <= 4
    static constexpr int32_t body_b = 0;        // no family body (0): no lanes
    using row                       = row_dec;  // unused: int_rows

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return q == 0 ? d_off + ((j >> 1) * 8 + rs) * 16 : (int64_t(q - 1 - NA) * gu_threads + rec) * 16;
    }

    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    // int_rows: a chunk taken against every routed row's codes, the row's integer sums finished by row_out_q8 (the
    // kernel's own arrays and values, passed as they are: through a unit struct the compiler placed the partials'
    // addresses differently, TTTS3 sass/b1..b4)
    template <int B>
    static __device__ __forceinline__ void take_rows(int32_t       q,
                                                     const uint4 & v,
                                                     float (&dh8)[8],
                                                     int32_t (&is8)[B],
                                                     const uint32_t (&act)[B][8],
                                                     const float (&da)[B],
                                                     [[maybe_unused]] const int32_t (&sq)[B],
                                                     uint32_t mask,
                                                     float *  red_buf,
                                                     int32_t  rs,
                                                     int32_t  w,
                                                     int32_t  lane) {
        if (q == 0) {
            take_d(v, dh8);
            return;
        }
        const int32_t c = q - 1 - NA;
#pragma unroll
        for (int32_t h = 0; h < 2; ++h) {
            const int32_t  mrow = 2 * c + h;
            const uint32_t w0 = h == 0 ? v.x : v.z, w1 = h == 0 ? v.y : v.w;
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
                if (((mask >> b) & 1u) == 0) {
                    continue;
                }
                int32_t acc = -sq[b];
#pragma unroll
                for (int32_t k = 0; k < 4; ++k) {
                    acc = __dp4a(static_cast<int32_t>(q2_codes(w0, k)), static_cast<int32_t>(act[b][k]), acc);
                    acc = __dp4a(static_cast<int32_t>(q2_codes(w1, k)), static_cast<int32_t>(act[b][4 + k]), acc);
                }
                is8[b] = acc;
            }
            row_out_q8<B>(is8, dh8[mrow], da, mask, red_buf, rs + 8 * mrow, w, lane);
        }
    }
};

// Q4_K (ME1, UD-Q4_K_XL's routed gate/up): chunk m = row m's 32 nibbles of sub-block j, byte t = weight t (low) and
// t + 16 (high). After the main chunks the scale plane sm [rec] of 16 B (byte 2m: row m's 6-bit sub-block scale,
// 2m + 1: its 6-bit min), then d and dmin, each [sb][rs] of 16 B (the 8 rows' FP16 d, dmin of superblock sb). Chunks:
// d, dmin, the scale plane (kq: both scaled where it arrives, exact products), then the rows. Row math: part, the
// min-correction form (f da_b) is - mn s_b (kq_part)
template <> struct gu_codec<fmt::q4_k> {
    static constexpr fmt     f           = fmt::q4_k;
    static constexpr int32_t block_bytes = 148, chunks = 8, grid_off = 0, grid_len = 0;
    static constexpr bool    aux8 = false, sc8 = false, kq = true;
    static constexpr int32_t NA = 2, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off, sm_off = aux8_off, d_off = sc8_off + gu_threads * 16,
                             dmin_off = d_off + 10 * 8 * 16;
    static constexpr bool    split = false, sgn = false, int_rows = false, code_sums = false, b4_two_copies = false;
    static constexpr int32_t max_b  = 5;
    static constexpr int32_t body_b = 1;  // TTTS3 item 3: the family body's smallest B (gate_up_kernel_u)
    using row                       = kq_row;

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return q == 0 ? d_off + ((j >> 3) * 8 + rs) * 16 :
               q == 1 ? dmin_off + ((j >> 3) * 8 + rs) * 16 :
               q == 2 ? sm_off + rec * 16 :
                        (int64_t(q - 3) * gu_threads + rec) * 16;
    }

    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    template <typename O>
    static __device__ __forceinline__ void take(int32_t       q,
                                                const uint4 & v,
                                                float (&dh8)[8],
                                                float (&mn8)[8],
                                                bytes8 &         ax8,
                                                bytes8 &         sc8,
                                                uint4 &          held,
                                                row_dec &        rd,
                                                const uint32_t * grid,
                                                int32_t          j,
                                                int32_t          rs,
                                                O &&             out) {
        if (q == 1) {  // the dmin chunk as the d chunk
#pragma unroll
            for (int32_t mr = 0; mr < 8; ++mr) {
                mn8[mr] = (mr & 1) ? half_hi(word_of(v, mr >> 1)) : half_lo(word_of(v, mr >> 1));
            }
            return;
        }
        if (q == 2) {  // the scale plane's bytes 2m, 2m + 1 (row m's scale, min) scale both (exact products)
#pragma unroll
            for (int32_t mr = 0; mr < 8; ++mr) {
                const uint32_t sw = word_of(v, mr >> 1) >> (16 * (mr & 1));
                dh8[mr] *= static_cast<float>(sw & 0xffu);
                mn8[mr] *= static_cast<float>((sw >> 8) & 0xffu);
            }
            return;
        }
        const int32_t mrow = q - 1 - NA;  // chunk c is row c
        kq_row        kr;
        dec_kq<false>(kr, v, 0u);
        kr.f  = dh8[mrow];
        kr.mn = mn8[mrow];
        out(kr, rs + 8 * mrow);
    }

    // row b's partial of a decoded row: act and da its codes and d, xs its staged sum (the run meta's y)
    static __device__ __forceinline__ float part(const kq_row & kr, const uint32_t (&act)[8], float da, float xs) {
        return kq_part(kr, act, da, xs);
    }
};

// Q5_K (ME1, UD-Q4_K_XL's layer 2): Q4_K's with each row's high bits, 4 B a row (bit t: bit 4 of weight t): main chunks
// [qh of rows 0..3][r0][r1][r2][r3] [qh of rows 4..7][r4]..[r7]; the qh chunks are held for their quad
template <> struct gu_codec<fmt::q5_k> {
    static constexpr fmt     f           = fmt::q5_k;
    static constexpr int32_t block_bytes = 180, chunks = 10, grid_off = 0, grid_len = 0;
    static constexpr bool    aux8 = false, sc8 = false, kq = true;
    static constexpr int32_t NA = 2, CC = 1 + NA + chunks;
    static constexpr int64_t T = int64_t(gu_threads) * block_bytes, aux8_off = int64_t(chunks) * gu_threads * 16,
                             sc8_off = aux8_off, sm_off = aux8_off, d_off = sc8_off + gu_threads * 16,
                             dmin_off = d_off + 10 * 8 * 16;
    static constexpr bool    split = false, sgn = false, int_rows = false, code_sums = false, b4_two_copies = false;
    static constexpr int32_t max_b  = 5;
    static constexpr int32_t body_b = 1;  // TTTS3 item 3: the family body's smallest B (gate_up_kernel_u)
    using row                       = kq_row;

    static __device__ __forceinline__ int64_t chunk_off(int32_t q, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return q == 0 ? d_off + ((j >> 3) * 8 + rs) * 16 :
               q == 1 ? dmin_off + ((j >> 3) * 8 + rs) * 16 :
               q == 2 ? sm_off + rec * 16 :
                        (int64_t(q - 3) * gu_threads + rec) * 16;
    }

    static __device__ __forceinline__ uint4
    load(const uint8_t * tl, int32_t q, uint32_t z, int32_t rec, int32_t j, int32_t sb, int32_t rs) {
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q, rec, j, sb, rs) + z));
    }

    template <typename O>
    static __device__ __forceinline__ void take(int32_t       q,
                                                const uint4 & v,
                                                float (&dh8)[8],
                                                float (&mn8)[8],
                                                bytes8 &         ax8,
                                                bytes8 &         sc8,
                                                uint4 &          held,
                                                row_dec &        rd,
                                                const uint32_t * grid,
                                                int32_t          j,
                                                int32_t          rs,
                                                O &&             out) {
        if (q == 1) {  // the dmin chunk as the d chunk
#pragma unroll
            for (int32_t mr = 0; mr < 8; ++mr) {
                mn8[mr] = (mr & 1) ? half_hi(word_of(v, mr >> 1)) : half_lo(word_of(v, mr >> 1));
            }
            return;
        }
        if (q == 2) {  // the scale plane's bytes 2m, 2m + 1 (row m's scale, min) scale both (exact products)
#pragma unroll
            for (int32_t mr = 0; mr < 8; ++mr) {
                const uint32_t sw = word_of(v, mr >> 1) >> (16 * (mr & 1));
                dh8[mr] *= static_cast<float>(sw & 0xffu);
                mn8[mr] *= static_cast<float>((sw >> 8) & 0xffu);
            }
            return;
        }
        const int32_t c = q - 1 - NA;
        if (c == 0 || c == 5) {  // the high bits of rows 0..3 and 4..7
            held = v;
            return;
        }
        const int32_t mrow = c - 1 - (c > 5 ? 1 : 0);
        kq_row        kr;
        dec_kq<true>(kr, v, word_of(held, mrow & 3));
        kr.f  = dh8[mrow];
        kr.mn = mn8[mrow];
        out(kr, rs + 8 * mrow);
    }

    static __device__ __forceinline__ float part(const kq_row & kr, const uint32_t (&act)[8], float da, float xs) {
        return kq_part(kr, act, da, xs);
    }
};

// ---- the packed gate/up layouts from the codecs (the host's packing, the tables' buffer) ---------------------------
struct gu_layout {
    int32_t block_bytes;  // packed bytes per 256 weights (GGUF's but for the K-quants, 4 more: the scales unpacked)
    int32_t chunks;       // main chunks per record
    bool    aux8, sc8;
    int32_t grid_off, grid_len;  // in the grid buffer
};

template <fmt F> __host__ __device__ constexpr gu_layout gu_layout_c() {
    using C = gu_codec<F>;
    return gu_layout{ C::block_bytes, C::chunks, C::aux8, C::sc8, C::grid_off, C::grid_len };
}

// a run-time format's layout: the deployed formats' chain, whose run-time form the IQ bodies take for their table
// (more arms in it changed their SASS); the K-quants and Q2_0 are out of it (gu_layout_k)
__host__ __device__ constexpr gu_layout gu_layout_of(fmt f) {
    return f == fmt::iq2_xxs ? gu_layout_c<fmt::iq2_xxs>() :
           f == fmt::iq2_xs  ? gu_layout_c<fmt::iq2_xs>() :
           f == fmt::iq2_s   ? gu_layout_c<fmt::iq2_s>() :
           f == fmt::iq3_xxs ? gu_layout_c<fmt::iq3_xxs>() :
           f == fmt::iq3_s   ? gu_layout_c<fmt::iq3_s>() :
           f == fmt::q8_0    ? gu_layout_c<fmt::q8_0>() :
                               gu_layout{ 0, 0, false, false, 0, 0 };
}

__host__ __device__ constexpr bool is_kq(fmt f) {
    return f == fmt::q4_k || f == fmt::q5_k;
}

__host__ __device__ constexpr gu_layout gu_layout_k(fmt f) {
    return f == fmt::q4_k ? gu_layout_c<fmt::q4_k>() :
           f == fmt::q5_k ? gu_layout_c<fmt::q5_k>() :
           f == fmt::q2_0 ? gu_layout_c<fmt::q2_0>() :
                            gu_layout_of(f);
}

__host__ __device__ constexpr int64_t tile_bytes(fmt f) {
    return int64_t(gu_threads) * gu_layout_k(f).block_bytes;
}

__host__ __device__ constexpr int64_t aux8_off(fmt f) {
    return int64_t(gu_layout_k(f).chunks) * gu_threads * 16;
}

__host__ __device__ constexpr int64_t sc8_off(fmt f) {
    return aux8_off(f) + (gu_layout_of(f).aux8 ? gu_threads * 8 : 0);
}

__host__ __device__ constexpr int64_t sm_off(fmt f) {  // a K-quant's scale plane is where the aux8 plane would be
    return aux8_off(f);
}

__host__ __device__ constexpr int64_t d_off(fmt f) {
    return sc8_off(f) + (gu_layout_of(f).sc8 ? (n_sub / 2) * 8 * 8 : 0) + (is_kq(f) ? gu_threads * 16 : 0);
}

__host__ __device__ constexpr int64_t dmin_off(fmt f) {
    return d_off(f) + 10 * 8 * 16;
}

static_assert(kvalues_off + 16 == gu_layout_of(fmt::iq2_xxs).grid_off &&
                  gu_layout_of(fmt::iq2_xxs).grid_off + gu_layout_of(fmt::iq2_xxs).grid_len ==
                      gu_layout_of(fmt::iq2_xs).grid_off &&
                  gu_layout_of(fmt::iq2_xs).grid_off + gu_layout_of(fmt::iq2_xs).grid_len ==
                      gu_layout_of(fmt::iq2_s).grid_off &&
                  gu_layout_of(fmt::iq2_s).grid_off + gu_layout_of(fmt::iq2_s).grid_len ==
                      static_cast<int32_t>(grid_bytes),
              "grid buffer layout: the IQ2 selector tables after kvalues_iq4nl");

// each codec's offsets against the layout's (one place a format: the two forms agree)
template <fmt F> constexpr bool gu_codec_ok() {
    using C = gu_codec<F>;
    return C::T == tile_bytes(F) && C::aux8_off == aux8_off(F) && C::sc8_off == sc8_off(F) && C::d_off == d_off(F) &&
           C::CC == 1 + C::NA + C::chunks;
}

static_assert(gu_codec_ok<fmt::iq2_xxs>() && gu_codec_ok<fmt::iq2_xs>() && gu_codec_ok<fmt::iq2_s>() &&
                  gu_codec_ok<fmt::iq3_xxs>() && gu_codec_ok<fmt::iq3_s>() && gu_codec_ok<fmt::q8_0>() &&
                  gu_codec_ok<fmt::q2_0>() && gu_codec_ok<fmt::q4_k>() && gu_codec_ok<fmt::q5_k>(),
              "the codecs' offsets are the layout's");
static_assert(d_off(fmt::iq2_xxs) + 10 * 8 * 16 == tile_bytes(fmt::iq2_xxs), "IQ2_XXS tile bytes");
static_assert(d_off(fmt::iq2_xs) + 10 * 8 * 16 == tile_bytes(fmt::iq2_xs), "IQ2_XS tile bytes");
static_assert(d_off(fmt::iq2_s) + 10 * 8 * 16 == tile_bytes(fmt::iq2_s), "IQ2_S tile bytes");
static_assert(d_off(fmt::iq3_xxs) + 10 * 8 * 16 == tile_bytes(fmt::iq3_xxs), "IQ3_XXS tile bytes");
static_assert(d_off(fmt::iq3_s) + 10 * 8 * 16 == tile_bytes(fmt::iq3_s), "IQ3_S tile bytes");
static_assert(d_off(fmt::q8_0) + gu_threads * 16 == tile_bytes(fmt::q8_0), "Q8_0 tile bytes");
static_assert(dmin_off(fmt::q4_k) + 10 * 8 * 16 == tile_bytes(fmt::q4_k), "Q4_K tile bytes");
static_assert(d_off(fmt::q2_0) + (n_sub / 2) * 8 * 16 == tile_bytes(fmt::q2_0) && tile_bytes(fmt::q2_0) == 46080,
              "Q2_0 tile bytes: 64 rows of 2,560 weights at GGUF's 18 bytes a 64");
static_assert(dmin_off(fmt::q5_k) + 10 * 8 * 16 == tile_bytes(fmt::q5_k), "Q5_K tile bytes");

// routed format f's family body's smallest B (its codec's body_b; 0: no family body, so no lanes), for the host's
// checks: one switch over the codecs
constexpr int32_t gu_body_b(fmt f) {
    switch (f) {
        case fmt::iq2_xxs:
            return gu_codec<fmt::iq2_xxs>::body_b;
        case fmt::iq2_xs:
            return gu_codec<fmt::iq2_xs>::body_b;
        case fmt::iq2_s:
            return gu_codec<fmt::iq2_s>::body_b;
        case fmt::iq3_xxs:
            return gu_codec<fmt::iq3_xxs>::body_b;
        case fmt::iq3_s:
            return gu_codec<fmt::iq3_s>::body_b;
        case fmt::q8_0:
            return gu_codec<fmt::q8_0>::body_b;
        case fmt::q2_0:
            return gu_codec<fmt::q2_0>::body_b;
        case fmt::q4_k:
            return gu_codec<fmt::q4_k>::body_b;
        case fmt::q5_k:
            return gu_codec<fmt::q5_k>::body_b;
        default:
            return 0;
    }
}

}  // namespace
}  // namespace fn::experts
