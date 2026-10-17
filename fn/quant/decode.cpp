#include "fn/quant/decode.h"

#include "fn/kernels/experts/grids.h"

#include <cstring>

namespace fn::quant {

namespace {

// Each decoder below is written from the format's definition in ggml at 66e11c3: its block struct in
// ggml/src/ggml-common.h (formats.cpp names the lines) and its row decoders in ggml/src/ggml-quants.c (named
// beside each). The products run in the definition's order, in F32, so the output is bitwise the definition's. The IQ
// grids and the IQ4_NL values are FNFL's tables (fn/kernels/experts/grids.h).

uint16_t u16_at(const uint8_t * p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t u32_at(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

// byte j of a grid entry, as the definition reads the entry's little-endian bytes
uint32_t grid_byte(uint64_t entry, int32_t j) {
    return static_cast<uint32_t>(entry >> (8 * j)) & 0xffu;
}

// the IQ sign byte of a 7-bit index: the index, and as bit 7 its parity, so a group of 8 has an even number of
// negative signs (ggml-common.h lines 558-567, ksigns_iq2xs, written as its rule)
uint32_t sign_byte(uint32_t idx7) {
    uint32_t p = idx7;
    p ^= p >> 4;
    p ^= p >> 2;
    p ^= p >> 1;
    return idx7 | ((p & 1u) << 7);
}

float sign_of(uint32_t signs, int32_t j) {
    return (signs >> j) & 1u ? -1.0f : 1.0f;
}

// ggml-quants.c lines 439-457 (the Q2_0 row decoder): 64 values, 2 bits each, (q - 1) d; d F16 at 0, qs at 2
void decode_q2_0(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    for (int32_t j = 0; j < 64; ++j) {
        const int32_t q = (b[2 + j / 4] >> ((j % 4) * 2)) & 3;
        y[j] = static_cast<float>(q - 1) * d;
    }
}

// ggml-quants.c lines 553-567 (the Q8_0 row decoder): 32 int8 codes, q d; d F16 at 0, qs at 2
void decode_q8_0(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    for (int32_t j = 0; j < 32; ++j) {
        y[j] = static_cast<float>(static_cast<int8_t>(b[2 + j])) * d;
    }
}

// ggml-quants.c lines 880-887 (get_scale_min_k4): the 6-bit scale and min j of 8, packed in 12 bytes
void scale_min_k4(int32_t j, const uint8_t * q, uint32_t & sc, uint32_t & m) {
    if (j < 4) {
        sc = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        sc = (q[j + 4] & 0xfu) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ggml-quants.c lines 1529-1551 (the Q4_K row decoder): 8 runs of 32, d sc q - dmin m; d at 0, dmin at 2, scales at 4,
// qs at 16 (each 32 bytes hold two runs: low nibbles, then high)
void decode_q4_k(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    const float dmin = f16_to_f32(u16_at(b + 2));
    const uint8_t * q = b + 16;
    for (int32_t r = 0; r < 8; r += 2) {
        uint32_t sc = 0, m = 0;
        scale_min_k4(r, b + 4, sc, m);
        const float d1 = d * static_cast<float>(sc), m1 = dmin * static_cast<float>(m);
        scale_min_k4(r + 1, b + 4, sc, m);
        const float d2 = d * static_cast<float>(sc), m2 = dmin * static_cast<float>(m);
        for (int32_t l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>(q[l] & 0xf) - m1;
        }
        for (int32_t l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
        }
        q += 32;
    }
}

// ggml-quants.c lines 1731-1756 (the Q5_K row decoder): as Q4_K with a fifth bit from qh (bit r of byte l for run r);
// d at 0, dmin at 2, scales at 4, qh at 16, qs at 48
void decode_q5_k(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    const float dmin = f16_to_f32(u16_at(b + 2));
    const uint8_t * ql = b + 48;
    const uint8_t * qh = b + 16;
    for (int32_t r = 0; r < 8; r += 2) {
        uint32_t sc = 0, m = 0;
        scale_min_k4(r, b + 4, sc, m);
        const float d1 = d * static_cast<float>(sc), m1 = dmin * static_cast<float>(m);
        scale_min_k4(r + 1, b + 4, sc, m);
        const float d2 = d * static_cast<float>(sc), m2 = dmin * static_cast<float>(m);
        for (int32_t l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>((ql[l] & 0xf) + ((qh[l] >> r) & 1 ? 16 : 0)) - m1;
        }
        for (int32_t l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] >> (r + 1)) & 1 ? 16 : 0)) - m2;
        }
        ql += 32;
    }
}

// ggml-quants.c lines 1939-1968 (the Q6_K row decoder): 16 runs of 16, d sc q with q = 6 bits - 32; ql at 0, qh at 128,
// int8 scales at 192, d at 208
void decode_q6_k(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b + 208));
    const uint8_t * ql = b;
    const uint8_t * qh = b + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192); // int8 storage read as int8: same object type size
    for (int32_t n = 0; n < 256; n += 128) {
        for (int32_t l = 0; l < 32; ++l) {
            const int32_t is = l / 16;
            const int32_t q1 = static_cast<int32_t>((ql[l] & 0xf) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int32_t q2 = static_cast<int32_t>((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int32_t q3 = static_cast<int32_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int32_t q4 = static_cast<int32_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// ggml-quants.c lines 2488-2512 (the IQ2_XXS row decoder): 8 runs of 32; per run two little-endian u32, the first 4
// grid indices (8 bits each) and the second 4 x 7 sign bits and a 4-bit scale; db = d (0.5 + s) / 4; d at 0, qs at 2
void decode_iq2_xxs(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    for (int32_t r = 0; r < 8; ++r) {
        const uint8_t * q = b + 2 + int64_t{8} * r;
        const uint32_t aux1 = u32_at(q + 4);
        const float db = d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
        for (int32_t l = 0; l < 4; ++l) {
            const uint64_t g = fn::experts::iq2xxs_grid[q[l]];
            const uint32_t signs = sign_byte((aux1 >> (7 * l)) & 127u);
            for (int32_t j = 0; j < 8; ++j) {
                y[j] = db * static_cast<float>(grid_byte(g, j)) * sign_of(signs, j);
            }
            y += 8;
        }
    }
}

// ggml-quants.c lines 2516-2539 (the IQ2_XS row decoder): 8 runs of 32; 4 u16 per run, 9-bit grid index and 7 sign
// bits; a nibble scale per 16 values; d at 0, qs at 2, scales at 66
void decode_iq2_xs(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    for (int32_t r = 0; r < 8; ++r) {
        const uint32_t s = b[66 + r];
        const float db[2] = {d * (0.5f + static_cast<float>(s & 0xf)) * 0.25f,
                             d * (0.5f + static_cast<float>(s >> 4)) * 0.25f};
        for (int32_t l = 0; l < 4; ++l) {
            const uint32_t qv = u16_at(b + 2 + int64_t{2} * (4 * r + l));
            const uint64_t g = fn::experts::iq2xs_grid[qv & 511u];
            const uint32_t signs = sign_byte(qv >> 9);
            for (int32_t j = 0; j < 8; ++j) {
                y[j] = db[l / 2] * static_cast<float>(grid_byte(g, j)) * sign_of(signs, j);
            }
            y += 8;
        }
    }
}

// ggml-quants.c lines 2543-2571 (the IQ2_S row decoder): 8 runs of 32; 10-bit grid index (8 bits in qs, 2 in qh), 8
// explicit sign bits per 8 values; a nibble scale per 16; d at 0, qs at 2 (indices 0..31, signs 32..63), qh at 66,
// scales at 74
void decode_iq2_s(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    const uint8_t * qs = b + 2;
    const uint8_t * signs = b + 2 + 32;
    for (int32_t r = 0; r < 8; ++r) {
        const uint32_t s = b[74 + r];
        const float db[2] = {d * (0.5f + static_cast<float>(s & 0xf)) * 0.25f,
                             d * (0.5f + static_cast<float>(s >> 4)) * 0.25f};
        const uint32_t qh = b[66 + r];
        for (int32_t l = 0; l < 4; ++l) {
            const float dl = db[l / 2];
            const uint64_t g = fn::experts::iq2s_grid[qs[l] | ((qh << (8 - 2 * l)) & 0x300u)];
            for (int32_t j = 0; j < 8; ++j) {
                y[j] = dl * static_cast<float>(grid_byte(g, j)) * sign_of(signs[l], j);
            }
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

// ggml-quants.c lines 2575-2603 (the IQ3_XXS row decoder): 8 runs of 32; 8 grid indices (8 bits) per run, each 4
// values; a little-endian u32 per run of 4 x 7 sign bits and a 4-bit scale; db = d (0.5 + s) / 2; d at 0, qs at 2,
// the u32s at 66
void decode_iq3_xxs(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    const uint8_t * qs = b + 2;
    for (int32_t r = 0; r < 8; ++r) {
        const uint32_t aux = u32_at(b + 66 + int64_t{4} * r);
        const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
        for (int32_t l = 0; l < 4; ++l) {
            const uint32_t signs = sign_byte((aux >> (7 * l)) & 127u);
            const uint32_t g1 = fn::experts::iq3xxs_grid[qs[2 * l + 0]];
            const uint32_t g2 = fn::experts::iq3xxs_grid[qs[2 * l + 1]];
            for (int32_t j = 0; j < 4; ++j) {
                y[j + 0] = db * static_cast<float>(grid_byte(g1, j)) * sign_of(signs, j + 0);
                y[j + 4] = db * static_cast<float>(grid_byte(g2, j)) * sign_of(signs, j + 4);
            }
            y += 8;
        }
        qs += 8;
    }
}

// ggml-quants.c lines 2607-2646 (the IQ3_S row decoder): 8 runs of 32; 9-bit grid index (8 bits in qs, the ninth in
// qh), each 4 values; 8 explicit sign bits per 8 values; a nibble scale per run, db = d (1 + 2 s); d at 0, qs at 2,
// qh at 66, signs at 74, scales at 106
void decode_iq3_s(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    const uint8_t * qs = b + 2;
    const uint8_t * signs = b + 74;
    for (int32_t r = 0; r < 8; ++r) {
        const uint32_t s = b[106 + r / 2];
        const float db = d * static_cast<float>(1 + 2 * ((r % 2) == 0 ? (s & 0xf) : (s >> 4)));
        const uint32_t qh = b[66 + r];
        for (int32_t l = 0; l < 4; ++l) {
            const uint32_t g1 = fn::experts::iq3s_grid[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256u)];
            const uint32_t g2 = fn::experts::iq3s_grid[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256u)];
            for (int32_t j = 0; j < 4; ++j) {
                y[j + 0] = db * static_cast<float>(grid_byte(g1, j)) * sign_of(signs[l], j + 0);
                y[j + 4] = db * static_cast<float>(grid_byte(g2, j)) * sign_of(signs[l], j + 4);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
    }
}

// ggml-quants.c lines 2803-2819 (the IQ4_NL row decoder): 32 values, d kvalues[nibble]; byte j holds value j (low) and
// value j + 16 (high); d at 0, qs at 2
void decode_iq4_nl(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    for (int32_t j = 0; j < 16; ++j) {
        y[j + 0] = d * static_cast<float>(fn::experts::kvalues_iq4nl[b[2 + j] & 0xf]);
        y[j + 16] = d * static_cast<float>(fn::experts::kvalues_iq4nl[b[2 + j] >> 4]);
    }
}

// ggml-quants.c lines 2821-2842 (the IQ4_XS row decoder): 8 runs of 32 as IQ4_NL, each with a 6-bit scale ls (low 4
// bits in scales_l, high 2 in scales_h), dl = d (ls - 32); d at 0, scales_h (u16) at 2, scales_l at 4, qs at 8
void decode_iq4_xs(const uint8_t * b, float * y) {
    const float d = f16_to_f32(u16_at(b));
    const uint32_t sh = u16_at(b + 2);
    const uint8_t * qs = b + 8;
    for (int32_t r = 0; r < 8; ++r) {
        const int32_t ls = static_cast<int32_t>(((b[4 + r / 2] >> (4 * (r % 2))) & 0xf) | (((sh >> (2 * r)) & 3) << 4));
        const float dl = d * static_cast<float>(ls - 32);
        for (int32_t j = 0; j < 16; ++j) {
            y[j + 0] = dl * static_cast<float>(fn::experts::kvalues_iq4nl[qs[j] & 0xf]);
            y[j + 16] = dl * static_cast<float>(fn::experts::kvalues_iq4nl[qs[j] >> 4]);
        }
        y += 32;
        qs += 16;
    }
}

} // namespace

// BF16, F16 and F32: ggml/src/ggml.c lines 469-487 (ggml_fp16_to_fp32_row, ggml_bf16_to_fp32_row) and ggml-impl.h
// lines 401-421 and 611-618 (the scalar conversions): exact widening; F32 is the stored value
void decode_block(wfmt f, const uint8_t * block, float * out) {
    switch (f) {
        case wfmt::iq2_xxs:
            decode_iq2_xxs(block, out);
            return;
        case wfmt::iq2_xs:
            decode_iq2_xs(block, out);
            return;
        case wfmt::iq2_s:
            decode_iq2_s(block, out);
            return;
        case wfmt::iq3_xxs:
            decode_iq3_xxs(block, out);
            return;
        case wfmt::iq3_s:
            decode_iq3_s(block, out);
            return;
        case wfmt::iq4_nl:
            decode_iq4_nl(block, out);
            return;
        case wfmt::iq4_xs:
            decode_iq4_xs(block, out);
            return;
        case wfmt::q2_0:
            decode_q2_0(block, out);
            return;
        case wfmt::q4_k:
            decode_q4_k(block, out);
            return;
        case wfmt::q5_k:
            decode_q5_k(block, out);
            return;
        case wfmt::q6_k:
            decode_q6_k(block, out);
            return;
        case wfmt::q8_0:
            decode_q8_0(block, out);
            return;
        case wfmt::bf16:
            *out = bf16_to_f32(u16_at(block));
            return;
        case wfmt::f16:
            *out = f16_to_f32(u16_at(block));
            return;
        case wfmt::f32:
            std::memcpy(out, block, sizeof(float));
            return;
    }
}

bool decode_row(wfmt f, const void * src, float * dst, int64_t n) {
    if (src == nullptr || dst == nullptr || row_bytes(f, n) == 0) {
        return false;
    }
    const wfmt_info & t = info(f);
    const uint8_t * b = static_cast<const uint8_t *>(src);
    for (int64_t i = 0; i < n / t.block_values; ++i) {
        decode_block(f, b + i * t.block_bytes, dst + i * t.block_values);
    }
    return true;
}

} // namespace fn::quant
