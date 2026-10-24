#include "fn/quant/formats.h"

#include <cstring>

namespace fn::quant {

namespace {

// Block geometry from ggml/src/ggml-common.h at 66e11c3 (the block_* structs: q2_0 lines 195-198, q8_0 259-262,
// q4_K 334-344, q5_K 351-362, q6_K 369-374, iq2_xxs 388-391, iq2_xs 395-399, iq2_s 403-408, iq3_xxs 414-417, iq3_s
// 422-428, iq4_nl 493-496, iq4_xs 499-504; ME1: q5_1 244-255), and the type numbers from ggml/include/ggml.h lines
// 390-432
constexpr wfmt_info table[n_wfmt] = {
    {"iq2_xxs", 16, 256, 66}, {"iq2_xs", 17, 256, 74}, {"iq2_s", 22, 256, 82},   {"iq3_xxs", 18, 256, 98},
    {"iq3_s", 21, 256, 110},  {"iq4_nl", 20, 32, 18},  {"iq4_xs", 23, 256, 136}, {"q2_0", 42, 64, 18},
    {"q4_k", 12, 256, 144},   {"q5_k", 13, 256, 176},  {"q6_k", 14, 256, 210},   {"q8_0", 8, 32, 34},
    {"bf16", 30, 1, 2},       {"f16", 1, 1, 2},        {"f32", 0, 1, 4},    {"q5_1", 7, 32, 24},
};

uint32_t bits_of(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float float_of(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

} // namespace

const wfmt_info & info(wfmt f) {
    return table[static_cast<int32_t>(f)];
}

bool wfmt_of_gguf_type(int32_t gguf_type, wfmt & out) {
    for (int32_t i = 0; i < n_wfmt; ++i) {
        if (table[i].gguf_type == gguf_type) {
            out = static_cast<wfmt>(i);
            return true;
        }
    }
    return false;
}

bool wfmt_of_name(const char * name, wfmt & out) {
    for (int32_t i = 0; i < n_wfmt; ++i) {
        if (name != nullptr && std::strcmp(table[i].name, name) == 0) {
            out = static_cast<wfmt>(i);
            return true;
        }
    }
    return false;
}

size_t row_bytes(wfmt f, int64_t n) {
    const wfmt_info & t = info(f);
    if (n <= 0 || n % t.block_values != 0) {
        return 0;
    }
    return static_cast<size_t>(n / t.block_values) * static_cast<size_t>(t.block_bytes);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    if (e == 0) {
        // zero or subnormal: m x 2^-24, exact in F32
        const float v = static_cast<float>(m) * 0x1p-24f;
        return float_of(sign | bits_of(v));
    }
    if (e == 31) {
        return float_of(sign | 0x7f800000u | (m << 13) | (m != 0 ? 0x400000u : 0u));
    }
    return float_of(sign | ((e + 112u) << 23) | (m << 13));
}

uint16_t f32_to_f16(float f) {
    const uint32_t x = bits_of(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) {
        return static_cast<uint16_t>(sign | 0x7e00u | ((ax >> 13) & 0x3ffu));
    }
    if (ax >= 0x477ff000u) {
        return static_cast<uint16_t>(sign | 0x7c00u); // at or above 65520 (and infinity): infinity
    }
    const int32_t e = static_cast<int32_t>(ax >> 23) - 127;
    uint32_t mant = ax & 0x7fffffu;
    if (e < -25) {
        return static_cast<uint16_t>(sign); // below half the smallest subnormal
    }
    if (e < -14) {
        // subnormal half: the 24-bit significand shifted right, rounded to nearest even
        mant |= 0x800000u;
        const int32_t shift = -e - 1;
        const uint32_t q = mant >> shift, rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
        const uint32_t r = q + ((rem > half || (rem == half && (q & 1u) != 0)) ? 1u : 0u);
        return static_cast<uint16_t>(sign | r);
    }
    const uint32_t q = mant >> 13, rem = mant & 0x1fffu;
    uint32_t h = (static_cast<uint32_t>(e + 15) << 10) | q;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u) != 0)) {
        ++h; // a carry into the exponent is still the right value
    }
    return static_cast<uint16_t>(sign | h);
}

float bf16_to_f32(uint16_t h) {
    return float_of(static_cast<uint32_t>(h) << 16);
}

uint16_t f32_to_bf16(float f) {
    uint32_t u = bits_of(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((u >> 16) | 0x40u);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

} // namespace fn::quant
