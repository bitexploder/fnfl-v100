#include "fn/quant/kv_q8_ref.h"

#include <cmath>
#include <cstring>
#include <utility>

namespace fn::quant {

// Strata-V100 f16_bits.hpp f16_from_f32 (d931e1d7), adapted
uint16_t f16_bits_rn(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xffu;
    const int32_t exp = static_cast<int32_t>(rawexp) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (rawexp == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u | (man != 0 ? 0x200u : 0u)); // inf, or a quiet NaN
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u); // a finite overflow to inf
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign); // underflow to a signed zero
        }
        man |= 0x800000u; // the implicit 1
        const uint32_t sh = static_cast<uint32_t>(14 - exp);
        uint32_t h = (man >> sh) & 0x3ffu;
        const uint32_t rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u) != 0)) {
            ++h;
        }
        return static_cast<uint16_t>(sign | h);
    }
    uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0)) {
        ++h; // the carry may raise the exponent
    }
    return static_cast<uint16_t>(h);
}

// Strata-V100 f16_bits.hpp f32_from_f16 (d931e1d7), adapted
float f16_value(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t ex = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    uint32_t out = 0;
    if (ex == 0) {
        if (man == 0) {
            out = sign;
        } else {
            // a subnormal man 2^-24: normalized after s shifts
            uint32_t m = man;
            int32_t s = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++s;
            }
            out = sign | (static_cast<uint32_t>(113 - s) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (ex == 31) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((ex - 15 + 127) << 23) | (man << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, 4);
    return f;
}

void kv_q8_quantize_group(const float * x, int8_t * codes, uint16_t & scale) {
    float amax = 0.0f;
    for (int32_t i = 0; i < kv_q8_group; ++i) {
        amax = std::fmax(amax, std::fabs(x[i]));
    }
    scale = f16_bits_rn(amax / 127.0f);
    const float s = f16_value(scale);
    for (int32_t i = 0; i < kv_q8_group; ++i) {
        int32_t q = 0;
        if (s > 0.0f) {
            const float r = std::nearbyint(x[i] / s); // F32 division, then ties to even (__float2int_rn)
            q = r < -127.0f ? -127 : r > 127.0f ? 127 : static_cast<int32_t>(r);
        }
        codes[i] = static_cast<int8_t>(q);
    }
}

void kv_q8_quantize(const float * x, int32_t n, int8_t * codes, uint16_t * scales) {
    for (int32_t g = 0; g < n / kv_q8_group; ++g) {
        kv_q8_quantize_group(x + g * kv_q8_group, codes + g * kv_q8_group, scales[g]);
    }
}

float kv_q8_value(int8_t code, uint16_t scale) {
    return static_cast<float>(code) * f16_value(scale);
}

std::vector<float> kv_q8_stress_rows(uint64_t seed, int32_t random_rows) {
    uint64_t z = seed;
    const auto next = [&]() { // [0, 1)
        z += 0x9e3779b97f4a7c15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return static_cast<float>(static_cast<double>((x ^ (x >> 31)) >> 11) * (1.0 / 9007199254740992.0));
    };
    std::vector<std::vector<float>> g; // the special groups
    const auto add = [&](std::vector<float> v) { g.push_back(std::move(v)); };
    add(std::vector<float>(kv_q8_group, 0.0f)); // all zeros
    {
        std::vector<float> v(kv_q8_group, 0.0f); // one nonzero value
        v[17] = -3.7f;
        add(v);
    }
    {
        std::vector<float> v(kv_q8_group, 0.0f); // half steps: amax 127 / 8 makes s 1 / 8 exactly
        v[0] = 15.875f;
        for (int32_t i = 1; i < kv_q8_group; ++i) {
            const float k = static_cast<float>((i * 37) % 254 - 127) + 0.5f; // -126.5 .. 126.5
            v[size_t(i)] = (i % 2 ? k : -k) * 0.125f;
        }
        add(v);
    }
    for (int32_t r = 0; r < 4; ++r) {
        // +-amax at codes +-127, and amaxes whose amax / 127 FP16 rounds down (x / s just past 127)
        std::vector<float> v(kv_q8_group);
        const float amax = 1.0f + 0.37f * static_cast<float>(r) + 0.000123f * static_cast<float>(r * r);
        for (int32_t i = 0; i < kv_q8_group; ++i) {
            v[size_t(i)] = amax * (2.0f * next() - 1.0f);
        }
        v[3] = amax;
        v[40] = -amax;
        v[41] = -amax * 0.99999f;
        add(v);
    }
    for (const float amax : {1.0e-3f, 6.0e-5f * 127.0f, 1.0e-5f, 2.0e-5f, 3.0e-6f, 1.0e-6f, 1.0e-40f}) {
        // FP16-subnormal scales (1e-5 and 2e-5: coarse ones, x / s past 127.5, the clamp), a scale that underflows to
        // 0, F32 subnormal inputs
        std::vector<float> v(kv_q8_group);
        for (int32_t i = 0; i < kv_q8_group; ++i) {
            v[size_t(i)] = amax * (2.0f * next() - 1.0f);
        }
        v[9] = -amax;
        add(v);
    }
    for (const float amax : {127.0f * 65504.0f, 8.0e6f, 1.0e5f, 65504.0f}) { // large magnitudes
        std::vector<float> v(kv_q8_group);
        for (int32_t i = 0; i < kv_q8_group; ++i) {
            v[size_t(i)] = amax * (2.0f * next() - 1.0f);
        }
        v[60] = amax;
        add(v);
    }
    while (g.size() % 4 != 0) {
        add(std::vector<float>(kv_q8_group, 0.0f));
    }
    std::vector<float> out;
    for (const std::vector<float> & v : g) {
        out.insert(out.end(), v.begin(), v.end());
    }
    for (int32_t r = 0; r < random_rows * 4; ++r) { // random groups at ranges 1e-2 .. 1e3
        const float range = std::pow(10.0f, -2.0f + 5.0f * next());
        for (int32_t i = 0; i < kv_q8_group; ++i) {
            out.push_back(range * (2.0f * next() - 1.0f));
        }
    }
    return out;
}

} // namespace fn::quant
