#pragma once

// FMS2 item 7: the host's reference of the joint 16-bit X-A / X-F payload (fn/xchg/xchg.h send_half_joint16): an F32
// value's FP16 bits, round to nearest even, every NaN x16_nan, a finite value beyond FP16's range an infinity with over
// set; and an FP16 value back to F32. Bit-level, no host half type, for the tests' bitwise checks.

#include "fn/xchg/xchg.h"

#include <cstdint>
#include <cstring>

namespace fn::xchg {

inline uint16_t x16_ref(float f, bool & over) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    const uint32_t ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u) {
        return x16_nan;
    }
    if (ax == 0x7f800000u) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if (ax >= 0x477ff000u) { // 65,520: the tie above 65,504 (odd mantissa) and beyond round to an infinity
        over = true;
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    const int32_t e = static_cast<int32_t>(ax >> 23) - 127;
    if (e >= -14) { // an FP16 normal: 10 mantissa bits kept, 13 rounded (a carry steps the exponent)
        const uint32_t m = ax & 0x7fffffu;
        uint32_t v = (static_cast<uint32_t>(e + 15) << 10) | (m >> 13);
        const uint32_t rem = m & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (v & 1u) != 0u)) {
            ++v;
        }
        return static_cast<uint16_t>(sign | v);
    }
    if (e < -25) { // below half the smallest subnormal (2^-25): a zero (F32 subnormals too)
        return sign;
    }
    // an FP16 subnormal (unit 2^-24): the 24-bit significand shifted by -(e + 1), 14 .. 24, rounded to even
    const uint32_t mant = (ax & 0x7fffffu) | 0x800000u;
    const int32_t shift = -(e + 1);
    uint32_t v = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (v & 1u) != 0u)) {
        ++v;
    }
    return static_cast<uint16_t>(sign | v);
}

inline float x16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    uint32_t x = 0;
    if (e == 0x1fu) {
        x = sign | 0x7f800000u | (m << 13);
    } else if (e != 0u) {
        x = sign | ((e + 112u) << 23) | (m << 13);
    } else if (m == 0u) {
        x = sign;
    } else { // a subnormal: normalise
        uint32_t ee = 113u;
        while ((m & 0x400u) == 0u) {
            m <<= 1;
            --ee;
        }
        x = sign | (ee << 23) | ((m & 0x3ffu) << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &x, 4);
    return f;
}

} // namespace fn::xchg
