#include "fn/quant/synth.h"

#include <cmath>
#include <cstring>

namespace fn::quant {

namespace {

struct rng {
    uint64_t s;

    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15u);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9u;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebu;
        return z ^ (z >> 31);
    }

    // uniform in [0, 1), 53 bits
    double unit() { return static_cast<double>(next() >> 11) * 0x1p-53; }

    // N(0, 1), Box-Muller
    double normal() {
        const double u1 = 1.0 - unit(); // (0, 1]
        const double u2 = unit();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

// Amendment 1's range of every F16 scale
constexpr double wide_lo = 1e-4, wide_hi = 1e-1;

// The deployed |d| of the formats ep-test uses, counted on 8,192 blocks a format from two tensors of the deployed GGUF
// in work/C1b/ref/out (decode_ref, 2026-10-11; logs/deployed-d-stats.txt): {median / 10, 2 x largest}. IQ3_S's blocks
// there are token_embd and blk.47.attn_q, and IQ4_NL's blk.0.ffn_down_exps and per_layer_token_embd.
struct d_range {
    wfmt f;
    double lo, hi;
};
constexpr d_range deployed[] = {
    {wfmt::iq2_xxs, 2.16007233e-05, 2.81715393e-03}, {wfmt::iq2_xs, 2.52485275e-05, 2.38037109e-03},
    {wfmt::iq2_s, 2.27212906e-05, 1.54781342e-03},   {wfmt::iq3_xxs, 9.20891762e-06, 2.55393982e-03},
    {wfmt::iq3_s, 8.68141651e-06, 2.86102295e-03},   {wfmt::q2_0, 1.37634277e-03, 9.38110352e-02},
    {wfmt::iq4_nl, 1.69157982e-05, 2.52914429e-03},
};

// an F16 scale: magnitude log-uniform in [lo_v, hi_v], rounded to F16 (Amendment 1's range is always a normal F16
// number; a deployed range's low end can be subnormal)
uint16_t scale_f16(rng & r, bool positive, double lo_v, double hi_v) {
    const double lo = std::log(lo_v), hi = std::log(hi_v);
    const float mag = static_cast<float>(std::exp(lo + (hi - lo) * r.unit()));
    const bool neg = !positive && (r.next() >> 63) != 0;
    return f32_to_f16(neg ? -mag : mag);
}

void put_u16(uint8_t * p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v & 0xffu);
    p[1] = static_cast<uint8_t>(v >> 8);
}

// N(0, 1) rounded to the type; a non-finite value (F16 overflow, not reachable from N(0, 1) in practice) is drawn again
void synth_value(wfmt f, rng & r, uint8_t * b) {
    for (;;) {
        const float v = static_cast<float>(r.normal());
        if (f == wfmt::f32) {
            std::memcpy(b, &v, sizeof(v));
            return;
        }
        const uint16_t h = f == wfmt::bf16 ? f32_to_bf16(v) : f32_to_f16(v);
        const float back = f == wfmt::bf16 ? bf16_to_f32(h) : f16_to_f32(h);
        if (std::isfinite(back)) {
            put_u16(b, h);
            return;
        }
    }
}

bool fill_blocks(wfmt f, uint64_t seed, int64_t n_blocks, uint8_t * dst, double lo, double hi) {
    if (n_blocks < 0 || dst == nullptr) {
        return false;
    }
    rng r{seed};
    const size_t bb = static_cast<size_t>(info(f).block_bytes);
    for (int64_t i = 0; i < n_blocks; ++i) {
        uint8_t * b = dst + static_cast<size_t>(i) * bb;
        if (f == wfmt::bf16 || f == wfmt::f16 || f == wfmt::f32) {
            synth_value(f, r, b);
            continue;
        }
        // every bit uniform, then the F16 scales over their own fields
        for (size_t k = 0; k < bb; k += 8) {
            const uint64_t v = r.next();
            for (size_t j = 0; j < 8 && k + j < bb; ++j) {
                b[k + j] = static_cast<uint8_t>(v >> (8 * j));
            }
        }
        switch (f) {
            case wfmt::iq2_xxs:
            case wfmt::iq2_xs:
            case wfmt::iq2_s:
            case wfmt::iq3_xxs:
            case wfmt::iq3_s:
                put_u16(b, scale_f16(r, true, lo, hi));
                break;
            case wfmt::q4_k:
            case wfmt::q5_k:
                put_u16(b, scale_f16(r, true, lo, hi));
                put_u16(b + 2, scale_f16(r, true, lo, hi));
                break;
            case wfmt::q6_k:
                put_u16(b + 208, scale_f16(r, false, lo, hi));
                break;
            case wfmt::q2_0:
            case wfmt::q8_0:
            case wfmt::iq4_nl:
            case wfmt::iq4_xs:
                put_u16(b, scale_f16(r, false, lo, hi));
                break;
            case wfmt::bf16:
            case wfmt::f16:
            case wfmt::f32:
                break;
        }
    }
    return true;
}

} // namespace

bool synth_blocks(wfmt f, uint64_t seed, int64_t n_blocks, uint8_t * dst) {
    return fill_blocks(f, seed, n_blocks, dst, wide_lo, wide_hi);
}

bool deployed_d_range(wfmt f, double & lo, double & hi) {
    for (const d_range & d : deployed) {
        if (d.f == f) {
            lo = d.lo;
            hi = d.hi;
            return true;
        }
    }
    return false;
}

bool synth_blocks_deployed(wfmt f, uint64_t seed, int64_t n_blocks, uint8_t * dst) {
    double lo = 0.0, hi = 0.0;
    return deployed_d_range(f, lo, hi) && fill_blocks(f, seed, n_blocks, dst, lo, hi);
}

} // namespace fn::quant
