#pragma once

// The CPU reference of KV8's 8-bit KV format (--fnfl-kv-format q8; fn/kernels/qsa/kv_q8.cuh is the device side), the
// rule every writer of the format is checked against bitwise. No GPU code.
//
// A group is 64 consecutive F32 values of one cell's K or V row (a row of 256 is 4 groups):
//   amax = max |x| over the group (F32)
//   scale = FP16 round-to-nearest-even of amax / 127.0f (IEEE division); s = its exact F32 value
//   code = 0 when s == 0 (s > 0 fails), else clamp(nearbyint(x / s), -127, 127) (IEEE division in F32, ties to even)
//   decoded value = float(code) * s, exact in F32 (at most 7 + 11 significant bits, exponent >= -38)
// Inputs are finite and amax <= 127 * 65504 (a larger amax rounds the scale to +inf: out of contract).
//
// Reused from Strata-V100 (/home/jeremy/src/Strata-V100, d931e1d7): the group-64 format and its quantizer
// (src/kernels/cuda/kv_q8.cu quantize_group64 / kv_append_q8_cell) and the exact FP16 conversions
// (include/strata/kernels/f16_bits.hpp f16_from_f32, f32_from_f16), adapted. Deviation: Strata's gather rounds the
// decoded value to FP16 for its scratch; FNFL's decode attention keeps it exact in F32.

#include <cstdint>
#include <vector>

namespace fn::quant {

constexpr int32_t kv_q8_group = 64;

// F32 to FP16 bits, round to nearest even (subnormals and overflow to inf handled; inf and NaN kept)
uint16_t f16_bits_rn(float f);
// FP16 bits to F32, exact (subnormals included)
float f16_value(uint16_t h);

// one group of kv_q8_group values to its codes and FP16 scale bits
void kv_q8_quantize_group(const float * x, int8_t * codes, uint16_t & scale);
// n values (n a multiple of kv_q8_group) to codes [n] and scales [n / kv_q8_group]
void kv_q8_quantize(const float * x, int32_t n, int8_t * codes, uint16_t * scales);
// the value a code stands for under its group's scale bits
float kv_q8_value(int8_t code, uint16_t scale);

// KV8's quantizer test inputs: rows of 256 F32 values (4 groups), each group one case: all zeros; one nonzero value;
// values at exact half steps (x / s = k + 0.5, ties to even); +-amax and a scale rounded down by FP16 (x / s just past
// 127); FP16-subnormal scales, coarse ones past 127.5 (the clamp); a scale that underflows to 0 (and F32 subnormal
// inputs); large magnitudes up
// to 127 x 65504; random data at several ranges. Seeded; the count of rows is rows_out / 256
std::vector<float> kv_q8_stress_rows(uint64_t seed, int32_t random_rows);

} // namespace fn::quant
