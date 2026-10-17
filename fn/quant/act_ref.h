#pragma once

// The CPU reference of FNFL's decode activation format (fn/kernels/act/act.h; C2): the quantization rule in F32, the
// contract every producer of the format is checked against bitwise. No GPU code.

#include <cstdint>

namespace fn::quant {

// x F32 [rows][k] (k a multiple of 32) to codes int8 [rows][k] and scales F32 [rows][k / 32][2] ({d, s}): per run of
// 32 values amax = max |x|, d = amax / 127, id = amax > 0 ? 127 / amax : 0, q = round-to-nearest-even of x id (as
// cvt.rni.s32.f32: NaN to 0) clamped to [-127, 127], s = d (float) sum q. All in F32, divisions IEEE.
void act_quantize_ref(const float * x, int32_t rows, int32_t k, int8_t * codes, float * scales);

// the values a run of the format stands for: code x d, F32 (k values of one row)
void act_values_ref(const int8_t * codes, const float * scales, int32_t k, float * out);

} // namespace fn::quant
