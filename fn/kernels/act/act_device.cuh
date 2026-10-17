#pragma once

// FNFL's decode activation format (act.h) as a device function, for a producer's epilogue (DP2 Part A step 2): the
// quantize kernel (act.cu) and every epilogue call this one function, so they are the same code and bitwise equal to
// the CPU reference (fn/quant/act_ref).
//
// Contract (quantize_run8):
//   shape:    one run of 32 values, held by 8 consecutive lanes (lanes 8 m .. 8 m + 7, m = lane / 8), 4 values a lane
//             in value order. All 32 lanes of the warp call it together (the shuffles name the full mask); a lane whose
//             run is not real passes zeros and discards the result.
//   numerics: the rule of act.h: amax and sum q exact, d = __fdiv_rn(amax, 127), id = amax > 0 ? __fdiv_rn(127, amax)
//             : 0, q = __float2int_rn(x id) clamped to [-127, 127] (the clamp never binds, so it is not computed), s = d
//             (float) sum q. Inputs finite F32. The including file is built without fast math (--ftz=false
//             --prec-div=true --prec-sqrt=true --fmad=true).
//   output:   packed: the lane's 4 codes, value 0 in the low byte (one 32-bit store at codes + 4 lane); ds: {d, s} of
//             the run, the same in its 8 lanes.

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::act {

__device__ __forceinline__ void quantize_run8(float4 v, uint32_t & packed, float2 & ds) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int32_t o = 4; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float d = __fdiv_rn(amax, 127.0f);
    const float id = amax > 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
    // no clamp is needed: id <= (127 / amax)(1 + 2^-24) and |e| <= amax, so |e id| rounds to at most 127 (1 + 2^-24)^2
    // < 127.5, and its nearest integer is within [-127, 127]
    const auto code = [id](float e) { return __float2int_rn(e * id); };
    const int32_t q0 = code(v.x), q1 = code(v.y), q2 = code(v.z), q3 = code(v.w);
    int32_t sum = q0 + q1 + q2 + q3;
#pragma unroll
    for (int32_t o = 4; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    packed = (static_cast<uint32_t>(q0) & 0xffu) | ((static_cast<uint32_t>(q1) & 0xffu) << 8) |
             ((static_cast<uint32_t>(q2) & 0xffu) << 16) | (static_cast<uint32_t>(q3) << 24);
    ds = make_float2(d, d * static_cast<float>(sum));
}

} // namespace fn::act
