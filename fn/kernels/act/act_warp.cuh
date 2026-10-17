#pragma once

// FNFL's decode activation format written from a producer's epilogue (DP2): one warp quantizes one run of 32 values,
// lane i holding value i, by the rule of act.h (C2's kernel) and fn/quant/act_ref.h, bitwise. Shared by every producer
// epilogue (DP2's shared-expert GLU; HC3 and EX2 later).
//
// Contract (put_run_warp):
//   participants: all 32 lanes of the warp, converged (the shuffles use the full mask); lane is the caller's lane id.
//   output:       codes[lane] = q of the lane's value; lane 0 writes scales[0] = d, scales[1] = s. codes is one run of
//                 the row's codes (any alignment), scales the run's float2 (8-byte aligned).
//   numerics:     amax = max |x| (exact), d = amax / 127 and id = 127 / amax (0 when amax = 0) as __fdiv_rn,
//                 q = __float2int_rn(x id) clamped to [-127, 127], s = d (float) sum q with the sum exact in int32. The
//                 including file must build without fast math (--ftz=false --prec-div=true --prec-sqrt=true
//                 --fmad=true), as act.cu does; then the output is a function of the 32 values alone.
//   evidence:     C2 (the format and its CPU reference), DP2 (dp-test checks every epilogue run bitwise).

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::act {

__device__ __forceinline__ void put_run_warp(float v, int32_t lane, int8_t * codes, float * scales) {
    float amax = fabsf(v);
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    const float d = __fdiv_rn(amax, 127.0f);
    const float id = amax > 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
    const int32_t q = min(max(__float2int_rn(v * id), -127), 127);
    int32_t sum = q;
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    codes[lane] = static_cast<int8_t>(q);
    if (lane == 0) {
        *reinterpret_cast<float2 *>(scales) = make_float2(d, d * static_cast<float>(sum));
    }
}

} // namespace fn::act
