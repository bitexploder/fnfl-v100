#pragma once

// KV8 (--fnfl-kv-format q8): the 8-bit KV format's device side, shared by QA2's prepare and attend (attend.cu) and the
// prompt path's qsp_kv and attention (qsa_prompt_attn.cu). The rule and its CPU reference are fn/quant/kv_q8_ref.h:
// per 64 values amax, scale = fp16_rn(amax / 127), code = clamp(rint(x / s), -127, 127) (0 when s is 0), value =
// float(code) * s exact in F32. Built without fast math (--prec-div=true: both divisions are IEEE).
//
// Layout (state.h "target KV (K0)" in q8): a QSA layer's codes int8 [cell][256] in place of its BF16 K or V array,
// and the class's "K scales" and "V scales" parts FP16 [cell][64], cell-major: cell c's group g of QSA layer o is at
// [c * 64 + 4 o + g] (48 of the 64 used; 128 bytes a cell keeps a page's slice on the 2 MiB granularity). A kernel
// takes its layer's scale pointer (the part's base + 4 o) and reads [c * scale_stride + g].
//
// Reused from Strata-V100 (d931e1d7): the group-64 format and quantizer (src/kernels/cuda/kv_q8.cu), adapted to a
// warp of 8-value lanes; the decode is FNFL's (exact F32, no FP16 rounding of the value).

#include <cuda_fp16.h>

#include <cstdint>

namespace fn::qsa::kv8 {

constexpr int32_t group = 64;
constexpr int32_t scale_stride = 64; // FP16 scales a cell in the scale parts (12 QSA layers x 4 groups, padded)
constexpr int32_t row_groups = 4;    // groups of a 256-value row

// A warp quantizes one 256-value row, lane l holding values 8 l .. 8 l + 7: lanes 8 g .. 8 g + 7 are group g, whose
// amax is the lanes' own maxima by a 3-step xor butterfly (every lane takes part). codes: the lane's 8 codes, value
// 8 l + e in byte e; scale: the group's FP16 bits (every lane of the group holds them).
__device__ __forceinline__ void quantize8(const float (&x)[8], uint2 & codes, uint16_t & scale) {
    float a = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        a = fmaxf(a, fabsf(x[e]));
    }
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    }
    const __half h = __float2half_rn(a / 127.0f);
    const float s = __half2float(h);
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        int32_t q = 0;
        if (s > 0.0f) {
            q = min(max(__float2int_rn(x[e] / s), -127), 127);
        }
        w[e / 4] |= (static_cast<uint32_t>(q) & 0xffu) << (8 * (e % 4));
    }
    codes = make_uint2(w[0], w[1]);
    scale = __half_as_ushort(h);
}

// a group's scale bits as F32 (exact)
__device__ __forceinline__ float scale_value(uint32_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xffffu)));
}

// 4 codes (one word, value i in byte i) times s, exact in F32: each biased byte placed under 2^23's exponent (a PRMT)
// is 2^23 + 128 + code, less 2^23 + 128 the code exactly (an FADD), times s exactly (an FMUL); no conversion op
__device__ __forceinline__ void decode4(uint32_t w, float s, float * x) {
    const uint32_t b = w ^ 0x80808080u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[i] = (__uint_as_float(__byte_perm(b, 0x4b000000u, 0x7650u | static_cast<uint32_t>(i))) - 8388736.0f) * s;
    }
}

} // namespace fn::qsa::kv8
