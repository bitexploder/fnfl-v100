#pragma once

// The GDN convolution with SiLU as a producer epilogue (GD1 Amendment 2b, change 1). The convolution is depthwise:
// channel c's output needs only column c of qkv, its 3 history values and its 4 taps, which is exactly the column the
// GDN input product computes in one block. LP1 wires it into DP2's producer; fn::gdn::conv_silu (gdn.h) launches it
// alone for gd-test until then.
//
// Contract:
//   shape:    one thread a channel c in 0..5119 (card-local, gdn.h), rows B in 1..5 (a template parameter). The column
//             x[0..B-1] is finished: the product's complete value of each row, so it serves a single-pass block or a
//             split launch's last arriver alike.
//   use:      conv_column_load at the producer's start (the taps and history do not depend on the product), then
//             conv_silu_epilogue once the column is finished.
//   math:     qwen4exp.cpp build_conv_state_at (1968-2020) and ggml_ssm_conv at 66e11c3: y_b = SiLU(sum_k seq[b + k] w[k]),
//             seq = the history's 3 taps (oldest first) then the rows; SiLU(a) = a / (1 + expf(-a)). Writes history
//             versions 1..B of the channel: version v holds seq[v .. v + 2], the last 3 inputs after the first v rows.
//   versions: logical version v at physical slot fn::round::version_slot(cur, v) (SR1); the caller passes the committed
//             slot cur (0..5), read once from its device word.
//   numerics: F32; the taps in order, one FMUL then FMAs; expf and the division precise (the including file is built
//             without fast math: --ftz=false --prec-div=true --prec-sqrt=true --fmad=true). History versions are
//             copies, bitwise.
//   cost:     a channel reads 12 bytes of history and 16 of taps and writes 12 B bytes of versions: 0.33 MB a card at
//             B = 3 (Amendment 2b's +0.32 us at the stream rate [A]).
//   evidence: GD1 (gd-test conv cases against the FP64 reference).

#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/round/version.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::gdn {

struct conv_column {
    float h0, h1, h2; // history version 0, oldest first
    float4 w;         // the 4 taps, tap 0 oldest
};

// hist: [6][5120][3] (logical version 0, slot cur, read); conv_w: [5120][4], 16-byte aligned
__device__ __forceinline__ conv_column conv_column_load(const float * __restrict__ hist,
                                                        const float * __restrict__ conv_w, int32_t c, uint32_t cur) {
    const float * h = hist + fn::round::version_slot(cur, 0) * hist_version_floats + c * n_hist;
    conv_column cc;
    cc.h0 = h[0];
    cc.h1 = h[1];
    cc.h2 = h[2];
    cc.w = __ldg(reinterpret_cast<const float4 *>(conv_w) + c);
    return cc;
}

// x: the finished column (rows 0..B-1); y: its post-SiLU values; hist: [6][5120][3], logical versions 1..B of c
// written (slots version_slot(cur, 1..B))
template <int32_t B>
__device__ __forceinline__ void conv_silu_epilogue(const conv_column & cc, int32_t c, const float (&x)[B],
                                                   float * __restrict__ hist, uint32_t cur, float (&y)[B]) {
    static_assert(B >= 1 && B <= max_rows, "rows 1..5");
    float h0 = cc.h0, h1 = cc.h1, h2 = cc.h2;
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        float acc = h0 * cc.w.x;
        acc = fmaf(h1, cc.w.y, acc);
        acc = fmaf(h2, cc.w.z, acc);
        acc = fmaf(x[b], cc.w.w, acc);
        y[b] = acc / (1.0f + expf(-acc));
        h0 = h1;
        h1 = h2;
        h2 = x[b];
        float * hv = hist + fn::round::version_slot(cur, b + 1) * hist_version_floats + c * n_hist;
        hv[0] = h0;
        hv[1] = h1;
        hv[2] = h2;
    }
}

} // namespace fn::gdn
