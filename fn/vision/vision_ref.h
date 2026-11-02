#pragma once

// VIS1: FP64 CPU references of the vision encoder's steps and of the whole encoder (the semantics of ggml-vfast
// 66e11c3's qwen3vl.cpp graph, written down in VIS1's "Encoder semantics"). Inputs are the GPU's inputs (F32 or the
// FP16 values it stored), every operation is in F64 with exact math (the rotary angles, softmax and tanh from libm in
// F64), and BF16 weights are widened exactly. No GPU code.

#include "fn/vision/image.h"
#include "fn/vision/weights.h"

#include <cstdint>
#include <vector>

namespace fn::vision::ref {

// rows [P][1152] (merge order) of the position table resized to (gw, gh) with aligned corners
std::vector<double> pos_resize(const std::vector<float> & table, int32_t gw, int32_t gh);
// (rows . w + b) + pos for P rows of 768
std::vector<double> patch_embed(const float *               rows,
                                int32_t                     P,
                                const std::vector<float> &  w,
                                const std::vector<float> &  b,
                                const std::vector<double> & pos);
// LayerNorm over rows of n
std::vector<double> layernorm(const std::vector<double> & x,
                              int32_t                     rows,
                              int32_t                     n,
                              const std::vector<float> &  g,
                              const std::vector<float> &  b,
                              double                      eps);
// the vision rotary on one head's q or k row (72 values) in place, at grid row y and column x
void                rope(double * v, int32_t gy, int32_t gx);
// non-causal attention of one head: q, k, v [P][72], out [P][72], scale 1 / sqrt(72)
void                attention(const double * q, const double * k, const double * v, int32_t P, double * out);
double              gelu(double v);
// y[r][n] = sum_k W[n][k] x[r][k] (+ b[n]) for BF16 W [n][k]
std::vector<double> product(const std::vector<double> &   x,
                            int32_t                       rows,
                            int32_t                       k,
                            const std::vector<uint16_t> & W,
                            int32_t                       n,
                            const std::vector<float> &    b,
                            int32_t                       threads);

// one block over x [gw gh][1152] in place
void block(const block_host & b, std::vector<double> & x, int32_t gw, int32_t gh, double eps, int32_t threads);
// post_ln, the merge and the merger MLP: [P / 4][2560]
std::vector<double> merger(const host_weights &        w,
                           const std::vector<double> & x,
                           int32_t                     P,
                           double                      eps,
                           int32_t                     threads);
// FP16 storage emulation (off by default): when on, block, merger and attention round to FP16 where the GPU stores
// FP16 (LayerNorm outputs, q and k after the rotary, v, P for the PV product, the attention output, GELU outputs),
// and keep F64 everywhere else. Its distance from exact FP64 is what FP16 storage alone costs. Not thread safe.
void                set_fp16_storage(bool on);
// the whole encoder: rows [P / 4][2560] for one prepared image
std::vector<double> encoder(const host_weights & w, const prepared & img, int32_t threads);

}  // namespace fn::vision::ref
