#pragma once

// VIS1: the vision encoder's own kernels (FNFL's; the math adapted from ggml-vfast 66e11c3's qwen3vl.cpp graph and its
// CPU ops, never its kernels). The matrix products are the prompt path's pd_tc (fn/kernels/dense); these are the steps
// between them. Rows are patches in merge-group order (image.h); P patches, Pp = P rounded up to 64.
//
// Shared numerics: F32 arithmetic with the precise forms (no fast math: sqrtf, divisions, tanhf, exp2f and sincosf
// are IEEE or the CUDA library's accurate forms), FP16 outputs rounded to nearest even. Every FP16 operand is in range
// for any image by the load-time proof (weights.h range_check). References: fn/vision/vision_ref.cpp, FP64;
// tolerances in vision_test.cpp.
//
// pos_resize: x[p][c] = the 48 x 48 table resized bilinearly with aligned corners to the patch grid (gw x gh) and
//     permuted to merge order: for grid (gx, gy), sx = gx / ((gw - 1) / 47), x0 = floor(sx), dx = sx - x0 (clamped
//     taps), likewise y, value a (1 - dx)(1 - dy) + b dx (1 - dy) + c (1 - dx) dy + d dx dy (ggml's upscale, F32).
//     A thread a value. Bound 4e-5 x max |table| (the F32 coordinates, as in ggml; vision_test.h tol_pos).
// patch_embed: x[p][o] = (sum_k rows[p][k] w[o][k] + b[o]) + x[p][o], K = 768 in order, F32 FMA (SIMT, 64 x 64 a block,
//     4 x 4 a thread). Bound 1e-5 x the row's max |ref|.
// layernorm: y[r][c] = FP16((x - mean) rstd g + b), mean and var = mean (x - mean)^2 over 1152 (two passes),
//     rstd = 1 / sqrtf(var + eps). A warp a row. Bound 1e-3 x the row's max |ref| (FP16's half ulp is 4.9e-4).
// qkv_rope: from the qkv product rows (F32 [rows][3456], before bias) of image rows r0 .. r0 + rows: q and k plus bias
//     rotated by the vision rotary (head 72: dim i with i + 36 for i < 36; pairs 0-17 at the patch row y, 18-35 at the
//     column x; theta = pos 10000^(-(i % 18) / 18)), written FP16 to Q, K [16][Pp][72]; v plus bias written FP16
//     transposed to Vt [16][80][Pp] (dims 72-79 untouched). Rows from P to r0 + rows write zeros (so padded keys hold
//     zero values). Bound 1e-3 x the row's max |ref|.
// attention: O[p][72 h + d] = FP16(softmax_k(q_p . k_k / sqrt(72)) v_k[d]) over all P keys (no mask but the tail),
//     flash form: 64 queries a block (4 warps of 16), 64-key tiles double-buffered in shared memory (K [64][72], Vt
//     [80][64] chunk-swizzled), S = Q K^T and O += P V on mma.m8n8k4 (FP16 operands, F32 accumulators), an online
//     softmax in F32 (exp2 of the score x log2(e) / sqrt(72) less the running maximum), P rounded to FP16 through a
//     per-warp shared tile. Scores are never materialized beyond one 16 x 64 tile a warp. Bound 4e-3 x the row's max
//     |ref| against FP64 attention over the same FP16 Q, K, V.
// bias_residual: x[r][c] = (y[r][c] + b[c]) + x[r][c] (the reference's order). Bitwise to F32.
// bias_gelu: g[r][c] = FP16(gelu(y[r][c] + b[c])), the tanh form 0.5 v (1 + tanhf(0.79788456 v (1 + 0.044715 v^2))).
//     Bound 1e-3 relative to max |ref| of the row.
// bias_add: y[r][c] += b[c].

#include <cstdint>

namespace fn::vision {

struct rope_freqs {
    float f[18];  // 10000^(-k / 18), from F64
};

// the kernels' frequency table (a compile-time constant) and whether it is 10000^(-k / 18) rounded once from F64
rope_freqs make_rope_freqs();
bool       rope_table_exact();

bool k_pos_resize(const float * table, int32_t gw, int32_t gh, float * x, void * stream);
bool k_patch_embed(const float * rows, int32_t P, const float * w, const float * b, float * x, void * stream);
bool k_layernorm(const float * x, int32_t rows, const float * g, const float * b, float eps, uint16_t * y,
                 void * stream);
bool k_qkv_rope(const float * qkv, int32_t r0, int32_t rows, int32_t P, int32_t Pp, int32_t gw, const float * bias,
                uint16_t * q, uint16_t * k, uint16_t * vt, void * stream);
bool k_attention(const uint16_t * q, const uint16_t * k, const uint16_t * vt, int32_t P, int32_t Pp, uint16_t * o,
                 void * stream);
bool k_bias_residual(const float * y, int32_t rows, int32_t n, const float * b, float * x, void * stream);
bool k_bias_gelu(const float * y, int32_t rows, int32_t n, const float * b, uint16_t * g, void * stream);
bool k_bias_add(float * y, int32_t rows, int32_t n, const float * b, void * stream);

// the launch resources of each kernel (ptxas's view at run time), for the Result
struct k_resources {
    const char * name;
    int32_t      regs, local_bytes, shared_bytes, ctas_per_sm;
};

int32_t vision_kernel_resources(k_resources * out, int32_t max);

}  // namespace fn::vision
