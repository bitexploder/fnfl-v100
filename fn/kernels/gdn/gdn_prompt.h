#pragma once

// The GDN mixer's math over prompt rows (PB1; PK3's span, FC-PF3's FC-GDN1 row) on one card: between the input products
// (qkv, z, alpha, beta: fn::dense prompt launches) and ssm_out. A run of R rows (1 .. 4,096, a half) at consecutive
// positions reads logical version v_in of the recurrent state and the convolution history and leaves one version,
// v_out, at the run's end; nothing is written per row (FC-PF3).
//   gdn_prompt_conv:  the causal convolution with SiLU (gdn_conv.cuh's per-row arithmetic: one FMUL, three FMAs, x / (1
//                     + expf(-x))) over the run, from version v_in's 3-tap history; the run's last 3 inputs to v_out.
//                     PB3: the input F32 or the product's FP16 output (its values, as F32, are the history's too).
//                     PB3, fuse_norm: also prep's Q and K norms (bitwise the same values), the output written once.
//   gdn_prompt_prep:  the Q and K L2 norms (in place in the conv output; core's arithmetic: a lane's 4 values, the
//                     xor tree, rsqrt.approx) and per row and head the log decay g = softplus(alpha + dt) a and
//                     beta = sigmoid(beta logit) (core's forms).
//   gdn_prompt_recur: the gated delta rule in the chunked WY form, C = 64, in two launches (PB3). Per 64-row block,
//                     with G the block's cumulative log decay, A[t][s] = beta_t exp(G_t - G_s) (k_s . k_t) (s < t) and
//                     T = (I + A)^-1:
//                     prep (a CTA per block and head, all independent): W = T diag(beta exp(G)) K, U = T diag(beta) V,
//                       P[t][s] = exp(G_t - G_s) (q_t . k_s) (s <= t), exp(G) and exp(G_C - G_s) into the WY scratch;
//                       T by four 16 x 16 diagonal inverses, then the six off-diagonal blocks in three stages.
//                     scan (96 CTAs: 24 heads x 4 slices of 32 value rows, the state in registers, blocks in order):
//                       V' = U - W M^T, o = (diag(exp G) Q M^T + P V') / sqrt(128), M <- exp(G_C) M + (diag(exp(G_C -
//                       G)) V')^T K.
//   gdn_prompt_gate:  out_j = o_j / sqrt(sum o^2 / 128 + eps) ssm_norm[j] sigmoid(z_j) per row and head (z F32 or,
//                     PB3, the product's FP16 output), FP16 (the ssm_out operand; |x| > 65,504 sets bit 0 of *err)
//                     and optionally F32.
// Numerics: F32 state, solves, products and accumulators; expf, logf, sqrtf, divisions precise (no fast math; the
// norms' rsqrt and beta's sigmoid are core's approximate forms, so those per-row values are core's bitwise). Fixed
// orders: reruns are bitwise. Not batch-invariant (NC1 section 6). Check (gdp-test): the token-serial recurrence in
// FP64 from the stored inputs at a loose tolerance (PB3 Amendment 2: layout, indexing and range errors), over single
// runs, a 6-run chain and a 128-run chain (196,608 rows), slow-decay heads included.

#include <cstddef>
#include <cstdint>

namespace fn::gdn {

constexpr int32_t prompt_max_rows = 4096;
constexpr int32_t prompt_block    = 64;  // C

struct prompt_args {
    int32_t          rows     = 0;
    const float *    qkv      = nullptr;   // [rows][5120] the product's raw output
    // PB3: or the product's FP16 output, qkv16 [rows][ld16] (the convolution's input) and z16 at the same stride (the
    // gate's); each used when its F32 pointer is null
    const uint16_t * qkv16 = nullptr;
    const uint16_t * z16 = nullptr;
    int64_t ld16 = 0;
    float *          conv_out = nullptr;   // [rows][5120] post-SiLU; q and k normalized in place by prep
    const float *    conv_w   = nullptr;   // [5120][4]
    float *          hist     = nullptr;   // [5][5120][3]
    float *          state    = nullptr;   // [5][24][128][128]
    const uint32_t * cur      = nullptr;   // device: the committed slot
    int32_t          v_in = 0, v_out = 1;  // logical versions read and written
    const float *    alpha    = nullptr;   // [rows][24] raw
    const float *    beta_raw = nullptr;
    const float *    dt       = nullptr;   // [24]
    const float *    a        = nullptr;   // [24]
    float *          g        = nullptr;   // [rows][24] log decay
    float *          beta     = nullptr;   // [rows][24]
    float *          o        = nullptr;   // [rows][3072]
    // PB3: or o FP16, o16 [rows][3072] (each value rounded once; a magnitude past 65,504 sets error bit 0), used when o
    // is null; the scan writes it, the gated norm reads it
    uint16_t * o16 = nullptr;
    const float *    z        = nullptr;   // [rows][3072]
    const float *    norm_w   = nullptr;   // [128]
    uint16_t *       gated16  = nullptr;   // [rows][3072] FP16
    float *          gated    = nullptr;   // optional [rows][3072] F32
    uint32_t *       err      = nullptr;
    float *          wy       = nullptr;   // recur's scratch, prompt_wy_bytes(rows)

    bool fuse_norm = false; // PB3: conv normalizes q and k itself (one pass); prep then only gates
};

// the WY scratch of a run: per (block, head) W and U [64][128], P [64][64], exp(G) and the state update's weights [64]
// (PB3: then per (block, Q/K group) the scan's FP16 Q and K^T operands)
constexpr int64_t prompt_wy_item_floats = 2 * prompt_block * 128 + prompt_block * prompt_block + 2 * prompt_block;

int64_t prompt_wy_bytes(int32_t rows);

bool gdn_prompt_conv(const prompt_args & a, void * stream);
bool gdn_prompt_prep(const prompt_args & a, void * stream);
bool gdn_prompt_recur(const prompt_args & a, void * stream);
bool gdn_prompt_gate(const prompt_args & a, void * stream);

struct prompt_resources {
    int32_t regs, local_bytes, shared_bytes, dyn_shared_bytes;
};

// recur's two kernels: prep and scan
bool gdn_prompt_recur_resources(prompt_resources & prep, prompt_resources & scan);

}  // namespace fn::gdn
