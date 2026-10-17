#pragma once

// Dense products over prompt rows (PB1; FC-PF3's dense rows, FC-ACT1's FP16 prompt operands): R rows (1 .. 6,144)
// against one card-local weight matrix, read in its resident layout (DP2's packed layout, or EX1's down layout for the
// shared expert's down), so no weight is repacked or copied for the prompt.
//
// pd_product (quantized weights, tensor cores):
//   math:    out[r][n] = sum over the matrix's scale groups g of f[n][g] P[r][n][g] - m[n][g] S[r][g], with
//            P = sum_{k in g} a[r][k] code[n][k] and S = sum_{k in g} a[r][k] (the plan's scale-last form). The codes
//            are the format's integers (or IQ table values), converted exactly to FP16: an unsigned code u to
//            bitcast(0x6400 | u) - 1024, a signed code c to bitcast(0x6400 | (c + 128)) - 1152 (both exact: 1024 + u
//            and 1152 + c are FP16 integers). Q6_K takes q - 32, Q2_0 q - 1 as its signed code, so only Q4_K and Q5_K
//            have a min term m. f and m are the format's F32 group scales as fn/quant's decoder forms them (d x
//            sub-scale, dmin x min), so f x code - m is that decoder's F32 weight. Groups: 32 weights (16 for Q6_K).
//   numerics: a: FP16 [R][K] as the producer stored it (FC-PF3). P: mma.m8n8k4 FP16 x FP16 products (exact in F32)
//            accumulated in F32 over one group, then out += f P (one fmaf), then out -= m S (one fmaf); S summed in
//            F32 in K order. Groups in K order, so two runs are bitwise identical; rows are independent (no split
//            K). Reference (pd-test): FP64 on fn/quant's decoded weights and the same FP16 activation values; bound
//            max |err| / max |ref| <= 1e-5 per output row. The conversion is checked weight by weight against
//            fn/quant's decoder (pd-test codes): f x code - m in F32 equals the decoder's value bitwise.
//   formats: DP2's layout: Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q8_0 (K % 256 == 0); EX1's down layout (experts.h
//            pack_down, one expert): IQ4_NL, Q2_0 (K = 320, N = 2560). N % 32 == 0. Anything else is rejected.
//   budget:  256 threads (8 warps, a warp 32 columns: lane (qp, r) owns column 8 qp + r and unpacks it), a block 256
//            columns x a 64-row panel (8 tiles of 8 rows); the activation tile in shared memory (DP2: 128 K a stage,
//            two stages 32 KiB; EX1 down: 64 K a stage), swizzled by row so a quadpair's 8 row reads hit 8 bank groups.
//
// pd_convert and pd_tc (PB3, the prompt path's dense products; pd_product stays as the fallback):
//   pd_convert: a packed matrix (DP2's or EX1's down layout, the formats above) to FP16 [n][k] in K order, each weight
//            float(f x code) - m rounded once to FP16 (round to nearest even): the decoder's F32 weight (as pd_codes
//            checks it) rounded to FP16. The prompt converts each matrix once a chunk, at its layer, into the family
//            region, and both halves' products read it; no FP16 copy outlives the layer.
//   pd_tc:   out[r][n] = sum_k a[r][k] w16[n][k] by fn/kernels/tc's core (BN 128, 4 m-tiles a warp): FP16 operands,
//            F32 accumulators (Amendment 2), K in order; rows independent, reruns bitwise. Up to three column segments
//            write three outputs (the fused qkv and z, wq wk wv, shared gate and up), the optional row scale applied
//            after the sum. Check (pd-test): the conversion bitwise; the product against FP64 at a loose tolerance.
//
// pd_bf16 (BF16 weights, F32 SIMT): out[r][n] = sum_k x[r][k] W[n][k], x F32 [R][K], W BF16 GGUF rows widened
//            exactly; a thread's sum over each 32-K step in K order, then the steps' sums in order. Rows of W from up
//            to two base pointers (a card's resident half and its replica): row n of the logical matrix is row
//            (n / period) half + (n % period) % half of base[(n % period) / half]. Bound 1e-5 per output row.
// pd_glu: glu[r][c] = g / (1 + expf(-g)) x u, rounded to FP16 (precise expf). Built without fast math.

#include "fn/kernels/dense/dense.h"

#include <cstddef>
#include <cstdint>

namespace fn::dense {

constexpr int32_t pd_max_rows = 16384; // eh_proj: 4 stream rows of 4,096 positions (PB3's half)

enum class pd_layout : int32_t { dp2, ex1_down };

// The FP16 activation operand: row r, value k is a0[(r >> shift0) * ld0 + k] for k < k_split, else
// a1[(r >> shift1) * ld1 + k - k_split] (eh_proj's [e | h_s] row read from e [positions] and h [positions][4]
// without materializing it). One segment: k_split = K. Rows and k_split 16-byte aligned (multiples of 8 halves).
struct pd_act {
    const uint16_t * a0      = nullptr;
    int64_t          ld0     = 0;
    int32_t          shift0  = 0;
    const uint16_t * a1      = nullptr;
    int64_t          ld1     = 0;
    int32_t          shift1  = 0;
    int32_t          k_split = 0;
};

struct pd_args {
    fmt             f   = fmt::q4_k;
    pd_layout       lay = pd_layout::dp2;
    int32_t         k = 0, n = 0, rows = 0;
    const uint8_t * w     = nullptr;      // the resident packed matrix
    const uint8_t * grids = nullptr;      // grid_bytes (IQ3_S grid, IQ4 values)
    pd_act          x;
    float *         out       = nullptr;  // F32 [rows][ld_out]
    int64_t         ld_out    = 0;
    const float *   row_scale = nullptr;  // optional F32 [rows]: out[r][n] *= row_scale[r] (the shared down's sg)
};

bool pd_product(const pd_args & a, void * stream);

// pd_convert: a's f, lay, k, n, w and grids (rows, x and out unused) into wt FP16 [n][k] (16-byte aligned)
bool pd_convert(const pd_args & a, uint16_t * wt, void * stream);

// an output segment of pd_tc: columns [previous end, end) to out [rows][ld] (end a multiple of 8, ld of 2)
struct pt_seg {
    float * out = nullptr;
    int64_t ld  = 0;
    int32_t end = 0;
};

struct pt_args {
    const uint16_t * w = nullptr;  // FP16 [n][k] (pd_convert's)
    int32_t          n = 0, k = 0, rows = 0;
    pd_act           x;
    pt_seg           seg[3];
    int32_t          n_seg     = 1;
    const float *    row_scale = nullptr;  // optional F32 [rows], applied after the sum
    const float *    col_scale = nullptr;  // optional F32 [n], applied after the sum (pd_convert_bf16's 2^-e)
    // PB3, optional with one segment and no col_scale: the sum written as FP16 to out16 [rows][ld16] instead of F32 to
    // seg[0] (whose out may be null), rounded to nearest; a value beyond 65,504 or NaN sets err_bit in *err
    // (fn::xchg::ppack's arithmetic, so a P message's payload is written by its producer)
    uint16_t * out16 = nullptr;
    int64_t ld16 = 0;
    uint32_t * err = nullptr;
    uint32_t err_bit = 0;
};

bool pd_tc(const pt_args & a, void * stream);

// pd_convert_bf16 (PB3, the prompt's BF16 products: router, QSA index q and k, GDN alpha and beta; Amendment 2's one
// FP16 pass): w BF16 [n][k] (16-byte aligned, k a multiple of 8) to FP16 [n][k] for pd_tc, each row scaled by the
// power of two that brings its largest magnitude into [2^14, 2^15) (the range check: no row overflows or flushes for
// its range), col_s[row] = 2^-e for pd_tc's col_scale (e = 0 for a zero row). A warp a row.
bool pd_convert_bf16(const uint16_t * w, int32_t n, int32_t k, uint16_t * wt, float * col_s, void * stream);

struct pb_args {
    const uint16_t * w_lo   = nullptr;      // BF16 rows
    const uint16_t * w_hi   = nullptr;      // second base (null: w_lo)
    int32_t          period = 0, half = 0;  // row mapping (0: identity)
    int32_t          n = 0, k = 0, rows = 0;
    const float *    x      = nullptr;      // F32 [rows][ldx]
    int64_t          ldx    = 0;
    float *          out    = nullptr;      // F32 [rows][ld_out]
    int64_t          ld_out = 0;
};

bool pd_bf16(const pb_args & a, void * stream);

// glu[r][c] (FP16, [rows][ld_glu]) from gate and up F32 [rows][ld] (c < n)
bool pd_glu(int32_t       rows,
            int32_t       n,
            const float * gate,
            const float * up,
            int64_t       ld,
            uint16_t *    glu,
            int64_t       ld_glu,
            void *        stream);

// The router's epilogue over prompt rows (DP2's rule, dense.h): from logits [rows][512] and the shared gate's logit
// [rows] (pd_bf16 outputs; PB3: at gate_logit[r gate_ld], the router's tensor product's extra column), p = softmax (max, expf(l - max), the sum, p = e (1 / sum)), the 10 largest p descending with
// ties to the lower expert ID, w = p (1 / max(sum of the 10, 6.103515625e-5)), sg = 1 / (1 + expf(-g)). A warp a row.
bool pd_route(int32_t rows, const float * logits, const float * gate_logit, int32_t * ids, float * wts, float * sg,
              void * stream, int32_t gate_ld = 1);

// pd-test only: the prompt path's decode of every weight: code[n][k] (the FP16 code as F32), f[n][k] and m[n][k] (its
// group's scale and min), each value written at the K position the product pairs it with
bool pd_codes(fmt             f,
              pd_layout       lay,
              int32_t         k,
              int32_t         n,
              const uint8_t * w,
              const uint8_t * grids,
              float *         code,
              float *         scale,
              float *         mins,
              void *          stream);

// launch resources of pd_product for (f, lay) and pd_bf16 (f = bf16), for the Result
struct pd_resources {
    int32_t regs, local_bytes, shared_bytes, ctas_per_sm;
};

bool pd_kernel_resources(fmt f, pd_layout lay, pd_resources & r);
// pd_tc's kernel and pd_convert's for (f, lay)
bool pd_tc_resources(pd_resources & r);
bool pd_convert_resources(fmt f, pd_layout lay, pd_resources & r);

}  // namespace fn::dense
