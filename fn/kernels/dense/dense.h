#pragma once

// Dense quantized products at decode rows (DP1, DP2): B rows against one weight matrix, every weight decoded once and
// dotted with all B rows. One launch per input: up to 6 matrices that read the same activation, each with its own
// format, row count, split and output. Simulator families gdn_proj, gdn_out, qsa_build, qsa_out, qsa_index, route,
// vocab and the shared expert's products (seed graphs G, Q, F, V); fnfl/design/dense-requirements.md (v0).
//
// Contract (product):
//   shapes:  rows B in {1, 2, 3, 4, 5} (one kernel instance each; SP2: B = 5 at K <= 3,072, the eh_proj's K 5,120 at
//            B <= 4); K the same for every matrix of a launch, K <= 5120
//            and a split of at most max_split parts; quantized formats: M % 8 == 0; Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S:
//            K % 256 == 0; Q8_0, IQ4_NL, Q2_0: K % 64 == 0; BF16 and F32: K % 256 == 0, any M. Per card the model
//            runs: GDN input [2560 -> 5120, 3072, 24, 24], QSA input [2560 -> 6144, 256, 256, 512, 128], ssm_out and
//            attn_output [3072 -> 2560], shared gate/up [2560 -> 320] and down [320 -> 2560], vocabulary
//            [2560 -> 124160], router [2560 -> 512, 1] (both cards, XP1's V2); the draft block (DR1, Q8_0 unless
//            named): eh_proj [5120 -> 1280] (B_eff = 4 B stream rows: a caller with more than 4 runs launches of at
//            most 4, each row's result being independent of B), attention input [2560 -> 6144, 256, 256],
//            attn_output [3072 -> 2560], the shared expert's products as the target's, the vocabulary subset
//            [2560 -> 32768] (Q5_K, head.h) and the F32 router [2560 -> 512, 1]. Anything else that breaks a rule
//            above, a format outside the list, a null or misaligned pointer, or a bad split is rejected (returns
//            false and prints the reason, nothing enqueued).
//   formats: Q6_K, Q5_K, Q4_K, IQ4_XS, IQ3_S, Q8_0, IQ4_NL, Q2_0 (activations in FNFL's decode format, act.h), BF16 and
//            F32 (activations F32): the dense formats the deployed model and its draft (DR1) use.
//   math:    out[b][n] = sum_k W[n][k] x_b[k], with x_b row b's values code x d (quantized formats) or F32 values
//            (BF16); with mat.scale, out[b][n] = scale[b] x that (the shared down: sigmoid of the shared gate, R2 F7).
//            Shared expert (epi glu_gate / glu_up, a pair of matrices with equal M and split): glu[b][c] =
//            g / (1 + exp(-g)) * u for gate row g and up row u of channel c, written in FNFL's decode format.
//            GDN input with the convolution (epi conv_silu, LP1): the matrix's 5,120 output columns are the GDN
//            channels (fn/kernels/gdn/gdn.h); each finished column goes through fn::gdn::conv_silu_epilogue (the
//            depthwise convolution over the history and the B rows, then SiLU), which writes out = the post-SiLU values
//            and history versions 1..B (conv.hist, logical versions at slots fn::round::version_slot(*conv.cur, v),
//            SR1) instead of the products. The column is unsplit.
//            Router (route.logits, route.gate set; BF16 matrices for the target, F32 for the draft, DR1): after both
//            matrices' outputs, the launch's last block to finish computes, warp b for row b,
//            p = softmax(logits[512]) (ggml's rule: max, expf(l - max), sum, p = e x (1 / sum)), the 10 largest p in
//            descending order with ties to the lower expert ID, w_k = p_k x (1 / max(sum_k p_k, 6.103515625e-5)) and
//            gate = 1 / (1 + expf(-sg)) (topk-moe.cu, llama-graph.cpp build_moe_ffn and qwen4exp.cpp at 66e11c3; the
//            model sets no expert weights scale).
//   numerics: weights as stored (pack permutes bytes within 8-row tiles, IQ4 nibbles within 32-weight blocks and Q2_0
//            fields within 16-weight groups); integer dot products (IDP4A) exact; Q2_0's -1 offset is the exact integer
//            sum of the run's codes; F32 scales, products and sums, built without fast math (--ftz=false
//            --prec-div=true --prec-sqrt=true --fmad=true); expf and the format's divisions are the precise forms. A
//            warp owns whole output rows: each lane sums its K positions in order, then fixed shuffle trees, and a K
//            split (several warps per row) sums its parts in part order, so two runs are bitwise identical. Splits and
//            tiles depend on the matrix shape only, never on B, so each row's result is bitwise independent of B (NC1
//            section 2). Reference: FP64 on fn/quant's decoded weights and act_ref's activation values (or the F32
//            values); bound max |err| / max |ref| <= 1e-5 per output vector; the GLU's codes and scales bitwise equal
//            to fn/quant/act_ref of the kernel's own F32 GLU output; the router's IDs, weights and gate as DP2's
//            Amendment 1 states.
//   floor:   cold weights, 1,380 MHz: 1.949 us + bytes / 1,021.6 GB/s [M, HX12], plus 0.31 us (single pass) or 0.83 us
//            (a launch that finishes across blocks: a split, a glu pair or the router) [M, HX12]; gate 90 % (DP2).
//   budget:  72 blocks of 512 threads, __launch_bounds__(512, 1): at most 128 registers, no spills; dynamic shared
//            memory B x 1.5 K bytes of staged activations (+2,048 B for the IQ3_S grid, +B x 2,560 B for the router's
//            epilogue), at most 32,816 B [C] for the target's launches (the router's BF16 launch: 10,240 B at B = 4);
//            K 5,120 at B = 4 stages 30,720 B, and no launch the contract takes exceeds 43,008 B (dense.cu asserts it
//            under the 48 KB default). SP2: a B = 5 launch takes K <= 3,072 and stages at most 23,040 B plus the
//            grid and 12,800 B of router rows [C]. One step of loads ahead per lane (18-68 weight bytes a lane),
//            refilled before the step's FMAs. Issue: at most 0.389 warp instructions per weight byte, 0.194 on the
//            FMA pipe (sheet).
//   evidence: HX4 (borrowed kernel costs), HX5 (instruction counts), HX12 (machine sheet), HC1-HC1c and QA1 (bytes in
//            flight, data ahead of the stream), EX1 (activation loads), the dense sheet v0, DP1 and DP2 (measured).
//
// Packed weight layout (pack): rows in tiles of 8 (BF16 and F32: unchanged GGUF rows, in tiles of 4 for the kernel);
// a tile is the 8 rows' bytes, step by step over K (256 weights a step), and within a step the 32 lanes' 16-byte loads
// are contiguous (lane = 4 r + q: row r of the tile, q the quarter of the step's 256 weights). dense.cu lists each
// format's segments. Activations: x codes int8 [B][K] and scales float2 [B][K / 32] (act.h; codes 16-byte, scales
// 8-byte aligned), x_f32 F32 [B][K] (BF16, F32); out F32 [B][M]; glu codes int8 [B][M] and scales float2
// [B][M / 32]; glu_f32 (test only, nullable) F32 [B][M]; route ids int32 [B][10], wts F32 [B][10] and gate F32 [B]
// (EX1b's ids and wts; gate is the shared down's scale).
// work: partial sums, then arrival counters, then (SP2) the fifth row's partial sums, at offsets set by the launch's
// layout (layout_of). The partials of rows 0..3 keep their [8 rows][4 B] slots, so a B <= 4 launch's offsets are
// unchanged; row 4's partials are 8 floats a (tile, part) in the same order, a region a B <= 4 launch never touches.
// Zero it once before
// the first launch. Every launch leaves its own counters at zero, but its partial sums stay where a launch of another
// layout keeps counters, so a region zeroed once may be reused only by launches with the same layout (same_layout: the
// same matrices' tiles, splits, epilogues and counter mapping). A launch of another layout needs its own region, or
// the region zeroed again first. The region is its caller's private scratch (FC-D2, FC-D3).

#include <cstddef>
#include <cstdint>

namespace fn::dense {

enum class fmt : int32_t { q4_k, q5_k, q6_k, iq4_xs, iq3_s, q8_0, iq4_nl, q2_0, bf16, f32 };
enum class epi : int32_t { f32, glu_gate, glu_up, conv_silu };

constexpr int32_t max_rows = 5; // SP2: the five-row verification pass
constexpr int32_t max_mats = 6;
constexpr int32_t max_k = 5120;        // the draft's eh_proj (DR1); the target's largest K is 3,072
constexpr int32_t max_split = 12;      // K parts of a split: the target's most steps, so a sum of parts stays 12 loads
constexpr int32_t route_experts = 512; // the router's logits a row
constexpr int32_t route_used = 10;     // experts selected a row

// GGUF bytes of one row of k weights, 0 for a format or k the kernel does not take
int64_t row_bytes(fmt f, int32_t k);

// Load-time repack of an M x K matrix: src holds M GGUF rows, src_row_bytes apart; dst gets M x row_bytes(f, k) bytes
// (the same count, every value kept). BF16 and F32 are copied unchanged.
bool pack(fmt f, int32_t m, int32_t k, const uint8_t * src, size_t src_row_bytes, uint8_t * dst);

// the tables' buffer: iq3s_grid (512 x 4 B), then kvalues_iq4nl (16 B)
constexpr size_t grid_bytes = 2048 + 16;
bool grid_table(const void * iq3s_grid, size_t iq3s_bytes, const void * kvalues, size_t kvalues_bytes, uint8_t * dst);

struct mat {
    fmt f = fmt::q4_k;
    int32_t m = 0;               // output rows
    const uint8_t * w = nullptr; // packed, 16-byte aligned
    float * out = nullptr;       // F32 [B][m] (epi f32), else unused
    epi e = epi::f32;
    int32_t pair = -1;             // glu_gate: the index of its glu_up matrix, and the reverse
    int32_t split = 0;             // K parts (warps per row group); 0: choose_splits picks it
    const float * scale = nullptr; // epi f32: F32 [B] multiplies row b's output (the shared down's gate), or null
};

// the GDN convolution epilogue (epi conv_silu, at most one matrix a launch, M 5,120, split 1): taps [5120][4] F32
// (ssm_conv1d, 16-byte aligned), history [5][5120][3] F32 (logical version 0 read, versions 1..B written; gdn.h's
// layout), and the device word naming the committed slot (SR1)
struct conv_args {
    const float * w = nullptr;
    float * hist = nullptr;
    const uint32_t * cur = nullptr;
};

// the router's epilogue: logits and gate name two epi f32 matrices of the launch (M 512 and 1, unsplit or split)
struct route_args {
    int32_t logits = -1;
    int32_t gate = -1;
    int32_t * ids = nullptr; // [B][10]
    float * wts = nullptr;   // [B][10]
    float * sg = nullptr;    // [B] sigmoid of the shared gate
};

struct product_args {
    int32_t k = 0;
    int32_t n_mats = 0;
    mat mats[max_mats];
    const int8_t * x_codes = nullptr; // [B][k], for the quantized formats (act.h)
    const float * x_scales = nullptr; // [B][k / 32][2]
    const float * x_f32 = nullptr;    // F32 [B][k], for BF16 and F32
    const uint8_t * grids = nullptr;  // grid_bytes, for IQ3_S, IQ4_XS and IQ4_NL
    int8_t * glu_codes = nullptr;     // [B][m] of the glu pair (act.h)
    float * glu_scales = nullptr;     // [B][m / 32][2]
    float * glu_f32 = nullptr;        // F32 [B][m], test only
    route_args route;
    conv_args conv;        // for an epi conv_silu matrix
    void * work = nullptr; // work_bytes(), 16-byte aligned, zeroed once
    // HC3, X-HMIX's acquire (fn/xchg/xchg.h): non-null hmix is this card's X-HMIX slot of the sublayer, and x_codes,
    // x_scales and x_f32 are taken from its regions (leave them null). Every block first prefetches its warps' first
    // weights into L2, then polls the peer's half of the codes and scales it stages and, when a matrix is BF16, of the
    // F32 row. Null: the activations were written before the launch (no acquire).
    const float * hmix = nullptr;
    int32_t hmix_peer_half = 0; // the half the peer writes: 1 - card
    uint32_t * err = nullptr;   // the card's error word (xchg::err_poll_hmix)
};

// Sets every split left at 0: the largest divisor of a matrix's K steps that keeps the launch's units within one
// round of the card's 1,152 warps, 1 when the launch already fills them. A glu pair gets one split; a conv_silu
// matrix always 1.
void choose_splits(product_args & a);

// The work region's layout of a launch (its signature): per matrix in launch order, the tiles, split, epilogue, pair
// and where its partial sums and counters sit; then the totals and the router's counter. Splits must be set. Pointers
// and formats are not part of it (formats of one tile size tile and split alike).
struct mat_layout {
    int32_t m = 0;
    int32_t tile_rows = 0;
    int32_t tiles = 0;
    int32_t split = 0;
    epi e = epi::f32;
    int32_t pair = -1;
    int32_t part_off = -1; // floats into the partials, -1: none (SP2: row 4's at part_off / 4 in their region)
    int32_t cnt_off = -1;  // words into the counters (glu_up: its gate's), -1: none
};
struct work_layout {
    int32_t k = 0;
    int32_t n_mats = 0;
    mat_layout mats[max_mats];
    int32_t part_floats = 0;
    int32_t cnt_words = 0;    // the matrices' counters, after the partials
    int32_t route_cnt = -1;   // the router's blocks' counter, a word after them, -1: none
    int32_t part4_floats = 0; // SP2: row 4's partials (part_floats / 4), after the counters (and the router's)
};
// false (out cleared) for a matrix count, format, row count or split work_bytes cannot size
bool layout_of(const product_args & a, work_layout & out);
bool same_layout(const work_layout & x, const work_layout & y);

// bytes of work a launch needs (0 when no matrix is split and there is no glu pair or router); splits must be set
size_t work_bytes(const product_args & a);

// Enqueue on stream (a cudaStream_t). Return false, with nothing enqueued, for any input the contract rejects; false
// after a failed launch submission (the CUDA error is left for the caller).
bool product(int32_t rows, const product_args & a, void * stream);

const char * fmt_name(fmt f);

// FMS2 (build item 2, the joint catch-up's eh_proj): product(4, a), whose blocks all return at once, writing nothing,
// when the device word *gate <= at (read on the device, so a captured graph holds the launch whatever the round's
// rows): the lone launch's device code otherwise, so its outputs are product(4, a)'s bitwise. Four rows, no router
// and no F32 matrix (else false, with nothing enqueued).
bool product_gated(int32_t rows, const product_args & a, const int32_t * gate, int32_t at, void * stream);

// FMU2, the joint product: U lanes (2 .. max_lanes) of lane_rows rows in one launch that reads every weight once. Row
// b = lane_rows u + i is lane u's row i. Each row's arithmetic is the lone launch's at B = 5 (the same per-lane dot
// products, shuffle trees and part order; splits and tiles from the shape), so every output of lane u is bitwise equal
// to product(lane_rows, ...) on lane u's rows alone (dense_joint_test). Matrices, formats, K and the epilogues as
// product at B = 5 (K <= 3,072), except F32 weights (the draft's router), which the joint launch rejects. Every
// per-row buffer of product_args (x codes, scales and F32 values, outputs, the GLU's codes, scales and F32 values,
// the router's ids, weights and gate, a matrix's scale) holds the U lane_rows rows contiguously, lane u's at row
// lane_rows u. Per lane: X-HMIX's slot (lane u's at a.hmix + u hmix_lane_floats, its rows 0 .. 4) and the convolution's
// history and committed slot (conv[u]; the taps conv[0].w; a.conv is ignored). work: joint_work_bytes (its own
// layout, joint_layout_of), zeroed once, private to joint launches of the same layout (any U). Dynamic shared memory:
// U x 5 x 1.5 K bytes of staged activations (92,160 B at U = 4, K 3,072) plus the IQ3_S grid, or the router's 20 rows
// (51,200 B) when larger, above the 48 KB default (the launch sets the kernel's opt-in attribute).
// FMS13: joint_args::rows = R (2 .. lane_rows, the round's d + 1) rows a lane, every lane alike. The buffers keep the
// lane_rows stride (lane u's rows at row lane_rows u), and only rows 0 .. R - 1 of each lane are read or written; each
// row's arithmetic is the lone launch's at B = R (describe(R): its splits and checks), so with the non-bitwise switches
// off every output of lane u is bitwise product(R, ...) on lane u's rows. Inside the kernel the U R rows are packed
// (row b = R u + i), and the forms are keyed on U R: the integer step at up to 10 rows, FP16 (dense_f16) at 12 or more,
// BF16's shared wide form at up to 15 and its per-lane form above.
constexpr int32_t lane_rows = max_rows;
constexpr int32_t max_lanes = 4;
constexpr int32_t max_joint_rows = lane_rows * max_lanes;
constexpr int32_t max_lane_mats = 2; // matrices with lane outputs (joint_args::out)

struct joint_args {
    int32_t lanes = 0;
    int32_t       rows             = lane_rows;  // FMS13: R, every lane's rows (2 .. lane_rows)
    int32_t hmix_lane_floats = 0; // X-HMIX: from lane u's slot to lane u + 1's (a multiple of 4)
    conv_args conv[max_lanes];
    // lane outputs: F32-epilogue matrix i's lane u rows go to out[i][u] (lane_rows rows of m, 16-byte aligned;
    // mats[i].out is ignored), for the head's vocabulary and PLE's two products; all null: each matrix's U lane_rows
    // rows of mats[i].out
    float * out[max_lane_mats][max_lanes] = {};
    // per-lane F32 activations (PLE, FMU2 build item 4): lane u's lane_rows rows of K at x_f32[u] (16-byte aligned;
    // BF16 matrices only, a.x_f32 ignored); all null: a.x_f32's (or X-HMIX's) rows
    const float * x_f32[max_lanes] = {};
    // FMS2 item 6 (XC-1): X-A sent by the producer. Non-null: matrix 0 (the only matrix: F32 epilogue, split > 1, m =
    // 2 words_half, no lane outputs) also stores each finished value of its rows' half peer_half, as its tile's last
    // split arrives, to peer: canonical words, row b at b words_half (the peer's joint slot, as send_half_joint)
    uint32_t * peer = nullptr;
    int32_t peer_half = 0;
    // FMS2 item 7: non-null with peer, the send is 16-bit (xchg.h send_half_joint16's form: FP16 halves, row b at b
    // x16_row_halves halves) and a value beyond FP16's range sets err_x16_range here (the sending card's joint word)
    uint32_t * x16_err = nullptr;
    // FMS5 (engine_config::joint_router_tc; ends at FMS5's disposition): the router launch (route set, the target's
    // BF16 logits [2560 -> 512] and shared gate [2560 -> 1], no lane outputs, no peer) on the tensor cores. Not
    // bitwise the lone launch: each logit within 2^-14 x sum |w x| of FP64 on the same BF16 weights and F32
    // activations (dense_joint_test), the IDs, weights and gate then by the lone epilogue on those logits. Contract:
    //   weights: BF16 bits to F32 (exact), then FP16 round to nearest in registers (exact for 2^-14 <= |w| <= 65,504;
    //            the loader counts the rest); activations x as hi = f16(x) and lo = f16(x - hi) (about 22 bits;
    //            |x| < 65,504 assumed); products W hi + W lo by mma.sync m8n8k4 (HMMA.884) with F32 accumulators.
    //   layout:  72 blocks, 9 row groups (8 of 64 logit rows, then the gate's row) x 8 K slices of 320; warp w takes
    //            8 rows (m-tile w % 8) and K half w / 8 of its block's slice, the U x 5 rows padded to 8s on the N
    //            side; the quadpairs' K chunks summed by xor 4 then 8, the halves in shared memory (half 0 + half 1);
    //            each block's partials [slice][24 rows][576] F32 to work, the last of a group's 8 blocks summing the
    //            slices in slice order. Deterministic: two runs are bitwise identical.
    //   work:    joint_work_bytes reserves the group counters and partials after the joint layout's words.
    bool router_tc = false;
    // FMS6 (engine_config::joint_dense_f16; ends at FMS6's disposition): the quantized matrices at U = 3 and 4 in FP16
    // on the tensor cores (U = 2, the BF16 matrices and the epilogues as without it). Not bitwise the lone launch:
    // each output within 2^-8 x sum |w| |x| of FP64 on the same decoded weights and activations (code x d), with
    // |c f| + |m| for |w| in Q4_K and Q5_K (dense_joint_test). Contract:
    //   weights: the code's exact integer in FP16 (Q6_K: c - 32), times the part's scale RN16(f) (Q6_K: RN16(d sc)),
    //            Q4_K and Q5_K c RN16(f) - RN16(m) in one HFMA2, one rounding each;
    //   activations: the staged code s exactly in FP16, times RN16(d) of its 32-run, one rounding;
    //   products by mma.sync m8n8k4 (HMMA.884) with F32 accumulators carried over the unit's whole K, the quarters
    //   summed by xor 4 then 8; layout, splits and part order as without it. Deterministic.
    bool dense_f16 = false;
};

// The joint launch's work layout, the same at every U: lone layout_of's counters and router counter; a split or glu
// matrix's partial sums [tile][part][8 rows][max_joint_rows] (no row-4 region)
bool joint_layout_of(const product_args & a, work_layout & out);
size_t joint_work_bytes(const product_args & a);
bool product_joint(const product_args & a, const joint_args & j, void * stream);

} // namespace fn::dense
