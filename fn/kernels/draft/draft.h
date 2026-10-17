#pragma once

// The draft block's own kernels on one card (DR1 Stage C; FC-DR1, FC-DR2): the draft input (X-DRAFT-IN: both halves
// of the normalized [e | h] row, exchanged) and the draft's dense attention (DA-TC: the q/k norms with IMROPE, the
// F16 K and V cache writes, attention over every cell, the sigmoid gate). The rest of the draft runs on existing
// kernels: eh_proj and the attention and FFN products on fn::dense, the HC mixes on fn::hc's Q8_0 forms, the experts
// on fn::experts, the head on fn::head (fn/model/draft.h assembles the step and the catch-up).
//
// Pinned source (qwen4exp.cpp build_mtp, lines 686-1000 at the REL2 engine): the draft row at position p takes token
// p's embedding and the target's layer-47 output row p - 1 (the hidden carry, FC-HEAD3):
//   h_norm[s] = rms_norm(h[s]) * hnorm[s] over each stream's 2,560 features; e_norm = rms_norm(e) * enorm, repeated
//   to the 4 streams; eh_proj's input row of stream s is the concatenation [e_norm | h_norm[s]] (5,120 values).
//   attention: q[h] = imrope(rms_norm(qfull[h][0..255]) * q_norm), k = imrope(rms_norm(kv_k) * k_norm), v = kv_v;
//   cache[cell] = (f16(k), f16(v)); o[h] = softmax_c(q[h] . K[c] / 16) . V[c] over cells 0 .. cell (dense, causal);
//   out[h] = o[h] * sigmoid(gate[h]) with gate[h] = qfull[h][256..511]. The catch-up (n_outputs 0, NOREAD) builds
//   only k, v and the cache writes.
//
// ---- X-DRAFT-IN (draft_in_send, draft_in_recv) ----
//
// Contract:
//   shapes:   rows in 1..5 (the step: 1; the catch-up: B_c; 5: SP2). Row i's hidden input is h_nextn when chained
//             (draft steps 2 .. d), else hidden row j = hsrc + i - 1 of the round: the retained carry when j < 0, else
//             the target's carry row j (FC-HEAD3's [B][4][1280], B <= 5, stride 5,120). Anything else is rejected
//             (false, nothing enqueued).
//   payload:  words_draft (6,408) a row: [0, 1280) e * enorm (own half), [1280 + 1280 s, +1280) h[s] * hnorm[s] (own
//             half), [6400, 6405) the own half's raw sums of squares (e, h0..h3), [6405, 6408) zero.
//   numerics: F32. A segment's sum of squares: a thread's 4 values by fmaf chain, a 32-lane xor butterfly, then warp
//             0's thread 0 adds the 10 warps' sums in warp order. The receiver adds card 0's sum and card 1's (the
//             same order on both cards), rs = 1 / sqrtf(ss / 2560 + 1e-6), and value = (x * g) * rs. Both cards form
//             identical rows (the same words in the same order). Bound (draft-kernel-test, FP64 reference of the
//             pinned order rms_norm(x) * g): max |error| / max |ref| <= 1e-6 a row; the codes bitwise act_ref of
//             the F32 row; both cards' outputs bitwise equal. No fast math (fn/CMakeLists.txt).
//   budget:   send: rows x 5 blocks of 320 threads (block k: segment k, a float4 a thread). recv: a block of 512
//             threads a row (the 1,602 peer groups polled into 25.6 KB of shared memory), 40 runs of 32 a warp.
//   floor:    one PCIe crossing of 25.6 KB a row (HX1: 1.98 us + 25.6 KB at 6.05 GB/s = 6.2 us at one block; the
//             rows x 5 sending blocks share it) [E], plus two launch ramps (1.95 us each, HX12) [M].
//   evidence: DR1 Stage C (draft-kernel-test, draft-test).
//
// The four steps (xchg.h X-DRAFT-IN):
//   1. writes:      each block of the peer's draft_in_send stores its segment's 1,280 words of each row with 16-byte
//                   stores, and its sum of squares word with one 4-byte store; block 0 of a row also the 3 zero words.
//   2. publication: none: each word is its own flag (A3).
//   3. acquire:     each 16-byte group of the slot has one reading thread in draft_in_recv's block of its row, which
//                   polls it with volatile loads and keeps the polled registers (in shared memory).
//   4. release:     that thread resets the group to the sentinel right after reading it.
//
// ---- DA-TC (datc_prepare, datc_attend) ----
//
// Contract:
//   shapes:   one KV head a card (F16 K and V caches [max_cells][256]), its 12 query heads; prepare: rows in 1..5
//             at cells n_before + row0 + t (t < rows), positions pos[plane * stride + row0 + t]; attend: the one row
//             row0 (the draft step), cells 0 .. n_before + row0. A cell at or past max_cells sets err_draft_depth
//             and writes nothing. Anything else is rejected (false, nothing enqueued).
//   numerics: prepare: F32 norms and rope (QA2's norm_rope, copied), K and V rounded to F16 (nearest even); q
//             scaled by the exact 1/16 and rounded to F16 (the pinned engine's F16 attention rounds q*scale so).
//             Attend: S = q . K by mma.m8n8k4 (exact F16 products, F32 accumulation; a quadpair sums 16 k-chunks, the four
//             quadpairs by xor 4 then 8); the online softmax in F32 (expf), P rounded to F16 and l the sum of the
//             rounded P; P V by mma.m8n8k4; the 144 CTAs' (O, m, l) folded in CTA order by each output lane
//             (w = expf(m_c - M)), o = O / L * (1 / (1 + expf(-gate))). Every order is fixed, so reruns are bitwise.
//             Bound (FC-DR1; NC1 section 3's attention bound for DA-TC): max |error| / max |ref| <= 2e-3 a head
//             against FP64 of the pinned math over the kernel's stored F16 inputs (q16, K, V). The figure against
//             the FP64 q includes q's own F16 rounding, which the prepare test checks. K within one F16 ulp of the
//             FP64 k; V bitwise f16(kv_v); codes bitwise act_ref of the F32 output.
//   budget:   prepare: a block of 416 threads a row (12 query warps and one K/V warp; the catch-up: K/V only), plus
//             one block that resets the released X-HC and X-HMIX slots when given. Attend: one cooperative launch of
//             144 CTAs x 256 threads (2 an SM), tiles of 64 cells split evenly over the CTAs; a CTA's 8 warps take 8
//             cells each for S and 32 dims each for P V; about 15.5 KB of static shared memory; then a grid barrier
//             and 96 warps for the fold.
//   floor:    K and V of n cells: 1,024 n bytes at 1,021.6 GB/s (HX12) + 1.95 us ramp + the grid barrier [A]; Stage A
//             section 2 prices it at the four depths.
//   sparse:   SP2 (attend_args::body): the same numerics over the logical cells the body lists (at most 2,055), in
//             the same tile order; sparse_cta (33) CTAs, the tile's physical cells staged in shared memory (one more
//             barrier a tile). Bound as above against FP64 over the listed cells (draft-kernel-test).
//   evidence: DR1 Stage A section 2 (the pricing), Stage C (draft-kernel-test, draft-test, draft-bench).
//
// Publication: prepare and attend are separate launches in stream order, so attend's reads of the new cells and the
// prepared q follow prepare's writes (kernel boundary). Attend's partials are published to the fold by the grid
// barrier and read with L2 (.cg) loads.

#include "fn/kernels/dense/dense.h"

#include <cstddef>
#include <cstdint>

namespace fn::draft {

constexpr int32_t max_rows = 5; // SP2: the catch-up's B_c up to 5 (a verify of four drafts)
constexpr int32_t half = 1280;     // a card's features of one stream
constexpr int32_t n_stream = 4;    // HC streams
constexpr int32_t in_width = 5120; // eh_proj's input a stream: [e | h_s]
constexpr int32_t in_runs = in_width / 32;
constexpr int32_t n_seg = 5;   // X-DRAFT-IN segments: e, h_0 .. h_3
constexpr int32_t n_head = 12; // query heads a card
constexpr int32_t head_dim = 256;
constexpr int32_t q_stride = 512;                               // qfull: [q | gate] a head
constexpr int32_t out_width = n_head * head_dim;                // 3,072
constexpr int32_t tile = 64;                                    // cells a DA-TC tile
constexpr int32_t n_cta = 144;                                  // DA-TC's grid: 2 CTAs on each of the 72 SMs
constexpr int32_t part_floats = n_head * head_dim + 2 * n_head; // a CTA's O[12][256], m[12], l[12]
// SP2 (sparse draft attention, temporary at SP2's disposition): the selection's shape (fn::qsa's ratio and top 512
// blocks), the chained steps that reuse one selection, and the sparse grid: every logical cell (4 x 512 selected, 3
// tail, 4 chain) in tiles of 64, one tile a CTA at the deepest
constexpr int32_t sel_ratio = 4;
constexpr int32_t sel_blocks = 512;
constexpr int32_t max_sparse_steps = 4;
constexpr int32_t sparse_cta = (sel_ratio * sel_blocks + sel_ratio - 1 + max_sparse_steps + tile - 1) / tile; // 33

// ---- X-DRAFT-IN ----

struct send_args {
    const float * emb; // F32 [rows][emb_stride]: the embed's output, floats 0 .. 1279 of a row the own half
    int32_t emb_stride;
    const float * carry;      // F32 [5][4][1280] the target's hidden carry (FC-HEAD3; SP2: five rows)
    const float * carry_prev; // F32 [4][1280] the retained carry (row -1 of the round's hidden)
    const float * h_nextn;    // F32 [4][1280] the draft's previous step output; used instead of the carry when chained
    const int32_t * hsrc;     // device: the round descriptor's hsrc
    bool chained;
    const float * g_e; // [1280] enorm, own half
    const float * g_h; // [4][1280] hnorm, own halves
    float * peer;      // the peer's receive slot (fn::xchg::rings::slot(peer, msg::draft, s))
    float * own;       // this card's copy of its payload [max_rows][6408]
};

struct recv_args {
    const float * own; // send_args::own
    float * slot;      // this card's receive slot (the same s)
    int8_t * codes;    // [rows][4][5120] eh_proj's activation input (act.h runs of 32)
    float * scales;    // [rows][4][160][2]
    float * f32;       // optional [rows][4][5120] the F32 rows (tests)
    uint32_t * err;    // this card's error word: err_poll_draft
    int32_t card;
    // the retained carry's update (the catch-up): carry_prev <- carry row hsrc + rows - 1; carry_prev null: none
    float * carry_prev;
    const float * carry;
    const int32_t * hsrc;
};

bool draft_in_send(int32_t rows, const send_args & a, void * stream);
bool draft_in_recv(int32_t rows, const recv_args & a, void * stream);

// ---- DA-TC ----

struct prepare_args {
    const float * qfull;      // F32 [rows][12][512], or null: K and V only (the catch-up)
    const float * kv_k;       // F32 [rows][256], before the norm
    const float * kv_v;       // F32 [rows][256]
    const float * q_norm;     // [256] attn_q_norm
    const float * k_norm;     // [256] attn_k_norm
    const int32_t * n_before; // device: cells before the round's rows
    const int32_t * pos;      // device: the round's rope planes [4][stride] (4 planes)
    const int32_t * stride;   // device: the round's rows (the descriptor's rows)
    int32_t row0;             // host: this launch's first row in the round
    float theta_scale;        // fn::qsa::attn::rope_theta_scale()
    uint16_t * k_cache;       // F16 [max_cells][256]
    uint16_t * v_cache;
    int32_t max_cells;
    uint16_t * q16;       // F16 [rows][12][256] q / 16 (attend's input); unused without qfull
    float * q_f32;        // optional [rows][12][256] the F32 q before the scale (tests)
    float * k_f32;        // optional [rows][256] the F32 k before rounding (tests)
    uint32_t * err;       // this card's error word: err_draft_depth
    float * release_hc;   // optional: an X-HC slot of this card reset to the sentinel (the catch-up's own slot)
    float * release_hmix; // optional: an X-HMIX slot of this card reset to the sentinels
};

struct attend_args {
    const uint16_t * q16;     // prepare's q16 of the row
    const float * qfull;      // F32 [12][512] the row's gates
    const uint16_t * k_cache; // F16 [max_cells][256]
    const uint16_t * v_cache;
    const int32_t * n_before; // device
    int32_t row0;             // host: the row's index in the round; its cell is n_before + row0
    int32_t max_cells;
    float * part;   // [n_cta][part_floats] scratch
    float * out;    // F32 [3072], gated
    int8_t * codes; // [3072] the activation format
    float * scales; // [96][2]
    uint32_t * err; // this card's error word: err_draft_depth
    // SP2 (sparse): fn::qsa::select's body of one row made at depth N - 1 with the query at N (q_ahead), N =
    // n_before: U32 K = min(N / 4, 512), the tail count, then K ascending block IDs. The row attends the K blocks'
    // cells, then cells 4 (N / 4) .. N + row0 in order (the prefix's tail and the chain), each once; row0 < 4. A K
    // that is not min(N / 4, 512) sets err_draft_depth and writes nothing. Null: every cell 0 .. N + row0 (dense).
    const uint32_t * body = nullptr;
};

size_t part_bytes();

// ---- eh_proj over B_c positions (Stage C amendment 1) ----
//
// eh_proj's rows are the 4 streams of each position, so B_c positions are 4 B_c rows, beyond fn::dense's 4. The
// adapter makes up to five (SP2) ordered calls of the existing 4-row product on stream: call b takes stream rows 4 b ..
// 4 b + 3 (x_codes, x_scales and mats[0].out offset by those rows) and is otherwise base. The calls share base.work,
// which stream order makes safe only because every call has the same work layout (dense::same_layout, checked before
// anything is enqueued) and each launch leaves its counters at zero. Each row's result is bitwise independent of the
// call it lands in (dense.h: splits depend on the shape only). base: k 5120, one matrix, X-HMIX not used (hmix null).
// False, with nothing enqueued, for positions outside 1..5 or a base the rule rejects.
bool eh_proj(int32_t positions, const fn::dense::product_args & base, void * stream);

bool datc_prepare(int32_t rows, const prepare_args & a, void * stream);
bool datc_attend(const attend_args & a, void * stream);

// SP2 (sparse): *depth = *n_before - 1, the depth of the step's select (fn::qsa::select_args::q_ahead), one thread
bool select_depth(const int32_t * n_before, int32_t * depth, void * stream);

struct resources {
    int32_t regs;
    int32_t local_bytes;
    int32_t shared_bytes;
    int32_t ctas_per_sm;
};
// 0: send, 1: recv, 2: prepare, 3: attend, 4: the sparse attend (SP2)
bool kernel_resources(int32_t which, resources & r);

} // namespace fn::draft
