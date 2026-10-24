#pragma once

// QSA over prompt rows (PB1; FC-PF3's FC-QSA1 and FC-QSA2 rows), on one card. Rows of a half at positions pos0 .. pos0 +
// rows - 1 (rows <= 4,096), every cell's state written before the rows that read it run.
//
// The index (index owner; qsa.cu, built with -use_fast_math as the decode scorer):
//   qsp_keys:    the raw index keys bf16(ik) (round to nearest even) at the rows' cells.
//   summaries:   QA1b's cache-form summary_update over the blocks the half completes ([pos0 / 4, (pos0 + rows) / 4)), the
//                decode path's pooling bit for bit.
//   qsp_queries: each row's 4 index queries normed and roped at its position: the decode scorer's norm_rope, bitwise.
//   qsp_score:   slabs of up to 128 rows: score[t][b] = sum_h relu(q'[t][h] . summary[b]) (each head an fmaf chain in
//                element order, the heads ((h0 + h1) + h2) + h3, as the decode scorer), its selection key for every
//                block row t sees whole (else 0) and each 32-block tile's best two, in qsa.cu's workspace layout
//                ([128][max_blocks] keys: 32 MiB).
//   qsp_select:  the decode select's row algorithm (1,024 threads a row, launch 2 of select, unchanged) on the slab:
//                each row's top 512 visible blocks (ties: the lower block), ascending, then its tail cells, as
//                qsa.h's body, the slab's rows packed with their byte offsets. Bitwise the decode path's selection on
//                the same summaries and queries.
// The bodies of a half: slab k's at body + k * 128 * 517 words, its offsets at offsets + 129 k.
//
// The attention (both cards; qsa_prompt_attn.cu, no fast math):
//   qsp_kv:      per row, k = imrope(rms_norm(kv_k) k_norm, pos), v = kv_v, written BF16 (round to nearest even) at the
//                row's cell: QA2's prepare arithmetic (attend.cu's norm_rope with all four rope planes the position).
//   qsp_attend:  a CTA a row: q[h] = imrope(rms_norm(qfull[h][0..255]) q_norm, pos) / 16, attention over the row's
//                selected cells (its blocks' 4 cells each, then its tail; all written: K/V of the half's rows are
//                written before), out = o sigmoid(gate), F32 and FP16 (|x| > 65,504 sets bit 0 of *err). PB3: on the
//                tensor cores in tiles of 64 cells (Amendment 2): q / 16 and the cache's BF16 K and V rounded once to
//                FP16, S = Q K^T and O = P V by HMMA.884 with F32 accumulators, the online softmax F32 (expf), P
//                rounded to FP16 (the sum l of the rounded P). Check (qsp-test): against FP64 at a loose bug-check
//                tolerance for selection, mask and layout errors.
// KV8 (--fnfl-kv-format q8; k_scale and v_scale set): qsp_kv writes the 8-bit format (fn/kernels/qsa/kv_q8.cuh,
// attend.h's prepare rule: codes [cell][256], FP16 scales [cell][kv8::scale_stride]) and qsp_attend's operand is
// code x s rounded once to FP16 (nearest even) where it rounds BF16 once today: formed in FP16 pairs (the biased
// code under FP16's exponent, less the bias, times the scale; bit for bit the exact F32 product rounded by F2F), the
// tile's scales staged in shared memory with its cells. Check: the same bound
// against FP64 over the decoded cache, the written codes and scales bitwise the CPU quantizer of the FP64 k's F32
// (within its rounding) and of v.

#include "fn/kernels/qsa/qsa.h"

#include <cstddef>
#include <cstdint>

namespace fn::qsa {

constexpr int32_t prompt_slab       = 128;
constexpr int32_t prompt_max_rows = 4096;
constexpr int32_t prompt_slabs      = prompt_max_rows / prompt_slab;
constexpr int64_t prompt_slab_words = int64_t(prompt_slab) * max_row_words;
// the SEL message's payload: the half's bodies, then the slabs' offsets (I32 [12][129])
constexpr int64_t prompt_body_words = prompt_slabs * prompt_slab_words;
constexpr int64_t prompt_sel_bytes  = prompt_body_words * 4 + int64_t(prompt_slabs) * (prompt_slab + 1) * 4;

struct prompt_ws {
    uint32_t * keys  = nullptr;  // [128][max_blocks]
    uint32_t * best1 = nullptr;  // [128][max_tiles]
    uint32_t * best2 = nullptr;
    uint16_t * lanes = nullptr;
    uint16_t * q16 = nullptr;    // [128][4][128] the slab's prepared queries, FP16, each scaled by a power of two
    float * qs = nullptr;        // [128][4] the scales' inverses (PB3: the tensor scorer's operands)
    int32_t *  depth = nullptr;  // [prompt_slabs] the slabs' first positions (select reads its depth on the device)
    uint32_t * stats = nullptr;  // [128][n_stats]
    uint32_t * error = nullptr;  // select's self-check word
};

size_t    prompt_ws_bytes(int32_t max_cells);
prompt_ws prompt_carve(void * base, int32_t max_cells);

bool qsp_keys(int32_t rows, int32_t pos0, const float * ik, uint16_t * keys, int32_t max_cells, void * stream);

struct qsp_index_args {
    const float * summaries = nullptr;
    const float * iq        = nullptr;  // [rows][4][128] the half's index queries (before norm and rope)
    const float * q_norm    = nullptr;
    rope_params   rope      = {};
    int32_t       rows = 0, pos0 = 0, max_cells = 0;
    prompt_ws     ws;
    uint32_t *    body    = nullptr;  // the half's bodies (layout above)
    int32_t *     offsets = nullptr;  // [prompt_slabs][129]
};

// score and select every slab of the half
bool qsp_select(const qsp_index_args & a, void * stream);

struct qsp_kv_args {
    const float * kv_k   = nullptr;  // [rows][256]
    const float * kv_v   = nullptr;
    const float * k_norm = nullptr;
    int32_t       rows = 0, pos0 = 0, max_cells = 0;
    uint16_t *    k_cache = nullptr;
    uint16_t *    v_cache = nullptr;
    uint16_t * k_scale = nullptr; // KV8: both set, the caches are q8 (int8 codes) and these the layer's scales
    uint16_t * v_scale = nullptr;
};

bool qsp_kv(const qsp_kv_args & a, void * stream);

struct qsp_attend_args {
    const float *    qfull   = nullptr;  // [rows][12][512]
    const float *    q_norm  = nullptr;
    const uint32_t * body    = nullptr;
    const int32_t *  offsets = nullptr;
    const uint16_t * k_cache = nullptr;
    const uint16_t * v_cache = nullptr;
    const uint16_t * k_scale = nullptr; // KV8: both set, the caches are q8
    const uint16_t * v_scale = nullptr;
    int32_t          rows = 0, pos0 = 0, max_cells = 0;
    float *          out   = nullptr;  // optional F32 [rows][3072]
    uint16_t *       out16 = nullptr;  // FP16 [rows][3072]
    uint32_t *       err   = nullptr;
};

bool qsp_attend(const qsp_attend_args & a, void * stream);

struct qsp_resources {
    int32_t regs, local_bytes, shared_bytes;
};

// 0: score, 1: attend, 2: kv; KV8: 3 attend and 4 kv in q8
bool qsp_kernel_resources(int32_t which, qsp_resources & r);

}  // namespace fn::qsa
