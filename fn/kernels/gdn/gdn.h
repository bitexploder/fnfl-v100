#pragma once

// The GDN layer's own math between its input products (DP2: qkv, z, alpha, beta) and its output product (ssm_out), on
// one card (GD1): the convolution with SiLU and its history versions as the input product's epilogue (gdn_conv.cuh;
// GD1 Amendment 2b), then one launch, gdn_core: the Q and K L2 norms, the decay and beta gates, the recurrence with
// per-row state versions, and the gated RMS norm, written as F32 and in FNFL's activation format
// (fn/kernels/act/act.h). The math is the pinned engine's (66e11c3): qwen4exp.cpp
// build_layer_attn_linear (lines 1694-1819), build_conv_state_at (1968-2020) and build_norm_gated (1012-1021),
// models.h build_gdn_l2_norm (13-17) and ggml-cpu ops.cpp's gated delta net (10990-11142).
//
// Card-local numbering (fn/model/placement, S2): the card holds value heads l = 0..23 (global 16 (l / 8) + 8 r + l % 8
// on card r) and Q/K groups 0..7 (global 8 r + g); head l reads group l % 8, which is the pinned op's h mod 16 in
// global numbering. Channels: Q of group g at 128 g, K at 1024 + 128 g, V of head l at 2048 + 128 l; z and the
// output of head l at 128 l.
//
// State (R1's reservation, fn/model/state.h; SR1, ME11). The convolution history keeps six physical slots: logical
// version v (0: committed, 1..5: after the step's first v rows) is in slot fn::round::version_slot(*cur, v), cur the
// device word naming the committed slot:
//   history:   F32 [6][5120][3]: [slot][channel][tap], oldest first: logical version v holds the channel's last 3
//              inputs after the first v rows.
// The recurrent state keeps two banks and two record sets a layer and card (ME11), selected by the word *gsel beside
// cur (fn/kernels/round/version.h: bank b, set r, k steps); the committed matrix is bank b with the first k steps of
// set r replayed onto it, each cell m <- fmaf(k_i, delta_j, m * decay), the recurrence's own two operations:
//   banks:     F32 [2][24][128][128]: [bank][head][j][i], row j (value channel j) 512 contiguous bytes;
//   records:   F32 [2][5][rec_step_floats]: [set][step] the normalized k [8][128] of the card's Q/K groups, the decay
//              [24] (padded to 32), delta [24][128] (head, value row): 16,512 bytes a step.
// A target round reads version 0 (bank b, k records), stores it into bank 1 - b and writes its own steps' records into
// set 1 - r; the frame's commit moves cur and the selector (SR1: a selection, no copy).

#include "fn/kernels/act/act.h"

#include <cstddef>
#include <cstdint>

namespace fn::gdn {

constexpr int32_t n_head = 24;      // value heads a card
constexpr int32_t n_group = 8;      // Q/K groups a card
constexpr int32_t head_dim = 128;   // key and value width
constexpr int32_t n_channel = 5120; // convolution channels a card: Q 1,024, K 1,024, V 3,072
constexpr int32_t q_first = 0;      // channel offsets
constexpr int32_t k_first = 1024;
constexpr int32_t v_first = 2048;
constexpr int32_t n_value = 3072; // value channels a card (z, the output)
constexpr int32_t n_tap = 4;      // ssm_d_conv
constexpr int32_t n_hist = 3;     // history taps
constexpr int32_t n_version = 6;  // 1 + max_rows
constexpr int32_t max_rows = 5; // SP2
constexpr int32_t n_split = 6; // blocks a head
constexpr int64_t state_version_floats = int64_t(n_head) * head_dim * head_dim; // one bank
constexpr int64_t hist_version_floats = int64_t(n_channel) * n_hist;
// ME11: the compact recurrent state of a layer and card
constexpr int32_t n_bank = 2;
constexpr int32_t n_rec_set = 2;
constexpr int32_t rec_k = 0;                                // [8][128] the normalized k of each Q/K group
constexpr int32_t rec_decay = rec_k + n_group * head_dim;   // [24] each head's decay, padded to 32 floats
constexpr int32_t rec_delta = rec_decay + 32;               // [24][128] delta of each (head, value row)
constexpr int32_t rec_step_floats = rec_delta + n_head * head_dim;
constexpr int64_t rec_set_floats = int64_t(max_rows) * rec_step_floats;
constexpr int64_t banks_bytes = int64_t(n_bank) * state_version_floats * 4;
constexpr int64_t records_bytes = int64_t(n_rec_set) * rec_set_floats * 4;
static_assert(rec_step_floats * 4 == 16512 && rec_delta % 4 == 0 && rec_step_floats % 4 == 0,
              "a step is 16,512 bytes, its pieces 16-byte aligned");
static_assert(banks_bytes == 3145728 && records_bytes == 165120 && records_bytes % 256 == 0,
              "two banks and two five-step sets: 3,310,848 bytes a layer and card");

static_assert(q_first + n_group * head_dim == k_first && k_first + n_group * head_dim == v_first, "Q, K, V tile");
static_assert(v_first + n_value == n_channel && n_value == n_head * head_dim, "V is the value heads");
static_assert(n_head % n_group == 0, "heads group over the Q/K groups");

// the workspace: one counter a head [24] U32. Zero it once before the first launch; every complete launch leaves the
// counters at zero again.
size_t workspace_bytes();

struct core_args {
    const float * conv = nullptr;   // [rows][5120] the convolution's post-SiLU output (conv_silu_epilogue)
    const float * z = nullptr;      // [rows][3072] the output gate's input
    const float * alpha = nullptr;  // [rows][24] raw
    const float * beta = nullptr;   // [rows][24] raw (the logit)
    const float * dt = nullptr;     // [24] ssm_dt.bias
    const float * a = nullptr;      // [24] ssm_a (-exp(A_log))
    const float * norm_w = nullptr; // [128] ssm_norm
    float * state = nullptr;         // [2][24][128][128] the banks (ME11)
    float * rec = nullptr;           // [2][5][rec_step_floats] the record sets (ME11)
    const uint32_t * gsel = nullptr; // device: the selector (fn::round::gdn_word)
    float * gated = nullptr;        // [rows][3072] the gated norm's output, F32
    act::out_view gated_q = {};     // the same in the activation format: codes [rows][3072], scales [rows][96]
    void * ws = nullptr;            // workspace_bytes()
    int32_t rows = 0;               // 1..5
};

// Enqueue gdn_core on stream (a cudaStream_t). False, with nothing enqueued, for an argument outside the contract (rows
// outside 1..5, a null pointer, a pointer not 16-byte aligned, gated_q not rows x 3072, a null stream, banks and
// records that overlap); false after a
// failed launch submission (the CUDA error is left for the caller).
bool core(const core_args & a, void * stream);

// FMS2 (build item 3, W3): core() of lanes 2 .. 4 of the joint verify round in one launch, grid.y = U: lane u's
// 144 blocks are core()'s on a[u] (each lane its own state, records, selector, outputs and workspace), so each lane's
// results are its core()'s bitwise. Every lane at the same rows, 2 .. max_rows (FMS13: R = d + 1). False, with nothing
// enqueued, for an argument outside core()'s contract in any lane, lanes outside 2 .. 4 or lanes of unequal rows.
bool core_lanes(int32_t lanes, const core_args * a, void * stream);

// The convolution with SiLU alone: conv_silu_epilogue (gdn_conv.cuh) in a launch of its own, one thread a channel, for
// gd-test and until LP1 wires the epilogue into DP2's producer; never timed as a result.
struct conv_args {
    const float * qkv = nullptr;    // [rows][5120] the product's finished columns
    const float * conv_w = nullptr; // [5120][4] ssm_conv1d, tap 0 oldest
    float * hist = nullptr;         // [6][5120][3]: logical version 0 read, versions 1..rows written
    const uint32_t * cur = nullptr; // device: the committed slot, 0..5 (fn::round::version_slot)
    float * out = nullptr;          // [rows][5120] post-SiLU
    int32_t rows = 0;               // 1..5
};
// False, with nothing enqueued, for an argument outside the contract (rows, null or unaligned pointers, null stream);
// false after a failed launch submission.
bool conv_silu(const conv_args & a, void * stream);

// the kernel's launch shape and resources at rows (an instance a B; gd-bench's recount): threads a block, blocks, and
// the occupancy API's resident blocks an SM, registers a thread and static shared memory (false on a CUDA error)
struct core_resources {
    int32_t threads = 0, blocks = 0, blocks_per_sm = 0, regs = 0, local_bytes = 0;
    size_t shared_bytes = 0;
};
bool resources(int32_t rows, core_resources & out);
// FMS2: core_lanes' kernel at lanes (blocks: the whole grid, 144 a lane)
bool lanes_resources(int32_t lanes, core_resources & out);

} // namespace fn::gdn
