#pragma once

// Every buffer FNFL reserves at load beside the weights (R1), as classes of named parts with their per-card bytes.
// Each part's size is arithmetic from its kernel's size function or the design's formula, named beside it in
// state.cpp; nothing is a guess. A class is one allocation per card, its parts 256-byte aligned inside it.
//
// Capacity is in cells per slot (one slot: one user on two cards). R1's case: 196,608 + 2,048 output tokens + 64.

#include "fn/kernels/round/version.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::model {

constexpr int32_t n_qsa_layers = 12;   // target layers 3, 7, ..., 47
constexpr int32_t n_gdn_layers = 36;
constexpr int32_t n_index_owned = 6;   // target indexers a card owns (QSA ordinal % 2)
constexpr int32_t kv_head_dim = 256;   // one KV head a card a QSA layer (target and draft)
constexpr int32_t gdn_heads_card = 24; // value heads a card a GDN layer
constexpr int32_t gdn_head_dim = 128;  // key and value width: a head's state is 128 x 128 F32
constexpr int32_t gdn_conv_channels_card = 5120; // Q 1,024 + K 1,024 + V 3,072
constexpr int32_t gdn_conv_history = 3;          // ssm_d_conv - 1
// the convolution and PLE histories: the base and one per potentially accepted row (WP2 section 4); SP2: B <= 5, so six
// (fn::round::n_version). ME11: the GDN matrices keep two banks and two five-step record sets (fn/kernels/gdn/gdn.h)
constexpr int32_t gdn_versions = fn::round::n_version;
constexpr int32_t gdn_banks = 2;
constexpr int64_t gdn_record_step_bytes = 16512; // fn::gdn::rec_step_floats * 4
constexpr int64_t gdn_record_bytes = int64_t(2) * fn::round::gdn_max_steps * gdn_record_step_bytes;
constexpr int32_t round_rows = 5; // SP2: the most rows a round carries ([p, d1..d4]; fn::round::max_rows)
constexpr int32_t ple_channels_card = 5120; // four streams x 1,280 own features
// PF1: the prefill driver's checkpoints a slot (FC-ST1) and the rows of a device PLE staging slot (one chunk, FC-PF3)
constexpr int32_t prefill_checkpoints = 8;
constexpr int32_t prefill_stage_rows = 8192;

struct part {
    std::string name;
    int64_t bytes[2] = {0, 0}; // card 0, card 1
};

struct buffer_class {
    std::string name;
    std::string source; // where the sizes come from
    std::vector<part> parts;

    int64_t requested(int32_t card) const;     // the parts' bytes
    int64_t allocation(int32_t card) const;    // with 256-byte alignment of each part
    int64_t offset(int32_t p, int32_t card) const;
};

struct state_params {
    int32_t n_cells = 0;    // capacity a slot
    int32_t chunk_rows = 0; // prefill rows a chunk: 3,072 (two 1,536-row tiles) or 1,536 (one)
    int32_t ple_conv_kernel = 0;
    int32_t ple_ngram = 0;
    // ME1 (--kv-host): > 0, the host-backed KV tier's window W (fn/model/kv_tier.h): its "KV tier (ME1)" class on the
    // device; the target and draft KV classes are then allocated in mapped pinned host memory (host_class). ME6: the
    // raw index keys then leave the index classes for the host classes "target index keys (ME6)" and "draft index
    // keys (ME6)", with the device's "index key windows (ME6)"
    int32_t kv_window = 0;
    // ME1: both logical cards on one device (model::single): the "HC and PLE replicas (PB1)" class holds the draft's
    // full norms only (the prompt path reads the peer's HC and PLE tensors where they are resident)
    bool single = false;
    // KV8 (--fnfl-kv-format q8; the switch ends at KV8's disposition): "target KV (K0)" in the 8-bit format: each
    // QSA layer's K and V int8 codes [cells][256], then the parts "K scales" and "V scales", FP16 [cells][64], cell-major
    // (cell c, QSA layer o, group g at [c * 64 + 4 o + g]; fn/kernels/qsa/kv_q8.cuh). Two devices, resident only
    bool kv_q8 = false;
};

// TG1: the parts of the "round and pass (TG1)" class whose layout fn/model/pass.cpp carves (256-byte aligned pieces)
int64_t pass_round_bytes();  // the descriptor, then the round state
int64_t pass_commit_bytes(); // the commit's error-word pointers (SR1: no segments)
int64_t pass_embed_bytes();  // hres, hpost, hx of layer 0's input
int64_t pass_head_bytes();   // the head's moments, mixed half, full row and greedy pairs

// PF1 (FC-ST1): one checkpoint on one card: every GDN layer's state and convolution history, the PLE history (one
// version each) and the draft's retained carry [4][1280] F32; ME11: each GDN layer's state is its selected bank and
// up to five records (the selector's live steps), 1,572,864 + 82,560 bytes
int64_t checkpoint_bytes(const state_params & p);

// the per-cell bytes of the classes that grow with capacity, per card (KV, draft KV, index keys and summaries); KV8:
// kv_q8 the 8-bit target KV's (codes and the 128-byte scale rows)
int64_t bytes_per_cell(int32_t card, bool kv_q8 = false);
// KV8: the FP16 scales a cell in each of the q8 scale parts (12 QSA layers x 4 groups of 64, padded to 64: 128 bytes,
// so a 16,384-cell page's slice is 2 MiB, the VMM granularity)
constexpr int32_t kv_q8_scale_stride = 64;

// the classes in reservation order: target KV, draft KV, target index keys and summaries, draft index (SP2: both
// cards),
// the draft head copy (DR1: 57,671,680 + 131,072 bytes a card), GDN state, convolution histories, kernel workspaces
// and constants, the round and pass (TG1), the sampler (SM1), the prefill checkpoints and PLE staging (PF1)
std::vector<buffer_class> state_classes(const state_params & p);
// ME1: whether a class is allocated in pinned host memory: on one device the prefill checkpoints (PF1), and with the
// host-backed KV tier (kv_host) the target and draft KV (mapped: the tier's gathers and the draft's prompt K/V writes
// reach it from the device); ME6: and the target and draft index keys (mapped, the tier's key history)
bool host_class(const std::string & name, bool single, bool kv_host);

// the prefill chunk buffers of one chunk size (the prefill design's memory formulas; EP1's workspace)
buffer_class chunk_class(int32_t chunk_rows);

// MU1 (multi-slot serving, --parallel 2..4 on resident TP): a slot's own classes, its frame: the KV, the index keys and
// summaries, the GDN state and histories, the round and pass, the draft round and the sampler. A multi-slot load
// reserves them once a slot (model::slot_state); every other class (the weights' copies, the kernel workspaces, the
// prompt buffers, the checkpoints) is the engine's, once a load. A one-slot load is unchanged. MXMU1 (two slots on one
// card with the host-backed tier): also each slot's host key history (ME6) and device window (ME1, ME6)
bool slot_class(const std::string & name);
// MU1: the slot classes whose per-cell parts a multi-slot load takes from the shared KV pool (fn/model/kv_pool.h)
// instead of a reservation of their own: each slot sees the whole capacity, the slots together use it once
bool pooled_class(const std::string & name);
// MU1: the KV pool's page (cells): 16 pages of 16,384 at 262,144 cells; every pooled part's page slice is a multiple of
// the CUDA allocation granularity (2 MiB on sm_70)
constexpr int32_t kv_page_cells = 16384;
// MU1: the other-reader lane (a multi-slot load only): its own prompt arena, laid out for chunks of at most lane_rows
// rows in halves of lane_half_rows, and one PLE staging slot of lane_rows rows, so a reader TU serves out of turn runs
// a chunk while the live reader's chunk stays suspended in the main arena
constexpr int32_t lane_rows = 4096;
constexpr int32_t lane_half_rows = 2048;
std::vector<buffer_class> lane_classes();

} // namespace fn::model
