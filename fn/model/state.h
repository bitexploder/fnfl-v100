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
// the base and one per potentially accepted row (WP2 section 4); SP2: B <= 5, so six (fn::round::n_version)
constexpr int32_t gdn_versions = fn::round::n_version;
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
};

// TG1: the parts of the "round and pass (TG1)" class whose layout fn/model/pass.cpp carves (256-byte aligned pieces)
int64_t pass_round_bytes();  // the descriptor, then the round state
int64_t pass_commit_bytes(); // the commit's error-word pointers (SR1: no segments)
int64_t pass_embed_bytes();  // hres, hpost, hx of layer 0's input
int64_t pass_head_bytes();   // the head's moments, mixed half, full row and greedy pairs

// PF1 (FC-ST1): one checkpoint on one card: every GDN layer's state and convolution history, the PLE history (one
// version each) and the draft's retained carry [4][1280] F32
int64_t checkpoint_bytes(const state_params & p);

// the per-cell bytes of the classes that grow with capacity, per card (KV, draft KV, index keys and summaries)
int64_t bytes_per_cell(int32_t card);

// the classes in reservation order: target KV, draft KV, target index keys and summaries, draft index (SP2: both
// cards),
// the draft head copy (DR1: 57,671,680 + 131,072 bytes a card), GDN state, convolution histories, kernel workspaces
// and constants, the round and pass (TG1), the sampler (SM1), the prefill checkpoints and PLE staging (PF1)
std::vector<buffer_class> state_classes(const state_params & p);

// the prefill chunk buffers of one chunk size (the prefill design's memory formulas; EP1's workspace)
buffer_class chunk_class(int32_t chunk_rows);

} // namespace fn::model
