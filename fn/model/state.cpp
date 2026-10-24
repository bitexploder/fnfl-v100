#include "fn/model/state.h"

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/draft/draft.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_prompt.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/head/head.h"
#include "fn/kernels/ple/ple.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/samp/samp.h"
#include "fn/model/kv_tier.h"
#include "fn/model/layer.h"
#include "fn/model/prompt.h"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.h"

#include <algorithm>

namespace fn::model {

namespace {

int64_t align256(int64_t n) {
    return (n + 255) / 256 * 256;
}

part both(const std::string & name, int64_t bytes) {
    part p;
    p.name = name;
    p.bytes[0] = bytes;
    p.bytes[1] = bytes;
    return p;
}

part one(const std::string & name, int32_t card, int64_t bytes) {
    part p;
    p.name = name;
    p.bytes[card] = bytes;
    return p;
}

// the target indexer's owner: QSA ordinal % 2 (layer 3 is ordinal 0)
int32_t index_owner(int32_t layer) {
    return (layer / fn::full_attn_interval) % 2;
}

} // namespace

int64_t buffer_class::requested(int32_t card) const {
    int64_t n = 0;
    for (const part & p : parts) {
        n += p.bytes[card];
    }
    return n;
}

int64_t buffer_class::allocation(int32_t card) const {
    int64_t n = 0;
    for (const part & p : parts) {
        n += align256(p.bytes[card]);
    }
    return n;
}

int64_t buffer_class::offset(int32_t p, int32_t card) const {
    int64_t n = 0;
    for (int32_t i = 0; i < p; ++i) {
        n += align256(parts[i].bytes[card]);
    }
    return n;
}

int64_t pass_round_bytes() {
    return align256(sizeof(fn::round::desc)) + align256(sizeof(fn::round::state));
}

int64_t pass_commit_bytes() {
    return align256(int64_t(fn::round::max_err) * sizeof(void *));
}

static_assert(round_rows == fn::round::max_rows && round_rows == layer_max_rows, "SP2: one row capacity");

int64_t pass_embed_bytes() {
    return align256(int64_t(round_rows) * 4 * 1280 * 4) + align256(int64_t(round_rows) * 4 * 4) +
           align256(int64_t(round_rows) * 1280 * 4);
}

int64_t pass_head_bytes() {
    return align256(int64_t(round_rows) * fn::hc::n_mom * 4) + align256(int64_t(round_rows) * 1280 * 4) +
           align256(int64_t(round_rows) * 2560 * 4) + align256(int64_t(round_rows) * 2 * 4);
}

static_assert(gdn_record_step_bytes == int64_t(fn::gdn::rec_step_floats) * 4 &&
                  gdn_record_bytes == fn::gdn::records_bytes && gdn_banks == fn::gdn::n_bank,
              "ME11: state.h's GDN layout is gdn.h's");

int64_t checkpoint_bytes(const state_params & p) {
    const int64_t gdn = int64_t(gdn_heads_card) * gdn_head_dim * gdn_head_dim * 4 + gdn_record_bytes / 2;
    const int64_t conv = int64_t(gdn_conv_channels_card) * gdn_conv_history * 4;
    const int64_t ple = int64_t(ple_channels_card) * (p.ple_conv_kernel - 1) * p.ple_ngram * 4;
    return int64_t(n_gdn_layers) * (gdn + conv) + ple + int64_t(4) * 1280 * 4;
}

int64_t bytes_per_cell(int32_t card, bool kv_q8) {
    // K0: BF16 K and V; KV8: int8 codes and the two scale rows
    const int64_t kv = kv_q8 ? int64_t(n_qsa_layers) * 2 * kv_head_dim + int64_t(2) * kv_q8_scale_stride * 2
                             : int64_t(n_qsa_layers) * 2 * kv_head_dim * 2;
    const int64_t draft_kv = int64_t{2} * kv_head_dim * 2;          // F16 K and V
    const int64_t index = int64_t(n_index_owned) * (fn::qsa::head_dim * 2 + fn::qsa::head_dim * 4 / fn::qsa::ratio);
    // SP2: the draft index on both cards (card 1's weight copies are fixed bytes, not per cell)
    (void)card;
    const int64_t draft_index = fn::qsa::head_dim * 2 + fn::qsa::head_dim * 4 / fn::qsa::ratio;
    return kv + draft_kv + index + draft_index;
}

bool host_class(const std::string & name, bool single, bool kv_host) {
    return (single && name == "prefill checkpoints (PF1)") ||
           (kv_host && (name == "target KV (K0)" || name == "draft KV (F16)" ||
                        name == "target index keys (ME6)" || name == "draft index keys (ME6)"));
}

std::vector<buffer_class> state_classes(const state_params & p) {
    const int64_t cells = p.n_cells;
    const int32_t blocks = fn::qsa::blocks_complete(p.n_cells);
    std::vector<buffer_class> out;

    buffer_class kv{
        "target KV (K0)",
        p.kv_q8 ? "KV8 (q8): int8 K and V codes, 256 + 256 bytes a cell, one head a layer; FP16 scales a group of 64, "
                  "cell-major K and V scale rows of 64 (128 bytes each) for the 12 layers"
                : "kv-cache-design.md K0: BF16 K and V, 512 + 512 bytes a cell, one head a layer",
        {}};
    const int64_t kv_elem = p.kv_q8 ? 1 : 2;
    for (int32_t l = fn::full_attn_interval - 1; l < fn::n_layer; l += fn::full_attn_interval) {
        kv.parts.push_back(both("L" + std::to_string(l) + " K", cells * kv_head_dim * kv_elem));
        kv.parts.push_back(both("L" + std::to_string(l) + " V", cells * kv_head_dim * kv_elem));
    }
    if (p.kv_q8) {
        static_assert(kv_q8_scale_stride >= n_qsa_layers * 4, "KV8: a scale row holds every layer's 4 groups");
        kv.parts.push_back(both("K scales", cells * kv_q8_scale_stride * 2));
        kv.parts.push_back(both("V scales", cells * kv_q8_scale_stride * 2));
    }
    out.push_back(kv);

    out.push_back({"draft KV (F16)",
                   "kv-cache-design.md: the draft block's F16 K and V, one head a card",
                   {both("L48 K", cells * kv_head_dim * 2), both("L48 V", cells * kv_head_dim * 2)}});

    if (p.kv_window > 0) {
        const int64_t arena = kv_tier::arena_cells(p.kv_window, cells) * kv_head_dim * 2;
        out.push_back({"KV tier (ME1)",
                       "kv_tier.h: per card a K and a V arena of the 13 attention layers' windows (W cells each) and "
                       "the gather area (the whole slot while a prompt stages a layer), and the remap's tables",
                       {both("K arena", arena), both("V arena", arena), both("tables", kv_tier::table_bytes())}});
        // ME6: the raw index keys' device windows; their history is the two host classes below
        out.push_back({"index key windows (ME6)",
                       "kv_tier.h: per card a key arena of 7 windows (the 6 owned target layers, the draft) of W cells "
                       "x 128 BF16, or one prompt chunk and its first block if larger",
                       {both("key arena", kv_tier::key_arena_cells(p.kv_window) * fn::qsa::head_dim * 2)}});
    }
    // ME6: with the host-backed tier the raw index keys are their own (host) classes; the summaries stay here
    const bool key_tier = p.kv_window > 0;

    buffer_class index{"target index keys and summaries",
                       key_tier ? "qsa.h: summaries summary_bytes(cells / 4), F32; on the owner (ME6: the raw keys are "
                                  "the target index keys (ME6) class)"
                                : "qsa.h: raw keys BF16 [cell][128]; summaries summary_bytes(cells / 4), F32; on the "
                                  "owner",
                       {}};
    buffer_class target_keys{
        "target index keys (ME6)", "qsa.h: raw keys BF16 [cell][128] on the owner (kv_tier.h)", {}};
    for (int32_t l = fn::full_attn_interval - 1; l < fn::n_layer; l += fn::full_attn_interval) {
        const int32_t c = index_owner(l);
        (key_tier ? target_keys : index).parts.push_back(one("L" + std::to_string(l) + " keys", c,
                                                             cells * fn::qsa::head_dim * 2));
        index.parts.push_back(one("L" + std::to_string(l) + " summaries", c, (int64_t)fn::qsa::summary_bytes(blocks)));
    }
    out.push_back(index);
    if (key_tier) {
        out.push_back(target_keys);
    }

    // SP2 (sparse draft attention, temporary at SP2's disposition): each card computes the same index products and
    // makes the same selection, so the keys, summaries and the indexer's weights (GGUF on card 0) are on both cards
    buffer_class draft_index{"draft index (SP2)",
                             "qsa.h: raw keys BF16 [cell][128] (ME6: the draft index keys (ME6) class); summaries "
                             "summary_bytes(cells / 4), F32; blk.48.indexer's q_proj, k_proj (BF16) and norms (F32) "
                             "copied to card 1; the step's select depth (I32); on both cards",
                             {}};
    if (!key_tier) {
        draft_index.parts.push_back(both("L48 keys", cells * fn::qsa::head_dim * 2));
    }
    draft_index.parts.push_back(both("L48 summaries", (int64_t)fn::qsa::summary_bytes(blocks)));
    draft_index.parts.push_back(
        one("indexer q_proj copy", 1, int64_t(fn::qsa::n_idx_head) * fn::qsa::head_dim * fn::n_embd * 2));
    draft_index.parts.push_back(one("indexer k_proj copy", 1, int64_t(fn::qsa::head_dim) * fn::n_embd * 2));
    draft_index.parts.push_back(one("indexer q_norm copy", 1, int64_t(fn::qsa::head_dim) * 4));
    draft_index.parts.push_back(one("indexer k_norm copy", 1, int64_t(fn::qsa::head_dim) * 4));
    draft_index.parts.push_back(both("select depth", 4));
    out.push_back(draft_index);
    if (key_tier) {
        out.push_back({"draft index keys (ME6)",
                       "qsa.h: the draft index's raw keys BF16 [cell][128] on both cards (kv_tier.h)",
                       {both("L48 keys", cells * fn::qsa::head_dim * 2)}});
    }

    // DR1 Stage A's re-shard (98.68% of the subset's IDs fall in card 0's vocabulary shard): each card's copy of half
    // the draft subset's output.weight rows, filled at load by head::pack_draft_head from head::draft_ids
    out.push_back({"draft head copy (DR1)",
                   "head.h: 32,768 Q5_K rows of 2,560 (head::draft_head_bytes, 57,671,680 bytes) and their ID map, "
                   "I32 [32,768], on each card",
                   {both("draft head rows (Q5_K, packed)", fn::head::draft_head_bytes(fn::head::n_draft_card)),
                    both("draft head ID map", int64_t(fn::head::n_draft_card) * 4)}});

    // ME11: two banks and two five-step record sets a layer and card (fn/kernels/gdn/gdn.h), in place of six versions
    buffer_class gdn{"GDN state",
                     "WP2 section 4: 24 heads x 128 x 128 F32 a layer and card; ME11: two banks (Ln) and two record "
                     "sets of 5 steps x 16,512 bytes (Ln records), selected by the round state's GDN word",
                     {}};
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (l % fn::full_attn_interval != fn::full_attn_interval - 1) {
            gdn.parts.push_back(both("L" + std::to_string(l),
                                     int64_t(gdn_banks) * gdn_heads_card * gdn_head_dim * gdn_head_dim * 4));
            gdn.parts.push_back(both("L" + std::to_string(l) + " records", gdn_record_bytes));
        }
    }
    out.push_back(gdn);

    buffer_class conv{"convolution histories",
                      "GDN: 5,120 channels x 3 positions F32 a layer and card; PLE (layer 1): 5,120 channels x "
                      "(conv_kernel - 1) x ngram positions F32 (qwen4exp.cpp build_ple); base + 5 row versions (SP2)",
                      {}};
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (l % fn::full_attn_interval != fn::full_attn_interval - 1) {
            conv.parts.push_back(both("L" + std::to_string(l) + " GDN",
                                      int64_t(gdn_versions) * gdn_conv_channels_card * gdn_conv_history * 4));
        }
    }
    const int64_t ple_hist = int64_t(p.ple_conv_kernel - 1) * p.ple_ngram;
    conv.parts.push_back(both("L1 PLE", int64_t(gdn_versions) * ple_channels_card * ple_hist * 4));
    out.push_back(conv);

    // the selection body U32 [B][517] and its offsets I32 [B + 1], q F32 [B][4][128]
    const int64_t select_io = int64_t(fn::qsa::max_rows) * fn::qsa::max_row_words * 4 +
                              int64_t(fn::qsa::max_rows + 1) * 4 +
                              int64_t(fn::qsa::max_rows) * fn::qsa::n_idx_head * fn::qsa::head_dim * 4;
    out.push_back(
        {"kernel workspaces and constants",
         "hc::workspace_bytes, experts::workspace_bytes, qsa::workspace_bytes and its select I/O (the "
         "selection body and offsets, q), the EX1 grid table, R2's layer_ws_bytes (a decode layer's "
         "activations, LP1's), LP1's dense_work_bytes (DP2's work) and dense::grid_bytes, gdn::workspace_bytes, "
         "qsa::attn::workspace_bytes",
         {both("HC sublayer (B <= 5)", (int64_t)fn::hc::workspace_bytes()),
          both("EX1 decode (B <= 5)", (int64_t)fn::experts::workspace_bytes()),
          both("QSA select scratch", (int64_t)fn::qsa::workspace_bytes()), both("QSA select I/O (B <= 5)", select_io),
          both("EX1 grid table", (int64_t)fn::experts::grid_bytes),
          both("R2 layer activations (B <= 5)", (int64_t)layer_ws_bytes()),
          both("DP2 work (B <= 5)", (int64_t)dense_work_bytes()),
          both("DP2 grid table", (int64_t)fn::dense::grid_bytes),
          both("GDN core workspace", (int64_t)fn::gdn::workspace_bytes()),
          both("QSA attend workspace", (int64_t)fn::qsa::attn::workspace_bytes())}});
    // TG1: the pass's buffers outside the layers (fn/model/pass.h carves them)
    out.push_back(
        {"round and pass (TG1)",
         "round.h's descriptor, state (with SR1's committed slot) and commit error list; the PLE upload slot e "
         "[5][2560] F32 and "
         "ple::workspace_bytes; the embedding's boundary (hres [5][4][1280], hpost [5][4], hx [5][1280]); "
         "the hidden carry (FC-HEAD3) [5][4][1280]; the head's moments, mixed, full and greedy pairs; the "
         "logits [5][124160] F32; head::greedy_work_bytes (SP2: five rows)",
         {both("round descriptor and state", pass_round_bytes()), both("commit error list", pass_commit_bytes()),
          both("PLE upload slot (B <= 5)", int64_t(round_rows) * 2560 * 4),
          both("PLE workspace", (int64_t)fn::ple::workspace_bytes()),
          both("embedding boundary (B <= 5)", pass_embed_bytes()),
          both("hidden carry (FC-HEAD3, B <= 5)", int64_t(round_rows) * 4 * 1280 * 4),
          both("head activations (B <= 5)", pass_head_bytes()),
          both("logits (B <= 5)", int64_t(round_rows) * fn::head::n_vocab_card * 4),
          both("greedy work", (int64_t)fn::head::greedy_work_bytes())}});
    // DR1 Stage C: the draft rounds' buffers outside the layer workspace (fn/model/draft.h carves them); the draft
    // rounds reuse the layer workspace, the embedding boundary, the head's moments and pairs and the greedy work
    out.push_back(
        {"draft round (DR1)",
         "draft.h: the retained carry and the draft's hidden output [4][1280] F32 (4 streams); X-DRAFT-IN's own "
         "payload [5][6408]; eh_proj's input codes [5][4][5120] and scales [5][4][160][2] and output [5][4][1280]; "
         "DA-TC's q16 [5][12][256] F16 and partials; the draft logits [4][32768] F32 and the head product's work "
         "(SP2: five catch-up rows, four steps)",
         {both("retained carry", int64_t(4) * 1280 * 4), both("draft hidden output", int64_t(4) * 1280 * 4),
          both("X-DRAFT-IN own payload", int64_t(fn::draft::max_rows) * fn::xchg::words_draft * 4),
          both("eh_proj input codes", int64_t(fn::draft::max_rows) * 4 * 5120),
          both("eh_proj input scales", int64_t(fn::draft::max_rows) * 4 * 160 * 8),
          both("eh_proj output", int64_t(fn::draft::max_rows) * 4 * 1280 * 4),
          both("DA-TC q16", int64_t(fn::draft::max_rows) * 12 * 256 * 2),
          both("DA-TC partials", (int64_t)fn::draft::part_bytes()),
          both("draft logits", int64_t(fn::samp::max_drafts) * fn::head::n_draft_card * 4),
          both("draft head work",
               std::max<int64_t>(256, (int64_t)fn::head::logits_work_bytes(fn::head::n_draft_card)))}});
    // SM1: the sampler's per-slot state (samp.h): the options and decision record, the bias table and penalty counts
    // over the whole vocabulary (both cards hold them, so the re-sharded draft head reads them too), the window's ring
    // (up to the slot's capacity: penalty_last_n = -1), the retained q of 4 draft rows (SP2; kept lists, and the target
    // shard's probabilities on the full paths), and the sampler's workspace
    out.push_back(
        {"sampler (SM1)",
         "samp.h: params and record; bias and counts [248,320] each; the ring [cells] I32; q lists "
         "4 x sizeof(qlist); q shards [4][124160] F32; samp::workspace_bytes",
         {both("sampler params and record", align256(sizeof(fn::samp::params)) + align256(sizeof(fn::samp::record))),
          both("logit bias table", int64_t(fn::samp::n_vocab) * 4),
          both("penalty counts", int64_t(fn::samp::n_vocab) * 4), both("penalty ring", cells * 4),
          both("retained q lists", int64_t(fn::samp::max_drafts) * int64_t(sizeof(fn::samp::qlist))),
          both("retained q shards", int64_t(fn::samp::max_drafts) * fn::samp::n_card * 4),
          both("sampler workspace", (int64_t)fn::samp::workspace_bytes())}});
    // PF1: the slot's checkpoints at multiples of 8,192 and prompt ends (FC-ST1; KV, index keys and summaries are
    // restored by their valid lengths, so they are not copied), and two device PLE staging slots of one chunk's rows
    // each (one being read by the chunk's rounds, the next chunk's uploading)
    buffer_class ck{"prefill checkpoints (PF1)",
                    "FC-ST1: 36 GDN states 24 x 128 x 128 F32 with up to 5 records (ME11), 36 GDN histories 5,120 x 3 "
                    "F32, the PLE history "
                    "5,120 x (conv_kernel - 1) x ngram F32 and the draft's retained carry [4][1280] F32, a checkpoint "
                    "a card; 8 a slot",
                    {}};
    for (int32_t i = 0; i < prefill_checkpoints; ++i) {
        ck.parts.push_back(both("checkpoint " + std::to_string(i), checkpoint_bytes(p)));
    }
    out.push_back(ck);
    const int64_t pf_slot = int64_t(prefill_stage_rows) * 2560 * 4;
    out.push_back({"prefill staging (PF1)",
                   "FC-PLE1 at prompt rows: two slots of 8,192 rows x 2,560 F32 (one chunk's PLE input e)",
                   {p.single ? one("PLE staging slot 0", 0, pf_slot) : both("PLE staging slot 0", pf_slot),
                    p.single ? one("PLE staging slot 1", 0, pf_slot) : both("PLE staging slot 1", pf_slot)}});
    // PB1: the bulk prompt path's arena (fn/model/prompt.h: the streams, the messages, the family scratch) and the
    // replicas of the peer's HC and PLE coefficients (FC-PF3's replicated HC) and the draft's full norms
    out.push_back({"prefill arena (PB1)",
                   "prompt.h prompt_arena_bytes: S [8192][4][2560], embeddings, mixed rows, the partial, the prompt "
                   "messages (P, EMB, SEL, MOM, DMIX), PLE's full history, the family scratch",
                   {part{"prompt arena",
                         {prompt_arena_bytes(prompt_chunk_rows, prompt_half_rows, p.single, 0),
                          prompt_arena_bytes(prompt_chunk_rows, prompt_half_rows, p.single, 1)}}}});
    out.push_back({"HC and PLE replicas (PB1)",
                   "prompt.h prompt_replica_bytes: the peer's w_down, w_up and gamma of the 96 HC sublayers, its PLE "
                   "key, value, norms and taps, the draft's full enorm and hnorm",
                   {both("replicas", prompt_replica_bytes(p.single))}});
    return out;
}

bool slot_class(const std::string & name) {
    // MXMU1: with the host-backed tier each slot keeps its own history (the ME6 host keys) and device window
    return pooled_class(name) || name == "GDN state" || name == "convolution histories" ||
           name == "round and pass (TG1)" || name == "draft round (DR1)" || name == "sampler (SM1)" ||
           name == "target index keys (ME6)" || name == "draft index keys (ME6)" || name == "KV tier (ME1)" ||
           name == "index key windows (ME6)";
}

bool pooled_class(const std::string & name) {
    return name == "target KV (K0)" || name == "draft KV (F16)" || name == "target index keys and summaries" ||
           name == "draft index (SP2)";
}

std::vector<buffer_class> lane_classes() {
    return {{"other-reader arena (MU1)",
             "prompt.h prompt_arena_bytes(4096, 2048): the other-reader lane's streams, messages, PLE history and "
             "family scratch",
             {both("lane arena", prompt_arena_bytes(lane_rows, lane_half_rows))}},
            {"other-reader staging (MU1)",
             "FC-PLE1 at prompt rows: one slot of 4,096 rows x 2,560 F32 (the lane's chunk's PLE input e)",
             {both("lane PLE staging", int64_t(lane_rows) * 2560 * 4)}}};
}

buffer_class chunk_class(int32_t chunk_rows) {
    // 3,072-row chunks run as two 1,536-row tiles whose buffers are live together (one tile's transfer beside the
    // other's compute); the fallback is one 1,536-row tile (fnfl-prefill-design.md section 3)
    const int32_t tiles = chunk_rows > fn::experts::ep_max_rows ? 2 : 1;
    const int64_t b = chunk_rows / tiles;
    buffer_class c{"prefill chunk buffers (" + std::to_string(chunk_rows) + " rows)",
                   "wp4/prefill-design.json memory_formulas (F32), per tile; EP1's ep_workspace_bytes once a chunk; "
                   "two input upload slots (fnfl-prefill-design.md section 2)",
                   {}};
    for (int32_t t = 0; t < tiles; ++t) {
        const std::string tn = "tile " + std::to_string(t) + " ";
        c.parts.push_back(both(tn + "two residual buffers", 2 * b * 4 * fn::n_embd * 4));
        c.parts.push_back(both(tn + "output partial and peer scratch", 2 * b * fn::n_embd * 4));
        c.parts.push_back(both(tn + "shared full input", b * fn::n_embd * 4));
        c.parts.push_back(both(tn + "GDN projected values (> QSA Q, gate, K, V)", b * (5120 + 3072 + 48) * 4));
        c.parts.push_back(both(tn + "retained hidden for the draft (own half)", b * 4 * (fn::n_embd / 2) * 4));
    }
    c.parts.push_back(both("EP1 workspace (" + std::to_string(b) + " rows)",
                           (int64_t)fn::experts::ep_workspace_bytes(static_cast<int32_t>(b))));
    c.parts.push_back(
        both("input upload slots x 2 (ids, positions, PLE input F32)", 2LL * chunk_rows * (4 + 4 + fn::n_embd * 4)));
    return c;
}

} // namespace fn::model
