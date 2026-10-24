#pragma once

// R2, LP1: one target decode layer on one card of the S2 split: HC (HC2's kernels and exchanges), the token mixer (GDN
// or QSA), the FFN (DP2's router, then EX2's experts launches, which carry the shared expert), each sublayer ending
// with its destination-half exchange (X-A, X-F). Every piece runs an FNFL kernel (LP1): DP2's products, GD1's gdn_core
// with the convolution as the GDN input launch's epilogue, QA1b's summaries and selection with QA2's raw key write,
// QA2's attend, EX2's experts. HC3: HC up writes each sublayer's mixed input in both forms (F32 and FNFL's activation
// format) into X-HMIX, and each body's first launches acquire it there (fn/xchg), so there is no ready launch and no
// standalone quantize. A piece FNFL has not built returns not_built
// (unbuilt_pieces). Host code only: it enqueues on one stream a card; the two cards meet only through fn/xchg. The pinned layer is qwen4exp.cpp's build loop (lines 468-520 at
// 66e11c3); the layer map with every op, its tensors, shapes per card and the piece that runs it is in
// work/R2/RESULT.md.
//
// Per card r (fn/model/placement.h): HC features r*1280.. of every stream; GDN value heads with h mod 16 in 8r..8r+7
// (24 heads, their 8 Q/K groups, 5,120 convolution channels); QSA query heads 12r..12r+11 and KV head r; the indexer on
// card (layer / 4) % 2; the router and the shared-expert gate on both cards (XP1's V2, LP1), each computing all B rows'
// routing, so there is no X-ROUTE; expert channels 320r..320r+319.
//
// State (R1's reserved buffers): convolution history [6][5,120][3] F32 a layer (logical version 0 the committed
// history, version v = 1..B after row v - 1 of the last step, at slot fn::round::version_slot(*card_ctx::cur, v); SR1;
// SP2: six versions); ME11: the GDN state as two banks [2][24][128 x 128] F32 and two record sets [2][5][16,512 bytes]
// a layer, selected by the word *card_ctx::gsel (fn/kernels/round/version.h, fn/kernels/gdn/gdn.h); KV [cell][256] BF16
// for K and V; the indexer's raw keys [cell][128] BF16 and F32 summaries. A commit moves the committed-slot word and
// the GDN selector together (fn::round::commit in the model pass, fn::round::advance in layer-test's runner).

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/experts/experts.h"
#include "fn/model/model.h"
#include "fn/xchg/xchg.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace fn::model {

constexpr int32_t layer_max_rows = 5; // SP2: [p, d1..d4]

// A step's outcome: not_built when a piece it needs has no FNFL kernel yet (err names the pieces)
enum class step_status : int32_t {
    ok,
    not_built,
    failed,
};

// The pieces of a target layer's step FNFL has not built, in step order: its token mixer's (GDN or QSA), then its
// FFN's. Empty when the layer runs entirely on FNFL kernels.
std::vector<std::string> unbuilt_pieces(int32_t layer);

// one HC sublayer's module on a card (fn::hc layouts)
struct hc_module {
    const uint16_t * w_down = nullptr; // [4][324][1280] BF16, inject rows 320..323
    const uint16_t * w_up = nullptr;   // [1280][4][320] BF16
    const float * gamma = nullptr;     // [4][1280]
    // DR1: the draft block's Q8_0 weights in hc.h's planar layouts (down2_q8, up2_q8); when set, w_down and w_up are
    // not read
    const uint8_t * q8_down = nullptr;
    const uint8_t * q8_up = nullptr;
};

// a dense quantized weight on a card: n_out rows of k weights, in DP2's packed layout once pack_dense has run (the
// same bytes as the GGUF slice, permuted)
struct dense_weight {
    const uint8_t * w = nullptr;
    int32_t fmt = -1; // fn::dense::fmt
    int64_t k = 0;
    int64_t n_out = 0;
    int64_t row_bytes = 0;
};

struct layer_weights {
    int32_t layer = -1;
    bool gdn = false;
    hc_module hc_attn, hc_ffn;
    // GDN
    dense_weight qkv, z, ssm_out;
    const uint16_t * alpha = nullptr; // [24][2560] BF16
    const uint16_t * beta = nullptr;
    const float * conv1d = nullptr;   // [5120][4] (the GDN input launch's convolution epilogue)
    const float * dt = nullptr;       // [24]
    const float * a = nullptr;        // [24]
    const float * ssm_norm = nullptr; // [128]
    // QSA
    dense_weight wq, wk, wv, wo;
    const float * q_norm = nullptr; // [256]
    const float * k_norm = nullptr;
    const uint16_t * idx_q = nullptr; // [512][2560] BF16 (index owner)
    const uint16_t * idx_k = nullptr; // [128][2560] BF16 (index owner)
    const float * idx_q_norm = nullptr;
    const float * idx_k_norm = nullptr;
    // FFN
    const uint16_t * router = nullptr;    // [512][2560] BF16 (both cards)
    const uint16_t * sh_gate_w = nullptr; // [1][2560] BF16 (both cards)
    dense_weight sh_gate, sh_up;          // the shared expert's gate and up: DP2's packed layout (EX2 reads it)
    const uint8_t * ex_gate_up = nullptr; // [512][gate_up_bytes]
    const uint8_t * ex_down = nullptr;    // [512][down_bytes]
    const uint8_t * ex_sh_down = nullptr; // [down_bytes] the shared expert's down, fn::experts::pack_down's layout
    int32_t ex_gu_fmt = -1;               // fn::experts::fmt
    int32_t ex_down_fmt = -1;
    int32_t ex_sh_down_fmt = -1;
    int32_t index_owner = 0;
    int32_t router_owner = 0;
};

// the weights of a target layer on a card, from the loaded model; false (err set) when a tensor is missing or its
// format is not one the step runs
bool resolve_weights(const model & m, int32_t layer, int32_t card, layer_weights & out, std::string & err);

// Repacks a target layer's dense weights on both cards in place into DP2's layout (fn::dense::pack: the same bytes,
// permuted; BF16 is unchanged), and the shared expert's down into the experts' layout (fn::experts::pack_down, EX2),
// once per loaded model: m.dense_packed records it (1 DP2's layout, 2 the experts'). Synchronous (a host round trip of
// each slice). Until the loader packs at load (R3), every user of a layer calls this before its first step.
bool pack_dense(model & m, int32_t layer, std::string & err);
// The same for one tensor (placement index), for the weights outside the layers (TG1: output.weight)
bool pack_dense_tensor(model & m, int32_t idx, std::string & err);

// DR1: the draft block (layer 48, blk.48 with its nextn tensors) on a card, S2's split as a target QSA layer's
// (FC-DR1 "Cards"): eh_proj's 1,280 output features of each stream over the whole [e | h] row, query heads 12r..12r+11
// and KV head r, expert channels 320r..320r+319; the router, shared gate and attention norms whole on each card
struct draft_weights {
    hc_module hc_attn, hc_ffn, head; // Q8_0 (q8_down, q8_up), the head without inject rows
    dense_weight eh;                 // nextn.eh_proj: 1,280 rows of 5,120
    dense_weight wq, wk, wv, wo;     // 6,144, 256, 256 rows of 2,560; wo 2,560 rows of 3,072
    const float * q_norm = nullptr;  // [256]
    const float * k_norm = nullptr;
    const float * enorm = nullptr;     // [1280] this card's half
    const float * hnorm = nullptr;     // [4][1280] this card's halves
    const float * router = nullptr;    // [512][2560] F32 (both cards)
    const float * sh_gate_w = nullptr; // [1][2560] F32 (both cards)
    dense_weight sh_gate, sh_up;       // Q8_0, DP2's packed layout
    const uint8_t * ex_gate_up = nullptr;
    const uint8_t * ex_down = nullptr;
    const uint8_t * ex_sh_down = nullptr; // fn::experts::pack_down's layout
    // ME7: the routed gate/up and down format (fn::experts::fmt), as the loader resolved the draft file's: Q8_0, or the
    // opt-in Q2_0 (the shared expert stays Q8_0)
    int32_t ex_fmt = -1;
};
bool resolve_draft_weights(const model & m, int32_t card, draft_weights & out, std::string & err);
// SP2 (sparse draft attention, temporary at SP2's disposition): blk.48.indexer's weights on card 0, GGUF rows (BF16
// q_proj [512][2560] and k_proj [128][2560], F32 norms [128]); card 1 holds copies (draft_init)
struct draft_index_weights {
    const uint16_t * q_proj = nullptr;
    const uint16_t * k_proj = nullptr;
    const float * q_norm = nullptr;
    const float * k_norm = nullptr;
};
bool resolve_draft_index_weights(const model & m, draft_index_weights & out, std::string & err);
// pack_dense for the draft block's dense weights and its shared down (DP2's layout; the experts')
bool pack_draft(model & m, std::string & err);

// R1's state of a layer on a card
struct layer_state {
    float * gdn = nullptr;        // ME11: the banks [2][24][128 x 128]
    float * rec = nullptr;        // ME11: the record sets [2][5][fn::gdn::rec_step_floats]
    float * conv = nullptr;       // [6][5120][3]
    uint16_t * k_cache = nullptr; // [cells][256] BF16 (KV8: int8 codes [cells][256])
    uint16_t * v_cache = nullptr;
    // KV8 (q8 only, else null): the layer's FP16 scales in the class's scale parts, cell c's group g at
    // [c * kv_q8_scale_stride + g] (the part's base + 4 x the layer's QSA ordinal; fn/kernels/qsa/kv_q8.cuh)
    uint16_t * k_scale = nullptr;
    uint16_t * v_scale = nullptr;
    uint16_t * idx_keys = nullptr;   // [cells][128] BF16 (index owner)
    float * idx_summaries = nullptr; // tiled F32 (index owner)
    int32_t n_cells = 0;
    int32_t kv_attn = -1; // a QSA layer's attention index (layer / 4: kv_tier.h); -1 on a GDN layer
};
bool resolve_state(const model & m, int32_t layer, int32_t card, layer_state & out, std::string & err);

// The activations of one card's step (B <= 4), one allocation carved by the plan below: R1's "kernel workspaces and
// constants" class holds it as a part (layer_ws_bytes). The sublayers' quantized inputs and the routed experts' buffers
// are EX1b's decode workspace (card_ctx::ex).
struct layer_ws {
    // HC, two sets (attention and FFN sublayers alternate): new streams, moments, injection, the sublayer's completed
    // own output half. The mixed input is in the sublayer's X-HMIX slot (HC3).
    float * x[2];      // [B][4][1280]
    float * mom[2];    // [B][1300]
    float * inject[2]; // [B][4]
    float * out[2];    // [B][1280]
    uint32_t * hc_runs; // [40] HC up's run counters (hc.h up2_args::runs): zeroed by resolve_card and clear_workspaces
    float * zero;      // [B][1280] and [B][4]: hx and hpost of a teacher-forced layer input
    float * partial;   // [B][2560] the sublayer's local output contribution
    // the mixer's output in FNFL's activation format (gdn_core's or attend's), the output product's input
    int8_t * mid_codes; // [B][3072]
    float * mid_scales; // [B][96][2]
    // GDN (the input launch's outputs, then gdn_core's)
    float * conv;  // [B][5120] post-SiLU (the convolution epilogue)
    float * z;     // [B][3072]
    float * alpha; // [B][24] raw
    float * beta;  // [B][24] raw
    float * gated; // [B][3072] gdn_core's F32 output (records)
    // QSA
    float * qfull;       // [B][6144]: each head's q, then its gate
    float * kv_k;        // [B][256]
    float * kv_v;        // [B][256]
    float * attn;        // [B][3072] attend's gated output, F32 (records)
    float * ik;          // [B][128] index key projection
    float * iq;          // [B][4][128] index query projection
    uint32_t * sel_body; // [B][517] the selection body (fn::qsa layout): the index owner's from select, the peer's
                         // from X-SELECT
    int32_t * sel_off;   // [B + 1] its byte offsets
    int32_t * pos;       // [4][B] positions, one plane a rope section (layer-test's runner writes it before a step)
    int32_t * depth; // [1] cells before the rows (layer-test's runner writes it; the model pass reads the descriptor)
    uint32_t * cur;  // [2] the committed slot, then (ME11) the GDN selector (layer-test's runner; the model pass uses
                     // fn::round::state::cur and gdn)
    // FFN: the router launch's outputs (the IDs and weights go to the experts' workspace)
    float * logits;   // [B][512]
    float * sg_logit; // [B]
    float * sg;       // [B] sigmoid of the shared gate: the shared expert's weight in EX2's down
};
size_t layer_ws_bytes(int32_t rows = layer_max_rows); // FMU2: a joint frame's at fn::xchg::joint_rows
// DP2's work: one region a launch site, each serving launches of one layout (dense.h: a region zeroed once is reused
// only by launches with the same layout). The QSA input has a site a role: the index owner's 5 matrices and the
// peer's 3 split differently, and a card changes role every four QSA layers. The router's site serves both layer
// kinds, whose shapes, and so layouts, are equal. (The shared expert's products left with EX2.)
// DR1 adds the draft block's: eh_proj (one call of 4 stream rows a position, fn::draft::eh_proj), the attention input
// (q, k, v; the catch-up's k, v), the attention output and the F32 router.
// SP2 appends the sparse draft's two (temporary, at SP2's disposition): the step-0 input with the index query, and the
// catch-up's k, v with the index key. Appended after the baseline's, so the baseline's work offsets are unchanged.
enum class dense_site : int32_t {
    gdn_input,
    gdn_out,
    qsa_input_owner,
    qsa_input_peer,
    qsa_out,
    router,
    draft_eh,
    draft_qkv,
    draft_kv,
    draft_out,
    draft_router,
    draft_qkv_idx, // SP2
    draft_kv_idx,  // SP2
    count
};
constexpr int32_t n_dense_sites = static_cast<int32_t>(dense_site::count);
using dense_layouts = std::array<fn::dense::work_layout, n_dense_sites>;
const char * dense_site_name(dense_site site);
std::string dense_layout_text(const fn::dense::work_layout & l);
// Each site's layout over every launch shape it serves (frame initialization); false, with err naming the site and
// its layouts, when a site's shapes have more than one
bool dense_site_layouts(dense_layouts & out, std::string & err);
size_t dense_site_bytes(dense_site site);
size_t dense_work_offset(dense_site site); // 256-byte aligned regions in dense_site's order
size_t dense_work_bytes();                 // all of them
// DP2's launch in its site's region of work (dense_work_bytes, zeroed once), splits chosen from the shapes. Refused,
// with nothing enqueued, unless the launch's layout is the site's: the sites' shapes are written apart from the bodies'
// launches. A host compare; the bodies run it while a graph is captured, not each round.
bool dense_site_launch(void * work, const dense_layouts & layouts, dense_site site, fn::dense::product_args a,
                       int32_t rows, void * stream, std::string & err);
layer_ws carve_layer_ws(void * base, int32_t rows = layer_max_rows);
// FMU2: lane u's five-row view of a layer_ws carved at fn::xchg::joint_rows rows: every per-row buffer from row 5u (the
// selection offsets from entry 6u); the HC run counters, positions, depth and committed-slot words shared
layer_ws lane_ws(const layer_ws & joint, int32_t u);

// The experts' decode buffers on a card (fn::experts::workspace_offsets): the GLU slots (the routed pairs and the
// shared expert's rows), the routing, the status word. (x_codes and x_scales are unused since HC3: the FFN input is
// read from X-HMIX.)
struct ex_bufs {
    int8_t * x_codes = nullptr; // [4][2560]
    float * x_scales = nullptr; // [4][80][2]
    int8_t * glu_codes = nullptr;
    float * glu_scales = nullptr;
    int32_t * ids = nullptr;  // [4][10]
    float * wts = nullptr;    // [4][10]
    uint8_t * meta = nullptr; // the routing meta of ids (fn::experts::meta_bytes), the router's (XB2)
    uint32_t * status = nullptr;
};

// ME8: a non-owning view of one logical card's routing (its router's top-10 IDs and weights, its shared gate), read by
// the other logical card's experts when both cards run on one device and one compute stream: the router computed
// once. The buffers stay the producing card's; the next writer is its next router, after every consumer in stream
// order (fn::experts::use_route sets a launch pair's routing from it)
using route_view = fn::experts::route_ptrs;

struct card_ctx;
struct layer_weights;
class kv_tier;

// ME3: a byte range on one device (a prompt buffer's part the prompt writes)
struct byte_span {
    const void * p = nullptr;
    int64_t bytes = 0;
};

// ME4: one ring group of a prompt layer (expert_service::prompt_group): experts [e_lo, e_hi), their pairs (exact after
// a route-first layer's routing, else a bound), and each logical card's address tables (gate/up, down; 512 entries,
// only [e_lo, e_hi) meaningful) for ep_gate_up_args::wt and ep_down_args::wt
struct prompt_group_view {
    int32_t e_lo = 0, e_hi = 0;
    int64_t pairs = 0;
    const uint8_t * const * table[2][2] = {}; // [card][down]
};

// MX7: a logical card's ep_sum inputs the prompt CPU share's add-back writes (fn/runtime/prompt_share.h): its routing's
// sorted positions, its partials (F16 [R x 10][2560]) and its error word and bit (ep_down's)
struct prompt_share_card {
    const int32_t * pos = nullptr;
    uint16_t * part16 = nullptr;
    uint32_t * err = nullptr;
    uint32_t err_bit = 0;
    const float * wts = nullptr; // the route weights (TEST: the shadow comparison's report, mx7_shadow.h)
};

// ME1: the target FFN's routed experts through the expert cache (fn/runtime/expert_cache.h). With a service on both
// cards (one device holding both, or one each), layer_pair_step runs both FFN bodies as one: both routers, then
// ffn_pair, which enqueues card k's gate_up and down on c[k]'s stream. ga[k] and da[k] are those launches as the
// resident banks would run them; the service sets their weights (w, wt), passes (skip, run_if) and CPU slots
// (ext_slots_d).
// ME3, the graph executor: a call made while its stream is capturing enqueues only the window's GPU work (resolve, the
// window's descriptor copies and stamp, the passes and their flag waits, down, the combine), with no host action; the
// replay's host work is pass_begin (the round boundary, the zone's return, the previous replay's timings) and
// pass_posted (one service job a layer, each waiting for its window's stamp), both before the launch: a launch can
// block until the GPU has run part of the replay, which needs the service (ME3). pass_abandon after a launch that
// failed: the posted jobs end failed. On one device a capture enqueues through ffn_pair (the pair order); on two, each
// card's capture through ffn_card (layer_card_step's ffn_body), card c's own part of the window on its own device.
class expert_service {
  public:
    virtual ~expert_service() = default;
    virtual bool ffn_pair(card_ctx * const c[2], const layer_weights * const w[2], int32_t rows,
                          fn::experts::gate_up_args ga[2], fn::experts::down_args da[2], std::string & err) = 0;
    virtual bool ffn_card(card_ctx & c, const layer_weights & w, int32_t rows, fn::experts::gate_up_args & ga,
                          fn::experts::down_args & da, std::string & err) = 0;
    virtual bool pass_begin(void * const stream[2], std::string & err) = 0;
    virtual bool pass_posted(int32_t rows, std::string & err) = 0;
    virtual void pass_abandon(const std::string & why) = 0;
    // the executor the passes run under (graph: pass_begin and pass_posted frame each replay); set between passes
    virtual void set_executor(bool graph) = 0;
    // the prompt path (prompt.cpp): at target layer `layer`'s first expert launch of a chunk, after card 0's routing
    // (ids: its [rows][10] on card 0's device when the chunk is one half, else null), the layer's experts made readable
    // (staged) for the launches on card c's stream[c]; then prompt_table gives card's address table (512 entries; down
    // or gate/up) of that layer for ep_gate_up_args::wt and ep_down_args::wt, and prompt_bank a valid bank pointer on
    // card's device for their w. ME4: more, another chunk of the run follows (only then may the last layer stage the
    // next chunk's first ahead)
    virtual bool prompt_layer(int32_t layer, int32_t rows, const int32_t * ids, bool more, void * const stream[2],
                              std::string & err) = 0;
    virtual const uint8_t * const * prompt_table(int32_t layer, int32_t card, bool down) const = 0;
    virtual const uint8_t * prompt_bank(int32_t card) const = 0;
    // ME4 (fn/runtime/prompt_plan.h): the next loan stages through the ring (a run of one-half chunks), else ME3's two
    // layer buffers. After a ring layer's prompt_layer, prompt_groups gives its groups (0: not a ring layer: use
    // prompt_table). For each group in order: prompt_group makes each card's stream[d] wait for the group's weights and
    // tables (their copies after every earlier reader of those bytes) and returns them; the caller enqueues the
    // group's gate/up and down on both logical cards, then prompt_group_done records the group's release after those
    // readers. ep_sum follows the layer's last group. MX3: rows, the run's largest chunk (0: no ring), sizes the ring
    virtual void set_prompt_ring(int32_t rows) { (void)rows; }
    // ME4's comparison base arm (expires at ME4's disposition): ME3's prompt staging (no ring; the parity buffers
    // at the zone's start; routed-rows)
    virtual void set_me4_base(bool on) { (void)on; }
    virtual int32_t prompt_groups(int32_t layer) const {
        (void)layer;
        return 0;
    }
    virtual bool prompt_group(int32_t layer, int32_t g, void * const stream[2], prompt_group_view & v,
                              std::string & err) {
        (void)layer;
        (void)g;
        (void)stream;
        (void)v;
        err = "prompt_group: no ring";
        return false;
    }
    virtual bool prompt_group_done(int32_t layer, int32_t g, void * const stream[2], std::string & err) {
        (void)layer;
        (void)g;
        (void)stream;
        err = "prompt_group_done: no ring";
        return false;
    }
    // MX7 (fn/runtime/prompt_share.h): on a ring layer, after prompt_layer, prompt_share_begin with the layer's F16
    // input ([rows][2560] on card 0's device) posts the CPU's experts' work; prompt_share_skip gives the device mask
    // (512) of the experts whose tasks es_drop removes after each card's ep_group (null: none); after the layer's last
    // group, prompt_share_end adds the CPU's products into each card's partials before ep_sum
    virtual bool prompt_share_begin(int32_t layer, const uint16_t * x16, void * const stream[2], std::string & err) {
        (void)layer;
        (void)x16;
        (void)stream;
        (void)err;
        return true;
    }
    virtual const uint8_t * prompt_share_skip(int32_t layer) const {
        (void)layer;
        return nullptr;
    }
    virtual bool prompt_share_end(int32_t layer, const prompt_share_card card[2], void * const stream[2],
                                  std::string & err) {
        (void)layer;
        (void)card;
        (void)stream;
        (void)err;
        return true;
    }
    // MX7: on every exit of a layer that called prompt_share_begin (an error between begin and end included): the
    // layer's CPU job, if any, joined and its plan dropped, so the decode service never finds the pool in the share's
    // hands. Idempotent; after a prompt_share_end that succeeded, nothing remains
    virtual void prompt_share_settle(int32_t layer) { (void)layer; }
    // ME2b: the prompt buffers' region. On one card the expert pool lends PB1's arena and PF1's staging to its slots
    // between prompts (prompt_lent). prompt_hold(true) before the first write into them (a prompt's staging upload or
    // preparation, a test's poison): the region's residents retired from the tables, every reader of them and every
    // copy into the region finished. prompt_hold(false) after their last reader: every stream of the device drained,
    // the residents refilled as a pass at the next decode window (or drain). Both idempotent; with nothing lent they
    // return true at once.
    // ME3: need, the byte spans (on the region's device) the prompt writes. Only the region's slots they touch and the
    // staging the prompt's layers need are borrowed, and only their residents refilled; null: the whole region and
    // zone (ME2b's loan).
    virtual bool prompt_lent() const { return false; }
    virtual bool prompt_hold(bool on, void * const stream[2], const std::vector<byte_span> * need, std::string & err) {
        (void)on;
        (void)stream;
        (void)need;
        (void)err;
        return true;
    }
    // ME3: the startup half of the loan's one calculation: refused (err) unless the largest prompt's spans (need) and
    // its staging fit the loan cap and the kept slots
    virtual bool prompt_loan_check(const std::vector<byte_span> & need, std::string & err) {
        (void)need;
        (void)err;
        return true;
    }
    // ME3: the CPUs the prompt's host helpers run on (the cache's helper mask), null for no preference
    virtual const std::vector<int32_t> * helper_cpus() const { return nullptr; }
    // ME3: known-token short reads in progress (their windows counted apart); the base arm's legacy mechanisms (ME3's
    // comparison switch, expiring at its disposition)
    virtual void set_short_read(bool on) { (void)on; }
    virtual void set_legacy(bool on) { (void)on; }
    virtual bool legacy() const { return false; }
    // ME3: a request's accounting: request_begin at its prefill's entry, request_json its one line at its end
    virtual void request_begin() {}
    virtual std::string request_json() { return std::string(); }
    // ME3 (Fable's load G ruling, 2026-10-18; Strata #267 adapted): no GPU wait outlives an error. After a round's
    // completion wait timed out, or at teardown with work still on the streams (only_if_busy): the service's waiting
    // jobs ended, every word a window's or a replay's swap wait reads raised until the devices' streams drain or
    // timeout_ms passes, and every later window refused (the engine is started again, not computed on). note: what
    // happened, empty when only_if_busy found nothing running. True when the streams drained
    virtual bool release_gpu_waits(int32_t timeout_ms, bool only_if_busy, std::string & note) {
        (void)timeout_ms;
        (void)only_if_busy;
        note.clear();
        return true;
    }
};

// ME3: a call a service makes once during its init, after the buffers its windows' GPU work names exist and before its
// pool takes the device's remaining memory (the graph executor's capture: the graphs come out of the device's account
// ahead of the pool)
using service_hook = std::function<bool(expert_service &, std::string &)>;

// What one card needs to run steps: its workspaces from R1's reservation and the rings.
struct card_ctx {
    int32_t card = 0;
    int32_t device = -1; // the CUDA device of the card (made current before each of its phases)
    void * stream = nullptr;
    const fn::xchg::rings * rings = nullptr;
    layer_ws ws;
    // R1's "kernel workspaces and constants" parts: the EX1 decode buffers, the QSA select scratch and I/O, the grids,
    // DP2's work and grid table, gdn_core's and attend's workspaces
    ex_bufs ex;
    uint8_t * qsa_ws = nullptr;
    uint8_t * qsa_io = nullptr;
    const uint8_t * grids = nullptr;
    void * dense_work = nullptr;
    dense_layouts dense_sites{}; // each site's layout (dense_site_layouts)
    const uint8_t * dense_grids = nullptr;
    void * gdn_ws = nullptr;
    void * attn_ws = nullptr;
    // the round's N_before and rope planes [4][B] on this card's device (FC-RD1): the descriptor's fields in the model
    // pass, ws.depth and ws.pos in layer-test's runner. Every piece that needs the depth reads it from here on the
    // device, so a captured step serves every depth.
    const int32_t * n_before = nullptr;
    const int32_t * pos = nullptr;
    // SP4 (the expert gate/up mapping): fn::experts::gate_up_args::map for the target's FFN, read at enqueue (so fixed
    // in a captured graph); 0: SP2's, 1: SP4's paired 320-thread mapping (H1), 2: H2's IQ2 body at B = 4 and 5,
    // 3: XB3 C1's (2 plus the IQ3 body at B = 4 and 5). gate_up_for reads it only at B >= 4, so two-guess serving is
    // unchanged; spec ab arms set it explicitly
    int32_t ex_map = 3;
    // SR1: the committed slot of every versioned object (fn/kernels/round/version.h) on this card's device: the round
    // state's cur in the model pass, ws.cur in layer-test's runner
    const uint32_t * cur = nullptr;
    // ME11: the GDN state's selector (fn/kernels/round/version.h): the round state's gdn in the model pass, ws.cur + 1
    // in layer-test's runner
    const uint32_t * gsel = nullptr;
    // an optional observer, called after each piece is enqueued (layer-bench's per-piece events); null: none
    void (*observe)(void * user, int32_t card, const char * piece) = nullptr;
    void * observe_user = nullptr;
    // model-test's and layer-test's records: when set, a sublayer's full F32 mixed input [B][2560] is copied here
    // (device or pinned host memory, in stream order) after its body, before the next HC sublayer resets its X-HMIX
    // slot (HC3); null: none
    float * record_mixed[2] = {nullptr, nullptr};
    // ME10: set by the expert cache's window when its one-device finish wrote this card's ws.out[1] (combine and X-F's
    // exchange in one launch); layer_pair_step clears it before the FFN body and skips X-F's send and receive after it
    bool ffn_finished = false;
    // ME1: the expert cache's service (null: the resident banks)
    expert_service * ex_service = nullptr;
    // ME1 (--kv-host): the host-backed KV tier (kv_tier.h): the decode attention's K/V pointers and remapped bodies,
    // the prompt's staging (null: the resident KV classes)
    kv_tier * kv = nullptr;
};
bool resolve_card(const model & m, int32_t card, void * stream, card_ctx & out, std::string & err);
// Zeroes the workspaces whose counters every complete launch leaves at zero (QA1b's select scratch, DP2's work,
// gdn_core's and attend's, HC up's run counters): resolve_card does it once; a caller does it again after a failed or
// abandoned step.
// Synchronous.
bool clear_workspaces(const model & m, int32_t card, std::string & err);

// The inputs of a layer's first HC sublayer: the previous layer's streams, its injection and its output half (the
// combine that hc_down applies: x = hres + hpost * hx), as device pointers on this card.
struct layer_input {
    const float * hres = nullptr;  // [B][4][1280]
    const float * hpost = nullptr; // [B][4]
    const float * hx = nullptr;    // [B][1280]
};

// Where the layer's output stands after the step (on this card): the FFN sublayer's streams, injection and output
// half, which the next layer's first down2 combines. l_last = x + inject * out (per stream), computed by that down2.
layer_input layer_output(const card_ctx & c);

// The attention and FFN sublayers' completed own output halves are ws.out[0] and ws.out[1] after a step.

// A graph's X-HC / X-HMIX slot rule (xchg.h, "HC slots in draft rounds"). Every round's graph starts with slot 1 the
// only dirty slot and leaves it so: target: HC2's slot_of (the last sublayer slot 1); draft_step: G = 3 d sublayers,
// slot(k) = {1, 2, 0}[(G - 1 - k) % 3] (from slot 0, ending on slot 1); catchup: its one sublayer on slot 2, released
// by the K/V prepare (fn::draft::prepare_args::release_hc, release_hmix).
enum class hc_rule : int32_t { target, draft_step, catchup };

// Message use counters and the HC sublayer index: they pick the exchange slots and advance as a step runs.
struct use_counters {
    int64_t sub = 0; // HC sublayers so far
    int64_t a = 0, f = 0, select = 0;
    int64_t draft = 0;  // X-DRAFT-IN uses (DR1)
    bool graph = false; // a graph replays a fixed sequence: slots by position in it (HC2's slot_of)
    int32_t graph_subs = 0;
    hc_rule rule = hc_rule::target;
    // uses of X-A, X-F, X-SELECT and X-DRAFT-IN in one graph (slot = position % 3 of the use in the graph):
    // layer-test's pair has 2, 2 and 1; the model pass 48, 48 and 12; a draft step round d, d and d draft inputs, a
    // catch-up 1
    int32_t graph_a = 2, graph_f = 2, graph_select = 1, graph_draft = 1;
};

// The X-HC / X-HMIX slot of HC sublayer n under u's rule (n % 3 eagerly; hc_rule in a graph): the model pass uses it
// for the HC head, sublayer 96 of a round
int32_t hc_slot_of(const use_counters & u, int64_t n);
// the slot sublayer n's down2 resets (its predecessor's), -1 for none (the first sublayer of an eager run)
int32_t hc_prev_slot_of(const use_counters & u, int64_t n);
// the slot of use n of a message with per_graph uses a graph (n % 3 eagerly)
int32_t use_slot_of(const use_counters & u, int64_t n, int32_t per_graph);

// ---- pieces of a sublayer for the draft block (fn/model/draft.h), as the target layer runs them ----

// HC sublayer u.sub, part 0 (down2, sends X-HC) or 1 (up2, reads it and writes X-HMIX on both cards), into set's
// buffers (ws.x, ws.mom, ws.inject); in: the sublayer's input streams
bool hc_sublayer(card_ctx & c, const hc_module & h, int32_t set, const layer_input & in, int32_t rows,
                 const use_counters & u, int32_t part, std::string & err);
// this card's X-HMIX slot of HC sublayer u.sub (the body's input)
float * hmix_slot(const card_ctx & c, const use_counters & u);
// DP2's launch in site's region; acquire: the first launch of a body, reading X-HMIX of sublayer u.sub (HC3)
bool dense_run(card_ctx & c, fn::dense::product_args a, int32_t rows, dense_site site, bool acquire,
               const use_counters & u, const char * piece, std::string & err);
// the destination-half exchange of ws.partial (set 0: X-A, 1: X-F) on slot `slot`: send, or receive into ws.out[set]
bool half_exchange(card_ctx & c, int32_t set, int32_t rows, int32_t slot, bool send, std::string & err);
// card_ctx::observe
void note_piece(const card_ctx & c, const char * piece);

// ME8: whether card 1's FFN reads card 0's routing (both cards on one device and one non-null compute stream): its own
// router, IDs, weights, shared gate and logits are then not written, and a reader of card 1's routing reads card 0's
bool one_router(const card_ctx & c0, const card_ctx & c1);

// One layer of B rows at positions N_before .. N_before + B - 1 (c.n_before and c.pos, device memory; the caller
// checks N_before + B against the state's cells before it enqueues).
// layer_pair_step enqueues both cards' work in phases, every send before the receive that reads it (down2 and up2 of
// both cards, the body of the card that sends X-SELECT before its peer's, then both X-A or X-F
// sends before both receives): the host's issue order is then a valid sequential schedule, so the step also runs
// where kernels execute one at a time (Compute Sanitizer). layer_card_step enqueues one card's work alone, for a
// graph capture. Both advance u. failed on a failed enqueue (err says which; the CUDA error is left for the caller);
// not_built at the first body with a piece FNFL has not built, before that body enqueues anything.
step_status layer_pair_step(card_ctx * const c[2], const layer_weights * const w[2], const layer_state * const st[2],
                            const layer_input in[2], int32_t rows, use_counters u[2], std::string & err);
step_status layer_card_step(card_ctx & c, const layer_weights & w, const layer_state & st, const layer_input & in,
                            int32_t rows, use_counters & u, std::string & err);

// ---- FMU2: the joint round on one card (fn/xchg/xchg.h joint_rings) ----
//
// 2..4 slots' rounds of layer_max_rows rows each, as lanes of one stream: lane u's rows are the joint rows
// 5u .. 5u + 4. Each lane's frame (lane[u]) is its slot's card_ctx with the lane's ring (joint_rings::lane(u)), the
// joint layer_ws's lane view (lane_ws; lane 0's is the joint frame's base) and the joint routing buffers' lane view
// (ex.ids, ex.wts rows 5u ..); its state, descriptor depth and positions, committed slot and GDN selector stay its
// slot's. A layer runs both HC sublayers as one launch for every lane, each lane on its own ring (fn::hc::sub_joint,
// BF16 HC weights); the dense products once for every lane (fn::dense::product_joint: each weight tile decoded
// once, each row's result bitwise the lone launch's); the slot pieces per lane (gdn_core, QSA's summary, select,
// X-SELECT and attend, the experts); X-A and X-F once over every lane's rows on the joint slots. Every lane takes the
// same pieces in the same order on both cards with the lone slot rules (one use_counters for all lanes), so each
// lane's ring sees the lone sequence. Resident banks only (no expert service, no host-backed KV); for a graph capture
// (one card's whole round on its stream), or eagerly in layer_pair_step's order (layer_joint_pair_step).
struct joint_lanes {
    int32_t lanes = 0;
    card_ctx * lane[fn::xchg::max_lanes] = {};
    const fn::xchg::joint_rings * jr = nullptr;
    void * dense_work = nullptr; // joint_dense_work_bytes(), zeroed once, this card's
    dense_layouts dense_sites{}; // joint_dense_layouts (the target's six sites)
};
bool joint_dense_layouts(dense_layouts & out, std::string & err);
size_t joint_dense_work_bytes();
size_t joint_ple_work_offset(); // FMU2 build item 4: PLE's joint products' region (fn::ple::products_joint)
// the routing buffers' bytes on a card for joint rows: IDs [20][10] I32, then weights [20][10] F32 (joint_route_carve);
// FMU2 (build item 1): then each lane's experts' GLU slots (codes [55][320], scales [55][10][2]) and routing meta
// (fn::experts::meta_bytes), so the lanes' gate/up and down run in one launch (fn::experts::gate_up_lanes)
size_t joint_route_bytes();
void joint_route_carve(void * base, int32_t *& ids, float *& wts);
// lane u's GLU slots and meta in the joint routing region (base: joint_route_bytes), into ex
void joint_lane_ex(void * base, int32_t u, ex_bufs & ex);
step_status layer_joint_step(joint_lanes & j, const layer_weights & w, const layer_state * const st[],
                             const layer_input in[], use_counters & u, std::string & err);
// Both cards' joint lanes (j[0] on card 0, j[1] on card 1, the same lanes) in layer_pair_step's phases, sends before
// receives and the HC sublayers as modes 1 then 2, so the joint round also runs where kernels execute one at a time
// (Compute Sanitizer); layer_joint_step's launches on each card, each HC sublayer's mode-3 launch split in two.
// st[k], in[k]: card k's lanes'.
step_status layer_joint_pair_step(joint_lanes * const j[2], const layer_weights * const w[2],
                                  const layer_state * const * const st[2], const layer_input * const in[2],
                                  use_counters u[2], std::string & err);

} // namespace fn::model
