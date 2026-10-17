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
// State (R1's reserved buffers): GDN state [6][24][128 x 128] F32 a layer (logical version 0 the committed state,
// version v = 1..B the state after row v - 1 of the last step, at slot fn::round::version_slot(*card_ctx::cur, v); SR1)
// and convolution history [6][5,120][3] F32 alike (SP2: six versions); KV [cell][256] BF16 for K and V; the
// indexer's raw keys [cell][128] BF16 and F32 summaries. A commit moves the one committed-slot word every versioned
// object shares (fn::round::commit in the model pass, fn::round::advance in layer-test's runner).

#include "fn/kernels/dense/dense.h"
#include "fn/model/model.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
    float * gdn = nullptr;        // [5][24][128 x 128]
    float * conv = nullptr;       // [6][5120][3]
    uint16_t * k_cache = nullptr; // [cells][256] BF16
    uint16_t * v_cache = nullptr;
    uint16_t * idx_keys = nullptr;   // [cells][128] BF16 (index owner)
    float * idx_summaries = nullptr; // tiled F32 (index owner)
    int32_t n_cells = 0;
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
    uint32_t * cur;  // [1] the committed slot (layer-test's runner; the model pass uses fn::round::state::cur)
    // FFN: the router launch's outputs (the IDs and weights go to the experts' workspace)
    float * logits;   // [B][512]
    float * sg_logit; // [B]
    float * sg;       // [B] sigmoid of the shared gate: the shared expert's weight in EX2's down
};
size_t layer_ws_bytes();
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
layer_ws carve_layer_ws(void * base);

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
    // an optional observer, called after each piece is enqueued (layer-bench's per-piece events); null: none
    void (*observe)(void * user, int32_t card, const char * piece) = nullptr;
    void * observe_user = nullptr;
    // model-test's and layer-test's records: when set, a sublayer's full F32 mixed input [B][2560] is copied here
    // (device or pinned host memory, in stream order) after its body, before the next HC sublayer resets its X-HMIX
    // slot (HC3); null: none
    float * record_mixed[2] = {nullptr, nullptr};
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


} // namespace fn::model
