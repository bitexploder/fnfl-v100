#pragma once

// DR1 Stage C: the MTP draft block on both cards (FC-DR1, FC-DR2), framed by the round as the target pass is
// (fn/model/pass.h), in two round kinds (fn/kernels/round/round.h):
//   draft_step     rows = d chained draft steps of B = 1 at positions n_before + j: the embedding of the step's token
//                  (desc tok[0], then the previous step's pick desc argmax[j - 1]), X-DRAFT-IN of [e | h] (h: the
//                  retained carry at step 0, then the draft's own hidden output), eh_proj, the attention sublayer
//                  (HC in Q8_0, q/k/v, DA-TC with its K/V cache write at the step's cell, the output product, X-A), the
//                  FFN sublayer (HC, the F32 router, the Q8_0 experts, X-F), the draft's HC head, the logits on this
//                  card's re-sharded copy of half the draft subset and the greedy pick over the subset (lower ID on an
//                  exact tie), written to desc argmax[j] on both cards; at T > 0 (draft_card::sampled, SP1) SM1's
//                  top-k pick instead, which also writes q_list[j]. The cells it writes are speculative: the
//                  catch-up writes them again once the target has committed them.
//   draft_catchup  rows = B_c committed positions n_before + t (tokens desc tok[t]): the embedding, X-DRAFT-IN with
//                  the target's carry rows (row hsrc + t - 1; -1: the retained carry), eh_proj in B_c calls (Stage C
//                  amendment 1), the attention sublayer's HC and the K/V products, DA-TC's K/V write (NOREAD: no
//                  attention, FFN, head or pick); the retained carry becomes carry row hsrc + B_c - 1.
// Both kinds commit no version (k = 0) and emit nothing; their error words are the target's commit list (the exchange
// word carries err_poll_draft and err_draft_depth).
//
// Exchange slots (xchg.h, "The draft rounds"): X-HC / X-HMIX by hc_rule (a step graph from slot 0 to slot 1, the
// catch-up's one sublayer on slot 2, released by its K/V prepare); X-DRAFT-IN, X-A and X-F by use (step j: slot j % 3,
// the catch-up 0); T-GREEDY slot 0. Eagerly every slot is the use counter % 3, continuing the target's counters.
//
// Buffers: R1's "draft round (DR1)", "draft KV (F16)" and "draft head copy (DR1)" classes; the layer workspace, the
// embedding boundary, the head's moments and pairs and the greedy work are the target pass's, free between its rounds.

#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/model/pass.h"
#include "fn/xchg/xchg.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::model {

constexpr int32_t draft_max_steps = 4; // d (SP2: 1..4)

// one card's draft block resources
struct draft_card {
    draft_weights w;
    const uint8_t * head_w = nullptr;   // the draft head copy: 32,768 Q5_K rows, DP2's packed layout
    const int32_t * head_ids = nullptr; // their global IDs
    uint16_t * k_cache = nullptr;       // F16 [max_cells][256] (layer 48's KV head of this card)
    uint16_t * v_cache = nullptr;
    int32_t max_cells = 0;
    float * carry_prev = nullptr; // [4][1280] the retained carry: the target's row before the round's first
    float * h_nextn = nullptr;    // [4][1280] the draft's hidden output (the head mixer's streams)
    float * in_own = nullptr;     // [5][6408] X-DRAFT-IN's own payload (SP2: rows up to fn::draft::max_rows)
    int8_t * codes = nullptr;     // [5][4][5120] eh_proj's input
    float * scales = nullptr;     // [5][4][160][2]
    float * hres_d = nullptr;     // [5][4][1280] eh_proj's output: the attention sublayer's input streams
    uint16_t * q16 = nullptr;     // [5][12][256]
    float * part = nullptr;       // DA-TC's partials
    float * logits = nullptr;     // [4][32768] each step's logits over this card's half of the subset
    void * logits_work = nullptr;
    // SP1: T > 0 on the sampler's top-k path: step j's pick is SM1's (samp::pick over the ID map), which writes the
    // proposal and the retained q (q_list[j]) for verify row j, its Philox position read live from the descriptor
    // (n_before + j + 1); false: the greedy pick over the subset (T = 0, DR1 Stage C)
    bool sampled = false;
    // ME8: set by draft_pair_enqueue for card 1 while it runs on card 0's device and stream: card 0's routing
    // (route_view, non-owning), read by this card's experts in place of a second router; null otherwise
    const route_view * shared_route = nullptr;
    // SP2 (sparse draft attention, temporary at SP2's disposition): the draft's own index (blk.48.indexer, the
    // target QSA's math) on this card. sparse: this capture's mode, in which step 0 adds the index query to its q/k/v
    // product, selects up to 512 whole blocks of the prefix (fn::qsa::select at depth n_before - 1, the query at
    // n_before) and every step attends those blocks' cells, the prefix's tail and the chain's cells (DA-TC's sparse
    // form); the catch-up adds the index key to its k/v product and writes the keys and pools the summaries
    // (fn::qsa::summary_update's round form). false: DR1's dense form; the index is neither read nor written.
    bool sparse = false;
    draft_index_weights iw;        // card 0: the GGUF regions; card 1: the "draft index (SP2)" copies
    uint16_t * idx_keys = nullptr; // BF16 [max_cells][128]
    float * idx_summ = nullptr;    // tiled F32 summaries
    int32_t * sel_depth = nullptr; // device: n_before - 1, step 0's select depth
};

// Resolves the draft weights (pack_draft must have run), the buffers and the KV cache of card p.c.card; zeroes the
// round buffers. Synchronous.
bool draft_init(model & m, const pass_card & p, draft_card & d, std::string & err);

// The draft head copy on both cards from output.weight's GGUF rows (the loaded files' mapping) and the subset (65,536
// distinct IDs): head::draft_ids, head::pack_draft_head, uploaded. Without a mapped file (a dry load) the copy is
// zeroed and the ID map is still written. Synchronous.
bool draft_head_fill(model & m, const std::vector<int32_t> & subset, std::string & err);

// The draft subset: a text file of 65,536 distinct IDs, one a line (work/DR1/stageC/subset/draft-subset.ids)
bool read_draft_subset(const std::string & path, std::vector<int32_t> & ids, std::string & err);

// The graph counters of a draft round: hc_rule draft_step (3 d sublayers) or catchup (one)
use_counters draft_graph_counters(int32_t kind, int32_t rows);

// One card's draft round for a graph capture (kind draft_step: rows = d; draft_catchup: rows = B_c): begin, stamp,
// the work above, commit, finish. u: the card's counters (u.graph: slots by hc_rule and position)
step_status draft_card_enqueue(pass_card & p, draft_card & d, int32_t kind, int32_t rows, use_counters & u,
                               std::string & err);

// Both cards in phases (every send enqueued before the receive that reads it), eagerly
step_status draft_pair_enqueue(pass_card * const p[2], draft_card * const d[2], int32_t kind, int32_t rows,
                               use_counters u[2], std::string & err);

// FMU2 (build item 2): the joint draft step. U (2 .. 4) slots' draft steps of d chained steps each (all greedy, or
// all sampled (FMU4); one draft attention form) run as U rows of one step on lane 0's slot frame: the shared rings,
// workspaces and draft weights every slot's rounds use, so one weight stream serves every row and a shared piece's
// error word is in every lane's commit list. Row k is lane k's:
//   lane k's own: begin and stamp, commit and finish (its descriptor, round state and completion record); DA-TC's
//                 prepare and attend at its cells, positions and KV cache (its q16 and partials); at SP2's sparse
//                 step 0 the select over its draft index (its body and offsets in sel_body[k], sel_off[k]).
//   U rows:       the embedding (its token: desc tok[0], then its step j - 1 pick), X-DRAFT-IN (its hidden input: its
//                 retained carry at step 0, then its own head output row), eh_proj, both HC sublayers, the q/k/v,
//                 output, router and expert products, X-A, X-F, the HC head, the logits and the greedy pick (lower ID
//                 on an exact tie), whose kernels are bitwise per row at B = 1 .. 4 (each row its own launch's bits).
//   sampled (FMU4): the logits of the U rows in one launch, then lane k's own SM1 pick on row k (its sampler view,
//                 options, q_list[j], live Philox position, penalty deltas and desc argmax[j]; its list in row k of
//                 samp_a's slot, samp::pick_lanes_stage), the proposals gathered into tok row j + 1.
// The picks go to each lane's desc argmax[0 .. d - 1] before its commit. Buffers: tok, carry, h and the device word
// one (1) on this card, the joint frame's (fn/runtime/joint.h); lane 0's draft_card's input payload, eh_proj
// buffers and logits rows hold the U rows.
struct draft_joint {
    int32_t lanes = 0;
    pass_card * p[fn::xchg::max_lanes] = {};  // each lane's slot pass on this card
    draft_card * d[fn::xchg::max_lanes] = {}; // and its draft block
    int32_t * tok = nullptr;                  // device [draft_max_steps + 1][max_lanes]: step j's tokens row j
    float * carry = nullptr;                  // [max_lanes][4][1280] the lanes' retained carries (step 0's hidden)
    float * h = nullptr;                      // [max_lanes][4][1280] the head's output rows (steps 1 .. d - 1)
    const int32_t * one = nullptr;            // device word 1: X-DRAFT-IN reads hidden row i at h or carry row i
    uint32_t * sel_body[fn::xchg::max_lanes] = {};
    int32_t * sel_off[fn::xchg::max_lanes] = {};
};

// One card's joint draft step of d = rows steps for a graph capture (u: draft_graph_counters(draft_step, rows)), and
// both cards in phases eagerly (u: the shared rings' counters, continued). False (failed) with nothing enqueued for
// lanes outside 2 .. 4, rows outside 1 .. 4, lanes with different pick kinds or draft forms, or the host-backed KV.
step_status draft_joint_enqueue(draft_joint & j, int32_t rows, use_counters & u, std::string & err);
step_status draft_joint_pair_enqueue(draft_joint * const j[2], int32_t rows, use_counters u[2], std::string & err);

// FMS2 (build item 2): the joint catch-up. Lanes 2 .. 4, each a slot's catch-up of its own B_c rows (1 .. 5: its
// descriptor's rows, read on the device, so one captured graph serves every round of the membership), in one launch
// sequence a card on the joint frame: lane k's pass is its slot's with the lane frame (pass_lane_init: joint_rings'
// lane k, the lane's rows of the joint layer_ws, its error list), its draft block its slot's. Per lane, its own
// kernels: begin and stamp, the embedding (all five rows; the rows past B_c embed token 0 into the slot's boundary,
// which nothing reads), eh_proj's B_c calls (fn::draft::eh_proj_gated), commit and finish. Every lane in one launch:
// X-DRAFT-IN's send and receive (with the retained carry's update) and the K/V prepare (with the release of the
// lane's slot 2) at each lane's B_c, the draft index's keys and summaries (sparse) at each lane's B_c, the HC
// sublayer at B = 5 (fn::model::hc_sublayer_lanes), the K/V (and index key) product at five rows a lane
// (dense_joint_run, acquiring each lane's X-HMIX). A lane's row below its B_c is bitwise its lone catch-up's (the
// same device code at its B_c, or a kernel whose rows are each the row's own at any B: hc-test's rows alone, dense.h);
// the rows past B_c of the HC and K/V work are computed from scratch and never read. Slots by the catch-up's rule on
// every lane ring (catchup_joint_counters): the lane rings are left clean. Resident KV only, one draft form.
struct catchup_joint {
    int32_t lanes = 0;
    pass_card * p[fn::xchg::max_lanes] = {};
    draft_card * d[fn::xchg::max_lanes] = {};
    const fn::xchg::joint_rings * jr = nullptr;
    uint32_t * runs = nullptr;              // [max_lanes][fn::hc::n_runs_card] the lanes' up2 run counters, zeroed once
    void * kv_work[2] = {nullptr, nullptr}; // the K/V product's joint work, dense and sparse forms, zeroed once
    size_t kv_work_bytes[2] = {0, 0};
};
// the K/V product's joint work bytes of a draft form (fn::dense::joint_work_bytes of its shape)
size_t catchup_kv_work_bytes(bool sparse);
// the joint catch-up's counters, every round (a graph's, eagerly too): the catch-up's slot rule
use_counters catchup_joint_counters();
// One card's joint catch-up for a graph capture, and both cards in phases eagerly (the pair form's order). False
// (failed), with nothing enqueued, for lanes outside 2 .. 4, lanes on other cards, streams or draft forms, the
// host-backed KV, or a missing joint work region.
step_status catchup_joint_enqueue(catchup_joint & j, use_counters & u, std::string & err);
step_status catchup_joint_pair_enqueue(catchup_joint * const j[2], use_counters u[2], std::string & err);

} // namespace fn::model
