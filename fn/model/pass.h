#pragma once

// TG1: the target pass on both cards (FC-MODEL1): the token embedding (E1), layer 0, the PLE layer (E1), layers 1 .. 47
// (LP1's frame and bodies), the HC head, the vocabulary logits and greedy (V1), framed by the round's device descriptor,
// start, commit and completion (fn/kernels/round). The pinned order is qwen4exp.cpp's build loop (455-546 at 66e11c3):
// the embedding into all four streams, PLE before layer 1's first HC mix, the head mixer as the output norm.
//
// Host code only. Everything that needs N_before, positions, tokens or the PLE use reads them on the device from the
// round descriptor (FC-RD1), so a pass enqueued into a CUDA graph serves every depth and round. Two ways to enqueue:
//   card:  one card's whole pass on its stream, for a graph capture (one graph per card per B);
//   pair:  both cards' passes in phases, every send enqueued before the receive that reads it (the order
//          layer_pair_step uses), so the pass also runs where kernels execute one at a time (Compute Sanitizer).
//
// Buffers: R1's reservation, with the "round and pass (TG1)" class (fn/model/state.h). The hidden carry (FC-HEAD3) is
// the head mixer's stream output x [B][4][1280], layer 47's output streams combined (x = hres + hpost * hx), which
// nothing else writes; it survives into the next round.

#include "fn/kernels/ple/ple.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/samp/samp.h"
#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/model/shapes.h"

#include <cstdint>
#include <string>

namespace fn::model {

// a recorded layer's output and its two HC mixes (model-test's records, copied in stream order after the layer)
struct pass_record_layer {
    int32_t layer = -1;
    float * x = nullptr;          // [B][4][1280] the FFN sublayer's streams (B <= 5)
    float * inject = nullptr;     // [B][4]
    float * out = nullptr;        // [B][1280]
    float * mixed_attn = nullptr; // [B][2560], or null
    float * mixed_ffn = nullptr;  // [B][2560], or null
};
constexpr int32_t pass_max_records = 4;

struct pass_options {
    // the layers the pass runs: lo .. hi (the racecheck option: 8 .. 11, with layer lo's input the embedding and no
    // PLE unless lo <= 1 <= hi); the whole model: 0 .. 47
    int32_t layer_lo = 0;
    int32_t layer_hi = fn::n_layer - 1;
    // model-test's records on each card, copied (device to device) after their layers; null: none
    const pass_record_layer * records[2] = {nullptr, nullptr};
    int32_t n_records = 0;
    // SM1: the request's sampler (FC-DEC2): the decision's stages (samp.h) after the logits, with the decision hash;
    // null: V1's greedy as TG1 built it (model-test, model-bench). Forced rounds decide nothing either way.
    const fn::samp::options * sampling = nullptr;
};

// one card's resources for the pass
struct pass_card {
    card_ctx c;
    layer_weights w[fn::n_layer];
    layer_state s[fn::n_layer];
    // the embedding (FC-EMB1): token_embd's card slice, and layer 0's input boundary
    const uint8_t * token_embd = nullptr;
    float * emb_hres = nullptr;  // [5][4][1280] (SP2: five rows)
    float * emb_hpost = nullptr; // [5][4], written 0
    float * emb_hx = nullptr;    // [5][1280], written 0
    // the PLE layer (FC-PLE1, FC-PLE2)
    fn::ple::weights ple_w;
    float * ple_e = nullptr;    // [5][2560] the upload slot on this card
    float * ple_hist = nullptr; // [6][5120][9]
    uint8_t * ple_ws = nullptr;
    // the head (FC-HEAD1..3)
    hc_module head;                  // w_down [4][320][1280], w_up, gamma
    const uint8_t * out_w = nullptr; // output.weight's card rows, DP2's packed layout
    float * carry = nullptr;         // [5][4][1280] the hidden carry (FC-HEAD3)
    float * head_mom = nullptr;      // [5][1300]
    uint32_t * pair_own = nullptr;   // [5][2]
    float * logits = nullptr;        // [5][124160]
    void * greedy_work = nullptr;
    fn::samp::card_view samp; // SM1: the sampler's buffers (R1's "sampler (SM1)") and its exchange slots
    // the round
    fn::round::desc * desc = nullptr;
    fn::round::state * rstate = nullptr; // with cur, the committed slot every versioned object shares (SR1)
    const uint32_t ** errs = nullptr;    // device: the error words commit checks (0: the card's exchange word)
    int32_t n_err = 0;
    uint32_t * err_words[fn::round::max_err] = {}; // the same words, held by the host (it clears them, FC-RD2)
    // the host side of the round (mapped host memory, device pointers)
    const fn::round::desc * host_ring = nullptr;
    const uint64_t * flag = nullptr;
    fn::round::done * done = nullptr;
};

// Resolves one card's pass: the frame (resolve_card), every layer's weights and state, the embedding, PLE and head
// tensors, the reservation's buffers; writes the commit's error list, zeroes the PLE workspace, the
// greedy work and the frame's zero rows, and sets the round state's counter to next_round and its committed slot to 0. The host ring, flag and
// completion record are the runtime's (mapped host memory). The caller has packed every dense weight
// (pack_dense_all). Synchronous.
bool pass_init(model & m, int32_t card, void * stream, const fn::round::desc * host_ring, const uint64_t * flag,
               fn::round::done * done, uint64_t next_round, pass_card & p, std::string & err);

// DP2's packed layout for every dense weight of the pass (the 48 layers' and output.weight), once per loaded model
bool pack_dense_all(model & m, std::string & err);

// One card's pass of B rows, for a graph capture: begin, stamp, embed, layers lo .. hi with PLE after layer 0, the head,
// logits, the decision (V1's greedy, or the sampler's stages, o.sampling), commit, finish. u counts the exchange uses
// (u.graph: slots by position in the graph).
step_status pass_card_enqueue(pass_card & p, int32_t rows, const pass_options & o, use_counters & u, std::string & err);

// Both cards' passes in phases (sends before receives), eagerly
step_status pass_pair_enqueue(pass_card * const p[2], int32_t rows, const pass_options & o, use_counters u[2],
                              std::string & err);

// The bootstrap round (FC-RD3, FC-PF2): no layer work; the decision (greedy, or the sampler's, o.sampling) on row
// `row` of the retained logits (the last prompt row), both cards in phases, then commit (k = 0) and finish. Eager.
bool pass_bootstrap_enqueue(pass_card * const p[2], int32_t row, const pass_options & o, std::string & err);

// the GPU use counters of one round of the whole pass in a graph: HC sublayers (96 and the head), X-A, X-F, X-SELECT
use_counters pass_graph_counters(const pass_options & o);

// ---- FMU2: the joint verify round on one card (fn/model/layer.h joint_lanes) ----
//
// 2..4 slots' rounds of layer_max_rows rows, one graph a card: every lane's begin and embedding, then each layer joint
// (layer_joint_step; PLE per lane after layer 0), then the head HC of every lane in one launch (hc::head_sub_joint)
// and each lane's logits, decision, commit and finish. A lane's pieces outside the joint launches are its slot's pass
// with the lane frame, so each lane's round reads its own descriptor, writes its own state, logits, carry and sampler
// record and finishes into its own completion record; its lane ring sees the lone pass's sequence (slot rules by
// position, pass_graph_counters). The whole model only (layers 0 .. 47, no records).
struct joint_pass {
    int32_t lanes = 0;
    pass_card * lane[fn::xchg::max_lanes] = {};       // pass_lane_init's
    const pass_options * o[fn::xchg::max_lanes] = {}; // each lane's slot's (its sampler)
    joint_lanes j;                                    // pass_joint_enqueue points j.lane[u] at lane[u]->c
};

// Lane u's pass from its slot's (slot), on the same card: the lane ring (jr.lane(u)) for the frame, the sampler's
// exchange slots and error word; the joint layer_ws's lane view (lane_ws); the joint routing's rows 5u .. (ids
// [20][10], wts [20][10]) and the lane's GLU slots and meta (joint_lane_ex), from route (joint_route_bytes, on the
// card); the commit's error list in errs (device memory, fn::round::max_err pointers on the
// card): the lane ring's word, attend's, select's, the experts' status and the joint word. Synchronous.
bool pass_lane_init(const pass_card & slot, const fn::xchg::joint_rings & jr, int32_t u, const layer_ws & joint_ws,
                    void * route, const uint32_t ** errs, pass_card & out, std::string & err);

// One card's joint round, for a graph capture; u: the joint graph's counters (pass_graph_counters of the whole pass)
step_status pass_joint_enqueue(joint_pass & jp, use_counters & u, std::string & err);

// Both cards' joint rounds in phases (sends before receives, HC modes 1 then 2 on both cards), eagerly: the eager
// engine's and Compute Sanitizer's joint round, as pass_pair_enqueue is the lone pass's
step_status pass_joint_pair_enqueue(joint_pass * const jp[2], use_counters u[2], std::string & err);

} // namespace fn::model
