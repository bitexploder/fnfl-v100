#pragma once

// R2: a layer entry point for the test and the bench, not the seam. Two target layers (10, a GDN layer, and 11, a QSA
// layer) run on both cards as fn/model/layer.h builds them: each step decodes B rows at positions depth .. depth + B - 1,
// the first layer's input given (teacher-forced), the second's given or the first's own output (chained). Each card
// has its own stream; the cards meet only through fn/xchg.

#include "fn/model/layer.h"
#include "fn/model/model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime_layers {

constexpr int32_t max_rows = fn::model::layer_max_rows;

// What a step leaves for the checks, per layer and card, copied to pinned host memory in stream order.
struct layer_record {
    float x[max_rows * 4 * 1280];      // the FFN sublayer's streams (own half)
    float inject[max_rows * 4];        // its injection
    float out_ffn[max_rows * 1280];    // the FFN sublayer's completed own output half (X-F's result)
    float out_attn[max_rows * 1280];   // the attention sublayer's (X-A's result)
    float mixed_attn[max_rows * 2560]; // the full mixed inputs of the two sublayers (HC2's ready)
    float mixed_ffn[max_rows * 2560];
    int32_t ids[max_rows * 10]; // the experts the card ran (its own router's, XP1's V2)
    float wts[max_rows * 10];
    float logits[max_rows * 512];      // the router's scores
    uint32_t sel_body[max_rows * 517]; // the selection body (QSA layer: the owner's own, the peer's from X-SELECT)
    int32_t sel_off[max_rows + 1];     // its byte offsets (fn::qsa layout)
    // the GDN layer's intermediates on this card (debug: compared offline with the reference's debug dumps)
    float gdn_conv[max_rows * 5120]; // post-SiLU (the reference's conv_output_silu)
    float gdn_z[max_rows * 3072];
    float gdn_alpha[max_rows * 24]; // raw
    float gdn_beta[max_rows * 24];  // raw
    float gdn_gated[max_rows * 3072];
    int32_t sel_words;  // the body's words (0 on a GDN layer)
    uint32_t err;       // the card's exchange error word after the step
    uint32_t attn_err;  // attend's acquire error word (QSA layer)
    uint32_t sel_err;   // select's self-check error word (QSA layer, index owner)
    uint32_t ex_status; // EX1b's status word (an expert ID out of range)
};

// a piece's GPU time in a step, from CUDA events between consecutive pieces on one card (layer-bench)
struct piece_time {
    int32_t card;
    int32_t layer;
    std::string piece;
    float ms;
};

class pair_runner {
  public:
    pair_runner() = default;
    ~pair_runner();
    pair_runner(const pair_runner &) = delete;
    pair_runner & operator=(const pair_runner &) = delete;

    // streams, the layers' weights, state and workspaces on both cards
    bool init(model::model & m, int32_t layer0, int32_t layer1, std::string & err);
    // every state of both layers zeroed (KV, index keys and summaries, GDN state and history), the rings refilled,
    // the selection scratch cleared and the use counters reset; synchronous
    bool reset(std::string & err);
    // LP1's synthetic history (layer-bench): depth cells of random values on both cards, in place of a decoded one: the
    // QSA layer's K and V caches, the index owner's raw keys and summaries of the complete blocks, and the GDN layer's
    // committed state and convolution history (version 0); synchronous. The values only need to be finite: the bench
    // times the function proven by layer-test.
    bool fill_synthetic(int32_t depth, uint64_t seed, std::string & err);
    // both cards drained, every exchange slot refilled with the sentinel and the use counters reset (between eager
    // steps and graph replays, which leave different slots last written); synchronous
    bool reset_exchange(std::string & err);
    // one step on both cards: in0 [card] the first layer's input rows (own half, [B][4][1280]); in1 the second's, or
    // null for chained. rec[layer][card] receives the records. Synchronous: returns once both cards are done.
    bool step(int32_t rows, int32_t depth, const float * const in0[2], const float * const in1[2],
              layer_record * const rec[2][2], std::string & err);

    // per-piece times of one eager step (events after every piece), appended to out; synchronous
    bool step_timed(int32_t rows, int32_t depth, const float * const in0[2], std::vector<piece_time> & out,
                    std::string & err);

    // a CUDA graph per card of the pair (chained) at fixed rows and depth; the first layer's input staged once
    bool capture(int32_t rows, int32_t depth, const float * const in0[2], std::string & err);
    // replay both cards' graphs once; ms[card]: each card's span from its launch to its end (CUDA events)
    bool replay(float ms[2], std::string & err);
    void drop_graphs();

    // the last step, step_timed or capture stopped at a piece FNFL has not built (err names the pieces)
    bool not_built() const { return not_built_; }

    int32_t layer(int32_t k) const { return layer_[k]; }
    const model::layer_weights & weights(int32_t k, int32_t card) const { return w_[k][card]; }

  private:
    bool upload(int32_t rows, int32_t depth, const float * const in0[2], const float * const in1[2], std::string & err);
    bool sync(std::string & err);

    model::model * m_ = nullptr;
    int32_t layer_[2] = {-1, -1};
    void * st_[2] = {nullptr, nullptr};
    model::card_ctx cc_[2];
    model::layer_weights w_[2][2];
    model::layer_state s_[2][2];
    model::use_counters u_[2];
    float * in_dev_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}}; // [layer][card] [4][4][1280]
    float * in_host_[2] = {nullptr, nullptr};                         // pinned staging, [layer][4][4][1280] a card
    int32_t * pos_host_[2] = {nullptr, nullptr};                      // pinned [4][4]
    void * graph_[2] = {nullptr, nullptr};
    void * exec_[2] = {nullptr, nullptr};
    void * ev_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    bool not_built_ = false;
};

} // namespace fn::runtime_layers
