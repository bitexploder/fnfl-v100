#pragma once

// XP1: layers 10 (GDN) and 11 (QSA) decoding on both cards with synthetic weights in their deployed formats and no
// model load, in R2's order and with R2's exchanges, for three coordination variants:
//   V0  R2's shape: separate send and receive kernels for X-A, X-F, X-ROUTE and X-SELECT (C2's send_select and
//       recv_select with QA1b's compact body); the router and the indexer run on their owner only.
//   V1  X-A and X-F sent from the epilogue of the last kernel that produces each partial (put_run into the peer's
//       slot): the output projection stand-in for T-A, the shared expert's down stand-in (after the routed experts,
//       adding their partial) for T-F.
//   V2  V1 + the router stand-in on both cards, from the full router input each holds after X-HMIX; no X-ROUTE.
// (XP1's V3 and V4, the split indexer, were rejected in fable's review and removed in Amendment 4.)
// Real FNFL kernels: HC2 (down2, up2 arm 0, ready, with X-HC and X-HMIX), C2's activation quantize (fn::act, the
// stand-in producer before EX1b; HC2's epilogue writing the format is not built), EX1b (gate_up, down), QA1b
// (summary_update, select writing the compact body), fn/xchg's exchanges. Load stand-ins (fn/proto/xp1/standin.h)
// for every piece FNFL has not built. Ownership is the frame's: router_owner = layer % 2, index_owner = (layer / 4) %
// 2. The QSA index holds 58,581 cells of synthetic keys; layer 11 decodes rows at positions 58,581 .. 58,581 + B - 1.
// Amendment 3's traffic (hx7): the router emits HX7's captured experts and the attention stand-in reads HX7's captured
// cells, decode r mod 1,024 for round r; with synthetic traffic it reads each row's blocks from the body.
// DP2 (pair_config::dense, V2 only): DP2's launches (fn/kernels/dense) on synthetic weights in the deployed formats
// replace the stand-ins whose work is wholly a dense product: gdn_out and qsa_out (ssm_out and attn_output, each after
// C2's quantize of its input as the producer epilogue's stand-in), the router (DP2's router and shared-gate launch on
// both cards) and the shared expert (gate/up with the GLU epilogue, then the down scaled by the gate's sigmoid into the
// shared buffer EX1b's down adds); index_q_proj too, whose norm QA1b's select applies. The GDN input, QSA input and
// index key stand-ins keep only their piece's bytes outside DP2's input launch (dense_rest), unpadded. Neither DP2 nor
// EX1b sends from its epilogue, so X-A and X-F go by V0's send kernels. With HX7 traffic the round node copies the
// decode's captured IDs for EX1b; DP2's router writes its own.
//
// A program is both cards' operations in one list that is also a valid sequential schedule (every send before the
// receive that reads it), so it runs eagerly under Compute Sanitizer; a graph capture takes each card's operations in
// order. The cards meet only through fn/xchg.

#include "fn/proto/xp1/hx7.h"
#include "fn/proto/xp1/standin.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fn::experts {
struct synth;
}
namespace fn::dense {
struct synth;
}

namespace fn::xp1 {

enum class variant : int32_t { v0, v1, v2 };
constexpr int32_t n_variant = 3;
const char * variant_name(variant v);
bool parse_variant(const std::string & s, variant & v);

enum class traffic : int32_t { hx7, synthetic };

constexpr int32_t pair_depth = 58581;   // cells in the history before the rows
constexpr int32_t pair_cells = 58624;   // the KV and index buffers: 229 groups of 256 cells
constexpr int32_t max_rows = 4;
constexpr int32_t gdn_state_floats = 24 * 128 * 128; // one version of the card's GDN state
constexpr int32_t conv_floats = 5120 * 3;            // one version of the card's convolution history
constexpr int32_t proj_width = 5120 + 3072 + 48;     // gdn_proj_conv's output: conv out | z | beta, g

struct op {
    int32_t card = 0;
    int32_t layer = 0;
    std::string piece;   // the label the trace split uses
    std::string role;    // the exchanges it sends or receives: "send X-A", "recv X-HC; send X-HMIX", ...
    int32_t launches = 0; // kernel nodes it adds (0: a record copy, eager only)
    std::function<bool(void *)> fn;
};

struct pair_config {
    int32_t dev[2] = {0, 1};
    traffic tr = traffic::hx7;
    std::string synth_dir; // EX1b's synthetic expert weights (generated there if missing)
    std::string hx7_dir;   // HX7's capture
    bool glu_f32 = false;  // EX1b's test-only F32 GLU output (xp1-test)
    bool dense = false;    // DP2's launches in V2 (xp1-bench --dense)
    // GD1 step 7: 0, layer 10's recurrence stand-in (XP1); 1, GD1's gdn_core in its place, the projections' stand-in
    // without the convolution history and its weights (gdn_core's share); 2, as 1 with gdn_core omitted (the knock-out
    // its in-pair time is measured against)
    int32_t gdn = 0;
    // QA2 step 7 (with dense): 0, layer 11's QSA attention stand-in (XP1); 1, QA2's attend in its place on both cards
    // (it writes attn_out and its activation format, so the stand-in quantize before attn_output goes, and the index
    // owner's summary_update carries the raw index-key write every round); 2, as 1 with attend omitted (the knock-out)
    int32_t qsa = 0;
};

// DP2's dense mode: the pieces DP2's launches replace, and those that keep a stand-in for the rest of their bytes
bool dense_replaced(piece p);
bool dense_rest(piece p);

// what a recorded eager step leaves per card (pinned host memory)
struct card_record {
    float x[4][max_rows * 4 * 1280]; // per sublayer k: the new streams, injection, full mixed input, completed half
    float inject[4][max_rows * 4];
    float full[4][max_rows * 2560];
    float out[4][max_rows * 1280];
    float recur_out[max_rows * 3072]; // X-A's producers' inputs (layer 10, layer 11)
    float attn_out[max_rows * 3072];
    // per FFN (layer 10, 11): EX1b's input in FNFL's format (codes, scales), routing, GLU (codes, scales), the shared
    // expert
    int8_t xc[2][max_rows * 2560];
    float xs[2][max_rows * 80 * 2];
    int32_t ids[2][max_rows * 10];
    float wts[2][max_rows * 10];
    float sg[2][max_rows];
    int8_t gluc[2][44 * 320]; // EX2's GLU slots: the routed pairs, then the shared expert's rows
    float glus[2][44 * 10 * 2];
    float glu_f32[2][44 * 320];
    float sh_glu[2][max_rows * 320];
    // the selection this card's attention reads: QA1b's body and byte offsets on the index owner, the received X-SELECT
    // on the peer
    uint32_t body[max_rows * 517];
    int32_t off[max_rows + 1];
    uint32_t sink[n_piece * si_blocks * sink_block_words]; // every piece's sink words (the last run of each)
    uint32_t err;
};

class pair {
  public:
    pair();
    ~pair();
    pair(const pair &) = delete;
    pair & operator=(const pair &) = delete;

    bool init(const pair_config & cfg, std::string & err);

    // The program of variant v at B rows. graph: slots by position in a replayed graph (HC2's rule), else by the
    // running use counts. rec (eager only): record copies after the pieces the test checks, into rec[card].
    bool build(variant v, int32_t rows, bool graph, std::vector<op> & prog, std::string & err,
               card_record * const rec[2] = nullptr);
    bool run_eager(const std::vector<op> & prog, std::string & err);
    bool capture(const std::vector<op> & prog, std::string & err);
    // both cards' graphs once: ms[card] from events around each launch, start_ns[card] the round stamp (each card's
    // own globaltimer)
    bool replay(float ms[2], uint64_t start_ns[2], std::string & err);
    void drop_graphs();

    bool reset_exchange(std::string & err); // drains, refills the rings, resets the use counts
    bool reset_round(std::string & err);    // the decode counter to 0 on both cards
    bool set_round(int32_t r, std::string & err); // the decode counter to r on both cards
    // after a graph replay: each card's residual half after layer 11 (into x[3], inject[3], out[3]) and the selection
    // its attention reads (body, off), synchronously
    bool read_final(int32_t rows, card_record * const rec[2], std::string & err);
    bool reset_state(std::string & err);    // the cells at and after the depth cleared (KV, index keys)
    bool sync(std::string & err);

    void set_pad(piece p, int32_t card, int32_t pad);
    int32_t pad(piece p, int32_t card) const;
    traffic traffic_mode() const { return cfg_.tr; }
    void set_traffic(traffic t) { cfg_.tr = t; }

    // the bench's handles on a card
    int32_t device(int32_t card) const;
    void * stream(int32_t card) const;
    uint64_t * stamps(int32_t card) const; // 64 words of device memory
    void * flush_buffer(int32_t card) const;
    int64_t flush_size() const;
    // card 1's globaltimer minus card 0's, and the round trip, from n trips through peer memory
    bool clock_offset(int32_t n, double & offset_ns, double & rtt_ns, std::string & err);

    // An exchange isolated on both cards, both directions at once, rep times back to back with rotating slots (each
    // send follows its card's receipt of the peer's previous message): "X-A", "X-F", "X-ROUTE", "X-SELECT", "X-A V1"
    // (the layer-11 output stand-in sending from its epilogue, then the receive),
    // "HC" (HC2's sublayer, 2 rep sublayers; rep a multiple of 3). Each card's ops in order, for a graph.
    bool exchange_ops(const std::string & name, int32_t rows, int32_t rep, std::vector<op> & ops, std::string & err);

    // a piece alone on one card (its stand-in as the program would launch it at B rows), for calibration and the
    // alone figures
    bool piece_op(piece p, int32_t card, int32_t rows, op & o, std::string & err);

    // host-side facts for the checks and the report
    struct facts;
    const facts & info() const;

  private:
    bool piece_args(piece p, int32_t card, int32_t rows, standin_args & a) const;

    struct impl;
    std::unique_ptr<impl> m_;
    pair_config cfg_;
};

// the bytes of a piece's stand-in on one card: weights from R1's memory plan (fn/model/placement's card bytes; for a
// card that does not own the router in S2, the owner's), and its state or cell bytes at B rows
int64_t piece_weight_bytes(piece p, int32_t card);

struct pair::facts {
    int32_t dev[2];
    std::vector<int32_t> map;           // expert ID -> synthetic expert (EX1b's synth)
    const fn::experts::synth * gate_up[2] = {nullptr, nullptr}; // layers 10 (IQ2_XS) and 11 (IQ2_XXS)
    const fn::experts::synth * down = nullptr;                   // Q2_0, both layers
    // EX2's shared expert (layer 10: IQ3_S gate, IQ4_XS up; layer 11: Q4_K, Q4_K; IQ4_NL down): DP2's synthetic rows.
    // Card r's gate row i is sh_gate[l]'s row (i + sh_off[l][r][0]) % n_synth_rows, its up row (i + sh_off[l][r][1]),
    // and its down row n (the card's 320 columns) sh_down's row (n + sh_off[l][r][2]) % n_synth_rows.
    const fn::dense::synth * sh_gate[2] = {nullptr, nullptr};
    const fn::dense::synth * sh_up[2] = {nullptr, nullptr};
    const fn::dense::synth * sh_down = nullptr;
    int32_t sh_off[2][2][3] = {};
    uint32_t w_xor[n_piece][2][4];      // the XOR of each piece's weight words, per card
    hx7_traffic hx7;
    // host copies for the references
    std::vector<uint16_t> hc_down[4], hc_inject[4], hc_up[4]; // GGUF layouts per sublayer (hc_test's)
    std::vector<float> hc_gamma[4];
    std::vector<float> in0[max_rows + 1]; // the teacher-forced layer-10 input per B: [B][4][2560], full width
    float c_piece[n_piece];
};

} // namespace fn::xp1
