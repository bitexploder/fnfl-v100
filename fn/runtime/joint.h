#pragma once

// FMU2: the joint verify round (FMU1 ticket section 6, stage 2) on resident TP: 2..4 slots' verify rounds at five rows
// each (d = 4) run as lanes of one graph a card (fn/model/pass.h pass_joint_enqueue), every dense product once for
// all lanes (fn::dense::product_joint), each lane's other pieces its slot's on its own ring (fn::xchg::joint_rings).
// Each member keeps its own round: its descriptor, start flag, completion record, versions and FC-RD2 recovery
// (target::joint_verify_begin, joint_release, joint_verify_end), so every lane's tokens are its lone round's (I3).
//
// Graphs: one pair a membership (the members' slots in lane order, each slot's decision variant and expert map),
// captured at the membership's first round and kept in a small cache (least recently used out), since a captured
// graph names its members' state (no slot table: the deviation from section 6 the Result records). The resources are
// allocated once at engine init and listed in its startup ledger (bytes()); graph bytes are measured at capture.
// On the eager engine (Compute Sanitizer's runs) the round is enqueued both cards in phases instead
// (pass_joint_pair_enqueue): no graphs.
//
// Host code only.

#include "fn/model/model.h"
#include "fn/model/pass.h"
#include "fn/runtime/target.h"
#include "fn/xchg/xchg.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fn::runtime {

class joint_frame {
  public:
    joint_frame() = default;
    ~joint_frame();
    joint_frame(const joint_frame &) = delete;
    joint_frame & operator=(const joint_frame &) = delete;

    // The joint rings, each card's joint layer_ws, dense work, routing and lanes' error lists, on the engine's
    // streams; cache: graph pairs kept (1 .. 8); eager: no graphs, each round enqueued in phases. Synchronous.
    bool init(model::model & m, void * const streams[2], int32_t cache, bool eager, std::string & err);
    bool ready() const { return rings_ != nullptr; }
    bool eager() const { return eager_; }

    // One joint round of n (2..4) members t[k], each a verify of d[k] = 4 proposals prop[k] under budget[k]: the
    // graphs (captured on the membership's first round), every member staged, both cards' graphs launched, every
    // member released, every member collected into r[k] / ok[k] / why[k] (a member's failure leaves the others' results
    // valid). False, with err, when the round did not launch: no member's state or ledger changed then, except as
    // `staged` says (its members were abandoned; the engine treats them as lost).
    bool verify(target * const t[], int32_t n, const int32_t d[], const int32_t * const prop[], const int32_t budget[],
                round_result r[], bool ok[], std::string why[], bool & staged, std::string & err);

    // FMU2 (build item 2): one joint draft step of n (2..4) members t[k], each a draft step of d (1..4) chained steps
    // from its draft_n() as row k of one step (fn/model/draft.h draft_joint): the graphs (captured on the
    // membership's first joint draft, a cache like the verify's), every member staged, both cards' graphs launched,
    // every member released and collected into r[k] / ok[k] / why[k] (a member's failure leaves the others' results
    // valid). False, with err, when it did not launch: no member's state changed, so each drafts alone.
    // draft_fits: every member's pick greedy, or every member's sampled (FMU4), and the members in one draft attention
    // form (else they draft alone).
    bool draft(target * const t[], int32_t n, int32_t d, round_result r[], bool ok[], std::string why[],
               std::string & err);
    static bool draft_fits(target * const t[], int32_t n);
    int32_t draft_captures() const { return draft_captures_; }
    int64_t draft_rounds() const { return draft_rounds_; }

    // the resources' bytes on a card (init's), and the cached graphs' (measured at capture)
    int64_t bytes(int32_t card) const { return bytes_[card]; }
    int64_t graph_bytes(int32_t card) const;
    int32_t captures() const { return captures_; }
    int64_t rounds() const { return rounds_; }
    // the last verify did not launch because its membership's graphs failed to capture
    bool capture_failed() const { return capture_failed_; }
    // the last capture's free memory left on each card (cudaMemGetInfo after its upload), and its graph's nodes
    int64_t free_after_capture(int32_t card) const { return free_after_[card]; }
    int64_t last_nodes(int32_t card) const { return nodes_[card]; }
    int32_t last_lanes() const { return last_lanes_; }

  private:
    struct graph_pair {
        std::vector<int32_t> key; // per lane: slot, variant, expert map
        void * graph[2] = {nullptr, nullptr};
        void * exec[2] = {nullptr, nullptr};
        int64_t bytes[2] = {0, 0};
        uint64_t used = 0;
    };
    bool lanes(target * const t[], int32_t n, int32_t c, std::vector<std::unique_ptr<model::pass_card>> & lane,
               model::joint_pass & jp, std::string & err);
    bool capture(target * const t[], int32_t n, const std::vector<int32_t> & key, graph_pair & g, std::string & err);
    bool enqueue_eager(target * const t[], int32_t n, std::string & err);
    void draft_lanes(target * const t[], int32_t n, int32_t c, model::draft_joint & j);
    bool capture_draft(target * const t[], int32_t n, int32_t d, const std::vector<int32_t> & key, graph_pair & g,
                       std::string & err);
    graph_pair * cached(std::vector<graph_pair> & cache, const std::vector<int32_t> & key);
    void drop(graph_pair & g);
    bool clear_after_error(std::string & err);

    model::model * m_ = nullptr;
    void * st_[2] = {nullptr, nullptr};
    std::unique_ptr<fn::xchg::joint_rings> rings_;
    void * ws_[2] = {nullptr, nullptr};    // the joint layer_ws (fn::xchg::joint_rows rows)
    void * work_[2] = {nullptr, nullptr};  // the joint dense work
    void * route_[2] = {nullptr, nullptr}; // the joint routing (IDs, weights)
    void * errs_[2] = {nullptr, nullptr};  // max_lanes error lists of fn::round::max_err pointers
    void * jd_[2] = {nullptr, nullptr};    // the joint draft's rows: tokens, retained carries, head output, the word 1
    model::layer_ws lws_[2];
    model::dense_layouts layouts_{};
    uint64_t ple_use_[fn::xchg::max_lanes] = {}; // each lane ring's PLE uses (its X-PLE-STATS slot sequence)
    std::vector<graph_pair> cache_;
    std::vector<graph_pair> dcache_; // the joint draft's graphs (key: per lane its slot, then d, the draft form and
                                     // the pick kind)
    int32_t cache_max_ = 2;
    uint64_t use_clock_ = 0;
    int64_t bytes_[2] = {0, 0};
    int64_t free_after_[2] = {0, 0};
    int64_t nodes_[2] = {0, 0};
    int32_t last_lanes_ = 0;
    int32_t captures_ = 0;
    int32_t draft_captures_ = 0;
    int64_t draft_rounds_ = 0;
    int64_t rounds_ = 0;
    bool capture_failed_ = false;
    bool eager_ = false;
};

} // namespace fn::runtime
