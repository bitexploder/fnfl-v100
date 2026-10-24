#pragma once

// TG1: the target-only round driver on both cards (FC-RD1..3, FC-SLOT1, FC-ST2). One graph per card per B in 1..5,
// captured once (fn/model/pass.h), runs every forced and generate round at any depth: the host writes descriptor slot
// round % 4 in mapped host memory, gathers the round's PLE rows on the host (E1's FC-PLE1; PL1: after a WILLNEED hint
// on their pages) and copies them to both cards' upload slots in stream order, launches card 0's graph then card 1's,
// then sets the start flag; each graph copies its descriptor, runs the pass, commits version k when its error words are
// clear, and writes its completion record into mapped host memory, which the host polls. The bootstrap round (no layer
// work) is enqueued eagerly. "eager" mode enqueues both cards' passes in phases instead of graphs (Compute Sanitizer).
//
// The ledger (FC-SLOT1; fn/runtime/versions.h), host only, one slot: L (ledger.size()), N, U, pending, retained,
// prompt_len, O and A, the summaries' valid blocks and the committed slot. After every commit the driver checks
// L = N + U, n_valid = N / 4, U = 1 after a bootstrap or generate round and U = 0 after the last forced round of a
// prompt; a failed check fails the round. SR1: commit selects a version (the round state's cur), so the pre-step
// contents survive until the next round starts, which acknowledges the step (A <- O). cut truncates within the last step
// (versions.h's primitive). A recoverable error on either card (FC-RD2) puts both cards back at the pre-step slot and
// clears the error words before the round reports its failure; the ledger is unchanged. SM1: with a sampler, the
// decision runs the sampler's stages (o.sampling): the round's decision record is discarded on a failure, the penalty
// window starts from the prompt, and a decision-hash mismatch (err_hash) invalidates the slot until reset. SV1: with
// both_paths, init also captures the other decision variant's graphs (the greedy path's or the top-k path's decision
// stages and draft pick); set_sampling selects a variant between requests and writes the options, never capturing.

#include "fn/kernels/round/round.h"
#include "fn/model/draft.h"
#include "fn/model/kv_tier.h"
#include "fn/model/model.h"
#include "fn/model/pass.h"
#include "fn/runtime/ple_host.h"
#include "fn/runtime/versions.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fn::runtime {

// DR1 Stage C: the MTP draft block (fn/model/draft.h). Settings, printed at startup by the commands that use them.
struct draft_options {
    bool on = false;    // pack the draft, fill its head copy, resolve both cards' draft blocks and capture its graphs
    std::string subset; // the draft subset's ID file (fn/model/draft.h read_draft_subset)
    // "draft attention": dense (DA-TC, a full read of the draft's cells) is the serving path; fresh and reuse are
    // refused (QA-2 deferred, not built) and a missing value is an error. Switch of ticket DR1; it expires when QA-2
    // is decided (built, or dropped from the plan). SP2: sparse (the draft's own index selects up to 512 whole blocks
    // at step 0 and the chain's steps reuse them, fn/model/draft.h) captures both forms' draft graphs and starts in
    // the sparse one (target::set_draft_sparse switches); a temporary switch of ticket SP2, it expires at SP2's
    // disposition.
    std::string attention;
};

struct target_options {
    std::string model_path; // the first target shard (read_ple_params)
    bool eager = false;     // both cards' passes enqueued in phases each round (no graphs)
    // ME3: init captures no graphs; the caller captures them (target::capture) once the expert service's buffers
    // exist (the cache's init, before its pool), so the graphs come out of the device's account before the pool
    bool defer_capture = false;
    model::pass_options pass;
    // SM1: the request's sampler (FC-DEC2, FC-SAMP0); unset: V1's greedy as TG1 built it. Its options are refused
    // (init fails with the message) when samp::options::check does
    bool sample = false;
    fn::samp::options sampling;
    double timeout_s = 60.0; // a round's completion wait
    std::string graph_nodes; // a directory: each captured graph's node list is written there (SR1); empty: none
    draft_options draft;
    // SV1: also capture the other decision variant (greedy at T <= 0, the top-k path at T > 0), so set_sampling
    // switches a request between T = 0 and T > 0 with no capture; needs the sampler
    bool both_paths = false;
    // MU1: the slot this target is the frame of (its slot classes resolved under model::slot_scope); the engine's
    // compute streams, one a card (null: the target creates its own), so every slot's rounds and the prompt path run
    // in one order on each card; and the serialized counters of the shared exchange rings, which every slot's rounds
    // advance in the order they run (X-PLE-STATS's use, xchg.h; the eager executor's message uses [2]); null: the
    // target's own
    int32_t slot = 0;
    void * streams[2] = {nullptr, nullptr};
    uint64_t * ple_use = nullptr;
    model::use_counters * eager_uses = nullptr;
};

// false with the reason unless the draft settings are buildable (draft_options::attention)
bool check_draft_options(const draft_options & o, std::string & err);

// what the host sees of a round once both cards' completion records are in
struct round_result {
    uint64_t round = 0;
    int32_t kind = 0, rows = 0, n_before = 0;
    int32_t n_out = 0;
    int32_t out[fn::round::max_rows] = {-1, -1, -1, -1, -1}; // SP2: five rows
    int32_t argmax[2][fn::round::max_rows] = {{-1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}}; // each card's (they agree)
    uint32_t err[2][fn::round::max_err] = {};
    uint64_t t_begin[2] = {}, t_start[2] = {}, t_end[2] = {};
    uint64_t t_logits[2] = {}; // SM1: after the logits (0 without a sampler)
    // host times of the round, seconds (steady_clock): the PLE gather (with its indices and prefetch), the upload
    // enqueue, the launches and the flag
    double host_gather_s = 0, host_upload_s = 0, host_launch_s = 0, host_wait_s = 0;
    double t_done_s = 0;         // steady_clock seconds when the host saw both completion records
    float upload_ms[2] = {0, 0}; // the e copy's GPU time per card (events), when timed
    float span_ms[2] = {0, 0};   // each card's event span from before its graph launch to after it, when timed
    uint32_t cur[2] = {0, 0};    // each card's committed slot in its completion record (SR1)
    uint32_t gdn[2] = {0, 0};    // ME11: each card's GDN selector in its completion record
    bool recovered = false;      // FC-RD2: an error word was set; both cards were put back at the pre-step slot
};

// SP2 (from HC8's c57f527, for fnfl-drive spec ab; temporary at SP2's disposition): the slot's decode-ready state at
// a prefill's handoff (FC-PF2), held in host memory for the in-process A/B: the ledger (tokens, N, the pending and
// retained state, the summaries' valid blocks), the draft's valid length, the draft index's (index_n) and the last
// target batch's start, the sampling options, and per card the versioned objects' committed version (GDN states and
// convolution histories, the PLE history: PF1's checkpoint set), the draft round's buffers (its retained carry), the
// sampler's class (seed, request and record), the retained logits and the hidden carry. KV, index keys and summaries,
// the draft's K and V and the draft index are not copied: their valid lengths (N, the summaries' blocks, draft N,
// index_n) leave a later request's cells outside them.
struct ready_state {
    bool valid = false;
    ledger l;
    int64_t draft_n = 0, batch_start = 0, index_n = 0;
    fn::samp::options sampling;
    std::vector<uint8_t> dev[2]; // per card: the regions in target::ready_regions order
};

class target {
  public:
    target() = default;
    ~target();
    // MU1: the slot this target is the frame of
    int32_t slot_index() const { return o_.slot; }
    target(const target &) = delete;
    target & operator=(const target &) = delete;

    // packs every dense weight, resolves both cards' passes, allocates the mapped descriptor ring, flag and completion
    // records, and (unless eager) captures one graph per card per B
    bool init(model::model & m, const target_options & o, std::string & err);

    // every state object zeroed (KV, index keys and summaries, GDN state and histories, PLE history), the exchange
    // rings refilled, the workspaces and error words cleared, the ledger emptied; synchronous
    bool reset(std::string & err);

    // SV1: a request's sampling options, between requests (no round in flight): refused unless check passes and their
    // decision variant was captured; written to both cards on their streams (samp::write_params), synchronous
    bool set_sampling(const fn::samp::options & o, std::string & err);
    const fn::samp::options & sampling() const { return o_.sampling; }
    // SV1: the second variant's capture (both_paths): its graphs, seconds, and the device memory it took on each card
    int32_t variant_captures() const { return var_captures_; }
    double variant_capture_s() const { return var_capture_s_; }
    int64_t variant_bytes(int32_t card) const { return var_bytes_[card]; }

    // the slot's ledger: a prompt (or model-test's reference tokens), all unprocessed
    void set_prompt(const std::vector<int32_t> & tokens);
    // SV1 (RS1 section 2's pure extension): the N processed tokens become the prompt to extend; the unprocessed tail (a
    // pending token) is dropped and the outputs count as prompt. The device state is unchanged.
    bool keep_processed(std::string & err);
    const ledger & slot() const { return l_; }

    // a forced round of the next B ledger tokens (B <= U); a generate round (the pending token, B = 1). upload_e:
    // gather and upload the PLE rows (false only in model-bench's replays, whose rows stay on the cards). timed: CUDA
    // events around each card's upload and graph launch. Synchronous: returns with the round's completion records.
    bool forced(int32_t rows, round_result & r, std::string & err, bool upload_e = true, bool timed = false);
    // PF1: a forced round whose PLE rows are already on the cards (the prefill's device staging; e[c] on card c holds
    // the round's rows), copied into the upload slot on the round's stream in place of the host gather and upload
    bool forced_staged(int32_t rows, const float * const e[2], round_result & r, std::string & err);
    bool generate(round_result & r, std::string & err, bool timed = false);
    // TEST ONLY (SR1, rollback-test): a forced round of B rows that commits k (0..B) of them (round kind forced_k);
    // test_fail: bit r makes card r's commit see an injected error (FC-RD2's recoverable error)
    bool forced_k(int32_t rows, int32_t k, int32_t test_fail, round_result & r, std::string & err);
    // SP1: a verify round (FC-RD3, FC-DEC3): B = d + 1 rows (d 1..4, SP2), the pending token then the proposals
    // prop[0 .. d - 1]; the decision's a (SM1's accept step at T > 0, greedy's at T = 0) and out[0 .. a], with
    // FC-DEC3's EOS and budget cut; commit a + 1 (FC-ST2). budget: the tokens the request still allows (B <= budget,
    // else refused before launch). Needs the sampler (target_options::sample; T = 0 runs its greedy path). The ledger
    // takes out[0 .. a];
    // out[a] is the new pending token
    bool verify(int32_t d, const int32_t * prop, int32_t budget, round_result & r, std::string & err,
                bool timed = false);
    // FMU2: a verify round as a lane of a joint round (fn/runtime/joint.h). joint_verify_begin: verify's checks, the
    // proposals into the ledger and the round staged (descriptor, PLE rows uploaded on the streams, the versions' step
    // begun) with the lane ring's PLE use; the caller then launches the joint graphs and calls joint_release (the
    // start flag), or joint_abandon when the launch failed; joint_verify_end: the completion wait and verify's checks,
    // FC-RD2's recovery and the ledger, as verify. Graph executor only.
    struct staged_round {
        uint64_t rd = 0;
        int64_t n_before = 0;
        std::chrono::steady_clock::time_point t0, t1, t2, t3;
    };
    bool joint_verify_begin(int32_t d, const int32_t * prop, int32_t budget, uint64_t ple_use, staged_round & sr,
                            std::string & err);
    void joint_release(const staged_round & sr);
    void joint_abandon();
    bool joint_verify_end(int32_t d, const int32_t * prop, staged_round & sr, round_result & r, std::string & err);
    // FMU2 (build item 2): a draft step as a lane of a joint draft step (joint_frame::draft). joint_draft_begin:
    // draft_step's checks and its descriptor staged (eagerly the start flag too); the caller launches the joint draft
    // and calls joint_release; nothing to undo when the launch fails. joint_draft_end: the completion wait and
    // draft_step's checks (r.argmax[0][0 .. d - 1] the proposals). draft_pick_sampled: the draft's pick is SM1's.
    bool joint_draft_begin(int32_t d, staged_round & sr, std::string & err);
    bool joint_draft_end(int32_t d, const staged_round & sr, round_result & r, std::string & err);
    bool draft_pick_sampled() const { return dc_[0].sampled; }
    // the eager executor's message use counters (the shared rings' when the engine shares them)
    model::use_counters * eager_counters() { return u_; }
    // FMU2: the joint graph's inputs from this slot: its pass options (the sampler's path) and decision variant
    const model::pass_options & pass_options() const { return o_.pass; }
    int32_t variant() const { return var_; }
    // FMU2 (the memory account): the nodes of card's captured verify graph at rows in the slot's variant; -1 without it
    int64_t graph_node_count(int32_t rows, int32_t card) const;
    // TEST (R9-12's repair, the dry spec run's verify-recovery check): the next verify round's commit sees an
    // injected error on the cards of mask (bit r: card r; FC-RD2's recoverable error, as forced_k's test_fail)
    void test_fail_next_verify(int32_t mask) { test_fail_verify_ = mask; }
    // the primitive (versions.h): truncate within the last step to j kept rows and the ledger to n_keep tokens
    bool cut(int32_t j, int64_t n_keep, std::string & err);
    // the slot's tokens grow by these, as prompt tokens (unprocessed; rollback-test's continuation after a cut)
    void append(const std::vector<int32_t> & tokens);
    const slot_versions & versions() const { return v_; }
    // SM1 (FC-RD2's delayed guard): the closing check of the last decision's hash on both cards, before a step that
    // ends generation returns (and before a truncate, re-prefill or release); checked: the last record was compared
    bool close_check(bool & checked, std::string & err);
    // a decision-hash mismatch (err_hash) invalidated the slot: no round runs until reset (FC-RD2)
    bool invalid() const { return invalid_; }
    // TEST ONLY (ME11, prefill-test): the GDN selector on both cards, no step recorded (no round in flight); refused
    // outside the contract
    bool test_set_gdn(uint32_t gdn, std::string & err) { return v_.set_gdn(l_, gdn, err); }
    // TEST ONLY (SM1): flip card r's decision record (its hash), so the next comparison fails
    bool test_flip_record(int32_t r, std::string & err);
    // the bootstrap round: token zero from the retained logits (FC-PF2). test_fail (TEST ONLY, rollback-test retain):
    // bit r makes card r's commit see an injected error; the round then fails with the ledger and retained row kept
    bool bootstrap(round_result & r, std::string & err, int32_t test_fail = 0);

    // PF1 (FC-ST3), a checkpoint restore in two halves around the caller's copy of the checkpoint into version 0.
    // rewind: as reset, but the state objects (KV, index keys and summaries, GDN states and histories, draft KV) keep
    // their contents: the workspaces, error words, rings, sampler and draft round buffers cleared, cur 0 on both
    // cards, the ledger emptied; synchronous. restored: the ledger holds tokens with the first
    // n processed (the checkpoint's position; KV, index keys and summaries valid below n), and the draft's history is
    // n with the retained carry the checkpoint's (draft on)
    bool rewind(std::string & err);
    // PB1, the bulk prompt path (fn/model/prompt.h), one chunk at a time:
    // prompt_begin: the chunk is a step (the previous one acknowledged, SR1's record of this one);
    // prompt_commit: after both cards joined clean, the chunk's publication: cur <- version_slot(cur, k) on both cards
    //   (k: its halves; each card's word written in stream order, read back and checked), N += rows, the summaries
    //   settled, the draft at N with its batch start N; nothing is retained until the handoff; the step keeps no rows for a cut (slot_versions::committed_prompt);
    // prompt_head: the handoff (FC-PF2): the decode head at B = 1 on the layer-47 boundary row 0 the prompt path wrote
    //   (prompt_path::head_input), on X-HC / X-HMIX slot 2 with every ring slot at its sentinel before and after (no
    //   round is in flight); logits row 0 retained. Synchronous.
    bool prompt_begin(std::string & err);
    bool prompt_commit(int32_t rows, int32_t k, std::string & err);
    bool prompt_head(std::string & err);
    // ME11: gdn_steps, the checkpoint's live GDN records, which the load put into bank 0 and set 0: the selector
    // becomes (0, 0, gdn_steps) on both cards
    bool restored(int64_t n, const std::vector<int32_t> & tokens, int32_t gdn_steps, std::string & err);
    bool draft_on() const { return o_.draft.on; }

    // model-bench: set N (and L, U) to depth with the ledger's tokens, without running rounds (a synthetic history)
    bool set_depth(int64_t depth, std::string & err);

    // DR1 Stage C, the draft rounds (fn/model/draft.h), when target_options::draft.on. The draft's ledger: draft_n()
    // cells hold the draft's K and V of committed positions 0 .. draft_n() - 1 (the catch-up's); the last target
    // round's batch starts at batch_start() and its hidden carry rows stay on the cards until the next target round.
    // draft_catchup: the next rows (1..5) committed positions from draft_n(), all within the last target batch or
    // continuing the retained carry (draft_n() >= batch_start(), draft_n() + rows <= N); the retained carry becomes
    // the last row's. draft_step: d (1..4) chained steps from position draft_n() (its token the ledger's, the pending
    // token at N), writing d speculative cells without advancing draft_n(); the proposals are r.argmax[0][0 .. d - 1].
    // Neither touches the target's state or versions; a draft round's error word is cleared and the round fails.
    bool draft_catchup(int32_t rows, round_result & r, std::string & err);
    bool draft_step(int32_t d, round_result & r, std::string & err);
    int64_t draft_n() const { return draft_n_; }
    int64_t batch_start() const { return batch_start_; }
    // SP2 (draft attention sparse; temporary at SP2's disposition): the draft rounds' form, between requests (no
    // round in flight): on, the sparse graphs (refused unless captured); off, DR1's dense ones. The draft index holds
    // the keys and summaries of cells 0 .. index_n() - 1, consistent with the draft's K and V: a sparse catch-up or
    // prompt chunk from a cell at or below index_n() extends it to its end, a dense one cuts it back to its first
    // cell, as a truncate of the draft does. A sparse draft step needs index_n() >= draft_n() (else refused).
    bool set_draft_sparse(bool on, std::string & err);
    // SP2 (HC8's): the decode-ready state after a prefill's handoff (no step since), copied to host memory; and the
    // slot put back at it (as rewind: the rings, workspaces, sampler and draft round cleared, cur 0; then the
    // committed versions written to version 0, the other regions and the ledger restored). The sampling options are
    // the saved ones. Synchronous.
    bool save_ready(ready_state & s, std::string & err);
    bool restore_ready(const ready_state & s, std::string & err);
    bool draft_sparse() const { return dmode_ == 1; }
    // SP4 (the expert gate/up mapping): the target FFN's fn::experts::gate_up_args::map (0: SP2's, 1: SP4's H1, 2:
    // SP4's H2, 3: XB3 C1's), between requests (no round in flight). A change recaptures
    // every captured decision variant's B = 4 and 5 graphs, each under the sampling options it was captured with
    // (B 1..3 and the draft's Q8_0 experts ignore the map).
    bool set_expert_map(int32_t map, std::string & err);
    int32_t expert_map() const { return p_[0].c.ex_map; }
    int64_t index_n() const { return index_n_; }
    // model-bench's draft timing: draft_n() and batch_start() at depth (a synthetic draft history; set_depth first;
    // SP2: in the sparse form the draft index's cells too, holding what the memory holds, as the K and V do)
    bool draft_set_depth(int64_t depth, std::string & err);
    model::draft_card & draft(int32_t r) { return dc_[r]; }
    // ME1: the host-backed KV tier (null without --kv-host); kv_flush makes its host history whole (state checks read
    // the KV classes there), synchronous
    model::kv_tier * tier() const { return tier_.get(); }
    bool kv_flush(std::string & err) { return tier_ == nullptr || tier_->flush(st_[0], err); }

    // card r's pass (buffers for checks and synthetic fills)
    model::pass_card & card(int32_t r) { return p_[r]; }
    void * stream(int32_t r) const { return st_[r]; }
    int32_t captures() const { return captures_; }

    // ME3: the graphs init deferred (target_options::defer_capture): every variant's target graphs, the draft's step
    // and catch-up graphs, with the cards' current expert service. On one device (model::single) a graph holds both
    // logical cards' passes in the pair order on the one stream (pass_pair_enqueue, draft_pair_enqueue: the
    // producer-before-consumer schedule; a card's peer waits are always on work enqueued before them); on two, one
    // graph a card. graph_bytes: the device memory the capture took (per card; one device: card 0's)
    bool capture(std::string & err);
    int64_t graph_bytes(int32_t card) const { return graph_bytes_[card]; }
    // ME3: the executor of the next rounds, between requests: graph (the captured graphs, the serving default) or
    // eager (the phased enqueue: Compute Sanitizer's, and ME3's base arm); the expert service and the KV tier follow
    // it, and the slot is reset (the exchange rings' slot rules differ between the executors)
    bool set_executor(bool graph, std::string & err);
    bool graph_executor() const { return !eager_; }
    // ME3: the submissions since init: graph launches (target, draft step, catch-up) and eager rounds
    struct submissions {
        int64_t target_graphs = 0, draft_graphs = 0, catchup_graphs = 0, eager_rounds = 0, eager_draft_rounds = 0;
        int64_t joint_rounds = 0; // FMU2: verify rounds run as a joint round's lane
        int64_t joint_drafts = 0; // FMU2: draft steps run as a joint draft step's lane
    };
    const submissions & submitted() const { return sub_; }
    const ple_params & ple() const { return ple_; }

  private:
    bool round(int32_t kind, int32_t rows, bool upload_e, bool timed, round_result & r, std::string & err,
               int32_t test_k = 0, int32_t test_fail = 0, int32_t budget = 0, const float * const * staged = nullptr);
    bool round_stage(int32_t kind, int32_t rows, bool upload_e, bool timed, int32_t test_k, int32_t test_fail,
                     int32_t budget, const float * const * staged, uint64_t ple_use, staged_round & sr,
                     std::string & err);
    bool round_collect(int32_t kind, int32_t rows, bool upload_e, bool timed, int32_t test_k, const staged_round & sr,
                       round_result & r, std::string & err);
    bool verify_accept(const int32_t * prop, const round_result & r, std::string & err);
    bool forced_round(int32_t rows, round_result & r, std::string & err, bool upload_e, bool timed,
                      const float * const * staged);
    bool clear_slot(bool zero_state, std::string & err); // reset's body; rewind keeps the state objects
    bool save_carry(std::string & err);
    bool restore_carry(std::string & err);
    bool wait(uint64_t round, double timeout_s, std::string & err);
    bool check_ledger(std::string & err);
    bool clear_errors(std::string & err);
    bool write_nodes(int32_t v, int32_t rows, int32_t card, std::string & err) const;
    bool write_graph_nodes(void * graph, const std::string & path, std::string & err) const;
    bool capture_draft(int32_t card, int32_t kind, int32_t rows, std::string & err, int32_t v = 0, int32_t mode = 0);
    bool capture_draft_kind(int32_t kind, int32_t max_rows, int32_t v, std::string & err); // every captured form
    void index_written(int64_t first, int64_t rows); // SP2: index_n_ after a draft K/V write in the current form
    // SP2 (HC8's): card c's regions of the decode-ready state (ready_state): a device range, and its kind: plain, a
    // versioned object (fn/kernels/round/version.h: n_version slices, of which the committed one is copied), or (ME11)
    // a GDN layer's two banks or two record sets, of which the selector's bank or set is copied
    enum class region_kind : int32_t { plain, versioned, gdn_banks, gdn_records };
    struct ready_region {
        uint8_t * p;
        int64_t bytes;
        region_kind kind;
    };
    bool ready_regions(int32_t c, std::vector<ready_region> & out, std::string & err) const;
    // the bytes of a region a ready state copies: one version, one bank, one record set (ME11) or the whole region
    static int64_t ready_bytes(const ready_region & r);
    bool capture_variant(int32_t v, std::string & err);
    bool capture_target(int32_t v, int32_t b, int32_t c, std::string & err); // one of capture_variant's graphs
    void set_pick(int32_t v); // the draft's pick for variant v (draft_card::sampled)
    bool draft_round(int32_t kind, int32_t rows, int64_t first, int32_t hsrc, round_result & r, std::string & err);
    // draft_round's halves: the checks and the descriptor (eagerly the flag), and the wait and the checks
    bool draft_stage(int32_t kind, int32_t rows, int64_t first, int32_t hsrc, staged_round & sr, std::string & err);
    bool draft_collect(int32_t kind, int32_t rows, const staged_round & sr, round_result & r, std::string & err);
    bool draft_step_check(int32_t d, std::string & err) const;

    model::model * m_ = nullptr;
    target_options o_;
    void * st_[2] = {nullptr, nullptr};
    model::pass_card p_[2];
    fn::round::desc * ring_ = nullptr; // mapped host memory [ring]
    uint64_t * flag_ = nullptr;        // mapped host memory
    fn::round::done * done_ = nullptr; // mapped host memory [2]
    float * e_host_ = nullptr;         // pinned [ring][5][2560]
    // [variant][B][card] graph executables; variant 0: no sampler or its greedy path, 1: the other paths (SV1); B 1..5
    void * exec_[2][fn::round::max_rows + 1][2] = {};
    void * graph_[2][fn::round::max_rows + 1][2] = {};
    int32_t var_ = 0;               // the variant the current options run
    int32_t var_key_[2] = {-1, -1}; // the captured variant's decision path (samp::path), -1: not captured
    fn::samp::options var_opts_[2]; // SP4: the sampling options each variant was captured with (set_expert_map)
    int32_t var_captures_ = 0;      // SV1: the second variant's graphs
    double var_capture_s_ = 0;
    int64_t var_bytes_[2] = {0, 0};
    void * ev_[2][4] = {}; // per card: upload start/end, launch start/end
    int32_t captures_ = 0;
    bool eager_ = false;    // ME3: the executor the rounds run (set_executor)
    bool captured_ = false; // ME3: the graphs exist (capture)
    int64_t graph_bytes_[2] = {0, 0};
    submissions sub_;
    uint64_t next_round_ = 1;
    uint64_t own_ple_use_ = 0;
    uint64_t * ple_use_ = &own_ple_use_; // MU1: the engine's when shared (target_options::ple_use)
    bool own_streams_ = true;            // MU1: false when the engine's streams are used
    ple_params ple_;
    ledger l_;
    slot_versions v_;
    model::use_counters own_u_[2];     // eager mode's exchange counters
    model::use_counters * u_ = own_u_; // MU1: the engine's when shared (target_options::eager_uses)
    model::draft_card dc_[2];
    // [form: 0 dense, 1 sparse (SP2)][variant][0: step, 1: catch-up][d or B_c][card]; the catch-up has no pick, so
    // only variant 0 holds it
    void * dexec_[2][2][2][fn::round::max_rows + 1][2] = {};
    void * dgraph_[2][2][2][fn::round::max_rows + 1][2] = {};
    int32_t dmode_ = 0;          // SP2: the draft rounds' form (1: sparse)
    bool sparse_built_ = false;  // SP2: the sparse form's graphs were captured (draft attention sparse)
    int64_t index_n_ = 0;        // SP2: the draft index's valid cells (index_n())
    int64_t draft_n_ = 0;     // -1: the draft no longer matches the ledger (a cut below it; reset clears it)
    // SP1: the retained carry's pre-step copy (20 KB a card, outside R1's reservation: debt), taken before the first
    // catch-up after a target round, with draft_n() then; restored with draft_n() on a truncate below the draft and
    // after a failed catch-up (FC-RD2's restore covers the draft's caches by their valid length)
    float * carry_save_[2] = {nullptr, nullptr};
    int64_t carry_save_n_ = -1; // draft_n() when the copy was taken; -1: no copy for the last target round
    // R9-12: a verify round's proposals on the token vector, for the round's inputs only (verify); FC-RD2's recovery
    // takes them off before it checks the ledger
    int32_t verify_extra_ = 0;
    int32_t test_fail_verify_ = 0; // TEST: test_fail_next_verify
    int64_t batch_start_ = 0; // N before the last target round
    bool invalid_ = false;    // SM1: a decision-hash mismatch invalidated the slot
    bool released_ = false;   // ME3 (load G ruling): a completion timeout released the cache's GPU waits
    bool after_error(round_result & r, std::string & err); // FC-RD2's restore, the record discarded
    std::unique_ptr<model::kv_tier> tier_; // ME1 (--kv-host)
};

} // namespace fn::runtime
