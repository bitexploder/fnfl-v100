#pragma once

// The FNFL runtime: request flow per slot (RS1, cut as SV1 states). SV1: the speculative loop lives here, one prefill
// and one step per call, and fnfl-drive spec and the server (fn/glue) both call it; engine owns the loaded model and the
// target (fn/runtime/target.h) of the one slot.
//
// A step (step): the bootstrap after a prompt (token zero from the retained logits, FC-RD3); then each step is d
// draft steps (SP2: the engine's setting, 1..4, default 2; d = budget - 1 when the budget is smaller), the verify
// round with the budget (B = d + 1 <= budget), and the draft's catch-up of the committed rows; a budget of 1 runs a
// generate round. Three host waits a speculative step,
// one a bootstrap or generate step. An end-of-generation token of the set ends the step: the tokens after it are not
// returned and the state is cut back to it by version (target::cut) before the catch-up, on the host after the verify
// round returned; no work is added inside a round.
//
// The engine's slot (one): prefill keeps the slot when n_reuse is its processed length (a pure extension); any other
// reuse restores the latest checkpoint at or before n_reuse and re-prefills from it, or from token 0 when none fits
// (PF1, fn/runtime/prefill.h); both run PF1's chunk loop. With no new tokens (n_tokens = 0), the last kept token is
// processed again so its logits exist (FC-PF1). truncate within the last step selects a version (target::cut), then
// the draft catches up the kept rows; deeper, the next prefill restores a checkpoint. A failed step ends the request
// with its message and forgets the checkpoints; the next prefill resets the slot (target::reset) and re-prefills.

#include "fn/kernels/samp/samp.h"
#include "fn/runtime/joint.h"
#include "fn/runtime/prefill.h"
#include "fn/runtime/target.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fn::model {
class model;
}

namespace fn::runtime {

enum class run_status : int32_t {
    ok,
    not_built,
    bad_request,
    error,
    cancelled,  // BG1: a prefill stopped by its progress callback
    no_context, // MU1: the KV pool cannot hold the slot's next cells (every other slot active)
};

constexpr int32_t max_step_tokens = 5; // SP2: the sampled token and up to four accepted drafts
constexpr int32_t max_spec_d = 4;      // SP2: the most drafts a speculative step proposes (fn::model::draft_max_steps)
constexpr int32_t spec_d_default = 2;  // SP2: the serving default

// one step as the host saw it
struct step_info {
    int32_t kind = 0; // 0 bootstrap, 1 generate, 2 speculative (draft step, verify, catch-up)
    int64_t n_before = 0;
    int32_t d = 0, a = 0; // the verify round's drafts and accepted drafts (before an end-of-generation cut)
    int32_t n_out = 0;    // the tokens returned (after the cut)
    int32_t prop[max_spec_d] = {-1, -1, -1, -1};
    int32_t out[max_step_tokens] = {-1, -1, -1, -1, -1};
    bool end = false;  // out[n_out - 1] is an end-of-generation token
    bool cut = false;  // the verify round's tokens were cut back to an end-of-generation token
    int32_t waits = 0; // host waits: rounds, and the cut's synchronous write
    double wall_s = 0; // host clock around the step
    // SF1: seconds from the step's start to each round's completion wait, from the host clock the round already reads
    // there (round_result::t_done_s); -1 when the step has no such round. Speculative: the draft step, verify and the
    // catch-up; bootstrap and generate: their one round
    double to_draft_s = -1, to_verify_s = -1, to_catchup_s = -1, to_one_s = -1;
    bool joint = false; // FMU2: the verify round was a joint round's lane (step_joint)
};

// between a speculative step's draft step and its verify round (fnfl-drive spec's recorded rounds); r: the draft step's
using after_draft_fn = std::function<bool(const round_result & r)>;

// a prompt from empty state: reset, then forced rounds over it at B <= 4 (the last the remainder), each followed by
// the draft's catch-up of its rows
bool prefill(target & t, const std::vector<int32_t> & prompt, std::string & err);
// the slot continues: its processed tokens stay (target::keep_processed), then forced rounds over tokens as prefill's
bool extend(target & t, const int32_t * tokens, int32_t n, std::string & err);
// extend's first part (PF1's pure extension): the draft caught up to N, the processed tokens kept and tokens appended
bool extend_ledger(target & t, const int32_t * tokens, int32_t n, std::string & err);
// one step of at most budget (>= 1) tokens (above); d: the drafts a speculative step proposes (SP2: 1..4);
// eog: the end-of-generation IDs
bool step(target & t, int32_t budget, int32_t d, const std::vector<int32_t> & eog, step_info & s, std::string & err,
          const after_draft_fn * after_draft = nullptr);
// a speculative step's halves around its verify round (FMU2: the joint round's members take theirs in between):
// step_propose, the draft step's min(d, budget - 1) proposals into s (from s.n_before, set by the caller, and the
// step's start t0); step_close, from the verify round's result rv the step's tokens, the end-of-generation cut and the
// draft's catch-up
bool step_propose(target & t, int32_t budget, int32_t d, std::chrono::steady_clock::time_point t0, step_info & s,
                  std::string & err, const after_draft_fn * after_draft = nullptr);
bool step_close(target & t, const std::vector<int32_t> & eog, const round_result & rv,
                std::chrono::steady_clock::time_point t0, step_info & s, std::string & err);

// the engine's settings, printed at startup (STANDARDS "Switches"); the test switches name SV1 and end when SV1 merges
// with its checks recorded, unless the review keeps them as documented debug options
struct engine_config {
    std::string model_path;     // the first target shard (target_options::model_path)
    int32_t n_ctx_slot = 0;     // the slot's cells
    draft_options draft;        // on, dense, the subset file
    std::vector<int32_t> eog;   // the end-of-generation IDs (the vocabulary's set, then --fnfl-test-eog's)
    int32_t test_fail_step = 0; // TEST (SV1, --fnfl-test-fail-step K): step K of the next request fails once; 0 off
    bool log_steps = false;     // TEST (SV1, with --fnfl-check-state): one line a step; off in timing runs
    double timeout_s = 60.0;    // a round's completion wait
    // the launch sampling until a request sets its own (TK1: the server's --temp, --top-k, --top-p, --min-p)
    float temp = 1.0f;
    int32_t top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.05f;
    int32_t spec_d = spec_d_default; // SP2: the drafts a speculative step proposes, 1..4 (--fnfl-spec-d)
    // ME3 (--ex-executor, --fnfl-expert-executor): false, the graph executor (the default, cached or not, one device
    // or two); true, the eager paired enqueue (the explicit diagnostic and ME3's base arm). experts, when set, makes
    // the routed experts' service (fn/runtime/expert_cache.h) after every other allocation of the engine, which then
    // owns it and serves both cards' FFNs and the prompt's experts through it; the factory passes the hook to the
    // service, which captures the graph executor's graphs before its pool (model::service_hook; null for eager)
    bool eager = false;
    std::function<std::unique_ptr<model::expert_service>(model::model &, const model::service_hook &, std::string &)>
        experts;
    // ME3 (fn/cache_settings.h, with experts only): the prefiller's known-token short reads (rows; 0 none) and its
    // largest bulk chunk; the loan's startup check runs once the prefiller exists
    int32_t short_read = 0, prompt_chunk = 8192;
    // FMU2: the joint verify round for 2..4 generating slots on resident TP (step_joint; eager: phased, no graphs);
    // joint_cache, the membership graph pairs kept (1 .. 8)
    bool joint = true;
    int32_t joint_cache = 2;
};

// FMU2: one slot's step in step_joint
struct joint_step_out {
    int32_t out[max_step_tokens] = {};
    int32_t n_out = 0;
    bool end = false;
    bool joint = false; // its verify round was a joint round's lane
    run_status st = run_status::ok;
    std::string why; // the failure's reason (the engine's error() for this slot)
};

// a request's options (fnfl_set_sampling)
struct request_options {
    fn::samp::options sampling;
    int32_t max_tokens = -1; // the request's token limit; -1: the context room
    int32_t n_ctx = 0;       // the shell's slot context (the room's limit); 0: the engine's
};

// one request's steps (the per-request summary)
struct request_summary {
    int32_t steps = 0, boot = 0, gen = 0, spec = 0;
    int32_t tokens = 0, accepted = 0, spec_tokens = 0, waits_spec = 0, waits_one = 0, cuts = 0;
    double wall_median_ms = 0, wall_p90_ms = 0;
    // SF1: medians of step_info's phase times (ms; 0 when no step has the phase), and the depth range before the steps
    double draft_median_ms = 0, verify_median_ms = 0, catchup_median_ms = 0, one_median_ms = 0;
    int64_t n_before_min = 0, n_before_max = 0;
    double span_s = 0; // from the bootstrap's start to the last step's end
    std::string ended; // eog, budget, or the shell's (a stop string, its limit, a cancel or the next request)
};

// MU1 (the multi-slot engine): a read job's step outcome (fn/runtime/prefill.h pf_step) and what the scheduler reads
enum class read_status : int32_t { yielded, chunk_done, done, cancelled, error };
// FMU1: who else has model work when a read step runs (the server's presence rule): no one (lone), generating slots
// (writers), or only other readers (shared). Resident TP only; elsewhere every step is shared (MU1, MXMU1)
enum class read_mode : int32_t { shared, writers, lone };
struct read_info {
    int32_t slot = -1;
    int64_t job = 0;                           // FMU1: the read's job number (its staging tags)
    bool live = false;                         // a read is prepared and not ended
    bool cancel = false;                       // its cancellation is pending
    int64_t total = 0;                         // the read's tokens (the slot's whole prompt)
    int64_t from = 0;                          // where the read started (the kept or restored prefix)
    int64_t committed = 0;                     // the prompt tokens the slot's state holds (published chunks)
    int32_t chunk = 0, chunks = 0;             // the chunk in progress or next, of the read's chunks
    int32_t next_rows = 0;                     // that chunk's rows
    bool in_chunk = false;                     // a chunk is in progress (suspended between groups)
    int32_t lane = -1;                         // its lane
    int32_t group = 0, groups = 0, layers = 0; // its next group, its groups, layers a group
    int64_t groups_done = 0;                   // groups run (each step's are complete when it returns)
    int32_t steps = 0, chunks_done = 0, checkpoints = 0;
    double service_s = 0, longest_step_s = 0, last_group_s = -1; // host clock around the job's steps
};
// MU1: the shared KV pool as the engine sees it
struct pool_info {
    bool on = false;
    int32_t pages = 0, page_cells = 0, free_pages = 0;
    int32_t mapped[4] = {0, 0, 0, 0}; // each slot's mapped pages
    bool busy[4] = {false, false, false, false};
    int64_t capacity_cells = 0;
    int32_t reclaims = 0; // idle slots reclaimed since load
};

class engine {
  public:
    engine();
    ~engine();
    engine(const engine &) = delete;
    engine & operator=(const engine &) = delete;

    // takes the loaded model (fn::loader::load_for_run, or a test's zero weights): the target with the draft, the
    // sampler at the launch defaults and both decision variants (target_options::both_paths); false with the reason.
    // MU1: the model's slots (model::n_slots): above one, a target a slot (its frame, its graphs) on the engine's
    // compute streams and shared exchange counters, and the prefiller's other-reader lane
    bool init(std::unique_ptr<model::model> m, const engine_config & c, std::string & err);
    // the settings line printed at startup
    std::string settings() const;
    const target & tgt() const { return *s_[0].t; }
    // PF1: the last prefill's chunks (prefill_test and the server's log)
    const pf_run_stats & last_prefill() const { return s_[0].pf_stats; }

    // PB1: the last prefill call's phase line (as written to stderr)
    const std::string & last_phase_line() const { return s_[0].phase_line; }

    // TEST ONLY (prefill-test): the slot's prefiller
    prefiller & prefill_driver() { return *pf_; }

    int32_t n_slots() const { return static_cast<int32_t>(s_.size()); }

    // the request's options, before its prefill (between requests); MU1: the slot's request only
    run_status set_sampling(int32_t slot, const request_options & o);
    // n_processed: the tokens processed: n_tokens for a pure extension, n_reuse + n_tokens - c after restoring the
    // checkpoint at c (c = 0: from token 0); n_tokens = 0 processes the last kept token again. progress (BG1) runs
    // after every chunk but the last; when it stops the prefill, the slot keeps the prompt's tokens up to that chunk's
    // end without a valid state (the next prefill restores a checkpoint at or below them), n_processed is their count
    // and the status is cancelled. MU1: on a multi-slot engine the same through the slot's read job, every group in
    // one step (no other slot runs during the call)
    run_status prefill(int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse, int32_t & n_processed,
                       const prefill_progress & progress = {});
    // out holds max_step_tokens tokens. MU1: the slot's next cells mapped first (the KV pool; an idle slot reclaimed
    // when the pool is full; no_context when it stays full)
    run_status step(int32_t slot, int32_t * out, int32_t & n_out, bool & end);
    // FMU2: one step of each of n distinct slots (1..4), as step each: the slots whose step is a speculative one at
    // d = 4 take their verify rounds as one joint round (fn/runtime/joint.h) when two or more do, between their own
    // draft steps and catch-ups; every other slot's step runs alone. res[k] is slots[k]'s (step's outputs and status)
    void step_joint(const int32_t * slots, int32_t n, joint_step_out * res);
    run_status truncate(int32_t slot, int32_t n_keep);
    // the tokens the slot's state holds (processed) and its pending token (-1: none)
    run_status history(int32_t slot, std::vector<int32_t> & tokens, int32_t & pending) const;

    // ---- MU1: the multi-slot engine (n_slots() > 1) ----
    // a slot's read prepared as a job: the input validated and owned, the reuse decided (a pure extension, or the
    // slot's latest matching checkpoint restored, or a reset), the chunk plan made and the pool's pages for the whole
    // prompt and the next step mapped (an idle slot reclaimed when the pool is full). Returns 0 (prepared), 1 (the
    // context or the pool cannot hold it: the slot's state unchanged) or -1 (invalid: err). n_processed as prefill's.
    int32_t read_begin(int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse, int32_t & n_processed);
    // the job's next step: up to n_groups layer groups of its chunk in progress (or of its next chunk, started now in
    // the main arena when free, else in the other-reader lane when it fits), complete on both cards when it returns
    // FMU1: mode fixes a chunk's geometry when the step starts it (fn/runtime/prefill.h pf_mode)
    read_status read_step(int32_t slot, int32_t n_groups, read_mode mode = read_mode::shared);
    // FMU1 (P, resident TP): now_slot's read has the main lane's turn in progress or starting next, next_slot's read
    // follows it (-1: none): next's next chunk is staged ahead on the staging thread (prefiller::prefetch); never waits
    bool read_prefetch(int32_t now_slot, int32_t next_slot);
    // FMU1: resident TP (the KV pool, two cards, no expert service): the presence-based paths are on
    bool resident_tp() const;
    // BG1: the job stops at its chunk's end (or before its next chunk); never synchronous work here
    void read_cancel(int32_t slot);
    bool get_read_info(int32_t slot, read_info & out) const;
    // layers a group for chunks that start after the call (1 .. 48)
    bool set_read_group(int32_t layers);
    int32_t read_group() const { return read_group_; }
    // the largest chunk a lane runs (0: the main arena, 1: the other-reader lane; 0 rows: none) and the slot with a
    // chunk in progress there (-1: none)
    int32_t lane_rows(int32_t lane) const;
    int32_t lane_owner(int32_t lane) const;
    // MXMU1 (TEST ONLY, the drivers' two-slot one-card checks): with the expert service a read step runs its whole
    // chunk (the loan is returned before another slot's work); on, read_step runs n_groups groups as on resident TP,
    // so a check can suspend a cached chunk. Off in serving
    void test_partial_reads(bool on) { test_partial_ = on; }
    // MXMU1 (TEST ONLY): the slot's read ends as a failed step ends it (its chunk abandoned: the lane, staging and
    // prompt loan released; the slot's state unknown, re-read from token 0)
    void test_abandon_read(int32_t slot);
    // the shell released the slot (its request ended or was cancelled): its open request finished, the slot idle (its
    // pages reclaimable, least recently released first)
    run_status release_slot(int32_t slot);
    // the tokens the engine holds for the slot (0 after a reclaim)
    int64_t held(int32_t slot) const;
    bool get_pool_info(pool_info & out) const;

    // SP2 (temporary, at SP2's disposition), for fnfl-drive spec ab: the draft's d (1..4) and form (target::
    // set_draft_sparse) between requests; the slot's decode-ready state after a prefill (no step since) saved to host
    // memory and restored as the next request's start (target::save_ready, restore_ready; HC8's). TEST ONLY:
    // test_target and test_model give spec ab's checks the target and the model.
    run_status set_draft(int32_t d, bool sparse);
    run_status save_ready(ready_state & s);
    run_status restore_ready(const ready_state & s);
    target & test_target(int32_t slot = 0) { return *s_[size_t(slot)].t; }
    model::model & test_model() { return *m_; }
    // FMU2: the joint frame (nullptr when the engine has none) and whether a failed capture turned it off
    const joint_frame * test_joint() const { return joint_.get(); }
    bool test_joint_off() const { return joint_off_; }

    // the reason of the last call that failed
    const std::string & error() const { return err_; }
    // the last finished request's summary, and its line as printed
    const request_summary & last_summary() const { return s_[0].sum; }
    // SF1: the last finished request's steps (fnfl-drive spec run's rounds.csv; the server writes no per-round file)
    const std::vector<step_info> & last_steps() const { return s_[0].last_steps; }
    const std::string & last_summary_line() const { return s_[0].sum_line; }
    // ME3: the last finished request's expert-cache line (empty without the cache)
    const std::string & last_cache_line() const { return s_[0].cache_line; }
    model::expert_service * experts() const { return ex_.get(); }

  private:
    // one slot's request state (MU1: the one-slot engine's members, a slot each)
    struct slot_rt {
        std::unique_ptr<target> t;
        request_options req;
        bool valid = false;        // the device state matches the ledger (false: restore a checkpoint or re-prefill)
        std::vector<int32_t> keep; // while !valid: the tokens the slot holds, restored or re-prefilled next
        bool open = false;         // a request's steps have run and it has not finished
        bool unchecked = false;    // a decision ran since the last closing check
        int32_t n_gen = 0;         // the open request's tokens
        int32_t fail_step = 0;     // armed --fnfl-test-fail-step
        std::vector<step_info> steps, last_steps;
        std::chrono::steady_clock::time_point t_first; // the open request's first step's start
        double t_span_s = 0;                           // from it to the last step's end
        request_summary sum;
        std::string sum_line, cache_line;
        pf_run_stats pf_stats;
        std::string phase_line;
        // ME4: the target's submissions at the request's prefill entry and at decode-ready
        target::submissions sub0, sub_ready;
        // MU1: the slot's read job; busy from its read's begin to its release (not reclaimable); the release order
        pf_job job;
        bool busy = false;
        uint64_t released = 0;
        std::chrono::steady_clock::time_point job_entry;
        const char * job_how = "";
        int32_t job_reuse = 0, job_new = 0; // the read's kept and new tokens (its prefill line)
        // FMU1: the cells the pool has mapped for the slot (refreshed whenever its mapping changes), so a step's check
        // is one comparison
        int64_t cells = 0;
    };

    run_status failed(run_status s, const std::string & what);
    // step's halves around its rounds: the checks and cells (b: the budget), and the record of a step that ran or not
    run_status step_enter(int32_t slot, const int32_t * out, int32_t & b);
    run_status step_leave(int32_t slot, bool ran, const step_info & s, int32_t * out, int32_t & n_out, bool & end);
    // the slot's open request's closing check and summary (ended: why it ended)
    void finish(slot_rt & x, const char * ended);
    int32_t budget(const slot_rt & x) const;
    void lost(slot_rt & x); // a failure: the slot's state is unknown (re-prefill from token 0, no checkpoints)
    bool slot_ok(int32_t slot) const { return slot >= 0 && slot < n_slots(); }
    // MU1: the slot's first cells mapped (the pool), reclaiming idle slots when it is full; capacity: it stayed full
    bool ensure_cells(int32_t slot, int64_t cells, bool & capacity, std::string & err);
    void reclaim(int32_t slot, int32_t for_slot);
    void prefill_lines(slot_rt & x, const char * how, int64_t from, int32_t n_processed, int32_t n_reuse,
                       int32_t n_tokens, double secs, bool ok, std::chrono::steady_clock::time_point entry);

    std::unique_ptr<model::model> m_;
    std::unique_ptr<model::expert_service> ex_; // ME1 (destroyed after the targets and the prefiller)
    std::vector<slot_rt> s_;
    std::unique_ptr<prefiller> pf_;
    engine_config c_;
    std::string err_;
    // MU1: the engine's compute streams (multi-slot), the shared exchange counters, the jobs' numbers, the group
    void * st_[2] = {nullptr, nullptr};
    uint64_t ple_use_ = 0;
    model::use_counters u_[2];
    int64_t next_job_ = 1;
    int32_t read_group_ = 2; // layers a group: the server's TU policy default (MU1's G2 preset), until set_read_group
    bool test_partial_ = false; // MXMU1 (test_partial_reads)
    uint64_t release_seq_ = 0;
    int32_t reclaims_ = 0;
    std::unique_ptr<joint_frame> joint_; // FMU2: resident TP with two or more slots (engine_config::joint)
    bool joint_off_ = false;             // FMU2: a joint capture failed; later steps take turns
};

} // namespace fn::runtime
