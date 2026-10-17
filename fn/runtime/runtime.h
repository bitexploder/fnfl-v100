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
    cancelled, // BG1: a prefill stopped by its progress callback
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

class engine {
  public:
    engine();
    ~engine();
    engine(const engine &) = delete;
    engine & operator=(const engine &) = delete;

    // takes the loaded model (fn::loader::load_for_run, or a test's zero weights): the target with the draft, the
    // sampler at the launch defaults and both decision variants (target_options::both_paths); false with the reason
    bool init(std::unique_ptr<model::model> m, const engine_config & c, std::string & err);
    // the settings line printed at startup
    std::string settings() const;
    const target & tgt() const { return *t_; }
    // PF1: the last prefill's chunks (prefill_test and the server's log)
    const pf_run_stats & last_prefill() const { return pf_stats_; }

    // PB1: the last prefill call's phase line (as written to stderr)
    const std::string & last_phase_line() const { return phase_line_; }

    // TEST ONLY (prefill-test): the slot's prefiller
    prefiller & prefill_driver() { return *pf_; }

    int32_t n_slots() const { return 1; }

    // the request's options, before its prefill (between requests)
    run_status set_sampling(int32_t slot, const request_options & o);
    // n_processed: the tokens processed: n_tokens for a pure extension, n_reuse + n_tokens - c after restoring the
    // checkpoint at c (c = 0: from token 0); n_tokens = 0 processes the last kept token again. progress (BG1) runs
    // after every chunk but the last; when it stops the prefill, the slot keeps the prompt's tokens up to that chunk's
    // end without a valid state (the next prefill restores a checkpoint at or below them), n_processed is their count
    // and the status is cancelled
    run_status prefill(int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse, int32_t & n_processed,
                       const prefill_progress & progress = {});
    // out holds max_step_tokens tokens
    run_status step(int32_t slot, int32_t * out, int32_t & n_out, bool & end);
    run_status truncate(int32_t slot, int32_t n_keep);
    // the tokens the slot's state holds (processed) and its pending token (-1: none)
    run_status history(int32_t slot, std::vector<int32_t> & tokens, int32_t & pending) const;

    // SP2 (temporary, at SP2's disposition), for fnfl-drive spec ab: the draft's d (1..4) and form (target::
    // set_draft_sparse) between requests; the slot's decode-ready state after a prefill (no step since) saved to host
    // memory and restored as the next request's start (target::save_ready, restore_ready; HC8's). TEST ONLY:
    // test_target and test_model give spec ab's checks the target and the model.
    run_status set_draft(int32_t d, bool sparse);
    run_status save_ready(ready_state & s);
    run_status restore_ready(const ready_state & s);
    target & test_target() { return *t_; }
    model::model & test_model() { return *m_; }

    // the reason of the last call that failed
    const std::string & error() const { return err_; }
    // the last finished request's summary, and its line as printed
    const request_summary & last_summary() const { return sum_; }
    // SF1: the last finished request's steps (fnfl-drive spec run's rounds.csv; the server writes no per-round file)
    const std::vector<step_info> & last_steps() const { return last_steps_; }
    const std::string & last_summary_line() const { return sum_line_; }

  private:
    run_status failed(run_status s, const std::string & what);
    // the open request's closing check and summary (ended: why it ended)
    void finish(const char * ended);
    int32_t budget() const;

    std::unique_ptr<model::model> m_;
    std::unique_ptr<target> t_;
    std::unique_ptr<prefiller> pf_;
    pf_run_stats pf_stats_;
    std::string                   phase_line_;
    engine_config c_;
    request_options req_;
    bool valid_ = false; // the device state matches the ledger (false: restore a checkpoint or re-prefill)
    std::vector<int32_t>
        keep_;   // while !valid_: the tokens the slot holds, restored or re-prefilled at the next prefill
    void lost(); // a failure: the slot's state is unknown (re-prefill from token 0, no checkpoints)
    bool open_ = false;         // a request's steps have run and it has not finished
    bool unchecked_ = false;    // a decision ran since the last closing check
    int32_t n_gen_ = 0;         // the open request's tokens
    int32_t fail_step_ = 0;     // armed --fnfl-test-fail-step
    std::vector<step_info> steps_;
    std::vector<step_info> last_steps_;
    std::chrono::steady_clock::time_point t_first_; // the open request's first step's start
    double t_span_s_ = 0;                           // from it to the last step's end
    request_summary sum_;
    std::string sum_line_;
    std::string err_;
};

} // namespace fn::runtime
