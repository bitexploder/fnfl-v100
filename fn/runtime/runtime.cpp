#include "fn/runtime/runtime.h"

#include "fn/model/model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

using clk = std::chrono::steady_clock;

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

// SF1: a round's completion time (round_result::t_done_s, steady_clock seconds) after the step's start t0
double after(clk::time_point t0, double t_done_s) {
    return t_done_s - std::chrono::duration<double>(t0.time_since_epoch()).count();
}

bool is_eog(const std::vector<int32_t> & eog, int32_t tok) {
    return std::find(eog.begin(), eog.end(), tok) != eog.end();
}

// forced rounds over the ledger's unprocessed tokens at B <= 4 (the last the remainder), the draft's catch-up after each
bool forced_all(target & t, std::string & err) {
    round_result r;
    while (t.slot().u > 0) {
        const int32_t B = static_cast<int32_t>(std::min<int64_t>(4, t.slot().u));
        if (!t.forced(B, r, err) || !t.draft_catchup(B, r, err)) {
            return false;
        }
    }
    if (t.draft_n() != t.slot().n) {
        return fail(err, "prefill: the draft's catch-up is not at N");
    }
    return true;
}

double quantile(std::vector<double> v, double q) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t i = static_cast<size_t>(q * double(v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

} // namespace

bool prefill(target & t, const std::vector<int32_t> & prompt, std::string & err) {
    if (prompt.empty()) {
        return fail(err, "prefill: no tokens");
    }
    if (!t.reset(err)) {
        return false;
    }
    t.set_prompt(prompt);
    return forced_all(t, err);
}

bool extend(target & t, const int32_t * tokens, int32_t n, std::string & err) {
    return extend_ledger(t, tokens, n, err) && forced_all(t, err);
}

bool extend_ledger(target & t, const int32_t * tokens, int32_t n, std::string & err) {
    if (tokens == nullptr || n < 1) {
        return fail(err, "extend: no tokens");
    }
    if (t.draft_n() < t.slot().n) {
        // a request that ended in a generate round left the draft one row short (that step has no catch-up)
        round_result r;
        if (!t.draft_catchup(static_cast<int32_t>(t.slot().n - t.draft_n()), r, err)) {
            return false;
        }
    }
    if (t.draft_n() != t.slot().n) {
        return fail(err, "extend: the draft is not at N");
    }
    if (!t.keep_processed(err)) {
        return false;
    }
    t.append(std::vector<int32_t>(tokens, tokens + n));
    return true;
}

bool step(target & t, int32_t budget, int32_t d_max, const std::vector<int32_t> & eog, step_info & s,
          std::string & err, const after_draft_fn * after_draft) {
    s = step_info();
    const auto t0 = clk::now();
    s.n_before = t.slot().n;
    if (budget < 1) {
        return fail(err, "step: no token budget left");
    }
    if (d_max < 1 || d_max > max_spec_d) {
        return fail(err, "step: d " + std::to_string(d_max) + " outside 1 .. 4 (SP2)");
    }
    round_result r;
    if (t.slot().u == 0) {
        // the bootstrap: token zero from the retained logits of the prompt's last row (FC-RD3)
        if (!t.bootstrap(r, err)) {
            return false;
        }
        s.kind = 0;
        s.n_out = 1;
        s.out[0] = r.out[0];
        s.waits = 1;
        s.to_one_s = after(t0, r.t_done_s);
    } else if (budget == 1) {
        // one token left: a generate round (B <= budget, FC-RD1)
        if (!t.generate(r, err)) {
            return false;
        }
        s.kind = 1;
        s.n_out = 1;
        s.out[0] = r.out[0];
        s.waits = 1;
        s.to_one_s = after(t0, r.t_done_s);
    } else {
        const int32_t d = std::min(d_max, budget - 1);
        if (!t.draft_step(d, r, err)) {
            return false;
        }
        s.to_draft_s = after(t0, r.t_done_s);
        if (after_draft != nullptr && *after_draft && !(*after_draft)(r)) {
            return fail(err, "after_draft hook");
        }
        s.kind = 2;
        s.d = d;
        for (int32_t j = 0; j < d; ++j) {
            s.prop[j] = r.argmax[0][j];
        }
        round_result rv;
        if (!t.verify(d, s.prop, budget, rv, err)) {
            return false;
        }
        s.to_verify_s = after(t0, rv.t_done_s);
        s.a = rv.n_out - 1;
        s.n_out = rv.n_out;
        for (int32_t j = 0; j < rv.n_out; ++j) {
            s.out[j] = rv.out[j];
        }
        s.waits = 2;
        for (int32_t j = 0; j + 1 < s.n_out; ++j) {
            if (is_eog(eog, s.out[j])) {
                // the tokens after it are not returned: rows 0 .. j kept by version, out[j] the pending token
                if (!t.cut(j + 1, s.n_before + j + 2, err)) {
                    return false;
                }
                s.n_out = j + 1;
                s.cut = true;
                ++s.waits;
                break;
            }
        }
        if (!t.draft_catchup(s.n_out, r, err)) {
            return false;
        }
        s.to_catchup_s = after(t0, r.t_done_s);
        ++s.waits;
    }
    s.end = is_eog(eog, s.out[s.n_out - 1]);
    s.wall_s = since(t0);
    return true;
}

// ---- the engine: the one slot behind the seam ----

engine::engine() = default;

engine::~engine() {
    pf_.reset();
    t_.reset(); // the target synchronizes its streams and frees its graphs while the model is alive
    m_.reset();
}

void engine::lost() {
    const ledger & l = t_->slot();
    valid_ = false;
    keep_.assign(l.tok.begin(), l.tok.begin() + std::min<int64_t>(l.n, int64_t(l.tok.size())));
    if (pf_ != nullptr) {
        pf_->drop_all();
    }
}

bool engine::init(std::unique_ptr<model::model> m, const engine_config & c, std::string & err) {
    if (m == nullptr || c.n_ctx_slot < 1 || c.draft.subset.empty()) {
        return fail(err, "engine: no model, cells or draft subset");
    }
    if (c.spec_d < 1 || c.spec_d > max_spec_d) {
        return fail(err, "engine: the proposal count d " + std::to_string(c.spec_d) + " is outside 1 .. 4 (SP2)");
    }
    m_ = std::move(m);
    c_ = c;
    c_.draft.on = true; // speculation is the served path; there is no draft-less mode
    target_options o;
    o.model_path = c_.model_path;
    o.timeout_s = c_.timeout_s;
    o.sample = true;
    o.sampling = fn::samp::options(); // the launch sampling until a request sets its own
    o.sampling.temp = c_.temp;
    o.sampling.top_k = c_.top_k;
    o.sampling.top_p = c_.top_p;
    o.sampling.min_p = c_.min_p;
    o.draft = c_.draft;
    o.both_paths = true;
    t_ = std::make_unique<target>();
    if (!t_->init(*m_, o, err)) {
        return false;
    }
    pf_ = std::make_unique<prefiller>();
    if (!pf_->init(*t_, *m_, err)) {
        return false;
    }
    req_ = request_options();
    req_.sampling = o.sampling;
    fail_step_ = c_.test_fail_step;
    valid_ = false;
    keep_.clear();
    return true;
}

std::string engine::settings() const {
    std::string eog;
    for (size_t i = 0; i < c_.eog.size(); ++i) {
        eog += (i ? "," : "") + std::to_string(c_.eog[i]);
    }
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "FNFL settings: slots 1; cells %d; draft on, attention %s (SP2's switch), d %d, subset %s; "
             "end-of-generation IDs [%s]; launch sampler T %.2f top-k %d top-p %.2f min-p %.2f (per request: "
             "fnfl_set_sampling); decision variants captured: 2 (the second: %d graphs, %.3f s, %.1f / %.1f MiB on "
             "cards 0 / 1); test-fail-step %d (SV1 test switch, 0 off); step log %s (SV1 test switch)",
             c_.n_ctx_slot, c_.draft.attention.c_str(), c_.spec_d, c_.draft.subset.c_str(), eog.c_str(),
             double(req_.sampling.temp), req_.sampling.top_k, double(req_.sampling.top_p), double(req_.sampling.min_p),
             t_ ? t_->variant_captures() : 0, t_ ? t_->variant_capture_s() : 0.0,
             t_ ? double(t_->variant_bytes(0)) / 1048576.0 : 0.0, t_ ? double(t_->variant_bytes(1)) / 1048576.0 : 0.0,
             c_.test_fail_step, c_.log_steps ? "on" : "off");
    return buf;
}

run_status engine::failed(run_status s, const std::string & what) {
    err_ = what;
    fprintf(stderr, "FNFL: %s\n", what.c_str());
    fflush(stderr);
    return s;
}

int32_t engine::budget() const {
    const ledger & l = t_->slot();
    const int64_t n_ctx = req_.n_ctx > 0 ? std::min(req_.n_ctx, c_.n_ctx_slot) : c_.n_ctx_slot;
    int64_t left = n_ctx - static_cast<int64_t>(l.tok.size()); // the context room: the pending token is held
    if (req_.max_tokens >= 0) {
        left = std::min<int64_t>(left, int64_t(req_.max_tokens) - n_gen_);
    }
    return static_cast<int32_t>(std::max<int64_t>(-1, std::min<int64_t>(left, 1 << 30)));
}

void engine::finish(const char * ended) {
    if (!open_) {
        return;
    }
    open_ = false;
    std::string check = "not needed";
    if (unchecked_ && valid_) {
        bool checked = false;
        std::string e;
        if (!t_->close_check(checked, e)) {
            check = "FAILED: " + e;
            lost(); // the next request resets the slot and re-prefills
        } else {
            check = checked ? "checked" : "NOT CHECKED";
        }
    }
    unchecked_ = false;
    request_summary s;
    s.ended = ended;
    std::vector<double> wall, ph[4];
    double own_s = 0;
    for (const step_info & x : steps_) {
        const double to[4] = {x.to_draft_s, x.to_verify_s, x.to_catchup_s, x.to_one_s};
        for (int32_t i = 0; i < 4; ++i) {
            if (to[i] >= 0) {
                ph[i].push_back(to[i] * 1e3);
            }
        }
        s.n_before_min = s.steps == 0 ? x.n_before : std::min(s.n_before_min, x.n_before);
        s.n_before_max = s.steps == 0 ? x.n_before : std::max(s.n_before_max, x.n_before);
        ++s.steps;
        s.tokens += x.n_out;
        s.cuts += x.cut ? 1 : 0;
        wall.push_back(x.wall_s * 1e3);
        if (x.kind == 0) {
            ++s.boot;
            s.waits_one += x.waits;
        } else if (x.kind == 1) {
            ++s.gen;
            s.waits_one += x.waits;
        } else {
            ++s.spec;
            s.accepted += x.a;
            s.spec_tokens += x.n_out;
            s.waits_spec += x.waits;
        }
    }
    s.wall_median_ms = quantile(wall, 0.5);
    s.wall_p90_ms = quantile(wall, 0.9);
    s.draft_median_ms = quantile(ph[0], 0.5);
    s.verify_median_ms = quantile(ph[1], 0.5);
    s.catchup_median_ms = quantile(ph[2], 0.5);
    s.one_median_ms = quantile(ph[3], 0.5);
    for (const double w : wall) {
        own_s += w / 1e3;
    }
    s.span_s = steps_.empty() ? 0.0 : t_span_s_;
    char buf[1536];
    snprintf(buf, sizeof(buf),
             "FNFL request: ended %s; steps %d (bootstrap %d, speculative %d, generate %d); tokens %d; tokens a "
             "speculative step %.3f; accepted drafts a speculative step %.3f; step wall ms median %.3f p90 %.3f (host "
             "clock around each step, all steps); host waits a step: speculative %.2f, bootstrap and generate %.2f; "
             "ms from a step's start to its waits, medians: draft step %.3f, verify %.3f, catch-up %.3f "
             "(speculative), the one round %.3f (bootstrap and generate); depth before a step %lld..%lld; "
             "end-of-generation cuts %d; decode span %.4f s (the bootstrap's start to the last step's end; steps' "
             "own wall %.4f s): %.2f tokens/s (all tokens), %.2f tokens/s ((tokens - 1) / span: the server's "
             "predicted_per_second definition); closing check %s",
             s.ended.c_str(), s.steps, s.boot, s.spec, s.gen, s.tokens, s.spec ? double(s.spec_tokens) / s.spec : 0.0,
             s.spec ? double(s.accepted) / s.spec : 0.0, s.wall_median_ms, s.wall_p90_ms,
             s.spec ? double(s.waits_spec) / s.spec : 0.0,
             (s.boot + s.gen) ? double(s.waits_one) / (s.boot + s.gen) : 0.0, s.draft_median_ms, s.verify_median_ms,
             s.catchup_median_ms, s.one_median_ms, static_cast<long long>(s.n_before_min),
             static_cast<long long>(s.n_before_max), s.cuts, s.span_s, own_s, s.span_s > 0 ? s.tokens / s.span_s : 0.0,
             s.span_s > 0 ? (s.tokens - 1) / s.span_s : 0.0, check.c_str());
    sum_ = s;
    sum_line_ = buf;
    fprintf(stderr, "%s\n", buf);
    fflush(stderr);
    last_steps_.swap(steps_);
    steps_.clear();
}

run_status engine::set_sampling(int32_t slot, const request_options & o) {
    if (slot != 0) {
        return failed(run_status::bad_request, "slot " + std::to_string(slot) + ": FNFL serves one slot");
    }
    finish("the next request");
    if (!t_->set_sampling(o.sampling, err_)) {
        return failed(run_status::bad_request, err_);
    }
    req_ = o;
    return run_status::ok;
}

run_status engine::prefill(int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                           int32_t & n_processed, const prefill_progress & progress) {
    const auto entry = clk::now();
    n_processed = 0;
    finish("the next request");
    const ledger & l = t_->slot();
    const std::vector<int32_t> & held_tok = valid_ ? l.tok : keep_;
    const int64_t held = valid_ ? l.n : static_cast<int64_t>(keep_.size());
    if (slot != 0 || n_tokens < 0 || (n_tokens > 0 && tokens == nullptr) || n_reuse < 0 || n_reuse > held ||
        (n_tokens == 0 && n_reuse < 1) || int64_t(n_reuse) + n_tokens > c_.n_ctx_slot) {
        return failed(run_status::bad_request, "prefill: slot " + std::to_string(slot) + ", " +
                                                   std::to_string(n_tokens) + " tokens after " +
                                                   std::to_string(n_reuse) + " kept of " + std::to_string(held) +
                                                   " held, " + std::to_string(c_.n_ctx_slot) + " cells");
    }
    // FC-PF1: nothing new: keep n_reuse - 1 tokens and process the last kept one again, so its logits exist
    int32_t one = 0;
    if (n_tokens == 0) {
        one = held_tok[size_t(n_reuse - 1)];
        tokens = &one;
        n_tokens = 1;
        --n_reuse;
    }
    bool ok = false;
    const auto t0 = clk::now();
    // a pure extension continues the slot; the draft may be up to one step short (extend_ledger catches it up)
    const bool draft_ok = t_->draft_n() >= t_->batch_start() && t_->draft_n() <= l.n && l.n - t_->draft_n() <= 4;
    int64_t from = 0;
    const char * how = "a pure extension";
    if (valid_ && n_reuse == l.n && draft_ok && !t_->invalid()) {
        from = l.n;
        pf_stats_ = pf_run_stats();
        ok = extend_ledger(*t_, tokens, n_tokens, err_) && pf_->run(*t_, &pf_stats_, err_, progress);
    } else {
        std::vector<int32_t> all(held_tok.begin(), held_tok.begin() + n_reuse);
        all.insert(all.end(), tokens, tokens + n_tokens);
        pf_stats_ = pf_run_stats();
        ok = pf_->begin(*t_, all, n_reuse, from, err_) && pf_->run(*t_, &pf_stats_, err_, progress);
        how = from > 0 ? "from a checkpoint" : "from token 0";
    }
    n_processed = static_cast<int32_t>(int64_t(n_reuse) + n_tokens - from);
    const int64_t stopped = ok ? -1 : pf_stats_.stopped;
    std::string chunks;
    for (const pf_chunk_stats & c : pf_stats_.chunks) {
        chunks += (chunks.empty() ? "" : " ") + std::to_string(c.c.start) + "+" + std::to_string(c.c.rows) +
                  (c.checkpoint >= 0 ? "*" : "");
    }
    fprintf(stderr,
            "FNFL prefill: %s at %lld: %d tokens processed (%d kept by the shell, %d new; %.3f s); chunks (start+rows, "
            "* checkpoint): %s; checkpoints held %d%s\n",
            how, static_cast<long long>(from), n_processed, n_reuse, n_tokens, since(t0), chunks.c_str(),
            pf_->book().count(), stopped >= 0 ? ("; stopped (cancelled) at " + std::to_string(stopped)).c_str() : "");
    // PB1 item 6: the call's phase line, once, from host markers at existing completion boundaries (no added event,
    // synchronization or per-chunk write). Steady-clock ms. The target's completion is not separately observable: the
    // target and the draft end at one join at each chunk's end, so it is marked unavailable. The draft's marker is the
    // last chunk's join; decode-ready is after the head's join and the handoff check (the bootstrap round follows).
    {
        const auto ms = [](clk::time_point p) {
            return std::chrono::duration<double, std::milli>(p.time_since_epoch()).count();
        };
        const auto mark = [&](double s) {
            if (s < 0) {
                return std::string(pf_stats_.bulk || !ok ? "not reached" : "unavailable (the forced path)");
            }
            const clk::time_point p =
                pf_stats_.t0 + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(s));
            char b[96];
            snprintf(b, sizeof b, "%.3f (+%.3f)", ms(p), std::chrono::duration<double, std::milli>(p - entry).count());
            return std::string(b);
        };
        char b[160];
        snprintf(b, sizeof b, "FNFL prefill phases (steady-clock ms, + from entry): entry %.3f; target unavailable",
                 ms(entry));
        phase_line_ = std::string(b) + " (it joins with the draft at each chunk's end); draft " +
                      mark(pf_stats_.last_join_s) + " (the last chunk's join); decode-ready " +
                      mark(pf_stats_.ready_s) + "; rows " + std::to_string(ok ? n_processed : 0) +
                      "; restored checkpoint " +
                      (from > 0 && std::string(how) == "from a checkpoint" ? std::to_string(from) : "none") +
                      (pf_stats_.bulk ? "" : "; the forced path");
        fprintf(stderr, "%s\n", phase_line_.c_str());
    }
    fflush(stderr);
    valid_ = ok;
    keep_.clear();
    n_gen_ = 0;
    steps_.clear();
    if (stopped >= 0) {
        // BG1: the chunks before the stop and their checkpoints are kept; the next prefill restores the latest
        // checkpoint at or below the kept prompt (the stopped run's handoff never happened, so there is no valid state)
        const ledger & l = t_->slot();
        keep_.assign(l.tok.begin(), l.tok.begin() + std::min<int64_t>(stopped, int64_t(l.tok.size())));
        n_processed = static_cast<int32_t>(keep_.size());
        err_ = "prefill cancelled at " + std::to_string(stopped);
        return run_status::cancelled;
    }
    if (!ok) {
        pf_->drop_all();
        return failed(run_status::error, "prefill: " + err_);
    }
    return run_status::ok;
}

run_status engine::set_draft(int32_t d, bool sparse) {
    finish("the draft setting");
    if (d < 1 || d > max_spec_d) {
        return failed(run_status::bad_request, "set_draft: d " + std::to_string(d) + " outside 1 .. 4");
    }
    if (!t_->set_draft_sparse(sparse, err_)) {
        return failed(run_status::bad_request, "set_draft: " + err_);
    }
    c_.spec_d = d;
    return run_status::ok;
}

run_status engine::save_ready(ready_state & s) {
    if (!valid_ || open_ || !steps_.empty()) {
        return failed(run_status::bad_request, "save_ready: no prefill handoff (a prefill, then no step)");
    }
    return t_->save_ready(s, err_) ? run_status::ok : failed(run_status::error, "save_ready: " + err_);
}

run_status engine::restore_ready(const ready_state & s) {
    finish("the next request");
    if (!t_->restore_ready(s, err_)) {
        const std::string what = "restore_ready: " + err_;
        lost();
        return failed(run_status::error, what);
    }
    valid_ = true;
    keep_.clear();
    n_gen_ = 0;
    steps_.clear();
    return run_status::ok;
}

run_status engine::step(int32_t slot, int32_t * out, int32_t & n_out, bool & end) {
    n_out = 0;
    end = false;
    if (slot != 0 || out == nullptr) {
        return failed(run_status::bad_request, "step: slot " + std::to_string(slot));
    }
    if (!valid_) {
        return failed(run_status::bad_request, "step: the slot holds no processed prompt (prefill first)");
    }
    if (fail_step_ > 0 && static_cast<int32_t>(steps_.size()) + 1 == fail_step_) {
        // TEST (SV1): the failure a round's error returns, before the step's rounds (FC-RD2 leaves the pre-step
        // state); the request ends with this message and the next one resets the slot and re-prefills
        const int32_t k = fail_step_;
        fail_step_ = 0;
        lost();
        open_ = false;
        unchecked_ = false;
        steps_.clear();
        return failed(run_status::error, "step " + std::to_string(k) +
                                             ": injected failure (SV1 test switch --fnfl-test-fail-step); the next "
                                             "request resets the slot and re-prefills");
    }
    int32_t b = budget();
    if (b < 1) {
        if (!steps_.empty()) {
            return failed(run_status::bad_request, "step: no token budget left (max_tokens or the context)");
        }
        b = 1; // a request's first step (the bootstrap) always returns its token
    }
    if (steps_.empty()) {
        t_first_ = clk::now();
    }
    step_info s;
    if (!runtime::step(*t_, b, c_.spec_d, c_.eog, s, err_)) {
        const std::string what = "step: " + err_;
        lost();
        open_ = false;
        unchecked_ = false;
        steps_.clear();
        return failed(run_status::error, what);
    }
    t_span_s_ = since(t_first_);
    open_ = true;
    unchecked_ = true;
    n_gen_ += s.n_out;
    steps_.push_back(s);
    for (int32_t j = 0; j < s.n_out; ++j) {
        out[j] = s.out[j];
    }
    n_out = s.n_out;
    end = s.end;
    if (c_.log_steps) {
        fprintf(stderr, "FNFL step %zu: kind %s N %lld d %d a %d tokens", steps_.size(),
                s.kind == 0   ? "bootstrap"
                : s.kind == 1 ? "generate"
                              : "spec",
                static_cast<long long>(s.n_before), s.d, s.a);
        for (int32_t j = 0; j < s.n_out; ++j) {
            fprintf(stderr, " %d", s.out[j]);
        }
        fprintf(stderr, "%s%s; waits %d; %.3f ms\n", s.cut ? " (cut at end of generation)" : "", s.end ? " END" : "",
                s.waits, s.wall_s * 1e3);
        fflush(stderr);
    }
    if (s.end) {
        finish("eog");
    } else if (budget() < 1) {
        finish("budget");
    }
    return run_status::ok;
}

run_status engine::truncate(int32_t slot, int32_t n_keep) {
    if (slot != 0 || n_keep < 0) {
        return failed(run_status::bad_request, "truncate: slot " + std::to_string(slot));
    }
    finish("the shell (a stop string, its limit or a cancel)");
    if (!valid_) {
        if (static_cast<size_t>(n_keep) < keep_.size()) {
            keep_.resize(static_cast<size_t>(n_keep));
        }
        return run_status::ok;
    }
    const ledger & l = t_->slot();
    if (n_keep >= l.n) {
        // the processed tokens stay; the pending token goes (the next prefill brings it back as a prompt token)
        if (!t_->keep_processed(err_)) {
            lost();
        }
        return run_status::ok;
    }
    std::vector<int32_t> kept(l.tok.begin(), l.tok.begin() + n_keep);
    const step_record & last = t_->versions().last();
    if (last.valid && !last.prompt && n_keep >= last.n_before) {
        // within the last step: its version (SR1), then the draft catches up the kept rows (PB1: not within a bulk
        // prompt chunk, whose versions are half ends: the restore below)
        const int32_t j = static_cast<int32_t>(n_keep - last.n_before);
        const int64_t nk = static_cast<int64_t>(l.tok.size()) > n_keep ? int64_t(n_keep) + 1 : int64_t(n_keep);
        bool ok = t_->cut(j, nk, err_) && t_->keep_processed(err_);
        if (ok && t_->draft_n() >= 0 && t_->draft_n() < l.n) {
            round_result r;
            ok = t_->draft_catchup(static_cast<int32_t>(l.n - t_->draft_n()), r, err_);
        }
        if (ok && t_->draft_n() == l.n && l.n == n_keep) {
            return run_status::ok;
        }
        fprintf(stderr, "FNFL: truncate to %d within the last step: %s; restore at the next prefill instead\n", n_keep,
                ok ? "the draft did not follow" : err_.c_str());
        if (!ok) {
            pf_->drop_all(); // a failed write: the state is unknown, the next prefill re-prefills from token 0
        }
    }
    // deeper: the next prefill restores the latest checkpoint at or before the kept length and re-prefills from it
    // (PF1); the state below every kept checkpoint is unchanged by the cut
    valid_ = false;
    keep_ = std::move(kept);
    return run_status::ok;
}

run_status engine::history(int32_t slot, std::vector<int32_t> & tokens, int32_t & pending) const {
    pending = -1;
    if (slot != 0) {
        return run_status::bad_request;
    }
    if (!valid_) {
        tokens = keep_;
        return run_status::ok;
    }
    const ledger & l = t_->slot();
    tokens.assign(l.tok.begin(), l.tok.begin() + l.n);
    if (l.u == 1 && l.pending >= 0) {
        pending = l.pending;
    }
    return run_status::ok;
}

} // namespace fn::runtime
