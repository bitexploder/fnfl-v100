#include "fn/runtime/runtime.h"

#include "fn/model/kv_pool.h"
#include "fn/model/model.h"

#include <cuda_runtime.h>

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
    // TTTS1: a split's front holds no draft (the back's catches up there)
    if (!t.stage_front() && t.draft_n() < t.slot().n) {
        // a request that ended in a generate round left the draft one row short (that step has no catch-up)
        round_result r;
        if (!t.draft_catchup(static_cast<int32_t>(t.slot().n - t.draft_n()), r, err)) {
            return false;
        }
    }
    if (!t.stage_front() && t.draft_n() != t.slot().n) {
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
        if (!step_propose(t, budget, d_max, t0, s, err, after_draft)) {
            return false;
        }
        round_result rv;
        if (!t.verify(s.d, s.prop, budget, rv, err)) {
            return false;
        }
        return step_close(t, eog, rv, t0, s, err);
    }
    s.end = is_eog(eog, s.out[s.n_out - 1]);
    s.wall_s = since(t0);
    return true;
}

bool step_propose(target & t, int32_t budget, int32_t d_max, std::chrono::steady_clock::time_point t0, step_info & s,
                  std::string & err, const after_draft_fn * after_draft) {
    round_result r;
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
    return true;
}

bool step_close(target & t, const std::vector<int32_t> & eog, const round_result & rv,
                std::chrono::steady_clock::time_point t0, step_info & s, std::string & err) {
    round_result r;
    if (!step_cut(t, eog, rv, t0, s, err) || !t.draft_catchup(s.n_out, r, err)) {
        return false;
    }
    step_caught(eog, r, t0, s);
    return true;
}

bool step_cut(target & t, const std::vector<int32_t> & eog, const round_result & rv,
              std::chrono::steady_clock::time_point t0, step_info & s, std::string & err) {
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
    return true;
}

void step_caught(const std::vector<int32_t> & eog, const round_result & r, std::chrono::steady_clock::time_point t0,
                 step_info & s) {
    s.to_catchup_s = after(t0, r.t_done_s);
    ++s.waits;
    s.end = is_eog(eog, s.out[s.n_out - 1]);
    s.wall_s = since(t0);
}

// ---- the engine: the slots behind the seam (MU1: one or 2..4) ----

namespace {

// MU1: cells a step can write past the slot's tokens (the verify round's rows and the draft's speculative cells), and
// a guard for the reads of a tile (pages are 16,384 cells, every kernel tile a divisor)
constexpr int64_t step_cells = max_step_tokens + 64;

} // namespace

engine::engine() = default;

engine::~engine() {
    shutdown();
    m_.reset();
}

std::unique_ptr<model::model> engine::test_take_model() {
    shutdown();
    return std::move(m_);
}

void engine::shutdown() {
    // ME3 (Fable's load G ruling, Strata #267): no stream left waiting on a cache flag nobody writes, so the target's
    // synchronize ends; nothing when the streams are idle
    if (ex_ != nullptr) {
        std::string note;
        if (!ex_->release_gpu_waits(5000, true, note) || !note.empty()) {
            std::fprintf(stderr, "FNFL: shutdown: the expert cache: %s\n", note.c_str());
        }
    }
    pf_.reset();
    for (slot_rt & x : s_) {
        x.t.reset(); // each target synchronizes its streams and frees its graphs while the model is alive
    }
    joint_.reset(); // FMU2: the joint frame too, on the engine's streams (its graphs, rings and buffers)
    for (int32_t c = 0; c < 2 && m_ != nullptr; ++c) {
        if (st_[c] != nullptr && (c == 0 || st_[c] != st_[0])) { // MXMU1: one card's slots share card 0's stream
            cudaSetDevice(m_->device[c]);
            cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c]));
            cudaStreamDestroy(static_cast<cudaStream_t>(st_[c]));
        }
    }
    st_[0] = st_[1] = nullptr;
    s_.clear();
    ex_.reset();
}

void engine::lost(slot_rt & x) {
    const ledger & l = x.t->slot();
    x.valid = false;
    x.keep.assign(l.tok.begin(), l.tok.begin() + std::min<int64_t>(l.n, int64_t(l.tok.size())));
    if (pf_ != nullptr) {
        if (n_slots() > 1) {
            pf_->drop_owner(x.t->slot_index());
        } else {
            pf_->drop_all();
        }
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
    const int32_t n = m_->n_slots;
    // MU1: 2..4 slots on resident TP; MXMU1: two on one card with the expert cache and the host-backed KV
    const bool one_card = m_->single && m_->kv_host && c_.experts != nullptr;
    // TTTS1: a split stage serves up to 8 slots (resident TP on its window); one box up to 4
    const int32_t max_n    = c_.stage != 0 ? max_split_slots : 4;
    if (c_.stage != 0 && (int32_t(c_.stage_buf.size()) < n || c_.experts != nullptr || m_->single || m_->kv_host)) {
        return fail(err, "engine: a split stage needs every slot's boundary buffers, resident TP, no expert cache");
    }
    if (n < 1 || n > max_n || (n > 1 && !one_card && (c_.experts != nullptr || m_->single || m_->kv_host)) ||
        (one_card && n > 2)) {
        return fail(err, "engine: " + std::to_string(n) +
                             " slots (1 .. 4 on resident TP, MU1; 1 .. 2 on one card with the host-backed KV, MXMU1)");
    }
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
    o.draft.on       = c_.stage != 1;  // TTTS1: the draft is the back's
    o.both_paths = true;
    o.eager = c_.eager;
    o.pass.layer_lo  = m_->win.lo;                         // TTTS1: the box's window
    o.pass.layer_hi  = m_->win.hi;
    o.defer_capture = c_.experts != nullptr && !c_.eager; // ME3: the graphs inside the cache's init (capture below)
    if (n > 1) {
        // MU1: one compute stream a card for every slot's rounds and the prompt path, and the shared rings' counters.
        // MXMU1: one card's two logical cards share one stream (target::init's one-device rule; the expert cache's
        // windows and the single-card graphs need it)
        for (int32_t k = 0; k < 2; ++k) {
            if (k == 1 && m_->single) {
                st_[1] = st_[0];
                o.streams[1] = st_[0];
                continue;
            }
            cudaStream_t cs = nullptr;
            if (cudaSetDevice(m_->device[k]) != cudaSuccess ||
                cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess) {
                return fail(err, "engine: the compute streams");
            }
            st_[k] = cs;
            o.streams[k] = cs;
        }
        o.ple_use = &ple_use_;
        o.eager_uses = u_;
    }
    s_.clear();
    s_.resize(size_t(n));
    for (int32_t i = 0; i < n; ++i) {
        o.slot = i;
        if (c_.stage != 0) {
            // TTTS1: the slot's boundary buffers, fixed for its graphs (the front's send, the back's receive)
            o.pass.stage     = model::pass_stage();
            o.pass.stage.out = c_.stage == 1;
            o.pass.stage.in  = c_.stage == 2;
            for (int32_t c = 0; c < 2; ++c) {
                if (c_.stage == 1) {
                    o.pass.stage.send[c] = c_.stage_buf[size_t(i)][size_t(c)];
                } else {
                    o.pass.stage.recv[c] = c_.stage_buf[size_t(i)][size_t(c)];
                }
            }
        }
        s_[size_t(i)].t = std::make_unique<target>();
        if (!s_[size_t(i)].t->init(*m_, o, err)) {
            return n > 1 ? fail(err, "slot " + std::to_string(i) + ": " + err) : false;
        }
    }
    target * const t0 = s_[0].t.get();
    // ME2b: the expert cache before the prefiller: on one card it lends PB1's arena and PF1's staging to its pool (the
    // classes' buffers become views into it), so the prompt path carves the views. ME3: the graph executor's graphs
    // are captured through the service once its buffers exist, before its pool. MXMU1: every slot's cards take the one
    // service and every slot's graphs are captured there, so the pool is sized after all of them
    if (c_.experts) {
        const model::service_hook capture = [this](model::expert_service & sv, std::string & e) {
            sv.set_executor(true);
            for (slot_rt & x : s_) {
                x.t->card(0).c.ex_service = &sv;
                x.t->card(1).c.ex_service = &sv;
                if (!x.t->capture(e)) {
                    return false;
                }
            }
            return true;
        };
        ex_ = c_.experts(*m_, c_.eager ? model::service_hook() : capture, err);
        if (ex_ == nullptr) {
            return false;
        }
        for (slot_rt & x : s_) {
            x.t->card(0).c.ex_service = ex_.get();
            x.t->card(1).c.ex_service = ex_.get();
            if (!x.t->set_executor(!c_.eager, err)) {
                return false;
            }
        }
    }
    pf_ = std::make_unique<prefiller>();
    if (!pf_->init(*t0, *m_, err)) {
        return false;
    }
    if (ex_ != nullptr && (!pf_->set_options(c_.short_read, c_.prompt_chunk, err) || !pf_->loan_check(err))) {
        return false;
    }
    if (n > 1 && m_->pool != nullptr && !pf_->init_lane(*t0, err)) { // MXMU1: one card has no other-reader lane
        return false;
    }
    for (slot_rt & x : s_) {
        x.req = request_options();
        x.req.sampling = o.sampling;
        x.fail_step = c_.test_fail_step;
        x.valid = false;
        x.keep.clear();
    }
    // FMU2: the joint round's frame on resident TP with two or more slots (eager: enqueued in phases, no graphs)
    if (n > 1 && c_.joint && c_.experts == nullptr && !m_->single && !m_->kv_host) {
        joint_ = std::make_unique<joint_frame>();
        if (!joint_->init(*m_, st_, c_.joint_cache, c_.eager, c_.joint_x16, c_.joint_router_tc, c_.joint_dense_f16,
                          err)) {
            return false;
        }
    }
    return true;
}

std::string engine::settings() const {
    std::string eog;
    for (size_t i = 0; i < c_.eog.size(); ++i) {
        eog += (i ? "," : "") + std::to_string(c_.eog[i]);
    }
    const target * t = s_.empty() ? nullptr : s_[0].t.get();
    char buf[1536];
    snprintf(buf, sizeof(buf),
             "FNFL settings: slots %d; cells %d; draft on, attention %s (SP2's switch), d %d, subset %s; "
             "end-of-generation IDs [%s]; launch sampler T %.2f top-k %d top-p %.2f min-p %.2f (per request: "
             "fnfl_set_sampling); decision variants captured: 2 (the second: %d graphs, %.3f s, %.1f / %.1f MiB on "
             "cards 0 / 1); test-fail-step %d (SV1 test switch, 0 off); step log %s (SV1 test switch); executor %s "
             "(ME3; graphs %.1f / %.1f MiB on cards 0 / 1)",
             n_slots(), c_.n_ctx_slot, c_.draft.attention.c_str(), c_.spec_d, c_.draft.subset.c_str(), eog.c_str(),
             s_.empty() ? 0.0 : double(s_[0].req.sampling.temp), s_.empty() ? 0 : s_[0].req.sampling.top_k,
             s_.empty() ? 0.0 : double(s_[0].req.sampling.top_p), s_.empty() ? 0.0 : double(s_[0].req.sampling.min_p),
             t ? t->variant_captures() : 0, t ? t->variant_capture_s() : 0.0,
             t ? double(t->variant_bytes(0)) / 1048576.0 : 0.0, t ? double(t->variant_bytes(1)) / 1048576.0 : 0.0,
             c_.test_fail_step, c_.log_steps ? "on" : "off", t && t->graph_executor() ? "graph" : "eager",
             t ? double(t->graph_bytes(0)) / 1048576.0 : 0.0, t ? double(t->graph_bytes(1)) / 1048576.0 : 0.0);
    std::string out = buf;
    if (n_slots() > 1) {
        // MU1: each slot's graphs (captured against its own frame)
        for (int32_t i = 0; i < n_slots(); ++i) {
            const target & ti = *s_[size_t(i)].t;
            snprintf(buf, sizeof(buf), "; slot %d graphs %.1f / %.1f MiB", i, double(ti.graph_bytes(0)) / 1048576.0,
                     double(ti.graph_bytes(1)) / 1048576.0);
            out += buf;
        }
        snprintf(buf, sizeof(buf), "; read groups of %d layers; lanes: main %d rows, other-reader %d rows", read_group_,
                 lane_rows(0), lane_rows(1));
        out += buf;
        if (joint_ != nullptr) {
            if (joint_->eager()) {
                snprintf(buf, sizeof(buf),
                         "; joint decode on (FMU2: frame %.1f / %.1f MiB on cards 0 / 1, eager; X-A/X-F 16-bit %s; "
                         "router on the tensor cores %s (FMS5); dense FP16 at U 3, 4 %s (FMS6))",
                         double(joint_->bytes(0)) / 1048576.0, double(joint_->bytes(1)) / 1048576.0,
                         joint_->x16() ? "on" : "off", joint_->router_tc() ? "on" : "off",
                         joint_->dense_f16() ? "on" : "off");
            } else {
                snprintf(buf, sizeof(buf),
                         "; joint decode on (FMU2: frame %.1f / %.1f MiB on cards 0 / 1, %d graph "
                         "pairs kept; X-A/X-F 16-bit %s; router on the tensor cores %s (FMS5); dense FP16 at U 3, "
                         "4 %s (FMS6))",
                         double(joint_->bytes(0)) / 1048576.0, double(joint_->bytes(1)) / 1048576.0, c_.joint_cache,
                         joint_->x16() ? "on" : "off", joint_->router_tc() ? "on" : "off",
                         joint_->dense_f16() ? "on" : "off");
            }
        } else {
            snprintf(buf, sizeof(buf), "; joint decode off");
        }
        out += buf;
    }
    return out;
}

run_status engine::failed(run_status s, const std::string & what) {
    err_ = what;
    fprintf(stderr, "FNFL: %s\n", what.c_str());
    fflush(stderr);
    return s;
}

int32_t engine::budget(const slot_rt & x) const {
    const ledger & l = x.t->slot();
    const int64_t n_ctx = x.req.n_ctx > 0 ? std::min(x.req.n_ctx, c_.n_ctx_slot) : c_.n_ctx_slot;
    int64_t left = n_ctx - static_cast<int64_t>(l.tok.size()); // the context room: the pending token is held
    if (x.req.max_tokens >= 0) {
        left = std::min<int64_t>(left, int64_t(x.req.max_tokens) - x.n_gen);
    }
    return static_cast<int32_t>(std::max<int64_t>(-1, std::min<int64_t>(left, 1 << 30)));
}

void engine::finish(slot_rt & x, const char * ended) {
    if (!x.open) {
        return;
    }
    x.open = false;
    std::string check = "not needed";
    if (x.unchecked && x.valid && x.t->stage_front()) {
        check = "the back's (a front stage decides nothing)";  // TTTS1
    } else if (x.unchecked && x.valid) {
        bool checked = false;
        std::string e;
        if (!x.t->close_check(checked, e)) {
            check = "FAILED: " + e;
            lost(x); // the next request resets the slot and re-prefills
        } else {
            check = checked ? "checked" : "NOT CHECKED";
        }
    }
    x.unchecked = false;
    request_summary s;
    s.ended = ended;
    std::vector<double> wall, ph[4];
    double own_s = 0;
    for (const step_info & st : x.steps) {
        const double to[4] = {st.to_draft_s, st.to_verify_s, st.to_catchup_s, st.to_one_s};
        for (int32_t i = 0; i < 4; ++i) {
            if (to[i] >= 0) {
                ph[i].push_back(to[i] * 1e3);
            }
        }
        s.n_before_min = s.steps == 0 ? st.n_before : std::min(s.n_before_min, st.n_before);
        s.n_before_max = s.steps == 0 ? st.n_before : std::max(s.n_before_max, st.n_before);
        ++s.steps;
        s.tokens += st.n_out;
        s.cuts += st.cut ? 1 : 0;
        wall.push_back(st.wall_s * 1e3);
        if (st.kind == 0) {
            ++s.boot;
            s.waits_one += st.waits;
        } else if (st.kind == 1) {
            ++s.gen;
            s.waits_one += st.waits;
        } else {
            ++s.spec;
            s.accepted += st.a;
            s.spec_tokens += st.n_out;
            s.waits_spec += st.waits;
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
    s.span_s = x.steps.empty() ? 0.0 : x.t_span_s;
    char buf[1600];
    // MU1: the slot last, so the line's fields parse as the one-slot engine's (reconcile_run.py's FNFL_REQ_RE)
    // FMU2: and the speculative steps whose verify round was a joint round's lane, before it
    int32_t joints = 0;
    for (const step_info & st : x.steps) {
        joints += st.joint ? 1 : 0;
    }
    const std::string who =
        n_slots() > 1 ? "; joint rounds " + std::to_string(joints) + "; slot " + std::to_string(x.t->slot_index())
                      : std::string();
    snprintf(buf, sizeof(buf),
             "FNFL request: ended %s; steps %d (bootstrap %d, speculative %d, generate %d); tokens %d; tokens a "
             "speculative step %.3f; accepted drafts a speculative step %.3f; step wall ms median %.3f p90 %.3f (host "
             "clock around each step, all steps); host waits a step: speculative %.2f, bootstrap and generate %.2f; "
             "ms from a step's start to its waits, medians: draft step %.3f, verify %.3f, catch-up %.3f "
             "(speculative), the one round %.3f (bootstrap and generate); depth before a step %lld..%lld; "
             "end-of-generation cuts %d; decode span %.4f s (the bootstrap's start to the last step's end; steps' "
             "own wall %.4f s): %.2f tokens/s (all tokens), %.2f tokens/s ((tokens - 1) / span: the server's "
             "predicted_per_second definition); closing check %s%s",
             s.ended.c_str(), s.steps, s.boot, s.spec, s.gen, s.tokens, s.spec ? double(s.spec_tokens) / s.spec : 0.0,
             s.spec ? double(s.accepted) / s.spec : 0.0, s.wall_median_ms, s.wall_p90_ms,
             s.spec ? double(s.waits_spec) / s.spec : 0.0,
             (s.boot + s.gen) ? double(s.waits_one) / (s.boot + s.gen) : 0.0, s.draft_median_ms, s.verify_median_ms,
             s.catchup_median_ms, s.one_median_ms, static_cast<long long>(s.n_before_min),
             static_cast<long long>(s.n_before_max), s.cuts, s.span_s, own_s, s.span_s > 0 ? s.tokens / s.span_s : 0.0,
             s.span_s > 0 ? (s.tokens - 1) / s.span_s : 0.0, check.c_str(), who.c_str());
    x.sum = s;
    x.sum_line = buf;
    fprintf(stderr, "%s\n", buf);
    // ME3: the request's one expert-cache line (its settings and the counters since its prefill's entry)
    x.cache_line.clear();
    if (ex_ != nullptr) {
        x.cache_line = ex_->request_json();
        // ME4: the target's graph launches and eager rounds of the request, its prompt (to decode-ready) and its decode
        const target::submissions end = x.t->submitted();
        const auto d = [](const target::submissions & a, const target::submissions & b) {
            char t[256];
            std::snprintf(t, sizeof(t),
                          "{\"target_graphs\": %lld, \"draft_graphs\": %lld, \"catchup_graphs\": %lld, "
                          "\"eager_rounds\": %lld, \"eager_draft_rounds\": %lld}",
                          (long long)(b.target_graphs - a.target_graphs), (long long)(b.draft_graphs - a.draft_graphs),
                          (long long)(b.catchup_graphs - a.catchup_graphs),
                          (long long)(b.eager_rounds - a.eager_rounds),
                          (long long)(b.eager_draft_rounds - a.eager_draft_rounds));
            return std::string(t);
        };
        if (!x.cache_line.empty() && x.cache_line.back() == '}') {
            x.cache_line.pop_back();
            x.cache_line += ", \"submissions\": {\"prompt\": " + d(x.sub0, x.sub_ready) +
                            ", \"decode\": " + d(x.sub_ready, end) + "}}";
        }
        if (!x.cache_line.empty()) {
            fprintf(stderr, "FNFL expert cache request: %s\n", x.cache_line.c_str());
        }
    }
    fflush(stderr);
    x.last_steps.swap(x.steps);
    x.steps.clear();
}

run_status engine::set_sampling(int32_t slot, const request_options & o) {
    if (!slot_ok(slot)) {
        return failed(run_status::bad_request, "slot " + std::to_string(slot) + ": FNFL serves " +
                                                   std::to_string(n_slots()) + " slot" + (n_slots() > 1 ? "s" : ""));
    }
    slot_rt & x = s_[size_t(slot)];
    if (x.job.live) {
        return failed(run_status::bad_request, "set_sampling: slot " + std::to_string(slot) + "'s read is in progress");
    }
    finish(x, "the next request");
    if (!x.t->set_sampling(o.sampling, err_)) {
        return failed(run_status::bad_request, err_);
    }
    x.req = o;
    return run_status::ok;
}

void engine::prefill_lines(slot_rt & x, const char * how, int64_t from, int32_t n_processed, int32_t n_reuse,
                           int32_t n_tokens, double secs, bool ok, clk::time_point entry) {
    const int64_t stopped = ok ? -1 : x.pf_stats.stopped;
    std::string chunks;
    for (const pf_chunk_stats & c : x.pf_stats.chunks) {
        chunks += (chunks.empty() ? "" : " ") + std::to_string(c.c.start) + "+" + std::to_string(c.c.rows) +
                  (c.checkpoint >= 0 ? "*" : "");
    }
    const std::string who = n_slots() > 1 ? "slot " + std::to_string(x.t->slot_index()) + ": " : std::string();
    const int32_t held_ck = n_slots() > 1 ? pf_->book().count_owner(x.t->slot_index()) : pf_->book().count();
    fprintf(
        stderr,
        "FNFL prefill: %s%s at %lld: %d tokens processed (%d kept by the shell, %d new; %.3f s); chunks (start+rows, "
        "* checkpoint): %s; checkpoints held %d%s\n",
        who.c_str(), how, static_cast<long long>(from), n_processed, n_reuse, n_tokens, secs, chunks.c_str(), held_ck,
        stopped >= 0 ? ("; stopped (cancelled) at " + std::to_string(stopped)).c_str() : "");
    // PB1 item 6: the call's phase line, once, from host markers at existing completion boundaries (no added event,
    // synchronization or per-chunk write). Steady-clock ms. The target's completion is not separately observable: the
    // target and the draft end at one join at each chunk's end, so it is marked unavailable. The draft's marker is the
    // last chunk's join; decode-ready is after the head's join and the handoff check (the bootstrap round follows).
    const auto ms = [](clk::time_point p) {
        return std::chrono::duration<double, std::milli>(p.time_since_epoch()).count();
    };
    const auto mark = [&](double sec) {
        if (sec < 0) {
            return std::string(x.pf_stats.bulk || !ok ? "not reached" : "unavailable (the forced path)");
        }
        const clk::time_point p =
            x.pf_stats.t0 + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(sec));
        char b[96];
        snprintf(b, sizeof b, "%.3f (+%.3f)", ms(p), std::chrono::duration<double, std::milli>(p - entry).count());
        return std::string(b);
    };
    char b[200];
    snprintf(b, sizeof b, "FNFL prefill phases (steady-clock ms, + from entry): %sentry %.3f; target unavailable",
             who.c_str(), ms(entry));
    x.phase_line = std::string(b) + " (it joins with the draft at each chunk's end); draft " +
                   mark(x.pf_stats.last_join_s) + " (the last chunk's join); decode-ready " + mark(x.pf_stats.ready_s) +
                   "; rows " + std::to_string(ok ? n_processed : 0) + "; restored checkpoint " +
                   (from > 0 && std::string(how) == "from a checkpoint" ? std::to_string(from) : "none") +
                   (x.pf_stats.bulk ? "" : "; the forced path");
    fprintf(stderr, "%s\n", x.phase_line.c_str());
    fflush(stderr);
}

run_status engine::prefill(int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                           int32_t & n_processed, const prefill_progress & progress) {
    n_processed = 0;
    if (n_slots() > 1) {
        // MU1: the slot's read job, every group in one step and every chunk in turn (nothing else runs meanwhile)
        const int32_t r = read_begin(slot, tokens, n_tokens, n_reuse, n_processed);
        if (r != 0) {
            return failed(run_status::bad_request, err_);
        }
        slot_rt & x = s_[size_t(slot)];
        while (true) {
            const read_status st = read_step(slot, fn::n_layer);
            if (st == read_status::done) {
                return run_status::ok;
            }
            if (st == read_status::cancelled) {
                n_processed = static_cast<int32_t>(x.keep.size());
                return run_status::cancelled;
            }
            if (st == read_status::error) {
                return run_status::error;
            }
            if (st == read_status::chunk_done && progress && !progress(x.job.from, x.job.committed, x.job.to)) {
                read_cancel(slot);
            }
        }
    }
    const auto entry = clk::now();
    slot_rt & x = s_[0];
    finish(x, "the next request");
    if (ex_ != nullptr) {
        ex_->request_begin();
    }
    target & t = *x.t;
    x.sub0 = t.submitted();
    x.sub_ready = x.sub0;
    const ledger & l = t.slot();
    const std::vector<int32_t> & held_tok = x.valid ? l.tok : x.keep;
    const int64_t held = x.valid ? l.n : static_cast<int64_t>(x.keep.size());
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
    const bool draft_ok =
        t.stage_front() || (t.draft_n() >= t.batch_start() && t.draft_n() <= l.n && l.n - t.draft_n() <= 4);
    int64_t from = 0;
    const char * how = "a pure extension";
    if (x.valid && n_reuse == l.n && draft_ok && !t.invalid()) {
        from = l.n;
        x.pf_stats = pf_run_stats();
        ok = extend_ledger(t, tokens, n_tokens, err_) && pf_->run(t, &x.pf_stats, err_, progress);
    } else {
        std::vector<int32_t> all(held_tok.begin(), held_tok.begin() + n_reuse);
        all.insert(all.end(), tokens, tokens + n_tokens);
        x.pf_stats = pf_run_stats();
        ok = pf_->begin(t, all, n_reuse, from, err_) && pf_->run(t, &x.pf_stats, err_, progress);
        how = from > 0 ? "from a checkpoint" : "from token 0";
    }
    n_processed = static_cast<int32_t>(int64_t(n_reuse) + n_tokens - from);
    x.sub_ready = t.submitted();
    const int64_t stopped = ok ? -1 : x.pf_stats.stopped;
    prefill_lines(x, how, from, n_processed, n_reuse, n_tokens, since(t0), ok, entry);
    x.valid = ok;
    x.keep.clear();
    x.n_gen = 0;
    x.steps.clear();
    if (stopped >= 0) {
        // BG1: the chunks before the stop and their checkpoints are kept; the next prefill restores the latest
        // checkpoint at or below the kept prompt (the stopped run's handoff never happened, so there is no valid state)
        const ledger & lk = t.slot();
        x.keep.assign(lk.tok.begin(), lk.tok.begin() + std::min<int64_t>(stopped, int64_t(lk.tok.size())));
        n_processed = static_cast<int32_t>(x.keep.size());
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
    for (slot_rt & x : s_) {
        finish(x, "the draft setting");
    }
    if (d < 1 || d > max_spec_d) {
        return failed(run_status::bad_request, "set_draft: d " + std::to_string(d) + " outside 1 .. 4");
    }
    for (slot_rt & x : s_) {
        // TTTS1: a split's front holds no draft (its form is the back's); it keeps the d alone
        if (!x.t->stage_front() && !x.t->set_draft_sparse(sparse, err_)) {
            return failed(run_status::bad_request, "set_draft: " + err_);
        }
    }
    c_.spec_d = d;
    return run_status::ok;
}

run_status engine::save_ready(ready_state & s) {
    slot_rt & x = s_[0];
    if (!x.valid || x.open || !x.steps.empty()) {
        return failed(run_status::bad_request, "save_ready: no prefill handoff (a prefill, then no step)");
    }
    return x.t->save_ready(s, err_) ? run_status::ok : failed(run_status::error, "save_ready: " + err_);
}

run_status engine::restore_ready(const ready_state & s) {
    slot_rt & x = s_[0];
    finish(x, "the next request");
    if (!x.t->restore_ready(s, err_)) {
        const std::string what = "restore_ready: " + err_;
        lost(x);
        return failed(run_status::error, what);
    }
    x.valid = true;
    x.keep.clear();
    x.n_gen = 0;
    x.steps.clear();
    return run_status::ok;
}

// a step's checks and the slot's cells before its rounds (b: its token budget)
run_status engine::step_enter(int32_t slot, const int32_t * out, int32_t & b) {
    if (!slot_ok(slot) || out == nullptr) {
        return failed(run_status::bad_request, "step: slot " + std::to_string(slot));
    }
    slot_rt & x = s_[size_t(slot)];
    if (!x.valid || x.job.live) {
        return failed(run_status::bad_request, "step: the slot holds no processed prompt (prefill first)");
    }
    if (pf_ != nullptr && pf_->loan_job() >= 0) {
        // MXMU1: a suspended cached chunk holds the prompt loan (only under test_partial_reads); a decode then would
        // run against the borrowed zone
        return failed(run_status::bad_request, "step: slot " + std::to_string(slot) +
                                                   ": a read's chunk holds the prompt loan (its turn ends first)");
    }
    if (x.fail_step > 0 && static_cast<int32_t>(x.steps.size()) + 1 == x.fail_step) {
        // TEST (SV1): the failure a round's error returns, before the step's rounds (FC-RD2 leaves the pre-step
        // state); the request ends with this message and the next one resets the slot and re-prefills
        const int32_t k = x.fail_step;
        x.fail_step = 0;
        lost(x);
        x.open = false;
        x.unchecked = false;
        x.steps.clear();
        return failed(run_status::error, "step " + std::to_string(k) +
                                             ": injected failure (SV1 test switch --fnfl-test-fail-step); the next "
                                             "request resets the slot and re-prefills");
    }
    b = budget(x);
    if (b < 1) {
        if (!x.steps.empty()) {
            return failed(run_status::bad_request, "step: no token budget left (max_tokens or the context)");
        }
        b = 1; // a request's first step (the bootstrap) always returns its token
    }
    if (m_->pool != nullptr && int64_t(x.t->slot().tok.size()) + step_cells > x.cells) {
        // MU1: the step's cells mapped before its rounds (the pool's growth, reclaiming idle slots when full); FMU1:
        // only when the slot's mapped cells (cached) do not cover them
        bool capacity = false;
        if (!ensure_cells(slot, int64_t(x.t->slot().tok.size()) + step_cells, capacity, err_)) {
            return failed(capacity ? run_status::no_context : run_status::error, "step: " + err_);
        }
    }
    if (x.steps.empty()) {
        x.t_first = clk::now();
    }
    return run_status::ok;
}

run_status engine::step(int32_t slot, int32_t * out, int32_t & n_out, bool & end) {
    n_out = 0;
    end = false;
    int32_t b = 0;
    const run_status rs = step_enter(slot, out, b);
    if (rs != run_status::ok) {
        return rs;
    }
    step_info s;
    const bool ran = runtime::step(*s_[size_t(slot)].t, b, c_.spec_d, c_.eog, s, err_);
    return step_leave(slot, ran, s, out, n_out, end);
}

run_status engine::step_leave(int32_t slot, bool ran, const step_info & s, int32_t * out, int32_t & n_out, bool & end) {
    slot_rt & x = s_[size_t(slot)];
    if (!ran) {
        const std::string what = "step: " + err_;
        lost(x);
        x.open = false;
        x.unchecked = false;
        x.steps.clear();
        return failed(run_status::error, what);
    }
    x.t_span_s = since(x.t_first);
    x.open = true;
    x.unchecked = true;
    x.n_gen += s.n_out;
    x.steps.push_back(s);
    if (c_.record_hash && !x.t->stage_front()) {
        // TTTS1: the step's decision hash (spec run's records; a front stage's comes from the back)
        uint64_t    h = 0;
        std::string e;
        if (x.t->decision_hash(h, e, x.round_bias)) {
            x.steps.back().hash = h;
        }
    }
    for (int32_t j = 0; j < s.n_out; ++j) {
        out[j] = s.out[j];
    }
    n_out = s.n_out;
    end = s.end;
    if (c_.log_steps) {
        fprintf(stderr, "FNFL step %zu: %skind %s N %lld d %d a %d tokens", x.steps.size(),
                n_slots() > 1 ? ("slot " + std::to_string(slot) + " ").c_str() : "",
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
        finish(x, "eog");
    } else if (budget(x) < 1) {
        finish(x, "budget");
    }
    return run_status::ok;
}

void engine::step_joint(const int32_t * slots, int32_t n, joint_step_out * res) {
    for (int32_t k = 0; k < n; ++k) {
        res[k] = joint_step_out();
    }
    if (slots == nullptr || n < 1 || n > fn::xchg::max_lanes) {
        for (int32_t k = 0; k < n; ++k) {
            res[k].st = failed(run_status::bad_request, "step_joint: 1 .. 4 slots");
            res[k].why = err_;
        }
        return;
    }
    // the slots whose step can be a joint round's lane: a speculative step at d = 4 on the frame's executor
    std::vector<int32_t> mem, alone; // indices into slots
    int32_t b[fn::xchg::max_lanes] = {};
    for (int32_t k = 0; k < n; ++k) {
        bool dup = false;
        for (int32_t q = 0; q < k; ++q) {
            dup = dup || slots[q] == slots[k];
        }
        if (dup) {
            res[k].st = failed(run_status::bad_request, "step_joint: slot " + std::to_string(slots[k]) + " twice");
            res[k].why = err_;
            continue;
        }
        res[k].st = step_enter(slots[k], res[k].out, b[k]);
        if (res[k].st != run_status::ok) {
            res[k].why = err_;
            continue;
        }
        const target & t = *s_[size_t(slots[k])].t;
        const bool     lane = joint_ != nullptr && !joint_off_ && t.graph_executor() != joint_->eager() &&
                              t.slot().u != 0 && b[k] >= 2 &&
                              std::min(c_.spec_d, b[k] - 1) == c_.spec_d;  // FMS13: the round's full d, any d in 1 .. 4
        (lane ? mem : alone).push_back(k);
    }
    if (mem.size() < 2) {
        alone.insert(alone.end(), mem.begin(), mem.end());
        mem.clear();
    }
    const auto leave = [&](int32_t k, bool ran, const step_info & s) {
        res[k].st = step_leave(slots[k], ran, s, res[k].out, res[k].n_out, res[k].end);
        res[k].why = res[k].st == run_status::ok ? std::string() : err_;
    };
    if (!mem.empty()) {
        // the members' draft steps (one joint draft step when they fit, FMU2 build item 2; else each alone), then one
        // joint verify round, then each member's cut and catch-up
        const auto t0 = clk::now();
        step_info s[fn::xchg::max_lanes];
        std::vector<int32_t> go;
        target * dt[fn::xchg::max_lanes] = {};
        for (size_t i = 0; i < mem.size(); ++i) {
            dt[i] = s_[size_t(slots[mem[i]])].t.get();
            s[mem[i]].n_before = dt[i]->slot().n;
        }
        const int32_t nm = static_cast<int32_t>(mem.size());
        bool drafted = false;
        if (joint_frame::draft_fits(dt, nm)) {
            const int32_t dd = c_.spec_d;  // every member's min(spec_d, budget - 1) (FMS13: any d)
            round_result rr[fn::xchg::max_lanes];
            bool dok[fn::xchg::max_lanes] = {};
            std::string dwhy[fn::xchg::max_lanes], derr;
            drafted = joint_->draft(dt, nm, dd, rr, dok, dwhy, derr);
            if (!drafted) {
                fprintf(stderr, "FNFL joint: %s; these slots draft alone\n", derr.c_str());
            }
            for (int32_t i = 0; i < nm && drafted; ++i) {
                const int32_t k = mem[size_t(i)];
                if (!dok[i]) {
                    err_ = dwhy[i];
                    leave(k, false, s[k]);
                    continue;
                }
                s[k].to_draft_s = after(t0, rr[i].t_done_s);
                s[k].kind = 2;
                s[k].d = dd;
                for (int32_t j = 0; j < dd; ++j) {
                    s[k].prop[j] = rr[i].argmax[0][j];
                }
                go.push_back(k);
            }
        }
        for (int32_t i = 0; i < nm && !drafted; ++i) {
            const int32_t k = mem[size_t(i)];
            if (!step_propose(*dt[i], b[k], c_.spec_d, t0, s[k], err_)) {
                leave(k, false, s[k]);
                continue;
            }
            go.push_back(k);
        }
        const int32_t g = static_cast<int32_t>(go.size());
        target * tt[fn::xchg::max_lanes] = {};
        int32_t d[fn::xchg::max_lanes] = {}, bud[fn::xchg::max_lanes] = {};
        const int32_t * prop[fn::xchg::max_lanes] = {};
        round_result rv[fn::xchg::max_lanes];
        bool ok[fn::xchg::max_lanes] = {};
        std::string why[fn::xchg::max_lanes];
        for (int32_t i = 0; i < g; ++i) {
            tt[i] = s_[size_t(slots[go[size_t(i)]])].t.get();
            d[i] = s[go[size_t(i)]].d;
            bud[i] = b[go[size_t(i)]];
            prop[i] = s[go[size_t(i)]].prop;
        }
        bool joint = false, staged = false;
        std::string jerr;
        if (g >= 2) {
            joint = joint_->verify(tt, g, d, prop, bud, rv, ok, why, staged, jerr);
            if (!joint && !staged) {
                // the round did not start and no member changed: they verify alone; a failed capture (memory) turns
                // the joint round off for the engine's life
                joint_off_ = joint_off_ || joint_->capture_failed();
                fprintf(stderr, "FNFL joint: %s; these slots verify alone%s\n", jerr.c_str(),
                        joint_->capture_failed() ? " (joint decode off from now on)" : "");
            }
        }
        // each member's tokens and cut; then the catch-ups (FMS2 build item 2: one joint catch-up when two or more
        // members reach theirs, else each alone); then each member's step closed
        target * ct[fn::xchg::max_lanes] = {};
        int32_t crows[fn::xchg::max_lanes] = {}, cm[fn::xchg::max_lanes] = {};
        int32_t nc = 0;
        for (int32_t i = 0; i < g; ++i) {
            const int32_t k = go[size_t(i)];
            target & t = *tt[i];
            res[k].joint = joint;
            s[k].joint = joint;
            bool cut = false;
            if (joint) {
                cut = ok[i] && step_cut(t, c_.eog, rv[i], t0, s[k], err_);
                if (!ok[i]) {
                    err_ = why[i];
                }
            } else if (staged) {
                err_ = "joint round: " + jerr;
            } else {
                round_result r1;
                cut = t.verify(d[i], prop[i], bud[i], r1, err_) && step_cut(t, c_.eog, r1, t0, s[k], err_);
            }
            if (!cut) {
                leave(k, false, s[k]);
                continue;
            }
            ct[nc] = &t;
            crows[nc] = s[k].n_out;
            cm[nc] = k;
            ++nc;
        }
        round_result cr[fn::xchg::max_lanes];
        bool cok[fn::xchg::max_lanes] = {};
        std::string cwhy[fn::xchg::max_lanes], cerr;
        bool caught = false;
        if (nc >= 2 && !joint_off_ && joint_frame::catchup_fits(ct, nc)) {
            caught = joint_->catchup(ct, nc, crows, cr, cok, cwhy, cerr);
            if (!caught) {
                joint_off_ = joint_off_ || joint_->capture_failed();
                fprintf(stderr, "FNFL joint: %s; these slots catch up alone%s\n", cerr.c_str(),
                        joint_->capture_failed() ? " (joint decode off from now on)" : "");
            }
        }
        for (int32_t i = 0; i < nc; ++i) {
            const int32_t k = cm[i];
            bool ran = false;
            if (caught) {
                ran = cok[i];
                if (!ran) {
                    err_ = cwhy[i];
                }
            } else {
                ran = ct[i]->draft_catchup(crows[i], cr[i], err_);
            }
            if (ran) {
                step_caught(c_.eog, cr[i], t0, s[k]);
            }
            leave(k, ran, s[k]);
        }
    }
    for (int32_t k : alone) {
        step_info s;
        const bool ran = runtime::step(*s_[size_t(slots[k])].t, b[k], c_.spec_d, c_.eog, s, err_);
        leave(k, ran, s);
    }
}

// ---- TTTS1: a split stage's steps ----

run_status engine::split_enter(int32_t slot, int32_t & b) {
    int32_t out[max_step_tokens] = {};
    return step_enter(slot, out, b);
}

run_status engine::split_leave(int32_t           slot,
                               bool              ran,
                               const step_info & s,
                               int32_t *         out,
                               int32_t &         n_out,
                               bool &            end) {
    return step_leave(slot, ran, s, out, n_out, end);
}

uint64_t engine::split_last_hash(int32_t slot) const {
    if (!slot_ok(slot)) {
        return 0;
    }
    const slot_rt & x = s_[size_t(slot)];
    return !x.steps.empty() ? x.steps.back().hash : !x.last_steps.empty() ? x.last_steps.back().hash : 0;
}

int32_t engine::split_budget(int32_t slot) const {
    if (!slot_ok(slot)) {
        return -1;
    }
    const slot_rt & x = s_[size_t(slot)];
    const int32_t   b = budget(x);
    return b < 1 && x.steps.empty() ? 1 : b;  // a request's first step (the bootstrap) always returns its token
}

run_status engine::truncate(int32_t slot, int32_t n_keep) {
    if (!slot_ok(slot) || n_keep < 0) {
        return failed(run_status::bad_request, "truncate: slot " + std::to_string(slot));
    }
    slot_rt & x = s_[size_t(slot)];
    if (x.job.live) {
        return failed(run_status::bad_request, "truncate: slot " + std::to_string(slot) + "'s read is in progress");
    }
    finish(x, "the shell (a stop string, its limit or a cancel)");
    if (!x.valid) {
        if (static_cast<size_t>(n_keep) < x.keep.size()) {
            x.keep.resize(static_cast<size_t>(n_keep));
        }
        return run_status::ok;
    }
    target & t = *x.t;
    const ledger & l = t.slot();
    if (n_keep >= l.n) {
        // the processed tokens stay; the pending token goes (the next prefill brings it back as a prompt token)
        if (!t.keep_processed(err_)) {
            lost(x);
        }
        return run_status::ok;
    }
    std::vector<int32_t> kept(l.tok.begin(), l.tok.begin() + n_keep);
    const step_record & last = t.versions().last();
    if (last.valid && !last.prompt && n_keep >= last.n_before) {
        // within the last step: its version (SR1), then the draft catches up the kept rows (PB1: not within a bulk
        // prompt chunk, whose versions are half ends: the restore below)
        const int32_t j = static_cast<int32_t>(n_keep - last.n_before);
        const int64_t nk = static_cast<int64_t>(l.tok.size()) > n_keep ? int64_t(n_keep) + 1 : int64_t(n_keep);
        bool ok = t.cut(j, nk, err_) && t.keep_processed(err_);
        // TTTS1: a split's front holds no draft (the back's catches up there), so its cut alone keeps the rows
        const bool    draftless = t.stage_front();
        if (ok && !draftless && t.draft_n() >= 0 && t.draft_n() < l.n) {
            round_result r;
            ok = t.draft_catchup(static_cast<int32_t>(l.n - t.draft_n()), r, err_);
        }
        if (ok && (draftless || t.draft_n() == l.n) && l.n == n_keep) {
            return run_status::ok;
        }
        fprintf(stderr, "FNFL: truncate to %d within the last step: %s; restore at the next prefill instead\n", n_keep,
                ok ? "the draft did not follow" : err_.c_str());
        if (!ok) {
            // a failed write: the state is unknown, the next prefill re-prefills from token 0
            if (n_slots() > 1) {
                pf_->drop_owner(slot);
            } else {
                pf_->drop_all();
            }
        }
    }
    // deeper: the next prefill restores the latest checkpoint at or before the kept length and re-prefills from it
    // (PF1); the state below every kept checkpoint is unchanged by the cut
    x.valid = false;
    x.keep = std::move(kept);
    return run_status::ok;
}

bool engine::in_place(int32_t slot) const {
    return slot_ok(slot) && s_[size_t(slot)].valid;
}

void engine::to_restore(int32_t slot) {
    if (!in_place(slot)) {
        return;
    }
    slot_rt &      x = s_[size_t(slot)];
    const ledger & l = x.t->slot();
    x.keep.assign(l.tok.begin(), l.tok.begin() + l.n);
    x.valid = false;
}

run_status engine::history(int32_t slot, std::vector<int32_t> & tokens, int32_t & pending) const {
    pending = -1;
    if (!slot_ok(slot)) {
        return run_status::bad_request;
    }
    const slot_rt & x = s_[size_t(slot)];
    if (!x.valid) {
        tokens = x.keep;
        return run_status::ok;
    }
    const ledger & l = x.t->slot();
    tokens.assign(l.tok.begin(), l.tok.begin() + l.n);
    if (l.u == 1 && l.pending >= 0) {
        pending = l.pending;
    }
    return run_status::ok;
}

// ---- MU1: the multi-slot engine ----

int32_t engine::lane_rows(int32_t lane) const {
    return n_slots() > 1 && pf_ != nullptr ? pf_->lane_rows(lane) : 0;
}

int32_t engine::lane_owner(int32_t lane) const {
    return pf_ != nullptr ? pf_->lane_owner(lane) : -1;
}

bool engine::set_read_group(int32_t layers) {
    if (layers < 1 || layers > fn::n_layer) {
        return false;
    }
    read_group_ = layers;
    return true;
}

int64_t engine::held(int32_t slot) const {
    if (!slot_ok(slot)) {
        return 0;
    }
    const slot_rt & x = s_[size_t(slot)];
    return x.valid ? x.t->slot().n : static_cast<int64_t>(x.keep.size());
}

bool engine::get_pool_info(pool_info & out) const {
    out = pool_info();
    if (m_ == nullptr || m_->pool == nullptr) {
        return false;
    }
    const model::kv_pages & p = m_->pool->pages();
    out.on = true;
    out.pages = p.n_pages();
    out.page_cells = m_->pool->page_cells();
    out.free_pages = p.free_pages();
    out.capacity_cells = m_->pool->capacity_cells();
    out.reclaims = reclaims_;
    for (int32_t i = 0; i < n_slots() && i < max_split_slots; ++i) {
        out.mapped[i] = p.mapped(i);
        out.busy[i] = s_[size_t(i)].busy || s_[size_t(i)].job.live;
    }
    return true;
}

run_status engine::release_slot(int32_t slot) {
    if (!slot_ok(slot)) {
        return failed(run_status::bad_request, "release: slot " + std::to_string(slot));
    }
    slot_rt & x = s_[size_t(slot)];
    if (x.job.live) {
        return failed(run_status::bad_request, "release: slot " + std::to_string(slot) + "'s read is in progress");
    }
    finish(x, "the shell (the slot was released)");
    x.busy = false;
    x.released = ++release_seq_;
    return run_status::ok;
}

void engine::reclaim(int32_t slot, int32_t for_slot) {
    slot_rt & x = s_[size_t(slot)];
    const int64_t had = held(slot);
    const int32_t pages = m_->pool->pages().mapped(slot);
    const int32_t ck = pf_->book().count_owner(slot);
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])); // every consumer of its pages completed
    }
    std::string e;
    if (!m_->pool->trim(slot, 0, e)) {
        fprintf(stderr, "FNFL KV pool: reclaiming slot %d: %s\n", slot, e.c_str());
    }
    x.cells = m_->pool->mapped_cells(slot);
    finish(x, "the KV pool (the slot was reclaimed)");
    x.valid = false;
    x.keep.clear();
    pf_->drop_owner(slot);
    ++reclaims_;
    fprintf(stderr,
            "FNFL KV pool: reclaimed idle slot %d for slot %d: %d pages, its %lld held tokens and %d checkpoints "
            "dropped (its prefix and checkpoint lineage invalidated); %d of %d pages free\n",
            slot, for_slot, pages, static_cast<long long>(had), ck, m_->pool->pages().free_pages(),
            m_->pool->pages().n_pages());
    fflush(stderr);
}

bool engine::ensure_cells(int32_t slot, int64_t cells, bool & capacity, std::string & err) {
    capacity = false;
    while (true) {
        bool full = false;
        if (m_->pool->ensure(slot, cells, err, &full)) {
            s_[size_t(slot)].cells = m_->pool->mapped_cells(slot);
            return true;
        }
        if (!full) {
            return false;
        }
        // the pool is full: the least recently released idle slot holding pages is reclaimed, then retried; an active
        // slot (a request or a read on it) is never reclaimed
        std::vector<model::kv_slot_view> v(static_cast<size_t>(n_slots()));
        for (int32_t i = 0; i < n_slots(); ++i) {
            const slot_rt & y = s_[size_t(i)];
            v[size_t(i)].active = y.busy || y.job.live;
            v[size_t(i)].pages = m_->pool->pages().mapped(i);
            v[size_t(i)].released = y.released;
        }
        const int32_t victim = model::kv_reclaim_victim(v, slot);
        if (victim < 0) {
            capacity = true;
            return false;
        }
        reclaim(victim, slot);
    }
}

int32_t engine::read_begin(int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                           int32_t & n_processed) {
    const auto entry = clk::now();
    n_processed = 0;
    if (n_slots() < 2 || !slot_ok(slot)) {
        failed(run_status::bad_request, "read_begin: slot " + std::to_string(slot) + " (a multi-slot engine's)");
        return -1;
    }
    slot_rt & x = s_[size_t(slot)];
    if (x.job.live) {
        failed(run_status::bad_request, "read_begin: slot " + std::to_string(slot) + "'s read is in progress");
        return -1;
    }
    target & t = *x.t;
    const ledger & l = t.slot();
    const std::vector<int32_t> & held_tok = x.valid ? l.tok : x.keep;
    const int64_t held = x.valid ? l.n : static_cast<int64_t>(x.keep.size());
    if (n_tokens < 0 || (n_tokens > 0 && tokens == nullptr) || n_reuse < 0 || n_reuse > held ||
        (n_tokens == 0 && n_reuse < 1)) {
        failed(run_status::bad_request, "read_begin: slot " + std::to_string(slot) + ", " + std::to_string(n_tokens) +
                                            " tokens after " + std::to_string(n_reuse) + " kept of " +
                                            std::to_string(held) + " held");
        return -1;
    }
    if (int64_t(n_reuse) + n_tokens + 1 > c_.n_ctx_slot) {
        failed(run_status::bad_request, "read_begin: slot " + std::to_string(slot) + ": " +
                                            std::to_string(int64_t(n_reuse) + n_tokens) + " tokens exceed the " +
                                            std::to_string(c_.n_ctx_slot) + "-cell context");
        return 1;
    }
    // the pool's pages for the whole prompt and its next step, before anything changes on the slot (the rejection
    // leaves the slot's state intact); an idle slot is reclaimed when the pool is full
    const int64_t need = int64_t(n_reuse) + n_tokens + step_cells;
    bool capacity = false;
    if (m_->pool != nullptr && !ensure_cells(slot, need, capacity, err_)) { // MXMU1: one card's slots own their KV
        failed(capacity ? run_status::no_context : run_status::error, "read_begin: " + err_);
        return capacity ? 1 : -1;
    }
    finish(x, "the next request");
    if (ex_ != nullptr) {
        ex_->request_begin(); // MXMU1: the cache's request line counts from here (one service: every slot's work)
    }
    x.busy = true;
    x.sub0 = t.submitted();
    x.sub_ready = x.sub0;
    // FC-PF1: nothing new: keep n_reuse - 1 tokens and process the last kept one again, so its logits exist
    int32_t one = 0;
    if (n_tokens == 0) {
        one = held_tok[size_t(n_reuse - 1)];
        tokens = &one;
        n_tokens = 1;
        --n_reuse;
    }
    const bool draft_ok =
        t.stage_front() || (t.draft_n() >= t.batch_start() && t.draft_n() <= l.n && l.n - t.draft_n() <= 4);
    int64_t from = 0;
    bool ok = false;
    const char * how = "a pure extension";
    if (x.valid && n_reuse == l.n && draft_ok && !t.invalid()) {
        from = l.n;
        ok = extend_ledger(t, tokens, n_tokens, err_);
    } else {
        std::vector<int32_t> all(held_tok.begin(), held_tok.begin() + n_reuse);
        all.insert(all.end(), tokens, tokens + n_tokens);
        ok = pf_->begin(t, all, n_reuse, from, err_);
        how = from > 0 ? "from a checkpoint" : "from token 0";
    }
    x.valid = false;
    x.keep.clear();
    x.n_gen = 0;
    x.steps.clear();
    if (!ok) {
        pf_->drop_owner(slot);
        failed(run_status::error, "read_begin: " + err_);
        return -1;
    }
    // the slot's pages past the read's needs go back to the pool (its old cells there are no longer valid)
    if (m_->pool != nullptr &&
        m_->pool->pages().mapped(slot) >
            model::kv_pages::pages_for(need, m_->pool->page_cells(), m_->pool->pages().n_pages())) {
        for (int32_t c = 0; c < 2; ++c) {
            cudaSetDevice(m_->device[c]);
            cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c]));
        }
        std::string e;
        if (!m_->pool->trim(slot, need, e)) {
            fprintf(stderr, "FNFL KV pool: trimming slot %d: %s\n", slot, e.c_str());
        }
        x.cells = m_->pool->mapped_cells(slot);
    }
    pf_job & j = x.job;
    if (resident_tp()) {
        pf_->forget_job(j.id); // FMU1: the staging thread may still read the previous job's tokens (P)
    }
    j = pf_job();
    j.live = true;
    j.id = next_job_++;
    j.slot = slot;
    j.tok = t.slot().tok;
    j.from = t.slot().n;
    j.to = static_cast<int64_t>(j.tok.size());
    j.committed = j.from;
    // MXMU1: the plan as the one-slot prefill makes it (MX3's cached split; ME3's short read with the expert service)
    pf_->job_plan(j.from, j.to, ex_ != nullptr, j.plan, j.short_read);
    j.stats = pf_run_stats();
    j.stats.from = j.from;
    j.stats.to = j.to;
    j.stats.bulk = true;
    j.stats.t0 = clk::now();
    x.job_entry = entry;
    x.job_how = how;
    x.job_reuse = n_reuse;
    x.job_new = n_tokens;
    x.pf_stats = pf_run_stats();
    n_processed = static_cast<int32_t>(int64_t(n_reuse) + n_tokens - from);
    if (j.plan.empty()) {
        j.live = false;
        failed(run_status::error, "read_begin: no rows to read");
        return -1;
    }
    return 0;
}

bool engine::resident_tp() const {
    return n_slots() > 1 && pf_ != nullptr && m_->pool != nullptr && !m_->single && !m_->kv_host && ex_ == nullptr;
}

bool engine::read_prefetch(int32_t now_slot, int32_t next_slot) {
    if (!resident_tp()) {
        return true;
    }
    const pf_job * now = slot_ok(now_slot) && s_[size_t(now_slot)].job.live ? &s_[size_t(now_slot)].job : nullptr;
    const pf_job * next = slot_ok(next_slot) && s_[size_t(next_slot)].job.live ? &s_[size_t(next_slot)].job : nullptr;
    target & t = *s_[size_t(next != nullptr ? next_slot : 0)].t;
    if (!pf_->prefetch(t, now, next, err_)) {
        failed(run_status::error, "read prefetch: " + err_);
        return false;
    }
    return true;
}

read_status engine::read_step(int32_t slot, int32_t n_groups, read_mode mode) {
    if (!slot_ok(slot) || !s_[size_t(slot)].job.live) {
        failed(run_status::bad_request, "read_step: slot " + std::to_string(slot) + " has no read");
        return read_status::error;
    }
    slot_rt & x = s_[size_t(slot)];
    pf_job & j = x.job;
    // MXMU1: with the expert service a step runs the whole chunk, so its loan, ring and lent hold are returned before
    // another slot's work (TEST: test_partial_reads steps groups, for the checks of a suspended cached chunk)
    const int32_t groups = ex_ != nullptr && !test_partial_ ? fn::n_layer : n_groups;
    const pf_mode pm = mode == read_mode::lone      ? pf_mode::lone
                       : mode == read_mode::writers ? pf_mode::writers
                                                    : pf_mode::shared;
    const pf_step st = pf_->job_step(*x.t, j, groups, read_group_, err_, pm);
    const auto end_lines = [&](bool ok, int64_t stopped) {
        x.pf_stats = j.stats;
        x.pf_stats.stopped = stopped;
        x.pf_stats.last_join_s = ok ? j.service_s : -1;
        x.pf_stats.ready_s = ok ? std::chrono::duration<double>(clk::now() - j.stats.t0).count() : -1;
        prefill_lines(x, x.job_how, j.from, static_cast<int32_t>(j.to - j.from), x.job_reuse, x.job_new,
                      std::chrono::duration<double>(clk::now() - j.stats.t0).count(), ok, x.job_entry);
        fprintf(stderr,
                "FNFL read job: slot %d, rows %lld (from %lld), chunks %d, groups %lld of %d layers, steps %d, service "
                "ms %.1f, longest step ms %.1f, checkpoints %d, lane %s\n",
                slot, static_cast<long long>(j.to - j.from), static_cast<long long>(j.from), j.chunks_done,
                static_cast<long long>(j.groups_done), j.layers, j.steps, j.service_s * 1e3, j.longest_step_s * 1e3,
                j.checkpoints,
                ok             ? "done"
                : stopped >= 0 ? "cancelled"
                               : "failed");
        fflush(stderr);
    };
    switch (st) {
        case pf_step::yielded:
            return read_status::yielded;
        case pf_step::chunk_done:
            return read_status::chunk_done;
        case pf_step::done:
            x.sub_ready = x.t->submitted();
            end_lines(true, -1);
            j.live = false;
            x.valid = true;
            x.keep.clear();
            return read_status::done;
        case pf_step::cancelled: {
            // BG1: the chunks before the stop and their checkpoints are kept; the next read restores the latest
            // checkpoint at or below the kept prompt (no handoff happened, so there is no valid state)
            const ledger & l = x.t->slot();
            x.keep.assign(l.tok.begin(), l.tok.begin() + std::min<int64_t>(j.committed, int64_t(l.tok.size())));
            end_lines(false, j.committed);
            j.live = false;
            x.valid = false;
            err_ = "read cancelled at " + std::to_string(j.committed);
            return read_status::cancelled;
        }
        case pf_step::error:
        default: {
            const std::string what = "read: " + err_;
            end_lines(false, -1);
            j.live = false;
            x.valid = false;
            x.keep.clear();
            pf_->drop_owner(slot);
            failed(run_status::error, what);
            return read_status::error;
        }
    }
}

void engine::test_abandon_read(int32_t slot) {
    if (!slot_ok(slot) || !s_[size_t(slot)].job.live) {
        return;
    }
    slot_rt & x = s_[size_t(slot)];
    pf_->job_abandon(*x.t, x.job);
    x.job.live = false;
    x.valid = false;
    x.keep.clear();
    pf_->drop_owner(slot);
}

void engine::read_cancel(int32_t slot) {
    if (slot_ok(slot) && s_[size_t(slot)].job.live) {
        s_[size_t(slot)].job.cancel = true;
    }
}

bool engine::get_read_info(int32_t slot, read_info & out) const {
    out = read_info();
    if (!slot_ok(slot)) {
        return false;
    }
    const pf_job & j = s_[size_t(slot)].job;
    out.slot = slot;
    out.job = j.id;
    out.live = j.live;
    out.cancel = j.cancel;
    out.total = j.to;
    out.from = j.from;
    out.committed = j.committed;
    out.chunk = static_cast<int32_t>(j.chunk);
    out.chunks = static_cast<int32_t>(j.plan.size());
    out.next_rows = j.chunk < j.plan.size() ? j.plan[j.chunk].rows : 0;
    out.in_chunk = j.group >= 0;
    out.lane = j.lane;
    out.group = j.group;
    out.groups = j.group >= 0 ? j.groups : (fn::n_layer + read_group_ - 1) / read_group_;
    out.layers = j.group >= 0 || j.chunks_done > 0 ? j.layers : read_group_; // the chunk's own (fixed at its start)
    out.groups_done = j.groups_done;
    out.steps = j.steps;
    out.chunks_done = j.chunks_done;
    out.checkpoints = j.checkpoints;
    out.service_s = j.service_s;
    out.longest_step_s = j.longest_step_s;
    out.last_group_s = j.last_group_s;
    return true;
}

} // namespace fn::runtime
