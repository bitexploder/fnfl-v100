#include "fn/runtime/target.h"

#include "fn/kernels/draft/draft.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/head/head.h"
#include "fn/model/layer.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

constexpr int32_t E = ple_input; // 2560 PLE input values a row
using clock_t_ = std::chrono::steady_clock;

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

double seconds(clock_t_::time_point a, clock_t_::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int32_t part_of(const model::reserved_class * rc, const std::string & name) {
    for (int32_t i = 0; rc != nullptr && i < (int32_t)rc->plan.parts.size(); ++i) {
        if (rc->plan.parts[i].name == name) {
            return i;
        }
    }
    return -1;
}

fn::samp::path sp_path(const fn::samp::options & o) {
    return fn::samp::path_of(o);
}

// SV1: the decision variant of the options: 0 without the sampler or on its greedy path, 1 on its other paths
int32_t variant_of(const target_options & o) {
    return !o.sample || sp_path(o.sampling) == fn::samp::path::greedy ? 0 : 1;
}

constexpr int32_t step_kind = static_cast<int32_t>(fn::round::kind::draft_step);
constexpr int32_t max_b = fn::round::max_rows; // SP2: rows a round, 1..5 (graphs per B, the descriptor's rows)
constexpr int32_t catchup_kind = static_cast<int32_t>(fn::round::kind::draft_catchup);

} // namespace

bool check_draft_options(const draft_options & o, std::string & err) {
    if (!o.on) {
        return true;
    }
    if (o.attention.empty()) {
        return fail(err, "draft attention: no value (dense | sparse; fresh and reuse are not built)");
    }
    if (o.attention == "fresh" || o.attention == "reuse") {
        return fail(err, "draft attention " + o.attention + ": QA-2 deferred, not built (DR1; dense or SP2's sparse)");
    }
    if (o.attention != "dense" && o.attention != "sparse") {
        return fail(err, "draft attention " + o.attention + ": not dense or sparse (SP2's switch)");
    }
    if (o.subset.empty()) {
        return fail(err, "draft: no subset file");
    }
    return true;
}

target::~target() {
    for (int32_t c = 0; c < 2; ++c) {
        if (m_ == nullptr) {
            break;
        }
        cudaSetDevice(m_->device[c]);
        if (st_[c] != nullptr) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c]));
        }
        for (int32_t v = 0; v < 2; ++v) {
            for (int32_t b = 0; b <= max_b; ++b) {
                if (exec_[v][b][c] != nullptr) {
                    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_[v][b][c]));
                }
                if (graph_[v][b][c] != nullptr) {
                    cudaGraphDestroy(static_cast<cudaGraph_t>(graph_[v][b][c]));
                }
            }
            for (int32_t f = 0; f < 2; ++f) {
                for (int32_t k = 0; k < 2; ++k) {
                    for (int32_t b = 0; b <= max_b; ++b) {
                        if (dexec_[f][v][k][b][c] != nullptr) {
                            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(dexec_[f][v][k][b][c]));
                        }
                        if (dgraph_[f][v][k][b][c] != nullptr) {
                            cudaGraphDestroy(static_cast<cudaGraph_t>(dgraph_[f][v][k][b][c]));
                        }
                    }
                }
            }
        }
        for (void * e : ev_[c]) {
            if (e != nullptr) {
                cudaEventDestroy(static_cast<cudaEvent_t>(e));
            }
        }
        if (st_[c] != nullptr) {
            cudaStreamDestroy(static_cast<cudaStream_t>(st_[c]));
        }
    }
    for (int32_t c = 0; c < 2 && m_ != nullptr; ++c) {
        if (carry_save_[c] != nullptr) {
            cudaSetDevice(m_->device[c]);
            cudaFree(carry_save_[c]);
        }
    }
    cudaFreeHost(ring_);
    cudaFreeHost(flag_);
    cudaFreeHost(done_);
    cudaFreeHost(e_host_);
}

bool target::init(model::model & m, const target_options & o, std::string & err) {
    m_ = &m;
    o_ = o;
    o_.pass.sampling = o_.sample ? &o_.sampling : nullptr; // SM1: the decision's sampler, or V1's greedy
    if (!read_ple_params(o.model_path, ple_, err)) {
        return false;
    }
    if (m.ple_table == nullptr) {
        return fail(err, "the PLE table is not mapped");
    }
    if (!check_draft_options(o.draft, err) || !model::pack_dense_all(m, err)) {
        return false;
    }
    if (o.draft.on && o.sample && o.sampling.temp > 0.0f && fn::samp::path_of(o.sampling) != fn::samp::path::topk) {
        // SP1 (FC-DEC3): the draft's full-path pick (q_shard on the re-sharded head) is not built
        return fail(err, "speculation at T > 0 off the sampler's top-k path (top-k 0, or above 256) is refused: the "
                         "draft's full-path pick (q_shard) is not built (SP1); use top-k 1 .. 256");
    }
    std::vector<int32_t> subset;
    if (o.draft.on && (!model::pack_draft(m, err) || !model::read_draft_subset(o.draft.subset, subset, err) ||
                       !model::draft_head_fill(m, subset, err))) {
        return false;
    }
    // mapped host memory, visible to both devices (portable)
    const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
    if (!cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&ring_), sizeof(fn::round::desc) * fn::round::ring, fl),
                 "descriptor ring", err) ||
        !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&flag_), 64, fl), "start flag", err) ||
        !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&done_), sizeof(fn::round::done) * 2, fl),
                 "completion records", err) ||
        !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&e_host_), sizeof(float) * fn::round::ring * max_b * E,
                               cudaHostAllocPortable),
                 "PLE upload staging", err)) {
        return false;
    }
    std::memset(ring_, 0, sizeof(fn::round::desc) * fn::round::ring);
    std::memset(flag_, 0, 64);
    std::memset(done_, 0, sizeof(fn::round::done) * 2);
    fn::round::desc * ring_dev = nullptr;
    uint64_t * flag_dev = nullptr;
    fn::round::done * done_dev = nullptr;
    if (!cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void **>(&ring_dev), ring_, 0), "ring pointer", err) ||
        !cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void **>(&flag_dev), flag_, 0), "flag pointer", err) ||
        !cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void **>(&done_dev), done_, 0), "done pointer", err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        cudaStream_t s = nullptr;
        if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream", err)) {
            return false;
        }
        st_[c] = s;
        for (void *& e : ev_[c]) {
            cudaEvent_t ev = nullptr;
            if (!cuda_ok(cudaEventCreate(&ev), "event", err)) {
                return false;
            }
            e = ev;
        }
        if (!model::pass_init(m, c, s, ring_dev, flag_dev, done_dev + c, next_round_, p_[c], err)) {
            return false;
        }
        v_.bind(c, m.device[c], s, &p_[c].rstate->cur);
        if (o.draft.on && !model::draft_init(m, p_[c], dc_[c], err)) {
            return false;
        }
        if (o.draft.on) {
            void * cs = nullptr;
            if (!cuda_ok(cudaMalloc(&cs, sizeof(float) * 4 * 1280), "retained carry copy", err)) {
                return false;
            }
            carry_save_[c] = static_cast<float *>(cs);
            // SP1: T > 0 on the top-k path draws the draft's proposals with SM1's pick (draft_card::sampled)
            dc_[c].sampled =
                o_.sample && o_.sampling.temp > 0.0f && fn::samp::path_of(o_.sampling) == fn::samp::path::topk;
        }
    }
    if (o_.sample) {
        // SM1: the request's options (refused when FNFL does not honour them) on both cards
        const int32_t cells = p_[0].s[3].n_cells;
        if (!o_.sampling.check(cells, err)) {
            return false;
        }
        const fn::samp::params prm = fn::samp::make_params(o_.sampling, cells);
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
                !fn::samp::write_params(p_[c].samp, prm, o_.sampling, st_[c])) {
                return err.empty() ? fail(err, "sampler params") : false;
            }
        }
    }
    sparse_built_ = o.draft.on && o.draft.attention == "sparse"; // SP2
    if (o.eager) {
        dmode_ = sparse_built_ ? 1 : 0;
        for (int32_t c = 0; c < 2 && o.draft.on; ++c) {
            dc_[c].sparse = sparse_built_;
        }
        return reset(err);
    }
    // one graph per card per B, captured once, and the draft's step graphs; SV1: the other decision variant too
    var_ = variant_of(o_);
    if (!capture_variant(var_, err)) {
        return false;
    }
    if (o.both_paths) {
        if (!o_.sample || o_.sampling.transform() || sp_path(o_.sampling) == fn::samp::path::full) {
            return fail(err, "both decision variants need the sampler on its greedy or top-k path, with no penalties "
                             "or bias (SV1)");
        }
        // the other variant's graphs differ only in the decision's stages and the draft's pick (samp.h stages_of)
        const fn::samp::options keep = o_.sampling;
        fn::samp::options other = keep;
        if (var_ == 0) {
            other.temp = 1.0f;
            other.top_k = keep.top_k > 0 ? keep.top_k : fn::samp::options().top_k;
        } else {
            other.temp = 0.0f;
        }
        size_t free0[2] = {0, 0}, free1[2] = {0, 0}, total = 0;
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
                !cuda_ok(cudaMemGetInfo(&free0[c], &total), "memory info", err)) {
                return false;
            }
        }
        const auto t0 = clock_t_::now();
        const int32_t n0 = captures_;
        o_.sampling = other;
        set_pick(1 - var_);
        const bool ok = capture_variant(1 - var_, err);
        o_.sampling = keep;
        set_pick(var_);
        if (!ok) {
            return false;
        }
        var_capture_s_ = seconds(t0, clock_t_::now());
        var_captures_ = captures_ - n0;
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
                !cuda_ok(cudaMemGetInfo(&free1[c], &total), "memory info", err)) {
                return false;
            }
            var_bytes_[c] = static_cast<int64_t>(free0[c]) - static_cast<int64_t>(free1[c]);
        }
    }
    // the draft's catch-up graph per B_c (1..5) per card: no pick, one variant (SP2: each captured form)
    if (o.draft.on && !capture_draft_kind(catchup_kind, fn::draft::max_rows, 0, err)) {
        return false;
    }
    dmode_ = sparse_built_ ? 1 : 0;
    for (int32_t c = 0; c < 2 && o.draft.on; ++c) {
        dc_[c].sparse = sparse_built_;
    }
    return reset(err);
}

bool target::capture_draft_kind(int32_t kind, int32_t max_rows, int32_t v, std::string & err) {
    for (int32_t f = 0; f <= (sparse_built_ ? 1 : 0); ++f) {
        for (int32_t c = 0; c < 2; ++c) {
            dc_[c].sparse = f == 1; // SP2: the form this capture records
        }
        for (int32_t rows = 1; rows <= max_rows; ++rows) {
            for (int32_t c = 0; c < 2; ++c) {
                if (!capture_draft(c, kind, rows, err, v, f)) {
                    return false;
                }
            }
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        dc_[c].sparse = dmode_ == 1;
    }
    return true;
}

bool target::set_draft_sparse(bool on, std::string & err) {
    if (!o_.draft.on || (on && !sparse_built_)) {
        return fail(err, "set_draft_sparse: no draft, or the sparse form not captured (draft attention sparse, SP2)");
    }
    dmode_ = on ? 1 : 0;
    for (int32_t c = 0; c < 2; ++c) {
        dc_[c].sparse = on;
    }
    return true;
}

bool target::ready_regions(int32_t c, std::vector<ready_region> & out, std::string & err) const {
    out.clear();
    struct want {
        const char * cls;
        const char * part; // null: every part
        bool versioned;
    };
    const want wants[] = {{"GDN state", nullptr, true},
                          {"convolution histories", nullptr, true},
                          {"draft round (DR1)", nullptr, false},
                          {"sampler (SM1)", nullptr, false},
                          {"round and pass (TG1)", "logits (B <= 5)", false},
                          {"round and pass (TG1)", "hidden carry (FC-HEAD3, B <= 5)", false}};
    for (const want & w : wants) {
        const model::reserved_class * rc = m_->state_class(w.cls);
        if (rc == nullptr) {
            return fail(err, std::string(w.cls) + ": not reserved");
        }
        int32_t found = 0;
        for (int32_t p = 0; p < static_cast<int32_t>(rc->plan.parts.size()); ++p) {
            const int64_t n = rc->plan.parts[p].bytes[c];
            if (n <= 0 || (w.part != nullptr && rc->plan.parts[p].name != w.part)) {
                continue;
            }
            if (w.versioned && n % fn::round::n_version != 0) {
                return fail(err, std::string(w.cls) + ": a part is not " + std::to_string(fn::round::n_version) +
                                     " versions");
            }
            out.push_back({rc->part_ptr(p, c), n, w.versioned});
            ++found;
        }
        if (found == 0) {
            return fail(err, std::string(w.cls) + " / " + (w.part != nullptr ? w.part : "parts") + ": none");
        }
    }
    return true;
}

bool target::save_ready(ready_state & s, std::string & err) {
    s.valid = false;
    if (invalid_ || l_.n < 1 || l_.u != 0 || !l_.retained) {
        return fail(err, "save_ready: the slot is not at a prefill's handoff (N >= 1, U = 0, logits retained)");
    }
    for (int32_t c = 0; c < 2; ++c) {
        std::vector<ready_region> rg;
        if (!ready_regions(c, rg, err) || !cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "sync", err)) {
            return false;
        }
        int64_t total = 0;
        for (const ready_region & r : rg) {
            total += r.versioned ? r.bytes / fn::round::n_version : r.bytes;
        }
        s.dev[c].assign(static_cast<size_t>(total), 0);
        int64_t off = 0;
        for (const ready_region & r : rg) {
            const int64_t n = r.versioned ? r.bytes / fn::round::n_version : r.bytes;
            const uint8_t * src = r.p + (r.versioned ? int64_t(l_.cur) * n : 0);
            if (!cuda_ok(cudaMemcpy(s.dev[c].data() + off, src, size_t(n), cudaMemcpyDeviceToHost), "save_ready",
                         err)) {
                return false;
            }
            off += n;
        }
    }
    s.l = l_;
    s.draft_n = draft_n_;
    s.batch_start = batch_start_;
    s.index_n = index_n_;
    s.sampling = o_.sampling;
    s.valid = true;
    return true;
}

bool target::restore_ready(const ready_state & s, std::string & err) {
    if (!s.valid) {
        return fail(err, "restore_ready: no saved state");
    }
    if (!clear_slot(false, err) || (o_.sample && !set_sampling(s.sampling, err))) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        std::vector<ready_region> rg;
        if (!ready_regions(c, rg, err) || !cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err)) {
            return false;
        }
        int64_t off = 0;
        for (const ready_region & r : rg) {
            const int64_t n = r.versioned ? r.bytes / fn::round::n_version : r.bytes;
            if (off + n > static_cast<int64_t>(s.dev[c].size()) ||
                !cuda_ok(cudaMemcpy(r.p, s.dev[c].data() + off, size_t(n), cudaMemcpyHostToDevice), "restore_ready",
                         err)) {
                return err.empty() ? fail(err, "restore_ready: the saved state does not match the regions") : false;
            }
            off += n; // a versioned object's committed version goes to version 0 (cur 0 after the rewind)
        }
        if (off != static_cast<int64_t>(s.dev[c].size())) {
            return fail(err, "restore_ready: the saved state does not match the regions");
        }
    }
    l_ = s.l;
    l_.cur = 0;
    draft_n_ = s.draft_n;
    batch_start_ = s.batch_start;
    index_n_ = s.index_n;
    carry_save_n_ = -1;
    invalid_ = false;
    return check_ledger(err);
}

void target::index_written(int64_t first, int64_t rows) {
    if (dmode_ == 1 && index_n_ >= first) {
        index_n_ = first + rows;
    } else if (dmode_ == 0) {
        index_n_ = std::min(index_n_, first);
    }
}

bool target::capture_target(int32_t v, int32_t b, int32_t c, std::string & err) {
    cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
    if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
        !cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture", err)) {
        return false;
    }
    model::use_counters u = model::pass_graph_counters(o_.pass);
    const model::step_status ss = model::pass_card_enqueue(p_[c], b, o_.pass, u, err);
    cudaGraph_t g = nullptr;
    const cudaError_t e = cudaStreamEndCapture(s, &g);
    if (ss != model::step_status::ok || !cuda_ok(e, "end capture", err)) {
        if (g != nullptr) {
            cudaGraphDestroy(g);
        }
        if (err.empty()) {
            err = "capture";
        }
        return false;
    }
    graph_[v][b][c] = g;
    cudaGraphExec_t x = nullptr;
    if (!cuda_ok(cudaGraphInstantiate(&x, g, 0), "instantiate", err) ||
        !cuda_ok(cudaGraphUpload(x, s), "graph upload", err) ||
        !cuda_ok(cudaStreamSynchronize(s), "graph upload", err)) {
        return false;
    }
    exec_[v][b][c] = x;
    ++captures_;
    return o_.graph_nodes.empty() || write_nodes(v, b, c, err);
}

bool target::set_expert_map(int32_t map, std::string & err) {
    if (map < 0 || map > 3) {
        return fail(err,
                    "set_expert_map: map " + std::to_string(map) + " (0: SP2, 1: SP4's H1, 2: SP4's H2, 3: XB3 C1)");
    }
    if (map == p_[0].c.ex_map && map == p_[1].c.ex_map) {
        return true;
    }
    for (int32_t c = 0; c < 2; ++c) {
        p_[c].c.ex_map = map;
    }
    if (o_.eager) {
        return true; // eager steps enqueue with the current map
    }
    // each captured variant's graphs again under the options it was captured with (init's both_paths capture)
    const fn::samp::options keep = o_.sampling;
    bool ok = true;
    for (int32_t v = 0; v < 2 && ok; ++v) {
        if (exec_[v][1][0] == nullptr) {
            continue; // not captured
        }
        o_.sampling = var_opts_[v];
        for (int32_t b = 4; b <= max_b && ok; ++b) {
            for (int32_t c = 0; c < 2 && ok; ++c) {
                ok = cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) &&
                     cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "sync", err);
                if (exec_[v][b][c] != nullptr) {
                    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_[v][b][c]));
                    exec_[v][b][c] = nullptr;
                }
                if (graph_[v][b][c] != nullptr) {
                    cudaGraphDestroy(static_cast<cudaGraph_t>(graph_[v][b][c]));
                    graph_[v][b][c] = nullptr;
                }
                ok = ok && capture_target(v, b, c, err);
            }
        }
    }
    o_.sampling = keep;
    return ok;
}

bool target::capture_variant(int32_t v, std::string & err) {
    for (int32_t b = 1; b <= max_b; ++b) {
        for (int32_t c = 0; c < 2; ++c) {
            if (!capture_target(v, b, c, err)) {
                return false;
            }
        }
    }
    // the draft's step graphs per d (1..4) per card (the pick: draft_card::sampled; SP2: each captured form)
    if (o_.draft.on && !capture_draft_kind(step_kind, model::draft_max_steps, v, err)) {
        return false;
    }
    var_key_[v] = o_.sample ? static_cast<int32_t>(sp_path(o_.sampling)) : -1;
    var_opts_[v] = o_.sampling;
    return true;
}

void target::set_pick(int32_t v) {
    for (int32_t c = 0; c < 2 && o_.draft.on; ++c) {
        // SP1: T > 0 on the top-k path draws the draft's proposals with SM1's pick (draft_card::sampled)
        dc_[c].sampled = v == 1 && o_.sample && o_.sampling.temp > 0.0f;
    }
}

bool target::set_sampling(const fn::samp::options & o, std::string & err) {
    if (!o_.sample) {
        return fail(err, "set_sampling: the target runs without the sampler");
    }
    const int32_t cells = p_[0].s[3].n_cells;
    if (!o.check(cells, err)) {
        return false;
    }
    if (o.transform()) {
        return fail(err,
                    "sampling: penalties and logit bias are not served (SV1: their decision stages are not captured)");
    }
    if (o_.draft.on && o.temp > 0.0f && sp_path(o) != fn::samp::path::topk) {
        return fail(err, "speculation at T > 0 off the sampler's top-k path (top-k 0, or above 256) is refused: the "
                         "draft's full-path pick (q_shard) is not built (SP1); use top-k 1 .. 256");
    }
    const int32_t v = sp_path(o) == fn::samp::path::greedy ? 0 : 1;
    if (var_key_[v] != static_cast<int32_t>(sp_path(o))) {
        return fail(err, "set_sampling: this decision path's graphs were not captured (target_options::both_paths)");
    }
    const fn::samp::params prm = fn::samp::make_params(o, cells);
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !fn::samp::write_params(p_[c].samp, prm, o, st_[c])) {
            return err.empty() ? fail(err, "sampler params") : false;
        }
    }
    o_.sampling = o;
    var_ = v;
    set_pick(v);
    return true;
}

bool target::keep_processed(std::string & err) {
    if (l_.n < 0 || l_.n > static_cast<int64_t>(l_.tok.size())) {
        return fail(err, "keep_processed: N beyond the ledger");
    }
    l_.tok.resize(static_cast<size_t>(l_.n));
    l_.u = 0;
    l_.pending = -1;
    l_.prompt_len = l_.n;
    l_.o = 0;
    l_.a = 0;
    return check_ledger(err);
}

bool target::capture_draft(int32_t card, int32_t kind, int32_t rows, std::string & err, int32_t v, int32_t mode) {
    cudaStream_t s = static_cast<cudaStream_t>(st_[card]);
    if (!cuda_ok(cudaSetDevice(m_->device[card]), "cudaSetDevice", err) ||
        !cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture", err)) {
        return false;
    }
    model::use_counters u = model::draft_graph_counters(kind, rows);
    const model::step_status ss = model::draft_card_enqueue(p_[card], dc_[card], kind, rows, u, err);
    cudaGraph_t g = nullptr;
    const cudaError_t e = cudaStreamEndCapture(s, &g);
    if (ss != model::step_status::ok || !cuda_ok(e, "end capture", err)) {
        if (g != nullptr) {
            cudaGraphDestroy(g);
        }
        if (err.empty()) {
            err = "draft capture";
        }
        return false;
    }
    const int32_t k = kind == step_kind ? 0 : 1;
    dgraph_[mode][v][k][rows][card] = g;
    cudaGraphExec_t x = nullptr;
    if (!cuda_ok(cudaGraphInstantiate(&x, g, 0), "instantiate", err) ||
        !cuda_ok(cudaGraphUpload(x, s), "graph upload", err) ||
        !cuda_ok(cudaStreamSynchronize(s), "graph upload", err)) {
        return false;
    }
    dexec_[mode][v][k][rows][card] = x;
    ++captures_;
    return true;
}

bool target::reset(std::string & err) {
    return clear_slot(true, err);
}

bool target::rewind(std::string & err) {
    return clear_slot(false, err);
}

bool target::prompt_begin(std::string & err) {
    if (invalid_) {
        return fail(err, "the slot is invalidated (a decision-hash mismatch, FC-RD2): reset and re-prefill");
    }
    v_.begin_step(l_, true);  // the previous step is acknowledged; a chunk overwrites no retained logits until the head
    return true;
}

bool target::prompt_commit(int32_t rows, int32_t k, std::string & err) {
    if (rows < 1 || rows > l_.u || k < 1 || k > 2) {
        return fail(err, "prompt commit: rows " + std::to_string(rows) + " (U " + std::to_string(l_.u) + "), k " +
                             std::to_string(k));
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !(fn::round::advance(&p_[c].rstate->cur, k, st_[c]) ? true : fail(err, "prompt commit: advance")) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "prompt commit", err)) {
            return false;
        }
    }
    uint32_t reported[2] = { 0, 0 };
    if (!v_.read_cur(reported, err) || !v_.committed_prompt(l_, k, reported, err)) {
        return false;
    }
    index_written(l_.n, rows); // SP2: the chunk's draft index cells (the sparse form's prompt path)
    l_.n += rows;
    l_.u -= rows;
    l_.retained     = false;
    l_.retained_row = -1;
    slot_versions::settle(l_);
    if (o_.draft.on) {
        draft_n_ = l_.n;
    }
    batch_start_  = l_.n;
    carry_save_n_ = -1;
    return check_ledger(err);
}

bool target::prompt_head(std::string & err) {
    if (l_.u != 0 || l_.n < 1) {
        return fail(err, "prompt head: the prompt is not fully processed");
    }
    const fn::xchg::rings & rg = *m_->rings;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "prompt head", err)) {
            return false;
        }
    }
    if (!rg.reset_all()) {
        return fail(err, "prompt head: rings reset");
    }
    constexpr int32_t s = 2;
    for (int32_t part = 0; part < 3; ++part) {
        for (int32_t c = 0; c < 2; ++c) {
            model::pass_card & p = p_[c];
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err)) {
                return false;
            }
            bool ok = false;
            if (part < 2) {
                const model::layer_input in = model::layer_output(p.c);
                fn::hc::sub_args         a  = {};
                a.w_down                    = p.head.w_down;
                a.w_up                      = p.head.w_up;
                a.hres                      = in.hres;
                a.hx                        = in.hx;
                a.hpost                     = in.hpost;
                a.gamma                     = p.head.gamma;
                a.x                         = p.carry;
                a.hc_ring                   = rg.slot(c, fn::xchg::msg::hc, s);
                a.hc_peer                   = rg.slot(1 - c, fn::xchg::msg::hc, s);
                a.hc_reset                  = nullptr;
                a.hmix_reset                = nullptr;
                a.hmix_own                  = rg.slot(c, fn::xchg::msg::hmix, s);
                a.hmix_peer                 = rg.slot(1 - c, fn::xchg::msg::hmix, s);
                a.err                       = rg.err(c);
                a.card                      = c;
                ok                          = fn::hc::head_sub(1, part + 1, a, st_[c]);
            } else {
                const fn::head::logits_args la = {
                    rg.slot(c, fn::xchg::msg::hmix, s), rg.err(c), c, p.out_w, fn::head::n_vocab_card, p.logits, nullptr
                };
                ok = fn::head::logits(1, la, st_[c]);
            }
            if (!ok) {
                return fail(err, "prompt head: part " + std::to_string(part));
            }
        }
    }
    uint32_t e[2] = { 0, 0 };
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "prompt head", err) ||
            !rg.read_err(c, e[c])) {
            return err.empty() ? fail(err, "prompt head: error word") : false;
        }
    }
    if ((e[0] | e[1]) != 0u) {
        return fail(err, "prompt head: exchange error words " + std::to_string(e[0]) + " / " + std::to_string(e[1]));
    }
    if (!rg.reset_all()) {
        return fail(err, "prompt head: rings reset");
    }
    l_.retained     = true;
    l_.retained_row = 0;
    return check_ledger(err);
}

bool target::restored(int64_t n, const std::vector<int32_t> & tokens, std::string & err) {
    if (n < 1 || n >= static_cast<int64_t>(tokens.size()) || l_.n != 0 || !l_.tok.empty()) {
        return fail(err, "restored: position " + std::to_string(n) + " outside 1 .. L - 1, or the slot not rewound");
    }
    set_prompt(tokens);
    l_.n = n;
    l_.u = static_cast<int64_t>(tokens.size()) - n;
    slot_versions::settle(l_); // the summaries of the whole blocks below n are the checkpoint's prefix's, unchanged
    draft_n_ = o_.draft.on ? n : 0;
    index_n_ = std::min(index_n_, draft_n_); // SP2
    batch_start_ = n; // the next target round starts at n; its first catch-up reads the restored retained carry
    carry_save_n_ = -1;
    return check_ledger(err);
}

bool target::clear_slot(bool zero_state, std::string & err) {
    const char * classes[] = {"target KV (K0)", "target index keys and summaries",
                              "GDN state",      "convolution histories",
                              "draft KV (F16)", "draft round (DR1)"};
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "sync", err)) {
            return false;
        }
        for (const char * cls : classes) {
            // rewind keeps the state objects but zeroes the draft round's buffers as reset does (the caller then
            // writes the checkpoint's retained carry), so a restored slot's scratch starts as a fresh one's
            if (!zero_state && std::strcmp(cls, "draft round (DR1)") != 0) {
                continue;
            }
            const model::reserved_class * rc = m_->state_class(cls);
            if (rc == nullptr) {
                return fail(err, std::string(cls) + ": not reserved");
            }
            for (int32_t p = 0; p < (int32_t)rc->plan.parts.size(); ++p) {
                const int64_t n = rc->plan.parts[p].bytes[c];
                if (n > 0 && !cuda_ok(cudaMemset(rc->part_ptr(p, c), 0, size_t(n)), cls, err)) {
                    return false;
                }
            }
        }
        const model::reserved_class * rc = m_->state_class("round and pass (TG1)");
        const int32_t pw = part_of(rc, "PLE workspace");
        if (pw < 0 || !model::clear_workspaces(*m_, c, err) ||
            !cuda_ok(cudaMemset(p_[c].greedy_work, 0, fn::head::greedy_work_bytes()), "greedy work", err) ||
            !cuda_ok(cudaMemset(p_[c].c.ex.status, 0, 4), "status", err) ||
            !(fn::ple::prepare(p_[c].ple_ws) ? true : fail(err, "ple prepare")) ||
            !cuda_ok(cudaDeviceSynchronize(), "reset", err)) {
            return err.empty() ? fail(err, "reset") : false;
        }
        u_[c] = model::use_counters();
    }
    draft_n_ = 0;
    if (zero_state) {
        index_n_ = 0; // SP2: rewind keeps the draft index with the KV (restored cuts it back)
    }
    batch_start_ = 0;
    carry_save_n_ = -1;
    if (!m_->rings->reset_all()) {
        return fail(err, "rings reset");
    }
    // SM1: the sampler's workspace cleared, no decision record, an empty window
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) || !fn::samp::prepare(p_[c].samp, st_[c]) ||
            !fn::samp::load_window(p_[c].samp, 0, {}, st_[c])) {
            return err.empty() ? fail(err, "sampler reset") : false;
        }
    }
    invalid_ = false;
    return v_.reset(l_, err); // the ledger emptied and cur 0 on both cards
}

void target::set_prompt(const std::vector<int32_t> & tokens) {
    const uint32_t cur = l_.cur; // the cards' committed slot stays
    l_ = ledger();
    l_.cur = cur;
    l_.tok = tokens;
    l_.u = static_cast<int64_t>(tokens.size());
    l_.prompt_len = l_.u;
}

void target::append(const std::vector<int32_t> & tokens) {
    l_.tok.insert(l_.tok.end(), tokens.begin(), tokens.end());
    l_.u += static_cast<int64_t>(tokens.size());
    l_.prompt_len = static_cast<int64_t>(l_.tok.size()); // forced input (rollback-test), no outputs
}

bool target::clear_errors(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err)) {
            return false;
        }
        for (int32_t i = 0; i < p_[c].n_err; ++i) {
            if (!cuda_ok(cudaMemsetAsync(p_[c].err_words[i], 0, 4, s), "clear error word", err)) {
                return false;
            }
        }
        if (!cuda_ok(cudaStreamSynchronize(s), "clear error words", err)) {
            return false;
        }
    }
    return true;
}

// one line a node: its index, type, and for a kernel its function's name and launch shape (SR1: the commit's node)
bool target::write_nodes(int32_t v, int32_t rows, int32_t card, std::string & err) const {
    cudaGraph_t g = static_cast<cudaGraph_t>(graph_[v][rows][card]);
    size_t n = 0;
    if (!cuda_ok(cudaGraphGetNodes(g, nullptr, &n), "graph nodes", err)) {
        return false;
    }
    std::vector<cudaGraphNode_t> nodes(n);
    if (!cuda_ok(cudaGraphGetNodes(g, nodes.data(), &n), "graph nodes", err)) {
        return false;
    }
    const std::string path = o_.graph_nodes + "/graph-nodes-" + (v == var_ ? "" : "v" + std::to_string(v) + "-") + "B" +
                             std::to_string(rows) + "-card" + std::to_string(card) + ".txt";
    FILE * f = fopen(path.c_str(), "w");
    if (f == nullptr) {
        return fail(err, "cannot write " + path);
    }
    for (size_t i = 0; i < n; ++i) {
        cudaGraphNodeType t = cudaGraphNodeTypeEmpty;
        cudaGraphNodeGetType(nodes[i], &t);
        if (t == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams kp = {};
            const char * name = "?";
            if (cudaGraphKernelNodeGetParams(nodes[i], &kp) != cudaSuccess ||
                cudaFuncGetName(&name, kp.func) != cudaSuccess) {
                name = "?";
            }
            fprintf(f, "%zu kernel %s grid %u,%u,%u block %u,%u,%u\n", i, name, kp.gridDim.x, kp.gridDim.y,
                    kp.gridDim.z, kp.blockDim.x, kp.blockDim.y, kp.blockDim.z);
        } else {
            fprintf(f, "%zu type %d%s\n", i, static_cast<int32_t>(t),
                    t == cudaGraphNodeTypeMemcpy   ? " memcpy"
                    : t == cudaGraphNodeTypeMemset ? " memset"
                                                   : "");
        }
    }
    fclose(f);
    return true;
}

bool target::set_depth(int64_t depth, std::string & err) {
    if (depth < 0 || depth > static_cast<int64_t>(l_.tok.size())) {
        return fail(err, "set_depth: beyond the ledger");
    }
    l_.n = depth;
    l_.u = static_cast<int64_t>(l_.tok.size()) - depth;
    slot_versions::settle(l_); // the synthetic history's summaries cover its whole blocks
    return true;
}

bool target::check_ledger(std::string & err) {
    return slot_versions::check(l_, err);
}

bool target::wait(uint64_t round, double timeout_s, std::string & err) {
    const auto t0 = clock_t_::now();
    for (;;) {
        const uint64_t a = reinterpret_cast<volatile uint64_t *>(&done_[0].round)[0];
        const uint64_t b = reinterpret_cast<volatile uint64_t *>(&done_[1].round)[0];
        if (a == round && b == round) {
            std::atomic_thread_fence(std::memory_order_acquire);
            return true;
        }
        for (int32_t c = 0; c < 2; ++c) {
            cudaSetDevice(m_->device[c]);
            const cudaError_t e = cudaStreamQuery(static_cast<cudaStream_t>(st_[c]));
            if (e != cudaSuccess && e != cudaErrorNotReady) {
                return fail(err, "card " + std::to_string(c) + ": " + cudaGetErrorString(e));
            }
        }
        if (seconds(t0, clock_t_::now()) > timeout_s) {
            return fail(err, "round " + std::to_string(round) + ": no completion record in " +
                                 std::to_string(timeout_s) + " s (card 0 at " + std::to_string(a) + ", card 1 at " +
                                 std::to_string(b) + ")");
        }
    }
}

bool target::round(int32_t kind, int32_t rows, bool upload_e, bool timed, round_result & r, std::string & err,
                   int32_t test_k, int32_t test_fail, int32_t budget, const float * const * staged) {
    r = round_result();
    if (invalid_) {
        return fail(err, "the slot is invalidated (a decision-hash mismatch, FC-RD2): reset and re-prefill");
    }
    const auto t0 = clock_t_::now();
    const uint64_t rd = next_round_;
    const int64_t n_before = l_.n;
    const int32_t cells = p_[0].s[3].n_cells;
    if (rows < 1 || rows > max_b || n_before + rows > cells) {
        return fail(err, "round refused before launch: B " + std::to_string(rows) + " at N " +
                             std::to_string(n_before) + " beyond " + std::to_string(cells) + " cells (FC-RD1)");
    }
    // the round's tokens: the next ledger tokens (forced) or the pending token (generate)
    int32_t tok[max_b] = {};
    for (int32_t t = 0; t < rows; ++t) {
        tok[t] = l_.tok[size_t(n_before + t)];
    }
    // the descriptor slot: free, since every earlier round has completed (this driver waits for each)
    fn::round::desc d = {};
    d.round = rd;
    d.ple_use = ple_use_;
    d.kind = kind;
    d.rows = rows;
    d.n_before = static_cast<int32_t>(n_before);
    d.budget = budget > 0 ? budget : rows;
    d.a = 0;
    d.n_out = 0;
    d.test_k = test_k;
    d.test_fail = test_fail;
    for (int32_t p = 0; p < 4; ++p) {
        for (int32_t t = 0; t < rows; ++t) {
            d.pos[p * rows + t] = static_cast<int32_t>(n_before + t);
        }
    }
    for (int32_t t = 0; t < max_b; ++t) {
        d.tok[t] = t < rows ? tok[t] : 0;
        d.out[t] = -1;
        d.argmax[t] = -1;
    }
    float * e = e_host_ + size_t(rd % fn::round::ring) * max_b * E;
    if (upload_e && staged == nullptr) {
        int32_t idx[max_b * ple_heads];
        if (!ple_indices(ple_, l_.tok.data(), n_before, tok, rows, idx)) {
            return fail(err, "PLE indices");
        }
        int32_t n_ranges = 0;
        if (!ple_prefetch(ple_, m_->ple_table, idx, rows, n_ranges, err)) {
            return false;
        }
        if (!ple_gather(ple_, m_->ple_table, idx, rows, e)) {
            return fail(err, "PLE gather");
        }
    }
    const auto t1 = clock_t_::now();
    v_.begin_step(l_, true); // the previous step is acknowledged: this round may overwrite its pre-step slot
    batch_start_ = n_before; // the round's head writes the hidden carry rows the draft's catch-up reads
    carry_save_n_ = -1;      // the next catch-up takes a new pre-step copy of the retained carry
    std::memcpy(&ring_[rd % fn::round::ring], &d, sizeof(d));
    std::atomic_thread_fence(std::memory_order_release);
    const bool eager = o_.eager;
    if (eager) {
        // the start condition first: the phased enqueue below runs card by card
        reinterpret_cast<volatile uint64_t *>(flag_)[0] = rd;
    }
    for (int32_t c = 0; c < 2 && upload_e; ++c) {
        cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            (timed && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_[c][0]), s), "event", err)) ||
            !cuda_ok(staged != nullptr
                         ? cudaMemcpyAsync(p_[c].ple_e, staged[c], size_t(rows) * E * 4, cudaMemcpyDeviceToDevice, s)
                         : cudaMemcpyAsync(p_[c].ple_e, e, size_t(rows) * E * 4, cudaMemcpyHostToDevice, s),
                     "PLE upload", err) ||
            (timed && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_[c][1]), s), "event", err))) {
            return false;
        }
    }
    const auto t2 = clock_t_::now();
    if (eager) {
        for (int32_t c = 0; c < 2 && timed; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
                !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_[c][2]), static_cast<cudaStream_t>(st_[c])),
                         "event", err)) {
                return false;
            }
        }
        model::pass_card * pp[2] = {&p_[0], &p_[1]};
        const model::step_status ss = model::pass_pair_enqueue(pp, rows, o_.pass, u_, err);
        if (ss != model::step_status::ok) {
            return err.empty() ? fail(err, "pass enqueue") : false;
        }
        for (int32_t c = 0; c < 2 && timed; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
                !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_[c][3]), static_cast<cudaStream_t>(st_[c])),
                         "event", err)) {
                return false;
            }
        }
    } else {
        for (int32_t c = 0; c < 2; ++c) {
            cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
                (timed && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_[c][2]), s), "event", err)) ||
                !cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec_[var_][rows][c]), s), "graph launch", err) ||
                (timed && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_[c][3]), s), "event", err))) {
                return false;
            }
        }
        // both cards' work is enqueued: release them together (RT-9)
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<volatile uint64_t *>(flag_)[0] = rd;
    }
    const auto t3 = clock_t_::now();
    next_round_ = rd + 1;
    ++ple_use_;
    if (!wait(rd, o_.timeout_s, err)) {
        return false;
    }
    const auto t4 = clock_t_::now();
    r.round = rd;
    r.kind = kind;
    r.rows = rows;
    r.n_before = static_cast<int32_t>(n_before);
    r.n_out = done_[0].n_out;
    for (int32_t c = 0; c < 2; ++c) {
        const fn::round::done & dn = done_[c];
        for (int32_t t = 0; t < max_b; ++t) {
            r.argmax[c][t] = dn.argmax[t];
        }
        std::memcpy(r.err[c], dn.err, sizeof(dn.err));
        r.t_begin[c] = dn.t_begin;
        r.t_start[c] = dn.t_start;
        r.t_end[c] = dn.t_end;
        r.t_logits[c] = dn.t_logits;
        r.cur[c] = dn.cur;
    }
    for (int32_t t = 0; t < max_b; ++t) {
        r.out[t] = done_[0].out[t];
    }
    r.host_gather_s = seconds(t0, t1);
    r.host_upload_s = seconds(t1, t2);
    r.host_launch_s = seconds(t2, t3);
    r.host_wait_s = seconds(t3, t4);
    r.t_done_s = std::chrono::duration<double>(t4.time_since_epoch()).count();
    if (timed) {
        for (int32_t c = 0; c < 2; ++c) {
            cudaSetDevice(m_->device[c]);
            // the completion record can land before the graph's tail retires: wait for the end event
            if (!cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(ev_[c][3])), "event sync", err)) {
                return false;
            }
            if ((upload_e && !cuda_ok(cudaEventElapsedTime(&r.upload_ms[c], static_cast<cudaEvent_t>(ev_[c][0]),
                                                           static_cast<cudaEvent_t>(ev_[c][1])),
                                      "elapsed", err)) ||
                !cuda_ok(cudaEventElapsedTime(&r.span_ms[c], static_cast<cudaEvent_t>(ev_[c][2]),
                                              static_cast<cudaEvent_t>(ev_[c][3])),
                         "elapsed", err)) {
                return false;
            }
        }
    }
    // FC-RD2: an error word on either card means the round's commit does not take effect on either: a card whose own
    // words were clear has committed, so both cards go back to the pre-step slot and the error words are cleared; the
    // ledger is unchanged and the round reports its failure
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t i = 0; i < fn::round::max_err; ++i) {
            if (r.err[c][i] != 0u) {
                if (!after_error(r, err)) {
                    return false;
                }
                return fail(err, "round " + std::to_string(rd) + ": card " + std::to_string(c) + " error word " +
                                     std::to_string(i) + " = " + std::to_string(r.err[c][i]) +
                                     " (recovered: both cards at the pre-step slot)");
            }
        }
    }
    const bool verify = kind == static_cast<int32_t>(fn::round::kind::verify);
    if (verify) {
        // both cards emit the same a + 1 tokens (FC-DEC3), 1 .. B of them
        bool same = done_[1].n_out == r.n_out && r.n_out >= 1 && r.n_out <= rows;
        for (int32_t t = 0; t < max_b && same; ++t) {
            same = done_[1].out[t] == r.out[t];
        }
        if (!same) {
            return fail(err, "round " + std::to_string(rd) + ": the cards' verify outputs differ, or n_out " +
                                 std::to_string(r.n_out) + " is outside 1 .. B");
        }
    }
    if (!v_.committed(l_, fn::round::commit_versions(kind, rows, verify ? r.n_out - 1 : 0, test_k), r.cur, err)) {
        return false;
    }
    // both cards decide the same tokens (T-GREEDY)
    for (int32_t t = 0; t < rows; ++t) {
        if (r.argmax[0][t] != r.argmax[1][t]) {
            return fail(err, "round " + std::to_string(rd) + ": the cards' greedy tokens differ");
        }
    }
    return true;
}

bool target::forced(int32_t rows, round_result & r, std::string & err, bool upload_e, bool timed) {
    return forced_round(rows, r, err, upload_e, timed, nullptr);
}

bool target::forced_staged(int32_t rows, const float * const e[2], round_result & r, std::string & err) {
    if (e == nullptr || e[0] == nullptr || e[1] == nullptr) {
        return fail(err, "forced_staged: no staged rows");
    }
    return forced_round(rows, r, err, true, false, e);
}

bool target::forced_round(int32_t rows, round_result & r, std::string & err, bool upload_e, bool timed,
                          const float * const * staged) {
    if (rows > l_.u) {
        return fail(err, "forced round: B > U (FC-RD1)");
    }
    if (!round(static_cast<int32_t>(fn::round::kind::forced), rows, upload_e, timed, r, err, 0, 0, 0, staged)) {
        return false;
    }
    l_.n += rows;
    l_.u -= rows;
    l_.retained = true; // its last row's logits stay until the next round's head (the bootstrap reads them)
    l_.retained_row = rows - 1;
    slot_versions::settle(l_);
    return check_ledger(err);
}

bool target::forced_k(int32_t rows, int32_t k, int32_t test_fail, round_result & r, std::string & err) {
    if (rows > l_.u || k < 0 || k > rows) {
        return fail(err, "forced_k round: B > U or k outside 0 .. B");
    }
    if (!round(static_cast<int32_t>(fn::round::kind::forced_k), rows, true, false, r, err, k, test_fail)) {
        return false;
    }
    l_.n += k;
    l_.u -= k;
    l_.retained = k > 0; // row k - 1 is the last kept row; its logits are in the round's window
    l_.retained_row = k - 1;
    slot_versions::settle(l_);
    return check_ledger(err);
}

bool target::verify(int32_t d, const int32_t * prop, int32_t budget, round_result & r, std::string & err, bool timed) {
    if (!o_.sample) {
        return fail(err, "verify round: needs the sampler's verdict (T = 0 runs its greedy path)");
    }
    if (l_.u != 1 || l_.pending < 0 || d < 1 || d > fn::samp::max_drafts || prop == nullptr) {
        return fail(err, "verify round: no pending token, or d outside 1 .. 4");
    }
    if (d + 1 > budget) {
        return fail(err, "verify round refused before launch: B " + std::to_string(d + 1) + " beyond the budget " +
                             std::to_string(budget) + " (FC-RD1)");
    }
    // the rows' tokens: the pending token (the ledger's at N), then the proposals, only for the round's inputs
    const size_t len = l_.tok.size();
    for (int32_t t = 0; t < d; ++t) {
        l_.tok.push_back(prop[t]);
    }
    const bool ok = round(static_cast<int32_t>(fn::round::kind::verify), d + 1, true, timed, r, err, 0, 0, budget);
    l_.tok.resize(len);
    if (!ok) {
        return false;
    }
    const int32_t a = r.n_out - 1;
    for (int32_t t = 0; t < a; ++t) {
        if (r.out[t] != prop[t]) {
            return fail(err, "verify round: accepted token " + std::to_string(t) + " is not the proposal");
        }
    }
    // FC-SLOT1: the a accepted proposals and the correction or bonus join the ledger; N_after = N_before + a + 1
    l_.n += a + 1;
    for (int32_t t = 0; t <= a; ++t) {
        l_.tok.push_back(r.out[t]);
    }
    l_.pending = r.out[a];
    l_.o += a + 1;
    l_.retained = false;
    slot_versions::settle(l_);
    if (!check_ledger(err)) {
        return false;
    }
    ++l_.checks;
    if (l_.u != 1) {
        ++l_.failures;
        return fail(err, "ledger: U != 1 after a verify round");
    }
    return true;
}

bool target::save_carry(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemcpyAsync(carry_save_[c], dc_[c].carry_prev, sizeof(float) * 4 * 1280,
                                     cudaMemcpyDeviceToDevice, s),
                     "retained carry copy", err)) {
            return false;
        }
    }
    carry_save_n_ = draft_n_;
    return true;
}

bool target::restore_carry(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemcpyAsync(dc_[c].carry_prev, carry_save_[c], sizeof(float) * 4 * 1280,
                                     cudaMemcpyDeviceToDevice, s),
                     "retained carry restore", err) ||
            !cuda_ok(cudaStreamSynchronize(s), "retained carry restore", err)) {
            return false;
        }
    }
    draft_n_ = carry_save_n_;
    index_n_ = std::min(index_n_, draft_n_); // SP2
    return true;
}

bool target::cut(int32_t j, int64_t n_keep, std::string & err) {
    if (!v_.cut(l_, j, n_keep, err) || !check_ledger(err)) {
        return false;
    }
    if (draft_n_ > l_.n) {
        // SP1: back to the draft's state before the last step's catch-ups (its pre-step retained carry and length),
        // when that is within the kept prefix; the caller catches up the kept rows from there. Otherwise stale.
        if (o_.draft.on && carry_save_n_ >= 0 && carry_save_n_ <= l_.n) {
            return restore_carry(err);
        }
        draft_n_ = -1;
        index_n_ = 0; // SP2
    }
    return true;
}

bool target::generate(round_result & r, std::string & err, bool timed) {
    if (l_.u != 1 || l_.pending < 0) {
        return fail(err, "generate round: no pending token");
    }
    if (!round(static_cast<int32_t>(fn::round::kind::generate), 1, true, timed, r, err)) {
        return false;
    }
    if (r.n_out != 1 || r.out[0] < 0) {
        return fail(err, "generate round: no token emitted");
    }
    l_.n += 1;
    l_.tok.push_back(r.out[0]);
    l_.pending = r.out[0];
    l_.o += 1; // acknowledged by the next step's start (SR1; RS1 section 3)
    l_.retained = false;
    slot_versions::settle(l_);
    if (!check_ledger(err)) {
        return false;
    }
    ++l_.checks;
    if (l_.u != 1) {
        ++l_.failures;
        return fail(err, "ledger: U != 1 after a generate round");
    }
    return true;
}

bool target::after_error(round_result & r, std::string & err) {
    // SM1: a decision-hash mismatch on either card invalidates the slot (FC-RD2); the failed round's decision record is
    // dropped on both cards, so its tokens never join the penalty tables and the next comparison is of equal records
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t i = 0; i < fn::round::max_err; ++i) {
            if ((r.err[c][i] & fn::xchg::err_hash) != 0u) {
                invalid_ = true;
            }
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !fn::samp::discard_record(p_[c].samp, st_[c]) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "discard", err)) {
            return err.empty() ? fail(err, "sampler discard") : false;
        }
    }
    r.recovered = clear_errors(err) && v_.restore(l_, err) && check_ledger(err);
    return r.recovered;
}

bool target::close_check(bool & checked, std::string & err) {
    checked = false;
    for (int32_t pass = 0; pass < 2; ++pass) {
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
                !(pass == 0 ? fn::samp::close_send(p_[c].samp, st_[c]) : fn::samp::close_recv(p_[c].samp, st_[c]))) {
                return err.empty() ? fail(err, "closing check") : false;
            }
        }
    }
    fn::samp::record rec[2] = {};
    uint32_t e[2] = {0, 0};
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "closing check", err) ||
            !fn::samp::read_record(p_[c].samp, rec[c], st_[c]) || !m_->rings->read_err(c, e[c])) {
            return err.empty() ? fail(err, "closing check") : false;
        }
    }
    if (((e[0] | e[1]) & fn::xchg::err_hash) != 0u) {
        invalid_ = true;
        clear_errors(err);
        return fail(err, "closing check: the cards' decision records differ (FC-RD2): the slot is invalidated");
    }
    checked = rec[0].checked != 0u && rec[1].checked != 0u;
    return true;
}

bool target::test_flip_record(int32_t r, std::string & err) {
    fn::samp::record rec = {};
    if (!cuda_ok(cudaSetDevice(m_->device[r]), "cudaSetDevice", err) ||
        !fn::samp::read_record(p_[r].samp, rec, st_[r])) {
        return err.empty() ? fail(err, "read record") : false;
    }
    rec.words[2] ^= 1u;
    return cuda_ok(cudaMemcpy(p_[r].samp.rec, &rec, sizeof(rec), cudaMemcpyHostToDevice), "flip record", err);
}

bool target::bootstrap(round_result & r, std::string & err, int32_t test_fail) {
    r = round_result();
    if (invalid_) {
        return fail(err, "the slot is invalidated (a decision-hash mismatch, FC-RD2): reset and re-prefill");
    }
    if (o_.sample && o_.sampling.penalties()) {
        // SM1: the penalty window from the prompt (the pinned server accepts the prompt into the sampler)
        const int32_t cap = fn::samp::make_params(o_.sampling, p_[0].s[3].n_cells).cap;
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
                !fn::samp::load_window(p_[c].samp, cap, l_.tok, st_[c])) {
                return err.empty() ? fail(err, "penalty window") : false;
            }
        }
    }
    if (l_.u != 0 || !l_.retained || l_.retained_row < 0) {
        return fail(err, "bootstrap: the prompt is not fully processed, or no retained logits");
    }
    const uint64_t rd = next_round_;
    fn::round::desc d = {};
    d.round = rd;
    d.ple_use = ple_use_;
    d.kind = static_cast<int32_t>(fn::round::kind::bootstrap);
    d.rows = 1;
    d.n_before = static_cast<int32_t>(l_.n);
    d.budget = 1;
    d.test_fail = test_fail;
    for (int32_t t = 0; t < max_b; ++t) {
        d.out[t] = -1;
        d.argmax[t] = -1;
    }
    v_.begin_step(l_, false); // the bootstrap runs no head: the retained logits stay
    std::memcpy(&ring_[rd % fn::round::ring], &d, sizeof(d));
    std::atomic_thread_fence(std::memory_order_release);
    reinterpret_cast<volatile uint64_t *>(flag_)[0] = rd;
    model::pass_card * pp[2] = {&p_[0], &p_[1]};
    if (!model::pass_bootstrap_enqueue(pp, l_.retained_row, o_.pass, err)) {
        return false;
    }
    next_round_ = rd + 1;
    if (!wait(rd, o_.timeout_s, err)) {
        return false;
    }
    r.t_done_s = std::chrono::duration<double>(clock_t_::now().time_since_epoch()).count(); // SF1: as every round
    r.round = rd;
    r.kind = d.kind;
    r.rows = 1;
    r.n_before = d.n_before;
    r.n_out = done_[0].n_out;
    r.out[0] = done_[0].out[0];
    for (int32_t c = 0; c < 2; ++c) {
        r.argmax[c][0] = done_[c].argmax[0];
        std::memcpy(r.err[c], done_[c].err, sizeof(done_[c].err));
        r.t_begin[c] = done_[c].t_begin;
        r.t_start[c] = done_[c].t_start;
        r.t_end[c] = done_[c].t_end;
        r.cur[c] = done_[c].cur;
    }
    // FC-RD2, as every round: an error word on either card clears the words and restores (the bootstrap ran no head,
    // so the retained row stays); the ledger is unchanged and the round reports its failure
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t i = 0; i < fn::round::max_err; ++i) {
            if (r.err[c][i] != 0u) {
                if (!after_error(r, err)) {
                    return false;
                }
                return fail(err, "bootstrap: card " + std::to_string(c) + " error word " + std::to_string(i) +
                                     " (recovered: the ledger and the retained row kept)");
            }
        }
    }
    if (!v_.committed(l_, 0, r.cur, err)) { // the bootstrap commits no version (k = 0)
        return false;
    }
    if (r.n_out != 1 || r.out[0] < 0 || r.argmax[0][0] != r.argmax[1][0]) {
        return fail(err, "bootstrap: no token, or the cards differ");
    }
    // N unchanged; the token is appended as the pending token (FC-RD3)
    l_.tok.push_back(r.out[0]);
    l_.pending = r.out[0];
    l_.u += 1;
    l_.o += 1;
    l_.retained = false;
    if (!check_ledger(err)) {
        return false;
    }
    ++l_.checks;
    if (l_.u != 1) {
        ++l_.failures;
        return fail(err, "ledger: U != 1 after the bootstrap");
    }
    return true;
}

// ---- the draft rounds (DR1 Stage C) ----

bool target::draft_set_depth(int64_t depth, std::string & err) {
    if (!o_.draft.on || depth < 0 || depth > l_.n || depth >= dc_[0].max_cells) {
        return fail(err, "draft_set_depth: no draft, or beyond N or the draft's cells");
    }
    draft_n_ = depth;
    index_n_ = dmode_ == 1 ? depth : std::min(index_n_, depth); // SP2: the synthetic history includes the index
    batch_start_ = depth;
    return true;
}

bool target::draft_catchup(int32_t rows, round_result & r, std::string & err) {
    const int64_t first = draft_n_;
    if (!o_.draft.on || first < 0 || rows < 1 || rows > fn::draft::max_rows || first < batch_start_ ||
        first + rows > l_.n) {
        return fail(err, "draft catch-up refused: rows " + std::to_string(rows) + " from " + std::to_string(first) +
                             " (batch start " + std::to_string(batch_start_) + ", N " + std::to_string(l_.n) +
                             "): the rows must be committed and the carry the last target batch's");
    }
    if (carry_save_n_ < 0 && !save_carry(err)) {
        return false;
    }
    if (!draft_round(catchup_kind, rows, first, static_cast<int32_t>(first - batch_start_), r, err)) {
        // SP1: a failed catch-up may have overwritten the retained carry: back to the pre-step copy (a real error
        // still ends the request, until SV1)
        std::string e2;
        if (r.recovered && !restore_carry(e2)) {
            err += "; " + e2;
        }
        return false;
    }
    draft_n_ = first + rows;
    index_written(first, rows);
    return true;
}

bool target::draft_step(int32_t d, round_result & r, std::string & err) {
    const int64_t pos = draft_n_;
    if (!o_.draft.on || pos < 0 || d < 1 || d > model::draft_max_steps || pos > l_.n ||
        pos >= static_cast<int64_t>(l_.tok.size())) {
        return fail(err, "draft step refused: d " + std::to_string(d) + " at " + std::to_string(pos) + " (N " +
                             std::to_string(l_.n) + ", L " + std::to_string(l_.tok.size()) + ")");
    }
    if (dmode_ == 1 && index_n_ < pos) {
        return fail(err, "sparse draft step refused: the draft index holds " + std::to_string(index_n_) +
                             " cells, the draft " + std::to_string(pos) + " (SP2: a dense write since)");
    }
    return draft_round(step_kind, d, pos, 0, r, err);
}

// A draft round, framed as round() frames a target round without its versions: no step acknowledgment and no commit
// (k = 0, so the cards' cur must stay); the target's buffers it borrows are free between rounds. An error word on
// either card is cleared and the round fails (the draft's K and V cells it wrote are rewritten by a later catch-up).
bool target::draft_round(int32_t kind, int32_t rows, int64_t first, int32_t hsrc, round_result & r, std::string & err) {
    r = round_result();
    if (invalid_) {
        return fail(err, "the slot is invalidated (a decision-hash mismatch, FC-RD2): reset and re-prefill");
    }
    if (first + rows > dc_[0].max_cells) {
        return fail(err, "draft round refused before launch: cells " + std::to_string(first) + " + " +
                             std::to_string(rows) + " beyond " + std::to_string(dc_[0].max_cells));
    }
    const uint64_t rd = next_round_;
    const bool step = kind == step_kind;
    fn::round::desc d = {};
    d.round = rd;
    d.ple_use = ple_use_;
    d.kind = kind;
    d.rows = rows;
    d.n_before = static_cast<int32_t>(first);
    d.budget = rows;
    d.hsrc = hsrc;
    for (int32_t p = 0; p < 4; ++p) {
        for (int32_t t = 0; t < rows; ++t) {
            d.pos[p * rows + t] = static_cast<int32_t>(first + t);
        }
    }
    for (int32_t t = 0; t < max_b; ++t) {
        d.tok[t] = t < (step ? 1 : rows) ? l_.tok[size_t(first + t)] : 0;
        d.out[t] = -1;
        d.argmax[t] = -1;
    }
    const auto t0 = clock_t_::now();
    std::memcpy(&ring_[rd % fn::round::ring], &d, sizeof(d));
    std::atomic_thread_fence(std::memory_order_release);
    const int32_t k = step ? 0 : 1;
    if (o_.eager) {
        reinterpret_cast<volatile uint64_t *>(flag_)[0] = rd;
        model::pass_card * pp[2] = {&p_[0], &p_[1]};
        model::draft_card * dd[2] = {&dc_[0], &dc_[1]};
        if (model::draft_pair_enqueue(pp, dd, kind, rows, u_, err) != model::step_status::ok) {
            return err.empty() ? fail(err, "draft enqueue") : false;
        }
    } else {
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
                !cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(dexec_[dmode_][k == 0 ? var_ : 0][k][rows][c]),
                                         static_cast<cudaStream_t>(st_[c])),
                         "graph launch", err)) {
                return false;
            }
        }
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<volatile uint64_t *>(flag_)[0] = rd;
    }
    const auto t1 = clock_t_::now();
    next_round_ = rd + 1;
    if (!wait(rd, o_.timeout_s, err)) {
        return false;
    }
    const auto t2 = clock_t_::now();
    r.round = rd;
    r.kind = kind;
    r.rows = rows;
    r.n_before = d.n_before;
    r.n_out = done_[0].n_out;
    for (int32_t c = 0; c < 2; ++c) {
        const fn::round::done & dn = done_[c];
        for (int32_t t = 0; t < max_b; ++t) {
            r.argmax[c][t] = dn.argmax[t];
        }
        std::memcpy(r.err[c], dn.err, sizeof(dn.err));
        r.t_begin[c] = dn.t_begin;
        r.t_start[c] = dn.t_start;
        r.t_end[c] = dn.t_end;
        r.cur[c] = dn.cur;
    }
    r.host_launch_s = seconds(t0, t1);
    r.host_wait_s = seconds(t1, t2);
    r.t_done_s = std::chrono::duration<double>(t2.time_since_epoch()).count();
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t i = 0; i < fn::round::max_err; ++i) {
            if (r.err[c][i] != 0u) {
                r.recovered = clear_errors(err);
                if (!r.recovered) {
                    return false;
                }
                return fail(err, "draft round " + std::to_string(rd) + ": card " + std::to_string(c) + " error word " +
                                     std::to_string(i) + " = " + std::to_string(r.err[c][i]) + " (cleared)");
            }
        }
    }
    if (r.n_out != 0 || r.cur[0] != l_.cur || r.cur[1] != l_.cur) {
        return fail(err, "draft round " + std::to_string(rd) + ": an output or a committed slot changed");
    }
    for (int32_t t = 0; step && t < rows; ++t) {
        if (r.argmax[0][t] != r.argmax[1][t] || r.argmax[0][t] < 0) {
            return fail(err, "draft round " + std::to_string(rd) + ": the cards' proposals differ");
        }
    }
    return true;
}

} // namespace fn::runtime
