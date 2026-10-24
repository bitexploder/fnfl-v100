#include "fn/runtime/joint.h"

#include "fn/kernels/draft/draft.h"
#include "fn/kernels/round/round.h"
#include "fn/model/layer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

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

constexpr int32_t lane_rows = model::layer_max_rows;
constexpr size_t errs_bytes = size_t(fn::xchg::max_lanes) * fn::round::max_err * sizeof(const uint32_t *);

// the joint draft's rows on a card (model::draft_joint): tokens, retained carries, head output rows, the word 1
constexpr size_t jd_rows_bytes = size_t(fn::xchg::max_lanes) * fn::draft::n_stream * fn::draft::half * sizeof(float);
constexpr size_t jd_tok = 0;
constexpr size_t jd_carry = 256;
constexpr size_t jd_h = jd_carry + jd_rows_bytes;
constexpr size_t jd_one = jd_h + jd_rows_bytes;
constexpr size_t jd_bytes = jd_one + 256;
static_assert(size_t(model::draft_max_steps + 1) * fn::xchg::max_lanes * sizeof(int32_t) <= jd_carry,
              "the joint draft's token rows");

constexpr int32_t draft_step_kind = static_cast<int32_t>(fn::round::kind::draft_step);

} // namespace

joint_frame::~joint_frame() {
    for (int32_t c = 0; c < 2 && m_ != nullptr; ++c) {
        cudaSetDevice(m_->device[c]);
        cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c]));
    }
    for (graph_pair & g : cache_) {
        drop(g);
    }
    cache_.clear();
    for (graph_pair & g : dcache_) {
        drop(g);
    }
    dcache_.clear();
    for (int32_t c = 0; c < 2 && m_ != nullptr; ++c) {
        cudaSetDevice(m_->device[c]);
        for (void * p : {ws_[c], work_[c], route_[c], errs_[c], jd_[c]}) {
            if (p != nullptr) {
                cudaFree(p);
            }
        }
    }
    rings_.reset();
}

bool joint_frame::init(model::model & m, void * const streams[2], int32_t cache, bool eager, std::string & err) {
    m_ = &m;
    eager_ = eager;
    st_[0] = streams[0];
    st_[1] = streams[1];
    cache_max_ = std::clamp(cache, 1, 8);
    if (m.single || m.kv_host || st_[0] == nullptr || st_[1] == nullptr || st_[0] == st_[1]) {
        return fail(err, "joint frame: resident TP on two cards with the engine's streams only");
    }
    if (!model::joint_dense_layouts(layouts_, err)) {
        return false;
    }
    auto rings = std::make_unique<fn::xchg::joint_rings>(m.device[0], m.device[1]);
    if (!rings->ok()) {
        return fail(err, "joint frame: the joint rings");
    }
    const size_t ws_bytes = model::layer_ws_bytes(fn::xchg::joint_rows);
    const size_t work_bytes = model::joint_dense_work_bytes();
    const size_t route_bytes = model::joint_route_bytes();
    for (int32_t c = 0; c < 2; ++c) {
        const std::pair<void **, size_t> parts[] = {{&ws_[c], ws_bytes},
                                                    {&work_[c], work_bytes},
                                                    {&route_[c], route_bytes},
                                                    {&errs_[c], errs_bytes},
                                                    {&jd_[c], jd_bytes}};
        if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err)) {
            return false;
        }
        for (const auto & [p, bytes] : parts) {
            if (!cuda_ok(cudaMalloc(p, bytes), "joint frame cudaMalloc", err) ||
                !cuda_ok(cudaMemset(*p, 0, bytes), "joint frame clear", err)) {
                return false;
            }
        }
        const int32_t one = 1;
        if (!cuda_ok(cudaMemcpy(static_cast<uint8_t *>(jd_[c]) + jd_one, &one, sizeof(one), cudaMemcpyHostToDevice),
                     "joint draft word", err)) {
            return false;
        }
        lws_[c] = model::carve_layer_ws(ws_[c], fn::xchg::joint_rows);
        bytes_[c] =
            fn::xchg::joint_rings_card_bytes() + int64_t(ws_bytes + work_bytes + route_bytes + errs_bytes + jd_bytes);
        if (!cuda_ok(cudaDeviceSynchronize(), "joint frame init", err)) {
            return false;
        }
    }
    rings_ = std::move(rings);
    return true;
}

int64_t joint_frame::graph_bytes(int32_t card) const {
    int64_t n = 0;
    for (const graph_pair & g : cache_) {
        n += g.bytes[card];
    }
    for (const graph_pair & g : dcache_) {
        n += g.bytes[card];
    }
    return n;
}

void joint_frame::drop(graph_pair & g) {
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        if (g.exec[c] != nullptr) {
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(g.exec[c]));
            g.exec[c] = nullptr;
        }
        if (g.graph[c] != nullptr) {
            cudaGraphDestroy(static_cast<cudaGraph_t>(g.graph[c]));
            g.graph[c] = nullptr;
        }
    }
}

bool joint_frame::lanes(target * const t[], int32_t n, int32_t c, std::vector<std::unique_ptr<model::pass_card>> & lane,
                        model::joint_pass & jp, std::string & err) {
    // each lane's pass: its slot's with the lane frame (its error list written here, before the enqueue)
    jp.lanes = n;
    for (int32_t k = 0; k < n; ++k) {
        lane.push_back(std::make_unique<model::pass_card>());
        const uint32_t ** errs = static_cast<const uint32_t **>(errs_[c]) + k * fn::round::max_err;
        if (!model::pass_lane_init(t[k]->card(c), *rings_, k, lws_[c], route_[c], errs, *lane.back(), err)) {
            return false;
        }
        jp.lane[k] = lane.back().get();
        jp.o[k] = &t[k]->pass_options();
    }
    jp.j.jr = rings_.get();
    jp.j.dense_work = work_[c];
    jp.j.dense_sites = layouts_;
    return true;
}

bool joint_frame::enqueue_eager(target * const t[], int32_t n, std::string & err) {
    std::vector<std::unique_ptr<model::pass_card>> lane[2];
    model::joint_pass jp[2];
    model::use_counters u[2];
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "joint eager sync", err) ||
            !lanes(t, n, c, lane[c], jp[c], err)) {
            return false;
        }
        // the graph's counters each round: every round the lone sequence from its start, as a joint graph's replay
        u[c] = model::pass_graph_counters(*jp[c].o[0]);
    }
    model::joint_pass * const jpp[2] = {&jp[0], &jp[1]};
    const model::step_status ss = model::pass_joint_pair_enqueue(jpp, u, err);
    if (ss != model::step_status::ok) {
        if (err.empty()) {
            err = "joint eager enqueue";
        }
        return false;
    }
    return true;
}

bool joint_frame::capture(target * const t[], int32_t n, const std::vector<int32_t> & key, graph_pair & g,
                          std::string & err) {
    g.key = key;
    size_t free0[2] = {0, 0}, free1[2] = {0, 0}, total = 0;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "joint capture sync", err) ||
            !cuda_ok(cudaMemGetInfo(&free0[c], &total), "memory info", err)) {
            return false;
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        std::vector<std::unique_ptr<model::pass_card>> lane;
        model::joint_pass jp;
        if (!lanes(t, n, c, lane, jp, err)) {
            return false;
        }
        cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "joint begin capture", err)) {
            return false;
        }
        model::use_counters u = model::pass_graph_counters(*jp.o[0]);
        const model::step_status ss = model::pass_joint_enqueue(jp, u, err);
        cudaGraph_t graph = nullptr;
        const cudaError_t e = cudaStreamEndCapture(s, &graph);
        if (ss != model::step_status::ok || !cuda_ok(e, "joint end capture", err)) {
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
            if (err.empty()) {
                err = "joint capture";
            }
            return false;
        }
        g.graph[c] = graph;
        size_t nodes = 0;
        if (!cuda_ok(cudaGraphGetNodes(graph, nullptr, &nodes), "joint graph nodes", err)) {
            return false;
        }
        nodes_[c] = static_cast<int64_t>(nodes);
        cudaGraphExec_t x = nullptr;
        if (!cuda_ok(cudaGraphInstantiate(&x, graph, 0), "joint instantiate", err) ||
            !cuda_ok(cudaGraphUpload(x, s), "joint graph upload", err) ||
            !cuda_ok(cudaStreamSynchronize(s), "joint graph upload", err)) {
            if (x != nullptr) {
                cudaGraphExecDestroy(x);
            }
            return false;
        }
        g.exec[c] = x;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemGetInfo(&free1[c], &total), "memory info", err)) {
            return false;
        }
        g.bytes[c] = static_cast<int64_t>(free0[c]) - static_cast<int64_t>(free1[c]);
        free_after_[c] = static_cast<int64_t>(free1[c]);
    }
    last_lanes_ = n;
    ++captures_;
    std::string slots;
    for (size_t i = 0; i < key.size(); i += 3) {
        slots += (i > 0 ? "," : "") + std::to_string(key[i]);
    }
    fprintf(stderr,
            "FNFL joint: captured the membership of slots %s: graphs %.1f / %.1f MiB, nodes %lld / %lld, free after "
            "%.3f / %.3f GiB on cards 0 / 1; capture %d\n",
            slots.c_str(), double(g.bytes[0]) / 1048576.0, double(g.bytes[1]) / 1048576.0,
            static_cast<long long>(nodes_[0]), static_cast<long long>(nodes_[1]), double(free_after_[0]) / 1073741824.0,
            double(free_after_[1]) / 1073741824.0, captures_);
    return true;
}

bool joint_frame::clear_after_error(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "joint drain", err)) {
            return false;
        }
    }
    // every lane ring and joint slot back at the sentinels, every error word zero, the lanes' PLE uses from 0
    if (!rings_->reset_all()) {
        return fail(err, "joint rings reset");
    }
    std::fill(std::begin(ple_use_), std::end(ple_use_), 0);
    return true;
}

bool joint_frame::verify(target * const t[], int32_t n, const int32_t d[], const int32_t * const prop[],
                         const int32_t budget[], round_result r[], bool ok[], std::string why[], bool & staged,
                         std::string & err) {
    staged = false;
    capture_failed_ = false;
    if (!ready() || n < 2 || n > fn::xchg::max_lanes) {
        return fail(err, "joint round: no frame, or members outside 2 .. 4");
    }
    std::vector<int32_t> key;
    for (int32_t k = 0; k < n; ++k) {
        if (d[k] != lane_rows - 1) {
            return fail(err, "joint round: a member's d is not 4 (lanes of five rows)");
        }
        key.push_back(t[k]->slot_index());
        key.push_back(t[k]->variant());
        key.push_back(t[k]->expert_map());
    }
    graph_pair * g = nullptr;
    for (graph_pair & q : cache_) {
        if (eager_) {
            break;
        }
        if (q.key == key) {
            g = &q;
        }
    }
    if (g == nullptr && !eager_) {
        if (static_cast<int32_t>(cache_.size()) >= cache_max_) {
            auto lru = std::min_element(cache_.begin(), cache_.end(),
                                        [](const graph_pair & a, const graph_pair & b) { return a.used < b.used; });
            drop(*lru);
            cache_.erase(lru);
        }
        cache_.emplace_back();
        if (!capture(t, n, key, cache_.back(), err)) {
            drop(cache_.back());
            cache_.pop_back();
            capture_failed_ = true;
            return false;
        }
        g = &cache_.back();
    }
    if (g != nullptr) {
        g->used = ++use_clock_;
    }
    target::staged_round sr[fn::xchg::max_lanes];
    for (int32_t k = 0; k < n; ++k) {
        if (!t[k]->joint_verify_begin(d[k], prop[k], budget[k], ple_use_[k], sr[k], err)) {
            for (int32_t q = 0; q < k; ++q) {
                t[q]->joint_abandon();
            }
            staged = k > 0;
            err = "joint round: member " + std::to_string(k) + ": " + err;
            return false;
        }
    }
    staged = true;
    // eager: each member's start flag set at its staging (round_stage), before the phased enqueue
    if (eager_ && !enqueue_eager(t, n, err)) {
        for (int32_t k = 0; k < n; ++k) {
            t[k]->joint_abandon();
        }
        return false;
    }
    for (int32_t c = 0; c < 2 && !eager_; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g->exec[c]), static_cast<cudaStream_t>(st_[c])),
                     "joint graph launch", err)) {
            for (int32_t k = 0; k < n; ++k) {
                t[k]->joint_abandon();
            }
            return false;
        }
    }
    // both cards' work is enqueued: every member released together (RT-9)
    for (int32_t k = 0; k < n; ++k) {
        t[k]->joint_release(sr[k]);
        ++ple_use_[k];
    }
    bool clean = true;
    for (int32_t k = 0; k < n; ++k) {
        ok[k] = t[k]->joint_verify_end(d[k], prop[k], sr[k], r[k], why[k]);
        clean = clean && ok[k] && !r[k].recovered;
    }
    ++rounds_;
    if (!clean) {
        std::string e2;
        if (!clear_after_error(e2)) {
            fprintf(stderr, "FNFL joint: the rings' reset after a member's error failed: %s\n", e2.c_str());
        }
    }
    return true;
}

// ---- FMU2 (build item 2): the joint draft step ----

bool joint_frame::draft_fits(target * const t[], int32_t n) {
    for (int32_t k = 0; k < n; ++k) {
        if (t[k] == nullptr || !t[k]->draft_on() || t[k]->draft_pick_sampled() != t[0]->draft_pick_sampled() ||
            t[k]->draft_sparse() != t[0]->draft_sparse()) {
            return false;
        }
    }
    return n >= 2;
}

void joint_frame::draft_lanes(target * const t[], int32_t n, int32_t c, model::draft_joint & j) {
    uint8_t * b = static_cast<uint8_t *>(jd_[c]);
    j = model::draft_joint();
    j.lanes = n;
    for (int32_t k = 0; k < n; ++k) {
        j.p[k] = &t[k]->card(c);
        j.d[k] = &t[k]->draft(c);
        const model::layer_ws w = model::lane_ws(lws_[c], k); // the select bodies: the joint layer_ws, free here
        j.sel_body[k] = w.sel_body;
        j.sel_off[k] = w.sel_off;
    }
    j.tok = reinterpret_cast<int32_t *>(b + jd_tok);
    j.carry = reinterpret_cast<float *>(b + jd_carry);
    j.h = reinterpret_cast<float *>(b + jd_h);
    j.one = reinterpret_cast<const int32_t *>(b + jd_one);
}

joint_frame::graph_pair * joint_frame::cached(std::vector<graph_pair> & cache, const std::vector<int32_t> & key) {
    for (graph_pair & q : cache) {
        if (q.key == key) {
            return &q;
        }
    }
    return nullptr;
}

bool joint_frame::capture_draft(target * const t[], int32_t n, int32_t d, const std::vector<int32_t> & key,
                                graph_pair & g, std::string & err) {
    g.key = key;
    size_t free0[2] = {0, 0}, free1[2] = {0, 0}, total = 0;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "joint draft capture sync", err) ||
            !cuda_ok(cudaMemGetInfo(&free0[c], &total), "memory info", err)) {
            return false;
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        model::draft_joint j;
        draft_lanes(t, n, c, j);
        cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "joint draft begin capture", err)) {
            return false;
        }
        model::use_counters u = model::draft_graph_counters(draft_step_kind, d);
        const model::step_status ss = model::draft_joint_enqueue(j, d, u, err);
        cudaGraph_t graph = nullptr;
        const cudaError_t e = cudaStreamEndCapture(s, &graph);
        if (ss != model::step_status::ok || !cuda_ok(e, "joint draft end capture", err)) {
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
            if (err.empty()) {
                err = "joint draft capture";
            }
            return false;
        }
        g.graph[c] = graph;
        cudaGraphExec_t x = nullptr;
        if (!cuda_ok(cudaGraphInstantiate(&x, graph, 0), "joint draft instantiate", err) ||
            !cuda_ok(cudaGraphUpload(x, s), "joint draft graph upload", err) ||
            !cuda_ok(cudaStreamSynchronize(s), "joint draft graph upload", err)) {
            if (x != nullptr) {
                cudaGraphExecDestroy(x);
            }
            return false;
        }
        g.exec[c] = x;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemGetInfo(&free1[c], &total), "memory info", err)) {
            return false;
        }
        g.bytes[c] = static_cast<int64_t>(free0[c]) - static_cast<int64_t>(free1[c]);
    }
    ++draft_captures_;
    return true;
}

bool joint_frame::draft(target * const t[], int32_t n, int32_t d, round_result r[], bool ok[], std::string why[],
                        std::string & err) {
    if (!ready() || n < 2 || n > fn::xchg::max_lanes || d < 1 || d > model::draft_max_steps || !draft_fits(t, n)) {
        return fail(err, "joint draft: no frame, members outside 2 .. 4, d outside 1 .. 4, mixed pick kinds or mixed "
                         "draft forms");
    }
    std::vector<int32_t> key;
    for (int32_t k = 0; k < n; ++k) {
        key.push_back(t[k]->slot_index());
    }
    key.push_back(d);
    key.push_back(t[0]->draft_sparse() ? 1 : 0);
    key.push_back(t[0]->draft_pick_sampled() ? 1 : 0); // FMU4: the pick kind
    graph_pair * g = eager_ ? nullptr : cached(dcache_, key);
    if (g == nullptr && !eager_) {
        if (static_cast<int32_t>(dcache_.size()) >= cache_max_) {
            auto lru = std::min_element(dcache_.begin(), dcache_.end(),
                                        [](const graph_pair & a, const graph_pair & b) { return a.used < b.used; });
            drop(*lru);
            dcache_.erase(lru);
        }
        dcache_.emplace_back();
        if (!capture_draft(t, n, d, key, dcache_.back(), err)) {
            drop(dcache_.back());
            dcache_.pop_back();
            return false;
        }
        g = &dcache_.back();
    }
    if (g != nullptr) {
        g->used = ++use_clock_;
    }
    // staged: each member's descriptor (eagerly its start flag); nothing to undo when the launch does not happen
    target::staged_round sr[fn::xchg::max_lanes];
    for (int32_t k = 0; k < n; ++k) {
        if (!t[k]->joint_draft_begin(d, sr[k], err)) {
            err = "joint draft: member " + std::to_string(k) + ": " + err;
            return false;
        }
    }
    std::string lerr;
    bool launched = true;
    if (eager_) {
        model::draft_joint j[2];
        for (int32_t c = 0; c < 2; ++c) {
            draft_lanes(t, n, c, j[c]);
        }
        model::draft_joint * const jj[2] = {&j[0], &j[1]};
        launched = model::draft_joint_pair_enqueue(jj, d, t[0]->eager_counters(), lerr) == model::step_status::ok;
    } else {
        for (int32_t c = 0; c < 2 && launched; ++c) {
            launched =
                cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", lerr) &&
                cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g->exec[c]), static_cast<cudaStream_t>(st_[c])),
                        "joint draft graph launch", lerr);
            if (!launched && c == 0) {
                err = lerr; // nothing launched: the members draft alone
                return false;
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (int32_t k = 0; k < n; ++k) {
        t[k]->joint_release(sr[k]);
    }
    for (int32_t k = 0; k < n; ++k) {
        if (!launched) {
            ok[k] = false; // part of the step enqueued: every member's step fails
            why[k] = "joint draft: " + (lerr.empty() ? std::string("enqueue") : lerr);
            continue;
        }
        sr[k].t1 = t1;
        ok[k] = t[k]->joint_draft_end(d, sr[k], r[k], why[k]);
    }
    ++draft_rounds_;
    return true;
}

} // namespace fn::runtime
