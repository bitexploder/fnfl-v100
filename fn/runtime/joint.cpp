#include "fn/runtime/joint.h"

#include "fn/kernels/draft/draft.h"
#include "fn/kernels/hc/hc.h"
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

// FMS2 (build item 2): the joint catch-up's resources on a card: the lanes' up2 run counters, then the K/V product's
// joint work of the dense and the sparse draft form (each zeroed once; every use leaves them zero)
constexpr size_t cu_runs_bytes = size_t(fn::xchg::max_lanes) * fn::hc::n_runs_card * sizeof(uint32_t);
size_t cu_kv_at(int32_t form) {
    const size_t a = (cu_runs_bytes + 255) / 256 * 256;
    return form == 0 ? a : a + (model::catchup_kv_work_bytes(false) + 255) / 256 * 256;
}
size_t cu_bytes() {
    return cu_kv_at(1) + model::catchup_kv_work_bytes(true);
}

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
    for (graph_pair & g : ccache_) {
        drop(g);
    }
    ccache_.clear();
    for (int32_t c = 0; c < 2 && m_ != nullptr; ++c) {
        cudaSetDevice(m_->device[c]);
        for (void * p : {ws_[c], work_[c], route_[c], errs_[c], jd_[c], cu_[c], kw_[c]}) {
            if (p != nullptr) {
                cudaFree(p);
            }
        }
    }
    rings_.reset();
}

bool joint_frame::init(model::model & m,
                       void * const   streams[2],
                       int32_t        cache,
                       bool           eager,
                       bool           x16,
                       bool           router_tc,
                       bool           dense_f16,
                       std::string &  err) {
    m_ = &m;
    eager_ = eager;
    x16_       = x16;
    router_tc_ = router_tc;
    dense_f16_ = dense_f16;
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
                                                    {&jd_[c], jd_bytes},
                                                    {&cu_[c], cu_bytes()},
                                                    {&kw_[c], model::joint_kernel_ws_bytes()}};
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
            fn::xchg::joint_rings_card_bytes() + int64_t(ws_bytes + work_bytes + route_bytes + errs_bytes + jd_bytes +
                                                         cu_bytes() + model::joint_kernel_ws_bytes());
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
    for (const graph_pair & g : ccache_) {
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

bool joint_frame::lanes(target * const                                   t[],
                        int32_t                                          n,
                        int32_t                                          rows,
                        int32_t                                          c,
                        std::vector<std::unique_ptr<model::pass_card>> & lane,
                        model::joint_pass &                              jp,
                        std::string &                                    err) {
    // each lane's pass: its slot's with the lane frame (its error list written here, before the enqueue)
    jp.lanes = n;
    jp.j.rows = rows;
    for (int32_t k = 0; k < n; ++k) {
        lane.push_back(std::make_unique<model::pass_card>());
        const uint32_t ** errs = static_cast<const uint32_t **>(errs_[c]) + k * fn::round::max_err;
        if (!model::pass_lane_init(t[k]->card(c), *rings_, k, lws_[c], route_[c], kw_[c], errs, *lane.back(), err)) {
            return false;
        }
        jp.lane[k] = lane.back().get();
        jp.o[k] = &t[k]->pass_options();
    }
    jp.j.jr = rings_.get();
    jp.j.dense_work = work_[c];
    jp.j.dense_sites = layouts_;
    // FMS13: the 16-bit exchange exists at five rows only; at R < 5 it is refused (named once a row count) and the
    // round runs the default F32 exchange
    const bool five  = rows == model::layer_max_rows;
    if (x16_ && !five && c == 0 && !refused_noted_[rows]) {
        refused_noted_[rows] = true;
        fprintf(stderr, "FNFL joint: R = %d: joint_x16 refused (five-row form only); the round runs the default path\n",
                rows);
    }
    jp.j.x16       = x16_ && five;
    jp.j.router_tc = router_tc_;
    jp.j.dense_f16 = dense_f16_;
    return true;
}

bool joint_frame::enqueue_eager(target * const t[], int32_t n, int32_t rows, std::string & err) {
    std::vector<std::unique_ptr<model::pass_card>> lane[2];
    model::joint_pass jp[2];
    model::use_counters u[2];
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "joint eager sync", err) ||
            !lanes(t, n, rows, c, lane[c], jp[c], err)) {
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

bool joint_frame::capture(target * const               t[],
                          int32_t                      n,
                          int32_t                      rows,
                          const std::vector<int32_t> & key,
                          graph_pair &                 g,
                          std::string &                err) {
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
        if (!lanes(t, n, rows, c, lane, jp, err)) {
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
        // FMS2 (build item 3): the joint verify graph's node list (test_graph_nodes), the last capture at each U and
        // (FMS13) R
        const std::string path = nodes_dir_ + "/graph-nodes-joint-verify-U" + std::to_string(n) + "-R" +
                                 std::to_string(rows) + "-card" + std::to_string(c) + ".txt";
        if (!nodes_dir_.empty() && !t[0]->write_graph_nodes(graph, path, err)) {
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
    for (size_t i = 0; i + 2 < key.size(); i += 3) {
        slots += (i > 0 ? "," : "") + std::to_string(key[i]);
    }
    fprintf(stderr,
            "FNFL joint: captured the membership of slots %s at R = %d: graphs %.1f / %.1f MiB, nodes %lld / %lld, "
            "free after %.3f / %.3f GiB on cards 0 / 1; capture %d\n",
            slots.c_str(), rows, double(g.bytes[0]) / 1048576.0, double(g.bytes[1]) / 1048576.0,
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
    // every lane ring and joint slot back at the sentinels, every error word zero, the lanes' PLE uses from 0; FMS2
    // (build item 3): the lanes' kernel workspaces zero (their error words, attend's and gdn_core's counters)
    if (!rings_->reset_all()) {
        return fail(err, "joint rings reset");
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemset(kw_[c], 0, model::joint_kernel_ws_bytes()), "joint kernel workspaces", err)) {
            return false;
        }
    }
    std::fill(std::begin(ple_use_), std::end(ple_use_), 0);
    return true;
}

// the membership's verify graphs at R rows a lane (null when eager): the cached pair, else captured now (the least
// recently used pair dropped when the cache is full); false, with capture_failed_, when the capture failed. FMS13: the
// key ends with R, so one process holds graphs at several R (rollback-test's section 7 runs d = 1 .. 4)
bool joint_frame::verify_graphs(target * const t[], int32_t n, int32_t rows, graph_pair *& g, std::string & err) {
    g = nullptr;
    if (eager_) {
        return true;
    }
    std::vector<int32_t> key;
    for (int32_t k = 0; k < n; ++k) {
        key.push_back(t[k]->slot_index());
        key.push_back(t[k]->variant());
        key.push_back(t[k]->expert_map());
    }
    key.push_back(rows);
    for (graph_pair & q : cache_) {
        if (q.key == key) {
            g = &q;
        }
    }
    if (g == nullptr) {
        if (static_cast<int32_t>(cache_.size()) >= cache_max_) {
            auto lru = std::min_element(cache_.begin(), cache_.end(),
                                        [](const graph_pair & a, const graph_pair & b) { return a.used < b.used; });
            drop(*lru);
            cache_.erase(lru);
        }
        cache_.emplace_back();
        if (!capture(t, n, rows, key, cache_.back(), err)) {
            drop(cache_.back());
            cache_.pop_back();
            capture_failed_ = true;
            return false;
        }
        g = &cache_.back();
    }
    g->used = ++use_clock_;
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
    // FMS13: any d in 1 .. 4, the same for every member (step_joint admits a lane only at the round's full d); the
    // round's lanes are R = d + 1 rows
    for (int32_t k = 0; k < n; ++k) {
        if (d[k] < 1 || d[k] > model::layer_max_rows - 1 || d[k] != d[0]) {
            return fail(err, "joint round: a member's d is outside 1 .. 4 or differs from member 0's");
        }
    }
    const int32_t rows = d[0] + 1;
    graph_pair * g = nullptr;
    if (!verify_graphs(t, n, rows, g, err)) {
        return false;
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
    if (eager_ && !enqueue_eager(t, n, rows, err)) {
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

// ---- TTTS1: a front stage's joint verify ----

bool joint_frame::front_verify(target * const        t[],
                               int32_t               n,
                               const int32_t         d[],
                               const int32_t * const prop[],
                               const int32_t         budget[],
                               target::front_round   fr[],
                               round_result          r[],
                               bool                  ok[],
                               std::string           why[],
                               bool &                staged,
                               std::string &         err) {
    staged          = false;
    capture_failed_ = false;
    if (!ready() || n < 2 || n > fn::xchg::max_lanes) {
        return fail(err, "joint front round: no frame, or members outside 2 .. 4");
    }
    for (int32_t k = 0; k < n; ++k) {
        if (!t[k]->stage_front() || d[k] < 1 || d[k] > model::layer_max_rows - 1 || d[k] != d[0]) {
            return fail(err,
                        "joint front round: a member is not a front stage, or its d is outside 1 .. 4 or "
                        "differs from member 0's");
        }
    }
    const int32_t rows = d[0] + 1;
    graph_pair *  g    = nullptr;
    if (!verify_graphs(t, n, rows, g, err)) {
        return false;
    }
    for (int32_t k = 0; k < n; ++k) {
        if (!t[k]->joint_front_begin(d[k], prop[k], budget[k], ple_use_[k], fr[k], err)) {
            for (int32_t q = 0; q < k; ++q) {
                t[q]->front_abandon(fr[q]);
            }
            staged = k > 0;
            err    = "joint front round: member " + std::to_string(k) + ": " + err;
            return false;
        }
    }
    staged = true;
    if (eager_ && !enqueue_eager(t, n, rows, err)) {
        for (int32_t k = 0; k < n; ++k) {
            t[k]->front_abandon(fr[k]);
        }
        return false;
    }
    for (int32_t c = 0; c < 2 && !eager_; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g->exec[c]), static_cast<cudaStream_t>(st_[c])),
                     "joint graph launch", err)) {
            for (int32_t k = 0; k < n; ++k) {
                t[k]->front_abandon(fr[k]);
            }
            return false;
        }
    }
    for (int32_t k = 0; k < n; ++k) {
        t[k]->joint_release(fr[k].sr);
        ++ple_use_[k];
    }
    bool clean = true;
    for (int32_t k = 0; k < n; ++k) {
        ok[k] = t[k]->front_wait(fr[k], r[k], why[k]);
        clean = clean && ok[k];
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

bool joint_frame::front_forced(target * const      t[],
                               int32_t             n,
                               int32_t             rows,
                               target::front_round fr[],
                               round_result        r[],
                               bool                ok[],
                               std::string         why[],
                               bool &              staged,
                               std::string &       err) {
    staged          = false;
    capture_failed_ = false;
    if (!ready() || n < 2 || n > fn::xchg::max_lanes || rows < 2 || rows > model::layer_max_rows) {
        return fail(err, "joint front forced: no frame, members outside 2 .. 4 or rows outside 2 .. 5");
    }
    for (int32_t k = 0; k < n; ++k) {
        if (!t[k]->stage_front()) {
            return fail(err, "joint front forced: a member is not a front stage");
        }
    }
    graph_pair * g = nullptr;
    if (!verify_graphs(t, n, rows, g, err)) {
        return false;
    }
    for (int32_t k = 0; k < n; ++k) {
        if (!t[k]->joint_front_forced_begin(rows, ple_use_[k], fr[k], err)) {
            for (int32_t q = 0; q < k; ++q) {
                t[q]->front_abandon(fr[q]);
            }
            staged = k > 0;
            err    = "joint front forced: member " + std::to_string(k) + ": " + err;
            return false;
        }
    }
    staged = true;
    if (eager_ && !enqueue_eager(t, n, rows, err)) {
        for (int32_t k = 0; k < n; ++k) {
            t[k]->front_abandon(fr[k]);
        }
        return false;
    }
    for (int32_t c = 0; c < 2 && !eager_; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g->exec[c]), static_cast<cudaStream_t>(st_[c])),
                     "joint graph launch", err)) {
            for (int32_t k = 0; k < n; ++k) {
                t[k]->front_abandon(fr[k]);
            }
            return false;
        }
    }
    for (int32_t k = 0; k < n; ++k) {
        t[k]->joint_release(fr[k].sr);
        ++ple_use_[k];
    }
    bool clean = true;
    for (int32_t k = 0; k < n; ++k) {
        ok[k] = t[k]->front_wait(fr[k], r[k], why[k]);
        clean = clean && ok[k];
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

// ---- FMS2 item 7 (TEST ONLY, joint-kl): a joint forced round on the verify's graphs ----

void joint_frame::test_set_x16(bool on) {
    if (on == x16_) {
        return;
    }
    for (graph_pair & g : cache_) {
        drop(g);
    }
    cache_.clear();
    x16_ = on;
}

bool joint_frame::forced(target * const t[],
                         int32_t        n,
                         int32_t        rows,
                         round_result   r[],
                         bool           ok[],
                         std::string    why[],
                         std::string &  err) {
    capture_failed_ = false;
    if (!ready() || n < 2 || n > fn::xchg::max_lanes || rows < 2 || rows > model::layer_max_rows) {
        return fail(err, "joint forced: no frame, members outside 2 .. 4 or rows outside 2 .. 5");
    }
    graph_pair * g = nullptr;
    if (!verify_graphs(t, n, rows, g, err)) {
        return false;
    }
    target::staged_round sr[fn::xchg::max_lanes];
    for (int32_t k = 0; k < n; ++k) {
        if (!t[k]->joint_forced_begin(rows, ple_use_[k], sr[k], err)) {
            err = "joint forced: member " + std::to_string(k) + ": " + err;
            return false;
        }
    }
    if (eager_ && !enqueue_eager(t, n, rows, err)) {
        return false;
    }
    for (int32_t c = 0; c < 2 && !eager_; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g->exec[c]), static_cast<cudaStream_t>(st_[c])),
                     "joint graph launch", err)) {
            return false;
        }
    }
    for (int32_t k = 0; k < n; ++k) {
        t[k]->joint_release(sr[k]);
        ++ple_use_[k];
    }
    bool clean = true;
    for (int32_t k = 0; k < n; ++k) {
        ok[k] = t[k]->joint_forced_end(rows, sr[k], r[k], why[k]);
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

// ---- FMS2 (build item 2): the joint catch-up ----

bool joint_frame::catchup_fits(target * const t[], int32_t n) {
    for (int32_t k = 0; k < n; ++k) {
        if (t[k] == nullptr || !t[k]->draft_on() || t[k]->draft_sparse() != t[0]->draft_sparse()) {
            return false;
        }
    }
    return n >= 2;
}

bool joint_frame::catchup_lanes(target * const t[], int32_t n, int32_t c,
                                std::vector<std::unique_ptr<model::pass_card>> & lane, model::catchup_joint & j,
                                std::string & err) {
    // each lane's pass: its slot's with the lane frame, as the verify's (a lane's error list names the lane ring's,
    // the card's workspaces' and the joint word: the verify's list of lane k, so its region is shared)
    j = model::catchup_joint();
    j.lanes = n;
    for (int32_t k = 0; k < n; ++k) {
        lane.push_back(std::make_unique<model::pass_card>());
        const uint32_t ** errs = static_cast<const uint32_t **>(errs_[c]) + k * fn::round::max_err;
        if (!model::pass_lane_init(t[k]->card(c), *rings_, k, lws_[c], route_[c], kw_[c], errs, *lane.back(), err)) {
            return false;
        }
        j.p[k] = lane.back().get();
        j.d[k] = &t[k]->draft(c);
    }
    uint8_t * b = static_cast<uint8_t *>(cu_[c]);
    j.jr = rings_.get();
    j.runs = reinterpret_cast<uint32_t *>(b);
    for (int32_t f = 0; f < 2; ++f) {
        j.kv_work[f] = b + cu_kv_at(f);
        j.kv_work_bytes[f] = model::catchup_kv_work_bytes(f == 1);
    }
    return true;
}

bool joint_frame::capture_catchup(target * const t[], int32_t n, const std::vector<int32_t> & key, graph_pair & g,
                                  std::string & err) {
    g.key = key;
    size_t free0[2] = {0, 0}, free1[2] = {0, 0}, total = 0;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "joint catch-up capture sync", err) ||
            !cuda_ok(cudaMemGetInfo(&free0[c], &total), "memory info", err)) {
            return false;
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        std::vector<std::unique_ptr<model::pass_card>> lane;
        model::catchup_joint j;
        if (!catchup_lanes(t, n, c, lane, j, err)) {
            return false;
        }
        cudaStream_t s = static_cast<cudaStream_t>(st_[c]);
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "joint catch-up begin capture",
                     err)) {
            return false;
        }
        model::use_counters u = model::catchup_joint_counters();
        const model::step_status ss = model::catchup_joint_enqueue(j, u, err);
        cudaGraph_t graph = nullptr;
        const cudaError_t e = cudaStreamEndCapture(s, &graph);
        if (ss != model::step_status::ok || !cuda_ok(e, "joint catch-up end capture", err)) {
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
            if (err.empty()) {
                err = "joint catch-up capture";
            }
            return false;
        }
        g.graph[c] = graph;
        const std::string form = key.back() == 1 ? "sparse" : "dense";
        const std::string path = nodes_dir_ + "/graph-nodes-joint-catchup-" + form + "-U" + std::to_string(n) +
                                 "-card" + std::to_string(c) + ".txt";
        if (!nodes_dir_.empty() && !t[0]->write_graph_nodes(graph, path, err)) {
            return false;
        }
        cudaGraphExec_t x = nullptr;
        if (!cuda_ok(cudaGraphInstantiate(&x, graph, 0), "joint catch-up instantiate", err) ||
            !cuda_ok(cudaGraphUpload(x, s), "joint catch-up graph upload", err) ||
            !cuda_ok(cudaStreamSynchronize(s), "joint catch-up graph upload", err)) {
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
    ++catchup_captures_;
    return true;
}

bool joint_frame::catchup(target * const t[], int32_t n, const int32_t rows[], round_result r[], bool ok[],
                          std::string why[], std::string & err) {
    capture_failed_ = false;
    if (!ready() || n < 2 || n > fn::xchg::max_lanes || !catchup_fits(t, n)) {
        return fail(err, "joint catch-up: no frame, members outside 2 .. 4, or mixed draft forms");
    }
    for (int32_t k = 0; k < n; ++k) {
        if (rows[k] < 1 || rows[k] > fn::draft::max_rows) {
            return fail(err, "joint catch-up: a member's rows outside 1 .. 5");
        }
    }
    std::vector<int32_t> key;
    for (int32_t k = 0; k < n; ++k) {
        key.push_back(t[k]->slot_index());
    }
    key.push_back(t[0]->draft_sparse() ? 1 : 0);
    graph_pair * g = eager_ ? nullptr : cached(ccache_, key);
    if (g == nullptr && !eager_) {
        if (static_cast<int32_t>(ccache_.size()) >= cache_max_) {
            auto lru = std::min_element(ccache_.begin(), ccache_.end(),
                                        [](const graph_pair & a, const graph_pair & b) { return a.used < b.used; });
            drop(*lru);
            ccache_.erase(lru);
        }
        ccache_.emplace_back();
        if (!capture_catchup(t, n, key, ccache_.back(), err)) {
            drop(ccache_.back());
            ccache_.pop_back();
            capture_failed_ = true;
            return false;
        }
        g = &ccache_.back();
    }
    if (g != nullptr) {
        g->used = ++use_clock_;
    }
    // staged: each member's descriptor (eagerly its start flag) and its retained carry's pre-step copy (enqueued
    // before the launch, and kept when the members then catch up alone: draft_catchup takes no second copy)
    target::staged_round sr[fn::xchg::max_lanes];
    for (int32_t k = 0; k < n; ++k) {
        if (!t[k]->joint_catchup_begin(rows[k], sr[k], err)) {
            err = "joint catch-up: member " + std::to_string(k) + ": " + err;
            return false;
        }
    }
    std::string lerr;
    bool launched = true;
    if (eager_) {
        // every round the catch-up's sequence from its start on the lane rings, as a graph's replay
        std::vector<std::unique_ptr<model::pass_card>> lane[2];
        model::catchup_joint j[2];
        model::use_counters u[2];
        for (int32_t c = 0; c < 2 && launched; ++c) {
            launched = cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", lerr) &&
                       cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(st_[c])), "joint catch-up sync", lerr) &&
                       catchup_lanes(t, n, c, lane[c], j[c], lerr);
            u[c] = model::catchup_joint_counters();
        }
        if (!launched) {
            err = lerr; // nothing enqueued: the members catch up alone
            return false;
        }
        model::catchup_joint * const jj[2] = {&j[0], &j[1]};
        launched = model::catchup_joint_pair_enqueue(jj, u, lerr) == model::step_status::ok;
    } else {
        for (int32_t c = 0; c < 2 && launched; ++c) {
            launched =
                cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", lerr) &&
                cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g->exec[c]), static_cast<cudaStream_t>(st_[c])),
                        "joint catch-up graph launch", lerr);
            if (!launched && c == 0) {
                err = lerr; // nothing launched: the members catch up alone
                return false;
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (int32_t k = 0; k < n; ++k) {
        t[k]->joint_release(sr[k]);
    }
    bool clean = true;
    for (int32_t k = 0; k < n; ++k) {
        if (!launched) {
            ok[k] = false; // part of the catch-up enqueued: every member's catch-up fails
            why[k] = "joint catch-up: " + (lerr.empty() ? std::string("enqueue") : lerr);
            clean = false;
            continue;
        }
        sr[k].t1 = t1;
        ok[k] = t[k]->joint_catchup_end(rows[k], sr[k], r[k], why[k]);
        clean = clean && ok[k];
    }
    ++catchup_rounds_;
    if (!clean) {
        std::string e2;
        if (!clear_after_error(e2)) {
            fprintf(stderr, "FNFL joint: the rings' reset after a member's catch-up error failed: %s\n", e2.c_str());
        }
    }
    return true;
}

} // namespace fn::runtime
