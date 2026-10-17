#include "fn/runtime/layer_run.h"

#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/round/round.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace fn::runtime_layers {

namespace {

constexpr int32_t F = 1280;
static_assert(fn::qsa::max_row_words == 517 && fn::xchg::select_row_words == 517, "layer_record's selection body");

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

cudaStream_t stream(void * s) {
    return static_cast<cudaStream_t>(s);
}

int32_t part_of(const model::reserved_class * rc, const std::string & name) {
    for (int32_t p = 0; rc != nullptr && p < (int32_t)rc->plan.parts.size(); ++p) {
        if (rc->plan.parts[p].name == name) {
            return p;
        }
    }
    return -1;
}

// layer-bench's observer: an event after every piece, on the card's stream (the phase made its device current)
struct event_log {
    struct mark {
        cudaEvent_t ev;
        int32_t layer;
        const char * piece;
    };
    std::vector<mark> marks[2];
    cudaEvent_t start[2] = {nullptr, nullptr};
    void * stream[2] = {nullptr, nullptr};
    int32_t layer = 0;
    bool ok = true;
};

void observe_event(void * user, int32_t card, const char * piece) {
    event_log & lg = *static_cast<event_log *>(user);
    cudaEvent_t ev = nullptr;
    if (cudaEventCreate(&ev) != cudaSuccess) {
        lg.ok = false;
        return;
    }
    lg.marks[card].push_back({ev, lg.layer, piece});
    lg.ok = lg.ok && cudaEventRecord(ev, static_cast<cudaStream_t>(lg.stream[card])) == cudaSuccess;
}

} // namespace

pair_runner::~pair_runner() {
    drop_graphs();
    for (int32_t c = 0; c < 2; ++c) {
        if (m_ == nullptr) {
            break;
        }
        cudaSetDevice(m_->device[c]);
        if (st_[c] != nullptr) {
            cudaStreamSynchronize(stream(st_[c]));
        }
        for (int32_t k = 0; k < 2; ++k) {
            cudaFree(in_dev_[k][c]);
            if (ev_[c][k] != nullptr) {
                cudaEventDestroy(static_cast<cudaEvent_t>(ev_[c][k]));
            }
        }
        cudaFreeHost(in_host_[c]);
        cudaFreeHost(pos_host_[c]);
        if (st_[c] != nullptr) {
            cudaStreamDestroy(stream(st_[c]));
        }
    }
}

bool pair_runner::init(model::model & m, int32_t layer0, int32_t layer1, std::string & err) {
    m_ = &m;
    layer_[0] = layer0;
    layer_[1] = layer1;
    // DP2's packed layout for the layers' dense weights (once per loaded model)
    if (!model::pack_dense(m, layer0, err) || !model::pack_dense(m, layer1, err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        cudaStream_t s = nullptr;
        if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream", err)) {
            return false;
        }
        st_[c] = s;
        if (!model::resolve_card(m, c, s, cc_[c], err)) {
            return false;
        }
        // the layers read the depth and positions on the device: this runner writes them before each step (upload)
        cc_[c].n_before = cc_[c].ws.depth;
        cc_[c].pos = cc_[c].ws.pos;
        cc_[c].cur = cc_[c].ws.cur; // SR1: this runner's committed slot, advanced by each step's commit
        for (int32_t k = 0; k < 2; ++k) {
            if (!model::resolve_weights(m, layer_[k], c, w_[k][c], err) ||
                !model::resolve_state(m, layer_[k], c, s_[k][c], err)) {
                return false;
            }
            cudaEvent_t e = nullptr;
            if (!cuda_ok(cudaMalloc(&in_dev_[k][c], size_t(max_rows) * 4 * F * 4), "input", err) ||
                !cuda_ok(cudaEventCreate(&e), "event", err)) {
                return false;
            }
            ev_[c][k] = e;
        }
        if (!cuda_ok(cudaMallocHost(&in_host_[c], size_t(2) * max_rows * 4 * F * 4), "pinned input", err) ||
            !cuda_ok(cudaMallocHost(&pos_host_[c], size_t(4) * max_rows * 4 + 4), "pinned positions", err)) {
            return false;
        }
        if (!cuda_ok(cudaMemset(cc_[c].ws.zero, 0, size_t(max_rows) * F * 4), "zero", err)) {
            return false;
        }
    }
    return reset(err);
}

bool pair_runner::reset(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(stream(st_[c])), "sync", err)) {
            return false;
        }
        for (int32_t k = 0; k < 2; ++k) {
            const model::layer_state & s = s_[k][c];
            const std::string L = "L" + std::to_string(layer_[k]);
            const auto clear = [&](const char * cls, const std::string & part) {
                const model::reserved_class * rc = m_->state_class(cls);
                const int32_t p = part_of(rc, part);
                if (p < 0 || rc->plan.parts[p].bytes[c] == 0) {
                    return true;
                }
                return cuda_ok(cudaMemset(rc->part_ptr(p, c), 0, (size_t)rc->plan.parts[p].bytes[c]), cls, err);
            };
            bool ok = true;
            if (w_[k][c].gdn) {
                ok = clear("GDN state", L) && clear("convolution histories", L + " GDN");
            } else {
                ok = clear("target KV (K0)", L + " K") && clear("target KV (K0)", L + " V") &&
                     clear("target index keys and summaries", L + " keys") &&
                     clear("target index keys and summaries", L + " summaries");
            }
            (void)s;
            if (!ok) {
                return false;
            }
        }
        if (!model::clear_workspaces(*m_, c, err) || !cuda_ok(cudaMemset(cc_[c].ws.cur, 0, 4), "cur", err) ||
            !cuda_ok(cudaDeviceSynchronize(), "reset", err)) {
            return false;
        }
        u_[c] = model::use_counters();
    }
    if (!m_->rings->reset_all()) {
        err = "rings reset";
        return false;
    }
    return true;
}

bool pair_runner::fill_synthetic(int32_t depth, uint64_t seed, std::string & err) {
    uint64_t x = seed * 6364136223846793005ull + 1442695040888963407ull;
    const auto next = [&]() { // in [-1, 1)
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>(static_cast<double>(x >> 11) * (2.0 / 9007199254740992.0) - 1.0);
    };
    const auto bf16 = [](float v) { // truncation is enough for synthetic values
        uint32_t u = 0;
        std::memcpy(&u, &v, 4);
        return static_cast<uint16_t>(u >> 16);
    };
    std::vector<uint16_t> h16;
    std::vector<float> h32;
    const auto put16 = [&](uint16_t * dst, size_t n, float scale) {
        h16.resize(n);
        for (uint16_t & v : h16) {
            v = bf16(scale * next());
        }
        return cuda_ok(cudaMemcpy(dst, h16.data(), n * 2, cudaMemcpyHostToDevice), "synthetic bf16", err);
    };
    const auto put32 = [&](float * dst, size_t n, float scale) {
        h32.resize(n);
        for (float & v : h32) {
            v = scale * next();
        }
        return cuda_ok(cudaMemcpy(dst, h32.data(), n * 4, cudaMemcpyHostToDevice), "synthetic f32", err);
    };
    if (!sync(err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err)) {
            return false;
        }
        for (int32_t k = 0; k < 2; ++k) {
            const model::layer_state & st = s_[k][c];
            const model::layer_weights & w = w_[k][c];
            bool ok = true;
            if (w.gdn) {
                ok = put32(st.gdn, size_t(24) * 128 * 128, 0.05f) && put32(st.conv, size_t(5120) * 3, 1.0f);
            } else {
                if (depth > st.n_cells) {
                    err = "synthetic depth beyond the reserved cells";
                    return false;
                }
                ok = put16(st.k_cache, size_t(depth) * 256, 1.0f) && put16(st.v_cache, size_t(depth) * 256, 1.0f);
                if (ok && c == w.index_owner) {
                    const int32_t nb = fn::qsa::blocks_complete(depth);
                    ok = put16(st.idx_keys, size_t(depth) * fn::qsa::head_dim, 1.0f) &&
                         put32(st.idx_summaries, fn::qsa::summary_bytes(nb) / 4, 1.0f);
                }
            }
            if (!ok) {
                return false;
            }
        }
    }
    return reset_exchange(err);
}

bool pair_runner::reset_exchange(std::string & err) {
    if (!sync(err)) {
        return false;
    }
    if (!m_->rings->reset_all()) {
        err = "rings reset";
        return false;
    }
    u_[0] = model::use_counters();
    u_[1] = model::use_counters();
    return true;
}

bool pair_runner::upload(int32_t rows, int32_t depth, const float * const in0[2], const float * const in1[2],
                         std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(stream(st_[c])), "staging", err)) {
            return false;
        }
        const size_t n = size_t(rows) * 4 * F;
        std::memcpy(in_host_[c], in0[c], n * 4);
        if (in1 != nullptr) {
            std::memcpy(in_host_[c] + size_t(max_rows) * 4 * F, in1[c], n * 4);
        }
        for (int32_t s4 = 0; s4 < 4; ++s4) {
            for (int32_t b = 0; b < rows; ++b) {
                pos_host_[c][s4 * rows + b] = depth + b;
            }
        }
        pos_host_[c][size_t(4) * max_rows] = depth;
        if (!cuda_ok(cudaMemcpyAsync(in_dev_[0][c], in_host_[c], n * 4, cudaMemcpyHostToDevice, stream(st_[c])), "in0",
                     err) ||
            (in1 != nullptr && !cuda_ok(cudaMemcpyAsync(in_dev_[1][c], in_host_[c] + size_t(max_rows) * 4 * F, n * 4,
                                                        cudaMemcpyHostToDevice, stream(st_[c])),
                                        "in1", err)) ||
            !cuda_ok(cudaMemcpyAsync(cc_[c].ws.pos, pos_host_[c], size_t(4) * rows * 4, cudaMemcpyHostToDevice,
                                     stream(st_[c])),
                     "pos", err) ||
            !cuda_ok(cudaMemcpyAsync(cc_[c].ws.depth, pos_host_[c] + size_t(4) * max_rows, 4, cudaMemcpyHostToDevice,
                                     stream(st_[c])),
                     "depth", err)) {
            return false;
        }
    }
    return true;
}

bool pair_runner::sync(std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m_->device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamSynchronize(stream(st_[c])), "step", err)) {
            err = "card " + std::to_string(c) + ": " + err;
            return false;
        }
    }
    return true;
}

bool pair_runner::step(int32_t rows, int32_t depth, const float * const in0[2], const float * const in1[2],
                       layer_record * const rec[2][2], std::string & err) {
    not_built_ = false;
    if (!upload(rows, depth, in0, in1, err)) {
        return false;
    }
    model::card_ctx * cp[2] = {&cc_[0], &cc_[1]};
    for (int32_t k = 0; k < 2; ++k) {
        const model::layer_weights * wp[2] = {&w_[k][0], &w_[k][1]};
        const model::layer_state * sp[2] = {&s_[k][0], &s_[k][1]};
        model::layer_input in[2];
        for (int32_t c = 0; c < 2; ++c) {
            if (k == 1 && in1 == nullptr) {
                in[c] = model::layer_output(cc_[c]);
            } else {
                in[c].hres = in_dev_[k][c];
                in[c].hpost = cc_[c].ws.zero;
                in[c].hx = cc_[c].ws.zero;
            }
        }
        // the mixed inputs are recorded by the layer after each body, before the next HC sublayer resets their X-HMIX
        // slot (HC3; card_ctx::record_mixed)
        for (int32_t c = 0; c < 2; ++c) {
            layer_record * r = rec != nullptr ? rec[k][c] : nullptr;
            cc_[c].record_mixed[0] = r != nullptr ? r->mixed_attn : nullptr;
            cc_[c].record_mixed[1] = r != nullptr ? r->mixed_ffn : nullptr;
        }
        // each phase makes its card's device current (fn/model/layer.cpp)
        const model::step_status ss = model::layer_pair_step(cp, wp, sp, in, rows, u_, err);
        for (int32_t c = 0; c < 2; ++c) {
            cc_[c].record_mixed[0] = cc_[c].record_mixed[1] = nullptr;
        }
        if (ss != model::step_status::ok) {
            not_built_ = ss == model::step_status::not_built;
            return false;
        }
        // the records, in stream order before the next layer overwrites the workspace
        for (int32_t c = 0; c < 2; ++c) {
            layer_record * r = rec != nullptr ? rec[k][c] : nullptr;
            if (r == nullptr) {
                continue;
            }
            cudaSetDevice(m_->device[c]);
            const model::layer_ws & ws = cc_[c].ws;
            cudaStream_t s = stream(st_[c]);
            const model::layer_weights & w = w_[k][c];
            bool ok = cuda_ok(cudaMemcpyAsync(r->x, ws.x[1], size_t(rows) * 4 * F * 4, cudaMemcpyDeviceToHost, s), "x",
                              err) &&
                      cuda_ok(cudaMemcpyAsync(r->inject, ws.inject[1], size_t(rows) * 4 * 4, cudaMemcpyDeviceToHost, s),
                              "inj", err) &&
                      cuda_ok(cudaMemcpyAsync(r->out_ffn, ws.out[1], size_t(rows) * F * 4, cudaMemcpyDeviceToHost, s),
                              "out1", err) &&
                      cuda_ok(cudaMemcpyAsync(r->out_attn, ws.out[0], size_t(rows) * F * 4, cudaMemcpyDeviceToHost, s),
                              "out0", err) &&
                      cuda_ok(cudaMemcpyAsync(r->ids, cc_[c].ex.ids, size_t(rows) * 10 * 4, cudaMemcpyDeviceToHost, s),
                              "ids", err) &&
                      cuda_ok(cudaMemcpyAsync(r->wts, cc_[c].ex.wts, size_t(rows) * 10 * 4, cudaMemcpyDeviceToHost, s),
                              "wts", err) &&
                      cuda_ok(cudaMemcpyAsync(r->logits, ws.logits, size_t(rows) * 512 * 4, cudaMemcpyDeviceToHost, s),
                              "logits", err) &&
                      cuda_ok(cudaMemcpyAsync(&r->ex_status, cc_[c].ex.status, 4, cudaMemcpyDeviceToHost, s), "status",
                              err) &&
                      cuda_ok(cudaMemcpyAsync(&r->err, m_->rings->err(c), 4, cudaMemcpyDeviceToHost, s), "err", err);
            r->attn_err = 0;
            r->sel_err = 0;
            if (ok && w.gdn) {
                ok =
                    cuda_ok(cudaMemcpyAsync(r->gdn_conv, ws.conv, size_t(rows) * 5120 * 4, cudaMemcpyDeviceToHost, s),
                            "conv", err) &&
                    cuda_ok(cudaMemcpyAsync(r->gdn_z, ws.z, size_t(rows) * 3072 * 4, cudaMemcpyDeviceToHost, s), "z",
                            err) &&
                    cuda_ok(cudaMemcpyAsync(r->gdn_alpha, ws.alpha, size_t(rows) * 24 * 4, cudaMemcpyDeviceToHost, s),
                            "alpha", err) &&
                    cuda_ok(cudaMemcpyAsync(r->gdn_beta, ws.beta, size_t(rows) * 24 * 4, cudaMemcpyDeviceToHost, s),
                            "beta", err) &&
                    cuda_ok(cudaMemcpyAsync(r->gdn_gated, ws.gated, size_t(rows) * 3072 * 4, cudaMemcpyDeviceToHost, s),
                            "gated", err);
            }
            if (ok && !w.gdn) {
                const fn::qsa::attn::workspace aw = fn::qsa::attn::carve(cc_[c].attn_ws);
                ok = cuda_ok(cudaMemcpyAsync(&r->attn_err, aw.error, 4, cudaMemcpyDeviceToHost, s), "attn err", err);
                if (ok && c == w.index_owner) {
                    const fn::qsa::workspace qw = fn::qsa::carve(cc_[c].qsa_ws);
                    ok = cuda_ok(cudaMemcpyAsync(&r->sel_err, qw.error, 4, cudaMemcpyDeviceToHost, s), "sel err", err);
                }
            }
            r->sel_words = 0;
            if (ok && !w.gdn) {
                // both cards hold the body: the index owner's from select, the peer's from X-SELECT
                r->sel_words = fn::qsa::body_words(depth, rows);
                ok = cuda_ok(
                         cudaMemcpyAsync(r->sel_body, ws.sel_body, size_t(r->sel_words) * 4, cudaMemcpyDeviceToHost, s),
                         "sel_body", err) &&
                     cuda_ok(cudaMemcpyAsync(r->sel_off, ws.sel_off, size_t(rows + 1) * 4, cudaMemcpyDeviceToHost, s),
                             "sel_off", err);
            }
            if (!ok) {
                return false;
            }
        }
    }
    // the commit (SR1): both layers' versioned state share the card's one committed-slot word
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        if (!fn::round::advance(cc_[c].ws.cur, rows, st_[c])) {
            return cuda_ok(cudaGetLastError(), "commit", err);
        }
    }
    return sync(err);
}

bool pair_runner::step_timed(int32_t rows, int32_t depth, const float * const in0[2], std::vector<piece_time> & out,
                             std::string & err) {
    not_built_ = false;
    if (!upload(rows, depth, in0, nullptr, err)) {
        return false;
    }
    auto lgp = std::make_unique<event_log>(); // the observer holds it while the step is enqueued
    event_log & lg = *lgp;
    model::card_ctx * cp[2] = {&cc_[0], &cc_[1]};
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        lg.stream[c] = st_[c];
        if (!cuda_ok(cudaEventCreate(&lg.start[c]), "event", err) ||
            !cuda_ok(cudaEventRecord(lg.start[c], stream(st_[c])), "record", err)) {
            return false;
        }
        cc_[c].observe = observe_event;
        cc_[c].observe_user = lgp.get();
    }
    bool ok = true;
    for (int32_t k = 0; k < 2 && ok; ++k) {
        lg.layer = layer_[k];
        const model::layer_weights * wp[2] = {&w_[k][0], &w_[k][1]};
        const model::layer_state * sp[2] = {&s_[k][0], &s_[k][1]};
        model::layer_input in[2];
        for (int32_t c = 0; c < 2; ++c) {
            if (k == 0) {
                in[c].hres = in_dev_[0][c];
                in[c].hpost = cc_[c].ws.zero;
                in[c].hx = cc_[c].ws.zero;
            } else {
                in[c] = model::layer_output(cc_[c]);
            }
        }
        const model::step_status ss = model::layer_pair_step(cp, wp, sp, in, rows, u_, err);
        not_built_ = ss == model::step_status::not_built;
        ok = ss == model::step_status::ok;
    }
    for (int32_t c = 0; c < 2; ++c) {
        cc_[c].observe = nullptr;
        cc_[c].observe_user = nullptr;
    }
    ok = ok && lg.ok && sync(err);
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        cudaEvent_t prev = lg.start[c];
        for (const event_log::mark & mk : lg.marks[c]) {
            float ms = 0.0f;
            if (ok && cudaEventElapsedTime(&ms, prev, mk.ev) == cudaSuccess) {
                out.push_back({c, mk.layer, mk.piece, ms});
            } else {
                ok = false;
            }
            prev = mk.ev;
        }
        for (const event_log::mark & mk : lg.marks[c]) {
            cudaEventDestroy(mk.ev);
        }
        cudaEventDestroy(lg.start[c]);
    }
    if (!ok && err.empty()) {
        err = "timed step";
    }
    return ok;
}

bool pair_runner::capture(int32_t rows, int32_t depth, const float * const in0[2], std::string & err) {
    not_built_ = false;
    drop_graphs();
    if (!upload(rows, depth, in0, nullptr, err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        cudaStream_t s = stream(st_[c]);
        if (!cuda_ok(cudaStreamSynchronize(s), "sync", err) ||
            !cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture", err)) {
            return false;
        }
        model::use_counters u;
        u.graph = true;
        u.graph_subs = 4;
        bool ok = true;
        for (int32_t k = 0; k < 2 && ok; ++k) {
            model::layer_input in;
            if (k == 0) {
                in.hres = in_dev_[0][c];
                in.hpost = cc_[c].ws.zero;
                in.hx = cc_[c].ws.zero;
            } else {
                in = model::layer_output(cc_[c]);
            }
            const model::step_status ss = model::layer_card_step(cc_[c], w_[k][c], s_[k][c], in, rows, u, err);
            not_built_ = ss == model::step_status::not_built;
            ok = ss == model::step_status::ok;
        }
        cudaGraph_t g = nullptr;
        const cudaError_t e = cudaStreamEndCapture(s, &g);
        if (!ok || !cuda_ok(e, "end capture", err)) {
            if (g != nullptr) {
                cudaGraphDestroy(g);
            }
            return false;
        }
        graph_[c] = g;
        cudaGraphExec_t x = nullptr;
        if (!cuda_ok(cudaGraphInstantiate(&x, g, 0), "instantiate", err) ||
            !cuda_ok(cudaGraphUpload(x, s), "upload", err) || !cuda_ok(cudaStreamSynchronize(s), "upload sync", err)) {
            return false;
        }
        exec_[c] = x;
    }
    return true;
}

bool pair_runner::replay(float ms[2], std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        cudaStream_t s = stream(st_[c]);
        if (exec_[c] == nullptr || !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_[c][0]), s), "record", err) ||
            !cuda_ok(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec_[c]), s), "launch", err) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_[c][1]), s), "record", err)) {
            return false;
        }
    }
    if (!sync(err)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(
                cudaEventElapsedTime(&ms[c], static_cast<cudaEvent_t>(ev_[c][0]), static_cast<cudaEvent_t>(ev_[c][1])),
                "elapsed", err)) {
            return false;
        }
    }
    return true;
}

void pair_runner::drop_graphs() {
    for (int32_t c = 0; c < 2; ++c) {
        if (exec_[c] != nullptr) {
            cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_[c]));
            exec_[c] = nullptr;
        }
        if (graph_[c] != nullptr) {
            cudaGraphDestroy(static_cast<cudaGraph_t>(graph_[c]));
            graph_[c] = nullptr;
        }
    }
}

} // namespace fn::runtime_layers
