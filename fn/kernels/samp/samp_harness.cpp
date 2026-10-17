#include "fn/kernels/samp/samp_harness.h"

#include "fn/kernels/head/head.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

namespace fn::samp::harness {

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "samp: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

card_s::card_s(int32_t d) : dev(d), guard(d) {
    if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) {
        st = nullptr;
    }
}

card_s::~card_s() {
    cudaSetDevice(dev);
    if (st != nullptr) {
        cudaStreamDestroy(st);
    }
}

bool tester::init() {
    using fn::xchg::msg;
    for (int32_t r = 0; r < 2; ++r) {
        card_s & k = *c[r];
        card_view & v = k.v;
        v.card = r;
        v.prm = k.prm.p;
        v.bias = k.bias.p;
        v.counts = k.counts.p;
        v.ring = k.ring.p;
        v.rec = k.rec.p;
        v.q_list = k.ql.p;
        v.q_shard = k.qs.p;
        v.work = k.work.p;
        v.err = rg->err(r);
        for (int32_t q = 0; q < 2; ++q) {
            v.a_own[q] = rg->slot(r, msg::samp_a, q);
            v.a_peer[q] = rg->slot(1 - r, msg::samp_a, q);
            v.b_own[q] = rg->slot(r, msg::samp_b, q);
            v.b_peer[q] = rg->slot(1 - r, msg::samp_b, q);
            v.c_own[q] = rg->slot(r, msg::samp_c, q);
            v.c_peer[q] = rg->slot(1 - r, msg::samp_c, q);
            v.d_own[q] = rg->slot(r, msg::samp_d, q);
            v.d_peer[q] = rg->slot(1 - r, msg::samp_d, q);
        }
        v.x_own = rg->slot(r, msg::samp_x, 0);
        v.x_peer = rg->slot(1 - r, msg::samp_x, 0);
        if (!cuda_ok(cudaSetDevice(k.dev), "set device") || k.st == nullptr || !prepare(v, k.st) ||
            !cuda_ok(cudaMemsetAsync(k.qs.p, 0, k.qs.n * 4, k.st), "q") ||
            !cuda_ok(cudaMemsetAsync(k.ql.p, 0, k.ql.n * sizeof(qlist), k.st), "q list") ||
            !cuda_ok(cudaMemsetAsync(k.gwork.p, 0, k.gwork.n, k.st), "greedy work") ||
            !cuda_ok(cudaStreamSynchronize(k.st), "init")) {
            return false;
        }
    }
    return true;
}

bool tester::set_options(const options & o, const std::vector<int32_t> & led, int32_t window_bins) {
    drop_graphs();
    opt = o;
    ledger = led;
    prm = make_params(o, ring_cap);
    prm.window_bins = window_bins;
    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(c[r]->dev);
        if (!write_params(c[r]->v, prm, o, c[r]->st) || !load_window(c[r]->v, prm.cap, led, c[r]->st)) {
            return false;
        }
    }
    return true;
}

bool tester::put_logits(const std::vector<std::vector<float>> & rows) {
    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(c[r]->dev);
        for (size_t t = 0; t < rows.size(); ++t) {
            if (!cuda_ok(cudaMemcpy(c[r]->logits.p + t * NC, rows[t].data() + size_t(r) * NC, size_t(NC) * 4,
                                    cudaMemcpyHostToDevice),
                         "logits")) {
                return false;
            }
        }
    }
    return true;
}

bool tester::put_q(const std::vector<std::vector<int32_t>> & ids, const std::vector<std::vector<float>> & q,
                   const std::vector<std::vector<float>> & shard) {
    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(c[r]->dev);
        std::vector<qlist> ql(max_drafts);
        for (size_t t = 0; t < ids.size() && t < size_t(max_drafts); ++t) {
            memset(&ql[t], 0, sizeof(qlist));
            ql[t].n = static_cast<int32_t>(std::min<size_t>(ids[t].size(), max_k));
            for (int32_t i = 0; i < ql[t].n; ++i) {
                ql[t].id[i] = ids[t][size_t(i)];
                ql[t].q[i] = q[t][size_t(i)];
            }
        }
        if (!cuda_ok(cudaMemcpy(c[r]->ql.p, ql.data(), ql.size() * sizeof(qlist), cudaMemcpyHostToDevice), "ql")) {
            return false;
        }
        for (size_t t = 0; t < shard.size() && t < size_t(max_drafts); ++t) {
            if (!cuda_ok(cudaMemcpy(c[r]->qs.p + t * NC, shard[t].data() + size_t(r) * NC, size_t(NC) * 4,
                                    cudaMemcpyHostToDevice),
                         "q shard")) {
                return false;
            }
        }
    }
    return true;
}

bool tester::put_desc(const std::vector<dec_in> & in) {
    const size_t n = in.size();
    if (n == 0 || n > size_t(kmax)) {
        return false;
    }
    std::vector<fn::round::desc> d(n);
    static uint64_t round_no = 1;
    for (size_t i = 0; i < n; ++i) {
        memset(&d[i], 0, sizeof(fn::round::desc));
        d[i].round = round_no++;
        d[i].kind = in[i].kind;
        d[i].rows = in[i].rows;
        d[i].n_before = in[i].n_before;
        d[i].budget = in[i].rows;
        for (int32_t t = 0; t < max_rows; ++t) {
            d[i].tok[t] = in[i].tok[t];
            d[i].out[t] = -1;
            d[i].argmax[t] = -1;
        }
    }
    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(c[r]->dev);
        if (!cuda_ok(cudaMemcpy(c[r]->desc.p, d.data(), n * sizeof(fn::round::desc), cudaMemcpyHostToDevice), "desc")) {
            return false;
        }
    }
    return true;
}

bool tester::enqueue_batch(const std::vector<dec_in> & in, int32_t row0, bool discard, int32_t per_card) {
    const size_t n = in.size();
    const bool verify = in[0].kind == static_cast<int32_t>(fn::round::kind::verify);
    const std::vector<stage> ss = stages_of(opt, verify);
    const bool greedy = path_of(opt) == path::greedy;
    const auto mine = [per_card](int32_t r) { return per_card < 0 || per_card == r; };
    for (size_t i = 0; i < n; ++i) {
        const int32_t rows = in[i].rows;
        decide_io io[2];
        for (int32_t r = 0; r < 2; ++r) {
            io[r].logits = c[r]->logits.p + size_t(row0) * NC;
            io[r].n = NC;
            io[r].id0 = r * NC;
            io[r].desc = c[r]->desc.p + i;
        }
        if (greedy) {
            // V1's greedy with T-GREEDY's hash words, then the verdict
            for (int32_t r = 0; r < 2; ++r) {
                if (!mine(r)) {
                    continue;
                }
                cudaSetDevice(loop ? c[0]->dev : c[r]->dev);
                // T-GREEDY's slot rotates by decision: back to back, no pass orders one decision's read before the
                // peer's next write (xchg.h's T-GREEDY chain needs the head's X-HC between rounds)
                fn::head::reduce_args ra = {io[r].logits,
                                            NC,
                                            r * NC,
                                            c[r]->pair.p,
                                            rg->slot(1 - r, fn::xchg::msg::greedy, static_cast<int32_t>(i % 3)),
                                            c[r]->gwork.p};
                ra.hash_words = greedy_hash_words(c[r]->v);
                if (!fn::head::greedy_reduce(rows, ra, st(r))) {
                    fprintf(stderr, "samp: greedy reduce: %s\n", cudaGetErrorString(cudaGetLastError()));
                    return false;
                }
            }
            for (int32_t r = 0; r < 2; ++r) {
                if (!mine(r)) {
                    continue;
                }
                cudaSetDevice(loop ? c[0]->dev : c[r]->dev);
                fn::head::decide_args da = {c[r]->pair.p,
                                            rg->slot(r, fn::xchg::msg::greedy, static_cast<int32_t>(i % 3)),
                                            c[r]->desc.p[i].argmax,
                                            rg->err(r),
                                            nullptr,
                                            r};
                da.hash_words = greedy_hash_words(c[r]->v);
                da.checked = greedy_checked(c[r]->v);
                if (!fn::head::greedy_decide(rows, da, st(r))) {
                    fprintf(stderr, "samp: greedy decide: %s\n", cudaGetErrorString(cudaGetLastError()));
                    return false;
                }
            }
        }
        for (stage s : ss) {
            for (int32_t r = 0; r < 2; ++r) {
                if (!mine(r)) {
                    continue;
                }
                cudaSetDevice(loop ? c[0]->dev : c[r]->dev);
                if (!enqueue(s, rows, c[r]->v, io[r], st(r))) {
                    fprintf(stderr, "samp: enqueue stage %d refused or failed (%s)\n", static_cast<int>(s),
                            cudaGetErrorString(cudaGetLastError()));
                    return false;
                }
            }
        }
        if (discard) {
            for (int32_t r = 0; r < 2; ++r) {
                if (!mine(r)) {
                    continue;
                }
                cudaSetDevice(loop ? c[0]->dev : c[r]->dev);
                if (!discard_record(c[r]->v, st(r))) {
                    return false;
                }
            }
        }
    }
    return true;
}

void tester::drop_graphs() {
    for (int32_t r = 0; r < 2; ++r) {
        if (ge[r] != nullptr) {
            cudaSetDevice(c[r]->dev);
            cudaGraphExecDestroy(ge[r]);
            ge[r] = nullptr;
        }
    }
    ge_key.clear();
}

bool tester::run(const std::vector<dec_in> & in, std::vector<fn::round::desc> out[2], int32_t row0, bool discard) {
    const size_t n = in.size();
    if (!put_desc(in)) {
        return false;
    }
    if (!graphs || loop) {
        if (!enqueue_batch(in, row0, discard) || !sync()) {
            return false;
        }
    } else {
        // the batch's shape: its size, logits row, discard, kind and rows, and the options' stages; the descriptors'
        // contents (positions, drafts, round numbers) are read on the device, so one capture serves every batch
        std::vector<int32_t> key = {static_cast<int32_t>(n), row0, discard ? 1 : 0, in[0].kind, in[0].rows};
        for (stage st : stages_of(opt, in[0].kind == static_cast<int32_t>(fn::round::kind::verify))) {
            key.push_back(static_cast<int32_t>(st));
        }
        key.push_back(static_cast<int32_t>(path_of(opt)));
        if (key != ge_key) {
            drop_graphs();
            for (int32_t r = 0; r < 2; ++r) {
                cudaSetDevice(c[r]->dev);
                cudaGraph_t g = nullptr;
                if (!cuda_ok(cudaStreamBeginCapture(c[r]->st, cudaStreamCaptureModeThreadLocal), "capture")) {
                    return false;
                }
                const bool enq = enqueue_batch(in, row0, discard, r);
                const bool end = cuda_ok(cudaStreamEndCapture(c[r]->st, &g), "end capture");
                // uploaded before the first launch (as TG1's driver does), so neither card's first decision waits on
                // the other's graph upload past the poll budget
                const bool inst = enq && end && cuda_ok(cudaGraphInstantiate(&ge[r], g, 0), "instantiate") &&
                                  cuda_ok(cudaGraphUpload(ge[r], c[r]->st), "upload") &&
                                  cuda_ok(cudaStreamSynchronize(c[r]->st), "upload");
                if (g != nullptr) {
                    cudaGraphDestroy(g);
                }
                if (!inst) {
                    return false;
                }
            }
            ge_key = key;
        }
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(c[r]->dev);
            if (!cuda_ok(cudaGraphLaunch(ge[r], c[r]->st), "graph launch")) {
                return false;
            }
        }
        if (!sync()) {
            return false;
        }
    }
    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(c[r]->dev);
        out[r].resize(n);
        if (!cuda_ok(cudaMemcpy(out[r].data(), c[r]->desc.p, n * sizeof(fn::round::desc), cudaMemcpyDeviceToHost),
                     "desc out")) {
            return false;
        }
    }
    return true;
}

bool tester::sync() const {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        bool done = true;
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(c[r]->dev);
            const cudaError_t e = cudaStreamQuery(c[r]->st);
            if (e == cudaErrorNotReady) {
                done = false;
            } else if (e != cudaSuccess) {
                return cuda_ok(e, "stream");
            }
        }
        if (done) {
            return true;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 120.0) {
            fprintf(stderr, "samp-test: host timeout\n");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

bool tester::errs(uint32_t e[2]) {
    for (int32_t r = 0; r < 2; ++r) {
        if (!rg->read_err(r, e[r])) {
            return false;
        }
        cudaSetDevice(c[r]->dev);
        if (!cuda_ok(cudaMemset(rg->err(r), 0, 4), "clear err")) {
            return false;
        }
    }
    return true;
}

} // namespace fn::samp::harness
