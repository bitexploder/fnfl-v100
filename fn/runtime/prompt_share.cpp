#include "fn/runtime/prompt_share.h"

#include "fn/kernels/experts/experts_share.h"
#include "fn/runtime/cache_service.h"

#include <cuda_runtime.h>
#include <pthread.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace fn::runtime {

namespace {

constexpr int32_t D = 2560;
constexpr int64_t row16 = int64_t(D) * 2; // bytes of an F16 row

cudaEvent_t ev(void * e) {
    return static_cast<cudaEvent_t>(e);
}
cudaStream_t st(void * s) {
    return static_cast<cudaStream_t>(s);
}

} // namespace

bool prompt_share::init(const cache::devices & dv, const residency & r, const options & o, std::string & err) {
    o_ = o;
    dv_ = &dv;
    res_ = &r;
    if (!(o.share == -1.0 || (o.share >= 0.0 && o.share <= 1.0)) || o.max_rows < pplan::share_max_rows ||
        o.max_rows > cache::max_prompt_rows || o.maxt < 1 || o.maxt > cache::max_prompt_rows) {
        return cache::fail(err, "prompt CPU share: the share is auto, 0 or in (0, 1], its limit 1024..4096 rows, its "
                                "pair bound 1..4096");
    }
    if (dv.nd != 1 || o.share == 0.0) {
        return true; // two cards never share (MX7's boundary); 0: nothing changes
    }
    native_ = r.native_ready();
    const int64_t rows = o.max_rows, b[] = {rows * row16, int64_t(max_pairs) * row16, int64_t(max_pairs) * 4, rows * 4,
                                            int64_t(cache::n_ex)};
    void ** p[] = {reinterpret_cast<void **>(&x16_d_), reinterpret_cast<void **>(&out16_d_),
                   reinterpret_cast<void **>(&pairs_d_), reinterpret_cast<void **>(&rows_d_),
                   reinterpret_cast<void **>(&mask_d_)};
    if (!dv.set(0, err)) {
        return false;
    }
    for (size_t i = 0; i < 5; ++i) {
        if (!cache::cuda_ok(cudaMalloc(p[i], size_t(b[i])), "prompt CPU share buffers", err)) {
            return false;
        }
        dev_bytes_ += b[i];
    }
    cudaStream_t s = nullptr;
    cudaEvent_t e[5] = {};
    if (!cache::cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "prompt CPU share stream", err)) {
        return false;
    }
    side_ = s;
    if (!cache::cuda_ok(cudaEventCreateWithFlags(&e[0], cudaEventDisableTiming), "prompt CPU share events", err)) {
        return false;
    }
    ev_ready_ = e[0];
    if (!cache::cuda_ok(cudaEventCreateWithFlags(&e[1], cudaEventDisableTiming | cudaEventBlockingSync),
                        "prompt CPU share events", err)) {
        return false;
    }
    ev_x_ = e[1];
    for (int32_t i = 0; i < 3; ++i) {
        if (!cache::cuda_ok(cudaEventCreate(&e[2 + i]), "prompt CPU share events", err)) {
            return false;
        }
        ev_t_[i] = e[2 + i];
    }
    armed_ = true;
    return true;
}

bool prompt_share::start(window_service & w, std::string & err) {
    (void)err;
    if (!armed_) {
        return true;
    }
    if (w.workers() == 0) {
        armed_ = false;
        std::fprintf(stderr, "prompt CPU share: off (no CPU workers)\n");
        return true;
    }
    pool_ = &w.cpu_pool();
    cpu_ = w.service_cpu();
    th_ = std::thread([this] { worker(); });
    return true;
}

void prompt_share::release() {
    if (th_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        th_.join();
    }
    if (dv_ != nullptr && (x16_d_ != nullptr || side_ != nullptr)) {
        cudaSetDevice(dv_->dev[0]);
    }
    for (void * q : {static_cast<void *>(x16_d_), static_cast<void *>(out16_d_), static_cast<void *>(pairs_d_),
                     static_cast<void *>(rows_d_), static_cast<void *>(mask_d_)}) {
        if (q != nullptr) {
            cudaFree(q);
        }
    }
    for (void * q : {static_cast<void *>(x16_h_), static_cast<void *>(out16_h_), static_cast<void *>(pairs_h_),
                     static_cast<void *>(rows_h_), static_cast<void *>(mask_h_)}) {
        if (q != nullptr) {
            cudaFreeHost(q);
        }
    }
    x16_d_ = out16_d_ = x16_h_ = out16_h_ = nullptr;
    pairs_d_ = rows_d_ = pairs_h_ = rows_h_ = nullptr;
    mask_d_ = mask_h_ = nullptr;
    for (void ** e : {&ev_ready_, &ev_x_, &ev_t_[0], &ev_t_[1], &ev_t_[2]}) {
        if (*e != nullptr) {
            cudaEventDestroy(ev(*e));
            *e = nullptr;
        }
    }
    if (side_ != nullptr) {
        cudaStreamDestroy(st(side_));
        side_ = nullptr;
    }
    armed_ = false;
    dev_bytes_ = host_bytes_ = 0;
    cap_rows_ = cap_pairs_ = 0;
}

bool prompt_share::set_share(double share, std::string & err) {
    if (!(share == -1.0 || (share >= 0.0 && share <= 1.0)) || (share != 0.0 && !armed_)) {
        return cache::fail(err, "prompt CPU share: set_share needs auto, 0 or (0, 1], and a share armed at init");
    }
    join();
    o_.share = share;
    c_ = g_ = 0;
    nc_ = ng_ = 0;
    share_now_ = 0.5;
    check_ = pplan::share_check();
    pend_ = pending();
    cur_ = layer_plan();
    return true;
}

void prompt_share::off(const char * what) {
    dead_ = true;
    ++n_.pshare_dead;
    std::fprintf(stderr,
                 "prompt CPU share: off for this run: the host could not give %s (RAM is short); the GPU takes every "
                 "expert\n",
                 what);
}

// the pinned host buffers for rows gathered rows and pairs CPU pairs, grown when short (the previous layer's job
// joined, its copies done: each layer's pick follows its routing sync, after the previous layer's every copy)
bool prompt_share::host_buffers(int32_t rows, int32_t pairs) {
    if (mask_h_ == nullptr) {
        void * m = nullptr;
        void * pr = nullptr;
        void * rw = nullptr;
        if (cudaHostAlloc(&m, size_t(cache::n_ex), cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&pr, size_t(max_pairs) * 4, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&rw, size_t(o_.max_rows) * 4, cudaHostAllocDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            for (void * q : {m, pr, rw}) {
                if (q != nullptr) {
                    cudaFreeHost(q);
                }
            }
            off("the share's index buffers");
            return false;
        }
        mask_h_ = static_cast<uint8_t *>(m);
        pairs_h_ = static_cast<int32_t *>(pr);
        rows_h_ = static_cast<int32_t *>(rw);
        host_bytes_ += int64_t(cache::n_ex) + int64_t(max_pairs) * 4 + int64_t(o_.max_rows) * 4;
    }
    const auto grow = [&](uint16_t *& h, int32_t & cap, int32_t want, int32_t hi, const char * what) {
        if (cap >= want) {
            return true;
        }
        const int32_t n = std::min(hi, std::max(want, 2 * cap)); // headroom: few reallocations
        if (h != nullptr) {
            cudaFreeHost(h);
            host_bytes_ -= int64_t(cap) * row16;
            h = nullptr;
            cap = 0;
        }
        void * q = nullptr;
        if (cudaHostAlloc(&q, size_t(n) * size_t(row16), cudaHostAllocDefault) != cudaSuccess) {
            (void)cudaGetLastError();
            off(what);
            return false;
        }
        h = static_cast<uint16_t *>(q);
        cap = n;
        host_bytes_ += int64_t(n) * row16;
        return true;
    };
    return grow(x16_h_, cap_rows_, rows, o_.max_rows, "the activation buffer") &&
           grow(out16_h_, cap_pairs_, pairs, max_pairs, "the rows buffer");
}

void prompt_share::join() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] { return done_; });
}

int32_t prompt_share::pick(int32_t layer, int32_t rows, const int32_t * ids, const int32_t * counts,
                           const uint8_t * resident, void * const stream[2], uint8_t * cpu) {
    std::memset(cpu, 0, size_t(cache::n_ex));
    join(); // a job left by a failed layer
    // the last eligible layer's measurement: its events are complete (this layer's routing sync followed them)
    if (pend_.valid) {
        pend_.valid = false;
        float m01 = 0, m02 = 0;
        if (cudaEventElapsedTime(&m01, ev(ev_t_[0]), ev(ev_t_[1])) == cudaSuccess &&
            cudaEventElapsedTime(&m02, ev(ev_t_[0]), ev(ev_t_[2])) == cudaSuccess) {
            if (pend_.on) {
                pplan::share_mean(c_, nc_, pend_.cpu_ms / double(pend_.n_cpu));
                if (pend_.staged > 0) {
                    pplan::share_mean(g_, ng_, double(m01) / double(pend_.staged));
                }
            }
            check_.record(pend_.on, double(m02) / double(pend_.nonres));
            share_now_ = pplan::share_of(c_, g_);
        } else {
            (void)cudaGetLastError();
        }
    }
    cur_ = layer_plan();
    if (layer == 0) {
        chunk_shared_ = false;
    }
    if (!armed() || ids == nullptr || rows < 1 || rows >= o_.max_rows || stream[0] != stream[1]) {
        return 0;
    }
    int32_t nonres = 0;
    for (int32_t e = 0; e < cache::n_ex; ++e) {
        nonres += counts[e] > 0 && resident[e] == 0 ? 1 : 0;
    }
    if (nonres == 0) {
        return 0;
    }
    const bool fixed = o_.share > 0;
    const bool on = fixed || check_.next();
    cur_.layer = layer;
    cur_.nonres = nonres;
    cur_.eligible = !fixed; // auto measures its layers either way; a fixed share only its sharing ones
    if (!on) {
        ++n_.pshare_off_layers;
        return 0;
    }
    int32_t pairs = 0;
    const double x = fixed ? o_.share : share_now_;
    const int32_t n = pplan::share_pick(counts, resident, cache::n_ex, o_.maxt, x, max_pairs, cpu, nonres, pairs);
    // the CPU experts' token rows (ascending: the compact order) and their pairs (by expert, then pair)
    std::vector<int32_t> compact(size_t(rows), -1);
    int32_t n_rows = 0;
    for (int32_t p = 0; n > 0 && p < rows * 10; ++p) {
        const int32_t e = ids[p];
        if (e >= 0 && e < cache::n_ex && cpu[e] != 0) {
            compact[size_t(p / 10)] = 0;
        }
    }
    for (int32_t t = 0; t < rows; ++t) {
        if (compact[size_t(t)] == 0) {
            compact[size_t(t)] = n_rows++;
        }
    }
    if (n == 0 || !host_buffers(n_rows, pairs)) {
        std::memset(cpu, 0, size_t(cache::n_ex));
        cur_.eligible = !fixed && !dead_;
        if (!cur_.eligible) {
            cur_.layer = -1;
        }
        ++n_.pshare_off_layers;
        return 0;
    }
    for (int32_t t = 0; t < rows; ++t) {
        if (compact[size_t(t)] >= 0) {
            rows_h_[compact[size_t(t)]] = t;
        }
    }
    const loader::expert_layer & L = res_->layer(layer);
    gu_ = L.gu_fmt;
    dn_ = L.dn_fmt;
    if (out32_.size() < size_t(pairs) * D) {
        out32_.resize(size_t(pairs) * D);
    }
    // each CPU expert's pairs in pair order (distinct rows: top-10 of distinct experts), the experts ascending; a job
    // takes at most job_max_rows of them, so an expert of more pairs (only under run-dry's test bound) takes several
    std::vector<int32_t> first(size_t(cache::n_ex), 0), fill(size_t(cache::n_ex), 0);
    for (int32_t p = 0; p < rows * 10; ++p) {
        const int32_t e = ids[p];
        if (e >= 0 && e < cache::n_ex && cpu[e] != 0) {
            ++fill[size_t(e)];
        }
    }
    int32_t nj = 0, at = 0;
    for (int32_t e = 0; e < cache::n_ex; ++e) {
        mask_h_[e] = cpu[e];
        first[size_t(e)] = at; // the expert's first pair index
        at += fill[size_t(e)];
        nj += (fill[size_t(e)] + fn::experts::cpu::job_max_rows - 1) / fn::experts::cpu::job_max_rows;
        fill[size_t(e)] = 0;
    }
    for (int32_t p = 0; p < rows * 10; ++p) {
        const int32_t e = ids[p];
        if (e >= 0 && e < cache::n_ex && cpu[e] != 0) {
            pairs_h_[first[size_t(e)] + fill[size_t(e)]++] = p;
        }
    }
    jobs_.assign(size_t(nj), fn::experts::cpu::job());
    int32_t j = 0;
    for (int32_t e = 0; e < cache::n_ex; ++e) {
        for (int32_t q = 0; q < fill[size_t(e)]; ++q) {
            fn::experts::cpu::job & x = jobs_[size_t(j)];
            if (x.nt == 0) {
                if (native_) {
                    fn::experts::cpu::set_native(res_->native(layer), e, x);
                } else {
                    const uint8_t * src = res_->entry_src(layer, e, 0);
                    for (int32_t c = 0; c < 2; ++c) {
                        x.gu[c] = src + L.gu_at(c);
                        x.dn[c] = src + L.dn_at(c);
                    }
                }
            }
            const int32_t k = first[size_t(e)] + q;
            x.row[x.nt] = compact[size_t(pairs_h_[k] / 10)];
            x.out[x.nt] = out32_.data() + int64_t(k) * D;
            if (++x.nt == fn::experts::cpu::job_max_rows || q + 1 == fill[size_t(e)]) {
                ++j;
            }
        }
    }
    cur_.jobs = nj;
    cur_.on = true;
    cur_.eligible = true;
    cur_.n_cpu = n;
    cur_.pairs = pairs;
    cur_.rows = n_rows;
    ++n_.pshare_layers;
    n_.pshare_experts += n;
    n_.pshare_pairs += pairs;
    n_.pshare_rows += n_rows;
    n_.pshare_unstaged_bytes += int64_t(n) * L.entry_bytes();
    n_.pshare_chunks += chunk_shared_ ? 0 : 1;
    chunk_shared_ = true;
    if (shadow_) { // TEST: the ring stages and the GPU computes the CPU's experts too
        sh_ids_.assign(ids, ids + int64_t(rows) * 10);
        sh_compact_ = compact;
        std::memset(cpu, 0, size_t(cache::n_ex));
    }
    return n;
}

bool prompt_share::begin(int32_t layer, const uint16_t * x16, void * const stream[2], std::string & err) {
    if (cur_.layer != layer || !cur_.eligible) {
        return true;
    }
    const cudaStream_t s = st(stream[0]), x = st(side_);
    if (!dv_->set(0, err) || !cache::cuda_ok(cudaEventRecord(ev(ev_t_[0]), s), "prompt CPU share", err)) {
        return false;
    }
    if (!cur_.on) {
        return true;
    }
    if (!cache::cuda_ok(cudaMemcpyAsync(mask_d_, mask_h_, size_t(cache::n_ex), cudaMemcpyHostToDevice, s),
                        "prompt CPU share mask", err) ||
        !cache::cuda_ok(cudaMemcpyAsync(pairs_d_, pairs_h_, size_t(cur_.pairs) * 4, cudaMemcpyHostToDevice, s),
                        "prompt CPU share pairs", err) ||
        !cache::cuda_ok(cudaEventRecord(ev(ev_ready_), s), "prompt CPU share", err) ||
        !cache::cuda_ok(cudaStreamWaitEvent(x, ev(ev_ready_), 0), "prompt CPU share", err) ||
        !cache::cuda_ok(cudaMemcpyAsync(rows_d_, rows_h_, size_t(cur_.rows) * 4, cudaMemcpyHostToDevice, x),
                        "prompt CPU share rows", err)) {
        return false;
    }
    sh_x16_ = x16;
    if (!fn::experts::es_gather(x16, rows_d_, cur_.rows, x16_d_, x)) {
        return cache::fail(err, "prompt CPU share: the gather");
    }
    if (!cache::cuda_ok(cudaMemcpyAsync(x16_h_, x16_d_, size_t(cur_.rows) * size_t(row16), cudaMemcpyDeviceToHost, x),
                        "prompt CPU share download", err) ||
        !cache::cuda_ok(cudaEventRecord(ev(ev_x_), x), "prompt CPU share", err)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        go_ = true;
        done_ = false;
    }
    cv_.notify_all();
    return true;
}

void prompt_share::settle(int32_t layer) {
    if (!armed_) {
        return;
    }
    join();
    if (cur_.layer == layer) {
        cur_ = layer_plan();
    }
}

const uint8_t * prompt_share::skip(int32_t layer) const {
    return cur_.layer == layer && cur_.on && !shadow_ ? mask_d_ : nullptr;
}

bool prompt_share::end(int32_t layer, const model::prompt_share_card card[2], void * const stream[2],
                       std::string & err) {
    if (cur_.layer != layer || !cur_.eligible) {
        return true;
    }
    const cudaStream_t s = st(stream[0]);
    if (!dv_->set(0, err) || !cache::cuda_ok(cudaEventRecord(ev(ev_t_[1]), s), "prompt CPU share", err)) {
        return false;
    }
    double cpu_ms = 0;
    if (cur_.on) {
        const cache::clk::time_point t0 = cache::clk::now();
        join();
        n_.pshare_join_ms += cache::since(t0) * 1e3;
        if (!job_ok_) {
            return cache::fail(err, "prompt CPU share: the CPU experts' activation download or the pool failed");
        }
        cpu_ms = job_ms_;
        n_.pshare_cpu_ms += cpu_ms;
        if (shadow_) { // TEST: compared, not added back; the layer's output stays the GPU's
            mx7_shadow_in in;
            in.layer = layer;
            in.rows = int32_t(sh_compact_.size());
            in.pairs = cur_.pairs;
            in.gu = gu_;
            in.dn = dn_;
            in.ids = sh_ids_.data();
            in.pair_of = pairs_h_;
            in.cpu = out32_.data();
            in.act = act_.data();
            in.compact = sh_compact_.data();
            in.native = native_ ? &res_->native(layer) : nullptr;
            in.x16_d = sh_x16_;
            for (int32_t k = 0; k < 2; ++k) {
                in.pos_d[k] = card[k].pos;
                in.part16_d[k] = card[k].part16;
            }
            in.wts_d = card[0].wts;
            in.stream = stream[0];
            if (!mx7_shadow_layer(in, sh_out_, err)) {
                return false;
            }
            cur_ = layer_plan();
            return true;
        }
        if (!cache::cuda_ok(
                cudaMemcpyAsync(out16_d_, out16_h_, size_t(cur_.pairs) * size_t(row16), cudaMemcpyHostToDevice, s),
                "prompt CPU share rows", err)) {
            return false;
        }
        for (int32_t k = 0; k < 2; ++k) {
            if (!fn::experts::es_add(pairs_d_, k == 0 ? out16_d_ : nullptr, cur_.pairs, card[k].pos, card[k].part16,
                                     k == 0 && job_bad_ ? 1 : 0, card[k].err, card[k].err_bit, s)) {
                return cache::fail(err, "prompt CPU share: the add-back");
            }
        }
    }
    if (!cache::cuda_ok(cudaEventRecord(ev(ev_t_[2]), s), "prompt CPU share", err)) {
        return false;
    }
    if (o_.share < 0) {
        pend_ = pending();
        pend_.valid = true;
        pend_.on = cur_.on;
        pend_.n_cpu = cur_.n_cpu;
        pend_.staged = cur_.nonres - cur_.n_cpu;
        pend_.nonres = cur_.nonres;
        pend_.cpu_ms = cpu_ms;
    }
    cur_ = layer_plan();
    return true;
}

// the share's thread: each posted job waits for its rows' download, converts them, runs the pool and writes the
// products as F16; the timer starts after the download, so c is the CPU's own work (Strata's)
void prompt_share::worker() {
    if (cpu_ >= 0) {
        cache::pin_thread(pthread_self(), cpu_);
    }
    if (dv_ != nullptr) {
        cudaSetDevice(dv_->dev[0]);
    }
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return go_ || stop_; });
            if (stop_) {
                return;
            }
            go_ = false;
        }
        bool ok = cudaEventSynchronize(ev(ev_x_)) == cudaSuccess;
        const cache::clk::time_point t0 = cache::clk::now();
        bool good = true;
        if (ok) {
            if (act_.size() < size_t(cur_.rows) * fn::experts::cpu::n_q8k) {
                act_.resize(size_t(cur_.rows) * fn::experts::cpu::n_q8k);
            }
            for (int32_t r = 0; r < cur_.rows; ++r) {
                fn::experts::cpu::gate_up_act_f16(x16_h_ + int64_t(r) * D,
                                                  act_.data() + int64_t(r) * fn::experts::cpu::n_q8k);
            }
            fn::experts::cpu::run_stats rs;
            ok = pool_->run(gu_, dn_, act_.data(), jobs_.data(), cur_.jobs, rs);
            good = ok && fn::experts::cpu::to_f16(out32_.data(), int64_t(cur_.pairs) * D, out16_h_);
        }
        const double ms = cache::since(t0) * 1e3;
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ok_ = ok;
            job_bad_ = !good;
            job_ms_ = ms;
            done_ = true;
        }
        cv_.notify_all();
    }
}

std::string prompt_share::setting() const {
    if (o_.share < 0) {
        return "auto";
    }
    char b[32];
    std::snprintf(b, sizeof(b), "%g", o_.share);
    return b;
}

std::string prompt_share::plan_line() const {
    char b[512];
    if (!armed_) {
        std::snprintf(b, sizeof(b), "  prompt CPU share (MX7): off (setting %s%s)\n", setting().c_str(),
                      dv_ != nullptr && dv_->nd == 2 ? "; two cards never share" : "");
        return b;
    }
    std::snprintf(b, sizeof(b),
                  "  prompt CPU share (MX7): %s, route-first ring chunks below %d rows, experts of at most %d pairs; "
                  "device buffers %s; the CPU thread on CPU %d; the CPU reads the %s\n",
                  setting().c_str(), o_.max_rows, pplan::share_maxt, cache::gib(dev_bytes_).c_str(), cpu_,
                  native_ ? "native views" : "packed backing (no native views: a test)");
    return b;
}

std::string prompt_share::state_json() const {
    char b[640];
    std::snprintf(b, sizeof(b),
                  "\"setting\": \"%s\", \"max_rows\": %d, \"armed\": %s, \"share_now\": %.4f, \"c_ms\": %.5f, "
                  "\"g_ms\": %.5f, \"c_n\": %lld, \"g_n\": %lld, \"check\": {\"warm\": %s, \"prefer_on\": %s, "
                  "\"cost_on_ms\": %.6f, \"cost_off_ms\": %.6f, \"measured_on\": %lld, \"measured_off\": %lld, "
                  "\"chose_on\": %lld, \"chose_off\": %lld, \"probes\": %lld}",
                  setting().c_str(), o_.max_rows, armed() ? "true" : "false", o_.share > 0 ? o_.share : share_now_, c_,
                  g_, (long long)nc_, (long long)ng_, check_.warm() ? "true" : "false",
                  check_.pref_on() ? "true" : "false", check_.cost(true), check_.cost(false),
                  (long long)check_.measured(true), (long long)check_.measured(false), (long long)check_.chose_on(),
                  (long long)check_.chose_off(), (long long)check_.probes());
    return b;
}

} // namespace fn::runtime
