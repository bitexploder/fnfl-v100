#include "fn/kernels/head/head_bench.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_cold.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/head/head.h"
#include "fn/model/shapes.h"
#include "fn/quant/formats.h"
#include "fn/quant/synth.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int32_t n_unit = 64;     // launches in a warm graph
constexpr int32_t n_warm = 3;      // untimed replays (warm) and trials (cold)
constexpr int32_t n_timed = 20;    // timed replays and trials
constexpr int32_t n_cold_rep = 20; // launches in a cold graph (XP1's rep)
constexpr int32_t n_round = 4;     // rounds in a whole-head graph (4 % 3 == 1: the last round takes slot 1)
constexpr int32_t n_set = 32;      // head weight sets a card: 32 x 6.55 MB = 210 MB rotated
constexpr double warm_s = 2.0;     // HX12: a process starts at 1,260 MHz for 0.3-1 s
constexpr double timeout_s = 30.0; // the host's bound on one replay
constexpr int64_t flush_used = 12ll << 20;
constexpr double ramp_us = 1.949;       // HX12 card 0 at 1,380 MHz
constexpr double dram_gbps = 1021.6;    // HX12
constexpr double tail_single_us = 0.31; // HX12: a single-pass launch's tail
constexpr double tail_across_us = 0.83; // HX12: a launch that finishes across blocks

constexpr int32_t D = fn::n_embd;
constexpr int32_t S = fn::n_hc;
constexpr int32_t F = fn::hc::n_own;
constexpr int32_t MH = fn::hc::n_mom_head;
constexpr int32_t R = fn::hc::max_rows;
constexpr int32_t VC = fn::head::n_vocab_card;
constexpr size_t vocab_bytes = size_t(VC) * (D / 256) * 176; // 218,521,600

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "v-bench: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool sent(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "v-bench: %s not enqueued: %s\n", what, cudaGetErrorString(cudaGetLastError()));
    }
    return ok;
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm = {};
    gmtime_r(&t, &tm);
    char buf[64];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms));
    return buf;
}

template <typename T> struct dev_buf {
    T * p = nullptr;
    size_t n = 0;

    explicit dev_buf(size_t count) : n(std::max<size_t>(count, 16)) {
        if (!cuda_ok(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc")) {
            p = nullptr;
        }
    }

    ~dev_buf() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dev_buf(const dev_buf &) = delete;
    dev_buf & operator=(const dev_buf &) = delete;
};

// The slot of round n in a graph that repeats: n % 3, the last round slot 1 (hc_bench's rule, xchg.h s(n+1) != s(n))
int32_t slot_of(int32_t n) {
    static_assert(n_round % fn::xchg::n_slot == 1, "the last round takes slot 1");
    return n == n_round - 1 ? 1 : n % fn::xchg::n_slot;
}

struct device_guard {
    explicit device_guard(int32_t d) { cudaSetDevice(d); }
};

struct card_b {
    int32_t dev;
    device_guard guard;
    cudaStream_t st = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    dev_buf<uint16_t> w_down{size_t(n_set) * fn::hc::w_down_head_elems}, w_up{size_t(n_set) * fn::hc::w_up_elems};
    dev_buf<float> gamma{size_t(S) * F};
    dev_buf<float> hres{size_t(R) * S * F}, hx{size_t(R) * F}, hpost{size_t(R) * S};
    dev_buf<float> x{size_t(R) * S * F}, mom{size_t(R) * MH}, full{size_t(R) * D};
    dev_buf<int8_t> codes{size_t(R) * D};
    dev_buf<float> scales{size_t(R) * D / 32 * 2};
    dev_buf<uint8_t> w_vocab{vocab_bytes};
    dev_buf<float> logits{size_t(R) * VC};
    dev_buf<uint32_t> pair{size_t(R) * 2};
    dev_buf<int32_t> token{size_t(R)};
    dev_buf<uint8_t> gwork{fn::head::greedy_work_bytes()}, lwork{fn::head::logits_work_bytes(VC)};
    // sinks the peer writes in the alone points, and this card's copies of a landed X-HC and X-HMIX slot; HC3: up2
    // writes both forms into a slot (own_hmix takes its own half in the alone points), with its run counters
    static constexpr size_t n_hmix = fn::xchg::hmix_layout::slot_words;
    dev_buf<float> sink_hc{size_t(R) * MH}, sink_hmix{n_hmix}, pre_hc{size_t(R) * MH}, pre_hmix{n_hmix},
        own_hmix{n_hmix};
    dev_buf<uint32_t> runs{size_t(fn::hc::n_runs_card)};
    dev_buf<uint8_t> flush{size_t(flush_used)};
    dev_buf<float> evict{16};

    explicit card_b(int32_t d) : dev(d), guard(d) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream") ||
            !cuda_ok(cudaEventCreate(&e0), "event") || !cuda_ok(cudaEventCreate(&e1), "event")) {
            st = nullptr;
        }
    }

    ~card_b() {
        cudaSetDevice(dev);
        for (cudaEvent_t e : {e0, e1}) {
            if (e != nullptr) {
                cudaEventDestroy(e);
            }
        }
        if (st != nullptr) {
            cudaStreamDestroy(st);
        }
    }

    card_b(const card_b &) = delete;
    card_b & operator=(const card_b &) = delete;

    bool cold() const { return fn::dense::cold_flush(flush.p, flush_used, st) && fn::dense::cold_icache(evict.p, st); }
};

struct bench {
    card_b * c[2];
    fn::xchg::rings * rg;

    // round n's slot; pre: the alone form (sends go to the peer's sinks, the receive slots are this card's landed
    // copies and are not reset)
    bool down(int32_t rows, int32_t r, int32_t n, bool pre) const {
        card_b & k = *c[r];
        fn::hc::down2_args d2;
        d2.d = {k.w_down.p + size_t(n % n_set) * fn::hc::w_down_head_elems,
                k.hres.p,
                k.hx.p,
                k.hpost.p,
                k.gamma.p,
                k.x.p,
                k.mom.p};
        d2.hc_peer = pre ? c[1 - r]->sink_hc.p : rg->slot(1 - r, fn::xchg::msg::hc, slot_of(n));
        d2.hc_reset = pre ? nullptr : rg->slot(r, fn::xchg::msg::hc, slot_of((n + n_round - 1) % n_round));
        d2.hmix_reset = pre ? nullptr : rg->slot(r, fn::xchg::msg::hmix, slot_of((n + n_round - 1) % n_round));
        return sent(fn::hc::head_down2(rows, d2, k.st), "head_down2");
    }

    bool up(int32_t rows, int32_t r, int32_t n, bool pre) const {
        card_b & k = *c[r];
        const fn::hc::up2_args u2 = {k.w_up.p + size_t(n % n_set) * fn::hc::w_up_elems,
                                     k.mom.p,
                                     pre ? k.pre_hc.p : rg->slot(r, fn::xchg::msg::hc, slot_of(n)),
                                     k.x.p,
                                     k.gamma.p,
                                     nullptr,
                                     pre ? k.own_hmix.p : rg->slot(r, fn::xchg::msg::hmix, slot_of(n)),
                                     pre ? c[1 - r]->sink_hmix.p : rg->slot(1 - r, fn::xchg::msg::hmix, slot_of(n)),
                                     k.runs.p,
                                     rg->err(r),
                                     r};
        return sent(fn::hc::head_up2(rows, u2, k.st), "head_up2");
    }

    // the X-HMIX stand-in consumer (no longer in the head, HC3): the row, codes and scales from a slot
    bool ready(int32_t rows, int32_t r, int32_t n, bool pre) const {
        card_b & k = *c[r];
        const fn::hc::ready_args ra = {pre ? k.pre_hmix.p : rg->slot(r, fn::xchg::msg::hmix, slot_of(n)),
                                       k.full.p,
                                       k.codes.p,
                                       k.scales.p,
                                       rg->err(r),
                                       r,
                                       0};
        return sent(fn::hc::ready(rows, ra, k.st), "ready");
    }

    // logits acquire the head's X-HMIX slot (HC3): round n's, or (pre) the landed copy
    bool logits(int32_t rows, int32_t r, int32_t n, bool pre) const {
        card_b & k = *c[r];
        const fn::head::logits_args la = {pre ? k.pre_hmix.p : rg->slot(r, fn::xchg::msg::hmix, slot_of(n)),
                                          rg->err(r),
                                          r,
                                          k.w_vocab.p,
                                          VC,
                                          k.logits.p,
                                          k.lwork.p};
        return sent(fn::head::logits(rows, la, k.st), "logits");
    }

    bool quantize(int32_t rows, int32_t r) const {
        card_b & k = *c[r];
        return sent(fn::act::quantize(k.full.p, {k.codes.p, k.scales.p, rows, D}, k.st), "quantize");
    }

    // the product alone: head.cu's vocab_args (one Q5_K matrix, m = n, K = 2560, DP2's splits)
    bool product(int32_t rows, int32_t r) const {
        card_b & k = *c[r];
        fn::dense::product_args p;
        p.k = D;
        p.n_mats = 1;
        p.mats[0].f = fn::dense::fmt::q5_k;
        p.mats[0].m = VC;
        p.mats[0].w = k.w_vocab.p;
        p.mats[0].out = k.logits.p;
        p.x_codes = k.codes.p;
        p.x_scales = k.scales.p;
        fn::dense::choose_splits(p);
        p.work = k.lwork.p;
        return sent(fn::dense::product(rows, p, k.st), "product");
    }

    // loop: the pair goes to this card's own T-GREEDY slot 0 (the alone forms); else the peer's slot of round n
    bool reduce(int32_t rows, int32_t r, int32_t n, bool loop) const {
        card_b & k = *c[r];
        const fn::head::reduce_args ga = {
            k.logits.p, VC, r * VC, k.pair.p, rg->slot(loop ? r : 1 - r, fn::xchg::msg::greedy, loop ? 0 : slot_of(n)),
            k.gwork.p};
        return sent(fn::head::greedy_reduce(rows, ga, k.st), "greedy_reduce");
    }

    bool decide(int32_t rows, int32_t r, int32_t n, bool loop) const {
        card_b & k = *c[r];
        const fn::head::decide_args da = {
            k.pair.p, rg->slot(r, fn::xchg::msg::greedy, loop ? 0 : slot_of(n)), k.token.p, rg->err(r), nullptr, r};
        return sent(fn::head::greedy_decide(rows, da, k.st), "greedy_decide");
    }
};

struct stats {
    double median = 0.0, p90 = 0.0;
    std::vector<double> us;
    std::string t0, t1;

    void finish() {
        std::vector<double> s = us;
        std::sort(s.begin(), s.end());
        const size_t n = s.size();
        median = n % 2 == 0 ? 0.5 * (s[n / 2 - 1] + s[n / 2]) : s[n / 2];
        p90 = s[(9 * n + 9) / 10 - 1];
    }
};

bool capture(card_b & k, int32_t n, const std::function<bool(int32_t)> & unit, cudaGraphExec_t & ge) {
    cudaGraph_t graph = nullptr;
    bool ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
              cuda_ok(cudaStreamBeginCapture(k.st, cudaStreamCaptureModeThreadLocal), "begin capture");
    for (int32_t i = 0; i < n && ok; ++i) {
        ok = unit(i);
    }
    ok = cuda_ok(cudaStreamEndCapture(k.st, &graph), "end capture") && ok;
    ok = ok && cuda_ok(cudaGraphInstantiate(&ge, graph, 0), "instantiate");
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    return ok;
}

bool replay_us(card_b & k, cudaGraphExec_t ge, double & us) {
    float ms = 0.0f;
    const bool ok = cuda_ok(cudaEventRecord(k.e0, k.st), "record") && cuda_ok(cudaGraphLaunch(ge, k.st), "replay") &&
                    cuda_ok(cudaEventRecord(k.e1, k.st), "record") && cuda_ok(cudaEventSynchronize(k.e1), "sync") &&
                    cuda_ok(cudaEventElapsedTime(&ms, k.e0, k.e1), "elapsed");
    us = 1e3 * double(ms);
    return ok;
}

// warm (DP1's method): n_unit units back to back in one graph; a unit's time is a replay's time over n_unit
bool time_warm(card_b & k, const std::function<bool(int32_t)> & unit, stats & s) {
    cudaGraphExec_t ge = nullptr;
    bool ok = capture(k, n_unit, unit, ge);
    double us = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = replay_us(k, ge, us);
    }
    s.us.clear();
    s.t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        ok = replay_us(k, ge, us);
        s.us.push_back(us / n_unit);
    }
    s.t1 = utc_now();
    if (ge != nullptr) {
        cudaGraphExecDestroy(ge);
    }
    if (ok) {
        s.finish();
    }
    return ok;
}

// cold (R17, XP1's alone_timer, as dp-bench): A = n_cold_rep x [flush, eviction, unit], B = n_cold_rep x [flush,
// eviction]; a trial is (A - B) / n_cold_rep
bool time_cold(card_b & k, const std::function<bool(int32_t)> & unit, stats & s) {
    cudaGraphExec_t ga = nullptr, gb = nullptr;
    bool ok = capture(
                  k, n_cold_rep, [&](int32_t i) { return k.cold() && unit(i); }, ga) &&
              capture(k, n_cold_rep, [&](int32_t) { return k.cold(); }, gb);
    double ta = 0.0, tb = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = replay_us(k, ga, ta) && replay_us(k, gb, tb);
    }
    s.us.clear();
    s.t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        ok = replay_us(k, gb, tb) && replay_us(k, ga, ta);
        s.us.push_back((ta - tb) / n_cold_rep);
    }
    s.t1 = utc_now();
    for (cudaGraphExec_t g : {ga, gb}) {
        if (g != nullptr) {
            cudaGraphExecDestroy(g);
        }
    }
    if (ok) {
        s.finish();
    }
    return ok;
}

void print_us(const char * key, const std::vector<double> & us) {
    printf(", \"%s\": [", key);
    for (size_t i = 0; i < us.size(); ++i) {
        printf("%s%.3f", i == 0 ? "" : ", ", us[i]);
    }
    printf("]");
}

// bytes: the unit's DRAM bytes (weights, activations in and out); floor: ramp + bytes / rate + tail
double floor_of(double bytes, bool across) {
    return ramp_us + bytes / (dram_gbps * 1e3) + (across ? tail_across_us : tail_single_us);
}

bool alone(bench & bn, int32_t r, const char * point, const char * what, int32_t rows, double bytes, bool across,
           const std::function<bool(int32_t)> & unit, bool & ok) {
    if (!ok) {
        return false;
    }
    card_b & k = *bn.c[r];
    stats w, cd;
    ok = time_warm(k, unit, w) && time_cold(k, unit, cd);
    uint32_t err[2] = {0u, 0u};
    ok = ok && bn.rg->read_err(0, err[0]) && bn.rg->read_err(1, err[1]);
    if (ok && (err[0] != 0u || err[1] != 0u)) {
        fprintf(stderr, "v-bench: %s: error words 0x%x 0x%x\n", point, err[0], err[1]);
        ok = false;
    }
    if (!ok) {
        return false;
    }
    const double fl = bytes > 0.0 ? floor_of(bytes, across) : 0.0;
    printf("{\"ticket\": \"V1\", \"point\": \"%s_B%d_card%d\", \"what\": \"%s\", \"rows\": %d, \"card\": %d, "
           "\"median_us\": %.3f, \"p90_us\": %.3f, \"cold_median_us\": %.3f, \"cold_p90_us\": %.3f, \"n\": %d, "
           "\"units_per_replay\": %d, \"warm_replays\": %d, \"bytes\": %.0f, \"floor_us\": %.3f, "
           "\"hw_pct_warm\": %.1f, \"hw_pct_cold\": %.1f, \"warm\": \"head weights rotated over 32 sets (210 MB); the "
           "vocabulary matrix is 218.5 MB; activations and code warm\", \"cold\": \"(A - B) / %d: 12 MB L2 flush and "
           "instruction-cache eviction before each launch\", \"t_start\": \"%s\", \"t_end\": \"%s\"",
           point, rows, r, what, rows, r, w.median, w.p90, cd.median, cd.p90, n_timed, n_unit, n_warm, bytes, fl,
           fl > 0.0 ? 100.0 * fl / w.median : 0.0, fl > 0.0 ? 100.0 * fl / cd.median : 0.0, n_cold_rep, w.t0.c_str(),
           cd.t1.c_str());
    print_us("us", w.us);
    print_us("cold_us", cd.us);
    printf("}\n");
    fflush(stdout);
    return true;
}

bool wait_events(card_b * c[2]) {
    const auto t0 = std::chrono::steady_clock::now();
    bool done[2] = {false, false};
    while (!(done[0] && done[1])) {
        for (int32_t r = 0; r < 2; ++r) {
            if (done[r]) {
                continue;
            }
            const cudaError_t e = cudaEventQuery(c[r]->e1);
            if (e == cudaSuccess) {
                done[r] = true;
            } else if (e != cudaErrorNotReady) {
                return cuda_ok(e, "event");
            }
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
            fprintf(stderr, "v-bench: host timeout after %.0f s\n", timeout_s);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

// the whole head (down2, up2, ready, logits, reduce, decide) n_round rounds in one graph a card, both cards' graphs
// replayed together; a round's time is a replay's time over n_round
bool whole_head(bench & bn, int32_t rows, bool & ok) {
    if (!ok) {
        return false;
    }
    cudaGraphExec_t ge[2] = {nullptr, nullptr};
    ok = bn.rg->reset_all();
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = capture(
            *bn.c[r], n_round,
            [&](int32_t n) {
                return bn.down(rows, r, n, false) && bn.up(rows, r, n, false) && bn.logits(rows, r, n, false) &&
                       bn.reduce(rows, r, n, false) && bn.decide(rows, r, n, false);
            },
            ge[r]);
    }
    std::vector<double> us[2];
    std::string t0;
    for (int32_t it = 0; it < n_warm + n_timed && ok; ++it) {
        if (it == n_warm) {
            t0 = utc_now();
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card_b & k = *bn.c[r];
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && cuda_ok(cudaEventRecord(k.e0, k.st), "record") &&
                 cuda_ok(cudaGraphLaunch(ge[r], k.st), "replay") && cuda_ok(cudaEventRecord(k.e1, k.st), "record");
        }
        ok = ok && wait_events(bn.c);
        for (int32_t r = 0; r < 2 && ok && it >= n_warm; ++r) {
            float ms = 0.0f;
            ok = cuda_ok(cudaEventElapsedTime(&ms, bn.c[r]->e0, bn.c[r]->e1), "elapsed");
            us[r].push_back(1e3 * double(ms) / n_round);
        }
    }
    const std::string t1 = utc_now();
    for (int32_t r = 0; r < 2; ++r) {
        if (ge[r] != nullptr) {
            cudaSetDevice(bn.c[r]->dev);
            cudaGraphExecDestroy(ge[r]);
        }
    }
    uint32_t err[2] = {0u, 0u};
    ok = ok && bn.rg->read_err(0, err[0]) && bn.rg->read_err(1, err[1]);
    if (ok && (err[0] != 0u || err[1] != 0u)) {
        fprintf(stderr, "v-bench: whole head B%d: error words 0x%x 0x%x\n", rows, err[0], err[1]);
        ok = false;
    }
    if (!ok) {
        return false;
    }
    printf("{\"ticket\": \"V1\", \"point\": \"head_B%d\", \"what\": \"whole head a round: head_down2, head_up2, ready, "
           "logits (quantize, product), greedy_reduce, greedy_decide on both cards, %d rounds a graph, both graphs "
           "replayed together\", \"rows\": %d, \"n\": %d, \"warm_replays\": %d",
           rows, n_round, rows, n_timed, n_warm);
    for (int32_t r = 0; r < 2; ++r) {
        stats s;
        s.us = us[r];
        s.finish();
        printf(", \"card%d\": {\"median_us\": %.3f, \"p90_us\": %.3f, \"ms_a_round\": %.4f", r, s.median, s.p90,
               s.median / 1e3);
        print_us("us", us[r]);
        printf("}");
    }
    printf(", \"t_start\": \"%s\", \"t_end\": \"%s\"}\n", t0.c_str(), t1.c_str());
    fflush(stdout);
    return true;
}

// random BF16 weights U(-a, a) into set 0, copied to every set (distinct addresses; the values do not matter)
bool fill_sets(dev_buf<uint16_t> & d, int64_t elems, float a, std::mt19937 & rng, cudaStream_t st) {
    std::uniform_real_distribution<float> u(-a, a);
    std::vector<uint16_t> h(static_cast<size_t>(elems));
    for (auto & v : h) {
        v = fn::quant::f32_to_bf16(u(rng));
    }
    bool ok = cuda_ok(cudaMemcpy(d.p, h.data(), h.size() * 2, cudaMemcpyHostToDevice), "weights");
    for (int32_t s = 1; s < n_set && ok; ++s) {
        ok = cuda_ok(cudaMemcpyAsync(d.p + size_t(s) * elems, d.p, h.size() * 2, cudaMemcpyDeviceToDevice, st), "set");
    }
    return ok;
}

bool fill_f32(dev_buf<float> & d, size_t n, float lo, float hi, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> h(n);
    for (auto & v : h) {
        v = u(rng);
    }
    return cuda_ok(cudaMemcpy(d.p, h.data(), n * 4, cudaMemcpyHostToDevice), "f32");
}

// each card's X-HC and X-HMIX as they land, written straight into the peer's pre_hc and pre_hmix (no waits), at
// max_rows; then the full rows, codes and logits once, so the alone points read real values
bool prefill(bench & bn) {
    bool ok = bn.rg->reset_all();
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card_b & k = *bn.c[r];
        fn::hc::down2_args d2;
        d2.d = {k.w_down.p, k.hres.p, k.hx.p, k.hpost.p, k.gamma.p, k.x.p, k.mom.p};
        d2.hc_peer = bn.c[1 - r]->pre_hc.p;
        d2.hc_reset = nullptr;
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && sent(fn::hc::head_down2(R, d2, k.st), "prefill down") &&
             cuda_ok(cudaStreamSynchronize(k.st), "prefill down");
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card_b & k = *bn.c[r];
        const fn::hc::up2_args u2 = {
            k.w_up.p, k.mom.p,       k.pre_hc.p, k.x.p, k.gamma.p, nullptr, k.pre_hmix.p, bn.c[1 - r]->pre_hmix.p,
            k.runs.p, bn.rg->err(r), r};
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && sent(fn::hc::head_up2(R, u2, k.st), "prefill up") &&
             cuda_ok(cudaStreamSynchronize(k.st), "prefill up");
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(bn.c[r]->dev), "cudaSetDevice") && bn.ready(R, r, 0, true) &&
             bn.logits(R, r, 0, true) && cuda_ok(cudaStreamSynchronize(bn.c[r]->st), "prefill logits");
    }
    return ok && bn.rg->reset_all();
}

bool busy_both(card_b * c[2], double secs) {
    bool ok = true;
    void * scratch[2] = {nullptr, nullptr};
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") &&
             cuda_ok(cudaMalloc(&scratch[r], size_t(64) << 20), "scratch");
    }
    const auto t0 = std::chrono::steady_clock::now();
    int64_t n = 0;
    while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        for (int32_t i = 0; i < 16 && ok; ++i, ++n) {
            for (int32_t r = 0; r < 2 && ok; ++r) {
                ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") &&
                     cuda_ok(cudaMemsetAsync(scratch[r], static_cast<int>(n & 0xff), size_t(64) << 20, c[r]->st),
                             "memset");
            }
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") && cuda_ok(cudaStreamSynchronize(c[r]->st), "busy");
        }
    }
    for (int32_t r = 0; r < 2; ++r) {
        if (scratch[r] != nullptr) {
            cudaSetDevice(c[r]->dev);
            cudaFree(scratch[r]);
        }
    }
    printf("# v-bench: both cards busy for %.2f s (%lld memsets of 64 MiB each), timing from %s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), static_cast<long long>(n),
           utc_now().c_str());
    return ok;
}

} // namespace

int fnfl_v_bench(int dev0, int dev1, const char * mode) {
    if (std::string(mode) != "all") {
        fprintf(stderr, "v-bench: mode is all\n");
        return 2;
    }
    if (dev0 == dev1 || !fn::xchg::enable_peer_access(dev0, dev1)) {
        fprintf(stderr, "v-bench: two devices with peer access\n");
        return 1;
    }
    for (const int32_t d : {dev0, dev1}) {
        cudaDeviceProp prop = {};
        int driver = 0, runtime = 0;
        if (!cuda_ok(cudaGetDeviceProperties(&prop, d), "properties") ||
            !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
            return 1;
        }
        printf("# v-bench: device %d: %s, sm_%d%d, %d SMs, PCI %02x:%02x, driver API %d, runtime %d, start %s\n", d,
               prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.pciBusID, prop.pciDeviceID, driver,
               runtime, utc_now().c_str());
    }
    // the synthetic vocabulary shard: fn/quant's Q5_K blocks (C1b's ranges), one matrix packed for both cards
    std::vector<uint8_t> raw(vocab_bytes), packed(vocab_bytes);
    bool ok = fn::quant::synth_blocks(fn::quant::wfmt::q5_k, fn::quant::synth_seed(9100), int64_t(VC) * (D / 256),
                                      raw.data()) &&
              fn::dense::pack(fn::dense::fmt::q5_k, VC, D, raw.data(), size_t(D / 256) * 176, packed.data());
    raw.clear();
    raw.shrink_to_fit();
    {
        card_b k0(dev0), k1(dev1);
        card_b * c[2] = {&k0, &k1};
        fn::xchg::rings rg(dev0, dev1);
        bench bn = {{&k0, &k1}, &rg};
        ok = ok && rg.ok() && k0.st != nullptr && k1.st != nullptr;
        std::mt19937 rng(11);
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card_b & k = *c[r];
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
                 fill_sets(k.w_down, fn::hc::w_down_head_elems, 0.07f, rng, k.st) &&
                 fill_sets(k.w_up, fn::hc::w_up_elems, 0.25f, rng, k.st) &&
                 fill_f32(k.hres, size_t(R) * S * F, -2.0f, 2.0f, rng) &&
                 fill_f32(k.hx, size_t(R) * F, -2.0f, 2.0f, rng) && fill_f32(k.gamma, size_t(S) * F, 0.5f, 1.5f, rng) &&
                 cuda_ok(cudaMemcpy(k.w_vocab.p, packed.data(), packed.size(), cudaMemcpyHostToDevice), "vocab") &&
                 cuda_ok(cudaMemset(k.gwork.p, 0, k.gwork.n), "gwork") &&
                 cuda_ok(cudaMemset(k.runs.p, 0, k.runs.n * sizeof(uint32_t)), "runs") &&
                 cuda_ok(cudaMemset(k.lwork.p, 0, k.lwork.n), "lwork") &&
                 cuda_ok(cudaMemset(k.flush.p, 0x5a, k.flush.n), "flush buffer");
        }
        std::vector<float> post(size_t(R) * S);
        std::uniform_real_distribution<float> d_post(0.5f, 1.5f);
        for (auto & v : post) {
            v = d_post(rng);
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") &&
                 cuda_ok(cudaMemcpy(c[r]->hpost.p, post.data(), post.size() * 4, cudaMemcpyHostToDevice), "hpost") &&
                 cuda_ok(cudaDeviceSynchronize(), "setup");
        }
        ok = ok && prefill(bn) && busy_both(c, warm_s);

        // DRAM bytes a launch moves (C): weights plus activations in and out
        const double w_down = double(fn::hc::w_down_head_elems) * 2, w_up = double(fn::hc::w_up_elems) * 2;
        for (const int32_t rows : {1, 3}) {
            const double b_x = double(rows) * S * F * 4, b_hx = double(rows) * F * 4, b_mom = double(rows) * MH * 4;
            const double b_full = double(rows) * D * 4, b_codes = double(rows) * D * (1 + 0.25);
            const double b_log = double(rows) * VC * 4;
            for (int32_t r = 0; r < 2; ++r) {
                if (r == 0) {
                    alone(
                        bn, 0, "head_down2", "head_down2 alone, sends to a sink", rows,
                        w_down + b_x + b_hx + b_x + 2 * b_mom, false,
                        [&bn, rows](int32_t n) { return bn.down(rows, 0, n, true); }, ok);
                    alone(
                        bn, 0, "head_up2", "head_up2 (arm 0) alone, X-HC pre-filled, sends to a sink", rows,
                        w_up + b_x + 2 * b_mom + 2 * b_hx, false,
                        [&bn, rows](int32_t n) { return bn.up(rows, 0, n, true); }, ok);
                    alone(
                        bn, 0, "ready", "ready alone, X-HMIX pre-filled, no reset", rows, 2 * b_hx + b_full, false,
                        [&bn, rows](int32_t n) { return bn.ready(rows, 0, n, true); }, ok);
                    alone(
                        bn, 0, "quantize", "act::quantize of the full row alone", rows, b_full + b_codes, false,
                        [&bn, rows](int32_t) { return bn.quantize(rows, 0); }, ok);
                }
                alone(
                    bn, r, "product", "dense::product Q5_K [124160 x 2560] alone", rows,
                    double(vocab_bytes) + b_codes + b_log, false,
                    [&bn, rows, r](int32_t) { return bn.product(rows, r); }, ok);
                if (r == 0) {
                    alone(
                        bn, 0, "logits", "logits (the product, acquiring a landed X-HMIX copy) alone", rows,
                        double(vocab_bytes) + b_codes + b_log, false,
                        [&bn, rows](int32_t) { return bn.logits(rows, 0, 0, true); }, ok);
                    alone(
                        bn, 0, "greedy_reduce", "greedy_reduce alone over 124,160 logits, pair to its own slot", rows,
                        b_log, true, [&bn, rows](int32_t n) { return bn.reduce(rows, 0, n, true); }, ok);
                    alone(
                        bn, 0, "greedy_pair", "greedy_reduce then greedy_decide, loopback through its own slot", rows,
                        b_log, true,
                        [&bn, rows](int32_t n) { return bn.reduce(rows, 0, n, true) && bn.decide(rows, 0, n, true); },
                        ok);
                }
            }
        }
        ok = ok && rg.reset_all();
        for (int32_t rows = 1; rows <= R && ok; ++rows) {
            whole_head(bn, rows, ok);
        }
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(c[r]->dev);
            ok = cuda_ok(cudaDeviceSynchronize(), "final sync") && ok;
        }
    }
    printf("# v-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
