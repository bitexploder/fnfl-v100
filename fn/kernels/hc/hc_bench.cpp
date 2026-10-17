#include "fn/kernels/hc/hc_bench.h"

#include "fn/kernels/dense/dense_cold.h"
#include "fn/kernels/hc/hc.h"
#include "fn/quant/formats.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int32_t n_sets = 64;  // weight sets, one per launch of a graph: 212 MB of hc_down weights, 210 MB of hc_up
constexpr int32_t n_unit = 64;  // units (one launch, or one sequence) captured in a graph
constexpr int32_t n_warm = 3;   // untimed replays before timing
constexpr int32_t n_timed = 20; // timed replays
constexpr double warm_s = 2.0;  // HX12: a process starts at 1,260 MHz for 0.3-1 s, so the card is kept busy first
constexpr int32_t S = 4;
constexpr int32_t F = fn::hc::n_own;
constexpr int32_t M = fn::hc::n_mom;
constexpr int32_t R = fn::hc::max_rows;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "hc-bench: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
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

// owns one device allocation of n elements of T
template <typename T> struct dev_buf {
    T * p = nullptr;

    explicit dev_buf(size_t n) {
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

// owns a captured graph's executable and the two timing events
struct timed_graph {
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    timed_graph() = default;

    ~timed_graph() {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
        }
        if (e0 != nullptr) {
            cudaEventDestroy(e0);
        }
        if (e1 != nullptr) {
            cudaEventDestroy(e1);
        }
    }

    timed_graph(const timed_graph &) = delete;
    timed_graph & operator=(const timed_graph &) = delete;
};

// one card's activations and outputs, sized for the largest row count
struct card_bufs {
    static constexpr size_t n_x = size_t(R) * S * F, n_row = size_t(R) * F, n_gamma = size_t(S) * F;
    dev_buf<float> hres{n_x}, hx{n_row}, gamma{n_gamma}, x{n_x};
    dev_buf<float> mom{size_t(R) * M}, inject{size_t(R) * 4}, mixed{n_row};
};

// a kernel instance: the shipped down() or up(), or the kernels at a stream depth
struct which {
    bool shipped;
    int32_t depth;
};

struct bench {
    cudaStream_t st = nullptr;
    dev_buf<uint16_t> w_down{static_cast<size_t>(n_sets) * fn::hc::w_down_elems};
    dev_buf<uint16_t> w_up{static_cast<size_t>(n_sets) * fn::hc::w_up_elems};
    dev_buf<float> hpost{static_cast<size_t>(R) * S};
    dev_buf<uint32_t> probe_out{size_t(fn::hc::probe_out_elems)};
    card_bufs card[2];
    // launches of each kernel instance so far, so a profiler can skip to a timed launch
    std::map<std::string, int64_t> launches;

    static std::string key(const char * kernel, int32_t depth, int32_t rows) {
        const std::string b = rows > 0 ? " B" + std::to_string(rows) : "";
        return std::string(kernel) + " d" + std::to_string(depth) + b;
    }

    bool down(int32_t rows, int32_t r, int32_t set, which v = {true, 0}) {
        const fn::hc::down_args a = {w_down.p + static_cast<size_t>(set) * fn::hc::w_down_elems,
                                     card[r].hres.p,
                                     card[r].hx.p,
                                     hpost.p,
                                     card[r].gamma.p,
                                     card[r].x.p,
                                     card[r].mom.p};
        const fn::hc::variant_id sh = fn::hc::shipped();
        const int32_t depth = v.shipped ? sh.depth_down : v.depth;
        ++launches[key("down", depth, rows)];
        const bool sent = v.shipped ? fn::hc::down(rows, a, st) : fn::hc::down_variant(depth, rows, a, st);
        if (!sent) {
            fprintf(stderr, "hc-bench: hc down not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        return true;
    }

    bool up(int32_t rows, int32_t r, int32_t set, which v = {true, 0}) {
        const fn::hc::up_args a = {w_up.p + static_cast<size_t>(set) * fn::hc::w_up_elems,
                                   card[r].mom.p,
                                   card[1 - r].mom.p,
                                   card[r].x.p,
                                   card[r].gamma.p,
                                   card[r].inject.p,
                                   card[r].mixed.p};
        const fn::hc::variant_id sh = fn::hc::shipped();
        const int32_t depth = v.shipped ? sh.depth_up : v.depth;
        ++launches[key("up", depth, rows)];
        const bool sent = v.shipped ? fn::hc::up(rows, a, st) : fn::hc::up_variant(depth, rows, a, st);
        if (!sent) {
            fprintf(stderr, "hc-bench: hc up not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        return true;
    }

    bool probe(int32_t depth, int32_t set) {
        ++launches[key("probe", depth, 0)];
        if (!fn::hc::probe(depth, w_down.p + static_cast<size_t>(set) * fn::hc::w_down_elems, probe_out.p, st)) {
            fprintf(stderr, "hc-bench: probe not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        return true;
    }
};

bool fill(dev_buf<float> & d, size_t n, std::mt19937 & rng, float lo, float hi, cudaStream_t st) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> h(n);
    for (auto & v : h) {
        v = dist(rng);
    }
    return cuda_ok(cudaMemcpyAsync(d.p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice, st), "fill") &&
           cuda_ok(cudaStreamSynchronize(st), "fill sync");
}

// BF16 weights in [-a, a): set 0 from the host, then copied to the other sets on the device
bool fill_weights(dev_buf<uint16_t> & d, int64_t elems, std::mt19937 & rng, float a, cudaStream_t st) {
    std::uniform_real_distribution<float> dist(-a, a);
    std::vector<uint16_t> h(static_cast<size_t>(elems));
    for (auto & v : h) {
        v = fn::quant::f32_to_bf16(dist(rng));
    }
    const size_t bytes = static_cast<size_t>(elems) * sizeof(uint16_t);
    if (!cuda_ok(cudaMemcpyAsync(d.p, h.data(), bytes, cudaMemcpyHostToDevice, st), "weights")) {
        return false;
    }
    for (int32_t i = 1; i < n_sets; ++i) {
        if (!cuda_ok(cudaMemcpyAsync(d.p + static_cast<size_t>(i) * static_cast<size_t>(elems), d.p, bytes,
                                     cudaMemcpyDeviceToDevice, st),
                     "weight sets")) {
            return false;
        }
    }
    return cuda_ok(cudaStreamSynchronize(st), "weights sync");
}

// Captures n_unit units of work, replays n_warm untimed and n_timed timed, and prints one JSON line. The time of a
// unit is a replay's event time over n_unit; median and p90 (nearest rank) over the timed replays.
bool time_point(bench & bn, const char * point, const char * what, int32_t rows, int32_t launches_per_unit,
                const std::function<bool(int32_t)> & unit) {
    cudaGraph_t graph = nullptr;
    timed_graph tg;
    if (!cuda_ok(cudaStreamBeginCapture(bn.st, cudaStreamCaptureModeThreadLocal), "begin capture")) {
        return false;
    }
    bool ok = true;
    const std::map<std::string, int64_t> before = bn.launches;
    for (int32_t i = 0; i < n_unit && ok; ++i) {
        ok = unit(i);
    }
    // capture enqueues nothing: undo the count and add it back per replay
    std::map<std::string, int64_t> per_replay;
    for (const auto & kv : bn.launches) {
        const auto it = before.find(kv.first);
        const int64_t n = kv.second - (it == before.end() ? 0 : it->second);
        if (n > 0) {
            per_replay[kv.first] = n;
        }
    }
    bn.launches = before;
    const auto replayed = [&]() {
        for (const auto & kv : per_replay) {
            bn.launches[kv.first] += kv.second;
        }
    };
    ok = cuda_ok(cudaStreamEndCapture(bn.st, &graph), "end capture") && ok;
    if (!ok) {
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        return false;
    }
    ok = cuda_ok(cudaGraphInstantiate(&tg.exec, graph, 0), "instantiate");
    cudaGraphDestroy(graph);
    ok = ok && cuda_ok(cudaEventCreate(&tg.e0), "event") && cuda_ok(cudaEventCreate(&tg.e1), "event");
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = cuda_ok(cudaGraphLaunch(tg.exec, bn.st), "warm replay") &&
             cuda_ok(cudaStreamSynchronize(bn.st), "warm sync");
        replayed();
    }
    std::map<std::string, int64_t> first_timed;
    for (const auto & kv : per_replay) {
        first_timed[kv.first] = bn.launches[kv.first];
    }
    std::vector<double> us;
    const std::string t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        float ms = 0.0f;
        ok = cuda_ok(cudaEventRecord(tg.e0, bn.st), "record") && cuda_ok(cudaGraphLaunch(tg.exec, bn.st), "replay") &&
             cuda_ok(cudaEventRecord(tg.e1, bn.st), "record") && cuda_ok(cudaEventSynchronize(tg.e1), "sync") &&
             cuda_ok(cudaEventElapsedTime(&ms, tg.e0, tg.e1), "elapsed");
        replayed();
        us.push_back(1000.0 * ms / n_unit);
    }
    const std::string t1 = utc_now();
    if (!ok) {
        return false;
    }
    std::vector<double> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    const double median = 0.5 * (sorted[n_timed / 2 - 1] + sorted[n_timed / 2]);
    const double p90 = sorted[(9 * n_timed + 9) / 10 - 1];
    printf("{\"ticket\": \"HC1c\", \"point\": \"%s\", \"what\": \"%s\", \"rows\": %d, "
           "\"median_us\": %.3f, \"p90_us\": %.3f, \"n\": %d, \"units_per_replay\": %d, \"launches_per_unit\": %d, "
           "\"warm_replays\": %d, \"first_timed_launch\": {",
           point, what, rows, median, p90, n_timed, n_unit, launches_per_unit, n_warm);
    bool first = true;
    for (const auto & kv : first_timed) {
        printf("%s\"%s\": %lld", first ? "" : ", ", kv.first.c_str(), static_cast<long long>(kv.second));
        first = false;
    }
    printf("}, \"t_start\": \"%s\", \"t_end\": \"%s\", \"us\": [", t0.c_str(), t1.c_str());
    for (int32_t i = 0; i < n_timed; ++i) {
        printf("%s%.3f", i == 0 ? "" : ", ", us[static_cast<size_t>(i)]);
    }
    printf("]}\n");
    fflush(stdout);
    return true;
}

// ---- HC2: the two-card sublayer ----

constexpr int32_t n_sub = 64;          // sublayers in a graph, and weight sets a card (422 MB)
constexpr double hc2_timeout_s = 30.0; // the host's bound on one replay pair
constexpr int32_t n_cold_rep = 20;     // HC4 cold mode: units of a cold graph (V1's and XP1's rep)
constexpr int64_t flush_bytes = 12ll << 20; // HC4 cold mode: the L2 flush (twice L2)

// The slot of sublayer n in a graph that repeats: n % 3, except the last (64 is not a multiple of 3), which must
// differ from both its neighbours across the repeat (xchg.h: s(n+1) != s(n)): slots 2 (n = 62) and 0 (n = 0).
int32_t slot_of(int32_t n) {
    static_assert(n_sub % fn::xchg::n_slot == 1, "the last sublayer takes slot 1");
    return n == n_sub - 1 ? 1 : n % fn::xchg::n_slot;
}

struct device_guard {
    explicit device_guard(int32_t d) { cudaSetDevice(d); }
};

// one card's buffers, on its own device (the guard sets it before the members are allocated)
struct card2 {
    static constexpr size_t n_x = size_t(R) * S * F, n_row = size_t(R) * F, n_gamma = size_t(S) * F;
    int32_t dev;
    device_guard guard;
    cudaStream_t st = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    dev_buf<uint16_t> w_down{static_cast<size_t>(n_sub) * fn::hc::w_down_elems};
    dev_buf<uint16_t> w_up{static_cast<size_t>(n_sub) * fn::hc::w_up_elems};
    static constexpr size_t n_hmix = fn::xchg::hmix_layout::slot_words;
    dev_buf<float> hres{n_x}, hx{n_row}, hpost{size_t(R) * S}, gamma{n_gamma}, x{n_x};
    dev_buf<float> mom{size_t(R) * M}, inject{size_t(R) * 4}, full{2 * n_row};
    // written by the peer in the pre-filled points (sinks), and this card's copies of a landed X-HC and X-HMIX slot;
    // own_hmix takes up2's own half in the pre-filled points (HC3: it writes both forms into a slot)
    dev_buf<float> sink_hc{size_t(R) * M}, sink_hmix{n_hmix}, pre_hc{size_t(R) * M}, pre_hmix{n_hmix}, own_hmix{n_hmix};
    dev_buf<uint32_t> runs{size_t(fn::hc::n_runs_card)}; // up2's run counters (zeroed once)
    // DR1: Q8_0 weight sets (q8 mode only: 113 MB of down2_q8 weights and 111 MB of up2_q8 a card); the head forms
    // read set n's leading bytes with the head's layout
    const bool q8;
    dev_buf<uint8_t> q_down{q8 ? static_cast<size_t>(n_sub) * fn::hc::w_down_q8_bytes : 0};
    dev_buf<uint8_t> q_up{q8 ? static_cast<size_t>(n_sub) * fn::hc::w_up_q8_bytes : 0};
    // HC4 cold mode (R17, v-bench's method): a 12 MB L2 flush and the instruction-cache evictor before a cold launch
    dev_buf<uint8_t> flush{size_t(flush_bytes)};
    dev_buf<float> evict{16};
    // HC4 (B3): the pre-filled one-launch sublayer reads a private X-HC slot a unit (the peer's region landed, its own
    // region the sentinel, as hc::sub expects), restored from pre_sub_init before every replay
    static constexpr size_t n_hc_slot = fn::xchg::slot_words(fn::xchg::msg::hc);
    dev_buf<float> pre_sub{size_t(n_sub) * n_hc_slot}, pre_sub_init{size_t(n_sub) * n_hc_slot};

    bool cold() const { return fn::dense::cold_flush(flush.p, flush_bytes, st) && fn::dense::cold_icache(evict.p, st); }

    card2(int32_t d, bool with_q8) : dev(d), guard(d), q8(with_q8) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream") ||
            !cuda_ok(cudaEventCreate(&e0), "event") || !cuda_ok(cudaEventCreate(&e1), "event")) {
            st = nullptr;
        }
    }

    ~card2() {
        cudaSetDevice(dev);
        if (e0 != nullptr) {
            cudaEventDestroy(e0);
        }
        if (e1 != nullptr) {
            cudaEventDestroy(e1);
        }
        if (st != nullptr) {
            cudaStreamDestroy(st);
        }
    }

    card2(const card2 &) = delete;
    card2 & operator=(const card2 &) = delete;
};

struct bench2 {
    card2 * c[2];
    fn::xchg::rings * rg;

    // the real sublayer n on card r: down2, up2, ready (DR1: with Q8_0 weights, and the head forms)
    bool down(int32_t rows, int32_t r, int32_t n, bool pre, bool q8 = false, bool head = false) const {
        card2 & k = *c[r];
        const fn::hc::down_args da = {q8 ? nullptr : k.w_down.p + static_cast<size_t>(n) * fn::hc::w_down_elems,
                                      k.hres.p,
                                      k.hx.p,
                                      k.hpost.p,
                                      k.gamma.p,
                                      k.x.p,
                                      k.mom.p};
        const int32_t prev = slot_of((n + n_sub - 1) % n_sub);
        const fn::hc::down2_args d2 = {da, pre ? c[1 - r]->sink_hc.p : rg->slot(1 - r, fn::xchg::msg::hc, slot_of(n)),
                                       pre ? nullptr : rg->slot(r, fn::xchg::msg::hc, prev),
                                       pre ? nullptr : rg->slot(r, fn::xchg::msg::hmix, prev)};
        if (q8) {
            const fn::hc::down2_q8_args dq = {d2, k.q_down.p + static_cast<size_t>(n) * fn::hc::w_down_q8_bytes};
            return report(head ? fn::hc::head_down2_q8(rows, dq, k.st) : fn::hc::down2_q8(rows, dq, k.st), "down2_q8");
        }
        return report(head ? fn::hc::head_down2(rows, d2, k.st) : fn::hc::down2(rows, d2, k.st), "down2");
    }

    bool up(int32_t rows, int32_t r, int32_t n, fn::hc::up_arm arm, bool pre, bool q8 = false,
            bool head = false) const {
        card2 & k = *c[r];
        const fn::hc::up2_args ua = {q8 ? nullptr : k.w_up.p + static_cast<size_t>(n) * fn::hc::w_up_elems,
                                     k.mom.p,
                                     pre ? k.pre_hc.p : rg->slot(r, fn::xchg::msg::hc, slot_of(n)),
                                     k.x.p,
                                     k.gamma.p,
                                     head ? nullptr : k.inject.p,
                                     pre ? k.own_hmix.p : rg->slot(r, fn::xchg::msg::hmix, slot_of(n)),
                                     pre ? c[1 - r]->sink_hmix.p : rg->slot(1 - r, fn::xchg::msg::hmix, slot_of(n)),
                                     k.runs.p,
                                     rg->err(r),
                                     r};
        if (q8) {
            const fn::hc::up2_q8_args uq = {ua, k.q_up.p + static_cast<size_t>(n) * fn::hc::w_up_q8_bytes};
            return report(arm == fn::hc::up_arm::ring &&
                              (head ? fn::hc::head_up2_q8(rows, uq, k.st) : fn::hc::up2_q8(rows, uq, k.st)),
                          "up2_q8");
        }
        if (head) {
            return report(arm == fn::hc::up_arm::ring && fn::hc::head_up2(rows, ua, k.st), "head_up2");
        }
        return report(fn::hc::up2(arm, rows, ua, k.st), "up2");
    }

    // HC4 (B3): the sublayer in one launch (hc::sub, mode 3); pre: the peer's moments landed (private slot n of
    // pre_sub), sends to sinks on the peer, no resets
    bool fused(int32_t rows, int32_t r, int32_t n, bool pre, bool head = false) const {
        using fn::xchg::msg;
        card2 & k = *c[r];
        const int32_t prev = slot_of((n + n_sub - 1) % n_sub);
        fn::hc::sub_args a = {};
        a.w_down = k.w_down.p + static_cast<size_t>(n) * fn::hc::w_down_elems;
        a.w_up = k.w_up.p + static_cast<size_t>(n) * fn::hc::w_up_elems;
        a.hres = k.hres.p;
        a.hx = k.hx.p;
        a.hpost = k.hpost.p;
        a.gamma = k.gamma.p;
        a.x = k.x.p;
        a.inject = k.inject.p;
        a.hc_ring = pre ? k.pre_sub.p + static_cast<size_t>(n) * card2::n_hc_slot : rg->slot(r, msg::hc, slot_of(n));
        a.hc_peer = pre ? c[1 - r]->sink_hc.p : rg->slot(1 - r, msg::hc, slot_of(n));
        a.hc_reset = pre ? nullptr : rg->slot(r, msg::hc, prev);
        a.hmix_reset = pre ? nullptr : rg->slot(r, msg::hmix, prev);
        a.hmix_own = pre ? k.own_hmix.p : rg->slot(r, msg::hmix, slot_of(n));
        a.hmix_peer = pre ? c[1 - r]->sink_hmix.p : rg->slot(1 - r, msg::hmix, slot_of(n));
        a.err = rg->err(r);
        a.card = r;
        return report(head ? fn::hc::head_sub(rows, 3, a, k.st) : fn::hc::sub(rows, 3, a, k.st), "sub");
    }

    // restores every card's pre_sub from pre_sub_init (before a replay of pre-filled one-launch units)
    bool restore_pre_sub(const bool use[2]) const {
        bool ok = true;
        for (int32_t r = 0; r < 2 && ok; ++r) {
            if (use[r]) {
                card2 & k = *c[r];
                ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
                     cuda_ok(cudaMemcpyAsync(k.pre_sub.p, k.pre_sub_init.p, size_t(n_sub) * card2::n_hc_slot * 4,
                                             cudaMemcpyDeviceToDevice, k.st),
                             "pre_sub");
            }
        }
        return ok;
    }

    // the stand-in consumer: polls the peer's half of the slot (all three regions, HC3) and assembles the F32 row
    bool ready(int32_t rows, int32_t r, int32_t n, bool pre) const {
        card2 & k = *c[r];
        const fn::hc::ready_args ra = {pre ? k.pre_hmix.p : rg->slot(r, fn::xchg::msg::hmix, slot_of(n)),
                                       k.full.p,
                                       nullptr,
                                       nullptr,
                                       rg->err(r),
                                       r,
                                       pre ? 0 : 1};
        return report(fn::hc::ready(rows, ra, k.st), "ready");
    }

    static bool report(bool sent, const char * what) {
        if (!sent) {
            fprintf(stderr, "hc-bench: %s not enqueued: %s\n", what, cudaGetErrorString(cudaGetLastError()));
        }
        return sent;
    }
};

// waits for both cards' end events with a host timeout
bool wait_events(card2 * c[2], const bool use[2]) {
    const auto t0 = std::chrono::steady_clock::now();
    bool done[2] = {!use[0], !use[1]};
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
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > hc2_timeout_s) {
            fprintf(stderr, "hc-bench: host timeout after %.0f s\n", hc2_timeout_s);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

// Captures n_sub units on each card in use, replays both cards' graphs together (card 0 launched first) n_warm times
// untimed and n_timed timed, and prints one JSON line: per card, median and p90 of the replay's event time / n_sub.
// HC4: restore, when set, runs before every replay, outside the timed events (the pre-filled one-launch points)
bool time_point2(bench2 & bn, const char * point, const char * what, const char * arm, int32_t rows, const bool use[2],
                 int32_t warm, int32_t timed, const std::function<bool(int32_t, int32_t)> & unit,
                 bool restore = false) {
    cudaGraphExec_t exec[2] = {nullptr, nullptr};
    bool ok = true;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        if (!use[r]) {
            continue;
        }
        cudaGraph_t graph = nullptr;
        ok = cuda_ok(cudaSetDevice(bn.c[r]->dev), "cudaSetDevice") &&
             cuda_ok(cudaStreamBeginCapture(bn.c[r]->st, cudaStreamCaptureModeThreadLocal), "begin capture");
        for (int32_t i = 0; i < n_sub && ok; ++i) {
            ok = unit(r, i);
        }
        ok = cuda_ok(cudaStreamEndCapture(bn.c[r]->st, &graph), "end capture") && ok;
        ok = ok && cuda_ok(cudaGraphInstantiate(&exec[r], graph, 0), "instantiate");
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
    }
    std::vector<double> us[2];
    std::string t0;
    for (int32_t it = 0; it < warm + timed && ok; ++it) {
        if (it == warm) {
            t0 = utc_now();
        }
        ok = !restore || bn.restore_pre_sub(use);
        for (int32_t r = 0; r < 2 && ok; ++r) {
            if (use[r]) {
                card2 & k = *bn.c[r];
                ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && cuda_ok(cudaEventRecord(k.e0, k.st), "record") &&
                     cuda_ok(cudaGraphLaunch(exec[r], k.st), "replay") &&
                     cuda_ok(cudaEventRecord(k.e1, k.st), "record");
            }
        }
        ok = ok && wait_events(bn.c, use);
        for (int32_t r = 0; r < 2 && ok && it >= warm; ++r) {
            if (use[r]) {
                float ms = 0.0f;
                ok = cuda_ok(cudaEventElapsedTime(&ms, bn.c[r]->e0, bn.c[r]->e1), "elapsed");
                us[r].push_back(1000.0 * ms / n_sub);
            }
        }
    }
    const std::string t1 = utc_now();
    for (int32_t r = 0; r < 2; ++r) {
        if (exec[r] != nullptr) {
            cudaSetDevice(bn.c[r]->dev);
            cudaGraphExecDestroy(exec[r]);
        }
    }
    uint32_t err[2] = {0u, 0u};
    ok = ok && bn.rg->read_err(0, err[0]) && bn.rg->read_err(1, err[1]);
    if (ok && (err[0] != 0u || err[1] != 0u)) {
        fprintf(stderr, "hc-bench: %s: error words 0x%x 0x%x\n", point, err[0], err[1]);
        ok = false;
    }
    if (!ok) {
        return false;
    }
    printf("{\"ticket\": \"HC2\", \"point\": \"%s\", \"what\": \"%s\", \"arm\": \"%s\", \"rows\": %d, "
           "\"units_per_replay\": %d, \"warm_replays\": %d, \"n\": %d",
           point, what, arm, rows, n_sub, warm, timed);
    for (int32_t r = 0; r < 2; ++r) {
        if (!use[r]) {
            continue;
        }
        std::vector<double> sorted = us[r];
        std::sort(sorted.begin(), sorted.end());
        const size_t n = sorted.size();
        const double median = n % 2 == 0 ? 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) : sorted[n / 2];
        const double p90 = sorted[(9 * n + 9) / 10 - 1];
        printf(", \"card%d\": {\"median_us\": %.3f, \"p90_us\": %.3f, \"us\": [", r, median, p90);
        for (size_t i = 0; i < n; ++i) {
            printf("%s%.3f", i == 0 ? "" : ", ", us[r][i]);
        }
        printf("]}");
    }
    printf(", \"t_start\": \"%s\", \"t_end\": \"%s\"}\n", t0.c_str(), t1.c_str());
    fflush(stdout);
    return true;
}

// HC4 cold mode (R17, v-bench's (A - B) method on both cards together): per card in use, graph A = n_cold_rep x [L2
// flush, instruction-cache eviction, unit], graph B = n_cold_rep x [flush, eviction]; each trial replays B on both
// cards, then A on both (card 0 launched first), and a card's cold unit is (A - B) / n_cold_rep. Prints one JSON line
// as time_point2's, with "cold": true.
bool time_point2_cold(bench2 & bn, const char * point, const char * what, const char * arm, int32_t rows,
                      const bool use[2], const std::function<bool(int32_t, int32_t)> & unit, bool restore = false) {
    cudaGraphExec_t ga[2] = {nullptr, nullptr}, gb[2] = {nullptr, nullptr};
    bool ok = true;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        if (!use[r]) {
            continue;
        }
        card2 & k = *bn.c[r];
        for (const bool with_unit : {true, false}) {
            cudaGraph_t graph = nullptr;
            ok = ok && cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
                 cuda_ok(cudaStreamBeginCapture(k.st, cudaStreamCaptureModeThreadLocal), "begin capture");
            for (int32_t i = 0; i < n_cold_rep && ok; ++i) {
                ok = k.cold() && (!with_unit || unit(r, i));
            }
            ok = cuda_ok(cudaStreamEndCapture(k.st, &graph), "end capture") && ok;
            ok = ok && cuda_ok(cudaGraphInstantiate(with_unit ? &ga[r] : &gb[r], graph, 0), "instantiate");
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
        }
    }
    const auto replay = [&](cudaGraphExec_t g[2], double out_us[2]) {
        bool rok = true;
        for (int32_t r = 0; r < 2 && rok; ++r) {
            if (use[r]) {
                card2 & k = *bn.c[r];
                rok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
                      cuda_ok(cudaEventRecord(k.e0, k.st), "record") &&
                      cuda_ok(cudaGraphLaunch(g[r], k.st), "replay") && cuda_ok(cudaEventRecord(k.e1, k.st), "record");
            }
        }
        rok = rok && wait_events(bn.c, use);
        for (int32_t r = 0; r < 2 && rok; ++r) {
            if (use[r]) {
                float ms = 0.0f;
                rok = cuda_ok(cudaEventElapsedTime(&ms, bn.c[r]->e0, bn.c[r]->e1), "elapsed");
                out_us[r] = 1000.0 * ms;
            }
        }
        return rok;
    };
    std::vector<double> us[2];
    std::string t0;
    for (int32_t it = 0; it < n_warm + n_timed && ok; ++it) {
        if (it == n_warm) {
            t0 = utc_now();
        }
        double ta[2] = {0.0, 0.0}, tb[2] = {0.0, 0.0};
        ok = replay(gb, tb) && (!restore || bn.restore_pre_sub(use)) && replay(ga, ta);
        for (int32_t r = 0; r < 2 && ok && it >= n_warm; ++r) {
            if (use[r]) {
                us[r].push_back((ta[r] - tb[r]) / n_cold_rep);
            }
        }
    }
    const std::string t1 = utc_now();
    for (int32_t r = 0; r < 2; ++r) {
        for (cudaGraphExec_t g : {ga[r], gb[r]}) {
            if (g != nullptr) {
                cudaSetDevice(bn.c[r]->dev);
                cudaGraphExecDestroy(g);
            }
        }
    }
    uint32_t err[2] = {0u, 0u};
    ok = ok && bn.rg->read_err(0, err[0]) && bn.rg->read_err(1, err[1]);
    if (ok && (err[0] != 0u || err[1] != 0u)) {
        fprintf(stderr, "hc-bench: %s: error words 0x%x 0x%x\n", point, err[0], err[1]);
        ok = false;
    }
    if (!ok) {
        return false;
    }
    printf("{\"ticket\": \"HC4\", \"point\": \"%s\", \"what\": \"%s\", \"arm\": \"%s\", \"rows\": %d, \"cold\": true, "
           "\"method\": \"(A - B) / %d: 12 MB L2 flush and instruction-cache eviction before each unit\", "
           "\"warm_replays\": %d, \"n\": %d",
           point, what, arm, rows, n_cold_rep, n_warm, n_timed);
    for (int32_t r = 0; r < 2; ++r) {
        if (!use[r]) {
            continue;
        }
        std::vector<double> sorted = us[r];
        std::sort(sorted.begin(), sorted.end());
        const size_t n = sorted.size();
        const double median = n % 2 == 0 ? 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) : sorted[n / 2];
        const double p90 = sorted[(9 * n + 9) / 10 - 1];
        printf(", \"card%d\": {\"median_us\": %.3f, \"p90_us\": %.3f, \"us\": [", r, median, p90);
        for (size_t i = 0; i < n; ++i) {
            printf("%s%.3f", i == 0 ? "" : ", ", us[r][i]);
        }
        printf("]}");
    }
    printf(", \"t_start\": \"%s\", \"t_end\": \"%s\"}\n", t0.c_str(), t1.c_str());
    fflush(stdout);
    return true;
}

// Each card's X-HC and X-HMIX slots as they land, copied to pre_hc and pre_hmix: down2 on both cards (sends only),
// then up2 on both (its poll finds the moments already there), at max_rows. Nothing waits on the other card.
bool prefill(bench2 & bn) {
    using fn::xchg::msg;
    bool ok = bn.rg->reset_all();
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(bn.c[r]->dev), "cudaSetDevice") && bn.down(R, r, 1, false) &&
             cuda_ok(cudaStreamSynchronize(bn.c[r]->st), "prefill down");
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(bn.c[r]->dev), "cudaSetDevice") && bn.up(R, r, 1, fn::hc::up_arm::ring, false) &&
             cuda_ok(cudaStreamSynchronize(bn.c[r]->st), "prefill up");
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card2 & k = *bn.c[r];
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
             cuda_ok(cudaMemcpy(k.pre_hc.p, bn.rg->slot(r, msg::hc, slot_of(1)), size_t(R) * M * sizeof(float),
                                cudaMemcpyDeviceToDevice),
                     "pre_hc") &&
             cuda_ok(cudaMemcpy(k.pre_hmix.p, bn.rg->slot(r, msg::hmix, slot_of(1)), card2::n_hmix * sizeof(float),
                                cudaMemcpyDeviceToDevice),
                     "pre_hmix");
    }
    // HC4: pre_sub_init, n_sub private X-HC slots a card: the peer's region as landed (pre_hc), the own region the
    // sentinel
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card2 & k = *bn.c[r];
        std::vector<float> landed(size_t(R) * M);
        std::vector<uint32_t> slot(card2::n_hc_slot, fn::xchg::sentinel);
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
             cuda_ok(cudaMemcpy(landed.data(), k.pre_hc.p, landed.size() * sizeof(float), cudaMemcpyDeviceToHost),
                     "pre_hc");
        memcpy(slot.data(), landed.data(), landed.size() * sizeof(float));
        for (int32_t n = 0; n < n_sub && ok; ++n) {
            ok = cuda_ok(cudaMemcpy(k.pre_sub_init.p + size_t(n) * card2::n_hc_slot, slot.data(), slot.size() * 4,
                                    cudaMemcpyHostToDevice),
                         "pre_sub_init");
        }
    }
    return ok && bn.rg->reset_all();
}

// keeps both cards busy for warm_s: memsets of a 64 MiB scratch buffer on each, no kernel of ours
bool busy_both(card2 * c[2], double secs) {
    bool ok = true;
    void * scratch[2] = {nullptr, nullptr};
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") &&
             cuda_ok(cudaMalloc(&scratch[r], size_t(64) << 20), "scratch");
    }
    const auto t0 = std::chrono::steady_clock::now();
    int64_t n_memset = 0;
    while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        for (int32_t i = 0; i < 16 && ok; ++i, ++n_memset) {
            for (int32_t r = 0; r < 2 && ok; ++r) {
                ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") &&
                     cuda_ok(cudaMemsetAsync(scratch[r], static_cast<int>(n_memset & 0xff), size_t(64) << 20, c[r]->st),
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
    printf("# hc2-bench: both cards busy for %.2f s (%lld memsets of 64 MiB each), timing from %s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
           static_cast<long long>(n_memset), utc_now().c_str());
    return ok;
}

// DR1: Q8_0 weights for n_sub sets of `bytes` each (`elems` weights: the int8 plane, then the FP16 scale plane): set 0
// from the host (uniform int8, scales amax / 127 times [0.5, 1)), copied to the other sets on the device
bool fill_q8(dev_buf<uint8_t> & d, int64_t bytes, int64_t elems, std::mt19937 & rng, float amax, cudaStream_t st) {
    std::vector<uint8_t> h(static_cast<size_t>(bytes));
    std::uniform_int_distribution<int32_t> dq(-127, 127);
    std::uniform_real_distribution<float> dd(0.5f, 1.0f);
    for (int64_t i = 0; i < elems; ++i) {
        h[static_cast<size_t>(i)] = static_cast<uint8_t>(dq(rng));
    }
    for (int64_t i = elems; i + 1 < bytes; i += 2) {
        const uint16_t v = fn::quant::f32_to_f16(amax / 127.0f * dd(rng));
        memcpy(&h[static_cast<size_t>(i)], &v, sizeof(v));
    }
    bool ok = d.p != nullptr && cuda_ok(cudaMemcpy(d.p, h.data(), h.size(), cudaMemcpyHostToDevice), "q8 set 0");
    for (int32_t n = 1; n < n_sub && ok; ++n) {
        ok = cuda_ok(cudaMemcpyAsync(d.p + n * bytes, d.p, static_cast<size_t>(bytes), cudaMemcpyDeviceToDevice, st),
                     "q8 sets");
    }
    return ok && cuda_ok(cudaStreamSynchronize(st), "q8 fill");
}

} // namespace

int fnfl_hc_bench(int device) {
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaDeviceProp prop = {};
    int driver = 0, runtime = 0;
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "properties") ||
        !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
        return 1;
    }
    printf("# hc-bench: %s, sm_%d%d, %d SMs, driver API %d, runtime %d, start %s\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, driver, runtime, utc_now().c_str());

    bool ok = true;
    {
        bench bn;
        ok = cuda_ok(cudaStreamCreateWithFlags(&bn.st, cudaStreamNonBlocking), "stream");
        std::mt19937 rng(7);
        ok = ok && bn.w_down.p != nullptr && bn.w_up.p != nullptr;
        ok = ok && fill_weights(bn.w_down, fn::hc::w_down_elems, rng, 0.07f, bn.st) &&
             fill_weights(bn.w_up, fn::hc::w_up_elems, rng, 0.25f, bn.st) &&
             fill(bn.hpost, size_t(R) * S, rng, 0.5f, 1.5f, bn.st);
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = fill(bn.card[r].hres, card_bufs::n_x, rng, -2.0f, 2.0f, bn.st) &&
                 fill(bn.card[r].hx, card_bufs::n_row, rng, -2.0f, 2.0f, bn.st) &&
                 fill(bn.card[r].gamma, card_bufs::n_gamma, rng, 0.5f, 1.5f, bn.st);
        }
        // the peer moments and card 0's own inputs to hc_up, once per row count (card 1's weights are sets 32..63)
        for (const int32_t rows : {1, 3}) {
            ok = ok && bn.down(rows, 1, n_sets / 2) && bn.down(rows, 0, 0);
        }
        ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "prefill");

        // keep the card busy for warm_s before timing: memsets of a 64 MiB scratch buffer, no kernel of ours
        if (ok) {
            dev_buf<uint8_t> scratch(size_t(64) << 20);
            const auto t0 = std::chrono::steady_clock::now();
            int64_t n_memset = 0;
            while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
                for (int32_t i = 0; i < 32 && ok; ++i, ++n_memset) {
                    const int value = static_cast<int>(n_memset & 0xff);
                    ok = cuda_ok(cudaMemsetAsync(scratch.p, value, size_t(64) << 20, bn.st), "memset");
                }
                ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "warm sync");
            }
            printf("# hc-bench: busy for %.2f s (%lld memsets of 64 MiB), timing from %s\n",
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                   static_cast<long long>(n_memset), utc_now().c_str());
        }

        // HC1c step 6: the stream-only probe and the kernels at depths 1, 2, 3 and 6, B = 3, card 0's data
        static const int32_t depths[] = {1, 2, 3, 6};
        for (const int32_t d : depths) {
            const std::string p = "probe_d" + std::to_string(d);
            ok = ok && time_point(bn, p.c_str(), "stream-only probe", 3, 1, [&](int32_t i) { return bn.probe(d, i); });
        }
        for (const int32_t d : depths) {
            const std::string p = "A_down_d" + std::to_string(d) + "_B3";
            ok = ok && time_point(bn, p.c_str(), "hc_down at a depth", 3, 1,
                                  [&](int32_t i) { return bn.down(3, 0, i, {false, d}); });
        }
        for (const int32_t d : depths) {
            const std::string p = "A_up_d" + std::to_string(d) + "_B3";
            ok = ok && time_point(bn, p.c_str(), "hc_up at a depth", 3, 1,
                                  [&](int32_t i) { return bn.up(3, 0, i, {false, d}); });
        }
        // the shipped kernels: alone at B = 1 and 3, then with their surrounding parts
        const fn::hc::variant_id sh = fn::hc::shipped();
        printf("# hc-bench: shipped hc_down depth %d, hc_up depth %d\n", sh.depth_down, sh.depth_up);
        for (const int32_t rows : {1, 3}) {
            const std::string p = "iso_down_B" + std::to_string(rows);
            ok = ok && time_point(bn, p.c_str(), "hc_down (shipped)", rows, 1,
                                  [&](int32_t i) { return bn.down(rows, 0, i); });
        }
        for (const int32_t rows : {1, 3}) {
            const std::string p = "iso_up_B" + std::to_string(rows);
            ok = ok &&
                 time_point(bn, p.c_str(), "hc_up (shipped)", rows, 1, [&](int32_t i) { return bn.up(rows, 0, i); });
        }
        for (const int32_t rows : {1, 3}) {
            const std::string p = "ctx_card_B" + std::to_string(rows);
            ok = ok && time_point(bn, p.c_str(), "hc_down + hc_up (card 0)", rows, 2,
                                  [&](int32_t i) { return bn.down(rows, 0, i) && bn.up(rows, 0, i); });
        }
        for (const int32_t rows : {1, 3}) {
            const std::string p = "full_width_B" + std::to_string(rows);
            const auto unit = [&](int32_t i) {
                const int32_t j = (i + n_sets / 2) % n_sets;
                return bn.down(rows, 0, i) && bn.down(rows, 1, j) && bn.up(rows, 0, i) && bn.up(rows, 1, j);
            };
            ok = ok && time_point(bn, p.c_str(), "hc_down x2 + hc_up x2 (both halves on card 0)", rows, 4, unit);
        }
        ok = cuda_ok(cudaStreamSynchronize(bn.st), "final sync") && ok;
        if (bn.st != nullptr) {
            cudaStreamDestroy(bn.st);
        }
    }
    printf("# hc-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}

int fnfl_hc2_bench(int dev0, int dev1, const char * mode) {
    const std::string m = mode;
    if (m != "all" && m != "ncu" && m != "nsys" && m != "q8" && m != "cold" && m != "ncu-up" && m != "ncu-fused") {
        fprintf(stderr, "hc2-bench: MODE is all, ncu, ncu-up, ncu-fused, nsys, q8 or cold\n");
        return 2;
    }
    if (!fn::xchg::enable_peer_access(dev0, dev1)) {
        return 1;
    }
    for (const int32_t d : {dev0, dev1}) {
        cudaDeviceProp prop = {};
        int driver = 0, runtime = 0;
        if (!cuda_ok(cudaGetDeviceProperties(&prop, d), "properties") ||
            !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
            return 1;
        }
        printf("# hc2-bench: device %d: %s, sm_%d%d, %d SMs, PCI %02x:%02x, driver API %d, runtime %d, mode %s, start "
               "%s\n",
               d, prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.pciBusID, prop.pciDeviceID, driver,
               runtime, mode, utc_now().c_str());
    }
    using fn::hc::up_arm;
    bool ok = true;
    {
        card2 k0(dev0, m == "q8"), k1(dev1, m == "q8");
        card2 * c[2] = {&k0, &k1};
        fn::xchg::rings rg(dev0, dev1);
        bench2 bn = {{&k0, &k1}, &rg};
        ok = rg.ok() && k0.st != nullptr && k1.st != nullptr;
        std::mt19937 rng(7);
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card2 & k = *c[r];
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::hc::prepare2() && k.w_down.p != nullptr &&
                 cuda_ok(cudaMemset(k.runs.p, 0, size_t(fn::hc::n_runs_card) * sizeof(uint32_t)), "runs") &&
                 k.w_up.p != nullptr && fill_weights(k.w_down, fn::hc::w_down_elems, rng, 0.07f, k.st) &&
                 fill_weights(k.w_up, fn::hc::w_up_elems, rng, 0.25f, k.st) &&
                 fill(k.hres, card2::n_x, rng, -2.0f, 2.0f, k.st) && fill(k.hx, card2::n_row, rng, -2.0f, 2.0f, k.st) &&
                 fill(k.gamma, card2::n_gamma, rng, 0.5f, 1.5f, k.st);
            ok = ok &&
                 (!k.q8 ||
                  (fill_q8(k.q_down, fn::hc::w_down_q8_bytes, int64_t(S) * fn::hc::n_rows_down * F, rng, 0.07f, k.st) &&
                   fill_q8(k.q_up, fn::hc::w_up_q8_bytes, int64_t(F) * S * fn::hc::n_down, rng, 0.25f, k.st)));
        }
        // hpost is replicated: the same values on both cards
        std::vector<float> post(size_t(R) * S);
        std::uniform_real_distribution<float> d_post(0.5f, 1.5f);
        for (auto & v : post) {
            v = d_post(rng);
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") &&
                 cuda_ok(cudaMemcpy(c[r]->hpost.p, post.data(), post.size() * sizeof(float), cudaMemcpyHostToDevice),
                         "hpost") &&
                 cuda_ok(cudaDeviceSynchronize(), "hpost"); // a pageable copy may return before the DMA lands
        }
        ok = ok && prefill(bn);
        ok = ok && busy_both(c, m == "ncu" ? 0.5 : warm_s);

        const bool both[2] = {true, true};
        const bool card0[2] = {true, false};
        const auto arm_name = [](up_arm a) { return a == up_arm::ring ? "0" : "A"; };
        const auto sub = [&](int32_t rows, up_arm a) {
            return [&bn, rows, a](int32_t r, int32_t n) {
                return bn.down(rows, r, n, false) && bn.up(rows, r, n, a, false) && bn.ready(rows, r, n, false);
            };
        };
        if (m == "ncu") {
            // card 0's pre-filled arm A hc_up at B = 3: 64 launches a replay, one warm replay and one profiled
            ok = ok &&
                 time_point2(bn, "ncu_pre_up_armA_B3", "hc_up2 arm A, pre-filled X-HC (card 0 only)", "A", 3, card0, 1,
                             1, [&bn](int32_t r, int32_t n) { return bn.up(3, r, n, up_arm::staged, true); });
        } else if (m == "ncu-up" || m == "ncu-fused") {
            // HC4: card 0's pre-filled up2 arm 0 (B2's shape) or one-launch sublayer at B = 3: 64 launches a replay,
            // one warm replay and one profiled
            const bool f = m == "ncu-fused";
            ok = ok && time_point2(
                           bn, f ? "ncu_pre_fused_B3" : "ncu_pre_up_arm0_B3",
                           f ? "sub alone, the peer's X-HC pre-filled (card 0 only)"
                             : "up2 arm 0, pre-filled X-HC (card 0 only)",
                           f ? "F" : "0", 3, card0, 1, 1,
                           [&bn, f](int32_t r, int32_t n) {
                               return f ? bn.fused(3, r, n, true) : bn.up(3, r, n, up_arm::ring, true);
                           },
                           f);
        } else if (m == "q8") {
            // DR1: the Q8_0 forms beside the BF16 ones in one process, the layer (arm 0) and the head, B = 1 and 3:
            // the sublayer on both cards, then down and up alone with the peer's data pre-filled
            for (const bool head : {false, true}) {
                for (const int32_t rows : {1, 3}) {
                    for (const bool q8 : {false, true}) {
                        const std::string p =
                            std::string(head ? "head_" : "") + (q8 ? "q8_" : "bf16_") + "sub_B" + std::to_string(rows);
                        ok = ok && time_point2(bn, p.c_str(), "sublayer (down2, up2, ready) on both cards", "0", rows,
                                               both, n_warm, n_timed, [&bn, rows, q8, head](int32_t r, int32_t n) {
                                                   return bn.down(rows, r, n, false, q8, head) &&
                                                          bn.up(rows, r, n, up_arm::ring, false, q8, head) &&
                                                          bn.ready(rows, r, n, false);
                                               });
                    }
                }
                for (const int32_t rows : {1, 3}) {
                    for (const bool q8 : {false, true}) {
                        const std::string sfx = std::string(q8 ? "q8_" : "bf16_") + "B" + std::to_string(rows);
                        const std::string pd = std::string(head ? "head_" : "") + "pre_down_" + sfx;
                        const std::string pu = std::string(head ? "head_" : "") + "pre_up_" + sfx;
                        ok = ok && time_point2(bn, pd.c_str(), "down2 alone, sends to a sink", "-", rows, both, n_warm,
                                               n_timed, [&bn, rows, q8, head](int32_t r, int32_t n) {
                                                   return bn.down(rows, r, n, true, q8, head);
                                               });
                        const auto up_unit = [&bn, rows, q8, head](int32_t r, int32_t n) {
                            return bn.up(rows, r, n, up_arm::ring, true, q8, head);
                        };
                        ok = ok && time_point2(bn, pu.c_str(), "up2 alone, X-HC pre-filled, sends to a sink", "0", rows,
                                               both, n_warm, n_timed, up_unit);
                    }
                }
            }
        } else if (m == "cold") {
            // HC4: arm 0's launches and the sublayer cold (R17), B = 1 and 3, both cards: down2 and up2 alone with the
            // peer's data pre-filled, and the sublayer (down2, up2, ready) on both cards
            for (const int32_t rows : {1, 3}) {
                const std::string pd = "cold_pre_down_B" + std::to_string(rows);
                ok = ok && time_point2_cold(bn, pd.c_str(), "down2 alone, sends to a sink", "-", rows, both,
                                            [&bn, rows](int32_t r, int32_t n) { return bn.down(rows, r, n, true); });
                const std::string pu = "cold_pre_up_arm0_B" + std::to_string(rows);
                ok = ok && time_point2_cold(
                               bn, pu.c_str(), "up2 alone, X-HC pre-filled, sends to a sink", "0", rows, both,
                               [&bn, rows](int32_t r, int32_t n) { return bn.up(rows, r, n, up_arm::ring, true); });
                const std::string ps = "cold_sub_arm0_B" + std::to_string(rows);
                ok = ok && time_point2_cold(bn, ps.c_str(), "sublayer (down2, up2, ready) on both cards", "0", rows,
                                            both, sub(rows, up_arm::ring));
                // HC4 (B3): the one-launch sublayer, in context and pre-filled
                const std::string pf = "cold_sub_fused_B" + std::to_string(rows);
                ok = ok && time_point2_cold(bn, pf.c_str(), "sublayer (sub, ready) on both cards", "F", rows, both,
                                            [&bn, rows](int32_t r, int32_t n) {
                                                return bn.fused(rows, r, n, false) && bn.ready(rows, r, n, false);
                                            });
                const std::string pp = "cold_pre_fused_B" + std::to_string(rows);
                ok = ok && time_point2_cold(
                               bn, pp.c_str(), "sub alone, the peer's X-HC pre-filled, sends to a sink", "F", rows,
                               both, [&bn, rows](int32_t r, int32_t n) { return bn.fused(rows, r, n, true); }, true);
            }
        } else if (m == "nsys") {
            ok = ok && time_point2(bn, "nsys_sub_armA_B3", "sublayer (down2, up2, ready) on both cards", "A", 3, both,
                                   3, 2, sub(3, up_arm::staged));
        } else {
            for (const up_arm a : {up_arm::ring, up_arm::staged}) {
                for (const int32_t rows : {1, 3}) {
                    const std::string p = std::string("sub_arm") + arm_name(a) + "_B" + std::to_string(rows);
                    ok = ok && time_point2(bn, p.c_str(), "sublayer (down2, up2, ready) on both cards", arm_name(a),
                                           rows, both, n_warm, n_timed, sub(rows, a));
                }
            }
            // pre-filled: the peer's data already landed (X-HC and X-HMIX copies), sends go to a sink on the peer
            for (const int32_t rows : {1, 3}) {
                const std::string p = "pre_down_B" + std::to_string(rows);
                ok = ok && time_point2(bn, p.c_str(), "down2 alone, sends to a sink", "-", rows, both, n_warm, n_timed,
                                       [&bn, rows](int32_t r, int32_t n) { return bn.down(rows, r, n, true); });
            }
            for (const up_arm a : {up_arm::ring, up_arm::staged}) {
                for (const int32_t rows : {1, 3}) {
                    const std::string p = std::string("pre_up_arm") + arm_name(a) + "_B" + std::to_string(rows);
                    ok = ok && time_point2(bn, p.c_str(), "up2 alone, X-HC pre-filled, sends to a sink", arm_name(a),
                                           rows, both, n_warm, n_timed,
                                           [&bn, rows, a](int32_t r, int32_t n) { return bn.up(rows, r, n, a, true); });
                }
            }
            for (const int32_t rows : {1, 3}) {
                const std::string p = "pre_ready_B" + std::to_string(rows);
                ok = ok &&
                     time_point2(bn, p.c_str(), "ready alone, X-HMIX pre-filled, no reset", "-", rows, both, n_warm,
                                 n_timed, [&bn, rows](int32_t r, int32_t n) { return bn.ready(rows, r, n, true); });
            }
            // HC4 (B3): the sublayer in one launch, in context (with ready) and pre-filled (alone)
            for (const int32_t rows : {1, 3}) {
                const std::string p = "sub_fused_B" + std::to_string(rows);
                ok = ok && time_point2(bn, p.c_str(), "sublayer (sub, ready) on both cards", "F", rows, both, n_warm,
                                       n_timed, [&bn, rows](int32_t r, int32_t n) {
                                           return bn.fused(rows, r, n, false) && bn.ready(rows, r, n, false);
                                       });
                const std::string pf = "pre_fused_B" + std::to_string(rows);
                ok = ok && time_point2(
                               bn, pf.c_str(), "sub alone, the peer's X-HC pre-filled, sends to a sink", "F", rows,
                               both, n_warm, n_timed,
                               [&bn, rows](int32_t r, int32_t n) { return bn.fused(rows, r, n, true); }, true);
            }
            for (const up_arm a : {up_arm::ring, up_arm::staged}) {
                for (const int32_t rows : {1, 3}) {
                    const std::string p = std::string("pre_sub_arm") + arm_name(a) + "_B" + std::to_string(rows);
                    ok = ok && time_point2(bn, p.c_str(), "down2 + up2 + ready, all pre-filled (no wait)", arm_name(a),
                                           rows, both, n_warm, n_timed, [&bn, rows, a](int32_t r, int32_t n) {
                                               return bn.down(rows, r, n, true) && bn.up(rows, r, n, a, true) &&
                                                      bn.ready(rows, r, n, true);
                                           });
                }
            }
        }
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(c[r]->dev);
            ok = cuda_ok(cudaDeviceSynchronize(), "final sync") && ok;
        }
    }
    printf("# hc2-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
