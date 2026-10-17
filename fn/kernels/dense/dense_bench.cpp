#include "fn/kernels/dense/dense_bench.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_cold.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/gdn/gdn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace fn::dense;

constexpr int32_t n_unit = 64;     // launches in a warm graph
constexpr int32_t n_seq_rep = 16;  // layer repeats in a step-7 graph
constexpr int32_t n_warm = 3;      // untimed replays (warm) and trials (cold)
constexpr int32_t n_timed = 20;    // timed replays (warm) and trials (cold)
constexpr int32_t n_cold_rep = 20; // launches in a cold graph (XP1's rep)
constexpr double warm_s = 2.0;     // HX12 Later 15: a process starts at 1,260 MHz for 0.3-1 s
constexpr size_t min_cold = size_t(200) << 20;
constexpr int64_t flush_used = 12ll << 20; // > 6 MiB of L2 (XP1)
constexpr double ramp_us = 1.949;          // HX12 card 0 at 1,380 MHz
constexpr double dram_gbps = 1021.6;       // HX12
constexpr double tail_single_us = 0.31;    // HX12: a single-pass launch's tail
constexpr double tail_across_us = 0.83;    // HX12: a launch that finishes across blocks
constexpr int32_t D = 2560;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "dp-bench: %s: %s\n", what, cudaGetErrorString(e));
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

// owns one device allocation of n bytes
struct dev_mem {
    uint8_t * p = nullptr;
    size_t n = 0;

    explicit dev_mem(size_t bytes) : n(std::max<size_t>(bytes, 16)) {
        if (!cuda_ok(cudaMalloc(&p, n), "cudaMalloc")) {
            p = nullptr;
        }
    }

    ~dev_mem() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dev_mem(const dev_mem &) = delete;
    dev_mem & operator=(const dev_mem &) = delete;

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); }
    bool put(const void * h, size_t bytes) const {
        return p != nullptr && bytes <= n && cuda_ok(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice), "put");
    }
};

// owns a graph executable
struct graph_exec {
    cudaGraphExec_t x = nullptr;

    graph_exec() = default;

    ~graph_exec() {
        if (x != nullptr) {
            cudaGraphExecDestroy(x);
        }
    }

    graph_exec(const graph_exec &) = delete;
    graph_exec & operator=(const graph_exec &) = delete;
};

// owns two timing events
struct event_pair {
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    event_pair() = default;

    ~event_pair() {
        if (e0 != nullptr) {
            cudaEventDestroy(e0);
        }
        if (e1 != nullptr) {
            cudaEventDestroy(e1);
        }
    }

    event_pair(const event_pair &) = delete;
    event_pair & operator=(const event_pair &) = delete;

    bool make() { return cuda_ok(cudaEventCreate(&e0), "event") && cuda_ok(cudaEventCreate(&e1), "event"); }
};

struct mdef {
    const char * name;
    fmt f;
    int32_t m;
    epi e;
    int32_t pair;
    bool scaled; // the shared down: scaled by the shared gate's sigmoid
};

mdef plain(const char * name, fmt f, int32_t m) {
    return {name, f, m, epi::f32, -1, false};
}

struct group_def {
    std::string name; // grouping and layer
    int32_t k;
    std::vector<mdef> mats;
    bool route = false; // mats 0 and 1 are the router's logits and the shared gate, with the epilogue
    bool conv = false;  // mats 0 is the GDN input's qkv with the convolution epilogue (LP1)
    std::vector<int32_t> rows = {1, 3}; // group mode's row counts (eh_proj adds 4: a draft step's B_eff, DR1)
};

// one grouping's weights: n_copies of each matrix, launch c reads copy c % n_copies; outputs and work shared
struct group {
    group_def def;
    int32_t n_copies = 0;
    size_t weight_bytes = 0;
    std::vector<std::unique_ptr<dev_mem>> w, out;
    std::vector<size_t> bytes;
    std::unique_ptr<dev_mem> gc, gs, ids, wts, sg, work, taps, hist, cur;
    product_args base; // splits chosen; weights of copy 0
    std::string splits;
    bool across = false; // finishes across blocks: a split, a glu pair or the router

    product_args args(int32_t c, const int8_t * codes, const float * scales, const float * xf, const uint8_t * grids,
                      const float * scale) const {
        product_args a = base;
        for (int32_t i = 0; i < a.n_mats; ++i) {
            a.mats[i].w = w[size_t(i)]->p + size_t(c % n_copies) * bytes[size_t(i)];
            a.mats[i].scale = def.mats[size_t(i)].scaled ? scale : nullptr;
        }
        bool any_q = false, any_bf = false;
        for (const mdef & m : def.mats) {
            const bool wide = m.f == fmt::bf16 || m.f == fmt::f32; // F32 activations
            any_q = any_q || !wide;
            any_bf = any_bf || wide;
        }
        a.x_codes = any_q ? codes : nullptr;
        a.x_scales = any_q ? scales : nullptr;
        a.x_f32 = any_bf ? xf : nullptr;
        a.grids = grids;
        return a;
    }

    // the activation bytes a launch reads and the output bytes it writes, at rows rows
    size_t io_bytes(int32_t rows) const {
        size_t b = 0;
        bool any_q = false, any_bf = false;
        for (const mdef & m : def.mats) {
            const bool wide = m.f == fmt::bf16 || m.f == fmt::f32; // F32 activations
            any_q = any_q || !wide;
            any_bf = any_bf || wide;
            b += m.e == epi::f32 || m.e == epi::conv_silu ? size_t(rows) * m.m * 4 : 0;
            // the convolution epilogue: history version 0 and the taps read, versions 1..rows written
            b += m.e == epi::conv_silu
                     ? size_t(m.m) * (fn::gdn::n_hist + fn::gdn::n_tap) * 4 + size_t(rows) * m.m * fn::gdn::n_hist * 4
                     : 0;
            b += m.e == epi::glu_gate ? size_t(rows) * m.m + size_t(rows) * (m.m / 32) * 8 : 0;
            b += m.scaled ? size_t(rows) * 4 : 0;
        }
        b += any_q ? size_t(rows) * def.k + size_t(rows) * (def.k / 32) * 8 : 0;
        b += any_bf ? size_t(rows) * def.k * 4 : 0;
        b += def.route ? size_t(rows) * route_used * 8 + size_t(rows) * 4 : 0;
        return b;
    }
};

double floor_us(size_t bytes, bool across) {
    return ramp_us + double(bytes) / (dram_gbps * 1e3) + (across ? tail_across_us : tail_single_us);
}

struct bench {
    cudaStream_t st = nullptr;
    std::map<int64_t, synth> sets;
    dev_mem grids{grid_bytes};
    dev_mem xf{size_t(max_rows) * max_k * 4}; // layer input, F32
    dev_mem xc{size_t(max_rows) * max_k};     // and in FNFL's decode format
    dev_mem xs{size_t(max_rows) * (max_k / 32) * 8};
    dev_mem xc2{size_t(max_rows) * max_k}; // the stand-ins' outputs
    dev_mem xs2{size_t(max_rows) * (max_k / 32) * 8};
    dev_mem scale{size_t(max_rows) * 4}; // the shared down's scale in "all"
    dev_mem flush{size_t(flush_used)};
    dev_mem evict{16};
    dev_mem flush_ef{size_t(flush_used)};       // CR1: the evict-first flush's own buffer
    dev_mem xf_p{size_t(max_rows) * max_k * 4}; // CR1: the pre-touch's own activations (copies of xf, xc, xs, scale)
    dev_mem xc_p{size_t(max_rows) * max_k};
    dev_mem xs_p{size_t(max_rows) * (max_k / 32) * 8};
    dev_mem scale_p{size_t(max_rows) * 4};

    const synth * get(fmt f, int32_t k) {
        const int64_t key = int64_t(f) * 65536 + k;
        auto it = sets.find(key);
        if (it != sets.end()) {
            return &it->second;
        }
        synth s;
        if (!make_synth(f, k, s)) {
            fprintf(stderr, "dp-bench: no synthetic %s at K %d\n", fmt_name(f), k);
            return nullptr;
        }
        return &(sets[key] = std::move(s));
    }

    // copies: 0 for enough to rotate over min_cold bytes
    bool make(const group_def & d, int32_t copies, group & g) {
        g.def = d;
        g.base = product_args();
        g.base.k = d.k;
        g.base.n_mats = static_cast<int32_t>(d.mats.size());
        g.weight_bytes = 0;
        for (const mdef & m : d.mats) {
            g.weight_bytes += size_t(row_bytes(m.f, d.k)) * m.m;
        }
        g.n_copies = copies > 0 ? copies : static_cast<int32_t>((min_cold + g.weight_bytes - 1) / g.weight_bytes);
        int32_t glu_m = 0;
        for (size_t i = 0; i < d.mats.size(); ++i) {
            const mdef & md = d.mats[i];
            const synth * s = get(md.f, d.k);
            std::vector<uint8_t> packed;
            if (s == nullptr || !pack_synth(*s, md.m, int32_t(37 * i + 11) % n_synth_rows, n_synth_rows, packed)) {
                return false;
            }
            g.bytes.push_back(packed.size());
            g.w.push_back(std::make_unique<dev_mem>(packed.size() * size_t(g.n_copies)));
            g.out.push_back(std::make_unique<dev_mem>(size_t(max_rows) * md.m * 4));
            if (!g.w.back()->put(packed.data(), packed.size())) {
                return false;
            }
            for (int32_t c = 1; c < g.n_copies; ++c) {
                if (!cuda_ok(cudaMemcpy(g.w.back()->p + size_t(c) * packed.size(), g.w.back()->p, packed.size(),
                                        cudaMemcpyDeviceToDevice),
                             "copy weights")) {
                    return false;
                }
            }
            mat & m = g.base.mats[i];
            m.f = md.f;
            m.m = md.m;
            m.e = md.e;
            m.pair = md.pair;
            m.split = 0;
            m.w = g.w.back()->p;
            m.out = md.e == epi::f32 || md.e == epi::conv_silu ? g.out.back()->as<float>() : nullptr;
            glu_m = md.e == epi::glu_gate ? md.m : glu_m;
        }
        choose_splits(g.base);
        g.gc = std::make_unique<dev_mem>(size_t(max_rows) * std::max(glu_m, 32));
        g.gs = std::make_unique<dev_mem>(size_t(max_rows) * (std::max(glu_m, 32) / 32) * 8);
        g.ids = std::make_unique<dev_mem>(size_t(max_rows) * route_used * 4);
        g.wts = std::make_unique<dev_mem>(size_t(max_rows) * route_used * 4);
        g.sg = std::make_unique<dev_mem>(size_t(max_rows) * 4);
        if (glu_m > 0) {
            g.base.glu_codes = g.gc->as<int8_t>();
            g.base.glu_scales = g.gs->as<float>();
        }
        if (d.route) {
            g.base.route.logits = 0;
            g.base.route.gate = 1;
            g.base.route.ids = g.ids->as<int32_t>();
            g.base.route.wts = g.wts->as<float>();
            g.base.route.sg = g.sg->as<float>();
        }
        if (d.conv) { // zero taps and history: the epilogue's work and bytes do not depend on the values
            g.taps = std::make_unique<dev_mem>(size_t(fn::gdn::n_channel) * fn::gdn::n_tap * 4);
            g.hist = std::make_unique<dev_mem>(size_t(fn::gdn::n_version) * fn::gdn::hist_version_floats * 4);
            g.cur = std::make_unique<dev_mem>(4);
            if (!cuda_ok(cudaMemset(g.taps->p, 0, g.taps->n), "taps") ||
                !cuda_ok(cudaMemset(g.hist->p, 0, g.hist->n), "hist") || !cuda_ok(cudaMemset(g.cur->p, 0, 4), "cur")) {
                return false;
            }
            g.base.conv.w = g.taps->as<float>();
            g.base.conv.hist = g.hist->as<float>();
            g.base.conv.cur = g.cur->as<uint32_t>(); // slot 0 throughout (SR1)
        }
        g.work = std::make_unique<dev_mem>(work_bytes(g.base));
        g.base.work = g.work->p;
        g.splits.clear();
        g.across = glu_m > 0 || d.route;
        for (int32_t i = 0; i < g.base.n_mats; ++i) {
            g.splits += std::string(i ? "," : "") + std::to_string(g.base.mats[i].split);
            g.across = g.across || g.base.mats[i].split > 1;
        }
        return cuda_ok(cudaMemset(g.work->p, 0, g.work->n), "work");
    }

    bool launch(const group & g, int32_t rows, int32_t c, const int8_t * codes, const float * scales, const float * sc,
                const float * x = nullptr) {
        const product_args a = g.args(c, codes, scales, x != nullptr ? x : xf.as<float>(), grids.p, sc);
        if (!product(rows, a, st)) {
            fprintf(stderr, "dp-bench: %s not enqueued: %s\n", g.def.name.c_str(),
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        return true;
    }
    bool launch(const group & g, int32_t rows, int32_t c) {
        return launch(g, rows, c, xc.as<int8_t>(), xs.as<float>(), scale.as<float>());
    }

    // CR1's pre-touch: the unit's kernel on another grouping's weights, outputs and work, with its own activations
    bool launch_pre(const group & g, int32_t rows, int32_t c) {
        return launch(g, rows, c, xc_p.as<int8_t>(), xs_p.as<float>(), scale_p.as<float>(), xf_p.as<float>());
    }

    // the cold state before a cold launch (R17)
    bool cold() { return cold_flush(flush.p, flush_used, st) && cold_icache(evict.as<float>(), st); }

    // CR1: the cold state of R4 variant v (dense_cold.h)
    bool cold_r4(int32_t v, const std::function<bool()> & pre) {
        return fn::dense::cold_r4(v, {flush.p, flush_ef.p, flush_used, evict.as<float>()}, st, pre);
    }
};

struct stats {
    double median = 0.0, p90 = 0.0;
    std::vector<double> us;
    std::string t0, t1;

    void finish() {
        std::vector<double> s = us;
        std::sort(s.begin(), s.end());
        median = 0.5 * (s[n_timed / 2 - 1] + s[n_timed / 2]);
        p90 = s[(9 * n_timed + 9) / 10 - 1];
    }
};

bool capture(bench & bn, int32_t n, const std::function<bool(int32_t)> & unit, graph_exec & ge) {
    cudaGraph_t graph = nullptr;
    if (!cuda_ok(cudaStreamBeginCapture(bn.st, cudaStreamCaptureModeThreadLocal), "begin capture")) {
        return false;
    }
    bool ok = true;
    for (int32_t i = 0; i < n && ok; ++i) {
        ok = unit(i);
    }
    ok = cuda_ok(cudaStreamEndCapture(bn.st, &graph), "end capture") && ok;
    ok = ok && cuda_ok(cudaGraphInstantiate(&ge.x, graph, 0), "instantiate");
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    return ok;
}

bool replay_us(bench & bn, const graph_exec & ge, event_pair & ev, double & us) {
    float ms = 0.0f;
    const bool ok = cuda_ok(cudaEventRecord(ev.e0, bn.st), "record") &&
                    cuda_ok(cudaGraphLaunch(ge.x, bn.st), "replay") &&
                    cuda_ok(cudaEventRecord(ev.e1, bn.st), "record") && cuda_ok(cudaEventSynchronize(ev.e1), "sync") &&
                    cuda_ok(cudaEventElapsedTime(&ms, ev.e0, ev.e1), "elapsed");
    us = 1e3 * double(ms);
    return ok;
}

// Warm (DP1's method): units back to back in one graph, n_warm untimed replays and n_timed timed; a unit's time is a
// replay's event time over the units
bool time_warm(bench & bn, int32_t units, const std::function<bool(int32_t)> & unit, stats & s) {
    graph_exec ge;
    event_pair ev;
    bool ok = capture(bn, units, unit, ge) && ev.make();
    double us = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = replay_us(bn, ge, ev, us);
    }
    s.us.clear();
    s.t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        ok = replay_us(bn, ge, ev, us);
        s.us.push_back(us / units);
    }
    s.t1 = utc_now();
    if (ok) {
        s.finish();
    }
    return ok;
}

// Cold (R17, XP1's alone_timer): graph A = n_cold_rep x [flush, eviction, launch c], graph B = n_cold_rep x [flush,
// eviction]; per trial (A - B) / n_cold_rep, so a figure is the launch's time plus one node boundary with its weights,
// activations and code cold
bool time_cold(bench & bn, const std::function<bool(int32_t)> & unit, stats & s) {
    graph_exec ga, gb;
    event_pair ev;
    bool ok = capture(
                  bn, n_cold_rep, [&](int32_t c) { return bn.cold() && unit(c); }, ga) &&
              capture(bn, n_cold_rep, [&](int32_t) { return bn.cold(); }, gb) && ev.make();
    double ta = 0.0, tb = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = replay_us(bn, ga, ev, ta) && replay_us(bn, gb, ev, tb);
    }
    s.us.clear();
    s.t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        ok = replay_us(bn, gb, ev, tb) && replay_us(bn, ga, ev, ta);
        s.us.push_back((ta - tb) / n_cold_rep);
    }
    s.t1 = utc_now();
    if (ok) {
        s.finish();
    }
    return ok;
}

// CR1 (R4 on the bench): time_cold's (A - B) / n_cold_rep for each of the four cold states of dense_cold.h's cold_r4,
// graph A = n_cold_rep x [state v, unit c], B = n_cold_rep x [state v]; pre(c) is unit c's pre-touch. A trial replays
// B then A of V0, V1, V2 and V3 in turn, so the four see the same clock drift.
bool time_r4(bench & bn, const std::function<bool(int32_t)> & unit, const std::function<bool(int32_t)> & pre,
             stats s[4]) {
    graph_exec ga[4], gb[4];
    event_pair ev;
    bool ok = ev.make();
    for (int32_t v = 0; v < 4 && ok; ++v) {
        ok =
            capture(
                bn, n_cold_rep, [&](int32_t c) { return bn.cold_r4(v, [&]() { return pre(c); }) && unit(c); }, ga[v]) &&
            capture(bn, n_cold_rep, [&](int32_t c) { return bn.cold_r4(v, [&]() { return pre(c); }); }, gb[v]);
    }
    double ta = 0.0, tb = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        for (int32_t v = 0; v < 4 && ok; ++v) {
            ok = replay_us(bn, gb[v], ev, tb) && replay_us(bn, ga[v], ev, ta);
        }
    }
    for (int32_t v = 0; v < 4; ++v) {
        s[v].us.clear();
        s[v].t0 = utc_now();
    }
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        for (int32_t v = 0; v < 4 && ok; ++v) {
            ok = replay_us(bn, gb[v], ev, tb) && replay_us(bn, ga[v], ev, ta);
            s[v].us.push_back((ta - tb) / n_cold_rep);
        }
    }
    for (int32_t v = 0; v < 4 && ok; ++v) {
        s[v].t1 = utc_now();
        s[v].finish();
    }
    return ok;
}

double p10_of(const stats & s) {
    std::vector<double> v = s.us;
    std::sort(v.begin(), v.end());
    return v[size_t(n_timed / 10)];
}

std::string formats_of(const group_def & d) {
    std::string s;
    for (size_t i = 0; i < d.mats.size(); ++i) {
        s += std::string(i ? " " : "") + d.mats[i].name + "=" + fmt_name(d.mats[i].f) + "[" + std::to_string(d.k) +
             "->" + std::to_string(d.mats[i].m) + "]" + (d.mats[i].scaled ? "*sg" : "");
    }
    return s + (d.route ? " +route" : "");
}

struct point_info {
    std::string point, what, formats, splits;
    int32_t rows = 0;
    int32_t units = n_unit; // launches (warm) a replay
    int32_t launches_per_unit = 1;
    double weight_bytes = 0.0; // a unit's
    double io_bytes = 0.0;     // activation and output bytes a unit
    double floor = 0.0;        // the sum of the unit's launches' floors
    int32_t n_copies = 0;
};

void print_us(const char * key, const std::vector<double> & us) {
    printf(", \"%s\": [", key);
    for (size_t i = 0; i < us.size(); ++i) {
        printf("%s%.3f", i == 0 ? "" : ", ", us[i]);
    }
    printf("]");
}

void print_point(const point_info & pi, const stats & warm, const stats * cold) {
    printf("{\"ticket\": \"DP2\", \"point\": \"%s\", \"what\": \"%s\", \"formats\": \"%s\", \"splits\": \"%s\", "
           "\"rows\": %d, \"median_us\": %.3f, \"p90_us\": %.3f, \"n\": %d, \"units_per_replay\": %d, "
           "\"launches_per_unit\": %d, \"warm_replays\": %d, \"weight_bytes_per_unit\": %.0f, "
           "\"io_bytes_per_unit\": %.0f, \"floor_us\": %.3f, \"hw_pct_warm\": %.1f, \"weight_copies\": %d, "
           "\"warm\": \"weights rotated over >= 200 MB, activations and code warm\", \"t_start\": \"%s\", "
           "\"t_end\": \"%s\"",
           pi.point.c_str(), pi.what.c_str(), pi.formats.c_str(), pi.splits.c_str(), pi.rows, warm.median, warm.p90,
           n_timed, pi.units, pi.launches_per_unit, n_warm, pi.weight_bytes, pi.io_bytes, pi.floor,
           pi.floor > 0.0 ? 100.0 * pi.floor / warm.median : 0.0, pi.n_copies, warm.t0.c_str(), warm.t1.c_str());
    if (cold != nullptr) {
        printf(", \"cold_median_us\": %.3f, \"cold_p90_us\": %.3f, \"hw_pct_cold\": %.1f, "
               "\"cold\": \"(A - B) / %d: 12 MB L2 flush and instruction-cache eviction before each launch\", "
               "\"cold_t_start\": \"%s\", \"cold_t_end\": \"%s\"",
               cold->median, cold->p90, pi.floor > 0.0 ? 100.0 * pi.floor / cold->median : 0.0, n_cold_rep,
               cold->t0.c_str(), cold->t1.c_str());
    }
    print_us("us", warm.us);
    if (cold != nullptr) {
        print_us("cold_us", cold->us);
    }
    printf("}\n");
    fflush(stdout);
}

std::string point_name(const std::string & name, int32_t rows) {
    std::string p = name + "_B" + std::to_string(rows);
    for (char & c : p) {
        c = (c == ' ' || c == '/' || c == '+') ? '_' : c;
    }
    return p;
}

point_info info_of(const group & g, int32_t rows) {
    point_info pi;
    pi.point = point_name(g.def.name, rows);
    pi.what = g.def.name + " alone";
    pi.formats = formats_of(g.def);
    pi.splits = g.splits;
    pi.rows = rows;
    pi.weight_bytes = double(g.weight_bytes);
    pi.io_bytes = double(g.io_bytes(rows));
    pi.floor = floor_us(g.weight_bytes + g.io_bytes(rows), g.across);
    pi.n_copies = g.n_copies;
    return pi;
}

// the per-card groupings at deployed layers' formats (fnfl/sim/data/deployed-tensor-inventory.json)
group_def gdn(const char * layer, fmt qkv, fmt gate) {
    return {std::string("GDN input L") + layer,
            D,
            {plain("attn_qkv", qkv, 5120), plain("attn_gate", gate, 3072), plain("alpha", fmt::bf16, 24),
             plain("beta", fmt::bf16, 24)}};
}
// the GDN input with the convolution epilogue on qkv (LP1)
group_def gdn_conv(const char * layer, fmt qkv, fmt gate) {
    return {std::string("GDN input conv L") + layer,
            D,
            {{"attn_qkv", qkv, fn::gdn::n_channel, epi::conv_silu, -1, false},
             plain("attn_gate", gate, 3072),
             plain("alpha", fmt::bf16, 24),
             plain("beta", fmt::bf16, 24)},
            false,
            true};
}
group_def qsa(const char * layer, fmt q, fmt k, fmt v) {
    return {std::string("QSA input L") + layer,
            D,
            {plain("attn_q", q, 6144), plain("attn_k", k, 256), plain("attn_v", v, 256), plain("idx_q", fmt::bf16, 512),
             plain("idx_k", fmt::bf16, 128)}};
}
group_def out_def(const char * name, const char * layer, fmt f) {
    return {std::string(name) + " L" + layer, 3072, {plain(name, f, 2560)}};
}
group_def shexp(const char * layer, fmt gate, fmt up) {
    return {std::string("shexp gate/up L") + layer,
            D,
            {{"gate", gate, 320, epi::glu_gate, 1, false}, {"up", up, 320, epi::glu_up, 0, false}}};
}
group_def down_def(const char * layer, fmt f) {
    return {std::string("shexp down L") + layer, 320, {{"down", f, 2560, epi::f32, -1, true}}};
}
group_def router_def() {
    return {"router + shared gate", D, {plain("router", fmt::bf16, 512), plain("shexp_gate", fmt::bf16, 1)}, true};
}
// the router's products without the epilogue (DP2 Amendment 3: the epilogue's share is the difference); group mode only
group_def router_products_def() {
    return {"router + shared gate products", D, {plain("router", fmt::bf16, 512), plain("shexp_gate", fmt::bf16, 1)}};
}
group_def vocab_def() {
    return {"vocabulary", D, {plain("vocab", fmt::q5_k, 124160)}};
}
// the draft block's (DR1 Stage B): eh_proj [5120 -> 1280] at B_eff = 4 B stream rows, the F32 router with its
// epilogue, and the draft head's vocabulary subset (32,768 Q5_K rows a card, head.h)
group_def eh_proj_def() {
    return {"eh_proj L48", 5120, {plain("eh_proj", fmt::q8_0, 1280)}, false, false, {1, 3, 4}};
}
group_def router_f32_def() {
    return {
        "F32 router + shared gate L48", D, {plain("router", fmt::f32, 512), plain("shexp_gate", fmt::f32, 1)}, true};
}
group_def draft_vocab_def() {
    return {"draft vocabulary", D, {plain("vocab", fmt::q5_k, 32768)}};
}

std::vector<group_def> all_defs() {
    return {
        gdn("0", fmt::iq4_xs, fmt::iq3_s),
        gdn("10", fmt::q6_k, fmt::q6_k),
        gdn_conv("10", fmt::q6_k, fmt::q6_k),
        gdn("14", fmt::q4_k, fmt::q4_k),
        gdn("46", fmt::q5_k, fmt::q5_k),
        qsa("11", fmt::q4_k, fmt::q4_k, fmt::q4_k),
        qsa("27", fmt::q6_k, fmt::q6_k, fmt::q6_k),
        qsa("47", fmt::iq3_s, fmt::iq3_s, fmt::q6_k),
        qsa("48", fmt::q8_0, fmt::q8_0, fmt::q8_0),
        out_def("ssm_out", "25", fmt::q6_k),
        out_def("ssm_out", "10", fmt::q5_k),
        out_def("ssm_out", "1", fmt::iq3_s),
        out_def("ssm_out", "0", fmt::iq4_xs),
        out_def("ssm_out", "2", fmt::q4_k),
        out_def("attn_output", "11", fmt::q6_k),
        out_def("attn_output", "7", fmt::q5_k),
        out_def("attn_output", "48", fmt::q8_0),
        shexp("0", fmt::iq4_xs, fmt::iq3_s),
        shexp("11", fmt::q4_k, fmt::q4_k),
        shexp("48", fmt::q8_0, fmt::q8_0),
        down_def("0", fmt::iq4_nl),
        down_def("3", fmt::q2_0),
        down_def("48", fmt::q8_0),
        router_def(),
        vocab_def(),
        eh_proj_def(),
        router_f32_def(),
        draft_vocab_def(),
    };
}

// "all", one point warm only (only: its name, for ncu), or one grouping's points warm and cold (group: its points'
// name without _B<rows> in set, for DP2 Amendment 3's router timing)
bool run_all(bench & bn, const char * only, const char * set = nullptr) {
    bool found = only == nullptr && set == nullptr;
    std::vector<group_def> defs = all_defs();
    if (set != nullptr) {
        defs.push_back(router_products_def());
    }
    for (const group_def & d : defs) {
        const bool any = set != nullptr
                             ? point_name(d.name, 1) == std::string(set) + "_B1"
                             : only == nullptr || point_name(d.name, 1) == only || point_name(d.name, 3) == only;
        if (!any) {
            continue;
        }
        group g;
        if (!bn.make(d, 0, g)) {
            return false;
        }
        for (const int32_t rows : d.rows) {
            const point_info pi = info_of(g, rows);
            if (only != nullptr && pi.point != only) {
                continue;
            }
            found = true;
            const auto unit = [&](int32_t c) { return bn.launch(g, rows, c); };
            stats warm, cold;
            if (!time_warm(bn, n_unit, unit, warm) || (only == nullptr && !time_cold(bn, unit, cold))) {
                fprintf(stderr, "dp-bench: %s failed\n", pi.point.c_str());
                return false;
            }
            print_point(pi, warm, only == nullptr ? &cold : nullptr);
        }
    }
    if (!found) {
        fprintf(stderr, "dp-bench: no point %s\n", only != nullptr ? only : set);
    }
    return found;
}

// Step 7: the layer's order, with C2's quantize or a stand-in between the launches (the stand-ins quantize the
// previous launch's F32 output as [B][3072]). Each repeat reads copy rep % 5 of every grouping (5 x the repeat's
// weights > 200 MB).
bool run_seq(bench & bn, bool alone_too) {
    struct seq_groups {
        group gdn, ssm, qsa, att, rt, sh, dn;
    } sg;
    constexpr int32_t copies = 5;
    bool ok = bn.make(gdn("10", fmt::q6_k, fmt::q6_k), copies, sg.gdn) &&
              bn.make(out_def("ssm_out", "10", fmt::q5_k), copies, sg.ssm) &&
              bn.make(qsa("11", fmt::q4_k, fmt::q4_k, fmt::q4_k), copies, sg.qsa) &&
              bn.make(out_def("attn_output", "11", fmt::q6_k), copies, sg.att) &&
              bn.make(router_def(), copies, sg.rt) && bn.make(shexp("11", fmt::q4_k, fmt::q4_k), copies, sg.sh) &&
              bn.make(down_def("11", fmt::iq4_nl), copies, sg.dn);
    if (!ok) {
        return false;
    }
    const group * gs[] = {&sg.gdn, &sg.ssm, &sg.qsa, &sg.att, &sg.rt, &sg.sh, &sg.dn};
    size_t rep_bytes = 0;
    for (const group * g : gs) {
        rep_bytes += g->weight_bytes;
    }
    printf("# dp-bench: seq: %zu weight bytes a repeat, %d copies (%.1f MB rotated)\n", rep_bytes, copies,
           double(rep_bytes) * copies / 1e6);
    const float * xf = bn.xf.as<float>();
    const float * qkv = sg.gdn.out[0]->as<float>();
    const float * q_out = sg.qsa.out[0]->as<float>();
    int8_t * c1 = bn.xc.as<int8_t>(); // quantize -> GDN input, QSA input, router, shexp gate/up
    float * s1 = bn.xs.as<float>();
    int8_t * c2 = bn.xc2.as<int8_t>(); // recurrence stand-in -> ssm_out; attention stand-in -> attn_output
    float * s2 = bn.xs2.as<float>();
    for (const int32_t rows : {1, 3}) {
        const fn::act::out_view in{c1, s1, rows, D}, mid{c2, s2, rows, 3072};
        const auto q = [&](const float * x, const fn::act::out_view & o) { return fn::act::quantize(x, o, bn.st); };
        const float * gate = sg.rt.sg->as<float>();
        // the twelve launches of one repeat, in order
        const auto rep = [&](int32_t c) {
            return q(xf, in) && bn.launch(sg.gdn, rows, c, c1, s1, nullptr) && q(qkv, mid) &&
                   bn.launch(sg.ssm, rows, c, c2, s2, nullptr) && q(xf, in) &&
                   bn.launch(sg.qsa, rows, c, c1, s1, nullptr) && q(q_out, mid) &&
                   bn.launch(sg.att, rows, c, c2, s2, nullptr) && q(xf, in) &&
                   bn.launch(sg.rt, rows, c, c1, s1, nullptr) && bn.launch(sg.sh, rows, c, c1, s1, nullptr) &&
                   bn.launch(sg.dn, rows, c, sg.sh.gc->as<int8_t>(), sg.sh.gs->as<float>(), gate);
        };
        point_info pi;
        pi.point = "seq_B" + std::to_string(rows);
        pi.what = "quantize, GDN input L10, recurrence stand-in, ssm_out L10, quantize, QSA input L11, attention "
                  "stand-in, attn_output L11, quantize, router + shared gate, shexp gate/up L11, shexp down L11";
        pi.rows = rows;
        pi.units = n_seq_rep;
        pi.launches_per_unit = 12;
        pi.n_copies = copies;
        for (const group * g : gs) {
            pi.formats += std::string(pi.formats.empty() ? "" : " | ") + formats_of(g->def);
            pi.splits += std::string(pi.splits.empty() ? "" : " | ") + g->splits;
            pi.weight_bytes += double(g->weight_bytes);
            pi.io_bytes += double(g->io_bytes(rows));
            pi.floor += floor_us(g->weight_bytes + g->io_bytes(rows), g->across);
        }
        printf("# dp-bench: seq_B%d graph order per repeat: quantize_kernel, product_kernel<%d> (GDN), quantize_kernel "
               "(recurrence stand-in), product_kernel<%d> (ssm_out), quantize_kernel, product_kernel<%d> (QSA), "
               "quantize_kernel (attention stand-in), product_kernel<%d> (attn_output), quantize_kernel, "
               "product_kernel<%d> (router), product_kernel<%d> (shexp gate/up), product_kernel<%d> (shexp down); "
               "floor_us is the DP2 launches' floors only\n",
               rows, rows, rows, rows, rows, rows, rows, rows);
        stats warm;
        if (!time_warm(bn, n_seq_rep, rep, warm)) {
            return false;
        }
        print_point(pi, warm, nullptr);
        if (!alone_too) {
            continue;
        }
        // the quantize and stand-in launches alone (DP2's launches alone are the "all" points)
        const auto alone = [&](const char * name, const char * what, const std::function<bool()> & f) {
            point_info p;
            p.point = std::string(name) + "_B" + std::to_string(rows);
            p.what = what;
            p.rows = rows;
            stats w, cd;
            const auto unit = [&](int32_t) { return f(); };
            if (!time_warm(bn, n_unit, unit, w) || !time_cold(bn, unit, cd)) {
                return false;
            }
            print_point(p, w, &cd);
            return true;
        };
        ok = alone("quantize_k2560", "C2 quantize F32 [B][2560] alone", [&]() { return q(xf, in); }) &&
             alone("standin_recurrence", "recurrence stand-in: C2 quantize F32 [B][3072] alone",
                   [&]() { return q(qkv, mid); }) &&
             alone("standin_attention", "attention stand-in: C2 quantize F32 [B][3072] alone",
                   [&]() { return q(q_out, mid); });
        if (!ok) {
            return false;
        }
    }
    return true;
}

// CR1 step 3: each named point (a comma list) warm, then V0..V3 (time_r4). The pre-touch is the same kernel on a
// second grouping of its own (weights, outputs, work, router outputs) and its own activations, so it shares only the
// kernel's code, its parameters' layout and the grid tables with the unit. Unit c reads weight copy c of the first
// grouping.
bool run_r4(bench & bn, const char * points) {
    const std::string sel = points;
    int32_t found = 0;
    for (const group_def & d : all_defs()) {
        for (const int32_t rows : d.rows) {
            const std::string pt = point_name(d.name, rows);
            const std::string padded = "," + sel + ",";
            if (padded.find("," + pt + ",") == std::string::npos) {
                continue;
            }
            group g, gp;
            if (!bn.make(d, 0, g) || !bn.make(d, 0, gp) || !cuda_ok(cudaDeviceSynchronize(), "groups")) {
                return false;
            }
            const point_info pi = info_of(g, rows);
            const auto unit = [&](int32_t c) { return bn.launch(g, rows, c); };
            const auto pre = [&](int32_t c) { return bn.launch_pre(gp, rows, c); };
            stats warm, v[4];
            if (!time_warm(bn, n_unit, unit, warm) || !time_r4(bn, unit, pre, v)) {
                fprintf(stderr, "dp-bench: %s failed\n", pt.c_str());
                return false;
            }
            printf("{\"ticket\": \"CR1\", \"point\": \"%s\", \"formats\": \"%s\", \"splits\": \"%s\", \"rows\": %d, "
                   "\"weight_bytes_per_unit\": %.0f, \"io_bytes_per_unit\": %.0f, \"weight_copies\": %d, "
                   "\"warm_median_us\": %.3f, \"warm_p10_us\": %.3f, \"warm_p90_us\": %.3f",
                   pt.c_str(), pi.formats.c_str(), pi.splits.c_str(), rows, pi.weight_bytes, pi.io_bytes, pi.n_copies,
                   warm.median, p10_of(warm), warm.p90);
            for (int32_t k = 0; k < 4; ++k) {
                printf(", \"V%d_median_us\": %.3f, \"V%d_p10_us\": %.3f, \"V%d_p90_us\": %.3f", k, v[k].median, k,
                       p10_of(v[k]), k, v[k].p90);
            }
            printf(", \"cold\": \"(A - B) / %d; V0: flush, evict | V1: flush, evict, pre-touch, ef flush, evict | V2: "
                   "flush, evict, ef flush, evict | V3: flush, evict, pre-touch, flush, evict; flushes 12 MB\", "
                   "\"warm\": \"graph of %d launches, weights rotated over >= 200 MB, activations and code warm\", "
                   "\"t_start\": \"%s\", \"t_end\": \"%s\"",
                   n_cold_rep, n_unit, warm.t0.c_str(), v[3].t1.c_str());
            print_us("warm_us", warm.us);
            for (int32_t k = 0; k < 4; ++k) {
                const std::string key = "V" + std::to_string(k) + "_us";
                print_us(key.c_str(), v[k].us);
            }
            printf("}\n");
            fflush(stdout);
            ++found;
        }
    }
    if (found == 0) {
        fprintf(stderr, "dp-bench: no point in %s\n", points);
    }
    return found > 0;
}

// CR1 step 2 (l2keep): does L2 keep normal-priority lines against an evict-first stream? A trial of an arm: reset (the
// 12 MB normal flush), fill (the 1 MB keep buffer read with normal loads), the arm's stream, then the walk times a load
// of every 32-byte sector of the keep buffer (l2keep_walk). Arms: warm (no stream), cold (the 12 MB normal flush), ef3
// and ef12 (3 or 12 MB read evict-first), and ef12+N (12 MB evict-first interleaved with N MB of normal reads, N =
// 0.5, 1, 2, 4). The arms take turns within a trial.
bool run_l2keep(bench & bn) {
    constexpr int64_t mb = 1ll << 20;
    constexpr int64_t keep_bytes = mb;
    constexpr int32_t n_walks = 64;
    constexpr int32_t n_sectors = int32_t(keep_bytes / 32);
    constexpr int32_t n_steps = n_sectors / n_walks;
    constexpr int32_t n_trials = 20;
    struct arm {
        const char * name;
        int64_t ef, normal; // bytes; ef < 0: the normal flush instead
    };
    const arm arms[] = {{"warm", 0, 0},
                        {"cold", -1, 0},
                        {"ef3", 3 * mb, 0},
                        {"ef12", 12 * mb, 0},
                        {"ef12+n0.5", 12 * mb, mb / 2},
                        {"ef12+n1", 12 * mb, mb},
                        {"ef12+n2", 12 * mb, 2 * mb},
                        {"ef12+n4", 12 * mb, 4 * mb}};
    constexpr int32_t n_arms = int32_t(sizeof(arms) / sizeof(arms[0]));
    dev_mem keep{size_t(keep_bytes)}, normal{size_t(4 * mb)}, cyc{size_t(n_sectors) * 4}, wcyc{size_t(n_walks) * 4};
    // sector s's word 0: the next sector of its walk (s + n_walks); the other words a pattern
    std::vector<uint32_t> h(size_t(keep_bytes / 4));
    for (size_t i = 0; i < h.size(); ++i) {
        h[i] = i % 8 == 0 ? uint32_t(i / 8 + n_walks) % uint32_t(n_sectors) : uint32_t(i * 2654435761u);
    }
    bool ok = keep.put(h.data(), h.size() * 4) && cuda_ok(cudaMemsetAsync(normal.p, 0x3c, normal.n, bn.st), "normal") &&
              cuda_ok(cudaMemsetAsync(bn.flush_ef.p, 0x69, bn.flush_ef.n, bn.st), "flush_ef") &&
              cuda_ok(cudaStreamSynchronize(bn.st), "l2keep setup");
    // per arm and trial: the walk's mean cycles a load, and every load's cycles (for the threshold)
    std::vector<std::vector<double>> mean(n_arms), walk(n_arms);
    std::vector<std::vector<std::vector<uint32_t>>> loads(n_arms);
    std::vector<uint32_t> hc(static_cast<size_t>(n_sectors)), hw(static_cast<size_t>(n_walks));
    const std::string t0 = utc_now();
    for (int32_t t = 0; t < n_trials + 1 && ok; ++t) { // trial 0: untimed warm-up
        for (int32_t a = 0; a < n_arms && ok; ++a) {
            const arm & r = arms[a];
            ok = cold_flush(bn.flush.p, flush_used, bn.st) && cold_flush(keep.p, keep_bytes, bn.st);
            if (r.ef < 0) {
                ok = ok && cold_flush(bn.flush.p, flush_used, bn.st);
            } else if (r.ef > 0) {
                ok = ok && l2keep_stream(bn.flush_ef.p, r.ef, normal.p, r.normal, bn.st);
            }
            ok = ok &&
                 l2keep_walk(keep.as<uint32_t>(), n_walks, n_steps, cyc.as<uint32_t>(), wcyc.as<uint32_t>(), bn.st) &&
                 cuda_ok(cudaMemcpyAsync(hc.data(), cyc.p, cyc.n, cudaMemcpyDeviceToHost, bn.st), "cycles") &&
                 cuda_ok(cudaMemcpyAsync(hw.data(), wcyc.p, wcyc.n, cudaMemcpyDeviceToHost, bn.st), "walk") &&
                 cuda_ok(cudaStreamSynchronize(bn.st), "l2keep trial");
            if (ok && t > 0) {
                double sum = 0.0, wsum = 0.0;
                for (const uint32_t c : hc) {
                    sum += c;
                }
                for (const uint32_t c : hw) {
                    wsum += c;
                }
                mean[size_t(a)].push_back(sum / n_sectors);
                walk[size_t(a)].push_back(wsum / n_walks / n_steps);
                loads[size_t(a)].push_back(hc);
            }
        }
    }
    if (!ok) {
        return false;
    }
    const auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    };
    const auto load_q = [&](int32_t a, double q) { // quantile q of every load of arm a
        std::vector<uint32_t> all;
        for (const auto & l : loads[size_t(a)]) {
            all.insert(all.end(), l.begin(), l.end());
        }
        std::sort(all.begin(), all.end());
        return double(all[std::min(all.size() - 1, size_t(q * double(all.size())))]);
    };
    // a load is an L2 hit below the midpoint of the warm and cold arms' median loads
    const double hit_med = load_q(0, 0.5), miss_med = load_q(1, 0.5), thr = 0.5 * (hit_med + miss_med);
    const double m_warm = med(mean[0]), m_cold = med(mean[1]);
    printf("# dp-bench l2keep: keep %lld B (%d sectors of 32 B), %d walks of %d loads, %d trials an arm (after 1 "
           "untimed), threshold %.1f cycles (warm median load %.0f, cold %.0f), %s to %s\n",
           static_cast<long long>(keep_bytes), n_sectors, n_walks, n_steps, n_trials, thr, hit_med, miss_med,
           t0.c_str(), utc_now().c_str());
    for (int32_t a = 0; a < n_arms; ++a) {
        std::vector<double> hits;
        for (const auto & l : loads[size_t(a)]) {
            int64_t n = 0;
            for (const uint32_t c : l) {
                n += double(c) < thr ? 1 : 0;
            }
            hits.push_back(double(n) / n_sectors);
        }
        std::vector<double> hs = hits;
        std::sort(hs.begin(), hs.end());
        const double m = med(mean[size_t(a)]);
        printf("{\"ticket\": \"CR1\", \"probe\": \"l2keep\", \"arm\": \"%s\", \"ef_bytes\": %lld, "
               "\"normal_bytes\": %lld, \"normal_flush\": %s, \"trials\": %d, \"mean_load_cycles_median\": %.1f, "
               "\"walk_cycles_per_load_median\": %.1f, \"load_cycles_p10\": %.0f, \"load_cycles_p50\": %.0f, "
               "\"load_cycles_p90\": %.0f, \"survive_by_time\": %.4f, \"survive_by_class_median\": %.4f, "
               "\"survive_by_class_min\": %.4f, \"survive_by_class_max\": %.4f",
               arms[a].name, static_cast<long long>(std::max<int64_t>(arms[a].ef, 0)),
               static_cast<long long>(arms[a].normal), arms[a].ef < 0 ? "true" : "false", n_trials, m,
               med(walk[size_t(a)]), load_q(a, 0.1), load_q(a, 0.5), load_q(a, 0.9), (m_cold - m) / (m_cold - m_warm),
               med(hits), hs.front(), hs.back());
        print_us("mean_load_cycles", mean[size_t(a)]);
        print_us("survive_by_class", hits);
        printf("}\n");
    }
    fflush(stdout);
    return true;
}

// CR3 step 4 (l2keep:cr3): CR1's probe at the retention limit. For each keep set (CR3: 0.5, 0.75 and 1 MB; XB1's probe
// 5, l2keep:xb1: 0.1 MB, 102,400 B, the experts' own code size) the arms warm
// (no stream), cold (the 12 MB normal flush) and ef12+N (12 MB evict-first with N MB of normal reads interleaved, N =
// 3.5, 4.5, 5, 5.5 and 6) take turns within a trial; a trial of an arm is CR1's: reset (12 MB normal flush), fill (the
// keep set read with normal loads), the arm's stream, the walk over every 32-byte sector of the keep set. The hit
// threshold is the midpoint of the keep set's own warm and cold median loads.
bool run_l2keep_cr3(bench & bn, const std::vector<int64_t> & keeps, const char * tag) {
    constexpr int64_t mb = 1ll << 20;
    constexpr int32_t n_walks = 64;
    constexpr int32_t n_trials = 20;
    struct arm {
        std::string name;
        int64_t ef, normal; // bytes; ef < 0: the normal flush instead
    };
    std::vector<arm> arms = {{"warm", 0, 0}, {"cold", -1, 0}};
    for (const int64_t n2 : {7, 9, 10, 11, 12}) { // N in half MB
        const std::string n = std::to_string(n2 / 2) + (n2 % 2 ? ".5" : "");
        arms.push_back({"ef12+n" + n, 12 * mb, n2 * mb / 2});
    }
    const int32_t n_arms = static_cast<int32_t>(arms.size());
    dev_mem keep{size_t(mb)}, normal{size_t(6 * mb)}, cyc{size_t(mb / 32) * 4}, wcyc{size_t(n_walks) * 4};
    bool ok = cuda_ok(cudaMemsetAsync(normal.p, 0x3c, normal.n, bn.st), "normal") &&
              cuda_ok(cudaMemsetAsync(bn.flush_ef.p, 0x69, bn.flush_ef.n, bn.st), "flush_ef") &&
              cuda_ok(cudaStreamSynchronize(bn.st), "l2keep setup");
    const std::string t0 = utc_now();
    for (const int64_t keep_bytes : keeps) {
        const int32_t n_sectors = int32_t(keep_bytes / 32), n_steps = n_sectors / n_walks;
        // sector s's word 0: the next sector of its walk (s + n_walks); the other words a pattern
        std::vector<uint32_t> h(size_t(keep_bytes / 4));
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] = i % 8 == 0 ? uint32_t(i / 8 + n_walks) % uint32_t(n_sectors) : uint32_t(i * 2654435761u);
        }
        ok = ok && keep.put(h.data(), h.size() * 4);
        std::vector<std::vector<double>> mean(static_cast<size_t>(n_arms));
        std::vector<std::vector<std::vector<uint32_t>>> loads(static_cast<size_t>(n_arms));
        std::vector<uint32_t> hc(static_cast<size_t>(n_sectors)), hw(static_cast<size_t>(n_walks));
        for (int32_t t = 0; t < n_trials + 1 && ok; ++t) { // trial 0: untimed warm-up
            for (int32_t a = 0; a < n_arms && ok; ++a) {
                const arm & r = arms[size_t(a)];
                ok = cold_flush(bn.flush.p, flush_used, bn.st) && cold_flush(keep.p, keep_bytes, bn.st);
                if (r.ef < 0) {
                    ok = ok && cold_flush(bn.flush.p, flush_used, bn.st);
                } else if (r.ef > 0) {
                    ok = ok && l2keep_stream(bn.flush_ef.p, r.ef, normal.p, r.normal, bn.st);
                }
                ok = ok &&
                     l2keep_walk(keep.as<uint32_t>(), n_walks, n_steps, cyc.as<uint32_t>(), wcyc.as<uint32_t>(),
                                 bn.st) &&
                     cuda_ok(cudaMemcpyAsync(hc.data(), cyc.p, hc.size() * 4, cudaMemcpyDeviceToHost, bn.st),
                             "cycles") &&
                     cuda_ok(cudaMemcpyAsync(hw.data(), wcyc.p, wcyc.n, cudaMemcpyDeviceToHost, bn.st), "walk") &&
                     cuda_ok(cudaStreamSynchronize(bn.st), "l2keep trial");
                if (ok && t > 0) {
                    double sum = 0.0;
                    for (const uint32_t c : hc) {
                        sum += c;
                    }
                    mean[size_t(a)].push_back(sum / n_sectors);
                    loads[size_t(a)].push_back(hc);
                }
            }
        }
        if (!ok) {
            return false;
        }
        const auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
        };
        const auto load_q = [&](int32_t a, double q) {
            std::vector<uint32_t> all;
            for (const auto & l : loads[size_t(a)]) {
                all.insert(all.end(), l.begin(), l.end());
            }
            std::sort(all.begin(), all.end());
            return double(all[std::min(all.size() - 1, size_t(q * double(all.size())))]);
        };
        const double hit_med = load_q(0, 0.5), miss_med = load_q(1, 0.5), thr = 0.5 * (hit_med + miss_med);
        const double m_warm = med(mean[0]), m_cold = med(mean[1]);
        printf("# dp-bench %s: keep %lld B (%d sectors of 32 B), %d walks of %d loads, %d trials an arm (after "
               "1 untimed), threshold %.1f cycles (warm median load %.0f, cold %.0f)\n",
               tag, static_cast<long long>(keep_bytes), n_sectors, n_walks, n_steps, n_trials, thr, hit_med, miss_med);
        for (int32_t a = 0; a < n_arms; ++a) {
            std::vector<double> hits;
            for (const auto & l : loads[size_t(a)]) {
                int64_t n = 0;
                for (const uint32_t c : l) {
                    n += double(c) < thr ? 1 : 0;
                }
                hits.push_back(double(n) / n_sectors);
            }
            std::vector<double> hs = hits;
            std::sort(hs.begin(), hs.end());
            const double m = med(mean[size_t(a)]);
            printf("{\"ticket\": \"%s\", \"probe\": \"l2keep\", \"keep_bytes\": %lld, \"arm\": \"%s\", "
                   "\"ef_bytes\": %lld, \"normal_bytes\": %lld, \"normal_flush\": %s, \"trials\": %d, "
                   "\"threshold_cycles\": %.1f, \"mean_load_cycles_median\": %.1f, \"load_cycles_p10\": %.0f, "
                   "\"load_cycles_p50\": %.0f, \"load_cycles_p90\": %.0f, \"survive_by_time\": %.4f, "
                   "\"survive_by_class_median\": %.4f, \"survive_by_class_min\": %.4f, "
                   "\"survive_by_class_max\": %.4f",
                   tag, static_cast<long long>(keep_bytes), arms[size_t(a)].name.c_str(),
                   static_cast<long long>(std::max<int64_t>(arms[size_t(a)].ef, 0)),
                   static_cast<long long>(arms[size_t(a)].normal), arms[size_t(a)].ef < 0 ? "true" : "false", n_trials,
                   thr, m, load_q(a, 0.1), load_q(a, 0.5), load_q(a, 0.9), (m_cold - m) / (m_cold - m_warm), med(hits),
                   hs.front(), hs.back());
            print_us("mean_load_cycles", mean[size_t(a)]);
            print_us("survive_by_class", hits);
            printf("}\n");
        }
        fflush(stdout);
    }
    printf("# dp-bench %s: %s to %s\n", tag, t0.c_str(), utc_now().c_str());
    return ok;
}

} // namespace

int fnfl_dp_bench(int device, const char * mode) {
    const bool ncu = strncmp(mode, "ncu:", 4) == 0;
    const bool one_group = strncmp(mode, "group:", 6) == 0;
    const bool r4 = strncmp(mode, "r4:", 3) == 0, l2keep = strcmp(mode, "l2keep") == 0; // CR1
    const bool l2keep_cr3 = strcmp(mode, "l2keep:cr3") == 0;                            // CR3
    const bool l2keep_xb1 = strcmp(mode, "l2keep:xb1") == 0;                            // XB1
    if (strcmp(mode, "all") != 0 && strcmp(mode, "seq") != 0 && strcmp(mode, "nsys") != 0 && !ncu && !one_group &&
        !r4 && !l2keep && !l2keep_cr3 && !l2keep_xb1) {
        fprintf(stderr, "dp-bench: mode is all, seq, nsys, ncu:POINT, group:NAME, r4:POINT[,POINT], l2keep, "
                        "l2keep:cr3 or l2keep:xb1\n");
        return 2;
    }
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaDeviceProp prop = {};
    int driver = 0, runtime = 0;
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "properties") ||
        !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
        return 1;
    }
    printf("# dp-bench: %s, sm_%d%d, %d SMs, driver API %d, runtime %d, mode %s, start %s\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, driver, runtime, mode, utc_now().c_str());
    bool ok = true;
    {
        bench bn;
        ok = cuda_ok(cudaStreamCreateWithFlags(&bn.st, cudaStreamNonBlocking), "stream");
        std::vector<uint8_t> g;
        ok = ok && grid_buffer(g) && bn.grids.put(g.data(), g.size());
        // the layer input: fixed synthetic F32 rows and their decode format (written once)
        std::vector<float> x(size_t(max_rows) * max_k);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = static_cast<float>(int32_t((i * 2654435761u) >> 20) % 2001 - 1000) * 1e-3f;
        }
        const std::vector<float> sc = {0.731058598f, 0.182425521f, 0.5f, 0.952574134f};
        ok = ok && bn.xf.put(x.data(), x.size() * 4) && bn.scale.put(sc.data(), sc.size() * 4) &&
             fn::act::quantize(bn.xf.as<float>(), {bn.xc.as<int8_t>(), bn.xs.as<float>(), max_rows, max_k}, bn.st) &&
             cuda_ok(cudaMemcpyAsync(bn.xc2.p, bn.xc.p, bn.xc.n, cudaMemcpyDeviceToDevice, bn.st), "xc2") &&
             cuda_ok(cudaMemcpyAsync(bn.xs2.p, bn.xs.p, bn.xs.n, cudaMemcpyDeviceToDevice, bn.st), "xs2") &&
             cuda_ok(cudaMemcpyAsync(bn.xf_p.p, bn.xf.p, bn.xf.n, cudaMemcpyDeviceToDevice, bn.st), "xf_p") &&
             cuda_ok(cudaMemcpyAsync(bn.xc_p.p, bn.xc.p, bn.xc.n, cudaMemcpyDeviceToDevice, bn.st), "xc_p") &&
             cuda_ok(cudaMemcpyAsync(bn.xs_p.p, bn.xs.p, bn.xs.n, cudaMemcpyDeviceToDevice, bn.st), "xs_p") &&
             cuda_ok(cudaMemcpyAsync(bn.scale_p.p, bn.scale.p, bn.scale.n, cudaMemcpyDeviceToDevice, bn.st),
                     "scale_p") &&
             cuda_ok(cudaMemsetAsync(bn.flush_ef.p, 0x69, bn.flush_ef.n, bn.st), "flush_ef buffer") &&
             cuda_ok(cudaMemsetAsync(bn.flush.p, 0x5a, bn.flush.n, bn.st), "flush buffer") &&
             cuda_ok(cudaStreamSynchronize(bn.st), "x");
        // keep the card busy for warm_s before timing: memsets of a 64 MiB scratch buffer
        if (ok) {
            dev_mem scratch(size_t(64) << 20);
            const auto t0 = std::chrono::steady_clock::now();
            int64_t n = 0;
            while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
                for (int32_t i = 0; i < 32 && ok; ++i, ++n) {
                    ok = cuda_ok(cudaMemsetAsync(scratch.p, static_cast<int>(n & 0xff), scratch.n, bn.st), "memset");
                }
                ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "warm sync");
            }
            printf("# dp-bench: busy for %.2f s (%lld memsets of 64 MiB), timing from %s\n",
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                   static_cast<long long>(n), utc_now().c_str());
        }
        if (ok) {
            if (r4) {
                ok = run_r4(bn, mode + 3);
            } else if (l2keep_cr3) {
                ok = run_l2keep_cr3(bn, {1ll << 19, 3ll << 18, 1ll << 20}, "CR3");
            } else if (l2keep_xb1) { // XB1's probe 5: the experts' own code size, a 0.1 MB keep set
                ok = run_l2keep_cr3(bn, {102400}, "XB1");
            } else if (l2keep) {
                ok = run_l2keep(bn);
            } else if (ncu) {
                ok = run_all(bn, mode + 4);
            } else if (one_group) {
                ok = run_all(bn, nullptr, mode + 6);
            } else if (strcmp(mode, "all") == 0) {
                ok = run_all(bn, nullptr);
            } else {
                ok = run_seq(bn, strcmp(mode, "seq") == 0);
            }
        }
        ok = cuda_ok(cudaStreamSynchronize(bn.st), "final sync") && ok;
        if (bn.st != nullptr) {
            cudaStreamDestroy(bn.st);
        }
    }
    printf("# dp-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
