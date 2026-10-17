#include "fn/proto/xp1/xp1_bench.h"

#include "fn/proto/xp1/hx7.h"
#include "fn/proto/xp1/pair.h"
#include "fn/proto/xp1/standin.h"
#include "fn/proto/xp1/timing.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

namespace p1 = fn::xp1;

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

double median(std::vector<double> v) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2]; // the upper median of an even count
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(q * double(v.size())))];
}

struct jsonl {
    FILE * f = nullptr;
    ~jsonl() {
        if (f != nullptr) {
            fclose(f);
        }
    }
    void line(const std::string & s) {
        if (f != nullptr) {
            fprintf(f, "%s\n", s.c_str());
            fflush(f);
        }
    }
};

// the SM clock of a card now (MHz), from the probe kernel, outside any timed span
double sm_mhz(p1::pair & pr, int32_t c) {
    cudaSetDevice(pr.device(c));
    cudaStream_t s = static_cast<cudaStream_t>(pr.stream(c));
    uint64_t * buf = pr.stamps(c) + 8;
    uint64_t h[2] = {0u, 0u};
    if (!p1::clock_probe(buf, s) || cudaStreamSynchronize(s) != cudaSuccess ||
        cudaMemcpy(h, buf, 16, cudaMemcpyDeviceToHost) != cudaSuccess || h[1] == 0u) {
        return NAN;
    }
    return 1e3 * double(h[0]) / double(h[1]);
}

std::string num(double v) {
    char b[64];
    snprintf(b, sizeof(b), std::isfinite(v) ? "%.3f" : "null", v);
    return b;
}

// the settings: targets.json's per-piece figures (us), the shared expert split between its two stand-ins by bytes
struct settings {
    std::map<std::pair<int32_t, std::string>, std::pair<double, double>> by_key; // (layer, key) -> (target, today)
    bool load(const std::string & path) {
        FILE * f = fopen(path.c_str(), "r");
        if (f == nullptr) {
            return false;
        }
        int32_t layer = 0;
        char key[128];
        double t = 0.0, d = 0.0;
        while (fscanf(f, "%d %127s %lf %lf", &layer, key, &t, &d) == 4) {
            by_key[{layer, key}] = {t, d};
        }
        fclose(f);
        return !by_key.empty();
    }
    // the piece's setting at a speed, us; NAN when targets.json has none
    double of(p1::piece p, int32_t card, bool today) const {
        const p1::piece_info & pi = p1::piece_table[static_cast<int32_t>(p)];
        const auto it = by_key.find({pi.layer, pi.target_key});
        if (it == by_key.end()) {
            return NAN;
        }
        double v = today ? it->second.second : it->second.first;
        if (std::strcmp(pi.target_key, "shared_expert") == 0) {
            const bool gu = p == p1::piece::sh_gate_up10 || p == p1::piece::sh_gate_up11;
            const p1::piece pg = pi.layer == 10 ? p1::piece::sh_gate_up10 : p1::piece::sh_gate_up11;
            const p1::piece pd = pi.layer == 10 ? p1::piece::sh_down10 : p1::piece::sh_down11;
            const double bg = double(p1::piece_weight_bytes(pg, card)), bd = double(p1::piece_weight_bytes(pd, card));
            v *= (gu ? bg : bd) / (bg + bd); // [A] split by bytes
        }
        return v;
    }
};

// One op alone on its card, as HX2's harness times a kernel: a graph of rep x [L2 flush, op] against a graph of
// rep x [L2 flush], each launched between two events (the 1.024 us event tick and the launch are amortized over rep);
// per trial (A - B) / rep, so a figure is the op's time plus one node boundary, with its weights cold.
constexpr int64_t flush_used = 12ll << 20; // > 6 MiB of L2
struct alone_timer {
    p1::pair & pr;
    cudaEvent_t ev[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    bool init(std::string & err) {
        for (int32_t c = 0; c < 2; ++c) {
            cudaSetDevice(pr.device(c));
            for (int32_t i = 0; i < 2; ++i) {
                if (!cuda_ok(cudaEventCreate(&ev[c][i]), "event", err)) {
                    return false;
                }
            }
        }
        return true;
    }
    ~alone_timer() {
        for (int32_t c = 0; c < 2; ++c) {
            cudaSetDevice(pr.device(c));
            for (int32_t i = 0; i < 2; ++i) {
                if (ev[c][i] != nullptr) {
                    cudaEventDestroy(ev[c][i]);
                }
            }
        }
    }
    bool graph_of(int32_t c, const std::function<bool(void *)> * fn, int32_t rep, cudaGraphExec_t & x,
                  std::string & err) {
        cudaStream_t s = static_cast<cudaStream_t>(pr.stream(c));
        cudaGraph_t g = nullptr;
        bool ok = cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture", err);
        for (int32_t i = 0; ok && i < rep; ++i) {
            ok = p1::flush_l2(pr.flush_buffer(c), flush_used, s) && (fn == nullptr || (*fn)(s));
        }
        const cudaError_t e = cudaStreamEndCapture(s, &g);
        ok = ok && cuda_ok(e, "end capture", err) && cuda_ok(cudaGraphInstantiate(&x, g, 0), "instantiate", err);
        if (g != nullptr) {
            cudaGraphDestroy(g);
        }
        return ok || (err.empty() && (err = "alone graph", false));
    }
    bool time(int32_t c, cudaGraphExec_t x, double & us, std::string & err) {
        cudaStream_t s = static_cast<cudaStream_t>(pr.stream(c));
        float ms = 0.0f;
        const bool ok = cuda_ok(cudaEventRecord(ev[c][0], s), "record", err) &&
                        cuda_ok(cudaGraphLaunch(x, s), "launch", err) &&
                        cuda_ok(cudaEventRecord(ev[c][1], s), "record", err) &&
                        cuda_ok(cudaStreamSynchronize(s), "sync", err) &&
                        cuda_ok(cudaEventElapsedTime(&ms, ev[c][0], ev[c][1]), "elapsed", err);
        us = 1e3 * ms;
        return ok;
    }
    bool run(int32_t c, const std::function<bool(void *)> & fn, int32_t trials, std::vector<double> & us,
             std::string & err, int32_t rep = 20) {
        us.clear();
        cudaSetDevice(pr.device(c));
        cudaGraphExec_t xa = nullptr, xb = nullptr;
        bool ok = cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(pr.stream(c))), "sync", err) &&
                  graph_of(c, &fn, rep, xa, err) && graph_of(c, nullptr, rep, xb, err);
        double ta = 0.0, tb = 0.0;
        ok = ok && time(c, xa, ta, err) && time(c, xb, tb, err); // warm both
        for (int32_t i = 0; ok && i < trials; ++i) {
            ok = time(c, xb, tb, err) && time(c, xa, ta, err);
            if (ok) {
                us.push_back((ta - tb) / rep);
            }
        }
        if (xa != nullptr) {
            cudaGraphExecDestroy(xa);
        }
        if (xb != nullptr) {
            cudaGraphExecDestroy(xb);
        }
        return ok || (err.empty() && (err = "alone run failed", false));
    }
};

// An exchange isolated on both cards (pair::exchange_ops: rep repetitions in one graph a card, both directions at
// once): per trial each card's span between its launch events, divided by rep.
struct pair_timer {
    p1::pair & pr;
    bool run(const std::vector<p1::op> & ops, int32_t rep, int32_t trials, std::vector<double> span[2],
             std::string & err) {
        cudaGraphExec_t x[2] = {nullptr, nullptr};
        cudaEvent_t ev[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
        bool ok = pr.reset_exchange(err);
        for (int32_t c = 0; c < 2 && ok; ++c) {
            cudaSetDevice(pr.device(c));
            cudaStream_t s = static_cast<cudaStream_t>(pr.stream(c));
            cudaGraph_t g = nullptr;
            ok = cuda_ok(cudaEventCreate(&ev[c][0]), "event", err) &&
                 cuda_ok(cudaEventCreate(&ev[c][1]), "event", err) &&
                 cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture", err);
            for (const p1::op & o : ops) {
                ok = ok && (o.card != c || o.fn(s));
            }
            const cudaError_t e = cudaStreamEndCapture(s, &g);
            ok = ok && cuda_ok(e, "end capture", err) && cuda_ok(cudaGraphInstantiate(&x[c], g, 0), "instantiate", err);
            if (g != nullptr) {
                cudaGraphDestroy(g);
            }
        }
        span[0].clear();
        span[1].clear();
        for (int32_t i = 0; ok && i < trials + 1; ++i) {
            for (int32_t c = 1; c >= 0 && ok; --c) {
                cudaSetDevice(pr.device(c));
                cudaStream_t s = static_cast<cudaStream_t>(pr.stream(c));
                ok = cuda_ok(cudaEventRecord(ev[c][0], s), "record", err) &&
                     cuda_ok(cudaGraphLaunch(x[c], s), "launch", err) &&
                     cuda_ok(cudaEventRecord(ev[c][1], s), "record", err);
            }
            ok = ok && pr.sync(err);
            for (int32_t c = 0; c < 2 && ok && i > 0; ++c) {
                float ms = 0.0f;
                cudaSetDevice(pr.device(c));
                ok = cuda_ok(cudaEventElapsedTime(&ms, ev[c][0], ev[c][1]), "elapsed", err);
                span[c].push_back(1e3 * ms / rep);
            }
        }
        for (int32_t c = 0; c < 2; ++c) {
            cudaSetDevice(pr.device(c));
            if (x[c] != nullptr) {
                cudaGraphExecDestroy(x[c]);
            }
            for (cudaEvent_t e : ev[c]) {
                if (e != nullptr) {
                    cudaEventDestroy(e);
                }
            }
        }
        return ok && pr.reset_exchange(err);
    }
};

int32_t piece_of(const std::string & name, int32_t layer) {
    for (int32_t p = 0; p < p1::n_piece; ++p) {
        if (name == p1::piece_table[p].name && layer == p1::piece_table[p].layer) {
            return p;
        }
    }
    return -1;
}

// the mean distinct experts a row set routes to over the replayed decodes (Amendment 3's report)
double distinct_experts(const p1::pair::facts & fx, bool hx7, int32_t l, int32_t B, int32_t first, int32_t n) {
    double sum = 0.0;
    for (int32_t r = first; r < first + n; ++r) {
        std::set<int32_t> u;
        for (int32_t b = 0; b < B; ++b) {
            for (int32_t q = 0; q < 10; ++q) {
                u.insert(hx7 ? fx.hx7.routes[l][(size_t(r % fx.hx7.n_decode) * 3 + b) * 10 + q]
                             : (37 * q + 13 * b + 7 * (10 + l)) % 512);
            }
        }
        sum += double(u.size());
    }
    return sum / double(n);
}

} // namespace

int fnfl_xp1_bench(const fnfl_xp1_bench_args & a) {
    std::string err;
    const bool today = a.speed == "today";
    if (!today && a.speed != "target") {
        fprintf(stderr, "xp1-bench: --stand-in-speed is target or today\n");
        return 2;
    }
    settings st;
    if (!st.load(a.targets)) {
        fprintf(stderr, "xp1-bench: cannot read the settings %s\n", a.targets.c_str());
        return 2;
    }
    p1::pair pr;
    p1::pair_config cfg;
    cfg.dev[0] = a.dev0;
    cfg.dev[1] = a.dev1;
    cfg.tr = a.synthetic ? p1::traffic::synthetic : p1::traffic::hx7;
    cfg.synth_dir = a.synth_dir;
    cfg.hx7_dir = a.hx7_dir;
    cfg.dense = a.dense;
    if (a.dense && (a.variants.size() != 1 || a.variants[0] != "V2" || a.calibrate_only)) {
        fprintf(stderr, "xp1-bench: --dense times V2 only (--variants V2)\n");
        return 2;
    }
    cfg.gdn = a.gdn;
    cfg.qsa = a.qsa;
    if (!pr.init(cfg, err)) {
        fprintf(stderr, "xp1-bench: init: %s\n", err.c_str());
        return 1;
    }
    const p1::pair::facts & fx = pr.info();
    jsonl out;
    out.f = fopen((a.out_dir + "/bench.jsonl").c_str(), "a");
    if (out.f == nullptr) {
        fprintf(stderr, "xp1-bench: cannot write %s/bench.jsonl\n", a.out_dir.c_str());
        return 2;
    }
    const std::string gdn = a.gdn == 1 ? "core" : a.gdn == 2 ? "omit" : "standin";
    const std::string qsa = a.qsa == 1 ? "attend" : a.qsa == 2 ? "omit" : "standin";
    const std::string base = std::string("\"speed\":\"") + a.speed + "\",\"traffic\":\"" +
                             (a.synthetic ? "synthetic" : "hx7") + "\",\"dense\":" + (a.dense ? "true" : "false") +
                             ",\"gdn\":\"" + gdn + "\",\"qsa\":\"" + qsa + "\"";
    double offset_ns = 0.0, rtt_ns = 0.0;
    if (!pr.clock_offset(64, offset_ns, rtt_ns, err)) {
        fprintf(stderr, "xp1-bench: %s\n", err.c_str());
        return 1;
    }
    printf("xp1-bench: clock offset card 1 - card 0 %.0f ns, round trip %.0f ns (median of 63)\n", offset_ns, rtt_ns);
    out.line("{\"kind\":\"clock\",\"offset_ns\":" + num(offset_ns) + ",\"rtt_ns\":" + num(rtt_ns) + "}");

    // an eager V0 step fills the buffers the isolated runs read: the owner's QA1b body and the peer's received copy,
    // the routing
    for (int32_t B : a.rows) {
        for (const p1::variant v : {p1::variant::v0}) {
            std::vector<p1::op> prog;
            if (!pr.build(v, B, false, prog, err) || !pr.run_eager(prog, err)) {
                fprintf(stderr, "xp1-bench: warm step: %s\n", err.c_str());
                return 1;
            }
        }
    }
    alone_timer at{pr};
    if (!at.init(err)) {
        fprintf(stderr, "xp1-bench: %s\n", err.c_str());
        return 1;
    }

    // ---- calibration: the pads, read or measured ----
    std::map<std::tuple<int32_t, int32_t, int32_t>, int32_t> pads; // (B, card, piece) -> pad
    bool have_calib = false;
    if (FILE * f = fopen(a.calib.c_str(), "r")) {
        char sp[16];
        int32_t B = 0, c = 0, p = 0, pad = 0;
        double t0 = 0.0, set = 0.0, got = 0.0;
        while (fscanf(f, "%15s %d %d %d %d %lf %lf %lf", sp, &B, &c, &p, &pad, &t0, &set, &got) == 8) {
            if (a.speed == sp) {
                pads[{B, c, p}] = pad;
                have_calib = true;
            }
        }
        fclose(f);
    }
    if (!have_calib && a.dense) {
        fprintf(stderr, "xp1-bench: --dense reads an existing calibration (%s has none at %s)\n", a.calib.c_str(),
                a.speed.c_str());
        return 2;
    }
    if (!have_calib) {
        FILE * f = fopen(a.calib.c_str(), "a");
        if (f == nullptr) {
            fprintf(stderr, "xp1-bench: cannot write %s\n", a.calib.c_str());
            return 2;
        }
        printf("xp1-bench: calibrating the stand-ins at %s\n", a.speed.c_str());
        for (int32_t B : a.rows) {
            for (int32_t c = 0; c < 2; ++c) {
                for (int32_t p = 0; p < p1::n_piece; ++p) {
                    const p1::piece pc = static_cast<p1::piece>(p);
                    const double set = st.of(pc, c, today);
                    std::vector<double> us;
                    p1::op o;
                    const auto time_at = [&](int32_t pad, double & t) {
                        pr.set_pad(pc, c, pad);
                        if (!pr.piece_op(pc, c, B, o, err) || !at.run(c, o.fn, 7, us, err)) {
                            return false;
                        }
                        t = median(us);
                        return true;
                    };
                    double t0 = 0.0, t = 0.0;
                    if (!time_at(0, t0)) {
                        fprintf(stderr, "xp1-bench: calibration: %s\n", err.c_str());
                        return 1;
                    }
                    int32_t pad = 0;
                    t = t0;
                    if (std::isfinite(set) && t0 < 0.98 * set) {
                        // padding grows linearly: a probe, then the secant, until within 2%
                        int32_t pa = 64;
                        double ta = 0.0;
                        if (!time_at(pa, ta)) {
                            return 1;
                        }
                        double slope = std::max((ta - t0) / pa, 1e-4);
                        pad = std::max(0, static_cast<int32_t>(std::lround((set - t0) / slope)));
                        for (int32_t it = 0; it < 6; ++it) {
                            if (!time_at(pad, t)) {
                                return 1;
                            }
                            if (std::fabs(t - set) <= 0.02 * set) {
                                break;
                            }
                            slope = pad > 0 ? std::max((t - t0) / pad, 1e-4) : slope;
                            pad = std::max(0, static_cast<int32_t>(std::lround((set - t0) / slope)));
                        }
                    }
                    pr.set_pad(pc, c, pad);
                    if (!time_at(pad, t)) {
                        return 1;
                    }
                    pads[{B, c, p}] = pad;
                    fprintf(f, "%s %d %d %d %d %.3f %.3f %.3f\n", a.speed.c_str(), B, c, p, pad, t0, set, t);
                    fflush(f);
                }
            }
        }
        fclose(f);
    }
    const auto apply_pads = [&](int32_t B) {
        for (int32_t c = 0; c < 2; ++c) {
            for (int32_t p = 0; p < p1::n_piece; ++p) {
                const auto it = pads.find({B, c, p});
                const bool rest = a.dense && p1::dense_rest(static_cast<p1::piece>(p)); // its bytes only, unpadded
                pr.set_pad(static_cast<p1::piece>(p), c, it == pads.end() || rest ? 0 : it->second);
            }
        }
    };
    if (a.calibrate_only) {
        return 0;
    }

    // ---- the stand-ins alone at their settings, and the FNFL kernels and exchanges alone ----
    if (a.alone) {
        for (int32_t B : a.rows) {
            apply_pads(B);
            for (int32_t c = 0; c < 2; ++c) {
                for (int32_t p = 0; p < p1::n_piece; ++p) {
                    const p1::piece pc = static_cast<p1::piece>(p);
                    if (a.dense && p1::dense_replaced(pc)) {
                        continue; // DP2's launches, timed below
                    }
                    p1::op o;
                    std::vector<double> us;
                    if (!pr.piece_op(pc, c, B, o, err) || !at.run(c, o.fn, 7, us, err)) {
                        fprintf(stderr, "xp1-bench: alone: %s\n", err.c_str());
                        return 1;
                    }
                    const double set = a.dense && p1::dense_rest(pc) ? NAN : st.of(pc, c, today), med = median(us);
                    const double mhz = sm_mhz(pr, c);
                    const int64_t bytes = p1::piece_weight_bytes(pc, c);
                    printf("xp1-bench: alone B=%d card %d %-24s L%d %8.2f us (setting %6.2f, %+.1f%%), pad %d, "
                           "%.1f GB/s of weights\n",
                           B, c, p1::piece_table[p].name, p1::piece_table[p].layer, med, set, 100.0 * (med / set - 1.0),
                           pr.pad(pc, c), double(bytes) / (med * 1e3));
                    out.line("{\"kind\":\"alone_standin\"," + base + ",\"B\":" + std::to_string(B) + ",\"card\":" +
                             std::to_string(c) + ",\"piece\":\"" + p1::piece_table[p].name + "\",\"layer\":" +
                             std::to_string(p1::piece_table[p].layer) + ",\"median_us\":" + num(med) + ",\"p90_us\":" +
                             num(pct(us, 0.9)) + ",\"setting_us\":" + num(set) + ",\"pad\":" +
                             std::to_string(pr.pad(pc, c)) +
                             ",\"weight_bytes\":" + std::to_string(bytes) + ",\"sm_mhz\":" + num(mhz) + "}");
                }
            }
            // FNFL kernels alone: the ops of V0's and V1's programs that run on one card without a peer
            std::set<std::string> done;
            const std::vector<p1::variant> kv = a.dense ? std::vector<p1::variant>{p1::variant::v2}
                                                        : std::vector<p1::variant>{p1::variant::v0, p1::variant::v1};
            for (const p1::variant v : kv) {
                std::vector<p1::op> prog;
                if (!pr.reset_exchange(err) || !pr.build(v, B, true, prog, err)) {
                    fprintf(stderr, "xp1-bench: alone build: %s\n", err.c_str());
                    return 1;
                }
                for (const p1::op & o : prog) {
                    const bool single = o.role.find("recv") == std::string::npos && o.launches > 0 && o.piece != "HC" &&
                                        o.piece != "round";
                    const bool producer_send = piece_of(o.piece, o.layer) >= 0 && !o.role.empty();
                    if (!single || (piece_of(o.piece, o.layer) >= 0 && !producer_send)) {
                        continue; // receivers need their peer; plain stand-ins are timed above
                    }
                    const std::string key = std::string(p1::variant_name(v)) + "|" + o.piece + "|" + o.role + "|" +
                                            std::to_string(o.layer) + "|" + std::to_string(o.card);
                    const std::string key0 = o.piece + "|" + o.role + "|" + std::to_string(o.layer) + "|" +
                                             std::to_string(o.card);
                    if (done.count(key0) != 0) {
                        continue;
                    }
                    done.insert(key0);
                    std::vector<double> us;
                    if (!at.run(o.card, o.fn, 7, us, err)) {
                        fprintf(stderr, "xp1-bench: alone %s: %s\n", key.c_str(), err.c_str());
                        return 1;
                    }
                    printf("xp1-bench: alone B=%d card %d %-24s L%d %-14s %8.2f us (%s)\n", B, o.card, o.piece.c_str(),
                           o.layer, o.role.c_str(), median(us), p1::variant_name(v));
                    out.line("{\"kind\":\"alone_kernel\"," + base + ",\"B\":" + std::to_string(B) + ",\"card\":" +
                             std::to_string(o.card) + ",\"piece\":\"" + o.piece + "\",\"role\":\"" + o.role +
                             "\",\"layer\":" + std::to_string(o.layer) + ",\"variant\":\"" + p1::variant_name(v) +
                             "\",\"median_us\":" + num(median(us)) + ",\"p90_us\":" + num(pct(us, 0.9)) +
                             ",\"sm_mhz\":" +
                             num(sm_mhz(pr, o.card)) + "}");
                }
            }
            if (!pr.reset_exchange(err)) {
                return 1;
            }
            // exchanges and HC2's sublayer, isolated on both cards, both directions at once
            pair_timer ptm{pr};
            constexpr int32_t rep = 21; // a multiple of 3: the slots rotate cleanly
            for (const char * name : {"X-A", "X-F", "X-ROUTE", "X-SELECT", "X-A V1", "HC"}) {
                std::vector<p1::op> ops;
                std::vector<double> span[2];
                if (!pr.exchange_ops(name, B, rep, ops, err) || !ptm.run(ops, rep, 7, span, err)) {
                    fprintf(stderr, "xp1-bench: alone %s: %s\n", name, err.c_str());
                    return 1;
                }
                printf("xp1-bench: alone B=%d %-10s isolated, both directions: card 0 %7.2f us, card 1 %7.2f us "
                       "(median of 7 trials of %d)\n",
                       B, name, median(span[0]), median(span[1]), rep);
                out.line("{\"kind\":\"alone_exchange\"," + base + ",\"B\":" + std::to_string(B) + ",\"exchange\":\"" +
                         name + "\",\"card_us\":[" + num(median(span[0])) + "," + num(median(span[1])) + "],\"rep\":" +
                         std::to_string(rep) + ",\"sm_mhz\":[" + num(sm_mhz(pr, 0)) + "," + num(sm_mhz(pr, 1)) + "]}");
            }
        }
    }

    // ---- the pair: untraced spans and start skew, or one traced run; rounds interleave the variants ----
    const int32_t n_timed = a.replays;
    std::vector<p1::variant> vs;
    for (const std::string & vn : a.variants) {
        p1::variant v;
        if (!p1::parse_variant(vn, v)) {
            fprintf(stderr, "xp1-bench: unknown variant %s\n", vn.c_str());
            return 2;
        }
        vs.push_back(v);
    }
    struct acc {
        std::vector<double> span[2], skew, mhz[2], round_med[2];
    };
    std::map<std::pair<int32_t, int32_t>, acc> res; // (variant, B)
    const int32_t rounds = a.trace ? 1 : std::max(1, a.rounds);
    for (int32_t rd = 0; rd < rounds; ++rd) {
        for (const p1::variant v : vs) {
            const std::string vn = p1::variant_name(v);
            for (int32_t B : a.rows) {
                apply_pads(B);
                std::vector<p1::op> prog;
                if (!pr.reset_exchange(err) || !pr.reset_state(err) || !pr.build(v, B, true, prog, err) ||
                    !pr.capture(prog, err) || !pr.reset_round(err)) {
                    fprintf(stderr, "xp1-bench: %s B=%d: %s\n", vn.c_str(), B, err.c_str());
                    return 1;
                }
                if (rd == 0) {
                    // the node manifest: each card's kernel nodes in order, with the piece and its exchange roles
                    FILE * mf = fopen((a.out_dir + "/manifest-" + vn + "-B" + std::to_string(B) + ".tsv").c_str(), "w");
                    if (mf != nullptr) {
                        fprintf(mf, "card\tnode\tlayer\tpiece\trole\n");
                        int32_t n[2] = {0, 0};
                        for (const p1::op & o : prog) {
                            for (int32_t i = 0; i < o.launches; ++i) {
                                fprintf(mf, "%d\t%d\t%d\t%s\t%s\n", o.card, n[o.card]++, o.layer, o.piece.c_str(),
                                        i == o.launches - 1 ? o.role.c_str() : "");
                            }
                        }
                        fclose(mf);
                    }
                }
                acc & ac = res[{static_cast<int32_t>(v), B}];
                std::vector<double> sp_round[2];
                float ms[2];
                uint64_t t0[2];
                bool ok = true;
                for (int32_t i = 0; i < a.warmup + n_timed && ok; ++i) {
                    ok = pr.replay(ms, t0, err);
                    if (ok && i >= a.warmup) {
                        for (int32_t c = 0; c < 2; ++c) {
                            ac.span[c].push_back(1e3 * ms[c]);
                            sp_round[c].push_back(1e3 * ms[c]);
                        }
                        ac.skew.push_back(1e-3 * (double(t0[1]) - offset_ns - double(t0[0]))); // card 1 after card 0
                        if (!a.trace) { // the clock beside each replay, outside its span (not in a traced run)
                            ac.mhz[0].push_back(sm_mhz(pr, 0));
                            ac.mhz[1].push_back(sm_mhz(pr, 1));
                        }
                    }
                }
                pr.drop_graphs();
                if (!ok) {
                    fprintf(stderr, "xp1-bench: %s B=%d replay: %s\n", vn.c_str(), B, err.c_str());
                    return 1;
                }
                ac.round_med[0].push_back(median(sp_round[0]));
                ac.round_med[1].push_back(median(sp_round[1]));
            }
        }
    }
    const auto list = [](const std::vector<double> & v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) {
            s += (i ? "," : "") + num(v[i]);
        }
        return s + "]";
    };
    for (const p1::variant v : vs) {
        const std::string vn = p1::variant_name(v);
        for (int32_t B : a.rows) {
            const acc & ac = res[{static_cast<int32_t>(v), B}];
            const bool hx7 = !a.synthetic;
            const double u10 = distinct_experts(fx, hx7, 0, B, a.warmup, n_timed);
            const double u11 = distinct_experts(fx, hx7, 1, B, a.warmup, n_timed);
            printf("xp1-bench: %s B=%d %s%s: span card 0 %.2f us (p90 %.2f), card 1 %.2f us (p90 %.2f); start "
                   "skew %+.2f us (card 1 after card 0); SM MHz %.0f / %.0f; rounds %d; distinct experts L10 %.2f, "
                   "L11 %.2f\n",
                   vn.c_str(), B, a.speed.c_str(), a.trace ? " (traced)" : "", median(ac.span[0]), pct(ac.span[0], 0.9),
                   median(ac.span[1]), pct(ac.span[1], 0.9), median(ac.skew), median(ac.mhz[0]), median(ac.mhz[1]),
                   rounds, u10, u11);
            out.line("{\"kind\":\"pair\"," + base + ",\"variant\":\"" + vn + "\",\"B\":" + std::to_string(B) +
                     ",\"traced\":" + (a.trace ? "true" : "false") + ",\"rounds\":" + std::to_string(rounds) +
                     ",\"span_us\":[" + num(median(ac.span[0])) + "," + num(median(ac.span[1])) +
                     "],\"span_p90_us\":[" +
                     num(pct(ac.span[0], 0.9)) + "," + num(pct(ac.span[1], 0.9)) + "],\"round_medians_us\":[" +
                     list(ac.round_med[0]) + "," + list(ac.round_med[1]) + "],\"start_skew_us\":" +
                     num(median(ac.skew)) +
                     ",\"sm_mhz_median\":[" + num(median(ac.mhz[0])) + "," + num(median(ac.mhz[1])) +
                     "],\"sm_mhz_min\":[" + num(pct(ac.mhz[0], 0.0)) + "," + num(pct(ac.mhz[1], 0.0)) +
                     "],\"distinct_experts\":[" + num(u10) + "," + num(u11) + "],\"warmup\":" +
                     std::to_string(a.warmup) +
                     ",\"replays_per_round\":" + std::to_string(n_timed) + "}");
            out.line("{\"kind\":\"pair_replays\"," + base + ",\"variant\":\"" + vn + "\",\"B\":" + std::to_string(B) +
                     ",\"traced\":" + (a.trace ? "true" : "false") + ",\"card0_us\":" + list(ac.span[0]) +
                     ",\"card1_us\":" + list(ac.span[1]) + ",\"card0_mhz\":" + list(ac.mhz[0]) + ",\"card1_mhz\":" +
                     list(ac.mhz[1]) + "}");
        }
    }
    return 0;
}
