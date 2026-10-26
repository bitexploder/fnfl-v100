// fnfl-drive joint-kl (FMS2 item 7): the 16-bit joint X-A / X-F judged by KL, on one loaded model. Four lanes, each
// its own text: IDS (HX7's deep.ids) at offsets 0, 1,024, 2,048 and 3,072, ceil(1,024 / R) forced rounds of R rows a
// lane (every row committed) cover the reported positions 0 .. 1,023 of each lane. FMS13: the whole comparison runs at
// each d = 1 .. 4 in turn (R = d + 1; at d = 4, 205 rounds of five rows, FMS2's). On a four-slot engine whose state is
// laid out again in q8 KV (no weight reload):
//   lone:  each lane alone, forced rounds of R rows from empty state (the reference, its logits kept);
//   joint: the four lanes as one joint forced round a step (joint_frame::forced), at engine_config's defaults (FMS5:
//          X-A/X-F 16-bit off, the router on the tensor cores; FMS2 ran the 16-bit switch on);
//   yardstick: the state laid out again in BF16 KV, the lone pass again.
// Reported over every position (4 x 1,024): KL(lone || joint) and KL(lone BF16 KV || lone q8 KV) over the full
// vocabulary (FP64), p50, p99 and max; top-1 differences (the host's argmax, the lower ID on a tie) of each pair; each
// lane's first greedy divergence (joint against lone; reported, not gated); every round's error words (the joint word's
// err_x16_range among them). PASS: the joint's p50, p99, max and top-1 differences each at most the yardstick's, no
// error word set, at every d. Writes OUT/joint-kl-d<d>.json and OUT/joint-kl-d<d>-per-position.tsv for each d.
//
// Run in load 1 after rollback-test mu1 (rollback-test IDS OUT mu1 --joint-kl DEEP: the same process, the model taken
// from its engine), or standalone: fnfl-drive joint-kl DEEP OUT [real|dry] (dry: zero weights, plumbing only; the
// logits all equal, so every KL is 0).

#include "fn/kernels/head/head.h"
#include "fn/loader/run_load.h"
#include "fn/model/model.h"
#include "fn/runtime/joint.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/stage_cmds.h"
#include "fn/runtime/target_cmds.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace rt = fn::runtime;
namespace md = fn::model;

constexpr int32_t lanes = 4;
constexpr int32_t max_rows  = fn::round::max_rows;     // 5
constexpr int32_t positions = 1024;                    // reported a lane
constexpr int32_t V         = fn::head::n_vocab;       // 248,320
constexpr int32_t VC        = fn::head::n_vocab_card;  // 124,160

// FMS13: the forced rounds of R rows that cover the reported positions, and the tokens a lane they take
constexpr int32_t rounds_of(int32_t rows) {
    return (positions + rows - 1) / rows;
}  // 205 at R = 5

constexpr int32_t lane_tokens_of(int32_t rows) {
    return rounds_of(rows) * rows;
}                                          // 1,025 at R = 5

constexpr int32_t max_lane_tokens = 1026;  // R = 3's, the most
static_assert(lane_tokens_of(2) <= max_lane_tokens && lane_tokens_of(3) <= max_lane_tokens &&
                  lane_tokens_of(4) <= max_lane_tokens && lane_tokens_of(5) <= max_lane_tokens,
              "the source holds every R's lane tokens");

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "joint-kl: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool read_ids(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    out.clear();
    long v = -1;
    for (const char ch : s) {
        if (ch >= '0' && ch <= '9') {
            v = (v < 0 ? 0 : v) * 10 + (ch - '0');
        } else {
            if (v >= 0) {
                out.push_back(static_cast<int32_t>(v));
            }
            v = -1;
        }
    }
    if (v >= 0) {
        out.push_back(static_cast<int32_t>(v));
    }
    return true;
}

double percentile(std::vector<double> v, double q) { // target_cmds.cpp's (linear between the ranks)
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const double pos = q / 100.0 * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

// KL(p || q) over the full vocabulary, F32 logits, FP64 sums (target_cmds.cpp kl_div's)
double kl(const float * p, const float * q) {
    double mp = -HUGE_VAL, mq = -HUGE_VAL;
    for (int32_t i = 0; i < V; ++i) {
        mp = std::max(mp, double(p[i]));
        mq = std::max(mq, double(q[i]));
    }
    double sp = 0.0, sq = 0.0;
    for (int32_t i = 0; i < V; ++i) {
        sp += std::exp(double(p[i]) - mp);
        sq += std::exp(double(q[i]) - mq);
    }
    const double lsp = std::log(sp), lsq = std::log(sq);
    double k = 0.0;
    for (int32_t i = 0; i < V; ++i) {
        const double lp = double(p[i]) - mp - lsp;
        const double lq = double(q[i]) - mq - lsq;
        k += std::exp(lp) * (lp - lq);
    }
    return k;
}

int32_t top1(const float * p) { // the lower ID on a tie
    int32_t best = 0;
    for (int32_t i = 1; i < V; ++i) {
        best = p[i] > p[best] ? i : best;
    }
    return best;
}

// a pass's logits: lane k position p at (k * positions + p) * V, card 0's shard then card 1's
struct logit_store {
    std::vector<float> v;
    void init() { v.assign(size_t(lanes) * positions * V, 0.0f); }
    float * at(int32_t k, int32_t p) { return v.data() + (size_t(k) * positions + size_t(p)) * V; }
};

// rows 0 .. rows - 1 of a round of lane k from position p0: the slot's logits buffers into the store (positions past
// the reported ones dropped)
bool take_logits(rt::target & t, md::model & m, int32_t rows, int32_t k, int32_t p0, float * pinned, logit_store & s) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaMemcpy(pinned + size_t(c) * rows * VC, t.card(c).logits, size_t(rows) * VC * 4,
                                cudaMemcpyDeviceToHost),
                     "logits")) {
            return false;
        }
    }
    for (int32_t b = 0; b < rows && p0 + b < positions; ++b) {
        float * dst = s.at(k, p0 + b);
        std::memcpy(dst, pinned + size_t(b) * VC, size_t(VC) * 4);
        std::memcpy(dst + VC, pinned + size_t(rows) * VC + size_t(b) * VC, size_t(VC) * 4);
    }
    return true;
}

// a round's error words: every word 0; x16: the joint word's range bit seen
bool clean(const rt::round_result & r, int32_t & words, bool & x16) {
    bool ok = true;
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t i = 0; i < fn::round::max_err; ++i) {
            words += r.err[c][i] != 0u ? 1 : 0;
            x16 = x16 || (r.err[c][i] & fn::xchg::err_x16_range) != 0u;
            ok = ok && r.err[c][i] == 0u;
        }
    }
    return ok;
}

bool lay_out(md::model & m, int32_t cells, bool kv_q8) {
    return fnfl_lay_out(m, cells, lanes, kv_q8);
}

// a four-slot engine on m (joint frame, graphs, d: rollback-test mu1's at FMS13's d), engine_config's joint switches,
// and each lane's pages mapped by a read of its text (the forced passes then run from empty state on those pages)
bool make_engine(const fnfl_target_args &   a,
                 std::unique_ptr<md::model> m,
                 int32_t                    cells,
                 int32_t                    d,
                 const std::vector<int32_t> (&tok)[lanes],
                 rt::engine & e) {
    fnfl_target_args da = a;
    da.draft_attention = "sparse"; // the serving setting
    da.draft_attention_given = true;
    rt::draft_options dopt;
    if (!fnfl_draft_settings(da, true, "joint-kl", dopt)) {
        return false;
    }
    rt::engine_config c;
    c.model_path = a.model_path;
    c.n_ctx_slot = cells;
    c.draft = dopt;
    c.eog = {248046};
    c.spec_d     = d;
    c.timeout_s = 60.0;
    std::string err;
    if (!e.init(std::move(m), c, err)) {
        fprintf(stderr, "joint-kl: engine: %s\n", err.c_str());
        return false;
    }
    printf("joint-kl: %s\n", e.settings().c_str());
    if (e.test_joint_frame() == nullptr || !e.test_joint_frame()->ready()) {
        fprintf(stderr, "joint-kl: the engine has no joint frame\n");
        return false;
    }
    rt::request_options ro;
    ro.sampling.temp = 0.0f;
    ro.max_tokens = 100000;
    for (int32_t k = 0; k < lanes; ++k) {
        int32_t np = 0;
        if (e.set_sampling(k, ro) != rt::run_status::ok || e.truncate(k, 0) != rt::run_status::ok ||
            e.read_begin(k, tok[k].data(), int32_t(tok[k].size()), 0, np) != 0) {
            fprintf(stderr, "joint-kl: lane %d's read: %s\n", k, e.error().c_str());
            return false;
        }
        rt::read_status st = rt::read_status::yielded;
        while (st == rt::read_status::yielded || st == rt::read_status::chunk_done) {
            st = e.read_step(k, fn::n_layer);
        }
        if (st != rt::read_status::done) {
            fprintf(stderr, "joint-kl: lane %d's read did not finish\n", k);
            return false;
        }
    }
    return true;
}

// each lane alone: reset, its text as the ledger, ceil(1,024 / R) forced rounds of R rows; logits into s
bool lone_pass(rt::engine & e,
               int32_t      rows,
               const std::vector<int32_t> (&tok)[lanes],
               float *       pinned,
               logit_store & s,
               int32_t &     err_words,
               bool &        x16) {
    md::model & m = e.test_model();
    std::string err;
    for (int32_t k = 0; k < lanes; ++k) {
        rt::target & t = e.test_target(k);
        if (!t.reset(err)) {
            fprintf(stderr, "joint-kl: lane %d reset: %s\n", k, err.c_str());
            return false;
        }
        t.set_prompt(tok[k]);
        for (int32_t i = 0; i < rounds_of(rows); ++i) {
            rt::round_result r;
            if (!t.forced(rows, r, err)) {
                fprintf(stderr, "joint-kl: lane %d lone round %d: %s\n", k, i, err.c_str());
                return false;
            }
            clean(r, err_words, x16);
            if (!take_logits(t, m, rows, k, i * rows, pinned, s)) {
                return false;
            }
        }
    }
    return true;
}

// the four lanes jointly: each reset with its text as the ledger, then ceil(1,024 / R) joint forced rounds of R rows
// each
bool joint_pass(rt::engine & e,
                int32_t      rows,
                const std::vector<int32_t> (&tok)[lanes],
                float *       pinned,
                logit_store & s,
                int32_t &     err_words,
                bool &        x16) {
    md::model & m = e.test_model();
    rt::joint_frame * jf = e.test_joint_frame();
    std::string err;
    rt::target * tt[lanes];
    for (int32_t k = 0; k < lanes; ++k) {
        tt[k] = &e.test_target(k);
        if (!tt[k]->reset(err)) {
            fprintf(stderr, "joint-kl: lane %d reset: %s\n", k, err.c_str());
            return false;
        }
        tt[k]->set_prompt(tok[k]);
    }
    for (int32_t i = 0; i < rounds_of(rows); ++i) {
        rt::round_result r[lanes];
        bool ok[lanes] = {};
        std::string why[lanes];
        if (!jf->forced(tt, lanes, rows, r, ok, why, err)) {
            fprintf(stderr, "joint-kl: joint round %d did not launch: %s\n", i, err.c_str());
            return false;
        }
        for (int32_t k = 0; k < lanes; ++k) {
            clean(r[k], err_words, x16);
            if (!ok[k]) {
                fprintf(stderr, "joint-kl: joint round %d lane %d: %s\n", i, k, why[k].c_str());
                return false;
            }
            if (!take_logits(*tt[k], m, rows, k, i * rows, pinned, s)) {
                return false;
            }
        }
    }
    return true;
}

// KL(p || q) and both top-1s at every position, over threads
void compare(logit_store & p, logit_store & q, std::vector<double> & k, std::vector<int32_t> & tp,
             std::vector<int32_t> & tq) {
    const int32_t n = lanes * positions;
    k.assign(size_t(n), 0.0);
    tp.assign(size_t(n), -1);
    tq.assign(size_t(n), -1);
    const int32_t nt = std::max(1, std::min(32, int32_t(std::thread::hardware_concurrency())));
    std::vector<std::thread> th;
    for (int32_t w = 0; w < nt; ++w) {
        th.emplace_back([&, w]() {
            for (int32_t i = w; i < n; i += nt) {
                const float * a = p.at(i / positions, i % positions);
                const float * b = q.at(i / positions, i % positions);
                k[size_t(i)] = kl(a, b);
                tp[size_t(i)] = top1(a);
                tq[size_t(i)] = top1(b);
            }
        });
    }
    for (std::thread & t : th) {
        t.join();
    }
}

} // namespace

namespace {

// FMS13: one d's comparison (R = d + 1): the lone and joint passes in q8 KV, the yardstick's lone pass in BF16 KV, the
// statistics and the outputs. m comes back for the next d (null after a failure). False when a pass did not run.
bool kl_at(const fnfl_target_args &     a,
           std::unique_ptr<md::model> & m,
           int32_t                      cells,
           const std::vector<int32_t> & src,
           int32_t                      d,
           float *                      pinned,
           logit_store &                lone,
           logit_store &                other,
           bool &                       pass) {
    const int32_t        rows        = d + 1;
    const int32_t        rounds      = rounds_of(rows);
    const int32_t        lane_tokens = lane_tokens_of(rows);
    std::vector<int32_t> tok[lanes];
    for (int32_t k = 0; k < lanes; ++k) {
        tok[k].assign(src.begin() + k * positions, src.begin() + k * positions + lane_tokens);
    }
    printf(
        "joint-kl: d = %d: four lanes at offsets 0, 1024, 2048, 3072, %d tokens each (%d forced rounds of %d rows), "
        "positions 0 .. %d of each reported\n",
        d, lane_tokens, rounds, rows, positions - 1);
    int32_t w_lone = 0, w_joint = 0, w_yard = 0;
    bool x_lone = false, x_joint = false, x_yard = false;
    std::vector<double> kj, ky;
    std::vector<int32_t> t_lone, t_joint, t_lone2, t_bf16;
    // q8 KV: the lone reference, then the joint pass
    if (!lay_out(*m, cells, true)) {
        return false;
    }
    {
        rt::engine e;
        if (!make_engine(a, std::move(m), cells, d, tok, e) || !lone_pass(e, rows, tok, pinned, lone, w_lone, x_lone)) {
            return false;
        }
        printf("joint-kl: d = %d: lone pass (q8 KV): %d lanes x %d rounds; error words set %d\n", d, lanes, rounds,
               w_lone);
        const int64_t r0 = e.test_joint_frame()->rounds();
        if (!joint_pass(e, rows, tok, pinned, other, w_joint, x_joint)) {
            return false;
        }
        printf(
            "joint-kl: d = %d: joint pass (q8 KV, X-A/X-F 16-bit %s, router on the tensor cores %s, dense FP16 %s): "
            "%lld joint rounds of %d lanes at R = %d, %d captures; error words set %d; err_x16_range %s\n",
            d, e.test_joint_frame()->x16() ? "on" : "off", e.test_joint_frame()->router_tc() ? "on" : "off",
            e.test_joint_frame()->dense_f16() ? "on" : "off",
            static_cast<long long>(e.test_joint_frame()->rounds() - r0), lanes, rows, e.test_joint_frame()->captures(),
            w_joint, x_joint ? "SET" : "never set");
        compare(lone, other, kj, t_lone, t_joint);
        m = e.test_take_model();
    }
    // BF16 KV: the yardstick's lone pass
    if (m == nullptr || !lay_out(*m, cells, false)) {
        return false;
    }
    {
        rt::engine e;
        if (!make_engine(a, std::move(m), cells, d, tok, e) ||
            !lone_pass(e, rows, tok, pinned, other, w_yard, x_yard)) {
            return false;
        }
        printf("joint-kl: d = %d: lone pass (BF16 KV, the yardstick): %d lanes x %d rounds; error words set %d\n", d,
               lanes, rounds, w_yard);
        compare(other, lone, ky, t_bf16, t_lone2);
        m = e.test_take_model();
    }
    // the statistics
    const int32_t n = lanes * positions;
    int32_t dj = 0, dy = 0;
    int32_t first[lanes];
    for (int32_t k = 0; k < lanes; ++k) {
        first[k] = -1;
        for (int32_t p = 0; p < positions; ++p) {
            const size_t i = size_t(k) * positions + size_t(p);
            if (t_joint[i] != t_lone[i] && first[k] < 0) {
                first[k] = p;
            }
        }
    }
    for (int32_t i = 0; i < n; ++i) {
        dj += t_joint[size_t(i)] != t_lone[size_t(i)] ? 1 : 0;
        dy += t_bf16[size_t(i)] != t_lone2[size_t(i)] ? 1 : 0;
    }
    const double j50 = percentile(kj, 50), j99 = percentile(kj, 99), jmax = *std::max_element(kj.begin(), kj.end());
    const double y50 = percentile(ky, 50), y99 = percentile(ky, 99), ymax = *std::max_element(ky.begin(), ky.end());
    printf("joint-kl: d = %d: KL(lone || joint) over %d positions: p50 %.6g p99 %.6g max %.6g; top-1 differs at %d\n",
           d, n, j50, j99, jmax, dj);
    printf(
        "joint-kl: d = %d: yardstick KL(lone BF16 KV || lone q8 KV): p50 %.6g p99 %.6g max %.6g; top-1 differs at "
        "%d\n",
        d, y50, y99, ymax, dy);
    printf("joint-kl: d = %d: each lane's first greedy divergence (joint against lone; reported, not gated):", d);
    for (int32_t k = 0; k < lanes; ++k) {
        printf(" lane %d %s%d", k, first[k] < 0 ? "none (" : "position ", first[k] < 0 ? positions : first[k]);
        printf("%s", first[k] < 0 ? " positions)" : "");
    }
    printf("\n");
    const bool p50 = j50 <= y50, p99 = j99 <= y99, pmax = jmax <= ymax, ptop = dj <= dy;
    const bool words = w_lone == 0 && w_joint == 0 && w_yard == 0 && !x_joint;
    printf(
        "joint-kl: d = %d: joint p50 %s, p99 %s, max %s, top-1 %s the yardstick's; the overflow flag %s; every "
        "error word %s\n",
        d, p50 ? "<=" : "ABOVE", p99 ? "<=" : "ABOVE", pmax ? "<=" : "ABOVE", ptop ? "<=" : "ABOVE",
        x_joint ? "SET" : "never set", words ? "0" : "NOT 0");
    const std::string stem = a.out_dir + "/joint-kl-d" + std::to_string(d);
    FILE *            jf   = fopen((stem + ".json").c_str(), "w");
    FILE *            tf   = fopen((stem + "-per-position.tsv").c_str(), "w");
    if (jf == nullptr || tf == nullptr) {
        fprintf(stderr, "joint-kl: cannot write the outputs in %s\n", a.out_dir.c_str());
        if (jf != nullptr) {
            fclose(jf);
        }
        if (tf != nullptr) {
            fclose(tf);
        }
        return false;
    }
    fprintf(jf,
            "{\"d\": %d, \"rows\": %d, \"positions\": %d, \"joint\": {\"p50\": %.9g, \"p99\": %.9g, \"max\": %.9g, "
            "\"top1_differ\": %d}, \"yardstick\": {\"p50\": %.9g, \"p99\": %.9g, \"max\": %.9g, \"top1_differ\": %d}, "
            "\"first_divergence\": [%d, %d, %d, %d], \"error_words\": [%d, %d, %d], \"x16_range\": %s, \"dry\": %s}\n",
            d, rows, n, j50, j99, jmax, dj, y50, y99, ymax, dy, first[0], first[1], first[2], first[3], w_lone, w_joint,
            w_yard, x_joint ? "true" : "false", a.mode == "dry" ? "true" : "false");
    fprintf(tf, "lane\tpos\tkl_lone_joint\tkl_bf16_q8\ttop1_lone\ttop1_joint\ttop1_bf16\n");
    for (int32_t i = 0; i < n; ++i) {
        fprintf(tf, "%d\t%d\t%.9g\t%.9g\t%d\t%d\t%d\n", i / positions, i % positions, kj[size_t(i)], ky[size_t(i)],
                t_lone[size_t(i)], t_joint[size_t(i)], t_bf16[size_t(i)]);
    }
    fclose(jf);
    fclose(tf);
    pass = p50 && p99 && pmax && ptop && words;
    printf("joint-kl: d = %d: %s\n", d, pass ? "PASS" : "FAIL");
    return true;
}

// ---- TTTS1: joint-kl on the split (this box the front, the follower the back, over the link) ----

namespace st = fn::runtime::stage;

// the split's engine on m (a front stage on the session's buffers and crossing), the follower's rebuilt alike with
// the same d, KV format and switches; each lane's pages mapped by a read of its text, as make_engine
bool make_engine_split(const fnfl_target_args &   a,
                       std::unique_ptr<md::model> m,
                       int32_t                    cells,
                       int32_t                    d,
                       bool                       kv_q8,
                       const std::vector<int32_t> (&tok)[lanes],
                       fnfl_split &                  sp,
                       std::unique_ptr<rt::engine> & e) {
    fnfl_target_args da      = a;
    da.draft_attention       = "sparse";
    da.draft_attention_given = true;
    rt::draft_options dopt;
    if (!fnfl_draft_settings(da, true, "joint-kl", dopt)) {
        return false;
    }
    rt::engine_config c;
    c.model_path = a.model_path;
    c.n_ctx_slot = cells;
    c.draft      = dopt;
    c.eog        = { 248046 };
    c.spec_d     = d;
    c.timeout_s  = 60.0;
    c.stage      = 1;
    c.stage_buf  = sp.buf.slot;
    std::string err;
    e = std::make_unique<rt::engine>();
    if (!e->init(std::move(m), c, err)) {
        fprintf(stderr, "joint-kl: engine: %s\n", err.c_str());
        return false;
    }
    e->prefill_driver().set_stage(&sp.cross);
    st::leader & L = *sp.lead;
    L.rebind(*e);
    st::ctrl rc;
    rc.a    = kv_q8 ? 1 : 0;
    rc.b    = d;
    rc.c    = (c.joint_x16 ? st::sw_joint_x16 : 0) | (c.joint_router_tc ? st::sw_router_tc : 0) |
              (c.joint_dense_f16 ? st::sw_dense_f16 : 0);
    rc.base = cells;
    if (L.reconfigure(rc) != rt::run_status::ok) {
        fprintf(stderr, "joint-kl: the follower's engine: %s\n", L.error().c_str());
        return false;
    }
    printf("joint-kl: %s; the follower rebuilt alike (d %d, %s KV)\n", e->settings().c_str(), d, kv_q8 ? "q8" : "BF16");
    if (e->test_joint_frame() == nullptr || !e->test_joint_frame()->ready()) {
        fprintf(stderr, "joint-kl: the engine has no joint frame\n");
        return false;
    }
    rt::request_options ro;
    ro.sampling.temp = 0.0f;
    ro.max_tokens    = 100000;
    for (int32_t k = 0; k < lanes; ++k) {
        int32_t np = 0;
        if (L.set_sampling(k, ro) != rt::run_status::ok || L.truncate(k, 0) != rt::run_status::ok ||
            L.read_begin(k, tok[k].data(), int32_t(tok[k].size()), 0, np) != 0) {
            fprintf(stderr, "joint-kl: lane %d's read: %s\n", k, L.error().c_str());
            return false;
        }
        rt::read_status rs = rt::read_status::yielded;
        while (rs == rt::read_status::yielded || rs == rt::read_status::chunk_done) {
            rs = L.read_step(k, fn::n_layer, rt::read_mode::shared);
        }
        if (rs != rt::read_status::done) {
            fprintf(stderr, "joint-kl: lane %d's read did not finish: %s\n", k, L.error().c_str());
            return false;
        }
    }
    return true;
}

// a forced member's back logits (card 0's rows, then card 1's) into the store at lane k from position p0
void store_logits(const std::vector<float> & lg, int32_t rows, int32_t k, int32_t p0, logit_store & s) {
    for (int32_t b = 0; b < rows && p0 + b < positions; ++b) {
        float * dst = s.at(k, p0 + b);
        std::memcpy(dst, lg.data() + size_t(b) * VC, size_t(VC) * 4);
        std::memcpy(dst + VC, lg.data() + size_t(rows) * VC + size_t(b) * VC, size_t(VC) * 4);
    }
}

// each lane alone (a forced step group of one a round), from a reset with its text as the ledger on both boxes
bool lone_pass_split(st::leader & L,
                     int32_t      rows,
                     const std::vector<int32_t> (&tok)[lanes],
                     logit_store & s,
                     int32_t &     failed) {
    for (int32_t k = 0; k < lanes; ++k) {
        if (L.test_prompt(k, tok[k]) != rt::run_status::ok) {
            fprintf(stderr, "joint-kl: lane %d reset: %s\n", k, L.error().c_str());
            return false;
        }
        for (int32_t i = 0; i < rounds_of(rows); ++i) {
            rt::joint_step_out r;
            const int32_t      one = 1;
            L.forced_groups(&k, &one, 1, rows, &r);
            if (r.st != rt::run_status::ok) {
                ++failed;
                fprintf(stderr, "joint-kl: lane %d lone round %d: %s\n", k, i, r.why.c_str());
                return false;
            }
            store_logits(L.forced_logits(k), rows, k, i * rows, s);
        }
    }
    return true;
}

// the four lanes jointly: groups 1, one U4 group a round; groups 2, two U2 groups pipelined (the second's front half
// runs while the first's back half does), each a joint round of its two lanes
bool joint_pass_split(st::leader & L,
                      int32_t      groups,
                      int32_t      rows,
                      const std::vector<int32_t> (&tok)[lanes],
                      logit_store & s,
                      int32_t &     failed,
                      int32_t &     joint_lanes) {
    for (int32_t k = 0; k < lanes; ++k) {
        if (L.test_prompt(k, tok[k]) != rt::run_status::ok) {
            fprintf(stderr, "joint-kl: lane %d reset: %s\n", k, L.error().c_str());
            return false;
        }
    }
    const int32_t all[lanes] = { 0, 1, 2, 3 };
    for (int32_t i = 0; i < rounds_of(rows); ++i) {
        rt::joint_step_out r[lanes];
        const int32_t      sizes[2] = { groups == 1 ? lanes : 2, 2 };
        L.forced_groups(all, sizes, groups, rows, r);
        for (int32_t k = 0; k < lanes; ++k) {
            if (r[k].st != rt::run_status::ok) {
                ++failed;
                fprintf(stderr, "joint-kl: joint round %d lane %d: %s\n", i, k, r[k].why.c_str());
                return false;
            }
            joint_lanes += r[k].joint ? 1 : 0;
            store_logits(L.forced_logits(k), rows, k, i * rows, s);
        }
    }
    return true;
}

// the statistics of KL(p || q) and the top-1s; the outputs at stem
struct kl_stats {
    double  p50 = 0, p99 = 0, max = 0;
    int32_t top1         = 0;
    int32_t first[lanes] = { -1, -1, -1, -1 };
};

kl_stats stats_of(const std::vector<double> & k, const std::vector<int32_t> & tp, const std::vector<int32_t> & tq) {
    kl_stats x;
    x.p50 = percentile(k, 50);
    x.p99 = percentile(k, 99);
    x.max = *std::max_element(k.begin(), k.end());
    for (size_t i = 0; i < k.size(); ++i) {
        x.top1 += tp[i] != tq[i] ? 1 : 0;
        const int32_t lane = int32_t(i) / positions;
        if (tp[i] != tq[i] && x.first[lane] < 0) {
            x.first[lane] = int32_t(i) % positions;
        }
    }
    return x;
}

// one d on the split: the lone pass (q8 KV), each requested joint pass (U4; U2 + U2 pipelined), the yardstick's lone
// pass (BF16 KV on both boxes), each joint judged against the yardstick as kl_at judges one box's
bool kl_at_split(const fnfl_target_args &     a,
                 std::unique_ptr<md::model> & m,
                 int32_t                      cells,
                 const std::vector<int32_t> & src,
                 int32_t                      d,
                 fnfl_split &                 sp,
                 logit_store &                lone,
                 logit_store &                other,
                 bool &                       pass) {
    const int32_t        rows        = d + 1;
    const int32_t        rounds      = rounds_of(rows);
    const int32_t        lane_tokens = lane_tokens_of(rows);
    std::vector<int32_t> tok[lanes];
    for (int32_t k = 0; k < lanes; ++k) {
        tok[k].assign(src.begin() + k * positions, src.begin() + k * positions + lane_tokens);
    }
    printf(
        "joint-kl: d = %d (the split, k %d): four lanes at offsets 0, 1024, 2048, 3072, %d tokens each (%d forced "
        "rounds of %d rows), positions 0 .. %d of each reported\n",
        d, a.split_k, lane_tokens, rounds, rows, positions - 1);
    st::leader &         L = *sp.lead;
    std::vector<int32_t> gs;
    if (a.split_groups == 0 || a.split_groups == 1) {
        gs.push_back(1);
    }
    if (a.split_groups == 0 || a.split_groups == 2) {
        gs.push_back(2);
    }
    // q8 KV: the lone reference, then each joint pass
    if (!lay_out(*m, cells, true)) {
        return false;
    }
    std::vector<kl_stats>             js;
    std::vector<std::vector<double>>  jk;
    std::vector<std::vector<int32_t>> jt;
    std::vector<int32_t>              t_lone, w_joint(gs.size(), 0);
    int32_t                           w_lone = 0;
    {
        std::unique_ptr<rt::engine> e;
        if (!make_engine_split(a, std::move(m), cells, d, true, tok, sp, e) ||
            !lone_pass_split(L, rows, tok, lone, w_lone)) {
            return false;
        }
        printf("joint-kl: d = %d: lone pass (q8 KV, the split): %d lanes x %d rounds\n", d, lanes, rounds);
        for (size_t q = 0; q < gs.size(); ++q) {
            int32_t       joint_lanes = 0;
            const int64_t r0          = e->test_joint_frame()->rounds();
            if (!joint_pass_split(L, gs[q], rows, tok, other, w_joint[q], joint_lanes)) {
                return false;
            }
            printf(
                "joint-kl: d = %d: joint pass (q8 KV, the split, %s; X-A/X-F 16-bit %s, router on the tensor cores "
                "%s, dense FP16 %s): %lld front joint rounds, %d joint lane rounds; failed rounds %d\n",
                d, gs[q] == 1 ? "one U4 group" : "two pipelined U2 groups", e->test_joint_frame()->x16() ? "on" : "off",
                e->test_joint_frame()->router_tc() ? "on" : "off", e->test_joint_frame()->dense_f16() ? "on" : "off",
                static_cast<long long>(e->test_joint_frame()->rounds() - r0), joint_lanes, w_joint[q]);
            std::vector<double>  k;
            std::vector<int32_t> tl, tj;
            compare(lone, other, k, tl, tj);
            js.push_back(stats_of(k, tl, tj));
            jk.push_back(std::move(k));
            jt.push_back(std::move(tj));
            t_lone = tl;
        }
        m = e->test_take_model();
    }
    // BF16 KV on both boxes: the yardstick's lone pass
    if (m == nullptr || !lay_out(*m, cells, false)) {
        return false;
    }
    std::vector<double>  ky;
    std::vector<int32_t> t_bf16, t_lone2;
    int32_t              w_yard = 0;
    {
        std::unique_ptr<rt::engine> e;
        if (!make_engine_split(a, std::move(m), cells, d, false, tok, sp, e) ||
            !lone_pass_split(L, rows, tok, other, w_yard)) {
            return false;
        }
        printf("joint-kl: d = %d: lone pass (BF16 KV, the split, the yardstick): %d lanes x %d rounds\n", d, lanes,
               rounds);
        compare(other, lone, ky, t_bf16, t_lone2);
        m = e->test_take_model();
    }
    const kl_stats y = stats_of(ky, t_bf16, t_lone2);
    printf(
        "joint-kl: d = %d: yardstick KL(lone BF16 KV || lone q8 KV): p50 %.6g p99 %.6g max %.6g; top-1 differs at "
        "%d\n",
        d, y.p50, y.p99, y.max, y.top1);
    pass = true;
    for (size_t q = 0; q < gs.size(); ++q) {
        const kl_stats & j    = js[q];
        const char *     name = gs[q] == 1 ? "U4" : "U2+U2";
        printf(
            "joint-kl: d = %d: %s: KL(lone || joint) over %d positions: p50 %.6g p99 %.6g max %.6g; top-1 differs "
            "at %d\n",
            d, name, lanes * positions, j.p50, j.p99, j.max, j.top1);
        printf("joint-kl: d = %d: %s: each lane's first greedy divergence (joint against lone; reported, not gated):",
               d, name);
        for (int32_t k = 0; k < lanes; ++k) {
            printf(" lane %d %s%d%s", k, j.first[k] < 0 ? "none (" : "position ",
                   j.first[k] < 0 ? positions : j.first[k], j.first[k] < 0 ? " positions)" : "");
        }
        printf("\n");
        const bool p50 = j.p50 <= y.p50, p99 = j.p99 <= y.p99, pmax = j.max <= y.max, ptop = j.top1 <= y.top1;
        const bool words = w_lone == 0 && w_joint[q] == 0 && w_yard == 0;
        printf("joint-kl: d = %d: %s: joint p50 %s, p99 %s, max %s, top-1 %s the yardstick's; every round %s\n", d,
               name, p50 ? "<=" : "ABOVE", p99 ? "<=" : "ABOVE", pmax ? "<=" : "ABOVE", ptop ? "<=" : "ABOVE",
               words ? "clean (a set error word fails its round on either box)" : "NOT clean");
        const std::string stem = a.out_dir + "/joint-kl-d" + std::to_string(d) + "-g" + std::to_string(gs[q]);
        FILE *            jf   = fopen((stem + ".json").c_str(), "w");
        FILE *            tf   = fopen((stem + "-per-position.tsv").c_str(), "w");
        if (jf == nullptr || tf == nullptr) {
            fprintf(stderr, "joint-kl: cannot write the outputs in %s\n", a.out_dir.c_str());
            if (jf != nullptr) {
                fclose(jf);
            }
            if (tf != nullptr) {
                fclose(tf);
            }
            return false;
        }
        fprintf(jf,
                "{\"d\": %d, \"rows\": %d, \"positions\": %d, \"split\": {\"k\": %d, \"groups\": %d}, \"joint\": "
                "{\"p50\": %.9g, \"p99\": %.9g, \"max\": %.9g, \"top1_differ\": %d}, \"yardstick\": {\"p50\": %.9g, "
                "\"p99\": %.9g, \"max\": %.9g, \"top1_differ\": %d}, \"first_divergence\": [%d, %d, %d, %d], "
                "\"failed_rounds\": [%d, %d, %d], \"dry\": %s}\n",
                d, rows, lanes * positions, a.split_k, gs[q], j.p50, j.p99, j.max, j.top1, y.p50, y.p99, y.max, y.top1,
                j.first[0], j.first[1], j.first[2], j.first[3], w_lone, w_joint[q], w_yard,
                a.mode == "dry" ? "true" : "false");
        fprintf(tf, "lane\tpos\tkl_lone_joint\tkl_bf16_q8\ttop1_lone\ttop1_joint\ttop1_bf16\n");
        for (int32_t i = 0; i < lanes * positions; ++i) {
            fprintf(tf, "%d\t%d\t%.9g\t%.9g\t%d\t%d\t%d\n", i / positions, i % positions, jk[q][size_t(i)],
                    ky[size_t(i)], t_lone[size_t(i)], jt[q][size_t(i)], t_bf16[size_t(i)]);
        }
        fclose(jf);
        fclose(tf);
        const bool ok = p50 && p99 && pmax && ptop && words;
        printf("joint-kl: d = %d: %s: %s\n", d, name, ok ? "PASS" : "FAIL");
        pass = pass && ok;
    }
    return true;
}
}  // namespace

// TTTS1: joint-kl's lay_out, any slot count (the split follower's reconfigure lays its state out alike)
// the state laid out again (release, reserve in kv_q8 or BF16, the EX1 grids), the weights kept
bool fnfl_lay_out(fn::model::model & m, int32_t cells, int32_t slots, bool kv_q8) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") || !cuda_ok(cudaDeviceSynchronize(), "drain")) {
            return false;
        }
    }
    fn::loader::release_state(m);
    fn::model::state_params sp;
    sp.n_cells             = cells;
    sp.ple_conv_kernel     = m.ple_conv_kernel;
    sp.ple_ngram           = m.ple_ngram;
    sp.single              = m.single;
    sp.kv_q8               = kv_q8;
    sp.win                 = m.win;  // TTTS1
    int64_t     host_bytes = 0;
    std::string err;
    if (!fn::loader::reserve_state(m, sp, slots, host_bytes, err) || !fn::loader::fill_ex1_grids(m, err)) {
        fprintf(stderr, "joint-kl: the %s reservation: %s\n", kv_q8 ? "q8" : "BF16", err.c_str());
        return false;
    }
    size_t f[2] = { 0, 0 }, tot = 0;
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m.device[c]);
        cudaMemGetInfo(&f[c], &tot);
    }
    printf(
        "joint-kl: the state laid out again in %s KV (the model's format now %s), %d slots of %d cells, the "
        "weights kept (no load): free %.3f / %.3f GiB\n",
        kv_q8 ? "q8" : "BF16", m.kv_q8 ? "q8" : "BF16", slots, cells, double(f[0]) / 1073741824.0,
        double(f[1]) / 1073741824.0);
    return m.kv_q8 == kv_q8;
}

int fnfl_joint_kl_on(const fnfl_target_args & a, std::unique_ptr<fn::model::model> m, int32_t cells, fnfl_split * sp) {
    std::error_code ec;
    std::filesystem::create_directories(a.out_dir, ec);
    if (ec) {
        fprintf(stderr, "joint-kl: cannot make %s: %s\n", a.out_dir.c_str(), ec.message().c_str());
        return 1;
    }
    std::vector<int32_t> src;
    const int32_t        need      = 3 * positions + max_lane_tokens;
    const bool           from_file = read_ids(a.joint_kl, src) && int32_t(src.size()) >= need;
    if (!from_file) {
        if (a.mode != "dry") {
            fprintf(stderr, "joint-kl: %s holds fewer than %d tokens\n", a.joint_kl.c_str(), need);
            return 1;
        }
        src.resize(size_t(need));
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = int32_t(1000 + (i * 7919u) % 150000u);  // dry: a seeded sequence of vocabulary IDs
        }
    }
    printf("joint-kl: %s; d = 1 .. 4 in turn%s\n", from_file ? a.joint_kl.c_str() : "a seeded sequence (dry)",
           a.mode == "dry" ? "; DRY: zero weights (plumbing only)" : "");
    float * pinned = nullptr;
    if (!cuda_ok(cudaMallocHost(&pinned, size_t(2) * max_rows * VC * 4), "pinned logits")) {
        return 1;
    }
    std::unique_ptr<float, decltype(&cudaFreeHost)> hold(pinned, cudaFreeHost);
    logit_store                                     lone, other;
    lone.init();
    other.init();
    bool all = true;
    for (int32_t d = 1; d <= max_rows - 1; ++d) {
        bool pass = false;
        if (sp != nullptr ? !kl_at_split(a, m, cells, src, d, *sp, lone, other, pass) :
                            !kl_at(a, m, cells, src, d, pinned, lone, other, pass)) {
            if (sp != nullptr) {
                sp->lead->bye();
            }
            return 1;
        }
        all = all && pass;
    }
    if (sp != nullptr) {
        sp->lead->bye();
    }
    printf("joint-kl: %s (d = 1 .. 4)\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}

int fnfl_joint_kl(const fnfl_target_args & a) {
    const int32_t cells = 65536; // rollback-test mu1's slots
    if (a.split_role != 0) {
        fprintf(stderr,
                "joint-kl: on the split it runs after rollback-test mu1 --split (--joint-kl DEEP), on its "
                "session\n");
        return 2;
    }
    fnfl_target_args aa = a;
    aa.n_slots = lanes;
    auto m = std::make_unique<md::model>();
    if (a.mode == "dry") {
        if (!fnfl_load_dry(aa, cells, *m)) {
            return 1;
        }
    } else {
        fn::loader::load_params lp;
        lp.model_path = a.model_path;
        lp.draft_path = a.draft_path;
        lp.ple_table = a.ple_table;
        lp.n_slots = lanes;
        fn::loader::run_load_report lr;
        std::string err;
        if (!fn::loader::load_for_run(lp, cells, *m, lr, err)) {
            fprintf(stderr, "joint-kl: load: %s\n", err.c_str());
            return 1;
        }
        printf("joint-kl: model loaded in %.2f s\n", lr.total_s);
    }
    return fnfl_joint_kl_on(a, std::move(m), cells);
}
