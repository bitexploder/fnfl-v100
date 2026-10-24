// SP1: fnfl-drive spec (fn/runtime/target_cmds.h): real sampled speculation on both cards. Each round: d draft steps
// (DR1's draft block; at T > 0 SM1's top-k pick, which retains q), the verify round (B = d + 1; SM1's accept step, the
// EOS and budget cut, SR1's commit of a + 1), then DR1's catch-up of the a + 1 committed rows.

#include "fn/kernels/head/head.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/samp/philox.h"
#include "fn/kernels/samp/samp.h"
#include "fn/kernels/samp/samp_ref.h"
#include "fn/model/draft.h"
#include "fn/model/model.h"
#include "fn/quant/formats.h"
#include "fn/runtime/json_check.h"
#include "fn/runtime/mx7_check.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/target.h"
#include "fn/runtime/target_cmds.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace rt = fn::runtime;
namespace md = fn::model;
namespace sp = fn::samp;

constexpr int32_t VC = fn::head::n_vocab_card; // 124,160
constexpr int32_t NV = fn::head::n_draft_card; // 32,768
constexpr int32_t V = sp::n_vocab;             // 248,320
constexpr int32_t eos_tokenizer = 248046;      // the tokenizer's EOS (TG1's generate report; not PLE's 248,044)
constexpr double tol = 2e-5;                   // SM1's listed CDF boundary (samp-test)
using clk = std::chrono::steady_clock;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "spec: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool fail_msg(std::string & err, const std::string & what) {
    err = what;
    return false;
}

// v[0..n) as "v0,v1,.." (sep between)
std::string list_of(const int32_t * v, int32_t n, const char * sep = ",") {
    std::string s;
    for (int32_t i = 0; i < n; ++i) {
        s += (i ? sep : "") + std::to_string(v[i]);
    }
    return s;
}

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

bool read_ids(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "spec: cannot read %s\n", path.c_str());
        return false;
    }
    out.clear();
    long v = -1;
    char ch = 0;
    while (f.get(ch)) {
        if (ch >= '0' && ch <= '9') {
            v = (v < 0 ? 0 : v) * 10 + (ch - '0');
        } else if (v >= 0) {
            out.push_back(static_cast<int32_t>(v));
            v = -1;
        }
    }
    if (v >= 0) {
        out.push_back(static_cast<int32_t>(v));
    }
    return !out.empty();
}

bool write_tokens(const std::string & path, const std::vector<int32_t> & t) {
    FILE * f = fopen(path.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "spec: cannot write %s\n", path.c_str());
        return false;
    }
    for (size_t i = 0; i < t.size(); ++i) {
        fprintf(f, "%s%d", i ? "," : "", t[i]);
    }
    fprintf(f, "\n");
    fclose(f);
    return true;
}

bool write_bin(const std::string & path, const void * p, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "spec: cannot write %s\n", path.c_str());
        return false;
    }
    const bool ok = fwrite(p, 1, n, f) == n;
    fclose(f);
    return ok;
}

uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

uint64_t hash_bytes(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, b + i, 8);
        h = mix(h ^ w);
    }
    for (; i < n; ++i) {
        h = mix(h ^ b[i]);
    }
    return mix(h ^ n);
}

template <typename T> bool get(std::vector<T> & v, const T * src, size_t n) {
    v.resize(n);
    // cudaMemcpyDefault: a source in device or (ME1: the host-backed KV, the PF1 checkpoints) pinned host memory
    return n == 0 || cuda_ok(cudaMemcpy(v.data(), src, n * sizeof(T), cudaMemcpyDefault), "download");
}

bool sync_cards(rt::target & t, md::model & m) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "sync")) {
            return false;
        }
    }
    // ME1: with the host-backed KV the KV classes are its host history, made whole for the reads that follow
    std::string err;
    if (!t.kv_flush(err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return false;
    }
    return true;
}

// C4e's state: rollback-test's seven target classes, then the draft's K and V of cells < draft_n() and the retained
// carry, both cards; SP2: the draft index's keys of cells < index_n() and summaries of its whole blocks
constexpr int32_t n_hash = 12;
const char * const hash_names[n_hash] = {"GDN states",  "GDN histories", "PLE history",      "K cache",
                                         "V cache",     "index keys",    "summaries",        "draft K",
                                         "draft V",     "draft carry",   "draft index keys", "draft index summaries"};
struct state_hash {
    int64_t n = 0, draft_n = 0, index_n = 0;
    uint64_t h[n_hash] = {};
    bool index_cards_equal = true; // SP2: both cards' draft index bitwise equal
};

// SP2: card c's draft index: the raw keys of cells < n and the summaries of blocks < n / 4 (one block's 128 values
// gathered from the tiled layout, as rollback-test's state hashes)
bool index_hash(rt::target & t, md::model & m, int32_t c, int64_t n, uint64_t & hk, uint64_t & hs) {
    const md::draft_card & d = t.draft(c);
    const int32_t blocks = fn::qsa::blocks_complete(static_cast<int32_t>(n));
    std::vector<uint16_t> u;
    std::vector<float> f;
    std::vector<float> one(size_t(fn::qsa::head_dim));
    if (d.idx_keys == nullptr) {
        return true; // a dense-only target holds no draft index
    }
    if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
        !get(u, static_cast<const uint16_t *>(d.idx_keys), size_t(n) * fn::qsa::head_dim) ||
        !get(f, static_cast<const float *>(d.idx_summ), fn::qsa::summary_bytes(blocks) / 4)) {
        return false;
    }
    hk = hash_bytes(hk, u.data(), u.size() * 2);
    for (int32_t b = 0; b < blocks; ++b) {
        for (int32_t e = 0; e < fn::qsa::head_dim; ++e) {
            one[size_t(e)] = f[size_t(b / 32) * 4096 + size_t(e / 4) * 128 + size_t(b % 32) * 4 + size_t(e % 4)];
        }
        hs = hash_bytes(hs, one.data(), one.size() * 4);
    }
    return true;
}

bool snapshot(rt::target & t, md::model & m, state_hash & s) {
    s = state_hash();
    s.n = t.slot().n;
    s.draft_n = t.draft_n();
    if (!sync_cards(t, m) || !fnfl_state_hashes(t, m, s.h)) {
        return false;
    }
    std::vector<uint16_t> u;
    std::vector<float> f;
    for (int32_t i = 7; i < n_hash; ++i) {
        s.h[i] = uint64_t(i) + 1;
    }
    for (int32_t c = 0; c < 2; ++c) {
        md::draft_card & d = t.draft(c);
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") || s.draft_n < 0 ||
            !get(u, d.k_cache, size_t(s.draft_n) * 256)) {
            return false;
        }
        s.h[7] = hash_bytes(s.h[7] ^ uint64_t(c), u.data(), u.size() * 2);
        if (!get(u, d.v_cache, size_t(s.draft_n) * 256)) {
            return false;
        }
        s.h[8] = hash_bytes(s.h[8] ^ uint64_t(c), u.data(), u.size() * 2);
        if (!get(f, d.carry_prev, size_t(4) * 1280)) {
            return false;
        }
        s.h[9] = hash_bytes(s.h[9] ^ uint64_t(c), f.data(), f.size() * 4);
    }
    // SP2: the draft index, each card's from the same seed (the cards hold identical copies)
    s.index_n = t.index_n();
    uint64_t hk[2] = {11, 11}, hs[2] = {12, 12};
    for (int32_t c = 0; c < 2; ++c) {
        if (!index_hash(t, m, c, s.index_n, hk[c], hs[c])) {
            return false;
        }
    }
    s.h[10] = hk[0];
    s.h[11] = hs[0];
    s.index_cards_equal = hk[0] == hk[1] && hs[0] == hs[1];
    return true;
}

// C4a: the runtime's prefill (fn/runtime/runtime.h): forced rounds at B = 4 (the last the remainder), the draft's
// catch-up after each
bool prefill(rt::target & t, const std::vector<int32_t> & prompt, double & secs, std::string & err) {
    const auto t0 = clk::now();
    const bool ok = rt::prefill(t, prompt, err);
    secs = since(t0);
    return ok;
}

struct spec_round {
    int64_t n_before = 0;
    int32_t kind = 0; // 0 bootstrap, 1 generate, 2 speculative (draft step, verify, catch-up)
    int32_t d = 0, a = 0, n_out = 0;
    int32_t prop[rt::max_spec_d] = {-1, -1, -1, -1};
    int32_t out[rt::max_step_tokens] = {-1, -1, -1, -1, -1};
    double t_done = 0; // seconds since the run's start
};

// hooks of a recorded run (C4d, C4e): after the draft step (proposals in r), after the round's catch-up
struct spec_hooks {
    std::function<bool(int32_t, const rt::round_result &)> after_draft;
    std::function<bool(int32_t, const spec_round &)> after_round;
};

// the runtime's steps (fn/runtime/runtime.h, SV1: the one implementation): the bootstrap, then speculative steps of d
// drafts (a generate round at a budget of 1) until max_tokens emitted, an EOS, max_rounds or the time limit; a failed
// step ends the run (FC-RD2's restore has run). through_eog (CHECK ONLY, SP2 Amendment 1, expires with SP2's
// closure): no end-of-generation ID, so the run continues past an EOS; the caller also sets the sampler's eos to -1
// (no verify cut), and the reference reads that from t.sampling()
bool spec_run(rt::target & t, int32_t d, int32_t max_tokens, int32_t max_rounds, double limit_s,
              std::vector<int32_t> & toks, std::vector<spec_round> & rounds, double & secs, bool & eos,
              std::string & err, const spec_hooks * hooks = nullptr, bool through_eog = false) {
    toks.clear();
    rounds.clear();
    eos = false;
    const std::vector<int32_t> eog = through_eog ? std::vector<int32_t>{} : std::vector<int32_t>{eos_tokenizer};
    int32_t n_spec = 0;
    const rt::after_draft_fn after_draft = [&](const rt::round_result & r) {
        return hooks == nullptr || !hooks->after_draft || hooks->after_draft(n_spec, r);
    };
    const auto t0 = clk::now();
    do {
        rt::step_info x;
        if (!rt::step(t, max_tokens - static_cast<int32_t>(toks.size()), d, eog, x, err, &after_draft)) {
            return false;
        }
        spec_round s;
        s.n_before = x.n_before;
        s.kind = x.kind;
        s.d = x.d;
        s.a = x.a;
        s.n_out = x.n_out;
        std::memcpy(s.prop, x.prop, sizeof(s.prop));
        std::memcpy(s.out, x.out, sizeof(s.out));
        s.t_done = since(t0);
        toks.insert(toks.end(), x.out, x.out + x.n_out);
        eos = x.end;
        rounds.push_back(s);
        if (x.kind != 0) {
            if (hooks != nullptr && hooks->after_round && !hooks->after_round(n_spec, s)) {
                return fail_msg(err, "after_round hook");
            }
            ++n_spec;
        }
    } while (!eos && static_cast<int32_t>(toks.size()) < max_tokens && n_spec < max_rounds && since(t0) < limit_s);
    secs = since(t0);
    return true;
}

// ---- C4d: the recorded rounds and the CPU reference ----

struct c4d_round {
    int64_t n_before = 0;
    int32_t d = 0, a = -1, n_out = 0;
    int32_t prop[rt::max_spec_d] = {-1, -1, -1, -1};
    int32_t out[rt::max_step_tokens] = {-1, -1, -1, -1, -1};
    std::vector<float> dl;    // [d][2][NV] the draft step's logits, each card's half of the subset
    std::vector<sp::qlist> q; // [d][2] the retained q lists, each card's
    std::vector<float> tl;    // [d + 1][V] the verify rows' target logits (card 0's shard, then card 1's)
};

bool record_draft(rt::target & t, md::model & m, int32_t d, c4d_round & x) {
    x.d = d;
    x.dl.assign(size_t(d) * 2 * NV, 0.0f);
    x.q.assign(size_t(d) * 2, sp::qlist());
    if (!sync_cards(t, m)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device")) {
            return false;
        }
        for (int32_t j = 0; j < d; ++j) {
            if (!cuda_ok(cudaMemcpy(&x.dl[(size_t(j) * 2 + size_t(c)) * NV], t.draft(c).logits + size_t(j) * NV,
                                    size_t(NV) * 4, cudaMemcpyDeviceToHost),
                         "draft logits") ||
                !cuda_ok(cudaMemcpy(&x.q[size_t(j) * 2 + size_t(c)], t.card(c).samp.q_list + j, sizeof(sp::qlist),
                                    cudaMemcpyDeviceToHost),
                         "q list")) {
                return false;
            }
        }
    }
    return true;
}

bool record_verify(rt::target & t, md::model & m, c4d_round & x) {
    x.tl.assign(size_t(x.d + 1) * V, 0.0f);
    if (!sync_cards(t, m)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device")) {
            return false;
        }
        for (int32_t r = 0; r <= x.d; ++r) {
            if (!cuda_ok(cudaMemcpy(&x.tl[size_t(r) * V + size_t(c) * VC], t.card(c).logits + size_t(r) * VC,
                                    size_t(VC) * 4, cudaMemcpyDeviceToHost),
                         "target logits")) {
                return false;
            }
        }
    }
    return true;
}

uint64_t uni(const sp::options & o, uint32_t pos, uint32_t purpose) {
    uint64_t u0 = 0, u1 = 0;
    sp::uniforms(o.seed, o.request, pos, purpose, u0, u1);
    return u0;
}

double draw_gap(const sp::ref_draw_out & rd) {
    return std::min(std::fabs(rd.target - rd.before), std::fabs(rd.target - rd.after));
}

struct c4d_result {
    int32_t picks = 0, pick_mism = 0, pick_listed = 0, q_bad = 0, q_cards_differ = 0;
    int32_t rounds = 0, acc_mism = 0, acc_listed = 0;
    int32_t by_a[rt::max_step_tokens] = {0, 0, 0, 0, 0};
    bool pass() const {
        return picks > 0 && rounds > 0 && pick_mism == pick_listed && q_bad == 0 && q_cards_differ == 0 &&
               acc_mism == acc_listed;
    }
};

// the producer: q and the proposal rebuilt from the real draft logits, the ID map and the settings; the acceptance:
// p, the accept test, the residual and the bonus (ref_verify), with FC-DEC3's EOS cut, under SM1's listing rule
c4d_result check_c4d(const sp::options & o, const std::vector<c4d_round> & rs, const std::vector<int32_t> ids[2],
                     FILE * log) {
    c4d_result res;
    std::string err;
    for (size_t i = 0; i < rs.size(); ++i) {
        const c4d_round & x = rs[i];
        std::vector<sp::ref_q> rq(size_t(x.d));
        std::vector<int32_t> xs(size_t(x.d));
        for (int32_t j = 0; j < x.d; ++j) {
            std::vector<float> sub(size_t(V), NAN); // the subset's logits, NaN elsewhere (never kept)
            for (int32_t c = 0; c < 2; ++c) {
                for (int32_t k = 0; k < NV; ++k) {
                    sub[size_t(ids[c][size_t(k)])] = x.dl[(size_t(j) * 2 + size_t(c)) * NV + size_t(k)];
                }
            }
            sp::ref_dist dq;
            const sp::qlist & q0 = x.q[size_t(j) * 2];
            const sp::qlist & q1 = x.q[size_t(j) * 2 + 1];
            ++res.picks;
            res.q_cards_differ += memcmp(&q0, &q1, sizeof(sp::qlist)) != 0 ? 1 : 0;
            if (!sp::ref_distribution(o, sub.data(), V, VC, dq, err)) {
                ++res.q_bad;
                fprintf(log, "c4d round %zu step %d: reference: %s\n", i, j, err.c_str());
                continue;
            }
            bool qok = q0.n == static_cast<int32_t>(dq.id.size());
            double qmax = 0;
            for (int32_t k = 0; qok && k < q0.n; ++k) {
                qok = q0.id[k] == dq.id[size_t(k)];
                qmax = std::max(qmax, std::fabs(double(q0.q[k]) - dq.prob[size_t(k)]));
            }
            qok = qok && qmax <= 1e-5;
            res.q_bad += qok ? 0 : 1;
            const uint32_t pos = static_cast<uint32_t>(x.n_before + j + 1);
            const sp::ref_draw_out rd = sp::ref_draw(dq, uni(o, pos, sp::p_draft), 0);
            const bool same = rd.token == x.prop[j];
            if (!same) {
                ++res.pick_mism;
                res.pick_listed += draw_gap(rd) <= tol ? 1 : 0;
            }
            fprintf(
                log,
                "c4d round %zu step %d pos %u: kept %d (CPU %zu), q max |diff| %.3g %s; proposal GPU %d CPU %d %s\n", i,
                j, pos, q0.n, dq.id.size(), qmax, qok ? "ok" : "BAD", x.prop[j], rd.token,
                same ? "same" : (draw_gap(rd) <= tol ? "listed" : "UNLISTED"));
            xs[size_t(j)] = x.prop[j];
            rq[size_t(j)].id.assign(q0.id, q0.id + q0.n);
            rq[size_t(j)].q.assign(q0.q, q0.q + q0.n);
        }
        std::vector<sp::ref_dist> p(size_t(x.d + 1));
        bool ok = true;
        for (int32_t t = 0; t <= x.d && ok; ++t) {
            ok = sp::ref_distribution(o, &x.tl[size_t(t) * V], V, VC, p[size_t(t)], err);
        }
        if (!ok) {
            ++res.acc_mism;
            fprintf(log, "c4d round %zu: target reference: %s\n", i, err.c_str());
            continue;
        }
        const uint32_t pos0 = static_cast<uint32_t>(x.n_before + 1);
        sp::ref_verify_out r = sp::ref_verify(o, p, xs, rq, pos0);
        for (int32_t j = 0; j < r.a; ++j) {
            if (r.out[size_t(j)] == o.eos) { // FC-DEC3's cut, as the device's verdict
                r.a = j;
                r.out.resize(size_t(j + 1));
                break;
            }
        }
        ++res.rounds;
        if (x.a >= 0 && x.a <= rt::max_spec_d) {
            res.by_a[x.a]++;
        }
        bool match = r.a == x.a;
        for (int32_t t = 0; t <= r.a && match; ++t) {
            match = x.out[t] == r.out[size_t(t)];
        }
        const char * how = "same";
        if (!match) {
            ++res.acc_mism;
            bool listed = false;
            if (r.a != x.a) {
                // the accept test at the first differing row: u * q(x) against p(x)
                const int32_t t = std::min(r.a, std::max(0, x.a));
                const double ua = sp::unit_double(uni(o, pos0 + uint32_t(t), sp::p_accept));
                const double px = t < x.d ? sp::ref_prob(p[size_t(t)], xs[size_t(t)]) : 0;
                const double qx = t < x.d ? rq[size_t(t)].at(xs[size_t(t)]) : 0;
                listed = qx > 0 && std::fabs(ua * qx - px) <= tol * qx + tol * px;
            } else if (x.a < x.d) {
                // the residual draw at the rejected row
                const sp::ref_resid rr = sp::ref_make_resid(p[size_t(x.a)], rq[size_t(x.a)]);
                const sp::ref_draw_out rd =
                    sp::ref_residual_draw(p[size_t(x.a)], rr, uni(o, pos0 + uint32_t(x.a), sp::p_resid_shard), 0);
                listed = draw_gap(rd) <= tol;
            } else {
                // the bonus from the last row's p
                const sp::ref_draw_out rd = sp::ref_draw(p[size_t(x.d)], uni(o, pos0 + uint32_t(x.d), sp::p_shard), 0);
                listed = draw_gap(rd) <= tol;
            }
            res.acc_listed += listed ? 1 : 0;
            how = listed ? "listed" : "UNLISTED";
        }
        fprintf(log, "c4d round %zu N %lld: GPU a %d out", i, static_cast<long long>(x.n_before), x.a);
        for (int32_t t = 0; t <= x.a; ++t) {
            fprintf(log, " %d", x.out[t]);
        }
        fprintf(log, "; CPU a %d out", r.a);
        for (int32_t v : r.out) {
            fprintf(log, " %d", v);
        }
        fprintf(log, " %s\n", how);
    }
    return res;
}

std::string sm_clocks() {
    FILE * p = popen("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null", "r");
    if (p == nullptr) {
        return "unknown";
    }
    char buf[128] = {};
    std::string s;
    while (fgets(buf, sizeof(buf), p) != nullptr) {
        for (char * c = buf; *c != 0; ++c) {
            if (*c == '\n') {
                *c = ' ';
            }
        }
        s += buf;
    }
    pclose(p);
    return s;
}

} // namespace

int fnfl_spec(const fnfl_target_args & a) {
    return fnfl_spec_on(a, nullptr);
}

int fnfl_spec_on(const fnfl_target_args & a, md::model * loaded) {
    const bool dry = a.mode == "dry";
    std::vector<int32_t> prompt;
    if (!read_ids(a.ids, prompt)) {
        return 1;
    }
    if (dry) {
        prompt.resize(std::min<size_t>(prompt.size(), 13)); // forced rounds of 4, 4, 4, 1
    }
    // dry: 15 and 41 tokens end on a d = 1 verify (B = 2) and a generate round (one token of budget left)
    const int32_t n_b = dry ? 15 : 64, n_d = dry ? 2 : 8, n_e = dry ? 4 : 32, n_f = dry ? 16 : 128;
    const int32_t c_max = dry ? 41 : 16384;
    const double c_limit = dry ? 30.0 : a.decode_s;
    const uint64_t seed = a.sample ? a.sampling.seed : 20261013ull;
    const int32_t d_spec = a.spec_d; // SP2: --spec-d, 1..4 (default 2)
    // the T = 1 settings: the command line's --temp, --top-k, --top-p and --min-p (TK1)
    const sp::options & given = a.run_settings.front().sampling;
    sp::options s1;
    s1.temp = given.temp;
    s1.top_k = given.top_k;
    s1.top_p = given.top_p;
    s1.min_p = given.min_p;
    if (s1.temp <= 0.0f || sp::path_of(s1) != sp::path::topk) {
        fprintf(stderr, "spec: the T = 1 segments need T > 0 on the top-k path (top-k 1 .. 256)\n");
        return 1;
    }
    printf("spec: settings: prompt %s (%zu tokens); %s; d %d; T = 0 diagnostic %d tokens; T = 1 segments: T %.2f, "
           "top-k %d, top-p %.2f, min-p %.2f, penalties %s, seed %llu, request 0; EOS %d (the tokenizer's); C4d %d "
           "rounds, C4c up to %.0f s or %d tokens, C4e %d rounds, C4f %d tokens\n",
           a.ids.c_str(), prompt.size(), dry ? "dry (zero weights)" : "the model", d_spec, n_b, s1.temp, s1.top_k,
           s1.top_p, s1.min_p, s1.penalties() ? "on" : "off", static_cast<unsigned long long>(seed), eos_tokenizer, n_d,
           c_limit, c_max, n_e, n_f);
    rt::draft_options dopt;
    if (!fnfl_draft_settings(a, true, "spec", dopt)) {
        return 1;
    }
    const int32_t cells = dry ? 8192 : a.n_cells;
    // KV8: a model loaded by the caller (model-test's load 4) is used as it is, its state as reserved
    std::unique_ptr<md::model> own;
    md::model * m = loaded;
    if (m == nullptr) {
        own = std::make_unique<md::model>();
        m = own.get();
        if (!(dry ? fnfl_load_dry(a, cells, *m) : fnfl_load(a, cells, *m, "spec"))) {
            return 1;
        }
    } else {
        printf("spec: on the caller's loaded model (KV format %s), no load of its own\n", m->kv_q8 ? "q8" : "bf16");
    }
    FILE * log = fopen((a.out_dir + "/spec-checks.log").c_str(), "w");
    if (log == nullptr) {
        fprintf(stderr, "spec: cannot write %s/spec-checks.log\n", a.out_dir.c_str());
        return 1;
    }
    std::string err;
    bool all = true;
    const auto verdict = [&](const char * name, bool ok, const std::string & what) {
        printf("spec: %s: %s: %s\n", name, ok ? "PASS" : "FAIL", what.c_str());
        fprintf(log, "%s: %s: %s\n", name, ok ? "PASS" : "FAIL", what.c_str());
        fflush(stdout);
        fflush(log);
        all = all && ok;
    };
    const auto closing = [&](rt::target & t, const char * name) {
        bool checked = false;
        const bool ok = t.close_check(checked, err);
        verdict(name, ok && checked,
                ok ? (checked ? "the last decision's hash checked on both cards" : "NOT CHECKED") : err);
    };

    // item 5: T > 0 off the top-k path (top-k 0: the full path) is refused at init, before any graph is captured
    {
        rt::target_options orf;
        orf.model_path = a.model_path;
        orf.sample = true;
        orf.sampling.temp = 1.0f;
        orf.sampling.top_k = 0;
        orf.draft = dopt;
        rt::target tr;
        std::string e2;
        const bool refused = !tr.init(*m, orf, e2) && e2.find("refused") != std::string::npos;
        verdict("refusal (T = 1, top-k 0)", refused, e2);
    }

    // ---- T = 0: the target with the sampler's greedy path ----
    rt::target_options oa;
    oa.model_path = a.model_path;
    oa.timeout_s = dry ? 30.0 : 60.0;
    oa.sample = true;
    oa.sampling.temp = 0.0f;
    oa.sampling.eos = eos_tokenizer;
    oa.draft = dopt;
    auto t = std::make_unique<rt::target>();
    if (!t->init(*m, oa, err)) {
        fprintf(stderr, "spec: init (T = 0): %s\n", err.c_str());
        return 1;
    }
    printf("spec: T = 0 target: %d graph captures (path %s)\n", t->captures(),
           sp::path_of(oa.sampling) == sp::path::greedy ? "greedy" : "other");
    double pre_s = 0;
    const bool pre_ok = prefill(*t, prompt, pre_s, err);
    verdict("C4a prefill", pre_ok,
            pre_ok ? std::to_string(prompt.size()) +
                         " prompt tokens as forced rounds at B = 4 with the catch-up after "
                         "each: N " +
                         std::to_string(t->slot().n) + ", draft N " + std::to_string(t->draft_n()) + ", " +
                         std::to_string(pre_s) + " s"
                   : err);
    if (!pre_ok) {
        return 1;
    }
    // C4b: generate rounds (TG1's loop: the bootstrap, then B = 1) against spec at T = 0
    std::vector<int32_t> gen;
    {
        rt::round_result r;
        bool ok = t->bootstrap(r, err);
        gen.push_back(r.out[0]);
        for (int32_t i = 1; ok && i < n_b; ++i) {
            ok = t->generate(r, err);
            gen.push_back(r.out[0]);
        }
        if (!ok) {
            verdict("C4b generate", false, err);
            return 1;
        }
        closing(*t, "C4b generate closing check");
    }
    std::vector<int32_t> s0;
    std::vector<spec_round> rounds;
    double secs = 0;
    bool eos = false;
    bool ok = prefill(*t, prompt, pre_s, err) && spec_run(*t, d_spec, n_b, 1 << 30, 1e9, s0, rounds, secs, eos, err);
    if (!ok) {
        verdict("C4b spec", false, err);
        return 1;
    }
    closing(*t, "C4b spec closing check");
    {
        // equal tokens; a spec run cut at an EOS ends with it, and generate (EOS not a stop) has it there too
        bool same = s0.size() <= gen.size() && (s0.size() == size_t(n_b) || eos);
        for (size_t i = 0; same && i < s0.size(); ++i) {
            same = s0[i] == gen[i];
        }
        int32_t acc = 0, spec_rounds = 0;
        for (const spec_round & s : rounds) {
            if (s.kind == 2) {
                acc += s.a;
                ++spec_rounds;
            }
        }
        verdict("C4b T = 0 diagnostic", same,
                std::to_string(s0.size()) + " spec tokens (" + std::to_string(spec_rounds) + " speculative rounds, " +
                    std::to_string(acc) + " accepted drafts" + (eos ? ", ended at EOS" : "") + ") against " +
                    std::to_string(gen.size()) + " generate tokens: " + (same ? "equal" : "DIFFERENT"));
        write_tokens(a.out_dir + "/c4b-generate-tokens.txt", gen);
        write_tokens(a.out_dir + "/c4b-spec-tokens.txt", s0);
    }
    t.reset(); // the T = 1 target's graphs replace these (the model's buffers are reused)

    // ---- T = 1 ----
    rt::target_options ob = oa;
    ob.sampling = s1;
    ob.sampling.seed = seed;
    ob.sampling.eos = eos_tokenizer;
    t = std::make_unique<rt::target>();
    if (!t->init(*m, ob, err)) {
        fprintf(stderr, "spec: init (T = 1): %s\n", err.c_str());
        return 1;
    }
    printf("spec: T = 1 target: %d graph captures (path %s, penalties %s)\n", t->captures(),
           sp::path_of(ob.sampling) == sp::path::topk ? "top-k" : "other", ob.sampling.penalties() ? "on" : "off");
    // C4d: 8 recorded rounds (untimed)
    {
        std::vector<int32_t> ids[2];
        for (int32_t c = 0; c < 2; ++c) {
            ids[c].resize(size_t(NV));
            cudaSetDevice(m->device[c]);
            if (!cuda_ok(cudaMemcpy(ids[c].data(), t->draft(c).head_ids, size_t(NV) * 4, cudaMemcpyDeviceToHost),
                         "ID map")) {
                return 1;
            }
        }
        std::vector<c4d_round> rec;
        spec_hooks hk;
        hk.after_draft = [&](int32_t, const rt::round_result & r) {
            rec.emplace_back();
            rec.back().n_before = t->slot().n;
            for (int32_t j = 0; j < r.rows; ++j) {
                rec.back().prop[j] = r.argmax[0][j];
            }
            return record_draft(*t, *m, r.rows, rec.back());
        };
        hk.after_round = [&](int32_t, const spec_round & s) {
            c4d_round & x = rec.back();
            x.a = s.a;
            x.n_out = s.n_out;
            std::memcpy(x.out, s.out, sizeof(x.out));
            return record_verify(*t, *m, x);
        };
        std::vector<int32_t> toks;
        ok = prefill(*t, prompt, pre_s, err) &&
             spec_run(*t, d_spec, 1 << 20, n_d, 1e9, toks, rounds, secs, eos, err, &hk);
        if (!ok) {
            verdict("C4d", false, err);
            return 1;
        }
        closing(*t, "C4d closing check");
        // record_verify ran after the catch-up: the target logits are the verify round's (the catch-up runs no head)
        const c4d_result cr = check_c4d(ob.sampling, rec, ids, log);
        char what[512];
        snprintf(what, sizeof(what),
                 "%d rounds (a = 0..%d: %s), %d draft picks: q lists differing from the reference %d, cards' q "
                 "lists differ %d, proposals same-uniform mismatches %d (listed %d); acceptance mismatches %d "
                 "(listed %d)",
                 cr.rounds, d_spec, list_of(cr.by_a, d_spec + 1, "/").c_str(), cr.picks, cr.q_bad, cr.q_cards_differ,
                 cr.pick_mism, cr.pick_listed, cr.acc_mism, cr.acc_listed);
        verdict("C4d producer and acceptance", cr.pass(), what);
        // the raw rows: per round the draft logits, q lists and target logits; the keys and decisions in the TSV
        const std::string dir = a.out_dir + "/c4d";
        FILE * kf = fopen((dir + "/rounds.tsv").c_str(), "w");
        if (kf == nullptr || !write_bin(dir + "/idmap-card0.i32", ids[0].data(), ids[0].size() * 4) ||
            !write_bin(dir + "/idmap-card1.i32", ids[1].data(), ids[1].size() * 4)) {
            fprintf(stderr, "spec: cannot write %s\n", dir.c_str());
            return 1;
        }
        fprintf(kf, "round\tN_before\td\tprop\ta\tout\tseed\trequest\tdraft_positions(p_draft=6)\t"
                    "verify_positions(p_accept=3,p_shard=1,p_resid_shard=4)\n");
        for (size_t i = 0; i < rec.size(); ++i) {
            const c4d_round & x = rec[i];
            const std::string b = dir + "/round" + std::to_string(i);
            write_bin(b + "-draft-logits.f32", x.dl.data(), x.dl.size() * 4);
            write_bin(b + "-qlists.bin", x.q.data(), x.q.size() * sizeof(sp::qlist));
            write_bin(b + "-target-logits.f32", x.tl.data(), x.tl.size() * 4);
            fprintf(kf, "%zu\t%lld\t%d\t%s\t%d\t", i, static_cast<long long>(x.n_before), x.d,
                    list_of(x.prop, x.d).c_str(), x.a);
            for (int32_t j = 0; j <= x.a; ++j) {
                fprintf(kf, "%s%d", j ? "," : "", x.out[j]);
            }
            fprintf(kf, "\t%llu\t0\t%lld..%lld\t%lld..%lld\n", static_cast<unsigned long long>(seed),
                    static_cast<long long>(x.n_before + 1), static_cast<long long>(x.n_before + x.d),
                    static_cast<long long>(x.n_before + 1), static_cast<long long>(x.n_before + x.d + 1));
        }
        fclose(kf);
    }
    // C4c: the timed run (counters only)
    std::vector<int32_t> c_toks;
    std::vector<spec_round> c_rounds;
    double c_secs = 0;
    bool c_eos = false;
    int64_t c_n0 = 0, c_n1 = 0;
    std::string clk0, clk1;
    {
        ok = prefill(*t, prompt, pre_s, err);
        c_n0 = t->slot().n;
        clk0 = sm_clocks();
        ok = ok && spec_run(*t, d_spec, c_max, 1 << 30, c_limit, c_toks, c_rounds, c_secs, c_eos, err);
        clk1 = sm_clocks();
        c_n1 = t->slot().n;
        if (!ok) {
            verdict("C4c", false, err);
            return 1;
        }
        closing(*t, "C4c closing check");
        int32_t n_spec = 0, acc = 0, hist[rt::max_step_tokens] = {}, tok_hist[rt::max_step_tokens + 1] = {};
        for (const spec_round & s : c_rounds) {
            if (s.kind == 2) {
                ++n_spec;
                acc += s.a;
                hist[std::min(rt::max_spec_d, s.a)]++;
                tok_hist[std::min(rt::max_step_tokens, s.n_out)]++;
            }
        }
        char what[768];
        snprintf(what, sizeof(what),
                 "%zu tokens in %.3f s = %.2f tokens/s (measured, bootstrap included); %d speculative rounds, %.3f "
                 "tokens a round (n_out 1..%d: %s), %.3f accepted drafts a round (a = 0..%d: %s); depth "
                 "%lld -> %lld; %s; SM clocks before [%s] after [%s]; every round's cards agreed and error words were "
                 "clear (a failed round would have ended the run)",
                 c_toks.size(), c_secs, double(c_toks.size()) / c_secs, n_spec,
                 n_spec ? double(acc + n_spec) / n_spec : 0.0, d_spec + 1,
                 list_of(tok_hist + 1, d_spec + 1, "/").c_str(), n_spec ? double(acc) / n_spec : 0.0, d_spec,
                 list_of(hist, d_spec + 1, "/").c_str(), static_cast<long long>(c_n0),
                 static_cast<long long>(c_n1), c_eos ? "ended at EOS" : "ended at the time or token limit",
                 clk0.c_str(), clk1.c_str());
        verdict("C4c T = 1 timed run", true, what);
        write_tokens(a.out_dir + "/c4c-tokens.txt", c_toks);
        FILE * f = fopen((a.out_dir + "/c4c-rounds.tsv").c_str(), "w");
        if (f != nullptr) {
            fprintf(f, "round\tkind\tN_before\td\tprop\ta\tn_out\tout\tt_done_s\n");
            for (size_t i = 0; i < c_rounds.size(); ++i) {
                const spec_round & s = c_rounds[i];
                fprintf(f, "%zu\t%s\t%lld\t%d\t%s\t%d\t%d\t", i,
                        s.kind == 0   ? "bootstrap"
                        : s.kind == 1 ? "generate"
                                      : "spec",
                        static_cast<long long>(s.n_before), s.d, list_of(s.prop, s.d).c_str(), s.a, s.n_out);
                for (int32_t j = 0; j < s.n_out; ++j) {
                    fprintf(f, "%s%d", j ? "," : "", s.out[j]);
                }
                fprintf(f, "\t%.6f\n", s.t_done);
            }
            fclose(f);
        }
    }
    // C4e: 32 rounds against a clean forced replay of the accepted tokens, bitwise
    {
        std::vector<state_hash> got;
        spec_hooks hk;
        hk.after_round = [&](int32_t, const spec_round &) {
            got.emplace_back();
            return snapshot(*t, *m, got.back());
        };
        std::vector<int32_t> toks;
        std::vector<spec_round> e_rounds;
        ok = prefill(*t, prompt, pre_s, err) &&
             spec_run(*t, d_spec, 1 << 20, n_e, 1e9, toks, e_rounds, secs, eos, err, &hk);
        if (!ok) {
            verdict("C4e", false, err);
            return 1;
        }
        closing(*t, "C4e closing check");
        const std::vector<int32_t> ledger = t->slot().tok;
        const int64_t n_final = t->slot().n;
        // the replay: the prompt as C4a, then each round's committed rows as one forced round and its catch-up
        ok = t->reset(err);
        t->set_prompt(std::vector<int32_t>(ledger.begin(), ledger.begin() + n_final));
        rt::round_result r;
        while (ok && t->slot().n < static_cast<int64_t>(prompt.size())) {
            const int32_t B = static_cast<int32_t>(std::min<int64_t>(4, int64_t(prompt.size()) - t->slot().n));
            ok = t->forced(B, r, err) && t->draft_catchup(B, r, err);
        }
        int32_t equal = 0, unequal = 0, rejected = 0;
        std::string rej_rows, bad;
        size_t gi = 0;
        for (size_t i = 1; ok && i < e_rounds.size(); ++i) {
            const spec_round & s = e_rounds[i];
            const int32_t rows = s.n_out;
            if (s.kind == 2 && s.a < s.d) {
                ++rejected;
                rej_rows += " " + std::to_string(i - 1) + ":row" + std::to_string(s.a);
            }
            if (t->slot().n + rows > n_final) {
                break; // the last round's last token is pending (no state); its rows end at n_final
            }
            ok = t->forced(rows, r, err) && t->draft_catchup(rows, r, err);
            state_hash h;
            ok = ok && gi < got.size() && snapshot(*t, *m, h);
            if (!ok) {
                break;
            }
            const state_hash & g = got[gi++];
            bool same = g.n == h.n && g.draft_n == h.draft_n && g.index_n == h.index_n && g.index_cards_equal &&
                        h.index_cards_equal;
            for (int32_t k = 0; k < n_hash; ++k) {
                if (g.h[k] != h.h[k]) {
                    same = false;
                    bad += std::string(" round ") + std::to_string(i - 1) + ":" + hash_names[k];
                }
            }
            (same ? equal : unequal)++;
            fprintf(log, "c4e round %zu: N %lld draft N %lld a %d: %s\n", i - 1, static_cast<long long>(h.n),
                    static_cast<long long>(h.draft_n), s.a, same ? "bitwise equal" : "DIFFERENT");
        }
        if (!ok) {
            verdict("C4e replay", false, err);
            return 1;
        }
        // zero weights (dry) give p = q, so every draft is accepted: no rejection coverage is possible there
        const bool pass = unequal == 0 && equal == static_cast<int32_t>(got.size()) && (rejected > 0 || dry);
        verdict("C4e rejection replay", pass,
                std::to_string(equal) + " of " + std::to_string(got.size()) +
                    " rounds' states bitwise equal to the clean forced replay (GDN, histories, PLE, K, V, index keys, "
                    "summaries, draft K and V, retained carry, draft index)" +
                    (unequal ? "; differing:" + bad : std::string()) + "; rounds with a rejection " +
                    std::to_string(rejected) + " (round:row" + (rej_rows.empty() ? " none" : rej_rows) + ")" +
                    (rejected == 0 ? (dry ? " (zero weights: p = q, information)" : " -- MISSING COVERAGE, not a pass")
                                   : ""));
    }
    // C4f: determinism against C4c's prefix
    {
        std::vector<int32_t> toks;
        ok = prefill(*t, prompt, pre_s, err) && spec_run(*t, d_spec, n_f, 1 << 30, 1e9, toks, rounds, secs, eos, err);
        if (!ok) {
            verdict("C4f", false, err);
            return 1;
        }
        closing(*t, "C4f closing check");
        bool same = toks.size() <= c_toks.size();
        for (size_t i = 0; same && i < toks.size(); ++i) {
            same = toks[i] == c_toks[i];
        }
        // an EOS inside the segment is at the same position (the prefix covers it); a stop at n_f is not compared
        // with the long run's later stop
        same = same && (eos ? toks.size() == c_toks.size() || c_toks[toks.size() - 1] == eos_tokenizer : true);
        verdict("C4f determinism", same,
                std::to_string(toks.size()) + " tokens from a fresh prefill and the same seed against C4c's prefix: " +
                    (same ? "equal" : "DIFFERENT") + (eos ? " (ended at EOS)" : ""));
        write_tokens(a.out_dir + "/c4f-tokens.txt", toks);
    }
    t.reset();
    fclose(log);
    printf("spec: %s\n", all ? "PASS (the retained checks run here)" : "FAILED");
    return all ? 0 : 1;
}

namespace {

double median_of(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

// run-dry's prefix paths with zero weights: a cut within the last step then a pure extension, a deep truncate then the
// re-prefill from token 0, and the switch to the other decision variant; the history checked after every call
bool run_dry_paths(rt::engine & e, const std::vector<int32_t> & prompt, int32_t top_k, FILE * log) {
    int32_t out[rt::max_step_tokens] = {};
    int32_t n = 0, pending = -1, np = 0;
    bool end = false;
    std::vector<int32_t> held, h;
    bool ok = true;
    const auto expect = [&](const char * what) {
        int32_t ep = -1;
        e.history(0, h, ep);
        const bool same = h == held && ep == pending;
        fprintf(log, "run-dry: %s: history %zu tokens (expected %zu), pending %d (expected %d): %s\n", what, h.size(),
                held.size(), ep, pending, same ? "OK" : "MISMATCH");
        printf("spec run-dry: %s: history %zu tokens, pending %d: %s\n", what, h.size(), ep, same ? "OK" : "MISMATCH");
        ok = ok && same;
    };
    const auto steps = [&](int32_t k) {
        for (int32_t i = 0; i < k && ok; ++i) {
            if (pending >= 0) {
                held.push_back(pending);
            }
            ok = e.step(0, out, n, end) == rt::run_status::ok && n >= 1;
            held.insert(held.end(), out, out + n - 1);
            pending = out[n - 1];
        }
    };
    rt::request_options o;
    o.sampling.temp = 0.0f;
    o.max_tokens = 64;
    ok = e.set_sampling(0, o) == rt::run_status::ok && e.truncate(0, 0) == rt::run_status::ok &&
         e.prefill(0, prompt.data(), static_cast<int32_t>(prompt.size()), 0, np) == rt::run_status::ok;
    held = prompt;
    pending = -1;
    expect("prefill");
    steps(4);
    expect("4 steps (T = 0)");
    // within the last step: keep one fewer token, then extend by two (a pure extension, no re-prefill)
    const int32_t keep = static_cast<int32_t>(held.size()) - 1;
    ok = ok && e.truncate(0, keep) == rt::run_status::ok;
    held.resize(size_t(keep));
    pending = -1;
    expect("truncate within the last step");
    const int32_t more[2] = {11, 12};
    ok = ok && e.prefill(0, more, 2, keep, np) == rt::run_status::ok && np == 2;
    held.insert(held.end(), more, more + 2);
    expect("extension");
    steps(3);
    expect("3 steps");
    // deep: the kept tokens re-prefilled from token 0
    ok = ok && e.truncate(0, 5) == rt::run_status::ok;
    held.resize(5);
    pending = -1;
    expect("deep truncate");
    ok = ok && e.prefill(0, more, 2, 5, np) == rt::run_status::ok && np == 7;
    held.insert(held.end(), more, more + 2);
    expect("re-prefill from token 0");
    // the other variant (T > 0, top-k) on the same slot, then back to T = 0
    o.sampling.temp = 1.0f;
    o.sampling.top_k = top_k;
    o.sampling.seed = 7;
    ok = ok && e.set_sampling(0, o) == rt::run_status::ok;
    steps(3);
    expect("3 steps at T = 1 (the other variant)");
    o.sampling.temp = 0.0f;
    const int32_t n_held = static_cast<int32_t>(held.size());
    ok = ok && e.set_sampling(0, o) == rt::run_status::ok && e.truncate(0, n_held) == rt::run_status::ok &&
         e.prefill(0, more, 1, n_held, np) == rt::run_status::ok && np == 1;
    held.push_back(more[0]);
    pending = -1;
    expect("back to T = 0, extended");
    steps(2);
    expect("2 steps");
    // a request that ends in a generate round (a budget of 1 after the bootstrap), then a pure extension
    o.max_tokens = 2;
    const int32_t n_held2 = static_cast<int32_t>(held.size());
    ok = ok && e.set_sampling(0, o) == rt::run_status::ok && e.truncate(0, n_held2) == rt::run_status::ok &&
         e.prefill(0, more, 1, n_held2, np) == rt::run_status::ok && np == 1;
    held.push_back(more[0]);
    pending = -1;
    steps(2);
    expect("max_tokens 2: the bootstrap and a generate round");
    const int32_t n_held3 = static_cast<int32_t>(held.size());
    ok = ok && e.truncate(0, n_held3) == rt::run_status::ok &&
         e.prefill(0, more + 1, 1, n_held3, np) == rt::run_status::ok && np == 1;
    held.push_back(more[1]);
    pending = -1;
    expect("a pure extension after the generate round");
    if (!ok) {
        printf("spec run-dry: FAILED (%s)\n", e.error().c_str());
    }
    return ok;
}

} // namespace

bool fnfl_spec_request(rt::engine &                 e,
                       const std::vector<int32_t> & prompt,
                       int32_t                      n_reuse,
                       const rt::request_options &  o,
                       fnfl_request_out &           out) {
    out              = fnfl_request_out();
    int32_t    n_pre = 0;
    const auto tp    = clk::now();
    if (n_reuse < 0 || n_reuse >= static_cast<int32_t>(prompt.size()) || e.set_sampling(0, o) != rt::run_status::ok ||
        e.truncate(0, n_reuse) != rt::run_status::ok ||
        e.prefill(0, prompt.data() + n_reuse, static_cast<int32_t>(prompt.size()) - n_reuse, n_reuse, n_pre) !=
            rt::run_status::ok) {
        return false;
    }
    out.pre_s                           = since(tp);
    out.n_processed                     = n_pre;
    int32_t    buf[rt::max_step_tokens] = {};
    int32_t    n                        = 0;
    bool       end                      = false;
    const auto t0                       = clk::now();
    while (!end && static_cast<int32_t>(out.toks.size()) < o.max_tokens) {
        if (e.step(0, buf, n, end) != rt::run_status::ok) {
            return false;
        }
        out.toks.insert(out.toks.end(), buf, buf + n);
    }
    out.gen_s = since(t0);  // after the prefill to the last step's return, as the server's timing
    return true;
}

namespace {

// ME1: the device's and the process's memory now: the device's used bytes, the host's resident and peak resident
// bytes, and the page-locked bytes the cache holds (its backing and buffers)
std::string me1_memory(const fn::runtime::expert_cache & x) {
    std::string dev;
    for (int32_t d = 0; d < x.devices(); ++d) {
        size_t fr = 0, tot = 0;
        int32_t id = 0;
        cudaGetDevice(&id);
        cudaSetDevice(x.device_id(d));
        cudaMemGetInfo(&fr, &tot);
        cudaSetDevice(id);
        char v[160];
        snprintf(v, sizeof(v), "%sdevice %d used %.3f of %.3f GiB (cache %.3f, its pool %.3f)", d == 0 ? "" : "; ",
                 x.device_id(d), double(tot - fr) / 1073741824.0, double(tot) / 1073741824.0,
                 double(x.device_bytes(d)) / 1073741824.0, double(x.pool_bytes()) / 1073741824.0);
        dev += v;
    }
    int64_t rss = 0, hwm = 0;
    if (FILE * f = fopen("/proc/self/status", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f) != nullptr) {
            long long v = 0;
            if (sscanf(line, "VmRSS: %lld kB", &v) == 1) {
                rss = v << 10;
            } else if (sscanf(line, "VmHWM: %lld kB", &v) == 1) {
                hwm = v << 10;
            }
        }
        fclose(f);
    }
    const double g = 1073741824.0;
    char b[768];
    snprintf(b, sizeof(b),
             "memory: %s; host RSS %.3f GiB, peak %.3f GiB; cache page-locked %.3f GiB (backing %.3f, buffers %.3f)",
             dev.c_str(), double(rss) / g, double(hwm) / g, double(x.backing_bytes() + x.pinned_bytes()) / g,
             double(x.backing_bytes()) / g, double(x.pinned_bytes()) / g);
    return b;
}

// ME1's model check (--ex-check PROMPT): the same T = 0 request with every decode miss on the CPU, every miss a GPU
// copy (both from an empty cache, no admission), then the normal policy from an empty cache and again warm: the
// tokens bitwise equal (the serving placement does not change the answer); the first run's tokens against
// --ex-ref's (an earlier run's tokens file, the two-card resident engine's; information, not a gate)
// ME1 amendment 3: the target's logits at the first token k where the T = 0 run (own) and --ex-ref's (ref) differ.
// The same T = 0 request stopped after its k equal tokens (the history the two runs share, made by the same rounds),
// then one generate round of the pending token (B = 1): its decision must be own again; printed: both tokens' logits,
// their difference and the row's two largest logits; the row ([248,320] F32) written to OUT/me1-divergence-logits.f32.
// The slot is re-prefilled by the next request.
bool me1_divergence(rt::engine & e, const std::vector<int32_t> & prompt, size_t k, int32_t own, int32_t ref,
                    const fnfl_target_args & a) {
    rt::request_options o;
    o.sampling.temp = 0.0f;
    o.max_tokens = static_cast<int32_t>(k);
    fnfl_request_out ro;
    if (!fnfl_spec_request(e, prompt, 0, o, ro) || ro.toks.size() != k) {
        fprintf(stderr, "spec run: divergence: the request to %zu tokens: %s\n", k, e.error().c_str());
        return false;
    }
    rt::target & t = e.test_target();
    rt::round_result r;
    std::string err;
    if (!t.generate(r, err)) {
        fprintf(stderr, "spec run: divergence: %s\n", err.c_str());
        return false;
    }
    std::vector<float> row(static_cast<size_t>(V));
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaMemcpy(row.data() + size_t(c) * VC, t.card(c).logits, size_t(VC) * 4, cudaMemcpyDefault),
                     "logits")) {
            return false;
        }
    }
    bool checked = false;
    if (!t.close_check(checked, err)) {
        fprintf(stderr, "spec run: divergence: closing check: %s\n", err.c_str());
        return false;
    }
    int32_t top[2] = {-1, -1};
    for (int32_t v = 0; v < V; ++v) {
        if (top[0] < 0 || row[size_t(v)] > row[size_t(top[0])]) {
            top[1] = top[0];
            top[0] = v;
        } else if (top[1] < 0 || row[size_t(v)] > row[size_t(top[1])]) {
            top[1] = v;
        }
    }
    const float lo = row[size_t(own)], lr = row[size_t(ref)];
    printf("spec run: ME1 divergence at token %zu (position %zu): this run %d, the reference %d; a B = 1 round over "
           "the shared history decides %d (%s); logits: %d %.6f, %d %.6f, difference %.6f (%.3g of |%.6f|); the row's "
           "largest %d %.6f, second %d %.6f\n",
           k, prompt.size() + k, own, ref, r.out[0], r.out[0] == own ? "as the run" : "NOT as the run", own,
           double(lo), ref, double(lr), double(lo - lr), double((lo - lr) / std::fabs(lo)), double(lo), top[0],
           double(row[size_t(top[0])]), top[1], double(row[size_t(top[1])]));
    write_bin(a.out_dir + "/me1-divergence-logits.f32", row.data(), row.size() * 4);
    return true;
}

bool me1_check(rt::engine & e, fn::runtime::expert_cache & x, const fnfl_target_args & a) {
    std::vector<int32_t> prompt;
    if (!read_ids(a.ex_check, prompt)) {
        return false;
    }
    static const char * const name[4] = {"CPU misses", "GPU misses", "normal, cold", "normal, warm"};
    static const int32_t mode[4] = {1, 2, 0, 0};
    std::vector<int32_t> toks[4];
    std::string err;
    bool same = true;
    for (int32_t i = 0; i < 4; ++i) {
        if ((i < 3 && !x.reset_to_profile(err)) || !x.set_mode(mode[i], err)) {
            fprintf(stderr, "spec run: check: %s\n", err.c_str());
            return false;
        }
        x.reset_counters();
        rt::request_options o;
        o.sampling.temp = 0.0f;
        o.max_tokens = 128;
        fnfl_request_out ro;
        if (!fnfl_spec_request(e, prompt, 0, o, ro)) {
            fprintf(stderr, "spec run: check: %s\n", e.error().c_str());
            return false;
        }
        toks[i] = ro.toks;
        same = same && toks[i] == toks[0];
        printf("spec run: ME1 check %d (%s): %zu tokens, %s run 0's; prefill %.3f s; %s; expert cache: %s\n", i,
               name[i], toks[i].size(), toks[i] == toks[0] ? "equal to" : "DIFFERENT from", ro.pre_s,
               e.last_summary_line().c_str(), x.counters_json().c_str());
        write_tokens(a.out_dir + "/me1-check-" + std::to_string(i) + "-tokens.txt", toks[i]);
    }
    printf("spec run: ME1 check: placement invariance (4 runs, T = 0, %s): %s\n", a.ex_check.c_str(),
           same ? "PASS" : "FAIL");
    if (!a.ex_ref.empty()) {
        std::vector<int32_t> ref;
        if (read_ids(a.ex_ref, ref)) {
            size_t k = 0;
            while (k < ref.size() && k < toks[0].size() && ref[k] == toks[0][k]) {
                ++k;
            }
            printf("spec run: ME1 check against %s (%zu tokens): the first %zu of %zu tokens equal%s\n",
                   a.ex_ref.c_str(), ref.size(), k, toks[0].size(),
                   k == ref.size() && k == toks[0].size() ? " (all)" : "");
            if (k > 0 && k < ref.size() && k < toks[0].size() && !me1_divergence(e, prompt, k, toks[0][k], ref[k], a)) {
                return false;
            }
        }
    }
    fflush(stdout);
    return same;
}

// ME2's model-level check (--ex-kl IDS): NC1 section 5's teacher-forced KL against the pinned engine's decode-mode
// reference, in normal mode from the profile state, then with every decode miss on the CPU from the profile state.
// A FAIL is reported (the line and the summary), not fatal: the requests still run. ME4 (refill_held, with --me4-arms):
// before the normal pass a loan of the layer buffers alone is taken and returned and its refill held to one batch
// (test_refill_limit; the drain after the pass lifts it), so the pass's windows run with a refill outstanding (the
// real-weights check of windows served during a refill); their count is printed
bool me2_kl(rt::engine & e, fn::runtime::expert_cache & x, const fnfl_target_args & a, bool refill_held = false) {
    static const char * const name[2] = {"normal", "all-CPU"};
    std::string err;
    bool all = true;
    for (int32_t i = 0; i < 2; ++i) {
        if (!x.reset_to_profile(err) || !x.set_mode(i, err)) {
            fprintf(stderr, "spec run: KL check: %s\n", err.c_str());
            return false;
        }
        x.reset_counters();
        if (refill_held && i == 0) {
            void * const s[2] = {e.test_target().stream(0), e.test_target().stream(1)};
            const std::vector<md::byte_span> none;
            x.set_prompt_ring(0);
            if (!x.prompt_hold(true, s, &none, err) || !x.prompt_hold(false, s, nullptr, err)) {
                fprintf(stderr, "spec run: KL check: the held refill's loan: %s\n", err.c_str());
                return false;
            }
            x.test_refill_limit(1);
        }
        bool pass = false;
        if (!fnfl_kl_check(e.test_target(), a.ex_kl, a.ex_kl_ref, a.ex_kl_spread, a.out_dir, name[i], pass) ||
            !x.drain(err)) {
            fprintf(stderr, "spec run: KL check (%s) did not run: %s\n", name[i], err.c_str());
            return false;
        }
        all = all && pass;
        printf("spec run: ME2 KL check %d (%s): expert cache: %s\n", i, name[i], x.counters_json().c_str());
        if (refill_held && i == 0) {
            const fn::runtime::expert_cache_counters c = x.counters();
            printf("spec run: ME4 KL check with a refill outstanding: %lld lane waits of windows served while it was "
                   "outstanding (lane_wait_refill), %lld refill batches; %s\n",
                   (long long)c.lane_wait_refill.n, (long long)c.refill_batches,
                   c.lane_wait_refill.n > 0 ? "exercised" : "NOT EXERCISED");
        }
    }
    printf("spec run: ME2 KL check (normal and all-CPU, %s against %s): %s\n", a.ex_kl.c_str(), a.ex_kl_ref.c_str(),
           all ? "PASS" : "FAIL");
    fflush(stdout);
    return true;
}


// ME3 (--me3-arms, expires at ME3's disposition): an arm's boot state. The slot emptied and its checkpoints dropped;
// the arm's mechanisms (base: ME2b's whole loan, its refill enqueued by the decode caller, the staging thread's
// inherited mask, no short reads; final: ME3's) and executor (base eager, final the graphs; the target reset); the
// cache's profile placement, usage, heat, rounds, split share and refinement, mode and counters; the K/V tier's
// counters
bool me3_arm(rt::engine & e, fn::runtime::expert_cache & x, const std::string & arm) {
    const bool base = arm == "base";
    std::string err;
    rt::prefiller * pf = &e.prefill_driver();
    if (e.truncate(0, 0) != rt::run_status::ok) {
        fprintf(stderr, "spec run: ME3 arm %s: %s\n", arm.c_str(), e.error().c_str());
        return false;
    }
    pf->drop_all();
    x.set_legacy(base);
    pf->set_legacy(base);
    md::kv_tier * kt = e.test_target().tier();
    if (!e.test_target().set_executor(!base, err) || !x.reset_to_profile(err) || !x.set_mode(0, err) ||
        (kt != nullptr && !kt->reset_counters(err))) {
        fprintf(stderr, "spec run: ME3 arm %s: %s\n", arm.c_str(), err.c_str());
        return false;
    }
    x.reset_counters();
    printf("spec run: ME3 arm %s: the boot state restored (slot empty, checkpoints dropped, profile placement, usage, "
           "heat, split share %.3f, counters); executor %s; mechanisms %s\n",
           arm.c_str(), x.share(), e.test_target().graph_executor() ? "graph" : "eager",
           base ? "ME2b's (whole loan, refill enqueued by the decode caller, inherited staging mask, no short reads; "
                  "not reproduced: ME2b's zero-product down work for CPU slots, removed in step 2)"
                : "ME3's (sized loan, bounded refill, helper mask, short reads)");
    fflush(stdout);
    return true;
}

// ME4 (--me4-arms, expires at ME4's disposition): ME3's final boot state and mechanisms (me3_arm: the split's share,
// refinement and rates reset to the settings' start, so no arm inherits another's), then the arm's prompt staging and
// CPU path: base main's (no ring, the parity buffers at the zone's start, routed-rows, ME3's wrap: layer 0 staged
// again after the run's last layer; the per-miss reconstruction
// from the packed backing), final ME4's (the native views when the residency holds them: not in a dry run)
bool me4_arm(rt::engine & e, fn::runtime::expert_cache & x, const std::string & arm) {
    if (!me3_arm(e, x, "final")) {
        return false;
    }
    x.set_me4_base(arm == "base");
    printf("spec run: ME4 arm %s: prompt staging %s; CPU experts %s\n", arm.c_str(),
           arm == "base" ? "ME3's (layer buffers at the zone's start, routed-rows, the wrap)"
                         : "ME4's (the ring for one-half runs, the layer buffers by usage)",
           x.cpu_native() ? "the native views in place"
                          : "the reconstruction from the packed backing per miss (main's path)");
    fflush(stdout);
    return true;
}

// TEST (--me3-stall, Fable's load G ruling; expires at ME4's disposition): a completion wait that times out, then the
// engine's release. From the boot state, the refill held to no batches (expert_cache::test_refill_limit) so the decode
// after base's bulk read misses, and the service's first window with misses withheld (no copies, CPU work or flags:
// the GPU waits on flags nobody writes); base's request then fails on its round's completion timeout with the cache's
// GPU waits released and the GPU finished, and a second request is refused. The caller returns at once: the engine's
// teardown follows on a released cache (the wrapper checks the exit and the cards' release)
bool me3_stall_check(rt::engine & e, fn::runtime::expert_cache & x, const std::vector<int32_t> & base) {
    if (!me3_arm(e, x, "final")) {
        return false;
    }
    rt::request_options o;
    o.sampling.temp = 0.0f;
    o.max_tokens = 8;
    fnfl_request_out ro;
    x.test_refill_limit(0);
    x.test_withhold(1);
    const bool first = fnfl_spec_request(e, base, 0, o, ro);
    const std::string e1 = e.error();
    const int32_t layer = x.test_withheld();
    const bool second = fnfl_spec_request(e, base, 0, o, ro);
    const std::string e2 = e.error();
    const auto has = [](const std::string & t, const char * w) { return t.find(w) != std::string::npos; };
    const bool ok = layer >= 0 && !first && has(e1, "no completion record") && has(e1, "were released") &&
                    has(e1, "the GPU finished") && !second && has(e2, "released") && has(e2, "restart");
    printf("spec run: ME3 stall check: window withheld at layer %d; the request: %s (%s); the next request: %s (%s): "
           "%s\n",
           layer, first ? "served" : "failed", e1.c_str(), second ? "served" : "refused", e2.c_str(),
           ok ? "PASS" : "FAIL");
    fflush(stdout);
    return ok;
}

// ME3: a request's tokens: a fresh prompt (n_reuse 0), or an append (--next-append): the slot's history, its pending
// token, then ids, n_reuse the history's length. ME4 (--append-close): close >= 0 is read after the pending token
bool me3_request_tokens(rt::engine & e, const std::vector<int32_t> & ids, bool append, std::vector<int32_t> & all,
                        int32_t & n_reuse, int32_t close = -1) {
    all.clear();
    n_reuse = 0;
    if (!append) {
        all = ids;
        return true;
    }
    std::vector<int32_t> hist;
    int32_t pending = -1;
    if (e.history(0, hist, pending) != rt::run_status::ok || hist.empty()) {
        fprintf(stderr, "spec run: an append without a previous request's state\n");
        return false;
    }
    all = hist;
    if (pending >= 0) {
        all.push_back(pending);
    }
    if (close >= 0) {
        all.push_back(close);
    }
    all.insert(all.end(), ids.begin(), ids.end());
    n_reuse = static_cast<int32_t>(hist.size());
    return true;
}

// ME3: the request's prefill and affinity account: the path (short read or bulk), the arena's layout, the loan's spans,
// the staging thread's mask read back against the cache's helper, service and decode CPUs
std::string me3_prefill_text(rt::engine & e, const fn::runtime::expert_cache & x) {
    const rt::pf_run_stats & s = e.last_prefill();
    const std::vector<int32_t> * hc = x.helper_cpus();
    uint64_t want = 0;
    for (const int32_t c : *hc) {
        want |= c < 64 ? uint64_t(1) << c : 0;
    }
    const auto in_mask = [](uint64_t m, int32_t c) { return c >= 0 && c < 64 && (m >> c & 1) != 0; };
    std::string aff = "no staging thread ran";
    if (s.helper_mask != 0) {
        const bool on = (s.helper_mask & ~want) == 0;
        const bool off = !in_mask(s.helper_mask, x.service_cpu()) && !in_mask(s.helper_mask, x.decode_cpu());
        char b[256];
        snprintf(b, sizeof(b), "staging thread mask 0x%llx (helpers 0x%llx): on the helper CPUs %s, off the service "
                               "CPU %d and decode CPU %d %s",
                 (unsigned long long)s.helper_mask, (unsigned long long)want, on ? "yes" : "NO", x.service_cpu(),
                 x.decode_cpu(), off ? "yes" : "NO");
        aff = b;
    }
    char b[512];
    snprintf(b, sizeof(b), "ME3 prefill: %s; layout %d rows in halves of %d; loan spans %lld bytes (-1 the whole "
                           "region); %s",
             s.short_read ? "a known-token short read (target windows, no loan)" : (s.bulk ? "bulk" : "forced"),
             s.layout_rows, s.layout_half, (long long)s.loan_span_bytes, aff.c_str());
    return b;
}

// ME3: the request's boundaries (prompt entry to decode-ready, the first step and the first speculative step, entry to
// the completed continuation)
std::string me3_boundaries(rt::engine & e, const fnfl_request_out & ro) {
    double first = -1, first_spec = -1;
    for (const rt::step_info & x : e.last_steps()) {
        if (first < 0) {
            first = x.wall_s;
        }
        if (first_spec < 0 && x.kind == 2) {
            first_spec = x.wall_s;
        }
    }
    char b[320];
    snprintf(b, sizeof(b), "ME3 boundaries (driver clock): prompt entry to decode-ready %.4f s; first step %.3f ms; "
                           "first speculative step %.3f ms; entry through the completed continuation %.4f s",
             ro.pre_s, first * 1e3, first_spec * 1e3, ro.pre_s + ro.gen_s);
    return b;
}

// ME3: an append is a pure extension: only its new tokens processed (a deeper reuse restores a checkpoint and
// processes more)
bool me3_append_ok(const fnfl_request_out & ro, size_t n_req, int32_t n_reuse) {
    return ro.n_processed == static_cast<int32_t>(n_req) - n_reuse;
}

// ME3: tokens [n_reuse, all) prefilled with no continuation (the T = 0 settings); n_pre the tokens processed
bool me3_prefill(rt::engine & e, const std::vector<int32_t> & all, int32_t n_reuse, int32_t & n_pre) {
    rt::request_options o;
    o.sampling.temp = 0.0f;
    n_pre = 0;
    return e.set_sampling(0, o) == rt::run_status::ok && e.truncate(0, n_reuse) == rt::run_status::ok &&
           e.prefill(0, all.data() + n_reuse, static_cast<int32_t>(all.size()) - n_reuse, n_reuse, n_pre) ==
               rt::run_status::ok;
}

// ME3: the slot's T = 0 continuation of n tokens (its settings set by the prefill)
bool me3_continue(rt::engine & e, int32_t n, std::vector<int32_t> & toks) {
    int32_t buf[rt::max_step_tokens] = {};
    int32_t got = 0;
    bool end = false;
    while (!end && static_cast<int32_t>(toks.size()) < n) {
        if (e.step(0, buf, got, end) != rt::run_status::ok) {
            return false;
        }
        toks.insert(toks.end(), buf, buf + got);
    }
    return true;
}

// TEST (--me9-check, ME9; expires at ME9's disposition): admission on the dry one-device path, in the final arm.
//   1. a fresh prompt (base), 8 tokens and a drain (its loan and refill settled), then 32 tokens: rounds admit, each
//      layer's reservations made; held after = held before + retired + adopted - published over those 32 tokens;
//   2. every reserved slot poisoned (0xFF), then a known eligible GPU miss a layer (Shell 0a1b4eb5, request 11): of
//      each layer's last decode window's experts (the dry route repeats them) the most used resident evicted and its
//      lifecycle tracked, every miss GPU-served with the split's mode kept 0, the adapt pass held so no swap returns
//      it; 80 tokens more and a drain: every evicted key missed, was assigned, copied into a poisoned reserved slot and
//      published, and is resident by admission or was replaced after; every admitted resident's bytes equal its
//      backing (published only after its copy landed);
//   3. a bulk append (more: a loan, which may borrow reserved slots) and 8 tokens with its refill held outstanding
//      (test_refill_limit 0): no resident retired to replace a reservation, the reservations held = before less those
//      published at the loan's settle, rounds off for the refill, nothing admitted; after the drain (the refill
//      completed) every zone resident and every admitted one equals its backing;
//   the copy bytes: every GPU miss's copy in lane_bytes whatever its destination, admission's a subset (2, 3);
//   4. the boot state again (reset_to_profile): no reservation held.
//   The cache's request and counter records (as printed at each request's end) parse as JSON after 2 and 3.
// A dry route repeats every window, so without the eviction phase 2 would hold no new miss (R9-08).
bool me9_check(rt::engine & e, fn::runtime::expert_cache & x, const std::vector<int32_t> & base,
               const std::vector<int32_t> & more) {
    using cc = fn::runtime::expert_cache_counters;
    std::string err;
    bool ok = true;
    const auto line = [&](bool pass, const std::string & what) {
        ok = ok && pass;
        printf("spec run: ME9 admission check: %s: %s\n", what.c_str(), pass ? "ok" : "FAIL");
    };
    if (!me3_arm(e, x, "final")) {
        return false;
    }
    line(x.test_adm_held() == 0, "the boot state holds no reservation");
    std::vector<int32_t> toks;
    int32_t n_pre = 0;
    // the prompt's own loan and refill settled first (drain), then rounds with neither
    if (!me3_prefill(e, base, 0, n_pre) || !me3_continue(e, 8, toks) || !x.drain(err)) {
        fprintf(stderr, "spec run: ME9 check: %s%s\n", e.error().c_str(), err.c_str());
        return false;
    }
    const cc c0 = x.counters();
    const int32_t held0 = x.test_adm_held();
    if (!me3_continue(e, 32, toks) || !x.drain(err)) {
        fprintf(stderr, "spec run: ME9 check: %s%s\n", e.error().c_str(), err.c_str());
        return false;
    }
    const cc d1 = fn::runtime::counters_delta(x.counters(), c0);
    const int32_t held1 = x.test_adm_held();
    // Shell (request 11): over the same interval, reservations held = before + retired + adopted - published (a
    // collision re-reserves its slot; an invalidation keeps it)
    line(d1.adm_rounds_on > 0 && held1 > 0 && held1 == held0 + d1.adm_retired + d1.adm_adopted - d1.adm_published,
         "rounds admitting " + std::to_string(d1.adm_rounds_on) + ", reservations " + std::to_string(held0) + " -> " +
             std::to_string(held1) + " (retired " + std::to_string(d1.adm_retired) + ", adopted " +
             std::to_string(d1.adm_adopted) + ", published " + std::to_string(d1.adm_published) + ")");
    int32_t n_forced = 0;
    if (!x.test_adm_poison(err) || !x.test_adm_force(n_forced, err)) {
        fprintf(stderr, "spec run: ME9 check: %s\n", err.c_str());
        return false;
    }
    const cc c1 = x.counters();
    if (!me3_continue(e, 80, toks) || !x.drain(err)) {
        fprintf(stderr, "spec run: ME9 check: %s%s\n", e.error().c_str(), err.c_str());
        return false;
    }
    x.test_adm_force_end();
    const cc d2 = fn::runtime::counters_delta(x.counters(), c1);
    // the records as printed, strictly parsed (Shell 0a1b4eb5)
    const auto records = [&](const char * when) {
        size_t at_r = 0, at_c = 0;
        const std::string rj = x.request_json(), cj = x.counters_json();
        const bool vr = fn::runtime::json_valid(rj, at_r), vc = fn::runtime::json_valid(cj, at_c);
        line(vr && vc, std::string("records as JSON ") + when + ": request record " +
                           (vr ? "parses" : "NOT JSON at byte " + std::to_string(at_r)) + ", counters record " +
                           (vc ? "parses" : "NOT JSON at byte " + std::to_string(at_c)) + " (" +
                           std::to_string(x.share_attempts().size()) + " share attempts logged)");
    };
    records("after the forced misses");
    int64_t checked = 0, bad = 0;
    if (!x.test_adm_verify(checked, bad, err)) {
        fprintf(stderr, "spec run: ME9 check: %s\n", err.c_str());
        return false;
    }
    // Shell (fad3e540): every GPU miss's copy in lane_bytes whatever its destination (one device: a miss's entry is
    // one copy), the admitted ones a subset
    line(d2.lane_bytes == d2.gpu_miss_bytes && d2.adm_bytes > 0 && d2.adm_bytes <= d2.lane_bytes,
         "copy bytes with admission: lane " + std::to_string(d2.lane_bytes) + " = GPU miss " +
             std::to_string(d2.gpu_miss_bytes) + ", admitted " + std::to_string(d2.adm_bytes) + " of them");
    // Shell (request 11): every tracked key missed, was assigned, copied and published, and is resident by admission
    // now or left its slot after (legitimately replaced); a key short of that is listed with its events
    int32_t n_adm_now = 0, n_replaced = 0;
    std::string forced_fail;
    x.test_adm_forced(n_adm_now, n_replaced, forced_fail);
    line(n_forced > 0 && forced_fail.empty() && n_adm_now > 0 && d2.adm_started >= n_forced,
         "known misses: " + std::to_string(n_forced) + " layers' routed residents evicted, each missed, copied and "
             "published; " + std::to_string(n_adm_now) + " resident by admission now, " + std::to_string(n_replaced) +
             " replaced after publication; " + std::to_string(d2.gpu_miss) + " GPU misses" +
             (forced_fail.empty() ? std::string() : "; NOT SO: " + forced_fail));
    line(d2.adm_started > 0 && d2.adm_published > 0 && checked > 0 && bad == 0,
         "into poisoned reserved slots: " + std::to_string(d2.adm_started) + " copies started, " +
             std::to_string(d2.adm_published) + " published, " + std::to_string(d2.adm_hits) +
             " later hits; admitted residents " + std::to_string(checked) + " checked, " + std::to_string(bad) +
             " differ from their backing");
    // 3: the loan
    const int32_t held2 = x.test_adm_held();
    const cc c2 = x.counters();
    std::vector<int32_t> all;
    int32_t n_reuse = 0;
    std::vector<int32_t> t3;
    // the refill held to no batch, so the whole decode runs with it outstanding (its restore set's keys miss and are
    // served by the lane or the CPU, nothing admitted); the drain below lifts the hold
    x.test_refill_limit(0);
    if (!me3_request_tokens(e, more, true, all, n_reuse) || !me3_prefill(e, all, n_reuse, n_pre) ||
        !me3_continue(e, 8, t3)) {
        fprintf(stderr, "spec run: ME9 check: %s\n", e.error().c_str());
        return false;
    }
    const cc d3 = fn::runtime::counters_delta(x.counters(), c2);
    const int32_t held3 = x.test_adm_held();
    line(d3.lane_bytes == d3.gpu_miss_bytes && d3.adm_bytes == 0,
         "copy bytes with admission off (the refill): lane " + std::to_string(d3.lane_bytes) + " = GPU miss " +
             std::to_string(d3.gpu_miss_bytes) + ", admitted " + std::to_string(d3.adm_bytes));
    line(d3.loans >= 1 && d3.adm_retired == 0 && held3 + d3.adm_published == held2 && d3.adm_off_refill > 0,
         "a loan (" + std::to_string(d3.loans) + ") and decode during its refill: retired " +
             std::to_string(d3.adm_retired) + ", reservations " + std::to_string(held2) + " -> " +
             std::to_string(held3) + " with " + std::to_string(d3.adm_published) + " published at the settle, rounds "
             "off for the refill " + std::to_string(d3.adm_off_refill));
    int64_t zc = 0, zb = 0;
    if (!x.drain(err) || !x.test_verify_zone(zc, zb, err) || !x.test_adm_verify(checked, bad, err)) {
        fprintf(stderr, "spec run: ME9 check: %s\n", err.c_str());
        return false;
    }
    line(zb == 0 && bad == 0 && zc > 0, "after the drain: zone residents " + std::to_string(zc) + " checked, " +
                                            std::to_string(zb) + " differ; admitted " + std::to_string(checked) +
                                            " checked, " + std::to_string(bad) + " differ");
    records("after the loan");
    if (!me3_arm(e, x, "final")) {
        return false;
    }
    line(x.test_adm_held() == 0, "reset to the profile: no reservation held");
    printf("spec run: ME9 admission check: %s\n", ok ? "PASS" : "FAIL");
    fflush(stdout);
    return ok;
}

// TEST (R9-12's repair, Shell f056e89f; every dry run with ME3's loan and short-read checks): FC-RD2 in a verify
// round. A verify round carries its proposals on the token vector for its inputs only; an error word's recovery checks
// the ledger inside the round, so it must take them off first (R9-12 reported "ledger: L != N + U" in place of the
// round's error word). A fresh base request, 8 tokens at T 0 (the reference); the same request with its first verify
// round's commit seeing an injected error on card 1: the request ends with that round's error word reported as
// recovered (not a ledger failure) and the ledger holds L = N + U at the pre-step state; then the same request again
// equals the reference's tokens.
bool verify_recovery_check(rt::engine & e, fn::runtime::expert_cache & x, const std::vector<int32_t> & base) {
    rt::request_options o;
    o.sampling.temp = 0.0f;
    o.max_tokens = 8;
    std::string err;
    fnfl_request_out ref, bad, again;
    if (!me3_arm(e, x, "final") || !fnfl_spec_request(e, base, 0, o, ref) || !x.drain(err)) {
        fprintf(stderr, "spec run: verify-recovery check: %s%s\n", e.error().c_str(), err.c_str());
        return false;
    }
    e.test_target().test_fail_next_verify(2);
    const bool failed = !fnfl_spec_request(e, base, 0, o, bad);
    const std::string msg = e.error();
    const fn::runtime::ledger & l = e.test_target().slot();
    const int64_t L = static_cast<int64_t>(l.tok.size()), N = l.n, U = l.u;
    e.test_target().test_fail_next_verify(0);
    const bool drained = x.drain(err);
    const bool ran = drained && fnfl_spec_request(e, base, 0, o, again) && x.drain(err);
    const bool reported = failed && msg.find("error word") != std::string::npos &&
                          msg.find("recovered") != std::string::npos && msg.find("ledger") == std::string::npos;
    const bool pass = reported && L == N + U && ran && again.toks == ref.toks && !ref.toks.empty();
    printf("spec run: verify-recovery check (an injected commit error on card 1 in the first verify round): the "
           "request %s with \"%s\"; ledger after it L %lld = N %lld + U %lld: %s; the next request's %zu tokens %s "
           "the reference's %zu: %s\n",
           failed ? "ended" : "DID NOT FAIL", msg.c_str(), (long long)L, (long long)N, (long long)U,
           L == N + U ? "yes" : "NO", again.toks.size(), again.toks == ref.toks ? "equal" : "DIFFER FROM",
           ref.toks.size(), pass ? "PASS" : "FAIL");
    if (!drained || !ran) {
        fprintf(stderr, "spec run: verify-recovery check: %s%s\n", e.error().c_str(), err.c_str());
    }
    fflush(stdout);
    return pass;
}

// TEST (--me3-cancel, Fable's ruling on ME3's completeness gaps; expires at ME4's disposition): prefill-test's two
// cancellations (its dry B4, the prefiller at 8,192 and the resend from its checkpoint; its engine E7, the engine's
// callback at 8,192 and the resend) with the expert cache on, in the final arm. Each cancelled prompt runs with its
// borrowed slots poisoned (expert_cache::test_poison_lent). Right after the cancel the prompt's hold has ended, its
// restore set is exactly its borrowed residents and every zone resident never borrowed equals its backing. B4's
// loan returns at a drain, then its resend equals the prefiller's uninterrupted run's state. E7's resend follows with
// no drain (its hold returns the cancelled zone first) and equals the engine's uninterrupted prompt's state and T = 0
// continuation (8 tokens). After each drain every loan has returned exactly the borrowed residents' bytes, nothing
// carried, every zone resident equals its backing, no refill is outstanding and the target's streams are idle
std::string me3_json_field(const std::string & js, const char * key) {
    const std::string k = std::string("\"") + key + "\": ";
    const size_t p = js.find(k);
    if (p == std::string::npos) {
        return "?";
    }
    const size_t a = p + k.size();
    return js.substr(a, js.find_first_of(",}", a) - a);
}

std::string me3_state_diff(const state_hash & a, const state_hash & b) {
    std::string d;
    if (a.n != b.n || a.draft_n != b.draft_n || a.index_n != b.index_n) {
        d = " N " + std::to_string(a.n) + "/" + std::to_string(b.n) + ", draft N " + std::to_string(a.draft_n) + "/" +
            std::to_string(b.draft_n) + ", index N " + std::to_string(a.index_n) + "/" + std::to_string(b.index_n);
    }
    for (int32_t i = 0; i < n_hash; ++i) {
        d += a.h[i] == b.h[i] ? "" : std::string(" ") + hash_names[i];
    }
    return d.empty() ? "equal" : "DIFFERENT:" + d;
}

bool me3_cancel_check(rt::engine & e, fn::runtime::expert_cache & x, const std::vector<int32_t> & P) {
    constexpr int32_t at = 8192, n_cont = 8;
    const int32_t L = static_cast<int32_t>(P.size());
    rt::target & t = e.test_target();
    md::model & m = e.test_model();
    rt::prefiller & pf = e.prefill_driver();
    rt::request_options o;
    o.sampling.temp = 0.0f;
    std::string err;
    const rt::prefill_progress stop = [](int64_t, int64_t done, int64_t) { return done < at; };
    const auto streams_idle = [&]() {
        bool idle = true;
        for (int32_t c = 0; c < 2; ++c) {
            idle = idle && cudaSetDevice(m.device[c]) == cudaSuccess &&
                   cudaStreamQuery(static_cast<cudaStream_t>(t.stream(c))) == cudaSuccess;
        }
        return idle;
    };
    // right after a cancel: the prompt's hold ended and its zone still out (expert_cache::prompt_hold: its keys refill
    // at the next decode window, the next prompt's hold or a drain, as after a whole prompt), the restore set exactly
    // the one loan's borrowed residents, and every zone resident never borrowed its backing
    const auto after_cancel = [&](const char * form, const fn::runtime::expert_cache_counters & c0) {
        int64_t k = 0, b = 0;
        const std::string j = x.request_json();
        const int64_t rk = std::atoll(me3_json_field(j, "refill_outstanding_keys").c_str());
        const bool v = x.test_verify_zone(k, b, err);
        const fn::runtime::expert_cache_counters c = fn::runtime::counters_delta(x.counters(), c0);
        const bool ok = v && !x.prompt_held() && me3_json_field(j, "zone_taken") == "true" && c.loans == 1 &&
                        c.refills == 0 && c.borrowed_slots > 0 && c.borrowed_slots < x.zone_slots() &&
                        rk == c.borrowed_residents && k > 0 && b == 0;
        printf("spec run: ME3 cancel check (%s): after the cancel the hold %s, the zone taken %s; loans %lld (%lld of "
               "the zone's %lld slots, %lld residents, %lld bytes), returns %lld; restore set %lld keys; %lld zone "
               "residents never borrowed, %lld differ from their backing: %s%s\n",
               form, x.prompt_held() ? "HELD" : "ended", me3_json_field(j, "zone_taken").c_str(), (long long)c.loans,
               (long long)c.borrowed_slots, (long long)x.zone_slots(), (long long)c.borrowed_residents,
               (long long)c.borrowed_resident_bytes, (long long)c.refills, (long long)rk, (long long)k, (long long)b,
               ok ? "ok" : "FAIL", err.empty() ? "" : (" [" + err + "]").c_str());
        return ok;
    };
    // after a drain: every loan since c0 returned, the refill's bytes exactly the borrowed residents', nothing carried,
    // every zone resident its backing, no refill outstanding and the zone not taken
    const auto after_drain = [&](const char * form, const fn::runtime::expert_cache_counters & c0, int64_t loans) {
        int64_t k = 0, b = 0;
        const bool v = x.drain(err) && x.test_verify_zone(k, b, err);
        const std::string j = x.request_json();
        const fn::runtime::expert_cache_counters c = fn::runtime::counters_delta(x.counters(), c0);
        const bool ok = v && c.loans == loans && c.refills == loans && c.refill_bytes == c.borrowed_resident_bytes &&
                        c.refill_carried == 0 && k > 0 && b == 0 &&
                        me3_json_field(j, "refill_outstanding_keys") == "0" && me3_json_field(j, "zone_taken") == "false";
        printf("spec run: ME3 cancel check (%s): after the drain loans %lld, returns %lld, borrowed residents %lld "
               "bytes, refill %lld bytes, carried %lld; %lld zone residents, %lld differ from their backing; refill "
               "outstanding %s keys, zone taken %s: %s%s\n",
               form, (long long)c.loans, (long long)c.refills, (long long)c.borrowed_resident_bytes,
               (long long)c.refill_bytes, (long long)c.refill_carried, (long long)k, (long long)b,
               me3_json_field(j, "refill_outstanding_keys").c_str(), me3_json_field(j, "zone_taken").c_str(),
               ok ? "ok" : "FAIL", err.empty() ? "" : (" [" + err + "]").c_str());
        return ok;
    };
    // the uninterrupted prompt: its state and T = 0 continuation
    state_hash h_ref, h;
    std::vector<int32_t> ref, toks;
    int32_t np = 0;
    bool ok = me3_arm(e, x, "final") && e.set_sampling(0, o) == rt::run_status::ok &&
              e.prefill(0, P.data(), L, 0, np) == rt::run_status::ok && np == L && snapshot(t, m, h_ref) &&
              me3_continue(e, n_cont, ref) && x.drain(err);
    if (!ok) {
        fprintf(stderr, "spec run: ME3 cancel check: the uninterrupted prompt: %s%s\n", e.error().c_str(), err.c_str());
        return false;
    }
    // B4's form: the prefiller's uninterrupted run (its state, the reference), then its run stopped at 8,192 through
    // its callback and begun again from its checkpoint
    bool pass = me3_arm(e, x, "final") && e.set_sampling(0, o) == rt::run_status::ok;
    int64_t from = -1;
    rt::pf_run_stats st;
    state_hash h_pf;
    pass = pass && pf.begin(t, P, 0, from, err) && pf.run(t, &st, err) && from == 0 && snapshot(t, m, h_pf) &&
           x.drain(err) && me3_arm(e, x, "final") && e.set_sampling(0, o) == rt::run_status::ok;
    const std::string d_pf = pass ? me3_state_diff(h_ref, h_pf) : "not run";
    fn::runtime::expert_cache_counters c0 = x.counters();
    x.test_poison_lent(true);
    const bool begun = pass && pf.begin(t, P, 0, from, err);
    const bool stopped = begun && !pf.run(t, &st, err, stop);
    x.test_poison_lent(false);
    const bool at_ok = stopped && st.stopped == at && t.slot().n == at && t.draft_n() == at &&
                       pf.book().latest(at) >= 0 && pf.book().at(pf.book().latest(at)).pos == at;
    const int64_t stopped_at = st.stopped, n_at = t.slot().n, draft_at = t.draft_n();
    err.clear();
    const bool loan_b4 = stopped && after_cancel("prefiller, B4's form", c0) &&
                         after_drain("prefiller, B4's form: the cancelled prompt's loan", c0, 1);
    const bool resent = loan_b4 && pf.begin(t, P, at, from, err) && pf.run(t, &st, err) && from == at &&
                        snapshot(t, m, h) && x.drain(err);
    const std::string d4 = resent ? me3_state_diff(h_pf, h) : "not resent";
    const bool b4 = pass && at_ok && loan_b4 && resent && d4 == "equal" && streams_idle();
    printf("spec run: ME3 cancel check (prefiller, B4's form): its uninterrupted run against the engine's prompt: %s; "
           "stopped at %lld (N %lld, draft N %lld, checkpoint %s); resend from %lld; state against its uninterrupted "
           "run: %s; streams idle after: %s: %s%s\n",
           d_pf.c_str(), (long long)stopped_at, (long long)n_at, (long long)draft_at, at_ok ? "at it" : "WRONG",
           (long long)from, d4.c_str(), streams_idle() ? "yes" : "NO", b4 ? "PASS" : "FAIL",
           err.empty() ? "" : (" [" + err + "]").c_str());
    // E7's form: the engine's prefill cancelled at 8,192 through its callback, then the resend from its checkpoint
    err.clear();
    pass = me3_arm(e, x, "final") && e.set_sampling(0, o) == rt::run_status::ok;
    c0 = x.counters();
    x.test_poison_lent(true);
    const rt::run_status cs = pass ? e.prefill(0, P.data(), L, 0, np, stop) : rt::run_status::error;
    x.test_poison_lent(false);
    const int32_t np_c = np;
    const std::string pl = e.last_phase_line();
    const bool phase_ok = pl.find("decode-ready not reached") != std::string::npos &&
                          pl.find("; rows 0;") != std::string::npos &&
                          pl.find("restored checkpoint none") != std::string::npos;
    err.clear();
    const bool loan_e7 = cs == rt::run_status::cancelled && np_c == at && after_cancel("engine, E7's form", c0);
    toks.clear();
    // the serving path: the resend with no drain between, its hold returning the cancelled prompt's zone first
    const bool resent7 = loan_e7 && e.prefill(0, P.data() + at, L - at, at, np) == rt::run_status::ok &&
                         np == L - at && e.last_prefill().from == at && snapshot(t, m, h) &&
                         me3_continue(e, n_cont, toks) &&
                         after_drain("engine, E7's form: the cancelled prompt's and the resend's loans", c0, 2);
    const std::string d7 = resent7 ? me3_state_diff(h_ref, h) : "not resent";
    const std::string j_end = x.request_json();
    const bool e7 = phase_ok && loan_e7 && resent7 && d7 == "equal" && toks == ref && streams_idle() &&
                    me3_json_field(j_end, "refill_outstanding_keys") == "0" &&
                    me3_json_field(j_end, "zone_taken") == "false";
    printf("spec run: ME3 cancel check (engine, E7's form): status %s, %d processed; phase line %s; resend from %lld, "
           "%d processed; state against the uninterrupted prompt: %s; its %zu tokens %s; at the end refill outstanding "
           "%s keys, zone taken %s, streams idle %s: %s%s\n",
           cs == rt::run_status::cancelled ? "cancelled" : "NOT cancelled", np_c,
           phase_ok ? "ok" : ("WRONG (" + pl + ")").c_str(),
           (long long)e.last_prefill().from, np, d7.c_str(), toks.size(), toks == ref ? "equal" : "DIFFER",
           me3_json_field(j_end, "refill_outstanding_keys").c_str(), me3_json_field(j_end, "zone_taken").c_str(),
           streams_idle() ? "yes" : "NO", e7 ? "PASS" : "FAIL",
           (err + e.error()).empty() ? "" : (" [" + err + e.error() + "]").c_str());
    printf("spec run: ME3 cancel check: %s\n", b4 && e7 ? "PASS" : "FAIL");
    fflush(stdout);
    return b4 && e7;
}

// ME3 (in the final arm's mode): the loan check. A base prompt, then a bulk append of more with the borrowed slots and
// spans poisoned: right after the append's prefill every zone slot still holding a key (none borrowed) equals its
// backing bytes, then after the drain (the refill) every zone resident does; the borrowed set is smaller than the
// zone, and the refill's bytes are the borrowed residents'; the append's T = 0 continuation (8 tokens) equals the
// unpoisoned one's
bool me3_loan_check(rt::engine & e, fn::runtime::expert_cache & x, const std::vector<int32_t> & base,
                    const std::vector<int32_t> & more) {
    rt::request_options o;
    o.sampling.temp = 0.0f;
    o.max_tokens = 8;
    std::vector<int32_t> toks[2];
    bool ok = true;
    for (int32_t poison = 1; poison >= 0; --poison) {
        if (!me3_arm(e, x, "final")) {
            return false;
        }
        fnfl_request_out ro;
        std::vector<int32_t> all;
        int32_t n_reuse = 0, n_pre = 0;
        std::string err;
        if (!fnfl_spec_request(e, base, 0, o, ro) || !x.drain(err) ||
            !me3_request_tokens(e, more, true, all, n_reuse)) {
            fprintf(stderr, "spec run: ME3 loan check: %s%s\n", e.error().c_str(), err.c_str());
            return false;
        }
        x.test_poison_lent(poison != 0);
        const fn::runtime::expert_cache_counters c0 = x.counters();
        const bool pre = me3_prefill(e, all, n_reuse, n_pre);
        x.test_poison_lent(false);
        int64_t checked = 0, bad = 0, checked2 = 0, bad2 = 0;
        if (!pre || n_pre != static_cast<int32_t>(all.size()) - n_reuse || !x.test_verify_zone(checked, bad, err)) {
            fprintf(stderr, "spec run: ME3 loan check: the append (%d processed): %s%s\n", n_pre, e.error().c_str(),
                    err.c_str());
            return false;
        }
        const rt::pf_run_stats ps = e.last_prefill();
        if (!me3_continue(e, o.max_tokens, toks[poison]) || !x.drain(err) ||
            !x.test_verify_zone(checked2, bad2, err)) {
            fprintf(stderr, "spec run: ME3 loan check: %s%s\n", e.error().c_str(), err.c_str());
            return false;
        }
        const fn::runtime::expert_cache_counters c = fn::runtime::counters_delta(x.counters(), c0);
        const bool sized = c.loans == 1 && c.borrowed_slots > 0 && c.borrowed_slots < x.zone_slots() &&
                           c.refill_bytes == c.borrowed_resident_bytes;
        const bool pass = bad == 0 && bad2 == 0 && checked > 0 && checked2 > checked && sized && !ps.short_read;
        ok = ok && pass;
        printf("spec run: ME3 loan check (%s): the append's loan %lld of the zone's %lld slots, %lld residents (%lld "
               "bytes), refill %lld bytes, spans %lld bytes, layout %d / %d; after its prefill %lld zone residents "
               "never borrowed, %lld differ from their backing; after the refill %lld zone residents, %lld differ: "
               "%s\n",
               poison ? "poisoned" : "unpoisoned", (long long)c.borrowed_slots, (long long)x.zone_slots(),
               (long long)c.borrowed_residents, (long long)c.borrowed_resident_bytes, (long long)c.refill_bytes,
               (long long)ps.loan_span_bytes, ps.layout_rows, ps.layout_half, (long long)checked, (long long)bad,
               (long long)checked2, (long long)bad2, pass ? "ok" : "FAIL");
    }
    const bool same = toks[0] == toks[1];
    ok = ok && same;
    printf("spec run: ME3 loan check (sized loan, poison, retained and refilled slots): the poisoned append's %zu "
           "tokens %s the unpoisoned one's: %s\n",
           toks[1].size(), same ? "equal" : "DIFFER from", ok ? "PASS" : "FAIL");
    fflush(stdout);
    return ok;
}

// the first position where a and b differ (their common length when one is a prefix of the other)
size_t first_difference(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) {
        ++i;
    }
    return i;
}

// ME3 (in the final arm's mode): the short-read check over a base prompt (bulk) and more (at least 400 tokens):
//   - a read of short_read tokens (its first short_read - 1 appended after the base request's pending output) read
//     short (no loan, no refill, the short windows counted), and its T = 0 continuation (8 tokens) against the same
//     append read in bulk (short reads off for it);
//   - the threshold, counted in tokens read: short_read + 1 read in bulk (one loan), short_read short;
//   - a prior outstanding loan: a bulk append of 200 tokens, its refill held to one batch (the cleanup's test hook,
//     expert_cache::test_refill_limit), then at once (no step, its refill outstanding) a 13-token append: short, no
//     new loan, the outstanding refill reported at the request's entry (not as zero work); then a second bulk append
//     of 200 at once: its loan stops the held refill (its batch landed and published, the moves never submitted
//     carried back to the restore set), and after the drain (the limit lifted) every zone resident equals its backing;
//   - a deeper reuse than the history fails the pure-extension test.
// Each short read's handoff (N = L, the draft at N) is checked in prefiller::run_short. No graph is captured again
// TEST (ME10): X-F's receive slots on both cards at the sentinel, read synchronously between requests. On one device
// the cached FFN's finish replaces X-F's exchange (its slots unused); the draft's FFN exchange still uses and resets
// them. A slot left written (a send without its receive) shows here
bool me10_xf_check(rt::engine & e) {
    const fn::xchg::rings & rg = *e.test_model().rings;
    constexpr int32_t n = fn::xchg::slot_words(fn::xchg::msg::f);
    std::vector<uint32_t> h(static_cast<size_t>(n));
    int64_t checked = 0, bad = 0;
    for (int32_t card = 0; card < 2; ++card) {
        for (int32_t s = 0; s < fn::xchg::slots_of(fn::xchg::msg::f); ++s) {
            if (cudaMemcpy(h.data(), rg.slot(card, fn::xchg::msg::f, s), size_t(n) * 4, cudaMemcpyDefault) !=
                cudaSuccess) {
                fprintf(stderr, "spec run: ME10 X-F check: %s\n", cudaGetErrorString(cudaGetLastError()));
                return false;
            }
            for (const uint32_t w : h) {
                ++checked;
                bad += w != fn::xchg::sentinel ? 1 : 0;
            }
        }
    }
    printf("spec run: ME10 X-F slots between requests: %lld words on both cards, %lld not the sentinel: %s\n",
           (long long)checked, (long long)bad, bad == 0 ? "ok" : "FAIL");
    return bad == 0;
}

bool me3_short_check(rt::engine & e, fn::runtime::expert_cache & x, const std::vector<int32_t> & base,
                     const std::vector<int32_t> & more) {
    rt::prefiller & pf = e.prefill_driver();
    const int32_t sr = pf.short_read(), chunk = pf.chunk_rows();
    if (sr < 1 || static_cast<int32_t>(more.size()) < std::max(sr + 1, 400)) {
        fprintf(stderr, "spec run: ME3 short-read check: short reads off or fewer than %d tokens to append\n",
                std::max(sr + 1, 400));
        return false;
    }
    const int32_t g0 = e.test_target().captures();
    rt::request_options o;
    o.sampling.temp = 0.0f;
    o.max_tokens = 8;
    std::string err;
    // one append to a fresh base request: the counters' change, the read's path, the continuation
    struct appended {
        fn::runtime::expert_cache_counters c;
        bool short_read = false;
        int32_t read = 0; // the tokens the prefill read (a continued request's pending token among them)
        std::vector<int32_t> toks;
        std::string line;
    };
    // limit (>= 0): the refill's batches held to it from this append on (a drain lifts it)
    const auto append = [&](const std::vector<int32_t> & ids, bool fresh, int32_t steps, appended & out,
                            int32_t limit = -1) {
        fnfl_request_out ro;
        std::vector<int32_t> all;
        int32_t n_reuse = 0, n_pre = 0;
        if (fresh && (!me3_arm(e, x, "final") || !fnfl_spec_request(e, base, 0, o, ro) || !x.drain(err))) {
            return false;
        }
        if (limit >= 0) {
            x.test_refill_limit(limit);
        }
        const fn::runtime::expert_cache_counters c0 = x.counters();
        x.request_begin();
        if (!me3_request_tokens(e, ids, true, all, n_reuse) || !me3_prefill(e, all, n_reuse, n_pre) ||
            n_pre != static_cast<int32_t>(all.size()) - n_reuse) {
            return false;
        }
        out.short_read = e.last_prefill().short_read;
        out.read = n_pre;
        out.line = x.request_json();
        if (!me3_continue(e, steps, out.toks)) {
            return false;
        }
        out.c = fn::runtime::counters_delta(x.counters(), c0);
        return true;
    };
    // a fresh base request leaves its last output pending, which the append's read includes: short_read - 1 new
    // tokens read short_read, short_read new ones short_read + 1; after an append without steps nothing is pending
    const std::vector<int32_t> m_sr1(more.begin(), more.begin() + sr - 1), m_sr(more.begin(), more.begin() + sr),
        m_bulk(more.begin(), more.begin() + 200), m_13(more.begin() + 200 - 13, more.begin() + 200),
        m_bulk2(more.begin() + 200, more.begin() + 400);
    appended s1, b1, over, at, bulk, after, bulk2;
    bool ok = append(m_sr1, true, o.max_tokens, s1);
    ok = ok && pf.set_options(0, chunk, err) && append(m_sr1, true, o.max_tokens, b1) &&
         pf.set_options(sr, chunk, err);
    ok = ok && append(m_sr, true, 0, over) && append(m_sr, false, 0, at);
    ok = ok && append(m_bulk, true, 0, bulk, 1) && append(m_13, false, 0, after) && append(m_bulk2, false, 0, bulk2) &&
         x.drain(err);
    int64_t z_checked = 0, z_bad = 0;
    ok = ok && x.test_verify_zone(z_checked, z_bad, err);
    if (!ok) {
        x.test_refill_limit(-1);
        pf.set_options(sr, chunk, err);
        fprintf(stderr, "spec run: ME3 short-read check: %s%s\n", e.error().c_str(), err.c_str());
        return false;
    }
    const auto none = [](const fn::runtime::expert_cache_counters & c) {
        return c.loans == 0 && c.borrowed_slots == 0 && c.borrowed_bytes == 0 && c.short_windows > 0;
    };
    const bool short_ok = s1.read == sr && s1.short_read && none(s1.c) && s1.c.refill_bytes == 0 && b1.read == sr &&
                          !b1.short_read && b1.c.loans == 1;
    const std::string diff_at =
        s1.toks == b1.toks ? "" : " (first at " + std::to_string(first_difference(s1.toks, b1.toks)) + ")";
    const bool threshold = over.read == sr + 1 && !over.short_read && over.c.loans == 1 && at.read == sr &&
                           at.short_read && none(at.c);
    const bool outstanding = !bulk.short_read && bulk.c.loans == 1 && after.short_read && none(after.c);
    // the held refill (one batch) stopped by the second loan: its unsubmitted moves carried back
    const bool stopped = !bulk2.short_read && bulk2.c.loans == 1 && bulk2.c.refill_carried > 0 &&
                         bulk.c.refill_batches + after.c.refill_batches <= 1 && z_checked > 0 && z_bad == 0;
    // a deeper reuse: the history kept to 100 tokens fewer, so the prefill restores a checkpoint and processes more
    fnfl_request_out ro;
    std::vector<int32_t> all;
    int32_t n_reuse = 0;
    ok = me3_request_tokens(e, m_13, true, all, n_reuse);
    const int32_t deep = std::max(n_reuse - 100, 1);
    ok = ok && fnfl_spec_request(e, all, deep, o, ro) && x.drain(err);
    const bool refused = ok && ro.n_processed > static_cast<int32_t>(all.size()) - deep &&
                         !me3_append_ok(ro, all.size(), deep);
    const int32_t g1 = e.test_target().captures();
    const bool pass = short_ok && threshold && outstanding && stopped && refused && g1 == g0;
    printf("spec run: ME3 short-read check (threshold %d): the %d-token read %s (loans %lld, borrowed %lld bytes, "
           "refill %lld bytes, %lld short windows of %lld rows); its 8 tokens %s the bulk read's%s\n",
           sr, s1.read, s1.short_read ? "short" : "IN BULK", (long long)s1.c.loans,
           (long long)s1.c.borrowed_bytes, (long long)s1.c.refill_bytes, (long long)s1.c.short_windows,
           (long long)s1.c.short_rows, s1.toks == b1.toks ? "equal" : "differ from", diff_at.c_str());
    printf("spec run: ME3 short-read check: a %d-token read %s (loans %lld), a %d-token read %s (loans %lld); after "
           "a bulk append of 200 (loans %lld), 13 tokens at once %s (loans %lld, borrowed %lld bytes), entry %s\n",
           over.read, over.short_read ? "SHORT" : "in bulk", (long long)over.c.loans, at.read,
           at.short_read ? "short" : "IN BULK", (long long)at.c.loans, (long long)bulk.c.loans,
           after.short_read ? "short" : "IN BULK", (long long)after.c.loans, (long long)after.c.borrowed_bytes,
           after.line.substr(0, after.line.find(", \"settings\"")).c_str());
    printf("spec run: ME3 short-read check: the refill held to one batch (%lld batches before the second loan); a "
           "second bulk append of 200 at once (loans %lld) stopped it, %lld moves carried back; after the drain %lld "
           "zone residents, %lld differ from their backing: %s\n",
           (long long)(bulk.c.refill_batches + after.c.refill_batches), (long long)bulk2.c.loans,
           (long long)bulk2.c.refill_carried, (long long)z_checked, (long long)z_bad, stopped ? "ok" : "FAIL");
    printf("spec run: ME3 short-read check: a reuse of %d of %d kept tokens processed %d, not its %d new: %s; graph "
           "captures %d before, %d after: %s\n",
           deep, n_reuse, ro.n_processed, static_cast<int32_t>(all.size()) - deep,
           refused ? "refused as not a pure extension" : "NOT REFUSED", g0, g1, pass ? "PASS" : "FAIL");
    fflush(stdout);
    return pass;
}

} // namespace

int fnfl_spec_run(const fnfl_target_args & a) {
    const bool dry = a.mode == "run-dry";
    // ME2 (--next-ids): the prompts in order, PROMPT_IDS first
    std::vector<std::string> paths{a.ids};
    paths.insert(paths.end(), a.next_ids.begin(), a.next_ids.end());
    // ME3: which prompts are appends (--next-append), and the arms (--me3-arms; none: the run as configured)
    std::vector<bool> append(paths.size(), false);
    std::vector<int32_t> rows(paths.size(), -1); // ME3 (--next-append-rows): the read's rows, checked
    for (size_t i = 0; i < a.next_append.size() && i + 1 < paths.size(); ++i) {
        append[i + 1] = a.next_append[i];
        rows[i + 1] = i < a.next_rows.size() ? a.next_rows[i] : -1;
    }
    std::vector<std::string> arms;
    const bool me4 = !a.me4_arms.empty(); // ME4's arms (--me4-arms), else ME3's
    const std::string & arm_list = me4 ? a.me4_arms : a.me3_arms;
    for (size_t q = 0; q < arm_list.size();) {
        const size_t c = std::min(arm_list.find(',', q), arm_list.size());
        arms.push_back(arm_list.substr(q, c - q));
        q = c + 1;
    }
    if (arms.empty()) {
        arms.emplace_back();
    }
    std::vector<std::vector<int32_t>> prompts(paths.size());
    for (size_t p = 0; p < paths.size(); ++p) {
        if (!read_ids(paths[p], prompts[p])) {
            return 1;
        }
        if (rows[p] > 0 && static_cast<int32_t>(prompts[p].size()) < rows[p]) {
            fprintf(stderr, "spec run: %s has fewer than %d IDs (--next-append-rows)\n", paths[p].c_str(), rows[p]);
            return 1;
        }
        if (dry && rows[p] < 0) {
            prompts[p].resize(std::min<size_t>(prompts[p].size(), 13));
        }
    }
    const std::vector<int32_t> & prompt = prompts[0];
    rt::draft_options dopt;
    if (!fnfl_draft_settings(a, true, "spec run", dopt)) {
        return 1;
    }
    std::vector<fnfl_target_args::run_setting> settings = a.run_settings;
    if (settings.empty()) {
        settings.emplace_back();
        settings.back().sampling.temp = 0.0f;
    }
    // ME1: a cached dry run takes --cells (the memory plan at the served capacity)
    const int32_t cells = dry && !a.cached() ? 8192 : a.n_cells;
    auto m = std::make_unique<md::model>();
    if (!(dry ? fnfl_load_dry(a, cells, *m) : fnfl_load(a, cells, *m, "spec run"))) {
        return 1;
    }
    rt::engine_config c;
    c.model_path = a.model_path;
    c.n_ctx_slot = cells;
    c.draft = dopt;
    c.eog = a.eog.empty() ? std::vector<int32_t>{eos_tokenizer} : a.eog;
    c.timeout_s = dry ? 30.0 : 60.0;
    c.spec_d = a.spec_d;
    // ME1: the expert cache (one device, or cached TP), made after the engine's other allocations. ME3: the executor
    // --ex-executor names (the graphs unless eager); the cache captures the graphs before its pool (service_hook)
    fn::runtime::expert_cache * xc = nullptr;
    c.eager = a.ex_eager;
    c.short_read = a.cached() ? a.cache.short_read : 0; // ME3
    c.prompt_chunk = a.cache.prompt_chunk;
    if (a.cached()) {
        c.experts = [&a, dry, &xc](md::model & mm, const md::service_hook & hook,
                                   std::string & e2) -> std::unique_ptr<md::expert_service> {
            std::unique_ptr<fn::runtime::expert_cache> x = fnfl_expert_cache(a, mm, dry, "spec run", hook);
            if (x == nullptr) {
                e2 = "the expert cache";
                return nullptr;
            }
            xc = x.get();
            return x;
        };
    }
    rt::engine e;
    std::string err;
    if (!e.init(std::move(m), c, err)) {
        fprintf(stderr, "spec run: engine: %s\n", err.c_str());
        return 1;
    }
    printf("spec run: %s\n", e.settings().c_str());
    // ME3: the graphs captured at init; none after (the end's line compares)
    const int32_t captures0 = e.test_target().captures();
    printf("spec run: ME3 executor: %s; %d graph captures at init\n",
           e.test_target().graph_executor() ? "graph" : "eager", captures0);
    if (xc != nullptr) {
        printf("spec run: ME1 %s, %s executor, expert cache on; %s\n",
               a.single_card >= 0 ? ("single card: device " + std::to_string(a.single_card)).c_str() : "cached TP",
               e.test_target().graph_executor() ? "graph" : "eager", me1_memory(*xc).c_str());
        // ME3: the model checks in the final arm's mode, then (with the arms) the loan check
        if (!arms.front().empty() && !(me4 ? me4_arm(e, *xc, "final") : me3_arm(e, *xc, "final"))) {
            return 1;
        }
        if (!a.ex_check.empty() && !me1_check(e, *xc, a)) {
            return 1;
        }
        if (!a.ex_kl.empty() && !me2_kl(e, *xc, a, me4)) {
            return 1;
        }
        if (!a.mx7_check.empty()) { // TEST (--mx7-check): MX7's model checks alone
            rt::mx7_check_args ma;
            ma.ids = a.mx7_check;
            ma.out_dir = a.out_dir;
            for (size_t p0 = 0; p0 <= a.mx7_shares.size();) {
                const size_t p1 = std::min(a.mx7_shares.find(',', p0), a.mx7_shares.size());
                if (p1 > p0) {
                    ma.shares.push_back(a.mx7_shares.substr(p0, p1 - p0));
                }
                p0 = p1 + 1;
            }
            ma.set_share = [xc](double s, std::string & e2) { return xc->set_prompt_cpu_share(s, e2); };
            ma.counters = [xc]() { return xc->request_json(); };
            ma.set_shadow = [xc](bool on) { xc->set_prompt_cpu_share_shadow(on); };
            ma.take_shadow = [xc]() { return xc->take_prompt_cpu_share_shadow(); };
            return rt::mx7_check(e, ma) ? 0 : 1;
        }
        // ME3: the loan and short-read checks, on the KL fixture's tokens (with the arms) or, dry, on synthetic ones
        std::vector<int32_t> kl;
        if (!arms.front().empty() && !a.ex_kl.empty() && (!read_ids(a.ex_kl, kl) || kl.size() < 3048)) {
            fprintf(stderr, "spec run: ME3 checks: %s has fewer than 3,048 tokens\n", a.ex_kl.c_str());
            return 1;
        }
        if (dry) {
            for (int32_t i = 0; i < 3048; ++i) {
                kl.push_back(1000 + (i * 7919) % 200000);
            }
        }
        const size_t n_kb = kl.empty() ? 0 : dry ? 300 : 2048, n_km = kl.empty() ? 0 : 1000;
        const std::vector<int32_t> kb(kl.begin(), kl.begin() + int64_t(n_kb));
        const std::vector<int32_t> km(kl.begin() + int64_t(n_kb), kl.begin() + int64_t(n_kb + n_km));
        if (a.me9_check) { // TEST (--me9-check): the admission check alone (one device)
            return !kb.empty() && me9_check(e, *xc, kb, km) ? 0 : 1;
        }
        if (a.me3_stall) { // TEST (--me3-stall): the stall check alone, then the teardown
            return !kb.empty() && me3_stall_check(e, *xc, kb) ? 0 : 1;
        }
        if (a.me3_cancel) { // TEST (--me3-cancel): the cancellation check alone, on prefill-test's 9,000 tokens' length
            std::vector<int32_t> p9;
            for (int32_t i = 0; i < 9000; ++i) {
                p9.push_back(1000 + (i * 7919) % 200000);
            }
            return me3_cancel_check(e, *xc, p9) ? 0 : 1;
        }
        if (!kl.empty() && (!me3_loan_check(e, *xc, kb, km) || !me3_short_check(e, *xc, kb, km) ||
                            !verify_recovery_check(e, *xc, kb) || !me10_xf_check(e))) {
            return 1;
        }
        // the checks' cache contents and counts do not reach the requests below (ME2: the profile state again)
        md::kv_tier * kt = e.test_target().tier();
        if (!xc->reset_to_profile(err) || !xc->set_mode(0, err) || (kt != nullptr && !kt->reset_counters(err))) {
            fprintf(stderr, "spec run: %s\n", err.c_str());
            return 1;
        }
        xc->reset_counters();
    }
    for (size_t p = 0; p < paths.size(); ++p) {
        printf("spec run: prompt %zu: %s (%zu tokens)%s\n", p, paths[p].c_str(), prompts[p].size(),
               append[p] ? ", an append to the previous request's state (ME3)" : "");
    }
    printf("spec run: %zu prompts x %zu settings x %d runs, each from a fresh prefill\n", paths.size(),
           settings.size(), a.runs);
    FILE * log = fopen((a.out_dir + "/run-summary.txt").c_str(), "w");
    if (log == nullptr) {
        fprintf(stderr, "spec run: cannot write %s/run-summary.txt\n", a.out_dir.c_str());
        return 1;
    }
    fprintf(log, "%s\n", e.settings().c_str());
    // SF1: one row a step of every request; ms from the step's start to each completion wait (empty: no such round)
    FILE * rc = fopen((a.out_dir + "/rounds.csv").c_str(), "w");
    if (rc == nullptr) {
        fprintf(stderr, "spec run: cannot write %s/rounds.csv\n", a.out_dir.c_str());
        fclose(log);
        return 1;
    }
    fprintf(rc, "setting,run,step,kind,n_before,d,a,n_out,cut,waits,to_draft_ms,to_verify_ms,to_catchup_ms,to_one_ms,"
                "wall_ms\n");
    const auto ms = [](double s_) {
        char b[32] = "";
        if (s_ >= 0) {
            snprintf(b, sizeof(b), "%.4f", s_ * 1e3);
        }
        return std::string(b);
    };
    bool all = true;
    // ME2: setting g of prompt p is numbered p x settings + g in the files and lines below (prompt 0's as before);
    // ME3: arm i's from i x prompts x settings
    const size_t per_arm = paths.size() * settings.size();
    for (size_t ai = 0; ai < arms.size(); ++ai) {
        if (!arms[ai].empty()) {
            if (!(me4 ? me4_arm(e, *xc, arms[ai]) : me3_arm(e, *xc, arms[ai]))) {
                return 1;
            }
            fprintf(log, "%s arm %s: requests %zu .. %zu\n", me4 ? "ME4" : "ME3", arms[ai].c_str(), ai * per_arm,
                    ai * per_arm + per_arm - 1);
        }
        for (size_t pg = 0; pg < per_arm; ++pg) {
            const size_t g = ai * per_arm + pg, p = pg / settings.size();
            const fnfl_target_args::run_setting & st = settings[pg % settings.size()];
            char what[384];
            snprintf(what, sizeof(what),
                     "setting %zu: T %.2f top-k %d top-p %.2f min-p %.2f seed %llu max_tokens %d; prompt %zu (%zu "
                     "tokens)",
                     g, double(st.sampling.temp), st.sampling.top_k, double(st.sampling.top_p),
                     double(st.sampling.min_p), static_cast<unsigned long long>(st.sampling.seed), st.max_tokens, p,
                     prompts[p].size());
            printf("spec run: %s\n", what);
            fprintf(log, "%s\n", what);
            std::vector<int32_t> first;
            std::vector<double> med, tps_engine, tps_driver;
            for (int32_t r = 0; r < a.runs; ++r) {
                rt::request_options o;
                o.sampling = st.sampling;
                o.max_tokens = st.max_tokens;
                const std::string clk0 = sm_clocks();
                fnfl_request_out  ro;
                std::vector<int32_t> req;
                int32_t n_reuse = 0;
                // ME3 (--next-append-rows): the IDs that make the read rows[p] rows with the slot's pending token
                std::vector<int32_t> ids = prompts[p];
                int32_t pend = -1;
                if (rows[p] > 0) {
                    std::vector<int32_t> hist;
                    if (e.history(0, hist, pend) != rt::run_status::ok) {
                        fprintf(stderr, "spec run: %s\n", e.error().c_str());
                        return 1;
                    }
                    ids.resize(size_t(rows[p] - (pend >= 0 ? 1 : 0)));
                }
                // ME4 (--append-close): the previous request ended at its budget, so its turn is closed here
                const bool closed = append[p] && a.append_close >= 0 && e.last_summary().ended == "budget";
                if (closed) {
                    printf("spec run: ME4 append: the previous request ended at its budget; its turn closed with ID %d "
                           "before the recorded IDs (one row more than recorded)\n",
                           a.append_close);
                }
                if (!me3_request_tokens(e, ids, append[p], req, n_reuse, closed ? a.append_close : -1) ||
                    !fnfl_spec_request(e, req, n_reuse, o, ro)) {
                    fprintf(stderr, "spec run: %s\n", e.error().c_str());
                    return 1;
                }
                if (rows[p] > 0) {
                    const bool is_short = e.last_prefill().short_read;
                    const bool want_short = rows[p] <= e.prefill_driver().short_read();
                    const bool rows_ok = ro.n_processed == rows[p] && is_short == want_short;
                    all = all && rows_ok;
                    printf("spec run: ME3 rows: a read of %d rows asked (the pending token %s, %zu IDs): %d processed, "
                           "%s (threshold %d: %s expected): %s\n",
                           rows[p], pend >= 0 ? "kept" : "none", ids.size(), ro.n_processed,
                           is_short ? "short" : "bulk", e.prefill_driver().short_read(), want_short ? "short" : "bulk",
                           rows_ok ? "as expected" : "UNEXPECTED");
                }
                // ME3: an append is a pure extension (only its new tokens processed), or the request fails
                if (append[p] && !me3_append_ok(ro, req.size(), n_reuse)) {
                    fprintf(stderr, "spec run: the append processed %d tokens, not its %d new ones (not a pure "
                                    "extension)\n", ro.n_processed, static_cast<int32_t>(req.size()) - n_reuse);
                    return 1;
                }
                if (append[p]) {
                    printf("spec run: ME3 append: %d tokens kept, %d new (the pending token and %zu IDs), %d "
                           "processed: a pure extension; populated depth %zu\n",
                           n_reuse, static_cast<int32_t>(req.size()) - n_reuse, ids.size(), ro.n_processed,
                           req.size());
                }
                const double                 pre_s = ro.pre_s, gen_s = ro.gen_s;
                const std::vector<int32_t> & toks = ro.toks;
                const std::string clk1 = sm_clocks();
                const rt::request_summary & sm = e.last_summary();
                printf("spec run: ME4 request %zu.%d ended: %s\n", g, r, sm.ended.c_str());
                fprintf(log, "request %zu.%d ended: %s\n", g, r, sm.ended.c_str());
                med.push_back(sm.wall_median_ms);
                tps_engine.push_back(sm.span_s > 0 ? (sm.tokens - 1) / sm.span_s : 0.0);
                tps_driver.push_back(gen_s > 0 ? (double(toks.size()) - 1) / gen_s : 0.0);
                const bool same = r == 0 || toks == first;
                if (r == 0) {
                    first = toks;
                }
                all = all && same;
                char line[512];
                snprintf(line, sizeof(line),
                         "run %zu.%d: %zu tokens; prefill %.3f s; driver: after the prefill to the last step %.4f "
                         "s, %.2f tokens/s ((tokens - 1) / time, the server's definition); tokens %s run 0's; SM "
                         "clocks before [%s] after [%s]",
                         g, r, toks.size(), pre_s, gen_s, tps_driver.back(), same ? "equal to" : "DIFFERENT from",
                         clk0.c_str(), clk1.c_str());
                printf("spec run: %s\nspec run: %s\n", line, e.last_summary_line().c_str());
                fprintf(log, "%s\n%s\n", line, e.last_summary_line().c_str());
                if (xc != nullptr) {
                    // ME3: the request's prefill path, loan and affinity, its boundaries and its cache line
                    const std::string m3 = me3_prefill_text(e, *xc) + "\n" + me3_boundaries(e, ro) +
                                           "\nexpert cache request: " + e.last_cache_line();
                    printf("spec run: %s\n", m3.c_str());
                    fprintf(log, "%s\n", m3.c_str());
                }
                if (xc != nullptr) {
                    // ME2: after the request (outside its timing): the last windows' times read, the pass
                    // committed, the learned profile saved (--ex-profile-save)
                    std::string derr;
                    if (!xc->drain(derr)) {
                        fprintf(stderr, "spec run: %s\n", derr.c_str());
                        return 1;
                    }
                    const std::string xs = "expert cache: " + xc->counters_json() + "; " + me1_memory(*xc);
                    printf("spec run: %s\n", xs.c_str());
                    fprintf(log, "%s\n", xs.c_str());
                    xc->reset_counters();
                }
                if (md::kv_tier * kt = e.test_target().tier()) { // ME1 (--kv-host): the KV moved for this request
                    std::string kerr;
                    if (!kt->read_counters(kerr)) {
                        fprintf(stderr, "spec run: %s\n", kerr.c_str());
                        return 1;
                    }
                    printf("spec run: %s\n", kt->counters_text().c_str());
                    fprintf(log, "%s\n", kt->counters_text().c_str());
                    if (!kt->reset_counters(kerr)) {
                        fprintf(stderr, "spec run: %s\n", kerr.c_str());
                        return 1;
                    }
                }
                write_tokens(a.out_dir + "/run-" + std::to_string(g) + "-" + std::to_string(r) + "-tokens.txt", toks);
                int32_t k = 0;
                for (const rt::step_info & x : e.last_steps()) {
                    fprintf(rc, "%zu,%d,%d,%s,%lld,%d,%d,%d,%d,%d,%s,%s,%s,%s,%.4f\n", g, r, k++,
                            x.kind == 0   ? "bootstrap"
                            : x.kind == 1 ? "generate"
                                          : "spec",
                            static_cast<long long>(x.n_before), x.d, x.a, x.n_out, x.cut ? 1 : 0, x.waits,
                            ms(x.to_draft_s).c_str(), ms(x.to_verify_s).c_str(), ms(x.to_catchup_s).c_str(),
                            ms(x.to_one_s).c_str(), x.wall_s * 1e3);
                }
                fflush(rc);
                fflush(stdout);
                fflush(log);
            }
            const auto spread = [](const std::vector<double> & v) {
                return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end());
            };
            char sum[512];
            snprintf(sum, sizeof(sum),
                     "setting %zu over %d runs: median step wall ms median %.4f (max-min %.4f); tokens/s (engine "
                     "span, (tokens - 1) / span) median %.3f (max-min %.3f); tokens/s (driver, the server's "
                     "definition) median %.3f (max-min %.3f); tokens equal across runs: %s",
                     g, a.runs, median_of(med), spread(med), median_of(tps_engine), spread(tps_engine),
                     median_of(tps_driver), spread(tps_driver), all ? "yes" : "NO");
            printf("spec run: %s\n", sum);
            fprintf(log, "%s\n", sum);
        }
    }
    fclose(rc);
    if (dry) {
        // the other variant at the first setting's top-k (TK1), or the default when that setting is off the top-k path
        const int32_t k = a.run_settings.front().sampling.top_k;
        all = run_dry_paths(e, prompt, k > 0 ? k : sp::options().top_k, log) && all;
    }
    const int32_t captures1 = e.test_target().captures();
    printf("spec run: ME3 executor: %d graph captures at the end, %d after init (rebuilds)\n", captures1,
           captures1 - captures0);
    all = all && captures1 == captures0;
    fclose(log);
    printf("spec run: %s\n", all ? "PASS" : "FAILED");
    return all ? 0 : 1;
}

namespace {

// ---- SP2, spec ab (from HC8's c57f527): the checks, then the decode-ready A/B ----

constexpr int32_t ab_d_base = 2, ab_d_cand = 4; // the base arm: d = 2, dense; the candidate: d = 4, sparse

// K3's and the measured part's arms (SP2 Amendment 3, --arms; expires at SP2's disposition): d and the draft form;
// SP4: the target's expert gate/up mapping (target::set_expert_map; expires at SP4's disposition)
struct ab_arm {
    int32_t d = 0;
    bool sparse = false;
    int32_t ex_map = 0; // SP4: 0 SP2's, 1 SP4's (H1), 2 H2's; XB3: 3 C1's
    std::string name;   // "d4-sparse", with ":EX" given "d4-sparse-sp4" (or -h2, -xb3)
};

// "D:FORM[:EX][,D:FORM[:EX]..]", D 1..4, FORM dense or sparse, EX sp2, sp4, h2 or xb3 (SP4, XB3; sp2 unless given),
// 1..8 arms; empty: the base and candidate arms
bool ab_arms(const std::string & spec, std::vector<ab_arm> & arms) {
    arms.clear();
    const std::string v = spec.empty() ? std::to_string(ab_d_base) + ":dense," + std::to_string(ab_d_cand) + ":sparse"
                                       : spec;
    size_t p = 0;
    while (p <= v.size()) {
        const size_t q = std::min(v.find(',', p), v.size());
        const std::string item = v.substr(p, q - p);
        const size_t c = item.find(':');
        const size_t c2 = item.find(':', c == std::string::npos ? c : c + 1);
        const std::string form = item.substr(c + 1, c2 == std::string::npos ? c2 : c2 - c - 1);
        const std::string ex = c2 == std::string::npos ? "" : item.substr(c2 + 1);
        if (c != 1 || item[0] < '1' || item[0] > '4' || (form != "dense" && form != "sparse") ||
            (c2 != std::string::npos && ex != "sp2" && ex != "sp4" && ex != "h2" && ex != "xb3")) {
            return false;
        }
        ab_arm x;
        x.d = item[0] - '0';
        x.sparse = form == "sparse";
        x.ex_map = ex == "sp4" ? 1 : ex == "h2" ? 2 : ex == "xb3" ? 3 : 0;
        x.name = "d" + item.substr(0, 1) + "-" + form + (ex.empty() ? "" : "-" + ex);
        arms.push_back(x);
        p = q + 1;
    }
    return !arms.empty() && arms.size() <= 8;
}

// a check's device rows on both cards: the logits [5][124,160], the hidden carry [5][4][1280] and the draft's
// retained carry [4][1280], with the state hashes (the draft index's included)
struct ab_rows {
    std::vector<float> logits[2], carry[2], draft_carry[2];
    state_hash h;
};

bool ab_capture(rt::engine & e, ab_rows & x) {
    rt::target & t = e.test_target();
    md::model & m = e.test_model();
    if (!snapshot(t, m, x.h)) {
        return false;
    }
    const size_t rows = size_t(rt::max_step_tokens);
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !get(x.logits[c], static_cast<const float *>(t.card(c).logits), rows * VC) ||
            !get(x.carry[c], static_cast<const float *>(t.card(c).carry), rows * 4 * 1280) ||
            !get(x.draft_carry[c], static_cast<const float *>(t.draft(c).carry_prev), size_t(4) * 1280)) {
            return false;
        }
    }
    return true;
}

// the parts that differ bitwise ("" when none)
std::string ab_diff(const ab_rows & a, const ab_rows & b) {
    std::string d;
    const auto bits = [](const std::vector<float> & x, const std::vector<float> & y) {
        return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * 4) == 0);
    };
    for (int32_t c = 0; c < 2; ++c) {
        const std::string card = " (card " + std::to_string(c) + ")";
        d += bits(a.logits[c], b.logits[c]) ? "" : " logits" + card;
        d += bits(a.carry[c], b.carry[c]) ? "" : " carry" + card;
        d += bits(a.draft_carry[c], b.draft_carry[c]) ? "" : " draft-carry" + card;
    }
    d += a.h.n == b.h.n && a.h.draft_n == b.h.draft_n ? "" : " N or draft N";
    d += a.h.index_n == b.h.index_n ? "" : " index_n";
    d += a.h.index_cards_equal && b.h.index_cards_equal ? "" : " draft index cards differ";
    for (int32_t i = 0; i < n_hash; ++i) {
        d += a.h.h[i] == b.h.h[i] ? "" : std::string(" ") + hash_names[i];
    }
    return d;
}

// one request from the saved decode-ready state with d drafts in the given draft attention form and expert mapping
// (SP4: a changed mapping recaptures the target's B = 4 and 5 graphs first, outside the clock): its steps until an
// end of generation or max_tokens, as the server drives them; wall_s: the driver's clock around the steps (the last
// step's closing check included)
bool ab_request(rt::engine & e, const rt::ready_state & rs, const ab_arm & arm, const rt::request_options & o,
                std::vector<int32_t> & toks, std::vector<rt::step_info> & steps, rt::request_summary & sum,
                double & wall_s, std::string & err) {
    toks.clear();
    if (!e.test_target().set_expert_map(arm.ex_map, err)) {
        return false;
    }
    const int32_t d = arm.d;
    const bool sparse = arm.sparse;
    if (e.set_draft(d, sparse) != rt::run_status::ok || e.set_sampling(0, o) != rt::run_status::ok ||
        e.restore_ready(rs) != rt::run_status::ok) {
        return fail_msg(err, e.error());
    }
    if (e.test_target().index_n() != rs.index_n) {
        return fail_msg(err, "restore: index_n " + std::to_string(e.test_target().index_n()) + ", saved " +
                                 std::to_string(rs.index_n));
    }
    int32_t out[rt::max_step_tokens] = {};
    int32_t n = 0;
    bool end = false;
    const auto t0 = clk::now();
    while (!end && static_cast<int32_t>(toks.size()) < o.max_tokens) {
        if (e.step(0, out, n, end) != rt::run_status::ok) {
            return fail_msg(err, e.error());
        }
        toks.insert(toks.end(), out, out + n);
    }
    wall_s = since(t0);
    steps = e.last_steps();
    sum = e.last_summary();
    return true;
}

bool same_step(const rt::step_info & x, const rt::step_info & y) {
    bool same = x.kind == y.kind && x.n_before == y.n_before && x.d == y.d && x.a == y.a && x.n_out == y.n_out;
    for (int32_t j = 0; same && j < x.n_out; ++j) {
        same = x.out[j] == y.out[j];
    }
    return same;
}

double mean_of(const std::vector<double> & v) {
    double s = 0;
    for (const double x : v) {
        s += x;
    }
    return v.empty() ? 0.0 : s / double(v.size());
}

// an arm's report (measured: the host clock the steps read; counted: tokens, steps, acceptance): delivered tokens over
// the decode span, the full-d speculative steps' wall and segments, the accepted-prefix histogram and per proposal
// position the steps that reached it (a >= j - 1) and accepted it (a >= j); the bootstrap and the tails (generate
// rounds and speculative steps of fewer than d drafts, the budget's end) apart
std::string ab_report(const char * arm, int32_t d, const std::vector<int32_t> & toks,
                      const std::vector<rt::step_info> & steps, const rt::request_summary & sum, double wall_s) {
    std::vector<double> wall, draft, verify, catchup, after;
    int32_t hist[rt::max_step_tokens] = {}, reached[rt::max_spec_d] = {}, accepted[rt::max_spec_d] = {};
    int32_t full_tokens = 0;
    std::string tails, boot;
    char b[256];
    for (const rt::step_info & x : steps) {
        if (x.kind == 2 && x.d == d) {
            wall.push_back(x.wall_s * 1e3);
            draft.push_back(x.to_draft_s * 1e3);
            verify.push_back((x.to_verify_s - x.to_draft_s) * 1e3);
            catchup.push_back((x.to_catchup_s - x.to_verify_s) * 1e3);
            after.push_back((x.wall_s - x.to_catchup_s) * 1e3);
            hist[std::min(d, std::max(0, x.a))]++;
            for (int32_t j = 0; j < d; ++j) {
                reached[j] += x.a >= j ? 1 : 0;
                accepted[j] += x.a >= j + 1 ? 1 : 0;
            }
            full_tokens += x.n_out;
        } else if (x.kind == 0) {
            snprintf(b, sizeof(b), "%.3f ms", x.wall_s * 1e3);
            boot += b;
        } else {
            snprintf(b, sizeof(b), "%s[%s d %d a %d n_out %d %.3f ms]", tails.empty() ? "" : " ",
                     x.kind == 1 ? "generate" : "spec", x.d, x.a, x.n_out, x.wall_s * 1e3);
            tails += b;
        }
    }
    std::string pos;
    for (int32_t j = 0; j < d; ++j) {
        snprintf(b, sizeof(b), "%s%d: %d/%d", j ? ", " : "", j + 1, accepted[j], reached[j]);
        pos += b;
    }
    const double n = static_cast<double>(toks.size());
    char line[2048];
    snprintf(line, sizeof(line),
             "%s (d %d): delivered %zu tokens; decode span %.4f s (the bootstrap's start to the last step's end): "
             "%.3f tokens/s (delivered / span; (tokens - 1) / span %.3f); driver wall around the steps %.4f s "
             "(with the last step's closing check): %.3f tokens/s; ended %s; steps %d (bootstrap %d, speculative %d, "
             "generate %d), depth before a step %lld..%lld | full-d speculative steps %zu, %d tokens (%.3f a step): "
             "step wall ms median %.3f mean %.3f; segments ms median (mean): draft %.3f (%.3f), verify %.3f (%.3f), "
             "catch-up %.3f (%.3f), after the catch-up's wait %.3f (%.3f) | accepted prefix a = 0..%d: %s | per "
             "proposal position, accepted/reached: %s | bootstrap: %s | tails: %s",
             arm, d, toks.size(), sum.span_s, sum.span_s > 0 ? n / sum.span_s : 0.0,
             sum.span_s > 0 ? (n - 1) / sum.span_s : 0.0, wall_s, wall_s > 0 ? n / wall_s : 0.0, sum.ended.c_str(),
             sum.steps, sum.boot, sum.spec, sum.gen, static_cast<long long>(sum.n_before_min),
             static_cast<long long>(sum.n_before_max), wall.size(), full_tokens,
             wall.empty() ? 0.0 : double(full_tokens) / double(wall.size()), median_of(wall), mean_of(wall),
             median_of(draft), mean_of(draft), median_of(verify), mean_of(verify), median_of(catchup),
             mean_of(catchup), median_of(after), mean_of(after), d, list_of(hist, d + 1, "/").c_str(), pos.c_str(),
             boot.empty() ? "none" : boot.c_str(), tails.empty() ? "none" : tails.c_str());
    return line;
}

} // namespace

int fnfl_spec_ab(const fnfl_target_args & a) {
    const bool dry = a.mode == "ab-dry";
    std::vector<ab_arm> arms;
    if (!ab_arms(a.arms, arms)) {
        fprintf(stderr,
                "spec ab: --arms %s: D:FORM[:EX][,D:FORM[:EX]..], D 1..4, FORM dense or sparse, EX sp2, sp4, h2 or "
                "xb3, at most 8 arms\n",
                a.arms.c_str());
        return 1;
    }
    std::string arm_list;
    for (const ab_arm & x : arms) {
        arm_list += (arm_list.empty() ? "" : ", ") + x.name;
    }
    std::vector<int32_t> history;
    if (!read_ids(a.ids, history)) {
        return 1;
    }
    // dry: zero weights, 8,192 cells: 2,100 history tokens (525 whole blocks: more than the 512 selected)
    const size_t n_check = dry ? 40 : 2600; // K1 and K2's prompt: 650 whole blocks, more than 512
    if (dry) {
        history.resize(std::min<size_t>(history.size(), 2100));
    }
    if (history.size() < n_check || a.run_settings.size() != 1) {
        fprintf(stderr, "spec ab: the history holds %zu tokens (at least %zu), %zu settings (one)\n", history.size(),
                n_check, a.run_settings.size());
        return 1;
    }
    const std::vector<int32_t> check_prompt(history.begin(), history.begin() + std::ptrdiff_t(n_check));
    const int32_t n_k1 = dry ? 2 : 8, n_k2 = dry ? 4 : 32, k3_tokens = dry ? 8 : 16;
    rt::request_options o;
    o.sampling = a.run_settings[0].sampling;
    o.sampling.eos = eos_tokenizer;
    o.max_tokens = dry ? std::min(a.run_settings[0].max_tokens, 32) : a.run_settings[0].max_tokens;
    rt::draft_options dopt;
    if (!fnfl_draft_settings(a, true, "spec ab", dopt)) {
        return 1;
    }
    if (dopt.attention != "sparse" || o.max_tokens < 1 || o.max_tokens > 256) {
        fprintf(stderr, "spec ab: needs --draft-attention sparse (both forms captured) and --max-tokens 1..256\n");
        return 1;
    }
    const int32_t cells = dry ? 8192 : a.n_cells;
    auto m = std::make_unique<md::model>();
    if (!(dry ? fnfl_load_dry(a, cells, *m) : fnfl_load(a, cells, *m, "spec ab"))) {
        return 1;
    }
    rt::engine_config c;
    c.model_path = a.model_path;
    c.n_ctx_slot = cells;
    c.draft = dopt;
    c.eog = a.eog.empty() ? std::vector<int32_t>{eos_tokenizer} : a.eog;
    c.timeout_s = dry ? 30.0 : 60.0;
    c.spec_d = ab_d_base;
    rt::engine e;
    std::string err;
    if (!e.init(std::move(m), c, err)) {
        fprintf(stderr, "spec ab: engine: %s\n", err.c_str());
        return 1;
    }
    FILE * log = fopen((a.out_dir + "/ab-summary.txt").c_str(), "w");
    if (log == nullptr) {
        fprintf(stderr, "spec ab: cannot write %s/ab-summary.txt\n", a.out_dir.c_str());
        return 1;
    }
    bool all = true;
    const auto say = [&](const std::string & s) {
        printf("spec ab: %s\n", s.c_str());
        fprintf(log, "%s\n", s.c_str());
        fflush(stdout);
        fflush(log);
    };
    const auto verdict = [&](const char * name, bool ok, const std::string & what) {
        say(std::string(name) + ": " + (ok ? "PASS" : "FAIL") + ": " + what);
        all = all && ok;
    };
    rt::target & t = e.test_target();
    md::model & mm = e.test_model();
    const auto closing = [&](const char * name) {
        bool checked = false;
        const bool ok = t.close_check(checked, err);
        verdict(name, ok && checked, ok ? (checked ? "checked" : "not checked") : err);
    };
    char what[1024];
    snprintf(what, sizeof(what),
             "%s; history %s (%zu tokens%s); setting T %.2f top-k %d top-p %.2f min-p %.2f seed %llu max_tokens %d; "
             "arms %s; checks (d %d sparse): K1 %d recorded rounds and K2 %d replayed rounds "
             "on the history's first %zu tokens (candidate form, through end-of-generation), K3 %d-token requests "
             "from the saved state twice per arm",
             e.settings().c_str(), a.ids.c_str(), history.size(), dry ? ", dry: zero weights, cut to 2,100" : "",
             double(o.sampling.temp), o.sampling.top_k, double(o.sampling.top_p), double(o.sampling.min_p),
             static_cast<unsigned long long>(o.sampling.seed), o.max_tokens, arm_list.c_str(), ab_d_cand, n_k1, n_k2,
             n_check, k3_tokens);
    say(what);
    // K1 and K2's check requests continue through end-of-generation (SP2 Amendment 1): the sampler's eos -1 (no
    // verify cut) and spec_run's through_eog; the normal options again before K4 and the measured part
    rt::request_options oc = o;
    oc.sampling.eos = -1;
    if (e.set_draft(ab_d_cand, true) != rt::run_status::ok || e.set_sampling(0, oc) != rt::run_status::ok) {
        fprintf(stderr, "spec ab: %s\n", e.error().c_str());
        return 1;
    }
    // SP4: K1 and K2 run in the last arm's expert mapping (the candidate's with the default order)
    if (!t.set_expert_map(arms.back().ex_map, err)) {
        fprintf(stderr, "spec ab: %s\n", err.c_str());
        return 1;
    }
    say(std::string("K1 and K2 expert mapping: ") +
        (t.expert_map() == 1   ? "sp4"
         : t.expert_map() == 2 ? "h2"
         : t.expert_map() == 3 ? "xb3"
                               : "sp2"));
    double pre_s = 0, secs = 0;
    bool eos = false;

    // K1: C4d at d = 4 in the sparse form: 8 recorded rounds against the CPU reference (proposals and acceptance)
    {
        std::vector<int32_t> ids[2];
        for (int32_t cd = 0; cd < 2; ++cd) {
            ids[cd].resize(size_t(NV));
            if (!cuda_ok(cudaSetDevice(mm.device[cd]), "device") ||
                !cuda_ok(cudaMemcpy(ids[cd].data(), t.draft(cd).head_ids, size_t(NV) * 4, cudaMemcpyDeviceToHost),
                         "ID map")) {
                return 1;
            }
        }
        std::vector<c4d_round> rec;
        spec_hooks hk;
        hk.after_draft = [&](int32_t, const rt::round_result & r) {
            rec.emplace_back();
            rec.back().n_before = t.slot().n;
            for (int32_t j = 0; j < r.rows; ++j) {
                rec.back().prop[j] = r.argmax[0][j];
            }
            return record_draft(t, mm, r.rows, rec.back());
        };
        hk.after_round = [&](int32_t, const spec_round & s) {
            c4d_round & x = rec.back();
            x.a = s.a;
            x.n_out = s.n_out;
            std::memcpy(x.out, s.out, sizeof(x.out));
            return record_verify(t, mm, x);
        };
        std::vector<int32_t> toks;
        std::vector<spec_round> rounds;
        const bool ok = prefill(t, check_prompt, pre_s, err) &&
                        spec_run(t, ab_d_cand, 1 << 20, n_k1, 1e9, toks, rounds, secs, eos, err, &hk, true);
        if (!ok) {
            verdict("K1", false, err);
            return 1;
        }
        closing("K1 closing check");
        const c4d_result cr = check_c4d(t.sampling(), rec, ids, log);
        snprintf(what, sizeof(what),
                 "%d rounds at d %d, sparse (a = 0..%d: %s), %d draft picks: q lists differing from the reference "
                 "%d, cards' q lists differ %d, proposals same-uniform mismatches %d (listed %d); acceptance "
                 "mismatches %d (listed %d); index_n %lld at N %lld",
                 cr.rounds, ab_d_cand, ab_d_cand, list_of(cr.by_a, ab_d_cand + 1, "/").c_str(), cr.picks, cr.q_bad,
                 cr.q_cards_differ, cr.pick_mism, cr.pick_listed, cr.acc_mism, cr.acc_listed,
                 static_cast<long long>(t.index_n()), static_cast<long long>(t.slot().n));
        verdict("K1 producer and acceptance (C4d at d = 4)", cr.pass() && cr.rounds == n_k1, what);
    }

    // K2: C4e at d = 4 in the sparse form: each round's state (the draft index's keys and summaries, both cards
    // equal, included) against a clean forced replay of the committed rows in the same grouping, bitwise
    {
        std::vector<state_hash> got;
        spec_hooks hk;
        hk.after_round = [&](int32_t, const spec_round &) {
            got.emplace_back();
            return snapshot(t, mm, got.back());
        };
        std::vector<int32_t> toks;
        std::vector<spec_round> rounds;
        bool ok = prefill(t, check_prompt, pre_s, err) &&
                  spec_run(t, ab_d_cand, 1 << 20, n_k2, 1e9, toks, rounds, secs, eos, err, &hk, true);
        if (!ok) {
            verdict("K2", false, err);
            return 1;
        }
        closing("K2 closing check");
        const std::vector<int32_t> ledger = t.slot().tok;
        const int64_t n_final = t.slot().n;
        ok = t.reset(err);
        t.set_prompt(std::vector<int32_t>(ledger.begin(), ledger.begin() + n_final));
        rt::round_result r;
        while (ok && t.slot().n < static_cast<int64_t>(check_prompt.size())) {
            const int32_t B = static_cast<int32_t>(std::min<int64_t>(4, int64_t(check_prompt.size()) - t.slot().n));
            ok = t.forced(B, r, err) && t.draft_catchup(B, r, err);
        }
        int32_t equal = 0, unequal = 0, rejected = 0, by_n[rt::max_step_tokens + 1] = {};
        std::string bad;
        size_t gi = 0;
        for (size_t i = 1; ok && i < rounds.size(); ++i) {
            const spec_round & s = rounds[i];
            const int32_t rows = s.n_out;
            rejected += s.kind == 2 && s.a < s.d ? 1 : 0;
            if (t.slot().n + rows > n_final) {
                break; // the last round's last token is pending (no state); its rows end at n_final
            }
            by_n[std::min(rt::max_step_tokens, rows)]++;
            ok = t.forced(rows, r, err) && t.draft_catchup(rows, r, err);
            state_hash h;
            ok = ok && gi < got.size() && snapshot(t, mm, h);
            if (!ok) {
                break;
            }
            const state_hash & g = got[gi++];
            bool same = g.n == h.n && g.draft_n == h.draft_n && g.index_n == h.index_n && g.index_cards_equal &&
                        h.index_cards_equal;
            for (int32_t k = 0; k < n_hash; ++k) {
                if (g.h[k] != h.h[k]) {
                    same = false;
                    bad += std::string(" round ") + std::to_string(i - 1) + ":" + hash_names[k];
                }
            }
            (same ? equal : unequal)++;
            fprintf(log, "K2 round %zu: N %lld draft N %lld index_n %lld a %d rows %d: %s\n", i - 1,
                    static_cast<long long>(h.n), static_cast<long long>(h.draft_n), static_cast<long long>(h.index_n),
                    s.a, rows, same ? "bitwise equal" : "DIFFERENT");
        }
        if (!ok) {
            verdict("K2 replay", false, err);
            return 1;
        }
        const bool pass = unequal == 0 && equal == static_cast<int32_t>(got.size()) && (rejected > 0 || dry);
        snprintf(what, sizeof(what),
                 "%d of %zu rounds' states bitwise equal to the clean forced replay (the target's classes, draft K, "
                 "V and carry, the draft index's keys and summaries; both cards' draft index equal), replayed rows "
                 "1..5: %s; rounds with a rejection %d%s%s",
                 equal, got.size(), list_of(by_n + 1, rt::max_step_tokens, "/").c_str(), rejected,
                 rejected > 0 ? ""
                 : dry        ? " (zero weights: p = q, information)"
                              : " -- MISSING COVERAGE, not a pass",
                 unequal ? (std::string("; differing:") + bad).c_str() : "");
        verdict("K2 rejection replay (C4e at d = 4)", pass, what);
    }
    if (e.set_sampling(0, o) != rt::run_status::ok || t.sampling().eos != eos_tokenizer) {
        verdict("normal end-of-generation restored", false, e.error());
        return 1;
    }

    // K4 (a diagnostic, no pass bound: SP2 Amendment 1): the bulk prompt path's draft state (fn/model/prompt.cpp)
    // against the decode path's (K2's forced rounds and catch-ups over the same tokens), both cards. The two paths
    // feed the draft different target rows (NC1's prompt-against-decode difference), so no bound separates a defect
    // here; prefill-test vcheck's sparse form checks the arithmetic on identical inputs. Per cell (per block for the
    // summaries) max |prompt - decode| / max |decode| (vcheck's cache_error metric): p50, p99, max, the count over
    // 2e-2 and the first such cell, for the index keys, the summaries and the draft K and V
    {
        const int64_t n = static_cast<int64_t>(n_check);
        const int32_t nb = fn::qsa::blocks_complete(static_cast<int32_t>(n));
        const int32_t hd = fn::qsa::head_dim;
        struct k4_state {
            std::vector<uint16_t> k[2], kv[2], vv[2];
            std::vector<float> s[2];
        } dec, pro;
        const auto capture = [&](k4_state & x) {
            bool ok = t.index_n() >= n && t.draft_n() >= n && sync_cards(t, mm);
            for (int32_t c = 0; c < 2 && ok; ++c) {
                const md::draft_card & dc = t.draft(c);
                ok = cuda_ok(cudaSetDevice(mm.device[c]), "device") &&
                     get(x.k[c], static_cast<const uint16_t *>(dc.idx_keys), size_t(n) * hd) &&
                     get(x.s[c], static_cast<const float *>(dc.idx_summ), fn::qsa::summary_bytes(nb) / 4) &&
                     get(x.kv[c], static_cast<const uint16_t *>(dc.k_cache), size_t(n) * 256) &&
                     get(x.vv[c], static_cast<const uint16_t *>(dc.v_cache), size_t(n) * 256);
            }
            return ok;
        };
        int32_t np = 0;
        const bool ok = capture(dec) && e.truncate(0, 0) == rt::run_status::ok &&
                        e.prefill(0, check_prompt.data(), static_cast<int32_t>(n), 0, np) == rt::run_status::ok &&
                        capture(pro);
        if (!ok) {
            verdict("K4", false,
                    "index_n " + std::to_string(t.index_n()) + ", draft N " + std::to_string(t.draft_n()) + ": " +
                        e.error());
            return 1;
        }
        const auto f16 = [](uint16_t u) { return double(fn::quant::f16_to_f32(u)); };
        const auto bf = [](uint16_t u) { return double(fn::quant::bf16_to_f32(u)); };
        const auto summ = [&](const std::vector<float> & f, int32_t b, int32_t e_) {
            return double(f[size_t(b / 32) * 4096 + size_t(e_ / 4) * 128 + size_t(b % 32) * 4 + size_t(e_ % 4)]);
        };
        // one quantity: per unit (cell or block) of both cards the relative error, then its line
        const auto line = [&](const char * name, int64_t units, int32_t width,
                              const std::function<double(int32_t, int64_t, int32_t, bool)> & at) {
            std::vector<double> r;
            int64_t over = 0, bits = 0;
            std::string first = "none";
            bool finite = true;
            for (int32_t c = 0; c < 2; ++c) {
                for (int64_t i = 0; i < units; ++i) {
                    double dm = 0, ym = 0;
                    bool same = true;
                    for (int32_t j = 0; j < width; ++j) {
                        const double x = at(c, i, j, true), y = at(c, i, j, false);
                        finite = finite && std::isfinite(x) && std::isfinite(y);
                        dm = std::max(dm, std::fabs(x - y));
                        ym = std::max(ym, std::fabs(y));
                        same = same && x == y;
                    }
                    const double rel = ym > 0 ? dm / ym : dm;
                    r.push_back(rel);
                    bits += same ? 1 : 0;
                    if (rel > 2e-2 && over++ == 0) {
                        first = "card " + std::to_string(c) + " " + std::to_string(i) + " (" + std::to_string(rel) +
                                ")";
                    }
                }
            }
            std::sort(r.begin(), r.end());
            const auto q = [&](double p) {
                const double a = p / 100.0 * double(r.size() - 1);
                const size_t lo = static_cast<size_t>(std::floor(a)), hi = std::min(lo + 1, r.size() - 1);
                return r[lo] + (r[hi] - r[lo]) * (a - double(lo));
            };
            char s[512];
            snprintf(s, sizeof(s),
                     "%s: %lld x 2 cards: p50 %.3g p99 %.3g max %.3g; over 2e-2 %lld, the first %s; bitwise equal "
                     "%lld%s",
                     name, static_cast<long long>(units), q(50), q(99), q(100), static_cast<long long>(over),
                     first.c_str(), static_cast<long long>(bits), finite ? "" : "; NOT FINITE");
            return std::string(s);
        };
        const std::string lk = line("index keys (cells)", n, hd, [&](int32_t c, int64_t i, int32_t j, bool p) {
            return bf((p ? pro : dec).k[c][size_t(i) * hd + size_t(j)]);
        });
        const std::string ls = line("summaries (blocks)", nb, hd, [&](int32_t c, int64_t i, int32_t j, bool p) {
            return summ((p ? pro : dec).s[c], static_cast<int32_t>(i), j);
        });
        const std::string lkv = line("draft K (cells)", n, 256, [&](int32_t c, int64_t i, int32_t j, bool p) {
            return f16((p ? pro : dec).kv[c][size_t(i) * 256 + size_t(j)]);
        });
        const std::string lvv = line("draft V (cells)", n, 256, [&](int32_t c, int64_t i, int32_t j, bool p) {
            return f16((p ? pro : dec).vv[c][size_t(i) * 256 + size_t(j)]);
        });
        say("K4 prompt path against decode path (diagnostic, no pass bound; per unit max |prompt - decode| / max "
            "|decode|): " +
            lk + "; " + ls + "; " + lkv + "; " + lvv);
    }

    // the history's one bulk prefill in the candidate form (the draft index maintained), then its decode-ready state
    rt::ready_state rs;
    int32_t np = 0;
    const auto tp = clk::now();
    if (e.truncate(0, 0) != rt::run_status::ok ||
        e.prefill(0, history.data(), static_cast<int32_t>(history.size()), 0, np) != rt::run_status::ok ||
        e.save_ready(rs) != rt::run_status::ok) {
        verdict("prefill and save", false, e.error());
        return 1;
    }
    snprintf(what, sizeof(what), "%d tokens in %.3f s (measured, incidental); N %lld, draft N %lld, index_n %lld",
             np, since(tp), static_cast<long long>(rs.l.n), static_cast<long long>(rs.draft_n),
             static_cast<long long>(rs.index_n));
    verdict("prefill and save", rs.index_n == rs.l.n && rs.draft_n == rs.l.n, what);
    say("prefill phase line: " + e.last_phase_line());
    ab_rows saved;
    if (!ab_capture(e, saved)) {
        verdict("K3", false, "a device read failed");
        return 1;
    }

    // K3: the saved state restores bitwise: per arm two short requests equal (tokens, steps, end rows and state), the
    // arms in reverse order; after them a restore in the candidate form gives the saved rows and state again; the end
    // index_n shows a dense arm's catch-ups leave the draft index unextended
    {
        rt::request_options os = o;
        os.max_tokens = k3_tokens;
        const size_t na = arms.size();
        std::vector<std::vector<int32_t>> tk(na * 2);
        std::vector<std::vector<rt::step_info>> st(na * 2);
        std::vector<ab_rows> end(na * 2);
        rt::request_summary sm;
        double w = 0;
        for (size_t i = na; i-- > 0;) { // with the default arms: the candidate, then the base
            for (size_t rep = 0; rep < 2; ++rep) {
                if (!ab_request(e, rs, arms[i], os, tk[i * 2 + rep], st[i * 2 + rep], sm, w, err) ||
                    !ab_capture(e, end[i * 2 + rep])) {
                    verdict("K3", false, err.empty() ? "a device read failed" : err);
                    return 1;
                }
            }
        }
        // the first arm ran last before this restore of the candidate form
        ab_rows back;
        if (e.set_draft(ab_d_cand, true) != rt::run_status::ok || e.restore_ready(rs) != rt::run_status::ok ||
            !ab_capture(e, back)) {
            verdict("K3", false, e.error());
            return 1;
        }
        for (size_t i = 0; i < na; ++i) {
            const std::vector<rt::step_info> & s0 = st[i * 2];
            const std::vector<rt::step_info> & s1 = st[i * 2 + 1];
            bool steps_same = s0.size() == s1.size();
            for (size_t j = 0; steps_same && j < s0.size(); ++j) {
                steps_same = same_step(s0[j], s1[j]);
            }
            const std::string d = ab_diff(end[i * 2], end[i * 2 + 1]);
            const bool toks_same = tk[i * 2] == tk[i * 2 + 1];
            snprintf(what, sizeof(what),
                     "%s arm: %zu and %zu tokens %s, %zu steps %s, end rows and state %s%s; index_n at the end %lld",
                     arms[i].name.c_str(), tk[i * 2].size(), tk[i * 2 + 1].size(), toks_same ? "equal" : "DIFFERENT",
                     s0.size(), steps_same ? "equal" : "DIFFERENT", d.empty() ? "bitwise equal" : "DIFFERENT:",
                     d.c_str(), static_cast<long long>(end[i * 2 + 1].h.index_n));
            verdict(("K3 restore (" + arms[i].name + ")").c_str(), toks_same && steps_same && d.empty(), what);
            write_tokens(a.out_dir + "/k3-" + arms[i].name + "-tokens.txt", tk[i * 2]);
        }
        // SP4: arms that differ only in the expert mapping give the same tokens, steps, end rows and state (the
        // mapping keeps the arithmetic)
        for (size_t i = 0; i < na; ++i) {
            for (size_t j = i + 1; j < na; ++j) {
                if (arms[i].d != arms[j].d || arms[i].sparse != arms[j].sparse || arms[i].ex_map == arms[j].ex_map) {
                    continue;
                }
                const std::vector<rt::step_info> & si = st[i * 2];
                const std::vector<rt::step_info> & sj = st[j * 2];
                bool steps_same = si.size() == sj.size();
                for (size_t k = 0; steps_same && k < si.size(); ++k) {
                    steps_same = same_step(si[k], sj[k]);
                }
                const std::string dd = ab_diff(end[i * 2], end[j * 2]);
                const bool toks_same = tk[i * 2] == tk[j * 2];
                snprintf(what, sizeof(what), "%s against %s: tokens %s, steps %s, end rows and state %s%s",
                         arms[i].name.c_str(), arms[j].name.c_str(), toks_same ? "equal" : "DIFFERENT",
                         steps_same ? "equal" : "DIFFERENT", dd.empty() ? "bitwise equal" : "DIFFERENT:",
                         dd.c_str());
                verdict("K3 expert mappings bitwise (SP4)", toks_same && steps_same && dd.empty(), what);
            }
        }
        const std::string d = ab_diff(saved, back);
        verdict("K3 restore after the requests", d.empty(),
                d.empty() ? "the saved rows and state (the draft index's keys of cells < " +
                                std::to_string(rs.index_n) + " and summaries included) bitwise equal after the arms"
                          : "DIFFERENT:" + d);
    }
    if (!all) {
        say("FAILED: a check failed; the measured part does not run");
        fclose(log);
        return 1;
    }

    // the measured part: one continuation per arm from the saved state, in the arms' order
    FILE * rc = fopen((a.out_dir + "/ab-rounds.csv").c_str(), "w");
    if (rc == nullptr) {
        fprintf(stderr, "spec ab: cannot write %s/ab-rounds.csv\n", a.out_dir.c_str());
        fclose(log);
        return 1;
    }
    fprintf(rc, "arm,step,kind,n_before,d,a,n_out,cut,waits,to_draft_ms,to_verify_ms,to_catchup_ms,to_one_ms,"
                "wall_ms\n");
    const auto ms = [](double s_) {
        char b[32] = "";
        if (s_ >= 0) {
            snprintf(b, sizeof(b), "%.4f", s_ * 1e3);
        }
        return std::string(b);
    };
    for (const ab_arm & arm : arms) {
        const char * name = arm.name.c_str();
        const int32_t d = arm.d;
        std::vector<int32_t> toks;
        std::vector<rt::step_info> steps;
        rt::request_summary sm;
        double w = 0;
        const std::string clk0 = sm_clocks();
        if (!ab_request(e, rs, arm, o, toks, steps, sm, w, err)) {
            verdict(name, false, err);
            fclose(rc);
            fclose(log);
            return 1;
        }
        const std::string clk1 = sm_clocks();
        say(ab_report(name, d, toks, steps, sm, w) + " | SM clocks before [" + clk0 + "] after [" + clk1 + "]");
        say(std::string(name) + ": " + e.last_summary_line());
        write_tokens(a.out_dir + "/ab-" + name + "-tokens.txt", toks);
        int32_t k = 0;
        for (const rt::step_info & x : steps) {
            fprintf(rc, "%s,%d,%s,%lld,%d,%d,%d,%d,%d,%s,%s,%s,%s,%.4f\n", name, k++,
                    x.kind == 0   ? "bootstrap"
                    : x.kind == 1 ? "generate"
                                  : "spec",
                    static_cast<long long>(x.n_before), x.d, x.a, x.n_out, x.cut ? 1 : 0, x.waits,
                    ms(x.to_draft_s).c_str(), ms(x.to_verify_s).c_str(), ms(x.to_catchup_s).c_str(),
                    ms(x.to_one_s).c_str(), x.wall_s * 1e3);
        }
        fflush(rc);
    }
    fclose(rc);
    fclose(log);
    printf("spec ab: %s\n", all ? "PASS (the checks; the arms measured)" : "FAILED");
    return all ? 0 : 1;
}
