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
#include "fn/runtime/runtime.h"
#include "fn/runtime/target.h"
#include "fn/runtime/target_cmds.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
    return n == 0 || cuda_ok(cudaMemcpy(v.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
}

bool sync_cards(rt::target & t, md::model & m) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "sync")) {
            return false;
        }
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
    auto m = std::make_unique<md::model>();
    if (!(dry ? fnfl_load_dry(a, cells, *m) : fnfl_load(a, cells, *m, "spec"))) {
        return 1;
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

int fnfl_spec_run(const fnfl_target_args & a) {
    const bool dry = a.mode == "run-dry";
    std::vector<int32_t> prompt;
    if (!read_ids(a.ids, prompt)) {
        return 1;
    }
    if (dry) {
        prompt.resize(std::min<size_t>(prompt.size(), 13));
    }
    rt::draft_options dopt;
    if (!fnfl_draft_settings(a, true, "spec run", dopt)) {
        return 1;
    }
    std::vector<fnfl_target_args::run_setting> settings = a.run_settings;
    if (settings.empty()) {
        settings.emplace_back();
        settings.back().sampling.temp = 0.0f;
    }
    const int32_t cells = dry ? 8192 : a.n_cells;
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
    rt::engine e;
    std::string err;
    if (!e.init(std::move(m), c, err)) {
        fprintf(stderr, "spec run: engine: %s\n", err.c_str());
        return 1;
    }
    printf("spec run: %s\n", e.settings().c_str());
    printf("spec run: prompt %s (%zu tokens); %zu settings x %d runs, each from a fresh prefill\n", a.ids.c_str(),
           prompt.size(), settings.size(), a.runs);
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
    for (size_t g = 0; g < settings.size(); ++g) {
        const fnfl_target_args::run_setting & st = settings[g];
        char what[256];
        snprintf(what, sizeof(what), "setting %zu: T %.2f top-k %d top-p %.2f min-p %.2f seed %llu max_tokens %d", g,
                 double(st.sampling.temp), st.sampling.top_k, double(st.sampling.top_p), double(st.sampling.min_p),
                 static_cast<unsigned long long>(st.sampling.seed), st.max_tokens);
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
            if (!fnfl_spec_request(e, prompt, 0, o, ro)) {
                fprintf(stderr, "spec run: %s\n", e.error().c_str());
                return 1;
            }
            const double                 pre_s = ro.pre_s, gen_s = ro.gen_s;
            const std::vector<int32_t> & toks = ro.toks;
            const std::string clk1 = sm_clocks();
            const rt::request_summary & sm = e.last_summary();
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
                     "run %zu.%d: %zu tokens; prefill %.3f s; driver: after the prefill to the last step %.4f s, %.2f "
                     "tokens/s ((tokens - 1) / time, the server's definition); tokens %s run 0's; SM clocks before "
                     "[%s] after [%s]",
                     g, r, toks.size(), pre_s, gen_s, tps_driver.back(), same ? "equal to" : "DIFFERENT from",
                     clk0.c_str(), clk1.c_str());
            printf("spec run: %s\nspec run: %s\n", line, e.last_summary_line().c_str());
            fprintf(log, "%s\n%s\n", line, e.last_summary_line().c_str());
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
                 "setting %zu over %d runs: median step wall ms median %.4f (max-min %.4f); tokens/s (engine span, "
                 "(tokens - 1) / span) median %.3f (max-min %.3f); tokens/s (driver, the server's definition) median "
                 "%.3f (max-min %.3f); tokens equal across runs: %s",
                 g, a.runs, median_of(med), spread(med), median_of(tps_engine), spread(tps_engine),
                 median_of(tps_driver), spread(tps_driver), all ? "yes" : "NO");
        printf("spec run: %s\n", sum);
        fprintf(log, "%s\n", sum);
    }
    fclose(rc);
    if (dry) {
        // the other variant at the first setting's top-k (TK1), or the default when that setting is off the top-k path
        const int32_t k = a.run_settings.front().sampling.top_k;
        all = run_dry_paths(e, prompt, k > 0 ? k : sp::options().top_k, log) && all;
    }
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
