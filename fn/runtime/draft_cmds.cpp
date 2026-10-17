// DR1 Stage C: fnfl-drive draft-test and draft-bench (fn/runtime/target_cmds.h).

#include "fn/kernels/head/head.h"
#include "fn/model/draft.h"
#include "fn/model/model.h"
#include "fn/proto/xp1/timing.h"
#include "fn/runtime/target.h"
#include "fn/runtime/target_cmds.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace rt = fn::runtime;
namespace md = fn::model;

constexpr int32_t NS = fn::head::n_draft_vocab; // 65,536: the subset
constexpr int32_t NC = fn::head::n_draft_card;  // 32,768 a card
constexpr int32_t F = 1280;
constexpr int32_t D = 2560;
constexpr int32_t W = 4 * D; // a hidden row of the reference (4 streams)

using clk = std::chrono::steady_clock;

double since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "draft: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool read_ids(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path);
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
    if (out.empty()) {
        fprintf(stderr, "draft: cannot read %s\n", path.c_str());
    }
    return !out.empty();
}

// a gzip file's contents (gzip -dc), exactly want bytes
bool gz_read(const std::string & path, std::vector<uint8_t> & out, size_t want) {
    out.assign(want, 0);
    const std::string cmd = "gzip -dc '" + path + "'";
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) {
        fprintf(stderr, "draft-test: cannot run %s\n", cmd.c_str());
        return false;
    }
    const size_t got = fread(out.data(), 1, want, p);
    const int extra = fgetc(p);
    const int rc = pclose(p);
    if (got != want || extra != EOF || rc != 0) {
        fprintf(stderr, "draft-test: %s: %zu bytes (want %zu), more %d, rc %d\n", path.c_str(), got, want,
                extra != EOF ? 1 : 0, rc);
        return false;
    }
    return true;
}

// numpy's default (linear) percentile
double percentile(std::vector<double> v, double q) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const double pos = q / 100.0 * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

uint64_t hash_bytes(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 0x100000001b3ull;
    }
    return h;
}

// KL(p || q) over the subset from F32 logits, in FP64
double kl_sub(const float * p, const float * q) {
    double mp = -HUGE_VAL, mq = -HUGE_VAL;
    for (int32_t i = 0; i < NS; ++i) {
        mp = std::max(mp, double(p[i]));
        mq = std::max(mq, double(q[i]));
    }
    double sp = 0.0, sq = 0.0;
    for (int32_t i = 0; i < NS; ++i) {
        sp += std::exp(double(p[i]) - mp);
        sq += std::exp(double(q[i]) - mq);
    }
    const double lsp = std::log(sp), lsq = std::log(sq);
    double kl = 0.0;
    for (int32_t i = 0; i < NS; ++i) {
        const double lp = double(p[i]) - mp - lsp;
        const double lq = double(q[i]) - mq - lsq;
        kl += std::exp(lp) * (lp - lq);
    }
    return kl;
}

struct top_rec {
    int32_t top1;
    float l1;
    int32_t top2;
    float l2;
};

struct chain_line {
    int32_t q = 0, step = 0, tok_in = 0, top1 = 0, top2 = 0;
    float l1 = 0, l2 = 0;
};

struct fixture {
    std::string name;
    int32_t n = 0, w0 = 0, stride = 1;
    bool selected(int32_t p) const { return p >= w0 && (p - w0) % stride == stride - 1; }
    int32_t index(int32_t p) const { return (p - w0) / stride; }
    int32_t n_selected() const { return n > w0 ? (n - w0) / stride : 0; }
};

// one fixture's reference (src/ref_dump_draft.cpp): decode mode in full, prompt mode's top and logits
struct reference {
    std::vector<uint8_t> top_dec, top_pr, lg_dec, lg_pr, h_in, h_nextn, chain_lg;
    std::vector<chain_line> chain;
    std::map<int32_t, int32_t> chain_at; // q -> its step 2 line
    const top_rec & top(bool dec, int32_t p) const {
        return reinterpret_cast<const top_rec *>((dec ? top_dec : top_pr).data())[p];
    }
    const float * logits(bool dec, int32_t k) const {
        return reinterpret_cast<const float *>((dec ? lg_dec : lg_pr).data()) + size_t(k) * NS;
    }
};

bool load_reference(const std::string & dir, const fixture & f, reference & r) {
    const std::string b = dir + "/" + f.name + "/";
    const size_t ns = size_t(f.n_selected());
    if (!gz_read(b + "decode/top.bin.gz", r.top_dec, size_t(f.n) * 16) ||
        !gz_read(b + "prompt/top.bin.gz", r.top_pr, size_t(f.n) * 16) ||
        !gz_read(b + "decode/logits.bin.gz", r.lg_dec, ns * NS * 4) ||
        !gz_read(b + "prompt/logits.bin.gz", r.lg_pr, ns * NS * 4) ||
        !gz_read(b + "decode/h_input.bin.gz", r.h_in, ns * W * 4) ||
        !gz_read(b + "decode/h_nextn.bin.gz", r.h_nextn, ns * W * 4)) {
        return false;
    }
    // the selected positions as the dump wrote them
    for (const char * mode : {"decode", "prompt"}) {
        std::ifstream pf(b + mode + "/positions.tsv");
        int32_t p = 0, k = 0;
        while (pf >> p) {
            if (!f.selected(p) || f.index(p) != k) {
                fprintf(stderr, "draft-test: %s %s positions.tsv: %d at line %d is not the selection\n", f.name.c_str(),
                        mode, p, k);
                return false;
            }
            ++k;
        }
        if (size_t(k) != ns) {
            fprintf(stderr, "draft-test: %s %s positions.tsv: %d positions, want %zu\n", f.name.c_str(), mode, k, ns);
            return false;
        }
    }
    std::ifstream cf(b + "decode/chain.tsv");
    std::string line;
    std::getline(cf, line); // the header
    while (std::getline(cf, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream ls(line);
        chain_line c;
        ls >> c.q >> c.step >> c.tok_in >> c.top1 >> c.l1 >> c.top2 >> c.l2;
        if (!ls || (c.step != 2 && c.step != 3)) {
            fprintf(stderr, "draft-test: %s chain.tsv: bad line '%s'\n", f.name.c_str(), line.c_str());
            return false;
        }
        if (c.step == 2) {
            r.chain_at[c.q] = static_cast<int32_t>(r.chain.size());
        }
        r.chain.push_back(c);
    }
    return gz_read(b + "decode/chain.bin.gz", r.chain_lg, r.chain.size() * NS * 4);
}

// the draft head's columns: card c's column i is subset index sidx[c][i]
struct head_map {
    std::vector<int32_t> ids[2], sidx[2];
};

bool make_map(const std::vector<int32_t> & subset, head_map & h) {
    std::vector<int32_t> inv(size_t(fn::head::n_vocab), -1);
    for (size_t i = 0; i < subset.size(); ++i) {
        inv[size_t(subset[i])] = static_cast<int32_t>(i);
    }
    for (int32_t c = 0; c < 2; ++c) {
        h.ids[c].assign(size_t(NC), 0);
        h.sidx[c].assign(size_t(NC), 0);
        if (!fn::head::draft_ids(subset.data(), NS, c, h.ids[c].data())) {
            return false;
        }
        for (int32_t i = 0; i < NC; ++i) {
            h.sidx[c][size_t(i)] = inv[size_t(h.ids[c][size_t(i)])];
        }
    }
    return true;
}

// the d steps' logits of both cards ([c][j][32768])
bool read_logits(rt::target & t, int32_t d, std::vector<float> & lg) {
    lg.resize(size_t(2) * 3 * NC);
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(t.card(c).c.device), "device") ||
            !cuda_ok(cudaMemcpy(lg.data() + size_t(c) * 3 * NC, t.draft(c).logits, size_t(d) * NC * 4,
                                cudaMemcpyDeviceToHost),
                     "draft logits")) {
            return false;
        }
    }
    return true;
}

// a [4][1280]-a-card buffer of both cards as [4][2560] (card c's features at c * 1280)
bool read_rows(rt::target & t, float * const src0, float * const src1, std::vector<float> & out) {
    out.resize(size_t(W));
    std::vector<float> h(size_t(4) * F);
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(t.card(c).c.device), "device") ||
            !cuda_ok(cudaMemcpy(h.data(), c == 0 ? src0 : src1, h.size() * 4, cudaMemcpyDeviceToHost), "rows")) {
            return false;
        }
        for (int32_t s = 0; s < 4; ++s) {
            std::memcpy(out.data() + size_t(s) * D + size_t(c) * F, h.data() + size_t(s) * F, size_t(F) * 4);
        }
    }
    return true;
}

// max |a - b| / max |b| over n
double rel_max(const float * a, const float * b, size_t n) {
    double e = 0.0, m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        e = std::max(e, std::fabs(double(a[i]) - double(b[i])));
        m = std::max(m, std::fabs(double(b[i])));
    }
    return m > 0.0 ? e / m : e;
}

int32_t accepted(const int32_t * prop, int32_t d, const std::vector<int32_t> & tok, size_t p) {
    int32_t a = 0;
    while (a < d && p + size_t(a) + 1 < tok.size() && prop[a] == tok[p + size_t(a) + 1]) {
        ++a;
    }
    return a;
}

struct fixture_out {
    std::vector<uint64_t> hash; // a step round: its logits on both cards and its proposals
    std::vector<double> kl, kl_chain;
    int32_t top1_diff = 0, steps = 0, chain_steps = 0, chain_same_input = 0, chain_top1_diff = 0;
    int32_t chain_qs = 0, chain_missing = 0;
    int64_t rounds = 0, catchups = 0;
    std::vector<double> carry_rel, hnext_rel;
    int64_t tf_ours = 0, tf_ref = 0, tf_n = 0; // teacher-forced accepted tokens of the chains, and the chains
    double seconds = 0.0;
};

// One pass over the fixture's tokens (module comment in target_cmds.h). ref null: the hashes only.
bool run_fixture(rt::target & t, const std::vector<int32_t> & ids, const fixture & f, const reference * ref,
                 const head_map & hm, fixture_out & o, FILE * tsv, FILE * ctsv) {
    std::string err;
    const auto t0 = clk::now();
    if (!t.reset(err)) {
        fprintf(stderr, "draft-test: reset: %s\n", err.c_str());
        return false;
    }
    t.set_prompt(std::vector<int32_t>(ids.begin(), ids.begin() + f.n));
    std::vector<float> lg, ours(static_cast<size_t>(NS)), carry, hnext;
    rt::round_result r;
    const auto catch_to = [&](int64_t e) {
        while (t.draft_n() < e) {
            const int32_t rows = static_cast<int32_t>(std::min<int64_t>(4, e - t.draft_n()));
            if (!t.draft_catchup(rows, r, err)) {
                fprintf(stderr, "draft-test: %s catch-up at %lld: %s\n", f.name.c_str(),
                        static_cast<long long>(t.draft_n()), err.c_str());
                return false;
            }
            ++o.catchups;
        }
        return true;
    };
    for (int32_t pos = 0, ci = 0; pos < f.n; ++ci) {
        const int32_t B = std::min(1 + ci % 4, f.n - pos);
        if (!t.forced(B, r, err)) {
            fprintf(stderr, "draft-test: %s round at %d (B %d): %s\n", f.name.c_str(), pos, B, err.c_str());
            return false;
        }
        ++o.rounds;
        const int32_t e = pos + B;
        for (int32_t q = pos; q < e; ++q) {
            if (!f.selected(q)) {
                continue;
            }
            if (!catch_to(q)) {
                return false;
            }
            if (ref != nullptr && !read_rows(t, t.draft(0).carry_prev, t.draft(1).carry_prev, carry)) {
                return false;
            }
            const int32_t d = q == e - 1 ? 3 : 1;
            if (!t.draft_step(d, r, err)) {
                fprintf(stderr, "draft-test: %s step at %d (d %d): %s\n", f.name.c_str(), q, d, err.c_str());
                return false;
            }
            ++o.steps;
            if (!read_logits(t, d, lg)) {
                return false;
            }
            uint64_t h = 0xcbf29ce484222325ull;
            for (int32_t c = 0; c < 2; ++c) {
                h = hash_bytes(lg.data() + size_t(c) * 3 * NC, size_t(d) * NC * 4, h);
            }
            o.hash.push_back(hash_bytes(r.argmax[0], size_t(d) * 4, h));
            if (ref == nullptr) {
                continue;
            }
            const int32_t k = f.index(q);
            const auto assemble = [&](int32_t j) {
                for (int32_t c = 0; c < 2; ++c) {
                    const float * row = lg.data() + size_t(c) * 3 * NC + size_t(j) * NC;
                    for (int32_t i = 0; i < NC; ++i) {
                        ours[size_t(hm.sidx[c][size_t(i)])] = row[i];
                    }
                }
            };
            assemble(0);
            const double kl = kl_sub(ref->logits(true, k), ours.data());
            const double klp = kl_sub(ref->logits(true, k), ref->logits(false, k));
            o.kl.push_back(kl);
            const top_rec & td = ref->top(true, q);
            const top_rec & tp = ref->top(false, q);
            o.top1_diff += r.argmax[0][0] != td.top1 ? 1 : 0;
            o.carry_rel.push_back(
                rel_max(carry.data(), reinterpret_cast<const float *>(ref->h_in.data()) + size_t(k) * W, size_t(W)));
            if (d == 1) {
                if (!read_rows(t, t.draft(0).h_nextn, t.draft(1).h_nextn, hnext)) {
                    return false;
                }
                o.hnext_rel.push_back(rel_max(
                    hnext.data(), reinterpret_cast<const float *>(ref->h_nextn.data()) + size_t(k) * W, size_t(W)));
            }
            fprintf(tsv, "%d\t%d\t%d\t%d\t%d\t%.6e\t%.6e\t%.4f\n", q, d, r.argmax[0][0], td.top1, tp.top1, kl, klp,
                    double(td.l1) - double(td.l2));
            if (d == 3) {
                const auto it = ref->chain_at.find(q);
                if (it == ref->chain_at.end()) {
                    ++o.chain_missing;
                    continue;
                }
                ++o.chain_qs;
                int32_t props[3] = {r.argmax[0][0], r.argmax[0][1], r.argmax[0][2]};
                int32_t refp[3] = {td.top1, 0, 0};
                for (int32_t j = 1; j < 3; ++j) {
                    const chain_line & cl = ref->chain[size_t(it->second + j - 1)];
                    refp[j] = cl.top1;
                    assemble(j);
                    const double klc =
                        kl_sub(reinterpret_cast<const float *>(ref->chain_lg.data()) + size_t(it->second + j - 1) * NS,
                               ours.data());
                    const bool same = cl.q == q && cl.step == j + 1 && props[j - 1] == cl.tok_in;
                    ++o.chain_steps;
                    if (same) {
                        ++o.chain_same_input;
                        o.kl_chain.push_back(klc);
                        o.chain_top1_diff += props[j] != cl.top1 ? 1 : 0;
                    }
                    fprintf(ctsv, "%d\t%d\t%d\t%d\t%d\t%d\t%.6e\t%d\n", q, j + 1, props[j - 1], cl.tok_in, props[j],
                            cl.top1, klc, same ? 1 : 0);
                }
                o.tf_ours += accepted(props, 3, ids, size_t(q));
                o.tf_ref += accepted(refp, 3, ids, size_t(q));
                ++o.tf_n;
            }
        }
        if (!catch_to(e)) {
            return false;
        }
        pos = e;
    }
    o.seconds = since(t0);
    return true;
}

// the yardstick: KL(decode || prompt) and prompt's top-1 differences at the selected positions
void yardstick(const fixture & f, const reference & ref, std::vector<double> & kl, int32_t & top1_diff) {
    kl.clear();
    top1_diff = 0;
    for (int32_t q = 0; q < f.n; ++q) {
        if (!f.selected(q)) {
            continue;
        }
        kl.push_back(kl_sub(ref.logits(true, f.index(q)), ref.logits(false, f.index(q))));
        top1_diff += ref.top(true, q).top1 != ref.top(false, q).top1 ? 1 : 0;
    }
}

bool judge(const fixture & f, const fixture_out & o, const std::vector<double> & ykl, int32_t ytop) {
    const double s[3] = {percentile(o.kl, 50), percentile(o.kl, 99), *std::max_element(o.kl.begin(), o.kl.end())};
    const double y[3] = {percentile(ykl, 50), percentile(ykl, 99), *std::max_element(ykl.begin(), ykl.end())};
    const bool kl_ok = s[0] <= 2 * y[0] && s[1] <= 2 * y[1] && s[2] <= 2 * y[2];
    const bool top_ok = o.top1_diff <= 2 * ytop;
    printf("draft-test: %s: %d positions; KL(decode || FNFL) p50 %.4e p99 %.4e max %.4e; yardstick KL(decode || "
           "prompt) p50 %.4e p99 %.4e max %.4e; ratios %.3f %.3f %.3f: %s\n",
           f.name.c_str(), static_cast<int32_t>(o.kl.size()), s[0], s[1], s[2], y[0], y[1], y[2], s[0] / y[0],
           s[1] / y[1], s[2] / y[2], kl_ok ? "within 2x" : "BEYOND 2x");
    printf("draft-test: %s: top-1 differences FNFL %d, prompt %d: %s\n", f.name.c_str(), o.top1_diff, ytop,
           top_ok ? "within 2x" : "BEYOND 2x");
    if (!o.kl_chain.empty()) {
        printf("draft-test: %s: chains: %d positions (%d not in the reference), %d chained steps, %d with the "
               "reference's input token; their KL p50 %.4e p99 %.4e max %.4e, top-1 differences %d (information: no "
               "prompt-mode chain)\n",
               f.name.c_str(), o.chain_qs, o.chain_missing, o.chain_steps, o.chain_same_input,
               percentile(o.kl_chain, 50), percentile(o.kl_chain, 99),
               *std::max_element(o.kl_chain.begin(), o.kl_chain.end()), o.chain_top1_diff);
    }
    if (o.tf_n > 0) {
        printf("draft-test: %s: teacher-forced acceptance of d = 3 chains against the fixture's tokens: FNFL %.3f, "
               "reference %.3f tokens a chain over %lld chains (offline, T = 0)\n",
               f.name.c_str(), double(o.tf_ours) / double(o.tf_n), double(o.tf_ref) / double(o.tf_n),
               static_cast<long long>(o.tf_n));
    }
    printf("draft-test: %s: diagnostics: retained carry against h_input rel max p50 %.3e max %.3e; h_nextn (d = 1) "
           "rel max p50 %.3e max %.3e\n",
           f.name.c_str(), percentile(o.carry_rel, 50),
           o.carry_rel.empty() ? 0.0 : *std::max_element(o.carry_rel.begin(), o.carry_rel.end()),
           percentile(o.hnext_rel, 50),
           o.hnext_rel.empty() ? 0.0 : *std::max_element(o.hnext_rel.begin(), o.hnext_rel.end()));
    printf("draft-test: %s: %lld target rounds, %lld catch-ups, %d steps in %.1f s\n", f.name.c_str(),
           static_cast<long long>(o.rounds), static_cast<long long>(o.catchups), o.steps, o.seconds);
    fflush(stdout);
    return kl_ok && top_ok && o.chain_missing == 0;
}

// offline T = 0 acceptance: after r4k's pass (the draft caught up), the bootstrap, then n generate rounds each
// preceded by the catch-up and a d = 3 step at the pending token; proposals against the target's later greedy tokens
bool acceptance(rt::target & t, int32_t n, FILE * f) {
    std::string err;
    rt::round_result r;
    if (!t.bootstrap(r, err)) {
        fprintf(stderr, "draft-test: bootstrap: %s\n", err.c_str());
        return false;
    }
    std::vector<std::vector<int32_t>> props;
    std::vector<int64_t> at;
    for (int32_t i = 0; i < n + 3; ++i) {
        while (t.draft_n() < t.slot().n) {
            if (!t.draft_catchup(static_cast<int32_t>(std::min<int64_t>(4, t.slot().n - t.draft_n())), r, err)) {
                fprintf(stderr, "draft-test: acceptance catch-up: %s\n", err.c_str());
                return false;
            }
        }
        if (i < n) {
            if (!t.draft_step(3, r, err)) {
                fprintf(stderr, "draft-test: acceptance step: %s\n", err.c_str());
                return false;
            }
            props.push_back({r.argmax[0][0], r.argmax[0][1], r.argmax[0][2]});
            at.push_back(t.draft_n());
        }
        if (!t.generate(r, err)) {
            fprintf(stderr, "draft-test: generate: %s\n", err.c_str());
            return false;
        }
    }
    int64_t total = 0, hist[4] = {0, 0, 0, 0}, first[3] = {0, 0, 0};
    for (size_t i = 0; i < props.size(); ++i) {
        const int32_t a = accepted(props[i].data(), 3, t.slot().tok, size_t(at[i]));
        total += a;
        ++hist[a];
        for (int32_t j = 0; j < 3; ++j) {
            first[j] += a > j ? 1 : 0;
        }
        fprintf(f, "%lld\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", static_cast<long long>(at[i]), t.slot().tok[size_t(at[i])],
                props[i][0], props[i][1], props[i][2], t.slot().tok[size_t(at[i]) + 1], t.slot().tok[size_t(at[i]) + 2],
                a);
    }
    printf("draft-test: offline T = 0 acceptance after r4k (d = 3, the target's greedy continuation from 4,096): %.3f "
           "tokens a step over %d steps; accepted 0/1/2/3: %lld %lld %lld %lld; step 1/2/3 reached %.3f %.3f %.3f\n",
           double(total) / double(n), n, static_cast<long long>(hist[0]), static_cast<long long>(hist[1]),
           static_cast<long long>(hist[2]), static_cast<long long>(hist[3]), double(first[0]) / double(n),
           double(first[1]) / double(n), double(first[2]) / double(n));
    return true;
}

// the draft's K and V cells 0 .. n - 1 and its retained carry on both cards
bool read_draft_state(rt::target & t, int32_t n, std::vector<uint16_t> & kv, std::vector<float> & carry) {
    kv.resize(size_t(4) * n * 256);
    carry.resize(size_t(2) * 4 * F);
    for (int32_t c = 0; c < 2; ++c) {
        const md::draft_card & d = t.draft(c);
        if (!cuda_ok(cudaSetDevice(t.card(c).c.device), "device") ||
            (n > 0 && (!cuda_ok(cudaMemcpy(kv.data() + size_t(2 * c) * n * 256, d.k_cache, size_t(n) * 512,
                                           cudaMemcpyDeviceToHost),
                                "K") ||
                       !cuda_ok(cudaMemcpy(kv.data() + size_t(2 * c + 1) * n * 256, d.v_cache, size_t(n) * 512,
                                           cudaMemcpyDeviceToHost),
                                "V"))) ||
            !cuda_ok(
                cudaMemcpy(carry.data() + size_t(c) * 4 * F, d.carry_prev, size_t(4) * F * 4, cudaMemcpyDeviceToHost),
                "carry")) {
            return false;
        }
    }
    return true;
}

// the catch-up against a clean replay at B = 1: every forced_k (B, k) commit (k = 0..B, B = 1..4), its catch-up of k
// and a d = 3 step; after each the cells 0 .. N - 1 and the retained carry equal the replay's bitwise
bool catchup_bitwise(rt::target & t, const std::vector<int32_t> & ids, int32_t & compares, int32_t & differ) {
    static const int32_t pairs[14][2] = {{1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}, {3, 0}, {3, 1},
                                         {3, 2}, {3, 3}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}};
    int32_t total = 0;
    for (const auto & pk : pairs) {
        total += pk[1];
    }
    std::string err;
    rt::round_result r;
    const std::vector<int32_t> prompt(ids.begin(), ids.begin() + 64);
    if (!t.reset(err)) {
        return false;
    }
    t.set_prompt(prompt);
    std::vector<std::vector<float>> clean_carry;
    std::vector<uint16_t> clean_kv, kv;
    std::vector<float> carry;
    for (int32_t p = 0; p < total; ++p) {
        if (!t.forced(1, r, err) || !t.draft_catchup(1, r, err) || !read_draft_state(t, 0, kv, carry)) {
            fprintf(stderr, "draft-test: clean replay at %d: %s\n", p, err.c_str());
            return false;
        }
        clean_carry.push_back(carry);
    }
    if (!read_draft_state(t, total, clean_kv, carry) || !t.reset(err)) {
        return false;
    }
    t.set_prompt(prompt);
    compares = differ = 0;
    for (const auto & pk : pairs) {
        if (!t.forced_k(pk[0], pk[1], 0, r, err) || (pk[1] > 0 && !t.draft_catchup(pk[1], r, err)) ||
            !t.draft_step(3, r, err)) {
            fprintf(stderr, "draft-test: forced_k (%d, %d): %s\n", pk[0], pk[1], err.c_str());
            return false;
        }
        const int32_t n = static_cast<int32_t>(t.draft_n());
        if (!read_draft_state(t, n, kv, carry)) {
            return false;
        }
        bool same = n == 0 || std::memcmp(carry.data(), clean_carry[size_t(n - 1)].data(), carry.size() * 4) == 0;
        for (int32_t part = 0; part < 4; ++part) {
            same = same && std::memcmp(kv.data() + size_t(part) * n * 256, clean_kv.data() + size_t(part) * total * 256,
                                       size_t(n) * 512) == 0;
        }
        ++compares;
        differ += same ? 0 : 1;
        printf("draft-test: catch-up after forced_k (B %d, k %d) at N %d: cells and retained carry %s\n", pk[0], pk[1],
               n, same ? "bitwise equal" : "DIFFER");
    }
    return true;
}

} // namespace

int fnfl_draft_test(const fnfl_target_args & a) {
    std::vector<int32_t> ids, subset;
    std::string err;
    rt::target_options o;
    o.model_path = a.model_path;
    if (!fnfl_draft_settings(a, true, "draft-test", o.draft) || !read_ids(a.ids, ids) ||
        !md::read_draft_subset(o.draft.subset, subset, err)) {
        if (!err.empty()) {
            fprintf(stderr, "draft-test: %s\n", err.c_str());
        }
        return 2;
    }
    head_map hm;
    if (!make_map(subset, hm)) {
        fprintf(stderr, "draft-test: draft_ids\n");
        return 2;
    }
    const fixture r4k = {"r4k", 4096, 0, 8};
    const fixture deep = {"deep", 16384, 16128, 1};
    if (ids.size() < size_t(deep.n) + 8) {
        fprintf(stderr, "draft-test: %s holds too few tokens\n", a.ids.c_str());
        return 2;
    }
    // the reference first (no model load while a dump is unreadable)
    const auto tr = clk::now();
    auto ref = std::make_unique<reference>();
    std::vector<double> ykl;
    int32_t ytop = 0;
    if (!load_reference(a.ref_draft, r4k, *ref)) {
        return 2;
    }
    yardstick(r4k, *ref, ykl, ytop);
    printf("draft-test: r4k reference read and its yardstick in %.1f s (%zu chain lines)\n", since(tr),
           ref->chain.size());
    auto m = std::make_unique<md::model>();
    if (!fnfl_load(a, a.n_cells, *m, "draft-test")) {
        return 1;
    }
    rt::target t;
    if (!t.init(*m, o, err)) {
        fprintf(stderr, "draft-test: init: %s\n", err.c_str());
        return 1;
    }
    printf("draft-test: %d graph captures (target B 1..4, draft step d 1..3, catch-up B_c 1..4)\n", t.captures());
    FILE * tsv = fopen((a.out_dir + "/draft-test-r4k.tsv").c_str(), "w");
    FILE * ctsv = fopen((a.out_dir + "/draft-test-r4k-chain.tsv").c_str(), "w");
    FILE * atsv = fopen((a.out_dir + "/draft-test-acceptance.tsv").c_str(), "w");
    if (tsv == nullptr || ctsv == nullptr || atsv == nullptr) {
        fprintf(stderr, "draft-test: cannot write in %s\n", a.out_dir.c_str());
        return 1;
    }
    fprintf(tsv, "q\td\tfnfl_top1\tdecode_top1\tprompt_top1\tkl_fnfl\tkl_prompt\tdecode_margin\n");
    fprintf(ctsv, "q\tstep\tfnfl_input\tref_input\tfnfl_top1\tref_top1\tkl\tsame_input\n");
    fprintf(atsv, "pos\ttoken\tprop1\tprop2\tprop3\tnext1\tnext2\taccepted\n");
    fixture_out o1;
    bool ok = run_fixture(t, ids, r4k, ref.get(), hm, o1, tsv, ctsv);
    bool pass = ok && judge(r4k, o1, ykl, ytop);
    fclose(tsv);
    fclose(ctsv);
    ok = ok && acceptance(t, 128, atsv);
    fclose(atsv);
    // deep
    ref = std::make_unique<reference>();
    ok = ok && load_reference(a.ref_draft, deep, *ref);
    if (ok) {
        yardstick(deep, *ref, ykl, ytop);
        tsv = fopen((a.out_dir + "/draft-test-deep.tsv").c_str(), "w");
        ctsv = fopen((a.out_dir + "/draft-test-deep-chain.tsv").c_str(), "w");
        ok = tsv != nullptr && ctsv != nullptr;
        fixture_out od;
        ok = ok && run_fixture(t, ids, deep, ref.get(), hm, od, tsv, ctsv);
        pass = pass && ok && judge(deep, od, ykl, ytop);
        if (tsv != nullptr) {
            fclose(tsv);
        }
        if (ctsv != nullptr) {
            fclose(ctsv);
        }
    }
    ref.reset();
    // the catch-up against a clean replay
    int32_t compares = 0, differ = 0;
    ok = ok && catchup_bitwise(t, ids, compares, differ);
    printf("draft-test: catch-up bitwise: %d of %d commits differ from the clean replay\n", differ, compares);
    pass = pass && ok && differ == 0 && compares == 14;
    // r4k again: bitwise
    fixture_out o2;
    ok = ok && run_fixture(t, ids, r4k, nullptr, hm, o2, nullptr, nullptr);
    const bool same = ok && o2.hash == o1.hash;
    printf("draft-test: r4k twice: %zu step rounds, logits and proposals %s\n", o1.hash.size(),
           same ? "bitwise equal" : "DIFFER");
    pass = pass && same;
    printf("draft-test: both cards' proposals equal in every step round (each round checks it)\n");
    printf("draft-test: %s\n", ok && pass ? "PASS" : "FAIL");
    return ok && pass ? 0 : 1;
}

// ---- draft-bench ----

namespace {

struct point {
    std::vector<double> us[2];
};

double p90(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t k = static_cast<size_t>(std::ceil(0.9 * double(v.size()))) - 1;
    return v[std::min(k, v.size() - 1)];
}

bool sm_mhz(rt::target & t, uint64_t * const probe[2], double mhz[2]) {
    for (int32_t c = 0; c < 2; ++c) {
        uint64_t h[2] = {0, 0};
        cudaStream_t s = static_cast<cudaStream_t>(t.stream(c));
        if (!cuda_ok(cudaSetDevice(t.card(c).c.device), "device") || !fn::xp1::clock_probe(probe[c], s) ||
            !cuda_ok(cudaMemcpyAsync(h, probe[c], 16, cudaMemcpyDeviceToHost, s), "probe") ||
            !cuda_ok(cudaStreamSynchronize(s), "probe")) {
            return false;
        }
        mhz[c] = h[1] > 0 ? double(h[0]) * 1e3 / double(h[1]) : 0.0;
    }
    return true;
}

// one round of the point: a step of d (kind 0) or a catch-up of rows (kind 1) at depth
bool bench_round(rt::target & t, int32_t kind, int32_t rows, int32_t depth, rt::round_result & r, std::string & err) {
    if (!t.draft_set_depth(depth, err)) {
        return false;
    }
    return kind == 0 ? t.draft_step(rows, r, err) : t.draft_catchup(rows, r, err);
}

bool bench_depth(rt::target & t, uint64_t * const probe[2], FILE * jf, int32_t depth, const char * label) {
    std::string err;
    rt::round_result r;
    if (!t.set_depth(depth + 4, err)) {
        fprintf(stderr, "draft-bench: set_depth: %s\n", err.c_str());
        return false;
    }
    // the clocks up: about 2 s of rounds, untimed
    const auto t0 = clk::now();
    while (since(t0) < 2.0) {
        if (!t.forced(1, r, err, false, false) || !t.set_depth(depth + 4, err) ||
            !bench_round(t, 0, 3, depth, r, err)) {
            fprintf(stderr, "draft-bench: warm-up: %s\n", err.c_str());
            return false;
        }
    }
    for (int32_t kind = 0; kind < 2; ++kind) {
        for (int32_t rows = 1; rows <= (kind == 0 ? 3 : 4); ++rows) {
            double mhz[2] = {0, 0};
            if (!sm_mhz(t, probe, mhz)) {
                return false;
            }
            point pt;
            for (int32_t i = 0; i < 23; ++i) {
                if (!bench_round(t, kind, rows, depth, r, err)) {
                    fprintf(stderr, "draft-bench: %s at %d: %s\n", kind == 0 ? "step" : "catch-up", depth, err.c_str());
                    return false;
                }
                if (i < 3) {
                    continue;
                }
                for (int32_t c = 0; c < 2; ++c) {
                    pt.us[c].push_back(double(r.t_end[c] - r.t_start[c]) / 1000.0);
                }
                fprintf(jf,
                        "{\"label\": \"%s\", \"depth\": %d, \"kind\": \"%s\", \"rows\": %d, \"replay\": %d, \"round\": "
                        "%llu, \"span_us\": [%.2f, %.2f]}\n",
                        label, depth, kind == 0 ? "step" : "catchup", rows, i - 3,
                        static_cast<unsigned long long>(r.round), pt.us[0].back(), pt.us[1].back());
            }
            printf("draft-bench: %-9s depth %6d %-8s %s %d  span us card0 %8.2f (p90 %8.2f) card1 %8.2f "
                   "(p90 %8.2f)  SM MHz %.0f / %.0f\n",
                   label, depth, kind == 0 ? "step" : "catch-up", kind == 0 ? "d  " : "B_c", rows,
                   percentile(pt.us[0], 50), p90(pt.us[0]), percentile(pt.us[1], 50), p90(pt.us[1]), mhz[0], mhz[1]);
            fflush(stdout);
            fflush(jf);
        }
    }
    return true;
}

} // namespace

int fnfl_draft_bench(const fnfl_target_args & a) {
    std::vector<int32_t> ids;
    const int32_t deepest = 196608;
    rt::target_options o;
    o.model_path = a.model_path;
    if (!fnfl_draft_settings(a, true, "draft-bench", o.draft) || !read_ids(a.ids, ids) || ids.size() < 4096) {
        return 2;
    }
    for (size_t i = 0; ids.size() < size_t(deepest) + 16; ++i) {
        ids.push_back(ids[i]);
    }
    auto m = std::make_unique<md::model>();
    if (!fnfl_load(a, a.n_cells, *m, "draft-bench")) {
        return 1;
    }
    rt::target t;
    std::string err;
    if (!t.init(*m, o, err)) {
        fprintf(stderr, "draft-bench: init: %s\n", err.c_str());
        return 1;
    }
    printf("draft-bench: %d graph captures; no capture after this line\n", t.captures());
    uint64_t * probe[2] = {nullptr, nullptr};
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(t.card(c).c.device), "device") ||
            !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&probe[c]), 16), "probe")) {
            return 1;
        }
    }
    FILE * jf = fopen((a.out_dir + "/draft-bench.jsonl").c_str(), "w");
    if (jf == nullptr) {
        return 1;
    }
    // the real history: 4,096 positions, B cycling, each batch caught up
    t.set_prompt(ids);
    bool ok = true;
    rt::round_result r;
    const auto th = clk::now();
    for (int32_t pos = 0, ci = 0; ok && pos < 4096; ++ci) {
        const int32_t B = std::min(1 + ci % 4, 4096 - pos);
        ok = t.forced(B, r, err);
        while (ok && t.draft_n() < pos + B) {
            ok = t.draft_catchup(static_cast<int32_t>(std::min<int64_t>(4, pos + B - t.draft_n())), r, err);
        }
        pos += B;
    }
    if (!ok) {
        fprintf(stderr, "draft-bench: history: %s\n", err.c_str());
        return 1;
    }
    printf("draft-bench: real history of 4096 positions (draft caught up) in %.1f s\n", since(th));
    ok = bench_depth(t, probe, jf, 4096, "real");
    if (ok) {
        const auto tf = clk::now();
        ok = t.reset(err) && fnfl_fill_synthetic(*m, t, deepest);
        t.set_prompt(ids);
        printf("draft-bench: synthetic history (the draft's K and V too) in %.1f s\n", since(tf));
    }
    for (const int32_t d : {58581, 131072, 196608}) {
        ok = ok && bench_depth(t, probe, jf, d, "synthetic");
    }
    fclose(jf);
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(t.card(c).c.device);
        cudaFree(probe[c]);
    }
    printf("draft-bench: %s; graph captures %d\n", ok ? "done" : "FAILED", t.captures());
    return ok ? 0 : 1;
}
