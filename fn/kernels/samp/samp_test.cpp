#include "fn/kernels/samp/samp_test.h"

#include "fn/kernels/head/head.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/samp/philox.h"
#include "fn/kernels/samp/samp.h"
#include "fn/kernels/samp/samp_harness.h"
#include "fn/kernels/samp/samp_ref.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdarg>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

// fnfl-drive samp-test (SM1 steps 2-5, 8): the sampler's kernels and exchanges on two cards against the CPU reference
// (samp_ref, FP64), with synthetic logits and TG1's recorded rows.

namespace {

using namespace fn::samp;
using namespace fn::samp::harness;

constexpr double tol = 2e-5; // a CDF boundary within F32 rounding (normalized; SM1's listed cases)

int g_fail = 0;
FILE * g_log = nullptr; // the listed cases and per-case lines
std::string g_only;     // SAMP_ONLY: run only the cases whose name contains it (diagnosis)
bool g_san = false;     // the san mode's few decisions

bool wanted(const std::string & name) {
    return g_only.empty() || name.find(g_only) != std::string::npos;
}

void line(const char * fmt, ...) __attribute__((format(printf, 1, 2)));
void line(const char * fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    printf("samp-test: %s\n", buf);
    if (g_log != nullptr) {
        fprintf(g_log, "%s\n", buf);
        fflush(g_log);
    }
}

void expect(bool ok, const std::string & what) {
    line("%-100s %s", what.c_str(), ok ? "PASS" : "FAIL");
    g_fail += ok ? 0 : 1;
}

uint64_t uni(const options & o, uint32_t pos, uint32_t purpose) {
    uint64_t u0 = 0, u1 = 0;
    uniforms(o.seed, o.request, pos, purpose, u0, u1);
    return u0;
}

// ---- synthetic rows ----

std::vector<float> gauss_row(uint32_t seed, float sigma, int32_t peaks) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, sigma);
    std::vector<float> l(static_cast<size_t>(V));
    for (float & x : l) {
        x = nd(rng);
    }
    std::uniform_int_distribution<int32_t> ud(0, V - 1);
    for (int32_t i = 0; i < peaks; ++i) {
        l[size_t(ud(rng))] = 12.0f + static_cast<float>(i % 7); // a few strong candidates on both cards
    }
    return l;
}

// TG1's recorded rows (the pinned engine's decode-mode logits, whole rows), read in place through gzip
bool read_recorded(const std::string & path, int32_t n_rows, std::vector<std::vector<float>> & rows) {
    const std::string cmd = "gzip -dc '" + path + "'";
    FILE * f = popen(cmd.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    rows.assign(static_cast<size_t>(n_rows), std::vector<float>(static_cast<size_t>(V)));
    bool ok = true;
    for (int32_t i = 0; i < n_rows && ok; ++i) {
        ok = fread(rows[size_t(i)].data(), 4, size_t(V), f) == size_t(V);
    }
    pclose(f);
    return ok;
}

// ---- the plain-draw cases ----

struct case_res {
    int32_t mism = 0, listed = 0, cards_differ = 0;
    double pv = 0;
};

// N draws (generate, B = 1) of row 0; the same-uniform comparison, both cards, chi-squared; reruns of the first batch
case_res draws(tester & T, const std::string & name, const std::vector<float> & row, int32_t N, bool discard,
               int32_t pos_base = 0) {
    const auto tc0 = std::chrono::steady_clock::now();
    case_res cr;
    std::string err;
    ref_pen pen;
    std::vector<uint32_t> counts;
    std::vector<int32_t> ring;
    if (T.opt.penalties()) {
        ref_window(T.prm.cap, T.ledger, counts, ring);
        pen.counts = counts.data();
    }
    std::vector<float> lt;
    ref_transform(T.opt, row.data(), V, pen, lt);
    ref_dist d;
    if (!ref_distribution(T.opt, lt.data(), V, NC, d, err) || !T.put_logits({row})) {
        expect(false, name + ": reference or upload: " + err);
        cr.mism = -1;
        return cr;
    }
    std::vector<int32_t> toks;
    std::vector<int32_t> first_batch;
    uint32_t batch_errs = 0u;
    for (int32_t i0 = 0; i0 < N; i0 += kmax) {
        const int32_t n = std::min(kmax, N - i0);
        std::vector<dec_in> in(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            in[size_t(i)].n_before = pos_base + 8 * (i0 + i);
            in[size_t(i)].tok[0] = 1;
        }
        std::vector<fn::round::desc> out[2];
        if (!T.run(in, out, 0, discard)) {
            expect(false, name + ": run");
            cr.mism = -1;
            return cr;
        }
        for (int32_t i = 0; i < n; ++i) {
            const int32_t g0 = out[0][size_t(i)].argmax[0], g1 = out[1][size_t(i)].argmax[0];
            cr.cards_differ += g0 != g1 ? 1 : 0;
            toks.push_back(g0);
            const uint32_t pos = static_cast<uint32_t>(in[size_t(i)].n_before + 1);
            const ref_draw_out rd = ref_draw(d, uni(T.opt, pos, p_shard), uni(T.opt, pos, p_local));
            if (rd.token != g0) {
                ++cr.mism;
                const double gap = std::min(std::fabs(rd.target - rd.before), std::fabs(rd.target - rd.after));
                const bool cut = d.cut_at >= 0 && (std::fabs(d.cut_at - d.cut_target) <= tol ||
                                                   std::fabs(d.cut_before - d.cut_target) <= tol);
                if (gap <= tol || cut) {
                    ++cr.listed;
                    line("  listed %s draw %d pos %u: GPU %d CPU %d stage %d u %.17g cum before %.17g after %.17g "
                         "|u - boundary| %.3g%s",
                         name.c_str(), i0 + i, pos, g0, rd.token, rd.stage, rd.target, rd.before, rd.after, gap,
                         cut ? " (top-p cut within F32 rounding)" : "");
                } else {
                    line("  UNLISTED %s draw %d pos %u: GPU %d CPU %d u %.17g before %.17g after %.17g p(GPU) %.6g "
                         "p(CPU) %.6g",
                         name.c_str(), i0 + i, pos, g0, rd.token, rd.target, rd.before, rd.after, ref_prob(d, g0),
                         ref_prob(d, rd.token));
                }
            }
        }
        if (i0 == 0) {
            first_batch.assign(toks.begin(), toks.end());
        }
        uint32_t eb[2] = {0, 0};
        if (!T.errs(eb)) {
            cr.mism = -1;
            return cr;
        }
        if ((eb[0] | eb[1]) != 0u) {
            line("  %s: errors 0x%x 0x%x in the batch of draws %d .. %d", name.c_str(), eb[0], eb[1], i0, i0 + n - 1);
            batch_errs |= eb[0] | eb[1];
        }
    }
    uint32_t e[2] = {batch_errs, batch_errs};
    const bool errs_ok = batch_errs == 0u;
    // the chi-squared statistic over the reference's kept set (bins under 5 expected pooled)
    {
        std::map<int32_t, double> obs;
        for (int32_t t : toks) {
            obs[t] += 1;
        }
        double chi = 0, pool_o = 0, pool_e = 0;
        int32_t bins = 0;
        for (size_t i = 0; i < d.id.size(); ++i) {
            const double ex = N * d.prob[i];
            const double ob = obs.count(d.id[i]) ? obs[d.id[i]] : 0.0;
            obs.erase(d.id[i]);
            if (ex >= 5) {
                chi += (ob - ex) * (ob - ex) / ex;
                ++bins;
            } else {
                pool_o += ob;
                pool_e += ex;
            }
        }
        for (const auto & kv : obs) {
            pool_o += kv.second; // outside the reference's kept set
        }
        if (pool_e >= 5) {
            chi += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
            ++bins;
        } else if (pool_o > 0 && pool_e < 1e-6) {
            chi = 1e30; // draws outside the support
        }
        cr.pv = bins > 1 ? chi2_q(chi, bins - 1) : (chi > 1e29 ? 0.0 : 1.0);
        if (d.greedy || d.id.size() == 1) {
            cr.pv = cr.mism == 0 ? 1.0 : 0.0; // a point mass: every draw is the one token
        }
    }
    // reruns bitwise: the first batch again
    bool rerun = true;
    {
        const int32_t n = std::min(kmax, N);
        std::vector<dec_in> in(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            in[size_t(i)].n_before = pos_base + 8 * i;
            in[size_t(i)].tok[0] = 1;
        }
        std::vector<fn::round::desc> out[2];
        if (T.opt.penalties() && !T.set_options(T.opt, T.ledger, T.prm.window_bins)) {
            rerun = false;
        }
        rerun = rerun && T.run(in, out, 0, discard);
        for (int32_t i = 0; i < n && rerun; ++i) {
            rerun = out[0][size_t(i)].argmax[0] == first_batch[size_t(i)];
        }
        uint32_t er[2] = {0, 0};
        const bool er_ok = T.errs(er);
        if ((er[0] | er[1]) != 0u) {
            line("  %s: errors 0x%x 0x%x in the rerun", name.c_str(), er[0], er[1]);
        }
        rerun = rerun && er_ok && (er[0] | er[1]) == 0u;
    }
    char what[256];
    snprintf(what, sizeof(what),
             "%s: %d draws, chi2 p = %.4f, same-uniform mismatches %d (listed %d), cards differ %d, rerun %s, errors "
             "0x%x 0x%x (%.1f s)",
             name.c_str(), N, cr.pv, cr.mism, cr.listed, cr.cards_differ, rerun ? "bitwise" : "DIFFERS", e[0], e[1],
             std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count());
    expect(cr.pv > 0.001 && cr.mism == cr.listed && cr.cards_differ == 0 && rerun && errs_ok, what);
    return cr;
}

// batch invariance: rows 0 .. 2 decided in one verify round equal each row decided alone at the same position
void batch_invariance(tester & T, const std::string & name, const std::vector<std::vector<float>> & rows) {
    if (!wanted(name + ": batch invariance")) {
        return;
    }
    if (!T.put_logits(rows)) {
        expect(false, name + ": upload");
        return;
    }
    // arbitrary drafts: whatever the accept bits, each row's plain draw is made and recorded (desc.argmax[t])
    const int32_t nb = g_san ? 4 : 64;
    std::vector<dec_in> v(static_cast<size_t>(nb));
    for (int32_t i = 0; i < nb; ++i) {
        v[size_t(i)].kind = static_cast<int32_t>(fn::round::kind::verify);
        v[size_t(i)].rows = 3;
        v[size_t(i)].n_before = 8 * i;
        v[size_t(i)].tok[0] = 1;
        v[size_t(i)].tok[1] = 2;
        v[size_t(i)].tok[2] = 3;
    }
    std::vector<fn::round::desc> vo[2];
    bool ok = T.run(v, vo, 0, true);
    int32_t differ = 0;
    for (int32_t t = 0; t < 3 && ok; ++t) {
        std::vector<dec_in> one(static_cast<size_t>(nb));
        for (int32_t i = 0; i < nb; ++i) {
            one[size_t(i)].n_before = 8 * i + t;
        }
        std::vector<fn::round::desc> go[2];
        ok = T.run(one, go, t, true);
        for (int32_t i = 0; i < nb && ok; ++i) {
            differ += go[0][size_t(i)].argmax[0] != vo[0][size_t(i)].argmax[t] ? 1 : 0;
            differ += go[1][size_t(i)].argmax[0] != vo[1][size_t(i)].argmax[t] ? 1 : 0;
        }
    }
    uint32_t e[2] = {0, 0};
    ok = ok && T.errs(e) && e[0] == 0 && e[1] == 0;
    expect(ok && differ == 0, name + ": batch invariance, 3 rows x " + std::to_string(nb) +
                                  " rounds at B = 3 equal B = 1 (" + std::to_string(differ) + " differ)");
}

// ---- the accept step (step 3) ----

struct verify_case {
    std::string name;
    options o;
    int32_t qkind = 0; // 0: q = p, 1: a different q, 2: q = 1.5 p (the residual empty)
};

void accept_step(tester & T, const verify_case & vc, const std::vector<std::vector<float>> & rows,
                 const std::vector<std::vector<float>> & qrows, int32_t N) {
    if (!wanted(vc.name)) {
        return;
    }
    const auto tc0 = std::chrono::steady_clock::now();
    const int32_t d = max_drafts; // SP2: four drafts, five rows
    std::string err;
    std::vector<ref_dist> p(static_cast<size_t>(d + 1)), qd(static_cast<size_t>(d));
    bool ok = T.set_options(vc.o, T.ledger) && T.put_logits(rows);
    for (int32_t t = 0; t <= d && ok; ++t) {
        ok = ref_distribution(vc.o, rows[size_t(t)].data(), V, NC, p[size_t(t)], err);
    }
    // q: the draft's distribution through the same chain (of p's rows, or other rows), scaled by 1.5 in case 2
    std::vector<std::vector<int32_t>> qid(static_cast<size_t>(d));
    std::vector<std::vector<float>> qp(static_cast<size_t>(d)),
        qsh(static_cast<size_t>(d), std::vector<float>(static_cast<size_t>(V), 0.0f));
    std::vector<ref_q> rq(static_cast<size_t>(d));
    const double scale = vc.qkind == 2 ? 1.5 : 1.0;
    for (int32_t t = 0; t < d && ok; ++t) {
        ok = ref_distribution(vc.o, (vc.qkind == 1 ? qrows[size_t(t)] : rows[size_t(t)]).data(), V, NC, qd[size_t(t)],
                              err);
        for (size_t i = 0; ok && i < qd[size_t(t)].id.size(); ++i) {
            const float qv = static_cast<float>(qd[size_t(t)].prob[i] * scale);
            qid[size_t(t)].push_back(qd[size_t(t)].id[i]);
            qp[size_t(t)].push_back(qv);
            qsh[size_t(t)][size_t(qd[size_t(t)].id[i])] = qv;
        }
    }
    const bool full = path_of(vc.o) == path::full;
    for (int32_t t = 0; t < d && ok; ++t) {
        if (full) {
            rq[size_t(t)].shard[0] = qsh[size_t(t)].data();
            rq[size_t(t)].shard[1] = qsh[size_t(t)].data() + NC;
            rq[size_t(t)].split = NC;
        } else {
            rq[size_t(t)].id = qid[size_t(t)];
            rq[size_t(t)].q = qp[size_t(t)];
        }
    }
    ok = ok && T.put_q(qid, qp, qsh);
    std::vector<ref_resid> rcache;
    for (int32_t t = 0; t < d && ok; ++t) {
        rcache.push_back(ref_make_resid(p[static_cast<size_t>(t)], rq[static_cast<size_t>(t)]));
    }
    // the F32 rounding of the GPU's probabilities, summed per card a row: lt = l / T rounded (|lt| 2^-23 relative in
    // the weight), expf and the integer weight (2^-22): the bound a listed CDF-boundary case is held to
    std::vector<std::array<double, 2>> berr(static_cast<size_t>(d + 1), std::array<double, 2>{0.0, 0.0});
    for (int32_t t = 0; t <= d && ok; ++t) {
        const ref_dist & pd = p[static_cast<size_t>(t)];
        const float T = vc.o.temp > 0.0f ? vc.o.temp : 1.0f;
        for (size_t j = 0; j < pd.id.size(); ++j) {
            const float lt = rows[static_cast<size_t>(t)][static_cast<size_t>(pd.id[j])] / T;
            berr[static_cast<size_t>(t)][pd.id[j] < NC ? 0 : 1] +=
                pd.prob[j] * (std::fabs(static_cast<double>(lt)) * 1.1920928955078125e-7 + 2.384185791015625e-7);
        }
    }
    if (!ok) {
        expect(false, vc.name + ": setup: " + err);
        return;
    }
    // drafts x_{t+1} ~ q_t (normalized), host random numbers
    std::mt19937_64 rng(0x5eed + static_cast<uint64_t>(vc.qkind));
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const auto draw_q = [&](int32_t t) {
        const ref_dist & q = qd[size_t(t)];
        if (q.greedy) {
            return q.id[0];
        }
        const double u = ud(rng);
        double cum = 0;
        for (size_t i = 0; i < q.id.size(); ++i) {
            cum += q.prob[i];
            if (cum > u) {
                return q.id[i];
            }
        }
        return q.id.back();
    };
    std::vector<std::vector<int32_t>> emitted(static_cast<size_t>(d + 1)); // out[t] where present
    int32_t mism = 0, listed = 0, differ = 0, by_a[max_rows] = {};
    for (int32_t i0 = 0; i0 < N && ok; i0 += kmax) {
        const int32_t n = std::min(kmax, N - i0);
        std::vector<dec_in> in(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            dec_in & x = in[size_t(i)];
            x.kind = static_cast<int32_t>(fn::round::kind::verify);
            x.rows = d + 1;
            x.n_before = 8 * (i0 + i);
            x.tok[0] = 1;
            for (int32_t t = 0; t < d; ++t) {
                x.tok[t + 1] = draw_q(t);
            }
        }
        std::vector<fn::round::desc> out[2];
        ok = T.run(in, out, 0, true);
        for (int32_t i = 0; i < n && ok; ++i) {
            const fn::round::desc & g = out[0][size_t(i)];
            const fn::round::desc & g1 = out[1][size_t(i)];
            bool same = g.a == g1.a && g.n_out == g1.n_out;
            for (int32_t t = 0; t < max_rows; ++t) {
                same = same && g.out[t] == g1.out[t];
            }
            differ += same ? 0 : 1;
            const std::vector<int32_t> xs(in[size_t(i)].tok + 1, in[size_t(i)].tok + 1 + d);
            const ref_verify_out r =
                ref_verify(vc.o, p, xs, rq, static_cast<uint32_t>(in[size_t(i)].n_before + 1), &rcache);
            bool match = r.a == g.a && g.n_out == r.a + 1;
            for (int32_t t = 0; t <= r.a && match; ++t) {
                match = g.out[t] == r.out[size_t(t)];
            }
            bool same_prefix = match || r.a == g.a;
            for (int32_t t = 0; t < r.a && same_prefix; ++t) {
                same_prefix = g.out[t] == r.out[size_t(t)];
            }
            if (!match && !same_prefix) {
                ++mism;
                // the accept bits differ: the accept test at the first differing row
                const int32_t t = std::min(r.a, std::max(0, g.a));
                const uint32_t pos = static_cast<uint32_t>(in[size_t(i)].n_before + 1 + t);
                const double ua = unit_double(uni(vc.o, pos, p_accept));
                const double px = ref_prob(p[size_t(t)], t < d ? xs[size_t(t)] : -1);
                const double qx = t < d ? rq[size_t(t)].at(xs[size_t(t)]) : 0;
                const double gap = qx > 0 ? std::fabs(ua * qx - px) : 1.0;
                if (gap <= tol * std::max(qx, 1e-30) + tol * px) {
                    ++listed;
                    line("  listed %s round %d row %d (accept test): GPU a %d CPU a %d u %.17g q(x) %.9g p(x) %.9g "
                         "|u q - p| %.3g",
                         vc.name.c_str(), i0 + i, t, g.a, r.a, ua, qx, px, gap);
                } else {
                    line("  UNLISTED %s round %d: GPU a %d out %d %d %d %d %d, CPU a %d out %d .. %d", vc.name.c_str(),
                         i0 + i, g.a, g.out[0], g.out[1], g.out[2], g.out[3], g.out[4], r.a, r.out[0], r.out.back());
                }
            } else if (!match) {
                // the same accepted run, a different final token (the residual, its fallback, or the bonus): the CPU's
                // target against the GPU token's interval in the CPU's running sums, and the F32 bound of the row
                ++mism;
                const int32_t a = r.a, gt = g.out[a];
                const uint32_t pos = static_cast<uint32_t>(in[size_t(i)].n_before + 1 + a);
                const ref_dist & pd = p[size_t(a)];
                const bool full = pd.p == path::full;
                const bool resid = a < d && !rcache[size_t(a)].empty;
                const auto interval = [](const std::vector<int32_t> & ids, const std::vector<double> & cum, int32_t tk,
                                         double & lo, double & hi) {
                    for (size_t j = 0; j < ids.size(); ++j) {
                        if (ids[j] == tk) {
                            lo = j == 0 ? 0.0 : cum[j - 1];
                            hi = cum[j];
                            return true;
                        }
                    }
                    return false;
                };
                double target = 0, lo = 0, hi = 0, total = 1, bound_num = 0;
                const char * stage = "";
                bool found = false;
                const int32_t gc = gt < NC ? 0 : 1;
                if (resid) {
                    const ref_resid & rr = rcache[size_t(a)];
                    const double tot = rr.R[0] + rr.R[1];
                    if (!full) {
                        stage = "residual over the list";
                        target = unit_double(uni(vc.o, pos, p_resid_shard)) * tot;
                        found = interval(rr.id, rr.cum, gt, lo, hi);
                        total = tot;
                        bound_num = berr[size_t(a)][0] + berr[size_t(a)][1];
                    } else {
                        const double us = unit_double(uni(vc.o, pos, p_resid_shard));
                        if ((us * tot < rr.R[0] ? 0 : 1) != gc) {
                            stage = "residual's shard";
                            target = us * tot;
                            lo = hi = rr.R[0];
                            found = true;
                            total = tot;
                            bound_num = berr[size_t(a)][0] + berr[size_t(a)][1];
                        } else {
                            stage = "residual inside the shard";
                            target = unit_double(uni(vc.o, pos, p_resid_local)) * rr.R[gc];
                            found = interval(rr.cid[gc], rr.ccum[gc], gt, lo, hi);
                            total = rr.R[gc];
                            bound_num = berr[size_t(a)][gc];
                        }
                    }
                } else {
                    // a plain draw from p: the bonus row, or the fallback of an empty residual
                    const uint32_t ps = a == d || full ? p_shard : p_resid_shard;
                    if (!full) {
                        stage = a == d ? "bonus over the list" : "fallback over the list";
                        target = unit_double(uni(vc.o, pos, ps));
                        found = interval(pd.id, pd.cum, gt, lo, hi);
                        bound_num = berr[size_t(a)][0] + berr[size_t(a)][1];
                    } else {
                        const double us = unit_double(uni(vc.o, pos, p_shard));
                        if ((us < pd.mass[0] ? 0 : 1) != gc) {
                            stage = "draw's shard";
                            target = us;
                            lo = hi = pd.mass[0];
                            found = true;
                            bound_num = berr[size_t(a)][0] + berr[size_t(a)][1];
                        } else {
                            stage = "draw inside the shard";
                            target = unit_double(uni(vc.o, pos, p_local)) * pd.mass[gc];
                            found = interval(pd.cid[gc], pd.ccum[gc], gt, lo, hi);
                            total = pd.mass[gc];
                            bound_num = berr[size_t(a)][gc];
                        }
                    }
                }
                const double gap = !found ? 1.0 : (target < lo ? lo - target : target > hi ? target - hi : 0.0) / total;
                const double bound = bound_num / total;
                const bool l = found && gap <= bound;
                listed += l ? 1 : 0;
                line("  %s %s round %d row %d (%s): GPU %d CPU %d, target %.17g, GPU token's running sums %.17g .. "
                     "%.17g, gap %.3g of the total, F32 bound %.3g",
                     l ? "listed" : "UNLISTED", vc.name.c_str(), i0 + i, a, stage, gt, r.out[size_t(a)], target / total,
                     lo / total, hi / total, gap, bound);
            }
            if (g.a >= 0 && g.a <= d) {
                by_a[g.a]++;
            }
            for (int32_t t = 0; t <= g.a && t <= d; ++t) {
                emitted[size_t(t)].push_back(g.out[t]);
            }
        }
    }
    uint32_t e[2] = {0, 0};
    ok = ok && T.errs(e) && e[0] == 0 && e[1] == 0;
    // each emitted position follows its row's p (rows independent): chi-squared per position with enough draws
    std::string pvs;
    bool pass_chi = true;
    for (int32_t t = 0; t <= d; ++t) {
        const std::vector<int32_t> & em = emitted[size_t(t)];
        if (em.size() < 2000) {
            pvs += " out[" + std::to_string(t) + "]: " + std::to_string(em.size()) + " draws (not tested)";
            continue;
        }
        std::map<int32_t, double> obs;
        for (int32_t x : em) {
            obs[x] += 1;
        }
        const ref_dist & pd = p[size_t(t)];
        double chi = 0, pool_o = 0, pool_e = 0;
        int32_t bins = 0;
        for (size_t i = 0; i < pd.id.size(); ++i) {
            const double ex = em.size() * pd.prob[i];
            const double ob = obs.count(pd.id[i]) ? obs[pd.id[i]] : 0.0;
            obs.erase(pd.id[i]);
            if (ex >= 5) {
                chi += (ob - ex) * (ob - ex) / ex;
                ++bins;
            } else {
                pool_o += ob;
                pool_e += ex;
            }
        }
        for (const auto & kv : obs) {
            pool_o += kv.second;
        }
        if (pool_e >= 5) {
            chi += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
            ++bins;
        } else if (pool_o > 0 && pool_e < 1e-6) {
            chi = 1e30;
        }
        const double pv = pd.greedy ? 1.0 : bins > 1 ? chi2_q(chi, bins - 1) : 1.0;
        pass_chi = pass_chi && pv > 0.001;
        char b[96];
        snprintf(b, sizeof(b), " out[%d] p = %.4f (%zu)", t, pv, em.size());
        pvs += b;
    }
    char what[512];
    snprintf(what, sizeof(what),
             "%s: %d rounds, a = 0/1/2/3/4: %d/%d/%d/%d/%d;%s; mismatches %d (listed %d), cards differ %d (%.1f s)",
             vc.name.c_str(), N, by_a[0], by_a[1], by_a[2], by_a[3], by_a[4], pvs.c_str(), mism, listed, differ,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count());
    const bool greedy = vc.o.temp <= 0.0f;
    // coverage (all accepted for q = p; a rejection at each row otherwise) needs more rounds than san's few
    bool rejected_each = true;
    for (int32_t t = 0; t < d; ++t) {
        rejected_each = rejected_each && by_a[t] > 0;
    }
    const bool cover = greedy || g_san || (vc.qkind == 0 ? by_a[d] > 0 : rejected_each);
    expect(ok && pass_chi && mism == listed && differ == 0 && cover, what);
}

// the accept step with penalties: verify row t sees drafts x_1 .. x_t as accepted and the tokens they evict from the
// window (the deltas); the reference recomputes each round's rows with the same deltas (the window kept static)
void accept_penalties(tester & T, const std::vector<std::vector<float>> & rows, int32_t N) {
    const std::string name = "accept, top-k defaults with penalties (row deltas over a full window of 8)";
    if (!wanted(name)) {
        return;
    }
    options o;
    o.seed = 31;
    o.temp = 1.0f;
    o.penalty_last_n = 8;
    o.penalty_repeat = 1.6f;
    o.penalty_freq = 0.4f;
    o.penalty_present = 0.3f;
    std::string err;
    // q: each row's distribution without deltas; the window: the last 8 of a ledger holding some of q's tokens
    const int32_t d = max_drafts; // SP2
    std::vector<ref_dist> q0(static_cast<size_t>(d));
    bool ok = true;
    for (int32_t t = 0; t < d && ok; ++t) {
        ok = ref_distribution(o, rows[static_cast<size_t>(t)].data(), V, NC, q0[static_cast<size_t>(t)], err);
    }
    std::vector<int32_t> led;
    for (int32_t i = 0; i < 20 && ok; ++i) {
        led.push_back(q0[static_cast<size_t>(i % 3)].id[static_cast<size_t>(i % 5)]); // rows 0..2 as before
    }
    std::vector<std::vector<int32_t>> qid(static_cast<size_t>(d));
    std::vector<std::vector<float>> qp(static_cast<size_t>(d)), qsh;
    std::vector<ref_q> rq(static_cast<size_t>(d));
    for (int32_t t = 0; t < d && ok; ++t) {
        for (size_t i = 0; i < q0[static_cast<size_t>(t)].id.size(); ++i) {
            qid[static_cast<size_t>(t)].push_back(q0[static_cast<size_t>(t)].id[i]);
            qp[static_cast<size_t>(t)].push_back(static_cast<float>(q0[static_cast<size_t>(t)].prob[i]));
        }
        rq[static_cast<size_t>(t)].id = qid[static_cast<size_t>(t)];
        rq[static_cast<size_t>(t)].q = qp[static_cast<size_t>(t)];
    }
    ok = ok && T.set_options(o, led) && T.put_logits(rows) && T.put_q(qid, qp, qsh);
    std::vector<uint32_t> counts;
    std::vector<int32_t> ring;
    ref_window(8, led, counts, ring);
    std::mt19937 rng(17);
    int32_t mism = 0, listed = 0, differ = 0, by_a[max_rows] = {};
    for (int32_t i0 = 0; i0 < N && ok; i0 += 64) {
        const int32_t n = std::min(64, N - i0);
        std::vector<dec_in> in(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            dec_in & x = in[static_cast<size_t>(i)];
            x.kind = static_cast<int32_t>(fn::round::kind::verify);
            x.rows = d + 1;
            x.n_before = 8 * (i0 + i);
            x.tok[0] = 1;
            for (int32_t t = 0; t < d; ++t) {
                // drafts among each row's top 6, so the window's and the drafts' penalties move the kept set
                x.tok[t + 1] = q0[static_cast<size_t>(t)].id[static_cast<size_t>(rng() % 6)];
            }
        }
        std::vector<fn::round::desc> out[2];
        ok = T.run(in, out, 0, true);
        for (int32_t i = 0; i < n && ok; ++i) {
            const dec_in & x = in[static_cast<size_t>(i)];
            const std::vector<int32_t> xs(x.tok + 1, x.tok + 1 + d);
            // the rows through the transform with their deltas: x_1 .. x_t accepted, the oldest t window tokens out
            std::vector<ref_dist> p(static_cast<size_t>(d + 1));
            for (int32_t t = 0; t <= d && ok; ++t) {
                ref_pen pen;
                pen.counts = counts.data();
                for (int32_t k = 0; k < t; ++k) {
                    pen.deltas.emplace_back(xs[static_cast<size_t>(k)], 1);
                    const int64_t pos = static_cast<int64_t>(led.size()) + k;
                    if (pos >= 8) {
                        pen.deltas.emplace_back(led[static_cast<size_t>(pos - 8)], -1);
                    }
                }
                std::vector<float> lt;
                ref_transform(o, rows[static_cast<size_t>(t)].data(), V, pen, lt);
                ok = ref_distribution(o, lt.data(), V, NC, p[static_cast<size_t>(t)], err);
            }
            const ref_verify_out r = ref_verify(o, p, xs, rq, static_cast<uint32_t>(x.n_before + 1));
            const fn::round::desc & g = out[0][static_cast<size_t>(i)];
            const fn::round::desc & g1 = out[1][static_cast<size_t>(i)];
            differ += g.a != g1.a || g.out[0] != g1.out[0] || g.out[g.a] != g1.out[g1.a] ? 1 : 0;
            bool match = r.a == g.a;
            for (int32_t t = 0; t <= r.a && match; ++t) {
                match = g.out[t] == r.out[static_cast<size_t>(t)];
            }
            if (g.a >= 0 && g.a <= d) {
                by_a[g.a]++;
            }
            if (!match) {
                ++mism;
                const int32_t t = std::min(r.a, std::max(0, g.a));
                const uint32_t pos = static_cast<uint32_t>(x.n_before + 1 + t);
                const double ua = unit_double(uni(o, pos, p_accept));
                const double px = ref_prob(p[static_cast<size_t>(t)], t < d ? xs[static_cast<size_t>(t)] : -1);
                const double qx = t < d ? rq[static_cast<size_t>(t)].at(xs[static_cast<size_t>(t)]) : 0;
                const double gap = qx > 0 ? std::fabs(ua * qx - px) : 1.0;
                const bool l = gap <= tol * std::max(qx, 1e-30) + tol * px;
                listed += l ? 1 : 0;
                line("  %s %s round %d: GPU a %d CPU a %d u %.17g q(x) %.9g p(x) %.9g", l ? "listed" : "UNLISTED",
                     name.c_str(), i0 + i, g.a, r.a, ua, qx, px);
            }
        }
    }
    uint32_t e[2] = {0, 0};
    ok = ok && T.errs(e) && e[0] == 0 && e[1] == 0;
    char what[400];
    snprintf(what, sizeof(what),
             "%s: %d rounds, a = 0/1/2/3/4: %d/%d/%d/%d/%d; same-uniform mismatches %d (listed %d), "
             "cards differ %d",
             name.c_str(), N, by_a[0], by_a[1], by_a[2], by_a[3], by_a[4], mism, listed, differ);
    expect(ok && mism == listed && differ == 0 && by_a[1] + by_a[2] + by_a[3] + by_a[4] > 0, what);
}

// ---- the decision hash (step 4) ----

bool flip_record(tester & T, int32_t r) {
    record rec = {};
    cudaSetDevice(T.c[r]->dev);
    if (!read_record(T.c[r]->v, rec, T.c[r]->st)) {
        return false;
    }
    rec.words[2] ^= 1u;
    return cuda_ok(cudaMemcpy(T.c[r]->rec.p, &rec, sizeof(rec), cudaMemcpyHostToDevice), "flip");
}

void hash_tests(tester & T, const std::string & name, const std::vector<float> & row) {
    if (!wanted(name)) {
        return;
    }
    bool ok = T.put_logits({row});
    std::vector<dec_in> one(1);
    std::vector<fn::round::desc> out[2];
    uint32_t e[2] = {0, 0};
    record r0 = {}, r1 = {};
    // a decision, then the next: the records agree, the first is checked by the second, no error
    one[0].n_before = 100;
    ok = ok && T.run(one, out) && T.errs(e);
    const bool clean0 = e[0] == 0 && e[1] == 0;
    ok = ok && read_record(T.c[0]->v, r0, T.c[0]->st) && read_record(T.c[1]->v, r1, T.c[1]->st);
    const bool unchecked = r0.checked == 0 && r1.checked == 0 && r0.valid == 1 && r0.hash == r1.hash;
    one[0].n_before = 108;
    ok = ok && T.run(one, out) && T.errs(e);
    const bool clean1 = e[0] == 0 && e[1] == 0;
    expect(ok && clean0 && clean1 && unchecked,
           name + ": records equal; a record reads unchecked until the next decision compares it");
    // the closing check runs: checked, no error
    for (int32_t r = 0; r < 2 && ok; ++r) {
        cudaSetDevice(T.loop ? T.c[0]->dev : T.c[r]->dev);
        ok = close_send(T.c[r]->v, T.st(r));
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        cudaSetDevice(T.loop ? T.c[0]->dev : T.c[r]->dev);
        ok = close_recv(T.c[r]->v, T.st(r));
    }
    ok =
        ok && T.sync() && T.errs(e) && read_record(T.c[0]->v, r0, T.c[0]->st) && read_record(T.c[1]->v, r1, T.c[1]->st);
    expect(ok && e[0] == 0 && e[1] == 0 && r0.checked == 1 && r1.checked == 1,
           name + ": the closing check runs before the last step returns: checked on both cards, no error");
    // a flipped hash on card 0: the next decision sets err_hash on both cards (the round commits nothing, FC-RD2)
    ok = ok && flip_record(T, 0);
    one[0].n_before = 116;
    ok = ok && T.run(one, out) && T.errs(e);
    expect(ok && (e[0] & err_hash) != 0 && (e[1] & err_hash) != 0,
           name + ": a flipped hash on card 0 sets err_hash on both cards at the next decision");
    // a flipped hash on card 1 caught by the closing check
    one[0].n_before = 124;
    ok = ok && T.run(one, out) && T.errs(e) && flip_record(T, 1);
    for (int32_t r = 0; r < 2 && ok; ++r) {
        cudaSetDevice(T.loop ? T.c[0]->dev : T.c[r]->dev);
        ok = close_send(T.c[r]->v, T.st(r));
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        cudaSetDevice(T.loop ? T.c[0]->dev : T.c[r]->dev);
        ok = close_recv(T.c[r]->v, T.st(r));
    }
    ok = ok && T.sync() && T.errs(e) && read_record(T.c[0]->v, r0, T.c[0]->st);
    expect(ok && (e[0] & err_hash) != 0 && (e[1] & err_hash) != 0 && r0.checked == 0,
           name + ": a flipped hash on card 1 fails the closing check on both cards; the record stays unchecked");
    // the record of a failed round discarded on both cards: the next decision compares equal invalid records
    for (int32_t r = 0; r < 2 && ok; ++r) {
        cudaSetDevice(T.c[r]->dev);
        ok = discard_record(T.c[r]->v, T.c[r]->st) && cuda_ok(cudaStreamSynchronize(T.c[r]->st), "discard");
    }
    one[0].n_before = 132;
    ok = ok && T.run(one, out) && T.errs(e);
    expect(ok && e[0] == 0 && e[1] == 0, name + ": after both cards discard the record, the next decision is clean");
}

// ---- penalties (step 5): the tables after commits, and a truncate's rewind, against the clean replay ----

void penalty_tables(tester & T, const std::vector<float> & row) {
    if (!wanted("penalties: tables")) {
        return;
    }
    options o;
    o.penalty_last_n = 64;
    o.penalty_repeat = 1.3f;
    o.penalty_freq = 0.2f;
    o.penalty_present = 0.1f;
    o.top_k = 20;
    std::mt19937 rng(99);
    std::uniform_int_distribution<int32_t> ud(0, V - 1);
    std::vector<int32_t> led(100);
    for (int32_t & t : led) {
        t = ud(rng);
    }
    bool ok = T.set_options(o, led) && T.put_logits({row});
    // 40 generate decisions; each emitted token joins the ledger (the device applies it at the next decision)
    for (int32_t i = 0; i < 40 && ok; ++i) {
        std::vector<dec_in> one(1);
        one[0].n_before = 1000 + 8 * i;
        std::vector<fn::round::desc> out[2];
        ok = T.run(one, out);
        led.push_back(out[0][0].argmax[0]);
    }
    std::vector<uint32_t> rc;
    std::vector<int32_t> rr;
    const auto compare = [&](const std::vector<int32_t> & ledger, const char * what) {
        std::vector<uint32_t> gc(static_cast<size_t>(V));
        std::vector<int32_t> gr(64);
        int32_t bad = 0;
        ref_window(64, ledger, rc, rr);
        for (int32_t r = 0; r < 2 && ok; ++r) {
            cudaSetDevice(T.c[r]->dev);
            ok = cuda_ok(cudaMemcpy(gc.data(), T.c[r]->counts.p, size_t(V) * 4, cudaMemcpyDeviceToHost), "counts") &&
                 cuda_ok(cudaMemcpy(gr.data(), T.c[r]->ring.p, 64 * 4, cudaMemcpyDeviceToHost), "ring");
            bad += gc != rc ? 1 : 0;
            bad += gr != std::vector<int32_t>(rr.begin(), rr.begin() + 64) ? 1 : 0;
        }
        expect(ok && bad == 0, std::string("penalties: ") + what);
    };
    // the last decision's tokens joined by a rewind that removes nothing (it applies the record first)
    for (int32_t r = 0; r < 2 && ok; ++r) {
        cudaSetDevice(T.c[r]->dev);
        ok = rewind(T.c[r]->v, 64, led, static_cast<int64_t>(led.size()), T.c[r]->st);
    }
    compare(led, "counts and ring after 40 commits equal the clean replay of the ledger, bitwise, both cards");
    // a truncate of 7 tokens (deeper than the window's churn) rewinds to the clean replay of the kept ledger
    const std::vector<int32_t> kept(led.begin(), led.end() - 7);
    for (int32_t r = 0; r < 2 && ok; ++r) {
        cudaSetDevice(T.c[r]->dev);
        ok = rewind(T.c[r]->v, 64, led, static_cast<int64_t>(kept.size()), T.c[r]->st);
    }
    compare(kept, "a truncate of 7 tokens rewinds them bitwise to the clean replay's");
    uint32_t e[2] = {0, 0};
    expect(T.errs(e) && e[0] == 0 && e[1] == 0, "penalties: no error words");
}

// ---- the draft's pick (SM0 section 5, last item): top-k path over the re-sharded draft head's ID map ----

void draft_pick(tester & T, int32_t N) {
    if (!wanted("draft pick")) {
        return;
    }
    options o; // the pinned defaults
    o.seed = 77;
    bool ok = T.set_options(o, {});
    // the draft subset: card r takes the sorted subset's ranks r, r + 2, ... (DR1); here 32,768 IDs a card
    constexpr int32_t nd = 32768;
    std::vector<int32_t> subset(2 * nd);
    for (int32_t i = 0; i < 2 * nd; ++i) {
        subset[size_t(i)] = i * 3 + (i % 3); // strictly increasing, spread over both target shards
    }
    std::vector<float> full = gauss_row(4242, 3.0f, 30);
    std::vector<float> dl[2] = {std::vector<float>(nd), std::vector<float>(nd)};
    std::vector<int32_t> ids[2] = {std::vector<int32_t>(nd), std::vector<int32_t>(nd)};
    std::vector<float> sub_logits(static_cast<size_t>(V),
                                  NAN); // the reference: the subset's logits, NaN elsewhere (never kept)
    for (int32_t i = 0; i < 2 * nd; ++i) {
        const int32_t r = i % 2;
        ids[r][size_t(i / 2)] = subset[size_t(i)];
        dl[r][size_t(i / 2)] = full[size_t(subset[size_t(i)])];
        sub_logits[size_t(subset[size_t(i)])] = full[size_t(subset[size_t(i)])];
    }
    std::string err;
    ref_dist d;
    ok = ok && ref_distribution(o, sub_logits.data(), V, NC, d, err);
    int32_t * dids[2] = {nullptr, nullptr};
    for (int32_t r = 0; r < 2 && ok; ++r) {
        cudaSetDevice(T.c[r]->dev);
        ok = cuda_ok(cudaMalloc(&dids[r], size_t(nd) * 4), "ids") &&
             cuda_ok(cudaMemcpy(dids[r], ids[r].data(), size_t(nd) * 4, cudaMemcpyHostToDevice), "ids") &&
             cuda_ok(cudaMemcpy(T.c[r]->logits.p, dl[r].data(), size_t(nd) * 4, cudaMemcpyHostToDevice), "logits");
    }
    std::vector<int32_t> toks;
    int32_t mism = 0, listed = 0, differ = 0, qbad = 0;
    for (int32_t i = 0; i < N && ok; ++i) {
        for (int32_t stg = 0; stg < 2 && ok; ++stg) {
            for (int32_t r = 0; r < 2 && ok; ++r) {
                cudaSetDevice(T.loop ? T.c[0]->dev : T.c[r]->dev);
                pick_io io;
                io.logits = T.c[r]->logits.p;
                io.n = nd;
                io.ids = dids[r];
                io.position = static_cast<uint32_t>(5000 + i);
                io.t = i % 3;
                io.proposal = T.c[r]->prop.p;
                ok = pick_stage(T.c[r]->v, io, stg, T.st(r));
            }
        }
        if (i % 3 != 2 && i + 1 < N) {
            continue; // three picks (rows 0, 1, 2) then a read-back
        }
        ok = ok && T.sync();
        int32_t pr[2] = {-1, -1};
        qlist ql[2];
        for (int32_t r = 0; r < 2 && ok; ++r) {
            cudaSetDevice(T.c[r]->dev);
            ok = cuda_ok(cudaMemcpy(&pr[r], T.c[r]->prop.p, 4, cudaMemcpyDeviceToHost), "proposal") &&
                 cuda_ok(cudaMemcpy(&ql[r], T.c[r]->ql.p + (i % 3), sizeof(qlist), cudaMemcpyDeviceToHost), "q");
        }
        differ += pr[0] != pr[1] ? 1 : 0;
        toks.push_back(pr[0]);
        const uint32_t pos = static_cast<uint32_t>(5000 + i);
        const ref_draw_out rd = ref_draw(d, uni(o, pos, p_draft), 0);
        if (rd.token != pr[0]) {
            ++mism;
            const double gap = std::min(std::fabs(rd.target - rd.before), std::fabs(rd.target - rd.after));
            listed += gap <= tol ? 1 : 0;
            line("  %s draft pick %d: GPU %d CPU %d u %.17g before %.17g after %.17g",
                 gap <= tol ? "listed" : "UNLISTED", i, pr[0], rd.token, rd.target, rd.before, rd.after);
        }
        // q equals the reference's kept list and probabilities (F32 of the integer weights against FP64)
        if (ql[0].n != static_cast<int32_t>(d.id.size()) || memcmp(&ql[0], &ql[1], sizeof(qlist)) != 0) {
            ++qbad;
        } else {
            for (int32_t j = 0; j < ql[0].n; ++j) {
                if (ql[0].id[j] != d.id[size_t(j)] || std::fabs(ql[0].q[j] - d.prob[size_t(j)]) > 1e-5) {
                    ++qbad;
                    break;
                }
            }
        }
    }
    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(T.c[r]->dev);
        cudaFree(dids[r]);
    }
    uint32_t e[2] = {0, 0};
    ok = ok && T.errs(e) && e[0] == 0 && e[1] == 0;
    expect(ok && mism == listed && differ == 0 && qbad == 0,
           "draft pick (ID map, 32,768 a card): " + std::to_string(toks.size()) + " picks checked, same-uniform " +
               std::to_string(mism) + " (listed " + std::to_string(listed) + "), cards differ " +
               std::to_string(differ) + ", q lists differing from the reference " + std::to_string(qbad));
}

// a row whose top-p cut falls on a single-member bin (a cut on a bin edge): five strong tokens 1/4 logit apart
std::vector<float> edge_row() {
    std::vector<float> l = gauss_row(31, 1.0f, 0);
    for (float & x : l) {
        x -= 30.0f;
    }
    const int32_t ids[5] = {7, 130000, 55, 200000, 124000};
    const float vals[5] = {10.0f, 9.5f, 9.0f, 8.5f, 8.0f};
    for (int32_t i = 0; i < 5; ++i) {
        l[size_t(ids[i])] = vals[i];
    }
    return l;
}

} // namespace

int fnfl_samp_test(int dev0, int dev1, const char * mode, const char * recorded, const char * out_path) {
    const std::string md = mode;
    const bool san = md == "san";
    if (!san && md != "all" && md != "quick") {
        fprintf(stderr, "samp-test: mode is all, quick or san\n");
        return 2;
    }
    if (dev0 == dev1) {
        printf("samp-test: loopback on device %d: both cards' kernels on one stream, sends before reads\n", dev0);
    } else if (!fn::xchg::enable_peer_access(dev0, dev1)) {
        printf("samp-test: FAIL (no peer access)\n");
        return 1;
    }
    g_fail = 0;
    printf(
        "samp-test: R1's sampler parts a card (counted): params and record %zu + %zu B, bias %d B, counts %d B, ring "
        "4 B a cell, q lists %zu B, q shards %d B, workspace %zu B; exchange rings %lld B a card with the sampler's "
        "slots\n",
        sizeof(params), sizeof(record), 4 * V, 4 * V, size_t(max_drafts) * sizeof(qlist), 4 * max_drafts * NC,
        workspace_bytes(), static_cast<long long>(fn::xchg::rings_card_bytes()));
    g_san = san;
    setvbuf(stdout, nullptr, _IOLBF, 0); // every case's line out as it completes
    const char * only = getenv("SAMP_ONLY");
    g_only = only != nullptr ? only : "";
    g_log = out_path != nullptr && out_path[0] != 0 ? fopen(out_path, "w") : nullptr;
    // san: a few decisions a case, every kernel and exchange still taken (Compute Sanitizer)
    const int32_t N = san ? 4 : md == "quick" ? 5000 : 100000;
    const int32_t NR = san ? 2 : md == "quick" ? 2000 : 10000;
    const int32_t NV = san ? 8 : md == "quick" ? 5000 : 100000;
    {
        card_s k0(dev0), k1(dev1);
        fn::xchg::rings rg(dev0, dev1);
        tester T;
        T.c[0] = &k0;
        T.c[1] = &k1;
        T.rg = &rg;
        T.loop = dev0 == dev1;
        if (!rg.ok() || !T.init()) {
            printf("samp-test: FAIL (setup)\n");
            return 1;
        }
        const std::vector<float> g1 = gauss_row(1, 3.0f, 40), g2 = gauss_row(2, 2.0f, 0), g3 = gauss_row(3, 4.0f, 10);
        options base;
        base.seed = 1234;
        base.request = 7;
        const auto run_case = [&](const std::string & name, options o, const std::vector<float> & row, int32_t n,
                                  int32_t wbins = hist_bins, const std::vector<int32_t> & led = {}) {
            if (!wanted(name)) {
                return;
            }
            if (!T.set_options(o, led, wbins)) {
                expect(false, name + ": options");
                return;
            }
            draws(T, name, row, n, o.penalties());
        };
        // step 2: the paths
        {
            options o = base;
            o.temp = 0.0f;
            run_case("greedy (T = 0, V1's kernels with the hash)", o, g1, san ? 4 : 2000);
            o.bias = {{5, 30.0f}};
            run_case("greedy with a bias (top-k 1, greedy mode)", o, g1, san ? 4 : 2000);
        }
        {
            options o = base; // the pinned server's defaults
            run_case("top-k 20, top-p 0.95, min-p 0.05, T 0.8 (defaults)", o, g1, N);
            o.top_k = 256;
            o.top_p = 1.0f;
            o.min_p = 0.0f;
            o.temp = 1.3f;
            std::vector<float> ties = g2;
            for (float & x : ties) {
                x = std::round(x * 4.0f) / 4.0f; // many exact ties
            }
            run_case("top-k 256 alone, ties to the lower ID, T 1.3", o, ties, N);
            if (wanted("top-k 256 alone")) {
                // each card's sent list against the CPU's top 256 of its shard, by (logit, lower ID)
                int32_t bad = 0;
                for (int32_t r = 0; r < 2; ++r) {
                    std::vector<uint32_t> w;
                    cudaSetDevice(T.c[r]->dev);
                    if (!read_list(T.c[r]->v, 0, w, T.c[r]->st)) {
                        ++bad;
                        continue;
                    }
                    std::vector<std::pair<float, int32_t>> cpu;
                    for (int32_t i = 0; i < NC; ++i) {
                        cpu.emplace_back(ties[size_t(r) * NC + size_t(i)], r * NC + i);
                    }
                    std::partial_sort(cpu.begin(), cpu.begin() + 256, cpu.end(), [](const auto & x, const auto & y) {
                        return x.first > y.first || (x.first == y.first && x.second < y.second);
                    });
                    const int32_t n = static_cast<int32_t>(w[0]);
                    for (int32_t i = 0; i < 256; ++i) {
                        float f = 0.0f;
                        memcpy(&f, &w[size_t(4 + 2 * i)], 4); // the sent logit's F32 bits
                        if (n != 256 || static_cast<int32_t>(w[size_t(5 + 2 * i)]) != cpu[size_t(i)].second ||
                            f != cpu[size_t(i)].first) {
                            if (bad < 6) {
                                line("  card %d list entry %d: GPU (%g, %d) CPU (%g, %d)", r, i, f,
                                     static_cast<int32_t>(w[size_t(5 + 2 * i)]), cpu[size_t(i)].first,
                                     cpu[size_t(i)].second);
                            }
                            ++bad;
                        }
                    }
                }
                expect(bad == 0, "top-k 256 with ties: each card's sent list equals the CPU's top 256 of its shard (" +
                                     std::to_string(bad) + " differ)");
            }
        }
        {
            options o = base;
            o.top_k = 0;
            o.top_p = 1.0f;
            o.min_p = 0.0f;
            run_case("no filter, T 0.8", o, g2, N);
            o.min_p = 0.05f;
            run_case("full-vocabulary min-p 0.05", o, g1, N);
            o.min_p = 0.0f;
            o.top_p = 0.95f;
            run_case("full-vocabulary top-p 0.95 (the split bin)", o, g2, N);
            o.min_p = 0.05f;
            run_case("full-vocabulary top-p 0.95 and min-p 0.05", o, g1, N);
            o.min_p = 0.0f;
            o.top_p = 0.9f;
            o.temp = 1.0f;
            run_case("top-p 0.9: the cut on a bin edge (a single-member bin)", o, edge_row(), N);
            o.top_p = 0.95f;
            // the tail: 50 strong tokens over [8, 10] (a few inside a 16-bin window, 1/4 logit), 2,000 tokens over
            // [4.5, 5.5] on both cards, the rest at -40 (no mass): about 2,050 members below the window hold the cut
            std::vector<float> tail(static_cast<size_t>(V), -40.0f);
            {
                std::mt19937 rng(77);
                std::uniform_real_distribution<float> top(8.0f, 10.0f), mid(4.5f, 5.5f);
                for (int32_t i = 0; i < 50; ++i) {
                    tail[static_cast<size_t>(i * 4801 + 3)] = top(rng);
                }
                for (int32_t i = 0; i < 2000; ++i) {
                    tail[static_cast<size_t>(i * 123 + 7)] = mid(rng);
                }
            }
            run_case("top-p 0.95, window 16 bins (1/4 logit): the cut in the tail bin, its members", o, tail, N, 16);
            std::vector<float> flat(static_cast<size_t>(V), -40.0f);
            for (int32_t i = 0; i < 6000; ++i) {
                flat[size_t(i * 41)] = 0.0f; // 6,000 equal logits on both cards: one bin, ties by ID
            }
            o.top_p = 0.5f;
            run_case("top-p 0.5 over 6,000 tied logits (members on both cards, ties by ID)", o, flat, N);
        }
        {
            options o = base;
            o.penalty_last_n = 64;
            o.penalty_repeat = 1.5f;
            o.penalty_freq = 0.3f;
            o.penalty_present = 0.2f;
            o.bias = {{17, -INFINITY}, {200000, 2.0f}};
            std::vector<int32_t> led;
            std::mt19937 rng(5);
            std::uniform_int_distribution<int32_t> ud(0, 2000);
            for (int32_t i = 0; i < 200; ++i) {
                led.push_back(ud(rng) * 97 % V);
            }
            run_case("penalties and bias, top-k defaults", o, g3, N, hist_bins, led);
            o.top_k = 0;
            run_case("penalties and bias, full top-p and min-p", o, g3, N, hist_bins, led);
        }
        // recorded rows (TG1's decode-mode reference logits)
        if (recorded != nullptr && recorded[0] != 0) {
            std::vector<std::vector<float>> rec;
            if (!read_recorded(recorded, 4, rec)) {
                expect(false, std::string("recorded rows: ") + recorded);
            } else {
                for (int32_t i = 0; i < 4; ++i) {
                    options o = base;
                    run_case("recorded row " + std::to_string(i) + " (position " + std::to_string(8 * i + 7) +
                                 "), defaults",
                             o, rec[size_t(i)], NR);
                    o.top_k = 0;
                    run_case("recorded row " + std::to_string(i) + ", full top-p 0.95 and min-p 0.05", o,
                             rec[size_t(i)], NR);
                }
            }
        }
        // batch invariance on the top-k and full paths
        {
            options o = base;
            T.set_options(o, {});
            batch_invariance(T, "top-k defaults", {g1, g2, g3});
            o.top_k = 0;
            T.set_options(o, {});
            batch_invariance(T, "full top-p and min-p", {g1, g2, g3});
            o.top_p = 1.0f;
            o.min_p = 0.0f;
            T.set_options(o, {});
            batch_invariance(T, "no filter", {g1, g2, g3});
        }
        // step 3: the accept step
        {
            // SP2: five rows, four drafts
            const std::vector<std::vector<float>> rows = {gauss_row(11, 2.5f, 20), gauss_row(12, 2.5f, 20),
                                                          gauss_row(13, 2.5f, 20), gauss_row(14, 2.5f, 20),
                                                          gauss_row(15, 2.5f, 20)};
            // q's rows: p's rows perturbed (each logit plus noise of 0.7), so drafts are rejected at every row
            std::vector<std::vector<float>> qrows(static_cast<size_t>(max_drafts));
            for (int32_t t = 0; t < max_drafts; ++t) {
                const std::vector<float> noise = gauss_row(21 + t, 0.7f, 0);
                qrows[size_t(t)] = rows[size_t(t)];
                for (size_t i = 0; i < noise.size(); ++i) {
                    qrows[size_t(t)][i] += noise[i];
                }
            }
            std::vector<verify_case> vcs;
            options o = base;
            o.temp = 1.0f;
            const char * qn[3] = {"q = p", "q different", "q = 1.5 p (empty residual)"};
            for (int32_t qk = 0; qk < 3; ++qk) {
                vcs.push_back({std::string("accept, top-k defaults, ") + qn[qk], o, qk});
            }
            options f = o;
            f.top_k = 0;
            f.top_p = 1.0f;
            f.min_p = 0.0f;
            for (int32_t qk = 0; qk < 3; ++qk) {
                vcs.push_back({std::string("accept, no filter, ") + qn[qk], f, qk});
            }
            f.min_p = 0.05f;
            vcs.push_back({"accept, full min-p, q different", f, 1});
            f.min_p = 0.0f;
            f.top_p = 0.95f;
            for (int32_t qk = 0; qk < 3; ++qk) {
                vcs.push_back({std::string("accept, full top-p 0.95, ") + qn[qk], f, qk});
            }
            for (const verify_case & vc : vcs) {
                accept_step(T, vc, rows, qrows, NV);
            }
            accept_penalties(T, rows, san ? 8 : 300);
            options g = base;
            g.temp = 0.0f;
            accept_step(T, {"accept, greedy (T = 0), drafts = the argmax (all accepted)", g, 0}, rows, qrows,
                        san ? 4 : 2000);
            accept_step(T, {"accept, greedy (T = 0), other drafts (accepted iff the argmax)", g, 1}, rows, qrows,
                        san ? 4 : 2000);
        }
        // step 4: the hash
        {
            options o = base;
            T.set_options(o, {});
            hash_tests(T, "hash, top-k path", g1);
            o.top_k = 0;
            T.set_options(o, {});
            hash_tests(T, "hash, full path", g1);
            o.temp = 0.0f;
            T.set_options(o, {});
            hash_tests(T, "hash, greedy path (T-GREEDY)", g1);
        }
        // step 5: penalties
        penalty_tables(T, g1);
        // the draft's pick
        draft_pick(T, san ? 3 : 3000);
    }
    printf("samp-test: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    if (g_log != nullptr) {
        fprintf(g_log, "samp-test: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
        fclose(g_log);
    }
    return g_fail == 0 ? 0 : 1;
}
