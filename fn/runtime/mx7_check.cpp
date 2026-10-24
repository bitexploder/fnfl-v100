// MX7's model-level checks (mx7_check.h)

#include "fn/runtime/mx7_check.h"

#include "fn/kernels/head/head.h"
#include "fn/runtime/target.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

constexpr int32_t V = fn::head::n_vocab;
constexpr int32_t VC = fn::head::n_vocab_card;
constexpr int32_t n_prefix = 20000;
constexpr int32_t n_ends = 32;        // the append's prompt ends before its full length: N - 1 .. N - 32
constexpr int32_t appends[1] = {650}; // the KL amendment's one append

bool read_ids(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path);
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

uint64_t fnv(const void * p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 0x100000001b3ull;
    }
    return h;
}

// KL(p || q) of two logit rows [V]
double kl(const float * p, const float * q) {
    double mp = -HUGE_VAL, mq = -HUGE_VAL;
    for (int32_t i = 0; i < V; ++i) {
        mp = std::max(mp, double(p[i]));
        mq = std::max(mq, double(q[i]));
    }
    double sp = 0, sq = 0;
    for (int32_t i = 0; i < V; ++i) {
        sp += std::exp(double(p[i]) - mp);
        sq += std::exp(double(q[i]) - mq);
    }
    const double lsp = std::log(sp), lsq = std::log(sq);
    double k = 0;
    for (int32_t i = 0; i < V; ++i) {
        const double lp = double(p[i]) - mp - lsp, lq = double(q[i]) - mq - lsq;
        k += std::exp(lp) * (lp - lq);
    }
    return k;
}

int32_t top1(const float * p) {
    return int32_t(std::max_element(p, p + V) - p);
}

// one run's evidence: the decode-ready state's bytes (both cards) and the 33 logit rows [33][V]
struct run_out {
    std::string share;
    int32_t n = 0;
    std::vector<uint8_t> state;
    std::vector<float> logits;
    std::string counters;
    int32_t ends = n_ends;               // the shorter prompts run (shadow: none)
    std::vector<mx7_shadow_pair> shadow; // the shadow's pairs
};

// the card logits' row 0 (the retained row after a prefill or a one-row forced round) into row [V]
bool logits_row(target & t, float * row, std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        if (cudaSetDevice(t.card(c).c.device) != cudaSuccess ||
            cudaMemcpy(row + int64_t(c) * VC, t.card(c).logits, size_t(VC) * 4, cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
            err = "the logits' copy";
            return false;
        }
    }
    return true;
}

bool one_run(engine & e, const mx7_check_args & a, const ready_state & s0, const std::vector<int32_t> & ids, int32_t n,
             const std::string & share, run_out & o, std::string & err) {
    o.share = share;
    o.n = n;
    const bool shadow = share == "shadow";
    const double x = share == "auto" ? -1.0 : shadow ? 1.0 : std::atof(share.c_str());
    if (!a.set_share(x, err)) {
        return false;
    }
    if (shadow && (!a.set_shadow || !a.take_shadow)) {
        err = "the shadow needs the cache's shadow hooks";
        return false;
    }
    if (a.set_shadow) {
        a.set_shadow(shadow);
    }
    o.ends = shadow ? 0 : n_ends;
    o.logits.assign(size_t(n_ends + 1) * V, 0.0f);
    // the full append first (its state and counters), then the shorter ones, each from the kept prefix
    for (int32_t k = 0; k <= o.ends; ++k) {
        int32_t np = 0;
        if (e.restore_ready(s0) != run_status::ok) {
            err = e.error();
            return false;
        }
        if (e.prefill(0, ids.data() + n_prefix, n - k, n_prefix, np) != run_status::ok || np != n - k) {
            err = "the append's prefill (" + std::to_string(n - k) + " rows): " + e.error();
            return false;
        }
        if (!logits_row(e.test_target(), o.logits.data() + int64_t(k) * V, err)) {
            return false;
        }
        if (k == 0) {
            o.counters = a.counters ? a.counters() : std::string();
            ready_state s;
            if (e.save_ready(s) != run_status::ok) {
                err = e.error();
                return false;
            }
            o.state = s.dev[0];
            o.state.insert(o.state.end(), s.dev[1].begin(), s.dev[1].end());
        }
    }
    if (shadow) {
        o.shadow = a.take_shadow();
        a.set_shadow(false);
    }
    return true;
}

// nearest rank: the smallest value at or above fraction q of n sorted values
double rank(std::vector<double> v, double q) {
    if (v.empty()) {
        return 0;
    }
    std::sort(v.begin(), v.end());
    const size_t i = size_t(std::max(1.0, std::ceil(q * double(v.size())))) - 1;
    return v[std::min(i, v.size() - 1)];
}

double entropy(const float * p) {
    double mp = p[0];
    for (int32_t i = 1; i < V; ++i) {
        mp = std::max(mp, double(p[i]));
    }
    double z = 0;
    for (int32_t i = 0; i < V; ++i) {
        z += std::exp(double(p[i]) - mp);
    }
    const double lz = std::log(z);
    double h = 0;
    for (int32_t i = 0; i < V; ++i) {
        const double lp = double(p[i]) - mp - lz;
        h -= std::exp(lp) * lp;
    }
    return h;
}

// a group of pairs' CPU-GPU errors (and the referenced ones' against FP64): count, p50, p99, max
void group_line(const char * what, const std::vector<const mx7_shadow_pair *> & g) {
    std::vector<double> cg, cpu, gpu, q1;
    for (const mx7_shadow_pair * r : g) {
        cg.push_back(r->e_cg);
        if (r->e_cpu >= 0) {
            cpu.push_back(r->e_cpu);
            gpu.push_back(r->e_gpu);
            q1.push_back(r->e_q1);
        }
    }
    printf("mx7-check: shadow %s: %zu pairs, CPU-GPU p50 %.4g p99 %.4g max %.4g; %zu referenced: CPU-ref p50 %.4g max "
           "%.4g, GPU-ref p50 %.4g max %.4g, Q8_K-ref p50 %.4g max %.4g\n",
           what, cg.size(), rank(cg, 0.5), rank(cg, 0.99), rank(cg, 1.0), cpu.size(), rank(cpu, 0.5), rank(cpu, 1.0),
           rank(gpu, 0.5), rank(gpu, 1.0), rank(q1, 0.5), rank(q1, 1.0));
}

// the shadow's report: the pairs' TSV, the groups, the 10 worst; true when no pair looks like a wiring fault (CPU-GPU
// and, where referenced, CPU-ref below 0.5: a wrong row, expert or weight gives about 1 or more)
bool shadow_report(const mx7_check_args & a, const std::vector<mx7_shadow_pair> & sp) {
    const std::string path = a.out_dir + "/shadow-pairs.tsv";
    FILE * f = fopen(path.c_str(), "w");
    if (f != nullptr) {
        fprintf(f, "layer\texpert\trow\tslot\tjob_row\tgu\tdn\tweight\te_cpu_gpu\te_cpu_ref\te_gpu_ref\te_q8k_ref\n");
        for (const mx7_shadow_pair & r : sp) {
            fprintf(f, "%d\t%d\t%d\t%d\t%d\t%s\t%s\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\n", r.layer, r.expert, r.row, r.slot,
                    r.job_row, fn::experts::fmt_name(r.gu), fn::experts::fmt_name(r.dn), r.w, r.e_cg, r.e_cpu, r.e_gpu,
                    r.e_q1);
        }
        fclose(f);
    }
    std::vector<const mx7_shadow_pair *> all;
    for (const mx7_shadow_pair & r : sp) {
        all.push_back(&r);
    }
    printf("mx7-check: shadow: %zu CPU pairs compared (%s %s)\n", sp.size(), path.c_str(),
           f != nullptr ? "written" : "NOT written");
    group_line("all", all);
    char what[96];
    for (int32_t l = 0; l < 64; ++l) {
        std::vector<const mx7_shadow_pair *> g;
        for (const mx7_shadow_pair * r : all) {
            if (r->layer == l) {
                g.push_back(r);
            }
        }
        if (!g.empty()) {
            snprintf(what, sizeof(what), "layer %d (%s / %s)", l, fn::experts::fmt_name(g[0]->gu),
                     fn::experts::fmt_name(g[0]->dn));
            group_line(what, g);
        }
    }
    for (int32_t gu = 0; gu < 11; ++gu) {
        for (int32_t dn = 0; dn < 11; ++dn) {
            std::vector<const mx7_shadow_pair *> g;
            for (const mx7_shadow_pair * r : all) {
                if (int32_t(r->gu) == gu && int32_t(r->dn) == dn) {
                    g.push_back(r);
                }
            }
            if (!g.empty()) {
                snprintf(what, sizeof(what), "class %s / %s", fn::experts::fmt_name(fn::experts::fmt(gu)),
                         fn::experts::fmt_name(fn::experts::fmt(dn)));
                group_line(what, g);
            }
        }
    }
    for (int32_t j = 0; j < fn::experts::cpu::job_max_rows; ++j) {
        std::vector<const mx7_shadow_pair *> g;
        for (const mx7_shadow_pair * r : all) {
            if (r->job_row == j) {
                g.push_back(r);
            }
        }
        snprintf(what, sizeof(what), "job row %d", j);
        group_line(what, g);
    }
    for (int32_t k = 0; k < 10; ++k) {
        std::vector<const mx7_shadow_pair *> g;
        for (const mx7_shadow_pair * r : all) {
            if (r->slot == k) {
                g.push_back(r);
            }
        }
        snprintf(what, sizeof(what), "route slot %d", k);
        group_line(what, g);
    }
    std::vector<const mx7_shadow_pair *> w = all;
    std::sort(w.begin(), w.end(),
              [](const mx7_shadow_pair * x, const mx7_shadow_pair * y) { return x->e_cg > y->e_cg; });
    for (size_t i = 0; i < std::min<size_t>(10, w.size()); ++i) {
        const mx7_shadow_pair & r = *w[i];
        printf("mx7-check: shadow worst %zu: layer %d expert %d row %d slot %d job row %d (%s / %s) weight %.4g: "
               "CPU-GPU %.4g, CPU-ref %.4g, GPU-ref %.4g, Q8_K-ref %.4g\n",
               i + 1, r.layer, r.expert, r.row, r.slot, r.job_row, fn::experts::fmt_name(r.gu),
               fn::experts::fmt_name(r.dn), r.w, r.e_cg, r.e_cpu, r.e_gpu, r.e_q1);
    }
    bool wiring = !sp.empty();
    for (const mx7_shadow_pair & r : sp) {
        wiring = wiring && std::isfinite(r.e_cg) && r.e_cg < 0.5f && (r.e_cpu < 0 || r.e_cpu < 0.5f);
    }
    printf("mx7-check: shadow wiring (every pair's CPU-GPU and CPU-ref error below 0.5): %s\n",
           wiring ? "PASS" : "FAIL");
    return wiring;
}

} // namespace

bool mx7_check(engine & e, const mx7_check_args & a) {
    std::vector<int32_t> ids;
    if (!read_ids(a.ids, ids) || int32_t(ids.size()) < n_prefix + appends[0] || a.shares.empty() || !a.set_share) {
        fprintf(stderr, "mx7-check: %s needs %d IDs, and a share list\n", a.ids.c_str(), n_prefix + appends[0]);
        return false;
    }
    std::string err;
    int32_t np = 0;
    ready_state s0;
    if (!a.set_share(0.0, err) || e.prefill(0, ids.data(), n_prefix, 0, np) != run_status::ok || np != n_prefix ||
        e.save_ready(s0) != run_status::ok) {
        fprintf(stderr, "mx7-check: the %d-token prefix: %s %s\n", n_prefix, err.c_str(), e.error().c_str());
        return false;
    }
    printf("mx7-check: prefix %d tokens of %s kept (decode-ready state %zu + %zu bytes)\n", n_prefix, a.ids.c_str(),
           s0.dev[0].size(), s0.dev[1].size());
    bool all = true;
    for (const int32_t n : appends) {
        std::vector<run_out> runs;
        for (const std::string & sh : a.shares) {
            run_out o;
            if (!one_run(e, a, s0, ids, n, sh, o, err)) {
                fprintf(stderr, "mx7-check: append %d share %s: %s\n", n, sh.c_str(), err.c_str());
                return false;
            }
            printf("mx7-check: append %d share %s: state %zu bytes fnv %016llx; prompt-final logits fnv %016llx top-1 "
                   "%d; counters %s\n",
                   n, sh.c_str(), o.state.size(), (unsigned long long)fnv(o.state.data(), o.state.size()),
                   (unsigned long long)fnv(o.logits.data(), size_t(V) * 4), top1(o.logits.data()), o.counters.c_str());
            if (sh == "0" && runs.end() == std::find_if(runs.begin(), runs.end(),
                                                        [](const run_out & r) { return r.share == "0"; })) {
                const std::string path = a.out_dir + "/state-" + std::to_string(n) + "-share0.bin";
                FILE * f = fopen(path.c_str(), "wb");
                const bool wrote = f != nullptr && fwrite(o.state.data(), 1, o.state.size(), f) == o.state.size();
                const std::string lpath = a.out_dir + "/logits-" + std::to_string(n) + "-share0.bin";
                FILE * g = fopen(lpath.c_str(), "wb");
                const bool lw = g != nullptr && fwrite(o.logits.data(), 4, o.logits.size(), g) == o.logits.size();
                if (f != nullptr) {
                    fclose(f);
                }
                if (g != nullptr) {
                    fclose(g);
                }
                if (!wrote || !lw) {
                    fprintf(stderr, "mx7-check: cannot write %s or %s\n", path.c_str(), lpath.c_str());
                    return false;
                }
            }
            runs.push_back(std::move(o));
        }
        const auto ref = std::find_if(runs.begin(), runs.end(), [](const run_out & r) { return r.share == "0"; });
        for (size_t i = 0; i < runs.size(); ++i) {
            const run_out & r = runs[i];
            // against share 0 (KL bound) and, for a repeated share, against its first run (bitwise for 0 or a fixed x)
            const auto first = std::find_if(runs.begin(), runs.begin() + int64_t(i),
                                            [&](const run_out & q) { return q.share == r.share; });
            if (ref != runs.end() && &*ref != &r && r.share == "shadow") {
                // the shadow's output is the GPU's alone: share 0's state and prompt-final logits, bitwise
                const bool same =
                    r.state == ref->state && std::equal(r.logits.begin(), r.logits.begin() + V, ref->logits.begin());
                all = all && same;
                printf("mx7-check: append %d, shadow against share 0: state and prompt-final logits %s: %s\n", n,
                       same ? "bitwise equal" : "DIFFER", same ? "PASS" : "FAIL");
                all = shadow_report(a, r.shadow) && all;
            } else if (ref != runs.end() && &*ref != &r) {
                double sum = 0, mx = 0;
                int32_t agree = 0;
                std::vector<double> ks;
                for (int32_t q = 0; q <= n_ends; ++q) {
                    const float * p0 = ref->logits.data() + int64_t(q) * V;
                    const float * p1 = r.logits.data() + int64_t(q) * V;
                    const double k = kl(p0, p1);
                    ks.push_back(k);
                    sum += k;
                    mx = std::max(mx, k);
                    agree += top1(p0) == top1(p1) ? 1 : 0;
                    if (r.share != "0") {
                        printf("mx7-check: append %d, share %s, prompt end %d rows: KL %.6g, share-0 entropy %.4g "
                               "nats, top-1 %d (share 0) %d (share %s)\n",
                               n, r.share.c_str(), n - q, k, entropy(p0), top1(p0), top1(p1), r.share.c_str());
                    }
                }
                const double mean = sum / (n_ends + 1);
                const double k0 = kl(ref->logits.data(), r.logits.data());
                const double p50 = rank(ks, 0.5), p99 = rank(ks, 0.99);
                const bool same = r.state == ref->state && r.logits == ref->logits;
                const bool nc1 = p50 <= 0.0272 && p99 <= 5.25 && mx <= 19.18;
                const bool pass = r.share == "0" ? same : nc1;
                all = all && pass;
                printf("mx7-check: append %d, share %s against share 0: prompt-final KL %.6g, KL over %d prompt ends "
                       "(%d to %d rows) p50 %.6g p99 %.6g max %.6g mean %.6g, top-1 agreement %d of %d; state and "
                       "logits %s; NC1 (p50 <= 0.0272, p99 <= 5.25, max <= 19.18) %s, Strata's bound (mean <= 0.01, "
                       "max <= 0.05, reference only) %s: %s\n",
                       n, r.share.c_str(), k0, n_ends + 1, n - n_ends, n, p50, p99, mx, mean, agree, n_ends + 1,
                       same ? "bitwise equal" : "differ", nc1 ? "met" : "NOT met",
                       mean <= 0.01 && mx <= 0.05 ? "met" : "not met", pass ? "PASS" : "FAIL");
            }
            if (first != runs.begin() + int64_t(i) && r.share != "auto" && r.share != "shadow") {
                const bool same = r.state == first->state && r.logits == first->logits;
                all = all && same;
                printf("mx7-check: append %d, share %s repeated: state and logits %s: %s\n", n, r.share.c_str(),
                       same ? "bitwise equal" : "DIFFER", same ? "PASS" : "FAIL");
            }
        }
    }
    printf("mx7-check: %s\n", all ? "PASS" : "FAIL");
    return all;
}

} // namespace fn::runtime
