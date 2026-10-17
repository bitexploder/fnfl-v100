#include "fn/kernels/samp/samp_ref.h"

#include "fn/kernels/samp/philox.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace fn::samp {
namespace {

uint32_t f32_bits(float f) {
    uint32_t u = 0;
    memcpy(&u, &f, 4); // the F32 bit pattern, for the order key
    return u;
}

// the order key: U32 order equals F32 order, every NaN at 0, -0 equal to +0 (as the kernels')
uint32_t order_key(float f) {
    const uint32_t u = f32_bits(f);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
        return 0u;
    }
    if ((u & 0x7fffffffu) == 0u) {
        return 0x80000000u;
    }
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

uint64_t composite(float l, int32_t id) {
    return (static_cast<uint64_t>(order_key(l)) << 32) | (0xffffffffu - static_cast<uint32_t>(id));
}

bool finite_num(float f) {
    return f == f && f != -INFINITY && f != INFINITY;
}

uint64_t uni(const options & o, uint32_t pos, uint32_t purpose) {
    uint64_t u0 = 0, u1 = 0;
    uniforms(o.seed, o.request, pos, purpose, u0, u1);
    return u0;
}

// the first entry whose running weight exceeds target (weights in order); the last positive one when none does
ref_draw_out scan(const std::vector<int32_t> & ids, const std::vector<double> & w, double target, double total) {
    ref_draw_out r;
    r.target = total > 0 ? target / total : 0;
    double cum = 0;
    int32_t last = -1;
    double last_before = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!(w[i] > 0)) {
            continue;
        }
        const double before = cum;
        cum += w[i];
        last = static_cast<int32_t>(i);
        last_before = before;
        if (cum > target) {
            r.token = ids[i];
            r.before = before / total;
            r.after = cum / total;
            return r;
        }
    }
    if (last >= 0) {
        r.token = ids[size_t(last)];
        r.before = last_before / total;
        r.after = 1.0;
    }
    return r;
}

} // namespace

void ref_transform(const options & o, const float * logits, int32_t V, const ref_pen & pen, std::vector<float> & out) {
    // the bias table (duplicate entries summed), added once to every logit as the device does
    out.assign(logits, logits + V);
    if (!o.bias.empty()) {
        std::vector<float> table(static_cast<size_t>(V), 0.0f);
        for (const auto & b : o.bias) {
            table[size_t(b.first)] += b.second;
        }
        for (int32_t i = 0; i < V; ++i) {
            out[size_t(i)] = logits[i] + table[size_t(i)];
        }
    }
    if (pen.counts == nullptr || !o.penalties()) {
        return;
    }
    for (int32_t i = 0; i < V; ++i) {
        int32_t c = static_cast<int32_t>(pen.counts[i]);
        for (const auto & d : pen.deltas) {
            c += d.first == i ? d.second : 0;
        }
        if (c > 0) {
            float l = out[size_t(i)];
            l = l <= 0.0f ? l * o.penalty_repeat : l / o.penalty_repeat;
            const float cf = static_cast<float>(c) * o.penalty_freq;
            const float sub = cf + o.penalty_present;
            out[size_t(i)] = l - sub;
        }
    }
}

bool ref_distribution(const options & o, const float * l, int32_t V, int32_t split, ref_dist & d, std::string & err) {
    d = ref_dist();
    d.split = split;
    d.p = path_of(o);
    d.greedy = o.temp <= 0.0f;
    // every candidate by (logit, lower ID): NaN last; the order the pinned sort gives, with FNFL's tie rule
    std::vector<uint64_t> keys(static_cast<size_t>(V));
    for (int32_t i = 0; i < V; ++i) {
        keys[size_t(i)] = composite(l[i], i);
    }
    const auto id_of = [](uint64_t k) { return static_cast<int32_t>(0xffffffffu - static_cast<uint32_t>(k)); };
    const int32_t k = d.greedy ? 1 : o.top_k;
    std::vector<uint64_t> sorted;
    if (k > 0) {
        const int32_t kk = std::min(k, V);
        std::partial_sort(keys.begin(), keys.begin() + kk, keys.end(), std::greater<uint64_t>());
        sorted.assign(keys.begin(), keys.begin() + kk);
    } else {
        std::sort(keys.begin(), keys.end(), std::greater<uint64_t>());
        sorted = keys;
    }
    const float M = l[id_of(sorted[0])];
    if (!finite_num(M)) {
        err = "no candidate: the largest logit is not finite";
        return false;
    }
    // top-p on the candidates (T = 1, at their maximum): the first index whose running probability reaches top_p
    size_t n1 = sorted.size();
    if (!d.greedy && o.top_p < 1.0f) {
        std::vector<double> w(sorted.size());
        double z = 0;
        for (size_t i = 0; i < sorted.size(); ++i) {
            const float li = l[id_of(sorted[i])];
            w[i] = finite_num(li) ? std::exp(static_cast<double>(li) - static_cast<double>(M)) : 0.0;
            z += w[i];
        }
        const double tgt = static_cast<double>(o.top_p) * z;
        double cum = 0;
        for (size_t i = 0; i < sorted.size(); ++i) {
            const double before = cum;
            cum += w[i];
            if (cum >= tgt) {
                n1 = i + 1;
                d.cut_before = before / z;
                d.cut_at = cum / z;
                d.cut_target = static_cast<double>(o.top_p);
                break;
            }
        }
    }
    // min-p: logit >= M + logf(min_p), in F32 as the pinned code; the first always kept
    size_t n2 = sorted.size();
    if (!d.greedy && o.min_p > 0.0f) {
        const float thr = M + logf(o.min_p);
        for (size_t i = 1; i < sorted.size(); ++i) {
            if (!(l[id_of(sorted[i])] >= thr)) {
                n2 = i;
                break;
            }
        }
    }
    const size_t nk = std::min(n1, n2);
    // temperature: logit / T in F32, then exp(lt - Mt) in double
    const float T = d.greedy ? 1.0f : o.temp;
    const float Mt = M / T;
    std::vector<int32_t> ids;
    std::vector<double> w;
    for (size_t i = 0; i < nk; ++i) {
        const int32_t id = id_of(sorted[i]);
        const float li = l[id];
        if (!(li == li)) {
            continue;
        }
        ids.push_back(id);
        const float lt = li / T;
        w.push_back(finite_num(li) ? std::exp(static_cast<double>(lt) - static_cast<double>(Mt)) : 0.0);
    }
    if (d.greedy) {
        ids.resize(1);
        w.assign(1, 1.0);
    }
    if (d.p == path::full) {
        // the kept set in ascending ID order
        std::vector<std::pair<int32_t, double>> v;
        for (size_t i = 0; i < ids.size(); ++i) {
            v.emplace_back(ids[i], w[i]);
        }
        std::sort(v.begin(), v.end());
        ids.clear();
        w.clear();
        for (const auto & e : v) {
            ids.push_back(e.first);
            w.push_back(e.second);
        }
    }
    double z = 0;
    for (double x : w) {
        z += x;
    }
    if (!(z > 0)) {
        err = "no candidate: the kept set has no mass";
        return false;
    }
    d.id = ids;
    d.prob.resize(w.size());
    d.cum.resize(w.size());
    d.pid.assign(static_cast<size_t>(V), 0.0);
    double run = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        d.prob[i] = w[i] / z;
        d.mass[ids[i] < split ? 0 : 1] += d.prob[i];
        run += d.prob[i];
        d.cum[i] = run;
        d.pid[static_cast<size_t>(ids[i])] = d.prob[i];
        const int32_t c = ids[i] < split ? 0 : 1;
        d.cid[c].push_back(ids[i]);
        d.ccum[c].push_back((d.ccum[c].empty() ? 0.0 : d.ccum[c].back()) + d.prob[i]);
    }
    return true;
}

namespace {

// the first entry whose running sum exceeds target (a binary search over running sums), as scan
ref_draw_out scan_cum(const std::vector<int32_t> & ids, const std::vector<double> & cum, double target, double total) {
    ref_draw_out r;
    r.target = total > 0 ? target / total : 0;
    if (ids.empty()) {
        return r;
    }
    size_t i = static_cast<size_t>(std::upper_bound(cum.begin(), cum.end(), target) - cum.begin());
    if (i >= ids.size()) {
        i = ids.size() - 1;
        while (i > 0 && cum[i] == cum[i - 1]) {
            --i; // the last positive weight
        }
    }
    r.token = ids[i];
    r.before = (i == 0 ? 0.0 : cum[i - 1]) / total;
    r.after = cum[i] / total;
    return r;
}

} // namespace

double ref_prob(const ref_dist & d, int32_t id) {
    return id >= 0 && static_cast<size_t>(id) < d.pid.size() ? d.pid[static_cast<size_t>(id)] : 0.0;
}

ref_draw_out ref_draw(const ref_dist & d, uint64_t u_shard, uint64_t u_local) {
    if (d.p != path::full) {
        return scan_cum(d.id, d.cum, unit_double(u_shard), d.cum.empty() ? 1.0 : d.cum.back());
    }
    // the shard by its mass (u * Z < Z_0), then the ascending-ID scan inside it with the second uniform
    const double u = unit_double(u_shard);
    const int32_t card = u < d.mass[0] ? 0 : 1;
    ref_draw_out r = scan_cum(d.cid[card], d.ccum[card], unit_double(u_local) * d.mass[card], d.mass[card]);
    r.stage = 1;
    // a shard choice within rounding of the boundary is the stage that matters
    if (std::fabs(u - d.mass[0]) < 1e-5) {
        r.stage = 0;
        r.target = u;
        r.before = d.mass[0];
        r.after = d.mass[0];
    }
    return r;
}

double ref_q::at(int32_t token) const {
    if (shard[0] != nullptr) {
        const int32_t c = token < split ? 0 : 1;
        return shard[c][token - (c == 0 ? 0 : split)];
    }
    for (size_t i = 0; i < id.size(); ++i) {
        if (id[i] == token) {
            return q[i];
        }
    }
    return 0.0;
}

ref_draw_out ref_residual(const ref_dist & d, const ref_q & q, uint64_t u_shard, uint64_t u_local, bool & empty) {
    std::vector<double> r(d.id.size());
    double R[2] = {0, 0};
    for (size_t i = 0; i < d.id.size(); ++i) {
        r[i] = std::max(0.0, d.prob[i] - q.at(d.id[i]));
        R[d.id[i] < d.split ? 0 : 1] += r[i];
    }
    empty = !(R[0] + R[1] > 0);
    if (empty) {
        return ref_draw_out();
    }
    if (d.p != path::full) {
        return scan(d.id, r, unit_double(u_shard) * (R[0] + R[1]), R[0] + R[1]);
    }
    const double u = unit_double(u_shard);
    const int32_t card = u * (R[0] + R[1]) < R[0] ? 0 : 1;
    std::vector<int32_t> ids;
    std::vector<double> w;
    for (size_t i = 0; i < d.id.size(); ++i) {
        if ((d.id[i] < d.split ? 0 : 1) == card) {
            ids.push_back(d.id[i]);
            w.push_back(r[i]);
        }
    }
    ref_draw_out o = scan(ids, w, unit_double(u_local) * R[card], R[card]);
    o.stage = 1;
    return o;
}

ref_resid ref_make_resid(const ref_dist & d, const ref_q & q) {
    ref_resid r;
    double run = 0, run_c[2] = {0, 0};
    for (size_t i = 0; i < d.id.size(); ++i) {
        const double v = std::max(0.0, d.prob[i] - q.at(d.id[i]));
        const int32_t c = d.id[i] < d.split ? 0 : 1;
        r.R[c] += v;
        run += v;
        r.id.push_back(d.id[i]);
        r.cum.push_back(run);
        run_c[c] += v;
        r.cid[c].push_back(d.id[i]);
        r.ccum[c].push_back(run_c[c]);
    }
    r.empty = !(r.R[0] + r.R[1] > 0);
    return r;
}

ref_draw_out ref_residual_draw(const ref_dist & d, const ref_resid & r, uint64_t u_shard, uint64_t u_local) {
    if (d.p != path::full) {
        const double tot = r.R[0] + r.R[1];
        return scan_cum(r.id, r.cum, unit_double(u_shard) * tot, tot);
    }
    const double u = unit_double(u_shard);
    const int32_t card = u * (r.R[0] + r.R[1]) < r.R[0] ? 0 : 1;
    ref_draw_out o = scan_cum(r.cid[card], r.ccum[card], unit_double(u_local) * r.R[card], r.R[card]);
    o.stage = 1;
    return o;
}

ref_verify_out ref_verify(const options & o, const std::vector<ref_dist> & p, const std::vector<int32_t> & x,
                          const std::vector<ref_q> & q, uint32_t pos0, const std::vector<ref_resid> * rc) {
    ref_verify_out r;
    const int32_t nd = static_cast<int32_t>(x.size());
    for (int32_t t = 0; t < nd; ++t) {
        const uint32_t pos = pos0 + static_cast<uint32_t>(t);
        if (p[size_t(t)].greedy) {
            const int32_t am = p[size_t(t)].id[0];
            if (x[size_t(t)] != am) {
                r.a = t;
                r.out.push_back(am);
                return r;
            }
            r.out.push_back(x[size_t(t)]);
            continue;
        }
        const double px = ref_prob(p[size_t(t)], x[size_t(t)]);
        const double qx = q[size_t(t)].at(x[size_t(t)]);
        if (unit_double(uni(o, pos, p_accept)) * qx < px) {
            r.out.push_back(x[size_t(t)]);
            continue;
        }
        bool empty = false;
        ref_draw_out rd;
        if (rc != nullptr) {
            empty = (*rc)[size_t(t)].empty;
            if (!empty) {
                rd = ref_residual_draw(p[size_t(t)], (*rc)[size_t(t)], uni(o, pos, p_resid_shard),
                                       uni(o, pos, p_resid_local));
            }
        } else {
            rd =
                ref_residual(p[size_t(t)], q[size_t(t)], uni(o, pos, p_resid_shard), uni(o, pos, p_resid_local), empty);
        }
        if (empty) {
            // p itself: the top-k path draws it with the residual's uniform; the full path takes the row's plain draw
            rd = p[size_t(t)].p == path::full ? ref_draw(p[size_t(t)], uni(o, pos, p_shard), uni(o, pos, p_local))
                                              : ref_draw(p[size_t(t)], uni(o, pos, p_resid_shard), 0);
        }
        r.a = t;
        r.out.push_back(rd.token);
        return r;
    }
    r.a = nd;
    const uint32_t pos = pos0 + static_cast<uint32_t>(nd);
    r.out.push_back(ref_draw(p[size_t(nd)], uni(o, pos, p_shard), uni(o, pos, p_local)).token);
    return r;
}

void ref_window(int32_t cap, const std::vector<int32_t> & ledger, std::vector<uint32_t> & counts,
                std::vector<int32_t> & ring) {
    counts.assign(n_vocab, 0u);
    ring.assign(static_cast<size_t>(std::max(cap, 1)), 0);
    if (cap <= 0) {
        return;
    }
    const int64_t L = static_cast<int64_t>(ledger.size());
    for (int64_t pos = std::max<int64_t>(0, L - cap); pos < L; ++pos) {
        counts[size_t(ledger[size_t(pos)])] += 1u;
        ring[size_t(pos % cap)] = ledger[size_t(pos)];
    }
}

double chi2_q(double x, int32_t k) {
    // Q(a, x/2) with a = k/2: the series for x < a + 1, else the continued fraction (Numerical Recipes, gammq)
    const double a = 0.5 * k, z = 0.5 * x;
    if (z <= 0) {
        return 1.0;
    }
    const double gln = std::lgamma(a);
    if (z < a + 1) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int32_t n = 0; n < 10000; ++n) {
            ap += 1;
            del *= z / ap;
            sum += del;
            if (std::fabs(del) < std::fabs(sum) * 1e-15) {
                break;
            }
        }
        return 1.0 - sum * std::exp(-z + a * std::log(z) - gln);
    }
    double b = z + 1 - a, c = 1.0 / 1e-300, dd = 1.0 / b, h = dd;
    for (int32_t i = 1; i < 10000; ++i) {
        const double an = -i * (i - a);
        b += 2;
        dd = an * dd + b;
        if (std::fabs(dd) < 1e-300) {
            dd = 1e-300;
        }
        c = b + an / c;
        if (std::fabs(c) < 1e-300) {
            c = 1e-300;
        }
        dd = 1.0 / dd;
        const double del = dd * c;
        h *= del;
        if (std::fabs(del - 1) < 1e-15) {
            break;
        }
    }
    return std::exp(-z + a * std::log(z) - gln) * h;
}

// ---- fnfl-drive sampler-ref ----

namespace {

int g_fail = 0;

void expect(bool ok, const char * what) {
    printf("sampler-ref: %-72s %s\n", what, ok ? "PASS" : "FAIL");
    g_fail += ok ? 0 : 1;
}

bool near(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

// a small vocabulary of V logits: the natural logs of the given probabilities, then -20
std::vector<float> logs_of(const std::vector<double> & p, int32_t V) {
    std::vector<float> l(static_cast<size_t>(V), -20.0f);
    for (size_t i = 0; i < p.size(); ++i) {
        l[i] = static_cast<float>(std::log(p[i]));
    }
    return l;
}

// a chi-squared test of counts against probabilities (bins under 5 expected pooled); its p-value
double chi2_p(const std::vector<int32_t> & ids, const std::vector<double> & prob, const std::vector<int32_t> & draws) {
    std::vector<double> obs(ids.size() + 1, 0.0);
    for (int32_t t : draws) {
        size_t i = 0;
        while (i < ids.size() && ids[i] != t) {
            ++i;
        }
        obs[i] += 1; // ids.size(): outside the support
    }
    const double n = static_cast<double>(draws.size());
    double chi = 0, pool_o = obs[ids.size()], pool_e = 0;
    int32_t bins = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        const double e = n * prob[i];
        if (e >= 5) {
            chi += (obs[i] - e) * (obs[i] - e) / e;
            ++bins;
        } else {
            pool_o += obs[i];
            pool_e += e;
        }
    }
    if (pool_e >= 5) {
        chi += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
        ++bins;
    } else if (pool_o > 0 && pool_e < 1e-9) {
        return 0.0; // draws outside the support
    }
    return bins > 1 ? chi2_q(chi, bins - 1) : 1.0;
}

} // namespace

int fnfl_sampler_ref() {
    g_fail = 0;
    constexpr int32_t V = 16, split = 8;
    std::string err;
    // 1. top-k with ties: three equal logits after the first, k = 3 keeps the two lower IDs
    {
        std::vector<float> l(V, -5.0f);
        l[9] = 2.0f;
        l[3] = 1.0f;
        l[1] = 1.0f;
        l[12] = 1.0f;
        options o;
        o.temp = 1.0f;
        o.top_k = 3;
        o.top_p = 1.0f;
        o.min_p = 0.0f;
        ref_dist d;
        const bool ok = ref_distribution(o, l.data(), V, split, d, err);
        expect(ok && d.id == std::vector<int32_t>({9, 1, 3}), "top-k 3 with a three-way tie keeps the lower IDs 1, 3");
        const double e1 = std::exp(1.0 - 2.0);
        expect(ok && near(d.prob[0], 1 / (1 + 2 * e1), 1e-12) && near(d.prob[1], e1 / (1 + 2 * e1), 1e-12),
               "top-k list probabilities (T = 1)");
    }
    // 2. top-p: probabilities 0.5, 0.3, 0.15, 0.05: p = 0.75 keeps two (0.5 + 0.3 >= 0.75), renormalized
    {
        const std::vector<float> l = logs_of({0.5, 0.3, 0.15, 0.05}, V);
        options o;
        o.temp = 1.0f;
        o.top_k = 0;
        o.top_p = 0.75f;
        o.min_p = 0.0f;
        ref_dist d;
        const bool ok = ref_distribution(o, l.data(), V, split, d, err);
        expect(ok && d.id == std::vector<int32_t>({0, 1}) && near(d.prob[0], 0.625, 1e-6),
               "full-path top-p 0.75: the cut includes the token that reaches p");
        o.top_k = 20;
        ref_dist d2;
        const bool ok2 = ref_distribution(o, l.data(), V, split, d2, err);
        expect(ok2 && d2.id == std::vector<int32_t>({0, 1}) && near(d2.prob[1], 0.375, 1e-6),
               "top-k path top-p 0.75 on the list, the same kept set");
        o.top_p = 0.0f;
        ref_dist d3;
        expect(ref_distribution(o, l.data(), V, split, d3, err) && d3.id.size() == 1 && d3.id[0] == 0,
               "top-p 0 keeps the first token only");
    }
    // 3. min-p 0.25: logit >= M + log 0.25 keeps 0.5, 0.3, 0.15 (0.125 is the bar)
    {
        const std::vector<float> l = logs_of({0.5, 0.3, 0.15, 0.05}, V);
        options o;
        o.temp = 1.0f;
        o.top_k = 0;
        o.top_p = 1.0f;
        o.min_p = 0.25f;
        ref_dist d;
        const bool ok = ref_distribution(o, l.data(), V, split, d, err);
        expect(ok && d.id.size() == 3 && near(d.prob[2], 0.15 / 0.95, 1e-6), "min-p 0.25 drops the 0.05 token");
    }
    // 4. temperature 2: probabilities proportional to p^(1/2), after the filters
    {
        const std::vector<float> l = logs_of({0.64, 0.36}, V);
        options o;
        o.temp = 2.0f;
        o.top_k = 2;
        o.top_p = 1.0f;
        o.min_p = 0.0f;
        ref_dist d;
        const bool ok = ref_distribution(o, l.data(), V, split, d, err);
        expect(ok && near(d.prob[0], 0.8 / 1.4, 1e-6), "temperature 2 on (0.64, 0.36): 0.8 / 1.4");
    }
    // 5. penalties as llama_sampler_penalties_apply: repeat on a positive and a negative logit, freq and present
    {
        std::vector<float> l(V, 0.0f);
        l[2] = 4.0f;
        l[5] = -2.0f;
        std::vector<uint32_t> counts(V, 0u);
        counts[2] = 2;
        counts[5] = 1;
        options o;
        o.penalty_repeat = 2.0f;
        o.penalty_freq = 0.5f;
        o.penalty_present = 0.25f;
        ref_pen pen;
        pen.counts = counts.data();
        pen.deltas = {{5, 1}};
        std::vector<float> out;
        ref_transform(o, l.data(), V, pen, out);
        expect(out[2] == 4.0f / 2.0f - (2 * 0.5f + 0.25f) && out[5] == -2.0f * 2.0f - (2 * 0.5f + 0.25f) &&
                   out[0] == 0.0f,
               "penalties: l / r or l * r, then - (c * freq + present); a delta adds to the count");
        o.bias = {{7, -INFINITY}, {0, 1.5f}};
        ref_transform(o, l.data(), V, ref_pen(), out);
        expect(out[7] == -INFINITY && out[0] == 1.5f, "logit bias, -inf bans");
    }
    // 6. the two-stage draw (full path) reproduces the distribution: chi-squared over 200,000 uniforms
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.5f);
        std::vector<float> l(V);
        for (float & v : l) {
            v = nd(rng);
        }
        options o;
        o.temp = 0.8f;
        o.top_k = 0;
        o.top_p = 0.9f;
        o.min_p = 0.02f;
        ref_dist d;
        bool ok = ref_distribution(o, l.data(), V, split, d, err);
        std::vector<int32_t> draws;
        for (uint32_t i = 0; i < 200000 && ok; ++i) {
            draws.push_back(ref_draw(d, uni(o, i, p_shard), uni(o, i, p_local)).token);
        }
        const double pv = ok ? chi2_p(d.id, d.prob, draws) : 0.0;
        printf("sampler-ref: two-stage draw chi2 p = %.4f\n", pv);
        expect(ok && pv > 0.001, "the shard-then-local draw is exact in distribution (chi2 p > 0.001)");
    }
    // 7. the accept step: the first emitted token follows p for q = p, a different q, and q = 1.5 p (the residual
    // empty, so p itself), on both paths; chi-squared over 200,000 rounds of one draft
    for (int32_t pathk = 0; pathk < 2; ++pathk) {
        for (int32_t qk = 0; qk < 3; ++qk) {
            std::mt19937 rng(11 + 3 * pathk + qk);
            std::normal_distribution<float> nd(0.0f, 1.5f);
            std::vector<float> lp(V), lq(V);
            for (int32_t i = 0; i < V; ++i) {
                lp[size_t(i)] = nd(rng);
                lq[size_t(i)] = nd(rng);
            }
            options o;
            o.temp = 1.0f;
            o.top_k = pathk == 0 ? 6 : 0;
            o.top_p = 0.95f;
            o.min_p = 0.0f;
            ref_dist p, qd;
            bool ok = ref_distribution(o, lp.data(), V, split, p, err) &&
                      ref_distribution(o, (qk == 1 ? lq : lp).data(), V, split, qd, err);
            ref_q q;
            std::vector<float> qs[2] = {std::vector<float>(split, 0.0f), std::vector<float>(V - split, 0.0f)};
            const double scale = qk == 2 ? 1.5 : 1.0;
            for (size_t i = 0; i < qd.id.size(); ++i) {
                q.id.push_back(qd.id[i]);
                q.q.push_back(static_cast<float>(qd.prob[i] * scale));
                const int32_t c = qd.id[i] < split ? 0 : 1;
                qs[c][size_t(qd.id[i] - (c == 0 ? 0 : split))] = static_cast<float>(qd.prob[i] * scale);
            }
            if (pathk == 1) {
                q.shard[0] = qs[0].data();
                q.shard[1] = qs[1].data();
                q.split = split;
            }
            std::vector<int32_t> first;
            std::uniform_real_distribution<double> ud(0.0, 1.0);
            int32_t rejected = 0;
            for (uint32_t i = 0; i < 200000 && ok; ++i) {
                // the draft x ~ q (normalized), then the verify step of one draft and the bonus row (p again)
                const double u = ud(rng);
                double cum = 0;
                int32_t x = qd.id.back();
                for (size_t j = 0; j < qd.id.size(); ++j) {
                    cum += qd.prob[j];
                    if (cum > u) {
                        x = qd.id[j];
                        break;
                    }
                }
                const ref_verify_out r = ref_verify(o, {p, p}, {x}, {q}, 2 * i);
                first.push_back(r.out[0]);
                rejected += r.a == 0 ? 1 : 0;
            }
            const double pv = ok ? chi2_p(p.id, p.prob, first) : 0.0;
            char what[160];
            snprintf(what, sizeof(what), "accept step, %s path, q %s: out[0] ~ p (p = %.4f, %d rejections)",
                     pathk == 0 ? "top-k" : "full",
                     qk == 0   ? "= p"
                     : qk == 1 ? "different"
                               : "= 1.5 p (empty)",
                     pv, rejected);
            expect(ok && pv > 0.001 && (qk != 0 || rejected == 0) && (qk == 0 || rejected > 0), what);
        }
    }
    // 8. the residual: empty when q >= p everywhere; the fallback then draws from p
    {
        const std::vector<float> l = logs_of({0.5, 0.3, 0.2}, V);
        options o;
        o.temp = 1.0f;
        o.top_k = 3;
        o.top_p = 1.0f;
        o.min_p = 0.0f;
        ref_dist p;
        const bool ok = ref_distribution(o, l.data(), V, split, p, err);
        ref_q q;
        for (size_t i = 0; i < p.id.size(); ++i) {
            q.id.push_back(p.id[i]);
            q.q.push_back(static_cast<float>(p.prob[i] * 1.5));
        }
        bool empty = false;
        ref_residual(p, q, 1, 2, empty);
        expect(ok && empty, "the residual of q = 1.5 p has no mass");
    }
    // 9. the window: counts of the last cap tokens and the ring by position
    {
        std::vector<uint32_t> c;
        std::vector<int32_t> ring;
        ref_window(3, {4, 5, 4, 6, 4}, c, ring);
        expect(c[4] == 2 && c[6] == 1 && c[5] == 0 && ring[0] == 6 && ring[1] == 4 && ring[2] == 4,
               "the window of the last 3 tokens (ring[pos % 3])");
    }
    // 10. chi2_q against known values
    expect(near(chi2_q(3.841458820694124, 1), 0.05, 1e-9) && near(chi2_q(18.307038053275146, 10), 0.05, 1e-9),
           "chi2 upper tail at the 5% points of 1 and 10 degrees of freedom");
    printf("sampler-ref: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}

} // namespace fn::samp
