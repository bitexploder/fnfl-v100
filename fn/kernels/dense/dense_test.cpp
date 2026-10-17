#include "fn/kernels/dense/dense_test.h"

#include "fn/kernels/experts/grids.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/round/version.h"
#include "fn/model/layer.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/quant/synth.h"
#include "fn/tools/test_dump.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fn::dense {
namespace {

// the synthetic sets' case indices: "DP" in the high half, so no other test's seeded blocks repeat here
constexpr uint64_t dp_seed_index = 0x44500000u;

quant::wfmt to_wfmt(fmt f) {
    switch (f) {
        case fmt::q4_k:
            return quant::wfmt::q4_k;
        case fmt::q5_k:
            return quant::wfmt::q5_k;
        case fmt::q6_k:
            return quant::wfmt::q6_k;
        case fmt::iq4_xs:
            return quant::wfmt::iq4_xs;
        case fmt::iq3_s:
            return quant::wfmt::iq3_s;
        case fmt::q8_0:
            return quant::wfmt::q8_0;
        case fmt::iq4_nl:
            return quant::wfmt::iq4_nl;
        case fmt::q2_0:
            return quant::wfmt::q2_0;
        case fmt::bf16:
            return quant::wfmt::bf16;
        case fmt::f32:
            return quant::wfmt::f32;
    }
    return quant::wfmt::f32;
}

// runs fn(i) for i in [0, n) on up to 12 threads
template <typename F> void parallel_for(int32_t n, const F & fn) {
    const int32_t nt = std::min<int32_t>(n, 12);
    std::atomic<int32_t> next{0};
    std::vector<std::thread> th;
    th.reserve(static_cast<size_t>(nt));
    for (int32_t t = 0; t < nt; ++t) {
        th.emplace_back([&]() {
            for (int32_t i = next++; i < n; i = next++) {
                fn(i);
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
}

// a 64-bit LCG; uniform in [0, 1)
struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    double next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

// FNV-1a of a case name: the activations' seed, the same on every host
uint64_t name_seed(const std::string & s) {
    uint64_t h = 1469598103934665603ull;
    for (const char c : s) {
        h = (h ^ static_cast<uint8_t>(c)) * 1099511628211ull;
    }
    return h;
}

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "dp-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
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
    bool get(void * h, size_t bytes) const {
        return p != nullptr && bytes <= n && cuda_ok(cudaMemcpy(h, p, bytes, cudaMemcpyDeviceToHost), "get");
    }
    // 0xff bytes (a NaN in every F32 slot), so an output the kernel misses cannot pass
    bool poison() const { return p != nullptr && cuda_ok(cudaMemset(p, 0xff, n), "memset"); }
};

// one matrix of a test case
struct mspec {
    const char * name;
    fmt f;
    int32_t m;
    epi e;
    int32_t pair;
    int32_t split;  // 0: choose_splits
    int32_t period; // synthetic rows before they repeat
    bool scaled;    // the shared down's per-row scale
};

mspec plain(const char * name, fmt f, int32_t m, int32_t split = 0) {
    return {name, f, m, epi::f32, -1, split, n_synth_rows, false};
}

mspec scaled(const char * name, fmt f, int32_t m) {
    return {name, f, m, epi::f32, -1, 0, n_synth_rows, true};
}

mspec glu(const char * name, fmt f, epi e, int32_t pair, int32_t split = 0) {
    return {name, f, 320, e, pair, split, n_synth_rows, false};
}

// the GDN input's qkv with the convolution epilogue (LP1): 5,120 channels, unsplit
mspec conv(const char * name, fmt f) {
    return {name, f, fn::gdn::n_channel, epi::conv_silu, -1, 1, n_synth_rows, false};
}

struct tcase {
    std::string name;
    int32_t k;
    std::vector<mspec> mats;
    bool zero = false;       // all-zero activations
    bool route = false;      // mats[0] the router's logits, mats[1] the shared gate, with the epilogue
    double xscale = 1.0;     // the activations' scale: 0.02 gives the router realistic logits (|l| ~ 1)
    int32_t nan_expert = -1; // a router row with a NaN weight: that expert's logit is NaN in every row
    bool conv = false;       // mats[0] is the convolution epilogue's matrix (epi conv_silu)
};

// the synthetic sets and their decoded rows, made once
struct synth_cache {
    std::map<int64_t, synth> sets;
    std::map<int64_t, std::vector<float>> deq; // [n_synth_rows][k]

    static int64_t key(fmt f, int32_t k) { return int64_t(f) * 65536 + k; }

    const synth * get(fmt f, int32_t k) {
        const int64_t kk = key(f, k);
        auto it = sets.find(kk);
        if (it != sets.end()) {
            return &it->second;
        }
        synth s;
        if (!make_synth(f, k, s)) {
            fprintf(stderr, "dp-test: no synthetic %s at K %d\n", fmt_name(f), k);
            return nullptr;
        }
        std::vector<float> d(size_t(n_synth_rows) * k);
        std::atomic<bool> ok{true};
        parallel_for(n_synth_rows, [&](int32_t i) {
            if (!quant::decode_row(to_wfmt(f), s.row(i), d.data() + size_t(i) * k, k)) {
                ok = false;
            }
        });
        if (!ok) {
            return nullptr;
        }
        deq[kk] = std::move(d);
        return &(sets[kk] = std::move(s));
    }
};

// what one launch wrote, rows [0, rows)
struct outputs {
    int32_t rows = 0;
    std::vector<std::vector<float>> out; // per matrix [rows][m] (epi f32)
    std::vector<int8_t> glu_codes;       // [rows][glu_m]
    std::vector<float> glu_scales;       // [rows][glu_m / 32][2]
    std::vector<float> glu_f32;          // [rows][glu_m]
    std::vector<int32_t> ids;            // [rows][10]
    std::vector<float> wts, sg;          // [rows][10], [rows]
    std::vector<float> hist;             // the convolution's history versions 1..rows [rows][5120][3]
};

// row ra of x against row rb of y, bitwise: every output the launch wrote
bool same_row(const outputs & x, int32_t ra, const outputs & y, int32_t rb) {
    bool same = true;
    const auto cmp = [&](const void * a, const void * b, size_t n) { same = same && memcmp(a, b, n) == 0; };
    for (size_t i = 0; i < x.out.size(); ++i) {
        const size_t m = x.out[i].size() / size_t(x.rows);
        cmp(x.out[i].data() + size_t(ra) * m, y.out[i].data() + size_t(rb) * m, m * 4);
    }
    if (!x.glu_codes.empty()) {
        const size_t m = x.glu_codes.size() / size_t(x.rows);
        cmp(x.glu_codes.data() + size_t(ra) * m, y.glu_codes.data() + size_t(rb) * m, m);
        cmp(x.glu_scales.data() + size_t(ra) * m / 16, y.glu_scales.data() + size_t(rb) * m / 16, m / 16 * 4);
        cmp(x.glu_f32.data() + size_t(ra) * m, y.glu_f32.data() + size_t(rb) * m, m * 4);
    }
    if (!x.hist.empty()) {
        const size_t v = size_t(fn::gdn::hist_version_floats);
        cmp(x.hist.data() + size_t(ra) * v, y.hist.data() + size_t(rb) * v, v * 4);
    }
    if (!x.ids.empty()) {
        cmp(x.ids.data() + size_t(ra) * route_used, y.ids.data() + size_t(rb) * route_used, size_t(route_used) * 4);
        cmp(x.wts.data() + size_t(ra) * route_used, y.wts.data() + size_t(rb) * route_used, size_t(route_used) * 4);
        cmp(x.sg.data() + ra, y.sg.data() + rb, 4);
    }
    return same;
}

bool same_all(const outputs & x, const outputs & y) {
    bool same = x.rows == y.rows;
    for (int32_t b = 0; same && b < x.rows; ++b) {
        same = same_row(x, b, y, b);
    }
    return same;
}

// a case's device buffers, sized for max_rows, and its product_args
struct case_run {
    const tcase & tc;
    std::vector<std::unique_ptr<dev_mem>> w, out;
    std::unique_ptr<dev_mem> xc, xs, xf, grid, gc, gs, gf, ids, wts, sg, scale, work, taps, hist, cur;
    product_args a;
    // SR1, conv cases: each launch starts at the next committed slot (0, 1, .. 4, 0, ..), with history version 0
    // there and every other slot poisoned; the versions are read back from their slots
    const std::vector<float> * hist0 = nullptr;
    uint32_t cur_h = 0;
    int32_t conv_launches = 0;
    uint32_t slots_seen = 0;
    int32_t glu_m = 0;
    std::vector<int32_t> off;                               // each matrix's first synthetic row
    std::function<bool(int32_t, const product_args &)> via; // the launch in place of product (frame sites), or empty

    explicit case_run(const tcase & t) : tc(t) {}

    // one launch of rows rows whose activations (and scale) start at row row0; outputs land at row 0
    bool launch(int32_t rows, int32_t row0, outputs & o) {
        product_args l = a;
        const int32_t k = tc.k;
        l.x_codes = a.x_codes != nullptr ? a.x_codes + size_t(row0) * k : nullptr;
        l.x_scales = a.x_scales != nullptr ? a.x_scales + size_t(row0) * (k / 32) * 2 : nullptr;
        l.x_f32 = a.x_f32 != nullptr ? a.x_f32 + size_t(row0) * k : nullptr;
        for (int32_t i = 0; i < a.n_mats; ++i) {
            l.mats[i].scale = a.mats[i].scale != nullptr ? a.mats[i].scale + row0 : nullptr;
        }
        bool ok = true;
        for (const auto & d : out) {
            ok = ok && d->poison();
        }
        ok = ok && gc->poison() && gs->poison() && gf->poison() && ids->poison() && wts->poison() && sg->poison();
        if (tc.conv) {
            cur_h = static_cast<uint32_t>(conv_launches++ % fn::gdn::n_version);
            slots_seen |= 1u << cur_h;
            ok =
                ok && cur->put(&cur_h, 4) && hist->poison() &&
                cuda_ok(cudaMemcpy(hist->as<float>() + fn::round::version_slot(cur_h, 0) * fn::gdn::hist_version_floats,
                                   hist0->data(), hist0->size() * 4, cudaMemcpyHostToDevice),
                        "hist0");
        }
        if (!ok || !(via ? via(rows, l) : product(rows, l, nullptr)) || !cuda_ok(cudaDeviceSynchronize(), "product")) {
            fprintf(stderr, "dp-test: %s B%d: launch failed\n", tc.name.c_str(), rows);
            return false;
        }
        o.rows = rows;
        o.out.assign(size_t(a.n_mats), {});
        for (int32_t i = 0; i < a.n_mats; ++i) {
            if (tc.mats[size_t(i)].e == epi::f32 || tc.mats[size_t(i)].e == epi::conv_silu) {
                o.out[size_t(i)].resize(size_t(rows) * tc.mats[size_t(i)].m);
                ok = ok && out[size_t(i)]->get(o.out[size_t(i)].data(), o.out[size_t(i)].size() * 4);
            }
        }
        if (glu_m > 0) {
            o.glu_codes.resize(size_t(rows) * glu_m);
            o.glu_scales.resize(size_t(rows) * glu_m / 16);
            o.glu_f32.resize(size_t(rows) * glu_m);
            ok = ok && gc->get(o.glu_codes.data(), o.glu_codes.size()) &&
                 gs->get(o.glu_scales.data(), o.glu_scales.size() * 4) &&
                 gf->get(o.glu_f32.data(), o.glu_f32.size() * 4);
        }
        if (tc.conv) {
            const size_t v = size_t(fn::gdn::hist_version_floats);
            o.hist.resize(size_t(rows) * v);
            for (int32_t ver = 1; ok && ver <= rows; ++ver) {
                ok = cuda_ok(cudaMemcpy(o.hist.data() + size_t(ver - 1) * v,
                                        hist->as<float>() + fn::round::version_slot(cur_h, ver) * int64_t(v), v * 4,
                                        cudaMemcpyDeviceToHost),
                             "hist");
            }
        }
        if (tc.route) {
            o.ids.resize(size_t(rows) * route_used);
            o.wts.resize(size_t(rows) * route_used);
            o.sg.resize(size_t(rows));
            ok = ok && ids->get(o.ids.data(), o.ids.size() * 4) && wts->get(o.wts.data(), o.wts.size() * 4) &&
                 sg->get(o.sg.data(), o.sg.size() * 4);
        }
        const std::string t = "dp-" + tc.name + "-B" + std::to_string(rows) + "-r" + std::to_string(row0);
        for (size_t i = 0; ok && i < o.out.size(); ++i) {
            ok = fn::tools::test_dump(t + "-out" + std::to_string(i), o.out[i].data(), o.out[i].size() * 4);
        }
        ok = ok && fn::tools::test_dump(t + "-glu-codes", o.glu_codes.data(), o.glu_codes.size()) &&
             fn::tools::test_dump(t + "-glu-scales", o.glu_scales.data(), o.glu_scales.size() * 4) &&
             fn::tools::test_dump(t + "-glu-f32", o.glu_f32.data(), o.glu_f32.size() * 4) &&
             fn::tools::test_dump(t + "-hist", o.hist.data(), o.hist.size() * 4) &&
             fn::tools::test_dump(t + "-ids", o.ids.data(), o.ids.size() * 4) &&
             fn::tools::test_dump(t + "-wts", o.wts.data(), o.wts.size() * 4) &&
             fn::tools::test_dump(t + "-sg", o.sg.data(), o.sg.size() * 4);
        return ok;
    }
};

// the reference of one case: FP64 dots of every matrix's rows for all max_rows activation rows
struct reference {
    std::vector<std::vector<double>> dot; // per matrix [max_rows][m]
    std::vector<float> x;                 // F32 activations [max_rows][k]
    std::vector<int8_t> codes;            // their decode format
    std::vector<float> scales;
    std::vector<float> scale; // the shared down's per-row scale [max_rows]
    std::vector<float> taps;  // the convolution's taps [5120][4] and history version 0 [5120][3] (conv cases)
    std::vector<float> hist0;
};

// rows' max |err| / max |ref| over m outputs, worst row; ok when every row is within 1e-5 (an all-zero reference
// needs an exact zero)
struct bound {
    double worst = 0.0, err = 0.0, ref = 0.0;
    bool ok = true;
    void row(const float * got, const double * want, int32_t m) {
        double emax = 0.0, rmax = 0.0;
        for (int32_t r = 0; r < m; ++r) {
            const double g = got[r], w = want[r];
            if (std::isnan(w)) { // a NaN the case planted: the output must be NaN too
                emax = std::isnan(g) ? emax : INFINITY;
                continue;
            }
            emax = std::isfinite(g) ? std::max(emax, std::fabs(g - w)) : INFINITY;
            rmax = std::max(rmax, std::fabs(w));
        }
        const double rel = rmax > 0.0 ? emax / rmax : emax;
        ok = ok && (rmax > 0.0 ? rel <= 1e-5 : emax == 0.0);
        if (rel >= worst) {
            worst = rel;
            err = emax;
            ref = rmax;
        }
    }
    void print(const std::string & what) const {
        printf("  %-34s max|err| %.3e  max|ref| %.3e  rel %.3e (worst row)  %s\n", what.c_str(), err, ref, worst,
               ok ? "ok" : "FAIL");
    }
};

// The router's check for one row (DP2 Amendment 1): the FP64 softmax and top 10 of the reference logits, the sets
// compared (a difference explained when every expert in only one set has an FP64 logit within 2x the row's own logit
// error of the reference's 10th), the kernel's order (weights non-increasing, equal logits by ascending ID), and in a
// matching row the weights within 1e-5 relative; the gate within 1e-5 relative in every row.
struct route_check {
    int32_t rows_ok = 0, explained = 0, failed = 0;
    double w_worst = 0.0, g_worst = 0.0, own_worst = 0.0;

    void row(int32_t b, const float * logit, const double * ref, const int32_t * ids, const float * wts, float sg,
             double sg_ref) {
        double own = 0.0, mx = -HUGE_VAL;
        for (int32_t e = 0; e < route_experts; ++e) {
            own = std::max(own, std::isfinite(logit[e]) ? std::fabs(double(logit[e]) - ref[e]) : INFINITY);
            mx = std::max(mx, ref[e]);
        }
        own_worst = std::max(own_worst, own);
        std::vector<double> p(route_experts);
        double sum = 0.0;
        for (int32_t e = 0; e < route_experts; ++e) {
            p[size_t(e)] = std::exp(ref[e] - mx);
            sum += p[size_t(e)];
        }
        std::vector<int32_t> order(route_experts);
        for (int32_t e = 0; e < route_experts; ++e) {
            order[size_t(e)] = e;
            p[size_t(e)] /= sum;
        }
        std::stable_sort(order.begin(), order.end(), [&](int32_t x, int32_t y) { return p[size_t(x)] > p[size_t(y)]; });
        double psum = 0.0;
        for (int32_t j = 0; j < route_used; ++j) {
            psum += p[size_t(order[size_t(j)])];
        }
        const double inv = 1.0 / std::max(psum, 6.103515625e-5);
        bool order_ok = true, ids_ok = true;
        for (int32_t j = 0; j < route_used; ++j) {
            ids_ok = ids_ok && ids[j] >= 0 && ids[j] < route_experts;
            for (int32_t i = 0; i < j; ++i) {
                ids_ok = ids_ok && ids[i] != ids[j];
            }
        }
        if (!ids_ok) {
            printf("  route b%d: IDs out of range or repeated: FAIL\n", b);
            ++failed;
            return;
        }
        for (int32_t j = 0; j + 1 < route_used; ++j) {
            const bool tie = logit[ids[j]] == logit[ids[j + 1]];
            order_ok = order_ok && wts[j] >= wts[j + 1] && (!tie || ids[j] < ids[j + 1]);
        }
        std::vector<int32_t> want(order.begin(), order.begin() + route_used), got(ids, ids + route_used);
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        const double g_rel = std::fabs(double(sg) - sg_ref) / std::fabs(sg_ref);
        g_worst = std::max(g_worst, std::isfinite(sg) ? g_rel : INFINITY);
        bool ok = order_ok && std::isfinite(sg) && g_rel <= 1e-5;
        if (want == got) {
            double w_rel = 0.0;
            for (int32_t j = 0; j < route_used; ++j) {
                const double wr = p[size_t(ids[j])] * inv;
                w_rel = std::max(w_rel, std::isfinite(wts[j]) ? std::fabs(double(wts[j]) - wr) / wr : INFINITY);
            }
            w_worst = std::max(w_worst, w_rel);
            if (w_rel > 1e-5) {
                printf("  route b%d: weights rel %.3e with own logit error %.3e: FAIL\n", b, w_rel, own);
                ok = false;
            }
        } else {
            const double l10 = ref[order[route_used - 1]];
            std::vector<int32_t> diff;
            std::set_symmetric_difference(want.begin(), want.end(), got.begin(), got.end(), std::back_inserter(diff));
            double gap = 0.0;
            for (const int32_t e : diff) {
                gap = std::max(gap, std::fabs(ref[e] - l10));
            }
            const bool expl = gap <= 2.0 * own;
            printf("  route b%d: sets differ in %zu experts, gap %.3e, own logit error %.3e: %s\n", b, diff.size(), gap,
                   own, expl ? "explained" : "UNEXPLAINED: FAIL");
            explained += expl ? 1 : 0;
            ok = ok && expl;
        }
        if (!order_ok) {
            printf("  route b%d: order not by weight with ties to the lower ID: FAIL\n", b);
        }
        if (!(g_rel <= 1e-5)) {
            printf("  route b%d: gate rel %.3e with own logit error %.3e: FAIL\n", b, g_rel, own);
        }
        rows_ok += ok ? 1 : 0;
        failed += ok ? 0 : 1;
    }

    // a row with a NaN logit: the softmax's sum is NaN, so every p is sanitized to -FLT_MAX (ggml's rule) and all 512
    // tie: the IDs are 0 .. 9 and each weight is -FLT_MAX / max(the FP32 sum of ten -FLT_MAX, the floor), bitwise
    void nan_row(int32_t b, const int32_t * ids, const float * wts, float sg, double sg_ref) {
        float ws = 0.0f;
        for (int32_t j = 0; j < route_used; ++j) {
            ws += -FLT_MAX;
        }
        const float w = -FLT_MAX * (1.0f / std::max(ws, 6.103515625e-5f));
        bool ok = true;
        for (int32_t j = 0; j < route_used; ++j) {
            ok = ok && ids[j] == j && memcmp(&wts[j], &w, 4) == 0;
        }
        const double g_rel = std::fabs(double(sg) - sg_ref) / std::fabs(sg_ref);
        g_worst = std::max(g_worst, std::isfinite(sg) ? g_rel : INFINITY);
        if (!ok) {
            printf("  route b%d: NaN row not IDs 0 .. 9 with weight %g: FAIL\n", b, double(w));
        }
        if (!(g_rel <= 1e-5)) {
            printf("  route b%d: gate rel %.3e: FAIL\n", b, g_rel);
            ok = false;
        }
        rows_ok += ok ? 1 : 0;
        failed += ok ? 0 : 1;
    }
};

bool setup(case_run & cr, synth_cache & sc, const std::vector<uint8_t> & grids, reference & ref) {
    const tcase & tc = cr.tc;
    const int32_t k = tc.k, n = static_cast<int32_t>(tc.mats.size());
    product_args & a = cr.a;
    a.k = k;
    a.n_mats = n;
    bool any_bf = false, any_q = false;
    cr.off.assign(size_t(n), 0);
    for (int32_t i = 0; i < n; ++i) {
        const mspec & ms = tc.mats[size_t(i)];
        const synth * s = sc.get(ms.f, k);
        if (s == nullptr) {
            return false;
        }
        cr.off[size_t(i)] = (37 * i + 11) % n_synth_rows;
        std::vector<uint8_t> packed;
        if (!pack_synth(*s, ms.m, cr.off[size_t(i)], ms.period, packed)) {
            fprintf(stderr, "dp-test: pack %s failed\n", ms.name);
            return false;
        }
        if (i == 0 && tc.nan_expert >= 0) { // BF16 and F32 rows are packed as they are: the row's first weight a
                                            // quiet NaN (its little-endian bytes)
            uint8_t * w0 = packed.data() + size_t(tc.nan_expert) * s->row_bytes;
            if (ms.f == fmt::f32) {
                w0[0] = 0x00;
                w0[1] = 0x00;
                w0 += 2;
            }
            w0[0] = 0xc0;
            w0[1] = 0x7f;
        }
        cr.w.push_back(std::make_unique<dev_mem>(packed.size()));
        cr.out.push_back(std::make_unique<dev_mem>(size_t(max_rows) * ms.m * 4));
        if (!cr.w.back()->put(packed.data(), packed.size())) {
            return false;
        }
        mat & m = a.mats[i];
        m.f = ms.f;
        m.m = ms.m;
        m.w = cr.w.back()->p;
        m.out = ms.e == epi::f32 || ms.e == epi::conv_silu ? cr.out.back()->as<float>() : nullptr;
        m.e = ms.e;
        m.pair = ms.pair;
        m.split = ms.split;
        cr.glu_m = ms.e == epi::glu_gate ? ms.m : cr.glu_m;
        const bool wide = ms.f == fmt::bf16 || ms.f == fmt::f32; // F32 activations
        any_bf = any_bf || wide;
        any_q = any_q || !wide;
    }
    choose_splits(a);

    // activations: max_rows F32 rows from the case's name (B = r uses the first r), and their decode format
    ref.x.assign(size_t(max_rows) * k, 0.0f);
    if (!tc.zero) {
        lcg g(name_seed(tc.name));
        for (size_t i = 0; i < ref.x.size(); ++i) {
            const double v = (g.next() + g.next() + g.next() - 1.5) * (1.0 + 3.0 * double(i % 97 == 0));
            ref.x[i] = static_cast<float>(v * tc.xscale);
        }
    }
    ref.codes.resize(ref.x.size());
    ref.scales.resize(ref.x.size() / 16);
    quant::act_quantize_ref(ref.x.data(), max_rows, k, ref.codes.data(), ref.scales.data());
    std::vector<float> xv(ref.x.size());
    for (int32_t b = 0; b < max_rows; ++b) {
        quant::act_values_ref(ref.codes.data() + size_t(b) * k, ref.scales.data() + size_t(b) * k / 16, k,
                              xv.data() + size_t(b) * k);
    }
    // the shared down's scales: sigmoid-like values in (0, 1)
    ref.scale = {0.731058598f, 0.182425521f, 0.5f, 0.952574134f, 0.268941421f};

    const size_t gm = size_t(std::max(cr.glu_m, 32));
    cr.xc = std::make_unique<dev_mem>(ref.codes.size());
    cr.xs = std::make_unique<dev_mem>(ref.scales.size() * 4);
    cr.xf = std::make_unique<dev_mem>(ref.x.size() * 4);
    cr.grid = std::make_unique<dev_mem>(grid_bytes);
    cr.gc = std::make_unique<dev_mem>(max_rows * gm);
    cr.gs = std::make_unique<dev_mem>(max_rows * gm / 16 * 4);
    cr.gf = std::make_unique<dev_mem>(max_rows * gm * 4);
    cr.ids = std::make_unique<dev_mem>(size_t(max_rows) * route_used * 4);
    cr.wts = std::make_unique<dev_mem>(size_t(max_rows) * route_used * 4);
    cr.sg = std::make_unique<dev_mem>(size_t(max_rows) * 4);
    cr.scale = std::make_unique<dev_mem>(size_t(max_rows) * 4);
    if (!cr.xc->put(ref.codes.data(), ref.codes.size()) || !cr.xs->put(ref.scales.data(), ref.scales.size() * 4) ||
        !cr.xf->put(ref.x.data(), ref.x.size() * 4) || !cr.grid->put(grids.data(), grids.size()) ||
        !cr.scale->put(ref.scale.data(), ref.scale.size() * 4)) {
        return false;
    }
    a.x_codes = any_q ? cr.xc->as<int8_t>() : nullptr;
    a.x_scales = any_q ? cr.xs->as<float>() : nullptr;
    a.x_f32 = any_bf ? cr.xf->as<float>() : nullptr;
    a.grids = cr.grid->p;
    a.glu_codes = cr.glu_m > 0 ? cr.gc->as<int8_t>() : nullptr;
    a.glu_scales = cr.glu_m > 0 ? cr.gs->as<float>() : nullptr;
    a.glu_f32 = cr.glu_m > 0 ? cr.gf->as<float>() : nullptr;
    for (int32_t i = 0; i < n; ++i) {
        a.mats[i].scale = tc.mats[size_t(i)].scaled ? cr.scale->as<float>() : nullptr;
    }
    if (tc.route) {
        a.route.logits = 0;
        a.route.gate = 1;
        a.route.ids = cr.ids->as<int32_t>();
        a.route.wts = cr.wts->as<float>();
        a.route.sg = cr.sg->as<float>();
    }
    if (tc.conv) {
        // taps and history from the case's name: the scale of the model's (|w| < 1, history like the products)
        lcg g(name_seed(tc.name + " conv"));
        ref.taps.resize(size_t(fn::gdn::n_channel) * fn::gdn::n_tap);
        ref.hist0.resize(size_t(fn::gdn::hist_version_floats));
        for (float & v : ref.taps) {
            v = static_cast<float>(g.next() - 0.5);
        }
        for (float & v : ref.hist0) {
            v = static_cast<float>((g.next() - 0.5) * 8.0);
        }
        cr.taps = std::make_unique<dev_mem>(ref.taps.size() * 4);
        cr.hist = std::make_unique<dev_mem>(size_t(fn::gdn::n_version) * fn::gdn::hist_version_floats * 4);
        cr.cur = std::make_unique<dev_mem>(4);
        cr.hist0 = &ref.hist0;
        if (!cr.taps->put(ref.taps.data(), ref.taps.size() * 4)) {
            return false;
        }
        a.conv.w = cr.taps->as<float>();
        a.conv.hist = cr.hist->as<float>();
        a.conv.cur = cr.cur->as<uint32_t>();
    }
    cr.work = std::make_unique<dev_mem>(work_bytes(a));
    if (!cuda_ok(cudaMemset(cr.work->p, 0, cr.work->n), "memset work")) {
        return false;
    }
    a.work = cr.work->p;

    // FP64 reference: each used synthetic row's dot with every activation row
    ref.dot.assign(size_t(n), {});
    for (int32_t i = 0; i < n; ++i) {
        const mspec & ms = tc.mats[size_t(i)];
        const std::vector<float> & w = sc.deq[synth_cache::key(ms.f, k)];
        const std::vector<float> & xa = ms.f == fmt::bf16 || ms.f == fmt::f32 ? ref.x : xv;
        std::vector<double> & d = ref.dot[size_t(i)];
        d.assign(size_t(max_rows) * ms.m, 0.0);
        // rows repeat every period and every n_synth_rows: only the distinct ones are summed
        const int32_t period = std::min(ms.period, n_synth_rows), distinct = std::min(ms.m, period);
        parallel_for(distinct, [&](int32_t r) {
            const float * wr = w.data() + size_t((r + cr.off[size_t(i)]) % n_synth_rows) * k;
            for (int32_t b = 0; b < max_rows; ++b) {
                double s = 0.0;
                for (int32_t j = 0; j < k; ++j) {
                    s += double(wr[j]) * double(xa[size_t(b) * k + j]);
                }
                d[size_t(b) * ms.m + r] = ms.scaled ? double(ref.scale[size_t(b)]) * s : s;
            }
        });
        for (int32_t b = 0; b < max_rows; ++b) {
            for (int32_t r = distinct; r < ms.m; ++r) {
                d[size_t(b) * ms.m + r] = d[size_t(b) * ms.m + r % period];
            }
            if (i == 0 && tc.nan_expert >= 0) {
                d[size_t(b) * ms.m + size_t(tc.nan_expert)] = NAN;
            }
        }
    }
    return true;
}

// one case at every row count of rows_mask: two runs each (bitwise), the reference checks, then each row at B = 1
// against the same row in every other run (batch invariance)
bool run_case(const tcase & tc, uint32_t rows_mask, synth_cache & sc, const std::vector<uint8_t> & grids) {
    case_run cr(tc);
    reference ref;
    if (!setup(cr, sc, grids, ref)) {
        return false;
    }
    const int32_t k = tc.k, n = static_cast<int32_t>(tc.mats.size());
    std::string splits;
    for (int32_t i = 0; i < n; ++i) {
        splits += std::string(i ? "," : "") + std::to_string(cr.a.mats[i].split);
    }
    printf("dp-test: %s K %d splits %s%s\n", tc.name.c_str(), k, splits.c_str(), tc.zero ? " (zero x)" : "");
    bool pass = true;
    std::map<int32_t, outputs> runs;
    int32_t max_b = 0;
    for (int32_t rows = 1; rows <= max_rows; ++rows) {
        if (((rows_mask >> (rows - 1)) & 1u) == 0) {
            continue;
        }
        if (rows == 5 && k > 3072) { // SP2: describe rejects it (dense.cu max_k_b5; the eh_proj runs at B <= 4)
            printf(" B5: not a launch (K above 3,072)\n");
            continue;
        }
        max_b = rows;
        outputs o1, o2;
        if (!cr.launch(rows, 0, o1) || !cr.launch(rows, 0, o2)) {
            return false;
        }
        const bool rerun = same_all(o1, o2);
        printf(" B%d: second run %s\n", rows, rerun ? "bitwise" : "DIFFERS: FAIL");
        pass = pass && rerun;
        for (int32_t i = 0; i < n; ++i) {
            const mspec & ms = tc.mats[size_t(i)];
            if (ms.e != epi::f32) {
                continue;
            }
            bound bd;
            for (int32_t b = 0; b < rows; ++b) {
                bd.row(o1.out[size_t(i)].data() + size_t(b) * ms.m, ref.dot[size_t(i)].data() + size_t(b) * ms.m, ms.m);
            }
            bd.print(std::string(ms.name) + " " + fmt_name(ms.f) + (ms.scaled ? " (scaled)" : ""));
            pass = pass && bd.ok;
        }
        if (cr.glu_m > 0) {
            int32_t gi = 0, ui = 0;
            for (int32_t i = 0; i < n; ++i) {
                gi = tc.mats[size_t(i)].e == epi::glu_gate ? i : gi;
                ui = tc.mats[size_t(i)].e == epi::glu_up ? i : ui;
            }
            const int32_t m = cr.glu_m;
            bound bd;
            for (int32_t b = 0; b < rows; ++b) {
                std::vector<double> want(static_cast<size_t>(m));
                for (int32_t c = 0; c < m; ++c) {
                    const double g = ref.dot[size_t(gi)][size_t(b) * m + c], u = ref.dot[size_t(ui)][size_t(b) * m + c];
                    want[size_t(c)] = g / (1.0 + std::exp(-g)) * u;
                }
                bd.row(o1.glu_f32.data() + size_t(b) * m, want.data(), m);
            }
            bd.print("glu (F32)");
            std::vector<int8_t> qc(o1.glu_codes.size());
            std::vector<float> qs(o1.glu_scales.size());
            quant::act_quantize_ref(o1.glu_f32.data(), rows, m, qc.data(), qs.data());
            const bool bitwise = qc == o1.glu_codes && memcmp(qs.data(), o1.glu_scales.data(), qs.size() * 4) == 0;
            printf("  %-34s %s\n", "glu codes and scales against act_ref", bitwise ? "bitwise" : "DIFFER: FAIL");
            pass = pass && bd.ok && bitwise;
        }
        if (tc.conv) {
            // the products without the epilogue (the same launch with epi f32): the history versions are bitwise
            // copies of them; the post-SiLU values against FP64 on the reference products
            const int32_t C = fn::gdn::n_channel, H = fn::gdn::n_hist;
            product_args l = cr.a;
            l.mats[0].e = epi::f32;
            dev_mem xo(size_t(max_rows) * C * 4);
            l.mats[0].out = xo.as<float>();
            std::vector<float> xk(size_t(rows) * C);
            if (!product(rows, l, nullptr) || !cuda_ok(cudaDeviceSynchronize(), "product") ||
                !xo.get(xk.data(), xk.size() * 4)) {
                return false;
            }
            bool hist_ok = true;
            bound by;
            std::vector<double> want(static_cast<size_t>(C));
            for (int32_t b = 0; b < rows; ++b) {
                for (int32_t c = 0; c < C; ++c) {
                    // seq: the 3 history taps (oldest first), then the rows
                    const auto seq_f = [&](int32_t i) {
                        return i < H ? ref.hist0[size_t(c) * H + i] : xk[size_t(i - H) * C + c];
                    };
                    const auto seq_d = [&](int32_t i) {
                        return i < H ? double(ref.hist0[size_t(c) * H + i]) : ref.dot[0][size_t(i - H) * C + c];
                    };
                    for (int32_t t = 0; t < H; ++t) {
                        const float got = o1.hist[size_t(b) * fn::gdn::hist_version_floats + size_t(c) * H + t];
                        const float exp = seq_f(b + 1 + t);
                        hist_ok = hist_ok && memcmp(&got, &exp, 4) == 0;
                    }
                    double acc = 0.0;
                    for (int32_t t = 0; t < fn::gdn::n_tap; ++t) {
                        acc += seq_d(b + t) * double(ref.taps[size_t(c) * fn::gdn::n_tap + t]);
                    }
                    want[size_t(c)] = acc / (1.0 + std::exp(-acc));
                }
                by.row(o1.out[0].data() + size_t(b) * C, want.data(), C);
            }
            by.print("conv + SiLU epilogue (F32)");
            printf("  %-34s %s\n", "history versions (copies)", hist_ok ? "bitwise" : "DIFFER: FAIL");
            pass = pass && by.ok && hist_ok;
        }
        if (tc.route) {
            route_check rc;
            for (int32_t b = 0; b < rows; ++b) {
                const double sg_ref = 1.0 / (1.0 + std::exp(-ref.dot[1][size_t(b)]));
                if (tc.nan_expert >= 0) {
                    rc.nan_row(b, o1.ids.data() + size_t(b) * route_used, o1.wts.data() + size_t(b) * route_used,
                               o1.sg[size_t(b)], sg_ref);
                    continue;
                }
                rc.row(b, o1.out[0].data() + size_t(b) * route_experts, ref.dot[0].data() + size_t(b) * route_experts,
                       o1.ids.data() + size_t(b) * route_used, o1.wts.data() + size_t(b) * route_used, o1.sg[size_t(b)],
                       sg_ref);
            }
            printf("  %-34s %d rows match, %d explained, %d failed; weights rel %.3e, gate rel %.3e, own logit "
                   "error %.3e (worst)  %s\n",
                   "route IDs, weights, gate", rc.rows_ok - rc.explained, rc.explained, rc.failed, rc.w_worst,
                   rc.g_worst, rc.own_worst, rc.failed == 0 ? "ok" : "FAIL");
            pass = pass && rc.failed == 0;
        }
        runs[rows] = std::move(o1);
    }
    // batch invariance: each row r alone at B = 1 against row r in every run of more rows (not the convolution: row r
    // reads rows 0 .. r - 1 through the history, so its products' invariance is the plain GDN input case's)
    if (!tc.conv && (runs.size() > 1 || (max_b > 1 && runs.count(1) == 0))) {
        bool inv = true;
        for (int32_t r = 0; r < max_b; ++r) {
            outputs alone;
            if (!cr.launch(1, r, alone)) {
                return false;
            }
            for (const auto & [rows, o] : runs) {
                inv = inv && (rows <= r || same_row(alone, 0, o, r));
            }
        }
        printf("  %-34s %s\n", "each row at B = 1 against every B", inv ? "bitwise" : "DIFFERS: FAIL");
        pass = pass && inv;
    }
    // SR1: the largest run again at the committed slots the runs did not reach, bitwise (outputs and versions)
    if (tc.conv && !runs.empty()) {
        const auto & [rows, o] = *runs.rbegin();
        bool same = true;
        int32_t extra = 0;
        const uint32_t all_slots = (1u << fn::gdn::n_version) - 1u; // SP2: six
        while (cr.slots_seen != all_slots && extra < fn::gdn::n_version) {
            outputs again;
            if (!cr.launch(rows, 0, again)) {
                return false;
            }
            same = same && again.out[0] == o.out[0] && again.hist == o.hist;
            ++extra;
        }
        printf("  %-34s %s (%d of %d slots)\n", "conv at every committed slot",
               same && cr.slots_seen == all_slots ? "bitwise" : "DIFFERS: FAIL", __builtin_popcount(cr.slots_seen),
               fn::gdn::n_version);
        pass = pass && same && cr.slots_seen == all_slots;
    }
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// LP1 Amendment 3: the frame's DP2 work sites (fn/model/layer.h). On one card, the QSA input's owner shape, then the
// peer's, then the owner's again, each through its own site in one frame work buffer zeroed once, must be bitwise
// equal to the same launch alone on freshly zeroed work, at B = 1 and 4. Negative controls through one shared region:
// the frame's launch must refuse the peer's layout at the owner's site with nothing enqueued, and the same sequence
// run unchecked must corrupt the second owner launch (the peer's partials over the owner's counters), so the bitwise
// comparison reaches the path.
bool run_site_case(uint32_t rows_mask, synth_cache & sc, const std::vector<uint8_t> & grids) {
    namespace fm = fn::model;
    const tcase owner_tc = {"QSA input owner (layer 11 formats)",
                            2560,
                            {plain("attn_q", fmt::q4_k, 6144), plain("attn_k", fmt::q4_k, 256),
                             plain("attn_v", fmt::q4_k, 256), plain("idx_q", fmt::bf16, 512),
                             plain("idx_k", fmt::bf16, 128)}};
    const tcase peer_tc = {
        "QSA input peer (layer 11 formats)",
        2560,
        {plain("attn_q", fmt::q4_k, 6144), plain("attn_k", fmt::q4_k, 256), plain("attn_v", fmt::q4_k, 256)}};
    printf("frame sites: QSA input owner, peer, owner (K 2560, layer 11 formats)\n");
    case_run oc(owner_tc), pc(peer_tc);
    reference oref, pref;
    fm::dense_layouts layouts;
    std::string err;
    if (!setup(oc, sc, grids, oref) || !setup(pc, sc, grids, pref) || !fm::dense_site_layouts(layouts, err)) {
        printf("  setup failed %s\n  FAIL\n", err.c_str());
        return false;
    }
    const auto site_of = [&](fm::dense_site s) { return layouts[size_t(s)]; };
    work_layout ol, pl;
    const bool lay_ok = layout_of(oc.a, ol) && layout_of(pc.a, pl) &&
                        same_layout(ol, site_of(fm::dense_site::qsa_input_owner)) &&
                        same_layout(pl, site_of(fm::dense_site::qsa_input_peer)) && !same_layout(ol, pl);
    printf("  owner: %s\n  peer:  %s\n  %-34s %s\n", fm::dense_layout_text(ol).c_str(),
           fm::dense_layout_text(pl).c_str(), "the sites' layouts", lay_ok ? "ok, two" : "FAIL");
    bool pass = lay_ok;

    dev_mem frame(fm::dense_work_bytes());
    if (!cuda_ok(cudaMemset(frame.p, 0, frame.n), "memset frame work")) {
        return false;
    }
    const auto zero = [](const dev_mem & m) { return cuda_ok(cudaMemset(m.p, 0, m.n), "memset work"); };
    const auto at_site = [&](case_run & cr, fm::dense_site s) {
        cr.via = [&frame, &layouts, s](int32_t rows, const product_args & l) {
            std::string why;
            const bool ok = fm::dense_site_launch(frame.p, layouts, s, l, rows, nullptr, why);
            if (!ok) {
                fprintf(stderr, "dp-test: %s\n", why.c_str());
            }
            return ok;
        };
    };
    const auto in_region = [](case_run & cr, uint8_t * work) {
        cr.via = [work](int32_t rows, const product_args & l) {
            product_args x = l;
            x.work = work;
            return product(rows, x, nullptr);
        };
    };
    int32_t n_rows = 0;
    for (const int32_t rows : {1, 4}) {
        if ((rows_mask >> (rows - 1) & 1u) == 0) {
            continue;
        }
        ++n_rows;
        // each launch alone on freshly zeroed work
        outputs o_alone, p_alone;
        oc.via = nullptr;
        pc.via = nullptr;
        if (!zero(*oc.work) || !oc.launch(rows, 0, o_alone) || !zero(*pc.work) || !pc.launch(rows, 0, p_alone)) {
            printf("  B%d: alone launches failed\n  FAIL\n", rows);
            return false;
        }
        // the frame's sites: owner, peer, owner, in the one frame buffer zeroed once for the case
        outputs o1, p1, o2;
        at_site(oc, fm::dense_site::qsa_input_owner);
        at_site(pc, fm::dense_site::qsa_input_peer);
        const bool site_ran = oc.launch(rows, 0, o1) && pc.launch(rows, 0, p1) && oc.launch(rows, 0, o2);
        const bool site_ok = site_ran && same_all(o1, o_alone) && same_all(p1, p_alone) && same_all(o2, o_alone);
        printf(" B%d: %-38s %s\n", rows, "frame sites owner, peer, owner",
               site_ok ? "bitwise equal to each alone" : "DIFFER: FAIL");
        pass = pass && site_ok;

        // negative control 1: the peer's launch at the owner's site (one shared region) is refused by the frame's
        // runtime layout compare, before anything is enqueued (the region's bytes and the peer's outputs unchanged)
        std::vector<uint8_t> before(frame.n), after(frame.n);
        bool refused = false;
        std::string why;
        if (pc.out[0]->poison() && frame.get(before.data(), before.size())) {
            product_args l = pc.a;
            refused = !fm::dense_site_launch(frame.p, layouts, fm::dense_site::qsa_input_owner, l, rows, nullptr, why);
            std::vector<uint8_t> out0(pc.out[0]->n);
            refused = refused && cuda_ok(cudaDeviceSynchronize(), "refused launch") &&
                      frame.get(after.data(), after.size()) && pc.out[0]->get(out0.data(), out0.size()) &&
                      before == after && std::all_of(out0.begin(), out0.end(), [](uint8_t b) { return b == 0xff; });
        }
        printf(" B%d: %-38s %s\n", rows, "checked, peer at the owner's site",
               refused ? "refused, nothing enqueued" : "NOT REFUSED: FAIL");
        if (rows == 1 || !refused) {
            printf("      %s\n", why.c_str());
        }
        pass = pass && refused;

        // negative control 2: the same sequence unchecked in one shared region zeroed once corrupts the second owner
        // launch, so the bitwise comparison above reaches the path
        dev_mem shared(std::max(work_bytes(oc.a), work_bytes(pc.a)));
        outputs u1, u2, u3;
        in_region(oc, shared.p);
        in_region(pc, shared.p);
        const bool u_ran = zero(shared) && oc.launch(rows, 0, u1) && pc.launch(rows, 0, u2) && oc.launch(rows, 0, u3);
        const bool caught = u_ran && same_all(u1, o_alone) && same_all(u2, p_alone) && !same_all(u3, o_alone);
        printf(" B%d: %-38s %s\n", rows, "unchecked, one shared region",
               caught ? "second owner launch differs (as expected)" : "NOT CAUGHT: FAIL");
        pass = pass && caught;
        oc.via = nullptr;
        pc.via = nullptr;
    }
    if (n_rows == 0) {
        printf("  no row count of 1 or 4 selected\n");
    }
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

bool make_synth(fmt f, int32_t k, synth & out) {
    const quant::wfmt q = to_wfmt(f);
    out.f = f;
    out.k = k;
    out.row_bytes = quant::row_bytes(q, k);
    if (out.row_bytes == 0 || int64_t(out.row_bytes) != row_bytes(f, k)) {
        return false;
    }
    out.bytes.assign(out.row_bytes * n_synth_rows, 0);
    const int64_t n_blocks = int64_t(n_synth_rows) * (k / quant::info(q).block_values);
    const uint64_t index = dp_seed_index + uint64_t(f) * 4096 + uint64_t(k);
    return quant::synth_blocks(q, quant::synth_seed(index), n_blocks, out.bytes.data());
}

bool pack_synth(const synth & s, int32_t m, int32_t off, int32_t period, std::vector<uint8_t> & out) {
    if (period < 1) {
        return false;
    }
    std::vector<uint8_t> rows(size_t(m) * s.row_bytes);
    for (int32_t i = 0; i < m; ++i) {
        memcpy(rows.data() + size_t(i) * s.row_bytes, s.row((i % period + off) % n_synth_rows), s.row_bytes);
    }
    out.assign(rows.size(), 0);
    return pack(s.f, m, s.k, rows.data(), s.row_bytes, out.data());
}

bool grid_buffer(std::vector<uint8_t> & out) {
    out.assign(grid_bytes, 0);
    return grid_table(experts::iq3s_grid, sizeof(experts::iq3s_grid), experts::kvalues_iq4nl,
                      sizeof(experts::kvalues_iq4nl), out.data());
}

} // namespace fn::dense

int fnfl_dp_test(int device, uint32_t rows_mask, const char * mode) {
    using namespace fn::dense;
    const bool full = strcmp(mode, "full") == 0 || strcmp(mode, "all") == 0;
    const bool san = strcmp(mode, "san") == 0 || strcmp(mode, "all") == 0;
    if (!full && !san) {
        fprintf(stderr, "dp-test: mode is full, san or all\n");
        return 2;
    }
    if (cudaSetDevice(device) != cudaSuccess) {
        fprintf(stderr, "dp-test: cannot use device %d\n", device);
        return 1;
    }
    synth_cache sc;
    std::vector<uint8_t> grids;
    if (!grid_buffer(grids)) {
        fprintf(stderr, "dp-test: grid tables\n");
        return 1;
    }
    std::vector<tcase> cases;
    // every format alone: a small shape and a deployed one
    for (const fmt f : {fmt::q4_k, fmt::q5_k, fmt::q6_k, fmt::iq4_xs, fmt::iq3_s}) {
        cases.push_back({std::string("small ") + fmt_name(f), 512, {plain("w", f, 64)}});
        cases.push_back({std::string("attn_qkv ") + fmt_name(f), 2560, {plain("attn_qkv", f, 5120)}});
    }
    cases.push_back({"small Q8_0 (tail step)", 320, {plain("w", fmt::q8_0, 64)}});
    cases.push_back({"small Q8_0", 512, {plain("w", fmt::q8_0, 64)}});
    cases.push_back({"attn_q Q8_0", 2560, {plain("attn_q", fmt::q8_0, 6144)}});
    cases.push_back({"small IQ4_NL", 512, {plain("w", fmt::iq4_nl, 64)}});
    cases.push_back({"small Q2_0", 512, {plain("w", fmt::q2_0, 64)}});
    cases.push_back({"small Q2_0 (tail step)", 320, {plain("w", fmt::q2_0, 64)}});
    cases.push_back({"small BF16 (tail rows)", 512, {plain("w", fmt::bf16, 22)}});
    cases.push_back({"router BF16 (no epilogue)", 2560, {plain("router", fmt::bf16, 512)}});
    // the draft block's shapes and formats (DR1 Stage B): F32 weights, eh_proj's K 5,120 split and unsplit
    cases.push_back({"small F32 (tail rows)", 512, {plain("w", fmt::f32, 22)}});
    cases.push_back({"router F32 (no epilogue)", 2560, {plain("router", fmt::f32, 512)}});
    cases.push_back({"eh_proj Q8_0 K 5120 split (layer 48)", 5120, {plain("eh_proj", fmt::q8_0, 1280)}});
    cases.push_back({"eh_proj Q8_0 K 5120 unsplit", 5120, {plain("eh_proj", fmt::q8_0, 1280, 1)}});
    cases.push_back({"small Q8_0 K 5120 split 10", 5120, {plain("w", fmt::q8_0, 64)}});
    // the groupings, mixed formats (deployed layers' formats where named)
    const std::vector<mspec> gdn0 = {plain("attn_qkv", fmt::iq4_xs, 5120), plain("attn_gate", fmt::iq3_s, 3072),
                                     plain("alpha", fmt::bf16, 24), plain("beta", fmt::bf16, 24)};
    cases.push_back({"GDN input (layer 0)", 2560, gdn0});
    cases.push_back({"GDN input (layer 10)",
                     2560,
                     {plain("attn_qkv", fmt::q6_k, 5120), plain("attn_gate", fmt::q6_k, 3072),
                      plain("alpha", fmt::bf16, 24), plain("beta", fmt::bf16, 24)}});
    cases.push_back(
        {"QSA input (mixed)",
         2560,
         {plain("attn_q", fmt::q4_k, 6144), plain("attn_k", fmt::iq4_xs, 256), plain("attn_v", fmt::q6_k, 256),
          plain("idx_q", fmt::bf16, 512), plain("idx_k", fmt::bf16, 128)}});
    cases.push_back(
        {"QSA input (layer 48, Q8_0)",
         2560,
         {plain("attn_q", fmt::q8_0, 6144), plain("attn_k", fmt::q8_0, 256), plain("attn_v", fmt::q8_0, 256),
          plain("idx_q", fmt::bf16, 512), plain("idx_k", fmt::bf16, 128)}});
    cases.push_back({"ssm_out Q5_K split", 3072, {plain("ssm_out", fmt::q5_k, 2560)}});
    cases.push_back({"ssm_out Q5_K unsplit", 3072, {plain("ssm_out", fmt::q5_k, 2560, 1)}});
    cases.push_back({"ssm_out IQ3_S split", 3072, {plain("ssm_out", fmt::iq3_s, 2560)}});
    cases.push_back({"attn_output Q6_K split", 3072, {plain("attn_output", fmt::q6_k, 2560)}});
    cases.push_back({"attn_output Q8_0 split", 3072, {plain("attn_output", fmt::q8_0, 2560)}});
    cases.push_back({"attn_output IQ4_XS split", 3072, {plain("attn_output", fmt::iq4_xs, 2560)}});
    cases.push_back({"ssm_out Q4_K split", 3072, {plain("ssm_out", fmt::q4_k, 2560)}});
    const std::vector<mspec> shx0 = {glu("gate", fmt::iq4_xs, epi::glu_gate, 1), glu("up", fmt::iq3_s, epi::glu_up, 0)};
    cases.push_back({"shexp gate/up GLU (layer 0) split", 2560, shx0});
    cases.push_back({"shexp gate/up GLU (layer 0) unsplit",
                     2560,
                     {glu("gate", fmt::iq4_xs, epi::glu_gate, 1, 1), glu("up", fmt::iq3_s, epi::glu_up, 0, 1)}});
    cases.push_back({"shexp gate/up GLU (layer 11)",
                     2560,
                     {glu("gate", fmt::q4_k, epi::glu_gate, 1), glu("up", fmt::q4_k, epi::glu_up, 0)}});
    cases.push_back({"shexp gate/up GLU (layer 48)",
                     2560,
                     {glu("up", fmt::q8_0, epi::glu_up, 1), glu("gate", fmt::q8_0, epi::glu_gate, 0)}});
    cases.push_back({"shexp gate/up GLU (Q5_K / Q6_K)",
                     2560,
                     {glu("gate", fmt::q5_k, epi::glu_gate, 1), glu("up", fmt::q6_k, epi::glu_up, 0)}});
    cases.push_back({"shexp down IQ4_NL (layer 0)", 320, {scaled("down", fmt::iq4_nl, 2560)}});
    cases.push_back({"shexp down Q2_0 (layer 3)", 320, {scaled("down", fmt::q2_0, 2560)}});
    cases.push_back({"shexp down Q8_0 (layer 48)", 320, {scaled("down", fmt::q8_0, 2560)}});
    cases.push_back({"shexp down Q2_0 unscaled", 320, {plain("down", fmt::q2_0, 2560)}});
    if (san) {
        cases.push_back({"vocabulary Q5_K (8,192 rows)", 2560, {plain("vocab", fmt::q5_k, 8192)}});
    }
    if (full) {
        cases.push_back({"vocabulary Q5_K", 2560, {plain("vocab", fmt::q5_k, 124160)}});
    }
    // the router at its real shape with the epilogue: realistic logits, split and unsplit, exact ties (rows repeating
    // every 128, so equal logits meet at the 10th place), a NaN logit (every p sanitized: all 512 tied) and zero
    // activations (all 512 tied)
    const std::vector<mspec> router = {plain("router", fmt::bf16, 512), plain("shexp_gate", fmt::bf16, 1)};
    std::vector<mspec> router_ties = router;
    router_ties[0].period = 128;
    cases.push_back({"router + shared gate split", 2560, router, false, true, 0.02});
    cases.push_back({"router + shared gate unsplit",
                     2560,
                     {plain("router", fmt::bf16, 512, 1), plain("shexp_gate", fmt::bf16, 1, 1)},
                     false,
                     true,
                     0.02});
    cases.push_back({"router + shared gate (wide logits)", 2560, router, false, true, 0.2});
    cases.push_back({"router + shared gate (ties)", 2560, router_ties, false, true, 0.02});
    cases.push_back({"router + shared gate (NaN logit)", 2560, router, false, true, 0.02, 300});
    // the draft's router (DR1): the same epilogue on F32 matrices, the same five cases
    const std::vector<mspec> router_f32 = {plain("router", fmt::f32, 512), plain("shexp_gate", fmt::f32, 1)};
    std::vector<mspec> router_f32_ties = router_f32;
    router_f32_ties[0].period = 128;
    cases.push_back({"F32 router + shared gate split (layer 48)", 2560, router_f32, false, true, 0.02});
    cases.push_back({"F32 router + shared gate unsplit",
                     2560,
                     {plain("router", fmt::f32, 512, 1), plain("shexp_gate", fmt::f32, 1, 1)},
                     false,
                     true,
                     0.02});
    cases.push_back({"F32 router + shared gate (wide logits)", 2560, router_f32, false, true, 0.2});
    cases.push_back({"F32 router + shared gate (ties)", 2560, router_f32_ties, false, true, 0.02});
    cases.push_back({"F32 router + shared gate (NaN logit)", 2560, router_f32, false, true, 0.02, 300});
    // the GDN input with the convolution epilogue (LP1) at layer 10's formats and layer 0's
    cases.push_back({"GDN input + conv epilogue (layer 10)",
                     2560,
                     {conv("attn_qkv", fmt::q6_k), plain("attn_gate", fmt::q6_k, 3072), plain("alpha", fmt::bf16, 24),
                      plain("beta", fmt::bf16, 24)},
                     false,
                     false,
                     1.0,
                     -1,
                     true});
    cases.push_back({"GDN input + conv epilogue (layer 0)",
                     2560,
                     {conv("attn_qkv", fmt::iq4_xs), plain("attn_gate", fmt::iq3_s, 3072),
                      plain("alpha", fmt::bf16, 24), plain("beta", fmt::bf16, 24)},
                     false,
                     false,
                     1.0,
                     -1,
                     true});
    cases.push_back({"GDN input (layer 0)", 2560, gdn0, true});
    cases.push_back({"shexp gate/up GLU (layer 0)", 2560, shx0, true});
    cases.push_back({"router + shared gate", 2560, router, true, true});
    cases.push_back({"F32 router + shared gate", 2560, router_f32, true, true});

    int32_t n_run = 0, n_fail = 0;
    for (const tcase & tc : cases) {
        const bool ok = run_case(tc, rows_mask, sc, grids);
        ++n_run;
        n_fail += ok ? 0 : 1;
    }
    const bool sites_ok = run_site_case(rows_mask, sc, grids); // LP1 Amendment 3
    ++n_run;
    n_fail += sites_ok ? 0 : 1;
    printf("dp-test: %d cases, %d failed\n", n_run, n_fail);
    printf("dp-test: %s\n", n_fail == 0 ? "PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
