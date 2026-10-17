#include "fn/proto/xp1/xp1_test.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_test.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/proto/xp1/hx7.h"
#include "fn/proto/xp1/pair.h"
#include "fn/proto/xp1/standin.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace p1 = fn::xp1;

constexpr int32_t D        = 2560;
constexpr int32_t F        = 1280;
constexpr int32_t S        = 4;
constexpr int32_t K        = S * D;
constexpr int32_t L        = 320;
constexpr int32_t J        = 4;
constexpr int32_t M        = 1300;
constexpr int32_t NU       = 10;
constexpr int32_t NFF      = 640;
constexpr double  hc_bound = 1e-6;  // HC2's test bound (max |err| / max |ref|)
constexpr double  ex_bound = 1e-5;  // EX1b's test bound

struct checks {
    int32_t n = 0, bad = 0;

    void ok(bool pass, const std::string & what) {
        ++n;
        if (!pass) {
            ++bad;
            printf("xp1-test: FAIL %s\n", what.c_str());
        }
    }
};

template <typename F> void parallel_for(int32_t n, const F & fn) {
    const int32_t            nt = std::min<int32_t>(n, 12);
    std::atomic<int32_t>     next{ 0 };
    std::vector<std::thread> th;
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

double from_bf16(uint16_t h) {
    const uint32_t u = uint32_t(h) << 16;
    float          f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct err_acc {
    double e = 0.0, r = 0.0;
    bool   finite = true;

    void add(double got, double ref) {
        finite = finite && std::isfinite(got);
        e      = std::max(e, std::fabs(got - ref));
        r      = std::max(r, std::fabs(ref));
    }

    double rel() const { return finite ? (r > 0.0 ? e / r : e) : INFINITY; }
};

// the stand-ins' defined function (standin.h), on the CPU
float f_out(const float * in, int32_t in_width, int32_t b, int32_t j, float c) {
    const float h = 0.5f * in[b * in_width + j % in_width];  // exact (a power-of-two scale)
    return h + c;
}

// ---- HC2's FP64 reference (hc_test.cpp's make_reference, here from the recorded full-width inputs) ----
struct hc_ref {
    std::vector<double> x, inject, mixed;
};

void hc_reference(const p1::pair::facts &    fx,
                  int32_t                    k,
                  int32_t                    B,
                  const std::vector<float> & hres,
                  const std::vector<float> & hpost,
                  const std::vector<float> & hx,
                  hc_ref &                   ref) {
    ref.x.assign(size_t(B) * S * D, 0.0);
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t s = 0; s < S; ++s) {
            for (int32_t d = 0; d < D; ++d) {
                const size_t at = (size_t(b) * S + s) * D + d;
                ref.x[at]       = double(hres[at]) + double(hpost[size_t(b) * S + s]) * hx[size_t(b) * D + d];
            }
        }
    }
    std::vector<double> mom(size_t(2) * B * M, 0.0);
    parallel_for(2 * B, [&](int32_t i) {
        const int32_t r = i % 2, b = i / 2;
        double *      m = &mom[(size_t(r) * B + b) * M];
        for (int32_t s = 0; s < S; ++s) {
            const double * xs = &ref.x[(size_t(b) * S + s) * D + r * F];
            double         v  = 0.0;
            for (int32_t q = 0; q < F; ++q) {
                v += xs[q] * xs[q];
            }
            m[s] = v;
            for (int32_t j = 0; j < L + J; ++j) {
                const uint16_t * wr = j < L ? &fx.hc_down[k][size_t(j) * K] : &fx.hc_inject[k][size_t(j - L) * K];
                double           u  = 0.0;
                for (int32_t q = 0; q < F; ++q) {
                    u += from_bf16(wr[s * D + r * F + q]) * double(fx.hc_gamma[k][s * D + r * F + q]) * xs[q];
                }
                m[j < L ? 4 + L * s + j : 4 + L * S + J * s + (j - L)] = u;
            }
        }
    });
    ref.inject.assign(size_t(B) * J, 0.0);
    ref.mixed.assign(size_t(B) * D, 0.0);
    for (int32_t b = 0; b < B; ++b) {
        const double * m0 = &mom[(size_t(0) * B + b) * M];
        const double * m1 = &mom[(size_t(1) * B + b) * M];
        double         rms[S];
        for (int32_t s = 0; s < S; ++s) {
            rms[s] = std::sqrt((m0[s] + m1[s]) / D + 1e-6);
        }
        std::vector<double> low(L);
        for (int32_t j = 0; j < L; ++j) {
            double z = 0.0;
            for (int32_t s = 0; s < S; ++s) {
                z += (m0[4 + L * s + j] + m1[4 + L * s + j]) / rms[s];
            }
            z *= 0.25;
            low[j] = z / (1.0 + std::exp(-z));
        }
        for (int32_t q = 0; q < J; ++q) {
            double z = 0.0;
            for (int32_t s = 0; s < S; ++s) {
                z += (m0[4 + L * S + J * s + q] + m1[4 + L * S + J * s + q]) / rms[s];
            }
            ref.inject[size_t(b) * J + q] = 2.0 / (1.0 + std::exp(-0.25 * z));
        }
        parallel_for(D / 64, [&](int32_t chunk) {
            for (int32_t d = chunk * 64; d < chunk * 64 + 64; ++d) {
                double mix = 0.0;
                for (int32_t s = 0; s < S; ++s) {
                    const uint16_t * wr = &fx.hc_up[k][size_t(s * D + d) * L];
                    double           z  = 0.0;
                    for (int32_t j = 0; j < L; ++j) {
                        z += from_bf16(wr[j]) * low[j];
                    }
                    const double gate = 1.0 / (1.0 + std::exp(-z));
                    mix += double(fx.hc_gamma[k][s * D + d]) * ref.x[(size_t(b) * S + s) * D + d] / rms[s] * gate;
                }
                ref.mixed[size_t(b) * D + d] = 0.25 * mix;
            }
        });
    }
}

// ---- EX1b's FP64 references (experts_test.cpp's glu_reference and down_reference, mirrored) ----
struct meta {
    int32_t              n_u = 0, n_slots = 0;
    std::vector<int32_t> e, mask, slot_base, u_of;

    int32_t slot(int32_t u, int32_t b) const {
        int32_t s = slot_base[size_t(u)];
        for (int32_t bb = 0; bb < b; ++bb) {
            s += (mask[size_t(u)] >> bb) & 1;
        }
        return s;
    }
};

meta make_meta(const int32_t * ids, int32_t rows) {
    meta m;
    for (int32_t p = 0; p < rows * NU; ++p) {
        int32_t u = -1;
        for (int32_t i = 0; i < m.n_u; ++i) {
            if (m.e[size_t(i)] == ids[p]) {
                u = i;
            }
        }
        if (u < 0) {
            u = m.n_u++;
            m.e.push_back(ids[p]);
            m.mask.push_back(0);
        }
        m.mask[size_t(u)] |= 1 << (p / NU);
        m.u_of.push_back(u);
    }
    for (int32_t u = 0; u < m.n_u; ++u) {
        m.slot_base.push_back(m.n_slots);
        for (int32_t b = 0; b < rows; ++b) {
            m.n_slots += (m.mask[size_t(u)] >> b) & 1;
        }
    }
    return m;
}

// the weights' CPU decoders (experts_test.cpp's mapping, in fmt's order)
fn::quant::wfmt to_quant(fn::experts::fmt f) {
    using fn::quant::wfmt;
    static const wfmt q[7] = { wfmt::iq2_xxs, wfmt::iq2_xs, wfmt::iq2_s, wfmt::iq3_xxs,
                               wfmt::iq3_s,   wfmt::q2_0,   wfmt::iq4_nl };  // fmt's order
    return q[static_cast<int32_t>(f)];
}

// rows of k values of FNFL's format (codes, then {d, s} scales a run) to F64
void act_values(const int8_t * codes, const float * scales, int32_t rows, int32_t k, double * out) {
    std::vector<float> v(static_cast<size_t>(k));
    for (int32_t b = 0; b < rows; ++b) {
        fn::quant::act_values_ref(codes + size_t(b) * k, scales + size_t(b) * (k / 32) * 2, k, v.data());
        for (int32_t i = 0; i < k; ++i) {
            out[size_t(b) * k + i] = v[size_t(i)];
        }
    }
}

// one card's routed partial in FP64 from its recorded input (FNFL's format), routing and GLU (the format); glu_rel:
// the F32 GLU against FP64; glu_act: its codes and scales bitwise equal to fn/quant/act_ref on the kernel's own F32 GLU
fn::quant::wfmt dense_quant(fn::dense::fmt f) {
    return f == fn::dense::fmt::iq3_s    ? fn::quant::wfmt::iq3_s
           : f == fn::dense::fmt::iq4_xs ? fn::quant::wfmt::iq4_xs
           : f == fn::dense::fmt::q4_k   ? fn::quant::wfmt::q4_k
                                         : fn::quant::wfmt::iq4_nl;
}

// EX2's FP64 reference of one card: the routed experts and the shared expert (GLU slots n_slots + b), routed is the
// weighted sum with the shared term sg[b] x its down product added
void ex_reference(const p1::pair::facts & fx,
                  int32_t                 l,
                  int32_t                 card,
                  int32_t                 B,
                  const p1::card_record & R,
                  std::vector<double> &   routed,
                  double &                glu_rel,
                  bool &                  glu_act) {
    const fn::experts::synth & sg = *fx.gate_up[l];
    const fn::experts::synth & sd = *fx.down;
    const fn::quant::wfmt      qg = to_quant(sg.f), qd = to_quant(sd.f);
    const meta                 m = make_meta(R.ids[l], B);
    std::vector<double>        xd(size_t(B) * D);
    act_values(R.xc[l], R.xs[l], B, D, xd.data());
    std::vector<double> glu(size_t(m.n_slots) * L, 0.0);
    parallel_for(m.n_u, [&](int32_t u) {
        const uint8_t *    e = sg.expert(fx.map[size_t(m.e[size_t(u)])]);
        std::vector<float> wg(D), wu(D);
        for (int32_t ch = 0; ch < L; ++ch) {
            const int32_t row = card * L + ch;
            fn::quant::decode_row(qg, e + size_t(row) * sg.row_bytes, wg.data(), D);
            fn::quant::decode_row(qg, e + size_t(NFF + row) * sg.row_bytes, wu.data(), D);
            for (int32_t b = 0; b < B; ++b) {
                if (((m.mask[size_t(u)] >> b) & 1) == 0) {
                    continue;
                }
                double g = 0.0, v = 0.0;
                for (int32_t q = 0; q < D; ++q) {
                    g += double(wg[q]) * xd[size_t(b) * D + q];
                    v += double(wu[q]) * xd[size_t(b) * D + q];
                }
                glu[size_t(m.slot(u, b)) * L + ch] = g / (1.0 + std::exp(-g)) * v;
            }
        }
    });
    // the shared expert: card rows i of gate and up are synthetic rows (i + off) % n_synth_rows (pair.h)
    glu.resize(size_t(m.n_slots + B) * L, 0.0);
    {
        const fn::dense::synth & shg = *fx.sh_gate[l];
        const fn::dense::synth & shu = *fx.sh_up[l];
        const int32_t n_sr = fn::dense::n_synth_rows;
        parallel_for(L, [&](int32_t ch) {
            std::vector<float> wg(D), wu(D);
            fn::quant::decode_row(dense_quant(shg.f), shg.row((ch + fx.sh_off[l][card][0]) % n_sr), wg.data(), D);
            fn::quant::decode_row(dense_quant(shu.f), shu.row((ch + fx.sh_off[l][card][1]) % n_sr), wu.data(), D);
            for (int32_t b = 0; b < B; ++b) {
                double g = 0.0, v = 0.0;
                for (int32_t q = 0; q < D; ++q) {
                    g += double(wg[q]) * xd[size_t(b) * D + q];
                    v += double(wu[q]) * xd[size_t(b) * D + q];
                }
                glu[size_t(m.n_slots + b) * L + ch] = g / (1.0 + std::exp(-g)) * v;
            }
        });
    }
    err_acc ge;
    glu_act = true;
    for (int32_t sl = 0; sl < m.n_slots + B; ++sl) {
        for (int32_t ch = 0; ch < L; ++ch) {
            ge.add(R.glu_f32[l][sl * L + ch], glu[size_t(sl) * L + ch]);
        }
        int8_t wc[L];
        float  ws[L / 32 * 2];
        fn::quant::act_quantize_ref(&R.glu_f32[l][sl * L], 1, L, wc, ws);
        glu_act = glu_act && std::memcmp(wc, &R.gluc[l][sl * L], sizeof(wc)) == 0 &&
                  std::memcmp(ws, &R.glus[l][sl * (L / 32) * 2], sizeof(ws)) == 0;
    }
    glu_rel = ge.rel();
    std::vector<double> gv(size_t(m.n_slots + B) * L);
    act_values(R.gluc[l], R.glus[l], m.n_slots + B, L, gv.data());
    std::vector<double> dot(size_t(m.n_slots) * D, 0.0);
    parallel_for(m.n_u, [&](int32_t u) {
        const uint8_t *    e = sd.expert(fx.map[size_t(m.e[size_t(u)])]);
        std::vector<float> wrow(NFF);
        for (int32_t n = 0; n < D; ++n) {
            fn::quant::decode_row(qd, e + size_t(n) * sd.row_bytes, wrow.data(), NFF);
            for (int32_t b = 0; b < B; ++b) {
                if (((m.mask[size_t(u)] >> b) & 1) == 0) {
                    continue;
                }
                const int32_t sl  = m.slot(u, b);
                double        acc = 0.0;
                for (int32_t ch = 0; ch < L; ++ch) {
                    acc += double(wrow[card * L + ch]) * gv[size_t(sl) * L + ch];
                }
                dot[size_t(sl) * D + n] = acc;
            }
        }
    });
    routed.assign(size_t(B) * D, 0.0);
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t n = 0; n < D; ++n) {
            double acc = 0.0;
            for (int32_t q = 0; q < NU; ++q) {
                acc += double(R.wts[l][b * NU + q]) * dot[size_t(m.slot(m.u_of[size_t(b) * NU + q], b)) * D + n];
            }
            routed[size_t(b) * D + n] = acc;
        }
    }
    // the shared down: row n is synthetic row (n + off) % n_synth_rows of the card's 320 columns, scaled by sg[b]
    const fn::dense::synth & shd = *fx.sh_down;
    parallel_for(B, [&](int32_t b) {
        std::vector<float> wrow(L);
        for (int32_t n = 0; n < D; ++n) {
            fn::quant::decode_row(dense_quant(shd.f), shd.row((n + fx.sh_off[l][card][2]) % fn::dense::n_synth_rows),
                                  wrow.data(), L);
            double acc = 0.0;
            for (int32_t ch = 0; ch < L; ++ch) {
                acc += double(wrow[ch]) * gv[size_t(m.n_slots + b) * L + ch];
            }
            routed[size_t(b) * D + n] += double(R.sg[l][b]) * acc;
        }
    });
}

int32_t piece_id(const std::string & name, int32_t layer) {
    for (int32_t p = 0; p < p1::n_piece; ++p) {
        if (name == p1::piece_table[p].name && layer == p1::piece_table[p].layer) {
            return p;
        }
    }
    return -1;
}

// a number from routes.txt.sel.report.json's layer-11 per_row block
bool report_value(const std::string & text, const char * key, double & v) {
    const size_t l11 = text.find("\"11\"");
    const size_t pr  = l11 == std::string::npos ? l11 : text.find("\"per_row\"", l11);
    const size_t k   = pr == std::string::npos ? pr : text.find(std::string("\"") + key + "\"", pr);
    if (k == std::string::npos) {
        return false;
    }
    return sscanf(text.c_str() + text.find(':', k) + 1, "%lf", &v) == 1;
}

struct host_records {
    p1::card_record * r[2] = { nullptr, nullptr };

    ~host_records() {
        for (auto * p : r) {
            cudaFreeHost(p);
        }
    }

    bool alloc() {
        return cudaMallocHost(&r[0], sizeof(p1::card_record)) == cudaSuccess &&
               cudaMallocHost(&r[1], sizeof(p1::card_record)) == cudaSuccess;
    }
};

// the selection a card's attention reads, from its records: the body's words, then its B + 1 byte offsets
std::vector<uint32_t> selection(const p1::card_record & R, int32_t B) {
    const int32_t         bw = fn::qsa::body_words(p1::pair_depth, B);
    std::vector<uint32_t> v(R.body, R.body + bw);
    for (int32_t t = 0; t <= B; ++t) {
        v.push_back(static_cast<uint32_t>(R.off[t]));
    }
    return v;
}

}  // namespace

int fnfl_xp1_test(const fnfl_xp1_test_args & a) {
    checks          ck;
    std::string     err;
    p1::pair        pr;
    p1::pair_config cfg;
    cfg.dev[0]    = a.dev0;
    cfg.dev[1]    = a.dev1;
    cfg.tr        = a.synthetic ? p1::traffic::synthetic : p1::traffic::hx7;
    cfg.synth_dir = a.synth_dir;
    cfg.hx7_dir   = a.hx7_dir;
    cfg.glu_f32   = true;
    if (!pr.init(cfg, err)) {
        printf("xp1-test: init: %s\nxp1-test: FAIL\n", err.c_str());
        return 1;
    }
    const p1::pair::facts & fx  = pr.info();
    const bool              hx7 = !a.synthetic;
    printf("xp1-test: devices %d and %d; traffic %s; depth %d; steps B =", a.dev0, a.dev1, hx7 ? "hx7" : "synthetic",
           p1::pair_depth);
    for (int32_t B : a.steps) {
        printf(" %d", B);
    }
    printf("\n");

    // Amendment 3: the loader against the files (16 decodes) and the layer-11 counts against the report
    {
        std::string e2;
        ck.ok(fx.hx7.n_decode == 1024 && fn::xp1::check_hx7(a.hx7_dir, fx.hx7, 16, e2), "hx7 loader: " + e2);
        FILE *      fp = fopen((a.hx7_dir + "/routes.txt.sel.report.json").c_str(), "r");
        std::string text;
        if (fp != nullptr) {
            char   buf[4096];
            size_t n = 0;
            while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
                text.append(buf, n);
            }
            fclose(fp);
        }
        double     mean = 0.0, mn = 0.0, mx = 0.0;
        const bool have =
            report_value(text, "mean", mean) && report_value(text, "min", mn) && report_value(text, "max", mx);
        double  sum = 0.0;
        int32_t lo = INT_MAX, hi = INT_MIN;
        for (const int32_t c : fx.hx7.counts) {
            sum += c;
            lo = std::min(lo, c);
            hi = std::max(hi, c);
        }
        const double m       = fx.hx7.counts.empty() ? 0.0 : sum / double(fx.hx7.counts.size());
        bool         first16 = true;
        for (int32_t i = 0; i < 16 * p1::hx7_rows && i < int32_t(fx.hx7.counts.size()); ++i) {
            first16 = first16 && fx.hx7.counts[i] >= mn && fx.hx7.counts[i] <= mx;
        }
        printf("xp1-test: hx7 layer-11 cells a row: mean %.4f min %d max %d (report: %.4f, %.0f, %.0f)\n", m, lo, hi,
               mean, mn, mx);
        ck.ok(have && std::fabs(m - mean) < 1e-3 && lo == int32_t(mn) && hi == int32_t(mx) && first16,
              "hx7 counts against routes.txt.sel.report.json");
    }

    host_records hr;
    if (!hr.alloc()) {
        printf("xp1-test: pinned records\nxp1-test: FAIL\n");
        return 1;
    }
    const int32_t owner   = (11 / 4) % 2;
    const int32_t n_steps = static_cast<int32_t>(a.steps.size());

    // per variant, step and card: the residual halves and the selection
    struct kept {
        std::vector<float>    x, inject, out;
        std::vector<uint32_t> sel;
    };

    std::vector<std::vector<std::array<kept, 2>>> keep(p1::n_variant);
    std::vector<bool>                             ran(p1::n_variant, false);

    for (const std::string & vn : a.variants) {
        p1::variant v;
        if (!p1::parse_variant(vn, v)) {
            printf("xp1-test: unknown variant %s\nxp1-test: FAIL\n", vn.c_str());
            return 2;
        }
        const int32_t vi = static_cast<int32_t>(v);
        ran[vi]          = true;
        keep[vi].assign(size_t(n_steps), {});
        if (!pr.reset_exchange(err) || !pr.reset_round(err) || !pr.reset_state(err)) {
            printf("xp1-test: reset: %s\nxp1-test: FAIL\n", err.c_str());
            return 1;
        }
        std::set<std::pair<int32_t, int32_t>> pieces_run;  // (piece, card)
        for (int32_t si = 0; si < n_steps; ++si) {
            const int32_t B = a.steps[size_t(si)];
            std::memset(hr.r[0], 0, sizeof(p1::card_record));
            std::memset(hr.r[1], 0, sizeof(p1::card_record));
            std::vector<p1::op> prog;
            if (!pr.build(v, B, false, prog, err, hr.r) || !pr.run_eager(prog, err)) {
                printf("xp1-test: %s step %d (B = %d): %s\nxp1-test: FAIL\n", vn.c_str(), si, B, err.c_str());
                return 1;
            }
            for (const p1::op & o : prog) {
                const int32_t id = piece_id(o.piece, o.layer);
                if (id >= 0) {
                    pieces_run.insert({ id, o.card });
                }
            }
            const std::string tag = vn + " step " + std::to_string(si) + " B=" + std::to_string(B);
            for (int32_t r = 0; r < 2; ++r) {
                const p1::card_record & R = *hr.r[r];
                ck.ok(R.err == 0, tag + " card " + std::to_string(r) + " error word " + std::to_string(R.err));
                kept & kp = keep[vi][size_t(si)][size_t(r)];
                kp.x.assign(R.x[3], R.x[3] + size_t(B) * S * F);
                kp.inject.assign(R.inject[3], R.inject[3] + size_t(B) * S);
                kp.out.assign(R.out[3], R.out[3] + size_t(B) * F);
                kp.sel = selection(R, B);
            }
            // the selection: the same on both cards (X-SELECT's round trip)
            ck.ok(keep[vi][size_t(si)][0].sel == keep[vi][size_t(si)][1].sel, tag + " selection equal on both cards");
            // X-ROUTE (or V2's router on both cards): the same routing on both cards, the router's rule
            for (int32_t l = 0; l < 2; ++l) {
                bool same = true, rule = true;
                for (int32_t b = 0; b < B; ++b) {
                    for (int32_t q = 0; q < NU; ++q) {
                        const int32_t i    = b * NU + q;
                        same               = same && hr.r[0]->ids[l][i] == hr.r[1]->ids[l][i] &&
                                             hr.r[0]->wts[l][i] == hr.r[1]->wts[l][i];
                        const int32_t d    = (fx.hx7.n_decode > 0 ? si % fx.hx7.n_decode : 0);
                        const int32_t want = hx7 ? fx.hx7.routes[l][(size_t(d) * 3 + b) * NU + q] :
                                                   (37 * q + 13 * b + 7 * (10 + l)) % 512;
                        rule               = rule && hr.r[0]->ids[l][i] == want && hr.r[0]->wts[l][i] == 0.1f;
                    }
                    same = same && hr.r[0]->sg[l][b] == hr.r[1]->sg[l][b];
                    rule = rule && hr.r[0]->sg[l][b] == 0.5f;
                }
                ck.ok(same && rule, tag + " layer " + std::to_string(10 + l) + " routing on both cards");
            }

            if (v != p1::variant::v0 || si >= 3) {
                continue;
            }
            // ---- V0 against the references ----
            // HC2: each sublayer from its recorded inputs
            for (int32_t k = 0; k < 4; ++k) {
                std::vector<float> hres(size_t(B) * S * D), hpost(size_t(B) * S, 0.0f), hx(size_t(B) * D, 0.0f);
                for (int32_t b = 0; b < B; ++b) {
                    for (int32_t s = 0; s < S; ++s) {
                        for (int32_t r = 0; r < 2; ++r) {
                            for (int32_t q = 0; q < F; ++q) {
                                const size_t at = (size_t(b) * S + s) * D + r * F + q;
                                hres[at] = k == 0 ? fx.in0[B][at] : hr.r[r]->x[k - 1][(size_t(b) * S + s) * F + q];
                            }
                        }
                        if (k > 0) {
                            hpost[size_t(b) * S + s] = hr.r[0]->inject[k - 1][b * S + s];
                        }
                    }
                    for (int32_t r = 0; r < 2 && k > 0; ++r) {
                        for (int32_t q = 0; q < F; ++q) {
                            hx[size_t(b) * D + r * F + q] = hr.r[r]->out[k - 1][b * F + q];
                        }
                    }
                }
                if (k > 0) {
                    ck.ok(std::memcmp(hr.r[0]->inject[k - 1], hr.r[1]->inject[k - 1], size_t(B) * S * 4) == 0,
                          tag + " sublayer " + std::to_string(k - 1) + " injection equal on both cards");
                }
                hc_ref ref;
                hc_reference(fx, k, B, hres, hpost, hx, ref);
                err_acc ex, ei, em;
                for (int32_t r = 0; r < 2; ++r) {
                    const p1::card_record & R = *hr.r[r];
                    for (int32_t b = 0; b < B; ++b) {
                        for (int32_t s = 0; s < S; ++s) {
                            for (int32_t q = 0; q < F; ++q) {
                                ex.add(R.x[k][(size_t(b) * S + s) * F + q], ref.x[(size_t(b) * S + s) * D + r * F + q]);
                            }
                            ei.add(R.inject[k][b * S + s], ref.inject[size_t(b) * J + s]);
                        }
                        for (int32_t d = 0; d < D; ++d) {
                            em.add(R.full[k][size_t(b) * D + d], ref.mixed[size_t(b) * D + d]);
                        }
                    }
                }
                char buf[160];
                snprintf(buf, sizeof(buf), " HC sublayer %d: x %.2e inject %.2e mixed %.2e (bound %.0e)", k, ex.rel(),
                         ei.rel(), em.rel(), hc_bound);
                if (si == 2) {
                    printf("xp1-test: V0 B=%d%s\n", B, buf);
                }
                ck.ok(ex.rel() <= hc_bound && ei.rel() <= hc_bound && em.rel() <= hc_bound, tag + buf);
            }
            // X-A: the stand-ins' partials summed, bitwise
            for (int32_t li = 0; li < 2; ++li) {
                const int32_t k   = 2 * li;
                const float   c   = fx.c_piece[static_cast<int32_t>(li == 0 ? p1::piece::gdn_out : p1::piece::qsa_out)];
                int32_t       bad = 0;
                for (int32_t r = 0; r < 2; ++r) {
                    for (int32_t b = 0; b < B; ++b) {
                        for (int32_t q = 0; q < F; ++q) {
                            const int32_t j = r * F + q;
                            float         v2[2];
                            for (int32_t t = 0; t < 2; ++t) {
                                const float * in = li == 0 ? hr.r[t]->recur_out : hr.r[t]->attn_out;
                                v2[t]            = f_out(in, 3072, b, j, c);
                            }
                            const float want = v2[r] + v2[1 - r];
                            bad += std::memcmp(&want, &hr.r[r]->out[k][b * F + q], 4) != 0;
                        }
                    }
                }
                ck.ok(bad == 0, tag + " layer " + std::to_string(10 + li) + " X-A against the stand-ins' function (" +
                                    std::to_string(bad) + " words differ)");
            }
            // X-F: EX2's FP64 reference (the routed and the shared expert); C2's quantize bitwise
            for (int32_t li = 0; li < 2; ++li) {
                const int32_t k = 2 * li + 1;
                std::vector<double> part[2];
                for (int32_t r = 0; r < 2; ++r) {
                    const p1::card_record & R = *hr.r[r];
                    std::vector<int8_t>     wc(size_t(B) * D);
                    std::vector<float>      ws(size_t(B) * (D / 32) * 2);
                    fn::quant::act_quantize_ref(R.full[k], B, D, wc.data(), ws.data());
                    ck.ok(std::memcmp(wc.data(), R.xc[li], wc.size()) == 0 &&
                              std::memcmp(ws.data(), R.xs[li], ws.size() * 4) == 0,
                          tag + " card " + std::to_string(r) + " layer " + std::to_string(10 + li) +
                              " C2's quantize against fn/quant/act_ref");
                    double glu_rel = 0.0;
                    bool   glu_act = false;
                    ex_reference(fx, li, r, B, R, part[r], glu_rel, glu_act);
                    ck.ok(glu_rel <= ex_bound && glu_act, tag + " card " + std::to_string(r) + " layer " +
                                                              std::to_string(10 + li) + " EX2 GLU against FP64 " +
                                                              std::to_string(glu_rel));
                }
                err_acc ef;
                for (int32_t r = 0; r < 2; ++r) {
                    for (int32_t b = 0; b < B; ++b) {
                        for (int32_t q = 0; q < F; ++q) {
                            const size_t at = size_t(b) * D + r * F + q;
                            ef.add(hr.r[r]->out[k][b * F + q], part[0][at] + part[1][at]);
                        }
                    }
                }
                if (si == 2) {
                    printf("xp1-test: V0 B=%d layer %d X-F against EX2's FP64 (routed and shared): %.2e "
                           "(bound %.0e)\n",
                           B, 10 + li, ef.rel(), ex_bound);
                }
                ck.ok(ef.rel() <= ex_bound,
                      tag + " layer " + std::to_string(10 + li) + " X-F " + std::to_string(ef.rel()));
            }
            // X-SELECT's round trip: the peer's body and offsets equal the owner's QA1b output, and the offsets are
            // the host's
            {
                bool off_ok = true;
                for (int32_t t = 0; t <= B; ++t) {
                    off_ok = off_ok && hr.r[owner]->off[t] == fn::qsa::body_offset(p1::pair_depth, t);
                }
                ck.ok(selection(*hr.r[owner], B) == selection(*hr.r[1 - owner], B) && off_ok,
                      tag + " X-SELECT round trip");
            }
        }
        // every stand-in's weights streamed: the XOR of its sink over the blocks
        {
            std::string bad;
            for (int32_t r = 0; r < 2; ++r) {
                const p1::card_record & R = *hr.r[r];
                for (const auto & pc : pieces_run) {
                    if (pc.second != r) {
                        continue;
                    }
                    uint32_t xr[4] = { 0u, 0u, 0u, 0u };
                    for (int32_t blk = 0; blk < p1::si_blocks; ++blk) {
                        for (int32_t q = 0; q < 4; ++q) {
                            xr[q] ^= R.sink[(size_t(pc.first) * p1::si_blocks + blk) * p1::sink_block_words + q];
                        }
                    }
                    if (std::memcmp(xr, fx.w_xor[pc.first][r], 16) != 0) {
                        bad += std::string(" ") + p1::piece_table[pc.first].name + "/" + std::to_string(r);
                    }
                }
            }
            ck.ok(bad.empty(), vn + " stand-in weight XORs" + bad);
        }
        // a repeated B in the sequence gives the same bits only with the same traffic: synthetic routes repeat; HX7's
        // decode differs per step, so the determinism check is the graph's
        if (!a.no_graph) {
            // the graph at B = 3 replays the step whose decode index matches
            int32_t si3 = -1;
            for (int32_t si = 0; si < n_steps; ++si) {
                si3 = si3 < 0 && a.steps[size_t(si)] == 3 ? si : si3;
            }
            std::vector<p1::op> prog;
            if (si3 >= 0) {
                bool     ok = pr.reset_exchange(err) && pr.build(v, 3, true, prog, err) && pr.capture(prog, err);
                float    ms[2];
                uint64_t st[2];
                for (int32_t rep = 0; rep < 2 && ok; ++rep) {
                    ok = pr.set_round(si3, err) && pr.replay(ms, st, err) && pr.read_final(3, hr.r, err);
                    for (int32_t r = 0; r < 2 && ok; ++r) {
                        const kept &                kp  = keep[vi][size_t(si3)][size_t(r)];
                        const p1::card_record &     R   = *hr.r[r];
                        const std::vector<uint32_t> sel = selection(R, 3);
                        const bool same = std::memcmp(kp.x.data(), R.x[3], kp.x.size() * 4) == 0 &&
                                          std::memcmp(kp.inject.data(), R.inject[3], kp.inject.size() * 4) == 0 &&
                                          std::memcmp(kp.out.data(), R.out[3], kp.out.size() * 4) == 0 &&
                                          kp.sel == sel && R.err == 0;
                        ck.ok(same, vn + " graph replay " + std::to_string(rep) + " card " + std::to_string(r) +
                                        " equals the eager step");
                    }
                }
                pr.drop_graphs();
                if (!ok) {
                    printf("xp1-test: %s graph: %s\n", vn.c_str(), err.c_str());
                    ck.ok(false, vn + " graph");
                }
                ok = pr.reset_exchange(err);
            }
        }
        printf("xp1-test: %s done (%d checks so far, %d failed)\n", vn.c_str(), ck.n, ck.bad);
    }
    // every variant against V0, step by step: each card's residual half after layer 11 and the selection
    if (ran[0]) {
        for (int32_t vi = 1; vi < p1::n_variant; ++vi) {
            if (!ran[size_t(vi)]) {
                continue;
            }
            int32_t diff = 0;
            for (int32_t si = 0; si < n_steps; ++si) {
                for (int32_t r = 0; r < 2; ++r) {
                    const kept & a0 = keep[0][size_t(si)][size_t(r)];
                    const kept & av = keep[size_t(vi)][size_t(si)][size_t(r)];
                    diff += a0.x != av.x || a0.inject != av.inject || a0.out != av.out || a0.sel != av.sel;
                }
            }
            ck.ok(diff == 0, std::string(p1::variant_name(static_cast<p1::variant>(vi))) +
                                 " bitwise equal to V0 (step-cards differing: " + std::to_string(diff) + ")");
            printf("xp1-test: %s against V0: %s\n", p1::variant_name(static_cast<p1::variant>(vi)),
                   diff == 0 ? "bitwise equal on every step and card" : "DIFFERS");
        }
    }
    printf("xp1-test: %d checks, %d failed\n", ck.n, ck.bad);
    printf("xp1-test: %s\n", ck.bad == 0 ? "PASS" : "FAIL");
    return ck.bad == 0 ? 0 : 1;
}
