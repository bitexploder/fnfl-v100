#include "fn/kernels/dense/dense_prompt_test.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/dense/dense_ladder.h"
#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/experts/experts.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/tools/prompt_test_util.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fn::dense {
namespace {

bool cuda_ok(cudaError_t e, const char * what) {
    return fn::test::cuda_ok(e, "pd-test", what);
}

// TFC: the prompt tests' shared device buffer, generator, work split and verdict (fn/tools/prompt_test_util.h)
using fn::ptest::dbuf;
using fn::ptest::parallel_for;
using fn::ptest::rng;
using fn::ptest::verdict;

quant::wfmt wf(fmt f) {
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
        default:
            return quant::wfmt::f32;
    }
}

experts::fmt ef(fmt f) {
    return f == fmt::q2_0 ? experts::fmt::q2_0 : f == fmt::q8_0 ? experts::fmt::q8_0 : experts::fmt::iq4_nl;  // TTTS2
}

std::string fmt_e(double v) {
    return fn::ptest::e3(v);
}

// a weight matrix of a case: n GGUF rows of k (synthetic rows i % 1024), its decoded values and its device copy in the
// resident layout
struct matrix {
    fmt                  f;
    pd_layout            lay;
    int32_t              n = 0, k = 0;
    std::vector<float>   deq;  // [n][k]
    std::vector<uint8_t> packed;
};

bool make_matrix(fmt f, pd_layout lay, int32_t n, int32_t k, int32_t off, matrix & m) {
    m.f   = f;
    m.lay = lay;
    m.n   = n;
    m.k   = k;
    synth s;
    if (!make_synth(f, k, s)) {
        fprintf(stderr, "pd-test: no synthetic rows for %s at K %d\n", fmt_name(f), k);
        return false;
    }
    m.deq.assign(size_t(n) * k, 0.0f);
    std::atomic<bool> ok{ true };
    parallel_for(n, [&](int32_t i) {
        if (!quant::decode_row(wf(f), s.row((i + off) % n_synth_rows), m.deq.data() + size_t(i) * k, k)) {
            ok = false;
        }
    });
    if (!ok) {
        return false;
    }
    if (lay == pd_layout::dp2) {
        return pack_synth(s, n, off, n_synth_rows, m.packed);
    }
    // EX1's down: the 2,560 GGUF rows of the card's 320 columns, then pack_down
    std::vector<uint8_t> rows(size_t(n) * s.row_bytes);
    for (int32_t i = 0; i < n; ++i) {
        std::memcpy(rows.data() + size_t(i) * s.row_bytes, s.row((i + off) % n_synth_rows), s.row_bytes);
    }
    m.packed.assign(size_t(experts::down_bytes(ef(f))), 0);
    return experts::pack_down(ef(f), rows.data(), s.row_bytes, m.packed.data());
}

// FP16 activations N(0, sd), as stored (bits) and as their exact values
void make_act(int64_t count, double sd, uint64_t seed, std::vector<uint16_t> & bits, std::vector<float> & vals) {
    rng g(seed);
    bits.resize(static_cast<size_t>(count));
    vals.resize(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        bits[size_t(i)] = quant::f32_to_f16(static_cast<float>(g.normal() * sd));
        vals[size_t(i)] = quant::f16_to_f32(bits[size_t(i)]);
    }
}

// max over rows of max |out - ref| / max |ref| (FP64 reference ref[r][n] = sum_k x[r][k] w[n][k]); NaN: inf
double row_error(const std::vector<float> & out, const std::vector<double> & ref, int32_t rows, int32_t n) {
    double worst = 0.0;
    for (int32_t r = 0; r < rows; ++r) {
        double e = 0.0, mref = 0.0;
        for (int32_t j = 0; j < n; ++j) {
            const double o = out[size_t(r) * n + j], f = ref[size_t(r) * n + j];
            if (!std::isfinite(o)) {
                return INFINITY;
            }
            e    = std::max(e, std::fabs(o - f));
            mref = std::max(mref, std::fabs(f));
        }
        worst = std::max(worst, mref > 0 ? e / mref : e);
    }
    return worst;
}

void reference(const std::vector<float> & x,
               int32_t                    rows,
               int32_t                    k,
               const std::vector<float> & w,
               int32_t                    n,
               const std::vector<float> * scale,
               std::vector<double> &      ref) {
    ref.assign(size_t(rows) * n, 0.0);
    parallel_for(rows, [&](int32_t r) {
        for (int32_t j = 0; j < n; ++j) {
            double        s  = 0.0;
            const float * xr = x.data() + size_t(r) * k;
            const float * wr = w.data() + size_t(j) * k;
            for (int32_t i = 0; i < k; ++i) {
                s += double(xr[i]) * double(wr[i]);
            }
            ref[size_t(r) * n + j] = s * (scale != nullptr ? double((*scale)[size_t(r)]) : 1.0);
        }
    });
}

// The codes of every weight against fn/quant's decoder: f x code - m in F32 must be its value bitwise; the distinct
// codes seen (domain: every code of the format must appear, as the synthetic rows give; a real matrix's are reported)
bool check_codes(const matrix & m, const dbuf & w, const dbuf & grids, verdict & v, bool domain = true) {
    const size_t nk = size_t(m.n) * m.k;
    dbuf         c(nk * 4), s(nk * 4), mm(nk * 4);
    if (!c.poison() || !s.poison() || !mm.poison() ||
        !pd_codes(m.f, m.lay, m.k, m.n, w.as<uint8_t>(), grids.as<uint8_t>(), c.as<float>(), s.as<float>(),
                  mm.as<float>(), fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "pd_codes")) {
        v.put(std::string("codes ") + fmt_name(m.f), false, "launch failed");
        return false;
    }
    std::vector<float> code(nk), sc(nk), mn(nk);
    if (!c.get(code.data(), nk * 4) || !s.get(sc.data(), nk * 4) || !mm.get(mn.data(), nk * 4)) {
        return false;
    }
    int64_t           bad = 0, first = -1;
    std::set<int32_t> seen;
    for (size_t i = 0; i < nk; ++i) {
        // f x code rounded once (an exact product in FP64, then to F32), then - m in F32, as the decoder computes
        const float fc  = static_cast<float>(double(sc[i]) * double(code[i]));
        const float val = fc - mn[i];
        uint32_t    a = 0, b = 0;
        std::memcpy(&a, &val, 4);
        std::memcpy(&b, &m.deq[i], 4);
        const bool integral = std::isfinite(code[i]) && code[i] == std::floor(code[i]);
        // a zero scale has two signed zeros' ways to equal a decoded zero
        if (!integral || (a != b && !(val == 0.0f && m.deq[i] == 0.0f))) {
            ++bad;
            if (first < 0) {
                first = static_cast<int64_t>(i);
            }
        }
        if (integral) {
            seen.insert(static_cast<int32_t>(code[i]));
        }
    }
    // the domains: Q4_K 0..15, Q5_K 0..31, Q6_K -32..31, IQ4 the 16 table values, IQ3_S the grid's 8 odd magnitudes
    // with both signs, Q8_0 -128..127, Q2_0 -1..2
    const int32_t want = m.f == fmt::q4_k ? 16 :
                         m.f == fmt::q5_k ? 32 :
                         m.f == fmt::q6_k ? 64 :
                         m.f == fmt::q8_0 ? 256 :
                         m.f == fmt::q2_0 ? 4 :
                                            16;
    const int32_t lo = *seen.begin(), hi = *seen.rbegin();
    std::string   what = std::to_string(nk) + " weights, " + std::to_string(bad) + " differ";
    if (first >= 0) {
        what += " (first: row " + std::to_string(first / m.k) + " k " + std::to_string(first % m.k) + ": code " +
                std::to_string(code[size_t(first)]) + " f " + fmt_e(sc[size_t(first)]) + " m " +
                fmt_e(mn[size_t(first)]) + " -> decoder " + fmt_e(m.deq[size_t(first)]) + ")";
    }
    what += "; " + std::to_string(seen.size()) + " distinct codes of " + std::to_string(want) + " in [" +
            std::to_string(lo) + ", " + std::to_string(hi) + "]";
    const bool ok = bad == 0 && (!domain || static_cast<int32_t>(seen.size()) == want);
    v.put(std::string("codes ") + fmt_name(m.f) + (m.lay == pd_layout::ex1_down ? " (EX1 down)" : ""), ok, what);
    return ok;
}

// one product case: rows of activations (one segment), the bound, and (rerun) a second launch compared bitwise
bool check_product(const matrix & m,
                   const dbuf &   w,
                   const dbuf &   grids,
                   int32_t        rows,
                   uint64_t       seed,
                   bool           scaled,
                   bool           rerun,
                   verdict &      v) {
    std::vector<uint16_t> xb;
    std::vector<float>    xv;
    make_act(int64_t(rows) * m.k, 1.0, seed, xb, xv);
    std::vector<float> sc;
    if (scaled) {
        rng g(seed ^ 0x5c5c);
        sc.resize(static_cast<size_t>(rows));
        for (float & s : sc) {
            s = static_cast<float>(g.uniform());
        }
    }
    dbuf    x(xb.size() * 2), o(size_t(rows) * m.n * 4), s(std::max<size_t>(sc.size(), 1) * 4);
    pd_args a;
    a.f         = m.f;
    a.lay       = m.lay;
    a.k         = m.k;
    a.n         = m.n;
    a.rows      = rows;
    a.w         = w.as<uint8_t>();
    a.grids     = grids.as<uint8_t>();
    a.x.a0      = x.as<uint16_t>();
    a.x.ld0     = m.k;
    a.x.k_split = m.k;
    a.out       = o.as<float>();
    a.ld_out    = m.n;
    a.row_scale = scaled ? s.as<float>() : nullptr;
    std::vector<float> out(size_t(rows) * m.n), out2;
    const std::string  name = std::string("product ") + fmt_name(m.f) + (m.lay == pd_layout::ex1_down ? " (EX1)" : "") +
                              " " + std::to_string(rows) + " x " + std::to_string(m.k) + " -> " + std::to_string(m.n) +
                              (scaled ? " scaled" : "");
    if (!x.put(xb.data(), xb.size() * 2) || (scaled && !s.put(sc.data(), sc.size() * 4)) || !o.poison() ||
        !pd_product(a, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "pd_product") ||
        !o.get(out.data(), out.size() * 4)) {
        v.put(name, false, "launch failed");
        return false;
    }
    bool same = true;
    if (rerun) {
        out2.resize(out.size());
        same = o.poison() && pd_product(a, fn::ptest::test_stream()) &&
               cuda_ok(cudaDeviceSynchronize(), "pd_product") && o.get(out2.data(), out2.size() * 4) &&
               std::memcmp(out.data(), out2.data(), out.size() * 4) == 0;
    }
    std::vector<double> ref;
    reference(xv, rows, m.k, m.deq, m.n, scaled ? &sc : nullptr, ref);
    const double e = row_error(out, ref, rows, m.n);
    v.put(name, e <= 1e-5 && same,
          "max err / max ref " + fmt_e(e) + " (bound 1e-5)" +
              (rerun ? same ? "; rerun bitwise" : "; RERUN DIFFERS" : ""));
    return e <= 1e-5 && same;
}

// eh_proj's operand: positions p of e [p][k_split] and h [p][4][k - k_split]; row r = 4 p + s reads e[p] then h[p][s]
bool check_two_segment(const matrix & m, const dbuf & w, const dbuf & grids, int32_t positions, verdict & v) {
    const int32_t         ks = m.k / 2, rows = 4 * positions;
    std::vector<uint16_t> eb, hb;
    std::vector<float>    ev, hv;
    make_act(int64_t(positions) * ks, 1.0, 0xe0e0, eb, ev);
    make_act(int64_t(rows) * (m.k - ks), 1.0, 0xa1a1, hb, hv);
    std::vector<float> xv(size_t(rows) * m.k);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t i = 0; i < m.k; ++i) {
            xv[size_t(r) * m.k + i] = i < ks ? ev[size_t(r / 4) * ks + i] : hv[size_t(r) * (m.k - ks) + i - ks];
        }
    }
    dbuf    e(eb.size() * 2), h(hb.size() * 2), o(size_t(rows) * m.n * 4);
    pd_args a;
    a.f         = m.f;
    a.lay       = m.lay;
    a.k         = m.k;
    a.n         = m.n;
    a.rows      = rows;
    a.w         = w.as<uint8_t>();
    a.grids     = grids.as<uint8_t>();
    a.x.a0      = e.as<uint16_t>();
    a.x.ld0     = ks;
    a.x.shift0  = 2;
    a.x.a1      = h.as<uint16_t>();
    a.x.ld1     = m.k - ks;
    a.x.shift1  = 0;
    a.x.k_split = ks;
    a.out       = o.as<float>();
    a.ld_out    = m.n;
    std::vector<float> out(size_t(rows) * m.n);
    const std::string  name = std::string("two-segment ") + fmt_name(m.f) + " " + std::to_string(positions) +
                              " positions x 4 streams, K " + std::to_string(m.k) + " split " + std::to_string(ks);
    if (!e.put(eb.data(), eb.size() * 2) || !h.put(hb.data(), hb.size() * 2) || !o.poison() ||
        !pd_product(a, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "pd_product") ||
        !o.get(out.data(), out.size() * 4)) {
        v.put(name, false, "launch failed");
        return false;
    }
    std::vector<double> ref;
    reference(xv, rows, m.k, m.deq, m.n, nullptr, ref);
    const double err = row_error(out, ref, rows, m.n);
    v.put(name, err <= 1e-5, "max err / max ref " + fmt_e(err) + " (bound 1e-5)");
    return err <= 1e-5;
}

// PB3: the conversion bitwise against fn/quant's decoder then FP16 rounding (round to nearest even), and pd_tc over
// it: rows of activations (one segment, or eh_proj's two when positions > 0), the output split into segs column
// segments, against FP64 on the FP16 weights at a loose tolerance (Amendment 2: layout, indexing and range errors)
bool g_san = false;  // pd-test san: the ladder at two split counts

// APF2: pl_tc of q (its pt_args' segments outs, at ends) on every tile and split count (san: 1 and the most): each
// product's error err(output) (against FP64 of the same FP16 weights) within the loose tolerance (1e-2) and at most
// twice the converted route's (e_conv), a rerun bitwise, at one split count every tile bitwise the others (each
// split's K in the same order); with one segment the 8-row tile's FP16 output the F32 output rounded, the error word
// clear. Reported: split 1 against the converted route's output conv (bitwise or not)
bool ladder_sweep(const pq_args &                                          q,
                  const std::vector<int32_t> &                             ends,
                  const std::vector<std::unique_ptr<dbuf>> &               outs,
                  const std::function<double(const std::vector<float> &)> & err,
                  double                                                   e_conv,
                  const std::vector<float> &                               conv,
                  const std::string &                                      name,
                  verdict &                                                v) {
    const int32_t rows = q.t.rows, n = q.t.n, segs = static_cast<int32_t>(ends.size());
    const int32_t steps = q.t.k / 32, max_s = std::min(pl_max_splits, steps);
    int64_t       part = 0;
    int32_t       cnt  = 0;
    for (int32_t t = 0; t < pl_n_tiles; ++t) {
        part = std::max(part, lad_split_floats(rows, n, t, max_s));
        cnt  = std::max(cnt, lad_tile_count(rows, n, t));
    }
    dbuf  wsp(size_t(std::max<int64_t>(part, 1)) * 4), wsc(size_t(cnt) * 4);
    pl_ws ws;
    ws.part        = wsp.as<float>();
    ws.part_floats = part;
    ws.count       = wsc.as<int32_t>();
    ws.count_len   = cnt;
    if (!cuda_ok(cudaMemset(wsc.p, 0, wsc.n), "the ladder's counters")) {
        return false;
    }
    const auto run_pl = [&](const pq_args & qq, int32_t t, int32_t sp, std::vector<float> & o) {
        for (const auto & b : outs) {
            if (!b->poison()) {
                return false;
            }
        }
        if (!pl_tc(qq, t, sp, ws, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "pl_tc")) {
            return false;
        }
        o.assign(size_t(rows) * n, 0.0f);
        int32_t st = 0;
        for (int32_t i = 0; i < segs; ++i) {
            const int64_t      ld = qq.t.seg[i].ld;
            std::vector<float> sg(size_t(rows) * size_t(ld));
            if (!outs[size_t(i)]->get(sg.data(), sg.size() * 4)) {
                return false;
            }
            for (int32_t r = 0; r < rows; ++r) {
                for (int32_t j = st; j < ends[size_t(i)]; ++j) {
                    o[size_t(r) * n + j] = sg[size_t(r) * size_t(ld) + size_t(j - st)];
                }
            }
            st = ends[size_t(i)];
        }
        return true;
    };
    const auto tile_name = [](int32_t t, int32_t sp) {
        return "t" + std::to_string(pl_tiles[t].bm) + "x" + std::to_string(pl_tiles[t].bn) + " s" + std::to_string(sp);
    };
    std::vector<int32_t> sl;
    for (int32_t sp = 1; sp <= max_s; ++sp) {
        if (!g_san || sp == 1 || sp == max_s) {
            sl.push_back(sp);
        }
    }
    bool        bound_ok = true, rerun_ok = true, across = true, conv_same = true;
    double      worst = 0.0;
    std::string worst_at, fails;
    for (const int32_t sp : sl) {
        std::vector<float> first;
        for (int32_t t = 0; t < pl_n_tiles; ++t) {
            std::vector<float> o1, o2;
            if (!run_pl(q, t, sp, o1) || !run_pl(q, t, sp, o2)) {
                v.put(name + " ladder", false, "pl_tc launch failed at " + tile_name(t, sp));
                return false;
            }
            const double e = err(o1);
            const bool   r = std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0;
            const bool   b = e <= 1e-2 && e <= 2.0 * e_conv + 1e-6;
            if (e > worst) {
                worst    = e;
                worst_at = tile_name(t, sp);
            }
            if (!b || !r) {
                fails += " " + tile_name(t, sp) + (b ? "" : " " + fmt_e(e)) + (r ? "" : " rerun");
            }
            bound_ok = bound_ok && b;
            rerun_ok = rerun_ok && r;
            if (t == 0) {
                first     = o1;
                conv_same = conv_same && (sp != 1 || std::memcmp(o1.data(), conv.data(), o1.size() * 4) == 0);
            } else {
                across = across && std::memcmp(o1.data(), first.data(), o1.size() * 4) == 0;
            }
        }
    }
    bool        l16 = true;
    std::string l16_note;
    // the FP16 output (out16) is pd_tc's contract with one segment and no column scale (BF16 weights carry one)
    if (segs == 1 && q.t.col_scale == nullptr) {
        const int64_t  ld16 = n + 8;
        dbuf           o16(size_t(rows) * size_t(ld16) * 2), ew(4);
        const uint32_t zero = 0;
        pq_args        q16  = q;
        q16.t.out16         = o16.as<uint16_t>();
        q16.t.ld16          = ld16;
        q16.t.err           = ew.as<uint32_t>();
        q16.t.err_bit       = 1u << 26;
        std::vector<float>    f32;
        std::vector<uint16_t> got(size_t(rows) * size_t(ld16));
        uint32_t              ev_word = 1;
        if (!run_pl(q, 0, sl.back(), f32) || !o16.poison() || !ew.put(&zero, 4) ||
            !pl_tc(q16, 0, sl.back(), ws, fn::ptest::test_stream()) ||
            !cuda_ok(cudaDeviceSynchronize(), "pl_tc out16") || !o16.get(got.data(), got.size() * 2) ||
            !ew.get(&ev_word, 4)) {
            v.put(name + " ladder", false, "pl_tc out16 launch failed");
            return false;
        }
        int64_t diff = 0;
        for (int32_t r = 0; r < rows; ++r) {
            for (int32_t j = 0; j < n; ++j) {
                diff += got[size_t(r) * size_t(ld16) + size_t(j)] != quant::f32_to_f16(f32[size_t(r) * n + j]);
            }
        }
        l16      = diff == 0 && ev_word == 0;
        l16_note = "; FP16 output " + std::to_string(diff) + " differ, error word " + std::to_string(ev_word);
    }
    const bool ok = bound_ok && rerun_ok && across && l16;
    v.put(name + " ladder", ok,
          std::to_string(pl_n_tiles) + " tiles x " + std::to_string(sl.size()) + " split counts: worst error " +
              fmt_e(worst) + " (" + worst_at + "), the converted route's " + fmt_e(e_conv) +
              " (bound 1e-2 and 2x the converted route's)" + (rerun_ok ? "; reruns bitwise" : "; RERUN DIFFERS") +
              (across ? "; at each split count every tile bitwise the others" : "; TILES DIFFER at a split count") +
              (conv_same ? "; split 1 bitwise the converted route" : "; split 1 differs from the converted route") +
              l16_note + (fails.empty() ? "" : "; failing:" + fails));
    return ok;
}

bool check_tc(const matrix & m,
              const dbuf &   w,
              const dbuf &   grids,
              int32_t        rows,
              int32_t        positions,
              int32_t        segs,
              bool           scaled,
              uint64_t       seed,
              verdict &      v) {
    const size_t nk = size_t(m.n) * m.k;
    dbuf         wt(nk * 2);
    std::string  name = std::string("tc ") + fmt_name(m.f) + (m.lay == pd_layout::ex1_down ? " (EX1)" : "") + " " +
                        std::to_string(positions > 0 ? 4 * positions : rows) + " x " + std::to_string(m.k) + " -> " +
                        std::to_string(m.n) + (positions > 0 ? " two-segment" : "") +
                        (segs > 1 ? " in " + std::to_string(segs) + " segments" : "") + (scaled ? " scaled" : "");
    pd_args      c;
    c.f     = m.f;
    c.lay   = m.lay;
    c.k     = m.k;
    c.n     = m.n;
    c.w     = w.as<uint8_t>();
    c.grids = grids.as<uint8_t>();
    std::vector<uint16_t> w16(nk);
    if (!wt.poison() || !pd_convert(c, wt.as<uint16_t>(), fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "pd_convert") || !wt.get(w16.data(), nk * 2)) {
        v.put(name, false, "conversion launch failed");
        return false;
    }
    int64_t            bad = 0;
    std::vector<float> wv(nk);
    for (size_t i = 0; i < nk; ++i) {
        const uint16_t want = quant::f32_to_f16(m.deq[i]);
        // a decoded zero may carry either sign
        if (w16[i] != want && !((w16[i] & 0x7fffu) == 0 && (want & 0x7fffu) == 0)) {
            ++bad;
        }
        wv[i] = quant::f16_to_f32(w16[i]);
    }
    const int32_t         r_all = positions > 0 ? 4 * positions : rows;
    const int32_t         ks    = positions > 0 ? m.k / 2 : m.k;
    std::vector<uint16_t> eb, hb;
    std::vector<float>    ev, hv, xv(size_t(r_all) * m.k);
    make_act(int64_t(positions > 0 ? positions : rows) * ks, 1.0, seed, eb, ev);
    if (positions > 0) {
        make_act(int64_t(r_all) * (m.k - ks), 1.0, seed ^ 0xa1a1, hb, hv);
    }
    for (int32_t r = 0; r < r_all; ++r) {
        for (int32_t i = 0; i < m.k; ++i) {
            xv[size_t(r) * m.k + i] = positions == 0 ? ev[size_t(r) * m.k + i] :
                                      i < ks         ? ev[size_t(r / 4) * ks + i] :
                                                       hv[size_t(r) * (m.k - ks) + i - ks];
        }
    }
    std::vector<float> sc;
    if (scaled) {
        rng g(seed ^ 0x5c5c);
        sc.resize(static_cast<size_t>(r_all));
        for (float & f : sc) {
            f = static_cast<float>(g.uniform());
        }
    }
    // segment ends: n split in segs parts at multiples of 8; each segment its own buffer with a padded stride
    std::vector<int32_t> ends;
    for (int32_t i = 1; i <= segs; ++i) {
        ends.push_back(i == segs ? m.n : (m.n * i / segs) / 8 * 8);
    }
    std::vector<std::unique_ptr<dbuf>> outs;
    dbuf    e(eb.size() * 2), h(std::max<size_t>(hb.size(), 1) * 2), s(std::max<size_t>(sc.size(), 1) * 4);
    pt_args a;
    a.w           = wt.as<uint16_t>();
    a.n           = m.n;
    a.k           = m.k;
    a.rows        = r_all;
    a.x.a0        = e.as<uint16_t>();
    a.x.ld0       = ks;
    a.x.shift0    = positions > 0 ? 2 : 0;
    a.x.a1        = positions > 0 ? h.as<uint16_t>() : nullptr;
    a.x.ld1       = m.k - ks;
    a.x.k_split   = ks;
    a.n_seg       = segs;
    a.row_scale   = scaled ? s.as<float>() : nullptr;
    int32_t start = 0;
    for (int32_t i = 0; i < segs; ++i) {
        const int64_t ld = ends[size_t(i)] - start + 8;
        outs.push_back(std::make_unique<dbuf>(size_t(r_all) * size_t(ld) * 4));
        a.seg[i].out = outs.back()->as<float>();
        a.seg[i].ld  = ld;
        a.seg[i].end = ends[size_t(i)];
        start        = ends[size_t(i)];
    }
    if (!e.put(eb.data(), eb.size() * 2) || (positions > 0 && !h.put(hb.data(), hb.size() * 2)) ||
        (scaled && !s.put(sc.data(), sc.size() * 4))) {
        return false;
    }
    std::vector<float> out(size_t(r_all) * m.n), out2;
    const auto         run = [&](std::vector<float> & o) {
        for (const auto & b : outs) {
            if (!b->poison()) {
                return false;
            }
        }
        if (!pd_tc(a, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "pd_tc")) {
            return false;
        }
        o.resize(size_t(r_all) * m.n);
        int32_t st = 0;
        for (int32_t i = 0; i < segs; ++i) {
            const int64_t      ld = a.seg[i].ld;
            std::vector<float> seg(size_t(r_all) * size_t(ld));
            if (!outs[size_t(i)]->get(seg.data(), seg.size() * 4)) {
                return false;
            }
            for (int32_t r = 0; r < r_all; ++r) {
                for (int32_t j = st; j < ends[size_t(i)]; ++j) {
                    o[size_t(r) * m.n + j] = seg[size_t(r) * size_t(ld) + size_t(j - st)];
                }
            }
            st = ends[size_t(i)];
        }
        return true;
    };
    if (!run(out) || !run(out2)) {
        v.put(name, false, "launch failed");
        return false;
    }
    const bool          same = std::memcmp(out.data(), out2.data(), out.size() * 4) == 0;
    // MUP1 piece 2: at most 64 rows run on the 64 x 64 tile; the 128 x 128 tile's outputs must equal them bitwise
    bool                tiles = true;
    std::string         tiles_note;
    if (r_all <= 64) {
        std::vector<float> big;
        a.big_tile     = true;
        const bool ran = run(big);
        a.big_tile     = false;
        if (!ran) {
            v.put(name, false, "128 x 128 tile launch failed");
            return false;
        }
        tiles = std::memcmp(out.data(), big.data(), out.size() * 4) == 0;
        tiles_note =
            tiles ? "; the 64 x 64 tile bitwise the 128 x 128 tile's" : "; 64 x 64 TILE DIFFERS from 128 x 128";
    }
    // PB3: one segment also written as FP16 (out16, the P payload): bitwise the F32 output rounded to nearest even,
    // the error word clear (every value is in range)
    bool p16 = true;
    std::string p16_note;
    if (segs == 1) {
        const int64_t ld16 = m.n + 8;
        dbuf o16(size_t(r_all) * size_t(ld16) * 2), ew(4);
        const uint32_t zero = 0;
        pt_args b = a;
        b.out16 = o16.as<uint16_t>();
        b.ld16 = ld16;
        b.err = ew.as<uint32_t>();
        b.err_bit = 1u << 26;
        std::vector<uint16_t> got(size_t(r_all) * size_t(ld16));
        uint32_t ev_word = 1;
        if (!o16.poison() || !ew.put(&zero, 4) || !pd_tc(b, fn::ptest::test_stream()) ||
            !cuda_ok(cudaDeviceSynchronize(), "pd_tc out16") || !o16.get(got.data(), got.size() * 2) ||
            !ew.get(&ev_word, 4)) {
            v.put(name, false, "out16 launch failed");
            return false;
        }
        int64_t diff = 0;
        for (int32_t r = 0; r < r_all; ++r) {
            for (int32_t j = 0; j < m.n; ++j) {
                diff += got[size_t(r) * size_t(ld16) + size_t(j)] != quant::f32_to_f16(out[size_t(r) * m.n + j]);
            }
        }
        p16 = diff == 0 && ev_word == 0;
        p16_note = "; FP16 output " + std::to_string(diff) + " differ from the F32 output rounded, error word " +
                   std::to_string(ev_word);
    }
    // APF1 Part 4: the fused route (pq_tc) bitwise the converted route, from the packed matrix as one source and (DP2,
    // N of at least 96) as three sources over its column tiles; with one segment its FP16 output too
    bool        fused = true;
    std::string fused_note;
    {
        pq_args q;
        q.f     = m.f;
        q.lay   = m.lay;
        q.grids = grids.as<uint8_t>();
        q.t     = a;
        q.t.w   = nullptr;
        const int64_t tile_bytes =
            m.lay == pd_layout::dp2 ? int64_t(m.packed.size()) / (m.n / 8) : 0;  // a tile of 8 columns, all its steps
        const auto run_pq = [&](int32_t n_src, std::vector<float> & o) {
            q.n_src    = n_src;
            int32_t c0 = 0;
            for (int32_t i = 0; i < n_src; ++i) {
                const int32_t cols = i + 1 < n_src ? (m.n / n_src) / 32 * 32 : m.n - c0;
                q.src[i].w         = w.as<uint8_t>() + int64_t(c0 / 8) * tile_bytes;
                q.src[i].n         = cols;
                c0 += cols;
            }
            for (const auto & b : outs) {
                if (!b->poison()) {
                    return false;
                }
            }
            if (!pq_tc(q, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "pq_tc")) {
                return false;
            }
            o.resize(size_t(r_all) * m.n);
            int32_t st = 0;
            for (int32_t i = 0; i < segs; ++i) {
                const int64_t      ld = q.t.seg[i].ld;
                std::vector<float> sg(size_t(r_all) * size_t(ld));
                if (!outs[size_t(i)]->get(sg.data(), sg.size() * 4)) {
                    return false;
                }
                for (int32_t r = 0; r < r_all; ++r) {
                    for (int32_t j = st; j < ends[size_t(i)]; ++j) {
                        o[size_t(r) * m.n + j] = sg[size_t(r) * size_t(ld) + size_t(j - st)];
                    }
                }
                st = ends[size_t(i)];
            }
            return true;
        };
        std::vector<float> f1, f3;
        const bool         three = m.lay == pd_layout::dp2 && m.n >= 96;
        if (!run_pq(1, f1) || (three && !run_pq(3, f3))) {
            v.put(name, false, "pq_tc launch failed");
            return false;
        }
        fused      = std::memcmp(out.data(), f1.data(), out.size() * 4) == 0 &&
                     (!three || std::memcmp(out.data(), f3.data(), out.size() * 4) == 0);
        fused_note = fused ? "; fused route bitwise (one source" + std::string(three ? ", three sources)" : ")") :
                             "; FUSED ROUTE DIFFERS from the converted route";
        if (segs == 1) {
            const int64_t  ld16 = m.n + 8;
            dbuf           o16(size_t(r_all) * size_t(ld16) * 2), ew(4);
            const uint32_t zero = 0;
            pq_args        q16  = q;
            q16.n_src           = 1;
            q16.src[0].w        = w.as<uint8_t>();
            q16.src[0].n        = m.n;
            q16.t.out16         = o16.as<uint16_t>();
            q16.t.ld16          = ld16;
            q16.t.err           = ew.as<uint32_t>();
            q16.t.err_bit       = 1u << 26;
            std::vector<uint16_t> got(size_t(r_all) * size_t(ld16));
            uint32_t              ev_word = 1;
            if (!o16.poison() || !ew.put(&zero, 4) || !pq_tc(q16, fn::ptest::test_stream()) ||
                !cuda_ok(cudaDeviceSynchronize(), "pq_tc out16") || !o16.get(got.data(), got.size() * 2) ||
                !ew.get(&ev_word, 4)) {
                v.put(name, false, "pq_tc out16 launch failed");
                return false;
            }
            int64_t diff = 0;
            for (int32_t r = 0; r < r_all; ++r) {
                for (int32_t j = 0; j < m.n; ++j) {
                    diff += got[size_t(r) * size_t(ld16) + size_t(j)] != quant::f32_to_f16(out[size_t(r) * m.n + j]);
                }
            }
            fused = fused && diff == 0 && ev_word == 0;
            fused_note += "; fused FP16 output " + std::to_string(diff) + " differ";
        }
    }
    std::vector<double> ref;
    reference(xv, r_all, m.k, wv, m.n, scaled ? &sc : nullptr, ref);
    const double e_conv = row_error(out, ref, r_all, m.n);
    // APF2: the ladder on every tile and split count from the packed matrix (ladder_sweep)
    ladder_sweep(
        [&]() {
            pq_args q;
            q.f        = m.f;
            q.lay      = m.lay;
            q.grids    = grids.as<uint8_t>();
            q.t        = a;
            q.t.w      = nullptr;
            q.n_src    = 1;
            q.src[0].w = w.as<uint8_t>();
            q.src[0].n = m.n;
            return q;
        }(),
        ends, outs, [&](const std::vector<float> & o) { return row_error(o, ref, r_all, m.n); }, e_conv, out, name,
        v);
    const bool loose = e_conv <= 1e-2;
    const bool ok    = bad == 0 && loose && same && p16 && tiles && fused;
    v.put(name, ok,
          "conversion " + std::to_string(nk) + " weights, " + std::to_string(bad) +
              " differ from decode-then-FP16; product " + (loose ? "within" : "OUTSIDE") +
              " the loose bug-check tolerance (1e-2 a row, finite)" + (same ? "; rerun bitwise" : "; RERUN DIFFERS") +
              p16_note + tiles_note + fused_note);
    return ok;
}

// PB3: BF16 weights to FP16 with the per-row range scale (pd_convert_bf16), then pd_tc with col_scale. Rows at 2^20,
// 2^-30 and zero among unit rows. The conversion bitwise against the CPU's same exponent rule; the product against FP64
// of the BF16 weights on the same FP16 activations at the loose bug-check tolerance, a column's error against its own
// magnitude (so a wrong scale on a 2^-30 column shows); a rerun bitwise
bool check_bf16_tc(int32_t n, int32_t k, int32_t rows, const std::vector<int32_t> & ends, verdict & v) {
    rng g(0xbf1c + uint64_t(n) * 3 + uint64_t(rows));
    const size_t nk = size_t(n) * k;
    std::vector<uint16_t> wb(nk), want(nk);
    std::vector<double> wv(nk);
    std::vector<float> want_s(static_cast<size_t>(n));
    for (int32_t j = 0; j < n; ++j) {
        const double sc = j % 7 == 0 ? std::ldexp(1.0, 20) : j % 7 == 1 ? std::ldexp(1.0, -30) : j % 7 == 2 ? 0.0 : 1.0;
        float m = 0.0f;
        for (int32_t i = 0; i < k; ++i) {
            const size_t at = size_t(j) * k + i;
            wb[at] = quant::f32_to_bf16(static_cast<float>(g.normal() * 0.05 * sc));
            wv[at] = quant::bf16_to_f32(wb[at]);
            m = std::max(m, std::fabs(static_cast<float>(wv[at])));
        }
        int32_t x = 0;
        (void)std::frexp(m, &x);
        const int32_t e = m > 0.0f ? std::max(-100, std::min(100, 15 - x)) : 0;
        for (int32_t i = 0; i < k; ++i) {
            const size_t at = size_t(j) * k + i;
            want[at] = quant::f32_to_f16(std::ldexp(static_cast<float>(wv[at]), e));
        }
        want_s[size_t(j)] = std::ldexp(1.0f, -e);
    }
    std::string name = "tc BF16 " + std::to_string(rows) + " x " + std::to_string(k) + " -> " + std::to_string(n) +
                       " in " + std::to_string(ends.size()) + " segments, range rows";
    dbuf w(nk * 2), wt(nk * 2), cs(size_t(n) * 4);
    std::vector<uint16_t> got(nk);
    std::vector<float> got_s(static_cast<size_t>(n));
    if (!w.put(wb.data(), nk * 2) || !wt.poison() || !cs.poison() ||
        !pd_convert_bf16(w.as<uint16_t>(), n, k, wt.as<uint16_t>(), cs.as<float>(), fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "pd_convert_bf16") || !wt.get(got.data(), nk * 2) ||
        !cs.get(got_s.data(), size_t(n) * 4)) {
        v.put(name, false, "conversion launch failed");
        return false;
    }
    int64_t bad = 0;
    for (size_t i = 0; i < nk; ++i) {
        bad += got[i] != want[i] && !((got[i] & 0x7fffu) == 0 && (want[i] & 0x7fffu) == 0) ? 1 : 0;
    }
    for (int32_t j = 0; j < n; ++j) {
        bad += got_s[size_t(j)] != want_s[size_t(j)] ? 1 : 0;
    }
    std::vector<uint16_t> eb;
    std::vector<float> ev;
    make_act(int64_t(rows) * k, 1.0, 0xbf1d + uint64_t(rows), eb, ev);
    dbuf e(eb.size() * 2);
    pt_args a;
    a.w = wt.as<uint16_t>();
    a.n = n;
    a.k = k;
    a.rows = rows;
    a.x.a0 = e.as<uint16_t>();
    a.x.ld0 = k;
    a.x.k_split = k;
    a.n_seg = static_cast<int32_t>(ends.size());
    a.col_scale = cs.as<float>();
    std::vector<std::unique_ptr<dbuf>> outs;
    int32_t start = 0;
    for (size_t i = 0; i < ends.size(); ++i) {
        const int64_t ld = ends[i] - start + 8;
        outs.push_back(std::make_unique<dbuf>(size_t(rows) * size_t(ld) * 4));
        a.seg[i] = {outs.back()->as<float>(), ld, ends[i]};
        start = ends[i];
    }
    if (!e.put(eb.data(), eb.size() * 2)) {
        return false;
    }
    std::vector<float> out, out2;
    const auto run = [&](std::vector<float> & o) {
        for (const auto & b : outs) {
            if (!b->poison()) {
                return false;
            }
        }
        if (!pd_tc(a, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "pd_tc")) {
            return false;
        }
        o.assign(size_t(rows) * n, 0.0f);
        int32_t st = 0;
        for (size_t i = 0; i < ends.size(); ++i) {
            const int64_t ld = a.seg[i].ld;
            std::vector<float> seg(size_t(rows) * size_t(ld));
            if (!outs[i]->get(seg.data(), seg.size() * 4)) {
                return false;
            }
            for (int32_t r = 0; r < rows; ++r) {
                for (int32_t j = st; j < ends[i]; ++j) {
                    o[size_t(r) * n + j] = seg[size_t(r) * size_t(ld) + size_t(j - st)];
                }
            }
            st = ends[i];
        }
        return true;
    };
    if (!run(out) || !run(out2)) {
        v.put(name, false, "launch failed");
        return false;
    }
    const bool same = std::memcmp(out.data(), out2.data(), out.size() * 4) == 0;
    double worst = 0.0;
    for (int32_t j = 0; j < n; ++j) {
        double em = 0.0, mm = 0.0;
        for (int32_t r = 0; r < rows; ++r) {
            double ref = 0.0;
            for (int32_t i = 0; i < k; ++i) {
                ref += double(ev[size_t(r) * k + i]) * wv[size_t(j) * k + i];
            }
            const double o = out[size_t(r) * n + j];
            em = std::isfinite(o) ? std::max(em, std::fabs(o - ref)) : INFINITY;
            mm = std::max(mm, std::fabs(ref));
        }
        worst = std::max(worst, mm > 0.0 ? em / mm : em);
    }
    const bool loose = worst <= 1e-2;
    // APF1 Part 4: pd_bf16_scales bitwise pd_convert_bf16's scales, and the fused BF16 route (the rows read as two
    // bases split at the first segment's end) bitwise the converted route
    dbuf               ps(size_t(n) * 4);
    std::vector<float> prep(static_cast<size_t>(n));
    std::vector<float> fo;
    pq_args            q;
    q.b0          = w.as<uint16_t>();
    q.n0          = ends.size() > 1 ? ends[0] : n;
    q.b1          = w.as<uint16_t>() + size_t(q.n0) * k;
    q.n1          = n - q.n0;
    q.b_scale     = ps.as<float>();
    q.t           = a;
    q.t.w         = nullptr;
    bool fused_ok = ps.poison() && pd_bf16_scales(w.as<uint16_t>(), n, k, ps.as<float>(), fn::ptest::test_stream()) &&
                    cuda_ok(cudaDeviceSynchronize(), "pd_bf16_scales") && ps.get(prep.data(), size_t(n) * 4);
    const bool scales_same = fused_ok && std::memcmp(prep.data(), got_s.data(), size_t(n) * 4) == 0;
    bool       fused_same  = false;
    if (fused_ok) {
        for (const auto & b : outs) {
            fused_ok = fused_ok && b->poison();
        }
        fused_ok = fused_ok && pq_tc(q, fn::ptest::test_stream()) && cuda_ok(cudaDeviceSynchronize(), "pq_tc");
        if (fused_ok) {
            fo.assign(size_t(rows) * n, 0.0f);
            int32_t st = 0;
            for (size_t i = 0; i < ends.size() && fused_ok; ++i) {
                const int64_t      ld = a.seg[i].ld;
                std::vector<float> seg(size_t(rows) * size_t(ld));
                fused_ok = outs[i]->get(seg.data(), seg.size() * 4);
                for (int32_t r = 0; r < rows && fused_ok; ++r) {
                    for (int32_t j = st; j < ends[i]; ++j) {
                        fo[size_t(r) * n + j] = seg[size_t(r) * size_t(ld) + size_t(j - st)];
                    }
                }
                st = ends[i];
            }
            fused_same = fused_ok && std::memcmp(fo.data(), out.data(), out.size() * 4) == 0;
        }
    }
    const bool fused = scales_same && fused_same;
    // APF2: the ladder on every tile and split count from the two BF16 bases (ladder_sweep), the column error
    if (fused_ok) {
        std::vector<double> refs(size_t(rows) * n, 0.0);
        for (int32_t j = 0; j < n; ++j) {
            for (int32_t r = 0; r < rows; ++r) {
                double ref = 0.0;
                for (int32_t i = 0; i < k; ++i) {
                    ref += double(ev[size_t(r) * k + i]) * wv[size_t(j) * k + i];
                }
                refs[size_t(r) * n + j] = ref;
            }
        }
        const auto col_err = [&](const std::vector<float> & o) {
            double wst = 0.0;
            for (int32_t j = 0; j < n; ++j) {
                double em = 0.0, mm = 0.0;
                for (int32_t r = 0; r < rows; ++r) {
                    const double ref = refs[size_t(r) * n + j];
                    const double x   = o[size_t(r) * n + j];
                    em             = std::isfinite(x) ? std::max(em, std::fabs(x - ref)) : INFINITY;
                    mm             = std::max(mm, std::fabs(ref));
                }
                wst = std::max(wst, mm > 0.0 ? em / mm : em);
            }
            return wst;
        };
        ladder_sweep(q, ends, outs, col_err, worst, out, name, v);
    }
    const bool ok    = bad == 0 && loose && same && fused;
    v.put(name, ok,
          "conversion " + std::to_string(nk) + " weights and " + std::to_string(n) + " scales, " + std::to_string(bad) +
              " differ; product " + (loose ? "within" : "OUTSIDE") +
              " the loose bug-check tolerance (1e-2 a column, finite)" +
              (loose ? "" : " [" + fn::ptest::e3(worst) + "]") + (same ? "; rerun bitwise" : "; RERUN DIFFERS") +
              (scales_same ? "; prepared scales bitwise" : "; PREPARED SCALES DIFFER") +
              (fused_same ? "; fused route bitwise (two bases)" : "; FUSED ROUTE DIFFERS"));
    return ok;
}

// APF1 Part 4: the router's region as the prompt lays it out (fn/model/prompt.cpp): 512 router rows, the shared gate's
// row, 7 zero rows with scale 0; the converted route (pd_convert_bf16 twice, the pad zeroed, pd_tc) against the fused
// route reading the two BF16 bases with prepared scales (zero past row 513), bitwise, at rows rows
bool check_router_tc(int32_t rows, verdict & v) {
    constexpr int32_t     ne = 512, n = 520, k = 2560;
    rng                   g(0x5047 + uint64_t(rows));
    std::vector<uint16_t> rb(size_t(ne) * k), gb(k);
    for (uint16_t & u : rb) {
        u = quant::f32_to_bf16(static_cast<float>(g.normal() * 0.02));
    }
    for (uint16_t & u : gb) {
        u = quant::f32_to_bf16(static_cast<float>(g.normal() * 0.5));
    }
    const std::string name = "tc router region " + std::to_string(rows) + " rows (512 + gate + 7 zero rows)";
    dbuf              r(rb.size() * 2), gt(gb.size() * 2), wt(size_t(n) * k * 2), cs(size_t(n) * 4), ps(size_t(n) * 4);
    std::vector<uint16_t> eb;
    std::vector<float>    ev;
    make_act(int64_t(rows) * k, 1.0, 0x5048 + uint64_t(rows), eb, ev);
    dbuf               e(eb.size() * 2), o1(size_t(rows) * ne * 4), o2(size_t(rows) * 8 * 4);
    const cudaStream_t st = static_cast<cudaStream_t>(fn::ptest::test_stream());
    bool ok = r.put(rb.data(), rb.size() * 2) && gt.put(gb.data(), gb.size() * 2) && e.put(eb.data(), eb.size() * 2) &&
              wt.poison() && cs.poison() && ps.poison() &&
              pd_convert_bf16(r.as<uint16_t>(), ne, k, wt.as<uint16_t>(), cs.as<float>(), st) &&
              pd_convert_bf16(gt.as<uint16_t>(), 1, k, wt.as<uint16_t>() + size_t(ne) * k, cs.as<float>() + ne, st) &&
              cuda_ok(cudaMemsetAsync(wt.as<uint16_t>() + size_t(ne + 1) * k, 0, size_t(7) * k * 2, st), "pad") &&
              cuda_ok(cudaMemsetAsync(cs.as<float>() + ne + 1, 0, 7 * 4, st), "pad") &&
              cuda_ok(cudaMemsetAsync(ps.as<float>(), 0, size_t(n) * 4, st), "pad") &&
              pd_bf16_scales(r.as<uint16_t>(), ne, k, ps.as<float>(), st) &&
              pd_bf16_scales(gt.as<uint16_t>(), 1, k, ps.as<float>() + ne, st) &&
              cuda_ok(cudaDeviceSynchronize(), "router conversion");
    pt_args a;
    a.w         = wt.as<uint16_t>();
    a.n         = n;
    a.k         = k;
    a.rows      = rows;
    a.x.a0      = e.as<uint16_t>();
    a.x.ld0     = k;
    a.x.k_split = k;
    a.n_seg     = 2;
    a.seg[0]    = { o1.as<float>(), ne, ne };
    a.seg[1]    = { o2.as<float>(), 8, n };
    a.col_scale = cs.as<float>();
    std::vector<float> c1(size_t(rows) * ne), c2(size_t(rows) * 8), f1(c1.size()), f2(c2.size());
    ok = ok && o1.poison() && o2.poison() && pd_tc(a, st) && cuda_ok(cudaDeviceSynchronize(), "pd_tc") &&
         o1.get(c1.data(), c1.size() * 4) && o2.get(c2.data(), c2.size() * 4);
    pq_args q;
    q.b0          = r.as<uint16_t>();
    q.n0          = ne;
    q.b1          = gt.as<uint16_t>();
    q.n1          = 1;
    q.b_scale     = ps.as<float>();
    q.t           = a;
    q.t.w         = nullptr;
    q.t.col_scale = ps.as<float>();
    ok            = ok && o1.poison() && o2.poison() && pq_tc(q, st) && cuda_ok(cudaDeviceSynchronize(), "pq_tc") &&
                    o1.get(f1.data(), f1.size() * 4) && o2.get(f2.data(), f2.size() * 4);
    if (!ok) {
        v.put(name, false, "launch failed");
        return false;
    }
    const bool same =
        std::memcmp(c1.data(), f1.data(), c1.size() * 4) == 0 && std::memcmp(c2.data(), f2.data(), c2.size() * 4) == 0;
    bool pad_zero = true;
    for (int32_t i = 0; i < rows; ++i) {
        for (int32_t j = 1; j < 8; ++j) {
            pad_zero = pad_zero && f2[size_t(i) * 8 + j] == 0.0f;
        }
    }
    v.put(name, same && pad_zero,
          std::string(same ? "fused bitwise the converted route" : "FUSED ROUTE DIFFERS") +
              (pad_zero ? ", the pad columns zero" : ", PAD COLUMNS NOT ZERO"));
    return same && pad_zero;
}

// BF16 products: identity rows, and the two-base mapping (period 2,560, half 1,280: PLE's key layout)
bool check_bf16(int32_t n, int32_t k, int32_t rows, bool mapped, verdict & v) {
    rng                   g(0xbf16 + uint64_t(n) * 7 + rows);
    const int32_t         nb = mapped ? n / 2 : n;  // rows a base holds
    std::vector<uint16_t> wl(size_t(nb) * k), wh(mapped ? size_t(nb) * k : 0);
    for (uint16_t & u : wl) {
        u = quant::f32_to_bf16(static_cast<float>(g.normal() * 0.05));
    }
    for (uint16_t & u : wh) {
        u = quant::f32_to_bf16(static_cast<float>(g.normal() * 0.05));
    }
    // the logical matrix's row j
    std::vector<float> w(size_t(n) * k);
    for (int32_t j = 0; j < n; ++j) {
        const uint16_t * src = wl.data() + size_t(j) * k;
        if (mapped) {
            const int32_t mm = j % 2560, h = mm / 1280, row = (j / 2560) * 1280 + mm % 1280;
            src = (h == 0 ? wl.data() : wh.data()) + size_t(row) * k;
        }
        for (int32_t i = 0; i < k; ++i) {
            w[size_t(j) * k + i] = quant::bf16_to_f32(src[i]);
        }
    }
    std::vector<float> x(size_t(rows) * k);
    for (float & f : x) {
        f = static_cast<float>(g.normal());
    }
    dbuf    dl(wl.size() * 2), dh(std::max<size_t>(wh.size(), 1) * 2), dx(x.size() * 4), o(size_t(rows) * n * 4);
    pb_args a;
    a.w_lo   = dl.as<uint16_t>();
    a.w_hi   = mapped ? dh.as<uint16_t>() : nullptr;
    a.period = mapped ? 2560 : 0;
    a.half   = mapped ? 1280 : 0;
    a.n      = n;
    a.k      = k;
    a.rows   = rows;
    a.x      = dx.as<float>();
    a.ldx    = k;
    a.out    = o.as<float>();
    a.ld_out = n;
    std::vector<float> out(size_t(rows) * n);
    const std::string  name = std::string("bf16 ") + (mapped ? "two-base " : "") + std::to_string(rows) + " x " +
                              std::to_string(k) + " -> " + std::to_string(n);
    if (!dl.put(wl.data(), wl.size() * 2) || (mapped && !dh.put(wh.data(), wh.size() * 2)) ||
        !dx.put(x.data(), x.size() * 4) || !o.poison() || !pd_bf16(a, fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "pd_bf16") || !o.get(out.data(), out.size() * 4)) {
        v.put(name, false, "launch failed");
        return false;
    }
    std::vector<double> ref;
    reference(x, rows, k, w, n, nullptr, ref);
    const double e = row_error(out, ref, rows, n);
    v.put(name, e <= 1e-5, "max err / max ref " + fmt_e(e) + " (bound 1e-5)");
    return e <= 1e-5;
}

bool check_glu(int32_t rows, int32_t n, verdict & v) {
    rng                g(0x61u);
    std::vector<float> gate(size_t(rows) * n), up(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = static_cast<float>(g.normal() * 3.0);
        up[i]   = static_cast<float>(g.normal() * 3.0);
    }
    dbuf                  dg(gate.size() * 4), du(up.size() * 4), dout(gate.size() * 2);
    std::vector<uint16_t> out(gate.size());
    if (!dg.put(gate.data(), gate.size() * 4) || !du.put(up.data(), up.size() * 4) ||
        !pd_glu(rows, n, dg.as<float>(), du.as<float>(), n, dout.as<uint16_t>(), n, fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "pd_glu") || !dout.get(out.data(), out.size() * 2)) {
        v.put("glu", false, "launch failed");
        return false;
    }
    int64_t bad = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double   gg = gate[i], ref = gg / (1.0 + std::exp(-gg)) * double(up[i]);
        const uint16_t want = quant::f32_to_f16(static_cast<float>(ref));
        // within one FP16 step of the FP64 value rounded to FP16 (F32 rounding before the FP16 one)
        if (std::abs(int32_t(out[i]) - int32_t(want)) > 1 || ((out[i] ^ want) & 0x8000u) != 0) {
            ++bad;
        }
    }
    v.put("glu " + std::to_string(rows) + " x " + std::to_string(n), bad == 0,
          std::to_string(bad) + " of " + std::to_string(out.size()) + " beyond one FP16 step of FP64");
    return bad == 0;
}

// the router's epilogue against FP64: the IDs exactly (random logits have no near ties), weights and gate 1e-6
bool check_route(int32_t rows, verdict & v) {
    rng                g(0x707);
    std::vector<float> lg(size_t(rows) * route_experts), sgl(static_cast<size_t>(rows));
    for (float & f : lg) {
        f = static_cast<float>(g.normal() * 1.5);
    }
    for (float & f : sgl) {
        f = static_cast<float>(g.normal());
    }
    dbuf dl(lg.size() * 4), ds(sgl.size() * 4), di(size_t(rows) * route_used * 4), dw(size_t(rows) * route_used * 4),
        dg(size_t(rows) * 4);
    std::vector<int32_t> ids(size_t(rows) * route_used);
    std::vector<float>   wts(ids.size()), sg(static_cast<size_t>(rows));
    if (!dl.put(lg.data(), lg.size() * 4) || !ds.put(sgl.data(), sgl.size() * 4) ||
        !pd_route(rows, dl.as<float>(), ds.as<float>(), di.as<int32_t>(), dw.as<float>(), dg.as<float>(),
                  fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "pd_route") || !di.get(ids.data(), ids.size() * 4) ||
        !dw.get(wts.data(), wts.size() * 4) || !dg.get(sg.data(), sg.size() * 4)) {
        v.put("router", false, "launch failed");
        return false;
    }
    int32_t bad_ids = 0;
    double  ew = 0.0, eg = 0.0;
    for (int32_t r = 0; r < rows; ++r) {
        const float * l  = lg.data() + size_t(r) * route_experts;
        double        mx = -HUGE_VAL, sum = 0.0;
        for (int32_t i = 0; i < route_experts; ++i) {
            mx = std::max(mx, double(l[i]));
        }
        std::vector<std::pair<double, int32_t>> p(route_experts);
        for (int32_t i = 0; i < route_experts; ++i) {
            p[size_t(i)] = { std::exp(double(l[i]) - mx), i };
            sum += p[size_t(i)].first;
        }
        std::sort(p.begin(), p.end(), [](const auto & a, const auto & b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        double tot = 0.0;
        for (int32_t k = 0; k < route_used; ++k) {
            tot += p[size_t(k)].first / sum;
        }
        for (int32_t k = 0; k < route_used; ++k) {
            bad_ids += ids[size_t(r) * route_used + k] != p[size_t(k)].second;
            const double w = p[size_t(k)].first / sum / tot;
            ew             = std::max(ew, std::fabs(wts[size_t(r) * route_used + k] - w) / w);
        }
        const double gs = 1.0 / (1.0 + std::exp(-double(sgl[size_t(r)])));
        eg              = std::max(eg, std::fabs(sg[size_t(r)] - gs) / gs);
    }
    const bool ok = bad_ids == 0 && ew <= 1e-6 && eg <= 1e-6;
    v.put("router " + std::to_string(rows) + " rows", ok,
          std::to_string(bad_ids) + " IDs differ from FP64's top 10; weights " + fmt_e(ew) + ", gate " + fmt_e(eg) +
              " (relative, bound 1e-6)");
    return ok;
}

// TTTS3 item 7 (pd-test DEVICE q8=GGUF): TTTS2's EX1-down Q8_0 on UD-Q4_K_XL's real shared downs (ffn_down_shexp,
// Q8_0 [2560][640]; card r's matrix the 2,560 rows' columns 320 r .. 320 r + 319, packed by pack_down as the loader
// does) of layers 0, 25 and 47: the codes bitwise the CPU decoder, the products at 1, 65 (scaled, rerun bitwise) and
// 129 rows against FP64, and the conversion bitwise with pd_tc over it (the prompt's path)
bool real_ex1_q8(const std::string & model, const dbuf & grids, verdict & v) {
    std::vector<std::string>   shards;
    std::vector<fnfl_gguf_dir> dirs;
    std::string                err;
    bool                       ok = fnfl_gguf_shard_paths(model, shards, err);
    for (size_t i = 0; ok && i < shards.size(); ++i) {
        dirs.emplace_back();
        ok = fnfl_gguf_read_dir(shards[i], dirs.back(), err);
    }
    if (!ok) {
        v.put("real EX1 down", false, err);
        return false;
    }
    constexpr int32_t N = 2560, K = 320;
    const size_t      rb = size_t(K / 32 * 34), full = 2 * rb;
    for (const int32_t layer : { 0, 25, 47 }) {
        const std::string        name = "blk." + std::to_string(layer) + ".ffn_down_shexp.weight";
        const fnfl_gguf_tensor * t    = nullptr;
        const fnfl_gguf_dir *    td   = nullptr;
        for (const fnfl_gguf_dir & d : dirs) {
            for (const fnfl_gguf_tensor & x : d.tensors) {
                if (x.name == name) {
                    t  = &x;
                    td = &d;
                }
            }
        }
        std::vector<uint8_t> all(full * N);
        FILE *               fp   = t != nullptr ? fopen(td->path.c_str(), "rb") : nullptr;
        const bool           read = fp != nullptr && t->type_name == "q8_0" && t->bytes == all.size() &&
                                    fseeko(fp, static_cast<off_t>(t->offset), SEEK_SET) == 0 &&
                                    fread(all.data(), 1, all.size(), fp) == all.size();
        if (fp != nullptr) {
            fclose(fp);
        }
        if (!read) {
            v.put(name, false, "missing, not Q8_0 [2560][640] or unreadable");
            ok = false;
            continue;
        }
        for (int32_t card = 0; card < 2; ++card) {
            printf("pd-test: real EX1 down: layer %d card %d (%s columns %d .. %d)\n", layer, card, name.c_str(),
                   K * card, K * card + K - 1);
            matrix m;
            m.f   = fmt::q8_0;
            m.lay = pd_layout::ex1_down;
            m.n   = N;
            m.k   = K;
            std::vector<uint8_t> rows(rb * N);
            for (int32_t n = 0; n < N; ++n) {
                std::memcpy(rows.data() + rb * size_t(n), all.data() + full * size_t(n) + rb * size_t(card), rb);
            }
            m.deq.assign(size_t(N) * K, 0.0f);
            bool dec = true;
            for (int32_t n = 0; n < N; ++n) {
                dec = dec && quant::decode_row(quant::wfmt::q8_0, rows.data() + rb * size_t(n),
                                               m.deq.data() + size_t(n) * K, K);
            }
            m.packed.assign(size_t(experts::down_bytes(experts::fmt::q8_0)), 0);
            dbuf w(m.packed.size());
            if (!dec || !experts::pack_down(experts::fmt::q8_0, rows.data(), rb, m.packed.data()) ||
                !w.put(m.packed.data(), m.packed.size())) {
                v.put(name, false, "decode or pack_down");
                ok = false;
                continue;
            }
            ok = check_codes(m, w, grids, v, false) && ok;
            for (const int32_t r : { 1, 65, 129 }) {
                ok = check_product(m, w, grids, r, 0x7100 + uint64_t(layer) * 17 + uint64_t(card) * 5 + uint64_t(r),
                                   r == 65, r == 65, v) &&
                     ok;
            }
            ok = check_tc(m, w, grids, 129, 0, 1, true, 0x7900 + uint64_t(layer) * 3 + uint64_t(card), v) && ok;
            ok = check_tc(m, w, grids, 56, 0, 1, true, 0x7a00 + uint64_t(layer) * 3 + uint64_t(card), v) && ok;  // MUP1
        }
    }
    return ok;
}

}  // namespace

}  // namespace fn::dense

int fnfl_pd_test(int device, const char * mode) {
    using namespace fn::dense;
    const bool san = mode != nullptr && std::strcmp(mode, "san") == 0;
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    verdict              v("pd-test");
    std::vector<uint8_t> gt;
    if (!grid_buffer(gt)) {
        fprintf(stderr, "pd-test: grid buffer\n");
        return 1;
    }
    dbuf grids(gt.size());
    if (!grids.put(gt.data(), gt.size())) {
        return 1;
    }
    if (mode != nullptr && std::strncmp(mode, "q8=", 3) == 0) {  // TTTS3 item 7: real Q8_0 shared downs only
        printf("pd-test: device %d, real EX1-down Q8_0 from %s\n", device, mode + 3);
        real_ex1_q8(std::string(mode + 3), grids, v);
        printf("pd-test: %s (%d checks)\n", v.all ? "PASS" : "FAIL", v.n);
        return v.all ? 0 : 1;
    }
    printf("pd-test: device %d, mode %s\n", device, san ? "san" : "full");
    g_san = san;
    if (!pl_attributes()) {
        fprintf(stderr, "pd-test: the ladder's shared memory opt-in\n");
        return 1;
    }
    const fmt dp2[] = { fmt::q4_k, fmt::q5_k, fmt::q6_k, fmt::iq4_xs, fmt::iq3_s, fmt::q8_0 };
    for (const fmt f : dp2) {
        // the codes over 512 columns of K 2,560 (every lane position of 64 tiles); the products at row counts that
        // cover a partial panel, a full one and several, at one and two column blocks
        matrix        m;
        const int32_t n = san ? 64 : 512;
        if (!make_matrix(f, pd_layout::dp2, n, 2560, 3, m)) {
            v.put(std::string("matrix ") + fmt_name(f), false, "synthetic rows");
            continue;
        }
        dbuf w(m.packed.size());
        if (!w.put(m.packed.data(), m.packed.size())) {
            return 1;
        }
        check_codes(m, w, grids, v);
        const std::vector<int32_t> rows =
            san ? std::vector<int32_t>{ 1, 65 } : std::vector<int32_t>{ 1, 5, 64, 65, 130 };
        for (const int32_t r : rows) {
            check_product(m, w, grids, r, 0x5000 + uint64_t(f) * 131 + r, false, r == 65, v);
        }
        // MUP1 piece 2: 33 and 64 rows (the 64 x 64 tile against the 128 x 128 tile); APF1 Part 4: each also the fused
        // route, and around the fused tile's 64 rows and the ladder's 512-row dense crossover (fn/model/prompt.h)
        std::vector<int32_t> tr = san ? std::vector<int32_t>{ 33, 65 } : std::vector<int32_t>{ 1, 33, 64, 129, 300 };
        if (!san && f == fmt::q8_0) {
            // APF2: just below, at and above each ladder tile's rows (Q4's dense format)
            for (const int32_t r : { 7, 8, 9, 15, 16, 17, 31, 32, 63, 65, 127, 128, 255, 256, 257, 511, 512, 513 }) {
                tr.push_back(r);
            }
        }
        for (const int32_t r : tr) {
            check_tc(m, w, grids, r, 0, r == 300 ? 3 : 1, false, 0x5800 + uint64_t(f) * 131 + r, v);
        }
    }
    // the deployed shapes the dense products take beyond K 2,560: ssm_out and wo (3,072 -> 2,560) and eh_proj's
    // two-segment operand (Q8_0, 5,120)
    {
        matrix m;
        if (make_matrix(fmt::q6_k, pd_layout::dp2, san ? 64 : 2560, 3072, 11, m)) {
            dbuf w(m.packed.size());
            if (w.put(m.packed.data(), m.packed.size())) {
                check_product(m, w, grids, san ? 9 : 97, 0x6001, false, false, v);
            }
        }
        matrix q8;
        if (make_matrix(fmt::q8_0, pd_layout::dp2, san ? 64 : 1280, 5120, 17, q8)) {
            dbuf w(q8.packed.size());
            if (w.put(q8.packed.data(), q8.packed.size())) {
                check_two_segment(q8, w, grids, san ? 3 : 41, v);
                check_tc(q8, w, grids, 0, san ? 3 : 41, 1, false, 0xe0e0, v);
            }
        }
        matrix q4;
        if (make_matrix(fmt::q4_k, pd_layout::dp2, 320, 2560, 29, q4)) {
            dbuf w(q4.packed.size());
            if (w.put(q4.packed.data(), q4.packed.size())) {
                check_product(q4, w, grids, san ? 7 : 70, 0x6002, true, false, v);
                check_tc(q4, w, grids, san ? 7 : 70, 0, 2, true, 0x6003, v);
            }
        }
    }
    // the shared expert's down in EX1's layout (TTTS3 item 7: TTTS2's Q8_0 too)
    for (const fmt f : { fmt::iq4_nl, fmt::q2_0, fmt::q8_0 }) {
        matrix m;
        if (!make_matrix(f, pd_layout::ex1_down, 2560, 320, 5, m)) {
            v.put(std::string("matrix ") + fmt_name(f), false, "synthetic rows");
            continue;
        }
        dbuf w(m.packed.size());
        if (!w.put(m.packed.data(), m.packed.size())) {
            return 1;
        }
        check_codes(m, w, grids, v);
        for (const int32_t r : san ? std::vector<int32_t>{ 3 } : std::vector<int32_t>{ 1, 65, 129 }) {
            check_product(m, w, grids, r, 0x7000 + uint64_t(f) * 17 + r, r == 65, r == 65, v);
        }
        check_tc(m, w, grids, san ? 3 : 129, 0, 1, true, 0x7800 + uint64_t(f), v);
    }
    check_bf16(san ? 64 : 513, 2560, san ? 3 : 65, false, v);
    check_bf16(san ? 5120 : 10240, 2560, san ? 2 : 9, true, v);
    // PB3: the BF16 products' shapes: GDN alpha + beta (24 + 24), QSA index q + k (512 + 128), the router (512)
    check_bf16_tc(48, 2560, san ? 3 : 65, {24, 48}, v);
    if (!san) {
        check_bf16_tc(640, 2560, 129, {512, 640}, v);
        check_bf16_tc(512, 2560, 300, {512}, v);
    }
    // APF1 Part 4: the router's region (two bases and zero rows), around the fused tile's rows and the crossover
    for (const int32_t r : san ? std::vector<int32_t>{ 3 } : std::vector<int32_t>{ 1, 64, 65, 512 }) {
        check_router_tc(r, v);
    }
    if (!san) {
        check_bf16_tc(640, 2560, 513, { 512, 640 }, v);
    }
    check_glu(san ? 3 : 200, 320, v);
    check_route(san ? 3 : 1536, v);
    pd_resources pr{};
    if (pd_tc_resources(pr)) {
        // reported, not judged (PB3 Amendment 3: resources are a cost)
        printf("pd-test: pd_tc resources: %d regs, %d B local, %d B shared, %d CTA/SM\n", pr.regs, pr.local_bytes,
               pr.shared_bytes, pr.ctas_per_sm);
    }
    // APF1 Part 4: the fused route's instances (reported, not judged)
    for (const fmt f : { fmt::q8_0, fmt::q4_k, fmt::q5_k, fmt::q6_k, fmt::iq4_xs, fmt::iq3_s }) {
        if (pq_tc_resources(false, f, pd_layout::dp2, pr)) {
            printf("pd-test: pq_tc %s resources: %d regs, %d B local, %d B shared, %d CTA/SM\n", fmt_name(f), pr.regs,
                   pr.local_bytes, pr.shared_bytes, pr.ctas_per_sm);
        }
    }
    if (pq_tc_resources(true, fmt::bf16, pd_layout::dp2, pr)) {
        printf("pd-test: pq_tc BF16 resources: %d regs, %d B local, %d B shared, %d CTA/SM\n", pr.regs, pr.local_bytes,
               pr.shared_bytes, pr.ctas_per_sm);
    }
    printf("pd-test: %s (%d checks)\n", v.all ? "PASS" : "FAIL", v.n);
    return v.all ? 0 : 1;
}
