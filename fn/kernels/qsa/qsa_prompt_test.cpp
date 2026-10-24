#include "fn/kernels/qsa/qsa_prompt_test.h"

#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/qsa/qsa_prompt.h"
#include "fn/model/state.h"
#include "fn/quant/formats.h"
#include "fn/quant/kv_q8_ref.h"
#include "fn/tools/prompt_test_util.h"
#include "fn/tools/test_dump.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace fn::ptest;
namespace qsa        = fn::qsa;
constexpr int32_t KD = qsa::head_dim;  // 128
constexpr int32_t HD = 256, NH = 12, QS = 512;

// the rope angle as the kernels form it: the position times theta_scale^i by the F32 product chain
double angle(int32_t pos, int32_t i, float ts) {
    float theta = static_cast<float>(pos);
    for (int32_t k = 0; k < i; ++k) {
        theta = theta * ts;
    }
    return theta;
}

// rms_norm(x) w then the IMROPE of a 256-wide row at pos (dims 0..63 in pairs (i, i + 32)), FP64
void norm_rope_ref(const float * x, const float * w, int32_t pos, float ts, double * y) {
    double ss = 0.0;
    for (int32_t e = 0; e < HD; ++e) {
        ss += double(x[e]) * x[e];
    }
    const double r = 1.0 / std::sqrt(ss / HD + 1e-6);
    for (int32_t e = 0; e < HD; ++e) {
        y[e] = double(x[e]) * r * w[e];
    }
    for (int32_t i = 0; i < 32; ++i) {
        const double th = angle(pos, i, ts), c = std::cos(th), s = std::sin(th);
        const double a = y[i], b = y[i + 32];
        y[i]      = a * c - b * s;
        y[i + 32] = a * s + b * c;
    }
}

struct fixture {
    int32_t               cells = 16384;
    std::vector<uint16_t> keys;  // the raw index keys [cells][128] BF16
    dbuf                  dkeys{ size_t(16384) * KD * 2 }, dsum{ qsa::summary_bytes(16384 / 4) }, dkn{ size_t(KD) * 4 },
        dqn{ size_t(KD) * 4 };
};

// The selection of rows [pos0, pos0 + n): the prompt path's against the decode select's, one row at a time. PB3
// (Amendment 2; the prompt scorer's FP16 products): a bug check. Each body's counts and tail equal the decode
// select's, its blocks ascend strictly below the row's visibility, and at least 95% of them are the decode select's
// (all of them for a row that takes every block, and with exact set, bitwise); a rerun is bitwise.
bool select_case(fixture & fx, int32_t pos0, int32_t n, verdict & v, const char * tag = "", bool exact = false) {
    rng                r(0x5e1 + uint64_t(pos0));
    std::vector<float> iq(size_t(n) * 4 * KD);
    for (float & f : iq) {
        f = static_cast<float>(r.normal());
    }
    dbuf diq(iq.size() * 4), ws(qsa::prompt_ws_bytes(fx.cells)), body(static_cast<size_t>(qsa::prompt_sel_bytes)),
        dws(qsa::workspace_bytes()), dbody(size_t(qsa::max_row_words) * 4 * 4), doff(64), ddepth(256);
    if (!diq.put(iq) || !body.poison() || !dws.zero()) {
        return false;
    }
    qsa::qsp_index_args a;
    a.summaries = fx.dsum.as<float>();
    a.iq        = diq.as<float>();
    a.q_norm    = fx.dqn.as<float>();
    a.rope      = qsa::model_rope();
    a.rows      = n;
    a.pos0      = pos0;
    a.max_cells = fx.cells;
    a.ws        = qsa::prompt_carve(ws.p, fx.cells);
    a.body      = body.as<uint32_t>();
    a.offsets   = reinterpret_cast<int32_t *>(a.body + qsa::prompt_body_words);
    const std::string name =
        std::string(tag) + "selection rows " + std::to_string(pos0) + " .. " + std::to_string(pos0 + n);
    std::vector<uint32_t> all;
    if (!qsa::qsp_select(a, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "qsp_select") ||
        !body.get(all, size_t(qsa::prompt_sel_bytes / 4))) {
        v.put(name, false, "launch failed");
        return false;
    }
    std::vector<uint32_t> again;
    if (!body.poison() || !qsa::qsp_select(a, fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "qsp_select") || !body.get(again, size_t(qsa::prompt_sel_bytes / 4))) {
        v.put(name, false, "launch failed");
        return false;
    }
    const int32_t * offs = reinterpret_cast<const int32_t *>(all.data() + qsa::prompt_body_words);
    int32_t bad = 0, first = -1, selected = 0, n_diff = 0;
    double worst = 1.0;
    for (int32_t t = 0; t < n; ++t) {
        const int32_t    depth = pos0 + t;
        qsa::select_args sa;
        sa.summaries = fx.dsum.as<float>();
        sa.q         = diq.as<float>() + int64_t(t) * 4 * KD;
        sa.q_norm    = fx.dqn.as<float>();
        sa.depth     = ddepth.as<int32_t>();
        sa.max_cells = fx.cells;
        sa.rope      = qsa::model_rope();
        sa.ws        = qsa::carve(dws.p);
        sa.body      = dbody.as<uint32_t>();
        sa.offsets   = doff.as<int32_t>();
        std::vector<uint32_t> want;
        if (!ddepth.put(std::vector<int32_t>{ depth }) || !qsa::select(1, sa, fn::ptest::test_stream()) ||
            !cuda_ok(cudaDeviceSynchronize(), "select") || !dbody.get(want, size_t(qsa::row_words(depth)))) {
            v.put(name, false, "decode select failed");
            return false;
        }
        const int32_t    s = t / qsa::prompt_slab, rr = t % qsa::prompt_slab;
        const uint32_t * got =
            all.data() + int64_t(s) * qsa::prompt_slab_words + offs[s * (qsa::prompt_slab + 1) + rr] / 4;
        const int32_t n_vis = (depth + 1) / 4, n_sel = static_cast<int32_t>(want[0]);
        bool ok = got[0] == want[0] && got[1] == want[1] &&
                  std::memcmp(got + 2 + n_sel, want.data() + 2 + n_sel, size_t(want[1]) * 4) == 0;
        int32_t common = 0;
        for (int32_t i = 0, j = 0; ok && i < n_sel; ++i) {
            ok = got[2 + i] < uint32_t(n_vis) && (i == 0 || got[2 + i] > got[1 + i]);
            while (j < n_sel && want[2 + j] < got[2 + i]) {
                ++j;
            }
            common += j < n_sel && want[2 + j] == got[2 + i] ? 1 : 0;
        }
        const bool same = std::memcmp(got, want.data(), want.size() * 4) == 0;
        n_diff += same ? 0 : 1;
        if (ok && n_sel > 0) {
            worst = std::min(worst, double(common) / n_sel);
        }
        ok = ok && (exact || n_vis <= qsa::top_blocks ? same : common * 100 >= n_sel * 95);
        selected += n_vis > qsa::top_blocks ? 1 : 0;
        if (!ok) {
            ++bad;
            if (first < 0) {
                first = depth;
            }
        }
    }
    const bool rerun = all == again;
    char ov[64];
    snprintf(ov, sizeof(ov), "%.4f", worst);
    v.put(name, bad == 0 && rerun,
          std::to_string(n) + " rows (" + std::to_string(selected) + " select top 512 of more), " +
              std::to_string(bad) + " bodies fail (" + std::to_string(n_diff) +
              " differ from the decode select's; the least overlap " + ov + ")" +
              (first >= 0 ? " (first at " + std::to_string(first) + ")" : "") + "; rerun " +
              (rerun ? "bitwise" : "DIFFERS"));
    return bad == 0 && rerun;
}

// KV8: QSA layer 7's column of the scale rows (fn/model/state.h kv_q8_scale_stride)
constexpr int32_t SS = fn::model::kv_q8_scale_stride, q8_col = 4 * 7;

// the caches of a case in either format: BF16 [cells][256], or KV8's codes [cells][256] and scale rows [cells][SS]
struct caches {
    bool q8 = false;
    std::vector<uint16_t> k, v; // BF16, or (q8) the scale rows
    std::vector<int8_t> k8, v8; // q8: the codes

    double kval(int64_t c, int32_t e) const {
        return q8 ? double(fn::quant::kv_q8_value(k8[size_t(c * HD + e)], k[size_t(c * SS + q8_col + e / 64)]))
                  : double(fn::quant::bf16_to_f32(k[size_t(c * HD + e)]));
    }

    double vval(int64_t c, int32_t e) const {
        return q8 ? double(fn::quant::kv_q8_value(v8[size_t(c * HD + e)], v[size_t(c * SS + q8_col + e / 64)]))
                  : double(fn::quant::bf16_to_f32(v[size_t(c * HD + e)]));
    }

    // q8: a row's group scale (the decode's s)
    double kscale(int64_t c, int32_t e) const {
        return double(fn::quant::f16_value(k[size_t(c * SS + q8_col + e / 64)]));
    }
};

// K and V of n rows at pos0 against FP64 (K within one BF16 step, V bitwise; KV8: K decoded within half a
// quantization step of FP64 (plus 1e-5 of the row's largest value for the kernel's F32 k), V's codes and scales
// bitwise the CPU quantizer's), then attention of those rows over the prompt path's own selection against FP64 over
// the decoded cache (PB3: a loose bug-check tolerance). vrows (optional): the rows' v (the quantizer's stress rows; the
// write checked alone, the attention run for the dumps only)
bool kv_attend_case(fixture & fx, int32_t pos0, int32_t n, verdict & v, bool q8, const float * vrows = nullptr,
                    const char * tag = "") {
    rng                r(0xa77 + uint64_t(pos0));
    const int32_t      cells = pos0 + n;
    std::vector<float> kk(size_t(n) * HD), vv(size_t(n) * HD), kn(HD), qn(HD), qfull(size_t(n) * NH * QS);
    for (float & f : kk) {
        f = static_cast<float>(r.normal());
    }
    for (float & f : vv) {
        f = static_cast<float>(r.normal());
    }
    if (vrows != nullptr) {
        std::copy(vrows, vrows + size_t(n) * HD, vv.begin());
    }
    for (float & f : kn) {
        f = static_cast<float>(r.uniform(0.5, 1.5));
    }
    for (float & f : qn) {
        f = static_cast<float>(r.uniform(0.5, 1.5));
    }
    for (float & f : qfull) {
        f = static_cast<float>(r.normal());
    }
    // the caches below pos0: random (earlier rows' cells), BF16 or (KV8) quantized by the CPU reference
    caches h;
    h.q8 = q8;
    if (q8) {
        h.k8.assign(size_t(fx.cells) * HD, 0);
        h.v8.assign(size_t(fx.cells) * HD, 0);
        h.k.assign(size_t(fx.cells) * SS, 0);
        h.v.assign(size_t(fx.cells) * SS, 0);
        std::vector<float> rk(HD), rv(HD);
        for (int64_t c = 0; c < pos0; ++c) {
            for (int32_t e = 0; e < HD; ++e) {
                rk[size_t(e)] = static_cast<float>(r.normal());
                rv[size_t(e)] = static_cast<float>(r.normal());
            }
            fn::quant::kv_q8_quantize(rk.data(), HD, h.k8.data() + c * HD, h.k.data() + c * SS + q8_col);
            fn::quant::kv_q8_quantize(rv.data(), HD, h.v8.data() + c * HD, h.v.data() + c * SS + q8_col);
        }
    } else {
        h.k.assign(size_t(fx.cells) * HD, 0);
        h.v.assign(size_t(fx.cells) * HD, 0);
        for (int64_t i = 0; i < int64_t(pos0) * HD; ++i) {
            h.k[size_t(i)] = fn::quant::f32_to_bf16(static_cast<float>(r.normal()));
            h.v[size_t(i)] = fn::quant::f32_to_bf16(static_cast<float>(r.normal()));
        }
    }
    dbuf dk(kk.size() * 4), dv(vv.size() * 4), dkn(size_t(HD) * 4), dqn(size_t(HD) * 4), dkc(size_t(fx.cells) * HD * 2),
        dvc(size_t(fx.cells) * HD * 2), dks(size_t(fx.cells) * SS * 2), dvs(size_t(fx.cells) * SS * 2),
        dq(qfull.size() * 4), out(size_t(n) * NH * HD * 4), out16(size_t(n) * NH * HD * 2), err(256);
    const bool up = q8 ? dkc.put(h.k8) && dvc.put(h.v8) && dks.put(h.k) && dvs.put(h.v) : dkc.put(h.k) && dvc.put(h.v);
    if (!dk.put(kk) || !dv.put(vv) || !dkn.put(kn) || !dqn.put(qn) || !up || !dq.put(qfull) || !out.poison() ||
        !err.zero()) {
        return false;
    }
    const float      ts = qsa::attn::rope_theta_scale();
    qsa::qsp_kv_args ka;
    ka.kv_k      = dk.as<float>();
    ka.kv_v      = dv.as<float>();
    ka.k_norm    = dkn.as<float>();
    ka.rows      = n;
    ka.pos0      = pos0;
    ka.max_cells = fx.cells;
    ka.k_cache   = dkc.as<uint16_t>();
    ka.v_cache   = dvc.as<uint16_t>();
    if (q8) {
        ka.k_scale = dks.as<uint16_t>() + q8_col;
        ka.v_scale = dvs.as<uint16_t>() + q8_col;
    }
    caches g; // the caches as the launch left them
    g.q8 = q8;
    const bool got = qsa::qsp_kv(ka, fn::ptest::test_stream()) && cuda_ok(cudaDeviceSynchronize(), "qsp_kv") &&
                     (q8 ? dkc.get(g.k8, h.k8.size()) && dvc.get(g.v8, h.v8.size()) && dks.get(g.k, h.k.size()) &&
                               dvs.get(g.v, h.v.size())
                         : dkc.get(g.k, h.k.size()) && dvc.get(g.v, h.v.size()));
    if (!got) {
        v.put(std::string(tag) + "kv", false, "launch failed");
        return false;
    }
    int64_t kbad = 0, vbad = 0;
    for (int32_t t = 0; t < n; ++t) {
        double y[HD];
        norm_rope_ref(kk.data() + size_t(t) * HD, kn.data(), pos0 + t, ts, y);
        const int64_t cell = int64_t(pos0) + t;
        if (q8) {
            double ymax = 0.0;
            for (int32_t e = 0; e < HD; ++e) {
                ymax = std::max(ymax, std::fabs(y[e]));
            }
            for (int32_t e = 0; e < HD; ++e) {
                kbad += std::fabs(g.kval(cell, e) - y[e]) <= 0.5 * g.kscale(cell, e) + 1e-5 * ymax ? 0 : 1;
            }
            int8_t rc[HD];
            uint16_t rs[4];
            fn::quant::kv_q8_quantize(vv.data() + size_t(t) * HD, HD, rc, rs);
            vbad += std::memcmp(rc, g.v8.data() + cell * HD, HD) == 0 &&
                            std::memcmp(rs, g.v.data() + cell * SS + q8_col, 8) == 0
                        ? 0
                        : 1;
            continue;
        }
        for (int32_t e = 0; e < HD; ++e) {
            const size_t at = size_t(cell) * HD + e;
            const double gk = fn::quant::bf16_to_f32(g.k[at]);
            const double ulp = std::ldexp(1.0, std::ilogb(std::max(std::fabs(y[e]), 1e-30)) - 7);
            kbad += std::fabs(gk - y[e]) <= ulp ? 0 : 1;
            vbad += g.v[at] == fn::quant::f32_to_bf16(vv[size_t(t) * HD + e]) ? 0 : 1;
        }
    }
    v.put(std::string(tag) + "K/V write rows " + std::to_string(pos0) + " + " + std::to_string(n),
          kbad == 0 && vbad == 0,
          q8 ? std::to_string(kbad) + " K values beyond half a quantization step of FP64, " + std::to_string(vbad) +
                   " V rows not the CPU quantizer's codes and scales (q8)"
             : std::to_string(kbad) + " K values beyond one BF16 step of FP64, " + std::to_string(vbad) +
                   " V values not bf16(v)");
    // the selection of the rows (the prompt path's own; checked against decode's above)
    std::vector<float> iq(size_t(n) * 4 * KD);
    for (float & f : iq) {
        f = static_cast<float>(r.normal());
    }
    dbuf diq(iq.size() * 4), ws(qsa::prompt_ws_bytes(fx.cells)), body(static_cast<size_t>(qsa::prompt_sel_bytes));
    qsa::qsp_index_args ia;
    ia.summaries = fx.dsum.as<float>();
    ia.iq        = diq.as<float>();
    ia.q_norm    = fx.dqn.as<float>();
    ia.rope      = qsa::model_rope();
    ia.rows      = n;
    ia.pos0      = pos0;
    ia.max_cells = fx.cells;
    ia.ws        = qsa::prompt_carve(ws.p, fx.cells);
    ia.body      = body.as<uint32_t>();
    ia.offsets   = reinterpret_cast<int32_t *>(ia.body + qsa::prompt_body_words);
    qsa::qsp_attend_args aa;
    aa.qfull     = dq.as<float>();
    aa.q_norm    = dqn.as<float>();
    aa.body      = ia.body;
    aa.offsets   = ia.offsets;
    aa.k_cache   = dkc.as<uint16_t>();
    aa.v_cache   = dvc.as<uint16_t>();
    aa.k_scale = ka.k_scale;
    aa.v_scale = ka.v_scale;
    aa.rows      = n;
    aa.pos0      = pos0;
    aa.max_cells = fx.cells;
    aa.out       = out.as<float>();
    aa.out16     = out16.as<uint16_t>();
    aa.err       = err.as<uint32_t>();
    std::vector<float> gout, gout2;
    std::vector<uint32_t> bodies, ew;
    if (!diq.put(iq) || !qsa::qsp_select(ia, fn::ptest::test_stream()) ||
        !qsa::qsp_attend(aa, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "attend") ||
        !out.get(gout, size_t(n) * NH * HD) || !body.get(bodies, size_t(qsa::prompt_sel_bytes / 4)) ||
        !err.get(ew, 1) || !qsa::qsp_attend(aa, fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "attend") || !out.get(gout2, size_t(n) * NH * HD)) {
        v.put(std::string(tag) + "attend", false, "launch failed");
        return false;
    }
    // KV8's correction check: the caches written and both launches' outputs, for a byte comparison across binaries
    // (test_dump: FNFL_TEST_DUMP set)
    {
        std::vector<uint16_t> g16;
        const std::string d = std::string("qsp-") + (q8 ? "q8" : "bf16") + (vrows != nullptr ? "-stress" : "") + "-" +
                              std::to_string(pos0) + "+" + std::to_string(n);
        const bool dumped = out16.get(g16, size_t(n) * NH * HD) &&
                            fn::tools::test_dump(d + "-out", gout.data(), gout.size() * 4) &&
                            fn::tools::test_dump(d + "-out-rerun", gout2.data(), gout2.size() * 4) &&
                            fn::tools::test_dump(d + "-out16", g16.data(), g16.size() * 2) &&
                            fn::tools::test_dump(d + "-k", g.k.data(), g.k.size() * 2) &&
                            fn::tools::test_dump(d + "-v", g.v.data(), g.v.size() * 2) &&
                            (!q8 || (fn::tools::test_dump(d + "-k8", g.k8.data(), g.k8.size()) &&
                                     fn::tools::test_dump(d + "-v8", g.v8.data(), g.v8.size())));
        if (!dumped) {
            v.put(std::string(tag) + "dump", false, "a dump failed");
            return false;
        }
    }
    if (vrows != nullptr) {
        // the stress rows' large groups pass FP16's range: the prompt attention (FP16 operands) is out of contract;
        // its outputs are dumped for the byte comparison only, the write checked alone
        return kbad == 0 && vbad == 0;
    }
    const int32_t *     offs = reinterpret_cast<const int32_t *>(bodies.data() + qsa::prompt_body_words);
    std::vector<double> ref(gout.size());
    parallel_for(n, [&](int64_t t) {
        const int32_t pos = pos0 + static_cast<int32_t>(t);
        const int32_t s = static_cast<int32_t>(t) / qsa::prompt_slab, rr = static_cast<int32_t>(t) % qsa::prompt_slab;
        const uint32_t * b =
            bodies.data() + int64_t(s) * qsa::prompt_slab_words + offs[s * (qsa::prompt_slab + 1) + rr] / 4;
        std::vector<int32_t> sel;
        for (uint32_t i = 0; i < b[0]; ++i) {
            for (int32_t q = 0; q < 4; ++q) {
                sel.push_back(4 * static_cast<int32_t>(b[2 + i]) + q);
            }
        }
        for (uint32_t i = 0; i < b[1]; ++i) {
            sel.push_back(static_cast<int32_t>(b[2 + b[0] + i]));
        }
        for (int32_t hh = 0; hh < NH; ++hh) {
            double q[HD];
            norm_rope_ref(qfull.data() + (size_t(t) * NH + hh) * QS, qn.data(), pos, ts, q);
            std::vector<double> sc(sel.size());
            double              mx = -HUGE_VAL;
            for (size_t c = 0; c < sel.size(); ++c) {
                double d = 0.0;
                for (int32_t e = 0; e < HD; ++e) {
                    d += q[e] / 16.0 * g.kval(sel[c], e);
                }
                sc[c] = d;
                mx    = std::max(mx, d);
            }
            double l = 0.0;
            for (double & x : sc) {
                x = std::exp(x - mx);
                l += x;
            }
            for (int32_t e = 0; e < HD; ++e) {
                double o = 0.0;
                for (size_t c = 0; c < sel.size(); ++c) {
                    o += sc[c] * g.vval(sel[c], e);
                }
                const double gate = qfull[(size_t(t) * NH + hh) * QS + HD + e];
                ref[(size_t(t) * NH + hh) * HD + e] = o / l / (1.0 + std::exp(-gate));
            }
        }
    });
    const double ea = unit_error(gout.data(), ref.data(), int64_t(n) * NH, HD);
    const bool same = std::memcmp(gout.data(), gout2.data(), gout.size() * 4) == 0;
    // PB3 Amendment 2: a loose bug check (FP16 operands), the error printed on a failure only
    const bool loose = ea <= 2e-2;
    const bool ok = loose && same && ew[0] == 0;
    v.put(std::string(tag) + "attention rows " + std::to_string(pos0) + " + " + std::to_string(n), ok,
          std::string("against FP64 ") + (loose ? "within" : "OUTSIDE") +
              " the loose bug-check tolerance (2e-2 a row and head)" + (loose ? "" : " [" + e3(ea) + "]") + "; rerun " +
              (same ? "bitwise" : "DIFFERS") + "; error word " + std::to_string(ew[0]));
    (void) cells;
    return ok;
}

}  // namespace

int fnfl_qsp_test(int device, const char * mode, bool q8) {
    const bool san = mode != nullptr && std::strcmp(mode, "san") == 0;
    verdict    v("qsp-test");
    printf("qsp-test: KV format %s\n", q8 ? "q8 (KV8)" : "bf16");
    for (const int32_t which : {1, 2, 3, 4}) {
        qsa::qsp_resources rs{};
        if (qsa::qsp_kernel_resources(which, rs)) {
            printf("qsp-test: %s: %d registers, %d B local, %d B shared\n",
                   which == 1   ? "attend (bf16)"
                   : which == 2 ? "kv (bf16)"
                   : which == 3 ? "attend (q8)"
                                : "kv (q8)",
                   rs.regs, rs.local_bytes, rs.shared_bytes);
        }
    }
    if (!cuda_ok(cudaSetDevice(device), "device")) {
        v.put("setup", false, "a launch or copy failed");
        return v.finish();
    }
    fixture fx;
    rng     r(0x51a);
    fx.keys.resize(size_t(fx.cells) * KD);
    for (uint16_t & k : fx.keys) {
        k = fn::quant::f32_to_bf16(static_cast<float>(r.normal()));
    }
    std::vector<float> kn(KD), qn(KD);
    for (float & f : kn) {
        f = static_cast<float>(r.uniform(0.5, 1.5));
    }
    for (float & f : qn) {
        f = static_cast<float>(r.uniform(0.5, 1.5));
    }
    // the raw keys: the prompt write (bitwise bf16 of the F32 keys) over the first cells, the rest as stored
    std::vector<float> ik(size_t(1536) * KD);
    for (size_t i = 0; i < ik.size(); ++i) {
        ik[i] = static_cast<float>(r.normal());
    }
    dbuf                  dik(ik.size() * 4);
    std::vector<uint16_t> kback;
    if (!fx.dkeys.put(fx.keys) || !fx.dkn.put(kn) || !fx.dqn.put(qn) || !dik.put(ik) ||
        !qsa::qsp_keys(1536, 4000, dik.as<float>(), fx.dkeys.as<uint16_t>(), fx.cells, fn::ptest::test_stream()) ||
        !cuda_ok(cudaDeviceSynchronize(), "keys") || !fx.dkeys.get(kback, fx.keys.size())) {
        v.put("setup", false, "a launch or copy failed");
        return v.finish();
    }
    int64_t kbad = 0;
    for (int64_t i = 0; i < int64_t(1536) * KD; ++i) {
        kbad += kback[size_t(4000) * KD + size_t(i)] == fn::quant::f32_to_bf16(ik[size_t(i)]) ? 0 : 1;
        fx.keys[size_t(4000) * KD + size_t(i)] = kback[size_t(4000) * KD + size_t(i)];
    }
    v.put("raw keys", kbad == 0, std::to_string(kbad) + " of " + std::to_string(1536 * KD) + " not bf16(ik)");
    // the summaries of every block by QA1b's cache form (the prompt path's pooling)
    qsa::update_args ua;
    ua.keys      = fx.dkeys.as<uint16_t>();
    ua.k_norm    = fx.dkn.as<float>();
    ua.summaries = fx.dsum.as<float>();
    ua.first     = 0;
    ua.count     = fx.cells / 4;
    ua.rope      = qsa::model_rope();
    if (!qsa::summary_update(ua, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "summaries")) {
        v.put("setup", false, "a launch or copy failed");
        return v.finish();
    }
    // KV8's quantizer stress rows (fn/quant/kv_q8_ref.h) as the rows' v: in q8 the codes and scales bitwise the CPU
    // quantizer's, in BF16 bitwise bf16(v)
    const std::vector<float> stress = fn::quant::kv_q8_stress_rows(0x6b5638ull, san ? 0 : 8);
    const int32_t n_stress = static_cast<int32_t>(stress.size() / HD);
    if (san) {
        select_case(fx, 2040, 20, v);
        kv_attend_case(fx, 2100, 8, v, q8);
        kv_attend_case(fx, 3000, n_stress, v, q8, stress.data(), "quantizer stress: ");
    } else {
        select_case(fx, 1990, 130, v);   // rows crossing 512 visible blocks (position 2,047)
        select_case(fx, 8000, 300, v);   // three slabs, deep rows
        select_case(fx, 15000, 129, v);  // a slab and one row
        kv_attend_case(fx, 1000, 64, v, q8);
        kv_attend_case(fx, 9000, 130, v, q8);
        kv_attend_case(fx, 3000, n_stress, v, q8, stress.data(), "quantizer stress: ");
    }
    // deterministic ties and pathological inputs: cells 0 .. 4,095 rewritten, the summaries recomputed, the selection
    // against the decode select. Ties: every key the same value, so every block's summary and score tie. Pathological:
    // blocks in turn all zero, one-hot at 1e19 (its squares near F32's top), the smallest BF16 subnormal, and -0.0
    const auto rewrite = [&](int32_t which) {
        for (int64_t c = 0; c < 4096; ++c) {
            const int64_t b = c / 4;
            for (int32_t e = 0; e < KD; ++e) {
                uint16_t k = fn::quant::f32_to_bf16(0.25f);
                if (which == 1) {
                    const int64_t kind = b % 4;
                    k                  = kind == 0 ? uint16_t(0) :
                                         kind == 1 ? (e == b % KD ? fn::quant::f32_to_bf16(1e19f) : uint16_t(0)) :
                                         kind == 2 ? uint16_t(0x0001) :
                                                     uint16_t(0x8000);
                }
                fx.keys[size_t(c) * KD + size_t(e)] = k;
            }
        }
        ua.count = fx.cells / 4;
        return fx.dkeys.put(fx.keys) && qsa::summary_update(ua, fn::ptest::test_stream()) &&
               cuda_ok(cudaDeviceSynchronize(), "summaries");
    };
    if (!rewrite(0)) {
        v.put("setup", false, "a launch or copy failed");
        return v.finish();
    }
    select_case(fx, 2040, san ? 12 : 130, v, "ties: "); // FP16 scores: near-zero relu terms may flip (Amendment 2)
    if (!rewrite(1)) {
        v.put("setup", false, "a launch or copy failed");
        return v.finish();
    }
    select_case(fx, 3000, san ? 12 : 200, v, "pathological: ");
    return v.finish();
}
