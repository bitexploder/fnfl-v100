#include "fn/kernels/hc/hc_prompt_test.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/hc/hc_prompt.h"
#include "fn/quant/formats.h"
#include "fn/tools/prompt_test_util.h"
#include "fn/xchg/prompt_xchg.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace fn::ptest;
constexpr int32_t D = 2560, F = 1280, J = 324, NL = 320;

// full-width HC coefficients (BF16 bits and their values), hc_test's distributions
struct hcw {
    std::vector<uint16_t> down, up;  // [4][324][2560], [2560][4][320]
    std::vector<float>    gamma;     // [4][2560]
    std::vector<double>   downv, upv;
};

void make_w(uint64_t seed, hcw & w) {
    rng g(seed);
    w.down.resize(size_t(4) * J * D);
    w.downv.resize(w.down.size());
    for (size_t i = 0; i < w.down.size(); ++i) {
        w.down[i]  = fn::quant::f32_to_bf16(static_cast<float>(g.uniform(-0.07, 0.07)));
        w.downv[i] = fn::quant::bf16_to_f32(w.down[i]);
    }
    w.up.resize(size_t(D) * 4 * NL);
    w.upv.resize(w.up.size());
    for (size_t i = 0; i < w.up.size(); ++i) {
        w.up[i]  = fn::quant::f32_to_bf16(static_cast<float>(g.uniform(-0.25, 0.25)));
        w.upv[i] = fn::quant::bf16_to_f32(w.up[i]);
    }
    w.gamma.resize(size_t(4) * D);
    for (float & v : w.gamma) {
        v = static_cast<float>(g.uniform(0.5, 1.5));
    }
}

// the streams: U(-2, 2); every third row has stream 1 scaled by 1e3 and stream 2 by 1e-3 (hc_test's range case)
void make_x(int32_t rows, uint64_t seed, std::vector<float> & x) {
    rng g(seed);
    x.resize(size_t(rows) * 4 * D);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t s = 0; s < 4; ++s) {
            const double sc = r % 3 == 2 ? (s == 1 ? 1e3 : s == 2 ? 1e-3 : 1.0) : 1.0;
            for (int32_t i = 0; i < D; ++i) {
                x[(size_t(r) * 4 + s) * D + i] = static_cast<float>(g.uniform(-2.0, 2.0) * sc);
            }
        }
    }
}

double sig(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// hc.h's formulas in FP64 over the full width: mixed [rows][2560], inject [rows][4]; optionally the down products u
// [rows][4][324]
void reference(const hcw &                w,
               const std::vector<float> & x,
               int32_t                    rows,
               std::vector<double> &      mixed,
               std::vector<double> &      inject,
               std::vector<double> *      u_out = nullptr) {
    mixed.assign(size_t(rows) * D, 0.0);
    inject.assign(size_t(rows) * 4, 0.0);
    if (u_out != nullptr) {
        u_out->assign(size_t(rows) * 4 * J, 0.0);
    }
    parallel_for(rows, [&](int64_t r) {
        double              v[4] = { 0, 0, 0, 0 }, rms[4];
        std::vector<double> u(size_t(4) * J, 0.0);
        for (int32_t s = 0; s < 4; ++s) {
            const float * xs = x.data() + (size_t(r) * 4 + s) * D;
            for (int32_t i = 0; i < D; ++i) {
                v[s] += double(xs[i]) * xs[i];
            }
            for (int32_t j = 0; j < J; ++j) {
                double         a  = 0.0;
                const double * wr = w.downv.data() + (size_t(s) * J + j) * D;
                for (int32_t i = 0; i < D; ++i) {
                    a += wr[i] * double(w.gamma[size_t(s) * D + i]) * xs[i];
                }
                u[size_t(s) * J + j] = a;
            }
            rms[s] = std::sqrt(v[s] / D + 1e-6);
        }
        if (u_out != nullptr) {
            std::copy(u.begin(), u.end(), u_out->begin() + r * 4 * J);
        }
        std::vector<double> low(NL);
        for (int32_t j = 0; j < J; ++j) {
            double z = 0.0;
            for (int32_t s = 0; s < 4; ++s) {
                z += u[size_t(s) * J + j] / rms[s];
            }
            z *= 0.25;
            if (j < NL) {
                low[size_t(j)] = z * sig(z);
            } else {
                inject[size_t(r) * 4 + (j - NL)] = 2.0 * sig(z);
            }
        }
        for (int32_t c = 0; c < D; ++c) {
            double m = 0.0;
            for (int32_t s = 0; s < 4; ++s) {
                double         zs = 0.0;
                const double * ur = w.upv.data() + (size_t(c) * 4 + s) * NL;
                for (int32_t j = 0; j < NL; ++j) {
                    zs += ur[j] * low[size_t(j)];
                }
                const double xv = x[(size_t(r) * 4 + s) * D + c];
                m += double(w.gamma[size_t(s) * D + c]) * xv / rms[s] * sig(zs);
            }
            mixed[size_t(r) * D + c] = 0.25 * m;
        }
    });
}

// the card halves in the decode layout
void halves(const hcw & w,
            std::vector<uint16_t> (&dn)[2],
            std::vector<uint16_t> (&up)[2],
            std::vector<float> (&gm)[2]) {
    for (int32_t h = 0; h < 2; ++h) {
        dn[h].resize(size_t(4) * J * F);
        up[h].resize(size_t(F) * 4 * NL);
        gm[h].resize(size_t(4) * F);
        for (int32_t s = 0; s < 4; ++s) {
            for (int32_t j = 0; j < J; ++j) {
                std::memcpy(&dn[h][(size_t(s) * J + j) * F], &w.down[(size_t(s) * J + j) * D + size_t(h) * F],
                            size_t(F) * 2);
            }
            std::memcpy(&gm[h][size_t(s) * F], &w.gamma[size_t(s) * D + size_t(h) * F], size_t(F) * 4);
        }
        std::memcpy(up[h].data(), &w.up[size_t(h) * F * 4 * NL], up[h].size() * 2);
    }
}

bool replicated_case(int32_t rows, verdict & v, const hcw & w) {
    std::vector<uint16_t> dn[2], up[2];
    std::vector<float>    gm[2];
    halves(w, dn, up, gm);
    std::vector<float> x;
    make_x(rows, 0x4c0 + uint64_t(rows), x);
    dbuf ddn0(dn[0].size() * 2), ddn1(dn[1].size() * 2), dup0(up[0].size() * 2), dup1(up[1].size() * 2),
        dg0(gm[0].size() * 4), dg1(gm[1].size() * 4), dx(x.size() * 4), mom(size_t(rows) * fn::hc::hcp_mom * 4),
        low(size_t(rows) * NL * 4), rinv(size_t(rows) * 16), inj(size_t(rows) * 16), mixed(size_t(rows) * D * 4),
        m16(size_t(rows) * D * 2), err(256);
    if (!ddn0.put(dn[0]) || !ddn1.put(dn[1]) || !dup0.put(up[0]) || !dup1.put(up[1]) || !dg0.put(gm[0]) ||
        !dg1.put(gm[1]) || !dx.put(x) || !mixed.poison() || !inj.poison() || !m16.poison() || !err.zero()) {
        return false;
    }
    fn::hc::hcp_args a;
    a.w.form     = fn::hc::hcp_form::replicated;
    a.w.down[0]  = ddn0.as<uint16_t>();
    a.w.down[1]  = ddn1.as<uint16_t>();
    a.w.up[0]    = dup0.as<uint16_t>();
    a.w.up[1]    = dup1.as<uint16_t>();
    a.w.gamma[0] = dg0.as<float>();
    a.w.gamma[1] = dg1.as<float>();
    a.rows       = rows;
    a.x          = dx.as<float>();
    a.mom        = mom.as<float>();
    a.low        = low.as<float>();
    a.rinv       = rinv.as<float>();
    a.inject     = inj.as<float>();
    a.mixed      = mixed.as<float>();
    a.ld_mixed   = D;
    a.mixed16    = m16.as<uint16_t>();
    a.ld_mixed16 = D;
    a.err        = err.as<uint32_t>();
    std::vector<float>    gm1, gi1, gm2, gi2;
    std::vector<uint16_t> g16;
    std::vector<uint32_t> ew;
    const auto            run = [&](std::vector<float> & om, std::vector<float> & oi) {
        return fn::hc::hcp_down(a, fn::ptest::test_stream()) && fn::hc::hcp_mid(a, fn::ptest::test_stream()) &&
               fn::hc::hcp_up(a, fn::ptest::test_stream()) && cuda_ok(cudaDeviceSynchronize(), "hcp") &&
               mixed.get(om, size_t(rows) * D) && inj.get(oi, size_t(rows) * 4);
    };
    const std::string name = "replicated " + std::to_string(rows) + " rows";
    if (!run(gm1, gi1) || !m16.get(g16, size_t(rows) * D) || !err.get(ew, 1) || !run(gm2, gi2)) {
        v.put(name, false, "launch failed");
        return false;
    }
    std::vector<double> rm, ri;
    reference(w, x, rows, rm, ri);
    const double em = unit_error(gm1.data(), rm.data(), rows, D), ei = unit_error(gi1.data(), ri.data(), rows, 4);
    bool         f16 = true;
    for (size_t i = 0; i < g16.size(); ++i) {
        f16 = f16 && g16[i] == fn::quant::f32_to_f16(gm1[i]);
    }
    const bool same = std::memcmp(gm1.data(), gm2.data(), gm1.size() * 4) == 0 &&
                      std::memcmp(gi1.data(), gi2.data(), gi1.size() * 4) == 0;
    const bool ok   = em <= 1e-6 && ei <= 1e-6 && f16 && same && ew[0] == 0;
    v.put(name, ok,
          "mixed " + e3(em) + ", inject " + e3(ei) + " (bound 1e-6 a row); FP16 copy " + (f16 ? "bitwise" : "DIFFERS") +
              "; rerun " + (same ? "bitwise" : "DIFFERS") + "; error word " + std::to_string(ew[0]));
    return ok;
}

// PB3: the tensor path (hct_convert + hct_mix) on x [rows][4][2560] against FP64 at a loose tolerance (Amendment 2: a
// check for layout, indexing and range errors, not a precision bound); NaN or Inf fails; the FP16 copy and a rerun
// bitwise.
// The tensor path against FP64. linear: only the down products u (mom) are compared, mixed and inject only checked
// finite: coefficient rows at 2^20 make |g| ~ 1e6, where sigmoid is a step and FP16's rounding of a near-zero g flips it
// (an arithmetic property of such rows, not a layout error). The errors are printed on a failure only.
bool tensor_case(const std::string &        name,
                 const hcw &                w,
                 const std::vector<float> & x,
                 int32_t                    rows,
                 verdict &                  v,
                 bool                       linear = false) {
    std::vector<uint16_t> dn[2], up[2];
    std::vector<float>    gm[2];
    halves(w, dn, up, gm);
    dbuf ddn0(dn[0].size() * 2), ddn1(dn[1].size() * 2), dup0(up[0].size() * 2), dup1(up[1].size() * 2),
        dg0(gm[0].size() * 4), dg1(gm[1].size() * 4), dx(x.size() * 4), mom(size_t(rows) * fn::hc::hcp_mom * 4),
        low(size_t(rows) * NL * 4), rinv(size_t(rows) * 16), inj(size_t(rows) * 16), mixed(size_t(rows) * D * 4),
        m16(size_t(rows) * D * 2), err(256), w16(static_cast<size_t>(fn::hc::hct_weight_bytes())),
        scr(static_cast<size_t>(fn::hc::hct_scratch_bytes(rows)));
    if (!ddn0.put(dn[0]) || !ddn1.put(dn[1]) || !dup0.put(up[0]) || !dup1.put(up[1]) || !dg0.put(gm[0]) ||
        !dg1.put(gm[1]) || !dx.put(x) || !err.zero() || !w16.poison() || !scr.poison()) {
        return false;
    }
    fn::hc::hcp_args a;
    a.w.form                     = fn::hc::hcp_form::replicated;
    a.w.down[0]                  = ddn0.as<uint16_t>();
    a.w.down[1]                  = ddn1.as<uint16_t>();
    a.w.up[0]                    = dup0.as<uint16_t>();
    a.w.up[1]                    = dup1.as<uint16_t>();
    a.w.gamma[0]                 = dg0.as<float>();
    a.w.gamma[1]                 = dg1.as<float>();
    a.rows                       = rows;
    a.x                          = dx.as<float>();
    a.mom                        = mom.as<float>();
    a.low                        = low.as<float>();
    a.rinv                       = rinv.as<float>();
    a.inject                     = inj.as<float>();
    a.mixed                      = mixed.as<float>();
    a.ld_mixed                   = D;
    a.mixed16                    = m16.as<uint16_t>();
    a.ld_mixed16                 = D;
    a.err                        = err.as<uint32_t>();
    const fn::hc::hct_weights hw = fn::hc::hct_carve(w16.as<uint8_t>());
    std::vector<float>        tm1, ti1, tm2, ti2, tmom;
    std::vector<uint16_t>     g16;
    std::vector<uint32_t>     ew;
    const auto                run = [&](std::vector<float> & om, std::vector<float> & oi) {
        return mixed.poison() && inj.poison() && m16.poison() &&
               fn::hc::hct_convert(a.w, hw, fn::ptest::test_stream()) &&
               fn::hc::hct_mix(a, fn::hc::hct_combine{}, hw, scr.as<uint8_t>(), fn::ptest::test_stream()) &&
               cuda_ok(cudaDeviceSynchronize(), "hct") && mixed.get(om, size_t(rows) * D) &&
               inj.get(oi, size_t(rows) * 4);
    };
    if (!run(tm1, ti1) || !m16.get(g16, size_t(rows) * D) || !err.get(ew, 1) ||
        !mom.get(tmom, size_t(rows) * fn::hc::hcp_mom) || !run(tm2, ti2)) {
        v.put(name, false, "launch failed");
        return false;
    }
    std::vector<double> rm, ri, ru;
    reference(w, x, rows, rm, ri, &ru);
    // u per (row, stream) from mom's layout (u[s][j] at 4 + 320 s + j, the injection rows' q[s][k] at 1,284 + 4 s + k)
    std::vector<float> gu(ru.size());
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t st = 0; st < 4; ++st) {
            for (int32_t j = 0; j < J; ++j) {
                gu[(size_t(r) * 4 + st) * J + j] =
                    tmom[size_t(r) * fn::hc::hcp_mom + (j < NL ? 4 + NL * st + j : 4 + 4 * NL + 4 * st + (j - NL))];
            }
        }
    }
    const double eu = unit_error(gu.data(), ru.data(), int64_t(rows) * 4, J);
    const double em = unit_error(tm1.data(), rm.data(), rows, D), ei = unit_error(ti1.data(), ri.data(), rows, 4);
    const bool   loose = eu <= 2e-2 && (linear ? std::isfinite(em) && std::isfinite(ei) : em <= 2e-2 && ei <= 2e-2);
    bool         f16   = true;
    for (size_t i = 0; i < g16.size(); ++i) {
        f16 = f16 && g16[i] == fn::quant::f32_to_f16(tm1[i]);
    }
    const bool same = std::memcmp(tm1.data(), tm2.data(), tm1.size() * 4) == 0 &&
                      std::memcmp(ti1.data(), ti2.data(), ti1.size() * 4) == 0;
    const bool ok   = loose && f16 && same && ew[0] == 0;
    v.put(name, ok,
          std::string("tensor: ") + (linear ? "u (mom); mixed and inject finite" : "u (mom), mixed and inject") + " " +
              (loose ? "within" : "OUTSIDE") + " the loose bug-check tolerance (2e-2 a row)" +
              (loose ? "" : " [u " + e3(eu) + ", mixed " + e3(em) + ", inject " + e3(ei) + "]") + "; FP16 copy " +
              (f16 ? "bitwise" : "DIFFERS") + "; rerun " + (same ? "bitwise" : "DIFFERS") + "; error word " +
              std::to_string(ew[0]));
    return ok;
}

// coefficient range stress: rows of down and up scaled by powers of two that FP16 cannot hold unscaled (2^20, 2^-30),
// so the per-row range handling carries them; small: the up rows' 2^-26 only (sigmoid stays smooth: the full check)
void stress_w(const hcw & w0, hcw & w, bool small) {
    w = w0;
    for (int32_t s = 0; s < 4 && !small; ++s) {
        for (int32_t j = 0; j < J; ++j) {
            const int32_t k  = (s * J + j) % 7;
            const double  sc = k == 0 ? std::ldexp(1.0, 20) : k == 1 ? std::ldexp(1.0, -30) : 1.0;
            for (int32_t i = 0; i < D; ++i) {
                const size_t at = (size_t(s) * J + j) * D + i;
                w.down[at]      = fn::quant::f32_to_bf16(static_cast<float>(w.downv[at] * sc));
                w.downv[at]     = fn::quant::bf16_to_f32(w.down[at]);
            }
        }
    }
    for (int32_t r = 0; r < 4 * D; r += 5) {
        const double sc = r % 2 == 0 && !small ? std::ldexp(1.0, 18) : std::ldexp(1.0, -26);
        for (int32_t j = 0; j < NL; ++j) {
            const size_t at = size_t(r) * NL + j;
            w.up[at]        = fn::quant::f32_to_bf16(static_cast<float>(w.upv[at] * sc));
            w.upv[at]       = fn::quant::bf16_to_f32(w.up[at]);
        }
    }
}

// activation stress: make_x's rows plus a row with one element 1e4 among U(-2, 2), and a row of streams near zero
void stress_x(int32_t rows, std::vector<float> & x) {
    make_x(rows, 0x5a5a, x);
    for (int32_t s = 0; s < 4; ++s) {
        x[(size_t(1) * 4 + s) * D + 17 + s] = 1e4f;
        for (int32_t i = 0; i < D; ++i) {
            x[(size_t(4) * 4 + s) * D + i] *= 1e-20f;
        }
    }
}

// PB1's saved real inputs (manifest.tsv in dir): every (chunk, layer, call, half) group's x rows, with that layer's
// coefficients read on the CPU from the manifest's GGUF
bool real_cases(const std::string & dir, verdict & v) {
    std::ifstream mf(dir + "/manifest.tsv");
    std::string   line, model;

    struct group {
        int32_t     layer = 0, rows = 0;
        std::string call, x;
    };

    std::map<std::string, group> groups;
    while (std::getline(mf, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            const size_t at = line.rfind(' ');
            if (at != std::string::npos && line.size() > 5 && line.compare(line.size() - 5, 5, ".gguf") == 0) {
                model = line.substr(at + 1);
            }
            continue;
        }
        std::istringstream       ls(line);
        std::vector<std::string> f;
        std::string              c;
        while (std::getline(ls, c, '\t')) {
            f.push_back(c);
        }
        if (f.size() < 11 || f[0] == "file") {
            continue;
        }
        const std::string key = "c" + f[1] + "-L" + f[3] + "-" + f[4] + "-h" + f[5];
        group &           g   = groups[key];
        g.layer               = std::stoi(f[3]);
        g.call                = f[4];
        g.rows                = std::stoi(f[8]);
        if (f[0].find("-x.f32") != std::string::npos) {
            g.x = f[0];
        }
    }
    if (model.empty() || groups.empty()) {
        v.put("real inputs", false, "no manifest or model path in " + dir);
        return false;
    }
    std::vector<std::string> shards;
    std::string              gerr;
    if (!fnfl_gguf_shard_paths(model, shards, gerr)) {
        v.put("real inputs", false, gerr);
        return false;
    }
    std::vector<fnfl_gguf_dir> dirs(shards.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        if (!fnfl_gguf_read_dir(shards[i], dirs[i], gerr)) {
            v.put("real inputs", false, gerr);
            return false;
        }
    }
    const auto read_tensor = [&](const std::string & name, std::vector<uint8_t> & out) {
        for (size_t i = 0; i < dirs.size(); ++i) {
            for (const fnfl_gguf_tensor & t : dirs[i].tensors) {
                if (t.name == name) {
                    std::ifstream in(shards[i], std::ios::binary);
                    out.resize(static_cast<size_t>(t.bytes));
                    in.seekg(static_cast<std::streamoff>(t.offset));
                    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(t.bytes));
                    return static_cast<bool>(in);
                }
            }
        }
        return false;
    };
    const auto read_f32 = [&](const std::string & path, size_t n, std::vector<float> & out) {
        std::ifstream in(dir + "/" + path, std::ios::binary);
        out.resize(n);
        in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(n * 4));
        return static_cast<bool>(in);
    };
    std::map<std::string, hcw> weights;
    bool                       all = true;
    for (const auto & [key, g] : groups) {
        const std::string wk = std::to_string(g.layer) + g.call;
        if (weights.find(wk) == weights.end()) {
            const std::string    pre = "blk." + std::to_string(g.layer) + ".hc_" + g.call + "_";
            std::vector<uint8_t> dn, inj, nm, upb;
            if (!read_tensor(pre + "down.weight", dn) || !read_tensor(pre + "inject.weight", inj) ||
                !read_tensor(pre + "norm.weight", nm) || !read_tensor(pre + "up.weight", upb) ||
                dn.size() != size_t(NL) * 4 * D * 2 || inj.size() != size_t(4) * 4 * D * 2 ||
                nm.size() != size_t(4) * D * 4 || upb.size() != size_t(4) * D * NL * 2) {
                v.put("real inputs " + key, false, "the coefficients of " + pre + "* not read");
                all = false;
                continue;
            }
            hcw w;
            w.down.resize(size_t(4) * J * D);
            w.downv.resize(w.down.size());
            w.up.resize(size_t(D) * 4 * NL);
            w.upv.resize(w.up.size());
            w.gamma.resize(size_t(4) * D);
            const uint16_t * dn16 = reinterpret_cast<const uint16_t *>(dn.data());
            const uint16_t * in16 = reinterpret_cast<const uint16_t *>(inj.data());
            const uint16_t * up16 = reinterpret_cast<const uint16_t *>(upb.data());
            // GGUF: down row j (inject row j - 320) holds column s 2560 + i; up row s 2560 + c holds j; norm s 2560 + i
            for (int32_t s = 0; s < 4; ++s) {
                for (int32_t j = 0; j < J; ++j) {
                    for (int32_t i = 0; i < D; ++i) {
                        const uint16_t b                    = j < NL ? dn16[size_t(j) * 4 * D + size_t(s) * D + i] :
                                                                       in16[size_t(j - NL) * 4 * D + size_t(s) * D + i];
                        w.down[(size_t(s) * J + j) * D + i] = b;
                    }
                }
                for (int32_t cc = 0; cc < D; ++cc) {
                    std::memcpy(&w.up[(size_t(cc) * 4 + s) * NL], &up16[(size_t(s) * D + cc) * NL], size_t(NL) * 2);
                }
            }
            std::memcpy(w.gamma.data(), nm.data(), nm.size());
            for (size_t i = 0; i < w.down.size(); ++i) {
                w.downv[i] = fn::quant::bf16_to_f32(w.down[i]);
            }
            for (size_t i = 0; i < w.up.size(); ++i) {
                w.upv[i] = fn::quant::bf16_to_f32(w.up[i]);
            }
            weights.emplace(wk, std::move(w));
        }
        std::vector<float> x;
        if (g.x.empty() || !read_f32(g.x, size_t(g.rows) * 4 * D, x)) {
            v.put("real inputs " + key, false, "dump files not read");
            all = false;
            continue;
        }
        all =
            tensor_case("real " + key + " (" + std::to_string(g.rows) + " rows)", weights.at(wk), x, g.rows, v) && all;
    }
    return all;
}

// Q8_0 in hc.h's planar layouts, from the full-width values (the draft's split form), and the decoded values
void q8_of(const std::vector<double> & vals,
           int64_t                     rows,
           int64_t                     width,
           std::vector<uint8_t> &      out,
           std::vector<double> &       dec,
           uint64_t                    seed) {
    // planar: rows x width int8 codes, then rows x width / 32 FP16 scales
    rng g(seed);
    out.assign(size_t(rows * width + rows * (width / 32) * 2), 0);
    dec.assign(size_t(rows * width), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < width / 32; ++b) {
            double amax = 0.0;
            for (int64_t i = 0; i < 32; ++i) {
                amax = std::max(amax, std::fabs(vals[size_t(r * width + b * 32 + i)]));
            }
            const uint16_t dh = fn::quant::f32_to_f16(static_cast<float>(amax / 127.0 * g.uniform(0.9, 1.1) + 1e-6));
            const double   d  = fn::quant::f16_to_f32(dh);
            std::memcpy(&out[size_t(rows * width + (r * (width / 32) + b) * 2)], &dh, 2);
            for (int64_t i = 0; i < 32; ++i) {
                const int64_t at = r * width + b * 32 + i;
                const int32_t q =
                    static_cast<int32_t>(std::max(-127.0, std::min(127.0, std::nearbyint(vals[size_t(at)] / d))));
                out[size_t(at)] = static_cast<uint8_t>(static_cast<int8_t>(q));
                dec[size_t(at)] = double(static_cast<float>(d) * static_cast<float>(q));
            }
        }
    }
}

bool split_case(int32_t rows, verdict & v, const hcw & w0) {
    // the draft's form: each card's Q8_0 half; the reference is the full-width formula on the decoded weights
    hcw                  w = w0;
    std::vector<uint8_t> qd[2], qu[2];
    std::vector<double>  full_down(size_t(4) * J * D), full_up(size_t(D) * 4 * NL);
    for (int32_t h = 0; h < 2; ++h) {
        std::vector<double> dv(size_t(4) * J * F), uv(size_t(F) * 4 * NL), ddec, udec;
        for (int32_t s = 0; s < 4; ++s) {
            for (int32_t j = 0; j < J; ++j) {
                for (int32_t i = 0; i < F; ++i) {
                    dv[(size_t(s) * J + j) * F + i] = w.downv[(size_t(s) * J + j) * D + size_t(h) * F + i];
                }
            }
        }
        std::memcpy(uv.data(), &w.upv[size_t(h) * F * 4 * NL], uv.size() * 8);
        q8_of(dv, int64_t(4) * J, F, qd[h], ddec, 0x51 + uint64_t(h));
        q8_of(uv, int64_t(F) * 4, NL, qu[h], udec, 0x61 + uint64_t(h));
        for (int32_t s = 0; s < 4; ++s) {
            for (int32_t j = 0; j < J; ++j) {
                for (int32_t i = 0; i < F; ++i) {
                    full_down[(size_t(s) * J + j) * D + size_t(h) * F + i] = ddec[(size_t(s) * J + j) * F + i];
                }
            }
        }
        std::memcpy(&full_up[size_t(h) * F * 4 * NL], udec.data(), udec.size() * 8);
    }
    w.downv = full_down;
    w.upv   = full_up;
    std::vector<float> x;
    make_x(rows, 0x7a0 + uint64_t(rows), x);
    std::vector<double> rm, ri;
    reference(w, x, rows, rm, ri);
    std::vector<float> xh[2], gh[2];
    for (int32_t h = 0; h < 2; ++h) {
        xh[h].resize(size_t(rows) * 4 * F);
        gh[h].resize(size_t(4) * F);
        for (int32_t r = 0; r < rows; ++r) {
            for (int32_t s = 0; s < 4; ++s) {
                std::memcpy(&xh[h][(size_t(r) * 4 + s) * F], &x[(size_t(r) * 4 + s) * D + size_t(h) * F],
                            size_t(F) * 4);
            }
        }
        for (int32_t s = 0; s < 4; ++s) {
            std::memcpy(&gh[h][size_t(s) * F], &w.gamma[size_t(s) * D + size_t(h) * F], size_t(F) * 4);
        }
    }

    // both "cards" on this device: down on each, then mid with the other's moments, then up
    struct card {
        std::unique_ptr<dbuf> qd, qu, g, x, mom, low, rinv, inj, mixed;
    } cd[2];

    fn::hc::hcp_args a[2];
    for (int32_t h = 0; h < 2; ++h) {
        cd[h].qd    = std::make_unique<dbuf>(qd[h].size());
        cd[h].qu    = std::make_unique<dbuf>(qu[h].size());
        cd[h].g     = std::make_unique<dbuf>(gh[h].size() * 4);
        cd[h].x     = std::make_unique<dbuf>(xh[h].size() * 4);
        cd[h].mom   = std::make_unique<dbuf>(size_t(rows) * fn::hc::hcp_mom * 4);
        cd[h].low   = std::make_unique<dbuf>(size_t(rows) * NL * 4);
        cd[h].rinv  = std::make_unique<dbuf>(size_t(rows) * 16);
        cd[h].inj   = std::make_unique<dbuf>(size_t(rows) * 16);
        cd[h].mixed = std::make_unique<dbuf>(size_t(rows) * F * 4);
        if (!cd[h].qd->put(qd[h]) || !cd[h].qu->put(qu[h]) || !cd[h].g->put(gh[h]) || !cd[h].x->put(xh[h]) ||
            !cd[h].mixed->poison()) {
            return false;
        }
        a[h].w.form      = fn::hc::hcp_form::split;
        a[h].w.q8_down   = cd[h].qd->as<uint8_t>();
        a[h].w.q8_up     = cd[h].qu->as<uint8_t>();
        a[h].w.gamma_own = cd[h].g->as<float>();
        a[h].rows        = rows;
        a[h].x           = cd[h].x->as<float>();
        a[h].mom         = cd[h].mom->as<float>();
        a[h].card        = h;
        a[h].low         = cd[h].low->as<float>();
        a[h].rinv        = cd[h].rinv->as<float>();
        a[h].inject      = cd[h].inj->as<float>();
        a[h].mixed       = cd[h].mixed->as<float>();
        a[h].ld_mixed    = F;
    }
    a[0].mom_peer = a[1].mom;
    a[1].mom_peer = a[0].mom;
    bool ok = fn::hc::hcp_down(a[0], fn::ptest::test_stream()) && fn::hc::hcp_down(a[1], fn::ptest::test_stream()) &&
              fn::hc::hcp_mid(a[0], fn::ptest::test_stream()) && fn::hc::hcp_mid(a[1], fn::ptest::test_stream()) &&
              fn::hc::hcp_up(a[0], fn::ptest::test_stream()) && fn::hc::hcp_up(a[1], fn::ptest::test_stream()) &&
              cuda_ok(cudaDeviceSynchronize(), "hcp split");
    std::vector<float> mh[2], ih[2];
    ok = ok && cd[0].mixed->get(mh[0], size_t(rows) * F) && cd[1].mixed->get(mh[1], size_t(rows) * F) &&
         cd[0].inj->get(ih[0], size_t(rows) * 4) && cd[1].inj->get(ih[1], size_t(rows) * 4);
    const std::string name = "split Q8_0 " + std::to_string(rows) + " rows";
    if (!ok) {
        v.put(name, false, "launch failed");
        return false;
    }
    std::vector<float> full(size_t(rows) * D);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t h = 0; h < 2; ++h) {
            std::memcpy(&full[size_t(r) * D + size_t(h) * F], &mh[h][size_t(r) * F], size_t(F) * 4);
        }
    }
    const double em = unit_error(full.data(), rm.data(), rows, D), ei = unit_error(ih[0].data(), ri.data(), rows, 4);
    const bool   same_inj = std::memcmp(ih[0].data(), ih[1].data(), ih[0].size() * 4) == 0;
    const bool   pass     = em <= 1e-5 && ei <= 1e-5 && same_inj;
    v.put(name, pass,
          "mixed " + e3(em) + ", inject " + e3(ei) + " (bound 1e-5 a row, FC-DR1); both cards' inject " +
              (same_inj ? "bitwise equal" : "DIFFER"));
    return pass;
}

// ppack and pcombine bitwise against the CPU, and a wrong generation refused
bool exchange_case(int32_t rows, verdict & v) {
    rng                g(0xe7);
    std::vector<float> part[2], inj(size_t(rows) * 4), S(size_t(rows) * 4 * D);
    for (int32_t c = 0; c < 2; ++c) {
        part[c].resize(size_t(rows) * D);
        for (float & f : part[c]) {
            f = static_cast<float>(g.normal() * 3.0);
        }
    }
    for (float & f : inj) {
        f = static_cast<float>(g.uniform(0.0, 2.0));
    }
    for (float & f : S) {
        f = static_cast<float>(g.normal());
    }
    const size_t slot = static_cast<size_t>(fn::xchg::pmsg_bytes(int64_t(rows) * D * 2));
    dbuf         dp0(part[0].size() * 4), dp1(part[1].size() * 4), s0{ slot }, s1{ slot }, di(inj.size() * 4),
        dS0(S.size() * 4), dS1(S.size() * 4), err(256);
    const uint64_t gen = fn::xchg::pgen(5, 7, 1, 1, fn::xchg::pkind::ffn);
    bool           ok =
        dp0.put(part[0]) && dp1.put(part[1]) && di.put(inj) && dS0.put(S) && dS1.put(S) && err.zero() &&
        fn::xchg::ppack(rows, dp0.as<float>(), s0.as<uint8_t>(), gen, err.as<uint32_t>(), fn::ptest::test_stream()) &&
        fn::xchg::ppack(rows, dp1.as<float>(), s1.as<uint8_t>(), gen, err.as<uint32_t>(), fn::ptest::test_stream()) &&
        // card 0: own slot 0, peer slot 1; card 1: own slot 1, peer slot 0
        fn::xchg::pcombine(rows, 0, s0.as<uint8_t>(), s1.as<uint8_t>(), gen, di.as<float>(), dS0.as<float>(),
                           err.as<uint32_t>(), fn::ptest::test_stream()) &&
        fn::xchg::pcombine(rows, 1, s1.as<uint8_t>(), s0.as<uint8_t>(), gen, di.as<float>(), dS1.as<float>(),
                           err.as<uint32_t>(), fn::ptest::test_stream()) &&
        cuda_ok(cudaDeviceSynchronize(), "exchange");
    std::vector<float>    g0, g1;
    std::vector<uint32_t> ew;
    ok                      = ok && dS0.get(g0, S.size()) && dS1.get(g1, S.size()) && err.get(ew, 1);
    std::vector<float> want = S;
    for (int32_t r = 0; r < rows && ok; ++r) {
        for (int32_t c = 0; c < D; ++c) {
            const float a = fn::quant::f16_to_f32(fn::quant::f32_to_f16(part[0][size_t(r) * D + c]));
            const float b = fn::quant::f16_to_f32(fn::quant::f32_to_f16(part[1][size_t(r) * D + c]));
            const float o = a + b;
            for (int32_t s = 0; s < 4; ++s) {
                float & x = want[(size_t(r) * 4 + s) * D + c];
                x         = std::fma(inj[size_t(r) * 4 + s], o, x);
            }
        }
    }
    const bool            b0 = ok && std::memcmp(g0.data(), want.data(), want.size() * 4) == 0;
    const bool            b1 = ok && std::memcmp(g1.data(), want.data(), want.size() * 4) == 0;
    // a stale generation: nothing written, the error bit set
    std::vector<float>    g2;
    std::vector<uint32_t> ew2;
    const bool refused = ok && err.zero() && dS0.put(S) &&
                         fn::xchg::pcombine(rows, 0, s0.as<uint8_t>(), s1.as<uint8_t>(), gen + 1, di.as<float>(),
                                            dS0.as<float>(), err.as<uint32_t>(), fn::ptest::test_stream()) &&
                         cuda_ok(cudaDeviceSynchronize(), "exchange") && dS0.get(g2, S.size()) && err.get(ew2, 1) &&
                         std::memcmp(g2.data(), S.data(), S.size() * 4) == 0 && (ew2[0] & fn::xchg::err_pmsg) != 0;
    const bool pass    = b0 && b1 && ew[0] == 0 && refused;
    v.put("exchange " + std::to_string(rows) + " rows", pass,
          std::string("card 0 ") + (b0 ? "bitwise" : "DIFFERS") + ", card 1 " + (b1 ? "bitwise" : "DIFFERS") +
              " against f16 + f16 in card order and the FMA; a stale generation " +
              (refused ? "refused, nothing written" : "NOT REFUSED"));
    return pass;
}

// PB3: the exchange folded into hct_mix's prescale against pcombine then hct_mix, per card: the streams and every
// output bitwise; a stale generation refused (err_pmsg, the streams untouched)
bool fused_case(int32_t rows, verdict & v, const hcw & w) {
    rng                g(0xf5 + uint64_t(rows));
    std::vector<float> part[2], inj(size_t(rows) * 4), S;
    for (int32_t c = 0; c < 2; ++c) {
        part[c].resize(size_t(rows) * D);
        for (float & f : part[c]) {
            f = static_cast<float>(g.normal() * 3.0);
        }
    }
    for (float & f : inj) {
        f = static_cast<float>(g.uniform(0.0, 2.0));
    }
    make_x(rows, 0x7f0 + uint64_t(rows), S);
    std::vector<uint16_t> dn[2], up[2];
    std::vector<float>    gm[2];
    halves(w, dn, up, gm);
    const size_t slot = static_cast<size_t>(fn::xchg::pmsg_bytes(int64_t(rows) * D * 2));
    dbuf         ddn0(dn[0].size() * 2), ddn1(dn[1].size() * 2), dup0(up[0].size() * 2), dup1(up[1].size() * 2),
        dg0(gm[0].size() * 4), dg1(gm[1].size() * 4), dp0(part[0].size() * 4), dp1(part[1].size() * 4), s0{ slot },
        s1{ slot }, di(inj.size() * 4), dS(S.size() * 4), mom(size_t(rows) * fn::hc::hcp_mom * 4),
        low(size_t(rows) * NL * 4), rinv(size_t(rows) * 16), io(size_t(rows) * 16), mixed(size_t(rows) * D * 4),
        m16(size_t(rows) * D * 2), err(256), w16(static_cast<size_t>(fn::hc::hct_weight_bytes())),
        scr(static_cast<size_t>(fn::hc::hct_scratch_bytes(rows)));
    const uint64_t gen = fn::xchg::pgen(3, 9, 0, 1, fn::xchg::pkind::attn);
    if (!ddn0.put(dn[0]) || !ddn1.put(dn[1]) || !dup0.put(up[0]) || !dup1.put(up[1]) || !dg0.put(gm[0]) ||
        !dg1.put(gm[1]) || !dp0.put(part[0]) || !dp1.put(part[1]) || !di.put(inj) || !err.zero() || !w16.poison() ||
        !fn::xchg::ppack(rows, dp0.as<float>(), s0.as<uint8_t>(), gen, err.as<uint32_t>(), fn::ptest::test_stream()) ||
        !fn::xchg::ppack(rows, dp1.as<float>(), s1.as<uint8_t>(), gen, err.as<uint32_t>(), fn::ptest::test_stream())) {
        v.put("fused exchange " + std::to_string(rows) + " rows", false, "setup failed");
        return false;
    }
    fn::hc::hcp_args a;
    a.w.form     = fn::hc::hcp_form::replicated;
    a.w.down[0]  = ddn0.as<uint16_t>();
    a.w.down[1]  = ddn1.as<uint16_t>();
    a.w.up[0]    = dup0.as<uint16_t>();
    a.w.up[1]    = dup1.as<uint16_t>();
    a.w.gamma[0] = dg0.as<float>();
    a.w.gamma[1] = dg1.as<float>();
    a.rows       = rows;
    a.x          = dS.as<float>();
    a.mom        = mom.as<float>();
    a.low        = low.as<float>();
    a.rinv       = rinv.as<float>();
    a.inject     = io.as<float>();
    a.mixed      = mixed.as<float>();
    a.ld_mixed   = D;
    a.mixed16    = m16.as<uint16_t>();
    a.ld_mixed16 = D;
    a.err        = err.as<uint32_t>();
    const fn::hc::hct_weights hw = fn::hc::hct_carve(w16.as<uint8_t>());
    struct outs {
        std::vector<float>    S, mom, low, rinv, inj, mixed;
        std::vector<uint16_t> m16;
    };
    // fused: the combine inside hct_mix; else pcombine first
    const auto run = [&](int32_t card, bool fused, uint64_t g_use, outs & o) {
        const uint8_t *     own  = card == 0 ? s0.as<uint8_t>() : s1.as<uint8_t>();
        const uint8_t *     peer = card == 0 ? s1.as<uint8_t>() : s0.as<uint8_t>();
        fn::hc::hct_combine cb;
        if (fused) {
            cb.card    = card;
            cb.own     = own;
            cb.peer    = peer;
            cb.gen     = g_use;
            cb.inject  = di.as<float>();
            cb.streams = dS.as<float>();
        }
        return dS.put(S) && mom.poison() && low.poison() && rinv.poison() && io.poison() && mixed.poison() &&
               m16.poison() && scr.poison() && fn::hc::hct_convert(a.w, hw, fn::ptest::test_stream()) &&
               (fused || fn::xchg::pcombine(rows, card, own, peer, g_use, di.as<float>(), dS.as<float>(),
                                            err.as<uint32_t>(), fn::ptest::test_stream())) &&
               fn::hc::hct_mix(a, cb, hw, scr.as<uint8_t>(), fn::ptest::test_stream()) &&
               cuda_ok(cudaDeviceSynchronize(), "fused") && dS.get(o.S, S.size()) &&
               mom.get(o.mom, size_t(rows) * fn::hc::hcp_mom) && low.get(o.low, size_t(rows) * NL) &&
               rinv.get(o.rinv, size_t(rows) * 4) && io.get(o.inj, size_t(rows) * 4) &&
               mixed.get(o.mixed, size_t(rows) * D) && m16.get(o.m16, size_t(rows) * D);
    };
    const auto eq = [](const auto & x, const auto & y) {
        return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0;
    };
    bool        ok = true, same = true;
    std::string what;
    for (int32_t card = 0; card < 2 && ok; ++card) {
        outs ref, got;
        ok = run(card, false, gen, ref) && run(card, true, gen, got);
        const bool c = ok && eq(ref.S, got.S) && eq(ref.mom, got.mom) && eq(ref.low, got.low) &&
                       eq(ref.rinv, got.rinv) && eq(ref.inj, got.inj) && eq(ref.mixed, got.mixed) &&
                       eq(ref.m16, got.m16) && !eq(ref.S, S);
        same = same && c;
        what += std::string(card == 0 ? "card 0 " : ", card 1 ") + (c ? "bitwise" : "DIFFERS");
    }
    std::vector<uint32_t> ew, ew2;
    ok = ok && err.get(ew, 1);
    // a stale generation: the error bit, the streams untouched
    outs       st;
    const bool refused = ok && err.zero() && run(0, true, gen + 1, st) && err.get(ew2, 1) && eq(st.S, S) &&
                         (ew2[0] & fn::xchg::err_pmsg) != 0;
    const bool pass = ok && same && ew[0] == 0 && refused;
    v.put("fused exchange " + std::to_string(rows) + " rows", pass,
          what + " (streams, mom, low, rinv, inject, mixed, FP16 copy) against pcombine then hct_mix; a stale " +
              "generation " + (refused ? "refused, the streams untouched" : "NOT REFUSED"));
    return pass;
}

}  // namespace

int fnfl_hcp_test(int device, const char * mode) {
    const bool san = mode != nullptr && std::strcmp(mode, "san") == 0;
    verdict    v("hcp-test");
    if (!cuda_ok(cudaSetDevice(device), "device")) {
        return 1;
    }
    hcw w;
    make_w(0x48437031, w);
    for (const int32_t r : san ? std::vector<int32_t>{ 3 } : std::vector<int32_t>{ 1, 5, 64, 65, 200 }) {
        replicated_case(r, v, w);
    }
    for (const int32_t r : san ? std::vector<int32_t>{ 2 } : std::vector<int32_t>{ 1, 65, 130 }) {
        split_case(r, v, w);
    }
    for (const int32_t r : san ? std::vector<int32_t>{ 3 } : std::vector<int32_t>{ 1, 65, 200, 1536 }) {
        std::vector<float> x;
        make_x(r, 0x4c0 + uint64_t(r), x);
        tensor_case("tensor " + std::to_string(r) + " rows", w, x, r, v);
    }
    {
        hcw sw, sm;
        stress_w(w, sw, false);
        stress_w(w, sm, true);
        std::vector<float> x, xs;
        make_x(san ? 5 : 67, 0x5a5a, x);
        stress_x(san ? 5 : 67, xs);
        tensor_case("tensor coefficient range stress (rows at 2^20, 2^-30, 2^18, 2^-26)", sw, x, san ? 5 : 67, v, true);
        tensor_case("tensor row range stress (an outlier row, a near-zero row, up rows at 2^-26)", sm, xs, san ? 5 : 67,
                    v);
    }
    // the exchange cases before the real-row mode's return (PB3: the cycle runs hcp-test real=...)
    for (const int32_t r : san ? std::vector<int32_t>{ 3 } : std::vector<int32_t>{ 1, 1536 }) {
        exchange_case(r, v);
    }
    for (const int32_t r : san ? std::vector<int32_t>{ 3 } : std::vector<int32_t>{ 1, 65, 1536 }) {
        fused_case(r, v, w);
    }
    if (mode != nullptr && std::strncmp(mode, "real=", 5) == 0) {
        real_cases(std::string(mode + 5), v);
    }
    return v.finish();
}
