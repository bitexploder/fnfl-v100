#include "fn/kernels/draft/draft_prompt_test.h"

#include "fn/kernels/draft/draft_prompt.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/quant/formats.h"
#include "fn/tools/prompt_test_util.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace fn::ptest;
constexpr int32_t D = 2560, F = 1280, HD = 256;

// within one FP16 step of the FP64 value plus the F32 rounding of the terms that formed it (the kernel computes in F32
// and rounds once; a rope pair y c - o s near zero cancels, so its F32 error scales with |y c| + |o s|, not the result)
bool f16_near(uint16_t got, double ref, double terms) {
    const double g   = fn::quant::f16_to_f32(got);
    const double ulp = std::ldexp(1.0, std::max(std::ilogb(std::max(std::fabs(ref), 6.1e-5)), -14) - 10);
    return std::fabs(g - ref) <= ulp + std::ldexp(terms, -21);
}

bool norm_case(int32_t rows, verdict & v) {
    rng                r(0xd0 + uint64_t(rows));
    std::vector<float> emb(size_t(rows) * D), hid(size_t(rows) * 4 * D), pred(size_t(4) * D), en(D), hn(size_t(4) * D);
    for (float & f : emb) {
        f = static_cast<float>(r.normal());
    }
    for (float & f : hid) {
        f = static_cast<float>(r.normal() * 3.0);
    }
    for (float & f : pred) {
        f = static_cast<float>(r.normal() * 3.0);
    }
    for (float & f : en) {
        f = static_cast<float>(r.uniform(0.5, 1.5));
    }
    for (float & f : hn) {
        f = static_cast<float>(r.uniform(0.5, 1.5));
    }
    dbuf de(emb.size() * 4), dh(hid.size() * 4), dp(pred.size() * 4), den(size_t(D) * 4), dhn(hn.size() * 4),
        e16(size_t(rows) * D * 2), h16(size_t(rows) * 4 * D * 2), err(256);
    if (!de.put(emb) || !dh.put(hid) || !dp.put(pred) || !den.put(en) || !dhn.put(hn) || !err.zero()) {
        return false;
    }
    fn::draft::norm_args a;
    a.rows   = rows;
    a.emb    = de.as<float>();
    a.hidden = dh.as<float>();
    a.pred   = dp.as<float>();
    a.enorm  = den.as<float>();
    a.hnorm  = dhn.as<float>();
    a.e16    = e16.as<uint16_t>();
    a.h16    = h16.as<uint16_t>();
    a.err    = err.as<uint32_t>();
    std::vector<uint16_t> ge, gh;
    if (!fn::draft::dcp_norm(a, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "dcp_norm") ||
        !e16.get(ge, size_t(rows) * D) || !h16.get(gh, size_t(rows) * 4 * D)) {
        v.put("norm", false, "launch failed");
        return false;
    }
    int64_t    bad = 0;
    const auto seg = [&](const float * x, const float * g, const uint16_t * got) {
        double ss = 0.0;
        for (int32_t i = 0; i < D; ++i) {
            ss += double(x[i]) * x[i];
        }
        const double rs = 1.0 / std::sqrt(ss / D + 1e-6);
        for (int32_t i = 0; i < D; ++i) {
            bad += f16_near(got[i], double(x[i]) * g[i] * rs, 0.0) ? 0 : 1;
        }
    };
    for (int32_t t = 0; t < rows; ++t) {
        seg(&emb[size_t(t) * D], en.data(), &ge[size_t(t) * D]);
        for (int32_t s = 0; s < 4; ++s) {
            const float * h = t == 0 ? &pred[size_t(s) * D] : &hid[(size_t(t - 1) * 4 + s) * D];
            seg(h, &hn[size_t(s) * D], &gh[(size_t(t) * 4 + s) * D]);
        }
    }
    v.put("norm " + std::to_string(rows) + " positions", bad == 0,
          std::to_string(bad) + " values beyond one FP16 step of FP64 (e and h, row t's h the hidden row t - 1)");
    return bad == 0;
}

bool mixed_prepare_case(int32_t rows, verdict & v) {
    rng                r(0xd7 + uint64_t(rows));
    std::vector<float> half[2], kk(size_t(rows) * HD), vv(size_t(rows) * HD), kn(HD);
    for (int32_t h = 0; h < 2; ++h) {
        half[h].resize(size_t(rows) * F);
        for (float & f : half[h]) {
            f = static_cast<float>(r.normal());
        }
    }
    for (float & f : kk) {
        f = static_cast<float>(r.normal());
    }
    for (float & f : vv) {
        f = static_cast<float>(r.normal());
    }
    for (float & f : kn) {
        f = static_cast<float>(r.uniform(0.5, 1.5));
    }
    const int32_t cells = 8192, pos0 = 3000; // PB3: room for a 4,096-row half after pos0
    dbuf          h0(half[0].size() * 4), h1(half[1].size() * 4), m16(size_t(rows) * D * 2), dk(kk.size() * 4),
        dv(vv.size() * 4), dkn(size_t(HD) * 4), kc(size_t(cells) * HD * 2), vc(size_t(cells) * HD * 2), err(256);
    if (!h0.put(half[0]) || !h1.put(half[1]) || !dk.put(kk) || !dv.put(vv) || !dkn.put(kn) || !kc.zero() ||
        !vc.zero() || !err.zero()) {
        return false;
    }
    std::vector<uint16_t> g16[2];
    bool                  ok = true;
    for (int32_t card = 0; card < 2 && ok; ++card) {
        ok = fn::draft::dcp_mixed(rows, card, card == 0 ? h0.as<float>() : h1.as<float>(),
                                  card == 0 ? h1.as<float>() : h0.as<float>(), m16.as<uint16_t>(), err.as<uint32_t>(),
                                  fn::ptest::test_stream()) &&
             cuda_ok(cudaDeviceSynchronize(), "dcp_mixed") && m16.get(g16[card], size_t(rows) * D);
    }
    int64_t mbad = 0;
    for (int32_t t = 0; ok && t < rows; ++t) {
        for (int32_t c = 0; c < D; ++c) {
            const uint16_t want = fn::quant::f32_to_f16(half[c / F][size_t(t) * F + c % F]);
            mbad += (g16[0][size_t(t) * D + c] != want) + (g16[1][size_t(t) * D + c] != want);
        }
    }
    fn::draft::kvp_args ka;
    ka.rows        = rows;
    ka.pos0        = pos0;
    ka.max_cells   = cells;
    ka.kv_k        = dk.as<float>();
    ka.kv_v        = dv.as<float>();
    ka.k_norm      = dkn.as<float>();
    ka.k_cache     = kc.as<uint16_t>();
    ka.v_cache     = vc.as<uint16_t>();
    ka.theta_scale = fn::qsa::attn::rope_theta_scale();
    std::vector<uint16_t> gk, gv;
    ok           = ok && fn::draft::dcp_prepare(ka, fn::ptest::test_stream()) &&
                   cuda_ok(cudaDeviceSynchronize(), "dcp_prepare") && kc.get(gk, size_t(cells) * HD) &&
                   vc.get(gv, size_t(cells) * HD);
    int64_t kbad = 0, vbad = 0;
    for (int32_t t = 0; ok && t < rows; ++t) {
        double ss = 0.0;
        for (int32_t e = 0; e < HD; ++e) {
            ss += double(kk[size_t(t) * HD + e]) * kk[size_t(t) * HD + e];
        }
        const double rn = 1.0 / std::sqrt(ss / HD + 1e-6);
        double       y[HD], terms[HD];
        for (int32_t e = 0; e < HD; ++e) {
            y[e]     = kk[size_t(t) * HD + e] * rn * kn[size_t(e)];
            terms[e] = std::fabs(y[e]);
        }
        for (int32_t i = 0; i < 32; ++i) {
            float theta = static_cast<float>(pos0 + t);
            for (int32_t q = 0; q < i; ++q) {
                theta = theta * ka.theta_scale;
            }
            const double c = std::cos(double(theta)), s = std::sin(double(theta)), a = y[i], b = y[i + 32];
            y[i]          = a * c - b * s;
            y[i + 32]     = a * s + b * c;
            terms[i]      = std::fabs(a * c) + std::fabs(b * s);
            terms[i + 32] = std::fabs(a * s) + std::fabs(b * c);
        }
        for (int32_t e = 0; e < HD; ++e) {
            const size_t at = (size_t(pos0) + t) * HD + e;
            kbad += f16_near(gk[at], y[e], terms[e]) ? 0 : 1;
            vbad += gv[at] == fn::quant::f32_to_f16(vv[size_t(t) * HD + e]) ? 0 : 1;
        }
    }
    const bool pass = ok && mbad == 0 && kbad == 0 && vbad == 0;
    v.put("mixed and K/V " + std::to_string(rows) + " positions", pass,
          std::to_string(mbad) + " mixed values not f16 of their card's half; " + std::to_string(kbad) +
              " K beyond one F16 step of FP64 plus F32 term rounding; " + std::to_string(vbad) + " V not f16(v)");
    return pass;
}

}  // namespace

int fnfl_dcp_test(int device, const char * mode) {
    const bool san = mode != nullptr && std::strcmp(mode, "san") == 0;
    verdict    v("dcp-test");
    if (!cuda_ok(cudaSetDevice(device), "device")) {
        return 1;
    }
    for (const int32_t rows : san ? std::vector<int32_t>{2} : std::vector<int32_t>{1, 2, 300, 4096}) {
        norm_case(rows, v);
        mixed_prepare_case(rows, v);
    }
    return v.finish();
}
