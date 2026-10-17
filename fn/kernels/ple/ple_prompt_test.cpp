#include "fn/kernels/ple/ple_prompt_test.h"

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/embed/embed.h"
#include "fn/kernels/ple/ple.h"
#include "fn/kernels/ple/ple_prompt.h"
#include "fn/kernels/round/version.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/quant/synth.h"
#include "fn/tools/prompt_test_util.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace fn::ptest;
constexpr int32_t D = 2560, F = 1280, CH = 4 * D, NH = 9;

// the embedding: rows of two card tables (synthetic IQ3_S, 4,096 tokens a table), each card's half bitwise
// fn/quant's decoder; then the join gives both cards the full rows
bool embed_case(int32_t rows, verdict & v) {
    constexpr int32_t    V = 4096;
    std::vector<uint8_t> tab[2];
    for (int32_t h = 0; h < 2; ++h) {
        tab[h].resize(size_t(V) * fn::embed::row_bytes);
        if (!fn::quant::synth_blocks(fn::quant::wfmt::iq3_s, fn::quant::synth_seed(0x454d42u + h), int64_t(V) * 5,
                                     tab[h].data())) {
            return false;
        }
    }
    rng                  r(0xe3 + uint64_t(rows));
    std::vector<int32_t> tok(static_cast<size_t>(rows));
    for (int32_t & t : tok) {
        t = static_cast<int32_t>(r.next() % V);
    }
    std::vector<uint8_t> gt;
    if (!fn::dense::grid_buffer(gt)) {
        return false;
    }
    dbuf dg(gt.size()), dt0(tab[0].size()), dt1(tab[1].size()), dtok(tok.size() * 4), own0(size_t(rows) * F * 4),
        own1(size_t(rows) * F * 4), full0(size_t(rows) * D * 4), full1(size_t(rows) * D * 4),
        s0(size_t(rows) * 4 * D * 4), s1(size_t(rows) * 4 * D * 4), err(256);
    if (!dg.put(gt) || !dt0.put(tab[0]) || !dt1.put(tab[1]) || !dtok.put(tok) || !full0.poison() || !full1.poison() ||
        !s0.poison() || !s1.poison() || !err.zero()) {
        return false;
    }
    fn::embed::rows_args a[2];
    for (int32_t h = 0; h < 2; ++h) {
        a[h].table   = h == 0 ? dt0.as<uint8_t>() : dt1.as<uint8_t>();
        a[h].grids   = dg.as<uint8_t>();
        a[h].tok     = dtok.as<int32_t>();
        a[h].rows    = rows;
        a[h].card    = h;
        a[h].own     = h == 0 ? own0.as<float>() : own1.as<float>();
        a[h].full    = h == 0 ? full0.as<float>() : full1.as<float>();
        a[h].streams = h == 0 ? s0.as<float>() : s1.as<float>();
        a[h].err     = err.as<uint32_t>();
    }
    std::vector<float>    f0, f1, st0, st1;
    std::vector<uint32_t> ew;
    const bool            ok0 = fn::embed::embed_rows(a[0], fn::ptest::test_stream()) &&
                                fn::embed::embed_rows(a[1], fn::ptest::test_stream()) &&
                                fn::embed::embed_join(a[0], own1.as<float>(), fn::ptest::test_stream()) &&
                                fn::embed::embed_join(a[1], own0.as<float>(), fn::ptest::test_stream()) &&
                                cuda_ok(cudaDeviceSynchronize(), "embed") && full0.get(f0, size_t(rows) * D) &&
                                full1.get(f1, size_t(rows) * D) && s0.get(st0, size_t(rows) * 4 * D) &&
                                s1.get(st1, size_t(rows) * 4 * D) && err.get(ew, 1);
    if (!ok0) {
        v.put("embed", false, "launch failed");
        return false;
    }
    int64_t            bad = 0;
    std::vector<float> want(static_cast<size_t>(D));
    for (int32_t t = 0; t < rows; ++t) {
        for (int32_t h = 0; h < 2; ++h) {
            fn::quant::decode_row(fn::quant::wfmt::iq3_s, tab[h].data() + size_t(tok[size_t(t)]) * fn::embed::row_bytes,
                                  want.data() + ptrdiff_t(h) * F, F);
        }
        bad += std::memcmp(&f0[size_t(t) * D], want.data(), size_t(D) * 4) != 0;
        bad += std::memcmp(&f1[size_t(t) * D], want.data(), size_t(D) * 4) != 0;
        for (int32_t s = 0; s < 4; ++s) {
            bad += std::memcmp(&st0[(size_t(t) * 4 + s) * D], want.data(), size_t(D) * 4) != 0;
            bad += std::memcmp(&st1[(size_t(t) * 4 + s) * D], want.data(), size_t(D) * 4) != 0;
        }
    }
    v.put("embedding " + std::to_string(rows) + " rows", bad == 0 && ew[0] == 0,
          std::to_string(bad) + " rows differ from fn/quant's IQ3_S decode (the full row on both cards, all streams)");
    return bad == 0;
}

double sig(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

bool ple_case(int32_t rows, verdict & v) {
    rng                   r(0x504c + uint64_t(rows));
    // full-width coefficients, as their two card halves
    std::vector<uint16_t> key(size_t(CH) * D), val(size_t(D) * D);
    for (uint16_t & u : key) {
        u = fn::quant::f32_to_bf16(static_cast<float>(r.normal() * 0.02));
    }
    for (uint16_t & u : val) {
        u = fn::quant::f32_to_bf16(static_cast<float>(r.normal() * 0.02));
    }
    std::vector<float>    nk(CH), nq(CH), nc(CH);
    std::vector<uint16_t> taps(size_t(CH) * 4);
    for (int32_t i = 0; i < CH; ++i) {
        nk[size_t(i)] = static_cast<float>(r.uniform(0.5, 1.5));
        nq[size_t(i)] = static_cast<float>(r.uniform(0.5, 1.5));
        nc[size_t(i)] = static_cast<float>(r.uniform(0.5, 1.5));
    }
    for (uint16_t & u : taps) {
        u = fn::quant::f32_to_f16(static_cast<float>(r.normal() * 0.5));
    }
    std::vector<uint16_t> keyh[2], valh[2], tapsh[2];
    std::vector<float>    nkh[2], nqh[2], nch[2];
    for (int32_t h = 0; h < 2; ++h) {
        keyh[h].resize(size_t(4) * F * D);
        valh[h].resize(size_t(F) * D);
        tapsh[h].resize(size_t(4) * F * 4);
        nkh[h].resize(size_t(4) * F);
        nqh[h].resize(size_t(4) * F);
        nch[h].resize(size_t(4) * F);
        for (int32_t s = 0; s < 4; ++s) {
            for (int32_t i = 0; i < F; ++i) {
                const int32_t full = s * D + h * F + i, card = s * F + i;
                std::memcpy(&keyh[h][size_t(card) * D], &key[size_t(full) * D], size_t(D) * 2);
                std::memcpy(&tapsh[h][size_t(card) * 4], &taps[size_t(full) * 4], 8);
                nkh[h][size_t(card)] = nk[size_t(full)];
                nqh[h][size_t(card)] = nq[size_t(full)];
                nch[h][size_t(card)] = nc[size_t(full)];
            }
        }
        std::memcpy(valh[h].data(), &val[size_t(h) * F * D], valh[h].size() * 2);
    }
    std::vector<float> e(size_t(rows) * D), S(size_t(rows) * 4 * D), hin(size_t(CH) * NH);
    for (float & f : e) {
        f = static_cast<float>(r.normal());
    }
    for (float & f : S) {
        f = static_cast<float>(r.normal());
    }
    for (float & f : hin) {
        f = static_cast<float>(r.normal() * 0.5);
    }
    // the products (pd_bf16, two bases) on this device, as card 0 with card 1's half as its replica
    dbuf dk0(keyh[0].size() * 2), dk1(keyh[1].size() * 2), dv0(valh[0].size() * 2), dv1(valh[1].size() * 2),
        dt0(tapsh[0].size() * 2), dt1(tapsh[1].size() * 2), dnk0(size_t(F) * 16), dnk1(size_t(F) * 16),
        dnq0(size_t(F) * 16), dnq1(size_t(F) * 16), dnc0(size_t(F) * 16), dnc1(size_t(F) * 16), de(e.size() * 4),
        dS(S.size() * 4), dkey(size_t(rows) * CH * 4), dval(size_t(rows) * D * 4), dsc(size_t(rows) * 32),
        dx(size_t(rows) * CH * 4), dhi(hin.size() * 4), dho(hin.size() * 4),
        dst(static_cast<size_t>(fn::ple::hist_bytes)), dcur(256);
    const uint32_t cur = 2;
    if (!dk0.put(keyh[0]) || !dk1.put(keyh[1]) || !dv0.put(valh[0]) || !dv1.put(valh[1]) || !dt0.put(tapsh[0]) ||
        !dt1.put(tapsh[1]) || !dnk0.put(nkh[0]) || !dnk1.put(nkh[1]) || !dnq0.put(nqh[0]) || !dnq1.put(nqh[1]) ||
        !dnc0.put(nch[0]) || !dnc1.put(nch[1]) || !de.put(e) || !dS.put(S) || !dhi.put(hin) || !dho.poison() ||
        !dst.poison() || !dcur.put(std::vector<uint32_t>{ cur })) {
        return false;
    }
    fn::dense::pb_args kb;
    kb.w_lo               = dk0.as<uint16_t>();
    kb.w_hi               = dk1.as<uint16_t>();
    kb.period             = D;
    kb.half               = F;
    kb.n                  = CH;
    kb.k                  = D;
    kb.rows               = rows;
    kb.x                  = de.as<float>();
    kb.ldx                = D;
    kb.out                = dkey.as<float>();
    kb.ld_out             = CH;
    fn::dense::pb_args vb = kb;
    vb.w_lo               = dv0.as<uint16_t>();
    vb.w_hi               = dv1.as<uint16_t>();
    vb.n                  = D;
    vb.out                = dval.as<float>();
    vb.ld_out             = D;
    fn::ple::prompt_args pa;
    pa.w.norm_key[0]   = dnk0.as<float>();
    pa.w.norm_key[1]   = dnk1.as<float>();
    pa.w.norm_query[0] = dnq0.as<float>();
    pa.w.norm_query[1] = dnq1.as<float>();
    pa.w.norm_conv[0]  = dnc0.as<float>();
    pa.w.norm_conv[1]  = dnc1.as<float>();
    pa.w.conv[0]       = dt0.as<uint16_t>();
    pa.w.conv[1]       = dt1.as<uint16_t>();
    pa.rows            = rows;
    pa.key             = dkey.as<float>();
    pa.value           = dval.as<float>();
    pa.streams         = dS.as<float>();
    pa.scal            = dsc.as<float>();
    pa.x               = dx.as<float>();
    pa.hist_in         = dhi.as<float>();
    pa.hist_out        = dho.as<float>();
    pa.state           = dst.as<float>();
    pa.cur             = dcur.as<uint32_t>();
    pa.v_out           = 2;
    pa.card            = 0;
    std::vector<float> Sg, xg, ho, sto;
    const std::string  name = "PLE " + std::to_string(rows) + " rows";
    if (!fn::dense::pd_bf16(kb, fn::ptest::test_stream()) || !fn::dense::pd_bf16(vb, fn::ptest::test_stream()) ||
        !fn::ple::ple_prompt_stats(pa, fn::ptest::test_stream()) ||
        !fn::ple::ple_prompt_x(pa, fn::ptest::test_stream()) ||
        !fn::ple::ple_prompt_update(pa, fn::ptest::test_stream()) ||
        !fn::ple::ple_prompt_hist(pa, fn::ptest::test_stream()) || !cuda_ok(cudaDeviceSynchronize(), "ple") ||
        !dS.get(Sg, S.size()) || !dx.get(xg, size_t(rows) * CH) || !dho.get(ho, hin.size()) ||
        !dst.get(sto, size_t(fn::ple::hist_bytes / 4))) {
        v.put(name, false, "launch failed");
        return false;
    }
    // FP64 of E1's pinned form, full width: the products, the statistics and scalars, x, the convolution, the update
    std::vector<double> hout(S.size()), xr(size_t(rows) * CH), gated(size_t(rows) * CH);
    parallel_for(rows, [&](int64_t t) {
        std::vector<double> kk(CH), vv(D);
        for (int32_t n = 0; n < CH; ++n) {
            double a = 0.0;
            for (int32_t k = 0; k < D; ++k) {
                a += double(fn::quant::bf16_to_f32(key[size_t(n) * D + k])) * e[size_t(t) * D + k];
            }
            kk[size_t(n)] = a;
        }
        double V = 0.0;
        for (int32_t n = 0; n < D; ++n) {
            double a = 0.0;
            for (int32_t k = 0; k < D; ++k) {
                a += double(fn::quant::bf16_to_f32(val[size_t(n) * D + k])) * e[size_t(t) * D + k];
            }
            vv[size_t(n)] = a;
            V += a * a;
        }
        for (int32_t s = 0; s < 4; ++s) {
            double K = 0, H = 0, KQ = 0;
            for (int32_t f = 0; f < D; ++f) {
                const double k = kk[size_t(s) * D + f], hv = S[(size_t(t) * 4 + s) * D + f];
                K += k * k;
                H += hv * hv;
                KQ += k * nk[size_t(s) * D + f] * hv * nq[size_t(s) * D + f];
            }
            const double rk = 1.0 / std::sqrt(K / D + 1e-6), rq = 1.0 / std::sqrt(H / D + 1e-6);
            const double score = KQ * rk * rq / std::sqrt(double(D));
            const double mag   = std::sqrt(std::max(std::fabs(score), 1e-6));
            const double gate  = sig(score > 0 ? mag : score < 0 ? -mag : 0.0);
            const double xinv  = 1.0 / std::sqrt(gate * gate * V / D + 1e-6);
            for (int32_t f = 0; f < D; ++f) {
                gated[size_t(t) * CH + size_t(s) * D + f] = vv[size_t(f)] * gate;
                xr[size_t(t) * CH + size_t(s) * D + f]    = vv[size_t(f)] * gate * xinv * nc[size_t(s) * D + f];
            }
        }
    });
    for (int32_t t = 0; t < rows; ++t) {
        for (int32_t ch = 0; ch < CH; ++ch) {
            const auto xa = [&](int32_t rr) {
                return rr >= 0 ? xr[size_t(rr) * CH + ch] : double(hin[size_t(ch) * NH + NH + rr]);
            };
            const uint16_t * w   = &taps[size_t(ch) * 4];
            const double     acc = xa(t - 9) * fn::quant::f16_to_f32(w[0]) + xa(t - 6) * fn::quant::f16_to_f32(w[1]) +
                                   xa(t - 3) * fn::quant::f16_to_f32(w[2]) + xa(t) * fn::quant::f16_to_f32(w[3]);
            const int32_t    s = ch / D, f = ch % D;
            const double     hv               = S[(size_t(t) * 4 + s) * D + f];
            hout[(size_t(t) * 4 + s) * D + f] = hv + (gated[size_t(t) * CH + ch] + acc * sig(acc));
        }
    }
    const double ex   = unit_error(xg.data(), xr.data(), int64_t(rows) * 4, D);
    const double eh   = unit_error(Sg.data(), hout.data(), int64_t(rows) * 4, D);
    // the history: the last 9 x of each channel, bitwise the kernel's own x (or the older history)
    int64_t      hbad = 0, sbad = 0;
    for (int32_t ch = 0; ch < CH; ++ch) {
        for (int32_t j = 0; j < NH; ++j) {
            const int32_t rr   = rows - NH + j;
            const float   want = rr >= 0 ? xg[size_t(rr) * CH + ch] : hin[size_t(ch) * NH + NH + rr];
            hbad += std::memcmp(&ho[size_t(ch) * NH + j], &want, 4) != 0;
            const int32_t s = ch / D, f = ch % D;
            if (f < F) {  // card 0's channels into its state version
                const float got = sto[size_t(fn::round::version_slot(cur, 2)) * fn::ple::hist_version_floats +
                                      size_t(s * F + f) * NH + j];
                sbad += std::memcmp(&got, &want, 4) != 0;
            }
        }
    }
    const bool ok = ex <= 1e-5 && eh <= 1e-5 && hbad == 0 && sbad == 0;
    v.put(name, ok,
          "x " + e3(ex) + ", streams " + e3(eh) + " (bound 1e-5 a row and stream); history " +
              (hbad == 0 ? "bitwise" : std::to_string(hbad) + " DIFFER") + "; card 0's state version " +
              (sbad == 0 ? "bitwise" : std::to_string(sbad) + " DIFFER"));
    return ok;
}

}  // namespace

int fnfl_pe_test(int device, const char * mode) {
    const bool san = mode != nullptr && std::strcmp(mode, "san") == 0;
    verdict    v("pe-test");
    if (!cuda_ok(cudaSetDevice(device), "device")) {
        return 1;
    }
    for (const int32_t rows : san ? std::vector<int32_t>{3} : std::vector<int32_t>{1, 7, 1536, 3072, 8192}) {
        embed_case(rows, v);
    }
    for (const int32_t rows : san ? std::vector<int32_t>{ 5 } : std::vector<int32_t>{ 1, 5, 9, 10, 40 }) {
        ple_case(rows, v);
    }
    return v.finish();
}
