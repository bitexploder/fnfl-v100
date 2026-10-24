#include "fn/kernels/gdn/gdn_prompt_test.h"

#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/gdn/gdn_prompt.h"
#include "fn/kernels/gdn/gdn_ref.h"
#include "fn/kernels/round/version.h"
#include "fn/quant/formats.h"
#include "fn/tools/prompt_test_util.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace fn::ptest;
constexpr int32_t NH = 24, HD = 128, NCH = 5120, NV = 3072, NG = 8, RMAX = 4096; // PB3's half
constexpr int32_t RCH = 1536; // the chains' halves (their row totals as before PB3)
constexpr int64_t SV = int64_t(NH) * HD * HD;
constexpr int32_t LQZ = NCH + NV; // PB3: the input product's FP16 output, [rows][qkv | z]
// PB3 Amendment 2: the recurrence is checked for layout, indexing and range errors against FP64 at a loose
// tolerance (max |err| / max |ref| a head), not a precision bound
constexpr double  gdp_loose = 2e-2;

double sig(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// the recurrence's inputs for one run: conv rows (q and k normalized, v), the log decay, beta, z
struct run_in {
    std::vector<float> conv, g, beta, z;
};

void make_run(int32_t rows, uint64_t seed, bool zero, run_in & in) {
    rng r(seed);
    in.conv.assign(size_t(rows) * NCH, 0.0f);
    in.g.assign(size_t(rows) * NH, 0.0f);
    in.beta.assign(size_t(rows) * NH, 0.0f);
    in.z.assign(size_t(rows) * NV, 0.0f);
    if (zero) {
        return;
    }
    for (int32_t t = 0; t < rows; ++t) {
        float * row = in.conv.data() + size_t(t) * NCH;
        for (int32_t v = 0; v < 2 * NG; ++v) {
            double x[HD], ss = 0.0;
            for (int32_t i = 0; i < HD; ++i) {
                x[i] = r.normal();
                ss += x[i] * x[i];
            }
            const double n = 1.0 / std::sqrt(ss + 1e-6);
            for (int32_t i = 0; i < HD; ++i) {
                row[v * HD + i] = static_cast<float>(x[i] * n);  // q groups at 0, k groups at 1024
            }
        }
        for (int32_t i = 2048; i < NCH; ++i) {
            row[i] = static_cast<float>(r.normal());
        }
        for (int32_t h = 0; h < NH; ++h) {
            // PK3's distribution; heads 3 and 11 decay slowly (log decay -1e-3), where state errors persist longest
            const bool slow = h == 3 || h == 11;
            in.g[size_t(t) * NH + h] =
                slow ? -1e-3f : static_cast<float>(-std::exp(r.uniform(-5.0, 1.0) + 0.5 * r.normal()) * 0.3);
            in.beta[size_t(t) * NH + h] = static_cast<float>(sig(r.normal()));
        }
        for (int32_t i = 0; i < NV; ++i) {
            in.z[size_t(t) * NV + i] = static_cast<float>(r.normal());
        }
    }
}

// the gated norm in FP64 from the reference o
void gated_ref(const std::vector<double> & o,
               const std::vector<float> &  z,
               const std::vector<float> &  w,
               int32_t                     rows,
               std::vector<double> &       out) {
    out.assign(size_t(rows) * NV, 0.0);
    for (int32_t t = 0; t < rows; ++t) {
        for (int32_t h = 0; h < NH; ++h) {
            double ss = 0.0;
            for (int32_t j = 0; j < HD; ++j) {
                const double v = o[size_t(t) * NV + size_t(h) * HD + j];
                ss += v * v;
            }
            const double r = 1.0 / std::sqrt(ss / HD + 1e-6);
            for (int32_t j = 0; j < HD; ++j) {
                const size_t at = size_t(t) * NV + size_t(h) * HD + j;
                out[at]         = o[at] * r * double(w[size_t(j)]) * sig(double(z[at]));
            }
        }
    }
}

// per head, max |err| / max |ref| over the run's rows (gated: [rows][24 x 128]; state [24][128 x 128])
double head_error(const float * got, const double * ref, int64_t rows, int64_t row_stride, int32_t width) {
    double worst = 0.0;
    for (int32_t h = 0; h < NH; ++h) {
        double e = 0.0, m = 0.0;
        for (int64_t t = 0; t < rows; ++t) {
            for (int32_t j = 0; j < width; ++j) {
                const double gv = got[t * row_stride + int64_t(h) * width + j],
                             rv = ref[t * row_stride + int64_t(h) * width + j];
                if (!std::isfinite(gv)) {
                    return INFINITY;
                }
                e = std::max(e, std::fabs(gv - rv));
                m = std::max(m, std::fabs(rv));
            }
        }
        worst = std::max(worst, m > 0 ? e / m : e);
    }
    return worst;
}

struct dev {
    dbuf conv{size_t(RMAX) * NCH * 4}, g{size_t(RMAX) * NH * 4}, beta{size_t(RMAX) * NH * 4}, o{size_t(RMAX) * NV * 4},
        z{size_t(RMAX) * NV * 4}, g16{size_t(RMAX) * NV * 2}, gated{size_t(RMAX) * NV * 4},
        state{size_t(fn::gdn::n_bank) * SV * 4}, // ME11: the two banks
        rec{size_t(fn::gdn::n_rec_set) * fn::gdn::rec_set_floats * 4}, // ME11: the two record sets
        st64{size_t(SV) * 8}, o64{size_t(RMAX) * NV * 8}, cur{256}, w{size_t(HD) * 4}, err{256},
        qz16{size_t(RMAX) * LQZ * 2}, wy{static_cast<size_t>(fn::gdn::prompt_wy_bytes(RMAX))},
        o16{size_t(RMAX) * NV * 2};
};

struct chain {
    dev & d;
    std::vector<float> w;
    uint32_t cur = 0;
    uint32_t gsel = 0;   // ME11: the GDN selector (the scan's banks and, at version 0, its replayed records)
    int32_t ver = 0;     // the logical version the next run reads
    bool half_z = false; // PB3: z from the FP16 [rows][qkv | z] array (the reference uses its values)
    bool half_o = false; // PB3: o FP16 (o16) between the scan and the gated norm
    double worst_g = 0, worst_s = 0;
    std::vector<float> end_state = {}; // the last run's end state (ME11)

    // one run of rows from version ver to ver + 1 (the driver's halves; a publication moves cur by 2 after the second)
    bool run(const run_in & in, int32_t rows, std::string & why) {
        fn::gdn::prompt_args a;
        a.rows     = rows;
        a.conv_out = d.conv.as<float>();
        a.state    = d.state.as<float>();
        a.rec      = d.rec.as<float>();
        a.gsel     = d.cur.as<uint32_t>() + 1;
        a.cur      = d.cur.as<uint32_t>();
        a.v_in     = ver;
        a.v_out    = ver + 1;
        a.g        = d.g.as<float>();
        a.beta     = d.beta.as<float>();
        a.o        = d.o.as<float>();
        a.z        = d.z.as<float>();
        std::vector<float> z = in.z;
        if (half_z) {
            std::vector<uint16_t> qz(size_t(rows) * LQZ, 0);
            for (int32_t t = 0; t < rows; ++t) {
                for (int32_t i = 0; i < NV; ++i) {
                    const uint16_t hv = fn::quant::f32_to_f16(in.z[size_t(t) * NV + i]);
                    qz[size_t(t) * LQZ + NCH + i] = hv;
                    z[size_t(t) * NV + i] = fn::quant::f16_to_f32(hv);
                }
            }
            if (!d.qz16.put(qz)) {
                why = "upload";
                return false;
            }
            a.z = nullptr;
            a.z16 = d.qz16.as<uint16_t>() + NCH;
            a.ld16 = LQZ;
        }
        if (half_o) {
            a.o = nullptr;
            a.o16 = d.o16.as<uint16_t>();
        }
        a.norm_w   = d.w.as<float>();
        a.gated16  = d.g16.as<uint16_t>();
        a.gated    = d.gated.as<float>();
        a.err      = d.err.as<uint32_t>();
        a.wy       = d.wy.as<float>();
        if (!d.conv.put(in.conv) || !d.g.put(in.g) || !d.beta.put(in.beta) || !d.z.put(in.z) ||
            !d.cur.put(std::vector<uint32_t>{ cur, gsel }) || !fn::gdn::gdn_prompt_recur(a, fn::ptest::test_stream()) ||
            !fn::gdn::gdn_prompt_gate(a, fn::ptest::test_stream()) ||
            !gdp_reference(rows, d.conv.as<float>(), d.g.as<float>(), d.beta.as<float>(), d.st64.as<double>(),
                           d.o64.as<double>(), fn::ptest::test_stream()) ||
            !cuda_ok(cudaDeviceSynchronize(), "gdp run")) {
            why = "launch failed";
            return false;
        }
        std::vector<float>  gated, st;
        std::vector<double> o64, s64, gref;
        const int32_t       slot = fn::round::gdn_bank_of(gsel, ver + 1); // ME11: the output bank
        if (!d.gated.get(gated, size_t(rows) * NV) || !d.o64.get(o64, size_t(rows) * NV) ||
            !d.st64.get(s64, size_t(SV)) ||
            !cuda_ok(cudaMemcpy((st.resize(static_cast<size_t>(SV)), st.data()),
                                d.state.as<float>() + int64_t(slot) * SV, size_t(SV) * 4, cudaMemcpyDeviceToHost),
                     "state")) {
            why = "readback";
            return false;
        }
        gated_ref(o64, z, w, rows, gref);
        const double eg = head_error(gated.data(), gref.data(), rows, NV, HD);
        const double es = head_error(st.data(), s64.data(), 1, SV, HD * HD);
        worst_g = std::max(worst_g, eg);
        worst_s = std::max(worst_s, es);
        end_state = st;
        if (++ver == 2) {
            cur = static_cast<uint32_t>(fn::round::version_slot(cur, 2));
            gsel = fn::round::gdn_after_prompt(gsel);
            ver = 0;
        }
        return true;
    }
};

// the state: N(0, 0.5) in version 0 (ME11: the committed bank of gsel, no records), the same values in the FP64 state
bool init_state(dev & d, uint32_t gsel, uint64_t seed, bool zero) {
    rng                 r(seed);
    std::vector<float>  s(static_cast<size_t>(SV));
    std::vector<double> s64(static_cast<size_t>(SV));
    for (size_t i = 0; i < s.size(); ++i) {
        s[i]   = zero ? 0.0f : static_cast<float>(r.normal() * 0.5);
        s64[i] = s[i];
    }
    return d.state.poison() &&
           cuda_ok(cudaMemcpy(d.state.as<float>() + int64_t(fn::round::gdn_bank(gsel)) * SV, s.data(),
                              s.size() * 4, cudaMemcpyHostToDevice),
                   "state") &&
           d.st64.put(s64);
}

bool single(dev & d, int32_t rows, bool zero, verdict & v, const std::vector<float> & w, bool half_z = false,
            bool half_o = false) {
    const uint32_t cur = static_cast<uint32_t>(rows % 5);
    const uint32_t gsel = fn::round::gdn_word(rows & 1, (rows >> 1) & 1, 0); // ME11: both banks in turn
    if (!init_state(d, gsel, 0x5a + uint64_t(rows), zero)) {
        return false;
    }
    run_in in;
    make_run(rows, 0x700 + uint64_t(rows), zero, in);
    chain c{ d, w };
    c.cur = cur;
    c.gsel = gsel;
    c.half_z = half_z;
    c.half_o = half_o;
    std::string       why;
    const std::string name = std::string(zero ? "zero input " : "run ") + std::to_string(rows) + " rows" +
                             (half_z ? " (FP16 z)" : "") + (half_o ? " (FP16 o)" : "");
    if (!c.run(in, rows, why)) {
        v.put(name, false, why);
        return false;
    }
    std::vector<float> o1, o2, s1, s2;
    bool               same = d.gated.get(o1, size_t(rows) * NV);
    // a rerun from the same state is bitwise
    chain              c2{ d, w };
    c2.cur       = cur;
    c2.gsel      = gsel;
    c2.half_z = half_z;
    c2.half_o = half_o;
    same         = same && init_state(d, gsel, 0x5a + uint64_t(rows), zero) && c2.run(in, rows, why) &&
                   d.gated.get(o2, size_t(rows) * NV) && std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0;
    bool zero_ok = true;
    if (zero) {
        for (const float f : o1) {
            zero_ok = zero_ok && f == 0.0f;
        }
    }
    const bool loose = c.worst_g <= gdp_loose && c.worst_s <= gdp_loose;
    const bool ok    = loose && same && zero_ok;
    v.put(name, ok,
          std::string("gated output and end state ") + (loose ? "within" : "OUTSIDE") +
              " the loose bug-check tolerance; rerun " + (same ? "bitwise" : "DIFFERS") +
              (zero ? std::string(zero_ok ? "; all zero" : "; NOT ZERO") : ""));
    return ok;
}

bool chain_case(dev & d, int32_t runs, verdict & v, const std::vector<float> & w) {
    if (!init_state(d, fn::round::gdn_word(1, 1, 0), 0xc4a1 + uint64_t(runs), false)) {
        return false;
    }
    chain c{ d, w };
    c.cur = 2;
    c.gsel = fn::round::gdn_word(1, 1, 0);
    std::string why;
    for (int32_t i = 0; i < runs; ++i) {
        run_in in;
        make_run(RCH, 0x9000 + uint64_t(runs) * 1000 + i, false, in);
        if (!c.run(in, RCH, why)) {
            v.put(std::to_string(runs) + "-run chain", false, why + " at run " + std::to_string(i));
            return false;
        }
    }
    const bool ok = c.worst_g <= gdp_loose && c.worst_s <= gdp_loose;
    v.put(std::to_string(runs) + "-run chain of " + std::to_string(RCH) + "-row halves (" +
              std::to_string(int64_t(runs) * RCH) + " rows, slow-decay heads 3 and 11)",
          ok,
          std::string("gated output and end state ") + (ok ? "within" : "OUTSIDE") +
              " the loose bug-check tolerance at every run");
    return ok;
}

// ME11: the scan's replayed start. A chunk of two halves from bank 1 with the first k (0 .. 5) steps of record set 1
// replayed (synthetic records: normalized k, decays in [0.3, 1), deltas, with denormal decays at k = 5), against the
// same chunk from bank 0 holding that matrix as the host replay makes it (gdn_ref.h, the device's operations) with no
// records: the outputs of both halves and the end state bitwise, and the end state against FP64 loosely. The second
// half works in place in the output bank. Set 0 is poisoned, so a read of the wrong set shows.
bool replayed_start(dev & d, int32_t k, verdict & v, const std::vector<float> & w) {
    rng r(0x11e0 + uint64_t(k));
    std::vector<float> s(static_cast<size_t>(SV)), rec(size_t(fn::gdn::n_rec_set) * fn::gdn::rec_set_floats, NAN);
    for (float & f : s) {
        f = static_cast<float>(r.normal() * 0.5);
    }
    float * set1 = rec.data() + fn::gdn::rec_set_floats;
    for (int32_t t = 0; t < fn::gdn::max_rows; ++t) {
        float * st = set1 + int64_t(t) * fn::gdn::rec_step_floats;
        for (int32_t g = 0; g < NG; ++g) {
            double x[HD], ss = 0.0;
            for (int32_t i = 0; i < HD; ++i) {
                x[i] = r.normal();
                ss += x[i] * x[i];
            }
            for (int32_t i = 0; i < HD; ++i) {
                st[fn::gdn::rec_k + g * HD + i] = static_cast<float>(x[i] / std::sqrt(ss));
            }
        }
        for (int32_t h = 0; h < NH; ++h) {
            st[fn::gdn::rec_decay + h] = t == 4 && h == 5 ? 1e-40f : static_cast<float>(r.uniform(0.3, 1.0));
        }
        for (int32_t i = 0; i < NH * HD; ++i) {
            st[fn::gdn::rec_delta + i] = static_cast<float>(r.normal() * 0.3);
        }
    }
    std::vector<float> m(s.size());
    fn::gdn::ref::replay_host(s.data(), set1, k, m.data());
    std::vector<double> m64(m.begin(), m.end());
    std::vector<run_in> halves(2);
    make_run(700, 0x11e7 + uint64_t(k), false, halves[0]);
    make_run(65, 0x11e8 + uint64_t(k), false, halves[1]);
    std::vector<float> out[2], end[2];
    double worst_g = 0, worst_s = 0;
    std::string why;
    for (int32_t arm = 0; arm < 2; ++arm) {
        // arm 0: bank 1 and k records of set 1; arm 1: bank 0 holding the replayed matrix, no records
        const uint32_t gsel = arm == 0 ? fn::round::gdn_word(1, 1, k) : fn::round::gdn_word(0, 1, 0);
        if (!d.state.poison() ||
            !cuda_ok(cudaMemcpy(d.state.as<float>() + int64_t(fn::round::gdn_bank(gsel)) * SV,
                                (arm == 0 ? s : m).data(), size_t(SV) * 4, cudaMemcpyHostToDevice),
                     "state") ||
            !d.rec.put(rec) || !d.st64.put(m64)) {
            return false;
        }
        chain c{ d, w };
        c.cur = 1;
        c.gsel = gsel;
        for (int32_t h = 0; h < 2; ++h) {
            std::vector<float> g;
            if (!c.run(halves[size_t(h)], static_cast<int32_t>(halves[size_t(h)].g.size() / NH), why) ||
                !d.gated.get(g, halves[size_t(h)].z.size())) {
                v.put("replayed start k " + std::to_string(k), false, why);
                return false;
            }
            out[arm].insert(out[arm].end(), g.begin(), g.end());
        }
        end[arm] = c.end_state;
        worst_g = std::max(worst_g, c.worst_g);
        worst_s = std::max(worst_s, c.worst_s);
        if (c.gsel != fn::round::gdn_after_prompt(gsel)) {
            v.put("replayed start k " + std::to_string(k), false, "the chunk's selector");
            return false;
        }
    }
    const bool same = out[0] == out[1] && std::memcmp(end[0].data(), end[1].data(), end[0].size() * 4) == 0;
    const bool loose = worst_g <= gdp_loose && worst_s <= gdp_loose;
    v.put("replayed start k " + std::to_string(k) + " (two halves, the second in place; ME11)", same && loose,
          std::string("outputs and end state against the materialized start ") + (same ? "bitwise" : "DIFFER") +
              "; against FP64 " + (loose ? "within" : "OUTSIDE") + " the loose tolerance");
    return same && loose;
}

// the convolution against FP64, its new history bitwise; the norms and the gates against FP64
// PB3: half_in, the input as the product's FP16 output ([rows][qkv | z]; the reference and history use its values)
bool conv_prep(dev & d, int32_t rows, verdict & v, bool half_in = false) {
    rng                r(0xc0 + uint64_t(rows));
    // SP2: the history holds every version (6)
    std::vector<float> qkv(size_t(rows) * NCH), cw(size_t(NCH) * 4), hist(size_t(fn::round::n_version) * NCH * 3),
        al(size_t(rows) * NH), be(size_t(rows) * NH), dt(NH), aa(NH);
    std::vector<uint16_t> qz(half_in ? size_t(rows) * LQZ : 0, 0);
    for (size_t i = 0; i < qkv.size(); ++i) {
        qkv[i] = static_cast<float>(r.normal());
        if (half_in) {
            const uint16_t hv = fn::quant::f32_to_f16(qkv[i]);
            qz[(i / NCH) * LQZ + i % NCH] = hv;
            qkv[i] = fn::quant::f16_to_f32(hv);
        }
    }
    for (float & f : cw) {
        f = static_cast<float>(r.normal() * 0.5);
    }
    for (float & f : hist) {
        f = static_cast<float>(r.normal());
    }
    for (size_t i = 0; i < al.size(); ++i) {
        al[i] = static_cast<float>(r.normal() * 4.0);  // spans softplus's two branches
        be[i] = static_cast<float>(r.normal() * 3.0);
    }
    for (int32_t h = 0; h < NH; ++h) {
        dt[size_t(h)] = static_cast<float>(r.normal());
        aa[size_t(h)] = static_cast<float>(-std::exp(r.normal()));
    }
    const uint32_t cur = 3;
    dbuf dq(qkv.size() * 4), dw(cw.size() * 4), dh(hist.size() * 4), dal(al.size() * 4), dbe(be.size() * 4), ddt(256),
        daa(256), dg(al.size() * 4), dbt(al.size() * 4);
    if (!dq.put(qkv) || !dw.put(cw) || !dh.put(hist) || !dal.put(al) || !dbe.put(be) || !ddt.put(dt) || !daa.put(aa) ||
        !d.cur.put(std::vector<uint32_t>{cur}) || !d.conv.poison() || (half_in && !d.qz16.put(qz))) {
        return false;
    }
    fn::gdn::prompt_args a;
    a.rows     = rows;
    a.qkv = half_in ? nullptr : dq.as<float>();
    a.qkv16 = half_in ? d.qz16.as<uint16_t>() : nullptr;
    a.ld16 = half_in ? LQZ : 0;
    a.conv_out = d.conv.as<float>();
    a.conv_w   = dw.as<float>();
    a.hist     = dh.as<float>();
    a.cur      = d.cur.as<uint32_t>();
    a.v_in     = 1;
    a.v_out    = 2;
    a.alpha    = dal.as<float>();
    a.beta_raw = dbe.as<float>();
    a.dt       = ddt.as<float>();
    a.a        = daa.as<float>();
    a.g        = dg.as<float>();
    a.beta     = dbt.as<float>();
    std::vector<float> conv_pre, conv_post, hout, gg, bb;
    const bool         ok0 = fn::gdn::gdn_prompt_conv(a, fn::ptest::test_stream()) &&
                             cuda_ok(cudaDeviceSynchronize(), "conv") && d.conv.get(conv_pre, size_t(rows) * NCH) &&
                             fn::gdn::gdn_prompt_prep(a, fn::ptest::test_stream()) &&
                             cuda_ok(cudaDeviceSynchronize(), "prep") && d.conv.get(conv_post, size_t(rows) * NCH) &&
                             dh.get(hout, hist.size()) && dg.get(gg, al.size()) && dbt.get(bb, al.size());
    if (!ok0) {
        v.put("conv and prep " + std::to_string(rows), false, "launch failed");
        return false;
    }
    const float *       h0 = hist.data() + size_t(fn::round::version_slot(cur, 1)) * NCH * 3;
    std::vector<double> ref(size_t(rows) * NCH);
    for (int32_t t = 0; t < rows; ++t) {
        for (int32_t c = 0; c < NCH; ++c) {
            const auto x = [&](int32_t k) {
                return k >= 0 ? double(qkv[size_t(k) * NCH + c]) : double(h0[size_t(c) * 3 + 3 + k]);
            };
            const double acc         = x(t - 3) * cw[size_t(c) * 4] + x(t - 2) * cw[size_t(c) * 4 + 1] +
                                       x(t - 1) * cw[size_t(c) * 4 + 2] + x(t) * cw[size_t(c) * 4 + 3];
            ref[size_t(t) * NCH + c] = acc * sig(acc);
        }
    }
    const double  ec      = unit_error(conv_pre.data(), ref.data(), rows, NCH);
    bool          hist_ok = true;
    const float * ho      = hout.data() + size_t(fn::round::version_slot(cur, 2)) * NCH * 3;
    for (int32_t c = 0; c < NCH; ++c) {
        for (int32_t j = 0; j < 3; ++j) {
            const int32_t t    = rows - 3 + j;
            const float   want = t >= 0 ? qkv[size_t(t) * NCH + c] : h0[size_t(c) * 3 + 3 + t];
            hist_ok            = hist_ok && std::memcmp(&ho[size_t(c) * 3 + j], &want, 4) == 0;
        }
    }
    // the norms of the conv output's q and k groups, and the gates
    std::vector<double> nref(size_t(rows) * 2048);
    for (int32_t t = 0; t < rows; ++t) {
        for (int32_t v8 = 0; v8 < 16; ++v8) {
            double ss = 0.0;
            for (int32_t i = 0; i < HD; ++i) {
                const double x = conv_pre[size_t(t) * NCH + size_t(v8) * HD + i];
                ss += x * x;
            }
            const double rn = 1.0 / std::sqrt(ss + 1e-6);
            for (int32_t i = 0; i < HD; ++i) {
                nref[size_t(t) * 2048 + size_t(v8) * HD + i] = conv_pre[size_t(t) * NCH + size_t(v8) * HD + i] * rn;
            }
        }
    }
    std::vector<float> ngot(size_t(rows) * 2048);
    for (int32_t t = 0; t < rows; ++t) {
        std::memcpy(&ngot[size_t(t) * 2048], &conv_post[size_t(t) * NCH], size_t(2048) * 4);
    }
    const double        en = unit_error(ngot.data(), nref.data(), rows, 2048);
    std::vector<double> gref(al.size()), bref(al.size());
    for (int32_t t = 0; t < rows; ++t) {
        for (int32_t h = 0; h < NH; ++h) {
            const double xg          = double(al[size_t(t) * NH + h]) + dt[size_t(h)];
            const double sp          = xg > 20.0 ? xg : std::log1p(std::exp(xg));
            gref[size_t(t) * NH + h] = sp * aa[size_t(h)];
            bref[size_t(t) * NH + h] = sig(be[size_t(t) * NH + h]);
        }
    }
    const double eg = unit_error(gg.data(), gref.data(), rows, NH), eb = unit_error(bb.data(), bref.data(), rows, NH);
    // PB3: fuse_norm (the convolution normalizes q and k itself) gives the two passes' output and history bitwise
    std::vector<float> conv_f, hout_f;
    a.fuse_norm = true;
    const bool fused_ok = d.conv.poison() && fn::gdn::gdn_prompt_conv(a, fn::ptest::test_stream()) &&
                          fn::gdn::gdn_prompt_prep(a, fn::ptest::test_stream()) &&
                          cuda_ok(cudaDeviceSynchronize(), "fused") && d.conv.get(conv_f, size_t(rows) * NCH) &&
                          dh.get(hout_f, hist.size()) &&
                          std::memcmp(conv_f.data(), conv_post.data(), conv_f.size() * 4) == 0 &&
                          std::memcmp(hout_f.data(), hout.data(), hout.size() * 4) == 0;
    const bool ok = ec <= 1e-6 && hist_ok && en <= 1e-6 && eg <= 1e-6 && eb <= 1e-6 && fused_ok;
    v.put("conv and prep " + std::to_string(rows) + " rows" + (half_in ? " (FP16 input)" : ""), ok,
          "conv " + e3(ec) + ", history " + (hist_ok ? "bitwise" : "DIFFERS") + ", q/k norms " + e3(en) +
              ", log decay " + e3(eg) + ", beta " + e3(eb) + " (bound 1e-6 a row); fused conv and norms " +
              (fused_ok ? "bitwise" : "DIFFER"));
    return ok;
}

}  // namespace

int fnfl_gdp_test(int device, const char * mode) {
    const bool san = mode != nullptr && std::strcmp(mode, "san") == 0;
    verdict    v("gdp-test");
    if (!cuda_ok(cudaSetDevice(device), "device")) {
        return 1;
    }
    auto               d = std::make_unique<dev>();
    std::vector<float> w(HD);
    rng                r(0x5757);
    for (float & f : w) {
        f = static_cast<float>(r.uniform(0.5, 1.5));
    }
    if (!d->w.put(w) || !d->err.zero()) {
        return 1;
    }
    for (const int32_t rows : san ? std::vector<int32_t>{2, 70} : std::vector<int32_t>{1, 2, 3, 70, 1536, 4096}) {
        conv_prep(*d, rows, v);
    }
    for (const int32_t rows : san ? std::vector<int32_t>{70} : std::vector<int32_t>{1, 2, 70, 4096}) {
        conv_prep(*d, rows, v, true);
    }
    for (const int32_t rows : san ? std::vector<int32_t>{65} : std::vector<int32_t>{1, 63, 64, 65, 100, 1536, 4096}) {
        single(*d, rows, false, v, w);
    }
    for (const int32_t rows : san ? std::vector<int32_t>{65} : std::vector<int32_t>{1, 65, 4096}) {
        single(*d, rows, false, v, w, true);
    }
    for (const int32_t rows : san ? std::vector<int32_t>{65} : std::vector<int32_t>{1, 65, 4096}) {
        single(*d, rows, false, v, w, true, true);
    }
    single(*d, san ? 65 : 1536, true, v, w);
    for (int32_t k = san ? 5 : 0; k <= fn::gdn::max_rows; ++k) {
        replayed_start(*d, k, v, w);
    }
    if (!san) {
        chain_case(*d, 6, v, w);
        chain_case(*d, 128, v, w);
    }
    fn::gdn::prompt_resources rp{}, rs{};
    if (fn::gdn::gdn_prompt_recur_resources(rp, rs)) {
        // reported, not judged (PB3 Amendment 3: resources are a cost)
        printf(
            "gdp-test: recur resources: prep %d regs, %d B local, %d B shared; scan %d regs, %d B local, %d B shared\n",
            rp.regs, rp.local_bytes, rp.dyn_shared_bytes, rs.regs, rs.local_bytes, rs.dyn_shared_bytes);
    }
    std::vector<uint32_t> ew;
    if (d->err.get(ew, 1)) {
        v.put("error word", ew[0] == 0, std::to_string(ew[0]));
    }
    return v.finish();
}
