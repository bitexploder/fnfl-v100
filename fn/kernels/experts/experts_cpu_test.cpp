#include "fn/kernels/experts/experts_cpu_test.h"

#include "fn/kernels/experts/experts_cpu.h"
#include "fn/kernels/experts/experts_test.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/runtime/expert_cache.h"
#include "fn/runtime/loan.h"
#include "fn/runtime/prefill.h"
#include "fn/runtime/admission_test.h"
#include "fn/runtime/miss_split_test.h"
#include "fn/runtime/prompt_plan_test.h"
#include "fn/runtime/slots_test.h"
#include "fn/runtime/service_loop.h"

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using fn::experts::fmt;
namespace cpu = fn::experts::cpu;

constexpr int32_t D = 2560;
constexpr int32_t NFF = 640;
constexpr int32_t NCH = 320;
constexpr int32_t R = cpu::job_max_rows; // MX7: a job's rows, 1..8 (Strata's MAXT); decode windows take 1..5

// The bound (ME2's numerical contract): each output is a sum of exact integer block products, each scaled by an F32
// product of at most three exactly representable factors (the weight block's FP16 scale or its sub-scale, the
// activation block's scale), accumulated in F32 over at most 80 32-value blocks (vector lanes, then one horizontal
// sum). Each term thus carries at most 3 + 80 roundings of u = 2^-24 relative to its magnitude, so
// |error| <= gamma x A, gamma = 128 u (83 rounded up), A = sum over the terms of |term|. A term's magnitude is at most
// (|w_i| + o_i) |x_i| per value, o_i the block's offset that the kernels apply separately (Q2_0: d, Q4_K / Q5_K:
// dmin x m of the sub-block, Q5_1: m; else 0), so A <= sum_i (|w_i| + o_i) |x_i| (+ |m| |s| a Q5_1 block, its
// activation sum taken from the block). SwiGLU (F32 g / (1 + expf(-g)) x u): the gate's and up's bounds propagated
// to first order plus their product, and 8 u of |h| for expf, the division and the product.
constexpr double u32 = 1.0 / 16777216.0;
constexpr double gamma_k = 128 * u32;

fn::quant::wfmt qf(fmt f) {
    using fn::quant::wfmt;
    static const wfmt q[11] = {wfmt::iq2_xxs, wfmt::iq2_xs, wfmt::iq2_s, wfmt::iq3_xxs, wfmt::iq3_s, wfmt::q2_0,
                               wfmt::iq4_nl,  wfmt::q8_0,   wfmt::q4_k,  wfmt::q5_k,    wfmt::q5_1}; // fmt's order
    return q[static_cast<int32_t>(f)];
}

uint16_t u16(const uint8_t * p) {
    uint16_t v = 0;
    std::memcpy(&v, p, 2);
    return v;
}

// ggml's get_scale_min_k4: sub-block j's 6-bit scale and min of a K-quant block's 12 scale bytes
void scale_min_k4(int32_t j, const uint8_t * q, int32_t & sc, int32_t & m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// one GGUF row's values and each value's offset o_i (the bound's), n values
void decode_row(fmt f, const uint8_t * row, int32_t n, std::vector<double> & w, std::vector<double> & o) {
    const fn::quant::wfmt q = qf(f);
    std::vector<float> v(static_cast<size_t>(n));
    fn::quant::decode_row(q, row, v.data(), n);
    w.assign(v.begin(), v.end());
    o.assign(size_t(n), 0.0);
    const int32_t bv = fn::quant::info(q).block_values, bb = fn::quant::info(q).block_bytes;
    for (int32_t b = 0; b < n / bv; ++b) {
        const uint8_t * blk = row + size_t(b) * size_t(bb);
        for (int32_t i = 0; i < bv; ++i) {
            double off = 0;
            if (f == fmt::q2_0) {
                off = std::fabs(double(fn::quant::f16_to_f32(u16(blk))));
            } else if (f == fmt::q5_1) {
                off = std::fabs(double(fn::quant::f16_to_f32(u16(blk + 2))));
            } else if (f == fmt::q4_k || f == fmt::q5_k) {
                int32_t sc = 0, m = 0;
                scale_min_k4(i / 32, blk + 4, sc, m);
                off = std::fabs(double(fn::quant::f16_to_f32(u16(blk + 2))) * m);
            }
            o[size_t(b) * bv + size_t(i)] = off;
        }
    }
}

// random activations in FNFL's format: codes in [-127, 127], each run's d in [1e-5, 1e-3) and s = d x sum. The
// synthetic weights are larger than a model's; with d up to 1e-1 the SwiGLU outputs reach 1e6 and the down
// activation's F16 scale (Q8_0 d, Q8_1 s, as ggml's) overflows
void act_rows(std::mt19937 & rng, int32_t rows, std::vector<int8_t> & codes, std::vector<float> & sc) {
    std::uniform_int_distribution<int32_t> c(-127, 127);
    std::uniform_real_distribution<float> d(1e-5f, 1e-3f);
    codes.assign(size_t(rows) * D, 0);
    sc.assign(size_t(rows) * (D / 32) * 2, 0.0f);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t q = 0; q < D / 32; ++q) {
            int32_t sum = 0;
            for (int32_t i = 0; i < 32; ++i) {
                const int32_t v = c(rng);
                codes[size_t(r) * D + q * 32 + i] = static_cast<int8_t>(v);
                sum += v;
            }
            const float dd = d(rng);
            sc[(size_t(r) * (D / 32) + q) * 2] = dd;
            sc[(size_t(r) * (D / 32) + q) * 2 + 1] = dd * static_cast<float>(sum);
        }
    }
}

// the scaled errors of one product kind: |err| / max |ref| (max, p99) and |err| / bound (max)
struct scaled {
    std::vector<double> err, ref, bnd;
    int64_t bad_got = 0, bad_ref = 0; // non-finite outputs and references: a failure
    void add(double got, double want, double bound) {
        bad_got += std::isfinite(got) ? 0 : 1;
        bad_ref += std::isfinite(want) && std::isfinite(bound) ? 0 : 1;
        err.push_back(std::isfinite(got - want) ? std::fabs(got - want) : HUGE_VAL);
        ref.push_back(std::isfinite(want) ? std::fabs(want) : 0.0);
        bnd.push_back(bound);
    }
    void report(const char * what, bool & ok) const {
        double mref = 0, worst = 0;
        for (size_t i = 0; i < err.size(); ++i) {
            mref = std::max(mref, ref[i]);
            worst = std::max(worst, bnd[i] > 0 ? err[i] / bnd[i] : (err[i] > 0 ? HUGE_VAL : 0.0));
        }
        std::vector<double> s(err.size());
        for (size_t i = 0; i < err.size(); ++i) {
            s[i] = mref > 0 ? err[i] / mref : err[i];
        }
        std::sort(s.begin(), s.end());
        const double p99 = s.empty() ? 0 : s[std::min(s.size() - 1, size_t(0.99 * double(s.size())))];
        const bool good = worst <= 1.0 && !s.empty() && bad_got == 0 && bad_ref == 0;
        printf("ex-cpu-test:   %s: %zu outputs, |err| / max|ref| max %.3g p99 %.3g; max |err| / bound %.3g; "
               "non-finite: %lld outputs, %lld references %s\n",
               what, err.size(), s.empty() ? 0 : s.back(), p99, worst, (long long) bad_got, (long long) bad_ref,
               good ? "ok" : "FAIL");
        ok = ok && good;
    }
};

bool load(fmt f, const std::string & dir, fn::experts::synth & s) {
    const bool down = f == fmt::q8_0 || f == fmt::q2_0 || f == fmt::iq4_nl || f == fmt::q5_1;
    if (!fn::experts::load_synth(f, dir, s, false, down)) {
        printf("ex-cpu-test: %s: the synthetic experts could not be loaded or made in %s\n",
               fn::experts::fmt_name(f), dir.c_str());
        return false;
    }
    return true;
}

// card c's packed gate/up and down of synthetic expert e
void pack_gu(const fn::experts::synth & s, int32_t e, int32_t c, std::vector<uint8_t> & out) {
    out.assign(size_t(fn::experts::gate_up_bytes(s.f)), 0);
    const uint8_t * x = s.expert(e);
    fn::experts::pack_gate_up(s.f, x + size_t(c) * NCH * s.row_bytes, x + (size_t(NFF) + size_t(c) * NCH) * s.row_bytes,
                              s.row_bytes, out.data());
}
void pack_dn(const fn::experts::synth & s, int32_t e, int32_t c, std::vector<uint8_t> & out) {
    out.assign(size_t(fn::experts::down_bytes(s.f)), 0);
    fn::experts::pack_down(s.f, s.expert(e) + size_t(c) * (s.row_bytes / 2), s.row_bytes, out.data());
}

bool round_trip_gu(const fn::experts::synth & s) {
    if (cpu::gu_row_bytes(s.f) != int64_t(s.row_bytes)) {
        printf("ex-cpu-test: %s: row bytes %lld, GGUF's %zu\n", fn::experts::fmt_name(s.f),
               (long long)cpu::gu_row_bytes(s.f), s.row_bytes);
        return false;
    }
    std::vector<uint8_t> packed, tile(size_t(cpu::scratch_bytes));
    int64_t rows = 0;
    for (const int32_t e : {0, 7, 16}) {
        for (int32_t c = 0; c < 2; ++c) {
            pack_gu(s, e, c, packed);
            const uint8_t * gate = s.expert(e) + size_t(c) * NCH * s.row_bytes;
            const uint8_t * up = s.expert(e) + (size_t(NFF) + size_t(c) * NCH) * s.row_bytes;
            for (int32_t g = 0; g < NCH / 32; ++g) {
                cpu::unpack_gate_up_tile(s.f, packed.data(), g, tile.data());
                for (int32_t r = 0; r < 64; ++r) {
                    const uint8_t * want = (r < 32 ? gate : up) + size_t(32 * g + (r & 31)) * s.row_bytes;
                    if (std::memcmp(tile.data() + size_t(r) * s.row_bytes, want, s.row_bytes) != 0) {
                        printf("ex-cpu-test: %s round trip: expert %d card %d tile %d row %d differs\n",
                               fn::experts::fmt_name(s.f), e, c, g, r);
                        return false;
                    }
                    ++rows;
                }
            }
        }
    }
    printf("ex-cpu-test: %s gate/up round trip: %lld rows of %zu bytes bitwise (3 experts, both cards, 10 tiles of 64 "
           "rows; scratch %lld bytes)\n",
           fn::experts::fmt_name(s.f), (long long)rows, s.row_bytes, (long long)cpu::gu_scratch_bytes(s.f));
    return true;
}

bool round_trip_dn(const fn::experts::synth & s) {
    if (cpu::dn_row_bytes(s.f) != int64_t(s.row_bytes)) {
        printf("ex-cpu-test: %s: row bytes %lld, GGUF's %zu\n", fn::experts::fmt_name(s.f),
               (long long)cpu::dn_row_bytes(s.f), s.row_bytes);
        return false;
    }
    std::vector<uint8_t> p0, p1, piece(size_t(cpu::scratch_bytes));
    int64_t rows = 0;
    for (const int32_t e : {0, 7, 16}) {
        pack_dn(s, e, 0, p0);
        pack_dn(s, e, 1, p1);
        for (int32_t n0 = 0; n0 < D; n0 += cpu::down_piece) {
            cpu::unpack_down_rows(s.f, p0.data(), p1.data(), n0, cpu::down_piece, piece.data());
            for (int32_t i = 0; i < cpu::down_piece; ++i) {
                if (std::memcmp(piece.data() + size_t(i) * s.row_bytes, s.expert(e) + size_t(n0 + i) * s.row_bytes,
                                s.row_bytes) != 0) {
                    printf("ex-cpu-test: %s round trip: expert %d row %d differs\n", fn::experts::fmt_name(s.f), e,
                           n0 + i);
                    return false;
                }
                ++rows;
            }
        }
    }
    printf("ex-cpu-test: %s down round trip: %lld rows of %zu bytes bitwise (3 experts, 20 pieces of %d rows; scratch "
           "%lld bytes)\n",
           fn::experts::fmt_name(s.f), (long long)rows, s.row_bytes, cpu::down_piece,
           (long long)cpu::dn_scratch_bytes(s.f));
    return true;
}

// the down activation's values (and, Q5_1's, each 32-value block's sum s as the kernel takes it)
void act_values(fmt df, const void * a, std::vector<double> & x, std::vector<double> & bsum) {
    x.assign(NFF, 0.0);
    bsum.assign(NFF / 32, 0.0);
    if (df == fmt::q2_0) {
        const cpu::act_q & q = *static_cast<const cpu::act_q *>(a);
        for (int32_t i = 0; i < NFF; ++i) {
            x[size_t(i)] = double(q.scale[i / 32]) * q.q[i];
        }
        return;
    }
    for (int32_t b = 0; b < NFF / 32; ++b) {
        if (df == fmt::q5_1) {
            const cpu::block_q8_1 & q = static_cast<const cpu::block_q8_1 *>(a)[b];
            for (int32_t i = 0; i < 32; ++i) {
                x[size_t(32 * b + i)] = double(fn::quant::f16_to_f32(q.d)) * q.qs[i];
            }
            bsum[size_t(b)] = double(fn::quant::f16_to_f32(q.s));
        } else {
            const cpu::block_q8_0 & q = static_cast<const cpu::block_q8_0 *>(a)[b];
            for (int32_t i = 0; i < 32; ++i) {
                x[size_t(32 * b + i)] = double(fn::quant::f16_to_f32(q.d)) * q.qs[i];
            }
        }
    }
}

// ME6 rider: Q2_0 down width invariance. The pool's R down activations of expert e's job j against its GGUF rows:
// each row alone (nt = 1, the outputs cut at 13) and all R in one call (nt = R; MX7: 8, was 5) give the same bits,
// and those bits are the pool's own outputs (its pieces and card halves through the packed scratch)
bool q2_width(const fn::experts::synth & sd, int32_t e, int32_t j, const cpu::pool & pl, const cpu::job & jb) {
    const cpu::act_q * a[R];
    for (int32_t t = 0; t < R; ++t) {
        a[t] = static_cast<const cpu::act_q *>(pl.test_act(j, t));
    }
    std::vector<float> o1(size_t(R) * D), o5(size_t(R) * D);
    float * p1[R];
    float * p5[R];
    for (int32_t t = 0; t < R; ++t) {
        p1[t] = o1.data() + size_t(t) * D;
        p5[t] = o5.data() + size_t(t) * D;
    }
    const uint8_t * w = sd.expert(e);
    for (int32_t t = 0; t < R; ++t) {
        cpu::q2_down_rows(w, sd.row_bytes, NFF / 64, a + t, 1, p1 + t, 0, 13);
        cpu::q2_down_rows(w, sd.row_bytes, NFF / 64, a + t, 1, p1 + t, 13, D);
    }
    cpu::q2_down_rows(w, sd.row_bytes, NFF / 64, a, R, p5, 0, D);
    int64_t diff1 = 0, diffp = 0;
    for (int32_t t = 0; t < R; ++t) {
        diff1 += std::memcmp(p1[t], p5[t], size_t(D) * sizeof(float)) != 0 ? 1 : 0;
        diffp += std::memcmp(jb.out[t], p5[t], size_t(D) * sizeof(float)) != 0 ? 1 : 0;
    }
    const bool good = diff1 == 0 && diffp == 0;
    printf("ex-cpu-test:   q2_0 down width (expert %d, %d rows x %d outputs): nt 1 vs nt %d %lld rows differ, the "
           "pool's vs nt %d %lld rows differ %s\n",
           e, R, D, R, (long long)diff1, R, (long long)diffp, good ? "ok" : "FAIL");
    return good;
}

// MX7: whole-expert width invariance through the pool, every format: jobs[2] at nt = R (their outputs out8, SwiGLU
// h8 [2][R][640] as the pool left them) against each of their rows alone, 2 R jobs of nt = 1 in one run: the same
// SwiGLU and output bits (the gate/up and down kernels, the down quantizer and the pool's cut alike)
bool width_jobs(fmt gf, fmt df, const cpu::block_q8_k * act, const cpu::job * jobs, const std::vector<float> & h8,
                cpu::pool & pl) {
    cpu::job one[2 * R];
    std::vector<float> o1(size_t(2) * R * D, 1.0f);
    for (int32_t j = 0; j < 2; ++j) {
        for (int32_t t = 0; t < R; ++t) {
            cpu::job & x = one[j * R + t];
            x = jobs[j];
            x.nt = 1;
            x.row[0] = jobs[j].row[t];
            x.out[0] = o1.data() + (size_t(j) * R + size_t(t)) * D;
        }
    }
    cpu::run_stats st;
    if (!pl.run(gf, df, act, one, 2 * R, st)) {
        printf("ex-cpu-test: the pool stalled\n");
        return false;
    }
    int64_t dh = 0, dout = 0;
    for (int32_t j = 0; j < 2; ++j) {
        for (int32_t t = 0; t < R; ++t) {
            dh += std::memcmp(pl.test_h(j * R + t, 0), h8.data() + (size_t(j) * R + size_t(t)) * NFF,
                              NFF * sizeof(float)) != 0;
            dout += std::memcmp(one[j * R + t].out[0], jobs[j].out[t], D * sizeof(float)) != 0;
        }
    }
    const bool good = dh == 0 && dout == 0;
    printf("ex-cpu-test:   width (2 experts x %d rows): nt 1 vs nt %d SwiGLU %lld rows differ, outputs %lld rows "
           "differ %s\n",
           R, R, (long long)dh, (long long)dout, good ? "ok" : "FAIL");
    return good;
}

// whole experts 0 and 7 through the pool at B = 1..R (MX7: 1..8, was 1..5) against FP64
bool fp64_pair(const fn::experts::synth & sg, const fn::experts::synth & sd, cpu::pool & pl) {
    const fmt gf = sg.f, df = sd.f;
    printf("ex-cpu-test: %s / %s, experts 0 and 7 at B = 1..%d:\n", fn::experts::fmt_name(gf),
           fn::experts::fmt_name(df), R);
    std::vector<uint8_t> gu[2][2], dn[2][2];
    const int32_t ex[2] = {0, 7};
    for (int32_t j = 0; j < 2; ++j) {
        for (int32_t c = 0; c < 2; ++c) {
            pack_gu(sg, ex[j], c, gu[j][c]);
            pack_dn(sd, ex[j], c, dn[j][c]);
        }
    }
    // the experts' GGUF values (and offsets): gate/up rows [640] of 2560 (n = card x 320 + channel), down [2560] of 640
    std::vector<std::vector<double>> gw[2], go[2], uw[2], uo[2], dw[2], dof[2];
    for (int32_t j = 0; j < 2; ++j) {
        gw[j].resize(NFF), go[j].resize(NFF), uw[j].resize(NFF), uo[j].resize(NFF), dw[j].resize(D), dof[j].resize(D);
        for (int32_t n = 0; n < NFF; ++n) {
            decode_row(gf, sg.expert(ex[j]) + size_t(n) * sg.row_bytes, D, gw[j][size_t(n)], go[j][size_t(n)]);
            decode_row(gf, sg.expert(ex[j]) + size_t(NFF + n) * sg.row_bytes, D, uw[j][size_t(n)], uo[j][size_t(n)]);
        }
        for (int32_t n = 0; n < D; ++n) {
            decode_row(df, sd.expert(ex[j]) + size_t(n) * sd.row_bytes, NFF, dw[j][size_t(n)], dof[j][size_t(n)]);
        }
    }
    std::mt19937 rng(20261017u + 31u * uint32_t(gf) + uint32_t(df));
    std::vector<int8_t> codes;
    std::vector<float> scales;
    act_rows(rng, R, codes, scales);
    std::vector<cpu::block_q8_k> act(size_t(R) * cpu::n_q8k);
    for (int32_t b = 0; b < R; ++b) {
        cpu::gate_up_act(codes.data() + size_t(b) * D, scales.data() + size_t(b) * (D / 32) * 2,
                         act.data() + size_t(b) * cpu::n_q8k);
    }
    std::vector<double> xg(size_t(R) * D);
    for (int32_t b = 0; b < R; ++b) {
        for (int32_t i = 0; i < D; ++i) {
            const cpu::block_q8_k & q = act[size_t(b) * cpu::n_q8k + size_t(i / 256)];
            xg[size_t(b) * D + size_t(i)] = double(q.d) * q.qs[i % 256];
        }
    }
    scaled sh, sdn;
    double hmax = 0;
    std::vector<float> out(size_t(2) * R * D);
    for (int32_t nt = 1; nt <= R; ++nt) {
        cpu::job jobs[2];
        for (int32_t j = 0; j < 2; ++j) {
            jobs[j].gu[0] = gu[j][0].data();
            jobs[j].gu[1] = gu[j][1].data();
            jobs[j].dn[0] = dn[j][0].data();
            jobs[j].dn[1] = dn[j][1].data();
            jobs[j].nt = nt;
            for (int32_t t = 0; t < nt; ++t) {
                jobs[j].row[t] = j == 0 ? t : R - 1 - t; // expert 7's rows from the window's last
                jobs[j].out[t] = out.data() + (size_t(j) * R + size_t(t)) * D;
            }
        }
        cpu::run_stats st;
        if (!pl.run(gf, df, act.data(), jobs, 2, st)) {
            printf("ex-cpu-test: the pool stalled\n");
            return false;
        }
        if (nt == 5) { // ME2b: the diagnostic per-unit clocks change nothing computed; their thread-ms printed
            const std::vector<float> first = out;
            cpu::run_stats su;
            pl.set_unit_timing(true);
            const bool ran = pl.run(gf, df, act.data(), jobs, 2, su);
            pl.set_unit_timing(false);
            const bool same = ran && std::memcmp(first.data(), out.data(), out.size() * sizeof(float)) == 0;
            printf("ex-cpu-test:   unit clocks on (diagnostic, B=5, 2 experts): outputs %s; thread-ms gate/up "
                   "un-permute %.3f, product %.3f; down un-permute %.3f, product %.3f\n",
                   same ? "bitwise the same" : "DIFFERENT", su.gu_unpack_ms, su.gu_compute_ms, su.dn_unpack_ms,
                   su.dn_compute_ms);
            if (!same) {
                return false;
            }
#ifdef NDEBUG
            // MX7: a held pool (another caller inside run) refuses the run and writes nothing; released, it runs
            std::vector<float> mark(out.size(), -7.0f);
            out = mark;
            cpu::run_stats sh;
            pl.test_hold(true);
            const bool refused = !pl.run(gf, df, act.data(), jobs, 2, sh);
            pl.test_hold(false);
            const bool untouched = std::memcmp(mark.data(), out.data(), out.size() * sizeof(float)) == 0;
            // the run writes each job's nt rows; the rest keep the mark
            std::vector<float> want = first;
            for (int32_t j = 0; j < 2; ++j) {
                for (int32_t t = nt; t < R; ++t) {
                    std::fill_n(want.begin() + int64_t((size_t(j) * R + size_t(t)) * D), D, -7.0f);
                }
            }
            const bool again = pl.run(gf, df, act.data(), jobs, 2, sh) &&
                               std::memcmp(want.data(), out.data(), out.size() * sizeof(float)) == 0;
            printf("ex-cpu-test:   MX7 one caller: a held pool %s the run (outputs %s); released, the run %s %s\n",
                   refused ? "refuses" : "TAKES", untouched ? "untouched" : "WRITTEN",
                   again ? "repeats bitwise" : "DIFFERS", refused && untouched && again ? "PASS" : "FAIL");
            if (!(refused && untouched && again)) {
                return false;
            }
#endif
        }
        if (nt == R) { // MX7: width invariance at nt 1 against R, before the SwiGLU check reads the pool's h
            for (int32_t j = 0; j < 2 && df == fmt::q2_0; ++j) {
                if (!q2_width(sd, ex[j], j, pl, jobs[j])) {
                    return false;
                }
            }
            std::vector<float> h8(size_t(2) * R * NFF);
            for (int32_t j = 0; j < 2; ++j) {
                for (int32_t t = 0; t < R; ++t) {
                    std::memcpy(h8.data() + (size_t(j) * R + size_t(t)) * NFF, pl.test_h(j, t), NFF * sizeof(float));
                }
            }
            if (!width_jobs(gf, df, act.data(), jobs, h8, pl)) {
                return false;
            }
            // the FP64 check below reads the pool's SwiGLU and down activations of jobs: rerun them
            cpu::run_stats sr;
            if (!pl.run(gf, df, act.data(), jobs, 2, sr)) {
                printf("ex-cpu-test: the pool stalled\n");
                return false;
            }
        }
        for (int32_t j = 0; j < 2; ++j) {
            for (int32_t t = 0; t < nt; ++t) {
                const double * x = xg.data() + size_t(jobs[j].row[t]) * D;
                const float * h = pl.test_h(j, t);
                for (int32_t n = 0; n < NFF; ++n) {
                    double g = 0, ga = 0, uu = 0, ua = 0;
                    for (int32_t i = 0; i < D; ++i) {
                        g += gw[j][size_t(n)][size_t(i)] * x[i];
                        ga += (std::fabs(gw[j][size_t(n)][size_t(i)]) + go[j][size_t(n)][size_t(i)]) * std::fabs(x[i]);
                        uu += uw[j][size_t(n)][size_t(i)] * x[i];
                        ua += (std::fabs(uw[j][size_t(n)][size_t(i)]) + uo[j][size_t(n)][size_t(i)]) * std::fabs(x[i]);
                    }
                    const double sg_ = 1.0 / (1.0 + std::exp(-g)), silu = g * sg_, ref = silu * uu;
                    const double dsilu = std::fabs(sg_ * (1.0 + g * (1.0 - sg_)));
                    const double bg = gamma_k * ga, bu = gamma_k * ua;
                    hmax = std::max(hmax, std::fabs(ref));
                    sh.add(double(h[n]), ref, dsilu * bg * std::fabs(uu) + std::fabs(silu) * bu + 1.1 * bg * bu +
                                                 8 * u32 * std::fabs(ref));
                }
                // down, against the pool's own down activation
                std::vector<double> xd, bs;
                act_values(df, pl.test_act(j, t), xd, bs);
                const float * o = jobs[j].out[t];
                for (int32_t n = 0; n < D; ++n) {
                    double r = 0, a = 0;
                    for (int32_t i = 0; i < NFF; ++i) {
                        const double w = dw[j][size_t(n)][size_t(i)], off = dof[j][size_t(n)][size_t(i)];
                        if (df == fmt::q5_1) { // w = d q + m: d q x through the kernel's integer sum, m s per block
                            const double m = double(fn::quant::f16_to_f32(
                                u16(sd.expert(ex[j]) + size_t(n) * sd.row_bytes + size_t(i / 32) * 24 + 2)));
                            r += (w - m) * xd[size_t(i)];
                            a += std::fabs(w - m) * std::fabs(xd[size_t(i)]);
                            if (i % 32 == 0) {
                                r += m * bs[size_t(i / 32)];
                                a += std::fabs(m) * std::fabs(bs[size_t(i / 32)]);
                            }
                        } else {
                            r += w * xd[size_t(i)];
                            a += (std::fabs(w) + off) * std::fabs(xd[size_t(i)]);
                        }
                    }
                    sdn.add(double(o[n]), r, gamma_k * a);
                }
            }
        }
    }
    bool ok = true;
    printf("ex-cpu-test:   max |SwiGLU output| %.3g\n", hmax);
    sh.report("gate/up with SwiGLU", ok);
    sdn.report("down", ok);
    return ok;
}

// ME4: the native views against the reference path (Shell's addition; correctness only, no rates). A 512-expert
// layer's address space, reserved (only the probed experts' pages written): gate and up interleaved per expert at a
// padded stride (the synthetic files' layout: gate's 640 rows, then up's, then 4,096 bytes), down at its own padded
// stride (4,160 bytes past its 2,560 rows); boundary and middle experts 0, 1, 255, 256, 510, 511 (synthetic e % 17).
//   bytes:    native_equal: both cards' ten tiles and every down piece against the reconstruction of the packed parts;
//   products: the pool on the native rows and on the reconstruction, B = 1..8 (MX7; was 1..5), two experts a run
//             (0 and 511, 1 and 510, 255 and 256; expert two's rows from the window's last), outputs and SwiGLU
//             bitwise; the per-unit clocks on once: the same outputs, and no un-permute time on the native path;
//   lifetime: the views made unreadable (PROT_NONE) after the runs, then a reference run: the pool reads no view once
//             its run has returned (a read would fault), and its outputs are the earlier ones.
bool me4_native(const fn::experts::synth & sg, const fn::experts::synth & sd, cpu::pool & pl) {
    const fmt gf = sg.f, df = sd.f;
    const int64_t grb = int64_t(sg.row_bytes), drb = int64_t(sd.row_bytes);
    const int64_t gmat = NFF * grb, dmat = D * drb;
    const int64_t gstride = 2 * gmat + 4096, dstride = dmat + 4160;
    const size_t gbytes = size_t(512 * gstride), dbytes = size_t(512 * dstride);
    void * gm = mmap(nullptr, gbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    void * dm = mmap(nullptr, dbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (gm == MAP_FAILED || dm == MAP_FAILED) {
        printf("ex-cpu-test: ME4 native %s / %s: cannot reserve the layer\n", fn::experts::fmt_name(gf),
               fn::experts::fmt_name(df));
        return false;
    }
    uint8_t * g8 = static_cast<uint8_t *>(gm);
    uint8_t * d8 = static_cast<uint8_t *>(dm);
    cpu::native_layer v;
    v.gate = g8;
    v.up = g8 + gmat;
    v.down = d8;
    v.gu_row = grb;
    v.gu_expert = gstride;
    v.dn_row = drb;
    v.dn_expert = dstride;
    const int32_t probe[6] = {0, 1, 255, 256, 510, 511};
    std::vector<uint8_t> gu[6][2], dn[6][2];
    for (int32_t k = 0; k < 6; ++k) {
        const int32_t e = probe[k], x = e % 17;
        std::memcpy(g8 + size_t(e) * size_t(gstride), sg.expert(x), size_t(2 * gmat));
        std::memcpy(d8 + size_t(e) * size_t(dstride), sd.expert(x), size_t(dmat));
        for (int32_t c = 0; c < 2; ++c) {
            pack_gu(sg, x, c, gu[k][c]);
            pack_dn(sd, x, c, dn[k][c]);
        }
    }
    const auto ref_job = [&](int32_t k, cpu::job & j) {
        j = cpu::job();
        for (int32_t c = 0; c < 2; ++c) {
            j.gu[c] = gu[k][c].data();
            j.dn[c] = dn[k][c].data();
        }
    };
    bool ok = true;
    // bytes
    std::vector<uint8_t> scratch(size_t(cpu::scratch_bytes));
    for (int32_t k = 0; k < 6; ++k) {
        cpu::job j;
        ref_job(k, j);
        cpu::set_native(v, probe[k], j);
        std::string where;
        if (!cpu::native_equal(gf, df, j, scratch.data(), where)) {
            printf("ex-cpu-test: ME4 native %s / %s: expert %d's native rows differ at %s\n", fn::experts::fmt_name(gf),
                   fn::experts::fmt_name(df), probe[k], where.c_str());
            ok = false;
        }
    }
    // products
    std::mt19937 rng(20261018u + 31u * uint32_t(gf) + uint32_t(df));
    std::vector<int8_t> codes;
    std::vector<float> scales;
    act_rows(rng, R, codes, scales);
    std::vector<cpu::block_q8_k> act(size_t(R) * cpu::n_q8k);
    for (int32_t b = 0; b < R; ++b) {
        cpu::gate_up_act(codes.data() + size_t(b) * D, scales.data() + size_t(b) * (D / 32) * 2,
                         act.data() + size_t(b) * cpu::n_q8k);
    }
    const int32_t pairs[3][2] = {{0, 5}, {1, 4}, {2, 3}}; // probe indices: 0 / 511, 1 / 510, 255 / 256
    std::vector<float> o_ref(size_t(2) * R * D), o_nat(size_t(2) * R * D), h_ref(size_t(2) * R * NFF);
    std::vector<float> keep;
    int32_t runs = 0, same = 0;
    bool clocks = true;
    for (const auto & pr : pairs) {
        for (int32_t nt = 1; nt <= R; ++nt) {
            cpu::job rj[2], nj[2];
            for (int32_t i = 0; i < 2; ++i) {
                ref_job(pr[i], rj[i]);
                rj[i].nt = nt;
                for (int32_t t = 0; t < nt; ++t) {
                    rj[i].row[t] = i == 0 ? t : R - 1 - t;
                    rj[i].out[t] = o_ref.data() + (size_t(i) * R + size_t(t)) * D;
                }
                nj[i] = rj[i];
                cpu::set_native(v, probe[pr[i]], nj[i]);
                for (int32_t t = 0; t < nt; ++t) {
                    nj[i].out[t] = o_nat.data() + (size_t(i) * R + size_t(t)) * D;
                }
            }
            cpu::run_stats st;
            std::fill(o_ref.begin(), o_ref.end(), 0.0f);
            std::fill(o_nat.begin(), o_nat.end(), 1.0f);
            if (!pl.run(gf, df, act.data(), rj, 2, st)) {
                printf("ex-cpu-test: the pool stalled\n");
                munmap(gm, gbytes);
                munmap(dm, dbytes);
                return false;
            }
            for (int32_t i = 0; i < 2; ++i) {
                for (int32_t t = 0; t < nt; ++t) {
                    std::memcpy(h_ref.data() + (size_t(i) * R + size_t(t)) * NFF, pl.test_h(i, t), NFF * sizeof(float));
                }
            }
            const bool timed = nt == 5 && pr[0] == 0;
            pl.set_unit_timing(timed);
            const bool ran = pl.run(gf, df, act.data(), nj, 2, st);
            pl.set_unit_timing(false);
            bool eq = ran;
            for (int32_t i = 0; i < 2 && eq; ++i) {
                for (int32_t t = 0; t < nt && eq; ++t) {
                    eq = std::memcmp(rj[i].out[t], nj[i].out[t], D * sizeof(float)) == 0 &&
                         std::memcmp(h_ref.data() + (size_t(i) * R + size_t(t)) * NFF, pl.test_h(i, t),
                                     NFF * sizeof(float)) == 0;
                }
            }
            if (timed) {
                clocks = st.gu_unpack_ms == 0.0 && st.dn_unpack_ms == 0.0 && st.gu_compute_ms > 0.0;
                printf("ex-cpu-test:   native unit clocks (B=5, experts 0 and 511): thread-ms gate/up un-permute %.3f, "
                       "product %.3f; down un-permute %.3f, product %.3f %s\n",
                       st.gu_unpack_ms, st.gu_compute_ms, st.dn_unpack_ms, st.dn_compute_ms,
                       clocks ? "(no reconstruction: ok)" : "FAIL");
            }
            if (nt == R && pr[0] == 2) {
                keep.assign(o_ref.begin(), o_ref.end());
            }
            ++runs;
            same += eq ? 1 : 0;
            if (!eq) {
                printf("ex-cpu-test: ME4 native %s / %s: experts %d and %d at B=%d: outputs DIFFER from the "
                       "reconstruction's\n",
                       fn::experts::fmt_name(gf), fn::experts::fmt_name(df), probe[pr[0]], probe[pr[1]], nt);
            }
        }
    }
    // lifetime: the views unreadable, then the reference run of the last pair at B = R again
    bool life = mprotect(gm, gbytes, PROT_NONE) == 0 && mprotect(dm, dbytes, PROT_NONE) == 0;
    if (life) {
        cpu::job rj[2];
        for (int32_t i = 0; i < 2; ++i) {
            ref_job(pairs[2][i], rj[i]);
            rj[i].nt = R;
            for (int32_t t = 0; t < R; ++t) {
                rj[i].row[t] = i == 0 ? t : R - 1 - t;
                rj[i].out[t] = o_ref.data() + (size_t(i) * R + size_t(t)) * D;
            }
        }
        cpu::run_stats st;
        std::fill(o_ref.begin(), o_ref.end(), 0.0f);
        life = pl.run(gf, df, act.data(), rj, 2, st) && keep.size() == o_ref.size() &&
               std::memcmp(keep.data(), o_ref.data(), keep.size() * sizeof(float)) == 0;
    }
    munmap(gm, gbytes);
    munmap(dm, dbytes);
    const bool good = ok && same == runs && clocks && life;
    printf("ex-cpu-test: ME4 native %s / %s: 6 experts (0, 1, 255, 256, 510, 511) bytes %s; %d of %d runs (B=1..%d, "
           "3 pairs) bitwise the reconstruction's; strides gate/up %lld, down %lld bytes; views unreadable after the "
           "runs, reference run %s: %s\n",
           fn::experts::fmt_name(gf), fn::experts::fmt_name(df), ok ? "equal" : "DIFFER", same, runs, R,
           (long long)gstride, (long long)dstride, life ? "the same" : "FAILED", good ? "PASS" : "FAIL");
    return good;
}

// ME4: the loader's extent rule (native_fits) at its edges
bool me4_extents() {
    const uint64_t mb = 1000;
    const int64_t m = int64_t(mb);
    const bool ok = cpu::native_fits(0, 512 * mb, m, 512) && !cpu::native_fits(0, 512 * mb - 1, m, 512) &&
                    cpu::native_fits(7, 512 * mb + 7, m, 512) && !cpu::native_fits(8, 512 * mb + 7, m, 512) &&
                    !cpu::native_fits(9, 8, 1, 1) && !cpu::native_fits(8, 8, 0, 1) &&
                    !cpu::native_fits(0, UINT64_MAX, INT64_MAX, 512) &&
                    cpu::native_fits(0, UINT64_MAX, int64_t(UINT64_MAX / 512), 512) &&
                    !cpu::native_fits(0, 512 * mb, m, 0);
    printf("ex-cpu-test: ME4 native extents (exact fit, one byte short, offset past the end, zero sizes, no overflow "
           "at 2^64): %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

// MX7 (the prompt CPU share's host conversions): to_f16 bitwise fn::quant::f32_to_f16 (round to nearest even) over
// normal, subnormal, signed-zero and tail values, its flag for a value beyond 65,504, infinity or NaN in the vector
// part and in the tail; gate_up_act_f16 bitwise gate_up_act_f16 of the same row (deterministic) and each block's
// codes within a step of the F16 values (half, but one where Q8_K's +128 clamps to 127)
bool mx7_f16() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int64_t n = 2560 + 5; // a tail of 5
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const float m = i % 7 == 0 ? 1e-6f : i % 7 == 1 ? 60000.0f : i % 7 == 2 ? 3e-5f : 1.0f;
        x[size_t(i)] = u(rng) * m;
    }
    x[3] = 0.0f;
    x[4] = -0.0f;
    x[5] = 65504.0f;
    x[6] = -65504.0f;
    std::vector<uint16_t> h(static_cast<size_t>(n)), want(static_cast<size_t>(n));
    bool ok = cpu::to_f16(x.data(), n, h.data());
    int64_t diff = 0;
    for (int64_t i = 0; i < n; ++i) {
        want[size_t(i)] = fn::quant::f32_to_f16(x[size_t(i)]);
        diff += h[size_t(i)] != want[size_t(i)] ? 1 : 0;
    }
    ok = ok && diff == 0;
    // the flag: beyond 65,504, infinity and NaN, in the vector part (index 10) and the tail (index n - 2)
    const float bad[3] = {65520.0f, INFINITY, NAN};
    for (const float v : bad) {
        for (const int64_t at : {int64_t(10), n - 2}) {
            std::vector<float> y = x;
            y[size_t(at)] = v;
            ok = ok && !cpu::to_f16(y.data(), n, h.data());
        }
    }
    // gate_up_act_f16 on the converted row
    std::vector<cpu::block_q8_k> a(cpu::n_q8k), b(cpu::n_q8k);
    cpu::gate_up_act_f16(want.data(), a.data());
    cpu::gate_up_act_f16(want.data(), b.data());
    ok = ok && std::memcmp(a.data(), b.data(), a.size() * sizeof(cpu::block_q8_k)) == 0;
    double worst = 0;
    for (int32_t k = 0; k < cpu::n_q8k; ++k) {
        for (int32_t j = 0; j < 256; ++j) {
            const double v = fn::quant::f16_to_f32(want[size_t(k * 256 + j)]);
            const double e = std::fabs(v - double(a[size_t(k)].d) * a[size_t(k)].qs[j]);
            const double d = std::fabs(double(a[size_t(k)].d));
            worst = std::max(worst, d > 0 ? e / d : e);
        }
    }
    ok = ok && worst <= 1.0 + 1e-3; // half a step, or one where Q8_K clamps +128 to 127
    printf("ex-cpu-test: MX7 F16 conversions: to_f16 %lld of %lld values differ from f32_to_f16, flags beyond "
           "65,504 / infinity / NaN (vector part and tail); gate_up_act_f16 repeats, worst code error %.4f steps: %s\n",
           (long long)diff, (long long)n, worst, ok ? "PASS" : "FAIL");
    return ok;
}

// the pool's rate on one deployed class: 64 windows of 8 distinct experts at B = 1 from a 512-expert layer backing
// (expert e is synthetic e % 17), the pool's wall time a window; GB/s of packed expert bytes
void throughput(fmt gf, fmt df, const fn::experts::synth & sg, const fn::experts::synth & sd, cpu::pool & pl) {
    const int64_t gb = fn::experts::gate_up_bytes(gf), db = fn::experts::down_bytes(df), entry = 2 * (gb + db);
    std::vector<uint8_t> layer(size_t(512) * size_t(entry));
    std::vector<uint8_t> tmp;
    for (int32_t s = 0; s < 17; ++s) {
        for (int32_t c = 0; c < 2; ++c) {
            pack_gu(sg, s, c, tmp);
            for (int32_t e = s; e < 512; e += 17) {
                std::memcpy(layer.data() + size_t(e) * entry + size_t(c) * (gb + db), tmp.data(), size_t(gb));
            }
            pack_dn(sd, s, c, tmp);
            for (int32_t e = s; e < 512; e += 17) {
                std::memcpy(layer.data() + size_t(e) * entry + size_t(c) * (gb + db) + gb, tmp.data(), size_t(db));
            }
        }
    }
    std::mt19937 rng(7u);
    std::vector<int8_t> codes;
    std::vector<float> scales;
    act_rows(rng, 1, codes, scales);
    std::vector<cpu::block_q8_k> act(cpu::n_q8k);
    cpu::gate_up_act(codes.data(), scales.data(), act.data());
    std::vector<float> out(size_t(8) * D);
    cpu::run_stats st;
    std::vector<int32_t> perm(512);
    for (int32_t i = 0; i < 512; ++i) {
        perm[size_t(i)] = i;
    }
    double wall = 0;
    for (int32_t w = 0; w < 64 + 4; ++w) { // 4 windows first, untimed
        std::shuffle(perm.begin(), perm.end(), rng);
        cpu::job jobs[8];
        for (int32_t j = 0; j < 8; ++j) {
            const uint8_t * x = layer.data() + size_t(perm[size_t(j)]) * entry;
            jobs[j].gu[0] = x;
            jobs[j].dn[0] = x + gb;
            jobs[j].gu[1] = x + gb + db;
            jobs[j].dn[1] = x + 2 * gb + db;
            jobs[j].nt = 1;
            jobs[j].row[0] = 0;
            jobs[j].out[0] = out.data() + size_t(j) * D;
        }
        if (w == 4) {
            st = cpu::run_stats();
        }
        const auto t0 = std::chrono::steady_clock::now();
        pl.run(gf, df, act.data(), jobs, 8, st);
        if (w >= 4) {
            wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    printf("ex-cpu-test: throughput %s / %s (diagnostic): 64 windows of 8 experts at B = 1, %d workers and the "
           "host: %.2f GB/s of expert bytes (%.3f ms a window: gate/up %.3f, down activation %.3f, down %.3f; "
           "synthetic values)\n",
           fn::experts::fmt_name(gf), fn::experts::fmt_name(df), pl.workers(), double(st.bytes) / wall / 1e9,
           wall * 1e3 / 64, st.gu_ms / 64, st.act_ms / 64, st.dn_ms / 64);
}

// ---- ME3: the prompt loan's, the short read's, the refill's and the helpers' rules (no model, no device) ----------

namespace loan = fn::runtime::loan;

// a synthetic pool: 40 slots, the zone its last 20 (100 bytes each from offset 1,000), zone slot j holding an expert
// of layer j % 4; 4 layers of 8 experts, all resident, 50 bytes an expert; staging buffers of 400 bytes at the zone's
// start (layers 0 and 2) and 400 bytes on (layers 1 and 3); the lent region the zone's slots 8..19 (1,800..3,000)
loan::plan_input loan_pool() {
    loan::plan_input in;
    for (int32_t j = 0; j < 20; ++j) {
        in.zone.push_back({1000 + 100 * int64_t(j), 100, j % 4});
    }
    in.stage_off[0] = 1000;
    in.stage_off[1] = 1400;
    in.resident.assign(4, 8);
    in.dentry.assign(4, 50);
    in.n_ex = 8;
    in.n_slots = 40;
    in.lend_pct = 85;
    in.min_keep = 4;
    return in;
}

std::vector<int32_t> taken(const loan::plan_output & o) {
    std::vector<int32_t> t;
    for (size_t j = 0; j < o.take.size(); ++j) {
        if (o.take[j] != 0) {
            t.push_back(int32_t(j));
        }
    }
    return t;
}

// the plan's invariants: a slot is taken exactly when it meets a span or a staging prefix; the staging need covers
// its layers' experts not resident outside the loan (the fixed point); bytes and residents counted once a slot
bool loan_invariants(const loan::plan_input & in, const loan::plan_output & o) {
    std::vector<int32_t> res = in.resident;
    int64_t n = 0, bytes = 0, residents = 0;
    for (size_t j = 0; j < in.zone.size(); ++j) {
        const int64_t a = in.zone[j].off, b = a + in.zone[j].bytes;
        bool meets = false;
        for (const auto & x : in.spans) {
            meets = meets || (a < x.second && x.first < b);
        }
        for (int32_t k = 0; k < 2; ++k) {
            meets = meets || (o.stage_need[k] > 0 && a < in.stage_off[k] + o.stage_need[k] && in.stage_off[k] < b);
        }
        if (meets != (o.take[j] != 0)) {
            return false;
        }
        if (o.take[j] != 0) {
            ++n;
            bytes += in.zone[j].bytes;
            residents += in.zone[j].layer >= 0 ? 1 : 0;
            if (in.zone[j].layer >= 0 && !in.worst) {
                --res[size_t(in.zone[j].layer)];
            }
        }
    }
    for (size_t l = 0; l < in.dentry.size(); ++l) {
        if (int64_t(in.worst ? in.n_ex : in.n_ex - res[l]) * in.dentry[l] > o.stage_need[l & 1]) {
            return false;
        }
    }
    return n == o.n_taken && bytes == o.taken_bytes && residents == o.taken_residents;
}

bool me3_loan() {
    struct loan_case {
        const char * what;
        std::vector<std::pair<int64_t, int64_t>> spans;
        bool worst;
        std::vector<int32_t> want; // empty: invariants only
        int64_t need0, need1;
    };
    const loan_case cases[] = {
        {"small part (50 bytes in slot 10)", {{2000, 2050}}, false, {0, 10}, 50, 0},
        {"small part, its span twice (overlap)", {{2000, 2050}, {2000, 2050}}, false, {0, 10}, 50, 0},
        {"tail (the region's last 50 bytes)", {{2950, 3000}}, false, {0, 4, 19}, 100, 50},
        {"bulk (the whole region)", {{1800, 3000}}, false, {0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                             19},
         250, 250},
        {"bulk, overlapping spans", {{1800, 2500}, {2400, 3000}}, false, {}, 0, 0},
        {"worst (startup: every expert staged), no spans", {}, true, {0, 1, 2, 3, 4, 5, 6, 7}, 400, 400},
    };
    bool ok = true;
    std::vector<int32_t> bulk;
    for (const loan_case & c : cases) {
        loan::plan_input in = loan_pool();
        in.spans = c.spans;
        in.worst = c.worst;
        loan::plan_output o;
        std::string err;
        const bool planned = loan::plan(in, o, err);
        const std::vector<int32_t> t = planned ? taken(o) : std::vector<int32_t>();
        bool good = planned && loan_invariants(in, o);
        if (!c.want.empty()) {
            good = good && t == c.want && o.stage_need[0] == c.need0 && o.stage_need[1] == c.need1;
        } else {
            good = good && t == bulk; // the overlap: the same slots as the one span
        }
        if (c.spans.size() == 1 && c.spans[0].first == 1800) {
            bulk = t;
        }
        ok = ok && good;
        printf("ex-cpu-test: me3 loan %s: %lld slots (%lld bytes, %lld residents), staging %lld + %lld bytes: %s\n",
               c.what, (long long)o.n_taken, (long long)o.taken_bytes, (long long)o.taken_residents,
               (long long)o.stage_need[0], (long long)o.stage_need[1], good ? "as expected" : "DIFFER");
    }
    // the cap and min-keep refuse the bulk loan (18 of 40 slots)
    for (int32_t k = 0; k < 2; ++k) {
        loan::plan_input in = loan_pool();
        in.spans = {{1800, 3000}};
        (k == 0 ? in.lend_pct : in.min_keep) = k == 0 ? 40 : 23;
        loan::plan_output o;
        std::string err;
        const bool refused = !loan::plan(in, o, err);
        ok = ok && refused;
        printf("ex-cpu-test: me3 loan cap (%s): %s%s\n", k == 0 ? "lend-pct 40, 16 slots" : "min-keep 23, 22 left",
               refused ? "refused: " : "ACCEPTED", err.c_str());
    }
    return ok;
}

bool me3_short_read() {
    bool ok = true;
    struct gate {
        int64_t from, to;
        int32_t short_read;
        bool service, legacy, want;
    };
    // ME3 (the cleanup): a follow-up processes the pending token (the previous request's last, not yet read by the
    // target) and its appended IDs: 63 IDs are 64 rows (a short read), 64 IDs 65 rows (the bulk path)
    const int64_t done = 1774;
    const gate gates[] = {{0, 64, 64, true, false, true},
                          {100, 165, 64, true, false, false},
                          {100, 164, 64, true, false, true},
                          {0, 13, 0, true, false, false},
                          {0, 13, 64, false, false, false},
                          {0, 13, 64, true, true, false},
                          {done, done + 1 + 63, 64, true, false, true},
                          {done, done + 1 + 64, 64, true, false, false}};
    int32_t n_gate = 0;
    for (const gate & g : gates) {
        n_gate += fn::runtime::pf_short_read(g.from, g.to, g.short_read, g.service, g.legacy) == g.want ? 1 : 0;
    }
    ok = ok && n_gate == int32_t(sizeof(gates) / sizeof(gates[0]));
    // the rounds over a part: each 1..5 rows, the part covered, no round across a multiple of 8,192
    const std::pair<int64_t, int64_t> parts[] = {{0, 13}, {8190, 64}, {16380, 4}, {24571, 5}, {100, 1}};
    std::string walks;
    for (const auto & p : parts) {
        int64_t n = p.first, u = p.second;
        std::string w;
        while (u > 0) {
            const int32_t B = fn::runtime::pf_short_rows(n, u);
            const bool crosses = n / fn::runtime::pf_span != (n + B - 1) / fn::runtime::pf_span;
            ok = ok && B >= 1 && B <= 5 && B <= u && !crosses;
            w += (w.empty() ? "" : ",") + std::to_string(B);
            n += B;
            u -= B;
            if (B < 1) {
                break;
            }
        }
        ok = ok && n == p.first + p.second;
        walks += " " + std::to_string(p.first) + "+" + std::to_string(p.second) + ": " + w + ";";
    }
    printf("ex-cpu-test: me3 short read: the gate %d of %zu cases (threshold 64 at 64 and 65 tokens, off, no service, "
           "the base arm, the pending token and 63 or 64 IDs: 64 or 65 rows); rounds%s %s\n",
           n_gate, sizeof(gates) / sizeof(gates[0]), walks.c_str(), ok ? "as expected" : "DIFFER");
    return ok;
}

bool me3_refill_batches() {
    const std::vector<int32_t> layer = {0, 0, 1, 3, 3, 3, 7, 8};
    const struct {
        int32_t layers;
        std::vector<size_t> ends;
    } cases[] = {{1, {2, 3, 6, 7, 8}}, {2, {3, 6, 8}}, {48, {8}}, {0, {2, 3, 6, 7, 8}}};
    bool ok = true;
    std::string s;
    for (const auto & c : cases) {
        std::vector<size_t> ends;
        for (size_t a = 0; a < layer.size();) {
            const size_t b = loan::batch_end(layer.size(), a, c.layers, [&](size_t i) { return layer[i]; });
            if (b <= a) {
                ok = false;
                break;
            }
            ends.push_back(b);
            a = b;
        }
        ok = ok && ends == c.ends;
        s += " " + std::to_string(c.layers) + ":" + std::to_string(ends.size());
    }
    printf("ex-cpu-test: me3 refill batches (8 moves of layers 0,0,1,3,3,3,7,8; batches per layer bound%s): %s\n",
           s.c_str(), ok ? "as expected" : "DIFFER");
    return ok;
}

// ---- ME3 (the cleanup): the refill's state and the service loop's protocol, with fakes (no device) ----------------

namespace svc = fn::runtime::svc;
using ms = std::chrono::milliseconds;

// pred within limit (a hung protocol fails the check instead of the test)
bool within(const std::function<bool()> & pred, int32_t limit_ms) {
    const auto end = std::chrono::steady_clock::now() + ms(limit_ms);
    while (!pred()) {
        if (std::chrono::steady_clock::now() > end) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
}

bool me3_refill_state() {
    loan::refill_state st;
    bool ok = !st.active() && !st.ready() && !st.take(0, 1);
    const uint32_t g1 = st.start(5);
    ok = ok && st.active() && st.ready() && !st.take(1, 3) && !st.take(0, 6) && st.take(0, 3) && !st.ready();
    ok = ok && !st.trim(0) && !st.trim(4) && st.trim(2) && st.next() == 2;      // the yield: a shorter batch
    ok = ok && !st.landed(g1 + 1) && st.landed(g1) && st.landed_end() == 2;     // a stale landing changes nothing
    ok = ok && !st.landed(g1) && !st.published(3) && st.published(2) && st.end();
    ok = ok && st.start(4) != g1 && st.ready() && st.take(0, 2);                 // a new refill (the previous ended)
    const uint32_t g2 = st.gen();
    ok = ok && !st.end() && !st.landed(g1);                                      // in flight: no end, the old gen stale
    st.stop();
    ok = ok && st.state() == loan::refill_state::phase::stopping && !st.settled() && !st.ready();
    ok = ok && st.landed(g2) && st.settled() && st.published(2) && !st.complete() && st.outstanding() == 2;
    ok = ok && st.end() && !st.active() && st.outstanding() == 0;
    // a lost batch (its submission failed before its landing was enqueued): no further batch, settled
    st.start(3);
    ok = ok && st.take(0, 3);
    st.lost();
    ok = ok && st.failed() && !st.ready() && st.settled() && st.next() == 0 && st.end();
    // every move submitted and landed: settled while running, complete once published
    st.start(2);
    ok = ok && st.take(0, 2) && !st.settled() && st.landed(st.gen()) && st.settled() && st.published(2) &&
         st.complete() && st.end();
    printf("ex-cpu-test: me3 refill state (start, take, trim, stale and current landings, stop, lost, end) %s\n",
           ok ? "as expected" : "DIFFER");
    return ok;
}

// a window service: job seq's descriptor comes once the test lets it (desc_upto); serve fails at fail_seq; every call
// logged; an unblock before its descriptor counted (expected only when the descriptor never comes)
struct fake_work final : svc::work {
    std::mutex mu;
    std::vector<std::string> log;
    std::atomic<uint32_t> desc_upto{0};
    uint32_t fail_seq = 0;
    std::atomic<int32_t> early{0}, served{0};
    void note(const std::string & s) {
        std::lock_guard<std::mutex> lk(mu);
        log.push_back(s);
    }
    bool await(const svc::job & j, const std::atomic<bool> & quit, std::string & err) override {
        while (desc_upto.load() < j.seq) {
            if (j.stamp && quit) {
                err.clear();
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        return true;
    }
    bool serve(const svc::job & j, std::string & err) override {
        note("serve " + std::to_string(j.seq));
        if (j.seq == fail_seq) {
            err = "fake failure";
            return false;
        }
        ++served;
        return true;
    }
    void unblock(const svc::job & j) override {
        early += desc_upto.load() < j.seq ? 1 : 0;
        note("unblock " + std::to_string(j.seq));
    }
};

// a refill of moves in layers (batch_layers each), its moves "copied" one at a time while the test allows (allowed),
// landed at once by its submission or later by the test (the host callback's thread)
struct fake_refill final : svc::refill {
    svc::loop * lp = nullptr;
    fake_work * w = nullptr;
    std::mutex mu;
    std::condition_variable cv;
    loan::refill_state st;
    std::vector<int32_t> layer;
    std::atomic<size_t> allowed{size_t(1) << 40}, copied{0};
    bool land_now = true, pending = false;
    std::vector<std::pair<size_t, size_t>> batches;
    std::string failed_err;
    int32_t failed_calls = 0;
    bool ready() override {
        std::lock_guard<std::mutex> lk(mu);
        return st.ready();
    }
    void take(size_t & a, size_t & b) override {
        std::lock_guard<std::mutex> lk(mu);
        a = st.next();
        b = loan::batch_end(layer.size(), a, 1, [&](size_t i) { return layer[i]; });
        st.take(a, b);
    }
    bool submit(size_t a, size_t b, const std::function<bool()> & yield, std::string &) override {
        size_t end = a;
        while (end < b) {
            while (copied.load() >= allowed.load()) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            ++copied;
            ++end;
            if (end < b && yield()) {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            if (end < b) {
                st.trim(end);
            }
            batches.emplace_back(a, end);
            if (land_now) {
                st.landed(st.gen());
            } else {
                pending = true;
            }
        }
        w->note("batch " + std::to_string(a) + "-" + std::to_string(end));
        cv.notify_all();
        lp->notify();
        return true;
    }
    void failed(const std::string & e) override {
        {
            std::lock_guard<std::mutex> lk(mu);
            st.fail();
            failed_err = e;
            ++failed_calls;
        }
        cv.notify_all();
    }
    void begin(size_t n) { // the decode thread's start (a zone return)
        {
            std::lock_guard<std::mutex> lk(mu);
            st.start(n);
        }
        lp->notify();
    }
    void land() { // the callback: this mutex released before the loop's notify (the lock order)
        {
            std::lock_guard<std::mutex> lk(mu);
            st.landed(st.gen());
            pending = false;
        }
        cv.notify_all();
        lp->notify();
    }
    // the stopper (a new loan: stop; a drain: not): settled within the limit
    bool stop_wait(bool finish, int32_t limit_ms) {
        std::unique_lock<std::mutex> lk(mu);
        if (!finish) {
            st.stop();
        }
        return cv.wait_for(lk, ms(limit_ms), [&]() { return st.settled(); });
    }
};

size_t index_of(const std::vector<std::string> & log, const std::string & s) {
    return size_t(std::find(log.begin(), log.end(), s) - log.begin());
}

bool me3_service_loop() {
    bool all = true;
    std::string out;
    const auto arm = [&](const char * name, bool ok) {
        out += std::string(" ") + name + (ok ? " ok;" : " DIFFER;");
        all = all && ok;
    };
    {
        // miss-service progress: a refill batch in progress (its second move held) when a window is posted ends after
        // that move; the window is served before the refill's next batch
        svc::loop lp;
        fake_work w;
        fake_refill r;
        r.lp = &lp;
        r.w = &w;
        r.layer.assign(6, 0);
        lp.start(&w, &r, 4);
        r.allowed = 1;
        r.begin(6);
        bool ok = within([&]() { return r.copied.load() == 1; }, 2000);
        w.desc_upto = 1;
        lp.post(2, 1, 0);
        r.allowed = size_t(1) << 40;
        ok = ok && within([&]() { return w.served.load() == 1; }, 2000);
        ok = ok && within([&]() { return lp.batches() == 2; }, 2000);
        lp.wait_idle();
        lp.stop();
        ok = ok && r.batches.size() == 2 && r.batches[0] == std::make_pair(size_t(0), size_t(2)) &&
             r.batches[1] == std::make_pair(size_t(2), size_t(6));
        ok = ok && index_of(w.log, "batch 0-2") < index_of(w.log, "serve 1") &&
             index_of(w.log, "serve 1") < index_of(w.log, "batch 2-6");
        arm("yield to a queued window", ok);
    }
    {
        // the decode thread's wait for the service ends a batch after its current move too
        svc::loop lp;
        fake_work w;
        fake_refill r;
        r.lp = &lp;
        r.w = &w;
        r.layer.assign(6, 0);
        lp.start(&w, &r, 4);
        r.allowed = 1;
        r.begin(6);
        bool ok = within([&]() { return r.copied.load() == 1; }, 2000);
        std::atomic<bool> idle{false};
        std::thread t([&]() {
            lp.wait_idle();
            idle = true;
        });
        ok = ok && within([&]() { return lp.yield(); }, 2000);
        r.allowed = size_t(1) << 40;
        ok = ok && within([&]() { return idle.load(); }, 2000);
        t.join();
        ok = ok && !r.batches.empty() && r.batches[0] == std::make_pair(size_t(0), size_t(2));
        lp.wait_idle();
        lp.stop();
        arm("yield to wait_idle", ok);
    }
    {
        // a prior outstanding refill (its first batch in flight, not landed) at a new loan: no further batch, the
        // stop waits for the landing, the rest carried
        svc::loop lp;
        fake_work w;
        fake_refill r;
        r.lp = &lp;
        r.w = &w;
        r.layer = {0, 0, 1, 1, 2, 2};
        r.land_now = false;
        lp.start(&w, &r, 4);
        r.begin(6);
        bool ok = within([&]() { return lp.batches() == 1; }, 2000);
        lp.wait_idle();
        ok = ok && !r.stop_wait(false, 50); // in flight: the stop waits
        std::thread cb([&]() { r.land(); });
        ok = ok && r.stop_wait(false, 2000);
        cb.join();
        lp.wait_idle();
        size_t carried = 0;
        {
            std::lock_guard<std::mutex> lk(r.mu);
            ok = ok && r.st.published(r.st.landed_end());
            carried = r.st.size() - r.st.pub();
            ok = ok && r.st.end();
        }
        lp.stop();
        ok = ok && r.batches.size() == 1 && carried == 4;
        arm("outstanding refill stopped by a loan", ok);
    }
    {
        // replay and catch-up: a pass's stamp jobs served in layer order as their stamps come, then eager windows
        svc::loop lp;
        fake_work w;
        lp.start(&w, nullptr, 4);
        lp.post_pass(4, 1);
        bool ok = true;
        for (uint32_t s = 1; s <= 4; ++s) {
            w.desc_upto = s;
            ok = ok && within([&]() { return w.served.load() == int32_t(s); }, 2000);
        }
        w.desc_upto = 6;
        lp.post(0, 2, 1);
        lp.post(1, 2, 1);
        lp.wait_layer(1);
        lp.wait_idle();
        lp.stop();
        ok = ok && w.served.load() == 6 && !lp.failed() && w.early.load() == 0;
        arm("replay and catch-up", ok);
    }
    {
        // abandon: the pass's jobs whose stamps never come end at once (unblocked), the failure names the launch;
        // a later eager window still awaits its descriptor before its unblock
        svc::loop lp;
        fake_work w;
        fake_refill r;
        r.lp = &lp;
        r.w = &w;
        lp.start(&w, &r, 4);
        lp.post_pass(4, 1);
        w.desc_upto = 1;
        bool ok = within([&]() { return w.served.load() == 1; }, 2000);
        lp.abandon("cudaGraphLaunch: fake");
        lp.wait_idle();
        ok = ok && lp.failed() && lp.error().find("the replay's launch failed: cudaGraphLaunch: fake") !=
                                      std::string::npos;
        ok = ok && lp.unblocked() == 3 && w.early.load() == 3 && r.failed_calls == 1;
        lp.post(0, 1, 1);
        ok = ok && !within([&]() { return lp.unblocked() == 4; }, 50); // its descriptor first
        w.desc_upto = 5;
        lp.wait_idle();
        ok = ok && lp.unblocked() == 4 && w.early.load() == 3 && w.served.load() == 1;
        lp.stop();
        arm("abandon", ok);
    }
    {
        // error unwind: a failed window, then the later ones each unblocked only after its descriptor; the refill
        // told once
        svc::loop lp;
        fake_work w;
        fake_refill r;
        r.lp = &lp;
        r.w = &w;
        lp.start(&w, &r, 4);
        w.fail_seq = 1;
        lp.post(0, 1, 0);
        lp.post(1, 1, 0);
        lp.post(2, 1, 0);
        w.desc_upto = 1;
        bool ok = within([&]() { return lp.failed(); }, 2000);
        ok = ok && !within([&]() { return lp.unblocked() > 1; }, 50);
        w.desc_upto = 3;
        lp.wait_idle();
        ok = ok && lp.unblocked() == 3 && w.early.load() == 0 && lp.error() == "fake failure" &&
             r.failed_calls == 1 && r.failed_err == "fake failure";
        // a decode-side failure after its descriptor was enqueued (post_failed): awaited, then unblocked
        lp.post_failed(3, 1, 0, "enqueue failed");
        w.desc_upto = 4;
        lp.wait_idle();
        ok = ok && lp.unblocked() == 4 && w.early.load() == 0 && lp.error() == "fake failure";
        lp.stop();
        arm("error unwind", ok);
    }
    {
        // shutdown: queued replay jobs whose stamps never come unblocked, the thread joined; a second stop a no-op
        svc::loop lp;
        fake_work w;
        lp.start(&w, nullptr, 4);
        lp.post_pass(3, 1);
        lp.stop();
        lp.stop();
        const bool ok = lp.unblocked() == 3 && w.early.load() == 3 && w.served.load() == 0;
        arm("shutdown", ok);
    }
    printf("ex-cpu-test: me3 service loop (fakes):%s %s\n", out.c_str(), all ? "as expected" : "DIFFER");
    return all;
}

uint64_t mask_of(const std::vector<int32_t> & cpus) {
    uint64_t m = 0;
    for (const int32_t c : cpus) {
        m |= c >= 0 && c < 64 ? uint64_t(1) << c : 0;
    }
    return m;
}

bool me3_affinity() {
    bool ok = true;
    // the default mask's rule
    const struct {
        std::vector<int32_t> allowed, workers;
        int32_t service, decode;
        std::vector<int32_t> want;
    } rules[] = {{{0, 1, 2, 3, 4, 5, 6, 7}, {2, 4, 6}, 0, 1, {3, 5, 7}},
                 {{0, 1, 2}, {2}, 0, 1, {2}},
                 {{0, 1}, {}, 0, 1, {0, 1}},
                 {{}, {}, -1, -1, {}}};
    int32_t n_rule = 0;
    for (const auto & r : rules) {
        n_rule += fn::runtime::helper_mask(r.allowed, r.workers, r.service, r.decode) == r.want ? 1 : 0;
    }
    ok = ok && n_rule == 4;
    // the range checks: each given CPU within the allowed mask
    const std::vector<int32_t> al = {0, 1, 2, 3};
    int32_t n_ref = 0;
    for (int32_t k = 0; k < 4; ++k) {
        fn::runtime::expert_cache_options o;
        (k == 0 ? o.worker_cpus : k == 1 ? o.helper_cpus : o.worker_cpus) = k < 2 ? std::vector<int32_t>{1, 9} : al;
        o.service_cpu = k == 2 ? 9 : -1;
        o.decode_cpu = k == 3 ? 9 : -1;
        std::string err;
        n_ref += fn::runtime::cpus_allowed(al, o, err) ? 0 : 1;
    }
    fn::runtime::expert_cache_options good;
    good.worker_cpus = {1, 2};
    good.helper_cpus = {3};
    good.service_cpu = 0;
    std::string err;
    const bool accepts = fn::runtime::cpus_allowed(al, good, err);
    ok = ok && n_ref == 4 && accepts;
    // the live rule: the mask derived before a decode thread pins itself; helpers that thread starts on two later
    // "requests" take the mask (read back), one left as inherited shows the pinned CPU
    const std::vector<std::vector<int32_t>> cores = fn::runtime::physical_cores();
    std::vector<int32_t> allowed;
    for (const auto & c : cores) {
        allowed.insert(allowed.end(), c.begin(), c.end());
    }
    if (allowed.size() < 2) {
        printf("ex-cpu-test: me3 affinity: fewer than 2 allowed CPUs, the live check not run: FAIL\n");
        return false;
    }
    const int32_t service = allowed[0], decode = allowed[1];
    const std::vector<int32_t> helpers = fn::runtime::helper_mask(allowed, {}, service, decode);
    uint64_t got[3] = {0, 0, 0};
    size_t seen_pinned = 0;
    std::thread dec([&]() {
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(decode, &one);
        pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
        seen_pinned = fn::runtime::physical_cores().size(); // what a derivation after pinning would see
        for (int32_t q = 0; q < 3; ++q) {
            std::thread h([&, q]() { got[q] = fn::runtime::pf_helper_affinity(helpers, q < 2); });
            h.join();
        }
    });
    dec.join();
    const bool live = !helpers.empty() && got[0] == mask_of(helpers) && got[1] == mask_of(helpers) &&
                      got[2] == mask_of({decode}) && (mask_of(helpers) & mask_of({service, decode})) == 0;
    ok = ok && live;
    printf("ex-cpu-test: me3 affinity: the rule %d of 4; range refusals %d of 4, a legal set %s; live: %zu allowed "
           "CPUs, decode pinned to %d (a derivation after pinning sees %zu core), helpers on requests 2 and 3 read "
           "back 0x%llx and 0x%llx (want 0x%llx, neither the service's nor decode's CPU), one inheriting 0x%llx: %s\n",
           n_rule, n_ref, accepts ? "accepted" : "REFUSED", allowed.size(), decode, seen_pinned,
           (unsigned long long)got[0], (unsigned long long)got[1], (unsigned long long)mask_of(helpers),
           (unsigned long long)got[2], ok ? "as expected" : "DIFFER");
    return ok;
}

} // namespace

int fnfl_ex_cpu_test(const char * dir) {
    std::map<int32_t, fn::experts::synth> s;
    std::map<int32_t, bool> good;
    bool ok = true;
    const fmt gus[7] = {fmt::iq2_xxs, fmt::iq2_xs, fmt::iq2_s, fmt::iq3_xxs, fmt::iq3_s, fmt::q4_k, fmt::q5_k};
    const fmt dns[4] = {fmt::q2_0, fmt::iq4_nl, fmt::q5_1, fmt::q8_0};
    for (const fmt f : gus) {
        const bool r = cpu::gate_up_supported(f) && load(f, dir, s[int32_t(f)]) && round_trip_gu(s[int32_t(f)]);
        good[int32_t(f)] = r;
        ok = ok && r;
    }
    for (const fmt f : dns) {
        const bool r = cpu::down_supported(f) && load(f, dir, s[int32_t(f)]) && round_trip_dn(s[int32_t(f)]);
        good[int32_t(f)] = r;
        ok = ok && r;
    }
    if (!ok) {
        printf("ex-cpu-test: FAIL\n");
        return 1;
    }
    cpu::pool pl;
    std::string err;
    if (!pl.start(std::vector<int32_t>(5, -1), err)) {
        printf("ex-cpu-test: %s\n", err.c_str());
        return 1;
    }
    // every gate/up format once, every down format at least once
    const fmt pairs[7][2] = {{fmt::iq2_xxs, fmt::q2_0}, {fmt::iq2_xs, fmt::iq4_nl}, {fmt::iq2_s, fmt::q5_1},
                             {fmt::iq3_xxs, fmt::q8_0}, {fmt::iq3_s, fmt::q2_0},    {fmt::q4_k, fmt::q5_1},
                             {fmt::q5_k, fmt::q8_0}};
    for (const auto & p : pairs) {
        const bool r = fp64_pair(s[int32_t(p[0])], s[int32_t(p[1])], pl);
        good[int32_t(p[0])] = good[int32_t(p[0])] && r;
        good[int32_t(p[1])] = good[int32_t(p[1])] && r;
        ok = ok && r;
    }
    for (const fmt f : gus) {
        printf("ex-cpu-test: %s B=1..%d %s\n", fn::experts::fmt_name(f), R, good[int32_t(f)] ? "PASS" : "FAIL");
    }
    for (const fmt f : dns) {
        printf("ex-cpu-test: %s B=1..%d %s\n", fn::experts::fmt_name(f), R, good[int32_t(f)] ? "PASS" : "FAIL");
    }
    // ME4: the native views against the reconstruction, the same pairs (every format), and the extent rule
    for (const auto & p : pairs) {
        ok = me4_native(s[int32_t(p[0])], s[int32_t(p[1])], pl) && ok;
    }
    ok = me4_extents() && ok;
    ok = mx7_f16() && ok; // MX7: the prompt CPU share's conversions
    // ME2b: the host drained with the workers (its own scratch, index workers): the tasks it claimed over the runs
    const bool host = pl.host_tasks() > 0;
    printf("ex-cpu-test: host drain: %lld tasks claimed by the host over the runs above (%d workers, 3 x (workers + "
           "host) tasks a phase) %s\n",
           (long long)pl.host_tasks(), pl.workers(), host ? "PASS" : "FAIL");
    ok = ok && host;
    // ME3: the loan, short-read, refill-batch and helper-affinity arms
    ok = me3_loan() && ok;
    ok = me3_short_read() && ok;
    ok = me3_refill_batches() && ok;
    ok = me3_refill_state() && ok;
    ok = me3_service_loop() && ok;
    ok = me3_affinity() && ok;
    // ME4: the prompt planner (chunks, paths, ring and loan, groups, the byte ring, the switch)
    ok = fn::runtime::prompt_plan_test() && ok;
    ok = fn::runtime::slots_cpu_test() && ok; // MU1
    // ME5: the row-aware miss split
    ok = fn::runtime::miss_split_test() && ok;
    // ME9: the admission records
    ok = fn::runtime::admission_test() && ok;
    // the deployed classes' throughput, the pool pinned as the cache pins it (CPUs 1..5: each core's first)
    pl.stop();
    const std::vector<std::vector<int32_t>> cores = fn::runtime::physical_cores();
    std::vector<int32_t> cpus;
    for (size_t c = 1; c < cores.size() && cpus.size() < 5; ++c) {
        cpus.push_back(cores[c][0]);
    }
    if (!pl.start(cpus, err)) {
        printf("ex-cpu-test: %s\n", err.c_str());
        return 1;
    }
    std::string masks;
    for (int32_t i = 0; i < pl.workers(); ++i) {
        char m[40];
        snprintf(m, sizeof(m), "%s%d:0x%llx", i == 0 ? "" : " ", cpus[size_t(i)], (unsigned long long)pl.affinity(i));
        masks += m;
    }
    printf("ex-cpu-test: throughput pool: %d workers, CPU:mask %s\n", pl.workers(), masks.c_str());
    const fmt classes[10][2] = {{fmt::iq2_xs, fmt::iq4_nl},  {fmt::iq2_xxs, fmt::q2_0},  {fmt::iq2_xs, fmt::q2_0},
                                {fmt::iq3_s, fmt::q2_0},     {fmt::iq2_s, fmt::q2_0},    {fmt::iq3_s, fmt::iq4_nl},
                                {fmt::iq2_s, fmt::iq4_nl},   {fmt::iq2_xxs, fmt::iq4_nl}, {fmt::iq3_xxs, fmt::iq4_nl},
                                {fmt::iq3_xxs, fmt::q2_0}};
    for (const auto & c : classes) {
        throughput(c[0], c[1], s[int32_t(c[0])], s[int32_t(c[1])], pl);
    }
    printf("ex-cpu-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
