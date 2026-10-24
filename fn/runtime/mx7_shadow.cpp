// MX7's shadow comparison (mx7_shadow.h)

#include "fn/runtime/mx7_shadow.h"

#include "fn/quant/decode.h"
#include "fn/quant/formats.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace fn::runtime {

namespace {

constexpr int32_t D = fn::experts::cpu::d_model;
constexpr int32_t NFF = fn::experts::cpu::n_ff;
constexpr int32_t worst_per_layer = 10;
constexpr int32_t sample_every = 16;

fn::quant::wfmt wfmt_of(fn::experts::fmt f) {
    using fn::quant::wfmt;
    static const wfmt q[11] = {wfmt::iq2_xxs, wfmt::iq2_xs, wfmt::iq2_s, wfmt::iq3_xxs, wfmt::iq3_s, wfmt::q2_0,
                               wfmt::iq4_nl,  wfmt::q8_0,   wfmt::q4_k,  wfmt::q5_k,    wfmt::q5_1}; // fmt's order
    return q[static_cast<int32_t>(f)];
}

template <typename A, typename B> double rel_l2(const A * a, const B * b, int32_t n) {
    double num = 0, den = 0;
    for (int32_t i = 0; i < n; ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(b[i]) * double(b[i]);
    }
    return den > 0 ? std::sqrt(num / den) : (num > 0 ? INFINITY : 0.0);
}

// one expert's GGUF values: gate and up [640][2560], down [2560][640]
struct expert_values {
    int32_t e = -1;
    std::vector<float> gate, up, down;
};

bool decode_expert(const fn::experts::cpu::native_layer & v, fn::experts::fmt gu, fn::experts::fmt dn, int32_t e,
                   expert_values & x) {
    if (x.e == e) {
        return true;
    }
    x.gate.resize(size_t(NFF) * D);
    x.up.resize(size_t(NFF) * D);
    x.down.resize(size_t(D) * NFF);
    const fn::quant::wfmt g = wfmt_of(gu), d = wfmt_of(dn);
    for (int32_t n = 0; n < NFF; ++n) {
        const int64_t at = int64_t(e) * v.gu_expert + int64_t(n) * v.gu_row;
        if (!fn::quant::decode_row(g, v.gate + at, x.gate.data() + int64_t(n) * D, D) ||
            !fn::quant::decode_row(g, v.up + at, x.up.data() + int64_t(n) * D, D)) {
            return false;
        }
    }
    for (int32_t j = 0; j < D; ++j) {
        const int64_t at = int64_t(e) * v.dn_expert + int64_t(j) * v.dn_row;
        if (!fn::quant::decode_row(d, v.down + at, x.down.data() + int64_t(j) * NFF, NFF)) {
            return false;
        }
    }
    x.e = e;
    return true;
}

// the FP64 product of one expert against one activation row x [2560]: gate, up, SwiGLU, down
void reference(const expert_values & v, const double * x, std::vector<double> & out) {
    std::vector<double> h(NFF);
    for (int32_t n = 0; n < NFF; ++n) {
        const float * gr = v.gate.data() + int64_t(n) * D;
        const float * ur = v.up.data() + int64_t(n) * D;
        double g = 0, u = 0;
        for (int32_t i = 0; i < D; ++i) {
            g += double(gr[i]) * x[i];
            u += double(ur[i]) * x[i];
        }
        h[size_t(n)] = g / (1.0 + std::exp(-g)) * u;
    }
    out.assign(D, 0.0);
    for (int32_t j = 0; j < D; ++j) {
        const float * dr = v.down.data() + int64_t(j) * NFF;
        double s = 0;
        for (int32_t n = 0; n < NFF; ++n) {
            s += double(dr[n]) * h[size_t(n)];
        }
        out[size_t(j)] = s;
    }
}

} // namespace

bool mx7_shadow_layer(const mx7_shadow_in & in, std::vector<mx7_shadow_pair> & out, std::string & err) {
    if (in.pairs < 1 || in.rows < 1) {
        err = "mx7 shadow: no pairs";
        return false;
    }
    const int64_t np = int64_t(in.rows) * 10;
    std::vector<int32_t> pos[2] = {std::vector<int32_t>(size_t(np)), std::vector<int32_t>(size_t(np))};
    std::vector<uint16_t> part[2] = {std::vector<uint16_t>(size_t(np) * D), std::vector<uint16_t>(size_t(np) * D)};
    std::vector<float> wts(static_cast<size_t>(np));
    std::vector<uint16_t> x16(size_t(in.rows) * D);
    const cudaStream_t s = static_cast<cudaStream_t>(in.stream);
    bool ok = cudaStreamSynchronize(s) == cudaSuccess;
    for (int32_t k = 0; ok && k < 2; ++k) {
        ok = cudaMemcpy(pos[k].data(), in.pos_d[k], size_t(np) * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(part[k].data(), in.part16_d[k], size_t(np) * D * 2, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    ok = ok && cudaMemcpy(wts.data(), in.wts_d, size_t(np) * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(x16.data(), in.x16_d, x16.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) {
        (void)cudaGetLastError();
        err = "mx7 shadow: the GPU partials' copy";
        return false;
    }
    // every CPU pair against the GPU's
    const size_t first = out.size();
    std::vector<float> gpu(size_t(in.pairs) * D);
    int32_t in_expert = 0; // the pair's index among its expert's pairs (the CPU's pairs are by expert, then pair)
    for (int32_t k = 0; k < in.pairs; ++k) {
        const int32_t p = in.pair_of[k];
        in_expert = k > 0 && in.ids[in.pair_of[k - 1]] == in.ids[p] ? in_expert + 1 : 0;
        float * g = gpu.data() + int64_t(k) * D;
        const uint16_t * a = part[0].data() + int64_t(pos[0][size_t(p)]) * D;
        const uint16_t * b = part[1].data() + int64_t(pos[1][size_t(p)]) * D;
        for (int32_t j = 0; j < D; ++j) {
            g[j] = fn::quant::f16_to_f32(a[j]) + fn::quant::f16_to_f32(b[j]);
        }
        mx7_shadow_pair r;
        r.layer = in.layer;
        r.expert = in.ids[p];
        r.row = p / 10;
        r.slot = p % 10;
        r.job_row = in_expert % fn::experts::cpu::job_max_rows;
        r.gu = in.gu;
        r.dn = in.dn;
        r.w = wts[size_t(p)];
        r.e_cg = float(rel_l2(in.cpu + int64_t(k) * D, g, D));
        out.push_back(r);
    }
    if (in.native == nullptr || !in.native->valid()) {
        return true; // no native views (run-dry's zero backing): no reference (e_cpu stays -1; the report counts them)
    }
    // the referenced pairs: one in 16, and the layer's 10 largest CPU-GPU errors; by expert, so each decodes once
    std::vector<int32_t> by_err(size_t(in.pairs));
    std::iota(by_err.begin(), by_err.end(), 0);
    std::sort(by_err.begin(), by_err.end(),
              [&](int32_t a, int32_t b) { return out[first + size_t(a)].e_cg > out[first + size_t(b)].e_cg; });
    std::vector<uint8_t> want(size_t(in.pairs), 0);
    for (int32_t k = 0; k < in.pairs; k += sample_every) {
        want[size_t(k)] = 1;
    }
    for (int32_t i = 0; i < std::min(worst_per_layer, in.pairs); ++i) {
        want[size_t(by_err[size_t(i)])] = 1;
    }
    std::vector<int32_t> refd;
    for (int32_t k = 0; k < in.pairs; ++k) {
        if (want[size_t(k)] != 0) {
            refd.push_back(k);
        }
    }
    std::stable_sort(refd.begin(), refd.end(), [&](int32_t a, int32_t b) {
        return out[first + size_t(a)].expert < out[first + size_t(b)].expert;
    });
    expert_values ev;
    std::vector<double> x(D), xq(D), ref, refq;
    for (const int32_t k : refd) {
        mx7_shadow_pair & r = out[first + size_t(k)];
        if (!decode_expert(*in.native, in.gu, in.dn, r.expert, ev)) {
            err = "mx7 shadow: an expert's GGUF rows (decode)";
            return false;
        }
        const int32_t c = in.compact[r.row];
        if (c < 0) {
            err = "mx7 shadow: a CPU pair's row has no compact row";
            return false;
        }
        for (int32_t i = 0; i < D; ++i) {
            x[size_t(i)] = double(fn::quant::f16_to_f32(x16[size_t(r.row) * D + size_t(i)]));
            const fn::experts::cpu::block_q8_k & q = in.act[int64_t(c) * fn::experts::cpu::n_q8k + i / 256];
            xq[size_t(i)] = double(q.d) * q.qs[i % 256];
        }
        reference(ev, x.data(), ref);
        reference(ev, xq.data(), refq);
        r.e_cpu = float(rel_l2(in.cpu + int64_t(k) * D, ref.data(), D));
        r.e_gpu = float(rel_l2(gpu.data() + int64_t(k) * D, ref.data(), D));
        r.e_q1 = float(rel_l2(refq.data(), ref.data(), D));
    }
    return true;
}

} // namespace fn::runtime
