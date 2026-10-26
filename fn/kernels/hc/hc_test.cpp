#include "fn/kernels/hc/hc_test.h"

#include "fn/kernels/hc/hc.h"
#include "fn/model/shapes.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/formats.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

// the pass bound for every output: max |gpu - ref| over max |ref|, against the FP64 reference
constexpr double rel_bound = 1e-6;

constexpr int32_t D = fn::n_embd;           // 2,560 features a stream
constexpr int32_t S = fn::n_hc;             // 4 streams
constexpr int32_t K = S * D;                // 10,240: the GGUF row length of down and inject
constexpr int32_t F = fn::hc::n_own;        // 1,280 features a card owns
constexpr int32_t L = fn::hc::n_down;       // 320
constexpr int32_t J = fn::hc::n_inject;     // 4
constexpr int32_t M = fn::hc::n_mom;        // 1,300
constexpr int32_t RD = fn::hc::n_rows_down; // 324

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "hc-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

uint16_t to_bf16(float f) {
    return fn::quant::f32_to_bf16(f);
}

double from_bf16(uint16_t h) {
    return fn::quant::bf16_to_f32(h);
}

// owns one device allocation of n elements of T
template <typename T> struct dev_buf {
    T * p = nullptr;
    size_t n = 0;

    explicit dev_buf(size_t count) : n(count) {
        if (!cuda_ok(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc")) {
            p = nullptr;
        }
    }

    ~dev_buf() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dev_buf(const dev_buf &) = delete;
    dev_buf & operator=(const dev_buf &) = delete;

    bool put(const std::vector<T> & h, cudaStream_t st) const {
        return h.size() == n && cuda_ok(cudaMemcpyAsync(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice, st), "put");
    }

    bool get(std::vector<T> & h, cudaStream_t st) const {
        h.assign(n, T{});
        return cuda_ok(cudaMemcpyAsync(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost, st), "get");
    }

    // NaN bytes for F32, so an element the kernel never writes fails the comparison
    bool poison(cudaStream_t st) const { return cuda_ok(cudaMemsetAsync(p, 0xff, n * sizeof(T), st), "poison"); }
};

// One HC sublayer's weights in the GGUF layouts (ne0 contiguous), and their two card repacks (hc.h)
struct weights {
    std::vector<uint16_t> down, inject, up; // [320][10240], [4][10240], [10240][320]
    std::vector<float> gamma;               // [10240]
    std::vector<uint16_t> w_down[2], w_up[2];
    std::vector<float> gamma_card[2];
    // the weights' exact values, which the reference reads: the BF16 widened, or (DR1) the Q8_0 decoded
    std::vector<float> down_f, inject_f, up_f;
    // DR1: the Q8_0 card repacks (hc.h), for make_weights_q8's weights
    std::vector<uint8_t> q_down[2], q_up[2];
};

// the BF16 card repacks of down, inject and up, and gamma's
void repack(weights & w) {
    for (int32_t r = 0; r < 2; ++r) {
        w.w_down[r].resize(fn::hc::w_down_elems);
        w.w_up[r].resize(fn::hc::w_up_elems);
        w.gamma_card[r].resize(static_cast<size_t>(S) * F);
        for (int32_t s = 0; s < S; ++s) {
            for (int32_t j = 0; j < RD; ++j) {
                const uint16_t * src =
                    j < L ? &w.down[static_cast<size_t>(j) * K] : &w.inject[static_cast<size_t>(j - L) * K];
                for (int32_t i = 0; i < F; ++i) {
                    w.w_down[r][(static_cast<size_t>(s) * RD + j) * F + i] = src[s * D + r * F + i];
                }
            }
            for (int32_t i = 0; i < F; ++i) {
                w.gamma_card[r][static_cast<size_t>(s) * F + i] = w.gamma[s * D + r * F + i];
            }
        }
        for (int32_t c = 0; c < F; ++c) {
            for (int32_t s = 0; s < S; ++s) {
                const uint16_t * src = &w.up[static_cast<size_t>(s * D + r * F + c) * L];
                std::copy(src, src + L, &w.w_up[r][(static_cast<size_t>(c) * S + s) * L]);
            }
        }
    }
}

void make_weights(weights & w) {
    std::mt19937 rng(20261010);
    std::uniform_real_distribution<float> d_down(-0.07f, 0.07f), d_up(-0.25f, 0.25f), d_gamma(0.5f, 1.5f);
    w.down.resize(static_cast<size_t>(L) * K);
    w.inject.resize(static_cast<size_t>(J) * K);
    w.up.resize(static_cast<size_t>(K) * L);
    w.gamma.resize(K);
    for (auto & v : w.down) {
        v = to_bf16(d_down(rng));
    }
    for (auto & v : w.inject) {
        v = to_bf16(d_down(rng));
    }
    for (auto & v : w.up) {
        v = to_bf16(d_up(rng));
    }
    for (auto & v : w.gamma) {
        v = d_gamma(rng);
    }
    const auto widen = [](const std::vector<uint16_t> & h, std::vector<float> & f) {
        f.resize(h.size());
        for (size_t i = 0; i < h.size(); ++i) {
            f[i] = static_cast<float>(from_bf16(h[i]));
        }
    };
    widen(w.down, w.down_f);
    widen(w.inject, w.inject_f);
    widen(w.up, w.up_f);
    repack(w);
}

// DR1: Q8_0 GGUF rows (blocks of 32 along ne0: an FP16 d and 32 int8 q, q in -127..127 as the quantizer gives, and
// -128 in a few places to cover the byte's full range), their exact F32 values, and the Q8_0 card repacks (hc.h).
// pow2: every d a power of two, so every weight is also exactly a BF16 and the BF16 fields hold the same weights (the
// Q8_0 kernels must then match the BF16 kernels bitwise); otherwise d is a general FP16 value and the BF16 fields are
// left empty.
struct q8_rows {
    std::vector<int8_t> q;   // [rows][n]
    std::vector<uint16_t> d; // [rows][n / 32]
};

void make_q8(std::mt19937 & rng, int32_t rows, int32_t n, float amax, bool pow2, q8_rows & out,
             std::vector<float> & f) {
    std::uniform_int_distribution<int32_t> dq(-127, 127), dexp(0, 2), dmin(0, 4095);
    std::uniform_real_distribution<float> dscale(0.5f, 1.0f);
    out.q.resize(static_cast<size_t>(rows) * n);
    out.d.resize(static_cast<size_t>(rows) * n / 32);
    f.resize(out.q.size());
    const float base = amax / 127.0f;
    for (size_t b = 0; b < out.d.size(); ++b) {
        // pow2: 2^floor(log2 base) / 2^(0..2); general: base times (0.5 .. 1), rounded to FP16
        const float dv = pow2 ? std::ldexp(1.0f, static_cast<int32_t>(std::floor(std::log2(base))) - dexp(rng))
                              : fn::quant::f16_to_f32(fn::quant::f32_to_f16(base * dscale(rng)));
        out.d[b] = fn::quant::f32_to_f16(dv);
        for (int32_t i = 0; i < 32; ++i) {
            const size_t at = b * 32 + i;
            out.q[at] = static_cast<int8_t>(dmin(rng) == 0 ? -128 : dq(rng));
            f[at] = static_cast<float>(out.q[at]) * fn::quant::f16_to_f32(out.d[b]); // exact
        }
    }
}

void make_weights_q8(weights & w, bool pow2) {
    std::mt19937 rng(pow2 ? 20261012 : 20261013);
    std::uniform_real_distribution<float> d_gamma(0.5f, 1.5f);
    q8_rows down, inject, up;
    make_q8(rng, L, K, 0.07f, pow2, down, w.down_f);
    make_q8(rng, J, K, 0.07f, pow2, inject, w.inject_f);
    make_q8(rng, K, L, 0.25f, pow2, up, w.up_f);
    w.gamma.resize(K);
    for (auto & v : w.gamma) {
        v = d_gamma(rng);
    }
    if (pow2) {
        const auto narrow = [](const std::vector<float> & f, std::vector<uint16_t> & h) {
            h.resize(f.size());
            for (size_t i = 0; i < f.size(); ++i) {
                h[i] = to_bf16(f[i]);
            }
        };
        narrow(w.down_f, w.down);
        narrow(w.inject_f, w.inject);
        narrow(w.up_f, w.up);
        repack(w);
    } else {
        w.down.clear();
        w.inject.clear();
        w.up.clear();
        for (int32_t r = 0; r < 2; ++r) {
            w.gamma_card[r].resize(static_cast<size_t>(S) * F);
            for (int32_t s = 0; s < S; ++s) {
                for (int32_t i = 0; i < F; ++i) {
                    w.gamma_card[r][static_cast<size_t>(s) * F + i] = w.gamma[s * D + r * F + i];
                }
            }
        }
    }
    constexpr int32_t FB = F / 32, LB = L / 32;
    for (int32_t r = 0; r < 2; ++r) {
        // w_down: qs[s][j][i], then d[s][j][i / 32]
        std::vector<uint8_t> & qd = w.q_down[r];
        qd.assign(static_cast<size_t>(fn::hc::w_down_q8_bytes), 0);
        uint8_t * dd = qd.data() + static_cast<size_t>(S) * RD * F;
        for (int32_t s = 0; s < S; ++s) {
            for (int32_t j = 0; j < RD; ++j) {
                const q8_rows & src = j < L ? down : inject;
                const int32_t row = j < L ? j : j - L;
                const size_t col0 = static_cast<size_t>(s) * D + int64_t(r) * F;
                for (int32_t i = 0; i < F; ++i) {
                    qd[(static_cast<size_t>(s) * RD + j) * F + i] =
                        static_cast<uint8_t>(src.q[static_cast<size_t>(row) * K + col0 + i]);
                }
                for (int32_t ib = 0; ib < FB; ++ib) {
                    memcpy(dd + ((static_cast<size_t>(s) * RD + j) * FB + ib) * 2,
                           &src.d[static_cast<size_t>(row) * (K / 32) + col0 / 32 + ib], 2);
                }
            }
        }
        // w_up: qs[c][s][j], then d[c][s][j / 32] from GGUF row s*2560 + r*1280 + c
        std::vector<uint8_t> & qu = w.q_up[r];
        qu.assign(static_cast<size_t>(fn::hc::w_up_q8_bytes), 0);
        uint8_t * du = qu.data() + static_cast<size_t>(F) * S * L;
        for (int32_t c = 0; c < F; ++c) {
            for (int32_t s = 0; s < S; ++s) {
                const size_t row = static_cast<size_t>(s) * D + int64_t(r) * F + c;
                const size_t at = static_cast<size_t>(c) * S + s;
                memcpy(qu.data() + at * L, &up.q[row * L], L);
                memcpy(du + at * LB * 2, &up.d[row * LB], LB * 2);
            }
        }
    }
}

// full-width activations for B rows
struct inputs {
    int32_t rows = 0;
    std::vector<float> hres, hx, hpost; // [B][4][2560], [B][2560], [B][4]
};

// ME12's special: signed zeros, cancellation-sensitive sums and NaNs, compared bitwise only (between two GPU orders)
enum class kind { plain, stress, zero_hx, special };

const char * kind_name(kind k) {
    return k == kind::plain ? "plain" : k == kind::stress ? "stress" : k == kind::zero_hx ? "zero-hx" : "special";
}

void make_inputs(inputs & in, int32_t rows, kind k) {
    std::mt19937 rng(static_cast<uint32_t>(1000 * rows + static_cast<int>(k)));
    std::uniform_real_distribution<float> d_act(-2.0f, 2.0f), d_post(0.5f, 1.5f);
    in.rows = rows;
    in.hres.resize(static_cast<size_t>(rows) * S * D);
    in.hx.resize(static_cast<size_t>(rows) * D);
    in.hpost.resize(static_cast<size_t>(rows) * S);
    for (auto & v : in.hres) {
        v = d_act(rng);
    }
    for (auto & v : in.hx) {
        v = k == kind::zero_hx ? 0.0f : d_act(rng);
    }
    for (auto & v : in.hpost) {
        v = d_post(rng);
    }
    if (k == kind::stress) {
        // stream 1 scaled by 1e3 and stream 2 by 1e-3: both hres and hpost, so the whole stream x scales
        for (int32_t b = 0; b < rows; ++b) {
            for (int32_t s = 1; s <= 2; ++s) {
                const float f = s == 1 ? 1e3f : 1e-3f;
                for (int32_t d = 0; d < D; ++d) {
                    in.hres[(static_cast<size_t>(b) * S + s) * D + d] *= f;
                }
                in.hpost[static_cast<size_t>(b) * S + s] *= f;
            }
        }
    }
    if (k == kind::special) {
        // every 7th hres and 11th hx feature -0, stream 3's hres +-1e7 about a small residue (sums that cancel),
        // row 0's stream 2 hpost -0, and NaNs in the last row (hres stream 1, hx)
        for (int32_t b = 0; b < rows; ++b) {
            for (int32_t d = 0; d < D; ++d) {
                for (int32_t s = 0; s < S; ++s) {
                    float & v = in.hres[(static_cast<size_t>(b) * S + s) * D + d];
                    v = s == 3 ? ((d & 1) != 0 ? 1e7f : -1e7f) + 1e-3f * v : d % 7 == 0 ? -0.0f : v;
                }
                if (d % 11 == 0) {
                    in.hx[static_cast<size_t>(b) * D + d] = -0.0f;
                }
            }
        }
        in.hpost[2] = -0.0f;
        const size_t last = static_cast<size_t>(rows) - 1;
        in.hres[(last * S + 1) * D + 5] = std::numeric_limits<float>::quiet_NaN();
        in.hx[last * D + 9] = std::numeric_limits<float>::quiet_NaN();
    }
}

// the FP64 reference of hc.h's equations, per card where the kernels are per card
struct reference {
    std::vector<double> x;      // [B][4][2560]
    std::vector<double> mom[2]; // [B][1300] per card
    std::vector<double> inject; // [B][4]
    std::vector<double> mixed;  // [B][2560]
};

void make_reference(const weights & w, const inputs & in, reference & ref) {
    const int32_t B = in.rows;
    ref.x.assign(static_cast<size_t>(B) * S * D, 0.0);
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t s = 0; s < S; ++s) {
            for (int32_t d = 0; d < D; ++d) {
                const size_t at = (static_cast<size_t>(b) * S + s) * D + d;
                const double post = in.hpost[static_cast<size_t>(b) * S + s];
                ref.x[at] = static_cast<double>(in.hres[at]) + post * in.hx[static_cast<size_t>(b) * D + d];
            }
        }
    }
    std::vector<double> xg(static_cast<size_t>(B) * K);
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t e = 0; e < K; ++e) {
            const size_t at = static_cast<size_t>(b) * K + e;
            xg[at] = static_cast<double>(w.gamma[e]) * ref.x[at];
        }
    }
    for (int32_t r = 0; r < 2; ++r) {
        ref.mom[r].assign(static_cast<size_t>(B) * M, 0.0);
        for (int32_t b = 0; b < B; ++b) {
            double * m = &ref.mom[r][static_cast<size_t>(b) * M];
            for (int32_t s = 0; s < S; ++s) {
                const double * xs = &ref.x[(static_cast<size_t>(b) * S + s) * D + int64_t(r) * F];
                const double * gs = &xg[static_cast<size_t>(b) * K + int64_t(s) * D + int64_t(r) * F];
                double v = 0.0;
                for (int32_t i = 0; i < F; ++i) {
                    v += xs[i] * xs[i];
                }
                m[s] = v;
                for (int32_t j = 0; j < RD; ++j) {
                    const float * wr =
                        j < L ? &w.down_f[static_cast<size_t>(j) * K] : &w.inject_f[static_cast<size_t>(j - L) * K];
                    double u = 0.0;
                    for (int32_t i = 0; i < F; ++i) {
                        u += static_cast<double>(wr[s * D + r * F + i]) * gs[i];
                    }
                    m[j < L ? 4 + L * s + j : 4 + L * S + J * s + (j - L)] = u;
                }
            }
        }
    }
    ref.inject.assign(static_cast<size_t>(B) * J, 0.0);
    ref.mixed.assign(static_cast<size_t>(B) * D, 0.0);
    for (int32_t b = 0; b < B; ++b) {
        const double * m0 = &ref.mom[0][static_cast<size_t>(b) * M];
        const double * m1 = &ref.mom[1][static_cast<size_t>(b) * M];
        double rms[S];
        for (int32_t s = 0; s < S; ++s) {
            rms[s] = std::sqrt((m0[s] + m1[s]) / D + static_cast<double>(fn::rms_eps));
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
        for (int32_t k = 0; k < J; ++k) {
            double z = 0.0;
            for (int32_t s = 0; s < S; ++s) {
                z += (m0[4 + L * S + J * s + k] + m1[4 + L * S + J * s + k]) / rms[s];
            }
            ref.inject[static_cast<size_t>(b) * J + k] = 2.0 / (1.0 + std::exp(-0.25 * z));
        }
        for (int32_t d = 0; d < D; ++d) {
            double mix = 0.0;
            for (int32_t s = 0; s < S; ++s) {
                const float * wr = &w.up_f[static_cast<size_t>(s * D + d) * L];
                double z = 0.0;
                for (int32_t j = 0; j < L; ++j) {
                    z += static_cast<double>(wr[j]) * low[j];
                }
                const double gate = 1.0 / (1.0 + std::exp(-z));
                const double xv = ref.x[(static_cast<size_t>(b) * S + s) * D + d];
                mix += static_cast<double>(w.gamma[s * D + d]) * xv / rms[s] * gate;
            }
            ref.mixed[static_cast<size_t>(b) * D + d] = 0.25 * mix;
        }
    }
}

// the outputs of both cards' kernels, run on one device: card 0's data, then card 1's
struct gpu_out {
    std::vector<float> x[2], mom[2], inject[2], mixed[2];
};

struct device_weights {
    static constexpr size_t n_wd = fn::hc::w_down_elems, n_wu = fn::hc::w_up_elems;
    dev_buf<uint16_t> w_down0{n_wd}, w_down1{n_wd};
    dev_buf<uint16_t> w_up0{n_wu}, w_up1{n_wu};
    dev_buf<float> gamma0{static_cast<size_t>(S) * F}, gamma1{static_cast<size_t>(S) * F};
};

// a kernel pair under test: the shipped down() and up(), or the kernels at a stream depth (HC1c)
struct candidate {
    const char * name;
    bool shipped;
    int32_t depth;
};

const candidate candidates[] = {
    {"shipped", true, 0}, {"d1", false, 1}, {"d2", false, 2}, {"d3", false, 3}, {"d6", false, 6},
};

bool same_bits(const gpu_out & a, const gpu_out & b);

// Runs both cards' kernels once and reads the outputs; then, if replays > 0, captures the same sequence (outputs
// poisoned first) in a CUDA graph and replays it that many times, counting the replays whose outputs differ in any
// bit from the first run's (HC1b's check).
bool run_gpu(const device_weights & dw, const inputs & in, gpu_out & out, cudaStream_t st, const candidate & cand,
             int32_t replays = 0, int32_t * replay_diffs = nullptr) {
    const int32_t B = in.rows;
    const size_t nx = static_cast<size_t>(B) * S * F;
    dev_buf<float> hres0(nx), hres1(nx), hx0(static_cast<size_t>(B) * F), hx1(static_cast<size_t>(B) * F);
    dev_buf<float> hpost(static_cast<size_t>(B) * S), x0(nx), x1(nx);
    dev_buf<float> mom0(static_cast<size_t>(B) * M), mom1(static_cast<size_t>(B) * M);
    dev_buf<float> inj0(static_cast<size_t>(B) * J), inj1(static_cast<size_t>(B) * J);
    dev_buf<float> mix0(static_cast<size_t>(B) * F), mix1(static_cast<size_t>(B) * F);
    const dev_buf<float> * hres[2] = {&hres0, &hres1};
    const dev_buf<float> * hx[2] = {&hx0, &hx1};
    const dev_buf<float> * x[2] = {&x0, &x1};
    const dev_buf<float> * mom[2] = {&mom0, &mom1};
    const dev_buf<float> * inj[2] = {&inj0, &inj1};
    const dev_buf<float> * mix[2] = {&mix0, &mix1};
    const dev_buf<uint16_t> * wd[2] = {&dw.w_down0, &dw.w_down1};
    const dev_buf<uint16_t> * wu[2] = {&dw.w_up0, &dw.w_up1};
    const dev_buf<float> * gm[2] = {&dw.gamma0, &dw.gamma1};

    for (int32_t r = 0; r < 2; ++r) {
        std::vector<float> h(nx), v(static_cast<size_t>(B) * F);
        for (int32_t b = 0; b < B; ++b) {
            for (int32_t s = 0; s < S; ++s) {
                for (int32_t i = 0; i < F; ++i) {
                    const size_t bs = static_cast<size_t>(b) * S + s;
                    h[bs * F + i] = in.hres[bs * D + int64_t(r) * F + i];
                }
            }
            for (int32_t i = 0; i < F; ++i) {
                v[static_cast<size_t>(b) * F + i] = in.hx[static_cast<size_t>(b) * D + int64_t(r) * F + i];
            }
        }
        if (!hres[r]->put(h, st) || !hx[r]->put(v, st) || !x[r]->poison(st) || !mom[r]->poison(st) ||
            !inj[r]->poison(st) || !mix[r]->poison(st)) {
            return false;
        }
    }
    if (!hpost.put(in.hpost, st)) {
        return false;
    }
    const auto enqueue = [&]() {
        for (int32_t r = 0; r < 2; ++r) {
            const fn::hc::down_args da = {wd[r]->p, hres[r]->p, hx[r]->p, hpost.p, gm[r]->p, x[r]->p, mom[r]->p};
            const bool sent = cand.shipped ? fn::hc::down(B, da, st) : fn::hc::down_variant(cand.depth, B, da, st);
            if (!sent) {
                fprintf(stderr, "hc-test: hc down card %d not enqueued: %s\n", r,
                        cudaGetErrorString(cudaGetLastError()));
                return false;
            }
        }
        for (int32_t r = 0; r < 2; ++r) {
            const fn::hc::up_args ua = {wu[r]->p, mom[r]->p, mom[1 - r]->p, x[r]->p, gm[r]->p, inj[r]->p, mix[r]->p};
            const bool sent = cand.shipped ? fn::hc::up(B, ua, st) : fn::hc::up_variant(cand.depth, B, ua, st);
            if (!sent) {
                fprintf(stderr, "hc-test: hc up card %d not enqueued: %s\n", r, cudaGetErrorString(cudaGetLastError()));
                return false;
            }
        }
        return true;
    };
    const auto read = [&](gpu_out & o) {
        for (int32_t r = 0; r < 2; ++r) {
            if (!x[r]->get(o.x[r], st) || !mom[r]->get(o.mom[r], st) || !inj[r]->get(o.inject[r], st) ||
                !mix[r]->get(o.mixed[r], st)) {
                return false;
            }
        }
        return cuda_ok(cudaStreamSynchronize(st), "hc kernels");
    };
    if (!enqueue() || !read(out)) {
        return false;
    }
    if (replays <= 0) {
        return true;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok = cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = x[r]->poison(st) && mom[r]->poison(st) && inj[r]->poison(st) && mix[r]->poison(st);
    }
    ok = ok && enqueue();
    ok = cuda_ok(cudaStreamEndCapture(st, &graph), "end capture") && ok;
    // the captured graph has the four kernel nodes and no cooperative one
    if (ok) {
        size_t n_nodes = 0;
        ok = cuda_ok(cudaGraphGetNodes(graph, nullptr, &n_nodes), "graph nodes");
        std::vector<cudaGraphNode_t> nodes(n_nodes);
        ok = ok && cuda_ok(cudaGraphGetNodes(graph, nodes.data(), &n_nodes), "graph nodes");
        int32_t kernels = 0, cooperative = 0;
        for (size_t i = 0; i < n_nodes && ok; ++i) {
            cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
            ok = cuda_ok(cudaGraphNodeGetType(nodes[i], &type), "node type");
            if (ok && type == cudaGraphNodeTypeKernel) {
                cudaLaunchAttributeValue v = {};
                ok = cuda_ok(cudaGraphKernelNodeGetAttribute(nodes[i], cudaLaunchAttributeCooperative, &v), "attr");
                ++kernels;
                cooperative += v.cooperative != 0 ? 1 : 0;
            }
        }
        const int32_t want = 0;
        if (ok && (kernels != 4 || cooperative != want)) {
            fprintf(stderr, "hc-test: graph has %d kernel nodes, %d cooperative (want 4 and %d)\n", kernels,
                    cooperative, want);
            ok = false;
        }
    }
    ok = ok && cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "graph instantiate");
    int32_t diffs = 0;
    for (int32_t i = 0; i < replays && ok; ++i) {
        gpu_out again;
        ok = cuda_ok(cudaGraphLaunch(exec, st), "graph launch") && read(again);
        diffs += ok && !same_bits(out, again) ? 1 : 0;
    }
    if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
    }
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    if (replay_diffs != nullptr) {
        *replay_diffs = diffs;
    }
    return ok;
}

// max |gpu - ref| / max |ref| over n elements, where gpu[i] pairs with ref[at(i)]; a non-finite gpu value fails
template <typename At> double rel_err(const float * gpu, const double * ref, size_t n, At at) {
    double err = 0.0, mag = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double g = gpu[i];
        const double r = ref[at(i)];
        if (!std::isfinite(g)) {
            return INFINITY;
        }
        err = std::max(err, std::fabs(g - r));
        mag = std::max(mag, std::fabs(r));
    }
    return mag > 0.0 ? err / mag : err;
}

// card -1: the assembled full width
bool check(const char * what, int32_t card, double rel, const char * label) {
    const bool pass = rel <= rel_bound;
    const std::string half = card < 0 ? "full" : "card " + std::to_string(card);
    printf("hc-test %-19s %-6s %-6s: %.3e%s\n", label, what, half.c_str(), rel, pass ? "" : "  FAIL");
    return pass;
}

bool compare(const inputs & in, const reference & ref, const gpu_out & out, const char * label) {
    const int32_t B = in.rows;
    bool ok = true;
    for (int32_t r = 0; r < 2; ++r) {
        // x: card r's [B][4][1280] against the full-width reference
        ok = check("x", r,
                   rel_err(out.x[r].data(), ref.x.data(), out.x[r].size(),
                           [&](size_t i) {
                               const size_t bs = i / F;
                               return bs * D + int64_t(r) * F + i % F;
                           }),
                   label) &&
             ok;
        // the moments: v, u and q separately
        std::vector<float> g_v, g_u, g_q;
        std::vector<double> r_v, r_u, r_q;
        for (int32_t b = 0; b < B; ++b) {
            for (int32_t e = 0; e < M; ++e) {
                const float g = out.mom[r][static_cast<size_t>(b) * M + e];
                const double rv = ref.mom[r][static_cast<size_t>(b) * M + e];
                if (e < 4) {
                    g_v.push_back(g), r_v.push_back(rv);
                } else if (e < 4 + L * S) {
                    g_u.push_back(g), r_u.push_back(rv);
                } else {
                    g_q.push_back(g), r_q.push_back(rv);
                }
            }
        }
        const auto same = [](size_t i) { return i; };
        ok = check("v", r, rel_err(g_v.data(), r_v.data(), g_v.size(), same), label) && ok;
        ok = check("u", r, rel_err(g_u.data(), r_u.data(), g_u.size(), same), label) && ok;
        ok = check("b", r, rel_err(g_q.data(), r_q.data(), g_q.size(), same), label) && ok;
        const double e_inject = rel_err(out.inject[r].data(), ref.inject.data(), out.inject[r].size(), same);
        ok = check("inject", r, e_inject, label) && ok;
        ok = check("mixed", r,
                   rel_err(out.mixed[r].data(), ref.mixed.data(), out.mixed[r].size(),
                           [&](size_t i) { return (i / F) * D + int64_t(r) * F + i % F; }),
                   label) &&
             ok;
    }
    // the assembled full-width mixed: card 0's half, then card 1's
    std::vector<float> full(static_cast<size_t>(B) * D);
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t r = 0; r < 2; ++r) {
            std::copy_n(&out.mixed[r][static_cast<size_t>(b) * F], F,
                        &full[static_cast<size_t>(b) * D + int64_t(r) * F]);
        }
    }
    const double e_full = rel_err(full.data(), ref.mixed.data(), full.size(), [](size_t i) { return i; });
    ok = check("mixed", -1, e_full, label) && ok;
    return ok;
}

bool same_bits(const gpu_out & a, const gpu_out & b) {
    const auto eq = [](const std::vector<float> & p, const std::vector<float> & q) {
        return p.size() == q.size() && memcmp(p.data(), q.data(), p.size() * sizeof(float)) == 0;
    };
    bool ok = true;
    for (int32_t r = 0; r < 2; ++r) {
        ok = ok && eq(a.x[r], b.x[r]) && eq(a.mom[r], b.mom[r]);
        ok = ok && eq(a.inject[r], b.inject[r]) && eq(a.mixed[r], b.mixed[r]);
    }
    return ok;
}

// unsupported row counts and null or misaligned pointers enqueue nothing and return false
bool check_rejects(const device_weights & dw, cudaStream_t st) {
    dev_buf<float> buf(static_cast<size_t>(fn::hc::max_rows) * S * F);
    const fn::hc::down_args da = {dw.w_down0.p, buf.p, buf.p, buf.p, buf.p, buf.p, buf.p};
    const fn::hc::up_args ua = {dw.w_up0.p, buf.p, buf.p, buf.p, buf.p, buf.p, buf.p};
    fn::hc::down_args bad = da;
    bad.hx = buf.p + 1;
    bool ok = !fn::hc::down(0, da, st) && !fn::hc::down(5, da, st) && !fn::hc::up(0, ua, st) &&
              !fn::hc::up(5, ua, st) && !fn::hc::down(3, bad, st) && !fn::hc::down(3, da, nullptr);
    ok = ok && !fn::hc::down_variant(4, 3, da, st) && !fn::hc::up_variant(5, 3, ua, st) &&
         !fn::hc::down_variant(0, 3, da, st) && !fn::hc::probe(4, dw.w_down0.p, nullptr, st);
    printf("hc-test rejects B = 0 and 5, a misaligned pointer, a null stream and depths 0, 4 and 5: %s\n",
           ok ? "yes" : "NO  FAIL");
    return ok;
}

// ---- HC2: the sublayer on both cards ----

constexpr int32_t n_sub = 6;           // sublayers a case, back to back: each slot used twice
constexpr double hc2_timeout_s = 30.0; // the host's bound on one case

// sets the device before the members that follow it are built (and the owner sets it again before they are freed)
struct device_guard {
    explicit device_guard(int32_t d) { cudaSetDevice(d); }
};

// one card's weights, inputs and per-sublayer outputs (each sized for max_rows), on its own device
struct card2 {
    static constexpr size_t n_x = size_t(fn::hc::max_rows) * S * F, n_m = size_t(fn::hc::max_rows) * M;
    static constexpr size_t n_j = size_t(fn::hc::max_rows) * J, n_h = size_t(fn::hc::max_rows) * F;
    static constexpr size_t n_full = size_t(fn::hc::max_rows) * D;
    int32_t dev;
    device_guard guard;
    cudaStream_t st = nullptr;
    dev_buf<uint16_t> w_down{size_t(fn::hc::w_down_elems)}, w_up{size_t(fn::hc::w_up_elems)};
    dev_buf<float> gamma{size_t(S) * F}, hres{n_x}, hx{n_h}, hpost{size_t(fn::hc::max_rows) * S};
    dev_buf<float> x{n_sub * n_x}, mom{n_sub * n_m}, inj{n_sub * n_j}, full{n_sub * n_full};
    // HC3: each sublayer's codes and scales as ready assembled them from X-HMIX, and up2's run counters
    static constexpr size_t n_sc = size_t(fn::hc::max_rows) * (D / 32) * 2;
    dev_buf<int8_t> codes{n_sub * n_full};
    dev_buf<float> scales{n_sub * n_sc};
    dev_buf<uint32_t> runs{size_t(fn::hc::n_runs_card)};
    // DR1: the Q8_0 weights (hc.h), and the head's down weights (no injection rows) in both formats
    dev_buf<uint8_t> q_down{size_t(fn::hc::w_down_q8_bytes)}, q_up{size_t(fn::hc::w_up_q8_bytes)};
    dev_buf<uint16_t> hw_down{size_t(fn::hc::w_down_head_elems)};
    dev_buf<uint8_t> hq_down{size_t(fn::hc::w_down_head_q8_bytes)};

    explicit card2(int32_t d) : dev(d), guard(d) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) {
            st = nullptr;
        }
    }

    ~card2() {
        cudaSetDevice(dev);
        if (st != nullptr) {
            cudaStreamDestroy(st);
        }
    }

    card2(const card2 &) = delete;
    card2 & operator=(const card2 &) = delete;
};

bool wait_two(card2 * c[2], double limit_s) {
    const auto t0 = std::chrono::steady_clock::now();
    bool done[2] = {false, false};
    while (!(done[0] && done[1])) {
        for (int32_t r = 0; r < 2; ++r) {
            if (done[r]) {
                continue;
            }
            const cudaError_t e = cudaStreamQuery(c[r]->st);
            if (e == cudaSuccess) {
                done[r] = true;
            } else if (e != cudaErrorNotReady) {
                return cuda_ok(e, "card stream");
            }
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > limit_s) {
            fprintf(stderr, "hc-test: host timeout after %.0f s\n", limit_s);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

// a slice of n elements at offset at of a device buffer, read to the host
template <typename T> bool get_part(const dev_buf<T> & d, size_t at, size_t n, std::vector<T> & h, cudaStream_t st) {
    h.assign(n, T{});
    return cuda_ok(cudaMemcpyAsync(h.data(), d.p + at, n * sizeof(T), cudaMemcpyDeviceToHost, st), "get part");
}

// the stream card r's kernels go on: in loopback (both cards on one device) card 0's, so every send precedes its read
cudaStream_t launch_stream(card2 * c[2], int32_t r) {
    return c[0]->dev == c[1]->dev ? c[0]->st : c[r]->st;
}

// sublayer 0's outputs on both cards, for the batch-invariance check
struct pair_out {
    std::vector<float> x[2], inject[2], full[2], scales[2];
    std::vector<int8_t> codes[2];
};

// every word of X-HMIX slot sl on both cards is a sentinel (code words code_sentinel): down2's reset (HC3)
bool hmix_slot_reset(fn::xchg::rings & rg, int32_t sl, int64_t & bad) {
    using hl = fn::xchg::hmix_layout;
    bad = 0;
    for (int32_t r = 0; r < 2; ++r) {
        std::vector<uint32_t> w(static_cast<size_t>(hl::slot_words));
        if (!cuda_ok(cudaMemcpy(w.data(), rg.slot(r, fn::xchg::msg::hmix, sl), w.size() * 4, cudaMemcpyDeviceToHost),
                     "slot")) {
            return false;
        }
        for (size_t i = 0; i < w.size(); ++i) {
            const bool code = i >= size_t(hl::code_at) && i < size_t(hl::scale_at);
            bad += w[i] != (code ? fn::xchg::code_sentinel : fn::xchg::sentinel) ? 1 : 0;
        }
    }
    return true;
}

// each card's half of the inputs (enqueued on its stream); rows in.rows .. 4 hold tail (FMS13's sentinel test: NaN)
bool put_inputs(card2 * c[2], const inputs & in, float tail = 0.0f) {
    bool ok = true;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        std::vector<float> h(card2::n_x, tail), v(card2::n_h, tail), post(size_t(fn::hc::max_rows) * S, tail);
        for (int32_t b = 0; b < in.rows; ++b) {
            for (int32_t s = 0; s < S; ++s) {
                for (int32_t i = 0; i < F; ++i) {
                    const size_t bs = static_cast<size_t>(b) * S + s;
                    h[bs * F + i] = in.hres[bs * D + int64_t(r) * F + i];
                }
                post[static_cast<size_t>(b) * S + s] = in.hpost[static_cast<size_t>(b) * S + s];
            }
            for (int32_t i = 0; i < F; ++i) {
                v[static_cast<size_t>(b) * F + i] = in.hx[static_cast<size_t>(b) * D + int64_t(r) * F + i];
            }
        }
        ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") && c[r]->hres.put(h, c[r]->st) &&
             c[r]->hx.put(v, c[r]->st) && c[r]->hpost.put(post, c[r]->st);
    }
    return ok;
}

// HC4: hc::sub's arguments for card r's sublayer n of a case (slot n % 3; the previous sublayer's slots reset), with
// the head's weights and no inject in head mode
fn::hc::sub_args sub_args_of(card2 & k, fn::xchg::rings & rg, int32_t r, int32_t n, bool head) {
    using fn::xchg::msg;
    const int32_t sl = n % fn::xchg::n_slot;
    const int32_t prev = (n + fn::xchg::n_slot - 1) % fn::xchg::n_slot;
    fn::hc::sub_args a = {};
    a.w_down = head ? k.hw_down.p : k.w_down.p;
    a.w_up = k.w_up.p;
    a.hres = k.hres.p;
    a.hx = k.hx.p;
    a.hpost = k.hpost.p;
    a.gamma = k.gamma.p;
    a.x = k.x.p + n * card2::n_x;
    a.inject = head ? nullptr : k.inj.p + n * card2::n_j;
    a.hc_ring = rg.slot(r, msg::hc, sl);
    a.hc_peer = rg.slot(1 - r, msg::hc, sl);
    a.hc_reset = n > 0 ? rg.slot(r, msg::hc, prev) : nullptr;
    a.hmix_reset = n > 0 ? rg.slot(r, msg::hmix, prev) : nullptr;
    a.hmix_own = rg.slot(r, msg::hmix, sl);
    a.hmix_peer = rg.slot(1 - r, msg::hmix, sl);
    a.err = rg.err(r);
    a.card = r;
    return a;
}

// Runs n_sub sublayers of one arm on both cards and checks every sublayer against the HC1c reference bits and the
// first against the FP64 reference; HC3: both forms of mixed, the codes and scales bitwise act_ref of the full F32 row
// on both cards (C2's quantize of HC2's row, which act-test holds bitwise to act_ref), and down2's X-HMIX resets.
// DR1: q8 runs down2_q8 and up2_q8 (arm 0) on the cards' Q8_0 weights (the BF16 weight pointers passed null). Without
// bits (Q8_0 weights BF16 cannot hold) every sublayer is checked bitwise against sublayer 0 on card 0 instead, and the
// codes against act_ref of that row.
// HC4: fused runs hc::sub instead (mode 3 on both cards at once; in loopback mode 1 on both, then mode 2 on both),
// and copies each card's moments from the own region of its X-HC slot after the launch, before the next sublayer's
// reset, so they are checked as down2's are.
bool run_hc2_case(card2 * c[2], fn::xchg::rings & rg, const inputs & in, const reference & ref, const gpu_out * bits,
                  fn::hc::up_arm arm, const std::string & label, pair_out * keep = nullptr, bool q8 = false,
                  bool fused = false) {
    using fn::xchg::msg;
    const int32_t B = in.rows;
    bool ok = rg.reset_all() && put_inputs(c, in);
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") && c[r]->x.poison(c[r]->st) &&
             c[r]->mom.poison(c[r]->st) && c[r]->inj.poison(c[r]->st) && c[r]->full.poison(c[r]->st) &&
             c[r]->codes.poison(c[r]->st) && c[r]->scales.poison(c[r]->st) &&
             cuda_ok(cudaMemsetAsync(c[r]->runs.p, 0, c[r]->runs.n * sizeof(uint32_t), c[r]->st), "runs") &&
             cuda_ok(cudaStreamSynchronize(c[r]->st), "inputs");
    }
    // sublayer n: down2 on both cards, up2 on both, ready on both (each card's stream orders its own three); ready
    // leaves the slot, and down2(n + 1) resets X-HMIX(n) as in the model
    const bool loopback = c[0]->dev == c[1]->dev;
    for (int32_t n = 0; n < n_sub && ok && fused; ++n) {
        const int32_t sl = n % fn::xchg::n_slot;
        for (const int32_t mode : {loopback ? 1 : 3, loopback ? 2 : 0}) {
            for (int32_t r = 0; r < 2 && ok && mode != 0; ++r) {
                ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") &&
                     fn::hc::sub(B, mode, sub_args_of(*c[r], rg, r, n, false), launch_stream(c, r));
            }
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card2 & k = *c[r];
            const fn::hc::ready_args ra = {rg.slot(r, msg::hmix, sl),
                                           k.full.p + n * card2::n_full,
                                           k.codes.p + n * card2::n_full,
                                           k.scales.p + n * card2::n_sc,
                                           rg.err(r),
                                           r,
                                           0};
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
                 cuda_ok(cudaMemcpyAsync(k.mom.p + n * card2::n_m, rg.slot(r, msg::hc, sl) + fn::xchg::hc_own_at,
                                         static_cast<size_t>(B) * M * sizeof(float), cudaMemcpyDeviceToDevice,
                                         launch_stream(c, r)),
                         "own moments") &&
                 fn::hc::ready(B, ra, launch_stream(c, r));
        }
        if (!ok) {
            fprintf(stderr, "hc-test: %s sublayer %d not enqueued: %s\n", label.c_str(), n,
                    cudaGetErrorString(cudaGetLastError()));
        }
    }
    for (int32_t n = 0; n < n_sub && ok && !fused; ++n) {
        const int32_t sl = n % fn::xchg::n_slot;
        const int32_t prev = (n + fn::xchg::n_slot - 1) % fn::xchg::n_slot;
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card2 & k = *c[r];
            const fn::hc::down_args da = {
                q8 ? nullptr : k.w_down.p, k.hres.p, k.hx.p, k.hpost.p, k.gamma.p, k.x.p + n * card2::n_x,
                k.mom.p + n * card2::n_m};
            const fn::hc::down2_args d2 = {da, rg.slot(1 - r, msg::hc, sl), n > 0 ? rg.slot(r, msg::hc, prev) : nullptr,
                                           n > 0 ? rg.slot(r, msg::hmix, prev) : nullptr};
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
                 (q8 ? fn::hc::down2_q8(B, {d2, k.q_down.p}, launch_stream(c, r))
                     : fn::hc::down2(B, d2, launch_stream(c, r)));
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card2 & k = *c[r];
            const fn::hc::up2_args ua = {q8 ? nullptr : k.w_up.p,
                                         k.mom.p + n * card2::n_m,
                                         rg.slot(r, msg::hc, sl),
                                         k.x.p + n * card2::n_x,
                                         k.gamma.p,
                                         k.inj.p + n * card2::n_j,
                                         rg.slot(r, msg::hmix, sl),
                                         rg.slot(1 - r, msg::hmix, sl),
                                         k.runs.p,
                                         rg.err(r),
                                         r};
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
                 (q8 ? arm == fn::hc::up_arm::ring && fn::hc::up2_q8(B, {ua, k.q_up.p}, launch_stream(c, r))
                     : fn::hc::up2(arm, B, ua, launch_stream(c, r)));
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card2 & k = *c[r];
            const fn::hc::ready_args ra = {rg.slot(r, msg::hmix, sl),
                                           k.full.p + n * card2::n_full,
                                           k.codes.p + n * card2::n_full,
                                           k.scales.p + n * card2::n_sc,
                                           rg.err(r),
                                           r,
                                           0};
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::hc::ready(B, ra, launch_stream(c, r));
        }
        if (!ok) {
            fprintf(stderr, "hc-test: %s sublayer %d not enqueued: %s\n", label.c_str(), n,
                    cudaGetErrorString(cudaGetLastError()));
        }
    }
    ok = ok && wait_two(c, hc2_timeout_s);
    uint32_t err[2] = {~0u, ~0u};
    ok = ok && rg.read_err(0, err[0]) && rg.read_err(1, err[1]);
    if (ok && (err[0] != 0u || err[1] != 0u)) {
        printf("hc-test %s: error words 0x%x 0x%x  FAIL\n", label.c_str(), err[0], err[1]);
        return false;
    }
    if (!ok) {
        return false;
    }
    // the assembled full width the HC1c halves give (without bits: sublayer 0's on card 0, set below)
    std::vector<float> want_full(static_cast<size_t>(B) * D);
    for (int32_t b = 0; b < B && bits != nullptr; ++b) {
        for (int32_t r = 0; r < 2; ++r) {
            std::copy_n(&bits->mixed[r][static_cast<size_t>(b) * F], F,
                        &want_full[static_cast<size_t>(b) * D + int64_t(r) * F]);
        }
    }
    // the activation format of that row (act_ref), which C2's quantize gives bitwise
    std::vector<int8_t> want_codes(static_cast<size_t>(B) * D);
    std::vector<float> want_scales(static_cast<size_t>(B) * (D / 32) * 2);
    if (bits != nullptr) {
        fn::quant::act_quantize_ref(want_full.data(), B, D, want_codes.data(), want_scales.data());
    }
    bool pass = true;
    int32_t n_bitwise = 0, n_format = 0;
    gpu_out first; // without bits: sublayer 0's outputs
    for (int32_t n = 0; n < n_sub && ok; ++n) {
        gpu_out g;
        std::vector<float> full[2], scales[2];
        std::vector<int8_t> codes[2];
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card2 & k = *c[r];
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
                 get_part(k.x, n * card2::n_x, static_cast<size_t>(B) * S * F, g.x[r], k.st) &&
                 get_part(k.mom, n * card2::n_m, static_cast<size_t>(B) * M, g.mom[r], k.st) &&
                 get_part(k.inj, n * card2::n_j, static_cast<size_t>(B) * J, g.inject[r], k.st) &&
                 get_part(k.full, n * card2::n_full, static_cast<size_t>(B) * D, full[r], k.st) &&
                 get_part(k.codes, n * card2::n_full, static_cast<size_t>(B) * D, codes[r], k.st) &&
                 get_part(k.scales, n * card2::n_sc, want_scales.size(), scales[r], k.st) &&
                 cuda_ok(cudaStreamSynchronize(k.st), "read");
            // the card's own half of mixed, from the full row
            g.mixed[r].resize(static_cast<size_t>(B) * F);
            for (int32_t b = 0; b < B && ok; ++b) {
                std::copy_n(&full[r][static_cast<size_t>(b) * D + int64_t(r) * F], F,
                            &g.mixed[r][static_cast<size_t>(b) * F]);
            }
        }
        if (!ok) {
            break;
        }
        if (n == 0 && bits == nullptr) {
            first = g;
            want_full = full[0];
            fn::quant::act_quantize_ref(want_full.data(), B, D, want_codes.data(), want_scales.data());
        }
        if (n == 0 && keep != nullptr) {
            for (int32_t r = 0; r < 2; ++r) {
                keep->x[r] = g.x[r];
                keep->inject[r] = g.inject[r];
                keep->full[r] = full[r];
                keep->codes[r] = codes[r];
                keep->scales[r] = scales[r];
            }
        }
        bool format = true;
        for (int32_t r = 0; r < 2; ++r) {
            const bool cr = codes[r] == want_codes;
            const bool sr = memcmp(scales[r].data(), want_scales.data(), want_scales.size() * sizeof(float)) == 0;
            if (!cr || !sr) {
                size_t ic = 0, is = 0;
                while (ic < want_codes.size() && codes[r][ic] == want_codes[ic]) {
                    ++ic;
                }
                while (is < want_scales.size() && memcmp(&scales[r][is], &want_scales[is], sizeof(float)) == 0) {
                    ++is;
                }
                int32_t n_bad_runs = 0;
                for (size_t j = 0; j < want_scales.size() / 2; ++j) {
                    n_bad_runs += memcmp(&scales[r][2 * j], &want_scales[2 * j], 8) != 0 ? 1 : 0;
                }
                printf(
                    "hc-test %-19s sublayer %d card %d: first code mismatch at %zu (%d vs %d), first scale mismatch at "
                    "%zu (%g vs %g), %d runs' scales differ\n",
                    label.c_str(), n, r, ic, ic < want_codes.size() ? codes[r][ic] : 0,
                    ic < want_codes.size() ? want_codes[ic] : 0, is, is < want_scales.size() ? scales[r][is] : 0.0,
                    is < want_scales.size() ? want_scales[is] : 0.0, n_bad_runs);
            }
            format = format && cr && sr;
        }
        n_format += format ? 1 : 0;
        if (n == 0) {
            pass = compare(in, ref, g, label.c_str()) && pass;
            for (int32_t r = 0; r < 2; ++r) {
                const double e = rel_err(full[r].data(), ref.mixed.data(), full[r].size(), [](size_t i) { return i; });
                pass = check("full", r, e, label.c_str()) && pass;
            }
        }
        const bool same = same_bits(g, bits != nullptr ? *bits : first) && full[0].size() == want_full.size() &&
                          memcmp(full[0].data(), want_full.data(), want_full.size() * sizeof(float)) == 0 &&
                          memcmp(full[1].data(), want_full.data(), want_full.size() * sizeof(float)) == 0;
        n_bitwise += same ? 1 : 0;
    }
    printf("hc-test %-19s sublayers bitwise identical to %s (x, moments, inject, mixed and the full mixed, both "
           "cards): %d of %d%s\n",
           label.c_str(), bits != nullptr ? "HC1c's two halves on one card" : "sublayer 0 on card 0", n_bitwise, n_sub,
           n_bitwise == n_sub ? "" : "  FAIL");
    printf("hc-test %-19s sublayers whose codes and scales (X-HMIX, both cards) are bitwise act_ref of that full row: "
           "%d of %d%s\n",
           label.c_str(), n_format, n_sub, n_format == n_sub ? "" : "  FAIL");
    // slots 0 .. n_slot - 1 were last used by sublayers n_sub - 3 .. n_sub - 1; all but the last were reset by the
    // down2 after their readers
    int64_t not_reset = 0;
    for (int32_t n = n_sub - fn::xchg::n_slot; n < n_sub - 1 && ok; ++n) {
        int64_t bad = 0;
        ok = hmix_slot_reset(rg, n % fn::xchg::n_slot, bad);
        not_reset += bad;
    }
    printf("hc-test %-19s X-HMIX slots reset by the next down2 (both cards, every word): %s\n", label.c_str(),
           ok && not_reset == 0 ? "yes" : "NO  FAIL");
    return ok && pass && n_bitwise == n_sub && n_format == n_sub && not_reset == 0;
}

// NC1 section 2: each row of the B = max_rows case alone (B = 1) and with the rows before it (B = 2 .. max_rows - 1)
// gives bitwise the same x, inject, F32 mixed, codes and scales on both cards, for both arms. DR1: q8 runs the Q8_0
// forms (arm 0) on the cards' Q8_0 weights, without HC1c's bits (dw is not read). SP2: HC1c's bits only for rows up to
// hc1_max_rows (its kernels' rows).
bool batch_invariance(card2 * c[2], fn::xchg::rings & rg, const weights & w, const device_weights & dw, cudaStream_t st,
                      int32_t dev0, bool q8 = false) {
    inputs in4;
    make_inputs(in4, fn::hc::max_rows, kind::plain);
    bool ok = true;
    int32_t n_same = 0, n_cmp = 0;
    const std::string pre = q8 ? "hc2 q8 batch B" : "hc2 batch B";
    // HC4: variant 2 is the one-launch sublayer (hc::sub), BF16 only
    std::vector<int32_t> variants = {0};
    if (!q8) {
        variants.push_back(1);
        variants.push_back(2);
    }
    for (const int32_t var : variants) {
        const fn::hc::up_arm arm = var == 1 ? fn::hc::up_arm::staged : fn::hc::up_arm::ring;
        const bool fused = var == 2;
        const std::string name = var == 0 ? "arm0" : var == 1 ? "armA" : "fused";
        pair_out o4;
        reference ref4;
        make_reference(w, in4, ref4);
        gpu_out bits4;
        ok = ok && cuda_ok(cudaSetDevice(dev0), "cudaSetDevice") &&
             (q8 || in4.rows > fn::hc::hc1_max_rows || run_gpu(dw, in4, bits4, st, candidates[0])) &&
             run_hc2_case(c, rg, in4, ref4, q8 || in4.rows > fn::hc::hc1_max_rows ? nullptr : &bits4, arm,
                          pre + std::to_string(in4.rows) + " " + name, &o4, q8, fused);
        for (int32_t B = 1; B < fn::hc::max_rows && ok; ++B) {
            inputs in;
            in.rows = B;
            in.hres.assign(in4.hres.begin(), in4.hres.begin() + static_cast<int64_t>(B) * S * D);
            in.hx.assign(in4.hx.begin(), in4.hx.begin() + static_cast<int64_t>(B) * D);
            in.hpost.assign(in4.hpost.begin(), in4.hpost.begin() + static_cast<int64_t>(B) * S);
            reference ref;
            make_reference(w, in, ref);
            gpu_out bits;
            pair_out o;
            ok = cuda_ok(cudaSetDevice(dev0), "cudaSetDevice") && (q8 || run_gpu(dw, in, bits, st, candidates[0])) &&
                 run_hc2_case(c, rg, in, ref, q8 ? nullptr : &bits, arm, pre + std::to_string(B) + " " + name, &o, q8,
                              fused);
            for (int32_t r = 0; r < 2 && ok; ++r) {
                const auto prefix = [](const auto & a, const auto & b) {
                    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin(), [](auto u, auto v) {
                               return memcmp(&u, &v, sizeof(u)) == 0;
                           });
                };
                const bool same = prefix(o.x[r], o4.x[r]) && prefix(o.inject[r], o4.inject[r]) &&
                                  prefix(o.full[r], o4.full[r]) && prefix(o.codes[r], o4.codes[r]) &&
                                  prefix(o.scales[r], o4.scales[r]);
                n_same += same ? 1 : 0;
                ++n_cmp;
            }
        }
    }
    printf("hc-test %s batch invariance: rows at B = 1 .. %d against the same rows inside B = %d (x, inject, F32 "
           "mixed, codes, scales; both cards, %s): %d of %d bitwise%s\n",
           q8 ? "hc2 q8" : "hc2", fn::hc::max_rows - 1, fn::hc::max_rows,
           q8 ? "arm 0" : "both arms and the fused launch", n_same, n_cmp,
           ok && n_same == n_cmp ? "" : "  FAIL");
    return ok && n_same == n_cmp;
}

bool check_rejects2(card2 & k, fn::xchg::rings & rg) {
    using fn::xchg::msg;
    cudaSetDevice(k.dev);
    const fn::hc::down_args da = {k.w_down.p, k.hres.p, k.hx.p, k.hpost.p, k.gamma.p, k.x.p, k.mom.p};
    const fn::hc::down2_args d2 = {da, rg.slot(1, msg::hc, 0), nullptr};
    fn::hc::down2_args d2_bad = d2, d2_hmix = d2;
    d2_bad.hc_peer = nullptr;
    d2_hmix.hmix_reset = rg.slot(0, msg::hmix, 0) + 1;
    const fn::hc::up2_args ua = {k.w_up.p,
                                 k.mom.p,
                                 rg.slot(0, msg::hc, 0),
                                 k.x.p,
                                 k.gamma.p,
                                 k.inj.p,
                                 rg.slot(0, msg::hmix, 0),
                                 rg.slot(1, msg::hmix, 0),
                                 k.runs.p,
                                 rg.err(0),
                                 0};
    fn::hc::up2_args ua_card = ua, ua_err = ua, ua_runs = ua, ua_own = ua;
    ua_card.card = 2;
    ua_err.err = nullptr;
    ua_runs.runs = nullptr;
    ua_own.hmix_own = nullptr;
    const fn::hc::ready_args ra = {rg.slot(0, msg::hmix, 0), k.full.p, nullptr, nullptr, rg.err(0), 0, 0};
    fn::hc::ready_args ra_codes = ra;
    ra_codes.codes = k.codes.p; // scales null
    const bool ok = !fn::hc::down2(0, d2, k.st) && !fn::hc::down2(6, d2, k.st) && !fn::hc::down2(3, d2_bad, k.st) &&
                    !fn::hc::down2(3, d2_hmix, k.st) && !fn::hc::up2(fn::hc::up_arm::ring, 6, ua, k.st) &&
                    !fn::hc::up2(static_cast<fn::hc::up_arm>(2), 3, ua, k.st) &&
                    !fn::hc::up2(fn::hc::up_arm::staged, 3, ua_card, k.st) &&
                    !fn::hc::up2(fn::hc::up_arm::staged, 3, ua_err, k.st) &&
                    !fn::hc::up2(fn::hc::up_arm::ring, 3, ua_runs, k.st) &&
                    !fn::hc::up2(fn::hc::up_arm::ring, 3, ua_own, k.st) && !fn::hc::ready(6, ra, k.st) &&
                    !fn::hc::ready(3, ra, nullptr) && !fn::hc::ready(3, ra_codes, k.st);
    printf("hc-test hc2 rejects B = 0 and 6, arm 2, card 2, a null peer slot, own slot, run counters, error word or "
           "stream, a misaligned X-HMIX reset and codes without scales: %s\n",
           ok ? "yes" : "NO  FAIL");
    // HC4: hc::sub and head_sub
    const fn::hc::sub_args sa = sub_args_of(k, rg, 0, 1, false);
    fn::hc::sub_args sa_ring = sa, sa_reset = sa, sa_card = sa, sa_err = sa, sa_inject = sa;
    sa_ring.hc_ring = nullptr;
    sa_reset.hmix_reset = sa.hmix_reset + 1;
    sa_card.card = 2;
    sa_err.err = nullptr;
    sa_inject.inject = nullptr;
    const bool ok_sub =
        !fn::hc::sub(0, 3, sa, k.st) && !fn::hc::sub(6, 3, sa, k.st) && !fn::hc::sub(3, 0, sa, k.st) &&
        !fn::hc::sub(3, 4, sa, k.st) && !fn::hc::sub(3, 3, sa_ring, k.st) && !fn::hc::sub(3, 3, sa_reset, k.st) &&
        !fn::hc::sub(3, 3, sa_card, k.st) && !fn::hc::sub(3, 3, sa_err, k.st) && !fn::hc::sub(3, 3, sa_inject, k.st) &&
        !fn::hc::sub(3, 3, sa, nullptr) && !fn::hc::head_sub(6, 3, sa, k.st) && !fn::hc::head_sub(3, 0, sa, k.st);
    printf("hc-test hc2 sub rejects B = 0 and 6, modes 0 and 4, a null X-HC slot, a misaligned X-HMIX reset, card 2, a "
           "null error word, inject or stream: %s\n",
           ok_sub ? "yes" : "NO  FAIL");
    return ok && ok_sub;
}

// DR1: the HC head on both cards, one pass at slot 0 (head_down2, head_up2 and ready, or their Q8_0 forms on the
// cards' head weights), giving each card's full mixed row with its codes and scales
struct head_out {
    std::vector<float> x[2], full[2], scales[2];
    std::vector<int8_t> codes[2];
};

bool run_head(card2 * c[2], fn::xchg::rings & rg, const inputs & in, bool q8, head_out & out, bool fused = false) {
    using fn::xchg::msg;
    const int32_t B = in.rows;
    bool ok = rg.reset_all() && put_inputs(c, in);
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") && c[r]->x.poison(c[r]->st) &&
             c[r]->mom.poison(c[r]->st) && c[r]->full.poison(c[r]->st) && c[r]->codes.poison(c[r]->st) &&
             c[r]->scales.poison(c[r]->st) &&
             cuda_ok(cudaMemsetAsync(c[r]->runs.p, 0, c[r]->runs.n * sizeof(uint32_t), c[r]->st), "runs") &&
             cuda_ok(cudaStreamSynchronize(c[r]->st), "inputs");
    }
    // HC4: the head's one-launch form (hc::head_sub, BF16), mode 3 on both cards, or modes 1 then 2 in loopback
    const bool loopback = c[0]->dev == c[1]->dev;
    for (const int32_t mode : {loopback ? 1 : 3, loopback ? 2 : 0}) {
        for (int32_t r = 0; r < 2 && ok && fused && mode != 0; ++r) {
            ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") &&
                 fn::hc::head_sub(B, mode, sub_args_of(*c[r], rg, r, 0, true), launch_stream(c, r));
        }
    }
    for (int32_t r = 0; r < 2 && ok && !fused; ++r) {
        card2 & k = *c[r];
        const fn::hc::down_args da = {
            q8 ? nullptr : k.hw_down.p, k.hres.p, k.hx.p, k.hpost.p, k.gamma.p, k.x.p, k.mom.p};
        const fn::hc::down2_args d2 = {da, rg.slot(1 - r, msg::hc, 0), nullptr, nullptr};
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
             (q8 ? fn::hc::head_down2_q8(B, {d2, k.hq_down.p}, launch_stream(c, r))
                 : fn::hc::head_down2(B, d2, launch_stream(c, r)));
    }
    for (int32_t r = 0; r < 2 && ok && !fused; ++r) {
        card2 & k = *c[r];
        const fn::hc::up2_args ua = {q8 ? nullptr : k.w_up.p,
                                     k.mom.p,
                                     rg.slot(r, msg::hc, 0),
                                     k.x.p,
                                     k.gamma.p,
                                     nullptr,
                                     rg.slot(r, msg::hmix, 0),
                                     rg.slot(1 - r, msg::hmix, 0),
                                     k.runs.p,
                                     rg.err(r),
                                     r};
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
             (q8 ? fn::hc::head_up2_q8(B, {ua, k.q_up.p}, launch_stream(c, r))
                 : fn::hc::head_up2(B, ua, launch_stream(c, r)));
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card2 & k = *c[r];
        const fn::hc::ready_args ra = {rg.slot(r, msg::hmix, 0), k.full.p, k.codes.p, k.scales.p, rg.err(r), r, 0};
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::hc::ready(B, ra, launch_stream(c, r));
    }
    ok = ok && wait_two(c, hc2_timeout_s);
    uint32_t err[2] = {~0u, ~0u};
    ok = ok && rg.read_err(0, err[0]) && rg.read_err(1, err[1]);
    if (ok && (err[0] != 0u || err[1] != 0u)) {
        printf("hc-test head: error words 0x%x 0x%x  FAIL\n", err[0], err[1]);
        return false;
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card2 & k = *c[r];
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") &&
             get_part(k.x, 0, static_cast<size_t>(B) * S * F, out.x[r], k.st) &&
             get_part(k.full, 0, static_cast<size_t>(B) * D, out.full[r], k.st) &&
             get_part(k.codes, 0, static_cast<size_t>(B) * D, out.codes[r], k.st) &&
             get_part(k.scales, 0, static_cast<size_t>(B) * (D / 32) * 2, out.scales[r], k.st) &&
             cuda_ok(cudaStreamSynchronize(k.st), "read");
    }
    return ok;
}

// the head's bound (hc.h V1): the full mixed row a row, max |err| / max |ref| <= 1e-5
constexpr double head_bound = 1e-5;

// DR1: head_down2_q8 and head_up2_q8 against the FP64 reference (the layer's x and mixed: the head is the sublayer
// without injection), both cards equal, codes and scales act_ref of the row, and with bf16 (pow2 weights) the BF16
// head kernels' full rows and x bitwise
bool head_q8_case(card2 * c[2], fn::xchg::rings & rg, const inputs & in, const reference & ref, bool bf16,
                  const std::string & label) {
    head_out q, h, f;
    bool ok = run_head(c, rg, in, true, q) &&
              (!bf16 || (run_head(c, rg, in, false, h) && run_head(c, rg, in, false, f, true)));
    if (!ok) {
        return false;
    }
    const int32_t B = in.rows;
    double worst = 0.0;
    for (int32_t b = 0; b < B; ++b) {
        const size_t o = static_cast<size_t>(b) * D;
        worst = std::max(worst, rel_err(q.full[0].data() + o, ref.mixed.data() + o, D, [](size_t i) { return i; }));
    }
    const auto bitwise = [](const auto & a, const auto & b) {
        return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
    };
    std::vector<int8_t> want_codes(static_cast<size_t>(B) * D);
    std::vector<float> want_scales(static_cast<size_t>(B) * (D / 32) * 2);
    fn::quant::act_quantize_ref(q.full[0].data(), B, D, want_codes.data(), want_scales.data());
    const bool cards = bitwise(q.full[0], q.full[1]);
    const bool format = bitwise(q.codes[0], want_codes) && bitwise(q.codes[1], want_codes) &&
                        bitwise(q.scales[0], want_scales) && bitwise(q.scales[1], want_scales);
    bool same = true;
    bool same_fused = true; // HC4: the head's one-launch form, bitwise the two-launch BF16 head and the Q8_0 head
    for (int32_t r = 0; r < 2 && bf16; ++r) {
        same = same && bitwise(q.full[r], h.full[r]) && bitwise(q.x[r], h.x[r]);
        same_fused = same_fused && bitwise(f.full[r], h.full[r]) && bitwise(f.x[r], h.x[r]) &&
                     bitwise(f.codes[r], h.codes[r]) && bitwise(f.scales[r], h.scales[r]);
    }
    printf("hc-test %-24s full mixed rel %.3e (worst row, bound %.0e)%s; cards equal %s; "
           "codes and scales act_ref %s%s%s\n",
           label.c_str(), worst, head_bound, worst <= head_bound ? "" : "  FAIL", cards ? "yes" : "NO  FAIL",
           format ? "yes" : "NO  FAIL",
           bf16 ? (same ? "; full rows and x bitwise the BF16 head's: yes" : "; BF16 head bitwise: NO  FAIL") : "",
           bf16 ? (same_fused ? "; the fused head (x, full rows, codes, scales) bitwise: yes"
                              : "; the fused head bitwise: NO  FAIL")
                : "");
    return worst <= head_bound && cards && format && same && same_fused;
}

// FMU2 (build item 2, the joint draft step: lane k is row k of one U-row step): the Q8_0 sublayer and head at B = 2
// .. 5, each row bitwise the same row run alone at B = 1 (the sublayer's x, inject, F32 mixed, codes and scales; the
// head's x, mixed rows, codes and scales), on both cards, for plain and stress rows
bool row_alone_q8(card2 * c[2], fn::xchg::rings & rg, const weights & w, int32_t dev0) {
    bool ok = true;
    int32_t n_same = 0, n_cmp = 0;
    const auto row = [](const auto & v, int32_t B, int32_t r) {
        const size_t n = v.size() / static_cast<size_t>(B);
        return std::make_pair(v.data() + static_cast<size_t>(r) * n, n);
    };
    const auto same_row = [&](const auto & vb, int32_t B, int32_t r, const auto & v1) {
        const auto [p, n] = row(vb, B, r);
        return vb.size() % static_cast<size_t>(B) == 0 && n == v1.size() && memcmp(p, v1.data(), n * sizeof(*p)) == 0;
    };
    for (const kind kd : {kind::plain, kind::stress}) {
        inputs in5;
        make_inputs(in5, fn::hc::max_rows, kd);
        const auto slice = [&](int32_t r0, int32_t B) {
            inputs in;
            in.rows = B;
            in.hres.assign(in5.hres.begin() + int64_t(r0) * S * D, in5.hres.begin() + int64_t(r0 + B) * S * D);
            in.hx.assign(in5.hx.begin() + int64_t(r0) * D, in5.hx.begin() + int64_t(r0 + B) * D);
            in.hpost.assign(in5.hpost.begin() + int64_t(r0) * S, in5.hpost.begin() + int64_t(r0 + B) * S);
            return in;
        };
        pair_out o1[fn::hc::max_rows];
        head_out h1[fn::hc::max_rows];
        for (int32_t r = 0; r < fn::hc::max_rows && ok; ++r) {
            const inputs in = slice(r, 1);
            reference ref;
            make_reference(w, in, ref);
            ok = cuda_ok(cudaSetDevice(dev0), "cudaSetDevice") &&
                 run_hc2_case(c, rg, in, ref, nullptr, fn::hc::up_arm::ring,
                              std::string("hc2 q8 alone ") + kind_name(kd) + " row " + std::to_string(r), &o1[r],
                              true) &&
                 run_head(c, rg, in, true, h1[r]);
        }
        for (int32_t B = 2; B <= fn::hc::max_rows && ok; ++B) {
            const inputs in = slice(0, B);
            reference ref;
            make_reference(w, in, ref);
            pair_out o;
            head_out h;
            ok = cuda_ok(cudaSetDevice(dev0), "cudaSetDevice") &&
                 run_hc2_case(c, rg, in, ref, nullptr, fn::hc::up_arm::ring,
                              std::string("hc2 q8 rows ") + kind_name(kd) + " B" + std::to_string(B), &o, true) &&
                 run_head(c, rg, in, true, h);
            for (int32_t r = 0; r < B && ok; ++r) {
                for (int32_t k = 0; k < 2; ++k) {
                    const bool sub =
                        same_row(o.x[k], B, r, o1[r].x[k]) && same_row(o.inject[k], B, r, o1[r].inject[k]) &&
                        same_row(o.full[k], B, r, o1[r].full[k]) && same_row(o.codes[k], B, r, o1[r].codes[k]) &&
                        same_row(o.scales[k], B, r, o1[r].scales[k]);
                    const bool hd = same_row(h.x[k], B, r, h1[r].x[k]) && same_row(h.full[k], B, r, h1[r].full[k]) &&
                                    same_row(h.codes[k], B, r, h1[r].codes[k]) &&
                                    same_row(h.scales[k], B, r, h1[r].scales[k]);
                    n_same += (sub ? 1 : 0) + (hd ? 1 : 0);
                    n_cmp += 2;
                }
            }
        }
    }
    printf("hc-test hc2 q8 rows alone: every row of B = 2 .. %d against the same row alone at B = 1 (the sublayer's x, "
           "inject, F32 mixed, codes, scales; the head's x, mixed rows, codes, scales; both cards; plain and stress): "
           "%d of %d bitwise%s\n",
           fn::hc::max_rows, n_same, n_cmp, ok && n_same == n_cmp ? "" : "  FAIL");
    return ok && n_same == n_cmp;
}

// the head's down weights from a sublayer's (rows 0..319 of each stream: no injection rows), BF16 and Q8_0
void head_weights(const weights & w, int32_t r, std::vector<uint16_t> & bf, std::vector<uint8_t> & q8) {
    constexpr int32_t FB = F / 32;
    if (!w.w_down[r].empty()) {
        bf.resize(static_cast<size_t>(fn::hc::w_down_head_elems));
        for (int32_t s = 0; s < S; ++s) {
            std::copy_n(&w.w_down[r][static_cast<size_t>(s) * RD * F], static_cast<size_t>(L) * F,
                        &bf[static_cast<size_t>(s) * L * F]);
        }
    }
    q8.assign(static_cast<size_t>(fn::hc::w_down_head_q8_bytes), 0);
    const uint8_t * dsrc = w.q_down[r].data() + static_cast<size_t>(S) * RD * F;
    uint8_t * ddst = q8.data() + static_cast<size_t>(S) * L * F;
    for (int32_t s = 0; s < S; ++s) {
        std::copy_n(&w.q_down[r][static_cast<size_t>(s) * RD * F], static_cast<size_t>(L) * F,
                    &q8[static_cast<size_t>(s) * L * F]);
        std::copy_n(dsrc + static_cast<size_t>(s) * RD * FB * 2, static_cast<size_t>(L) * FB * 2,
                    ddst + static_cast<size_t>(s) * L * FB * 2);
    }
}

// DR1: the Q8_0 forms reject B = 0 and 6, null or misaligned Q8_0 weights (and accept a null BF16 pointer)
bool check_rejects_q8(card2 & k, fn::xchg::rings & rg) {
    using fn::xchg::msg;
    cudaSetDevice(k.dev);
    const fn::hc::down_args da = {nullptr, k.hres.p, k.hx.p, k.hpost.p, k.gamma.p, k.x.p, k.mom.p};
    const fn::hc::down2_q8_args d2 = {{da, rg.slot(1, msg::hc, 0), nullptr}, k.q_down.p};
    fn::hc::down2_q8_args d2_null = d2, d2_mis = d2;
    d2_null.w = nullptr;
    d2_mis.w = k.q_down.p + 8;
    const fn::hc::up2_args ua = {nullptr,
                                 k.mom.p,
                                 rg.slot(0, msg::hc, 0),
                                 k.x.p,
                                 k.gamma.p,
                                 k.inj.p,
                                 rg.slot(0, msg::hmix, 0),
                                 rg.slot(1, msg::hmix, 0),
                                 k.runs.p,
                                 rg.err(0),
                                 0};
    const fn::hc::up2_q8_args u2 = {ua, k.q_up.p};
    fn::hc::up2_q8_args u2_null = u2, u2_mis = u2, u2_card = u2;
    u2_null.w = nullptr;
    u2_mis.w = k.q_up.p + 2;
    u2_card.a.card = 2;
    const bool ok = !fn::hc::down2_q8(0, d2, k.st) && !fn::hc::down2_q8(6, d2, k.st) &&
                    !fn::hc::down2_q8(3, d2_null, k.st) && !fn::hc::down2_q8(3, d2_mis, k.st) &&
                    !fn::hc::head_down2_q8(6, d2, k.st) && !fn::hc::head_down2_q8(3, d2_mis, k.st) &&
                    !fn::hc::up2_q8(0, u2, k.st) && !fn::hc::up2_q8(6, u2, k.st) && !fn::hc::up2_q8(3, u2_null, k.st) &&
                    !fn::hc::up2_q8(3, u2_mis, k.st) && !fn::hc::up2_q8(3, u2_card, k.st) &&
                    !fn::hc::head_up2_q8(6, u2, k.st) && !fn::hc::head_up2_q8(3, u2_mis, k.st) &&
                    !fn::hc::down2_q8(3, d2, nullptr) && !fn::hc::up2_q8(3, u2, nullptr);
    printf("hc-test hc2 q8 rejects B = 0 and 6, null or misaligned Q8_0 weights, card 2 and a null stream: %s\n",
           ok ? "yes" : "NO  FAIL");
    return ok;
}

// DR1: the Q8_0 forms on both cards. pow2 weights (each also a BF16): bitwise against the BF16 HC1c kernels' two halves
// on the same weights, and FP64. General FP16 scales: FP64, every sublayer bitwise sublayer 0 and the cards equal; then
// batch invariance on them.
bool run_q8(card2 * c[2], fn::xchg::rings & rg, cudaStream_t st, int32_t dev0, uint32_t rows_mask) {
    bool ok = check_rejects_q8(*c[0], rg);
    for (const bool pow2 : {true, false}) {
        weights w;
        make_weights_q8(w, pow2);
        device_weights dw;
        ok = ok && cuda_ok(cudaSetDevice(dev0), "cudaSetDevice");
        if (pow2) {
            ok = ok && dw.w_down0.put(w.w_down[0], st) && dw.w_down1.put(w.w_down[1], st) &&
                 dw.w_up0.put(w.w_up[0], st) && dw.w_up1.put(w.w_up[1], st) && dw.gamma0.put(w.gamma_card[0], st) &&
                 dw.gamma1.put(w.gamma_card[1], st) && cuda_ok(cudaStreamSynchronize(st), "weights");
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            std::vector<uint16_t> hbf;
            std::vector<uint8_t> hq;
            head_weights(w, r, hbf, hq);
            ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") && c[r]->q_down.put(w.q_down[r], c[r]->st) &&
                 c[r]->q_up.put(w.q_up[r], c[r]->st) && c[r]->gamma.put(w.gamma_card[r], c[r]->st) &&
                 c[r]->hq_down.put(hq, c[r]->st) && (!pow2 || c[r]->hw_down.put(hbf, c[r]->st)) &&
                 (!pow2 || c[r]->w_up.put(w.w_up[r], c[r]->st)) && cuda_ok(cudaStreamSynchronize(c[r]->st), "weights");
        }
        for (const kind kd : {kind::plain, kind::stress, kind::zero_hx}) {
            for (int32_t rows = 1; rows <= fn::hc::max_rows && ok; ++rows) {
                if ((rows_mask >> (rows - 1) & 1u) == 0) {
                    continue;
                }
                inputs in;
                make_inputs(in, rows, kd);
                reference ref;
                make_reference(w, in, ref);
                gpu_out bits;
                const bool hc1 = pow2 && rows <= fn::hc::hc1_max_rows; // SP2: HC1c's bits within its rows
                ok = cuda_ok(cudaSetDevice(dev0), "cudaSetDevice") &&
                     (!hc1 || run_gpu(dw, in, bits, st, candidates[0]));
                const std::string label =
                    std::string(pow2 ? "hc2 q8 pow2 " : "hc2 q8 fp16 ") + kind_name(kd) + " B" + std::to_string(rows);
                ok = ok &&
                     run_hc2_case(c, rg, in, ref, hc1 ? &bits : nullptr, fn::hc::up_arm::ring, label, nullptr, true);
                ok = ok && head_q8_case(c, rg, in, ref, pow2,
                                        std::string(pow2 ? "hc2 q8 head pow2 " : "hc2 q8 head fp16 ") + kind_name(kd) +
                                            " B" + std::to_string(rows));
            }
        }
        if (ok && !pow2 && (rows_mask >> (fn::hc::max_rows - 1) & 1u) != 0) {
            ok = batch_invariance(c, rg, w, dw, st, dev0, true) && row_alone_q8(c, rg, w, dev0);
        }
    }
    return ok;
}

// ME12: the paired launch (hc::pair, head_pair) against the four-launch order it replaces on one device (hc::sub or
// head_sub, mode 1 on both cards, then mode 2 on both, one stream). Sublayer n of a sequence takes slot n % 3 and
// resets slot (n - 1) % 3 (sub_args_of); its x and inject go to entry n % n_sub.
fn::hc::sub_args pair_sub_args(card2 & k, fn::xchg::rings & rg, int32_t r, int32_t n, bool head) {
    fn::hc::sub_args a = sub_args_of(k, rg, r, n % n_sub, head);
    const fn::hc::sub_args at = sub_args_of(k, rg, r, n, head);
    a.hc_ring = at.hc_ring;
    a.hc_peer = at.hc_peer;
    a.hc_reset = at.hc_reset;
    a.hmix_reset = at.hmix_reset;
    a.hmix_own = at.hmix_own;
    a.hmix_peer = at.hmix_peer;
    return a;
}

// appends n words at device address p (current device)
bool append_words(const void * p, size_t n, std::vector<uint32_t> & out) {
    const size_t at = out.size();
    out.resize(at + n);
    return cuda_ok(cudaMemcpy(out.data() + at, p, n * sizeof(uint32_t), cudaMemcpyDeviceToHost), "snapshot");
}

// every word the paired launch is held to after sublayer n, both cards: all three X-HC slots (both regions of the
// written one, the reset one, the untouched one) and X-HMIX slots (F32 mixed, codes, scales), sublayer n's x and
// inject entries, and the error word
bool snapshot(card2 * c[2], fn::xchg::rings & rg, int32_t n, bool head, std::vector<uint32_t> & out) {
    using fn::xchg::msg;
    bool ok = cuda_ok(cudaStreamSynchronize(c[0]->st), "sublayer");
    for (int32_t r = 0; r < 2 && ok; ++r) {
        for (int32_t sl = 0; sl < fn::xchg::n_slot && ok; ++sl) {
            ok = append_words(rg.slot(r, msg::hc, sl), size_t(fn::xchg::slot_words(msg::hc)), out) &&
                 append_words(rg.slot(r, msg::hmix, sl), size_t(fn::xchg::slot_words(msg::hmix)), out);
        }
        const size_t i = static_cast<size_t>(n % n_sub);
        ok = ok && append_words(c[r]->x.p + i * card2::n_x, card2::n_x, out) &&
             (head || append_words(c[r]->inj.p + i * card2::n_j, card2::n_j, out));
        uint32_t e = ~0u;
        ok = ok && rg.read_err(r, e);
        out.push_back(e);
    }
    return ok;
}

// sublayer n of a sequence at rows, paired or in the four-launch order, on card 0's stream
bool pair_enqueue(card2 * c[2], fn::xchg::rings & rg, int32_t n, int32_t rows, bool head, bool paired,
                  float * scratch) {
    const fn::hc::pair_args pa = {{pair_sub_args(*c[0], rg, 0, n, head), pair_sub_args(*c[1], rg, 1, n, head)},
                                  scratch};
    if (paired) {
        return head ? fn::hc::head_pair(rows, pa, c[0]->st) : fn::hc::pair(rows, pa, c[0]->st);
    }
    bool ok = true;
    for (const int32_t mode : {1, 2}) {
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = head ? fn::hc::head_sub(rows, mode, pa.s[r], c[0]->st) : fn::hc::sub(rows, mode, pa.s[r], c[0]->st);
        }
    }
    return ok;
}

// fresh rings (every slot its sentinels, error words zero) and poisoned outputs and scratch
bool pair_fresh(card2 * c[2], fn::xchg::rings & rg, const dev_buf<float> & scratch) {
    bool ok = rg.reset_all() && cuda_ok(cudaDeviceSynchronize(), "rings reset") && scratch.poison(c[0]->st);
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = c[r]->x.poison(c[0]->st) && c[r]->inj.poison(c[0]->st);
    }
    return ok && cuda_ok(cudaStreamSynchronize(c[0]->st), "poison");
}

// the inputs on both cards' buffers, complete before card 0's stream runs
bool pair_inputs(card2 * c[2], const inputs & in) {
    return put_inputs(c, in) && cuda_ok(cudaStreamSynchronize(c[0]->st), "inputs") &&
           cuda_ok(cudaStreamSynchronize(c[1]->st), "inputs");
}

// one sequence at widths, a snapshot after every sublayer
bool pair_sequence(card2 * c[2], fn::xchg::rings & rg, const std::vector<int32_t> & widths, bool head, bool paired,
                   const dev_buf<float> & scratch, std::vector<uint32_t> & out) {
    out.clear();
    bool ok = pair_fresh(c, rg, scratch);
    for (int32_t n = 0; n < static_cast<int32_t>(widths.size()) && ok; ++n) {
        ok = pair_enqueue(c, rg, n, widths[n], head, paired, scratch.p) && snapshot(c, rg, n, head, out);
    }
    return ok;
}

// the words of two snapshots that differ (all of them if the sizes do)
size_t words_differ(const std::vector<uint32_t> & a, const std::vector<uint32_t> & b) {
    if (a.size() != b.size()) {
        return std::max(a.size(), b.size());
    }
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        bad += a[i] != b[i] ? 1 : 0;
    }
    return bad;
}

// a sequence paired and in the four-launch order: every snapshot word bitwise; the error words zero except with
// special inputs (there both orders' words are held equal instead)
bool pair_case(card2 * c[2], fn::xchg::rings & rg, const inputs & in, kind kd, const std::vector<int32_t> & widths,
               bool head, const dev_buf<float> & scratch, const std::string & label) {
    std::vector<uint32_t> want, got;
    bool ok = pair_inputs(c, in) && pair_sequence(c, rg, widths, head, false, scratch, want) &&
              pair_sequence(c, rg, widths, head, true, scratch, got);
    if (!ok) {
        printf("hc-test %-28s not run: %s  FAIL\n", label.c_str(), cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    const size_t bad = words_differ(want, got);
    // the last snapshot's error words: card 1's ends it, card 0's ends card 0's part (snapshot)
    using fn::xchg::msg;
    const size_t slots = size_t(fn::xchg::slot_words(msg::hc) + fn::xchg::slot_words(msg::hmix));
    const size_t card_words = size_t(fn::xchg::n_slot) * slots + card2::n_x + (head ? 0 : card2::n_j) + 1;
    const bool clean = kd == kind::special || (got.back() == 0u && got[got.size() - 1 - card_words] == 0u);
    printf("hc-test %-28s paired launch bitwise the four-launch order after each of %zu sublayers (X-HC and X-HMIX "
           "slots, x%s, error words; both cards; %zu words): %s%s\n",
           label.c_str(), widths.size(), head ? "" : ", inject", want.size(), bad == 0 ? "yes" : "NO  FAIL",
           clean ? "" : "; error words not zero  FAIL");
    return bad == 0 && clean;
}

// rejected before any launch: B = 0 and 6, a null stream, the cards swapped, a peer slot not the other card's own (X-HC
// and X-HMIX), both shards one card's slots, a reset of a written slot (either shard's current X-HC or X-HMIX slot,
// from either shard), a null error word, and a null or misaligned scratch; head_pair the same
bool check_rejects_pair(card2 * c[2], fn::xchg::rings & rg, float * scratch) {
    using fn::xchg::msg;
    const fn::hc::pair_args pa = {{pair_sub_args(*c[0], rg, 0, 1, false), pair_sub_args(*c[1], rg, 1, 1, false)},
                                  scratch};
    fn::hc::pair_args swap = pa, peer = pa, hpeer = pa, same = pa, reset = pa, hreset = pa, err = pa, mis = pa,
                      none = pa, x_reset01 = pa, x_reset10 = pa, x_hreset01 = pa, x_hreset10 = pa;
    swap.s[0] = pa.s[1];
    swap.s[1] = pa.s[0];
    peer.s[0].hc_peer = rg.slot(1, msg::hc, 2);
    hpeer.s[1].hmix_peer = rg.slot(0, msg::hmix, 2);
    same.s[1] = pa.s[0];
    same.s[1].card = 1;
    reset.s[1].hc_reset = pa.s[1].hc_ring;
    hreset.s[0].hmix_reset = pa.s[0].hmix_own;
    // the other shard's current slots, both directions
    x_reset01.s[0].hc_reset = pa.s[1].hc_ring;
    x_reset10.s[1].hc_reset = pa.s[0].hc_ring;
    x_hreset01.s[0].hmix_reset = pa.s[1].hmix_own;
    x_hreset10.s[1].hmix_reset = pa.s[0].hmix_own;
    err.s[1].err = nullptr;
    mis.scratch = scratch + 1;
    none.scratch = nullptr;
    const cudaStream_t st = c[0]->st;
    bool ok = true;
    for (const bool head : {false, true}) {
        const auto run = [&](int32_t rows, const fn::hc::pair_args & a, cudaStream_t s) {
            return head ? fn::hc::head_pair(rows, a, s) : fn::hc::pair(rows, a, s);
        };
        ok = ok && !run(0, pa, st) && !run(6, pa, st) && !run(3, pa, nullptr) && !run(3, swap, st) &&
             !run(3, peer, st) && !run(3, hpeer, st) && !run(3, same, st) && !run(3, reset, st) &&
             !run(3, hreset, st) && !run(3, x_reset01, st) && !run(3, x_reset10, st) && !run(3, x_hreset01, st) &&
             !run(3, x_hreset10, st) && !run(3, err, st) && !run(3, mis, st) && !run(3, none, st);
    }
    // nothing was enqueued: the stream is idle and no launch error is pending
    ok = ok && cudaStreamQuery(st) == cudaSuccess && cudaPeekAtLastError() == cudaSuccess;
    printf("hc-test me12 pair and head_pair reject B = 0 and 6, a null stream, swapped cards, a peer X-HC or X-HMIX "
           "slot not the other card's, one card's slots twice, a reset of either shard's written X-HC or X-HMIX slot "
           "(own and cross-shard, both directions), a null error word, a null or misaligned scratch, before any "
           "launch: %s\n",
           ok ? "yes" : "NO  FAIL");
    return ok;
}

// ME12: the paired launch in a captured graph, as the one-device verify graph holds it: n_graph_sub sublayers at
// max_rows, each one kernel node with the cooperative launch attribute. Replays (two on the first inputs, one on
// changed inputs) are bitwise the eager paired order on the same inputs; then a replay while another stream runs a
// full-device cooperative grid sequence (the four-launch order on a second ring set: one 640-thread block on every SM
// at a time) completes, both bitwise their eager results.
constexpr int32_t n_graph_sub = 3;

bool final_state(card2 * c[2], fn::xchg::rings & rg, std::vector<uint32_t> & out) {
    out.clear();
    bool ok = true;
    for (int32_t n = 0; n < n_graph_sub && ok; ++n) {
        ok = snapshot(c, rg, n, false, out);
    }
    return ok;
}

bool pair_graph(card2 * c[2], fn::xchg::rings & rg, const dev_buf<float> & scratch, card2 * o[2],
                fn::xchg::rings & rg2, const dev_buf<float> & scratch2) {
    constexpr int32_t B = fn::hc::max_rows;
    inputs in_a, in_b;
    make_inputs(in_a, B, kind::plain);
    make_inputs(in_b, B, kind::stress);
    const std::vector<int32_t> widths(n_graph_sub, B);
    const auto eager = [&](const inputs & in, std::vector<uint32_t> & st) {
        std::vector<uint32_t> each;
        return pair_inputs(c, in) && pair_sequence(c, rg, widths, false, true, scratch, each) && final_state(c, rg, st);
    };
    std::vector<uint32_t> want_a, want_b;
    bool ok = eager(in_a, want_a) && eager(in_b, want_b);
    // capture
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    ok = ok && cuda_ok(cudaStreamBeginCapture(c[0]->st, cudaStreamCaptureModeThreadLocal), "begin capture");
    for (int32_t n = 0; n < n_graph_sub && ok; ++n) {
        ok = pair_enqueue(c, rg, n, B, false, true, scratch.p);
    }
    ok = cuda_ok(cudaStreamEndCapture(c[0]->st, &g), "end capture") && ok;
    size_t n_nodes = 0;
    ok = ok && cuda_ok(cudaGraphGetNodes(g, nullptr, &n_nodes), "graph nodes");
    std::vector<cudaGraphNode_t> nodes(n_nodes);
    ok = ok && cuda_ok(cudaGraphGetNodes(g, nodes.data(), &n_nodes), "graph nodes");
    int32_t n_coop = 0;
    for (size_t i = 0; i < n_nodes && ok; ++i) {
        cudaGraphNodeType t = cudaGraphNodeTypeEmpty;
        cudaKernelNodeAttrValue v = {};
        ok = cuda_ok(cudaGraphNodeGetType(nodes[i], &t), "node type");
        if (ok && t == cudaGraphNodeTypeKernel) {
            ok = cuda_ok(cudaGraphKernelNodeGetAttribute(nodes[i], cudaLaunchAttributeCooperative, &v), "node attr");
            n_coop += v.cooperative != 0 ? 1 : 0;
        }
    }
    const bool shape = ok && n_nodes == size_t(n_graph_sub) && n_coop == n_graph_sub;
    printf("hc-test me12 graph: %d paired sublayers captured as %zu nodes, %d of them cooperative kernel nodes: %s\n",
           n_graph_sub, n_nodes, n_coop, shape ? "yes" : "NO  FAIL");
    ok = ok && cuda_ok(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    const auto replay = [&](const inputs & in, std::vector<uint32_t> & st) {
        return pair_inputs(c, in) && pair_fresh(c, rg, scratch) && cuda_ok(cudaGraphLaunch(ge, c[0]->st), "launch") &&
               final_state(c, rg, st);
    };
    std::vector<uint32_t> got_a1, got_a2, got_b;
    ok = ok && replay(in_a, got_a1) && replay(in_a, got_a2) && replay(in_b, got_b);
    const size_t d_a1 = words_differ(want_a, got_a1), d_a2 = words_differ(want_a, got_a2);
    const size_t d_b = words_differ(want_b, got_b), d_ab = words_differ(want_a, want_b);
    const bool replays = ok && d_a1 == 0 && d_a2 == 0 && d_b == 0 && d_ab > 0;
    printf("hc-test me12 graph replays bitwise the eager paired order (inputs A twice, then changed inputs B; A and B "
           "differ in %zu of %zu words): %s\n",
           d_ab, want_a.size(), replays ? "yes" : "NO  FAIL");
    // co-residency: the other stream's full-device cooperative grids are enqueued first, the replay behind them on
    // its own stream; the replay's grids wait for SMs and both finish within the host's bound
    constexpr int32_t n_other = 12;
    const std::vector<int32_t> other_widths(n_other, B);
    std::vector<uint32_t> other_each;
    ok = ok && pair_inputs(o, in_b) && pair_sequence(o, rg2, other_widths, false, false, scratch2, other_each);
    // its last snapshot: the state the concurrent run must end in
    const std::vector<uint32_t> other_want(other_each.end() - static_cast<int64_t>(other_each.size() / n_other),
                                           other_each.end());
    std::vector<uint32_t> got_c, other_got;
    ok = ok && pair_inputs(c, in_a) && pair_fresh(c, rg, scratch) && pair_fresh(o, rg2, scratch2);
    for (int32_t n = 0; n < n_other && ok; ++n) {
        ok = pair_enqueue(o, rg2, n, B, false, false, scratch2.p);
    }
    ok = ok && cuda_ok(cudaGraphLaunch(ge, c[0]->st), "launch");
    card2 * both[2] = {c[0], o[0]};
    // the host's bound: four cases' (the other stream's launches are twice a case's)
    ok = ok && wait_two(both, 4 * hc2_timeout_s) && final_state(c, rg, got_c) &&
         snapshot(o, rg2, n_other - 1, false, other_got);
    const size_t d_c = words_differ(want_a, got_c), d_o = words_differ(other_want, other_got);
    const bool co = ok && d_c == 0 && d_o == 0;
    printf("hc-test me12 graph replay behind %d full-device cooperative grids on another stream: completes, bitwise "
           "(replay %zu words differ, other stream %zu): %s\n",
           4 * n_other, d_c, d_o, co ? "yes" : "NO  FAIL");
    if (ge != nullptr) {
        cudaGraphExecDestroy(ge);
    }
    if (g != nullptr) {
        cudaGraphDestroy(g);
    }
    return ok && shape && replays && co;
}

// ME12 on one device (loopback): the static and residency checks (pair_prepare), rejects, every B and both modes for
// each input kind with fixed widths, a sequence of changing widths, and the graph checks
bool run_pair(card2 * c[2], fn::xchg::rings & rg, const weights & w, uint32_t rows_mask) {
    dev_buf<float> scratch(size_t(fn::hc::pair_scratch_bytes) / sizeof(float));
    bool ok = cuda_ok(cudaSetDevice(c[0]->dev), "cudaSetDevice") && scratch.p != nullptr;
    const bool prep = ok && fn::hc::pair_prepare();
    printf("hc-test me12 pair_prepare (cooperative launch, every B and mode co-resident at 72 x 640 by the occupancy "
           "API, no local memory): %s\n",
           prep ? "yes" : "NO  FAIL");
    ok = ok && prep && check_rejects_pair(c, rg, scratch.p);
    // the head's down weights: rows 0..319 of each stream of the sublayer's (head_weights' BF16 part)
    for (int32_t r = 0; r < 2 && ok; ++r) {
        std::vector<uint16_t> hbf(static_cast<size_t>(fn::hc::w_down_head_elems));
        for (int32_t s = 0; s < S; ++s) {
            std::copy_n(&w.w_down[r][static_cast<size_t>(s) * RD * F], static_cast<size_t>(L) * F,
                        &hbf[static_cast<size_t>(s) * L * F]);
        }
        ok = c[r]->hw_down.put(hbf, c[0]->st) && cuda_ok(cudaStreamSynchronize(c[0]->st), "head weights");
    }
    for (const kind kd : {kind::plain, kind::stress, kind::zero_hx, kind::special}) {
        for (int32_t rows = 1; rows <= fn::hc::max_rows && ok; ++rows) {
            if ((rows_mask >> (rows - 1) & 1u) == 0) {
                continue;
            }
            inputs in;
            make_inputs(in, rows, kd);
            for (const bool head : {false, true}) {
                const std::string label = std::string(head ? "me12 head " : "me12 pair ") + kind_name(kd) + " B" +
                                          std::to_string(rows);
                ok = pair_case(c, rg, in, kd, std::vector<int32_t>(n_sub, rows), head, scratch, label) && ok;
            }
        }
    }
    // changing widths over twelve sublayers (each slot four times): a wider sublayer's rows in a slot before a
    // narrower one's, so a row the narrower one does not write must stay unlanded (its reset words)
    const std::vector<int32_t> widths = {5, 3, 1, 4, 2, 5, 1, 5, 2, 4, 3, 1};
    for (const kind kd : {kind::plain, kind::special}) {
        inputs in;
        make_inputs(in, fn::hc::max_rows, kd);
        for (const bool head : {false, true}) {
            const std::string label =
                std::string(head ? "me12 head " : "me12 pair ") + kind_name(kd) + " widths 5..1";
            ok = ok && pair_case(c, rg, in, kd, widths, head, scratch, label);
        }
    }
    // the graph checks, with a second card pair and ring set on the same device for the other stream
    if (ok) {
        card2 o0(c[0]->dev), o1(c[0]->dev);
        card2 * o[2] = {&o0, &o1};
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = o[r]->st != nullptr && o[r]->w_down.put(w.w_down[r], o[0]->st) &&
                 o[r]->w_up.put(w.w_up[r], o[0]->st) && o[r]->gamma.put(w.gamma_card[r], o[0]->st) &&
                 cuda_ok(cudaStreamSynchronize(o[0]->st), "weights");
        }
        fn::xchg::rings rg2(c[0]->dev, c[0]->dev);
        dev_buf<float> scratch2(size_t(fn::hc::pair_scratch_bytes) / sizeof(float));
        ok = ok && rg2.ok() && scratch2.p != nullptr && pair_graph(c, rg, scratch, o, rg2, scratch2);
        cudaSetDevice(c[0]->dev);
    }
    return ok;
}

// FMU2: the joint sublayer (hc::sub_joint, head_sub_joint) against each lane's own launches. Lanes of max_rows rows,
// each its own inputs, outputs and ring set (as the joint round's lane rings), all on lane 0's weights. A sequence of
// sublayers (slot n % 3, the previous one's slots reset: sub_args_of) runs once per lane alone (hc::sub or head_sub on
// both cards) and twice jointly (one joint launch a card a sublayer); after every sublayer each lane's snapshot (every
// X-HC and X-HMIX slot, x and inject, the error words; both cards) is held bitwise its lone run's, and the second joint
// run bitwise the first. In loopback (DEV0 == DEV1) every launch goes on one stream, modes 1 then 2.
constexpr int32_t n_joint_sub = 5, n_joint_head = 2;

struct jlane {
    card2 k0, k1;
    card2 * c[2] = {&k0, &k1};
    fn::xchg::rings rg;

    jlane(int32_t d0, int32_t d1) : k0(d0), k1(d1), rg(d0, d1) {}
};

using jlanes = std::vector<std::unique_ptr<jlane>>;

// lane l's arguments for card r's sublayer n, on the weights of w (lane 0's card r)
fn::hc::sub_args joint_lane_args(jlane & l, const card2 & w, int32_t r, int32_t n, bool head) {
    fn::hc::sub_args a = sub_args_of(*l.c[r], l.rg, r, n, head);
    a.w_down = head ? w.hw_down.p : w.w_down.p;
    a.w_up = w.w_up.p;
    a.gamma = w.gamma.p;
    return a;
}

bool devices_idle(const jlanes & ln) {
    bool ok = true;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(ln[0]->c[r]->dev), "cudaSetDevice") && cuda_ok(cudaDeviceSynchronize(), "idle");
    }
    return ok;
}

// every lane's rings fresh (sentinels, error words zero), its x and inject poisoned
bool joint_fresh(const jlanes & ln) {
    bool ok = true;
    for (size_t k = 0; k < ln.size() && ok; ++k) {
        ok = ln[k]->rg.reset_all();
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card2 & c = *ln[k]->c[r];
            ok = cuda_ok(cudaSetDevice(c.dev), "cudaSetDevice") && c.x.poison(c.st) && c.inj.poison(c.st);
        }
    }
    return ok && devices_idle(ln);
}

// lane k's sequence alone at rows (FMS13: R, 2 .. 5), a snapshot after every sublayer
bool joint_lone(const jlanes & ln, size_t k, bool head, int32_t subs, int32_t rows, std::vector<uint32_t> & out) {
    jlane & l = *ln[k];
    const bool loopback = l.c[0]->dev == l.c[1]->dev;
    out.clear();
    bool ok = joint_fresh(ln);
    for (int32_t n = 0; n < subs && ok; ++n) {
        for (const int32_t mode : {loopback ? 1 : 3, loopback ? 2 : 0}) {
            for (int32_t r = 0; r < 2 && ok && mode != 0; ++r) {
                const fn::hc::sub_args a = joint_lane_args(l, *ln[0]->c[r], r, n, head);
                ok                       = cuda_ok(cudaSetDevice(l.c[r]->dev), "cudaSetDevice") &&
                                           (head ? fn::hc::head_sub(rows, mode, a, launch_stream(l.c, r)) :
                                                   fn::hc::sub(rows, mode, a, launch_stream(l.c, r)));
            }
        }
        ok = ok && wait_two(l.c, hc2_timeout_s) && snapshot(l.c, l.rg, n, head, out);
    }
    return ok;
}

// the lanes' sequence jointly at rows a lane (lanes' launches on lane 0's streams), each lane's snapshots
bool joint_run(const jlanes & ln, int32_t lanes, bool head, int32_t subs, int32_t rows, std::vector<uint32_t> * out) {
    const bool loopback = ln[0]->c[0]->dev == ln[0]->c[1]->dev;
    for (int32_t k = 0; k < lanes; ++k) {
        out[k].clear();
    }
    bool ok = joint_fresh(ln);
    for (int32_t n = 0; n < subs && ok; ++n) {
        for (const int32_t mode : {loopback ? 1 : 3, loopback ? 2 : 0}) {
            for (int32_t r = 0; r < 2 && ok && mode != 0; ++r) {
                fn::hc::sub_args a[fn::xchg::max_lanes] = {};
                for (int32_t k = 0; k < lanes; ++k) {
                    a[k] = joint_lane_args(*ln[k], *ln[0]->c[r], r, n, head);
                }
                ok = cuda_ok(cudaSetDevice(ln[0]->c[r]->dev), "cudaSetDevice") &&
                     (head ? fn::hc::head_sub_joint(lanes, rows, mode, a, launch_stream(ln[0]->c, r)) :
                             fn::hc::sub_joint(lanes, rows, mode, a, launch_stream(ln[0]->c, r)));
            }
        }
        ok = ok && wait_two(ln[0]->c, hc2_timeout_s);
        for (int32_t k = 0; k < lanes && ok; ++k) {
            ok = snapshot(ln[k]->c, ln[k]->rg, n, head, out[k]);
        }
    }
    return ok;
}

// FMS2 item 5b: a joint run's snapshots with each X-HC region (peer and own, every slot, both cards) moved from the
// joint layout (xchg.h hcj_run_at) to the lone one (rows of words_hc words, or the head's 1,284), every word to its
// lone place and the rest the sentinel; a short head block's unused words must be zero or the sentinel (bad counts
// the others)
void hcj_to_lone(std::vector<uint32_t> & v, bool head, size_t card_words, size_t & bad) {
    using fn::xchg::msg;
    constexpr int32_t region = fn::xchg::hc_own_at;
    const int32_t stride = head ? 1284 : fn::xchg::words_hc;
    const size_t slot = size_t(fn::xchg::slot_words(msg::hc) + fn::xchg::slot_words(msg::hmix));
    std::vector<uint32_t> lone(static_cast<size_t>(region));
    for (size_t c0 = 0; c0 + card_words <= v.size(); c0 += card_words) {
        for (int32_t sl = 0; sl < fn::xchg::n_slot; ++sl) {
            for (int32_t half = 0; half < 2; ++half) {
                uint32_t * w = v.data() + c0 + size_t(sl) * slot + size_t(half) * region;
                std::fill(lone.begin(), lone.end(), fn::xchg::sentinel);
                for (int32_t jw = 0; jw < region; ++jw) {
                    int32_t at = -1;
                    if (jw < fn::xchg::hcj_v_words) {
                        at = (jw / 4) * stride + jw % 4;
                    } else {
                        const int32_t vb = (jw - fn::xchg::hcj_v_words) / fn::xchg::hcj_block_words;
                        const int32_t rem = (jw - fn::xchg::hcj_v_words) % fn::xchg::hcj_block_words;
                        const int32_t b = rem / fn::xchg::hcj_rows_block, r = rem % fn::xchg::hcj_rows_block;
                        const int32_t s = vb % 4, j = (vb / 4) * fn::xchg::hcj_rows_block + r;
                        if (!head || j < 320) {
                            at = b * stride + (j < 320 ? 4 + 320 * s + j : 1284 + 4 * s + (j - 320));
                        }
                    }
                    if (at >= 0) {
                        lone[size_t(at)] = w[jw];
                    } else if (w[jw] != 0u && w[jw] != fn::xchg::sentinel) {
                        ++bad;
                    }
                }
                std::copy(lone.begin(), lone.end(), w);
            }
        }
    }
}

// FMS13's sentinel test: the words of rows rows .. 4 in a lone-layout snapshot (hcj_to_lone's) that are no longer
// their initial values: every slot's X-HC regions (the sentinel) and X-HMIX regions (the sentinels), x and inject (the
// poison)
size_t tail_touched(const std::vector<uint32_t> & v, bool head, size_t card_words, int32_t rows) {
    using fn::xchg::msg;
    using hl                   = fn::xchg::hmix_layout;
    constexpr int32_t region   = fn::xchg::hc_own_at;
    const int32_t     stride   = head ? 1284 : fn::xchg::words_hc;
    const size_t      hc_words = size_t(fn::xchg::slot_words(msg::hc)), slot = hc_words + size_t(hl::slot_words);
    size_t            bad = 0;
    const auto        run = [&](size_t at, size_t n, uint32_t want) {
        for (size_t i = 0; i < n && at + i < v.size(); ++i) {
            bad += v[at + i] != want ? 1 : 0;
        }
    };
    for (size_t c0 = 0; c0 + card_words <= v.size(); c0 += card_words) {
        for (int32_t b = rows; b < fn::hc::max_rows; ++b) {
            for (int32_t sl = 0; sl < fn::xchg::n_slot; ++sl) {
                const size_t s0 = c0 + size_t(sl) * slot;
                for (int32_t half = 0; half < 2; ++half) {
                    run(s0 + size_t(half) * region + size_t(b) * stride, size_t(stride), fn::xchg::sentinel);
                }
                const size_t m0 = s0 + hc_words;
                run(m0 + size_t(b) * hl::width, size_t(hl::width), fn::xchg::sentinel);
                run(m0 + size_t(hl::code_at) + size_t(b) * hl::width / 4, size_t(hl::width / 4),
                    fn::xchg::code_sentinel);
                run(m0 + size_t(hl::scale_at) + size_t(b) * hl::runs * 2, size_t(hl::runs * 2), fn::xchg::sentinel);
            }
            const size_t x0 = c0 + size_t(fn::xchg::n_slot) * slot;
            run(x0 + size_t(b) * S * F, size_t(S) * F, ~0u);
            if (!head) {
                run(x0 + card2::n_x + size_t(b) * J, size_t(J), ~0u);
            }
        }
    }
    return bad;
}

// U lanes at rows a lane (FMS13: 2 .. 5), a sequence each way: every lane's snapshots bitwise its lone run's at B =
// rows, the second joint run bitwise the first, rows .. 4 untouched; the error words zero except a lane of special
// inputs (held equal instead)
bool joint_case(const jlanes & ln, int32_t lanes, const kind * kinds, bool head, int32_t rows) {
    const int32_t subs = head ? n_joint_head : n_joint_sub;
    const std::string label =
        std::string(head ? "hcj head" : "hcj sub") + " R" + std::to_string(rows) + " U" + std::to_string(lanes);
    std::vector<uint32_t> want[fn::xchg::max_lanes], got[fn::xchg::max_lanes], again[fn::xchg::max_lanes];
    bool ok = true;
    for (int32_t k = 0; k < lanes && ok; ++k) {
        ok = joint_lone(ln, size_t(k), head, subs, rows, want[k]);
    }
    ok = ok && joint_run(ln, lanes, head, subs, rows, got) && joint_run(ln, lanes, head, subs, rows, again);
    if (!ok) {
        printf("hc-test %-14s not run: %s  FAIL\n", label.c_str(), cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    using fn::xchg::msg;
    const size_t slots = size_t(fn::xchg::slot_words(msg::hc) + fn::xchg::slot_words(msg::hmix));
    const size_t card_words = size_t(fn::xchg::n_slot) * slots + card2::n_x + (head ? 0 : card2::n_j) + 1;
    size_t       bad = 0, bad2 = 0, words = 0, gap = 0, touched = 0;
    bool clean = true;
    for (int32_t k = 0; k < lanes; ++k) {
        hcj_to_lone(got[k], head, card_words, gap);
        hcj_to_lone(again[k], head, card_words, gap);
        touched += tail_touched(got[k], head, card_words, rows);
        bad += words_differ(want[k], got[k]);
        bad2 += words_differ(got[k], again[k]);
        words += want[k].size();
        clean = clean &&
                (kinds[k] == kind::special || (got[k].back() == 0u && got[k][got[k].size() - 1 - card_words] == 0u));
    }
    printf(
        "hc-test %-14s each lane bitwise its lone launches after each of %d sublayers (X-HC and X-HMIX slots, x%s, "
        "error words; both cards; %zu words; X-HC from the joint layout, %zu unused words not zero%s): %s; second "
        "joint run: %s%s; rows %d .. 4 words touched %zu%s\n",
        label.c_str(), subs, head ? "" : ", inject", words, gap, gap == 0 ? "" : "  FAIL",
        bad == 0 ? "yes" : "NO  FAIL", bad2 == 0 ? "bitwise" : "DIFFERS  FAIL",
        clean ? "" : "; error words not zero  FAIL", rows, touched, touched == 0 ? "" : "  FAIL");
    return bad == 0 && bad2 == 0 && clean && gap == 0 && touched == 0;
}

// refused with nothing enqueued (every lane's X-HMIX slots still the sentinels): lanes 1 and 5, mode 0, two lanes on
// one ring, lanes on different cards, a lane's reset of a slot another lane writes, a lane on other weights (down, up
// or gamma: the launch streams lane 0's), the head form's one lane, and (FMS13) rows 1 and 6
bool joint_rejects(const jlanes & ln) {
    const card2 & w = *ln[0]->c[0];
    fn::hc::sub_args a[fn::xchg::max_lanes] = {};
    for (size_t k = 0; k < ln.size(); ++k) {
        a[k] = joint_lane_args(*ln[k], w, 0, 1, false);
    }
    cudaStream_t st = ln[0]->c[0]->st;
    constexpr int32_t R5      = fn::hc::max_rows;
    bool ok = joint_fresh(ln) && cuda_ok(cudaSetDevice(w.dev), "cudaSetDevice");
    int32_t refused = 0, cases = 0;
    const auto refuse = [&](bool launched) {
        ++cases;
        refused += launched ? 0 : 1;
    };
    refuse(fn::hc::sub_joint(1, R5, 3, a, st));
    refuse(fn::hc::sub_joint(5, R5, 3, a, st));
    refuse(fn::hc::sub_joint(2, R5, 0, a, st));
    fn::hc::sub_args b[fn::xchg::max_lanes];
    std::copy_n(a, fn::xchg::max_lanes, b);
    b[1] = joint_lane_args(*ln[0], w, 0, 1, false); // lane 1 on lane 0's ring
    b[1].x = a[1].x;
    b[1].inject = a[1].inject;
    refuse(fn::hc::sub_joint(2, R5, 3, b, st));
    std::copy_n(a, fn::xchg::max_lanes, b);
    b[1] = joint_lane_args(*ln[1], *ln[0]->c[1], 1, 1, false); // lane 1 card 1's
    refuse(fn::hc::sub_joint(2, R5, 3, b, st));
    std::copy_n(a, fn::xchg::max_lanes, b);
    b[2].hc_reset = b[0].hc_ring; // lane 2 resets lane 0's X-HC slot
    refuse(fn::hc::sub_joint(3, R5, 3, b, st));
    std::copy_n(a, fn::xchg::max_lanes, b);
    b[3].hmix_reset = b[1].hmix_own; // lane 3 resets lane 1's X-HMIX slot
    refuse(fn::hc::sub_joint(4, R5, 3, b, st));
    for (int32_t which = 0; which < 3; ++which) { // a lane on other weights (16-byte aligned, valid addresses)
        std::copy_n(a, fn::xchg::max_lanes, b);
        if (which == 0) {
            b[1].w_down = a[0].w_down + 8;
        } else if (which == 1) {
            b[1].w_up = a[0].w_up + 8;
        } else {
            b[1].gamma = a[0].gamma + 4;
        }
        refuse(fn::hc::sub_joint(2, R5, 3, b, st));
    }
    refuse(fn::hc::head_sub_joint(1, R5, 3, a, st));
    refuse(fn::hc::sub_joint(2, 1, 3, a, st));
    refuse(fn::hc::sub_joint(2, R5 + 1, 3, a, st));
    refuse(fn::hc::head_sub_joint(2, 1, 3, a, st));
    ok = ok && cuda_ok(cudaStreamSynchronize(st), "refused") && cuda_ok(cudaGetLastError(), "refused");
    // nothing reached the rings: every lane's slots still the sentinels
    int64_t touched = 0;
    for (size_t k = 0; k < ln.size() && ok; ++k) {
        for (int32_t sl = 0; sl < fn::xchg::n_slot && ok; ++sl) {
            int64_t bad = 0;
            ok = hmix_slot_reset(ln[k]->rg, sl, bad);
            touched += bad;
        }
    }
    printf("hc-test hcj rejects: %d of %d refused, rings untouched: %s\n", refused, cases,
           ok && touched == 0 ? "yes" : "NO  FAIL");
    return ok && refused == cases && touched == 0;
}

} // namespace

int fnfl_hc_test(int device, uint32_t rows_mask) {
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaStream_t st = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate")) {
        return 1;
    }
    bool ok = true;
    {
        weights w;
        make_weights(w);
        device_weights dw;
        ok = dw.w_down0.put(w.w_down[0], st) && dw.w_down1.put(w.w_down[1], st) && dw.w_up0.put(w.w_up[0], st) &&
             dw.w_up1.put(w.w_up[1], st) && dw.gamma0.put(w.gamma_card[0], st) && dw.gamma1.put(w.gamma_card[1], st);
        ok = ok && check_rejects(dw, st);
        bool cuda_failed = !ok;
        for (const kind k : {kind::plain, kind::stress, kind::zero_hx}) {
            for (int32_t rows = 1; rows <= fn::hc::hc1_max_rows && !cuda_failed; ++rows) {
                if ((rows_mask >> (rows - 1) & 1u) == 0) {
                    continue;
                }
                inputs in;
                make_inputs(in, rows, k);
                reference ref;
                make_reference(w, in, ref);
                for (const candidate & cand : candidates) {
                    gpu_out first, second;
                    const std::string label =
                        std::string(cand.name) + " " + std::string(kind_name(k)) + " B" + std::to_string(rows);
                    // the plain case also replays the sequence 64 times in a graph (HC1b)
                    const int32_t replays = k == kind::plain ? 64 : 0;
                    int32_t diffs = 0;
                    if (!run_gpu(dw, in, first, st, cand) || !run_gpu(dw, in, second, st, cand, replays, &diffs)) {
                        ok = false;
                        cuda_failed = true;
                        break;
                    }
                    ok = compare(in, ref, first, label.c_str()) && ok;
                    const bool same = same_bits(first, second);
                    printf("hc-test %-19s second run bitwise identical: %s\n", label.c_str(),
                           same ? "yes" : "NO  FAIL");
                    ok = same && ok;
                    if (replays > 0) {
                        printf("hc-test %-19s %d graph replays bitwise identical to the first run: %s\n", label.c_str(),
                               replays, diffs == 0 ? "yes" : "NO  FAIL");
                        ok = diffs == 0 && ok;
                    }
                }
            }
        }
        ok = cuda_ok(cudaStreamSynchronize(st), "final sync") && ok;
    }
    cudaStreamDestroy(st);
    printf("hc-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int fnfl_hc2_test(int dev0, int dev1, uint32_t rows_mask) {
    // loopback (dev0 == dev1, for Compute Sanitizer): both cards on one device, one stream, sends before reads
    if (dev0 == dev1) {
        printf("hc2-test: loopback on device %d: both cards' kernels on one stream, sends before reads\n", dev0);
    } else if (!fn::xchg::enable_peer_access(dev0, dev1)) {
        printf("hc2-test: FAIL (no peer access)\n");
        return 1;
    }
    bool ok = true;
    {
        weights w;
        make_weights(w);
        // the reference run (HC1c's kernels, both halves) on dev0
        ok = cuda_ok(cudaSetDevice(dev0), "cudaSetDevice");
        cudaStream_t st = nullptr;
        ok = ok && cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
        device_weights dw;
        ok = ok && dw.w_down0.put(w.w_down[0], st) && dw.w_down1.put(w.w_down[1], st) && dw.w_up0.put(w.w_up[0], st) &&
             dw.w_up1.put(w.w_up[1], st) && dw.gamma0.put(w.gamma_card[0], st) && dw.gamma1.put(w.gamma_card[1], st);
        // the pair: card r on device dev[r]
        card2 k0(dev0), k1(dev1);
        card2 * c[2] = {&k0, &k1};
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") && c[r]->st != nullptr && fn::hc::prepare2() &&
                 c[r]->w_down.put(w.w_down[r], c[r]->st) && c[r]->w_up.put(w.w_up[r], c[r]->st) &&
                 c[r]->gamma.put(w.gamma_card[r], c[r]->st) && cuda_ok(cudaStreamSynchronize(c[r]->st), "weights");
        }
        fn::xchg::rings rg(dev0, dev1);
        ok = ok && rg.ok() && check_rejects2(k0, rg);
        for (const kind kd : {kind::plain, kind::stress, kind::zero_hx}) {
            for (int32_t rows = 1; rows <= fn::hc::max_rows && ok; ++rows) {
                if ((rows_mask >> (rows - 1) & 1u) == 0) {
                    continue;
                }
                inputs in;
                make_inputs(in, rows, kd);
                reference ref;
                make_reference(w, in, ref);
                gpu_out bits;
                const bool hc1 = rows <= fn::hc::hc1_max_rows; // SP2: HC1c's bits within its rows
                ok = cuda_ok(cudaSetDevice(dev0), "cudaSetDevice") &&
                     (!hc1 || run_gpu(dw, in, bits, st, candidates[0]));
                for (const fn::hc::up_arm arm : {fn::hc::up_arm::ring, fn::hc::up_arm::staged}) {
                    if (!ok) {
                        break;
                    }
                    const std::string label = std::string(arm == fn::hc::up_arm::ring ? "hc2 arm0 " : "hc2 armA ") +
                                              kind_name(kd) + " B" + std::to_string(rows);
                    ok = run_hc2_case(c, rg, in, ref, hc1 ? &bits : nullptr, arm, label) && ok;
                }
                // HC4: the one-launch sublayer (hc::sub), against the same bits
                ok = ok && run_hc2_case(c, rg, in, ref, hc1 ? &bits : nullptr, fn::hc::up_arm::ring,
                                        std::string("hc2 fused ") + kind_name(kd) + " B" + std::to_string(rows),
                                        nullptr, false, true);
            }
        }
        if (ok && (rows_mask >> (fn::hc::max_rows - 1) & 1u) != 0) {
            ok = batch_invariance(c, rg, w, dw, st, dev0);
        }
        ok = ok && run_q8(c, rg, st, dev0, rows_mask);
        // ME12: the paired launch is one device's (dev0 == dev1); two devices keep hc::sub, checked above
        if (dev0 == dev1) {
            // run_q8 left the cards' weights its own: the BF16 sublayer's again
            for (int32_t r = 0; r < 2 && ok; ++r) {
                ok = c[r]->w_down.put(w.w_down[r], c[r]->st) && c[r]->w_up.put(w.w_up[r], c[r]->st) &&
                     c[r]->gamma.put(w.gamma_card[r], c[r]->st) && cuda_ok(cudaStreamSynchronize(c[r]->st), "weights");
            }
            ok = ok && run_pair(c, rg, w, rows_mask);
        } else {
            printf("hc2-test: two devices: the paired launch (ME12) is one device's; hc::sub cases above\n");
        }
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(c[r]->dev);
            ok = cuda_ok(cudaDeviceSynchronize(), "final sync") && ok;
        }
        cudaSetDevice(dev0);
        if (st != nullptr) {
            cudaStreamDestroy(st);
        }
    }
    printf("hc2-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int fnfl_hcj_test(int dev0, int dev1) {
    if (dev0 == dev1) {
        printf("hcj-test: loopback on device %d: both cards' launches on one stream, modes 1 then 2\n", dev0);
    } else if (!fn::xchg::enable_peer_access(dev0, dev1)) {
        printf("hcj-test: FAIL (no peer access)\n");
        return 1;
    }
    bool ok = true;
    {
        weights w;
        make_weights(w);
        jlanes ln;
        for (int32_t k = 0; k < fn::xchg::max_lanes; ++k) {
            ln.push_back(std::make_unique<jlane>(dev0, dev1));
            ok = ok && ln.back()->rg.ok() && ln.back()->k0.st != nullptr && ln.back()->k1.st != nullptr;
        }
        // lane 0's cards hold the weights every lane reads: the sublayer's and the head's down (no injection rows)
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card2 & c = *ln[0]->c[r];
            std::vector<uint16_t> hbf(static_cast<size_t>(fn::hc::w_down_head_elems));
            for (int32_t s = 0; s < S; ++s) {
                std::copy_n(&w.w_down[r][static_cast<size_t>(s) * RD * F], static_cast<size_t>(L) * F,
                            &hbf[static_cast<size_t>(s) * L * F]);
            }
            ok = cuda_ok(cudaSetDevice(c.dev), "cudaSetDevice") && fn::hc::prepare2() &&
                 c.w_down.put(w.w_down[r], c.st) && c.w_up.put(w.w_up[r], c.st) && c.gamma.put(w.gamma_card[r], c.st) &&
                 c.hw_down.put(hbf, c.st) && cuda_ok(cudaStreamSynchronize(c.st), "weights");
        }
        // each lane its own inputs, at R rows (FMS13: 2 .. 5; rows R .. 4 NaN); the rejects at five rows
        const kind kinds[fn::xchg::max_lanes] = {kind::plain, kind::stress, kind::zero_hx, kind::special};
        const auto lanes_inputs               = [&](int32_t rows) {
            bool good = true;
            for (int32_t k = 0; k < fn::xchg::max_lanes && good; ++k) {
                inputs in;
                make_inputs(in, rows, kinds[k]);
                good = put_inputs(ln[size_t(k)]->c, in, std::numeric_limits<float>::quiet_NaN()) && devices_idle(ln);
            }
            return good;
        };
        ok = ok && lanes_inputs(fn::hc::max_rows) && joint_rejects(ln);
        for (int32_t rows = 2; rows <= fn::hc::max_rows && ok; ++rows) {
            ok = lanes_inputs(rows);
            for (int32_t lanes = 2; lanes <= fn::xchg::max_lanes && ok; ++lanes) {
                for (const bool head : { false, true }) {
                    ok = joint_case(ln, lanes, kinds, head, rows) && ok;
                }
            }
        }
        ok = devices_idle(ln) && ok;
        cudaSetDevice(dev0);
    }
    printf("hcj-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
