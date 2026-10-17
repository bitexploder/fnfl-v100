#include "fn/kernels/head/head_test.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/head/head.h"
#include "fn/model/shapes.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <strings.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double head_bound = 1e-5;   // FC-HEAD1: the full mixed row, max |err| / max |ref| a row
constexpr double mom_bound = 1e-6;    // the moments and x, as HC2's test checks them
constexpr double logits_bound = 1e-5; // FC-HEAD2: DP2's product bound a row
constexpr double timeout_s = 30.0;    // the host's bound on one round

constexpr int32_t D = fn::n_embd;          // 2,560
constexpr int32_t S = fn::n_hc;            // 4 streams
constexpr int32_t K = S * D;               // 10,240
constexpr int32_t F = fn::hc::n_own;       // 1,280
constexpr int32_t L = fn::hc::n_down;      // 320
constexpr int32_t MH = fn::hc::n_mom_head; // 1,284
constexpr int32_t R = fn::hc::max_rows;
constexpr int32_t VC = fn::head::n_vocab_card;
constexpr int32_t q5_row_bytes = D / 256 * 176; // 1,760: a GGUF Q5_K row of output.weight
constexpr uint16_t f16_nan = 0x7e00;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "v-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// owns one device allocation of n elements of T (on the current device)
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

    bool put(const T * h, size_t count, cudaStream_t st) const {
        return count <= n && cuda_ok(cudaMemcpyAsync(p, h, count * sizeof(T), cudaMemcpyHostToDevice, st), "put");
    }

    bool get(std::vector<T> & h, size_t count, cudaStream_t st) const {
        h.assign(count, T{});
        return count <= n &&
               cuda_ok(cudaMemcpyAsync(h.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost, st), "get");
    }

    // NaN bytes for F32 (and -1 for integers), so an element the kernel never writes fails the comparison
    bool poison(cudaStream_t st) const { return cuda_ok(cudaMemsetAsync(p, 0xff, n * sizeof(T), st), "poison"); }

    bool zero(cudaStream_t st) const { return cuda_ok(cudaMemsetAsync(p, 0, n * sizeof(T), st), "zero"); }
};

// ---- the real weights, from the GGUF on the CPU ----

const fnfl_gguf_tensor * find_tensor(const std::vector<fnfl_gguf_dir> & dirs, const std::string & name, size_t & d) {
    for (d = 0; d < dirs.size(); ++d) {
        for (const fnfl_gguf_tensor & t : dirs[d].tensors) {
            if (t.name == name) {
                return &t;
            }
        }
    }
    return nullptr;
}

// bytes [at, at + n) of a tensor's data
bool read_bytes(const std::vector<fnfl_gguf_dir> & dirs, const std::string & name, const char * type, uint64_t want,
                uint64_t at, uint64_t n, void * out) {
    size_t d = 0;
    const fnfl_gguf_tensor * t = find_tensor(dirs, name, d);
    if (t == nullptr || strcasecmp(t->type_name.c_str(), type) != 0 || t->bytes != want || at + n > want) {
        fprintf(stderr, "v-test: %s: missing, not %s or not %llu bytes\n", name.c_str(), type,
                static_cast<unsigned long long>(want));
        return false;
    }
    FILE * f = fopen(dirs[d].path.c_str(), "rb");
    const bool ok = f != nullptr && fseeko(f, static_cast<off_t>(t->offset + at), SEEK_SET) == 0 &&
                    fread(out, 1, size_t(n), f) == size_t(n);
    if (f != nullptr) {
        fclose(f);
    }
    if (!ok) {
        fprintf(stderr, "v-test: %s: read failed\n", name.c_str());
    }
    return ok;
}

// the head's weights in the GGUF layouts (ne0 contiguous) and their two card repacks (hc.h's head mode)
struct head_weights {
    std::vector<uint16_t> down, up; // [320][10240], [10240][320] BF16
    std::vector<float> gamma;       // [10240]
    std::vector<uint16_t> w_down[2], w_up[2];
    std::vector<float> gamma_card[2];
};

bool read_head(const std::vector<fnfl_gguf_dir> & dirs, head_weights & w) {
    w.down.resize(size_t(L) * K);
    w.up.resize(size_t(K) * L);
    w.gamma.resize(K);
    const uint64_t nb = uint64_t(L) * K * 2;
    if (!read_bytes(dirs, "output_hc_down.weight", "bf16", nb, 0, nb, w.down.data()) ||
        !read_bytes(dirs, "output_hc_up.weight", "bf16", nb, 0, nb, w.up.data()) ||
        !read_bytes(dirs, "output_hc_norm.weight", "f32", uint64_t(K) * 4, 0, uint64_t(K) * 4, w.gamma.data())) {
        return false;
    }
    for (int32_t r = 0; r < 2; ++r) {
        w.w_down[r].resize(fn::hc::w_down_head_elems);
        w.w_up[r].resize(fn::hc::w_up_elems);
        w.gamma_card[r].resize(size_t(S) * F);
        for (int32_t s = 0; s < S; ++s) {
            for (int32_t j = 0; j < L; ++j) {
                for (int32_t i = 0; i < F; ++i) {
                    w.w_down[r][(size_t(s) * L + j) * F + i] =
                        w.down[size_t(j) * K + size_t(s) * D + size_t(r) * F + i];
                }
            }
            for (int32_t i = 0; i < F; ++i) {
                w.gamma_card[r][size_t(s) * F + i] = w.gamma[s * D + r * F + i];
            }
        }
        for (int32_t c = 0; c < F; ++c) {
            for (int32_t s = 0; s < S; ++s) {
                const uint16_t * src = &w.up[size_t(s * D + r * F + c) * L];
                std::copy(src, src + L, &w.w_up[r][(size_t(c) * S + s) * L]);
            }
        }
    }
    return true;
}

// card r's slice of output.weight: GGUF rows r * 124160 .. + n - 1 (global IDs), as raw Q5_K rows
bool read_vocab_slice(const std::vector<fnfl_gguf_dir> & dirs, int32_t r, int32_t n, std::vector<uint8_t> & rows) {
    rows.resize(size_t(n) * q5_row_bytes);
    const uint64_t total = uint64_t(fn::head::n_vocab) * q5_row_bytes;
    return read_bytes(dirs, "output.weight", "q5_K", total, uint64_t(r) * VC * q5_row_bytes, rows.size(), rows.data());
}

// a Q5_K row's first block gets a NaN d: every logit of that vocabulary row is NaN for every input
void plant_nan_row(std::vector<uint8_t> & rows, int32_t i) {
    memcpy(&rows[size_t(i) * q5_row_bytes], &f16_nan, 2);
}

// ---- inputs and the FP64 references ----

struct inputs {
    std::vector<float> hres, hx, hpost; // [R][4][2560], [R][2560], [R][4]
};

void make_inputs(inputs & in, uint32_t seed, bool stress) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d_act(-2.0f, 2.0f), d_post(0.5f, 1.5f);
    in.hres.resize(size_t(R) * S * D);
    in.hx.resize(size_t(R) * D);
    in.hpost.resize(size_t(R) * S);
    for (auto & v : in.hres) {
        v = d_act(rng);
    }
    for (auto & v : in.hx) {
        v = d_act(rng);
    }
    for (auto & v : in.hpost) {
        v = d_post(rng);
    }
    if (stress) {
        // streams 1 and 2 scaled by 1e3 and 1e-3 (hres and hpost), as HC2's stress case
        for (int32_t b = 0; b < R; ++b) {
            for (int32_t s = 1; s <= 2; ++s) {
                const float f = s == 1 ? 1e3f : 1e-3f;
                for (int32_t d = 0; d < D; ++d) {
                    in.hres[(size_t(b) * S + s) * D + d] *= f;
                }
                in.hpost[size_t(b) * S + s] *= f;
            }
        }
    }
}

// build_hc_mix without injection in FP64 (qwen4exp.cpp 327-396 at 66e11c3), from layer 47's boundary
struct head_ref {
    std::vector<double> x;      // [R][4][2560]
    std::vector<double> mom[2]; // [R][1284] per card
    std::vector<double> mixed;  // [R][2560]
};

void make_head_ref(const head_weights & w, const inputs & in, head_ref & ref) {
    ref.x.assign(size_t(R) * S * D, 0.0);
    for (int32_t b = 0; b < R; ++b) {
        for (int32_t s = 0; s < S; ++s) {
            for (int32_t d = 0; d < D; ++d) {
                const size_t at = (size_t(b) * S + s) * D + d;
                ref.x[at] = double(in.hres[at]) + double(in.hpost[size_t(b) * S + s]) * in.hx[size_t(b) * D + d];
            }
        }
    }
    for (int32_t r = 0; r < 2; ++r) {
        ref.mom[r].assign(size_t(R) * MH, 0.0);
        for (int32_t b = 0; b < R; ++b) {
            double * m = &ref.mom[r][size_t(b) * MH];
            for (int32_t s = 0; s < S; ++s) {
                const double * xs = &ref.x[(size_t(b) * S + s) * D + size_t(r) * F];
                std::vector<double> xg(F);
                double v = 0.0;
                for (int32_t i = 0; i < F; ++i) {
                    v += xs[i] * xs[i];
                    xg[i] = double(w.gamma[s * D + r * F + i]) * xs[i];
                }
                m[s] = v;
                for (int32_t j = 0; j < L; ++j) {
                    const uint16_t * wr = &w.down[size_t(j) * K + size_t(s) * D + size_t(r) * F];
                    double u = 0.0;
                    for (int32_t i = 0; i < F; ++i) {
                        u += double(fn::quant::bf16_to_f32(wr[i])) * xg[i];
                    }
                    m[4 + L * s + j] = u;
                }
            }
        }
    }
    ref.mixed.assign(size_t(R) * D, 0.0);
    for (int32_t b = 0; b < R; ++b) {
        const double * m0 = &ref.mom[0][size_t(b) * MH];
        const double * m1 = &ref.mom[1][size_t(b) * MH];
        double rms[S];
        for (int32_t s = 0; s < S; ++s) {
            rms[s] = std::sqrt((m0[s] + m1[s]) / D + double(fn::rms_eps));
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
        for (int32_t d = 0; d < D; ++d) {
            double mix = 0.0;
            for (int32_t s = 0; s < S; ++s) {
                const uint16_t * wr = &w.up[size_t(s * D + d) * L];
                double z = 0.0;
                for (int32_t j = 0; j < L; ++j) {
                    z += double(fn::quant::bf16_to_f32(wr[j])) * low[j];
                }
                const double xv = ref.x[(size_t(b) * S + s) * D + d];
                mix += double(w.gamma[s * D + d]) * xv / rms[s] / (1.0 + std::exp(-z));
            }
            ref.mixed[size_t(b) * D + d] = 0.25 * mix;
        }
    }
}

// max |gpu - ref| / max |ref| over n elements; a NaN on either side where the other is not is reported as infinity
double rel_err(const float * g, const double * r, size_t n) {
    double e = 0.0, m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const bool gn = std::isnan(g[i]), rn = std::isnan(r[i]);
        if (gn != rn) {
            return INFINITY;
        }
        if (rn) {
            continue;
        }
        e = std::max(e, std::fabs(double(g[i]) - r[i]));
        m = std::max(m, std::fabs(r[i]));
    }
    return m > 0.0 ? e / m : e;
}

// the decoded Q5_K slice of one card, F32 [n][2560]
std::vector<float> decode_slice(const std::vector<uint8_t> & rows, int32_t n) {
    std::vector<float> out(size_t(n) * D);
    for (int32_t i = 0; i < n; ++i) {
        fn::quant::decode_row(fn::quant::wfmt::q5_k, &rows[size_t(i) * q5_row_bytes], &out[size_t(i) * D], D);
    }
    return out;
}

// FP64 logits of B rows of act values against a decoded slice
std::vector<double> logits_ref(const std::vector<float> & w, int32_t n, const std::vector<float> & xv, int32_t B) {
    std::vector<double> out(size_t(B) * n);
    for (int32_t b = 0; b < B; ++b) {
        const float * x = &xv[size_t(b) * D];
        for (int32_t i = 0; i < n; ++i) {
            const float * wr = &w[size_t(i) * D];
            double acc = 0.0;
            for (int32_t k = 0; k < D; ++k) {
                acc += double(wr[k]) * double(x[k]);
            }
            out[size_t(b) * n + i] = acc;
        }
    }
    return out;
}

// ---- the greedy rule on the host (FC-DEC1), written from the contract, not from the kernel ----

struct pick {
    int32_t id = -1; // global ID, -1 when every value is NaN
    double value = 0.0;
    bool tied = false; // the maximum is attained at more than one ID
};

// ascending IDs: a strictly larger value replaces, so an exact tie (-0 == +0 included) keeps the lower ID; NaN never
// replaces. Column i's ID is id0 + i, or ids[i] with DR1's ID map (ascending, as draft_ids makes it).
template <typename T> void scan(const T * v, int32_t n, int32_t id0, pick & p, const int32_t * ids = nullptr) {
    for (int32_t i = 0; i < n; ++i) {
        const double x = double(v[i]);
        if (std::isnan(x)) {
            continue;
        }
        if (p.id < 0 || x > p.value) {
            p.id = ids != nullptr ? ids[i] : id0 + i;
            p.value = x;
            p.tied = false;
        } else if (x == p.value) {
            p.tied = true;
        }
    }
}

// the pair a card should send: the canonical bits of its maximum (+0 for either zero), its ID; NaN bits for none
uint32_t canonical_bits(double v) {
    const float f = static_cast<float>(v);
    if (f == 0.0f) {
        return 0u;
    }
    uint32_t u = 0;
    memcpy(&u, &f, 4); // F32 bits of the value
    return u;
}

// ---- both cards ----

struct device_guard {
    explicit device_guard(int32_t d) { cudaSetDevice(d); }
};

constexpr int32_t n_slice = 8192; // vocabulary rows a card in the head cases (DP2's slice size)

struct card_v {
    int32_t dev;
    device_guard guard;
    cudaStream_t st = nullptr;
    dev_buf<uint16_t> w_down{size_t(fn::hc::w_down_head_elems)}, w_up{size_t(fn::hc::w_up_elems)};
    dev_buf<float> gamma{size_t(S) * F};
    dev_buf<float> hres{size_t(R) * S * F}, hx{size_t(R) * F}, hpost{size_t(R) * S};
    dev_buf<float> x{size_t(R) * S * F}, mom{size_t(R) * MH}, full{size_t(R) * D};
    // HC3: the head's mixed row in both forms as ready assembles it from X-HMIX, and up2's run counters
    dev_buf<int8_t> codes{size_t(R) * D};
    dev_buf<float> scales{size_t(R) * D / 32 * 2};
    dev_buf<uint32_t> runs{size_t(fn::hc::n_runs_card)};
    dev_buf<uint8_t> w_vocab{size_t(n_slice) * q5_row_bytes};
    dev_buf<float> logits{size_t(R) * VC};
    dev_buf<uint8_t> lwork{std::max<size_t>(16, fn::head::logits_work_bytes(n_slice))};
    dev_buf<uint8_t> gwork{fn::head::greedy_work_bytes()};
    dev_buf<uint32_t> pair{size_t(R) * 2}, seen{size_t(R) * 2};
    dev_buf<int32_t> token{size_t(R)};

    explicit card_v(int32_t d) : dev(d), guard(d) {
        if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) {
            st = nullptr;
        }
    }

    ~card_v() {
        cudaSetDevice(dev);
        if (st != nullptr) {
            cudaStreamDestroy(st);
        }
    }

    card_v(const card_v &) = delete;
    card_v & operator=(const card_v &) = delete;
};

bool wait_two(card_v * c[2]) {
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
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
            fprintf(stderr, "v-test: host timeout after %.0f s\n", timeout_s);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

// in loopback (both cards on one device) card 0's stream, so every send is enqueued before its read
cudaStream_t launch_stream(card_v * c[2], int32_t r) {
    return c[0]->dev == c[1]->dev ? c[0]->st : c[r]->st;
}

// one round on both cards: the head (if head), the logits (if head), greedy; slot k % 3, and the head's down phase
// resets the previous round's X-HC slot (prev >= 0)
struct round_cfg {
    int32_t rows = 1;
    int32_t n = n_slice;
    bool head = true;
    int32_t slot = 0;
    int32_t prev = -1;
    // DR1's draft head: card r's packed subset copy, its ID map on the device and the product's work (null: the
    // vocabulary slice, id0 = r * VC), and the map on the host for the checks
    const uint8_t * w[2] = {nullptr, nullptr};
    const int32_t * ids_dev[2] = {nullptr, nullptr};
    void * lwork[2] = {nullptr, nullptr};
    const int32_t * ids[2] = {nullptr, nullptr};
};

// the card and column of global ID t in round rc; false when no card holds it
bool locate(const round_cfg & rc, int32_t t, int32_t & r, int32_t & col) {
    for (r = 0; r < 2; ++r) {
        if (rc.ids[r] == nullptr) {
            col = t - r * VC;
            if (col >= 0 && col < rc.n) {
                return true;
            }
            continue;
        }
        const int32_t * e = rc.ids[r] + rc.n;
        const int32_t * f = std::lower_bound(rc.ids[r], e, t);
        if (f != e && *f == t) {
            col = static_cast<int32_t>(f - rc.ids[r]);
            return true;
        }
    }
    return false;
}

bool enqueue_round(card_v * c[2], const fn::xchg::rings & rg, const round_cfg & rc) {
    using fn::xchg::msg;
    const int32_t B = rc.rows, s = rc.slot;
    bool ok = true;
    if (rc.head) {
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card_v & k = *c[r];
            fn::hc::down2_args d2;
            d2.d = {k.w_down.p, k.hres.p, k.hx.p, k.hpost.p, k.gamma.p, k.x.p, k.mom.p};
            d2.hc_peer = rg.slot(1 - r, msg::hc, s);
            d2.hc_reset = rc.prev >= 0 ? rg.slot(r, msg::hc, rc.prev) : nullptr;
            d2.hmix_reset = rc.prev >= 0 ? rg.slot(r, msg::hmix, rc.prev) : nullptr;
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::hc::head_down2(B, d2, launch_stream(c, r));
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card_v & k = *c[r];
            const fn::hc::up2_args u2 = {k.w_up.p,
                                         k.mom.p,
                                         rg.slot(r, msg::hc, s),
                                         k.x.p,
                                         k.gamma.p,
                                         nullptr,
                                         rg.slot(r, msg::hmix, s),
                                         rg.slot(1 - r, msg::hmix, s),
                                         k.runs.p,
                                         rg.err(r),
                                         r};
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::hc::head_up2(B, u2, launch_stream(c, r));
        }
        // the test's view of the row (F32, codes, scales) from the slot; the slot is left for logits (HC3)
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card_v & k = *c[r];
            const fn::hc::ready_args ra = {rg.slot(r, msg::hmix, s), k.full.p, k.codes.p, k.scales.p, rg.err(r), r, 0};
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::hc::ready(B, ra, launch_stream(c, r));
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card_v & k = *c[r];
            const fn::head::logits_args la = {rg.slot(r, msg::hmix, s),
                                              rg.err(r),
                                              r,
                                              rc.w[r] != nullptr ? rc.w[r] : k.w_vocab.p,
                                              rc.n,
                                              k.logits.p,
                                              rc.lwork[r] != nullptr ? rc.lwork[r] : k.lwork.p};
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::head::logits(B, la, launch_stream(c, r));
        }
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card_v & k = *c[r];
        fn::head::reduce_args ga = {k.logits.p, rc.n, r * VC, k.pair.p, rg.slot(1 - r, msg::greedy, s), k.gwork.p};
        ga.ids = rc.ids_dev[r];
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::head::greedy_reduce(B, ga, launch_stream(c, r));
    }
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card_v & k = *c[r];
        const fn::head::decide_args da = {k.pair.p, rg.slot(r, msg::greedy, s), k.token.p, rg.err(r), k.seen.p, r};
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && fn::head::greedy_decide(B, da, launch_stream(c, r));
    }
    if (!ok) {
        fprintf(stderr, "v-test: round not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    return ok;
}

// what a round leaves on each card
struct round_out {
    std::vector<float> x[2], mom[2], full[2], logits[2], scales[2], hres[2], hx[2], hpost[2];
    std::vector<int8_t> codes[2];
    std::vector<uint32_t> pair[2], seen[2], ring_hc[2];
    std::vector<int32_t> token[2];
    uint32_t err[2] = {~0u, ~0u};
};

bool read_round(card_v * c[2], const fn::xchg::rings & rg, const round_cfg & rc, round_out & o) {
    const int32_t B = rc.rows;
    bool ok = wait_two(c) && rg.read_err(0, o.err[0]) && rg.read_err(1, o.err[1]);
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card_v & k = *c[r];
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && k.pair.get(o.pair[r], size_t(B) * 2, k.st) &&
             k.seen.get(o.seen[r], size_t(B) * 2, k.st) && k.token.get(o.token[r], size_t(B), k.st) &&
             k.logits.get(o.logits[r], size_t(B) * rc.n, k.st);
        if (ok && rc.head) {
            ok = k.x.get(o.x[r], size_t(B) * S * F, k.st) && k.mom.get(o.mom[r], size_t(B) * MH, k.st) &&
                 k.full.get(o.full[r], size_t(B) * D, k.st) && k.codes.get(o.codes[r], size_t(B) * D, k.st) &&
                 k.scales.get(o.scales[r], size_t(B) * D / 32 * 2, k.st) &&
                 k.hres.get(o.hres[r], size_t(B) * S * F, k.st) && k.hx.get(o.hx[r], size_t(B) * F, k.st) &&
                 k.hpost.get(o.hpost[r], size_t(B) * S, k.st);
            // this card's X-HC receive slot: the peer's moments as delivered (read before the next round resets it)
            o.ring_hc[r].assign(size_t(B) * MH, 0u);
            ok = ok && cuda_ok(cudaMemcpyAsync(o.ring_hc[r].data(), rg.slot(r, fn::xchg::msg::hc, rc.slot),
                                               o.ring_hc[r].size() * 4, cudaMemcpyDeviceToHost, k.st),
                               "ring");
        }
        ok = ok && cuda_ok(cudaStreamSynchronize(k.st), "read");
    }
    return ok;
}

// the boundary inputs a card holds (layer 47's FC-LAY1 halves), uploaded for rows 0..R-1
bool put_inputs(card_v * c[2], const inputs & in, std::vector<float> (&hres)[2], std::vector<float> (&hx)[2],
                std::vector<float> & hpost) {
    bool ok = true;
    hpost = in.hpost;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        hres[r].assign(size_t(R) * S * F, 0.0f);
        hx[r].assign(size_t(R) * F, 0.0f);
        for (int32_t b = 0; b < R; ++b) {
            for (int32_t s = 0; s < S; ++s) {
                const size_t bs = size_t(b) * S + s;
                std::copy_n(&in.hres[bs * D + size_t(r) * F], F, &hres[r][bs * F]);
            }
            std::copy_n(&in.hx[size_t(b) * D + size_t(r) * F], F, &hx[r][size_t(b) * F]);
        }
        card_v & k = *c[r];
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && k.hres.put(hres[r].data(), hres[r].size(), k.st) &&
             k.hx.put(hx[r].data(), hx[r].size(), k.st) && k.hpost.put(hpost.data(), hpost.size(), k.st) &&
             k.x.poison(k.st) && k.mom.poison(k.st) && k.full.poison(k.st) && k.codes.poison(k.st) &&
             k.scales.poison(k.st) &&
             cuda_ok(cudaMemsetAsync(k.runs.p, 0, size_t(fn::hc::n_runs_card) * sizeof(uint32_t), k.st), "runs") &&
             k.logits.poison(k.st) && k.pair.poison(k.st) && k.seen.poison(k.st) && k.token.poison(k.st) &&
             cuda_ok(cudaStreamSynchronize(k.st), "inputs");
    }
    return ok;
}

bool put_vocab(card_v * c[2], const std::vector<uint8_t> (&raw)[2]) {
    bool ok = true;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        std::vector<uint8_t> packed(raw[r].size());
        card_v & k = *c[r];
        ok = fn::dense::pack(fn::dense::fmt::q5_k, n_slice, D, raw[r].data(), q5_row_bytes, packed.data()) &&
             cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && k.w_vocab.put(packed.data(), packed.size(), k.st) &&
             cuda_ok(cudaStreamSynchronize(k.st), "vocab");
    }
    return ok;
}

bool report(const char * what, bool pass, const std::string & detail = "") {
    printf("v-test %-58s %s%s\n", what, detail.c_str(), pass ? "" : "  FAIL");
    return pass;
}

std::string fmt_e(double e) {
    char b[32];
    snprintf(b, sizeof(b), "%.3g", e);
    return b;
}

// the expected greedy result of a round from its own GPU logits (exact), and the checks of the pairs and tokens on
// both cards; per row: the token, and whether the maximum was tied
struct greedy_check {
    std::vector<int32_t> token;
    std::vector<bool> tied;
    int32_t cross_ties = 0;       // rows whose two cards' maxima are equal (the lower ID must win)
    int32_t cross_ties_card1 = 0; // those decided to card 1 (only with DR1's ID map: card 0's IDs are lower otherwise)
    bool pass = true;
};

greedy_check check_greedy(const round_cfg & rc, const round_out & o, const char * label, bool quiet = false) {
    greedy_check g;
    const int32_t B = rc.rows;
    int32_t bad_pair = 0, bad_seen = 0, bad_token = 0;
    for (int32_t b = 0; b < B; ++b) {
        pick card[2], both;
        for (int32_t r = 0; r < 2; ++r) {
            scan(&o.logits[r][size_t(b) * rc.n], rc.n, r * VC, card[r], rc.ids[r]);
            const uint32_t v = card[r].id < 0 ? fn::xchg::canonical_nan : canonical_bits(card[r].value);
            const size_t p = 2 * size_t(b);
            const bool same_value = o.pair[r][p] == v;
            const bool same_id = card[r].id < 0 || int32_t(o.pair[r][p + 1]) == card[r].id;
            bad_pair += same_value && same_id ? 0 : 1;
            // T-GREEDY: the peer's pair as polled equals the pair the peer kept
            bad_seen += o.seen[r][p] == o.pair[1 - r][p] && o.seen[r][p + 1] == o.pair[1 - r][p + 1] ? 0 : 1;
        }
        both = card[0];
        if (card[1].id >= 0 && (both.id < 0 || card[1].value > both.value)) {
            both = card[1];
        } else if (card[1].id >= 0 && both.id >= 0 && card[1].value == both.value) {
            both.tied = true;
            ++g.cross_ties;
            if (card[1].id < both.id) { // the lower ID on an exact tie (FC-DEC1, FC-DR1)
                both.id = card[1].id;
                ++g.cross_ties_card1;
            }
        }
        g.token.push_back(both.id);
        g.tied.push_back(both.tied);
        for (int32_t r = 0; r < 2; ++r) {
            bad_token += o.token[r][b] == both.id ? 0 : 1;
        }
    }
    g.pass = bad_pair == 0 && bad_seen == 0 && bad_token == 0;
    if (!quiet || !g.pass) {
        report((std::string(label) + ": greedy pairs, T-GREEDY and tokens").c_str(), g.pass,
               "pairs off " + std::to_string(bad_pair) + ", T-GREEDY off " + std::to_string(bad_seen) +
                   ", tokens off " + std::to_string(bad_token) + " (of " + std::to_string(B) + " rows x 2 cards)");
    }
    return g;
}

// the head's and the logits' checks against FP64 for one end-to-end round, and the token against the FP64 argmax
// under NC1 section 3's own-error rule
bool check_head_round(const head_ref & ref, const std::vector<float> (&wdec)[2], const round_cfg & rc,
                      const round_out & o, const std::vector<float> (&hres)[2], const std::vector<float> (&hx)[2],
                      const std::vector<float> & hpost, const char * label) {
    const int32_t B = rc.rows;
    bool pass = o.err[0] == 0u && o.err[1] == 0u;
    report((std::string(label) + ": error words").c_str(), pass,
           std::to_string(o.err[0]) + " " + std::to_string(o.err[1]));
    double e_x = 0.0, e_mom = 0.0, e_full = 0.0, e_log = 0.0;
    int32_t bad_ring = 0, bad_codes = 0, bad_boundary = 0;
    bool full_same = true;
    for (int32_t r = 0; r < 2; ++r) {
        for (int32_t b = 0; b < B; ++b) {
            for (int32_t s = 0; s < S; ++s) {
                std::vector<double> xr(F);
                for (int32_t i = 0; i < F; ++i) {
                    xr[i] = ref.x[(size_t(b) * S + s) * D + size_t(r) * F + i];
                }
                e_x = std::max(e_x, rel_err(&o.x[r][(size_t(b) * S + s) * F], xr.data(), F));
            }
            e_mom = std::max(e_mom, rel_err(&o.mom[r][size_t(b) * MH], &ref.mom[r][size_t(b) * MH], MH));
            e_full = std::max(e_full, rel_err(&o.full[r][size_t(b) * D], &ref.mixed[size_t(b) * D], D));
        }
        // X-HC delivered intact: this card's slot holds the peer's own moment words
        bad_ring += memcmp(o.ring_hc[r].data(), o.mom[1 - r].data(), size_t(B) * MH * 4) == 0 ? 0 : 1;
        // FC-HEAD3: layer 47's boundary bitwise unchanged
        bad_boundary += memcmp(o.hres[r].data(), hres[r].data(), size_t(B) * S * F * 4) == 0 &&
                                memcmp(o.hx[r].data(), hx[r].data(), size_t(B) * F * 4) == 0 &&
                                memcmp(o.hpost[r].data(), hpost.data(), size_t(B) * S * 4) == 0
                            ? 0
                            : 1;
    }
    full_same = memcmp(o.full[0].data(), o.full[1].data(), size_t(B) * D * 4) == 0;
    pass = report((std::string(label) + ": x and moments vs FP64 (<= 1e-6)").c_str(),
                  e_x <= mom_bound && e_mom <= mom_bound, "x " + fmt_e(e_x) + ", moments " + fmt_e(e_mom)) &&
           pass;
    pass = report((std::string(label) + ": full mixed row vs FP64 (<= 1e-5 a row)").c_str(), e_full <= head_bound,
                  fmt_e(e_full)) &&
           pass;
    pass = report((std::string(label) + ": full mixed row identical on both cards").c_str(), full_same,
                  full_same ? "yes" : "no") &&
           pass;
    pass = report((std::string(label) + ": X-HC slots hold the peer's moments bitwise").c_str(), bad_ring == 0,
                  std::to_string(2 - bad_ring) + " of 2") &&
           pass;
    pass = report((std::string(label) + ": layer 47's boundary (hres, hx, hpost) unchanged").c_str(), bad_boundary == 0,
                  std::to_string(2 - bad_boundary) + " of 2 cards bitwise") &&
           pass;
    // the logits: act codes bitwise to act_ref of the GPU's own full row, then FP64 on the decoded slice
    std::vector<double> lref[2];
    for (int32_t r = 0; r < 2; ++r) {
        std::vector<int8_t> codes(size_t(B) * D);
        std::vector<float> scales(size_t(B) * D / 32 * 2), xv(size_t(B) * D);
        fn::quant::act_quantize_ref(o.full[r].data(), B, D, codes.data(), scales.data());
        bad_codes += codes == o.codes[r] && memcmp(scales.data(), o.scales[r].data(), scales.size() * 4) == 0 ? 0 : 1;
        for (int32_t b = 0; b < B; ++b) {
            fn::quant::act_values_ref(&codes[size_t(b) * D], &scales[size_t(b) * D / 32 * 2], D, &xv[size_t(b) * D]);
        }
        lref[r] = logits_ref(wdec[r], rc.n, xv, B);
        for (int32_t b = 0; b < B; ++b) {
            e_log = std::max(e_log, rel_err(&o.logits[r][size_t(b) * rc.n], &lref[r][size_t(b) * rc.n], size_t(rc.n)));
        }
    }
    pass = report((std::string(label) + ": act codes and scales bitwise to act_ref").c_str(), bad_codes == 0,
                  std::to_string(2 - bad_codes) + " of 2") &&
           pass;
    pass = report((std::string(label) + ": logits vs FP64 (<= 1e-5 a row)").c_str(), e_log <= logits_bound,
                  fmt_e(e_log)) &&
           pass;
    // the token against the FP64 argmax: equal, or explained by the row's own error (NC1 section 3)
    const greedy_check g = check_greedy(rc, o, label);
    pass = g.pass && pass;
    int32_t equal = 0, explained = 0, unexplained = 0;
    for (int32_t b = 0; b < B; ++b) {
        pick fp;
        double own = 0.0;
        for (int32_t r = 0; r < 2; ++r) {
            scan(&lref[r][size_t(b) * rc.n], rc.n, r * VC, fp, rc.ids[r]);
            for (int32_t i = 0; i < rc.n; ++i) {
                const double d = double(o.logits[r][size_t(b) * rc.n + i]) - lref[r][size_t(b) * rc.n + i];
                if (!std::isnan(d)) {
                    own = std::max(own, std::fabs(d));
                }
            }
        }
        const int32_t t = g.token[b];
        if (t == fp.id) {
            ++equal;
            continue;
        }
        int32_t tr = 0, tc = 0;
        const bool held = t >= 0 && locate(rc, t, tr, tc);
        const double v_t = held ? lref[tr][size_t(b) * rc.n + tc] : NAN;
        if (held && fp.id >= 0 && std::fabs(fp.value - v_t) <= 2.0 * own) {
            ++explained;
        } else {
            ++unexplained;
        }
    }
    pass = report((std::string(label) + ": token vs the FP64 argmax").c_str(), unexplained == 0,
                  std::to_string(equal) + " equal, " + std::to_string(explained) + " by own error, " +
                      std::to_string(unexplained) + " unexplained") &&
           pass;
    return pass;
}

bool same_rows(const std::vector<float> & a, size_t row_a, const std::vector<float> & b, size_t row_b, size_t w) {
    return memcmp(&a[row_a * w], &b[row_b * w], w * 4) == 0;
}

// ---- the synthetic greedy cases (full shard, logits written directly) ----

float from_bits(uint32_t u) {
    float f = 0.0f;
    memcpy(&f, &u, 4); // F32 from bits
    return f;
}

// set 0: per row, (0) card 1's strict maximum is a denormal over card 0's zero: card 1 wins; (1) an exact tie within
// card 0 far apart (IDs 7 and 100,000) with NaNs of every kind scattered; (2) an exact tie across the cards (card 0's
// 120,000, card 1's 3): card 0 wins; (3) random values, unique maximum on card 1
// set 1: (0) card 0 all NaN, card 1 all NaN but one -inf: the -inf wins; (1) all NaN on both cards: token -1 and
// err_all_nan; (2) +inf on both cards: card 0 wins; (3) -0 and +0 as the maximum on card 0: the lower ID, value +0.
// SP2, row 4 of both sets: set 0, the maximum on card 1; set 1, an exact tie across the cards: card 0's lower ID
void make_greedy_set(int32_t set, std::vector<float> (&lg)[2]) {
    std::mt19937 rng(777u + uint32_t(set));
    std::uniform_real_distribution<float> d(-30.0f, 30.0f);
    for (int32_t r = 0; r < 2; ++r) {
        lg[r].assign(size_t(R) * VC, 0.0f);
        for (auto & v : lg[r]) {
            v = d(rng);
        }
    }
    auto at = [&](int32_t r, int32_t b, int32_t i) -> float & { return lg[r][size_t(b) * VC + i]; };
    const uint32_t nans[4] = {0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xffffffffu};
    if (set == 0) {
        for (int32_t i = 0; i < VC; ++i) {
            at(0, 0, i) = -1.0f - float(i % 13);
            at(1, 0, i) = -1.0f - float(i % 11);
        }
        at(0, 0, 500) = 0.0f;
        at(1, 0, 77) = from_bits(1u); // the smallest positive denormal
        for (int32_t i = 0; i < VC; i += 997) {
            at(0, 1, i) = from_bits(nans[(i / 997) % 4]);
            at(1, 1, i) = from_bits(nans[(i / 997 + 1) % 4]);
        }
        at(0, 1, 7) = at(0, 1, 100000) = 40.0f;
        at(1, 1, 5) = 39.5f;
        at(0, 2, 120000) = at(1, 2, 3) = 41.25f;
        at(1, 3, 123456) = 45.0f;
        at(1, 4, 4242) = 50.0f;
    } else {
        for (int32_t i = 0; i < VC; ++i) {
            at(0, 0, i) = at(1, 0, i) = from_bits(nans[i % 4]);
            at(0, 1, i) = at(1, 1, i) = from_bits(nans[(i + 1) % 4]);
            at(0, 3, i) = -1.0f;
            at(1, 3, i) = -1e-30f;
        }
        at(1, 0, 50000) = -INFINITY;
        at(0, 2, 5) = at(1, 2, 9) = INFINITY;
        at(0, 3, 1000) = -0.0f;
        at(0, 3, 2000) = 0.0f;
        at(0, 4, 2) = at(1, 4, 2) = 33.0f;
    }
}

bool put_logits(card_v * c[2], const std::vector<float> (&lg)[2], int32_t rows, int32_t first_row) {
    bool ok = true;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        card_v & k = *c[r];
        ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && k.pair.poison(k.st) && k.seen.poison(k.st) &&
             k.token.poison(k.st) && k.logits.put(&lg[r][size_t(first_row) * VC], size_t(rows) * VC, k.st) &&
             cuda_ok(cudaStreamSynchronize(k.st), "logits");
    }
    return ok;
}

bool run_greedy_cases(card_v * c[2], const fn::xchg::rings & rg, bool san) {
    bool pass = true;
    int32_t k = 0; // rounds since the last reset of the rings: slot k % 3
    static_assert(R == 5, "the planted rows (SP2)");
    const int32_t want_token[2][R] = {{VC + 77, 7, 120000, VC + 123456, VC + 4242}, {VC + 50000, -1, 5, 1000, 2}};
    for (int32_t set = 0; set < 2; ++set) {
        std::vector<float> lg[2];
        make_greedy_set(set, lg);
        round_out o4;
        round_cfg rc;
        rc.rows = R;
        rc.n = VC;
        rc.head = false;
        bool ok = rg.reset_all();
        k = 0;
        rc.slot = 0;
        ok = ok && put_logits(c, lg, R, 0) && enqueue_round(c, rg, rc) && read_round(c, rg, rc, o4);
        if (!ok) {
            return report("greedy synthetic set: round", false);
        }
        const std::string label = "greedy set " + std::to_string(set) + " B" + std::to_string(R);
        const greedy_check g = check_greedy(rc, o4, label.c_str());
        int32_t want = 0;
        for (int32_t b = 0; b < R; ++b) {
            want += g.token[b] == want_token[set][b] && o4.token[0][b] == want_token[set][b] ? 1 : 0;
        }
        pass =
            report((label + ": tokens as the cases plant them").c_str(), want == R,
                   std::to_string(want) + " of " + std::to_string(R)) &&
            g.pass && pass;
        const uint32_t want_err = set == 1 ? fn::head::err_all_nan : 0u;
        pass = report((label + ": error words").c_str(), o4.err[0] == want_err && o4.err[1] == want_err,
                      std::to_string(o4.err[0]) + " " + std::to_string(o4.err[1]) + " (want " +
                          std::to_string(want_err) + ")") &&
               pass;
        if (set == 1) {
            // the -0 / +0 tie is sent as +0
            pass = report((label + ": a -0 / +0 maximum is sent as +0").c_str(), o4.pair[0][2 * size_t{3}] == 0u,
                          std::to_string(o4.pair[0][2 * size_t{3}])) &&
                   pass;
            continue;
        }
        if (san) {
            continue;
        }
        // batch invariance: each row alone at B = 1, and a bitwise rerun at B = R (slots rotated)
        int32_t same_b1 = 0;
        for (int32_t b = 0; b < R && ok; ++b) {
            round_cfg r1 = rc;
            r1.rows = 1;
            r1.slot = ++k % fn::xchg::n_slot;
            round_out o1;
            ok = put_logits(c, lg, 1, b) && enqueue_round(c, rg, r1) && read_round(c, rg, r1, o1);
            bool same = ok;
            const size_t p = 2 * size_t(b);
            for (int32_t r = 0; r < 2 && ok; ++r) {
                same = same && o1.token[r][0] == o4.token[r][b] && o1.pair[r][0] == o4.pair[r][p] &&
                       o1.pair[r][1] == o4.pair[r][p + 1];
            }
            same_b1 += same ? 1 : 0;
        }
        pass = report((label + ": each row alone at B1 bitwise as in B" + std::to_string(R)).c_str(), same_b1 == R,
                      std::to_string(same_b1) + " of " + std::to_string(R)) &&
               pass;
        round_out o4b;
        rc.slot = ++k % fn::xchg::n_slot;
        ok = ok && put_logits(c, lg, R, 0) && enqueue_round(c, rg, rc) && read_round(c, rg, rc, o4b);
        const bool rerun = ok && o4b.token[0] == o4.token[0] && o4b.token[1] == o4.token[1] &&
                           o4b.pair[0] == o4.pair[0] && o4b.pair[1] == o4.pair[1];
        pass = report((label + ": rerun bitwise (another slot)").c_str(), rerun, rerun ? "yes" : "no") && pass;
    }
    return rg.reset_all() && pass;
}

// ---- DR1: the draft head (FC-DR1 "Cards"): the re-shard's ID maps, the packed copy, greedy with the map ----

constexpr int32_t n_subset = 2 * n_slice; // the test's subset: 16,384 IDs (8,192 a card)
constexpr int32_t n_dcard = n_subset / 2;

// a subset of n distinct IDs, 1.32% of them in card 1's shard as in the real one (Stage A: 98.68% on card 0), with
// the first and last vocabulary rows, in no order
std::vector<int32_t> make_subset(int32_t n) {
    std::mt19937 rng(4242u);
    std::vector<uint8_t> taken(size_t(fn::head::n_vocab), 0);
    std::vector<int32_t> s = {0, fn::head::n_vocab - 1};
    taken[0] = taken[size_t(fn::head::n_vocab) - 1] = 1;
    const int32_t n1 = n * 132 / 10000;
    auto take = [&](int32_t lo, int32_t hi, int32_t count) {
        std::uniform_int_distribution<int32_t> d(lo, hi - 1);
        while (count > 0) {
            const int32_t id = d(rng);
            if (taken[size_t(id)] == 0) {
                taken[size_t(id)] = 1;
                s.push_back(id);
                --count;
            }
        }
    };
    take(0, VC, n - n1 - 1);
    take(VC, fn::head::n_vocab, n1 - 1);
    std::shuffle(s.begin(), s.end(), rng);
    return s;
}

// card r's packed copy, its ID map and the product's work, on the card's device
struct draft_bufs {
    device_guard guard;
    dev_buf<uint8_t> w{size_t(n_dcard) * q5_row_bytes};
    dev_buf<int32_t> ids{size_t(n_dcard)};
    dev_buf<uint8_t> lwork{std::max<size_t>(16, fn::head::logits_work_bytes(n_dcard))};

    explicit draft_bufs(int32_t d) : guard(d) {}
};

// pack_draft_head of both cards' maps from the full output.weight (vocab, GGUF rows), uploaded; also checks the
// copy bitwise against dense::pack of the gathered rows at once (gathered: those rows, for the FP64 decode)
bool put_draft_head(card_v * c[2], draft_bufs * db[2], const std::vector<uint8_t> & vocab,
                    const std::vector<int32_t> (&ids)[2], std::vector<uint8_t> (&gathered)[2], bool & same_pack) {
    same_pack = true;
    bool ok = true;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        std::vector<uint8_t> packed(size_t(n_dcard) * q5_row_bytes), whole(packed.size());
        gathered[r].resize(packed.size());
        for (int32_t i = 0; i < n_dcard; ++i) {
            memcpy(&gathered[r][size_t(i) * q5_row_bytes], &vocab[size_t(ids[r][size_t(i)]) * q5_row_bytes],
                   q5_row_bytes);
        }
        ok = fn::head::pack_draft_head(vocab.data(), q5_row_bytes, ids[r].data(), n_dcard, packed.data()) &&
             fn::dense::pack(fn::dense::fmt::q5_k, n_dcard, D, gathered[r].data(), q5_row_bytes, whole.data());
        same_pack = same_pack && ok && packed == whole;
        card_v & k = *c[r];
        ok = ok && cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && db[r]->w.put(packed.data(), packed.size(), k.st) &&
             db[r]->ids.put(ids[r].data(), ids[r].size(), k.st) && cuda_ok(cudaStreamSynchronize(k.st), "draft head");
    }
    return ok;
}

// The draft head's cases: draft_ids against the rule written from the contract (ranks card, card + 2, ... of the
// sorted subset) and its rejects; the packed copy bitwise dense::pack of the gathered rows; one end-to-end round
// (the HC head, the logits on the copy against FP64 on the decoded rows, greedy with the ID map against the host
// rule and the FP64 argmax); then planted cross-card ties, each row's winner copied to the other card at a lower
// or a higher ID, so that the lower ID wins on either card.
bool run_draft_head_cases(card_v * c[2], const fn::xchg::rings & rg, const std::vector<fnfl_gguf_dir> & dirs,
                          const head_weights & hw, bool san) {
    const std::vector<int32_t> subset = make_subset(n_subset);
    std::vector<int32_t> sorted = subset;
    std::sort(sorted.begin(), sorted.end());
    std::vector<int32_t> ids[2];
    bool ok = true;
    int32_t bad = 0, below = 0;
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ids[r].assign(size_t(n_dcard), -1);
        ok = fn::head::draft_ids(subset.data(), n_subset, r, ids[r].data());
        for (int32_t i = 0; i < n_dcard && ok; ++i) {
            bad += ids[r][size_t(i)] == sorted[size_t(2 * i + r)] ? 0 : 1;
            below += ids[r][size_t(i)] < VC ? 1 : 0;
        }
    }
    bool pass = report("draft head: draft_ids, ranks card, card + 2, ... of the subset", ok && bad == 0,
                       std::to_string(bad) + " of " + std::to_string(n_subset) + " IDs off; " + std::to_string(below) +
                           " in card 0's shard");
    {
        std::vector<int32_t> dup = subset, out = subset, tmp(static_cast<size_t>(n_dcard));
        dup[1] = dup[0];
        out[0] = fn::head::n_vocab;
        const bool rejects = !fn::head::draft_ids(dup.data(), n_subset, 0, tmp.data()) &&
                             !fn::head::draft_ids(out.data(), n_subset, 0, tmp.data()) &&
                             !fn::head::draft_ids(subset.data(), n_subset - 2, 0, tmp.data()) &&
                             !fn::head::draft_ids(subset.data(), n_subset, 2, tmp.data()) &&
                             !fn::head::pack_draft_head(nullptr, q5_row_bytes, ids[0].data(), 8, nullptr);
        pass = report("draft head: rejects a repeated or out-of-range ID, n / 2 % 8 != 0, card 2, null rows", rejects,
                      rejects ? "yes" : "no") &&
               pass;
        const int64_t bytes = fn::head::draft_head_bytes(fn::head::n_draft_card);
        pass = report("draft head: draft_head_bytes(32,768) = 57,671,680", bytes == 57671680, std::to_string(bytes)) &&
               pass;
    }
    std::vector<uint8_t> vocab(size_t(fn::head::n_vocab) * q5_row_bytes);
    ok = ok && read_bytes(dirs, "output.weight", "q5_K", vocab.size(), 0, vocab.size(), vocab.data());
    if (!ok) {
        return report("draft head: inputs", false);
    }
    draft_bufs d0(c[0]->dev), d1(c[1]->dev);
    draft_bufs * db[2] = {&d0, &d1};
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(c[r]->dev), "cudaSetDevice") && db[r]->lwork.zero(c[r]->st) &&
             cuda_ok(cudaStreamSynchronize(c[r]->st), "draft work");
    }
    std::vector<uint8_t> gathered[2];
    bool same_pack = false;
    ok = ok && put_draft_head(c, db, vocab, ids, gathered, same_pack);
    pass = report("draft head: pack_draft_head bitwise dense::pack of the gathered rows", ok && same_pack,
                  same_pack ? "yes, both cards" : "no") &&
           pass;
    const std::vector<float> wdec[2] = {decode_slice(gathered[0], n_dcard), decode_slice(gathered[1], n_dcard)};

    inputs in;
    make_inputs(in, 4713u, false);
    head_ref ref;
    make_head_ref(hw, in, ref);
    std::vector<float> hres[2], hx[2], hpost;
    round_cfg rc;
    rc.rows = san ? 3 : R;
    rc.n = n_dcard;
    for (int32_t r = 0; r < 2; ++r) {
        rc.w[r] = db[r]->w.p;
        rc.ids_dev[r] = db[r]->ids.p;
        rc.lwork[r] = db[r]->lwork.p;
        rc.ids[r] = ids[r].data();
    }
    round_out oa;
    ok = ok && rg.reset_all() && put_inputs(c, in, hres, hx, hpost) && enqueue_round(c, rg, rc) &&
         read_round(c, rg, rc, oa);
    if (!ok) {
        return report("draft head: round", false);
    }
    const std::string label = "draft head B" + std::to_string(rc.rows);
    pass = check_head_round(ref, wdec, rc, oa, hres, hx, hpost, label.c_str()) && pass;

    // planted cross-card ties: row b's winner (card cw, ID idw) gets an exact copy on the other card, at a lower ID
    // (b even: the other card must win) or a higher one (b odd: the winner's card keeps it); target columns are no
    // row's winner on their card and not planted before
    std::vector<int32_t> win_col[2];
    for (int32_t r = 0; r < 2; ++r) {
        for (int32_t b = 0; b < rc.rows; ++b) {
            pick p;
            scan(&oa.logits[r][size_t(b) * rc.n], rc.n, 0, p);
            win_col[r].push_back(p.id);
        }
    }
    std::vector<int32_t> used[2] = {win_col[0], win_col[1]};
    std::vector<int32_t> want(size_t(rc.rows), -1);
    int32_t planted = 0, want_card1 = 0;
    for (int32_t b = 0; b < rc.rows; ++b) {
        pick card[2];
        for (int32_t r = 0; r < 2; ++r) {
            scan(&oa.logits[r][size_t(b) * rc.n], rc.n, 0, card[r], ids[r].data());
        }
        const int32_t cw =
            card[1].value > card[0].value || (card[1].value == card[0].value && card[1].id < card[0].id) ? 1 : 0;
        const int32_t idw = card[cw].id, o = 1 - cw;
        const bool lower = b % 2 == 0;
        const std::vector<int32_t> & m = ids[o];
        int32_t j = static_cast<int32_t>(std::lower_bound(m.begin(), m.end(), idw) - m.begin()); // first ID > idw
        if (lower) {
            --j;
        }
        auto taken = [&](int32_t col) { return std::find(used[o].begin(), used[o].end(), col) != used[o].end(); };
        while (j >= 0 && j < n_dcard && taken(j)) {
            j += lower ? -1 : 1;
        }
        if (j < 0 || j >= n_dcard) {
            continue;
        }
        used[o].push_back(j);
        memcpy(&vocab[size_t(m[size_t(j)]) * q5_row_bytes], &vocab[size_t(idw) * q5_row_bytes], q5_row_bytes);
        want[size_t(b)] = lower ? m[size_t(j)] : idw;
        want_card1 += (lower ? o : cw) == 1 ? 1 : 0;
        ++planted;
    }
    round_out ob;
    rc.slot = 1;
    ok = put_draft_head(c, db, vocab, ids, gathered, same_pack) && put_inputs(c, in, hres, hx, hpost) &&
         enqueue_round(c, rg, rc) && read_round(c, rg, rc, ob);
    if (!ok) {
        return report("draft head: planted round", false);
    }
    const greedy_check g = check_greedy(rc, ob, "draft head planted ties");
    int32_t as_planted = 0;
    for (int32_t b = 0; b < rc.rows; ++b) {
        as_planted += want[size_t(b)] >= 0 && ob.token[0][b] == want[size_t(b)] && ob.token[1][b] == want[size_t(b)];
    }
    pass = g.pass && pass;
    pass = report("draft head planted ties: cross-card, the lower ID wins on either card",
                  planted == rc.rows && as_planted == rc.rows && g.cross_ties == rc.rows &&
                      g.cross_ties_card1 == want_card1,
                  std::to_string(as_planted) + " of " + std::to_string(rc.rows) + " rows as planted; " +
                      std::to_string(g.cross_ties) + " cross-card ties, " + std::to_string(g.cross_ties_card1) +
                      " to card 1 (want " + std::to_string(want_card1) + ")") &&
           pass;
    pass = report("draft head: error words", (oa.err[0] | oa.err[1] | ob.err[0] | ob.err[1]) == 0u,
                  std::to_string(oa.err[0]) + " " + std::to_string(oa.err[1]) + " / " + std::to_string(ob.err[0]) +
                      " " + std::to_string(ob.err[1])) &&
           pass;
    return rg.reset_all() && pass;
}

bool check_rejects(card_v & k, const fn::xchg::rings & rg) {
    using fn::xchg::msg;
    cudaSetDevice(k.dev);
    fn::hc::down2_args d2;
    d2.d = {k.w_down.p, k.hres.p, k.hx.p, k.hpost.p, k.gamma.p, k.x.p, k.mom.p};
    d2.hc_peer = rg.slot(1, msg::hc, 0);
    d2.hc_reset = nullptr;
    fn::hc::down2_args d2_bad = d2;
    d2_bad.hc_peer = nullptr;
    const fn::hc::up2_args u2 = {k.w_up.p,
                                 k.mom.p,
                                 rg.slot(0, msg::hc, 0),
                                 k.x.p,
                                 k.gamma.p,
                                 nullptr,
                                 rg.slot(0, msg::hmix, 0),
                                 rg.slot(1, msg::hmix, 0),
                                 k.runs.p,
                                 rg.err(0),
                                 0};
    fn::hc::up2_args u2_card = u2, u2_err = u2;
    u2_card.card = 2;
    u2_err.err = nullptr;
    const fn::head::logits_args la = {
        rg.slot(0, msg::hmix, 0), rg.err(0), 0, k.w_vocab.p, n_slice, k.logits.p, k.lwork.p};
    fn::head::logits_args la_n = la, la_big = la, la_w = la, la_card = la, la_err = la;
    la_n.n = 1000 + 4; // not a multiple of 8
    la_big.n = VC + 8;
    la_w.w = k.w_vocab.p + 1;
    la_card.card = 2;
    la_err.err = nullptr;
    const fn::head::reduce_args ga = {k.logits.p, VC, 0, k.pair.p, rg.slot(1, msg::greedy, 0), k.gwork.p};
    fn::head::reduce_args ga_n = ga, ga_id = ga, ga_w = ga;
    ga_n.n = 6;
    ga_id.id0 = fn::head::n_vocab - VC + 4;
    ga_w.work = nullptr;
    const fn::head::decide_args da = {k.pair.p, rg.slot(0, msg::greedy, 0), k.token.p, rg.err(0), nullptr, 0};
    fn::head::decide_args da_card = da, da_err = da;
    da_card.card = 2;
    da_err.err = nullptr;
    const bool ok =
        !fn::hc::head_down2(0, d2, k.st) && !fn::hc::head_down2(6, d2, k.st) && !fn::hc::head_down2(3, d2_bad, k.st) &&
        !fn::hc::head_up2(6, u2, k.st) && !fn::hc::head_up2(3, u2_card, k.st) && !fn::hc::head_up2(3, u2_err, k.st) &&
        !fn::head::logits(0, la, k.st) && !fn::head::logits(6, la, k.st) && !fn::head::logits(1, la_n, k.st) &&
        !fn::head::logits(1, la_big, k.st) && !fn::head::logits(1, la_w, k.st) && !fn::head::logits(1, la, nullptr) &&
        !fn::head::logits(1, la_card, k.st) && !fn::head::logits(1, la_err, k.st) &&
        !fn::head::greedy_reduce(0, ga, k.st) && !fn::head::greedy_reduce(6, ga, k.st) &&
        !fn::head::greedy_reduce(1, ga_n, k.st) && !fn::head::greedy_reduce(1, ga_id, k.st) &&
        !fn::head::greedy_reduce(1, ga_w, k.st) && !fn::head::greedy_decide(6, da, k.st) &&
        !fn::head::greedy_decide(1, da_card, k.st) && !fn::head::greedy_decide(1, da_err, k.st) &&
        cuda_ok(cudaStreamSynchronize(k.st), "rejects");
    return report("rejects bad B, n, id0, card, alignment, null slot/error/work/stream", ok, ok ? "yes" : "no");
}

} // namespace

int fnfl_v_test(int dev0, int dev1, const char * mode, const char * model) {
    const std::string md = mode;
    const bool san = md == "san";
    if (!san && md != "all") {
        fprintf(stderr, "v-test: mode is all or san\n");
        return 2;
    }
    if (dev0 == dev1) {
        printf("v-test: loopback on device %d: both cards' kernels on one stream, sends before reads\n", dev0);
    } else if (!fn::xchg::enable_peer_access(dev0, dev1)) {
        printf("v-test: FAIL (no peer access)\n");
        return 1;
    }
    std::vector<std::string> paths;
    std::vector<fnfl_gguf_dir> dirs;
    std::string err;
    if (!fnfl_gguf_shard_paths(model, paths, err)) {
        fprintf(stderr, "v-test: %s\n", err.c_str());
        return 1;
    }
    for (const std::string & p : paths) {
        dirs.emplace_back();
        if (!fnfl_gguf_read_dir(p, dirs.back(), err)) {
            fprintf(stderr, "v-test: %s\n", err.c_str());
            return 1;
        }
    }
    head_weights hw;
    std::vector<uint8_t> raw[2];
    bool ok =
        read_head(dirs, hw) && read_vocab_slice(dirs, 0, n_slice, raw[0]) && read_vocab_slice(dirs, 1, n_slice, raw[1]);
    if (!ok) {
        printf("v-test: FAIL (weights)\n");
        return 1;
    }
    bool pass = true;
    {
        card_v k0(dev0), k1(dev1);
        card_v * c[2] = {&k0, &k1};
        for (int32_t r = 0; r < 2 && ok; ++r) {
            card_v & k = *c[r];
            ok = cuda_ok(cudaSetDevice(k.dev), "cudaSetDevice") && k.st != nullptr &&
                 k.w_down.put(hw.w_down[r].data(), hw.w_down[r].size(), k.st) &&
                 k.w_up.put(hw.w_up[r].data(), hw.w_up[r].size(), k.st) &&
                 k.gamma.put(hw.gamma_card[r].data(), hw.gamma_card[r].size(), k.st) && k.lwork.zero(k.st) &&
                 k.gwork.zero(k.st) && cuda_ok(cudaStreamSynchronize(k.st), "weights");
        }
        fn::xchg::rings rg(dev0, dev1);
        ok = ok && rg.ok() && put_vocab(c, raw) && check_rejects(k0, rg);
        pass = ok;
        std::vector<float> wdec[2] = {decode_slice(raw[0], n_slice), decode_slice(raw[1], n_slice)};

        // the end-to-end cases: plain inputs at B = 1..4 (rows 0..B-1 of one input set), then stress at B = 4
        for (int32_t kind = 0; kind < 2 && ok; ++kind) {
            inputs in;
            make_inputs(in, kind == 0 ? 4711u : 4712u, kind == 1);
            head_ref ref;
            make_head_ref(hw, in, ref);
            std::vector<float> hres[2], hx[2], hpost;
            round_out o4;
            bool have4 = false;
            int32_t k = 0;
            ok = rg.reset_all();
            for (int32_t B = kind == 0 ? 1 : R; B <= R && ok; ++B) {
                if (san && B != 1 && B != 3) {
                    continue;
                }
                round_cfg rc;
                rc.rows = B;
                rc.slot = k % fn::xchg::n_slot;
                rc.prev = k == 0 ? -1 : (k - 1) % fn::xchg::n_slot;
                ++k;
                round_out o;
                ok = put_inputs(c, in, hres, hx, hpost) && enqueue_round(c, rg, rc) && read_round(c, rg, rc, o);
                if (!ok) {
                    break;
                }
                const std::string label = std::string(kind == 0 ? "plain" : "stress") + " B" + std::to_string(B);
                pass = check_head_round(ref, wdec, rc, o, hres, hx, hpost, label.c_str()) && pass;
                if (B == R) {
                    o4 = o;
                    have4 = true;
                }
            }
            if (!ok || san || !have4) {
                continue;
            }
            // batch invariance: every row of B = 1..R-1 bitwise equal to the same row in B = R (full row, logits,
            // pairs, token), and a bitwise rerun of B = R in another slot
            int32_t inv_rows = 0, inv_same = 0;
            for (int32_t B = 1; B < R && kind == 0 && ok; ++B) {
                round_cfg rc;
                rc.rows = B;
                rc.slot = k % fn::xchg::n_slot;
                rc.prev = (k - 1) % fn::xchg::n_slot;
                ++k;
                round_out o;
                ok = put_inputs(c, in, hres, hx, hpost) && enqueue_round(c, rg, rc) && read_round(c, rg, rc, o);
                for (int32_t b = 0; b < B && ok; ++b) {
                    bool same = true;
                    const size_t p = 2 * size_t(b);
                    for (int32_t r = 0; r < 2; ++r) {
                        same = same && same_rows(o.full[r], b, o4.full[r], b, D) &&
                               same_rows(o.logits[r], b, o4.logits[r], b, n_slice) && o.pair[r][p] == o4.pair[r][p] &&
                               o.pair[r][p + 1] == o4.pair[r][p + 1] && o.token[r][b] == o4.token[r][b];
                    }
                    ++inv_rows;
                    inv_same += same ? 1 : 0;
                }
            }
            if (kind == 0) {
                pass = report(("plain: rows of B1..B" + std::to_string(R - 1) + " bitwise equal to the same rows in B" +
                               std::to_string(R))
                                  .c_str(),
                              inv_same == inv_rows,
                              std::to_string(inv_same) + " of " + std::to_string(inv_rows)) &&
                       pass;
            }
            round_cfg rc;
            rc.rows = R;
            rc.slot = k % fn::xchg::n_slot;
            rc.prev = (k - 1) % fn::xchg::n_slot;
            ++k;
            round_out o;
            ok = ok && put_inputs(c, in, hres, hx, hpost) && enqueue_round(c, rg, rc) && read_round(c, rg, rc, o);
            bool rerun = ok;
            for (int32_t r = 0; r < 2 && ok; ++r) {
                rerun = rerun && o.x[r] == o4.x[r] &&
                        memcmp(o.mom[r].data(), o4.mom[r].data(), o.mom[r].size() * 4) == 0 &&
                        memcmp(o.full[r].data(), o4.full[r].data(), o.full[r].size() * 4) == 0 &&
                        memcmp(o.logits[r].data(), o4.logits[r].data(), o.logits[r].size() * 4) == 0 &&
                        o.pair[r] == o4.pair[r] && o.token[r] == o4.token[r];
            }
            pass = report((std::string(kind == 0 ? "plain B" : "stress B") + std::to_string(R) +
                           ": rerun bitwise (another slot, X-HC reset path)")
                              .c_str(),
                          rerun, rerun ? "yes" : "no") &&
                   pass;
        }

        // planted ties and NaN rows in the vocabulary slice (B = R, plain inputs): NaN rows first, then exact
        // duplicates of each row's winner at a lower ID on its card, and of card 0 winners on card 1 (a tie across
        // the cards); then every row NaN
        if (ok) {
            inputs in;
            make_inputs(in, 4711u, false);
            head_ref ref;
            make_head_ref(hw, in, ref);
            std::vector<float> hres[2], hx[2], hpost;
            std::vector<uint8_t> planted[2] = {raw[0], raw[1]};
            const int32_t nan_rows[2][3] = {{0, 1, 4000}, {0, 2, 6000}};
            for (int32_t r = 0; r < 2; ++r) {
                for (const int32_t i : nan_rows[r]) {
                    plant_nan_row(planted[r], i);
                }
            }
            round_cfg rc;
            rc.rows = san ? 3 : R;
            ok = rg.reset_all() && put_vocab(c, planted) && put_inputs(c, in, hres, hx, hpost) &&
                 enqueue_round(c, rg, rc);
            round_out oa;
            ok = ok && read_round(c, rg, rc, oa);
            int32_t nan_seen = 0;
            for (int32_t r = 0; r < 2 && ok; ++r) {
                for (int32_t b = 0; b < rc.rows; ++b) {
                    for (const int32_t i : nan_rows[r]) {
                        nan_seen += std::isnan(oa.logits[r][size_t(b) * n_slice + i]) ? 1 : 0;
                    }
                }
            }
            pass = report("planted NaN rows: those logits are NaN", nan_seen == 2 * 3 * rc.rows,
                          std::to_string(nan_seen) + " of " + std::to_string(2 * 3 * rc.rows)) &&
                   pass;
            const greedy_check ga = ok ? check_greedy(rc, oa, "planted NaN rows") : greedy_check{};
            pass = ga.pass && pass;
            // the duplicates, from round A's own logits
            std::vector<int32_t> winners[2];
            for (int32_t r = 0; r < 2 && ok; ++r) {
                for (int32_t b = 0; b < rc.rows; ++b) {
                    pick p;
                    scan(&oa.logits[r][size_t(b) * n_slice], n_slice, 0, p);
                    winners[r].push_back(p.id);
                }
            }
            int32_t planted_within = 0, planted_across = 0;
            std::vector<int32_t> used[2] = {{0, 1, 2, 4000, 6000}, {0, 1, 2, 4000, 6000}};
            auto is_used = [&](int32_t r, int32_t i) {
                return std::find(used[r].begin(), used[r].end(), i) != used[r].end() ||
                       std::find(winners[r].begin(), winners[r].end(), i) != winners[r].end();
            };
            for (int32_t b = 0; b < rc.rows && ok; ++b) {
                for (int32_t r = 0; r < 2; ++r) {
                    const int32_t w = winners[r][b];
                    int32_t p = 3 + 5 * b + r;
                    while (p < w && is_used(r, p)) {
                        ++p;
                    }
                    if (w > 0 && p < w) {
                        memcpy(&planted[r][size_t(p) * q5_row_bytes], &raw[r][size_t(w) * q5_row_bytes], q5_row_bytes);
                        used[r].push_back(p);
                        ++planted_within;
                    }
                }
                // across: card 0's winner of row b copied to card 1 (its IDs are higher, so card 0 must win)
                int32_t q = n_slice - 1 - 3 * b;
                while (is_used(1, q)) {
                    --q;
                }
                memcpy(&planted[1][size_t(q) * q5_row_bytes], &raw[0][size_t(winners[0][b]) * q5_row_bytes],
                       q5_row_bytes);
                used[1].push_back(q);
                ++planted_across;
            }
            round_out ob;
            rc.slot = 1;
            ok = ok && put_vocab(c, planted) && put_inputs(c, in, hres, hx, hpost) && enqueue_round(c, rg, rc) &&
                 read_round(c, rg, rc, ob);
            const greedy_check gb = ok ? check_greedy(rc, ob, "planted ties") : greedy_check{};
            pass = gb.pass && pass;
            // every row's maximum is now tied; the rule then gives each row's lower ID, and both cards agree
            int32_t tied = 0;
            for (int32_t b = 0; b < rc.rows && ok; ++b) {
                tied += gb.tied[b] ? 1 : 0;
            }
            pass = report("planted ties realized (the maximum attained at 2+ IDs)", ok && tied == rc.rows,
                          std::to_string(tied) + " of " + std::to_string(rc.rows) + " rows (" +
                              std::to_string(gb.cross_ties) + " decided across the cards); " +
                              std::to_string(planted_within) + " within-card and " + std::to_string(planted_across) +
                              " cross-card copies") &&
                   pass;
            pass = report("planted NaN rows and ties: error words",
                          ok && (oa.err[0] | oa.err[1] | ob.err[0] | ob.err[1]) == 0u,
                          std::to_string(oa.err[0]) + " " + std::to_string(oa.err[1]) + " / " +
                              std::to_string(ob.err[0]) + " " + std::to_string(ob.err[1])) &&
                   pass;
            // all NaN: every row of both slices
            for (int32_t r = 0; r < 2; ++r) {
                for (int32_t i = 0; i < n_slice; ++i) {
                    plant_nan_row(planted[r], i);
                }
            }
            round_out oc;
            rc.slot = 2;
            ok = ok && put_vocab(c, planted) && put_inputs(c, in, hres, hx, hpost) && enqueue_round(c, rg, rc) &&
                 read_round(c, rg, rc, oc);
            int32_t minus = 0;
            for (int32_t b = 0; b < rc.rows && ok; ++b) {
                minus += oc.token[0][b] == fn::head::no_token && oc.token[1][b] == fn::head::no_token ? 1 : 0;
            }
            pass = report("all-NaN logits: token -1 on both cards", ok && minus == rc.rows,
                          std::to_string(minus) + " of " + std::to_string(rc.rows)) &&
                   pass;
            pass = report("all-NaN logits: err_all_nan reported on both cards",
                          ok && oc.err[0] == fn::head::err_all_nan && oc.err[1] == fn::head::err_all_nan,
                          std::to_string(oc.err[0]) + " " + std::to_string(oc.err[1])) &&
                   pass;
            ok = ok && rg.reset_all() && put_vocab(c, raw);
        }

        // greedy on synthetic full-shard logits
        pass = ok && run_greedy_cases(c, rg, san) && pass;
        // DR1's draft head: the re-shard's maps, the packed copy and greedy with the ID map
        pass = ok && run_draft_head_cases(c, rg, dirs, hw, san) && pass;
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(c[r]->dev);
            ok = cuda_ok(cudaDeviceSynchronize(), "final sync") && ok;
        }
    }
    pass = pass && ok;
    printf("v-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
