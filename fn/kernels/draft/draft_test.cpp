#include "fn/kernels/draft/draft_test.h"

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/draft/draft.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/shapes.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/quant/synth.h"
#include "fn/xchg/xchg.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// The references are written from the pinned lines (qwen4exp.cpp build_mtp, lines 686-1000 at the REL2 engine), never
// from a GPU kernel (R16):
//   734-747: h_norm = rms_norm(h) over each stream's 2,560 features, times hnorm (10,240: stream-major); e_norm =
//       rms_norm(e) times enorm, repeated to the 4 streams; 747: concat [e_norm | h_norm[s]] a stream.
//   855-900: q = rms_norm(q part) * attn_q_norm, gate = the head's second 256, k = rms_norm(k) * attn_k_norm, v not
//       normed; q and k through ggml_rope_multi (IMROPE, the target's rope: QA2's reference, copied below).
//   903-938: kq_scale = 1 / sqrt(256) (no attention.scale), the K and V store, attention over the cells, the output
//       times sigmoid(gate).
// rms_norm: x / sqrt(mean(x^2) + 1e-6). The rope angles use the kernel's F32 product chain (part of the model's
// arithmetic, as QA2's qa2-test); everything else is FP64. The attention reads the F16 K and V the kernel wrote.

namespace {

namespace dr = fn::draft;
namespace xc = fn::xchg;

constexpr int32_t HD = dr::head_dim;
constexpr int32_t NH = dr::n_head;
constexpr int32_t QS = dr::q_stride;
constexpr int32_t H = dr::half;
constexpr int32_t W = xc::words_draft;
constexpr double in_bound = 1e-6;
constexpr double attn_bound = 2e-3; // FC-DR1: NC1 section 2's attention bound for DA-TC
constexpr double f32_bound = 1e-6;
constexpr double eh_bound = 1e-5;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "draft-kernel-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// device memory on the current device
template <typename T> struct dev_buf {
    T * p = nullptr;
    size_t n = 0;

    explicit dev_buf(size_t count) : n(count > 0 ? count : 1) {
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

    bool put(const T * h, size_t count, cudaStream_t st, size_t at_elem = 0) {
        return p != nullptr && at_elem + count <= n &&
               cuda_ok(cudaMemcpyAsync(p + at_elem, h, count * sizeof(T), cudaMemcpyHostToDevice, st), "upload");
    }

    bool put(const std::vector<T> & h, cudaStream_t st) { return put(h.data(), h.size(), st); }

    bool get(T * h, size_t count, cudaStream_t st, size_t at_elem = 0) const {
        return p != nullptr && at_elem + count <= n &&
               cuda_ok(cudaMemcpyAsync(h, p + at_elem, count * sizeof(T), cudaMemcpyDeviceToHost, st), "download") &&
               cuda_ok(cudaStreamSynchronize(st), "download sync");
    }

    bool get(std::vector<T> & h, size_t count, cudaStream_t st, size_t at_elem = 0) const {
        h.resize(count);
        return get(h.data(), count, st, at_elem);
    }
};

// a fixed sequence (splitmix64): the cases are reproducible
struct rng {
    uint64_t s;

    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    float uniform(float lo, float hi) {
        return lo + (hi - lo) * static_cast<float>(next() >> 40) * (1.0f / 16777216.0f);
    }

    void fill(std::vector<float> & v, size_t n, float lo, float hi) {
        v.resize(n);
        for (float & x : v) {
            x = uniform(lo, hi);
        }
    }
};

uint16_t f16_rn(float f) {
    return __half_as_ushort(__float2half_rn(f));
}

float from_f16(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

struct tally {
    int64_t checks = 0, failed = 0;
    double in_err = 0.0, attn_err = 0.0, attn_err64 = 0.0, k_ulp = 0.0, f32_err = 0.0, eh_err = 0.0;
    std::string attn_at;

    void check(bool ok, const std::string & what) {
        ++checks;
        if (!ok) {
            ++failed;
            printf("    FAIL %s\n", what.c_str());
        }
    }
};

double norm_err(const double * ref, const float * got, int32_t n) {
    double rmax = 0.0, emax = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        rmax = std::max(rmax, std::fabs(ref[i]));
        const double e = std::fabs(double(got[i]) - ref[i]);
        emax = std::isnan(e) ? INFINITY : std::max(emax, e);
    }
    return rmax > 0.0 ? emax / rmax : emax;
}

// ---- X-DRAFT-IN ----

struct card_in {
    // [4][5120], [4][4][1280], [4][1280] x2, [1280], [4][1280]
    std::vector<float> emb, carry, carry_prev, h_nextn, g_e, g_h;
};

struct in_dev {
    cudaStream_t st = nullptr;
    dev_buf<float> emb{size_t(dr::max_rows) * dr::n_stream * H}, carry{size_t(dr::max_rows) * dr::n_stream * H};
    dev_buf<float> carry_prev{size_t(dr::n_stream) * H}, h_nextn{size_t(dr::n_stream) * H};
    dev_buf<float> g_e{size_t(H)}, g_h{size_t(dr::n_stream) * H}, own{size_t(dr::max_rows) * W};
    dev_buf<int32_t> hsrc{1};
    dev_buf<int8_t> codes{size_t(dr::max_rows) * dr::n_stream * dr::in_width};
    dev_buf<float> scales{size_t(dr::max_rows) * dr::n_stream * dr::in_runs * 2};
    dev_buf<float> f32{size_t(dr::max_rows) * dr::n_stream * dr::in_width};

    bool ready() const {
        return emb.p && carry.p && carry_prev.p && h_nextn.p && g_e.p && g_h.p && own.p && hsrc.p && codes.p &&
               scales.p && f32.p;
    }
};

struct in_case {
    std::string name;
    int32_t rows = 1, hsrc = 0, slot = 0;
    bool chained = false, update = false;
};

struct in_out {
    std::vector<float> f32, scales, carry_prev;
    std::vector<int8_t> codes;
    std::vector<uint32_t> slot;
    uint32_t err = 0;
};

// the hidden row of a case (draft.h): chained, else row hsrc + i - 1 of the carry, the retained carry when negative
const float * hidden_ref(const card_in & c, const in_case & k, int32_t i) {
    if (k.chained) {
        return c.h_nextn.data();
    }
    const int32_t j = k.hsrc + i - 1;
    return j < 0 ? c.carry_prev.data() : c.carry.data() + size_t(j) * dr::n_stream * H;
}

bool run_in(in_dev * d[2], xc::rings & rg, const card_in (&ci)[2], const in_case & k, in_out (&o)[2]) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(rg.device(c)), "set device") || !d[c]->emb.put(ci[c].emb, d[c]->st) ||
            !d[c]->carry.put(ci[c].carry, d[c]->st) || !d[c]->carry_prev.put(ci[c].carry_prev, d[c]->st) ||
            !d[c]->h_nextn.put(ci[c].h_nextn, d[c]->st) || !d[c]->g_e.put(ci[c].g_e, d[c]->st) ||
            !d[c]->g_h.put(ci[c].g_h, d[c]->st) || !d[c]->hsrc.put(&k.hsrc, 1, d[c]->st) ||
            !cuda_ok(cudaMemsetAsync(d[c]->f32.p, 0xff, d[c]->f32.n * 4, d[c]->st), "junk") ||
            !cuda_ok(cudaStreamSynchronize(d[c]->st), "upload")) {
            return false;
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        dr::send_args a = {};
        a.emb = d[c]->emb.p;
        a.emb_stride = dr::n_stream * H;
        a.carry = d[c]->carry.p;
        a.carry_prev = d[c]->carry_prev.p;
        a.h_nextn = d[c]->h_nextn.p;
        a.hsrc = d[c]->hsrc.p;
        a.chained = k.chained;
        a.g_e = d[c]->g_e.p;
        a.g_h = d[c]->g_h.p;
        a.peer = rg.slot(1 - c, xc::msg::draft, k.slot);
        a.own = d[c]->own.p;
        if (!cuda_ok(cudaSetDevice(rg.device(c)), "set device") || !dr::draft_in_send(k.rows, a, d[c]->st)) {
            fprintf(stderr, "draft-kernel-test: %s: send not enqueued\n", k.name.c_str());
            return false;
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        dr::recv_args a = {};
        a.own = d[c]->own.p;
        a.slot = rg.slot(c, xc::msg::draft, k.slot);
        a.codes = d[c]->codes.p;
        a.scales = d[c]->scales.p;
        a.f32 = d[c]->f32.p;
        a.err = rg.err(c);
        a.card = c;
        if (k.update) {
            a.carry_prev = d[c]->carry_prev.p;
            a.carry = d[c]->carry.p;
            a.hsrc = d[c]->hsrc.p;
        }
        if (!cuda_ok(cudaSetDevice(rg.device(c)), "set device") || !dr::draft_in_recv(k.rows, a, d[c]->st)) {
            fprintf(stderr, "draft-kernel-test: %s: recv not enqueued\n", k.name.c_str());
            return false;
        }
    }
    const size_t n_rows = size_t(k.rows) * dr::n_stream;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(rg.device(c)), "set device") ||
            !cuda_ok(cudaStreamSynchronize(d[c]->st), "draft-in") ||
            !d[c]->f32.get(o[c].f32, n_rows * dr::in_width, d[c]->st) ||
            !d[c]->codes.get(o[c].codes, n_rows * dr::in_width, d[c]->st) ||
            !d[c]->scales.get(o[c].scales, n_rows * dr::in_runs * 2, d[c]->st) ||
            !d[c]->carry_prev.get(o[c].carry_prev, size_t(dr::n_stream) * H, d[c]->st) || !rg.read_err(c, o[c].err)) {
            return false;
        }
        o[c].slot.resize(size_t(k.rows) * W);
        if (!cuda_ok(cudaMemcpy(o[c].slot.data(), rg.slot(c, xc::msg::draft, k.slot), o[c].slot.size() * 4,
                                cudaMemcpyDeviceToHost),
                     "slot")) {
            return false;
        }
    }
    return true;
}

void check_in(const card_in (&ci)[2], const in_case & k, const in_out (&o)[2], tally & t) {
    std::vector<double> ref(dr::in_width);
    for (int32_t c = 0; c < 2; ++c) {
        t.check(o[c].err == 0, k.name + ": card " + std::to_string(c) + " error word " + std::to_string(o[c].err));
        bool reset = true;
        for (const uint32_t w : o[c].slot) {
            reset = reset && w == xc::sentinel;
        }
        t.check(reset, k.name + ": card " + std::to_string(c) + "'s slot not reset to the sentinel");
    }
    for (int32_t i = 0; i < k.rows; ++i) {
        // the full rows: e = [card 0's own half | card 1's], h[s] likewise
        double sse = 0.0, ssh[dr::n_stream] = {};
        for (int32_t c = 0; c < 2; ++c) {
            const float * e = ci[c].emb.data() + size_t(i) * dr::n_stream * H;
            const float * h = hidden_ref(ci[c], k, i);
            for (int32_t f = 0; f < H; ++f) {
                sse += double(e[f]) * double(e[f]);
                for (int32_t s = 0; s < dr::n_stream; ++s) {
                    ssh[s] += double(h[s * H + f]) * double(h[s * H + f]);
                }
            }
        }
        const double re = 1.0 / std::sqrt(sse / fn::n_embd + double(fn::rms_eps));
        for (int32_t s = 0; s < dr::n_stream; ++s) {
            const double rh = 1.0 / std::sqrt(ssh[s] / fn::n_embd + double(fn::rms_eps));
            for (int32_t c = 0; c < 2; ++c) {
                const float * e = ci[c].emb.data() + size_t(i) * dr::n_stream * H;
                const float * h = hidden_ref(ci[c], k, i);
                for (int32_t f = 0; f < H; ++f) {
                    ref[size_t(c * H + f)] = double(e[f]) * re * double(ci[c].g_e[size_t(f)]);
                    ref[size_t(2 * H + c * H + f)] = double(h[s * H + f]) * rh * double(ci[c].g_h[size_t(s * H + f)]);
                }
            }
            const size_t r = size_t(i) * dr::n_stream + size_t(s);
            const double err = norm_err(ref.data(), o[0].f32.data() + r * dr::in_width, dr::in_width);
            t.in_err = std::max(t.in_err, err);
            t.check(err <= in_bound, k.name + ": row " + std::to_string(i) + " stream " + std::to_string(s) +
                                         " outside 1e-6 (" + std::to_string(err) + ")");
        }
    }
    const int32_t n_rows = k.rows * dr::n_stream;
    std::vector<int8_t> rc(size_t(n_rows) * dr::in_width);
    std::vector<float> rs(size_t(n_rows) * dr::in_runs * 2);
    fn::quant::act_quantize_ref(o[0].f32.data(), n_rows, dr::in_width, rc.data(), rs.data());
    t.check(rc == o[0].codes && memcmp(rs.data(), o[0].scales.data(), rs.size() * 4) == 0,
            k.name + ": the codes are not act_ref of the F32 rows");
    t.check(memcmp(o[0].f32.data(), o[1].f32.data(), o[0].f32.size() * 4) == 0 && o[0].codes == o[1].codes &&
                memcmp(o[0].scales.data(), o[1].scales.data(), o[0].scales.size() * 4) == 0,
            k.name + ": the cards' rows differ");
    for (int32_t c = 0; c < 2; ++c) {
        const float * want = ci[c].carry_prev.data();
        if (k.update) {
            want = ci[c].carry.data() + size_t(k.hsrc + k.rows - 1) * dr::n_stream * H;
        }
        t.check(memcmp(o[c].carry_prev.data(), want, size_t(dr::n_stream) * H * 4) == 0,
                k.name + ": card " + std::to_string(c) + "'s retained carry is not " +
                    (k.update ? "the round's last hidden row" : "unchanged"));
    }
}

bool test_in(int32_t dev0, int32_t dev1, tally & t) {
    if (!xc::enable_peer_access(dev0, dev1)) {
        return false;
    }
    xc::rings rg(dev0, dev1);
    if (!rg.ok()) {
        fprintf(stderr, "draft-kernel-test: rings\n");
        return false;
    }
    cudaSetDevice(dev0);
    in_dev d0;
    cudaStreamCreateWithFlags(&d0.st, cudaStreamNonBlocking);
    cudaSetDevice(dev1);
    in_dev d1;
    cudaStreamCreateWithFlags(&d1.st, cudaStreamNonBlocking);
    in_dev * d[2] = {&d0, &d1};
    bool ok = d0.ready() && d1.ready() && d0.st != nullptr && d1.st != nullptr;
    rng r{0x44523143ull};
    card_in ci[2];
    for (card_in & c : ci) {
        r.fill(c.emb, size_t(dr::max_rows) * dr::n_stream * H, -0.08f, 0.08f);
        r.fill(c.carry, size_t(dr::max_rows) * dr::n_stream * H, -6.0f, 6.0f);
        r.fill(c.carry_prev, size_t(dr::n_stream) * H, -3.0f, 3.0f);
        r.fill(c.h_nextn, size_t(dr::n_stream) * H, -12.0f, 12.0f);
        r.fill(c.g_e, size_t(H), 0.2f, 2.5f);
        r.fill(c.g_h, size_t(dr::n_stream) * H, 0.05f, 1.5f);
    }
    // a large-magnitude stream and a zero-ish one: the sums of squares span orders of magnitude
    for (int32_t f = 0; f < H; ++f) {
        ci[0].carry[size_t(2 * H + f)] *= 40.0f;
        ci[1].carry[size_t(2 * H + f)] *= 40.0f;
        ci[0].carry[size_t(5 * H + f)] *= 1e-3f;
        ci[1].carry[size_t(5 * H + f)] *= 1e-3f;
    }
    const in_case cases[] = {
        {"draft-in step, hsrc 0 (retained carry)", 1, 0, 0, false, false},
        {"draft-in step, hsrc 3", 1, 3, 1, false, false},
        {"draft-in step, chained", 1, 0, 2, true, false},
        {"draft-in catch-up B_c 1, hsrc 0", 1, 0, 0, false, true},
        {"draft-in catch-up B_c 2, hsrc 0", 2, 0, 1, false, true},
        {"draft-in catch-up B_c 2, hsrc 2", 2, 2, 2, false, true},
        {"draft-in catch-up B_c 3, hsrc 1", 3, 1, 0, false, true},
        {"draft-in catch-up B_c 4, hsrc 0", 4, 0, 1, false, true},
        {"draft-in rows 4, no update", 4, 0, 2, false, false},
        {"draft-in catch-up B_c 5, hsrc 0 (SP2)", 5, 0, 1, false, true},
        {"draft-in rows 5, no update (SP2)", 5, 0, 2, false, false},
    };
    for (const in_case & k : cases) {
        if (!ok) {
            break;
        }
        in_out o[2], o2[2];
        ok = run_in(d, rg, ci, k, o) && run_in(d, rg, ci, k, o2);
        if (ok) {
            check_in(ci, k, o, t);
            t.check(memcmp(o[0].f32.data(), o2[0].f32.data(), o[0].f32.size() * 4) == 0 && o[0].codes == o2[0].codes,
                    k.name + ": a rerun is not bitwise identical");
        }
    }
    // FMU2 (build item 2, the joint draft step): rows 2 .. 4 at hsrc 1, not chained (row i's hidden: carry row i), each
    // row bitwise the same row alone at B = 1 in the lone step's two forms (step 0 on the retained carry; the later
    // steps chained on h_nextn): its F32 row, codes and scales on both cards
    int32_t n_alone = 0, n_alone_same = 0;
    for (int32_t U = 2; U <= 4 && ok; ++U) {
        const in_case kj = {"draft-in joint rows " + std::to_string(U), U, 1, U % 3, false, false};
        in_out oj[2];
        ok = run_in(d, rg, ci, kj, oj);
        for (int32_t i = 0; i < U && ok; ++i) {
            for (const bool chained : {false, true}) {
                card_in c1[2] = {ci[0], ci[1]};
                const size_t row = size_t(dr::n_stream) * H;
                for (int32_t c = 0; c < 2; ++c) {
                    std::copy_n(ci[c].emb.begin() + std::ptrdiff_t(size_t(i) * row), row, c1[c].emb.begin());
                    std::copy_n(ci[c].carry.begin() + std::ptrdiff_t(size_t(i) * row), row,
                                (chained ? c1[c].h_nextn : c1[c].carry_prev).begin());
                }
                const in_case k1 = {"draft-in row alone", 1, 0, (i + 1) % 3, chained, false};
                in_out o1[2];
                ok = run_in(d, rg, c1, k1, o1);
                bool same = ok;
                const size_t nf = size_t(dr::n_stream) * dr::in_width, ns = size_t(dr::n_stream) * dr::in_runs * 2;
                for (int32_t c = 0; c < 2 && ok; ++c) {
                    same = same && memcmp(oj[c].f32.data() + size_t(i) * nf, o1[c].f32.data(), nf * 4) == 0 &&
                           memcmp(oj[c].codes.data() + size_t(i) * nf, o1[c].codes.data(), nf) == 0 &&
                           memcmp(oj[c].scales.data() + size_t(i) * ns, o1[c].scales.data(), ns * 4) == 0 &&
                           oj[c].err == 0u && o1[c].err == 0u;
                }
                ++n_alone;
                n_alone_same += same ? 1 : 0;
                t.check(same, kj.name + ": row " + std::to_string(i) + (chained ? " (chained)" : " (retained carry)") +
                                  " is not bitwise the row alone");
            }
        }
    }
    printf("draft-kernel-test: FMU2 joint rows 2 .. 4 (hsrc 1), each row against the row alone in both lone forms: %d "
           "of %d bitwise\n",
           n_alone_same, n_alone);
    // rejects: nothing enqueued
    if (ok) {
        dr::send_args a = {};
        a.emb = d0.emb.p;
        a.emb_stride = dr::n_stream * H;
        a.carry = d0.carry.p;
        a.carry_prev = d0.carry_prev.p;
        a.h_nextn = d0.h_nextn.p;
        a.hsrc = d0.hsrc.p;
        a.g_e = d0.g_e.p;
        a.g_h = d0.g_h.p;
        a.peer = rg.slot(1, xc::msg::draft, 0);
        a.own = d0.own.p;
        cudaSetDevice(dev0);
        t.check(!dr::draft_in_send(0, a, d0.st) && !dr::draft_in_send(6, a, d0.st), "send rejects rows 0 and 6");
        a.chained = true;
        t.check(!dr::draft_in_send(2, a, d0.st), "send rejects a chained launch of 2 rows");
        dr::recv_args b = {};
        b.own = d0.own.p;
        b.slot = rg.slot(0, xc::msg::draft, 0);
        b.codes = d0.codes.p;
        b.scales = d0.scales.p;
        b.err = rg.err(0);
        b.card = 2;
        t.check(!dr::draft_in_recv(1, b, d0.st), "recv rejects card 2");
    }
    cudaSetDevice(dev0);
    cudaStreamDestroy(d0.st);
    cudaSetDevice(dev1);
    cudaStreamDestroy(d1.st);
    return ok;
}

// ---- DA-TC ----

// pair i's angle: the F32 chain and the IMROPE sector rule (QA2's qa2-test, ggml_mrope_cache_init)
float theta_f32(const int32_t (&pos4)[4], int i, float ts) {
    float th[4];
    for (int p = 0; p < 4; ++p) {
        th[p] = static_cast<float>(pos4[p]);
    }
    for (int k = 0; k < i; ++k) {
        for (int p = 0; p < 4; ++p) {
            th[p] *= ts;
        }
    }
    const int s = i;
    return s % 3 == 1 && s < 33 ? th[1] : s % 3 == 2 && s < 30 ? th[2] : s % 3 == 0 && s < 33 ? th[0] : th[3];
}

void norm_rope_ref(const float * x, const float * w, const int32_t (&pos4)[4], float ts, double * y) {
    double ss = 0.0;
    for (int d = 0; d < HD; ++d) {
        ss += double(x[d]) * double(x[d]);
    }
    const double r = 1.0 / std::sqrt(ss / HD + double(fn::rms_eps));
    for (int d = 0; d < HD; ++d) {
        y[d] = double(x[d]) * r * double(w[d]);
    }
    for (int i = 0; i < fn::qsa::n_rot / 2; ++i) {
        const double th = double(theta_f32(pos4, i, ts)), c = std::cos(th), s = std::sin(th);
        const double x0 = y[i], x1 = y[i + fn::qsa::n_rot / 2];
        y[i] = x0 * c - x1 * s;
        y[i + fn::qsa::n_rot / 2] = x0 * s + x1 * c;
    }
}

// one F16 ulp of y (subnormals: 2^-24)
double f16_ulp(double y) {
    if (y == 0.0) {
        return std::ldexp(1.0, -24);
    }
    return std::ldexp(1.0, std::max(std::ilogb(y), -14) - 10);
}

struct attn_card {
    int32_t dev = 0;
    cudaStream_t st = nullptr;
    int32_t n_cells = 0;
    dev_buf<uint16_t> kc, vc;
    dev_buf<float> qfull{size_t(dr::max_rows) * NH * QS}, kvk{size_t(dr::max_rows) * HD},
        kvv{size_t(dr::max_rows) * HD};
    dev_buf<float> qn{HD}, kn{HD}, qf32{size_t(dr::max_rows) * NH * HD}, kf32{size_t(dr::max_rows) * HD};
    dev_buf<uint16_t> q16{size_t(dr::max_rows) * NH * HD};
    dev_buf<int32_t> nb{1}, pos{size_t(4) * dr::max_rows}, stride{1};
    dev_buf<float> part{dr::part_bytes() / 4}, out{size_t(dr::out_width)}, scales{size_t(dr::out_width) / 16};
    dev_buf<int8_t> codes{size_t(dr::out_width)};
    dev_buf<uint32_t> err{1};
    dev_buf<float> rel_hc{size_t(xc::slot_words(xc::msg::hc))}, rel_hmix{size_t(xc::slot_words(xc::msg::hmix))};
    dev_buf<uint32_t> body{size_t(2 + dr::sel_blocks + dr::sel_ratio - 1)}; // SP2: a select body of one row

    explicit attn_card(int32_t cells) : n_cells(cells), kc(size_t(cells) * HD), vc(size_t(cells) * HD) {}

    bool ready() const {
        return kc.p && vc.p && qfull.p && kvk.p && kvv.p && qn.p && kn.p && qf32.p && kf32.p && q16.p && nb.p &&
               pos.p && stride.p && part.p && out.p && scales.p && codes.p && err.p && rel_hc.p && rel_hmix.p &&
               body.p;
    }
};

struct attn_case {
    std::string name;
    int32_t n_before = 0, rows = 1, row0 = 0, stride = 1;
    bool step = true;    // q and attend; else the catch-up (K and V only, with the release)
    bool planes = false; // distinct rope planes (else all four the position)
    bool sparse = false; // SP2: attend over a select body of random blocks of the prefix (the step only)
    int32_t bad_k = 0;   // SP2: added to the body's K (a body not made at this depth)
};

struct attn_in {
    std::vector<float> qfull, kv_k, kv_v; // [rows][12][512], [rows][256] x2
    std::vector<int32_t> pos;             // [4][stride]
    std::vector<uint32_t> body;           // SP2: the select body (sparse)
    std::vector<int32_t> cells;           // the attended cells in the kernel's logical order
};

struct attn_out {
    std::vector<float> out, scales, qf32, kf32;
    std::vector<int8_t> codes;
    std::vector<uint16_t> q16, k_new, v_new;
    std::vector<float> rel_hc, rel_hmix;
    uint32_t err = 0;
};

attn_in make_attn_in(const attn_case & k, rng & r) {
    attn_in in;
    r.fill(in.qfull, size_t(k.rows) * NH * QS, -3.0f, 3.0f);
    r.fill(in.kv_k, size_t(k.rows) * HD, -2.0f, 2.0f);
    r.fill(in.kv_v, size_t(k.rows) * HD, -1.5f, 1.5f);
    in.pos.assign(size_t(4) * k.stride, 0);
    for (int32_t t = 0; t < k.stride; ++t) {
        const int32_t p = k.n_before + t;
        for (int32_t pl = 0; pl < 4; ++pl) {
            in.pos[size_t(pl) * k.stride + t] = k.planes ? p + 17 * pl - (pl == 3 ? 3 * p / 4 : 0) : p;
        }
    }
    const int32_t n = k.n_before + k.row0 + 1;
    if (!k.sparse) {
        in.cells.resize(size_t(n));
        for (int32_t c = 0; c < n; ++c) {
            in.cells[size_t(c)] = c;
        }
        return in;
    }
    // SP2: K = min(N / 4, 512) distinct blocks of the N / 4 whole ones, ascending (all of them up to 512), then the
    // tail and the chain's cells 4 (N / 4) .. N + row0
    const int32_t n_vis = k.n_before / dr::sel_ratio, K = std::min(n_vis, dr::sel_blocks);
    std::vector<int32_t> blk(static_cast<size_t>(n_vis));
    for (int32_t b = 0; b < n_vis; ++b) {
        blk[size_t(b)] = b;
    }
    for (int32_t i = 0; i < K && n_vis > dr::sel_blocks; ++i) {
        const int32_t j = i + static_cast<int32_t>(r.next() % uint64_t(n_vis - i));
        std::swap(blk[size_t(i)], blk[size_t(j)]);
    }
    blk.resize(size_t(K));
    std::sort(blk.begin(), blk.end());
    in.body = {uint32_t(K + k.bad_k), uint32_t(k.n_before % dr::sel_ratio)};
    for (const int32_t b : blk) {
        in.body.push_back(uint32_t(b));
        for (int32_t c = 0; c < dr::sel_ratio; ++c) {
            in.cells.push_back(dr::sel_ratio * b + c);
        }
    }
    for (int32_t c = dr::sel_ratio * n_vis; c < k.n_before; ++c) {
        in.body.push_back(uint32_t(c));
    }
    for (int32_t c = dr::sel_ratio * n_vis; c < n; ++c) {
        in.cells.push_back(c);
    }
    return in;
}

bool run_attn(attn_card & c, const attn_case & k, const attn_in & in, attn_out & o) {
    const float ts = fn::qsa::attn::rope_theta_scale();
    const size_t cell = size_t(k.n_before) + size_t(k.row0);
    bool ok = cuda_ok(cudaSetDevice(c.dev), "set device") && c.nb.put(&k.n_before, 1, c.st) &&
              c.stride.put(&k.stride, 1, c.st) && c.pos.put(in.pos, c.st) && c.qfull.put(in.qfull, c.st) &&
              c.kvk.put(in.kv_k, c.st) && c.kvv.put(in.kv_v, c.st) &&
              cuda_ok(cudaMemsetAsync(c.out.p, 0xff, c.out.n * 4, c.st), "junk") &&
              cuda_ok(cudaMemsetAsync(c.err.p, 0, 4, c.st), "error word") &&
              cuda_ok(cudaMemsetAsync(c.rel_hc.p, 0, c.rel_hc.n * 4, c.st), "slot fill") &&
              cuda_ok(cudaMemsetAsync(c.rel_hmix.p, 0, c.rel_hmix.n * 4, c.st), "slot fill") &&
              (!k.sparse || c.body.put(in.body, c.st));
    if (ok && cell + size_t(k.rows) <= size_t(c.n_cells)) {
        // the new cells hold junk: prepare must write them
        ok = cuda_ok(cudaMemsetAsync(c.kc.p + cell * HD, 0x7f, size_t(k.rows) * HD * 2, c.st), "junk") &&
             cuda_ok(cudaMemsetAsync(c.vc.p + cell * HD, 0x7f, size_t(k.rows) * HD * 2, c.st), "junk");
    }
    dr::prepare_args p = {};
    p.qfull = k.step ? c.qfull.p : nullptr;
    p.kv_k = c.kvk.p;
    p.kv_v = c.kvv.p;
    p.q_norm = c.qn.p;
    p.k_norm = c.kn.p;
    p.n_before = c.nb.p;
    p.pos = c.pos.p;
    p.stride = c.stride.p;
    p.row0 = k.row0;
    p.theta_scale = ts;
    p.k_cache = c.kc.p;
    p.v_cache = c.vc.p;
    p.max_cells = c.n_cells;
    p.q16 = c.q16.p;
    p.q_f32 = c.qf32.p;
    p.k_f32 = c.kf32.p;
    p.err = c.err.p;
    p.release_hc = k.step ? nullptr : c.rel_hc.p;
    p.release_hmix = k.step ? nullptr : c.rel_hmix.p;
    if (!ok || !dr::datc_prepare(k.rows, p, c.st)) {
        fprintf(stderr, "draft-kernel-test: %s: prepare not enqueued\n", k.name.c_str());
        return false;
    }
    if (k.step) {
        dr::attend_args a = {};
        a.q16 = c.q16.p;
        a.qfull = c.qfull.p;
        a.k_cache = c.kc.p;
        a.v_cache = c.vc.p;
        a.n_before = c.nb.p;
        a.row0 = k.row0;
        a.max_cells = c.n_cells;
        a.part = c.part.p;
        a.out = c.out.p;
        a.codes = c.codes.p;
        a.scales = c.scales.p;
        a.err = c.err.p;
        a.body = k.sparse ? c.body.p : nullptr;
        if (!dr::datc_attend(a, c.st)) {
            fprintf(stderr, "draft-kernel-test: %s: attend not enqueued: %s\n", k.name.c_str(),
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
    }
    // the q, codes and scales only where the kernels write them (a step case with a good depth; check_attn reads them
    // there only): copying them otherwise reads memory nothing wrote (Compute Sanitizer initcheck)
    const bool attended = k.step && k.bad_k == 0;
    ok = cuda_ok(cudaStreamSynchronize(c.st), "DA-TC") && c.err.get(&o.err, 1, c.st) &&
         c.out.get(o.out, size_t(dr::out_width), c.st) &&
         (!attended || (c.codes.get(o.codes, size_t(dr::out_width), c.st) &&
                        c.scales.get(o.scales, size_t(dr::out_width) / 16, c.st) &&
                        c.qf32.get(o.qf32, size_t(k.rows) * NH * HD, c.st) &&
                        c.q16.get(o.q16, size_t(k.rows) * NH * HD, c.st))) &&
         c.kf32.get(o.kf32, size_t(k.rows) * HD, c.st) && c.rel_hc.get(o.rel_hc, c.rel_hc.n, c.st) &&
         c.rel_hmix.get(o.rel_hmix, c.rel_hmix.n, c.st);
    if (ok && cell + size_t(k.rows) <= size_t(c.n_cells)) {
        ok = c.kc.get(o.k_new, size_t(k.rows) * HD, c.st, cell * HD) &&
             c.vc.get(o.v_new, size_t(k.rows) * HD, c.st, cell * HD);
    }
    return ok;
}

struct attn_mirror {
    std::vector<uint16_t> k, v; // the caches as the cards hold them
    std::vector<float> q_norm, k_norm;
    float f16[65536];
};

// FP64 attention of one head over the listed cells of the mirror (dense: 0 .. n - 1; SP2's sparse: the selected
// blocks' cells, the tail and the chain): softmax(q . k / 16) V, times sigmoid(gate)
void attn_ref(const attn_mirror & m, const std::vector<double> & q, const float * gate,
              const std::vector<int32_t> & cells, std::vector<double> & s, std::vector<double> & out) {
    double mx = -std::numeric_limits<double>::infinity();
    const int32_t n = static_cast<int32_t>(cells.size());
    for (int32_t c = 0; c < n; ++c) {
        const uint16_t * kr = m.k.data() + int64_t(cells[size_t(c)]) * HD;
        double dot = 0.0;
        for (int32_t d = 0; d < HD; ++d) {
            dot += q[size_t(d)] * double(m.f16[kr[d]]);
        }
        s[size_t(c)] = dot / 16.0;
        mx = std::max(mx, s[size_t(c)]);
    }
    std::fill(out.begin(), out.end(), 0.0);
    double l = 0.0;
    for (int32_t c = 0; c < n; ++c) {
        const double p = std::exp(s[size_t(c)] - mx);
        l += p;
        const uint16_t * vr = m.v.data() + int64_t(cells[size_t(c)]) * HD;
        for (int32_t d = 0; d < HD; ++d) {
            out[size_t(d)] += p * double(m.f16[vr[d]]);
        }
    }
    for (int32_t d = 0; d < HD; ++d) {
        out[size_t(d)] = out[size_t(d)] / l * (1.0 / (1.0 + std::exp(-double(gate[d]))));
    }
}

void check_attn(attn_mirror & m, const attn_case & k, const attn_in & in, const attn_out & o, tally & t) {
    const float ts = fn::qsa::attn::rope_theta_scale();
    t.check(k.bad_k != 0 || o.err == 0, k.name + ": error word " + std::to_string(o.err));
    std::vector<double> y(HD);
    for (int32_t r = 0; r < k.rows; ++r) {
        const int32_t col = k.row0 + r;
        const int32_t p4[4] = {in.pos[size_t(col)], in.pos[size_t(k.stride + col)], in.pos[size_t(2 * k.stride + col)],
                               in.pos[size_t(3 * k.stride + col)]};
        norm_rope_ref(in.kv_k.data() + size_t(r) * HD, m.k_norm.data(), p4, ts, y.data());
        const double kerr = norm_err(y.data(), o.kf32.data() + size_t(r) * HD, HD);
        t.f32_err = std::max(t.f32_err, kerr);
        t.check(kerr <= f32_bound, k.name + ": the F32 k outside 1e-6 of FP64");
        // the cache holds f16(the F32 k) bitwise; the distance from the FP64 k in F16 ulps is reported for the values
        // of at least 1e-3 of the row's largest (smaller ones are dominated by the F32 row's normalized error)
        double ulps = 0.0, ymax = 0.0;
        bool k_ok = true, v_ok = true;
        for (int32_t d = 0; d < HD; ++d) {
            ymax = std::max(ymax, std::fabs(y[size_t(d)]));
        }
        for (int32_t d = 0; d < HD; ++d) {
            const double kb = double(from_f16(o.k_new[size_t(r) * HD + d]));
            if (std::fabs(y[size_t(d)]) >= 1e-3 * ymax) {
                ulps = std::max(ulps, std::fabs(kb - y[size_t(d)]) / f16_ulp(y[size_t(d)]));
            }
            k_ok = k_ok && o.k_new[size_t(r) * HD + d] == f16_rn(o.kf32[size_t(r) * HD + d]);
            v_ok = v_ok && o.v_new[size_t(r) * HD + d] == f16_rn(in.kv_v[size_t(r) * HD + d]);
        }
        t.k_ulp = std::max(t.k_ulp, ulps);
        t.check(k_ok, k.name + ": the K cache is not f16(the F32 k)");
        t.check(ulps <= 1.0, k.name + ": a K cache value more than one F16 ulp from the FP64 k");
        t.check(v_ok, k.name + ": the V cache is not f16(v)");
        const size_t cell = size_t(k.n_before) + size_t(col);
        std::copy(o.k_new.begin() + int64_t(r) * HD, o.k_new.begin() + int64_t(r + 1) * HD,
                  m.k.begin() + int64_t(cell) * HD);
        std::copy(o.v_new.begin() + int64_t(r) * HD, o.v_new.begin() + int64_t(r + 1) * HD,
                  m.v.begin() + int64_t(cell) * HD);
    }
    if (!k.step) {
        bool hc = true, hmix = true;
        for (const float f : o.rel_hc) {
            uint32_t u = 0;
            memcpy(&u, &f, 4);
            hc = hc && u == xc::sentinel;
        }
        for (size_t i = 0; i < o.rel_hmix.size(); ++i) {
            uint32_t u = 0;
            memcpy(&u, &o.rel_hmix[i], 4);
            const bool code = i >= size_t(xc::hmix_layout::code_at) && i < size_t(xc::hmix_layout::scale_at);
            hmix = hmix && u == (code ? xc::code_sentinel : xc::sentinel);
        }
        t.check(hc && hmix, k.name + ": the released X-HC or X-HMIX slot is not reset");
        return;
    }
    if (k.bad_k != 0) {
        bool junk = true;
        for (const float f : o.out) {
            uint32_t u = 0;
            memcpy(&u, &f, 4);
            junk = junk && u == 0xffffffffu;
        }
        t.check(o.err == xc::err_draft_depth && junk, k.name + ": err_draft_depth and no output");
        return;
    }
    // q: F32 against FP64, q16 bitwise f16(q / 16)
    const int32_t n = static_cast<int32_t>(in.cells.size());
    const int32_t p4[4] = {in.pos[size_t(k.row0)], in.pos[size_t(k.stride + k.row0)],
                           in.pos[size_t(2 * k.stride + k.row0)], in.pos[size_t(3 * k.stride + k.row0)]};
    std::vector<double> qd(HD), q16d(HD), s(static_cast<size_t>(n)), ref(HD), ref64(HD);
    bool q16_ok = true;
    for (int32_t h = 0; h < NH; ++h) {
        const float * qrow = in.qfull.data() + size_t(h) * QS; // the attended row is the case's first (rows 1)
        norm_rope_ref(qrow, m.q_norm.data(), p4, ts, qd.data());
        const double qerr = norm_err(qd.data(), o.qf32.data() + size_t(h) * HD, HD);
        t.f32_err = std::max(t.f32_err, qerr);
        t.check(qerr <= f32_bound, k.name + ": the F32 q outside 1e-6 of FP64");
        for (int32_t d = 0; d < HD; ++d) {
            q16_ok = q16_ok && o.q16[size_t(h) * HD + d] == f16_rn(o.qf32[size_t(h) * HD + d] * 0.0625f);
        }
        // NC1 section 3: FP64 on the stored inputs the kernel reads, the F16 q tile (q16 x 16, exact) and the F16 K
        // and V; the same attention from the FP64 q is reported beside it (it adds q16's rounding)
        for (int32_t d = 0; d < HD; ++d) {
            q16d[size_t(d)] = 16.0 * double(m.f16[o.q16[size_t(h) * HD + d]]);
        }
        attn_ref(m, q16d, qrow + HD, in.cells, s, ref);
        attn_ref(m, qd, qrow + HD, in.cells, s, ref64);
        const double err = norm_err(ref.data(), o.out.data() + size_t(h) * HD, HD);
        const double err64 = norm_err(ref64.data(), o.out.data() + size_t(h) * HD, HD);
        if (!(err <= t.attn_err)) {
            t.attn_err = err;
            t.attn_at = k.name + " head " + std::to_string(h);
        }
        t.attn_err64 = std::max(t.attn_err64, err64);
        t.check(err <= attn_bound,
                k.name + ": head " + std::to_string(h) + " outside 2e-3 (" + std::to_string(err) + ")");
    }
    t.check(q16_ok, k.name + ": q16 is not f16(q / 16)");
    std::vector<int8_t> rc(size_t(dr::out_width));
    std::vector<float> rs(size_t(dr::out_width) / 16);
    fn::quant::act_quantize_ref(o.out.data(), 1, dr::out_width, rc.data(), rs.data());
    t.check(rc == o.codes && memcmp(rs.data(), o.scales.data(), rs.size() * 4) == 0,
            k.name + ": the codes are not act_ref of the F32 output");
}

bool same_attn(const attn_out & a, const attn_out & b) {
    return memcmp(a.out.data(), b.out.data(), a.out.size() * 4) == 0 && a.codes == b.codes &&
           memcmp(a.scales.data(), b.scales.data(), a.scales.size() * 4) == 0 && a.k_new == b.k_new &&
           a.v_new == b.v_new && a.q16 == b.q16;
}

bool test_attn(int32_t dev0, int32_t dev1, bool san, tally & t) {
    const int32_t max_n = san ? 4096 : 196608;
    const int32_t cells = max_n + 16;
    attn_mirror m;
    for (uint32_t u = 0; u < 65536; ++u) {
        m.f16[u] = from_f16(static_cast<uint16_t>(u));
    }
    rng r{0x44415443ull};
    r.fill(m.q_norm, HD, 0.3f, 1.8f);
    r.fill(m.k_norm, HD, 0.3f, 1.8f);
    m.k.resize(size_t(cells) * HD);
    m.v.resize(size_t(cells) * HD);
    for (size_t i = 0; i < m.k.size(); ++i) {
        m.k[i] = f16_rn(r.uniform(-2.0f, 2.0f));
        m.v[i] = f16_rn(r.uniform(-1.5f, 1.5f));
    }
    // a band of large keys deep in the history: the softmax is dominated by a few cells far from the end
    for (int32_t c = std::min(1000, max_n / 3); c < std::min(1000, max_n / 3) + 3; ++c) {
        for (int32_t d = 0; d < HD; ++d) {
            m.k[size_t(c) * HD + d] = f16_rn(6.0f * from_f16(m.k[size_t(c) * HD + d]));
        }
    }
    cudaSetDevice(dev0);
    attn_card c0(cells);
    cudaStreamCreateWithFlags(&c0.st, cudaStreamNonBlocking);
    c0.dev = dev0;
    cudaSetDevice(dev1);
    attn_card c1(cells);
    cudaStreamCreateWithFlags(&c1.st, cudaStreamNonBlocking);
    c1.dev = dev1;
    attn_card * cs[2] = {&c0, &c1};
    bool ok = c0.ready() && c1.ready();
    for (attn_card * c : cs) {
        ok = ok && cuda_ok(cudaSetDevice(c->dev), "set device") && c->kc.put(m.k, c->st) && c->vc.put(m.v, c->st) &&
             c->qn.put(m.q_norm, c->st) && c->kn.put(m.k_norm, c->st) &&
             cuda_ok(cudaStreamSynchronize(c->st), "upload");
    }
    std::vector<attn_case> cases;
    const int32_t steps_all[] = {0, 1, 62, 63, 64, 127, 200, 4095, 9215, 9216, 9217, 58580, 131071, 196607};
    const int32_t steps_san[] = {0, 63, 64, 1000, 4095};
    for (const int32_t q :
         san ? std::vector<int32_t>(steps_san, steps_san + 5) : std::vector<int32_t>(steps_all, steps_all + 14)) {
        cases.push_back({"step n " + std::to_string(q + 1), q, 1, 0, 1, true, false});
    }
    // chained steps j = 1, 2 of a d = 3 round: row j's cell n_before + j, its position column j
    cases.push_back({"step d 3 j 1", 300, 1, 1, 3, true, false});
    cases.push_back({"step d 3 j 2 (planes)", 300, 1, 2, 3, true, true});
    cases.push_back({"step d 2 j 1 deep", max_n - 2, 1, 1, 2, true, false});
    cases.push_back({"step d 4 j 3 (planes, SP2)", 300, 1, 3, 4, true, true});
    for (int32_t b = 1; b <= dr::max_rows; ++b) {
        cases.push_back({"catch-up B_c " + std::to_string(b), 500 + 7 * b, b, 0, b, false, b == 3});
    }
    cases.push_back({"catch-up B_c 5 at the end", max_n + 11, 5, 0, 5, false, false}); // SP2: five rows
    // SP2's sparse attend: step j of a d = 4 chain over a select body at N = n_before (all blocks up to 512 of them,
    // then 512 random ones), the tail of N % 4 cells and the chain's j + 1
    const int32_t sp_all[] = {1, 7, 64, 2048, 2051, 2052, 2055, 9217, 58581, 131070, 196600};
    const int32_t sp_san[] = {6, 2052, 4000};
    int32_t sp_i = 0;
    const std::vector<int32_t> sp =
        san ? std::vector<int32_t>(sp_san, sp_san + 3) : std::vector<int32_t>(sp_all, sp_all + 11);
    for (const int32_t nb : sp) {
        const int32_t j = sp_i++ % dr::max_sparse_steps;
        cases.push_back(
            {"sparse N " + std::to_string(nb) + " j " + std::to_string(j), nb, 1, j, 4, true, j == 2, true});
    }
    cases.push_back({"sparse body K not this depth's", 3000, 1, 1, 4, true, false, true, 1});
    for (const attn_case & k : cases) {
        if (!ok) {
            break;
        }
        const attn_in in = make_attn_in(k, r);
        attn_out o0, o0b, o1;
        ok = run_attn(c0, k, in, o0) && run_attn(c0, k, in, o0b) && run_attn(c1, k, in, o1);
        if (ok) {
            check_attn(m, k, in, o0, t);
            t.check(same_attn(o0, o0b), k.name + ": a rerun is not bitwise identical");
            t.check(same_attn(o0, o1), k.name + ": the cards differ");
        }
    }
    // a cell outside the cache: the error word, nothing written
    if (ok) {
        const attn_case k{"depth past the cache", cells - 1, 2, 0, 2, false, false};
        const attn_in in = make_attn_in(k, r);
        attn_out o;
        std::vector<uint16_t> before, after;
        ok = c0.kc.get(before, HD, c0.st, size_t(cells - 1) * HD) && run_attn(c0, k, in, o) &&
             c0.kc.get(after, HD, c0.st, size_t(cells - 1) * HD);
        t.check(ok && o.err == xc::err_draft_depth && before == after,
                "prepare past the cache: err_draft_depth and no write");
        const attn_case ka{"attend past the cache", cells - 1, 1, 1, 2, true, false};
        const attn_in ina = make_attn_in(ka, r);
        attn_out oa;
        ok = ok && run_attn(c0, ka, ina, oa);
        bool junk = true;
        for (const float f : oa.out) {
            uint32_t u = 0;
            memcpy(&u, &f, 4);
            junk = junk && u == 0xffffffffu;
        }
        t.check(ok && oa.err == xc::err_draft_depth && junk, "attend past the cache: err_draft_depth and no output");
    }
    if (ok) {
        dr::prepare_args p = {};
        p.kv_k = c0.kvk.p;
        p.kv_v = c0.kvv.p;
        p.k_norm = c0.kn.p;
        p.n_before = c0.nb.p;
        p.pos = c0.pos.p;
        p.stride = c0.stride.p;
        p.k_cache = c0.kc.p;
        p.v_cache = c0.vc.p;
        p.max_cells = cells;
        p.err = c0.err.p;
        cudaSetDevice(dev0);
        t.check(!dr::datc_prepare(0, p, c0.st) && !dr::datc_prepare(6, p, c0.st), "prepare rejects rows 0 and 6");
        p.row0 = 3;
        t.check(!dr::datc_prepare(3, p, c0.st), "prepare rejects row0 + rows > 5");
        dr::attend_args a = {};
        a.q16 = c0.q16.p;
        a.qfull = c0.qfull.p;
        a.k_cache = c0.kc.p;
        a.v_cache = c0.vc.p;
        a.n_before = c0.nb.p;
        a.row0 = dr::max_rows;
        a.max_cells = cells;
        a.part = c0.part.p;
        a.out = c0.out.p;
        a.codes = c0.codes.p;
        a.scales = c0.scales.p;
        a.err = c0.err.p;
        t.check(!dr::datc_attend(a, c0.st), "attend rejects row0 5");
        a.row0 = dr::max_sparse_steps;
        a.body = c0.body.p;
        t.check(!dr::datc_attend(a, c0.st), "the sparse attend rejects row0 4");
    }
    for (attn_card * c : cs) {
        cudaSetDevice(c->dev);
        cudaStreamDestroy(c->st);
    }
    return ok;
}

// ---- eh_proj's B_c adapter (amendment 1) ----

bool test_eh(int32_t dev, tally & t) {
    namespace dn = fn::dense;
    constexpr int32_t K = dr::in_width, M = H, R = dr::max_rows * dr::n_stream; // 20 stream rows
    const fn::quant::wfmt q = fn::quant::wfmt::q8_0;
    const size_t rb = fn::quant::row_bytes(q, K);
    std::vector<uint8_t> raw(size_t(M) * rb), packed(size_t(M) * rb);
    if (!fn::quant::synth_blocks(q, fn::quant::synth_seed(0x4548u), int64_t(M) * (K / 32), raw.data()) ||
        !dn::pack(dn::fmt::q8_0, M, K, raw.data(), rb, packed.data())) {
        fprintf(stderr, "draft-kernel-test: eh_proj weights\n");
        return false;
    }
    std::vector<float> wf(size_t(M) * K);
    for (int32_t n = 0; n < M; ++n) {
        if (!fn::quant::decode_row(q, raw.data() + size_t(n) * rb, wf.data() + size_t(n) * K, K)) {
            return false;
        }
    }
    rng r{0x45485052ull};
    std::vector<float> x;
    r.fill(x, size_t(R) * K, -2.0f, 2.0f);
    std::vector<int8_t> codes(size_t(R) * K);
    std::vector<float> scales(size_t(R) * (K / 32) * 2), xv(size_t(R) * K);
    fn::quant::act_quantize_ref(x.data(), R, K, codes.data(), scales.data());
    for (int32_t i = 0; i < R; ++i) {
        fn::quant::act_values_ref(codes.data() + size_t(i) * K, scales.data() + size_t(i) * (K / 32) * 2, K,
                                  xv.data() + size_t(i) * K);
    }
    cudaSetDevice(dev);
    cudaStream_t st = nullptr;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    dev_buf<uint8_t> w(packed.size());
    dev_buf<int8_t> xc_(codes.size());
    dev_buf<float> xs(scales.size()), out(size_t(R) * M);
    dn::product_args base = {};
    base.k = K;
    base.n_mats = 1;
    base.mats[0].f = dn::fmt::q8_0;
    base.mats[0].m = M;
    base.mats[0].w = w.p;
    base.mats[0].out = out.p;
    base.x_codes = xc_.p;
    base.x_scales = xs.p;
    dn::choose_splits(base);
    dev_buf<uint8_t> work(std::max<size_t>(dn::work_bytes(base), 16));
    base.work = work.p;
    bool ok = st != nullptr && w.p && xc_.p && xs.p && out.p && work.p && w.put(packed, st) && xc_.put(codes, st) &&
              xs.put(scales, st) && cuda_ok(cudaMemsetAsync(work.p, 0, work.n, st), "work");
    printf("  eh_proj: Q8_0 [5120 -> 1280], split %d, work %zu bytes\n", base.mats[0].split, dn::work_bytes(base));
    // the reference rows (FP64 of the decoded weights and the activation values)
    std::vector<double> ref(size_t(R) * M);
    for (int32_t i = 0; ok && i < R; ++i) {
        for (int32_t n = 0; n < M; ++n) {
            double acc = 0.0;
            const float * wr = wf.data() + size_t(n) * K;
            const float * xr = xv.data() + size_t(i) * K;
            for (int32_t kk = 0; kk < K; ++kk) {
                acc += double(wr[kk]) * double(xr[kk]);
            }
            ref[size_t(i) * M + n] = acc;
        }
    }
    std::vector<float> single(size_t(R) * M);
    // each position alone (B_c = 1 at that position's rows), then composed at B_c = 1..5: bitwise the same rows
    for (int32_t p = 0; ok && p < dr::max_rows; ++p) {
        dn::product_args a = base;
        a.x_codes = xc_.p + size_t(p) * dr::n_stream * K;
        a.x_scales = xs.p + size_t(p) * dr::n_stream * (K / 32) * 2;
        a.mats[0].out = out.p + size_t(p) * dr::n_stream * M;
        ok = dr::eh_proj(1, a, st) && cuda_ok(cudaStreamSynchronize(st), "eh_proj");
    }
    ok = ok && out.get(single.data(), single.size(), st);
    for (int32_t b = 1; ok && b <= dr::max_rows; ++b) {
        ok = cuda_ok(cudaMemsetAsync(out.p, 0xff, out.n * 4, st), "junk") && dr::eh_proj(b, base, st) &&
             cuda_ok(cudaStreamSynchronize(st), "eh_proj");
        std::vector<float> got;
        ok = ok && out.get(got, size_t(R) * M, st);
        if (!ok) {
            break;
        }
        for (int32_t i = 0; i < b * dr::n_stream; ++i) {
            const double err = norm_err(ref.data() + size_t(i) * M, got.data() + size_t(i) * M, M);
            t.eh_err = std::max(t.eh_err, err);
            t.check(err <= eh_bound, "eh_proj B_c " + std::to_string(b) + " row " + std::to_string(i) +
                                         " outside 1e-5 (" + std::to_string(err) + ")");
        }
        t.check(memcmp(got.data(), single.data(), size_t(b) * dr::n_stream * M * 4) == 0,
                "eh_proj B_c " + std::to_string(b) + ": rows differ from each position alone");
        bool untouched = true;
        for (size_t i = size_t(b) * dr::n_stream * M; i < got.size(); ++i) {
            uint32_t u = 0;
            memcpy(&u, &got[i], 4);
            untouched = untouched && u == 0xffffffffu;
        }
        t.check(untouched, "eh_proj B_c " + std::to_string(b) + ": rows past B_c written");
    }
    if (ok) {
        t.check(!dr::eh_proj(0, base, st) && !dr::eh_proj(6, base, st), "eh_proj rejects B_c 0 and 6");
        dn::product_args bad = base;
        bad.k = 2560;
        t.check(!dr::eh_proj(1, bad, st), "eh_proj rejects K 2560");
    }
    cudaStreamDestroy(st);
    return ok;
}

} // namespace

int fnfl_draft_kernel_test(int dev0, int dev1, const char * mode) {
    const bool san = mode != nullptr && strcmp(mode, "san") == 0;
    if (!san && (mode == nullptr || strcmp(mode, "all") != 0)) {
        fprintf(stderr, "draft-kernel-test: mode is all or san\n");
        return 2;
    }
    printf("draft-kernel-test: devices %d and %d, mode %s\n", dev0, dev1, san ? "san" : "all");
    for (int32_t which = 0; which < 5; ++which) {
        dr::resources rs = {};
        cudaSetDevice(dev0);
        if (dr::kernel_resources(which, rs)) {
            static const char * names[] = {"draft_in send", "draft_in recv", "DA-TC prepare", "DA-TC attend",
                                           "DA-TC sparse attend (SP2)"};
            printf("  %s: %d registers, %d local bytes, %d static shared bytes, %d CTAs an SM\n", names[which], rs.regs,
                   rs.local_bytes, rs.shared_bytes, rs.ctas_per_sm);
        }
    }
    tally t;
    const bool ok = test_in(dev0, dev1, t) && test_attn(dev0, dev1, san, t) && test_eh(dev0, t);
    printf("  draft-in worst %.3g (bound 1e-6); attention worst %.3g at %s (bound 2e-3; from the "
           "FP64 q %.3g); K worst %.3f F16 ulp; F32 q, k "
           "worst %.3g (1e-6); eh_proj worst %.3g (1e-5)\n",
           t.in_err, t.attn_err, t.attn_at.c_str(), t.attn_err64, t.k_ulp, t.f32_err, t.eh_err);
    printf("  %lld checks, %lld failed\n", static_cast<long long>(t.checks), static_cast<long long>(t.failed));
    const bool pass = ok && t.failed == 0;
    printf("draft-kernel-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
