#include "fn/kernels/qsa/qsa_test.h"

#include "fn/kernels/qsa/qsa.h"
#include "fn/model/shapes.h"
#include "fn/quant/formats.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace q = fn::qsa;

constexpr int32_t HD = q::head_dim;
constexpr int32_t NH = q::n_idx_head;
constexpr int32_t R = q::ratio;
constexpr int32_t fixture_width = 2051; // the HX7 fixture's rows: the pinned source's top_k width (2,048 + 3)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "qa-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// owns one device allocation of n elements of T (at least one)
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

    bool put(const std::vector<T> & h, cudaStream_t st) {
        return h.size() <= n &&
               cuda_ok(cudaMemcpyAsync(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice, st), "upload");
    }

    bool get(std::vector<T> & h, size_t count, cudaStream_t st) const {
        h.resize(count);
        return count <= n &&
               cuda_ok(cudaMemcpyAsync(h.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost, st), "download") &&
               cuda_ok(cudaStreamSynchronize(st), "download sync");
    }
};

// ---- F32 as the device computes it with -use_fast_math: round to nearest, subnormal inputs and results flushed ----

float ftz(float x) {
    return std::fabs(x) < FLT_MIN ? std::copysign(0.0f, x) : x;
}

float fadd(float a, float b) {
    return ftz(ftz(a) + ftz(b));
}

float fmul(float a, float b) {
    return ftz(ftz(a) * ftz(b));
}

float ffma(float a, float b, float c) {
    return ftz(std::fma(ftz(a), ftz(b), ftz(c)));
}

uint32_t bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

uint16_t to_bf16(float f) {
    return fn::quant::f32_to_bf16(f);
}

float from_bf16(uint16_t h) {
    return fn::quant::bf16_to_f32(h);
}

// F32 through IEEE binary16 storage: round to nearest even, subnormals kept, overflow to infinity
float f16_round(float f) {
    const float a = std::fabs(f);
    if (std::isnan(f)) {
        return f;
    }
    if (a >= 65520.0f) {
        return std::copysign(INFINITY, f);
    }
    int e = 0;
    std::frexp(a, &e);
    const int quantum = std::max(e - 1 - 10, -24); // the F16 spacing at a's exponent (subnormal spacing below 2^-14)
    const float scale = std::ldexp(1.0f, -quantum);
    return std::copysign(std::nearbyint(a * scale) / scale, f); // exact scalings; nearbyint rounds to nearest even
}

// the selection key: the score as an order-preserving U32, subnormals as +0, -inf and NaN as 0
// (after ggml_qsa_select_key)
uint32_t select_key(float v) {
    if (!(v > -INFINITY)) {
        return 0;
    }
    uint32_t b = bits(v);
    if ((b & 0x7f800000u) == 0) {
        b = 0;
    }
    return b ^ ((b >> 31) ? 0xffffffffu : 0x80000000u);
}

// The device's results of its fast-math instructions for arguments computed here. MUFU.RSQ, MUFU.RCP, MUFU.LG2/EX2 and
// MUFU.SIN/COS are hardware approximations that no C++ library reproduces bit for bit, so the CPU reference computes
// every argument itself and takes only those instructions' results from the device (qsa.h fast_math_oracle).
struct oracle {
    cudaStream_t st;

    bool eval(int32_t op, const std::vector<float> & x, const std::vector<float> & y, std::vector<float> & out) const {
        const size_t n = x.size();
        if (n == 0) {
            out.clear();
            return true;
        }
        dev_buf<float> dx(n), dy(n), dout(n);
        std::vector<float> yy = y;
        yy.resize(n, 0.0f);
        return dx.p != nullptr && dy.p != nullptr && dout.p != nullptr && dx.put(x, st) && dy.put(yy, st) &&
               q::fast_math_oracle(op, dx.p, dy.p, dout.p, static_cast<int32_t>(n), st) && dout.get(out, n, st);
    }
};

// XOR butterfly over 32 lanes (warp_reduce_sum): every lane ends with the same sum
float butterfly(std::vector<float> v) {
    for (int off = 16; off > 0; off >>= 1) {
        std::vector<float> w(32);
        for (int l = 0; l < 32; ++l) {
            w[l] = fadd(v[l], v[l ^ off]);
        }
        v = w;
    }
    return v[0];
}

// RMS norm times w and the indexer rope of n rows of 128 (in place), as qa1's norm_rope and ggml-cuda's
// qsa_pool_rope_f32 compute it; pos[i] is row i's position.
bool norm_rope_ref(std::vector<float> & x, const std::vector<int32_t> & pos, const std::vector<float> & w,
                   const q::rope_params & rp, const oracle & orc) {
    const size_t n = pos.size();
    std::vector<float> one(1, 1.0f), d128(1, static_cast<float>(HD)), rcp;
    if (!orc.eval(1, one, d128, rcp)) {
        return false;
    }
    std::vector<float> ts(32, rp.theta_scale), kk(32), pw;
    for (int k = 0; k < 32; ++k) {
        kk[k] = static_cast<float>(k);
    }
    if (!orc.eval(2, ts, kk, pw)) {
        return false;
    }
    std::vector<float> me(n), zero(n, 0.0f), rs;
    for (size_t i = 0; i < n; ++i) {
        const float * r = &x[i * HD];
        float ws[4];
        for (int m = 0; m < 4; ++m) {
            std::vector<float> v(32);
            for (int l = 0; l < 32; ++l) {
                v[l] = fmul(r[m * 32 + l], r[m * 32 + l]);
            }
            ws[m] = butterfly(v);
        }
        std::vector<float> v(32, 0.0f);
        for (int m = 0; m < 4; ++m) {
            v[m] = ws[m];
        }
        me[i] = ffma(butterfly(v), rcp[0], fn::rms_eps);
    }
    if (!orc.eval(0, me, zero, rs)) {
        return false;
    }
    std::vector<float> theta(n * 32), cs, sn;
    for (size_t i = 0; i < n; ++i) {
        for (int k = 0; k < 32; ++k) {
            theta[i * 32 + k] = fmul(rp.freq_scale, fmul(static_cast<float>(pos[i]), pw[k]));
        }
    }
    if (!orc.eval(3, theta, zero, cs) || !orc.eval(4, theta, zero, sn)) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        float * r = &x[i * HD];
        float y[HD];
        for (int e = 0; e < HD; ++e) {
            y[e] = fmul(fmul(rs[i], r[e]), w[e]);
        }
        for (int k = 0; k < 32; ++k) {
            const float c = fmul(cs[i * 32 + k], rp.attn_factor);
            const float s = fmul(sn[i * 32 + k], rp.attn_factor);
            r[k] = ffma(y[k], c, -fmul(y[k + 32], s));
            r[k + 32] = ffma(y[k], s, fmul(y[k + 32], c));
        }
        for (int e = q::n_rot; e < HD; ++e) {
            r[e] = y[e];
        }
    }
    return true;
}

// FNFL's selection, one row at position qt, as WP3's body: the min(n_vis, 512) visible blocks with the largest keys
// (equal keys: the lower block first) ascending, then the tail cells 4 n_vis .. qt; appended to body
void select_ref_row(const float * sc, int32_t qt, std::vector<uint32_t> & body) {
    const int32_t n_vis = (qt + 1) / R;
    std::vector<std::pair<uint32_t, int32_t>> cand;
    for (int32_t b = 0; b < n_vis; ++b) {
        cand.emplace_back(select_key(sc[b]), b);
    }
    const auto before = [](const std::pair<uint32_t, int32_t> & a, const std::pair<uint32_t, int32_t> & b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    };
    if (static_cast<int32_t>(cand.size()) > q::top_blocks) {
        std::nth_element(cand.begin(), cand.begin() + q::top_blocks, cand.end(), before);
        cand.resize(static_cast<size_t>(q::top_blocks));
    }
    std::vector<int32_t> sel;
    for (const auto & c : cand) {
        sel.push_back(c.second);
    }
    std::sort(sel.begin(), sel.end());
    body.push_back(static_cast<uint32_t>(sel.size()));
    body.push_back(static_cast<uint32_t>(qt + 1 - n_vis * R));
    for (const int32_t b : sel) {
        body.push_back(static_cast<uint32_t>(b));
    }
    for (int32_t c = n_vis * R; c <= qt; ++c) {
        body.push_back(static_cast<uint32_t>(c));
    }
}

// a test case's inputs, generated on the host
struct case_data {
    std::string name;
    int32_t depth = 0, rows = 0;
    std::vector<uint16_t> keys; // [n_kv][128]
    std::vector<float> qv;      // [rows][4][128]
    std::vector<float> k_norm, q_norm;
    std::vector<std::set<int32_t>> expect; // the HX7 case: each row's captured blocks
};

float bf16_round(float f) {
    return from_bf16(to_bf16(f));
}

// Synthetic keys with deliberate exact ties. Rows' queries share a pattern P on dims 64..127 (which the rope leaves
// alone). Blocks: "super" (2P plus a little noise, a contiguous run, so whole tiles of winners), "tie" (one identical
// set of four keys, zero on dims 0..63, so equal summaries and exactly equal scores; the 512th block falls inside them),
// "anti" (-P, zero on 0..63: every head's dot is negative, so the score is exactly 0, a large tie at zero) and random.
case_data make_synthetic(int32_t depth, int32_t rows) {
    case_data cd;
    cd.name = "synthetic d" + std::to_string(depth) + " B" + std::to_string(rows);
    cd.depth = depth;
    cd.rows = rows;
    std::mt19937 rng(static_cast<uint32_t>(depth * 7 + rows));
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f), u01(0.0f, 1.0f);
    const int32_t n_kv = depth + rows;
    const int32_t nb = n_kv / R;
    std::vector<float> P(HD, 0.0f), F(HD);
    for (int e = 64; e < HD; ++e) {
        P[e] = 0.5f + u01(rng);
    }
    for (auto & f : F) {
        f = u11(rng);
    }
    cd.k_norm.resize(HD);
    cd.q_norm.resize(HD);
    for (int e = 0; e < HD; ++e) {
        cd.k_norm[e] = 0.5f + u01(rng);
        cd.q_norm[e] = 0.5f + u01(rng);
    }
    // each row also leans to its own pattern Qt (random signs on dims 64..127), so the rows select different blocks
    std::vector<std::vector<float>> Q(static_cast<size_t>(rows), std::vector<float>(HD, 0.0f));
    for (auto & qt : Q) {
        for (int e = 64; e < HD; ++e) {
            qt[e] = (u01(rng) < 0.5f ? -0.6f : 0.6f);
        }
    }
    cd.qv.resize(static_cast<size_t>(rows) * NH * HD);
    for (size_t i = 0; i < cd.qv.size(); ++i) {
        cd.qv[i] = P[i % HD] + Q[i / (int64_t(NH) * HD)][i % HD] + 0.3f * u11(rng);
    }
    // block kinds by depth; ROW + t: blocks that row t prefers
    enum { RANDOM, SUPER, TIE, ANTI, ROW };
    std::vector<int> kind(static_cast<size_t>(nb), RANDOM);
    int32_t n_super = 0, super0 = 0, n_tie = 0, tie0 = 0, tie_stride = 1, n_anti = 0;
    if (nb >= 513 && nb < 1024) {
        n_super = 10, super0 = 200, n_tie = 30, tie0 = 5, tie_stride = 9, n_anti = 40;
    } else if (nb >= 1024 && nb < 8192) {
        n_super = 64, super0 = 100, n_tie = 600, tie0 = 200, tie_stride = 3, n_anti = 800;
    } else if (nb >= 8192 && nb < 32768) {
        n_super = 128, super0 = 5000, n_tie = 600, tie0 = 7, tie_stride = 27, n_anti = 4000;
    } else if (nb >= 32768) {
        n_super = 256, super0 = 30000, n_tie = 600, tie0 = 11, tie_stride = 80, n_anti = 10000;
    }
    for (int32_t i = 0; i < n_super && super0 + i < nb; ++i) {
        kind[int64_t(super0) + i] = SUPER;
    }
    for (int32_t i = 0, b = tie0; i < n_tie && b < nb; b += tie_stride) {
        if (kind[static_cast<size_t>(b)] == RANDOM) {
            kind[static_cast<size_t>(b)] = TIE;
            ++i;
        }
    }
    // 60 blocks a row, spread, that its own pattern favours
    for (int32_t t = 0; t < rows && nb >= 513; ++t) {
        for (int32_t i = 0, b = 13 + 5 * t; i < 60 && b < nb; b += std::max(nb / 61, 7)) {
            if (kind[static_cast<size_t>(b)] == RANDOM) {
                kind[static_cast<size_t>(b)] = ROW + t;
                ++i;
            }
        }
    }
    for (int32_t i = 0, b = 1; i < n_anti && b < nb; b += 2) {
        if (kind[static_cast<size_t>(b)] == RANDOM) {
            kind[static_cast<size_t>(b)] = ANTI;
            ++i;
        }
    }
    std::vector<float> tie_key(static_cast<size_t>(R) * HD);
    for (int r = 0; r < R; ++r) {
        for (int e = 0; e < HD; ++e) {
            tie_key[r * HD + e] = e < 64 ? 0.0f : 1.5f * P[e] + 0.4f * F[e] + 0.05f * u11(rng);
        }
    }
    cd.keys.resize(static_cast<size_t>(n_kv) * HD);
    for (int32_t c = 0; c < n_kv; ++c) {
        const int32_t b = c / R;
        const int k = b < nb ? kind[static_cast<size_t>(b)] : RANDOM;
        for (int e = 0; e < HD; ++e) {
            float v = u11(rng);
            if (k == SUPER) {
                v = e < 64 ? 0.05f * v : 2.0f * P[e] + 0.05f * v;
            } else if (k == TIE) {
                v = tie_key[(c % R) * HD + e];
            } else if (k == ANTI) {
                v = e < 64 ? 0.0f : -(0.8f + 0.4f * u01(rng)) * P[e];
            } else if (k >= ROW) {
                v = e < 64 ? 0.05f * v : 1.2f * P[e] + 2.5f * Q[static_cast<size_t>(k - ROW)][e] + 0.05f * v;
            }
            cd.keys[static_cast<size_t>(c) * HD + e] = to_bf16(v);
        }
    }
    return cd;
}

// Every block tied: one key for every cell, zero on the rotated dims 0..63, so every summary and score is equal. At
// 262,140 cells (65,535 blocks, the 2,048th tile short) every tile fails, which takes the failing-tile loop past the
// registers and the 9-bit passes from the top (no 16-bit counts), and the 512 lowest blocks are taken.
case_data make_all_tied(int32_t depth, int32_t rows) {
    case_data cd;
    cd.name = "all tied d" + std::to_string(depth) + " B" + std::to_string(rows);
    cd.depth = depth;
    cd.rows = rows;
    std::mt19937 rng(static_cast<uint32_t>(depth + rows));
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f), u01(0.0f, 1.0f);
    cd.k_norm.resize(HD);
    cd.q_norm.resize(HD);
    std::vector<float> key(HD, 0.0f);
    for (int e = 0; e < HD; ++e) {
        cd.k_norm[e] = 0.5f + u01(rng);
        cd.q_norm[e] = 0.5f + u01(rng);
        key[e] = e < 64 ? 0.0f : u11(rng);
    }
    cd.qv.resize(static_cast<size_t>(rows) * NH * HD);
    for (size_t i = 0; i < cd.qv.size(); ++i) {
        cd.qv[i] = key[i % HD] + 0.3f * u11(rng);
    }
    const int32_t n_kv = depth + rows;
    cd.keys.resize(static_cast<size_t>(n_kv) * HD);
    for (int32_t c = 0; c < n_kv; ++c) {
        for (int e = 0; e < HD; ++e) {
            cd.keys[static_cast<size_t>(c) * HD + e] = to_bf16(key[e]);
        }
    }
    return cd;
}

// The HX7 case (fixture from work/QA1/src/make_fixtures.py): the real layer-3 index norms, and keys shaped so each
// row's 512 best blocks are the blocks HX7 captured for it (three nearly orthogonal row patterns on dims 64..127).
// depth 0 keeps the capture's depth (128,000); 196,608 gives qa-bench's clustered point (load_clustered), key for key.
bool make_hx7(const std::string & path, int32_t depth, case_data & cd) {
    std::ifstream f(path, std::ios::binary);
    int32_t hdr[4] = {};
    if (!f.read(reinterpret_cast<char *>(hdr), sizeof(hdr)) || hdr[0] != 0x31415141 || hdr[2] != 3 ||
        hdr[3] != fixture_width) {
        fprintf(stderr, "qa-test: %s is not a QA1 HX7 fixture\n", path.c_str());
        return false;
    }
    cd.depth = depth > 0 ? depth : hdr[1];
    cd.name = "hx7 layer 3 decode 0 d" + std::to_string(cd.depth);
    cd.rows = hdr[2];
    std::vector<int32_t> cells(static_cast<size_t>(cd.rows) * fixture_width);
    cd.k_norm.resize(HD);
    cd.q_norm.resize(HD);
    if (!f.read(reinterpret_cast<char *>(cells.data()), static_cast<std::streamsize>(cells.size() * 4)) ||
        !f.read(reinterpret_cast<char *>(cd.k_norm.data()), int64_t(HD) * 4) ||
        !f.read(reinterpret_cast<char *>(cd.q_norm.data()), int64_t(HD) * 4)) {
        fprintf(stderr, "qa-test: %s is short\n", path.c_str());
        return false;
    }
    cd.expect.resize(static_cast<size_t>(cd.rows));
    for (int32_t t = 0; t < cd.rows; ++t) {
        for (int32_t j = 0; j < q::top_blocks * R; j += R) {
            cd.expect[static_cast<size_t>(t)].insert(cells[static_cast<size_t>(t) * fixture_width + j] / R);
        }
    }
    std::mt19937 rng(128000);
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f), u01(0.0f, 1.0f);
    std::vector<std::vector<float>> P(static_cast<size_t>(cd.rows), std::vector<float>(HD, 0.0f));
    for (auto & p : P) {
        for (int e = 64; e < HD; ++e) {
            p[e] = (u01(rng) < 0.5f ? -1.0f : 1.0f) * (0.5f + u01(rng));
        }
    }
    cd.qv.resize(static_cast<size_t>(cd.rows) * NH * HD);
    for (int32_t t = 0; t < cd.rows; ++t) {
        for (int i = 0; i < NH * HD; ++i) {
            cd.qv[static_cast<size_t>(t) * NH * HD + i] = P[static_cast<size_t>(t)][i % HD] + 0.2f * u11(rng);
        }
    }
    const int32_t n_kv = cd.depth + cd.rows;
    std::vector<std::vector<char>> in(static_cast<size_t>(cd.rows),
                                      std::vector<char>(static_cast<size_t>(n_kv / R + 1), 0));
    for (int32_t t = 0; t < cd.rows; ++t) {
        for (const int32_t b : cd.expect[static_cast<size_t>(t)]) {
            in[static_cast<size_t>(t)][static_cast<size_t>(b)] = 1;
        }
    }
    cd.keys.resize(static_cast<size_t>(n_kv) * HD);
    for (int32_t c = 0; c < n_kv; ++c) {
        const int32_t b = c / R;
        for (int e = 0; e < HD; ++e) {
            float v = 0.3f * u11(rng);
            for (int32_t t = 0; t < cd.rows; ++t) {
                if (e >= 64 && in[static_cast<size_t>(t)][static_cast<size_t>(b)] != 0) {
                    v += P[static_cast<size_t>(t)][e];
                }
            }
            cd.keys[static_cast<size_t>(c) * HD + e] = to_bf16(v);
        }
    }
    return true;
}

struct checker {
    int failures = 0;

    void expect(bool ok, const std::string & what) {
        if (!ok) {
            ++failures;
            printf("  FAIL %s\n", what.c_str());
        }
    }
};

template <typename T> size_t count_diff(const std::vector<T> & a, const std::vector<T> & b) {
    if (a.size() != b.size()) {
        return SIZE_MAX;
    }
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    }
    return d;
}

// the outputs of one run of the kernels
struct run_out {
    std::vector<float> summaries;
    std::vector<uint32_t> body;
    std::vector<int32_t> offsets;
    std::vector<uint32_t> keys;
    std::vector<uint32_t> error;
    std::vector<uint32_t> stats;
    // SP2: one row at depth D: select's q_ahead form at depth D - 1 (the query at D, the D / 4 whole blocks) against
    // the plain form's blocks at D, which see the same blocks when D % 4 != 3: -1 not run, 1 bitwise equal, 0 differ
    int32_t q_ahead_same = -1;
};

struct device_case {
    cudaStream_t st;
    const case_data & cd;
    int32_t n_kv, nb;
    dev_buf<uint16_t> keys;
    dev_buf<float> summaries, qv, k_norm, q_norm;
    dev_buf<uint8_t> ws_mem;
    dev_buf<uint32_t> body; // exactly body_words: a write past the body is the sanitizer's to find
    dev_buf<int32_t> offsets;
    dev_buf<int32_t> depth; // [1]: select reads the depth on the device (TG1)
    q::workspace ws;

    device_case(cudaStream_t s, const case_data & c)
        : st(s), cd(c), n_kv(c.depth + c.rows), nb(n_kv / R), keys(c.keys.size()),
          summaries(q::summary_bytes(n_kv / R) / 4), qv(c.qv.size()), k_norm(HD), q_norm(HD),
          ws_mem(q::workspace_bytes()), body(static_cast<size_t>(q::body_words(c.depth, c.rows))),
          offsets(static_cast<size_t>(c.rows) + 1), depth(1), ws(q::carve(ws_mem.p)) {}

    bool ready() const {
        return keys.p && summaries.p && qv.p && k_norm.p && q_norm.p && ws_mem.p && body.p && offsets.p && depth.p;
    }

    bool upload() {
        return keys.put(cd.keys, st) && qv.put(cd.qv, st) && depth.put(std::vector<int32_t>{cd.depth}, st) &&
               k_norm.put(cd.k_norm, st) && q_norm.put(cd.q_norm, st) &&
               cuda_ok(cudaMemsetAsync(ws_mem.p, 0, q::workspace_bytes(), st), "workspace") &&
               // the tiles' lists as a deeper earlier call would leave them: the largest key everywhere, so a select that
               // reads a list launch 1 did not write this time takes junk
               cuda_ok(cudaMemsetAsync(ws.best1, 0xff,
                                       static_cast<size_t>(q::max_rows) * q::max_blocks / q::tile_blocks * 4, st),
                       "stale lists") &&
               cuda_ok(cudaMemsetAsync(ws.best2, 0xff,
                                       static_cast<size_t>(q::max_rows) * q::max_blocks / q::tile_blocks * 4, st),
                       "stale lists") &&
               cuda_ok(cudaMemsetAsync(ws.lanes, 0x05,
                                       static_cast<size_t>(q::max_rows) * q::max_blocks / q::tile_blocks * 2, st),
                       "stale lists") &&
               cuda_ok(cudaMemsetAsync(summaries.p, 0, summaries.n * 4, st), "summaries");
    }

    // build every summary, rebuild the last three after a rollback, select; read everything back
    bool run(run_out & o) {
        bool ok = cuda_ok(cudaMemsetAsync(body.p, 0xab, body.n * 4, st), "clear") &&
                  cuda_ok(cudaMemsetAsync(offsets.p, 0xab, offsets.n * 4, st), "clear");
        q::summary_index idx;
        if (ok && nb > 0) {
            const q::update_args u = {keys.p, k_norm.p, summaries.p, 0, nb, q::model_rope()};
            ok = q::summary_update(u, st) && q::mark_updated(idx, 0, nb);
            if (ok && nb >= 3) {
                // a rollback of the last three blocks: their summaries are stale until recomputed
                q::invalidate_from(idx, nb - 3);
                const size_t tile0 = static_cast<size_t>(nb - 3) / q::tile_blocks * q::tile_blocks;
                ok = idx.n_valid == nb - 3 &&
                     cuda_ok(cudaMemsetAsync(summaries.p + tile0 * HD, 0x7f, (summaries.n - tile0 * HD) * 4, st),
                             "stale");
                // the round-sized updates: at most two blocks a launch
                const q::update_args u1 = {keys.p, k_norm.p, summaries.p, nb - 3, 1, q::model_rope()};
                const q::update_args u2 = {keys.p, k_norm.p, summaries.p, nb - 2, 2, q::model_rope()};
                ok = ok && q::summary_update(u1, st) && q::mark_updated(idx, nb - 3, 1) && q::summary_update(u2, st) &&
                     q::mark_updated(idx, nb - 2, 2) && idx.n_valid == nb;
                // the rollback cleared whole tiles: rebuild the rest of the first one
                if (ok && tile0 < static_cast<size_t>(nb - 3)) {
                    const q::update_args u3 = {keys.p,
                                               k_norm.p,
                                               summaries.p,
                                               static_cast<int32_t>(tile0),
                                               nb - 3 - static_cast<int32_t>(tile0),
                                               q::model_rope()};
                    ok = q::summary_update(u3, st);
                }
            }
        }
        if (!ok) {
            fprintf(stderr, "qa-test: summary update not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        // the grid of the deepest launch (TG1: one captured select serves every depth); the summaries cover the depth
        if (idx.n_valid < n_kv / R) {
            fprintf(stderr, "qa-test: summaries %d do not cover the depth's %d blocks\n", idx.n_valid, n_kv / R);
            return false;
        }
        const q::select_args sa = {summaries.p,     qv.p, q_norm.p, depth.p,  q::max_cells,
                                   q::model_rope(), ws,   body.p,   offsets.p};
        if (!q::select(cd.rows, sa, st)) {
            fprintf(stderr, "qa-test: select not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        if (!cuda_ok(cudaStreamSynchronize(st), "run") || !cuda_ok(cudaGetLastError(), "run") || !cuda_get(o)) {
            return false;
        }
        if (cd.rows != 1 || cd.depth < 1 || cd.depth % R == R - 1) {
            return true;
        }
        // SP2: the q_ahead form at depth D - 1; its count and block IDs against the plain form's (the tails differ:
        // the plain row's own cell D is in its tail)
        q::select_args sq = sa;
        sq.q_ahead = 1;
        std::vector<uint32_t> qbody;
        std::vector<int32_t> qoff;
        const uint32_t n0 = o.body.empty() ? 0u : o.body[0];
        ok = depth.put(std::vector<int32_t>{cd.depth - 1}, st) &&
             cuda_ok(cudaMemsetAsync(body.p, 0xab, body.n * 4, st), "clear") && q::select(1, sq, st) &&
             cuda_ok(cudaStreamSynchronize(st), "q_ahead") && body.get(qbody, body.n, st) &&
             offsets.get(qoff, offsets.n, st) && depth.put(std::vector<int32_t>{cd.depth}, st) &&
             cuda_ok(cudaStreamSynchronize(st), "q_ahead");
        if (!ok) {
            fprintf(stderr, "qa-test: q_ahead select: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        bool same = !qbody.empty() && qbody[0] == n0 && size_t(n0) + 2 <= qbody.size();
        for (uint32_t j = 0; same && j < n0; ++j) {
            same = qbody[2 + j] == o.body[2 + j];
        }
        o.q_ahead_same = same ? 1 : 0;
        return true;
    }

    bool cuda_get(run_out & o) {
        bool ok = summaries.get(o.summaries, summaries.n, st) && body.get(o.body, body.n, st) &&
                  offsets.get(o.offsets, offsets.n, st);
        o.keys.resize(static_cast<size_t>(q::max_rows) * q::max_blocks);
        o.error.resize(1);
        o.stats.resize(static_cast<size_t>(q::max_rows) * q::n_stats);
        // on the kernels' own stream: st is non-blocking, so a legacy-stream copy is not ordered with its work
        ok = ok &&
             cuda_ok(cudaMemcpyAsync(o.keys.data(), ws.keys, o.keys.size() * 4, cudaMemcpyDeviceToHost, st), "keys") &&
             cuda_ok(cudaMemcpyAsync(o.error.data(), ws.error, 4, cudaMemcpyDeviceToHost, st), "error") &&
             cuda_ok(cudaMemcpyAsync(o.stats.data(), ws.stats, o.stats.size() * 4, cudaMemcpyDeviceToHost, st),
                     "stats") &&
             cuda_ok(cudaStreamSynchronize(st), "download sync");
        return ok;
    }
};

// the selected blocks of row t of a body (its words from the row's byte offset)
std::set<int32_t> blocks_of(const std::vector<uint32_t> & body, int32_t off_bytes) {
    std::set<int32_t> s;
    const size_t w = static_cast<size_t>(off_bytes) / 4;
    for (uint32_t j = 0; w < body.size() && j < body[w] && w + 2 + j < body.size(); ++j) {
        s.insert(static_cast<int32_t>(body[w + 2 + j]));
    }
    return s;
}

// score[t][b] as the kernels compute it from a summary s (each element through store() first, when given): per head one
// fmaf chain in element order from 0, relu as (d > 0 ? d : 0), the heads summed ((h0 + h1) + h2) + h3
float score_ref(const float * s, const std::vector<float> & qref, int32_t t, float (*store)(float)) {
    float d[NH];
    for (int h = 0; h < NH; ++h) {
        const float * qq = &qref[(static_cast<size_t>(t) * NH + h) * HD];
        float acc = 0.0f;
        for (int e = 0; e < HD; ++e) {
            acc = ffma(store != nullptr ? store(s[e]) : s[e], qq[e], acc);
        }
        d[h] = acc > 0.0f ? acc : 0.0f;
    }
    return fadd(fadd(fadd(d[0], d[1]), d[2]), d[3]);
}

// the top 512 of keys (key descending, block ascending), as a set of blocks
std::set<int32_t> top_set(const std::vector<uint32_t> & keys) {
    std::vector<std::pair<uint32_t, int32_t>> v;
    for (size_t b = 0; b < keys.size(); ++b) {
        if (keys[b] != 0) {
            v.emplace_back(keys[b], static_cast<int32_t>(b));
        }
    }
    const size_t k = std::min<size_t>(v.size(), q::top_blocks);
    std::partial_sort(
        v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end(),
        [](const auto & x, const auto & y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
    std::set<int32_t> out;
    for (size_t i = 0; i < k; ++i) {
        out.insert(v[i].second);
    }
    return out;
}

std::string json_list(const std::vector<int> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? ", " : "") + std::to_string(v[i]);
    }
    return s + "]";
}

bool run_case(const case_data & cd, cudaStream_t st, const oracle & orc, checker & ck) {
    const int32_t B = cd.rows;
    const int32_t n_kv = cd.depth + B;
    const int32_t nb = n_kv / R;                 // complete blocks: the summaries
    const int32_t n_blocks = (n_kv + R - 1) / R; // the pinned source's blocks, the incomplete one included
    const q::rope_params rp = q::model_rope();
    const int fails0 = ck.failures;

    device_case dc(st, cd);
    if (!dc.ready() || !dc.upload()) {
        return false;
    }

    // CPU reference: pooled keys (F32), the summaries
    std::vector<float> pooled(static_cast<size_t>(nb) * HD);
    std::vector<int32_t> bpos(static_cast<size_t>(nb));
    for (int32_t b = 0; b < nb; ++b) {
        bpos[static_cast<size_t>(b)] = b * R;
        for (int e = 0; e < HD; ++e) {
            float x = 0.0f;
            for (int r = 0; r < R; ++r) {
                const float v = from_bf16(cd.keys[(static_cast<size_t>(b) * R + r) * HD + e]);
                x = r == 0 ? v : fadd(x, v);
            }
            pooled[static_cast<size_t>(b) * HD + e] = ffma(x, 1.0f / static_cast<float>(R), 0.0f);
        }
    }
    if (!norm_rope_ref(pooled, bpos, cd.k_norm, rp, orc)) {
        return false;
    }
    std::vector<float> qref = cd.qv;
    std::vector<int32_t> qpos(static_cast<size_t>(B) * NH);
    for (int32_t i = 0; i < B * NH; ++i) {
        qpos[static_cast<size_t>(i)] = cd.depth + i / NH;
    }
    if (!norm_rope_ref(qref, qpos, cd.q_norm, rp, orc)) {
        return false;
    }

    // CPU reference: scores from the F32 pooled keys, then FNFL's selection as the body, and its offsets
    std::vector<float> score(static_cast<size_t>(B) * n_blocks, 0.0f);
    for (int32_t t = 0; t < B; ++t) {
        for (int32_t b = 0; b < nb; ++b) {
            score[static_cast<size_t>(t) * n_blocks + b] =
                score_ref(&pooled[static_cast<size_t>(b) * HD], qref, t, nullptr);
        }
    }
    std::vector<uint32_t> body_ref;
    std::vector<int32_t> off_ref;
    for (int32_t t = 0; t < B; ++t) {
        off_ref.push_back(static_cast<int32_t>(body_ref.size() * 4));
        select_ref_row(&score[static_cast<size_t>(t) * n_blocks], cd.depth + t, body_ref);
    }
    off_ref.push_back(static_cast<int32_t>(body_ref.size() * 4));
    bool host_off = static_cast<int32_t>(body_ref.size()) == q::body_words(cd.depth, B);
    for (int32_t t = 0; t <= B; ++t) {
        host_off = host_off && q::body_offset(cd.depth, t) == off_ref[static_cast<size_t>(t)];
    }
    ck.expect(host_off, "body_offset and body_words vs the CPU body");

    // the kernels, twice
    run_out o1, o2;
    if (!dc.run(o1) || !dc.run(o2)) {
        return false;
    }
    if (o1.q_ahead_same >= 0) {
        printf("  q_ahead (SP2): depth %d - 1 with the query at %d against depth %d: blocks %s\n", cd.depth, cd.depth,
               cd.depth, o1.q_ahead_same == 1 && o2.q_ahead_same == 1 ? "bitwise equal" : "DIFFER");
        ck.expect(o1.q_ahead_same == 1 && o2.q_ahead_same == 1, "q_ahead select's blocks (SP2)");
    }

    // summaries: the stored F32 against the CPU F32
    std::vector<uint32_t> pc(pooled.size()), mine(static_cast<size_t>(nb) * HD);
    for (size_t i = 0; i < pooled.size(); ++i) {
        pc[i] = bits(pooled[i]);
    }
    for (int32_t b = 0; b < nb; ++b) {
        for (int e = 0; e < HD; ++e) {
            mine[static_cast<size_t>(b) * HD + e] =
                bits(o1.summaries[static_cast<size_t>(b / q::tile_blocks) * q::tile_blocks * HD +
                                  static_cast<size_t>(e / 4) * q::tile_blocks * 4 +
                                  static_cast<size_t>(b % q::tile_blocks) * 4 + e % 4]);
        }
    }
    const size_t d_sum_cpu = count_diff(mine, pc);
    ck.expect(d_sum_cpu == 0, "summaries vs CPU: " + std::to_string(d_sum_cpu) + " differ");

    // keys (launch 1 runs only past 512 blocks)
    size_t d_keys = 0;
    if (q::select_launches(B, cd.depth) == 2) {
        for (int32_t t = 0; t < B; ++t) {
            const int32_t n_vis = (cd.depth + t + 1) / R;
            for (int32_t b = 0; b < n_vis; ++b) {
                d_keys += o1.keys[static_cast<size_t>(t) * q::max_blocks + b] !=
                          select_key(score[static_cast<size_t>(t) * n_blocks + b]);
            }
        }
    }
    ck.expect(d_keys == 0, "selection keys vs CPU: " + std::to_string(d_keys) + " differ");

    // selection: the body (counts, blocks, order, tail) and the offsets, bitwise
    const size_t d_body_cpu = count_diff(o1.body, body_ref);
    ck.expect(d_body_cpu == 0, "body vs CPU: " + std::to_string(d_body_cpu) + " words differ");
    ck.expect(o1.offsets == off_ref, "offsets vs CPU");

    ck.expect(o1.error[0] == 0, "device error word " + std::to_string(o1.error[0]));

    // the second run, bitwise
    const bool same = count_diff(o1.summaries, o2.summaries) == 0 && o1.body == o2.body && o1.offsets == o2.offsets &&
                      o1.keys == o2.keys && o2.error[0] == 0;
    ck.expect(same, "second run not bitwise identical");

    // diagnostics, not pass lines: the blocks a row's selection moves when the summaries are stored as BF16 or as F16
    // (round to nearest even) instead of F32, or when the scores are summed in FP64, against the F32 selection
    std::vector<int> mv_bf16, mv_f16, mv_f64;
    for (int32_t t = 0; t < B; ++t) {
        const int32_t n_vis = (cd.depth + t + 1) / R;
        if (n_vis <= q::top_blocks) {
            mv_bf16.push_back(0);
            mv_f16.push_back(0);
            mv_f64.push_back(0);
            continue;
        }
        const std::set<int32_t> ref = blocks_of(body_ref, off_ref[static_cast<size_t>(t)]);
        std::vector<uint32_t> kb(static_cast<size_t>(n_vis)), kh(static_cast<size_t>(n_vis));
        std::vector<std::pair<double, int32_t>> s64(static_cast<size_t>(n_vis));
        for (int32_t b = 0; b < n_vis; ++b) {
            const float * sb = &pooled[static_cast<size_t>(b) * HD];
            kb[static_cast<size_t>(b)] = select_key(score_ref(sb, qref, t, bf16_round));
            kh[static_cast<size_t>(b)] = select_key(score_ref(sb, qref, t, f16_round));
            double a64 = 0.0;
            for (int h = 0; h < NH; ++h) {
                const float * qq = &qref[(static_cast<size_t>(t) * NH + h) * HD];
                double d64 = 0.0;
                for (int e = 0; e < HD; ++e) {
                    d64 += static_cast<double>(sb[e]) * qq[e];
                }
                a64 += d64 > 0 ? d64 : 0;
            }
            s64[static_cast<size_t>(b)] = {-a64, b};
        }
        std::sort(s64.begin(), s64.end());
        std::set<int32_t> s_f64;
        for (int32_t i = 0; i < q::top_blocks; ++i) {
            s_f64.insert(s64[static_cast<size_t>(i)].second);
        }
        const std::set<int32_t> s_bf16 = top_set(kb), s_f16 = top_set(kh);
        int db = 0, dh = 0, d64 = 0;
        for (const int32_t b : ref) {
            db += s_bf16.count(b) == 0;
            dh += s_f16.count(b) == 0;
            d64 += s_f64.count(b) == 0;
        }
        mv_bf16.push_back(db);
        mv_f16.push_back(dh);
        mv_f64.push_back(d64);
    }

    // coverage: the tie at the 512th place (equal keys there, and how many of them are taken), and the tiles launch 2
    // reads back (the rule of select_row, recomputed here)
    std::string cover;
    for (int32_t t = 0; t < B; ++t) {
        const int32_t n_vis = (cd.depth + t + 1) / R;
        if (n_vis <= q::top_blocks) {
            continue;
        }
        std::vector<std::pair<uint32_t, int32_t>> ks;
        for (int32_t b = 0; b < n_vis; ++b) {
            ks.emplace_back(select_key(score[static_cast<size_t>(t) * n_blocks + b]), b);
        }
        std::vector<std::pair<uint32_t, int32_t>> srt = ks;
        std::sort(srt.begin(), srt.end(), [](const auto & x, const auto & y) {
            return x.first != y.first ? x.first > y.first : x.second < y.second;
        });
        const uint32_t kth = srt[q::top_blocks - 1].first;
        int eq = 0, eq_taken = 0;
        for (int32_t i = 0; i < n_vis; ++i) {
            if (srt[static_cast<size_t>(i)].first == kth) {
                ++eq;
                eq_taken += i < q::top_blocks;
            }
        }
        // listed: each tile's best two; T1 = the 512th listed; a tile fails when it has more than two blocks and its
        // second is at or above T1
        std::vector<std::pair<uint32_t, int32_t>> listed;
        std::vector<std::pair<uint32_t, int32_t>> second;
        for (int32_t w = 0; w * 32 < n_vis; ++w) {
            std::vector<std::pair<uint32_t, int32_t>> tl(ks.begin() + int64_t(w) * 32,
                                                         ks.begin() + std::min(n_vis, w * 32 + 32));
            std::sort(tl.begin(), tl.end(), [](const auto & x, const auto & y) {
                return x.first != y.first ? x.first > y.first : x.second < y.second;
            });
            listed.push_back(tl[0]);
            if (tl.size() > 1) {
                listed.push_back(tl[1]);
            }
            second.push_back(tl.size() > 2 ? tl[1] : std::make_pair(0u, INT32_MAX));
        }
        std::sort(listed.begin(), listed.end(), [](const auto & x, const auto & y) {
            return x.first != y.first ? x.first > y.first : x.second < y.second;
        });
        int fails = 0;
        for (const auto & z : second) {
            if (z.first == 0) {
                continue;
            }
            fails += static_cast<int32_t>(listed.size()) <= q::top_blocks ||
                     z.first > listed[q::top_blocks - 1].first ||
                     (z.first == listed[q::top_blocks - 1].first && z.second <= listed[q::top_blocks - 1].second);
        }
        cover += " r" + std::to_string(t) + ": tie " + std::to_string(eq_taken) + "/" + std::to_string(eq) +
                 ", failing tiles " + std::to_string(fails) + ";";
    }

    printf("  %-26s n_kv %7d body %5d B launches %d: summaries %zu, keys %zu, body %zu, rerun %s; blocks moved a "
           "row: BF16 storage %s, F16 storage %s, FP64 sums %s\n",
           cd.name.c_str(), n_kv, off_ref[static_cast<size_t>(B)], q::select_launches(B, cd.depth), d_sum_cpu, d_keys,
           d_body_cpu, same ? "identical" : "DIFFERS", json_list(mv_bf16).c_str(), json_list(mv_f16).c_str(),
           json_list(mv_f64).c_str());
    printf("    json {\"case\": \"%s\", \"depth\": %d, \"rows\": %d, \"moved_bf16\": %s, \"moved_f16\": %s, "
           "\"moved_fp64\": %s}\n",
           cd.name.c_str(), cd.depth, B, json_list(mv_bf16).c_str(), json_list(mv_f16).c_str(),
           json_list(mv_f64).c_str());
    if (!cover.empty()) {
        printf("    coverage:%s\n", cover.c_str());
    }
    std::string path;
    for (int32_t t = 0; t < B; ++t) {
        const uint32_t * x = &o1.stats[static_cast<size_t>(t) * q::n_stats];
        if (x[7] != 0) {
            path += " r" + std::to_string(t) + ": listed " + std::to_string(x[0]) + ", T1 bin " +
                    std::to_string(static_cast<int32_t>(x[1])) + ", failing " + std::to_string(x[2]) +
                    (x[3] != 0 ? ", T2 bin " + std::to_string(x[4]) : std::string(", unpacked")) + ", 9-bit passes " +
                    std::to_string(x[5]) + ", warp-finished " + std::to_string(x[6]) + ";";
        }
    }
    if (!path.empty()) {
        printf("    select path:%s\n", path.c_str());
    }
    if (!cd.expect.empty()) {
        for (int32_t t = 0; t < B; ++t) {
            const std::set<int32_t> got = blocks_of(o1.body, o1.offsets[static_cast<size_t>(t)]);
            int match = 0;
            for (const int32_t b : got) {
                match += cd.expect[static_cast<size_t>(t)].count(b) != 0;
            }
            printf("    hx7 row %d: %d of 512 selected blocks are the captured ones\n", t, match);
        }
    }
    fflush(stdout);
    if (ck.failures != fails0) {
        printf("    (%d failed checks in this case)\n", ck.failures - fails0);
    }
    return true;
}

} // namespace

int fnfl_qsa_test(int device, const std::vector<int32_t> & depths_in, const std::vector<int32_t> & rows_in,
                  const std::string & fixture) {
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaDeviceProp prop = {};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "properties")) {
        return 1;
    }
    printf("# qa-test: %s, sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    const std::vector<int32_t> depths =
        depths_in.empty() ? std::vector<int32_t>{0, 4, 2051, 8192, 65536, 196608} : depths_in;
    const std::vector<int32_t> rows = rows_in.empty() ? std::vector<int32_t>{1, 3, 4} : rows_in;
    cudaStream_t st = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) {
        return 1;
    }
    const oracle orc = {st};
    checker ck;
    bool ok = true;
    int n_cases = 0;
    for (const int32_t d : depths) {
        for (const int32_t b : rows) {
            ok = ok && run_case(make_synthetic(d, b), st, orc, ck);
            ++n_cases;
        }
    }
    if (depths_in.empty() && rows_in.empty()) {
        ok = ok && run_case(make_all_tied(262140, 1), st, orc, ck);
        ++n_cases;
    }
    if (!fixture.empty()) {
        // the capture's depth, then qa-bench's clustered point (196,608 cells, B = 3)
        for (const int32_t d : {0, 196608}) {
            case_data cd;
            ok = ok && make_hx7(fixture, d, cd) && run_case(cd, st, orc, ck);
            ++n_cases;
        }
    } else {
        printf("  (no fixture: the HX7 case is skipped)\n");
    }
    cudaStreamDestroy(st);
    const bool pass = ok && ck.failures == 0;
    printf("qa-test: %s (%d cases, %d failed checks%s)\n", pass ? "PASS" : "FAIL", n_cases, ck.failures,
           ok ? "" : ", a run did not complete");
    return pass ? 0 : 1;
}
