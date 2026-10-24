#include "fn/kernels/gdn/gdn_test.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/act/act.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/gdn/gdn_ref.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/round/version.h"
#include "fn/model/placement.h"
#include "fn/model/shapes.h"
#include "fn/quant/act_ref.h"
#include "fn/tools/test_dump.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

// The references are written from the pinned lines (66e11c3), never from a GPU kernel (R16):
//   qwen4exp.cpp 1694-1819 (build_layer_attn_linear): beta = sigmoid(beta); g = softplus(alpha + ssm_dt) * ssm_a; the
//       convolution over [history | rows], SiLU; Q, K, V views of channels 0.., 2,048.., 4,096.. (global); the L2 norms;
//       the recurrence; build_norm_gated(output, ssm_norm, z).
//   qwen4exp.cpp 1968-2020 (build_conv_state_at): the history is the last 3 columns of [history | rows].
//   models.h 13-17: build_gdn_l2_norm = rms_norm(x, eps / n) / sqrt(n), that is x / sqrt(sum x^2 + eps).
//   qwen4exp.cpp 1012-1021: build_norm_gated = rms_norm(x) * ssm_norm * sigmoid(z) ("not silu").
//   ops.cpp 10990-11142: M = exp(g) M; delta_j = (v_j - sum_i M[j][i] k_i) beta; M[j][i] += k_i delta_j;
//       o_j = (sum_i M[j][i] q_i) / sqrt(S_v); head iv1 reads Q/K group iv1 % neq1.
//   ggml-impl.h 107-109 / unary-ops.cpp 80-82: softplus(x) = x > 20 ? x : log(1 + exp(x)).
// ref<double, false> is the FP64 reference (sums in index order). ref<float, true> is an F32 reference in gdn_core's
// order (gdn.cu's contract), reported only for scale.

namespace {

namespace gd = fn::gdn;

constexpr int32_t S = gd::head_dim;
constexpr double bound = 1e-4;
// GD1 Amendment 1: a row's output bound is bound * max(1, cond_floor / cond), cond from the FP64 reference; cond exactly
// 0 (zero q) requires every output exactly 0
constexpr double cond_floor = 1e-3;
constexpr int32_t n_ref_threads = 8;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "gd-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// owns one device allocation
struct dev_buf {
    void * p = nullptr;
    size_t n = 0;

    explicit dev_buf(size_t bytes) : n(bytes) {
        if (!cuda_ok(cudaMalloc(&p, bytes), "cudaMalloc")) {
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

    float * f() const { return static_cast<float *>(p); }
};

// ---- the layer's real weights, from the GGUF on the CPU ----

struct layer_weights {
    std::vector<float> conv, dt, a, norm; // conv [n_ch][4], dt and a [n_head], norm [128]
};

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

bool read_f32(const std::vector<fnfl_gguf_dir> & dirs, const std::string & name, int64_t n, std::vector<float> & out) {
    size_t d = 0;
    const fnfl_gguf_tensor * t = find_tensor(dirs, name, d);
    if (t == nullptr || t->type_name != "f32" || t->bytes != uint64_t(n) * 4) {
        fprintf(stderr, "gd-test: %s: missing, not F32 or not %lld values\n", name.c_str(), static_cast<long long>(n));
        return false;
    }
    out.assign(size_t(n), 0.0f);
    FILE * f = fopen(dirs[d].path.c_str(), "rb");
    const bool ok = f != nullptr && fseeko(f, static_cast<off_t>(t->offset), SEEK_SET) == 0 &&
                    fread(out.data(), 4, size_t(n), f) == size_t(n);
    if (f != nullptr) {
        fclose(f);
    }
    if (!ok) {
        fprintf(stderr, "gd-test: %s: read failed\n", name.c_str());
    }
    return ok;
}

// the global weights of one layer (48 heads, 10,240 channels)
bool read_layer(const std::vector<fnfl_gguf_dir> & dirs, int32_t layer, layer_weights & w) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    return read_f32(dirs, p + "ssm_conv1d.weight", int64_t(2) * gd::n_channel * gd::n_tap, w.conv) &&
           read_f32(dirs, p + "ssm_dt.bias", int64_t{2} * gd::n_head, w.dt) &&
           read_f32(dirs, p + "ssm_a", int64_t{2} * gd::n_head, w.a) &&
           read_f32(dirs, p + "ssm_norm.weight", S, w.norm);
}

// the global indices a card holds of a tensor, in slice order: its rows (ne1) or its columns (ne0)
std::vector<int32_t> card_index(const char * name, int32_t card, bool rows) {
    std::vector<int32_t> out;
    for (int32_t t = 0; t < fn::model::n_tensors; ++t) {
        if (strcmp(fn::model::tensors[t].name, name) != 0) {
            continue;
        }
        const fn::model::card_place & cp = fn::model::tensors[t].card[card];
        const int32_t first = rows ? cp.row_first : cp.col_first, count = rows ? cp.row_count : cp.col_count;
        for (int32_t i = 0; i < count; ++i) {
            const fn::model::range & r = fn::model::ranges[first + i];
            for (int32_t e = 0; e < r.count; ++e) {
                out.push_back(r.first + e);
            }
        }
    }
    return out;
}

// a card's channel, value and head maps (local index -> global), from the placement of layer 10 (every GDN layer is
// placed alike; checked for the layers the test reads)
struct card_map {
    std::vector<int32_t> ch, val, head;
};

bool make_map(int32_t layer, int32_t card, card_map & m) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    m.ch = card_index((p + "ssm_conv1d.weight").c_str(), card, true);
    m.val = card_index((p + "attn_gate.weight").c_str(), card, true);
    m.head = card_index((p + "ssm_dt.bias").c_str(), card, false);
    const std::vector<int32_t> qkv = card_index((p + "attn_qkv.weight").c_str(), card, true);
    const std::vector<int32_t> al = card_index((p + "ssm_alpha.weight").c_str(), card, true);
    const std::vector<int32_t> aa = card_index((p + "ssm_a").c_str(), card, false);
    bool ok = int32_t(m.ch.size()) == gd::n_channel && int32_t(m.val.size()) == gd::n_value &&
              int32_t(m.head.size()) == gd::n_head && qkv == m.ch && al == m.head && aa == m.head;
    // the card-local layout gdn.h states: V of head l at 2,048 + 128 l is global value head head[l]; z alike
    for (int32_t l = 0; ok && l < gd::n_head; ++l) {
        for (int32_t j = 0; j < S; ++j) {
            ok = ok && m.ch[gd::v_first + l * S + j] == 2 * gd::v_first + m.head[l] * S + j &&
                 m.val[l * S + j] == m.head[l] * S + j;
        }
        ok = ok && m.ch[size_t(l % gd::n_group) * S] == m.head[l] % 16 * S;
    }
    if (!ok) {
        fprintf(stderr, "gd-test: layer %d card %d: the placement is not gdn.h's card-local layout\n", layer, card);
    }
    return ok;
}

// the card's slice of a layer's weights
layer_weights card_weights(const layer_weights & g, const card_map & m) {
    layer_weights w;
    w.conv.resize(size_t(gd::n_channel) * gd::n_tap);
    for (int32_t c = 0; c < gd::n_channel; ++c) {
        for (int32_t k = 0; k < gd::n_tap; ++k) {
            w.conv[size_t(c) * gd::n_tap + k] = g.conv[size_t(m.ch[c]) * gd::n_tap + k];
        }
    }
    for (int32_t l = 0; l < gd::n_head; ++l) {
        w.dt.push_back(g.dt[m.head[l]]);
        w.a.push_back(g.a[m.head[l]]);
    }
    w.norm = g.norm;
    return w;
}

// ---- the references ----

struct shape {
    int32_t n_head, n_group, k_first, v_first, n_ch;
    bool interleave; // the wrong mapping, for the mapping check's contrast: head h reads group h / (n_head / n_group)

    int32_t n_val() const { return n_head * S; }
    int32_t group_of(int32_t h) const { return interleave ? h / (n_head / n_group) : h % n_group; }
};

constexpr shape card_shape = {gd::n_head, gd::n_group, gd::k_first, gd::v_first, gd::n_channel, false};
constexpr shape global_shape = {2 * gd::n_head, 16, 2 * gd::k_first, 2 * gd::v_first, 2 * gd::n_channel, false};

struct step_inputs {
    int32_t rows = 0;
    std::vector<float> qkv, z, alpha, beta; // [rows][n_ch], [rows][n_val], [rows][n_head] x 2
};

template <typename T> struct ref_out {
    std::vector<T> gated;     // [rows][n_val]
    std::vector<T> state;     // [rows][n_head][128][128]: versions 1..rows
    std::vector<float> hist;  // [rows][n_ch][3]: versions 1..rows
    std::vector<double> cond; // [rows][n_head]: rms(o) / (max_j |M_j| / sqrt(128)), in [0, 1]; small: o cancels
    std::vector<T> conv;      // [rows][n_ch]: the convolution's post-SiLU output (the epilogue's, Amendment 2b)
};

template <typename T> T dot_in_order(const T * a, const T * b) {
    T s = T(0);
    for (int32_t i = 0; i < S; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

// gdn_core's dot: lane l's 4 products (FMUL, then FMAs), then the xor butterfly 16, 8, 4, 2, 1 (lane 0's value)
float dot_kernel_order(const float * a, const float * b) {
    float p[32];
    for (int32_t l = 0; l < 32; ++l) {
        float s = a[size_t(4) * l] * b[size_t(4) * l];
        for (int32_t e = 1; e < 4; ++e) {
            s = std::fma(a[4 * l + e], b[4 * l + e], s);
        }
        p[l] = s;
    }
    for (int32_t o = 16; o > 0; o >>= 1) {
        float q[32];
        for (int32_t l = 0; l < 32; ++l) {
            q[l] = p[l] + p[l ^ o];
        }
        memcpy(p, q, sizeof(p));
    }
    return p[0];
}

float butterfly32(const float * v) {
    float p[32];
    memcpy(p, v, sizeof(p));
    for (int32_t o = 16; o > 0; o >>= 1) {
        float q[32];
        for (int32_t l = 0; l < 32; ++l) {
            q[l] = p[l] + p[l ^ o];
        }
        memcpy(p, q, sizeof(p));
    }
    return p[0];
}

// gdn_core's sums of squares of 128 values (the q and k norms, the gated norm): lane l's 4 values in order (FMUL, then
// FMAs), then the xor butterfly
template <typename T> float sum_sq_kernel_order(const T * x) {
    float p[32];
    for (int32_t l = 0; l < 32; ++l) {
        const T * xl = x + size_t(4) * l;
        float s = float(xl[0]) * float(xl[0]);
        for (int32_t e = 1; e < 4; ++e) {
            s = std::fma(float(xl[e]), float(xl[e]), s);
        }
        p[l] = s;
    }
    return butterfly32(p);
}

// One step of rows from the committed state0 (T [n_head][128][128]) and hist0 (F32 [n_ch][3]).
template <typename T, bool kernel_order>
void ref_step(const shape & sh, const layer_weights & w, const step_inputs & in, const T * state0, const float * hist0,
              ref_out<T> & out) {
    const int32_t B = in.rows, nc = sh.n_ch, nv = sh.n_val(), nh = sh.n_head;
    const T eps = T(fn::rms_eps);
    // the convolution with SiLU, and the history versions
    std::vector<T> y(size_t(B) * nc);
    out.hist.assign(size_t(B) * nc * gd::n_hist, 0.0f);
    for (int32_t c = 0; c < nc; ++c) {
        float seq[gd::n_hist + gd::max_rows];
        for (int32_t k = 0; k < gd::n_hist; ++k) {
            seq[k] = hist0[size_t(c) * gd::n_hist + k];
        }
        for (int32_t b = 0; b < B; ++b) {
            seq[gd::n_hist + b] = in.qkv[size_t(b) * nc + c];
        }
        const float * wc = &w.conv[size_t(c) * gd::n_tap];
        for (int32_t b = 0; b < B; ++b) {
            T acc;
            if (kernel_order) {
                acc = T(seq[b] * wc[0]);
                for (int32_t k = 1; k < gd::n_tap; ++k) {
                    acc = T(std::fma(seq[b + k], wc[k], float(acc)));
                }
            } else {
                acc = T(0);
                for (int32_t k = 0; k < gd::n_tap; ++k) {
                    acc += T(seq[b + k]) * T(wc[k]);
                }
            }
            y[size_t(b) * nc + c] = acc / (T(1) + std::exp(-acc));
        }
        for (int32_t r = 1; r <= B; ++r) {
            for (int32_t k = 0; k < gd::n_hist; ++k) {
                out.hist[(size_t(r - 1) * nc + c) * gd::n_hist + k] = seq[r + k];
            }
        }
    }
    out.conv = y;
    // the L2 norms of each group's q and k
    std::vector<T> qn(size_t(B) * sh.n_group * S), kn(size_t(B) * sh.n_group * S);
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t gq = 0; gq < sh.n_group; ++gq) {
            for (int32_t which = 0; which < 2; ++which) {
                const T * x = &y[size_t(b) * nc + (which == 0 ? 0 : sh.k_first) + size_t(gq) * S];
                T tot = T(0);
                if (kernel_order) {
                    tot = T(sum_sq_kernel_order(x));
                } else {
                    for (int32_t i = 0; i < S; ++i) {
                        tot += x[i] * x[i];
                    }
                }
                T * dst = &(which == 0 ? qn : kn)[(size_t(b) * sh.n_group + gq) * S];
                const T den = std::sqrt(tot + eps);
                const T r = T(1) / den; // the kernel's x rsqrt(sum + eps)
                for (int32_t i = 0; i < S; ++i) {
                    dst[i] = kernel_order ? x[i] * r : x[i] / den;
                }
            }
        }
    }
    // the gates
    std::vector<T> decay(size_t(B) * nh), beta(size_t(B) * nh);
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t h = 0; h < nh; ++h) {
            const T x =
                kernel_order ? T(in.alpha[size_t(b) * nh + h] + w.dt[h]) : T(in.alpha[size_t(b) * nh + h]) + T(w.dt[h]);
            const T sp = x > T(20) ? x : std::log(T(1) + std::exp(x));
            decay[size_t(b) * nh + h] = std::exp(sp * T(w.a[h]));
            beta[size_t(b) * nh + h] = T(1) / (T(1) + std::exp(-T(in.beta[size_t(b) * nh + h])));
        }
    }
    // the recurrence, a head at a time (threads over heads), and the gated norm
    out.state.assign(size_t(B) * nh * S * S, T(0));
    out.gated.assign(size_t(B) * nv, T(0));
    out.cond.assign(size_t(B) * nh, 0.0);
    const T scale = kernel_order ? T(1.0f / std::sqrt(128.0f)) : T(1) / std::sqrt(T(S));
    auto head = [&](int32_t h) {
        std::vector<T> m(state0 + size_t(h) * S * S, state0 + size_t(h + 1) * S * S);
        std::vector<T> o(size_t(B) * S);
        const int32_t gq = sh.group_of(h);
        for (int32_t b = 0; b < B; ++b) {
            const T * k = &kn[(size_t(b) * sh.n_group + gq) * S];
            const T * q = &qn[(size_t(b) * sh.n_group + gq) * S];
            const T * v = &y[size_t(b) * nc + sh.v_first + size_t(h) * S];
            const T dec = decay[size_t(b) * nh + h], bt = beta[size_t(b) * nh + h];
            for (int32_t j = 0; j < S; ++j) {
                T * mj = &m[size_t(j) * S];
                for (int32_t i = 0; i < S; ++i) {
                    mj[i] *= dec;
                }
                T p;
                if constexpr (kernel_order) {
                    p = dot_kernel_order(mj, k);
                } else {
                    p = dot_in_order(mj, k);
                }
                const T delta = (v[j] - p) * bt;
                for (int32_t i = 0; i < S; ++i) {
                    if constexpr (kernel_order) {
                        mj[i] = std::fma(k[i], delta, mj[i]);
                    } else {
                        mj[i] += k[i] * delta;
                    }
                }
                if constexpr (kernel_order) {
                    o[size_t(b) * S + j] = dot_kernel_order(mj, q) * scale;
                } else {
                    o[size_t(b) * S + j] = dot_in_order(mj, q) * scale;
                }
            }
            std::copy(m.begin(), m.end(), out.state.begin() + (size_t(b) * nh + h) * S * S);
            double mx = 0.0, oo = 0.0;
            for (int32_t j = 0; j < S; ++j) {
                double r = 0.0;
                for (int32_t i = 0; i < S; ++i) {
                    r += double(m[size_t(j) * S + i]) * double(m[size_t(j) * S + i]);
                }
                mx = std::max(mx, std::sqrt(r));
                oo += double(o[size_t(b) * S + j]) * double(o[size_t(b) * S + j]);
            }
            out.cond[size_t(b) * nh + h] = mx > 0.0 ? std::sqrt(oo / S) / (mx / std::sqrt(double(S))) : 1.0;
        }
        for (int32_t b = 0; b < B; ++b) {
            const T * ob = &o[size_t(b) * S];
            T ss = T(0);
            if (kernel_order) {
                ss = T(sum_sq_kernel_order(ob));
            } else {
                for (int32_t j = 0; j < S; ++j) {
                    ss += ob[j] * ob[j];
                }
            }
            const T rs = T(1) / std::sqrt(ss / T(S) + eps);
            for (int32_t j = 0; j < S; ++j) {
                const T zv = T(in.z[size_t(b) * nv + size_t(h) * S + j]);
                out.gated[size_t(b) * nv + size_t(h) * S + j] =
                    ob[j] * rs * T(w.norm[j]) * (T(1) / (T(1) + std::exp(-zv)));
            }
        }
    };
    std::vector<std::thread> pool;
    for (int32_t t = 0; t < n_ref_threads; ++t) {
        pool.emplace_back([&, t] {
            for (int32_t h = t; h < nh; h += n_ref_threads) {
                head(h);
            }
        });
    }
    for (std::thread & t : pool) {
        t.join();
    }
}

// max |got - ref| / max |ref| over n values (the absolute error when the reference is all zero)
template <typename A, typename R> double norm_err(const A * got, const R * ref, size_t n) {
    double e = 0.0, m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        e = std::max(e, std::fabs(double(got[i]) - double(ref[i])));
        m = std::max(m, std::fabs(double(ref[i])));
    }
    return m > 0.0 ? e / m : e;
}

// the worst normalized errors seen, the kernel's and the F32 reference's (both against FP64)
struct worst {
    double out = 0.0, out32 = 0.0, state = 0.0, state32 = 0.0, conv = 0.0, conv32 = 0.0;
    double win = 0.0, win32 = 0.0; // the chained run's current window of positions
    // Amendment 1: rows judged under the widened bound, rows with cond exactly 0, the smallest positive cond, and the
    // worst error / bound ratio (all and in the current case or window)
    int64_t n_widened = 0, n_zero = 0, case_widened = 0;
    double min_cond = 1.0, ratio = 0.0, case_ratio = 0.0;
    std::vector<double> conds; // every checked (row, head)'s cond
    std::string out_at, state_at;
    int64_t n_fail = 0;
    bool ok = true;

    // every failure counts; the first 20 are printed
    void fail(const std::string & what) {
        if (n_fail < 20) {
            fprintf(stderr, "gd-test: FAIL: %s\n", what.c_str());
        }
        ++n_fail;
        ok = false;
    }
};

// the outputs of one step: per row and head
void check_outputs(const std::string & where, const float * got, const ref_out<double> & r64,
                   const ref_out<float> & r32, int32_t rows, int32_t nh, worst & w) {
    for (int32_t b = 0; b < rows; ++b) {
        for (int32_t h = 0; h < nh; ++h) {
            const size_t o = size_t(b) * nh * S + size_t(h) * S;
            const double e = norm_err(got + o, r64.gated.data() + o, S);
            const double e32 = norm_err(r32.gated.data() + o, r64.gated.data() + o, S);
            w.out32 = std::max(w.out32, e32);
            w.conds.push_back(r64.cond[size_t(b) * nh + h]);
            w.win = std::max(w.win, e);
            w.win32 = std::max(w.win32, e32);
            if (e > w.out) {
                w.out = e;
                w.out_at = where + " row " + std::to_string(b) + " head " + std::to_string(h);
            }
            const double cond = r64.cond[size_t(b) * nh + h];
            if (cond == 0.0) {
                ++w.n_zero;
                for (int32_t j = 0; j < S; ++j) {
                    if (got[o + j] != 0.0f) {
                        char buf[256];
                        snprintf(buf, sizeof(buf), "%s: output row %d head %d: cond 0 but output %d is %.3e, not 0",
                                 where.c_str(), b, h, j, double(got[o + j]));
                        w.fail(buf);
                        break;
                    }
                }
                continue;
            }
            const double bd = bound * std::max(1.0, cond_floor / cond);
            if (cond < cond_floor) {
                ++w.n_widened;
                ++w.case_widened;
            }
            w.min_cond = std::min(w.min_cond, cond);
            w.ratio = std::max(w.ratio, e / bd);
            w.case_ratio = std::max(w.case_ratio, e / bd);
            if (!(e <= bd)) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "%s: output row %d head %d: %.3e > bound %.3e (F32 reference %.3e); cond %.3e", where.c_str(),
                         b, h, e, bd, e32, cond);
                w.fail(buf);
            }
        }
    }
}

// the convolution epilogue's post-SiLU output (got [rows][5120]): per row and 128-channel group
constexpr double conv_bound = 1e-5;
void check_conv(const std::string & where, const float * got, const ref_out<double> & r64, const ref_out<float> & r32,
                int32_t rows, worst & w) {
    for (int32_t b = 0; b < rows; ++b) {
        for (int32_t gch = 0; gch < gd::n_channel / S; ++gch) {
            const size_t o = size_t(b) * gd::n_channel + size_t(gch) * S;
            const double e = norm_err(got + o, r64.conv.data() + o, S);
            w.conv = std::max(w.conv, e);
            w.conv32 = std::max(w.conv32, norm_err(r32.conv.data() + o, r64.conv.data() + o, S));
            if (!(e <= conv_bound)) {
                char buf[256];
                snprintf(buf, sizeof(buf), "%s: convolution row %d channels %d..: %.3e > 1e-5", where.c_str(), b,
                         gch * S, e);
                w.fail(buf);
            }
        }
    }
}

// the state versions 1..rows (got [rows][n_head][128][128]): per version and head
void check_states(const std::string & where, const float * got, const ref_out<double> & r64, const ref_out<float> & r32,
                  int32_t rows, int32_t nh, worst & w) {
    for (int32_t v = 0; v < rows; ++v) {
        for (int32_t h = 0; h < nh; ++h) {
            const size_t o = (size_t(v) * nh + h) * S * S;
            const double e = norm_err(got + o, r64.state.data() + o, size_t(S) * S);
            const double e32 = norm_err(r32.state.data() + o, r64.state.data() + o, size_t(S) * S);
            w.state32 = std::max(w.state32, e32);
            if (e > w.state) {
                w.state = e;
                w.state_at = where + " version " + std::to_string(v + 1) + " head " + std::to_string(h);
            }
            if (!(e <= bound)) {
                char buf[256];
                snprintf(buf, sizeof(buf), "%s: state version %d head %d: %.3e > 1e-4 (F32 reference %.3e)",
                         where.c_str(), v + 1, h, e, e32);
                w.fail(buf);
            }
        }
    }
}

// ---- the kernel on one card ----

struct card_gpu {
    cudaStream_t st = nullptr;
    dev_buf qkv{size_t(gd::max_rows) * gd::n_channel * 4}, z{size_t(gd::max_rows) * gd::n_value * 4};
    dev_buf cv{size_t(gd::max_rows) * gd::n_channel * 4}; // the epilogue's post-SiLU output, gdn_core's input
    dev_buf alpha{size_t(gd::max_rows) * gd::n_head * 4}, beta{size_t(gd::max_rows) * gd::n_head * 4};
    dev_buf conv{size_t(gd::n_channel) * gd::n_tap * 4}, dt{size_t(gd::n_head) * 4}, a{size_t(gd::n_head) * 4},
        norm{size_t(S) * 4};
    // ME11: the compact state, two banks and two record sets (gdn.h)
    dev_buf state{size_t(gd::n_bank) * gd::state_version_floats * 4};
    dev_buf rec{size_t(gd::n_rec_set) * gd::rec_set_floats * 4};
    dev_buf hist{size_t(gd::n_version) * gd::hist_version_floats * 4};
    dev_buf gated{size_t(gd::max_rows) * gd::n_value * 4};
    dev_buf codes{fn::act::codes_bytes(gd::max_rows, gd::n_value)};
    dev_buf scales{fn::act::scales_bytes(gd::max_rows, gd::n_value)};
    dev_buf ws{gd::workspace_bytes()};
    dev_buf cur{8};          // the committed slot (SR1), then the GDN selector (ME11)
    uint32_t cur_h = 0;      // their host mirrors
    uint32_t gsel_h = 0;
    uint32_t slots_seen = 0; // bit s: a launch ran with the committed slot at s
    uint32_t orient_seen = 0; // ME11: bit bank + 2 set: a launch ran from that bank and record set
    // ME11 test reference (gdn_ref.h): the pre-ME11 kernel's six versions, its slot word and outputs
    dev_buf ref_state{size_t(gd::n_version) * gd::state_version_floats * 4};
    dev_buf ref_cur{4};
    dev_buf ref_gated{size_t(gd::max_rows) * gd::n_value * 4};
    dev_buf ref_codes{fn::act::codes_bytes(gd::max_rows, gd::n_value)};
    dev_buf ref_scales{fn::act::scales_bytes(gd::max_rows, gd::n_value)};
    uint32_t ref_cur_h = 0;

    bool ready() const {
        return st != nullptr && qkv.p && cv.p && z.p && alpha.p && beta.p && conv.p && dt.p && a.p && norm.p &&
               state.p && rec.p && hist.p && gated.p && codes.p && scales.p && ws.p && cur.p && ref_state.p &&
               ref_cur.p && ref_gated.p && ref_codes.p && ref_scales.p;
    }
    float * bank(int32_t b) const { return state.f() + b * gd::state_version_floats; }
    float * rec_set(int32_t r) const { return rec.f() + r * gd::rec_set_floats; }
    // the committed bank: the committed matrix when the selector has no steps
    float * state_v0() const { return bank(fn::round::gdn_bank(gsel_h)); }
    float * hist_v(int32_t v) const { return hist.f() + fn::round::version_slot(cur_h, v) * gd::hist_version_floats; }
    float * ref_v(int32_t v) const {
        return ref_state.f() + fn::round::version_slot(ref_cur_h, v) * gd::state_version_floats;
    }
    const uint32_t * cur_word() const { return static_cast<const uint32_t *>(cur.p); }
    const uint32_t * gsel_word() const { return cur_word() + 1; }
};

bool put(void * dst, const void * src, size_t n, cudaStream_t st) {
    return cuda_ok(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, st), "upload");
}

bool get(void * dst, const void * src, size_t n, cudaStream_t st) {
    return cuda_ok(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, st), "download");
}

bool put_weights(card_gpu & g, const layer_weights & w) {
    return put(g.conv.p, w.conv.data(), w.conv.size() * 4, g.st) && put(g.dt.p, w.dt.data(), w.dt.size() * 4, g.st) &&
           put(g.a.p, w.a.data(), w.a.size() * 4, g.st) && put(g.norm.p, w.norm.data(), w.norm.size() * 4, g.st);
}

bool put_inputs(card_gpu & g, const step_inputs & in) {
    return put(g.qkv.p, in.qkv.data(), in.qkv.size() * 4, g.st) && put(g.z.p, in.z.data(), in.z.size() * 4, g.st) &&
           put(g.alpha.p, in.alpha.data(), in.alpha.size() * 4, g.st) &&
           put(g.beta.p, in.beta.data(), in.beta.size() * 4, g.st);
}

// the convolution epilogue (alone, as gdn.h's conv_silu), then gdn_core
bool run_kernel(card_gpu & g, int32_t rows) {
    gd::conv_args c;
    c.qkv = g.qkv.f();
    c.conv_w = g.conv.f();
    c.hist = g.hist.f();
    c.cur = g.cur_word();
    c.out = g.cv.f();
    c.rows = rows;
    if (!gd::conv_silu(c, g.st)) {
        fprintf(stderr, "gd-test: conv_silu not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    gd::core_args a;
    a.conv = g.cv.f();
    a.z = g.z.f();
    a.alpha = g.alpha.f();
    a.beta = g.beta.f();
    a.dt = g.dt.f();
    a.a = g.a.f();
    a.norm_w = g.norm.f();
    a.state = g.state.f();
    a.rec = g.rec.f();
    a.gsel = g.gsel_word();
    a.gated = g.gated.f();
    a.gated_q = {static_cast<int8_t *>(g.codes.p), g.scales.f(), rows, gd::n_value};
    a.ws = g.ws.p;
    a.rows = rows;
    if (!gd::core(a, g.st)) {
        fprintf(stderr, "gd-test: gdn_core not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    g.slots_seen |= 1u << g.cur_h;
    g.orient_seen |= 1u << (fn::round::gdn_bank(g.gsel_h) + 2 * fn::round::gdn_set(g.gsel_h));
    return cuda_ok(cudaStreamSynchronize(g.st), "gdn_core");
}

// ME11: the pre-ME11 kernel on the same convolution output and inputs (gdn_ref.h), into its own state and outputs
bool run_ref(card_gpu & g, int32_t rows) {
    gd::ref::core_ref_args a;
    a.conv = g.cv.f();
    a.z = g.z.f();
    a.alpha = g.alpha.f();
    a.beta = g.beta.f();
    a.dt = g.dt.f();
    a.a = g.a.f();
    a.norm_w = g.norm.f();
    a.state = g.ref_state.f();
    a.cur = static_cast<const uint32_t *>(g.ref_cur.p);
    a.gated = g.ref_gated.f();
    a.gated_q = {static_cast<int8_t *>(g.ref_codes.p), g.ref_scales.f(), rows, gd::n_value};
    a.ws = g.ws.p;
    a.rows = rows;
    if (!gd::ref::core_ref(a, g.st)) {
        fprintf(stderr, "gd-test: the reference not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return cuda_ok(cudaStreamSynchronize(g.st), "reference");
}

// the committed slot and the GDN selector set from the host (SR1: the short cases start each case at another slot;
// ME11: and another bank and record set)
bool set_cur(card_gpu & g, uint32_t cur, uint32_t gsel = 0) {
    g.cur_h = cur;
    g.gsel_h = gsel;
    const uint32_t w[2] = {cur, gsel};
    return cuda_ok(cudaMemcpyAsync(g.cur.p, w, 8, cudaMemcpyHostToDevice, g.st), "cur") &&
           cuda_ok(cudaStreamSynchronize(g.st), "cur");
}

// the commit (SR1, fn::round::advance as layer-test's runner does it): logical version c becomes the committed one by
// moving the slot word and (ME11) the GDN selector; the device words are checked against the host mirrors
bool commit(card_gpu & g, int32_t c) {
    uint32_t got[2] = {0, 0};
    g.cur_h = static_cast<uint32_t>(fn::round::version_slot(g.cur_h, c));
    g.gsel_h = c >= 1 ? fn::round::gdn_after_round(g.gsel_h, c) : g.gsel_h;
    uint32_t * w = static_cast<uint32_t *>(g.cur.p);
    return (fn::round::advance(w, w + 1, c, false, g.st) ? true : cuda_ok(cudaGetLastError(), "commit")) &&
           cuda_ok(cudaMemcpyAsync(got, g.cur.p, 8, cudaMemcpyDeviceToHost, g.st), "cur") &&
           cuda_ok(cudaStreamSynchronize(g.st), "commit") && got[0] == g.cur_h && got[1] == g.gsel_h;
}

// ME11: the selector's steps dropped (bank and set kept) after the committed matrix is written to its bank by the host
bool set_plain(card_gpu & g) {
    return set_cur(g, g.cur_h, fn::round::gdn_word(fn::round::gdn_bank(g.gsel_h), fn::round::gdn_set(g.gsel_h), 0));
}

// what one launch wrote
struct step_result {
    std::vector<float> gated, scales, state, hist, conv; // state and hist: versions 1..rows; conv [rows][5120]
    std::vector<int8_t> codes;
    std::vector<float> pre, recs; // ME11: the bank the launch stored (its pre-step matrix) and its record set
};

bool get_result(card_gpu & g, int32_t rows, bool versions, step_result & r) {
    r.gated.resize(size_t(rows) * gd::n_value);
    r.codes.resize(fn::act::codes_bytes(rows, gd::n_value));
    r.scales.resize(fn::act::scales_bytes(rows, gd::n_value) / 4);
    r.conv.resize(size_t(rows) * gd::n_channel);
    bool ok = get(r.gated.data(), g.gated.p, r.gated.size() * 4, g.st) &&
              get(r.codes.data(), g.codes.p, r.codes.size(), g.st) &&
              get(r.scales.data(), g.scales.p, r.scales.size() * 4, g.st) &&
              get(r.conv.data(), g.cv.p, r.conv.size() * 4, g.st);
    if (versions) {
        // ME11: the launch's pre-step bank and record set (from the selector it ran with, still g.gsel_h); version v
        // is that bank with the first v records replayed (gdn_ref.h's replay_host, the device's own operations)
        r.state.resize(size_t(rows) * gd::state_version_floats);
        r.hist.resize(size_t(rows) * gd::hist_version_floats);
        r.pre.resize(size_t(gd::state_version_floats));
        r.recs.resize(size_t(gd::rec_set_floats));
        ok = ok && get(r.pre.data(), g.bank(fn::round::gdn_bank_of(g.gsel_h, 1)), r.pre.size() * 4, g.st) &&
             get(r.recs.data(), g.rec_set(1 - fn::round::gdn_set(g.gsel_h)), r.recs.size() * 4, g.st);
        for (int32_t v = 1; ok && v <= rows; ++v) {
            ok = get(r.hist.data() + size_t(v - 1) * gd::hist_version_floats, g.hist_v(v),
                     size_t(gd::hist_version_floats) * 4, g.st);
        }
    }
    ok = ok && cuda_ok(cudaStreamSynchronize(g.st), "download");
    for (int32_t v = 1; ok && versions && v <= rows; ++v) {
        gd::ref::replay_host(r.pre.data(), r.recs.data(), v, r.state.data() + size_t(v - 1) * gd::state_version_floats);
    }
    const std::string t = "gd-B" + std::to_string(rows) + "-cur" + std::to_string(g.cur_h);
    return ok && fn::tools::test_dump(t + "-gated", r.gated.data(), r.gated.size() * 4) &&
           fn::tools::test_dump(t + "-codes", r.codes.data(), r.codes.size()) &&
           fn::tools::test_dump(t + "-scales", r.scales.data(), r.scales.size() * 4) &&
           fn::tools::test_dump(t + "-conv", r.conv.data(), r.conv.size() * 4) &&
           fn::tools::test_dump(t + "-state", r.state.data(), r.state.size() * 4) &&
           fn::tools::test_dump(t + "-hist", r.hist.data(), r.hist.size() * 4);
}

// the activation format against act_ref applied to the kernel's own F32 output, bitwise
bool act_bitwise(const step_result & r, int32_t rows) {
    std::vector<int8_t> rc(r.codes.size());
    std::vector<float> rs(r.scales.size());
    fn::quant::act_quantize_ref(r.gated.data(), rows, gd::n_value, rc.data(), rs.data());
    return memcmp(rc.data(), r.codes.data(), rc.size()) == 0 && memcmp(rs.data(), r.scales.data(), rs.size() * 4) == 0;
}

// ---- synthetic inputs ----

void gen_inputs(std::mt19937_64 & rng, const shape & sh, int32_t rows, step_inputs & in) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    in.rows = rows;
    in.qkv.resize(size_t(rows) * sh.n_ch);
    in.z.resize(size_t(rows) * sh.n_val());
    in.alpha.resize(size_t(rows) * sh.n_head);
    in.beta.resize(size_t(rows) * sh.n_head);
    for (float & x : in.qkv) {
        x = 1.5f * u(rng);
    }
    for (float & x : in.z) {
        x = 3.0f * u(rng);
    }
    for (float & x : in.alpha) {
        x = -1.0f + 3.0f * u(rng);
    }
    for (float & x : in.beta) {
        x = 3.0f * u(rng);
    }
}

void gen_state(std::mt19937_64 & rng, size_t n_state, size_t n_hist, std::vector<float> & state,
               std::vector<float> & hist) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    state.resize(n_state);
    hist.resize(n_hist);
    for (float & x : state) {
        x = 0.2f * u(rng);
    }
    for (float & x : hist) {
        x = 1.5f * u(rng);
    }
}

// ---- the checks ----

const char * const edge_names[] = {"random", "zero q and k heads", "inputs x 1e3", "g below -80", "g near 0"};

// short cases: B = 1..4 from a random committed state, the edge rows, each run twice (bitwise reruns)
bool short_cases(card_gpu & g, int32_t layer, const layer_weights & w, worst & wr) {
    if (!put_weights(g, w)) {
        return false;
    }
    std::mt19937_64 rng(0x6744314a00ull + uint64_t(layer));
    bool ok = true;
    for (int32_t edge = 0; edge < 5; ++edge) {
        for (int32_t rows = 1; rows <= gd::max_rows; ++rows) {
            const std::string where =
                "short L" + std::to_string(layer) + " " + edge_names[edge] + " B" + std::to_string(rows);
            std::vector<float> state, hist;
            gen_state(rng, gd::state_version_floats, gd::hist_version_floats, state, hist);
            step_inputs in;
            gen_inputs(rng, card_shape, rows, in);
            if (edge == 1) { // group 0's Q and group 1's K are all zeros: the norm's eps path
                for (int32_t i = 0; i < S; ++i) {
                    for (int32_t k = 0; k < gd::n_hist; ++k) {
                        hist[size_t(gd::q_first + i) * gd::n_hist + k] = 0.0f;
                        hist[size_t(gd::k_first + S + i) * gd::n_hist + k] = 0.0f;
                    }
                    for (int32_t b = 0; b < rows; ++b) {
                        in.qkv[size_t(b) * gd::n_channel + gd::q_first + i] = 0.0f;
                        in.qkv[size_t(b) * gd::n_channel + gd::k_first + S + i] = 0.0f;
                    }
                }
            } else if (edge == 2) {
                for (float & x : in.qkv) {
                    x *= 1e3f;
                }
                for (float & x : hist) {
                    x *= 1e3f;
                }
            } else if (edge == 3 || edge == 4) {
                // alpha with g = softplus(alpha + dt) a near -100 (decay to zero), or alpha + dt = -30 (g near 0)
                for (int32_t b = 0; b < rows; ++b) {
                    for (int32_t h = 0; h < gd::n_head; ++h) {
                        const double sp = -100.0 / double(w.a[h]); // softplus(x) = x above 20
                        const double x = sp > 20.0 ? sp : std::log(std::expm1(sp));
                        in.alpha[size_t(b) * gd::n_head + h] =
                            edge == 3 ? float(x - double(w.dt[h])) : -30.0f - w.dt[h];
                    }
                }
            }
            ref_out<double> r64;
            ref_out<float> r32;
            {
                const std::vector<double> s64(state.begin(), state.end());
                ref_step<double, false>(card_shape, w, in, s64.data(), hist.data(), r64);
                ref_step<float, true>(card_shape, w, in, state.data(), hist.data(), r32);
            }
            step_result res[2];
            // SR1: each case and rerun starts at another committed slot, so every reader runs at all five; every
            // other slot is poisoned. ME11: and from each bank and record set in turn (no records: the state is
            // the bank's); the other bank and both record sets are poisoned, so a record element the launch does
            // not write shows as NaN in a replayed version
            for (int32_t rep = 0; rep < 2 && ok; ++rep) {
                const int32_t o = edge * gd::max_rows + rows - 1 + rep;
                ok = set_cur(g, static_cast<uint32_t>((2 * (edge * gd::max_rows + rows - 1) + rep) % gd::n_version),
                             fn::round::gdn_word(o & 1, (o >> 1) & 1, 0)) &&
                     cuda_ok(cudaMemsetAsync(g.state.p, 0xff, g.state.n, g.st), "poison") &&
                     cuda_ok(cudaMemsetAsync(g.rec.p, 0xff, g.rec.n, g.st), "poison") &&
                     put(g.state_v0(), state.data(), state.size() * 4, g.st) &&
                     put(g.hist_v(0), hist.data(), hist.size() * 4, g.st) && put_inputs(g, in) &&
                     cuda_ok(cudaMemsetAsync(g.gated.p, 0xff, g.gated.n, g.st), "poison") &&
                     cuda_ok(cudaMemsetAsync(g.codes.p, 0x5a, g.codes.n, g.st), "poison");
                for (int32_t v = 1; ok && v < gd::n_version; ++v) {
                    ok = cuda_ok(cudaMemsetAsync(g.hist_v(v), 0xff, gd::hist_version_floats * 4, g.st), "poison");
                }
                ok = ok && run_kernel(g, rows) && get_result(g, rows, true, res[rep]);
            }
            if (!ok) {
                return false;
            }
            check_outputs(where, res[0].gated.data(), r64, r32, rows, gd::n_head, wr);
            check_conv(where, res[0].conv.data(), r64, r32, rows, wr);
            check_states(where, res[0].state.data(), r64, r32, rows, gd::n_head, wr);
            if (memcmp(res[0].pre.data(), state.data(), state.size() * 4) != 0) {
                wr.fail(where + ": the stored pre-step bank is not the committed state (ME11)");
            }
            if (memcmp(res[0].hist.data(), r64.hist.data(), r64.hist.size() * 4) != 0) {
                wr.fail(where + ": history versions differ from the reference");
            }
            if (!act_bitwise(res[0], rows)) {
                wr.fail(where + ": the activation format differs from act_ref of the kernel's output");
            }
            const bool same = res[0].gated == res[1].gated && res[0].codes == res[1].codes &&
                              memcmp(res[0].scales.data(), res[1].scales.data(), res[0].scales.size() * 4) == 0 &&
                              memcmp(res[0].state.data(), res[1].state.data(), res[0].state.size() * 4) == 0 &&
                              memcmp(res[0].hist.data(), res[1].hist.data(), res[0].hist.size() * 4) == 0 &&
                              memcmp(res[0].conv.data(), res[1].conv.data(), res[0].conv.size() * 4) == 0;
            if (!same) {
                wr.fail(where + ": the rerun is not bitwise identical");
            }
            // gated's F32 output: NaN-free and not the poison
            for (const float x : res[0].gated) {
                if (!std::isfinite(x)) {
                    wr.fail(where + ": a non-finite output");
                    break;
                }
            }
        }
        printf("gd-test: short L%d %-20s B 1-4: worst output %.3e, state %.3e so far (F32 reference %.3e, %.3e); "
               "%lld rows under the widened bound, worst error / bound %.3f\n",
               layer, edge_names[edge], wr.out, wr.state, wr.out32, wr.state32, static_cast<long long>(wr.case_widened),
               wr.case_ratio);
        wr.case_widened = 0;
        wr.case_ratio = 0.0;
    }
    return true;
}

// the chained run: positions from zero state in steps of the cycle's rows; every 7th step commits version 1 (a
// rejection), the others version B. Outputs and the activation format every step, state and history versions every
// check_every steps, and the committed state and history at the end.
bool chained(card_gpu & g, int32_t layer, const layer_weights & w, int32_t positions,
             const std::vector<int32_t> & cycle, int32_t check_every, const char * name, worst & wr) {
    if (!put_weights(g, w) || !set_plain(g) ||
        !cuda_ok(cudaMemsetAsync(g.state_v0(), 0, gd::state_version_floats * 4, g.st), "zero state") ||
        !cuda_ok(cudaMemsetAsync(g.hist_v(0), 0, gd::hist_version_floats * 4, g.st), "zero history")) {
        return false;
    }
    std::mt19937_64 rng(0x6744314a10ull + uint64_t(layer) * 7 + uint64_t(positions));
    std::vector<double> s64(gd::state_version_floats, 0.0);
    std::vector<float> s32(gd::state_version_floats, 0.0f), h64(gd::hist_version_floats, 0.0f),
        h32(gd::hist_version_floats, 0.0f);
    int32_t pos = 0, step = 0, rejections = 0, checks = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (pos < positions) {
        const int32_t rows = std::min(cycle[size_t(step) % cycle.size()], positions - pos);
        step_inputs in;
        gen_inputs(rng, card_shape, rows, in);
        const bool check = step % check_every == 0;
        step_result res;
        if (!put_inputs(g, in) || !run_kernel(g, rows) || !get_result(g, rows, check, res)) {
            return false;
        }
        ref_out<double> r64;
        ref_out<float> r32;
        ref_step<double, false>(card_shape, w, in, s64.data(), h64.data(), r64);
        ref_step<float, true>(card_shape, w, in, s32.data(), h32.data(), r32);
        const std::string where = std::string(name) + " L" + std::to_string(layer) + " step " + std::to_string(step) +
                                  " (position " + std::to_string(pos) + ", B" + std::to_string(rows) + ")";
        check_outputs(where, res.gated.data(), r64, r32, rows, gd::n_head, wr);
        check_conv(where, res.conv.data(), r64, r32, rows, wr);
        if (!act_bitwise(res, rows)) {
            wr.fail(where + ": the activation format differs from act_ref of the kernel's output");
        }
        if (check) {
            ++checks;
            check_states(where, res.state.data(), r64, r32, rows, gd::n_head, wr);
            if (memcmp(res.hist.data(), r64.hist.data(), r64.hist.size() * 4) != 0) {
                wr.fail(where + ": history versions differ from the reference");
            }
        }
        const int32_t c = step % 7 == 6 ? 1 : rows;
        rejections += c < rows ? 1 : 0;
        const auto so = static_cast<std::ptrdiff_t>(int64_t(c - 1) * gd::state_version_floats);
        const auto ho = static_cast<std::ptrdiff_t>(int64_t(c - 1) * gd::hist_version_floats);
        std::copy(r64.state.begin() + so, r64.state.begin() + so + gd::state_version_floats, s64.begin());
        std::copy(r32.state.begin() + so, r32.state.begin() + so + gd::state_version_floats, s32.begin());
        std::copy(r64.hist.begin() + ho, r64.hist.begin() + ho + gd::hist_version_floats, h64.begin());
        std::copy(r32.hist.begin() + ho, r32.hist.begin() + ho + gd::hist_version_floats, h32.begin());
        if (!commit(g, c)) {
            return false;
        }
        // the worst output error of each window of 512 positions, the kernel's and the F32 reference's
        if ((pos + c) / 512 != pos / 512 || pos + c >= positions) {
            printf("gd-test: %s L%d positions to %5d: worst output %.3e (F32 reference %.3e), %lld rows under the "
                   "widened bound, worst error / bound %.3f, %lld failures so far\n",
                   name, layer, pos + c, wr.win, wr.win32, static_cast<long long>(wr.case_widened), wr.case_ratio,
                   static_cast<long long>(wr.n_fail));
            wr.win = wr.win32 = 0.0;
            wr.case_widened = 0;
            wr.case_ratio = 0.0;
        }
        pos += c;
        ++step;
    }
    // the committed state (ME11: the selector's bank with its records replayed) and history at the end
    std::vector<float> st(gd::state_version_floats), hi(gd::hist_version_floats);
    if (!get(hi.data(), g.hist_v(0), hi.size() * 4, g.st) || !cuda_ok(cudaStreamSynchronize(g.st), "download") ||
        !gd::ref::read_committed(g.state.f(), g.rec.f(), g.gsel_h, st)) {
        return false;
    }
    ref_out<double> e64;
    ref_out<float> e32;
    e64.state = s64;
    e32.state = s32;
    check_states(std::string(name) + " end", st.data(), e64, e32, 1, gd::n_head, wr);
    if (memcmp(hi.data(), h64.data(), hi.size() * 4) != 0) {
        wr.fail(std::string(name) + ": the committed history at the end differs from the reference");
    }
    printf("gd-test: %s L%d: %d positions, %d steps (%d rejections), versions checked at %d steps and the end, "
           "%.1f s: worst output %.3e, state %.3e so far (F32 reference %.3e, %.3e)\n",
           name, layer, pos, step, rejections, checks,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), wr.out, wr.state, wr.out32,
           wr.state32);
    return true;
}

// the mapping: distinct random values for every global group and head; the card's outputs and states against the
// global FP64 reference (head h reads group h % 16), and the same against the interleaved mapping for contrast
bool mapping(card_gpu & g, int32_t card, int32_t layer, const layer_weights & wg, const card_map & m, worst & wr) {
    const layer_weights w = card_weights(wg, m);
    if (!put_weights(g, w)) {
        return false;
    }
    std::mt19937_64 rng(0x6744314a20ull + uint64_t(layer));
    bool ok = true;
    for (const int32_t rows : {1, 3}) {
        std::vector<float> gs, gh;
        gen_state(rng, size_t(global_shape.n_head) * S * S, size_t(global_shape.n_ch) * gd::n_hist, gs, gh);
        step_inputs gi;
        gen_inputs(rng, global_shape, rows, gi);
        // the card's slices
        std::vector<float> cs(gd::state_version_floats), ch(gd::hist_version_floats);
        step_inputs ci;
        ci.rows = rows;
        ci.qkv.resize(size_t(rows) * gd::n_channel);
        ci.z.resize(size_t(rows) * gd::n_value);
        ci.alpha.resize(size_t(rows) * gd::n_head);
        ci.beta.resize(size_t(rows) * gd::n_head);
        for (int32_t c = 0; c < gd::n_channel; ++c) {
            for (int32_t k = 0; k < gd::n_hist; ++k) {
                ch[size_t(c) * gd::n_hist + k] = gh[size_t(m.ch[c]) * gd::n_hist + k];
            }
            for (int32_t b = 0; b < rows; ++b) {
                ci.qkv[size_t(b) * gd::n_channel + c] = gi.qkv[size_t(b) * global_shape.n_ch + m.ch[c]];
            }
        }
        for (int32_t b = 0; b < rows; ++b) {
            for (int32_t v = 0; v < gd::n_value; ++v) {
                ci.z[size_t(b) * gd::n_value + v] = gi.z[size_t(b) * global_shape.n_val() + m.val[v]];
            }
            for (int32_t l = 0; l < gd::n_head; ++l) {
                ci.alpha[size_t(b) * gd::n_head + l] = gi.alpha[size_t(b) * global_shape.n_head + m.head[l]];
                ci.beta[size_t(b) * gd::n_head + l] = gi.beta[size_t(b) * global_shape.n_head + m.head[l]];
            }
        }
        for (int32_t l = 0; l < gd::n_head; ++l) {
            std::copy(gs.begin() + std::ptrdiff_t(m.head[l]) * S * S,
                      gs.begin() + std::ptrdiff_t(m.head[l] + 1) * S * S, cs.begin() + std::ptrdiff_t(l) * S * S);
        }
        step_result res;
        ok = set_plain(g) && put(g.state_v0(), cs.data(), cs.size() * 4, g.st) &&
             put(g.hist_v(0), ch.data(), ch.size() * 4, g.st) && put_inputs(g, ci) && run_kernel(g, rows) &&
             get_result(g, rows, true, res);
        if (!ok) {
            return false;
        }
        double err[2] = {0.0, 0.0}, serr[2] = {0.0, 0.0};
        for (int32_t variant = 0; variant < 2; ++variant) {
            shape sh = global_shape;
            sh.interleave = variant == 1;
            const std::vector<double> s64(gs.begin(), gs.end());
            ref_out<double> r64;
            ref_step<double, false>(sh, wg, gi, s64.data(), gh.data(), r64);
            for (int32_t b = 0; b < rows; ++b) {
                for (int32_t l = 0; l < gd::n_head; ++l) {
                    const double * ref = &r64.gated[size_t(b) * global_shape.n_val() + size_t(m.head[l]) * S];
                    err[variant] =
                        std::max(err[variant], norm_err(&res.gated[size_t(b) * gd::n_value + size_t(l) * S], ref, S));
                    const double * rst = &r64.state[(size_t(b) * global_shape.n_head + m.head[l]) * S * S];
                    serr[variant] = std::max(
                        serr[variant], norm_err(&res.state[(size_t(b) * gd::n_head + l) * S * S], rst, size_t(S) * S));
                }
            }
        }
        printf("gd-test: map L%d card %d B%d: against head h reading global group h %% 16: output %.3e, state %.3e; "
               "against h / 3 (contrast): output %.3e, state %.3e\n",
               layer, card, rows, err[0], serr[0], err[1], serr[1]);
        if (!(err[0] <= bound && serr[0] <= bound)) {
            wr.fail("map: the card's heads do not match the global reference with group h % 16");
        }
        if (!(err[1] > 1e-2)) {
            wr.fail("map: the contrast mapping is not distinguished");
        }
    }
    return true;
}

// batch invariance (NC1 section 2): the same positions in steps of B = 1 and in the cycle 1, 2, 3, 4, every step
// committing its last version: outputs, the activation format and the final state and history bitwise equal
bool invariance(card_gpu & g, int32_t layer, const layer_weights & w, worst & wr) {
    constexpr int32_t positions = 512;
    if (!put_weights(g, w)) {
        return false;
    }
    std::vector<step_inputs> per_pos(positions);
    {
        std::mt19937_64 rng(0x6744314a30ull + uint64_t(layer));
        for (step_inputs & in : per_pos) {
            gen_inputs(rng, card_shape, 1, in);
        }
    }
    std::vector<float> out[2], sc[2], st[2], hi[2];
    std::vector<int8_t> cd[2];
    for (int32_t run = 0; run < 2; ++run) {
        if (!set_plain(g) ||
            !cuda_ok(cudaMemsetAsync(g.state_v0(), 0, gd::state_version_floats * 4, g.st), "zero state") ||
            !cuda_ok(cudaMemsetAsync(g.hist_v(0), 0, gd::hist_version_floats * 4, g.st), "zero history")) {
            return false;
        }
        int32_t pos = 0, step = 0;
        while (pos < positions) {
            const int32_t rows = run == 0 ? 1 : std::min(1 + step % 4, positions - pos);
            step_inputs in;
            in.rows = rows;
            for (int32_t b = 0; b < rows; ++b) {
                const step_inputs & p = per_pos[size_t(pos) + size_t(b)];
                in.qkv.insert(in.qkv.end(), p.qkv.begin(), p.qkv.end());
                in.z.insert(in.z.end(), p.z.begin(), p.z.end());
                in.alpha.insert(in.alpha.end(), p.alpha.begin(), p.alpha.end());
                in.beta.insert(in.beta.end(), p.beta.begin(), p.beta.end());
            }
            step_result res;
            if (!put_inputs(g, in) || !run_kernel(g, rows) || !get_result(g, rows, false, res) || !commit(g, rows)) {
                return false;
            }
            out[run].insert(out[run].end(), res.gated.begin(), res.gated.end());
            cd[run].insert(cd[run].end(), res.codes.begin(), res.codes.end());
            sc[run].insert(sc[run].end(), res.scales.begin(), res.scales.end());
            pos += rows;
            ++step;
        }
        st[run].resize(gd::state_version_floats);
        hi[run].resize(gd::hist_version_floats);
        if (!get(hi[run].data(), g.hist_v(0), hi[run].size() * 4, g.st) ||
            !cuda_ok(cudaStreamSynchronize(g.st), "download") ||
            !gd::ref::read_committed(g.state.f(), g.rec.f(), g.gsel_h, st[run])) {
            return false;
        }
    }
    const bool same_out = memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
    const bool same_act = cd[0] == cd[1] && memcmp(sc[0].data(), sc[1].data(), sc[0].size() * 4) == 0;
    const bool same_state = memcmp(st[0].data(), st[1].data(), st[0].size() * 4) == 0 &&
                            memcmp(hi[0].data(), hi[1].data(), hi[0].size() * 4) == 0;
    printf("gd-test: inv L%d: %d positions, B = 1 against the cycle 1-4: outputs %s, activation format %s, final state "
           "and history %s\n",
           layer, positions, same_out ? "bitwise" : "DIFFER", same_act ? "bitwise" : "DIFFER",
           same_state ? "bitwise" : "DIFFER");
    if (!(same_out && same_act && same_state)) {
        wr.fail("inv: a row's results depend on B");
    }
    return true;
}

// ME11: the compact state against the pre-ME11 kernel (gdn_ref.h), bitwise. A chain of rounds from one random state:
// for B = 1..5 and every k = 0..B, a round of B rows (both kernels on the same convolution output and inputs, the
// edge rows in turn), then the commit of k (k = 0: the zero-version commit, the selector unchanged), and a last round.
// Each round: the outputs (F32, codes, scales) bitwise; the bank the round stored (the device's replay of the previous
// round's k records, its materialized start) bitwise the reference's version 0; and every accepted prefix v = 1..B,
// the host replay of the round's records, bitwise the reference's version v. Both record sets start poisoned (NaN).
bool replay_chain(card_gpu & g, int32_t layer, const layer_weights & w, int32_t start_bank, int32_t max_b, worst & wr) {
    if (!put_weights(g, w)) {
        return false;
    }
    std::mt19937_64 rng(0x6744314a40ull + uint64_t(layer) * 3 + uint64_t(start_bank));
    std::vector<float> state, hist;
    gen_state(rng, gd::state_version_floats, gd::hist_version_floats, state, hist);
    g.ref_cur_h = static_cast<uint32_t>(start_bank + 2);
    if (!set_cur(g, static_cast<uint32_t>(start_bank), fn::round::gdn_word(start_bank, 0, 0)) ||
        !cuda_ok(cudaMemsetAsync(g.state.p, 0xff, g.state.n, g.st), "poison") ||
        !cuda_ok(cudaMemsetAsync(g.rec.p, 0xff, g.rec.n, g.st), "poison") ||
        !put(g.state_v0(), state.data(), state.size() * 4, g.st) ||
        !put(g.hist_v(0), hist.data(), hist.size() * 4, g.st) ||
        !put(g.ref_cur.p, &g.ref_cur_h, 4, g.st) || !put(g.ref_v(0), state.data(), state.size() * 4, g.st) ||
        !cuda_ok(cudaStreamSynchronize(g.st), "upload")) {
        return false;
    }
    std::vector<std::pair<int32_t, int32_t>> plan; // (B, k)
    for (int32_t b = 1; b <= max_b; ++b) {
        for (int32_t k = 0; k <= b; ++k) {
            plan.emplace_back(b, k);
        }
    }
    plan.emplace_back(max_b, 0); // the last round checks the last commit's replay
    int32_t same_out = 0, same_pre = 0, prefixes = 0, same_prefix = 0, round = 0;
    const size_t sv = size_t(gd::state_version_floats);
    std::vector<float> ref(sv), rg, rsc;
    std::vector<int8_t> rcd;
    for (const auto & [rows, k] : plan) {
        step_inputs in;
        gen_inputs(rng, card_shape, rows, in);
        const int32_t edge = round % 5;
        for (int32_t b = 0; b < rows; ++b) {
            for (int32_t h = 0; h < gd::n_head; ++h) {
                const double sp = -100.0 / double(w.a[h]);
                const double x = sp > 20.0 ? sp : std::log(std::expm1(sp));
                float & al = in.alpha[size_t(b) * gd::n_head + h];
                al = edge == 2 ? float(x - double(w.dt[h])) : edge == 3 ? -30.0f - w.dt[h] : al;
            }
        }
        if (edge == 1) {
            for (float & x : in.qkv) {
                x *= 1e3f;
            }
        }
        step_result res;
        if (!put_inputs(g, in) || !run_kernel(g, rows) || !run_ref(g, rows) || !get_result(g, rows, true, res)) {
            return false;
        }
        rg.resize(size_t(rows) * gd::n_value);
        rcd.resize(res.codes.size());
        rsc.resize(res.scales.size());
        bool ok = get(rg.data(), g.ref_gated.p, rg.size() * 4, g.st) &&
                  get(rcd.data(), g.ref_codes.p, rcd.size(), g.st) &&
                  get(rsc.data(), g.ref_scales.p, rsc.size() * 4, g.st) && get(ref.data(), g.ref_v(0), sv * 4, g.st) &&
                  cuda_ok(cudaStreamSynchronize(g.st), "download");
        if (!ok) {
            return false;
        }
        const std::string where = "replay L" + std::to_string(layer) + " bank " + std::to_string(start_bank) +
                                  " round " + std::to_string(round) + " (B" + std::to_string(rows) + ", commit " +
                                  std::to_string(k) + ")";
        if (memcmp(rg.data(), res.gated.data(), rg.size() * 4) == 0 && rcd == res.codes &&
            memcmp(rsc.data(), res.scales.data(), rsc.size() * 4) == 0) {
            ++same_out;
        } else {
            wr.fail(where + ": the outputs differ from the pre-ME11 kernel's");
        }
        if (memcmp(ref.data(), res.pre.data(), sv * 4) == 0) {
            ++same_pre;
        } else {
            wr.fail(where + ": the stored pre-step matrix (the device replay) differs from the reference's version 0");
        }
        for (int32_t v = 1; v <= rows; ++v) {
            ++prefixes;
            if (!get(ref.data(), g.ref_v(v), sv * 4, g.st) || !cuda_ok(cudaStreamSynchronize(g.st), "download")) {
                return false;
            }
            if (memcmp(ref.data(), res.state.data() + size_t(v - 1) * sv, sv * 4) == 0) {
                ++same_prefix;
            } else {
                wr.fail(where + ": accepted prefix " + std::to_string(v) + " differs from the reference's version");
            }
        }
        g.ref_cur_h = static_cast<uint32_t>(fn::round::version_slot(g.ref_cur_h, k));
        if (!commit(g, k) || !put(g.ref_cur.p, &g.ref_cur_h, 4, g.st) ||
            !cuda_ok(cudaStreamSynchronize(g.st), "commit")) {
            fprintf(stderr, "gd-test: %s: commit\n", where.c_str());
            return false;
        }
        ++round;
    }
    printf("gd-test: replay L%d from bank %d set 0: %d rounds (B 1..%d, every commit 0..B): outputs bitwise %d/%d, "
           "stored pre-step matrix (device replay) bitwise %d/%d, accepted prefixes (host replay) bitwise %d/%d\n",
           layer, start_bank, round, max_b, same_out, round, same_pre, round, same_prefix, prefixes);
    return true;
}

// arguments outside the contract are rejected with nothing enqueued
bool rejects(card_gpu & g) {
    gd::core_args a;
    a.conv = g.cv.f();
    a.z = g.z.f();
    a.alpha = g.alpha.f();
    a.beta = g.beta.f();
    a.dt = g.dt.f();
    a.a = g.a.f();
    a.norm_w = g.norm.f();
    a.state = g.state.f();
    a.rec = g.rec.f();
    a.gsel = g.gsel_word();
    a.gated = g.gated.f();
    a.gated_q = {static_cast<int8_t *>(g.codes.p), g.scales.f(), 1, gd::n_value};
    a.ws = g.ws.p;
    a.rows = 1;
    gd::core_args r0 = a, r6 = a, mis = a, nul = a, qk = a, nocur = a, norec = a, overlap = a;
    nocur.gsel = nullptr;
    norec.rec = nullptr;
    overlap.rec = g.state.f() + gd::state_version_floats; // ME11: the records inside the second bank
    r0.rows = 0;
    r6.rows = 6; // SP2: rows 1..5
    r6.gated_q.rows = 6;
    mis.state = g.state.f() + 1;
    nul.ws = nullptr;
    qk.gated_q.k = 2560;
    gd::conv_args c;
    c.qkv = g.qkv.f();
    c.conv_w = g.conv.f();
    c.hist = g.hist.f();
    c.cur = g.cur_word();
    c.out = g.cv.f();
    c.rows = 0;
    gd::conv_args c1 = c, c2 = c;
    c1.rows = 1;
    c1.hist = g.hist.f() + 1;
    c2.rows = 1;
    c2.cur = nullptr;
    const bool rej = !gd::core(r0, g.st) && !gd::core(r6, g.st) && !gd::core(mis, g.st) && !gd::core(nul, g.st) &&
                     !gd::core(nocur, g.st) && !gd::conv_silu(c2, g.st) && !gd::core(qk, g.st) &&
                     !gd::core(a, nullptr) && !gd::conv_silu(c, g.st) && !gd::conv_silu(c1, g.st) &&
                     !gd::core(norec, g.st) && !gd::core(overlap, g.st);
    printf("gd-test: rejected arguments: %s\n", rej ? "all twelve rejected" : "FAIL");
    return rej && cuda_ok(cudaStreamSynchronize(g.st), "sync");
}

} // namespace

int fnfl_gd_test(int device, const char * mode, const char * model) {
    const std::string md = mode;
    const bool all = md == "all", san = md == "san";
    if (!all && !san && md != "short" && md != "chain" && md != "map" && md != "inv" && md != "replay") {
        fprintf(stderr, "gd-test: mode is all, short, chain, map, inv, replay or san\n");
        return 2;
    }
    if (device != 0 && device != 1) {
        fprintf(stderr, "gd-test: the device is the card, 0 or 1\n");
        return 2;
    }
    const int32_t card = device;
    std::vector<std::string> paths;
    std::vector<fnfl_gguf_dir> dirs;
    std::string err;
    if (!fnfl_gguf_shard_paths(model, paths, err)) {
        fprintf(stderr, "gd-test: %s\n", err.c_str());
        return 1;
    }
    for (const std::string & p : paths) {
        dirs.emplace_back();
        if (!fnfl_gguf_read_dir(p, dirs.back(), err)) {
            fprintf(stderr, "gd-test: %s\n", err.c_str());
            return 1;
        }
    }
    const int32_t layers[2] = {10, 40};
    layer_weights wg[2], wc[2];
    card_map maps[2];
    for (int32_t i = 0; i < 2; ++i) {
        if (!read_layer(dirs, layers[i], wg[i]) || !make_map(layers[i], card, maps[i])) {
            return 1;
        }
        wc[i] = card_weights(wg[i], maps[i]);
    }
    printf(
        "gd-test: card %d, mode %s, weights of layers 10 and 40 from %s (%zu shards; CPU file read, no model load)\n",
        card, mode, model, paths.size());
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaDeviceProp prop = {};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "props")) {
        return 1;
    }
    for (int32_t rows = 1; rows <= gd::max_rows; ++rows) {
        gd::core_resources res;
        if (!gd::resources(rows, res)) {
            return 1;
        }
        printf("gd-test: %s, %d SMs; gdn_core<%d>: %d blocks of %d threads, %d registers, %zu B static shared, %d B "
               "local, %d blocks an SM (occupancy API)\n",
               prop.name, prop.multiProcessorCount, rows, res.blocks, res.threads, res.regs, res.shared_bytes,
               res.local_bytes, res.blocks_per_sm);
    }
    worst wr;
    bool ok = true;
    {
        card_gpu g;
        ok = cuda_ok(cudaStreamCreateWithFlags(&g.st, cudaStreamNonBlocking), "stream") && g.ready() &&
             cuda_ok(cudaMemsetAsync(g.ws.p, 0, g.ws.n, g.st), "zero workspace") && set_cur(g, 0);
        if (ok && (all || md == "short")) {
            ok = rejects(g) && short_cases(g, 10, wc[0], wr) && short_cases(g, 40, wc[1], wr);
        }
        if (ok && (all || md == "chain")) {
            ok = chained(g, 10, wc[0], 4096, {1, 2, 3, 4}, 64, "chain", wr);
        }
        if (ok && (all || md == "map")) {
            ok = mapping(g, card, 10, wg[0], maps[0], wr) && mapping(g, card, 40, wg[1], maps[1], wr);
        }
        if (ok && (all || md == "inv")) {
            ok = invariance(g, 10, wc[0], wr) && invariance(g, 40, wc[1], wr);
        }
        if (ok && (all || md == "replay")) {
            // ME11: from bank 0 and from bank 1 (set 0 both): every bank and set orientation (the round flips both)
            ok = replay_chain(g, 10, wc[0], 0, gd::max_rows, wr) && replay_chain(g, 10, wc[0], 1, gd::max_rows, wr) &&
                 replay_chain(g, 40, wc[1], 0, gd::max_rows, wr) && replay_chain(g, 40, wc[1], 1, gd::max_rows, wr);
            if (ok) {
                printf("gd-test: bank and record-set orientations the launches ran from: %d of 4 (ME11)\n",
                       __builtin_popcount(g.orient_seen));
                if (g.orient_seen != 15u) {
                    wr.fail("the launches did not run from every bank and record set");
                }
            }
        }
        if (ok && san) {
            ok = chained(g, 10, wc[0], 64, {1, 3}, 1, "san", wr) && replay_chain(g, 10, wc[0], 1, 3, wr);
        }
        // SR1: the readers ran at every committed slot
        if (ok && (all || md == "short")) {
            printf("gd-test: committed slots the launches ran at: %d of %d (SR1)\n", __builtin_popcount(g.slots_seen),
                   gd::n_version);
            if (g.slots_seen != (1u << gd::n_version) - 1u) { // SP2: six
                wr.fail("the launches did not run at every committed slot");
            }
        }
        // the counters are back at zero after every launch
        if (ok) {
            std::vector<uint32_t> cnt(gd::n_head);
            ok = get(cnt.data(), static_cast<uint8_t *>(g.ws.p) + gd::workspace_bytes() - cnt.size() * 4,
                     cnt.size() * 4, g.st) &&
                 cuda_ok(cudaStreamSynchronize(g.st), "download");
            for (const uint32_t c : cnt) {
                if (c != 0) {
                    wr.fail("a head's counter is not back at zero");
                    break;
                }
            }
        }
        if (g.st != nullptr) {
            cudaStreamDestroy(g.st);
        }
    }
    ok = ok && wr.ok;
    if (!wr.conds.empty()) {
        std::vector<double> c = wr.conds;
        std::sort(c.begin(), c.end());
        printf("gd-test: cond = rms(o) / (max_j |M_j| / sqrt(128)) over %zu checked rows and heads: min %.3e, p0.01 "
               "%.3e, p1 %.3e, median %.3e\n",
               c.size(), c[0], c[c.size() / 10000], c[c.size() / 100], c[c.size() / 2]);
    }
    printf("gd-test: worst normalized error: outputs %.3e (%s), states %.3e (%s); the F32 reference's: outputs %.3e, "
           "states %.3e; bound 1e-4 (outputs: x max(1, 1e-3 / cond), Amendment 1); %lld failed checks\n",
           wr.out, wr.out_at.c_str(), wr.state, wr.state_at.c_str(), wr.out32, wr.state32,
           static_cast<long long>(wr.n_fail));
    printf("gd-test: the convolution epilogue's post-SiLU output: worst %.3e per row and 128 channels (F32 reference "
           "%.3e), bound 1e-5\n",
           wr.conv, wr.conv32);
    printf("gd-test: outputs under Amendment 1: %lld rows judged under the widened bound (smallest cond %.3e, "
           "non-blocking), %lld rows with cond 0 (outputs exactly 0), worst error / bound %.3f\n",
           static_cast<long long>(wr.n_widened), wr.min_cond, static_cast<long long>(wr.n_zero), wr.ratio);
    printf("gd-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
