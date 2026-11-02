#include "fn/vision/vision_ref.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace fn::vision::ref {

namespace {

template <typename F> void parallel_rows(int32_t rows, int32_t threads, F f) {
    threads = std::max(1, std::min(threads, rows));
    std::vector<std::thread> t;
    for (int32_t i = 0; i < threads; ++i) {
        t.emplace_back([=, &f] {
            for (int32_t r = i; r < rows; r += threads) {
                f(r);
            }
        });
    }
    for (auto & x : t) {
        x.join();
    }
}

void grid_of(int32_t p, int32_t gw, int32_t & gx, int32_t & gy) {
    const int32_t g = p / 4, s = p % 4;
    gx = 2 * (g % (gw / 2)) + s % 2;
    gy = 2 * (g / (gw / 2)) + s / 2;
}

bool fp16_storage = false;

// v rounded to the nearest FP16 value (ties to even; FP16 subnormals below 2^-14)
double r16(double v) {
    const double a = std::fabs(v);
    if (a == 0 || !std::isfinite(a)) {
        return v;
    }
    int e = 0;
    std::frexp(a, &e);  // a in [2^(e-1), 2^e)
    const double step = std::ldexp(1.0, std::max(e - 11, -24));
    return std::nearbyint(v / step) * step;
}

void r16_all(std::vector<double> & v) {
    if (fp16_storage) {
        for (double & e : v) {
            e = r16(e);
        }
    }
}

}  // namespace

void set_fp16_storage(bool on) {
    fp16_storage = on;
}

std::vector<double> pos_resize(const std::vector<float> & table, int32_t gw, int32_t gh) {
    const int32_t       P = gw * gh;
    std::vector<double> out(size_t(P) * width);
    for (int32_t p = 0; p < P; ++p) {
        int32_t gx, gy;
        grid_of(p, gw, gx, gy);
        const double  sx = double(gx) * (pos_side - 1) / double(gw - 1);
        const double  sy = double(gy) * (pos_side - 1) / double(gh - 1);
        int32_t       x0 = int32_t(std::floor(sx)), y0 = int32_t(std::floor(sy));
        const int32_t x1 = std::min(x0 + 1, pos_side - 1), y1 = std::min(y0 + 1, pos_side - 1);
        x0              = std::min(x0, pos_side - 1);
        y0              = std::min(y0, pos_side - 1);
        const double dx = std::clamp(sx - x0, 0.0, 1.0), dy = std::clamp(sy - y0, 0.0, 1.0);
        for (int32_t c = 0; c < width; ++c) {
            const double a             = table[size_t(y0 * pos_side + x0) * width + c];
            const double b             = table[size_t(y0 * pos_side + x1) * width + c];
            const double e             = table[size_t(y1 * pos_side + x0) * width + c];
            const double d             = table[size_t(y1 * pos_side + x1) * width + c];
            out[size_t(p) * width + c] = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + e * (1 - dx) * dy + d * dx * dy;
        }
    }
    return out;
}

std::vector<double> patch_embed(const float *               rows,
                                int32_t                     P,
                                const std::vector<float> &  w,
                                const std::vector<float> &  b,
                                const std::vector<double> & pos) {
    std::vector<double> out(size_t(P) * width);
    parallel_rows(P, 8, [&](int32_t p) {
        const float * x = rows + size_t(p) * patch_vals;
        for (int32_t o = 0; o < width; ++o) {
            const float * wo = w.data() + size_t(o) * patch_vals;
            double        s  = 0;
            for (int32_t k = 0; k < patch_vals; ++k) {
                s += double(wo[k]) * double(x[k]);
            }
            out[size_t(p) * width + o] = (s + b[size_t(o)]) + pos[size_t(p) * width + o];
        }
    });
    return out;
}

std::vector<double> layernorm(const std::vector<double> & x,
                              int32_t                     rows,
                              int32_t                     n,
                              const std::vector<float> &  g,
                              const std::vector<float> &  b,
                              double                      eps) {
    std::vector<double> y(size_t(rows) * n);
    for (int32_t r = 0; r < rows; ++r) {
        const double * v = x.data() + size_t(r) * n;
        double         m = 0, q = 0;
        for (int32_t i = 0; i < n; ++i) {
            m += v[i];
        }
        m /= n;
        for (int32_t i = 0; i < n; ++i) {
            q += (v[i] - m) * (v[i] - m);
        }
        const double rs = 1.0 / std::sqrt(q / n + eps);
        for (int32_t i = 0; i < n; ++i) {
            y[size_t(r) * n + i] = (v[i] - m) * rs * g[size_t(i)] + b[size_t(i)];
        }
    }
    return y;
}

void rope(double * v, int32_t gy, int32_t gx) {
    for (int32_t i = 0; i < 36; ++i) {
        const double pos = i < rope_pairs ? gy : gx;
        const double th  = pos * std::pow(10000.0, -double(i % rope_pairs) / double(rope_pairs));
        const double c = std::cos(th), s = std::sin(th);
        const double x0 = v[i], x1 = v[i + 36];
        v[i]      = x0 * c - x1 * s;
        v[i + 36] = x0 * s + x1 * c;
    }
}

void attention(const double * q, const double * k, const double * v, int32_t P, double * out) {
    const double scale = 1.0 / std::sqrt(double(head_dim));
    parallel_rows(P, 8, [&](int32_t i) {
        std::vector<double> s(static_cast<size_t>(P));
        double              m = -INFINITY;
        for (int32_t j = 0; j < P; ++j) {
            double d = 0;
            for (int32_t c = 0; c < head_dim; ++c) {
                d += q[size_t(i) * head_dim + c] * k[size_t(j) * head_dim + c];
            }
            s[size_t(j)] = d * scale;
            m            = std::max(m, s[size_t(j)]);
        }
        double l = 0;
        for (int32_t j = 0; j < P; ++j) {
            s[size_t(j)] = std::exp(s[size_t(j)] - m);
            l += s[size_t(j)];
            if (fp16_storage) {
                s[size_t(j)] = r16(s[size_t(j)]);  // the GPU sums l before rounding P for the PV product
            }
        }
        for (int32_t c = 0; c < head_dim; ++c) {
            double a = 0;
            for (int32_t j = 0; j < P; ++j) {
                a += s[size_t(j)] * v[size_t(j) * head_dim + c];
            }
            out[size_t(i) * head_dim + c] = fp16_storage ? r16(a / l) : a / l;
        }
    });
}

double gelu(double v) {
    return 0.5 * v * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (v + 0.044715 * v * v * v)));
}

std::vector<double> product(const std::vector<double> &   x,
                            int32_t                       rows,
                            int32_t                       k,
                            const std::vector<uint16_t> & W,
                            int32_t                       n,
                            const std::vector<float> &    b,
                            int32_t                       threads) {
    std::vector<double> wd(size_t(n) * k);
    for (size_t i = 0; i < wd.size(); ++i) {
        wd[i] = bf16_f32(W[i]);
    }
    std::vector<double> y(size_t(rows) * n);
    parallel_rows(rows, threads, [&](int32_t r) {
        const double * xr = x.data() + size_t(r) * k;
        for (int32_t o = 0; o < n; ++o) {
            const double * wo = wd.data() + size_t(o) * k;
            double         s  = 0;
            for (int32_t i = 0; i < k; ++i) {
                s += wo[i] * xr[i];
            }
            y[size_t(r) * n + o] = s + (b.empty() ? 0.0 : double(b[size_t(o)]));
        }
    });
    return y;
}

void block(const block_host & b, std::vector<double> & x, int32_t gw, int32_t gh, double eps, int32_t threads) {
    const int32_t       P = gw * gh;
    std::vector<double> q(size_t(P) * head_dim), k(q.size()), v(q.size()), o(q.size());
    std::vector<double> h = layernorm(x, P, width, b.ln1_w, b.ln1_b, eps);
    r16_all(h);
    const std::vector<double> qkv = product(h, P, width, b.qkv, qkv_width, b.qkv_b, threads);
    std::vector<double>       att(size_t(P) * width);
    for (int32_t hd = 0; hd < heads; ++hd) {
        for (int32_t p = 0; p < P; ++p) {
            int32_t gx, gy;
            grid_of(p, gw, gx, gy);
            for (int32_t c = 0; c < head_dim; ++c) {
                const size_t base           = size_t(p) * qkv_width + size_t(hd) * head_dim + c;
                q[size_t(p) * head_dim + c] = qkv[base];
                k[size_t(p) * head_dim + c] = qkv[base + width];
                v[size_t(p) * head_dim + c] = qkv[base + 2 * width];
            }
            rope(&q[size_t(p) * head_dim], gy, gx);
            rope(&k[size_t(p) * head_dim], gy, gx);
        }
        r16_all(q);
        r16_all(k);
        r16_all(v);
        attention(q.data(), k.data(), v.data(), P, o.data());
        for (int32_t p = 0; p < P; ++p) {
            std::copy(o.begin() + int64_t(p) * head_dim, o.begin() + int64_t(p + 1) * head_dim,
                      att.begin() + int64_t(p) * width + int64_t(hd) * head_dim);
        }
    }
    const std::vector<double> ao = product(att, P, width, b.o, width, b.o_b, threads);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = ao[i] + x[i];
    }
    std::vector<double> h2 = layernorm(x, P, width, b.ln2_w, b.ln2_b, eps);
    r16_all(h2);
    std::vector<double> u = product(h2, P, width, b.up, ffn, b.up_b, threads);
    for (double & e : u) {
        e = gelu(e);
    }
    r16_all(u);
    const std::vector<double> dn = product(u, P, ffn, b.down, width, b.down_b, threads);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = dn[i] + x[i];
    }
}

std::vector<double> merger(const host_weights &        w,
                           const std::vector<double> & x,
                           int32_t                     P,
                           double                      eps,
                           int32_t                     threads) {
    // post_ln, then four consecutive patch rows are one 4608-wide token row
    std::vector<double> h = layernorm(x, P, width, w.post_w, w.post_b, eps);
    r16_all(h);
    std::vector<double> m = product(h, P / 4, merged, w.mm0, merged, w.mm0_b, threads);
    for (double & e : m) {
        e = gelu(e);
    }
    r16_all(m);
    return product(m, P / 4, merged, w.mm2, out_width, w.mm2_b, threads);
}

std::vector<double> encoder(const host_weights & w, const prepared & img, int32_t threads) {
    const int32_t       P   = img.gw * img.gh;
    const double        eps = vision_profiles[w.profile].eps;
    std::vector<double> x   = patch_embed(img.rows.data(), P, w.patch, w.patch_b, pos_resize(w.pos, img.gw, img.gh));
    for (const block_host & b : w.blk) {
        block(b, x, img.gw, img.gh, eps, threads);
    }
    return merger(w, x, P, eps, threads);
}

}  // namespace fn::vision::ref
