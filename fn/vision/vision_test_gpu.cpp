#include "fn/kernels/dense/dense_prompt.h"
#include "fn/vision/encoder.h"
#include "fn/vision/kernels.h"
#include "fn/vision/vision_ref.h"
#include "fn/vision/vision_test.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>

namespace fn::vision::test {

namespace {

struct dbuf {
    void * p = nullptr;
    dbuf()   = default;

    explicit dbuf(size_t n) {
        if (cudaMalloc(&p, n) != cudaSuccess) {
            p = nullptr;
        }
    }

    ~dbuf() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dbuf(const dbuf &)             = delete;
    dbuf & operator=(const dbuf &) = delete;

    template <typename T> T * as() const { return static_cast<T *>(p); }
};

float h2f(uint16_t h) {
    const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t       u;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {  // subnormal: normalize
            int32_t  ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 1024) == 0);
            u = s | (uint32_t(127 - 15 - ee) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f2bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  // nearest even (finite inputs)
}

template <typename T> bool down(std::vector<T> & h, const void * d, size_t n) {
    h.resize(n);
    return cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
}

// A pageable cudaMemcpy to the device can return before its DMA lands, and the tests' non-blocking stream does not
// wait for the legacy stream, so the upload is complete on the device before any kernel is queued.
template <typename T> bool up(void * d, const std::vector<T> & h) {
    return cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
}

// per row max |a - ref| / max |ref|, the worst row
double row_maxrel(const float * a, const double * ref, int32_t rows, int32_t n) {
    double worst = 0;
    for (int32_t r = 0; r < rows; ++r) {
        double e = 0, m = 0;
        for (int32_t c = 0; c < n; ++c) {
            e = std::max(e, std::fabs(double(a[size_t(r) * n + c]) - ref[size_t(r) * n + c]));
            m = std::max(m, std::fabs(ref[size_t(r) * n + c]));
        }
        worst = std::max(worst, m > 0 ? e / m : e);
    }
    return worst;
}

int report(const char * what, int32_t gw, int32_t gh, double err, double tol) {
    const bool ok = err <= tol && std::isfinite(err);
    std::printf("kernel %-14s grid %2dx%-2d (P %4d): worst row max|err|/max|ref| %.3e (tolerance %.0e) %s\n", what, gw,
                gh, gw * gh, err, tol, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

std::vector<float> randn(std::mt19937 & g, size_t n, float sd, float mean = 0.0f) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<float>              v(n);
    for (auto & x : v) {
        x = d(g);
    }
    return v;
}

std::vector<float> randu(std::mt19937 & g, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float>                    v(n);
    for (auto & x : v) {
        x = d(g);
    }
    return v;
}

std::vector<uint16_t> rand_bf16(std::mt19937 & g, size_t n, float sd) {
    const std::vector<float> f = randn(g, n, sd);
    std::vector<uint16_t>    b(n);
    for (size_t i = 0; i < n; ++i) {
        b[i] = f2bf16(f[i]);
    }
    return b;
}

void grid_of(int32_t p, int32_t gw, int32_t & gx, int32_t & gy) {
    const int32_t g = p / 4, s = p % 4;
    gx = 2 * (g % (gw / 2)) + s % 2;
    gy = 2 * (g / (gw / 2)) + s / 2;
}

int32_t round64(int32_t x) {
    return (x + 63) / 64 * 64;
}

// the single-kernel tests at one grid
int kernels_at(int32_t gw, int32_t gh, std::mt19937 & g, cudaStream_t s) {
    const int32_t            P = gw * gh, Pp = round64(P);
    int                      fails = 0;
    // pos_resize
    const std::vector<float> table = randu(g, size_t(pos_side) * pos_side * width, -2.0f, 2.0f);
    dbuf                     dt(table.size() * 4), dx(size_t(P) * width * 4);
    std::vector<float>       pos;
    if (!up(dt.p, table) || !k_pos_resize(dt.as<float>(), gw, gh, dx.as<float>(), s) || cudaStreamSynchronize(s) ||
        !down(pos, dx.p, size_t(P) * width)) {
        return 1;
    }
    fails +=
        report("pos_resize", gw, gh, row_maxrel(pos.data(), ref::pos_resize(table, gw, gh).data(), P, width), tol_pos);
    // patch_embed on top of the resized table
    const std::vector<float> rows = randu(g, size_t(P) * patch_vals, -1.0f, 1.0f);
    const std::vector<float> w = randn(g, size_t(width) * patch_vals, 0.03f), b = randn(g, width, 0.1f);
    dbuf                     dr(rows.size() * 4), dw(w.size() * 4), db(b.size() * 4);
    std::vector<float>       pe;
    if (!up(dr.p, rows) || !up(dw.p, w) || !up(db.p, b) ||
        !k_patch_embed(dr.as<float>(), P, dw.as<float>(), db.as<float>(), dx.as<float>(), s) ||
        cudaStreamSynchronize(s) || !down(pe, dx.p, size_t(P) * width)) {
        return 1;
    }
    const std::vector<double> pos64(pos.begin(), pos.end());
    fails += report("patch_embed", gw, gh,
                    row_maxrel(pe.data(), ref::patch_embed(rows.data(), P, w, b, pos64).data(), P, width), tol_patch);
    // layernorm
    std::vector<float> xl = randn(g, size_t(P) * width, 2.0f);
    for (int32_t r = 0; r < P; ++r) {
        for (int32_t c = 0; c < width; ++c) {
            xl[size_t(r) * width + c] += float(r % 7) - 3.0f;  // a row offset
        }
    }
    const std::vector<float> lg = randu(g, width, 0.5f, 2.0f), lb = randn(g, width, 0.2f);
    dbuf                     dl(xl.size() * 4), dlg(width * 4), dlb(width * 4), dy(size_t(Pp) * width * 2);
    std::vector<uint16_t>    y16;
    if (!up(dl.p, xl) || !up(dlg.p, lg) || !up(dlb.p, lb) ||
        !k_layernorm(dl.as<float>(), P, dlg.as<float>(), dlb.as<float>(), 1e-6f, dy.as<uint16_t>(), s) ||
        cudaStreamSynchronize(s) || !down(y16, dy.p, size_t(P) * width)) {
        return 1;
    }
    {
        std::vector<float> y(y16.size());
        std::transform(y16.begin(), y16.end(), y.begin(), h2f);
        const std::vector<double> x64(xl.begin(), xl.end());
        fails += report("layernorm", gw, gh,
                        row_maxrel(y.data(), ref::layernorm(x64, P, width, lg, lb, 1e-6).data(), P, width), tol_fp16);
    }
    // qkv_rope over 256-row chunks (the encoder's chunk loop), then attention over its Q, K and Vt
    const std::vector<float> qkv = randn(g, size_t(Pp) * qkv_width, 1.5f), qb = randn(g, qkv_width, 0.3f);
    const size_t             hq = size_t(heads) * Pp * head_dim, hv = size_t(heads) * head_pad * Pp;
    dbuf                     dq(qkv.size() * 4), dqb(qb.size() * 4), dQ(hq * 2), dK(hq * 2), dV(hv * 2);
    std::vector<uint16_t>    ones(std::max(hq, hv), 0x3c00);  // 1.0: the pads must come back zero
    ones.resize(hq);
    bool ok = up(dq.p, qkv) && up(dqb.p, qb) && up(dQ.p, ones) && up(dK.p, ones);
    ones.assign(hv, 0x3c00);
    ok = ok && up(dV.p, ones);
    for (int32_t r0 = 0; r0 < Pp && ok; r0 += 256) {
        ok = k_qkv_rope(dq.as<float>() + size_t(r0) * qkv_width, r0, std::min(256, Pp - r0), P, Pp, gw, dqb.as<float>(),
                        dQ.as<uint16_t>(), dK.as<uint16_t>(), dV.as<uint16_t>(), s);
    }
    std::vector<uint16_t> Q, K, V;
    if (!ok || cudaStreamSynchronize(s) || !down(Q, dQ.p, hq) || !down(K, dK.p, hq) || !down(V, dV.p, hv)) {
        return 1;
    }
    {
        std::vector<double> rq(size_t(P) * width), rk(rq.size()), rv(rq.size());
        std::vector<float>  gq(rq.size()), gk(rq.size()), gv(rq.size());
        size_t              pad_bad = 0;
        for (int32_t p = 0; p < Pp; ++p) {
            int32_t gx = 0, gy = 0;
            if (p < P) {
                grid_of(p, gw, gx, gy);
            }
            for (int32_t h = 0; h < heads; ++h) {
                double vq[head_dim], vk[head_dim];
                for (int32_t c = 0; c < head_dim; ++c) {
                    const size_t qi = (size_t(h) * Pp + p) * head_dim + c, vi = (size_t(h) * head_pad + c) * Pp + p;
                    if (p >= P) {
                        pad_bad += Q[qi] != 0 || K[qi] != 0 || V[vi] != 0;
                        continue;
                    }
                    const size_t col = size_t(h) * head_dim + c;
                    vq[c]            = double(qkv[size_t(p) * qkv_width + col]) + qb[col];
                    vk[c]            = double(qkv[size_t(p) * qkv_width + width + col]) + qb[width + col];
                    rv[size_t(p) * width + col] =
                        double(qkv[size_t(p) * qkv_width + 2 * width + col]) + qb[2 * width + col];
                    gq[size_t(p) * width + col] = h2f(Q[qi]);
                    gk[size_t(p) * width + col] = h2f(K[qi]);
                    gv[size_t(p) * width + col] = h2f(V[vi]);
                }
                if (p < P) {
                    ref::rope(vq, gy, gx);
                    ref::rope(vk, gy, gx);
                    std::copy(vq, vq + head_dim, rq.begin() + int64_t(p) * width + h * head_dim);
                    std::copy(vk, vk + head_dim, rk.begin() + int64_t(p) * width + h * head_dim);
                }
            }
        }
        const double e =
            std::max({ row_maxrel(gq.data(), rq.data(), P, width), row_maxrel(gk.data(), rk.data(), P, width),
                       row_maxrel(gv.data(), rv.data(), P, width) });
        fails += report("qkv_rope", gw, gh, pad_bad ? 1.0 : e, tol_fp16);
        // attention against FP64 over the same FP16 values
        dbuf                  dO(size_t(P) * width * 2);
        std::vector<uint16_t> O;
        if (!k_attention(dQ.as<uint16_t>(), dK.as<uint16_t>(), dV.as<uint16_t>(), P, Pp, dO.as<uint16_t>(), s) ||
            cudaStreamSynchronize(s) || !down(O, dO.p, size_t(P) * width)) {
            return 1;
        }
        std::vector<double> ro(size_t(P) * width), hq64(size_t(P) * head_dim), hk64(hq64.size()), hv64(hq64.size()),
            ho(hq64.size());
        for (int32_t h = 0; h < heads; ++h) {
            for (int32_t p = 0; p < P; ++p) {
                for (int32_t c = 0; c < head_dim; ++c) {
                    hq64[size_t(p) * head_dim + c] = gq[size_t(p) * width + h * head_dim + c];
                    hk64[size_t(p) * head_dim + c] = gk[size_t(p) * width + h * head_dim + c];
                    hv64[size_t(p) * head_dim + c] = gv[size_t(p) * width + h * head_dim + c];
                }
            }
            ref::attention(hq64.data(), hk64.data(), hv64.data(), P, ho.data());
            for (int32_t p = 0; p < P; ++p) {
                std::copy(ho.begin() + int64_t(p) * head_dim, ho.begin() + int64_t(p + 1) * head_dim,
                          ro.begin() + int64_t(p) * width + h * head_dim);
            }
        }
        std::vector<float> go(O.size());
        std::transform(O.begin(), O.end(), go.begin(), h2f);
        fails += report("attention", gw, gh, row_maxrel(go.data(), ro.data(), P, width), tol_attn);
    }
    // bias_gelu, bias_residual, bias_add
    {
        const std::vector<float> yv = randn(g, size_t(P) * ffn_pad, 3.0f), bv = randn(g, ffn_pad, 0.5f);
        dbuf                     d1(yv.size() * 4), d2(bv.size() * 4), d3(yv.size() * 2);
        std::vector<uint16_t>    g16;
        if (!up(d1.p, yv) || !up(d2.p, bv) ||
            !k_bias_gelu(d1.as<float>(), P, ffn_pad, d2.as<float>(), d3.as<uint16_t>(), s) ||
            cudaStreamSynchronize(s) || !down(g16, d3.p, yv.size())) {
            return 1;
        }
        std::vector<float>  gg(g16.size());
        std::vector<double> rg(g16.size());
        std::transform(g16.begin(), g16.end(), gg.begin(), h2f);
        for (size_t i = 0; i < rg.size(); ++i) {
            rg[i] = ref::gelu(double(yv[i]) + bv[i % ffn_pad]);
        }
        fails += report("bias_gelu", gw, gh, row_maxrel(gg.data(), rg.data(), P, ffn_pad), tol_fp16);
        const std::vector<float> xr = randn(g, size_t(P) * width, 1.0f), yr = randn(g, size_t(P) * width, 1.0f),
                                 br = randn(g, width, 0.5f);
        dbuf                     e1(xr.size() * 4), e2(yr.size() * 4), e3(br.size() * 4);
        std::vector<float>       got, got2;
        if (!up(e1.p, xr) || !up(e2.p, yr) || !up(e3.p, br) ||
            !k_bias_residual(e2.as<float>(), P, width, e3.as<float>(), e1.as<float>(), s) ||
            !k_bias_add(e2.as<float>(), P, width, e3.as<float>(), s) || cudaStreamSynchronize(s) ||
            !down(got, e1.p, xr.size()) || !down(got2, e2.p, yr.size())) {
            return 1;
        }
        size_t bad = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const float want = (yr[i] + br[i % width]) + xr[i], want2 = yr[i] + br[i % width];
            bad += std::memcmp(&want, &got[i], 4) != 0 || std::memcmp(&want2, &got2[i], 4) != 0;
        }
        std::printf("kernel %-14s grid %2dx%-2d (P %4d): %zu of %zu values differ from F32 on the host %s\n",
                    "bias_res/add", gw, gh, P, bad, 2 * got.size(), bad ? "FAIL" : "PASS");
        fails += bad != 0;
    }
    return fails;
}

// one synthetic block and the merger through the encoder (its chunk loops at chunk 192), against FP64
int block_and_merger(int32_t device, std::mt19937 & g) {
    host_weights w;
    w.profile = 0;
    w.blk.resize(1);
    block_host & b = w.blk[0];
    b.ln1_w        = randu(g, width, 0.5f, 1.5f);
    b.ln1_b        = randn(g, width, 0.1f);
    b.ln2_w        = randu(g, width, 0.5f, 1.5f);
    b.ln2_b        = randn(g, width, 0.1f);
    b.qkv          = rand_bf16(g, size_t(qkv_width) * width, 1.0f / std::sqrt(float(width)));
    b.qkv_b        = randn(g, qkv_width, 0.2f);
    b.o            = rand_bf16(g, size_t(width) * width, 1.0f / std::sqrt(float(width)));
    b.o_b          = randn(g, width, 0.1f);
    b.up           = rand_bf16(g, size_t(ffn) * width, 1.0f / std::sqrt(float(width)));
    b.up_b         = randn(g, ffn, 0.2f);
    b.down         = rand_bf16(g, size_t(width) * ffn, 1.0f / std::sqrt(float(ffn)));
    b.down_b       = randn(g, width, 0.1f);
    w.post_w       = randu(g, width, 0.5f, 1.5f);
    w.post_b       = randn(g, width, 0.1f);
    w.mm0          = rand_bf16(g, size_t(merged) * merged, 1.0f / std::sqrt(float(merged)));
    w.mm0_b        = randn(g, merged, 0.1f);
    w.mm2          = rand_bf16(g, size_t(out_width) * merged, 1.0f / std::sqrt(float(merged)));
    w.mm2_b        = randn(g, out_width, 0.1f);
    device_weights dw;
    encoder        enc;
    std::string    err;
    if (!dw.upload(w, device, 0, 1, false, true, err) || !enc.init(device, 64 * 40, err, 192)) {
        std::printf("block: %s\n", err.c_str());
        return 1;
    }
    int fails = 0;
    for (const auto & gr : { std::make_pair(30, 30), std::make_pair(64, 40) }) {
        const int32_t      gw = gr.first, gh = gr.second, P = gw * gh;
        std::vector<float> x0 = randn(g, size_t(P) * width, 1.0f), x1, out;
        dbuf               dout(size_t(P / 4) * out_width * 4);
        if (!enc.begin_from_residual(x0.data(), gw, gh) || !enc.run_blocks(dw, 0, 1) || !enc.sync() ||
            !down(x1, enc.residual(), x0.size()) || !enc.merge(dw, dout.as<float>()) || !enc.sync() ||
            !down(out, dout.p, size_t(P / 4) * out_width)) {
            return 1;
        }
        std::vector<double> rx(x0.begin(), x0.end());
        ref::block(b, rx, gw, gh, double(1e-6f), 8);
        fails += report("block", gw, gh, row_maxrel(x1.data(), rx.data(), P, width), tol_block);
        // the merger from the GPU's own block output, so its error is the merger's
        const std::vector<double> x164(x1.begin(), x1.end());
        fails += report("merger", gw, gh,
                        row_maxrel(out.data(), ref::merger(w, x164, P, double(1e-6f), 8).data(), P / 4, out_width),
                        tol_merger);
    }
    return fails;
}

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string base_of(const std::string & path) {
    const size_t sl = path.find_last_of('/');
    std::string  b  = sl == std::string::npos ? path : path.substr(sl + 1);
    const size_t d  = b.find_last_of('.');
    return d == std::string::npos ? b : b.substr(0, d);
}

}  // namespace

int run_kernels(int32_t device) {
    if (cudaSetDevice(device) != cudaSuccess) {
        return 1;
    }
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        return 1;
    }
    std::mt19937 g(20261027u);
    int          fails = 0;
    const bool   tab   = rope_table_exact();
    std::printf("kernel rope table: 10000^(-k/18) rounded once from F64: %s\n", tab ? "PASS" : "FAIL");
    fails += !tab;
    for (const auto & gr : { std::make_pair(30, 30), std::make_pair(46, 18), std::make_pair(18, 46) }) {
        fails += kernels_at(gr.first, gr.second, g, s);
    }
    cudaStreamDestroy(s);
    fails += block_and_merger(device, g);
    std::printf("kernels: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

int run_encoder(const std::string &              mmproj,
                const std::vector<std::string> & images,
                const std::string &              ref_dir,
                int32_t                          device,
                int32_t                          threads,
                bool                             fp64) {
    (void) fp64;
    host_weights w;
    std::string  err;
    range_report rr;
    if (!load_host(mmproj, w, err) || !range_check(w, rr)) {
        std::printf("encoder: %s\n", err.empty() ? "the range check failed" : err.c_str());
        return 1;
    }
    std::vector<prepared> imgs(images.size());
    int32_t               maxp = 4;
    for (size_t i = 0; i < images.size(); ++i) {
        const std::vector<uint8_t> bytes = read_file(images[i]);
        if (prepare(bytes.data(), bytes.size(), limits(), imgs[i]) != status::ok) {
            std::printf("encoder: cannot prepare %s\n", images[i].c_str());
            return 1;
        }
        maxp = std::max(maxp, imgs[i].gw * imgs[i].gh);
    }
    device_weights dw;
    encoder        enc;
    if (!dw.upload(w, device, 0, blocks, true, true, err) || !enc.init(device, maxp, err)) {
        std::printf("encoder: %s\n", err.c_str());
        return 1;
    }
    int fails = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        const prepared &   p = imgs[i];
        const int32_t      T = p.nx * p.ny;
        dbuf               dout(size_t(T) * out_width * 4);
        std::vector<float> out;
        if (!enc.encode(dw, p, dout.as<float>()) || !enc.sync() || !down(out, dout.p, size_t(T) * out_width)) {
            std::printf("encoder %s: %s\n", base_of(images[i]).c_str(), enc.error() ? enc.error() : "failed");
            return 1;
        }
        const std::string b = base_of(images[i]);
        {
            // the GPU's rows, for ref16's offline comparison against the FP16-storage reference
            std::ofstream f(ref_dir + "/" + b + ".gpu.bin", std::ios::binary);
            f.write(reinterpret_cast<const char *>(out.data()), int64_t(out.size() * 4));
        }
        // the FP64 rows from ref64 (fnfl-vision-host), or computed here when fp64 is set; otherwise not judged
        std::vector<double> r64(size_t(T) * out_width);
        std::ifstream       f(ref_dir + "/" + b + ".fp64.bin", std::ios::binary);
        bool have = bool(f.read(reinterpret_cast<char *>(r64.data()), int64_t(r64.size() * sizeof(double))));
        if (!have && fp64) {
            r64  = ref::encoder(w, p, threads);
            have = true;
        }
        // the engineer's rule (VIS1, 2026-10-27): the GPU's mean against FP64 at most 1.5x the FP16-storage
        // reference's (ref16, REF_DIR/<image>.fp16e.bin), and its worst row at most the reference engine's own worst
        // row against FP64 (its plain-attention rows, the arm on file for every image)
        std::vector<double> e16(have ? r64.size() : 0);
        std::ifstream       f16(ref_dir + "/" + b + ".fp16e.bin", std::ios::binary);
        const bool          have16 =
            have && bool(f16.read(reinterpret_cast<char *>(e16.data()), int64_t(e16.size() * sizeof(double))));
        dump       eng;
        const bool have_eng =
            have && read_dump(ref_dir + "/" + b + ".rows.bin", eng) && eng.dims[0] == out_width && eng.dims[1] == T;
        if (have && (!have16 || !have_eng)) {
            std::printf("encoder %-16s %4d tokens: FP64 rows but no %s; run ref16 / vis-ref first FAIL\n", b.c_str(), T,
                        have16 ? "reference-engine rows" : "ref16 rows");
            ++fails;
        } else if (have) {
            const row_err      e = compare_rows(out.data(), r64.data(), T, out_width);
            std::vector<float> e16f(e16.begin(), e16.end());
            const row_err      s16     = compare_rows(e16f.data(), r64.data(), T, out_width);
            const row_err      ref     = compare_rows(eng.f32.data(), r64.data(), T, out_width);
            const row_err      g16     = compare_rows(out.data(), e16.data(), T, out_width);
            const bool         ok_mean = e.mean_rel <= tol_enc_mean_x * s16.mean_rel;
            const bool         ok_max  = e.max_rel <= ref.max_rel;
            std::printf(
                "encoder %-16s %4d tokens, GPU against FP64: mean %.3e = %.2fx ref16's %.3e (at most %.1fx) %s; "
                "worst row %.3e (row %d; ref16's worst row %d at %.3e) against the reference engine's %.3e "
                "(row %d) %s\n",
                b.c_str(), T, e.mean_rel, s16.mean_rel > 0 ? e.mean_rel / s16.mean_rel : 0.0, s16.mean_rel,
                tol_enc_mean_x, ok_mean ? "PASS" : "FAIL", e.max_rel, e.worst_row, s16.worst_row, s16.max_rel,
                ref.max_rel, ref.worst_row, ok_max ? "PASS" : "FAIL");
            std::printf("encoder %-16s GPU against ref16: per-row rel L2 max %.3e (row %d) mean %.3e (reported)\n",
                        b.c_str(), g16.max_rel, g16.worst_row, g16.mean_rel);
            fails += !(ok_mean && ok_max);
        } else {
            std::printf("encoder %-16s %4d tokens: no FP64 rows (not required at this size)\n", b.c_str(), T);
        }
        const bool finite = std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
        if (!finite) {
            std::printf("encoder %-16s non-finite output FAIL\n", b.c_str());
            ++fails;
        }
        for (const char * arm : { "rows", "fa.rows" }) {
            dump d;
            if (read_dump(ref_dir + "/" + b + "." + arm + ".bin", d) && d.dims[0] == out_width && d.dims[1] == T) {
                const row_err x = compare_rows(out.data(), d.f32.data(), T, out_width);
                std::printf(
                    "encoder %-16s against the reference engine (CPU, %s attention): per-row rel L2 max %.3e "
                    "mean %.3e, max abs / max %.3e (reported)\n",
                    b.c_str(), std::strcmp(arm, "rows") == 0 ? "plain" : "flash", x.max_rel, x.mean_rel,
                    x.max_abs_over_max);
            }
        }
    }
    std::printf("encoder: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

int run_cap(const std::string & mmproj, const std::string & image, int32_t device) {
    host_weights w;
    std::string  err;
    if (!load_host(mmproj, w, err) || cudaSetDevice(device) != cudaSuccess) {
        std::printf("cap: %s\n", err.c_str());
        return 1;
    }
    const std::vector<uint8_t> bytes = read_file(image);
    prepared                   p;
    if (prepare(bytes.data(), bytes.size(), limits(), p) != status::ok) {
        std::printf("cap: cannot prepare %s\n", image.c_str());
        return 1;
    }
    size_t free0 = 0, free1 = 0, free2 = 0, total = 0;
    cudaFree(nullptr);
    cudaMemGetInfo(&free0, &total);
    device_weights dw;
    if (!dw.upload(w, device, 0, blocks, true, true, err)) {
        std::printf("cap: %s\n", err.c_str());
        return 1;
    }
    cudaMemGetInfo(&free1, &total);
    encoder enc;
    if (!enc.init(device, max_patches, err)) {
        std::printf("cap: %s\n", err.c_str());
        return 1;
    }
    const int32_t T = p.nx * p.ny;
    dbuf          dout(size_t(max_tokens) * out_width * 4);
    cudaMemGetInfo(&free2, &total);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, static_cast<cudaStream_t>(enc.stream()));
    const bool ok = enc.encode(dw, p, dout.as<float>());
    cudaEventRecord(b, static_cast<cudaStream_t>(enc.stream()));
    if (!ok || !enc.sync()) {
        std::printf("cap: encode failed\n");
        return 1;
    }
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    std::vector<float> out;
    down(out, dout.p, size_t(T) * out_width);
    const bool finite = std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); });
    std::printf("cap: image %s %dx%d -> %dx%d, %d patches, %d tokens\n", base_of(image).c_str(), p.src.w, p.src.h,
                p.size.w, p.size.h, p.gw * p.gh, T);
    std::printf(
        "cap: weights on the card %.1f MiB (measured %.1f MiB); scratch planned %.1f MiB for %d patches, "
        "measured %.1f MiB with the %d-token output rows (%.1f MiB)\n",
        dw.bytes / 1048576.0, (free0 - free1) / 1048576.0, enc.scratch_bytes() / 1048576.0, max_patches,
        (free1 - free2) / 1048576.0, max_tokens, double(max_tokens) * out_width * 4 / 1048576.0);
    std::printf("cap: one encode %.1f ms (printed, not acted on); output finite: %s\n", ms, finite ? "yes" : "NO");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return finite && T <= max_tokens ? 0 : 1;
}

int run_resources(int32_t device) {
    if (cudaSetDevice(device) != cudaSuccess) {
        return 1;
    }
    k_resources   r[16];
    const int32_t n = vision_kernel_resources(r, 16);
    for (int32_t i = 0; i < n; ++i) {
        std::printf("resources %-14s %3d registers, %d local bytes, %5d shared bytes, %d CTAs an SM\n", r[i].name,
                    r[i].regs, r[i].local_bytes, r[i].shared_bytes, r[i].ctas_per_sm);
    }
    fn::dense::pd_resources t{};
    if (fn::dense::pd_tc_resources(t)) {
        std::printf("resources %-14s %3d registers, %d local bytes, %5d shared bytes, %d CTAs an SM\n", "pd_tc", t.regs,
                    t.local_bytes, t.shared_bytes, t.ctas_per_sm);
    }
    return n > 0 ? 0 : 1;
}

}  // namespace fn::vision::test
