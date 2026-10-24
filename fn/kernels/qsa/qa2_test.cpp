#include "fn/kernels/qsa/qa2_test.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/shapes.h"
#include "fn/model/state.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/kv_q8_ref.h"
#include "fn/tools/test_dump.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

// The reference is written from the pinned lines (66e11c3), never from a GPU kernel (R16):
//   qwen4exp.cpp 1607-1692 (build_layer_attn): Qcur_full = wq x; Qcur = view of each head's first 256, rms-normed
//       with attn_q_norm; Kcur rms-normed with attn_k_norm; gate = each head's second 256; V not normed; Q and K
//       through ggml_rope_multi (IMROPE, sections 11, 11, 10, 0); kq_scale = 1 / sqrt(256) unless attention.scale;
//       the attention output times sigmoid(gate).
//   qwen4exp.cpp 1364-1605 (build_attn_qsa): K and V stored to the cache, then attention over each row's selected cells
//       (the union form keeps each row's own mask, qwen4exp-qsa-compact.h 148-156: the softmax is over the row's set).
//   ops.cpp 5984-5999 (rope_yarn): theta = freq_scale theta_extrap (ext_factor 0); cos and sin times mscale.
//   ops.cpp 6017-6087 (ggml_mrope_cache_init): theta_{t,h,w,e} start at the positions and are multiplied by
//       theta_scale after each pair (F32); sector = (i0 / 2) % 32; IMROPE: h when sector % 3 == 1 and sector < 33, w
//       when % 3 == 2 and < 30, t when % 3 == 0 and < 33, else e.
//   ops.cpp 6089-6106 (rotate_pairs, NEOX/IMROPE offset n_dims / 2): x0 cos - x1 sin, x0 sin + x1 cos.
//   ops.cpp 6152: theta_scale = powf(freq_base, -2.0f / n_dims).
// The rope angles use the same F32 product chain as the kernel (part of the model's arithmetic); sin, cos and every
// other operation are FP64. The attention reads the BF16 K and V the kernel wrote.
// KV8 (--kv-format q8): the caches in the 8-bit format, the scales in QSA layer 5's column of the scale rows; the
// reference reads the decoded cache (float(code) * s), the new cells' codes and scales are bitwise the CPU quantizer
// (fn/quant/kv_q8_ref) of the launch's own F32 k and of v. Both formats: the quantizer's stress rows as v.

namespace {

namespace at = fn::qsa::attn;
namespace qs = fn::qsa;

constexpr int32_t HD = at::head_dim;
constexpr int32_t NH = at::n_head;
constexpr int32_t QS = at::q_stride;
constexpr int32_t IHD = qs::head_dim;
constexpr int32_t R = qs::ratio;
constexpr int32_t max_depth = 196608;
constexpr int32_t n_cells = max_depth + 16;
constexpr double out_bound = 1e-5;
constexpr double kf32_bound = 1e-6;
constexpr int32_t SS = fn::model::kv_q8_scale_stride; // KV8: the scale rows' stride
constexpr int32_t q8_col = 4 * 5;                     // KV8: QSA layer 5's 4 scales in a row

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "qa2-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

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
        return at_elem + count <= n &&
               cuda_ok(cudaMemcpyAsync(p + at_elem, h, count * sizeof(T), cudaMemcpyHostToDevice, st), "upload");
    }

    bool put(const std::vector<T> & h, cudaStream_t st) { return put(h.data(), h.size(), st); }

    bool get(T * h, size_t count, cudaStream_t st, size_t at_elem = 0) const {
        return at_elem + count <= n &&
               cuda_ok(cudaMemcpyAsync(h, p + at_elem, count * sizeof(T), cudaMemcpyDeviceToHost, st), "download") &&
               cuda_ok(cudaStreamSynchronize(st), "download sync");
    }
};

uint16_t bf16_rn(float f) {
    uint32_t u = 0;
    memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float from_bf16(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f = 0.0f;
    memcpy(&f, &u, 4);
    return f;
}

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
};

// ---- the model's norms and keys ----

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
        fprintf(stderr, "qa2-test: %s: missing, not F32 or not %lld values\n", name.c_str(), static_cast<long long>(n));
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
        fprintf(stderr, "qa2-test: %s: read failed\n", name.c_str());
    }
    return ok;
}

// layer 11's q and k norms; the rope and attention keys, which must be what the kernel implements
bool read_model(const std::string & model, std::vector<float> & q_norm, std::vector<float> & k_norm) {
    std::vector<std::string> paths;
    std::vector<fnfl_gguf_dir> dirs;
    std::string err;
    if (!fnfl_gguf_shard_paths(model, paths, err)) {
        fprintf(stderr, "qa2-test: %s\n", err.c_str());
        return false;
    }
    for (const std::string & p : paths) {
        dirs.emplace_back();
        if (!fnfl_gguf_read_dir(p, dirs.back(), err)) {
            fprintf(stderr, "qa2-test: %s\n", err.c_str());
            return false;
        }
    }
    if (!read_f32(dirs, "blk.11.attn_q_norm.weight", HD, q_norm) ||
        !read_f32(dirs, "blk.11.attn_k_norm.weight", HD, k_norm)) {
        return false;
    }
    const std::string a = std::string(fn::arch) + ".";
    const std::string & f0 = paths[0];
    double base = 0.0, eps = 0.0;
    std::vector<int64_t> sections;
    const int64_t n_rot = fnfl_gguf_read_int(f0, a + "rope.dimension_count");
    const int64_t key_len = fnfl_gguf_read_int(f0, a + "attention.key_length");
    const bool ok_base = fnfl_gguf_read_float(f0, a + "rope.freq_base", base);
    const bool ok_eps = fnfl_gguf_read_float(f0, a + "attention.layer_norm_rms_epsilon", eps);
    const bool ok_sec = fnfl_gguf_read_ints(f0, a + "rope.dimension_sections", sections);
    printf("  gguf: rope.freq_base %.1f, rope.dimension_count %lld, rope.dimension_sections [", base,
           static_cast<long long>(n_rot));
    for (size_t i = 0; i < sections.size(); ++i) {
        printf("%s%lld", i ? ", " : "", static_cast<long long>(sections[i]));
    }
    printf("], attention.layer_norm_rms_epsilon %.9g, attention.key_length %lld\n", eps,
           static_cast<long long>(key_len));
    bool ok = ok_base && static_cast<float>(base) == 10000000.0f && n_rot == qs::n_rot && ok_eps &&
              static_cast<float>(eps) == fn::rms_eps && ok_sec && sections.size() == 4 && sections[0] == 11 &&
              sections[1] == 11 && sections[2] == 10 && sections[3] == 0 && key_len == HD;
    // keys whose presence would change the arithmetic: each must be absent or neutral
    struct neutral {
        const char * key;
        double value;
    };
    const neutral keys[] = {{"rope.scaling.factor", 1.0},      {"rope.scale_linear", 1.0},
                            {"rope.scaling.attn_factor", 1.0}, {"rope.scaling.yarn_ext_factor", 0.0},
                            {"attention.scale", 0.0},          {"attn_logit_softcapping", 0.0},
                            {"final_logit_softcapping", 0.0},  {"attention.max_alibi_bias", 0.0}};
    for (const neutral & k : keys) {
        double v = 0.0;
        const bool has = fnfl_gguf_has_key(f0, a + k.key);
        const bool num = has && fnfl_gguf_read_float(f0, a + k.key, v);
        printf("  gguf: %s %s\n", k.key, !has ? "absent" : num ? std::to_string(v).c_str() : "present (not a float)");
        ok = ok && (!has || (num && v == k.value));
    }
    const bool has_type = fnfl_gguf_has_key(f0, a + "rope.scaling.type");
    printf("  gguf: rope.scaling.type %s; kq_scale = 1 / sqrt(256), freq_scale 1, ext_factor 0, attn_factor 1\n",
           has_type ? "present" : "absent");
    ok = ok && !has_type;
    if (!ok) {
        fprintf(stderr, "qa2-test: the model's rope or attention keys are not the ones this kernel implements\n");
    }
    return ok;
}

// ---- the FP64 reference ----

// pair i's angle: the F32 chain and the IMROPE sector rule (ggml_mrope_cache_init)
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
    for (int i = 0; i < qs::n_rot / 2; ++i) {
        const double th = double(theta_f32(pos4, i, ts)), c = std::cos(th), s = std::sin(th);
        const double x0 = y[i], x1 = y[i + qs::n_rot / 2];
        y[i] = x0 * c - x1 * s;
        y[i + qs::n_rot / 2] = x0 * s + x1 * c;
    }
}

// the cells of row t of a body, in order
std::vector<int32_t> row_cells(const std::vector<uint32_t> & body, const std::vector<int32_t> & offsets, int32_t t) {
    const uint32_t * w = body.data() + offsets[size_t(t)] / 4;
    const int32_t K = int32_t(w[0]), tail = int32_t(w[1]);
    std::vector<int32_t> c;
    for (int32_t i = 0; i < K; ++i) {
        for (int32_t r = 0; r < R; ++r) {
            c.push_back(R * int32_t(w[2 + i]) + r);
        }
    }
    for (int32_t i = 0; i < tail; ++i) {
        c.push_back(int32_t(w[2 + K + i]));
    }
    return c;
}

// ---- one case ----

struct case_in {
    std::string name;
    int32_t depth = 0, rows = 0;
    std::vector<float> qfull, kv_k, kv_v; // [B][12][512], [B][256], [B][256]
    std::vector<int32_t> pos;             // [4][B]
    std::vector<uint32_t> body;
    std::vector<int32_t> offsets;
};

struct case_out {
    std::vector<float> out, k_f32;
    std::vector<int8_t> codes;
    std::vector<float> scales;
    std::vector<uint16_t> k_new, v_new; // [B][256]
    std::vector<int8_t> k8, v8;         // KV8: the new cells' codes [B][256] and scales [B][4]
    std::vector<uint16_t> ks, vs;
    uint32_t error = 0;
};

struct worst {
    double out = 0.0, k_ulp = 0.0, k_f32 = 0.0;
    std::string out_at, k_at;
    int64_t checks = 0, failed = 0;

    void check(bool ok, const char * what, const std::string & name) {
        ++checks;
        if (!ok) {
            ++failed;
            printf("    FAIL %s: %s\n", name.c_str(), what);
        }
    }
};

struct bench_ctx {
    cudaStream_t st = nullptr;
    bool q8 = false;              // KV8: the caches in the 8-bit format
    std::vector<uint16_t> hk, hv; // the caches' host mirror
    std::vector<int8_t> hk8, hv8; // KV8: its codes [cells][256] and the layer's scales [cells][4]
    std::vector<uint16_t> hks, hvs;
    dev_buf<uint16_t> kc{size_t(n_cells) * HD}, vc{size_t(n_cells) * HD};
    dev_buf<uint16_t> ksc{size_t(n_cells) * SS}, vsc{size_t(n_cells) * SS}; // KV8: the scale rows
    dev_buf<float> qfull{size_t(at::max_rows) * NH * QS}, kvk{size_t(at::max_rows) * HD},
        kvv{size_t(at::max_rows) * HD};
    dev_buf<float> qn{HD}, kn{HD}, out{size_t(at::max_rows) * at::out_width}, kf32{size_t(at::max_rows) * HD};
    dev_buf<float> scales{size_t(at::max_rows) * at::out_width / 16};
    dev_buf<int8_t> codes{size_t(at::max_rows) * at::out_width};
    dev_buf<int32_t> pos{size_t(4) * at::max_rows}, offsets{size_t(at::max_rows) + 1};
    dev_buf<int32_t> dep{1}; // the depth on the device (TG1: select and attend read it there)
    dev_buf<uint32_t> body{size_t(at::max_rows) * qs::max_row_words};
    dev_buf<uint8_t> aws{at::workspace_bytes()};
    at::workspace ws = {};
    float ts = at::rope_theta_scale();

    bool ready() const {
        return kc.p && vc.p && qfull.p && kvk.p && kvv.p && qn.p && kn.p && out.p && kf32.p && scales.p && codes.p &&
               pos.p && offsets.p && body.p && aws.p && dep.p && ksc.p && vsc.p;
    }

    // the mirror's K or V value of a cell (BF16, or KV8's decoded code)
    double kval(int64_t cell, int32_t e) const {
        return q8 ? double(fn::quant::kv_q8_value(hk8[size_t(cell * HD + e)], hks[size_t(cell * 4 + e / 64)]))
                  : double(from_bf16(hk[size_t(cell * HD + e)]));
    }
    double vval(int64_t cell, int32_t e) const {
        return q8 ? double(fn::quant::kv_q8_value(hv8[size_t(cell * HD + e)], hvs[size_t(cell * 4 + e / 64)]))
                  : double(from_bf16(hv[size_t(cell * HD + e)]));
    }

    // KV8: cells c0 .. c0 + n - 1 of a mirror to the device (codes, and the scales into the layer's column)
    bool put_q8(bool v, int64_t c0, int64_t n) {
        const std::vector<int8_t> & h8 = v ? hv8 : hk8;
        const std::vector<uint16_t> & hs = v ? hvs : hks;
        uint16_t * dc = v ? vc.p : kc.p;
        uint16_t * ds = v ? vsc.p : ksc.p;
        return cuda_ok(cudaMemcpyAsync(reinterpret_cast<int8_t *>(dc) + c0 * HD, h8.data() + c0 * HD, size_t(n) * HD,
                                       cudaMemcpyHostToDevice, st),
                       "upload codes") &&
               cuda_ok(cudaMemcpy2DAsync(ds + c0 * SS + q8_col, size_t(SS) * 2, hs.data() + c0 * 4, 8, 8, size_t(n),
                                         cudaMemcpyHostToDevice, st),
                       "upload scales");
    }
    // KV8: a row of F32 values into the mirror's cell, quantized by the CPU reference
    void set_q8(bool v, int64_t cell, const float * x) {
        fn::quant::kv_q8_quantize(x, HD, (v ? hv8 : hk8).data() + cell * HD, (v ? hvs : hks).data() + cell * 4);
    }
};

bool run_attend(bench_ctx & c, const case_in & in, case_out & o) {
    const int32_t B = in.rows;
    bool ok =
        c.dep.put(std::vector<int32_t>{in.depth}, c.st) && c.qfull.put(in.qfull, c.st) && c.kvk.put(in.kv_k, c.st) &&
        c.kvv.put(in.kv_v, c.st) && c.pos.put(in.pos, c.st) && c.body.put(in.body, c.st) &&
        c.offsets.put(in.offsets, c.st) &&
        // the new cells hold junk: the launch must write them, and must not read them before it does
        cuda_ok(cudaMemsetAsync(c.kc.p + size_t(in.depth) * HD / (c.q8 ? 2 : 1), 0x7f, size_t(B) * HD * (c.q8 ? 1 : 2),
                                c.st),
                "junk") &&
        cuda_ok(cudaMemsetAsync(c.vc.p + size_t(in.depth) * HD / (c.q8 ? 2 : 1), 0x7f, size_t(B) * HD * (c.q8 ? 1 : 2),
                                c.st),
                "junk") &&
        (!c.q8 ||
         (cuda_ok(cudaMemset2DAsync(c.ksc.p + size_t(in.depth) * SS + q8_col, size_t(SS) * 2, 0x7f, 8, size_t(B), c.st),
                  "junk") &&
          cuda_ok(cudaMemset2DAsync(c.vsc.p + size_t(in.depth) * SS + q8_col, size_t(SS) * 2, 0x7f, 8, size_t(B), c.st),
                  "junk"))) &&
        cuda_ok(cudaMemsetAsync(c.out.p, 0xff, c.out.n * 4, c.st), "junk") &&
        cuda_ok(cudaMemsetAsync(c.ws.error, 0, 4, c.st), "error word");
    at::args a = {};
    a.qfull = c.qfull.p;
    a.kv_k = c.kvk.p;
    a.kv_v = c.kvv.p;
    a.q_norm = c.qn.p;
    a.k_norm = c.kn.p;
    a.body = c.body.p;
    a.offsets = c.offsets.p;
    a.pos = c.pos.p;
    a.k_cache = c.kc.p;
    a.v_cache = c.vc.p;
    a.depth = c.dep.p;
    a.max_cells = n_cells;
    a.theta_scale = c.ts;
    a.out = c.out.p;
    a.codes = c.codes.p;
    a.scales = c.scales.p;
    a.k_f32 = c.kf32.p;
    a.ws = c.ws;
    if (c.q8) {
        a.k_scale = c.ksc.p + q8_col;
        a.v_scale = c.vsc.p + q8_col;
    }
    if (!ok || !at::attend(B, a, c.st)) {
        fprintf(stderr, "qa2-test: %s: attend not enqueued: %s\n", in.name.c_str(),
                cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    o.out.resize(size_t(B) * at::out_width);
    o.codes.resize(size_t(B) * at::out_width);
    o.scales.resize(size_t(B) * at::out_width / 16);
    o.k_f32.resize(size_t(B) * HD);
    bool got = cuda_ok(cudaStreamSynchronize(c.st), "attend") && c.out.get(o.out.data(), o.out.size(), c.st) &&
               c.codes.get(o.codes.data(), o.codes.size(), c.st) &&
               c.scales.get(o.scales.data(), o.scales.size(), c.st) &&
               c.kf32.get(o.k_f32.data(), o.k_f32.size(), c.st) &&
               cuda_ok(cudaMemcpyAsync(&o.error, c.ws.error, 4, cudaMemcpyDeviceToHost, c.st), "error") &&
               cuda_ok(cudaStreamSynchronize(c.st), "error");
    if (c.q8) {
        // KV8: the new cells' codes and the layer's scales
        o.k8.resize(size_t(B) * HD);
        o.v8.resize(size_t(B) * HD);
        o.ks.resize(size_t(B) * 4);
        o.vs.resize(size_t(B) * 4);
        const int64_t d = in.depth;
        got = got &&
              cuda_ok(cudaMemcpy(o.k8.data(), reinterpret_cast<const int8_t *>(c.kc.p) + d * HD, o.k8.size(),
                                 cudaMemcpyDeviceToHost),
                      "codes") &&
              cuda_ok(cudaMemcpy(o.v8.data(), reinterpret_cast<const int8_t *>(c.vc.p) + d * HD, o.v8.size(),
                                 cudaMemcpyDeviceToHost),
                      "codes") &&
              cuda_ok(cudaMemcpy2D(o.ks.data(), 8, c.ksc.p + d * SS + q8_col, size_t(SS) * 2, 8, size_t(B),
                                   cudaMemcpyDeviceToHost),
                      "scales") &&
              cuda_ok(cudaMemcpy2D(o.vs.data(), 8, c.vsc.p + d * SS + q8_col, size_t(SS) * 2, 8, size_t(B),
                                   cudaMemcpyDeviceToHost),
                      "scales");
    } else {
        o.k_new.resize(size_t(B) * HD);
        o.v_new.resize(size_t(B) * HD);
        got = got && c.kc.get(o.k_new.data(), o.k_new.size(), c.st, size_t(in.depth) * HD) &&
              c.vc.get(o.v_new.data(), o.v_new.size(), c.st, size_t(in.depth) * HD);
    }
    const std::string t = "qa2-" + in.name + "-B" + std::to_string(B) + "-d" + std::to_string(in.depth);
    return got && fn::tools::test_dump(t + "-out", o.out.data(), o.out.size() * 4) &&
           fn::tools::test_dump(t + "-codes", o.codes.data(), o.codes.size()) &&
           fn::tools::test_dump(t + "-scales", o.scales.data(), o.scales.size() * 4) &&
           fn::tools::test_dump(t + "-kf32", o.k_f32.data(), o.k_f32.size() * 4) &&
           fn::tools::test_dump(t + "-knew", c.q8 ? static_cast<const void *>(o.k8.data()) : o.k_new.data(),
                                c.q8 ? o.k8.size() : o.k_new.size() * 2) &&
           fn::tools::test_dump(t + "-vnew", c.q8 ? static_cast<const void *>(o.v8.data()) : o.v_new.data(),
                                c.q8 ? o.v8.size() : o.v_new.size() * 2) &&
           (!c.q8 || (fn::tools::test_dump(t + "-kscale", o.ks.data(), o.ks.size() * 2) &&
                      fn::tools::test_dump(t + "-vscale", o.vs.data(), o.vs.size() * 2))) &&
           fn::tools::test_dump(t + "-error", &o.error, sizeof(o.error));
}

int32_t pos_of(const case_in & in, int p, int32_t t) {
    return in.pos[size_t(p) * in.rows + t];
}

// checks one run against the FP64 reference; updates the caches' mirror with the new cells
void check_case(bench_ctx & c, const case_in & in, const case_out & o, const std::vector<float> & q_norm,
                const std::vector<float> & k_norm, worst & w) {
    const int32_t B = in.rows;
    w.check(o.error == 0, "the acquire timed out", in.name);
    std::vector<double> y(HD);
    for (int32_t t = 0; t < B; ++t) {
        const int32_t p4[4] = {pos_of(in, 0, t), pos_of(in, 1, t), pos_of(in, 2, t), pos_of(in, 3, t)};
        norm_rope_ref(in.kv_k.data() + size_t(t) * HD, k_norm.data(), p4, c.ts, y.data());
        double kmax = 0.0, kerr = 0.0, ulp_ratio = 0.0;
        bool v_ok = true, k8_ok = true;
        for (int32_t d = 0; d < HD; ++d) {
            kmax = std::max(kmax, std::fabs(y[size_t(d)]));
            kerr = std::max(kerr, std::fabs(double(o.k_f32[size_t(t) * HD + d]) - y[size_t(d)]));
            if (c.q8) {
                continue;
            }
            const double kb = double(from_bf16(o.k_new[size_t(t) * HD + d]));
            const double ulp = y[size_t(d)] == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(y[size_t(d)]) - 7);
            ulp_ratio = std::max(ulp_ratio, ulp > 0.0 ? std::fabs(kb - y[size_t(d)]) / ulp : (kb == 0.0 ? 0.0 : 1e9));
            v_ok = v_ok && o.v_new[size_t(t) * HD + d] == bf16_rn(in.kv_v[size_t(t) * HD + d]);
        }
        if (c.q8) {
            // KV8: the codes and scales bitwise the CPU quantizer of the launch's own F32 k, and of v
            int8_t rc[HD];
            uint16_t rs[4];
            fn::quant::kv_q8_quantize(o.k_f32.data() + size_t(t) * HD, HD, rc, rs);
            k8_ok =
                memcmp(rc, o.k8.data() + size_t(t) * HD, HD) == 0 && memcmp(rs, o.ks.data() + size_t(t) * 4, 8) == 0;
            fn::quant::kv_q8_quantize(in.kv_v.data() + size_t(t) * HD, HD, rc, rs);
            v_ok = memcmp(rc, o.v8.data() + size_t(t) * HD, HD) == 0 && memcmp(rs, o.vs.data() + size_t(t) * 4, 8) == 0;
        }
        const double kn_err = kmax > 0.0 ? kerr / kmax : kerr;
        if (kn_err > w.k_f32) {
            w.k_f32 = kn_err;
        }
        if (ulp_ratio > w.k_ulp) {
            w.k_ulp = ulp_ratio;
            w.k_at = in.name + " row " + std::to_string(t);
        }
        w.check(kn_err <= kf32_bound, "the kernel's F32 k outside 1e-6 of the FP64 k", in.name);
        if (c.q8) {
            w.check(k8_ok, "the K codes or scales are not the CPU quantizer's of the launch's F32 k", in.name);
            w.check(v_ok, "the V codes or scales are not the CPU quantizer's of v", in.name);
            // the mirror: the new cells as the kernel wrote them (the reference reads them)
            const int64_t cell = int64_t(in.depth) + t;
            std::copy(o.k8.begin() + int64_t(t) * HD, o.k8.begin() + int64_t(t + 1) * HD, c.hk8.begin() + cell * HD);
            std::copy(o.v8.begin() + int64_t(t) * HD, o.v8.begin() + int64_t(t + 1) * HD, c.hv8.begin() + cell * HD);
            std::copy(o.ks.begin() + int64_t(t) * 4, o.ks.begin() + int64_t(t + 1) * 4, c.hks.begin() + cell * 4);
            std::copy(o.vs.begin() + int64_t(t) * 4, o.vs.begin() + int64_t(t + 1) * 4, c.hvs.begin() + cell * 4);
            continue;
        }
        w.check(ulp_ratio <= 1.0, "a K cache value more than one BF16 ulp from the FP64 k", in.name);
        w.check(v_ok, "the V cache is not bf16(v)", in.name);
        // the mirror: the new cells as the kernel wrote them (the reference reads them)
        std::copy(o.k_new.begin() + int64_t(t) * HD, o.k_new.begin() + int64_t(t + 1) * HD,
                  c.hk.begin() + (int64_t(in.depth) + t) * HD);
        std::copy(o.v_new.begin() + int64_t(t) * HD, o.v_new.begin() + int64_t(t + 1) * HD,
                  c.hv.begin() + (int64_t(in.depth) + t) * HD);
    }
    std::vector<double> qd(HD), s, ref(HD);
    for (int32_t t = 0; t < B; ++t) {
        const int32_t p4[4] = {pos_of(in, 0, t), pos_of(in, 1, t), pos_of(in, 2, t), pos_of(in, 3, t)};
        const std::vector<int32_t> cells = row_cells(in.body, in.offsets, t);
        bool causal = true;
        for (const int32_t cc : cells) {
            causal = causal && cc <= in.depth + t;
        }
        w.check(causal, "a selected cell after the row's position", in.name);
        for (int32_t h = 0; h < NH; ++h) {
            const float * qrow = in.qfull.data() + (size_t(t) * NH + h) * QS;
            norm_rope_ref(qrow, q_norm.data(), p4, c.ts, qd.data());
            s.assign(cells.size(), 0.0);
            double m = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < cells.size(); ++i) {
                double d = 0.0;
                for (int32_t e = 0; e < HD; ++e) {
                    d += qd[size_t(e)] * c.kval(cells[i], e);
                }
                s[i] = d / 16.0;
                m = std::max(m, s[i]);
            }
            std::fill(ref.begin(), ref.end(), 0.0);
            double l = 0.0;
            for (size_t i = 0; i < cells.size(); ++i) {
                const double p = std::exp(s[i] - m);
                l += p;
                for (int32_t e = 0; e < HD; ++e) {
                    ref[size_t(e)] += p * c.vval(cells[i], e);
                }
            }
            double rmax = 0.0, emax = 0.0;
            for (int32_t e = 0; e < HD; ++e) {
                const double g = double(qrow[HD + e]);
                ref[size_t(e)] = ref[size_t(e)] / l * (1.0 / (1.0 + std::exp(-g)));
                rmax = std::max(rmax, std::fabs(ref[size_t(e)]));
                emax = std::max(emax, std::fabs(double(o.out[size_t(t) * at::out_width + size_t(h) * HD + size_t(e)]) -
                                                ref[size_t(e)]));
            }
            const double err = rmax > 0.0 ? emax / rmax : emax;
            if (!(err <= w.out) || std::isnan(err)) {
                w.out = std::isnan(err) ? INFINITY : std::max(w.out, err);
                w.out_at = in.name + " row " + std::to_string(t) + " head " + std::to_string(h);
            }
            w.check(err <= out_bound,
                    ("output outside 1e-5 (row " + std::to_string(t) + " head " + std::to_string(h) + ", " +
                     std::to_string(err) + ")")
                        .c_str(),
                    in.name);
        }
    }
    // the activation format: bitwise act_ref of the kernel's own F32 output
    std::vector<int8_t> rc(o.codes.size());
    std::vector<float> rs(o.scales.size());
    fn::quant::act_quantize_ref(o.out.data(), B, at::out_width, rc.data(), rs.data());
    w.check(rc == o.codes && memcmp(rs.data(), o.scales.data(), rs.size() * 4) == 0,
            "the activation format is not act_ref of the F32 output", in.name);
}

bool bitwise_equal(const case_out & a, const case_out & b) {
    return memcmp(a.out.data(), b.out.data(), a.out.size() * 4) == 0 && a.codes == b.codes &&
           memcmp(a.scales.data(), b.scales.data(), a.scales.size() * 4) == 0 && a.k_new == b.k_new &&
           a.v_new == b.v_new && memcmp(a.k_f32.data(), b.k_f32.data(), a.k_f32.size() * 4) == 0 && a.k8 == b.k8 &&
           a.v8 == b.v8 && a.ks == b.ks && a.vs == b.vs;
}

// one row of a batch as its own B = 1 step at depth + t
case_in row_alone(const case_in & in, int32_t t) {
    case_in r;
    r.name = in.name + " row " + std::to_string(t) + " alone";
    r.depth = in.depth + t;
    r.rows = 1;
    r.qfull.assign(in.qfull.begin() + int64_t(t) * NH * QS, in.qfull.begin() + int64_t(t + 1) * NH * QS);
    r.kv_k.assign(in.kv_k.begin() + int64_t(t) * HD, in.kv_k.begin() + int64_t(t + 1) * HD);
    r.kv_v.assign(in.kv_v.begin() + int64_t(t) * HD, in.kv_v.begin() + int64_t(t + 1) * HD);
    r.pos = {pos_of(in, 0, t), pos_of(in, 1, t), pos_of(in, 2, t), pos_of(in, 3, t)};
    r.body.assign(in.body.begin() + in.offsets[size_t(t)] / 4, in.body.begin() + in.offsets[size_t(t) + 1] / 4);
    r.offsets = {0, in.offsets[size_t(t) + 1] - in.offsets[size_t(t)]};
    return r;
}

bool run_and_check(bench_ctx & c, const case_in & in, const std::vector<float> & q_norm,
                   const std::vector<float> & k_norm, worst & w, bool invariance) {
    case_out o1, o2;
    if (!run_attend(c, in, o1)) {
        return false;
    }
    check_case(c, in, o1, q_norm, k_norm, w);
    if (!run_attend(c, in, o2)) {
        return false;
    }
    w.check(bitwise_equal(o1, o2), "a rerun is not bitwise identical", in.name);
    if (invariance) {
        for (int32_t t = 0; t < in.rows; ++t) {
            case_out o;
            if (!run_attend(c, row_alone(in, t), o)) {
                return false;
            }
            const bool eq =
                memcmp(o.out.data(), o1.out.data() + size_t(t) * at::out_width, size_t(at::out_width) * 4) == 0 &&
                std::equal(o.codes.begin(), o.codes.end(), o1.codes.begin() + int64_t(t) * at::out_width) &&
                memcmp(o.scales.data(), o1.scales.data() + size_t(t) * at::out_width / 16,
                       size_t(at::out_width) / 16 * 4) == 0 &&
                std::equal(o.k_new.begin(), o.k_new.end(), o1.k_new.begin() + int64_t(t) * HD) &&
                std::equal(o.v_new.begin(), o.v_new.end(), o1.v_new.begin() + int64_t(t) * HD) &&
                std::equal(o.k8.begin(), o.k8.end(), o1.k8.begin() + int64_t(t) * HD) &&
                std::equal(o.v8.begin(), o.v8.end(), o1.v8.begin() + int64_t(t) * HD) &&
                std::equal(o.ks.begin(), o.ks.end(), o1.ks.begin() + int64_t(t) * 4) &&
                std::equal(o.vs.begin(), o.vs.end(), o1.vs.begin() + int64_t(t) * 4);
            w.check(eq, ("row " + std::to_string(t) + " at B = 1 differs from the batch").c_str(), in.name);
        }
    }
    printf("  %-44s rows %d, %zu + %zu cells: worst so far %.3g (%s)\n", in.name.c_str(), in.rows,
           row_cells(in.body, in.offsets, 0).size(),
           in.rows > 1 ? row_cells(in.body, in.offsets, in.rows - 1).size() : 0, w.out, w.out_at.c_str());
    fflush(stdout);
    return true;
}

void fill_inputs(case_in & in, rng & r) {
    const int32_t B = in.rows;
    in.qfull.resize(size_t(B) * NH * QS);
    for (float & x : in.qfull) {
        x = r.uniform(-1.5f, 1.5f);
    }
    in.kv_k.resize(size_t(B) * HD);
    in.kv_v.resize(size_t(B) * HD);
    for (float & x : in.kv_k) {
        x = r.uniform(-2.0f, 2.0f);
    }
    for (float & x : in.kv_v) {
        x = r.uniform(-2.0f, 2.0f);
    }
    in.pos.resize(size_t(4) * B);
    for (int p = 0; p < 4; ++p) {
        for (int32_t t = 0; t < B; ++t) {
            in.pos[size_t(p) * B + t] = in.depth + t;
        }
    }
}

// the selection from QA1b's select on the synthetic index keys, read back
struct selector {
    cudaStream_t st = nullptr;
    dev_buf<uint16_t> keys{size_t(n_cells) * IHD};
    dev_buf<float> sums{qs::summary_bytes(n_cells / R) / 4}, iq{size_t(at::max_rows) * qs::n_idx_head * IHD};
    dev_buf<float> ikn{IHD}, iqn{IHD};
    dev_buf<uint8_t> wsm{qs::workspace_bytes()};
    std::vector<uint16_t> hkeys;

    bool init(rng & r) {
        hkeys.resize(size_t(n_cells) * IHD);
        for (uint16_t & k : hkeys) {
            k = bf16_rn(r.uniform(-1.0f, 1.0f));
        }
        std::vector<float> q(iq.n), kn(IHD), qn(IHD);
        for (float & x : q) {
            x = r.uniform(-1.0f, 1.0f);
        }
        for (int32_t e = 0; e < IHD; ++e) {
            kn[size_t(e)] = r.uniform(0.5f, 1.5f);
            qn[size_t(e)] = r.uniform(0.5f, 1.5f);
        }
        const qs::update_args u = {keys.p, ikn.p, sums.p, 0, n_cells / R, qs::model_rope()};
        return keys.p && sums.p && iq.p && ikn.p && iqn.p && wsm.p && keys.put(hkeys, st) && iq.put(q, st) &&
               ikn.put(kn, st) && iqn.put(qn, st) &&
               cuda_ok(cudaMemsetAsync(wsm.p, 0, qs::workspace_bytes(), st), "select workspace") &&
               qs::summary_update(u, st) && cuda_ok(cudaStreamSynchronize(st), "summaries");
    }

    bool select(case_in & in, bench_ctx & c) {
        const qs::select_args sa = {sums.p,           iq.p,     iqn.p,      c.dep.p, n_cells, qs::model_rope(),
                                    qs::carve(wsm.p), c.body.p, c.offsets.p};
        in.body.resize(size_t(qs::body_words(in.depth, in.rows)));
        in.offsets.resize(size_t(in.rows) + 1);
        return c.dep.put(std::vector<int32_t>{in.depth}, st) && qs::select(in.rows, sa, st) &&
               c.body.get(in.body.data(), in.body.size(), st) &&
               c.offsets.get(in.offsets.data(), in.offsets.size(), st);
    }
};

// the HX7 case: the three rows' selections of QA1's fixture (layer 3, decode 0), as bodies at depth
bool make_hx7(const std::string & path, case_in & in) {
    std::ifstream f(path, std::ios::binary);
    int32_t hdr[4] = {};
    if (!f.read(reinterpret_cast<char *>(hdr), sizeof(hdr)) || hdr[0] != 0x31415141 || hdr[2] != 3 || hdr[3] != 2051) {
        fprintf(stderr, "qa2-test: %s is not QA1's HX7 fixture\n", path.c_str());
        return false;
    }
    std::vector<int32_t> cells(size_t(3) * 2051);
    if (!f.read(reinterpret_cast<char *>(cells.data()), static_cast<std::streamsize>(cells.size() * 4))) {
        return false;
    }
    in.body.clear();
    in.offsets = {0};
    for (int32_t t = 0; t < 3; ++t) {
        std::set<int32_t> blocks;
        for (int32_t i = 0; i < 2048; i += R) {
            blocks.insert(cells[size_t(t) * 2051 + i] / R);
        }
        const int32_t q = in.depth + t, tail = (q + 1) % R;
        in.body.push_back(uint32_t(blocks.size()));
        in.body.push_back(uint32_t(tail));
        for (const int32_t b : blocks) {
            in.body.push_back(uint32_t(b));
        }
        for (int32_t i = 0; i < tail; ++i) {
            in.body.push_back(uint32_t((q + 1) / R * R + i));
        }
        in.offsets.push_back(int32_t(in.body.size() * 4));
    }
    return in.body.size() == size_t(qs::body_words(in.depth, 3));
}

// the raw index-key write inside summary_update: across a block that completes in the step, and with no block
bool key_write_cases(selector & sel, rng & r, worst & w) {
    struct kc {
        int32_t depth, rows;
    };
    const kc cases[] = {{8194, 3}, {8193, 2}, {196606, 4}};
    dev_buf<float> ik{size_t(at::max_rows) * IHD}, s1{qs::summary_bytes(n_cells / R) / 4},
        s2{qs::summary_bytes(n_cells / R) / 4};
    dev_buf<float> ikn{IHD};
    dev_buf<int32_t> dep{1};
    if (!ik.p || !s1.p || !s2.p || !ikn.p || !dep.p) {
        return false;
    }
    std::vector<float> kn(IHD, 1.0f);
    bool ok = ikn.put(kn, sel.st);
    for (const kc & k : cases) {
        std::vector<float> h(size_t(k.rows) * IHD);
        for (float & x : h) {
            x = r.uniform(-1.0f, 1.0f);
        }
        const int32_t first = k.depth / R;                    // the block of the first new cell
        const int32_t count = (k.depth + k.rows) / R - first; // blocks completed in the step (0 or 1)
        qs::round_update_args u;
        u.keys = sel.keys.p;
        u.k_norm = ikn.p;
        u.summaries = s1.p;
        u.rope = qs::model_rope();
        u.ik = ik.p;
        u.depth = dep.p;
        u.rows = k.rows;
        u.max_cells = n_cells;
        // the cache holds junk in the new cells: the launch must take them from ik
        ok = ok && ik.put(h, sel.st) && dep.put(std::vector<int32_t>{k.depth}, sel.st) &&
             cuda_ok(cudaMemsetAsync(sel.keys.p + size_t(k.depth) * IHD, 0x55, size_t(k.rows) * IHD * 2, sel.st),
                     "junk") &&
             qs::summary_update(u, sel.st);
        std::vector<uint16_t> got(size_t(k.rows) * IHD);
        ok = ok && sel.keys.get(got.data(), got.size(), sel.st, size_t(k.depth) * IHD);
        bool keys_ok = true;
        for (size_t i = 0; i < got.size(); ++i) {
            keys_ok = keys_ok && got[i] == bf16_rn(h[i]);
        }
        const std::string name = "key write d" + std::to_string(k.depth) + " B" + std::to_string(k.rows);
        w.check(keys_ok, "the raw index keys are not bf16(ik)", name);
        if (count > 0) {
            // QA1b's path from the cache, now that the keys are in it
            const qs::update_args u2 = {sel.keys.p, ikn.p, s2.p, first, count, qs::model_rope()};
            std::vector<float> a, b;
            const size_t t0 = size_t(first) / qs::tile_blocks * qs::tile_blocks * IHD;
            const size_t n =
                (size_t(first + count + qs::tile_blocks - 1) / qs::tile_blocks * qs::tile_blocks) * IHD - t0;
            a.resize(n);
            b.resize(n);
            ok = ok && qs::summary_update(u2, sel.st) && s1.get(a.data(), n, sel.st, t0) &&
                 s2.get(b.data(), n, sel.st, t0);
            bool same = true;
            for (int32_t bk = first; bk < first + count; ++bk) {
                for (int32_t e = 0; e < IHD; ++e) {
                    const size_t idx = size_t(bk / qs::tile_blocks) * size_t(qs::n_chunk * qs::tile_blocks * 4) +
                                       size_t(e / 4) * size_t(qs::tile_blocks * 4) + size_t(bk % qs::tile_blocks) * 4 +
                                       size_t(e % 4) - t0;
                    same = same && memcmp(&a[idx], &b[idx], 4) == 0;
                }
            }
            w.check(same, "a pooled summary differs from the cache path's", name);
        }
        printf("  %-44s %d new cells, %d blocks pooled\n", name.c_str(), k.rows, count);
        // restore the mirror's keys for the selections that follow
        ok = ok && sel.keys.put(sel.hkeys.data() + size_t(k.depth) * IHD, size_t(k.rows) * IHD, sel.st,
                                size_t(k.depth) * IHD);
    }
    return ok && cuda_ok(cudaStreamSynchronize(sel.st), "key write");
}

} // namespace

int fnfl_qa2_test(int device, const std::string & mode, const std::string & model, const std::string & fixture,
                  bool q8) {
    if (mode != "all" && mode != "san") {
        fprintf(stderr, "qa2-test: mode is all or san\n");
        return 2;
    }
    std::vector<float> q_norm, k_norm;
    if (!read_model(model, q_norm, k_norm)) {
        return 1;
    }
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    printf("  KV format: %s\n", q8 ? "q8 (KV8: int8 codes, FP16 scales in QSA layer 5's column)" : "bf16");
    for (const bool f : {false, true}) {
        at::resources res = {};
        if (at::kernel_resources(res, f)) {
            printf("  attend (%s): %d registers, %d B local, %d B shared, %d CTAs an SM\n", f ? "q8" : "bf16", res.regs,
                   res.local_bytes, res.shared_bytes, res.ctas_per_sm);
        }
    }
    bench_ctx c;
    if (!c.ready() || !cuda_ok(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking), "stream")) {
        return 1;
    }
    c.ws = at::carve(c.aws.p);
    c.q8 = q8;
    rng r{20261012};
    bool ok = true;
    if (q8) {
        // KV8: each history cell's F32 K and V row quantized by the CPU reference
        c.hk8.resize(size_t(n_cells) * HD);
        c.hv8.resize(size_t(n_cells) * HD);
        c.hks.resize(size_t(n_cells) * 4);
        c.hvs.resize(size_t(n_cells) * 4);
        std::vector<float> row(HD);
        for (int32_t v = 0; v < 2; ++v) {
            for (int64_t cell = 0; cell < n_cells; ++cell) {
                for (float & x : row) {
                    x = r.uniform(-2.0f, 2.0f);
                }
                c.set_q8(v == 1, cell, row.data());
            }
        }
        ok = cuda_ok(cudaMemsetAsync(c.ksc.p, 0, c.ksc.n * 2, c.st), "scales") &&
             cuda_ok(cudaMemsetAsync(c.vsc.p, 0, c.vsc.n * 2, c.st), "scales") && c.put_q8(false, 0, n_cells) &&
             c.put_q8(true, 0, n_cells);
    } else {
        c.hk.resize(size_t(n_cells) * HD);
        c.hv.resize(size_t(n_cells) * HD);
        for (uint16_t & k : c.hk) {
            k = bf16_rn(r.uniform(-2.0f, 2.0f));
        }
        for (uint16_t & v : c.hv) {
            v = bf16_rn(r.uniform(-2.0f, 2.0f));
        }
        ok = c.kc.put(c.hk, c.st) && c.vc.put(c.hv, c.st);
    }
    selector sel;
    sel.st = c.st;
    ok = ok && c.qn.put(q_norm, c.st) && c.kn.put(k_norm, c.st) &&
         cuda_ok(cudaMemsetAsync(c.aws.p, 0, at::workspace_bytes(), c.st), "workspace") && sel.init(r);
    worst w;
    const std::vector<int32_t> depths =
        mode == "san" ? std::vector<int32_t>{65536} : std::vector<int32_t>{0, 1, 5, 2047, 2051, 8192, 65536, 196608};
    for (const int32_t d : depths) {
        for (int32_t B = 1; B <= at::max_rows && ok; ++B) {
            if (mode == "san" && B != 3) {
                continue;
            }
            case_in in;
            in.name = "d" + std::to_string(d) + " B" + std::to_string(B);
            in.depth = d;
            in.rows = B;
            fill_inputs(in, r);
            ok = sel.select(in, c) && run_and_check(c, in, q_norm, k_norm, w, B >= 3);
        }
    }
    if (ok && mode == "all") {
        // HX7's selection pattern at 196,608 cells, B = 3
        case_in in;
        in.name = "hx7 d196608 B3";
        in.depth = max_depth;
        in.rows = 3;
        fill_inputs(in, r);
        ok = make_hx7(fixture, in) && run_and_check(c, in, q_norm, k_norm, w, true);
    }
    if (ok && mode == "all") {
        // distinct rope position planes (the sector rule)
        case_in in;
        in.name = "planes t h w e d8192 B2";
        in.depth = 8192;
        in.rows = 2;
        fill_inputs(in, r);
        for (int32_t t = 0; t < 2; ++t) {
            in.pos[size_t(1) * 2 + t] = 3 * (in.depth + t) + 7;
            in.pos[size_t(2) * 2 + t] = (in.depth + t) / 2 + 11;
            in.pos[size_t(3) * 2 + t] = 123 + t;
        }
        ok = sel.select(in, c) && run_and_check(c, in, q_norm, k_norm, w, true);
    }
    if (ok && mode == "all") {
        // one cell's K scaled so its scores reach about +-1e4: the max subtraction and the others' underflow
        case_in in;
        in.name = "extreme score d8192 B3";
        in.depth = 8192;
        in.rows = 3;
        fill_inputs(in, r);
        ok = sel.select(in, c);
        const int32_t cell = ok ? row_cells(in.body, in.offsets, 0)[100] : 0;
        std::vector<double> qd(HD);
        const int32_t p4[4] = {in.depth, in.depth, in.depth, in.depth};
        norm_rope_ref(in.qfull.data(), q_norm.data(), p4, c.ts, qd.data());
        if (q8) {
            // KV8: the row of +-1024 quantized (codes +-127 under one scale a group)
            std::vector<float> row(HD);
            for (int32_t e = 0; e < HD; ++e) {
                row[size_t(e)] = qd[size_t(e)] >= 0.0 ? 1024.0f : -1024.0f;
            }
            const std::vector<int8_t> saved8(c.hk8.begin() + int64_t(cell) * HD,
                                             c.hk8.begin() + int64_t(cell + 1) * HD);
            const std::vector<uint16_t> saveds(c.hks.begin() + int64_t(cell) * 4,
                                               c.hks.begin() + int64_t(cell + 1) * 4);
            c.set_q8(false, cell, row.data());
            ok = ok && c.put_q8(false, cell, 1) && run_and_check(c, in, q_norm, k_norm, w, false);
            std::copy(saved8.begin(), saved8.end(), c.hk8.begin() + int64_t(cell) * HD);
            std::copy(saveds.begin(), saveds.end(), c.hks.begin() + int64_t(cell) * 4);
            ok = ok && c.put_q8(false, cell, 1);
        } else {
            std::vector<uint16_t> row(HD);
            for (int32_t e = 0; e < HD; ++e) {
                row[size_t(e)] = bf16_rn(qd[size_t(e)] >= 0.0 ? 1024.0f : -1024.0f);
            }
            const std::vector<uint16_t> saved(c.hk.begin() + int64_t(cell) * HD, c.hk.begin() + int64_t(cell + 1) * HD);
            std::copy(row.begin(), row.end(), c.hk.begin() + int64_t(cell) * HD);
            ok = ok && c.kc.put(row.data(), HD, c.st, size_t(cell) * HD) &&
                 run_and_check(c, in, q_norm, k_norm, w, false);
            std::copy(saved.begin(), saved.end(), c.hk.begin() + int64_t(cell) * HD);
            ok = ok && c.kc.put(saved.data(), HD, c.st, size_t(cell) * HD);
        }
    }
    if (ok && mode == "all") {
        // every score tied: one K row in every history cell the rows select
        case_in in;
        in.name = "tied scores d8192 B3";
        in.depth = 8192;
        in.rows = 3;
        fill_inputs(in, r);
        ok = sel.select(in, c);
        // the history's scores tie (every selected cell below depth); the rows' own new cells are normed and roped
        // from kv_k, so they cannot be made equal to the cached row
        if (q8) {
            // KV8: cell 0's codes and scales in every history cell
            const std::vector<int8_t> saved8(c.hk8.begin(), c.hk8.begin() + int64_t(in.depth) * HD);
            const std::vector<uint16_t> saveds(c.hks.begin(), c.hks.begin() + int64_t(in.depth) * 4);
            for (int32_t cc = 1; cc < in.depth; ++cc) {
                std::copy(c.hk8.begin(), c.hk8.begin() + HD, c.hk8.begin() + int64_t(cc) * HD);
                std::copy(c.hks.begin(), c.hks.begin() + 4, c.hks.begin() + int64_t(cc) * 4);
            }
            ok = ok && c.put_q8(false, 0, in.depth) && run_and_check(c, in, q_norm, k_norm, w, false);
            std::copy(saved8.begin(), saved8.end(), c.hk8.begin());
            std::copy(saveds.begin(), saveds.end(), c.hks.begin());
            ok = ok && c.put_q8(false, 0, in.depth);
        } else {
            std::vector<uint16_t> saved(size_t(in.depth) * HD), same(size_t(in.depth) * HD);
            std::copy(c.hk.begin(), c.hk.begin() + int64_t(in.depth) * HD, saved.begin());
            for (int32_t cc = 0; cc < in.depth; ++cc) {
                std::copy(c.hk.begin(), c.hk.begin() + HD, same.begin() + int64_t(cc) * HD);
            }
            std::copy(same.begin(), same.end(), c.hk.begin());
            ok = ok && c.kc.put(same.data(), same.size(), c.st) && run_and_check(c, in, q_norm, k_norm, w, false);
            std::copy(saved.begin(), saved.end(), c.hk.begin());
            ok = ok && c.kc.put(saved.data(), saved.size(), c.st);
        }
    }
    if (ok && mode == "all") {
        // the causal rule: later rows' new cells hold extreme values that must not reach earlier rows
        case_in in;
        in.name = "causal d8192 B3";
        in.depth = 8192;
        in.rows = 3;
        fill_inputs(in, r);
        for (int32_t t = 1; t < 3; ++t) {
            for (int32_t e = 0; e < HD; ++e) {
                in.kv_k[size_t(t) * HD + e] = e % 2 ? 300.0f : -300.0f;
                in.kv_v[size_t(t) * HD + e] = 3000.0f;
            }
        }
        ok = sel.select(in, c) && run_and_check(c, in, q_norm, k_norm, w, true);
    }
    if (ok) {
        // KV8's quantizer stress rows (fn/quant/kv_q8_ref.h) as the new cells' v, B = 4 a launch: in q8 the codes and
        // scales bitwise the CPU quantizer's, in BF16 bitwise bf16(v); the attention checked as every case
        const std::vector<float> st = fn::quant::kv_q8_stress_rows(0x6b5638ull, mode == "san" ? 0 : 8);
        const int32_t n_rows = static_cast<int32_t>(st.size() / HD);
        for (int32_t r0 = 0; ok && r0 < n_rows; r0 += 4) {
            case_in in;
            in.rows = std::min(4, n_rows - r0);
            in.depth = 4096 + 8 * r0;
            in.name = "quantizer stress rows " + std::to_string(r0) + " d" + std::to_string(in.depth);
            fill_inputs(in, r);
            std::copy(st.begin() + int64_t(r0) * HD, st.begin() + int64_t(r0 + in.rows) * HD, in.kv_v.begin());
            ok = sel.select(in, c) && run_and_check(c, in, q_norm, k_norm, w, false);
        }
    }
    if (ok && mode == "all") {
        ok = key_write_cases(sel, r, w);
    }
    if (c.st != nullptr) {
        cudaStreamDestroy(c.st);
    }
    if (!ok) {
        fprintf(stderr, "qa2-test: FAILED to run (%s)\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    printf("qa2-test: %s%s (%lld checks, %lld failed): worst output error %.3g normalized (bound 1e-5; %s), worst K "
           "cache %.3f BF16 ulp (bound 1; %s), worst F32 k %.3g normalized (bound 1e-6)\n",
           w.failed == 0 ? "PASS" : "FAIL", q8 ? " [q8: K and V codes and scales bitwise the CPU quantizer's]" : "",
           static_cast<long long>(w.checks), static_cast<long long>(w.failed), w.out, w.out_at.c_str(), w.k_ulp,
           w.k_at.c_str(), w.k_f32);
    return w.failed == 0 ? 0 : 1;
}
