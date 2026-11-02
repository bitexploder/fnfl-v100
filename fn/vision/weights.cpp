#include "fn/vision/weights.h"

#include "fn/borrowed/gguf_tensors.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace fn::vision {

namespace {

std::string block_name(const vtensor_rule & r, int32_t b) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), r.name, b);
    return buf;
}

const char * fmt_type(vfmt f) {
    return f == vfmt::bf16 ? "bf16" : "f32";
}

// the first mismatch between the file and profile p, or empty
std::string mismatch(const std::string & path, const fnfl_gguf_dir & dir, const vision_profile & p) {
    std::string s;
    if (!fnfl_gguf_read_string(path, "clip.projector_type", s) || s != p.projector_type) {
        return "clip.projector_type '" + s + "', not '" + p.projector_type + "'";
    }

    const struct {
        const char * key;
        int64_t      want;
    } ints[] = {
        { "clip.vision.block_count",          p.n_block    },
        { "clip.vision.embedding_length",     p.n_embd     },
        { "clip.vision.feed_forward_length",  p.n_ff       },
        { "clip.vision.attention.head_count", p.n_head     },
        { "clip.vision.patch_size",           p.patch_size },
        { "clip.vision.spatial_merge_size",   p.merge_size },
        { "clip.vision.projection_dim",       p.proj_dim   },
    };

    for (const auto & k : ints) {
        if (fnfl_gguf_read_int(path, k.key) != k.want) {
            return std::string(k.key) + " is not " + std::to_string(k.want);
        }
    }
    double eps = 0;
    if (!fnfl_gguf_read_float(path, "clip.vision.attention.layer_norm_epsilon", eps) || eps != p.eps) {
        return "clip.vision.attention.layer_norm_epsilon is not 1e-6";
    }
    std::vector<double> mean, sd;
    if (!fnfl_gguf_read_floats(path, "clip.vision.image_mean", mean) ||
        !fnfl_gguf_read_floats(path, "clip.vision.image_std", sd) || mean.size() != 3 || sd.size() != 3) {
        return "clip.vision.image_mean / image_std are not three floats";
    }
    for (int32_t c = 0; c < 3; ++c) {
        if (float(mean[c]) != p.mean || float(sd[c]) != p.std_) {
            return "clip.vision.image_mean / image_std are not the profile's";
        }
    }
    std::vector<bool> b;
    if (!fnfl_gguf_read_bools(path, "clip.use_gelu", b) || b.size() != 1 || b[0] != p.use_gelu) {
        return "clip.use_gelu is not the profile's";
    }
    if (fnfl_gguf_read_bools(path, "clip.use_silu", b) && !b.empty() && b[0]) {
        return "clip.use_silu is set";
    }
    if (fnfl_gguf_read_bools(path, "clip.vision.is_deepstack_layers", b)) {
        const bool any = std::find(b.begin(), b.end(), true) != b.end();
        if (any != p.deepstack) {
            return "clip.vision.is_deepstack_layers differs from the profile";
        }
    }
    size_t expected = 0;
    for (int32_t i = 0; i < vp_count; ++i) {
        const vtensor_rule & r = p.t[i];
        const int32_t        n = r.per_block ? p.n_block : 1;
        expected += size_t(n);
        for (int32_t k = 0; k < n; ++k) {
            const std::string name = r.per_block ? block_name(r, k) : std::string(r.name);
            const auto        it   = std::find_if(dir.tensors.begin(), dir.tensors.end(),
                                                  [&](const fnfl_gguf_tensor & t) { return t.name == name; });
            if (it == dir.tensors.end()) {
                return "no tensor " + name;
            }
            if (it->type_name != fmt_type(r.f)) {
                return name + " is " + it->type_name + ", not " + fmt_type(r.f);
            }
            for (int32_t d = 0; d < 4; ++d) {
                if (it->ne[d] != r.ne[d]) {
                    return name + " has another shape";
                }
            }
        }
    }
    if (dir.tensors.size() != expected) {
        return "the file has " + std::to_string(dir.tensors.size()) + " tensors, the profile " +
               std::to_string(expected);
    }
    return "";
}

struct file_reader {
    int fd = -1;

    ~file_reader() {
        if (fd >= 0) {
            close(fd);
        }
    }

    bool read(uint64_t off, void * dst, size_t n) const {
        auto * p = static_cast<char *>(dst);
        while (n > 0) {
            const ssize_t k = pread(fd, p, n, static_cast<off_t>(off));
            if (k <= 0) {
                return false;
            }
            p += k;
            off += uint64_t(k);
            n -= size_t(k);
        }
        return true;
    }
};

}  // namespace

bool load_host(const std::string & path, host_weights & w, std::string & err) {
    w = host_weights();
    fnfl_gguf_dir dir;
    if (!fnfl_gguf_read_dir(path, dir, err)) {
        return false;
    }
    std::string why;
    for (int32_t i = 0; i < n_vision_profiles && w.profile < 0; ++i) {
        const std::string m = mismatch(path, dir, vision_profiles[i]);
        if (m.empty()) {
            w.profile = i;
        } else if (why.empty()) {
            why = m;
        }
    }
    if (w.profile < 0) {
        err = "'" + path + "' matches no compiled vision profile (" + vision_profiles[0].name + ": " + why + ")";
        return false;
    }
    const vision_profile & p = vision_profiles[w.profile];
    file_reader            f;
    f.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f.fd < 0) {
        err = "cannot open '" + path + "'";
        return false;
    }
    const auto find = [&](const std::string & name) -> const fnfl_gguf_tensor & {
        return *std::find_if(dir.tensors.begin(), dir.tensors.end(),
                             [&](const fnfl_gguf_tensor & t) { return t.name == name; });
    };
    bool       ok    = true;
    const auto get32 = [&](int32_t part, int32_t b, std::vector<float> & out) {
        const fnfl_gguf_tensor & t = find(p.t[part].per_block ? block_name(p.t[part], b) : p.t[part].name);
        out.resize(t.bytes / sizeof(float));
        ok = ok && t.bytes % sizeof(float) == 0 && f.read(t.offset, out.data(), t.bytes);
    };
    const auto get16 = [&](int32_t part, int32_t b, std::vector<uint16_t> & out) {
        const fnfl_gguf_tensor & t = find(p.t[part].per_block ? block_name(p.t[part], b) : p.t[part].name);
        out.resize(t.bytes / sizeof(uint16_t));
        ok = ok && t.bytes % sizeof(uint16_t) == 0 && f.read(t.offset, out.data(), t.bytes);
    };
    std::vector<float> p0, p1;
    get32(vp_patch0, 0, p0);
    get32(vp_patch1, 0, p1);
    get32(vp_patch_b, 0, w.patch_b);
    get32(vp_pos, 0, w.pos);
    if (ok) {
        // ne (16, 16, 3, 1152): element (kx, ky, c, o) at ((o 3 + c) 16 + ky) 16 + kx, so [o][768] already
        w.patch.resize(p0.size());
        for (size_t i = 0; i < p0.size(); ++i) {
            w.patch[i] = static_cast<float>(double(p0[i]) + double(p1[i]));
        }
    }
    w.blk.resize(size_t(p.n_block));
    for (int32_t b = 0; b < p.n_block && ok; ++b) {
        block_host & k = w.blk[size_t(b)];
        get32(vp_ln1_w, b, k.ln1_w);
        get32(vp_ln1_b, b, k.ln1_b);
        get16(vp_qkv_w, b, k.qkv);
        get32(vp_qkv_b, b, k.qkv_b);
        get16(vp_o_w, b, k.o);
        get32(vp_o_b, b, k.o_b);
        get32(vp_ln2_w, b, k.ln2_w);
        get32(vp_ln2_b, b, k.ln2_b);
        get16(vp_up_w, b, k.up);
        get32(vp_up_b, b, k.up_b);
        get16(vp_down_w, b, k.down);
        get32(vp_down_b, b, k.down_b);
    }
    get32(vp_post_w, 0, w.post_w);
    get32(vp_post_b, 0, w.post_b);
    get16(vp_mm0_w, 0, w.mm0);
    get32(vp_mm0_b, 0, w.mm0_b);
    get16(vp_mm2_w, 0, w.mm2);
    get32(vp_mm2_b, 0, w.mm2_b);
    if (!ok) {
        err = "cannot read the tensors of '" + path + "'";
        w   = host_weights();
        return false;
    }
    w.path = path;
    return true;
}

namespace {

// the LayerNorm output's bound per channel: |gamma| sqrt(n - 1) + |beta|
std::vector<double> ln_bound(const std::vector<float> & g, const std::vector<float> & b) {
    const double        z = std::sqrt(double(g.size()) - 1.0);
    std::vector<double> y(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        y[i] = std::fabs(double(g[i])) * z + std::fabs(double(b[i]));
    }
    return y;
}

// sum_k |W[i][k]| y[k] + |b[i]| for rows [r0, r1), their maximum
double product_bound(const std::vector<uint16_t> & W,
                     const std::vector<float> &    b,
                     const std::vector<double> &   y,
                     int32_t                       r0,
                     int32_t                       r1) {
    const size_t k = y.size();
    double       m = 0;
    for (int32_t i = r0; i < r1; ++i) {
        double s = std::fabs(double(b[size_t(i)]));
        for (size_t j = 0; j < k; ++j) {
            s += std::fabs(double(bf16_f32(W[size_t(i) * k + j]))) * y[j];
        }
        m = std::max(m, s);
    }
    return m;
}

// the tanh GELU's magnitude bound for |u| <= m: GELU is between -0.17 and u for u >= 0, and between -0.17 and 0 below
double gelu_bound(double m) {
    return std::max(m, 0.17);
}

}  // namespace

bool range_check(const host_weights & w, range_report & r) {
    r                    = range_report();
    constexpr double lim = 65504.0;
    for (int32_t b = 0; b < int32_t(w.blk.size()); ++b) {
        const block_host & k  = w.blk[size_t(b)];
        const auto         y1 = ln_bound(k.ln1_w, k.ln1_b);
        const auto         y2 = ln_bound(k.ln2_w, k.ln2_b);
        const double ln = std::max(*std::max_element(y1.begin(), y1.end()), *std::max_element(y2.begin(), y2.end()));
        const double qk = std::sqrt(2.0) * product_bound(k.qkv, k.qkv_b, y1, 0, 2 * width);
        const double v  = product_bound(k.qkv, k.qkv_b, y1, 2 * width, qkv_width);
        const double g  = gelu_bound(product_bound(k.up, k.up_b, y2, 0, ffn));
        if (ln > r.ln) {
            r.ln     = ln;
            r.ln_blk = b;
        }
        if (std::max(qk, v) > r.qkv) {
            r.qkv     = std::max(qk, v);
            r.qkv_blk = b;
        }
        if (g > r.gelu) {
            r.gelu     = g;
            r.gelu_blk = b;
        }
    }
    const auto          yp = ln_bound(w.post_w, w.post_b);
    std::vector<double> y4;
    for (int32_t i = 0; i < 4; ++i) {
        y4.insert(y4.end(), yp.begin(), yp.end());
    }
    r.ln          = std::max(r.ln, *std::max_element(yp.begin(), yp.end()));
    r.merger_gelu = gelu_bound(product_bound(w.mm0, w.mm0_b, y4, 0, merged));
    r.ok          = r.ln < lim && r.qkv < lim && r.gelu < lim && r.merger_gelu < lim;
    return r.ok;
}

}  // namespace fn::vision
