#include "fn/kernels/dense/dense_prompt.h"
#include "fn/vision/weights.h"

#include <cuda_runtime.h>

#include <algorithm>

namespace fn::vision {

namespace {

constexpr size_t al = 256;

size_t up_to(size_t x) {
    return (x + al - 1) / al * al;
}

// the BF16 matrix [n][k] padded to [n_pad][k_pad] with zeros
std::vector<uint16_t> pad16(const std::vector<uint16_t> & w, int32_t n, int32_t k, int32_t n_pad, int32_t k_pad) {
    std::vector<uint16_t> o(size_t(n_pad) * k_pad, 0);
    for (int32_t i = 0; i < n; ++i) {
        std::copy(w.begin() + int64_t(i) * k, w.begin() + int64_t(i + 1) * k, o.begin() + int64_t(i) * k_pad);
    }
    return o;
}

}  // namespace

device_weights::~device_weights() {
    release();
}

void device_weights::release() {
    if (owned_ && arena_ != nullptr) {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(device);
        cudaFree(arena_);
        cudaSetDevice(prev);
    }
    arena_ = nullptr;
    owned_ = false;
    blk.clear();
    patch = patch_b = pos = post_w = post_b = mm0_s = mm2_s = mm0_b = mm2_b = nullptr;
    mm0 = mm2 = nullptr;
    has_embed = has_merger = false;
    bytes                  = 0;
}

bool device_weights::upload(const host_weights & w,
                            int32_t              dev,
                            int32_t              first,
                            int32_t              last,
                            bool                 embed,
                            bool                 mrg,
                            std::string &        err,
                            void *               arena,
                            void *               ext_stage) {
    release();
    if (w.profile < 0 || first < 0 || last > int32_t(w.blk.size()) || first > last) {
        err = "vision upload: no weights or a bad block range";
        return false;
    }
    // the plan: every piece's byte offset in one allocation, 256-byte aligned
    size_t     total = 0;
    const auto take  = [&](size_t b) {
        const size_t o = total;
        total          = up_to(total + b);
        return o;
    };

    struct mat {
        size_t                w16, scale;
        int32_t               n, k;
        std::vector<uint16_t> src;  // BF16 [n][k], padded
    };

    struct f32 {
        size_t                     off;
        const std::vector<float> * v;
        size_t                     n;  // elements to reserve (>= v->size(); the tail zero)
    };

    std::vector<mat> mats;
    std::vector<f32> vecs;
    const auto       add_mat = [&](std::vector<uint16_t> src, int32_t n, int32_t k) {
        mat m{ take(size_t(n) * k * 2), take(size_t(n) * 4), n, k, std::move(src) };
        mats.push_back(std::move(m));
        return mats.size() - 1;
    };
    const auto add_vec = [&](const std::vector<float> & v, size_t n) {
        vecs.push_back({ take(n * 4), &v, n });
        return vecs.size() - 1;
    };

    struct blk_idx {
        size_t ln1_w, ln1_b, ln2_w, ln2_b, qkv_b, o_b, up_b, down_b, qkv, o, up, down;
    };

    std::vector<blk_idx> bi;
    for (int32_t b = first; b < last; ++b) {
        const block_host & h = w.blk[size_t(b)];
        blk_idx            x;
        x.ln1_w  = add_vec(h.ln1_w, width);
        x.ln1_b  = add_vec(h.ln1_b, width);
        x.ln2_w  = add_vec(h.ln2_w, width);
        x.ln2_b  = add_vec(h.ln2_b, width);
        x.qkv_b  = add_vec(h.qkv_b, qkv_width);
        x.o_b    = add_vec(h.o_b, width);
        x.up_b   = add_vec(h.up_b, ffn_pad);
        x.down_b = add_vec(h.down_b, width);
        x.qkv    = add_mat(h.qkv, qkv_width, width);
        x.o      = add_mat(h.o, width, width);
        x.up     = add_mat(pad16(h.up, ffn, width, ffn_pad, width), ffn_pad, width);
        x.down   = add_mat(pad16(h.down, width, ffn, width, ffn_pad), width, ffn_pad);
        bi.push_back(x);
    }
    size_t patch_i = 0, patch_b_i = 0, pos_i = 0, post_w_i = 0, post_b_i = 0, mm0_b_i = 0, mm2_b_i = 0, mm0_i = 0,
           mm2_i = 0;
    if (embed) {
        patch_i   = add_vec(w.patch, w.patch.size());
        patch_b_i = add_vec(w.patch_b, width);
        pos_i     = add_vec(w.pos, w.pos.size());
    }
    if (mrg) {
        post_w_i = add_vec(w.post_w, width);
        post_b_i = add_vec(w.post_b, width);
        mm0_b_i  = add_vec(w.mm0_b, merged);
        mm2_b_i  = add_vec(w.mm2_b, out_width);
        mm0_i    = add_mat(w.mm0, merged, merged);
        mm2_i    = add_mat(w.mm2, out_width, merged);
    }

    size_t stage_bytes = 0;
    for (const mat & m : mats) {
        stage_bytes = std::max(stage_bytes, m.src.size() * 2);
    }
    if (total != weight_bytes(first, last, embed, mrg) || stage_bytes > weight_stage_bytes(first, last, mrg)) {
        err = "vision upload: the layout differs from sizes.h's plan";
        return false;
    }
    if ((arena == nullptr) != (ext_stage == nullptr)) {
        err = "vision upload: an arena needs its staging";
        return false;
    }
    int prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(dev) != cudaSuccess) {
        err = "vision upload: cannot select device " + std::to_string(dev);
        return false;
    }
    owned_             = arena == nullptr;
    void *       stage = ext_stage;
    cudaStream_t s     = nullptr;
    bool         ok    = true;
    if (owned_) {
        ok = cudaMalloc(&arena_, total) == cudaSuccess && cudaMalloc(&stage, stage_bytes) == cudaSuccess;
    } else {
        arena_ = arena;
    }
    ok          = ok && cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess;
    auto * base = static_cast<uint8_t *>(arena_);
    if (ok) {
        ok = cudaMemsetAsync(arena_, 0, total, s) == cudaSuccess;
    }
    for (const f32 & v : vecs) {
        ok =
            ok && cudaMemcpyAsync(base + v.off, v.v->data(), v.v->size() * 4, cudaMemcpyHostToDevice, s) == cudaSuccess;
    }
    for (const mat & m : mats) {
        // one staging buffer: each conversion finishes before the next copy overwrites it
        ok = ok && cudaMemcpyAsync(stage, m.src.data(), m.src.size() * 2, cudaMemcpyHostToDevice, s) == cudaSuccess &&
             fn::dense::pd_convert_bf16(static_cast<const uint16_t *>(stage), m.n, m.k,
                                        reinterpret_cast<uint16_t *>(base + m.w16),
                                        reinterpret_cast<float *>(base + m.scale), s) &&
             cudaStreamSynchronize(s) == cudaSuccess;
    }
    ok = ok && cudaStreamSynchronize(s) == cudaSuccess;
    if (s != nullptr) {
        cudaStreamDestroy(s);
    }
    if (owned_ && stage != nullptr) {
        cudaFree(stage);
    }
    cudaSetDevice(prev);
    if (!ok) {
        err    = std::string("vision upload: ") + cudaGetErrorString(cudaGetLastError());
        device = dev;
        release();
        return false;
    }
    const auto F = [&](size_t i) {
        return reinterpret_cast<const float *>(base + vecs[i].off);
    };
    const auto H = [&](size_t i) {
        return reinterpret_cast<const uint16_t *>(base + mats[i].w16);
    };
    const auto S = [&](size_t i) {
        return reinterpret_cast<const float *>(base + mats[i].scale);
    };
    device = dev;
    b0     = first;
    b1     = last;
    for (const blk_idx & x : bi) {
        dev_block d;
        d.ln1_w  = F(x.ln1_w);
        d.ln1_b  = F(x.ln1_b);
        d.ln2_w  = F(x.ln2_w);
        d.ln2_b  = F(x.ln2_b);
        d.qkv_b  = F(x.qkv_b);
        d.o_b    = F(x.o_b);
        d.up_b   = F(x.up_b);
        d.down_b = F(x.down_b);
        d.qkv    = H(x.qkv);
        d.qkv_s  = S(x.qkv);
        d.o      = H(x.o);
        d.o_s    = S(x.o);
        d.up     = H(x.up);
        d.up_s   = S(x.up);
        d.down   = H(x.down);
        d.down_s = S(x.down);
        blk.push_back(d);
    }
    if (embed) {
        has_embed = true;
        patch     = F(patch_i);
        patch_b   = F(patch_b_i);
        pos       = F(pos_i);
    }
    if (mrg) {
        has_merger = true;
        post_w     = F(post_w_i);
        post_b     = F(post_b_i);
        mm0_b      = F(mm0_b_i);
        mm2_b      = F(mm2_b_i);
        mm0        = H(mm0_i);
        mm0_s      = S(mm0_i);
        mm2        = H(mm2_i);
        mm2_s      = S(mm2_i);
    }
    bytes = total;
    return true;
}

}  // namespace fn::vision
