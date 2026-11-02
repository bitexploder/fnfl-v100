#include "fn/vision/encoder.h"

#include "fn/kernels/dense/dense_prompt.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace fn::vision {

namespace {

int32_t round64(int32_t x) {
    return (x + 63) / 64 * 64;
}

}  // namespace

encoder::~encoder() {
    release();
}

void encoder::release() {
    if (device_ < 0) {
        return;
    }
    int prev = 0;
    cudaGetDevice(&prev);
    cudaSetDevice(device_);
    if (stream_ != nullptr) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    }
    if (owned_ && arena_ != nullptr) {
        cudaFree(arena_);
    }
    if (owned_ && pinned_ != nullptr) {
        cudaFreeHost(pinned_);
    }
    owned_ = false;
    cudaSetDevice(prev);
    stream_ = arena_ = nullptr;
    pinned_          = nullptr;
    device_          = -1;
}

bool encoder::init(int32_t       device,
                   int32_t       max_patches,
                   std::string & err,
                   int32_t       chunk,
                   void *        scratch,
                   float *       staging) {
    release();
    if (max_patches < 4 || max_patches > fn::vision::max_patches) {
        err = "vision encoder: max_patches out of range";
        return false;
    }
    if (chunk < 64 || chunk % 64 != 0) {
        err = "vision encoder: the chunk is not a positive multiple of 64";
        return false;
    }
    plan_    = plan_scratch(max_patches, chunk);
    int prev = 0;
    cudaGetDevice(&prev);
    if (cudaSetDevice(device) != cudaSuccess) {
        err = "vision encoder: cannot select device " + std::to_string(device);
        return false;
    }
    device_        = device;
    cudaStream_t s = nullptr;
    owned_         = scratch == nullptr;
    if (!owned_) {
        arena_  = scratch;
        pinned_ = staging;
    }
    const bool ok =
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess &&
        (!owned_ ||
         (cudaMalloc(&arena_, plan_.total) == cudaSuccess &&
          cudaMallocHost(reinterpret_cast<void **>(&pinned_), patch_stage_bytes(max_patches)) == cudaSuccess)) &&
        cudaMemsetAsync(arena_, 0, plan_.total, s) == cudaSuccess && cudaStreamSynchronize(s) == cudaSuccess;
    stream_ = s;
    cudaSetDevice(prev);
    if (!ok) {
        err = std::string("vision encoder: ") + cudaGetErrorString(cudaGetLastError());
        release();
        return false;
    }
    auto * base = static_cast<uint8_t *>(arena_);
    x_          = reinterpret_cast<float *>(base + plan_.x);
    h_          = reinterpret_cast<uint16_t *>(base + plan_.h);
    q_          = reinterpret_cast<uint16_t *>(base + plan_.qkv);
    f32_        = reinterpret_cast<float *>(base + plan_.f32);
    f16_        = reinterpret_cast<uint16_t *>(base + plan_.f16);
    return true;
}

bool encoder::fail(const char * what) {
    err_ = what;
    std::fprintf(stderr, "fnfl vision: encoder: %s\n", what);
    return false;
}

bool encoder::product(const uint16_t * a,
                      int32_t          rows,
                      int32_t          k,
                      const uint16_t * w,
                      const float *    cs,
                      int32_t          n,
                      float *          out,
                      int64_t          ld_out) {
    fn::dense::pt_args t;
    t.w          = w;
    t.n          = n;
    t.k          = k;
    t.rows       = rows;
    t.x.a0       = a;
    t.x.ld0      = k;
    t.x.k_split  = k;
    t.seg[0].out = out;
    t.seg[0].ld  = ld_out;
    t.seg[0].end = n;
    t.n_seg      = 1;
    t.col_scale  = cs;
    return fn::dense::pd_tc(t, stream_);
}

bool encoder::shape(int32_t gw, int32_t gh) {
    const int32_t P = gw * gh;
    if (gw < 2 || gh < 2 || gw % 2 || gh % 2 || P > plan_.max_p) {
        return fail("the image does not fit the scratch");
    }
    p_  = P;
    pp_ = round64(P);
    gw_ = gw;
    gh_ = gh;
    k_  = q_ + size_t(pp_) * heads * head_dim;
    vt_ = k_ + size_t(pp_) * heads * head_dim;
    return true;
}

bool encoder::begin_from_residual(const float * x, int32_t gw, int32_t gh) {
    if (x == nullptr || cudaSetDevice(device_) != cudaSuccess || !shape(gw, gh)) {
        return fail("begin_from_residual");
    }
    if (cudaMemcpyAsync(x_, x, size_t(p_) * width * 4, cudaMemcpyDefault, static_cast<cudaStream_t>(stream_)) !=
        cudaSuccess) {
        return fail("begin_from_residual: copy");
    }
    return true;
}

bool encoder::embed(const device_weights & w, const prepared & img) {
    if (!w.has_embed || w.device != device_ || pinned_ == nullptr) {
        return fail("embed: the weights have no patch embedding on this card, or no staging");
    }
    const int32_t P = img.gw * img.gh;
    if (int64_t(img.rows.size()) != int64_t(P) * patch_vals || cudaSetDevice(device_) != cudaSuccess ||
        !shape(img.gw, img.gh)) {
        return fail("embed: the image does not fit the scratch");
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    // the previous image's upload may still be reading the staging buffer
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        return fail("embed: stream");
    }
    std::memcpy(pinned_, img.rows.data(), img.rows.size() * sizeof(float));
    auto * rows = reinterpret_cast<float *>(q_);  // before block 0 the qkv region is free
    if (cudaMemcpyAsync(rows, pinned_, img.rows.size() * sizeof(float), cudaMemcpyHostToDevice, s) != cudaSuccess) {
        return fail("embed: upload");
    }
    if (!k_pos_resize(w.pos, gw_, gh_, x_, s) || !k_patch_embed(rows, P, w.patch, w.patch_b, x_, s)) {
        return fail("embed: kernels");
    }
    return true;
}

bool encoder::run_blocks(const device_weights & w, int32_t first, int32_t last) {
    if (w.device != device_ || first < w.b0 || last > w.b1 || first > last || p_ < 4) {
        return fail("run_blocks: the range is not on this card, or no image");
    }
    if (cudaSetDevice(device_) != cudaSuccess) {
        return fail("run_blocks: cudaSetDevice");
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int32_t      P = p_, Pp = pp_, R = plan_.chunk;
    const float        eps = float(vision_profiles[0].eps);
    for (int32_t b = first; b < last; ++b) {
        const dev_block & d = w.blk[size_t(b - w.b0)];
        if (!k_layernorm(x_, P, d.ln1_w, d.ln1_b, eps, h_, s)) {
            return fail("block: ln1");
        }
        for (int32_t r0 = 0; r0 < Pp; r0 += R) {
            const int32_t rows = std::min(R, Pp - r0), real = std::min(rows, P - r0);
            if (real > 0 &&
                !product(h_ + size_t(r0) * width, real, width, d.qkv, d.qkv_s, qkv_width, f32_, qkv_width)) {
                return fail("block: qkv product");
            }
            if (!k_qkv_rope(f32_, r0, rows, P, Pp, gw_, d.qkv_b, q_, k_, vt_, s)) {
                return fail("block: qkv_rope");
            }
        }
        if (!k_attention(q_, k_, vt_, P, Pp, h_, s)) {
            return fail("block: attention");
        }
        for (int32_t r0 = 0; r0 < P; r0 += R) {
            const int32_t real = std::min(R, P - r0);
            if (!product(h_ + size_t(r0) * width, real, width, d.o, d.o_s, width, f32_, width) ||
                !k_bias_residual(f32_, real, width, d.o_b, x_ + size_t(r0) * width, s)) {
                return fail("block: attention output");
            }
        }
        if (!k_layernorm(x_, P, d.ln2_w, d.ln2_b, eps, h_, s)) {
            return fail("block: ln2");
        }
        for (int32_t r0 = 0; r0 < P; r0 += R) {
            const int32_t real = std::min(R, P - r0);
            if (!product(h_ + size_t(r0) * width, real, width, d.up, d.up_s, ffn_pad, f32_, ffn_pad) ||
                !k_bias_gelu(f32_, real, ffn_pad, d.up_b, f16_, s) ||
                !product(f16_, real, ffn_pad, d.down, d.down_s, width, f32_, width) ||
                !k_bias_residual(f32_, real, width, d.down_b, x_ + size_t(r0) * width, s)) {
                return fail("block: feed-forward");
            }
        }
    }
    return true;
}

bool encoder::merge(const device_weights & w, float * out) {
    if (!w.has_merger || w.device != device_ || p_ < 4 || out == nullptr) {
        return fail("merge: the weights have no merger on this card, or no image");
    }
    if (cudaSetDevice(device_) != cudaSuccess) {
        return fail("merge: cudaSetDevice");
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream_);
    const int32_t      T = p_ / 4, R = plan_.chunk;
    if (!k_layernorm(x_, p_, w.post_w, w.post_b, float(vision_profiles[0].eps), h_, s)) {
        return fail("merge: post_ln");
    }
    // four consecutive patch rows are one token's 4608 values
    for (int32_t t0 = 0; t0 < T; t0 += R) {
        const int32_t real = std::min(R, T - t0);
        if (!product(h_ + size_t(t0) * merged, real, merged, w.mm0, w.mm0_s, merged, f32_, merged) ||
            !k_bias_gelu(f32_, real, merged, w.mm0_b, f16_, s) ||
            !product(f16_, real, merged, w.mm2, w.mm2_s, out_width, out + size_t(t0) * out_width, out_width) ||
            !k_bias_add(out + size_t(t0) * out_width, real, out_width, w.mm2_b, s)) {
            return fail("merge: merger");
        }
    }
    return true;
}

bool encoder::encode(const device_weights & w, const prepared & img, float * out) {
    return embed(w, img) && run_blocks(w, w.b0, w.b1) && merge(w, out);
}

bool encoder::sync() {
    if (cudaSetDevice(device_) != cudaSuccess ||
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)) != cudaSuccess) {
        return fail(cudaGetErrorString(cudaGetLastError()));
    }
    return true;
}

}  // namespace fn::vision
