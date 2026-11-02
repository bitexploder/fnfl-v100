#include "fn/vision/sizes.h"

#include "fn/vision/image.h"
#include "fn/vision/profile.h"

#include <algorithm>

namespace fn::vision {

namespace {

size_t up_to(size_t x) {
    return (x + 255) / 256 * 256;
}

int32_t round64(int32_t x) {
    return (x + 63) / 64 * 64;
}

}  // namespace

scratch_plan plan_scratch(int32_t max_patches, int32_t chunk) {
    scratch_plan p;
    p.max_p         = max_patches;
    p.max_pp        = round64(max_patches);
    p.chunk         = chunk;
    size_t     off  = 0;
    const auto take = [&](size_t b) {
        const size_t o = off;
        off            = up_to(off + b);
        return o;
    };
    p.x       = take(size_t(p.max_p) * width * 4);
    p.h       = take(size_t(p.max_pp) * width * 2);
    p.q_bytes = size_t(p.max_pp) * (2 * heads * head_dim + heads * head_pad) * 2;
    p.q_bytes = std::max(p.q_bytes, size_t(p.max_p) * patch_vals * 4);
    p.qkv     = take(p.q_bytes);
    p.f32     = take(size_t(chunk) * merged * 4);
    p.f16     = take(size_t(chunk) * merged * 2);
    p.total   = off;
    return p;
}

size_t weight_bytes(int32_t b0, int32_t b1, bool embed, bool merger) {
    // the order of device_weights::upload: each block's vectors then its matrices (FP16 words, then F32 column
    // scales), then the embedding's vectors, then the merger's vectors and matrices
    size_t     total = 0;
    const auto vec   = [&](size_t n) {
        total = up_to(total + n * 4);
    };
    const auto mat = [&](size_t n, size_t k) {
        total = up_to(total + n * k * 2);
        total = up_to(total + n * 4);
    };
    for (int32_t b = b0; b < b1; ++b) {
        vec(width);
        vec(width);
        vec(width);
        vec(width);
        vec(qkv_width);
        vec(width);
        vec(ffn_pad);
        vec(width);
        mat(qkv_width, width);
        mat(width, width);
        mat(ffn_pad, width);
        mat(width, ffn_pad);
    }
    if (embed) {
        vec(size_t(width) * patch_vals);
        vec(width);
        vec(size_t(pos_side) * pos_side * width);
    }
    if (merger) {
        vec(width);
        vec(width);
        vec(merged);
        vec(out_width);
        mat(merged, merged);
        mat(out_width, merged);
    }
    return total;
}

size_t weight_stage_bytes(int32_t b0, int32_t b1, bool merger) {
    size_t s = 0;
    if (b1 > b0) {
        s = std::max({ size_t(qkv_width) * width, size_t(width) * width, size_t(ffn_pad) * width }) * 2;
    }
    if (merger) {
        s = std::max({ s, size_t(merged) * merged * 2, size_t(out_width) * merged * 2 });
    }
    return s;
}

size_t patch_stage_bytes(int32_t max_patches) {
    return size_t(max_patches) * patch_vals * 4;
}

}  // namespace fn::vision
