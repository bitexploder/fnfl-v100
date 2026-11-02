#pragma once

// VIS1: the projector file's weights. load_host reads the file through the GGUF reader (fn/borrowed), resolves it
// against the compiled profiles (profile.h) and keeps every tensor as stored: BF16 matrices as their 16-bit words, F32
// as F32. The two patch kernels are folded into one (summed in F64, rounded once to F32): for one image both see the
// same frame. range_check is the load-time proof that every FP16 operand of the encoder stays finite for any image;
// upload places a block range (and optionally the patch embedding and the merger) on one card, the matrices converted
// once to FP16 for the prompt path's tensor-core product (pd_convert_bf16, with its per-row 2^-e column scale).

#include "fn/vision/profile.h"
#include "fn/vision/sizes.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace fn::vision {

struct block_host {
    std::vector<float>    ln1_w, ln1_b, qkv_b, o_b, ln2_w, ln2_b, up_b, down_b;
    std::vector<uint16_t> qkv, o, up, down;  // BF16 [n][k]: qkv [3456][1152], o [1152][1152], up [4304][1152],
                                             // down [1152][4304]
};

struct host_weights {
    std::string             path;
    int32_t                 profile = -1;
    std::vector<float>      patch;    // [1152][768] (c, ky, kx), the two kernels' sum
    std::vector<float>      patch_b;  // [1152]
    std::vector<float>      pos;      // [2304][1152], row 48 y + x
    std::vector<block_host> blk;
    std::vector<float>      post_w, post_b, mm0_b, mm2_b;
    std::vector<uint16_t>   mm0, mm2;  // BF16 [4608][4608], [2560][4608]
};

bool load_host(const std::string & path, host_weights & w, std::string & err);

inline float bf16_f32(uint16_t h) {
    const uint32_t u = uint32_t(h) << 16;
    float          f;
    static_assert(sizeof(f) == sizeof(u), "F32");
    std::memcpy(&f, &u, sizeof(f));  // a BF16 word is the F32's high half
    return f;
}

// The largest magnitude each FP16 operand can reach for any input, from the weights: a LayerNorm output is at most
// |gamma| sqrt(n - 1) + |beta| (a standardized value of n samples is at most sqrt(n - 1)); a product's output at most
// sum |W| x that bound + |bias|; the rotary step at most sqrt(2) x its pair's bound; GELU at most its input's bound.
struct range_report {
    double  ln = 0, qkv = 0, gelu = 0, merger_gelu = 0;  // maxima over the blocks
    int32_t ln_blk = -1, qkv_blk = -1, gelu_blk = -1;
    bool    ok = false;                                  // every bound under 65,504
};

bool range_check(const host_weights & w, range_report & r);

// One card's share: blocks [b0, b1), the patch embedding and position table when embed, post_ln and the merger when
// merger. Owns one device allocation.
struct dev_block {
    const float *   ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;
    const uint16_t *qkv = nullptr, *o = nullptr, *up = nullptr, *down = nullptr;  // FP16 [n][k]; up [4320][1152],
                                                                                  // down [1152][4320] zero-padded
    const float *   qkv_s = nullptr, *o_s = nullptr, *up_s = nullptr, *down_s = nullptr;  // column scales 2^-e
    const float *   qkv_b = nullptr, *o_b = nullptr, *up_b = nullptr, *down_b = nullptr;  // up_b [4320]
};

class device_weights {
  public:
    device_weights() = default;
    ~device_weights();
    device_weights(const device_weights &)             = delete;
    device_weights & operator=(const device_weights &) = delete;

    // VIS1 Stage B: with arena (weight_bytes(b0, b1, embed, merger) device bytes on this card, 256-byte aligned) and
    // stage (weight_stage_bytes(b0, b1, merger) device bytes, free during the call: the encoder's scratch at load) the
    // share is placed there and nothing is allocated; without, the share owns one allocation (the tests')
    bool upload(const host_weights & w,
                int32_t              device,
                int32_t              b0,
                int32_t              b1,
                bool                 embed,
                bool                 merger,
                std::string &        err,
                void *               arena = nullptr,
                void *               stage = nullptr);
    void release();

    int32_t                device = -1, b0 = 0, b1 = 0;
    bool                   has_embed = false, has_merger = false;
    std::vector<dev_block> blk;  // blk[i] is block b0 + i
    const float *          patch = nullptr, *patch_b = nullptr, *pos = nullptr;
    const float *          post_w = nullptr, *post_b = nullptr;
    const uint16_t *       mm0 = nullptr, *mm2 = nullptr;
    const float *          mm0_s = nullptr, *mm2_s = nullptr, *mm0_b = nullptr, *mm2_b = nullptr;
    size_t                 bytes = 0;  // the allocation

  private:
    void * arena_ = nullptr;
    bool   owned_ = false;
};

}  // namespace fn::vision
