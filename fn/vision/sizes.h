#pragma once

// VIS1 Stage B: the vision module's memory, as arithmetic (host only, no CUDA), so the loader can declare it as a
// state class before anything is placed: the encoder's scratch (encoder.h) and a card's share of the weights
// (weights.h device_weights::upload, whose layout is this plan: upload refuses a mismatch).

#include <cstddef>
#include <cstdint>

namespace fn::vision {

constexpr int32_t enc_chunk    = 2048;  // rows a product chunk
constexpr int32_t vision_split = 14;    // the two-card placement: blocks 0 .. 13 on card 0, 14 .. 26 on card 1

struct scratch_plan {
    int32_t max_p = 0, max_pp = 0, chunk = 0;
    size_t  x = 0, h = 0, qkv = 0, q_bytes = 0, f32 = 0, f16 = 0, total = 0;  // offsets and sizes
};

scratch_plan plan_scratch(int32_t max_patches, int32_t chunk = enc_chunk);

// the bytes of one card's share: blocks [b0, b1), the patch embedding and position table when embed, post_ln and the
// merger when merger (every piece 256-byte aligned, as upload lays them out)
size_t weight_bytes(int32_t b0, int32_t b1, bool embed, bool merger);

// the largest matrix's BF16 bytes in a share (upload's staging, which the encoder's scratch holds at load)
size_t weight_stage_bytes(int32_t b0, int32_t b1, bool merger);

// the patch rows' host staging of one image (encoder::embed): max_patches x 768 F32
size_t patch_stage_bytes(int32_t max_patches);

}  // namespace fn::vision
