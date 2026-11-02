#pragma once

// VIS1: the vision encoder's shapes and the projector files' profiles (TTTS3's pattern, fn/kernels/experts/profiles.h).
// A profile is one projector file's declaration: its projector type, its hyperparameters and each tensor family's
// name, stored format and shape. The loader resolves a file against the compiled profiles (vision_profile_of) and
// refuses anything none of them holds, naming the first mismatch; nothing downstream branches on a format. One
// profile today: the Qwen3-VL merger in BF16, which serves both text files (VIS1's projector decision).

#include <cstdint>

namespace fn::vision {

// the encoder's fixed shapes (the vision file's metadata, VIS1's "What the files say")
constexpr int32_t width      = 1152;  // the vision transformer
constexpr int32_t heads      = 16;
constexpr int32_t head_dim   = 72;
constexpr int32_t ffn        = 4304;
constexpr int32_t ffn_pad    = 4320;          // pd_tc's K % 32: zero rows of up and zero columns of down (exact)
constexpr int32_t blocks     = 27;
constexpr int32_t merged     = 4 * width;     // 4608: a token's 2 x 2 patches
constexpr int32_t out_width  = 2560;          // the text model's width
constexpr int32_t qkv_width  = 3 * width;
constexpr int32_t pos_side   = 48;            // the learned 48 x 48 position table
constexpr int32_t rope_pairs = head_dim / 4;  // 18: pairs 0-17 on the row, 18-35 on the column
constexpr int32_t head_pad   = 80;            // V's dims padded to 5 n16 tiles in the attention kernel
static_assert(heads * head_dim == width && ffn_pad % 32 == 0 && ffn_pad >= ffn && merged % 32 == 0, "shapes");

enum class vfmt : int32_t { f32, bf16 };

// one tensor family: its name (a %d for the block index when per_block), format and shape (ne, fastest first)
struct vtensor_rule {
    const char * name;
    bool         per_block;
    vfmt         f;
    int64_t      ne[4];
};

// the families, in the order every profile lists them
enum vpart : int32_t {
    vp_patch0,
    vp_patch1,
    vp_patch_b,
    vp_pos,
    vp_ln1_w,
    vp_ln1_b,
    vp_qkv_w,
    vp_qkv_b,
    vp_o_w,
    vp_o_b,
    vp_ln2_w,
    vp_ln2_b,
    vp_up_w,
    vp_up_b,
    vp_down_w,
    vp_down_b,
    vp_post_w,
    vp_post_b,
    vp_mm0_w,
    vp_mm0_b,
    vp_mm2_w,
    vp_mm2_b,
    vp_count
};

struct vision_profile {
    const char * name;
    const char * projector_type;  // clip.projector_type
    int32_t      n_block, n_embd, n_ff, n_head, patch_size, merge_size, proj_dim;
    double       eps;             // clip.vision.attention.layer_norm_epsilon, as F32
    float        mean, std_;      // clip.vision.image_mean / image_std, all three channels
    bool         use_gelu;        // clip.use_gelu: the tanh GELU in the blocks and the merger
    bool         deepstack;       // any clip.vision.is_deepstack_layers entry true
    vtensor_rule t[vp_count];
};

inline constexpr vision_profile vision_profiles[] = {
    { "Qwen3-VL merger, BF16 matrices",
     "qwen3vl_merger", blocks,
     width, ffn,
     heads, 16,
     2, out_width,
     double(1e-6f),
     0.5f, 0.5f,
     true, false,
     {
          { "v.patch_embd.weight", false, vfmt::f32, { 16, 16, 3, width } },
          { "v.patch_embd.weight.1", false, vfmt::f32, { 16, 16, 3, width } },
          { "v.patch_embd.bias", false, vfmt::f32, { width, 1, 1, 1 } },
          { "v.position_embd.weight", false, vfmt::f32, { width, pos_side * pos_side, 1, 1 } },
          { "v.blk.%d.ln1.weight", true, vfmt::f32, { width, 1, 1, 1 } },
          { "v.blk.%d.ln1.bias", true, vfmt::f32, { width, 1, 1, 1 } },
          { "v.blk.%d.attn_qkv.weight", true, vfmt::bf16, { width, qkv_width, 1, 1 } },
          { "v.blk.%d.attn_qkv.bias", true, vfmt::f32, { qkv_width, 1, 1, 1 } },
          { "v.blk.%d.attn_out.weight", true, vfmt::bf16, { width, width, 1, 1 } },
          { "v.blk.%d.attn_out.bias", true, vfmt::f32, { width, 1, 1, 1 } },
          { "v.blk.%d.ln2.weight", true, vfmt::f32, { width, 1, 1, 1 } },
          { "v.blk.%d.ln2.bias", true, vfmt::f32, { width, 1, 1, 1 } },
          { "v.blk.%d.ffn_up.weight", true, vfmt::bf16, { width, ffn, 1, 1 } },
          { "v.blk.%d.ffn_up.bias", true, vfmt::f32, { ffn, 1, 1, 1 } },
          { "v.blk.%d.ffn_down.weight", true, vfmt::bf16, { ffn, width, 1, 1 } },
          { "v.blk.%d.ffn_down.bias", true, vfmt::f32, { width, 1, 1, 1 } },
          { "v.post_ln.weight", false, vfmt::f32, { width, 1, 1, 1 } },
          { "v.post_ln.bias", false, vfmt::f32, { width, 1, 1, 1 } },
          { "mm.0.weight", false, vfmt::bf16, { merged, merged, 1, 1 } },
          { "mm.0.bias", false, vfmt::f32, { merged, 1, 1, 1 } },
          { "mm.2.weight", false, vfmt::bf16, { merged, out_width, 1, 1 } },
          { "mm.2.bias", false, vfmt::f32, { out_width, 1, 1, 1 } },
      } },
};
inline constexpr int32_t n_vision_profiles = int32_t(sizeof(vision_profiles) / sizeof(vision_profiles[0]));

}  // namespace fn::vision
