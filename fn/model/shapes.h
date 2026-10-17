#pragma once

// Qwen3.8-Flash-Next (qwen4exp) shapes. Counted from the deployed IQ3_XXS GGUF's metadata (FL2); the glue checks the file
// against them at load.

#include <cstdint>

namespace fn {

constexpr const char * arch = "qwen4exp";

constexpr int32_t n_layer = 48;
constexpr int32_t n_embd = 2560; // D
constexpr int32_t n_ctx_train = 262144;

constexpr int32_t n_hc = 4; // hyper-connection streams
constexpr int32_t n_hc_lowrank = 320;

constexpr int32_t n_head = 24;
constexpr int32_t n_head_kv = 2;
constexpr int32_t n_embd_head = 256; // key and value length
constexpr int32_t full_attn_interval = 4;

constexpr int32_t n_expert = 512;
constexpr int32_t n_expert_used = 10;
constexpr int32_t n_ff_expert = 640;
constexpr int32_t n_ff_shared = 640;

constexpr int32_t ssm_d_conv = 4;
constexpr int32_t ssm_d_state = 128;
constexpr int32_t ssm_n_group = 16;
constexpr int32_t ssm_dt_rank = 48;
constexpr int32_t ssm_d_inner = 6144;

constexpr float rms_eps = 1e-6f;

static_assert(n_layer % full_attn_interval == 0, "full-attention layers must tile the layer sequence");
static_assert(n_head % n_head_kv == 0, "query heads must group over the KV heads");

} // namespace fn
