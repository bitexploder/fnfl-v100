#pragma once

// The deployed model's weight formats (fnfl/sim/data/deployed-tensor-inventory.json) and their block geometry, as the
// GGUF format defines them (ggml/include/ggml.h and ggml/src/ggml-common.h at 66e11c3), plus the scalar F16 and BF16
// conversions FNFL's CPU references use. No GPU code.

#include <cstddef>
#include <cstdint>

namespace fn::quant {

enum class wfmt : int32_t {
    iq2_xxs,
    iq2_xs,
    iq2_s,
    iq3_xxs,
    iq3_s,
    iq4_nl,
    iq4_xs,
    q2_0,
    q4_k,
    q5_k,
    q6_k,
    q8_0,
    bf16,
    f16,
    f32,
};

constexpr int32_t n_wfmt = 15;

struct wfmt_info {
    const char * name;    // "iq2_xxs", ..., "f32"
    int32_t gguf_type;    // the GGUF tensor type number (ggml.h's enum ggml_type)
    int32_t block_values; // values per block
    int32_t block_bytes;  // bytes per block
};

const wfmt_info & info(wfmt f);

// the format of a GGUF tensor type number or a name; false for any other
bool wfmt_of_gguf_type(int32_t gguf_type, wfmt & out);
bool wfmt_of_name(const char * name, wfmt & out);

// bytes of n values (n a positive multiple of the block's values), 0 otherwise
size_t row_bytes(wfmt f, int64_t n);

// IEEE binary16 and bfloat16 storage. To F32 is exact (a NaN keeps its payload and is made quiet); from F32 rounds to
// nearest even, overflows to infinity (F16) and keeps a NaN a quiet NaN.
float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);
float bf16_to_f32(uint16_t h);
uint16_t f32_to_bf16(float f);

} // namespace fn::quant
