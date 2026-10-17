#pragma once

// FNFL's CPU decoders of the deployed weight formats (formats.h), written from the GGUF format's definition (ggml at
// 66e11c3; decode.cpp names each decoder's source). The reference every FNFL test and mock decodes weights with.
// Plain F32 arithmetic in the definition's order, so the output is bitwise the definition's on any IEEE host without
// contracted multiply-adds. No GPU code.

#include "fn/quant/formats.h"

#include <cstdint>

namespace fn::quant {

// one block (info(f).block_bytes bytes, no alignment required) to info(f).block_values F32 values
void decode_block(wfmt f, const uint8_t * block, float * out);

// n values (a positive multiple of the block's values) from consecutive blocks; false otherwise
bool decode_row(wfmt f, const void * src, float * dst, int64_t n);

} // namespace fn::quant
