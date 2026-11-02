#pragma once

// VIS1 Stage B: image_rows' CPU test (no CUDA, no model). Its reference walks the prompt token by token as the
// reference engine assigns positions (mtmd's M-RoPE helper: a text token at the running position p, then p + 1; an image
// of nx x ny at p0 = p, then p = p0 + max(nx, ny)), independent of image_rows' offset formula. Returns 0 for PASS.

#include <cstdint>

namespace fn::runtime {

int image_rows_test(uint32_t seed);

}  // namespace fn::runtime
