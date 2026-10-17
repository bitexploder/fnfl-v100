#pragma once

// The IQ grid tables of the routed experts' formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) and the IQ4_NL values,
// FNFL's own constants: the GGUF formats' definition, copied once from ggml/src/ggml-common.h at 66e11c3 (grids.cpp
// names each table's lines). grid_table() (experts.h) packs them into the kernels' grid buffer at load; grid-test
// compares each with that definition byte for byte.

#include <cstddef>
#include <cstdint>

namespace fn::experts {

extern const uint64_t iq2xxs_grid[256];
extern const uint64_t iq2xs_grid[512];
extern const uint64_t iq2s_grid[1024];
extern const uint32_t iq3xxs_grid[256];
extern const uint32_t iq3s_grid[512];
extern const int8_t kvalues_iq4nl[16];

// grid_table()'s sources in its order, and their bytes
extern const void * const grid_src[6];
extern const size_t grid_src_bytes[6];

} // namespace fn::experts
