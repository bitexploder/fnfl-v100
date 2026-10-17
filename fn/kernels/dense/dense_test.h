#pragma once

// The dense products against an FP64 CPU reference (DP1 step 3, DP2 step 2), and the synthetic weights the test and
// the bench share: n_synth_rows distinct rows per (format, K) of valid random blocks from fn/quant's seeded generator
// (C1b), so no quantizer runs and nothing is cached; the reference decodes them with fn/quant's decoders.

#include "fn/kernels/dense/dense.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fn::dense {

constexpr int32_t n_synth_rows = 1024;

// n_synth_rows GGUF rows of k weights in format f
struct synth {
    fmt f = fmt::q4_k;
    int32_t k = 0;
    size_t row_bytes = 0;
    std::vector<uint8_t> bytes;
    const uint8_t * row(int32_t i) const { return bytes.data() + size_t(i) * row_bytes; }
};

// The synthetic rows of (f, k), seeded from f and k (fn/quant/synth.h). False for a format or k the kernel does not
// take.
bool make_synth(fmt f, int32_t k, synth & out);

// The packed bytes of an m-row matrix whose row i is synthetic row (i % period + off) % n_synth_rows; a period below m
// repeats rows exactly (the router's tie case)
bool pack_synth(const synth & s, int32_t m, int32_t off, int32_t period, std::vector<uint8_t> & out);

// The tables' buffer (grid_bytes) from FNFL's own tables (fn/kernels/experts/grids.h)
bool grid_buffer(std::vector<uint8_t> & out);

} // namespace fn::dense

// dp-test: every format at small and deployed shapes, every grouping with mixed formats, split and unsplit paths,
// the shared GLU's codes against act_ref bitwise, the router's IDs, weights and gate against FP64 (DP2 Amendment 1),
// the scaled shared down, the GDN input's convolution epilogue (LP1: post-SiLU values against FP64, history versions
// bitwise), all-zero activations, a second identical run compared bitwise, and each row at B = 1
// against the same row at every other B run (Amendment 2), at each row count in rows_mask (bit B-1); the draft
// block's shapes (DR1 Stage B: eh_proj Q8_0 at K 5,120 split and unsplit, the F32 router and shared gate with the
// epilogue's ties, NaN and zero cases, F32 tail rows); and the frame's
// work sites (LP1 Amendment 3: the QSA input's owner, peer and owner launches through fn/model's sites, bitwise equal
// to each alone, at B = 1 and 4, with the shared-region negative controls). mode: "full"
// (the vocabulary at 124,160 rows), "san" (8,192 rows, for Compute Sanitizer's time) or "all" (both). Prints
// "dp-test: PASS" and returns 0 when every check passes.
int fnfl_dp_test(int device, uint32_t rows_mask, const char * mode);
