#pragma once

// The prompt expert kernels (EP1) against FP64 CPU references, and the host pieces ep-test and ep-bench share: FN4's
// routes as prompt rows and the expert sort. Synthetic weights: EX1's layout (experts_test.h, n_synth distinct
// experts; expert e uses synthetic expert e % n_synth), with each block's d from the format's deployed range on
// ep-test's own seed stream (load_synth's deployed, C1b Amendment 4), so the F16 GLU sees realistic magnitudes.

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_test.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::experts {

// FN4's routes.txt ("decode row layer e0 .. e9"): per layer, every row's 10 IDs in delivery order
struct ep_fn4 {
    std::vector<std::vector<int32_t>> by_layer;
    int32_t n_rows = 0; // rows every layer has
};
bool ep_read_fn4(const char * path, ep_fn4 & out);

// rows consecutive delivered rows of layer from row start: [rows][10] IDs
std::vector<int32_t> ep_window(const ep_fn4 & r, int32_t layer, int32_t start, int32_t rows);

// MUL_MAT_ID's expert-sorted pair order (stable by pair index p = 10 row + slot): bounds [513], sorted [pairs]
struct ep_sort {
    std::vector<int32_t> bounds, sorted;
};
ep_sort ep_sort_pairs(const std::vector<int32_t> & ids);

} // namespace fn::experts

// ep-prep: the FP64 references of every ep-test case, written to dir (CPU only; reads the synthetic weights from
// syn_dir), for formats "all" (the deployed ones) or "q4" (ME1: the Q4 candidate's gate/up Q4_K, Q5_K and down Q5_1,
// Q8_0, synthetic scales from synth.h's chosen ranges). Returns 0 on success.
int fnfl_ep_prep(const char * dir, const char * syn_dir, const char * routes, const char * formats);

// ep-test: rows (128, 512 or 1536, as listed), formats "all" (the deployed ones), "q4" (ME1's, as ep-prep) or a list of
// gate/up and down formats: every listed gate/up format against every listed down format. mode "full": both runs
// bitwise, the zero and bad-ID cases; "san": one run (Compute Sanitizer). Prints "ep-test: PASS" and returns 0 when
// every check passes.
int fnfl_ep_test(int device, const char * rows, const char * formats, const char * dir, const char * syn_dir,
                 const char * routes, const char * mode);
