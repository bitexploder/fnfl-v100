#pragma once

// dp-bench (DP1 steps 6-8, DP2 steps 5-7). mode:
//   all       every grouping at deployed layers' formats alone, B = 1 and 3, each launch timed warm and cold (below)
//   seq       one graph per B of the launches in a layer's order with C2's quantize and the stand-ins between them
//             (DP1 step 7, DP2 step 7), and the quantize and stand-in launches alone
//   nsys      the seq graphs only (for nsys --cuda-graph-trace=node)
//   ncu:NAME  one point of "all" alone, warm only (for ncu); NAME as its JSON "point"
// Warm (DP1's method): a CUDA graph of 64 launches back to back, weights rotated over at least 200 MB, 3 warm-up
// replays and 20 timed. Cold (DP2 rule R17, XP1's alone method): a graph of 20 x [12 MB L2 flush, instruction-cache
// eviction, launch] against one of 20 x [flush, eviction], per trial (A - B) / 20, 3 warm-up trials and 20 timed. One
// JSON line per point on stdout, with the floor of its weight, activation and output bytes. Returns 0 on success.
int fnfl_dp_bench(int device, const char * mode);
