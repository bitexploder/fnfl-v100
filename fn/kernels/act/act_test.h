#pragma once

// act-test (C2): FNFL's quantize kernel (fn/kernels/act/act.h) against its CPU reference (fn/quant/act_ref.h),
// bitwise, on random rows at K = 320, 2,560, 3,072 and 6,144 and B = 1 to 4, plus all zeros, one large value and values
// exactly at rounding midpoints. Prints "act-test: PASS" and returns 0 when every case matches.
int fnfl_act_test(int device);

// act-bench (C2): the quantize kernel's time at each point, untraced, median of 20 graph replays; a replay (a round)
// is 512 launches, each on its own input and output (cold: the rotation spans more than L2) or all on one (hot).
// mode: "all" (B = 1 to 4 at K = 320, 2,560, 3,072, 6,144) or "acceptance" (B = 3 at K = 2,560 and 6,144).
int fnfl_act_bench(int device, const char * mode);
