#pragma once

// gd-bench (GD1): gdn_core's time at B = 1..4, untraced, alone. warm: a graph of launches back to back, each on its
// own state, history and inputs (rotated over more than 200 MB); cold (R17): every launch after a 12 MB L2 flush and
// an instruction-cache evictor, timed as the difference of graphs with and without the kernel. Median and p90 of 20
// replays after 3 warm-ups, SM clock sampled. mode: "all", "ncu" (one graph at B = 3 for an ncu capture) or "probe" (GD1
// Amendment 2b step 0: the write rate W at gdn_core's shape) or "diag" (step 4's check: warm, with the finish disabled).
int fnfl_gd_bench(int device, const char * mode);
