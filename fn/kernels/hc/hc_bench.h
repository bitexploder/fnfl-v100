#pragma once

// The HC kernels timed alone and with their surrounding parts, in CUDA graphs over cold weights (HC1 steps 6 and 7).
// Prints one JSON line per point and returns 0 when every launch succeeded.
int fnfl_hc_bench(int device);

// HC2: the sublayer on both cards (dev0 card 0, dev1 card 1), both arms, B = 1 and 3: graphs of 64 sublayers on each
// card launched together from this thread, weights rotated over 422 MB a card; and each card's kernels with the peer's
// data pre-filled (no wait). mode "all"; "ncu": only card 0's pre-filled arm A hc_up at B = 3 (nothing waits, so the
// profiler's serialization cannot stall it); "nsys": only the arm A sublayer graph at B = 3. Prints one JSON line a
// point; returns 0 when every launch succeeded and no poll gave up.
int fnfl_hc2_bench(int dev0, int dev1, const char * mode);
