#pragma once

// The prompt expert kernels (EP1), timed in CUDA graphs of 16 layers over cold weights with FN4's decode routes as
// prompt rows. Prints one JSON line a point; returns 0 when every launch succeeded. mode "all": every point; "ncu": at
// 1,536 rows (IQ3_XXS/IQ4_NL).
int fnfl_ep_bench(int device, const char * mode, const char * cache_dir, const char * routes_path);
