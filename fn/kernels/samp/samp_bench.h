#pragma once

// fnfl-drive samp-bench [DEV0] [DEV1] [OUT] (SM1 step 6): each sampler path alone on both cards at B = 1 (generate)
// and B = 3 (verify, d = 2), in CUDA graphs of 16 decisions a card; median and p90 of 20 replays; OUT: JSON lines.
int fnfl_samp_bench(int dev0, int dev1, const char * out_path);
