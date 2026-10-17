#pragma once

// The QSA kernels timed alone and with their surrounding parts, in CUDA graphs over cold summaries (QA1 steps 6 and 7). mode "all": every point; "acceptance": only score + select + union at
// 196,608 cells and B = 3 (the ncu run). fixture (optional): the HX7 case's file, for one more point at 196,608 cells
// and B = 3 with HX7's selection pattern. Prints one JSON line per point and returns 0 when every launch succeeded.
int fnfl_qsa_bench(int device, const char * mode, const char * fixture);
