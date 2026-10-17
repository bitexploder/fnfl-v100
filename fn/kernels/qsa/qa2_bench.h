#pragma once

#include <string>

// QA2 step 4: attend (two launches) timed alone in CUDA graphs. mode "all": 8,192, 65,536 and 196,608 cells at B = 1
// and 3 (rows select disjoint blocks), plus HX7's selection at 196,608 cells and B = 3; each point warm (K and V
// rotated over at least 200 MB, code warm) and cold (12 MB L2 flush and the instruction-cache evictor before each
// launch). mode "ncu": HX7's point for one Nsight Compute capture of the two launches. fixture: QA1's
// hx7-l3-w0.bin. Prints one JSON line a point and returns 0 when every launch succeeded.
int fnfl_qa2_bench(int device, const std::string & mode, const std::string & fixture);
