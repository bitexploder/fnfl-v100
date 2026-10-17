#pragma once

#include <string>

// QA2: attend (fn/kernels/qsa/attend.h) against an FP64 CPU reference written from the pinned lines, and the raw
// index-key write in summary_update. mode "all": every case; "san": 65,536 cells at B = 3 only (Compute Sanitizer).
// model: the GGUF's first shard (layer 11's attn_q_norm and attn_k_norm and the rope and attention keys, read on the
// CPU); fixture: QA1's HX7 selection (hx7-l3-w0.bin). Prints "qa2-test: PASS" with the worst normalized errors and
// returns 0 when every check passes.
int fnfl_qa2_test(int device, const std::string & mode, const std::string & model, const std::string & fixture);
