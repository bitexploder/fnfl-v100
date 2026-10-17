#pragma once

// hcp-test (PB1): the prompt HC (hc_prompt.h) against FP64 of hc.h's formulas, replicated (1e-6 a row) and split Q8_0
// (1e-5), the FP16 copy bitwise, a rerun bitwise; the prompt exchange's pack and combine bitwise against the CPU and a
// stale generation refused. PB3: the tensor path (hct_*) against FP64 at a loose bug-check tolerance (Amendment 2),
// synthetic and range-stress rows. mode "san": small cases (Compute Sanitizer); "real=DIR": the full cases, then PB1's saved real
// inputs in DIR (manifest.tsv; the coefficients read on the CPU from its GGUF). Prints "hcp-test: PASS", returns 0.
int fnfl_hcp_test(int device, const char * mode);
