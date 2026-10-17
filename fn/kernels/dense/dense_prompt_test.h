#pragma once

// pd-test (PB1): the prompt dense products (dense_prompt.h) against FP64 on fn/quant's decoded weights and the same
// FP16 activations, every deployed format in its resident layout; the exact code conversion weight by weight against
// fn/quant's decoder (every code value of each format's domain seen, every packed lane position); the two-segment
// activation operand (eh_proj), the row scale, the BF16 products with both row mappings, the GLU; a second run
// bitwise. mode "san": fewer and smaller cases (Compute Sanitizer). Prints "pd-test: PASS" and returns 0 when every
// check passes.
int fnfl_pd_test(int device, const char * mode);
