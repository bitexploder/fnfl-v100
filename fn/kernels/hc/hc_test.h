#pragma once

#include <cstdint>

// The HC kernels against an FP64 CPU reference (HC1 step 3). Prints "hc-test: PASS" and returns 0 when every case
// passes. rows_mask: bit B-1 selects row count B (0xf: all four).
int fnfl_hc_test(int device, uint32_t rows_mask);

// HC2: the sublayer on both cards (dev0 holds card 0's half, dev1 card 1's), both arms of hc_up, 6 sublayers back to
// back per case, against the HC1c kernels run as two halves on dev0 (bitwise) and the FP64 reference (1e-6). Prints
// "hc2-test: PASS" and returns 0 when every case passes. rows_mask as fnfl_hc_test. DR1: then the Q8_0 forms (arm 0):
// with power-of-two scales bitwise the BF16 kernels on the same weights, with general FP16 scales against FP64 (1e-6),
// every sublayer bitwise sublayer 0, both cards equal, batch invariance and the rejects.
int fnfl_hc2_test(int dev0, int dev1, uint32_t rows_mask);
