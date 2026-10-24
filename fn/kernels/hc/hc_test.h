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

// FMU2: hc::sub_joint and head_sub_joint for 2..4 lanes of max_rows rows (each its own inputs, outputs and rings, all
// on one set of weights) bitwise each lane's own sub or head_sub launches after every sublayer of a sequence (every
// X-HC and X-HMIX slot, x and inject, the error words; both cards), a second joint run bitwise the first, and the
// refusals.
// Prints "hcj-test: PASS" and returns 0 when every case passes. DEV0 == DEV1: loopback, for Compute Sanitizer.
int fnfl_hcj_test(int dev0, int dev1);
