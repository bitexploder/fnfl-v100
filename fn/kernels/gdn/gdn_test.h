#pragma once

// gd-test (GD1): the convolution epilogue (conv_silu) and gdn_core (fn/kernels/gdn/gdn.h) on one card's head set
// against a CPU FP64 reference written from the pinned lines (gdn_test.cpp), with a CPU F32 reference in the kernel's order beside it for scale. The real ssm_conv1d,
// ssm_dt.bias, ssm_a and ssm_norm of layers 10 and 40 are read on the CPU from the deployed GGUF (no model load);
// everything else is synthetic. The card is the device index (0 or 1).
// mode: "all" (short, chain, map, inv), one of those, or "san" (64 positions at B = 1 and 3, for Compute Sanitizer).
// Outputs are judged per row and head against 1e-4 max(1, 1e-3 / cond) (GD1 Amendment 1). Prints "gd-test: PASS" with
// the worst normalized errors and returns 0 when every check passes.
int fnfl_gd_test(int device, const char * mode, const char * model);
