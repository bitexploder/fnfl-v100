#pragma once

// FMU2: the joint dense product (dense.h product_joint) at U = 2, 3 and 4 lanes against the lone product at B = 5 on
// each lane's rows: every output bitwise equal (products, the scaled shared down, the GLU's codes, scales and F32
// values, the router's IDs, weights and gate, the convolution's post-SiLU values and every history version with a
// different committed slot and history a lane), through activations in memory and through per-lane X-HMIX slots; a
// second joint run bitwise; the F32 products against FP64 on the decoded synthetic weights (dense.h's bound). mode:
// "full" or "san" (fewer cases, for Compute Sanitizer); only: run just the cases whose name contains one of its
// '|'-separated parts (null: all; FMU2 capture 3's ncu of chosen sites). FMU4: then the case "sampled pick", the joint
// draft step's sampled pick (fn/kernels/samp/samp_joint_test.h) on this device and the other (san: loopback on this
// one). Prints "dpj-test: PASS" and returns 0 when every check passes.
int fnfl_dpj_test(int device, const char * mode, const char * only);
