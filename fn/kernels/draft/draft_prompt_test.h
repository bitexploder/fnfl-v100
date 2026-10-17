#pragma once

// dcp-test (PB1): the draft's catch-up pieces (draft_prompt.h): e and h normalized against FP64 (one FP16 step), row
// t's h the hidden row t - 1 and row 0's the predecessor; the mixed row bitwise f16 of each card's half on both cards;
// K within one F16 step of FP64 (the norm and IMROPE), V bitwise f16(v). mode "san": small cases. "dcp-test: PASS".
int fnfl_dcp_test(int device, const char * mode);
