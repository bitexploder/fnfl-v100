#pragma once

// qsp-test (PB1): the prompt QSA (qsa_prompt.h): the raw key write bitwise; the prompt selection bitwise against the
// decode select run one row at a time on the same summaries and queries (rows below, across and above 512 visible
// blocks, across slabs; deterministic ties, every block's score equal; pathological keys: zero, one-hot 1e19, the
// smallest subnormal and -0.0 blocks); the K/V write against FP64 (K within one BF16 step, V bitwise); attention over
// the prompt's own selection against FP64 (1e-5 a row and head) and a rerun bitwise. mode "san": small cases.
// "qsp-test: PASS".
int fnfl_qsp_test(int device, const char * mode);
