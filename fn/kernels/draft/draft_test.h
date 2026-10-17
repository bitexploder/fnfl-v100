#pragma once

// DR1 Stage C: the draft block's own kernels (fn/kernels/draft/draft.h) against FP64 CPU references written from the
// pinned lines (qwen4exp.cpp build_mtp): X-DRAFT-IN on both cards (dev0 card 0, dev1 card 1), DA-TC's prepare and
// attend on each card at depths up to 196,608 cells, and eh_proj's B_c adapter (amendment 1) at B_c = 1..4. mode
// "all": every case; "san": small depths only (Compute Sanitizer). Prints "draft-kernel-test: PASS" with the worst
// normalized errors and returns 0 when every check passes.
int fnfl_draft_kernel_test(int dev0, int dev1, const char * mode);
