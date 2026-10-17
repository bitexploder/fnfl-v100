#pragma once

// gdp-test (PB1): the prompt GDN core (gdn_prompt.h) against FP64: the convolution (outputs, the new history bitwise),
// the norms and gates, and the chunked recurrence with the gated norm against the token-serial FP64 recurrence
// (gdp_reference, on the device, FP64) at PB3 Amendment 2's loose bug-check tolerance: single runs (1, 63, 64, 65,
// 1,536 rows and an all-zero input), a 6-run chain (9,216 rows) and a 128-run chain of 1,536-row halves (196,608 rows),
// slow-decay heads in both chains; a rerun bitwise; PB3's recurrence kernels' resources (no local memory). mode "san":
// small cases. Prints "gdp-test: PASS".
#include <cstdint>

int fnfl_gdp_test(int device, const char * mode);

// the FP64 reference (gdn_prompt_ref.cu): rows rows from conv [rows][5120] (q, k normalized), g and beta [rows][24];
// state [24][128][128] FP64 in and out; o [rows][3072] FP64 (with the 1 / sqrt(128) scale)
bool gdp_reference(int32_t       rows,
                   const float * conv,
                   const float * g,
                   const float * beta,
                   double *      state,
                   double *      o,
                   void *        stream);
