#pragma once

#include <cstdint>

// xchg-test (HC2 step 2): both messages between dev0 and dev1, rounds back to back at each row count in rows_mask
// (bit B-1; B = 1 and 3 are built), every payload word checked on both cards, including rounds that carry NaNs (the
// sentinel pattern among them) and infinities. Prints "xchg-test: PASS" and returns 0 when every check passes.
int fnfl_xchg_test(int dev0, int dev1, int32_t rounds, uint32_t rows_mask);
