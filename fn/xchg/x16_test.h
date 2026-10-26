#pragma once

// fnfl-drive x16-test (FMS2 item 7): the joint 16-bit X-A / X-F (fn/xchg/xchg.h send_half_joint16, recv_add_joint16)
// on two cards with synthetic payloads, no model. At joint rows 10, 15 and 20 (U = 2 .. 4), both cards send the
// peer's half of a partial and receive their own: the edge values (+-65,504, 65,519, 65,520 and beyond, FLT_MAX, the
// smallest normal, subnormals, the ties at both ranges, F32 subnormals, NaNs, +-inf, +-0) among random values and
// random ties. Checked bitwise against the host's rounding (fn/xchg/x16_ref.h): every sent half and the rows past the
// payload still the sentinel; every output own + the peer's value (NaN as any NaN); the slots back to the sentinel; the
// error words 0, except err_x16_range on the sender of an overflowing value (and only there). Prints
// "x16-test: PASS" and returns 0. Two distinct cards with peer access.

#include <cstdint>

int fnfl_x16_test(int32_t dev0, int32_t dev1);
