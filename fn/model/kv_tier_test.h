#pragma once

#include <string>

// ME1: the host-backed KV tier (fn/model/kv_tier.h) against a resident cache, no model data. One slot's life on both
// logical cards of DEVICE (tp / tp-san: physical devices 0 and 1): target verify rounds of B = 1..5 rows on a QSA
// layer (attend.h) with selections that reach
// below the window, across its edge and onto its four-cell block boundary, sparse draft chains of 1..4 steps
// (DA-TC; step 0 gathers, the later steps reuse it) and catch-ups, rejected rows, checkpoint restores below and above
// the window, prompt chunks through the staging (two halves, the draft's K and V straight to host memory) and a
// divergent prefix after a restore, up to the 262,144-cell limit. Every attention output (out, codes, scales) and
// error word bitwise equal to the same launch on the resident cache; the staged history, and the host history after
// a flush, bitwise equal to the resident cells. mode "all": 262,144 cells, W 4,096; "san": 32,768 cells, W 1,024,
// fewer rounds (Compute Sanitizer). MEKV covers eager and graph-window forms in every case. Prints "kv-tier-test: PASS"
// and returns 0 when every check passes.
int fnfl_kv_tier_test(int device, const std::string & mode);
