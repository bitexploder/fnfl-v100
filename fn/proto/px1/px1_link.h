#pragma once

// PX1 Part A (prototype, not engine code): large two-way card-to-card messages while both cards compute. Both
// directions at once (card 0 -> card 1 and card 1 -> card 0), 1, 2, 4, 8 and 16 MiB, two protocols:
//   A3: SM-driven stores through fn/xchg's put_run (72 sender blocks of 128 threads, the X-HC / X-HMIX grid), and a
//       receiver that polls every 16-byte word until it is not the sentinel and resets it (X-HMIX's four steps);
//   A5: cudaMemcpyPeerAsync on a stream of its own on each card.
// Three conditions on both cards: idle; a plain coalesced HBM read kernel (HX6's); EP1's ep_gate_up at 1,536 rows,
// repeated in a CUDA graph, on synthetic weights. Prints one JSON line a point.

#include <cstdint>

// mode: "all" (both protocols x 5 sizes x 3 conditions, 3 warm-up and 20 timed iterations each), "nsys" (the 8 MiB
// points only, 1 warm-up and 5 timed, for one nsys capture) or "san" (the A3 sender and receiver at 1 MiB, card by card
// in stream order with no concurrency, for Compute Sanitizer). cache: the synthetic expert weights (ex-prep); routes:
// FN4's routes.txt. Returns 0 when every point ran with no mismatch and no poll error, 4 on a host or device timeout.
int fnfl_px1_link(int32_t dev0, int32_t dev1, const char * mode, const char * cache, const char * routes);
