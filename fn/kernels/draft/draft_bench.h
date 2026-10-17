#pragma once

// DR1 Stage C: the standalone bench of the draft block's own kernels (fn/kernels/draft/draft.h; STANDARDS "a
// standalone bench, timed in isolation"), no model data. Each point is timed in CUDA graphs, 3 warm-up and 20 timed
// replays, median and p90 of the event span, with the SM clock read before and after (fn::xp1::clock_probe):
//   DA-TC, the draft step's row (prepare with q, then attend) at 4,096, 58,581, 131,072 and 196,608 cells on dev0:
//          warm = the code warm, K and V rotated over at least 200 MB of cache copies (every cell read from DRAM, as
//          in a step: the cells were written rounds ago); cold = an L2 flush and the instruction-cache evictor before
//          each launch (fn/kernels/dense/dense_cold.h), less the graph of those alone. Beside Stage A section 3's
//          frozen DA-TC low / mid / high.
//   DA-TC prepare alone: the step's row (q and K/V) and the catch-up's B_c = 1..4 (K/V only, with the slot release).
//   X-DRAFT-IN: send then receive on both cards (dev0 card 0, dev1 card 1) at rows 1 (the step) and 4 (B_c = 4), 32
//          exchanges a graph over slots 0..2 in turn, each card's span a exchange (both graphs launched before
//          either is waited on). Beside Stage A section 5's draft-input line (6.0 / 9.0, which includes the embed).
// Writes one JSON line a point to stdout.
int fnfl_draft_kernel_bench(int dev0, int dev1);
