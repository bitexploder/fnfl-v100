#pragma once

// ME5 (row-aware miss service): which of a decode window's nonresident experts the GPU lane takes. The count is the
// split's (policy::gpu_misses of the learned share, or a forced mode's), unchanged; ME5 changes only which misses
// those are. A miss serving several of the window's rows costs the lane the same bytes as one serving a single row
// (every expert entry of a layer has one extent, copied once), but costs the CPU a product per row. So the lane takes
// the misses with the most rows: at the same count and the same copied bytes, the CPU's row incidences (the sum of
// its experts' row counts) are the fewest any choice of that count gives. Fewer CPU rows is not a latency claim.
//
// Ties keep the existing rule (cache_service.cpp at e1c0a99: the last n_gpu misses in route order go to the lane):
// among equal row counts the later in route order goes first, so a window whose misses all serve the same number of
// rows (every B = 1 window) is assigned exactly as before. The chosen misses keep route order for their lane
// positions and launches; nothing else about the window (IDs, masks, slot bases, gate weights) changes. CPU only, no
// state: the host tests call it directly.

#include <cstdint>

namespace fn::runtime::split {

constexpr int32_t max_misses = 50; // experts::max_slots: a window's distinct experts at most

// misses: the window's nonresident distinct experts' indices, in route (first-appearance) order, nm of them; mask[u]:
// expert u's row mask (bit b, row b). to_gpu[i] (i < nm) set to 1 for the n_gpu chosen, 0 for the rest. n_gpu is
// clamped to [0, nm]; returns the count chosen (an empty window or n_gpu 0 chooses none)
int32_t choose_gpu(const int32_t * misses, int32_t nm, const int32_t * mask, int32_t n_gpu, uint8_t * to_gpu);

// the rows expert u serves (mask's population)
inline int32_t rows_of(int32_t mask) {
    return __builtin_popcount(uint32_t(mask));
}

} // namespace fn::runtime::split
