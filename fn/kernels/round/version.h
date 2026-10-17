#pragma once

// SR1: the versioned state's physical slots (FC-ST1, FC-ST2). Every versioned object (the GDN recurrent state and
// convolution history of each GDN layer, and the PLE history) keeps n_version physical slots. One device word cur
// (0 .. 5, fn::round::state::cur in the model pass) names the slot of the committed version; logical version v
// (0: committed, 1 .. B: the state after the round's first v rows) lives in slot (cur + v) mod 6. Readers read logical
// version 0 and write 1 .. B through version_slot, the one function that maps a logical version to a physical slot;
// no kernel computes a slot itself. Commit of k sets cur to version_slot(cur, k); a rollback within the last step to j
// kept rows sets cur to version_slot(cur_pre, j). Six slots suffice (SP2, B <= 5): a round writes the five slots other
// than cur, and the pre-step slot is overwritten only by the next round, which starts after the step is acknowledged
// (RS1 section 3).
//
// Host and device: the CUDA compiler sees __host__ __device__; a host compiler sees a plain inline function.

#include <cstdint>

#if defined(__CUDACC__)
#define FN_ROUND_HD __host__ __device__
#else
#define FN_ROUND_HD
#endif

namespace fn::round {

constexpr int32_t n_version = 6; // 1 + the most rows a round writes (SP2: 5)

// the physical slot of logical version v (0 .. 5) when the committed version is at slot cur (0 .. 5): one add and one
// conditional subtract, no division
FN_ROUND_HD inline int32_t version_slot(uint32_t cur, int32_t v) {
    const int32_t s = static_cast<int32_t>(cur) + v;
    return s >= n_version ? s - n_version : s;
}

} // namespace fn::round

#undef FN_ROUND_HD
