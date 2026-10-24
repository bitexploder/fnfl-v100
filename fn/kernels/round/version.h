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

// ME11: the GDN recurrent state's selector, one word beside cur. The convolution and PLE histories keep cur's six
// slots; the GDN matrices keep two banks and two record sets a layer and card (fn/kernels/gdn/gdn.h). The word is
// bank | set << 1 | k << 2: the committed matrix is bank `bank` with the first k (0 .. 5) steps of record set `set`
// replayed onto it. These functions are the one mapping, on the host and the device; no other code decodes the word.
//   a target round (gdn_core) reads logical version 0 (the bank and k records), stores it into the other bank and
//     records its own steps into the other set: its commit of k >= 1 selects (1 - bank, 1 - set, k);
//   a prompt chunk (the scan) reads version 0 into the other bank (half 0), then works on that bank in place (half 1):
//     its commit selects (1 - bank, set, 0);
//   a cut to j >= 1 rows of the last step selects (1 - bank_pre, 1 - set_pre, j); a cut to 0 or a failure the
//     pre-step word; k = 0 (bootstrap, draft rounds, a zero-version commit) leaves the word.
// A round never writes its committed bank or set, so the pre-step word stays valid until the next round starts.
constexpr int32_t gdn_max_steps = 5; // SP2: a round's rows

FN_ROUND_HD inline bool gdn_valid(uint32_t g) {
    return (g >> 2) <= static_cast<uint32_t>(gdn_max_steps);
}

FN_ROUND_HD inline uint32_t gdn_word(int32_t bank, int32_t set, int32_t k) {
    return static_cast<uint32_t>(bank & 1) | static_cast<uint32_t>(set & 1) << 1 | static_cast<uint32_t>(k) << 2;
}

FN_ROUND_HD inline int32_t gdn_bank(uint32_t g) {
    return static_cast<int32_t>(g & 1u);
}

FN_ROUND_HD inline int32_t gdn_set(uint32_t g) {
    return static_cast<int32_t>((g >> 1) & 1u);
}

// the records to replay; 0 for a word outside the contract, so a reader never leaves its set (the commit refuses such
// a word, fn::round::commit)
FN_ROUND_HD inline int32_t gdn_steps(uint32_t g) {
    return gdn_valid(g) ? static_cast<int32_t>(g >> 2) : 0;
}

// the bank holding logical version v's matrix: v = 0 the committed bank (replayed), v >= 1 the other bank
FN_ROUND_HD inline int32_t gdn_bank_of(uint32_t g, int32_t v) {
    return v == 0 ? gdn_bank(g) : 1 - gdn_bank(g);
}

// after a target round's commit of k (1 .. 5), or a cut of the last such step to j = k rows
FN_ROUND_HD inline uint32_t gdn_after_round(uint32_t g, int32_t k) {
    return gdn_word(1 - gdn_bank(g), 1 - gdn_set(g), k);
}

// after a prompt chunk's commit
FN_ROUND_HD inline uint32_t gdn_after_prompt(uint32_t g) {
    return gdn_word(1 - gdn_bank(g), gdn_set(g), 0);
}

} // namespace fn::round

#undef FN_ROUND_HD
