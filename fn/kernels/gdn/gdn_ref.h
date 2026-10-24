#pragma once

// ME11 TEST REFERENCE (gd-test, gdp-test, state-test, rollback-test, prefill-test): never in a model pass.
//   core_ref: gdn_core as it was before ME11 (eee8f38 fn/kernels/gdn/gdn.cu, verbatim), writing every row's state as a
//     logical version of a six-slot buffer [6][24][128][128] at fn::round::version_slot(*cur, v). The compact
//     recurrence (gdn.h) is checked bitwise against it: the outputs of every round and the matrix the replay
//     materializes for every accepted prefix.
//   replay_host: the committed matrix of a compact layer on the CPU: bank b with the first k steps of set r replayed,
//     each cell m <- fmaf(k_i, delta_j, m * decay) in F32, round to nearest, no flush (the device replay's IEEE
//     operations, so the same bits).
//   read_committed: a layer's committed matrix from the device (its selector's bank and set, then replay_host).

#include "fn/kernels/act/act.h"

#include <cstdint>
#include <vector>

namespace fn::gdn::ref {

struct core_ref_args {
    const float * conv = nullptr;
    const float * z = nullptr;
    const float * alpha = nullptr;
    const float * beta = nullptr;
    const float * dt = nullptr;
    const float * a = nullptr;
    const float * norm_w = nullptr;
    float * state = nullptr;        // [6][24][128][128]
    const uint32_t * cur = nullptr; // device: the committed slot, 0..5
    float * gated = nullptr;
    act::out_view gated_q = {};
    void * ws = nullptr; // gdn::workspace_bytes()
    int32_t rows = 0;
};

// false, with nothing enqueued, for rows outside 1..5, a null pointer or a null stream
bool core_ref(const core_ref_args & a, void * stream);

// out [24][128][128] from bank [24][128][128] and the record set [5][rec_step_floats], steps 0..5
void replay_host(const float * bank, const float * rec_set, int32_t steps, float * out);

// layer state banks [2][24][128][128] and records [2][5][...] on the device: the matrix selector gsel names, into out
// (24 x 128 x 128); synchronous; false on a CUDA error or a selector outside the contract
bool read_committed(const float * banks, const float * recs, uint32_t gsel, std::vector<float> & out);

} // namespace fn::gdn::ref
