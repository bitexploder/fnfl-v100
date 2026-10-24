#pragma once

// fn/loader for running (R2): one model load on both cards, ready for kernels, without load-check's read-back and
// account. The weights are placed as R1 places them; every state and workspace class of fn/model/state.h is reserved
// at the capacity and zeroed (STANDARDS: initialize all state); the EX1 grid table is filled; peer access is enabled
// both ways and the exchange rings are allocated and prefilled. ME1: with device[0] == device[1] both logical cards'
// classes are on the one device (no peer access; the prefill checkpoints in pinned host memory).

#include "fn/loader/loader.h"
#include "fn/model/model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::loader {

struct run_load_report {
    double open_s = 0;    // open_model: mappings, table checks, region plan
    double fill_s = 0;    // allocate and fill the weights
    double reserve_s = 0; // state classes, grids, peer access, rings
    double total_s = 0;   // open to ready (steady_clock)
    int64_t free_after[2] = {0, 0}; // cudaMemGetInfo once ready
    int64_t host_state_bytes = 0;   // ME1: state classes held in pinned host memory (one device: the checkpoints)
    // PN1: the PLE table's residency, the bytes pinned (the page-aligned range; 0 when lazy) and the loader's times
    runtime::ple_residency ple_table = runtime::ple_residency::lazy;
    uint64_t ple_pinned_bytes = 0;
    int64_t ple_foll_pin[2] = {-1, -1}; // the kernel's net FOLL_PIN pages just before and after the registration
    load_times times;
};

// PN1: with p.ple_table full, the PLE table is populated on its own thread beside the weight fill and registered
// after it, before load_for_run returns (pinning is synchronous: nothing runs on the model until it completes); any
// failure fails the load, never falling back to lazy.
bool load_for_run(const load_params & p, int32_t n_cells, model::model & m, run_load_report & rep, std::string & err);

// PN1: the load's PLE lines, "PLE table: full, <bytes> pinned (<mechanism>)" and the load_times ple_pin_s line
std::vector<std::string> ple_table_lines(const run_load_report & rep);

// the EX1 grid table (fn::experts::grid_table) into the "EX1 grid table" part of both cards
bool fill_ex1_grids(const model::model & m, std::string & err);

// every state class of sp (fn/model/state.h) reserved and zeroed (the host classes too, their bytes added to
// host_bytes); MU1, n_slots > 1: slot 0's slot classes in m.state, slot s's in m.slot_state[s - 1], the pooled ones as
// views of the KV pool (created here, its fixed tails zeroed, no page mapped), and the other-reader lane's classes.
// KV8: pool_cells (n_slots > 1) the pool's physical cells (0: sp.n_cells; kv_pool::init).
// MXMU1, two slots on one card with the host-backed tier: every slot class reserved per slot (no pool, no lane),
// refused before any allocation when MemAvailable cannot hold every slot's pinned history
bool reserve_state(model::model & m, const model::state_params & sp, int32_t n_slots, int64_t & host_bytes,
                   std::string & err, int32_t pool_cells = 0);
// KV8: every state class released (the slots' and the pool too), on drained devices, so reserve_state can lay the
// state out again (one loaded model's bf16 and q8 passes); the weights and the rings stay, and the caller fills the
// EX1 grid table again (fill_ex1_grids) after the new reservation
void release_state(model::model & m);

} // namespace fn::loader
