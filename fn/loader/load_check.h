#pragma once

// fnfl-drive load-plan and load-check (R1).
//
// load-plan (CPU only, no GPU): opens the deployed files, checks them against the placement, hashes every tensor's
// card slices (written to OUT/gguf-slices.sha256 for load-check) and runs the loader's fill and the placement check on
// the host (verify_host), so a load on the cards is spent only on a plan that already checks out.
//
// load-check (both cards): one model load. The CUDA context, the weights (allocated, filled, read back and hashed
// against the GGUF slices), every state and workspace class, the exchange rings, the prefill chunk buffers for 1,536
// and 3,072 rows, and the graph-instance allowance, with cudaMemGetInfo before and after each; the account per class
// and card, and the 192K chunk decision. Prints "load-check: PASS" and returns 0 when every placement sum and hash
// holds and every class is reserved. No kernel runs.

#include <cstdint>
#include <string>

struct fnfl_load_check_args {
    std::string out_dir;
    std::string model_path;
    std::string draft_path;
    int32_t n_cells = 0; // capacity a slot
    int32_t threads_per_card = 5;
};

int fnfl_load_plan(const fnfl_load_check_args & a);
int fnfl_load_check(const fnfl_load_check_args & a);
