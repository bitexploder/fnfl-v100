#pragma once

// xp1-test: XP1's pair on both cards. HX7's traffic loader against its files (Amendment 3); V0's exchanged halves
// against the CPU references (the stand-ins' defined function bitwise; HC2's and EX1b's FP64 references at their own
// tests' bounds; C2's quantize bitwise against fn/quant/act_ref); X-ROUTE's and X-SELECT's round trips (the body
// bitwise); every stand-in's weight XOR; then every variant's outputs on both cards bitwise equal to V0's (each card's
// residual half after layer 11 and the final selection), at B alternating between consecutive steps (R15), eagerly and
// (unless no_graph) from a captured graph. Prints "xp1-test: PASS" and returns 0 when every check passes.

#include <cstdint>
#include <string>
#include <vector>

struct fnfl_xp1_test_args {
    int32_t dev0 = 0, dev1 = 1;
    std::string synth_dir;
    std::string hx7_dir;
    std::vector<std::string> variants = {"V0", "V1", "V2"};
    std::vector<int32_t> steps = {1, 2, 3, 1, 2, 3}; // B of each consecutive step
    bool no_graph = false;                          // Compute Sanitizer: eager steps only
    bool synthetic = false;                         // --traffic synthetic
};

int fnfl_xp1_test(const fnfl_xp1_test_args & a);
