#pragma once

// MX7's shadow comparison (TEST ONLY: fnfl-drive spec run --mx7-check with the share "shadow"; expires at MX7's
// disposition; engineer disposition 2026-10-21, load 6). On a sharing ring layer the share plans its CPU experts as
// usual, but the ring stages and the GPU computes every routed pair (no mask), the CPU computes its pairs into its own
// rows, and nothing is added back: the layer's output is the GPU's alone (share 0's). Each CPU pair's product is then
// compared with the GPU's product of the same pair (the two logical cards' partial rows at the pair's positions,
// summed), and, for a sample of pairs (one in 16 of the layer's CPU pairs, and the layer's 10 largest CPU-GPU
// errors), with an FP64 reference independent of both: the expert's GGUF rows decoded (the native views), the pair's
// token row of the layer's F16 input, gate and up, SwiGLU and down in FP64. The same reference with the CPU's own Q8_K
// gate/up activations (its first quantization) shows how much of the CPU-GPU difference that quantization explains.
// Errors are relative L2: ||a - b|| / ||b||.

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_cpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime {

struct mx7_shadow_pair {
    int32_t layer = 0, expert = 0, row = 0, slot = 0;
    int32_t job_row = 0; // the pair's row within its CPU job (0..7; MX7 added 6 to 8 rows a job)
    fn::experts::fmt gu = fn::experts::fmt::iq2_xxs, dn = fn::experts::fmt::q2_0;
    float w = 0;      // the pair's route weight
    float e_cg = 0;   // CPU against GPU
    float e_cpu = -1; // CPU against the FP64 reference (-1: not referenced)
    float e_gpu = -1; // GPU against the FP64 reference
    float e_q1 = -1;  // the reference with the CPU's Q8_K gate/up activations against the reference
};

struct mx7_shadow_in {
    int32_t layer = 0, rows = 0, pairs = 0;
    fn::experts::fmt gu = fn::experts::fmt::iq2_xxs, dn = fn::experts::fmt::q2_0;
    const int32_t * ids = nullptr;                      // host [rows x 10]: the layer's routing (card 0's)
    const int32_t * pair_of = nullptr;                  // host [pairs]: CPU pair k's pair index p = 10 row + slot
    const float * cpu = nullptr;                        // host [pairs][2560]: the CPU's products, F32
    const fn::experts::cpu::block_q8_k * act = nullptr; // host: the CPU's gate/up activations, by compact row
    const int32_t * compact = nullptr;                  // host [rows]: a token row's compact row (-1: none)
    const fn::experts::cpu::native_layer * native = nullptr;
    const uint16_t * x16_d = nullptr;  // device [rows][2560] F16: the layer's input (card 0's)
    const int32_t * pos_d[2] = {};     // device [rows x 10]: each logical card's sorted positions
    const uint16_t * part16_d[2] = {}; // device [rows x 10][2560] F16: each logical card's partials
    const float * wts_d = nullptr;     // device [rows x 10]: the route weights
    void * stream = nullptr;           // the layer's compute stream (both logical cards': one device)
};

// one layer's comparison, appended to out; false (err) on a copy's or a decode's failure. Without native views
// (run-dry's zero backing) the pairs are compared with the GPU's only
bool mx7_shadow_layer(const mx7_shadow_in & in, std::vector<mx7_shadow_pair> & out, std::string & err);

} // namespace fn::runtime
