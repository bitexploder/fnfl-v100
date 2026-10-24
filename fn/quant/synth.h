#pragma once

// Seeded generators of valid random blocks of the deployed weight formats (formats.h), for FNFL's synthetic test
// weights in place of a quantizer (C1b; the ranges are C1b's Amendment 1). No GPU code.
//   - Every F16 block scale (d, and dmin for Q4_K and Q5_K): magnitude log-uniform in [1e-4, 1e-1], rounded to F16.
//     Positive where the format stores a magnitude (the IQ2 and IQ3 grids, whose signs are separate bits, and Q4_K's
//     and Q5_K's d and dmin); a random sign elsewhere (Q2_0, Q8_0, IQ4_NL, IQ4_XS, Q6_K, and ME1's Q5_1 d and m).
//   - Every other bit (sub-scales, mins, nibbles, sign bits, grid indices, codes) uniform over its whole range.
//   - BF16, F16 and F32 values: N(0, 1) rounded to the type, finite.
// The generator is splitmix64 with Box-Muller and exp/log in FP64, so a seed gives the same blocks on every run.
//
// synth_blocks_deployed (C1b Amendment 4, ep-test only): the same blocks except that each block's d is drawn
// log-uniformly between the deployed median |d| / 10 and 2 x the deployed largest |d| of the format, so the prompt
// experts' F16 GLU sees realistic magnitudes. Only the formats ep-test uses have these ranges. ME1: the Q4 candidate's
// formats (Q4_K, Q5_K, Q5_1, Q8_0) have ranges chosen to match IQ4_NL's deployed weight magnitudes, not measured
// (synth.cpp), with their second scale (dmin, Q5_1's m) from its own range.

#include "fn/quant/formats.h"

#include <cstdint>

namespace fn::quant {

// the seed of a test's case index: 0x464E464C ("FNFL") plus the index
constexpr uint64_t synth_seed_base = 0x464E464Cu;
constexpr uint64_t synth_seed(uint64_t case_index) {
    return synth_seed_base + case_index;
}

// ep-test's own seed stream: 0x45503031 ("EP01") plus the case index, so no other test's blocks change
constexpr uint64_t synth_seed_ep_base = 0x45503031u;
constexpr uint64_t synth_seed_ep(uint64_t case_index) {
    return synth_seed_ep_base + case_index;
}

// n_blocks valid random blocks of f from seed into dst (n_blocks x info(f).block_bytes bytes); false on n_blocks < 0
// or a null dst
bool synth_blocks(wfmt f, uint64_t seed, int64_t n_blocks, uint8_t * dst);

// the deployed |d| range of f (synth.cpp names the source; ME1's are chosen); false for a format without one
bool deployed_d_range(wfmt f, double & lo, double & hi);

// as synth_blocks with each block's d from the deployed range; false also for a format without one
bool synth_blocks_deployed(wfmt f, uint64_t seed, int64_t n_blocks, uint8_t * dst);

} // namespace fn::quant
