#pragma once

// APF2: the dense family's registry of fused tile configurations (the kernel-size ladder's dense entries).
// pl_tc: pq_tc's product (fn/kernels/dense/dense_prompt.h: the same sources, activation operand and epilogue) on one of
// the registry's tiles, with split K. Each entry is a compiled kernel the prompt path launches through the selector
// (fn/model/ladder.h); pd_convert + pd_tc (the retained route) stays the selector's fallback candidate.
//   tiles:   TurboMind's sm70 dense list (1Cat-vLLM 1d7321bf2d, kernels/gemm/kernel/sm70_884_4.cu:160-240): 8, 16, 32,
//            64 and 128 rows by 128 columns, and 128 by 256, K 32 a stage, two stages (fn/kernels/tc/tc_ladder.cuh).
//   weights: decoded in the tile loader: Q8_0 (DP2's layout and EX1's down layout) and BF16 rows (with the prepared
//            column scales pd_bf16_scales writes), Q4's dense inventory, on their fast loaders; the IQ3 inventory's other
//            DP2 formats (Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S) and EX1's IQ4_NL and Q2_0 through the device decoder's
//            quarter decode (a correct, simple loader: the IQ3 file runs the retained routes, with no table).
//            Each weight's FP16 value is pd_convert's (or pd_convert_bf16's): code x scale (minus the min) rounded once.
//   split K: splits 1 .. pl_max_splits (at most the K stages); partials F32 in the caller's workspace, summed in split
//            order by the last split of each tile (no float atomics; tc_ladder.cuh). splits 1 writes directly.
//   numerics: FP16 operands, F32 accumulators, each split's K in order then the splits in order: a rerun is bitwise; a
//            different tile is the same order, a different split count another. Bound (pd-test): the family's FP64
//            contract, max |err| / max |ref| a row, for every tile and split count.
//   budget:  the ptxas reports (pl_resources: registers, spills, resident CTAs an SM for the cost model).

#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/tc/ladder.h"

#include <cstdint>

namespace fn::dense {

constexpr int32_t pl_n_tiles    = fn::tc::lad_n_tiles;
constexpr int32_t pl_max_splits = fn::tc::lad_max_splits;
using pl_tile                   = fn::tc::lad_tile;
using pl_ws                     = fn::tc::lad_ws;
inline constexpr const pl_tile (&pl_tiles)[pl_n_tiles] = fn::tc::lad_tiles;

// the output tiles of a product of rows x n on tile t, and the partial floats splits > 1 needs
using fn::tc::lad_split_floats;
using fn::tc::lad_tile_count;

// whether the registry takes a's sources (format and layout)
bool pl_takes(const pq_args & a);
// the product of a on tile t with splits (1 .. min(pl_max_splits, K / 32)); false (nothing enqueued) on an argument
// pq_tc rejects, an unknown tile or split count, or a workspace too small for the split
bool pl_tc(const pq_args & a, int32_t t, int32_t splits, const pl_ws & ws, void * stream);
// every instance's shared memory opt-in on the current device (before any stream capture)
bool pl_attributes();
// tile t's instance for a's sources: registers, local bytes, shared bytes and resident CTAs an SM
bool pl_resources(const pq_args & a, int32_t t, pd_resources & r);

}  // namespace fn::dense
