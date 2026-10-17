#pragma once

// fnfl-drive layer-test and layer-bench (R2).
//
// layer-test: one model load. Layers 10 and 11 decode the first N positions of the reference (default 4,096) on both
// cards in rows of B = 1, 2, 3, 4, 1, ... from empty state, teacher-forced (each layer's input rows are the pinned
// engine's l_last of the layer before), twice, then chained (layer 11 fed by FNFL's own layer 10). Checks against the
// pinned engine's decode-mode dump (ref_dump.cpp), with its prompt-mode dump as the yardstick:
//   outputs (LP1: l_last, X-A's and X-F's results, both halves): per position max |FNFL - ref| over the row / max |ref|;
//            p50, p99 and max over positions each at most 2x the same statistic of prompt mode against decode mode
//            (ref-spread.json, R2's ref_spread.py over the same dumps);
//   routing: each row's 10 experts equal the reference's as sets, or the reference's scores of the experts that differ
//            are within 2x the router's own error (that row's largest prompt-against-decode score difference);
//   selection (layer 11): each row's cells equal the reference's, or the reference's scores of the blocks that differ
//            are within 2x the indexer's own error (that row's largest prompt-against-decode block-score difference),
//            and the differing rows at most 2x the pinned engine's own (ref-spread.json select-11);
//   determinism: the second teacher-forced pass is bitwise identical to the first;
//   batch invariance (LP1): the first 512 positions again at B = 1, every recorded output bitwise equal to the cycled
//            pass's at the same position;
//   exchanges and kernels: both cards' exchange error words, attend's and select's error words and EX1b's status stay
//            zero; the two cards' routing (each its own router) and selection bodies are equal.
// Prints "layer-test: PASS" and returns 0 when every check holds. Mode san: N positions (64), one pass, no PASS
// statistics (the Compute Sanitizer run); its state is reserved for 8,192 cells, not R1's capacity, so the tool has
// device memory beside the weights.
//
// Both tools first list the pieces of layers 10 and 11 FNFL has not built (fn::model::unbuilt_pieces); when there are
// any they print each one and return fnfl_layer_not_built, before reading a reference or loading the model.
//
// layer-bench: one model load. A teacher-forced pass over the 4,096 positions builds the history, then for B = 1..4 a
// CUDA graph per card of the chained pair at depth 4,096 (3 warm-up replays, 20 timed: median and p90 per card), and
// one eager step per B with an event after every piece (per-piece times, host gaps included). Then (LP1) the history
// replaced by 131,072 cells of synthetic values (K and V caches, index keys and summaries, GDN states and histories
// random), and the same graphs at B = 1 and 3 at that depth.

#include "fn/runtime/ple_host.h"

#include <cstdint>
#include <string>

struct fnfl_layer_test_args {
    std::string ref_decode;  // ref_dump's decode-mode directory (raw .bin files)
    std::string ref_prompt;  // its prompt-mode directory
    std::string spread_json; // the yardstick (R2's ref-spread.json); default: beside ref_prompt
    std::string out_dir;
    std::string model_path;
    std::string draft_path;
    int32_t n_pos = 4096;
    int32_t n_cells = 198720; // R1's capacity
    bool san = false;
    fn::runtime::ple_residency ple_table = fn::runtime::ple_residency::lazy; // PN1: --ple-table
};

// the exit status of a tool stopped by a piece FNFL has not built (0 PASS, 1 failed, 2 usage)
constexpr int fnfl_layer_not_built = 3;

// prints "<tool>: not_built: layer L: <piece>" for each unbuilt piece of layers 10 and 11; true when there is one
bool fnfl_layer_report_unbuilt(const char * tool);
int fnfl_layer_test(const fnfl_layer_test_args & a);
int fnfl_layer_bench(const fnfl_layer_test_args & a);
