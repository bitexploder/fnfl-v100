#pragma once

// xp1-bench: XP1's pair timed on both cards (the ticket's xp1-bench), one CUDA graph per card per variant and B,
// launched together.
//   calibrate: each stand-in's padding so its alone time lands on its setting (targets.json: target or today); the
//              pads are written to (or read from) a calibration file, so every later process times the same settings.
//   untraced:  per variant and B, 3 warm-up and 20 timed replays: each card's span (events around its launch) and the
//              start skew (the cards' round stamps on one clock, through the measured offset); the mean distinct
//              experts the replayed decodes route to.
//   alone:     each stand-in and FNFL kernel alone (7 trials of a graph of 20 x [L2 flush, op] less a graph of 20
//              flushes), each exchange at the same bytes isolated on both cards (21 back to back a graph, both
//              directions at once), and HC2's sublayer; the SM clock read beside each figure.
//   rounds:    the variant loop repeated and interleaved (--rounds), each replay with the SM clock read beside it.
//   trace:     one variant and B for an nsys capture (--cuda-graph-trace=node), with the node manifest the split
//              script (work/XP1/src/xp1_split.py) maps the trace by.
//   dense:     DP2's launches in V2 (pair.h); V2 only, with an existing calibration file; the dense_rest stand-ins
//              unpadded, the replaced ones not timed alone, and the FNFL kernels alone those of V2's program.
// Results are JSON lines in OUT/bench.jsonl; a summary goes to stdout.

#include <cstdint>
#include <string>
#include <vector>

struct fnfl_xp1_bench_args {
    int32_t dev0 = 0, dev1 = 1;
    std::string out_dir;
    std::string synth_dir;
    std::string hx7_dir;
    std::string targets;  // tickets/XP1/targets.json reduced to "layer key target today" lines (work/XP1/src)
    std::string calib;    // the calibration file: read if it exists, else written
    std::string speed = "target";
    std::vector<std::string> variants = {"V0", "V1", "V2"};
    std::vector<int32_t> rows = {1, 3};
    bool synthetic = false;
    bool alone = false;
    bool trace = false;
    bool calibrate_only = false;
    bool dense = false;
    int32_t warmup = 3;
    int32_t replays = 20;
    int32_t rounds = 1; // the variant loop repeated, interleaved: each variant's figure is the median over all rounds
    int32_t gdn = 0;    // GD1 step 7 (pair_config::gdn): 0 the stand-ins, 1 gdn_core (--gdn core), 2 omitted (omit)
    int32_t qsa = 0;    // QA2 step 7 (pair_config::qsa): 0 the stand-in, 1 attend (--qsa attend), 2 omitted (omit)
};

int fnfl_xp1_bench(const fnfl_xp1_bench_args & a);
