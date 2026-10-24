#pragma once

// ME2: the expert cache's host policy, from Strata-V100 3d4ce4e0 (MIT; the notice is in expert_policy.cpp): the
// profile file and the learned ranking (src/core/expert_cache.cpp read_expert_profile, rank_learned_profile,
// write_expert_profile; tools/make_profile.py), the frequency-ranked swaps (src/program/generate.cpp adapt, near
// lines 9400-9440 and 9540-9560), the PCIe share of the misses (generate.cpp PcieAuto, line 1894; expert_source.cpp
// near line 3640) and the boot placement in rank order. No device work: expert_cache.cpp applies the decisions, and
// ex-test checks them without a model.
//
// Keys are layer x 512 + expert.

#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime::policy {

constexpr int32_t n_expert = 512;

// STRP v1: "STRP", uint32 version 1, n_layers, n_expert, slots, n_ranked, then n_ranked (uint16 layer, uint16 expert)
// pairs, then the n_layers x n_expert int32 table of each pair's rank (-1 unranked). ranked: keys. read checks the
// header against n_layers and 512, every pair in range and no pair twice
bool read_profile(const std::string & path, int32_t n_layers, std::vector<int32_t> & ranked, std::string & err);
// written to path + ".tmp" and renamed over path once complete
bool write_profile(const std::string & path, int32_t n_layers, const std::vector<int32_t> & ranked, std::string & err);
// every key ranked: resident first, then by heat descending, then by the prior ranking (an unranked key after every
// ranked one), then by key (rank_learned_profile's order)
std::vector<int32_t> rank_learned(int32_t n_layers, const std::vector<uint8_t> & resident,
                                  const std::vector<double> & heat, const std::vector<int32_t> & prior);

// The boot placement: class slots filled in rank order; once a class's slots are full its further keys take zone
// slots (fixed offsets in the zone, mixed classes, in rank order) while they fit; every key at most once. Main slots
// are numbered class by class (class k's first at first[k]), zone slots after them in creation order.
struct place_in {
    std::vector<int32_t> class_of_layer; // per layer
    std::vector<int64_t> entry;          // per class: a slot's bytes
    std::vector<int32_t> count;          // per class: its main slots
    int64_t zone_bytes = 0;
};
struct place_out {
    std::vector<int32_t> first;     // per class
    std::vector<int32_t> slot_cls;  // per slot (main, then zone)
    std::vector<int64_t> zone_at;   // per zone slot: its byte offset in the zone
    std::vector<int32_t> slot_key;  // per slot: the key placed, -1 empty
    int32_t n_main = 0;
};
place_out place(const place_in & in, const std::vector<int32_t> & ranked);

// The swap rule (adapt): per layer, the non-resident experts with usage >= min_use (candidates, descending) against
// the layer's residents (victims, ascending over the first min(candidates, victims)); pair i swaps while
// cand_i >= vict_i + min_gain. All layers' swaps sorted by gain (cand - vict) descending, the first max_swaps kept,
// then (layer_order) stably sorted by layer.
struct swap {
    float gain = 0;
    int32_t layer = 0, in = 0, out = 0; // experts of the layer: in becomes resident in out's slot
};
struct adapt_options {
    float min_use = 2.0f;
    float min_gain = 1.5f;
    float decay = 0.7f;
    int32_t max_swaps = 96;
    int32_t every = 4; // rounds between choices
    bool layer_order = true;
};
// usage: [n_layers][512]; slot_of: [n_layers][512], -1 when not resident
std::vector<swap> choose_swaps(int32_t n_layers, const float * usage, const int32_t * slot_of, const adapt_options & o);
// after a choice: heat += usage (when heat is kept), then usage *= decay (generate.cpp 9544, 9555)
void decay_usage(std::vector<float> & usage, std::vector<double> & heat, float decay);

// The PCIe share of the misses (PcieAuto's schedule): after each decode round, each leg's cumulative clock over its own
// experts (ME5 rider, Fable 2026-10-18): the lane's device-0 copy-event time and bytes (ME9's admission copies
// included, the same bytes), the CPU's input conversion plus pool phases and its experts. The share is refined after
// windows[0], then windows[1], .. more rounds, the last entry repeating for the whole run (48, 96, 96, 96, ..), to
// L / (L + k R), L = the lane's bytes / its copy time, R = blob bytes / the CPU's time per expert. It changes only when
// that unrounded reading is at least min_change (0.0375, three quarters of a step) from the current share; then it is
// rounded to a multiple of round (0.05) and clamped to [lo, hi] ([0.10, 0.60]), and kept when that is the current
// share. One update a window at most; fewer than min_cpu (300) CPU experts or no lane copy since the last update
// defers it (off after off_after (2,000) such rounds). ME2b: L / (L + R) is where the lane's and the pool's times for
// a window's misses are equal. ME3: the schedule's numbers are settings (fn/cache_settings.h split-*), k = 1 unless
// given; every attempt is reported (split_reading), unchanged ones too. link_gbps (the start-up probe) is reported
// beside the lane's reading; the share does not use it.
struct split_options {
    std::vector<int32_t> windows = {48, 96, 96}; // rounds before each update; the last repeats for the whole run
    int32_t min_cpu = 300, off_after = 2000;
    // min_change: the deadband on the unrounded reading (ME5 rider: three quarters of round)
    double round = 0.05, min_change = 0.0375, lo = 0.10, hi = 0.60, k = 1.0;
};
// one attempted refinement: its update (1-based), the CPU experts read, L, the time per CPU expert, R, the share before
// and after, and why it changed or stayed (reason null: no attempt this round)
struct split_reading {
    int32_t update = 0;
    int64_t samples = 0;
    // link_gbps: L, the lane's own reading (ME5 rider); probe_gbps: the start-up probe, for comparison; raw_share: the
    // unrounded L / (L + k R) the deadband compares
    double link_gbps = 0, probe_gbps = 0, tc_us = -1, rate_gbps = -1, raw_share = -1, old_share = 0, new_share = 0;
    double lane_ms = 0;     // the lane's copy time read (device 0)
    int64_t lane_bytes = 0; // and its bytes
    const char * reason = nullptr;
};
struct pcie_auto {
    bool on = false;
    split_options o;
    double link_gbps = 0.0, blob_bytes = 0.0; // link_gbps: the start-up probe (reported only)
    double mark_ms = 0.0, mark_lane_ms = 0.0;
    int64_t mark_jobs = 0, mark_lane_bytes = 0;
    double acc_ms = 0.0, acc_lane_ms = 0.0;
    int64_t acc_jobs = 0, acc_lane_bytes = 0;
    int32_t acc_win = 0, updates = 0;
    bool deferred = false; // this window's deferral reported
    // the legs' cumulative clocks now: the CPU's ms and experts, the lane's copy ms and bytes
    void mark(double ms_run, int64_t jobs, double lane_ms, int64_t lane_bytes) {
        mark_ms = ms_run;
        mark_jobs = jobs;
        mark_lane_ms = lane_ms;
        mark_lane_bytes = lane_bytes;
    }
    // the new share, or -1 while there is nothing to change; rd gets the attempt (reason null: none this round)
    double window(double ms_run, int64_t jobs, double lane_ms, int64_t lane_bytes, double cur, split_reading & rd);
};

// the share's 1/256 units and the window's GPU misses: the last (n_miss x num) >> 8 misses in routing order
int32_t share_num(double share);
inline int32_t gpu_misses(int32_t n_miss, int32_t num) {
    return (n_miss * num) >> 8;
}

} // namespace fn::runtime::policy
