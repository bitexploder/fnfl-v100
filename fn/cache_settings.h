#pragma once

// ME3: the expert cache's and the host K/V tier's settings, one table for both entry points: fnfl-drive spells a
// setting --ex-<name> (K/V: --kv-<name>), the server --fnfl-expert-<name> (--fnfl-kv-<name>). Each has a default, a
// unit and a legal range; parse refuses anything outside it; readback is the startup line and json the per-request
// provenance (each setting's value and whether it was given). Header only and free of CUDA, ggml and llama types:
// fn/fnfl.h includes it, so common/arg.cpp, the glue and the driver share the definitions. The values reach the
// engine as fn::runtime::expert_cache_options (residency, split, threads, the prompt's loan and refill), the prefiller
// (short reads, the chunk) and the load (the K/V window); a "derived" default is computed at init and read back there.
// The settings that existed before ME3 (--ex-pool-mib, --ex-reserve-mib, --ex-profile, --ex-profile-save,
// --ex-swap-overlap, --ex-cpu, --kv-host, --device-cap-mib, --single-card, --cached-tp, --spec-d) keep their own
// parsing and spellings. Every setting here names ME3; at ME3's disposition the ones not independently useful are
// removed (fnfl/STANDARDS.md "Switches").

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace fn::settings {

struct entry {
    const char * name;  // as in the request JSON
    const char * drive; // fnfl-drive's spelling
    const char * serve; // the server's
    const char * metavar;
    const char * unit;
    const char * def;   // the default as text ("derived": computed at init and read back)
    const char * range;
    const char * help;
    int32_t scope;      // 0: any engine; 1: the expert cache's (needs the cache); 2: the host K/V tier's (--kv-host)
};

// clang-format off
constexpr entry table[] = {
    // residency (adapt_options, Strata's starting policy; not swept)
    {"swap-every", "--ex-swap-every", "--fnfl-expert-swap-every", "N", "rounds", "4", "1..1000000",
     "rounds between swap choices", 1},
    {"swap-cap", "--ex-swap-cap", "--fnfl-expert-swap-cap", "N", "swaps", "96", "0..4096",
     "swaps a choice admits at most (a binding cap is a diagnostic)", 1},
    {"swap-min-gain", "--ex-swap-min-gain", "--fnfl-expert-swap-min-gain", "X", "uses", "1.5", "0..1000000",
     "a swap's usage gain over its victim, additive", 1},
    {"swap-min-use", "--ex-swap-min-use", "--fnfl-expert-swap-min-use", "X", "uses", "2.0", "0..1000000",
     "a candidate's usage at least", 1},
    {"swap-decay", "--ex-swap-decay", "--fnfl-expert-swap-decay", "X", "factor", "0.7", "0..1",
     "usage decay after each choice", 1},
    // the miss split (pcie_auto)
    {"split", "--ex-split", "--fnfl-expert-split", "F", "share", "0.30", "0..1",
     "the misses' PCIe share at start", 1},
    {"split-fixed", "--ex-split-fixed", "--fnfl-expert-split-fixed", "F", "share", "off", "off or 0..1",
     "a fixed share, refinement off", 1},
    {"split-auto", "--ex-split-auto", "--fnfl-expert-split-auto", "0|1", "switch", "1", "0..1",
     "the share's refinement", 1},
    {"split-windows", "--ex-split-windows", "--fnfl-expert-split-windows", "LIST", "rounds", "48,96,96",
     "1..8 entries of 1..100000", "rounds before each refinement; the last repeats for the whole run (ME5 rider)", 1},
    {"split-min-cpu", "--ex-split-min-cpu", "--fnfl-expert-split-min-cpu", "N", "CPU experts", "300", "0..1000000000",
     "CPU experts since the last update before another is taken", 1},
    {"split-off-after", "--ex-split-off-after", "--fnfl-expert-split-off-after", "N", "rounds", "2000",
     "1..1000000000", "refinement off after this many deferred rounds", 1},
    {"split-round", "--ex-split-round", "--fnfl-expert-split-round", "X", "share", "0.05", "0.001..0.5",
     "the new share rounded to a multiple of this", 1},
    {"split-min-change", "--ex-split-min-change", "--fnfl-expert-split-min-change", "X", "share", "0.0375", "0..1",
     "the unrounded reading within this of the share keeps it (ME5 rider)", 1},
    {"split-clamp", "--ex-split-clamp", "--fnfl-expert-split-clamp", "LO,HI", "share", "0.10,0.60",
     "0 <= LO <= HI <= 1", "the refined share's bounds", 1},
    {"split-k", "--ex-split-k", "--fnfl-expert-split-k", "K", "factor", "1", "0.01..100",
     "k in L / (L + k R)", 1},
    // threads
    {"worker-cpus", "--ex-worker-cpus", "--fnfl-expert-worker-cpus", "LIST", "CPUs", "derived",
     "allowed CPUs, each once", "the CPU product workers' CPUs, one a worker (default: each core's first CPU but "
     "core 0's)", 1},
    {"service-cpu", "--ex-service-cpu", "--fnfl-expert-service-cpu", "N", "CPU", "derived", "-1 or an allowed CPU",
     "the draining service thread's CPU (default, one card: core 0's first CPU)", 1},
    {"decode-cpu", "--ex-decode-cpu", "--fnfl-expert-decode-cpu", "N", "CPU", "derived", "-1 or an allowed CPU",
     "the decode thread's CPU (default, one card: core 0's second CPU)", 1},
    {"helper-cpus", "--ex-helper-cpus", "--fnfl-expert-helper-cpus", "LIST", "CPUs", "derived",
     "allowed CPUs, each once", "the helpers' mask (the prefill gather; default: the original allowed mask less "
     "the workers', service and decode CPUs, else less the service and decode CPUs)", 1},
    {"park-ms", "--ex-park-ms", "--fnfl-expert-park-ms", "N", "ms", "20", "0..10000",
     "the workers' spin before they sleep", 1},
    // the prompt
    {"short-read", "--ex-short-read", "--fnfl-expert-short-read", "N", "tokens", "64", "0..8192",
     "a prompt part of at most N tokens reads through the target windows, without a loan (0: always bulk)", 1},
    {"routed-rows", "--ex-routed-rows", "--fnfl-expert-routed-rows", "N", "rows", "256", "0..4096",
     "a prompt half of at most N rows stages only its routed experts", 1},
    {"prompt-chunk", "--ex-prompt-chunk", "--fnfl-expert-prompt-chunk", "N", "rows", "8192",
     "256..8192, a multiple of 256", "the bulk prompt's largest chunk", 1},
    {"lend-pct", "--ex-lend-pct", "--fnfl-expert-lend-pct", "N", "% of slots", "85", "1..100",
     "the largest loan (the prompt zone and the lent region) as a share of the pool's slots", 1},
    {"min-keep", "--ex-min-keep", "--fnfl-expert-min-keep", "N", "slots", "128", "0..1000000",
     "slots never lent", 1},
    {"lend-region", "--ex-lend-region", "--fnfl-expert-lend-region", "0|1", "switch", "1", "0..1",
     "one card: PB1's arena and PF1's staging lent to the pool between prompts", 1},
    {"refill-overlap", "--ex-refill-overlap", "--fnfl-expert-refill-overlap", "0|1", "switch", "1", "0..1",
     "the refill's copies overlap decode (0: copied and published before the first decode round)", 1},
    {"refill-batch-layers", "--ex-refill-batch-layers", "--fnfl-expert-refill-batch-layers", "N", "layers", "4",
     "1..48", "layers a refill submission copies at most at a time (MX3: and at most 64 MiB a device)", 1},
    // ME4: a one-half prompt run's experts through a bounded ring (fn/runtime/prompt_plan.h)
    {"ring-mib", "--ex-ring-mib", "--fnfl-expert-ring-mib", "N", "MiB", "256", "0 or 2 x group-mib..4096",
     "one-half prompt runs stage their experts through a ring of N MiB at the zone's coldest end, in expert-ID "
     "groups (0: ME3's two layer buffers and routed-rows)", 1},
    {"group-mib", "--ex-group-mib", "--fnfl-expert-group-mib", "N", "MiB", "64", "4..1024",
     "a ring group's copied experts at most", 1},
    {"route-cover", "--ex-route-cover", "--fnfl-expert-route-cover", "N", "per mille", "875", "0..1000",
     "a ring chunk streams its later layers ahead once a layer's routes cover this much of its nonresident bytes "
     "(0: from its second layer; 1000: only full cover)", 1},
    // MX7: the prompt CPU share (fn/runtime/prompt_share.h; one device: the single-card cached configuration)
    {"prompt-cpu-share", "--ex-prompt-cpu-share", "--fnfl-prompt-cpu-share", "auto|X", "share", "auto",
     "auto, 0 or 0..1",
     "on one device, a route-first ring layer's CPU takes this share of its routed nonresident experts of at most 8 "
     "pairs, fewest first (auto: from the CPU's and the GPU's measured cost an expert, self-checked; 0: off; two "
     "cards: off)", 1},
    {"prompt-cpu-share-max", "--ex-prompt-cpu-share-max", "--fnfl-prompt-cpu-share-max", "N", "rows", "1024",
     "1024..4096", "the CPU share serves prompt chunks of fewer rows", 1},
    // the executor
    {"executor", "--ex-executor", "--fnfl-expert-executor", "graph|eager", "mode", "graph", "graph or eager",
     "graph replays the captured target, draft and catch-up graphs; eager enqueues each round (a diagnostic)", 0},
    // the host K/V tier
    {"kv-window", "--kv-window", "--fnfl-kv-window", "CELLS", "cells", "32768", "1024..262144, a multiple of 1024",
     "the decode window of each attention layer", 2},
    {"kv-gather", "--kv-gather", "--fnfl-kv-gather", "CELLS", "cells", "10240", "10240 (the kernels' size)",
     "the gather area", 2},
};
// clang-format on
constexpr int32_t n_entries = int32_t(sizeof(table) / sizeof(table[0]));
static_assert(n_entries <= 64, "values::given holds a bit a setting");

struct values {
    int32_t swap_every = 4, swap_cap = 96;
    double swap_min_gain = 1.5, swap_min_use = 2.0, swap_decay = 0.7;
    double split = 0.30, split_fixed = -1.0; // -1: off
    int32_t split_auto = 1;
    std::vector<int32_t> split_windows = {48, 96, 96};
    int32_t split_min_cpu = 300, split_off_after = 2000;
    double split_round = 0.05, split_min_change = 0.0375, split_lo = 0.10, split_hi = 0.60, split_k = 1.0;
    std::vector<int32_t> worker_cpus, helper_cpus; // empty: derived
    int32_t service_cpu = -1, decode_cpu = -1;     // -1: derived
    int32_t park_ms = 20;
    int32_t short_read = 64, routed_rows = 256, prompt_chunk = 8192, lend_pct = 85, min_keep = 128;
    int32_t lend_region = 1, refill_overlap = 1, refill_batch_layers = 4; // MX3: 4 layers a batch (was 1)
    int32_t ring_mib = 256, group_mib = 64, route_cover = 875;            // ME4
    double prompt_cpu_share = -1.0;                                       // MX7: -1 auto
    int32_t prompt_cpu_share_max = 1024;
    std::string executor = "graph";
    int32_t kv_window = 32768, kv_gather = 10240;
    uint64_t given = 0; // bit i: table[i] was given
    // a setting of the scope given
    bool any(int32_t scope) const {
        for (int32_t i = 0; i < n_entries; ++i) {
            if ((given >> i & 1) != 0 && table[i].scope == scope) {
                return true;
            }
        }
        return false;
    }
};

// the entry spelled flag (either entry point's spelling), -1 none
inline int32_t find(const char * flag) {
    const std::string f = flag != nullptr ? flag : "";
    for (int32_t i = 0; i < n_entries; ++i) {
        if (f == table[i].drive || f == table[i].serve) {
            return i;
        }
    }
    return -1;
}

namespace detail {

inline bool to_int(const std::string & s, int64_t lo, int64_t hi, int64_t & out) {
    if (s.empty()) {
        return false;
    }
    char * end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || v < lo || v > hi) {
        return false;
    }
    out = v;
    return true;
}

inline bool to_real(const std::string & s, double lo, double hi, double & out) {
    if (s.empty()) {
        return false;
    }
    char * end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == nullptr || *end != '\0' || !(v >= lo && v <= hi)) {
        return false;
    }
    out = v;
    return true;
}

inline std::vector<std::string> split_list(const std::string & s) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at <= s.size()) {
        const size_t c = s.find(',', at);
        out.push_back(s.substr(at, c == std::string::npos ? std::string::npos : c - at));
        if (c == std::string::npos) {
            break;
        }
        at = c + 1;
    }
    return out;
}

inline bool to_ints(const std::string & s, size_t n_lo, size_t n_hi, int64_t lo, int64_t hi, bool unique,
                    std::vector<int32_t> & out) {
    std::vector<int32_t> v;
    for (const std::string & p : split_list(s)) {
        int64_t x = 0;
        if (!to_int(p, lo, hi, x)) {
            return false;
        }
        for (const int32_t y : v) {
            if (unique && y == x) {
                return false;
            }
        }
        v.push_back(int32_t(x));
    }
    if (v.size() < n_lo || v.size() > n_hi) {
        return false;
    }
    out = v;
    return true;
}

inline std::string ints_text(const std::vector<int32_t> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s;
}

inline std::string real_text(double x) {
    char b[32];
    std::snprintf(b, sizeof b, "%.6g", x);
    return b;
}

} // namespace detail

// table[i] set from text, its range checked; false with the reason
inline bool parse(int32_t i, const std::string & text, values & v, std::string & err) {
    using namespace detail;
    if (i < 0 || i >= n_entries) {
        err = "no such setting";
        return false;
    }
    const std::string n = table[i].name;
    int64_t k = 0;
    bool ok = false;
    if (n == "swap-every") {
        ok = to_int(text, 1, 1000000, k) && ((v.swap_every = int32_t(k)), true);
    } else if (n == "swap-cap") {
        ok = to_int(text, 0, 4096, k) && ((v.swap_cap = int32_t(k)), true);
    } else if (n == "swap-min-gain") {
        ok = to_real(text, 0, 1e6, v.swap_min_gain);
    } else if (n == "swap-min-use") {
        ok = to_real(text, 0, 1e6, v.swap_min_use);
    } else if (n == "swap-decay") {
        ok = to_real(text, 0, 1, v.swap_decay);
    } else if (n == "split") {
        ok = to_real(text, 0, 1, v.split);
    } else if (n == "split-fixed") {
        ok = text == "off" ? ((v.split_fixed = -1.0), true) : to_real(text, 0, 1, v.split_fixed);
    } else if (n == "split-auto") {
        ok = to_int(text, 0, 1, k) && ((v.split_auto = int32_t(k)), true);
    } else if (n == "split-windows") {
        ok = to_ints(text, 1, 8, 1, 100000, false, v.split_windows);
    } else if (n == "split-min-cpu") {
        ok = to_int(text, 0, 1000000000, k) && ((v.split_min_cpu = int32_t(k)), true);
    } else if (n == "split-off-after") {
        ok = to_int(text, 1, 1000000000, k) && ((v.split_off_after = int32_t(k)), true);
    } else if (n == "split-round") {
        ok = to_real(text, 0.001, 0.5, v.split_round);
    } else if (n == "split-min-change") {
        ok = to_real(text, 0, 1, v.split_min_change);
    } else if (n == "split-clamp") {
        const std::vector<std::string> p = split_list(text);
        double lo = 0, hi = 0;
        ok = p.size() == 2 && to_real(p[0], 0, 1, lo) && to_real(p[1], 0, 1, hi) && lo <= hi;
        if (ok) {
            v.split_lo = lo;
            v.split_hi = hi;
        }
    } else if (n == "split-k") {
        ok = to_real(text, 0.01, 100, v.split_k);
    } else if (n == "worker-cpus") {
        ok = to_ints(text, 1, 1024, 0, 1023, true, v.worker_cpus);
    } else if (n == "service-cpu") {
        ok = to_int(text, -1, 1023, k) && ((v.service_cpu = int32_t(k)), true);
    } else if (n == "decode-cpu") {
        ok = to_int(text, -1, 1023, k) && ((v.decode_cpu = int32_t(k)), true);
    } else if (n == "helper-cpus") {
        ok = to_ints(text, 1, 1024, 0, 1023, true, v.helper_cpus);
    } else if (n == "park-ms") {
        ok = to_int(text, 0, 10000, k) && ((v.park_ms = int32_t(k)), true);
    } else if (n == "short-read") {
        ok = to_int(text, 0, 8192, k) && ((v.short_read = int32_t(k)), true);
    } else if (n == "routed-rows") {
        ok = to_int(text, 0, 4096, k) && ((v.routed_rows = int32_t(k)), true);
    } else if (n == "prompt-chunk") {
        ok = to_int(text, 256, 8192, k) && k % 256 == 0 && ((v.prompt_chunk = int32_t(k)), true);
    } else if (n == "lend-pct") {
        ok = to_int(text, 1, 100, k) && ((v.lend_pct = int32_t(k)), true);
    } else if (n == "min-keep") {
        ok = to_int(text, 0, 1000000, k) && ((v.min_keep = int32_t(k)), true);
    } else if (n == "lend-region") {
        ok = to_int(text, 0, 1, k) && ((v.lend_region = int32_t(k)), true);
    } else if (n == "refill-overlap") {
        ok = to_int(text, 0, 1, k) && ((v.refill_overlap = int32_t(k)), true);
    } else if (n == "refill-batch-layers") {
        ok = to_int(text, 1, 48, k) && ((v.refill_batch_layers = int32_t(k)), true);
    } else if (n == "ring-mib") {
        ok = to_int(text, 0, 4096, k) && ((v.ring_mib = int32_t(k)), true);
    } else if (n == "group-mib") {
        ok = to_int(text, 4, 1024, k) && ((v.group_mib = int32_t(k)), true);
    } else if (n == "route-cover") {
        ok = to_int(text, 0, 1000, k) && ((v.route_cover = int32_t(k)), true);
    } else if (n == "prompt-cpu-share") {
        ok = text == "auto" ? ((v.prompt_cpu_share = -1.0), true) : to_real(text, 0, 1, v.prompt_cpu_share);
    } else if (n == "prompt-cpu-share-max") {
        ok = to_int(text, 1024, 4096, k) && ((v.prompt_cpu_share_max = int32_t(k)), true);
    } else if (n == "executor") {
        ok = text == "graph" || text == "eager";
        if (ok) {
            v.executor = text;
        }
    } else if (n == "kv-window") {
        ok = to_int(text, 1024, 262144, k) && k % 1024 == 0 && ((v.kv_window = int32_t(k)), true);
    } else if (n == "kv-gather") {
        ok = to_int(text, 10240, 10240, k) && ((v.kv_gather = int32_t(k)), true);
    }
    if (!ok) {
        err = std::string(table[i].drive) + " (" + table[i].serve + "): " + table[i].metavar + " is " +
              table[i].range + ", not '" + text + "'";
        return false;
    }
    v.given |= uint64_t(1) << i;
    return true;
}

// table[i]'s value as text
inline std::string text(const values & v, int32_t i) {
    using namespace detail;
    const std::string n = i >= 0 && i < n_entries ? table[i].name : "";
    if (n == "swap-every") {
        return std::to_string(v.swap_every);
    }
    if (n == "swap-cap") {
        return std::to_string(v.swap_cap);
    }
    if (n == "swap-min-gain") {
        return real_text(v.swap_min_gain);
    }
    if (n == "swap-min-use") {
        return real_text(v.swap_min_use);
    }
    if (n == "swap-decay") {
        return real_text(v.swap_decay);
    }
    if (n == "split") {
        return real_text(v.split);
    }
    if (n == "split-fixed") {
        return v.split_fixed < 0 ? "off" : real_text(v.split_fixed);
    }
    if (n == "split-auto") {
        return std::to_string(v.split_auto);
    }
    if (n == "split-windows") {
        return ints_text(v.split_windows);
    }
    if (n == "split-min-cpu") {
        return std::to_string(v.split_min_cpu);
    }
    if (n == "split-off-after") {
        return std::to_string(v.split_off_after);
    }
    if (n == "split-round") {
        return real_text(v.split_round);
    }
    if (n == "split-min-change") {
        return real_text(v.split_min_change);
    }
    if (n == "split-clamp") {
        return real_text(v.split_lo) + "," + real_text(v.split_hi);
    }
    if (n == "split-k") {
        return real_text(v.split_k);
    }
    if (n == "worker-cpus") {
        return v.worker_cpus.empty() ? "derived" : ints_text(v.worker_cpus);
    }
    if (n == "service-cpu") {
        return v.service_cpu < 0 ? "derived" : std::to_string(v.service_cpu);
    }
    if (n == "decode-cpu") {
        return v.decode_cpu < 0 ? "derived" : std::to_string(v.decode_cpu);
    }
    if (n == "helper-cpus") {
        return v.helper_cpus.empty() ? "derived" : ints_text(v.helper_cpus);
    }
    if (n == "park-ms") {
        return std::to_string(v.park_ms);
    }
    if (n == "short-read") {
        return std::to_string(v.short_read);
    }
    if (n == "routed-rows") {
        return std::to_string(v.routed_rows);
    }
    if (n == "prompt-chunk") {
        return std::to_string(v.prompt_chunk);
    }
    if (n == "lend-pct") {
        return std::to_string(v.lend_pct);
    }
    if (n == "min-keep") {
        return std::to_string(v.min_keep);
    }
    if (n == "lend-region") {
        return std::to_string(v.lend_region);
    }
    if (n == "refill-overlap") {
        return std::to_string(v.refill_overlap);
    }
    if (n == "refill-batch-layers") {
        return std::to_string(v.refill_batch_layers);
    }
    if (n == "ring-mib") {
        return std::to_string(v.ring_mib);
    }
    if (n == "group-mib") {
        return std::to_string(v.group_mib);
    }
    if (n == "route-cover") {
        return std::to_string(v.route_cover);
    }
    if (n == "prompt-cpu-share") {
        return v.prompt_cpu_share < 0 ? "auto" : real_text(v.prompt_cpu_share);
    }
    if (n == "prompt-cpu-share-max") {
        return std::to_string(v.prompt_cpu_share_max);
    }
    if (n == "executor") {
        return v.executor;
    }
    if (n == "kv-window") {
        return std::to_string(v.kv_window);
    }
    if (n == "kv-gather") {
        return std::to_string(v.kv_gather);
    }
    return "?";
}

// the startup read-back: every setting of the scope as name=value, given ones marked (*)
inline std::string readback(const values & v, int32_t scope) {
    std::string s;
    for (int32_t i = 0; i < n_entries; ++i) {
        if (table[i].scope != scope) {
            continue;
        }
        s += (s.empty() ? "" : " ") + std::string(table[i].name) + "=" + text(v, i) +
             ((v.given >> i & 1) != 0 ? "*" : "");
    }
    return s;
}

// the request JSON's provenance: {"name": {"value": "...", "given": true|false}, ...}, every scope
inline std::string json(const values & v) {
    std::string s = "{";
    for (int32_t i = 0; i < n_entries; ++i) {
        s += std::string(i ? ", " : "") + "\"" + table[i].name + "\": {\"value\": \"" + text(v, i) +
             "\", \"given\": " + ((v.given >> i & 1) != 0 ? "true" : "false") + "}";
    }
    return s + "}";
}

} // namespace fn::settings
