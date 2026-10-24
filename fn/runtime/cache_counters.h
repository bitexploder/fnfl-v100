#pragma once

// The expert cache's counters (expert_cache.h). ME3 (the cleanup): each unit keeps its own instance and writes only its
// own fields (the residency its passes and swap waits, the loan its prompts, loans and refill, the window service its
// windows, misses, CPU work and GPU waits, the cache its rounds and the executor's); the cache's snapshot is their
// sum, read between requests.

#include "fn/runtime/expert_policy.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime {

// the window counts of one kind of exposed wait
struct wait_count {
    int64_t n = 0;
    double s = 0;
};

struct expert_cache_counters {
    // decode windows (one a target layer a round): experts looked up and how each was served
    int64_t windows = 0, lookups = 0, hits = 0, gpu_miss = 0, cpu_miss = 0;
    int64_t hit_bytes = 0, gpu_miss_bytes = 0, cpu_miss_bytes = 0; // entry bytes (both shards)
    int64_t cpu_windows = 0, rounds = 0;
    int64_t device_windows = 0; // windows every expert of which was resident: served without the host
    int64_t cpu_rows[6] = {};   // CPU experts by rows (1..5)
    int64_t gpu_rows[6] = {};   // ME5: lane (GPU-served) misses by rows (1..5); row incidences: sum of rows x count
    // the four costs (ME2 amendment 1, item 3), never summed:
    //   CPU service: the pool's wall time (gate/up, the down activation, down) and the expert bytes it read
    double cpu_gu_ms = 0, cpu_act_ms = 0, cpu_dn_ms = 0;
    int64_t cpu_bytes = 0;
    //   ME2b (diagnostic; zero and reported disabled unless the pool's unit clocks are on, never in serving): thread-ms
    //   of the un-permutes and the products, gate/up and down
    double cpu_gu_unpack_ms = 0, cpu_gu_compute_ms = 0, cpu_dn_unpack_ms = 0, cpu_dn_compute_ms = 0;
    //   the GPU's waits on the CPU (a window with CPU experts: its compute stream at the down flag, timing events
    //   around the stream wait; device 0)
    wait_count cpu_wait;
    double cpu_q8_ms = 0; // the service's gate/up activation (dequantize, Q8_K) for windows with CPU experts
    //   H2D: bytes and copy-stream time of the lane misses (copy stream), the swaps and the refills (swap stream);
    //   the time is device 0's. ME9: lane_bytes is every GPU miss's copy whatever its destination (the scratch lane
    //   or a reserved slot), the bytes of lane_copy_ms's interval; adm_bytes is the reserved slots' subset
    int64_t lane_bytes = 0, swap_bytes = 0, refill_bytes = 0;
    double lane_copy_ms = 0, swap_copy_ms = 0, refill_copy_ms = 0;
    int64_t swaps = 0, passes = 0, refills = 0;
    //   exposed waits by kind: the GPU at the gate/up flag in a window with lane misses (the lane copies), the GPU
    //   waiting for a layer's swap or refill copies (its own events), both device 0's; the host publishing a pass
    //   (overlap off: with the copies); the service waiting for a window's event (not a GPU wait)
    wait_count lane_wait, swap_wait, publish_wait, desc_wait;
    // prefill: layers staged (of them by their routing), bytes copied (copy stream) and bytes served from resident
    // cache entries, host waits for a staging buffer's table or a layer's routing, zone evictions
    int64_t prompt_layers = 0, prompt_routed_layers = 0, prompt_staged_bytes = 0, prompt_resident_bytes = 0;
    int64_t zone_evictions = 0;
    double prompt_wait_s = 0;
    int64_t lent_holds = 0; // ME2b: borrows of the lent region (prompt_hold)
    // ME3, the graph executor: windows recorded in captures (card 0's), replays served (pass_posted)
    int64_t graph_windows_captured = 0, graph_replays = 0;
    // ME3, the prompt's loan: loans taken, zone slots borrowed and their bytes (a device's), of them the slots that
    // held a resident and those residents' payload bytes (both shards: what the refill copies back)
    int64_t loans = 0, borrowed_slots = 0, borrowed_bytes = 0, borrowed_residents = 0, borrowed_resident_bytes = 0;
    // ME3, the bounded refill (refills counts the returns): submission batches, the service's host time submitting
    // them, moves skipped at publication (their key resident again), keys carried back to the zone by a later prompt's
    // loan (not yet submitted); the lane waits of windows served while a refill was outstanding (also counted in
    // lane_wait)
    int64_t refill_batches = 0, refill_skipped = 0, refill_carried = 0;
    double refill_submit_s = 0;
    wait_count lane_wait_refill;
    // ME3: windows of known-token short reads (also counted in windows) and their rows
    int64_t short_windows = 0, short_rows = 0;
    // ME4, the ring: loans staging through it, layers staged route-first and streamed ahead, groups, and the route-first
    // layers' cover (routed nonresident over nonresident bytes, per mille, summed: their mean is sum / layers)
    int64_t ring_loans = 0, ring_routed_layers = 0, ring_ahead_layers = 0, ring_groups = 0, ring_cover_pm = 0;
    // MX3: the ring loans' bytes (a device's, summed)
    int64_t ring_loan_bytes = 0;
    // MX7, the prompt CPU share (prompt_share.h): ring chunks and layers it served, eligible layers the self-check (or
    // a short pick) left to the GPU, its experts, pairs and gathered rows, the expert bytes not staged, its turn-off
    // (the host refused pinned memory), the CPU's ms (its thread's convert, pool and F16 pass) and the compute thread's
    // wait
    int64_t pshare_chunks = 0, pshare_layers = 0, pshare_off_layers = 0, pshare_experts = 0, pshare_pairs = 0,
            pshare_rows = 0, pshare_unstaged_bytes = 0, pshare_dead = 0;
    double pshare_cpu_ms = 0, pshare_join_ms = 0;
    // ME4, a two-half run's parity buffers: loans placed by usage (parity_tried), and of those, placed away from the
    // zone's start (parity_moved; the rest found no interval outside the spans and kept ME3's place)
    int64_t parity_tried = 0, parity_moved = 0;
    // ME9, admission (fn/runtime/admission.h): residents retired into reservations and orphaned slots adopted; misses
    // admitted (assigned a reserved slot), their copies started (copy and event enqueued) and their bytes; records
    // landed, published, invalidated (a failure, an abandoned pass, released waits) and published-over (the key or slot
    // taken meanwhile: the slot stays reserved); later hits on admitted keys and misses on keys retired for a
    // reservation; rounds admission was off, by why (mode or devices: never on; then loan, refill, pending pass, a kept
    // failure)
    int64_t adm_retired = 0, adm_adopted = 0, adm_assigned = 0, adm_started = 0, adm_bytes = 0;
    int64_t adm_landed = 0, adm_published = 0, adm_invalidated = 0, adm_collisions = 0;
    int64_t adm_hits = 0, adm_lost = 0;
    int64_t adm_rounds_on = 0, adm_off_loan = 0, adm_off_refill = 0, adm_off_pass = 0, adm_off_failed = 0,
            adm_off_other = 0;
};

// ME3: a - b (sign -1) or a + b (sign 1) field by field (a request's counters from the run's; the units' sum)
expert_cache_counters counters_add(const expert_cache_counters & a, const expert_cache_counters & b, int32_t sign);
inline expert_cache_counters counters_delta(const expert_cache_counters & a, const expert_cache_counters & b) {
    return counters_add(a, b, -1);
}

// one attempted refinement of the miss share (PcieAuto), at a round, changed or not (ME3)
struct share_attempt {
    int64_t round = 0;
    policy::split_reading r;
};

// ME9 (Shell 0a1b4eb5): the counters record's last members and its closing brace: "admission" (the adm_ fields and
// the slots reserved now), then "share_attempts", the log's entries from `from` (an array, possibly empty). A
// non-finite reading is written null, a missing reason "". expert_cache::counters_json ends with it; ex-cpu-test
// parses it as JSON with an empty and a populated log
std::string counters_tail_json(int32_t reserved_now, const expert_cache_counters & c,
                               const std::vector<share_attempt> & log, size_t from);

} // namespace fn::runtime
