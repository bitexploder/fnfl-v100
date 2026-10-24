// ME3 (the cleanup): the expert cache's configuration and reporting (expert_cache.h): the settings into the options,
// the CPUs' masks, the counters' sums and JSON, the request line and the memory plan, read from the units' read-only
// accessors between requests. Moved here from expert_cache.cpp.

#include "fn/runtime/expert_cache.h"

#include "fn/runtime/cache_parts.h"

#include <sched.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <utility>

namespace fn::runtime {

using cache::fail;
using cache::gib;
using cache::n_ex;

std::vector<std::vector<int32_t>> physical_cores() {
    cpu_set_t set;
    CPU_ZERO(&set);
    std::vector<std::vector<int32_t>> out;
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return out;
    }
    std::map<std::pair<int32_t, int32_t>, size_t> at;
    for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) {
            continue;
        }
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        int32_t pkg = 0, core = cpu;
        std::ifstream(base + "physical_package_id") >> pkg;
        std::ifstream(base + "core_id") >> core;
        const auto key = std::make_pair(pkg, core);
        const auto it = at.find(key);
        if (it == at.end()) {
            at[key] = out.size();
            out.push_back({cpu});
        } else {
            out[it->second].push_back(cpu);
        }
    }
    return out;
}

bool cpus_allowed(const std::vector<int32_t> & allowed, const expert_cache_options & o, std::string & err) {
    const auto in_allowed = [&](int32_t cpu) {
        return std::find(allowed.begin(), allowed.end(), cpu) != allowed.end();
    };
    for (const int32_t cpu : o.worker_cpus) {
        if (!in_allowed(cpu)) {
            return fail(err, "worker CPU " + std::to_string(cpu) + " is outside the process's allowed mask");
        }
    }
    for (const int32_t cpu : {o.service_cpu, o.decode_cpu}) {
        if (cpu >= 0 && !in_allowed(cpu)) {
            return fail(err, "service or decode CPU " + std::to_string(cpu) + " is outside the allowed mask");
        }
    }
    for (const int32_t cpu : o.helper_cpus) {
        if (!in_allowed(cpu)) {
            return fail(err, "helper CPU " + std::to_string(cpu) + " is outside the process's allowed mask");
        }
    }
    return true;
}

std::vector<int32_t> helper_mask(const std::vector<int32_t> & allowed, const std::vector<int32_t> & workers,
                                 int32_t service_cpu, int32_t decode_cpu) {
    std::vector<int32_t> out;
    for (int32_t pass = 0; pass < 3 && out.empty(); ++pass) {
        for (const int32_t cpu : allowed) {
            const bool worker = std::find(workers.begin(), workers.end(), cpu) != workers.end();
            if ((pass == 0 && worker) || (pass < 2 && (cpu == service_cpu || cpu == decode_cpu))) {
                continue;
            }
            out.push_back(cpu);
        }
    }
    return out;
}

void apply_settings(const fn::settings::values & v, expert_cache_options & o) {
    o.adapt.every = v.swap_every;
    o.adapt.max_swaps = v.swap_cap;
    o.adapt.min_gain = float(v.swap_min_gain);
    o.adapt.min_use = float(v.swap_min_use);
    o.adapt.decay = float(v.swap_decay);
    o.share0 = v.split_fixed >= 0 ? v.split_fixed : v.split;
    o.refine = v.split_fixed < 0 && v.split_auto != 0;
    o.split.windows = v.split_windows;
    o.split.min_cpu = v.split_min_cpu;
    o.split.off_after = v.split_off_after;
    o.split.round = v.split_round;
    o.split.min_change = v.split_min_change;
    o.split.lo = v.split_lo;
    o.split.hi = v.split_hi;
    o.split.k = v.split_k;
    o.worker_cpus = v.worker_cpus;
    o.helper_cpus = v.helper_cpus;
    o.service_cpu = v.service_cpu;
    o.decode_cpu = v.decode_cpu;
    o.park_ms = v.park_ms;
    o.routed_rows = v.routed_rows;
    o.lend_pct = v.lend_pct;
    o.min_keep = v.min_keep;
    o.lend_region = v.lend_region != 0;
    o.refill_overlap = v.refill_overlap != 0;
    o.refill_batch_layers = v.refill_batch_layers;
    o.ring_bytes = int64_t(v.ring_mib) << 20;
    o.group_bytes = int64_t(v.group_mib) << 20;
    o.route_cover = v.route_cover;
    o.prompt_cpu_share = v.prompt_cpu_share; // MX7
    o.prompt_cpu_share_max = v.prompt_cpu_share_max;
}

expert_cache_counters counters_add(const expert_cache_counters & a, const expert_cache_counters & b, int32_t sign) {
    expert_cache_counters d = a;
    const auto w = [sign](wait_count & x, const wait_count & y) {
        x.n += sign * y.n;
        x.s += sign * y.s;
    };
    d.windows += sign * b.windows;
    d.lookups += sign * b.lookups;
    d.hits += sign * b.hits;
    d.gpu_miss += sign * b.gpu_miss;
    d.cpu_miss += sign * b.cpu_miss;
    d.hit_bytes += sign * b.hit_bytes;
    d.gpu_miss_bytes += sign * b.gpu_miss_bytes;
    d.cpu_miss_bytes += sign * b.cpu_miss_bytes;
    d.cpu_windows += sign * b.cpu_windows;
    d.rounds += sign * b.rounds;
    d.device_windows += sign * b.device_windows;
    for (int32_t i = 0; i < 6; ++i) {
        d.cpu_rows[i] += sign * b.cpu_rows[i];
        d.gpu_rows[i] += sign * b.gpu_rows[i];
    }
    d.cpu_gu_ms += sign * b.cpu_gu_ms;
    d.cpu_act_ms += sign * b.cpu_act_ms;
    d.cpu_dn_ms += sign * b.cpu_dn_ms;
    d.cpu_bytes += sign * b.cpu_bytes;
    d.cpu_gu_unpack_ms += sign * b.cpu_gu_unpack_ms;
    d.cpu_gu_compute_ms += sign * b.cpu_gu_compute_ms;
    d.cpu_dn_unpack_ms += sign * b.cpu_dn_unpack_ms;
    d.cpu_dn_compute_ms += sign * b.cpu_dn_compute_ms;
    w(d.cpu_wait, b.cpu_wait);
    d.cpu_q8_ms += sign * b.cpu_q8_ms;
    d.lane_bytes += sign * b.lane_bytes;
    d.swap_bytes += sign * b.swap_bytes;
    d.refill_bytes += sign * b.refill_bytes;
    d.lane_copy_ms += sign * b.lane_copy_ms;
    d.swap_copy_ms += sign * b.swap_copy_ms;
    d.refill_copy_ms += sign * b.refill_copy_ms;
    d.swaps += sign * b.swaps;
    d.passes += sign * b.passes;
    d.refills += sign * b.refills;
    w(d.lane_wait, b.lane_wait);
    w(d.swap_wait, b.swap_wait);
    w(d.publish_wait, b.publish_wait);
    w(d.desc_wait, b.desc_wait);
    d.ring_loans += sign * b.ring_loans;
    d.ring_routed_layers += sign * b.ring_routed_layers;
    d.ring_ahead_layers += sign * b.ring_ahead_layers;
    d.ring_groups += sign * b.ring_groups;
    d.ring_cover_pm += sign * b.ring_cover_pm;
    d.ring_loan_bytes += sign * b.ring_loan_bytes;
    d.pshare_chunks += sign * b.pshare_chunks;
    d.pshare_layers += sign * b.pshare_layers;
    d.pshare_off_layers += sign * b.pshare_off_layers;
    d.pshare_experts += sign * b.pshare_experts;
    d.pshare_pairs += sign * b.pshare_pairs;
    d.pshare_rows += sign * b.pshare_rows;
    d.pshare_unstaged_bytes += sign * b.pshare_unstaged_bytes;
    d.pshare_dead += sign * b.pshare_dead;
    d.pshare_cpu_ms += sign * b.pshare_cpu_ms;
    d.pshare_join_ms += sign * b.pshare_join_ms;
    d.parity_tried += sign * b.parity_tried;
    d.parity_moved += sign * b.parity_moved;
    d.adm_retired += sign * b.adm_retired;
    d.adm_adopted += sign * b.adm_adopted;
    d.adm_assigned += sign * b.adm_assigned;
    d.adm_started += sign * b.adm_started;
    d.adm_bytes += sign * b.adm_bytes;
    d.adm_landed += sign * b.adm_landed;
    d.adm_published += sign * b.adm_published;
    d.adm_invalidated += sign * b.adm_invalidated;
    d.adm_collisions += sign * b.adm_collisions;
    d.adm_hits += sign * b.adm_hits;
    d.adm_lost += sign * b.adm_lost;
    d.adm_rounds_on += sign * b.adm_rounds_on;
    d.adm_off_loan += sign * b.adm_off_loan;
    d.adm_off_refill += sign * b.adm_off_refill;
    d.adm_off_pass += sign * b.adm_off_pass;
    d.adm_off_failed += sign * b.adm_off_failed;
    d.adm_off_other += sign * b.adm_off_other;
    d.prompt_layers += sign * b.prompt_layers;
    d.prompt_routed_layers += sign * b.prompt_routed_layers;
    d.prompt_staged_bytes += sign * b.prompt_staged_bytes;
    d.prompt_resident_bytes += sign * b.prompt_resident_bytes;
    d.zone_evictions += sign * b.zone_evictions;
    d.prompt_wait_s += sign * b.prompt_wait_s;
    d.lent_holds += sign * b.lent_holds;
    d.graph_windows_captured += sign * b.graph_windows_captured;
    d.graph_replays += sign * b.graph_replays;
    d.loans += sign * b.loans;
    d.borrowed_slots += sign * b.borrowed_slots;
    d.borrowed_bytes += sign * b.borrowed_bytes;
    d.borrowed_residents += sign * b.borrowed_residents;
    d.borrowed_resident_bytes += sign * b.borrowed_resident_bytes;
    d.refill_batches += sign * b.refill_batches;
    d.refill_skipped += sign * b.refill_skipped;
    d.refill_carried += sign * b.refill_carried;
    d.refill_submit_s += sign * b.refill_submit_s;
    w(d.lane_wait_refill, b.lane_wait_refill);
    d.short_windows += sign * b.short_windows;
    d.short_rows += sign * b.short_rows;
    return d;
}

std::string expert_cache::counters_json() const {
    return counters_json(counters(), 0);
}

std::string expert_cache::counters_json(const expert_cache_counters & c, size_t share_from) const {
    const int64_t all = c.hit_bytes + c.gpu_miss_bytes + c.cpu_miss_bytes;
    std::string s;
    char b[1024];
    std::snprintf(b, sizeof(b),
                  "{\"devices\": %d, \"windows\": %lld, \"device_windows\": %lld, \"rounds\": %lld, \"lookups\": %lld, "
                  "\"hits\": %lld, \"gpu_miss\": %lld, \"cpu_miss\": %lld, \"hit_bytes\": %lld, "
                  "\"gpu_miss_bytes\": %lld, \"cpu_miss_bytes\": %lld, \"byte_hit_rate\": %.6f, \"cpu_windows\": %lld, "
                  "\"cpu_rows\": [%lld, %lld, %lld, %lld, %lld], \"gpu_rows\": [%lld, %lld, %lld, %lld, %lld], ",
                  p_->dv.nd, (long long)c.windows, (long long)c.device_windows, (long long)c.rounds,
                  (long long)c.lookups,
                  (long long)c.hits, (long long)c.gpu_miss, (long long)c.cpu_miss, (long long)c.hit_bytes,
                  (long long)c.gpu_miss_bytes, (long long)c.cpu_miss_bytes,
                  all > 0 ? double(c.hit_bytes) / double(all) : 0.0, (long long)c.cpu_windows,
                  (long long)c.cpu_rows[1], (long long)c.cpu_rows[2], (long long)c.cpu_rows[3],
                  (long long)c.cpu_rows[4], (long long)c.cpu_rows[5], (long long)c.gpu_rows[1],
                  (long long)c.gpu_rows[2], (long long)c.gpu_rows[3], (long long)c.gpu_rows[4],
                  (long long)c.gpu_rows[5]);
    s += b;
    // ME2b: the per-unit thread-ms only when the pool's diagnostic clocks ran (off in serving); otherwise disabled
    char um[256] = "\"unit_thread_ms\": \"disabled\"";
    if (p_->win.unit_timing()) {
        std::snprintf(um, sizeof(um),
                      "\"unit_thread_ms\": {\"gu_unpack\": %.3f, \"gu_compute\": %.3f, \"dn_unpack\": %.3f, "
                      "\"dn_compute\": %.3f}",
                      c.cpu_gu_unpack_ms, c.cpu_gu_compute_ms, c.cpu_dn_unpack_ms, c.cpu_dn_compute_ms);
    }
    std::snprintf(b, sizeof(b),
                  "\"cpu_service\": {\"gu_ms\": %.3f, \"act_ms\": %.3f, \"dn_ms\": %.3f, \"q8_ms\": %.3f, "
                  "\"bytes\": %lld, %s}, \"cpu_wait\": {\"n\": %lld, \"ms\": %.3f}, "
                  "\"h2d\": {\"lane_bytes\": %lld, \"lane_ms\": %.3f, \"swap_bytes\": %lld, \"swap_ms\": %.3f, "
                  "\"refill_bytes\": %lld, \"refill_ms\": %.3f}, \"swaps\": %lld, \"passes\": %lld, "
                  "\"refills\": %lld, ",
                  c.cpu_gu_ms, c.cpu_act_ms, c.cpu_dn_ms, c.cpu_q8_ms, (long long)c.cpu_bytes, um,
                  (long long)c.cpu_wait.n, c.cpu_wait.s * 1e3, (long long)c.lane_bytes, c.lane_copy_ms,
                  (long long)c.swap_bytes, c.swap_copy_ms, (long long)c.refill_bytes, c.refill_copy_ms,
                  (long long)c.swaps, (long long)c.passes, (long long)c.refills);
    s += b;
    std::snprintf(b, sizeof(b),
                  "\"waits\": {\"lane\": {\"n\": %lld, \"ms\": %.3f}, \"swap\": {\"n\": %lld, \"ms\": %.3f}, "
                  "\"publish\": {\"n\": %lld, \"ms\": %.3f}, \"desc\": {\"n\": %lld, \"ms\": %.3f}}, "
                  "\"host_bytes\": {\"copy_stream\": %lld, \"swap_stream\": %lld, \"cpu_pool\": %lld}, ",
                  (long long)c.lane_wait.n, c.lane_wait.s * 1e3, (long long)c.swap_wait.n, c.swap_wait.s * 1e3,
                  (long long)c.publish_wait.n, c.publish_wait.s * 1e3, (long long)c.desc_wait.n, c.desc_wait.s * 1e3,
                  (long long)(c.lane_bytes + c.prompt_staged_bytes), (long long)(c.swap_bytes + c.refill_bytes),
                  (long long)c.cpu_bytes);
    s += b;
    std::snprintf(b, sizeof(b),
                  "\"threads\": {\"service_cpu\": %d, \"service_mask\": \"0x%llx\", \"decode_cpu\": %d, "
                  "\"decode_mask\": \"0x%llx\", \"device_flags\": [\"0x%x\", \"0x%x\"], \"waits_quiet\": %s}, "
                  "\"executor\": {\"graph\": %s, \"windows_captured\": %lld, \"replays\": %lld}, ",
                  p_->win.service_cpu(), (unsigned long long)p_->win.service_mask(), p_->win.decode_cpu(),
                  (unsigned long long)p_->win.decode_mask(), p_->dv.flags[0], p_->dv.flags[1],
                  p_->dv.quiet ? "true" : "false", graph_ ? "true" : "false",
                  (long long)c.graph_windows_captured, (long long)c.graph_replays);
    s += b;
    std::snprintf(b, sizeof(b),
                  "\"loan\": {\"loans\": %lld, \"slots\": %lld, \"bytes\": %lld, \"residents\": %lld, "
                  "\"resident_bytes\": %lld, \"stage_need\": [%lld, %lld]}, \"refill\": {\"returns\": %lld, "
                  "\"batches\": %lld, \"bytes\": %lld, \"submit_ms\": %.3f, \"copy_ms\": %.3f, \"skipped\": %lld, "
                  "\"carried\": %lld, \"overlap\": %s, \"batch_layers\": %d}, \"lane_wait_refill\": {\"n\": %lld, "
                  "\"ms\": %.3f}, \"lane_wait_no_refill\": {\"n\": %lld, \"ms\": %.3f}, \"short_read\": "
                  "{\"windows\": %lld, \"rows\": %lld}, ",
                  (long long)c.loans, (long long)c.borrowed_slots, (long long)c.borrowed_bytes,
                  (long long)c.borrowed_residents, (long long)c.borrowed_resident_bytes,
                  (long long)p_->loan.stage_need(0),
                  (long long)p_->loan.stage_need(1), (long long)c.refills, (long long)c.refill_batches,
                  (long long)c.refill_bytes, c.refill_submit_s * 1e3, c.refill_copy_ms, (long long)c.refill_skipped,
                  (long long)c.refill_carried, legacy_ ? "\"legacy\"" : (o_.refill_overlap ? "true" : "false"),
                  o_.refill_batch_layers, (long long)c.lane_wait_refill.n, c.lane_wait_refill.s * 1e3,
                  (long long)(c.lane_wait.n - c.lane_wait_refill.n), (c.lane_wait.s - c.lane_wait_refill.s) * 1e3,
                  (long long)c.short_windows, (long long)c.short_rows);
    s += b;
    std::snprintf(b, sizeof(b),
                  "\"prompt_layers\": %lld, \"prompt_routed_layers\": %lld, \"prompt_staged_bytes\": %lld, "
                  "\"prompt_resident_bytes\": %lld, \"ring\": {\"loans\": %lld, \"routed_layers\": %lld, "
                  "\"ahead_layers\": %lld, \"groups\": %lld, \"cover_pm_sum\": %lld, \"bytes\": %lld, "
                  "\"loan_bytes\": %lld}, "
                  "\"parity\": {\"tried\": %lld, \"moved\": %lld}, \"zone_evictions\": %lld, \"prompt_wait_s\": %.6f, "
                  "\"lent\": {\"bytes\": %lld, \"holds\": %lld}, \"resident\": %d, \"share\": %.3f, ",
                  (long long)c.prompt_layers, (long long)c.prompt_routed_layers, (long long)c.prompt_staged_bytes,
                  (long long)c.prompt_resident_bytes, (long long)c.ring_loans, (long long)c.ring_routed_layers,
                  (long long)c.ring_ahead_layers, (long long)c.ring_groups, (long long)c.ring_cover_pm,
                  (long long)o_.ring_bytes, (long long)c.ring_loan_bytes, (long long)c.parity_tried,
                  (long long)c.parity_moved, (long long)c.zone_evictions, c.prompt_wait_s,
                  (long long)p_->loan.lent_bytes(), (long long)c.lent_holds, resident(), share());
    s += b;
    // MX7: the prompt CPU share's counters and its state now
    std::snprintf(b, sizeof(b),
                  "\"prompt_share\": {\"chunks\": %lld, \"layers\": %lld, \"off_layers\": %lld, \"experts\": %lld, "
                  "\"pairs\": %lld, \"rows\": %lld, \"unstaged_bytes\": %lld, \"dead\": %lld, \"cpu_ms\": %.3f, "
                  "\"join_ms\": %.3f, ",
                  (long long)c.pshare_chunks, (long long)c.pshare_layers, (long long)c.pshare_off_layers,
                  (long long)c.pshare_experts, (long long)c.pshare_pairs, (long long)c.pshare_rows,
                  (long long)c.pshare_unstaged_bytes, (long long)c.pshare_dead, c.pshare_cpu_ms, c.pshare_join_ms);
    s += b + p_->share.state_json() + "}, ";
    // ME9 (Shell 0a1b4eb5): admission, then the share attempts' array, closing the record
    return s + counters_tail_json(p_->res.adm_held(), c, p_->win.share_log(), share_from);
}

namespace {
// a reading as a JSON number, or null when it is not finite
std::string json_num(double v, const char * fmt) {
    if (!std::isfinite(v)) {
        return "null";
    }
    char b[64];
    std::snprintf(b, sizeof(b), fmt, v);
    return b;
}
} // namespace

std::string counters_tail_json(int32_t reserved_now, const expert_cache_counters & c,
                               const std::vector<share_attempt> & log, size_t from) {
    std::string s;
    char b[1024];
    std::snprintf(b, sizeof(b),
                  "\"admission\": {\"reserved_now\": %d, \"retired\": %lld, \"adopted\": %lld, \"assigned\": %lld, "
                  "\"started\": %lld, \"bytes\": %lld, \"landed\": %lld, \"published\": %lld, \"invalidated\": %lld, "
                  "\"collisions\": %lld, \"hits\": %lld, \"lost\": %lld, \"rounds_on\": %lld, \"off\": {\"loan\": "
                  "%lld, \"refill\": %lld, \"pass\": %lld, \"failed\": %lld, \"other\": %lld}}, "
                  "\"share_attempts\": [",
                  reserved_now, (long long)c.adm_retired, (long long)c.adm_adopted, (long long)c.adm_assigned,
                  (long long)c.adm_started, (long long)c.adm_bytes, (long long)c.adm_landed, (long long)c.adm_published,
                  (long long)c.adm_invalidated, (long long)c.adm_collisions, (long long)c.adm_hits,
                  (long long)c.adm_lost, (long long)c.adm_rounds_on, (long long)c.adm_off_loan,
                  (long long)c.adm_off_refill, (long long)c.adm_off_pass, (long long)c.adm_off_failed,
                  (long long)c.adm_off_other);
    s += b;
    for (size_t i = from; i < log.size(); ++i) {
        const share_attempt & x = log[i];
        std::snprintf(b, sizeof(b), "%s{\"round\": %lld, \"update\": %d, \"samples\": %lld, ", i == from ? "" : ", ",
                      (long long)x.round, x.r.update, (long long)x.r.samples);
        s += b;
        // ME5 rider: L is the lane's own reading (its bytes and device-0 copy time), the probe beside it, and the
        // unrounded share the deadband compares
        s += "\"link_gbps\": " + json_num(x.r.link_gbps, "%.3f") + ", \"lane_bytes\": " +
             std::to_string(x.r.lane_bytes) + ", \"lane_ms\": " + json_num(x.r.lane_ms, "%.3f") +
             ", \"probe_gbps\": " + json_num(x.r.probe_gbps, "%.3f") + ", \"tc_us\": " + json_num(x.r.tc_us, "%.2f") +
             ", \"rate_gbps\": " + json_num(x.r.rate_gbps, "%.3f") + ", \"raw\": " + json_num(x.r.raw_share, "%.4f") +
             ", \"old\": " + json_num(x.r.old_share, "%.3f") +
             ", \"new\": " + json_num(x.r.new_share, "%.3f") + ", \"reason\": \"" +
             (x.r.reason != nullptr ? x.r.reason : "") + "\"}";
    }
    return s + "]}";
}

// ME3: the request's one line: the effective settings, the executor and arm, the refill outstanding at its entry and
// now (the next request's or the drain's), and the counters since its entry
std::string expert_cache::request_json() {
    int64_t bytes = 0;
    const int64_t keys = p_->loan.restore_keys(bytes);
    char b[512];
    std::snprintf(b, sizeof(b),
                  "{\"request\": %lld, \"executor\": \"%s\", \"legacy\": %s, \"refill_at_entry\": {\"keys\": %lld, "
                  "\"bytes\": %lld}, \"refill_outstanding_keys\": %lld, \"zone_taken\": %s, \"settings\": ",
                  (long long)requests_, graph_ ? "graph" : "eager", legacy_ ? "true" : "false",
                  (long long)req_rf_keys_, (long long)req_rf_bytes_, (long long)keys,
                  p_->loan.taken() ? "true" : "false");
    return std::string(b) + (settings_json_.empty() ? std::string("{}") : settings_json_) +
           ", \"counters\": " + counters_json(counters_delta(counters(), req_n_), req_share_) +
           ", \"native_pages\": " + p_->res.native_json() + "}";
}

std::string expert_cache::plan_text() const {
    const parts & p = *p_;
    const cache::devices & dv = p.dv;
    std::string s = p.res.plan_pool(dv, o_.reserve_bytes, o_.routed_rows) + p.loan.plan_lines() +
                    p.res.plan_ranking(o_.profile, o_.profile_save, o_.adapt) + p.win.plan_share(p.res.link_gbps()) +
                    p.share.plan_line();
    char b[640];
    for (int32_t d = 0; d < dv.nd; ++d) {
        const int64_t all = device_bytes(d);
        std::snprintf(b, sizeof(b), "  device %d: decode lane %s; other buffers %s; all cache device bytes %s\n",
                      dv.dev[d], gib(p.win.lane_bytes()).c_str(),
                      gib(all - p.res.pool_bytes() - p.win.lane_bytes()).c_str(), gib(all).c_str());
        s += b;
    }
    const std::string cpus = p.win.cpu_list();
    std::snprintf(b, sizeof(b),
                  "  host backing %s page-locked (packed in %.1f s, registered in %.1f s); pinned buffers %s; "
                  "CPU workers %d (pinned to CPUs %s, the masks read back)\n",
                  gib(p.res.backing_bytes()).c_str(), p.res.pack_s(), p.res.register_s(), gib(pinned_bytes()).c_str(),
                  p.win.workers(), cpus.empty() ? "-" : cpus.c_str());
    s += b + p.res.native_line() + p.res.adm_line();
    return s + p.win.plan_threads();
}

} // namespace fn::runtime
