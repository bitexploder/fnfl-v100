#pragma once

// ME3 (the cleanup): the expert cache's prompt loan and its refill (expert_cache.h describes the cache). The one owner
// of:
//   - the lent region's parts (one card: PF1's staging and PB1's arena as views of the pool's tail) and the hold;
//   - the loan: the zone slots borrowed (taken), the exact restore set (each borrowed slot's former key), each staging
//     buffer's need, the staging tables and their events, the routed prompt's IDs;
//   - the refill: its moves, its progress (loan::refill_state: published, landed, in flight, unsubmitted), the
//     landing events and the host callback's records, and its stop.
// The residency's published state changes only through the residency: a loan retires its slots' keys (retire, the
// rows uploaded in the compute streams' order), a return publishes the restore set (a pass, or the refill's landed
// moves at the round boundary: publish_landed). As the service's svc::refill, the refill's batches are submitted on
// the service thread: ready and take under the loop's lock (then this mutex: the loop's before this, never the
// reverse), submit outside it. Producers of its waits: refill_stop waits for the batch in flight to land (the
// swap stream's host callback, which takes this mutex, releases it, then notifies the loop) or for the remaining
// batches to be submitted (the service), and ends on a failure (the loop's failed). The callback's record is one of
// two, alternating per batch (one in flight), naming the refill's generation; the owner (the cache) outlives every
// callback: its release synchronizes the devices first.

#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/model/shapes.h"
#include "fn/runtime/cache_counters.h"
#include "fn/runtime/cache_residency.h"
#include "fn/runtime/cache_util.h"
#include "fn/runtime/loan.h"
#include "fn/runtime/prompt_plan.h"
#include "fn/runtime/service_loop.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fn::runtime {

class prompt_loan final : public svc::refill {
  public:
    struct options {
        int32_t routed_rows = 256, lend_pct = 85, min_keep = 128;
        bool refill_overlap = true;
        int32_t refill_batch_layers = 4;                         // MX3 (was 1)
        int64_t refill_batch_bytes = int64_t(64) << 20;          // MX3: a refill batch's bytes on one device at most
        int64_t ring_bytes = 0, group_bytes = int64_t(64) << 20; // ME4: the ring (0: none) and a group's bytes
        int32_t route_cover = 875;
    };

    prompt_loan() = default;
    prompt_loan(const prompt_loan &) = delete;
    prompt_loan & operator=(const prompt_loan &) = delete;

    // init, in the cache's order: the lent region's parts collected and their own allocations released (one card,
    // lend_region; before the pool is measured, so it takes their bytes); the staging tables, the prompt IDs and the
    // events (before the pool); after the pool, the parts' views of its tail and the zone's restore set
    void collect_region(model::model & m, const cache::devices & dv, bool lend_region);
    int64_t lent_bytes() const { return lent_bytes_; }
    bool init(const cache::devices & dv, residency & r, svc::loop & lp, const options & o, std::string & err);
    void attach();
    void release(const cache::devices & dv) { own_.release(dv); }

    // the loan (the decode thread, the service idle; the caller committed the pending pass and stopped a refill).
    // plan: the zone slots the spans and the staging need touch (worst: every expert of a layer staged) and the
    // staging need of each buffer, refused beyond lend_pct and min_keep. take: the loan (need null or legacy: the
    // whole zone, ME2b's), the borrowed slots' keys retired and kept, every earlier reader of the zone marked on the
    // compute streams, the rows uploaded in their order. give_back: the restore set returned (synchronous with null
    // streams, legacy (enqueued as a pass) or refill_overlap 0; else the bounded refill started)
    bool plan(const std::vector<model::byte_span> & need, bool worst, std::vector<uint8_t> & take,
              int64_t stage_need[2], std::string & err) const;
    bool check(const std::vector<model::byte_span> & need, std::string & err) const; // the startup check
    // ME4: plan with the parity buffers placed by current usage (by_usage) and their starts
    bool plan_at(const std::vector<model::byte_span> & need, bool worst, bool by_usage, std::vector<uint8_t> & take,
                 int64_t stage_need[2], int64_t stage_off[2], std::string & err) const;
    // ME4: a ring loan's slots and the ring's place (false, err empty: no place for the ring)
    bool plan_ring(const std::vector<model::byte_span> & need, std::vector<uint8_t> & take, int64_t & ring_lo,
                   std::string & err) const;
    bool take(void * const stream[2], const std::vector<model::byte_span> * need, bool legacy, std::string & err);
    bool give_back(void * const stream[2], bool legacy, std::string & err);
    // the hold's device side after the loan (on) or before a return (off): lent region drained (lent_sync), the
    // poison (TEST) in the borrowed slots and spans; without a lent region only the poison. On a failure the hold
    // stays as it was
    bool hold(bool on, const std::vector<model::byte_span> * need, bool legacy, std::string & err);
    bool taken() const { return taken_; }
    bool held() const { return held_; }
    void set_poison(bool on) { poison_ = on; }
    // the keys a loan holds out (the learned ranking counts them resident); empty when no loan
    std::vector<int32_t> lent_keys() const { return taken_ ? zone_keys_ : std::vector<int32_t>(); }
    int64_t restore_keys(int64_t & bytes) const; // the restore set and the refill's unpublished moves

    // prefill: layer's staged buffer (routed: its chunk's experts; else every expert, and the next layer's staged
    // into the other buffer; ME4: after the last layer only when more, another chunk, follows), the layer's readers
    // after its copies; the staged tables
    bool prompt_layer(int32_t layer, int32_t rows, const int32_t * ids, bool more, void * const stream[2],
                      std::string & err);
    const uint8_t * const * prompt_table(int32_t layer, int32_t card, bool down) const;

    // ME4, the ring (prompt_plan.h; expert_service's prompt_group calls). set_ring: the next loan stages through the
    // ring (taken only with ring_bytes > 0 and a place off the spans; else the two layer buffers). A ring layer is
    // planned at its prompt_layer (route-first: its routing read, the routed nonresident experts) or, once a layer's
    // cover reached route_cover, at the previous layer's last group_done (every nonresident expert, streamed ahead).
    // Its groups' entries [table][payload] are copied into the ring in order as space allows: each copy after the
    // releases of the entries whose bytes it reuses (and, first, after the loan's earlier readers), its table first,
    // then a copied event. group: each device's compute stream waits for the group's copied event; group_done: the
    // group's release recorded on each device's compute stream (after both logical cards' gate/up and down).
    // MX3: rows, the run's largest chunk (0: no ring), sizes the ring (pplan::ring_for)
    void set_ring(int32_t rows) { ring_rows_next_ = rows; }
    // ME4's comparison base arm (expires at ME4's disposition): the parity buffers at ME3's place, no ring
    void set_me3_place(bool on) { me3_place_ = on; }
    bool ring() const { return ring_on_; }
    int32_t prompt_groups(int32_t layer) const;
    bool prompt_group(int32_t layer, int32_t g, void * const stream[2], model::prompt_group_view & v,
                      std::string & err);
    bool prompt_group_done(int32_t layer, int32_t g, void * const stream[2], std::string & err);
    // MX7 (fn/runtime/prompt_share.h): a route-first ring layer's CPU experts, chosen after its routing sync (layer,
    // rows, the host IDs [rows][10], each expert's pairs, the layer's residency mask, the compute streams; cpu: [512]
    // set nonzero for each). The ring plan leaves them out: neither copied nor counted in a group's pairs
    using cpu_pick_fn = std::function<int32_t(int32_t, int32_t, const int32_t *, const int32_t *, const uint8_t *,
                                              void * const *, uint8_t *)>;
    void set_cpu_pick(cpu_pick_fn f) { cpu_pick_ = std::move(f); }

    // the refill (the decode thread): its landed moves published (the round boundary, the service idle; null streams:
    // synchronous), stopped (finish: every batch submitted, landed and published; else the one in flight landed and
    // published, the rest carried back to the restore set); active: an atomic read (the service's windows)
    bool refill_publish(void * const stream[2], std::string & err);
    bool refill_stop(bool finish, std::string & err);
    bool refill_active() const { return active_.load(); }
    // TEST (run-dry's deterministic outstanding refill): at most limit batches of each refill submitted (-1: no
    // limit), so the rest stays unsubmitted until a loan carries it; lifted before a drain (which waits for every
    // batch), the service notified
    void test_refill_limit(int32_t limit);

    // svc::refill (the service thread)
    bool ready() override;
    void take(size_t & a, size_t & b) override;
    bool submit(size_t a, size_t b, const std::function<bool()> & yield, std::string & err) override;
    void failed(const std::string & err) override;

    std::string plan_lines() const; // the lent region's line
    expert_cache_counters counters() const;
    void reset_counters();
    int64_t stage_need(int32_t b) const { return stage_need_[b]; }
    int64_t device_bytes(int32_t d) const { return own_.dev_bytes[d]; }
    int64_t host_bytes() const { return own_.host_bytes; }

  private:
    struct lent_part {
        model::device_buffer * buf = nullptr;
        std::string name;
        int32_t card = 0;
        int64_t off = 0, bytes = 0;
    };
    struct landing {
        prompt_loan * owner = nullptr;
        uint32_t gen = 0;
    };
    static void landed(void * rec); // the swap stream's host callback: no CUDA call
    bool stage(int32_t l, int32_t b, const uint8_t * used, std::string & err); // used: [512] or null (all)
    void end_refill(); // under mu_
    // ME4, the ring: a group, a planned layer, the plan (used and counts null: streamed ahead), the copies issued as
    // space allows, the state reset (a loan's take and return)
    static constexpr int32_t n_ring_ev = 256; // events a device: a ring entry is at least ring_bytes / n_ring_ev
    struct ring_group {
        pplan::group g;
        int64_t pairs = 0;
        int32_t id = -1;  // the byte ring's entry, -1 not yet issued
        int64_t off = 0;  // its entry's offset in the ring: the table, then the payload
    };
    struct ring_layer {
        int32_t layer = -1, rows = 0;
        std::vector<ring_group> g;
        std::vector<uint8_t> copy; // per expert: copied into the ring
        size_t next = 0, done = 0; // the first group not issued, not released
    };
    bool ring_plan(int32_t layer, int32_t rows, const int32_t * counts, const uint8_t * cpu, std::string & err);
    bool ring_issue(std::string & err);
    void ring_reset();

    cpu_pick_fn cpu_pick_; // MX7
    const cache::devices * dv_ = nullptr;
    residency * res_ = nullptr;
    svc::loop * loop_ = nullptr;
    options o_;
    cache::owned own_;
    // the lent region (one card)
    std::vector<lent_part> lent_;
    int64_t lent_bytes_ = 0;
    bool held_ = false, poison_ = false;
    // the loan
    bool taken_ = false;
    std::vector<int32_t> zone_keys_; // per zone slot: the key retired at the loan, -1 none (the restore set)
    std::vector<uint8_t> taken_slots_;
    int64_t stage_need_[2] = {0, 0};
    const uint8_t ** ptab_[2][2] = {};   // [device][staging buffer]: [parts][512]
    const uint8_t ** ptab_h_[2][2] = {};
    int32_t ptab_layer_[2] = {-1, -1};
    bool ptab_full_[2] = {false, false}; // every expert staged (else the routed ones only)
    int32_t * pids_h_ = nullptr;          // [4096][10] a routed prompt's IDs
    void * ev_free_[2][2] = {};           // [device][buffer] a staging buffer's last reader enqueued (compute stream)
    void * ev_staged_[2][2] = {};         // [device][buffer] its layer copied (copy stream)
    void * ev_route_ = nullptr;           // a routed prompt's IDs read (device 0)
    void * ev_cp_[2] = {nullptr, nullptr}; // the refill's copies after the copy stream's staging
    expert_cache_counters n_;             // the decode thread's: loans, holds, prompt staging, refills returned
    // the refill: moves (written by the decode thread while idle), its state and the service's counters (mu_)
    std::vector<residency::move> moves_;
    mutable std::mutex mu_;
    std::condition_variable cv_; // refill_stop's wait
    loan::refill_state st_;
    std::atomic<bool> active_{false};
    bool timed_ = false, read_prev_ = false; // a batch's copy time recorded and not yet read
    std::string err_;                        // the failure the loop reported
    expert_cache_counters svc_n_;            // the service's: batches, bytes, submission time, copy time (mu_)
    landing rec_[2];
    int32_t rec_at_ = 0;
    int32_t limit_ = -1, batches_ = 0; // TEST: the limit, this refill's batches
    void * ev_rf_[2] = {nullptr, nullptr};    // per device: the batch's copies landed
    void * ev_rft_[2] = {nullptr, nullptr};   // device 0 (timing): around the batch's copies
    void * ev_lease_[2] = {nullptr, nullptr}; // per device: the prompt's last readers (compute stream)
    // ME4: the parity buffers' starts (this loan's), ME3's place (the base arm)
    int64_t stage_off_[2] = {0, 0};
    bool me3_place_ = false;
    // ME4, the ring (MX3: ring_rows_next_ the next loan's largest chunk, ring_cur_ this loan's ring bytes)
    bool ring_on_ = false, ring_first_ = true, ahead_ = false;
    int32_t ring_rows_next_ = 0;
    int64_t ring_lo_ = 0, ring_min_ = 0, tab_bytes_ = 0, ring_cur_ = 0;
    std::unique_ptr<pplan::byte_ring> ring_;
    ring_layer rl_[2];                       // [layer & 1]
    uint8_t * rtab_h_[2] = {nullptr, nullptr}; // per device: pinned host tables [n_ring_ev][tab_bytes_]
    void * ev_rcp_[2][n_ring_ev] = {};       // per device and entry slot: its table and copies landed (copy stream)
    void * ev_rrel_[2][n_ring_ev] = {};      // its last readers enqueued before (compute stream)
};

} // namespace fn::runtime
