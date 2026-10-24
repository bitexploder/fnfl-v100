#pragma once

// ME3 (the cleanup): the expert cache's residency and placement (expert_cache.h describes the cache). The one owner of:
//   - the backing: every target layer's 512 experts packed once in host memory and page-locked;
//   - each device's pool (the class slots, then the prompt zone's two staging buffers and the lent region) and the
//     slots' layout and keys;
//   - the word table the GPU reads (the residency rows, the windows' descriptor words) and its host mirrors;
//   - the swap passes: their patch buffers, swap streams, events and per-layer counts, the pending pass, the usage and
//     heat, the boot ranking and the learned one.
// Proposed placements are values (choose: an adapt pass's moves; the loan's restore set) until published. Published
// residency (the slots' keys, each key's slot, the rows) changes only at a publication boundary: fill (synchronous),
// publish (a pass: its moves resident at once, each layer's readers after its copies by the layer's wait), publish
// landed (a refill's moves, only after their copies landed) and retire (a loan's slots, the rows after the zone's
// earlier readers). Every mutator runs on the decode thread with the service idle; while windows run the service
// reads the published state (its plan) and counts usage, which the round boundary reads and decays with the service
// idle.

#include "fn/loader/loader.h"
#include "fn/model/model.h"
#include "fn/model/shapes.h"
#include "fn/runtime/cache_counters.h"
#include "fn/runtime/cache_passes.h"
#include "fn/runtime/admission.h"
#include "fn/runtime/cache_util.h"
#include "fn/runtime/expert_policy.h"
#include "fn/runtime/native_pages.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fn::runtime {

class residency {
  public:
    using move = std::pair<int32_t, int32_t>; // (slot, key): key resident in slot, the slot's key evicted

    struct pool_options {
        int64_t pool_bytes = -1, reserve_bytes = 0, device_cap_bytes = -1;
        int32_t max_swaps = 96;
        bool swap_overlap = true;
    };

    residency() = default;
    residency(const residency &) = delete;
    residency & operator=(const residency &) = delete;

    // init, in the cache's order: the layers, classes and boot ranking, the backing packed (zero_backing: left zero)
    // and registered; each device's word table, zero expert, rows' mirrors, swap stream, events and counts; then
    // (after the cache's other buffers and the capture) the pool out of what is left (fr_min: the smaller device's
    // free bytes under its ceiling), the pass buffers, the slots and the boot placement; the link probe and the fill
    // ME4: native_hold (TEST, the gate's lock check) holds the native pages even with zero_backing, not serving
    bool init_layers(model::model & m, int32_t nd, const std::string & profile, int32_t pack_threads, bool zero_backing,
                     std::string & err, bool native_hold = false);
    bool init_devices(const cache::devices & dv, std::string & err);
    bool measure(const cache::devices & dv, int64_t device_cap_bytes, int64_t & fr_min, std::string & err);
    bool init_pool(const cache::devices & dv, int64_t fr_min, const pool_options & po, int64_t lent_bytes,
                   std::string & err);
    bool probe_link(const cache::devices & dv, std::string & err);
    bool fill_boot(const cache::devices & dv, std::string & err) { return fill(dv, boot_.slot_key, err); }
    void release(const cache::devices & dv); // the devices idle: the buffers freed, the backing unregistered

    // the published state (the decode thread; the service reads it while windows run)
    const loader::expert_layer & layer(int32_t l) const { return lay_[size_t(l)]; }
    int32_t slot_of(int32_t key) const { return entry_slot_[size_t(key)]; }
    int32_t key_of(int32_t s) const { return slot_key_[size_t(s)]; }
    int64_t slot_off(int32_t s) const { return slot_off_[size_t(s)]; }
    int64_t slot_bytes(int32_t s) const { return cls_[size_t(boot_.slot_cls[size_t(s)])].entry; }
    int32_t n_slots() const { return int32_t(slot_off_.size()); }
    int32_t zone_first() const { return zone_first_; }
    int32_t zone_slots() const { return n_slots() - zone_first_; }
    uint8_t * pool(int32_t d) const { return pool_[d]; }
    const uint8_t * slot_addr(int32_t d, int32_t s) const { return pool_[d] + slot_off_[size_t(s)]; }
    const uint8_t * entry_src(int32_t layer, int32_t e, int32_t d) const; // device d's bytes of the entry
    const uint8_t * backing(int32_t layer) const { return backing_[size_t(layer)]; }
    // ME4: the layer's native views (the loader's, from the model's mappings), and whether the CPU path reads them:
    // built and checked for every layer at init; held resident (native_pages) unless the backing is left zero (the
    // dry runs: their experts are zero, the files' are not, so the reference path serves and the report says so)
    const fn::experts::cpu::native_layer & native(int32_t layer) const { return native_[size_t(layer)]; }
    bool native_ready() const { return native_ready_; }
    std::string native_line() const; // the report's lines on the views, their check and their pages
    // ME4: the native pages' residency now (each request's line): locked, pages resident, RssFile, major faults; null
    // when none are held
    std::string native_json() const;
    int64_t part_off(int32_t layer, int32_t card, bool down) const; // card's part in a device entry
    int64_t pool_bytes() const { return pool_bytes_; }
    int64_t zone_off() const { return zone_off_; }
    int64_t zone_buf() const { return zone_buf_; }
    int64_t lent_off() const { return lent_off_; }
    int64_t max_entry() const { return max_entry_; } // a device's largest entry
    int64_t backing_bytes() const { return backing_bytes_; }
    int32_t resident() const;
    double link_gbps() const { return link_gbps_; }
    bool pending() const { return pending_; }
    // the word table of a device (8-byte words): residency rows [48][512], tab [parts][512], tab2 [parts][512], the
    // CPU slot word, the run word
    uint64_t * words(int32_t d) const { return words_[d]; }
    void * swap_stream(int32_t d) const { return swap_[d]; } // the passes' and the refill's copies
    const uint8_t * zero(int32_t d) const { return zero_[d]; }
    static int64_t w_res(int32_t layer, int32_t e) { return int64_t(layer) * 512 + e; }
    int64_t w_tab(int32_t part, int32_t e) const { return int64_t(48) * 512 + int64_t(part) * 512 + e; }
    int64_t w_tab2(int32_t part, int32_t e) const { return w_tab(parts_ + part, e); }
    int64_t w_ext() const { return w_tab(2 * parts_, 0); }
    int64_t w_run() const { return w_ext() + 1; }

    // the swap waits a window's GPU work enqueues (the decode thread or a capture): eager, the layer's wait for this
    // round's pass copies (once); a replay, a kernel waiting for the layer's count, its times in mapped memory
    bool eager_wait(int32_t d, int32_t layer, void * stream, std::string & err);
    bool replay_wait(int32_t d, int32_t layer, void * stream, std::string & err) const;
    void mark_replay_waits(); // pass_begin: the layers whose copies the pass enqueued are the replay's waits
    // ME3 (load G ruling): each device's per-layer copy counts ([48], device memory) a replay's swap wait reads, for
    // the cache's release of the GPU's waits
    uint32_t * swap_done(int32_t d) const { return sw_done_[d]; }
    bool read_swap_waits(const cache::devices & dv, std::string & err);

    // the service's usage count (mode 0): +1 a row and ID
    void count(int32_t layer, int32_t e) { usage_[size_t(layer * cache::n_ex + e)] += 1.0f; }
    // ME4: a key's current (decayed) usage, the swap chooser's measure (the ring's place)
    float usage(int32_t key) const { return usage_[size_t(key)]; }

    // publication (the decode thread, the service idle). fill: every slot to target's key (-1 empty), copied
    // synchronously where it changes, the rows uploaded. publish: a pass of moves (refill: a zone return's, its copies
    // also after the copy streams' staging, counted by the loan; else an adapt pass, counted here), on the compute
    // streams (null: synchronous); bytes its copies (both devices). commit
    // (wait: synchronize for it). publish_landed: a refill's landed moves, skipping a key resident again or a slot
    // filled (skipped), each device's row patches on its compute stream after the previous ones (null: uploaded).
    // retire: a zone slot's key evicted from the tables (returned); upload_rows_ordered: each device's rows from a
    // snapshot, in its compute stream's order
    bool fill(const cache::devices & dv, const std::vector<int32_t> & target, std::string & err);
    bool publish(const cache::devices & dv, std::vector<move> moves, bool refill, void * const stream[2],
                 int64_t & bytes, std::string & err);
    bool try_commit(const cache::devices & dv, bool wait, std::string & err);
    bool publish_landed(const cache::devices & dv, const move * mv, size_t n_mv, void * const stream[2],
                        int64_t & skipped, std::string & err);
    int32_t retire(int32_t s);

    // ME9, admission (the one-device path; fn/runtime/admission.h): up to adm::per_layer reserved slots a layer, each
    // record with a completion event of its own. The decode thread, the service idle:
    //   adm_publish: the records whose copy completed landed (sync: their events synchronized) and published, each a
    //                key installed in its slot with no copy, the rows patched in one batch on stream[0] (null:
    //                uploaded); a collision keeps the slot reserved; failed (or an abandon/release since) invalidates
    //                every record copying or landed instead
    //   adm_reserve: each layer's reservations topped up: an orphaned slot adopted, else the resident of least decayed
    //                usage retired (key, then slot, breaking ties), its row nulled in one batch on stream[0]; only in a
    //                round that admits (no loan, refill or pending pass), so no other writer owns these slots
    //   adm_loan_returned: a loan gave back what it borrowed; every reserved slot's bytes are garbage (a new
    //                generation)
    //   adm_reset: every record discarded (before the boot placement's fill)
    //   adm_poison: any thread (an abandoned pass, released waits): the next adm_publish invalidates instead
    // The service, in its windows: adm_free (a layer's empty records), adm_start (after the slot's copy and its event
    // were enqueued), adm_flag (1: admitted and resident since, 2: retired for a reservation and not resident since)
    bool adm_on() const { return adm_on_; }
    bool adm_publish(const cache::devices & dv, bool failed, bool sync, void * const stream[2], std::string & err);
    bool adm_reserve(const cache::devices & dv, void * const stream[2], std::string & err);
    void adm_loan_returned() { adm_.return_from_loan(); }
    void adm_reset();
    void adm_poison() { adm_poisoned_.store(true); }
    int32_t adm_free(int32_t layer, int32_t (&out)[adm::per_layer]) const;
    bool adm_in_flight(int32_t key) const { return adm_.in_flight(key); }
    int32_t adm_slot(int32_t i) const { return adm_.at(i).slot; }
    void * adm_event(int32_t i) const { return adm_ev_[i]; }
    bool adm_start(int32_t i, int32_t key) const { return adm_.start(i, key); } // the records, not the maps
    uint8_t adm_flag(int32_t key) const { return adm_flag_.empty() ? 0 : adm_flag_[size_t(key)]; }
    int32_t adm_held() const { return adm_.held_all(); }
    std::string adm_line() const; // the plan text's line: reserved slots and their bytes
    // TEST (--me9-check): every reserved slot filled with 0xFF; every admitted resident (flag 1) compared with its
    // backing bytes (synchronous; checked, bad)
    bool adm_test_poison(const cache::devices & dv, std::string & err) const;
    bool adm_test_verify(const cache::devices & dv, int64_t & checked, int64_t & bad, std::string & err) const;
    // TEST (--me9-check, Shell 0a1b4eb5 and request 11): a known miss a layer. routed [48][10]: each layer's last
    // decode window's row-0 experts (the dry route repeats them every window); of those, the resident with the most
    // usage and no copy in flight leaves its slot (left empty, its rows zeroed, uploaded synchronously). keys gets
    // them, and from then each one's lifecycle is tracked (adm_test_note's events, test-only, one key a layer at most)
    bool adm_test_evict(const cache::devices & dv, const std::vector<int32_t> & routed, std::vector<int32_t> & keys,
                        std::string & err);
    static constexpr uint32_t ev_missed = 1u, ev_assigned = 2u, ev_started = 4u, ev_published = 8u,
                              ev_collided = 16u, ev_retired = 32u, ev_left = 64u;
    void adm_test_note(int32_t key, uint32_t ev) const {
        if (!test_idx_.empty() && test_idx_[size_t(key)] >= 0) {
            test_ev_[test_idx_[size_t(key)]].fetch_or(ev);
        }
    }
    // the tracked keys: each must have missed (GPU-served), been assigned and copied (started) and published; then
    // be resident by admission now (n_admitted) or have left its slot after (n_replaced, legitimately replaced).
    // failures lists every other key with its events
    void adm_test_forced(int32_t & n_admitted, int32_t & n_replaced, std::string & failures) const;
    bool upload_rows_ordered(const cache::devices & dv, void * const stream[2], std::string & err);
    bool upload_rows(const cache::devices & dv, std::string & err); // synchronous

    // the round boundary's adapt choice (a proposal) and the usage's decay; the boot state's usage and heat
    std::vector<move> choose(const policy::adapt_options & a);
    void reset_usage();
    // the learned ranking: residents (and the keys a loan holds out, lent) first by heat, then heat, then the prior
    void mark_learned() { learned_dirty_ = true; }
    bool learned_dirty() const { return learned_dirty_; }
    bool save_profile(const std::string & path, const std::vector<int32_t> & lent, std::string & err);

    // TEST (the loan's checks): every zone slot holding a key against its backing on each device; a slot poisoned
    bool verify_zone(const cache::devices & dv, int64_t & checked, int64_t & bad, std::string & err) const;
    bool poison_slot(const cache::devices & dv, int32_t s, std::string & err) const;

    // the memory plan's lines: each device's free memory, the pool, its classes and the zone; the ranking's
    std::string plan_pool(const cache::devices & dv, int64_t reserve, int32_t routed_rows) const;
    std::string plan_ranking(const std::string & profile, const std::string & profile_save,
                             const policy::adapt_options & a) const;
    double pack_s() const { return pack_s_; }
    double register_s() const { return register_s_; }
    const expert_cache_counters & counters() const { return n_; }
    void reset_counters() { n_ = expert_cache_counters(); }
    int64_t device_bytes(int32_t d) const { return own_.dev_bytes[d]; }
    int64_t host_bytes() const { return own_.host_bytes; }

  private:
    struct cls {
        int32_t gu_fmt = -1, dn_fmt = -1;
        int64_t entry = 0;   // bytes a slot on one device (both shards on one device; a shard on two)
        int64_t payload = 0; // the class's experts' bytes (all its layers), likewise
        int32_t layers = 0;
        int32_t count = 0; // its main slots
    };
    void place_key(int32_t s, int32_t key);
    bool init_native(int32_t threads, bool check, std::string & err);
    void unplace(int32_t s);
    // ME9: batch b's (0: publications, 1: retirements) row patches: its previous submission awaited, then fill(idx,
    // val) gives n patches, applied on stream (null: the rows uploaded)
    template <typename F>
    bool adm_batch(const cache::devices & dv, int32_t b, void * stream, F fill, std::string & err);

    cache::owned own_;
    int32_t nd_ = 1, parts_ = 4;
    int32_t dev_[2] = {-1, -1};
    std::vector<loader::expert_layer> lay_; // per target layer
    std::vector<int32_t> cls_of_;           // per target layer
    std::vector<cls> cls_;
    int64_t max_entry_ = 0, max_part_ = 0;
    // backing (page-locked)
    uint8_t * backing_base_ = nullptr;
    size_t backing_map_ = 0;
    bool backing_registered_ = false;
    std::vector<uint8_t *> backing_; // per layer
    int64_t backing_bytes_ = 0;
    double pack_s_ = 0, register_s_ = 0;
    // ME9: the admission records, their completion events, the two row-patch batches (device and pinned host) and
    // their events; per key its admission flag (adm_flag)
    mutable adm::table adm_; // the service starts records through the const view it holds (never the maps)
    bool adm_on_ = false;
    std::atomic<bool> adm_poisoned_{false};
    void * adm_ev_[adm::max_recs] = {};
    int32_t * adm_idx_[2] = {nullptr, nullptr}, * adm_idx_h_[2] = {nullptr, nullptr};
    uint64_t * adm_val_[2] = {nullptr, nullptr}, * adm_val_h_[2] = {nullptr, nullptr};
    void * adm_bev_[2] = {nullptr, nullptr};
    bool adm_bev_live_[2] = {false, false};
    std::vector<uint8_t> adm_flag_;
    std::vector<int8_t> test_idx_;                    // TEST: a tracked key's index, -1 (adm_test_evict)
    std::vector<int32_t> test_keys_;                  // TEST: the tracked keys
    mutable std::atomic<uint32_t> test_ev_[fn::n_layer] = {}; // TEST: their events
    // ME4: the native views and their pages (released after the backing; the model's mappings outlive both)
    std::vector<fn::experts::cpu::native_layer> native_;
    native_pages native_pages_;
    bool native_ready_ = false;
    int32_t native_checked_ = 0; // experts checked bytewise against the backing at init
    bool native_dry_hold_ = false; // TEST (--ex-native-hold): held with a zero backing, not serving
    double native_check_s_ = 0;
    // the pools and their slots (offsets are a device's: the same on both)
    uint8_t * pool_[2] = {nullptr, nullptr};
    int64_t pool_bytes_ = 0, main_bytes_ = 0, free_before_[2] = {0, 0}, used_before_[2] = {0, 0};
    int64_t device_cap_ = -1;
    int64_t zone_off_ = 0, zone_buf_ = 0, lent_off_ = 0, lent_bytes_ = 0;
    policy::place_out boot_;        // the profile's placement (slots and their boot keys)
    std::vector<int32_t> profile_;  // the boot ranking (keys)
    std::vector<int64_t> slot_off_;
    std::vector<int32_t> slot_key_;   // layer x 512 + expert, -1 empty
    std::vector<int32_t> entry_slot_; // per key, -1 when not resident (the devices' residency rows mirrored)
    int32_t zone_first_ = 0;          // the first zone slot (boot_.n_main)
    // the swaps' state: usage and heat per key, the pending pass and its kind
    std::vector<float> usage_;
    std::vector<double> heat_;
    bool pending_ = false, pending_refill_ = false;
    bool swap_overlap_ = true;
    bool learned_dirty_ = false;
    std::vector<uint8_t> wait_has_[2]; // per device, per layer: this pass's copies not yet waited for
    std::vector<uint8_t> waited_[2];   // per device, per layer: waited for (1 its events, 2 its wait kernel)
    double link_gbps_ = 0;
    // the word table and its host mirrors
    uint64_t * words_[2] = {nullptr, nullptr};
    uint8_t * zero_[2] = {nullptr, nullptr};          // a zero expert (the largest gate/up and down)
    const uint8_t ** res_h_[2] = {nullptr, nullptr};  // per device: its residency rows [48][512] mirrored
    uint64_t * res_snap_h_[2] = {nullptr, nullptr};   // per device: the rows retire uploads (a snapshot)
    // a pass's patches (device and pinned host); the refill publications' patches, a slot each
    int32_t max_moves_ = 0;
    int32_t * pass_idx_[2] = {nullptr, nullptr}, * pass_idx_h_[2] = {nullptr, nullptr};
    uint64_t * pass_val_[2] = {nullptr, nullptr}, * pass_val_h_[2] = {nullptr, nullptr};
    int32_t * rf_idx_[2] = {nullptr, nullptr}, * rf_idx_h_[2] = {nullptr, nullptr};
    uint64_t * rf_val_[2] = {nullptr, nullptr}, * rf_val_h_[2] = {nullptr, nullptr};
    // the swap streams and their events
    void * swap_[2] = {nullptr, nullptr};
    void * ev_pub_[2] = {nullptr, nullptr};
    void * ev_cp_[2] = {nullptr, nullptr};  // a refill pass's copies after the copy stream's staging
    void * ev_rfp_[2] = {nullptr, nullptr}; // a refill publication's patches applied (the host reuses their buffers)
    std::vector<void *> ev_sw_[2];          // per device, per layer: its pass copies landed
    std::vector<void *> ev_sww_[2];         // per device, [layer][2] (timing): around the compute stream's wait
    void * ev_pt_[2][2] = {};               // per device (timing): around a pass's copies
    // ME3 (graph executor): a replay's swap waits are kernels: per device, per layer, the pass copies' counts (the
    // host's, the swap stream's landed, the compute stream's needed) and the waits' globaltimer pairs (mapped)
    uint32_t sw_count_[2][48] = {};
    uint32_t * sw_done_[2] = {nullptr, nullptr};
    uint32_t * sw_need_[2] = {nullptr, nullptr};
    uint64_t * swt_h_ = nullptr; // [2][48][2]
    uint64_t * swt_d_[2] = {nullptr, nullptr};
    expert_cache_counters n_; // the residency's: passes (swaps and refill passes), swap waits, publications
};

} // namespace fn::runtime
