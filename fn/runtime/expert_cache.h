#pragma once

// ME1: the target's routed experts served from host memory through a byte-capped GPU cache. Two placements:
//   one device (fn::model::model::single) holding both logical cards, a slot holding an expert's both shards;
//   two devices (cached TP), device c holding card c's shard of every resident expert, the slots alike on both.
// The two logical 320-channel shards and every kernel's arithmetic stay as they are; only where an expert's packed
// bytes are read changes.
//
// Backing: every target layer's 512 experts packed once in host memory (fn::loader::pack_expert_entry), an entry being
// one expert with both shards: [gate/up 0][down 0][gate/up 1][down 1], each the bytes the resident bank would hold.
// Immutable after init and page-locked (cudaHostRegister), so every upload is one DMA from it (card c's half of an
// entry is contiguous: a device of the two-device placement copies its own half).
//
// ME2 (Strata-V100 3d4ce4e0's residency and miss path; expert_policy.h holds the host rules, experts_cpu.h the CPU
// path):
//
// GPU cache: one pool a device. Its tail is the prompt zone (two layer buffers, zone_bytes); the rest is divided among
// the layers' format classes (gate/up format, down format) by payload, each class into slots of its entry size on
// that device; slot s is the same expert on both devices. Boot (and reset_to_profile): the profile's ranking fills
// the class slots, then the zone's slots (fixed offsets, mixed classes) with the next-ranked experts that fit
// (policy::place); the profile is a STRP file (--ex-profile) or, without one, expert order.
//
// Residency changes only by swaps. Each decode window counts its routed IDs (+1 a row and ID; prefill does not count).
// Every adapt_every rounds (a round: one layer-0 window) with no pass pending, policy::choose_swaps picks at most 96
// (frequency gain 1.5, usage 2.0, decay 0.7), and the pass is published at that layer-0 window, after the service is
// idle:
//   the host tables: each victim becomes a miss and its candidate resident in the victim's slot;
//   the devices' residency rows patched (experts_cache.h apply) on the compute stream, then an event there: every
//   reader of a victim slot so far precedes it (the lease), and the swap stream waits for it;
//   the copies on the swap stream in layer order, an event after each layer's; a layer of this round with copies
//   makes the compute stream wait for its own event before its resolve (only that layer, only once).
// The pass commits without blocking (event queries) at a later layer-0 window; no pass starts while one is pending.
// --ex-swap-overlap 0 (ME2) instead synchronizes the swap stream at publication. The learned ranking (residents
// first by heat, then heat, then the prior) is saved (--ex-profile-save) by save_profile, at drain, at a prompt's start
// after decode rounds (between requests) and at destruction.
//
// The miss split: a window's distinct misses in routing order, the last (n_miss x num) >> 8 copied to the GPU lane
// (num = the share in 1/256: 0.30 at start, then policy::pcie_auto's refinements from the init's link probe and the
// pool's measured rate), the rest computed whole on the CPU (experts_cpu.h pool, into card 0's ext_ds_, which the
// combine adds after down; ME3: the cache's down instantiation runs nothing for those slots).
//
// Decode (ffn_pair, model::expert_service), per window (one layer of one round): the host enqueues on each device's
// stream, without waiting:
//   0. (a layer with copies in this round's pass) a stream wait for its copies;
//   1. resolve (experts_cache.h): the window's IDs against the device's residency rows: the hits' table entries, the
//      misses' left null; when every expert is resident it writes the window's two flags itself;
//   2. the IDs and the FFN input to pinned staging (device 0), and the window's event (ME3, a replay: its stamp);
//   3. both shards' gate_up on the hits (gate_up_args::skip 1);
//   4. a stream wait on the gate/up flag (cuStreamWaitValue32);
//   5. both shards' gate_up on the lane misses (skip 2, the table the service filled; run_if: none), then an event;
//   6. (ME2b) both shards' down at once, the CPU-served slots (the slot word, written before the gate/up flag) reading
//      their ds from zeros, so they add nothing;
//   7. on card 0's device only: a stream wait on the down flag, then combine (experts_cache.h), which adds the CPU
//      experts' weighted products into card 0's down output in route order.
// The cache's service thread takes the windows in order: it waits for the window's event (a replay's: its stamp),
// plans (the CPU jobs and the slot word with it), copies the lane misses on each device's copy stream, applies the
// table, run and slot words, writes the gate/up flags, then runs the CPU experts (the pool, the service draining with
// its workers), uploads their ds and writes the down flags. Steps 4 and 7's waits are each between two timing events:
// the GPU's exposed waits (lane copies; CPU experts), read at the layer's next window. A failure is kept (failed_) and
// still writes the flags, so no stream waits for ever. ME3, a replay (the graph executor): steps 0, 4 and 7's waits are
// kernels polling the layer's swap count and the flags (experts_cache.h wait_word, their times in mapped memory) and
// step 2's stamp a kernel's write, so a capture holds kernels and copies only: a launch of a graph with stream memory
// operations or event nodes can block in the driver until the GPU has run part of it, holding a lock the service's
// copies need while the GPU waits for them (load B's hang).
//
// Prefill (prompt_layer): PB3's prompt experts of a layer read one staged layer per device: its experts not resident
// are copied, on the copy stream, into one of the zone's two buffers (the layer's parity). At the prompt's first layer
// any pending pass is committed (the swap stream synchronized) and the zone's residents are evicted (their keys kept);
// at the next decode window (or drain) the zone is refilled with them as a pass (the same publication, kind refill).
//
// CPU workers: by default one a physical core of the process's affinity less one (at least one), each pinned to its
// core's first logical CPU (no SMT sibling). ME2b: the first core's first CPU takes the service thread (it drains the
// pool's phases with the workers), its second the decode thread (the caller of ffn_pair, pinned at its first window).
//
// ME3 (the cleanup): the cache is four units, each the one owner of its state, behind this facade (which keeps the
// settings, the request accounting, the round count and the executor's mode, and orders the units' calls):
//   - the residency (cache_residency.h): the backing, the pools, the slots and the word table; proposed placements
//     (an adapt pass's moves, a loan's restore set) become resident only at its publication boundary;
//   - the prompt loan (cache_loan.h): the lent region, the borrowed slots and their exact restore set, the staging, and
//     the bounded refill's progress, landing and stop;
//   - the window service (cache_service.h, cache_window.cpp): the windows' GPU work (the decode thread's or a
//     capture's) apart from the host service's (the service loop's thread), the CPU pool and the miss share;
//   - the service loop (service_loop.h): the service thread's queue, its producers and waits, the failure kept, the
//     abandon and the stop.
// Counters are kept per unit and summed here (counters, between requests).

#include "fn/cache_settings.h"
#include "fn/loader/loader.h"
#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/model/shapes.h"
#include "fn/runtime/cache_counters.h"
#include "fn/runtime/cache_passes.h"
#include "fn/runtime/expert_policy.h"
#include "fn/runtime/mx7_shadow.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fn::runtime {

struct expert_cache_options {
    int64_t pool_bytes = -1;                    // a device's GPU cache; -1: its free memory less reserve_bytes
    int64_t reserve_bytes = int64_t(768) << 20; // left free on each device (driver, transient and late allocations)
    // ME1 (the 16 GB plan on a larger card): a ceiling on a device's whole use (every allocation and the context, as
    // cudaMemGetInfo counts it: total less free); the pool shrinks so that use plus the reserve stays under it; -1 none
    int64_t device_cap_bytes = -1;
    int32_t cpu_workers = -1;                   // CPU product threads; -1: physical cores less one; 0: none
    int32_t pack_threads = 10;                  // threads packing the backing at init
    int32_t routed_rows = 256;                  // a one-half prompt of at most this many rows stages its routed experts
    bool zero_backing = false;                  // TEST (model-dry, run-dry): the backing left zero, nothing packed
    // TEST (--ex-native-hold, ME4's gate: the host-memory residency check, expires at ME4's disposition): with
    // zero_backing, the native expert pages still held as a load holds them (read in, locked when the memlock limit
    // allows, reported); not serving
    bool native_hold = false;
    // ME2 (--ex-profile PATH, --ex-profile-save PATH, --ex-swap-overlap 0|1; expire at ME2's disposition): the boot
    // ranking (empty: expert order), where save_profile writes the learned one (empty: nowhere), and the swaps' overlap
    std::string profile;
    std::string profile_save;
    bool swap_overlap = true;
    double share0 = 0.30;              // the PCIe share of the misses at start
    // ME3: init calls it once its windows' buffers, events and streams exist, before the pool is measured (the graph
    // executor's capture, model::service_hook); null: none
    model::service_hook before_pool;
    policy::adapt_options adapt;       // Strata's starting policy (expert_policy.h), not swept
    // ME3 (fn/cache_settings.h; the settings name ME3): the split's schedule; refine false keeps share0 (split-auto 0,
    // split-fixed)
    policy::split_options split;
    bool refine = true;
    // ME3, threads: the workers' CPUs (empty: one a physical core but the first's, each core's first CPU; cpu_workers
    // then their count), the service's and decode's CPUs (-1: one card, core 0's first and second CPUs; two cards
    // unpinned), the helpers' mask (empty: the original allowed mask less the workers', service and decode CPUs, else
    // less the service and decode CPUs), the workers' spin before sleeping
    std::vector<int32_t> worker_cpus, helper_cpus;
    int32_t service_cpu = -1, decode_cpu = -1;
    int32_t park_ms = 20;
    // ME3, the prompt's loan (one card's lent region and the zone's staging buffers): the largest loan's share of the
    // pool's slots at most lend_pct %, min_keep slots never lent; lend_region false keeps PB1's arena and PF1's
    // staging in their own allocations (two cards have none to lend); the refill's copies overlap decode
    // (refill_overlap; else copied and published synchronously at the return), submitted by the service at most
    // refill_batch_layers layers at a time
    int32_t lend_pct = 85, min_keep = 128;
    bool lend_region = true, refill_overlap = true;
    int32_t refill_batch_layers = 4; // MX3 (was 1); each batch at most 64 MiB a device (prompt_loan::options)
    // ME4: a one-half prompt run's ring (0: ME3's two layer buffers), its groups' copied bytes at most, and the cover
    // (per mille of a layer's nonresident bytes) after which the run's later layers stream ahead (prompt_plan.h)
    int64_t ring_bytes = int64_t(256) << 20, group_bytes = int64_t(64) << 20;
    int32_t route_cover = 875;
    // MX7 (fn/runtime/prompt_share.h): the prompt CPU share (-1 auto, 0 off, else fixed in (0, 1]; one device only)
    // and the chunk rows below which it serves
    double prompt_cpu_share = -1.0;
    int32_t prompt_cpu_share_max = 1024;
    int32_t prompt_cpu_share_test_maxt = 8; // TEST (run-dry's --mx7-dry-maxt): the candidates' pair bound
};

// ME3: the settings (fn/cache_settings.h) into the options; the pre-ME3 fields (pool, reserve, cap, workers, profile,
// swap overlap) stay the caller's
void apply_settings(const fn::settings::values & v, expert_cache_options & o);

// the process's allowed logical CPUs grouped by physical core (package, core), cores in order of their first CPU
std::vector<std::vector<int32_t>> physical_cores();

// ME3: the helpers' default mask: the allowed CPUs less the workers', service and decode CPUs; else (all taken) less
// the service and decode CPUs; else the allowed mask. Empty only when allowed is
std::vector<int32_t> helper_mask(const std::vector<int32_t> & allowed, const std::vector<int32_t> & workers,
                                 int32_t service_cpu, int32_t decode_cpu);

// ME3: the given CPUs (workers, helpers, the service's and decode's when not -1) each in the allowed mask; false and
// err naming the first that is not
bool cpus_allowed(const std::vector<int32_t> & allowed, const expert_cache_options & o, std::string & err);

class expert_cache final : public model::expert_service {
  public:
    expert_cache();
    ~expert_cache() override;
    expert_cache(const expert_cache &) = delete;
    expert_cache & operator=(const expert_cache &) = delete;

    // after the load (load_for_run with expert_cache) and every other device allocation of the run: the backing packed
    // from the mapping and registered, each device's lane, tables and buffers, then its pool filled from the profile,
    // the link probed, and the service and workers started. Synchronous.
    bool init(model::model & m, const expert_cache_options & o, std::string & err);

    bool ffn_pair(model::card_ctx * const c[2], const model::layer_weights * const w[2], int32_t rows,
                  fn::experts::gate_up_args ga[2], fn::experts::down_args da[2], std::string & err) override;
    // ME3, the graph executor (model::expert_service): card c's part of a window in a two-device capture; a replay's
    // host work before its launch (pass_abandon: after a failed one). set_executor selects which executor's windows
    // the cache serves (eager: ffn_pair's per-window host work; graph: pass_begin and pass_posted before each replay),
    // the service idle
    bool ffn_card(model::card_ctx & c, const model::layer_weights & w, int32_t rows, fn::experts::gate_up_args & ga,
                  fn::experts::down_args & da, std::string & err) override;
    bool pass_begin(void * const stream[2], std::string & err) override;
    bool pass_posted(int32_t rows, std::string & err) override;
    void pass_abandon(const std::string & why) override;
    void set_executor(bool graph) override;
    bool graph_executor() const { return graph_; }
    bool prompt_layer(int32_t layer, int32_t rows, const int32_t * ids, bool more, void * const stream[2],
                      std::string & err) override;
    const uint8_t * const * prompt_table(int32_t layer, int32_t card, bool down) const override;
    // ME4: the ring (model::expert_service; cache_loan.h)
    void set_prompt_ring(int32_t rows) override;
    void set_me4_base(bool on) override; // ME4: base's prompt staging and CPU path (main's), else ME4's
    bool cpu_native() const;              // ME4: the CPU experts read the native views (else the reconstruction)
    int32_t prompt_groups(int32_t layer) const override;
    bool prompt_group(int32_t layer, int32_t g, void * const stream[2], model::prompt_group_view & v,
                      std::string & err) override;
    bool prompt_group_done(int32_t layer, int32_t g, void * const stream[2], std::string & err) override;
    // MX7: the prompt CPU share (prompt_share.h); TEST ONLY set_prompt_cpu_share (--mx7-check), between requests
    bool set_prompt_cpu_share(double share, std::string & err);
    // MX7 TEST ONLY (--mx7-check's "shadow"): the share's shadow comparison (prompt_share::set_shadow), between
    // requests
    void set_prompt_cpu_share_shadow(bool on);
    std::vector<mx7_shadow_pair> take_prompt_cpu_share_shadow();
    bool prompt_share_begin(int32_t layer, const uint16_t * x16, void * const stream[2], std::string & err) override;
    const uint8_t * prompt_share_skip(int32_t layer) const override;
    bool prompt_share_end(int32_t layer, const model::prompt_share_card card[2], void * const stream[2],
                          std::string & err) override;
    void prompt_share_settle(int32_t layer) override;
    const uint8_t * prompt_bank(int32_t card) const override;
    // ME2b item 6: on one card, PF1's staging and PB1's arena are a region of the pool at its tail (lent_off_), whose
    // slots are zone slots: filled at boot, retired with the zone when a prompt borrows the region, refilled after it.
    // Held, the zone is not returned (ffn_pair, drain) and reset_to_profile refuses.
    bool prompt_lent() const override;
    bool prompt_hold(bool on, void * const stream[2], const std::vector<model::byte_span> * need,
                     std::string & err) override;
    bool prompt_held() const;
    // ME3: the startup check of the largest loan (model::expert_service), through loan_plan
    bool prompt_loan_check(const std::vector<model::byte_span> & need, std::string & err) override;
    // ME3: short reads' windows counted apart; ME3's base arm (expires at ME3's disposition): ME2b's whole loan, its
    // refill enqueued by the decode caller and the helpers' inherited mask
    void set_short_read(bool on) override { short_read_ = on; }
    void set_legacy(bool on) override;
    bool legacy() const override { return legacy_; }
    // ME3: the request's accounting (model::expert_service): the counters, shares and refill state at its entry; its
    // line is the settings and the deltas since
    void request_begin() override;
    std::string request_json() override;
    // ME3 (Fable's load G ruling; Strata #267's Verifier::release_gpu_waits adapted, its notice in expert_cache.cpp):
    // model::expert_service's release. The service's kept failure set (its waiting jobs end, every later call
    // refused), then every flag and swap-count word the devices' waits read written again each millisecond on a
    // release stream of its own until the decode streams the cache last saw, the copy streams and the swap streams
    // drain or timeout_ms passes. only_if_busy (teardown): nothing when those streams are idle, and the decode streams
    // forgotten after (their owner goes next)
    bool release_gpu_waits(int32_t timeout_ms, bool only_if_busy, std::string & note) override;
    // ME3: the settings' read-back the request lines carry (the caller's, fn/cache_settings.h)
    void set_settings_json(const std::string & j) { settings_json_ = j; }
    // TEST (the loan's integrity check): every zone slot holding a key compared with its backing bytes on each device;
    // checked the slots compared, bad those that differ. Synchronous, between requests; no drain, so a refill not yet
    // published is not checked (its slots hold no key)
    bool test_verify_zone(int64_t & checked, int64_t & bad, std::string & err);
    int64_t zone_slots() const; // the prompt zone's slots
    // ME3: the helpers' CPUs (model::expert_service), derived at init from the original allowed mask
    const std::vector<int32_t> * helper_cpus() const override;
    int32_t service_cpu() const;
    int32_t decode_cpu() const;
    // TEST (the poison check): every borrow fills the borrowed slots with 0xFF before the prompt writes them
    void test_poison_lent(bool on);
    // TEST (run-dry's outstanding-refill check, ME3's cleanup): at most limit batches of each refill submitted (-1
    // none); lifted before a drain
    void test_refill_limit(int32_t limit);
    // TEST (--me9-check, ME9): the admission reservations held now; every reserved slot poisoned (0xFF); every
    // admitted resident's bytes against its backing (synchronous, between requests)
    int32_t test_adm_held() const;
    bool test_adm_poison(std::string & err);
    bool test_adm_verify(int64_t & checked, int64_t & bad, std::string & err);
    // TEST (--me9-check, Shell 0a1b4eb5 and request 11): a known eligible GPU miss a layer. Of each layer's last decode
    // window's experts, the most used resident is evicted (n gets the count) and tracked; every miss GPU-served (mode 0
    // kept) and the adapt pass held (no swap returns an evicted key) until test_adm_force_end. test_adm_forced: each
    // tracked key's lifecycle (residency::adm_test_forced)
    bool test_adm_force(int32_t & n, std::string & err);
    void test_adm_force_end();
    void test_adm_forced(int32_t & n_admitted, int32_t & n_replaced, std::string & failures) const;
    // TEST (ex-test's lent-region arm): the region's offset in the pool and its bytes; a slot's device address
    int64_t lent_offset() const;
    int64_t lent_bytes() const;
    const uint8_t * test_pool(int32_t d) const;
    // TEST (spec run's stall check, ME3 load G ruling; expires at ME3's disposition): the service's n-th window with
    // misses from now withheld (window_service::test_withhold); the withheld window's layer, -1 before
    void test_withhold(int32_t n);
    int32_t test_withheld() const;

    // every window served, every queued copy finished, the pass committed and the zone refilled (between requests,
    // after a failure or abandoned prompt, before destruction); the learned profile saved (--ex-profile-save). Reports
    // a kept failure. Synchronous.
    bool drain(std::string & err);
    // ME2: the boot state again: the profile's placement uploaded and copied, the usage, heat, rounds and share reset;
    // the counters stay. Synchronous.
    bool reset_to_profile(std::string & err);
    // ME2: the learned ranking to opts' profile_save (nothing without one). Synchronous.
    bool save_profile(std::string & err);

    // TEST (spec run's checks): how decode misses are served from now on: 0 the split; 1 every miss on the CPU, 2
    // every miss a GPU copy (no usage, swaps or share change in either). 1 needs CPU workers
    bool set_mode(int32_t mode, std::string & err);
    // the units' counters summed (a snapshot, between requests: the service idle)
    const expert_cache_counters & counters() const;
    void reset_counters();
    // the memory plan (pools, classes and slots, zone, lanes, tables and buffers, backing, pinned bytes, workers)
    std::string plan_text() const;
    // the counters as one JSON object
    std::string counters_json() const;
    std::string counters_json(const expert_cache_counters & c, size_t share_from) const;
    int32_t devices() const;
    int32_t device_id(int32_t d) const;
    int64_t pool_bytes() const;            // a device's
    int64_t device_bytes(int32_t d) const; // all the cache's bytes on device d
    int64_t pinned_bytes() const;
    int64_t backing_bytes() const;
    int32_t cpu_workers() const;
    int32_t resident() const; // entries resident now
    double share() const;
    const std::vector<share_attempt> & share_attempts() const;

  private:
    struct parts; // the units (expert_cache.cpp)
    // the round boundary (a layer-0 window, the service idle): the swap waits read, the pending pass committed when it
    // has landed, the refill's landed moves published, the share refined, then every adapt_every rounds with nothing
    // pending, no refill outstanding and no loan the adapt pass chosen and published
    bool round_start(void * const stream[2], std::string & err);
    bool admission_round(void * const stream[2], bool failed_now, std::string & err); // ME9 (round_start's last step)
    // the loan: the pending pass committed, an outstanding refill stopped, the learned profile saved (between
    // requests), then the loan; the return: the pending pass committed, then the return
    bool zone_take(void * const stream[2], const std::vector<model::byte_span> * need, std::string & err);
    bool zone_return(void * const stream[2], std::string & err);
    bool failed(std::string & err); // the service's kept failure as err ("a window failed: ...")
    void release();

    std::unique_ptr<parts> p_;
    model::model * m_ = nullptr;
    expert_cache_options o_;
    int64_t round_ = 0;
    bool short_read_ = false, legacy_ = false, graph_ = false;
    bool test_hold_adapt_ = false;    // TEST: test_adm_force
    std::vector<int32_t> test_forced_; // TEST: test_adm_force's evicted keys
    void * decode_[2] = {nullptr, nullptr}; // ME3 (load G ruling): the decode streams last seen (the target's)
    expert_cache_counters n_;            // the facade's: rounds, the executor's windows and replays, short reads
    mutable expert_cache_counters sum_;  // counters' snapshot
    // ME3: the request's state at its entry, the settings' read-back
    expert_cache_counters req_n_;
    size_t req_share_ = 0;
    int64_t req_rf_keys_ = 0, req_rf_bytes_ = 0, requests_ = 0;
    std::string settings_json_;
};

} // namespace fn::runtime
