#pragma once

// ME3 (the cleanup): the expert cache's decode windows (expert_cache.h describes them). The one owner of:
//   - the windows' device side: each device's lane, the flags, the second gate_up's meta, the table patches, card 0's
//     CPU products, the windows' events and their timing, a replay's stamps and its flag waits' times;
//   - the host service: the svc::loop (its thread, queue, stamps' jobs, failure and stop), the window plan, the
//     pinned staging (IDs, input, CPU products, activations), the CPU pool, the threads' CPUs, the miss share
//     (PcieAuto) and the mode.
// Two halves, apart: enqueue is the decode thread's (or a capture's) GPU work for a window and touches only device
// work and the per-layer parity; the svc::work calls (await, serve, unblock) are the service thread's. The service
// reads the residency's published state (and counts usage, mode 0) while windows run; the decode thread changes it
// only with the service idle. The refill's batches are the loan's (svc::refill); the service submits them when it
// has no window and yields to a queued window.

#include "fn/kernels/experts/experts_cpu.h"
#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/runtime/cache_counters.h"
#include "fn/runtime/cache_loan.h"
#include "fn/runtime/cache_residency.h"
#include "fn/runtime/cache_util.h"
#include "fn/runtime/expert_policy.h"
#include "fn/runtime/service_loop.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime {

class window_service final : public svc::work {
  public:
    struct options {
        int32_t cpu_workers = -1;
        std::vector<int32_t> worker_cpus, helper_cpus;
        int32_t service_cpu = -1, decode_cpu = -1, park_ms = 20;
        double share0 = 0.30;
        bool refine = true;
        policy::split_options split;
    };
    // ME3: the flags' set value (experts_cache.h resolve_args::set): resolve resets them, the service or resolve sets
    static constexpr uint32_t flag_set = 1;

    window_service() = default;
    ~window_service() override { stop(); }
    window_service(const window_service &) = delete;
    window_service & operator=(const window_service &) = delete;

    // init (before the pool): the pinned staging, the lane, flags, meta, patches and products, the events, the flags'
    // stream memory operations checked on each copy stream. start (after the boot fill): the CPU workers on their
    // CPUs (cores and the allowed mask read before the decode thread pins itself), the share, the service thread on
    // its CPU and the helpers' mask
    bool init(const cache::devices & dv, residency & r, const prompt_loan & ln, std::string & err);
    bool start(const options & o, const std::vector<std::vector<int32_t>> & cores, prompt_loan & rf, std::string & err);
    void set_helpers(const std::vector<int32_t> & cpus) { helper_cpus_ = cpus; } // the cache's derivation
    const std::vector<int32_t> & worker_cpus() const { return worker_cpus_; }
    void stop(); // the service joined (every queued job ended), then the pool
    void release(const cache::devices & dv) { own_.release(dv); }
    svc::loop & loop() { return loop_; }

    // the decode thread. enqueue: the window's GPU work on the devices of only (-1: both cards; else that card's
    // device, two devices); cap: inside a capture (the waits kernels, the stamp a kernel's write, the window times at
    // parity 0); desc set once its descriptor (event or stamp) is enqueued
    bool enqueue(model::card_ctx * const c[2], const model::layer_weights * const w[2], int32_t rows,
                 fn::experts::gate_up_args ga[2], fn::experts::down_args da[2], int32_t only, int32_t par, bool cap,
                 bool & desc, std::string & err);
    int32_t next_par(int32_t layer) { return int32_t(wpar_[size_t(layer)]++ & 1); }
    bool decode_pinned() const { return decode_pinned_; }
    void pin_decode();
    // the replay boundary (the service idle): the stamps cleared (every window of the previous replay was served), the
    // previous replay's window times (written before its round's completion record)
    void clear_stamps();
    void read_replay_times();
    void read_all_times(); // drain: every device synchronized
    // the round boundary (the service idle): the share refined (PcieAuto) from the pool's readings since
    void refine_share(int64_t completed);
    void reset_share(); // reset_to_profile
    bool set_mode(int32_t mode, std::string & err); // the service idle
    int32_t mode() const { return mode_; }

    // ME3 (load G ruling): each device's flags ([48][2], device memory) for the cache's release of the GPU's waits
    uint32_t * flags(int32_t d) const { return flags_[d]; }
    // TEST (spec run's stall check, ME3 load G ruling; expires at ME3's disposition): the n-th served window from now
    // with misses (lane or CPU work) is withheld: neither its copies, its CPU work nor its flags (0: off). withheld:
    // that window's layer, -1 before
    void test_withhold(int32_t n) { withhold_.store(n); }
    // TEST (--me9-check, Shell 0a1b4eb5): every miss GPU-served (the lane, or a reserved slot when the round admits)
    // while the split's mode stays 0; set between requests, the service idle
    void test_gpu_all(bool on) { gpu_all_.store(on); }
    // TEST (--me9-check): layer's last window's row-0 experts (the host copy of its routing; the service idle)
    void test_routed(int32_t layer, int32_t * ids10) const {
        for (int32_t i = 0; i < 10; ++i) {
            ids10[i] = ids_h_[layer * max_b * 10 + i];
        }
    }
    // ME4: the CPU jobs read the residency's native views (on: only when it has them ready) or the reference path's
    // reconstruction of the packed backing (main's path: --me4-arms base). Set between requests, the service idle
    void set_native(bool on) { native_ = on; }
    // ME9: whether this round's lane misses may land in reserved slots (expert_cache::admission_round, at the
    // boundary, the service idle); the plan reads it, never the loan's or refill's state
    void set_admit(bool on) { admit_.store(on); }
    bool native() const { return native_; }
    int32_t test_withheld() const { return withheld_.load(); }

    // svc::work (the service thread)
    bool await(const svc::job & j, const std::atomic<bool> & quit, std::string & err) override;
    bool serve(const svc::job & j, std::string & err) override;
    void unblock(const svc::job & j) override;

    // reporting (between requests)
    std::string plan_share(double link_gbps) const; // the share line
    std::string plan_threads() const;               // the threads, helpers and waits lines
    std::string cpu_list() const;                   // the workers' CPUs and masks
    const expert_cache_counters & counters() const { return n_; }
    void reset_counters() { n_ = expert_cache_counters(); }
    int64_t device_bytes(int32_t d) const { return own_.dev_bytes[d]; }
    int64_t host_bytes() const { return own_.host_bytes; }
    int64_t lane_bytes() const { return lane_bytes_; }
    int32_t workers() const { return cpu_.workers(); }
    // MX7: the pool for the prompt CPU share's thread, its only caller while a prompt is read (no window runs then)
    fn::experts::cpu::pool & cpu_pool() { return cpu_; }
    bool unit_timing() const { return cpu_.unit_timing(); }
    double share() const { return share_; }
    const std::vector<share_attempt> & share_log() const { return share_log_; }
    const std::vector<int32_t> & helper_cpus() const { return helper_cpus_; }
    int32_t service_cpu() const { return service_cpu_; }
    int32_t decode_cpu() const { return decode_cpu_; }
    uint64_t service_mask() const { return service_mask_; }
    uint64_t decode_mask() const { return decode_mask_; }

  private:
    static constexpr int32_t max_u = fn::experts::max_slots;
    static_assert(max_u <= 50, "ME5: miss_split's bound (split::max_misses) and the arrays below");
    static constexpr int32_t max_b = fn::experts::max_rows;
    struct win_rec {
        bool live = false, lane = false, cpu = false;
        bool refill = false; // ME3: a refill outstanding while it was served
        bool graph = false;  // ME3: a replay's window (its wait times in wt_h_, not events)
        int64_t lane_bytes = 0; // ME5 rider: its lane and admission copies' bytes, every device
    };
    // a replay's flag waits' globaltimer pairs in mapped host memory, per device [layer][parity][flag][2]
    static constexpr int32_t wt_n = 48 * 2 * 2 * 2;
    static int32_t wt_flag(int32_t layer, int32_t par, int32_t f) { return ((layer * 2 + par) * 2 + f) * 2; }
    void plan(int32_t layer, int32_t rows);
    bool write_flags(const svc::job & j, int32_t which, std::string & err); // which: 0 gate/up, 1 down, 2 both
    void read_window_times(int32_t layer, int32_t par);
    cache::clk::time_point await_t0_; // the current job's await (its descriptor wait, counted when served)

    const cache::devices * dv_ = nullptr;
    residency * res_ = nullptr;
    const prompt_loan * loan_ = nullptr;
    cache::owned own_;
    svc::loop loop_;
    options o_;
    // device buffers (per device)
    uint8_t * lane_[2] = {nullptr, nullptr};
    int64_t lane_bytes_ = 0;
    uint32_t * flags_[2] = {nullptr, nullptr}; // [48][2]: gate/up, down
    uint8_t * meta2_[2] = {nullptr, nullptr};  // per card: its second gate_up's meta (not down's)
    int32_t * apply_idx_[2] = {nullptr, nullptr};
    uint64_t * apply_val_[2] = {nullptr, nullptr};
    float * ext_ds_ = nullptr; // ME3: card 0's device: the CPU's ds [50][2560], the combine's
    // pinned host buffers
    int8_t * x_codes_h_ = nullptr; // [48][5][2560] (the window's staging, per layer)
    float * x_scales_h_ = nullptr; // [48][5][80][2]
    int32_t * ids_h_ = nullptr;    // [48][5][10]
    float * ds_h_ = nullptr;       // [50][2560] the CPU experts' products
    int32_t * apply_idx_h_[2] = {nullptr, nullptr};
    uint64_t * apply_val_h_[2] = {nullptr, nullptr};
    fn::experts::cpu::block_q8_k * act_ = nullptr; // [5][10] the window's gate/up activations
    // ME3 (graph executor): per device, per layer, a replay's window stamp in mapped host memory: clear_stamps clears
    // it, the replay writes 1 after the window's descriptor copies (a kernel's write after its stream's earlier work),
    // the service's job waits for it. stamp_d_: each device's address of it. wt_h_: the flag waits' times
    uint32_t * stamp_h_ = nullptr; // [2][48]
    uint32_t * stamp_d_[2] = {nullptr, nullptr};
    uint64_t * wt_h_ = nullptr; // [2][wt_n]
    uint64_t * wt_d_[2] = {nullptr, nullptr};
    // events: per device, per layer, the window's resolve and staging enqueued; per device, [layer][parity][4]
    // (timing): around the compute stream's gate/up flag wait, then its down flag wait; [layer][parity][2] (timing):
    // around the window's lane copies on the copy stream. A window's parity alternates per layer; the service reads a
    // window's times at the layer's next window (both complete by then)
    std::vector<void *> ev_desc_[2], ev_fw_[2], ev_lane_[2];
    std::vector<win_rec> win_;   // [layer][parity] (the service's)
    std::vector<uint32_t> wpar_; // per layer: windows posted (the decode thread's)
    // the window's plan (the service's)
    int32_t n_u_ = 0, n_slots_ = 0;
    int32_t e_[50] = {}, mask_[50] = {}, base_[50] = {};
    int8_t how_[50] = {};    // 0 resident (hit), 1 lane, 2 CPU, 3 (ME9) a reserved slot: admitted
    int32_t adm_rec_[50] = {}; // ME9: how 3's admission record
    int32_t n_adm_ = 0;        // ME9: this window's admitted misses (GPU-served, copied into reserved slots)
    std::atomic<bool> admit_{false};
    std::atomic<bool> gpu_all_{false}; // TEST: test_gpu_all
    int64_t where_[50] = {}; // pool or lane byte offset (a device's)
    bool bad_ids_ = false;   // an ID outside [0, 512): the resolve wrote no flag
    bool native_ = false;    // ME4: set_native
    int32_t n_lane_ = 0, n_cpu_ = 0;
    fn::experts::cpu::job jobs_[50];
    std::atomic<int32_t> withhold_{0}, withheld_{-1}; // TEST (the stall check): test_withhold
    // the miss share (the round boundary's, the service idle) and the pool's readings since init (the service's)
    policy::pcie_auto auto_;
    double share_ = 0.30, link_gbps_ = 0;
    int32_t share_num_ = 77;
    std::vector<share_attempt> share_log_;
    double cpu_ms_all_ = 0; // ME5 rider: the CPU leg's input conversion plus pool phases
    int64_t cpu_jobs_all_ = 0;
    double lane_ms_all_ = 0; // ME5 rider: the lane leg's device-0 copy-event time and the bytes of those intervals
    int64_t lane_bytes_all_ = 0;
    int32_t mode_ = 0;
    // CPU workers and threads
    fn::experts::cpu::pool cpu_;
    std::vector<int32_t> worker_cpus_; // the CPU each worker is pinned to, -1 unpinned
    int32_t service_cpu_ = -1, decode_cpu_ = -1;
    uint64_t service_mask_ = 0, decode_mask_ = 0;
    std::vector<int32_t> helper_cpus_;
    bool decode_pinned_ = false;
    expert_cache_counters n_; // the service's: windows, misses, CPU work, GPU waits
};

} // namespace fn::runtime
