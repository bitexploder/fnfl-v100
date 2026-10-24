#pragma once

// MX7: the prompt CPU share (Strata-V100 69713b61's set_cpu_pool and STRATA_PREFILL_CPU_SHARE, src/prefill/
// prefill.cpp 178-225, 3333-3720 and 4395-4410; rewritten as FNFL code, nothing borrowed). One device only (the
// single-card cached configuration); two cards never arm it. On a route-first ring layer (prompt_loan's) of a chunk
// below max_rows rows, the CPU pool (experts_cpu.h), idle while a prompt is read, takes whole some of the layer's
// routed nonresident experts that at most pplan::share_maxt pairs route to (pplan::share_pick, fewest first); the
// ring neither stages nor the GPU computes them. Per layer, in the compute stream's order:
//   pick   (prompt_loan::prompt_layer, after the routing sync): the previous layer's events read (complete by then)
//          into the running means and the self-check, then this layer's side and its CPU experts, which the ring
//          plan leaves out;
//   begin  (prompt.cpp, after prompt_layer): the expert work's first event; the skip mask and the pairs uploaded;
//          the token rows the CPU's experts read gathered (es_gather) and downloaded on the share's own stream; the
//          CPU job posted to the share's thread, which waits for the download, converts the rows (gate_up_act_f16),
//          runs the pool on the native views and writes the products as F16 (to_f16);
//   skip   the device mask es_drop takes after each card's ep_group;
//   end    (prompt.cpp, after the last ring group): the GPU work's event, the CPU job joined, its rows uploaded and
//          written into each card's partials (es_add: card 0 the products, card 1 zeros), the expert work's last
//          event; ep_sum then weights them in route order.
// The settings: share -1 (auto: share_of the running means c, the CPU's ms an expert, and g, the GPU's ms a staged
// expert, measured on sharing layers; and the self-check, pplan::share_check, on every eligible layer), 0 (off: no
// buffers, nothing changes) or x in (0, 1] (fixed: every eligible layer shares x, no self-check, so a fixed x repeats
// run to run). The share's host buffers are pinned when first needed, mid-request; one refused turns the share off
// for the run (one line on stderr) and the GPU takes every expert. The CPU's products differ from the GPU's in their
// arithmetic (Q8_K activations, the CPU kernels' rounding): a lossy split, accepted for this path by the operator
// (tickets/MX7-strata-prompt-cpu-share.md). The CPU jobs read the native views (residency::native); only with the
// backing left zero (zero_backing tests, where no views are held) do they read the packed backing.

#include "fn/kernels/experts/experts_cpu.h"
#include "fn/model/layer.h"
#include "fn/runtime/cache_counters.h"
#include "fn/runtime/cache_residency.h"
#include "fn/runtime/cache_util.h"
#include "fn/runtime/mx7_shadow.h"
#include "fn/runtime/prompt_plan.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fn::runtime {

class window_service;

class prompt_share {
  public:
    struct options {
        double share = -1.0;                      // -1: auto; 0: off; else the fixed share, (0, 1]
        int32_t max_rows = pplan::share_max_rows; // chunks of fewer rows share
        // TEST (run-dry, zero weights: --mx7-dry-maxt): the candidates' pair bound in place of MAXT, so the zero
        // weights' few, heavily routed experts reach the CPU (an expert then spans several jobs)
        int32_t maxt = pplan::share_maxt;
    };
    static constexpr int32_t max_pairs = cache::n_ex * pplan::share_maxt; // a layer's CPU pairs at most

    prompt_share() = default;
    ~prompt_share() { release(); }
    prompt_share(const prompt_share &) = delete;
    prompt_share & operator=(const prompt_share &) = delete;

    // at the cache's init, before the pool's measurement (so its device bytes are in the device's use): one device and
    // a share other than 0 arm it (its device buffers, stream and events); else nothing is allocated
    bool init(const cache::devices & dv, const residency & r, const options & o, std::string & err);
    // after the pool's workers started: the share's thread (the pool's caller during a prompt) on the service's CPU,
    // idle while a prompt is read; no workers: the share stays off
    bool start(window_service & w, std::string & err);
    void release();
    bool armed() const { return armed_ && !dead_ && o_.share != 0.0; }
    // TEST ONLY (fnfl-drive spec run --mx7-check), between requests: the share (-1 auto, 0 off, else fixed), its means,
    // self-check and measurement started afresh; a share other than 0 needs one armed at init
    bool set_share(double share, std::string & err);
    // TEST ONLY (--mx7-check's "shadow", mx7_shadow.h), between requests: on, a sharing layer's CPU experts stay in the
    // ring and the GPU's work (no mask, no add-back) and each CPU pair is compared with the GPU's; take collects them
    void set_shadow(bool on) {
        shadow_ = on;
        sh_out_.clear();
    }
    std::vector<mx7_shadow_pair> take_shadow() { return std::move(sh_out_); }

    int32_t pick(int32_t layer, int32_t rows, const int32_t * ids, const int32_t * counts, const uint8_t * resident,
                 void * const stream[2], uint8_t * cpu);
    bool begin(int32_t layer, const uint16_t * x16, void * const stream[2], std::string & err);
    const uint8_t * skip(int32_t layer) const;
    bool end(int32_t layer, const model::prompt_share_card card[2], void * const stream[2], std::string & err);
    // every exit of a layer begun (prompt.cpp): its job, if posted, joined (the pool back in no one's hands) and its
    // plan dropped; nothing after an end that succeeded
    void settle(int32_t layer);

    expert_cache_counters counters() const { return n_; }
    void reset_counters() { n_ = expert_cache_counters(); }
    std::string setting() const; // "auto", "0" or the fixed share
    std::string plan_line() const;
    std::string state_json() const; // the share in force, c, g and the self-check now
    int64_t device_bytes(int32_t d) const { return d == 0 ? dev_bytes_ : 0; }
    int64_t host_bytes() const { return host_bytes_; }

  private:
    void worker();
    void join(); // the CPU job posted, if any, finished
    bool host_buffers(int32_t rows, int32_t pairs);
    void off(const char * what);

    options o_;
    const cache::devices * dv_ = nullptr;
    const residency * res_ = nullptr;
    fn::experts::cpu::pool * pool_ = nullptr;
    bool armed_ = false, dead_ = false, native_ = false;
    int32_t cpu_ = -1; // the thread's CPU
    // device: the gathered rows F16 [max_rows][2560], the products F16 [max_pairs][2560], the pairs, the rows, the mask
    uint16_t * x16_d_ = nullptr;
    uint16_t * out16_d_ = nullptr;
    int32_t * pairs_d_ = nullptr;
    int32_t * rows_d_ = nullptr;
    uint8_t * mask_d_ = nullptr;
    int64_t dev_bytes_ = 0;
    // host, pinned when first needed: the same rows and products, the pairs, the rows, the mask
    uint16_t * x16_h_ = nullptr;
    uint16_t * out16_h_ = nullptr;
    int32_t * pairs_h_ = nullptr;
    int32_t * rows_h_ = nullptr;
    uint8_t * mask_h_ = nullptr;
    int32_t cap_rows_ = 0, cap_pairs_ = 0;
    int64_t host_bytes_ = 0;
    void * side_ = nullptr;                        // the download's stream
    void * ev_ready_ = nullptr;                    // the layer's input written (compute stream)
    void * ev_x_ = nullptr;                        // its rows downloaded (the side stream; a blocking event)
    void * ev_t_[3] = {nullptr, nullptr, nullptr}; // the expert work's start, the GPU's end, the add-back's end

    // this layer (pick to end)
    struct layer_plan {
        int32_t layer = -1;
        bool eligible = false, on = false;
        int32_t n_cpu = 0, jobs = 0, pairs = 0, rows = 0, nonres = 0;
    } cur_;
    std::vector<fn::experts::cpu::job> jobs_;
    std::vector<fn::experts::cpu::block_q8_k> act_;
    std::vector<float> out32_;
    fn::experts::fmt gu_ = fn::experts::fmt::iq3_xxs, dn_ = fn::experts::fmt::q2_0;
    // the last eligible layer's measurement, read at the next pick
    struct pending {
        bool valid = false, on = false;
        int32_t n_cpu = 0, staged = 0, nonres = 0;
        double cpu_ms = 0;
    } pend_;
    double c_ = 0, g_ = 0; // auto's running means: the CPU's ms an expert, the GPU's a staged expert
    int64_t nc_ = 0, ng_ = 0;
    double share_now_ = 0.5;
    pplan::share_check check_;
    bool chunk_shared_ = false;
    // TEST (set_shadow): the layer's routing and compact rows kept from pick, its input from begin, the comparisons
    bool shadow_ = false;
    std::vector<int32_t> sh_ids_, sh_compact_;
    const uint16_t * sh_x16_ = nullptr;
    std::vector<mx7_shadow_pair> sh_out_;

    // the thread
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool go_ = false, done_ = true, stop_ = false;
    bool job_ok_ = true, job_bad_ = false;
    double job_ms_ = 0;

    expert_cache_counters n_;
};

} // namespace fn::runtime
