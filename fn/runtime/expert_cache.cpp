// ME1's expert cache with ME2's residency and miss path (expert_cache.h). ME3 (the cleanup): the facade over the
// units; it keeps the settings, the request accounting, the round count and the executor's mode, and orders the units'
// calls. The swap rule, the profile and the share are Strata-V100 3d4ce4e0's (expert_policy.cpp carries its MIT
// notice); the pass ordering follows its STRATA_ADAPT_OVERLAP path (src/program/generate.cpp near lines 9540-9900) on
// FNFL's streams.

#include "fn/runtime/expert_cache.h"

#include "fn/runtime/cache_parts.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <thread>

namespace fn::runtime {

using cache::cuda_ok;
using cache::fail;

namespace {

bool stream_capturing(void * stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &st) == cudaSuccess &&
           st == cudaStreamCaptureStatusActive;
}

} // namespace

expert_cache::expert_cache() : p_(new parts()) {}

expert_cache::~expert_cache() {
    if (m_ != nullptr && !o_.profile_save.empty()) {
        std::string ignored;
        if (!save_profile(ignored)) {
            std::fprintf(stderr, "%s\n", ignored.c_str());
        }
    }
    release();
}

// the service stopped (every queued job ended: no stream is left waiting on a flag) and the pool, then every device
// idle (every queued consumer and host callback done), then each unit's buffers, events and streams
void expert_cache::release() {
    parts & p = *p_;
    std::string note; // ME3 (load G ruling): no device synchronize waits on a flag nobody writes
    if (!release_gpu_waits(5000, true, note) || !note.empty()) {
        std::fprintf(stderr, "expert cache: release: %s\n", note.c_str());
    }
    p.win.stop();
    for (int32_t d = 0; d < 2; ++d) {
        if (p.dv.dev[d] >= 0) {
            cudaSetDevice(p.dv.dev[d]);
            cudaDeviceSynchronize();
        }
    }
    p.share.release(); // MX7: its thread (idle) before the pool's
    p.win.release(p.dv);
    p.loan.release(p.dv);
    p.res.release(p.dv);
    for (int32_t d = 0; d < 2; ++d) {
        if (p.dv.copy[d] != nullptr) {
            cudaSetDevice(p.dv.dev[d]);
            cudaStreamDestroy(static_cast<cudaStream_t>(p.dv.copy[d]));
            p.dv.copy[d] = nullptr;
        }
    }
}

bool expert_cache::init(model::model & m, const expert_cache_options & o, std::string & err) {
    const cache::device_guard keep;
    if (!m.expert_cache || (m.single && m.device[0] != m.device[1]) || (!m.single && m.device[0] == m.device[1])) {
        return fail(err, "needs a load with the routed experts left to the cache (one device, or two)");
    }
    if (o.cpu_workers < -1 || o.cpu_workers > 64 || o.pack_threads < 1 || o.reserve_bytes < 0 || o.routed_rows < 0 ||
        o.routed_rows > cache::max_prompt_rows || !(o.share0 >= 0.0 && o.share0 <= 1.0) || o.adapt.every < 1 ||
        o.adapt.max_swaps < 0 || o.adapt.max_swaps > 4096 || !(o.adapt.decay >= 0.0f && o.adapt.decay <= 1.0f) ||
        o.split.windows.empty() || o.lend_pct < 1 || o.lend_pct > 100 || o.min_keep < 0 || o.refill_batch_layers < 1 ||
        o.refill_batch_layers > fn::n_layer || o.park_ms < 0 ||
        (o.ring_bytes != 0 && (o.ring_bytes < 2 * o.group_bytes || o.ring_bytes % 256 != 0)) ||
        o.group_bytes < (int64_t(4) << 20) || o.route_cover < 0 || o.route_cover > 1000 ||
        (!o.worker_cpus.empty() && o.cpu_workers >= 0 && o.cpu_workers != int32_t(o.worker_cpus.size())) ||
        !(o.prompt_cpu_share == -1.0 || (o.prompt_cpu_share >= 0.0 && o.prompt_cpu_share <= 1.0)) ||
        o.prompt_cpu_share_max < 1024 || o.prompt_cpu_share_max > cache::max_prompt_rows) {
        return fail(err, "options out of range (or --ex-cpu disagrees with --ex-worker-cpus' count)");
    }
    if (!cache::memops(err)) {
        return false;
    }
    // MEKV: the generic device account below excludes mapped history; retain only each owner's tier allocation.
    if (m.kv_host && !m.single) {
        const model::reserved_class * tier = m.state_class("KV tier (ME1)");
        for (int32_t c = 0; c < 2; ++c) {
            if (tier == nullptr || tier->buf[c].host() || tier->buf[c].device() != m.device[c]) {
                return fail(err, "two-card host KV has no device-local tier");
            }
            std::fprintf(stderr, "MEKV device %d: KV tier %.6f GiB (%zu bytes), full history in mapped host memory\n",
                         m.device[c], double(tier->buf[c].size()) / double(int64_t(1) << 30), tier->buf[c].size());
        }
    }
    m_ = &m;
    o_ = o;
    parts & p = *p_;
    cache::devices & dv = p.dv;
    dv.nd = m.single ? 1 : 2;
    dv.parts = dv.nd == 1 ? 4 : 2;
    dv.dev[0] = m.device[0];
    dv.dev[1] = dv.nd == 2 ? m.device[1] : -1;
    // ME2b item 3, one card only (Amendment 1: two-card runs unchanged): no CUDA wait on the device spins (stream and
    // device synchronizations block; event waits block where the event was made with cudaEventBlockingSync). The load
    // made the primary context already, so the runtime's flags call overwrites its scheduling flags (CUDA 12.8
    // cudaSetDeviceFlags); read back for the plan
    dv.quiet = dv.nd == 1;
    for (int32_t d = 0; d < dv.nd; ++d) {
        unsigned int fl = 0;
        if (!dv.set(d, err) ||
            (dv.quiet && !cuda_ok(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), "cudaSetDeviceFlags", err)) ||
            !cuda_ok(cudaGetDeviceFlags(&fl), "cudaGetDeviceFlags", err)) {
            return false;
        }
        dv.flags[d] = fl;
    }
    // the layers, classes, boot ranking and backing; the copy streams; each unit's buffers other than the pool and
    // the pass buffers, its events and streams
    if (!p.res.init_layers(m, dv.nd, o.profile, o.pack_threads, o.zero_backing, err, o.native_hold)) {
        return false;
    }
    for (int32_t d = 0; d < dv.nd; ++d) {
        cudaStream_t cs = nullptr;
        if (!dv.set(d, err) || !cuda_ok(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "copy stream", err)) {
            return false;
        }
        dv.copy[d] = cs;
    }
    prompt_loan::options lo;
    lo.routed_rows = o.routed_rows;
    lo.lend_pct = o.lend_pct;
    lo.min_keep = o.min_keep;
    lo.refill_overlap = o.refill_overlap;
    lo.refill_batch_layers = o.refill_batch_layers;
    lo.ring_bytes = o.ring_bytes;
    lo.group_bytes = o.group_bytes;
    lo.route_cover = o.route_cover;
    if (!p.res.init_devices(dv, err) || !p.loan.init(dv, p.res, p.win.loop(), lo, err) ||
        !p.win.init(dv, p.res, p.loan, err)) {
        return false;
    }
    p.win.set_native(p.res.native_ready()); // ME4: the native views serve whenever the residency holds them
    p.loan.collect_region(m, dv, o.lend_region);
    // MX7: the prompt CPU share's device buffers before the pool's measurement (one device and a share other than 0)
    prompt_share::options so;
    so.share = o.prompt_cpu_share;
    so.max_rows = o.prompt_cpu_share_max;
    so.maxt = o.prompt_cpu_share_test_maxt;
    if (!p.share.init(dv, p.res, so, err)) {
        return false;
    }
    // ME3: the graph executor's capture, with the windows' buffers in place and before the pool's measurement, so the
    // graphs' memory is in each device's use (and under device_cap_bytes) ahead of the pool
    if (o.before_pool && !o.before_pool(*this, err)) {
        return err.empty() ? fail(err, "the capture before the pool") : false;
    }
    // the pool: what each device has left less the reserve (the smaller); its tail the prompt zone and the lent
    // region, the rest divided among the classes by payload
    int64_t fr_min = -1;
    residency::pool_options po;
    po.pool_bytes = o.pool_bytes;
    po.reserve_bytes = o.reserve_bytes;
    po.device_cap_bytes = o.device_cap_bytes;
    po.max_swaps = o.adapt.max_swaps;
    po.swap_overlap = o.swap_overlap;
    if (!p.res.measure(dv, o.device_cap_bytes, fr_min, err) ||
        !p.res.init_pool(dv, fr_min, po, p.loan.lent_bytes(), err)) {
        return false;
    }
    p.loan.attach();
    // the link (into the empty pool), then the boot placement
    if (!p.res.probe_link(dv, err) || !p.res.fill_boot(dv, err)) {
        return false;
    }
    // the threads: the allowed mask read here, before the decode thread pins itself (its first window), so the
    // helpers' mask derives from the original one
    const std::vector<std::vector<int32_t>> cores = physical_cores();
    std::vector<int32_t> allowed;
    for (const auto & core : cores) {
        allowed.insert(allowed.end(), core.begin(), core.end());
    }
    if (!cpus_allowed(allowed, o, err)) {
        return false;
    }
    window_service::options wo;
    wo.cpu_workers = o.cpu_workers;
    wo.worker_cpus = o.worker_cpus;
    wo.helper_cpus = o.helper_cpus;
    wo.service_cpu = o.service_cpu;
    wo.decode_cpu = o.decode_cpu;
    wo.park_ms = o.park_ms;
    wo.share0 = o.share0;
    wo.refine = o.refine;
    wo.split = o.split;
    if (!p.win.start(wo, cores, p.loan, err)) {
        return false;
    }
    // MX7: the share's thread, the pool's caller during a prompt; the loan's route-first ring layers ask it
    if (!p.share.start(p.win, err)) {
        return false;
    }
    if (p.share.armed()) {
        prompt_share * const sh = &p.share;
        p.loan.set_cpu_pick([sh](int32_t l, int32_t rows, const int32_t * ids, const int32_t * counts,
                                 const uint8_t * resident, void * const * stream, uint8_t * cpu) {
            void * const st[2] = {stream[0], stream[1]};
            return sh->pick(l, rows, ids, counts, resident, st, cpu);
        });
    }
    // ME3: the helpers' mask (the prefill gather thread; expert_service::helper_cpus): given, else the allowed mask
    // less the workers', service and decode CPUs, else less the service and decode CPUs; never empty
    p.win.set_helpers(o.helper_cpus.empty()
                          ? helper_mask(allowed, p.win.worker_cpus(), p.win.service_cpu(), p.win.decode_cpu())
                          : o.helper_cpus);
    if (p.win.helper_cpus().empty()) {
        return fail(err, "no CPU for the helpers (the allowed mask is empty)");
    }
    return true;
}

bool expert_cache::failed(std::string & err) {
    return fail(err, "a window failed: " + p_->win.loop().error());
}

// --- the round boundary and the loan's order ---

bool expert_cache::round_start(void * const stream[2], std::string & err) {
    parts & p = *p_;
    decode_[0] = stream[0];
    decode_[1] = stream[1];
    if (!p.res.read_swap_waits(p.dv, err) || !p.res.try_commit(p.dv, false, err) ||
        !p.loan.refill_publish(stream, err)) {
        return false;
    }
    // ME9: the landed admissions published before the adapt choice (it sees them resident); a kept failure, an
    // abandoned pass or released waits invalidate the records instead
    const bool failed_now = p.win.loop().failed();
    if (!p.res.adm_publish(p.dv, failed_now, false, stream, err)) {
        return false;
    }
    const int64_t completed = round_;
    ++round_;
    ++n_.rounds;
    if (p.win.mode() == 0 && completed != 0) {
        if (!o_.profile_save.empty()) {
            p.res.mark_learned();
        }
        p.win.refine_share(completed);
        // ME3: no choice while a refill is outstanding or a prompt holds the zone (its staging need was sized to the
        // residency at its loan)
        if (completed % o_.adapt.every == 0 && !p.res.pending() && !p.loan.refill_active() && !p.loan.taken() &&
            !test_hold_adapt_) {
            const std::vector<residency::move> moves = p.res.choose(o_.adapt);
            int64_t bytes = 0;
            if (!moves.empty() && !p.res.publish(p.dv, moves, false, stream, bytes, err)) {
                return false;
            }
        }
    }
    return admission_round(stream, failed_now, err);
}

// ME9: the round's admission, decided once here (the service idle) and handed to the service's plan: on with one
// device, mode 0, no loan taken or held, no refill outstanding, no pending pass (this round's adapt pass included)
// and no kept failure; then each layer's reservations topped up (after the pass, whose destinations are its own)
bool expert_cache::admission_round(void * const stream[2], bool failed_now, std::string & err) {
    parts & p = *p_;
    const bool on = p.res.adm_on() && p.win.mode() == 0 && !p.loan.taken() && !p.loan.refill_active() &&
                    !p.res.pending() && !failed_now;
    if (!on) {
        p.win.set_admit(false);
        if (!p.res.adm_on() || p.win.mode() != 0) {
            ++n_.adm_off_other;
        } else if (p.loan.taken()) {
            ++n_.adm_off_loan;
        } else if (p.loan.refill_active()) {
            ++n_.adm_off_refill;
        } else if (p.res.pending()) {
            ++n_.adm_off_pass;
        } else {
            ++n_.adm_off_failed;
        }
        return true;
    }
    if (!p.res.adm_reserve(p.dv, stream, err)) {
        return false;
    }
    ++n_.adm_rounds_on;
    p.win.set_admit(true);
    return true;
}

bool expert_cache::zone_take(void * const stream[2], const std::vector<model::byte_span> * need, std::string & err) {
    parts & p = *p_;
    if (!p.res.try_commit(p.dv, true, err) || !p.loan.refill_stop(false, err)) {
        return false;
    }
    // ME9: before the loan can borrow a reserved slot, every admission copy settled (its event synchronized) and
    // published, so no copy lands in bytes the prompt stages into; nothing admitted while the loan is out
    p.win.loop().wait_idle();
    p.win.set_admit(false);
    // (the rows uploaded synchronously: the prompt's streams are not the decode stream the next pass reads them on)
    if (!p.res.adm_publish(p.dv, p.win.loop().failed(), true, nullptr, err)) {
        return false;
    }
    if (p.res.learned_dirty()) { // between requests: the learned ranking saved (a failure printed, not fatal)
        std::string e2;
        if (!save_profile(e2)) {
            std::fprintf(stderr, "%s\n", e2.c_str());
        }
    }
    return p.loan.take(stream, need, legacy_, err);
}

bool expert_cache::zone_return(void * const stream[2], std::string & err) {
    parts & p = *p_;
    if (!p.res.try_commit(p.dv, true, err) || !p.loan.give_back(stream, legacy_, err)) {
        return false;
    }
    p.res.adm_loan_returned(); // ME9: the reserved slots the loan borrowed come back to admission, their bytes garbage
    return true;
}

// --- decode ---

bool expert_cache::ffn_pair(model::card_ctx * const c[2], const model::layer_weights * const w[2], int32_t rows,
                            fn::experts::gate_up_args ga[2], fn::experts::down_args da[2], std::string & err) {
    const cache::device_guard keep;
    parts & p = *p_;
    const cache::devices & dv = p.dv;
    const int32_t layer = w[0]->layer;
    const bool placed = dv.nd == 1 ? c[0]->stream == c[1]->stream && c[0]->device == dv.dev[0] &&
                                         c[1]->device == dv.dev[0]
                                   : c[0]->device == dv.dev[0] && c[1]->device == dv.dev[1];
    if (layer < 0 || layer >= fn::n_layer || w[1]->layer != layer || rows < 1 || rows > fn::experts::max_rows ||
        !placed) {
        return fail(err, "a window outside the cache's layers, rows, devices or streams");
    }
    svc::loop & lp = p.win.loop();
    if (lp.failed()) {
        return failed(err);
    }
    bool desc = false;
    if (stream_capturing(c[0]->stream)) {
        if (dv.nd != 1) {
            return fail(err, "a two-device capture enqueues each card's part (ffn_card)");
        }
        n_.graph_windows_captured += 1;
        return p.win.enqueue(c, w, rows, ga, da, -1, 0, true, desc, err);
    }
    if (graph_) {
        return fail(err, "an eager window while the graph executor runs the cache (set_executor)");
    }
    if (dv.quiet && !p.win.decode_pinned()) { // ME2b item 3: the decode thread on the service's sibling CPU, once
        p.win.pin_decode();
    }
    void * const streams[2] = {c[0]->stream, dv.nd == 2 ? c[1]->stream : nullptr};
    // ME2b: a held region (the forced prompt path's rounds and catch-up) stays out
    if (p.loan.taken() && !p.loan.held()) {
        lp.wait_idle();
        if (!zone_return(streams, err)) {
            return false;
        }
    }
    if (layer == 0) {
        lp.wait_idle();
        if (!round_start(streams, err)) {
            return false;
        }
    }
    // the layer's staging and events are free once the service has taken the layer's previous window
    lp.wait_layer(layer);
    const int32_t par = p.win.next_par(layer);
    if (!p.win.enqueue(c, w, rows, ga, da, -1, par, false, desc, err)) {
        if (desc) { // its flag waits may be enqueued: the service unblocks them after its descriptor
            lp.post_failed(layer, rows, par, err);
        }
        return false;
    }
    n_.short_windows += short_read_ ? 1 : 0;
    n_.short_rows += short_read_ ? rows : 0;
    lp.post(layer, rows, par);
    return true;
}

bool expert_cache::ffn_card(model::card_ctx & c, const model::layer_weights & w, int32_t rows,
                            fn::experts::gate_up_args & ga, fn::experts::down_args & da, std::string & err) {
    const cache::device_guard keep;
    parts & p = *p_;
    if (p.dv.nd != 2 || c.card < 0 || c.card > 1 || c.device != p.dv.dev[c.card] || w.layer < 0 ||
        w.layer >= fn::n_layer || rows < 1 || rows > fn::experts::max_rows) {
        return fail(err, "a per-card cached window outside two devices, the cache's layers or rows");
    }
    if (!stream_capturing(c.stream)) {
        return fail(err, "the per-card cached window runs in a graph capture (the eager pair order: ffn_pair)");
    }
    if (p.win.loop().failed()) {
        return failed(err);
    }
    model::card_ctx * cc[2] = {nullptr, nullptr};
    const model::layer_weights * ww[2] = {nullptr, nullptr};
    fn::experts::gate_up_args g2[2];
    fn::experts::down_args d2[2];
    cc[c.card] = &c;
    ww[c.card] = &w;
    g2[c.card] = ga;
    d2[c.card] = da;
    if (c.card == 0) {
        n_.graph_windows_captured += 1;
    }
    bool desc = false;
    return p.win.enqueue(cc, ww, rows, g2, d2, c.card, 0, true, desc, err);
}

// ME3: the graph executor's replay boundary, before its launch: the service idle (every window of the previous
// replay served), the zone returned, the previous replay's window times read, the round boundary (the commit, the
// share, the adapt pass, published on the compute streams ahead of the launch); the layers whose copies that pass
// enqueued are the replay's measured swap waits
bool expert_cache::pass_begin(void * const stream[2], std::string & err) {
    const cache::device_guard keep;
    parts & p = *p_;
    svc::loop & lp = p.win.loop();
    if (lp.failed()) {
        return failed(err);
    }
    if (!graph_) {
        return fail(err, "pass_begin: the cache runs the eager executor");
    }
    if (p.dv.quiet && !p.win.decode_pinned()) {
        p.win.pin_decode();
    }
    lp.wait_idle();
    p.win.clear_stamps();
    void * const streams[2] = {stream[0], p.dv.nd == 2 ? stream[1] : nullptr};
    if (p.loan.taken() && !p.loan.held() && !zone_return(streams, err)) {
        return false;
    }
    p.win.read_replay_times();
    if (!round_start(streams, err)) {
        return false;
    }
    p.res.mark_replay_waits();
    return true;
}

// ME3: before the replay's launch (after pass_begin): one job a layer, in layer order, at parity 0 (the replay's
// records), each waiting for its window's stamp
bool expert_cache::pass_posted(int32_t rows, std::string & err) {
    if (!graph_ || rows < 1 || rows > fn::experts::max_rows) {
        return fail(err, "pass_posted: the eager executor, or rows outside 1..5");
    }
    p_->win.loop().post_pass(fn::n_layer, rows);
    n_.graph_replays += 1;
    n_.short_windows += short_read_ ? fn::n_layer : 0;
    n_.short_rows += short_read_ ? int64_t(rows) * fn::n_layer : 0;
    return true;
}

// ME3: the replay's launch failed after pass_posted: its jobs end failed (their flags written), and so does every
// later call
void expert_cache::pass_abandon(const std::string & why) {
    p_->res.adm_poison(); // ME9: no admission of this pass's copies is published
    p_->win.loop().abandon(why);
}

// ME3 (Fable's load G ruling): adapted from Strata-V100 69713b61, src/core/verify.cpp (Verifier::release_gpu_waits,
// #267): there the words are mapped host memory, stored once past every ring; here they are device memory (the flags,
// a replay's swap counts), written by a stream of the cache's own, and written again each millisecond because a later
// window's resolve resets its flags. Strata's notice:
//
// MIT License
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
bool expert_cache::release_gpu_waits(int32_t timeout_ms, bool only_if_busy, std::string & note) {
    const cache::device_guard keep;
    parts & p = *p_;
    const cache::devices & dv = p.dv;
    note.clear();
    // each device's streams the waits hold up: the decode stream, the copy stream (the lane, the flags), the swap
    // stream (the passes, the refill)
    const auto busy = [&](int32_t d) {
        void * const s[3] = {decode_[d], dv.copy[d], p.res.swap_stream(d)};
        bool any = false;
        cudaSetDevice(dv.dev[d]);
        for (void * x : s) {
            any = any || (x != nullptr && cudaStreamQuery(static_cast<cudaStream_t>(x)) == cudaErrorNotReady);
        }
        return any;
    };
    const auto any_busy = [&]() {
        bool any = false;
        for (int32_t d = 0; d < dv.nd; ++d) {
            any = any || (dv.dev[d] >= 0 && busy(d));
        }
        return any;
    };
    const auto forget = [&]() {
        if (only_if_busy) {
            decode_[0] = decode_[1] = nullptr;
        }
    };
    if (m_ == nullptr || (only_if_busy && !any_busy())) {
        forget();
        return true;
    }
    p.res.adm_poison(); // ME9: no admission is published after the waits are released
    p.win.loop().release("the cache's GPU waits were released after a completion timeout or at teardown; restart "
                         "the engine (ME3, Strata #267)");
    void * rs[2] = {nullptr, nullptr};
    for (int32_t d = 0; d < dv.nd; ++d) {
        cudaSetDevice(dv.dev[d]);
        cudaStream_t s = nullptr;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess) {
            rs[d] = s;
        }
    }
    const auto t0 = cache::clk::now();
    const auto ms = [&]() { return std::chrono::duration<double, std::milli>(cache::clk::now() - t0).count(); };
    bool drained = false;
    std::string e;
    for (;;) {
        for (int32_t d = 0; d < dv.nd; ++d) {
            if (rs[d] == nullptr) {
                continue;
            }
            cudaSetDevice(dv.dev[d]);
            uint32_t * const fl = p.win.flags(d);
            uint32_t * const sw = p.res.swap_done(d);
            for (int32_t i = 0; i < 2 * fn::n_layer && fl != nullptr; ++i) {
                cache::stream_write(rs[d], fl + i, window_service::flag_set, "release flag", e);
            }
            for (int32_t l = 0; l < fn::n_layer && sw != nullptr; ++l) {
                cache::stream_write(rs[d], sw + l, 0x40000000u, "release swap count", e); // past every count
            }
            cudaStreamSynchronize(static_cast<cudaStream_t>(rs[d]));
        }
        if (!any_busy()) {
            drained = true;
            break;
        }
        if (ms() > timeout_ms) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const double took = ms();
    for (int32_t d = 0; d < dv.nd; ++d) {
        if (rs[d] != nullptr) {
            cudaSetDevice(dv.dev[d]);
            cudaStreamDestroy(static_cast<cudaStream_t>(rs[d]));
        }
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf), drained ? "its GPU waits were released and the GPU finished in %.0f ms (ME3, %s)"
                                            : "its GPU waits were released but the GPU did not finish within %.0f ms "
                                              "(ME3, %s)",
                  drained ? took : double(timeout_ms), "Strata #267");
    note = buf;
    if (!e.empty()) {
        note += "; " + e;
    }
    forget();
    return drained;
}

void expert_cache::test_withhold(int32_t n) {
    p_->win.test_withhold(n);
}

int32_t expert_cache::test_withheld() const {
    return p_->win.test_withheld();
}

void expert_cache::set_executor(bool graph) {
    p_->win.loop().wait_idle();
    graph_ = graph;
}

void expert_cache::set_legacy(bool on) {
    p_->win.loop().wait_idle();
    legacy_ = on;
}

bool expert_cache::set_mode(int32_t mode, std::string & err) {
    p_->win.loop().wait_idle();
    return p_->win.set_mode(mode, err);
}

// --- prefill ---

// ME2b item 6, ME3: the prompt's loan taken (on) or ended (off). On: a zone still out from an earlier prompt returned
// first (synchronously), then the loan (zone_take: need's spans and the staging need, or the whole zone). With a lent
// region each device is then drained, so no reader of a former resident, no copy into a borrowed slot and no table
// upload is outstanding when the prompt's first write enters; poison (TEST) fills the borrowed slots and spans. Off:
// with a lent region each device drained (the prompt's last readers, its staging and message copies), the zone left
// taken: its keys refill at the next decode window or drain. Without a lent region nothing is drained or held (the
// staging copies wait on the zone's events). On a failure the hold stays as it was.
bool expert_cache::prompt_hold(bool on, void * const stream[2], const std::vector<model::byte_span> * need,
                               std::string & err) {
    parts & p = *p_;
    if (on == p.loan.held() && (p.loan.lent_bytes() > 0 || !on)) {
        return true;
    }
    const cache::device_guard keep;
    p.win.loop().wait_idle();
    if (p.win.loop().failed()) {
        return failed(err);
    }
    if (on && p.loan.taken() && !zone_return(nullptr, err)) {
        return false;
    }
    if (on && !zone_take(stream, need, err)) {
        return false;
    }
    return p.loan.hold(on, need, legacy_, err);
}

bool expert_cache::prompt_loan_check(const std::vector<model::byte_span> & need, std::string & err) {
    return p_->loan.check(need, err);
}

bool expert_cache::prompt_layer(int32_t layer, int32_t rows, const int32_t * ids, bool more, void * const stream[2],
                                std::string & err) {
    const cache::device_guard keep;
    parts & p = *p_;
    if (layer < 0 || layer >= fn::n_layer || stream[0] == nullptr || (p.dv.nd == 2 && stream[1] == nullptr) ||
        (ids != nullptr && (rows < 1 || rows > cache::max_prompt_rows))) {
        return fail(err, "prompt layer " + std::to_string(layer));
    }
    p.win.loop().wait_idle();
    if (p.win.loop().failed()) {
        return failed(err);
    }
    // a prompt without a hold (prefill-test's callers): its loan here, the staging only
    const std::vector<model::byte_span> none;
    if (!p.loan.taken() && !zone_take(stream, &none, err)) {
        return false;
    }
    return p.loan.prompt_layer(layer, rows, ids, more, stream, err);
}

const uint8_t * const * expert_cache::prompt_table(int32_t layer, int32_t card, bool down) const {
    return p_->loan.prompt_table(layer, card, down);
}

void expert_cache::set_prompt_ring(int32_t rows) {
    p_->loan.set_ring(legacy_ ? 0 : rows);
}

void expert_cache::set_me4_base(bool on) {
    p_->loan.set_me3_place(on);
    p_->win.set_native(!on && p_->res.native_ready()); // base: main's CPU path (the reconstruction per miss)
}

bool expert_cache::cpu_native() const {
    return p_->win.native();
}

int32_t expert_cache::prompt_groups(int32_t layer) const {
    return p_->loan.prompt_groups(layer);
}

bool expert_cache::prompt_group(int32_t layer, int32_t g, void * const stream[2], model::prompt_group_view & v,
                                std::string & err) {
    const cache::device_guard keep;
    return p_->loan.prompt_group(layer, g, stream, v, err);
}

bool expert_cache::prompt_group_done(int32_t layer, int32_t g, void * const stream[2], std::string & err) {
    const cache::device_guard keep;
    return p_->loan.prompt_group_done(layer, g, stream, err);
}

bool expert_cache::set_prompt_cpu_share(double share, std::string & err) {
    return p_->share.set_share(share, err);
}

void expert_cache::set_prompt_cpu_share_shadow(bool on) {
    p_->share.set_shadow(on);
}

std::vector<mx7_shadow_pair> expert_cache::take_prompt_cpu_share_shadow() {
    return p_->share.take_shadow();
}

bool expert_cache::prompt_share_begin(int32_t layer, const uint16_t * x16, void * const stream[2], std::string & err) {
    const cache::device_guard keep;
    return p_->share.begin(layer, x16, stream, err);
}

const uint8_t * expert_cache::prompt_share_skip(int32_t layer) const {
    return p_->share.skip(layer);
}

bool expert_cache::prompt_share_end(int32_t layer, const model::prompt_share_card card[2], void * const stream[2],
                                    std::string & err) {
    const cache::device_guard keep;
    return p_->share.end(layer, card, stream, err);
}

void expert_cache::prompt_share_settle(int32_t layer) {
    p_->share.settle(layer);
}

const uint8_t * expert_cache::prompt_bank(int32_t card) const {
    return p_->res.zero(p_->dv.dev_of(card));
}

// --- between requests ---

bool expert_cache::drain(std::string & err) {
    const cache::device_guard keep;
    parts & p = *p_;
    p.win.loop().wait_idle();
    for (int32_t d = 0; d < p.dv.nd; ++d) {
        if (!p.dv.set(d, err) || !cuda_ok(cudaDeviceSynchronize(), "drain", err)) {
            return false;
        }
    }
    p.win.read_all_times();
    // held (ME2b): the prompt still owns the lent region, so the zone stays out until it returns it; ME3: an
    // outstanding refill completed first (a test's batch limit lifted: the drain waits for every batch)
    p.loan.test_refill_limit(-1);
    if (!p.res.read_swap_waits(p.dv, err) || !p.res.try_commit(p.dv, true, err) || !p.loan.refill_stop(true, err) ||
        (p.loan.taken() && !p.loan.held() && !zone_return(nullptr, err))) {
        return false;
    }
    // ME9: the synchronized settle point: completed admissions published (rows uploaded) without a kept failure,
    // discarded with one
    p.win.set_admit(false);
    if (!p.res.adm_publish(p.dv, p.win.loop().failed(), true, nullptr, err)) {
        return false;
    }
    if (p.res.learned_dirty() && !save_profile(err)) {
        return false;
    }
    if (p.win.loop().failed()) {
        return failed(err);
    }
    return true;
}

bool expert_cache::save_profile(std::string & err) {
    return p_->res.save_profile(o_.profile_save, p_->loan.lent_keys(), err);
}

bool expert_cache::reset_to_profile(std::string & err) {
    const cache::device_guard keep;
    parts & p = *p_;
    if (p.loan.held()) { // ME2b: the boot placement fills the lent region, which a prompt holds
        return fail(err, "reset to the profile while a prompt holds the lent region");
    }
    if (!drain(err)) {
        return false;
    }
    p.res.adm_reset(); // ME9: every reservation and record discarded: the boot placement exactly, no holes
    if (!p.res.fill_boot(p.dv, err)) {
        return false;
    }
    p.res.reset_usage();
    round_ = 0;
    p.win.reset_share();
    return true;
}

bool expert_cache::test_verify_zone(int64_t & checked, int64_t & bad, std::string & err) {
    const cache::device_guard keep;
    // no drain: a refill not yet published leaves its slots keyless, so before one this checks only the slots
    // the loan never took; the caller drains first to check the refilled ones too
    p_->win.loop().wait_idle();
    return p_->res.verify_zone(p_->dv, checked, bad, err);
}

void expert_cache::request_begin() {
    req_n_ = counters();
    req_share_ = p_->win.share_log().size();
    req_rf_keys_ = p_->loan.restore_keys(req_rf_bytes_);
    ++requests_;
}

// --- accessors ---

bool expert_cache::prompt_lent() const {
    return p_->loan.lent_bytes() > 0;
}
bool expert_cache::prompt_held() const {
    return p_->loan.held();
}
int64_t expert_cache::zone_slots() const {
    return p_->res.zone_slots();
}
const std::vector<int32_t> * expert_cache::helper_cpus() const {
    return &p_->win.helper_cpus();
}
int32_t expert_cache::service_cpu() const {
    return p_->win.service_cpu();
}
int32_t expert_cache::decode_cpu() const {
    return p_->win.decode_cpu();
}
void expert_cache::test_poison_lent(bool on) {
    p_->loan.set_poison(on);
}
void expert_cache::test_refill_limit(int32_t limit) {
    p_->loan.test_refill_limit(limit);
}
int32_t expert_cache::test_adm_held() const {
    return p_->res.adm_held();
}
bool expert_cache::test_adm_poison(std::string & err) {
    const cache::device_guard keep;
    p_->win.loop().wait_idle();
    return p_->res.adm_test_poison(p_->dv, err);
}
bool expert_cache::test_adm_verify(int64_t & checked, int64_t & bad, std::string & err) {
    const cache::device_guard keep;
    p_->win.loop().wait_idle();
    return p_->res.adm_test_verify(p_->dv, checked, bad, err);
}
bool expert_cache::test_adm_force(int32_t & n, std::string & err) {
    const cache::device_guard keep;
    p_->win.loop().wait_idle();
    test_hold_adapt_ = true;
    p_->win.test_gpu_all(true);
    std::vector<int32_t> routed(size_t(fn::n_layer) * 10);
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        p_->win.test_routed(l, routed.data() + size_t(l) * 10);
    }
    const bool ok = p_->res.adm_test_evict(p_->dv, routed, test_forced_, err);
    n = int32_t(test_forced_.size());
    return ok;
}
void expert_cache::test_adm_force_end() {
    p_->win.loop().wait_idle();
    test_hold_adapt_ = false;
    p_->win.test_gpu_all(false);
}
void expert_cache::test_adm_forced(int32_t & n_admitted, int32_t & n_replaced, std::string & failures) const {
    p_->res.adm_test_forced(n_admitted, n_replaced, failures);
}
int64_t expert_cache::lent_offset() const {
    return p_->res.lent_off();
}
int64_t expert_cache::lent_bytes() const {
    return p_->loan.lent_bytes();
}
const uint8_t * expert_cache::test_pool(int32_t d) const {
    return p_->res.pool(d);
}
int32_t expert_cache::devices() const {
    return p_->dv.nd;
}
int32_t expert_cache::device_id(int32_t d) const {
    return p_->dv.dev[d];
}
int64_t expert_cache::pool_bytes() const {
    return p_->res.pool_bytes();
}
int64_t expert_cache::device_bytes(int32_t d) const {
    return p_->res.device_bytes(d) + p_->loan.device_bytes(d) + p_->win.device_bytes(d) + p_->share.device_bytes(d);
}
int64_t expert_cache::pinned_bytes() const {
    return p_->res.host_bytes() + p_->loan.host_bytes() + p_->win.host_bytes() + p_->share.host_bytes();
}
int64_t expert_cache::backing_bytes() const {
    return p_->res.backing_bytes();
}
int32_t expert_cache::cpu_workers() const {
    return p_->win.workers();
}
int32_t expert_cache::resident() const {
    return p_->res.resident();
}
double expert_cache::share() const {
    return p_->win.share();
}
const std::vector<share_attempt> & expert_cache::share_attempts() const {
    return p_->win.share_log();
}

const expert_cache_counters & expert_cache::counters() const {
    const parts & p = *p_;
    sum_ = counters_add(counters_add(counters_add(p.res.counters(), p.loan.counters(), 1), p.win.counters(), 1), n_, 1);
    sum_ = counters_add(sum_, p.share.counters(), 1); // MX7
    return sum_;
}

void expert_cache::reset_counters() {
    p_->res.reset_counters();
    p_->loan.reset_counters();
    p_->win.reset_counters();
    p_->share.reset_counters();
    n_ = expert_cache_counters();
}

} // namespace fn::runtime
