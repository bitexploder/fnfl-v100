// ME3 (the cleanup): the expert cache's decode windows, the host service's half (cache_service.h); ME1's, ME2's and
// ME2b's code moved here from expert_cache.cpp. The window's GPU work is cache_window.cpp's.

#include "fn/runtime/cache_service.h"

#include "fn/kernels/experts/experts_cache.h"
#include "fn/runtime/miss_split.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>

namespace fn::runtime {

using cache::cuda_ok;
using cache::D;
using cache::fail;
using cache::n_ex;

bool window_service::init(const cache::devices & dv, residency & r, const prompt_loan & ln, std::string & err) {
    dv_ = &dv;
    res_ = &r;
    loan_ = &ln;
    const auto vp = [](auto ** p) { return reinterpret_cast<void **>(p); };
    const size_t ext_bytes = size_t(max_u) * D * 4;
    if (!own_.halloc(size_t(fn::n_layer) * max_b * D, vp(&x_codes_h_), false, err) ||
        !own_.halloc(size_t(fn::n_layer) * max_b * (D / 32) * 2 * 4, vp(&x_scales_h_), false, err) ||
        !own_.halloc(size_t(fn::n_layer) * max_b * 10 * 4, vp(&ids_h_), false, err) ||
        !own_.halloc(ext_bytes, vp(&ds_h_), false, err) ||
        !own_.halloc(size_t(max_b) * fn::experts::cpu::n_q8k * sizeof(fn::experts::cpu::block_q8_k), vp(&act_), false,
                     err) ||
        // ME3: the replays' window stamps and flag waits' times, mapped (each device's address of them below)
        !own_.halloc(size_t(2) * fn::n_layer * 4, vp(&stamp_h_), true, err) ||
        !own_.halloc(size_t(2) * wt_n * 8, vp(&wt_h_), true, err)) {
        return false;
    }
    lane_bytes_ = max_u * r.max_entry();
    for (int32_t d = 0; d < dv.nd; ++d) {
        if (!dv.set(d, err) ||
            !cuda_ok(cudaHostGetDevicePointer(vp(&stamp_d_[d]), stamp_h_ + d * fn::n_layer, 0), "stamp pointer",
                     err) ||
            !cuda_ok(cudaHostGetDevicePointer(vp(&wt_d_[d]), wt_h_ + d * wt_n, 0), "wait times pointer", err) ||
            !own_.dalloc(dv, d, size_t(lane_bytes_), vp(&lane_[d]), true, err) ||
            !own_.dalloc(dv, d, size_t(fn::n_layer) * 2 * 4, vp(&flags_[d]), true, err) ||
            !own_.dalloc(dv, d, size_t(fn::experts::max_apply) * 4, vp(&apply_idx_[d]), true, err) ||
            !own_.dalloc(dv, d, size_t(fn::experts::max_apply) * 8, vp(&apply_val_[d]), true, err) ||
            !own_.halloc(size_t(fn::experts::max_apply) * 4, vp(&apply_idx_h_[d]), false, err) ||
            !own_.halloc(size_t(fn::experts::max_apply) * 8, vp(&apply_val_h_[d]), false, err)) {
            return false;
        }
        for (int32_t card = 0; card < 2; ++card) {
            if (dv.dev_of(card) == d && !own_.dalloc(dv, d, size_t(fn::experts::meta_bytes), vp(&meta2_[card]), true,
                                                     err)) {
                return false;
            }
        }
        // ME3: the CPU's products on card 0's device only (the down kernel leaves the CPU's slots out; the combine adds
        // them)
        if (d == dv.dev_of(0) && !own_.dalloc(dv, d, ext_bytes, vp(&ext_ds_), true, err)) {
            return false;
        }
        ev_desc_[d].assign(fn::n_layer, nullptr);
        ev_fw_[d].assign(size_t(fn::n_layer) * 2 * 4, nullptr);
        ev_lane_[d].assign(size_t(fn::n_layer) * 2 * 2, nullptr);
        for (auto & e : ev_desc_[d]) {
            if (!own_.event(dv, d, &e, false, false, err)) {
                return false;
            }
        }
        for (auto * v : {&ev_fw_[d], &ev_lane_[d]}) {
            for (auto & e : *v) {
                if (!own_.event(dv, d, &e, true, false, err)) {
                    return false;
                }
            }
        }
        // the stream memory operations on this device: a write the copy stream then waits for
        const cudaStream_t cs = static_cast<cudaStream_t>(dv.copy[d]);
        if (!dv.set(d, err) || !cache::stream_write(cs, flags_[d], 1, "flag write", err) ||
            !cache::stream_wait(cs, flags_[d], 1, "flag wait", err) ||
            !cuda_ok(cudaMemsetAsync(flags_[d], 0, size_t(fn::n_layer) * 2 * 4, cs), "flags", err) ||
            !cuda_ok(cudaStreamSynchronize(cs), "flag check", err)) {
            return false;
        }
    }
    wpar_.assign(fn::n_layer, 0);
    win_.assign(size_t(fn::n_layer) * 2, win_rec());
    return true;
}

// the CPU workers: one a physical core less one, each pinned to the first CPU of a core other than the first (ME3:
// or the given CPUs, the caller checked them against the allowed mask, read before the decode thread pins itself)
bool window_service::start(const options & o, const std::vector<std::vector<int32_t>> & cores, prompt_loan & rf,
                           std::string & err) {
    o_ = o;
    const int32_t nc = int32_t(cores.size());
    if (!o.worker_cpus.empty()) {
        worker_cpus_ = o.worker_cpus;
    } else {
        const int32_t nw = o.cpu_workers >= 0 ? o.cpu_workers : std::max(1, nc - 1);
        for (int32_t i = 0; i < nw; ++i) {
            const int32_t core = nc > 1 ? 1 + i % (nc - 1) : -1;
            worker_cpus_.push_back(core >= 0 ? cores[size_t(core)][0] : -1);
        }
    }
    const int32_t nw = int32_t(worker_cpus_.size());
    if (nw > 0 && !cpu_.start(worker_cpus_, err, o.park_ms)) {
        return fail(err, err);
    }
    // the miss share (PcieAuto): the link as the devices together copy an entry, the blob a mean entry
    link_gbps_ = res_->link_gbps();
    reset_share();
    loop_.start(this, &rf, fn::n_layer);
    // ME2b item 3 (one card): the service (which drains with the workers) on core 0's first CPU, the core the workers
    // leave, and the decode thread on its sibling (ME3: or the given CPUs)
    const bool quiet = dv_->quiet;
    service_cpu_ = o.service_cpu >= 0 ? o.service_cpu : quiet && nc > 0 ? cores[0][0] : -1;
    service_mask_ = cache::pin_thread(loop_.handle(), service_cpu_);
    decode_cpu_ = o.decode_cpu >= 0 ? o.decode_cpu : quiet && nc > 0 && cores[0].size() > 1 ? cores[0][1] : -1;
    return true;
}

void window_service::stop() {
    loop_.stop();
    cpu_.stop();
}

void window_service::pin_decode() {
    decode_pinned_ = true;
    decode_mask_ = cache::pin_thread(pthread_self(), decode_cpu_);
    std::fprintf(stderr, "expert cache: decode thread pinned to CPU %d (mask 0x%llx, read back)\n", decode_cpu_,
                 (unsigned long long)decode_mask_);
}

void window_service::clear_stamps() {
    for (int32_t i = 0; i < 2 * fn::n_layer; ++i) {
        reinterpret_cast<volatile uint32_t *>(stamp_h_)[i] = 0;
    }
    std::atomic_thread_fence(std::memory_order_release);
}

void window_service::read_replay_times() {
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (win_[size_t(l * 2)].live) {
            read_window_times(l, 0);
        }
    }
}

void window_service::read_all_times() {
    for (int32_t i = 0; i < fn::n_layer * 2; ++i) {
        read_window_times(i / 2, i % 2);
    }
}

void window_service::refine_share(int64_t completed) {
    policy::split_reading rd;
    const double f = auto_.window(cpu_ms_all_, cpu_jobs_all_, lane_ms_all_, lane_bytes_all_, share_, rd);
    if (f >= 0) {
        share_ = f;
        share_num_ = policy::share_num(f);
    }
    if (rd.reason != nullptr) {
        // ME3: every attempted refinement, unchanged ones too (L, R, the samples, the shares and the reason)
        share_log_.push_back({completed, rd});
        std::fprintf(stderr,
                     "expert cache: split refinement %d at round %lld: L %.3f GB/s (the lane's %lld bytes in %.3f ms; "
                     "probe %.3f), R %.3f GB/s (tc %.2f us, %lld CPU experts), reading %.4f, share %.3f -> %.3f: "
                     "%s\n",
                     rd.update, (long long)completed, rd.link_gbps, (long long)rd.lane_bytes, rd.lane_ms,
                     rd.probe_gbps, rd.rate_gbps, rd.tc_us, (long long)rd.samples, rd.raw_share, rd.old_share,
                     rd.new_share, rd.reason);
    }
}

void window_service::reset_share() {
    share_ = o_.share0;
    share_num_ = policy::share_num(share_);
    share_log_.clear();
    auto_ = policy::pcie_auto();
    auto_.on = cpu_.workers() > 0 && o_.refine;
    auto_.o = o_.split;
    auto_.link_gbps = link_gbps_ * dv_->nd;
    auto_.blob_bytes = double(res_->backing_bytes()) / double(fn::n_layer * n_ex);
    auto_.mark(cpu_ms_all_, cpu_jobs_all_, lane_ms_all_, lane_bytes_all_);
}

bool window_service::set_mode(int32_t mode, std::string & err) {
    if (mode < 0 || mode > 2 || (mode == 1 && cpu_.workers() == 0)) {
        return fail(err, "mode " + std::to_string(mode) + " (0, 1 with CPU workers, or 2)");
    }
    mode_ = mode;
    return true;
}

// the window's distinct experts in first-appearance order over p = 10 b + k (the routing order), their row masks and
// slot bases (the kernels' routing meta), the usage (+1 a row and ID, mode 0), and how each is served: resident, or
// of the misses (n_miss x num) >> 8 by the GPU lane and the rest by the CPU. ME5: the lane's are the misses serving the
// most rows, ties to the later in route order (miss_split.h; the last n_gpu before ME5), in route order on the lane
void window_service::plan(int32_t layer, int32_t rows) {
    const int32_t * ids = ids_h_ + layer * max_b * 10;
    n_u_ = fn::experts::cpu::window_meta(ids, rows, e_, mask_, base_, n_slots_);
    bad_ids_ = false;
    for (int32_t p = 0; p < rows * 10; ++p) {
        const bool ok = ids[p] >= 0 && ids[p] < n_ex;
        bad_ids_ = bad_ids_ || !ok;
        if (ok && mode_ == 0) {
            res_->count(layer, ids[p]);
        }
    }
    const int64_t entry = res_->layer(layer).entry_bytes();
    int32_t miss[max_u];
    int32_t nm = 0;
    for (int32_t u = 0; u < n_u_; ++u) {
        const int32_t s = res_->slot_of(layer * n_ex + e_[u]);
        ++n_.lookups;
        if (s >= 0) {
            how_[u] = 0;
            where_[u] = res_->slot_off(s);
            ++n_.hits;
            n_.hit_bytes += entry;
            n_.adm_hits += res_->adm_flag(layer * n_ex + e_[u]) == 1 ? 1 : 0; // ME9: a hit on an admitted key
        } else {
            miss[nm++] = u;
            n_.adm_lost += res_->adm_flag(layer * n_ex + e_[u]) == 2 ? 1 : 0; // ME9: a miss on a retired key
        }
    }
    const int32_t n_gpu = mode_ == 2 || cpu_.workers() == 0 || gpu_all_.load() ? nm
                          : mode_ == 1                                       ? 0
                                                                              : policy::gpu_misses(nm, share_num_);
    const int64_t dentry = entry / dv_->nd;
    uint8_t to_gpu[max_u];
    split::choose_gpu(miss, nm, mask_, n_gpu, to_gpu);
    n_lane_ = 0;
    n_cpu_ = 0;
    n_adm_ = 0;
    // ME9: in a round that admits, the layer's empty reserved slots take the GPU-served misses with the most rows
    // (route order breaking ties), a key once (none already copying or landed); the copy is the one the lane would
    // make, into the slot instead of the scratch lane. The others take the lane as before
    uint8_t admit_u[max_u] = {};
    int32_t rec_u[max_u] = {};
    if (admit_.load() && res_->adm_on()) {
        int32_t free[adm::per_layer];
        const int32_t n_free = res_->adm_free(layer, free);
        std::vector<adm::miss> cand;
        for (int32_t i = 0; i < nm && n_free > 0; ++i) {
            const int32_t u = miss[i];
            if (to_gpu[i] != 0 && !res_->adm_in_flight(layer * n_ex + e_[u])) {
                cand.push_back({u, split::rows_of(mask_[u]), i});
            }
        }
        const std::vector<adm::miss> chosen = adm::admit(cand, n_free);
        for (size_t k = 0; k < chosen.size(); ++k) {
            admit_u[chosen[k].u] = 1;
            rec_u[chosen[k].u] = free[k];
        }
        n_.adm_assigned += int64_t(chosen.size());
    }
    for (int32_t i = 0; i < nm; ++i) {
        const int32_t u = miss[i];
        if (to_gpu[i] != 0) {
            const uint32_t ev = residency::ev_missed | (admit_u[u] != 0 ? residency::ev_assigned : 0u);
            res_->adm_test_note(layer * n_ex + e_[u], ev);
            if (admit_u[u] != 0) {
                how_[u] = 3;
                adm_rec_[u] = rec_u[u];
                where_[u] = res_->slot_off(res_->adm_slot(rec_u[u]));
                ++n_adm_;
            } else {
                how_[u] = 1;
                where_[u] = int64_t(n_lane_++) * dentry;
            }
            ++n_.gpu_miss;
            ++n_.gpu_rows[split::rows_of(mask_[u])];
            n_.gpu_miss_bytes += entry;
        } else {
            how_[u] = 2;
            ++n_cpu_;
        }
    }
}

bool window_service::write_flags(const svc::job & j, int32_t which, std::string & err) {
    for (int32_t d = 0; d < dv_->nd; ++d) {
        if (!dv_->set(d, err)) {
            return false;
        }
        for (int32_t f = 0; f < 2; ++f) {
            if ((which == 0 && f == 1) || (which == 1 && f == 0)) {
                continue;
            }
            if (!cache::stream_write(dv_->copy[d], flags_[d] + 2 * j.layer + f, flag_set, "flag write", err)) {
                return false;
            }
        }
    }
    return true;
}

// a served window's GPU waits and lane copy time (device 0), once its events are complete
void window_service::read_window_times(int32_t layer, int32_t par) {
    win_rec & w = win_[size_t(layer * 2 + par)];
    if (!w.live) {
        return;
    }
    w.live = false;
    if (!w.lane && !w.cpu) {
        return;
    }
    const auto el = [&](void * a, void * b) {
        float ms = 0;
        cudaSetDevice(dv_->dev[0]);
        return cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(a), static_cast<cudaEvent_t>(b)) == cudaSuccess
                   ? double(ms)
                   : 0.0;
    };
    // ME3: a replay's flag waits from its wait kernels' times (device 0), else the events
    const auto flag_s = [&](int32_t fl) {
        if (!w.graph) {
            const size_t f = size_t(layer * 2 + par) * 4 + size_t(2 * fl);
            return el(ev_fw_[0][f], ev_fw_[0][f + 1]) * 1e-3;
        }
        const volatile uint64_t * const t = wt_h_ + wt_flag(layer, par, fl);
        return t[1] > t[0] ? double(t[1] - t[0]) * 1e-9 : 0.0;
    };
    const size_t c = size_t(layer * 2 + par) * 2;
    if (w.lane) {
        const double lw = flag_s(0);
        n_.lane_wait.n += 1;
        n_.lane_wait.s += lw;
        if (w.refill) {
            n_.lane_wait_refill.n += 1;
            n_.lane_wait_refill.s += lw;
        }
        const double lane_ms = el(ev_lane_[0][c], ev_lane_[0][c + 1]);
        n_.lane_copy_ms += lane_ms;
        // ME5 rider: the split's lane leg, its bytes credited with their own interval's time
        lane_ms_all_ += lane_ms;
        lane_bytes_all_ += w.lane_bytes;
    }
    if (w.cpu) {
        n_.cpu_wait.n += 1;
        n_.cpu_wait.s += flag_s(1);
    }
}

// the window's descriptor on every device: an eager window's event (ME2b item 3, one card: polled with _mm_pause on
// the service's own core, its next work following at once; the device's blocking flag would otherwise put this wait to
// sleep), a replay's stamp (posted before the launch, which can block until the replay's windows are served); quit
// (abandon, stop) ends a stamp's wait: false with err empty
bool window_service::await(const svc::job & j, const std::atomic<bool> & quit, std::string & err) {
    await_t0_ = cache::clk::now();
    for (int32_t d = 0; d < dv_->nd; ++d) {
        if (!dv_->set(d, err)) {
            return false;
        }
        if (j.stamp) {
            const volatile uint32_t * const st = stamp_h_ + d * fn::n_layer + j.layer;
            while (*st == 0) {
                if (quit) {
                    err.clear();
                    return false;
                }
                _mm_pause();
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            continue;
        }
        const auto e = static_cast<cudaEvent_t>(ev_desc_[d][size_t(j.layer)]);
        cudaError_t q = dv_->quiet ? cudaErrorNotReady : cudaEventSynchronize(e);
        while (q == cudaErrorNotReady && (q = cudaEventQuery(e)) == cudaErrorNotReady) {
            _mm_pause();
        }
        if (!cuda_ok(q, "window wait", err)) {
            return false;
        }
    }
    return true;
}

void window_service::unblock(const svc::job & j) {
    std::string ignored;
    write_flags(j, 2, ignored); // no stream waits for ever; the next call reports the failure
}

// one window (after its descriptor): the plan; when every expert is resident (and every ID valid) the devices have
// written the flags; otherwise the lane copies, the tables and the run word, the gate/up flags, the CPU experts (the
// pool), their ds and slot word, the down flags
bool window_service::serve(const svc::job & j, std::string & err) {
    const cache::devices & dv = *dv_;
    const residency & r = *res_;
    const int32_t layer = j.layer;
    const loader::expert_layer & L = r.layer(layer);
    const int64_t entry = L.entry_bytes(), dentry = entry / dv.nd;
    n_.desc_wait.n += 1;
    n_.desc_wait.s += cache::since(await_t0_);
    read_window_times(layer, 1 - j.par);
    ++n_.windows;
    plan(layer, j.rows);
    win_rec & w = win_[size_t(layer * 2 + j.par)];
    w.live = true;
    w.lane = n_lane_ + n_adm_ > 0;
    w.lane_bytes = int64_t(n_lane_ + n_adm_) * entry; // every device's copies (ME9's admissions included)
    w.cpu = n_cpu_ > 0;
    w.refill = loan_->refill_active();
    w.graph = j.stamp;
    if (n_lane_ + n_adm_ == 0 && n_cpu_ == 0 && !bad_ids_) {
        ++n_.device_windows;
        return true;
    }
    // TEST (the stall check): this window withheld, so the GPU waits on flags nobody writes
    if (withhold_.load() > 0 && withhold_.fetch_sub(1) == 1) {
        withheld_.store(layer);
        return true;
    }
    // ME2b: the CPU experts' jobs and the slot word at the plan, so the word goes with the table patches (before the
    // gate/up flag) and the GPU's down runs while the pool computes
    int32_t nj = 0, lo = max_u, hi = 0;
    uint64_t slots = 0;
    for (int32_t u = 0; u < n_u_; ++u) {
        if (how_[u] != 2) {
            continue;
        }
        fn::experts::cpu::job & jb = jobs_[nj++];
        const uint8_t * src = r.backing(layer) + e_[u] * entry;
        for (int32_t card = 0; card < 2; ++card) {
            jb.gu[card] = src + L.gu_at(card);
            jb.dn[card] = src + L.dn_at(card);
        }
        if (native_) { // ME4: the expert's native rows in place (no per-miss reconstruction)
            fn::experts::cpu::set_native(r.native(layer), e_[u], jb);
        } else {
            jb.ngate = jb.nup = jb.ndown = nullptr;
        }
        jb.nt = 0;
        for (int32_t b = 0; b < max_b; ++b) {
            if (((mask_[u] >> b) & 1) == 0) {
                continue;
            }
            const int32_t slot = base_[u] + jb.nt;
            jb.row[jb.nt] = b;
            jb.out[jb.nt] = ds_h_ + int64_t(slot) * D;
            slots |= uint64_t(1) << slot;
            lo = std::min(lo, slot);
            hi = std::max(hi, slot + 1);
            ++jb.nt;
        }
        ++n_.cpu_rows[jb.nt];
    }
    // the lane misses' copies, then each device's table patches, run word and slot word
    for (int32_t d = 0; d < dv.nd; ++d) {
        const cudaStream_t cs = static_cast<cudaStream_t>(dv.copy[d]);
        if (!dv.set(d, err)) {
            return false;
        }
        const size_t c = size_t(layer * 2 + j.par) * 2;
        const int32_t n_copy = n_lane_ + n_adm_; // ME9: the lane's copies and the admitted ones (into reserved slots)
        if (n_copy > 0 && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_lane_[d][c]), cs), "lane", err)) {
            return false;
        }
        for (int32_t u = 0; u < n_u_ && n_copy > 0; ++u) {
            if (how_[u] == 1 && !cuda_ok(cudaMemcpyAsync(lane_[d] + where_[u], r.entry_src(layer, e_[u], d),
                                                         size_t(dentry), cudaMemcpyHostToDevice, cs),
                                         "lane copy", err)) {
                return false;
            }
            if (how_[u] == 3) {
                // ME9: the complete entry into the reserved slot, then the record's own completion event; the record
                // is started only once both are enqueued (a failure leaves it empty: nothing to publish)
                if (!cuda_ok(cudaMemcpyAsync(r.pool(d) + where_[u], r.entry_src(layer, e_[u], d), size_t(dentry),
                                             cudaMemcpyHostToDevice, cs),
                             "admission copy", err) ||
                    !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(r.adm_event(adm_rec_[u])), cs), "admission",
                             err)) {
                    return false;
                }
                if (r.adm_start(adm_rec_[u], layer * n_ex + e_[u])) {
                    r.adm_test_note(layer * n_ex + e_[u], residency::ev_started);
                    ++n_.adm_started;
                    n_.adm_bytes += dentry;
                }
            }
        }
        if (n_copy > 0 && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_lane_[d][c + 1]), cs), "lane", err)) {
            return false;
        }
        // every miss copy of the window, whatever its destination (ME9: the scratch lane or a reserved slot), the
        // bytes of the ev_lane_ interval above; adm_bytes is the reserved slots' subset (Shell, fad3e540)
        n_.lane_bytes += int64_t(n_copy) * dentry;
        int32_t n = 0;
        int32_t * idx = apply_idx_h_[d];
        uint64_t * val = apply_val_h_[d];
        const auto put = [&](int64_t wd, const uint8_t * a) {
            idx[n] = int32_t(wd);
            val[n] = uint64_t(reinterpret_cast<uintptr_t>(a));
            ++n;
        };
        for (int32_t u = 0; u < n_u_; ++u) {
            if (how_[u] != 1 && how_[u] != 3) {
                continue;
            }
            const uint8_t * a = how_[u] == 3 ? r.pool(d) + where_[u] : lane_[d] + where_[u]; // ME9: the slot's bytes
            for (int32_t card = 0; card < 2; ++card) {
                if (dv.dev_of(card) != d) {
                    continue;
                }
                put(r.w_tab(dv.part_of(card, true), e_[u]), a + r.part_off(layer, card, true));
                put(r.w_tab2(dv.part_of(card, false), e_[u]), a + r.part_off(layer, card, false));
            }
        }
        idx[n] = int32_t(r.w_run());
        val[n] = n_lane_ + n_adm_ > 0 ? 1 : 0;
        ++n;
        idx[n] = int32_t(r.w_ext());
        val[n] = slots;
        ++n;
        bool ok = cuda_ok(cudaMemcpyAsync(apply_idx_[d], idx, size_t(n) * 4, cudaMemcpyHostToDevice, cs), "patches",
                          err) &&
                  cuda_ok(cudaMemcpyAsync(apply_val_[d], val, size_t(n) * 8, cudaMemcpyHostToDevice, cs), "patches",
                          err);
        if (ok && !fn::experts::apply(r.words(d), apply_idx_[d], apply_val_[d], n, cs)) {
            ok = fail(err, "apply");
        }
        if (!ok) {
            return false;
        }
    }
    if (!write_flags(j, 0, err)) {
        return false;
    }
    if (n_cpu_ > 0) {
        // the window's gate/up activations, the CPU experts' jobs, the pool
        const cache::clk::time_point tq = cache::clk::now();
        const int8_t * xc = x_codes_h_ + int64_t(layer) * max_b * D;
        const float * xs = x_scales_h_ + int64_t(layer) * max_b * (D / 32) * 2;
        for (int32_t b = 0; b < j.rows; ++b) {
            fn::experts::cpu::gate_up_act(xc + int64_t(b) * D, xs + int64_t(b) * (D / 32) * 2,
                                          act_ + int64_t(b) * fn::experts::cpu::n_q8k);
        }
        const double q8_ms = cache::since(tq) * 1e3;
        n_.cpu_q8_ms += q8_ms;
        fn::experts::cpu::run_stats st;
        if (!cpu_.run(L.gu_fmt, L.dn_fmt, act_, jobs_, nj, st)) {
            return fail(err, "the CPU pool stalled");
        }
        n_.cpu_gu_ms += st.gu_ms;
        n_.cpu_act_ms += st.act_ms;
        n_.cpu_dn_ms += st.dn_ms;
        n_.cpu_bytes += st.bytes;
        n_.cpu_gu_unpack_ms += st.gu_unpack_ms;
        n_.cpu_gu_compute_ms += st.gu_compute_ms;
        n_.cpu_dn_unpack_ms += st.dn_unpack_ms;
        n_.cpu_dn_compute_ms += st.dn_compute_ms;
        cpu_ms_all_ += q8_ms + st.gu_ms + st.act_ms + st.dn_ms; // ME5 rider: the input conversion and the pool phases
        cpu_jobs_all_ += nj;
        n_.cpu_miss += nj;
        n_.cpu_miss_bytes += nj * entry;
        ++n_.cpu_windows;
        // the whole products to card 0's ext_ds, read by the window's combine after the down flag
        const int32_t d0 = dv.dev_of(0);
        if (!dv.set(d0, err) ||
            !cuda_ok(cudaMemcpyAsync(ext_ds_ + int64_t(lo) * D, ds_h_ + int64_t(lo) * D, size_t(hi - lo) * D * 4,
                                     cudaMemcpyHostToDevice, static_cast<cudaStream_t>(dv.copy[d0])),
                     "CPU ds", err)) {
            return false;
        }
    }
    return write_flags(j, 1, err);
}

std::string window_service::plan_share(double link_gbps) const {
    char b[256];
    std::snprintf(b, sizeof(b), "  miss share %.2f at start; link probe %.2f GB/s (device 0), PcieAuto %s\n", o_.share0,
                  link_gbps, auto_.on ? "on" : "off");
    return b;
}

std::string window_service::cpu_list() const {
    std::string cpus;
    for (size_t i = 0; i < worker_cpus_.size(); ++i) {
        char m[48];
        std::snprintf(m, sizeof(m), "%s%d (mask 0x%llx)", i == 0 ? "" : ", ", worker_cpus_[i],
                      i < size_t(cpu_.workers()) ? (unsigned long long)cpu_.affinity(int32_t(i)) : 0ull);
        cpus += m;
    }
    return cpus;
}

std::string window_service::plan_threads() const {
    std::string s;
    char b[640]; // ME3: the waits line is longer than 400
    std::snprintf(b, sizeof(b),
                  "  threads (ME2b): service on CPU %d (mask 0x%llx, read back), drains with the workers; decode "
                  "thread to CPU %d at its first window (read back then)\n",
                  service_cpu_, (unsigned long long)service_mask_, decode_cpu_);
    s += b;
    std::string hc;
    for (size_t i = 0; i < helper_cpus_.size(); ++i) {
        hc += (i ? "," : "") + std::to_string(helper_cpus_[i]);
    }
    s += "  helpers (ME3): CPUs " + hc + " (" + (o_.helper_cpus.empty() ? "derived from the allowed mask" : "given") +
         "; each helper reads its mask back)\n";
    if (dv_->quiet) {
        std::snprintf(b, sizeof(b),
                      "  waits (ME2b): device flags 0x%x read back (cudaDeviceScheduleBlockingSync 0x%x: stream and "
                      "device synchronizations block); the service polls each window's event (cudaEventQuery, "
                      "_mm_pause; ME3, a replay's window: its stamp in mapped memory); the commit, swap-wait, staging "
                      "and prompt-ID events BlockingSync; the decode thread's round wait backs off (target::wait); the "
                      "pool's host and workers spin (workers park after %d ms)\n",
                      dv_->flags[0], unsigned(cudaDeviceScheduleBlockingSync), o_.park_ms);
    } else {
        std::snprintf(b, sizeof(b), "  waits: two cards, as ME2 (device flags 0x%x / 0x%x read back)\n",
                      dv_->flags[0], dv_->flags[1]);
    }
    s += b;
    return s;
}

} // namespace fn::runtime
