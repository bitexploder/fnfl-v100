// ME3 (the cleanup): the expert cache's residency and placement (cache_residency.h), ME1's and ME2's code moved here
// from expert_cache.cpp. The swap rule, the profile and the share are Strata-V100 3d4ce4e0's (expert_policy.cpp carries
// its MIT notice); the pass ordering (publication after the window's readers, copies in layer order, a layer's own
// wait, a non-blocking commit) follows its STRATA_ADAPT_OVERLAP path (src/program/generate.cpp near lines 9540-9900).

#include "fn/runtime/cache_residency.h"

#include "fn/kernels/experts/experts_cache.h"
#include "fn/kernels/experts/experts_cpu.h"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

namespace fn::runtime {

using cache::cuda_ok;
using cache::fail;
using cache::n_ex;

bool publish_pass(const pass_dev & p, const std::vector<pass_copy> & copies, uint8_t * has, std::string & err) {
    const cudaStream_t cs = static_cast<cudaStream_t>(p.compute), ss = static_cast<cudaStream_t>(p.swap);
    if (p.n > 0 &&
        (!cuda_ok(cudaMemcpyAsync(p.idx_d, p.idx_h, size_t(p.n) * 4, cudaMemcpyHostToDevice, cs), "pass patches",
                  err) ||
         !cuda_ok(cudaMemcpyAsync(p.val_d, p.val_h, size_t(p.n) * 8, cudaMemcpyHostToDevice, cs), "pass patches",
                  err))) {
        return false;
    }
    for (int32_t at = 0; at < p.n; at += fn::experts::max_apply) {
        const int32_t n = std::min(fn::experts::max_apply, p.n - at);
        if (!fn::experts::apply(p.words, p.idx_d + at, p.val_d + at, n, cs)) {
            return fail(err, "pass apply");
        }
    }
    if (!cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(p.ev_pub), cs), "publication event", err) ||
        !cuda_ok(cudaStreamWaitEvent(ss, static_cast<cudaEvent_t>(p.ev_pub), 0), "publication wait", err) ||
        !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(p.ev_t0), ss), "pass timing", err)) {
        return false;
    }
    for (size_t i = 0; i < copies.size(); ++i) {
        const pass_copy & c = copies[i];
        if (i > 0 && c.layer < copies[i - 1].layer) {
            return fail(err, "pass copies out of layer order");
        }
        if (!cuda_ok(cudaMemcpyAsync(c.dst, c.src, size_t(c.bytes), cudaMemcpyHostToDevice, ss), "pass copy", err)) {
            return false;
        }
        if (i + 1 == copies.size() || copies[i + 1].layer != c.layer) {
            if (!cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(p.ev_layer[c.layer]), ss), "layer event", err)) {
                return false;
            }
            has[c.layer] = 1;
            // ME3: the replay's swap wait: the count landed after the copies, needed by the next replay (no counts:
            // ex-test's direct passes)
            if (p.sw_count != nullptr) {
                const uint32_t n = ++p.sw_count[c.layer];
                if (!cache::stream_write(ss, p.sw_done + c.layer, n, "swap count", err) ||
                    !cache::stream_write(cs, p.sw_need + c.layer, n, "swap need", err)) {
                    return false;
                }
            }
        }
    }
    return cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(p.ev_t1), ss), "pass timing", err);
}

bool wait_layer(void * compute, void * ev_layer, void * ev_w0, void * ev_w1, std::string & err) {
    const cudaStream_t cs = static_cast<cudaStream_t>(compute);
    return cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_w0), cs), "swap wait timing", err) &&
           cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(ev_layer), 0), "swap wait", err) &&
           cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_w1), cs), "swap wait timing", err);
}

bool residency::init_layers(model::model & m, int32_t nd, const std::string & profile, int32_t pack_threads,
                            bool zero_backing, std::string & err, bool native_hold) {
    nd_ = nd;
    parts_ = nd == 1 ? 4 : 2;
    // the layers and their classes (entry and payload: one device's bytes)
    lay_.resize(fn::n_layer);
    cls_of_.assign(fn::n_layer, -1);
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (!loader::expert_layer_of(m, l, lay_[size_t(l)], err)) {
            return fail(err, err);
        }
        const loader::expert_layer & L = lay_[size_t(l)];
        const int64_t dentry = L.entry_bytes() / nd_;
        int32_t k = 0;
        while (k < int32_t(cls_.size()) &&
               (cls_[size_t(k)].gu_fmt != int32_t(L.gu_fmt) || cls_[size_t(k)].dn_fmt != int32_t(L.dn_fmt))) {
            ++k;
        }
        if (k == int32_t(cls_.size())) {
            cls c;
            c.gu_fmt = int32_t(L.gu_fmt);
            c.dn_fmt = int32_t(L.dn_fmt);
            c.entry = dentry;
            cls_.push_back(c);
        }
        cls_[size_t(k)].payload += n_ex * dentry;
        cls_[size_t(k)].layers += 1;
        cls_of_[size_t(l)] = k;
        backing_bytes_ += n_ex * L.entry_bytes();
        max_entry_ = std::max(max_entry_, dentry);
        max_part_ = std::max({max_part_, L.gu_bytes, L.dn_bytes});
        if (!fn::experts::cpu::gate_up_supported(L.gu_fmt) || !fn::experts::cpu::down_supported(L.dn_fmt)) {
            return fail(err, "layer " + std::to_string(l) + ": a format the CPU products do not take");
        }
    }
    // ME4: every layer's native views (pointer arithmetic on the model's mappings, extents checked; nothing read)
    native_.assign(size_t(fn::n_layer), fn::experts::cpu::native_layer());
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (!loader::native_expert_layer(m, lay_[size_t(l)], native_[size_t(l)], err)) {
            return fail(err, err);
        }
    }
    // the boot ranking: the profile, or expert order
    if (!profile.empty()) {
        if (!policy::read_profile(profile, fn::n_layer, profile_, err)) {
            return fail(err, err);
        }
    } else {
        profile_.resize(size_t(fn::n_layer) * n_ex);
        for (size_t i = 0; i < profile_.size(); ++i) {
            profile_[i] = int32_t(i);
        }
    }
    // the backing: packed by threads from the mapping, then page-locked
    backing_map_ = size_t(backing_bytes_);
    void * map = mmap(nullptr, backing_map_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) {
        return fail(err, "cannot map " + std::to_string(backing_bytes_) + " bytes of backing");
    }
    backing_base_ = static_cast<uint8_t *>(map);
    madvise(backing_base_, backing_map_, MADV_HUGEPAGE);
    backing_.resize(fn::n_layer);
    {
        size_t off = 0;
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            backing_[size_t(l)] = backing_base_ + off;
            off += size_t(n_ex * lay_[size_t(l)].entry_bytes());
        }
    }
    const cache::clk::time_point tp = cache::clk::now();
    std::atomic<int32_t> next{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> pk;
    for (int32_t t = 0; t < (zero_backing ? 0 : pack_threads); ++t) { // an anonymous mapping reads as zero
        pk.emplace_back([&]() {
            for (int32_t i = next++; i < fn::n_layer * n_ex && !bad; i = next++) {
                const int32_t l = i / n_ex, e = i % n_ex;
                uint8_t * const dst = backing_[size_t(l)] + size_t(e) * size_t(lay_[size_t(l)].entry_bytes());
                if (!loader::pack_expert_entry(m, lay_[size_t(l)], e, dst)) {
                    bad = true;
                }
            }
        });
    }
    for (auto & t : pk) {
        t.join();
    }
    pack_s_ = cache::since(tp);
    if (bad) {
        return fail(err, "packing an expert entry failed");
    }
    const cache::clk::time_point tr = cache::clk::now();
    if (!cuda_ok(cudaHostRegister(backing_base_, backing_map_, cudaHostRegisterPortable), "cudaHostRegister", err)) {
        return false;
    }
    backing_registered_ = true;
    register_s_ = cache::since(tr);
    if (zero_backing && native_hold) { // TEST: the lock check alone (the zero backing differs from the files)
        native_dry_hold_ = init_native(pack_threads, false, err);
        return native_dry_hold_;
    }
    return zero_backing || init_native(pack_threads, true, err);
}

// ME4: the native views' pages held, then checked: each layer's boundary and middle experts' native rows against the
// reference path's reconstruction of the packed backing, byte for byte (a failure fails the init)
bool residency::init_native(int32_t threads, bool check, std::string & err) {
    std::vector<page_range> spans;
    for (const fn::experts::cpu::native_layer & v : native_) {
        // reason: the views' byte spans as addresses, for the page arithmetic
        spans.push_back({reinterpret_cast<uintptr_t>(v.gate), reinterpret_cast<uintptr_t>(v.gate) +
                                                                  uintptr_t(n_ex) * uintptr_t(v.gu_expert)});
        spans.push_back({reinterpret_cast<uintptr_t>(v.up), reinterpret_cast<uintptr_t>(v.up) +
                                                                uintptr_t(n_ex) * uintptr_t(v.gu_expert)});
        spans.push_back({reinterpret_cast<uintptr_t>(v.down), reinterpret_cast<uintptr_t>(v.down) +
                                                                  uintptr_t(n_ex) * uintptr_t(v.dn_expert)});
    }
    if (!native_pages_.hold(spans, threads, err)) {
        return fail(err, err + " (the CPU experts read these pages in place; nothing serves without them)");
    }
    if (!check) {
        return true;
    }
    const cache::clk::time_point tc = cache::clk::now();
    std::vector<uint8_t> scratch(size_t(fn::experts::cpu::scratch_bytes));
    const int32_t probe[6] = {0, 1, n_ex / 2 - 1, n_ex / 2, n_ex - 2, n_ex - 1};
    native_checked_ = 0;
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        const loader::expert_layer & L = lay_[size_t(l)];
        for (const int32_t e : probe) {
            fn::experts::cpu::job j;
            const uint8_t * src = backing_[size_t(l)] + size_t(e) * size_t(L.entry_bytes());
            for (int32_t card = 0; card < 2; ++card) {
                j.gu[card] = src + L.gu_at(card);
                j.dn[card] = src + L.dn_at(card);
            }
            fn::experts::cpu::set_native(native_[size_t(l)], e, j);
            std::string where;
            if (!fn::experts::cpu::native_equal(L.gu_fmt, L.dn_fmt, j, scratch.data(), where)) {
                native_pages_.release();
                return fail(err, "layer " + std::to_string(l) + " expert " + std::to_string(e) +
                                     ": the native rows differ from the packed backing's reconstruction at " + where);
            }
            ++native_checked_;
        }
    }
    native_check_s_ = cache::since(tc);
    native_ready_ = true;
    return true;
}

std::string residency::native_json() const {
    return native_pages_.held() ? native_pages_.json(native_pages_.snapshot()) : std::string("null");
}

std::string residency::native_line() const {
    char b[384];
    if (native_dry_hold_) {
        std::snprintf(b, sizeof(b),
                      "  native CPU expert views: %zu layers built and bounds-checked; pages held for the residency "
                      "check (--ex-native-hold); NOT serving: the backing is left zero (dry run)\n",
                      native_.size());
        return b + ("  " + native_pages_.line()) + "\n";
    }
    if (!native_ready_) {
        std::snprintf(b, sizeof(b),
                      "  native CPU expert views: %zu layers built and bounds-checked; NOT serving: the backing is "
                      "left zero (dry run), so the CPU experts use the reference path (reconstruction from the "
                      "backing)\n",
                      native_.size());
        return b;
    }
    std::snprintf(b, sizeof(b),
                  "  native CPU expert views: %zu layers, serving; %d experts' rows equal to the backing's "
                  "reconstruction, bytewise (%.2f s)\n",
                  native_.size(), native_checked_, native_check_s_);
    return b + ("  " + native_pages_.line()) + "\n";
}

bool residency::init_devices(const cache::devices & dv, std::string & err) {
    const auto vp = [](auto ** p) { return reinterpret_cast<void **>(p); };
    const size_t res_bytes = size_t(fn::n_layer) * n_ex * sizeof(void *);
    const size_t n_words = size_t(w_run() + 1);
    // ME3: the replays' swap-wait times, mapped (each device's address of them below)
    if (!own_.halloc(size_t(2) * fn::n_layer * 2 * 8, vp(&swt_h_), true, err)) {
        return false;
    }
    for (int32_t d = 0; d < nd_; ++d) {
        dev_[d] = dv.dev[d];
        if (!dv.set(d, err) ||
            !cuda_ok(cudaHostGetDevicePointer(vp(&swt_d_[d]), swt_h_ + d * fn::n_layer * 2, 0), "swap wait times",
                     err) ||
            !own_.dalloc(dv, d, size_t(max_part_), vp(&zero_[d]), true, err) ||
            !own_.dalloc(dv, d, n_words * 8, vp(&words_[d]), true, err) ||
            !own_.dalloc(dv, d, size_t(fn::n_layer) * 4, vp(&sw_done_[d]), true, err) ||
            !own_.dalloc(dv, d, size_t(fn::n_layer) * 4, vp(&sw_need_[d]), true, err) ||
            !own_.halloc(res_bytes, vp(&res_h_[d]), false, err) || !own_.halloc(res_bytes, vp(&res_snap_h_[d]), false,
                                                                                err)) {
            return false;
        }
        ev_sw_[d].assign(fn::n_layer, nullptr);
        ev_sww_[d].assign(size_t(fn::n_layer) * 2, nullptr);
        for (auto & e : ev_sw_[d]) {
            if (!own_.event(dv, d, &e, false, false, err)) {
                return false;
            }
        }
        for (auto & e : ev_sww_[d]) { // read_swap_waits synchronizes on them
            if (!own_.event(dv, d, &e, true, true, err)) {
                return false;
            }
        }
        // host waits: ev_pt_ (the commit, the link probe), ev_rfp_ (a refill publication's patches applied)
        if (!own_.event(dv, d, &ev_pub_[d], false, false, err) || !own_.event(dv, d, &ev_cp_[d], false, false, err) ||
            !own_.event(dv, d, &ev_pt_[d][0], true, true, err) || !own_.event(dv, d, &ev_pt_[d][1], true, true, err) ||
            !own_.event(dv, d, &ev_rfp_[d], false, true, err)) {
            return false;
        }
        cudaStream_t ss = nullptr;
        if (!dv.set(d, err) || !cuda_ok(cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking), "swap stream", err)) {
            return false;
        }
        swap_[d] = ss;
        wait_has_[d].assign(fn::n_layer, 0);
        waited_[d].assign(fn::n_layer, 0);
    }
    // ME9: admission on the one-device path: a completion event a record, the two patch batches
    if (nd_ == 1) {
        for (void *& e : adm_ev_) {
            if (!own_.event(dv, 0, &e, false, true, err)) {
                return false;
            }
        }
        for (int32_t b = 0; b < 2; ++b) {
            constexpr size_t n = adm::max_recs + 16;
            if (!own_.dalloc(dv, 0, n * 4, vp(&adm_idx_[b]), false, err) ||
                !own_.dalloc(dv, 0, n * 8, vp(&adm_val_[b]), false, err) ||
                !own_.halloc(n * 4, vp(&adm_idx_h_[b]), false, err) ||
                !own_.halloc(n * 8, vp(&adm_val_h_[b]), false, err) ||
                !own_.event(dv, 0, &adm_bev_[b], false, true, err)) {
                return false;
            }
        }
        adm_flag_.assign(size_t(fn::n_layer) * n_ex, 0);
        adm_on_ = true;
    }
    return true;
}

bool residency::measure(const cache::devices & dv, int64_t device_cap_bytes, int64_t & fr_min, std::string & err) {
    device_cap_ = device_cap_bytes;
    fr_min = -1;
    for (int32_t d = 0; d < nd_; ++d) {
        size_t fr = 0, tot = 0;
        if (!dv.set(d, err) || !cuda_ok(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo", err)) {
            return false;
        }
        free_before_[d] = int64_t(fr);
        used_before_[d] = int64_t(tot) - int64_t(fr);
        // the ceiling (device_cap_bytes): what the device may still take under it, counted as free
        const int64_t left =
            device_cap_bytes >= 0 ? std::min(int64_t(fr), device_cap_bytes - used_before_[d]) : int64_t(fr);
        fr_min = fr_min < 0 ? left : std::min(fr_min, left);
    }
    return true;
}

bool residency::init_pool(const cache::devices & dv, int64_t fr_min, const pool_options & po, int64_t lent_bytes,
                          std::string & err) {
    const auto vp = [](auto ** p) { return reinterpret_cast<void **>(p); };
    swap_overlap_ = po.swap_overlap;
    lent_bytes_ = lent_bytes;
    // the pass buffers come out of what is left: at most 2 x (max(swaps, zone slots)) patches a device, the zone's
    // slots at most its bytes over the smallest entry
    zone_buf_ = 0;
    int64_t min_entry = max_entry_;
    for (const auto & L : lay_) {
        zone_buf_ = std::max(zone_buf_, n_ex * (L.entry_bytes() / nd_));
        min_entry = std::min(min_entry, L.entry_bytes() / nd_);
    }
    max_moves_ = std::max<int32_t>(po.max_swaps, int32_t((2 * zone_buf_ + lent_bytes_) / min_entry));
    const size_t pass_bytes = size_t(3) * size_t(max_moves_) * 12; // ME3: and the refill's patches
    pool_bytes_ = po.pool_bytes >= 0 ? po.pool_bytes : fr_min - po.reserve_bytes - int64_t(pass_bytes) - (1 << 20);
    pool_bytes_ = pool_bytes_ / 256 * 256;
    if (pool_bytes_ < 2 * zone_buf_ + lent_bytes_ + 64 * max_entry_) {
        return fail(err, "the pool (" + cache::gib(pool_bytes_) +
                             ") holds less than the prompt's two staging buffers (" +
                             cache::gib(2 * zone_buf_) + "), the lent region (" + cache::gib(lent_bytes_) +
                             ") and 64 entries");
    }
    for (int32_t d = 0; d < nd_; ++d) {
        // ME3: the refill's publications, a patch a zone slot
        if (!own_.dalloc(dv, d, size_t(pool_bytes_), vp(&pool_[d]), false, err) ||
            !own_.dalloc(dv, d, size_t(2 * max_moves_) * 4, vp(&pass_idx_[d]), false, err) ||
            !own_.dalloc(dv, d, size_t(2 * max_moves_) * 8, vp(&pass_val_[d]), false, err) ||
            !own_.halloc(size_t(2 * max_moves_) * 4, vp(&pass_idx_h_[d]), false, err) ||
            !own_.halloc(size_t(2 * max_moves_) * 8, vp(&pass_val_h_[d]), false, err) ||
            !own_.dalloc(dv, d, size_t(max_moves_) * 4, vp(&rf_idx_[d]), false, err) ||
            !own_.dalloc(dv, d, size_t(max_moves_) * 8, vp(&rf_val_[d]), false, err) ||
            !own_.halloc(size_t(max_moves_) * 4, vp(&rf_idx_h_[d]), false, err) ||
            !own_.halloc(size_t(max_moves_) * 8, vp(&rf_val_h_[d]), false, err)) {
            return false;
        }
    }
    // the zone's slots span the staging buffers and the lent region after them (zone_at: offsets from zone_off_)
    lent_off_ = pool_bytes_ - lent_bytes_;
    zone_off_ = lent_off_ - 2 * zone_buf_;
    main_bytes_ = zone_off_;
    policy::place_in pi;
    pi.class_of_layer = cls_of_;
    pi.zone_bytes = 2 * zone_buf_ + lent_bytes_;
    for (auto & c : cls_) {
        const int64_t share = int64_t(double(main_bytes_) * double(c.payload) / double(backing_bytes_ / nd_));
        c.count = int32_t(std::min<int64_t>(share / c.entry, n_ex * c.layers));
        pi.entry.push_back(c.entry);
        pi.count.push_back(c.count);
    }
    boot_ = policy::place(pi, profile_);
    zone_first_ = boot_.n_main;
    if (int32_t(boot_.slot_key.size()) - zone_first_ > max_moves_) {
        return fail(err, "the zone's slots exceed the pass buffers");
    }
    {
        int64_t at = 0;
        for (int32_t s = 0; s < boot_.n_main; ++s) {
            slot_off_.push_back(at);
            at += cls_[size_t(boot_.slot_cls[size_t(s)])].entry;
        }
        for (size_t j = 0; j < boot_.zone_at.size(); ++j) {
            slot_off_.push_back(zone_off_ + boot_.zone_at[j]);
        }
    }
    slot_key_.assign(slot_off_.size(), -1);
    entry_slot_.assign(size_t(fn::n_layer) * n_ex, -1);
    usage_.assign(size_t(fn::n_layer) * n_ex, 0.0f);
    heat_.assign(size_t(fn::n_layer) * n_ex, 0.0);
    return true;
}

// PcieAuto's link probe (generate.cpp near line 1894): 256 MiB from the page-locked backing into device 0's pool, a
// warm-up, then four event-timed bursts; the best rate
bool residency::probe_link(const cache::devices & dv, std::string & err) {
    const int64_t bytes = std::min<int64_t>(int64_t(256) << 20, std::min(backing_bytes_, pool_bytes_));
    const cudaStream_t cs = static_cast<cudaStream_t>(dv.copy[0]);
    cudaEvent_t a = static_cast<cudaEvent_t>(ev_pt_[0][0]), b = static_cast<cudaEvent_t>(ev_pt_[0][1]);
    if (!dv.set(0, err) ||
        !cuda_ok(cudaMemcpyAsync(pool_[0], backing_base_, size_t(bytes), cudaMemcpyHostToDevice, cs), "probe", err) ||
        !cuda_ok(cudaStreamSynchronize(cs), "probe", err)) {
        return false;
    }
    double best = 0;
    for (int32_t i = 0; i < 4; ++i) {
        float ms = 0;
        if (!cuda_ok(cudaEventRecord(a, cs), "probe", err) ||
            !cuda_ok(cudaMemcpyAsync(pool_[0], backing_base_, size_t(bytes), cudaMemcpyHostToDevice, cs), "probe",
                     err) ||
            !cuda_ok(cudaEventRecord(b, cs), "probe", err) || !cuda_ok(cudaEventSynchronize(b), "probe", err) ||
            !cuda_ok(cudaEventElapsedTime(&ms, a, b), "probe", err)) {
            return false;
        }
        if (ms > 0) {
            best = std::max(best, double(bytes) / (double(ms) * 1e-3) / 1e9);
        }
    }
    link_gbps_ = best;
    return true;
}

void residency::release(const cache::devices & dv) {
    for (int32_t d = 0; d < 2; ++d) {
        if (swap_[d] != nullptr) {
            cudaSetDevice(dv.dev[d]);
            cudaStreamDestroy(static_cast<cudaStream_t>(swap_[d]));
            swap_[d] = nullptr;
        }
    }
    own_.release(dv);
    native_ready_ = false;
    native_dry_hold_ = false;
    if (native_pages_.held()) { // Fable's R4-15 ruling: the residency at the end, beside the hold's and each request's
        std::fprintf(stderr, "expert cache: native expert pages at release: %s\n", native_json().c_str());
    }
    native_pages_.release(); // the pool stopped (expert_cache::release): no job reads a view
    if (backing_registered_) {
        cudaHostUnregister(backing_base_);
        backing_registered_ = false;
    }
    if (backing_base_ != nullptr) {
        munmap(backing_base_, backing_map_);
        backing_base_ = nullptr;
    }
}

int64_t residency::part_off(int32_t layer, int32_t card, bool down) const {
    const loader::expert_layer & L = lay_[size_t(layer)];
    if (nd_ == 2) {
        return down ? L.gu_bytes : 0;
    }
    return down ? L.dn_at(card) : L.gu_at(card);
}

const uint8_t * residency::entry_src(int32_t layer, int32_t e, int32_t d) const {
    const int64_t entry = lay_[size_t(layer)].entry_bytes();
    return backing_[size_t(layer)] + e * entry + (nd_ == 2 ? d * (entry / 2) : 0);
}

int32_t residency::resident() const {
    int32_t n = 0;
    for (const int32_t k : slot_key_) {
        n += k >= 0;
    }
    return n;
}

void residency::place_key(int32_t s, int32_t key) {
    if (!adm_flag_.empty() && adm_flag_[size_t(key)] == 2) {
        adm_flag_[size_t(key)] = 0; // ME9: resident again
    }
    slot_key_[size_t(s)] = key;
    entry_slot_[size_t(key)] = s;
    for (int32_t d = 0; d < nd_; ++d) {
        res_h_[d][key] = slot_addr(d, s);
    }
}

void residency::unplace(int32_t s) {
    const int32_t key = slot_key_[size_t(s)];
    if (key >= 0) {
        adm_test_note(key, ev_left);
        if (!adm_flag_.empty() && adm_flag_[size_t(key)] == 1) {
            adm_flag_[size_t(key)] = 0; // ME9: an admitted key evicted
        }
        entry_slot_[size_t(key)] = -1;
        for (int32_t d = 0; d < nd_; ++d) {
            res_h_[d][key] = nullptr;
        }
    }
    slot_key_[size_t(s)] = -1;
}

bool residency::upload_rows(const cache::devices & dv, std::string & err) {
    for (int32_t d = 0; d < nd_; ++d) {
        if (!dv.set(d, err) || !cuda_ok(cudaDeviceSynchronize(), "rows", err) ||
            !cuda_ok(cudaMemcpy(words_[d] + w_res(0, 0), res_h_[d], size_t(fn::n_layer) * n_ex * 8,
                                cudaMemcpyHostToDevice),
                     "rows", err)) {
            return false;
        }
    }
    return true;
}

bool residency::fill(const cache::devices & dv, const std::vector<int32_t> & target, std::string & err) {
    for (int32_t d = 0; d < nd_; ++d) {
        if (!dv.set(d, err) || !cuda_ok(cudaDeviceSynchronize(), "fill", err)) {
            return false;
        }
    }
    std::vector<int32_t> moved;
    for (size_t s = 0; s < slot_key_.size(); ++s) {
        const int32_t k = s < target.size() ? target[s] : -1;
        if (slot_key_[s] != k) {
            unplace(int32_t(s));
            moved.push_back(int32_t(s));
        }
    }
    for (const int32_t s : moved) {
        const int32_t k = target[size_t(s)];
        if (k < 0) {
            continue;
        }
        place_key(s, k);
        const int32_t l = k / n_ex;
        for (int32_t d = 0; d < nd_; ++d) {
            if (!dv.set(d, err) ||
                !cuda_ok(cudaMemcpyAsync(pool_[d] + slot_off_[size_t(s)], entry_src(l, k % n_ex, d),
                                         size_t(lay_[size_t(l)].entry_bytes() / nd_), cudaMemcpyHostToDevice,
                                         static_cast<cudaStream_t>(dv.copy[d])),
                         "fill", err)) {
                return false;
            }
        }
    }
    for (int32_t d = 0; d < nd_; ++d) {
        if (!dv.set(d, err) || !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(dv.copy[d])), "fill", err)) {
            return false;
        }
    }
    return upload_rows(dv, err);
}

// a pass: each (slot, key) move makes the slot's key a miss and key resident in it. The host tables at once; on each
// device the rows patched on the compute stream, then (after the lease event) the copies on the swap stream in layer
// order. Null streams: everything synchronous (drain). An adapt pass counts its swaps here; the loan counts a refill's
bool residency::publish(const cache::devices & dv, std::vector<move> moves, bool refill, void * const stream[2],
                        int64_t & bytes, std::string & err) {
    const cache::clk::time_point t0 = cache::clk::now();
    std::stable_sort(moves.begin(), moves.end(),
                     [](const auto & a, const auto & b) { return a.second / n_ex < b.second / n_ex; });
    int32_t n = 0;
    std::vector<pass_copy> copies[2];
    bytes = 0;
    for (const auto & mv : moves) {
        const int32_t s = mv.first, k = mv.second, old = slot_key_[size_t(s)], l = k / n_ex;
        if (k < 0 || entry_slot_[size_t(k)] >= 0 || (old >= 0 && old / n_ex != l)) {
            return fail(err, "a pass move out of rule");
        }
        for (int32_t d = 0; d < nd_; ++d) {
            if (old >= 0) {
                pass_idx_h_[d][n] = int32_t(w_res(old / n_ex, old % n_ex));
                pass_val_h_[d][n] = 0;
            }
            pass_idx_h_[d][n + (old >= 0)] = int32_t(w_res(l, k % n_ex));
            pass_val_h_[d][n + (old >= 0)] = uint64_t(reinterpret_cast<uintptr_t>(slot_addr(d, s)));
            const int64_t dentry = lay_[size_t(l)].entry_bytes() / nd_;
            copies[d].push_back({l, pool_[d] + slot_off_[size_t(s)], entry_src(l, k % n_ex, d), dentry});
            bytes += dentry;
        }
        n += 1 + (old >= 0);
        unplace(s);
        place_key(s, k);
    }
    if (!refill) {
        n_.swap_bytes += bytes;
        n_.swaps += int64_t(moves.size());
        ++n_.passes;
    }
    if (moves.empty()) {
        return true;
    }
    if (stream == nullptr) {
        for (int32_t d = 0; d < nd_; ++d) {
            if (!dv.set(d, err) || !cuda_ok(cudaDeviceSynchronize(), "pass", err)) {
                return false;
            }
            for (const pass_copy & c : copies[d]) {
                if (!cuda_ok(cudaMemcpy(c.dst, c.src, size_t(c.bytes), cudaMemcpyHostToDevice), "pass copy", err)) {
                    return false;
                }
            }
        }
        n_.publish_wait.n += 1;
        n_.publish_wait.s += cache::since(t0);
        return upload_rows(dv, err);
    }
    for (int32_t d = 0; d < nd_; ++d) {
        pass_dev p;
        p.words = words_[d];
        p.idx_d = pass_idx_[d];
        p.val_d = pass_val_[d];
        p.idx_h = pass_idx_h_[d];
        p.val_h = pass_val_h_[d];
        p.n = n;
        p.compute = stream[d];
        p.swap = swap_[d];
        p.ev_pub = ev_pub_[d];
        p.ev_layer = ev_sw_[d].data();
        p.ev_t0 = ev_pt_[d][0];
        p.ev_t1 = ev_pt_[d][1];
        p.sw_count = sw_count_[d];
        p.sw_done = sw_done_[d];
        p.sw_need = sw_need_[d];
        if (!dv.set(d, err)) {
            return false;
        }
        // a refill's copies also after the staging copies into the zone (the copy stream)
        if (refill && (!cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_cp_[d]),
                                                static_cast<cudaStream_t>(dv.copy[d])),
                                "refill event", err) ||
                       !cuda_ok(cudaStreamWaitEvent(static_cast<cudaStream_t>(swap_[d]),
                                                    static_cast<cudaEvent_t>(ev_cp_[d]), 0),
                                "refill wait", err))) {
            return false;
        }
        if (!publish_pass(p, copies[d], wait_has_[d].data(), err)) {
            return false;
        }
    }
    pending_ = true;
    pending_refill_ = refill;
    if (!swap_overlap_) {
        for (int32_t d = 0; d < nd_; ++d) {
            if (!dv.set(d, err) || !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(swap_[d])), "pass", err)) {
                return false;
            }
        }
        if (!try_commit(dv, true, err)) {
            return false;
        }
    }
    n_.publish_wait.n += 1;
    n_.publish_wait.s += cache::since(t0);
    return true;
}

// the pending pass committed once every device's copies have landed (wait: synchronize for them); the waits not
// yet enqueued are not needed after it
bool residency::try_commit(const cache::devices & dv, bool wait, std::string & err) {
    if (!pending_) {
        return true;
    }
    for (int32_t d = 0; d < nd_; ++d) {
        const cudaEvent_t e = static_cast<cudaEvent_t>(ev_pt_[d][1]);
        if (!dv.set(d, err)) {
            return false;
        }
        if (wait) {
            if (!cuda_ok(cudaEventSynchronize(e), "commit", err)) {
                return false;
            }
            continue;
        }
        const cudaError_t q = cudaEventQuery(e);
        if (q == cudaErrorNotReady) {
            return true;
        }
        if (!cuda_ok(q, "commit", err)) {
            return false;
        }
    }
    float ms = 0;
    if (!dv.set(0, err) || !cuda_ok(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(ev_pt_[0][0]),
                                                         static_cast<cudaEvent_t>(ev_pt_[0][1])),
                                    "pass time", err)) {
        return false;
    }
    (pending_refill_ ? n_.refill_copy_ms : n_.swap_copy_ms) += double(ms);
    for (int32_t d = 0; d < nd_; ++d) {
        std::fill(wait_has_[d].begin(), wait_has_[d].end(), uint8_t(0));
    }
    pending_ = false;
    return true;
}

bool residency::eager_wait(int32_t d, int32_t layer, void * stream, std::string & err) {
    if (wait_has_[d][size_t(layer)] == 0) {
        return true;
    }
    if (!wait_layer(stream, ev_sw_[d][size_t(layer)], ev_sww_[d][size_t(2 * layer)],
                    ev_sww_[d][size_t(2 * layer + 1)], err)) {
        return false;
    }
    wait_has_[d][size_t(layer)] = 0;
    waited_[d][size_t(layer)] = 1;
    return true;
}

// ME3: a replay waits for the layer's latest copies (their count landed), already complete unless a pass was
// published before its launch (pass_begin); a kernel, its times read for the layers the pass copied
bool residency::replay_wait(int32_t d, int32_t layer, void * stream, std::string & err) const {
    const fn::experts::wait_args wa{sw_done_[d] + layer, sw_need_[d] + layer, 0, swt_d_[d] + 2 * layer};
    return fn::experts::wait_word(wa, stream) || fail(err, "swap wait");
}

void residency::mark_replay_waits() {
    for (int32_t d = 0; d < nd_; ++d) {
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            if (wait_has_[d][size_t(l)] != 0) {
                wait_has_[d][size_t(l)] = 0;
                waited_[d][size_t(l)] = 2; // the replay's wait kernel
            }
        }
    }
}

// the compute streams' swap waits enqueued so far (every one before the service's last window event, so complete
// once the service is idle; a replay's once its round has completed)
bool residency::read_swap_waits(const cache::devices & dv, std::string & err) {
    for (int32_t d = 0; d < nd_; ++d) {
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            if (waited_[d][size_t(l)] == 0) {
                continue;
            }
            const bool replay = waited_[d][size_t(l)] == 2;
            waited_[d][size_t(l)] = 0;
            float ms = 0;
            if (replay) {
                const volatile uint64_t * const t = swt_h_ + (d * fn::n_layer + l) * 2;
                ms = t[1] > t[0] ? float(double(t[1] - t[0]) * 1e-6) : 0.0f;
            } else if (!dv.set(d, err) ||
                       !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(ev_sww_[d][size_t(2 * l + 1)])),
                                "swap wait", err) ||
                       !cuda_ok(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(ev_sww_[d][size_t(2 * l)]),
                                                     static_cast<cudaEvent_t>(ev_sww_[d][size_t(2 * l + 1)])),
                                "swap wait", err)) {
                return false;
            }
            if (d == 0) {
                n_.swap_wait.n += 1;
                n_.swap_wait.s += double(ms) * 1e-3;
            }
        }
    }
    return true;
}

// ME3: a refill's landed moves into the tables: host placement, then each device's row patches on its compute stream
// (after the previous publication's, whose host buffers they reuse); null streams: the rows uploaded synchronously.
// A move whose key is resident again or whose slot was filled is skipped
bool residency::publish_landed(const cache::devices & dv, const move * mv, size_t n_mv, void * const stream[2],
                               int64_t & skipped, std::string & err) {
    if (n_mv == 0) {
        return true;
    }
    for (int32_t d = 0; d < nd_ && stream != nullptr; ++d) {
        if (!dv.set(d, err) ||
            !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(ev_rfp_[d])), "refill patches", err)) {
            return false;
        }
    }
    int32_t n = 0;
    for (size_t i = 0; i < n_mv; ++i) {
        const int32_t s = mv[i].first, k = mv[i].second;
        if (entry_slot_[size_t(k)] >= 0 || slot_key_[size_t(s)] >= 0) {
            ++skipped;
            continue;
        }
        place_key(s, k);
        for (int32_t d = 0; d < nd_; ++d) {
            rf_idx_h_[d][n] = int32_t(w_res(k / n_ex, k % n_ex));
            rf_val_h_[d][n] = uint64_t(reinterpret_cast<uintptr_t>(slot_addr(d, s)));
        }
        ++n;
    }
    if (n > 0 && stream == nullptr && !upload_rows(dv, err)) {
        return false;
    }
    for (int32_t d = 0; d < nd_ && n > 0 && stream != nullptr; ++d) {
        const cudaStream_t cs = static_cast<cudaStream_t>(stream[d]);
        if (!dv.set(d, err) ||
            !cuda_ok(cudaMemcpyAsync(rf_idx_[d], rf_idx_h_[d], size_t(n) * 4, cudaMemcpyHostToDevice, cs),
                     "refill patches", err) ||
            !cuda_ok(cudaMemcpyAsync(rf_val_[d], rf_val_h_[d], size_t(n) * 8, cudaMemcpyHostToDevice, cs),
                     "refill patches", err)) {
            return false;
        }
        for (int32_t at = 0; at < n; at += fn::experts::max_apply) {
            if (!fn::experts::apply(words_[d], rf_idx_[d] + at, rf_val_[d] + at,
                                    std::min(fn::experts::max_apply, n - at), cs)) {
                return fail(err, "refill apply");
            }
        }
        if (!cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_rfp_[d]), cs), "refill patches", err)) {
            return false;
        }
    }
    return true;
}

// ME9: admission's two row-patch batches (fn/runtime/admission.h). A batch's host buffers are reused only after its
// previous submission completed (one host wait on a round-old event); null stream: the rows uploaded synchronously
template <typename F>
bool residency::adm_batch(const cache::devices & dv, int32_t b, void * stream, F fill, std::string & err) {
    if (!dv.set(0, err)) {
        return false;
    }
    if (adm_bev_live_[b] && !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(adm_bev_[b])), "admission", err)) {
        return false;
    }
    adm_bev_live_[b] = false;
    const int32_t n = fill(adm_idx_h_[b], adm_val_h_[b]);
    if (n == 0) {
        return true;
    }
    if (stream == nullptr) {
        return upload_rows(dv, err);
    }
    const cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if (!cuda_ok(cudaMemcpyAsync(adm_idx_[b], adm_idx_h_[b], size_t(n) * 4, cudaMemcpyHostToDevice, cs), "admission",
                 err) ||
        !cuda_ok(cudaMemcpyAsync(adm_val_[b], adm_val_h_[b], size_t(n) * 8, cudaMemcpyHostToDevice, cs), "admission",
                 err)) {
        return false;
    }
    if (!fn::experts::apply(words_[0], adm_idx_[b], adm_val_[b], n, cs)) {
        return fail(err, "admission apply");
    }
    if (!cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(adm_bev_[b]), cs), "admission", err)) {
        return false;
    }
    adm_bev_live_[b] = true;
    return true;
}

bool residency::adm_publish(const cache::devices & dv, bool failed, bool sync, void * const stream[2],
                            std::string & err) {
    if (!adm_on_) {
        return true;
    }
    if (failed || adm_poisoned_.exchange(false)) { // a kept failure, an abandoned pass or released waits
        for (int32_t i = 0; i < adm::max_recs; ++i) {
            n_.adm_invalidated += adm_.invalidate(i) ? 1 : 0;
        }
        return true;
    }
    if (!dv.set(0, err)) {
        return false;
    }
    for (int32_t i = 0; i < adm::max_recs; ++i) {
        if (adm_.at(i).st != adm::state::copying) {
            continue;
        }
        const cudaEvent_t e = static_cast<cudaEvent_t>(adm_ev_[i]);
        const cudaError_t q = sync ? cudaEventSynchronize(e) : cudaEventQuery(e);
        if (q == cudaErrorNotReady) {
            continue; // still copying: the next boundary
        }
        if (!cuda_ok(q, "admission copy", err)) {
            return false;
        }
        n_.adm_landed += adm_.land(i) ? 1 : 0;
    }
    return adm_batch(dv, 0, stream == nullptr ? nullptr : stream[0], [&](int32_t * idx, uint64_t * val) {
        int32_t n = 0;
        for (int32_t i = 0; i < adm::max_recs; ++i) {
            if (adm_.at(i).st != adm::state::landed) {
                continue;
            }
            const int32_t layer = adm_.at(i).layer;
            int32_t s = -1, k = -1;
            adm_.publish(i, s, k);
            if (entry_slot_[size_t(k)] >= 0 || slot_key_[size_t(s)] >= 0) {
                ++n_.adm_collisions; // the key resident or the slot filled meanwhile: the slot stays reserved
                adm_test_note(k, ev_collided);
                adm_.reserve(layer, s);
                continue;
            }
            place_key(s, k);
            adm_flag_[size_t(k)] = 1;
            adm_test_note(k, ev_published);
            idx[n] = int32_t(w_res(k / n_ex, k % n_ex));
            val[n] = uint64_t(reinterpret_cast<uintptr_t>(slot_addr(0, s)));
            ++n;
            ++n_.adm_published;
        }
        return n;
    }, err);
}

bool residency::adm_reserve(const cache::devices & dv, void * const stream[2], std::string & err) {
    if (!adm_on_) {
        return true;
    }
    // orphaned slots (empty, unreserved) by their boot key's layer: adopted before any resident is retired
    std::vector<std::vector<int32_t>> orphans(fn::n_layer);
    for (int32_t s = 0; s < n_slots(); ++s) {
        const int32_t bk = boot_.slot_key[size_t(s)];
        if (slot_key_[size_t(s)] < 0 && bk >= 0 && adm_.of_slot(s) < 0) {
            orphans[size_t(bk / n_ex)].push_back(s);
        }
    }
    return adm_batch(dv, 1, stream == nullptr ? nullptr : stream[0], [&](int32_t * idx, uint64_t * val) {
        int32_t n = 0;
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            int32_t need = adm::per_layer - adm_.held(l);
            for (size_t o = 0; need > 0 && o < orphans[size_t(l)].size(); ++o) {
                if (adm_.reserve(l, orphans[size_t(l)][o])) {
                    ++n_.adm_adopted;
                    --need;
                }
            }
            if (need <= 0) {
                continue;
            }
            std::vector<adm::cand> c;
            for (int32_t e = 0; e < n_ex; ++e) {
                const int32_t key = l * n_ex + e, s = entry_slot_[size_t(key)];
                if (s >= 0) {
                    c.push_back({s, key, usage_[size_t(key)]});
                }
            }
            for (const adm::cand & v : adm::victims(c, need)) {
                adm_test_note(v.key, ev_retired);
                unplace(v.slot);
                adm_flag_[size_t(v.key)] = 2;
                adm_.reserve(l, v.slot);
                idx[n] = int32_t(w_res(l, v.key % n_ex));
                val[n] = 0;
                ++n;
                ++n_.adm_retired;
            }
        }
        return n;
    }, err);
}

void residency::adm_reset() {
    adm_.discard_all();
    std::fill(adm_flag_.begin(), adm_flag_.end(), uint8_t(0));
    adm_poisoned_.store(false);
}

int32_t residency::adm_free(int32_t layer, int32_t (&out)[adm::per_layer]) const {
    int32_t n = 0;
    for (int32_t i = layer * adm::per_layer; adm_on_ && i < (layer + 1) * adm::per_layer; ++i) {
        if (adm_.at(i).st == adm::state::empty) {
            out[n++] = i;
        }
    }
    return n;
}

std::string residency::adm_line() const {
    if (!adm_on_) {
        return "  admission (ME9): off (two devices)\n";
    }
    int64_t bytes = 0;
    for (int32_t i = 0; i < adm::max_recs; ++i) {
        if (adm_.at(i).st != adm::state::none) {
            bytes += slot_bytes(adm_.at(i).slot);
        }
    }
    char b[200];
    std::snprintf(b, sizeof(b),
                  "  admission (ME9): %d of %d slots reserved now (%.3f GiB, out of the published capacity), up to "
                  "%d a layer\n",
                  adm_.held_all(), n_slots(), double(bytes) / 1073741824.0, adm::per_layer);
    return b;
}

int32_t residency::retire(int32_t s) {
    const int32_t k = slot_key_[size_t(s)];
    if (k >= 0) {
        unplace(s);
    }
    return k;
}

bool residency::upload_rows_ordered(const cache::devices & dv, void * const stream[2], std::string & err) {
    for (int32_t d = 0; d < nd_; ++d) {
        std::memcpy(res_snap_h_[d], res_h_[d], size_t(fn::n_layer) * n_ex * 8);
        if (!dv.set(d, err) ||
            !cuda_ok(cudaMemcpyAsync(words_[d] + w_res(0, 0), res_snap_h_[d], size_t(fn::n_layer) * n_ex * 8,
                                     cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream[d])),
                     "residency rows", err)) {
            return false;
        }
    }
    return true;
}

std::vector<residency::move> residency::choose(const policy::adapt_options & a) {
    // ME9: a key whose admission copy is still in flight is no candidate (it lands in its reserved slot): the
    // chooser reads a copy of the usage with those keys' zeroed; the decay below is the real usage's
    std::vector<float> masked;
    const float * use = usage_.data();
    if (adm_on_ && adm_.in_state(adm::state::copying) > 0) {
        masked = usage_;
        for (int32_t i = 0; i < adm::max_recs; ++i) {
            if (adm_.at(i).st == adm::state::copying) {
                masked[size_t(adm_.at(i).key)] = 0.0f;
            }
        }
        use = masked.data();
    }
    const std::vector<policy::swap> sw = policy::choose_swaps(fn::n_layer, use, entry_slot_.data(), a);
    policy::decay_usage(usage_, heat_, a.decay);
    std::vector<move> moves;
    for (const policy::swap & x : sw) {
        moves.emplace_back(entry_slot_[size_t(x.layer * n_ex + x.out)], x.layer * n_ex + x.in);
    }
    return moves;
}

void residency::reset_usage() {
    std::fill(usage_.begin(), usage_.end(), 0.0f);
    std::fill(heat_.begin(), heat_.end(), 0.0);
}

bool residency::save_profile(const std::string & path, const std::vector<int32_t> & lent, std::string & err) {
    if (path.empty()) {
        return true;
    }
    std::vector<uint8_t> res(size_t(fn::n_layer) * n_ex, 0);
    for (const int32_t k : slot_key_) {
        if (k >= 0) {
            res[size_t(k)] = 1;
        }
    }
    for (const int32_t k : lent) { // a prompt's lent slots count as resident (Strata saves with them back)
        if (k >= 0) {
            res[size_t(k)] = 1;
        }
    }
    const std::vector<int32_t> ranked = policy::rank_learned(fn::n_layer, res, heat_, profile_);
    learned_dirty_ = false;
    if (!policy::write_profile(path, fn::n_layer, ranked, err)) {
        return fail(err, err);
    }
    std::fprintf(stderr, "expert cache: learned profile saved to %s\n", path.c_str());
    return true;
}

bool residency::verify_zone(const cache::devices & dv, int64_t & checked, int64_t & bad, std::string & err) const {
    checked = bad = 0;
    for (int32_t d = 0; d < nd_; ++d) {
        if (!dv.set(d, err) || !cuda_ok(cudaDeviceSynchronize(), "verify", err)) {
            return false;
        }
    }
    std::vector<uint8_t> h(static_cast<size_t>(max_entry_), 0);
    for (size_t s = size_t(zone_first_); s < slot_key_.size(); ++s) {
        const int32_t k = slot_key_[s];
        if (k < 0) {
            continue;
        }
        const int32_t l = k / n_ex;
        const int64_t dentry = lay_[size_t(l)].entry_bytes() / nd_;
        bool same = true;
        for (int32_t d = 0; d < nd_; ++d) {
            if (!dv.set(d, err) ||
                !cuda_ok(cudaMemcpy(h.data(), pool_[d] + slot_off_[s], size_t(dentry), cudaMemcpyDeviceToHost),
                         "verify", err)) {
                return false;
            }
            same = same && std::memcmp(h.data(), entry_src(l, k % n_ex, d), size_t(dentry)) == 0;
        }
        ++checked;
        bad += same ? 0 : 1;
    }
    return true;
}

bool residency::adm_test_poison(const cache::devices & dv, std::string & err) const {
    for (int32_t i = 0; i < adm::max_recs; ++i) {
        if (adm_.at(i).st != adm::state::none && !poison_slot(dv, adm_.at(i).slot, err)) {
            return false;
        }
    }
    return true;
}

bool residency::adm_test_verify(const cache::devices & dv, int64_t & checked, int64_t & bad, std::string & err) const {
    checked = bad = 0;
    for (int32_t d = 0; d < nd_; ++d) {
        if (!dv.set(d, err) || !cuda_ok(cudaDeviceSynchronize(), "verify", err)) {
            return false;
        }
    }
    std::vector<uint8_t> h(static_cast<size_t>(max_entry_), 0);
    for (size_t s = 0; s < slot_key_.size(); ++s) {
        const int32_t k = slot_key_[s];
        if (k < 0 || adm_flag(k) != 1) {
            continue;
        }
        const int32_t l = k / n_ex;
        const int64_t dentry = lay_[size_t(l)].entry_bytes() / nd_;
        bool same = true;
        for (int32_t d = 0; d < nd_; ++d) {
            if (!dv.set(d, err) ||
                !cuda_ok(cudaMemcpy(h.data(), pool_[d] + slot_off_[s], size_t(dentry), cudaMemcpyDeviceToHost),
                         "verify", err)) {
                return false;
            }
            same = same && std::memcmp(h.data(), entry_src(l, k % n_ex, d), size_t(dentry)) == 0;
        }
        ++checked;
        bad += same ? 0 : 1;
    }
    return true;
}

bool residency::adm_test_evict(const cache::devices & dv, const std::vector<int32_t> & routed,
                               std::vector<int32_t> & keys, std::string & err) {
    keys.clear();
    if (!adm_on_) {
        return fail(err, "adm_test_evict: admission is off (two devices)");
    }
    if (routed.size() != size_t(fn::n_layer) * 10) {
        return fail(err, "adm_test_evict: the routed experts of every layer");
    }
    const bool ok = adm_batch(dv, 1, nullptr, [&](int32_t * idx, uint64_t * val) {
        int32_t n = 0;
        for (int32_t l = 0; l < fn::n_layer; ++l) {
            int32_t best = -1;
            for (int32_t i = 0; i < 10; ++i) {
                const int32_t e = routed[size_t(l) * 10 + size_t(i)];
                const int32_t key = l * n_ex + e;
                if (e >= 0 && e < n_ex && entry_slot_[size_t(key)] >= 0 && !adm_.in_flight(key) &&
                    (best < 0 || usage_[size_t(key)] > usage_[size_t(best)])) {
                    best = key;
                }
            }
            if (best < 0) {
                continue;
            }
            unplace(entry_slot_[size_t(best)]);
            idx[n] = int32_t(w_res(l, best % n_ex));
            val[n] = 0;
            ++n;
            keys.push_back(best);
        }
        return n;
    }, err);
    test_idx_.assign(size_t(fn::n_layer) * n_ex, int8_t(-1));
    test_keys_ = keys;
    for (size_t i = 0; i < keys.size(); ++i) {
        test_idx_[size_t(keys[i])] = int8_t(i);
        test_ev_[i].store(0u);
    }
    return ok;
}

void residency::adm_test_forced(int32_t & n_admitted, int32_t & n_replaced, std::string & failures) const {
    n_admitted = n_replaced = 0;
    failures.clear();
    const uint32_t need = ev_missed | ev_assigned | ev_started | ev_published;
    for (size_t i = 0; i < test_keys_.size(); ++i) {
        const int32_t k = test_keys_[i];
        const uint32_t ev = test_ev_[i].load();
        const bool admitted = entry_slot_[size_t(k)] >= 0 && adm_flag(k) == 1;
        const bool replaced = !admitted && (ev & ev_left) != 0;
        if ((ev & need) == need && (admitted || replaced)) {
            n_admitted += admitted ? 1 : 0;
            n_replaced += replaced ? 1 : 0;
            continue;
        }
        char b[192];
        std::snprintf(b, sizeof(b), "%skey %d (layer %d): missed %d assigned %d started %d published %d collided %d "
                      "retired %d left %d, resident %d flag %d",
                      failures.empty() ? "" : "; ", k, k / n_ex, (ev & ev_missed) != 0, (ev & ev_assigned) != 0,
                      (ev & ev_started) != 0, (ev & ev_published) != 0, (ev & ev_collided) != 0,
                      (ev & ev_retired) != 0, (ev & ev_left) != 0, entry_slot_[size_t(k)] >= 0, int(adm_flag(k)));
        failures += b;
    }
}

bool residency::poison_slot(const cache::devices & dv, int32_t s, std::string & err) const {
    for (int32_t d = 0; d < nd_; ++d) {
        if (!dv.set(d, err) ||
            !cuda_ok(cudaMemset(pool_[d] + slot_off_[size_t(s)], 0xff, size_t(slot_bytes(s))), "poison", err)) {
            return false;
        }
    }
    return true;
}

std::string residency::plan_pool(const cache::devices & dv, int64_t reserve, int32_t routed_rows) const {
    std::string s;
    char b[640];
    for (int32_t d = 0; d < nd_; ++d) {
        std::snprintf(b, sizeof(b), "expert cache: device %d (%s), free before the pool %s, reserve %s\n", dv.dev[d],
                      nd_ == 1 ? "both cards" : d == 0 ? "card 0's shards" : "card 1's shards",
                      cache::gib(free_before_[d]).c_str(), cache::gib(reserve).c_str());
        s += b;
        if (device_cap_ >= 0) {
            std::snprintf(b, sizeof(b), "  device ceiling %s: in use before the pool %s (every allocation and the "
                          "context), so the pool takes at most the ceiling less that and the reserve\n",
                          cache::gib(device_cap_).c_str(), cache::gib(used_before_[d]).c_str());
            s += b;
        }
    }
    int64_t used = 0;
    for (int32_t s2 = 0; s2 < zone_first_; ++s2) {
        used += cls_[size_t(boot_.slot_cls[size_t(s2)])].entry;
    }
    std::snprintf(b, sizeof(b),
                  "  pool %s (%lld bytes) a device: class slots %s in %d slots (rounding %s), zone %s, lent region "
                  "%s\n",
                  cache::gib(pool_bytes_).c_str(), (long long)pool_bytes_, cache::gib(used).c_str(), zone_first_,
                  cache::gib(main_bytes_ - used).c_str(), cache::gib(2 * zone_buf_).c_str(),
                  cache::gib(lent_bytes_).c_str());
    s += b;
    for (size_t k = 0; k < cls_.size(); ++k) {
        const cls & c = cls_[k];
        int32_t zn = 0;
        for (size_t s2 = size_t(zone_first_); s2 < boot_.slot_cls.size(); ++s2) {
            zn += boot_.slot_cls[s2] == int32_t(k);
        }
        std::snprintf(b, sizeof(b),
                      "  class %zu: %s/%s, %d layers, entry %lld bytes a device, %d slots and %d zone slots of %d "
                      "experts (%.1f%%)\n",
                      k, fn::experts::fmt_name(fn::experts::fmt(c.gu_fmt)),
                      fn::experts::fmt_name(fn::experts::fmt(c.dn_fmt)), c.layers, (long long)c.entry, c.count, zn,
                      n_ex * c.layers, 100.0 * (c.count + zn) / double(n_ex * c.layers));
        s += b;
    }
    std::snprintf(b, sizeof(b),
                  "  prompt zone: %zu slots in 2 x %s at pool offset %lld (lent to a prompt's staging, refilled "
                  "after it); a prompt of one half of at most %d rows stages its routed experts\n",
                  slot_off_.size() - size_t(zone_first_), cache::gib(zone_buf_).c_str(), (long long)zone_off_,
                  routed_rows);
    s += b;
    return s;
}

std::string residency::plan_ranking(const std::string & profile, const std::string & profile_save,
                                    const policy::adapt_options & a) const {
    char b[640];
    std::snprintf(b, sizeof(b),
                  "  boot ranking: %s (%zu keys ranked); learned profile to %s; swaps every %d rounds, at most %d, "
                  "gain %.2f, usage %.2f, decay %.2f, overlap %s\n",
                  profile.empty() ? "expert order" : profile.c_str(), profile_.size(),
                  profile_save.empty() ? "-" : profile_save.c_str(), a.every, a.max_swaps, double(a.min_gain),
                  double(a.min_use), double(a.decay), swap_overlap_ ? "on" : "off");
    return b;
}

} // namespace fn::runtime
