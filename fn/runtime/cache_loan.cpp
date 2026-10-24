// ME3 (the cleanup): the expert cache's prompt loan and refill (cache_loan.h), ME2b's and ME3's code moved here from
// expert_cache.cpp; the refill's progress is loan::refill_state.

#include "fn/runtime/cache_loan.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace fn::runtime {

using cache::cuda_ok;
using cache::fail;
using cache::n_ex;

bool lent_sync(const int32_t * dev, int32_t nd, bool borrow, uint8_t * region, int64_t bytes, bool poison,
               std::string & err) {
    for (int32_t d = 0; d < nd; ++d) {
        if (!cuda_ok(cudaSetDevice(dev[d]), "cudaSetDevice", err) ||
            !cuda_ok(cudaDeviceSynchronize(), borrow ? "borrow" : "return", err)) {
            return false;
        }
    }
    if (borrow && poison &&
        (!cuda_ok(cudaSetDevice(dev[0]), "cudaSetDevice", err) ||
         !cuda_ok(cudaMemset(region, 0xff, size_t(bytes)), "poison", err) ||
         !cuda_ok(cudaDeviceSynchronize(), "poison", err))) {
        return false;
    }
    return true;
}

// ME2b item 6: on one card the prompt buffers decode never reads (PF1's staging, PB1's arena: both logical cards'
// parts) leave their own allocations for a region at the pool's tail. Freed here, so the pool measured after takes
// their bytes; the classes' buffers become fixed views of the region once the pool exists (the pool the one owner)
void prompt_loan::collect_region(model::model & m, const cache::devices & dv, bool lend_region) {
    lent_.clear();
    lent_bytes_ = 0;
    for (const char * name : {"prefill staging (PF1)", "prefill arena (PB1)"}) {
        for (auto & rc : m.state) {
            for (int32_t c = 0; c < 2 && dv.nd == 1 && lend_region && rc->plan.name == name; ++c) {
                model::device_buffer & b = rc->buf[c];
                if (b.data() == nullptr || b.host() || b.device() != dv.dev[0]) {
                    continue;
                }
                lent_.push_back({&b, rc->plan.name, c, lent_bytes_, int64_t(b.size())});
                lent_bytes_ += (int64_t(b.size()) + 255) / 256 * 256;
                b.release();
            }
        }
    }
}

bool prompt_loan::init(const cache::devices & dv, residency & r, svc::loop & lp, const options & o,
                       std::string & err) {
    dv_ = &dv;
    res_ = &r;
    loop_ = &lp;
    o_ = o;
    const auto vp = [](auto ** p) { return reinterpret_cast<void **>(p); };
    const size_t ptab_bytes = size_t(dv.parts) * n_ex * sizeof(void *);
    if (!own_.halloc(size_t(cache::max_prompt_rows) * 10 * 4, vp(&pids_h_), false, err)) {
        return false;
    }
    for (int32_t d = 0; d < dv.nd; ++d) {
        if (!own_.dalloc(dv, d, ptab_bytes, vp(&ptab_[d][0]), true, err) ||
            !own_.dalloc(dv, d, ptab_bytes, vp(&ptab_[d][1]), true, err) ||
            !own_.halloc(ptab_bytes, vp(&ptab_h_[d][0]), false, err) ||
            !own_.halloc(ptab_bytes, vp(&ptab_h_[d][1]), false, err)) {
            return false;
        }
        for (int32_t i = 0; i < dv.parts * n_ex; ++i) {
            ptab_h_[d][0][i] = r.zero(d);
        }
        if (!cuda_ok(cudaMemcpy(ptab_[d][0], ptab_h_[d][0], ptab_bytes, cudaMemcpyHostToDevice), "table", err) ||
            !cuda_ok(cudaMemcpy(ptab_[d][1], ptab_h_[d][0], ptab_bytes, cudaMemcpyHostToDevice), "table", err)) {
            return false;
        }
        // host waits: ev_staged_ (staging), ev_route_ (a prompt's IDs). The refill: ev_rf_ (a batch landed),
        // ev_lease_ (the prompt's last readers), ev_rft_ (device 0, timing: a batch's copies)
        if (!own_.event(dv, d, &ev_rf_[d], false, false, err) || !own_.event(dv, d, &ev_lease_[d], false, false, err) ||
            !own_.event(dv, d, &ev_cp_[d], false, false, err) ||
            (d == 0 && (!own_.event(dv, d, &ev_rft_[0], true, false, err) ||
                        !own_.event(dv, d, &ev_rft_[1], true, false, err))) ||
            !own_.event(dv, d, &ev_free_[d][0], false, false, err) ||
            !own_.event(dv, d, &ev_free_[d][1], false, false, err) ||
            !own_.event(dv, d, &ev_staged_[d][0], false, true, err) ||
            !own_.event(dv, d, &ev_staged_[d][1], false, true, err) ||
            (d == 0 && !own_.event(dv, d, &ev_route_, false, true, err))) {
            return false;
        }
    }
    // ME4: the ring's pinned host tables and its entries' events (an entry at least ring_bytes / n_ring_ev, so at most
    // n_ring_ev live entries and their slots distinct)
    tab_bytes_ = (int64_t(ptab_bytes) + 255) / 256 * 256;
    ring_min_ = (o.ring_bytes / n_ring_ev + 255) / 256 * 256;
    for (int32_t d = 0; d < dv.nd && o.ring_bytes > 0; ++d) {
        if (!own_.halloc(size_t(n_ring_ev) * size_t(tab_bytes_), vp(&rtab_h_[d]), false, err)) {
            return false;
        }
        for (int32_t i = 0; i < n_ring_ev; ++i) {
            if (!own_.event(dv, d, &ev_rcp_[d][i], false, false, err) ||
                !own_.event(dv, d, &ev_rrel_[d][i], false, false, err)) {
                return false;
            }
        }
    }
    return true;
}

void prompt_loan::attach() {
    for (const lent_part & x : lent_) {
        x.buf->view(dv_->dev[0], res_->pool(0) + res_->lent_off() + x.off, size_t(x.bytes));
    }
    zone_keys_.assign(size_t(res_->zone_slots()), -1);
    taken_slots_.assign(zone_keys_.size(), 0);
}

// ME3: the loan's one calculation. The spans (need: the bytes a prompt writes in the lent region, device 0) and each
// staging buffer's need (the largest of its layers' experts not resident outside the loan; worst: all of them) give the
// zone slots borrowed, each slot counted once however many intervals touch it; the staging need grows with the
// residents the loan retires, so the two are iterated to a fixed point (monotone, bounded by the buffers). Refused
// beyond the loan cap (lend_pct % of the pool's slots) or below min_keep kept slots.
bool prompt_loan::plan(const std::vector<model::byte_span> & need, bool worst, std::vector<uint8_t> & take,
                       int64_t stage_need[2], std::string & err) const {
    int64_t so[2] = {0, 0};
    return plan_at(need, worst, false, take, stage_need, so, err);
}

// ME4: as plan, and by_usage: the parity buffers in the contiguous interval of least current usage (prompt_plan.h
// place_parity; no such interval: ME3's place at the zone's start); stage_off their starts
bool prompt_loan::plan_at(const std::vector<model::byte_span> & need, bool worst, bool by_usage,
                          std::vector<uint8_t> & take, int64_t stage_need[2], int64_t stage_off[2],
                          std::string & err) const {
    const residency & r = *res_;
    loan::plan_input in;
    std::vector<double> score;
    const uintptr_t base = reinterpret_cast<uintptr_t>(r.pool(0));
    for (const model::byte_span & x : need) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(x.p);
        if (x.bytes <= 0 || lent_bytes_ == 0 || p < base + uintptr_t(r.lent_off()) ||
            p + uintptr_t(x.bytes) > base + uintptr_t(r.pool_bytes())) {
            return fail(err, "a prompt span outside the lent region");
        }
        in.spans.emplace_back(int64_t(p - base), int64_t(p - base) + x.bytes);
    }
    for (int32_t s = r.zone_first(); s < r.n_slots(); ++s) {
        const int32_t k = r.key_of(s);
        in.zone.push_back({r.slot_off(s), r.slot_bytes(s), k >= 0 ? k / n_ex : -1});
        score.push_back(k >= 0 ? double(r.usage(k)) : 0.0);
    }
    in.stage_off[0] = r.zone_off();
    in.stage_off[1] = r.zone_off() + r.zone_buf();
    in.resident.assign(fn::n_layer, 0);
    for (int32_t s = 0; s < r.n_slots(); ++s) {
        if (r.key_of(s) >= 0) {
            ++in.resident[size_t(r.key_of(s) / n_ex)];
        }
    }
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        in.dentry.push_back(r.layer(l).entry_bytes() / dv_->nd);
    }
    in.n_ex = n_ex;
    in.worst = worst;
    in.n_slots = int64_t(r.n_slots());
    in.lend_pct = o_.lend_pct;
    in.min_keep = o_.min_keep;
    loan::plan_output out;
    stage_off[0] = in.stage_off[0];
    stage_off[1] = in.stage_off[1];
    bool placed = false;
    if (by_usage && !worst) {
        std::string e2;
        placed = pplan::place_parity(in, score, r.zone_off(), r.pool_bytes(), stage_off, out, e2);
        if (!placed && !e2.empty()) {
            return fail(err, e2);
        }
    }
    if (!placed && !loan::plan(in, out, err)) {
        return false;
    }
    take = std::move(out.take);
    stage_need[0] = out.stage_need[0];
    stage_need[1] = out.stage_need[1];
    return true;
}

bool prompt_loan::check(const std::vector<model::byte_span> & need, std::string & err) const {
    std::vector<uint8_t> take;
    int64_t sn[2] = {0, 0};
    if (!plan(need, true, take, sn, err)) {
        return false;
    }
    int64_t n = 0, bytes = 0, span = 0;
    for (size_t j = 0; j < take.size(); ++j) {
        n += take[j];
        bytes += take[j] != 0 ? res_->slot_bytes(res_->zone_first() + int32_t(j)) : 0;
    }
    for (const model::byte_span & x : need) {
        span += x.bytes;
    }
    std::fprintf(stderr,
                 "expert cache: the largest loan (startup, every expert staged): %lld of %zu zone slots (%s), of %d "
                 "slots; spans %s, staging %s + %s; cap %d %%, min-keep %d\n",
                 (long long)n, take.size(), cache::gib(bytes).c_str(), res_->n_slots(), cache::gib(span).c_str(),
                 cache::gib(sn[0]).c_str(), cache::gib(sn[1]).c_str(), o_.lend_pct, o_.min_keep);
    return true;
}

// ME4: a ring loan's slots: the ring's place off the spans by the residents' current usage (ties to the highest
// offset), borrowed with the spans through loan::plan with no layer buffers (each slot once). false with err empty: no
// place for the ring (the caller plans the layer buffers instead); with err: refused (the cap, min_keep, a span)
bool prompt_loan::plan_ring(const std::vector<model::byte_span> & need, std::vector<uint8_t> & take, int64_t & ring_lo,
                            std::string & err) const {
    const residency & r = *res_;
    loan::plan_input in;
    const uintptr_t base = reinterpret_cast<uintptr_t>(r.pool(0));
    for (const model::byte_span & x : need) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(x.p);
        if (x.bytes <= 0 || lent_bytes_ == 0 || p < base + uintptr_t(r.lent_off()) ||
            p + uintptr_t(x.bytes) > base + uintptr_t(r.pool_bytes())) {
            return fail(err, "a prompt span outside the lent region");
        }
        in.spans.emplace_back(int64_t(p - base), int64_t(p - base) + x.bytes);
    }
    std::vector<double> score;
    for (int32_t s = r.zone_first(); s < r.n_slots(); ++s) {
        const int32_t k = r.key_of(s);
        in.zone.push_back({r.slot_off(s), r.slot_bytes(s), k >= 0 ? k / n_ex : -1});
        score.push_back(k >= 0 ? double(r.usage(k)) : 0.0);
    }
    if (!pplan::place_ring_scored(in.zone, score, r.zone_off(), r.pool_bytes(), in.spans, ring_cur_, ring_lo)) {
        return false;
    }
    in.spans.emplace_back(ring_lo, ring_lo + ring_cur_);
    in.resident.assign(fn::n_layer, 0);
    in.dentry.assign(fn::n_layer, 0);
    in.n_ex = n_ex;
    in.n_slots = int64_t(r.n_slots());
    in.lend_pct = o_.lend_pct;
    in.min_keep = o_.min_keep;
    loan::plan_output out;
    if (!loan::plan(in, out, err)) {
        return false;
    }
    take = std::move(out.take);
    return true;
}

// the prompt's loan out of the pool (the caller committed the pending pass and stopped an outstanding refill): every
// earlier reader of the zone's slots marked on the compute streams (the staging copies wait for it), the borrowed
// slots' keys retired (kept for the return) and the residency rows uploaded in stream order. need null (or ME3's base
// arm): the whole zone, ME2b's loan
bool prompt_loan::take(void * const stream[2], const std::vector<model::byte_span> * need, bool legacy,
                       std::string & err) {
    if (refill_active()) {
        return fail(err, "a loan while a refill is outstanding");
    }
    // set_ring applies to this loan only; MX3: its ring sized by the run's largest chunk
    const int32_t ring_rows = ring_rows_next_;
    ring_rows_next_ = 0;
    const bool ring_asked = ring_rows > 0;
    ring_cur_ = ring_asked ? pplan::ring_for(ring_rows, o_.ring_bytes, o_.group_bytes, tab_bytes_) : o_.ring_bytes;
    ring_reset();
    ring_on_ = false;
    stage_off_[0] = res_->zone_off();
    stage_off_[1] = res_->zone_off() + res_->zone_buf();
    if (need == nullptr || legacy) {
        taken_slots_.assign(zone_keys_.size(), 1);
        stage_need_[0] = stage_need_[1] = res_->zone_buf();
    } else {
        // ME4: the ring at its place (prompt_plan.h place_ring_scored), borrowed with the spans and no layer buffers;
        // with no place for it, ME3's layer buffers
        std::string e2;
        if (ring_asked && !me3_place_ && o_.ring_bytes > 0 && plan_ring(*need, taken_slots_, ring_lo_, e2)) {
            ring_on_ = true;
            stage_need_[0] = stage_need_[1] = 0;
            ++n_.ring_loans;
            n_.ring_loan_bytes += ring_cur_;
        } else if (!e2.empty()) {
            return fail(err, e2);
        } else if (!plan_at(*need, false, !me3_place_, taken_slots_, stage_need_, stage_off_, err)) {
            return false;
        } else if (!me3_place_) {
            ++n_.parity_tried;
            n_.parity_moved += stage_off_[0] != res_->zone_off() ? 1 : 0;
        }
    }
    ++n_.loans;
    for (size_t j = 0; j < taken_slots_.size(); ++j) {
        const int32_t s = res_->zone_first() + int32_t(j);
        if (taken_slots_[j] == 0) {
            continue;
        }
        ++n_.borrowed_slots;
        n_.borrowed_bytes += res_->slot_bytes(s);
        const int32_t k = res_->retire(s);
        if (k >= 0) {
            zone_keys_[j] = k;
            ++n_.borrowed_residents;
            n_.borrowed_resident_bytes += res_->layer(k / n_ex).entry_bytes();
            ++n_.zone_evictions;
        }
    }
    for (int32_t d = 0; d < dv_->nd; ++d) {
        const cudaStream_t s = static_cast<cudaStream_t>(stream[d]);
        if (!dv_->set(d, err) || !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_free_[d][0]), s), "zone", err) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_free_[d][1]), s), "zone", err)) {
            return false;
        }
    }
    if (!res_->upload_rows_ordered(*dv_, stream, err)) {
        return false;
    }
    ptab_layer_[0] = ptab_layer_[1] = -1;
    taken_ = true;
    return true;
}

// the zone back after a prompt (the caller committed the pending pass): its restore set in layer order. Null streams
// (drain), ME3's refill-overlap 0: copied and published synchronously. ME3's base arm: ME2b's pass, enqueued here (the
// compute streams' layer waits). Otherwise ME3's bounded refill: the swap streams after the prompt's last readers (the
// lease, the copy streams' staging), the batches submitted by the service when it has no window, each published once
// it has landed
bool prompt_loan::give_back(void * const stream[2], bool legacy, std::string & err) {
    if (refill_active()) {
        return fail(err, "a return while a refill is outstanding");
    }
    std::vector<residency::move> moves;
    for (size_t j = 0; j < zone_keys_.size(); ++j) {
        const int32_t k = zone_keys_[j], s = res_->zone_first() + int32_t(j);
        if (k >= 0 && res_->slot_of(k) < 0 && res_->key_of(s) < 0) {
            moves.emplace_back(s, k);
        }
        zone_keys_[j] = -1;
    }
    ptab_layer_[0] = ptab_layer_[1] = -1;
    taken_ = false;
    ring_reset();
    ring_on_ = false;
    if (stream == nullptr || legacy || !o_.refill_overlap || moves.empty()) {
        int64_t bytes = 0;
        const bool ok = res_->publish(*dv_, moves, true, stream != nullptr && legacy ? stream : nullptr, bytes, err);
        n_.refill_bytes += bytes;
        ++n_.refills;
        return ok;
    }
    std::stable_sort(moves.begin(), moves.end(),
                     [](const auto & a, const auto & b) { return a.second / n_ex < b.second / n_ex; });
    for (int32_t d = 0; d < dv_->nd; ++d) {
        const cudaStream_t ss = static_cast<cudaStream_t>(res_->swap_stream(d));
        if (!dv_->set(d, err) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_lease_[d]), static_cast<cudaStream_t>(stream[d])),
                     "refill lease", err) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_cp_[d]), static_cast<cudaStream_t>(dv_->copy[d])),
                     "refill event", err) ||
            !cuda_ok(cudaStreamWaitEvent(ss, static_cast<cudaEvent_t>(ev_lease_[d]), 0), "refill wait", err) ||
            !cuda_ok(cudaStreamWaitEvent(ss, static_cast<cudaEvent_t>(ev_cp_[d]), 0), "refill wait", err)) {
            return false;
        }
    }
    ++n_.refills;
    {
        std::lock_guard<std::mutex> lk(mu_);
        moves_ = std::move(moves);
        st_.start(moves_.size());
        timed_ = false;
        batches_ = 0;
        active_ = true;
    }
    loop_->notify(); // after this mutex is released (the lock order)
    return true;
}

bool prompt_loan::hold(bool on, const std::vector<model::byte_span> * need, bool legacy, std::string & err) {
    const cache::devices & dv = *dv_;
    const auto poison_taken = [&]() {
        for (size_t j = 0; j < taken_slots_.size(); ++j) {
            if (taken_slots_[j] != 0 && !res_->poison_slot(dv, res_->zone_first() + int32_t(j), err)) {
                return false;
            }
        }
        return true;
    };
    if (lent_bytes_ == 0 && on && poison_) {
        // ME3 (TEST): without a lent region (cached TP) the poison fills the borrowed zone slots, each device drained
        // first; the hold is unchanged
        for (int32_t d = 0; d < dv.nd; ++d) {
            if (!dv.set(d, err) || !cuda_ok(cudaDeviceSynchronize(), "poison", err)) {
                return false;
            }
        }
        if (!poison_taken()) {
            return false;
        }
        for (int32_t d = 0; d < dv.nd; ++d) {
            if (!dv.set(d, err) || !cuda_ok(cudaDeviceSynchronize(), "poison", err)) {
                return false;
            }
        }
        return true;
    }
    if (lent_bytes_ == 0) {
        return true;
    }
    if (!lent_sync(dv.dev, dv.nd, on, res_->pool(0) + res_->lent_off(), lent_bytes_, false, err)) {
        return false;
    }
    if (on && poison_) {
        if (!poison_taken()) {
            return false;
        }
        for (size_t i = 0; need != nullptr && !legacy && i < need->size(); ++i) {
            if (!dv.set(0, err) ||
                !cuda_ok(cudaMemset(const_cast<void *>((*need)[i].p), 0xff, size_t((*need)[i].bytes)), "poison",
                         err)) {
                return false;
            }
        }
        if ((need == nullptr || legacy) &&
            !cuda_ok(cudaMemset(res_->pool(0) + res_->lent_off(), 0xff, size_t(lent_bytes_)), "poison", err)) {
            return false;
        }
        if (!cuda_ok(cudaDeviceSynchronize(), "poison", err)) {
            return false;
        }
    }
    held_ = on;
    n_.lent_holds += on ? 1 : 0;
    return true;
}

int64_t prompt_loan::restore_keys(int64_t & bytes) const {
    int64_t keys = 0;
    bytes = 0;
    std::lock_guard<std::mutex> lk(mu_);
    if (active_) {
        for (size_t i = st_.pub(); i < moves_.size(); ++i) {
            ++keys;
            bytes += res_->layer(moves_[i].second / n_ex).entry_bytes();
        }
    }
    for (size_t j = 0; j < zone_keys_.size() && taken_; ++j) {
        if (zone_keys_[j] >= 0) {
            ++keys;
            bytes += res_->layer(zone_keys_[j] / n_ex).entry_bytes();
        }
    }
    return keys;
}

// --- the refill ---

// the swap stream's host callback after a refill batch (both devices' copies complete): no CUDA call here. The
// record's generation names the refill whose batch it was
void prompt_loan::landed(void * rec) {
    const landing * const r = static_cast<const landing *>(rec);
    prompt_loan * const x = r->owner;
    const uint32_t gen = r->gen;
    {
        std::lock_guard<std::mutex> lk(x->mu_);
        x->st_.landed(gen);
    }
    x->cv_.notify_all();
    x->loop_->notify();
}

bool prompt_loan::ready() {
    std::lock_guard<std::mutex> lk(mu_);
    return st_.ready() && (limit_ < 0 || batches_ < limit_);
}

// the service's next batch (under the loop's lock, ready): from the first unsubmitted move, at most
// refill_batch_layers layers and (MX3) refill_batch_bytes on a device
void prompt_loan::take(size_t & a, size_t & b) {
    std::lock_guard<std::mutex> lk(mu_);
    a = st_.next();
    b = loan::batch_end_bytes(
        moves_.size(), a, o_.refill_batch_layers, o_.refill_batch_bytes,
        [&](size_t i) { return moves_[i].second / n_ex; },
        [&](size_t i) { return res_->layer(moves_[i].second / n_ex).entry_bytes() / dv_->nd; });
    st_.take(a, b);
    read_prev_ = timed_;
    timed_ = true;
    ++batches_;
}

// the service's batch [a, b): the previous batch's copy time read, then each move's copies on every device's swap
// stream (one in flight: the previous landed), ending after the current move when the service must yield to a window
// (at least one move), each device's landing event, then on device 0 (after device 1's) the host callback
bool prompt_loan::submit(size_t a, size_t b, const std::function<bool()> & yield, std::string & err) {
    const cache::clk::time_point t0 = cache::clk::now();
    const cache::devices & dv = *dv_;
    const auto lose = [&]() {
        std::lock_guard<std::mutex> lk(mu_);
        st_.lost();
        err_ = err;
        return false;
    };
    float prev_ms = 0;
    if (!dv.set(0, err)) {
        return lose();
    }
    if (read_prev_ && !cuda_ok(cudaEventElapsedTime(&prev_ms, static_cast<cudaEvent_t>(ev_rft_[0]),
                                                    static_cast<cudaEvent_t>(ev_rft_[1])),
                               "refill time", err)) {
        prev_ms = 0;
    }
    if (!cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_rft_[0]), static_cast<cudaStream_t>(res_->swap_stream(0))),
                 "refill time", err)) {
        return lose();
    }
    int64_t bytes = 0;
    size_t end = a;
    while (end < b) {
        const int32_t s = moves_[end].first, k = moves_[end].second, l = k / n_ex;
        const int64_t dentry = res_->layer(l).entry_bytes() / dv.nd;
        for (int32_t d = 0; d < dv.nd; ++d) {
            if (!dv.set(d, err) ||
                !cuda_ok(cudaMemcpyAsync(res_->pool(d) + res_->slot_off(s), res_->entry_src(l, k % n_ex, d),
                                         size_t(dentry), cudaMemcpyHostToDevice,
                                         static_cast<cudaStream_t>(res_->swap_stream(d))),
                         "refill copy", err)) {
                return lose();
            }
            bytes += dentry;
        }
        ++end;
        if (end < b && yield()) {
            break;
        }
    }
    uint32_t gen = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (end < b) {
            st_.trim(end);
        }
        gen = st_.gen();
    }
    for (int32_t d = 0; d < dv.nd; ++d) {
        const cudaStream_t ss = static_cast<cudaStream_t>(res_->swap_stream(d));
        if (!dv.set(d, err) ||
            (d == 0 && !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_rft_[1]), ss), "refill time", err)) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_rf_[d]), ss), "refill landed", err)) {
            return lose();
        }
    }
    landing * const rec = &rec_[rec_at_];
    rec_at_ ^= 1;
    rec->owner = this;
    rec->gen = gen;
    const cudaStream_t s0 = static_cast<cudaStream_t>(res_->swap_stream(0));
    if (!dv.set(0, err) ||
        (dv.nd == 2 && !cuda_ok(cudaStreamWaitEvent(s0, static_cast<cudaEvent_t>(ev_rf_[1]), 0), "refill wait", err)) ||
        !cuda_ok(cudaLaunchHostFunc(s0, &prompt_loan::landed, rec), "refill callback", err)) {
        return lose();
    }
    std::lock_guard<std::mutex> lk(mu_);
    svc_n_.refill_copy_ms += double(prev_ms);
    svc_n_.refill_bytes += bytes;
    ++svc_n_.refill_batches;
    svc_n_.refill_submit_s += cache::since(t0);
    return true;
}

void prompt_loan::failed(const std::string & err) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        st_.fail();
        if (err_.empty()) {
            err_ = err;
        }
    }
    cv_.notify_all();
}

void prompt_loan::test_refill_limit(int32_t limit) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        limit_ = limit;
    }
    loop_->notify();
}

void prompt_loan::end_refill() {
    st_.end();
    moves_.clear();
    timed_ = false;
    active_ = false;
}

// ME3: the landed moves into the tables (the decode thread, the service without a window), through the residency's
// publish_landed; a slot is published only once its copy has landed. Complete: the refill ended and its last batch's
// copy time read
bool prompt_loan::refill_publish(void * const stream[2], std::string & err) {
    if (!refill_active()) {
        return true;
    }
    size_t pub = 0, end = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        pub = st_.pub();
        end = st_.landed_end();
    }
    if (end > pub) {
        if (!res_->publish_landed(*dv_, moves_.data() + pub, end - pub, stream, n_.refill_skipped, err)) {
            return false;
        }
        std::lock_guard<std::mutex> lk(mu_);
        st_.published(end);
    }
    bool timed = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!st_.complete()) {
            return true;
        }
        timed = timed_;
    }
    // complete: nothing in flight and nothing to submit, so the service takes no batch while the time is read (no CUDA
    // call under this mutex, which the callback takes)
    float ms = 0;
    if (timed && dv_->set(0, err) &&
        cuda_ok(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(ev_rft_[0]), static_cast<cudaEvent_t>(ev_rft_[1])),
                "refill time", err)) {
        n_.refill_copy_ms += double(ms);
    }
    std::lock_guard<std::mutex> lk(mu_);
    end_refill();
    return true;
}

// ME3: an outstanding refill ended. finish (drain): every batch submitted and landed, then published synchronously.
// Otherwise (a new loan): no further batch, the one in flight landed and published, the moves never submitted carried
// back to the restore set (their slots empty, their keys borrowed again: the next return refills them)
bool prompt_loan::refill_stop(bool finish, std::string & err) {
    if (!refill_active()) {
        return true;
    }
    {
        std::unique_lock<std::mutex> lk(mu_);
        if (!finish) {
            st_.stop();
        }
        cv_.wait(lk, [&]() { return st_.settled(); });
        if (st_.failed()) {
            return fail(err, "a window failed: " + err_);
        }
    }
    if (!refill_publish(nullptr, err)) {
        return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (!active_) {
        return true;
    }
    for (size_t i = st_.pub(); i < moves_.size(); ++i) {
        zone_keys_[size_t(moves_[i].first - res_->zone_first())] = moves_[i].second;
        ++n_.refill_carried;
    }
    end_refill();
    return true;
}

// --- prefill ---

// layer l's experts into staging buffer b on every device (used: the experts to stage, null for all): wait for the
// buffer's table copy and last reader, copy each run of experts not resident, then the buffer's table
bool prompt_loan::stage(int32_t l, int32_t b, const uint8_t * used, std::string & err) {
    const cache::devices & dv = *dv_;
    const residency & r = *res_;
    const loader::expert_layer & L = r.layer(l);
    const int64_t entry = L.entry_bytes(), dentry = entry / dv.nd;
    for (int32_t d = 0; d < dv.nd; ++d) {
        const cudaStream_t cs = static_cast<cudaStream_t>(dv.copy[d]);
        const cache::clk::time_point t0 = cache::clk::now();
        if (!dv.set(d, err) ||
            !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(ev_staged_[d][b])), "table wait", err)) {
            return false;
        }
        n_.prompt_wait_s += cache::since(t0);
        if (!cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(ev_free_[d][b]), 0), "buffer wait", err)) {
            return false;
        }
        uint8_t * const buf = r.pool(d) + stage_off_[b];
        const uint8_t ** tab = ptab_h_[d][b];
        int64_t off = 0, run_off = 0;
        int32_t run = -1;
        const auto flush = [&](int32_t end) {
            if (run < 0) {
                return true;
            }
            const int32_t n = end - run;
            if (run_off + n * dentry > stage_need_[b]) { // ME3: never past the loan's staging
                run = -1;
                return fail(err, "layer " + std::to_string(l) + "'s staging exceeds the loan's (" +
                                     std::to_string(stage_need_[b]) + " bytes)");
            }
            n_.prompt_staged_bytes += n * dentry;
            const bool ok =
                dv.nd == 1 ? cuda_ok(cudaMemcpyAsync(buf + run_off, r.entry_src(l, run, d), size_t(n * dentry),
                                                     cudaMemcpyHostToDevice, cs),
                                     "staging", err)
                           : cuda_ok(cudaMemcpy2DAsync(buf + run_off, size_t(dentry), r.entry_src(l, run, d),
                                                       size_t(entry), size_t(dentry), size_t(n),
                                                       cudaMemcpyHostToDevice, cs),
                                     "staging", err);
            run = -1;
            return ok;
        };
        for (int32_t e = 0; e < n_ex; ++e) {
            const int32_t sl = r.slot_of(l * n_ex + e);
            const uint8_t * a = nullptr;
            if (used != nullptr && used[e] == 0) {
                if (!flush(e)) {
                    return false;
                }
            } else if (sl >= 0) {
                if (!flush(e)) {
                    return false;
                }
                a = r.slot_addr(d, sl);
                n_.prompt_resident_bytes += dentry;
            } else {
                if (run < 0) {
                    run = e;
                    run_off = off;
                }
                a = buf + off;
                off += dentry;
            }
            for (int32_t card = 0; card < 2; ++card) {
                if (dv.dev_of(card) != d) {
                    continue;
                }
                for (int32_t dn = 0; dn < 2; ++dn) {
                    tab[dv.part_of(card, dn != 0) * n_ex + e] =
                        a != nullptr ? a + r.part_off(l, card, dn != 0) : r.zero(d);
                }
            }
        }
        if (!flush(n_ex) ||
            !cuda_ok(cudaMemcpyAsync(ptab_[d][b], tab, size_t(dv.parts) * n_ex * sizeof(void *),
                                     cudaMemcpyHostToDevice, cs),
                     "staging table", err) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_staged_[d][b]), cs), "staged", err)) {
            return false;
        }
    }
    ptab_layer_[b] = l;
    ptab_full_[b] = used == nullptr;
    ++n_.prompt_layers;
    n_.prompt_routed_layers += used != nullptr;
    return true;
}

bool prompt_loan::prompt_layer(int32_t layer, int32_t rows, const int32_t * ids, bool more, void * const stream[2],
                               std::string & err) {
    const cache::devices & dv = *dv_;
    if (ring_on_) {
        // ME4: a ring layer. Each chunk starts route-first; a layer planned ahead (streamed) needs no routing. The ring
        // never stages ahead across a chunk's end, so more changes nothing here
        (void)more;
        if (layer == 0) {
            ahead_ = false;
        }
        if (rl_[layer & 1].layer != layer) {
            if (ids == nullptr || rows < 1) {
                return fail(err, "ring layer " + std::to_string(layer) + " without its routing");
            }
            const cache::clk::time_point t0 = cache::clk::now();
            if (!dv.set(0, err) ||
                !cuda_ok(cudaMemcpyAsync(pids_h_, ids, size_t(rows) * 10 * 4, cudaMemcpyDeviceToHost,
                                         static_cast<cudaStream_t>(stream[0])),
                         "prompt ids", err) ||
                !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_route_), static_cast<cudaStream_t>(stream[0])),
                         "prompt ids", err) ||
                !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(ev_route_)), "prompt ids", err)) {
                return false;
            }
            n_.prompt_wait_s += cache::since(t0);
            int32_t counts[n_ex] = {};
            for (int32_t q = 0; q < rows * 10; ++q) {
                if (pids_h_[q] >= 0 && pids_h_[q] < n_ex) {
                    ++counts[pids_h_[q]];
                }
            }
            // MX7: the CPU share's experts (none without a share or on two devices)
            uint8_t cpu[n_ex] = {};
            if (cpu_pick_) {
                uint8_t resident[n_ex];
                for (int32_t e = 0; e < n_ex; ++e) {
                    resident[e] = res_->slot_of(layer * n_ex + e) >= 0 ? 1 : 0;
                }
                cpu_pick_(layer, rows, pids_h_, counts, resident, stream, cpu);
            }
            if (!ring_plan(layer, rows, counts, cpu, err)) {
                return false;
            }
        }
        return ring_issue(err);
    }
    const int32_t b = layer & 1;
    const bool routed = ids != nullptr && rows <= o_.routed_rows;
    const bool ready = ptab_layer_[b] == layer && ptab_full_[b];
    if (routed && !ready) {
        // the chunk's routing (one half), then only its experts
        const cache::clk::time_point t0 = cache::clk::now();
        if (!dv.set(0, err) ||
            !cuda_ok(cudaMemcpyAsync(pids_h_, ids, size_t(rows) * 10 * 4, cudaMemcpyDeviceToHost,
                                     static_cast<cudaStream_t>(stream[0])),
                     "prompt ids", err) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_route_), static_cast<cudaStream_t>(stream[0])),
                     "prompt ids", err) ||
            !cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(ev_route_)), "prompt ids", err)) {
            return false;
        }
        n_.prompt_wait_s += cache::since(t0);
        uint8_t used[n_ex] = {};
        for (int32_t p = 0; p < rows * 10; ++p) {
            if (pids_h_[p] >= 0 && pids_h_[p] < n_ex) {
                used[pids_h_[p]] = 1;
            }
        }
        if (!stage(layer, b, used, err)) {
            return false;
        }
    } else if (!routed && !ready && !stage(layer, b, nullptr, err)) {
        return false;
    }
    // the layer's readers after its copies; the other buffer's last reader (the previous layer) marked; a full
    // prompt stages the next layer into it now. ME4: the last chunk's last layer stages nothing (ME3 staged layer 0
    // again there, a whole layer's nonresident experts with no reader: the return discards it)
    for (int32_t d = 0; d < dv.nd; ++d) {
        const cudaStream_t s = static_cast<cudaStream_t>(stream[d]);
        if (!dv.set(d, err) ||
            !cuda_ok(cudaStreamWaitEvent(s, static_cast<cudaEvent_t>(ev_staged_[d][b]), 0), "staged wait", err) ||
            !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_free_[d][1 - b]), s), "buffer free", err)) {
            return false;
        }
    }
    // ME4 --me4-arms base (me3_place_): ME3's wrap, layer 0 staged again after the run's last layer
    const bool last = layer + 1 == fn::n_layer && !more && !me3_place_;
    return routed || last || stage((layer + 1) % fn::n_layer, 1 - b, nullptr, err);
}

// --- ME4: the ring ---

void prompt_loan::ring_reset() {
    rl_[0] = ring_layer();
    rl_[1] = ring_layer();
    ring_min_ = (ring_cur_ / n_ring_ev + 255) / 256 * 256; // MX13: this loan's ring, so at most n_ring_ev live entries
    ring_.reset(o_.ring_bytes > 0 ? new pplan::byte_ring(ring_cur_) : nullptr);
    ring_first_ = true;
    ahead_ = false;
}

// layer's groups: route-first (counts: its pairs an expert) the routed nonresident experts, then the cover decides
// whether the chunk's later layers stream ahead; streamed (counts null) every nonresident expert. The parity's previous
// layer must be released (its groups done)
bool prompt_loan::ring_plan(int32_t layer, int32_t rows, const int32_t * counts, const uint8_t * cpu,
                            std::string & err) {
    ring_layer & L = rl_[layer & 1];
    if (L.layer >= 0 && L.done < L.g.size()) {
        return fail(err, "ring layer " + std::to_string(layer) + ": layer " + std::to_string(L.layer) +
                             "'s groups are still in use");
    }
    const residency & r = *res_;
    const int64_t entry = r.layer(layer).entry_bytes(), dentry = entry / dv_->nd;
    L = ring_layer();
    L.layer = layer;
    L.rows = rows;
    L.copy.assign(n_ex, 0);
    int32_t nonres = 0, copied = 0;
    for (int32_t e = 0; e < n_ex; ++e) {
        const bool resident = r.slot_of(layer * n_ex + e) >= 0;
        const bool used = counts == nullptr || counts[e] > 0;
        const bool on_cpu = cpu != nullptr && cpu[e] != 0; // MX7: the CPU share's (routed, nonresident)
        nonres += resident ? 0 : 1;
        L.copy[size_t(e)] = used && !resident && !on_cpu ? 1 : 0;
        copied += L.copy[size_t(e)];
        n_.prompt_resident_bytes += used && resident ? entry : 0;
    }
    std::vector<pplan::group> gs;
    if (!pplan::plan_groups(L.copy.data(), n_ex, dentry, o_.group_bytes, gs, err)) {
        return false;
    }
    for (const pplan::group & x : gs) {
        ring_group rg;
        rg.g = x;
        if (counts != nullptr) {
            for (int32_t e = x.e_lo; e < x.e_hi; ++e) {
                rg.pairs += cpu != nullptr && cpu[e] != 0 ? 0 : counts[e];
            }
        } else {
            rg.pairs = int64_t(10) * rows; // a bound: the half's pairs
        }
        L.g.push_back(rg);
    }
    ++n_.prompt_layers;
    if (counts != nullptr) {
        ++n_.prompt_routed_layers;
        ++n_.ring_routed_layers;
        n_.ring_cover_pm += nonres > 0 ? int64_t(copied) * 1000 / nonres : 1000;
        ahead_ = ahead_ || pplan::stream_after(int64_t(copied) * dentry, int64_t(nonres) * dentry, o_.route_cover);
    } else {
        ++n_.ring_ahead_layers;
    }
    return true;
}

// the planned groups' entries into the ring in order (the lower layer first), as long as the ring has released space
// for them: each copy stream first waits for the loan's earlier readers (the first entry) and for the releases of the
// entries whose bytes it reuses; then the entry's table (each logical card's gate/up and down addresses: resident
// slots, the payload, or the zero expert), its payload (runs of consecutive copied experts), and its copied event
bool prompt_loan::ring_issue(std::string & err) {
    const cache::devices & dv = *dv_;
    const residency & r = *res_;
    for (;;) {
        ring_layer * L = nullptr;
        for (ring_layer & x : rl_) {
            if (x.layer >= 0 && x.next < x.g.size() && (L == nullptr || x.layer < L->layer)) {
                L = &x;
            }
        }
        if (L == nullptr) {
            return true;
        }
        ring_group & G = L->g[L->next];
        int64_t off = 0;
        int32_t id = -1;
        std::vector<int32_t> waits;
        if (ring_ == nullptr || !ring_->alloc(std::max(tab_bytes_ + G.g.bytes, ring_min_), off, waits, id)) {
            return true; // no released space yet: the next group_done issues it
        }
        const int32_t sl = id % n_ring_ev, l = L->layer;
        const int64_t entry = r.layer(l).entry_bytes(), dentry = entry / dv.nd;
        for (int32_t d = 0; d < dv.nd; ++d) {
            const cudaStream_t cs = static_cast<cudaStream_t>(dv.copy[d]);
            if (!dv.set(d, err) ||
                (ring_first_ &&
                 !cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(ev_free_[d][0]), 0), "ring wait", err))) {
                return false;
            }
            for (const int32_t w : waits) {
                if (!cuda_ok(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(ev_rrel_[d][w % n_ring_ev]), 0),
                             "ring wait", err)) {
                    return false;
                }
            }
            // the host table's slot: its previous entry's table copy landed
            if (!cuda_ok(cudaEventSynchronize(static_cast<cudaEvent_t>(ev_rcp_[d][sl])), "ring table", err)) {
                return false;
            }
            uint8_t * const dst = r.pool(d) + ring_lo_ + off;
            const uint8_t ** tab = reinterpret_cast<const uint8_t **>(rtab_h_[d] + int64_t(sl) * tab_bytes_);
            for (int32_t i = 0; i < dv.parts * n_ex; ++i) {
                tab[i] = r.zero(d);
            }
            int64_t at = tab_bytes_, run_at = 0;
            int32_t run = -1;
            const auto flush = [&](int32_t end) {
                if (run < 0) {
                    return true;
                }
                const int32_t n = end - run;
                const bool ok =
                    dv.nd == 1 ? cuda_ok(cudaMemcpyAsync(dst + run_at, r.entry_src(l, run, d), size_t(n * dentry),
                                                         cudaMemcpyHostToDevice, cs),
                                         "ring copy", err)
                               : cuda_ok(cudaMemcpy2DAsync(dst + run_at, size_t(dentry), r.entry_src(l, run, d),
                                                           size_t(entry), size_t(dentry), size_t(n),
                                                           cudaMemcpyHostToDevice, cs),
                                         "ring copy", err);
                run = -1;
                return ok;
            };
            for (int32_t e = G.g.e_lo; e < G.g.e_hi; ++e) {
                const int32_t sk = r.slot_of(l * n_ex + e);
                const uint8_t * a = nullptr;
                if (L->copy[size_t(e)] != 0) {
                    if (run < 0) {
                        run = e;
                        run_at = at;
                    }
                    a = dst + at;
                    at += dentry;
                } else {
                    if (!flush(e)) {
                        return false;
                    }
                    a = sk >= 0 ? r.slot_addr(d, sk) : nullptr;
                }
                for (int32_t card = 0; card < 2; ++card) {
                    for (int32_t dn = 0; dn < 2 && dv.dev_of(card) == d; ++dn) {
                        tab[dv.part_of(card, dn != 0) * n_ex + e] =
                            a != nullptr ? a + r.part_off(l, card, dn != 0) : r.zero(d);
                    }
                }
            }
            if (!flush(G.g.e_hi) ||
                !cuda_ok(cudaMemcpyAsync(dst, tab, size_t(dv.parts) * n_ex * sizeof(void *), cudaMemcpyHostToDevice,
                                         cs),
                         "ring table", err) ||
                !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_rcp_[d][sl]), cs), "ring copied", err)) {
                return false;
            }
            n_.prompt_staged_bytes += int64_t(G.g.copies) * dentry;
        }
        ring_first_ = false;
        G.id = id;
        G.off = off;
        ++L->next;
    }
}

int32_t prompt_loan::prompt_groups(int32_t layer) const {
    if (!ring_on_ || layer < 0 || layer >= fn::n_layer) {
        return 0;
    }
    const ring_layer & L = rl_[layer & 1];
    return L.layer == layer ? int32_t(L.g.size()) : 0;
}

// group g of layer (in order, after the earlier groups' done): issued if it was not yet, each device's compute stream
// waits for its copied event (the table and payload landed after every earlier reader of their bytes)
bool prompt_loan::prompt_group(int32_t layer, int32_t g, void * const stream[2], model::prompt_group_view & v,
                               std::string & err) {
    const cache::devices & dv = *dv_;
    if (prompt_groups(layer) <= g || g < 0 || size_t(g) != rl_[layer & 1].done) {
        return fail(err, "ring group " + std::to_string(g) + " of layer " + std::to_string(layer) + " out of order");
    }
    ring_layer & L = rl_[layer & 1];
    if (L.g[size_t(g)].id < 0 && !ring_issue(err)) {
        return false;
    }
    const ring_group & G = L.g[size_t(g)];
    if (G.id < 0) {
        return fail(err, "ring group " + std::to_string(g) + " of layer " + std::to_string(layer) + " not issued");
    }
    const int32_t sl = G.id % n_ring_ev;
    for (int32_t d = 0; d < dv.nd; ++d) {
        if (!dv.set(d, err) || !cuda_ok(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream[d]),
                                                            static_cast<cudaEvent_t>(ev_rcp_[d][sl]), 0),
                                        "ring group wait", err)) {
            return false;
        }
    }
    v = model::prompt_group_view();
    v.e_lo = G.g.e_lo;
    v.e_hi = G.g.e_hi;
    v.pairs = G.pairs;
    for (int32_t card = 0; card < 2; ++card) {
        const int32_t d = dv.dev_of(card);
        const uint8_t * const * base = reinterpret_cast<const uint8_t * const *>(res_->pool(d) + ring_lo_ + G.off);
        v.table[card][0] = base + dv.part_of(card, false) * n_ex;
        v.table[card][1] = base + dv.part_of(card, true) * n_ex;
    }
    return true;
}

// group g's release: recorded on each device's compute stream after both logical cards' gate/up and down (the
// caller's order), so a later copy into its bytes waits for them; the layer's last group may plan the next layer ahead
bool prompt_loan::prompt_group_done(int32_t layer, int32_t g, void * const stream[2], std::string & err) {
    const cache::devices & dv = *dv_;
    if (prompt_groups(layer) <= g || g < 0 || size_t(g) != rl_[layer & 1].done || rl_[layer & 1].g[size_t(g)].id < 0) {
        return fail(err, "ring group " + std::to_string(g) + " of layer " + std::to_string(layer) + " not acquired");
    }
    ring_layer & L = rl_[layer & 1];
    const int32_t id = L.g[size_t(g)].id, sl = id % n_ring_ev;
    for (int32_t d = 0; d < dv.nd; ++d) {
        if (!dv.set(d, err) || !cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ev_rrel_[d][sl]),
                                                        static_cast<cudaStream_t>(stream[d])),
                                        "ring release", err)) {
            return false;
        }
    }
    ring_->release(id);
    ++L.done;
    ++n_.ring_groups;
    if (L.done == L.g.size() && ahead_ && layer + 1 < fn::n_layer && rl_[(layer + 1) & 1].layer != layer + 1 &&
        !ring_plan(layer + 1, L.rows, nullptr, nullptr, err)) {
        return false;
    }
    return ring_issue(err);
}

const uint8_t * const * prompt_loan::prompt_table(int32_t layer, int32_t card, bool down) const {
    const int32_t b = layer & 1;
    if (layer < 0 || ptab_layer_[b] != layer || card < 0 || card > 1) {
        return nullptr;
    }
    return ptab_[dv_->dev_of(card)][b] + dv_->part_of(card, down) * n_ex;
}

std::string prompt_loan::plan_lines() const {
    if (lent_bytes_ == 0) {
        return "  lent region: none (two devices, or no device prompt buffers)\n";
    }
    const residency & r = *res_;
    int32_t in_lent = 0;
    int64_t lent_used = 0;
    for (int32_t s = r.zone_first(); s < r.n_slots(); ++s) {
        if (r.slot_off(s) >= r.lent_off()) {
            ++in_lent;
            lent_used += r.slot_bytes(s);
        }
    }
    std::string s;
    char b[640];
    std::snprintf(b, sizeof(b),
                  "  lent region (ME2b): %s (%lld bytes) at pool offset %lld, %d of the zone's slots (%s); "
                  "borrowed by a prompt from before its first write to after its last reader, refilled after "
                  "it:",
                  cache::gib(lent_bytes_).c_str(), (long long)lent_bytes_, (long long)r.lent_off(), in_lent,
                  cache::gib(lent_used).c_str());
    s += b;
    for (const lent_part & x : lent_) {
        std::snprintf(b, sizeof(b), " %s card %d %lld bytes at +%lld;", x.name.c_str(), x.card, (long long)x.bytes,
                      (long long)x.off);
        s += b;
    }
    return s + "\n";
}

expert_cache_counters prompt_loan::counters() const {
    std::lock_guard<std::mutex> lk(mu_);
    return counters_add(n_, svc_n_, 1);
}

void prompt_loan::reset_counters() {
    n_ = expert_cache_counters();
    std::lock_guard<std::mutex> lk(mu_);
    svc_n_ = expert_cache_counters();
}

} // namespace fn::runtime
