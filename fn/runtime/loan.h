#pragma once

// ME3: the prompt loan's and the refill's arithmetic and state, free of the cache's buffers so ex-cpu-test checks them
// directly (cache_loan.h calls these). Offsets are bytes in the pool of device 0.

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fn::runtime::loan {

struct zone_slot {
    int64_t off = 0;    // the slot's byte offset in the pool
    int64_t bytes = 0;  // its size
    int32_t layer = -1; // the layer of the expert it holds; -1 none
};

struct plan_input {
    std::vector<zone_slot> zone;                    // the zone's slots in order
    std::vector<std::pair<int64_t, int64_t>> spans; // [lo, hi): the bytes the prompt writes in the lent region
    int64_t stage_off[2] = {0, 0};                  // the staging buffers' starts (buffer b: the layers l % 2 == b)
    std::vector<int32_t> resident;                  // per layer: its experts resident anywhere in the pool
    std::vector<int64_t> dentry;                    // per layer: one expert's bytes on this device
    int32_t n_ex = 512;                             // experts a layer
    bool worst = false;                             // every expert staged (the startup check)
    int64_t n_slots = 0;                            // the pool's slots (the cap's base)
    int32_t lend_pct = 85;                          // the loan's cap, % of n_slots
    int32_t min_keep = 128;                         // slots never lent
};

struct plan_output {
    std::vector<uint8_t> take; // per zone slot: borrowed
    int64_t stage_need[2] = {0, 0};
    int64_t n_taken = 0;
    int64_t taken_bytes = 0;
    int64_t taken_residents = 0;
};

// the zone slots a loan borrows: those intersecting a span or a staging buffer's prefix [stage_off[b],
// + stage_need[b]), each counted once however many intervals touch it. stage_need[b] is the largest, over its layers,
// of the experts not resident outside the loan (worst: all of them) times the layer's entry; the loan retires the
// residents of the slots it takes, which grows the need, so the two are iterated to a fixed point (monotone: a slot,
// once taken, stays taken; bounded by the zone). Refused when the loan exceeds lend_pct % of the pool's slots or
// leaves fewer than min_keep
inline bool plan(const plan_input & in, plan_output & out, std::string & err) {
    const size_t nz = in.zone.size();
    std::vector<int32_t> res = in.resident;
    if (res.size() != in.dentry.size()) {
        err = "loan plan: resident and entry rows differ";
        return false;
    }
    out = plan_output();
    out.take.assign(nz, 0);
    int64_t want[2] = {0, 0};
    for (;;) {
        for (size_t j = 0; j < nz; ++j) {
            const int64_t a = in.zone[j].off, b = a + in.zone[j].bytes;
            bool hit = false;
            for (const auto & x : in.spans) {
                hit = hit || (a < x.second && x.first < b);
            }
            for (int32_t k = 0; k < 2; ++k) {
                const int64_t lo = in.stage_off[k];
                hit = hit || (want[k] > 0 && a < lo + want[k] && lo < b);
            }
            if (hit && out.take[j] == 0) {
                out.take[j] = 1;
                const int32_t l = in.zone[j].layer;
                if (l >= 0) {
                    ++out.taken_residents;
                    if (!in.worst) {
                        --res[size_t(l)];
                    }
                }
            }
        }
        int64_t got[2] = {0, 0};
        for (size_t l = 0; l < in.dentry.size(); ++l) {
            got[l & 1] = std::max(got[l & 1], int64_t(in.worst ? in.n_ex : in.n_ex - res[l]) * in.dentry[l]);
        }
        if (got[0] <= want[0] && got[1] <= want[1]) {
            break;
        }
        want[0] = std::max(want[0], got[0]);
        want[1] = std::max(want[1], got[1]);
    }
    for (size_t j = 0; j < nz; ++j) {
        out.n_taken += out.take[j];
        out.taken_bytes += out.take[j] != 0 ? in.zone[j].bytes : 0;
    }
    if (out.n_taken * 100 > int64_t(in.lend_pct) * in.n_slots || in.n_slots - out.n_taken < in.min_keep) {
        err = "the prompt's loan (" + std::to_string(out.n_taken) + " of " + std::to_string(in.n_slots) +
              " slots) exceeds the loan cap (" + std::to_string(in.lend_pct) + " %) or keeps fewer than min-keep (" +
              std::to_string(in.min_keep) + ") slots";
        return false;
    }
    out.stage_need[0] = want[0];
    out.stage_need[1] = want[1];
    return true;
}

// the refill's next batch [a, end) of n moves in layer order (layer_of(i): move i's layer): from move a, every move
// whose layer is below a's plus layers (at least 1); at least one move (a < n)
template <typename F> size_t batch_end(size_t n, size_t a, int32_t layers, F layer_of) {
    size_t b = a;
    const int32_t l0 = layer_of(a);
    while (b < n && layer_of(b) < l0 + std::max(layers, 1)) {
        ++b;
    }
    return b;
}

// MX3: as batch_end, and the batch's bytes (bytes_of(i): move i's bytes on one device) at most max_bytes (0: no cap).
// MX13: whole layers only: the first layer always (a layer above the cap is a batch alone), each later layer only if
// the batch stays within max_bytes
template <typename F, typename G>
size_t batch_end_bytes(size_t n, size_t a, int32_t layers, int64_t max_bytes, F layer_of, G bytes_of) {
    const size_t e = batch_end(n, a, layers, layer_of);
    if (a >= e || max_bytes <= 0) {
        return e;
    }
    const int32_t l0 = layer_of(a);
    size_t b = a;
    int64_t sum = 0;
    while (b < e) {
        const int32_t lb = layer_of(b);
        size_t next_l = b;
        int64_t l_bytes = 0;
        while (next_l < e && layer_of(next_l) == lb) {
            l_bytes += bytes_of(next_l);
            ++next_l;
        }
        if (lb != l0 && sum + l_bytes > max_bytes) {
            break;
        }
        sum += l_bytes;
        b = next_l;
    }
    return b;
}

// ME3 (the cleanup): one refill's progress over its n moves (cache_loan.h owns it under its mutex). The moves are four
// ranges: [0, pub) published, [pub, landed) landed, [landed, flight) the batch in flight, [flight, n) unsubmitted.
// Phases: idle (no refill), running (batches may be submitted), stopping (no further batch). Each refill has
// a generation; a landing names the generation whose batch it was, and a stale one (a refill ended since) changes
// nothing.
// Legal transitions:
//   idle -start-> running; running -take-> (a batch in flight) -trim-> (shorter) -landed or -lost-> running;
//   running or stopping -stop-> stopping; any active -end-> idle (only once settled: no batch in flight).
// A failure (fail) keeps the phase and ends the waits: ready false, settled once nothing is in flight.
class refill_state {
  public:
    enum class phase { idle, running, stopping };

    uint32_t start(size_t n) { // idle only
        ph_ = phase::running;
        n_ = n;
        pub_ = landed_ = flight_ = 0;
        inflight_ = failed_ = false;
        return ++gen_;
    }
    bool active() const { return ph_ != phase::idle; }
    phase state() const { return ph_; }
    uint32_t gen() const { return gen_; }
    bool ready() const { return ph_ == phase::running && !inflight_ && !failed_ && flight_ < n_; }
    // the batch [a, b) in flight (ready, a the first unsubmitted, a < b <= n)
    bool take(size_t a, size_t b) {
        if (!ready() || a != flight_ || b <= a || b > n_) {
            return false;
        }
        flight_ = b;
        inflight_ = true;
        return true;
    }
    // the batch in flight ends at b instead (the yield: at least one move, landed_ < b <= flight_)
    bool trim(size_t b) {
        if (!inflight_ || b <= landed_ || b > flight_) {
            return false;
        }
        flight_ = b;
        return true;
    }
    // the batch of generation g landed (its callback); false: stale or none in flight
    bool landed(uint32_t g) {
        if (g != gen_ || !inflight_) {
            return false;
        }
        landed_ = flight_;
        inflight_ = false;
        return true;
    }
    // the batch in flight never enqueued its landing (a submission failed): no further batch
    void lost() {
        flight_ = landed_;
        inflight_ = false;
        failed_ = true;
    }
    void fail() { failed_ = true; }
    void stop() {
        if (ph_ == phase::running) {
            ph_ = phase::stopping;
        }
    }
    // a stopper's wait ends: no batch in flight, and stopping, failed or every move submitted
    bool settled() const { return !inflight_ && (ph_ == phase::stopping || failed_ || flight_ >= n_); }
    bool failed() const { return failed_; }
    size_t pub() const { return pub_; }
    size_t landed_end() const { return landed_; }
    size_t next() const { return flight_; } // the first move not yet submitted
    size_t size() const { return n_; }
    bool complete() const { return pub_ >= n_; }
    // [pub, end) published (end <= landed)
    bool published(size_t end) {
        if (end < pub_ || end > landed_) {
            return false;
        }
        pub_ = end;
        return true;
    }
    size_t outstanding() const { return active() ? n_ - pub_ : 0; }
    bool end() {
        if (inflight_) {
            return false;
        }
        ph_ = phase::idle;
        n_ = pub_ = landed_ = flight_ = 0;
        failed_ = false;
        return true;
    }

  private:
    phase ph_ = phase::idle;
    size_t n_ = 0, pub_ = 0, landed_ = 0, flight_ = 0;
    bool inflight_ = false, failed_ = false;
    uint32_t gen_ = 0;
};

} // namespace fn::runtime::loan
