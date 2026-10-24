#pragma once

// ME4: the prompt policy's host planning model: no CUDA, no model, no route of a layer before that layer has computed
// it. One calculation for a prompt part [from, to): its path (known-token windows, or bulk chunks), the chunks and
// their halves (moved here from prefill.cpp unchanged), the arena layout and the bytes the prompt writes in the lent
// region (spans), each chunk's expert staging, the zone slots the loan borrows (loan::plan, each slot counted once
// however many intervals meet it), the restore set's bytes, and the byte bounds of the staging uploads. The runtime
// (prefill.cpp, cache_loan.cpp) and ex-cpu-test call it; work/ME4's derivation prints its accounts.
//
// The ME4 policy (tickets/ME4-cache-aware-prompt-staging.md, Result):
//   - a part of at most short_read rows: the target's windows (ME3's short read), no loan;
//   - a longer part: chunks of at most chunk_rows rows that never cross a multiple of 8,192 (PB3), halves above
//     1,536 rows; MX3: with the expert cache, halves only above 4,096 rows (pf_cached_split_rows), so every chunk up
//     to 4,096 rows is one half and stages through the ring;
//   - a one-half chunk (staging::ring): its experts through a bounded byte ring (MX3: two groups' entries for a run
//     of chunks of at most 512 rows, ring_for), one contiguous zone interval off the
//     prompt's spans chosen by the lowest current usage of the residents it would borrow (place_ring_scored; with no
//     usage, the highest offset: the zone's last boot-ranked slots. Address order is boot rank only: swaps refill
//     victims' slots in place, so it is not current heat). The experts go in expert-ID groups (each group's copies,
//     then its gate/up and down over that group's contiguous task range on both logical shards, then its space
//     released). A layer is route-first (only its routed nonresident experts, after its own router) until a layer's
//     routed set covers at least cover_pm per mille of that layer's nonresident bytes; the chunk's later layers are
//     then streamed ahead (every nonresident expert, issued while the previous layer computes). That switch is a
//     predictive scheduling heuristic: one layer's cover does not prove the next layer's, and the default is not a
//     demonstrated crossover; every routed expert is staged either way;
//   - a two-half chunk (staging::layers): both halves read one staged layer, so whole layers in the zone's two parity
//     buffers (ME3's arrangement and need); the last chunk stages no layer after its layer 47.
// Staging changes where an expert's packed bytes are read from, never its arithmetic: every task of the prompt
// expert kernels reads the same weights however they were staged or grouped.

#include "fn/runtime/loan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace fn::runtime {

// ---- the chunk geometry (moved from prefill.cpp; prefill-test's dry arms check it) ----

constexpr int64_t pf_span = 8192; // no chunk crosses an absolute multiple; checkpoints sit on them
// a chunk: two halves (FC-PF3). PB3: a whole 8,192 span a chunk (was 3,072), so each chunk's fixed cost (its sublayers'
// weight conversions, its expert weight passes, the pipeline's fill and drain, the join) is spread over more rows
constexpr int32_t pf_chunk_rows = 8192;
constexpr int32_t pf_half_rows = 4096;
constexpr int32_t pf_split_rows = 1536; // a chunk of at most this many rows runs as one half (as before PB3)
// MX3: with the expert cache (not its base arm) a chunk of at most a half's rows runs as one half, so it stages through
// the ring and its routing, not the parity buffers' whole layers; the halves' buffers already hold 4,096 rows
constexpr int32_t pf_cached_split_rows = pf_half_rows;
constexpr int32_t pf_slice_rows = 4;    // Stage 1's substitute: decode passes of B <= 4

struct pf_chunk {
    int64_t start = 0;
    int32_t rows = 0;
    int32_t half[2] = {0, 0}; // half[1] = 0 for a chunk of at most split_rows rows (plan_chunks)
};

// the chunks over positions [from, to); ME3: of at most max_rows rows (--ex-prompt-chunk); MX3: one half up to
// split_rows rows (pf_split_rows, or pf_cached_split_rows with the expert cache)
inline std::vector<pf_chunk> plan_chunks(int64_t from, int64_t to, int32_t max_rows = pf_chunk_rows,
                                         int32_t split_rows = pf_split_rows) {
    std::vector<pf_chunk> out;
    int64_t pos = std::max<int64_t>(from, 0);
    const int64_t most = std::min<int64_t>(std::max(max_rows, 1), pf_chunk_rows);
    while (pos < to) {
        // the tiling restarts at every multiple of 8,192 and at from
        const int64_t span_end = (pos / pf_span + 1) * pf_span;
        const int64_t end = std::min({pos + most, span_end, to});
        pf_chunk c;
        c.start = pos;
        c.rows = static_cast<int32_t>(end - pos);
        // up to split_rows one half; above, two near-equal halves, the first a multiple of 64 rows (a GDN block)
        c.half[0] = c.rows <= split_rows ? c.rows : std::min(pf_half_rows, ((c.rows + 1) / 2 + 63) / 64 * 64);
        c.half[1] = c.rows - c.half[0];
        out.push_back(c);
        pos = end;
    }
    return out;
}

// ME3: the largest half of any chunk of at most rows rows (plan_chunks' halves with split_rows)
inline int32_t pf_max_half(int32_t rows, int32_t split_rows = pf_split_rows) {
    const int32_t two = rows <= split_rows ? rows : std::min(pf_half_rows, ((rows + 1) / 2 + 63) / 64 * 64);
    return std::max(std::min(rows, split_rows), two);
}

// ME3: a known-token short read takes [from, to) when the expert service serves it (not the base arm) and the part is
// at most short_read tokens (0: never)
inline bool pf_short_read(int64_t from, int64_t to, int32_t short_read, bool service, bool legacy) {
    return service && !legacy && short_read > 0 && to - from <= short_read;
}
// ME3: a short read's next forced round at position n with u tokens left: at most five rows (the target windows'
// largest), none crossing a multiple of 8,192 (its checkpoint)
inline int32_t pf_short_rows(int64_t n, int64_t u) {
    const int64_t span_end = (n / pf_span + 1) * pf_span;
    return static_cast<int32_t>(std::min<int64_t>({5, u, span_end - n}));
}

// a checkpoint is taken at pos (a chunk's end) when pos is a multiple of 8,192 or the prompt's end
inline bool checkpoint_due(int64_t pos, int64_t prompt_end) {
    return pos > 0 && (pos % pf_span == 0 || pos == prompt_end);
}

namespace pplan {

constexpr int32_t n_used = 10;      // routed experts a row
constexpr int32_t task_pairs = 40;  // the prompt expert kernels' task (ep_task_rows)
constexpr int32_t pf_row_bytes = 2560 * 4; // a PF1 staging row (F32 PLE input)
constexpr int64_t align = 256;      // pool and slot alignment (every entry size is a multiple)
// MX3: a run whose chunks are at most this many rows takes the small ring (ring_for)
constexpr int32_t small_ring_rows = 512;

enum class path : int32_t { windows, bulk };
enum class staging : int32_t { ring, layers };

struct options {
    int32_t short_read = 64;                 // --ex-short-read (0: never)
    int32_t chunk_rows = pf_chunk_rows;      // --ex-prompt-chunk
    int64_t ring_bytes = int64_t(256) << 20; // ME4: a one-half run's ring
    int64_t group_bytes = int64_t(64) << 20; // ME4: a group's copied bytes at most (the ring holds at least two)
    int32_t cover_pm = 875;                  // ME4: stream ahead once a layer routes this per mille of its nonresident
    int32_t lend_pct = 85, min_keep = 128;   // the loan's cap and the slots never lent
    int32_t split_rows = pf_cached_split_rows; // MX3: the one-half bound (ME3's policy: pf_split_rows)
    int64_t table_bytes = 4 * 512 * 8;         // MX3: a ring entry's table (one device: four parts of 512 pointers)
    // ME3's policy (the comparison's base): a one-half chunk of at most routed_rows rows stages its routed experts in
    // the parity buffers, any other streams whole layers; the staging need is ME3's fixed point; the last chunk stages
    // layer 0 again after its layer 47 when it streamed
    bool me3 = false;
    int32_t routed_rows = 256;
};

// the pool of one device: offsets in bytes from its start
struct pool {
    std::vector<loan::zone_slot> zone; // the zone's slots in order (layer -1: empty)
    std::vector<double> score;         // per zone slot: its resident's current usage (empty or 0: none; place_ring)
    int64_t zone_off = 0, zone_buf = 0; // the parity buffers: [zone_off, + zone_buf), [zone_off + zone_buf, + zone_buf)
    int64_t pool_bytes = 0;            // the zone runs to the pool's end
    int64_t pf1[2] = {-1, -1};         // logical card c's PF1 staging (two slots of pf_chunk_rows rows); -1: not lent
    int64_t pb1[2] = {-1, -1};         // logical card c's PB1 arena; -1: not lent
    std::vector<int32_t> resident;     // per layer: residents anywhere in the pool before the loan
    std::vector<int64_t> dentry;       // per layer: one expert's bytes on this device
    int64_t n_slots = 0;
    int32_t n_ex = 512;
};

// the arena's bytes for chunks of at most rc rows in halves of at most rh (prompt_arena_bytes); MX6: per logical card
using arena_fn = int64_t (*)(int32_t rc, int32_t rh);
using card_arena_fn = int64_t (*)(int32_t rc, int32_t rh, int32_t card);

struct chunk_plan {
    pf_chunk c;
    staging s = staging::layers;
    bool routed_me3 = false; // ME3: the chunk stages its routed experts (one half, at most routed_rows rows)
};

struct plan {
    path p = path::windows;
    int64_t stage_off[2] = {0, 0};                  // the parity buffers' starts (layers runs)
    bool parity_moved = false;                      // ME4: placed by usage (else ME3's place at the zone's start)
    int32_t rounds = 0;                            // windows: the forced rounds
    std::vector<chunk_plan> chunks;                // bulk
    int32_t rc = 0, rh = 0;                        // the arena's layout
    std::vector<std::pair<int64_t, int64_t>> spans; // the prompt's writes in the lent region [lo, hi)
    int64_t ring_lo = 0, ring_hi = 0;               // ME4's ring [lo, hi) (lo == hi: none)
    loan::plan_output loan;
    int64_t span_bytes = 0;      // the spans' union
    int64_t refill_bytes = 0;    // the restore set's bytes (the borrowed slots' residents)
    std::vector<int32_t> res_out; // per layer: residents outside the loan
    int64_t stream_bytes = 0;    // one whole-layer chunk's uploads: every nonresident expert of every layer
    int64_t up_lo = 0, up_hi = 0; // the run's staging uploads, routes unknown
    int32_t stage_calls = 0;     // layer stagings (ME3's prompt_layers counter)
    int64_t wasted = 0;          // ME3: the last chunk's layer 0, staged again after its layer 47 for no chunk
};

inline int64_t al(int64_t n) {
    return (n + align - 1) / align * align;
}

// MX3: a run's ring for its largest chunk of rc rows: at most small_ring_rows, two entries of a group's bytes and its
// table (one copies while the other's group computes), else the configured ring
inline int64_t ring_for(int32_t rc, int64_t ring_bytes, int64_t group_bytes, int64_t table_bytes) {
    return rc <= small_ring_rows ? std::min(ring_bytes, 2 * al(group_bytes + table_bytes)) : ring_bytes;
}

// the union's bytes of intervals (each byte once)
inline int64_t union_bytes(std::vector<std::pair<int64_t, int64_t>> v) {
    std::sort(v.begin(), v.end());
    int64_t n = 0, end = INT64_MIN;
    for (const auto & x : v) {
        if (x.second <= x.first) {
            continue;
        }
        const int64_t lo = std::max(x.first, end);
        n += std::max<int64_t>(0, x.second - lo);
        end = std::max(end, x.second);
    }
    return n;
}

// ME4: the ring's place with no usage to go by: the highest bytes [lo, lo + bytes) in [zone_lo, zone_hi), lo aligned,
// meeting no span. policy::place fills the zone in boot rank order from its start, so this borrows the last boot-ranked
// slots; after swaps that is a boot-order heuristic, not the currently coldest
inline bool place_ring(int64_t zone_lo, int64_t zone_hi, const std::vector<std::pair<int64_t, int64_t>> & spans,
                       int64_t bytes, int64_t & lo) {
    std::vector<int64_t> tops = {zone_hi};
    for (const auto & x : spans) {
        tops.push_back(x.first);
    }
    std::sort(tops.rbegin(), tops.rend());
    for (const int64_t hi : tops) {
        const int64_t a = (hi - bytes) / align * align;
        if (hi > zone_hi || a < zone_lo) {
            continue;
        }
        bool meets = false;
        for (const auto & x : spans) {
            meets = meets || (a < x.second && x.first < a + bytes);
        }
        if (!meets) {
            lo = a;
            return true;
        }
    }
    return false;
}

// ME4: the ring's place by current usage: among the aligned intervals [lo, lo + bytes) in [zone_lo, zone_hi) meeting no
// span (starting at a zone slot, or ending at zone_hi or a span's start), the one whose met slots' score sums least
// (the residents the loan would retire, by the swap chooser's own usage measure); ties to the highest offset, so an
// all-zero score is place_ring's choice. The score predicts nothing beyond that measure
inline bool place_ring_scored(const std::vector<loan::zone_slot> & zone, const std::vector<double> & score,
                              int64_t zone_lo, int64_t zone_hi,
                              const std::vector<std::pair<int64_t, int64_t>> & spans, int64_t bytes, int64_t & lo) {
    std::vector<int64_t> cand = {(zone_hi - bytes) / align * align};
    for (const auto & x : spans) {
        cand.push_back((x.first - bytes) / align * align);
    }
    for (const auto & z : zone) {
        cand.push_back(z.off / align * align);
    }
    bool found = false;
    double best = 0;
    for (const int64_t a : cand) {
        if (a < zone_lo || a + bytes > zone_hi) {
            continue;
        }
        bool meets = false;
        for (const auto & x : spans) {
            meets = meets || (a < x.second && x.first < a + bytes);
        }
        if (meets) {
            continue;
        }
        double sc = 0;
        for (size_t j = 0; j < zone.size(); ++j) {
            if (zone[j].layer >= 0 && j < score.size() && a < zone[j].off + zone[j].bytes && zone[j].off < a + bytes) {
                sc += score[j];
            }
        }
        if (!found || sc < best || (sc == best && a > lo)) {
            found = true;
            best = sc;
            lo = a;
        }
    }
    return found;
}

// ME4: the parity buffers' place for a run with a two-half chunk, by the same rule as the ring: one contiguous interval
// [lo, lo + need0 + need1) of least current usage (place_ring_scored), buffer 0 then buffer 1, each sized by
// loan::plan's fixed point at that place (residents it retires raise the need). Iterated: the needs only grow, and a
// place whose needs fit its sizes is the answer (at most 8 rounds). in: the loan's input without staging offsets.
// false: no place fits (the caller keeps ME3's place at the zone's start); refused: err set (the cap, min_keep)
inline bool place_parity(const loan::plan_input & in, const std::vector<double> & score, int64_t zone_lo,
                         int64_t zone_hi, int64_t stage_off[2], loan::plan_output & out, std::string & err) {
    int64_t need[2] = {0, 0};
    for (size_t l = 0; l < in.dentry.size(); ++l) {
        need[l & 1] = std::max(need[l & 1], int64_t(in.worst ? in.n_ex : in.n_ex - in.resident[l]) * in.dentry[l]);
    }
    for (int32_t round = 0; round < 8; ++round) {
        int64_t lo = 0;
        const int64_t b0 = (need[0] + align - 1) / align * align;
        if (!place_ring_scored(in.zone, score, zone_lo, zone_hi, in.spans, b0 + need[1], lo)) {
            return false;
        }
        loan::plan_input q = in;
        q.stage_off[0] = lo;
        q.stage_off[1] = lo + b0;
        if (!loan::plan(q, out, err)) {
            return false;
        }
        if (out.stage_need[0] <= need[0] && out.stage_need[1] <= need[1]) {
            stage_off[0] = q.stage_off[0];
            stage_off[1] = q.stage_off[1];
            return true;
        }
        need[0] = std::max(need[0], out.stage_need[0]);
        need[1] = std::max(need[1], out.stage_need[1]);
    }
    return false;
}

// the spans of a run (prefiller::loan_spans): each PF1 staging slot's rows (the slot k % 2 takes chunk k), both logical
// cards, and each card's arena prefix for the layout
inline void spans_of(const pool & pl, const std::vector<pf_chunk> & chunks, const int64_t arena[2],
                     std::vector<std::pair<int64_t, int64_t>> & out) {
    int32_t rows[2] = {0, 0};
    for (size_t k = 0; k < chunks.size(); ++k) {
        rows[k % 2] = std::max(rows[k % 2], chunks[k].rows);
    }
    for (int32_t sl = 0; sl < 2; ++sl) {
        for (int32_t c = 0; c < 2 && rows[sl] > 0; ++c) {
            if (pl.pf1[c] >= 0) {
                const int64_t a = pl.pf1[c] + int64_t(sl) * pf_chunk_rows * pf_row_bytes;
                out.emplace_back(a, a + int64_t(rows[sl]) * pf_row_bytes);
            }
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (pl.pb1[c] >= 0) {
            out.emplace_back(pl.pb1[c], pl.pb1[c] + arena[c]);
        }
    }
}

inline void spans_of(const pool & pl, const std::vector<pf_chunk> & chunks, int64_t arena,
                     std::vector<std::pair<int64_t, int64_t>> & out) {
    const int64_t ar[2] = {arena, arena};
    spans_of(pl, chunks, ar, out);
}

// a layer's routed nonresident experts' bytes, bounds with routes unknown: at least the ten a row selects less the
// layer's residents, at most min(10 rows, nonresident)
inline int64_t routed_lo(int32_t rows, int32_t res, int64_t dentry) {
    return rows > 0 ? std::max<int64_t>(0, n_used - res) * dentry : 0;
}
inline int64_t routed_hi(int32_t rows, int32_t res, int32_t n_ex, int64_t dentry) {
    return std::min<int64_t>(int64_t(n_used) * rows, n_ex - res) * dentry;
}

// the run's plan (see the header). arena: prompt_arena_bytes. false and err: a part outside [0, to), a ring that has
// no place, or a loan refused by its cap or min_keep (loan::plan)
inline bool make(const options & o, const pool & pl, int64_t from, int64_t to, arena_fn arena, plan & out,
                 std::string & err, card_arena_fn card_arena = nullptr) {
    out = plan();
    const size_t nl = pl.dentry.size();
    if (from < 0 || to <= from || pl.resident.size() != nl || (arena == nullptr && card_arena == nullptr) ||
        o.group_bytes <= 0 || o.ring_bytes < 2 * o.group_bytes) {
        err = "prompt plan: a part outside [0, to), mismatched layers, or a ring that holds fewer than two groups";
        return false;
    }
    for (const int64_t e : pl.dentry) {
        if (e <= 0 || e % align != 0 || e > o.group_bytes) {
            err = "prompt plan: an expert entry not a positive multiple of 256 bytes or larger than a group";
            return false;
        }
    }
    if (pf_short_read(from, to, o.short_read, true, false)) {
        out.p = path::windows;
        for (int64_t n = from, u = to - from; u > 0;) {
            const int32_t b = pf_short_rows(n, u);
            n += b;
            u -= b;
            ++out.rounds;
        }
        out.res_out = pl.resident;
        return true;
    }
    out.p = path::bulk;
    const std::vector<pf_chunk> cs = plan_chunks(from, to, o.chunk_rows, o.me3 ? pf_split_rows : o.split_rows);
    bool two = false;
    for (const pf_chunk & c : cs) {
        out.rc = std::max(out.rc, c.rows);
        out.rh = std::max({out.rh, c.half[0], c.half[1]});
        two = two || c.half[1] > 0;
    }
    const bool lent = pl.pb1[0] >= 0 || pl.pf1[0] >= 0;
    if (lent) {
        const int64_t ar[2] = {card_arena != nullptr ? card_arena(out.rc, out.rh, 0) : arena(out.rc, out.rh),
                               card_arena != nullptr ? card_arena(out.rc, out.rh, 1) : arena(out.rc, out.rh)};
        spans_of(pl, cs, ar, out.spans);
    }
    out.span_bytes = union_bytes(out.spans);
    // the staging: ME3, or a run with a two-half chunk, the parity buffers and ME3's need (loan::plan's fixed point);
    // ME4's one-half runs, the ring as one more interval and no staging need
    const bool ring = !o.me3 && !two;
    loan::plan_input in;
    in.zone = pl.zone;
    in.spans = out.spans;
    in.stage_off[0] = pl.zone_off;
    in.stage_off[1] = pl.zone_off + pl.zone_buf;
    in.resident = pl.resident;
    in.dentry = pl.dentry;
    in.n_ex = pl.n_ex;
    in.n_slots = pl.n_slots;
    in.lend_pct = o.lend_pct;
    in.min_keep = o.min_keep;
    if (ring) {
        int64_t lo = 0;
        const int64_t rb = ring_for(out.rc, o.ring_bytes, o.group_bytes, o.table_bytes);
        if (!place_ring_scored(pl.zone, pl.score, pl.zone_off, pl.pool_bytes, out.spans, rb, lo)) {
            err = "prompt plan: no place in the zone for the ring outside the prompt's spans";
            return false;
        }
        out.ring_lo = lo;
        out.ring_hi = lo + rb;
        in.spans.emplace_back(out.ring_lo, out.ring_hi);
        in.dentry.assign(nl, 0); // no parity buffers: loan::plan's staging need stays 0
    }
    out.stage_off[0] = in.stage_off[0];
    out.stage_off[1] = in.stage_off[1];
    if (!ring && !o.me3) {
        // ME4: the parity buffers by usage; no place: ME3's
        std::string e2;
        int64_t so[2] = {0, 0};
        if (place_parity(in, pl.score, pl.zone_off, pl.pool_bytes, so, out.loan, e2)) {
            out.stage_off[0] = so[0];
            out.stage_off[1] = so[1];
            out.parity_moved = true;
        } else if (!e2.empty()) {
            err = e2;
            return false;
        }
    }
    if (!out.parity_moved && !loan::plan(in, out.loan, err)) {
        return false;
    }
    out.res_out = pl.resident;
    for (size_t j = 0; j < pl.zone.size(); ++j) {
        const int32_t l = pl.zone[j].layer;
        if (out.loan.take[j] != 0 && l >= 0) {
            --out.res_out[size_t(l)];
            out.refill_bytes += pl.dentry[size_t(l)];
        }
    }
    for (size_t l = 0; l < nl; ++l) {
        out.stream_bytes += int64_t(pl.n_ex - out.res_out[l]) * pl.dentry[l];
    }
    // per chunk: its staging and upload bounds. ME3: a streamed chunk's layer 47 stages layer 0 again (the next
    // chunk's, ready when it starts); after the last chunk that copy has no reader
    bool prev_streamed = false;
    for (size_t k = 0; k < cs.size(); ++k) {
        chunk_plan cp;
        cp.c = cs[k];
        cp.s = ring ? staging::ring : staging::layers;
        const bool one = cs[k].half[1] == 0;
        const int32_t r = cs[k].rows;
        cp.routed_me3 = o.me3 && one && r <= o.routed_rows;
        // ME4: every one-half chunk is route-first (a ring chunk may then stream its later layers ahead: its upper
        // bound is every nonresident expert); ME3: only up to routed_rows, never switching
        const bool routed = o.me3 ? cp.routed_me3 : one;
        for (size_t l = 0; l < nl; ++l) {
            const int32_t res = out.res_out[l];
            const int64_t all = int64_t(pl.n_ex - res) * pl.dentry[l];
            if (l == 0 && prev_streamed) {
                // the previous chunk's layer 47 staged this layer whole (ME3 and ME4's layers runs)
                out.up_lo += all;
                out.up_hi += all;
                continue;
            }
            ++out.stage_calls;
            out.up_lo += routed ? routed_lo(r, res, pl.dentry[l]) : all;
            out.up_hi += !routed || cp.s == staging::ring ? all : routed_hi(r, res, pl.n_ex, pl.dentry[l]);
        }
        const bool streamed = !routed;
        const bool last = k + 1 == cs.size();
        if (streamed && (o.me3 || !last)) {
            ++out.stage_calls; // layer 0 staged after layer 47
            if (last) {
                out.wasted = int64_t(pl.n_ex - out.res_out[0]) * pl.dentry[0];
                out.up_lo += out.wasted;
                out.up_hi += out.wasted;
            }
        }
        prev_streamed = streamed;
        out.chunks.push_back(cp);
    }
    return true;
}

inline bool make(const options & o, const pool & pl, int64_t from, int64_t to, card_arena_fn arena, plan & out,
                 std::string & err) {
    if (arena == nullptr) {
        err = "prompt plan: no arena function";
        return false;
    }
    return make(o, pl, from, to, nullptr, out, err, arena);
}

// ---- a ring chunk's layer: route-first or streamed ahead, its groups, the ring ----

// ME4: after layer l of a ring chunk routed its experts: stream the chunk's later layers ahead (every nonresident
// expert, before their routes) once this layer's routed nonresident bytes reach cover_pm per mille of its nonresident
// bytes. Route-first never copies more than streaming; streaming ahead only adds overlap of the copies with the layers'
// work before routing. A predictive heuristic: layer l's cover does not bound layer l + 1's, and cover_pm is a chosen
// default, not a measured crossover. Either way every routed expert is staged and computed
inline bool stream_after(int64_t routed_bytes, int64_t nonres_bytes, int32_t cover_pm) {
    return nonres_bytes > 0 && routed_bytes * 1000 >= int64_t(cover_pm) * nonres_bytes;
}

struct group {
    int32_t e_lo = 0, e_hi = 0; // experts [e_lo, e_hi): one contiguous task range of the sorted pairs
    int32_t copies = 0;         // its experts copied into the ring
    int64_t bytes = 0;          // their bytes
};

// ME4: a layer's expert-ID groups: copy[e] != 0 marks expert e copied (needed, not resident). The ranges partition
// [0, n_ex) in order, each copying at most group_bytes; a layer copying nothing is one group [0, n_ex). Resident and
// unrouted experts fall in whichever range covers them (their tasks read their slots, or do not exist)
inline bool plan_groups(const uint8_t * copy, int32_t n_ex, int64_t dentry, int64_t group_bytes,
                        std::vector<group> & out, std::string & err) {
    out.clear();
    if (copy == nullptr || n_ex < 1 || dentry <= 0 || dentry > group_bytes) {
        err = "prompt groups: no copy mask, or an entry larger than a group";
        return false;
    }
    group g;
    for (int32_t e = 0; e < n_ex; ++e) {
        if (copy[e] != 0 && g.bytes + dentry > group_bytes) {
            g.e_hi = e;
            out.push_back(g);
            g = group();
            g.e_lo = e;
        }
        if (copy[e] != 0) {
            ++g.copies;
            g.bytes += dentry;
        }
    }
    g.e_hi = n_ex;
    out.push_back(g);
    return true;
}

// the prompt expert kernels' tasks in a group of experts with these pairs (ep_group: an expert's pairs in tasks of at
// most 40), an upper bound for the group's grid when the pairs are not on the host: one partial task an expert
inline int32_t group_tasks_hi(int32_t experts, int64_t pairs) {
    return int32_t(std::min<int64_t>(experts, pairs) + (pairs + task_pairs - 1) / task_pairs);
}

// ME4: the prompt expert kernels' per-expert task offsets, as the cache instantiation's ep_task_offsets kernel computes
// them from ep_group's tasks (sorted by expert): off[e] = the first task whose expert is at least e, off[n_ex] = the
// task count. A group [e_lo, e_hi) runs tasks [off[e_lo], off[e_hi]), so groups that partition the experts
// partition the tasks: each task runs once, and ep_sum sees every pair
inline std::vector<int32_t> task_offsets(const std::vector<int32_t> & task_expert, int32_t n_ex) {
    std::vector<int32_t> off(size_t(n_ex) + 1, 0);
    for (int32_t e = 0; e <= n_ex; ++e) {
        off[size_t(e)] = int32_t(std::lower_bound(task_expert.begin(), task_expert.end(), e) - task_expert.begin());
    }
    return off;
}
// ep_group's task list for these per-expert pair counts (tasks of at most task_rows pairs, an expert's tasks together)
inline std::vector<int32_t> task_list(const std::vector<int32_t> & pairs, int32_t task_rows) {
    std::vector<int32_t> t;
    for (size_t e = 0; e < pairs.size(); ++e) {
        for (int32_t k = 0; k < (pairs[e] + task_rows - 1) / task_rows; ++k) {
            t.push_back(int32_t(e));
        }
    }
    return t;
}

// ME4: the ring's byte space and its entries (a group's copies each), first in first out. An entry is placed at the
// head, or at 0 when it does not fit before the end; it may reuse only the space of released entries (their reader's
// event recorded on the compute stream), whose events its copy waits for. So no copy overwrites bytes a kernel can
// still read, and every wait names an event already recorded (no copy is issued before the release it needs)
class byte_ring {
  public:
    explicit byte_ring(int64_t cap) : cap_(cap / align * align) {}

    // a place for bytes: false (nothing changed) when bytes exceed the ring or the space holds an entry not yet
    // released; else its offset, the released entries it reuses (waits) and its id
    bool alloc(int64_t bytes, int64_t & off, std::vector<int32_t> & waits, int32_t & id) {
        waits.clear();
        const int64_t b = al(bytes);
        if (b <= 0 || b > cap_) {
            return false;
        }
        const int64_t at = head_ + b <= cap_ ? head_ : 0;
        for (const entry & x : live_) {
            if (at < x.hi && x.lo < at + b && !x.released) {
                return false;
            }
        }
        for (auto it = live_.begin(); it != live_.end();) {
            if (at < it->hi && it->lo < at + b) {
                waits.push_back(it->id);
                it = live_.erase(it);
            } else {
                ++it;
            }
        }
        off = at;
        id = next_++;
        live_.push_back({id, at, at + b, false});
        head_ = at + b;
        return true;
    }
    bool release(int32_t id) {
        for (entry & x : live_) {
            if (x.id == id) {
                x.released = true;
                return true;
            }
        }
        return false;
    }
    int64_t cap() const { return cap_; }
    size_t live() const { return live_.size(); }

  private:
    struct entry {
        int32_t id;
        int64_t lo, hi;
        bool released;
    };
    int64_t cap_ = 0, head_ = 0;
    int32_t next_ = 0;
    std::deque<entry> live_;
};

// ---- MX7: the prompt CPU share (Strata-V100 69713b61 src/prefill/prefill.cpp 178-225 and 3333-3440, rewritten) ----
// On a route-first ring layer of a chunk below the share's limit, the CPU pool (experts_cpu.h) takes whole some
// routed nonresident experts that few of the chunk's tokens route to, instead of the ring staging them over PCIe.
// fn/runtime/prompt_share.h runs it; these are its host decisions.

constexpr int32_t share_maxt = 8;        // Strata's MAXT: a CPU expert's tokens at most (experts_cpu.h job_max_rows)
constexpr int32_t share_max_rows = 1024; // the default limit: chunks below it share (Strata's measured range)
constexpr double share_lo = 0.05, share_hi = 0.9; // auto's clamp
constexpr double share_alpha = 0.25;              // the running means' weight
constexpr int32_t share_checks = 3;               // the self-check's measured layers a side before it chooses
constexpr int32_t share_probe = 29;               // then every 29th layer runs the other side

// the CPU's experts of a route-first layer: counts[e] the pairs routed to expert e, resident[e] != 0 for a resident
// one. The candidates are the routed nonresident experts of at most maxt pairs, fewest first (ties by expert ID);
// the CPU takes llround(share x the routed nonresident count) of them, at most all, and at most max_pairs pairs in all
// (the buffers' bound). cpu[e] = 1 for a taken expert, else 0. Returns the taken count; nonres the routed nonresident
// count, pairs the taken experts' pairs
inline int32_t share_pick(const int32_t * counts, const uint8_t * resident, int32_t n_ex, int32_t maxt, double share,
                          int32_t max_pairs, uint8_t * cpu, int32_t & nonres, int32_t & pairs) {
    std::vector<std::pair<int32_t, int32_t>> cand; // (pairs, expert)
    nonres = 0;
    pairs = 0;
    for (int32_t e = 0; e < n_ex; ++e) {
        cpu[e] = 0;
        if (counts[e] < 1 || resident[e] != 0) {
            continue;
        }
        ++nonres;
        if (counts[e] <= maxt) {
            cand.emplace_back(counts[e], e);
        }
    }
    std::sort(cand.begin(), cand.end());
    const int64_t want = share > 0 ? std::llround(share * double(nonres)) : 0;
    const int32_t take = int32_t(std::min<int64_t>(int64_t(cand.size()), want));
    int32_t n = 0;
    for (int32_t i = 0; i < take && pairs + cand[size_t(i)].first <= max_pairs; ++i) {
        cpu[cand[size_t(i)].second] = 1;
        pairs += cand[size_t(i)].first;
        ++n;
    }
    return n;
}

// auto's share from the running means: the CPU's ms an expert c and the GPU's ms a staged expert g, so both sides end
// together, clamped to [share_lo, share_hi]; before both are measured, Strata's start (0.5)
inline double share_of(double c, double g) {
    return c > 0 && g > 0 ? std::clamp(g / (c + g), share_lo, share_hi) : 0.5;
}

// a running mean (share_alpha) taking a sample; the first sample is the mean
inline void share_mean(double & mean, int64_t & n, double x) {
    mean = n == 0 ? x : mean + share_alpha * (x - mean);
    ++n;
}

// auto's self-check (ticket MX7, the piece 2: Strata's): eligible layers alternate with and without the share,
// sharing first, until each side has share_checks measured layers; then each layer runs the side measured cheaper (a
// layer's cost: its expert work's span over its routed nonresident experts, a running mean a side), and every
// share_probe-th layer the other side, so the losing side stays measured. next() is the next eligible layer's side;
// record(on, cost) a measured layer's cost (layers are measured one layer later, after the next routing sync)
class share_check {
  public:
    bool next() {
        ++decided_;
        if (n_[0] < share_checks || n_[1] < share_checks) {
            const bool on = n_[1] < share_checks && (n_[0] >= share_checks || (decided_ - 1) % 2 == 0);
            return on;
        }
        const bool pref = pref_on();
        ++after_;
        if (after_ % share_probe == 0) {
            ++probes_;
            return !pref;
        }
        ++(pref ? chose_on_ : chose_off_);
        return pref;
    }
    void record(bool on, double cost) { share_mean(cost_[on ? 1 : 0], n_[on ? 1 : 0], cost); }
    bool warm() const { return n_[0] >= share_checks && n_[1] >= share_checks; }
    bool pref_on() const { return cost_[1] < cost_[0]; }
    double cost(bool on) const { return cost_[on ? 1 : 0]; }
    int64_t measured(bool on) const { return n_[on ? 1 : 0]; }
    int64_t chose_on() const { return chose_on_; }
    int64_t chose_off() const { return chose_off_; }
    int64_t probes() const { return probes_; }

  private:
    double cost_[2] = {0, 0}; // [off, on]
    int64_t n_[2] = {0, 0};
    int64_t decided_ = 0, after_ = 0, chose_on_ = 0, chose_off_ = 0, probes_ = 0;
};

} // namespace pplan
} // namespace fn::runtime
