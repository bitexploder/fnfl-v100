// ME4: the prompt planner's host checks (prompt_plan_test.h)

#include "fn/runtime/prompt_plan_test.h"

#include "fn/model/prompt.h"
#include "fn/runtime/loan.h"
#include "fn/runtime/prompt_plan.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace fn::runtime {

namespace {

namespace pp = fn::runtime::pplan;
namespace rt = fn::runtime;
namespace loan = fn::runtime::loan;

int32_t n_fail = 0, n_pass = 0;

void expect(bool ok, const std::string & what) {
    ++(ok ? n_pass : n_fail);
    std::printf("ex-cpu-test: me4 %s: %s\n", what.c_str(), ok ? "PASS" : "FAIL");
}

bool meets(int64_t a, int64_t b, const std::vector<std::pair<int64_t, int64_t>> & v) {
    for (const auto & x : v) {
        if (a < x.second && x.first < b) {
            return true;
        }
    }
    return false;
}

// a small pool: 4 layers of 8 experts, 256 bytes an expert; 4 class slots, then 20 zone slots of 256 bytes from
// 2,560 (slot j holding layer j % 4), so 6 residents a layer; the parity buffers [2,560, 3,584) and [3,584, 4,608);
// the lent region [4,608, 7,680) with logical card 0's and 1's arenas at 4,608 and 5,888 (no PF1 part: its slots are
// 8,192 rows)
pp::pool small_pool() {
    pp::pool pl;
    for (int32_t j = 0; j < 20; ++j) {
        pl.zone.push_back({2560 + 256 * int64_t(j), 256, j % 4});
    }
    pl.zone_off = 2560;
    pl.zone_buf = 1024;
    pl.pool_bytes = 7680;
    pl.pb1[0] = 4608;
    pl.pb1[1] = 5888;
    pl.resident.assign(4, 6);
    pl.dentry.assign(4, 256);
    pl.n_ex = 8;
    pl.n_slots = 24;
    return pl;
}

int64_t small_arena(int32_t rc, int32_t) {
    return 256 + rc / 8;
}

void check_chunks() {
    // invariants over many parts: chunks tile [from, to), never cross a multiple of 8,192, at most max rows, halves
    // above 1,536 rows (the first a multiple of 64, both at most 4,096); a restart at a multiple repeats the tiling
    const int64_t froms[] = {0, 1, 63, 8000, 8191, 8192, 13433, 16383};
    const int64_t lens[] = {1, 64, 65, 1536, 1537, 3000, 8192, 8193, 20000};
    const int32_t maxes[] = {256, 1536, 2048, 8192};
    bool ok = true;
    for (const int64_t f : froms) {
        for (const int64_t n : lens) {
            for (const int32_t m : maxes) {
                const auto cs = rt::plan_chunks(f, f + n, m);
                int64_t pos = f;
                for (const auto & c : cs) {
                    ok = ok && c.start == pos && c.rows >= 1 && c.rows <= m && c.rows <= rt::pf_chunk_rows;
                    ok = ok && c.start / rt::pf_span == (c.start + c.rows - 1) / rt::pf_span;
                    ok = ok && c.half[0] + c.half[1] == c.rows && c.half[0] <= rt::pf_half_rows &&
                         c.half[1] <= rt::pf_half_rows;
                    ok = ok && (c.rows <= rt::pf_split_rows ? c.half[1] == 0 : (c.half[1] > 0 && c.half[0] % 64 == 0));
                    pos += c.rows;
                }
                ok = ok && pos == f + n;
                const int64_t k = (f / rt::pf_span + 1) * rt::pf_span;
                if (k < f + n) {
                    const auto again = rt::plan_chunks(k, f + n, m);
                    size_t i = 0;
                    while (i < cs.size() && cs[i].start < k) {
                        ++i;
                    }
                    ok = ok && cs.size() - i == again.size();
                    for (size_t j = 0; ok && j < again.size(); ++j) {
                        ok = ok && again[j].start == cs[i + j].start && again[j].rows == cs[i + j].rows;
                    }
                }
            }
        }
    }
    expect(ok, "chunks: tiling, no crossing of 8,192, halves, a restart's partition");
    expect(rt::pf_max_half(1536) == 1536 && rt::pf_max_half(1537) == 1536 && rt::pf_max_half(8192) == 4096 &&
               rt::pf_max_half(3000) == 1536 && rt::pf_max_half(65) == 65,
           "chunks: the largest half (1,536 | 1,537 | 8,192 | 3,000 | 65)");
    // MX3's cached rule: one half to 4,096 rows, two halves above (the first a multiple of 64)
    bool cached = true;
    for (const int64_t f : froms) {
        for (const int64_t n : lens) {
            for (const auto & c : rt::plan_chunks(f, f + n, rt::pf_chunk_rows, rt::pf_cached_split_rows)) {
                cached = cached && (c.rows <= rt::pf_cached_split_rows
                                        ? c.half[1] == 0
                                        : (c.half[1] > 0 && c.half[0] % 64 == 0 && c.half[0] <= rt::pf_half_rows));
            }
        }
    }
    const int32_t cs = rt::pf_cached_split_rows;
    expect(cached && rt::pf_max_half(2049, cs) == 2049 && rt::pf_max_half(4096, cs) == 4096 &&
               rt::pf_max_half(8192, cs) == 4096 && rt::pf_max_half(65, cs) == 65,
           "chunks (MX3, split 4,096): one half to 4,096 rows; the largest half (2,049 | 4,096 | 8,192 | 65)");
    // the checkpoint arithmetic and the tail across a boundary
    const auto t = rt::plan_chunks(16184, 16584);
    expect(t.size() == 2 && t[0].rows == 200 && t[1].start == 16384 && rt::checkpoint_due(16384, 16584) &&
               rt::checkpoint_due(16584, 16584) && !rt::checkpoint_due(16583, 16584),
           "chunks: a 400-row part across 16,384 is 200 + 200, checkpoints at 16,384 and its end");
}

void check_paths() {
    const pp::pool pl = small_pool();
    pp::options o;
    o.ring_bytes = 512;
    o.group_bytes = 256;
    o.min_keep = 2;
    const auto ar = small_arena;
    pp::plan p;
    std::string err;
    // the pending token: a follow-up of 63 IDs processes 64 rows (windows), 64 IDs 65 rows (bulk)
    const bool w64 = pp::make(o, pl, 1774, 1774 + 64, ar, p, err) && p.p == pp::path::windows && p.rounds == 13;
    const bool b65 = pp::make(o, pl, 1774, 1774 + 65, ar, p, err) && p.p == pp::path::bulk && p.chunks.size() == 1 &&
                     p.chunks[0].s == pp::staging::ring;
    expect(w64 && b65, "paths: 64 rows windows (13 rounds), 65 rows one ring chunk");
    o.short_read = 0;
    expect(pp::make(o, pl, 0, 13, ar, p, err) && p.p == pp::path::bulk, "paths: short read 0 always bulk");
    o.short_read = 64;
    // a windows round never crosses a multiple of 8,192
    expect(pp::make(o, pl, 8190, 8190 + 13, ar, p, err) && p.rounds == 4, "paths: windows across 8,192: 2,5,5,1");
    // one-half runs ring, any two-half chunk layers (and its one-half tail routed in the parity buffers). Uncached
    // rule (split 1,536): 1,537 rows is two halves; MX3's cached rule (the default, split 4,096): one half to 4,096
    pp::options u = o;
    u.split_rows = rt::pf_split_rows;
    const bool one =
        pp::make(u, pl, 0, 1536, ar, p, err) && p.chunks[0].s == pp::staging::ring && p.ring_hi > p.ring_lo;
    const bool two = pp::make(u, pl, 0, 1537, ar, p, err) && p.chunks[0].s == pp::staging::layers &&
                     p.ring_hi == p.ring_lo && p.loan.stage_need[0] > 0;
    expect(one && two,
           "paths (split 1,536): 1,536 rows a ring chunk, 1,537 rows two halves through the parity buffers");
    bool cached = true;
    for (const int64_t n : {int64_t(1537), int64_t(2049), int64_t(4096)}) {
        cached = cached && pp::make(o, pl, 0, n, ar, p, err) && p.chunks.size() == 1 && p.chunks[0].c.half[1] == 0 &&
                 p.chunks[0].s == pp::staging::ring && p.ring_hi > p.ring_lo && p.loan.stage_need[0] == 0;
    }
    const bool c4097 = pp::make(o, pl, 0, 4097, ar, p, err) && p.chunks[0].c.half[1] > 0 &&
                       p.chunks[0].s == pp::staging::layers && p.ring_hi == p.ring_lo;
    const bool c8192 = pp::make(o, pl, 0, 8192, ar, p, err) && p.chunks.size() == 1 && p.chunks[0].c.half[0] == 4096 &&
                       p.chunks[0].c.half[1] == 4096 && p.chunks[0].s == pp::staging::layers;
    expect(cached && c4097 && c8192, "paths (MX3, split 4,096): 1,537, 2,049 and 4,096 rows one ring chunk (no staging "
                                     "need), 4,097 two halves, 8,192 two halves of 4,096");
    // ME3: routed only to 256 rows, then streamed with the wrap after the last chunk
    pp::options m3 = o;
    m3.me3 = true;
    const bool r256 = pp::make(m3, pl, 0, 256, ar, p, err) && p.chunks[0].routed_me3 && p.stage_calls == 4 &&
                      p.wasted == 0;
    const bool s257 = pp::make(m3, pl, 0, 257, ar, p, err) && !p.chunks[0].routed_me3 && p.stage_calls == 5 &&
                      p.wasted == int64_t(8 - p.res_out[0]) * 256 && p.wasted > 0;
    pp::plan q;
    const bool m4 = pp::make(o, pl, 0, 257, ar, q, err) && q.stage_calls == 4 && q.wasted == 0;
    expect(r256 && s257 && m4, "paths: ME3 routes 256 rows, streams 257 with a wasted wrap; ME4 has no wrap");
    // two streamed chunks: the wrap feeds the next chunk (ME3 and ME4), only the last one is wasted (ME3)
    const bool w2 = pp::make(m3, pl, 0, 16384, ar, p, err) && p.stage_calls == 9 && p.wasted > 0;
    const bool w4 = pp::make(o, pl, 0, 16384, ar, q, err) && q.stage_calls == 8 && q.wasted == 0;
    expect(w2 && w4, "paths: two streamed chunks: ME3 9 stagings (one wasted), ME4 8");
}

void check_ring_and_loan() {
    pp::pool pl = small_pool();
    pp::options o;
    o.ring_bytes = 512;
    o.group_bytes = 256;
    o.min_keep = 2;
    const auto ar = small_arena;
    std::mt19937 rng(1234);
    bool ok = true;
    for (int32_t i = 0; i < 400; ++i) {
        const int64_t from = int64_t(rng() % 20000);
        const int64_t to = from + 65 + int64_t(rng() % 1472); // one-half parts
        pp::plan p;
        std::string err;
        if (!pp::make(o, pl, from, to, ar, p, err)) {
            ok = false;
            break;
        }
        // the ring: inside the zone, aligned, meeting no span
        ok = ok && p.ring_lo >= pl.zone_off && p.ring_hi <= pl.pool_bytes && p.ring_lo % pp::align == 0 &&
             !meets(p.ring_lo, p.ring_hi, p.spans);
        // every slot meeting the ring or a span is borrowed, every other kept; residents and refill counted once
        std::vector<std::pair<int64_t, int64_t>> all = p.spans;
        all.emplace_back(p.ring_lo, p.ring_hi);
        int64_t refill = 0;
        std::vector<int32_t> res = pl.resident;
        for (size_t j = 0; j < pl.zone.size(); ++j) {
            const bool m = meets(pl.zone[j].off, pl.zone[j].off + pl.zone[j].bytes, all);
            ok = ok && m == (p.loan.take[j] != 0);
            if (m && pl.zone[j].layer >= 0) {
                refill += pl.dentry[size_t(pl.zone[j].layer)];
                --res[size_t(pl.zone[j].layer)];
            }
        }
        ok = ok && refill == p.refill_bytes && res == p.res_out;
    }
    expect(ok, "ring and loan: 400 random one-half parts: ring placed in the zone off the spans, the borrowed slots "
               "exactly those meeting them, refill and residents counted once");
    // the ring goes to the highest free bytes: the zone's end when free, else just below the highest span
    {
        int64_t lo = 0;
        const bool a = pp::place_ring(1024, 4096, {{2048, 2304}, {2560, 2816}}, 512, lo) && lo == 3584;
        const bool b = pp::place_ring(1024, 4096, {{2048, 2304}, {2560, 4096}}, 512, lo) && lo == 1536;
        const bool c = !pp::place_ring(1024, 1536, {{1024, 1536}}, 256, lo);
        expect(a && b && c, "ring placement: highest gap, below a span reaching the end, refused without a gap");
    }
    // overlap counted once: a span given twice borrows the same slots
    {
        pp::plan p1, p2;
        std::string err;
        const pp::pool & q = pl;
        pp::make(o, q, 0, 100, small_arena, p1, err);
        loan::plan_input in;
        in.zone = q.zone;
        in.spans = p1.spans;
        in.spans.insert(in.spans.end(), p1.spans.begin(), p1.spans.end());
        in.spans.emplace_back(p1.ring_lo, p1.ring_hi);
        in.spans.emplace_back(p1.ring_lo, p1.ring_hi);
        in.resident = q.resident;
        in.dentry.assign(4, 0);
        in.n_ex = 8;
        in.n_slots = 24;
        in.min_keep = 2;
        loan::plan_output out;
        expect(loan::plan(in, out, err) && out.take == p1.loan.take && out.taken_bytes == p1.loan.taken_bytes,
               "loan: intervals given twice borrow each slot once");
    }
    // the cap and min-keep refuse (nothing evicted beyond the plan)
    {
        pp::options c = o;
        c.lend_pct = 5;
        pp::plan p;
        std::string err;
        const bool cap = !pp::make(c, pl, 0, 100, ar, p, err);
        c = o;
        c.min_keep = 23;
        const bool keep = !pp::make(c, pl, 0, 100, ar, p, err);
        expect(cap && keep, "loan: refused over the cap (5 %) and under min-keep (23 of 24)");
    }
    // retained resident pointers: a kept slot's bytes meet neither the ring nor a span, so its experts' prompt
    // table entries stay valid through the prompt
    {
        pp::plan p;
        std::string err;
        pp::make(o, pl, 0, 300, ar, p, err);
        std::vector<std::pair<int64_t, int64_t>> all = p.spans;
        all.emplace_back(p.ring_lo, p.ring_hi);
        bool kept_ok = true;
        for (size_t j = 0; j < pl.zone.size(); ++j) {
            if (p.loan.take[j] == 0) {
                kept_ok = kept_ok && !meets(pl.zone[j].off, pl.zone[j].off + pl.zone[j].bytes, all);
            }
        }
        expect(kept_ok, "loan: every kept slot is untouched by the ring and the spans");
    }
}

void check_groups_and_ring() {
    std::string err;
    std::vector<pp::group> g;
    // empty, all-hit (nothing copied): one group over every expert
    std::vector<uint8_t> none(512, 0), all(512, 1);
    expect(pp::plan_groups(none.data(), 512, 2329600, int64_t(64) << 20, g, err) && g.size() == 1 && g[0].e_lo == 0 &&
               g[0].e_hi == 512 && g[0].copies == 0,
           "groups: nothing to copy (empty or all-hit routes) is one group, no copy");
    // all-miss: every group at most 64 MiB, ranges partition [0, 512)
    bool ok = pp::plan_groups(all.data(), 512, 2329600, int64_t(64) << 20, g, err);
    int32_t at = 0, copies = 0;
    for (const auto & x : g) {
        ok = ok && x.e_lo == at && x.e_hi > x.e_lo && x.bytes <= (int64_t(64) << 20) && x.copies > 0;
        at = x.e_hi;
        copies += x.copies;
    }
    expect(ok && at == 512 && copies == 512 && g.size() == 19, "groups: all-miss, 2,329,600-byte entries: 19 groups "
                                                                "of at most 64 MiB partitioning 0..512");
    // random masks for every entry size: partition, bound, copy count
    std::mt19937 rng(77);
    const int64_t entries[] = {1305600, 1408000, 1510400, 1715200, 1766400, 1868800, 1971200, 2176000, 2329600};
    ok = true;
    for (int32_t t = 0; t < 300; ++t) {
        std::vector<uint8_t> m(512);
        int32_t want = 0;
        const uint32_t p = rng() % 101;
        for (auto & b : m) {
            b = rng() % 100 < p ? 1 : 0;
            want += b;
        }
        const int64_t de = entries[t % 9];
        ok = ok && pp::plan_groups(m.data(), 512, de, int64_t(64) << 20, g, err);
        at = 0;
        copies = 0;
        for (const auto & x : g) {
            ok = ok && x.e_lo == at && x.bytes <= (int64_t(64) << 20) && x.bytes == int64_t(x.copies) * de;
            at = x.e_hi;
            copies += x.copies;
        }
        ok = ok && at == 512 && copies == want;
    }
    expect(ok, "groups: 300 random masks over the nine entry sizes partition 0..512 within 64 MiB a group");
    expect(!pp::plan_groups(all.data(), 512, int64_t(65) << 20, int64_t(64) << 20, g, err),
           "groups: refused when one entry exceeds a group");
    expect(pp::group_tasks_hi(64, 640) == 64 + 16 && pp::group_tasks_hi(512, 15360) == 512 + 384 &&
               pp::group_tasks_hi(10, 10) == 10 + 1,
           "groups: task bound (an expert's partial task plus 40-pair tasks)");

    // the ring, as the host would drive it: copies issued ahead while their space is released, each group's reader
    // enqueued in order then released. No live unreleased overlap, waits only on released entries, progress, and
    // with at least two groups' space a copy ahead of each compute
    for (const int64_t cap : {int64_t(128) << 20, int64_t(256) << 20, int64_t(512) << 20}) {
        pp::byte_ring r(cap);
        struct live {
            int32_t id;
            int64_t lo, hi;
            bool released;
        };
        std::vector<live> lv;
        std::vector<int64_t> sizes;
        for (int32_t l = 0; l < 48; ++l) {
            std::vector<uint8_t> m(512);
            for (auto & b : m) {
                b = rng() % 100 < 70 ? 1 : 0;
            }
            pp::plan_groups(m.data(), 512, entries[l % 9], int64_t(64) << 20, g, err);
            for (const auto & x : g) {
                if (x.bytes > 0) {
                    sizes.push_back(x.bytes);
                }
            }
        }
        size_t issued = 0, computed = 0;
        int64_t ahead = 0;
        bool good = true;
        std::vector<int32_t> ids(sizes.size(), -1);
        while (computed < sizes.size() && good) {
            // issue ahead
            for (;;) {
                if (issued >= sizes.size()) {
                    break;
                }
                int64_t off = 0;
                int32_t id = -1;
                std::vector<int32_t> waits;
                if (!r.alloc(sizes[issued], off, waits, id)) {
                    break;
                }
                for (const int32_t w : waits) {
                    bool rel = false;
                    for (auto it = lv.begin(); it != lv.end(); ++it) {
                        if (it->id == w) {
                            rel = it->released;
                            lv.erase(it);
                            break;
                        }
                    }
                    good = good && rel;
                }
                for (const auto & x : lv) {
                    good = good && !(off < x.hi && x.lo < off + pp::al(sizes[issued]));
                }
                good = good && off >= 0 && off + sizes[issued] <= r.cap() && off % pp::align == 0;
                lv.push_back({id, off, off + pp::al(sizes[issued]), false});
                ids[issued++] = id;
            }
            good = good && issued > computed; // progress: the next group's copy is issued
            ahead += int64_t(issued - computed - 1);
            r.release(ids[computed]);
            for (auto & x : lv) {
                x.released = x.released || x.id == ids[computed];
            }
            ++computed;
        }
        char b[256]; // room for the text with wide counts (-Wformat-truncation)
        std::snprintf(b, sizeof(b), "byte ring %" PRId64 " MiB: %zu groups, no overwrite of a live entry, waits on "
                      "released entries only, progress; copies ahead of compute on average %.2f",
                      cap >> 20, sizes.size(), double(ahead) / double(sizes.size()));
        expect(good && (cap < (int64_t(128) << 20) || ahead > 0), b);
    }
}

void check_switch() {
    // the route-first to streamed-ahead switch: deterministic in a layer's own routes
    expect(!pp::stream_after(0, 0, 875) && !pp::stream_after(0, 100, 875) && !pp::stream_after(874, 1000, 875) &&
               pp::stream_after(875, 1000, 875) && pp::stream_after(1000, 1000, 875),
           "switch (a predictive heuristic): no nonresident bytes never streams (all-hit), empty routes never, "
           "87.5 % switches");
}

// MX3: a Config D-like pool for the loan's sizes: 48 layers of 512 experts of 1,766,400 bytes (a layer's mean entry,
// counted), 4,802 zone slots from 0 holding layer j % 48 (100 residents a layer, two empty), the lent region at the
// pool's tail: logical cards 0 and 1's PF1 staging (two 8,192-row slots each) then their PB1 arenas at the full layout
// (prompt_arena_bytes). Spans and ring placed as the runtime places them (no usage: the highest free bytes)
pp::pool d_pool() {
    pp::pool pl;
    const int64_t de = 1766400;
    for (int32_t j = 0; j < 4802; ++j) {
        pl.zone.push_back({int64_t(j) * de, de, j < 4800 ? j % 48 : -1});
    }
    pl.zone_off = 0;
    pl.pool_bytes = int64_t(4802) * de;
    const int64_t pf1 = int64_t(2) * rt::pf_chunk_rows * pp::pf_row_bytes, arena = pp::al(model::prompt_arena_bytes());
    const int64_t lent = 2 * pf1 + 2 * arena;
    pl.pf1[0] = pl.pool_bytes - lent;
    pl.pf1[1] = pl.pf1[0] + pf1;
    pl.pb1[0] = pl.pf1[1] + pf1;
    pl.pb1[1] = pl.pb1[0] + arena;
    pl.resident.assign(48, 100);
    pl.dentry.assign(48, de);
    pl.n_ex = 512;
    pl.n_slots = 4802;
    pl.zone_buf = 512 * de;
    return pl;
}

int64_t d_arena(int32_t rc, int32_t rh) {
    return model::prompt_arena_bytes(rc, rh);
}

void check_mx3_plans() {
    const pp::pool pl = d_pool();
    pp::options o; // the deployed defaults: ring 256 MiB, groups 64 MiB, the cached split
    o.lend_pct = 98;
    o.min_keep = 64;
    pp::options u = o;
    u.split_rows = rt::pf_split_rows; // 134da68's rule
    const int64_t mib = int64_t(1) << 20, gib = int64_t(1) << 30;
    std::string err;
    char b[512];
    for (const int32_t n : {2049, 4096}) {
        pp::plan c, t;
        const bool ok = pp::make(o, pl, 131072, 131072 + n, d_arena, c, err) &&
                        pp::make(u, pl, 131072, 131072 + n, d_arena, t, err);
        const bool ring = ok && c.chunks.size() == 1 && c.chunks[0].c.half[1] == 0 &&
                          c.chunks[0].s == pp::staging::ring && c.ring_hi - c.ring_lo == 256 * mib &&
                          c.loan.stage_need[0] == 0 && c.loan.stage_need[1] == 0;
        const bool parity = ok && t.chunks[0].c.half[1] > 0 && t.chunks[0].s == pp::staging::layers;
        const int64_t less = ok ? t.loan.taken_bytes - c.loan.taken_bytes : 0;
        std::snprintf(b, sizeof(b),
                      "plan (MX3, a synthetic Config D-like pool): %d rows at 131,072 one half through a 256 MiB "
                      "ring, no staging need, borrowing %.3f GiB (%lld residents) against 134da68's two halves %.3f "
                      "GiB (%lld): %.3f GiB less (counted here; the run-dry reports Config D's own)",
                      n, double(c.loan.taken_bytes) / double(gib), (long long)c.loan.taken_residents,
                      double(t.loan.taken_bytes) / double(gib), (long long)t.loan.taken_residents,
                      double(less) / double(gib));
        expect(ring && parity && less > 0, b);
    }
    {
        pp::plan c, t;
        const bool ok = pp::make(o, pl, 131072, 131072 + 257, d_arena, c, err) &&
                        pp::make(o, pl, 131072, 131072 + 513, d_arena, t, err);
        const int64_t small = 2 * pp::al(64 * mib + o.table_bytes);
        std::snprintf(b, sizeof(b),
                      "plan (MX3): 257 rows a ring of two group entries (128 MiB and two 16 KiB tables, %lld bytes), "
                      "513 rows 256 MiB, a smaller configured ring kept; the 257-row loan %.1f MiB (%lld residents)",
                      (long long)(c.ring_hi - c.ring_lo), double(c.loan.taken_bytes) / double(mib),
                      (long long)c.loan.taken_residents);
        expect(ok && c.ring_hi - c.ring_lo == small && t.ring_hi - t.ring_lo == 256 * mib &&
                   pp::ring_for(512, 256 * mib, 64 * mib, o.table_bytes) == small &&
                   pp::ring_for(512, 96 * mib, 64 * mib, o.table_bytes) == 96 * mib,
               b);
    }
    {
        pp::plan c;
        const bool ok = pp::make(o, pl, 131072, 131072 + 8192, d_arena, c, err) && c.chunks.size() == 1 &&
                        c.chunks[0].c.half[0] == 4096 && c.chunks[0].c.half[1] == 4096 &&
                        c.chunks[0].s == pp::staging::layers && c.loan.stage_need[0] > 0;
        expect(ok, "plan (MX3): 8,192 rows two halves of 4,096 through the parity buffers");
    }
}

// MX3: the small ring holds two full group entries (a group's 64 MiB and its table) at once; a third waits for the
// first's release and reuses its bytes
void check_small_ring() {
    const int64_t e = (int64_t(64) << 20) + 16384;
    pp::byte_ring r(2 * pp::al(e));
    int64_t off = 0;
    int32_t id0 = -1, id1 = -1, id2 = -1;
    std::vector<int32_t> w;
    bool f = r.alloc(e, off, w, id0) && r.alloc(e, off, w, id1) && !r.alloc(e, off, w, id2);
    r.release(id0);
    f = f && r.alloc(e, off, w, id2) && off == 0 && w.size() == 1 && w[0] == id0 && !r.alloc(1, off, w, id2);
    expect(f, "byte ring (MX3): two 64 MiB + table entries in the small ring, a third only after the first's release");
}

void check_refill_ownership() {
    // a loan's restore set = the borrowed slots' residents; a refill stopped part-way publishes what landed and carries
    // the unsubmitted rest back, so published + carried is exactly the set (loan.h refill_state)
    loan::refill_state st;
    const size_t n = 37;
    st.start(n);
    size_t a = st.next(), b = std::min(n, a + 10);
    bool ok = st.take(a, b) && st.landed(st.gen()) && st.published(b);
    a = st.next();
    b = std::min(n, a + 10);
    ok = ok && st.take(a, b);
    st.stop();
    ok = ok && !st.settled() && st.landed(st.gen()) && st.settled() && st.published(st.landed_end());
    const size_t published = st.pub(), carried = n - published;
    ok = ok && published == 20 && carried == 17 && published + carried == n && st.end() && !st.active();
    // a stale landing (a refill ended since) changes nothing
    const uint32_t g0 = st.start(5);
    ok = ok && st.take(0, 5);
    st.lost();
    ok = ok && st.failed() && st.settled() && !st.landed(g0) && st.end();
    expect(ok, "refill: stop part-way publishes 20 landed, carries 17 back (37); a lost batch fails and settles");
}

void check_scored_and_tasks() {
    // the scored place: away from the residents with the most current usage, never on a span, the highest place when
    // nothing scores (place_ring's)
    const pp::pool pl = small_pool();
    const std::vector<std::pair<int64_t, int64_t>> spans = {{4608, 4864}, {5888, 6144}};
    std::vector<double> sc(pl.zone.size(), 0.0);
    int64_t a = 0, b = 0;
    bool ok = pp::place_ring_scored(pl.zone, sc, pl.zone_off, pl.pool_bytes, spans, 512, a) &&
              pp::place_ring(pl.zone_off, pl.pool_bytes, spans, 512, b) && a == b;
    sc.back() = 5.0; // the last slot [7424, 7680) is hot now
    ok = ok && pp::place_ring_scored(pl.zone, sc, pl.zone_off, pl.pool_bytes, spans, 512, a) && a + 512 <= 7424;
    std::mt19937 rng(9);
    for (int32_t t = 0; t < 200 && ok; ++t) {
        for (double & x : sc) {
            x = double(rng() % 7);
        }
        ok = pp::place_ring_scored(pl.zone, sc, pl.zone_off, pl.pool_bytes, spans, 512, a) && a >= pl.zone_off &&
             a + 512 <= pl.pool_bytes && a % pp::align == 0 && !meets(a, a + 512, spans);
        // no legal slot-started place scores less
        double best = 0;
        for (size_t j = 0; j < pl.zone.size(); ++j) {
            best += a < pl.zone[j].off + pl.zone[j].bytes && pl.zone[j].off < a + 512 ? sc[j] : 0.0;
        }
        for (const auto & z : pl.zone) {
            const int64_t c = z.off;
            if (c + 512 > pl.pool_bytes || meets(c, c + 512, spans)) {
                continue;
            }
            double s2 = 0;
            for (size_t j = 0; j < pl.zone.size(); ++j) {
                s2 += c < pl.zone[j].off + pl.zone[j].bytes && pl.zone[j].off < c + 512 ? sc[j] : 0.0;
            }
            ok = ok && s2 >= best;
        }
    }
    expect(ok, "ring placement by usage: as place_ring with no usage, off a hot slot, never on a span, no legal "
               "place scores less (200 random usages)");
    // groups partition the tasks: every task of ep_group's list in exactly one group's [off[e_lo], off[e_hi]), and the
    // group's grid bound covers its tasks, for 40-pair and 8-pair tasks
    ok = true;
    std::vector<pp::group> g;
    std::string err;
    for (int32_t t = 0; t < 300 && ok; ++t) {
        std::vector<int32_t> pairs(512, 0);
        const int32_t rows = 1 + int32_t(rng() % 1536);
        for (int32_t p = 0; p < rows * pp::n_used; ++p) {
            ++pairs[size_t(rng() % (t % 2 == 0 ? 512 : 37))]; // spread and concentrated routes
        }
        std::vector<uint8_t> m(512);
        for (int32_t e = 0; e < 512; ++e) {
            m[size_t(e)] = pairs[size_t(e)] > 0 && rng() % 3 != 0 ? 1 : 0;
        }
        for (const int32_t tr : {40, 8}) {
            const std::vector<int32_t> tasks = pp::task_list(pairs, tr);
            const std::vector<int32_t> off = pp::task_offsets(tasks, 512);
            ok = ok && pp::plan_groups(m.data(), 512, 2329600, int64_t(64) << 20, g, err);
            std::vector<int32_t> seen(tasks.size(), 0);
            for (const auto & x : g) {
                int64_t gp = 0;
                for (int32_t e = x.e_lo; e < x.e_hi; ++e) {
                    gp += pairs[size_t(e)];
                }
                const int32_t n = off[size_t(x.e_hi)] - off[size_t(x.e_lo)];
                ok = ok && n >= 0 && n <= x.e_hi - x.e_lo + (gp + tr - 1) / tr;
                for (int32_t k = off[size_t(x.e_lo)]; k < off[size_t(x.e_hi)]; ++k) {
                    ++seen[size_t(k)];
                    ok = ok && tasks[size_t(k)] >= x.e_lo && tasks[size_t(k)] < x.e_hi;
                }
            }
            for (const int32_t v : seen) {
                ok = ok && v == 1;
            }
        }
    }
    expect(ok, "groups partition ep_group's tasks: each task in exactly one group, of its experts, within the group's "
               "grid bound (300 random routings, 40- and 8-pair tasks)");
}

void check_parity() {
    // the parity buffers by usage. A zone of 300 slots of 256 bytes from 2,560, the last 40 holding layer j % 4 (10 of
    // 64 experts a layer resident), a span at slot 100. With no usage: the highest place, which borrows those 40 and
    // grows the need to whole layers (the fixed point); with those residents hot: below them, borrowing no resident.
    // Either way off the span and every staging prefix inside the borrowed slots; with no room, none
    loan::plan_input in;
    const int64_t zone_lo = 2560, zone_hi = zone_lo + 256 * 300;
    for (int32_t j = 0; j < 300; ++j) {
        in.zone.push_back({zone_lo + 256 * int64_t(j), 256, j >= 260 ? j % 4 : -1});
    }
    in.spans = {{zone_lo + 256 * 100, zone_lo + 256 * 101}};
    in.resident.assign(4, 10);
    in.dentry.assign(4, 256);
    in.n_ex = 64;
    in.n_slots = 310;
    in.lend_pct = 100;
    in.min_keep = 0;
    std::vector<double> sc(in.zone.size(), 0.0);
    int64_t so[2] = {0, 0};
    loan::plan_output out;
    std::string err;
    const auto inside = [&](const loan::plan_output & o, const int64_t s0[2]) {
        bool good = s0[1] >= s0[0] + o.stage_need[0];
        for (int32_t b = 0; b < 2; ++b) {
            good = good && !meets(s0[b], s0[b] + o.stage_need[b], in.spans);
            for (size_t j = 0; j < in.zone.size(); ++j) {
                const bool m = in.zone[j].off < s0[b] + o.stage_need[b] && s0[b] < in.zone[j].off + in.zone[j].bytes;
                good = good && (!m || o.take[j] != 0);
            }
        }
        return good;
    };
    bool ok = pp::place_parity(in, sc, zone_lo, zone_hi, so, out, err) && inside(out, so) &&
              so[1] + out.stage_need[1] == zone_hi && out.taken_residents == 40 && out.stage_need[0] == 64 * 256;
    for (size_t j = 0; j < in.zone.size(); ++j) {
        sc[j] = in.zone[j].layer >= 0 ? 9.0 : 0.0;
    }
    ok = ok && pp::place_parity(in, sc, zone_lo, zone_hi, so, out, err) && inside(out, so) &&
         out.taken_residents == 0 && out.stage_need[0] == 54 * 256;
    const bool none = !pp::place_parity(in, sc, zone_lo, zone_lo + 512, so, out, err) && err.empty();
    expect(ok && none, "parity buffers by usage: the highest place with no usage (the fixed point growing to whole "
                       "layers), below hot residents borrowing none, off the span, staging inside the borrowed "
                       "slots, none without room");
}

void check_cpu_share() {
    // MX7: the prompt CPU share's host decisions. 16 experts: 0 resident (3 pairs), 2 unrouted, 3 over MAXT (9 pairs);
    // the routed nonresident are 1, 3, 4, 5, 6, 7, 8 (7 of them), the candidates (pairs, expert) (1,4) (1,8) (2,1)
    // (2,5) (5,7) (8,6)
    const int32_t counts[16] = {3, 2, 0, 9, 1, 2, 8, 5, 1, 0, 0, 0, 0, 0, 0, 0};
    uint8_t resident[16] = {};
    resident[0] = 1;
    uint8_t cpu[16];
    int32_t nonres = 0, pairs = 0;
    const auto taken = [&](std::vector<int32_t> want) {
        for (int32_t e = 0; e < 16; ++e) {
            const bool in = std::find(want.begin(), want.end(), e) != want.end();
            if ((cpu[e] != 0) != in) {
                return false;
            }
        }
        return true;
    };
    // 0.5 x 7 = 3.5, rounded to 4 (llround: half away from zero): the four fewest, ties by ID
    bool ok = pp::share_pick(counts, resident, 16, pp::share_maxt, 0.5, 4096, cpu, nonres, pairs) == 4 && nonres == 7 &&
              pairs == 6 && taken({4, 8, 1, 5});
    // 1: capped at the candidates (6, not 7); the over-MAXT and resident experts never taken
    ok = ok && pp::share_pick(counts, resident, 16, pp::share_maxt, 1.0, 4096, cpu, nonres, pairs) == 6 &&
         pairs == 19 && taken({4, 8, 1, 5, 7, 6});
    // the pair bound stops the walk: 1 + 1 + 2 = 4 pairs, the next (2) would pass 5
    ok = ok && pp::share_pick(counts, resident, 16, pp::share_maxt, 0.5, 5, cpu, nonres, pairs) == 3 && pairs == 4 &&
         taken({4, 8, 1});
    // rounding: 0.07 x 7 = 0.49 takes none, 0.0715 x 7 = 0.5005 one; 0 none
    ok = ok && pp::share_pick(counts, resident, 16, pp::share_maxt, 0.07, 4096, cpu, nonres, pairs) == 0 && taken({}) &&
         pp::share_pick(counts, resident, 16, pp::share_maxt, 0.0715, 4096, cpu, nonres, pairs) == 1 && taken({4}) &&
         pp::share_pick(counts, resident, 16, pp::share_maxt, 0.0, 4096, cpu, nonres, pairs) == 0;
    // a smaller MAXT: only the experts of at most 2 pairs are candidates
    ok = ok && pp::share_pick(counts, resident, 16, 2, 1.0, 4096, cpu, nonres, pairs) == 4 && taken({4, 8, 1, 5});
    expect(ok, "MX7 CPU share pick: routed nonresident experts of at most MAXT pairs, fewest first (ties by ID), "
               "llround(share x routed nonresident) capped at the candidates, the pair bound, share 0 none");

    // auto's share: g / (c + g) clamped to [0.05, 0.9]; unmeasured 0.5. The running mean: the first sample, then 0.25
    double m = 0;
    int64_t n = 0;
    pp::share_mean(m, n, 4.0);
    const bool first = m == 4.0 && n == 1;
    pp::share_mean(m, n, 8.0);
    ok = first && m == 5.0 && n == 2 && pp::share_of(0, 1) == 0.5 && pp::share_of(1, 0) == 0.5 &&
         pp::share_of(1, 1) == 0.5 && pp::share_of(3, 1) == 0.25 && pp::share_of(1, 100) == pp::share_hi &&
         pp::share_of(100, 1) == pp::share_lo;
    expect(ok, "MX7 CPU share auto: g / (c + g) clamped to [0.05, 0.9], 0.5 before both means, the running means' "
               "first sample and alpha 0.25");

    // the self-check. Measured at once (cost on 1, off 2): on, off alternating until three a side, then on (cheaper)
    // with every 29th post-warm-up layer off; measured a layer late (as the runtime reads them) the same sequence. Then
    // the costs reversed: the first on layer after the warm-up moves on's mean past off's, and off is preferred
    std::string seq;
    pp::share_check sc;
    for (int32_t i = 0; i < 6 + 58; ++i) {
        const bool on = sc.next();
        seq += on ? '1' : '0';
        sc.record(on, on ? 1.0 : 2.0);
    }
    std::string want = "101010";
    for (int32_t i = 1; i <= 58; ++i) {
        want += i % pp::share_probe == 0 ? '0' : '1';
    }
    ok = seq == want && sc.warm() && sc.pref_on() && sc.probes() == 2 && sc.chose_on() == 56 && sc.chose_off() == 0;
    pp::share_check late;
    std::string lseq;
    bool prev = false, have = false;
    for (int32_t i = 0; i < 8; ++i) {
        if (have) {
            late.record(prev, prev ? 1.0 : 2.0);
        }
        prev = late.next();
        have = true;
        lseq += prev ? '1' : '0';
    }
    ok = ok && lseq == "10101011";
    // reversed: on now costs 10. Each on layer (preferred, then probes) moves its mean toward 10 until it passes off's
    pp::share_check rv;
    for (int32_t i = 0; i < 6; ++i) {
        const bool on = rv.next();
        rv.record(on, on ? 1.0 : 2.0);
    }
    int32_t flip = -1;
    for (int32_t i = 0; i < 40 && flip < 0; ++i) {
        const bool on = rv.next();
        rv.record(on, on ? 10.0 : 2.0);
        if (!rv.pref_on()) {
            flip = i;
        }
    }
    // on's mean from 1 toward 10 by 0.25: 3.25 (> 2) after the first; then off, its probe the 29th post-warm-up layer
    bool after = true;
    for (int32_t i = 0; i < 20; ++i) {
        const bool on = rv.next();
        after = after && (on == ((i + 2) % pp::share_probe == 0));
        rv.record(on, on ? 10.0 : 2.0);
    }
    ok = ok && flip == 0 && !rv.pref_on() && after;
    expect(ok, "MX7 CPU share self-check: on/off alternating until three measured a side (also measured a layer "
               "late), then the cheaper side with every 29th layer the other, following a reversal of the costs");
}

} // namespace

bool prompt_plan_test() {
    n_fail = n_pass = 0;
    check_chunks();
    check_paths();
    check_ring_and_loan();
    check_groups_and_ring();
    check_switch();
    check_mx3_plans();
    check_small_ring();
    check_scored_and_tasks();
    check_parity();
    check_refill_ownership();
    check_cpu_share();
    std::printf("ex-cpu-test: me4 prompt planner: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0;
}

} // namespace fn::runtime
