// fnfl-drive state-test mu1 (MU1, no model data): the multi-slot engine's memory and frame isolation on both cards.
//
// Part A, no model: the KV pool (fn/model/kv_pool.h) at the full capacity, 262,144 cells in 16 pages, four slots:
// one slot maps every page and reaches its last logical cell in every per-cell part; a second slot's reservation then
// fails with nothing taken; the peer card reads a mapped range; a trimmed slot keeps its kept pages' bytes; a slot
// grown across a page boundary reads zeros on its new pages (a physical page another slot held is zeroed when mapped)
// and its writes on both sides of the boundary; a mapping that fails part way on card 1 leaves the slot unmapped and
// the free list whole, and the retry maps; every slot's range is the whole capacity (not divided by four). Nonzero
// seeded bytes everywhere, so a zero cannot hide an alias.
//
// Part B, zero weights (model-dry's load: no model data on the cards), four slots at 65,536 cells: each slot's frame
// (its GDN state, histories, logits, sampler, retained carry and KV range) at distinct addresses, with its own
// graphs; a slot's reset, truncate and erase leave the other slots' seeded state, ledgers and committed versions as
// they were; a request option set on one slot does not finish another's request; a read cancelled before its first
// chunk keeps nothing new and one cancelled during a chunk finishes it and keeps it (BG1), then resumes from its
// checkpoint; a short read runs in the other-reader lane while a long read's chunk is suspended in the main arena;
// a full pool reclaims the least recently released idle slot and never an active one.
//
// KV8 (--kv-format q8): part A at the q8 geometry, the slots' 262,144-cell ranges over a 409,600-cell pool of 25
// pages (a second slot then takes the 9 pages the first leaves before the pool is full), part B's engine in q8 (its
// pool a slot's 65,536 cells, as in BF16).
//
// FMU2, state-test mu1 joint: part B's engine on graphs at d = 4, then step_joint's cases alone (part_b_fmu2);
// joint-eager the same eagerly (the joint rounds enqueued in phases); joint-san eagerly, D1 and one four-member round
// (Compute Sanitizer's run). FMU4 (D9): then four sampled slots' joint rounds (joint-san: one) against each slot's lone
// run, tokens and ledgers.
//
// usage: fnfl-drive state-test mu1 [graph|eager|read|joint|joint-eager|joint-san] [DEV0] [DEV1] [--kv-format q8]
//
// MXMU1, state-test mu1-one [graph|eager|c0] DEV (zero weights, one card, the expert cache over its zero backing, the
// host-backed KV; Config D's cache settings): part C0, no allocation: the refusals (three slots on one card, two
// without the host-backed KV, cached TP, two-card host KV, a host that cannot hold the slots' histories). Part C,
// two slots at 65,536 cells: each slot's frame, host history and device window at distinct addresses with its own
// graphs; seeded host history and window bytes kept by the peer's prompt, decode and reset; slot switches both ways
// mid-decode and after a prompt; every read step returns the prompt loan (no hold, no loan job) and decode runs
// between a long read's chunks; ME3's short read in one step; a chunk suspended under test_partial_reads holds the
// loan and the peer's step is refused until its cancel finishes the chunk; an abandoned suspended chunk returns its
// loan and its slot re-reads; erase and truncate on one slot leave the other's tokens and history; FMU2 Amendment 3
// (C11): two slots through engine::step_joint step alone, each slot's steps, tokens and ledger its engine::step run's.

#include "fn/kernels/round/version.h"
#include "fn/model/kv_pool.h"
#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/model/state.h"
#include "fn/loader/run_load.h"
#include "fn/runtime/expert_cache.h"
#include "fn/runtime/joint.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/target_cmds.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace md = fn::model;
namespace rt = fn::runtime;

int g_fail = 0;

bool put(const char * name, bool ok, const std::string & detail = std::string()) {
    printf("state-test mu1: %s: %s%s%s\n", name, ok ? "PASS" : "FAIL", detail.empty() ? "" : " (",
           detail.empty() ? "" : (detail + ")").c_str());
    g_fail += ok ? 0 : 1;
    return ok;
}

bool cu(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "state-test mu1: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

std::vector<uint8_t> pattern(uint64_t seed, size_t n) {
    std::vector<uint8_t> v(n);
    uint64_t z = seed * 0x9e3779b97f4a7c15ull + 1;
    for (size_t i = 0; i < n; ++i) {
        z ^= z << 13;
        z ^= z >> 7;
        z ^= z << 17;
        v[i] = static_cast<uint8_t>(z | 1); // never zero
    }
    return v;
}

bool write_at(int32_t dev, uint8_t * p, const std::vector<uint8_t> & v) {
    return cu(cudaSetDevice(dev), "device") && cu(cudaMemcpy(p, v.data(), v.size(), cudaMemcpyHostToDevice), "write");
}

bool read_at(int32_t dev, const uint8_t * p, std::vector<uint8_t> & v, size_t n) {
    v.assign(n, 0);
    return cu(cudaSetDevice(dev), "device") && cu(cudaMemcpy(v.data(), p, n, cudaMemcpyDeviceToHost), "read");
}

bool all_zero(const std::vector<uint8_t> & v) {
    for (const uint8_t b : v) {
        if (b != 0) {
            return false;
        }
    }
    return true;
}

// ---- part A: the pool alone ----

struct pooled_part {
    std::string cls;
    int32_t part = 0;
    int64_t off = 0, bytes = 0, slice = 0;
};

bool part_a(int32_t dev0, int32_t dev1, bool q8) {
    md::model m;
    m.device[0] = dev0;
    m.device[1] = dev1;
    // np: a slot's logical pages; phys: the pool's (KV8: 25 pages of the 409,600-cell candidate)
    const int32_t cells = 262144, pc = md::kv_page_cells, np = cells / pc, pool_cells = q8 ? 409600 : cells,
                  phys = pool_cells / pc;
    md::state_params sp;
    sp.n_cells = cells;
    sp.ple_conv_kernel = 4;
    sp.ple_ngram = 3;
    sp.kv_q8 = q8;
    md::state_params hp = sp;
    hp.n_cells = cells / 2;
    std::vector<md::buffer_class> full, half;
    const std::vector<md::buffer_class> a = md::state_classes(sp), b = md::state_classes(hp);
    for (size_t i = 0; i < a.size(); ++i) {
        if (md::pooled_class(a[i].name)) {
            full.push_back(a[i]);
            half.push_back(b[i]);
        }
    }
    md::kv_pool pool;
    std::string err;
    if (!put(q8 ? "A pool init (q8: 25 pages a card, 4 slots of 16 logical pages)"
                : "A pool init (16 pages a card, 4 slots)",
             pool.init(m, full, half, 4, cells, pc, err, pool_cells), err)) {
        return false;
    }
    for (const std::string & l : pool.ledger()) {
        printf("state-test mu1: %s\n", l.c_str());
    }
    // the per-cell parts on each card
    std::vector<pooled_part> parts[2];
    for (size_t k = 0; k < full.size(); ++k) {
        for (int32_t c = 0; c < 2; ++c) {
            for (int32_t p = 0; p < static_cast<int32_t>(full[k].parts.size()); ++p) {
                const int64_t bf = full[k].parts[size_t(p)].bytes[c], bh = half[k].parts[size_t(p)].bytes[c];
                if (bf > 0 && bf == 2 * bh) {
                    parts[c].push_back({full[k].name, p, full[k].offset(p, c), bf, bf / np});
                }
            }
        }
    }
    // A1: every slot's range is the whole capacity on both cards, at distinct addresses
    bool whole = true;
    for (size_t k = 0; k < full.size(); ++k) {
        for (int32_t c = 0; c < 2; ++c) {
            for (int32_t s = 0; s < 4; ++s) {
                whole = whole && pool.range_bytes(full[k].name, c) >= full[k].allocation(c) &&
                        pool.base(s, full[k].name, c) != nullptr;
                for (int32_t t = 0; t < s; ++t) {
                    whole = whole && pool.base(s, full[k].name, c) != pool.base(t, full[k].name, c);
                }
            }
        }
    }
    put("A every slot's range holds the whole 262,144 cells (not a quarter), distinct a slot", whole);
    // A2: slot 0 maps all 16 pages and reaches its last cell, cell 0 and every page boundary of every part
    bool ok = pool.ensure(0, cells, err);
    bool reach = ok && pool.pages().mapped(0) == np && pool.pages().free_pages() == phys - np;
    int64_t written = 0;
    for (int32_t c = 0; c < 2 && reach; ++c) {
        const int32_t dev = m.device[c];
        for (const pooled_part & pp : parts[c]) {
            uint8_t * base = pool.base(0, pp.cls, c) + pp.off;
            std::vector<int64_t> at = {0, pp.bytes - 64};
            for (int32_t j = 1; j < np; ++j) {
                at.push_back(int64_t(j) * pp.slice - 32);
            }
            for (const int64_t o : at) {
                const std::vector<uint8_t> v = pattern(uint64_t(o) ^ (uint64_t(c) << 40) ^ uint64_t(pp.part), 64);
                std::vector<uint8_t> got;
                reach = reach && write_at(dev, base + o, v) && read_at(dev, base + o, got, 64) && got == v;
                ++written;
            }
        }
    }
    put("A one slot maps all 16 pages: the last logical cell, cell 0 and every page boundary of every part", reach,
        std::to_string(written) + " ranges on both cards" + (ok ? "" : ": " + err));
    if (q8) {
        // KV8: the pool beyond one slot's range: slot 3 takes the 9 pages slot 0 leaves and reaches its last cell
        // and every page boundary there (physical pages 16 .. 24), and the pool is then full
        ok = pool.ensure(3, int64_t(phys - np) * pc, err);
        bool beyond = ok && pool.pages().mapped(3) == phys - np && pool.pages().free_pages() == 0 &&
                      pool.pages().physical(3, 0) == np && pool.pages().physical(3, phys - np - 1) == phys - 1;
        for (int32_t c = 0; c < 2 && beyond; ++c) {
            for (const pooled_part & pp : parts[c]) {
                uint8_t * base = pool.base(3, pp.cls, c) + pp.off;
                for (int32_t j = 1; j <= phys - np && beyond; ++j) {
                    const int64_t o = int64_t(j) * pp.slice - 32;
                    const std::vector<uint8_t> v = pattern(0x7700 + uint64_t(o) + uint64_t(c) * 3, 32);
                    std::vector<uint8_t> got;
                    beyond = write_at(m.device[c], base + o, v) && read_at(m.device[c], base + o, got, 32) && got == v;
                }
            }
        }
        put("A q8: a second slot takes the pool's 9 pages past one slot's range (cells 262,144 .. 409,599 of the "
            "pool) and reaches each; the pool is then full",
            beyond, ok ? "" : err);
    }
    // A3: a second slot's reservation fails with nothing taken
    bool full_pool = false;
    std::string e3;
    const bool refused = !pool.ensure(1, 1, e3, &full_pool);
    put("A a full pool refuses another slot's page, nothing taken",
        refused && full_pool && pool.pages().mapped(1) == 0 && pool.pages().free_pages() == 0, e3);
    // A4: the peer card reads card 0's mapped range
    {
        const pooled_part & pp = parts[0].front();
        uint8_t * src = pool.base(0, pp.cls, 0) + pp.off + pp.bytes - 64;
        void * dst = nullptr;
        std::vector<uint8_t> want, got(64, 0);
        bool p = read_at(m.device[0], src, want, 64) && cu(cudaSetDevice(m.device[1]), "device") &&
                 cu(cudaMalloc(&dst, 64), "peer buffer") &&
                 cu(cudaMemcpyPeer(dst, m.device[1], src, m.device[0], 64), "peer copy") &&
                 cu(cudaMemcpy(got.data(), dst, 64, cudaMemcpyDeviceToHost), "peer read") && got == want;
        if (dst != nullptr) {
            cudaFree(dst);
        }
        put("A the peer card reads a mapped range (access granted to both)", p);
    }
    // A5: trimmed to 3 pages, the kept pages' bytes stay (KV8: slot 3's pages returned first)
    ok = (!q8 || pool.trim(3, 0, err)) && pool.trim(0, int64_t(3) * pc, err);
    bool kept = ok && pool.pages().mapped(0) == 3 && pool.pages().free_pages() == phys - 3;
    for (int32_t c = 0; c < 2 && kept; ++c) {
        for (const pooled_part & pp : parts[c]) {
            const int64_t o = int64_t(2) * pp.slice - 32; // the boundary of pages 1 and 2
            std::vector<uint8_t> got;
            kept = kept && read_at(m.device[c], pool.base(0, pp.cls, c) + pp.off + o, got, 64) &&
                   got == pattern(uint64_t(o) ^ (uint64_t(c) << 40) ^ uint64_t(pp.part), 64);
        }
    }
    put(("A a trim to 3 pages frees the rest (" + std::to_string(phys - 3) + " free) and keeps the kept pages' bytes")
            .c_str(),
        kept, ok ? "" : err);
    // A6: a slot grown across a page boundary: its new pages read zero (pages 3 and 4, which slot 0 wrote), then its
    // writes on both sides of the boundary
    ok = pool.ensure(1, int64_t(pc) + 1, err);
    bool grow =
        ok && pool.pages().mapped(1) == 2 && pool.pages().physical(1, 0) == 3 && pool.pages().physical(1, 1) == 4;
    for (int32_t c = 0; c < 2 && grow; ++c) {
        for (const pooled_part & pp : parts[c]) {
            uint8_t * base = pool.base(1, pp.cls, c) + pp.off;
            std::vector<uint8_t> z0, z1, got;
            grow = grow && read_at(m.device[c], base, z0, 64) && all_zero(z0) &&
                   read_at(m.device[c], base + pp.slice - 32, z1, 64) && all_zero(z1);
            const std::vector<uint8_t> v = pattern(0x5100 + uint64_t(c) * 7 + uint64_t(pp.part), 64);
            grow = grow && write_at(m.device[c], base + pp.slice - 32, v) &&
                   read_at(m.device[c], base + pp.slice - 32, got, 64) && got == v;
        }
    }
    put("A a slot grown across a page boundary: a page another slot wrote reads zero, writes across the boundary hold",
        grow, ok ? "" : err);
    // A7: a mapping that fails part way on card 1 leaves the slot unmapped and the free list whole; the retry maps
    const int32_t free_before = pool.pages().free_pages();
    pool.test_fail_map(1, 7);
    std::string e7;
    const bool failed = !pool.ensure(2, int64_t(2) * pc, e7);
    const bool undone = failed && pool.pages().mapped(2) == 0 && pool.pages().free_pages() == free_before;
    ok = pool.ensure(2, int64_t(2) * pc, err);
    bool retry = ok && pool.pages().mapped(2) == 2;
    for (int32_t c = 0; c < 2 && retry; ++c) {
        for (const pooled_part & pp : parts[c]) {
            std::vector<uint8_t> z;
            retry = retry && read_at(m.device[c], pool.base(2, pp.cls, c) + pp.off + pp.slice, z, 64) && all_zero(z);
        }
    }
    put("A a both-card mapping failing part way on card 1 is undone; the retry maps zeroed pages", undone && retry,
        e7 + (ok ? "" : "; retry: " + err));
    // A8: every page back
    ok = pool.trim(0, 0, err) && pool.trim(1, 0, err) && pool.trim(2, 0, err);
    put("A every slot's pages back to the pool", ok && pool.pages().free_pages() == phys, ok ? "" : err);
    return true;
}

// ---- part B: a zero-weight engine of four slots ----

struct seeded {
    uint8_t * p;
    int32_t dev;
    std::vector<uint8_t> v;
};

bool same_bytes(const seeded & s) {
    std::vector<uint8_t> got;
    return read_at(s.dev, s.p, got, s.v.size()) && got == s.v;
}

// FMU1 (B9..B12): the presence-based read paths on the four-slot engine (resident TP): lone reads on every slot, a peer
// arriving mid-chunk (writers' 1-layer groups, the chunk's end and finish their own steps, then lone), P's staging of
// the reader to follow (its tags dropped when the read is cancelled and the slot reads again) and the other-reader lane
// taking a staged chunk on the cards
void part_b_fmu1(rt::engine & e, const rt::request_options & ro, const std::vector<int32_t> & p300,
                 const std::vector<int32_t> & p9000) {
    rt::prefiller & pf = e.prefill_driver();
    int32_t np = 0, pend = -1;
    for (int32_t s = 0; s < 4; ++s) {
        e.release_slot(s);
    }
    put("B FMU1 the four-slot pooled two-card engine is resident TP", e.resident_tp());
    const auto steps = [&](int32_t s, int32_t n) {
        int32_t out[rt::max_step_tokens];
        int32_t n_out = 0;
        bool end = false;
        for (int32_t i = 0; i < n; ++i) {
            if (e.step(s, out, n_out, end) != rt::run_status::ok) {
                return false;
            }
        }
        return true;
    };
    const auto begin = [&](int32_t s, const std::vector<int32_t> & tok) {
        return e.set_sampling(s, ro) == rt::run_status::ok && e.truncate(s, 0) == rt::run_status::ok &&
               e.read_begin(s, tok.data(), int32_t(tok.size()), 0, np) == 0;
    };
    const auto info = [&](int32_t s) {
        rt::read_info ri;
        e.get_read_info(s, ri);
        return ri;
    };
    const auto history_is = [&](int32_t s, const std::vector<int32_t> & tok) {
        std::vector<int32_t> h;
        return e.history(s, h, pend) == rt::run_status::ok && h == tok;
    };
    // stage the job's chunk (P never waits: the staging thread finishes, a later call uploads and tags it)
    const auto stage = [&](int32_t now, int32_t next, int64_t job, int64_t chunk) {
        for (int32_t i = 0; i < 20000 && pf.staged(job, chunk) < 0; ++i) {
            if (!e.read_prefetch(now, next)) {
                return false;
            }
            if (pf.staged(job, chunk) < 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
            }
        }
        return pf.staged(job, chunk) >= 0;
    };
    // B9: a lone read on every slot: each chunk in one step of one 48-layer group, its finish in the same step
    bool ok = true;
    std::string got;
    for (int32_t s = 0; s < 4; ++s) {
        bool one = begin(s, p9000);
        const rt::read_status a = one ? e.read_step(s, 1, rt::read_mode::lone) : rt::read_status::error;
        const rt::read_info r0 = info(s);
        const rt::read_status b = one ? e.read_step(s, 1, rt::read_mode::lone) : rt::read_status::error;
        const rt::read_info r1 = info(s);
        one = one && a == rt::read_status::chunk_done && r0.layers == 48 && r0.groups_done == 1 &&
              b == rt::read_status::done && r1.groups_done == 2 && r1.steps == 2 && history_is(s, p9000) &&
              steps(s, 2) && e.release_slot(s) == rt::run_status::ok;
        got += (got.empty() ? "" : ", ") + std::string("slot ") + std::to_string(s) + (one ? " ok" : " FAIL");
        ok = ok && one;
    }
    put("B FMU1 a lone read on every slot: one step and one 48-layer group a chunk, then steps", ok,
        got + "; " + e.error());
    // B10: a writer generates while slot 1 reads: 1-layer groups, the chunk's end group and its finish steps of their
    // own (the writer's round between every two); the next chunk starts with writers, then runs lone once they leave
    ok = begin(0, p300);
    rt::read_status st = rt::read_status::yielded;
    while (ok && st != rt::read_status::done) {
        st = e.read_step(0, 48);
        ok = st == rt::read_status::yielded || st == rt::read_status::chunk_done || st == rt::read_status::done;
    }
    ok = ok && steps(0, 1) && begin(1, p9000);
    int32_t n0 = 0;
    st = rt::read_status::yielded;
    rt::read_info w0;
    while (ok && st == rt::read_status::yielded) {
        st = e.read_step(1, 1, rt::read_mode::writers);
        if (++n0 == 1) {
            w0 = info(1);
        }
        ok = steps(0, 1);
    }
    const rt::read_info w1 = info(1);
    const bool chunk0 = ok && st == rt::read_status::chunk_done && w0.layers == 1 && w0.groups == 49 && n0 == 50 &&
                        w1.groups_done == 49 && w1.committed == 8192;
    st = ok ? e.read_step(1, 1, rt::read_mode::writers) : rt::read_status::error;
    ok = ok && st == rt::read_status::yielded && steps(0, 1) &&
         e.read_step(1, 1, rt::read_mode::writers) == rt::read_status::yielded;
    const rt::read_info w2 = info(1);
    st = ok ? e.read_step(1, 1, rt::read_mode::lone) : rt::read_status::error;
    const rt::read_info w3 = info(1);
    const bool chunk1 = ok && w2.layers == 1 && w2.groups_done == 51 && st == rt::read_status::done &&
                        w3.groups_done == 49 + 49 && history_is(1, p9000) && steps(1, 1) && steps(0, 1);
    put("B FMU1 writers beside a reader: 1-layer groups, the chunk's end and finish their own steps; lone once they "
        "leave",
        chunk0 && chunk1,
        "chunk 0 steps " + std::to_string(n0) + ", groups " + std::to_string(w1.groups_done) + "; chunk 1 groups " +
            std::to_string(w3.groups_done) + "; " + e.error());
    e.release_slot(0);
    e.release_slot(1);
    // B10b (FMU2's reader rule): a writers-mode step of five 1-layer groups (main's group beside writers) stops before
    // the chunk's end group: nine steps of five layers, one of the last three, the end group's own step, the finish
    ok = begin(0, p300);
    st = rt::read_status::yielded;
    while (ok && st != rt::read_status::done) {
        st = e.read_step(0, 48);
        ok = st == rt::read_status::yielded || st == rt::read_status::chunk_done || st == rt::read_status::done;
    }
    ok = ok && steps(0, 1) && begin(1, p9000);
    std::string seq;
    n0 = 0;
    st = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded && n0 < 64) {
        st = e.read_step(1, 5, rt::read_mode::writers);
        ++n0;
        seq += (seq.empty() ? "" : ",") + std::to_string(info(1).groups_done);
        ok = steps(0, 1);
    }
    const bool five =
        ok && st == rt::read_status::chunk_done && n0 == 12 && seq == "5,10,15,20,25,30,35,40,45,48,49,49";
    st = rt::read_status::yielded;
    while (five && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
        st = e.read_step(1, 1, rt::read_mode::lone);
    }
    put("B FMU2 writers beside a reader at five groups a step: the chunk's end group and finish stay their own steps",
        five && st == rt::read_status::done && history_is(1, p9000) && steps(0, 1),
        "groups done after each step " + seq + "; " + e.error());
    e.release_slot(0);
    e.release_slot(1);
    // B11: P stages the chunk of the reader to follow; a read cancelled before its chunk and its slot's next read drop
    // its staging; a staged chunk is the one its read starts with
    std::vector<int32_t> q9000(p9000.size());
    for (size_t i = 0; i < q9000.size(); ++i) {
        q9000[i] = int32_t(5000 + (i * 11) % 3001);
    }
    ok = begin(2, p9000) && begin(3, q9000);
    const int64_t b_old = info(3).job;
    const bool staged0 = ok && stage(2, 3, b_old, 0);
    e.read_cancel(3);
    ok = ok && e.read_step(3, 1) == rt::read_status::cancelled && begin(3, p9000);
    const int64_t b_new = info(3).job;
    const bool dropped = ok && pf.staged(b_old, 0) < 0 && pf.gather_job() != b_old && b_new != b_old;
    st = rt::read_status::yielded;
    while (ok && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
        ok = e.read_prefetch(2, 3);
        st = e.read_step(2, 1, rt::read_mode::lone);
    }
    ok = ok && st == rt::read_status::done && history_is(2, p9000);
    const bool staged1 = ok && stage(3, 3, b_new, 0);
    st = rt::read_status::yielded;
    while (ok && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
        ok = e.read_prefetch(3, 3);
        st = e.read_step(3, 1, rt::read_mode::lone);
    }
    ok = ok && st == rt::read_status::done && history_is(3, p9000) && steps(2, 1) && steps(3, 1);
    put("B FMU1 P stages the next reader's chunk; a cancelled read's staging is dropped when its slot reads again; "
        "both reads then complete",
        staged0 && dropped && staged1 && ok,
        std::string(staged0 ? "staged" : "not staged") + ", " + (dropped ? "dropped" : "not dropped") + ", " +
            (staged1 ? "staged" : "not staged") + "; " + e.error());
    e.release_slot(2);
    e.release_slot(3);
    // B12: a short read the main lane's staging holds runs in the other-reader lane (a copy on the cards) while a long
    // read's chunk is suspended in the main arena
    ok = begin(0, p9000) && e.read_step(0, 1, rt::read_mode::writers) == rt::read_status::yielded && begin(1, p300);
    const int64_t short_job = info(1).job;
    const bool staged2 = ok && stage(0, 1, short_job, 0);
    st = rt::read_status::yielded;
    while (staged2 && st == rt::read_status::yielded) {
        st = e.read_step(1, 4);
    }
    const bool lane1 = staged2 && st == rt::read_status::done && pf.staged(short_job, 0) < 0 && e.lane_owner(1) < 0 &&
                       e.lane_owner(0) == 0 && history_is(1, p300);
    st = rt::read_status::yielded;
    while (lane1 && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
        st = e.read_step(0, 1, rt::read_mode::lone);
    }
    put("B FMU1 the other-reader lane takes a chunk P staged in the main lane while a long chunk is suspended",
        lane1 && st == rt::read_status::done && history_is(0, p9000) && steps(0, 1) && steps(1, 1), e.error());
    e.release_slot(0);
    e.release_slot(1);
}

// FMU2 (D1..D8): step_joint on the four-slot engine (resident TP, graphs, d = 4; zero weights, so every proposal is
// accepted and the tokens say nothing of identity: rollback-test mu1 section 7 compares joint against lone on the real
// weights). The plumbing: bootstraps alone, the joint round for two to four members in any lane order, the graph cache
// (a hit, a capture, the least recently used out), a call mixing a member and a slot whose step is alone, a member's
// injected round error (FC-RD2: that slot's step fails and its slot is lost; the others' steps are joint and whole; the
// rings back at their sentinels for the next joint round), a duplicate slot refused, the frame's memory
// eager (Compute Sanitizer's runs): the joint rounds enqueued in phases (no captures); first: D1 and D2's first round
void part_b_fmu2(rt::engine & e, rt::request_options ro, bool eager, bool first) {
    for (int32_t s = 0; s < 4; ++s) {
        e.release_slot(s);
    }
    const rt::joint_frame * jf = e.test_joint();
    put("D FMU2 the four-slot engine has a joint frame",
        jf != nullptr && jf->ready() && jf->eager() == eager && !e.test_joint_off(), e.settings());
    if (jf == nullptr) {
        return;
    }
    std::vector<int32_t> p(300);
    for (size_t i = 0; i < p.size(); ++i) {
        p[i] = int32_t(1000 + i % 977);
    }
    ro.max_tokens = 100000;
    const auto read_as = [&](int32_t s, const rt::request_options & o) {
        int32_t np = 0;
        if (e.set_sampling(s, o) != rt::run_status::ok || e.truncate(s, 0) != rt::run_status::ok ||
            e.read_begin(s, p.data(), int32_t(p.size()), 0, np) != 0) {
            return false;
        }
        rt::read_status st = rt::read_status::yielded;
        while (st == rt::read_status::yielded || st == rt::read_status::chunk_done) {
            st = e.read_step(s, 48);
        }
        return st == rt::read_status::done;
    };
    const auto read = [&](int32_t s) { return read_as(s, ro); };
    // one step_joint call; want: per slot, whether its step was a joint lane (1), alone (0), or failed (-1)
    rt::joint_step_out res[4];
    const auto call = [&](std::vector<int32_t> slots, std::vector<int32_t> want, std::string & got) {
        e.step_joint(slots.data(), int32_t(slots.size()), res);
        bool ok = true;
        got.clear();
        for (size_t k = 0; k < slots.size(); ++k) {
            const int32_t is = res[k].st != rt::run_status::ok ? -1 : res[k].joint ? 1 : 0;
            ok = ok && is == want[k] && (is != 1 || res[k].n_out == 5) && (is < 0 || res[k].n_out >= 1);
            const char * how = is == 1 ? " joint " : is == 0 ? " alone " : " failed ";
            got += (k > 0 ? ", " : "") + std::to_string(slots[k]) + how + std::to_string(res[k].n_out) +
                   (is < 0 ? " (" + res[k].why + ")" : "");
        }
        return ok;
    };
    // a membership's first round captures its graphs (none eager)
    const auto caps = [&](int32_t n) { return jf->captures() == (eager ? 0 : n); };
    // D9 (FMU4): four sampled slots (T = 1, top-k 20, a seed each): the bootstraps alone, then `rounds` joint rounds
    // whose members draft in one joint sampled draft step each; then every slot read again and stepped alone
    // (engine::step) as many times: each slot's emitted tokens and ledger its lone run's
    const auto sampled = [&](int32_t rounds) {
        rt::request_options so = ro;
        so.sampling.temp = 1.0f;
        so.sampling.top_k = 20;
        const auto opts = [&](int32_t s) {
            rt::request_options o = so;
            o.sampling.seed = 500 + uint64_t(s);
            return o;
        };
        std::vector<int32_t> tj[4], tl[4], hj[4], hl[4];
        const int64_t dr0 = jf->draft_rounds(), r0 = jf->rounds();
        const int32_t dc0 = jf->draft_captures();
        bool ok = true, joint = true;
        for (int32_t s = 0; s < 4; ++s) {
            ok = ok && read_as(s, opts(s));
        }
        const int32_t all[4] = {0, 1, 2, 3};
        for (int32_t i = 0; i <= rounds && ok; ++i) {
            e.step_joint(all, 4, res);
            for (int32_t k = 0; k < 4; ++k) {
                ok = ok && res[k].st == rt::run_status::ok && res[k].n_out >= 1;
                joint = joint && res[k].joint == (i > 0);
                tj[k].insert(tj[k].end(), res[k].out, res[k].out + res[k].n_out);
            }
        }
        int32_t pend = -1;
        for (int32_t s = 0; s < 4 && ok; ++s) {
            ok = e.history(s, hj[s], pend) == rt::run_status::ok;
        }
        const int64_t drafts = jf->draft_rounds() - dr0, verifies = jf->rounds() - r0;
        const int32_t dcaps = jf->draft_captures() - dc0;
        for (int32_t s = 0; s < 4 && ok; ++s) {
            ok = read_as(s, opts(s));
            for (int32_t i = 0; i <= rounds && ok; ++i) {
                int32_t out[rt::max_step_tokens];
                int32_t n_out = 0;
                bool end = false;
                ok = e.step(s, out, n_out, end) == rt::run_status::ok && n_out >= 1;
                tl[s].insert(tl[s].end(), out, out + n_out);
            }
            ok = ok && e.history(s, hl[s], pend) == rt::run_status::ok;
        }
        bool same = ok, differ = false;
        std::string sizes;
        for (int32_t s = 0; s < 4; ++s) {
            same = same && tj[s] == tl[s] && hj[s] == hl[s] && hj[s].size() > p.size();
            differ = differ || (s > 0 && tj[s] != tj[0]);
            sizes += (s > 0 ? ", " : "") + std::to_string(tj[s].size()) + "/" + std::to_string(tl[s].size());
        }
        put("D FMU4 four sampled slots (T = 1, top-k 20, a seed each): each joint round's members drafted in one "
            "joint sampled draft step; every slot's tokens and ledger its lone run's",
            ok && joint && same && drafts == rounds && verifies == rounds && dcaps == (eager ? 0 : 1),
            "emitted joint/lone " + sizes + "; joint drafts " + std::to_string(drafts) + ", joint rounds " +
                std::to_string(verifies) + ", draft captures " + std::to_string(dcaps) + "; slots' streams " +
                (differ ? "differ" : "all equal") + (ok ? std::string() : "; " + e.error()));
        for (int32_t s = 0; s < 4; ++s) {
            e.release_slot(s);
        }
    };
    bool ok = read(0) && read(1) && read(2) && read(3);
    std::string got;
    // D1: every slot's first step is its bootstrap: alone
    const bool d1 = ok && call({0, 1, 2, 3}, {0, 0, 0, 0}, got) && caps(0);
    put("D FMU2 bootstraps step alone in a joint call", d1, got);
    if (first) {
        const bool d2 =
            call({0, 1, 2, 3}, {1, 1, 1, 1}, got) && caps(1) && jf->rounds() == 1 && jf->draft_rounds() == 1;
        put("D FMU2 four members in one joint round (five tokens each)", d2,
            got + "; captures " + std::to_string(jf->captures()) + ", rounds " + std::to_string(jf->rounds()) +
                ", joint drafts " + std::to_string(jf->draft_rounds()));
        for (int32_t s = 0; s < 4; ++s) {
            e.release_slot(s);
        }
        sampled(1);
        return;
    }
    // D2: four members, a capture; the same membership again, no capture
    const bool d2 = call({0, 1, 2, 3}, {1, 1, 1, 1}, got) && caps(1) && call({0, 1, 2, 3}, {1, 1, 1, 1}, got) &&
                    caps(1) && jf->rounds() == 2;
    put("D FMU2 four members in one joint round (five tokens each); the membership's graphs reused", d2,
        got + "; captures " + std::to_string(jf->captures()) + ", rounds " + std::to_string(jf->rounds()));
    // D3: another lane order is another membership; two members; the least recently used pair out (cache 2)
    const bool d3 = call({2, 0, 3, 1}, {1, 1, 1, 1}, got) && caps(2) && call({3, 1}, {1, 1}, got) && caps(3) &&
                    call({0, 1, 2, 3}, {1, 1, 1, 1}, got) && caps(4) && call({3, 1}, {1, 1}, got) && caps(4);
    put("D FMU2 lane orders and two-member rounds: a capture each new membership, two pairs kept (least recently used "
        "out)",
        d3, got + "; captures " + std::to_string(jf->captures()));
    // D4: three members beside a slot whose step is its bootstrap (alone)
    ok = read(2);
    const bool d4 = ok && call({0, 1, 2, 3}, {1, 1, 0, 1}, got) && call({2, 3}, {1, 1}, got);
    put("D FMU2 three members beside a bootstrap in one call; the bootstrapped slot a member next", d4, got);
    // D5: an injected round error on a member (card 1's commit): its step fails, its slot is lost; the others whole
    e.test_target(1).test_fail_next_verify(2);
    const bool d5a = call({0, 1, 3}, {1, -1, 1}, got);
    const std::string got5a = got;
    const bool d5b = call({0, 3}, {1, 1}, got) && read(1) && call({0, 1, 2, 3}, {1, 0, 1, 1}, got) &&
                     call({0, 1, 2, 3}, {1, 1, 1, 1}, got);
    put("D FMU2 a member's injected round error: its step fails (FC-RD2), the others' are joint and whole; the next "
        "joint rounds clean; the lost slot reads again and rejoins",
        d5a && d5b && !e.test_joint_off(), got5a + "; then " + got);
    // D6: a duplicate slot refused, the rest stepped; one slot alone
    rt::joint_step_out one[2];
    const int32_t dup[2] = {0, 0};
    e.step_joint(dup, 2, one);
    const bool d6 = one[0].st == rt::run_status::ok && !one[0].joint && one[1].st == rt::run_status::bad_request &&
                    call({3}, {0}, got);
    put("D FMU2 a duplicate slot refused, the first stepped alone; one slot steps alone", d6, got);
    // D7: every slot's ledger the same tokens (zero weights: the same stream on every slot, any lane)
    std::vector<int32_t> h[4];
    int32_t pend = -1;
    bool same = true;
    for (int32_t s = 0; s < 4; ++s) {
        same = same && e.history(s, h[s], pend) == rt::run_status::ok && h[s].size() > p.size();
    }
    put("D FMU2 the slots' ledgers after the joint rounds", same,
        std::to_string(h[0].size()) + ", " + std::to_string(h[1].size()) + ", " + std::to_string(h[2].size()) + ", " +
            std::to_string(h[3].size()) + " tokens");
    // D7b (build item 2): every joint round's members drafted in one joint draft step (all greedy), its graphs a
    // membership's as the verify's (none eager)
    put("D FMU2 every joint round's members drafted in one joint draft step",
        jf->draft_rounds() == jf->rounds() && jf->draft_captures() == (eager ? 0 : jf->captures()),
        "joint drafts " + std::to_string(jf->draft_rounds()) + ", joint rounds " + std::to_string(jf->rounds()) +
            ", draft captures " + std::to_string(jf->draft_captures()));
    sampled(3);
    // D8: the frame's memory (its startup bytes, the kept pairs' measured bytes, free memory after the last capture)
    char line[256];
    snprintf(line, sizeof(line),
             "frame %.1f / %.1f MiB, graphs kept %.1f / %.1f MiB, free after capture %.3f / %.3f GiB",
             double(jf->bytes(0)) / 1048576.0, double(jf->bytes(1)) / 1048576.0, double(jf->graph_bytes(0)) / 1048576.0,
             double(jf->graph_bytes(1)) / 1048576.0, double(jf->free_after_capture(0)) / 1073741824.0,
             double(jf->free_after_capture(1)) / 1073741824.0);
    put("D FMU2 the joint frame's memory", jf->bytes(0) > 0 && jf->bytes(1) > 0, line);
    printf("state-test mu1: D the last joint capture (%d lanes): %lld / %lld graph nodes; slot 0's lone five-row graph "
           "%lld / %lld\n",
           jf->last_lanes(), static_cast<long long>(jf->last_nodes(0)), static_cast<long long>(jf->last_nodes(1)),
           static_cast<long long>(e.test_target(0).graph_node_count(5, 0)),
           static_cast<long long>(e.test_target(0).graph_node_count(5, 1)));
    printf("state-test mu1: D the joint frame a card: rings %lld B, layer_ws at 20 rows %zu B (5 rows %zu B), dense "
           "work %zu B, routing %zu B\n",
           static_cast<long long>(fn::xchg::joint_rings_card_bytes()), md::layer_ws_bytes(fn::xchg::joint_rows),
           md::layer_ws_bytes(md::layer_max_rows), md::joint_dense_work_bytes(), md::joint_route_bytes());
    for (int32_t s = 0; s < 4; ++s) {
        e.release_slot(s);
    }
}

// read_only (FMU1, the sanitizers' run): the engine, then the FMU1 read-job cases alone; joint_only (FMU2): the
// engine on graphs at d = 4, then the joint step cases alone
bool part_b(const fnfl_target_args & a, bool eager, bool read_only, bool joint_only = false, bool joint_one = false) {
    const int32_t cells = 65536;
    auto m = std::make_unique<md::model>();
    fnfl_target_args aa = a;
    aa.n_slots = 4;
    // KV8: a cell's K codes (a.kv_q8) or BF16 K, for the seeded cell 1000
    const int64_t kv_cell = a.kv_q8 ? 256 : 512;
    aa.draft_attention = "sparse"; // the serving setting (MX0's recipe)
    aa.draft_attention_given = true;
    if (!fnfl_load_dry(aa, cells, *m)) {
        return put("B the zero-weight load of four slots", false);
    }
    rt::draft_options dopt;
    if (!fnfl_draft_settings(aa, true, "state-test mu1", dopt)) {
        return put("B the draft settings", false);
    }
    rt::engine_config c;
    c.model_path = a.model_path;
    c.n_ctx_slot = cells;
    c.draft = dopt;
    c.eog = {248046};
    c.timeout_s = joint_one ? 1800.0 : 30.0; // joint-san: a round under racecheck outlasts 30 s
    c.eager = eager;
    if (joint_only) {
        c.spec_d = 4; // the serving d
    }
    rt::engine e;
    std::string err;
    if (!put("B the engine of four slots", e.init(std::move(m), c, err) && e.n_slots() == 4, err)) {
        return false;
    }
    md::model & mm = e.test_model();
    // B1: four frames at distinct addresses, each with its own graphs
    bool distinct = true;
    std::vector<const void *> addr;
    for (int32_t s = 0; s < 4; ++s) {
        rt::target & t = e.test_target(s);
        for (int32_t card = 0; card < 2; ++card) {
            const void * mine[] = {t.card(card).s[0].gdn,    t.card(card).s[0].conv, t.card(card).logits,
                                   t.card(card).samp.prm,    t.card(card).carry,     t.card(card).s[3].k_cache,
                                   t.draft(card).carry_prev, t.draft(card).k_cache,  t.card(card).rstate};
            for (const void * p : mine) {
                for (const void * q : addr) {
                    distinct = distinct && p != q;
                }
                distinct = distinct && p != nullptr;
                addr.push_back(p);
            }
        }
        distinct = distinct && (eager || t.captures() > 0) && t.slot_index() == s;
    }
    put("B four slot frames at distinct addresses (GDN state, histories, logits, sampler, carries, KV, round state), "
        "each with its own graphs",
        distinct, std::to_string(addr.size()) + " addresses");
    // the requests' options: greedy, a budget
    rt::request_options ro;
    ro.sampling.temp = 0.0f;
    ro.max_tokens = 64;
    const auto read_all = [&](int32_t s, const std::vector<int32_t> & tok, int32_t reuse) {
        int32_t np = 0;
        if (e.read_begin(s, tok.data() + reuse, int32_t(tok.size()) - reuse, reuse, np) != 0) {
            return false;
        }
        rt::read_status st = rt::read_status::yielded;
        while (st == rt::read_status::yielded || st == rt::read_status::chunk_done) {
            st = e.read_step(s, 48);
        }
        return st == rt::read_status::done;
    };
    const auto steps = [&](int32_t s, int32_t n) {
        int32_t out[rt::max_step_tokens];
        int32_t n_out = 0;
        bool end = false;
        for (int32_t i = 0; i < n; ++i) {
            if (e.step(s, out, n_out, end) != rt::run_status::ok) {
                return false;
            }
        }
        return true;
    };
    std::vector<int32_t> p300(300), p9000(9000);
    for (size_t i = 0; i < p300.size(); ++i) {
        p300[i] = int32_t(1000 + i % 977);
    }
    for (size_t i = 0; i < p9000.size(); ++i) {
        p9000[i] = int32_t(2000 + (i * 7) % 4999);
    }
    if (read_only) {
        part_b_fmu1(e, ro, p300, p9000);
        return true;
    }
    if (joint_only) {
        part_b_fmu2(e, ro, eager, joint_one);
        return true;
    }
    bool ok = true;
    for (int32_t s = 0; s < 4; ++s) {
        ok = ok && e.set_sampling(s, ro) == rt::run_status::ok;
    }
    ok = ok && read_all(0, p300, 0) && steps(0, 3) && read_all(1, p300, 0) && steps(1, 2);
    put("B reads and steps on slots 0 and 1", ok, e.error());
    // B2: seeded bytes in every slot's GDN state, histories and KV; slot 1's reset clears its own only
    std::vector<seeded> seeds;
    for (int32_t s = 0; s < 4 && ok; ++s) {
        rt::target & t = e.test_target(s);
        for (int32_t card = 0; card < 2; ++card) {
            const int32_t dev = mm.device[card];
            uint8_t * gdn = reinterpret_cast<uint8_t *>(t.card(card).s[0].gdn) + 4096;
            uint8_t * conv = reinterpret_cast<uint8_t *>(t.card(card).s[0].conv);
            uint8_t * kv = reinterpret_cast<uint8_t *>(t.card(card).s[3].k_cache) + kv_cell * 1000; // cell 1000
            for (uint8_t * p : {gdn, conv, kv}) {
                if (s >= 2 && p == kv) {
                    continue; // slots 2 and 3 hold no pages yet
                }
                seeded x{p, dev, pattern(0x4200 + uint64_t(s) * 31 + uint64_t(card) * 7 + uint64_t(p - gdn), 256)};
                ok = ok && write_at(dev, p, x.v);
                seeds.push_back({p, dev, x.v});
            }
        }
    }
    rt::ledger l0 = e.test_target(0).slot();
    uint32_t w0[2] = {0, 0}, g0[2] = {0, 0};
    ok = ok && e.test_target(0).versions().read_words(w0, g0, err);
    ok = ok && e.test_target(1).reset(err);
    bool iso = ok;
    int32_t kept_seeds = 0, cleared = 0;
    for (const seeded & x : seeds) {
        // slot 1's (on either card) are zero now; every other slot's as written
        bool one = false;
        for (int32_t card = 0; card < 2; ++card) {
            rt::target & t = e.test_target(1);
            one = one || x.p == reinterpret_cast<uint8_t *>(t.card(card).s[0].gdn) + 4096 ||
                  x.p == reinterpret_cast<uint8_t *>(t.card(card).s[0].conv) ||
                  x.p == reinterpret_cast<uint8_t *>(t.card(card).s[3].k_cache) + kv_cell * 1000;
        }
        std::vector<uint8_t> got;
        if (one) {
            iso = iso && read_at(x.dev, x.p, got, x.v.size()) && all_zero(got);
            ++cleared;
        } else {
            iso = iso && same_bytes(x);
            ++kept_seeds;
        }
    }
    uint32_t w1[2] = {0, 0}, g1[2] = {0, 0};
    iso = iso && e.test_target(0).versions().read_words(w1, g1, err) && w1[0] == w0[0] && w1[1] == w0[1] &&
          g1[0] == g0[0] && g1[1] == g0[1] && e.test_target(0).slot().n == l0.n &&
          e.test_target(0).slot().tok == l0.tok;
    put("B a slot's reset clears its own state only (seeded GDN state, histories and KV; the others' ledgers and "
        "committed versions unchanged)",
        iso,
        std::to_string(cleared) + " cleared, " + std::to_string(kept_seeds) + " kept; slot 0 cur " +
            std::to_string(w0[0]) + " gdn " + std::to_string(g0[0]));
    // B3: setting slot 2's options leaves slot 0's open request open
    const std::string line0 = e.last_summary_line();
    ok = e.set_sampling(2, ro) == rt::run_status::ok && steps(0, 1);
    put("B a request option set on slot 2 does not finish slot 0's request", ok && e.last_summary_line() == line0,
        e.error());
    // B4: the state check a slot's versions away from its initial bank: slot 0 after steps, then slot 2 reads and steps;
    // slot 0's committed versions are unchanged
    ok = e.test_target(0).versions().read_words(w0, g0, err) && read_all(2, p300, 0) && steps(2, 2) &&
         e.test_target(0).versions().read_words(w1, g1, err);
    put("B another slot's rounds leave a slot's committed versions (away from the initial bank)",
        ok && w1[0] == w0[0] && w1[1] == w0[1] && g1[0] == g0[0] && g1[1] == g0[1] && (w0[0] != 0 || g0[0] != 0),
        "slot 0 cur " + std::to_string(w0[0]) + " gdn " + std::to_string(g0[0]));
    // B5: truncate and erase on one slot leave the others' tokens
    std::vector<int32_t> h0, h1, h2, h0b, h2b;
    int32_t pend = -1;
    e.history(0, h0, pend);
    e.history(2, h2, pend);
    ok = e.truncate(1, 0) == rt::run_status::ok && e.history(0, h0b, pend) == rt::run_status::ok &&
         e.history(1, h1, pend) == rt::run_status::ok;
    const bool erase_ok = ok && h1.empty() && h0b == h0;
    ok = e.truncate(0, 100) == rt::run_status::ok && e.history(0, h0b, pend) == rt::run_status::ok &&
         e.history(2, h2b, pend) == rt::run_status::ok;
    put("B a slot's erase and truncate leave the other slots' tokens", erase_ok && ok && h0b.size() == 100 && h2b == h2,
        e.error());
    // B6: cancellation: before the first chunk (nothing new kept), during a chunk (the chunk finishes, BG1), then the
    // resumed read from its checkpoint
    int32_t np = 0;
    ok = e.read_begin(3, p9000.data(), int32_t(p9000.size()), 0, np) == 0;
    e.read_cancel(3);
    const bool cancel0 = ok && e.read_step(3, 1) == rt::read_status::cancelled && e.held(3) == 0 &&
                         e.lane_owner(0) < 0 && e.lane_owner(1) < 0;
    ok = e.read_begin(3, p9000.data(), int32_t(p9000.size()), 0, np) == 0 &&
         e.read_step(3, 1) == rt::read_status::yielded;
    rt::read_info ri;
    e.get_read_info(3, ri);
    const bool mid = ok && ri.in_chunk && ri.lane == 0 && e.lane_owner(0) == 3;
    e.read_cancel(3);
    rt::read_status st = rt::read_status::yielded;
    int32_t n_after = 0;
    while (ok && st == rt::read_status::yielded) {
        st = e.read_step(3, 1);
        ++n_after;
    }
    const bool cancel1 = mid && st == rt::read_status::cancelled && e.held(3) == 8192 && e.lane_owner(0) < 0;
    // the next read keeps the 8,192 cancelled-at tokens: it restores the chunk's checkpoint and reads the rest
    ok = read_all(3, p9000, 8192);
    std::vector<int32_t> h3;
    e.history(3, h3, pend);
    put("B cancellation: before any chunk nothing new is kept; during a chunk it finishes (BG1), its lane released; "
        "the next read resumes from its checkpoint",
        cancel0 && cancel1 && ok && h3 == p9000,
        "groups after the cancel " + std::to_string(n_after) + "; " + e.error());
    // B7: a short read in the other-reader lane while a long read's chunk is suspended in the main arena
    ok = e.truncate(1, 0) == rt::run_status::ok && e.read_begin(1, p9000.data(), int32_t(p9000.size()), 0, np) == 0 &&
         e.read_step(1, 2) == rt::read_status::yielded && e.lane_owner(0) == 1;
    ok = ok && e.truncate(2, 0) == rt::run_status::ok && e.read_begin(2, p300.data(), 300, 0, np) == 0 &&
         e.read_step(2, 1) == rt::read_status::yielded && e.lane_owner(1) == 2 && e.lane_owner(0) == 1;
    st = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded) {
        st = e.read_step(2, 4);
    }
    const bool nested = ok && st == rt::read_status::done && e.lane_owner(1) < 0 && e.lane_owner(0) == 1;
    st = rt::read_status::yielded;
    while (nested && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
        st = e.read_step(1, 3);
    }
    std::vector<int32_t> h1b, h2c;
    e.history(1, h1b, pend);
    e.history(2, h2c, pend);
    put("B a short read runs in the other-reader lane while a long read's chunk is suspended in the main arena",
        nested && st == rt::read_status::done && h1b == p9000 && h2c == p300 && steps(1, 1) && steps(2, 1), e.error());
    // B8: the pool: four pages; the least recently released idle slot is reclaimed, an active one never
    rt::pool_info pi;
    e.get_pool_info(pi);
    const int32_t before = pi.free_pages;
    e.release_slot(2); // released first: the least recently used
    e.release_slot(0);
    e.release_slot(3);
    std::vector<int32_t> big(40000);
    for (size_t i = 0; i < big.size(); ++i) {
        big[i] = int32_t(3000 + (i * 13) % 7001);
    }
    // slot 1 stays active (busy since its read); a 40,000-token read on it needs 3 pages
    ok = e.truncate(1, 0) == rt::run_status::ok && read_all(1, big, 0);
    rt::pool_info pa;
    e.get_pool_info(pa);
    const bool reclaimed = ok && pa.reclaims >= 1 && e.held(2) == 0 && pa.mapped[1] == 3;
    // every other slot idle and holding pages is reclaimable; with slot 1 active, a slot needing all four pages fails
    e.set_sampling(0, ro);
    int32_t r = e.read_begin(0, big.data(), int32_t(big.size()), 0, np);
    const bool refused = r == 1 && e.held(1) == int64_t(big.size());
    put("B the pool reclaims the least recently released idle slot, never an active one", reclaimed && refused,
        "free before " + std::to_string(before) + ", reclaims " + std::to_string(pa.reclaims) + ", slot 1 pages " +
            std::to_string(pa.mapped[1]) + ", a 4-page read on slot 0 returned " + std::to_string(r));
    part_b_fmu1(e, ro, p300, p9000);
    return true;
}

// ---- part C (MXMU1): two slots on one card with the expert cache and the host-backed KV ----

bool host_same(const seeded & s) {
    return std::memcmp(s.p, s.v.data(), s.v.size()) == 0;
}

bool host_zero(const uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] != 0) {
            return false;
        }
    }
    return true;
}

void part_c0() {
    // the reservation's refusals come before any allocation (the model object holds no weights)
    const auto refused = [](bool single, bool kv_host, bool cache, int32_t slots, int32_t cells, std::string & err) {
        md::model m;
        m.single = single;
        m.kv_host = kv_host;
        m.expert_cache = cache;
        md::state_params sp;
        sp.n_cells = cells;
        sp.ple_conv_kernel = 4;
        sp.ple_ngram = 3;
        sp.kv_window = kv_host ? md::kv_tier::window_cells : 0;
        sp.single = single;
        int64_t hb = 0;
        err.clear();
        const bool ok = fn::loader::reserve_state(m, sp, slots, hb, err);
        return !ok && hb == 0 && m.state.empty();
    };
    std::string e1, e2, e3, e4, e5;
    const bool r1 = refused(true, true, true, 3, 65536, e1);
    const bool r2 = refused(true, false, true, 2, 65536, e2);
    const bool r3 = refused(false, true, true, 2, 65536, e3);
    const bool r4 = refused(false, true, false, 2, 65536, e4);
    // 32 x 262,144 cells: every slot's pinned history beyond any host, refused with the arithmetic
    const bool r5 = refused(true, true, true, 2, 32 * 262144, e5);
    put("C0 refused before any allocation: three slots on one card, two without the host-backed KV, cached TP, "
        "two-card host KV, and slots whose histories exceed the host",
        r1 && r2 && r3 && r4 && r5 && e5.find("MemAvailable") != std::string::npos,
        e1 + " | " + e2 + " | " + e3 + " | " + e4 + " | " + e5);
}

bool part_c(int32_t dev, bool eager) {
    const int32_t cells = 65536;
    fnfl_target_args aa;
    aa.n_slots = 2;
    aa.single_card = dev;
    aa.kv_host = true;
    aa.draft_attention = "sparse";
    aa.draft_attention_given = true;
    aa.draft_path = "/home/jeremy/src/flashnext/work/ME7/out/mtp-Qwen3.8-Flash-Next-shared-Q8_0-q2experts.gguf";
    aa.device_cap_mib = 16160; // Config D
    aa.ex_reserve_mib = 640;
    aa.cache.lend_pct = 98;
    aa.cache.min_keep = 64;
    aa.ex_eager = eager;
    aa.cache.executor = eager ? "eager" : "graph";
    auto m = std::make_unique<md::model>();
    if (!fnfl_load_dry(aa, cells, *m)) {
        return put("C the zero-weight load of two slots on one card", false);
    }
    rt::draft_options dopt;
    if (!fnfl_draft_settings(aa, true, "state-test mu1-one", dopt)) {
        return put("C the draft settings", false);
    }
    rt::engine_config c;
    c.model_path = aa.model_path;
    c.n_ctx_slot = cells;
    c.draft = dopt;
    c.eog = {248046};
    c.timeout_s = 30.0;
    c.eager = eager;
    c.short_read = aa.cache.short_read;
    c.prompt_chunk = aa.cache.prompt_chunk;
    fn::runtime::expert_cache * xc = nullptr;
    c.experts = [&aa, &xc](md::model & mm, const md::service_hook & hook,
                           std::string & e2) -> std::unique_ptr<md::expert_service> {
        std::unique_ptr<fn::runtime::expert_cache> x = fnfl_expert_cache(aa, mm, true, "state-test mu1-one", hook);
        if (x == nullptr) {
            e2 = "the expert cache";
            return nullptr;
        }
        xc = x.get();
        return x;
    };
    rt::engine e;
    std::string err;
    if (!put("C the engine of two slots on one card (expert cache, host KV)",
             e.init(std::move(m), c, err) && e.n_slots() == 2 && xc != nullptr, err)) {
        return false;
    }
    printf("state-test mu1-one: %s\n", e.settings().c_str());
    md::model & mm = e.test_model();
    // C1: frames, host histories and windows at distinct addresses, each slot with its own graphs and tier; one stream
    bool distinct = mm.pool == nullptr && e.lane_rows(1) == 0;
    std::vector<const void *> addr;
    for (int32_t s = 0; s < 2; ++s) {
        rt::target & t = e.test_target(s);
        const md::kv_tier * kt = t.tier();
        distinct = distinct && kt != nullptr && t.slot_index() == s && (eager || t.captures() > 0) &&
                   t.stream(0) == t.stream(1) && t.stream(0) == e.test_target(0).stream(0);
        for (int32_t card = 0; card < 2 && kt != nullptr; ++card) {
            const void * mine[] = {
                t.card(card).s[0].gdn, t.card(card).s[0].conv, t.card(card).logits,
                t.card(card).samp.prm, t.card(card).carry,     t.draft(card).carry_prev,
                t.card(card).rstate,   kt->host_k(card, 0),    kt->host_v(card, 0),
                kt->stage_k(card),     kt->stage_v(card),      kt->host_k(card, md::kv_tier::draft_attn)};
            for (const void * p : mine) {
                for (const void * q : addr) {
                    distinct = distinct && p != q;
                }
                distinct = distinct && p != nullptr;
                addr.push_back(p);
            }
        }
    }
    put("C1 two slot frames, host histories and device windows at distinct addresses, each with its own graphs and "
        "tier, one compute stream, no KV pool or other-reader lane",
        distinct, std::to_string(addr.size()) + " addresses");
    rt::request_options ro;
    ro.sampling.temp = 0.0f;
    ro.max_tokens = 256; // every check's steps fit one request's budget
    const auto loan_free = [&]() { return !xc->prompt_held() && e.prefill_driver().loan_job() < 0; };
    int32_t turns = 0, between = 0;
    bool settled = true;
    // a read in whole-chunk turns, slot `peer` stepping between its chunks (peer < 0: none)
    const auto read_turns = [&](int32_t s, const std::vector<int32_t> & tok, int32_t reuse, int32_t peer) {
        int32_t np = 0;
        if (e.read_begin(s, tok.data() + reuse, int32_t(tok.size()) - reuse, reuse, np) != 0) {
            return false;
        }
        rt::read_status st = rt::read_status::chunk_done;
        while (st == rt::read_status::chunk_done) {
            st = e.read_step(s, 1); // a cached engine runs the whole chunk whatever n_groups is
            ++turns;
            settled = settled && loan_free();
            int32_t out[rt::max_step_tokens];
            int32_t n_out = 0;
            bool end = false;
            if (st == rt::read_status::chunk_done && peer >= 0) {
                if (e.step(peer, out, n_out, end) != rt::run_status::ok) {
                    return false;
                }
                ++between;
            }
        }
        return st == rt::read_status::done;
    };
    std::vector<int32_t> toks[2];
    const auto steps = [&](int32_t s, int32_t n) {
        int32_t out[rt::max_step_tokens];
        int32_t n_out = 0;
        bool end = false;
        for (int32_t i = 0; i < n; ++i) {
            if (e.step(s, out, n_out, end) != rt::run_status::ok) {
                return false;
            }
            toks[s].insert(toks[s].end(), out, out + n_out);
        }
        return true;
    };
    std::vector<int32_t> p300(300), p9000(9000);
    for (size_t i = 0; i < p300.size(); ++i) {
        p300[i] = int32_t(1000 + i % 977);
    }
    for (size_t i = 0; i < p9000.size(); ++i) {
        p9000[i] = int32_t(2000 + (i * 7) % 4999);
    }
    bool ok = e.set_sampling(0, ro) == rt::run_status::ok && e.set_sampling(1, ro) == rt::run_status::ok;
    // C2: slot switches: slot 0 reads and decodes, slot 1 reads (0 -> 1 after a prompt), then alternate steps
    ok = ok && read_turns(0, p300, 0, -1) && steps(0, 3) && read_turns(1, p300, 0, -1) && steps(1, 2) && steps(0, 2) &&
         steps(1, 2) && steps(0, 1);
    put("C2 slot switches both ways, after a prompt and mid-decode", ok && loan_free(), e.error());
    // C3: seeded bytes in each slot's host history (target layer 0, draft) and device window; the peer's work and the
    // slot's own reset
    std::vector<seeded> host_seeds, win_seeds;
    for (int32_t s = 0; s < 2 && ok; ++s) {
        rt::target & t = e.test_target(s);
        md::kv_tier * kt = t.tier();
        ok = ok && t.kv_flush(err);
        for (int32_t card = 0; card < 2; ++card) {
            for (int32_t a : {0, int32_t(md::kv_tier::draft_attn)}) {
                uint8_t * h = reinterpret_cast<uint8_t *>(kt->host_k(card, a)) + size_t(512) * 40000; // cell 40,000
                seeded x{h, -1, pattern(0x4d58000 + uint64_t(s) * 97 + uint64_t(card) * 13 + uint64_t(a), 512)};
                std::memcpy(h, x.v.data(), x.v.size());
                host_seeds.push_back(x);
            }
            // a cell of layer 1's window the slot never reaches (its depth is a few hundred tokens)
            uint8_t * w = reinterpret_cast<uint8_t *>(kt->stage_k(card)) + size_t(512) * 60000;
            seeded y{w, mm.device[card], pattern(0x4d59000 + uint64_t(s) * 97 + uint64_t(card), 512)};
            ok = ok && write_at(y.dev, w, y.v);
            win_seeds.push_back(y);
        }
    }
    const int64_t lo0 = e.test_target(0).tier()->lo();
    // slot 1: a 9,000-token read (two chunks: a turn each, slot 0 decoding between them) and decode
    ok = ok && e.truncate(1, 0) == rt::run_status::ok && read_turns(1, p9000, 0, 0) && steps(1, 2);
    bool kept = ok;
    for (size_t i = 0; i < host_seeds.size(); ++i) {
        const bool mine = i < host_seeds.size() / 2; // slot 0's
        kept = kept && (!mine || host_same(host_seeds[i]));
    }
    for (size_t i = 0; i < win_seeds.size() / 2; ++i) {
        kept = kept && same_bytes(win_seeds[i]);
    }
    put("C3 the peer's long read (whole-chunk turns, decode between them) and decode leave a slot's seeded host "
        "history and device window",
        kept,
        "slot 0 lo " + std::to_string(lo0) + " -> " + std::to_string(e.test_target(0).tier()->lo()) + "; " + e.error());
    put("C4 every read step returned the prompt loan (no hold, no loan job) and the peer decoded between chunks",
        settled && between >= 1 && turns >= 3,
        std::to_string(turns) + " turns, " + std::to_string(between) + " peer steps between chunks");
    // C5: slot 1's reset clears its own host history only
    ok = e.test_target(1).reset(err);
    bool iso = ok;
    for (size_t i = 0; i < host_seeds.size(); ++i) {
        const bool mine = i < host_seeds.size() / 2;
        iso = iso && (mine ? host_same(host_seeds[i]) : host_zero(host_seeds[i].p, host_seeds[i].v.size()));
    }
    put("C5 a slot's reset zeroes its own host history only (seeded, both logical cards, target and draft layers)", iso,
        err);
    // C6: ME3's short read on slot 0 (30 new tokens after its processed ones: a pure extension, at most the 64-row
    // short read) in one step, then decode
    std::vector<int32_t> h1;
    int32_t pend = -1;
    e.history(0, h1, pend);
    std::vector<int32_t> app(h1);
    for (int32_t i = 0; i < 30; ++i) {
        app.push_back(3000 + i);
    }
    int32_t np = 0;
    const int32_t reuse = int32_t(h1.size());
    ok = e.read_begin(0, app.data() + reuse, int32_t(app.size()) - reuse, reuse, np) == 0;
    rt::read_info ri;
    const rt::read_status sst = ok ? e.read_step(0, 1) : rt::read_status::error;
    e.get_read_info(0, ri);
    put("C6 ME3's short read runs in one step (no loan) and decode follows",
        ok && sst == rt::read_status::done && ri.steps == 1 && e.last_prefill().short_read && loan_free() &&
            steps(0, 1),
        e.error());
    // C7: a suspended cached chunk (test_partial_reads) holds the loan; the peer's step is refused; the cancel finishes
    // the chunk and returns the loan; the peer then steps
    e.test_partial_reads(true);
    ok = e.truncate(1, 0) == rt::run_status::ok && e.read_begin(1, p9000.data(), int32_t(p9000.size()), 0, np) == 0 &&
         e.read_step(1, 1) == rt::read_status::yielded;
    const bool held = ok && xc->prompt_held() && e.prefill_driver().loan_job() >= 0 && e.lane_owner(0) == 1;
    int32_t out[rt::max_step_tokens];
    int32_t n_out = 0;
    bool end = false;
    const bool refused = held && e.step(0, out, n_out, end) == rt::run_status::bad_request;
    e.read_cancel(1);
    rt::read_status st = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded) {
        st = e.read_step(1, 4);
    }
    const bool cancelled = st == rt::read_status::cancelled && loan_free() && e.held(1) == 8192;
    put("C7 a suspended cached chunk holds the loan and refuses the peer's step; its cancel finishes the chunk, "
        "returns the loan, and the peer steps",
        held && refused && cancelled && steps(0, 1), e.error());
    // C8: an abandoned suspended chunk (a failed step's path) returns its loan; the slot re-reads from token 0
    ok = e.read_begin(1, p9000.data() + 8192, int32_t(p9000.size()) - 8192, 8192, np) == 0 &&
         e.read_step(1, 1) == rt::read_status::yielded && xc->prompt_held();
    e.test_abandon_read(1);
    const bool abandoned = ok && loan_free() && e.lane_owner(0) < 0 && e.held(1) == 0;
    e.test_partial_reads(false);
    ok = abandoned && steps(0, 1) && read_turns(1, p9000, 0, 0) && steps(1, 1);
    put("C8 an abandoned suspended chunk returns its loan; the peer steps and the slot re-reads from token 0", ok,
        e.error());
    // C9: erase and truncate on one slot leave the other's tokens
    std::vector<int32_t> h0, h0b, h1b;
    e.history(0, h0, pend);
    ok = e.truncate(1, 0) == rt::run_status::ok && e.history(0, h0b, pend) == rt::run_status::ok &&
         e.history(1, h1b, pend) == rt::run_status::ok;
    const bool erase_ok = ok && h1b.empty() && h0b == h0;
    ok = read_turns(1, p300, 0, -1) && e.truncate(0, 100) == rt::run_status::ok &&
         e.history(0, h0b, pend) == rt::run_status::ok && e.history(1, h1b, pend) == rt::run_status::ok;
    put("C9 a slot's erase and truncate leave the other slot's tokens",
        erase_ok && ok && h0b.size() == 100 && h1b == p300, e.error());
    // C11 (FMU2 Amendment 3): two slots through engine::step_joint on one card. Off resident TP each member takes the
    // alone branch (runtime::step between step_enter and step_leave); each slot's steps, tokens and ledger must equal
    // the same steps through engine::step in the same order (zero weights: the plumbing, not identity)
    struct trace {
        std::vector<int32_t> tok, n_out, end;
    };
    const auto reread = [&]() {
        bool r = true;
        for (int32_t s = 0; s < 2; ++s) {
            r = r && e.truncate(s, 0) == rt::run_status::ok && e.set_sampling(s, ro) == rt::run_status::ok &&
                read_turns(s, p300, 0, -1);
        }
        return r;
    };
    trace lone[2], joint[2];
    std::vector<int32_t> hl[2], hj[2];
    int32_t pl[2] = {-1, -1}, pj[2] = {-1, -1};
    bool c11 = reread();
    for (int32_t i = 0; i < 6 && c11; ++i) {
        for (int32_t s = 0; s < 2 && c11; ++s) {
            int32_t out[rt::max_step_tokens];
            int32_t n_out = 0;
            bool end = false;
            c11 = e.step(s, out, n_out, end) == rt::run_status::ok;
            lone[s].tok.insert(lone[s].tok.end(), out, out + n_out);
            lone[s].n_out.push_back(n_out);
            lone[s].end.push_back(end ? 1 : 0);
        }
    }
    for (int32_t s = 0; s < 2 && c11; ++s) {
        c11 = e.history(s, hl[s], pl[s]) == rt::run_status::ok;
    }
    c11 = c11 && reread();
    bool alone = true;
    const int32_t both[2] = {0, 1};
    for (int32_t i = 0; i < 6 && c11; ++i) {
        rt::joint_step_out res[2];
        e.step_joint(both, 2, res);
        for (int32_t s = 0; s < 2; ++s) {
            c11 = c11 && res[s].st == rt::run_status::ok;
            alone = alone && !res[s].joint;
            joint[s].tok.insert(joint[s].tok.end(), res[s].out, res[s].out + res[s].n_out);
            joint[s].n_out.push_back(res[s].n_out);
            joint[s].end.push_back(res[s].end ? 1 : 0);
        }
    }
    for (int32_t s = 0; s < 2 && c11; ++s) {
        c11 = e.history(s, hj[s], pj[s]) == rt::run_status::ok;
    }
    bool same = c11 && alone;
    for (int32_t s = 0; s < 2; ++s) {
        same = same && joint[s].tok == lone[s].tok && joint[s].n_out == lone[s].n_out && joint[s].end == lone[s].end &&
               hj[s] == hl[s] && pj[s] == pl[s] && !lone[s].tok.empty();
    }
    put("C11 FMU2 Amendment 3: two slots through step_joint on one card step alone, each slot's steps, tokens and "
        "ledger its engine::step run's",
        same && loan_free(),
        e.error() + " tokens " + std::to_string(lone[0].tok.size()) + "/" + std::to_string(joint[0].tok.size()) + ", " +
            std::to_string(lone[1].tok.size()) + "/" + std::to_string(joint[1].tok.size()) +
            (alone ? "" : "; a lane was joint"));
    std::string note;
    put("C10 the expert cache drains with no loan out", xc->drain(note) && loan_free(), note);
    return true;
}

} // namespace

int fnfl_state_test_mxmu1(const std::string & run, int32_t dev) {
    g_fail = 0;
    part_c0();
    if (run != "c0") { // c0: the refusals alone (no load: a host that cannot pin the expert backing)
        part_c(dev, run == "eager");
    }
    printf("state-test mu1-one: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}

int fnfl_state_test_mu1(const std::string & run, int32_t dev0, int32_t dev1, bool kv_q8) {
    g_fail = 0;
    printf("state-test mu1: KV format %s\n", kv_q8 ? "q8 (KV8)" : "bf16");
    // read (FMU1): part B's engine and its FMU1 read-job cases alone, eagerly (Compute Sanitizer's run)
    const bool read_only = run == "read";
    // joint (FMU2): part B's engine on graphs at d = 4 and the joint step cases alone; joint-eager: eagerly (phased);
    // joint-san: eagerly, D1 and one four-member round (Compute Sanitizer's run)
    const bool joint_only = run == "joint" || run == "joint-eager" || run == "joint-san";
    const bool joint_eager = run == "joint-eager" || run == "joint-san";
    if (!read_only && !joint_only) {
        part_a(dev0, dev1, kv_q8);
    }
    fnfl_target_args a;
    a.kv_q8 = kv_q8;
    part_b(a, run == "eager" || read_only || joint_eager, read_only, joint_only, run == "joint-san");
    printf("state-test mu1: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
