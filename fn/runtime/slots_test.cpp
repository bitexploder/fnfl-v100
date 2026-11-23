// MU1: the multi-slot engine's host checks (slots_test.h)

#include "fn/runtime/slots_test.h"

#include "fn/model/kv_pool.h"
#include "fn/runtime/image_rows.h"
#include "fn/runtime/prefill.h"
#include "fn/runtime/prompt_plan.h"

#include <cstdio>
#include <string>
#include <vector>

namespace fn::runtime {

namespace {

bool put(const char * name, bool ok, const std::string & detail) {
    std::printf("ex-cpu-test: mu1 %s: %s%s%s\n", name, ok ? "PASS" : "FAIL", detail.empty() ? "" : " (",
                detail.empty() ? "" : (detail + ")").c_str());
    return ok;
}

} // namespace

bool slots_cpu_test() {
    bool ok = true;
    using model::kv_pages;
    const int32_t pc = model::kv_page_cells, np = 262144 / pc;
    {
        // rounding: a prefix needs ceil(cells / page) pages, capped at the pool; zero cells none
        const bool r = kv_pages::pages_for(0, pc, np) == 0 && kv_pages::pages_for(1, pc, np) == 1 &&
                       kv_pages::pages_for(pc, pc, np) == 1 && kv_pages::pages_for(pc + 1, pc, np) == 2 &&
                       kv_pages::pages_for(262144, pc, np) == 16 && kv_pages::pages_for(262144 + 5, pc, np) == 16;
        // C6's arithmetic: R5 (32,768 + 128 output + the step's 69 cells) is 3 pages; four fit the 16
        const int32_t r5 = kv_pages::pages_for(32768 + 128 + 69, pc, np);
        // C4's: R4 (131,072 + 32 + 69) is 9 pages; two do not fit
        const int32_t r4 = kv_pages::pages_for(131072 + 32 + 69, pc, np);
        ok = put("pool page rounding", r && r5 == 3 && 4 * r5 <= np && r4 == 9 && 2 * r4 > np,
                 "R5 " + std::to_string(r5) + " pages, R4 " + std::to_string(r4)) &&
             ok;
    }
    {
        // a reservation that does not fit takes nothing; growth takes the lowest free pages; release returns them in
        // order, and the full capacity goes to one slot
        kv_pages p;
        p.init(4, np);
        std::vector<int32_t> t;
        bool a = p.take(0, 9, t) && t.size() == 9 && t.front() == 0 && p.free_pages() == 7;
        bool b = !p.take(1, 9, t) && t.empty() && p.mapped(1) == 0 && p.free_pages() == 7;
        bool c = p.take(1, 7, t) && t.front() == 9 && p.free_pages() == 0 && !p.take(2, 1, t);
        std::vector<int32_t> out;
        p.give(0, 3, out); // slot 0 keeps 3 pages, frees 6 (its last first)
        bool d = out.size() == 6 && out.front() == 8 && p.mapped(0) == 3 && p.free_pages() == 6;
        bool e = p.take(2, 2, t) && t.size() == 2 && t[0] == 3 && t[1] == 4 && p.physical(2, 1) == 4;
        p.give(0, 0, out);
        p.give(1, 0, out);
        p.give(2, 0, out);
        bool f = p.free_pages() == np && p.take(3, np, t) && p.mapped(3) == np && p.physical(3, np - 1) == np - 1;
        bool g = !p.take(3, np + 1, t) && p.mapped(3) == np;
        ok = put("pool reservation, rollback and release", a && b && c && d && e && f && g,
                 std::to_string(a) + std::to_string(b) + std::to_string(c) + std::to_string(d) + std::to_string(e) +
                     std::to_string(f) + std::to_string(g)) &&
             ok;
    }
    {
        // reclaim: the least recently released idle slot with pages; never an active slot or the requester; never
        // used (0) before any released one
        std::vector<model::kv_slot_view> v(4);
        v[0] = {false, 3, 7};
        v[1] = {false, 2, 4};
        v[2] = {true, 5, 1};
        v[3] = {false, 0, 2};
        const int32_t a = model::kv_reclaim_victim(v, 0);
        const int32_t b = model::kv_reclaim_victim(v, 1);
        v[1].active = true;
        const int32_t c = model::kv_reclaim_victim(v, 0);
        v[0].released = 0;
        const int32_t d = model::kv_reclaim_victim(v, 3);
        ok = put("reclaim order (least recently released idle slot; active slots kept)",
                 a == 1 && b == 0 && c == -1 && d == 0,
                 std::to_string(a) + " " + std::to_string(b) + " " + std::to_string(c) + " " + std::to_string(d)) &&
             ok;
    }
    {
        // BG3: one book a slot: a checkpoint restores its owner only; an owner's mismatch, drop and clear leave the
        // others'; another slot's writes, however many, never touch a slot's entries
        checkpoint_book b(3, model::prefill_checkpoints_per_slot(32768)); // 5 a slot
        const std::vector<int32_t> tok0(40000, 7), tok1(40000, 9);
        for (const int64_t pos : {int64_t(8192), int64_t(16384), int64_t(20000)}) {
            b.record(b.slot_for(pos, 0), pos, token_key(tok0.data(), pos), 0);
            b.record(b.slot_for(pos, 1), pos, token_key(tok1.data(), pos), 1);
        }
        const bool full = b.entries() == 15 && b.count() == 6 && b.count_owner(0) == 3 && b.count_owner(1) == 3;
        const int32_t c0 = choose_checkpoint(b, tok0, 18000, 0);
        const int32_t c1 = choose_checkpoint(b, tok1, 18000, 1);
        const bool own = c0 >= 0 && c0 / 5 == 0 && b.at(c0).owner == 0 && b.at(c0).pos == 16384 && c1 >= 0 &&
                         c1 / 5 == 1 && b.at(c1).owner == 1 && b.at(c1).pos == 16384;
        // slot 1 asks with slot 0's tokens: its entries at or below the reuse point do not match (dropped; 20,000 lies
        // past it and stays), and none of slot 0's is used
        const int32_t x = choose_checkpoint(b, tok0, 18000, 1);
        const bool sel = own && x < 0 && b.count_owner(1) == 1 && b.count_owner(0) == 3;
        // slot 2 writes far more than a book holds (every multiple to 40,960 and many prompt ends): slot 0's three
        // entries are unchanged, slot 2 keeps its newest five
        std::vector<pf_checkpoint> before;
        for (int32_t i = 0; i < 5; ++i) {
            before.push_back(b.at(i));
        }
        for (int32_t k = 1; k <= 30; ++k) {
            const int64_t pos = k % 3 == 0 ? int64_t(k / 3) * 8192 : int64_t(k) * 1000 + 1;
            b.record(b.slot_for(pos, 2), pos, uint64_t(k), 2);
        }
        bool kept = b.count_owner(0) == 3;
        for (int32_t i = 0; i < 5; ++i) {
            kept = kept && b.at(i).pos == before[size_t(i)].pos && b.at(i).key == before[size_t(i)].key;
        }
        const bool full2 = b.count_owner(2) == 5;
        // drop_after and clear are the owner's
        b.drop_after(8192, 0);
        const bool drop = b.count_owner(0) == 1 && b.count_owner(1) == 1 && b.count_owner(2) == 5;
        b.clear_owner(2);
        const bool clr = b.count_owner(2) == 0 && b.count_owner(0) == 1 && b.count_owner(1) == 1 &&
                         b.slot_for(8192, 3) < 0 && b.latest(99999, 3) < 0;
        ok = put("checkpoint books a slot (selection, mismatch, isolation, drop, clear)",
                 full && sel && kept && full2 && drop && clr,
                 std::to_string(full) + std::to_string(sel) + std::to_string(kept) + std::to_string(full2) +
                     std::to_string(drop) + std::to_string(clr)) &&
             ok;
    }
    {
        // BG3's keep rule: every multiple of 8,192 the slot holds and its newest prompt-end checkpoint (one before
        // the prompt's end): a follow-up turn's prompt end replaces the last turn's; the same position reuses its
        // entry; a restore's drop_after keeps what lies at or below it
        checkpoint_book b(1, model::prefill_checkpoints_per_slot(65536)); // 9
        const std::vector<int32_t> tok(70000, 3);
        const auto rec = [&](int64_t pos) { b.record(b.slot_for(pos, 0), pos, token_key(tok.data(), pos), 0); };
        rec(8192);
        rec(9999); // turn 1's prompt end (a 10,000-token prompt)
        const int32_t t1 = b.latest(20000);
        rec(16384);
        rec(24575); // turn 2's: replaces 9,999
        const int32_t t2 = b.latest(30000);
        const bool repl = t1 >= 0 && t2 == t1 && b.at(t2).pos == 24575 && b.count() == 3 && b.latest(9999) >= 0 &&
                          b.at(b.latest(9999)).pos == 8192;
        const int32_t m1 = b.slot_for(16384, 0);
        rec(16384);
        const bool same = b.slot_for(16384, 0) == m1 && b.count() == 3;
        const int32_t r = choose_checkpoint(b, tok, 24575, 0); // a follow-up keeping all but turn 2's last token
        b.drop_after(b.at(r).pos, 0);
        const bool at = r >= 0 && b.at(r).pos == 24575 && b.count() == 3;
        b.drop_after(16384, 0);
        const bool down = b.count() == 2 && b.at(b.latest(99999)).pos == 16384;
        ok = put("checkpoint keep rule (every multiple, the newest prompt end; restore at L - 1)",
                 repl && same && at && down,
                 std::to_string(repl) + std::to_string(same) + std::to_string(at) + std::to_string(down)) &&
             ok;
    }
    {
        // BG3 (cause 4): the split's two books receive the same writes in a different order across slots (the back
        // runs each chunk after the front, interleaved with other slots' work): each slot's book is the same on both
        const int32_t n_owner = 4, per = model::prefill_checkpoints_per_slot(40000);
        struct w {
            int32_t owner;
            int64_t pos;
        };
        std::vector<w> writes;
        for (int32_t s = 0; s < n_owner; ++s) {
            for (int64_t p = 8192; p <= 32768; p += 8192) {
                writes.push_back({s, p});
            }
            writes.push_back({s, 32768 + 1000 + s});
            writes.push_back({s, 32768 + 2000 + s}); // a second prompt end replaces the first
        }
        // the front's order: slot by slot; the back's: round robin from the last slot
        checkpoint_book front(n_owner, per), back(n_owner, per);
        for (const w & x : writes) {
            front.record(front.slot_for(x.pos, x.owner), x.pos, uint64_t(x.pos * 31 + x.owner), x.owner);
        }
        for (int32_t k = 0; k < 6; ++k) {
            for (int32_t s = n_owner - 1; s >= 0; --s) {
                const w & x = writes[size_t(s * 6 + k)];
                back.record(back.slot_for(x.pos, x.owner), x.pos, uint64_t(x.pos * 31 + x.owner), x.owner);
            }
        }
        bool same = true;
        for (int32_t s = 0; s < n_owner; ++s) {
            same = same && front.count_owner(s) == 5 && back.count_owner(s) == 5;
            for (const int64_t n : {int64_t(8191), int64_t(20000), int64_t(32768 + 1500), int64_t(99999)}) {
                const int32_t a = front.latest(n, s), c = back.latest(n, s);
                same = same && (a < 0) == (c < 0) &&
                       (a < 0 || (front.at(a).pos == back.at(c).pos && front.at(a).key == back.at(c).key));
            }
        }
        ok = put("checkpoint books under different write orders (the split's stages agree slot by slot)", same,
                 std::to_string(same)) &&
             ok;
    }
    {
        // VIS1 Stage B: with images the keys are the ledger's: the same token ids with another image (its digest)
        // match only below that image; the same image matches; no ledger reads token_key's keys as before
        std::vector<int32_t> tok(30000, 11);
        std::fill(tok.begin() + 9000, tok.begin() + 9000 + 64 * 32, image_token);
        image_rows      a, o;
        std::string     e;
        const bool      set = a.set(tok.data(), 30000,
                                    {
                                        { 9000, 64, 32, 0x1234 }
        },
                                    e) &&
                              o.set(tok.data(), 30000, { { 9000, 64, 32, 0x9999 } }, e);
        checkpoint_book b;
        for (const int64_t pos : { int64_t(8192), int64_t(16384), int64_t(24576) }) {
            b.record(b.slot_for(pos, 0), pos, a.key(tok.data(), pos), 0);
        }
        const int32_t same  = choose_checkpoint(b, tok, 29000, 0, &a);
        const bool    s_ok  = same >= 0 && b.at(same).pos == 24576;
        const int32_t other = choose_checkpoint(b, tok, 29000, 0, &o);  // drops 24,576 and 16,384 (another image)
        const bool    o_ok  = other >= 0 && b.at(other).pos == 8192 && b.count_owner(0) == 1;
        const bool    k_ok  = a.key(tok.data(), 8192) == token_key(tok.data(), 8192) &&
                              image_rows().key(tok.data(), 20000) == token_key(tok.data(), 20000);
        ok = put("checkpoint keys with images (the same image, another image, no image)", set && s_ok && o_ok && k_ok,
                 std::to_string(set) + std::to_string(s_ok) + std::to_string(o_ok) + std::to_string(k_ok)) &&
             ok;
    }
    {
        // the multi-slot read's plan is the one-slot read's (the same chunk partition, so the same numerics): lone and
        // concurrent reads both use plan_chunks at the main arena's 8,192 rows; a short read's chunks fit the
        // other-reader lane
        // BG3: the last token is its own chunk (the prompt end's checkpoint one before it)
        const std::vector<pf_chunk> p64 = plan_chunks(0, 65536 + 128);
        const std::vector<pf_chunk> r3 = plan_chunks(0, 2048);
        const std::vector<pf_chunk> mid = plan_chunks(6000, 10096);
        bool a = p64.size() == 10 && p64[0].rows == 8192 && p64[8].rows == 127 && p64[9].rows == 1;
        bool b = r3.size() == 2 && r3[0].rows == 2047 && r3[1].rows == 1 && r3[0].rows <= model::lane_rows &&
                 r3[0].half[0] <= model::lane_half_rows && r3[0].half[1] <= model::lane_half_rows;
        bool c = mid.size() == 3 && mid[0].rows == 2192 && mid[1].rows == 1903 && mid[2].rows == 1;
        bool d = true;
        for (int32_t rows = 1; rows <= model::lane_rows; rows += 37) {
            const pf_chunk ch = plan_chunks(0, rows)[0];
            d = d && ch.half[0] <= model::lane_half_rows && ch.half[1] <= model::lane_half_rows;
        }
        ok = put("read plans (the one-slot partition; short reads fit the lane)", a && b && c && d,
                 std::to_string(a) + std::to_string(b) + std::to_string(c) + std::to_string(d)) &&
             ok;
    }
    return ok;
}

} // namespace fn::runtime
