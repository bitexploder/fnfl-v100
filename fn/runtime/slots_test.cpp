// MU1: the multi-slot engine's host checks (slots_test.h)

#include "fn/runtime/slots_test.h"

#include "fn/model/kv_pool.h"
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
        // the shared book: entries owned by a slot; a checkpoint restores its owner only; an owner's drop and clear
        // leave the others'; the oldest written is replaced whatever its owner; owner 0 alone is the one-slot book
        checkpoint_book b;
        const std::vector<int32_t> tok0(20000, 7), tok1(20000, 9);
        for (int32_t k = 0; k < 4; ++k) {
            const int64_t pos = int64_t(k + 1) * 4096;
            b.record(b.slot_for(pos, 0), pos, token_key(tok0.data(), pos), 0);
            b.record(b.slot_for(pos, 1), pos, token_key(tok1.data(), pos), 1);
        }
        const bool full = b.count() == 8 && b.count_owner(0) == 4 && b.count_owner(1) == 4;
        const int32_t c0 = choose_checkpoint(b, tok0, 15000, 0);
        const int32_t c1 = choose_checkpoint(b, tok1, 15000, 1);
        const bool c1_ok = c1 >= 0 && b.at(c1).owner == 1 && b.at(c1).pos == 12288;
        // slot 1 asks with slot 0's tokens: its entries at or below the reuse point do not match (dropped; the one
        // at 16,384 lies past it and stays), and none of slot 0's is used
        std::vector<int32_t> other = tok0;
        const int32_t x = choose_checkpoint(b, other, 15000, 1);
        const bool sel = c0 >= 0 && b.at(c0).owner == 0 && b.at(c0).pos == 12288 && c1_ok && x < 0 &&
                         b.count_owner(1) == 1 && b.count_owner(0) == 4;
        // a new owner's write takes an empty entry (slot 1's dropped ones) before replacing the oldest written
        const int32_t e = b.slot_for(8192, 2);
        const bool repl = b.at(e).pos < 0;
        b.record(e, 8192, token_key(tok1.data(), 8192), 2);
        b.drop_after(8192, 0);
        const bool drop = b.count_owner(0) == 2 && b.count_owner(2) == 1 && b.count_owner(1) == 1;
        b.clear_owner(0);
        // with every entry written, the next new write replaces the oldest written whatever its owner
        checkpoint_book f;
        for (int32_t k = 0; k < pf_max_checkpoints; ++k) {
            f.record(f.slot_for(int64_t(k + 1) * 8192, k % 2), int64_t(k + 1) * 8192, 1, k % 2);
        }
        const int32_t old_e = f.slot_for(999, 3);
        const bool clr =
            b.count_owner(0) == 0 && b.count_owner(2) == 1 && f.at(old_e).pos == 8192 && f.at(old_e).owner == 0;
        ok = put("checkpoint book owners (selection, mismatch, replacement, drop, clear)",
                 full && sel && repl && drop && clr,
                 std::to_string(full) + std::to_string(sel) + std::to_string(repl) + std::to_string(drop) +
                     std::to_string(clr)) &&
             ok;
    }
    {
        // the multi-slot read's plan is the one-slot read's (the same chunk partition, so the same numerics): lone and
        // concurrent reads both use plan_chunks at the main arena's 8,192 rows; a short read's chunks fit the
        // other-reader lane
        const std::vector<pf_chunk> p64 = plan_chunks(0, 65536 + 128);
        const std::vector<pf_chunk> r3 = plan_chunks(0, 2048);
        const std::vector<pf_chunk> mid = plan_chunks(6000, 10096);
        bool a = p64.size() == 9 && p64[0].rows == 8192 && p64[8].rows == 128;
        bool b = r3.size() == 1 && r3[0].rows == 2048 && r3[0].rows <= model::lane_rows &&
                 r3[0].half[0] <= model::lane_half_rows && r3[0].half[1] <= model::lane_half_rows;
        bool c = mid.size() == 2 && mid[0].rows == 2192 && mid[1].rows == 1904;
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
