#include "fn/runtime/admission_test.h"

#include "fn/runtime/admission.h"
#include "fn/runtime/cache_counters.h"
#include "fn/runtime/json_check.h"
#include "fn/runtime/loan.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <random>
#include <set>
#include <vector>

namespace fn::runtime {

namespace {

int32_t n_pass = 0, n_fail = 0;

void expect(bool ok, const char * what) {
    (ok ? n_pass : n_fail) += 1;
    if (!ok) {
        std::printf("ex-cpu-test: me9 %s: FAIL\n", what);
    }
}

// the invariants: one record a slot, at most per_layer a layer (within its index range), a key in at most one
// record copying or landed, a key's layer the record's, a started record's key set and an empty one's not
bool invariants(const adm::table & t) {
    std::set<int32_t> slots, keys;
    for (int32_t i = 0; i < adm::max_recs; ++i) {
        const adm::rec & r = t.at(i);
        if (r.st == adm::state::none) {
            continue;
        }
        if (r.layer != i / adm::per_layer || r.slot < 0 || !slots.insert(r.slot).second) {
            return false;
        }
        if (r.st == adm::state::empty && r.key != -1) {
            return false;
        }
        if (r.st == adm::state::copying || r.st == adm::state::landed) {
            if (r.key < 0 || r.key / 512 != r.layer || !keys.insert(r.key).second) {
                return false;
            }
        }
    }
    return true;
}

void transitions() {
    adm::table t;
    expect(t.held_all() == 0 && t.in_flight(5) == false, "an empty table holds nothing");
    expect(t.reserve(2, 100) && t.reserve(2, 101) && t.reserve(2, 102), "three reservations a layer");
    expect(!t.reserve(2, 103), "a fourth reservation of the layer refused (capacity exhausted)");
    expect(!t.reserve(3, 101), "a slot already reserved refused (one owner a slot)");
    expect(!t.reserve(48, 7) && !t.reserve(-1, 7) && !t.reserve(0, -1), "a layer or slot out of range refused");
    expect(t.held(2) == 3 && t.held(3) == 0 && t.held_all() == 3, "held counts");
    const int32_t i0 = 2 * adm::per_layer, i1 = i0 + 1, i2 = i0 + 2;
    expect(t.of_slot(101) == i1 && t.of_slot(999) == -1, "the record of a slot");
    const uint32_t g_before = t.at(i0).gen;
    expect(!t.land(i0) && !t.invalidate(i0), "an empty record neither lands nor invalidates");
    expect(!t.start(i0, 3 * 512 + 7), "a key of another layer refused");
    expect(t.start(i0, 2 * 512 + 7) && t.at(i0).st == adm::state::copying, "empty -> copying");
    expect(!t.start(i1, 2 * 512 + 7) && t.in_flight(2 * 512 + 7),
           "a key copying refused a second slot (a repeat miss in the same pass)");
    expect(!t.start(i0, 2 * 512 + 8), "a record copying refused a second start");
    int32_t s = -1, k = -1;
    expect(!t.publish(i0, s, k), "a record copying not published (its copy unproven)");
    expect(t.land(i0) && t.at(i0).st == adm::state::landed && !t.land(i0), "copying -> landed, once");
    expect(t.publish(i0, s, k) && s == 100 && k == 2 * 512 + 7 && t.at(i0).st == adm::state::none,
           "landed -> published: the slot and key returned, the record free");
    expect(!t.in_flight(2 * 512 + 7) && t.held(2) == 2, "a published key no longer in flight; the slot left the table");
    expect(t.start(i1, 2 * 512 + 9) && t.invalidate(i1) && t.at(i1).st == adm::state::empty && t.at(i1).key == -1 &&
               t.at(i1).gen > g_before,
           "copying -> empty on invalidation, a new generation, the slot kept");
    expect(t.start(i2, 2 * 512 + 10) && t.land(i2) && t.invalidate(i2) && t.at(i2).st == adm::state::empty,
           "landed -> empty on invalidation (a failure after the copy)");
    // a loan returns what it borrowed: every reservation's bytes garbage
    expect(t.start(i1, 2 * 512 + 11), "a start before the loan");
    const uint32_t g1 = t.at(i1).gen, g2 = t.at(i2).gen;
    t.return_from_loan();
    expect(t.at(i1).st == adm::state::empty && t.at(i2).st == adm::state::empty && t.at(i1).gen > g1 &&
               t.at(i2).gen > g2 && t.held(2) == 2 && !t.in_flight(2 * 512 + 11),
           "a loan's return: the reservations kept, empty, new generations");
    // reset to the boot placement
    const std::vector<int32_t> back = t.discard_all();
    expect(back.size() == 2 && t.held_all() == 0, "discard: every slot returned, no record left (no holes)");
}

void choices() {
    std::vector<adm::cand> c = {{10, 512 + 3, 0.5f}, {11, 512 + 1, 0.5f}, {12, 512 + 2, 0.1f}, {13, 512 + 0, 2.0f}};
    const std::vector<adm::cand> v = adm::victims(c, 2);
    expect(v.size() == 2 && v[0].slot == 12 && v[1].slot == 11, "victims: least usage, then key");
    expect(adm::victims(c, 9).size() == 4 && adm::victims(c, 0).empty() && adm::victims({}, 3).empty(),
           "victims: the need clamped; none from none");
    std::vector<adm::cand> tie = {{21, 7, 1.0f}, {20, 7, 1.0f}};
    expect(adm::victims(tie, 1)[0].slot == 20, "victims: slot breaks a full tie");
    std::vector<adm::miss> m = {{4, 1, 0}, {9, 5, 1}, {2, 2, 2}, {7, 5, 3}, {1, 1, 4}};
    const std::vector<adm::miss> a = adm::admit(m, 3);
    expect(a.size() == 3 && a[0].u == 9 && a[1].u == 7 && a[2].u == 2, "admit: most rows, then route order");
    expect(adm::admit(m, 0).empty() && adm::admit({}, 3).empty() && adm::admit(m, 9).size() == 5,
           "admit: exhausted, empty, clamped");
}

void property() {
    std::mt19937 g(20261018u);
    adm::table t;
    std::uniform_int_distribution<int32_t> layer(0, 47), op(0, 7), slot(0, 4000), e(0, 511);
    bool ok = true, gens = true, admits_subset = true;
    uint32_t last_gen = t.gen();
    for (int32_t it = 0; it < 200000; ++it) {
        const int32_t l = layer(g), i = l * adm::per_layer + (it % adm::per_layer);
        switch (op(g)) {
            case 0:
            case 1:
                t.reserve(l, slot(g));
                break;
            case 2:
            case 3:
                t.start(i, l * 512 + e(g));
                break;
            case 4:
                t.land(i);
                break;
            case 5: {
                int32_t s = -1, k = -1;
                t.publish(i, s, k);
                break;
            }
            case 6:
                t.invalidate(i);
                break;
            default:
                if (it % 997 == 0) {
                    t.return_from_loan();
                }
                if (it % 4999 == 0) {
                    t.discard_all();
                }
        }
        ok = ok && invariants(t);
        gens = gens && t.gen() >= last_gen;
        last_gen = t.gen();
        if (it % 1000 == 0) { // admit picks only among its candidates, at most n_avail
            std::vector<adm::miss> m;
            for (int32_t u = 0; u < 12; ++u) {
                m.push_back({u, 1 + int32_t(g() % 5), u});
            }
            const int32_t n = int32_t(g() % 5);
            const std::vector<adm::miss> a = adm::admit(m, n);
            admits_subset = admits_subset && int32_t(a.size()) == std::min(n, 12);
            for (const adm::miss & x : a) {
                admits_subset = admits_subset && x.u >= 0 && x.u < 12;
            }
        }
    }
    expect(ok, "random operations keep one record a slot, per_layer a layer, a key in flight once");
    expect(gens, "generations never go back");
    expect(admits_subset, "admit returns only candidates (no transfer for admission alone), at most n_avail");
}

// ME4's R4-12 follow-up (Fable's merge note): the refill's batches. prompt_loan::give_back stable-sorts its moves by
// layer, and loan::batch_end bounds a batch by its first move's layer plus refill_batch_layers. So at 1 a batch is
// one layer's moves: R4-12's 288 moves in one batch were one layer's, and a loan over two or more layers leaves moves
// outstanding behind a one-batch hold (R7-05: 869 slots in 3 batches, 534 carried)
void refill_batches() {
    std::mt19937 g(4012u);
    std::vector<std::pair<int32_t, int32_t>> moves; // (slot, key)
    for (int32_t i = 0; i < 400; ++i) {
        moves.emplace_back(i, int32_t(g() % 6) * 512 + int32_t(g() % 512)); // layers 0..5, unordered
    }
    std::stable_sort(moves.begin(), moves.end(),
                     [](const auto & a, const auto & b) { return a.second / 512 < b.second / 512; }); // give_back's
    const auto layer_of = [&](size_t i) { return moves[i].second / 512; };
    bool one_layer = true, covers = true;
    int32_t batches = 0;
    for (size_t a = 0; a < moves.size();) {
        const size_t b = fn::runtime::loan::batch_end(moves.size(), a, 1, layer_of);
        one_layer = one_layer && b > a && layer_of(b - 1) == layer_of(a) &&
                    (b == moves.size() || layer_of(b) > layer_of(a));
        a = b;
        ++batches;
    }
    covers = batches == 6;
    std::vector<std::pair<int32_t, int32_t>> same(288, {0, 7 * 512 + 3}); // R4-12: every move of one layer
    const size_t b288 = fn::runtime::loan::batch_end(same.size(), 0, 1, [&](size_t i) { return same[i].second / 512; });
    const size_t b2 = fn::runtime::loan::batch_end(moves.size(), 0, 2, layer_of);
    expect(one_layer && covers, "refill batches at 1 layer: each batch exactly one layer's moves, in layer order");
    expect(b288 == 288, "R4-12's case: 288 moves of one layer are one batch (nothing left behind a one-batch hold)");
    expect(b2 > 0 && layer_of(b2 - 1) <= layer_of(0) + 1 && (b2 == moves.size() || layer_of(b2) >= layer_of(0) + 2),
           "refill batches at 2 layers: the first two layers' moves");
    // MX13: 4 layers and at most 64 MiB a device, whole layers only (1,766,400-byte moves): 16 moves a layer (28.3 MB)
    // pack two whole layers a batch and stop before the third; 60 a layer (106 MB, above the cap) are one layer a
    // batch, 48 batches over 48 layers (MX3's mid-layer split gave 76-77); the batches tile the moves in order; 0 is
    // no cap
    const int64_t cap = int64_t(64) << 20;
    const auto de = [](size_t) { return int64_t(1766400); };
    const auto tile = [&](int32_t per_layer, bool & whole, int32_t & batches, size_t & first) {
        std::vector<int32_t> lay;
        for (int32_t l = 0; l < 48; ++l) {
            lay.insert(lay.end(), size_t(per_layer), l);
        }
        const auto lof = [&](size_t i) { return lay[i]; };
        whole = true;
        batches = 0;
        first = 0;
        for (size_t a = 0; a < lay.size();) {
            const size_t b = fn::runtime::loan::batch_end_bytes(lay.size(), a, 4, cap, lof, de);
            const size_t l = fn::runtime::loan::batch_end(lay.size(), a, 4, lof);
            whole = whole && b > a && b <= l && (b == lay.size() || lof(b) != lof(b - 1)) &&
                    (a == 0 || lof(a) != lof(a - 1));
            first = first == 0 ? b : first;
            a = b;
            ++batches;
        }
    };
    bool whole16 = false, whole60 = false;
    int32_t n16 = 0, n60 = 0;
    size_t first16 = 0, first60 = 0;
    tile(16, whole16, n16, first16);
    tile(60, whole60, n60, first60);
    expect(whole16 && first16 == 32 && n16 == 24,
           "refill batches (MX13) at 4 layers and 64 MiB: 16 moves a layer pack two whole layers (56.5 MB) a batch");
    expect(whole60 && first60 == 60 && n60 == 48,
           "refill batches (MX13): 60 moves a layer (106 MB, above the cap) are one whole layer a batch, 48 batches");
    const size_t none = fn::runtime::loan::batch_end_bytes(moves.size(), 0, 4, 0, layer_of, de);
    const size_t one = fn::runtime::loan::batch_end_bytes(moves.size(), 0, 4, 1000, layer_of, de);
    expect(none == fn::runtime::loan::batch_end(moves.size(), 0, 4, layer_of) && one > 0 &&
               layer_of(one - 1) == layer_of(0) && (one == moves.size() || layer_of(one) > layer_of(0)),
           "refill batches (MX13): 0 is no cap; a cap below one layer's bytes still takes that whole layer");
}

// Shell 0a1b4eb5: the counters record's tail (counters_tail_json, the admission member then the share attempts) parses
// as JSON closing a record, with an empty log and with a populated one (a reading not finite, a reason missing, a
// window from `from`); the validator itself refuses R9-08's shape (a member inside the open array)
void records_json() {
    expert_cache_counters c;
    c.adm_started = 248;
    c.adm_published = 248;
    c.adm_off_other = 2;
    std::vector<share_attempt> log(3);
    log[0].round = 4;
    log[0].r.update = 1;
    log[0].r.samples = 96;
    log[0].r.link_gbps = 6.67;
    log[0].r.reason = "first";
    log[1].round = 8;
    log[1].r.tc_us = std::numeric_limits<double>::quiet_NaN();
    log[1].r.rate_gbps = std::numeric_limits<double>::infinity();
    log[2].round = 12;
    log[2].r.reason = "third";
    const auto valid = [](const std::string & t) {
        size_t at = 0;
        const bool ok = json_valid(t, at);
        if (!ok) {
            std::printf("ex-cpu-test: me9 record not JSON at byte %zu: %s\n", at, t.c_str());
        }
        return ok;
    };
    const std::string head = "{\"devices\": 1, \"share\": 0.300, ";
    expect(valid(head + counters_tail_json(0, c, {}, 0)), "counters record, no share attempts: JSON");
    expect(valid(head + counters_tail_json(144, c, log, 0)), "counters record, three share attempts: JSON");
    expect(valid(head + counters_tail_json(3, c, log, 1)), "counters record, the attempts from the second: JSON");
    expect(valid(head + counters_tail_json(3, c, log, 3)), "counters record, a request with none of its own: JSON");
    expect(counters_tail_json(0, c, {}, 0).find("\"share_attempts\": []}") != std::string::npos &&
               counters_tail_json(0, c, {}, 0).rfind("\"admission\": {", 0) == 0,
           "admission before the attempts' array, the array closing the record");
    size_t at = 0;
    expect(!json_valid("{\"share_attempts\": [\"admission\": {\"started\": 0}]}", at),
           "the validator refuses a member inside an array (R9-08's record)");
    expect(!json_valid("{\"a\": nan}", at) && !json_valid("{\"a\": 1,}", at) && !json_valid("{\"a\": 1} x", at) &&
               json_valid("{\"a\": [1, -2.5e3, \"b\\n\", true, null, {}]}", at),
           "the validator: non-numbers, trailing commas and trailing text refused; a mixed value accepted");
}

} // namespace

bool admission_test() {
    n_pass = n_fail = 0;
    transitions();
    choices();
    property();
    refill_batches();
    records_json();
    std::printf("ex-cpu-test: me9 admission: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0;
}

} // namespace fn::runtime
