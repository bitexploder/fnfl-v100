#include "fn/runtime/safety_test.h"

#include "fn/runtime/stage_wire.h"
#include "fn/xchg/life.h"
#include "fn/xchg/link.h"
#include "fn/xchg/link_fault.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace fn::runtime {

namespace {

namespace lk = fn::xchg::link;
namespace fl = fn::xchg::link::fault;
namespace st = fn::runtime::stage;

constexpr double quick = 0.2;  // a wait meant to fail (and the link's retire bound)
constexpr double slow  = 5.0;  // a wait meant to succeed

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// word i of the payload of (slot, request generation, sequence, card)
uint32_t pat(uint64_t slot, uint64_t gen, uint64_t seq, uint64_t card, uint64_t i) {
    return uint32_t(mix(mix(mix(mix(mix(slot) ^ gen) ^ seq) ^ card) ^ i));
}

void fill(void * p, size_t n, uint64_t slot, uint64_t gen, uint64_t seq, uint64_t card) {
    for (size_t i = 0; i < n / 4; ++i) {
        const uint32_t w = pat(slot, gen, seq, card, i);
        std::memcpy(static_cast<uint8_t *>(p) + i * 4, &w, 4);
    }
}

bool same(const void * p, size_t n, uint64_t slot, uint64_t gen, uint64_t seq, uint64_t card) {
    for (size_t i = 0; i < n / 4; ++i) {
        uint32_t w = 0;
        std::memcpy(&w, static_cast<const uint8_t *>(p) + i * 4, 4);
        if (w != pat(slot, gen, seq, card, i)) {
            return false;
        }
    }
    return true;
}

// an X-STEP answer that names its owner: slot, request generation, sequence
st::step_out answer(int32_t slot, int32_t gen, uint32_t seq) {
    st::step_out o;
    o.slot   = slot;
    o.status = 0;
    o.ran    = 1;
    o.n_out  = 1;
    o.out[0] = slot * 100000 + gen * 1000 + int32_t(seq);
    std::snprintf(o.why, sizeof o.why, "slot %d generation %d seq %u", slot, gen, seq);
    return o;
}

bool is_answer(const st::step_out & o, int32_t slot, int32_t gen, uint32_t seq) {
    const st::step_out w = answer(slot, gen, seq);
    return std::memcmp(&o, &w, sizeof o) == 0;
}

// the two ends over the test network: L the leader (the production wire paths), F the follower (scripted here)
struct pair_t {
    lk::link                  L, F;
    std::unique_ptr<st::port> pl, pf;

    bool start(std::string & err) {
        lk::options o;
        o.timeout_s   = quick;
        o.retire_s    = quick;
        o.me.slots    = 8;
        lk::options f = o;
        f.leader      = false;
        if (!fl::pair(L, F, o, f, err)) {
            return false;
        }
        pl = st::link_port(L);
        pf = st::link_port(F);
        return true;
    }
};

// the follower's scripted receive and send (its buffers live for the whole case)
bool frecv(lk::link & F, uint64_t tag, void * p, size_t n, double t) {
    std::string   e;
    const int32_t op = F.recv(tag, p, n, e);
    return op >= 0 && F.wait(op, e, t);
}

int32_t fsend(lk::link & F, uint64_t tag, const void * p, size_t n) {
    std::string e;
    return F.send(tag, p, n, e);
}

// polls (each call observes the link's ends) until done, bounded
bool settle(lk::link & l, const std::function<bool()> & done, double bound_s) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const bool idle = l.held() == 0;
        if (idle && done()) {
            return true;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > bound_s) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void poll_until(const std::function<bool()> & done, double bound_s) {
    const auto t0 = std::chrono::steady_clock::now();
    while (!done() && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < bound_s) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

st::member begin_member(int32_t slot) {
    st::member m;
    m.slot = slot;
    m.kind = int32_t(st::mkind::begin);
    return m;
}

st::member generate_member(int32_t slot) {
    st::member m;
    m.slot = slot;
    m.kind = int32_t(st::mkind::generate);
    m.rows = 1;
    return m;
}

// a command whose ack and tokens live in this frame (the base's callers' stack records)
__attribute__((noinline)) bool stack_command(st::control & ctl, double t) {
    int32_t tokens[3] = { 7, 8, 9 };
    st::ack a;
    life::own(&a, sizeof a, "a caller's stack ack");
    life::own(tokens, sizeof tokens, "a caller's stack tokens");
    st::ctrl c;
    c.op          = int32_t(st::op::set_sampling);
    c.slot        = 1;
    c.n_tokens    = 3;
    const bool ok = ctl.command(c, tokens, a, t);
    life::disown(tokens, "a caller's stack tokens");
    life::disown(&a, "a caller's stack ack");
    return ok;
}

// the ledger over one case: every owner and user the case made ends inside it
struct mark {
    life::totals t0 = life::get();

    bool clean(std::string & d) const {
        const life::totals t  = life::get();
        const int64_t      v  = t.violations - t0.violations;
        const bool         ok = v == 0 && t.live_users == t0.live_users && t.live_owners == t0.live_owners;
        if (life::on) {
            d += "; ledger: " + std::to_string(t.uses - t0.uses) + " uses, " + std::to_string(t.dones - t0.dones) +
                 " ended, " + std::to_string(v) + " violations, users left " +
                 std::to_string(t.live_users - t0.live_users) + ", owners left " +
                 std::to_string(t.live_owners - t0.live_owners);
            if (v > 0) {
                d += "; last: " + life::last();
            }
        }
        return ok;
    }
};

// ---- the cases ----

bool c_normal(std::string & d) {
    mark                 lm;
    constexpr size_t     B = 3000;
    std::vector<uint8_t> bnd[4][2], fb[4][2];
    for (int32_t s = 0; s < 4; ++s) {
        for (int32_t c = 0; c < 2; ++c) {
            bnd[s][c].assign(B, 0);
            fb[s][c].assign(B, 0);
            life::own(bnd[s][c].data(), B, "slot boundary buffer");
            fill(bnd[s][c].data(), B, uint64_t(s), 1, 0, uint64_t(c));
        }
    }
    std::vector<st::step_out> fo;
    fo.reserve(8);
    bool ok = true;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err)) {
            d = err;
            return false;
        }
        st::control ctl;
        ctl.init(*P.pl);
        ctl.send_timeout_s = slow;
        st::leader_stats stats;
        st::member       m[2][2] = {
            { generate_member(0), generate_member(1) },
            { generate_member(2), generate_member(3) }
        };
        st::boundary_src src[2][2];
        for (int32_t g = 0; g < 2; ++g) {
            for (int32_t k = 0; k < 2; ++k) {
                const int32_t s  = m[g][k].slot;
                src[g][k].buf[0] = bnd[s][0].data();
                src[g][k].buf[1] = bnd[s][1].data();
                src[g][k].bytes  = B;
            }
        }
        std::shared_ptr<st::group_wire> w[2] = { std::shared_ptr<st::group_wire>(new st::group_wire()),
                                                 std::shared_ptr<st::group_wire>(new st::group_wire()) };
        ok                                   = st::post_group(ctl, w[0], 0, m[0], 2, src[0], nullptr, stats) &&
                                               st::post_group(ctl, w[1], 1, m[1], 2, src[1], nullptr, stats);
        // the follower: group 1 first, then group 0
        std::vector<int32_t> fops;
        for (int32_t g = 1; g >= 0 && ok; --g) {
            const uint32_t seq = uint32_t(g + 1);
            st::ctrl       rc;
            ok = frecv(P.F, st::ctrl_tag(seq, 0), &rc, sizeof rc, slow) && rc.op == int32_t(st::op::step_group) &&
                 rc.group == g && rc.n_members == 2 && rc.m[0].slot == m[g][0].slot && rc.m[1].slot == m[g][1].slot;
            for (int32_t k = 0; k < 2 && ok; ++k) {
                const int32_t s = m[g][k].slot;
                for (int32_t c = 0; c < 2 && ok; ++c) {
                    ok = frecv(P.F, st::boundary_tag(g, seq, c, s), fb[s][c].data(), B, slow) &&
                         same(fb[s][c].data(), B, uint64_t(s), 1, 0, uint64_t(c));
                }
            }
            for (int32_t k = 0; k < 2 && ok; ++k) {
                fo.push_back(answer(m[g][k].slot, 1, seq));
                fops.push_back(fsend(P.F, st::step_tag(g, seq, m[g][k].slot), &fo.back(), sizeof(st::step_out)));
            }
        }
        if (!ok) {
            d = "the follower's side of the exchange failed";
        }
        // the leader consumes group 1 before group 0
        for (int32_t g = 1; g >= 0 && ok; --g) {
            std::string e;
            ok = st::group_sends(ctl, *w[g], slow);
            for (int32_t k = 0; k < 2 && ok; ++k) {
                ok = st::group_answer(ctl, *w[g], k, slow, e) &&
                     is_answer(w[g]->out[k], m[g][k].slot, 1, uint32_t(g + 1));
                if (!ok) {
                    d = "group " + std::to_string(g) + " member " + std::to_string(k) + ": " +
                        (e.empty() ? "a foreign answer: " + std::string(w[g]->out[k].why) : e);
                }
            }
            st::group_close(ctl, *w[g]);
            // the group's sends finished: its slots' boundary buffers may be refilled
            for (int32_t k = 0; k < 2 && ok; ++k) {
                for (int32_t c = 0; c < 2; ++c) {
                    life::reuse(bnd[m[g][k].slot][c].data(), B, "slot boundary buffer (the next round)");
                }
            }
        }
        for (int32_t op : fops) {
            std::string e;
            ok = ok && op >= 0 && P.F.wait(op, e, slow);
        }
        w[0].reset();
        w[1].reset();
        P.L.stop();
        P.F.stop();
    }
    for (int32_t s = 0; s < 4; ++s) {
        for (int32_t c = 0; c < 2; ++c) {
            life::disown(bnd[s][c].data(), "slot boundary buffer");
        }
    }
    if (ok) {
        d = "4 members' boundaries and answers matched their own (slot, generation, sequence, card) bytes";
    }
    return lm.clean(d) && ok;
}

bool c_partial(std::string & d) {
    mark                 lm;
    constexpr size_t     B = 2048;
    std::vector<uint8_t> bnd[2][2];
    std::vector<uint8_t> fbuf(sizeof(st::ctrl));
    st::step_out         late = answer(0, 1, 1);
    for (int32_t s = 0; s < 2; ++s) {
        for (int32_t c = 0; c < 2; ++c) {
            bnd[s][c].assign(B, 0);
            life::own(bnd[s][c].data(), B, "slot boundary buffer");
        }
    }
    bool ok = true;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err)) {
            d = err;
            return false;
        }
        st::control ctl;
        ctl.init(*P.pl);
        ctl.send_timeout_s = slow;
        st::leader_stats stats;
        st::member       m[2] = { generate_member(0), generate_member(1) };
        st::boundary_src src[2];
        for (int32_t k = 0; k < 2; ++k) {
            src[k].buf[0] = bnd[k][0].data();
            src[k].buf[1] = bnd[k][1].data();
            src[k].bytes  = B;
        }
        auto                                w  = std::shared_ptr<st::group_wire>(new st::group_wire());
        const std::weak_ptr<st::group_wire> ww = w;
        // submissions: receive 1, receive 2, the X-CTRL send, then the first boundary: refused
        fl::fail_submit(P.L, 4);
        const bool posted = st::post_group(ctl, w, 0, m, 2, src, nullptr, stats);
        st::group_sends(ctl, *w, quick);
        st::group_close(ctl, *w);
        w.reset();  // the pipeline's end_group drops the group
        // the follower takes the X-CTRL now (a send still outstanding would read the dropped record), then answers
        // member 0 (a receive still posted would write it)
        frecv(P.F, st::ctrl_tag(1, 0), fbuf.data(), fbuf.size(), quick);
        const int32_t fop = fsend(P.F, st::step_tag(0, 1, 0), &late, sizeof late);
        const bool    rec = settle(P.L, [&]() { return ww.expired(); }, 2.0);
        if (fop >= 0) {
            std::string e;
            P.F.wait(fop, e, quick);
        }
        ok = !posted && ctl.lost() && rec;
        d  = std::string("post_group ") + (posted ? "succeeded (expected the injected refusal)" : "refused") +
             "; the stage " + (ctl.lost() ? "lost" : "not lost") + "; the record " +
             (rec ? "reclaimed" : "not reclaimed");
        P.L.stop();
        P.F.stop();
    }
    for (int32_t s = 0; s < 2; ++s) {
        for (int32_t c = 0; c < 2; ++c) {
            life::disown(bnd[s][c].data(), "slot boundary buffer");
        }
    }
    return lm.clean(d) && ok;
}

bool c_queued(std::string & d) {
    mark                 lm;
    std::vector<int32_t> ft(3, 0);
    st::ack              fa;
    fa.status = 0x5a;
    bool ok   = true;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err)) {
            d = err;
            return false;
        }
        st::control ctl;
        ctl.init(*P.pl);
        ctl.send_timeout_s = quick;
        fl::hold_thread(P.L, true);
        fl::wait_parked(P.L);
        const bool r = stack_command(ctl, quick);  // its operations queued, never posted while it waits
        fl::hold_thread(P.L, false);
        // the follower takes the tokens and answers X-ACK 1: an operation posted late would touch the dead frame
        frecv(P.F, st::ctrl_tag(1, 1), ft.data(), 12, quick);
        const int32_t fop = fsend(P.F, st::ack_tag(1), &fa, sizeof fa);
        settle(P.L, []() { return true; }, 1.0);
        if (fop >= 0) {
            std::string e;
            P.F.wait(fop, e, quick);
        }
        ok = !r && ctl.lost();
        d  = std::string("the command ") + (r ? "succeeded (expected a timeout)" : "timed out") + "; the stage " +
             (ctl.lost() ? "lost" : "not lost");
        P.L.stop();
        P.F.stop();
    }
    return lm.clean(d) && ok;
}

bool c_posted(bool stuck, std::string & d) {
    mark         lm;
    st::step_out o = answer(2, 1, 1);
    st::ctrl     rc;
    bool         ok = true;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err)) {
            d = err;
            return false;
        }
        st::control ctl;
        ctl.init(*P.pl);
        ctl.send_timeout_s = slow;
        st::leader_stats                    stats;
        const st::member                    m  = begin_member(2);
        auto                                w  = std::shared_ptr<st::group_wire>(new st::group_wire());
        const std::weak_ptr<st::group_wire> ww = w;
        ok = st::post_group(ctl, w, 0, &m, 1, nullptr, nullptr, stats) &&
             frecv(P.F, st::ctrl_tag(1, 0), &rc, sizeof rc, slow) && st::group_sends(ctl, *w, slow);
        fl::hold_transfers(P.L, ~uint64_t(0), st::step_tag(0, 1, 2), stuck);
        const int32_t fop = fsend(P.F, st::step_tag(0, 1, 2), &o, sizeof o);
        poll_until([&]() { return fl::held_transfers(P.L) == 1; }, slow);
        fl::hold_thread(P.L, true);  // before the cancel
        fl::wait_parked(P.L);
        std::string e;
        const bool  got = st::group_answer(ctl, *w, 0, quick, e);
        st::group_close(ctl, *w);
        w.reset();  // the caller drops the record early
        const bool kept = !ww.expired();
        fl::hold_thread(P.L, false);
        if (stuck) {
            settle(P.L, []() { return false; }, quick);  // the link's quiesce had its turn; the copy is still held
            fl::release_transfers(P.L);
        }
        const bool rec = settle(P.L, [&]() { return ww.expired(); }, 2.0);
        if (fop >= 0) {
            std::string fe;
            P.F.wait(fop, fe, quick);
        }
        const int64_t copied = fl::copies(P.L), dropped = fl::dropped(P.L);
        ok = ok && !got && rec && (stuck ? copied > 0 : dropped > 0);
        d  = std::string(stuck ? "stuck" : "not stuck") + ": the answer's wait " + (got ? "succeeded" : "failed") +
             "; the record " + (kept ? "kept by the link" : "not kept") + " after the caller dropped it, then " +
             (rec ? "reclaimed" : "not reclaimed") + "; the held copy " +
             (copied > 0  ? "landed later" :
              dropped > 0 ? "ended without a copy" :
                            "never ended");
        P.L.stop();
        P.F.stop();
    }
    return lm.clean(d) && ok;
}

bool c_race(std::string & d) {
    mark                 lm;
    std::vector<uint8_t> rb(512, 0), sb(512, 0), b1(64, 0), b2(64, 0), s1(64, 0), s2(64, 0);
    life::own(rb.data(), rb.size(), "a receive buffer");
    life::own(b1.data(), b1.size(), "a receive buffer");
    life::own(b2.data(), b2.size(), "a receive buffer");
    fill(sb.data(), sb.size(), 5, 1, 1, 0);
    fill(s1.data(), s1.size(), 6, 1, 1, 0);
    fill(s2.data(), s2.size(), 6, 2, 2, 0);
    bool        raced = false, stale_ok = false;
    std::string sd;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err)) {
            d = err;
            return false;
        }
        // a completion racing the cancel: matched and held; the cancel's hook releases it just before the cancel
        const uint64_t T = lk::make_tag(lk::kind::test, 0, 1, 0, 0);
        fl::hold_transfers(P.L, ~uint64_t(0), T, false);
        std::string   e;
        const int32_t op  = P.L.recv(T, rb.data(), rb.size(), e);
        const int32_t fop = fsend(P.F, T, sb.data(), sb.size());
        poll_until([&]() { return fl::held_transfers(P.L) == 1; }, slow);
        fl::hold_thread(P.L, true);
        fl::wait_parked(P.L);
        fl::on_cancel(P.L, [&P]() { fl::release_transfers(P.L); });
        fl::hold_thread(P.L, false);
        const bool got = op >= 0 && P.L.wait(op, e, quick);
        raced          = got && same(rb.data(), rb.size(), 5, 1, 1, 0);
        sd             = std::string("the raced wait ") + (got ? "reported the completion" : "failed: " + e);
        settle(P.L, []() { return true; }, 1.0);
        if (fop >= 0) {
            std::string fe;
            P.F.wait(fop, fe, slow);
        }
        // a stale handle: op1 finished, its slot reused by op2; waiting on op1 must not take op2's completion
        const uint64_t T1 = lk::make_tag(lk::kind::test, 1, 1, 0, 0), T2 = lk::make_tag(lk::kind::test, 1, 2, 0, 0);
        const int32_t  op1 = P.L.recv(T1, b1.data(), b1.size(), e);
        const int32_t  f1  = fsend(P.F, T1, s1.data(), s1.size());
        const bool     w1  = op1 >= 0 && P.L.wait(op1, e, slow);
        const int32_t  op2 = P.L.recv(T2, b2.data(), b2.size(), e);
        const int32_t  f2  = fsend(P.F, T2, s2.data(), s2.size());
        std::string    fe;
        const bool     sent2 = f2 >= 0 && P.F.wait(f2, fe, slow);  // op2 has completed (not yet waited)
        std::string    se;
        const bool     stale = P.L.wait(op1, se, quick);
        const bool     fresh = op2 >= 0 && P.L.wait(op2, e, slow);
        stale_ok             = w1 && sent2 && !stale && se.find("stale") != std::string::npos && fresh &&
                               same(b2.data(), b2.size(), 6, 2, 2, 0);
        sd += std::string("; the stale handle's wait ") +
              (stale ? "returned the newer operation's completion" : "refused: " + se) + "; the newer operation " +
              (fresh ? "completed with its own bytes" : "failed: " + e);
        if (f1 >= 0) {
            P.F.wait(f1, fe, slow);
        }
        P.L.stop();
        P.F.stop();
    }
    life::disown(rb.data(), "a receive buffer");
    life::disown(b1.data(), "a receive buffer");
    life::disown(b2.data(), "a receive buffer");
    d = sd;
    return lm.clean(d) && raced && stale_ok;
}

bool c_final(std::string & d) {
    mark                      lm;
    std::vector<st::step_out> fo;
    fo.reserve(8);
    std::vector<uint8_t> longer(sizeof(st::step_out) + 8, 0x77);
    st::ctrl             rc[2];
    bool                 ok = true;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err)) {
            d = err;
            return false;
        }
        st::control ctl;
        ctl.init(*P.pl);
        ctl.send_timeout_s = slow;
        st::leader_stats                    stats;
        const st::member                    ma[3] = { begin_member(0), begin_member(1), begin_member(2) };
        const st::member                    mb[2] = { begin_member(4), begin_member(5) };
        auto                                wa    = std::shared_ptr<st::group_wire>(new st::group_wire());
        auto                                wb    = std::shared_ptr<st::group_wire>(new st::group_wire());
        const std::weak_ptr<st::group_wire> wwa = wa, wwb = wb;
        ok                  = st::post_group(ctl, wa, 0, ma, 3, nullptr, nullptr, stats) &&
                              st::post_group(ctl, wb, 1, mb, 2, nullptr, nullptr, stats) &&
                              frecv(P.F, st::ctrl_tag(1, 0), &rc[0], sizeof rc[0], slow) &&
                              frecv(P.F, st::ctrl_tag(2, 0), &rc[1], sizeof rc[1], slow);
        // group B's answers (group 1, seq 2) are held at the leader once matched
        const uint64_t mask = ~uint64_t(0xffff);
        fl::hold_transfers(P.L, mask, st::step_tag(1, 2, 0) & mask, false);
        std::vector<int32_t> fops;
        fo.push_back(answer(0, 1, 1));
        fops.push_back(fsend(P.F, st::step_tag(0, 1, 0), &fo.back(), sizeof(st::step_out)));
        fo.push_back(answer(1, 1, 1));
        fops.push_back(fsend(P.F, st::step_tag(0, 1, 1), &fo.back(), sizeof(st::step_out)));
        fops.push_back(fsend(P.F, st::step_tag(0, 1, 2), longer.data(), longer.size()));  // the last member's fails
        fo.push_back(answer(4, 1, 2));
        fops.push_back(fsend(P.F, st::step_tag(1, 2, 4), &fo.back(), sizeof(st::step_out)));
        fo.push_back(answer(5, 1, 2));
        fops.push_back(fsend(P.F, st::step_tag(1, 2, 5), &fo.back(), sizeof(st::step_out)));
        poll_until([&]() { return fl::held_transfers(P.L) == 2; }, slow);
        std::string e0, e1, e2;
        const bool  a0   = st::group_sends(ctl, *wa, slow) && st::group_answer(ctl, *wa, 0, slow, e0);
        const bool  a1   = a0 && st::group_answer(ctl, *wa, 1, slow, e1);
        const bool  a2   = a1 && st::group_answer(ctl, *wa, 2, slow, e2);
        const bool  mine = is_answer(wa->out[0], 0, 1, 1) && is_answer(wa->out[1], 1, 1, 1);
        st::group_close(ctl, *wa);
        wa.reset();
        // group B ends after A failed (the pipeline's next end)
        std::string eb;
        const bool  b_ok      = st::group_sends(ctl, *wb, slow) && st::group_answer(ctl, *wb, 0, quick, eb) &&
                                st::group_answer(ctl, *wb, 1, quick, eb);
        const bool  untouched = wb->out[0].slot == -1 && wb->out[1].slot == -1;
        st::group_close(ctl, *wb);
        wb.reset();
        fl::release_transfers(P.L);
        const bool rec = settle(P.L, [&]() { return wwa.expired() && wwb.expired(); }, 2.0);
        for (int32_t op : fops) {
            std::string e;
            if (op >= 0) {
                P.F.wait(op, e, quick);
            }
        }
        ok = ok && a1 && !a2 && mine && !b_ok && untouched && rec;
        d  = std::string("group A's members 0 and 1 ") + (mine ? "got their own answers" : "did not get their own") +
             "; member 2 " + (a2 ? "succeeded (expected a refusal)" : "refused: " + e2) + "; group B " +
             (b_ok ? "succeeded" : "failed (" + eb + ")") + ", its records " +
             (untouched ? "untouched by A's bytes" : "written") + "; both records " +
             (rec ? "reclaimed" : "not reclaimed");
        P.L.stop();
        P.F.stop();
    }
    return lm.clean(d) && ok;
}

bool c_ack(std::string & d) {
    mark                 lm;
    std::vector<int32_t> ft(3, 0);
    st::ack              fa;
    fa.status = 0x5a;
    fa.a      = 4242;
    st::ctrl rc;
    bool     toks = false, ok = true;
    int32_t  fop = -1;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err)) {
            d = err;
            return false;
        }
        st::control ctl;
        ctl.init(*P.pl);
        ctl.send_timeout_s = slow;
        fl::hold_transfers(P.L, ~uint64_t(0), st::ack_tag(1), false);  // the X-ACK's copy waits at the leader
        std::thread follower([&]() {
            toks = frecv(P.F, st::ctrl_tag(1, 0), &rc, sizeof rc, slow) &&
                   frecv(P.F, st::ctrl_tag(1, 1), ft.data(), 12, slow) && ft[0] == 7 && ft[1] == 8 && ft[2] == 9;
            fop  = fsend(P.F, st::ack_tag(1), &fa, sizeof fa);
        });
        const bool  r = stack_command(ctl, quick);  // its wait for the X-ACK fails; its frame ends
        follower.join();
        const int64_t c0   = fl::copies(P.L);
        const int32_t held = fl::held_transfers(P.L);
        fl::release_transfers(P.L);  // a receive still posted would write the dead frame now
        const bool rec       = settle(P.L, []() { return true; }, 2.0);
        const bool exercised = held == 1 || fl::dropped(P.L) > 0 || fl::copies(P.L) > c0;
        if (fop >= 0) {
            std::string e;
            P.F.wait(fop, e, quick);
        }
        ok = !r && toks && rec && exercised;
        d  = std::string("the X-ACK ") + (exercised ? "was matched and held past the wait" : "was never matched") +
             "; the follower " + (toks ? "received the X-CTRL and tokens 7, 8, 9" : "missed the tokens") +
             "; the command " + (r ? "succeeded (expected the X-ACK's timeout)" : "failed at its X-ACK") +
             "; every operation " + (rec ? "ended" : "not ended");
        P.L.stop();
        P.F.stop();
    }
    return lm.clean(d) && ok;
}

bool c_shutdown(std::string & d) {
    mark                 lm;
    constexpr size_t     B = 1024;
    std::vector<uint8_t> mb(4 * B, 0), mb2(512, 0);
    life::own(mb.data(), mb.size(), "mapped boundary buffer");
    life::own(mb2.data(), mb2.size(), "mapped buffer");
    bool        lost_ok = false, stop_ok = false;
    std::string sd;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err) || !P.L.map(mb.data(), mb.size(), err)) {
            d = err;
            return false;
        }
        st::control ctl;
        ctl.init(*P.pl);
        ctl.send_timeout_s = slow;
        st::leader_stats stats;
        const st::member m = generate_member(0);
        st::boundary_src src;
        src.buf[0]                              = mb.data();
        src.buf[1]                              = mb.data() + B;
        src.bytes                               = B;
        auto                                w   = std::shared_ptr<st::group_wire>(new st::group_wire());
        const std::weak_ptr<st::group_wire> ww  = w;
        const bool                          pst = st::post_group(ctl, w, 0, &m, 1, &src, nullptr, stats);
        fl::kill(P.L);  // the peer is lost with the group outstanding
        const bool  sent = st::group_sends(ctl, *w, slow);
        std::string e;
        const bool  got = sent && st::group_answer(ctl, *w, 0, slow, e);
        st::group_close(ctl, *w);
        w.reset();
        const bool rec = settle(P.L, [&]() { return ww.expired(); }, 2.0);
        lost_ok        = pst && !sent && !got && rec;
        sd = std::string("peer lost: the group's sends ") + (sent ? "finished" : "failed") + ", its record " +
             (rec ? "reclaimed" : "not reclaimed");
        P.L.stop();  // unmaps after its operations ended
        P.F.stop();
    }
    {
        pair_t      P;
        std::string err;
        if (!P.start(err) || !P.L.map(mb2.data(), mb2.size(), err)) {
            d = err;
            return false;
        }
        std::string   e;
        const int32_t s1 = P.L.send(lk::make_tag(lk::kind::test, 2, 1, 0, 0), mb2.data(), 256, e);
        const int32_t r1 = P.L.recv(lk::make_tag(lk::kind::test, 2, 2, 0, 0), mb2.data() + 256, 256, e);
        P.L.stop();  // both outstanding: the stop ends them before it unmaps
        P.F.stop();
        stop_ok = s1 >= 0 && r1 >= 0;
        sd += std::string("; teardown with a send and a receive outstanding on the mapped buffer: stopped");
    }
    life::disown(mb.data(), "mapped boundary buffer");
    life::disown(mb2.data(), "mapped buffer");
    d = sd;
    return lm.clean(d) && lost_ok && stop_ok;
}

bool c_reuse(std::string & d) {
    mark                 lm;
    constexpr size_t     H = 1024;
    std::vector<uint8_t> sb(2 * H, 0), lb0(H), lb1(H);
    life::own(sb.data(), sb.size(), "the follower's slot 3 boundary buffer");
    fill(lb0.data(), H, 3, 1, 5, 0);
    fill(lb1.data(), H, 3, 1, 5, 1);
    std::vector<st::step_out> fo;
    fo.reserve(4);
    st::ctrl    rc[2];
    bool        tag_ok = false, buf_ok = false;
    std::string sd;
    {
        pair_t      P;
        std::string err;
        if (!P.start(err)) {
            d = err;
            return false;
        }
        st::control ctl;
        ctl.init(*P.pl);
        ctl.send_timeout_s = slow;
        st::leader_stats stats;
        const st::member m  = begin_member(3);
        // request generation 1 on slot 3, answered, then its result dropped (the request cancelled)
        auto             w1 = std::shared_ptr<st::group_wire>(new st::group_wire());
        std::string      e;
        bool             ok = st::post_group(ctl, w1, 0, &m, 1, nullptr, nullptr, stats) &&
                              frecv(P.F, st::ctrl_tag(1, 0), &rc[0], sizeof rc[0], slow);
        fo.push_back(answer(3, 1, 1));
        const int32_t f1 = fsend(P.F, st::step_tag(0, 1, 3), &fo.back(), sizeof(st::step_out));
        ok               = ok && st::group_sends(ctl, *w1, slow) && st::group_answer(ctl, *w1, 0, slow, e) &&
                           is_answer(w1->out[0], 3, 1, 1);
        st::group_close(ctl, *w1);
        w1.reset();
        // generation 2 reuses the slot; a stale duplicate of generation 1's X-STEP arrives first
        auto w2 = std::shared_ptr<st::group_wire>(new st::group_wire());
        ok      = ok && st::post_group(ctl, w2, 0, &m, 1, nullptr, nullptr, stats) &&
                  frecv(P.F, st::ctrl_tag(2, 0), &rc[1], sizeof rc[1], slow);
        fo.push_back(answer(3, 1, 1));
        const int32_t f2 = fsend(P.F, st::step_tag(0, 1, 3), &fo.back(), sizeof(st::step_out));
        fo.push_back(answer(3, 2, 2));
        const int32_t f3 = fsend(P.F, st::step_tag(0, 2, 3), &fo.back(), sizeof(st::step_out));
        ok               = ok && st::group_sends(ctl, *w2, slow) && st::group_answer(ctl, *w2, 0, slow, e);
        const bool mine  = is_answer(w2->out[0], 3, 2, 2);
        st::group_close(ctl, *w2);
        w2.reset();
        poll_until([&]() { return fl::unexpected_count(P.L) == 1; }, 1.0);
        const int32_t unexp = fl::unexpected_count(P.L);
        tag_ok              = ok && mine && unexp == 1;
        sd = std::string("generation 2's record ") + (mine ? "holds its own answer" : "does not hold its own answer") +
             "; the stale generation-1 X-STEP " + (unexp == 1 ? "stayed unexpected" : "was taken");
        std::string fe;
        for (int32_t op : { f1, f3 }) {
            if (op >= 0) {
                P.F.wait(op, fe, slow);
            }
        }
        (void) f2;  // never received: it ends with the link
        // the follower's slot buffer: a group's two boundary receives, card 0's copy held; the group fails; the
        // slot's next request refills the buffer
        const uint64_t       t0 = st::boundary_tag(0, 5, 0, 3), t1 = st::boundary_tag(0, 5, 1, 3);
        std::vector<int32_t> fops = { P.pf->recv(t0, sb.data(), H, e), P.pf->recv(t1, sb.data() + H, H, e) };
        fl::hold_transfers(P.F, ~uint64_t(0), t0, false);
        const int64_t c0 = fl::copies(P.F);
        const int32_t l0 = P.L.send(t0, lb0.data(), H, e);
        const int32_t l1 = P.L.send(t1, lb1.data(), H, e);
        poll_until([&]() { return fl::held_transfers(P.F) == 1; }, slow);
        poll_until([&]() { return fl::copies(P.F) >= c0 + int64_t(H); }, slow);  // card 1's copy has landed
        const bool all = st::wait_all(*P.pf, fops, quick, e);
        life::reuse(sb.data(), sb.size(), "the follower's slot 3 boundary buffer (the next request)");
        fill(sb.data(), sb.size(), 3, 2, 6, 0);
        fl::release_transfers(P.F);  // a receive still posted would now write the old request's bytes
        settle(P.F, []() { return true; }, 2.0);
        for (int32_t op : { l0, l1 }) {
            if (op >= 0) {
                P.L.wait(op, fe, quick);
            }
        }
        buf_ok = !all && same(sb.data(), sb.size(), 3, 2, 6, 0);
        sd += std::string("; the failed group's buffer ") +
              (buf_ok ? "kept the next request's bytes" : "was written by the old request's held copy");
        P.L.stop();
        P.F.stop();
    }
    life::disown(sb.data(), "the follower's slot 3 boundary buffer");
    d = sd;
    return lm.clean(d) && tag_ok && buf_ok;
}

}  // namespace

int fnfl_safety_test(const std::string & cases) {
    life::set_abort(false);  // count, so every case reports
    const auto on = [&](const char * c) {
        return cases == "all" || ("," + cases + ",").find("," + std::string(c) + ",") != std::string::npos;
    };

    struct entry {
        const char *                       name;
        const char *                       schedule;
        std::function<bool(std::string &)> run;
    };

    const entry list[] = {
        { "normal",       "groups 0 {slots 0,1} and 1 {2,3}, boundaries 3,000 bytes a card; answered 1 then 0",                c_normal                                           },
        { "partial",
         "submission 4 refused (after 2 answer receives and the X-CTRL send); then the X-CTRL taken and "
          "member 0 answered",                                                                                                 c_partial                                          },
        { "queued",
         "link thread held; command (3 tokens, stack ack) times out at 0.2 s; thread released; tokens "
          "taken and X-ACK 1 sent",                                                                                            c_queued                                           },
        { "posted",
         "answer matched and its copy held (not stuck); link thread held; wait 0.2 s; record dropped; "
          "thread released",                                                                                                   [](std::string & d) { return c_posted(false, d); } },
        { "posted-stuck", "as posted, the copy stuck through the forced close, then released",
         [](std::string & d) { return c_posted(true, d); }                                                                                                                        },
        { "race",         "matched and held; the cancel's hook releases it; then op1 done, op2 on its slot, wait(op1)",
         c_race                                                                                                                                                                   },
        { "final",        "group A {0,1,2}: 0 and 1 answered, 2 longer by 8 bytes; group B {4,5} answers held",                c_final                                            },
        { "ack",          "X-ACK 1's copy held at the leader; command (3 stack tokens, stack ack) waits 0.2 s; copy released",
         c_ack                                                                                                                                                                    },
        { "shutdown",
         "peer killed with a group outstanding; then stop with a send and a receive outstanding on a "
          "mapped buffer",                                                                                                     c_shutdown                                         },
        { "reuse",
         "slot 3: generation 1 answered and dropped, stale duplicate then generation 2; then a follower "
          "group with card 0's copy held, wait_all 0.2 s, buffer refilled, copy released",                                     c_reuse                                            },
    };
    int32_t failed = 0, ran = 0;
    for (const entry & e : list) {
        if (!on(e.name) && !(std::string(e.name) == "posted-stuck" && on("posted"))) {
            continue;
        }
        ++ran;
        std::string d;
        const bool  ok = e.run(d);
        printf("safety-test %s: %s\n  schedule: %s\n  %s\n", e.name, ok ? "PASS" : "FAIL", e.schedule, d.c_str());
        fflush(stdout);
        failed += ok ? 0 : 1;
    }
    const life::totals t = life::get();
    printf(
        "safety-test: ledger %s: %lld owners, %lld uses, %lld ended, %lld reuses, %lld maps, %lld unmaps, "
        "%lld published copies; live owners %lld, live users %lld\n",
        life::on ? "on" : "off (not a lifetime build)", (long long) t.owns, (long long) t.uses, (long long) t.dones,
        (long long) t.reuses, (long long) t.maps, (long long) t.unmaps, (long long) t.publishes,
        (long long) t.live_owners, (long long) t.live_users);
    printf("safety-test: %s (%d of %d cases failed)\n", failed == 0 && ran > 0 ? "PASS" : "FAIL", failed, ran);
    fflush(stdout);
    return failed == 0 && ran > 0 ? 0 : 1;
}

}  // namespace fn::runtime
