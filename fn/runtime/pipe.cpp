#include "fn/runtime/pipe.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace fn::runtime::pipe {

namespace {

bool has(const std::vector<int32_t> & v, int32_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

void erase(std::vector<int32_t> & v, int32_t x) {
    v.erase(std::remove(v.begin(), v.end(), x), v.end());
}

}  // namespace

bool core::wanted(int32_t slot) const {
    return has(want_, slot);
}

void core::set_writers(const std::vector<int32_t> & slots) {
    want_.clear();
    for (int32_t s : slots) {
        if (s >= 0 && s < max_writers && !has(want_, s)) {
            want_.push_back(s);
        }
    }
}

void core::leave(int32_t slot) {
    erase(want_, slot);
    for (int32_t g = 0; g < 2; ++g) {
        if (!sent_[g]) {
            erase(members_[g], slot);
        }
    }
}

void core::read_start(int32_t slot) {
    if (!has(readers_, slot)) {
        readers_.push_back(slot);
    }
}

void core::front_done(int32_t slot) {
    erase(readers_, slot);
    if (next_reader_ >= readers_.size()) {
        next_reader_ = 0;
    }
}

void core::rebalance() {
    // leavers out of the groups not in flight
    for (int32_t g = 0; g < 2; ++g) {
        if (sent_[g]) {
            continue;
        }
        std::vector<int32_t> & m = members_[g];
        m.erase(std::remove_if(m.begin(), m.end(), [this](int32_t s) { return !wanted(s); }), m.end());
    }
    if (groups_ == 1) {
        // one group: every writer in A while it is idle
        for (int32_t s : want_) {
            if (!sent_[0] && !has(members_[0], s) && int32_t(members_[0].size()) < max_writers) {
                members_[0].push_back(s);
            }
        }
        return;
    }
    if (!sent_[0] && !sent_[1]) {
        // both idle: even them out (ceil and floor of n / 2)
        for (;;) {
            const int32_t big = members_[0].size() >= members_[1].size() ? 0 : 1;
            if (members_[big].size() < members_[1 - big].size() + 2) {
                break;
            }
            members_[1 - big].push_back(members_[big].back());
            members_[big].pop_back();
        }
    } else if (!sent_[0] || !sent_[1]) {
        // one in flight: an idle group two or more larger parks its last members, who join the other group at its
        // next begin (one step later than they would have stepped)
        const int32_t idle = sent_[0] ? 1 : 0;
        while (members_[idle].size() >= members_[1 - idle].size() + 2) {
            members_[idle].pop_back();
        }
    }
    // joiners (and the parked) into an idle group no larger than the other (four at most)
    for (int32_t s : want_) {
        if (has(members_[0], s) || has(members_[1], s)) {
            continue;
        }
        int32_t best = -1;
        for (int32_t g = 0; g < 2; ++g) {
            if (sent_[g] || int32_t(members_[g].size()) >= max_group || members_[g].size() > members_[1 - g].size()) {
                continue;
            }
            if (best < 0 || members_[g].size() < members_[size_t(best)].size()) {
                best = g;
            }
        }
        if (best >= 0) {
            members_[best].push_back(s);
        }
    }
}

action core::next() {
    rebalance();
    action     x;
    const bool writers = !want_.empty() || sent_[0] || sent_[1];
    // the read turn: one front read group between the leader's group actions (the whole chunk with no writer)
    if (!readers_.empty() && (read_turn_ || !writers)) {
        x.a    = act::front_read;
        x.slot = readers_[next_reader_ % readers_.size()];
        x.n    = writers ? 1 : 0;
        return x;
    }
    // a group to begin: idle with members (the one not begun last first)
    for (int32_t i = 0; i < 2; ++i) {
        const int32_t g = i == 0 ? 1 - last_ : last_;
        if (!sent_[g] && !members_[g].empty()) {
            x.a       = act::begin;
            x.g       = g;
            x.members = members_[g];
            return x;
        }
    }
    // the older sent group's answers
    if (sent_[0] || sent_[1]) {
        const int32_t g = sent_[0] && sent_[1] ? (order_[0] < order_[1] ? 0 : 1) : sent_[0] ? 0 : 1;
        x.a             = act::end;
        x.g             = g;
        x.members       = members_[g];
        return x;
    }
    return x;
}

void core::done(const action & x) {
    switch (x.a) {
        case act::begin:
            sent_[x.g]  = true;
            order_[x.g] = ++clock_;
            last_       = x.g;
            read_turn_  = !readers_.empty();
            break;
        case act::end:
            sent_[x.g] = false;
            break;
        case act::front_read:
            read_turn_ = false;
            if (!readers_.empty()) {
                next_reader_ = (next_reader_ + 1) % readers_.size();
            }
            break;
        case act::none:
            break;
    }
}

std::string core::describe() const {
    std::string s;
    for (int32_t g = 0; g < 2; ++g) {
        s += std::string(g ? "; " : "") + "group " + char('A' + g) + (sent_[g] ? " (in flight)" : "") + ":";
        for (int32_t m : members_[g]) {
            s += " " + std::to_string(m);
        }
    }
    s += "; readers " + std::to_string(readers_.size());
    return s;
}

// ---- pipe-test: the core against a fake leader and follower ----

namespace {

// The fake boxes. The leader runs one action at a time on its clock; the follower runs its queue in order on its
// own: a group's back half when its send arrived, then one read layer group (the follower's interleave), and read
// groups while nothing else waits. Abstract times: a member's front half 1.0, its back half 1.2, a read layer group
// 0.3 on either box
struct fake {
    struct rd {
        int32_t front_groups   = 0;  // the front groups left in the read
        int32_t per_chunk      = 3;  // layer groups a chunk (either box)
        int32_t back_groups    = 0;  // the back groups released (front chunks done) and not run
        int32_t back_left      = 0;  // the back groups of the whole read not run
        bool    front_finished = false;
    };

    core                                    c;
    double                                  t_lead = 0, t_follow = 0;
    std::map<int32_t, double>               answers;  // a group in flight: when its back half ends
    std::map<int32_t, std::vector<int32_t>> sent;     // its members when begun
    std::map<int32_t, int32_t>              steps;    // steps a slot took
    std::map<int32_t, rd>                   readers;
    std::vector<int32_t>                    writers;  // the generating slots (the caller's)
    std::set<int32_t>                       left;     // slots that left (the end of generation, a cancel, a failure)
    std::function<bool(int32_t slot)>       ends;     // the end of generation after this step (the test's choice)
    bool        fail_next         = false;            // the next group's answers fail (the follower's failure)
    int32_t     reads_since_group = 0;                // the leader's front read groups since its last group action
    int32_t     both_in_flight    = 0;  // actions taken with both groups in flight (the pipeline's overlap)
    int32_t     back_behind_max   = 0;  // the most read groups a back half waited behind
    std::string bad;

    void set(const std::vector<int32_t> & w) {
        writers = w;
        c.set_writers(w);
    }

    void drop(int32_t s) {
        erase(writers, s);
        left.insert(s);
        c.set_writers(writers);
        c.leave(s);
    }

    void check() {
        std::set<int32_t> seen;
        for (int32_t g = 0; g < 2; ++g) {
            if (int32_t(c.group(g).size()) > max_group) {
                bad = "a group of more than four";
            }
            for (int32_t m : c.group(g)) {
                if (!seen.insert(m).second) {
                    bad = "slot " + std::to_string(m) + " in two groups";
                }
            }
            if (c.in_flight(g) && sent.count(g) != 0 && c.group(g) != sent[g]) {
                bad = "a group's membership changed while in flight";
            }
        }
        both_in_flight += c.in_flight(0) && c.in_flight(1) ? 1 : 0;
    }

    // the follower's idle time until t: read groups of the released back chunks
    void follower_idle_until(double t) {
        for (auto & [s, r] : readers) {
            while (r.back_groups > 0 && t_follow + 0.3 <= t) {
                t_follow += 0.3;
                --r.back_groups;
                --r.back_left;
            }
        }
    }

    void readers_done() {
        for (auto it = readers.begin(); it != readers.end();) {
            const rd & r = it->second;
            if (r.front_finished && r.back_left == 0) {
                // the read ended on both boxes: the slot generates
                const int32_t s        = it->first;
                it                     = readers.erase(it);
                std::vector<int32_t> w = writers;
                w.push_back(s);
                set(w);
            } else {
                ++it;
            }
        }
    }

    bool run(int32_t n_actions, const std::function<void(int32_t)> & at) {
        for (int32_t i = 0; i < n_actions && bad.empty(); ++i) {
            if (at) {
                at(i);
            }
            readers_done();
            const action x = c.next();
            check();
            if (!bad.empty()) {
                break;
            }
            switch (x.a) {
                case act::none:
                    // the leader idle: the follower finishes what it holds
                    follower_idle_until(1e30);
                    break;
                case act::begin:
                    {
                        if (c.in_flight(x.g) || answers.count(x.g) != 0) {
                            bad = "a group begun before its last answers were applied";
                            break;
                        }
                        for (int32_t m : x.members) {
                            if (!has(writers, m)) {
                                bad = "slot " + std::to_string(m) + " begun after it left";
                            }
                        }
                        if (!writers.empty() && reads_since_group > 1) {
                            bad = "a writer's front half waited behind " + std::to_string(reads_since_group) +
                                  " front read groups";
                        }
                        reads_since_group = 0;
                        t_lead += 1.0 * double(x.members.size());
                        // its back half: after the follower's queue; the follower runs read groups while it waits
                        follower_idle_until(t_lead);
                        const double start = std::max(t_lead, t_follow);
                        t_follow           = start + 1.2 * double(x.members.size());
                        answers[x.g]       = t_follow;
                        // then one read layer group (the follower's interleave): the next back half waits behind it
                        int32_t behind     = 0;
                        for (auto & [s, r] : readers) {
                            if (r.back_groups > 0) {
                                t_follow += 0.3;
                                --r.back_groups;
                                --r.back_left;
                                ++behind;
                                break;
                            }
                        }
                        back_behind_max = std::max(back_behind_max, behind);
                        sent[x.g]       = x.members;
                        c.done(x);
                        break;
                    }
                case act::end:
                    {
                        auto it = answers.find(x.g);
                        if (it == answers.end()) {
                            bad = "an end of a group never begun";
                            break;
                        }
                        if (x.members != sent[x.g]) {
                            bad = "a group's membership changed while in flight";
                            break;
                        }
                        follower_idle_until(it->second);
                        t_lead = std::max(t_lead, it->second);
                        answers.erase(it);
                        sent.erase(x.g);
                        reads_since_group = 0;
                        c.done(x);
                        const bool failed = fail_next;
                        fail_next         = false;
                        for (int32_t m : x.members) {
                            if (failed) {
                                drop(m);
                                continue;
                            }
                            ++steps[m];
                            if (ends && ends(m)) {
                                drop(m);
                            }
                        }
                        break;
                    }
                case act::front_read:
                    {
                        auto it = readers.find(x.slot);
                        if (it == readers.end() || it->second.front_groups <= 0) {
                            bad = "a front read group with no front work";
                            break;
                        }
                        if (x.n == 0 && !writers.empty()) {
                            bad = "a whole front chunk while writers wait";
                            break;
                        }
                        rd &          r   = it->second;
                        const int32_t rem = r.front_groups % r.per_chunk;
                        const int32_t n   = x.n != 0 ? 1 : (rem == 0 ? r.per_chunk : rem);
                        t_lead += 0.3 * n;
                        const int32_t before = r.front_groups;
                        r.front_groups -= n;
                        // each chunk's front done releases its back groups to the follower
                        const int32_t chunks_before = (before + r.per_chunk - 1) / r.per_chunk;
                        const int32_t chunks_after  = (r.front_groups + r.per_chunk - 1) / r.per_chunk;
                        r.back_groups += (chunks_before - chunks_after) * r.per_chunk;
                        // a read with no writer present delays no one
                        reads_since_group = writers.empty() ? 0 : reads_since_group + 1;
                        c.done(x);
                        if (r.front_groups == 0) {
                            r.front_finished = true;
                            c.front_done(x.slot);
                        }
                        break;
                    }
            }
        }
        return bad.empty();
    }
};

std::vector<int32_t> first(int32_t n) {
    std::vector<int32_t> w;
    for (int32_t s = 0; s < n; ++s) {
        w.push_back(s);
    }
    return w;
}

}  // namespace

bool self_test(std::string & report) {
    int32_t    failed = 0;
    const auto line   = [&](const std::string & name, bool ok, const std::string & detail) {
        report +=
            "pipe-test: " + name + ": " + (ok ? "PASS" : "FAIL") + (detail.empty() ? "" : " (" + detail + ")") + "\n";
        failed += ok ? 0 : 1;
    };
    // 1 .. 8 writers, steady: every writer steps evenly, the groups ceil and floor of n / 2, both groups in flight
    // together from two writers
    for (int32_t n = 1; n <= max_writers; ++n) {
        fake f;
        f.set(first(n));
        f.run(400, nullptr);
        int32_t lo = 1 << 30, hi = 0;
        for (int32_t s = 0; s < n; ++s) {
            lo = std::min(lo, f.steps[s]);
            hi = std::max(hi, f.steps[s]);
        }
        const size_t a = f.c.group(0).size(), b = f.c.group(1).size();
        const bool   sizes   = std::max(a, b) == size_t((n + 1) / 2) && std::min(a, b) == size_t(n / 2);
        const bool   overlap = n < 2 || f.both_in_flight > 0;
        line(std::to_string(n) + " writer" + (n > 1 ? "s" : ""),
             f.bad.empty() && lo > 0 && hi - lo <= 1 && sizes && overlap,
             f.bad.empty() ?
                 "steps a writer " + std::to_string(lo) + " .. " + std::to_string(hi) + "; groups of " +
                     std::to_string(std::max(a, b)) + " and " + std::to_string(std::min(a, b)) +
                     (n >= 2 ? "; both groups in flight at " + std::to_string(f.both_in_flight) + " actions" : "") :
                 f.bad);
    }
    // joins at every point: a writer added before action i (0 .. 29), 0 .. 7 writers before it
    {
        bool        ok  = true;
        std::string why = "every joiner stepped";
        for (int32_t n = 0; n < max_writers && ok; ++n) {
            for (int32_t at = 0; at < 30 && ok; ++at) {
                fake f;
                f.set(first(n));
                f.run(300, [&](int32_t i) {
                    if (i == at) {
                        f.set(first(n + 1));
                    }
                });
                const size_t a = f.c.group(0).size(), b = f.c.group(1).size();
                ok = f.bad.empty() && f.steps[n] > 0 && (a > b ? a - b : b - a) <= 1;
                if (!ok) {
                    why = f.bad.empty() ? "the joiner (" + std::to_string(n) + " before, action " + std::to_string(at) +
                                              "): " + std::to_string(f.steps[n]) + " steps, groups of " +
                                              std::to_string(a) + " and " + std::to_string(b) :
                                          f.bad;
                }
            }
        }
        line("joins at every point (0 .. 7 writers, a joiner before actions 0 .. 29)", ok, why);
    }
    // leaves at every point (a cancel: the slot leaves at once, or after its step when its group is in flight)
    {
        bool        ok  = true;
        std::string why = "every leaver took at most its step in flight; the others went on";
        for (int32_t n = 2; n <= max_writers && ok; ++n) {
            for (int32_t at = 0; at < 30 && ok; ++at) {
                fake f;
                f.set(first(n));
                int32_t at_leave = -1;
                f.run(300, [&](int32_t i) {
                    if (i == at) {
                        at_leave = f.steps[0];
                        f.drop(0);
                    }
                });
                const size_t a = f.c.group(0).size(), b = f.c.group(1).size();
                ok = f.bad.empty() && f.steps[0] <= at_leave + 1 && f.steps[1] > 10 && (a > b ? a - b : b - a) <= 1;
                if (!ok) {
                    why = f.bad.empty() ?
                              "slot 0 stepped " + std::to_string(f.steps[0] - at_leave) + " times after it left (" +
                                  std::to_string(n) + " writers, action " + std::to_string(at) + "); groups of " +
                                  std::to_string(a) + " and " + std::to_string(b) :
                              f.bad;
                }
            }
        }
        line("leaves at every point (cancels: 2 .. 8 writers, a leaver before actions 0 .. 29)", ok, why);
    }
    // the end of generation: members end after their steps at random and a queued request takes the slot a few
    // actions later (a server's queue); every invariant holds and the groups stay within one of each other once
    // both are idle
    {
        fake f;
        f.set(first(8));
        uint64_t                                 x     = 12345;
        int32_t                                  ended = 0, uneven = 0;
        std::vector<std::pair<int32_t, int32_t>> queued;  // (slot, the action it rejoins at)
        int32_t                                  now = 0;
        f.ends                                       = [&](int32_t s) {
            x = x * 6364136223846793005ull + 1442695040888963407ull;
            if ((x >> 33) % 17 != 0) {
                return false;
            }
            ++ended;
            queued.push_back({ s, now + 1 + int32_t((x >> 40) % 5) });
            return true;
        };
        f.run(1500, [&](int32_t i) {
            now = i;
            for (size_t q = 0; q < queued.size();) {
                if (queued[q].second <= i) {
                    std::vector<int32_t> w = f.writers;
                    w.push_back(queued[q].first);
                    f.left.erase(queued[q].first);
                    f.set(w);
                    queued.erase(queued.begin() + int64_t(q));
                } else {
                    ++q;
                }
            }
            const size_t a = f.c.group(0).size(), b = f.c.group(1).size();
            if (!f.c.in_flight(0) && !f.c.in_flight(1) && (a > b ? a - b : b - a) > 1) {
                ++uneven;
            }
        });
        line("ends of generation (8 slots, requests ending at random and queued ones taking their slots)",
             f.bad.empty() && ended >= 20 && uneven == 0,
             f.bad.empty() ? std::to_string(ended) + " requests ended; groups uneven while both idle at " +
                                 std::to_string(uneven) + " actions" :
                             f.bad);
    }
    // readers interleaved with 0 .. 6 writers: each read finishes and its slot steps; with writers, a front half
    // waits behind at most one front read group and a back half behind at most one back read group; the writers keep
    // stepping meanwhile
    {
        bool        ok = true;
        std::string why;
        for (int32_t n = 0; n <= 6 && ok; n += 2) {
            fake f;
            f.set(first(n));
            for (int32_t r = 6; r < 8; ++r) {
                f.readers[r].front_groups = 3 * 4;
                f.readers[r].back_left    = 3 * 4;
                f.c.read_start(r);
            }
            int32_t steps_during = 0;
            f.run(2000, [&](int32_t) {
                if (!f.readers.empty()) {
                    int32_t t = 0;
                    for (int32_t s = 0; s < n; ++s) {
                        t += f.steps[s];
                    }
                    steps_during = t;
                }
            });
            ok  = f.bad.empty() && f.readers.empty() && f.back_behind_max <= 1 && (n == 0 || steps_during > 0) &&
                  f.steps[6] > 0 && f.steps[7] > 0;
            why = f.bad.empty() ? std::to_string(n) + " writers: " +
                                      (f.readers.empty() ? "both reads finished and their slots stepped" :
                                                           "a read did not finish") +
                                      "; a back half behind at most " + std::to_string(f.back_behind_max) +
                                      " read group; writer steps during the reads " + std::to_string(steps_during) :
                                  f.bad;
        }
        line("readers interleaved (two reads of four chunks, with 0, 2, 4, 6 writers)", ok, why);
    }
    // the follower's failure: a group's answers fail; its members leave and the other group goes on
    {
        fake f;
        f.set(first(4));
        f.run(20, nullptr);
        f.fail_next    = true;
        int32_t before = 0;
        for (int32_t s = 0; s < 4; ++s) {
            before += f.steps[s];
        }
        f.run(100, nullptr);
        int32_t after = 0, others = 0;
        for (int32_t s = 0; s < 4; ++s) {
            after += f.steps[s];
            others += f.left.count(s) == 0 ? 1 : 0;
        }
        line("the follower's failure (a group's answers failed)",
             f.bad.empty() && f.left.size() == 2 && others == 2 && after > before + 20,
             f.bad.empty() ? std::to_string(f.left.size()) + " members left; the other " + std::to_string(others) +
                                 " stepped on (" + std::to_string(after - before) + " steps)" :
                             f.bad);
    }
    report += std::string("pipe-test: ") + (failed == 0 ? "PASS" : "FAIL") + "\n";
    return failed == 0;
}

}  // namespace fn::runtime::pipe
