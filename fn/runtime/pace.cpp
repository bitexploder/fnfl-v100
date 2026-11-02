#include "fn/runtime/pace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace fn::runtime::pace {

bool pause_valid(int64_t ms) {
    return ms >= pause_min_ms && ms <= pause_max_ms;
}

double clock_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char * phase_name(phase p) {
    switch (p) {
        case phase::ready:
            return "ready";
        case phase::away:
            return "away";
        case phase::waiting:
            return "waiting";
        default:
            return "none";
    }
}

stats operator-(const stats & a, const stats & b) {
    stats d;
    d.steps         = a.steps - b.steps;
    d.known         = a.known - b.known;
    d.unknown       = a.unknown - b.unknown;
    d.zero          = a.zero - b.zero;
    d.budget_sum    = a.budget_sum - b.budget_sum;
    d.completions   = a.completions - b.completions;
    d.gaps          = a.gaps - b.gaps;
    d.over          = a.over - b.over;
    d.over_none     = a.over_none - b.over_none;
    d.over_budgeted = a.over_budgeted - b.over_budgeted;
    d.over_minimum  = a.over_minimum - b.over_minimum;
    d.over_small    = a.over_small - b.over_small;
    d.over_idle     = a.over_idle - b.over_idle;
    d.over_ms_sum   = a.over_ms_sum - b.over_ms_sum;
    // the longest gap is the later total's (a running maximum, not a difference)
    d.longest_ms    = a.longest_ms;
    return d;
}

bool pacer::set_pause(int32_t ms) {
    if (!pause_valid(ms)) {
        return false;
    }
    pause_ = ms;
    return true;
}

void pacer::slot_ended(int32_t slot) {
    if (slot >= 0 && slot < max_slots) {
        ++epoch_[slot];
    }
}

void pacer::clear(int32_t g) {
    if (g >= 0 && g < max_groups) {
        const int64_t gen = g_[g].gen;
        g_[g]             = group();
        g_[g].gen         = gen;  // generations only grow (the back tells a re-admitted group by it)
    }
}

void pacer::clear_all() {
    for (int32_t g = 0; g < max_groups; ++g) {
        clear(g);
    }
}

uint64_t pacer::key_of(const int32_t * slots, int32_t n) const {
    // a membership: the slots (as a set) and their epochs, so a slot reused by another request is a new member
    uint64_t mask = 0, mix = 0;
    for (int32_t k = 0; k < n; ++k) {
        const int32_t s = slots[k];
        if (s < 0 || s >= max_slots) {
            continue;
        }
        mask |= 1ull << uint32_t(s);
    }
    for (int32_t s = 0; s < max_slots; ++s) {
        if (mask & (1ull << uint32_t(s))) {
            mix = mix * 1000003ull + uint64_t(epoch_[s]) * 131ull + uint64_t(s) + 1ull;
        }
    }
    return (mix << 8) ^ mask ^ (mask != 0 ? 0x5bd1e995ull : 0);
}

void pacer::admit(group & x, uint64_t key, double now) {
    if (!x.live) {
        x.live = true;
        x.gen += 1;
        x.c        = now;  // seeded at local admission: no uninitialized deadline
        x.observed = false;
        x.read_ms  = 0;
        x.kinds    = 0;
    }
    x.key = key;  // a changed membership keeps C (the continuing writers' history)
}

void pacer::complete(group & x, double now) {
    ++st_.completions;
    if (x.observed) {
        const double gap = now - x.c;
        ++st_.gaps;
        st_.longest_ms = std::max(st_.longest_ms, gap);
        if (gap > double(pause_)) {
            ++st_.over;
            st_.over_ms_sum += gap - double(pause_);
            // the reason: the read work this box inserted since the last completion, by precedence
            const auto has = [&](read_kind k) {
                return (x.kinds & (1u << uint32_t(k))) != 0;
            };
            if (x.kinds == 0) {
                ++st_.over_none;
            } else if (has(read_kind::small)) {
                ++st_.over_small;
            } else if (has(read_kind::idle)) {
                ++st_.over_idle;
            } else if (has(read_kind::minimum)) {
                ++st_.over_minimum;
            } else {
                ++st_.over_budgeted;
            }
        }
    }
    x.c        = now;
    x.observed = true;
    x.read_ms  = 0;
    x.kinds    = 0;
}

void pacer::front_begin(int32_t g, const int32_t * slots, int32_t n, double t0) {
    if (g < 0 || g >= max_groups) {
        return;
    }
    group & x = g_[g];
    admit(x, key_of(slots, n), t0);
    x.f0 = t0;
}

void pacer::front_sent(int32_t g, double t1) {
    if (g < 0 || g >= max_groups || !g_[g].live || g_[g].f0 < 0) {
        return;
    }
    group & x = g_[g];
    x.f       = std::max(0.0, t1 - x.f0);
    x.f_key   = x.key;
    x.ph      = phase::away;
    x.sent    = t1;
    x.f0      = -1;
}

void pacer::answered(int32_t g, double now, double back_ms) {
    if (g < 0 || g >= max_groups || !g_[g].live) {
        return;
    }
    group & x = g_[g];
    complete(x, now);
    if (back_ms >= 0) {
        x.b     = back_ms;
        x.b_key = x.key;
    } else {
        x.b_key = 0;  // unknown for this membership
    }
    x.ph = phase::ready;
}

void pacer::back_ran(int32_t g, int32_t gen, const int32_t * slots, int32_t n, double t0, double t1, double front_ms) {
    if (g < 0 || g >= max_groups) {
        return;
    }
    group & x = g_[g];
    if (x.live && x.peer != gen) {
        clear(g);  // the front re-admitted this group: a new group here too
    }
    const uint64_t key = key_of(slots, n);
    admit(x, key, t0);
    x.peer = gen;
    complete(x, t1);
    x.b     = std::max(0.0, t1 - t0);
    x.b_key = key;
    if (front_ms >= 0) {
        x.f     = front_ms;
        x.f_key = key;
    } else {
        x.f_key = 0;
    }
    x.ph = phase::waiting;
}

void pacer::live_groups(uint32_t mask, const int32_t * gens, double now) {
    for (int32_t g = 0; g < max_groups; ++g) {
        group & x = g_[g];
        if ((mask & (1u << uint32_t(g))) == 0) {
            if (x.live) {
                clear(g);
            }
            continue;
        }
        if (x.live && x.peer != gens[g]) {
            clear(g);  // re-admitted on the front since
        }
        if (!x.live) {
            admit(x, 0, now);  // not seen here yet: its costs unknown (the bootstrap)
            x.peer = gens[g];
            x.ph   = phase::waiting;
        }
    }
}

void pacer::close_stage(group & x) {
    if (x.s0 < 0) {
        return;
    }
    uint32_t m = x.s_mask;
    int32_t  slots[max_slots];
    int32_t  n = 0;
    for (int32_t s = 0; s < max_slots; ++s) {
        if (m & (1u << uint32_t(s))) {
            slots[n++] = s;
        }
    }
    const uint64_t key = key_of(slots, n);
    admit(x, key, x.s0);
    complete(x, x.s1);
    x.f     = std::max(0.0, x.s1 - x.s0);
    x.f_key = key;
    x.ph    = phase::ready;
    x.s0 = x.s1 = -1;
    x.s_mask    = 0;
}

void pacer::writer_step(const int32_t * slots, int32_t n, double t0, double t1) {
    group &  x    = g_[0];
    uint32_t mask = 0;
    for (int32_t k = 0; k < n; ++k) {
        if (slots[k] >= 0 && slots[k] < max_slots) {
            mask |= 1u << uint32_t(slots[k]);
        }
    }
    if (x.s0 >= 0 && (x.s_mask & mask) != 0) {
        close_stage(x);  // a slot steps again: the writers' previous pass is complete
    }
    if (x.s0 < 0) {
        x.s0 = t0;
    }
    x.s1 = t1;
    x.s_mask |= mask;
}

void pacer::writers_left() {
    group & x = g_[0];
    close_stage(x);
    clear(0);
}

bool pacer::latest(int32_t g, double & ls, bool & projected, const char *& why) const {
    const group & x = g_[g];
    const double  P = double(pause_);
    projected       = false;
    const bool fk   = x.f >= 0 && x.f_key == x.key && x.key != 0;
    const bool bk   = x.b >= 0 && x.b_key == x.key && x.key != 0;
    switch (role_) {
        case role::resident:
            if (!fk) {
                why = "stage unmeasured";
                return false;
            }
            ls = x.c + P - x.f;
            return true;
        case role::front:
            {
                if (!fk) {
                    why = "front unmeasured";
                    return false;
                }
                if (!bk) {
                    why = "back unmeasured";
                    return false;
                }
                if (x.ph == phase::ready) {
                    ls = x.c + P - x.f - x.b;
                    return true;
                }
                // away: the back's earlier queued group first, then this one (projected), and the front work queued
                // ahead of its answers (the ready groups begin first)
                double q = 0, start = x.sent;
                for (int32_t h = 0; h < max_groups; ++h) {
                    const group & y = g_[h];
                    if (h == g || !y.live) {
                        continue;
                    }
                    const bool yf = y.f >= 0 && y.f_key == y.key && y.key != 0;
                    const bool yb = y.b >= 0 && y.b_key == y.key && y.key != 0;
                    if (y.ph == phase::ready) {
                        if (!yf) {
                            why = "front unmeasured";
                            return false;
                        }
                        q += y.f;
                    } else if (y.ph == phase::away && y.sent < x.sent) {
                        if (!yb) {
                            why = "back unmeasured";
                            return false;
                        }
                        start = std::max(start, y.sent + y.b);
                    }
                }
                const double e = start + x.b;
                projected      = true;
                ls             = std::min(x.c + P - q, e + P - x.f - x.b - q);
                return true;
            }
        case role::back:
            {
                if (!bk) {
                    why = "back unmeasured";
                    return false;
                }
                const double arrive = x.f >= 0 && x.f_key == x.key ? x.c + x.f : x.c;
                projected           = x.f >= 0 && x.f_key == x.key && arrive > x.c + P - x.b;
                ls                  = std::max(x.c + P - x.b, arrive);
                return true;
            }
    }
    why = "role";
    return false;
}

decision pacer::peek(double now) const {
    pacer t = *this;
    if (t.role_ == role::resident) {
        t.close_stage(t.g_[0]);
    }
    decision d;
    d.pause_ms       = t.pause_;
    bool         any = false, unknown = false;
    double       best      = 0;
    const char * why       = "no group";
    int32_t      unknown_g = -1;
    for (int32_t g = 0; g < max_groups; ++g) {
        const group & x = t.g_[g];
        if (!x.live) {
            continue;
        }
        double       ls = 0;
        bool         pj = false;
        const char * w  = "";
        if (!t.latest(g, ls, pj, w)) {
            if (!unknown) {
                why       = w;
                unknown_g = g;
            }
            unknown = true;
            continue;
        }
        if (!any || ls < best) {
            any         = true;
            best        = ls;
            d.group     = g;
            d.ph        = x.ph;
            d.projected = pj;
            d.c_ms      = x.c;
            d.f_ms      = x.f;
            d.b_ms      = x.b;
        }
    }
    // a known non-positive slack is zero whatever else is unknown; else any unknown group makes the budget unknown
    if (any && (!unknown || best - now <= 0)) {
        d.known     = true;
        d.slack_ms  = best - now;
        d.budget_ms = std::max(0.0, d.slack_ms);
        return d;
    }
    d.known     = false;
    d.budget_ms = -1;
    d.why       = why;
    if (unknown_g >= 0) {
        d.group = unknown_g;
        d.ph    = t.g_[unknown_g].ph;
        d.c_ms  = t.g_[unknown_g].c;
    }
    return d;
}

decision pacer::decide(double now) {
    if (role_ == role::resident) {
        close_stage(g_[0]);
    }
    const decision d = peek(now);
    ++st_.steps;
    if (d.known) {
        ++st_.known;
        st_.budget_sum += d.budget_ms;
        st_.zero += d.budget_ms <= 0 ? 1 : 0;
    } else {
        ++st_.unknown;
    }
    return d;
}

void pacer::read_ran(double t0, double t1, read_kind k) {
    for (group & x : g_) {
        if (x.live) {
            x.read_ms += std::max(0.0, t1 - t0);
            x.kinds |= 1u << uint32_t(k);
        }
    }
}

std::string pacer::summary(const stats & d, int32_t pause_ms) {
    char b[400];
    snprintf(b, sizeof b,
             "P %d ms: steps %lld (budget known %lld, zero %lld, unknown %lld; mean %.1f ms); writer completions %lld, "
             "gaps %lld, longest %.1f ms, over P %lld (no read here %lld, budgeted %lld, minimum unit %lld, small "
             "read %lld, idle drain %lld; excess %.1f ms)",
             pause_ms, (long long) d.steps, (long long) d.known, (long long) d.zero, (long long) d.unknown,
             d.known > 0 ? d.budget_sum / double(d.known) : -1.0, (long long) d.completions, (long long) d.gaps,
             d.longest_ms, (long long) d.over, (long long) d.over_none, (long long) d.over_budgeted,
             (long long) d.over_minimum, (long long) d.over_small, (long long) d.over_idle, d.over_ms_sum);
    return b;
}

bool unit_runs(int32_t ran,
               bool    budget,
               double  budget_ms,
               double  planned_ms,
               double  next_ms,
               bool    lagged,
               int32_t max_units) {
    if (ran == 0) {
        return true;  // the progress rule: at least one unit (measured when its prediction is unknown)
    }
    if (lagged) {
        return false;  // the idle drain: one unit a call
    }
    if (!budget) {
        return ran < max_units;
    }
    return budget_ms >= 0 && next_ms >= 0 && planned_ms + next_ms <= budget_ms;
}

geometry geometry_of(bool lone, bool writers, bool small) {
    if (lone || (writers && small)) {
        return geometry::whole;
    }
    return writers ? geometry::units : geometry::groups;
}

back_next back_choice(bool command_came, bool reads_waiting, bool reads_held) {
    if (command_came) {
        return back_next::command;
    }
    return reads_waiting && !reads_held ? back_next::idle_unit : back_next::wait;
}

}  // namespace fn::runtime::pace
