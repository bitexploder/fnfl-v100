// APF1: pipe-test's cases of the writer-pause book (fn/runtime/pace.h), on a fake clock; no CUDA
#include "fn/runtime/pace.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace fn::runtime::pace {

namespace {

// the front's begin and send of group g over slots, from t0 to t1
void fs(pacer & p, int32_t g, const int32_t * slots, int32_t n, double t0, double t1) {
    p.front_begin(g, slots, n, t0);
    p.front_sent(g, t1);
}

bool near(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

std::string num(double v) {
    char b[32];
    snprintf(b, sizeof b, "%.3g", v);
    return b;
}

}  // namespace

bool self_test(std::string & report) {
    int32_t    failed = 0;
    const auto line   = [&](const std::string & name, bool ok, const std::string & detail) {
        report +=
            "pace-test: " + name + ": " + (ok ? "PASS" : "FAIL") + (detail.empty() ? "" : " (" + detail + ")") + "\n";
        failed += ok ? 0 : 1;
    };
    const int32_t A[1] = { 0 }, AB[2] = { 0, 1 }, C1[1] = { 2 }, C2[2] = { 2, 3 }, G0[2] = { 0, 0 };

    // P's range: 10 .. 1,000 ms; an invalid update keeps the valid P
    {
        pacer      p(role::front);
        const bool range = !pause_valid(9) && pause_valid(10) && pause_valid(100) && pause_valid(1000) &&
                           !pause_valid(1001) && !pause_valid(-1) && !pause_valid(0);
        const bool keep  = p.pause() == pause_default_ms && p.set_pause(250) && !p.set_pause(5) && p.pause() == 250 &&
                           !p.set_pause(5000) && p.pause() == 250;
        line("P's range and a rejected update keeps P", range && keep, std::to_string(range) + std::to_string(keep));
    }
    // first use: no group, then a group whose costs are unknown: the budget is unknown (the one-unit probe), never
    // zero-cost permission
    {
        pacer          p(role::front);
        const decision d0 = p.decide(0);
        fs(p, 0, A, 1, 0, 6);  // admitted at 0, F 6 measured, B not yet
        const decision d1 = p.decide(7);
        const bool     ok =
            !d0.known && d0.budget_ms < 0 && !d1.known && d1.budget_ms < 0 && std::string(d1.why) == "back unmeasured";
        line("first use: no group, then B unknown: budget unknown", ok, std::string(d0.why) + "; " + d1.why);
    }
    // front, one group, ready after its answers: C + P - F - B; unequal costs (F 6, B 9)
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 6);
        p.answered(0, 20, 9);  // C 20
        const decision d  = p.decide(22);
        const bool     ok = d.known && d.ph == phase::ready && near(d.budget_ms, 20 + 100 - 6 - 9 - 22) && !d.projected;
        line("front ready: C + P - F - B (F 6, B 9)", ok, num(d.budget_ms) + " " + phase_name(d.ph));
    }
    // front, one group away (its step on the back): min(C + P, E + P - F - B), E projected = sent + B
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 6);
        p.answered(0, 20, 9);    // C 20
        fs(p, 0, A, 1, 21, 27);  // sent at 27: E = 36
        const decision d  = p.decide(28);
        const double   ls = std::min(20.0 + 100, 36.0 + 100 - 6 - 9);
        const bool     ok = d.known && d.ph == phase::away && d.projected && near(d.budget_ms, ls - 28);
        line("front away: min(C + P, E + P - F - B), E projected", ok, num(d.budget_ms) + " " + phase_name(d.ph));
    }
    // two groups: A ready, B away behind A's front (Q = F_A); the smallest latest start constrains
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 5);     // A: F 5
        fs(p, 1, C1, 1, 5, 12);   // B: F 7, sent 12
        p.answered(0, 30, 10);    // A: C 30, B 10 (ready)
        p.answered(1, 31, 8);     // B: C 31, B 8
        fs(p, 1, C1, 1, 32, 39);  // B away again, sent 39
        const decision d   = p.decide(40);
        const double   lsA = 30.0 + 100 - 5 - 10;
        const double   lsB = std::min(31.0 + 100 - 5, (39.0 + 8) + 100 - 7 - 8 - 5);
        const bool     ok  = d.known && near(d.budget_ms, std::min(lsA, lsB) - 40) && d.group == (lsA <= lsB ? 0 : 1);
        line("two groups: A ready, B away behind A's front", ok,
             num(d.budget_ms) + " group " + std::to_string(d.group) + " " + phase_name(d.ph));
    }
    // two groups both away: the later one's projected completion waits for the earlier's back stage (the queue)
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 5);
        p.answered(0, 30, 20);    // A: F 5, B 20
        fs(p, 1, C1, 1, 5, 10);
        p.answered(1, 31, 20);    // B: F 5, B 20
        fs(p, 0, A, 1, 32, 37);   // A sent 37: E_A = 57
        fs(p, 1, C1, 1, 37, 42);  // B sent 42: E_B = max(42, 57) + 20 = 77
        const decision d   = p.peek(43);
        const double   lsA = std::min(30.0 + 100, 57.0 + 100 - 5 - 20);
        const double   lsB = std::min(31.0 + 100, 77.0 + 100 - 5 - 20);
        const bool     ok  = d.known && near(d.budget_ms, std::min(lsA, lsB) - 43);
        line("two groups away: the back's queue in E", ok, num(d.budget_ms));
    }
    // zero and negative slack: known zero (the one unit), however late
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 6);
        p.answered(0, 20, 9);
        const decision d  = p.decide(20 + 100 - 15);       // exactly zero
        const decision e  = p.decide(20 + 100 - 15 + 40);  // 40 ms late
        const bool     ok = d.known && near(d.budget_ms, 0) && e.known && near(e.budget_ms, 0) && e.slack_ms < -39;
        line("zero and negative slack: a known zero", ok,
             num(d.budget_ms) + ", " + num(e.budget_ms) + " (" + num(e.slack_ms) + ")");
    }
    // writer work alone above P (F + B 120 > 100): a known zero; its completion gap is an overrun with no read here
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 60);
        p.answered(0, 120, 60);
        fs(p, 0, A, 1, 120, 180);
        p.answered(0, 240, 60);
        const decision d = p.decide(241);
        const stats &  s = p.totals();
        const bool ok = d.known && near(d.budget_ms, 0) && s.over == 1 && s.over_none == 1 && near(s.longest_ms, 120);
        line("writer work over P: zero budget, an overrun with no read here", ok,
             "over " + std::to_string(s.over) + ", longest " + num(s.longest_ms));
    }
    // a unit longer than the slack: the first unit runs (progress); no second unit once it would pass the budget, and
    // none with an unknown prediction or an unknown budget; lagged one a call; no budget up to max_units
    {
        const bool first = unit_runs(0, true, 0, 0, 80, false, 128) && unit_runs(0, true, -1, 0, -1, false, 128);
        const bool fit   = unit_runs(1, true, 50, 20, 30, false, 128) && !unit_runs(1, true, 50, 20, 31, false, 128);
        const bool unk   = !unit_runs(1, true, 50, 0, -1, false, 128) && !unit_runs(1, true, -1, 0, 1, false, 128);
        const bool zero  = !unit_runs(1, true, 0, 0, 0.5, false, 128);
        const bool lag   = !unit_runs(1, true, 1e9, 0, 1, true, 128);
        const bool none  = unit_runs(5, false, -1, 0, -1, false, 6) && !unit_runs(6, false, -1, 0, -1, false, 6);
        line("units: progress, fit, unknown, zero, lagged, no budget", first && fit && unk && zero && lag && none,
             std::to_string(first) + std::to_string(fit) + std::to_string(unk) + std::to_string(zero) +
                 std::to_string(lag) + std::to_string(none));
    }
    // a minimum unit inserted between completions that then overran: counted as the minimum unit's
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 6);
        p.answered(0, 20, 9);
        p.read_ran(21, 150, read_kind::minimum);
        fs(p, 0, A, 1, 150, 156);
        p.answered(0, 165, 9);
        const stats & s  = p.totals();
        const bool    ok = s.over == 1 && s.over_minimum == 1 && near(s.over_ms_sum, 45);
        line("an overrun after a minimum unit is the unit's", ok, "excess " + num(s.over_ms_sum));
    }
    // the small-read exception: a small read whole beside writers (the lone geometry), its overrun counted as such
    {
        const bool g =
            geometry_of(false, true, true) == geometry::whole && geometry_of(false, true, false) == geometry::units &&
            geometry_of(true, false, false) == geometry::whole && geometry_of(false, false, false) == geometry::groups;
        pacer p(role::resident);
        p.writer_step(A, 1, 0, 10);
        p.decide(10);
        p.read_ran(10, 200, read_kind::small);
        p.read_ran(10, 11, read_kind::minimum);
        p.writer_step(A, 1, 200, 210);
        p.decide(210);
        const stats & s = p.totals();
        line("small read whole beside writers: the documented exception", g && s.over == 1 && s.over_small == 1,
             std::to_string(g) + ", small overruns " + std::to_string(s.over_small));
    }
    // no writers: a lone reader's whole chunks (the geometry), and no group: unknown budget (writers mode needs one)
    {
        pacer          p(role::resident);
        const decision d  = p.peek(5);
        const bool     ok = geometry_of(true, false, false) == geometry::whole && !d.known && !p.group_live(0);
        line("no writers: whole chunks, no group", ok, d.why);
    }
    // joins and leaves keep C; the new membership's costs are unknown until measured (bootstrap); slot reuse is a new
    // member; an emptied group is cleared and its next admission seeds C afresh
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 6);
        p.answered(0, 20, 9);
        fs(p, 0, AB, 2, 21, 30);  // slot 1 joins: C kept (20); F measured for AB, B not
        const decision j    = p.peek(31);
        const bool     keep = near(p.group_c(0), 20) && !j.known && std::string(j.why) == "back unmeasured";
        p.answered(0, 45, 12);  // AB measured
        const bool known = p.peek(46).known;
        p.slot_ended(1);        // slot 1's request ended; a new request reuses it
        fs(p, 0, AB, 2, 46, 55);
        const bool reuse = !p.peek(56).known && near(p.group_c(0), 45);
        p.answered(0, 70, 12);
        const int64_t gen = p.group_gen(0);
        p.clear(0);                 // emptied
        const bool cleared = !p.group_live(0) && p.group_phase(0) == phase::none;
        fs(p, 0, C2, 2, 500, 506);  // a new group: seeded at its admission, not 70
        const bool seeded = near(p.group_c(0), 500) && p.group_gen(0) == gen + 1;
        p.answered(0, 520, 9);
        const stats & s  = p.totals();
        // the new group's first completion is no gap (its seed is no completion): no 450 ms overrun
        const bool    ok = keep && known && reuse && cleared && seeded && s.over == 0;
        line("joins, leaves, slot reuse, an emptied group", ok,
             std::to_string(keep) + std::to_string(known) + std::to_string(reuse) + std::to_string(cleared) +
                 std::to_string(seeded) + " over " + std::to_string(s.over));
    }
    // P changes apply at the next decision
    {
        pacer p(role::front);
        fs(p, 0, A, 1, 0, 6);
        p.answered(0, 20, 9);
        const double a = p.peek(22).budget_ms;
        p.set_pause(300);
        const double b = p.peek(22).budget_ms;
        line("a P update applies at the next decision", near(b - a, 200), num(a) + " -> " + num(b));
    }
    // the back: max(C + P - B, C + F); a group not yet seen is admitted with unknown costs; one the front no longer
    // has is cleared; two groups
    {
        pacer p(role::back);
        p.live_groups(1, G0, 0);
        const decision u = p.peek(1);
        p.back_ran(0, 0, A, 1, 2, 12, 6);  // B 10, C 12, F 6
        const decision d = p.peek(13);
        const bool     a = !u.known && d.known && near(d.budget_ms, 12 + 100 - 10 - 13) && d.ph == phase::waiting;
        p.live_groups(3, G0, 14);            // group 1 appears: unknown until it runs here
        const bool b = !p.peek(15).known;
        p.back_ran(1, 0, C1, 1, 20, 28, 7);  // B 8, C 28
        const decision e = p.peek(29);
        const bool     c = e.known && near(e.budget_ms, std::min(12.0 + 100 - 10, 28.0 + 100 - 8) - 29);
        p.live_groups(2, G0, 30);  // group 0 emptied on the front
        const bool dd = !p.group_live(0) && p.group_live(1);
        // F + B above P: the input arrives after the target; read work until then costs nothing more
        pacer      q(role::back);
        q.back_ran(0, 0, A, 1, 0, 70, 60);  // B 70, C 70, F 60: max(70 + 100 - 70, 70 + 60) = 130
        const decision f  = q.peek(71);
        const bool     ee = f.known && near(f.budget_ms, 130 - 71) && f.projected;
        line("back: max(C + P - B, C + F), admission, clearing, two groups", a && b && c && dd && ee,
             std::to_string(a) + std::to_string(b) + std::to_string(c) + std::to_string(dd) + std::to_string(ee));
    }
    // the back: a group the front re-admitted (another generation) is new here: its pause is no overrun
    {
        pacer         p(role::back);
        const int32_t g3[2] = { 3, 0 }, g4[2] = { 4, 0 };
        p.live_groups(1, g3, 0);
        p.back_ran(0, 3, A, 1, 0, 10, 5);
        p.back_ran(0, 3, A, 1, 50, 60, 5);  // a gap of 50
        p.live_groups(1, g4, 5000);         // re-admitted on the front: cleared, seeded at 5,000
        const bool fresh = !p.peek(5001).known;
        p.back_ran(0, 4, A, 1, 5000, 5010, 5);
        const stats & s  = p.totals();
        const bool    ok = fresh && s.gaps == 1 && s.over == 0 && near(p.group_c(0), 5010);
        line("back: another generation is a new group", ok, "gaps " + std::to_string(s.gaps));
    }
    // resident TP: the writers stepped between reads are the group; L its stage; C + P - L; a slot stepping again
    // closes the pass (no read between); the group cleared when the writers left
    {
        pacer p(role::resident);
        p.writer_step(AB, 2, 0, 15);      // a joint step of two
        const decision d = p.decide(16);  // the read closes the stage: C 15, L 15
        const bool     a = d.known && near(d.budget_ms, 15 + 100 - 15 - 16) && d.ph == phase::ready;
        p.writer_step(A, 1, 40, 48);      // five-plus writers step one by one
        p.writer_step(C1, 1, 48, 56);
        p.writer_step(A, 1, 70, 78);      // slot 0 again, no read between: the pass [40, 56] closed
        const bool b = near(p.group_c(0), 56);
        p.writers_left();
        const bool c = !p.group_live(0);
        line("resident TP: one local stage, C + P - L", a && b && c,
             num(d.budget_ms) + " " + std::to_string(a) + std::to_string(b) + std::to_string(c));
    }
    // the follower's command priority: a command that came runs before any read work; the idle drain only with none
    {
        const bool ok = back_choice(true, true, false) == back_next::command &&
                        back_choice(true, false, false) == back_next::command &&
                        back_choice(false, true, false) == back_next::idle_unit &&
                        back_choice(false, true, true) == back_next::wait &&
                        back_choice(false, false, false) == back_next::wait;
        line("follower: commands first, the idle drain only with none", ok, "");
    }
    // a completion gap: the stats' difference over a read job keeps the later running maximum
    {
        stats a, b;
        a.steps = 10, a.over = 3, a.longest_ms = 180;
        b.steps = 4, b.over = 1, b.longest_ms = 90;
        const stats d = a - b;
        line("stats difference", d.steps == 6 && d.over == 2 && near(d.longest_ms, 180), "");
    }
    report += std::string("pace-test: ") + (failed == 0 ? "all PASS" : std::to_string(failed) + " FAIL") + "\n";
    return failed == 0;
}

}  // namespace fn::runtime::pace
