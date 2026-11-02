#pragma once

// APF1: the writers' pause target P (--fnfl-writer-pause-ms, POST /props tu.writer_pause_ms; default 100 ms). A read
// step beside writers runs read units while they fit before the latest start that still lets every writer group's
// next completed step come within P of its last one. Best effort: a minimum unit, a small read run whole, a
// prediction error, or writer work that alone exceeds P can overrun it, and each overrun is counted with its reason.
//
// Each box keeps its own book (host clock only, milliseconds on a local monotonic clock; durations and group
// identity cross between boxes, never timestamps): the split's front in stage::pipeline, the back in stage::follower,
// and resident TP without a split in runtime::engine. The server's read step consumes the owner's decision.
//
// Latest start constraining this box's read work, per group (S35's phase distinctions with each box's own measured
// stage estimates; APF1's table):
//   front, the group needs its next front stage (ready):   C + P - F - B
//   front, its step is queued or running on the back:      min(C + P - Q, E + P - F - B - Q), E the projected back
//     completion (after the back's earlier queued group), Q the front work queued ahead of its answers (the ready
//     groups' F: the pipeline begins idle groups before it ends the older one). The first term is the step in flight
//     (its answers are applied only after this box's read), the second its next front stage
//   back, its next input has not come:                    max(C + P - B, C + F), C + F the earliest its input can
//     arrive (its front stage after this box's answer)
//   resident TP, no split:                                 C + P - L, L the local writer stage (the writers stepped
//     between two read steps)
// The step's budget is the smallest latest start less now. A known negative slack is zero; a group whose stage costs
// were not measured for its current membership makes the budget unknown (the one-unit probe), never zero-cost
// permission. Membership changes keep the group's last completion C; a group that empties is cleared; a slot whose
// request ended is a new member when it next joins (its epoch).
//
// No CUDA: pipe-test checks it with a fake clock (self_test).

#include <cstdint>
#include <string>

namespace fn::runtime::pace {

constexpr int32_t pause_default_ms = 100;
constexpr int32_t pause_min_ms     = 10;
constexpr int32_t pause_max_ms     = 1000;
constexpr int32_t max_groups       = 2;  // the split's pipelined groups (resident TP: one)
constexpr int32_t max_slots        = 8;

// P's range, the CLI's and POST /props' rule: an integer from 10 to 1,000 ms
bool pause_valid(int64_t ms);

// the books' clock: this box's monotonic clock in milliseconds (never compared across boxes)
double clock_ms();

enum class role : int32_t { front, back, resident };

// a group's phase as this box sees it
enum class phase : int32_t {
    none    = 0,  // no group
    ready   = 1,  // front or resident: it needs its next stage on this box
    away    = 2,  // front: its current step is queued or running on the back
    waiting = 3,  // back: its next input has not come (its front pending or running)
};
const char * phase_name(phase p);

// the read work a box inserted between two completions of a group (an overrun's reason, in this order of precedence)
enum class read_kind : int32_t {
    budgeted = 0,  // units under a known positive budget
    minimum  = 1,  // the one unit the progress rule runs (budget zero or unknown)
    small    = 2,  // a small read run whole beside writers (MUP1's rule, PRD1 item 7)
    idle     = 3,  // the follower's idle drain (no command waiting; up to two units in flight)
};

// one read step's decision
struct decision {
    bool         known     = false;                // every relevant group's inputs measured for its membership
    double       budget_ms = -1;                   // known: the slack, at least 0; unknown: -1
    double       slack_ms  = 0;                    // known: unclipped
    int32_t      pause_ms  = pause_default_ms;
    int32_t      group     = -1;                   // the constraining group (-1: none)
    phase        ph        = phase::none;
    bool         projected = false;                // its constraint used a projected completion or arrival
    double       c_ms = -1, f_ms = -1, b_ms = -1;  // its inputs (C on the local clock; F and B durations; -1 unknown)
    const char * why = "";                         // unknown: the missing input
};

// a box's totals (the read job lines take differences)
struct stats {
    int64_t steps = 0, known = 0, unknown = 0, zero = 0;  // decisions
    double  budget_sum  = 0;                              // over known decisions
    int64_t completions = 0, gaps = 0, over = 0;          // gaps: completions after an observed one
    int64_t over_none = 0, over_budgeted = 0, over_minimum = 0, over_small = 0, over_idle = 0;
    double  longest_ms = 0, over_ms_sum = 0;              // over_ms_sum: the overruns' excess over P
};

stats operator-(const stats & a, const stats & b);

class pacer {
  public:
    explicit pacer(role r = role::resident) : role_(r) {}

    void set_role(role r) { role_ = r; }

    // P for the next decisions (outside 10 .. 1,000: false, unchanged)
    bool set_pause(int32_t ms);

    int32_t pause() const { return pause_; }

    // a slot's request ended (or the slot was given to another request): it is a new member when it next joins
    void slot_ended(int32_t slot);
    // group g emptied: its timing and phase cleared
    void clear(int32_t g);
    void clear_all();

    // ---- the front (stage::pipeline) ----
    // group g's front stage over slots begins at t0: a new group is admitted (C seeded at t0); a changed membership
    // keeps C. front_sent: it ended at t1 (sent): F measured for this membership; the group is away
    void front_begin(int32_t g, const int32_t * slots, int32_t n, double t0);
    void front_sent(int32_t g, double t1);
    // g's answers applied at now (its observed completion); back_ms its back stage's service (< 0: unknown)
    void answered(int32_t g, double now, double back_ms);

    // ---- the back (stage::follower) ----
    // g's back stage over slots (the front's generation gen of that group) began at t0 (its input in; front_ms the
    // front's service carried with it, < 0 unknown) and ended at t1 (answers sent): B measured, C = t1, the group
    // waits for its next input. Another generation is a new group here too
    void back_ran(int32_t g, int32_t gen, const int32_t * slots, int32_t n, double t0, double t1, double front_ms);
    // the groups the front has (bit g: group g has members) and their generations: others are cleared; a live group
    // not yet seen here (or of another generation) is admitted (C seeded at now, its costs unknown)
    void live_groups(uint32_t mask, const int32_t * gens, double now);

    // ---- resident TP (runtime::engine) ----
    // a writer step of slots from t0 to t1: the open stage (the writers stepped since the last read) takes it; a slot
    // already in the open stage closes that stage first (a new pass of the writers)
    void writer_step(const int32_t * slots, int32_t n, double t0, double t1);
    // no generating slot remains (the server's lone reader or no work): the group is cleared
    void writers_left();

    // ---- reads ----
    // the next read step's budget at now (counted in the totals)
    decision decide(double now);
    // the same without counting (tests, logs)
    decision peek(double now) const;
    // read work this box inserted from t0 to t1 (every live group's account until its next completion)
    void     read_ran(double t0, double t1, read_kind k);

    const stats & totals() const { return st_; }

    // the line's summary of a stats difference: P, decisions and the completion gaps
    static std::string summary(const stats & d, int32_t pause_ms);

    // TEST: group g's state
    phase group_phase(int32_t g) const { return g >= 0 && g < max_groups ? g_[g].ph : phase::none; }

    double group_c(int32_t g) const { return g >= 0 && g < max_groups ? g_[g].c : -1; }

    bool group_live(int32_t g) const { return g >= 0 && g < max_groups && g_[g].live; }

    int64_t group_gen(int32_t g) const { return g >= 0 && g < max_groups ? g_[g].gen : 0; }

  private:
    struct group {
        bool     live     = false;
        int64_t  gen      = 0;      // +1 at each admission of an empty group
        uint64_t key      = 0;      // its membership: slots and their epochs
        phase    ph       = phase::none;
        double   c        = -1;     // the last completion, or the admission's seed
        bool     observed = false;  // c is a completion (not the seed)
        double   sent     = -1;     // front: when its current step left this box
        double   f0       = -1;     // front: when its current front stage began
        int64_t  peer     = -1;     // back: the front's generation of this group
        double   f = -1, b = -1;    // the stage estimates (front: F measured, B carried; back: B measured, F carried;
                                    // resident: L in f)
        uint64_t f_key = 0, b_key = 0;  // the memberships they were measured for
        double   read_ms = 0;           // read work inserted since its last completion
        uint32_t kinds   = 0;           // read_kind bits inserted since then
        // resident: the open stage
        double   s0 = -1, s1 = -1;
        uint32_t s_mask = 0;
    };

    uint64_t key_of(const int32_t * slots, int32_t n) const;
    void     admit(group & x, uint64_t key, double now);
    void     complete(group & x, double now);
    void     close_stage(group & x);
    // group x's latest start (false: unknown, why set)
    bool     latest(int32_t g, double & ls, bool & projected, const char *& why) const;

    role     role_  = role::resident;
    int32_t  pause_ = pause_default_ms;
    group    g_[max_groups];
    uint32_t epoch_[max_slots] = {};
    stats    st_;
};

// the units a read step runs (prefill's units_step, host-checkable): after `ran` units with `planned_ms` predicted,
// whether the next unit (predicted next_ms, < 0 unknown) also runs. The first always runs (the progress rule; its
// time measured when unknown). budget: no budget (a chunk begun in units whose writers left: up to max_units);
// budget_ms < 0 (unknown): the first only; lagged (the follower's idle drain): one a call
bool unit_runs(int32_t ran,
               bool    budget,
               double  budget_ms,
               double  planned_ms,
               double  next_ms,
               bool    lagged,
               int32_t max_units);

// a read step's chunk geometry (engine::read_step's rule): whole (the lone geometry: no writers, or a small read
// beside writers, MUP1 piece 3 as PRD1 item 7 left it), units (writers mode), or layer groups (shared)
enum class geometry : int32_t { whole, units, groups };
geometry geometry_of(bool lone, bool writers, bool small);

// the follower's next action in its command wait (stage::follower::serve): a command that came runs first (the read's
// in-flight drain joined before it); with none, one idle-drain unit while reads wait; else wait for a command
enum class back_next : int32_t { command, idle_unit, wait };
back_next back_choice(bool command_came, bool reads_waiting, bool reads_held);

// pipe-test's cases (a fake clock); appends one line a case to report; true when all pass
bool self_test(std::string & report);

}  // namespace fn::runtime::pace
