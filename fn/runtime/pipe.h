#pragma once

// TTTS1: the split's two-group pipeline, the leader's scheduling core (no CUDA: pipe-test drives it with fakes).
// Writers form two groups of ceil(n / 2) and floor(n / 2) (a group of one runs alone, of two to four as a joint
// round); a joiner goes to the smaller group, a leaver leaves when its group is not in flight, and membership changes
// only between a group's steps. The leader's box (the front) runs one thing at a time: a group's front half and its
// send (begin), then the follower runs the group's back half in arrival order; the leader takes a group's answers
// (end) before that group's next front half. So Carlos runs F(A, n), F(B, n), F(A, n + 1), ... while Carlin runs
// B(A, n), B(B, n), ...: F(A, n + 1) waits only for B(A, n) (vLLM's batch queue of PP-size batches; SGLang's
// micro-batch loop).
//
// Readers: a read's front chunks run on the leader in layer groups, one group between the leader's group actions
// (the whole chunk at once, in the lone geometry, while no writer is present). Its back chunks are the follower's:
// it runs one read layer group after each step group's back half and more while no command waits, so a writer's back
// half never waits behind more than one read layer group (fn/runtime/stage.h, the follower's read interleave).
//
// The core decides; the caller acts and reports: next() names the next action, done() records its completion. The
// invariants it keeps (pipe-test checks them on every action):
//   a group's front never starts before its last answers were applied;
//   the leader runs one action at a time; the follower's commands queue in order;
//   no slot is in two groups, and no group has more than four;
//   a group's membership never changes while it is in flight;
//   a writer's front half never waits behind more than one front read group (and the follower's back half behind
//   more than one back read group: pipe-test's fake follower).

#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime::pipe {

constexpr int32_t max_writers = 8;
constexpr int32_t max_group   = 4;

enum class act : int32_t {
    none  = 0,  // nothing to do now (no writer, no front read work)
    begin = 1,  // group g's front half and its send (its members: members)
    end   = 2,  // group g's answers (wait, then apply)
    front_read =
        3,      // reader slot's front work: one layer group (n = 1, the writers geometry) or its chunk (n = 0, lone)
};

struct action {
    act                  a    = act::none;
    int32_t              g    = -1;  // the group (begin, end)
    int32_t              slot = -1;  // the reader (front_read)
    int32_t              n    = 0;   // layer groups (front_read; 0: the whole chunk)
    std::vector<int32_t> members;
};

class core {
  public:
    // the generating slots now (each 0 .. 7, at most 8): a new one joins the smaller group (at that group's next
    // begin); one missing leaves its group once the group is not in flight (one in flight finishes its step first)
    void set_writers(const std::vector<int32_t> & slots);
    // a read on slot has front work (its next chunk, or the rest of one); front_done: the read's front is finished
    // (its back may still run on the follower: the caller makes the slot a writer once the read ended there)
    void read_start(int32_t slot);
    void front_done(int32_t slot);
    // a member's step failed or its request ended: it leaves its group (now when idle, else after its step)
    void leave(int32_t slot);

    // the groups: 2 (the pipeline), or 1 (--fnfl-split-groups 1: every writer in group A, up to eight, no overlap)
    void set_groups(int32_t n) { groups_ = n == 1 ? 1 : 2; }

    // the next action, given what is in flight; none when idle
    action next();
    // the action's completion (begin: sent; end: applied; front_read: its groups ran)
    void   done(const action & x);

    // state for checks and the startup line
    const std::vector<int32_t> & group(int32_t g) const { return members_[g]; }

    bool in_flight(int32_t g) const { return sent_[g]; }

    int32_t readers() const { return int32_t(readers_.size()); }

    std::string describe() const;

  private:
    void rebalance();
    bool wanted(int32_t slot) const;

    std::vector<int32_t> members_[2];
    std::vector<int32_t> want_;                 // the generating slots the caller gave
    bool                 sent_[2]  = { false, false };
    int64_t              order_[2] = { 0, 0 };  // when each group was sent (the older is ended first)
    int64_t              clock_    = 0;
    int32_t              last_     = 1;         // the group begun last
    std::vector<int32_t> readers_;
    size_t               next_reader_ = 0;      // the readers take the leader's read turns in rotation
    bool                 read_turn_   = false;  // a front read group is due before the next group action
    int32_t              groups_      = 2;
};

// pipe-test: the core against a fake leader and follower (CPU only): 1 .. 8 writers, joins and leaves at every point,
// cancels and ends of generation, readers interleaved and the follower's failure; every invariant checked at every
// action. Appends one line a case to report; true when all pass
bool self_test(std::string & report);

}  // namespace fn::runtime::pipe
