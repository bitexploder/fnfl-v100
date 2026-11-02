#pragma once

// TTTS1/TTTSBS: the leader's wire paths of the split (fn/runtime/stage.h has the protocol): its command channel and a
// step group's sends and answers. Host code only (no CUDA, no engine): the leader calls these, and the host safety
// tests (fn/runtime/safety_test.h) drive the same functions over the link's test network.
//
// Ownership (TTTSBS). A command's records (its X-CTRL, tokens and X-ACK) and a step group's (its X-CTRL, X-STEP
// answers and forced logits) live in one heap object, shared with the transport: every send and receive posted on
// it carries a reference (port's keep), so the object outlives every operation that can still read or write it,
// whatever the caller does. A failure never leaves an operation behind: every operation submitted for the command or
// the group is waited, or retired (port::retire: cancelled and its end observed), before the call returns. The
// boundary rows a group sends are read from the slot's pinned buffer, which the slot refills only at its next front
// round; that round follows the group's end (its sends finished or retired).

#include "fn/runtime/stage_msg.h"
#include "fn/xchg/link.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fn::runtime::stage {

uint64_t ctrl_tag(uint32_t seq, uint8_t part);
uint64_t ack_tag(uint32_t seq);
uint64_t boundary_tag(int32_t group, uint32_t seq, int32_t card, int32_t slot);
uint64_t step_tag(int32_t group, uint32_t seq, int32_t slot);
// a forced member's back logits (the drive's tests), and a probe's answer bytes
uint64_t logits_tag(int32_t group, uint32_t seq, int32_t slot);
uint64_t probe_tag(uint32_t seq);
uint64_t crossing_tag(int64_t n, int32_t piece);

// FNFL_SPLIT_TRACE=1: one stderr line a command and a step group's phase on either side (the protocol's order)
void trace(const char * who, const std::string & what);

struct leader_stats {
    int64_t groups = 0, members = 0, boundaries = 0, boundary_bytes = 0, crossings = 0, crossing_bytes = 0;
    double  back_wait_s = 0;  // the leader's time waiting for X-STEP and acknowledgments
};

// wait for every operation in ops (-1 entries skipped); on the first failure the rest are retired. Each entry is
// set to -1 once finished. false with err: the first failure's message
bool wait_all(port & p, std::vector<int32_t> & ops, double timeout_s, std::string & err);

// a command in flight: its records and its operations
struct cmd {
    cmd();
    ~cmd();
    cmd(const cmd &)             = delete;
    cmd & operator=(const cmd &) = delete;

    uint32_t             seq = 0;
    ctrl                 c;
    std::vector<int32_t> tokens;
    ack                  a;
    int32_t              ack_op = -1, c_op = -1, t_op = -1;
    bool                 answered = false;  // its X-ACK is in a
};

// the leader's command channel: sequence numbers, the lost stage's refusal, and each command's operations
class control {
  public:
    void init(port & p) { p_ = &p; }

    // post c (and n_tokens tokens): its X-ACK's receive first, then the X-CTRL and tokens sends, waited. nullptr when
    // the stage is lost (lost() then holds; why() the reason)
    std::shared_ptr<cmd> send(const ctrl & c, const int32_t * tokens);
    // the command's X-ACK (k.a): true when it came; a failure loses the stage
    bool                 wait(cmd & k, double timeout_s);
    // without waiting: done once the X-ACK came (then true); false: the stage is lost
    bool                 test(cmd & k, bool & done);
    // the answer is no longer wanted (the stage is lost or the command abandoned): its operations retired
    void                 drop(cmd & k);
    // send, then wait; the X-ACK copied into a
    bool                 command(ctrl & c, const int32_t * tokens, ack & a, double timeout_s);

    uint32_t take_seq() { return seq_++; }

    uint32_t last_seq() const { return seq_ - 1; }

    bool lost() const { return lost_; }

    const std::string & why() const { return why_; }

    // the stage lost: every later request refused (the refusal is not a memory fence: each path still retires its
    // own operations); the transport quiesced
    void lose(const std::string & why);

    double wait_s         = 0;     // time waiting for acknowledgments
    double send_timeout_s = 60.0;  // a command's sends' bound

    port * p() const { return p_; }

  private:
    port *      p_    = nullptr;
    uint32_t    seq_  = 1;
    bool        lost_ = false;
    std::string why_;
};

// a step group's wire records: the command and every member's answer (and a forced member's logits)
struct group_wire {
    group_wire();
    ~group_wire();
    group_wire(const group_wire &)             = delete;
    group_wire & operator=(const group_wire &) = delete;

    ctrl                 c;
    step_out             out[max_members];
    std::vector<float>   lg[max_members];
    int32_t              rops[max_members] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    int32_t              lops[max_members] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    std::vector<int32_t> sops;          // the command and the boundaries
    uint32_t             seq  = 0;
    bool                 sent = false;  // every send submitted (and, after group_sends, finished)
};

// one member's boundary rows: the slot's pinned buffer on each card, bytes each
struct boundary_src {
    const uint8_t * buf[2] = { nullptr, nullptr };
    size_t          bytes  = 0;
};

// the group to the follower: m[0 .. n) its members (kind < 0: not sent), the answers' receives posted first (a forced
// member's logits too: logits_n floats), then the command, then the fronted members' boundaries (src[k], when
// src[k].bytes > 0). Returns w.sent; a submission failure loses the stage and retires what was submitted
bool post_group(control &                           ctl,
                const std::shared_ptr<group_wire> & w,
                int32_t                             group,
                const member *                      m,
                int32_t                             n,
                const boundary_src *                src,
                const size_t *                      logits_n,
                leader_stats &                      st);
// the group's sends finished (the slot's boundary buffers free to refill); false: lost, every operation retired
bool group_sends(control & ctl, group_wire & w, double timeout_s);
// member k's X-STEP (and logits) arrived in w; false with e: lost, every operation retired
bool group_answer(control & ctl, group_wire & w, int32_t k, double timeout_s, std::string & e);
// the group's end: every operation not finished retired
void group_close(control & ctl, group_wire & w);

// the box-to-box link (the caller keeps it alive)
std::unique_ptr<port> link_port(fn::xchg::link::link & l);
// split-test's in-process pair: a send completes into the peer's matching receive by a host copy
void                  loop_pair(std::unique_ptr<port> & a, std::unique_ptr<port> & b);

}  // namespace fn::runtime::stage
