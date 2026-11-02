#pragma once

// TTTS1 link-test: the box-to-box link (fn/xchg/link.h) between a leader and a follower, one process on each box.
// No model. Both sides run the same case list in the same order and agree on every payload from its seed:
//   connect    the X-HELLO check and the transport line (a serving link shows rc_mlx5);
//   decode     X-BOUNDARY decode crossings of 1 .. 20 rows (lanes of up to 5 rows, a slot's fixed buffer each) over
//              every slot buffer, both groups and both cards, with pattern payloads holding the canonical NaN, both
//              infinities, denormals and -0; the follower copies each lane to its card, reads it back and compares
//              bytes;
//   prompt     the prompt form, chunks of 1 .. 8,192 rows in pieces of 1,024 rows through four piece buffers, each
//              piece copied to both cards and compared;
//   small      X-STEP, X-CTRL and X-ACK both ways;
//   stream     100,000 decode messages alternating the groups, every one compared, zero mismatches;
//   refuse     a message one byte longer, one byte shorter, and one with a tag nobody receives: each refused, named;
//   retire     (TTTSBS) a receive's wait times out with nothing sent; its buffer is reused at once; the message sent
//              after it arrives unexpected and intact, the buffer untouched;
//   kill       (last) the follower kills itself mid-transfer; the leader's wait ends within its bound, named;
//   stop       (TTTSBS, only when named, instead of kill) the leader stops its link with a receive and a 40 MiB send
//              outstanding; the stop ends them before unmapping, within its bound; the follower's wait ends.
// --host (TTTSBS): plain host buffers, every payload compared on the host, no card touched (the host safety build,
// fnfl-safety, has only this mode); the transport line is printed, not required.
//   time       (TTTS7; only when named, never in "all") the link's own cost at the split's real sizes, as the stage
//              orders its messages: a step group's hand-off at U = 2 and U = 4 lanes of time_rows rows (the leader
//              posts the members' X-STEP receives, sends X-CTRL and a boundary a member and card; the follower takes
//              X-CTRL, then the boundaries, then answers each member's X-STEP), timed on the leader from its first
//              post to the last X-STEP; and a whole prompt chunk's crossing (its head, waited, then pieces of at most
//              piece_bytes), timed from the head to the follower's acknowledgment of the last piece. No compute and no
//              card copies.
// Only the time case prints timings; the other cases print none, and none of theirs are read (the ticket's rule).
// Exit 0 when every case passes on this side.

#include <cstdint>
#include <string>

namespace fn::xchg::link {

struct test_options {
    bool        leader          = true;
    std::string addr            = "10.25.0.2";
    uint16_t    port            = 13371;
    double      timeout_s       = 10.0;
    std::string cases           = "all";  // all, or a comma list of the case names above
    int32_t     device[2]       = { 0, 1 };
    int64_t     stream_messages = 100000;
    bool        host            = false;  // --host
    // the time case: the stage's sizes (the drive passes model::stage_row_bytes, sizeof(stage::ctrl),
    // sizeof(stage::step_out), a whole prompt chunk's crossing_bytes and stage::piece_bytes), the lanes' rows, and
    // its samples: untimed then timed hand-offs at each U, untimed then timed chunks
    int64_t     row_bytes = 25616, ctrl_bytes = 512, step_bytes = 512, head_bytes = 48;
    int64_t     chunk_bytes = 0, piece_bytes = int64_t(42) << 20;
    int32_t     time_rows = 3, warm_rounds = 20, rounds = 200, warm_chunks = 1, chunks = 5;
};

int run_test(const test_options & o);

}  // namespace fn::xchg::link
