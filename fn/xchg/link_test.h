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
//   kill       (last) the follower kills itself mid-transfer; the leader's wait ends within its bound, named.
// It prints no timings; none are read (the ticket's rule). Exit 0 when every case passes on this side.

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
};

int run_test(const test_options & o);

}  // namespace fn::xchg::link
