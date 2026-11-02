#pragma once

// TTTSBS safety-test: the split's buffer lifetimes on the failure paths, host only (no CUDA, no model, no GPU). Each
// case runs the production link (fn/xchg/link.h) over its in-process delayed-copy test network (fn/xchg/link_fault.h)
// and the leader's production wire paths (fn/runtime/stage_wire.h), with a fixed fault schedule, and checks the
// lifetime ledger (fn/xchg/life.h): no owner destroyed, unmapped or reused while an operation can still touch it,
// every operation's end observed, every owner eventually reclaimed, and each payload's bytes against its own
// (slot, request generation, sequence, card) pattern. Cases:
//   normal         two groups of two members with per-slot boundaries; the answers consumed in the other order
//   partial        a group's fourth submission refused after its two answer receives and its X-CTRL send went; the
//                  X-CTRL send outstanding and an answer arriving after the failure was reported
//   queued         a command's wait times out before the link thread posts its operations (the thread held);
//                  its records on the caller's stack; the X-ACK sent after
//   posted         an answer's receive posted and matched, its copy held, the link thread held before the cancel:
//                  the record dropped by the caller early; non-stuck (a forced close ends it) and stuck (the copy
//                  lands later, into the record the link kept)
//   race           a completion racing the cancel of a timed-out wait; then a stale handle waited after its slot
//                  was reused by a newer operation
//   final          the last member of a group fails (a longer answer) while a second group's answers are held
//   ack            a command's X-ACK delayed past its wait's failure; its ack and tokens on the caller's stack
//   shutdown       a lost peer with a group outstanding; then a link stopped with a send and a receive outstanding
//                  on its mapped buffer (unmapping after quiescence)
//   reuse          a cancelled request's slot reused: a stale X-STEP of the old sequence never lands in the new
//                  group's record; a slot's boundary buffer refilled for the next request after a failed group
//                  receive whose copy was held
// Prints the schedule and outcome of each case and the ledger's totals; no timings. Exit 0 when every case passes.

#include <string>

namespace fn::runtime {

// cases: "all" or a comma list of the names above
int fnfl_safety_test(const std::string & cases);

}  // namespace fn::runtime
