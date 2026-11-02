#pragma once

// TTTSBS: the link's test hooks (-DFNFL_TEST_HOOKS only: the host safety tests; never in a serving build). They drive
// the production link (fn/xchg/link.h: its submission queue, link thread, operation slots, waits, cancellation and
// teardown) over an in-process delayed-copy network in place of UCX, so a fault schedule is deterministic:
//   pair            two links joined in this process (no UCX): a send is matched to the peer's posted receive with the
//                   full tag, as UCP's tag matching; a message with no receive waits as an unexpected one
//   delayed copy    a matched transfer's bytes are copied by the receiver's link thread when the transfer runs (the
//                   receive's completion), reading the sender's buffer then; the send completes after that copy
//   hold_transfers  matched transfers whose receive tag matches (mask, value) wait until release_transfers; stuck:
//                   they also survive a forced close (a transfer the transport cannot stop), else a forced close or
//                   a lost peer ends them with an error and no copy
//   hold_thread     the link thread parks before its next queue entry (its cancels and posts wait) until released
//   fail_submit     the n-th submission from now (1: the next) is refused, as a full operation table would be
//   on_cancel       a callback on the link thread as it takes a cancel, before acting on it
//   kill            the peer is lost on both sides: every operation not stuck ends with an error
// Receives are cancellable until matched; sends are not cancellable (UCP's eager and rendezvous sends end by
// completing or by the endpoint's error).

#include "fn/xchg/link.h"

#include <cstdint>
#include <functional>
#include <string>

#if defined(FNFL_TEST_HOOKS)

namespace fn::xchg::link::fault {

struct net;

// a and b joined in this process, a the leader; both started (their X-HELLO identities taken as exchanged)
bool    pair(link & a, link & b, const options & oa, const options & ob, std::string & err);
void    hold_transfers(link & receiver, uint64_t mask, uint64_t value, bool stuck);
// every held transfer of this receiver may run (stuck ones too)
void    release_transfers(link & receiver);
// matched transfers waiting on a hold (the receiver's)
int32_t held_transfers(link & receiver);
// messages sent to this side that no receive has matched
int32_t unexpected_count(link & receiver);
void    hold_thread(link & l, bool on);
// returns once the link thread is parked (hold_thread(l, true) first)
void    wait_parked(link & l);
void    fail_submit(link & l, int32_t nth);
// f runs on the link thread as it takes each cancel entry, before the cancel (a completion racing the cancel)
void    on_cancel(link & l, std::function<void()> f);
void    kill(link & l);
// bytes the delayed copies wrote, and transfers that ended without a copy (forced close, a lost peer)
int64_t copies(link & l);
int64_t dropped(link & l);

}  // namespace fn::xchg::link::fault

#endif
