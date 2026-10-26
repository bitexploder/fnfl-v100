#pragma once

// TTTS1: the split's drive commands (fn/runtime/stage.h). fnfl-drive stage: the follower, the back stage on this box
// (layers k .. 47 with the head and the draft), listening for the leader and serving its commands until X-BYE. The
// leader's side of spec run, rollback-test and joint-kl (--split) uses the session helpers below.

#include "fn/runtime/runtime.h"
#include "fn/runtime/stage.h"
#include "fn/runtime/target_cmds.h"
#include "fn/xchg/link.h"

#include <memory>
#include <string>

// one process's split session: the link, its port, the boundary buffers, the prompt crossing, the leader or follower
struct fnfl_split {
    fn::xchg::link::link                          link;
    std::unique_ptr<fn::runtime::stage::port>     port;
    fn::runtime::stage::buffers                   buf;
    fn::runtime::stage::crossing                  cross;
    std::unique_ptr<fn::runtime::stage::leader>   lead;
    std::unique_ptr<fn::runtime::stage::follower> follow;
};

// before the engine: the boundary buffers (pinned, not yet mapped) and the engine's stage fields
bool fnfl_split_prepare(const fnfl_target_args &     a,
                        int32_t                      n_slots,
                        fnfl_split &                 s,
                        fn::runtime::engine_config & c,
                        std::string &                err);
// after the engine's init: connect (the leader) or listen (the follower) with X-HELLO's identities (a's commit and
// digest; the model files' heads and the engine's settings hashed; k; the slots), the buffers mapped, the crossing on
// the engine's prefiller, the leader or follower made. Prints the transport line
bool fnfl_split_attach(const fnfl_target_args & a,
                       fn::runtime::engine &    e,
                       int32_t                  cells,
                       fnfl_split &             s,
                       std::string &            err);

// fnfl-drive stage: the follower process (a.split_role 2)
int fnfl_stage(const fnfl_target_args & a);
