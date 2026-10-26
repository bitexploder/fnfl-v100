#include "fn/runtime/stage_cmds.h"

#include "fn/model/model.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rt = fn::runtime;
namespace st = fn::runtime::stage;
namespace lk = fn::xchg::link;
namespace md = fn::model;

bool fnfl_split_prepare(const fnfl_target_args & a,
                        int32_t                  n_slots,
                        fnfl_split &             s,
                        rt::engine_config &      c,
                        std::string &            err) {
    if (a.split_role != 1 && a.split_role != 2) {
        err = "split: no stage";
        return false;
    }
    if (!s.buf.alloc(n_slots, nullptr, err)) {
        return false;
    }
    c.stage       = a.split_role;
    c.stage_buf   = s.buf.slot;
    c.record_hash = a.record_hash && a.split_role == 2;
    return true;
}

bool fnfl_split_attach(const fnfl_target_args & a, rt::engine & e, int32_t cells, fnfl_split & s, std::string & err) {
    const bool  leader = a.split_role == 1;
    lk::options lo;
    lo.leader    = leader;
    lo.addr      = a.split_addr;
    lo.port      = static_cast<uint16_t>(a.split_port);
    lo.timeout_s = 60.0;
    lo.connect_s = 1800.0;  // the leader retries while the follower loads; the follower waits as long
    lk::set_text(lo.me.commit, sizeof lo.me.commit, a.split_commit);
    lk::set_text(lo.me.digest, sizeof lo.me.digest, a.split_digest);
    const rt::engine_config & c = e.config();
    const std::string         settings =
        st::hello_settings(a.split_k, e.n_slots(), cells, a.pool_cells, a.kv_q8, c.spec_d, c.draft.attention, c.eog);
    printf("split: the shared settings: %s\n", settings.c_str());
    lk::set_text(lo.me.model, sizeof lo.me.model, st::hello_files(e.test_model()) + "/" + st::hello_hash(settings));
    lo.me.split = a.split_k;
    lo.me.slots = e.n_slots();
    printf("split: %s %s:%d, k %d, %d slots; UCX_TLS=%s UCX_NET_DEVICES=%s UCX_IB_GID_INDEX=%s\n",
           leader ? "the leader, connecting to" : "the follower, listening on", a.split_addr.c_str(), a.split_port,
           a.split_k, e.n_slots(), getenv("UCX_TLS") ? getenv("UCX_TLS") : "(unset)",
           getenv("UCX_NET_DEVICES") ? getenv("UCX_NET_DEVICES") : "(unset)",
           getenv("UCX_IB_GID_INDEX") ? getenv("UCX_IB_GID_INDEX") : "(unset)");
    fflush(stdout);
    if (!s.link.start(lo, err)) {
        return false;
    }
    printf("split: connected (X-HELLO: commit %s, device code %s, model and settings %s); transport %s\n",
           a.split_commit.empty() ? "(none given)" : a.split_commit.c_str(),
           a.split_digest.empty() ? "(none given)" : a.split_digest.c_str(), lo.me.model, s.link.transport().c_str());
    fflush(stdout);
    s.port = st::link_port(s.link);
    for (auto & sl : s.buf.slot) {
        for (uint8_t * p : sl) {
            if (!s.port->map(p, size_t(md::stage_bytes(fn::round::max_rows)), err)) {
                return false;
            }
        }
    }
    if (!s.cross.init(*s.port, leader, a.split_k, err)) {
        return false;
    }
    e.prefill_driver().set_stage(&s.cross);
    if (leader) {
        s.lead = std::make_unique<st::leader>();
        return s.lead->init(e, *s.port, s.cross, err);
    }
    s.follow = std::make_unique<st::follower>();
    return s.follow->init(e, *s.port, s.cross, err);
}

int fnfl_stage(const fnfl_target_args & a) {
    if (a.split_role != 2 || a.split_k < 2 || a.split_k > fn::n_layer - 1) {
        fprintf(stderr, "stage: the follower of a split at k in 2 .. 47 (--split-layer K)\n");
        return 2;
    }
    rt::draft_options dopt;
    if (!fnfl_draft_settings(a, true, "stage", dopt)) {
        return 1;
    }
    const int32_t cells = a.n_cells;
    auto          m     = std::make_unique<md::model>();
    if (a.mode == "dry" ? !fnfl_load_dry(a, cells, *m) : !fnfl_load(a, cells, *m, "stage")) {
        return 1;
    }
    if (a.mode == "dry") {
        printf("stage: DRY: zero weights, no model data on the cards (plumbing only)\n");
    }
    rt::engine_config c;
    c.model_path = a.model_path;
    c.n_ctx_slot = cells;
    c.draft      = dopt;
    c.eog        = a.eog.empty() ? std::vector<int32_t>{ 248046 } : a.eog;
    c.spec_d     = a.spec_d;
    fnfl_split  s;
    std::string err;
    if (!fnfl_split_prepare(a, a.n_slots, s, c, err)) {
        fprintf(stderr, "stage: %s\n", err.c_str());
        return 1;
    }
    auto e = std::make_unique<rt::engine>();
    if (!e->init(std::move(m), c, err)) {
        fprintf(stderr, "stage: engine: %s\n", err.c_str());
        return 1;
    }
    printf("stage: the back (layers %d .. 47, the head and the draft): %s\n", a.split_k, e->settings().c_str());
    fflush(stdout);
    if (!fnfl_split_attach(a, *e, cells, s, err)) {
        fprintf(stderr, "stage: %s\n", err.c_str());
        return 1;
    }
    // the drive's tests: the back's slot state hashes (rollback-test mu1), and the engine rebuilt on the leader's
    // order (its state laid out again in the given KV format, the d and the joint switches; joint-kl)
    s.follow->probe_fn = [&e](int32_t slot, int32_t what, std::vector<uint8_t> & out, std::string & perr) {
        if (what != 0 || slot < 0 || slot >= e->n_slots()) {
            perr = "probe " + std::to_string(what) + " of slot " + std::to_string(slot);
            return false;
        }
        if (!fnfl_mu1_slot_hash(*e, slot, out)) {
            perr = "slot " + std::to_string(slot) + "'s state hash";
            return false;
        }
        return true;
    };
    const rt::engine_config c0 = c;
    s.follow->reconfigure_fn   = [&e, &s, c0](const st::ctrl & rc, std::string & rerr) -> rt::engine * {
        const int32_t              slots = e->n_slots();
        std::unique_ptr<md::model> mm    = e->test_take_model();
        e.reset();
        const int32_t rcells = rc.base > 0 ? int32_t(rc.base) : c0.n_ctx_slot;
        if (mm == nullptr || !fnfl_lay_out(*mm, rcells, slots, rc.a != 0)) {
            rerr = "the state's layout";
            return nullptr;
        }
        rt::engine_config c2 = c0;
        c2.n_ctx_slot        = rcells;
        c2.spec_d            = rc.b > 0 ? rc.b : c0.spec_d;
        c2.joint_x16         = (rc.c & st::sw_joint_x16) != 0;
        c2.joint_router_tc   = (rc.c & st::sw_router_tc) != 0;
        c2.joint_dense_f16   = (rc.c & st::sw_dense_f16) != 0;
        c2.eager             = (rc.c & st::sw_eager) != 0;
        e                    = std::make_unique<rt::engine>();
        if (!e->init(std::move(mm), c2, rerr)) {
            return nullptr;
        }
        e->prefill_driver().set_stage(&s.cross);
        printf("stage: rebuilt: %s\n", e->settings().c_str());
        fflush(stdout);
        return e.get();
    };
    std::string        why;
    const bool         bye = s.follow->serve(why);
    const lk::counters t   = s.link.totals();
    printf(
        "stage: %s; link: sent %llu messages / %llu bytes, received %llu / %llu; prompt crossings %lld (%lld "
        "bytes)\n",
        bye ? "the leader said X-BYE" : ("the leader was lost: " + why).c_str(), (unsigned long long) t.sent_msgs,
        (unsigned long long) t.sent_bytes, (unsigned long long) t.recv_msgs, (unsigned long long) t.recv_bytes,
        (long long) s.cross.count(), (long long) s.cross.bytes());
    fflush(stdout);
    return bye ? 0 : 1;
}
