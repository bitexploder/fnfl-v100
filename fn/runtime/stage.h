#pragma once

// TTTS1: the split's two stages. Flash-Next across two boxes: tensor parallel across each box's two cards,
// the layers split between the boxes at k (model::window). Each box runs its own engine (fn/runtime/runtime.h) on its
// window: the leader's front (layers 0 .. k - 1, the embedding and PLE; the server, the tokenizer and the scheduler
// stay on its box), the follower's back (layers k .. 47, the head, the decision, the commit and the MTP draft). The
// leader drives; the follower executes what arrives, in arrival order, and answers each command.
//
// Messages (fn/xchg/link.h's tags; every receive's size is known before it is posted):
//   X-CTRL     leader -> follower, a fixed ctrl record (tag ctrl, group 0, the command's sequence), its tokens (if
//              any) in a second message (group 1, the same sequence): set_sampling, prefill, read_begin, read_step,
//              read_cancel, truncate, release_slot, pool, step_group, bye.
//   X-ACK      follower -> leader, an ack record for each command but step_group (tag ack, the command's sequence).
//   X-BOUNDARY a step group's members' front rounds, one message a member and card (tag boundary, the group, the
//              command's sequence, the card, the slot; model::stage_bytes(rows) from the slot's send buffer into the
//              follower's receive buffer of the slot).
//   X-STEP     follower -> leader, one step_out record a member (tag step, the group, the command's sequence, the
//              slot): the step's outcome (or its failure) and the member's next step (none, generate, or a
//              speculative step with the d proposals the back's draft made).
//   X-BOUNDARY (prompt form) a chunk's crossing (model::prompt_path::crossing_out), a crossing_head record (tag
//              prompt, crossing number n, aux 255) and pieces of at most piece_bytes (aux 0, 1, ..). A read's chunks
//              take the crossing numbers base + chunk index; the leader picks base and sends it in the read's command.
//
// A step group (step_group): 1 .. 8 members, each its slot's next step. A member whose next step is known (the last
// X-STEP's) runs its front round on the leader (a verify of the proposals, or a generate round; the full-d verifies of
// two or more members as one joint round when the group says so), then the follower runs the back half, the decision,
// the commit, the end-of-generation cut and the draft's catch-up (joint where the members fit), then each member's
// next step's draft (joint where they fit), and answers. A member whose next step is not known (after a prefill, a
// truncate or a release: begin) runs nothing on the leader: the follower runs the bootstrap (a whole step) or only the
// next step's draft. The leader then commits its front rounds with the back's verdict (target::front_end).
//
// The four steps of each crossing are fn/xchg/link.h's. The leader's send buffers are refilled only after their sends
// complete; the follower's receive buffers are posted before the data is expected and reused only after their graph
// copied them in (stream order: the next round of the slot follows the copy). Every wait is bounded; a lost peer
// fails every waiting call with "split stage lost: <the message>".
//
// No model constant lives here beyond the boundary's and the crossing's sizes (fn/model/pass.h, prompt.h), so the
// GLM port reuses the protocol.

#include "fn/runtime/pace.h"
#include "fn/runtime/pipe.h"
#include "fn/runtime/prefill.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/stage_msg.h"
#include "fn/runtime/stage_wire.h"
#include "fn/xchg/link.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace fn::runtime::stage {

// ---- X-HELLO ----

// X-HELLO's identities, computed the same way by every leader (the drive's, the server's) and the follower: the model
// files (each file's size and first 16 MiB: its GGUF header and tensor table; a 128-bit FNV-1a, an identity check,
// not a security boundary), and the settings both stages must share, as text (each side prints its own) and hashed.
// The hello's model field is hello_files(m) + "/" + hello_hash(the settings text)
std::string hello_files(const model::model & m);
std::string hello_settings(int32_t                      k,
                           int32_t                      n_slots,
                           int32_t                      cells,
                           int32_t                      pool_cells,
                           bool                         kv_q8,
                           int32_t                      spec_d,
                           const std::string &          attention,
                           const std::vector<int32_t> & eog,
                           bool                         images = false);  // VIS1: the front reads images
std::string hello_hash(const std::string & text);

// ---- the prompt crossing (prefiller's pf_stage) ----

class crossing : public pf_stage {
  public:
    // front: the leader's (sends); else the follower's (receives); k the split layer; bytes the largest crossing
    bool init(port & p, bool front, int32_t k, std::string & err);
    ~crossing() override;
    // the next read of slot takes crossing numbers base + its chunk index
    void expect(int32_t slot, int64_t base);
    bool out(target & t, model::prompt_path & pp, const model::prompt_chunk & c, std::string & err) override;
    bool in(target & t, model::prompt_path & pp, model::prompt_chunk & c, std::string & err) override;
    // the leader: every crossing's sends completed; PRD1 item 9: with a slot, those of the buffers that slot filled
    // last (before its read reports done)
    bool drain(std::string & err);
    bool drain(int32_t slot, std::string & err);

    // the leader: the crossings sent for slot's current read (its chunks' count); abort, when set, is asked before
    // each crossing: true stops the read there (the follower already failed it)
    int64_t sent(int32_t slot) const { return slot >= 0 && slot < max_split_slots ? sent_[slot] : 0; }

    // the follower: the crossings slot's current read has taken
    int64_t taken(int32_t slot) const { return slot >= 0 && slot < max_split_slots ? taken_[slot] - base_[slot] : 0; }

    std::function<bool()> abort;
    // the follower: slot's crossings base + i for i below count that it has not taken, received and dropped
    bool                  drop(int32_t slot, int64_t base, int64_t count, std::string & err);

    // PRD1 item 11 (the follower): the received-ahead crossing. plan: the slot's read has chunks chunks (set after
    // its read_begin; 0, as a prefill, takes none ahead). When chunk n's copy-in starts, chunk n + 1's header receive
    // is posted if it is in the plan; poll (between units, never waiting) posts its pieces once the header landed,
    // next says no other chunk's copy-in comes before it, and its buffer's previous copy-in completed. in validates a
    // header taken ahead as any other. retire_ahead ends a receive taken ahead for slot (the read's end or cancel)
    void plan(int32_t slot, int64_t chunks);

    int32_t ahead_slot() const { return ahead_.slot; }

    bool poll(bool next, std::string & err);
    void retire_ahead(int32_t slot);

    // the chunks of slot's current read whose pieces had all landed before in was entered (the follower's)
    int64_t received_ahead(int32_t slot) const override {
        return front_ ? -1 : slot >= 0 && slot < max_split_slots ? ahead_done_[slot] : 0;
    }

    // totals: crossings and bytes
    int64_t count() const { return n_; }

    int64_t bytes() const { return bytes_; }

    double timeout_s = 600.0;

  private:
    port *               p_      = nullptr;
    bool                 front_  = true;
    int32_t              k_      = 0;
    uint8_t *            buf_[2] = { nullptr, nullptr };
    int64_t              cap_    = 0;
    std::vector<int32_t> ops_[2];                       // the sends (front) of each buffer
    int32_t              owner_[2] = { -1, -1 };        // the slot whose crossing filled each buffer last (front)
    void *               done_[2][2]             = {};  // the back: each buffer's copy-in completed, on each card
    bool                 used_[2]                = { false, false };
    int64_t              base_[max_split_slots]  = {};
    int64_t              sent_[max_split_slots]  = {};  // the leader: crossings sent for the slot's read
    int64_t              taken_[max_split_slots] = {};  // the follower: the next crossing number the slot's read takes
    int64_t              n_ = 0, bytes_ = 0;

    // PRD1 item 11: the crossing taken ahead (slot -1: none), its header's receive (head: landed) and its pieces'
    struct ahead_rx {
        int32_t                        slot = -1;
        int64_t                        n    = -1;
        int32_t                        hop  = -1;
        bool                           head = false, posted = false;
        std::shared_ptr<crossing_head> hr;
        std::vector<int32_t>           ops;
    };

    ahead_rx ahead_;
    int64_t  plan_[max_split_slots]       = {};
    int64_t  ahead_done_[max_split_slots] = {};
    int32_t  dev_[2]                      = { -1, -1 };  // the back's cards (the copy-in events' devices)

    // every send of both buffers ended (retired) after a failure: none reads a buffer refilled later
    void retire_sends();
    // a header's pieces' receives into buffer b (posted; their ops in ops)
    bool post_pieces(const crossing_head & h, int64_t n, int32_t b, std::vector<int32_t> & ops, std::string & err);
    // the buffer's previous copy-in completed on both cards (wait: synchronize; else query)
    bool buffer_free(int32_t b, bool wait, std::string & err);
};

// ---- the boundary buffers ----

// pinned host memory for every slot's boundary (model::stage_bytes(5) a card), mapped on the port
struct buffers {
    std::vector<std::array<uint8_t *, 2>> slot;
    ~buffers();
    bool alloc(int32_t n_slots, port * p, std::string & err);
};

// ---- the leader (the front engine's driver) ----

// TTTS3: the split's joint refusals. A failed graph capture turns that joint path off for the engine's life (the
// one-box path's rule, runtime.cpp); every refusal's message is printed once per distinct cause, not once a round.
struct joint_refusals {
    bool                  off = false;  // a capture failed: no more joint rounds on this path
    std::set<std::string> said;
    // record a refusal: latch on a failed capture, then print "<tag>: <cause>; <tail>" if this cause is new
    void                  refuse(const char * tag, const std::string & cause, bool capture_failed, const char * tail);
};

class leader {
  public:
    // e: the front engine (engine_config::stage 1, its stage_buf b's); p: the link; x: the front's crossing, set on
    // the engine's prefiller
    bool init(engine & e, port & p, crossing & x, std::string & err);

    run_status set_sampling(int32_t slot, const request_options & o);
    // VIS1 Stage B: images as engine::prefill's; their spans (not their pixels) go to the follower after the tokens
    run_status  prefill(int32_t                           slot,
                        const int32_t *                   tokens,
                        int32_t                           n_tokens,
                        int32_t                           n_reuse,
                        int32_t &                         n_processed,
                        const std::vector<prompt_image> * images = nullptr);
    int32_t     read_begin(int32_t                           slot,
                           const int32_t *                   tokens,
                           int32_t                           n_tokens,
                           int32_t                           n_reuse,
                           int32_t &                         n_processed,
                           const std::vector<prompt_image> * images = nullptr);
    // MUP2: writers mode with a budget: the front's units under this box's budget, and the back's chunks budgeted on
    // the follower (APF1: by its own writer-pause deadline under the budget's P)
    // PRD1 item 9: beside writers or other readers, front_done once the front's read ended with its back halves out;
    // the slot is then polled: read_ready (no wait: every back half answered), read_poll (its end, or front_done)
    read_status read_step(int32_t slot, int32_t n_groups, read_mode mode, const read_budget * budget = nullptr);
    bool        read_ready(int32_t slot);
    read_status read_poll(int32_t slot);
    void        read_cancel(int32_t slot);
    // MUP2 piece 4: the back's units account of slot's read so far (its last answered back chunk's)
    void        back_info(int32_t slot, read_info & out) const;
    run_status  truncate(int32_t slot, int32_t n_keep);
    run_status  release_slot(int32_t slot);
    // the draft's d and form (engine::set_draft) on both boxes, between requests
    run_status  set_draft(int32_t d, bool sparse);
    // MUP1 piece 5: the slot's rotary position offset (engine::set_pos_offset) on both boxes, between its requests
    run_status  set_pos_offset(int32_t slot, int32_t off);
    // one step of each of n slots (1 .. 8), as engine::step_joint (the full-d verifies of 2 .. 4 lanes as joint
    // rounds); synchronous: returns with every member's step
    void        step_group(const int32_t * slots, int32_t n, joint_step_out * res);
    // the follower's pool
    bool        follower_pool(pool_info & out);
    // an orderly end: X-BYE, answered
    void        bye();

    bool lost() const { return ctl_.lost(); }

    const std::string & lost_why() const { return ctl_.why(); }

    const leader_stats & stats() const { return st_; }

    // PRD1 item 4 (TEST): the front round's joint verify refusals (the latch and the causes said)
    const joint_refusals & test_joint_verify() const { return joint_verify_; }

    const std::string & error() const { return err_; }

    // APF1: what the next step groups carry to the follower (the pipeline sets it before each begin): P, the front's
    // groups (bit g) and their generations; and the last ended group's back stage (ms, from its answers; -1: none)
    void set_pace(int32_t pause_ms, uint32_t live, const int32_t * gen);

    double last_back_ms() const { return last_back_ms_; }

    // the pipelined group step's halves (fn/runtime/pipe.h): begin runs the members' front rounds and sends the group,
    // end waits for its answers and closes the members' steps; one group in flight per member slot
    struct pending;
    std::unique_ptr<pending> group_begin(const int32_t * slots, int32_t n, int32_t group);
    void                     group_end(pending & g, joint_step_out * res);

    // ---- the drive's tests (rollback-test mu1, joint-kl) ----
    // a read's end waits for the back's chunks (one box's statuses call by call); off in serving
    void set_block_reads(bool on) { block_reads_ = on; }

    // the back's answer to a probe of slot (what: the drive's own meaning), its bytes
    bool                     probe(int32_t slot, int32_t what, std::vector<uint8_t> & out);
    // after the leader rebuilt its engine (rebind): the follower's rebuilt alike (c: op::reconfigure's fields)
    run_status               reconfigure(ctrl c);
    void                     rebind(engine & e);
    // both boxes' slot reset, its ledger the tokens (none processed); MUP2 keep: no reset, the processed tokens kept
    // (target::keep_processed) and the tokens appended to the ledger (joint-kl at depth)
    run_status               test_prompt(int32_t slot, const std::vector<int32_t> & tokens, bool keep = false);
    // APF2: both boxes' forced prompt routes (prefiller::test_force_route) for the chunks that start after the call;
    // back: the follower process's executed-variant totals (prompt_route_totals; collected from the first call)
    run_status test_force_route(const model::prompt_route_force & f, model::prompt_path::route_counts * back = nullptr);
    // a forced round of rows rows on each of n slots (2 .. 4 of the same rows: a joint round) as a step group; its
    // members' back logits (forced_logits) after group_end: card 0's rows x n_vocab_card F32, then card 1's
    std::unique_ptr<pending> group_begin_forced(const int32_t * slots, int32_t n, int32_t rows, int32_t group);

    const std::vector<float> & forced_logits(int32_t slot) const { return logits_[size_t(slot)]; }

    // groups of forced members (sizes[g] slots each, in order in slots): each begun in turn, then each ended in
    // turn (two groups: pipelined, the second's front half beside the first's back half); res in slots' order
    void forced_groups(const int32_t *  slots,
                       const int32_t *  sizes,
                       int32_t          n_groups,
                       int32_t          rows,
                       joint_step_out * res);

  private:
    struct next_step {
        bool    valid = false;
        int32_t kind = 0, d = 0;
        int32_t prop[4] = {};
    };

    // a command answered (its X-ACK in a); false: the stage is lost (err_)
    bool                 command(ctrl & c, const int32_t * tokens, ack & a, double timeout_s);
    std::shared_ptr<cmd> send_cmd(const ctrl & c, const int32_t * tokens);
    bool                 wait_cmd(cmd & k, double timeout_s);
    void                 dispatch(pending & g, int32_t group);
    // a failed read's crossings the follower did not take, dropped there (op::drain)
    void                 drain_read(int32_t slot, int64_t base);

    engine *                                            e_ = nullptr;
    port *                                              p_ = nullptr;
    crossing *                                          x_ = nullptr;
    std::vector<next_step>                              next_;
    // MUP1 piece 2: each slot's set_sampling command awaiting its X-ACK (sent without waiting; collected by the slot's
    // next command, read_begin's in the same follower turn, or a group's start when it has come)
    std::vector<std::shared_ptr<cmd>>                   samp_;
    run_status                                          settle(int32_t slot, bool wait);
    run_status                                          settle_all();
    std::vector<int64_t>                                read_base_;  // per slot: its read's first crossing number
    std::vector<std::vector<std::shared_ptr<cmd>>>      read_acks_;  // per slot: its read's back halves' commands
    // per slot: the front's read ended with this status while the back's chunks may still run (-1: none)
    std::vector<int32_t>                                read_end_;
    std::vector<read_info>                              back_acct_;  // per slot: MUP2's back account (back_info)
    // the read's end once the back's chunks are answered: blocking, or yielded while one is not
    read_status                                         read_finish(int32_t slot, bool block);
    bool                                                block_reads_ = false;
    std::vector<std::vector<float>>                     logits_;  // per slot: its last forced round's back logits
    control                                             ctl_;     // the command channel and the lost stage
    int64_t                                             base_ = 0;
    std::string                                         err_;
    leader_stats                                        st_;
    joint_refusals                                      joint_verify_;  // TTTS3: the front round's joint verify
    int32_t                                             pace_pause_   = pace::pause_default_ms;  // APF1 (set_pace)
    uint32_t                                            pace_live_    = 0;
    int32_t                                             pace_gen_[2]  = { 0, 0 };
    double                                              last_back_ms_ = -1;
};

// ---- the leader's pipeline (fn/runtime/pipe.h's core over the leader's group halves) ----

// TTTS8: the server's split calls (fn/glue's split branches) are this class's: each glue entry naming a slot forwards
// to one method here, so split-test's deliver case drives the server's own code. A step the engine finished for a
// slot reaches that slot's request exactly once, in order, whichever call takes it: a group's results are queued by
// slot when the group ends, each call that steps a slot takes its oldest first, and only end_request (the request
// ended) drops them. A slot whose request delivered its end (end of generation or a failure) steps again only after
// end_request: a step after it is refused, never run
class pipeline {
  public:
    explicit pipeline(leader & l);
    ~pipeline();  // its groups in flight dropped unanswered (settle_all first)

    // 2: the two pipelined groups; 1: one group (--fnfl-split-groups 1)
    void set_groups(int32_t n) { core_.set_groups(n); }

    // one turn over the generating slots (1 .. 8, distinct): each idle group with members begins (its front half
    // and its send), then the older group in flight ends (its answers applied). done[k]: slots[k]'s step finished in
    // this turn (res[k] its); false while its group is in flight. A step settle finished is returned at once (no
    // group begins in that turn)
    void step(const int32_t * slots, int32_t n, joint_step_out * res, bool * done);
    // one slot's step outside the pipeline (fnfl_step: the shell's lone writer): the slot's oldest finished step not
    // yet delivered (its group in flight ends first) is this call's; with none, every group in flight ends (their
    // members' steps queued for them) and the slot steps alone, a group of one, synchronous
    void    step_one(int32_t slot, joint_step_out & res);
    // slot's group, when in flight, ends now; its members' results are kept for their next step call
    void settle(int32_t slot);
    void settle_all();
    // the slot's request ended (end of generation, a stop, a cancel, a failure or a release; call names the server's
    // call for the line): its group in flight ends and its finished steps not delivered are dropped, the only place
    // they are, with a line when any was; returns how many
    int32_t end_request(int32_t slot, const char * call = "end_request");
    bool    in_flight(int32_t slot) const;
    // the slot's finished steps not yet delivered
    int32_t queued(int32_t slot) const;

    // APF1: the front's writer-pause book (fn/runtime/pace.h): each group's begin and send, its answers, an emptied
    // group, a slot's request end; fnfl_read_step_paced decides the front's read budget from it
    pace::pacer & pace() { return pace_; }

    const pace::pacer & pace() const { return pace_; }

    // the server's calls that name a slot between its requests (fnfl_set_sampling, fnfl_prefill, fnfl_read_begin,
    // fnfl_truncate, fnfl_release_slot): the slot's request ended first (end_request), then the leader's call
    run_status set_sampling(int32_t slot, const request_options & o);
    // VIS1 Stage B: images as engine::prefill's; their spans (not their pixels) go to the follower after the tokens
    run_status prefill(int32_t                           slot,
                       const int32_t *                   tokens,
                       int32_t                           n_tokens,
                       int32_t                           n_reuse,
                       int32_t &                         n_processed,
                       const std::vector<prompt_image> * images = nullptr);
    int32_t    read_begin(int32_t                           slot,
                          const int32_t *                   tokens,
                          int32_t                           n_tokens,
                          int32_t                           n_reuse,
                          int32_t &                         n_processed,
                          const std::vector<prompt_image> * images = nullptr);
    run_status truncate(int32_t slot, int32_t n_keep);
    run_status release_slot(int32_t slot);
    run_status set_pos_offset(int32_t slot, int32_t off);  // MUP1 piece 5

  private:
    void end_group(int32_t g);
    // the slot's oldest queued step into res (false: none); a delivered end or failure ends the slot's stepping
    bool take(int32_t slot, joint_step_out & res);
    void delivered(int32_t slot, const joint_step_out & res);
    bool ended(int32_t slot) const;

    leader *                                        l_;
    pipe::core                                      core_;
    std::unique_ptr<leader::pending>                fl_[2];
    std::vector<int32_t>                            members_[2];
    std::vector<std::pair<int32_t, joint_step_out>> kept_;  // finished steps not delivered, in the order they ended
    bool                                            ended_[pipe::max_writers] = {};
    pace::pacer                                     pace_{ pace::role::front };
    // the groups the front has (bit g: members or in flight), and each one's pacing generation, for set_pace
    uint32_t                                        live_mask() const;
};

// ---- the follower (the back engine's executor) ----

class follower {
  public:
    bool   init(engine & e, port & p, crossing & x, std::string & err);
    // the commands, in arrival order, until X-BYE (true) or the leader is lost (false, why set)
    bool   serve(std::string & why);
    double idle_poll_s = 1.0;
    // the drive's tests: op::probe answered by probe_fn (slot, what, the bytes); op::reconfigure by reconfigure_fn,
    // which rebuilds the engine and returns it (nullptr and err on a failure); the follower then serves it
    std::function<bool(int32_t slot, int32_t what, std::vector<uint8_t> & out, std::string & err)> probe_fn;
    std::function<engine *(const ctrl & c, std::string & err)>                                     reconfigure_fn;
    // split-test's injection (TTTS8, the truncate case): the next truncate of this slot fails on the follower as
    // engine::truncate's failed write inside the last step leaves it (the slot's checkpoints dropped, its state
    // restored at the next read); -1: off
    std::atomic<int32_t>                                                                           test_fail_cut{ -1 };
    // PRD1 items 8, 9 and 11 (TEST): the most idle-drain units in flight when a command arrived and the drains joined
    // at one; with test_order, the order (test_log): 'L' a read step that ran, 'C' a step group's arrival, 'Q' its
    // back half queued, 'P' and the slot's digit a read's end on this box; test_hold_reads: no read work runs here
    // (commands still do), so a front can finish reads before the back takes them
    std::atomic<bool>                                                                              test_order{ false };
    std::atomic<bool>    test_hold_reads{ false };
    std::atomic<int32_t> test_in_flight_max{ 0 };
    std::atomic<int64_t> test_settles{ 0 };
    std::string          test_log() const;

  private:
    mutable std::mutex test_mu_;
    std::string        test_log_;
    void               test_note(char c);
    struct pre {
        bool     valid = false;
        int32_t  kind = 0, d = 0;
        int32_t  prop[4]    = {};
        double   to_draft_s = -1;  // the draft's completion from its own start (the next step's draft phase)
        uint64_t rounds     = 0;   // the target rounds the early draft ran (counted when it is discarded)
    };

    // an early draft gone unused (the slot's next command is not its step): its rounds counted on the engine
    void discard(int32_t slot);
    void rebind(engine & e);

    // the back chunks of reads in arrival order (op::read_step): a lone chunk whole when it comes up, an interleaved
    // one a layer group at a time, after each step group and while no command waits; each answered at its end
    // MUP2: a budgeted chunk (a = 2) runs units after each step group under this box's writer-pause deadline (APF1:
    // pace_, from the step groups' arrivals and back stages here, P carried by the commands), and through an idle gap
    // while no command waits (no writer step pending: the floor). PRD1 item 7: a small read is a lone chunk (a = 0)
    struct iread {
        int32_t  slot = -1;
        uint32_t seq  = 0;
        bool     lone = false, cancel = false, cancel_sent = false;
        int64_t  want = 0;  // cancel: the chunks the front finished (their crossings sent), run before the cancel
        bool     budgeted = false;
    };

    std::deque<iread> reads_;
    // idle: no command waits (the floor); else after a step group (the budget)
    void                                  read_group(bool idle = false);
    void                                  read_answer(const iread & r, ack & a);
    // PRD1 item 8: a command arrived: the idle drain's in-flight units joined first, as read time (false: the read
    // failed, answered)
    bool                                  read_settle();
    // PRD1 item 11: the crossing taken ahead's pieces posted when they may be (between units, never waiting)
    void                                  poll_ahead();
    // APF1: the back's writer-pause book (fn/runtime/pace.h)
    pace::pacer                           pace_{ pace::role::back };
    void              step_group(const ctrl & c, uint32_t seq);
    void next_of(const std::vector<int32_t> & slots, std::vector<pre> & out, std::vector<std::string> & why);
    bool reply(uint32_t seq, const ack & a);

    engine *                                         e_ = nullptr;
    port *                                           p_ = nullptr;
    crossing *                                       x_ = nullptr;
    std::vector<pre>                                 pre_;
    std::vector<std::shared_ptr<step_out>>           outs_;       // X-STEP records in flight (their sends' keeps)
    std::vector<std::shared_ptr<std::vector<float>>> lgs_;        // forced members' logits in flight (kept too)
    std::vector<uint8_t>                             probe_out_;  // a probe's answer as filled (sent from a copy)
    void                                             forced_group(const ctrl & c, uint32_t seq);
    std::vector<int32_t>                             out_ops_;
    joint_refusals                                   joint_draft_, joint_catchup_;  // TTTS3
};

}  // namespace fn::runtime::stage
