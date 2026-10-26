#pragma once

// TTTS1: the split's two stages. Flash-Next across Carlos and Carlin: tensor parallel across each box's two cards,
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

#include "fn/runtime/pipe.h"
#include "fn/runtime/prefill.h"
#include "fn/runtime/runtime.h"
#include "fn/xchg/link.h"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fn::runtime::stage {

// ---- the transport ----

class port {
  public:
    virtual ~port()                                                                 = default;
    virtual int32_t send(uint64_t tag, const void * p, size_t n, std::string & err) = 0;
    virtual int32_t recv(uint64_t tag, void * p, size_t n, std::string & err)       = 0;
    virtual bool    wait(int32_t op, std::string & err, double timeout_s)           = 0;
    // without waiting: done set when op completed (then as wait: true for its bytes); op is finished either way
    virtual bool    test(int32_t op, bool & done, std::string & err)                = 0;
    virtual bool    map(void * p, size_t n, std::string & err)                      = 0;
    virtual bool    lost() const                                                    = 0;

    // the leader's side gives up on a lost stage (the in-process pair: the follower sees its peer lost; the link: the
    // process's exit does it)
    virtual void close() {}
};

// the box-to-box link (the caller keeps it alive)
std::unique_ptr<port> link_port(fn::xchg::link::link & l);

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
                           const std::vector<int32_t> & eog);
std::string hello_hash(const std::string & text);
// split-test's in-process pair: a send completes into the peer's matching receive by a host copy
void        loop_pair(std::unique_ptr<port> & a, std::unique_ptr<port> & b);

// ---- the messages ----

enum class op : int32_t {
    set_sampling = 1,
    prefill      = 2,
    read_begin   = 3,
    read_step    = 4,
    read_cancel  = 5,
    truncate     = 6,
    release_slot = 7,
    pool         = 8,
    step_group   = 9,
    bye          = 10,
    set_draft    = 11,
    drain        = 12,  // a failed read's crossings the follower did not take: received and dropped
    // the drive's tests (rollback-test mu1, joint-kl): a probe of the back's state (a: what; answered with the
    // ack's a bytes, then the bytes), the engine rebuilt (reconfigure: a the KV format q8, b the d, c the joint
    // switches, base the cells a slot), and a slot reset with a ledger of the given tokens (test_prompt)
    probe        = 13,
    reconfigure  = 14,
    test_prompt  = 15,
};

// reconfigure's c: the joint switches as engine_config holds them
constexpr int32_t sw_joint_x16 = 1, sw_router_tc = 2, sw_dense_f16 = 4, sw_eager = 8;

constexpr int32_t max_members = 8;

// a step group's member: its slot's next step as the leader runs it (forced: the drive's tests, a forced round of
// rows ledger rows; the back answers with its logits rows too)
enum class mkind : int32_t { begin = 0, generate = 1, verify = 2, forced = 3 };

struct member {
    int32_t slot    = -1;
    int32_t kind    = 0;  // mkind
    int32_t rows    = 0;  // its front round's rows (0: begin)
    int32_t d       = 0;  // verify: the proposals
    int32_t joint   = 0;  // verify: a lane of the group's joint round
    int32_t prop[4] = { 0, 0, 0, 0 };
};

struct ctrl {
    int32_t  op   = 0;
    int32_t  slot = -1;
    int32_t  a = 0, b = 0, c = 0, n_tokens = 0;  // per op (prefill / read_begin: n_reuse; read_step: groups, mode)
    int64_t  base = 0;                           // a read's first crossing number
    // set_sampling: the request's options
    uint64_t seed = 0;
    float    temp = 1.0f, top_p = 0.95f, min_p = 0.05f;
    int32_t  top_k = 20, max_tokens = -1, n_ctx = 0;
    // step_group
    int32_t  group = 0, n_members = 0;
    member   m[max_members];
    int32_t  pad[12] = {};
};

static_assert(sizeof(ctrl) % 8 == 0, "X-CTRL's record");

struct ack {
    int32_t status = 0;  // run_status, or the read's status
    int32_t a = 0, b = 0, c = 0;
    int64_t x = 0, y = 0;
    char    why[232] = {};
};

static_assert(sizeof(ack) % 8 == 0, "X-ACK's record");

// X-STEP: a member's step and its next step
struct step_out {
    int32_t  slot   = -1;
    int32_t  status = 0;  // run_status of the step (ok: ran or only drafted)
    int32_t  ran    = 0;  // a step ran (a begin may only draft)
    int32_t  kind = 0, d = 0, a = 0, n_out = 0, end = 0, cut = 0, waits = 0, joint = 0;
    int32_t  out[5]       = {};
    int32_t  round_n_out  = 0;   // the decided round's tokens before the end-of-generation cut (verify: a + 1)
    int32_t  round_out[5] = {};
    int32_t  cut_rows     = -1;  // the cut's kept rows (-1: no cut)
    int32_t  pad0         = 0;
    int64_t  n_keep       = 0;
    int64_t  n_before     = -1;  // -1: the step was not entered
    uint64_t hash         = 0;   // the step's decision hash (with engine_config::record_hash)
    double   to_draft_s = -1, to_verify_s = -1, to_catchup_s = -1, to_one_s = -1, wall_s = 0;
    int32_t  next_kind    = 0;   // mkind of the next step: 0 none (the request ended), 1 generate, 2 verify
    int32_t  next_d       = 0;
    int32_t  next_prop[4] = {};
    char     why[192]     = {};
};

static_assert(sizeof(step_out) % 8 == 0, "X-STEP's record");

struct crossing_head {
    int32_t slot = -1;
    int32_t rows = 0, half[2] = { 0, 0 };
    int64_t start = 0, seq = 0, bytes = 0;
    int32_t pieces = 0, pad = 0;
};

constexpr int64_t piece_bytes = int64_t(42) << 20;  // a crossing's piece (about 1,024 rows)

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
    // the leader: every crossing's sends completed (before a read reports done)
    bool drain(std::string & err);

    // the leader: the crossings sent for slot's current read (its chunks' count); abort, when set, is asked before
    // each crossing: true stops the read there (the follower already failed it)
    int64_t sent(int32_t slot) const { return slot >= 0 && slot < max_split_slots ? sent_[slot] : 0; }

    // the follower: the crossings slot's current read has taken
    int64_t taken(int32_t slot) const { return slot >= 0 && slot < max_split_slots ? taken_[slot] - base_[slot] : 0; }

    std::function<bool()> abort;
    // the follower: slot's crossings base + i for i below count that it has not taken, received and dropped
    bool                  drop(int32_t slot, int64_t base, int64_t count, std::string & err);

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
    void *               done_[2][2]             = {};  // the back: each buffer's copy-in completed, on each card
    bool                 used_[2]                = { false, false };
    int64_t              base_[max_split_slots]  = {};
    int64_t              sent_[max_split_slots]  = {};  // the leader: crossings sent for the slot's read
    int64_t              taken_[max_split_slots] = {};  // the follower: the next crossing number the slot's read takes
    int64_t              n_ = 0, bytes_ = 0;
    crossing_head        head_ = {};
};

// ---- the boundary buffers ----

// pinned host memory for every slot's boundary (model::stage_bytes(5) a card), mapped on the port
struct buffers {
    std::vector<std::array<uint8_t *, 2>> slot;
    ~buffers();
    bool alloc(int32_t n_slots, port * p, std::string & err);
};

// ---- the leader (the front engine's driver) ----

struct leader_stats {
    int64_t groups = 0, members = 0, boundaries = 0, boundary_bytes = 0, crossings = 0, crossing_bytes = 0;
    double  back_wait_s = 0;  // the leader's time waiting for X-STEP and acknowledgments
};

class leader {
  public:
    // e: the front engine (engine_config::stage 1, its stage_buf b's); p: the link; x: the front's crossing, set on
    // the engine's prefiller
    bool init(engine & e, port & p, crossing & x, std::string & err);

    run_status set_sampling(int32_t slot, const request_options & o);
    run_status prefill(int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse, int32_t & n_processed);
    int32_t read_begin(int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse, int32_t & n_processed);
    read_status read_step(int32_t slot, int32_t n_groups, read_mode mode);
    void        read_cancel(int32_t slot);
    run_status  truncate(int32_t slot, int32_t n_keep);
    run_status  release_slot(int32_t slot);
    // the draft's d and form (engine::set_draft) on both boxes, between requests
    run_status  set_draft(int32_t d, bool sparse);
    // one step of each of n slots (1 .. 8), as engine::step_joint (the full-d verifies of 2 .. 4 lanes as joint
    // rounds); synchronous: returns with every member's step
    void        step_group(const int32_t * slots, int32_t n, joint_step_out * res);
    // the follower's pool
    bool        follower_pool(pool_info & out);
    // an orderly end: X-BYE, answered
    void        bye();

    bool lost() const { return lost_; }

    const std::string & lost_why() const { return why_; }

    const leader_stats & stats() const { return st_; }

    const std::string & error() const { return err_; }

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
    // both boxes' slot reset, its ledger the tokens (none processed)
    run_status               test_prompt(int32_t slot, const std::vector<int32_t> & tokens);
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

    bool command(ctrl & c, const int32_t * tokens, ack & a, double timeout_s);
    void dispatch(pending & g, int32_t group);
    // a failed read's crossings the follower did not take, dropped there (op::drain)
    void drain_read(int32_t slot, int64_t base);
    bool send_ctrl(ctrl & c, const int32_t * tokens, int32_t & ack_op, ack & a);
    bool wait_ack(int32_t op, ack & a, double timeout_s);
    void lose(const std::string & why);

    engine *                                            e_ = nullptr;
    port *                                              p_ = nullptr;
    crossing *                                          x_ = nullptr;
    std::vector<next_step>                              next_;
    std::vector<int64_t>                                read_base_;  // per slot: its read's first crossing number
    std::vector<std::vector<std::pair<int32_t, ack *>>> read_acks_;
    // per slot: the front's read ended with this status while the back's chunks may still run (-1: none)
    std::vector<int32_t>                                read_end_;
    // the read's end once the back's chunks are answered: blocking, or yielded while one is not
    read_status                                         read_finish(int32_t slot, bool block);
    bool                                                block_reads_ = false;
    std::vector<std::vector<float>>                     logits_;  // per slot: its last forced round's back logits
    uint32_t                                            seq_  = 1;
    int64_t                                             base_ = 0;
    bool                                                lost_ = false;
    std::string                                         why_, err_;
    leader_stats                                        st_;
};

// ---- the leader's pipeline (fn/runtime/pipe.h's core over the leader's group halves) ----

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
    // slot's group, when in flight, ends now; its members' results are kept for their next step call
    void settle(int32_t slot);
    void settle_all();
    // a kept result dropped (the slot's request ended or was cancelled)
    void forget(int32_t slot);
    bool in_flight(int32_t slot) const;

  private:
    void end_group(int32_t g);

    leader *                                        l_;
    pipe::core                                      core_;
    std::unique_ptr<leader::pending>                fl_[2];
    std::vector<int32_t>                            members_[2];
    std::vector<std::pair<int32_t, joint_step_out>> kept_;
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

  private:
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
    struct iread {
        int32_t  slot = -1;
        uint32_t seq  = 0;
        bool     lone = false, cancel = false, cancel_sent = false;
        int64_t  want = 0;  // cancel: the chunks the front finished (their crossings sent), run before the cancel
    };

    std::deque<iread> reads_;
    void              read_group();
    void              step_group(const ctrl & c, uint32_t seq);
    void next_of(const std::vector<int32_t> & slots, std::vector<pre> & out, std::vector<std::string> & why);
    bool reply(uint32_t seq, const ack & a);

    engine *                                         e_ = nullptr;
    port *                                           p_ = nullptr;
    crossing *                                       x_ = nullptr;
    std::vector<pre>                                 pre_;
    std::vector<std::unique_ptr<step_out>>           outs_;       // X-STEP records in flight (their sends' buffers)
    std::vector<std::unique_ptr<std::vector<float>>> lgs_;        // forced members' logits in flight
    std::vector<uint8_t>                             probe_out_;  // a probe's answer bytes in flight
    void                                             forced_group(const ctrl & c, uint32_t seq);
    std::vector<int32_t>                             out_ops_;
};

}  // namespace fn::runtime::stage
