#pragma once

// TTTS1: the split's messages and its transport port (fn/runtime/stage.h has the protocol). Host code only: no CUDA,
// no engine, so the host safety tests (TTTSBS) build it alone.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace fn::runtime::stage {

// ---- the transport ----

class port {
  public:
    // TTTSBS: an operation's buffer belongs to the transport from its post until its end is observed (a completion,
    // or a cancel's acknowledged end); keep, the buffer's owner, is held by the transport until then, whatever the
    // caller does
    using keep_ref = std::shared_ptr<const void>;

    virtual ~port()                                                                                = default;
    virtual int32_t send(uint64_t tag, const void * p, size_t n, keep_ref keep, std::string & err) = 0;
    virtual int32_t recv(uint64_t tag, void * p, size_t n, keep_ref keep, std::string & err)       = 0;

    int32_t send(uint64_t tag, const void * p, size_t n, std::string & err) { return send(tag, p, n, nullptr, err); }

    int32_t recv(uint64_t tag, void * p, size_t n, std::string & err) { return recv(tag, p, n, nullptr, err); }

    // true when op completed with its bytes; false: it failed or the wait gave up. Either way op is finished and its
    // buffer released by the transport (a give-up cancels it and observes its end first), unless the message says it
    // is held: then the transport keeps it, and its keep, until it ends
    virtual bool    wait(int32_t op, std::string & err, double timeout_s) = 0;
    // without waiting: done set when op ended (then as wait: true for its bytes); op is finished either way
    virtual bool    test(int32_t op, bool & done, std::string & err)      = 0;
    // end every op of ops[0 .. n) not finished (-1 skipped): cancelled, its end observed, finished (its result
    // dropped). Returns how many stay held (their keeps with them); 0: every buffer released
    virtual int32_t retire(const int32_t * ops, int32_t n)                = 0;
    virtual bool    map(void * p, size_t n, std::string & err)            = 0;
    virtual bool    lost() const                                          = 0;

    // the leader gives up a lost stage: every operation's end follows promptly (the link: quiesced, the transport
    // stops touching memory; the in-process pair: closed, the follower sees its peer lost)
    virtual void close() {}
};

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
    test_prompt  = 15,  // MUP2: a 1 keeps the processed tokens and appends (no reset)
    pos_offset   = 16,  // MUP1 piece 5: the slot's rotary position offset (a), between its requests
    // APF2 (the drive's tests): the back's forced prompt routes (model::prompt_route_force: a dense, b hc, c expert,
    // base graph), answered with the follower process's executed-variant totals (model::prompt_route_totals,
    // collected from here on) in acct_i[0 .. route_count_fields - 1]
    test_route   = 17,
};

// APF2: test_route's answer: the totals' counts in model::prompt_path::route_counts' order (dense_ladder .. ple)
constexpr int32_t route_count_fields = 16;

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
    // per op (prefill / read_begin: n_reuse; read_step: a the back's geometry (0 whole, 1 interleaved, 2 budgeted by
    // the writer-pause deadline, APF1), b a cancel, c the chunks the front finished)
    int32_t  a = 0, b = 0, c = 0, n_tokens = 0;
    int64_t  base = 0;                           // a read's first crossing number
    // set_sampling: the request's options
    uint64_t seed = 0;
    float    temp = 1.0f, top_p = 0.95f, min_p = 0.05f;
    int32_t  top_k = 20, max_tokens = -1, n_ctx = 0;
    // step_group
    int32_t  group = 0, n_members = 0;
    member   m[max_members];
    // APF1: the writer pause P (read_step and step_group; the back's next read-step decision takes it). step_group
    // also carries the front's service for this step (ms; the back's projection of its next input), the groups the
    // front has (bit g) and each group's generation (a re-admitted group is new on the back too): durations and
    // identities, never timestamps. PRD1 item 7: a small read is a = 0 (whole), with or without a budget
    float    pause_ms   = 0;
    float    front_ms   = -1;
    uint32_t live       = 0;
    int32_t  gen[2]     = { 0, 0 };
    // APF2: prefill and read_begin carry the front's prompt-graph switch (model::prompt_graphs: 1 on, 0 off; -1 none),
    // so the back's reads follow the server's props switch
    int32_t  graphs     = -1;
    int32_t  ignore_eos = 0;   // VBENCH: set_sampling's (1: the back ends the request only at its budget)
    int32_t  pad[5]     = {};  // the record's size kept: 5a674fe1's pad[7] less graphs and ignore_eos
};

static_assert(sizeof(ctrl) % 8 == 0, "X-CTRL's record");

struct ack {
    int32_t status = 0;  // run_status, or the read's status
    int32_t a = 0, b = 0, c = 0;
    int64_t x = 0, y = 0;
    char    why[232] = {};
    // MUP2 piece 4: a back chunk's answer carries the read's units account on the back so far (the follower's
    // read_info): unit GPU ms, floor ms, the B mean, P, its read job's service ms, then APF1's pacing (the longest
    // writer completion gap, the overruns' excess over P, the mean known budget); units, steps, most units a step,
    // floor units, the smallest and largest piece's rows, PRD1 item 12's budget misses and pieces at the floor above
    // B (1: yes), then APF1's budgets zero and unknown, and the pacing's decisions (steps, known, zero, unknown),
    // completions, gaps, overruns and their reasons (no read here, budgeted, minimum unit, small read, idle drain)
    double  acct[8]    = {};
    int32_t acct_i[24] = {};
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
    float    back_ms      = -1;  // APF1: the group's back stage on the follower, its input in to its answers (ms)
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
    // MUP2 piece 1: the chunk's partition (model::prompt_chunk n_pieces and piece): the back runs the front's pieces
    // (the crossing's FFN messages are the front's pieces')
    int32_t n_cuts   = 0;
    int32_t cut[128] = {};  // model::prompt_max_pieces (stage.cpp checks)
    int32_t pad2     = 0;
};

static_assert(sizeof(crossing_head) % 8 == 0, "X-BOUNDARY's head record");

constexpr int64_t piece_bytes = int64_t(42) << 20;  // a crossing's piece (about 1,024 rows)

}  // namespace fn::runtime::stage
