#pragma once

// The round's device frame on one card (TG1, SR1; FC-RD1, FC-RD2, FC-ST2): the round descriptor and its copy from
// mapped host memory, the start of both cards together (RT-9), the commit of the versioned state by selection and the
// round's completion record for the host. No model math: copies, a poll, timer reads, one word of commit.
//
// Per card r, a round's graph runs, on one stream:
//   begin   polls the start flag in mapped host memory until it reaches the round's number (the host sets it after
//           launching both cards' graphs, so both cards start together), then copies descriptor slot round % ring from
//           mapped host memory into the device descriptor, which every later node reads. The round's number is the
//           card's own counter (state::next_round), so one captured graph serves every round.
//   stamp   writes %globaltimer into state::t_start (the start skew between the cards, and the pass time).
//   ...     the pass (fn/model/pass.h), whose kernels read N_before, positions, tokens and the PLE use from the device
//           descriptor and the committed slot from state::cur (version.h); the decision writes desc::argmax.
//   commit  (SR1) every versioned object makes logical version k current, k = N_after - N_before from the descriptor's
//           kind (FC-RD3: forced B, generate 1, bootstrap 0; the test-only forced_k its test_k), only when every listed
//           error word is clear (FC-RD2): state::cur <- version_slot(cur, k) and, ME11, state::gdn <-
//           gdn_after_round(gdn, k) (version.h), from the same thread; k = 0 leaves both. A selector outside the
//           contract sets err_gdn_sel and nothing is committed. Nothing is copied, so the
//           pre-step contents (slot cur before the commit) and the round's other versions survive until the next
//           round writes over them, which is after the step's acknowledgment (FC-ST2, RS1 section 3).
//   finish  sets desc::out and n_out by kind (generate and bootstrap: out[0] = argmax[0]; verify: the decision's
//           out[0 .. a] and n_out = a + 1, unchanged, SP1), then writes the completion
//           record into mapped host memory: outputs, argmax, error words, stamps and cur, then (after a system fence)
//           its round number, which the host polls (FC-RD2's completion). ME11: the record carries the selector beside
//           cur.
//
// Who writes cur: this card's commit node, in stream order after every reader of the round, and the host between
// rounds (a truncate within the last step, or the restore after a recoverable error on either card), when no round is
// in flight (FC-G8). Both cards apply the same k from the same descriptor, so they agree.
//
// Contract:
//   shapes:   rows B in 1..5 (SP2); the descriptor ring holds ring slots; up to max_err error words. Anything else is
//             rejected (false, nothing enqueued).
//   protocol: the host writes slot round % ring, then (release) the start flag; begin's threads each acquire the flag
//             (ld.acquire.sys) before reading their words of the slot, so a slot word is never read before the slot
//             was written. The host refills slot s only after the completion record of round s - ring (or a later
//             one) is seen (FC-G8: a round's inputs do not change while it is in flight). A flag not reached within
//             start_timeout_ns sets err_poll_start; a slot whose round differs from the counter sets err_round. Both
//             mean the round's commit does not take effect (commit reads the error word).
//   numerics: exact copies; cur in 0..5; gdn a word gdn_valid accepts.
//   budget:   begin, commit and finish: one warp; stamp: one thread.
//   floor:    commit reads the listed error words and the descriptor and writes one word: launch-bound (under the 1.95 us
//             ramp HX12 measured for a launch [E]). TG1's copy moved about 59 MB a card (floor about 116 us [A]).
//   evidence: TG1 (model-bench), SR1 (state-test, rollback-test, model-bench).

#include <cstddef>
#include <cstdint>

namespace fn::round {

constexpr int32_t max_rows = 5; // SP2: [p, d1..d4]
constexpr int32_t ring = 4;      // descriptor slots in mapped host memory
constexpr int32_t max_err = 8;   // error words commit checks
constexpr uint64_t start_timeout_ns = 2000000000ull;

// FC-RD3's round kinds. forced_k is TEST ONLY (SR1): a forced round of B rows whose commit takes the host-written
// desc::test_k (0..B) instead of B, standing in for a verify round's a + 1 until SP1 absorbs it into verify; only
// fnfl-drive state-test and rollback-test use it, never generate. The draft kinds (DR1) touch no versioned state, so
// their commit count is 0 and they emit nothing: draft_step runs rows = d draft steps of one position (step j's input
// token is tok[0] for j = 0, else argmax[j - 1]; its position pos[p * d + j]; its proposal argmax[j]); draft_catchup
// writes the draft's K and V of rows = B_c committed positions (tok[t], pos[p * B_c + t]) and proposes nothing.
enum class kind : int32_t {
    forced = 0,
    generate = 1,
    verify = 2,
    bootstrap = 3,
    forced_k = 4,
    draft_step = 5,
    draft_catchup = 6,
};

// FC-RD1's descriptor, identical on both cards. The host fills the input fields; the device writes argmax, out and
// n_out. pos is the rope planes t, h, w, e of each row as attend reads them: pos[p * B + t] (all planes N_before + t for
// text).
struct desc {
    uint64_t round;   // strictly increasing per slot; begin checks it against the card's counter
    uint64_t ple_use; // rounds with layer work before this one: X-PLE-STATS's use counter (its slot u % 3)
    int32_t kind;     // fn::round::kind
    int32_t rows;     // B
    int32_t n_before; // N_before: cells before the rows (the old host depth)
    int32_t budget;   // rows the request still allows (FC-SAMP0)
    int32_t a;        // verify only: accepted proposals, written by the decision (FC-DEC3), read by commit
    int32_t n_out;    // emitted tokens: 1 in generate and bootstrap, 0 in forced (written by finish); verify a + 1
                      // (written by the decision)
    int32_t pos[4 * max_rows];
    int32_t tok[max_rows];    // the input tokens
    int32_t out[max_rows];    // the emitted tokens (written by finish)
    int32_t argmax[max_rows]; // greedy's token of each decided row (written by the decision; in a forced round an
                              // observation, not emitted)
    int32_t test_k;           // TEST ONLY (SR1): forced_k's commit count, 0..B
    int32_t test_fail;        // TEST ONLY (SR1): bit r set: card r's commit sees err_test_commit (FC-RD2's
                              // recoverable error on one card or both); 0 in every other use
    int32_t hsrc;             // draft kinds (DR1): the target batch row of the round's first row; row i takes the
                              // hidden carry's row hsrc + i - 1, or the retained carry when that is -1 (draft.h)
    int32_t pad[4];
};
static_assert(sizeof(desc) % 16 == 0, "descriptor slots stay 16-byte aligned");
constexpr int32_t desc_words = static_cast<int32_t>(sizeof(desc) / 4);

// the card's device-side round state
struct state {
    uint64_t next_round; // the round begin expects next
    uint64_t t_begin;    // %globaltimer when begin started polling
    uint64_t t_start;    // %globaltimer after the start (stamp)
    uint64_t t_end;      // %globaltimer in finish
    uint64_t t_logits;   // SM1: %globaltimer after the logits (stamp_logits; the decision's start in context)
    uint32_t cur;        // SR1: the committed version's physical slot, 0..5 (version.h)
    uint32_t gdn;        // ME11: the GDN state's selector (version.h gdn_word), in a former pad word
    uint32_t pad[2];
};
static_assert(sizeof(state) == 56, "ME11: the selector takes a pad word; the round state keeps its size");

// the completion record in mapped host memory, one a card; round is written last
struct done {
    uint64_t round;
    uint64_t t_begin, t_start, t_end;
    uint64_t t_logits; // SM1: after the logits (0 when the round's graph has no logits stamp)
    int32_t n_out;
    int32_t out[max_rows];
    int32_t argmax[max_rows];
    uint32_t err[max_err]; // the listed error words at finish (0: the card's exchange word)
    uint32_t cur;          // SR1: state::cur after the commit
    uint32_t gdn;          // ME11: state::gdn after the commit
};

struct begin_args {
    const desc * host_ring; // [ring] mapped host memory (device pointer)
    const uint64_t * flag;  // mapped host memory: the start flag
    desc * dev;             // the device descriptor
    state * st;             // the card's round state
    uint32_t * err;         // the card's error word (fn::xchg::rings::err)
};

struct commit_args {
    const desc * dev;
    state * st;                    // its cur is the one word commit writes
    const uint32_t * const * errs; // [n_err] device pointers to error words, device memory (errs[0]: the card's
                                   // exchange word, where the test hook sets err_test_commit)
    int32_t n_err;
    int32_t card; // the card this commit runs on (desc::test_fail's bit)
};

struct finish_args {
    desc * dev;
    state * st;
    const uint32_t * const * errs; // [n_err], as commit's (errs[0]: the card's exchange word)
    int32_t n_err;
    done * host; // mapped host memory (device pointer)
};

bool begin(const begin_args & a, void * stream);
bool stamp(state * st, void * stream);
// SM1: %globaltimer into state::t_logits (after the logits, before the decision); one thread
bool stamp_logits(state * st, void * stream);
bool commit(const commit_args & a, void * stream);
bool finish(const finish_args & a, void * stream);

// cur <- version_slot(cur, k) on the device, in stream order: the commit of k (0..5) outside the round frame, for
// layer-test's runner and the kernel tests (gd-test, ple-test). One thread.
bool advance(uint32_t * cur, int32_t k, void * stream);
// ME11: the same with the GDN selector: for k >= 1, gdn <- gdn_after_round(gdn, k) (a target step, prompt false) or
// gdn_after_prompt(gdn) (a prompt chunk of k halves, prompt true); a selector outside the contract is left unchanged
// (the caller's read-back finds it). The prompt path's publication (target::prompt_commit) and layer-test's runner.
bool advance(uint32_t * cur, uint32_t * gdn, int32_t k, bool prompt, void * stream);

// k = N_after - N_before of a round kind with rows B (FC-RD3); verify takes a + 1, forced_k test_k (host mirror of the
// device rule)
int32_t commit_versions(int32_t kind, int32_t rows, int32_t a, int32_t test_k);

} // namespace fn::round
