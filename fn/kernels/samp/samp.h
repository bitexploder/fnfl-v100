#pragma once

// SM1: the exact sampler over the two-card vocabulary, the speculative accept step and the decision hash (FC-DEC2,
// FC-DEC3, FC-RD2's delayed guard; SM0 v2, fnfl/design/sampler-design.md). Both cards decide; no card broadcasts
// (SM0 section 6): every value entering a decision is exchanged bytes (a card uses the bits it sent, from its own copy),
// the deciding kernels are the same machine code on both cards, and every exp and log runs on the device.
//
// The pinned chain (common/common.h 261-271 and common/sampling.cpp common_sampler_init at 66e11c3; src/llama-sampler.cpp
// for each step), in its order: logit bias, penalties, top-k, top-p, min-p, temperature, the draw. top-k, top-p and
// min-p decide the kept set on untempered logits; temperature then shapes the probabilities inside it. DRY, top-n-sigma,
// typical-p, XTC, mirostat and grammar are refused (options::check; SM0 section 9's proposal).
//
// The decision paths (path_of):
//   greedy  T <= 0 with no bias and no penalties: V1's greedy (fn/kernels/head) with T-GREEDY, which carries the
//           decision hash; greedy_verdict writes the round's tokens (verify: a draft is accepted iff it equals the
//           previous row's argmax, NC1 section 7) and the decision record. 1 exchange.
//   topk    0 < top-k <= 256, or T <= 0 with a bias or penalties (top-k 1, greedy mode). select: each card's top k by
//           (logit, lower ID), then exchange A carries each card's sorted list; tk_decide merges them, applies top-p
//           and min-p to the merged list exactly as the pinned code does on a sorted list, tempers, normalizes and
//           draws; verify rows take the accept step on the lists (q is a lookup in the draft row's kept list). 1 exchange.
//   full    top-k 0: no filter, full-vocabulary min-p, full-vocabulary top-p, or both. Stages (each a launch; SM0
//           section 4):
//             max      (min-p, top-p) each card's maximum M_r; min-p alone sends it (A)
//             hist     (top-p) the absolute-grid histogram (w = 1/64, 2,048 bins from the card's top bin and a tail
//                      bin), counts exact, sums integer (below); sends A
//             plan     (top-p) both cards: the global maximum, the cut bin, min-p's threshold; members needed unless
//                      the cut bin has one member (a cut on a bin edge) or min-p's threshold lies above it
//             members  (top-p) each card's members of the cut bin (logit, ID); sends B
//             resolve  (top-p) both cards: the exact cut among the members, by a radix select over the composite key
//                      weighted by integer mass
//             settle   the kept set's mass Z_r and a local candidate (ascending-ID scan) on each card; verify rows add
//                      the owner's numerator and q at the draft; sends C
//             decide   both cards: the shard by its mass with one uniform, the token from that card's candidate;
//                      verify: the accept bits and the first rejection
//             resid    (verify, on a rejection) the residual max(0, p - q) on each card: its integer mass R_r and a
//                      local candidate; sends D
//             decide2  (verify, on a rejection) the shard by R, or p itself when R is 0
//           Exchanges: no filter 1 (2 on a rejection); min-p 2 (3); top-p 3 (4; 2 when the cut is on a bin edge).
//
// The tail bin: with 2,048 bins the window spans 32 logit units below the card's top bin, so the mass below it is at
// most 248,320 e^-32 = 3.1e-9 of Z [counted] and the cut of any F32 top_p < 1 (at most 1 - 6e-8) falls inside the
// window. The tail bin is still a bin: a cut there takes its members like any other (test hook window_bins).
//
// Numerics (NC1 section 7, exact in distribution; NC1 section 2, deterministic and batch-invariant):
//   transform  F32, each operation rounded (no contraction): l + bias[id]; then for count c > 0 (the window's count
//              plus the row's deltas): l <= 0 ? l * repeat : l / repeat, then l - (c * freq + present), as
//              llama_sampler_penalties_apply. A NaN logit orders below every number and has no mass.
//   order      the composite key (order_key(l) << 32 | ~id): larger logit first, then the lower ID; unique per token.
//   masses     weights w = expf(lt - Mt) in F32 (lt = l / T, Mt = M / T, both rounded; T = 1 for top-p's cut), then
//              integers: round(w * 2^s), s = 50 inside a chunk (settle), the top-k list and resolve's members, 46 for
//              a card's histogram, 60 for the residual's probabilities. Integer sums are exact and order-free (integer
//              atomics allowed, NC1 section 2); across chunks and cards the scales are combined in double, in a fixed
//              order (chunk order; card 0 first). Weights below 2^-(s+1) round to 0: at most 1.7e-9 of the mass
//              [counted: 124,160 x 2^-47 / 1].
//   draws      a 64-bit uniform u: an integer target floor(u * S / 2^64) for integer masses S, the first entry whose
//              running sum exceeds it; for double masses (the shard choice) u53 * Z < Z_0. The accept test is
//              u53 * q(x) < p(x) in double (common_sampler_rejection_step).
//   min-p      threshold M + logf(min_p) in F32, as the pinned code; kept: l >= threshold.
//   top-p      the first entry in (logit, lower ID) order whose running T = 1 mass reaches top_p * Z (pinned: cum_sum
//              >= p), in integers at the global scale.
//
// Random numbers (philox.h): (seed, request, position, purpose); position = the decided token's position: N_before
// + t + 1 for row t of a generate or verify round, N_before for the bootstrap (FC-RD3).
//
// The decision record and hash (FC-RD2's delayed guard): each card's decision writes its record (the round number, the
// emitted tokens, their hash). The next decision's first exchange carries both cards' records (four words: the round
// and the hash, 31 bits each); its first reader compares them and sets err_hash in the card's error word on a mismatch,
// so that round commits nothing (FC-RD2), and marks the record checked. close_send and close_recv compare the last
// record before a step that ends generation returns, and before a truncate, re-prefill or release (the closing check).
// A record whose check has not run reads unchecked (read_record).
//
// Penalty state (FC-ST1; SM0 section 3): per card, U32 counts for the whole vocabulary (both cards hold the same table,
// so the re-sharded draft head reads it too) and the window's ring: ring[pos % cap] holds the ledger token at pos for
// the window's positions [L - min(L, cap), L), L the tokens accepted so far. The decision record's tokens join the
// tables at the next decision's first node (apply), so a round whose commit does not take effect (FC-RD2) never
// changes them: the host discards the record (discard_record). A truncate first applies the record, then rewinds
// (rewind); the result equals load_window of the kept ledger (the clean replay). A verify row t sees drafts
// x_1 .. x_t as accepted (per-row deltas, at most max_drafts added and max_drafts evicted, never written).
//
// The retained q (FC-DR1; the engineer's decision, SM1's layout): per slot and card, for each draft row t (0 .. 3):
//   q_list[t]   the draft's kept list on the top-k path: n <= 256 (ID, probability F32), written by the draft's pick
//               (pick, tk_decide in draft mode), identical on both cards;
//   q_shard[t]  the draft's probabilities over the card's target vocabulary shard, F32 [124,160], on the full paths
//               (0 outside the draft subset).
//   Reuse chain (xchg.h's form): writer: the draft step that drew x_{t+1} (pick); readers: the next verify round's
//   tk_decide (q_list) or settle and resid (q_shard); the next write is the next round's draft pick, which runs after
//   (stream order on the card) that verify's last reader (decide2). q_shard's writer on the re-sharded draft head is
//   not built (Later: the draft's ID map interleaves the two cards, so the full-path pick would scatter q across them).
//
// Batch invariance: a row's blocks and arithmetic do not depend on B (32 chunks a row, the same chunk bounds); rows are
// independent except the verdict, which reads each row's results.
//
// Exchanges: fn/xchg's protocol; the sampler's messages, readers, resetters and reuse chain are in xchg.h (SM1). A
// decision uses slot (round & 1) of each message, so back-to-back decisions need no other exchange between them.
// Forced rounds decide nothing: every decision stage returns at once (both cards alike); apply still runs.
//
// R1's parts (state.cpp, "sampler (SM1)"): the params and record, the bias table and the counts (248,320 each a card),
// the ring (the slot's cells), the q lists and shards, the workspace (workspace_bytes); the exchange rings grow by the
// sampler's slots (fn::xchg::rings_card_bytes).
//
// Contract of every stage launcher:
//   shapes:  rows B in 1..5 (verify: B = d + 1, d in 1..4; SP2); n logits a row, n % 4 == 0, 128 <= n <= 124,160;
//            id0 + n <= 248,320, or an ID map of n strictly increasing IDs below 248,320.
//   budget:  grid stages: 32 x B blocks of 256 threads, __launch_bounds__(256), at most 26 KB of shared memory a block;
//            single-block stages 256 threads (ptxas reports in work/SM1/checks/).
//   floor:   a row's logits read once (stats) and once more by the block that rescans the chosen chunk: 0.5 MB a card a
//            row, 1.5 MB at B = 3, about 1.5 us at 1,021.6 GB/s [M, HX2] plus launches (1.95 us ramp each [M, HX12])
//            and the exchanges (HX1, about 2-6 us each [M]); SM0 section 8's forecast: top-k path 8-12 us a round at
//            B = 3, threshold paths 15-30 us [E].
//   numerics: above; the reference is samp_ref (FP64), never a borrowed kernel (R16).
//   evidence: SM1 (this ticket); simulator node: the sampler (SM0 section 8).
// False, with nothing enqueued, for an argument the contract rejects; false after a failed launch submission.

#include "fn/kernels/round/round.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fn::samp {

constexpr int32_t max_rows = 5;   // SP2: [p, d1..d4]
constexpr int32_t max_drafts = 4; // SP2
constexpr int32_t max_k = 256;
constexpr int32_t n_vocab = 248320;
constexpr int32_t n_card = 124160;  // a card's target shard
constexpr int32_t n_chunk = 32;     // chunks a row, whatever B
constexpr int32_t max_chunk = 3904; // ceil(124,160 / 32) rounded up to 64
constexpr int32_t hist_bins = 2048; // the window (w = 1/64): 32 logit units
constexpr int32_t n_thread = 256;

enum class path : int32_t { greedy = 0, topk = 1, full = 2 };

// error bits (fn::xchg's card error word, kernel data bits 16-31)
constexpr uint32_t err_hash = 1u << 21;  // the two cards' decision records differ (FC-RD2: the slot is invalidated)
constexpr uint32_t err_empty = 1u << 22; // a row with no candidate (every logit NaN or -inf): a recoverable error

// The request's sampling options (FC-SAMP0; RS1 section 5). Defaults: the pinned server's (T 0.8, top-p 0.95, min-p
// 0.05, penalty_last_n 64, repeat 1, freq 0, present 0), with top-k 20, the model's own default (TK1).
struct options {
    float temp = 0.8f;
    int32_t top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.05f;
    int32_t penalty_last_n = 64; // -1: the context (n_ctx); 0: off
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;
    uint64_t request = 0;
    int32_t eos = -1; // SP1 (FC-DEC3): the end-of-generation token; a verify round's emitted list ends at it (-1: none)
    std::vector<std::pair<int32_t, float>> bias; // logit bias (token, value), the vocabulary's suppressed tokens (-inf)
                                                 // appended by the caller
    // refused until the operator decides (SM0 section 9): nonzero means the request asked for them
    bool dry = false, top_n_sigma = false, typical = false, xtc = false, mirostat = false, grammar = false;

    // the options FNFL refuses, with a message naming each; n_ctx resolves penalty_last_n = -1
    bool check(int32_t n_ctx, std::string & err) const;
    bool penalties() const;
    bool transform() const { return penalties() || !bias.empty(); }
};

path path_of(const options & o);

// the device form of the options and the path's flags
struct params {
    float temp;
    float top_p;
    float min_p;
    int32_t top_k; // 1 .. 256 on the top-k path (1 in greedy mode)
    float rep;
    float freq;
    float present;
    int32_t cap; // the window (0: penalties off)
    uint32_t flags;
    int32_t window_bins; // hist_bins; TEST ONLY: fewer, to reach the tail bin
    uint32_t seed_lo, seed_hi, req_lo, req_hi;
    int32_t eos; // FC-DEC3: the end-of-generation token a verify verdict cuts at (-1: none)
    int32_t pad[1];
};
static_assert(sizeof(params) % 16 == 0, "params stay 16-byte aligned");
constexpr uint32_t f_bias = 1u << 0;
constexpr uint32_t f_pen = 1u << 1;
constexpr uint32_t f_greedy = 1u << 2; // T <= 0: the argmax of the kept set (top-k 1)
constexpr uint32_t f_topp = 1u << 3;   // full path: top_p < 1
constexpr uint32_t f_minp = 1u << 4;   // min_p > 0

params make_params(const options & o, int32_t n_ctx);

// the decision record (FC-RD2's hash) and the window's position, per card
struct record {
    uint32_t words[4]; // as exchanged: round and hash, 31 bits each; 0x7fffffff x 4 for an invalid record
    uint64_t round;
    uint64_t hash;  // 62 bits: two 31-bit halves (never the exchange sentinel)
    int64_t window; // L: tokens the penalty tables have taken
    int32_t n;      // emitted tokens 0..5 (SP2)
    int32_t tok[max_rows];
    uint32_t valid;   // the decision ran and its round has not failed (the host clears it, discard_record)
    uint32_t checked; // the peer's record was compared with this one (by the next decision or the closing check)
    uint32_t applied; // its tokens joined the penalty tables (apply)
    int32_t a;        // verify: accepted proposals
};
static_assert(sizeof(record) % 16 == 0 && offsetof(record, words) == 0, "the record's words stay 16-byte aligned");

// the draft's kept list for one draft row (top-k path)
struct qlist {
    int32_t n;
    int32_t pad[3];
    int32_t id[max_k];
    float q[max_k];
};

// One card's sampler buffers (non-owning; R1's "sampler (SM1)" class, or a test's). Every pointer 16-byte aligned.
struct card_view {
    int32_t card = 0;
    params * prm = nullptr;      // device copy of the options
    float * bias = nullptr;      // [248,320] F32, read when f_bias
    uint32_t * counts = nullptr; // [248,320] U32 window counts, read when f_pen
    int32_t * ring = nullptr;    // [cap] ledger tokens by position % cap
    record * rec = nullptr;
    qlist * q_list = nullptr;  // [max_drafts]
    float * q_shard = nullptr; // [max_drafts][124,160]
    uint8_t * work = nullptr;  // workspace_bytes()
    uint32_t * err = nullptr;  // the card's exchange error word
    // the exchange slots (fn::xchg::rings, msg::samp_*): this card's receive slots of each and the peer's; a decision
    // uses slot (round & 1), a draft pick slot (position & 1); the closing exchange has one
    float *a_own[2] = {nullptr, nullptr}, *a_peer[2] = {nullptr, nullptr};
    float *b_own[2] = {nullptr, nullptr}, *b_peer[2] = {nullptr, nullptr};
    float *c_own[2] = {nullptr, nullptr}, *c_peer[2] = {nullptr, nullptr};
    float *d_own[2] = {nullptr, nullptr}, *d_peer[2] = {nullptr, nullptr};
    float *x_own = nullptr, *x_peer = nullptr;
};

// the round's inputs a decision reads: the logits rows and the round descriptor (kind, B, N_before, tok, budget; the
// decision writes argmax, and in verify a, out and n_out)
struct decide_io {
    const float * logits = nullptr; // [B][n] row-major, row stride n
    int32_t n = n_card;
    int32_t id0 = 0;               // the global ID of column 0 (card * 124,160), when ids is null
    const int32_t * ids = nullptr; // or n strictly increasing global IDs (the draft head's ID map)
    fn::round::desc * desc = nullptr;
};

size_t workspace_bytes();
// zeroes the workspace's counters and accumulators (once, before the first launch; every launch leaves them zero)
bool prepare(const card_view & v, void * stream);

// ---- the stages ----
enum class stage : int32_t {
    apply = 0,  // the record's tokens join the penalty tables (f_pen)
    select = 1, // top-k path
    tk_decide = 2,
    max = 3, // full path
    hist = 4,
    plan = 5,
    members = 6,
    resolve = 7,
    settle = 8,
    decide = 9,
    resid = 10,
    decide2 = 11,
    greedy_verdict = 12, // greedy path, after V1's greedy_decide
};
// the stages of a decision, in order (each enqueued on both cards before the next: every send precedes its read)
std::vector<stage> stages_of(const options & o, bool verify);
bool enqueue(stage s, int32_t rows, const card_view & v, const decide_io & io, void * stream);

// the greedy path's hash words for T-GREEDY (head.h): the record's four words (record::words), sent by greedy_reduce,
// compared by greedy_decide, which marks the record checked (greedy_checked)
const uint32_t * greedy_hash_words(const card_view & v);
uint32_t * greedy_checked(const card_view & v);

// ---- the closing check (FC-RD2), eager: close_send on both cards, then close_recv on both ----
bool close_send(const card_view & v, void * stream);
bool close_recv(const card_view & v, void * stream);

// ---- the draft's pick (SM0 section 5, last item): top-k path only (B = 1); proposal and q_list[t] out ----
struct pick_io {
    const float * logits = nullptr; // [1][n]
    int32_t n = 0;
    int32_t id0 = 0;
    const int32_t * ids = nullptr;  // the draft head's ID map
    uint32_t position = 0;          // the proposal's position, when live is null
    // SP1: the draft round's device descriptor; when set, the position is live->n_before + live_offset, read at every
    // launch (a captured graph's kernel arguments are frozen at capture: a host value would fix the Philox key)
    const fn::round::desc * live = nullptr;
    int32_t live_offset = 0;
    int32_t t = 0;                  // the draft row (0 .. 3): q_list[t]
    const int32_t * dtok = nullptr; // the proposals so far (penalty deltas), nullable
    int32_t n_dtok = 0;
    int32_t * proposal = nullptr; // out, one I32
};
bool pick(const card_view & v, const pick_io & io, void * stream);
// pick's two stages (0: select, sends samp_a; 1: tk_decide, reads it), for a single stream serving both cards
// (loopback, Compute Sanitizer): stage 0 on both cards before stage 1 on either
bool pick_stage(const card_view & v, const pick_io & io, int32_t stage, void * stream);
// FMU4: one stage of the joint draft step's pick of `lanes` (1 .. max_rows) lanes on one stream, lane k's pick_stage on
// its own view v[k] and io[k], except that its list travels in row k of samp_a's slot (row 0 alone), so lanes whose
// views name the same rings never share a row; each lane's proposal and q_list row are its lone pick's
bool pick_lanes_stage(const card_view * const v[], const pick_io io[], int32_t lanes, int32_t stage, void * stream);

// ---- host-side state operations (synchronous; no round in flight) ----
// the window from the ledger's last min(L, cap) tokens: counts, ring, record.window = L, record cleared
bool load_window(const card_view & v, int32_t cap, const std::vector<int32_t> & ledger, void * stream);
// the record of a round whose commit did not take effect (FC-RD2) is dropped (enqueued; the caller synchronizes)
bool discard_record(const card_view & v, void * stream);
// a truncate of the window to n_keep tokens: apply the record's tokens, then remove the last tokens and restore the
// evicted ones from the ledger (the full ledger before the cut)
bool rewind(const card_view & v, int32_t cap, const std::vector<int32_t> & ledger_before, int64_t n_keep,
            void * stream);
bool read_record(const card_view & v, record & out, void * stream);
bool write_params(const card_view & v, const params & p, const options & o, void * stream);

// TEST ONLY: this card's top-k list of row (the words it sent in samp_a: n, then (logit, ID) pairs)
bool read_list(const card_view & v, int32_t row, std::vector<uint32_t> & words, void * stream);

// the record's hash of a decision (host mirror of the device's)
uint64_t decision_hash(uint64_t round, int32_t n, const int32_t * tok, int32_t a);

} // namespace fn::samp
