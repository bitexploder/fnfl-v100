#pragma once

// The output end of the target pass on one card of the two-card split (V1): the vocabulary logits (FC-HEAD2) and greedy
// selection with T-GREEDY (FC-DEC1). The HC head mixer before them (FC-HEAD1) is a mode of HC2's kernels (hc.h:
// head_down2, head_up2). Simulator nodes vocab and the greedy pick (seed graph V; WP2 section 7). Per card r, on one
// stream: hc::head_down2, hc::head_up2, logits (which acquires the head's X-HMIX, HC3), greedy_reduce, greedy_decide.
//
// Contract (logits, FC-HEAD2): card r's vocabulary shard, IDs r*124160 .. r*124160+124159 (the halves are concatenated,
//   not added).
//   shapes:  rows B in {1, 2, 3, 4, 5} (SP2); n vocabulary rows a card: 124,160 in the model, any n % 8 == 0 for
//            tests; K 2,560.
//   math:    x = the head's mixed row in C2's decode format, which hc::head_up2's epilogue wrote into X-HMIX (HC3;
//            bitwise act_ref of the F32 row), then logits[b][i] = sum_k W[i][k] x_b[k] with dense::product on the
//            Q5_K rows, which acquires the peer's half (dense.h; DP2's numerics and bound, 1e-5 of max |ref| a row
//            against FP64 on fn/quant's decode and act_ref's values).
//   life:    valid from the product's completion until the round's decision completes (FC-HEAD2); this module never
//            rewrites them.
//   floor:   DP2's: 218.5 MB a card, 1.949 us + bytes / 1,021.6 GB/s + 0.31 us = 216.65 us (B1) [M, HX12].
//
// Contract (greedy_reduce, then greedy_decide, FC-DEC1; the draft's greedy pick, FC-DR1):
//   shapes:  rows B in {1, 2, 3, 4, 5}; n logits a row, n % 4 == 0, 4 <= n <= 124,160; id0 + n <= 248,320.
//   math:    each card: the largest logit of each row and its global ID (id0 + column, or ids[column] with an ID map);
//            an exact tie (equal F32 values; -0 equals +0) goes to the lower ID; a NaN orders below every number
//            (below -inf). Then both cards: the larger of the two cards' values, the lower ID on an exact tie (card
//            0's in the vocabulary shard, whose IDs are lower), so both cards hold the same token. A row whose logits
//            are all NaN (on both cards) is out of contract: token -1 and err_all_nan in the card's error word (FC-RD2:
//            a recoverable error of the round).
//   numerics: exact (comparisons on an order-preserving U32 key; no arithmetic). The reduction is a max over a total
//            order, so its order is free: integer atomicMax (NC1 section 2 allows integer atomics) and any split give
//            bitwise the same pair, independent of B and of the run.
//   pair:    (value, ID) as U32 words: the winning value's canonical F32 bits (-0 sent as +0; an all-NaN row's value is
//            the canonical NaN, xchg.h) and the global ID. Neither word can equal xchg's sentinel.
//   exchange: T-GREEDY (xchg.h: its four steps, reader, resetter and reuse chain). reduce's last block sends the pair;
//            decide's lane b polls and resets the peer's pair of row b. SM1: with hash_words, the last block also sends
//            the decision record's four words, and decide's lane 31 polls, resets and compares them (FC-RD2). They are separate launches so that every send
//            is enqueued before any read on either card (loopback and Compute Sanitizer serialize launches).
//   floor:   reduce: B x n x 4 bytes, 1.49 MB at B = 3: 1.949 us + 1.46 us + 0.83 us (a launch that finishes across
//            blocks) = 4.24 us [E, HX12's ramp and rate]; decide: one poll of a peer write, HX1's 1.98 us fixed [M].
//   budget:  reduce: 72 blocks of 448 threads, __launch_bounds__(448, 1); decide: one warp. No shared memory beyond a
//            few words; no spills (ptxas -v, V1 checks/).
//   evidence: V1 (this ticket), WP2 section 7, FC1 (FC-HEAD2, FC-DEC1, FC-X1), NC1 section 3 (the own-error rule).
//
// The draft head (DR1, FC-DR1 "Cards"): the draft's vocabulary subset (65,536 IDs, 98.68% of them in card 0's shard)
//   is re-sharded for balance: card r holds a packed copy of the subset's rows of ranks r, r + 2, ... in ID order
//   (32,768 Q5_K rows, 57,671,680 bytes; draft_ids, pack_draft_head) and their IDs (the ID map, 131,072 bytes). logits
//   runs on the copy (n = 32,768) unchanged; greedy_reduce takes the map (reduce_args::ids), so the pair carries the
//   global ID and greedy_decide's lower-ID rule makes the pick the subset's greedy token on both cards.
//
// Buffers (card r; every pointer 16-byte aligned, buffers not overlapping): the head's X-HMIX slot (fn::xchg::rings);
// logits [B][n] F32; pair_own U32 [B][2]; the T-GREEDY slots from fn::xchg::rings;
// token I32 [B]; work: greedy_work_bytes (reduce) and logits_work_bytes (product), each zeroed once before the first
// launch and left at zero by every launch.

#include "fn/xchg/xchg.h"

#include <cstddef>
#include <cstdint>

namespace fn::head {

constexpr int32_t max_rows = 5; // SP2: the five-row verification pass
constexpr int32_t n_vocab = 248320;      // output.weight rows
constexpr int32_t n_vocab_card = 124160; // a card's shard
constexpr int32_t k_in = 2560;           // the mixed row
constexpr int32_t no_token = -1;         // an all-NaN row's token
constexpr int32_t n_draft_vocab = 65536; // the draft's vocabulary subset (DR1)
constexpr int32_t n_draft_card = n_draft_vocab / 2;
// ORed into the card's error word (fn::xchg::rings::err) by greedy_decide for an all-NaN row; not one of xchg's
// poll bits
constexpr uint32_t err_all_nan = fn::xchg::err_all_nan;

struct logits_args {
    const float * hmix; // this card's X-HMIX slot of the head (xchg.h): the mixed row in both forms, acquired here
    uint32_t * err;     // the card's error word (xchg::err_poll_hmix)
    int32_t card;       // 0 or 1
    const uint8_t * w;  // the card's Q5_K output.weight rows, dense::pack'ed: [n][2560]
    int32_t n;          // vocabulary rows on this card
    float * logits;     // [B][n] out
    void * work;        // logits_work_bytes(n) bytes, or null when that is 0
};

// bytes of the product's work buffer for n rows (0 when the product is unsplit, as at n = 124,160)
size_t logits_work_bytes(int32_t n);

struct reduce_args {
    const float * logits; // [B][n]
    int32_t n;            // logits a row
    int32_t id0;          // the global ID of column 0: card * n_vocab_card in the model (unused with ids)
    uint32_t * pair_own;  // [B][2] out: this card's pair a row (the peer receives the same words)
    float * pair_peer;    // the peer's T-GREEDY receive slot for this round (peer memory)
    void * work;          // greedy_work_bytes(), zeroed once
    // DR1: the ID map [n] (16-byte aligned; distinct IDs in [0, 248,320), disjoint from the peer's), column c's global
    // ID; null: id0 + c
    const int32_t * ids = nullptr;
    // SM1 (FC-RD2's decision hash): the card's decision record's four words (fn::samp::record::words), sent after the
    // rows (words 12 .. 15 of the slot, SP2) by the last block; null: none (V1's message, unchanged)
    const uint32_t * hash_words = nullptr;
};

struct decide_args {
    const uint32_t * pair_own; // [B][2] from greedy_reduce
    float * ring;              // this card's T-GREEDY receive slot for this round (written by the peer); reset as read
    int32_t * token;           // [B] out: the token, the same on both cards
    uint32_t * err;            // this card's error word: xchg::err_poll_greedy, err_all_nan
    uint32_t * pair_seen;      // [B][2] out, test only (nullable): the peer's pair as polled
    int32_t card;              // 0 or 1
    // SM1: the record's words, compared with the peer's (polled and reset by lane 31): a mismatch sets
    // fn::xchg::err_hash; a match of a valid record sets *checked; null: none (both cards alike)
    const uint32_t * hash_words = nullptr;
    uint32_t * checked = nullptr;
};

size_t greedy_work_bytes();

// DR1, the draft head's re-shard (host, load time). draft_ids: card's ID map from the subset (n distinct IDs in
// [0, 248,320), any order; n even, n / 2 % 8 == 0, n / 2 <= 124,160): the subset sorted ascending, ranks card,
// card + 2, ...; ids gets n / 2 IDs. False on an input outside that. draft_head_bytes(m): the packed copy of m Q5_K
// rows. pack_draft_head: output.weight's rows ids[0 .. m - 1] (vocab: the GGUF rows, row_bytes apart; m % 8 == 0)
// packed for logits; false on a bad input.
bool draft_ids(const int32_t * subset, int32_t n, int32_t card, int32_t * ids);
int64_t draft_head_bytes(int32_t m);
bool pack_draft_head(const uint8_t * vocab, size_t row_bytes, const int32_t * ids, int32_t m, uint8_t * dst);

// Enqueue on stream (a cudaStream_t). False, with nothing enqueued, for a row count, size or pointer the contract
// rejects; false after a failed launch submission (the CUDA error is left for the caller).
bool logits(int32_t rows, const logits_args & a, void * stream);
bool greedy_reduce(int32_t rows, const reduce_args & a, void * stream);
bool greedy_decide(int32_t rows, const decide_args & a, void * stream);

// FMU2: the logits of 2..4 lanes of max_rows rows in one launch that reads the card's vocabulary rows once
// (dense::product_joint, lane outputs). a[k]: lane k's arguments, its X-HMIX slot lane 0's plus k hmix_lane_floats and
// its own logits buffer; every lane's card, w and n the same, work null (the product unsplit). Lane k's logits are
// bitwise logits(max_rows, a[k]); a failed X-HMIX acquire raises err (the joint round's word) instead of a lane's.
// False, with nothing enqueued, on an input logits or the joint product rejects.
bool logits_joint(int32_t lanes, const logits_args * a, int32_t hmix_lane_floats, uint32_t * err, void * stream);

} // namespace fn::head
