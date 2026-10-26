#pragma once

// fn/xchg: the card-to-card protocol (STANDARDS: every cross-card hand-off goes through here). HX1's A3 sentinel
// protocol: each card owns receive rings in its own memory, prefilled with a sentinel word the payload never contains;
// the peer writes the payload with plain 16-byte stores (8- and 4-byte stores only at the unaligned ends of a run) and
// no fence; the receiver polls until no sentinel word remains, then the slot is reset to the sentinel. Three rotating
// slots per message. HX1: 1.98 us fixed + 6.05 GB/s for the simultaneous pattern at one sending block [M].
//
// Messages (seed graph H, fnfl/design/history/wp3/transfers.json; payloads per row, F32 words):
//   X-HC   (msg::hc):   a card's HC moments, 1,300 words a row (v[4], u[4][320], q[4][4]; the layout of fn::hc's mom),
//                       written by hc_down's blocks as each block finishes its rows; read by every block of the peer's
//                       hc_up.
//   X-HMIX (msg::hmix): the mixed row in both forms (HC3): F32 and FNFL's activation format (act.h: int8 codes and a
//                       {d, s} pair a run of 32), each card writing its half (1,280 values: 5,120 + 1,280 + 320 bytes a
//                       row) into the peer's slot and its own at the same offsets, so a slot holds the full row in the
//                       consumers' layout (hmix_layout below). Written by hc_up's blocks: the F32 words as each block
//                       finishes its features, a run's codes and scales by the block that completes the run. Read by
//                       the next sublayer's first kernels (the consumers), many blocks each, which poll the peer half.
//   X-A, X-F, X-ROUTE, X-SELECT: R2's layer exchanges (below).
//   X-PLE-STATS (msg::ple): the PLE layer's 13 statistics, 16 words a row (E1, at the end of this file): written by
//                       warp 0 of each block of fn::ple's stats; read by every block of the peer's update; reset by this
//                       card's next stats.
//   X-HC and X-HMIX also carry the HC head (V1, FC-HEAD1: hc.h's head mode): X-HC with 1,284 words a row (no q), in
//                       the same slots and with the same readers, resetter and chain as a sublayer's.
//   T-GREEDY (msg::greedy, V1, FC-DEC1): a card's greedy candidate a row, 2 words (the canonical F32 value bits and
//                       the global token ID as U32), rows packed (row b at words 2b, 2b+1); written by the last block
//                       of the card's greedy reduce; read once by the peer's greedy decide (fn/kernels/head).
//
// The four steps of each hand-off (STANDARDS "Memory and publication"):
//   X-HC, slot n % 3 of card r's ring for sublayer n (HC4: the peer's region and card r's own region, hc_own_at):
//     1. writes:       each block of the peer's hc_down(n) stores its finished moment words into the peer region of
//                      the slot (weak global stores to peer memory over PCIe); HC4: with the fused launch, each block of
//                      card r's own down phase stores the same words of card r's moments into the own region (weak local
//                      stores, with the same put_hc_rows). Every word is written exactly once.
//     2. publication:  none beyond the stores themselves: each word is its own flag, so no fence or flag store is
//                      needed (A3). A word is either the sentinel or its final value: stores are naturally aligned and
//                      at least 4 bytes, so no 4-byte word can tear.
//     3. acquire:      every reading lane of card r's hc_up(n) polls its own 16-byte words with volatile (system-scope,
//                      L1-bypassing) loads until none is the sentinel, and uses the polled registers directly; nothing
//                      is read through a cache that could hold an older copy. HC4: the fused launch's up phase (blocks
//                      0..39) polls the own region the same way, so it waits only for the words it reads; it reads x
//                      nowhere (it recomputes it from the kernel's inputs), so the polled words are its only data from
//                      another block of its grid.
//     4. release:      card r's hc_down(n+1) resets the slot (both regions, all max_rows rows) to the sentinel. Stream
//                      order puts it after every block of card r's hc_up(n), the slot's only readers. HC4: the fused
//                      launch(n+1) resets it in its down phase, before its own moments are published (below).
//   X-HMIX, slot n % 3 of card r's ring (HC3; the peer's half of the row, hmix_layout; card r's own half is written
//   into the same slot by card r's own hc_up(n), which stream order completes before any reader of the slot runs):
//     1. writes:       each block of the peer's hc_up(n) stores its finished F32 words, and the block that completes
//                      a run of 32 stores that run's codes (4-byte words) and its {d, s} pair (weak stores to peer
//                      memory). Every word is written exactly once.
//     2. publication:  none, as for X-HC: each word is its own flag. Code words have their own sentinel
//                      (code_sentinel: four -128 codes, which the format never produces); F32 and scale words are
//                      canonical F32 bits.
//     3. acquire:      every block of each consumer polls, with volatile 16-byte loads (hmix_poll_*), every word of the
//                      peer half it will read, after issuing its first weight loads: the codes and scales for quantized
//                      products, and also the F32 words when the launch has a BF16 matrix. It uses the polled
//                      registers, or (F32 words of a BF16 product) reads them after its block barrier through L1,
//                      which holds no line of the slot from before the poll (one block an SM, L1 invalidated at kernel
//                      start). The consumers of sublayer n on card r (its readers): the GDN input launch, or the QSA
//                      input launch (index owner and peer); the router-plus-shared-gate launch, then EX2's gate/up
//                      (codes and scales, the last reader of an FFN slot); the head's logits launch.
//     4. release:      card r's hc_down(n+1) resets the slot (both halves, all max_rows rows) to the sentinels.
//                      Stream order puts it after every reader above, all of which run before hc_down(n+1); the own
//                      half is rewritten by card r's next hc_up into the slot before any later reader.
//   T-GREEDY, slot s(k) of card r's ring for the k-th decided round (the caller's rings::slot(r, msg::greedy, s(k))):
//     1. writes:       the last block to finish the peer's greedy reduce stores each row's pair with one 8-byte
//                      store (weak, to peer memory). Every pair is written exactly once.
//     2. publication:  none, as for X-HC: both words of a pair are written by one naturally aligned 8-byte store.
//     3. acquire:      each row has exactly one reading lane in card r's greedy decide, which polls its 8-byte pair
//                      with volatile loads until neither word is the sentinel.
//     4. release:      that lane resets the pair to the sentinel right after reading it (same thread, same address).
//
// Why no slot is overwritten before its reset (the dependency chain; stream order on each card is
// down(n) < up(n) < consumer(n) < down(n+1)):
//   X-HC slot s(n) on card r is next written by the peer's down(m), the next sublayer m > n with s(m) = s(n). The peer
//   starts down(m) only after its consumer(m-1) has read all of card r's X-HMIX(m-1); card r's up(m-1) sends that, and
//   it runs after card r's down(n+1), which did the reset, whenever m - 1 >= n + 1. So two slots suffice (m >= n + 2);
//   the third is one sublayer of slack. With one slot (m = n + 1) the peer's down(n+1) could write before card r's
//   down(n+1) reset it.
//   X-HMIX slot s(n) on card r is next written by the peer's up(m), the next sublayer m > n with s(m) = s(n), so
//   m >= n + 2. The peer's up(m) sends only after it has polled card r's X-HC(m), which card r's down(m) sends; down(m)
//   follows (stream order, a kernel boundary) card r's down(n+1), which reset the slot after every reader of
//   sublayer n (m - 1 >= n + 1). So two slots suffice; the third is one sublayer of slack. (HC2's single reader reset
//   each word as it read it; with many readers the reset moved to down(n+1), as X-HC's.)
//   T-GREEDY slot s(k) on card r is next written by the peer's greedy reduce of a later round m > k. The peer
//   reduces round m's logits only after its head of round m, whose up phase polls card r's X-HC of that head; card r
//   sends that X-HC from its head down phase of round m, which follows (stream order) its decide of round k, the
//   slot's only reader and resetter. So one slot would suffice; slots rotate as X-HC's do, for symmetry.
//   HC4, the fused launch (down and up of sublayer n in one cooperative grid): the kernel boundary between down(n+1)
//   (the reset) and up(n+1) (the X-HMIX send) that the chains above lean on is gone. The resets of slot s(n) (X-HC,
//   both regions, and X-HMIX) are stores of launch(n+1)'s down phase, made by every thread before the block's second
//   barrier; after it, warps 0..B-1 (the block's only moment senders) run __threadfence() before any moment store.
//   Every up-phase block of launch(n+1) sends X-HMIX(n+1) only after it has polled moment words from all 72 down
//   blocks (the lowering lanes read every u, v and q row), and its sends depend on the polled values, so every reset
//   store is performed in card r's L2 before any X-HMIX(n+1) word leaves, and before the peer can start its sublayer
//   n+2, which writes slot s(n+2) (= s(n) when s(n+2) = s(n), as the target graph's 94, 95, 96: slots 1, 2, 1). The own
//   region is written only by card r's launch(m) with s(m) = s(n), m >= n + 2, after launch(n+1) in stream order.
//   The rule this needs from a schedule: consecutive sublayers use different slots (s(n+1) != s(n)). n % 3 does; a
//   repeating graph whose length is not a multiple of 3 must end on a slot different from both its first and its
//   second-to-last (hc_bench's slot_of).
//
// Payload rule: a non-finite value never equals the sentinel after canonicalization: every NaN is sent as the canonical
// quiet NaN 0x7fc00000; infinities and every finite value are sent unchanged. The sentinel 0xffffffff is a NaN, so no
// canonical payload word equals it. Both cards' local copies are canonicalized the same way, so both cards hold the
// same bits. X-HMIX's code words are four int8 codes in [-127, 127] (act.h), so none equals code_sentinel; a word of
// four -1 codes equals the F32 sentinel's bits, which is why code words have their own.
//
// Owners: rings owns both cards' receive rings and error words; the kernels take non-owning slot pointers. The owner
// must outlive every queued kernel that uses a slot (callers synchronize both devices before destroying it).

#include <cstdint>
#include <memory>

namespace fn::xchg {

enum class msg : int32_t {
    hc = 0,      // X-HC
    hmix = 1,    // X-HMIX
    a = 2,       // X-A (R2)
    f = 3,       // X-F (R2)
    route = 4,   // X-ROUTE (R2)
    select = 5,  // X-SELECT (R2)
    ple = 6,     // X-PLE-STATS (E1)
    greedy = 7,  // T-GREEDY (V1)
    samp_a = 8,  // SM1: the sampler's first exchange (top-k lists, maxima or histograms; the decision hash)
    samp_b = 9,  // SM1: the cut bin's members (top-p)
    samp_c = 10, // SM1: the settling exchange (kept mass, local candidates, verify numerators)
    samp_d = 11, // SM1: the residual (verify, on a rejection)
    samp_x = 12, // SM1: the closing check of the decision hash
    draft = 13,  // X-DRAFT-IN (DR1)
};
constexpr int32_t n_msg = 14;
constexpr int32_t n_slot = 3; // slots of every message but the sampler's (theirs: slots_of)
constexpr int32_t max_rows = 5; // SP2: the five-row verification pass
constexpr int32_t words_hc = 1300;   // F32 words a row
// HC4: an X-HC slot holds the peer's moments (max_rows rows, from word 0) and then this card's own (max_rows rows from
// hc_own_at), which HC's fused sublayer launch (fn::hc::sub) writes and polls in the same grid
constexpr int32_t hc_own_at = max_rows * words_hc;
// FMS2 item 5b: the joint X-HC layout (fn::hc's sub_joint only; the lone launches keep words_hc rows). A lane's region
// of the slot (the peer's or its own, hc_own_at words each) holds the v words, row b's four at 4 b (stream s at
// 4 b + s), then the 72 down blocks' moments block-major: block vb's 18 rows of row b at hcj_run_at(vb, b), so each
// block's moments of a lane are one contiguous run of 90 words (a short block's unused words are written as zero).
// The same words as the lone layout, placed by the readers; no padding fits the slot, so a block's run starts 8-byte
// aligned.
// Host and device: the CUDA compiler sees __host__ __device__; a host compiler sees a plain function.
#if defined(__CUDACC__)
#define FN_XCHG_HD __host__ __device__
#else
#define FN_XCHG_HD
#endif
constexpr int32_t hcj_rows_block = 18;                         // a down block's moment rows of one stream
constexpr int32_t hcj_block_words = max_rows * hcj_rows_block; // 90
constexpr int32_t hcj_v_words = 4 * max_rows;                  // 20
constexpr int32_t hcj_blocks = 72;
FN_XCHG_HD constexpr int32_t hcj_run_at(int32_t vb, int32_t b) {
    return hcj_v_words + hcj_block_words * vb + hcj_rows_block * b;
}
static_assert(hcj_run_at(hcj_blocks, 0) == hc_own_at && hcj_v_words % 4 == 0, "the joint X-HC layout fills the region");
constexpr int32_t words_hmix = 1280; // a card's half of the mixed row: F32 words, and codes a row of it
constexpr int32_t words_half = 1280; // X-A, X-F: F32 words a row (the destination half)
constexpr int32_t words_route = 24;  // X-ROUTE: 21 words a row (10 IDs, 10 weights, the shared gate), padded to 24
constexpr int32_t words_ple = 16;    // X-PLE-STATS: 13 F32 statistics a row, padded with 3 zero words
constexpr int32_t words_greedy = 2;  // T-GREEDY: (value, ID) a row, rows packed (8 bytes a row)
constexpr int32_t words_draft = 6408; // X-DRAFT-IN: 6,400 F32 values and 5 sums of squares a row, padded with 3 zeros
// SM1: T-GREEDY's decision hash: four words after the rows (SP2: words 12 .. 15, after five rows of two words), the
// slot 16 words
constexpr int32_t greedy_hash_at = 12;
// SM1: the sampler's messages (fn/kernels/samp: samp.cu states each layout). A header of 4 words (the decision
// record: round and hash, 31 bits each) leads samp_a and samp_c.
constexpr int32_t samp_hdr_words = 4;
constexpr int32_t samp_a_row_words = 4 + 4 * 2049; // a histogram row: 4 words, then (count, sum lo, sum hi, 0) a bin
constexpr int32_t samp_b_row_words = 2 * 124160;   // a row's members: (logit, ID) pairs, up to a whole shard
constexpr int32_t samp_c_row_words = 8;
constexpr int32_t samp_d_words = 8;
constexpr int32_t samp_x_words = 4;
// X-SELECT: QA1b's selected-block body (fn::qsa's layout) of up to max_rows rows. A row's body is at most 517 words
// (fn::qsa::max_row_words: 2 counts, 512 block IDs, 3 tail cells); the message adds 8 header words.
constexpr int32_t select_row_words = 517;
constexpr int32_t select_header_words = 8;
constexpr int32_t select_words(int32_t body_words) {
    return select_header_words + (body_words + 3) / 4 * 4;
}

constexpr uint32_t sentinel = 0xffffffffu;      // a NaN bit pattern
constexpr uint32_t canonical_nan = 0x7fc00000u; // every NaN payload word is sent as this
constexpr uint32_t code_sentinel = 0x80808080u; // X-HMIX code words: four -128 codes, which act.h never produces

// X-HMIX's slot (HC3): the full mixed row of max_rows rows in three regions, each in the consumers' layout (act.h's
// view and an F32 [B][2560] row): F32 [5][2560], then codes int8 [5][2560], then scales float2 {d, s} [5][80]. Card c
// writes values c*1280 .. c*1280+1279 of each row (runs c*40 .. c*40+39) into both cards' slots.
struct hmix_layout {
    static constexpr int32_t width = 2 * words_hmix;            // values a row: 2,560
    static constexpr int32_t runs = width / 32;                 // runs of 32 a row: 80
    static constexpr int32_t f32_words = max_rows * width;      // region 0
    static constexpr int32_t code_words = max_rows * width / 4; // region 1 (4 codes a word)
    static constexpr int32_t scale_words = max_rows * runs * 2; // region 2
    static constexpr int32_t code_at = f32_words;               // word offsets in the slot
    static constexpr int32_t scale_at = f32_words + code_words;
    static constexpr int32_t slot_words = f32_words + code_words + scale_words;
    static constexpr int32_t half_payload_bytes = words_hmix * 4 + words_hmix + words_hmix / 32 * 8; // 6,720 a row
};
static_assert(hmix_layout::slot_words == 16800 && hmix_layout::code_at % 4 == 0 && hmix_layout::scale_at % 4 == 0,
              "X-HMIX's regions stay 16-byte aligned");

// Poll errors use the low 16 bits, with 8 and 9 reserved for PX1's prototype in its separate word; kernel data errors
// use bits 16-31. Keep every allocated or reserved bit in the table below.
constexpr uint32_t err_poll_hc = 1u << 0;
constexpr uint32_t err_poll_hmix = 1u << 1;
constexpr uint32_t err_poll_test = 1u << 2;
constexpr uint32_t err_poll_a = 1u << 3;
constexpr uint32_t err_poll_f = 1u << 4;
constexpr uint32_t err_poll_route = 1u << 5;
constexpr uint32_t err_poll_select = 1u << 6;
constexpr uint32_t err_poll_ple = 1u << 7;
constexpr uint32_t err_reserved_px1_8 = 1u << 8;
constexpr uint32_t err_reserved_px1_9 = 1u << 9;
constexpr uint32_t err_poll_greedy = 1u << 10;
constexpr uint32_t err_poll_start = 1u << 11; // TG1: a round's start flag was not set in time (fn::round)
constexpr uint32_t err_token = 1u << 16;
constexpr uint32_t err_all_nan = 1u << 17;
constexpr uint32_t err_select_body = 1u << 18; // TG1: an X-SELECT body length outside 2 B .. B * 517 words
constexpr uint32_t err_round = 1u << 19;       // TG1: a round descriptor slot not the round's (fn::round)
constexpr uint32_t err_test_commit = 1u << 20; // SR1: test only, an error injected before commit (fn::round)
constexpr uint32_t err_poll_samp = 1u << 12;   // SM1: a sampler exchange's bounded poll gave up (fn::samp)
constexpr uint32_t err_poll_draft = 1u << 13;  // DR1: X-DRAFT-IN
constexpr uint32_t err_hash = 1u << 21;        // SM1: the cards' decision records differ (FC-RD2; fn::samp)
constexpr uint32_t err_samp_empty = 1u << 22;  // SM1: a row with no candidate (fn::samp)
constexpr uint32_t err_samp_count = 1u << 23;  // SM1: a cut bin's members differ in number from its count (fn::samp)
constexpr uint32_t err_draft_depth = 1u << 24; // DR1: a draft row's cell outside the draft cache (fn::draft)
constexpr uint32_t err_kv_tier = 1u << 27;     // ME1: a selection the KV tier cannot remap (fn::qsa::tier)
constexpr uint32_t err_gdn_sel = 1u << 28;     // ME11: the GDN selector outside the contract at commit (fn::round)
constexpr uint32_t err_x16_range = 1u << 29;   // FMS2 item 7: a finite joint X-A/X-F value beyond FP16 (sent as inf)

struct error_bit {
    const char * name;
    uint32_t value;
};
constexpr error_bit card_error_bits[] = {
    {"poll hc", err_poll_hc},
    {"poll hmix", err_poll_hmix},
    {"poll test", err_poll_test},
    {"poll a", err_poll_a},
    {"poll f", err_poll_f},
    {"poll route", err_poll_route},
    {"poll select", err_poll_select},
    {"poll ple", err_poll_ple},
    {"PX1 reserved bit 8", err_reserved_px1_8},
    {"PX1 reserved bit 9", err_reserved_px1_9},
    {"poll greedy", err_poll_greedy},
    {"poll start", err_poll_start},
    {"poll draft", err_poll_draft},
    {"token", err_token},
    {"all NaN", err_all_nan},
    {"select body", err_select_body},
    {"round", err_round},
    {"test commit", err_test_commit},
    {"poll samp", err_poll_samp},
    {"decision hash", err_hash},
    {"sampler empty row", err_samp_empty},
    {"sampler member count", err_samp_count},
    {"draft depth", err_draft_depth},
    {"KV tier", err_kv_tier},
    {"GDN selector", err_gdn_sel},
    {"X-A/X-F FP16 range", err_x16_range},
};
constexpr bool card_error_bits_are_unique() {
    constexpr int32_t n = sizeof(card_error_bits) / sizeof(card_error_bits[0]);
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = i + 1; j < n; ++j) {
            if (card_error_bits[i].value == card_error_bits[j].value) {
                return false;
            }
        }
    }
    return true;
}
static_assert(card_error_bits_are_unique(), "card error bits must be pairwise distinct");
static_assert((err_token & 0xffffu) == 0u && (err_all_nan & 0xffffu) == 0u && (err_select_body & 0xffffu) == 0u &&
                  (err_round & 0xffffu) == 0u && (err_poll_start & 0xffff0000u) == 0u && (err_hash & 0xffffu) == 0u &&
                  (err_samp_empty & 0xffffu) == 0u && (err_poll_samp & 0xffff0000u) == 0u &&
                  (err_draft_depth & 0xffffu) == 0u && (err_poll_draft & 0xffff0000u) == 0u,
              "kernel data errors must not overlap poll bits");

// words a row of a row message (X-HC, X-A, X-F, X-ROUTE, X-PLE-STATS, T-GREEDY, X-DRAFT-IN; X-HMIX: a slot's words /
// max_rows, its regions are hmix_layout's); 0 for X-SELECT
constexpr int32_t words_per_row(msg m) {
    return m == msg::hc       ? words_hc
           : m == msg::hmix   ? hmix_layout::slot_words / max_rows
           : m == msg::route  ? words_route
           : m == msg::select ? 0
           : m == msg::ple    ? words_ple
           : m == msg::greedy ? words_greedy
           : m == msg::draft  ? words_draft
                              : words_half;
}
// words a slot: max_rows rows, or the largest X-SELECT body (max_rows rows of select_row_words)
constexpr int32_t slot_words(msg m) {
    return m == msg::select   ? select_words(max_rows * select_row_words)
           : m == msg::hc     ? 2 * hc_own_at
           : m == msg::hmix   ? hmix_layout::slot_words
           : m == msg::greedy ? 16
           : m == msg::samp_a ? samp_hdr_words + max_rows * samp_a_row_words
           : m == msg::samp_b ? max_rows * samp_b_row_words
           : m == msg::samp_c ? samp_hdr_words + max_rows * samp_c_row_words
           : m == msg::samp_d ? samp_d_words
           : m == msg::samp_x ? samp_x_words
                              : max_rows * words_per_row(m);
}
// slots a message has: three for each but the sampler's (X-DRAFT-IN too); two for samp_a .. samp_d (by the decision's
// parity) and one for samp_x (the chains are below)
constexpr int32_t slots_of(msg m) {
    const int32_t i = static_cast<int32_t>(m);
    const bool by_parity = i >= static_cast<int32_t>(msg::samp_a) && i <= static_cast<int32_t>(msg::samp_d);
    return m == msg::samp_x ? 1 : by_parity ? 2 : n_slot;
}
static_assert(slots_of(msg::draft) == n_slot, "X-DRAFT-IN keeps a slot for each step (step j: slot j)");
static_assert(words_hc % 4 == 0 && words_hmix % 4 == 0 && words_half % 4 == 0 && words_route % 4 == 0 &&
                  words_ple % 4 == 0 && slot_words(msg::select) % 4 == 0 && slot_words(msg::greedy) % 4 == 0 &&
                  words_greedy == 2 && slot_words(msg::samp_a) % 4 == 0 && slot_words(msg::samp_b) % 4 == 0 &&
                  slot_words(msg::samp_c) % 4 == 0 && slot_words(msg::samp_d) % 4 == 0 &&
                  slot_words(msg::samp_x) % 4 == 0 && greedy_hash_at + 4 <= 16 && 2 * max_rows <= greedy_hash_at &&
                  words_draft % 4 == 0,
              "rows and slots stay 16-byte aligned (T-GREEDY: 8-byte rows in a 16-byte aligned slot)");

// Enables peer access both ways between dev0 and dev1 (already-enabled is fine). False, with a message on stderr, if
// either card cannot reach the other.
bool enable_peer_access(int32_t dev0, int32_t dev1);

// The receive rings and error words of both cards: card c's live in device dev[c]'s memory and are written by the other
// card. Constructed prefilled with the sentinels (reset_all) and with zero error words.
class rings {
  public:
    rings(int32_t dev0, int32_t dev1);
    // FMU2: a view of rings another owner allocated (joint_rings' lanes): buf[c] card_words floats (rings_card_bytes
    // less the error word) and err[c] one word on device dev[c]; not freed by the view, not prefilled by it
    rings(int32_t dev0, int32_t dev1, float * const buf[2], uint32_t * const err[2]);
    ~rings();
    rings(const rings &) = delete;
    rings & operator=(const rings &) = delete;

    bool ok() const { return ok_; }
    int32_t device(int32_t card) const { return dev_[card]; }

    // card's receive slot s (0 .. slots_of(m) - 1) of message m: slot_words(m) words, 16-byte aligned
    float * slot(int32_t card, msg m, int32_t s) const;
    uint32_t * err(int32_t card) const { return err_[card]; }

    // Synchronous (callers have drained both devices): refill every slot with the sentinel (X-HMIX's code words with
    // code_sentinel) and zero both error words.
    bool reset_all() const;
    // Synchronous read of card's error word.
    bool read_err(int32_t card, uint32_t & out) const;

  private:
    int32_t dev_[2] = {-1, -1};
    float * buf_[2] = {nullptr, nullptr}; // n_msg x n_slot slots a card
    uint32_t * err_[2] = {nullptr, nullptr};
    bool ok_ = false;
    bool owns_ = true;
};

// bytes rings allocates on each card: every slot of every message, and the error word
int64_t rings_card_bytes();

// ---- FMU2: the joint round's exchange memory ----
//
// A joint round (fn/model/layer.h joint_lanes) runs 2..4 slots' rounds of max_rows rows each as lanes on one stream a
// card. Every per-lane piece (HC, the bodies' per-slot kernels, PLE, the head, the decision) uses its lane's ring, a
// complete rings in the lone format with the lone protocol and slot rules, so lanes never share a slot; the joint dense
// products read every lane's X-HMIX slot at a fixed stride (lane_floats). The destination-half exchanges after the
// joint products (X-A, X-F) run once over all lanes' rows on the joint slots (n_slot each, joint_rows rows, the R2
// protocol and slot rule unchanged). Error words: one a lane ring (its pieces') and the joint word (X-A, X-F); a joint
// round's commit on every lane checks its lane's word and the joint word, so a joint error fails every lane's round.
constexpr int32_t max_lanes = 4;
constexpr int32_t joint_rows = max_lanes * max_rows;

class joint_rings {
  public:
    joint_rings(int32_t dev0, int32_t dev1);
    ~joint_rings();
    joint_rings(const joint_rings &) = delete;
    joint_rings & operator=(const joint_rings &) = delete;

    bool ok() const { return ok_; }
    // lane u's ring (0 .. max_lanes - 1)
    const rings & lane(int32_t u) const { return *lane_[u]; }
    // floats from lane u's ring to lane u + 1's (every slot at the same offset in each)
    static int64_t lane_floats();
    // card's joint receive slot s (0 .. n_slot - 1) of X-A or X-F: joint_rows rows of words_half words
    float * slot(int32_t card, msg m, int32_t s) const;
    uint32_t * err(int32_t card) const { return err_[card] == nullptr ? nullptr : err_[card] + max_lanes; }

    // Synchronous (callers have drained both devices): every lane ring and joint slot refilled with the sentinels and
    // every error word zeroed
    bool reset_all() const;

  private:
    int32_t dev_[2] = {-1, -1};
    float * buf_[2] = {nullptr, nullptr};    // max_lanes lane rings, then X-A's and X-F's joint slots
    uint32_t * err_[2] = {nullptr, nullptr}; // max_lanes lane words, then the joint word
    std::unique_ptr<rings> lane_[max_lanes];
    bool ok_ = false;
};

// bytes joint_rings allocates on each card
int64_t joint_rings_card_bytes();

// bytes of one slot of message m
constexpr int64_t slot_bytes(msg m) {
    return int64_t(slot_words(m)) * 4;
}

// ---- xchg-test only: the producer and consumer stand-ins that drive both messages with checked payloads ----

// Payload word w of round k from card c, message m, as F32 bits: a hash, except that some rounds carry non-finite
// values (rounds with k % 7 == 3: every 97th word is a NaN, alternately the sentinel pattern itself and other NaNs;
// every 89th word is +inf or -inf). Shared by host and device so the expected value is the same function.
uint32_t test_word_host(int32_t c, msg m, uint32_t k, int32_t w);

// X-HMIX's code word w of round k from card c (HC3): never code_sentinel; four -1 codes (the F32 sentinel's bits) in
// some words of the non-finite rounds.
uint32_t test_code_word_host(int32_t c, uint32_t k, int32_t w);

// Enqueue the stand-in producer: the senders' block decomposition (X-HC: hc_down's 72 blocks of 18 rows and the v
// words; X-HMIX: hc_up's 56 blocks of 18 features and 16 of 17 for the F32 values, and 40 blocks completing a run's
// codes and scales each, HC3), writing round k's payload into peer_slot with the protocol's stores. For X-HC, also
// resets reset_slot (this card's X-HC slot of the previous round) and hmix_reset (its X-HMIX slot of the previous
// round, as hc_down does; xchg.h step 4); null: none.
bool test_send(msg m, int32_t rows, int32_t card, uint32_t k, float * peer_slot, float * reset_slot, void * stream,
               float * hmix_reset = nullptr);
// Enqueue the stand-in consumer: X-HC as hc_up reads it (72 blocks, each polling every word of the slot); X-HMIX as
// the consumers read it (HC3: 72 blocks, each polling every word of the sender's half, no reset). Counts words that
// differ from the canonical expected value into mism[0] and words checked into mism[1] (64-bit counters).
bool test_recv(msg m, int32_t rows, int32_t sender, uint32_t k, float * slot, uint32_t * err, unsigned long long * mism,
               void * stream);

// ---- R2: the layer exchanges (X-A, X-F, X-ROUTE, X-SELECT), each a small kernel after its producer ----
//
// The four steps, for each (the protocol of X-HMIX: one reading lane a 16-byte word, which resets it after reading):
//   1. writes:      the sender's kernel stores each payload word once into the peer's slot (weak 16- or 4-byte stores
//                   to peer memory), after its producer finished (stream order).
//   2. publication: none beyond the stores: each word is its own flag (A3).
//   3. acquire:     each word has one reading lane in the receiver's kernel, which polls it with volatile loads until
//                   it is not the sentinel and uses the polled registers.
//   4. release:     that lane resets the word to the sentinel right after reading it.
// Why no slot is overwritten before its reset: every send on card c follows (stream order) card c's receipt of a peer
// message that the peer sent after its own receive of the previous use of the slot, so one slot would do; slots rotate
// by use (s = use % 3) for symmetry with HC2's rings.
// Payload words never equal the sentinel: F32 values are canonicalized (xchg.h payload rule); X-ROUTE's IDs are 0..511
// and its pad words 0; X-SELECT's words are counts, block and cell IDs and byte offsets (all below 2^18) and zero pad.
//
// Layouts:
//   X-A, X-F:  [B][1280] F32, row b: the receiver's own half (features r*1280 ..) of the sender's partial [B][2560].
//   X-ROUTE:   [B][24] words, row b: IDs 0..9 (I32), weights 10..19 (F32), the shared expert's gate 20 (F32, after its
//              sigmoid), pad 21..23 (0).
//   X-SELECT:  8 header words, then QA1b's body as select() wrote it (fn::qsa's layout: each row's counts, block IDs
//              and tail cells, rows packed), then zero words up to a multiple of 4. Header word t is row t's byte
//              offset in the body for t < B and the body's total bytes for t >= B. The body's length depends only on
//              the depth and B; TG1: it is read on the device (the sender from offsets[B], the receiver from header
//              word 4, which is always the total since B <= 4), so one captured launch of each serves every depth.
//              The receiver hands the body and offsets to its consumer unchanged: no expansion to cells and no union.
//              Readers and resetters are unchanged: one reading lane a 16-byte word, which resets it after reading;
//              the receiver's thread 1 reads header group 1 first, alone, and resets it before the others poll
//              theirs, so no word is polled after its reset. Words past the message are neither written nor read.

// X-A / X-F send: peer_slot row b = partial[b][peer_card * 1280 .. +1279] (partial [B][2560] F32, 16-byte aligned)
bool send_half(int32_t rows, int32_t peer_card, const float * partial, float * peer_slot, void * stream);
// X-A / X-F receive and sum: out[b][i] = partial[b][card * 1280 + i] + the peer's word (the sum is commutative, so the
// same on either card); the slot is polled and reset. code: err_poll_a or err_poll_f.
bool recv_add(int32_t rows, int32_t card, const float * partial, float * slot, float * out, uint32_t * err,
              uint32_t code, void * stream);
// FMU2: send_half and recv_add over a joint round's lanes (1 .. max_lanes) on a joint slot (joint_rings::slot), the
// layout [B][1280] a slot. FMS13: lane k's rows 0 .. lane_rows - 1 (1 .. max_rows) are rows max_rows k + i of the
// partial, the slot and out; only they are sent, polled, reset and written. lane_rows max_rows: send_half's kernels
bool               send_half_joint(int32_t       lanes,
                                   int32_t       lane_rows,
                                   int32_t       peer_card,
                                   const float * partial,
                                   float *       peer_slot,
                                   void *        stream);
bool               recv_add_joint(int32_t       lanes,
                                  int32_t       lane_rows,
                                  int32_t       card,
                                  const float * partial,
                                  float *       slot,
                                  float *       out,
                                  uint32_t *    err,
                                  uint32_t      code,
                                  void *        stream);
// FMS2 item 7: the joint X-A / X-F in 16 bits (joint rounds only, behind the joint frame's switch). The payload is the
// peer's half as FP16, round to nearest even (x16_bits), row b at b x16_row_halves halves of the joint slot (the F32
// slot's first half), and each 16-bit half is its own flag: the sentinel half 0xffff, a NaN, which no payload half
// equals (every NaN is sent as x16_nan), so writers may store single halves of a 32-bit word. A finite value beyond
// FP16's range is sent as an infinity and sets err_x16_range in the sender's err (the round fails). The receive polls
// each 16-byte group until none of its 8 halves is the sentinel, resets it, and adds in F32: out = partial (own) +
// the peer's value, today's order. send_half_joint16's err: the sending card's joint word.
constexpr int32_t x16_row_halves = words_half; // 1280 halves (640 words) a row
constexpr uint16_t x16_sentinel = 0xffffu;
constexpr uint16_t x16_nan = 0x7e00u;
bool send_half_joint16(int32_t rows, int32_t peer_card, const float * partial, float * peer_slot, uint32_t * err,
                       void * stream);
bool recv_add_joint16(int32_t rows, int32_t card, const float * partial, float * slot, float * out, uint32_t * err,
                      uint32_t code, void * stream);
// X-ROUTE send (router owner): ids_full [B][ids_stride] (the first 10 of each row), wts [B][10], sg [B]; also writes
// the compact ids_local [B][10] the experts read on this card
bool send_route(int32_t rows, const int32_t * ids_full, int32_t ids_stride, const float * wts, const float * sg,
                int32_t * ids_local, float * peer_slot, void * stream);
// X-ROUTE receive: ids [B][10], wts [B][10], sg [B]
bool recv_route(int32_t rows, float * slot, int32_t * ids, float * wts, float * sg, uint32_t * err, void * stream);
// X-SELECT send (index owner): offsets I32 [B + 1] and body U32 [offsets[B] / 4 words], as fn::qsa::select wrote them
// (body 16-byte aligned; from 2 B to B * select_row_words words). TG1: the length is read on the device from offsets[B]
// and travels in the header, so one captured launch serves every depth.
bool send_select(int32_t rows, const int32_t * offsets, const uint32_t * body, float * peer_slot, void * stream);
// X-SELECT receive (the peer): offsets I32 [B + 1] and the body, the sender's unchanged; the length is read from the
// header on the device (a length outside the contract sets err_select_body and writes no body). body has room for
// B * select_row_words words.
bool recv_select(int32_t rows, float * slot, int32_t * offsets, uint32_t * body, uint32_t * err, void * stream);
// FMS2 (build item 3, W3): every lane's X-SELECT of the joint verify round, lanes 2 .. 4, in one send and one receive:
// lane k's message is send_select's / recv_select's on its own arguments (offsets[k], body[k], its lane ring's slot
// and error word), the same device code, so every lane's words are its own launches' bitwise
bool send_select_lanes(int32_t lanes, int32_t rows, const int32_t * const * offsets, const uint32_t * const * body,
                       float * const * peer_slot, void * stream);
bool recv_select_lanes(int32_t lanes, int32_t rows, float * const * slot, int32_t * const * offsets,
                       uint32_t * const * body, uint32_t * const * err, void * stream);

// ---- E1: X-PLE-STATS (msg::ple), the PLE layer's 13 statistics a row (WP3 T-PLE; FC-PLE2) ----
//
// Content, row b: words 0..3 sum key_s^2, 4..7 sum h_s^2, 8..11 sum key_s nk_s h_s nq_s (s = stream), 12 sum value^2,
// each over the sender's 1,280 features; 13..15 zero. Sender: fn::ple's stats launch (warp 0 of block b, put_run),
// which also writes the same canonical words to its own card's copy. Receiver: every block of the peer's update launch
// reads every word of rows 0..B-1 (the X-HC pattern: many readers). Both cards sum card 0's partial first.
// The four steps, slot u % 3 of card r's ring for PLE use u (one use a round):
//   1. writes:      warp 0 of each block b of the peer's stats(u) stores row b's 16 words (four 16-byte stores) into
//                   the slot (weak stores to peer memory), after its own reductions; every word written once.
//   2. publication: none beyond the stores: each word is its own flag (A3).
//   3. acquire:     every block of card r's update(u) polls the B x 4 16-byte words (one lane a word) with volatile
//                   loads until none is the sentinel, and shares the polled values through shared memory after a
//                   barrier.
//   4. release:     card r's stats(u + 1) (its block 0) resets slot (u % 3) to the sentinel, all max_rows rows; stream
//                   order puts it after every block of card r's update(u), the slot's only readers. Resetter: card r's
//                   next stats launch. Readers: every block of card r's update(u).
// Why no slot is overwritten before its reset: slot s(u) on card r is next written by the peer's stats(m), m = u + 3
// (s = u % 3). The peer starts stats(m) after its update(m - 1) acquired card r's stats(m - 1) send. Card r's
// stats(m - 1) = stats(u + 2) runs after stats(u + 1) completed (stream order, a kernel boundary), and stats(u + 1)
// did the reset. So the reset is complete before the send the peer waits on. With two slots (m = u + 2) the reset and
// that send would be in the same launch (stats(u + 1)) with no order between them, so three are needed.
// A schedule that runs PLE more than once a round, or moves the reset or the send, restates this chain (FC-X1).
// TG1: in the model pass's captured graphs the use u is read on the device (fn::ple::step_args::use, the round
// descriptor's PLE use: rounds with layer work before this one, so consecutive PLE uses are consecutive u); the slots,
// readers, resetter and chain above are unchanged.
//
// TG1, T-GREEDY in the model pass: slot 0 in every round (graphs and the bootstrap). The chain above (one slot
// suffices) covers rounds with a head. The bootstrap, which has none, and the round after it are launched only after
// the host has seen both cards' completion records of the round before (fn/runtime/target.h), so the slot's previous
// reader and resetter (that round's decide) has completed before the next reduce writes it.

// ---- SM1: the sampler's exchanges (fn/kernels/samp; FC-DEC2, FC-DEC3, FC-RD2; FC1 section 12's "sampler exchanges") ----
//
// Messages (layouts in samp.cu): samp_a, the decision's first exchange (each card's top-k list, or its maximum, or its
// maximum and histogram; with a 4-word header carrying the sender's decision record when the launch is a decision's
// first sender); samp_b, the top-p cut bin's members; samp_c, the settling exchange (kept mass, local candidate, and a
// verify row's numerator and q at its draft; the header on the no-filter path); samp_d, the residual on a rejection;
// samp_x, the closing check (the record's four words). Words: F32 values canonical; 64-bit values as (low, high) with
// a low word of 0xffffffff sent as 0xfffffffe (the sender uses the adjusted value too); IDs below 2^18; a card without
// a candidate sends 0x7fffffff; counts, flags and 31-bit record halves. No payload word is the sentinel.
// Slots: a decision uses slot (round & 1) of samp_a .. samp_d (a draft pick: position & 1); samp_x has one.
//
// The four steps, for each message and path (card r's receive slot; the peer is the sender):
//   1. writes:      the peer's sending launch (its last block of a row's grid, or a decision's single block) stores each
//                   word once, with 16-byte stores (8-byte for a top-k list's odd last pair and for members), to card
//                   r's slot and the same words to its own copy (workspace), after its own work (block arrival order).
//   2. publication: none beyond the stores: each word is its own flag (A3).
//   3. acquire:     card r's reading launch polls each word it reads with volatile loads until it is not the sentinel and
//                   uses the polled registers.
//   4. release:     the reader resets each word to the sentinel right after reading it (one reader a word) for:
//                   top-k lists (tk_decide), histograms (plan), members (resolve), samp_c (decide), samp_d (decide2),
//                   samp_x (close_recv), and every header except min-p's. Min-p's samp_a words (each row's first group
//                   and the header) have many readers, every settle block; card r's decide resets them, after them in
//                   stream order.
// Readers and resetters (one decision, card r): top-k path: tk_decide (samp_a); full path: settle's blocks (samp_a,
// min-p alone; reset by decide), plan (samp_a, top-p), resolve (samp_b), decide (samp_c), decide2 (samp_d); greedy path:
// T-GREEDY (above) with the record's words at 8 .. 11, read and reset by greedy_decide's lane 31.
// Why no slot is overwritten before its reset: slot s = k & 1 of decision k on card r is next written by the peer's
// decision k + 2 (the next with the same parity). The peer's decision k + 2 sends only after it acquired card r's first
// message of decision k + 1 (its first stage polls it, or, with a pass between, the pass's HC exchanges order it), and
// card r sends that after its own decision k's last reader and resetter of the slot (stream order on card r). So two
// slots suffice even for back-to-back decisions with no other exchange between them (the tests, the bench, the draft's
// picks). samp_x is written once by each card's close_send, after the host has seen both cards' last round complete,
// and read and reset by close_recv before the host starts another round. A draft pick and a verify round may share a
// parity; a target pass (its HC exchanges) lies between them.
//
// ---- DR1: the draft rounds (round kinds draft_step and draft_catchup; fn/model/draft.h) ----
//
// The draft rounds use the target's rings, after and between its rounds on the same streams; the host launches a
// draft round only after both cards' completion records of the round before (fn/runtime/target.h), as for TG1.
//
// X-DRAFT-IN (msg::draft, words_draft a row: [e | h0..h3] of the sender's half, its 5 sums of squares, 3 zero words).
// The four steps, slot s of card r's ring:
//   1. writes:      each block of the peer's draft_in_send (block k: segment k of each row) stores the segment's 1,280
//                   words with 16-byte stores and its sum of squares word with one 4-byte store; block 0 of a row also
//                   stores the 3 zero words (weak stores to peer memory). Every word is written once.
//   2. publication: none beyond the stores: each word is its own flag (A3).
//   3. acquire:     each 16-byte group of rows 0..B-1 has one reading thread in card r's draft_in_recv (the block of
//                   its row), which polls it with volatile loads until no word is the sentinel and keeps the polled
//                   registers (in shared memory).
//   4. release:     that thread resets the group to the sentinel right after reading it (same thread, same address).
//   Readers and resetter: card r's draft_in_recv of the use, one thread a group. Slot by use: in a step graph step j
//   uses slot j (j < 3), the catch-up's graph slot 0; eagerly the draft use counter % 3, continuing across rounds.
//   Why no slot is overwritten before its reset: as R2. Every send on card c follows (stream order) card c's receipt
//   of a peer message the peer sent after its own receive of the previous use: step j's send after card c's X-F of
//   step j - 1 (or, at a round's first use, after the previous round, which the host serialized), and the peer's X-F
//   send follows its draft_in_recv of step j - 1. So one slot would do; the rotation is symmetry.
//
// X-HC / X-HMIX in the draft rounds (layer.h hc_rule; the four steps and the chain above are unchanged):
//   draft_step:  a graph of G = 3 d sublayers (each step's attention, FFN and head) uses slot
//                {1, 2, 0}[(G - 1 - k) % 3] for sublayer k: slots 0, 2, 1 repeating, from slot 0 to slot 1.
//                Consecutive sublayers differ, and the graph ends on slot 1 as every target graph does, so its first
//                sublayer resets slot 1 (the slot the round before left dirty) and each later sublayer resets its
//                predecessor's, as down(n+1) does.
//   draft_catchup: its one sublayer (the attention sublayer's HC) uses slot 2 and resets slot 1 (the previous
//                round's). It has no later sublayer to reset slot 2, so card r's datc_prepare releases its own X-HC
//                and X-HMIX slot 2 (all max_rows rows) after the slot's last readers on card r (up2 read X-HC; the K/V
//                product acquired X-HMIX), in stream order. The peer next writes card r's slot 2 at a sublayer of a
//                later round whose down or up waits on a card-r message sent after that prepare, so the release is
//                complete first; the next round's first sublayer resets slot 1 again, already clean (harmless).
//   eager:       sublayer n uses n % 3, continuing the target's counters, as HC2.
//
// X-A / X-F in a draft step: the R2 protocol and chain, slot by use (step j: slot j in a graph; eagerly use % 3). The
// catch-up has no attention output or FFN, so neither.
//
// T-GREEDY in a draft step: slot 0 for every step's pick, as TG1. The chain above (one slot suffices) holds between
// steps: the peer reduces step j + 1's logits only after its head of step j + 1 polled card r's X-HC of that head,
// which card r sends after its decide of step j, the slot's only reader and resetter. The draft head's pair is over
// card r's half of the subset in global IDs (lower ID on an exact tie), the same pair layout. The draft's greedy
// launches pass no decision-record words (SM1's words 8 .. 11 stay unwritten). The catch-up has none.

} // namespace fn::xchg
