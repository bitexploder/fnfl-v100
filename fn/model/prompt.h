#pragma once

// PB1: the bulk prompt path on both cards (FC-PF1, FC-PF3; fnfl/design/2026-10-14-prefill-plan.md, "Fix the execution
// shape first" .. "Publication, state and handoff"). A chunk (at most 3,072 rows, halves of at most 1,536, PF1's
// partition) runs sublayer-major: for each layer, the attention sublayer of half A, then of half B, then the FFN
// sublayer of A (after A's attention exchange), then of B; the next layer's A waits only for A's FFN exchange. Every
// sublayer ends with its full-width FP16 partial copied to the peer by the copy engine (fn/xchg/prompt_xchg.h); the
// other half computes meanwhile. Target HC is replicated (FC-PF3): each card holds both halves of every coefficient
// (its resident half and the peer's replica) and the full streams S [rows][4][2560] F32. After layer 47 the draft's
// catch-up runs over the chunk's positions, half by half (the eh_proj product, the split Q8_0 HC with its two
// messages, K and V into the draft cache). The state objects keep decode's layout and SR1's versions: half A writes
// logical version 1, half B version 2; the caller publishes the chunk (fn::round::advance by the chunk's halves)
// after both cards' work joined clean, so a failed chunk publishes nothing (design note, section 7).
//
// Host code only. Every launch is enqueued directly (no graph); the host computes each launch's rows and positions
// from the chunk, so nothing is allocated, reserved or synchronized between layers. Both cards are enqueued phase by
// phase (card 0, then card 1), every event recorded before the wait that reads it. The message protocol (four steps,
// design note section 6): producer -> its event -> the copy (after the peer freed its receive slot) -> its event -> the
// consumer, which checks the generation in the header; the producer's next write waits for the copy.
//
// Buffers: R1's "prefill arena (PB1)" and "HC and PLE replicas (PB1)" classes (fn/model/state.h), carved here.

#include "fn/model/draft.h"
#include "fn/model/model.h"
#include "fn/model/pass.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fn::model {

constexpr int32_t prompt_chunk_rows = 8192; // PB3: a whole 8,192 span (was 3,072 in halves of 1,536)
constexpr int32_t prompt_half_rows = 4096;

// the arena's bytes a card (n_cells sizes nothing here: the QSA slab is the full 262,144-cell scratch, 32 MiB)
int64_t prompt_arena_bytes();
// the replicas' bytes a card: the peer's half of the 96 target HC sublayers' coefficients and of PLE's, and the draft's
// full enorm and hnorm
int64_t prompt_replica_bytes();

// the declared bulk call sites (design note section 4): host counters, one an enqueue
enum class psite : int32_t {
    embed,
    hc_attn,
    hc_ffn,
    gdn_input,
    gdn_core,
    gdn_out,
    qsa_input,
    qsa_index,
    qsa_attend,
    qsa_out,
    ffn_router,
    ffn_experts,
    ffn_shared,
    ple,
    exchange,
    draft,
    head,
    count
};
constexpr int32_t n_psites = static_cast<int32_t>(psite::count);
const char *      psite_name(psite s);

// one chunk as the driver hands it over
struct prompt_chunk {
    int64_t         index   = 0;  // the run's chunk number (the HC dump's chunk selection, the token slot's parity)
    int64_t         seq     = 0;  // the prompt path's chunk number since init, never repeated (message generations)
    int64_t         start   = 0;  // its first position
    int32_t         rows    = 0;
    int32_t         half[2] = { 0, 0 };
    const int32_t * tok     = nullptr;               // host: the chunk's tokens
    const float *   e[2]    = { nullptr, nullptr };  // device: each card's PLE input rows [rows][2560] (PF1's staging)
    bool            draft   = true;
};

// what the HC dump (PB2's inputs) takes from one chunk: layers, and every stride-th row of each half
struct hc_dump_spec {
    std::vector<int32_t> layers;
    std::vector<int64_t> chunks;          // chunk indices of the run
    int32_t              row_stride = 0;  // 0: no dump
    std::string          dir;
};

class prompt_path {
  public:
    prompt_path();
    ~prompt_path();
    prompt_path(const prompt_path &)             = delete;
    prompt_path & operator=(const prompt_path &) = delete;

    // the arenas, replicas (filled now, card to card), copy streams and events; p[c] is the target's pass on card c
    // (its stream is the compute stream), d[c] its draft (null: no draft). Synchronous.
    bool init(model & m, pass_card * const p[2], draft_card * const d[2], std::string & err);

    // a run's start, both cards drained: the predecessor row (the draft's retained carry, both halves) and PLE's full
    // history from the committed version (cur: the cards' committed slot). Synchronous.
    bool run_start(uint32_t cur, std::string & err);

    // enqueue one chunk on both cards (no wait); the caller joins both compute streams, then checks errors
    bool chunk(const prompt_chunk & c, std::string & err);

    // after a clean join: both cards' prompt error words and the selection's and EP1's words; cleared after reading.
    // A set word fails the chunk (the caller publishes nothing).
    bool check_errors(std::string & err);

    // after a clean join and the publication: the retained carry (the chunk's last row's own halves) and the
    // predecessor row for the next chunk; enqueued on the compute streams
    bool publish(const prompt_chunk & c, std::string & err);

    // the last row's own half of the final streams into the pass's layer-47 boundary row 0 (hpost = hx = 0) for the
    // decode head (the caller runs it)
    bool head_input(const prompt_chunk & c, std::string & err);

    // TEST ONLY: card r's prompt error word set during the next chunk (a failed chunk's invalidation)
    void test_fail_next(int32_t card) { test_fail_ = card; }

    // TEST ONLY (prefill-test dry, PB1's determinism follow-up): every arena byte set to 0xFF (a NaN in F32, FP16
    // and BF16) and the error words cleared, so a run that reads arena data an earlier run left shows it even on zero
    // weights. Synchronous.
    bool test_poison(std::string & err);

    // TEST ONLY: the device final streams of chunk row r on card c (the draft check's hidden rows)
    const float * streams_row(int32_t card, int32_t r) const;
    const float * pred_row(int32_t card) const;

    // TEST ONLY (prefill-test vcheck, the PB1 review's V check): when set, each draft half first overwrites its rows of
    // the final streams with rows [pos0, pos0 + n) of host (pinned, [n_rows][4][2560] F32 by absolute position), in
    // stream order, so the bulk draft span reads given hidden rows; null: none
    void test_hidden(const float * host, int64_t n_rows) {
        test_h_      = host;
        test_h_rows_ = n_rows;
    }

    // TEST ONLY (SP2 Amendment 1, vcheck's sparse form): when set, each card's draft half copies its FP16 mixed rows
    // (the K, V and index-key products' operand) into rows [pos0, pos0 + n) of host[card] (pinned, [n_rows][2560]
    // FP16 by absolute position); null: none
    void test_mixed(uint16_t * host0, uint16_t * host1, int64_t n_rows) {
        test_m_[0]   = host0;
        test_m_[1]   = host1;
        test_m_rows_ = n_rows;
    }

    // TEST ONLY (SP2 Amendment 1 item 7, vcheck's localization; with test_mixed's rows): when set, each card's draft
    // half also copies its eh_proj output (the split HC's streams, [n_rows][4][1280] F32, this card's features) and
    // its HC low ([n_rows][320] F32) by absolute position; null: none
    void test_stages(float * eh0, float * eh1, float * low0, float * low1) {
        test_eh_[0]  = eh0;
        test_eh_[1]  = eh1;
        test_low_[0] = low0;
        test_low_[1] = low1;
    }

    void set_hc_dump(const hc_dump_spec & s) { dump_ = s; }

    const std::array<int64_t, n_psites> & site_counts() const { return sites_; }

    void clear_site_counts() { sites_.fill(0); }

  private:
    struct card_arena;
    struct link;
    bool fill_replicas(std::string & err);
    bool layer_half(const prompt_chunk & c, int32_t layer, int32_t sub, int32_t h, int32_t phase, std::string & err);
    bool draft_half(const prompt_chunk & c, int32_t h, int32_t phase, std::string & err);
    bool dump_hc(const prompt_chunk & c, int32_t layer, int32_t sub, int32_t h, std::string & err);

    void count(psite s) { ++sites_[static_cast<size_t>(s)]; }

    model *                       m_       = nullptr;
    pass_card *                   p_[2]    = { nullptr, nullptr };
    draft_card *                  d_[2]    = { nullptr, nullptr };
    void *                        copy_[2] = { nullptr, nullptr };
    std::vector<card_arena>       a_;
    std::vector<link>             links_;
    std::array<int64_t, n_psites> sites_{};
    int32_t                       hist_cur_  = 0;  // which of the two PLE full histories holds the latest
    int32_t                       test_fail_ = -1;
    const float *                 test_h_      = nullptr;
    int64_t                       test_h_rows_ = 0;
    uint16_t *                    test_m_[2]   = { nullptr, nullptr };
    int64_t                       test_m_rows_ = 0;
    float *                       test_eh_[2]  = { nullptr, nullptr };
    float *                       test_low_[2] = { nullptr, nullptr };
    hc_dump_spec                  dump_;
};

}  // namespace fn::model
