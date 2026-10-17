#pragma once

// PF1: bulk prefill's driver on the slot's target (FC-PF1, FC-PF2, FC-PF3's driver side, FC-ST1 and FC-ST3), Stage 1.
//
// Chunks: the ledger's unprocessed rows in chunks of at most 3,072 rows that never cross an absolute multiple of 8,192
// (RS1 section 2); inside each 8,192 span the tiling starts at the span's start or at the slot's processed length,
// whichever is later, so re-prefilling from a checkpoint at a multiple of 8,192 repeats the uninterrupted partition.
// Each chunk runs as two halves of at most 1,536 rows (FC-PF3). PB1: the bulk path (fn/model/prompt.h) runs each
// chunk sublayer-major on both cards, joins both cards at the chunk's end, checks the prompt's error words, then
// publishes the chunk (target::prompt_commit: cur by the chunk's halves, N += rows), the retained carry and the next
// chunk's predecessor; the draft is caught up inside the chunk; after the last chunk the decode head at B = 1 retains
// the last row's logits (target::prompt_head). A chunk that fails publishes nothing. The forced path below stays
// selectable (set_bulk(false)) as prefill-test's reference. PF1 Stage 1's substitute behind each half is the decode
// pass in B = 4 slices (the last takes the remainder), each followed by the draft's catch-up of its rows (DR1 Stage C;
// zero chunks behind, sealed at the chunk's end); NC1 section 2.2 makes the slices batch-invariant, so the state after
// a prompt equals the decode path's bitwise whatever the partition. A PK family replaces the slices of a half; the
// driver's loop, events and staging stay. The halves' exchange runs inside the decode passes in Stage 1 (the rings);
// each half's end is an event on each card's stream, where Stage 2's copy-stream exchange will synchronize.
//
// Staging (FC-PLE1 at prompt rows): the next chunk's PLE rows are gathered on a host thread (PL1's prefetch and
// gather, in the decode rounds' groups of four, so the rows are bitwise the decode path's) into one of two pinned host
// slots while the current chunk computes, then copied on each card's copy stream into one of two device staging slots;
// each round copies its rows from there into the pass's upload slot. Reuse rules, kept apart:
//   a pinned host slot is refilled only after its host-to-device copies completed (an event on each copy stream,
//   waited on the host before the gather starts);
//   a device slot is overwritten only after its last GPU reader completed: the copy stream waits for the event the
//   compute stream records after the chunk's last round (FC-D5), not for the copy's completion.
//
// Checkpoints (FC-ST1): at each absolute multiple of 8,192 a chunk ends on and at the prompt's end, at most 8 a slot
// (the oldest written dropped first), each the GDN states and histories and the PLE history at version 0 and the
// draft's retained carry, copied on each card's stream after the chunk's last catch-up. KV, index keys and summaries
// are not copied: a restore shortens their valid lengths. Each checkpoint carries a hash of its token prefix.
//
// Restore (FC-PF1, FC-ST3): a prompt that keeps n_reuse tokens of the slot and is not a pure extension restores the
// latest checkpoint at or before n_reuse whose token prefix matches, drops the later ones and re-prefills from it;
// with none, the slot is reset and the prompt re-prefilled from token 0.
//
// The handoff (FC-PF2): after the last chunk N = L = n, U = 0, O = A = 0, the last row's logits retained (the target
// round's) and the draft caught up to N with its retained carry the last row's; the bootstrap round follows.

#include "fn/model/model.h"
#include "fn/model/prompt.h"
#include "fn/runtime/target.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fn::runtime {

constexpr int64_t pf_span = 8192;       // no chunk crosses an absolute multiple; checkpoints sit on them
// a chunk: two halves (FC-PF3). PB3: a whole 8,192 span a chunk (was 3,072), so each chunk's fixed cost (its sublayers'
// weight conversions, its expert weight passes, the pipeline's fill and drain, the join) is spread over more rows
constexpr int32_t pf_chunk_rows = 8192;
constexpr int32_t pf_half_rows = 4096;
constexpr int32_t pf_split_rows = 1536; // a chunk of at most this many rows runs as one half (as before PB3)
constexpr int32_t pf_slice_rows = 4; // Stage 1's substitute: decode passes of B <= 4
constexpr int32_t pf_max_checkpoints = 8;

struct pf_chunk {
    int64_t start = 0;
    int32_t rows = 0;
    int32_t half[2] = {0, 0}; // half[1] = 0 for a chunk of at most pf_split_rows rows
};

// the chunks over positions [from, to)
std::vector<pf_chunk> plan_chunks(int64_t from, int64_t to);

// a checkpoint is taken at pos (a chunk's end) when pos is a multiple of 8,192 or the prompt's end
bool checkpoint_due(int64_t pos, int64_t prompt_end);

// FNV-1a over tokens [0, n): a checkpoint's token prefix
uint64_t token_key(const int32_t * tok, int64_t n);

struct pf_checkpoint {
    int64_t pos = -1; // -1: empty
    uint64_t key = 0; // token_key of the prefix
    uint64_t age = 0; // write order
};

// the slot's checkpoint bookkeeping, host only
class checkpoint_book {
  public:
    // the slot a checkpoint at pos is written into: the one already at pos, else an empty one, else the oldest written
    int32_t slot_for(int64_t pos) const;
    void record(int32_t slot, int64_t pos, uint64_t key);
    // the latest checkpoint at or before n, -1 when none
    int32_t latest(int64_t n) const;
    void drop(int32_t slot);
    void drop_after(int64_t n); // every checkpoint past n
    void clear();
    const pf_checkpoint & at(int32_t slot) const { return e_[slot]; }
    int32_t count() const;

  private:
    pf_checkpoint e_[pf_max_checkpoints];
    uint64_t next_age_ = 1;
};

// the latest checkpoint at or before min(n_reuse, |all| - 1) whose token prefix is all's (mismatches dropped), -1: none
int32_t choose_checkpoint(checkpoint_book & book, const std::vector<int32_t> & all, int64_t n_reuse);

// one chunk as the driver ran it
struct pf_chunk_stats {
    pf_chunk c;
    double gather_s = 0;     // the host gather of its rows (indices, prefetch, gather), on the staging thread
    double stall_s = 0;      // at its end, the main thread's wait for the next chunk's gather (0: staged under it)
    double compute_s = 0;    // its rounds and catch-ups, host clock
    int32_t checkpoint = -1; // the checkpoint slot written at its end, -1: none
};

// BG1: called at the end of every chunk but the last with (from, done, to); false stops the run there
using prefill_progress = std::function<bool(int64_t from, int64_t done, int64_t total)>;

struct pf_run_stats {
    int64_t from = 0, to = 0;
    int64_t stopped = -1; // BG1: the chunk end where the progress callback stopped the run, -1: not stopped
    int32_t rounds = 0, catchups = 0, checkpoints = 0;
    double wall_s = 0;
    std::vector<pf_chunk_stats> chunks;
    // PB1: the path (bulk, or the forced reference) and its host completion markers, seconds from the run's start
    // (steady clock): the last chunk's join (its target and draft work, one join), and decode-ready (the head's join
    // and the handoff check); -1 where not reached
    bool                                  bulk        = false;
    double                                last_join_s = -1, ready_s = -1;
    std::chrono::steady_clock::time_point t0;  // the run's start
};

class prefiller {
  public:
    prefiller() = default;
    ~prefiller();
    prefiller(const prefiller &) = delete;
    prefiller & operator=(const prefiller &) = delete;

    // the reserved checkpoint and staging classes, two pinned host slots, a copy stream and events a card
    bool init(target & t, model::model & m, std::string & err);

    // a prompt all that keeps the slot's first n_reuse tokens and is not a pure extension: restores the latest
    // matching checkpoint at or before n_reuse (from: its position) or resets the slot (from = 0); the ledger then
    // holds all with from processed. n_reuse = 0 always resets.
    bool begin(target & t, const std::vector<int32_t> & all, int64_t n_reuse, int64_t & from, std::string & err);
    // the ledger's unprocessed rows in chunks, with checkpoints, to the handoff (above); st optional. progress
    // (optional, BG1) runs after every chunk but the last; when it returns false the run stops there with the
    // chunk's work complete on both cards and its checkpoint (if due) written, st->stopped set and false returned
    bool run(target & t, pf_run_stats * st, std::string & err, const prefill_progress & progress = {});
    // every checkpoint forgotten (a failure left the slot's state unknown)
    void drop_all() { book_.clear(); }

    // PB1: the bulk path (the default) or PF1 Stage 1's forced B <= 4 slices (prefill-test's reference); the forced path
    // is kept selectable for the comparisons (PB1 item 9)
    void set_bulk(bool on) { bulk_ = on; }

    bool bulk() const { return bulk_; }

    model::prompt_path *    prompt() { return pp_.get(); }
    const checkpoint_book & book() const { return book_; }

    // TEST ONLY (prefill-test): a checkpoint of the slot's state at N now (synchronous)
    bool checkpoint_now(target & t, std::string & err);
    // PL1's gather over rows [start, start + rows) of tok in the decode rounds' groups of four (staging's form)
    static bool gather_rows(const ple_params & p, const uint8_t * table, const std::vector<int32_t> & tok,
                            int64_t start, int32_t rows, float * e, std::string & err);

  private:
    bool run_forced(target &                      t,
                    pf_run_stats &                s,
                    const std::vector<pf_chunk> & plan,
                    const prefill_progress &      progress,
                    std::string &                 err);
    bool run_bulk(target &                              t,
                  pf_run_stats &                        s,
                  const std::vector<pf_chunk> &         plan,
                  const prefill_progress &              progress,
                  std::chrono::steady_clock::time_point t0,
                  std::string &                         err);
    bool save(target & t, int32_t slot, std::string & err);
    bool load(target & t, int32_t slot, std::string & err);
    bool upload(int32_t s, int32_t rows, std::string & err);
    void start_gather(const ple_params & p, int32_t s, const pf_chunk & c);
    void join();

    model::model * m_ = nullptr;
    checkpoint_book book_;
    uint8_t * ck_[pf_max_checkpoints][2] = {}; // [checkpoint][card] in the reserved class
    int64_t ck_bytes_ = 0;                     // one checkpoint on one card (model::checkpoint_bytes)
    float * dev_[2][2] = {};                   // [slot][card] device staging
    float * host_[2] = {nullptr, nullptr};     // pinned [slot][3072][2560]
    void * copy_[2] = {nullptr, nullptr};      // a copy stream a card
    void * uploaded_[2][2] = {};               // [slot][card] its host-to-device copy completed
    void * read_[2][2] = {};                   // [slot][card] its chunk's last round completed (the last reader)
    void * half_[2][2] = {};                   // [half][card] the half's last round completed
    std::vector<int32_t> tok_;                 // the run's tokens, read by the staging thread
    std::thread gather_;                       // the staging thread (CPU only: no CUDA call)
    std::atomic<bool> gathered_{false};
    bool gather_ok_ = false;
    double gather_s_ = 0;
    std::string gather_err_;
    bool                 bulk_ = true;
    int64_t              seq_  = 0;  // the bulk path's chunks since init (prompt_chunk::seq)
    std::unique_ptr<model::prompt_path> pp_;
};

} // namespace fn::runtime
