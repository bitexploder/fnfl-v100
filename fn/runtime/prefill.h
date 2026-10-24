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
// (the oldest written dropped first), each the GDN states (ME11: the selector's bank and its live records, restored
// into bank 0 and set 0 with the same count) and histories and the PLE history at version 0 and the
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
#include "fn/runtime/prompt_plan.h"
#include "fn/runtime/target.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fn::runtime {

// the chunk geometry (pf_span, pf_chunk_rows, pf_half_rows, pf_split_rows, pf_slice_rows, pf_chunk, plan_chunks,
// pf_max_half, pf_short_read, pf_short_rows, checkpoint_due): fn/runtime/prompt_plan.h (ME4)
constexpr int32_t pf_max_checkpoints = 8;

// ME3: the calling thread's affinity to cpus (empty or !apply: left as inherited), read back as a mask of CPUs 0..63
uint64_t pf_helper_affinity(const std::vector<int32_t> & cpus, bool apply);

// FNV-1a over tokens [0, n): a checkpoint's token prefix
uint64_t token_key(const int32_t * tok, int64_t n);

struct pf_checkpoint {
    int64_t pos = -1;  // -1: empty
    uint64_t key = 0;  // token_key of the prefix
    uint64_t age = 0;  // write order
    int32_t owner = 0; // MU1: the engine slot whose state it holds (a checkpoint restores its owner only)
};

// the checkpoint bookkeeping, host only. MU1: one book for every engine slot, each entry owned by one; an entry is
// written into (slot_for) as before whatever its owner: the one the owner already has at pos, else an empty one, else
// the oldest written. Every query and drop names its owner; with one slot (owner 0) it is the one-slot book.
class checkpoint_book {
  public:
    // the entry a checkpoint at pos is written into: the owner's one already at pos, else an empty one, else the
    // oldest written (any owner's)
    int32_t slot_for(int64_t pos, int32_t owner = 0) const;
    void record(int32_t slot, int64_t pos, uint64_t key, int32_t owner = 0);
    // the owner's latest checkpoint at or before n, -1 when none
    int32_t latest(int64_t n, int32_t owner = 0) const;
    void drop(int32_t slot);
    void drop_after(int64_t n, int32_t owner = 0); // every one of the owner's checkpoints past n
    void clear();                                  // every entry
    void clear_owner(int32_t owner);               // MU1: every entry of the owner
    const pf_checkpoint & at(int32_t slot) const { return e_[slot]; }
    int32_t count() const;                    // every entry
    int32_t count_owner(int32_t owner) const; // MU1

  private:
    pf_checkpoint e_[pf_max_checkpoints];
    uint64_t next_age_ = 1;
};

// the owner's latest checkpoint at or before min(n_reuse, |all| - 1) whose token prefix is all's (mismatches
// dropped), -1: none
int32_t choose_checkpoint(checkpoint_book & book, const std::vector<int32_t> & all, int64_t n_reuse, int32_t owner = 0);

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
    // ME3: a known-token short read (target windows, no loan); the arena's layout and the loan's spans (bytes the
    // prompt writes in the lent region; -1: the whole region, 0: nothing lent); the staging thread's affinity read
    // back (CPUs 0..63; 0: no staging thread ran)
    bool     short_read = false;
    int32_t  layout_rows = 0, layout_half = 0;
    int64_t  loan_span_bytes = 0;
    uint64_t helper_mask = 0;
};

// MU1: a slot's resumable read on the multi-slot engine (the engine's slot holds it from its read's begin to its end):
// its tokens and chunk plan (PF1's, unchanged), the chunk in progress and its next layer group, and the counters the
// scheduler reads. A chunk runs in layer groups of `layers` layers (fixed when it starts); between two of its steps the
// chunk is suspended with every operand in its lane (fn/model/prompt.h chunk_layers), and other slots' rounds run.
struct pf_job {
    bool live = false; // begun and not ended (done, cancelled or failed)
    int64_t id = 0;    // the job's number (staging tags, lane continuation)
    int32_t slot = -1;
    std::vector<int32_t> tok; // the read's tokens (the slot's whole ledger)
    std::vector<pf_chunk> plan;
    int64_t from = 0, to = 0;
    size_t chunk = 0;   // the chunk in progress, or the next
    int32_t group = -1; // the chunk's next layer group; -1: no chunk in progress
    int32_t groups = 0; // the chunk's groups
    int32_t layers = 2; // its layers a group (set when the chunk starts)
    // FMU1: the chunk's end (layer 48's last phase and the draft's halves) runs as its own group after the layer groups
    // (writers present when it started), and its finish (errors, commit, publication, checkpoint) as its own step
    bool end_group = false;
    int32_t lane = -1;  // its lane (0: the main arena, 1: the other-reader lane)
    int32_t stage = -1; // its PLE staging slot (lane 0)
    model::prompt_chunk pc;
    bool short_read = false; // MXMU1: ME3's known-token short read (with the expert service): one step, no chunk
    bool cancel = false;   // BG1: stop at the chunk's end (or before the next chunk starts)
    int64_t committed = 0; // the prompt rows the slot's state holds (published chunks; from at the begin)
    // counters: groups run, chunks and checkpoints done, the job's host service time (its steps, synchronized)
    int64_t groups_done = 0;
    int32_t chunks_done = 0, checkpoints = 0, steps = 0;
    double service_s = 0, longest_step_s = 0, last_group_s = -1;
    pf_run_stats stats;
};

enum class pf_step : int32_t { yielded, chunk_done, done, cancelled, error };

// FMU1: how a read step runs (resident TP; with the expert service every step is MU1's shared form). A chunk's geometry
// is fixed when it starts:
//   shared:  layers groups of `layers` layers, the chunk's end with its last group (MU1);
//   writers: 1-layer groups, the chunk's end its own group, then its finish its own step (writers wait one layer);
//   lone:    the chunk in one group (run_bulk's kernel order and one join), its finish in the same step; a chunk that
//            started in another mode runs all its remaining groups and its finish in the step
enum class pf_mode : int32_t { shared, writers, lone };

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
    // every checkpoint forgotten (a failure left the slot's state unknown); MU1: the slot's (owner's) on a multi-slot
    // engine
    void drop_all() { book_.clear(); }
    void drop_owner(int32_t owner) { book_.clear_owner(owner); }

    // ---- MU1: the multi-slot engine's resumable reads ----
    // the other-reader lane: its prompt path on the lane's arena ("other-reader arena (MU1)"), laid out for chunks of
    // at most fn::model::lane_rows rows, and its staging; t is any slot's target (the lane binds each chunk's slot)
    bool init_lane(target & t, std::string & err);
    // the largest chunk a lane runs (0: no such lane)
    int32_t lane_rows(int32_t lane) const;
    // the slot whose chunk is in progress in a lane, -1: none
    int32_t lane_owner(int32_t lane) const { return lane < 0 || lane > 1 ? -1 : busy_[lane]; }
    // a job's next step on its slot's target t: start its next chunk if none is in progress (in lane 0 when free,
    // else in lane 1 when the chunk fits; an error when neither), then up to n_groups layer groups of it (the
    // group's layers set when the chunk starts), then a wait for both cards (and the lane's copy streams): the step
    // returns with its work complete. A chunk's last group is followed by its end (errors, commit, publication, the
    // checkpoint, the next chunk's staging), and the read's last chunk by the head (FC-PF2). cancelled: the job had
    // cancel set at a chunk's end or before its next chunk started (BG1: completed chunks and their checkpoints kept)
    pf_step job_step(target & t, pf_job & j, int32_t n_groups, int32_t layers, std::string & err,
                     pf_mode mode = pf_mode::shared);
    // FMU1 (P, resident TP): the staging of the reader whose turn follows the current one. now: the job whose lane-0
    // turn is in progress or starts next (null: none), its staged chunk kept; next: the job after it, whose next chunk
    // (after the one in progress) is gathered on the staging thread into a staging slot neither holds, then uploaded
    // and tagged by a later call. Never waits: a running gather, or a pinned slot whose upload has not completed,
    // leaves the work to a later call. Chunks of the next job start their own next chunk's gather only while it is the
    // job to follow (no call yet: always, PF1's pipeline)
    bool prefetch(target & t, const pf_job * now, const pf_job * next, std::string & err);
    // FMU1: the job's staging forgotten before its tokens change or after it ends: a gather reading its tokens
    // joined, its tags dropped
    void forget_job(int64_t id);
    // FMU1 (TEST ONLY, state-test): the staging slot holding the job's chunk (-1: none), and the running or finished
    // gather's job (-1: none)
    int32_t staged(int64_t id, int64_t chunk) const;
    int64_t gather_job() const { return gather_tag_job_; }
    // the job ended without its chunk's end (a failure): its lane and staging released, both cards drained; MXMU1: the
    // prompt loan its chunk held returned
    void job_abandon(target & t, pf_job & j);
    // MXMU1: a read's plan as run() makes it: PF1's chunks (MX3: with the expert service, chunks of at most 4,096 rows
    // as one half) and whether it is ME3's known-token short read
    void job_plan(int64_t from, int64_t to, bool cached, std::vector<pf_chunk> & plan, bool & short_read) const;
    // MXMU1: the job whose chunk holds the prompt loan (the expert service's lent region and staging), -1: none
    int64_t loan_job() const { return held_job_; }

    // PB1: the bulk path (the default) or PF1 Stage 1's forced B <= 4 slices (prefill-test's reference); the forced path
    // is kept selectable for the comparisons (PB1 item 9)
    void set_bulk(bool on) { bulk_ = on; }

    bool bulk() const { return bulk_; }

    // ME3 (fn/cache_settings.h): with an expert service, a prompt of at most short_read rows (0: none) reads through
    // the target's windows without a loan; bulk chunks of at most chunk_rows rows. set_legacy: ME3's base arm (expires
    // at ME3's disposition): ME2b's whole loan and the staging thread's inherited mask. loan_check: the startup half of
    // the loan's one calculation (the largest chunk's spans through the service; nothing lent: nothing to check)
    bool set_options(int32_t short_read, int32_t chunk_rows, std::string & err);
    void set_legacy(bool on) { legacy_ = on; }
    bool loan_check(std::string & err);
    int32_t short_read() const { return short_read_; }
    int32_t chunk_rows() const { return chunk_rows_; }

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
    bool run_short(target & t, pf_run_stats & s, std::string & err);
    // ME3: the loan's spans for a plan: its staging slots' rows (both cards) and the arena's layout (both cards)
    void loan_spans(const std::vector<pf_chunk> & plan, std::vector<model::byte_span> & out) const;
    bool save(target & t, int32_t slot, std::string & err);
    bool load(target & t, int32_t slot, std::string & err);
    bool upload(int32_t s, int32_t rows, std::string & err);
    void start_gather(const ple_params & p, int32_t s, const pf_chunk & c, const std::vector<int32_t> * tok = nullptr);
    void join();
    // MU1: the job's next chunk (FMU1: its geometry by mode)
    bool job_start(target & t, pf_job & j, int32_t layers, pf_mode mode, std::string & err);
    // FMU1: a finished gather (joined without waiting) uploaded and tagged; false only on a CUDA error
    bool pump(std::string & err);
    // FMU1: resident TP (the KV pool, two cards, no expert service): the paths FMU1 changes; else MU1's and MXMU1's
    bool resident(target & t) const;
    bool job_finish_chunk(target & t, pf_job & j, bool & done, std::string & err);
    // MXMU1 (with the expert service): the chunk's loan, taken before its first write into the lent region (the
    // staging slot d's upload, run_start), and returned after its last reader
    bool job_hold(target & t, model::prompt_path & pp, const pf_chunk & ch, int32_t d, pf_job & j, std::string & err);
    bool job_release(target & t, std::string & err);

    model::model * m_ = nullptr;
    checkpoint_book book_;
    uint8_t * ck_[pf_max_checkpoints][2] = {}; // [checkpoint][card] in the reserved class
    int64_t ck_bytes_ = 0;                     // one checkpoint on one card (model::checkpoint_bytes)
    int32_t ck_gdn_steps_[pf_max_checkpoints] = {}; // ME11: each checkpoint's live GDN records (the selector's k)
    float * dev_[2][2] = {};                   // [slot][card] device staging
    float * host_[2] = {nullptr, nullptr};     // pinned [slot][3072][2560]
    void * copy_[2] = {nullptr, nullptr};      // a copy stream a card
    void * uploaded_[2][2] = {};               // [slot][card] its host-to-device copy completed
    void * read_[2][2] = {};                   // [slot][card] its chunk's last round completed (the last reader)
    void * half_[2][2] = {};                   // [half][card] the half's last round completed
    std::vector<int32_t> tok_;                 // the run's tokens, read by the staging thread
    const std::vector<int32_t> * gather_tok_ = nullptr; // MU1: the tokens the running gather reads (tok_ or a job's)
    // MU1: the multi-slot reads. The lanes' prompt paths (0: pp_), the slot with a chunk in progress in each (-1:
    // none), each lane's continuation (the job and position whose predecessor and PLE history its arena holds, so the
    // job's next chunk there needs no run_start), lane 0's staging tags (the job and chunk each device slot and the
    // running gather holds, -1: none), and lane 1's staging (one device slot a card, a pinned host slot)
    std::unique_ptr<model::prompt_path> pp2_;
    int32_t busy_[2] = {-1, -1};
    int64_t cont_job_[2] = {-1, -1}, cont_pos_[2] = {-1, -1};
    int64_t dev_tag_job_[2] = {-1, -1}, dev_tag_chunk_[2] = {-1, -1};
    int64_t gather_tag_job_ = -1, gather_tag_chunk_ = -1;
    int32_t gather_slot_ = -1;
    int32_t gather_rows_ = 0; // FMU1: the running gather's rows
    int32_t live_stage_ = -1; // FMU1: the lane-0 chunk in progress's staging slot (-1: none)
    int64_t want_job_ = -1;   // FMU1: the job to follow the current turn (prefetch's next; -1: no call yet)
    float * lane_dev_[2] = {nullptr, nullptr};
    float * lane_host_ = nullptr;
    int64_t held_job_ = -1;                    // MXMU1: the job whose chunk holds the prompt loan (loan_job)
    std::thread gather_;                       // the staging thread (CPU only: no CUDA call)
    std::vector<int32_t> helper_cpus_;         // ME3: its mask (the expert service's; empty: inherited)
    uint64_t gather_mask_ = 0;                 // ME3: its affinity read back (CPUs 0..63), the last chunk's
    bool legacy_ = false;                      // ME3's base arm (set_legacy)
    int32_t short_read_ = 0, chunk_rows_ = pf_chunk_rows; // ME3 (set_options)
    std::atomic<bool> gathered_{false};
    bool gather_ok_ = false;
    double gather_s_ = 0;
    std::string gather_err_;
    bool                 bulk_ = true;
    int64_t              seq_  = 0;  // the bulk path's chunks since init (prompt_chunk::seq)
    std::unique_ptr<model::prompt_path> pp_;
};

} // namespace fn::runtime
