#pragma once

// The engine seam: everything the server needs from FNFL. Only fn/glue/ and the one server file that drives slots include
// this header. It names no CUDA, ggml or llama type, so touching an engine header never rebuilds the server.

#include "cache_settings.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct fnfl_engine;

enum fnfl_status : int32_t {
    FNFL_OK = 0,
    FNFL_NOT_BUILT = 1, // the engine does not run this yet (FL2: prefill and step)
    FNFL_BAD_REQUEST = 2,
    FNFL_ERROR = 3,
    FNFL_CANCELLED = 4,  // BG1: the progress callback stopped a prefill at a chunk's end
    FNFL_NO_CONTEXT = 5, // MU1: the shared KV pool cannot hold the slot's cells (every other slot active)
};

// the server's FNFL settings (SV1; STANDARDS "Switches": the engine prints them at startup). The test switches name
// SV1 and end when SV1 merges with its checks recorded, unless its review keeps them as documented debug options.
struct fnfl_config {
    std::string draft_subset; // the draft's subset ID file (--fnfl-draft-subset)
    std::string ple_table;    // PN1: the PLE table's residency, lazy or full (--fnfl-ple-table); empty: lazy
    int32_t spec_d = 2;       // SP2: the drafts a speculative step proposes, 1..4 (--fnfl-spec-d)
    // SP2's switch (--fnfl-draft-attention dense|sparse; empty: dense): the draft's attention over its cells; it
    // expires at Fable's disposition of SP2
    std::string draft_attention;
    bool check_state = false;      // TEST (--fnfl-check-state): the shell compares fnfl_slot_tokens after every call
    std::vector<int32_t> test_eog; // TEST (--fnfl-test-eog ID): more end-of-generation IDs
    int32_t test_fail_step = 0;    // TEST (--fnfl-test-fail-step K): step K of the next request fails once; 0 off
    // ME1 (--fnfl-single-card N): both logical cards on device N with the expert cache; -1 off
    int32_t single_card = -1;
    int32_t expert_pool_mib = -1; // ME1 (--fnfl-expert-pool-mib): the cache's GPU pool; -1: free less the reserve
    int32_t expert_cpu = -1;      // ME1 (--fnfl-expert-cpu): the CPU expert workers; -1: physical cores less one
    bool cached_tp = false;       // ME1 (--fnfl-cached-tp): the two cards with the routed experts from the cache
    bool kv_host = false;         // ME1 (--fnfl-kv-host, one device): the host-backed KV tier (fn/model/kv_tier.h)
    int32_t device_cap_mib = -1;  // ME1 (--fnfl-device-cap-mib): a ceiling on the device's whole use; -1 none
    // ME2 (--fnfl-expert-profile PATH, --fnfl-expert-profile-save PATH, --fnfl-expert-swap-overlap 0|1; expire at
    // ME2's disposition): the cache's boot ranking (a STRP file; empty: expert order), where it saves the learned
    // ranking (when a prompt starts after decode rounds, and at exit), the swaps' overlap
    std::string expert_profile;
    std::string expert_profile_save;
    bool expert_swap_overlap = true;
    // ME3: the settings of fn/cache_settings.h as given (each server spelling, --fnfl-expert-<name> or
    // --fnfl-kv-<name>, and its text, checked by the argument parser), parsed again by fnfl_load as the driver parses
    // its own (--fnfl-expert-executor among them); --fnfl-expert-reserve-mib, --ex-reserve-mib's server form
    std::vector<std::pair<std::string, std::string>> settings;
    int32_t expert_reserve_mib = 768;
    // KV8 (--fnfl-kv-format bf16|q8; empty: the default, fnfl_kv_format): the target KV's format. q8 serves
    // resident TP on two cards only; with more than one slot the KV pool is then n_ctx cells (each slot's context
    // stays at most the training context)
    std::string kv_format;
    // TTTS1 (--fnfl-split-layer K, 2 .. 47; 0: one box): the split across two boxes, resident TP on each. This server
    // is the leader, the front (layers 0 .. K - 1, the embeddings), connecting to its follower, `fnfl-drive stage`
    // (the back: layers K .. 47, the head and the draft), at split_addr:split_port (--fnfl-split-addr IP,
    // --fnfl-split-port P). Up to eight slots in two pipelined groups. split_commit, split_digest (--fnfl-split-commit,
    // --fnfl-split-digest): X-HELLO's source commit and device-code digest, the launch script's
    int32_t                                          split_layer = 0;
    std::string                                      split_addr  = "10.25.0.2";
    int32_t                                          split_port  = 13371;
    std::string                                      split_commit;
    std::string                                      split_digest;
};

// KV8 Amendment 1 (operator, 2026-10-22: q8 the serving default): the target KV's format from c alone (its kv_format
// and its --fnfl-single-card, --fnfl-cached-tp and --fnfl-kv-host settings), no model or device touched. Not given: q8
// on resident TP on two cards (Config A), bf16 on the others (not a refusal). Given: that value, q8 refused on the
// others and any value but bf16 or q8 refused (false, err set). line: the startup line's text, "q8 (default)", "bf16
// (default: <config> has no q8)", "q8" or "bf16".
struct fnfl_kv_choice {
    bool q8 = false;
    std::string line;
};
bool fnfl_kv_format(const fnfl_config & c, fnfl_kv_choice & out, std::string & err);

// ME3: the settings fnfl_load takes from c (c.settings, then the scope check: a cache setting needs
// --fnfl-single-card or --fnfl-cached-tp, a K/V one --fnfl-kv-host); text: their read-back (startup's lines) and
// the request lines' JSON. No model or device is touched.
bool fnfl_settings(const fnfl_config & c, fn::settings::values & v, std::string & text, std::string & err);

// the launch parameters the engine reads (the server's arguments, already parsed)
struct fnfl_load_params {
    std::string model_path; // the first GGUF shard; the others are found by its split name
    std::string draft_path; // the MTP draft GGUF (required: speculation is the served path)
    int32_t n_ctx = 0;      // total context over all slots, 0 for the model's training context
    // slots: 1, or MU1's 2..4 on resident TP (no expert cache, no host KV), where n_ctx is the shared KV pool's cells
    // and every slot's logical context (a lone user has all of it); or MXMU1's 2 on one card with the host-backed KV,
    // each slot's own history of n_ctx cells in host memory (no pool: pool_cells 0)
    int32_t n_parallel = 1;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    std::vector<int32_t> eog_ids; // the vocabulary's end-of-generation IDs (the tokenizer's is_eog set)
    fnfl_config config;
    // TK1: the launch sampling until a request sets its own (the server's --temp, --top-k, --top-p, --min-p)
    float temperature = 1.0f;
    int32_t top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.05f;
};

// a request's sampling and token budget (SV1), set before its prefill
struct fnfl_sampling {
    uint64_t seed = 0; // with request 0: the same request and seed give the same tokens
    float temperature = 1.0f;
    int32_t top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.05f;
    int32_t max_tokens = -1; // the request's n_predict; -1: unlimited, which becomes the context room
    int32_t n_ctx = 0;       // the shell's slot context (the room is n_ctx less the tokens the slot holds)
};

// what the server reports and renders with, read from the GGUF on the CPU
struct fnfl_model_meta {
    std::string arch;
    std::string name;
    std::string chat_template;          // tokenizer.chat_template, empty for none
    std::string chat_template_tool_use; // tokenizer.chat_template.tool_use, empty for none
    std::string format_name;            // the file type, as llama.cpp names it
    int32_t n_ctx_train = 0;
    int32_t n_embd = 0;
    uint64_t n_params = 0; // over all shards
    uint64_t size = 0;     // tensor bytes over all shards

    int32_t n_ctx_slot = 0; // the engine's context per slot, from the launch parameters
    // MU1: the slots, and with more than one the shared KV pool's physical cells (every slot's logical context is
    // n_ctx_slot, the pool's whole capacity) and its page
    int32_t n_slots = 1;
    int64_t pool_cells = 0;
    int32_t pool_page_cells = 0;
};

// one step's result: the 1-5 accepted tokens (SP2), and whether the slot's generation ended (an end-of-generation
// token)
struct fnfl_step_result {
    int32_t tokens[5] = {};
    int32_t n_tokens = 0;
    bool end = false;
};

const char * fnfl_status_str(fnfl_status status);

// returns nullptr and sets err on failure; meta is filled on success
fnfl_engine * fnfl_load(const fnfl_load_params & params, fnfl_model_meta & meta, std::string & err);

// the request's options, between requests and before its prefill; refused (FNFL_BAD_REQUEST) with the reason in err
fnfl_status fnfl_set_sampling(fnfl_engine * engine, int32_t slot, const fnfl_sampling & params, std::string & err);

// BG1: a prefill's progress, called on the caller's thread at the end of every chunk but the last. from: where the
// prefill started (the kept or restored prefix); done: the prompt tokens the slot then holds; total: the whole prompt.
// Returning false stops the prefill there (FNFL_CANCELLED).
using fnfl_progress = bool (*)(void * user, int32_t from, int32_t done, int32_t total);

// a slot's new tokens, after the n_reuse tokens of its cached prefix. n_processed (optional, SV1): the tokens the engine
// processed: n_tokens for a pure extension, n_reuse + n_tokens - c when it restored the checkpoint at c and re-prefilled
// from it (PF1; c = 0: from token 0). n_tokens = 0 (n_reuse >= 1) processes the last kept token again (FC-PF1).
// progress (optional, BG1) is called after each chunk; when it stops the prefill the call returns FNFL_CANCELLED and
// n_processed is the number of prompt tokens the slot keeps (the next prefill restores a checkpoint at or below it)
fnfl_status fnfl_prefill(fnfl_engine * engine, int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                         int32_t * n_processed = nullptr, fnfl_progress progress = nullptr, void * user = nullptr);

fnfl_status fnfl_step(fnfl_engine * engine, int32_t slot, fnfl_step_result & result);

// FMU2: one step of each of n distinct slots (1..4) at once: the slots whose step is speculative at d = 4 take their
// verify rounds as one joint round when two or more do (resident TP), every other slot's step alone; each slot's
// tokens are those of its fnfl_step. results[k], status[k] and (on a failure) errors[k] are slots[k]'s. Returns
// FNFL_BAD_REQUEST for no engine or arrays, else FNFL_OK (each slot's outcome in status[k]).
fnfl_status fnfl_step_joint(fnfl_engine * engine, const int32_t * slots, int32_t n, fnfl_step_result * results,
                            fnfl_status * status, std::string * errors);

// TTTS1: the engine is the split's leader (fnfl_config::split_layer)
bool        fnfl_is_split(const fnfl_engine * engine);
// TTTS1 (the split): one turn of the two-group pipeline over the generating slots (1 .. 8, distinct): each idle group
// with members begins (its front rounds and its send to the follower), then the older group in flight ends (its
// answers applied). done[k]: slots[k]'s step finished in this call (results[k], status[k], errors[k] then its);
// false while its group is in flight. Every other call naming a slot whose group is in flight ends that group first.
// Returns FNFL_BAD_REQUEST for no split engine or arrays, else FNFL_OK
fnfl_status fnfl_step_pipe(fnfl_engine *      engine,
                           const int32_t *    slots,
                           int32_t            n,
                           fnfl_step_result * results,
                           fnfl_status *      status,
                           std::string *      errors,
                           bool *             done);

// keep the slot's first n_keep tokens (cancellation and prefix reuse)
fnfl_status fnfl_truncate(fnfl_engine * engine, int32_t slot, int32_t n_keep);

// the reason of the engine's last failed call (a failed step ends its request with it)
std::string fnfl_last_error(const fnfl_engine * engine);

// TEST (--fnfl-check-state): the tokens the slot's state holds and its pending token (-1: none)
fnfl_status fnfl_slot_tokens(const fnfl_engine * engine, int32_t slot, std::vector<int32_t> & tokens,
                             int32_t & pending);

void fnfl_release(fnfl_engine * engine);

// ---- MU1: a multi-slot engine's reads as jobs (the server's two-user scheduling, TU1..TU6's port) ----
// superfast's llama_decode_job_* applied to a slot's whole read: begin prepares it, step advances it by layer groups of
// its chunk in progress (FNFL's chunk: PF1's partition, the --parallel 1 path's), cancel stops it at a chunk's end.

enum fnfl_read_status : int32_t {
    FNFL_READ_YIELDED = 0,    // the chunk has more groups
    FNFL_READ_CHUNK_DONE = 1, // the chunk is joined, committed, published and checkpointed; the read has more chunks
    FNFL_READ_DONE = 2,       // the last chunk, the draft catch-up and the head are done: the slot can step
    FNFL_READ_CANCELLED = 3,  // BG1: stopped at a chunk's end (or before any chunk); completed chunks kept
    FNFL_READ_ERROR = 4,      // a step failed (fnfl_last_error); the slot re-reads from token 0 or a checkpoint
};

// FMU1: who else has model work when a read step runs (resident TP; elsewhere every step is shared): lone, nobody (the
// chunk in one step, PF1's kernel order); writers, generating slots (1-layer groups, the chunk's end and its finish
// steps of their own); shared, other readers only (groups of fnfl_set_read_group's layers). Fixed per chunk at its
// start
enum fnfl_read_mode : int32_t {
    FNFL_READ_SHARED = 0,
    FNFL_READ_WRITERS = 1,
    FNFL_READ_LONE = 2,
};

// a slot's read as the scheduler sees it
struct fnfl_read_report {
    bool live = false;     // prepared and not ended
    bool cancel = false;   // its cancellation is pending
    int32_t total = 0;     // the read's tokens (the whole prompt)
    int32_t from = 0;      // where it started (the kept or restored prefix)
    int32_t committed = 0; // the prompt tokens the slot's state holds now (completed chunks)
    int32_t chunk = 0, chunks = 0;
    int32_t next_rows = 0; // the rows of the chunk in progress or next
    bool in_chunk = false; // a chunk is in progress (its groups so far complete)
    int32_t lane = -1;     // its lane: 0 the main arena, 1 the other-reader lane
    int32_t group = 0, groups = 0, layers = 0;
    int64_t groups_done = 0; // groups submitted and completed (a step returns with its work complete)
    int32_t steps = 0, chunks_done = 0, checkpoints = 0;
    double service_ms = 0, longest_step_ms = 0; // host clock around the read's steps
    double last_group_ms = -1;                  // the last step's time a group, -1 at a chunk's start or end
};

// the shared KV pool
struct fnfl_pool_report {
    bool on = false;
    int32_t pages = 0, page_cells = 0, free_pages = 0, reclaims = 0;
    int64_t capacity_cells = 0;
    int32_t mapped[8]      = {};  // TTTS1: up to eight slots on the split
    bool    busy[8]        = {};
};

// a slot's read prepared: the input owned, the reuse decided and executed (a pure extension, the slot's latest
// matching checkpoint restored, or a reset), the chunk plan made and the pool's pages for the prompt and its next step
// mapped (an idle slot reclaimed when the pool is full). 0: prepared; 1: the context or the pool cannot hold it (the
// slot's state unchanged); -1: invalid (fnfl_last_error). n_processed (optional) as fnfl_prefill's
int32_t fnfl_read_begin(fnfl_engine * engine, int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                        int32_t * n_processed = nullptr);
// up to n_groups layer groups of the read's chunk in progress (or of its next chunk, started now in the main arena
// when free, else in the other-reader lane when it fits; fnfl_lane_rows), complete on both cards when it returns
fnfl_read_status fnfl_read_step(fnfl_engine * engine, int32_t slot, int32_t n_groups,
                                fnfl_read_mode mode = FNFL_READ_SHARED);
// FMU1 (resident TP): now_slot's read has the main lane's turn in progress or starting next, next_slot's read follows
// it (-1: none); next's next chunk is staged ahead on the staging thread. Never waits; false: a CUDA failure
bool fnfl_read_prefetch(fnfl_engine * engine, int32_t now_slot, int32_t next_slot);
// FMU1: resident TP (the KV pool on two cards, no expert cache, no host KV): the presence-based paths apply
bool fnfl_resident_tp(const fnfl_engine * engine);
// BG1: marks the read's cancellation and returns; its next steps finish the chunk in progress, then CANCELLED
void fnfl_read_cancel(fnfl_engine * engine, int32_t slot);
bool fnfl_read_info(const fnfl_engine * engine, int32_t slot, fnfl_read_report & info);
// the layers a group of the chunks that start after the call (1 .. 48)
bool fnfl_set_read_group(fnfl_engine * engine, int32_t n_layers);
// the largest chunk a lane runs (0 rows: no such lane) and the slot with a chunk in progress there (-1: none)
int32_t fnfl_lane_rows(const fnfl_engine * engine, int32_t lane);
int32_t fnfl_lane_owner(const fnfl_engine * engine, int32_t lane);
// the shell released the slot (its request ended or was cancelled): idle, its pages reclaimable
fnfl_status fnfl_release_slot(fnfl_engine * engine, int32_t slot);
// the tokens the engine holds for the slot (fewer than the shell's after a reclaim: the shell keeps that prefix)
int32_t fnfl_slot_held(const fnfl_engine * engine, int32_t slot);
bool fnfl_pool(const fnfl_engine * engine, fnfl_pool_report & out);
