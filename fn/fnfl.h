#pragma once

// The engine seam: everything the server needs from FNFL. Only fn/glue/ and the one server file that drives slots include
// this header. It names no CUDA, ggml or llama type, so touching an engine header never rebuilds the server.

#include <cstdint>
#include <string>
#include <vector>

struct fnfl_engine;

enum fnfl_status : int32_t {
    FNFL_OK = 0,
    FNFL_NOT_BUILT = 1, // the engine does not run this yet (FL2: prefill and step)
    FNFL_BAD_REQUEST = 2,
    FNFL_ERROR = 3,
    FNFL_CANCELLED = 4, // BG1: the progress callback stopped a prefill at a chunk's end
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
};

// the launch parameters the engine reads (the server's arguments, already parsed)
struct fnfl_load_params {
    std::string model_path; // the first GGUF shard; the others are found by its split name
    std::string draft_path; // the MTP draft GGUF (required: speculation is the served path)
    int32_t n_ctx = 0;      // total context over all slots, 0 for the model's training context
    int32_t n_parallel = 1; // slots (FNFL serves one)
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

// keep the slot's first n_keep tokens (cancellation and prefix reuse)
fnfl_status fnfl_truncate(fnfl_engine * engine, int32_t slot, int32_t n_keep);

// the reason of the engine's last failed call (a failed step ends its request with it)
std::string fnfl_last_error(const fnfl_engine * engine);

// TEST (--fnfl-check-state): the tokens the slot's state holds and its pending token (-1: none)
fnfl_status fnfl_slot_tokens(const fnfl_engine * engine, int32_t slot, std::vector<int32_t> & tokens,
                             int32_t & pending);

void fnfl_release(fnfl_engine * engine);
