#pragma once

// fnfl-drive model-test, model-bench and generate (TG1): the whole target pass on both cards (fn/runtime/target.h).
//
// model-test: one model load. Forced rounds (FC-RD3) over the first N positions of the reference tokens (default
// 4,096) from empty state, B cycling 1, 2, 3, 4, with the real token IDs and no hidden-state forcing, three times: the
// cycled pass, the cycled pass again, and positions 0 .. 511 at B = 1. Checks (NC1 section 5):
//   distribution: KL(reference || FNFL) over the full vocabulary at the positions p % 8 == 7, p50, p99 and max each at
//            most 2x the same statistic of KL(decode mode || prompt mode) (ref-spread3.json);
//   top-1:   FNFL's greedy token (the round's argmax, an observation in a forced round) differs from the reference's
//            top-1 at no more than 2x the positions where prompt mode's does; each difference is listed with the
//            reference's margin;
//   determinism: the second pass's logits (both cards' shards) hash bitwise to the first's at every position;
//   batch invariance: the B = 1 pass's logits hash bitwise to the cycled pass's at positions 0 .. 511;
//   the ledger checks (FC-SLOT1), the error words of every round, both cards' tokens equal.
// It writes the hidden-state records (l_last of layers 11, 23, 35, 47 and the hc_mixed rows of layers 11 and 35) of the
// first pass for the offline statistics (information), and prints "model-test: PASS" with exit 0 when every check holds.
// Modes: san: 16 positions, both cards in phases (no graphs), state for 8,192 cells (Compute Sanitizer memcheck); race:
// the same with layers 8 .. 11 only (racecheck).
//
// model-bench: one model load. A forced pass over the first 4,096 tokens builds the real history; then the graphs at
// B = 1..4 at depth 4,096, 3 warm-up and 20 timed replays each, untraced; then a synthetic history (random K, V and
// index keys in their stored formats, summaries from QA1b's summary_update, random GDN and PLE state) and the same at
// 58,581, 131,072 and 196,608. Per replay: each card's event span, its pass (%globaltimer from after the start to the
// finish node, the commit included) and the start skew. Mode nsys: 131,072 cells at B = 3 only, between
// cudaProfilerStart and Stop; real (TR1) forces the real history at any DEPTH, from the file's first DEPTH + B tokens.
//
// generate: one model load. Forced rounds over the prompt at B = 4 (the last takes the remainder), the bootstrap
// (token zero from the retained logits), then 255 generate rounds at B = 1, each fed back through the embedding, PLE,
// the 48 layers, the head, greedy (or SM1's sampler, with the sampling options) and the commit; with the sampler the
// closing check compares the last decision's hash before the run returns. EOS is not a stop. Writes the tokens, the trace of every round, and
// the host and GPU times, with the process's page faults. PL1: optionally after evicting the PLE table (its page counts
// printed), with the run's inter-token and gather statistics, faults and storage reads.

#include "fn/kernels/samp/samp.h"
#include "fn/runtime/ple_host.h"

#include <cstdint>
#include <string>
#include <vector>

struct fnfl_target_args {
    std::string ref_dir;     // model-test: the decode-mode reference (raw logits.bin and top.bin of ref_dump3)
    std::string spread_json; // model-test: ref-spread3.json
    std::string ref_prompt;  // prefill-test model (PB1): TG1's prompt-mode dump (ref_dir: its decode-mode dump)
    std::string deep_ids;    // prefill-test model (PB1): the deep stage's prompt (empty: no deep stage)
    int32_t deep_restore = 0;  // the deep stage: 0, or first a request of this many tokens, then the prompt restoring
                               // its prompt-end checkpoint (the engine's restore and append)
    std::string append_ids;    // prefill-test model (PB3 closure): after the deep stage, the append point's prompt
    int32_t append_restore = 0; // its untimed prefix: a request of this many tokens, then the prompt restoring it
    std::string ids;         // the token IDs (model-test, model-bench: HX7's deep.ids; generate: the prompt)
    std::string out_dir;
    std::string mode; // model-test: full | san | race; model-bench: all | nsys
    // model-bench nsys (SF1): the traced point's depth and B (TG1's 131,072 and 3 when not given)
    int32_t bench_depth = 131072;
    int32_t bench_b = 3;
    bool bench_real = false; // model-bench nsys (TR1): a real history at any DEPTH; the file must hold DEPTH + B tokens
    std::string model_path = "/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
                             "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    std::string draft_path = "/home/jeremy/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
    int32_t n_pos = 4096;
    int32_t n_gen = 256;
    int32_t n_cells = 198720;
    bool evict_ple = false; // generate (PL1): the PLE table evicted before the run (page_cache.h), its counts printed
    // PN1: --ple-table, the loaded table's residency (refused with evict_ple in full: a pinned table stays resident)
    fn::runtime::ple_residency ple_table = fn::runtime::ple_residency::lazy;
    bool dry_paths = false; // generate (PL1): check every input and output path, print the plan and exit, no load
    // DR1 Stage C (fn/runtime/target.h draft_options): --draft-attention's value, required by draft-test and
    // draft-bench (a missing value is an error); model-dry runs the draft rounds when it is given
    std::string draft_attention;
    bool draft_attention_given = false;
    std::string draft_subset = "/home/jeremy/src/flashnext/work/DR1/stageC/subset/draft-subset.ids";
    std::string ref_draft; // draft-test: the reference dumps' directory (src/ref_dump_draft.cpp: r4k/, deep/)
    // generate (SM1): sample with these options (FC-DEC2); unset: V1's greedy as TG1 built it
    bool sample = false;
    fn::samp::options sampling;
    // spec (SP1): the decode time limit of the timed run (C4c), seconds
    double decode_s = 90.0;
    // spec run (SV1): each setting's requests (runs of each, from a fresh prefill), and the end-of-generation IDs
    struct run_setting {
        fn::samp::options sampling;
        int32_t max_tokens = 256;
    };
    std::vector<run_setting> run_settings;
    int32_t runs = 3;
    std::vector<int32_t> eog; // empty: the tokenizer's EOS (248046) alone
    int32_t spec_d = 2; // SP2 (--spec-d): the drafts a speculative step proposes, 1..4; expires at SP2's disposition
    // SP2 Amendment 3 (spec ab --arms D:FORM[,D:FORM..], FORM dense or sparse): K3's and the measured part's arms,
    // in order; empty: 2:dense,4:sparse; expires at SP2's disposition. SP4: an optional :EX per arm, sp2, sp4 (H1) or
    // h2 (H2), XB3: or xb3 (C1), the target's expert gate/up mapping (target::set_expert_map; sp2 unless given)
    std::string arms;
};

int fnfl_model_test(const fnfl_target_args & a);
int fnfl_model_bench(const fnfl_target_args & a);
// model-dry (test only, no model data): the loader's plan and allocation with zeroed weights, then forced rounds twice,
// a prompt, the bootstrap and generate rounds, and every B at the slot's last cells; mode graph or eager. Mode sample
// (SM1): sampled generate rounds through the round frame twice (bitwise), the forced decision-hash mismatch (the round
// commits nothing and the slot is invalidated) and the closing check
int fnfl_model_dry(const fnfl_target_args & a);
int fnfl_generate(const fnfl_target_args & a);
// rollback-test (SR1, fn/runtime/rollback_test.cpp): one model load; scripted commits of k of B rows, cuts within the
// last step and injected recoverable errors over 512 positions of the reference tokens, against a clean replay at B = 1.
// mode dry: model-dry's load instead (zero weights, no model data on the cards), a plumbing check before the load
int fnfl_rollback_test(const fnfl_target_args & a);

namespace fn::model {
class model;
}
namespace fn::runtime {
class target;
struct draft_options;
} // namespace fn::runtime
// rollback-test's state hashes (SR1), for spec's replay check: on both cards every GDN layer's state and convolution
// history (logical version 0), the PLE history, K and V of cells < N, the index keys of cells < N and the summaries of
// blocks < N / 4, in that order; false on a failed read or a committed slot that is not the ledger's
bool fnfl_state_hashes(fn::runtime::target & t, fn::model::model & m, uint64_t h[7]);
// SP1 (fn/runtime/spec_cmds.cpp). spec: one model load (mode dry: model-dry's zero weights, short segments; the C1
// plumbing check). Real sampled speculation on the prompt after forced-round prefill with the draft's catch-up (C4a):
// the T = 0 diagnostic against generate rounds (C4b), then at the command line's settings (TK1; default T 1, top-k 20,
// top-p 0.95, min-p 0.05), d = 2 and the seed: 8 recorded rounds checked by the CPU reference (C4d), the timed run
// to EOS or the time limit (C4c), 32 rounds against a clean forced replay bitwise (C4e) and the determinism prefix
// (C4f). Every segment starts from a fresh prefill. Writes the tokens, the rounds and the report into OUT.
int fnfl_spec(const fnfl_target_args & a);
// SV1, spec run (mode run; run-dry: zero weights, 13 prompt tokens, then the engine's prefix paths): the runtime's
// engine as the server drives it (fn/runtime/runtime.h), for each setting `runs` requests from a fresh prefill
// (set_sampling, truncate to 0, prefill, steps until an end of generation or max_tokens), printing each request's
// summary line and tokens. Writes run-<setting>-<run>-tokens.txt and run-summary.txt into OUT.
int  fnfl_spec_run(const fnfl_target_args & a);
// SP2, spec ab (mode ab; ab-dry: zero weights, the history cut to 2,100 tokens): one model load. Checks at d = 4 in
// the sparse form on the history's first 2,600 tokens: C4d's recorded rounds (K1) and C4e's replay with the draft
// index (K2), and the decode path's draft index against a bulk prefill's on the same tokens (K4); then the
// history's bulk prefill and its decode-ready state saved (HC8's), restored for every request:
// K3 (two short requests per arm, bitwise equal), then one continuation per arm (base: d = 2, dense; candidate:
// d = 4, sparse) of at most --max-tokens (<= 256). Writes ab-summary.txt, ab-rounds.csv and the tokens into OUT.
int  fnfl_spec_ab(const fnfl_target_args & a);

namespace fn::runtime {
class engine;
struct request_options;
}  // namespace fn::runtime

// SV1's request as spec run drives it (PB1: shared with prefill-test's deep stage): the options set, the slot truncated
// to n_reuse (0: a fresh prefill; else the next prefill keeps n_reuse tokens: a pure extension or a checkpoint
// restore, as the engine decides), the prompt's tokens after n_reuse prefilled, then steps until an end of generation
// or o.max_tokens. pre_s: the prefill call; gen_s: after it to the last step's return. false: e.error() says why
struct fnfl_request_out {
    std::vector<int32_t> toks;
    double               pre_s = 0, gen_s = 0;
};

bool fnfl_spec_request(fn::runtime::engine &                e,
                       const std::vector<int32_t> &         prompt,
                       int32_t                              n_reuse,
                       const fn::runtime::request_options & o,
                       fnfl_request_out &                   out);
// PF1, prefill-test (fn/runtime/prefill_cmds.cpp): bulk prefill's driver; PB1: the bulk path against the forced
// reference and the pinned engine. Mode plan: the chunk rules and the checkpoint bookkeeping, no device; dry and
// dry-eager: zero weights, 16,384 cells, the driver's paths on the target and (dry) the engine's prefill, truncate,
// history, phase line, cancellation and failure, and the headroom at 202,816 and 262,144 cells; model: one load, NC1
// section 6 on TG1's dumps (ref_dir, ref_prompt), the 20,000-token checks, then the deep stage (deep_ids). Prints
// "prefill-test: PASS" with exit 0 when every check holds; writes OUT/prefill-test.log
int fnfl_prefill_test(const fnfl_target_args & a);
// model-dry's load without model data (zero weights), for rollback-test's dry mode
bool fnfl_load_dry(const fnfl_target_args & a, int32_t n_cells, fn::model::model & m);
// the model load of model-test and model-bench (tool: the log's prefix), for draft-test and draft-bench
bool fnfl_load(const fnfl_target_args & a, int32_t n_cells, fn::model::model & m, const char * tool);
// model-bench's synthetic history (the draft's K and V too), for draft-bench
bool fnfl_fill_synthetic(fn::model::model & m, fn::runtime::target & t, int32_t depth);
// the draft settings of a and their startup line; false (printed) when the draft is asked for and not buildable
bool fnfl_draft_settings(const fnfl_target_args & a, bool required, const char * tool, fn::runtime::draft_options & o);

// DR1 Stage C (fn/runtime/draft_cmds.cpp).
// draft-test: one model load. The draft block against the pinned engine's reference dumps (src/ref_dump_draft.cpp) on
// two fixtures of HX7's deep.ids, r4k (4,096 tokens, every eighth position) and deep (16,384, the last 256): forced
// rounds with B cycling 1..4 as the reference's decode mode; after each batch the catch-up to every selected position
// in it, a draft step there (d = 3 where the position ends the batch: the reference's chain) and the catch-up to the
// batch's end. Checks: KL(decode || FNFL) over the subset, p50 / p99 / max each at most 2x KL(decode || prompt), and
// the top-1 differences at most 2x prompt's, per fixture; both cards' proposals equal (every round); r4k twice,
// bitwise; the catch-up's K / V cells and retained carry bitwise against a clean replay at B = 1 after every
// forced_k (B, k) commit of SR1's test path. Reports offline T = 0 acceptance (proposals against the target's
// later greedy tokens after r4k) and the teacher-forced acceptance of the reference's and FNFL's chains.
int fnfl_draft_test(const fnfl_target_args & a);
// draft-bench: one model load. The draft step at d = 1..3 and the catch-up at B_c = 1..4 at 4,096 (a real history)
// and 58,581, 131,072 and 196,608 (synthetic), 3 warm-up and 20 timed graph replays each, untraced: each card's device
// span (%globaltimer, start to finish), median and p90, with the SM clock read before each point.
int fnfl_draft_bench(const fnfl_target_args & a);
