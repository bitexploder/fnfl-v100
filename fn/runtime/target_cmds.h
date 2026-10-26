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

#include "fn/cache_settings.h"
#include "fn/kernels/samp/samp.h"
#include "fn/runtime/expert_cache.h"
#include "fn/runtime/ple_host.h"

#include <cstdint>
#include <memory>
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
    std::string joint_kl;    // FMS2 item 7 (joint-kl, rollback-test mu1 --joint-kl): the lanes' IDs (HX7's deep.ids)
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
    // MU1 (--slots N, model-dry): the slots the engine serves (1..4; above one: resident TP, the KV pool)
    int32_t n_slots = 1;
    // KV8 (--kv-format q8, global; the switch ends at KV8's disposition): the target KV in the 8-bit format
    // (state_params::kv_q8); pool_cells: a multi-slot load's KV pool cells (0: a slot's)
    bool kv_q8 = false;
    int32_t pool_cells = 0;
    // KV8 (model-test full with kv_q8: KV8's load 4): spec's C4a..C4f checks on this prompt after the q8 passes, on the
    // same loaded model (--spec-ids IDS, with --draft-attention and --decode-s); empty: none
    std::string spec_ids;
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
        int32_t           spec_d     = 0;  // TTTS1 (--spec-d inside a setting): the setting's d; 0: the process's
    };

    // TTTS1 (--split, anywhere after the command): the split's stage of this process: 0 one box, 1 the leader (the
    // front, layers 0 .. split_k - 1, connecting to split_addr), 2 the follower (fnfl-drive stage: the back, listening
    // on split_addr); X-HELLO's identities split_commit and split_digest (the launch script's git rev-parse and
    // cuobjdump md5); record_hash: each step's decision hash (spec run's rounds.csv); split_groups (joint-kl on the
    // split): the joint pass as 1 U4 group, as 2 pipelined U2 groups, or 0 both
    int32_t                  split_role = 0;
    int32_t                  split_k    = 27;
    std::string              split_addr = "10.25.0.2";
    int32_t                  split_port = 13371;
    std::string              split_commit, split_digest;
    bool                     record_hash  = false;
    int32_t                  split_groups = 0;
    std::vector<run_setting> run_settings;
    int32_t runs = 3;
    std::vector<int32_t> eog; // empty: the tokenizer's EOS (248046) alone
    int32_t spec_d = 2; // SP2 (--spec-d): the drafts a speculative step proposes, 1..4; expires at SP2's disposition
    // SP2 Amendment 3 (spec ab --arms D:FORM[,D:FORM..], FORM dense or sparse): K3's and the measured part's arms,
    // in order; empty: 2:dense,4:sparse; expires at SP2's disposition. SP4: an optional :EX per arm, sp2, sp4 (H1) or
    // h2 (H2), XB3: or xb3 (C1), the target's expert gate/up mapping (target::set_expert_map; sp2 unless given)
    std::string arms;
    // ME1 (--single-card N, global): both logical cards on CUDA device N with the routed experts through the expert
    // cache (fn/runtime/expert_cache.h); spec run and model-dry. --cached-tp 1: the two cards
    // (devices 0 and 1) with the routed experts through the cache. --ex-pool-mib (-1: the free memory less
    // --ex-reserve-mib), --ex-cpu (CPU product workers; -1: physical cores less one)
    int32_t single_card = -1;
    bool cached_tp = false;
    int64_t ex_pool_mib = -1;
    int64_t ex_reserve_mib = 768;
    int32_t ex_cpu = -1;
    // MEKV (--kv-host 1, with --single-card or --cached-tp): the host-backed KV tier (fn/model/kv_tier.h), the 16 GB
    // configuration's
    // KV; --device-cap-mib M: a ceiling on the device's whole use, the pool shrinking under it (the 16 GB plan on a
    // larger card: expert_cache_options::device_cap_bytes)
    bool kv_host = false;
    int64_t device_cap_mib = -1;
    // ME2 (--ex-profile PATH, --ex-profile-save PATH, --ex-swap-overlap 0|1; expire at ME2's disposition): the
    // cache's boot ranking (a STRP file; empty: expert order), where the learned ranking is saved, the swaps' overlap
    std::string ex_profile, ex_profile_save;
    bool ex_swap_overlap = true;
    // ME3 (--ex-executor graph|eager, global): the executor of spec run, model-dry and run-dry: the graphs (default) or
    // the eager paired enqueue (an explicit diagnostic and ME3's base arm), printed at startup
    bool ex_eager = false;
    // ME3 (global): the cache's and the K/V tier's settings (fn/cache_settings.h), ex_eager among them
    fn::settings::values cache;
    // spec run (ME2, --ex-kl IDS --ex-kl-ref DIR --ex-kl-spread JSON): NC1 section 5's teacher-forced KL check before
    // the requests, in normal mode then with every miss on the CPU, each from the profile state; the requests then
    // start from the profile state again
    std::string ex_kl, ex_kl_ref, ex_kl_spread;
    // spec run (MX7, --mx7-check IDS --mx7-shares LIST; TEST ONLY, expires at MX7's disposition): the prompt CPU
    // share's model checks (fn/runtime/mx7_check.h) in place of the requests
    std::string mx7_check, mx7_shares;
    int32_t mx7_dry_maxt = 0; // MX7 (run-dry only, --mx7-dry-maxt N): the share's candidates' pair bound (0: MAXT)
    bool cached() const { return single_card >= 0 || cached_tp; }
    std::string ex_check; // spec run (ME1, --ex-check IDS): the placement check's prompt before the requests
    std::string ex_ref;   // --ex-ref TOKENS: an earlier run's tokens of that prompt at T = 0 (information)
    // spec run (ME2, --next-ids FILE, repeatable): further prompts after PROMPT_IDS in the same process, each with every
    // setting and run, the engine's state (and the expert cache's) kept between them; expires at ME2's disposition
    std::vector<std::string> next_ids;
    // ME3 (--next-append FILE, repeatable, in order with --next-ids; kept): next_append[i] makes next_ids[i] an append
    // to the slot's state (its history, its pending token, then the file's IDs), required to be a pure extension.
    // --me3-arms base,final (ME3's comparison switch, driver only, expires at ME3's disposition): the whole sequence
    // once an arm, each from the boot state; base ME2b's mechanisms on the eager executor, final ME3's on the graphs
    std::vector<bool> next_append;
    // ME3 (--next-append-rows N FILE, ME3's cleanup check, expires at ME3's disposition): an append whose read is N
    // rows: N less the slot's pending token (when it has one) of FILE's first IDs, its processed rows and its path
    // (short at most the short-read threshold, else bulk) checked; next_rows[i] -1 for the others
    std::vector<int32_t> next_rows;
    std::string me3_arms;
    // ME4 (--me4-arms base,final, ME4's comparison switch, driver only, expires at ME4's disposition): as --me3-arms,
    // both arms on ME3's mechanisms and the graphs; base ME3's prompt staging (no ring, the parity buffers at the
    // zone's start, routed-rows; the CPU experts' reconstruction per miss, main's), final ME4's (the ring, the parity
    // buffers by usage; the native views). Not with --me3-arms
    std::string me4_arms;
    // ME4 (--append-close ID, the albums fixture's length-stop rule, driver only, expires at ME4's disposition): when
    // a request ended at its budget (its pending token a content token, not the turn's end), the next append reads
    // ID (the template's end of turn) after the pending token and before its recorded IDs, so the history it extends
    // is a closed turn: one more row than recorded, printed with each request's end. -1: off
    int32_t append_close = -1;
    // TEST (--me3-stall, Fable's load G ruling, driver only, expires at ME4's disposition): spec run's stall check
    // (me3_stall_check) in place of the ME3 checks and the requests, on a cached run
    bool me3_stall = false;
    // TEST (--me9-check, ME9, driver only, expires at ME9's disposition): spec run's admission check (me9_check) in
    // place of the ME3 checks and the requests, on a one-device cached run
    bool me9_check = false;
    // TEST (--ex-native-hold, ME4's gate, driver only, expires at ME4's disposition): a dry spec run holds the native
    // expert pages as the load will (expert_cache_options::native_hold), for the host-memory residency check
    bool ex_native_hold = false;
    // TEST (--me3-cancel, Fable's ruling on ME3's completeness gaps, driver only, expires at ME4's disposition): spec
    // run's cancellation check (me3_cancel_check) in place of the ME3 checks and the requests, on a cached run
    bool me3_cancel = false;
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
// MU1 (rollback-test IDS OUT mu1 | mu1-dry, fn/runtime/rollback_mu1_test.cpp): a read paused at every layer boundary
// with other slots' work between its groups against the same read straight through, on a four-slot engine; the
// short read out of turn, cancellation before, during and while suspended, and the resumed reads
int fnfl_rollback_mu1(const fnfl_target_args & a);

// TTTS1: rollback-test mu1's slot state hash of slot (the engine's own window's parts), as bytes: the split
// follower's answer to the leader's probe
namespace fn::runtime {
class engine;
}

bool fnfl_mu1_slot_hash(fn::runtime::engine & e, int32_t slot, std::vector<uint8_t> & out);

namespace fn::model {
class model;
}
namespace fn::runtime {
class target;
struct draft_options;
} // namespace fn::runtime
// FMS2 item 7 (fn/runtime/joint_kl.cpp): the 16-bit joint X-A / X-F judged by KL on one loaded model: four lanes of
// a.joint_kl alone and jointly (q8 KV, the switch on), and the lone pass in BF16 KV as the yardstick; PASS (exit 0)
// when the joint's KL p50, p99, max and top-1 differences are each at most the yardstick's and no error word is set.
// fnfl_joint_kl_on takes a model loaded with four slots of cells (rollback-test mu1's, after its PASS); fnfl_joint_kl
// loads one (a.mode dry: model-dry's zero weights)
// TTTS1: sp, the split's session (rollback-test mu1 --split's): the passes on the split, the follower rebuilt alike
struct fnfl_split;
int  fnfl_joint_kl_on(const fnfl_target_args &          a,
                      std::unique_ptr<fn::model::model> m,
                      int32_t                           cells,
                      fnfl_split *                      sp = nullptr);
// TTTS1: the model's state laid out again for slots slots of cells cells in q8 or BF16 KV (release, reserve, the EX1
// grids), the weights kept; true when the model's KV format is now kv_q8's
bool fnfl_lay_out(fn::model::model & m, int32_t cells, int32_t slots, bool kv_q8);
int fnfl_joint_kl(const fnfl_target_args & a);
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
// KV8: spec's checks on a model loaded by the caller (null: spec's own load)
int fnfl_spec_on(const fnfl_target_args & a, fn::model::model * loaded);
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
    int32_t              n_processed = 0; // ME3: the prefill's processed tokens
};

namespace fn::runtime::stage {
class leader;
}

// TTTS1: L (the split's leader, its front engine e) carries out the request's calls; null: e alone
bool fnfl_spec_request(fn::runtime::engine &                e,
                       const std::vector<int32_t> &         prompt,
                       int32_t                              n_reuse,
                       const fn::runtime::request_options & o,
                       fnfl_request_out &                   out,
                       fn::runtime::stage::leader *         L = nullptr);
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
// ME2: model-test's KL pass (NC1 section 5) on t: forced rounds, B cycling 1..4, over the first 4,096 IDs from empty
// state; KL(reference || t) at p % 8 == 7 and the top-1 differences against the reference (ref_dir's top.bin and
// logits.bin), each against 2x the yardstick (spread_json). Prints one line (label), writes OUT/kl-LABEL.tsv; pass
// set; false (printed) on a failure to run
bool fnfl_kl_check(fn::runtime::target & t, const std::string & ids_path, const std::string & ref_dir,
                   const std::string & spread_json, const std::string & out_dir, const std::string & label, bool & pass);
// ME1: the expert cache of a cached load (a.cached()), after every other allocation; dry: zero backing. Prints
// the memory plan; null (printed) on failure. ME3: before_pool, the graph executor's capture (service_hook)
std::unique_ptr<fn::runtime::expert_cache> fnfl_expert_cache(const fnfl_target_args & a, fn::model::model & m,
                                                             bool dry, const char * tool,
                                                             const fn::model::service_hook & before_pool = {});

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
