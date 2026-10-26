// fnfl-drive: runs FNFL pieces with no server. HC1: the HC kernels' test and bench.
// QA1: the QSA kernels' test and bench. EX1: the routed-expert kernels' calibration, test and bench. HC2: the
// card-to-card protocol's test and the two-card HC sublayer's test and bench. EP1: the prompt expert kernels' test and
// bench. R1: the loader's load-plan (CPU) and load-check (both cards). R2: layer-test and layer-bench (two decode
// layers on both cards against the pinned engine). C2: FNFL's activation quantize kernel's test and bench. C1b: FNFL's
// weight decoders against the pinned engine's decode, and the synthetic-block generators' check. DP1/DP2: the dense
// products' test and bench. GD1: the GDN layer's math (gdn_core): its test and bench. E1: the token embedding, the
// host PLE gather and the PLE layer: their test and bench.
//
// usage: fnfl-drive act-test [DEVICE]   (C2: the quantize kernel bitwise against the CPU reference)
//        fnfl-drive act-bench [DEVICE] [all|acceptance]
//        fnfl-drive gd-test [DEVICE] [all|short|chain|map|inv|replay|san] [MODEL]   (GD1: DEVICE is the card; the
//            weights of layers 10 and 40 are read from MODEL's file, no model load; ME11's replay: the compact state
//            bitwise against the pre-ME11 kernel)
//        fnfl-drive gd-bench [DEVICE] [all|ncu|probe|diag]
//        fnfl-drive ple-test [all|san] [DEV0] [DEV1] [MODEL]   (E1: defaults all 0 1; DEV0 == DEV1: loopback on one
//            device, for Compute Sanitizer; layer 1's PLE tensors, token_embd rows, PLE table rows and the ple.* keys
//            are read from MODEL's files, no model load)
//        fnfl-drive ple-bench [all|ncu:LAUNCH:B|host:off|on:B] [DEV0] [DEV1] [MODEL] [TOKENS]   (E1 step 4; TOKENS: a
//            JSON list of token IDs for the host gather, default FL2's fl1-tokenize.out; PL1: host:<off|on>:<1|3>,
//            the host gather alone from an evicted table, no device)
//        fnfl-drive ple-pin-test [probe|unit] [DIR] [DEVICE]   (PN1: the PLE table's residency on a temporary file in
//            DIR, default work/PN1; no model load)
//        fnfl-drive hc-test|hc-bench [DEVICE] [ROWS]   (hc-test only: ROWS = row counts, for example 1,3)
//        fnfl-drive qa-test [DEVICE] [DEPTHS|all] [ROWS|all] [FIXTURE]   (lists like 65536 or 1,3; FIXTURE: the HX7 case)
//        fnfl-drive qa-bench [DEVICE] [all|acceptance] [FIXTURE]
//        fnfl-drive qa2-test [DEVICE] [all|san] [MODEL] [FIXTURE]   (QA2: attend against its FP64 reference; layer 11's
//            norms and the rope and attention keys read from MODEL's file, no model load; FIXTURE: QA1's HX7 case)
//        fnfl-drive qa2-bench [DEVICE] [all|ncu] [FIXTURE]   (QA2 steps 4-5)
//        fnfl-drive kv-tier-test [DEVICE] [all|san]   (ME1: the host-backed KV tier against a resident cache, bitwise;
//            synthetic, no model data; san: 32,768 cells for Compute Sanitizer)
//        fnfl-drive draft-kernel-test [DEV0] [DEV1] [all|san]   (DR1 Stage C: the draft kernels against FP64)
//        fnfl-drive draft-kernel-bench [DEV0] [DEV1]   (DR1 Stage C: DA-TC at the four depths, X-DRAFT-IN on both)
//        fnfl-drive ex-calib [DEVICE]
//        fnfl-drive ex-prep [CACHE]   (CPU only: the synthetic expert weights)
//        fnfl-drive ex-cpu-test [CACHE]   (CPU only, ME2: the CPU miss path round trip, against FP64, a throughput line)
//        fnfl-drive ex-test [DEVICE] [ROWS] [FORMATS] [CACHE] [BASE]   (EX2: ROWS like 1,2,3,4; FORMATS all, san or
//            routed gate/up formats like iq2_xxs,iq3_s: every deployed instantiation with the shared expert; XB2:
//            BASE, 71b92a9's experts cubins, adds the xb3-bitwise and xb4-bitwise arms)
//        fnfl-drive ex-bench [DEVICE] [all|POINT] [CACHE] [HX7]   (EX2: every deployed launch warm and cold, HX7's
//            captured routing; HX7: the capture directory; XB2: xb2prof:POINTS, each point once a decode for ncu;
//            SP4 Amendment 1: sp4prof:POINTS:MAP, B = 5 on four decodes, cold, MAP 0 SP2 or 1 SP4, for ncu)
//        fnfl-drive ep-prep [REFS] [CACHE] [ROUTES] [all|q4]   (CPU only: ep-test's FP64 references; q4: ME1's)
//        fnfl-drive ep-test [DEVICE] [ROWS] [FORMATS] [full|san] [REFS] [CACHE] [ROUTES]   (ROWS like 128,1536)
//        fnfl-drive ep-bench [DEVICE] [all|ncu] [CACHE] [ROUTES]   (ROUTES: FN4's routes.txt)
//        fnfl-drive grid-test [HEADER]   (the grid tables against ggml-common.h at 66e11c3, default the pinned
//            tree's; FMS4: the IQ2 selector tables, and their recipe on device 0)
//        fnfl-drive decode-check DIR   (CPU only: fn/quant's decoders bitwise against work/C1b/ref's pinned dump)
//        fnfl-drive synth-check   (CPU only: fn/quant's synthetic-block generators)
//        fnfl-drive xchg-test [DEV0] [DEV1] [ROUNDS] [ROWS]   (defaults 0 1 10000 1,3)
//        fnfl-drive hc2-test [DEV0] [DEV1] [ROWS]   (default ROWS 1,2,3,4,5)
//            (DEV0 == DEV1: loopback on one device and one stream, sends before reads: for Compute Sanitizer)
//        fnfl-drive hcj-test [DEV0] [DEV1]   (FMU2: the joint sublayer against each lane's own; DEV0 == DEV1 loopback)
//        fnfl-drive qsaj-test [DEV0] [DEV1]   (FMS2: the joint round's merged QSA launches against each lane's own;
//            DEV0 == DEV1 loopback)
//        fnfl-drive gdnj-test [DEV]   (FMS2: the joint round's one GDN core launch against each lane's core())
//        fnfl-drive hc2-bench [DEV0] [DEV1] [all|ncu|ncu-up|ncu-fused|nsys|q8|cold]   (q8: DR1; cold, ncu-up, ncu-fused: HC4)
//        fnfl-drive load-plan [OUT] [MODEL] [DRAFT]   (CPU only: the placement, the GGUF slice hashes, the dry run)
//        fnfl-drive split-plan [--model GGUF] [--k K] [--slots N] [--slot-cells C] [--pool-cells P] [--cal ...]
//            [--free-front G0,G1] [--free-back G0,G1]   (TTTS1, CPU only: each box's weights and state, BF16 and q8,
//            and the headroom from a measured free figure; fn/loader/load_check.h)
//        fnfl-drive load-check [OUT] [CELLS] [MODEL] [DRAFT]   (devices 0 and 1; default CELLS 198,720)
//        fnfl-drive xchg2-test [DEV0] [DEV1] [ROUNDS]   (R2: X-A, X-F, X-ROUTE and X-SELECT, synthetic; default 0 1 200)
//        fnfl-drive x16-test [DEV0] [DEV1]   (FMS2 item 7: the joint 16-bit X-A / X-F against the host's rounding)
//        fnfl-drive px1-link [DEV0] [DEV1] [all|nsys|san] [CACHE] [ROUTES]   (PX1 Part A: large two-way messages
//            beside the read kernel and EP1's gate/up; default 0 1 all)
//        fnfl-drive dp-test [DEVICE] [ROWS|all] [full|san|all]   (DP1, DP2; ROWS like 1,3, default all; san: the
//            vocabulary at 8,192 rows, full: at 124,160, all: both; default full)
//        fnfl-drive dpj-test [DEVICE] [full|san|tc|f16] [CASE]   (FMU2: the joint product at U = 2, 3, 4 lanes
//            against each lane's B = 5 launch, bitwise, and FP64; FMU4: then "sampled pick", the joint draft step's
//            sampled pick against each lane's lone pick on DEVICE and the other, san: loopback; san: fewer cases, U = 2
//            and 4; tc (FMS5): the router cases on the tensor cores, against FP64's bound; f16 (FMS6): the quantized
//            cases at U = 3 and 4 with the dense FP16 switch on, against FP64's 2^-8 bound; CASE: only the cases whose
//            name contains one of its '|'-separated parts)
//        fnfl-drive dp-bench [DEVICE] [all|seq|nsys|ncu:POINT|group:NAME]   (DP1 steps 6-8, DP2 steps 5-7)
//        fnfl-drive pd-test|hcp-test|gdp-test|qsp-test|pe-test|dcp-test [DEVICE] [full|san]   (PB1: the prompt
//            kernels against their FP64 or bitwise references: dense and router, HC and the exchange, GDN, QSA,
//            embedding and PLE, the draft's catch-up pieces; san: fewer and smaller cases; hcp-test real=DIR: PB3's
//            tensor HC on PB1's saved real inputs in DIR as well)
//        fnfl-drive v-test [all|san] [MODEL]   (V1: the head, the logits and greedy on devices 0 and 1; san: fewer
//            cases, loopback on device 0, for Compute Sanitizer; the GGUF read on the CPU only)
//        fnfl-drive v-bench [DEV0] [DEV1]   (V1 step 5; synthetic weights, default 0 1)
//        fnfl-drive layer-test|layer-bench REF_DECODE REF_PROMPT OUT [full|san] [POSITIONS]   (R2: layers 10 and 11 on
//            devices 0 and 1 against ref_dump's decode and prompt dumps; san: one pass over 64 positions; exit 3 and no
//            model load while a piece of the two layers is not built)
//        fnfl-drive xp1-test DEV0 DEV1 [--variants V0,...] [--steps 1,2,3,...] [--no-graph] [--traffic hx7|synthetic]
//            [--synth DIR] [--hx7 DIR]   (XP1: the coordination prototype's checks; no model load)
//        fnfl-drive xp1-bench DEV0 DEV1 OUT --targets FILE --calib FILE [--stand-in-speed target|today]
//            [--variants V0,...] [--rows 1,3] [--traffic hx7|synthetic] [--alone] [--trace] [--calibrate-only]
//            [--rounds N] [--dense] [--gdn standin|core|omit] [--qsa standin|attend|omit]
//                (DP2: --dense times DP2's launches in V2; GD1: layer 10's recurrence stand-in, gdn_core, or neither;
//                QA2: layer 11's QSA attention stand-in, attend, or neither, with --dense)
//            [--synth DIR] [--hx7 DIR]   (XP1: the pair timed; no model load)
//        fnfl-drive model-test REF_DIR SPREAD_JSON IDS OUT [full|san|race] [POSITIONS]   (TG1: the whole target pass on
//            devices 0 and 1 against ref_dump3's decode-mode logits; san: 16 positions in phases; race: layers 8..11)
//        fnfl-drive model-bench IDS OUT [all|nsys [DEPTH B [real]]]   (TG1: the pass at 4,096, 58,581, 131,072 and
//            196,608 cells; nsys: one point inside cudaProfilerStart/Stop, by default 131,072 cells at B = 3; SF1: DEPTH
//            and B, a real history at DEPTH <= 4,096 and a synthetic one above, as all builds each point; TR1: real, a
//            real history at any DEPTH, refused when IDS holds fewer than DEPTH + B tokens)
//        fnfl-drive generate PROMPT_IDS OUT [N_TOKENS] [--evict-ple] [--dry-paths] [--temp T] [--top-k K] [--top-p P]
//            [--min-p P] [--repeat-penalty R] [--repeat-last-n N] [--presence-penalty X] [--frequency-penalty X]
//            [--seed S]   (TG1: the prompt at B = 4, the bootstrap, then greedy at B = 1; PL1: --evict-ple evicts the
//            PLE table first, --dry-paths checks the paths without a load; SM1: a sampling option samples instead)
//        fnfl-drive state-test [all|san] [graph|eager] [DEV0] [DEV1]   (SR1: commit by selection and the truncate
//            primitive through the round frame, random weights, no model load)
//        fnfl-drive spec PROMPT_IDS OUT [real|dry] --draft-attention dense [--seed S] [--decode-s SECONDS] [--temp T]
//            [--top-k K] [--top-p P] [--min-p P]   (SP1: real sampled speculation, C4a-C4f in one load; dry: zero
//            weights, short segments, the C1 plumbing check; TK1: the T = 1 segments' settings, default T 1, top-k 20,
//            top-p 0.95, min-p 0.05)
//        fnfl-drive spec PROMPT_IDS OUT run|run-dry --draft-attention dense [--runs N] [--eog IDS] SETTING [--and
//            SETTING ..]   (SV1: the runtime's engine as the server drives it, N requests a setting from a fresh
//            prefill; SETTING: [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S] [--max-tokens N]; SF1:
//            OUT/rounds.csv, one row a step with its depth and the host clock at each completion wait)
//        fnfl-drive spec HISTORY_IDS OUT ab|ab-dry --draft-attention sparse SETTING   (SP2: K1-K3 at d = 4, then one
//            continuation per arm, d = 2 dense and d = 4 sparse, from one saved decode-ready state)
//        fnfl-drive model-dry IDS OUT [graph|eager|sample|spec] [--draft-attention dense] [--temp T] [--top-k K]
//            [--top-p P] [--min-p P]   (TG1 / SR1 plumbing with zero weights; SM1: sample runs the sampler, TK1: with
//            these settings, default samp.h's options; DR1: the draft rounds too when --draft-attention is given)
//        fnfl-drive draft-test REF_DRAFT IDS OUT --draft-attention dense [--draft-subset FILE]   (DR1 Stage C: the
//            draft block against ref_dump_draft's dumps in REF_DRAFT; a missing --draft-attention is an error)
//        fnfl-drive draft-bench IDS OUT --draft-attention dense [--draft-subset FILE]   (DR1 Stage C: the draft step
//            and catch-up at 4,096, 58,581, 131,072 and 196,608 cells)
//        fnfl-drive sampler-ref   (SM1 step 1: the CPU reference sampler's own tests, CPU only)
//        fnfl-drive samp-test [all|quick|san] [DEV0] [DEV1] [RECORDED] [OUT]   (SM1 steps 2-5: the sampler against the
//            reference on devices 0 and 1; san: few draws, loopback on DEV0 == DEV1, for Compute Sanitizer)
//        fnfl-drive samp-bench [DEV0] [DEV1] [OUT]   (SM1 step 6: each path alone at B = 1 and 3, CUDA graphs)
//        fnfl-drive prefill-test IDS OUT [plan|dry|dry-eager|vcheck|model] [POSITIONS] --draft-attention dense
//            [--draft-subset FILE]; model (PB1): --ref-decode DIR --ref-prompt DIR [--cells N] [--deep IDS
//            [--deep-restore N] [--append IDS --append-restore N] [--max-tokens N] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S]]
//            (PF1: bulk prefill's driver against the decode path; plan: no device; dry: zero weights, a prompt of
//            POSITIONS tokens (default 4,096, at most 7,000); model: one load)
//        fnfl-drive rollback-test IDS OUT [real|dry|retain]   (SR1: scripted commits and truncates over 512 reference tokens on the real
//            model, against a clean replay)
//            FMS2 item 7: rollback-test IDS OUT mu1 --joint-kl DEEP runs joint-kl after mu1's PASS on its model
//        fnfl-drive joint-kl DEEP OUT [real|dry]   (FMS2 item 7: KL(lone || joint) of four lanes, 16-bit joint X-A /
//            X-F, against the lone BF16-KV yardstick; one load, or model-dry's zero weights)
//        fnfl-drive split-test [all|san] [graph|eager] [--k LIST] [--cases lone,joint]   (TTTS1: one box against the
//            split at k = 3, 27, 45, bitwise, on synthetic weights (no model data): lone rounds of 1 .. 5 rows, commits
//            and cuts, a reset, joint rounds of U = 2 .. 4 at every d, the HC slot rule; san: layers 0 .. 5 at k = 3,
//            eager, for Compute Sanitizer)
//        fnfl-drive stage --split-layer K --draft-attention sparse [--slots N] [--cells N] [--pool-cells N]
//            [--spec-d D] [--eog IDS] [--record-hash] [--split-addr IP]   (TTTS1: the split's follower, the back
//            stage, until the leader's X-BYE); --split [--split-layer K] on spec run (and rollback-test, joint-kl):
//            this process the leader, the front stage
//        fnfl-drive pipe-test   (TTTS1: the split's two-group pipeline core against a fake leader and follower, CPU
//            only: 1 .. 8 writers, joins and leaves at every point, cancels, ends of generation, readers interleaved,
//            the follower's failure; its invariants checked at every action)
//        fnfl-drive link-test leader|follower [--addr IP] [--port P] [--timeout S] [--cases all|LIST] [--stream N]
//            [--devices D0,D1]   (TTTS1: the box-to-box link, one process a box: X-HELLO, the transport line,
//            X-BOUNDARY decode and prompt forms with pattern payloads compared on the far cards, X-STEP, X-CTRL and
//            X-ACK, N (100,000) messages alternating groups, refusals, and the follower killed mid-transfer; no
//            timings printed)
//        --ple-table lazy|full, anywhere after the command (PN1): the PLE table's residency in every model load;
//            lazy (default, PN1 Amendment 1): the plain mapping and PL1's hints; full: read into RAM and pinned at
//            the load; generate --evict-ple needs lazy

#include "fn/fnfl.h"
#include "fn/kernels/act/act_test.h"
#include "fn/kernels/dense/dense_bench.h"
#include "fn/kernels/dense/dense_joint_test.h"
#include "fn/kernels/dense/dense_prompt_test.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/draft/draft_bench.h"
#include "fn/kernels/draft/draft_prompt_test.h"
#include "fn/kernels/draft/draft_test.h"
#include "fn/kernels/experts/experts_bench.h"
#include "fn/kernels/experts/experts_calib.h"
#include "fn/kernels/experts/experts_cpu_test.h"
#include "fn/kernels/experts/experts_prompt_bench.h"
#include "fn/kernels/experts/experts_prompt_test.h"
#include "fn/kernels/experts/experts_test.h"
#include "fn/kernels/experts/grids_test.h"
#include "fn/kernels/gdn/gdn_bench.h"
#include "fn/kernels/gdn/gdn_prompt_test.h"
#include "fn/kernels/gdn/gdn_test.h"
#include "fn/kernels/gdn/gdnj_test.h"
#include "fn/kernels/hc/hc_bench.h"
#include "fn/kernels/hc/hc_prompt_test.h"
#include "fn/kernels/hc/hc_test.h"
#include "fn/kernels/head/head_bench.h"
#include "fn/kernels/head/head_test.h"
#include "fn/kernels/ple/ple_bench.h"
#include "fn/kernels/ple/ple_prompt_test.h"
#include "fn/kernels/ple/ple_test.h"
#include "fn/kernels/qsa/qa2_bench.h"
#include "fn/kernels/qsa/qa2_test.h"
#include "fn/kernels/qsa/qsa_bench.h"
#include "fn/kernels/qsa/qsa_prompt_test.h"
#include "fn/kernels/qsa/qsa_test.h"
#include "fn/kernels/qsa/qsaj_test.h"
#include "fn/kernels/samp/samp_bench.h"
#include "fn/kernels/samp/samp_ref.h"
#include "fn/kernels/samp/samp_test.h"
#include "fn/loader/load_check.h"
#include "fn/model/kv_tier_test.h"
#include "fn/proto/px1/px1_link.h"
#include "fn/proto/xp1/xp1_bench.h"
#include "fn/proto/xp1/xp1_test.h"
#include "fn/quant/quant_test.h"
#include "fn/runtime/layer_test.h"
#include "fn/runtime/pipe.h"
#include "fn/runtime/ple_host.h"
#include "fn/runtime/ple_pin_test.h"
#include "fn/runtime/split_stage_test.h"
#include "fn/runtime/stage_cmds.h"
#include "fn/runtime/state_test.h"
#include "fn/runtime/target_cmds.h"
#include "fn/xchg/link_test.h"
#include "fn/xchg/x16_test.h"
#include "fn/xchg/xchg2_test.h"
#include "fn/xchg/xchg_test.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// "all" or a comma-separated list of non-negative integers; false on anything else
static bool parse_list(const char * s, std::vector<int32_t> & out) {
    out.clear();
    if (strcmp(s, "all") == 0) {
        return true;
    }
    long v = -1;
    for (const char * p = s;; ++p) {
        if (*p >= '0' && *p <= '9') {
            v = (v < 0 ? 0 : v) * 10 + (*p - '0');
            if (v > 1 << 30) {
                return false;
            }
        } else if (*p == ',' || *p == '\0') {
            if (v < 0) {
                return false;
            }
            out.push_back(static_cast<int32_t>(v));
            v = -1;
            if (*p == '\0') {
                return true;
            }
        } else {
            return false;
        }
    }
}

// a row-count list such as 1,3 as a mask (bit B-1); false on anything but digits 1..4 and commas
// DR1: --draft-attention V and --draft-subset FILE from argv[from ..] into a; the other arguments in order into rest.
// ME7: --draft-model FILE, the MTP draft GGUF (default: the Q8_0 draft; the Q2 draft is work/ME7's splice)
static bool draft_flags(int argc, char ** argv, int from, fnfl_target_args & a, std::vector<std::string> & rest) {
    for (int i = from; i < argc; ++i) {
        if (strcmp(argv[i], "--draft-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: no value\n", argv[i]);
                return false;
            }
            a.draft_path = argv[++i];
            continue;
        }
        const bool att = strcmp(argv[i], "--draft-attention") == 0;
        if (att || strcmp(argv[i], "--draft-subset") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: no value\n", argv[i]);
                return false;
            }
            (att ? a.draft_attention : a.draft_subset) = argv[++i];
            a.draft_attention_given = a.draft_attention_given || att;
            continue;
        }
        rest.emplace_back(argv[i]);
    }
    return true;
}

// TK1: --temp T, --top-k K, --top-p P or --min-p P at rest[i] (i moves to its value); false for anything else
static bool sampling_flag(const std::vector<std::string> & rest, size_t & i, fn::samp::options & s) {
    if (i + 1 >= rest.size()) {
        return false;
    }
    const std::string & f = rest[i];
    const char * v = rest[i + 1].c_str();
    if (f == "--temp") {
        s.temp = static_cast<float>(atof(v));
    } else if (f == "--top-k") {
        s.top_k = atoi(v);
    } else if (f == "--top-p") {
        s.top_p = static_cast<float>(atof(v));
    } else if (f == "--min-p") {
        s.min_p = static_cast<float>(atof(v));
    } else {
        return false;
    }
    ++i;
    return true;
}

static bool parse_rows(const char * s, uint32_t & mask) {
    mask = 0u;
    for (const char * p = s; *p != '\0'; ++p) {
        if (*p >= '1' && *p <= '5') { // SP2: 1..5
            mask |= 1u << (*p - '1');
        } else if (*p != ',') {
            return false;
        }
    }
    return mask != 0u;
}

// ME3 (fnfl-drive settings test; no model, no device): the server's settings path (fnfl_settings, the glue's) against
// the driver's parse. Each table entry given once in its server spelling must read back as the driver's spelling
// parses it (the same request JSON); a cache setting without the cache, a K/V one without --kv-host, a value out of
// range and an unknown name are refused; then the read-back and the request JSON of a server start with every
// setting at a value of its own
// KV8 Amendment 1: fnfl_kv_format over every combination of the given value (none, bf16, q8, an invalid one) and the
// configuration (Config A, one card, cached TP, one card with the host KV), against the table below; no model
static int kv_format_test() {
    struct cfg {
        const char * name;
        int32_t single_card;
        bool cached_tp, kv_host;
    };
    const cfg cfgs[] = {{"Config A", -1, false, false},
                        {"--fnfl-single-card 0", 0, false, false},
                        {"--fnfl-cached-tp", -1, true, false},
                        {"--fnfl-single-card 0 --fnfl-kv-host", 0, false, true}};
    struct want {
        bool ok, q8;
        const char * line; // the startup text, or a fragment of the refusal
    };
    // [value][configuration]
    const char * values[] = {"", "bf16", "q8", "q4"};
    const want table[4][4] = {
        {{true, true, "q8 (default)"},
         {true, false, "bf16 (default: --fnfl-single-card has no q8)"},
         {true, false, "bf16 (default: --fnfl-cached-tp has no q8)"},
         {true, false, "bf16 (default: --fnfl-single-card with --fnfl-kv-host has no q8)"}},
        {{true, false, "bf16"}, {true, false, "bf16"}, {true, false, "bf16"}, {true, false, "bf16"}},
        {{true, true, "q8"},
         {false, false, "serves resident TP on two cards (Config A) only"},
         {false, false, "serves resident TP on two cards (Config A) only"},
         {false, false, "serves resident TP on two cards (Config A) only"}},
        {{false, false, "is bf16 or q8, not 'q4'"},
         {false, false, "is bf16 or q8, not 'q4'"},
         {false, false, "is bf16 or q8, not 'q4'"},
         {false, false, "is bf16 or q8, not 'q4'"}}};
    int32_t bad = 0, n = 0;
    for (int32_t v = 0; v < 4; ++v) {
        for (int32_t k = 0; k < 4; ++k) {
            fnfl_config c;
            c.kv_format = values[v];
            c.single_card = cfgs[k].single_card;
            c.cached_tp = cfgs[k].cached_tp;
            c.kv_host = cfgs[k].kv_host;
            fnfl_kv_choice out;
            std::string err;
            const bool ok = fnfl_kv_format(c, out, err);
            const want & w = table[v][k];
            const bool pass = ok == w.ok && (ok ? out.q8 == w.q8 && out.line == w.line
                                                : err.find(w.line) != std::string::npos && !out.q8);
            ++n;
            bad += pass ? 0 : 1;
            printf("kv-format-test: --fnfl-kv-format %-4s %-38s -> %s: %s\n", values[v][0] ? values[v] : "(none)",
                   cfgs[k].name, ok ? out.line.c_str() : ("refused: " + err).c_str(), pass ? "PASS" : "FAIL");
        }
    }
    printf("kv-format-test: %s (%d of %d combinations as the table)\n", bad == 0 ? "PASS" : "FAIL", n - bad, n);
    return bad == 0 ? 0 : 1;
}

static int settings_test() {
    // a legal value for each entry, "derived" ones given a CPU (checked against the allowed set at init, not here)
    auto sample = [](const fn::settings::entry & e) -> std::string {
        const std::string n = e.name;
        if (n == "worker-cpus") {
            return "2,3";
        }
        if (n == "helper-cpus") {
            return "4,5";
        }
        if (n == "service-cpu") {
            return "1";
        }
        if (n == "decode-cpu") {
            return "0";
        }
        return e.def;
    };
    int32_t n_ok = 0, n_bad = 0;
    for (int32_t i = 0; i < fn::settings::n_entries; ++i) {
        const fn::settings::entry & e = fn::settings::table[i];
        const std::string val = sample(e);
        fn::settings::values vd, vs;
        std::string err, text;
        const bool dp = fn::settings::find(e.drive) == i && fn::settings::parse(i, val, vd, err);
        fnfl_config c;
        c.single_card = 0;
        c.kv_host = true;
        c.settings.emplace_back(e.serve, val);
        const bool sp = fn::settings::find(e.serve) == i && fnfl_settings(c, vs, text, err);
        const bool same = dp && sp && fn::settings::json(vd) == fn::settings::json(vs);
        n_ok += same ? 1 : 0;
        n_bad += same ? 0 : 1;
        if (!same) {
            printf("settings test: %s / %s %s: the driver %s, the server %s, %s (%s)\n", e.drive, e.serve,
                   val.c_str(), dp ? "parses" : "refuses", sp ? "parses" : "refuses",
                   dp && sp ? "different JSON" : "-", err.c_str());
        }
    }
    printf("settings test: %d of %d entries parse alike in both spellings (driver %s, server fnfl_settings)%s\n",
           n_ok, fn::settings::n_entries, "fn::settings::parse", n_bad == 0 ? "" : ": FAIL");
    struct refusal {
        const char * what;
        const char * name;
        const char * val;
        bool cache, kv;
    };
    const refusal rs[] = {
        {"a cache setting without the cache", "--fnfl-expert-short-read", "64", false, false},
        {"a K/V setting without --fnfl-kv-host", "--fnfl-kv-window", "32768", true, false},
        {"a value out of range", "--fnfl-expert-swap-decay", "1.5", true, true},
        {"a chunk not a multiple of 256", "--fnfl-expert-prompt-chunk", "1000", true, true},
        {"an unknown name", "--fnfl-expert-nothing", "1", true, true},
    };
    int32_t n_ref = 0;
    for (const refusal & r : rs) {
        fnfl_config c;
        c.single_card = r.cache ? 0 : -1;
        c.kv_host = r.kv;
        c.settings.emplace_back(r.name, r.val);
        fn::settings::values v;
        std::string err, text;
        const bool refused = !fnfl_settings(c, v, text, err);
        n_ref += refused ? 1 : 0;
        printf("settings test: %s (%s %s): %s%s%s\n", r.what, r.name, r.val, refused ? "refused: " : "ACCEPTED",
               refused ? err.c_str() : "", refused ? "" : ": FAIL");
    }
    // the executor (scope 0) without the cache: accepted
    fnfl_config c0;
    c0.settings.emplace_back("--fnfl-expert-executor", "eager");
    fn::settings::values v0;
    std::string e0, t0;
    const bool exec_ok = fnfl_settings(c0, v0, t0, e0) && v0.executor == "eager";
    printf("settings test: the executor without the cache (--fnfl-expert-executor eager): %s\n",
           exec_ok ? "accepted, eager" : "FAIL");
    fnfl_config ca;
    ca.single_card = 0;
    ca.kv_host = true;
    for (int32_t i = 0; i < fn::settings::n_entries; ++i) {
        ca.settings.emplace_back(fn::settings::table[i].serve, sample(fn::settings::table[i]));
    }
    fn::settings::values va;
    std::string ea, ta;
    const bool all_ok = fnfl_settings(ca, va, ta, ea);
    printf("settings test: a server start with every setting given (--fnfl-single-card 0 --fnfl-kv-host):\n%s",
           all_ok ? ta.c_str() : (ea + "\n").c_str());
    const bool pass = n_bad == 0 && n_ref == int32_t(sizeof(rs) / sizeof(rs[0])) && exec_ok && all_ok;
    printf("settings test (driver/server parity, scope and range refusals, read-back): %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// TTTS1: the split's options (taken out of argv in main), applied to a command's arguments
struct split_opts {
    bool        leader = false;  // --split
    int32_t     k      = 27;     // --split-layer K
    std::string addr   = "10.25.0.2";
    int32_t     port   = 13371;
    std::string commit, digest;
    bool        record_hash = false;
    int32_t     groups = 0;  // --fnfl-split-groups (joint-kl on the split: 1 one U4 group, 2 two pipelined U2; 0 both)
};

void apply_split(fnfl_target_args & a, const split_opts & so, int32_t role) {
    a.split_role   = role;
    a.split_k      = so.k;
    a.split_addr   = so.addr;
    a.split_port   = so.port;
    a.split_commit = so.commit;
    a.split_digest = so.digest;
    a.record_hash  = so.record_hash;
    a.split_groups = so.groups;
}

int main(int argc, char ** argv) {
    // PN1: --ple-table lazy|full anywhere after the command sets the residency of every model load; it is taken out
    // of argv here so each command's own parsing is unchanged
    fn::runtime::ple_residency ple_table = fn::runtime::ple_residency::lazy;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--ple-table") == 0) {
            if (i + 1 >= argc || !fn::runtime::ple_residency_parse(argv[i + 1], ple_table)) {
                fprintf(stderr, "%s: --ple-table is lazy or full\n", argv[0]);
                return 2;
            }
            for (int j = i; j + 2 <= argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            --i;
        }
    }
    // TTTS1: --split, --split-layer K, --split-addr IP, --split-port P, --split-commit HEX, --split-digest HEX,
    // --record-hash and --fnfl-split-groups N anywhere after the command, taken out of argv alike
    split_opts so;
    for (int i = 2; i < argc; ++i) {
        const std::string f    = argv[i];
        const bool        has  = i + 1 < argc;
        int               take = 0;
        if (f == "--split") {
            so.leader = true;
            take      = 1;
        } else if (f == "--record-hash") {
            so.record_hash = true;
            take           = 1;
        } else if (has && (f == "--split-layer" || f == "--fnfl-split-layer")) {
            so.k = atoi(argv[i + 1]);
            take = 2;
        } else if (has && f == "--split-addr") {
            so.addr = argv[i + 1];
            take    = 2;
        } else if (has && f == "--split-port") {
            so.port = atoi(argv[i + 1]);
            take    = 2;
        } else if (has && f == "--split-commit") {
            so.commit = argv[i + 1];
            take      = 2;
        } else if (has && f == "--split-digest") {
            so.digest = argv[i + 1];
            take      = 2;
        } else if (has && f == "--fnfl-split-groups") {
            so.groups = atoi(argv[i + 1]);
            take      = 2;
        }
        if (take > 0) {
            for (int j = i; j + take <= argc; ++j) {
                argv[j] = argv[j + take];
            }
            argc -= take;
            --i;
        }
    }
    if (so.k < 2 || so.k > 47 || so.groups < 0 || so.groups > 2) {
        fprintf(stderr, "%s: --split-layer is 2 .. 47 and --fnfl-split-groups 1 or 2\n", argv[0]);
        return 2;
    }
    // KV8: --kv-format bf16|q8 (or the server's --fnfl-kv-format) anywhere after the command, taken out of argv alike:
    // the target KV's format for the target commands, qa2-test, qsp-test and state-test mu1 (bf16 unless given).
    // TTTS1: also before the command (the pinned loads' form, fnfl-drive --kv-format q8 spec ...)
    bool kv_q8 = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--kv-format") == 0 || strcmp(argv[i], "--fnfl-kv-format") == 0) {
            if (i + 1 >= argc || (strcmp(argv[i + 1], "bf16") != 0 && strcmp(argv[i + 1], "q8") != 0)) {
                fprintf(stderr, "%s: --kv-format is bf16 or q8\n", argv[0]);
                return 2;
            }
            kv_q8 = strcmp(argv[i + 1], "q8") == 0;
            for (int j = i; j + 2 <= argc; ++j) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            --i;
        }
    }
    // ME1: --single-card N, --cached-tp 1, --ex-pool-mib M, --ex-reserve-mib M, --ex-cpu N (-1: physical cores
    // less one), --kv-host 1 (the host-backed KV tier, one or two devices) and --device-cap-mib M (a ceiling on the
    // device's whole use) anywhere after the command (spec run, model-dry), taken out of argv alike
    constexpr int32_t n_me1 = 7;
    int64_t me1[n_me1] = {-1, -1, 768, -1, 0, 0, -1};
    static const char * const me1_flag[n_me1] = {"--single-card", "--ex-pool-mib", "--ex-reserve-mib", "--ex-cpu",
                                                 "--cached-tp",    "--kv-host",     "--device-cap-mib"};
    for (int i = 2; i < argc; ++i) {
        int32_t f = 0;
        while (f < n_me1 && strcmp(argv[i], me1_flag[f]) != 0) {
            ++f;
        }
        if (f == n_me1) {
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "%s: %s needs a value\n", argv[0], me1_flag[f]);
            return 2;
        }
        me1[f] = strtoll(argv[i + 1], nullptr, 10);
        for (int j = i; j + 2 <= argc; ++j) {
            argv[j] = argv[j + 2];
        }
        argc -= 2;
        --i;
    }
    // ME2 (expire at ME2's disposition): --ex-profile PATH (the cache's boot ranking, a STRP file), --ex-profile-save
    // PATH (where the learned ranking is saved) and --ex-swap-overlap 0|1 (1 unless given) anywhere after the command,
    // as ME1's
    constexpr int32_t n_me2 = 3;
    std::string me2[n_me2] = {"", "", "1"};
    static const char * const me2_flag[n_me2] = {"--ex-profile", "--ex-profile-save", "--ex-swap-overlap"};
    for (int i = 2; i < argc; ++i) {
        int32_t f = 0;
        while (f < n_me2 && strcmp(argv[i], me2_flag[f]) != 0) {
            ++f;
        }
        if (f == n_me2) {
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "%s: %s needs a value\n", argv[0], me2_flag[f]);
            return 2;
        }
        me2[f] = argv[i + 1];
        for (int j = i; j + 2 <= argc; ++j) {
            argv[j] = argv[j + 2];
        }
        argc -= 2;
        --i;
    }
    if (me2[2] != "0" && me2[2] != "1") {
        fprintf(stderr, "%s: --ex-swap-overlap is 0 or 1\n", argv[0]);
        return 2;
    }
    // ME3: the cache's and the host K/V tier's settings (fn/cache_settings.h: --ex-<name>, --kv-<name>), anywhere
    // after the command, taken out of argv alike; the server's spellings are accepted too
    fn::settings::values cache_set;
    for (int i = 2; i < argc; ++i) {
        const int32_t f = fn::settings::find(argv[i]);
        if (f < 0) {
            continue;
        }
        std::string e;
        if (i + 1 >= argc || !fn::settings::parse(f, argv[i + 1], cache_set, e)) {
            fprintf(stderr, "%s: %s\n", argv[0], i + 1 >= argc ? (std::string(argv[i]) + " needs a value").c_str()
                                                               : e.c_str());
            return 2;
        }
        for (int j = i; j + 2 <= argc; ++j) {
            argv[j] = argv[j + 2];
        }
        argc -= 2;
        --i;
    }
    // ME1: the expert cache's streams wait on flags its service thread writes, and a kernel's first load under CUDA's
    // lazy loading waits for the device, so the cache needs every module loaded at the context's creation (before any
    // CUDA call here; the cache refuses a lazy context)
    if (me1[0] >= 0 || me1[4] != 0) {
        setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    }
    if (kv_q8 && (me1[0] >= 0 || me1[4] != 0 || me1[5] != 0)) {
        fprintf(stderr,
                "%s: --kv-format q8 serves resident TP on two cards only: not with --single-card, --cached-tp or "
                "--kv-host\n",
                argv[0]);
        return 2;
    }
    const auto me1_args = [&](fnfl_target_args & a) {
        a.single_card = int32_t(me1[0]);
        a.ex_pool_mib = me1[1];
        a.ex_reserve_mib = me1[2];
        a.ex_cpu = int32_t(me1[3]);
        a.cached_tp = me1[4] != 0 && me1[0] < 0;
        a.kv_host = me1[5] != 0;
        a.device_cap_mib = me1[6];
        a.ex_profile = me2[0];
        a.ex_profile_save = me2[1];
        a.ex_swap_overlap = me2[2] == "1";
        a.cache = cache_set;
        a.ex_eager = cache_set.executor == "eager";
    };
    if (argc >= 2 && strcmp(argv[1], "act-test") == 0) {
        return fnfl_act_test(argc >= 3 ? atoi(argv[2]) : 0);
    }
    if (argc >= 2 && strcmp(argv[1], "act-bench") == 0) {
        return fnfl_act_bench(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "all");
    }
    if (argc >= 2 && strcmp(argv[1], "gd-test") == 0) {
        static const char * model = "/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
                                    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
        return fnfl_gd_test(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "all", argc >= 5 ? argv[4] : model);
    }
    if (argc >= 2 && strcmp(argv[1], "gdnj-test") == 0) {
        return fnfl_gdnj_test(argc >= 3 ? atoi(argv[2]) : 0);
    }
    if (argc >= 2 && strcmp(argv[1], "gd-bench") == 0) {
        return fnfl_gd_bench(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "all");
    }
    static const char * fnfl_model = "/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
                                     "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    if (argc >= 2 && strcmp(argv[1], "ple-test") == 0) {
        return fnfl_ple_test(argc >= 3 ? argv[2] : "all", argc >= 4 ? atoi(argv[3]) : 0, argc >= 5 ? atoi(argv[4]) : 1,
                             argc >= 6 ? argv[5] : fnfl_model);
    }
    if (argc >= 2 && strcmp(argv[1], "ple-pin-test") == 0) {
        static const char * pn1_dir = "/home/jeremy/src/flashnext/work/PN1";
        return fnfl_ple_pin_test(argc >= 3 ? argv[2] : "unit", argc >= 4 ? argv[3] : pn1_dir,
                                 argc >= 5 ? atoi(argv[4]) : 0);
    }
    if (argc >= 2 && strcmp(argv[1], "ple-bench") == 0) {
        static const char * tokens = "/home/jeremy/src/flashnext/work/FL2/smoke/fl1-tokenize.out";
        return fnfl_ple_bench(argc >= 4 ? atoi(argv[3]) : 0, argc >= 5 ? atoi(argv[4]) : 1, argc >= 3 ? argv[2] : "all",
                              argc >= 6 ? argv[5] : fnfl_model, argc >= 7 ? argv[6] : tokens);
    }
    if (argc >= 2 && strcmp(argv[1], "hc-test") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        uint32_t rows_mask = argc >= 4 ? 0u : 0xfu;
        for (const char * p = argc >= 4 ? argv[3] : ""; *p != '\0'; ++p) {
            if (*p >= '1' && *p <= '4') {
                rows_mask |= 1u << (*p - '1');
            } else if (*p != ',') {
                fprintf(stderr, "hc-test: ROWS is a list of row counts 1..4, for example 1,3\n");
                return 2;
            }
        }
        return fnfl_hc_test(device, rows_mask);
    }
    if (argc >= 2 && strcmp(argv[1], "hc-bench") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        return fnfl_hc_bench(device);
    }
    if (argc >= 2 && strcmp(argv[1], "qa-test") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        std::vector<int32_t> depths, rows;
        if ((argc >= 4 && !parse_list(argv[3], depths)) || (argc >= 5 && !parse_list(argv[4], rows))) {
            fprintf(stderr, "qa-test: DEPTHS and ROWS are 'all' or lists such as 65536 or 1,3\n");
            return 2;
        }
        for (const int32_t r : rows) {
            if (r < 1 || r > 5) { // SP2: B = 5
                fprintf(stderr, "qa-test: ROWS are 1..5\n");
                return 2;
            }
        }
        return fnfl_qsa_test(device, depths, rows, argc >= 6 ? argv[5] : "");
    }
    if (argc >= 2 && strcmp(argv[1], "qa-bench") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        return fnfl_qsa_bench(device, argc >= 4 ? argv[3] : "all", argc >= 5 ? argv[4] : "");
    }
    static const char * hx7_fixture = "/home/jeremy/src/flashnext/work/QA1/fixtures/hx7-l3-w0.bin"; // QA1's HX7 case
    if (argc >= 2 && strcmp(argv[1], "qa2-test") == 0) {
        static const char * model = "/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
                                    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
        return fnfl_qa2_test(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "all", argc >= 5 ? argv[4] : model,
                             argc >= 6 ? argv[5] : hx7_fixture, kv_q8);
    }
    if (argc >= 2 && strcmp(argv[1], "kv-tier-test") == 0) {
        return fnfl_kv_tier_test(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "all");
    }
    if (argc >= 2 && strcmp(argv[1], "qa2-bench") == 0) {
        return fnfl_qa2_bench(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "all",
                              argc >= 5 ? argv[4] : hx7_fixture);
    }
    if (argc >= 2 && strcmp(argv[1], "draft-kernel-bench") == 0) {
        return fnfl_draft_kernel_bench(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? atoi(argv[3]) : 1);
    }
    if (argc >= 2 && strcmp(argv[1], "draft-kernel-test") == 0) {
        return fnfl_draft_kernel_test(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? atoi(argv[3]) : 1,
                                      argc >= 5 ? argv[4] : "all");
    }
    // ME3: the settings table (fn/cache_settings.h), both entry points' spellings, defaults, units and ranges
    if (argc >= 2 && strcmp(argv[1], "kv-format-test") == 0) {
        return kv_format_test();
    }
    if (argc >= 3 && strcmp(argv[1], "settings") == 0 && strcmp(argv[2], "test") == 0) {
        return settings_test();
    }
    if (argc >= 2 && strcmp(argv[1], "settings") == 0) {
        for (const fn::settings::entry & e : fn::settings::table) {
            printf("%-26s %-34s %-8s %-12s default %-10s range %s: %s\n", e.drive, e.serve, e.metavar, e.unit, e.def,
                   e.range, e.help);
        }
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "ex-calib") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        return fnfl_ex_calib(device);
    }
    static const char * ex_cache = "/home/jeremy/src/flashnext/work/C1b/cache"; // fn/quant's synthetic blocks (C1b)
    if (argc >= 2 && strcmp(argv[1], "ex-prep") == 0) {
        return fnfl_ex_prep(argc >= 3 ? argv[2] : ex_cache);
    }
    if (argc >= 2 && strcmp(argv[1], "ex-cpu-test") == 0) {
        return fnfl_ex_cpu_test(argc >= 3 ? argv[2] : ex_cache);
    }
    if (argc >= 2 && strcmp(argv[1], "ex-test") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        uint32_t rows_mask = argc >= 4 ? 0u : 0x5u; // B = 1 and 3 (EX1 step 4)
        for (const char * p = argc >= 4 ? argv[3] : ""; *p != '\0'; ++p) {
            if (*p >= '1' && *p <= '5') { // SP2: B = 5
                rows_mask |= 1u << (*p - '1');
            } else if (*p != ',') {
                fprintf(stderr, "ex-test: ROWS is a list of row counts 1..5, for example 1,3\n");
                return 2;
            }
        }
        return fnfl_ex_test(device, rows_mask, argc >= 5 ? argv[4] : "all", argc >= 6 ? argv[5] : ex_cache,
                            argc >= 7 ? argv[6] : nullptr);
    }
    if (argc >= 2 && strcmp(argv[1], "ex-bench") == 0) {
        static const char * hx7 = "/home/jeremy/src/flashnext/work/HX7/runs/capture-58581-20261009T090240Z";
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        return fnfl_ex_bench(device, argc >= 4 ? argv[3] : "all", argc >= 5 ? argv[4] : ex_cache,
                             argc >= 6 ? argv[5] : hx7);
    }
    static const char * fn4 = "/home/jeremy/src/flashnext-opt/work/FN4/20261007T170304Z/routes.txt";
    // FP16 input (C2), fn/quant's deployed-range weights (C1b Amendment 4)
    static const char * ep_refs = "/home/jeremy/src/flashnext/work/C1b/ep-refs-a4";
    if (argc >= 2 && strcmp(argv[1], "ep-prep") == 0) {
        return fnfl_ep_prep(argc >= 3 ? argv[2] : ep_refs, argc >= 4 ? argv[3] : ex_cache, argc >= 5 ? argv[4] : fn4,
                            argc >= 6 ? argv[5] : "all");
    }
    if (argc >= 2 && strcmp(argv[1], "ep-test") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        return fnfl_ep_test(device, argc >= 4 ? argv[3] : "128,1536", argc >= 5 ? argv[4] : "all",
                            argc >= 7 ? argv[6] : ep_refs, argc >= 8 ? argv[7] : ex_cache, argc >= 9 ? argv[8] : fn4,
                            argc >= 6 ? argv[5] : "full");
    }
    if (argc >= 2 && strcmp(argv[1], "ep-bench") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        return fnfl_ep_bench(device, argc >= 4 ? argv[3] : "all", argc >= 5 ? argv[4] : ex_cache,
                             argc >= 6 ? argv[5] : fn4);
    }
    if (argc >= 2 && strcmp(argv[1], "decode-check") == 0) {
        if (argc < 3) {
            fprintf(stderr, "decode-check: DIR (work/C1b/ref's dump) is required\n");
            return 2;
        }
        return fnfl_decode_check(argv[2]);
    }
    if (argc >= 2 && strcmp(argv[1], "synth-check") == 0) {
        return fnfl_synth_check();
    }
    if (argc >= 2 && strcmp(argv[1], "grid-test") == 0) {
        return fnfl_grid_test(argc >= 3 ? argv[2]
                                        : "/home/jeremy/src/flashnext-opt/work/REL2/engine/ggml/src/ggml-common.h");
    }
    if (argc >= 2 &&
        (strcmp(argv[1], "xchg-test") == 0 || strcmp(argv[1], "hc2-test") == 0 || strcmp(argv[1], "hc2-bench") == 0 ||
         strcmp(argv[1], "hcj-test") == 0 || strcmp(argv[1], "qsaj-test") == 0)) {
        const int dev0 = argc >= 3 ? atoi(argv[2]) : 0;
        const int dev1 = argc >= 4 ? atoi(argv[3]) : 1;
        if (strcmp(argv[1], "hcj-test") == 0) {
            return fnfl_hcj_test(dev0, dev1);
        }
        if (strcmp(argv[1], "qsaj-test") == 0) {
            return fnfl_qsaj_test(dev0, dev1);
        }
        if (strcmp(argv[1], "hc2-bench") == 0) {
            if (dev0 == dev1) {
                fprintf(stderr, "hc2-bench: DEV0 and DEV1 must differ\n");
                return 2;
            }
            return fnfl_hc2_bench(dev0, dev1, argc >= 5 ? argv[4] : "all");
        }
        const bool xt = strcmp(argv[1], "xchg-test") == 0;
        const char * rows = argc >= (xt ? 6 : 5) ? argv[xt ? 5 : 4] : (xt ? "1,3" : "1,2,3,4,5");
        uint32_t mask = 0u;
        if (!parse_rows(rows, mask)) {
            fprintf(stderr, "%s: ROWS is a list of row counts 1..5, for example 1,3\n", argv[1]);
            return 2;
        }
        if (xt) {
            const long rounds = argc >= 5 ? atol(argv[4]) : 10000;
            if (rounds < 1 || rounds > 1000000 || (mask & ~0x5u) != 0u) {
                fprintf(stderr, "xchg-test: ROUNDS is 1..1000000 and ROWS 1 and/or 3\n");
                return 2;
            }
            return fnfl_xchg_test(dev0, dev1, static_cast<int32_t>(rounds), mask);
        }
        return fnfl_hc2_test(dev0, dev1, mask);
    }
    if (argc >= 2 && strcmp(argv[1], "split-test") == 0) {
        return fn::runtime::fnfl_split_test(argc, argv);
    }
    if (argc >= 2 && strcmp(argv[1], "pipe-test") == 0) {
        std::string report;
        const bool  ok = fn::runtime::pipe::self_test(report);
        fputs(report.c_str(), stdout);
        return ok ? 0 : 1;
    }
    if (argc >= 2 && strcmp(argv[1], "link-test") == 0) {
        fn::xchg::link::test_options o;
        const bool role_ok = argc >= 3 && (strcmp(argv[2], "leader") == 0 || strcmp(argv[2], "follower") == 0);
        bool       ok      = role_ok;
        if (role_ok) {
            o.leader = strcmp(argv[2], "leader") == 0;
        }
        for (int i = 3; ok && i < argc; ++i) {
            const bool has = i + 1 < argc;
            if (strcmp(argv[i], "--addr") == 0 && has) {
                o.addr = argv[++i];
            } else if (strcmp(argv[i], "--port") == 0 && has) {
                o.port = static_cast<uint16_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "--timeout") == 0 && has) {
                o.timeout_s = atof(argv[++i]);
            } else if (strcmp(argv[i], "--cases") == 0 && has) {
                o.cases = argv[++i];
            } else if (strcmp(argv[i], "--stream") == 0 && has) {
                o.stream_messages = atol(argv[++i]);
            } else if (strcmp(argv[i], "--devices") == 0 && has) {
                ok = sscanf(argv[++i], "%d,%d", &o.device[0], &o.device[1]) == 2;
            } else {
                ok = false;
            }
        }
        if (!ok || o.timeout_s <= 0 || o.stream_messages < 1000) {
            fprintf(stderr,
                    "usage: fnfl-drive link-test leader|follower [--addr IP] [--port P] [--timeout S] "
                    "[--cases all|decode,prompt,small,stream,refuse,kill] [--stream N >= 1000] "
                    "[--devices D0,D1]\n");
            return 2;
        }
        return fn::xchg::link::run_test(o);
    }
    if (argc >= 2 && (strcmp(argv[1], "load-plan") == 0 || strcmp(argv[1], "load-check") == 0)) {
        const bool check = strcmp(argv[1], "load-check") == 0;
        fnfl_load_check_args a;
        a.out_dir = argc >= 3 ? argv[2] : "/home/jeremy/src/flashnext/work/R1/raw";
        int32_t i = 3;
        a.n_cells = 196608 + 2048 + 64;
        if (check && argc > i) {
            const long n = atol(argv[i++]);
            if (n < 4096 || n > (1L << 18)) {
                fprintf(stderr, "load-check: CELLS is 4096..262144\n");
                return 2;
            }
            a.n_cells = static_cast<int32_t>(n);
        }
        static const char * model = "/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
                                    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
        static const char * draft = "/home/jeremy/models/Qwen3.8-Flash-Next-GGUF/MTP/"
                                    "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
        a.model_path = argc > i ? argv[i] : model;
        a.draft_path = argc > i + 1 ? argv[i + 1] : draft;
        return check ? fnfl_load_check(a) : fnfl_load_plan(a);
    }
    if (argc >= 2 && strcmp(argv[1], "split-plan") == 0) {
        // TTTS1 step 8 (CPU only): the split's per-box plan (fn/loader/load_check.h)
        fnfl_split_plan_args a;
        a.model_path =
            "/home/jeremy/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
        a.draft_path    = "/home/jeremy/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
        const auto pair = [](const char * v, double out[2]) {
            return sscanf(v, "%lf,%lf", &out[0], &out[1]) == 2;
        };
        bool bad = false;
        for (int i = 2; i < argc && !bad; ++i) {
            const std::string f   = argv[i];
            const bool        has = i + 1 < argc;
            if (f == "--model" && has) {
                a.model_path = argv[++i];
            } else if (f == "--draft" && has) {
                a.draft_path = argv[++i];
            } else if (f == "--k" && has) {
                a.k = atoi(argv[++i]);
            } else if (f == "--slots" && has) {
                a.slots = atoi(argv[++i]);
            } else if (f == "--slot-cells" && has) {
                a.slot_cells = atoi(argv[++i]);
            } else if (f == "--pool-cells" && has) {
                a.pool_cells = atoi(argv[++i]);
            } else if (f == "--cal" && i + 6 < argc) {
                a.cal_model      = argv[++i];
                a.cal_draft      = argv[++i];
                a.cal_k          = atoi(argv[++i]);
                a.cal_slots      = atoi(argv[++i]);
                a.cal_slot_cells = atoi(argv[++i]);
                a.cal_pool_cells = atoi(argv[++i]);
            } else if (f == "--free-front" && has) {
                bad = !pair(argv[++i], a.free_front);
            } else if (f == "--free-back" && has) {
                bad = !pair(argv[++i], a.free_back);
            } else {
                bad = true;
            }
        }
        if (bad || a.k < 2 || a.k > 47 || a.slots < 1 || a.slots > 8) {
            fprintf(stderr,
                    "usage: fnfl-drive split-plan [--model GGUF] [--draft GGUF] [--k K] [--slots N] "
                    "[--slot-cells C] [--pool-cells P] [--cal MODEL DRAFT K SLOTS SLOT_CELLS POOL_CELLS] "
                    "[--free-front G0,G1] [--free-back G0,G1]\n");
            return 2;
        }
        return fnfl_split_plan(a);
    }
    if (argc >= 2 && strcmp(argv[1], "px1-link") == 0) {
        return fnfl_px1_link(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? atoi(argv[3]) : 1, argc >= 5 ? argv[4] : "all",
                             argc >= 6 ? argv[5] : ex_cache, argc >= 7 ? argv[6] : fn4);
    }
    if (argc >= 2 && strcmp(argv[1], "x16-test") == 0) {
        return fnfl_x16_test(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? atoi(argv[3]) : 1);
    }
    if (argc >= 2 && strcmp(argv[1], "xchg2-test") == 0) {
        const int32_t d0 = argc >= 3 ? atoi(argv[2]) : 0, d1 = argc >= 4 ? atoi(argv[3]) : 1;
        const long rounds = argc >= 5 ? atol(argv[4]) : 200;
        if (rounds < 1 || rounds > 100000) {
            fprintf(stderr, "xchg2-test: ROUNDS is 1..100000\n");
            return 2;
        }
        return fnfl_xchg2_test(d0, d1, static_cast<int32_t>(rounds));
    }
    if (argc >= 2 && strcmp(argv[1], "pd-test") == 0) {
        return fnfl_pd_test(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "full");
    }
    if (argc >= 2 && strcmp(argv[1], "qsp-test") == 0) {
        return fnfl_qsp_test(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "full", kv_q8); // KV8: its format
    }
    {
        const struct {
            const char * name;
            int (*fn)(int, const char *);
        } pb1[] = {{"hcp-test", fnfl_hcp_test},
                   {"gdp-test", fnfl_gdp_test},
                   {"pe-test", fnfl_pe_test},
                   {"dcp-test", fnfl_dcp_test}};

        for (const auto & t : pb1) {
            if (argc >= 2 && strcmp(argv[1], t.name) == 0) {
                return t.fn(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "full");
            }
        }
    }
    if (argc >= 2 && strcmp(argv[1], "dpj-test") == 0) { // FMU2: the joint product
        return fnfl_dpj_test(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "full",
                             argc >= 5 ? argv[4] : nullptr);
    }
    if (argc >= 2 && strcmp(argv[1], "dp-test") == 0) {
        uint32_t rows_mask = 0x1fu; // 1 .. 5 (SP2)
        if (argc >= 4 && strcmp(argv[3], "all") != 0 && !parse_rows(argv[3], rows_mask)) {
            fprintf(stderr, "dp-test: ROWS is all or a list of row counts 1..5, for example 1,3\n");
            return 2;
        }
        return fnfl_dp_test(argc >= 3 ? atoi(argv[2]) : 0, rows_mask, argc >= 5 ? argv[4] : "full");
    }
    if (argc >= 2 && strcmp(argv[1], "v-test") == 0) {
        const char * vmode = argc >= 3 ? argv[2] : "all";
        const bool loop = strcmp(vmode, "san") == 0;
        return fnfl_v_test(0, loop ? 0 : 1, vmode,
                           argc >= 4 ? argv[3]
                                     : "/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
                                       "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf");
    }
    if (argc >= 2 && strcmp(argv[1], "sampler-ref") == 0) {
        return fn::samp::fnfl_sampler_ref();
    }
    if (argc >= 2 && strcmp(argv[1], "samp-test") == 0) {
        const char * md = argc >= 3 ? argv[2] : "all";
        const int d0 = argc >= 4 ? atoi(argv[3]) : 0;
        const int d1 = argc >= 5 ? atoi(argv[4]) : (strcmp(md, "san") == 0 ? d0 : 1);
        return fnfl_samp_test(d0, d1, md,
                              argc >= 6 ? argv[5] : "/home/jeremy/src/flashnext/work/TG1/raw/ref-decode/logits.bin.gz",
                              argc >= 7 ? argv[6] : "");
    }
    if (argc >= 2 && strcmp(argv[1], "samp-bench") == 0) {
        return fnfl_samp_bench(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? atoi(argv[3]) : 1, argc >= 5 ? argv[4] : "");
    }
    if (argc >= 2 && strcmp(argv[1], "v-bench") == 0) {
        return fnfl_v_bench(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? atoi(argv[3]) : 1, "all");
    }
    if (argc >= 2 && strcmp(argv[1], "dp-bench") == 0) {
        return fnfl_dp_bench(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "all");
    }
    if (argc >= 2 && (strcmp(argv[1], "layer-test") == 0 || strcmp(argv[1], "layer-bench") == 0)) {
        if (argc < 5) {
            fprintf(stderr, "%s: REF_DECODE REF_PROMPT OUT [full|san] [POSITIONS]\n", argv[1]);
            return 2;
        }
        fnfl_layer_test_args a;
        a.ple_table = ple_table;
        a.ref_decode = argv[2];
        a.ref_prompt = argv[3];
        a.out_dir = argv[4];
        {
            // the yardstick beside the prompt dump's directory (R2's raw/ref-spread.json)
            std::string pd = a.ref_prompt;
            while (pd.size() > 1 && pd.back() == '/') {
                pd.pop_back();
            }
            const size_t sl = pd.rfind('/');
            a.spread_json = (sl == std::string::npos ? std::string(".") : pd.substr(0, sl)) + "/ref-spread.json";
        }
        a.san = argc >= 6 && strcmp(argv[5], "san") == 0;
        if (argc >= 7) {
            const long n = atol(argv[6]);
            if (n < 4 || n > 4096) {
                fprintf(stderr, "%s: POSITIONS is 4..4096\n", argv[1]);
                return 2;
            }
            a.n_pos = static_cast<int32_t>(n);
        }
        a.model_path = "/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
                       "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
        a.draft_path = "/home/jeremy/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
        return strcmp(argv[1], "layer-test") == 0 ? fnfl_layer_test(a) : fnfl_layer_bench(a);
    }
    if (argc >= 2 && strcmp(argv[1], "model-test") == 0) {
        fnfl_target_args a;
        a.kv_q8 = kv_q8; // KV8
        // KV8 (load 4, with --kv-format q8): --spec-ids IDS runs spec's checks on the same load after the q8 passes,
        // with --draft-attention, --draft-subset and --decode-s S as spec's
        std::vector<std::string> rest, pos;
        if (!draft_flags(argc, argv, 2, a, rest)) {
            return 2;
        }
        bool bad = false;
        for (size_t i = 0; i < rest.size(); ++i) {
            if (rest[i] == "--spec-ids" && i + 1 < rest.size()) {
                a.spec_ids = rest[++i];
            } else if (rest[i] == "--decode-s" && i + 1 < rest.size()) {
                a.decode_s = atof(rest[++i].c_str());
            } else if (rest[i].rfind("--", 0) == 0) {
                bad = true;
            } else {
                pos.push_back(rest[i]);
            }
        }
        if (bad || pos.size() < 4 || pos.size() > 6 || (!a.spec_ids.empty() && !kv_q8)) {
            fprintf(stderr, "model-test: REF_DIR SPREAD_JSON IDS OUT [full|san|race] [POSITIONS] [--spec-ids IDS "
                            "--draft-attention dense [--decode-s S]] (KV8: --spec-ids with --kv-format q8)\n");
            return 2;
        }
        a.ple_table = ple_table;
        a.ref_dir = pos[0];
        a.spread_json = pos[1];
        a.ids = pos[2];
        a.out_dir = pos[3];
        a.mode = pos.size() >= 5 ? pos[4] : "full";
        if (pos.size() >= 6) {
            const long n = atol(pos[5].c_str());
            if (n < 16 || n > 4096 || n % 8 != 0) {
                fprintf(stderr, "model-test: POSITIONS is 16..4096, a multiple of 8\n");
                return 2;
            }
            a.n_pos = static_cast<int32_t>(n);
        }
        return fnfl_model_test(a);
    }
    if (argc >= 2 && strcmp(argv[1], "model-bench") == 0) {
        fnfl_target_args a;
        a.kv_q8 = kv_q8; // KV8
        a.ple_table = ple_table;
        if (argc < 4 || argc == 6 || argc > 8 || (argc >= 7 && strcmp(argv[4], "nsys") != 0) ||
            (argc == 8 && strcmp(argv[7], "real") != 0)) {
            fprintf(stderr, "model-bench: IDS OUT [all|nsys [DEPTH B [real]]]\n");
            return 2;
        }
        a.ids = argv[2];
        a.out_dir = argv[3];
        a.mode = argc >= 5 ? argv[4] : "all";
        a.bench_real = argc == 8;
        if (argc >= 7) {
            char * e1 = nullptr;
            char * e2 = nullptr;
            const long depth = strtol(argv[5], &e1, 10);
            const long b = strtol(argv[6], &e2, 10);
            if (*e1 != '\0' || *e2 != '\0' || depth < 1 || depth > 196608 || b < 1 || b > 4) {
                fprintf(stderr, "model-bench nsys: DEPTH is 1..196608 and B is 1..4 (got %s %s)\n", argv[5], argv[6]);
                return 2;
            }
            a.bench_depth = static_cast<int32_t>(depth);
            a.bench_b = static_cast<int32_t>(b);
        }
        return fnfl_model_bench(a);
    }
    if (argc >= 2 && strcmp(argv[1], "state-test") == 0) {
        return fnfl_state_test(argc, argv, kv_q8);
    }
    if (argc >= 2 && strcmp(argv[1], "rollback-test") == 0) {
        if (argc < 4) {
            fprintf(stderr, "rollback-test: IDS OUT [real|dry|retain|mu1|mu1-dry|mu1-joint|mu1-joint-eager]\n");
            return 2;
        }
        fnfl_target_args a;
        a.kv_q8 = kv_q8; // KV8
        a.ple_table = ple_table;
        a.ids = argv[2];
        a.out_dir = argv[3];
        a.mode = argc >= 5 ? argv[4] : "real";
        for (int i = 5; i < argc; ++i) { // FMS2 item 7: --joint-kl DEEP after mu1 (load 1)
            if (strcmp(argv[i], "--joint-kl") == 0 && i + 1 < argc) {
                a.joint_kl = argv[++i];
            } else {
                fprintf(stderr, "rollback-test: unknown argument %s\n", argv[i]);
                return 2;
            }
        }
        if (!a.joint_kl.empty() && a.mode != "mu1" && a.mode != "mu1-dry") {
            fprintf(stderr, "rollback-test: --joint-kl follows mu1 or mu1-dry\n");
            return 2;
        }
        apply_split(a, so,
                    so.leader ? 1 : 0);  // TTTS1: --split: mu1 (and its joint-kl) on the split, this box the front
        if (a.split_role != 0 && a.mode != "mu1" && a.mode != "mu1-dry") {
            fprintf(stderr, "rollback-test: --split runs mu1 or mu1-dry\n");
            return 2;
        }
        return fnfl_rollback_test(a);
    }
    if (argc >= 2 && strcmp(argv[1], "joint-kl") == 0) {
        if (argc < 4 || (argc >= 5 && strcmp(argv[4], "real") != 0 && strcmp(argv[4], "dry") != 0)) {
            fprintf(stderr, "joint-kl: DEEP OUT [real|dry]\n");
            return 2;
        }
        fnfl_target_args a;
        a.ple_table = ple_table;
        a.joint_kl = argv[2];
        a.out_dir = argv[3];
        a.mode = argc >= 5 ? argv[4] : "real";
        apply_split(a, so, so.leader ? 1 : 0);  // TTTS1: --split [--fnfl-split-groups 1|2]
        return fnfl_joint_kl(a);
    }
    if (argc >= 2 && strcmp(argv[1], "prefill-test") == 0) {
        fnfl_target_args a;
        a.kv_q8 = kv_q8; // KV8
        a.ple_table = ple_table;
        std::vector<std::string> rest, pos;
        if (!draft_flags(argc, argv, 2, a, rest)) {
            return 2;
        }
        // PB1's model mode: the pinned dumps, the cells, and the deep stage's prompt, restore point and setting
        fnfl_target_args::run_setting cur;
        cur.sampling.temp           = 1.0f;
        cur.sampling.penalty_last_n = 0;
        cur.max_tokens              = 64;
        bool bad                    = false;
        for (size_t i = 0; i < rest.size(); ++i) {
            const bool more = i + 1 < rest.size();
            if (rest[i] == "--ref-decode" && more) {
                a.ref_dir = rest[++i];
            } else if (rest[i] == "--ref-prompt" && more) {
                a.ref_prompt = rest[++i];
            } else if (rest[i] == "--cells" && more) {
                a.n_cells = atoi(rest[++i].c_str());
            } else if (rest[i] == "--deep" && more) {
                a.deep_ids = rest[++i];
            } else if (rest[i] == "--deep-restore" && more) {
                a.deep_restore = atoi(rest[++i].c_str());
            } else if (rest[i] == "--append" && more) {
                a.append_ids = rest[++i];
            } else if (rest[i] == "--append-restore" && more) {
                a.append_restore = atoi(rest[++i].c_str());
            } else if (rest[i] == "--max-tokens" && more) {
                cur.max_tokens = atoi(rest[++i].c_str());
            } else if (rest[i] == "--temp" && more) {
                cur.sampling.temp = static_cast<float>(atof(rest[++i].c_str()));
            } else if (rest[i] == "--top-k" && more) {
                cur.sampling.top_k = atoi(rest[++i].c_str());
            } else if (rest[i] == "--top-p" && more) {
                cur.sampling.top_p = static_cast<float>(atof(rest[++i].c_str()));
            } else if (rest[i] == "--min-p" && more) {
                cur.sampling.min_p = static_cast<float>(atof(rest[++i].c_str()));
            } else if (rest[i] == "--seed" && more) {
                cur.sampling.seed = strtoull(rest[++i].c_str(), nullptr, 10);
            } else if (rest[i].rfind("--", 0) == 0) {
                bad = true;
            } else {
                pos.push_back(rest[i]);
            }
        }
        a.run_settings.push_back(cur);
        if (bad || pos.size() < 2 || pos.size() > 4 || cur.max_tokens < 1 || a.n_cells < 8192) {
            fprintf(stderr,
                    "prefill-test: IDS OUT [plan|dry|dry-eager|vcheck|model] [POSITIONS] --draft-attention dense "
                    "[--draft-subset FILE]; model: --ref-decode DIR --ref-prompt DIR [--cells N] [--deep IDS "
                    "[--deep-restore N] [--append IDS --append-restore N] [--max-tokens N] [--temp T] [--top-k K] "
                    "[--top-p P] [--min-p P] "
                    "[--seed S]]\n");
            return 2;
        }
        a.ids     = pos[0];
        a.out_dir = pos[1];
        a.mode    = pos.size() >= 3 ? pos[2] : "plan";
        if (pos.size() == 4) {
            a.n_pos = atoi(pos[3].c_str());
        }
        me1_args(a);
        return fnfl_prefill_test(a);
    }
    if (argc >= 2 && strcmp(argv[1], "model-dry") == 0) {
        fnfl_target_args a;
        a.kv_q8 = kv_q8; // KV8
        a.ple_table = ple_table;
        std::vector<std::string> rest, pos;
        bool bad = !draft_flags(argc, argv, 2, a, rest);
        for (size_t i = 0; i < rest.size() && !bad; ++i) {
            if (rest[i] == "--cells" && i + 1 < rest.size()) {
                a.n_cells = atoi(rest[++i].c_str());
            } else if (!sampling_flag(rest, i, a.sampling)) {
                bad = rest[i].rfind("--", 0) == 0;
                pos.push_back(rest[i]);
            }
        }
        if (bad || pos.size() < 2) {
            fprintf(stderr, "model-dry: IDS OUT [graph|eager|sample|spec] [--draft-attention dense] [--draft-subset "
                            "FILE] [--temp T] [--top-k K] [--top-p P] [--min-p P] (sample's settings)\n");
            return 2;
        }
        a.ids = pos[0];
        a.out_dir = pos[1];
        a.mode = pos.size() >= 3 ? pos[2] : "graph";
        me1_args(a);
        if ((!a.cached() && a.cache.any(1)) || (!a.kv_host && a.cache.any(2))) {
            fprintf(stderr, "model-dry: an expert cache setting needs --single-card or --cached-tp, a K/V one "
                            "--kv-host\n");
            return 2;
        }
        return fnfl_model_dry(a);
    }
    if (argc >= 2 && (strcmp(argv[1], "draft-test") == 0 || strcmp(argv[1], "draft-bench") == 0)) {
        const bool test = strcmp(argv[1], "draft-test") == 0;
        fnfl_target_args a;
        a.kv_q8 = kv_q8; // KV8
        a.ple_table = ple_table;
        std::vector<std::string> rest;
        if (!draft_flags(argc, argv, 2, a, rest) || rest.size() != (test ? 3u : 2u)) {
            fprintf(stderr, "%s: %sIDS OUT --draft-attention dense [--draft-subset FILE]\n", argv[1],
                    test ? "REF_DRAFT " : "");
            return 2;
        }
        size_t i = 0;
        if (test) {
            a.ref_draft = rest[i++];
        }
        a.ids = rest[i++];
        a.out_dir = rest[i++];
        return test ? fnfl_draft_test(a) : fnfl_draft_bench(a);
    }
    if (argc >= 2 && strcmp(argv[1], "stage") == 0) {
        // TTTS1: the follower (the back stage) of a split
        fnfl_target_args a;
        a.kv_q8     = kv_q8;
        a.ple_table = ple_table;
        apply_split(a, so, 2);
        std::vector<std::string> rest;
        if (!draft_flags(argc, argv, 2, a, rest)) {
            return 2;
        }
        bool bad = false;
        for (size_t i = 0; i < rest.size(); ++i) {
            const bool more = i + 1 < rest.size();
            if (rest[i] == "--dry") {
                a.mode = "dry";  // model-dry's load: zero weights, no model data (the no-load checks)
            } else if (rest[i] == "--slots" && more) {
                a.n_slots = atoi(rest[++i].c_str());
            } else if (rest[i] == "--cells" && more) {
                a.n_cells = atoi(rest[++i].c_str());
            } else if (rest[i] == "--pool-cells" && more) {
                a.pool_cells = atoi(rest[++i].c_str());
            } else if (rest[i] == "--spec-d" && more) {
                a.spec_d = atoi(rest[++i].c_str());
            } else if (rest[i] == "--eog" && more) {
                const std::string & v = rest[++i];
                for (size_t p = 0; p < v.size();) {
                    const size_t q = v.find(',', p);
                    a.eog.push_back(atoi(v.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str()));
                    p = q == std::string::npos ? v.size() : q + 1;
                }
            } else {
                bad = true;
            }
        }
        if (bad || a.n_slots < 1 || a.n_slots > 8 || a.spec_d < 1 || a.spec_d > 4) {
            fprintf(stderr,
                    "usage: fnfl-drive stage --split-layer K --draft-attention sparse [--dry] [--slots N (1 .. 8)] "
                    "[--cells N] [--pool-cells N] [--spec-d D] [--eog IDS] [--record-hash] [--split-addr IP] "
                    "[--split-port P] [--split-commit HEX] [--split-digest HEX] [--kv-format q8]\n");
            return 2;
        }
        return fnfl_stage(a);
    }
    if (argc >= 2 && strcmp(argv[1], "spec") == 0) {
        fnfl_target_args a;
        a.kv_q8 = kv_q8; // KV8
        a.ple_table = ple_table;
        apply_split(a, so, so.leader ? 1 : 0);
        std::vector<std::string> rest, pos;
        if (!draft_flags(argc, argv, 2, a, rest)) {
            return 2;
        }
        // SV1 run's settings: --and starts the next one (each from the defaults below)
        fnfl_target_args::run_setting cur;
        cur.sampling.temp = 1.0f;
        cur.sampling.penalty_last_n = 0;
        bool bad = false;
        for (size_t i = 0; i < rest.size(); ++i) {
            const bool more = i + 1 < rest.size();
            if (rest[i] == "--seed" && more) {
                a.sample = true;
                a.sampling.seed = strtoull(rest[++i].c_str(), nullptr, 10);
                cur.sampling.seed = a.sampling.seed;
            } else if (rest[i] == "--decode-s" && more) {
                a.decode_s = atof(rest[++i].c_str());
            } else if (rest[i] == "--runs" && more) {
                a.runs = atoi(rest[++i].c_str());
            } else if (sampling_flag(rest, i, cur.sampling)) {
                continue;
            } else if (rest[i] == "--spec-d" && more) {
                // SP2's switch: 1..4. TTTS1: inside a setting, the setting's d (the first setting's is the process's)
                cur.spec_d = atoi(rest[++i].c_str());
                if (a.run_settings.empty()) {
                    a.spec_d = cur.spec_d;
                }
            } else if (rest[i] == "--arms" && more) {
                a.arms = rest[++i]; // SP2 Amendment 3: spec ab's arms
            } else if (rest[i] == "--max-tokens" && more) {
                cur.max_tokens = atoi(rest[++i].c_str());
            } else if (rest[i] == "--ex-check" && more) {
                a.ex_check = rest[++i]; // ME1: spec run's placement check
            } else if (rest[i] == "--ex-ref" && more) {
                a.ex_ref = rest[++i];
            } else if (rest[i] == "--ex-kl" && more) {
                a.ex_kl = rest[++i]; // ME2: spec run's KL check (NC1 section 5)
            } else if (rest[i] == "--ex-kl-ref" && more) {
                a.ex_kl_ref = rest[++i];
            } else if (rest[i] == "--ex-kl-spread" && more) {
                a.ex_kl_spread = rest[++i];
            } else if (rest[i] == "--mx7-check" && more) {
                a.mx7_check = rest[++i]; // MX7: the prompt CPU share's model checks (mx7_check.h)
            } else if (rest[i] == "--mx7-shares" && more) {
                a.mx7_shares = rest[++i];
            } else if (rest[i] == "--mx7-dry-maxt" && more) {
                a.mx7_dry_maxt = atoi(rest[++i].c_str()); // MX7: run-dry's share candidates' pair bound
            } else if (rest[i] == "--next-ids" && more) {
                a.next_ids.push_back(rest[++i]); // ME2: spec run's further prompts
                a.next_append.push_back(false);
                a.next_rows.push_back(-1);
            } else if (rest[i] == "--next-append" && more) {
                a.next_ids.push_back(rest[++i]); // ME3: an append to the previous request's state
                a.next_append.push_back(true);
                a.next_rows.push_back(-1);
            } else if (rest[i] == "--next-append-rows" && i + 2 < rest.size()) {
                a.next_rows.push_back(atoi(rest[++i].c_str())); // ME3's cleanup check: an append of N rows read
                a.next_ids.push_back(rest[++i]);
                a.next_append.push_back(true);
            } else if (rest[i] == "--me3-cancel") {
                a.me3_cancel = true; // TEST: the cancellation check (expires at ME4's disposition)
            } else if (rest[i] == "--ex-native-hold") {
                a.ex_native_hold = true; // TEST: ME4's host-memory residency check in a dry run
            } else if (rest[i] == "--me9-check") {
                a.me9_check = true; // TEST: ME9's admission check
            } else if (rest[i] == "--me3-stall") {
                a.me3_stall = true; // TEST: the stall check (expires at ME4's disposition)
            } else if (rest[i] == "--me3-arms" && more) {
                a.me3_arms = rest[++i]; // ME3's comparison switch (expires at ME3's disposition)
            } else if (rest[i] == "--me4-arms" && more) {
                a.me4_arms = rest[++i]; // ME4's comparison switch (expires at ME4's disposition)
            } else if (rest[i] == "--append-close" && more) {
                a.append_close = atoi(rest[++i].c_str()); // ME4: the fixture's length-stop rule
            } else if (rest[i] == "--cells" && more) {
                a.n_cells = atoi(rest[++i].c_str()); // ME1: the slot's cells (198,720 unless given)
            } else if (rest[i] == "--and") {
                if (cur.spec_d < 0 || cur.spec_d > 4) {
                    bad = true;
                }
                a.run_settings.push_back(cur);
                cur = fnfl_target_args::run_setting();
                cur.sampling.temp = 1.0f;
                cur.sampling.penalty_last_n = 0;
            } else if (rest[i] == "--eog" && more) {
                // a comma list of end-of-generation IDs (the server's set)
                const std::string & v = rest[++i];
                size_t p = 0;
                while (p < v.size()) {
                    const size_t q = v.find(',', p);
                    a.eog.push_back(atoi(v.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str()));
                    p = q == std::string::npos ? v.size() : q + 1;
                }
            } else if (rest[i].rfind("--", 0) == 0) {
                bad = true;
            } else {
                pos.push_back(rest[i]);
            }
        }
        a.run_settings.push_back(cur);
        const bool run = pos.size() == 3 && (pos[2] == "run" || pos[2] == "run-dry");
        const bool ab = pos.size() == 3 && (pos[2] == "ab" || pos[2] == "ab-dry"); // SP2
        if (bad || pos.size() < 2 || pos.size() > 3 || a.decode_s <= 0.0 || a.decode_s > 120.0 || a.runs < 1 ||
            a.runs > 10 || a.spec_d < 1 || a.spec_d > 4 ||
            (pos.size() == 3 && pos[2] != "real" && pos[2] != "dry" && !run && !ab)) {
            fprintf(stderr, "spec: PROMPT_IDS OUT [real|dry] --draft-attention dense [--draft-subset FILE] [--seed S] "
                            "[--decode-s SECONDS (<= 120)] [--temp T] [--top-k K] [--top-p P] [--min-p P] (the T = 1 "
                            "segments' settings)\n"
                            "spec: PROMPT_IDS OUT run|run-dry --draft-attention dense [--draft-subset FILE] [--runs N] "
                            "[--eog ID,ID..] SETTING [--and SETTING ..], SETTING: [--temp T] [--top-k K] [--top-p P] "
                            "[--min-p P] [--seed S] [--max-tokens N] [--cells N]   (SV1: the runtime's engine as the "
                            "server drives it; ME1: --single-card N [--ex-pool-mib M] [--ex-reserve-mib M] [--ex-cpu N] "
                            "[--ex-check IDS [--ex-ref TOKENS]]; ME2: [--next-ids IDS ..], further prompts in the "
                            "same process; [--ex-kl IDS --ex-kl-ref DIR --ex-kl-spread JSON], the KL check; anywhere: "
                            "[--ex-profile STRP] [--ex-profile-save PATH] [--ex-swap-overlap 0|1]; ME3: "
                            "[--next-append IDS ..], an append to the previous request's state; "
                            "[--next-append-rows N IDS ..], an append whose read is N rows (checked); "
                            "[--me3-arms base,final]; [--me4-arms base,final]; [--me3-stall], the stall check "
                            "(a test); [--me3-cancel], the cancellation check (a test); "
                            "anywhere: the settings of fn/cache_settings.h, --ex-<name> V and --kv-<name> V, "
                            "fnfl-drive settings lists them)\n"
                            "spec: HISTORY_IDS OUT ab|ab-dry --draft-attention sparse [--draft-subset FILE] "
                            "[--arms D:FORM[:EX][,D:FORM[:EX]..]] SETTING   (SP2: the checks, then the arms, "
                            "2:dense,4:sparse unless given, from one decode-ready state; SP4: EX sp2, sp4, h2 or xb3, "
                            "the expert gate/up mapping, sp2 unless given)\n");
            return 2;
        }
        a.ids = pos[0];
        a.out_dir = pos[1];
        a.mode = pos.size() == 3 ? pos[2] : "real";
        me1_args(a);
        if ((!a.cached() && a.cache.any(1)) || (!a.kv_host && a.cache.any(2))) {
            fprintf(stderr, "spec: an expert cache setting needs --single-card or --cached-tp, a K/V one --kv-host\n");
            return 2;
        }
        if (a.cached() && !run) {
            fprintf(stderr, "spec: --single-card and --cached-tp serve spec run and run-dry only\n");
            return 2;
        }
        if (!a.ex_kl.empty() && (!run || !a.cached() || a.ex_kl_ref.empty() || a.ex_kl_spread.empty())) {
            fprintf(stderr, "spec: --ex-kl serves a cached spec run and needs --ex-kl-ref and --ex-kl-spread\n");
            return 2;
        }
        if (!a.mx7_check.empty() && (!run || !a.cached() || a.mx7_shares.empty())) {
            fprintf(stderr, "spec: --mx7-check serves a cached spec run and needs --mx7-shares\n");
            return 2;
        }
        if (a.mx7_dry_maxt != 0 &&
            (pos.size() != 3 || pos[2] != "run-dry" || !a.cached() || a.mx7_dry_maxt < 1 || a.mx7_dry_maxt > 4096)) {
            fprintf(stderr, "spec: --mx7-dry-maxt N (1..4096) serves a cached run-dry\n");
            return 2;
        }
        if (!a.next_ids.empty() && !run) {
            fprintf(stderr, "spec: --next-ids and --next-append serve spec run and run-dry only\n");
            return 2;
        }
        if (!a.me3_arms.empty() && (!run || !a.cached() ||
                                    (a.me3_arms != "base,final" && a.me3_arms != "base" && a.me3_arms != "final"))) {
            fprintf(stderr, "spec: --me3-arms base,final (or one of them) serves a cached spec run\n");
            return 2;
        }
        if (!a.me4_arms.empty() && (!run || !a.cached() || !a.me3_arms.empty() ||
                                    (a.me4_arms != "base,final" && a.me4_arms != "base" && a.me4_arms != "final"))) {
            fprintf(stderr, "spec: --me4-arms base,final (or one of them) serves a cached spec run, not with "
                            "--me3-arms\n");
            return 2;
        }
        // ME4 (Carlos request 1, R1-08): a run of an append extends the previous run's state, so its tokens cannot
        // equal run 0's; an append is read once
        if (a.runs != 1 && std::find(a.next_append.begin(), a.next_append.end(), true) != a.next_append.end()) {
            fprintf(stderr, "spec: --next-append and --next-append-rows take --runs 1 (each run would extend the "
                            "previous run's state)\n");
            return 2;
        }
        if (a.me9_check && (!run || a.single_card < 0)) {
            fprintf(stderr, "spec: --me9-check serves a cached spec run on one device (--single-card)\n");
            return 2;
        }
        if (a.me3_stall && (!run || !a.cached())) {
            fprintf(stderr, "spec: --me3-stall serves a cached spec run\n");
            return 2;
        }
        if (a.me3_cancel && (!run || !a.cached() || a.me3_stall)) {
            fprintf(stderr, "spec: --me3-cancel serves a cached spec run, without --me3-stall\n");
            return 2;
        }
        return ab ? fnfl_spec_ab(a) : run ? fnfl_spec_run(a) : fnfl_spec(a);
    }
    if (argc >= 2 && strcmp(argv[1], "generate") == 0) {
        const char * usage =
            "generate: PROMPT_IDS OUT [N_TOKENS] [--evict-ple] [--dry-paths] [--temp T] [--top-k K] [--top-p P] "
            "[--min-p P] [--repeat-penalty R] [--repeat-last-n N] [--presence-penalty X] [--frequency-penalty X] "
            "[--seed S]   (SM1: any sampling option selects the sampler; unset ones take the pinned server's "
            "defaults)\n";
        if (argc < 4) {
            fprintf(stderr, "%s", usage);
            return 2;
        }
        fnfl_target_args a;
        a.kv_q8 = kv_q8; // KV8
        a.ple_table = ple_table;
        a.ids = argv[2];
        a.out_dir = argv[3];
        for (int i = 4; i < argc; ++i) {
            const auto val = [&](float & f) {
                a.sample = true;
                f = i + 1 < argc ? static_cast<float>(atof(argv[++i])) : NAN;
            };
            if (strcmp(argv[i], "--evict-ple") == 0) {
                a.evict_ple = true;
            } else if (strcmp(argv[i], "--dry-paths") == 0) {
                a.dry_paths = true;
            } else if (strcmp(argv[i], "--temp") == 0) {
                val(a.sampling.temp);
            } else if (strcmp(argv[i], "--top-p") == 0) {
                val(a.sampling.top_p);
            } else if (strcmp(argv[i], "--min-p") == 0) {
                val(a.sampling.min_p);
            } else if (strcmp(argv[i], "--repeat-penalty") == 0) {
                val(a.sampling.penalty_repeat);
            } else if (strcmp(argv[i], "--presence-penalty") == 0) {
                val(a.sampling.penalty_present);
            } else if (strcmp(argv[i], "--frequency-penalty") == 0) {
                val(a.sampling.penalty_freq);
            } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
                a.sample = true;
                a.sampling.top_k = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--repeat-last-n") == 0 && i + 1 < argc) {
                a.sample = true;
                a.sampling.penalty_last_n = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
                a.sample = true;
                a.sampling.seed = strtoull(argv[++i], nullptr, 10);
            } else if (argv[i][0] == '-') {
                fprintf(stderr, "generate: unknown option %s\n%s", argv[i], usage);
                return 2;
            } else {
                a.n_gen = atoi(argv[i]);
                if (a.n_gen < 1 || a.n_gen > 2048) {
                    fprintf(stderr, "generate: N_TOKENS is 1..2048\n");
                    return 2;
                }
            }
        }
        return fnfl_generate(a);
    }
    if (argc >= 4 && (strcmp(argv[1], "xp1-test") == 0 || strcmp(argv[1], "xp1-bench") == 0)) {
        const bool test = strcmp(argv[1], "xp1-test") == 0;
        fnfl_xp1_test_args t;
        fnfl_xp1_bench_args b;
        t.dev0 = b.dev0 = atoi(argv[2]);
        t.dev1 = b.dev1 = atoi(argv[3]);
        t.synth_dir = b.synth_dir = "/home/jeremy/src/flashnext/work/XP1/synth";
        t.hx7_dir = b.hx7_dir = "/home/jeremy/src/flashnext/work/HX7/runs/capture-58581-20261009T090240Z";
        int i = 4;
        if (!test) {
            if (argc < 5) {
                fprintf(stderr, "xp1-bench: DEV0 DEV1 OUT ...\n");
                return 2;
            }
            b.out_dir = argv[i++];
        }
        const auto split = [](const char * s) {
            std::vector<std::string> v;
            std::string cur;
            for (const char * p = s;; ++p) {
                if (*p == ',' || *p == 0) {
                    if (!cur.empty()) {
                        v.push_back(cur);
                    }
                    cur.clear();
                    if (*p == 0) {
                        break;
                    }
                } else {
                    cur += *p;
                }
            }
            return v;
        };
        for (; i < argc; ++i) {
            const std::string o = argv[i];
            const bool has = i + 1 < argc;
            std::vector<int32_t> nums;
            if (o == "--variants" && has) {
                t.variants = b.variants = split(argv[++i]);
            } else if ((o == "--steps" || o == "--rows") && has && parse_list(argv[i + 1], nums) && !nums.empty()) {
                ++i;
                for (int32_t n : nums) {
                    if (n < 1 || n > 4) {
                        fprintf(stderr, "%s: rows are 1..4\n", argv[1]);
                        return 2;
                    }
                }
                (o == "--steps" ? t.steps : b.rows) = nums;
            } else if (o == "--no-graph") {
                t.no_graph = true;
            } else if (o == "--traffic" && has) {
                const std::string v = argv[++i];
                if (v != "hx7" && v != "synthetic") {
                    fprintf(stderr, "%s: --traffic is hx7 or synthetic\n", argv[1]);
                    return 2;
                }
                t.synthetic = b.synthetic = v == "synthetic";
            } else if (o == "--synth" && has) {
                t.synth_dir = b.synth_dir = argv[++i];
            } else if (o == "--hx7" && has) {
                t.hx7_dir = b.hx7_dir = argv[++i];
            } else if (o == "--targets" && has) {
                b.targets = argv[++i];
            } else if (o == "--calib" && has) {
                b.calib = argv[++i];
            } else if (o == "--stand-in-speed" && has) {
                b.speed = argv[++i];
            } else if (o == "--alone") {
                b.alone = true;
            } else if (o == "--trace") {
                b.trace = true;
            } else if (o == "--calibrate-only") {
                b.calibrate_only = true;
            } else if (o == "--dense" && !test) {
                b.dense = true;
            } else if (o == "--gdn" && has) {
                const std::string v = argv[++i];
                if (v != "standin" && v != "core" && v != "omit") {
                    fprintf(stderr, "xp1-bench: --gdn is standin, core or omit\n");
                    return 2;
                }
                b.gdn = v == "core" ? 1 : v == "omit" ? 2 : 0;
            } else if (o == "--qsa" && has) {
                const std::string v = argv[++i];
                if (v != "standin" && v != "attend" && v != "omit") {
                    fprintf(stderr, "xp1-bench: --qsa is standin, attend or omit\n");
                    return 2;
                }
                b.qsa = v == "attend" ? 1 : v == "omit" ? 2 : 0;
            } else if (o == "--rounds" && has) {
                b.rounds = atoi(argv[++i]);
                if (b.rounds < 1 || b.rounds > 20) {
                    fprintf(stderr, "xp1-bench: --rounds is 1..20\n");
                    return 2;
                }
            } else {
                fprintf(stderr, "%s: unknown option %s\n", argv[1], o.c_str());
                return 2;
            }
        }
        if (!test && (b.targets.empty() || b.calib.empty())) {
            fprintf(stderr, "xp1-bench: --targets and --calib are required\n");
            return 2;
        }
        return test ? fnfl_xp1_test(t) : fnfl_xp1_bench(b);
    }
    fprintf(
        stderr,
        "usage: %s "
        "act-test|act-bench|gd-test|gd-bench|hc-test|hc-bench|qa-test|qa-bench|qa2-test|qa2-bench|kv-tier-test|"
        "settings|kv-format-test|ex-calib|ex-prep|ex-cpu-test|ex-test|ex-bench|"
        "ep-prep|ep-test|ep-bench|decode-check|synth-check|grid-test|xchg-test|hc2-test|hcj-test|qsaj-test|gdnj-test|"
        "hc2-bench|load-plan|"
        "load-check|xchg2-test|px1-link|layer-test|layer-bench|xp1-test|xp1-bench|dp-test|dpj-test|dp-bench|pd-test|"
        "hcp-test|gdp-test|qsp-test|pe-test|dcp-test|ple-test|ple-pin-test|ple-bench|state-test|"
        "prefill-test|rollback-test|spec|draft-kernel-test|draft-kernel-bench|draft-test|draft-bench|sampler-ref|"
        "samp-test|samp-bench|v-test|v-bench [DEVICE] ...\n",
        argv[0]);
    return 2;
}
