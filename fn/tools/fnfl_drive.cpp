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
//        fnfl-drive gd-test [DEVICE] [all|short|chain|map|inv|san] [MODEL]   (GD1: DEVICE is the card; the weights
//            of layers 10 and 40 are read from MODEL's file, no model load)
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
//        fnfl-drive draft-kernel-test [DEV0] [DEV1] [all|san]   (DR1 Stage C: the draft kernels against FP64)
//        fnfl-drive draft-kernel-bench [DEV0] [DEV1]   (DR1 Stage C: DA-TC at the four depths, X-DRAFT-IN on both)
//        fnfl-drive ex-calib [DEVICE]
//        fnfl-drive ex-prep [CACHE]   (CPU only: the synthetic expert weights)
//        fnfl-drive ex-test [DEVICE] [ROWS] [FORMATS] [CACHE] [BASE]   (EX2: ROWS like 1,2,3,4; FORMATS all, san or
//            routed gate/up formats like iq2_xxs,iq3_s: every deployed instantiation with the shared expert; XB2:
//            BASE, 71b92a9's experts cubins, adds the xb3-bitwise and xb4-bitwise arms)
//        fnfl-drive ex-bench [DEVICE] [all|POINT] [CACHE] [HX7]   (EX2: every deployed launch warm and cold, HX7's
//            captured routing; HX7: the capture directory; XB2: xb2prof:POINTS, each point once a decode for ncu;
//            SP4 Amendment 1: sp4prof:POINTS:MAP, B = 5 on four decodes, cold, MAP 0 SP2 or 1 SP4, for ncu)
//        fnfl-drive ep-prep [REFS] [CACHE] [ROUTES]   (CPU only: ep-test's FP64 references)
//        fnfl-drive ep-test [DEVICE] [ROWS] [FORMATS] [full|san] [REFS] [CACHE] [ROUTES]   (ROWS like 128,1536)
//        fnfl-drive ep-bench [DEVICE] [all|ncu] [CACHE] [ROUTES]   (ROUTES: FN4's routes.txt)
//        fnfl-drive grid-test [HEADER]   (CPU only: the grid tables against ggml-common.h at 66e11c3; default the
//            pinned tree's)
//        fnfl-drive decode-check DIR   (CPU only: fn/quant's decoders bitwise against work/C1b/ref's pinned dump)
//        fnfl-drive synth-check   (CPU only: fn/quant's synthetic-block generators)
//        fnfl-drive xchg-test [DEV0] [DEV1] [ROUNDS] [ROWS]   (defaults 0 1 10000 1,3)
//        fnfl-drive hc2-test [DEV0] [DEV1] [ROWS]   (default ROWS 1,2,3,4,5)
//            (DEV0 == DEV1: loopback on one device and one stream, sends before reads: for Compute Sanitizer)
//        fnfl-drive hc2-bench [DEV0] [DEV1] [all|ncu|ncu-up|ncu-fused|nsys|q8|cold]   (q8: DR1; cold, ncu-up, ncu-fused: HC4)
//        fnfl-drive load-plan [OUT] [MODEL] [DRAFT]   (CPU only: the placement, the GGUF slice hashes, the dry run)
//        fnfl-drive load-check [OUT] [CELLS] [MODEL] [DRAFT]   (devices 0 and 1; default CELLS 198,720)
//        fnfl-drive xchg2-test [DEV0] [DEV1] [ROUNDS]   (R2: X-A, X-F, X-ROUTE and X-SELECT, synthetic; default 0 1 200)
//        fnfl-drive px1-link [DEV0] [DEV1] [all|nsys|san] [CACHE] [ROUTES]   (PX1 Part A: large two-way messages
//            beside the read kernel and EP1's gate/up; default 0 1 all)
//        fnfl-drive dp-test [DEVICE] [ROWS|all] [full|san|all]   (DP1, DP2; ROWS like 1,3, default all; san: the
//            vocabulary at 8,192 rows, full: at 124,160, all: both; default full)
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
//        --ple-table lazy|full, anywhere after the command (PN1): the PLE table's residency in every model load;
//            lazy (default, PN1 Amendment 1): the plain mapping and PL1's hints; full: read into RAM and pinned at
//            the load; generate --evict-ple needs lazy

#include "fn/kernels/act/act_test.h"
#include "fn/kernels/dense/dense_bench.h"
#include "fn/kernels/dense/dense_prompt_test.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/draft/draft_bench.h"
#include "fn/kernels/draft/draft_prompt_test.h"
#include "fn/kernels/draft/draft_test.h"
#include "fn/kernels/experts/experts_bench.h"
#include "fn/kernels/experts/experts_calib.h"
#include "fn/kernels/experts/experts_prompt_bench.h"
#include "fn/kernels/experts/experts_prompt_test.h"
#include "fn/kernels/experts/experts_test.h"
#include "fn/kernels/experts/grids_test.h"
#include "fn/kernels/gdn/gdn_bench.h"
#include "fn/kernels/gdn/gdn_prompt_test.h"
#include "fn/kernels/gdn/gdn_test.h"
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
#include "fn/kernels/samp/samp_bench.h"
#include "fn/kernels/samp/samp_ref.h"
#include "fn/kernels/samp/samp_test.h"
#include "fn/loader/load_check.h"
#include "fn/proto/px1/px1_link.h"
#include "fn/proto/xp1/xp1_bench.h"
#include "fn/proto/xp1/xp1_test.h"
#include "fn/quant/quant_test.h"
#include "fn/runtime/layer_test.h"
#include "fn/runtime/ple_host.h"
#include "fn/runtime/ple_pin_test.h"
#include "fn/runtime/state_test.h"
#include "fn/runtime/target_cmds.h"
#include "fn/xchg/xchg2_test.h"
#include "fn/xchg/xchg_test.h"

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
// DR1: --draft-attention V and --draft-subset FILE from argv[from ..] into a; the other arguments in order into rest
static bool draft_flags(int argc, char ** argv, int from, fnfl_target_args & a, std::vector<std::string> & rest) {
    for (int i = from; i < argc; ++i) {
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
                             argc >= 6 ? argv[5] : hx7_fixture);
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
    if (argc >= 2 && strcmp(argv[1], "ex-calib") == 0) {
        const int device = argc >= 3 ? atoi(argv[2]) : 0;
        return fnfl_ex_calib(device);
    }
    static const char * ex_cache = "/home/jeremy/src/flashnext/work/C1b/cache"; // fn/quant's synthetic blocks (C1b)
    if (argc >= 2 && strcmp(argv[1], "ex-prep") == 0) {
        return fnfl_ex_prep(argc >= 3 ? argv[2] : ex_cache);
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
        return fnfl_ep_prep(argc >= 3 ? argv[2] : ep_refs, argc >= 4 ? argv[3] : ex_cache, argc >= 5 ? argv[4] : fn4);
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
        (strcmp(argv[1], "xchg-test") == 0 || strcmp(argv[1], "hc2-test") == 0 || strcmp(argv[1], "hc2-bench") == 0)) {
        const int dev0 = argc >= 3 ? atoi(argv[2]) : 0;
        const int dev1 = argc >= 4 ? atoi(argv[3]) : 1;
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
    if (argc >= 2 && strcmp(argv[1], "px1-link") == 0) {
        return fnfl_px1_link(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? atoi(argv[3]) : 1, argc >= 5 ? argv[4] : "all",
                             argc >= 6 ? argv[5] : ex_cache, argc >= 7 ? argv[6] : fn4);
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
    {
        const struct {
            const char * name;
            int (*fn)(int, const char *);
        } pb1[] = {
            { "hcp-test", fnfl_hcp_test },
            { "gdp-test", fnfl_gdp_test },
            { "qsp-test", fnfl_qsp_test },
            { "pe-test",  fnfl_pe_test  },
            { "dcp-test", fnfl_dcp_test }
        };

        for (const auto & t : pb1) {
            if (argc >= 2 && strcmp(argv[1], t.name) == 0) {
                return t.fn(argc >= 3 ? atoi(argv[2]) : 0, argc >= 4 ? argv[3] : "full");
            }
        }
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
        if (argc < 6) {
            fprintf(stderr, "model-test: REF_DIR SPREAD_JSON IDS OUT [full|san|race] [POSITIONS]\n");
            return 2;
        }
        fnfl_target_args a;
        a.ple_table = ple_table;
        a.ref_dir = argv[2];
        a.spread_json = argv[3];
        a.ids = argv[4];
        a.out_dir = argv[5];
        a.mode = argc >= 7 ? argv[6] : "full";
        if (argc >= 8) {
            const long n = atol(argv[7]);
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
        return fnfl_state_test(argc, argv);
    }
    if (argc >= 2 && strcmp(argv[1], "rollback-test") == 0) {
        if (argc < 4) {
            fprintf(stderr, "rollback-test: IDS OUT [real|dry|retain]\n");
            return 2;
        }
        fnfl_target_args a;
        a.ple_table = ple_table;
        a.ids = argv[2];
        a.out_dir = argv[3];
        a.mode = argc >= 5 ? argv[4] : "real";
        return fnfl_rollback_test(a);
    }
    if (argc >= 2 && strcmp(argv[1], "prefill-test") == 0) {
        fnfl_target_args a;
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
        return fnfl_prefill_test(a);
    }
    if (argc >= 2 && strcmp(argv[1], "model-dry") == 0) {
        fnfl_target_args a;
        a.ple_table = ple_table;
        std::vector<std::string> rest, pos;
        bool bad = !draft_flags(argc, argv, 2, a, rest);
        for (size_t i = 0; i < rest.size() && !bad; ++i) {
            if (!sampling_flag(rest, i, a.sampling)) {
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
        return fnfl_model_dry(a);
    }
    if (argc >= 2 && (strcmp(argv[1], "draft-test") == 0 || strcmp(argv[1], "draft-bench") == 0)) {
        const bool test = strcmp(argv[1], "draft-test") == 0;
        fnfl_target_args a;
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
    if (argc >= 2 && strcmp(argv[1], "spec") == 0) {
        fnfl_target_args a;
        a.ple_table = ple_table;
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
                a.spec_d = atoi(rest[++i].c_str()); // SP2's switch: 1..4
            } else if (rest[i] == "--arms" && more) {
                a.arms = rest[++i]; // SP2 Amendment 3: spec ab's arms
            } else if (rest[i] == "--max-tokens" && more) {
                cur.max_tokens = atoi(rest[++i].c_str());
            } else if (rest[i] == "--and") {
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
                            "[--min-p P] [--seed S] [--max-tokens N]   (SV1: the runtime's engine as the server drives "
                            "it)\n"
                            "spec: HISTORY_IDS OUT ab|ab-dry --draft-attention sparse [--draft-subset FILE] "
                            "[--arms D:FORM[:EX][,D:FORM[:EX]..]] SETTING   (SP2: the checks, then the arms, "
                            "2:dense,4:sparse unless given, from one decode-ready state; SP4: EX sp2, sp4, h2 or xb3, "
                            "the expert gate/up mapping, sp2 unless given)\n");
            return 2;
        }
        a.ids = pos[0];
        a.out_dir = pos[1];
        a.mode = pos.size() == 3 ? pos[2] : "real";
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
        "act-test|act-bench|gd-test|gd-bench|hc-test|hc-bench|qa-test|qa-bench|qa2-test|qa2-bench|ex-calib|ex-prep|ex-"
        "test|ex-bench|"
        "ep-prep|ep-test|ep-bench|decode-check|synth-check|grid-test|xchg-test|hc2-test|hc2-bench|load-plan|load-check|"
        "xchg2-test|px1-link|layer-test|layer-bench|xp1-test|xp1-bench|dp-test|dp-bench|pd-test|hcp-test|gdp-test|qsp-"
        "test|pe-test|dcp-test|ple-test|ple-pin-test|ple-bench|state-test|"
        "prefill-test|rollback-test|spec|draft-kernel-test|draft-kernel-bench|draft-test|draft-bench|sampler-ref|"
        "samp-test|samp-bench|v-test|v-bench [DEVICE] ...\n",
        argv[0]);
    return 2;
}
