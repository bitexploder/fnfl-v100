#include "fn/tools/cases.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern char ** environ;

namespace fn::cases {
namespace {

#ifndef FNFL_SOURCE_DIR
#    error "FNFL_SOURCE_DIR (the engine's source tree) is set by fn/CMakeLists.txt"
#endif

// the fixtures the checks read; the defaults are the paths the commands had before TFC, under $HOME, so a command with no
// argument reads what it read before
const fixture fixtures[] = {
    { "model",          "FNFL_FIXTURE_MODEL",
     "~/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
      "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",                                                                        "the target's first shard, IQ3_XXS"                                   },
    { "model-q4",       "FNFL_FIXTURE_MODEL_Q4",
     "~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",                                  "the target's first shard, UD-Q4_K_XL"                                },
    { "draft",          "FNFL_FIXTURE_DRAFT",          "~/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
     "the MTP draft, Q8_0"                                                                                                                                                                                   },
    { "grid-header",    "FNFL_FIXTURE_GRID_HEADER",    FNFL_SOURCE_DIR "/ggml/src/ggml-common.h",
     "ggml-common.h at 66e11c3 (grid-test checks its SHA-256)"                                                                                                                                               },
    { "ex-cache",       "FNFL_FIXTURE_EX_CACHE",       "~/src/flashnext/work/C1b/cache",
     "fn/quant's synthetic experts (C1b); fnfl-drive ex-prep DIR makes them"                                                                                                                                 },
    { "ep-refs",        "FNFL_FIXTURE_EP_REFS",        "~/src/flashnext/work/C1b/ep-refs-a4",
     "ep-test's FP64 references (C1b Amendment 4); fnfl-drive ep-prep makes them"                                                                                                                            },
    { "fn4-routes",     "FNFL_FIXTURE_FN4_ROUTES",     "~/src/flashnext-opt/work/FN4/20261007T170304Z/routes.txt",
     "FN4's captured routes"                                                                                                                                                                                 },
    { "c1b-ref",        "FNFL_FIXTURE_C1B_REF",        "~/src/flashnext/work/C1b/ref/out",
     "the pinned engine's decode dump (C1b), decode-check's reference"                                                                                                                                       },
    { "hx7-case",       "FNFL_FIXTURE_HX7_CASE",       "~/src/flashnext/work/QA1/fixtures/hx7-l3-w0.bin",
     "QA1's HX7 case (qa2-test, qa2-bench)"                                                                                                                                                                  },
    { "hx7-capture",    "FNFL_FIXTURE_HX7_CAPTURE",    "~/src/flashnext/work/HX7/runs/capture-58581-20261009T090240Z",
     "HX7's captured routing (ex-bench, xp1)"                                                                                                                                                                },
    { "tg1-logits",     "FNFL_FIXTURE_TG1_LOGITS",     "~/src/flashnext/work/TG1/raw/ref-decode/logits.bin.gz",
     "TG1's recorded decode logits (samp-test, samp-bench)"                                                                                                                                                  },
    { "fl2-tokens",     "FNFL_FIXTURE_FL2_TOKENS",     "~/src/flashnext/work/FL2/smoke/fl1-tokenize.out",
     "FL2's token list (ple-bench's host gather)"                                                                                                                                                            },
    { "xp1-synth",      "FNFL_FIXTURE_XP1_SYNTH",      "~/src/flashnext/work/XP1/synth",                                               "XP1's synthetic traffic"                                             },
    { "draft-q2",       "FNFL_FIXTURE_DRAFT_Q2",
     "~/src/flashnext/work/ME7/out/mtp-Qwen3.8-Flash-Next-shared-Q8_0-q2experts.gguf",                                                 "ME7's Q2 draft (state-test mu1-one)"                                 },
    { "draft-subset",   "FNFL_FIXTURE_DRAFT_SUBSET",   "~/src/flashnext/work/DR1/stageC/subset/draft-subset.ids",
     "DR1's draft vocabulary subset (split-test)"                                                                                                                                                            },
    { "ids",            "FNFL_FIXTURE_IDS",            "",                                                                             "a prompt's token IDs (model-dry, spec, prefill-test, rollback-test)" },
    { "deep-ids",       "FNFL_FIXTURE_DEEP_IDS",       "",                                                                             "joint-kl's deep prompt IDs"                                          },
    { "tg1-ref",        "FNFL_FIXTURE_TG1_REF",        "",                                                                             "ref_dump3's decode-mode dumps (model-test REF_DIR)"                  },
    { "tg1-spread",     "FNFL_FIXTURE_TG1_SPREAD",     "",                                                                             "the pinned engine's spread JSON (model-test SPREAD_JSON)"            },
    { "r2-ref-decode",  "FNFL_FIXTURE_R2_REF_DECODE",  "",                                                                             "ref_dump's decode dump (layer-test)"                                 },
    { "r2-ref-prompt",  "FNFL_FIXTURE_R2_REF_PROMPT",  "",                                                                             "ref_dump's prompt dump (layer-test)"                                 },
    { "dr1-ref-draft",  "FNFL_FIXTURE_DR1_REF_DRAFT",  "",                                                                             "ref_dump_draft's dumps (draft-test)"                                 },
    { "pb1-ref-decode", "FNFL_FIXTURE_PB1_REF_DECODE", "",                                                                             "PB1's pinned decode dumps (prefill-test model)"                      },
    { "pb1-ref-prompt", "FNFL_FIXTURE_PB1_REF_PROMPT", "",                                                                             "PB1's pinned prompt dumps (prefill-test model)"                      },
    { "pb1-real",       "FNFL_FIXTURE_PB1_REAL",       "",                                                                             "PB1's saved real HC inputs (hcp-test real=DIR)"                      },
};

constexpr need H = need::host, HD = need::host_data, G = need::gpu, GD = need::gpu_data, M = need::model,
               S = need::serving, B = need::two_box;
constexpr kind C = kind::check, T = kind::tool;

// The inventory. ID: command[.mode]; each existing command, mode and internal case of a check has an ID here. A
// command's own default devices are 0 and 1
const entry entries[] = {
    // host checks (no device)
    { "grid-test",                     C, H,  0, "fnfl-drive",       "grid-test {grid-header}",                                            "",
     "IQ grid tables and IQ4_NL values byte for byte against ggml-common.h at 66e11c3; grid_table packing; the IQ2 "
      "selector tables against their derivation"                                                                                                                                                                                                                                                                  },
    { "synth-check",                   C, H,  0, "fnfl-drive",       "synth-check",                                                        "",                                                                                              "fn/quant's synthetic-block generators"                               },
    { "sampler-ref",                   C, H,  0, "fnfl-drive",       "sampler-ref",                                                        "",                                                                                              "the CPU reference sampler's own tests (SM1 step 1)"                  },
    { "settings-test",                 C, H,  0, "fnfl-drive",       "settings test",                                                      "",
     "driver/server settings parity, scope and range refusals, read-back (ME3)"                                                                                                                                                                                                                                   },
    { "kv-format-test",                C, H,  0, "fnfl-drive",       "kv-format-test",                                                     "",
     "--fnfl-kv-format against every configuration (KV8 Amendment 1)"                                                                                                                                                                                                                                             },
    { "pipe-test",                     C, H,  0, "fnfl-drive",       "pipe-test",                                                          "",
     "the split's two-group pipeline core against a fake leader and follower (TTTS1)"                                                                                                                                                                                                                             },
    { "link-test.host",                C, H,  0, "fnfl-drive",
     "link-test leader --host --addr 127.0.0.1 --port 13391 --timeout 60 --cases "
      "decode,prompt,small,stream,refuse,retire",                                                                                          "link-test follower --host --addr 127.0.0.1 --port 13391 --timeout 60 --cases decode,prompt,small,stream,refuse,"
      "retire",                                                                                                                                                                            "the box-to-box link on host buffers over loopback, no card (TTTSBS)" },
    { "prefill-test.plan",             C, H,  0, "fnfl-drive",       "prefill-test - {out} plan",                                          "",
     "bulk prefill's chunk plan, no device (PF1)"                                                                                                                                                                                                                                                                 },
    { "ex-cpu-test.extents",           C, H,  0, "fnfl-drive",       "ex-cpu-test {out} extents",                                          "",                                                                                              "ME4's extent rule"                                                   },
    { "ex-cpu-test.mx7-f16",           C, H,  0, "fnfl-drive",       "ex-cpu-test {out} mx7-f16",                                          "",
     "MX7's prompt CPU share's conversions"                                                                                                                                                                                                                                                                       },
    { "ex-cpu-test.loan",              C, H,  0, "fnfl-drive",       "ex-cpu-test {out} loan",                                             "",                                                                                              "ME3's loan plan invariants"                                          },
    { "ex-cpu-test.short-read",        C, H,  0, "fnfl-drive",       "ex-cpu-test {out} short-read",                                       "",                                                                                              "ME3's short reads"                                                   },
    { "ex-cpu-test.refill-batches",    C, H,  0, "fnfl-drive",       "ex-cpu-test {out} refill-batches",                                   "",
     "ME3's refill batches"                                                                                                                                                                                                                                                                                       },
    { "ex-cpu-test.refill-state",      C, H,  0, "fnfl-drive",       "ex-cpu-test {out} refill-state",                                     "",                                                                                              "ME3's refill state"                                                  },
    { "ex-cpu-test.service-loop",      C, H,  0, "fnfl-drive",       "ex-cpu-test {out} service-loop",                                     "",                                                                                              "ME3's service loop"                                                  },
    { "ex-cpu-test.affinity",          C, H,  0, "fnfl-drive",       "ex-cpu-test {out} affinity",                                         "",
     "ME3's helper affinity rule, refusals and live pinning"                                                                                                                                                                                                                                                      },
    { "ex-cpu-test.prompt-plan",       C, H,  0, "fnfl-drive",       "ex-cpu-test {out} prompt-plan",                                      "",
     "ME4's prompt planner (chunks, paths, ring, loan, groups, byte ring, switch)"                                                                                                                                                                                                                                },
    { "ex-cpu-test.slots",             C, H,  0, "fnfl-drive",       "ex-cpu-test {out} slots",                                            "",                                                                                              "MU1's slot rules on the CPU"                                         },
    { "ex-cpu-test.miss-split",        C, H,  0, "fnfl-drive",       "ex-cpu-test {out} miss-split",                                       "",
     "ME5's row-aware miss split"                                                                                                                                                                                                                                                                                 },
    { "ex-cpu-test.admission",         C, H,  0, "fnfl-drive",       "ex-cpu-test {out} admission",                                        "",                                                                                              "ME9's admission records"                                             },
    { "ex-cpu-test.formats",           C, HD, 0, "fnfl-drive",       "ex-cpu-test {ex-cache} formats",                                     "",
     "the CPU miss path: pack/unpack round trips, whole experts at B = 1..5 against FP64, ME4's native views, the "
      "host drain (ME2)"                                                                                                                                                                                                                                                                                          },
    { "decode-check",                  C, HD, 0, "fnfl-drive",       "decode-check {c1b-ref}",                                             "",
     "fn/quant's decoders bitwise against the pinned engine's dump (C1b)"                                                                                                                                                                                                                                         },
    { "load-plan",                     C, HD, 0, "fnfl-drive",       "load-plan {out} {model} {draft}",                                    "",
     "the placement, the GGUF slice hashes and the dry run on the CPU (R1)"                                                                                                                                                                                                                                       },
    { "split-plan",                    C, HD, 0, "fnfl-drive",       "split-plan --model {model-q4} --draft {draft}",                      "",
     "each box's weights and state for the split (TTTS1)"                                                                                                                                                                                                                                                         },
    { "safety-test",                   C, H,  0, "fnfl-safety",      "safety-test all",                                                    "",
     "TTTSBS: the split's buffer lifetimes on the failure paths, ten cases in one process; build fnfl-safety in "
      "plain, address and thread build directories (FNFL_SANITIZE)"                                                                                                                                                                                                                                               },
    { "safety-test.normal",            C, H,  0, "fnfl-safety",      "safety-test normal",                                                 "",                                                                                              "TTTSBS case normal"                                                  },
    { "safety-test.partial",           C, H,  0, "fnfl-safety",      "safety-test partial",                                                "",                                                                                              "TTTSBS case partial"                                                 },
    { "safety-test.queued",            C, H,  0, "fnfl-safety",      "safety-test queued",                                                 "",                                                                                              "TTTSBS case queued"                                                  },
    { "safety-test.posted",            C, H,  0, "fnfl-safety",      "safety-test posted",                                                 "",                                                                                              "TTTSBS case posted"                                                  },
    { "safety-test.race",              C, H,  0, "fnfl-safety",      "safety-test race",                                                   "",                                                                                              "TTTSBS case race"                                                    },
    { "safety-test.final",             C, H,  0, "fnfl-safety",      "safety-test final",                                                  "",                                                                                              "TTTSBS case final"                                                   },
    { "safety-test.ack",               C, H,  0, "fnfl-safety",      "safety-test ack",                                                    "",                                                                                              "TTTSBS case ack"                                                     },
    { "safety-test.shutdown",          C, H,  0, "fnfl-safety",      "safety-test shutdown",                                               "",                                                                                              "TTTSBS case shutdown"                                                },
    { "safety-test.reuse",             C, H,  0, "fnfl-safety",      "safety-test reuse",                                                  "",                                                                                              "TTTSBS case reuse"                                                   },
    { "safety-test.pipe",              C, H,  0, "fnfl-safety",      "pipe-test",                                                          "",                                                                                              "the pipeline core in the safety build"                               },
    { "safety-test.link-host",         C, H,  0, "fnfl-safety",
     "link-test leader --port 13392 --timeout 60 --cases decode,prompt,small,stream,refuse,retire",                                        "link-test follower --port 13392 --timeout 60 --cases decode,prompt,small,stream,refuse,retire",
     "the link-test's host-buffer mode with the lifetime ledger and the link's test hooks (TTTSBS)"                                                                                                                                                                                                               },
    { "splice-test",                   C, H,  0, "fnfl-splice-test", "",                                                                   "",
     "the server's answer splice and the two-user scheduling policy's pure parts (BG2, MU1)"                                                                                                                                                                                                                      },

    // GPU checks on synthetic data (no model data)
    { "grid-gpu-test",                 C, G,  1, "fnfl-drive",       "grid-gpu-test 0",                                                    "",
     "the decoders' IQ2 selector recipe on the device, 360,448 cases (FMS4 item 2; split out of grid-test)"                                                                                                                                                                                                       },
    { "act-test",                      C, G,  1, "fnfl-drive",       "act-test 0",                                                         "",
     "the activation quantize kernel bitwise against act_ref (C2)"                                                                                                                                                                                                                                                },
    { "hc-test",                       C, G,  1, "fnfl-drive",       "hc-test 0",                                                          "",                                                                                              "the HC kernels at rows 1..4 (HC1)"                                   },
    { "qa-test",                       C, G,  1, "fnfl-drive",       "qa-test 0 all all",                                                  "",                                                                                              "the QSA indexer summaries, scores, selection (QA1)"                  },
    { "kv-tier-test",                  C, G,  1, "fnfl-drive",       "kv-tier-test 0 all",                                                 "",
     "the host-backed KV tier against resident (ME1)"                                                                                                                                                                                                                                                             },
    { "kv-tier-test.san",              C, G,  1, "fnfl-drive",       "kv-tier-test 0 san",                                                 "",                                                                                              "kv-tier-test's Compute Sanitizer size"                               },
    { "draft-kernel-test",             C, G,  2, "fnfl-drive",       "draft-kernel-test 0 1 all",                                          "",
     "the draft kernels against FP64 (DR1 Stage C)"                                                                                                                                                                                                                                                               },
    { "draft-kernel-test.san",         C, G,  2, "fnfl-drive",       "draft-kernel-test 0 1 san",                                          "",                                                                                              "the Compute Sanitizer form"                                          },
    { "gdnj-test",                     C, G,  1, "fnfl-drive",       "gdnj-test 0",                                                        "",                                                                                              "the joint GDN core launch against each lane's (FMS2)"                },
    { "xchg-test",                     C, G,  2, "fnfl-drive",       "xchg-test 0 1 10000 1,3",                                            "",                                                                                              "X-HC and X-HMIX (HC2)"                                               },
    { "hc2-test",                      C, G,  2, "fnfl-drive",       "hc2-test 0 1",                                                       "",                                                                                              "the two-card HC sublayer at rows 1..5 (HC2)"                         },
    { "hc2-test.san",                  C, G,  1, "fnfl-drive",       "hc2-test 0 0",                                                       "",                                                                                              "hc2-test loopback on one card (Compute Sanitizer)"                   },
    { "hcj-test",                      C, G,  2, "fnfl-drive",       "hcj-test 0 1",                                                       "",
     "the joint HC sublayer and the Q8_0 head in lanes (FMU2)"                                                                                                                                                                                                                                                    },
    { "qsaj-test",                     C, G,  2, "fnfl-drive",       "qsaj-test 0 1",                                                      "",                                                                                              "the joint round's merged QSA launches (FMS2)"                        },
    { "xchg2-test",                    C, G,  2, "fnfl-drive",       "xchg2-test 0 1 200",                                                 "",                                                                                              "X-A, X-F, X-ROUTE, X-SELECT (R2)"                                    },
    { "x16-test",                      C, G,  2, "fnfl-drive",       "x16-test 0 1",                                                       "",                                                                                              "the joint 16-bit X-A / X-F (FMS2 item 7)"                            },
    { "dp-test",                       C, G,  1, "fnfl-drive",       "dp-test 0 all full",                                                 "",                                                                                              "the dense products at rows 1..5 (DP1, DP2)"                          },
    { "dp-test.san",                   C, G,  1, "fnfl-drive",       "dp-test 0 all san",                                                  "",                                                                                              "dp-test's Compute Sanitizer size"                                    },
    { "dpj-test",                      C, G,  2, "fnfl-drive",       "dpj-test 0 full",                                                    "",
     "the joint product at U = 2..4, sampled pick (FMU2/4)"                                                                                                                                                                                                                                                       },
    { "dpj-test.san",                  C, G,  1, "fnfl-drive",       "dpj-test 0 san",                                                     "",                                                                                              "dpj-test's sanitizer form, loopback"                                 },
    { "dpj-test.tc",                   C, G,  2, "fnfl-drive",       "dpj-test 0 tc",                                                      "",                                                                                              "the router cases on the tensor cores (FMS5)"                         },
    { "dpj-test.f16",                  C, G,  2, "fnfl-drive",       "dpj-test 0 f16",                                                     "",                                                                                              "the dense FP16 switch at U = 3, 4 (FMS6)"                            },
    { "pd-test",                       C, G,  1, "fnfl-drive",       "pd-test 0 full",                                                     "",                                                                                              "the prompt dense and router products (PB1)"                          },
    { "pd-test.san",                   C, G,  1, "fnfl-drive",       "pd-test 0 san",                                                      "",                                                                                              "pd-test's sanitizer form"                                            },
    { "hcp-test",                      C, G,  1, "fnfl-drive",       "hcp-test 0 full",                                                    "",                                                                                              "the prompt HC and the exchange (PB1)"                                },
    { "hcp-test.san",                  C, G,  1, "fnfl-drive",       "hcp-test 0 san",                                                     "",                                                                                              "hcp-test's sanitizer form"                                           },
    { "gdp-test",                      C, G,  1, "fnfl-drive",       "gdp-test 0 full",                                                    "",                                                                                              "the prompt GDN against FP64 (PB1)"                                   },
    { "gdp-test.san",                  C, G,  1, "fnfl-drive",       "gdp-test 0 san",                                                     "",                                                                                              "gdp-test's sanitizer form"                                           },
    { "qsp-test",                      C, G,  1, "fnfl-drive",       "qsp-test 0 full",                                                    "",                                                                                              "the prompt QSA (PB1; KV8: --kv-format)"                              },
    { "qsp-test.san",                  C, G,  1, "fnfl-drive",       "qsp-test 0 san",                                                     "",                                                                                              "qsp-test's sanitizer form"                                           },
    { "pe-test",                       C, G,  1, "fnfl-drive",       "pe-test 0 full",                                                     "",                                                                                              "the prompt embedding and PLE (PB1)"                                  },
    { "pe-test.san",                   C, G,  1, "fnfl-drive",       "pe-test 0 san",                                                      "",                                                                                              "pe-test's sanitizer form"                                            },
    { "dcp-test",                      C, G,  1, "fnfl-drive",       "dcp-test 0 full",                                                    "",                                                                                              "the draft's catch-up pieces (PB1)"                                   },
    { "dcp-test.san",                  C, G,  1, "fnfl-drive",       "dcp-test 0 san",                                                     "",                                                                                              "dcp-test's sanitizer form"                                           },
    { "ple-pin-test.unit",             C, G,  1, "fnfl-drive",       "ple-pin-test unit {out} 0",                                          "",
     "ple_host's residency on a file over RLIMIT_MEMLOCK (PN1)"                                                                                                                                                                                                                                                   },
    { "split-test",                    C, G,  2, "fnfl-drive",       "split-test all",                                                     "",
     "one box against the split at k = 3, 27, 45, bitwise, synthetic weights (TTTS1, TTTS8)"                                                                                                                                                                                                                      },
    { "split-test.san",                C, G,  1, "fnfl-drive",       "split-test san",                                                     "",                                                                                              "split-test's sanitizer form"                                         },
    { "split-test.pieces",             C, G,  2, "fnfl-drive",       "split-test all --cases pieces",                                      "",
     "MUP2 piece 1: reads in units and pieces (level 1, 2,048, 512 and 64 rows) bitwise today's lone read, one box "
      "and "
      "the split at k = 27"                                                                                                                                                                                                                                                                                       },
    { "split-test.latch",              C, G,  2, "fnfl-drive",       "split-test all --cases latch",                                       "",
     "PRD1: a forced joint-capture failure on the split's front: the latch, the refusal said once, later groups "
     "alone"                                                                                                                                                                                                                                                                                                      },
    { "split-test.reads",              C, G,  2, "fnfl-drive",       "split-test all --cases reads",                                       "",
     "PRD1: small reads whole beside a writer, no-writer progress and the idle drain, the front freed at its end, the "
      "crossing received ahead; bitwise each read's lone read, the split at k = 27"                                                                                                                                                                                                                               },
    { "state-test",                    C, G,  2, "fnfl-drive",       "state-test all graph 0 1",                                           "",
     "commit by selection and truncate through the round frame (SR1)"                                                                                                                                                                                                                                             },
    { "state-test.eager",              C, G,  2, "fnfl-drive",       "state-test all eager 0 1",                                           "",                                                                                              "state-test without CUDA graphs"                                      },
    { "state-test.san",                C, G,  2, "fnfl-drive",       "state-test san eager 0 1",                                           "",                                                                                              "state-test's sanitizer form"                                         },

    // GPU checks reading fixture files (no model load)
    { "state-test.mu1.graph",          C, GD, 2, "fnfl-drive",       "state-test mu1 graph 0 1",                                           "",
     "MU1's pool and frames; parts A and B (the dry load reads the model's metadata)"                                                                                                                                                                                                                             },
    { "state-test.mu1.eager",          C, GD, 2, "fnfl-drive",       "state-test mu1 eager 0 1",                                           "",                                                                                              "mu1 without CUDA graphs"                                             },
    { "state-test.mu1.read",           C, GD, 2, "fnfl-drive",       "state-test mu1 read 0 1",                                            "",                                                                                              "mu1's read case"                                                     },
    { "state-test.mu1.joint",          C, GD, 2, "fnfl-drive",       "state-test mu1 joint 0 1",                                           "",                                                                                              "mu1's joint rounds (FMS2)"                                           },
    { "state-test.mu1.joint-eager",    C, GD, 2, "fnfl-drive",       "state-test mu1 joint-eager 0 1",                                     "",
     "mu1's joint rounds without graphs"                                                                                                                                                                                                                                                                          },
    { "state-test.mu1.joint-san",      C, GD, 2, "fnfl-drive",       "state-test mu1 joint-san 0 1",                                       "",
     "mu1's joint rounds, sanitizer form"                                                                                                                                                                                                                                                                         },
    { "state-test.mu1.catchup",        C, GD, 2, "fnfl-drive",       "state-test mu1 catchup 0 1",                                         "",                                                                                              "the joint catch-up (FMS2)"                                           },
    { "state-test.mu1.catchup-eager",  C, GD, 2, "fnfl-drive",       "state-test mu1 catchup-eager 0 1",                                   "",
     "the joint catch-up without graphs"                                                                                                                                                                                                                                                                          },
    { "state-test.mu1-one.graph",      C, GD, 1, "fnfl-drive",       "state-test mu1-one graph 0",                                         "",
     "MXMU1: two slots on one card with the expert cache and the host-backed KV (reads {draft-q2})"                                                                                                                                                                                                               },
    { "state-test.mu1-one.eager",      C, GD, 1, "fnfl-drive",       "state-test mu1-one eager 0",                                         "",                                                                                              "mu1-one without graphs"                                              },
    { "state-test.mu1-one.c0",         C, GD, 1, "fnfl-drive",       "state-test mu1-one c0 0",                                            "",
     "mu1-one's refusals before any allocation"                                                                                                                                                                                                                                                                   },
    { "gd-test",                       C, GD, 1, "fnfl-drive",       "gd-test 0 all {model}",                                              "",
     "gdn_core on layers 10 and 40's weights read on the CPU (GD1; ME11 replay)"                                                                                                                                                                                                                                  },
    { "gd-test.san",                   C, GD, 1, "fnfl-drive",       "gd-test 0 san {model}",                                              "",                                                                                              "gd-test's sanitizer form"                                            },
    { "ple-test",                      C, GD, 2, "fnfl-drive",       "ple-test all 0 1 {model}",                                           "",                                                                                              "the embedding, host PLE gather, PLE (E1)"                            },
    { "ple-test.san",                  C, GD, 1, "fnfl-drive",       "ple-test san 0 0 {model}",                                           "",                                                                                              "ple-test loopback, sanitizer form"                                   },
    { "ple-test.q8",                   C, GD, 2, "fnfl-drive",       "ple-test q8 0 1 {model-q4}",                                         "",                                                                                              "the Q8_0 PLE key/value (TTTS3)"                                      },
    { "qa2-test",                      C, GD, 1, "fnfl-drive",       "qa2-test 0 all {model} {hx7-case}",                                  "",                                                                                              "QSA attend against FP64 (QA2)"                                       },
    { "qa2-test.san",                  C, GD, 1, "fnfl-drive",       "qa2-test 0 san {model} {hx7-case}",                                  "",                                                                                              "qa2-test's sanitizer form"                                           },
    { "ex-test",                       C, GD, 1, "fnfl-drive",       "ex-test 0 1,2,3,4,5 all {ex-cache}",                                 "",
     "every deployed expert instantiation with the shared expert (EX2)"                                                                                                                                                                                                                                           },
    { "ex-test.san",                   C, GD, 1, "fnfl-drive",       "ex-test 0 1,4,5 san {ex-cache}",                                     "",
     "ex-test's sanitizer covering set (B 4 and 5: ME10's pair and finish compare there)"                                                                                                                                                                                                                         },
    { "ex-test.q4",                    C, GD, 1, "fnfl-drive",       "ex-test 0 1,2,3,4,5 q4 {ex-cache}",                                  "",
     "the UD-Q4_K_XL instance lists (ME1)"                                                                                                                                                                                                                                                                        },
    { "ex-test.q2d",                   C, GD, 1, "fnfl-drive",       "ex-test 0 1,2,3,4,5 q2d {ex-cache}",                                 "",
     "the Q2 draft's own instance lists, gate/up and down (ME7)"                                                                                                                                                                                                                                                  },
    { "ep-test",                       C, GD, 1, "fnfl-drive",       "ep-test 0 128,1536 all full {ep-refs} {ex-cache} {fn4-routes}",      "",
     "the prompt expert kernels against FP64 (EP1)"                                                                                                                                                                                                                                                               },
    { "ep-test.san",                   C, GD, 1, "fnfl-drive",       "ep-test 0 128 all san {ep-refs} {ex-cache} {fn4-routes}",            "",
     "ep-test's sanitizer form"                                                                                                                                                                                                                                                                                   },
    { "hcmix-test",                    C, GD, 2, "fnfl-drive",       "hcmix-test 0 1 {model-q4}",                                          "",
     "TTTS2's mixed HC down on UD-Q4_K_XL's real modules (TTTS3 item 7)"                                                                                                                                                                                                                                          },
    { "dpj-test.q4",                   C, GD, 2, "fnfl-drive",       "dpj-test 0 q4",                                                      "",
     "UD-Q4_K_XL's and the draft's real rows, Q8_0 instances (TTTS6; reads {model-q4} and {draft})"                                                                                                                                                                                                               },
    { "pd-test.q8",                    C, GD, 1, "fnfl-drive",       "pd-test 0 q8={model-q4}",                                            "",                                                                                              "the EX1-down Q8_0 on real downs (TTTS3)"                             },
    { "hcp-test.q8",                   C, GD, 1, "fnfl-drive",       "hcp-test 0 q8={model-q4}",                                           "",
     "hct_convert from the Q8_0 halves (TTTS3)"                                                                                                                                                                                                                                                                   },
    { "hcp-test.real",                 C, GD, 1, "fnfl-drive",       "hcp-test 0 real={pb1-real}",                                         "",
     "PB3's tensor HC on PB1's saved real inputs"                                                                                                                                                                                                                                                                 },
    { "ex-test.q4-real",               C, GD, 1, "fnfl-drive",       "ex-test 0 1,2,3,4,5 q4@1 {ex-cache}",                                "",
     "layer 1's instances on its real routed experts from UD-Q4_K_XL, read on the CPU (TTTS3 step 8)"                                                                                                                                                                                                             },
    { "v-test",                        C, GD, 2, "fnfl-drive",       "v-test all {model}",                                                 "",                                                                                              "the head, the logits and greedy (V1)"                                },
    { "v-test.san",                    C, GD, 1, "fnfl-drive",       "v-test san {model}",                                                 "",                                                                                              "v-test loopback on device 0"                                         },
    { "v-test.q8",                     C, GD, 1, "fnfl-drive",       "v-test q8 {model-q4}",                                               "",                                                                                              "the Q8_0 logits and draft head (TTTS3)"                              },
    { "samp-test",                     C, GD, 2, "fnfl-drive",       "samp-test all 0 1 {tg1-logits}",                                     "",
     "the sampler against the reference on recorded logits (SM1)"                                                                                                                                                                                                                                                 },
    { "samp-test.quick",               C, GD, 2, "fnfl-drive",       "samp-test quick 0 1 {tg1-logits}",                                   "",                                                                                              "samp-test with fewer draws"                                          },
    { "samp-test.san",                 C, GD, 1, "fnfl-drive",       "samp-test san 0 0 {tg1-logits}",                                     "",                                                                                              "samp-test's sanitizer form"                                          },
    { "model-dry",                     C, GD, 2, "fnfl-drive",       "model-dry {ids} {out} graph --model {model}",                        "",
     "TG1/SR1 plumbing with zero weights"                                                                                                                                                                                                                                                                         },
    { "model-dry.eager",               C, GD, 2, "fnfl-drive",       "model-dry {ids} {out} eager --model {model}",                        "",                                                                                              "without graphs"                                                      },
    { "model-dry.sample",              C, GD, 2, "fnfl-drive",       "model-dry {ids} {out} sample --model {model}",                       "",
     "the sampler path, zero weights (SM1)"                                                                                                                                                                                                                                                                       },
    { "model-dry.spec",                C, GD, 2, "fnfl-drive",       "model-dry {ids} {out} spec --draft-attention dense --model {model}",
     "",                                                                                                                                                                                                                                    "the draft rounds, zero weights (DR1)"                                },
    { "prefill-test.dry",              C, GD, 2, "fnfl-drive",       "prefill-test {ids} {out} dry --draft-attention dense",               "",
     "bulk prefill against the decode path, zero weights (PF1)"                                                                                                                                                                                                                                                   },
    { "prefill-test.dry-eager",        C, GD, 2, "fnfl-drive",       "prefill-test {ids} {out} dry-eager --draft-attention dense",
     "",                                                                                                                                                                                                                                    "the same without graphs"                                             },
    { "prefill-test.vcheck",           C, GD, 2, "fnfl-drive",       "prefill-test {ids} {out} vcheck --draft-attention dense",            "",
     "the draft's catch-up by forced rounds and by the bulk span against each other and FP64 (a dry load)"                                                                                                                                                                                                        },
    { "prefill-test.model-plumb",      C, GD, 2, "fnfl-drive",
     "prefill-test {ids} {out} model-plumb --draft-attention dense --ref-decode {pb1-ref-decode} --ref-prompt "
      "{pb1-ref-prompt}",                                                                                                                  "",                                                                                              "prefill-test model's flow on zero weights before its one real load"  },
    { "rollback-test.dry",             C, GD, 2, "fnfl-drive",       "rollback-test {ids} {out} dry",                                      "",                                                                                              "SR1's script, zero weights"                                          },
    { "rollback-test.mu1-dry",         C, GD, 2, "fnfl-drive",       "rollback-test {ids} {out} mu1-dry",                                  "",
     "MU1's rollback, zero weights"                                                                                                                                                                                                                                                                               },
    { "joint-kl.dry",                  C, GD, 2, "fnfl-drive",       "joint-kl {deep-ids} {out} dry",                                      "",
     "joint-kl's plumbing, zero weights"                                                                                                                                                                                                                                                                          },
    { "spec.dry",                      C, GD, 2, "fnfl-drive",       "spec {ids} {out} dry --draft-attention dense",                       "",
     "real sampled speculation's plumbing, zero weights (SP1 C1)"                                                                                                                                                                                                                                                 },
    { "spec.run-dry",                  C, GD, 2, "fnfl-drive",       "spec {ids} {out} run-dry --draft-attention dense",                   "",
     "spec run's plumbing, zero weights (SV1)"                                                                                                                                                                                                                                                                    },
    { "spec.ab-dry",                   C, GD, 2, "fnfl-drive",       "spec {ids} {out} ab-dry --draft-attention sparse",                   "",
     "SP2's arms' plumbing, zero weights"                                                                                                                                                                                                                                                                         },

    // model loads
    { "load-check",                    C, M,  2, "fnfl-drive",       "load-check {out} 198720 {model} {draft}",                            "",
     "the loader on both cards (R1)"                                                                                                                                                                                                                                                                              },
    { "layer-test",                    C, M,  2, "fnfl-drive",       "layer-test {r2-ref-decode} {r2-ref-prompt} {out} full",              "",
     "layers 10 and 11 against the pinned dumps (R2)"                                                                                                                                                                                                                                                             },
    { "layer-test.san",                C, M,  2, "fnfl-drive",       "layer-test {r2-ref-decode} {r2-ref-prompt} {out} san",               "",
     "layer-test, one pass over 64 positions"                                                                                                                                                                                                                                                                     },
    { "model-test",                    C, M,  2, "fnfl-drive",       "model-test {tg1-ref} {tg1-spread} {ids} {out} full",                 "",
     "the whole target pass against ref_dump3 (TG1)"                                                                                                                                                                                                                                                              },
    { "model-test.san",                C, M,  2, "fnfl-drive",       "model-test {tg1-ref} {tg1-spread} {ids} {out} san",                  "",
     "16 positions in phases"                                                                                                                                                                                                                                                                                     },
    { "model-test.race",               C, M,  2, "fnfl-drive",       "model-test {tg1-ref} {tg1-spread} {ids} {out} race",                 "",
     "layers 8..11 for racecheck"                                                                                                                                                                                                                                                                                 },
    { "draft-test",                    C, M,  2, "fnfl-drive",       "draft-test {dr1-ref-draft} {ids} {out} --draft-attention dense",     "",
     "the draft block against ref_dump_draft (DR1)"                                                                                                                                                                                                                                                               },
    { "prefill-test.model",            C, M,  2, "fnfl-drive",
     "prefill-test {ids} {out} model --draft-attention dense --ref-decode {pb1-ref-decode} --ref-prompt "
      "{pb1-ref-prompt}",                                                                                                                  "",                                                                                              "bulk prefill against the pinned dumps, one load (PB1)"               },
    { "rollback-test",                 C, M,  2, "fnfl-drive",       "rollback-test {ids} {out} real",                                     "",
     "scripted commits and truncates against a clean replay (SR1)"                                                                                                                                                                                                                                                },
    { "rollback-test.retain",          C, M,  2, "fnfl-drive",       "rollback-test {ids} {out} retain",                                   "",                                                                                              "SR1's retain form"                                                   },
    { "rollback-test.mu1",             C, M,  2, "fnfl-drive",       "rollback-test {ids} {out} mu1",                                      "",                                                                                              "MU1's multi-slot rollback"                                           },
    { "rollback-test.mu1-joint",       C, M,  2, "fnfl-drive",       "rollback-test {ids} {out} mu1-joint",                                "",                                                                                              "mu1's joint form"                                                    },
    { "rollback-test.mu1-joint-eager", C, M,  2, "fnfl-drive",       "rollback-test {ids} {out} mu1-joint-eager",                          "",
     "mu1-joint without graphs"                                                                                                                                                                                                                                                                                   },
    { "joint-kl",                      C, M,  2, "fnfl-drive",       "joint-kl {deep-ids} {out} real",                                     "",
     "KL(lone || joint) of four lanes (FMS2)"                                                                                                                                                                                                                                                                     },
    { "spec.real",                     C, M,  2, "fnfl-drive",       "spec {ids} {out} real --draft-attention dense",                      "",
     "real sampled speculation's checks C4a-C4f in one load (SP1)"                                                                                                                                                                                                                                                },
    { "link-test",                     C, B,  2, "fnfl-drive",       "link-test leader --addr ADDR --port P --timeout 60 --cases all",
     "link-test follower --addr ADDR --port P --timeout 60 --cases all (on the other box)",                                                                                                                                                 "the box-to-box link with card buffers (TTTS1); one process a box"    },
    { "link-test.time",                T, B,  2, "fnfl-drive",       "link-test leader --addr ADDR --port P --timeout 60 --cases time",
     "link-test follower --addr ADDR --port P --timeout 60 --cases time (on the other box)",                                                                                                                                                "the link's own cost at the stage's sizes: a step group's hand-off at U = 2 and 4 (--time-rows R, default 3) "
      "and a whole prompt chunk's crossing; prints timings, never in all (TTTS7)"                                                                                                                                     },

    // serving (a server with the chosen model; tools/server/tests, the curated FNFL suite)
    { "serving",                       C, S,  2, "pytest",           "tools/server/tests/tests.sh -m 'not startup'",                       "",
     "one FNFL server, one model load: HTTP, auth, streaming, cancellation, errors, tool-call round trips, request "
      "isolation and the unsupported options' refusals (TFC item 5; tools/server/tests/README.md)"                                                                                                                                                                                                                },
    { "serving.startup",               C, S,  0, "pytest",           "tools/server/tests/tests.sh -m startup",                             "",
     "the server binary refuses --embedding, --reranking, --lora and --mmproj at launch; no card, no model data"                                                                                                                                                                                                  },

    // optional tools: fixture generators, probes, benches and prototypes (not checks)
    { "ex-prep",                       T, HD, 0, "fnfl-drive",       "ex-prep {ex-cache}",                                                 "",                                                                                              "makes the synthetic experts (C1b)"                                   },
    { "ep-prep",                       T, HD, 0, "fnfl-drive",       "ep-prep {ep-refs} {ex-cache} {fn4-routes} all",                      "",
     "makes ep-test's FP64 references"                                                                                                                                                                                                                                                                            },
    { "ex-cpu-test.throughput",        T, HD, 0, "fnfl-drive",       "ex-cpu-test {ex-cache} throughput",                                  "",
     "a CPU throughput diagnostic line per deployed class; not a check"                                                                                                                                                                                                                                           },
    { "settings",                      T, H,  0, "fnfl-drive",       "settings",                                                           "",                                                                                              "lists fn/cache_settings.h's settings"                                },
    { "ple-pin-test.probe",            T, G,  2, "fnfl-drive",       "ple-pin-test probe {out} 0",                                         "",
     "prints what the driver answers about read-only host registration; exits 0 whatever it sees"                                                                                                                                                                                                                 },
    { "stage",                         T, M,  2, "fnfl-drive",       "stage --split-layer K --draft-attention sparse",                     "",
     "the split's follower process (serving)"                                                                                                                                                                                                                                                                     },
    { "generate",                      T, M,  2, "fnfl-drive",       "generate {ids} {out}",                                               "",                                                                                              "the prompt, then greedy or sampled (TG1, SM1)"                       },
    { "model-bench",                   T, M,  2, "fnfl-drive",       "model-bench {ids} {out}",                                            "",                                                                                              "the target pass timed (TG1, SF1, TR1)"                               },
    { "draft-bench",                   T, M,  2, "fnfl-drive",       "draft-bench {ids} {out} --draft-attention dense",                    "",
     "the draft step and catch-up timed (DR1)"                                                                                                                                                                                                                                                                    },
    { "spec.run",                      T, M,  2, "fnfl-drive",       "spec {ids} {out} run --draft-attention dense",                       "",
     "the runtime as the server drives it (SV1); the full-loop measurement driver"                                                                                                                                                                                                                                },
    { "spec.ab",                       T, M,  2, "fnfl-drive",       "spec {ids} {out} ab --draft-attention sparse",                       "",                                                                                              "SP2's arms"                                                          },
    { "act-bench",                     T, G,  1, "fnfl-bench",       "act-bench 0",                                                        "",                                                                                              "C2's bench"                                                          },
    { "hc-bench",                      T, G,  1, "fnfl-bench",       "hc-bench 0",                                                         "",                                                                                              "HC1's bench"                                                         },
    { "hc2-bench",                     T, G,  2, "fnfl-bench",       "hc2-bench 0 1",                                                      "",                                                                                              "HC2/HC4/TTTS4's bench"                                               },
    { "qa-bench",                      T, G,  1, "fnfl-bench",       "qa-bench 0",                                                         "",                                                                                              "QA1's bench"                                                         },
    { "qa2-bench",                     T, GD, 1, "fnfl-bench",       "qa2-bench 0 all {hx7-case}",                                         "",                                                                                              "QA2's bench"                                                         },
    { "gd-bench",                      T, G,  1, "fnfl-bench",       "gd-bench 0",                                                         "",                                                                                              "GD1's bench"                                                         },
    { "ple-bench",                     T, GD, 2, "fnfl-bench",       "ple-bench all 0 1 {model} {fl2-tokens}",                             "",                                                                                              "E1's bench"                                                          },
    { "draft-kernel-bench",            T, G,  2, "fnfl-bench",       "draft-kernel-bench 0 1",                                             "",                                                                                              "DR1 Stage C's kernel bench"                                          },
    { "ex-calib",                      T, G,  1, "fnfl-bench",       "ex-calib 0",                                                         "",                                                                                              "EX1's machine-rate calibration"                                      },
    { "ex-bench",                      T, GD, 1, "fnfl-bench",       "ex-bench 0 all {ex-cache} {hx7-capture}",                            "",                                                                                              "EX2/XB2/SP4's bench"                                                 },
    { "ep-bench",                      T, GD, 1, "fnfl-bench",       "ep-bench 0 all {ex-cache} {fn4-routes}",                             "",                                                                                              "EP1's bench"                                                         },
    { "dp-bench",                      T, G,  1, "fnfl-bench",       "dp-bench 0",                                                         "",                                                                                              "DP1/DP2's bench"                                                     },
    { "dpj-test.q4p",                  T, GD, 1, "fnfl-drive",       "dpj-test 0 q4p",                                                     "",                                                                                              "TTTS6's served forms once, for ncu"                                  },
    { "dpj-test.q4bench",              T, GD, 1, "fnfl-drive",       "dpj-test 0 q4bench",                                                 "",                                                                                              "TTTS6's kernel points"                                               },
    { "v-bench",                       T, G,  2, "fnfl-bench",       "v-bench 0 1",                                                        "",                                                                                              "V1's bench"                                                          },
    { "samp-bench",                    T, GD, 2, "fnfl-bench",       "samp-bench 0 1",                                                     "",                                                                                              "SM1's bench (reads {tg1-logits})"                                    },
    { "layer-bench",                   T, M,  2, "fnfl-bench",       "layer-bench {r2-ref-decode} {r2-ref-prompt} {out}",                  "",                                                                                              "R2's bench"                                                          },
    { "px1-link",                      T, GD, 2, "fnfl-bench",       "px1-link 0 1 all {ex-cache} {fn4-routes}",                           "",                                                                                              "PX1 Part A's link bench"                                             },
    { "xp1-test",                      T, GD, 2, "fnfl-bench",       "xp1-test 0 1 --synth {xp1-synth} --hx7 {hx7-capture}",               "",
     "XP1's coordination prototype's checks (a prototype, not an engine check)"                                                                                                                                                                                                                                   },
    { "xp1-bench",                     T, GD, 2, "fnfl-bench",       "xp1-bench 0 1 {out} --targets FILE --calib FILE",                    "",                                                                                              "XP1's pair timed"                                                    },
};

const fixture * find(const std::string & name) {
    for (const fixture & f : fixtures) {
        if (name == f.name) {
            return &f;
        }
    }
    return nullptr;
}

bool exists(const std::string & path) {
    struct stat st;
    return !path.empty() && stat(path.c_str(), &st) == 0;
}

bool is_host(const entry & e) {
    return e.n == need::host || e.n == need::host_data;
}

// a group name, an ID, an ID prefix ending in '.' or '*', or "all"; checks only unless tools are named
bool selected(const entry & e, const std::string & filter) {
    if (filter.empty() || filter == "all") {
        return true;
    }
    if (filter == "checks" || filter == "tools") {
        return (e.k == kind::check) == (filter == "checks");
    }
    if (filter == need_name(e.n)) {
        return true;
    }
    if (filter == "host" && e.n == need::host_data) {
        return true;
    }
    if (filter == "gpu" && e.n == need::gpu_data) {
        return true;
    }
    if (filter.back() == '*') {
        return std::string(e.id).rfind(filter.substr(0, filter.size() - 1), 0) == 0;
    }
    return filter == e.id || std::string(e.id).rfind(filter + ".", 0) == 0;
}

// the arguments with {name} replaced by the fixture's path and {out} by out; the fixtures named, in missing
std::vector<std::string> expand(const std::string & args, const std::string & out, std::vector<std::string> & missing) {
    std::vector<std::string> v;
    std::string              cur;
    bool                     any = false;
    for (size_t i = 0; i <= args.size(); ++i) {
        if (i == args.size() || args[i] == ' ') {
            if (any) {
                v.push_back(cur);
            }
            cur.clear();
            any = false;
            continue;
        }
        any = true;
        if (args[i] == '{') {
            const size_t      j    = args.find('}', i);
            const std::string name = args.substr(i + 1, j - i - 1);
            if (name == "out") {
                cur += out;
            } else {
                const std::string p = fixture_path(name);
                if (!exists(p)) {
                    missing.push_back(name + (p.empty() ? " (unset: " : " (") +
                                      (p.empty() ? std::string(find(name) ? find(name)->env : "?") + ")" : p + ")"));
                }
                cur += p;
            }
            i = j;
            continue;
        }
        cur += args[i];
    }
    return v;
}

std::string dir_of(const char * self) {
    char          buf[4096];
    const ssize_t n  = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    std::string   p  = n > 0 ? std::string(buf, size_t(n)) : std::string(self);
    const size_t  sl = p.rfind('/');
    return sl == std::string::npos ? "." : p.substr(0, sl);
}

// spawn bin with args, its stdout and stderr appended to log, CUDA_VISIBLE_DEVICES empty (a host check that reaches a
// card fails instead of using it); the child's PID, or -1
pid_t spawn(const std::string & bin, const std::vector<std::string> & args, const std::string & log) {
    std::vector<std::string> env_s;
    for (char ** e = environ; *e != nullptr; ++e) {
        if (strncmp(*e, "CUDA_VISIBLE_DEVICES=", 21) != 0) {
            env_s.emplace_back(*e);
        }
    }
    env_s.emplace_back("CUDA_VISIBLE_DEVICES=");
    std::vector<char *> envp, argv;
    for (std::string & s : env_s) {
        envp.push_back(s.data());
    }
    envp.push_back(nullptr);
    std::vector<std::string> a = args;
    a.insert(a.begin(), bin);
    for (std::string & s : a) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    pid_t     pid = -1;
    const int rc  = posix_spawn(&pid, bin.c_str(), &fa, nullptr, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&fa);
    return rc == 0 ? pid : -1;
}

// the exit status of pid, or -1 when it was killed at the timeout (by its PID) or ended on a signal
int wait_for(pid_t pid, std::chrono::steady_clock::time_point until, bool & timed_out) {
    int st = 0;
    while (true) {
        const pid_t r = waitpid(pid, &st, WNOHANG);
        if (r == pid) {
            return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
        }
        if (r < 0) {
            return -1;
        }
        if (std::chrono::steady_clock::now() > until) {
            timed_out = true;
            kill(pid, SIGKILL);
            waitpid(pid, &st, 0);
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

std::string last_line(const std::string & log) {
    std::string last, line;
    if (FILE * f = fopen(log.c_str(), "r")) {
        char b[1024];
        while (fgets(b, sizeof(b), f) != nullptr) {
            line = b;
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                line.pop_back();
            }
            if (!line.empty()) {
                last = line;
            }
        }
        fclose(f);
    }
    return last;
}

// a default under the home directory: ~/ becomes $HOME/ (unchanged without HOME)
std::string home_path(const char * p) {
    const char * home = getenv("HOME");
    if (p[0] == '~' && p[1] == '/' && home != nullptr && home[0] != '\0') {
        return std::string(home) + (p + 1);
    }
    return std::string(p);
}

}  // namespace

std::string fixture_path(const std::string & name) {
    const fixture * f = find(name);
    if (f == nullptr) {
        return "";
    }
    const char * v = getenv(f->env);
    return v != nullptr && v[0] != '\0' ? std::string(v) : home_path(f->path);
}

const char * need_name(need n) {
    switch (n) {
        case need::host:
            return "host";
        case need::host_data:
            return "host-data";
        case need::gpu:
            return "gpu";
        case need::gpu_data:
            return "gpu-data";
        case need::model:
            return "model";
        case need::serving:
            return "serving";
        case need::two_box:
            return "two-box";
    }
    return "?";
}

int list(const std::string & filter) {
    int32_t n = 0;
    printf("%-30s %-5s %-9s %-5s %-16s %s\n", "ID", "KIND", "NEEDS", "CARDS", "TOOL", "ARGUMENTS / COVERS");
    for (const entry & e : entries) {
        if (!selected(e, filter)) {
            continue;
        }
        ++n;
        printf("%-30s %-5s %-9s %-5d %-16s %s\n", e.id, e.k == kind::check ? "check" : "tool", need_name(e.n), e.cards,
               e.tool, e.args);
        if (e.peer[0] != '\0') {
            printf("%-69s with: %s\n", "", e.peer);
        }
        printf("%-69s %s\n", "", e.covers);
    }
    printf(
        "cases: %d listed. host and host-data checks run with: fnfl-drive cases run [FILTER]. Cards: a GPU process "
        "holds gpu.lock, gpu0.lock and gpu1.lock; nothing here takes them.\n",
        n);
    return n > 0 ? 0 : 2;
}

int list_fixtures() {
    for (const fixture & f : fixtures) {
        const std::string p = fixture_path(f.name);
        printf("%-15s %-28s %-8s %s\n", f.name, f.env,
               p.empty() ? "unset" :
               exists(p) ? "present" :
                           "missing",
               p.empty() ? f.what : p.c_str());
    }
    return 0;
}

int run(const char * self, const std::string & filter, const std::string & out_dir, int32_t timeout_s) {
    const std::string bin_dir = dir_of(self);
    if (mkdir(out_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "cases run: cannot make %s\n", out_dir.c_str());
        return 2;
    }
    int32_t n_pass = 0, n_fail = 0, n_not = 0, n_sel = 0;
    for (const entry & e : entries) {
        if (!selected(e, filter) || e.k != kind::check) {
            continue;
        }
        if (!is_host(e)) {
            if (filter != "all" && !filter.empty()) {
                printf("cases: %s: not run here (%s, %d card(s)): its command needs the ticket's locks\n", e.id,
                       need_name(e.n), e.cards);
            }
            continue;
        }
        ++n_sel;
        const std::string        dir = out_dir + "/" + e.id;
        std::vector<std::string> missing;
        mkdir(dir.c_str(), 0755);
        const std::vector<std::string> args = expand(e.args, dir, missing);
        const std::vector<std::string> peer = expand(e.peer, dir, missing);
        const std::string              bin  = bin_dir + "/" + e.tool;
        if (!exists(bin)) {
            missing.push_back("tool " + bin + " (not built)");
        }
        if (!missing.empty()) {
            std::string m;
            for (const std::string & s : missing) {
                m += (m.empty() ? "" : ", ") + s;
            }
            printf("cases: %-30s NOT RUN: missing %s\n", e.id, m.c_str());
            ++n_not;
            fflush(stdout);
            continue;
        }
        const std::string log   = out_dir + "/" + e.id + ".log";
        const auto        until = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        remove(log.c_str());
        const pid_t pp = peer.empty() ? 0 : spawn(bin, peer, log + ".peer");
        if (!peer.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));  // the follower listens first
        }
        const pid_t p       = spawn(bin, args, log);
        bool        timed   = false;
        const int   rc      = p > 0 ? wait_for(p, until, timed) : -1;
        const int   rc_peer = pp > 0 ? wait_for(pp, until, timed) : (peer.empty() ? 0 : -1);
        const bool  pass    = rc == 0 && rc_peer == 0;
        n_pass += pass ? 1 : 0;
        n_fail += pass ? 0 : 1;
        printf("cases: %-30s %s", e.id, pass ? "PASS" : "FAIL");
        if (!pass) {
            printf(" (exit %d%s%s): %s", rc, peer.empty() ? "" : (", peer exit " + std::to_string(rc_peer)).c_str(),
                   timed ? ", killed at the timeout" : "", last_line(log).c_str());
        }
        printf("\n");
        fflush(stdout);
    }
    printf("cases: %d host checks: %d passed, %d failed, %d not run (logs in %s)\n", n_sel, n_pass, n_fail, n_not,
           out_dir.c_str());
    return n_sel == 0 ? 2 : n_fail > 0 ? 1 : n_not > 0 ? 3 : 0;
}

int main(int argc, char ** argv) {
    std::vector<std::string> pos;
    std::string              out;
    int32_t                  timeout_s = 3600;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out = argv[++i];
        } else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
            timeout_s = atoi(argv[++i]);
        } else {
            pos.emplace_back(argv[i]);
        }
    }
    if (!pos.empty() && pos[0] == "fixtures" && pos.size() == 1) {
        return list_fixtures();
    }
    if (!pos.empty() && pos[0] == "run" && pos.size() <= 2 && timeout_s > 0) {
        if (out.empty()) {
            char tmpl[] = "/tmp/fnfl-cases-XXXXXX";
            if (mkdtemp(tmpl) == nullptr) {
                fprintf(stderr, "cases run: cannot make a directory under /tmp\n");
                return 2;
            }
            out = tmpl;
        }
        return run(argv[0], pos.size() == 2 ? pos[1] : "host", out, timeout_s);
    }
    if (pos.size() <= 1) {
        return list(pos.empty() ? "all" : pos[0]);
    }
    fprintf(stderr,
            "usage: fnfl-drive cases [all|checks|tools|host|host-data|gpu|gpu-data|model|serving|two-box|ID|PREFIX*] | "
            "cases fixtures | cases run [FILTER] [--out DIR] [--timeout S]\n");
    return 2;
}

}  // namespace fn::cases
