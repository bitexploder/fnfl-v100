#pragma once

// MX7's model-level checks (fnfl-drive spec run --mx7-check IDS --mx7-shares LIST; TEST ONLY, expires at MX7's
// disposition): one load, the single-card cached configuration. IDS' first 20,000 tokens are prefilled once and the
// decode-ready state kept (engine::save_ready). Then for the append (650 of IDS' next tokens, the KL amendment's) and
// each share of LIST in order ("0", "auto" or a fixed x; set between requests through set_share): the kept state
// restored and the append read as a pure extension, its decode-ready state hashed (the share 0 one written to
// OUT/state-N-share0.bin for a byte comparison against another binary's) and its prompt-final logits kept; then the
// append's 32 shorter prompts (N - 1 down to N - 32 rows), each from the kept state, their prompt-final logits kept.
// Against the list's first share-0 run: KL(share 0 || this) at the 33 prompt ends (mean and max, ticket MX7's bound
// mean <= 0.01, max <= 0.05), top-1 agreement, and whether the state and logits are bitwise equal. A second run of the
// same fixed x is compared the same way (bitwise expected: a fixed share repeats). Prompt ends, not teacher-forced
// decode rounds: decode's miss split depends on the residency each run leaves (two share-0 runs' rounds differ, MX7
// load 2), and the prompt path's logits repeat bitwise. Judged on NC1's model-level limits (2x the pinned engine's
// spread: p50 <= 0.0272, p99 <= 5.25, max <= 19.18; the engineer's disposition 2026-10-21), every position printed with
// its share-0 entropy and both top-1s; Strata's bound (mean <= 0.01, max <= 0.05) reported for reference. The share
// "shadow" (load 6): share 1's CPU experts computed and compared with the GPU's (mx7_shadow.h), the full append only;
// its output is the GPU's alone, so its state and prompt-final logits equal share 0's bitwise. Its pairs are written
// to OUT/shadow-pairs.tsv and summarized by layer, format class, job row and route slot, with the 10 worst.

#include "fn/runtime/mx7_shadow.h"
#include "fn/runtime/runtime.h"

#include <functional>
#include <string>
#include <vector>

namespace fn::runtime {

struct mx7_check_args {
    std::string ids, out_dir;
    std::vector<std::string> shares;
    std::function<bool(double share, std::string & err)> set_share; // -1 auto
    std::function<std::string()> counters;                          // the request's cache counters (JSON)
    // the share's shadow comparison (mx7_shadow.h): on or off between requests, and the pairs collected
    std::function<void(bool on)> set_shadow;
    std::function<std::vector<mx7_shadow_pair>()> take_shadow;
};

// true when every comparison passes (printed as "mx7-check: ... PASS" or FAIL)
bool mx7_check(engine & e, const mx7_check_args & a);

} // namespace fn::runtime
