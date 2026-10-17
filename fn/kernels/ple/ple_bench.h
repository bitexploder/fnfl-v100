#pragma once

// ple-bench (E1 step 4): each launch alone (embed, products, stats, update; SR1: the commit is the frame's one-word
// selection, no copy) at B = 1 and 3, in a CUDA graph,
// warm (weights rotated over >= 200 MB; activations and code warm) and cold (R17: a 12 MB L2 flush and an
// instruction-cache eviction before each launch, (A - B) / n), 3 warm-up and 20 timed replays, median and p90; then
// the embedding and the PLE layer as one graph a card on both cards with X-PLE-STATS live (3 rounds a replay, the
// slot rotation); then the host gather over 1,000 steps of real token windows (TOKENS: a JSON list of token IDs),
// first touch and warm, with page faults. Synthetic weights (timing does not depend on their values); the host
// gather reads the model's PLE table through the host mapping. Prints one JSON object a point. mode: "all", or
// "ncu:<launch>:<B>" (one warm graph of that launch only, for ncu), or (PL1) "host:<off|on>:<B>": the host gather
// alone at B = 1 or 3 with the prefetch off or on, no device: the table evicted (fn/runtime/page_cache.h, its counts
// printed; void if it stays cached), then cold, page-cached first touch and warm passes of 1,000 steps with page faults
// and storage reads, then each cold step's e checked bitwise against the gather without the prefetch.
int fnfl_ple_bench(int dev0, int dev1, const char * mode, const char * model, const char * tokens);
