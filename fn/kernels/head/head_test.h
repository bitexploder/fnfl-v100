#pragma once

#include <cstdint>

// V1: the output end on both cards in one process (dev0 holds card 0, dev1 card 1; dev0 == dev1: loopback on one
// stream, sends before reads, for Compute Sanitizer): the HC head (hc.h's head mode) on the real output_hc_* weights
// against FP64, the vocabulary logits on a slice of the real output.weight against FP64, greedy with T-GREEDY (planted
// ties within and across the cards, planted NaN rows and all-NaN rows, synthetic full-shard cases), batch invariance,
// bitwise reruns and layer 47's boundary left unchanged; DR1's draft head (draft_ids, pack_draft_head against
// dense::pack, the logits on the packed copy against FP64, greedy with the ID map and cross-card ties to the lower
// ID). mode: "all" or "san" (fewer cases). model: the deployed GGUF's first shard (read on the CPU only). Prints
// "v-test: PASS" and returns 0 when every case passes.
int fnfl_v_test(int dev0, int dev1, const char * mode, const char * model);
