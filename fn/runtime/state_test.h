#pragma once

// fnfl-drive state-test (SR1, no model load): the versioned state's commit by selection and the truncate primitive
// (fn/runtime/versions.h) through the real round frame (fn/kernels/round: descriptor, begin, commit, finish, cur) on
// both cards, with random weights and inputs at the model's per-card shapes:
//   one GDN layer: DP2's GDN input launch (qkv Q6_K with the convolution epilogue, z Q6_K, alpha and beta BF16), then
//                  gdn_core; the state and convolution history in five slots;
//   the PLE layer: products, stats (X-PLE-STATS to the peer), update; the history in five slots;
//   one QSA indexer on card 0: the round form of summary_update (raw key write and pooling), 1,024 cells.
// One graph per card per B (eager: both cards in phases, for Compute Sanitizer). Each row's inputs are a seeded
// function of its position and token, so a replay of the ledger feeds the same rows.
//
// The script: forced rounds that commit k of B rows (the test-only kind forced_k) for every B 1..4 and k 0..B, each cut
// to every j 0..k (the ledger to N_before + j), then continued with fresh tokens; recoverable errors injected before
// commit on both cards and on each card alone; a truncate inside a summary block, continued past the block's end.
// After every operation the state is checked against a clean replay: the device state is saved, every state object
// zeroed, the ledger's first N tokens replayed in forced rounds at B = 1, then compared bitwise (each card's GDN state,
// GDN convolution history and PLE history as logical version 0; the index keys of cells < N and the summaries of
// blocks < N / 4; N and the summaries' valid count), and the saved state restored. A continuation round's output rows
// (GDN gated, PLE h_out) are compared with the replay's rows at their positions. A negative control (cur moved to the
// uncut version) must be detected. PASS (exit 0) needs every check bitwise, every k and j covered at B = 1..4, the
// committed slot at all five positions, and both cards back at cur_pre after each injected error.
//
// usage: fnfl-drive state-test [all|san] [graph|eager] [DEV0] [DEV1]   (san: a short script, for the sanitizers)

int fnfl_state_test(int argc, char ** argv);
