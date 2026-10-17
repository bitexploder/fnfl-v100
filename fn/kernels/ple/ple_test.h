#pragma once

// ple-test (E1): the host n-gram gather, the token embedding and the PLE layer on both cards against their references.
//   host:  the 16 row indices bitwise against an independent CPU implementation of the quoted pinned lines, on token
//          windows with EOS at each position, missing predecessors and the image token, and over a chained ledger;
//          the gathered rows bitwise fn/quant's IQ4_NL decode of rows read from the file; out-of-range IDs rejected.
//   embed: every row bitwise fn/quant's IQ3_S decode of the card's features; hpost and hx zero; each row at B = 1
//          equal to the same row at B = 3 and 4 in other slots; an ID outside [0, V) rejected on the host and, on the
//          device, flagged in the error word with nothing written; B outside 1..4 rejected.
//   PLE:   a chained run (512 positions; san: 64) with commits of k = B and k < B (draft rows past k), each row against
//          an FP64 implementation of the pinned form on the same e, h, weights and stored history: the score, the gate
//          (at the kernel's own score, and against FP64 by the amendment's bound), the stream update, x and the
//          history versions; both cards' scalars bitwise equal; B-row rounds bitwise equal to B one-row rounds; a k < B
//          commit bitwise equal to k one-row steps; the whole run repeated bitwise.
// Real weights (layer 1's PLE tensors, token_embd rows, per_layer_token_embd rows, the ple.* keys) are read from the
// model's files on the CPU; no model load. dev0 == dev1: loopback on one device and one stream (Compute Sanitizer).
// mode "all" or "san" (64 positions, B = 1 and 3). Prints "ple-test: PASS" and returns 0 when every check passes.
int fnfl_ple_test(const char * mode, int dev0, int dev1, const char * model);
