#pragma once

// TTTS1 split-test: one box against the split, bitwise, on synthetic weights (no model data: every weight from
// fn/quant's seeded valid blocks through the loader's own layouts, fn/loader fill_weights_synth; the PLE table a zero
// mapping). One process on both cards of one box: the reference model (the whole window), then for each split point k
// the front (layers 0 .. k - 1) and back (k .. 47 with the head) models together, their boundary carried between the
// front's send buffers and the back's receive buffers by a host copy (the link's place). The same scripted rounds on
// both; after every round: the tokens, each card's decided tokens and committed words, the logits and the hidden carry,
// and every layer's state (GDN banks and records, histories, K and V, index keys and summaries, the PLE history),
// hashed whole. Cases: lone rounds of 1 .. 5 rows (forced, the bootstrap, generate, verify at d = 1 .. 4 with
// rejected and accepted proposals), commits and cuts within the last step, a reset and a second prompt; joint verify
// rounds of U = 2 .. 4 lanes at every d. Also checks that every stage graph keeps the HC slot rule (consecutive
// sublayers differ, slot 1 last). In graph and eager modes. Prints no timings.

namespace fn::runtime {

int fnfl_split_test(int argc, char ** argv);

}  // namespace fn::runtime
