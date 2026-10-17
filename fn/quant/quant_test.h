#pragma once

// decode-check: FNFL's decoders (decode.h) against a dump of the pinned engine's own decode of real deployed blocks
// (C1b's one-time reference binary, work/C1b/ref/decode_ref), bitwise. Reads <dir>/manifest.tsv, <dir>/<type>.blocks
// and <dir>/<type>.f32. Prints each format's block and tensor counts and "decode-check: PASS" when every block of all
// 15 formats matches, at least 4,096 blocks a format from at least two tensors (one for F16, which the model has once).
int fnfl_decode_check(const char * dir);

// synth-check: the generators (synth.h): the same seed gives the same blocks, every block decodes to finite values,
// every F16 scale is in its range with its sign rule, every other byte of a block takes all 256 values over 4,096
// blocks, and BF16, F16 and F32 values have mean ~0 and deviation ~1. Prints "synth-check: PASS".
int fnfl_synth_check();
