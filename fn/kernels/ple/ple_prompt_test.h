#pragma once

// pe-test (PB1): the bulk embedding (embed.h embed_rows, embed_join) bitwise against fn/quant's IQ3_S decode, the full
// row on both cards and in all four streams; the full-width PLE (ple_prompt.h, with pd_bf16's two-base products)
// against FP64 of E1's form (x and the streams within 1e-5 a row and stream), its history bitwise the kernel's own x
// and the card's own channels in its state version. mode "san": small cases. Prints "pe-test: PASS".
int fnfl_pe_test(int device, const char * mode);
