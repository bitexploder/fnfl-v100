#pragma once

// V1 step 5: the output end's launches alone on card 0 (warm, and cold per R17: a 12 MB L2 flush and the instruction
// cache eviction before each launch) at B = 1 and 3, the vocabulary product also on card 1, then the whole head (down2,
// up2, ready, logits, greedy) as one graph a card on both cards at B = 1..4. Synthetic weights of the real shapes
// (the vocabulary: fn/quant's Q5_K blocks, 218.5 MB a card); no model load. One JSON line a point. mode: "all".
int fnfl_v_bench(int dev0, int dev1, const char * mode);
