#pragma once

// APF2: fnfl-drive prompt-tune MODEL DRAFT OUT [DEVICE] — the kernel-size ladder's calibration (the ticket's Load 1).
// One model load: the UD-Q4_K_XL target's layers 0 .. 3 and the head with the MTP draft (every distinct projection
// shape of the production split: GDN, QSA, each routed gate/up and down format, the draft's), both TP2 cards' halves
// on one device. For each ladder key (fn/model/prompt.h prompt_ladder_keys, both cards) at each rung its products
// reach (dense and HC 8 .. 4,096 rows: a piece's; grouped experts 8 .. 64 routed rows an expert), the cost model's
// first 8 configurations (fn/model/ladder.h lad_rank) and the retained route, each 5 warm-up launches then 20 timed
// launches (CUDA events around each), the median counted:
//   dense:   pl_tc on the tile and split count, against pd_convert + pd_tc (the retained route converts at its piece)
//   HC:      hct_mix_ladder with the product on the candidate and the other product on its retained tile, against
//            hct_convert + hct_mix; then each rung's two picks together against the retained route (the prompt path
//            takes the ladder only when both products do), the retained route kept when the pair is not faster
//   experts: ep_gate_up_ladder / ep_down_ladder on the tile against ep_gate_up / ep_down, over the layer's real expert
//            weights with SYNTHETIC routing in two layouts (uniform: every expert the rung's rows, within one; skewed:
//            half the pairs on 32 experts), the pick by the two medians' mean
// The winner of each (key, rung) is the fastest median, the retained route on a tie. Writes OUT/ladder_table.inc (the
// table: every measured candidate's median, the winners marked; fn/model/ladder_table.inc's form) and
// OUT/prompt-tune.tsv (every timing: each launch's microseconds). Performance evidence for the table only.

namespace fn::tools {

int fnfl_prompt_tune(int argc, char ** argv);

}  // namespace fn::tools
