#pragma once

// VIS1 Stage B, Load 2 (the ticket's image KL and the engineer's a0 ruling, lab 5aaf8fa3): fnfl-drive vis-kl REF MODEL
// DRAFT MMPROJ [--pool-cells P] [--subset FILE] [--a0 PROMPT_IDS REF_IDS]. One load, as the server loads IQ3_XXS on one
// box for fmu_scen.py's a0 (fnfl_load: four slots of 262,144 cells, q8 KV, the MTP draft with its subset, sparse draft
// attention, d 4, the vocabulary's end-of-generation IDs, the launch sampler T 1, top-k 20, top-p 0.95, min-p 0.05),
// with the projector and a KV pool of P cells (default 294,912: the served 409,600 leaves no room for the vision
// classes).
//   a0 (first, as the server ran it): the request IDs (work/VIS1/tools/a0-ids.py) read on slot 0 and 2,048 tokens
//     generated through the speculative path at T = 0 (seed 20261013); compared bitwise with the reference's tokens;
//     the first divergence printed.
//   KL: REF is the file work/VIS1/tools/vis-kl.py ref writes from Load 1 (the reference server): per prompt its text
//     pieces' token IDs, its image files, the reference's 32 greedy tokens and each step's top 20 logprobs. Per prompt:
//     the images prepared (fn/vision prepare) and the prompt built as the server builds it (each piece, then for each
//     image <|vision_start|>, nx ny image tokens and <|vision_end|>); its token count printed against the reference's
//     (N lines); the prompt read with its images (engine prefill, n_reuse 0); step 0 from the retained logits, then the
//     reference's tokens forced in rounds of up to five rows, each row's logits the next step's. Each step prints
//     `R ID STEP KL AGREE FNFL_TOP1 REF_TOP1`: the KL of the reference's top-20 distribution (plus the rest of the
//     vocabulary as one outcome) against FNFL's full softmax (FP64), and whether FNFL's argmax is the reference's
//     top 1. vis-kl.py report summarizes.
// Returns 0 when a0 is bitwise and every prompt ran.

#include <cstdint>
#include <string>

namespace fn::runtime {

struct vis_kl_args {
    std::string ref, model, draft, mmproj;
    std::string subset     = "~/src/flashnext/work/DR1/stageC/subset/draft-subset.ids";
    int32_t     pool_cells = 294912;
    std::string a0_prompt, a0_ref;  // empty: no a0
};

int fnfl_vis_kl(const vis_kl_args & a);

}  // namespace fn::runtime
