#pragma once

// The deployed model's expert launches (EX2): one gate/up instantiation per (routed gate/up format, shared gate format,
// shared up format) and one down instantiation per (routed down format, shared down format) that the target layers
// 0-47 use, and no others (EX2 A2). From fnfl/sim/data/deployed-tensor-inventory.json (GGUF headers of the deployed
// IQ3_XXS shards); each entry names its first layer (the bench routes that layer's HX7 traffic) and lists them all.
// DR1: the MTP draft layer 48 adds the Q8_0 entries (all Q8_0, ticket DR1 Stage A); HX7 captured no layer 48, so the
// bench routes layer 47's traffic in its place.
// X(routed, shared gate, shared up, first layer): fn::experts::fmt and fn::dense::fmt member names.
#define FN_EXPERTS_GATE_UP_INSTANCES(X)                                                                                \
    X(iq2_xxs, iq4_xs, iq3_s, 1)   /* layers 1,27,31 */                                                                \
    X(iq2_xxs, q4_k, q4_k, 3)      /* layers 3,11 */                                                                   \
    X(iq2_xxs, iq4_xs, q6_k, 6)    /* layers 6 */                                                                      \
    X(iq2_xxs, iq3_s, iq4_xs, 7)   /* layers 7 */                                                                      \
    X(iq2_xxs, iq3_s, iq3_s, 13)   /* layers 13 */                                                                     \
    X(iq2_xxs, q4_k, iq3_s, 16)    /* layers 16 */                                                                     \
    X(iq2_xs, iq4_xs, iq3_s, 0)    /* layers 0,4,5 */                                                                  \
    X(iq2_xs, q4_k, q5_k, 2)       /* layers 2 */                                                                      \
    X(iq2_xs, q4_k, iq3_s, 9)      /* layers 9 */                                                                      \
    X(iq2_xs, iq3_s, iq4_xs, 10)   /* layers 10 */                                                                     \
    X(iq2_xs, q5_k, q4_k, 14)      /* layers 14 */                                                                     \
    X(iq2_xs, iq4_xs, q4_k, 19)    /* layers 19 */                                                                     \
    X(iq2_xs, iq4_xs, iq4_xs, 25)  /* layers 25,39 */                                                                  \
    X(iq2_s, iq3_s, iq3_s, 12)     /* layers 12,38 */                                                                  \
    X(iq2_s, q4_k, iq3_s, 15)      /* layers 15,26 */                                                                  \
    X(iq2_s, iq4_xs, iq4_xs, 17)   /* layers 17,21 */                                                                  \
    X(iq2_s, iq4_xs, iq3_s, 29)    /* layers 29,40,46 */                                                               \
    X(iq2_s, iq3_s, iq4_xs, 42)    /* layers 42 */                                                                     \
    X(iq3_xxs, iq4_xs, q6_k, 35)   /* layers 35 */                                                                     \
    X(iq3_xxs, iq3_s, iq3_s, 36)   /* layers 36 */                                                                     \
    X(iq3_xxs, q4_k, iq3_s, 41)    /* layers 41 */                                                                     \
    X(iq3_xxs, iq3_s, q4_k, 44)    /* layers 44 */                                                                     \
    X(iq3_xxs, q6_k, iq3_s, 45)    /* layers 45 */                                                                     \
    X(iq3_xxs, iq4_xs, iq4_xs, 47) /* layers 47 */                                                                     \
    X(iq3_s, iq3_s, iq3_s, 8)      /* layers 8,18,24,28,43 */                                                          \
    X(iq3_s, q4_k, iq3_s, 20)      /* layers 20,32,37 */                                                               \
    X(iq3_s, iq4_xs, q6_k, 22)     /* layers 22 */                                                                     \
    X(iq3_s, iq4_xs, iq4_xs, 23)   /* layers 23 */                                                                     \
    X(iq3_s, iq4_xs, q4_k, 30)     /* layers 30,33 */                                                                  \
    X(iq3_s, q4_k, iq4_xs, 34)     /* layers 34 */                                                                     \
    X(q8_0, q8_0, q8_0, 48)        /* layer 48 (the MTP draft, DR1) */

// X(routed down, shared down, first layer): fn::experts::fmt member names
#define FN_EXPERTS_DOWN_INSTANCES(X)                                                                                   \
    X(iq4_nl, iq4_nl, 0) /* layers 0,2,28,29,30,31,32,34,35,37,38,39,42,43,44,46,47 */                                 \
    X(q2_0, iq4_nl, 1)   /* layers 1,5,7,8,9,10,11,15,16,17,18,21,22,23,24,25,26,27,33,36,40,41,45 */                  \
    X(q2_0, q2_0, 3)     /* layers 3,6,12,13,14,19,20 */                                                               \
    X(iq4_nl, q2_0, 4)   /* layers 4 */                                                                                \
    X(q8_0, q8_0, 48)    /* layer 48 (the MTP draft, DR1) */

// ME1: the Unsloth UD-Q4_K_XL candidate's routed pairs (tickets/ME1/unsloth-q4-header-inventory.json, revision
// 766911a6): gate/up Q4_K with down Q5_1 (43 layers), Q4_K with Q8_0 (layers 4, 30, 46, 47), Q5_K with Q8_0 (layer
// 2); its shared expert is Q8_0 (gate, up and down) in every layer. Not the deployed model's: the kernels and the
// tests take them, the bench does not. First layer: the candidate's.
#define FN_EXPERTS_Q4_GATE_UP_INSTANCES(X)                                                                             \
    X(q4_k, q8_0, q8_0, 0) /* layers 0,1,3-47 */                                                                       \
    X(q5_k, q8_0, q8_0, 2) /* layer 2 */

#define FN_EXPERTS_Q4_DOWN_INSTANCES(X)                                                                                \
    X(q5_1, q8_0, 0) /* 43 layers */                                                                                   \
    X(q8_0, q8_0, 2) /* layers 2,4,30,46,47 (the deployed list's draft entry, at B = 5 too) */

// ME7: the Q2 draft's routed pairs (work/ME7's splice of Strata-V100 69713b61's Q2_0 experts into our MTP draft): Q2_0
// gate/up and down with the draft's shared expert in Q8_0, as the Q8 draft's. Opt-in (the draft file's formats choose
// it, fn/loader); the draft runs B <= 4. First layer: the draft's
#define FN_EXPERTS_DRAFT_Q2_GATE_UP_INSTANCES(X) X(q2_0, q8_0, q8_0, 48) /* layer 48 (the Q2 MTP draft) */
#define FN_EXPERTS_DRAFT_Q2_DOWN_INSTANCES(X) X(q2_0, q8_0, 48)          /* layer 48 (the Q2 MTP draft) */
