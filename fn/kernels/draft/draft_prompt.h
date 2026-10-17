#pragma once

// The draft's catch-up over prompt positions (PB1; FC-DR2's pruned path over a chunk, FC-PF3's draft row), the pieces
// beside the products (fn::dense pd_product: eh_proj, K, V) and the split HC (fn::hc hcp_* in the split form):
//   dcp_norm:    for positions t of a slice, e = E[t] (token t's embedding row) and h = the target's final streams of
//                row t - 1 (the slice's first position: pred, the predecessor row), both full width on each card:
//                e' = (e enorm) rs_e, h'_s = (h_s hnorm_s) rs_h_s with rs = 1 / sqrtf(sum x^2 / 2560 + 1e-6) (draft.h's
//                X-DRAFT-IN arithmetic per value), written FP16 (eh_proj's operand: e' [P][2560], h' [P][4][2560]).
//   dcp_mixed:   the draft HC's mixed row assembled from both cards' halves (card 0's features first), FP16.
//   dcp_prepare: per position, k = imrope(rms_norm(kv_k) k_norm, p), v = kv_v, rounded to F16 (nearest even) into the
//                draft caches at cell p (DA-TC prepare's arithmetic, the rope planes all p).
// Numerics: F32, precise forms (no fast math). Sums in fixed order. Reference: draft-prompt cases of dcp-test, FP64;
// bounds DR1 Stage C's (the normalized rows 1e-6 a row; K within one F16 ulp of FP64; V bitwise f16(v)).

#include <cstddef>
#include <cstdint>

namespace fn::draft {

constexpr int32_t prompt_max_rows = 4096;

struct norm_args {
    int32_t       rows   = 0;
    const float * emb    = nullptr;  // [rows][2560] the slice's embeddings
    const float * hidden = nullptr;  // [rows][4][2560]: row t's input is hidden row t - 1 (t >= 1)
    const float * pred   = nullptr;  // [4][2560] the slice's first position's predecessor
    const float * enorm  = nullptr;  // [2560] full width
    const float * hnorm  = nullptr;  // [4][2560] full width
    uint16_t *    e16    = nullptr;  // [rows][2560]
    uint16_t *    h16    = nullptr;  // [rows][4][2560]
    uint32_t *    err    = nullptr;  // bit 0: an FP16 overflow
};

bool dcp_norm(const norm_args & a, void * stream);

// mixed16 [rows][2560] from card 0's and card 1's halves [rows][1280] F32 (own and peer by card)
bool dcp_mixed(int32_t       rows,
               int32_t       card,
               const float * own,
               const float * peer,
               uint16_t *    mixed16,
               uint32_t *    err,
               void *        stream);

struct kvp_args {
    int32_t       rows = 0, pos0 = 0, max_cells = 0;
    const float * kv_k        = nullptr;  // [rows][256]
    const float * kv_v        = nullptr;
    const float * k_norm      = nullptr;  // [256]
    uint16_t *    k_cache     = nullptr;  // F16 [max_cells][256]
    uint16_t *    v_cache     = nullptr;
    float         theta_scale = 0.0f;
};

bool dcp_prepare(const kvp_args & a, void * stream);

}  // namespace fn::draft
