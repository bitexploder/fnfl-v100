#pragma once

// HC over prompt rows (PB1; FC-HC1 and FC-PF3's HC rows): one HC sublayer's mix for R rows (1 .. 6,144), in two forms.
//   replicated (the target's prompt HC, FC-PF3): each card computes the full row from the full streams
//            [R][4][2560] with the full coefficients: its own resident half and the peer's half (the replica), read
//            through two base pointers in the decode layout (hc.h), BF16.
//   split (the draft's catch-up HC, DR1's Q8_0 modules): each card computes its own half's moments over its own
//            streams [R][4][1280]; the moments cross (the caller's message) and both cards sum them in card order;
//            each card then mixes its own half.
// Three launches a sublayer:
//   hcp_down: mom[r][1300] (hc.h's X-HC packet layout: v[4], u[4][320], q[4][4]) over the features the card sums:
//            v[s] = sum_i x[s][i]^2, u[s][j] = sum_i W[s][j][i] (gamma[s][i] x[s][i]) for j < 324 (rows 320..323 the
//            injection's, written as q).
//   hcp_mid: per row, from mom (and the peer's mom in the split form, summed card 0 first): rinv[s] = 1 / sqrtf(v[s] /
//            2560 + eps), z_j = 0.25 (u[0][j] rinv[0] + ... ) as decode HC's FMA chain, low[j] = z SiLU-form
//            z sigmoid(z) (j < 320), inject[k] = 2 sigmoid(z) (j = 320 + k).
//   hcp_up:  per row and output feature c: g[s] = sigmoid(sum_j Wup[c][s][j] low[j]), mixed[c] = 0.25 sum_s
//            fma((gamma[s][c] x[s][c]) rinv[s], g[s], .), written F32 and FP16 (round to nearest; |mixed| > 65,504
//            sets bit 0 of *err, the FP16 operand's range).
// The streams x are the combined FC-LAY1 streams (x = hres + hpost hx already formed by the caller's combine).
// Numerics: F32; weights BF16 widened exactly, or Q8_0 decoded exactly as d x q (hc.h). Sums over K in 32-term
// steps, each step's sum (in K order) added to the running sum with Kahan compensation, so the long K = 2,560 sums keep
// HC1's 1e-6. expf, sqrtf and the divisions precise (no fast math). Fixed orders, no atomics: reruns are bitwise.
// Reference (hcp-test): FP64 of hc.h's formulas on the same weights and streams; bound max |err| / max |ref| <= 1e-6
// a row for mixed and inject (the split form: the draft's 1e-5, FC-DR1).

//
// PB3, the replicated form on tensor cores (hct_*; PB3 Amendments 1 and 2: one FP16 tensor pass): the same three steps
// and outputs, the two products by fn/kernels/tc's core (HMMA.884, FP16 operands, F32 accumulators).
//   hct_convert: per sublayer and chunk, both halves' BF16 coefficients to FP16 in the products' layouts: down
//            [4][324][2560] and up [10240][320] (row b: feature 2 (b / 8) + ((b / 2) & 1), stream (b & 1) | ((b / 4) &
//            1) << 1, so a lane's four epilogue columns are one feature's four streams). The range check is per row:
//            each row is scaled by the power of two that brings its largest magnitude into [2^14, 2^15), so no value
//            overflows FP16; the row's 2^-e is undone in the F32 epilogue.
//   hct_mix: prescale: y = gamma x per (row, stream), scaled the same way and rounded once to FP16; v[s] = sum x^2 in
//            FP64, rounded once; the four injection rows' u from the same FP16 operands (F32 sums of the exact products
//            in a fixed order). down: u = y' W'^T per stream over the 320 low rows, both scales undone in the
//            epilogue. mid: hcp_mid's F32
//            arithmetic; low scaled and rounded to FP16 the same way. up: g = low' Wup'^T, then hcp_up's F32 epilogue
//            (rinv, gamma x, sigmoid, the four-stream sum, mixed F32 and FP16). Checks (hcp-test): against FP64 at a
//            loose tolerance for layout, indexing and range errors, on synthetic, range-stress and PB1's saved real
//            rows; the FP16 copy and a rerun bitwise; acceptance is NC1 section 6 at model level.

#include <cstddef>
#include <cstdint>

namespace fn::hc {

constexpr int32_t hcp_max_rows = 16384; // 4 x PB3's 4,096-row half
constexpr int32_t hcp_mom      = 1300;  // words a row of mom (hc.h n_mom)

enum class hcp_form : int32_t { replicated, split };

struct hcp_weights {
    hcp_form         form      = hcp_form::replicated;
    // replicated: card 0's and card 1's decode-layout halves (BF16): w_down [4][324][1280], w_up [1280][4][320],
    // gamma [4][1280]
    const uint16_t * down[2]   = { nullptr, nullptr };
    const uint16_t * up[2]     = { nullptr, nullptr };
    const float *    gamma[2]  = { nullptr, nullptr };
    // split: this card's Q8_0 modules in hc.h's planar layout and its gamma [4][1280]
    const uint8_t *  q8_down   = nullptr;
    const uint8_t *  q8_up     = nullptr;
    const float *    gamma_own = nullptr;
};

struct hcp_args {
    hcp_weights   w;
    int32_t       rows       = 0;
    const float * x          = nullptr;  // replicated: [rows][4][2560]; split: [rows][4][1280]
    float *       mom        = nullptr;  // [rows][1300] this card's moments
    const float * mom_peer   = nullptr;  // split: the peer's [rows][1300] (received)
    int32_t       card       = 0;
    float *       low        = nullptr;  // [rows][320]
    float *       rinv       = nullptr;  // [rows][4]
    float *       inject     = nullptr;  // [rows][4]
    // PB3: hct_mix may leave mixed null when mixed16 is set
    float *       mixed      = nullptr;  // F32 [rows][ld_mixed]: replicated 2,560 features; split this card's 1,280
    int64_t       ld_mixed   = 0;
    uint16_t *    mixed16    = nullptr;  // FP16 [rows][ld_mixed16], or null (the split form's caller assembles the row)
    int64_t       ld_mixed16 = 0;
    uint32_t *    err        = nullptr;  // bit 0: an FP16 overflow
};

// the FP16 coefficients of one sublayer (hct_convert's output; 16-byte aligned)
struct hct_weights {
    uint16_t * down   = nullptr;  // [4][324][2560]
    float *    down_s = nullptr;  // [4][324] each row's 2^-e
    uint16_t * up     = nullptr;  // [10240][320]
    float *    up_s   = nullptr;  // [10240]
};

constexpr int64_t hct_down_elems = int64_t(4) * 324 * 2560;
constexpr int64_t hct_up_elems   = int64_t(10240) * 320;

// the bytes of an hct_weights region (carved by hct_carve, 256-byte aligned parts)
int64_t     hct_weight_bytes();
hct_weights hct_carve(uint8_t * region);
// the scratch of hct_mix for rows
int64_t     hct_scratch_bytes(int32_t rows);

// PB3: the previous sublayer's exchange folded into hct_mix's prescale, so the streams are read once: before the
// prescale, S[r][s][c] = fma(inject[r][s], f32(f16 own) + f32(f16 peer) in card order, S[r][s][c]) and written back
// (fn::xchg::pcombine's arithmetic and header check, prompt_xchg.h: on a header that is not gen, err_pmsg is set in
// a.err and the prescale writes nothing). own == nullptr: the streams are already combined.
struct hct_combine {
    int32_t         card    = 0;
    const uint8_t * own     = nullptr;  // this card's send slot (header, FP16 payload [rows][2560])
    const uint8_t * peer    = nullptr;  // the received slot
    uint64_t        gen     = 0;
    const float *   inject  = nullptr;  // [rows][4], the previous sublayer's (not a.inject's new values)
    float *         streams = nullptr;  // a.x, written
};

// a's replicated weights (down, up, gamma unused) into w
bool hct_convert(const hcp_weights & a, const hct_weights & w, void * stream);
// the replicated sublayer: a's x, gamma, outputs as hcp_down + hcp_mid + hcp_up (mom, low, rinv, inject, mixed,
// mixed16, err), from w (converted from a's weights) and scratch; cb the folded exchange (cb.own null: none)
bool hct_mix(const hcp_args & a, const hct_combine & cb, const hct_weights & w, uint8_t * scratch, void * stream);

bool hcp_down(const hcp_args & a, void * stream);
bool hcp_mid(const hcp_args & a, void * stream);
bool hcp_up(const hcp_args & a, void * stream);

struct hcp_resources {
    int32_t regs, local_bytes, shared_bytes;
};

// 0: down (BF16), 1: down (Q8_0), 2: mid, 3: up (BF16), 4: up (Q8_0); PB3: 5 split, 6 tensor down, 7 tensor mid,
// 8 tensor up, 9 convert
bool hcp_kernel_resources(int32_t which, hcp_resources & r);

}  // namespace fn::hc
