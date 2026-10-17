#pragma once

// SM1: the CPU reference sampler (FP64), the reference every sampler kernel is tested against (NC1 section 7; never a
// borrowed kernel, R16). It implements the pinned chain's order for the options samp.h lists, read from the pinned
// sources at 66e11c3: common/sampling.cpp common_sampler_init (the chain: logit bias, penalties, top-k, top-p, min-p,
// temperature, dist) and common_sampler_rejection_step; src/llama-sampler.cpp llama_sampler_top_k_impl,
// llama_sampler_top_p_apply (the first index whose running probability reaches p), llama_sampler_min_p_apply (logit >=
// max + logf(p)), llama_sampler_temp_impl (logit / temp), llama_sampler_penalties_apply and _accept.
//
// FNFL's design choices it follows (SM0): ties go to the lower ID; the draw takes SM0's structure with the same
// uniforms as the kernels: on the top-k path one uniform over the kept list in (logit, lower ID) order; on the full
// path the shard by its mass with one uniform, then the token inside the shard by an ascending-ID scan with another.
// Probabilities are FP64 from the F32 transformed logits (exp in double, the kept set exact).
//
// The vocabulary is [0, V) split at `split` (card 0: IDs below it). The model: V = 248,320, split = 124,160.

#include "fn/kernels/samp/samp.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::samp {

// a row's penalty view: the window's counts (whole vocabulary) and the row's deltas (id, +1 or -1)
struct ref_pen {
    const uint32_t * counts = nullptr; // null: no penalties
    std::vector<std::pair<int32_t, int32_t>> deltas;
};

// the transform of a row: bias, then penalties (F32, each operation rounded as the device's)
void ref_transform(const options & o, const float * logits, int32_t V, const ref_pen & pen, std::vector<float> & out);

// a row's distribution after the chain
struct ref_dist {
    path p = path::topk;
    bool greedy = false;
    std::vector<int32_t> id;  // the kept tokens in draw order: top-k path (logit, lower ID); full path ascending ID
    std::vector<double> prob; // normalized
    double mass[2] = {0, 0};  // full path: each card's share
    int32_t split = 0;
    // the top-p cut's margin (FP64): the running mass before the cut token, at it, and top_p * Z, all over Z (for the
    // listing of cut-boundary cases); -1 without top-p
    double cut_before = -1, cut_at = -1, cut_target = -1;
    // precomputed for the draws: the running sums in draw order (top-k path), each card's IDs and running sums
    // (full path), and every ID's probability
    std::vector<double> cum;
    std::vector<int32_t> cid[2];
    std::vector<double> ccum[2];
    std::vector<double> pid;
};

// the distribution of transformed logits l [V]; false (with a message) for a row with no candidate
bool ref_distribution(const options & o, const float * l, int32_t V, int32_t split, ref_dist & d, std::string & err);
double ref_prob(const ref_dist & d, int32_t id);

// a plain draw with the kernels' uniforms (philox.h): u_shard, and on the full path u_local; the CDF position of the
// draw for the boundary listing: the target (in [0, 1) of the stage that chose) and the running sums around it
struct ref_draw_out {
    int32_t token = -1;
    int32_t stage = 0;            // 0: the list (top-k) or the shard choice; 1: the scan inside a shard
    double target = 0;            // the uniform times the stage's total, over that total
    double before = 0, after = 0; // the running sums (over the total) before and after the chosen token
};
ref_draw_out ref_draw(const ref_dist & d, uint64_t u_shard, uint64_t u_local);

// a residual's draw max(0, p - q) (or p when empty), with the kernels' uniforms; q(id) by function
struct ref_q {
    std::vector<int32_t> id; // top-k path: the draft's kept list
    std::vector<float> q;
    const float * shard[2] = {nullptr, nullptr}; // full path: each card's q over its shard (IDs from its split)
    int32_t split = 0;
    double at(int32_t token) const;
};
ref_draw_out ref_residual(const ref_dist & d, const ref_q & q, uint64_t u_shard, uint64_t u_local, bool & empty);

// a residual max(0, p - q) precomputed once for a row and its q (the verify tests' many rounds)
struct ref_resid {
    bool empty = true;
    double R[2] = {0, 0};
    std::vector<int32_t> id; // top-k path: the kept list's order
    std::vector<double> cum;
    std::vector<int32_t> cid[2]; // full path: each card's IDs and running sums
    std::vector<double> ccum[2];
};
ref_resid ref_make_resid(const ref_dist & d, const ref_q & q);
ref_draw_out ref_residual_draw(const ref_dist & d, const ref_resid & r, uint64_t u_shard, uint64_t u_local);

// the verify step of a round with rows 0 .. d: p[t] the rows' distributions, x[t] the drafts (t < d), q[t] the drafts'
// rows; uniforms by position pos0 + t (the kernels' positions). out: a and the a + 1 tokens.
struct ref_verify_out {
    int32_t a = 0;
    std::vector<int32_t> out;
};
ref_verify_out ref_verify(const options & o, const std::vector<ref_dist> & p, const std::vector<int32_t> & x,
                          const std::vector<ref_q> & q, uint32_t pos0, const std::vector<ref_resid> * rc = nullptr);

// the window (penalties) after a ledger: counts and ring (the clean replay)
void ref_window(int32_t cap, const std::vector<int32_t> & ledger, std::vector<uint32_t> & counts,
                std::vector<int32_t> & ring);

// the upper tail of the chi-squared distribution with k degrees of freedom at x (regularized gamma Q(k/2, x/2))
double chi2_q(double x, int32_t k);

// fnfl-drive sampler-ref: the reference's own tests (known small distributions, ties, empty residuals); 0 for PASS
int fnfl_sampler_ref();

} // namespace fn::samp
