#pragma once

// The PLE layer over prompt rows (PB1; FC-PF3's FC-PLE2 row), replicated: each card computes every channel of the
// half's rows from the full coefficients (its own resident half and the peer's replica), so no statistic crosses.
// Full channel ch = 2560 s + f (stream s, feature f); feature f's coefficients are card f / 1280's, at its channel
// 1280 s + f % 1280 (ple.h's card layout). The products (key [rows][10240], value [rows][2560]) are fn::dense::pd_bf16
// launches by the caller. Then:
//   ple_prompt_stats:  per row, the 13 statistics over the full width (h = the streams S, FC-LAY1 combined), and the
//                      row and stream scalars gate, xinv (E1's formulas, ple.h), scal [rows][8] (gate[4], xinv[4]);
//   ple_prompt_x:      x[r][ch] = (value[r][f] gate[s]) xinv[s] norm_conv[ch], the convolution's input;
//   ple_prompt_update: per row and channel, conv = w0 x[t - 9] + w1 x[t - 6] + w2 x[t - 3] + w3 x[t] (F16 taps
//                      widened exactly; x before the half from the history), S += gated + SiLU(conv), in place;
//   ple_prompt_hist:   the half's new history (the last 9 x of each channel, oldest first) into hist_out [10240][9]
//                      (the prompt's full copy) and the card's own channels into its state version (ple.h's
//                      [5][5120][9] layout at slot version_slot(*cur, v_out));
//   ple_prompt_hist_in: the full copy from the card's own state version 0 and the peer's (copied), at a run's start.
// Numerics: F32, E1's operation order per value; sums in fixed order (a thread's strided features in order, then a
// shuffle tree, then the warps in order). Not bitwise to decode (NC1 section 6). Reference (ple-prompt cases of
// pe-test): FP64 of the pinned form; bounds E1's (stream update <= 1e-5 of max |ref| a row and stream).

#include <cstddef>
#include <cstdint>

namespace fn::ple {

constexpr int32_t prompt_channels = 4 * 2560;  // 10,240
constexpr int32_t prompt_max_rows = 4096;

struct prompt_weights {
    const float *    norm_key[2]   = { nullptr, nullptr };  // card 0's and card 1's [4][1280]
    const float *    norm_query[2] = { nullptr, nullptr };
    const float *    norm_conv[2]  = { nullptr, nullptr };
    const uint16_t * conv[2]       = { nullptr, nullptr };  // [5120][4] F16
};

struct prompt_args {
    prompt_weights   w;
    int32_t          rows     = 0;
    const float *    key      = nullptr;  // [rows][10240]
    const float *    value    = nullptr;  // [rows][2560]
    float *          streams  = nullptr;  // [rows][4][2560] (the half's rows of S)
    float *          scal     = nullptr;  // [rows][8]
    float *          x        = nullptr;  // [rows][10240]
    const float *    hist_in  = nullptr;  // [10240][9] the history before the half's first row
    float *          hist_out = nullptr;  // [10240][9]
    float *          state    = nullptr;  // the card's history [5][5120][9]
    const uint32_t * cur      = nullptr;  // device: the committed slot
    int32_t          v_out    = 1;        // the logical version the half's history goes to
    int32_t          card     = 0;
};

bool ple_prompt_stats(const prompt_args & a, void * stream);
bool ple_prompt_x(const prompt_args & a, void * stream);
bool ple_prompt_update(const prompt_args & a, void * stream);
bool ple_prompt_hist(const prompt_args & a, void * stream);
// hist_out <- own (state version 0) and peer ([5120][9], the peer's version 0, copied)
bool ple_prompt_hist_in(const prompt_args & a, const float * peer, void * stream);

}  // namespace fn::ple
