#pragma once

// The PLE layer on one card of the S2 split (E1; FC-PLE2, simulator node PLE): at target layer 1, before its first HC
// mix, each stream gets a gated per-layer embedding of the row's hashed n-grams (e, FC-PLE1) plus a dilated causal
// convolution of it. Pinned source: qwen4exp.cpp build_ple (lines 2043-2131 at 66e11c3), called at line 471.
//
// Card r owns features r*1280 .. of every stream (fn/model/placement): ple_key rows s*2560 + r*1280 .. (5,120 card
// rows, stream-major), ple_value rows r*1280 .. (1,280), the norms' and ple_conv1d's channels s*2560 + r*1280 .. as
// card channel c = 1280 s + i. Three launches a step, one stream a card, then the frame's commit:
//   products: key [B][5120] = ple_key . e and value [B][1280] = ple_value . e, DP2's BF16 product (fn::dense), one
//             launch with both matrices.
//   stats:    per row, the 13 partial statistics over the card's features (WP2's factorization), with
//             h = hres + hpost * hx (FC-LAY1, one fmaf): sum key_s^2, sum h_s^2, sum key_s nk_s h_s nq_s (s = 0..3), sum
//             value^2. Sends them to the peer as X-PLE-STATS (fn/xchg/xchg.h has the protocol and its reuse chain) and
//             keeps the same canonical words; resets this card's X-PLE-STATS slot of the previous round.
//   update:   polls the peer's statistics, sums each with card 0's partial first (so both cards hold identical
//             scalars), then per row and stream: rk = 1 / sqrt(K / 2560 + eps), rq = 1 / sqrt(H / 2560 + eps),
//             score = KQ rk rq / sqrt(2560), gate = sigmoid(sign(score) sqrt(max(|score|, 1e-6))),
//             xinv = 1 / sqrt(gate^2 V / 2560 + eps); per channel and row in order: gated = value gate,
//             x = gated xinv norm_conv, conv = w0 x[t-9] + w1 x[t-6] + w2 x[t-3] + w3 x (F16 taps widened exactly),
//             h_out = h + (gated + SiLU(conv)); writes history versions 1..B.
//   commit:   the frame's (fn/kernels/round, SR1): logical version k (= N_after - N_before, 0..5; FC-ST2) becomes
//             current by moving the committed slot's word cur; nothing is copied.
//
// Contract:
//   shapes:   rows B in {1, .., 5} (FC-G1; 5: SP2); D = 2560, F = 1280, 4 streams, 5,120 channels, 4 taps dilated
//             by 3 (9 history values). Anything else, a null or misaligned pointer, is rejected (false, nothing
//             enqueued).
//   state:    history F32 [6][5120][9] (R1's "L1 PLE" part): logical version 0 committed; version v = the channel's
//             last 9 inputs after the first v rows, oldest first. A step reads version 0 and writes versions 1..B.
//             Logical version v is at slot fn::round::version_slot(*cur, v) (fn/kernels/round/version.h, SR1).
//   numerics: F32 storage, accumulation and outputs; weights BF16 (key, value), F32 norms, F16 taps, widened exactly.
//             Built without fast math (--ftz=false --prec-div=true --prec-sqrt=true --fmad=true); sqrtf, expf and the
//             divisions are the precise forms. Every reduction has a fixed order and no atomics (stats: a thread's 5
//             features in order, a shuffle-xor tree, the 8 warps in order), one block a row, so a row is bitwise
//             independent of B and its slot; the convolution runs the rows in order (NC1 section 2.2's B-steps form).
//             Reference: FP64 of the pinned form (full-width norms, no factorization) on the same e, h and weights
//             (ple_test.cpp): score |err| <= 1e-5 max(1, |score|); gate within 1e-6 of FP64 at the kernel's own score
//             and within 1e-6 + 2 slope(s) own_err of FP64 (E1 amendment); stream update <= 1e-5 of max |reference|
//             a row and stream; x within 1e-5; history versions bitwise copies of the kernel's own x.
//   out of contract: non-finite inputs (a NaN statistic is canonicalized by the exchange on both cards alike).
//   budget:   products: DP2's (72 x 512, no spills); stats: B blocks x 256 threads; update: 80 blocks x 64 threads,
//             template on B. Registers and spills in work/E1/sass.
//   floor:    products 32,768,000 weight bytes: 1.949 us + bytes / 1,021.6 GB/s = 34.0 us [A, HX12]; stats and update
//             move under 1 MB at B = 3, ramp-bound (about 1.95 us each [M, HX12]) plus the exposed exchange.
//   evidence: E1 (ple-test, ple-bench).
//
// Layouts (card r): e [B][2560] F32 (the round's upload slot on this card); hres, h_out [B][4][1280], hpost [B][4],
// hx [B][1280] F32 (FC-LAY1: layer 0's output boundary in, layer 1's hres out, which layer 1 reads with hpost = 0 and
// hx = 0); h_out may equal hres (in place: each value is read and written by the same thread). Every pointer 16-byte
// aligned; the X-PLE-STATS slots from fn::xchg::rings (msg::ple).

#include <cstddef>
#include <cstdint>

namespace fn::ple {

constexpr int32_t n_in = 2560;  // e, the 16 gathered rows of 160
constexpr int32_t n_own = 1280; // features a card owns per stream
constexpr int32_t n_stream = 4;
constexpr int32_t n_channel = n_stream * n_own; // 5,120 card channels
constexpr int32_t n_tap = 4;                    // ple.conv_kernel
constexpr int32_t dilation = 3;                 // ple.ngram_size
constexpr int32_t n_hist = (n_tap - 1) * dilation;
constexpr int32_t n_version = 6; // SP2: fn::round::n_version (1 + max_rows)
constexpr int32_t max_rows = 5; // SP2
constexpr int32_t n_stats = 13;
constexpr int64_t hist_version_floats = int64_t(n_channel) * n_hist;
constexpr int64_t hist_bytes = n_version * hist_version_floats * 4;

struct weights {
    const uint8_t * key = nullptr;      // [5120][2560] BF16, GGUF rows (fn::dense's BF16 pack is a copy)
    const uint8_t * value = nullptr;    // [1280][2560] BF16
    const float * norm_key = nullptr;   // [4][1280]
    const float * norm_query = nullptr; // [4][1280]
    const float * norm_conv = nullptr;  // [4][1280]
    const uint16_t * conv = nullptr;    // [5120][4] F16 bits
};

struct step_args {
    weights w;
    const float * e = nullptr;     // [B][2560]
    const float * hres = nullptr;  // [B][4][1280]
    const float * hpost = nullptr; // [B][4]
    const float * hx = nullptr;    // [B][1280]
    float * h_out = nullptr;       // [B][4][1280], may equal hres
    float * hist = nullptr;        // [5][5120][9]
    const uint32_t * cur = nullptr; // device: the committed slot, 0..4 (SR1)
    uint8_t * ws = nullptr;        // workspace_bytes(), prepared once
    float * stats_peer = nullptr;  // the peer's X-PLE-STATS slot of this round (peer memory)
    float * stats_slot = nullptr;  // this card's X-PLE-STATS slot of this round (the peer writes it)
    float * stats_reset = nullptr; // this card's slot of the previous round, reset by stats; null: none
    // TG1: the use counter u on the device (the round descriptor's PLE use, FC-RD1), so one captured launch serves every
    // round: stats_peer and stats_slot are then slot 0 of the peer's and this card's X-PLE-STATS rings, the launches
    // use slot u % 3 and stats resets this card's slot (u + 2) % 3 (the previous use's; stats_reset is not read). The
    // slots, readers, resetter and reuse chain are xchg.h's, unchanged. Null: the three pointers above are this use's.
    const uint64_t * use = nullptr;
    uint32_t * err = nullptr;      // this card's error word (fn::xchg::err_poll_ple)
    int32_t card = 0;
};

// The card's private workspace (FC-D2): key [4][5120], value [4][1280], own statistics [4][16], the per-row and
// stream scalars [4][4][4] (score, gate, xinv, 0), DP2's product work, each 256-byte aligned.
size_t workspace_bytes();
// zero the workspace once before the first step (DP2's work counters); synchronous on the current device
bool prepare(uint8_t * ws);

struct ws_view {
    float * key;
    float * value;
    float * stats; // [B][16] the canonical words this card sent
    float * scal;  // [B][4][4]: score, gate, xinv, 0
    uint8_t * work;
};
ws_view carve(uint8_t * ws);

// Enqueue on stream (a cudaStream_t). False, with nothing enqueued, for a rejected input; false after a failed launch
// submission (the CUDA error is left for the caller). step enqueues products, stats and update.
bool products(int32_t rows, const step_args & a, void * stream);
bool stats(int32_t rows, const step_args & a, void * stream);
bool update(int32_t rows, const step_args & a, void * stream);
bool step(int32_t rows, const step_args & a, void * stream);

// FMU2 (build item 4): every lane's products in one fn::dense::product_joint launch, each weight loaded once: lanes 2..4,
// lane u's a[u] at fn::dense::lane_rows (5) rows, its own e and workspace, the same weights. Lane u's key and value are
// bitwise its products(5, a[u]) (the joint product's contract, dense.h). work: joint_products_work_bytes(), zeroed
// once, private to these launches. Enqueued on stream; false as products.
size_t joint_products_work_bytes();
bool products_joint(int32_t lanes, const step_args * const a[], uint8_t * work, void * stream);

} // namespace fn::ple
