#pragma once

// FNFL's decode activation format (fnfl/design/dense-requirements.md, "FNFL's decode activation format"; AUD1 L3; C2)
// and its stand-in producer, quantize: one launch from F32 [B][K] to the format. Each producer writes the format in its
// own epilogue later (DP2, EX2, HC3), with act_device.cuh's quantize_run8 (GD1's gdn_core does); until then this
// kernel writes it.
//
// Format (B rows of K values, K a multiple of 32):
//   codes:  int8 [B][K], row b at codes + b K; with a 16-byte aligned base every row starts on a 16-byte boundary, so a
//           consumer reads 16 codes a 16-byte load.
//   scales: float2 {d, s} [B][K / 32] (8 bytes a run of 32 values), row b at scales + b K / 32.
//   rule, per run of 32 values x (the CPU reference is fn/quant/act_ref.h, bitwise):
//           amax = max |x|; d = amax / 127.0f; id = amax > 0 ? 127.0f / amax : 0.0f;
//           q = __float2int_rn(x id) (round to nearest, ties to even), clamped to [-127, 127]; s = d (float) sum q, with
//           sum q exact in int32. The divisions are IEEE (round to nearest), not the fast reciprocal.
//
// Contract (quantize):
//   shapes:   rows in [1, max_rows], K in [32, max_k] a multiple of 32; anything else, a null or misaligned pointer
//             (x and codes 16-byte, scales 8-byte) or a null stream is rejected (false, nothing enqueued).
//   numerics: inputs finite F32. Built without fast math (--ftz=false --prec-div=true --prec-sqrt=true --fmad=true);
//             the divisions are __fdiv_rn, the conversion __float2int_rn; amax and sum q are exact, so the output is a
//             function of the input alone and bitwise equal to the CPU reference (act-test).
//   floor:    bytes B K (4 + 1 + 8 / 32): 40,320 B at B = 3, K = 2,560 (39 ns at 1,021.6 GB/s): the launch dominates.
//   budget:   8 lanes a run (a float4 each), 64-thread blocks, one block per 8 runs; amax and the sum over shuffles.
//   measured: act-bench (C2 Result).
//   evidence: C2.

#include <cstddef>
#include <cstdint>

namespace fn::act {

constexpr int32_t run = 32;           // values a scale pair covers
constexpr int32_t max_rows = 1 << 16; // the quantize kernel's row limit
constexpr int32_t max_k = 1 << 16;    // and its K limit

constexpr bool valid(int32_t rows, int32_t k) {
    return rows >= 1 && rows <= max_rows && k >= run && k <= max_k && k % run == 0;
}
// bytes of the codes and of the scales of rows x k values
constexpr size_t codes_bytes(int32_t rows, int32_t k) {
    return size_t(rows) * size_t(k);
}
constexpr size_t scales_bytes(int32_t rows, int32_t k) {
    return size_t(rows) * size_t(k / run) * 8;
}

// what a consumer takes: codes int8 [rows][k], scales F32 [rows][k / 32][2] ({d, s} pairs)
struct view {
    const int8_t * codes;
    const float * scales;
    int32_t rows;
    int32_t k;
};

// what a producer writes
struct out_view {
    int8_t * codes;
    float * scales;
    int32_t rows;
    int32_t k;
};

// Enqueue on stream (a cudaStream_t): x F32 [o.rows][o.k] to o. False, with nothing enqueued, for an argument outside
// the contract; false after a failed launch submission (the CUDA error is left for the caller).
bool quantize(const float * x, const out_view & o, void * stream);

} // namespace fn::act
