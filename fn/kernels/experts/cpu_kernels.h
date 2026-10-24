#pragma once

// ME2: the routed experts' CPU row kernels, copied from Strata-V100 3d4ce4e0 (MIT; the notice is in each source file)
// and reshaped only at their boundary (FNFL names, FNFL's format enum, no environment switches). They read GGUF's
// native blocks, a row's blocks contiguous, so the caller un-permutes FNFL's packed layout into a scratch first
// (experts_cpu.h). The sources and their revision: cpu_iq_avx2.cpp (src/kernels/cpu/iq_avx2.cpp, iq_avx2_rows.inl),
// cpu_kq_avx2.cpp (src/kernels/cpu/kq_avx2.cpp; Q5_K and the Q8_0 / Q8_1 quantizers from ggml-cpu's
// arch/x86/quants.c at 66e11c3), cpu_q2_avx2.cpp (src/kernels/cpu/q2_avx2.cpp; its rows at 69713b61, ME6).
//
// Contract (every kernel): nt rows (1..max_nt = 8; MX7: a prompt job's, decode windows take 1..5) share one decode of
// each weight chunk: the weights are read once for all of them. Accumulators are F32 vectors, one horizontal sum per
// output row. Integer products are exact (maddubs cannot saturate for these formats); scales and sums are F32. Rows
// [r0, r1).
//   gate/up: ff[t][r] = g / (1 + exp(-g)) * u, g = gate row r . act[t], u = up row r . act[t]; gate row r at
//            gate + r gu_row, up row r at up + r gu_row (ME4: two bases, so the gate and up rows can sit in separate
//            tensors, the model's native mappings; was one blob and an up offset); n = 2560 columns; act[t]: n / 256
//            block_q8_k
//   down:    out[t][r] = row r . act[t]; row r at w + r row_bytes; n = 640 columns; act[t]: the down format's
//            activation (Q2_0: act_q, IQ4_NL and Q8_0: n / 32 block_q8_0, Q5_1: n / 32 block_q8_1)
// Reference: experts_cpu_test.cpp (FP64 on the same quantized operands). Compiled with -mavx2 -mfma -mf16c (Strata's
// GCC flags); Carlos's Ryzen 5 5600 has no AVX-512 or AVX-VNNI, so Strata's copies for those are not taken.

#include "fn/kernels/experts/experts.h"

#include <cstddef>
#include <cstdint>

namespace fn::experts::cpu {

// GGUF's activation blocks (ggml-common.h's layouts)
struct block_q8_k {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
struct block_q8_0 {
    uint16_t d; // FP16
    int8_t qs[32];
};
struct block_q8_1 {
    uint16_t d, s; // FP16: d, and d x the sum of qs
    int8_t qs[32];
};
static_assert(sizeof(block_q8_k) == 292 && sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36, "GGUF blocks");

// Strata's planar 32-value activation (src/kernels/cpu/expert.hpp ActQ, the down width): codes, the chunk scale s,
// the exact code sum, and hx = s x sum (Q2_0's -1 offset). ME6 rider: quantize_act_q also writes Strata's bit-plane
// image (69713b61), the order q2_down_rows reads; per pair p of 64-value blocks, qp[128p + 32k + 16h + i] =
// q[128p + 64h + 4i + k], psum[8p + 4h + m] = the sum of q[128p + 64h + 16m .. + 16), pscale[8p + 2c + e] =
// scale[4p + c]; bp_pairs = 5. The 32-byte alignment is the kernel's aligned loads'.
constexpr int32_t act_q_n = 640;
struct act_q {
    alignas(64) int8_t q[act_q_n];
    float scale[act_q_n / 32];
    int32_t sum[act_q_n / 32];
    float hx[act_q_n / 32];
    int32_t nchunks = 0;
    alignas(32) int8_t qp[act_q_n];
    alignas(32) int32_t psum[act_q_n / 16];
    alignas(32) float pscale[act_q_n / 16];
    int32_t bp_pairs = 0;
};

constexpr int32_t max_nt = 8;

// gate/up of IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S (Strata iq256_gu_rows, the plain AVX2 decode)
void iq_gu_rows(fmt f, const uint8_t * gate, const uint8_t * up, size_t gu_row, int n, const void * const * act,
                int nt, float * const * ff, int r0, int r1);
// gate/up of Q4_K, Q5_K (Strata kq256_gu_rows; Q5_K's dot from ggml)
void kq_gu_rows(fmt f, const uint8_t * gate, const uint8_t * up, size_t gu_row, int n, const void * const * act,
                int nt, float * const * ff, int r0, int r1);
// down of IQ4_NL (Strata iq4nl256_down_rows)
void iq4nl_down_rows(const uint8_t * w, size_t row_bytes, int n, const void * const * act, int nt, float * const * out,
                     int r0, int r1);
// down of Q5_1, Q8_0 (Strata kq256_rows)
void kq_down_rows(fmt f, const uint8_t * w, size_t row_bytes, int n, const void * const * act, int nt,
                  float * const * out, int r0, int r1);
// down of Q2_0 (ME6 rider: Strata's bit-plane rows, 69713b61): nblocks 64-value blocks a row, nblocks / 2 ==
// a[t]->bp_pairs; a row's output is the same bits at any nt and any [r0, r1)
void q2_down_rows(const uint8_t * w, size_t row_bytes, int nblocks, const act_q * const * a, int nt,
                  float * const * out, int r0, int r1);

// activations: ggml's quantize_row_q8_K in AVX2 (Strata q8k_quant_avx2), ggml-cpu's x86 quantize_row_q8_0 and
// quantize_row_q8_1, and Strata's act_quant_q8_1_avx2 with its bit-plane image (Q2_0's)
void quantize_q8_k(const float * x, void * y, int64_t n);
void quantize_q8_0(const float * x, void * y, int64_t n);
void quantize_q8_1(const float * x, void * y, int64_t n);
void quantize_act_q(const float * x, int n, act_q & a);

} // namespace fn::experts::cpu
