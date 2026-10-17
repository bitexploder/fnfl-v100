#pragma once

// The experts' device code shared by the gate/up instantiations (experts_gu.cuh, one file a routed format) and the down
// kernel (experts.cu): the shapes, the packed layouts, the launch's distinct experts (meta) and the decode helpers.
// Internal to fn/kernels/experts.

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_meta.cuh"
#include "fn/model/shapes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>

namespace fn::experts {
namespace {

constexpr int32_t n_sm = 72;      // the PG500-216 (GUIDE section 1); both kernels launch one block an SM
constexpr int32_t D = fn::n_embd; // 2560
constexpr int32_t n_used = fn::n_expert_used;
constexpr int32_t n_expert = fn::n_expert;
constexpr int32_t n_group = n_ff_card / 32; // 32-channel groups a card owns per expert
constexpr int32_t n_sub = D / 32;           // 32-weight sub-blocks of a gate/up row

constexpr int32_t x_codes = D;                          // bytes of a row's codes (FNFL's activation format, act.h)
constexpr int32_t x_scales = (D / 32) * 8;              // bytes of a row's {d, s} pairs
constexpr int32_t x_stage = x_codes + x_scales;         // bytes of a staged row: 3,200
constexpr int32_t glu_codes = glu_slots * n_ff_card;    // the GLU buffer: codes [55][320]
constexpr int32_t glu_scales = glu_slots * n_group * 8; // and {d, s} [55][10]
// SP2: the GLU slots of the routing capacity C (cap_of): 11 C, the first slots of the buffer
template <int C> constexpr int32_t glu_slots_c = (fn::n_expert_used + 1) * C;
template <int C> constexpr int32_t glu_codes_c = glu_slots_c<C> * n_ff_card;
template <int C> constexpr int32_t glu_scales_c = glu_slots_c<C> * n_group * 8;

static_assert(n_used * max_rows == max_slots, "slots are the routed pairs at the largest B");
static_assert(n_ff_card == 320 && D == 2560 && n_expert == 512, "the shapes this file's layouts are written for");

// ---- gate/up packed layout -------------------------------------------------------------------------------------
// One tile per (expert, group g): rows r = 0..31 are gate channels 32g + r, rows 32..63 up channels 32g + r - 32, all
// 2560 columns. A record rec = 8 j + rs (j = sub-block 0..79, rs = 0..7) holds sub-block j of rows rs + 8 m, m = 0..7;
// thread rec of the gate/up block owns record rec. A tile is: C main chunks [c][rec] of 16 B, then aux8 [rec] of 8 B,
// then sc8 [j / 2][rs] of 8 B, then the scales d [sb][rs] of 16 B (the 8 rows' FP16 d of superblock sb = j / 8).
// Main chunk contents (rows m of the record; q = grid index bytes, sg = sign bytes or the scale/sign word):
//   IQ2_XXS (C 4): chunk c = [row 2c: qs 8 B][row 2c+1: qs 8 B]                      (qs = 4 index bytes, aux word)
//   IQ2_XS  (C 4): chunk c = [row 2c: qs 8 B][row 2c+1: qs 8 B]; aux8 = scales byte (qs = 4 x (9-bit index, 7 signs))
//   IQ2_S   (C 5): chunk 0 = [qh of rows 0..7][scales of rows 0..7]; chunk 1+c = [row 2c: q 4, sg 4][row 2c+1: q, sg]
//   IQ3_XXS (C 6): rows 4h..4h+3: [q r0][sg r0][sg r1], [q r1][sg r2][sg r3], [q r2][q r3] (q 8 B, sg 4 B), so each
//                  chunk completes a row and at most three sign words wait in registers
//   IQ3_S   (C 6): as IQ3_XXS with sg = the 4 sign bytes; aux8 = qh byte; sc8 = scales byte j / 2 of rows 0..7
//   Q8_0   (C 16, DR1): chunk c = row c / 2's int8 weights 16 (c & 1) .. 16 (c & 1) + 15 of sub-block j; each
//                  32-weight block has its own d, so the d plane is [rec] of 16 B (the 8 rows' FP16 d of sub-block j)
constexpr int32_t gu_threads = 8 * n_sub; // 640: one thread a record
constexpr int32_t gu_warps = gu_threads / 32;
constexpr int32_t red_stride = gu_warps + 1; // floats per row of the reduction buffer (21: conflict-free columns)

struct gu_layout {
    int32_t block_bytes; // GGUF bytes per 256 weights
    int32_t chunks;      // main chunks per record
    bool aux8, sc8;
    int32_t grid_off, grid_len; // in the grid buffer
};

__host__ __device__ constexpr gu_layout gu_layout_of(fmt f) {
    return f == fmt::iq2_xxs   ? gu_layout{66, 4, false, false, 0, 2048}
           : f == fmt::iq2_xs  ? gu_layout{74, 4, true, false, 2048, 4096}
           : f == fmt::iq2_s   ? gu_layout{82, 5, false, false, 6144, 8192}
           : f == fmt::iq3_xxs ? gu_layout{98, 6, false, false, 14336, 1024}
           : f == fmt::iq3_s   ? gu_layout{110, 6, true, true, 15360, 2048}
           : f == fmt::q8_0    ? gu_layout{8 * 34, 16, false, false, 0, 0}
                               : gu_layout{0, 0, false, false, 0, 0};
}

constexpr int32_t kvalues_off = 17408;
static_assert(kvalues_off + 16 == static_cast<int32_t>(grid_bytes), "grid buffer layout");

__host__ __device__ constexpr int64_t tile_bytes(fmt f) {
    return int64_t(gu_threads) * gu_layout_of(f).block_bytes;
}
__host__ __device__ constexpr int64_t aux8_off(fmt f) {
    return int64_t(gu_layout_of(f).chunks) * gu_threads * 16;
}
__host__ __device__ constexpr int64_t sc8_off(fmt f) {
    return aux8_off(f) + (gu_layout_of(f).aux8 ? gu_threads * 8 : 0);
}
__host__ __device__ constexpr int64_t d_off(fmt f) {
    return sc8_off(f) + (gu_layout_of(f).sc8 ? (n_sub / 2) * 8 * 8 : 0);
}

static_assert(d_off(fmt::iq2_xxs) + 10 * 8 * 16 == tile_bytes(fmt::iq2_xxs), "IQ2_XXS tile bytes");
static_assert(d_off(fmt::iq2_xs) + 10 * 8 * 16 == tile_bytes(fmt::iq2_xs), "IQ2_XS tile bytes");
static_assert(d_off(fmt::iq2_s) + 10 * 8 * 16 == tile_bytes(fmt::iq2_s), "IQ2_S tile bytes");
static_assert(d_off(fmt::iq3_xxs) + 10 * 8 * 16 == tile_bytes(fmt::iq3_xxs), "IQ3_XXS tile bytes");
static_assert(d_off(fmt::iq3_s) + 10 * 8 * 16 == tile_bytes(fmt::iq3_s), "IQ3_S tile bytes");
static_assert(d_off(fmt::q8_0) + gu_threads * 16 == tile_bytes(fmt::q8_0), "Q8_0 tile bytes");

// ---- down packed layout ------------------------------------------------------------------------------------------
// Per expert: item j5 (0..4) covers input columns 64 j5 .. 64 j5 + 63 of every output row n.
//   Q2_0:   chunks [j5][n] 16 B (the block's qs), then d [j5][n] FP16
//   IQ4_NL: chunks [j10][n] 16 B (block j10's qs; item j5 = blocks 2 j5, 2 j5 + 1), then d [j5][n][2] FP16
//   Q8_0 (DR1): chunks [j5][c][n] 16 B (c = 0..3: block 2 j5 + c / 2's int8 weights 16 (c & 1) ..), then d as IQ4_NL
constexpr int32_t n_item_k = 5;

__host__ __device__ constexpr int32_t down_chunks(fmt f) {
    return f == fmt::q2_0 ? 1 : f == fmt::q8_0 ? 4 : 2;
}

// ---- the distinct experts of the launch (both kernels): experts_meta.cuh's meta, read at entry (XB2) ----------
template <typename M> __device__ __forceinline__ int32_t slot_of(const M & m, int32_t u, int32_t b) {
    return m.slot_base[u] + __popc(m.mask[u] & ((1u << b) - 1u));
}

// ---- decode helpers ---------------------------------------------------------------------------------------------
// The sign of each byte of x's MSBs replicated over the byte: PRMT's sign-replicate selector (not reachable through
// __byte_perm, which ignores bit 3 of each selector nibble).
__device__ __forceinline__ uint32_t msb_mask(uint32_t x) {
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0xBA98;" : "=r"(r) : "r"(x));
    return r;
}

// 0xff in byte i where bit i (lo) or bit 4 + i (hi) of the sign byte s is set; the multiply moves bit i to bit 8 i + 7
// without carries because the operand is masked to four bits first
__device__ __forceinline__ uint32_t sign_lo(uint32_t s) {
    return msb_mask((s & 0x0fu) * 0x10204080u);
}
__device__ __forceinline__ uint32_t sign_hi(uint32_t s) {
    return msb_mask((s & 0xf0u) * 0x01020408u);
}

// The masks of a 7-bit sign group (IQ2_XXS, IQ2_XS, IQ3_XXS): ksigns_iq2xs without the table, the eighth sign being
// the parity of the seven (C2: POPC issues at 0.5 a clock an SM on its own pipe). s7 < 128, so the shifted copies in
// each product never overlap and nothing carries into a byte's MSB: lo puts bits 0..3 at bits 7, 15, 23, 31; hi puts
// bits 4..6 at 7, 15, 23 and the parity at 31.
__device__ __forceinline__ void sign_masks7(uint32_t s7, uint32_t & lo, uint32_t & hi) {
    lo = msb_mask(s7 * 0x10204080u);
    hi = msb_mask(s7 * 0x00020408u + (static_cast<uint32_t>(__popc(s7)) << 31));
}

// sum over the 8 words of s(g) * a, where the sign masks m select the negative bytes: dp4a(g, a) - 2 dp4a(g & m, a).
// Grid bytes are positive and below 128, so every product is exact.
struct dot2 {
    int32_t p, n;
};
__device__ __forceinline__ void dot_word(dot2 & d, uint32_t g, uint32_t m, uint32_t a) {
    d.p = __dp4a(static_cast<int32_t>(g), static_cast<int32_t>(a), d.p);
    d.n = __dp4a(static_cast<int32_t>(g & m), static_cast<int32_t>(a), d.n);
}

__device__ __forceinline__ float half_lo(uint32_t h2) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xffffu)));
}
__device__ __forceinline__ float half_hi(uint32_t h2) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(h2 >> 16)));
}

__device__ __forceinline__ uint32_t word_of(const uint4 & v, int32_t i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// a zero that depends on v; mz is zero for every valid launch, so a refill address that carries it is issued only
// once v has arrived (HC1c), not hoisted ahead of the stream
__device__ __forceinline__ uint32_t order_after(uint32_t v, uint32_t mz) {
    return v & mz;
}

template <int D_> struct ring {
    uint4 s[D_];
};

} // namespace
} // namespace fn::experts

namespace fn::experts::detail {

// experts.cu: a kernel's opt-in shared memory, once a device
bool smem_attr_once(std::atomic<uint32_t> & done, const void * kernel, int32_t smem);
// experts_gu_<format>.cu: the gate/up launches of one routed format; false for a (fg, fu) not deployed with it
bool gate_up_iq2_xxs(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);
bool gate_up_iq2_xs(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);
bool gate_up_iq2_s(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);
bool gate_up_iq3_xxs(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);
bool gate_up_iq3_s(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);
bool gate_up_q8_0(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);

} // namespace fn::experts::detail
