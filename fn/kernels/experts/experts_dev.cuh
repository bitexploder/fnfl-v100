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
//   Q4_K   (C 8, ME1): chunk m = row m's 32 nibbles of sub-block j, byte t = weight t (low) and t + 16 (high); after
//                  the main chunks the scale plane sm [rec] of 16 B (byte 2m: row m's 6-bit sub-block scale, 2m + 1:
//                  its 6-bit min), then d and dmin, each [sb][rs] of 16 B (the 8 rows' FP16 d, dmin of superblock sb)
//   Q2_0   (C 4, ME7, the Q2 draft's): chunk c = [row 2c: its 8 code bytes of sub-block j][row 2c + 1's]; a GGUF Q2_0
//                  block (FP16 d, 16 code bytes, 64 weights) is sub-blocks 2i, 2i + 1, so the d plane is [j / 2][rs] of
//                  16 B (the 8 rows' FP16 d of the pair): 72 bytes a 256 weights, GGUF's (Strata-V100 69713b61
//                  mtp_pack.py q2_0 / ggml-quants.c: weight t's code at byte 2 + t / 4, bits 2 (t % 4); (code - 1) d)
//   Q5_K   (C 10, ME1): Q4_K's with each row's high bits, 4 B a row (bit t: bit 4 of weight t): [qh of rows 0..3]
//                  [r0][r1][r2][r3] [qh of rows 4..7][r4]..[r7]; the tile is 4 bytes a 256 weights over GGUF's
constexpr int32_t gu_threads = 8 * n_sub; // 640: one thread a record
constexpr int32_t gu_warps = gu_threads / 32;
constexpr int32_t red_stride = gu_warps + 1; // floats per row of the reduction buffer (21: conflict-free columns)

struct gu_layout {
    int32_t block_bytes; // packed bytes per 256 weights (GGUF's but for the K-quants, 4 more: the scales unpacked)
    int32_t chunks;      // main chunks per record
    bool aux8, sc8;
    int32_t grid_off, grid_len; // in the grid buffer
};

__host__ __device__ constexpr gu_layout gu_layout_of(fmt f) {
    return f == fmt::iq2_xxs   ? gu_layout{66, 4, false, false, 17424, 1024} // FMS4 item 2: the selector tables
           : f == fmt::iq2_xs  ? gu_layout{74, 4, true, false, 18448, 2048}
           : f == fmt::iq2_s   ? gu_layout{82, 5, false, false, 20496, 4096}
           : f == fmt::iq3_xxs ? gu_layout{98, 6, false, false, 14336, 1024}
           : f == fmt::iq3_s   ? gu_layout{110, 6, true, true, 15360, 2048}
           : f == fmt::q8_0    ? gu_layout{8 * 34, 16, false, false, 0, 0}
                               : gu_layout{0, 0, false, false, 0, 0};
}
// ME1: the K-quants (the Q4 candidate's routed gate/up) take their layout from gu_layout_k, every format's at compile
// time; gu_layout_of keeps the deployed formats' chain, whose run-time form the IQ bodies take (two more arms in it
// changed their SASS)
__host__ __device__ constexpr bool is_kq(fmt f) {
    return f == fmt::q4_k || f == fmt::q5_k;
}
// ME7: Q2_0 (the Q2 draft's routed gate/up) is here too, out of gu_layout_of's run-time chain for the same reason
__host__ __device__ constexpr gu_layout gu_layout_k(fmt f) {
    return f == fmt::q4_k   ? gu_layout{148, 8, false, false, 0, 0}
           : f == fmt::q5_k ? gu_layout{180, 10, false, false, 0, 0}
           : f == fmt::q2_0 ? gu_layout{72, 4, false, false, 0, 0}
                            : gu_layout_of(f);
}

constexpr int32_t kvalues_off = 17408;
static_assert(kvalues_off + 16 == gu_layout_of(fmt::iq2_xxs).grid_off &&
                  gu_layout_of(fmt::iq2_xxs).grid_off + gu_layout_of(fmt::iq2_xxs).grid_len ==
                      gu_layout_of(fmt::iq2_xs).grid_off &&
                  gu_layout_of(fmt::iq2_xs).grid_off + gu_layout_of(fmt::iq2_xs).grid_len ==
                      gu_layout_of(fmt::iq2_s).grid_off &&
                  gu_layout_of(fmt::iq2_s).grid_off + gu_layout_of(fmt::iq2_s).grid_len ==
                      static_cast<int32_t>(grid_bytes),
              "grid buffer layout: the IQ2 selector tables after kvalues_iq4nl");

__host__ __device__ constexpr int64_t tile_bytes(fmt f) {
    return int64_t(gu_threads) * gu_layout_k(f).block_bytes;
}
__host__ __device__ constexpr int64_t aux8_off(fmt f) {
    return int64_t(gu_layout_k(f).chunks) * gu_threads * 16;
}
__host__ __device__ constexpr int64_t sc8_off(fmt f) {
    return aux8_off(f) + (gu_layout_of(f).aux8 ? gu_threads * 8 : 0);
}
// ME1: a K-quant's scale plane is where the aux8 plane would be
__host__ __device__ constexpr int64_t sm_off(fmt f) {
    return aux8_off(f);
}
__host__ __device__ constexpr int64_t d_off(fmt f) {
    return sc8_off(f) + (gu_layout_of(f).sc8 ? (n_sub / 2) * 8 * 8 : 0) + (is_kq(f) ? gu_threads * 16 : 0);
}
__host__ __device__ constexpr int64_t dmin_off(fmt f) {
    return d_off(f) + 10 * 8 * 16;
}

static_assert(d_off(fmt::iq2_xxs) + 10 * 8 * 16 == tile_bytes(fmt::iq2_xxs), "IQ2_XXS tile bytes");
static_assert(d_off(fmt::iq2_xs) + 10 * 8 * 16 == tile_bytes(fmt::iq2_xs), "IQ2_XS tile bytes");
static_assert(d_off(fmt::iq2_s) + 10 * 8 * 16 == tile_bytes(fmt::iq2_s), "IQ2_S tile bytes");
static_assert(d_off(fmt::iq3_xxs) + 10 * 8 * 16 == tile_bytes(fmt::iq3_xxs), "IQ3_XXS tile bytes");
static_assert(d_off(fmt::iq3_s) + 10 * 8 * 16 == tile_bytes(fmt::iq3_s), "IQ3_S tile bytes");
static_assert(d_off(fmt::q8_0) + gu_threads * 16 == tile_bytes(fmt::q8_0), "Q8_0 tile bytes");
static_assert(dmin_off(fmt::q4_k) + 10 * 8 * 16 == tile_bytes(fmt::q4_k), "Q4_K tile bytes");
static_assert(d_off(fmt::q2_0) + (n_sub / 2) * 8 * 16 == tile_bytes(fmt::q2_0) && tile_bytes(fmt::q2_0) == 46080,
              "Q2_0 tile bytes: 64 rows of 2,560 weights at GGUF's 18 bytes a 64");
static_assert(dmin_off(fmt::q5_k) + 10 * 8 * 16 == tile_bytes(fmt::q5_k), "Q5_K tile bytes");

// ---- down packed layout ------------------------------------------------------------------------------------------
// Per expert: item j5 (0..4) covers input columns 64 j5 .. 64 j5 + 63 of every output row n.
//   Q2_0:   chunks [j5][n] 16 B (the block's qs), then d [j5][n] FP16
//   IQ4_NL: chunks [j10][n] 16 B (block j10's qs; item j5 = blocks 2 j5, 2 j5 + 1), then d [j5][n][2] FP16
//   Q8_0 (DR1): chunks [j5][c][n] 16 B (c = 0..3: block 2 j5 + c / 2's int8 weights 16 (c & 1) ..), then d as IQ4_NL
//   Q5_1 (ME1): chunks as IQ4_NL (block j10's qs: byte t = weight t (low) and t + 16 (high)), then [j5][n] of 16 B:
//               the two blocks' qh words, then FP16 d0, m0, d1, m1
constexpr int32_t n_item_k = 5;

__host__ __device__ constexpr int32_t down_chunks(fmt f) {
    return f == fmt::q2_0 ? 1 : f == fmt::q8_0 ? 4 : 2;
}
// bytes of item j5's d entry of one output row
__host__ __device__ constexpr int32_t down_d_bytes(fmt f) {
    return f == fmt::q2_0 ? 2 : f == fmt::q5_1 ? 16 : 4;
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

// FMS4 item 2: PRMT with a run-time selector in the default mode. __byte_perm clears bit 3 of each selector nibble
// first (one more LOP3 ... 0x7777 a call); the IQ2 selectors never set it, so the raw form is exact. prmt.b32 reads
// only the selector's low 16 bits.
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
// FMS4 item 2: the eight sign bits of a 7-bit group (IQ2_XXS, IQ2_XS): the parity of the seven at bit 7 (ggml's
// ksigns_iq2xs). s7 < 128, so nothing carries into bits 0..6; the bits above 7 are junk that iq2_signed drops. One
// multiply-add: written in C++, the compiler knows the bits are disjoint and emits a shift and a LOP3 merge instead.
__device__ __forceinline__ uint32_t sign8_of7(uint32_t s7) {
    uint32_t r;
    asm("mad.lo.u32 %0, %1, 128, %2;" : "=r"(r) : "r"(static_cast<uint32_t>(__popc(s7))), "r"(s7));
    return r;
}
// FMS4 item 2: a | (b & M) as one LOP3. Written in C++, the selector's two merges are reassociated into three LOP3s
// ((o & M2) | (e & M1), then | g32).
template <uint32_t M> __device__ __forceinline__ uint32_t or_and(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xf8;" : "=r"(r) : "r"(a), "r"(b), "n"(M));
    return r;
}
// FMS4 item 2: an IQ2 grid entry's 8 signed bytes, weights 0..3 in lo and 4..7 in hi, from its selector entry g32
// (nibble k: the level code of byte k) and its sign bits s8 (bit k: weight k negative; bits above 7 ignored). Sign k
// goes to bit 2 of nibble k (even k by the first product, odd k by the second, each masked on its own: one mask over
// their sum is wrong, spread2.py), so nibble k indexes the palette pos (8, 25, 43) or neg (-8, -25, -43). 0 of 360,448
// (entry, sign) cases wrong over the three grids (fnfl/evidence/fms4-decode-recipes/final.py; grid-test's device form).
// The merge order is the ticket's first, (g32 | e) | o.
__device__ __forceinline__ void iq2_signed(uint32_t g32, uint32_t s8, uint32_t & lo, uint32_t & hi) {
    constexpr uint32_t pos = 0x002b1908u, neg = 0x00d5e7f8u;
    const uint32_t sel = or_and<0x40404040u>(or_and<0x04040404u>(g32, (s8 & 0x55u) * 0x00104104u),
                                             (s8 & 0xaau) * 0x00820820u);
    lo = prmt(pos, neg, sel);
    hi = prmt(pos, neg, sel >> 16);
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

// ME1 (Q5_K, Q5_1): bits 0..3 of x to bit 0 of bytes 0..3 (the four shifted copies never overlap, so nothing carries)
__device__ __forceinline__ uint32_t spread4(uint32_t x) {
    return ((x & 0xfu) * 0x00204081u) & 0x01010101u;
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

// The GLU of one channel from its gate and up sums and FNFL's activation format of its 32-channel run (act.h's rule:
// lane i holds channel i of the run; the warp's 32 lanes take part): code and v out, d and s of the run in every lane.
// gate_up's glu_out and ME1's glu_finish both run it, so their GLU is one arithmetic.
__device__ __forceinline__ int8_t glu_run(float gs, float us, float & v, float & d, float & sv) {
    v = gs / (1.0f + expf(-gs)) * us;
    float amax = fabsf(v);
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    d = __fdiv_rn(amax, 127.0f);
    const float id = amax > 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
    const int32_t qv = min(max(__float2int_rn(v * id), -127), 127);
    int32_t sum = qv;
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    sv = d * static_cast<float>(sum);
    return static_cast<int8_t>(qv);
}

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
bool gate_up_q2_0(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);
bool gate_up_q4_k(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);
bool gate_up_q5_k(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st);
// ME10: the pair launches (gate_up_pair) of the IQ2 and IQ3 bodies at B = 4 and 5; the caller checked the arguments
bool gate_up_pair_iq2(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_pair_args & a,
                      cudaStream_t st);
bool gate_up_pair_iq3(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_pair_args & a,
                      cudaStream_t st);

// FMU2 (build item 1): a joint round's lanes' gate/up in one launch (gate_up_lanes) at B = rows (FMS13: the round's
// R; IQ2 2 .. 5, IQ3 4 .. 5), the IQ2 and IQ3 bodies (experts_gu_lanes.cu); the caller checked the arguments
bool gate_up_lanes_iq2(fmt                        f,
                       fn::dense::fmt             fg,
                       fn::dense::fmt             fu,
                       int32_t                    lanes,
                       int32_t                    rows,
                       const gate_up_lanes_args & a,
                       cudaStream_t               st);
bool gate_up_lanes_iq3(fmt                        f,
                       fn::dense::fmt             fg,
                       fn::dense::fmt             fu,
                       int32_t                    lanes,
                       int32_t                    rows,
                       const gate_up_lanes_args & a,
                       cudaStream_t               st);

} // namespace fn::experts::detail
