#pragma once

// APF2: the prompt experts' device helpers (moved verbatim from experts_prompt.cu, so the ladder's grouped kernels,
// fn/kernels/experts/experts_ladder.cu, unpack the same FP16 values): EX1's packed layouts, the gate/up and down
// loaders and unpackers, the m8n8k4 product and the task helpers. Device only.

#include "fn/kernels/experts/experts_prompt.h"
#include "fn/model/shapes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace fn::experts::epd {

constexpr int32_t D = fn::n_embd; // 2560
constexpr int32_t n_used = fn::n_expert_used;
constexpr int32_t n_expert = fn::n_expert;
constexpr int32_t n_group = n_ff_card / 32;       // gate/up tiles of 32 channels a card owns per expert
constexpr int32_t ks = 32;                        // K a step: one 32-weight sub-block
constexpr int32_t nt_threads = 640;               // gate/up: one weight column (channel of gate or up) a thread
constexpr int32_t dn_threads = 512;               // down: one output column a thread (a column group)
constexpr int32_t max_rt = ep_pass_rows / 8;      // row tiles of the largest task
constexpr int32_t a_tile = ep_pass_rows * ks * 2; // bytes of one FP16 activation tile
constexpr int32_t n_stage = ep_pass_rows * 4;     // threads staging the activation tile, 8 values each

static_assert(n_used * ep_max_rows == ep_max_pairs && n_ff_card == 320 && D == 2560 && n_expert == 512,
              "the shapes this file's layouts are written for");
static_assert(2 * n_ff_card == nt_threads && ep_col_group == dn_threads, "a thread a tile column");

// ---- EX1's packed gate/up layout (experts.cu): the facts this file reads -----------------------------------------
// Tile per (expert, 32-channel group g): rows 0..31 gate, 32..63 up; record rec = 8 j + rs (sub-block j, rows rs + 8 m)
// owns C main chunks [c][rec] of 16 bytes, then aux8 [rec] 8 B, sc8 [j / 2][rs] 8 B, d [j / 8][rs] 16 B.
struct gu_layout {
    int32_t block_bytes, chunks;
    bool aux8, sc8;
};

__host__ __device__ constexpr gu_layout gu_layout_of(fmt f) {
    return f == fmt::iq2_xxs   ? gu_layout{66, 4, false, false}
           : f == fmt::iq2_xs  ? gu_layout{74, 4, true, false}
           : f == fmt::iq2_s   ? gu_layout{82, 5, false, false}
           : f == fmt::iq3_xxs ? gu_layout{98, 6, false, false}
           : f == fmt::iq3_s   ? gu_layout{110, 6, true, true}
                               : gu_layout{0, 0, false, false};
}
__host__ __device__ constexpr int32_t tile_bytes(fmt f) {
    return nt_threads * gu_layout_of(f).block_bytes;
}
__host__ __device__ constexpr int32_t aux8_off(fmt f) {
    return gu_layout_of(f).chunks * nt_threads * 16;
}
__host__ __device__ constexpr int32_t sc8_off(fmt f) {
    return aux8_off(f) + (gu_layout_of(f).aux8 ? nt_threads * 8 : 0);
}
__host__ __device__ constexpr int32_t d_off(fmt f) {
    return sc8_off(f) + (gu_layout_of(f).sc8 ? 40 * 64 : 0);
}
static_assert(d_off(fmt::iq3_xxs) + 10 * 128 == tile_bytes(fmt::iq3_xxs) &&
                  d_off(fmt::iq3_s) + 10 * 128 == tile_bytes(fmt::iq3_s) &&
                  d_off(fmt::iq2_xxs) + 10 * 128 == tile_bytes(fmt::iq2_xxs) &&
                  d_off(fmt::iq2_xs) + 10 * 128 == tile_bytes(fmt::iq2_xs) &&
                  d_off(fmt::iq2_s) + 10 * 128 == tile_bytes(fmt::iq2_s),
              "EX1 tile bytes");

// ME1: the K-quants (the Q4 candidate's routed gate/up; experts_dev.cuh): C main chunks [c][rec] (Q4_K 8: chunk m =
// row m's 32 nibbles, byte t = weight t low and t + 16 high; Q5_K 10: [qh rows 0..3][r0..r3][qh rows 4..7][r4..r7],
// 4 qh bytes a row, bit t = bit 4 of weight t), then sm [rec] 16 B (byte 2 m: row m's 6-bit scale, 2 m + 1 its min),
// then d and dmin, each [j / 8][rs] 16 B
__host__ __device__ constexpr bool is_kq(fmt f) {
    return f == fmt::q4_k || f == fmt::q5_k;
}
__host__ __device__ constexpr int32_t kq_sm_off(fmt f) {
    return (f == fmt::q5_k ? 10 : 8) * nt_threads * 16;
}
__host__ __device__ constexpr int32_t kq_d_off(fmt f) {
    return kq_sm_off(f) + nt_threads * 16;
}
constexpr int32_t kq_dmin = 10 * 128; // dmin's plane after d's
// the bytes of one (expert, group) tile, every format
__host__ __device__ constexpr int32_t ep_tile_bytes(fmt f) {
    return is_kq(f) ? kq_d_off(f) + 2 * kq_dmin : tile_bytes(f);
}
static_assert(ep_tile_bytes(fmt::q4_k) == nt_threads * 148 && ep_tile_bytes(fmt::q5_k) == nt_threads * 180,
              "ME1 K-quant tile bytes (experts_dev.cuh)");

// the grid buffer (experts.h grid_table): iq2xxs, iq2xs, iq2s, iq3xxs, iq3s grids, kvalues_iq4nl
constexpr int32_t grid_off_iq2xxs = 0, grid_off_iq2xs = 2048, grid_off_iq2s = 6144, grid_off_iq3xxs = 14336,
                  grid_off_iq3s = 15360, kvalues_off = 17408;

// The grid tables, built per block from the grid buffer as rep bank-private copies: entry idx of copy c at
// (idx rep + c) eb bytes, and lane l reads copy l % rep, so a warp's random lookups do not conflict (eb 4: rep 32,
// each lane its own bank; eb 8: rep 16, each half-warp lane its own bank pair; IQ2_S: rep 8, half the conflicts of
// one copy). rep x eb is the stride of an index: 128 bytes (64 for IQ2_S).
struct grid_spec {
    int32_t entries, eb, rep, src_off;
};
__host__ __device__ constexpr grid_spec grid_of(fmt f) {
    return f == fmt::iq2_xxs   ? grid_spec{256, 8, 16, grid_off_iq2xxs}
           : f == fmt::iq2_xs  ? grid_spec{512, 8, 16, grid_off_iq2xs}
           : f == fmt::iq2_s   ? grid_spec{1024, 8, 8, grid_off_iq2s}
           : f == fmt::iq3_xxs ? grid_spec{256, 4, 32, grid_off_iq3xxs}
           : f == fmt::iq3_s   ? grid_spec{512, 4, 32, grid_off_iq3s}
                               : grid_spec{0, 0, 0, 0};
}
__host__ __device__ constexpr int32_t table_bytes(fmt f) {
    return grid_of(f).entries * grid_of(f).eb * grid_of(f).rep;
}
constexpr int32_t gu_tiles = 4; // gate/up's activation ring (PB3): tile j & 3 for step j, staged two steps ahead
__host__ __device__ constexpr int32_t gu_smem(fmt f) {
    return gu_tiles * a_tile + 16 * nt_threads + table_bytes(f);
}
static_assert(gu_smem(fmt::iq2_s) <= 96 * 1024 && gu_smem(fmt::iq3_s) <= 96 * 1024, "tables fit beside the tiles");

// ---- small helpers -----------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    const half2 r = __hmul2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r); // reinterpret: half2 is the 32-bit word's two halves
}
__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
    const half2 r = __hfma2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b),
                            *reinterpret_cast<const half2 *>(&c));
    return *reinterpret_cast<const uint32_t *>(&r);
}
__device__ __forceinline__ uint32_t hsub2(uint32_t a, uint32_t b) {
    const half2 r = __hsub2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r);
}
__device__ __forceinline__ float half_lo(uint32_t h2) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xffffu)));
}
// fp16(v) in both halves, rounded once from F32
__device__ __forceinline__ uint32_t h2_of(float v) {
    const uint32_t h = __half_as_ushort(__float2half_rn(v));
    return h | (h << 16);
}
// fp16(a) low, fp16(b) high, each rounded once from F32
__device__ __forceinline__ uint32_t h2_pack(float a, float b) {
    return static_cast<uint32_t>(__half_as_ushort(__float2half_rn(a))) |
           (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(b))) << 16);
}
__device__ __forceinline__ uint32_t byte_of(uint32_t x, int32_t k) {
    return (x >> (8 * k)) & 0xffu;
}

// A shared-memory tile row (64 bytes, 4 chunks of 16) as its 32-bit shared address with the swizzle s in bits 4..5:
// chunk q sits at row ^ (q << 4) (rows are 64-byte aligned, so the OR and the XOR do not carry). One register a row;
// the four chunk addresses are rebuilt with one LOP3 each rather than held across the K loop.
// threadIdx.x through a volatile read: not merged with an earlier read, so values derived from it are not held
__device__ __forceinline__ int32_t tid_fresh() {
    int32_t t;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
    return t;
}
// a thread's own int4 in shared memory, read where it is used (asm volatile: not hoisted into a register)
__device__ __forceinline__ int4 ld_ofs(const void * p) {
    int4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p)))
                 : "memory");
    return v;
}
// A line into L1 ahead of its __ldg (PB3): the staging threads' activation rows, two steps ahead, so the step's own
// load is an L1 hit rather than an L2 round trip ahead of the shared store and the barrier
__device__ __forceinline__ void prefetch_l1(const void * p) {
    asm volatile("prefetch.global.L1 [%0];" ::"l"(p));
}
// A line into L2 ahead of its __ldg (PB3): gate/up's weight lines, two steps ahead, so the load a step ahead waits on
// L2 rather than DRAM (EP1: long scoreboard 24 % on the weight loads' consumers)
__device__ __forceinline__ void prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
__device__ __forceinline__ uint32_t tile_row(const void * row, int32_t sw) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(row)) | (static_cast<uint32_t>(sw) << 4);
}
__device__ __forceinline__ void st_chunk(uint32_t row, int32_t q, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(row ^ (static_cast<uint32_t>(q) << 4)), "r"(a),
                 "r"(b), "r"(c), "r"(d)
                 : "memory");
}
// The activation tile's swizzle. A warp reads rows 8 t + r, one 16-byte chunk each, in quarter-warps of 8 lanes (rows
// 0..3 or 4..7, each twice: broadcasts): rows r and r + 2 swap chunks, so the 4 rows of a quarter-warp sit in 4 different
// bank groups and the loads and the staging stores are conflict-free (R6).
__device__ __forceinline__ int32_t asw(int32_t row) {
    return (row >> 1) & 1;
}

// D += A B: one m8n8k4 product per quadpair, FP16 inputs, F32 accumulators (HMMA.884.F32)
__device__ __forceinline__ void mma884(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// ---- table builders (once per block) -----------------------------------------------------------------------------
template <fmt F> __device__ void build_tables(uint8_t * tb, const uint8_t * grids) {
    constexpr grid_spec G = grid_of(F);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    for (int32_t i = tid; i < G.entries * G.rep; i += static_cast<int32_t>(blockDim.x)) {
        const int32_t idx = i / G.rep;
        if constexpr (G.eb == 4) {
            reinterpret_cast<uint32_t *>(tb)[i] = __ldg(reinterpret_cast<const uint32_t *>(grids + G.src_off) + idx);
        } else {
            reinterpret_cast<uint2 *>(tb)[i] = __ldg(reinterpret_cast<const uint2 *>(grids + G.src_off) + idx);
        }
    }
}

// ---- gate/up: one sub-block of one column, loaded a step ahead and unpacked to FP16 B fragments ------------------
// Thread constants: the column's row rs and row m of its EX1 record, and the byte offsets of its pieces.
struct gu_col {
    int32_t rs, m;
    int32_t o_a, o_b, o_c; // format-specific offsets in the tile (sub-block 0)
};

template <fmt F> __device__ __forceinline__ gu_col gu_col_of(int32_t rs, int32_t m) {
    gu_col c = {rs, m, 0, 0, 0};
    const int32_t rec16 = rs * 16; // record rs of sub-block 0
    if constexpr (is_kq(F)) {
        // ME1: o_a row m's main chunk, o_b its qh word (Q5_K), o_c its scale and min bytes
        const int32_t cm = F == fmt::q5_k ? (m < 4 ? 1 + m : 2 + m) : m;
        c.o_a = cm * nt_threads * 16 + rec16;
        c.o_b = (m < 4 ? 0 : 5) * nt_threads * 16 + rec16 + 4 * (m & 3);
        c.o_c = kq_sm_off(F) + rec16 + 2 * m;
        return c;
    } else if constexpr (F == fmt::iq3_xxs || F == fmt::iq3_s) {
        const int32_t h = m >> 2, i4 = m & 3;
        const int32_t qc = 3 * h + (i4 == 0 ? 0 : i4 == 1 ? 1 : 2), qb = i4 == 3 ? 8 : 0;
        const int32_t sc = 3 * h + (i4 < 2 ? 0 : 1), sb = (i4 & 1) ? 12 : 8;
        c.o_a = qc * nt_threads * 16 + rec16 + qb; // q: 8 bytes
        c.o_b = sc * nt_threads * 16 + rec16 + sb; // signs: 4 bytes
    } else if constexpr (F == fmt::iq2_xxs || F == fmt::iq2_xs) {
        c.o_a = (m >> 1) * nt_threads * 16 + rec16 + 8 * (m & 1); // 8 bytes
    } else {
        c.o_a = (1 + (m >> 1)) * nt_threads * 16 + rec16 + 8 * (m & 1); // q 4 bytes, signs 4 bytes
        c.o_b = rec16 + m;                                              // qh byte; scales byte at + 8
    }
    if constexpr (F == fmt::iq2_xs || F == fmt::iq3_s) {
        c.o_c = aux8_off(F) + rs * 8 + m; // aux8 byte: IQ2_XS scales, IQ3_S qh
    }
    return c;
}

struct gu_raw {
    uint2 q;
    uint32_t b; // signs (IQ3), qh | scales << 8 (IQ2_S), scales (IQ2_XS)
    uint32_t d; // FP16 d of the superblock (IQ3_S: qh << 16, scale nibble << 24 above it)
};

// w: the expert weights; o: this column's byte offsets in them for sub-block 0 (below 2^31: 512 x 10 tiles of at most
// 70,400 bytes): x the q bytes, y the sign or qh/scale bytes, z the FP16 d, w the aux8 byte (IQ2_XS, IQ3_S).
template <fmt F> __device__ __forceinline__ gu_raw gu_load(const uint8_t * w, const int4 & o, int32_t j) {
    gu_raw r;
    r.d = __ldg(reinterpret_cast<const unsigned short *>(w + (o.z + (j >> 3) * 128)));
    r.q = __ldg(reinterpret_cast<const uint2 *>(w + (o.x + j * 128))); // record 8 j + rs: 128 bytes a sub-block
    if constexpr (F == fmt::iq3_xxs) {
        r.b = __ldg(reinterpret_cast<const uint32_t *>(w + (o.y + j * 128)));
    } else if constexpr (F == fmt::iq3_s) {
        r.b = __ldg(reinterpret_cast<const uint32_t *>(w + (o.y + j * 128)));
        const uint32_t qh = __ldg(w + (o.w + j * 64));
        const uint32_t sc = __ldg(w + (o.w - aux8_off(F) + sc8_off(F) + (j >> 1) * 64));
        r.d |= (qh << 16) | (((sc >> (4 * (j & 1))) & 0xfu) << 24);
    } else if constexpr (F == fmt::iq2_xxs) {
        r.b = 0;
    } else if constexpr (F == fmt::iq2_xs) {
        r.b = __ldg(w + (o.w + j * 64));
    } else {
        r.b = __ldg(w + (o.y + j * 128)) | (static_cast<uint32_t>(__ldg(w + (o.y + j * 128 + 8))) << 8);
    }
    return r;
}

// the q and sign/qh lines gu_load reads for sub-block j (d and aux8 lines serve 8 and 2 sub-blocks), into L2
template <fmt F> __device__ __forceinline__ void gu_prefetch(const uint8_t * w, const int4 & o, int32_t j) {
    prefetch_l2(w + (o.x + j * 128));
    if constexpr (F == fmt::iq3_xxs || F == fmt::iq3_s || F == fmt::iq2_s) {
        prefetch_l2(w + (o.y + j * 128));
    }
}

// this column's offsets (gu_load) for tile offset to
template <fmt F> __device__ __forceinline__ int4 gu_offsets(const gu_col & c, int32_t to) {
    if constexpr (is_kq(F)) {
        // ME1: x the main chunk, y the qh word, z d (dmin kq_dmin after it), w the scale and min bytes
        return make_int4(to + c.o_a, to + c.o_b, to + kq_d_off(F) + c.rs * 16 + 2 * c.m, to + c.o_c);
    }
    return make_int4(to + c.o_a, to + c.o_b, to + d_off(F) + c.rs * 16 + 2 * c.m, to + c.o_c);
}

// ME1: one K-quant sub-block of the column: its 32 nibbles, the qh word (Q5_K), the superblock's FP16 d | dmin << 16
// and the sub-block's 6-bit scale | min << 8
struct kq_raw {
    uint4 q;
    uint32_t h, d, s;
};
template <fmt F> __device__ __forceinline__ kq_raw kq_load(const uint8_t * w, const int4 & o, int32_t j) {
    kq_raw r;
    r.d = __ldg(reinterpret_cast<const unsigned short *>(w + (o.z + (j >> 3) * 128))) |
          (static_cast<uint32_t>(__ldg(reinterpret_cast<const unsigned short *>(w + (o.z + kq_dmin + (j >> 3) * 128))))
           << 16);
    r.s = __ldg(reinterpret_cast<const unsigned short *>(w + (o.w + j * 128)));
    r.q = __ldg(reinterpret_cast<const uint4 *>(w + (o.x + j * 128)));
    r.h = F == fmt::q5_k ? __ldg(reinterpret_cast<const uint32_t *>(w + (o.y + j * 128))) : 0u;
    return r;
}
template <fmt F> __device__ __forceinline__ void kq_prefetch(const uint8_t * w, const int4 & o, int32_t j) {
    prefetch_l2(w + (o.x + j * 128));
    prefetch_l2(w + (o.w + j * 128));
    if constexpr (F == fmt::q5_k) {
        prefetch_l2(w + (o.y + j * 128));
    }
}
template <fmt F> using raw_of = std::conditional_t<is_kq(F), kq_raw, gu_raw>;
template <fmt F> __device__ __forceinline__ raw_of<F> raw_load(const uint8_t * w, const int4 & o, int32_t j) {
    if constexpr (is_kq(F)) {
        return kq_load<F>(w, o, j);
    } else {
        return gu_load<F>(w, o, j);
    }
}
template <fmt F> __device__ __forceinline__ void raw_prefetch(const uint8_t * w, const int4 & o, int32_t j) {
    if constexpr (is_kq(F)) {
        kq_prefetch<F>(w, o, j);
    } else {
        gu_prefetch<F>(w, o, j);
    }
}
// bits 0..3 of x to bit 0 of bytes 0..3 (the four shifted copies never overlap, so nothing carries)
__device__ __forceinline__ uint32_t spread4(uint32_t x) {
    return ((x & 0xfu) * 0x00204081u) & 0x01010101u;
}

// The sign mask of half-word k (weights 2 k, 2 k + 1 of an 8-weight group with sign bits s8): s8 x K puts bit 2 k at
// bit 15 and bit 2 k + 1 at bit 31 (two copies of s8 that cannot overlap or carry: s8 < 2^10); the caller's XOR keeps
// only those two bits. With s8 = s7 + 128 popc(s7), bit 7 is the parity sign of the 7-bit formats.
__device__ __forceinline__ uint32_t sign_xor(uint32_t h, uint32_t s8, int32_t k) {
    const uint32_t m = s8 * (k == 0 ? 0x40008000u : k == 1 ? 0x10002000u : k == 2 ? 0x04000800u : 0x01000200u);
    return h ^ (m & 0x80008000u); // one LOP3
}
__device__ __forceinline__ uint32_t sign8_of7(uint32_t s7) {
    return s7 + (static_cast<uint32_t>(__popc(s7)) << 7);
}

// The scale of chunks 2 h, 2 h + 1 (one per sub-block, or one per 16 weights for IQ2_XS and IQ2_S): s (FP16 in both
// halves) and c = -1024 s (exact)
struct gu_scale {
    uint32_t s, c;
};
template <fmt F> __device__ __forceinline__ gu_scale gu_scale_of(const gu_raw & r, int32_t h) {
    const float dh = half_lo(r.d);
    uint32_t s;
    if constexpr (F == fmt::iq3_xxs) {
        s = h2_of(dh * static_cast<float>((r.b >> 27) | 1u) * 0.25f); // d (2 ls + 1) / 4; bit 27 is a sign bit
    } else if constexpr (F == fmt::iq3_s) {
        s = h2_of(dh * static_cast<float>(1u + 2u * (r.d >> 24)));
    } else if constexpr (F == fmt::iq2_xxs) {
        s = h2_of(dh * static_cast<float>((r.q.y >> 27) | 1u) * 0.125f);
    } else {
        const uint32_t sc = F == fmt::iq2_xs ? r.b : (r.b >> 8);
        s = h2_of(dh * static_cast<float>(2u * ((sc >> (4 * h)) & 0xfu) + 1u) * 0.125f);
    }
    return {s, hmul2(s, 0xe400e400u)};
}
// ME1: a K-quant sub-block's scale s = d x sc and its min term c = -(dmin x min), each rounded once to FP16 from the
// exact F32 product
__device__ __forceinline__ gu_scale kq_scale(const kq_raw & r) {
    const float d = half_lo(r.d), dm = half_lo(r.d >> 16);
    return {h2_of(d * static_cast<float>(r.s & 0xffu)), h2_of(-(dm * static_cast<float>(r.s >> 8)))};
}
__host__ __device__ constexpr int32_t task_rows_of(fmt f) {
    return f == fmt::iq3_xxs || f == fmt::iq2_xxs || f == fmt::iq2_xs || f == fmt::iq2_s || f == fmt::iq3_s ||
                   is_kq(f)
               ? 40
               : 0;
}
__host__ __device__ constexpr bool two_scales(fmt f) {
    return f == fmt::iq2_xs || f == fmt::iq2_s;
}

// Chunk l of the column's sub-block: weights 8 l .. 8 l + 7 as the B fragments of two k4 products, FP16 code x scale
// rounded once. Each grid byte g under 0x64 is 1024 + g exactly; x s - 1024 s = g s; the sign flips the result's sign
// bit (exact: rounding is symmetric). tb: the replicated table; lo: this lane's copy offset.
template <fmt F>
__device__ __forceinline__ uint4 gu_chunk(const gu_raw & r, const gu_scale & p, const uint8_t * tb, uint32_t lo,
                                          int32_t l) {
    constexpr uint32_t c64 = 0x64646464u;
    const auto hw = [&](uint32_t g, uint32_t sel, uint32_t s2, uint32_t c2) {
        return hfma2(__byte_perm(g, c64, sel), s2, c2);
    };
    if constexpr (F == fmt::iq3_xxs || F == fmt::iq3_s) {
        const uint32_t qw = l < 2 ? r.q.x : r.q.y;
        uint32_t s8, i0, i1;
        if constexpr (F == fmt::iq3_xxs) {
            s8 = sign8_of7((r.b >> (7 * l)) & 0x7fu);
            i0 = byte_of(qw, 2 * (l & 1));
            i1 = byte_of(qw, 2 * (l & 1) + 1);
        } else {
            const uint32_t qh = (r.d >> 16) & 0xffu;
            s8 = (r.b >> (8 * l)) & 0xffu;
            i0 = byte_of(qw, 2 * (l & 1)) | (((qh >> (2 * l)) & 1u) << 8);
            i1 = byte_of(qw, 2 * (l & 1) + 1) | (((qh >> (2 * l + 1)) & 1u) << 8);
        }
        const uint32_t g0 = *reinterpret_cast<const uint32_t *>(tb + ((i0 << 7) | lo));
        const uint32_t g1 = *reinterpret_cast<const uint32_t *>(tb + ((i1 << 7) | lo));
        return make_uint4(sign_xor(hw(g0, 0x4140, p.s, p.c), s8, 0), sign_xor(hw(g0, 0x4342, p.s, p.c), s8, 1),
                          sign_xor(hw(g1, 0x4140, p.s, p.c), s8, 2), sign_xor(hw(g1, 0x4342, p.s, p.c), s8, 3));
    } else {
        uint32_t s8, idx;
        if constexpr (F == fmt::iq2_xxs) {
            s8 = sign8_of7((r.q.y >> (7 * l)) & 0x7fu);
            idx = byte_of(r.q.x, l) << 7;
        } else if constexpr (F == fmt::iq2_xs) {
            const uint32_t code = ((l < 2 ? r.q.x : r.q.y) >> (16 * (l & 1))) & 0xffffu;
            s8 = sign8_of7(code >> 9);
            idx = (code & 511u) << 7;
        } else {
            const uint32_t qh = r.b & 0xffu;
            s8 = (r.q.y >> (8 * l)) & 0xffu;
            idx = (byte_of(r.q.x, l) | ((qh << (8 - 2 * l)) & 0x300u)) << 6;
        }
        const uint32_t s2 = p.s, c2 = p.c;
        const uint2 g = *reinterpret_cast<const uint2 *>(tb + (idx | lo));
        return make_uint4(sign_xor(hw(g.x, 0x4140, s2, c2), s8, 0), sign_xor(hw(g.x, 0x4342, s2, c2), s8, 1),
                          sign_xor(hw(g.y, 0x4140, s2, c2), s8, 2), sign_xor(hw(g.y, 0x4342, s2, c2), s8, 3));
    }
}

// ME1: chunk l of a K-quant sub-block, weights 8 l .. 8 l + 7: code q (5 bits with qh) under 0x64 is 1024 + q exactly,
// minus 1024 is q; q x s + c rounds once
template <fmt F> __device__ __forceinline__ uint4 kq_chunk(const kq_raw & r, const gu_scale & p, int32_t l) {
    const int32_t sh = l >= 2 ? 4 : 0;
    uint32_t v0 = ((l & 1 ? r.q.z : r.q.x) >> sh) & 0x0f0f0f0fu;
    uint32_t v1 = ((l & 1 ? r.q.w : r.q.y) >> sh) & 0x0f0f0f0fu;
    if constexpr (F == fmt::q5_k) {
        const uint32_t hb = r.h >> (8 * l);
        v0 |= spread4(hb) << 4;
        v1 |= spread4(hb >> 4) << 4;
    }
    constexpr uint32_t c64 = 0x64646464u, c1024 = 0x64006400u;
    const auto hw = [&](uint32_t v, uint32_t sel) { return hfma2(hsub2(__byte_perm(v, c64, sel), c1024), p.s, p.c); };
    return make_uint4(hw(v0, 0x4140), hw(v0, 0x4342), hw(v1, 0x4140), hw(v1, 0x4342));
}

// ---- down: one 32-weight step of one output row -------------------------------------------------------------------
// EX1's packed down layout (experts.cu): Q2_0 chunks [j5][n] 16 B then d [j5][n] FP16; IQ4_NL chunks [j10][n] 16 B
// then d [j5][n][2] FP16 (j5 = 64 inputs, j10 = 32).
__host__ __device__ constexpr int32_t dn_bytes(fmt f) {
    return f == fmt::q5_1 ? 10 * D * 24 : f == fmt::q8_0 ? 10 * D * 34 : f == fmt::q2_0 ? 5 * D * 18 : 10 * D * 18;
}

struct dn_raw {
    uint4 q; // IQ4_NL: the block's 16 qs bytes; Q2_0: x, y the step's 8 qs bytes
    uint32_t d;
};
// ME1: Q5_1 (q as IQ4_NL's, d the block's FP16 d | m << 16, h its qh word); Q8_0 (q, q1 the block's int8 weights
// 0..15 and 16..31)
struct dn_raw_q5 {
    uint4 q;
    uint32_t d, h;
};
struct dn_raw_q8 {
    uint4 q, q1;
    uint32_t d;
};
template <fmt F>
using dn_raw_of = std::conditional_t<F == fmt::q5_1, dn_raw_q5, std::conditional_t<F == fmt::q8_0, dn_raw_q8, dn_raw>>;
// a staging thread's GLU values of one down step: Q2_0 a and b (16 halves), IQ4_NL a (8 halves)
struct glu_raw {
    uint4 a, b;
};

// eo: the expert's byte offset (below 2^31: 512 x 460,800 bytes); n16: 16 x the output row
template <fmt F>
__device__ __forceinline__ dn_raw_of<F> dn_load(const uint8_t * w, int32_t eo, int32_t n16, int32_t j) {
    dn_raw_of<F> r;
    if constexpr (F == fmt::q5_1) {
        // ME1: the qs as IQ4_NL's, then [j5][n] 16 B: qh0, qh1, d0 | m0 << 16, d1 | m1 << 16
        const int32_t e = eo + 10 * D * 16 + (j >> 1) * (D * 16) + n16 + 4 * (j & 1);
        r.q = __ldg(reinterpret_cast<const uint4 *>(w + (eo + n16 + j * (D * 16))));
        r.h = __ldg(reinterpret_cast<const uint32_t *>(w + e));
        r.d = __ldg(reinterpret_cast<const uint32_t *>(w + (e + 8)));
    } else if constexpr (F == fmt::q8_0) {
        // ME1: chunks [j5][c][n] (block 2 j5 + c / 2's weights 16 (c & 1) ..), then d [j5][n][2] FP16
        const int32_t o = eo + ((j >> 1) * 4 + 2 * (j & 1)) * (D * 16) + n16;
        r.q = __ldg(reinterpret_cast<const uint4 *>(w + o));
        r.q1 = __ldg(reinterpret_cast<const uint4 *>(w + (o + D * 16)));
        r.d = __ldg(reinterpret_cast<const unsigned short *>(
            w + (eo + 20 * D * 16 + (n16 >> 2) + (j >> 1) * (D * 4) + ((j & 1) << 1))));
    } else if constexpr (F == fmt::iq4_nl) {
        constexpr int32_t d_plane = 10 * D * 16;
        r.q = __ldg(reinterpret_cast<const uint4 *>(w + (eo + n16 + j * (D * 16))));
        r.d = __ldg(reinterpret_cast<const unsigned short *>(
            w + (eo + d_plane + (n16 >> 2) + (j >> 1) * (D * 4) + ((j & 1) << 1))));
    } else {
        constexpr int32_t d_plane = 5 * D * 16;
        const uint2 h = __ldg(reinterpret_cast<const uint2 *>(w + (eo + n16 + (j >> 1) * (D * 16) + ((j & 1) << 3))));
        r.q = make_uint4(h.x, h.y, 0u, 0u);
        r.d = __ldg(reinterpret_cast<const unsigned short *>(w + (eo + d_plane + (n16 >> 3) + (j >> 1) * (D * 2))));
    }
    return r;
}

// kvalues_iq4nl + 128 of the 8 nibbles of q (ggml's get_int_from_table_16 on the biased table): the low nibbles'
// values in lo, the high ones' in hi
__device__ __forceinline__ void table16(uint32_t q, const uint32_t (&t)[4], uint32_t & lo, uint32_t & hi) {
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const uint32_t l0 = __byte_perm(t[0], t[1], q), h0 = __byte_perm(t[2], t[3], q);
    const uint32_t l1 = __byte_perm(t[0], t[1], q >> 16), h1 = __byte_perm(t[2], t[3], q >> 16);
    const uint32_t x0 = __byte_perm(l0, h0, sel), x1 = __byte_perm(l1, h1, sel >> 16);
    lo = __byte_perm(x0, x1, 0x6420);
    hi = __byte_perm(x0, x1, 0x7531);
}

// Chunk c of one output row's 32-weight step, in a fixed K order per format (the activation tile is permuted alike):
// IQ4_NL chunk c holds weights 4 c .. 4 c + 3 and 16 + 4 c .. 16 + 4 c + 3 (qs word c alone); Q2_0 chunk 2 h + p holds
// weights 16 h + 2 p + (0, 4, 8, 12, 1, 5, 9, 13) of the step.
template <fmt F>
__device__ __forceinline__ uint4 dn_chunk(const dn_raw_of<F> & r, const uint32_t (&kv)[4], int32_t c) {
    if constexpr (F == fmt::q5_1 || F == fmt::q8_0) {
        // ME1, in IQ4_NL's K order. Q5_1: code q (5 bits with qh) under 0x64 is 1024 + q exactly, minus 1024 is q;
        // q x d + m rounds once. Q8_0: q + 128 under 0x64 is 1152 + q, minus 1152 is q; times d rounds once
        constexpr uint32_t c64 = 0x64646464u;
        const uint32_t d2 = (r.d & 0xffffu) | (r.d << 16);
        const uint32_t w = c == 0 ? r.q.x : c == 1 ? r.q.y : c == 2 ? r.q.z : r.q.w;
        uint32_t lo, hi;
        if constexpr (F == fmt::q5_1) {
            const uint32_t m2 = (r.d >> 16) | (r.d & 0xffff0000u);
            lo = (w & 0x0f0f0f0fu) | (spread4(r.h >> (4 * c)) << 4);
            hi = ((w >> 4) & 0x0f0f0f0fu) | (spread4(r.h >> (16 + 4 * c)) << 4);
            const auto h = [&](uint32_t v, uint32_t s) {
                return hfma2(hsub2(__byte_perm(v, c64, s), 0x64006400u), d2, m2);
            };
            return make_uint4(h(lo, 0x4140), h(lo, 0x4342), h(hi, 0x4140), h(hi, 0x4342));
        } else {
            lo = w ^ 0x80808080u;
            hi = (c == 0 ? r.q1.x : c == 1 ? r.q1.y : c == 2 ? r.q1.z : r.q1.w) ^ 0x80808080u;
            const auto h = [&](uint32_t v, uint32_t s) {
                return hmul2(hsub2(__byte_perm(v, c64, s), 0x64806480u), d2);
            };
            return make_uint4(h(lo, 0x4140), h(lo, 0x4342), h(hi, 0x4140), h(hi, 0x4342));
        }
    } else if constexpr (F == fmt::iq4_nl) {
        // 128 + code under 0x64 is 1152 + code exactly; minus 1152 is the code; times d rounds once
        const uint32_t d2 = (r.d & 0xffffu) | (r.d << 16);
        constexpr uint32_t c64 = 0x64646464u, c1152 = 0x64806480u;
        const auto h = [&](uint32_t v, uint32_t s) { return hmul2(hsub2(__byte_perm(v, c64, s), c1152), d2); };
        uint32_t lo, hi;
        table16(c == 0 ? r.q.x : c == 1 ? r.q.y : c == 2 ? r.q.z : r.q.w, kv, lo, hi);
        return make_uint4(h(lo, 0x4140), h(lo, 0x4342), h(hi, 0x4140), h(hi, 0x4342));
    } else {
        // Q2_0: value (code - 1) d for codes 0..3: -d, 0, d, 2d, all exact. ta:tb holds the four FP16 values; a byte
        // 34 code + 16 of a PRMT selector picks the two bytes of code's value. The codes of weights s, s + 4, s + 8,
        // s + 12 of a 16-weight word as bytes give, through the selector's low half, weights (s, s + 4), through its
        // high half (s + 8, s + 12).
        const uint32_t dh = r.d & 0xffffu;
        const uint32_t d2x = __half_as_ushort(__hadd(__ushort_as_half(static_cast<unsigned short>(dh)),
                                                     __ushort_as_half(static_cast<unsigned short>(dh))));
        const uint32_t ta = dh ^ 0x8000u, tb = dh | (d2x << 16);
        const uint32_t wv = c < 2 ? r.q.x : r.q.y;
        const int32_t s0 = 2 * (c & 1);
        const uint32_t z0 = ((wv >> (2 * s0)) & 0x03030303u) * 0x22u + 0x10101010u;
        const uint32_t z1 = ((wv >> (2 * s0 + 2)) & 0x03030303u) * 0x22u + 0x10101010u;
        return make_uint4(__byte_perm(ta, tb, z0), __byte_perm(ta, tb, z0 >> 16), __byte_perm(ta, tb, z1),
                          __byte_perm(ta, tb, z1 >> 16));
    }
}

// ---- the product, shared by gate/up and down ----------------------------------------------------------------------
// arow: the activation tile row as two plain pointers (even: for even chunks, odd: odd), so every A load is base +
// immediate (the row swizzle s is one bit: chunk q sits at q ^ s = q + s for even q, q - s for odd q)
struct a_rows {
    const uint8_t * even;
    const uint8_t * odd;
};
__device__ __forceinline__ a_rows a_row(const uint8_t * row, int32_t sw) {
    return {row + 16 * sw, row - 16 * sw};
}

// Chunk q (8 K values) of every row tile t < nt against the lane's B fragments b (its column's same 8 K values):
// quadpair qp holds 8 columns, all quadpairs the same 8 activation rows of tile t.
// NT row tiles, a compile-time count: a guard t < nt compiles to predicated HMMA, and a predicated-off HMMA still
// occupies the tensor pipe (EP1 change 1, ncu: 181.9M HMMA issued for 112.5M predicated on)
template <int NT>
__device__ __forceinline__ void mma_chunk(float (&acc)[NT][8], const uint4 & b, const a_rows & ar, int32_t q) {
    const uint8_t * ab = (q & 1 ? ar.odd : ar.even) + 16 * q;
#pragma unroll
    for (int32_t t = 0; t < NT; ++t) {
        const uint4 a = *reinterpret_cast<const uint4 *>(ab + 8 * t * 64);
        mma884(acc[t], a.x, a.y, b.x, b.y);
        mma884(acc[t], a.z, a.w, b.z, b.w);
    }
}

// run(std::integral_constant<int, nt>) for nt in [1, RT]: the task's row tiles as a compile-time count
template <int RT, typename F> __device__ __forceinline__ void with_nt(int32_t nt, const F & run) {
    switch (nt) {
        case 1:
            run(std::integral_constant<int, 1>{});
            break;
        case 2:
            if constexpr (RT >= 2) {
                run(std::integral_constant<int, 2>{});
            }
            break;
        case 3:
            if constexpr (RT >= 3) {
                run(std::integral_constant<int, 3>{});
            }
            break;
        case 4:
            if constexpr (RT >= 4) {
                run(std::integral_constant<int, 4>{});
            }
            break;
        case 5:
            if constexpr (RT >= 5) {
                run(std::integral_constant<int, 5>{});
            }
            break;
        case 6:
            if constexpr (RT >= 6) {
                run(std::integral_constant<int, 6>{});
            }
            break;
        default:
            break;
    }
}

__device__ __forceinline__ int32_t n_tasks_of(const ep_route & r) {
    return *reinterpret_cast<const volatile int32_t *>(r.counters); // written by ep_group, before this kernel
}

struct task_t {
    int32_t e, first, n;
};

__device__ __forceinline__ task_t task_of(const int32_t * tasks, int32_t i) {
    const int4 t = __ldg(reinterpret_cast<const int4 *>(tasks) + i);
    return {t.x, t.y, t.z};
}
// ME4: the block's task: a ring group's first task added (the cache instantiation with task_off), computed where it is
// needed rather than held across the K loop (gate/up has no register to spare at 96)
template <bool X, typename A> __device__ __forceinline__ int32_t block_task(const A & a) {
    if constexpr (X) {
        if (a.task_off != nullptr) {
            return static_cast<int32_t>(blockIdx.x) + __ldg(a.task_off + a.e_lo);
        }
    }
    return static_cast<int32_t>(blockIdx.x);
}
// ME4: block_task for gate/up's epilogue, the group's first task read volatile (as task_rows) rather than with
// __ldg: at 96 registers ptxas then allocates the IQ2_XXS and IQ3_S cache K loops without spills (ME4)
template <bool X, typename A> __device__ __forceinline__ int32_t block_task_fresh(const A & a) {
    if constexpr (X) {
        if (a.task_off != nullptr) {
            return static_cast<int32_t>(blockIdx.x) + *reinterpret_cast<const volatile int32_t *>(a.task_off + a.e_lo);
        }
    }
    return static_cast<int32_t>(blockIdx.x);
}

// the epilogue's (first, n), re-read rather than held across the K loop
__device__ __forceinline__ int2 task_rows(const int32_t * tasks, int32_t i) {
    const volatile int32_t * t = tasks + 4 * i;
    return make_int2(t[1], t[2]);
}

// the C element i of a lane: its row and column in the quadpair's 8 x 8 block (m8n8k4, F32 accumulators)
__device__ __forceinline__ int32_t c_row(int32_t lane, int32_t i) {
    return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}
__device__ __forceinline__ int32_t c_col(int32_t lane, int32_t i) {
    return (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
}

}  // namespace fn::experts::epd
