#include "fn/kernels/experts/experts_prompt.h"
#include "fn/model/shapes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <type_traits>

namespace fn::experts {
namespace {

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
    if constexpr (F == fmt::iq3_xxs || F == fmt::iq3_s) {
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
    return make_int4(to + c.o_a, to + c.o_b, to + d_off(F) + c.rs * 16 + 2 * c.m, to + c.o_c);
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
__host__ __device__ constexpr int32_t task_rows_of(fmt f) {
    return f == fmt::iq3_xxs || f == fmt::iq2_xxs || f == fmt::iq2_xs || f == fmt::iq2_s || f == fmt::iq3_s ? 40 : 0;
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

// ---- down: one 32-weight step of one output row -------------------------------------------------------------------
// EX1's packed down layout (experts.cu): Q2_0 chunks [j5][n] 16 B then d [j5][n] FP16; IQ4_NL chunks [j10][n] 16 B
// then d [j5][n][2] FP16 (j5 = 64 inputs, j10 = 32).
__host__ __device__ constexpr int32_t dn_bytes(fmt f) {
    return f == fmt::q2_0 ? 5 * D * 18 : 10 * D * 18; // = down_bytes(f)
}

struct dn_raw {
    uint4 q; // IQ4_NL: the block's 16 qs bytes; Q2_0: x, y the step's 8 qs bytes
    uint32_t d;
};
// a staging thread's GLU values of one down step: Q2_0 a and b (16 halves), IQ4_NL a (8 halves)
struct glu_raw {
    uint4 a, b;
};

// eo: the expert's byte offset (below 2^31: 512 x 460,800 bytes); n16: 16 x the output row
template <fmt F> __device__ __forceinline__ dn_raw dn_load(const uint8_t * w, int32_t eo, int32_t n16, int32_t j) {
    dn_raw r;
    if constexpr (F == fmt::iq4_nl) {
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
template <fmt F> __device__ __forceinline__ uint4 dn_chunk(const dn_raw & r, const uint32_t (&kv)[4], int32_t c) {
    if constexpr (F == fmt::iq4_nl) {
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

// ---- gate/up kernel ----------------------------------------------------------------------------------------------
// A block a task (grid ep_max_tasks; blocks past the task count exit at once), one resident an SM: 640 threads.
// Thread (w, lane) owns column c = 32 w + 8 qp + r of the warp's 32: channel 16 w + 8 (qp & 1) + r of this card, gate
// for qp < 2, up otherwise; its lane is that column's B fragment holder (quadpair qp, row r of m8n8k4), so it unpacks
// its own column's sub-block, chunk by chunk, straight into the products: each weight is unpacked once a task and no
// weight passes through shared memory. Threads below 192 stage the activation tiles (PB3: a ring of four, step j's
// tile staged during step j - 2, one barrier every second step):
// row t / 4 of the task, 8 values (t % 4) of the step's 32 FP16 values, as the producer wrote them.
template <fmt F> __global__ void __launch_bounds__(nt_threads, 1) ep_gate_up_kernel(ep_gate_up_args a) {
    extern __shared__ uint4 smem_u4[];
    uint8_t * smem = reinterpret_cast<uint8_t *>(smem_u4);
    uint8_t * at = smem;
    uint8_t * ofs = smem + gu_tiles * a_tile; // [640] int4: each column's load offsets
    uint8_t * tb = ofs + 16 * nt_threads;
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t w = tid >> 5, lane = tid & 31;
    const int32_t qp = (lane >> 2) & 3, r = (lane & 3) | ((lane >> 2) & 4);
    const a_rows arow = a_row(at + r * 64, asw(r)); // activation tile 0's row r; row tile t at + 512 t
    const bool up = qp >= 2;
    const int32_t ch = 16 * w + 8 * (qp & 1) + r; // channel of this card, 0..319
    const int32_t g = ch >> 5;
    const gu_col col = gu_col_of<F>(r, (up ? 4 : 0) + ((ch & 31) >> 3));
    constexpr grid_spec G = grid_of(F);
    constexpr int32_t RT = task_rows_of(F) / 8;
    const uint32_t lo = static_cast<uint32_t>((lane % G.rep) * G.eb);
    const int32_t srow = tid >> 2, sq = tid & 3; // staging
    const uint32_t srow_a = tile_row(at + srow * 64, asw(srow));
    constexpr int32_t n_steps = D / ks;
    constexpr int32_t TB = tile_bytes(F);
    const int32_t task = static_cast<int32_t>(blockIdx.x);
    if (task >= n_tasks_of(a.route)) {
        return;
    }
    build_tables<F>(tb, a.grids);
    {
        const task_t tk = task_of(a.route.tasks, task);
        const int32_t nt = (tk.n + 7) >> 3;
        // this column's load offsets, kept in shared memory and re-read each step (registers are the limit here)
        reinterpret_cast<int4 *>(ofs)[tid] = gu_offsets<F>(col, (tk.e * n_group + g) * TB);
        // the staged row's FP16 values (rows of 5,120 bytes, 16-byte aligned), as a 32-bit offset in halves
        const bool sreal = tid < n_stage && srow < tk.n;
        const int32_t xo = sreal ? (__ldg(a.route.sorted + tk.first + srow) / n_used) * D + 8 * sq : 0;
        const auto x_load = [&](int32_t j) -> uint4 {
            return sreal ? __ldg(reinterpret_cast<const uint4 *>(a.x + xo + j * ks)) : make_uint4(0, 0, 0, 0);
        };
        const auto x_stage = [&](const uint4 & v, uint32_t abuf) {
            if (tid < n_stage) {
                st_chunk(abuf, sq, v.x, v.y, v.z, v.w);
            }
        };
        // a step's 64 bytes of a row: one line, one of its 4 threads (not for IQ2_S and IQ3_S: at 96 registers they
        // spill with it)
        const auto x_prefetch = [&](int32_t j) {
            if (F != fmt::iq2_s && F != fmt::iq3_s && sreal && sq == 0) {
                prefetch_l1(a.x + xo + j * ks);
            }
        };

        with_nt<RT>(nt, [&](auto ntc) {
            constexpr int32_t NT = decltype(ntc)::value;
            float acc[NT][8];
#pragma unroll
            for (int32_t t = 0; t < NT; ++t) {
#pragma unroll
                for (int32_t i = 0; i < 8; ++i) {
                    acc[t][i] = 0.0f;
                }
            }
            // two raw slots with compile-time indices (R12): slot j & 1 holds sub-block j; sub-block j + 1 is loaded
            // during step j (a step ahead of its unpacking)
            gu_raw raw[2];
            {
                const int4 o = ld_ofs(ofs + 16 * tid);
                raw[0] = gu_load<F>(a.w, o, 0);
                gu_prefetch<F>(a.w, o, 1);
            }
            x_prefetch(2);
            x_stage(x_load(0), srow_a);
            x_stage(x_load(1), srow_a + a_tile);
            __syncthreads(); // the tables and activation tiles 0 and 1
            // Step j reads tile j & 3 and stages tile (j + 2) & 3, last read at step j - 2: the barrier after every odd
            // step orders that read before this write, and this write before step j + 2's read.
            const auto step = [&](auto tile, int32_t j) {
                constexpr int32_t T = decltype(tile)::value; // j & 3
                constexpr int32_t P = T & 1;                 // the raw slot
                if (j + 3 < n_steps) {
                    x_prefetch(j + 3);
                }
                if (j + 1 < n_steps) {
                    const int4 o = ld_ofs(ofs + 16 * tid);
                    raw[P ^ 1] = gu_load<F>(a.w, o, j + 1);
                    if (j + 2 < n_steps) {
                        gu_prefetch<F>(a.w, o, j + 2);
                    }
                }
                if (j + 2 < n_steps) {
                    x_stage(x_load(j + 2), srow_a + ((T + 2) & 3) * a_tile); // rows hit L1 (prefetched): x is a few MB
                }
                const a_rows ar = {arow.even + T * a_tile, arow.odd + T * a_tile};
                gu_scale sc = gu_scale_of<F>(raw[P], 0);
#pragma unroll
                for (int32_t l = 0; l < 4; ++l) {
                    if (two_scales(F) && l == 2) {
                        sc = gu_scale_of<F>(raw[P], 1);
                    }
                    mma_chunk<NT>(acc, gu_chunk<F>(raw[P], sc, tb, lo, l), ar, l);
                }
                if constexpr ((T & 1) != 0) {
                    __syncthreads();
                }
            };
            static_assert(n_steps % gu_tiles == 0, "whole rings");
            for (int32_t j = 0; j < n_steps; j += gu_tiles) {
                step(std::integral_constant<int32_t, 0>{}, j);
                step(std::integral_constant<int32_t, 1>{}, j + 1);
                step(std::integral_constant<int32_t, 2>{}, j + 2);
                step(std::integral_constant<int32_t, 3>{}, j + 3);
            }

            // GLU: lanes of quadpairs 0, 1 hold gate, their partners (lane ^ 8) up at the same places; each lane finishes
            // half of the elements (gate lanes 0..3, up lanes 4..7) and stores pairs of adjacent channels
            const int2 fr = task_rows(a.route.tasks, static_cast<int32_t>(blockIdx.x));
            // the thread's place, re-read rather than held across the K loop (one register short at 96)
            const int32_t etid = tid_fresh(), elane = etid & 31, ew = etid >> 5, eqp = (elane >> 2) & 3;
            const bool eup = eqp >= 2;
#pragma unroll
            for (int32_t t = 0; t < NT; ++t) {
                {
                    float gv[4], uv[4];
#pragma unroll
                    for (int32_t k = 0; k < 4; ++k) {
                        const float send = eup ? acc[t][k] : acc[t][k + 4];
                        const float got = __shfl_xor_sync(0xffffffffu, send, 8);
                        gv[k] = eup ? got : acc[t][k];
                        uv[k] = eup ? acc[t][k + 4] : got;
                    }
#pragma unroll
                    for (int32_t k = 0; k < 4; k += 2) {
                        const int32_t i = eup ? k + 4 : k;
                        const int32_t row = 8 * t + c_row(elane, i);
                        if (row < fr.y) {
                            const int32_t chan = 16 * ew + 8 * (eqp & 1) + c_col(elane, i);
                            const float v0 = gv[k] / (1.0f + expf(-gv[k])) * uv[k];
                            const float v1 = gv[k + 1] / (1.0f + expf(-gv[k + 1])) * uv[k + 1];
                            const int64_t o = int64_t(fr.x + row) * n_ff_card + chan;
                            const uint32_t hv = __half_as_ushort(__float2half_rn(v0)) |
                                                (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(v1))) << 16);
                            *reinterpret_cast<uint32_t *>(a.glu + o) = hv;
                            if (a.glu_f32 != nullptr) {
                                *reinterpret_cast<float2 *>(a.glu_f32 + o) = make_float2(v0, v1);
                            }
                        }
                    }
                }
            }
        });
    }
}

// ---- down kernel -------------------------------------------------------------------------------------------------
// As gate/up with 512 threads: column c of each 512-column group g of the card's [320 -> 2560] product (output
// 512 g + c), its lane unpacking the column's weights into its own B fragments; the F16 GLU rows are the activations,
// permuted as the weights. PB3: one launch takes the five groups (EP1's later item 3). The task's GLU rows (K = 320:
// 10 steps) are staged once, behind one barrier, so the K loops have none; the weight stream runs on across the groups
// (step s = 10 g + j sits in slot j & 1 and is loaded during step s - 1), so each task pays its prologue once; group g's
// partials are stored when its 10 steps end. Each partial is the same products in the same order as one launch a group.
template <fmt F> __global__ void __launch_bounds__(dn_threads, 1) ep_down_kernel(ep_down_args a) {
    extern __shared__ uint4 smem_u4[];
    uint8_t * at = reinterpret_cast<uint8_t *>(smem_u4); // [n_steps] activation tiles, step j at + j a_tile
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t w = tid >> 5, lane = tid & 31;
    const int32_t qp = (lane >> 2) & 3, r = (lane & 3) | ((lane >> 2) & 4);
    const int32_t c = 32 * w + 8 * qp + r;
    const a_rows arow = a_row(at + r * 64, asw(r));
    const int32_t n16 = 16 * c; // group g's column: n16 + g g16
    constexpr int32_t g16 = 16 * ep_col_group;
    const int32_t srow = tid >> 2, sq = tid & 3;
    constexpr int32_t n_steps = n_ff_card / ks;
    static_assert(n_steps % 2 == 0, "steps a pair: step 10 g + j sits in slot j & 1");
    const int32_t task = static_cast<int32_t>(blockIdx.x);
    if (task >= n_tasks_of(a.route)) {
        return;
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (F == fmt::iq4_nl) {
        const uint4 t = __ldg(reinterpret_cast<const uint4 *>(a.grids + kvalues_off));
        kv[0] = t.x ^ 0x80808080u; // kvalues + 128 as unsigned bytes
        kv[1] = t.y ^ 0x80808080u;
        kv[2] = t.z ^ 0x80808080u;
        kv[3] = t.w ^ 0x80808080u;
    }
    {
        const task_t tk = task_of(a.route.tasks, task);
        const int32_t nt = (tk.n + 7) >> 3;
        const int32_t eo = tk.e * dn_bytes(F);
        // two raw slots with compile-time indices (R12); step 0's weights load beside the staging
        dn_raw raw[2];
        raw[0] = dn_load<F>(a.w, eo, n16, 0);
        // the GLU rows: IQ4_NL threads below 192 stage chunk t % 4 of row t / 4 (halves 4 c .. 4 c + 3 and 16 + 4 c ..
        // 16 + 4 c + 3 of the step); Q2_0 threads below 96 stage 16 halves (row t / 2, chunks 2 (t % 2) and
        // 2 (t % 2) + 1, permuted as the weights). Every step's tile, in two batches of loads then stores (the stores
        // are asm volatile: a load after one is not moved ahead of it)
        constexpr bool q2 = F == fmt::q2_0;
        const int32_t xrow = q2 ? tid >> 1 : srow, xq = q2 ? 2 * (tid & 1) : sq;
        const bool stager = tid < (q2 ? n_stage / 2 : n_stage);
        const bool sreal = stager && xrow < tk.n;
        const int32_t go = (tk.first + (sreal ? xrow : 0)) * n_ff_card + (q2 ? 8 * xq : 4 * xq);        // F16 elements
        const uint32_t xrow_a = tile_row(at + xrow * 64, asw(xrow)) ^ (static_cast<uint32_t>(xq) << 4); // chunk xq
        const auto x_load = [&](int32_t j) -> glu_raw {
            const uint4 z4 = make_uint4(0, 0, 0, 0);
            if constexpr (q2) {
                return {sreal ? __ldg(reinterpret_cast<const uint4 *>(a.glu + (go + j * ks))) : z4,
                        sreal ? __ldg(reinterpret_cast<const uint4 *>(a.glu + (go + j * ks + 8))) : z4};
            } else {
                const uint2 z2 = make_uint2(0, 0);
                const uint2 u = sreal ? __ldg(reinterpret_cast<const uint2 *>(a.glu + (go + j * ks))) : z2;
                const uint2 v = sreal ? __ldg(reinterpret_cast<const uint2 *>(a.glu + (go + j * ks + 16))) : z2;
                return {make_uint4(u.x, u.y, v.x, v.y), z4};
            }
        };
        const auto x_stage = [&](const glu_raw & x, uint32_t abuf) {
            if constexpr (q2) {
                const uint32_t wk[8] = {x.a.x, x.a.y, x.a.z, x.a.w, x.b.x, x.b.y, x.b.z, x.b.w}; // (n0, n1), ...
#pragma unroll
                for (int32_t p = 0; p < 2; ++p) {
                    st_chunk(abuf, p, __byte_perm(wk[p], wk[p + 2], 0x5410), __byte_perm(wk[p + 4], wk[p + 6], 0x5410),
                             __byte_perm(wk[p], wk[p + 2], 0x7632), __byte_perm(wk[p + 4], wk[p + 6], 0x7632));
                }
            } else {
                st_chunk(abuf, 0, x.a.x, x.a.y, x.a.z, x.a.w);
            }
        };
        if (stager) {
            constexpr int32_t batch = n_steps / 2;
#pragma unroll
            for (int32_t j0 = 0; j0 < n_steps; j0 += batch) {
                glu_raw x[batch];
#pragma unroll
                for (int32_t k = 0; k < batch; ++k) {
                    x[k] = x_load(j0 + k);
                }
#pragma unroll
                for (int32_t k = 0; k < batch; ++k) {
                    x_stage(x[k], xrow_a + (j0 + k) * a_tile);
                }
            }
        }
        __syncthreads(); // every step's tile; read-only from here
        with_nt<max_rt>(nt, [&](auto ntc) {
            constexpr int32_t NT = decltype(ntc)::value;
            float acc[NT][8];
            // step j of group g: the next step's weights (j + 1, or the next group's step 0) into the other slot
            const auto step = [&](auto par, int32_t g, int32_t j) {
                constexpr int32_t P = decltype(par)::value;
                if (j + 1 < n_steps) {
                    raw[P ^ 1] = dn_load<F>(a.w, eo, n16 + g * g16, j + 1);
                } else if (g + 1 < ep_n_col_groups) {
                    raw[P ^ 1] = dn_load<F>(a.w, eo, n16 + (g + 1) * g16, 0);
                }
                const a_rows ar = {arow.even + j * a_tile, arow.odd + j * a_tile};
#pragma unroll
                for (int32_t cq = 0; cq < 4; ++cq) {
                    mma_chunk<NT>(acc, dn_chunk<F>(raw[P], kv, cq), ar, cq);
                }
            };
            for (int32_t g = 0; g < ep_n_col_groups; ++g) {
#pragma unroll
                for (int32_t t = 0; t < NT; ++t) {
#pragma unroll
                    for (int32_t i = 0; i < 8; ++i) {
                        acc[t][i] = 0.0f;
                    }
                }
#pragma unroll
                for (int32_t j = 0; j < n_steps; j += 2) {
                    step(std::integral_constant<int32_t, 0>{}, g, j);
                    step(std::integral_constant<int32_t, 1>{}, g, j + 1);
                }
                // group g's partials, pairs of adjacent columns: F32, or FP16 rounded once (PB3: half the bytes the
                // sum reads back)
                const int2 fr = task_rows(a.route.tasks, static_cast<int32_t>(blockIdx.x));
                bool big = false;
#pragma unroll
                for (int32_t t = 0; t < NT; ++t) {
#pragma unroll
                    for (int32_t i = 0; i < 8; i += 2) {
                        const int32_t row = 8 * t + c_row(lane, i);
                        if (row < fr.y) {
                            const int32_t cc = ep_col_group * g + 32 * w + 8 * qp + c_col(lane, i);
                            const int64_t o = int64_t(fr.x + row) * D + cc;
                            if (a.part16 != nullptr) {
                                big = big || !(fabsf(acc[t][i]) <= 65504.0f) || !(fabsf(acc[t][i + 1]) <= 65504.0f);
                                *reinterpret_cast<uint32_t *>(a.part16 + o) = h2_pack(acc[t][i], acc[t][i + 1]);
                            } else {
                                *reinterpret_cast<float2 *>(a.part + o) = make_float2(acc[t][i], acc[t][i + 1]);
                            }
                        }
                    }
                }
                if (big) {
                    atomicOr(a.err, a.err_bit);
                }
            }
        });
    }
}

// ---- sum ---------------------------------------------------------------------------------------------------------
// Thread (b, c4): 4 columns of row b; the 10 slots' partials in route order, then the shared partial.
constexpr int32_t sum_threads = 256;

__global__ void __launch_bounds__(sum_threads) ep_sum_kernel(ep_sum_args a, int32_t rows) {
    const int32_t i = static_cast<int32_t>(blockIdx.x) * sum_threads + static_cast<int32_t>(threadIdx.x);
    constexpr int32_t n4 = D / 4;
    if (i >= rows * n4 || __ldcg(a.route.counters + 1) != 0) {
        return;
    }
    const int32_t b = i / n4, c4 = i - b * n4;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int32_t k = 0; k < n_used; ++k) {
        const int32_t p = __ldg(a.route.pos + b * n_used + k);
        const float wk = __ldg(a.wts + b * n_used + k);
        float4 v;
        if (a.part16 != nullptr) {
            const uint2 hv = __ldcs(reinterpret_cast<const uint2 *>(a.part16 + int64_t(p) * D) + c4);
            const float2 lo = __half22float2(*reinterpret_cast<const half2 *>(&hv.x));
            const float2 hi = __half22float2(*reinterpret_cast<const half2 *>(&hv.y));
            v = make_float4(lo.x, lo.y, hi.x, hi.y);
        } else {
            v = __ldcs(reinterpret_cast<const float4 *>(a.part + int64_t(p) * D) + c4);
        }
        acc.x = fmaf(wk, v.x, acc.x);
        acc.y = fmaf(wk, v.y, acc.y);
        acc.z = fmaf(wk, v.z, acc.z);
        acc.w = fmaf(wk, v.w, acc.w);
    }
    const int64_t o = int64_t(b) * D + 4 * c4;
    if (a.shared != nullptr) {
        const float4 s = __ldg(reinterpret_cast<const float4 *>(a.shared + o));
        acc.x += s.x;
        acc.y += s.y;
        acc.z += s.z;
        acc.w += s.w;
    }
    if (a.out16 != nullptr) {
        if (!(fabsf(acc.x) <= 65504.0f) || !(fabsf(acc.y) <= 65504.0f) || !(fabsf(acc.z) <= 65504.0f) ||
            !(fabsf(acc.w) <= 65504.0f)) {
            atomicOr(a.err, a.err_bit);
        }
        *reinterpret_cast<uint2 *>(a.out16 + o) = make_uint2(h2_pack(acc.x, acc.y), h2_pack(acc.z, acc.w));
        return;
    }
    *reinterpret_cast<float4 *>(a.out + o) = acc;
}

// ---- group -------------------------------------------------------------------------------------------------------
// One block of 1024 threads (32 warps). Warp w takes the pairs [w P / 32, (w + 1) P / 32) in order, 32 a trip:
// __match_any_sync finds equal experts, the lowest lane of each adds the count (no atomics). Each expert's total, a
// scan over experts, and each warp's start inside each expert follow; the warps replay their pairs and place them,
// so the order is stable by pair index. Then each expert's pairs are split into ceil(n / 48) tasks of near-equal size.
constexpr int32_t grp_threads = 1024;
constexpr int32_t grp_warps = grp_threads / 32;

struct grp_smem {
    uint16_t hist[grp_warps][n_expert]; // counts, then each warp's next position minus offsets (fits: P < 65536)
    int32_t off[n_expert + 1];
    int32_t toff[n_expert + 1];
    int32_t wsum[grp_warps];
    int32_t bad;
};

// exclusive scan of v over the block's first 512 threads (others pass 0); total in *tot
__device__ int32_t block_scan(int32_t v, int32_t * wsum, int32_t * tot) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x), lane = tid & 31, w = tid >> 5;
    int32_t x = v;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1) {
        const int32_t y = __shfl_up_sync(0xffffffffu, x, o);
        x += lane >= o ? y : 0;
    }
    if (lane == 31) {
        wsum[w] = x;
    }
    __syncthreads();
    int32_t before = 0, all = 0;
    for (int32_t k = 0; k < grp_warps; ++k) {
        const int32_t s = wsum[k];
        before += k < w ? s : 0;
        all += s;
    }
    __syncthreads();
    *tot = all;
    return before + x - v;
}

__global__ void __launch_bounds__(grp_threads, 1) ep_group_kernel(const int32_t * ids, ep_route rt, uint32_t * status,
                                                                  int32_t rows, int32_t task_rows) {
    __shared__ grp_smem s;
    const int32_t tid = static_cast<int32_t>(threadIdx.x), lane = tid & 31, w = tid >> 5;
    const uint32_t lt = (1u << lane) - 1u;
    const int32_t P = rows * n_used;
    const int32_t p0 = int32_t((int64_t(P) * w) / grp_warps), p1 = int32_t((int64_t(P) * (w + 1)) / grp_warps);
    for (int32_t i = tid; i < grp_warps * n_expert; i += grp_threads) {
        (&s.hist[0][0])[i] = 0;
    }
    if (tid == 0) {
        s.bad = 0;
    }
    __syncthreads();
    int32_t bad = 0;
    for (int32_t p = p0; p < p1; p += 32) {
        const bool on = p + lane < p1;
        int32_t e = on ? __ldg(ids + p + lane) : -1;
        if (on && (e < 0 || e >= n_expert)) {
            bad = 1;
            e = 0;
        }
        const uint32_t mm = __match_any_sync(0xffffffffu, e);
        if (on && (mm & lt) == 0) {
            s.hist[w][e] = static_cast<uint16_t>(s.hist[w][e] + __popc(mm)); // off lanes match only each other
        }
        __syncwarp();
    }
    if (__any_sync(0xffffffffu, bad != 0) && lane == 0) {
        s.bad = 1; // any writer stores 1
    }
    __syncthreads();
    if (s.bad != 0) {
        if (tid == 0) {
            rt.counters[0] = 0;
            rt.counters[1] = 1;
            if (status != nullptr) {
                *status = 1u;
            }
        }
        return;
    }
    // totals and offsets
    int32_t tot_e = 0;
    if (tid < n_expert) {
        for (int32_t k = 0; k < grp_warps; ++k) {
            tot_e += s.hist[k][tid];
        }
    }
    int32_t all = 0;
    const int32_t off_e = block_scan(tid < n_expert ? tot_e : 0, s.wsum, &all);
    const int32_t np = (tot_e + task_rows - 1) / task_rows;
    int32_t n_tasks = 0;
    const int32_t toff_e = block_scan(tid < n_expert ? np : 0, s.wsum, &n_tasks);
    if (tid < n_expert) {
        s.off[tid] = off_e;
        s.toff[tid] = toff_e;
        rt.offsets[tid] = off_e;
        int32_t base = 0;
        for (int32_t k = 0; k < grp_warps; ++k) {
            const int32_t h = s.hist[k][tid];
            s.hist[k][tid] = static_cast<uint16_t>(base);
            base += h;
        }
        // the tasks of expert tid
        for (int32_t k = 0; k < np; ++k) {
            const int32_t q = tot_e / np, rem = tot_e - q * np;
            const int32_t first = off_e + k * q + (k < rem ? k : rem);
            *reinterpret_cast<int4 *>(rt.tasks + 4 * (toff_e + k)) = make_int4(tid, first, q + (k < rem ? 1 : 0), 0);
        }
    }
    if (tid == 0) {
        rt.offsets[n_expert] = all;
        rt.counters[0] = n_tasks;
        rt.counters[1] = 0;
    }
    __syncthreads();
    // placement: replay in order
    for (int32_t p = p0; p < p1; p += 32) {
        const bool on = p + lane < p1;
        const int32_t e = on ? __ldg(ids + p + lane) : -1;
        const uint32_t mm = __match_any_sync(0xffffffffu, e);
        if (on) {
            const int32_t at_ = s.off[e] + s.hist[w][e] + __popc(mm & lt);
            rt.sorted[at_] = p + lane;
            rt.pos[p + lane] = at_;
        }
        __syncwarp();
        if (on && (mm & lt) == 0) {
            s.hist[w][e] = static_cast<uint16_t>(s.hist[w][e] + __popc(mm));
        }
        __syncwarp();
    }
}

// ---- launch ------------------------------------------------------------------------------------------------------
bool aligned16(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}
bool aligned4(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 3u) == 0;
}
bool route_ok(const ep_route & r) {
    return aligned4(r.offsets) && aligned4(r.sorted) && aligned4(r.pos) && aligned16(r.tasks) && aligned4(r.counters);
}

// PB1: cudaFuncSetAttribute sets the current device's attribute only, and the prompt path runs these kernels on both
// cards (a process-wide static set it on the first card only, so card 1's launches failed): one bit a device, as the
// decode experts' detail::smem_attr_once keeps (experts.cu)
bool smem_per_device(std::atomic<uint32_t> & done, const void * kernel, int32_t smem) {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        return false;
    }
    const uint32_t bit = 1u << dev;
    if ((done.load(std::memory_order_acquire) & bit) != 0u) {
        return true;
    }
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
        return false;
    }
    done.fetch_or(bit, std::memory_order_release);
    return true;
}

template <fmt F> bool launch_gate_up(const ep_gate_up_args & a, cudaStream_t st) {
    constexpr int32_t smem = gu_smem(F);
    static std::atomic<uint32_t> done{ 0u };
    if (!smem_per_device(done, reinterpret_cast<const void *>(ep_gate_up_kernel<F>), smem)) {
        return false;
    }
    ep_gate_up_kernel<F><<<ep_max_tasks, nt_threads, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <fmt F> bool launch_down(const ep_down_args & a, cudaStream_t st) {
    constexpr int32_t smem = (n_ff_card / ks) * a_tile; // every step's GLU tile
    static std::atomic<uint32_t> done{ 0u };  // PB1: per device, as above
    if (!smem_per_device(done, reinterpret_cast<const void *>(ep_down_kernel<F>), smem)) {
        return false;
    }
    ep_down_kernel<F><<<ep_max_tasks, dn_threads, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

size_t ep_workspace_bytes(int32_t rows) {
    if (rows < 1 || rows > ep_max_rows) {
        return 0;
    }
    const size_t r = rows, pairs = r * 10;
    const size_t bytes[] = {
        pairs * 4,    513 * 4,         pairs * 4,     pairs * 4, size_t(ep_max_tasks) * 16, 8 * 4, // ids, the route
        r * D * 2,    pairs * 320 * 2, pairs * D * 4, pairs * 4,                                   // x, glu, part, wts
        r * 2560 * 4, r * 2560 * 4,                                                                // out, shared
    };
    size_t total = 0;
    for (const size_t n : bytes) {
        total += (n + 255) / 256 * 256;
    }
    return total;
}

int32_t ep_task_rows(fmt gate_up) {
    return task_rows_of(gate_up);
}

bool ep_group(int32_t rows, int32_t task_rows, const int32_t * ids, const ep_route & r, uint32_t * status,
              void * stream) {
    if (rows < 1 || rows > ep_max_rows || task_rows < 8 || task_rows > ep_pass_rows || task_rows % 8 != 0 ||
        !aligned4(ids) || !route_ok(r) || stream == nullptr || (status != nullptr && !aligned4(status))) {
        return false;
    }
    ep_group_kernel<<<1, grp_threads, 0, static_cast<cudaStream_t>(stream)>>>(ids, r, status, rows, task_rows);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool ep_gate_up(fmt f, int32_t rows, const ep_gate_up_args & a, void * stream) {
    if (rows < 1 || rows > ep_max_rows || !aligned16(a.w) || !aligned16(a.grids) || !aligned16(a.x) ||
        !route_ok(a.route) || !aligned4(a.glu) || stream == nullptr ||
        (a.glu_f32 != nullptr && !aligned16(a.glu_f32))) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (f) {
        case fmt::iq2_xxs:
            return launch_gate_up<fmt::iq2_xxs>(a, st);
        case fmt::iq2_xs:
            return launch_gate_up<fmt::iq2_xs>(a, st);
        case fmt::iq2_s:
            return launch_gate_up<fmt::iq2_s>(a, st);
        case fmt::iq3_xxs:
            return launch_gate_up<fmt::iq3_xxs>(a, st);
        case fmt::iq3_s:
            return launch_gate_up<fmt::iq3_s>(a, st);
        default:
            return false;
    }
}

bool ep_down(fmt f, int32_t rows, const ep_down_args & a, void * stream) {
    if (rows < 1 || rows > ep_max_rows || !aligned16(a.w) || !aligned16(a.grids) || !aligned16(a.glu) ||
        !route_ok(a.route) || (a.part16 != nullptr ? !aligned16(a.part16) || a.err == nullptr : !aligned16(a.part)) ||
        stream == nullptr) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (f) {
        case fmt::q2_0:
            return launch_down<fmt::q2_0>(a, st);
        case fmt::iq4_nl:
            return launch_down<fmt::iq4_nl>(a, st);
        default:
            return false;
    }
}

bool ep_sum(int32_t rows, const ep_sum_args & a, void * stream) {
    if (rows < 1 || rows > ep_max_rows ||
        !aligned16(a.part16 != nullptr ? static_cast<const void *>(a.part16) : a.part) || !route_ok(a.route) ||
        !aligned4(a.wts) || (a.out16 != nullptr ? !aligned16(a.out16) || a.err == nullptr : !aligned16(a.out)) ||
        stream == nullptr || (a.shared != nullptr && !aligned16(a.shared))) {
        return false;
    }
    const int32_t n = rows * (D / 4);
    ep_sum_kernel<<<(n + sum_threads - 1) / sum_threads, sum_threads, 0, static_cast<cudaStream_t>(stream)>>>(a, rows);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::experts
