#pragma once

// EX2's gate, up and GLU launch: the routed experts (EX1b's recipe) and the shared expert (DP2's recipe, as 10 more
// units) in one kernel per deployed (routed, shared gate, shared up) triple (experts_inst.h). Internal to
// fn/kernels/experts: each experts_gu_<routed format>.cu defines detail::gate_up_<format> from gate_up_for.

#include "fn/kernels/dense/dense_dec.cuh"
#include "fn/kernels/experts/experts_dev.cuh"
#include "fn/kernels/experts/experts_inst.h"
#include "fn/xchg/xchg.cuh"

#include <atomic>
#include <type_traits>

namespace fn::experts {
namespace {

using dfmt = fn::dense::fmt;

// One row's sub-block, decoded: 8 grid words, their 8 sign masks and the sub-block scale(s) (two for the formats
// with a 4-bit scale per 16 weights). dw1 is unused when split is false.
struct row_dec {
    uint32_t g[8];
    uint32_t m[8];
    float dw0, dw1;
};

// ---- gate/up ----------------------------------------------------------------------------------------------------
constexpr int32_t x_meta = n_sub * 16; // bytes of a row's staged run meta {d, s, sum of codes 0..15, 16..31}
constexpr int32_t gu_depth = 2;        // 16-byte loads a thread in flight: 640 x 2 x 16 = 20 KB an SM
constexpr int32_t iq3s_off = 15360;    // the IQ3_S grid in the grid buffer (gu_layout_of(fmt::iq3_s).grid_off)
static_assert(gu_layout_of(fmt::iq3_s).grid_off == iq3s_off && gu_layout_of(fmt::iq3_s).grid_len == 2048, "IQ3_S grid");

template <int C> struct gu_smem_view {
    uint8_t * grid;   // the routed format's table
    uint32_t * sgrid; // the IQ3_S grid for a shared matrix (the routed table itself when the routed format is IQ3_S)
    uint8_t * xs;     // the rows' codes, staged once (R3) in DP2's swizzled chunks: [B][2560]
    float4 * xm;      // per run of 32 {d, s, sum of codes 0..15, sum of 16..31} (fn::dense::act): [B][80]
    float * red;      // [2][B][64][red_stride]
    meta_c<C> * m;    // SP2: the launch's capacity's (cap_of(B))
};

template <fmt F, dfmt FG, dfmt FU> __host__ __device__ constexpr bool extra_iq3s() {
    return (FG == dfmt::iq3_s || FU == dfmt::iq3_s) && F != fmt::iq3_s;
}
template <fmt F, dfmt FG, dfmt FU> __host__ __device__ constexpr int32_t gu_grid_len() {
    return gu_layout_of(F).grid_len + (extra_iq3s<F, FG, FU>() ? 2048 : 0);
}
// XB2 (d): a block's units in its rank order, routed then shared (U16 each, at most gu_max_ranks: 410 units over 72
// blocks; SP2, capacity 5: 510 units, 8 ranks), then the routed count and the total (U16)
template <int C> constexpr int32_t gu_max_ranks = (max_units_of(C) + n_sm - 1) / n_sm;
template <int C> constexpr int32_t gu_list_bytes = C <= 4 ? 16 : 32;
template <int C> constexpr bool gu_list_ok() {
    return 2 * (gu_max_ranks<C> + 2) <= gu_list_bytes<C> && gu_max_ranks<C> <= 32;
}
static_assert(gu_list_ok<4>() && gu_list_ok<5>(), "a block's units: one warp's lanes");
// XB3: the struct alone (the order is not held)
template <int C> constexpr int32_t gu_meta_bytes = static_cast<int32_t>(sizeof(meta_c<C>));
static_assert(gu_meta_bytes<4> % 16 == 0 && gu_meta_bytes<5> % 16 == 0, "16-byte aligned list region");
template <fmt F, dfmt FG, dfmt FU, int B> __host__ __device__ constexpr int32_t gu_smem_bytes() {
    return gu_grid_len<F, FG, FU>() + B * (x_codes + x_meta) + 2 * B * 64 * red_stride * 4 +
           gu_meta_bytes<cap_of(B)> + gu_list_bytes<cap_of(B)>; // the meta and the block's units (XB2, XB3)
}

// XB3 part 1: block `first`'s units in its rank order (XB2 (d)'s snake deal over build_order's order: its k-th rank
// is 72 k + first for even k and 72 k + 71 - first for odd k), by warp 0 from the block's built meta m: each expert's
// class (its active rows; the shared class apart), the classes' sizes, their starts by class_before (a prefix over the
// classes), then for each of the block's ranks the expert whose 10 units hold it (a ballot) or the shared unit. Writes
// list as XB2's last warp did (routed units, then shared, each in rank order; then the routed count and the total)
// and, when order is set, each of the block's ranks' entries into the region's order (unit_order's form), so the
// region holds what write_meta writes.
template <int C>
__device__ __forceinline__ void block_units(const meta_c<C> & m, const class_table & cw, int32_t first, uint16_t * list,
                                            uint16_t * order) {
    constexpr int32_t n_classes = C + 1;                 // SP2: the capacity's classes
    constexpr int32_t order_shared = order_shared_of(C); // and first shared unit
    constexpr int32_t gu_ranks = gu_max_ranks<C>;
    const uint32_t full = 0xffffffffu;
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    const uint32_t lt = (1u << lane) - 1u;
    const int32_t n_u = m.n_u;
    const int32_t c0 = lane < n_u ? __popc(m.mask[lane]) - 1 : -1;
    const int32_t c1 = lane + 32 < n_u ? __popc(m.mask[lane + 32]) - 1 : -1;
    int32_t size[n_classes]; // units of each class
    int32_t r0 = 0, r1 = 0;  // the experts' ranks in their classes
#pragma unroll
    for (int32_t c = 0; c < n_classes - 1; ++c) {
        const uint32_t b0 = __ballot_sync(full, c0 == c);
        const uint32_t b1 = __ballot_sync(full, c1 == c);
        size[c] = unit_groups * (__popc(b0) + __popc(b1));
        r0 = c0 == c ? __popc(b0 & lt) : r0;
        r1 = c1 == c ? __popc(b0) + __popc(b1 & lt) : r1;
    }
    size[n_classes - 1] = unit_groups;
    int32_t start[n_classes];
#pragma unroll
    for (int32_t a = 0; a < n_classes; ++a) {
        start[a] = 0;
#pragma unroll
        for (int32_t b = 0; b < n_classes; ++b) {
            start[a] += b != a && class_before<C>(cw, b, a) ? size[b] : 0;
        }
    }
    int32_t s0 = 0, s1 = 0; // class starts of the lane's experts
#pragma unroll
    for (int32_t c = 0; c < n_classes - 1; ++c) {
        s0 = c0 == c ? start[c] : s0;
        s1 = c1 == c ? start[c] : s1;
    }
    const int32_t q0 = s0 + unit_groups * r0, q1 = s1 + unit_groups * r1; // the rank of each expert's unit g = 0
    const int32_t sh = start[n_classes - 1];
    const int32_t n_ranks = unit_groups * (n_u + 1);
    uint32_t t = 0xffffu; // lane k: the unit at the block's rank k
#pragma unroll
    for (int32_t k = 0; k < gu_ranks; ++k) {
        const int32_t r = k * n_sm + ((k & 1) != 0 ? n_sm - 1 - first : first);
        const uint32_t h0 = __ballot_sync(full, c0 >= 0 && r >= q0 && r < q0 + unit_groups);
        const uint32_t h1 = __ballot_sync(full, c1 >= 0 && r >= q1 && r < q1 + unit_groups);
        const int32_t src = h0 != 0 ? __ffs(h0) - 1 : __ffs(h1) - 1;
        const int32_t q = __shfl_sync(full, h0 != 0 ? q0 : q1, src < 0 ? 0 : src);
        const uint32_t u = static_cast<uint32_t>(src + (h0 != 0 ? 0 : 32));
        const uint32_t tk = r >= n_ranks                      ? 0xffffu
                            : r >= sh && r < sh + unit_groups ? static_cast<uint32_t>(order_shared + r - sh)
                                                              : unit_groups * u + static_cast<uint32_t>(r - q);
        t = lane == k ? tk : t;
    }
    const uint32_t routed = __ballot_sync(full, t < order_shared);
    const uint32_t shared = __ballot_sync(full, t != 0xffffu && t >= order_shared);
    if (t < order_shared) {
        list[__popc(routed & lt)] = static_cast<uint16_t>(t);
    } else if (t != 0xffffu) {
        list[__popc(routed) + __popc(shared & lt)] = static_cast<uint16_t>(t);
    }
    if (lane == 0) {
        list[gu_ranks] = static_cast<uint16_t>(__popc(routed));
        list[gu_ranks + 1] = static_cast<uint16_t>(__popc(routed) + __popc(shared));
    }
    if (order != nullptr && t != 0xffffu) {
        order[lane * n_sm + ((lane & 1) != 0 ? n_sm - 1 - first : first)] = static_cast<uint16_t>(t);
    }
}

// Row partials: for each routed row b, (dw * da_b) * isum, summed over the 4 records of the warp that share rs, then
// stored for the cross-warp sum: red[buf][b][r][w], r = rs + 8 m.
template <bool SPLIT, int B, typename DA>
__device__ __forceinline__ void row_out(const row_dec & rd, const uint32_t (&act)[B][8], const DA & da, uint32_t mask,
                                        float * red_buf, int32_t r, int32_t w, int32_t lane) {
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        if (((mask >> b) & 1u) == 0) {
            continue;
        }
        dot2 lo = {0, 0}, hi = {0, 0};
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            dot_word(i < 4 || !SPLIT ? lo : hi, rd.g[i], rd.m[i], act[b][i]);
        }
        float part;
        if (SPLIT) {
            part =
                (rd.dw0 * static_cast<float>(lo.p - 2 * lo.n) + rd.dw1 * static_cast<float>(hi.p - 2 * hi.n)) * da[b];
        } else {
            part = (rd.dw0 * da[b]) * static_cast<float>(lo.p - 2 * lo.n);
        }
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        part += __shfl_xor_sync(0xffffffffu, part, 16);
        if (lane < 8) {
            red_buf[(b * 64 + r) * red_stride + w] = part;
        }
    }
}

// XB3 K1 (K1's row_out1, k1 99460ed): row_out for a unit with one routed row b0, without the row test: act and da
// are row b0's, read for the unit, and red_row = red_buf + b0 * 64 * red_stride, so the partial lands where row_out
// puts row b0's (the same dots in the same order: bitwise row_out's)
template <bool SPLIT>
__device__ __forceinline__ void row_out1(const row_dec & rd, const uint32_t (&act)[8], float da, float * red_row,
                                         int32_t r, int32_t w, int32_t lane) {
    dot2 lo = {0, 0}, hi = {0, 0};
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
        dot_word(i < 4 || !SPLIT ? lo : hi, rd.g[i], rd.m[i], act[i]);
    }
    float part;
    if (SPLIT) {
        part = (rd.dw0 * static_cast<float>(lo.p - 2 * lo.n) + rd.dw1 * static_cast<float>(hi.p - 2 * hi.n)) * da;
    } else {
        part = (rd.dw0 * da) * static_cast<float>(lo.p - 2 * lo.n);
    }
    part += __shfl_xor_sync(0xffffffffu, part, 8);
    part += __shfl_xor_sync(0xffffffffu, part, 16);
    // a predicated store, so no branch (and no BSSY) around it ahead of the other rows' shuffles (K1b)
    const uint32_t at = static_cast<uint32_t>(__cvta_generic_to_shared(red_row + r * red_stride + w));
    asm volatile("{\n\t.reg .pred p;\n\tsetp.lt.s32 p, %0, 8;\n\t@p st.shared.f32 [%1], %2;\n\t}" ::"r"(lane), "r"(at),
                 "f"(part));
}

// XB3 K1b at B = 4: row_out's da read from shared memory at each use (rows b of the staged run meta from p on), not
// held in registers
struct smem_da {
    const float4 * p;
    __device__ __forceinline__ float operator[](int32_t b) const { return p[b * n_sub].x; }
};

// SP2, B = 5: row_out for the routed rows b0 + 1 + k (bit k of rest, k < 4) with each row's activation fragment of
// sub-block j and its d read from shared memory at the record (xs, xm: the staged rows), so no row's fragment is held
// in registers across the unit; each row's dots, partial and stores are row_out's (bitwise the same partials)
template <bool SPLIT>
__device__ __forceinline__ void row_out_s(const row_dec & rd, const uint8_t * xs, const float4 * xm, int32_t j,
                                          int32_t b1, uint32_t rest, float * red_buf, int32_t r, int32_t w,
                                          int32_t lane) {
#pragma unroll
    for (int32_t k = 0; k < 4; ++k) {
        if (((rest >> k) & 1u) == 0) {
            continue;
        }
        const int32_t b = b1 + k;
        const uint4 * xb = reinterpret_cast<const uint4 *>(xs + b * x_codes);
        const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
        const uint32_t act[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        const float da = xm[b * n_sub + j].x;
        dot2 lo = {0, 0}, hi = {0, 0};
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            dot_word(i < 4 || !SPLIT ? lo : hi, rd.g[i], rd.m[i], act[i]);
        }
        float part;
        if (SPLIT) {
            part = (rd.dw0 * static_cast<float>(lo.p - 2 * lo.n) + rd.dw1 * static_cast<float>(hi.p - 2 * hi.n)) * da;
        } else {
            part = (rd.dw0 * da) * static_cast<float>(lo.p - 2 * lo.n);
        }
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        part += __shfl_xor_sync(0xffffffffu, part, 16);
        if (lane < 8) {
            red_buf[(b * 64 + r) * red_stride + w] = part;
        }
    }
}

// SP4 H2 (map 2), the IQ2 body at B = 4 and 5: row_out for the routed rows after b0 (rest: their bits, row b at bit
// b), in ascending order through one non-unrolled loop, each row's activation fragment of sub-block j and its d read
// from shared memory (xs, xm: the staged rows) at its turn, so the unit holds no fragment but b0's. Each row's dots,
// partial and slot are row_out's (bitwise the same partials); the store is row_out1's predicated one. The loop's
// control is the unit's mask, block-uniform, so the full-mask shuffles see every lane.
template <bool SPLIT>
__device__ __forceinline__ void row_out_c(const row_dec & rd, const uint8_t * xs, const float4 * xm, int32_t j,
                                          uint32_t rest, float * red_buf, int32_t r, int32_t w, int32_t lane) {
#pragma unroll 1
    for (; rest != 0u; rest &= rest - 1u) {
        const int32_t b = __ffs(static_cast<int32_t>(rest)) - 1;
        const uint4 * xb = reinterpret_cast<const uint4 *>(xs + b * x_codes);
        const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
        const uint32_t act[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        const float da = xm[b * n_sub + j].x;
        dot2 lo = {0, 0}, hi = {0, 0};
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            dot_word(i < 4 || !SPLIT ? lo : hi, rd.g[i], rd.m[i], act[i]);
        }
        float part;
        if (SPLIT) {
            part = (rd.dw0 * static_cast<float>(lo.p - 2 * lo.n) + rd.dw1 * static_cast<float>(hi.p - 2 * hi.n)) * da;
        } else {
            part = (rd.dw0 * da) * static_cast<float>(lo.p - 2 * lo.n);
        }
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        part += __shfl_xor_sync(0xffffffffu, part, 16);
        const uint32_t at = static_cast<uint32_t>(__cvta_generic_to_shared(red_buf + (b * 64 + r) * red_stride + w));
        asm volatile("{\n\t.reg .pred p;\n\tsetp.lt.s32 p, %0, 8;\n\t@p st.shared.f32 [%1], %2;\n\t}" ::"r"(lane),
                     "r"(at), "f"(part));
    }
}

// Q8_0 (DR1): row r's sub-block j is 32 int8 weights against row b's 32 codes, an exact integer sum is[b] (IDP4A on
// signed bytes, PTX ISA dp4a), then (dh * da_b) * is[b] in F32 as the other formats' row_out; the warp reduction is
// row_out's.
template <int B>
__device__ __forceinline__ void row_out_q8(const int32_t (&is)[B], float dh, const float (&da)[B], uint32_t mask,
                                           float * red_buf, int32_t r, int32_t w, int32_t lane) {
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        if (((mask >> b) & 1u) == 0) {
            continue;
        }
        float part = (dh * da[b]) * static_cast<float>(is[b]);
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        part += __shfl_xor_sync(0xffffffffu, part, 16);
        if (lane < 8) {
            red_buf[(b * 64 + r) * red_stride + w] = part;
        }
    }
}

// Decoders, one row's sub-block each. dh: the row's FP16 d as F32.
__device__ __forceinline__ void dec_iq3_xxs(row_dec & rd, uint2 q, uint32_t sg, float dh, const uint32_t * grid) {
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
        rd.g[i] = grid[__byte_perm(i < 4 ? q.x : q.y, 0, 0x4440 + (i & 3))];
    }
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        sign_masks7((sg >> (7 * l)) & 0x7fu, rd.m[2 * l], rd.m[2 * l + 1]);
    }
    rd.dw0 = dh * (0.5f + static_cast<float>(sg >> 28)) * 0.5f;
}

__device__ __forceinline__ void dec_iq3_s(row_dec & rd, uint2 q, uint32_t sg, uint32_t qh, uint32_t sc, float dh,
                                          const uint32_t * grid) {
#pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
        const uint32_t lo = __byte_perm(i < 4 ? q.x : q.y, 0, 0x4440 + (i & 3));
        rd.g[i] = grid[lo | ((qh << (8 - i)) & 0x100u)];
    }
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        const uint32_t s = (sg >> (8 * l)) & 0xffu;
        rd.m[2 * l] = sign_lo(s);
        rd.m[2 * l + 1] = sign_hi(s);
    }
    rd.dw0 = dh * static_cast<float>(1 + 2 * static_cast<int32_t>(sc));
}

__device__ __forceinline__ void dec_iq2_xxs(row_dec & rd, uint2 q, float dh, const uint2 * grid) {
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        const uint2 g = grid[__byte_perm(q.x, 0, 0x4440 + l)];
        rd.g[2 * l] = g.x;
        rd.g[2 * l + 1] = g.y;
        sign_masks7((q.y >> (7 * l)) & 0x7fu, rd.m[2 * l], rd.m[2 * l + 1]);
    }
    rd.dw0 = dh * (0.5f + static_cast<float>(q.y >> 28)) * 0.25f;
}

__device__ __forceinline__ void dec_iq2_xs(row_dec & rd, uint2 q, uint32_t sc, float dh, const uint2 * grid) {
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        const uint32_t v = ((l < 2 ? q.x : q.y) >> (16 * (l & 1))) & 0xffffu;
        const uint2 g = grid[v & 511u];
        rd.g[2 * l] = g.x;
        rd.g[2 * l + 1] = g.y;
        sign_masks7(v >> 9, rd.m[2 * l], rd.m[2 * l + 1]);
    }
    rd.dw0 = dh * (0.5f + static_cast<float>(sc & 0xfu)) * 0.25f;
    rd.dw1 = dh * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
}

__device__ __forceinline__ void dec_iq2_s(row_dec & rd, uint32_t qs, uint32_t sg, uint32_t qh, uint32_t sc, float dh,
                                          const uint2 * grid) {
#pragma unroll
    for (int32_t l = 0; l < 4; ++l) {
        const uint32_t idx = __byte_perm(qs, 0, 0x4440 + l) | ((qh << (8 - 2 * l)) & 0x300u);
        const uint2 g = grid[idx];
        rd.g[2 * l] = g.x;
        rd.g[2 * l + 1] = g.y;
        const uint32_t s = (sg >> (8 * l)) & 0xffu;
        rd.m[2 * l] = sign_lo(s);
        rd.m[2 * l + 1] = sign_hi(s);
    }
    rd.dw0 = dh * (0.5f + static_cast<float>(sc & 0xfu)) * 0.25f;
    rd.dw1 = dh * (0.5f + static_cast<float>(sc >> 4)) * 0.25f;
}

// XB2 (c): a taken chunk's byte pair (x, y), one byte a row (row m < 4: byte m of x; else byte m - 4 of y), taken by
// two byte permutes into lo = [x0 y0 x1 y1], hi = [x2 y2 x3 y3]: real consumers where the chunk is taken. A plain
// copy of the pair (what holding the words compiled to) was placed by the compiler right after the chunk's load, so
// the warp waited for the load there (ring_copies.py), long before the data was used.
struct bytes8 {
    uint32_t lo, hi;
};
__device__ __forceinline__ bytes8 take8(uint32_t x, uint32_t y) {
    return {__byte_perm(x, y, 0x5140), __byte_perm(x, y, 0x7362)};
}
// row m's byte (m < 8, a constant in the unrolled bodies)
__device__ __forceinline__ uint32_t byte8(const bytes8 & b, int32_t m) {
    const int32_t k = m & 3;
    return ((k < 2 ? b.lo : b.hi) >> (8 * (2 * (k & 1) + (m >> 2)))) & 0xffu;
}

// One finished unit's row: warp row b sums the unit's NPART partials per channel in order (routed: the 20 warps;
// shared: the 10 steps' 4 quarter lanes), STRIDE floats a channel, applies the GLU and writes FNFL's activation format
// to slot (act.h's rule: lane i holds channel 32 g + i of the run).
template <int NPART, int STRIDE>
__device__ __forceinline__ void glu_out(const gate_up_args & a, const float * rb, int32_t slot, int32_t g,
                                        int32_t lane) {
    float gs = 0.0f, us = 0.0f;
#pragma unroll 4
    for (int32_t ww = 0; ww < NPART; ++ww) {
        gs += rb[lane * STRIDE + ww];
        us += rb[(32 + lane) * STRIDE + ww];
    }
    const float v = gs / (1.0f + expf(-gs)) * us;
    float amax = fabsf(v);
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    const float d = __fdiv_rn(amax, 127.0f);
    const float id = amax > 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
    const int32_t qv = min(max(__float2int_rn(v * id), -127), 127);
    int32_t sum = qv;
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    a.glu.codes[slot * n_ff_card + g * 32 + lane] = static_cast<int8_t>(qv);
    if (lane == 0) {
        reinterpret_cast<float2 *>(a.glu.scales)[slot * n_group + g] = make_float2(d, d * static_cast<float>(sum));
    }
    if (a.glu_f32 != nullptr) {
        a.glu_f32[slot * n_ff_card + g * 32 + lane] = v;
    }
}

// ---- the shared expert's units (DP2's recipe)
// ------------------------------------------------------------------------ Its gate and up on this card are 320 rows of
// 2560 each in DP2's packed tiles (8 rows; tile t's step s of 256 weights at (10 t + s) step_bytes). Shared unit g
// (channels 32 g .. 32 g + 31, the routed units' channel group) is gate tiles 4 g .. 4 g + 3 and up tiles alike: 80
// (tile, step) items. Warp w takes step s = w % 10 of the gate tiles 4 g + h and 4 g + h + 2 (h = w / 10) and of the up
// tiles alike; lane (r, q) decodes its 64 weights of a step (fn::dense::decode), dots the B rows' staged codes
// (fn::dense::dots) and stores its partial of row b at red[b][unit row][4 s + q] (rows 0..31 gate, 32..63 up, as a
// routed unit's channels; the whole partials region, sh_stride floats a row): no shuffles, whose divergent-path calls
// spilled at B = 4. The GLU warp sums the 40 in (step, quarter) order.
constexpr int32_t sh_parts = 4 * 10;        // a shared unit's partials a (row, channel): 10 steps x 4 quarter lanes
constexpr int32_t sh_stride = sh_parts + 1; // floats a row: odd, so the GLU warp's column reads are conflict-free
static_assert(sh_stride <= 2 * red_stride, "the shared unit's partials fit the double-buffered region");

template <dfmt FD> __device__ __forceinline__ const uint8_t * sh_step(const uint8_t * w, int32_t tile, int32_t s) {
    return w + (int64_t(tile) * (D / fn::dense::step_k) + s) * fn::dense::step_bytes(FD, 4);
}

// v, hidden from the optimizer (DP2's opaque): an item's addresses are computed at its load, not hoisted to the unit's
// start and held (four IQ3_S cursors of five pointers spilled at B = 4)
__device__ __forceinline__ int32_t opaque(int32_t v) {
    asm volatile("" : "+r"(v));
    return v;
}

template <dfmt FD>
__device__ __forceinline__ void sh_load(fn::dense::raw<FD> & v, const uint8_t * w, int32_t tile, int32_t s, int32_t r,
                                        int32_t q) {
    fn::dense::load_full<FD, 0>(v, fn::dense::make_cursor<FD>(sh_step<FD>(w, opaque(tile), s), r, q));
}

template <dfmt FD, int B>
__device__ __forceinline__ void sh_item(const fn::dense::raw<FD> & v, int32_t s, int32_t q, int32_t row,
                                        const uint8_t * grids, const uint32_t * grid, const fn::dense::act & x,
                                        float * red) {
    // the IQ4 table through L1 at each IQ4_XS item (an L1 hit after the first), as DP2 reads it a unit: held across
    // the other format's items it spilled at B = 4
    const uint4 kv =
        FD == dfmt::iq4_xs ? __ldg(reinterpret_cast<const uint4 *>(grids + kvalues_off)) : make_uint4(0, 0, 0, 0);
    fn::dense::dec64 o;
    fn::dense::decode<FD>(v, q, kv, grid, o);
    float acc[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        acc[b] = 0.0f;
    }
    fn::dense::dots<FD, B>(o, s, q, x, acc);
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        red[(b * 64 + row) * sh_stride + 4 * s + q] = acc[b];
    }
}

template <dfmt FG, dfmt FU, int B>
__device__ __forceinline__ void shared_unit(const gate_up_args & a, int32_t g, int32_t w, int32_t lane,
                                            const fn::dense::act & x, const uint32_t * grid, float * red) {
    const int32_t s = w % 10, h = w / 10;
    const int32_t r = lane >> 2, q = lane & 3;
    const int32_t t0 = 4 * g + h, t1 = t0 + 2;
    // both gate items' loads first, then each up item's as a gate item's registers free (one item ahead)
    fn::dense::raw<FG> g0, g1;
    sh_load<FG>(g0, a.sh_gate, t0, s, r, q);
    sh_load<FG>(g1, a.sh_gate, t1, s, r, q);
    fn::dense::raw<FU> u0, u1;
    sh_item<FG, B>(g0, s, q, 8 * h + r, a.grids, grid, x, red);
    sh_load<FU>(u0, a.sh_up, t0, s, r, q);
    sh_item<FG, B>(g1, s, q, 8 * (h + 2) + r, a.grids, grid, x, red);
    sh_load<FU>(u1, a.sh_up, t1, s, r, q);
    sh_item<FU, B>(u0, s, q, 32 + 8 * h + r, a.grids, grid, x, red);
    sh_item<FU, B>(u1, s, q, 32 + 8 * (h + 2) + r, a.grids, grid, x, red);
}

// A routed format's compile-time constants (gate_up_kernel's sections take them through with_f, XB3 part 6)
template <fmt F> struct gu_fc {
    static constexpr fmt f = F;
    static constexpr gu_layout L = gu_layout_of(F);
    static constexpr int32_t NA = (L.aux8 || L.sc8) ? 1 : 0; // the aux chunk: aux8 in x, y and sc8 in z, w
    static constexpr int32_t CC = 1 + NA + L.chunks;         // the d chunk, the aux chunk, then the main chunks
    static constexpr int64_t T = tile_bytes(F);
};

// XB3 part 6, the IQ2 body: one gate_up symbol a B for the three IQ2 routed formats and every shared pair they meet,
// the formats run-time data (gate_up_args' u_f, u_fg, u_fu). Its shared memory holds the largest IQ2 grid (IQ2_S's) and
// always the IQ3_S grid of a shared matrix after it.
constexpr int32_t gu_u_table = 8192;
static_assert(gu_layout_of(fmt::iq2_xxs).grid_len <= gu_u_table && gu_layout_of(fmt::iq2_xs).grid_len <= gu_u_table &&
                  gu_layout_of(fmt::iq2_s).grid_len <= gu_u_table,
              "the IQ2 grids fit the body's grid region");
// XB3 C1, the IQ3 body (gate_up_kernel_u<gu_iq3>, B = 4 and 5): the same body for the two IQ3 routed formats; its
// routed table region holds the larger IQ3 grid (IQ3_S's), then the IQ3_S grid of a shared matrix as the IQ2 body's
constexpr int32_t gu_u3_table = 2048;
static_assert(gu_layout_of(fmt::iq3_xxs).grid_len <= gu_u3_table && gu_layout_of(fmt::iq3_s).grid_len <= gu_u3_table,
              "the IQ3 grids fit the body's grid region");
// a body's format family: its routed table region (the switches over its formats and pairs are in gate_up_kernel_u)
struct gu_iq2 {
    static constexpr bool iq3 = false;
    static constexpr int32_t table = gu_u_table;
};
struct gu_iq3 {
    static constexpr bool iq3 = true;
    static constexpr int32_t table = gu_u3_table;
};
template <int B, typename FAM = gu_iq2> __host__ __device__ constexpr int32_t gu_u_smem_bytes() {
    return FAM::table + 2048 + B * (x_codes + x_meta) + 2 * B * 64 * red_stride * 4 + gu_meta_bytes<cap_of(B)> +
           gu_list_bytes<cap_of(B)>;
}
template <bool U, fmt F, dfmt FG, dfmt FU> __host__ __device__ constexpr int32_t gu_grid_region() {
    return U ? gu_u_table + 2048 : gu_grid_len<F, FG, FU>();
}

// Gate, up and GLU. One block an SM (72), 640 threads. Routed: thread rec = 8 j + rs owns sub-block j of rows rs + 8 m
// of every routed unit (u, g) the block takes (unit 10 u + g; which ones: Units below), with row b's activation codes
// of sub-block j in registers. Each thread streams its records' chunks (the d chunk, the aux chunk of IQ2_XS and
// IQ3_S, then the C main chunks, unit after unit) through a ring of DEPTH 16-byte slots, each refilled as soon as its
// data is taken. After a unit, one barrier; warp b then sums row b's 20 warp partials, applies the GLU and writes
// FNFL's activation format (double-buffered partials, so the next unit runs meanwhile). Shared: shared unit g (g < 10)
// runs after the block's routed units, as shared_unit says, in the whole partials region with a barrier before and
// after it; its GLU rows go to slots n_slots + b.
// Routing (XB3 part 1): warp 0 of every block builds the meta of a.ids (71b92a9's build_meta) and the block's units
// (block_units); block 0 writes the meta's struct to a.meta for down, each block its ranks' entries of the order.
// Units (XB2 (d)): the order sorts the 10 U + 10 units by cost class (experts.h, a.cw); block i runs ranks 72 k + i
// (k even) and 72 k + 71 - i (k odd), its routed units first, then each shared unit among them in rank order (one, or
// two when the shared class straddles a snake turn).
// Budget: __launch_bounds__(640, 1) caps registers at 96; shared memory gu_smem_bytes (dynamic).
template <fmt F, dfmt FG, dfmt FU, int B, int DEPTH>
__global__ void __launch_bounds__(gu_threads, 1) gate_up_kernel(gate_up_args a) {
    constexpr int32_t C = cap_of(B); // SP2: the routing capacity (experts.h)
    constexpr gu_layout L = gu_layout_of(F);
    constexpr int32_t NA = (L.aux8 || L.sc8) ? 1 : 0; // the aux chunk: aux8 in x, y and sc8 in z, w
    constexpr int32_t CC = 1 + NA + L.chunks;         // the d chunk, the aux chunk, then the main chunks
    constexpr int64_t T = tile_bytes(F);
    extern __shared__ uint4 smem_u4[];
    uint8_t * smem = reinterpret_cast<uint8_t *>(smem_u4);
    constexpr int32_t xs_off = gu_grid_len<F, FG, FU>();
    constexpr int32_t xm_off = xs_off + B * x_codes;
    constexpr int32_t red_off = xm_off + B * x_meta;
    constexpr int32_t meta_off = red_off + 2 * B * 64 * red_stride * 4;
    static_assert(xs_off % 16 == 0 && xm_off % 16 == 0 && red_off % 16 == 0 && meta_off % 16 == 0,
                  "16-byte aligned shared regions");
    const gu_smem_view<C> sv = {smem,
                             reinterpret_cast<uint32_t *>(smem + (F == fmt::iq3_s ? 0 : L.grid_len)),
                             smem + xs_off,
                             reinterpret_cast<float4 *>(smem + xm_off),
                             reinterpret_cast<float *>(smem + red_off),
                             reinterpret_cast<meta_c<C> *>(smem + meta_off)};
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t lane = tid & 31;
    const int32_t w = tid >> 5;
    const int32_t j = tid >> 3;
    const int32_t rs = tid & 7;
    const uint32_t mz = gridDim.x == n_sm ? 0u : ~0u;

    // The grid tables and the rows' activations first, neither depends on the routing: coalesced 16-byte loads through
    // L1 (R3: written before this kernel and read by every SM; EX1's 4-byte L1-bypassing loads cost 7.7x the ideal L2
    // sectors). Thread i < 80 B takes run (b, j) = (i / 80, i % 80) of the rows. They land in shared memory after the
    // first weight chunks are issued.
    constexpr int32_t n_grid16 = L.grid_len / 16;
    constexpr int32_t n_sgrid16 = extra_iq3s<F, FG, FU>() ? 2048 / 16 : 0;
    static_assert(n_grid16 + n_sgrid16 <= gu_threads, "one 16-byte load of the tables a thread");
    uint4 grid_v = {0, 0, 0, 0};
    if (tid < n_grid16) {
        grid_v = __ldg(reinterpret_cast<const uint4 *>(a.grids + L.grid_off) + tid);
    } else if (tid < n_grid16 + n_sgrid16) {
        grid_v = __ldg(reinterpret_cast<const uint4 *>(a.grids + iq3s_off) + (tid - n_grid16));
    }
    const bool stager = tid < B * n_sub;
    // HC3: with X-HMIX, a stager whose run is in the peer's half polls it after the weight ring is primed (below)
    const bool peer_run = stager && a.hmix != nullptr && (tid % n_sub) / (n_sub / 2) == a.hmix_peer_half;
    uint4 xc0 = {0, 0, 0, 0}, xc1 = {0, 0, 0, 0};
    float2 xsc = {0.0f, 0.0f};
    if (stager && !peer_run) {
        const uint4 * src = reinterpret_cast<const uint4 *>(a.x.codes) + 2 * tid; // codes [B][2560]: run tid
        xc0 = __ldg(src);
        xc1 = __ldg(src + 1);
        xsc = __ldg(reinterpret_cast<const float2 *>(a.x.scales) + tid);
    }
    // XB3 part 1: warp 0 builds the routing meta (71b92a9's build_meta) and the block's units (block_units) while the
    // other warps' loads are in flight; block 0 writes the struct for down (the kernel boundary orders it before down's
    // read). The block's first unit comes from the order, so the ring's first loads follow the barrier.
    uint16_t * list = reinterpret_cast<uint16_t *>(smem + meta_off + gu_meta_bytes<C>);
    if (w == 0) {
        build_meta_of<B>([&](int32_t i) { return __ldg(a.ids + i); }, *sv.m);
        __syncwarp();
        if (sv.m->bad == 0) {
            block_units<C>(*sv.m, a.cw, static_cast<int32_t>(blockIdx.x), list,
                        reinterpret_cast<uint16_t *>(a.meta + meta_order_of(C)));
        }
        if (blockIdx.x == 0) {
            const uint4 * src = reinterpret_cast<const uint4 *>(sv.m);
#pragma unroll 1
            for (int32_t i = lane; i < meta_words_c<C>; i += 32) {
                reinterpret_cast<uint4 *>(a.meta)[i] = src[i];
            }
        }
    }
    __syncthreads();
    const meta_c<C> & m = *sv.m;
    if (m.bad != 0) {
        if (tid == 0 && blockIdx.x == 0 && a.status != nullptr) {
            *a.status = 1u;
        }
        return;
    }
    if (list[gu_max_ranks<C> + 1] == 0) { // no rank of the launch's 10 U + 10 is this block's
        return;
    }
    // the block's first routed unit (its tile primes the ring), or -1
    const int32_t t_first = list[gu_max_ranks<C>] > 0 ? int32_t(list[0]) : -1;
    const int32_t sb = j >> 3;
    // byte offset of chunk q (0 = d, then aux when the format has one, then main) of this thread's record in a tile;
    // Q8_0's d chunk is the record's own (a d a sub-block), the others' their superblock's
    const auto chunk_off = [&](int32_t q) -> int64_t {
        return q == 0 ? d_off(F) + (F == fmt::q8_0 ? tid : sb * 8 + rs) * 16
                      : (int64_t(q - 1 - NA) * gu_threads + tid) * 16;
    };
    // a chunk of the sequence: the aux chunk is one or two 8-byte loads, every other chunk one 16-byte load
    const auto load_chunk = [&](const uint8_t * tl, int32_t q, uint32_t z) -> uint4 {
        if (NA == 1 && q == 1) {
            uint4 v = {0, 0, 0, 0};
            if (L.aux8) {
                const uint2 x = __ldcs(reinterpret_cast<const uint2 *>(tl + aux8_off(F) + tid * 8 + z));
                v.x = x.x;
                v.y = x.y;
            }
            if (L.sc8) {
                const uint2 x = __ldcs(reinterpret_cast<const uint2 *>(tl + sc8_off(F) + ((j >> 1) * 8 + rs) * 8 + z));
                v.z = x.x;
                v.w = x.y;
            }
            return v;
        }
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(q) + z));
    };
    const auto tile_of = [&](int32_t t) -> const uint8_t * {
        const int32_t u = t / n_group;
        return a.w + (int64_t(m.e[u]) * n_group + (t - u * n_group)) * T;
    };

    // prime the ring: the first DEPTH chunks of the sequence (units are at least CC >= DEPTH chunks long)
    static_assert(CC >= DEPTH, "a unit has at least DEPTH chunks");
    ring<DEPTH> rg;
    const uint8_t * tile = t_first >= 0 ? tile_of(t_first) : a.w;
    if (t_first >= 0) {
#pragma unroll
        for (int32_t q = 0; q < DEPTH; ++q) {
            rg.s[q] = load_chunk(tile, q, 0u);
        }
    }
    // X-HMIX's acquire (xchg.h): the peer half's runs, now that this block's first weight chunks are in flight
    if (peer_run) {
        fn::xchg::hmix_run(a.hmix, a.hmix_peer_half, tid / n_sub, tid % n_sub, xc0, xc1, xsc, a.err);
    }
    // the staged tables and activations into shared memory: the codes into DP2's swizzled chunks and each run's
    // {d, s} with the exact sums of its halves' codes (fn::dense::act, which the shared units read)
    if (tid < n_grid16) {
        reinterpret_cast<uint4 *>(sv.grid)[tid] = grid_v;
    } else if (tid < n_grid16 + n_sgrid16) {
        reinterpret_cast<uint4 *>(sv.sgrid)[tid - n_grid16] = grid_v;
    }
    if (stager) {
        const int32_t b = tid / n_sub, jj = tid - b * n_sub;
        int32_t slo = 0, shi = 0;
#pragma unroll
        for (int32_t t = 0; t < 4; ++t) {
            slo = fn::dense::dp4a(word_of(xc0, t), 0x01010101u, slo);
            shi = fn::dense::dp4a(word_of(xc1, t), 0x01010101u, shi);
        }
        uint4 * xs = reinterpret_cast<uint4 *>(sv.xs + b * x_codes);
        xs[fn::dense::swz(2 * jj)] = xc0;
        xs[fn::dense::swz(2 * jj + 1)] = xc1;
        sv.xm[tid] = make_float4(xsc.x, xsc.y, __int_as_float(slo), __int_as_float(shi)); // the sums' bits
    }
    __syncthreads();

    if (t_first >= 0) {
        // each thread's activations of sub-block j: the 8 lanes of a quarter-warp share j (a broadcast), and the four
        // quarter-warps' 16-byte reads fall in distinct bank groups (R6)
        uint32_t act[B][8];
        float da[B];
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            const uint4 * xb = reinterpret_cast<const uint4 *>(sv.xs + b * x_codes);
            const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
            act[b][0] = c0.x;
            act[b][1] = c0.y;
            act[b][2] = c0.z;
            act[b][3] = c0.w;
            act[b][4] = c1.x;
            act[b][5] = c1.y;
            act[b][6] = c1.z;
            act[b][7] = c1.w;
            da[b] = sv.xm[b * n_sub + j].x;
        }

        // the block's routed unit i (10 u + g), -1 past the last: a volatile load at each use, so no entry is held in
        // a register across a unit (the loops hold the index alone, as they held the unit before XB2; IQ2_S at B = 4
        // spilled at the 96-register cap otherwise)
        const volatile uint16_t * lv = list;
        const int32_t n_mine = list[gu_max_ranks<C>];
        const auto unit_at = [&](int32_t i) -> int32_t { return i < n_mine ? int32_t(lv[i]) : -1; };

        // The block's routed unit i, then its next one (none: -1). The ring slot of chunk q is (P + q) % DEPTH, a
        // constant: data in flight is never moved between registers (a move waits for its load; EX1 v1 lost the stream
        // depth that way). The refill of the last chunks targets the next routed unit (phase (P + CC) % DEPTH), or
        // the current tile again after the block's last one, so no refill is predicated.
        const auto run_unit = [&](auto phase, int32_t i, int32_t buf) {
            constexpr int32_t P = decltype(phase)::value;
            const uint32_t mask = m.mask[unit_at(i) / n_group];
            const int32_t tn = unit_at(i + 1);
            const uint8_t * tile_next = tn >= 0 ? tile_of(tn) : tile;
            float * red_buf = sv.red + buf * B * 64 * red_stride;

            // the d chunk: rows 0..7's d, converted where the chunk is taken (XB2 (c): held as words, it compiled to a
            // copy right after the next unit's load, a wait long before the data was used)
            float dh8[8];
            // the aux chunk's aux8 bytes (x, y) and sc8 bytes (z, w) of rows 0..7; IQ2_S: its qh (x, y) and scales
            // (z, w) chunk, main chunk 0
            [[maybe_unused]] bytes8 ax8 = {0, 0}, sc8 = {0, 0};
            [[maybe_unused]] uint4 held = {0, 0, 0, 0}; // IQ3: the sign words waiting
            [[maybe_unused]] int32_t is8[B];            // Q8_0: a row's integer sums, over its two chunks
            // the unit's chunks in order; out(rd, r) finishes row r of a record (Q8_0 finishes its rows itself)
            const auto chunks = [&](auto out) {
#pragma unroll
                for (int32_t q = 0; q < CC; ++q) {
                    const int32_t sl = (P + q) % DEPTH;
                    const uint4 v = rg.s[sl];
                    // refill the slot just taken: chunk q + DEPTH of the sequence, in this unit or the next
                    const int32_t qn = q + DEPTH;
                    const uint32_t z = order_after(v.x, mz);
                    rg.s[sl] = qn < CC ? load_chunk(tile, qn, z) : load_chunk(tile_next, qn - CC, z);
                    if (q == 0) {
#pragma unroll
                        for (int32_t mr = 0; mr < 8; ++mr) {
                            dh8[mr] = (mr & 1) ? half_hi(word_of(v, mr >> 1)) : half_lo(word_of(v, mr >> 1));
                        }
                        continue;
                    }
                    if (NA == 1 && q == 1) {
                        ax8 = take8(v.x, v.y);
                        sc8 = take8(v.z, v.w);
                        continue;
                    }
                    const int32_t c = q - 1 - NA;
                    if constexpr (F == fmt::q8_0) {
                        // chunk c: half c & 1 of row c / 2's sub-block, against codes 16 (c & 1) .. of each routed row
                        const int32_t mrow = c >> 1, hh = c & 1;
#pragma unroll
                        for (int32_t b = 0; b < B; ++b) {
                            if (((mask >> b) & 1u) == 0) {
                                continue;
                            }
                            int32_t acc = hh == 0 ? 0 : is8[b];
#pragma unroll
                            for (int32_t i = 0; i < 4; ++i) {
                                acc = __dp4a(static_cast<int32_t>(word_of(v, i)),
                                             static_cast<int32_t>(act[b][4 * hh + i]), acc);
                            }
                            is8[b] = acc;
                        }
                        if (hh == 1) {
                            row_out_q8<B>(is8, dh8[mrow], da, mask, red_buf, rs + 8 * mrow, w, lane);
                        }
                        continue;
                    }
                    row_dec rd;
                    if constexpr (F == fmt::iq2_xxs || F == fmt::iq2_xs) {
                        const uint2 * grid = reinterpret_cast<const uint2 *>(sv.grid);
#pragma unroll
                        for (int32_t h = 0; h < 2; ++h) {
                            const int32_t mrow = 2 * c + h;
                            const uint2 qq = h == 0 ? make_uint2(v.x, v.y) : make_uint2(v.z, v.w);
                            const float dh = dh8[mrow];
                            if constexpr (F == fmt::iq2_xxs) {
                                dec_iq2_xxs(rd, qq, dh, grid);
                                out(rd, rs + 8 * mrow);
                            } else {
                                const uint32_t sc = byte8(ax8, mrow);
                                dec_iq2_xs(rd, qq, sc, dh, grid);
                                out(rd, rs + 8 * mrow);
                            }
                        }
                    } else if constexpr (F == fmt::iq2_s) {
                        if (c == 0) {
                            ax8 = take8(v.x, v.y); // qh of rows 0..7
                            sc8 = take8(v.z, v.w); // scales of rows 0..7
                            continue;
                        }
                        const uint2 * grid = reinterpret_cast<const uint2 *>(sv.grid);
#pragma unroll
                        for (int32_t h = 0; h < 2; ++h) {
                            const int32_t mrow = 2 * (c - 1) + h;
                            const float dh = dh8[mrow];
                            const uint32_t qh = byte8(ax8, mrow);
                            const uint32_t sc = byte8(sc8, mrow);
                            dec_iq2_s(rd, h == 0 ? v.x : v.z, h == 0 ? v.y : v.w, qh, sc, dh, grid);
                            out(rd, rs + 8 * mrow);
                        }
                    } else {
                        // IQ3: chunk k of row quad hq: k 0 [q r0][sg r0][sg r1], k 1 [q r1][sg r2][sg r3],
                        // k 2 [q r2][q r3]
                        const uint32_t * grid = reinterpret_cast<const uint32_t *>(sv.grid);
                        const int32_t hq = c / 3;
                        const int32_t k = c - 3 * hq;
                        const auto row = [&](int32_t mrow, uint2 qq, uint32_t sg) {
                            const float dh = dh8[mrow];
                            if constexpr (F == fmt::iq3_xxs) {
                                dec_iq3_xxs(rd, qq, sg, dh, grid);
                            } else {
                                const uint32_t qh = byte8(ax8, mrow);
                                const uint32_t scb = byte8(sc8, mrow);
                                dec_iq3_s(rd, qq, sg, qh, (j & 1) ? scb >> 4 : scb & 0xfu, dh, grid);
                            }
                            out(rd, rs + 8 * mrow);
                        };
                        if (k == 0) {
                            held.x = v.w;
                            row(4 * hq + 0, make_uint2(v.x, v.y), v.z);
                        } else if (k == 1) {
                            held.y = v.z;
                            held.z = v.w;
                            row(4 * hq + 1, make_uint2(v.x, v.y), held.x);
                        } else {
                            row(4 * hq + 2, make_uint2(v.x, v.y), held.y);
                            row(4 * hq + 3, make_uint2(v.z, v.w), held.z);
                        }
                    }
                }
            };
            // XB3 K1b: one body. Every record's rows run the unit's first routed row b0 with no row test (row_out1 on
            // b0's codes and d, read once a unit from shared memory by b0's address, never act[b0]), then the other
            // routed rows under one unit-uniform test, inside which today's per-row guard runs over the rest. Each
            // row's partial is today's, in today's row order. Q8_0 (the draft's) keeps its own loop. At B = 4 the rows
            // after b0 are read a unit too, into slots k = row - b0 - 1 (row_out over B - 1 slots, its partials from
            // row b0 + 1 on), instead of today's act held a block: act1 beside all four rows spilled at 96 registers
            // (IQ2_S, IQ3_S).
            if constexpr (F == fmt::q8_0) {
                chunks([](const row_dec &, int32_t) {});
            } else {
                constexpr bool SPLIT = F == fmt::iq2_xs || F == fmt::iq2_s;
                const auto row_act = [&](int32_t b, uint32_t (&o)[8], float & d) {
                    const uint4 * xb = reinterpret_cast<const uint4 *>(sv.xs + b * x_codes);
                    const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
                    o[0] = c0.x;
                    o[1] = c0.y;
                    o[2] = c0.z;
                    o[3] = c0.w;
                    o[4] = c1.x;
                    o[5] = c1.y;
                    o[6] = c1.z;
                    o[7] = c1.w;
                    d = sv.xm[b * n_sub + j].x;
                };
                const int32_t b0 = __ffs(static_cast<int32_t>(mask)) - 1;
                uint32_t act1[8];
                float da1;
                row_act(b0, act1, da1);
                if constexpr (B == 5) {
                    // SP2: the rows after b0 read from shared memory at each record (row_out_s), so the live
                    // activation registers stay act1's (B = 4's actr beside act1 already sat at the cap)
                    chunks([&](const row_dec & rd, int32_t r) {
                        const int32_t bb = opaque(b0);
                        row_out1<SPLIT>(rd, act1, da1, red_buf + bb * 64 * red_stride, r, w, lane);
                        const uint32_t rest = mask >> (bb + 1); // bit k: row b0 + 1 + k
                        if (rest != 0u) {
                            row_out_s<SPLIT>(rd, sv.xs, sv.xm, j, bb + 1, rest, red_buf, r, w, lane);
                        }
                    });
                } else if constexpr (B == 4) {
                    uint32_t actr[B - 1][8];
                    float unused = 0.0f;
#pragma unroll
                    for (int32_t k = 0; k < B - 1; ++k) {
                        row_act(min(b0 + 1 + k, B - 1), actr[k], unused); // past row B - 1: never routed, not read
                    }
                    // the rows' places and the rest's mask from b0 at each use (opaque): held a unit, they spilled
                    chunks([&](const row_dec & rd, int32_t r) {
                        const int32_t bb = opaque(b0);
                        row_out1<SPLIT>(rd, act1, da1, red_buf + bb * 64 * red_stride, r, w, lane);
                        const uint32_t rest = mask >> (bb + 1); // bit k: row b0 + 1 + k
                        if (rest != 0u) {
                            const smem_da dar = {sv.xm + (bb + 1) * n_sub + j};
                            row_out<SPLIT, B - 1>(rd, actr, dar, rest, red_buf + (bb + 1) * 64 * red_stride, r, w,
                                                  lane);
                        }
                    });
                } else {
                    float * red1 = red_buf + b0 * 64 * red_stride;
                    const uint32_t rest = mask & (mask - 1u);
                    chunks([&](const row_dec & rd, int32_t r) {
                        row_out1<SPLIT>(rd, act1, da1, red1, r, w, lane);
                        if (rest != 0u) {
                            row_out<SPLIT, B>(rd, act, da, rest, red_buf, r, w, lane);
                        }
                    });
                }
            }
            tile = tile_next;
            __syncthreads();
            // epilogue of the unit: warp b (a routed row) sums its 20 warp partials per channel, GLU, FNFL's format
            if (w < B && ((mask >> w) & 1u) != 0) {
                const int32_t t = unit_at(i);
                const int32_t u = t / n_group;
                glu_out<gu_warps, red_stride>(a, red_buf + w * 64 * red_stride, slot_of(m, u, w), t - u * n_group,
                                              lane);
            }
        };
        // the ring's phase is a compile-time constant (R12): when a unit's chunks are a multiple of DEPTH every unit
        // starts at phase 0, so one copy of the unit body serves them all (EX2 B3: the executed code a launch fetches
        // cold, M9); an odd CC takes two units a trip, one copy a phase. IQ2_S at B = 4 keeps two copies: its single
        // copy's runtime buffer index spilled 8 bytes at the 96-register cap.
        static_assert((2 * CC) % DEPTH == 0, "the phase repeats every two units");
        if constexpr (CC % DEPTH == 0 && !(F == fmt::iq2_s && B == 4)) {
#pragma unroll 1
            for (int32_t i = 0; i < n_mine; ++i) {
                run_unit(std::integral_constant<int32_t, 0>{}, i, i & 1);
            }
        } else {
            for (int32_t i = 0; i < n_mine; i += 2) {
                run_unit(std::integral_constant<int32_t, 0>{}, i, 0);
                if (i + 1 < n_mine) {
                    run_unit(std::integral_constant<int32_t, CC % DEPTH>{}, i + 1, 1);
                }
            }
        }
    }
    // the block's shared ranks after its routed units, in rank order: two when the shared class straddles a snake turn
    // (XB2 Amendment 1). Each takes the whole partials region, so the unit before it must have finished its epilogue.
#pragma unroll 1
    for (int32_t i = list[gu_max_ranks<C>]; i < list[gu_max_ranks<C> + 1]; ++i) {
        const int32_t sh_g = list[i] - order_shared_of(C);
        if (i > 0) {
            __syncthreads();
        }
        const fn::dense::act x = {sv.xs, sv.xm, D};
        // the thread's warp and lane read again (opaque), not held across the routed units (K1b at B = 4: IQ3_XXS
        // spilled one register here)
        const int32_t tx = opaque(static_cast<int32_t>(threadIdx.x));
        const int32_t sw = tx >> 5, sl = tx & 31;
        shared_unit<FG, FU, B>(a, sh_g, sw, sl, x, sv.sgrid, sv.red);
        __syncthreads();
        if (sw < B) {
            glu_out<sh_parts, sh_stride>(a, sv.red + sw * 64 * sh_stride, m.n_slots + sw, sh_g, sl);
        }
    }
}

// XB3 part 6, the IQ2 body: gate_up_kernel's work for the three IQ2 routed formats in one symbol a B, the formats from
// a.u_f, a.u_fg, a.u_fu. The sections that depend on the routed format (the ring's priming, the signed table's copy,
// the unit loop) are each format's compile-time code behind one block-uniform switch (with_f), and the shared unit each
// pair's behind another (with_pair). Everything else, the meta and order, the staging, the shared memory layout, is one
// copy for all. It is gate_up_kernel's text with its sections in with_f and with_pair; gate_up_kernel itself is kept
// as it was (generic sections there moved ptxas's schedule of the IQ3 and Q8_0 instantiations). B = 1-3 (at B = 4 the
// body with every row's fragment held spilled 344 bytes at the 96-register cap); SP4 H2 (gate_up_args::map 2): B = 4
// and 5 too, holding b0's fragment alone and running the other routed rows through row_out_c's compact loop. XB3 C1:
// one kernel template on the format family FAM: gu_iq2 (the IQ2 body, B = 1-5) and gu_iq3 (the IQ3 formats at B = 4 and
// 5, map 3), the family setting the formats with_f takes, the routed table region and the shared pairs of the direct
// switch. A kernel, not a __device__ body under two kernels: through a forceinline body the IQ2 instances' SASS at
// B = 1-3 changed (the body is optimized alone before it is inlined); this form keeps it instruction for instruction.
template <typename FAM, int B, int DEPTH>
__global__ void __launch_bounds__(gu_threads, 1) gate_up_kernel_u(gate_up_args a) {
    constexpr int32_t C = cap_of(B); // SP2: the routing capacity (experts.h)
    constexpr bool U = true; // the generic sections' run-time form; F, FG and FU are unused placeholders
    constexpr fmt F = fmt::iq2_xxs;
    constexpr dfmt FG = dfmt::q4_k, FU = dfmt::q4_k;
    static_assert(!FAM::iq3 || B >= 4, "the IQ3 body: B = 4 and 5");
    // the routed format's sections: body(gu_fc<f>{}) for this launch's f
    const auto with_f = [&](auto && body) {
        if constexpr (!U) {
            body(gu_fc<F>{});
        } else if constexpr (FAM::iq3) {
            switch (static_cast<fmt>(a.u_f)) {
                case fmt::iq3_xxs:
                    body(gu_fc<fmt::iq3_xxs>{});
                    break;
                default:
                    body(gu_fc<fmt::iq3_s>{});
                    break;
            }
        } else {
            switch (static_cast<fmt>(a.u_f)) {
                case fmt::iq2_xxs:
                    body(gu_fc<fmt::iq2_xxs>{});
                    break;
                case fmt::iq2_xs:
                    body(gu_fc<fmt::iq2_xs>{});
                    break;
                default:
                    body(gu_fc<fmt::iq2_s>{});
                    break;
            }
        }
    };
    extern __shared__ uint4 smem_u4[];
    uint8_t * smem = reinterpret_cast<uint8_t *>(smem_u4);
    constexpr int32_t xs_off = U ? FAM::table + 2048 : gu_grid_len<F, FG, FU>();
    constexpr int32_t xm_off = xs_off + B * x_codes;
    constexpr int32_t red_off = xm_off + B * x_meta;
    constexpr int32_t meta_off = red_off + 2 * B * 64 * red_stride * 4;
    static_assert(xs_off % 16 == 0 && xm_off % 16 == 0 && red_off % 16 == 0 && meta_off % 16 == 0,
                  "16-byte aligned shared regions");
    const gu_smem_view<C> sv = {smem,
                             reinterpret_cast<uint32_t *>(smem + (U                         ? FAM::table
                                                                  : extra_iq3s<F, FG, FU>() ? gu_layout_of(F).grid_len
                                                                                            : 0)),
                             smem + xs_off,
                             reinterpret_cast<float4 *>(smem + xm_off),
                             reinterpret_cast<float *>(smem + red_off),
                             reinterpret_cast<meta_c<C> *>(smem + meta_off)};
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t lane = tid & 31;
    const int32_t w = tid >> 5;
    const int32_t j = tid >> 3;
    const int32_t rs = tid & 7;
    const uint32_t mz = gridDim.x == n_sm ? 0u : ~0u;

    // The grid tables and the rows' activations first, neither depends on the routing: coalesced 16-byte loads through
    // L1 (R3: written before this kernel and read by every SM; EX1's 4-byte L1-bypassing loads cost 7.7x the ideal L2
    // sectors). Thread i < 80 B takes run (b, j) = (i / 80, i % 80) of the rows. They land in shared memory after the
    // first weight chunks are issued.
    // the routed table (run-time data in the IQ2 body): a signed table is copied after the ring is primed
    const gu_layout gl = gu_layout_of(U ? static_cast<fmt>(a.u_f) : F);
    const bool ex_iq3s = U ? a.u_fg == int32_t(dfmt::iq3_s) || a.u_fu == int32_t(dfmt::iq3_s) : extra_iq3s<F, FG, FU>();
    const int32_t n_grid16 = gl.grid_len / 16;
    const int32_t n_sgrid16 = ex_iq3s ? 2048 / 16 : 0;
    static_assert(FAM::table / 16 + 2048 / 16 <= gu_threads, "one 16-byte load a thread");
    uint4 grid_v = {0, 0, 0, 0};
    if (tid < n_grid16) {
        grid_v = __ldg(reinterpret_cast<const uint4 *>(a.grids + gl.grid_off) + tid);
    } else if (tid < n_grid16 + n_sgrid16) {
        grid_v = __ldg(reinterpret_cast<const uint4 *>(a.grids + iq3s_off) + (tid - n_grid16));
    }
    const bool stager = tid < B * n_sub;
    // HC3: with X-HMIX, a stager whose run is in the peer's half polls it after the weight ring is primed (below)
    const bool peer_run = stager && a.hmix != nullptr && (tid % n_sub) / (n_sub / 2) == a.hmix_peer_half;
    uint4 xc0 = {0, 0, 0, 0}, xc1 = {0, 0, 0, 0};
    float2 xsc = {0.0f, 0.0f};
    if (stager && !peer_run) {
        const uint4 * src = reinterpret_cast<const uint4 *>(a.x.codes) + 2 * tid; // codes [B][2560]: run tid
        xc0 = __ldg(src);
        xc1 = __ldg(src + 1);
        xsc = __ldg(reinterpret_cast<const float2 *>(a.x.scales) + tid);
    }
    // XB3 part 1: warp 0 builds the routing meta (71b92a9's build_meta) and the block's units (block_units) while the
    // other warps' loads are in flight; block 0 writes the struct for down (the kernel boundary orders it before down's
    // read). The block's first unit comes from the order, so the ring's first loads follow the barrier.
    uint16_t * list = reinterpret_cast<uint16_t *>(smem + meta_off + gu_meta_bytes<C>);
    if (w == 0) {
        build_meta_of<B>([&](int32_t i) { return __ldg(a.ids + i); }, *sv.m);
        __syncwarp();
        if (sv.m->bad == 0) {
            block_units<C>(*sv.m, a.cw, static_cast<int32_t>(blockIdx.x), list,
                        reinterpret_cast<uint16_t *>(a.meta + meta_order_of(C)));
        }
        if (blockIdx.x == 0) {
            const uint4 * src = reinterpret_cast<const uint4 *>(sv.m);
#pragma unroll 1
            for (int32_t i = lane; i < meta_words_c<C>; i += 32) {
                reinterpret_cast<uint4 *>(a.meta)[i] = src[i];
            }
        }
    }
    __syncthreads();
    const meta_c<C> & m = *sv.m;
    if (m.bad != 0) {
        if (tid == 0 && blockIdx.x == 0 && a.status != nullptr) {
            *a.status = 1u;
        }
        return;
    }
    if (list[gu_max_ranks<C> + 1] == 0) { // no rank of the launch's 10 U + 10 is this block's
        return;
    }
    // the block's first routed unit (its tile primes the ring), or -1
    const int32_t t_first = list[gu_max_ranks<C>] > 0 ? int32_t(list[0]) : -1;
    const int32_t sb = j >> 3;
    // byte offset of chunk q (0 = d, then aux when the format has one, then main) of this thread's record in format
    // C::f's tile; Q8_0's d chunk is the record's own (a d a sub-block), the others' their superblock's
    const auto chunk_off = [&](auto fc, int32_t q) -> int64_t {
        using C = decltype(fc);
        return q == 0 ? d_off(C::f) + (C::f == fmt::q8_0 ? tid : sb * 8 + rs) * 16
                      : (int64_t(q - 1 - C::NA) * gu_threads + tid) * 16;
    };
    // a chunk of format C::f's sequence: the aux chunk is one or two 8-byte loads, every other chunk one 16-byte load
    const auto load_chunk = [&](auto fc, const uint8_t * tl, int32_t q, uint32_t z) -> uint4 {
        using C = decltype(fc);
        if (C::NA == 1 && q == 1) {
            uint4 v = {0, 0, 0, 0};
            if (C::L.aux8) {
                const uint2 x = __ldcs(reinterpret_cast<const uint2 *>(tl + aux8_off(C::f) + tid * 8 + z));
                v.x = x.x;
                v.y = x.y;
            }
            if (C::L.sc8) {
                const uint2 x =
                    __ldcs(reinterpret_cast<const uint2 *>(tl + sc8_off(C::f) + ((j >> 1) * 8 + rs) * 8 + z));
                v.z = x.x;
                v.w = x.y;
            }
            return v;
        }
        return __ldcs(reinterpret_cast<const uint4 *>(tl + chunk_off(fc, q) + z));
    };
    const auto tile_of = [&](auto fc, int32_t t) -> const uint8_t * {
        const int32_t u = t / n_group;
        return a.w + (int64_t(m.e[u]) * n_group + (t - u * n_group)) * decltype(fc)::T;
    };

    // prime the ring: the first DEPTH chunks of the sequence (units are at least CC >= DEPTH chunks long)
    ring<DEPTH> rg;
    const uint8_t * tile = a.w;
    with_f([&](auto fc) {
        static_assert(decltype(fc)::CC >= DEPTH, "a unit has at least DEPTH chunks");
        if (t_first >= 0) {
            tile = tile_of(fc, t_first);
#pragma unroll
            for (int32_t q = 0; q < DEPTH; ++q) {
                rg.s[q] = load_chunk(fc, tile, q, 0u);
            }
        }
    });
    // X-HMIX's acquire (xchg.h): the peer half's runs, now that this block's first weight chunks are in flight
    if (peer_run) {
        fn::xchg::hmix_run(a.hmix, a.hmix_peer_half, tid / n_sub, tid % n_sub, xc0, xc1, xsc, a.err);
    }
    // the staged tables and activations into shared memory: the codes into DP2's swizzled chunks and each run's
    // {d, s} with the exact sums of its halves' codes (fn::dense::act, which the shared units read)
    if (tid < n_grid16) {
        reinterpret_cast<uint4 *>(sv.grid)[tid] = grid_v;
    } else if (tid < n_grid16 + n_sgrid16) {
        reinterpret_cast<uint4 *>(sv.sgrid)[tid - n_grid16] = grid_v;
    }
    if (stager) {
        const int32_t b = tid / n_sub, jj = tid - b * n_sub;
        int32_t slo = 0, shi = 0;
#pragma unroll
        for (int32_t t = 0; t < 4; ++t) {
            slo = fn::dense::dp4a(word_of(xc0, t), 0x01010101u, slo);
            shi = fn::dense::dp4a(word_of(xc1, t), 0x01010101u, shi);
        }
        uint4 * xs = reinterpret_cast<uint4 *>(sv.xs + b * x_codes);
        xs[fn::dense::swz(2 * jj)] = xc0;
        xs[fn::dense::swz(2 * jj + 1)] = xc1;
        sv.xm[tid] = make_float4(xsc.x, xsc.y, __int_as_float(slo), __int_as_float(shi)); // the sums' bits
    }
    __syncthreads();

    if (t_first >= 0) {
        // each thread's activations of sub-block j: the 8 lanes of a quarter-warp share j (a broadcast), and the four
        // quarter-warps' 16-byte reads fall in distinct bank groups (R6). SP4 H2: at B = 4 and 5 no row's fragment is
        // held a block (act's BA = 1, and neither array is read: row_out_c reads each row's at its turn; da keeps B for
        // the Q8_0 section's row_out_q8 signature, which this body never runs)
        constexpr int32_t BA = B <= 3 ? B : 1;
        [[maybe_unused]] uint32_t act[BA][8];
        [[maybe_unused]] float da[B];
        if constexpr (B <= 3) {
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
                const uint4 * xb = reinterpret_cast<const uint4 *>(sv.xs + b * x_codes);
                const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
                act[b][0] = c0.x;
                act[b][1] = c0.y;
                act[b][2] = c0.z;
                act[b][3] = c0.w;
                act[b][4] = c1.x;
                act[b][5] = c1.y;
                act[b][6] = c1.z;
                act[b][7] = c1.w;
                da[b] = sv.xm[b * n_sub + j].x;
            }
        }

        // the block's routed unit i (10 u + g), -1 past the last: a volatile load at each use, so no entry is held in
        // a register across a unit (the loops hold the index alone, as they held the unit before XB2; IQ2_S at B = 4
        // spilled at the 96-register cap otherwise)
        const volatile uint16_t * lv = list;
        const int32_t n_mine = list[gu_max_ranks<C>];
        const auto unit_at = [&](int32_t i) -> int32_t { return i < n_mine ? int32_t(lv[i]) : -1; };

        with_f([&](auto fc) {
            using C = decltype(fc);
            // The block's routed unit i, then its next one (none: -1). The ring slot of chunk q is (P + q) % DEPTH, a
            // constant: data in flight is never moved between registers (a move waits for its load; EX1 v1 lost the
            // stream depth that way). The refill of the last chunks targets the next routed unit (phase (P + C::CC) %
            // DEPTH), or the current tile again after the block's last one, so no refill is predicated.
            const auto run_unit = [&](auto phase, int32_t i, int32_t buf) {
                constexpr int32_t P = decltype(phase)::value;
                const uint32_t mask = m.mask[unit_at(i) / n_group];
                const int32_t tn = unit_at(i + 1);
                const uint8_t * tile_next = tn >= 0 ? tile_of(fc, tn) : tile;
                float * red_buf = sv.red + buf * B * 64 * red_stride;

                // the d chunk: rows 0..7's d, converted where the chunk is taken (XB2 (c): held as words, it compiled
                // to a copy right after the next unit's load, a wait long before the data was used)
                float dh8[8];
                // the aux chunk's aux8 bytes (x, y) and sc8 bytes (z, w) of rows 0..7; IQ2_S: its qh (x, y) and scales
                // (z, w) chunk, main chunk 0
                [[maybe_unused]] bytes8 ax8 = {0, 0}, sc8 = {0, 0};
                [[maybe_unused]] uint4 held = {0, 0, 0, 0}; // IQ3: the sign words waiting
                [[maybe_unused]] int32_t is8[B];            // Q8_0: a row's integer sums, over its two chunks
                // the unit's chunks in order; out(rd, r) finishes row r of a record (Q8_0 finishes its rows itself)
                const auto chunks = [&](auto out) {
#pragma unroll
                    for (int32_t q = 0; q < C::CC; ++q) {
                        const int32_t sl = (P + q) % DEPTH;
                        const uint4 v = rg.s[sl];
                        // refill the slot just taken: chunk q + DEPTH of the sequence, in this unit or the next
                        const int32_t qn = q + DEPTH;
                        const uint32_t z = order_after(v.x, mz);
                        rg.s[sl] = qn < C::CC ? load_chunk(fc, tile, qn, z) : load_chunk(fc, tile_next, qn - C::CC, z);
                        if (q == 0) {
#pragma unroll
                            for (int32_t mr = 0; mr < 8; ++mr) {
                                dh8[mr] = (mr & 1) ? half_hi(word_of(v, mr >> 1)) : half_lo(word_of(v, mr >> 1));
                            }
                            continue;
                        }
                        if (C::NA == 1 && q == 1) {
                            ax8 = take8(v.x, v.y);
                            sc8 = take8(v.z, v.w);
                            continue;
                        }
                        const int32_t c = q - 1 - C::NA;
                        if constexpr (C::f == fmt::q8_0) {
                            // chunk c: half c & 1 of row c / 2's sub-block, against codes 16 (c & 1) .. of each routed
                            // row
                            const int32_t mrow = c >> 1, hh = c & 1;
#pragma unroll
                            for (int32_t b = 0; b < B; ++b) {
                                if (((mask >> b) & 1u) == 0) {
                                    continue;
                                }
                                int32_t acc = hh == 0 ? 0 : is8[b];
#pragma unroll
                                for (int32_t i = 0; i < 4; ++i) {
                                    acc = __dp4a(static_cast<int32_t>(word_of(v, i)),
                                                 static_cast<int32_t>(act[b][4 * hh + i]), acc);
                                }
                                is8[b] = acc;
                            }
                            if (hh == 1) {
                                row_out_q8<B>(is8, dh8[mrow], da, mask, red_buf, rs + 8 * mrow, w, lane);
                            }
                            continue;
                        }
                        row_dec rd;
                        if constexpr (C::f == fmt::iq2_xxs || C::f == fmt::iq2_xs) {
                            [[maybe_unused]] const uint2 * grid = reinterpret_cast<const uint2 *>(sv.grid); // IQ2_XS
#pragma unroll
                            for (int32_t h = 0; h < 2; ++h) {
                                const int32_t mrow = 2 * c + h;
                                const uint2 qq = h == 0 ? make_uint2(v.x, v.y) : make_uint2(v.z, v.w);
                                const float dh = dh8[mrow];
                                if constexpr (C::f == fmt::iq2_xxs) {
                                    dec_iq2_xxs(rd, qq, dh, grid);
                                    out(rd, rs + 8 * mrow);
                                } else {
                                    const uint32_t sc = byte8(ax8, mrow);
                                    dec_iq2_xs(rd, qq, sc, dh, grid);
                                    out(rd, rs + 8 * mrow);
                                }
                            }
                        } else if constexpr (C::f == fmt::iq2_s) {
                            if (c == 0) {
                                ax8 = take8(v.x, v.y); // qh of rows 0..7
                                sc8 = take8(v.z, v.w); // scales of rows 0..7
                                continue;
                            }
                            const uint2 * grid = reinterpret_cast<const uint2 *>(sv.grid);
#pragma unroll
                            for (int32_t h = 0; h < 2; ++h) {
                                const int32_t mrow = 2 * (c - 1) + h;
                                const float dh = dh8[mrow];
                                const uint32_t qh = byte8(ax8, mrow);
                                const uint32_t sc = byte8(sc8, mrow);
                                dec_iq2_s(rd, h == 0 ? v.x : v.z, h == 0 ? v.y : v.w, qh, sc, dh, grid);
                                out(rd, rs + 8 * mrow);
                            }
                        } else {
                            // IQ3: chunk k of row quad hq: k 0 [q r0][sg r0][sg r1], k 1 [q r1][sg r2][sg r3],
                            // k 2 [q r2][q r3]
                            const uint32_t * grid = reinterpret_cast<const uint32_t *>(sv.grid);
                            const int32_t hq = c / 3;
                            const int32_t k = c - 3 * hq;
                            const auto row = [&](int32_t mrow, uint2 qq, uint32_t sg) {
                                const float dh = dh8[mrow];
                                if constexpr (C::f == fmt::iq3_xxs) {
                                    dec_iq3_xxs(rd, qq, sg, dh, grid);
                                } else {
                                    const uint32_t qh = byte8(ax8, mrow);
                                    const uint32_t scb = byte8(sc8, mrow);
                                    dec_iq3_s(rd, qq, sg, qh, (j & 1) ? scb >> 4 : scb & 0xfu, dh, grid);
                                }
                                out(rd, rs + 8 * mrow);
                            };
                            if (k == 0) {
                                held.x = v.w;
                                row(4 * hq + 0, make_uint2(v.x, v.y), v.z);
                            } else if (k == 1) {
                                held.y = v.z;
                                held.z = v.w;
                                row(4 * hq + 1, make_uint2(v.x, v.y), held.x);
                            } else {
                                row(4 * hq + 2, make_uint2(v.x, v.y), held.y);
                                row(4 * hq + 3, make_uint2(v.z, v.w), held.z);
                            }
                        }
                    }
                };
                // XB3 K1b: one body. Every record's rows run the unit's first routed row b0 with no row test (row_out1
                // on b0's codes and d, read once a unit from shared memory by b0's address, never act[b0]), then the
                // other routed rows under one unit-uniform test, inside which today's per-row guard runs over the rest.
                // Each row's partial is today's, in today's row order. Q8_0 (the draft's) keeps its own loop. At B = 4
                // the rows after b0 are read a unit too, into slots k = row - b0 - 1 (row_out over B - 1 slots, its
                // partials from row b0 + 1 on), instead of today's act held a block: act1 beside all four rows spilled
                // at 96 registers (IQ2_S, IQ3_S).
                if constexpr (C::f == fmt::q8_0) {
                    chunks([](const row_dec &, int32_t) {});
                } else {
                    constexpr bool SPLIT = C::f == fmt::iq2_xs || C::f == fmt::iq2_s;
                    const auto row_act = [&](int32_t b, uint32_t (&o)[8], float & d) {
                        const uint4 * xb = reinterpret_cast<const uint4 *>(sv.xs + b * x_codes);
                        const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
                        o[0] = c0.x;
                        o[1] = c0.y;
                        o[2] = c0.z;
                        o[3] = c0.w;
                        o[4] = c1.x;
                        o[5] = c1.y;
                        o[6] = c1.z;
                        o[7] = c1.w;
                        d = sv.xm[b * n_sub + j].x;
                    };
                    const int32_t b0 = __ffs(static_cast<int32_t>(mask)) - 1;
                    uint32_t act1[8];
                    float da1;
                    row_act(b0, act1, da1);
                    if constexpr (B >= 4) {
                        // SP4 H2: b0 by row_out1, then the unit's other routed rows (the mask without b0, bit b: row
                        // b) through row_out_c's one compact loop, at every decoded row
                        float * red1 = red_buf + b0 * 64 * red_stride;
                        const uint32_t rest = mask & (mask - 1u);
                        chunks([&](const row_dec & rd, int32_t r) {
                            row_out1<SPLIT>(rd, act1, da1, red1, r, w, lane);
                            row_out_c<SPLIT>(rd, sv.xs, sv.xm, j, rest, red_buf, r, w, lane);
                        });
                    } else {
                        float * red1 = red_buf + b0 * 64 * red_stride;
                        const uint32_t rest = mask & (mask - 1u);
                        chunks([&](const row_dec & rd, int32_t r) {
                            row_out1<SPLIT>(rd, act1, da1, red1, r, w, lane);
                            if (rest != 0u) {
                                row_out<SPLIT, B>(rd, act, da, rest, red_buf, r, w, lane);
                            }
                        });
                    }
                }
                tile = tile_next;
                __syncthreads();
                // epilogue of the unit: warp b (a routed row) sums its 20 warp partials per channel, GLU, FNFL's format
                if (w < B && ((mask >> w) & 1u) != 0) {
                    const int32_t t = unit_at(i);
                    const int32_t u = t / n_group;
                    glu_out<gu_warps, red_stride>(a, red_buf + w * 64 * red_stride, slot_of(m, u, w), t - u * n_group,
                                                  lane);
                }
            };
            // the ring's phase is a compile-time constant (R12): when a unit's chunks are a multiple of DEPTH every
            // unit starts at phase 0, so one copy of the unit body serves them all (EX2 B3: the executed code a launch
            // fetches cold, M9); an odd C::CC takes two units a trip, one copy a phase. IQ2_S at B = 4 keeps two
            // copies: its single copy's runtime buffer index spilled 8 bytes at the 96-register cap (the per-triple
            // kernel's B = 4; this body ran B = 1-3 only before SP4 H2, whose B = 4 and 5 keep one copy).
            static_assert((2 * C::CC) % DEPTH == 0, "the phase repeats every two units");
            if constexpr (C::CC % DEPTH == 0) {
#pragma unroll 1
                for (int32_t i = 0; i < n_mine; ++i) {
                    run_unit(std::integral_constant<int32_t, 0>{}, i, i & 1);
                }
            } else {
                for (int32_t i = 0; i < n_mine; i += 2) {
                    run_unit(std::integral_constant<int32_t, 0>{}, i, 0);
                    if (i + 1 < n_mine) {
                        run_unit(std::integral_constant<int32_t, C::CC % DEPTH>{}, i + 1, 1);
                    }
                }
            }
        });
    }
    // the shared expert's pair: body(gate, up) as integral constants, this launch's (the IQ2 body: each pair's code
    // behind one block-uniform switch, the ten pairs the IQ2 layers meet; the launcher admits only deployed triples)
    const auto with_pair = [&](auto && body) {
        using std::integral_constant;
        if constexpr (!U) {
            body(integral_constant<dfmt, FG>{}, integral_constant<dfmt, FU>{});
        } else {
#define FN_GU_PAIR(g, u)                                                                                               \
    case int32_t(dfmt::g) * 16 + int32_t(dfmt::u):                                                                     \
        body(integral_constant<dfmt, dfmt::g>{}, integral_constant<dfmt, dfmt::u>{});                                  \
        break;
            switch (a.u_fg * 16 + a.u_fu) {
                FN_GU_PAIR(iq4_xs, iq3_s)
                FN_GU_PAIR(q4_k, q4_k)
                FN_GU_PAIR(iq4_xs, q6_k)
                FN_GU_PAIR(iq3_s, iq4_xs)
                FN_GU_PAIR(iq3_s, iq3_s)
                FN_GU_PAIR(q4_k, iq3_s)
                FN_GU_PAIR(q4_k, q5_k)
                FN_GU_PAIR(q5_k, q4_k)
                FN_GU_PAIR(iq4_xs, q4_k)
                FN_GU_PAIR(iq4_xs, iq4_xs)
                default:
                    break;
            }
#undef FN_GU_PAIR
        }
    };
    // the block's shared ranks after its routed units, in rank order: two when the shared class straddles a snake turn
    // (XB2 Amendment 1). Each takes the whole partials region, so the unit before it must have finished its epilogue.
#pragma unroll 1
    for (int32_t i = list[gu_max_ranks<C>]; i < list[gu_max_ranks<C> + 1]; ++i) {
        const int32_t sh_g = list[i] - order_shared_of(C);
        if (i > 0) {
            __syncthreads();
        }
        const fn::dense::act x = {sv.xs, sv.xm, D};
        // the thread's warp and lane read again (opaque), not held across the routed units (K1b at B = 4: IQ3_XXS
        // spilled one register here)
        const int32_t tx = opaque(static_cast<int32_t>(threadIdx.x));
        const int32_t sw = tx >> 5, sl = tx & 31;
        if constexpr (B <= 3) {
            with_pair([&](auto gc, auto uc) {
                shared_unit<decltype(gc)::value, decltype(uc)::value, B>(a, sh_g, sw, sl, x, sv.sgrid, sv.red);
            });
        } else {
            // SP4 H2: a direct switch (through with_pair ptxas outlined the pair at B = 4 and 5 in SP4's kernel)
#define FN_GU_PAIR(g, u)                                                                                               \
    case int32_t(dfmt::g) * 16 + int32_t(dfmt::u):                                                                     \
        shared_unit<dfmt::g, dfmt::u, B>(a, sh_g, sw, sl, x, sv.sgrid, sv.red);                                        \
        break;
            if constexpr (FAM::iq3) {
                // XB3 C1: the eight pairs the IQ3 layers meet (experts_inst.h)
                switch (a.u_fg * 16 + a.u_fu) {
                    FN_GU_PAIR(iq4_xs, q6_k)
                    FN_GU_PAIR(iq3_s, iq3_s)
                    FN_GU_PAIR(q4_k, iq3_s)
                    FN_GU_PAIR(iq3_s, q4_k)
                    FN_GU_PAIR(q6_k, iq3_s)
                    FN_GU_PAIR(iq4_xs, iq4_xs)
                    FN_GU_PAIR(iq4_xs, q4_k)
                    FN_GU_PAIR(q4_k, iq4_xs)
                    default:
                        break;
                }
            } else {
                switch (a.u_fg * 16 + a.u_fu) {
                    FN_GU_PAIR(iq4_xs, iq3_s)
                    FN_GU_PAIR(q4_k, q4_k)
                    FN_GU_PAIR(iq4_xs, q6_k)
                    FN_GU_PAIR(iq3_s, iq4_xs)
                    FN_GU_PAIR(iq3_s, iq3_s)
                    FN_GU_PAIR(q4_k, iq3_s)
                    FN_GU_PAIR(q4_k, q5_k)
                    FN_GU_PAIR(q5_k, q4_k)
                    FN_GU_PAIR(iq4_xs, q4_k)
                    FN_GU_PAIR(iq4_xs, iq4_xs)
                    default:
                        break;
                }
            }
#undef FN_GU_PAIR
        }
        __syncthreads();
        if (sw < B) {
            glu_out<sh_parts, sh_stride>(a, sv.red + sw * 64 * sh_stride, m.n_slots + sw, sh_g, sl);
        }
    }
}

// ---- SP4: the paired 320-thread mapping at B = 4 and 5 (gate_up_args::map 1) ---------------------------------------
// Thread tid = 4 j + s (j = sub-block 0..79, s = 0..3) owns packed records 8 j + s + 4 p of each routed unit, phase
// p = 0 then 1, so output rows s + 4 p + 8 m: the 640-record packed layout (gu_threads) is unchanged, only the launch's
// threads are. Each record's partial is gate_up_kernel's; XOR 4 then 8 sums the same four sub-blocks in the same tree
// as its XOR 8 then 16, and writer lanes (lane & 15) < 4 publish logical partial 2 w + (lane >> 4), so glu_out still
// folds the 20 partials (gp_parts) in order, stride 21 (tickets SP4/derive.py proves the ownership and the trees).
constexpr int32_t gp_threads = 4 * n_sub; // 320
constexpr int32_t gp_warps = gp_threads / 32;
constexpr int32_t gp_parts = gu_warps; // the ordered partials a channel: 20 logical, not the 10 warps
constexpr int32_t gp_depth = 4;        // 16-byte loads a thread in flight: the 640-thread kernel's 20 KB an SM
static_assert(gp_threads * gp_depth == gu_threads * gu_depth, "the load window's capacity");
static_assert(gp_warps * 2 == gp_parts && 2 * gp_warps == 20, "the shared unit's 20 virtual warps on 10 warps");

// row_out over the unit's active rows, held as compact fragments: slot k is the unit's k-th active row (slot 0 the
// first, always active), roff[k] its partials' offset; the slots past n_act are neither loaded nor read. Each row's
// dots and partial are row_out's.
template <bool SPLIT, int B>
__device__ __forceinline__ void rows_out_p(const row_dec & rd, const uint32_t (&act)[B][8], const float (&da)[B],
                                           const int32_t (&roff)[B], int32_t n_act, float * red_buf, int32_t r,
                                           int32_t part_w, int32_t lane) {
#pragma unroll
    for (int32_t k = 0; k < B; ++k) {
        if (k > 0 && k >= n_act) {
            break;
        }
        dot2 lo = {0, 0}, hi = {0, 0};
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            dot_word(i < 4 || !SPLIT ? lo : hi, rd.g[i], rd.m[i], act[k][i]);
        }
        float part;
        if (SPLIT) {
            part = (rd.dw0 * static_cast<float>(lo.p - 2 * lo.n) + rd.dw1 * static_cast<float>(hi.p - 2 * hi.n)) *
                   da[k];
        } else {
            part = (rd.dw0 * da[k]) * static_cast<float>(lo.p - 2 * lo.n);
        }
        part += __shfl_xor_sync(0xffffffffu, part, 4);
        part += __shfl_xor_sync(0xffffffffu, part, 8);
        if ((lane & 15) < 4) {
            red_buf[roff[k] + r * red_stride + part_w] = part;
        }
    }
}

// Gate, up and GLU at B = 4 or 5 on 320 threads: gate_up_kernel's work and order with the paired mapping above. U: the
// IQ2 body (one symbol a B for the three IQ2 routed formats and their shared pairs, formats from a.u_f, a.u_fg, a.u_fu,
// gate_up_kernel_u's with_f and with_pair); else the triple F, FG, FU (the IQ3 formats). Per routed unit each thread
// loads the unit's active rows' activation fragments of its sub-block once (row_act), and streams its two records'
// chunks (record 8 j + s, then 8 j + s + 4) through a ring of gp_depth slots refilled across records and units. The
// staging covers B x 80 runs and the tables' 16-byte vectors with stride-320 loops; the shared unit's virtual warp v
// runs on warp v % 10, v = w then w + 10. Budget: __launch_bounds__(320, 1); shared memory as the 640-thread kernels'.
template <bool U, fmt F, dfmt FG, dfmt FU, int B>
__global__ void __launch_bounds__(gp_threads, 1) gate_up_kernel_p(gate_up_args a) {
    static_assert(B == 4 || B == 5, "SP4's widths");
    static_assert(U || (F == fmt::iq3_xxs || F == fmt::iq3_s), "the triples outside the IQ2 body: IQ3");
    constexpr int32_t C = cap_of(B);
    const auto with_f = [&](auto && body) {
        if constexpr (!U) {
            body(gu_fc<F>{});
        } else {
            switch (static_cast<fmt>(a.u_f)) {
                case fmt::iq2_xxs:
                    body(gu_fc<fmt::iq2_xxs>{});
                    break;
                case fmt::iq2_xs:
                    body(gu_fc<fmt::iq2_xs>{});
                    break;
                default:
                    body(gu_fc<fmt::iq2_s>{});
                    break;
            }
        }
    };
    extern __shared__ uint4 smem_u4[];
    uint8_t * smem = reinterpret_cast<uint8_t *>(smem_u4);
    constexpr int32_t xs_off = gu_grid_region<U, F, FG, FU>();
    constexpr int32_t xm_off = xs_off + B * x_codes;
    constexpr int32_t red_off = xm_off + B * x_meta;
    constexpr int32_t meta_off = red_off + 2 * B * 64 * red_stride * 4;
    static_assert(xs_off % 16 == 0 && xm_off % 16 == 0 && red_off % 16 == 0 && meta_off % 16 == 0,
                  "16-byte aligned shared regions");
    const gu_smem_view<C> sv = {
        smem,
        reinterpret_cast<uint32_t *>(smem + (U ? gu_u_table : F == fmt::iq3_s ? 0 : gu_layout_of(F).grid_len)),
        smem + xs_off,
        reinterpret_cast<float4 *>(smem + xm_off),
        reinterpret_cast<float *>(smem + red_off),
        reinterpret_cast<meta_c<C> *>(smem + meta_off)};
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t lane = tid & 31;
    const int32_t w = tid >> 5;
    const int32_t j = tid >> 2;
    const int32_t s = tid & 3;
    const uint32_t mz = gridDim.x == n_sm ? 0u : ~0u;

    // the tables' vectors and the rows' runs, stride 320 (at most 640 vectors and 400 runs): loaded before the
    // routing, stored to shared memory after the ring is primed (gate_up_kernel's order)
    constexpr int32_t n_runs = B * n_sub;
    constexpr int32_t n_run_it = (n_runs + gp_threads - 1) / gp_threads;
    const gu_layout gl = gu_layout_of(U ? static_cast<fmt>(a.u_f) : F);
    const bool ex_iq3s = U ? a.u_fg == int32_t(dfmt::iq3_s) || a.u_fu == int32_t(dfmt::iq3_s) : extra_iq3s<F, FG, FU>();
    const int32_t n_grid16 = gl.grid_len / 16;
    const int32_t n_sgrid16 = ex_iq3s ? 2048 / 16 : 0;
    static_assert((gu_u_table + 2048) / 16 <= 2 * gp_threads && (8192 + 2048) / 16 <= 2 * gp_threads,
                  "two 16-byte table loads a thread");
    uint4 grid_v[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
    for (int32_t k = 0; k < 2; ++k) {
        const int32_t i = tid + k * gp_threads;
        if (i < n_grid16) {
            grid_v[k] = __ldg(reinterpret_cast<const uint4 *>(a.grids + gl.grid_off) + i);
        } else if (i < n_grid16 + n_sgrid16) {
            grid_v[k] = __ldg(reinterpret_cast<const uint4 *>(a.grids + iq3s_off) + (i - n_grid16));
        }
    }
    uint4 xc0[n_run_it], xc1[n_run_it];
    float2 xsc[n_run_it];
#pragma unroll
    for (int32_t k = 0; k < n_run_it; ++k) {
        const int32_t run = tid + k * gp_threads;
        // HC3: with X-HMIX, a run in the peer's half is polled after the weight ring is primed (below)
        const bool peer = a.hmix != nullptr && (run % n_sub) / (n_sub / 2) == a.hmix_peer_half;
        xc0[k] = make_uint4(0, 0, 0, 0);
        xc1[k] = make_uint4(0, 0, 0, 0);
        xsc[k] = make_float2(0.0f, 0.0f);
        if (run < n_runs && !peer) {
            const uint4 * src = reinterpret_cast<const uint4 *>(a.x.codes) + 2 * run; // codes [B][2560]: run
            xc0[k] = __ldg(src);
            xc1[k] = __ldg(src + 1);
            xsc[k] = __ldg(reinterpret_cast<const float2 *>(a.x.scales) + run);
        }
    }
    // warp 0: the routing meta and the block's units (gate_up_kernel's)
    uint16_t * list = reinterpret_cast<uint16_t *>(smem + meta_off + gu_meta_bytes<C>);
    if (w == 0) {
        build_meta_of<B>([&](int32_t i) { return __ldg(a.ids + i); }, *sv.m);
        __syncwarp();
        if (sv.m->bad == 0) {
            block_units<C>(*sv.m, a.cw, static_cast<int32_t>(blockIdx.x), list,
                           reinterpret_cast<uint16_t *>(a.meta + meta_order_of(C)));
        }
        if (blockIdx.x == 0) {
            const uint4 * src = reinterpret_cast<const uint4 *>(sv.m);
#pragma unroll 1
            for (int32_t i = lane; i < meta_words_c<C>; i += 32) {
                reinterpret_cast<uint4 *>(a.meta)[i] = src[i];
            }
        }
    }
    __syncthreads();
    const meta_c<C> & m = *sv.m;
    if (m.bad != 0) {
        if (tid == 0 && blockIdx.x == 0 && a.status != nullptr) {
            *a.status = 1u;
        }
        return;
    }
    if (list[gu_max_ranks<C> + 1] == 0) { // no rank of the launch's 10 U + 10 is this block's
        return;
    }
    const int32_t t_first = list[gu_max_ranks<C>] > 0 ? int32_t(list[0]) : -1;
    const int32_t sb = j >> 3;
    // chunk q2 of a unit's sequence of format C::f: record p = q2 / CC (8 j + s + 4 p), its chunk q = q2 % CC (0 = d,
    // then aux when the format has one, then main); the aux chunk is one or two 8-byte loads, every other one 16 bytes
    const auto load_chunk = [&](auto fc, const uint8_t * tl, int32_t q2, uint32_t z) -> uint4 {
        using FC = decltype(fc);
        const int32_t p = q2 / FC::CC, q = q2 - p * FC::CC;
        const int32_t rs = s + 4 * p, rec = 8 * j + rs;
        if (FC::NA == 1 && q == 1) {
            uint4 v = {0, 0, 0, 0};
            if (FC::L.aux8) {
                const uint2 x = __ldcs(reinterpret_cast<const uint2 *>(tl + aux8_off(FC::f) + rec * 8 + z));
                v.x = x.x;
                v.y = x.y;
            }
            if (FC::L.sc8) {
                const uint2 x =
                    __ldcs(reinterpret_cast<const uint2 *>(tl + sc8_off(FC::f) + ((j >> 1) * 8 + rs) * 8 + z));
                v.z = x.x;
                v.w = x.y;
            }
            return v;
        }
        const int64_t off = q == 0 ? d_off(FC::f) + (sb * 8 + rs) * 16 : (int64_t(q - 1 - FC::NA) * gu_threads + rec) * 16;
        return __ldcs(reinterpret_cast<const uint4 *>(tl + off + z));
    };
    const auto tile_of = [&](auto fc, int32_t t) -> const uint8_t * {
        const int32_t u = t / n_group;
        return a.w + (int64_t(m.e[u]) * n_group + (t - u * n_group)) * decltype(fc)::T;
    };

    // prime the ring: the first gp_depth chunks of the block's first unit (a unit is 2 CC >= 10 chunks)
    ring<gp_depth> rg;
    const uint8_t * tile = a.w;
    with_f([&](auto fc) {
        static_assert(2 * decltype(fc)::CC >= gp_depth, "a unit has at least gp_depth chunks");
        if (t_first >= 0) {
            tile = tile_of(fc, t_first);
#pragma unroll
            for (int32_t q = 0; q < gp_depth; ++q) {
                rg.s[q] = load_chunk(fc, tile, q, 0u);
            }
        }
    });
    // X-HMIX's acquire (xchg.h): each of the thread's runs in the peer's half, now that the first chunks are in flight
#pragma unroll
    for (int32_t k = 0; k < n_run_it; ++k) {
        const int32_t run = tid + k * gp_threads;
        if (run < n_runs && a.hmix != nullptr && (run % n_sub) / (n_sub / 2) == a.hmix_peer_half) {
            fn::xchg::hmix_run(a.hmix, a.hmix_peer_half, run / n_sub, run % n_sub, xc0[k], xc1[k], xsc[k], a.err);
        }
    }
#pragma unroll
    for (int32_t k = 0; k < 2; ++k) {
        const int32_t i = tid + k * gp_threads;
        if (i < n_grid16) {
            reinterpret_cast<uint4 *>(sv.grid)[i] = grid_v[k];
        } else if (i < n_grid16 + n_sgrid16) {
            reinterpret_cast<uint4 *>(sv.sgrid)[i - n_grid16] = grid_v[k];
        }
    }
#pragma unroll
    for (int32_t k = 0; k < n_run_it; ++k) {
        const int32_t run = tid + k * gp_threads;
        if (run < n_runs) {
            const int32_t b = run / n_sub, jj = run - b * n_sub;
            int32_t slo = 0, shi = 0;
#pragma unroll
            for (int32_t t = 0; t < 4; ++t) {
                slo = fn::dense::dp4a(word_of(xc0[k], t), 0x01010101u, slo);
                shi = fn::dense::dp4a(word_of(xc1[k], t), 0x01010101u, shi);
            }
            uint4 * xs = reinterpret_cast<uint4 *>(sv.xs + b * x_codes);
            xs[fn::dense::swz(2 * jj)] = xc0[k];
            xs[fn::dense::swz(2 * jj + 1)] = xc1[k];
            sv.xm[run] = make_float4(xsc[k].x, xsc[k].y, __int_as_float(slo), __int_as_float(shi)); // the sums' bits
        }
    }
    __syncthreads();

    if (t_first >= 0) {
        const volatile uint16_t * lv = list;
        const int32_t n_mine = list[gu_max_ranks<C>];
        const auto unit_at = [&](int32_t i) -> int32_t { return i < n_mine ? int32_t(lv[i]) : -1; };
        // row b's codes and d of sub-block j: the four lanes of a sub-block share it (a broadcast)
        const auto row_act = [&](int32_t b, uint32_t (&o)[8], float & d) {
            const uint4 * xb = reinterpret_cast<const uint4 *>(sv.xs + b * x_codes);
            const uint4 c0 = xb[fn::dense::swz(2 * j)], c1 = xb[fn::dense::swz(2 * j + 1)];
            o[0] = c0.x;
            o[1] = c0.y;
            o[2] = c0.z;
            o[3] = c0.w;
            o[4] = c1.x;
            o[5] = c1.y;
            o[6] = c1.z;
            o[7] = c1.w;
            d = sv.xm[b * n_sub + j].x;
        };
        const int32_t part_w = 2 * w + (lane >> 4); // the logical partial a writer lane publishes

        with_f([&](auto fc) {
            using FC = decltype(fc);
            constexpr int32_t CC = FC::CC;
            constexpr int32_t QQ = 2 * CC; // a unit's chunks: two records
            constexpr bool SPLIT = FC::f == fmt::iq2_xs || FC::f == fmt::iq2_s;
            // The block's routed unit i, then its next one (none: -1); the ring slot of chunk q2 is (P + q2) %
            // gp_depth, a constant, and the last refills target the next unit (or this tile again after the last)
            const auto run_unit = [&](auto phase, int32_t i, int32_t buf) {
                constexpr int32_t P = decltype(phase)::value;
                const uint32_t mask = m.mask[unit_at(i) / n_group];
                const int32_t tn = unit_at(i + 1);
                const uint8_t * tile_next = tn >= 0 ? tile_of(fc, tn) : tile;
                float * red_buf = sv.red + buf * B * 64 * red_stride;

                // the unit's active rows' fragments, once: slot k the k-th active row
                const int32_t n_act = __popc(mask);
                uint32_t act[B][8];
                float da[B];
                int32_t roff[B];
                uint32_t mm = mask;
#pragma unroll
                for (int32_t k = 0; k < B; ++k) {
                    if (k > 0 && k >= n_act) {
                        break;
                    }
                    const int32_t b = __ffs(static_cast<int32_t>(mm)) - 1;
                    mm &= mm - 1u;
                    row_act(b, act[k], da[k]);
                    roff[k] = b * 64 * red_stride;
                }
                const auto out = [&](const row_dec & rd, int32_t r) {
                    rows_out_p<SPLIT, B>(rd, act, da, roff, n_act, red_buf, r, part_w, lane);
                };

                float dh8[8];
                [[maybe_unused]] bytes8 ax8 = {0, 0}, sc8 = {0, 0};
                [[maybe_unused]] uint4 held = {0, 0, 0, 0}; // IQ3: the sign words waiting
#pragma unroll
                for (int32_t q2 = 0; q2 < QQ; ++q2) {
                    const int32_t sl = (P + q2) % gp_depth;
                    const uint4 v = rg.s[sl];
                    const int32_t qn = q2 + gp_depth;
                    const uint32_t z = order_after(v.x, mz);
                    rg.s[sl] = qn < QQ ? load_chunk(fc, tile, qn, z) : load_chunk(fc, tile_next, qn - QQ, z);
                    const int32_t p = q2 / CC, q = q2 - p * CC;
                    const int32_t rs = s + 4 * p;
                    if (q == 0) {
#pragma unroll
                        for (int32_t mr = 0; mr < 8; ++mr) {
                            dh8[mr] = (mr & 1) ? half_hi(word_of(v, mr >> 1)) : half_lo(word_of(v, mr >> 1));
                        }
                        continue;
                    }
                    if (FC::NA == 1 && q == 1) {
                        ax8 = take8(v.x, v.y);
                        sc8 = take8(v.z, v.w);
                        continue;
                    }
                    const int32_t c = q - 1 - FC::NA;
                    row_dec rd;
                    if constexpr (FC::f == fmt::iq2_xxs || FC::f == fmt::iq2_xs) {
                        const uint2 * grid = reinterpret_cast<const uint2 *>(sv.grid);
#pragma unroll
                        for (int32_t h = 0; h < 2; ++h) {
                            const int32_t mrow = 2 * c + h;
                            const uint2 qq = h == 0 ? make_uint2(v.x, v.y) : make_uint2(v.z, v.w);
                            if constexpr (FC::f == fmt::iq2_xxs) {
                                dec_iq2_xxs(rd, qq, dh8[mrow], grid);
                            } else {
                                dec_iq2_xs(rd, qq, byte8(ax8, mrow), dh8[mrow], grid);
                            }
                            out(rd, rs + 8 * mrow);
                        }
                    } else if constexpr (FC::f == fmt::iq2_s) {
                        if (c == 0) {
                            ax8 = take8(v.x, v.y); // qh of rows 0..7
                            sc8 = take8(v.z, v.w); // scales of rows 0..7
                            continue;
                        }
                        const uint2 * grid = reinterpret_cast<const uint2 *>(sv.grid);
#pragma unroll
                        for (int32_t h = 0; h < 2; ++h) {
                            const int32_t mrow = 2 * (c - 1) + h;
                            dec_iq2_s(rd, h == 0 ? v.x : v.z, h == 0 ? v.y : v.w, byte8(ax8, mrow), byte8(sc8, mrow),
                                      dh8[mrow], grid);
                            out(rd, rs + 8 * mrow);
                        }
                    } else {
                        // IQ3: chunk k of row quad hq: k 0 [q r0][sg r0][sg r1], k 1 [q r1][sg r2][sg r3],
                        // k 2 [q r2][q r3]
                        const uint32_t * grid = reinterpret_cast<const uint32_t *>(sv.grid);
                        const int32_t hq = c / 3;
                        const int32_t k = c - 3 * hq;
                        const auto row = [&](int32_t mrow, uint2 qq, uint32_t sg) {
                            if constexpr (FC::f == fmt::iq3_xxs) {
                                dec_iq3_xxs(rd, qq, sg, dh8[mrow], grid);
                            } else {
                                const uint32_t scb = byte8(sc8, mrow);
                                dec_iq3_s(rd, qq, sg, byte8(ax8, mrow), (j & 1) ? scb >> 4 : scb & 0xfu, dh8[mrow],
                                          grid);
                            }
                            out(rd, rs + 8 * mrow);
                        };
                        if (k == 0) {
                            held.x = v.w;
                            row(4 * hq + 0, make_uint2(v.x, v.y), v.z);
                        } else if (k == 1) {
                            held.y = v.z;
                            held.z = v.w;
                            row(4 * hq + 1, make_uint2(v.x, v.y), held.x);
                        } else {
                            row(4 * hq + 2, make_uint2(v.x, v.y), held.y);
                            row(4 * hq + 3, make_uint2(v.z, v.w), held.z);
                        }
                    }
                }
                tile = tile_next;
                __syncthreads();
                // epilogue of the unit: warp b (a routed row) sums its 20 logical partials per channel, GLU, format
                if (w < B && ((mask >> w) & 1u) != 0) {
                    const int32_t t = unit_at(i);
                    const int32_t u = t / n_group;
                    glu_out<gp_parts, red_stride>(a, red_buf + w * 64 * red_stride, slot_of(m, u, w),
                                                  t - u * n_group, lane);
                }
            };
            // a unit is 2 CC chunks: when that is a multiple of gp_depth every unit starts at phase 0 (one copy of
            // the body), else two units a trip, one copy a phase
            static_assert((2 * QQ) % gp_depth == 0, "the phase repeats every two units");
            if constexpr (QQ % gp_depth == 0) {
#pragma unroll 1
                for (int32_t i = 0; i < n_mine; ++i) {
                    run_unit(std::integral_constant<int32_t, 0>{}, i, i & 1);
                }
            } else {
#pragma unroll 1
                for (int32_t i = 0; i < n_mine; i += 2) {
                    run_unit(std::integral_constant<int32_t, 0>{}, i, 0);
                    if (i + 1 < n_mine) {
                        run_unit(std::integral_constant<int32_t, QQ % gp_depth>{}, i + 1, 1);
                    }
                }
            }
        });
    }
    // the block's shared ranks after its routed units (gate_up_kernel's); virtual warp v of shared_unit on warp v % 10,
    // v = w then w + 10, one item set live at a time. The IQ2 body's pair is a direct switch: through a generic lambda
    // (gate_up_kernel_u's with_pair) ptxas outlined it at B = 4 and 5, a call with a 312-byte stack frame
#pragma unroll 1
    for (int32_t i = list[gu_max_ranks<C>]; i < list[gu_max_ranks<C> + 1]; ++i) {
        const int32_t sh_g = list[i] - order_shared_of(C);
        if (i > 0) {
            __syncthreads();
        }
        const fn::dense::act x = {sv.xs, sv.xm, D};
        const int32_t tx = opaque(static_cast<int32_t>(threadIdx.x));
        const int32_t sw = tx >> 5, sl = tx & 31;
#pragma unroll 1
        for (int32_t h = 0; h < 2; ++h) {
            const int32_t v = sw + gp_warps * h;
            if constexpr (!U) {
                shared_unit<FG, FU, B>(a, sh_g, v, sl, x, sv.sgrid, sv.red);
            } else {
#define FN_GU_PAIR(g, u)                                                                                               \
    case int32_t(dfmt::g) * 16 + int32_t(dfmt::u):                                                                     \
        shared_unit<dfmt::g, dfmt::u, B>(a, sh_g, v, sl, x, sv.sgrid, sv.red);                                         \
        break;
                switch (a.u_fg * 16 + a.u_fu) {
                    FN_GU_PAIR(iq4_xs, iq3_s)
                    FN_GU_PAIR(q4_k, q4_k)
                    FN_GU_PAIR(iq4_xs, q6_k)
                    FN_GU_PAIR(iq3_s, iq4_xs)
                    FN_GU_PAIR(iq3_s, iq3_s)
                    FN_GU_PAIR(q4_k, iq3_s)
                    FN_GU_PAIR(q4_k, q5_k)
                    FN_GU_PAIR(q5_k, q4_k)
                    FN_GU_PAIR(iq4_xs, q4_k)
                    FN_GU_PAIR(iq4_xs, iq4_xs)
                    default:
                        break;
                }
#undef FN_GU_PAIR
            }
        }
        __syncthreads();
        if (sw < B) {
            glu_out<sh_parts, sh_stride>(a, sv.red + sw * 64 * sh_stride, m.n_slots + sw, sh_g, sl);
        }
    }
}

template <bool U, fmt F, dfmt FG, dfmt FU, int B> bool launch_gate_up_p(const gate_up_args & a, cudaStream_t st) {
    constexpr int32_t smem = U ? gu_u_smem_bytes<B>() : gu_smem_bytes<F, FG, FU, B>();
    static_assert(smem <= 96 * 1024, "one block's shared memory");
    static std::atomic<uint32_t> attr_done{0u};
    const void * k = reinterpret_cast<const void *>(gate_up_kernel_p<U, F, FG, FU, B>);
    if (!detail::smem_attr_once(attr_done, k, smem)) {
        return false;
    }
    gate_up_kernel_p<U, F, FG, FU, B><<<n_sm, gp_threads, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <fmt F, dfmt FG, dfmt FU, int B> bool launch_gate_up(const gate_up_args & a, cudaStream_t st) {
    constexpr int32_t smem = gu_smem_bytes<F, FG, FU, B>();
    static_assert(smem <= 96 * 1024, "one block's shared memory");
    static std::atomic<uint32_t> attr_done{0u};
    if (!detail::smem_attr_once(attr_done, reinterpret_cast<const void *>(gate_up_kernel<F, FG, FU, B, gu_depth>),
                                smem)) {
        return false;
    }
    gate_up_kernel<F, FG, FU, B, gu_depth><<<n_sm, gu_threads, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <fmt F, dfmt FG, dfmt FU> bool gate_up_rows(int32_t rows, const gate_up_args & a, cudaStream_t st) {
    switch (rows) {
        case 1:
            return launch_gate_up<F, FG, FU, 1>(a, st);
        case 2:
            return launch_gate_up<F, FG, FU, 2>(a, st);
        case 3:
            return launch_gate_up<F, FG, FU, 3>(a, st);
        case 4:
            return launch_gate_up<F, FG, FU, 4>(a, st);
        case 5: // SP2: the target's verification rows; the draft's Q8_0 layer runs B <= 4 (catch-up has no FFN)
            if constexpr (F != fmt::q8_0) {
                return launch_gate_up<F, FG, FU, 5>(a, st);
            }
            return false;
        default:
            return false;
    }
}

// XB3 part 6: the IQ2 body at B, its formats set in its arguments; XB3 C1: the IQ3 body (FAM gu_iq3) at B = 4 and 5
template <int B, typename FAM = gu_iq2> bool launch_gate_up_u(const gate_up_args & a, cudaStream_t st) {
    constexpr int32_t smem = gu_u_smem_bytes<B, FAM>();
    static_assert(smem <= 96 * 1024, "one block's shared memory");
    static std::atomic<uint32_t> attr_done{0u};
    if (!detail::smem_attr_once(attr_done, reinterpret_cast<const void *>(gate_up_kernel_u<FAM, B, gu_depth>), smem)) {
        return false;
    }
    gate_up_kernel_u<FAM, B, gu_depth><<<n_sm, gu_threads, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

constexpr bool is_iq2(fmt f) {
    return f == fmt::iq2_xxs || f == fmt::iq2_xs || f == fmt::iq2_s;
}

// routed format F's deployed triples (experts_inst.h); false for any other (fg, fu). The IQ2 formats launch the IQ2
// body (XB3 part 6) with the triple in its arguments at B = 1-3; at B = 4 the body (three formats' loops in one
// function at the 96-register cap) spilled 344 bytes, so B = 4 keeps the per-triple instantiations, and so does B = 5
// (SP2). SP4 H2 (map 2): the IQ2 body at B = 4 and 5 with the compact optional-row loop; the IQ3 and Q8_0 formats keep
// SP2's kernels under map 2. XB3 C1 (map 3): the IQ2 formats as map 2, the IQ3 formats' B = 4 and 5 on the IQ3 body.
template <fmt F> bool gate_up_for(dfmt fg, dfmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    // SP4 (map 1): B = 4 and 5 of the target formats on the paired 320-thread kernel, the IQ2 formats in its IQ2 body
    if constexpr (F != fmt::q8_0) {
        if (rows >= 4 && a.map == 1) {
            bool deployed = false;
#define FN_EXPERTS_GU_CASE(rf, sgf, suf, layer)                                                                        \
    deployed = deployed || (F == fmt::rf && fg == dfmt::sgf && fu == dfmt::suf);
            FN_EXPERTS_GATE_UP_INSTANCES(FN_EXPERTS_GU_CASE)
#undef FN_EXPERTS_GU_CASE
            if (!deployed) {
                return false;
            }
            if constexpr (is_iq2(F)) {
                gate_up_args u = a;
                u.u_f = int32_t(F);
                u.u_fg = int32_t(fg);
                u.u_fu = int32_t(fu);
                constexpr fmt PF = fmt::iq2_xxs; // the IQ2 body's placeholders
                constexpr dfmt PD = dfmt::q4_k;
                return rows == 4 ? launch_gate_up_p<true, PF, PD, PD, 4>(u, st)
                                 : launch_gate_up_p<true, PF, PD, PD, 5>(u, st);
            } else {
#define FN_EXPERTS_GU_CASE(rf, sgf, suf, layer)                                                                        \
    if constexpr (F == fmt::rf) {                                                                                      \
        if (fg == dfmt::sgf && fu == dfmt::suf) {                                                                      \
            return rows == 4 ? launch_gate_up_p<false, fmt::rf, dfmt::sgf, dfmt::suf, 4>(a, st)                        \
                             : launch_gate_up_p<false, fmt::rf, dfmt::sgf, dfmt::suf, 5>(a, st);                       \
        }                                                                                                              \
    }
                FN_EXPERTS_GATE_UP_INSTANCES(FN_EXPERTS_GU_CASE)
#undef FN_EXPERTS_GU_CASE
                return false;
            }
        }
    }
    if constexpr (is_iq2(F)) {
        if (rows >= 4 && a.map != 2 && a.map != 3) { // SP2: B = 5 as B = 4, the per-triple instantiations
#define FN_EXPERTS_GU_CASE(rf, sgf, suf, layer)                                                                        \
    if constexpr (F == fmt::rf) {                                                                                      \
        if (fg == dfmt::sgf && fu == dfmt::suf) {                                                                      \
            return rows == 4 ? launch_gate_up<fmt::rf, dfmt::sgf, dfmt::suf, 4>(a, st)                                 \
                             : launch_gate_up<fmt::rf, dfmt::sgf, dfmt::suf, 5>(a, st);                                \
        }                                                                                                              \
    }
            FN_EXPERTS_GATE_UP_INSTANCES(FN_EXPERTS_GU_CASE)
#undef FN_EXPERTS_GU_CASE
            return false;
        }
        bool deployed = false;
#define FN_EXPERTS_GU_CASE(rf, sgf, suf, layer)                                                                        \
    deployed = deployed || (F == fmt::rf && fg == dfmt::sgf && fu == dfmt::suf);
        FN_EXPERTS_GATE_UP_INSTANCES(FN_EXPERTS_GU_CASE)
#undef FN_EXPERTS_GU_CASE
        if (!deployed) {
            return false;
        }
        gate_up_args u = a;
        u.u_f = int32_t(F);
        u.u_fg = int32_t(fg);
        u.u_fu = int32_t(fu);
        switch (rows) {
            case 1:
                return launch_gate_up_u<1>(u, st);
            case 2:
                return launch_gate_up_u<2>(u, st);
            case 3:
                return launch_gate_up_u<3>(u, st);
            case 4: // SP4 H2 (map 2): the IQ2 body at B = 4 and 5
                return launch_gate_up_u<4>(u, st);
            case 5:
                return launch_gate_up_u<5>(u, st);
            default:
                return false;
        }
    } else {
        if constexpr (F == fmt::iq3_xxs || F == fmt::iq3_s) {
            if (rows >= 4 && a.map == 3) { // XB3 C1: the IQ3 body, the triple in its arguments
                bool deployed = false;
#define FN_EXPERTS_GU_CASE(rf, sgf, suf, layer)                                                                        \
    deployed = deployed || (F == fmt::rf && fg == dfmt::sgf && fu == dfmt::suf);
                FN_EXPERTS_GATE_UP_INSTANCES(FN_EXPERTS_GU_CASE)
#undef FN_EXPERTS_GU_CASE
                if (!deployed) {
                    return false;
                }
                gate_up_args u = a;
                u.u_f = int32_t(F);
                u.u_fg = int32_t(fg);
                u.u_fu = int32_t(fu);
                return rows == 4 ? launch_gate_up_u<4, gu_iq3>(u, st) : launch_gate_up_u<5, gu_iq3>(u, st);
            }
        }
#define FN_EXPERTS_GU_CASE(rf, sgf, suf, layer)                                                                        \
    if constexpr (F == fmt::rf) {                                                                                      \
        if (fg == dfmt::sgf && fu == dfmt::suf) {                                                                      \
            return gate_up_rows<fmt::rf, dfmt::sgf, dfmt::suf>(rows, a, st);                                           \
        }                                                                                                              \
    }
        FN_EXPERTS_GATE_UP_INSTANCES(FN_EXPERTS_GU_CASE)
#undef FN_EXPERTS_GU_CASE
        return false;
    }
}

} // namespace
} // namespace fn::experts
