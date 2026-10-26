#include "fn/kernels/experts/experts.h"

#include "fn/kernels/experts/experts_dev.cuh"
#include "fn/kernels/experts/experts_inst.h"
#include "fn/xchg/xchg.cuh"
#include "fn/xchg/xchg.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <type_traits>

namespace fn::experts {
namespace {

// ---- down -------------------------------------------------------------------------------------------------------
constexpr int32_t act_stride = 336;    // bytes per staged slot: 320 + 16, 21 x 16 B (odd), so two slots never share a
                                       // bank group in one 16-byte read
constexpr int32_t max_rows_block = 36; // ceil(2560 / 72)
constexpr int32_t dn_threads = 544;    // 17 warps: three groups of 5 R <= 180 item threads (XB4)
constexpr int32_t dn_depth_q2 = 3;     // Q2_0 items in flight a thread: 540 x 3 x 18 B = 28 KB an SM
constexpr int32_t dn_depth_nl = 2;     // IQ4_NL: 540 x 2 x 36 B = 38 KB an SM; Q8_0 (DR1): 512 x 2 x 68 B = 68 KB an SM
// Q8_0 down (the draft's layer) keeps 71b92a9's thread-strided walk at 512 threads: at 544 the cap is 96 registers
// (5 warps on one of the SM's four 16K-register partitions), and K5 spills 48 bytes there (XB4)
constexpr int32_t dn_threads_strided = 512;
template <fmt F> constexpr int32_t dn_threads_of() {
    return F == fmt::q8_0 ? dn_threads_strided : dn_threads;
}

// SP2: C, the routing capacity (cap_of(B)): capacity 4 is the layout B <= 4 had before SP2
template <int C> struct dn_smem {
    static constexpr int32_t gs = glu_slots_c<C>;
    uint8_t raw[glu_codes_c<C> + glu_scales_c<C>]; // the GLU slots as gate_up wrote them, staged once (R3): codes,
                                                   // then {d, s}
    uint8_t act[gs * act_stride];
    float da[gs][10];
    int32_t asum[gs][10];
    float part[gs][n_item_k][max_rows_block];
    float wts[meta_c<C>::slots]; // the router weights [B][10], read in the final sum
    float sg[C];                 // the shared expert's weights [B]
    meta_c<C> m;
};
static_assert(sizeof(dn_smem<4>) == 68416 && sizeof(dn_smem<5>) <= 96 * 1024, "down's shared memory [C]");

// 16 int8 codes of a half run, transposed for Q2_0's 2-bit fields: word s holds elements s, s + 4, s + 8, s + 12
__device__ __forceinline__ uint4 transpose16(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3) {
    const uint32_t a01 = __byte_perm(w0, w1, 0x5140), b01 = __byte_perm(w0, w1, 0x7362);
    const uint32_t a23 = __byte_perm(w2, w3, 0x5140), b23 = __byte_perm(w2, w3, 0x7362);
    // a01 = (e0, e4, e1, e5), b01 = (e2, e6, e3, e7), a23 = (e8, e12, e9, e13), b23 = (e10, e14, e11, e15)
    return make_uint4(__byte_perm(a01, a23, 0x5410), __byte_perm(a01, a23, 0x7632), __byte_perm(b01, b23, 0x5410),
                      __byte_perm(b01, b23, 0x7632));
}

// kvalues_iq4nl of the 8 nibbles of q (ggml's get_int_from_table_16): lo = the low nibbles' values, hi = the high
__device__ __forceinline__ void table16(uint32_t q, const uint32_t (&t)[4], uint32_t & lo, uint32_t & hi) {
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const uint32_t l0 = __byte_perm(t[0], t[1], q), h0 = __byte_perm(t[2], t[3], q);
    const uint32_t l1 = __byte_perm(t[0], t[1], q >> 16), h1 = __byte_perm(t[2], t[3], q >> 16);
    const uint32_t x0 = __byte_perm(l0, h0, sel), x1 = __byte_perm(l1, h1, sel >> 16);
    lo = __byte_perm(x0, x1, 0x6420);
    hi = __byte_perm(x0, x1, 0x7531);
}

template <fmt F> struct dn_slot {
    uint4 c[down_chunks(F)];
    // FP16 d (Q2_0: low half; IQ4_NL and Q8_0: both blocks); ME1, Q5_1: the blocks' qh words, then d0 m0, d1 m1
    std::conditional_t<F == fmt::q5_1, uint4, uint32_t> d;
};

template <fmt F> __host__ __device__ constexpr int64_t dn_expert_bytes() {
    return F == fmt::q2_0 ? 230400 : F == fmt::q8_0 ? 870400 : F == fmt::q5_1 ? 614400 : 460800;
}
static_assert(dn_expert_bytes<fmt::q8_0>() == int64_t(n_item_k) * (down_chunks(fmt::q8_0) * 16 + 4) * D,
              "Q8_0 down bytes");
static_assert(dn_expert_bytes<fmt::q5_1>() == int64_t(n_item_k) * (down_chunks(fmt::q5_1) * 16 + 16) * D,
              "Q5_1 down bytes (GGUF's: 24 bytes a 32 weights)");

// item (j5, n) of one expert's packed down: the byte offsets of its chunk-0 codes and of its d from the expert's first
// byte (at most 870,400)
template <fmt F> __device__ __forceinline__ int32_t dn_code_off(int32_t j5, int32_t n) {
    return (j5 * down_chunks(F) * D + n) * 16;
}
template <fmt F> __device__ __forceinline__ int32_t dn_d_off(int32_t j5, int32_t n) {
    constexpr int32_t d_plane = n_item_k * down_chunks(F) * D * 16;
    return d_plane + (j5 * D + n) * down_d_bytes(F);
}

// an item's codes at pc (chunk c at pc + c x D x 16, an immediate offset) and its d at pd
template <fmt F> __device__ __forceinline__ dn_slot<F> dn_load(const uint8_t * pc, const uint8_t * pd, uint32_t z) {
    dn_slot<F> v;
#pragma unroll
    for (int32_t c = 0; c < down_chunks(F); ++c) {
        v.c[c] = __ldcs(reinterpret_cast<const uint4 *>(pc + c * D * 16 + z));
    }
    if constexpr (F == fmt::q5_1) {
        v.d = __ldcs(reinterpret_cast<const uint4 *>(pd + z));
    } else if (F == fmt::q2_0) {
        v.d = __ldcs(reinterpret_cast<const unsigned short *>(pd + z));
    } else {
        v.d = __ldcs(reinterpret_cast<const uint32_t *>(pd + z));
    }
    return v;
}

// FMS4 item 3 (the ticket's fallback form): an item's rows, in mask order (their slots consecutive from slot; the row
// body reads only the slot): the first unguarded, since a routed unit always has a row, then rows 1..B-1 each guarded
// by the row count. B - 1 guards instead of B, and no guard at B = 1. ALL: the shared expert's call, every row (a
// constant mask), unrolled as before.
template <int B, bool ALL, typename R> __device__ __forceinline__ void dn_rows(uint32_t mask, int32_t slot, R && row) {
    if constexpr (ALL) {
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            row(slot + b);
        }
    } else {
        const int32_t nr = __popc(mask);
        row(slot);
#pragma unroll
        for (int32_t k = 1; k < B; ++k) {
            if (k < nr) {
                row(slot + k);
            }
        }
    }
}

// One item: output row n's dot with columns 64 j5 .. 64 j5 + 63 of one expert, for each row b in mask (their slots
// consecutive from slot), into part[slot][j5][n]. ALL: mask is every row's (the shared expert).
template <fmt F, int B, bool ALL = false, typename S>
__device__ __forceinline__ void dn_item(S & s, const dn_slot<F> & v, int32_t j5, int32_t n, uint32_t mask,
                                        int32_t slot, const uint32_t (&kv)[4]) {
    const int32_t kb = 2 * j5; // the 32-value runs of this item
    if constexpr (F == fmt::q2_0) {
        uint32_t x[16]; // x[4 i + s] = 2-bit field s of each byte of word i
#pragma unroll
        for (int32_t wi = 0; wi < 4; ++wi) {
            const uint32_t wv = word_of(v.c[0], wi);
#pragma unroll
            for (int32_t sh = 0; sh < 4; ++sh) {
                x[4 * wi + sh] = (wv >> (2 * sh)) & 0x03030303u;
            }
        }
        const float dw = half_lo(v.d);
        dn_rows<B, ALL>(mask, slot, [&](int32_t slot) {
            const uint4 * ap = reinterpret_cast<const uint4 *>(s.act + slot * act_stride + j5 * 64);
            int32_t i0 = 0, i1 = 0;
#pragma unroll
            for (int32_t wi = 0; wi < 4; ++wi) {
                const uint4 av = ap[wi];
                int32_t & acc = wi < 2 ? i0 : i1;
                acc = __dp4a(static_cast<int32_t>(x[4 * wi + 0]), static_cast<int32_t>(av.x), acc);
                acc = __dp4a(static_cast<int32_t>(x[4 * wi + 1]), static_cast<int32_t>(av.y), acc);
                acc = __dp4a(static_cast<int32_t>(x[4 * wi + 2]), static_cast<int32_t>(av.z), acc);
                acc = __dp4a(static_cast<int32_t>(x[4 * wi + 3]), static_cast<int32_t>(av.w), acc);
            }
            const float dot = dw * (s.da[slot][kb] * static_cast<float>(i0 - s.asum[slot][kb]) +
                                    s.da[slot][kb + 1] * static_cast<float>(i1 - s.asum[slot][kb + 1]));
            s.part[slot][j5][n] = dot;
        });
    } else if constexpr (F == fmt::q5_1) {
        // ME1: block kb + c's weights 0..15 (lo) and 16..31 (hi), 0..31 with bit 4 from the block's qh word; row b's
        // partial is (d_c da_c) is_c + (m_c da_c) asum_c a block (asum: the run's exact code sum), one fused rounding
        // each, then the two blocks' sum
        uint32_t lo[8], hi[8];
#pragma unroll
        for (int32_t c = 0; c < 2; ++c) {
            const uint32_t qh = c == 0 ? v.d.x : v.d.y;
#pragma unroll
            for (int32_t wi = 0; wi < 4; ++wi) {
                const uint32_t x = word_of(v.c[c], wi);
                lo[4 * c + wi] = (x & 0x0f0f0f0fu) | (spread4(qh >> (4 * wi)) << 4);
                hi[4 * c + wi] = ((x >> 4) & 0x0f0f0f0fu) | (spread4(qh >> (16 + 4 * wi)) << 4);
            }
        }
        const float d0 = half_lo(v.d.z), m0 = half_hi(v.d.z), d1 = half_lo(v.d.w), m1 = half_hi(v.d.w);
        dn_rows<B, ALL>(mask, slot, [&](int32_t slot) {
            const uint4 * ap = reinterpret_cast<const uint4 *>(s.act + slot * act_stride + j5 * 64);
            int32_t is[2] = {0, 0};
#pragma unroll
            for (int32_t c = 0; c < 2; ++c) {
                const uint4 a0 = ap[2 * c], a1 = ap[2 * c + 1]; // elements 0..15, 16..31 of block kb + c
#pragma unroll
                for (int32_t wi = 0; wi < 4; ++wi) {
                    is[c] = __dp4a(static_cast<int32_t>(lo[4 * c + wi]), static_cast<int32_t>(word_of(a0, wi)), is[c]);
                    is[c] = __dp4a(static_cast<int32_t>(hi[4 * c + wi]), static_cast<int32_t>(word_of(a1, wi)), is[c]);
                }
            }
            const float da0 = s.da[slot][kb], da1 = s.da[slot][kb + 1];
            const float t0 = __fmaf_rn(__fmul_rn(d0, da0), static_cast<float>(is[0]),
                                       __fmul_rn(__fmul_rn(m0, da0), static_cast<float>(s.asum[slot][kb])));
            const float t1 = __fmaf_rn(__fmul_rn(d1, da1), static_cast<float>(is[1]),
                                       __fmul_rn(__fmul_rn(m1, da1), static_cast<float>(s.asum[slot][kb + 1])));
            s.part[slot][j5][n] = __fadd_rn(t0, t1);
        });
    } else {
        // block kb + c's weights 0..15 (lo) and 16..31 (hi): IQ4_NL's table values, Q8_0's own int8 (DR1)
        uint32_t lo[8], hi[8];
#pragma unroll
        for (int32_t c = 0; c < 2; ++c) {
#pragma unroll
            for (int32_t wi = 0; wi < 4; ++wi) {
                if constexpr (F == fmt::q8_0) {
                    lo[4 * c + wi] = word_of(v.c[2 * c], wi);
                    hi[4 * c + wi] = word_of(v.c[2 * c + 1], wi);
                } else {
                    table16(word_of(v.c[c], wi), kv, lo[4 * c + wi], hi[4 * c + wi]);
                }
            }
        }
        const float d0 = half_lo(v.d), d1 = half_hi(v.d);
        dn_rows<B, ALL>(mask, slot, [&](int32_t slot) {
            const uint4 * ap = reinterpret_cast<const uint4 *>(s.act + slot * act_stride + j5 * 64);
            int32_t is[2] = {0, 0};
#pragma unroll
            for (int32_t c = 0; c < 2; ++c) {
                const uint4 a0 = ap[2 * c], a1 = ap[2 * c + 1]; // elements 0..15, 16..31 of block kb + c
#pragma unroll
                for (int32_t wi = 0; wi < 4; ++wi) {
                    is[c] = __dp4a(static_cast<int32_t>(lo[4 * c + wi]), static_cast<int32_t>(word_of(a0, wi)), is[c]);
                    is[c] = __dp4a(static_cast<int32_t>(hi[4 * c + wi]), static_cast<int32_t>(word_of(a1, wi)), is[c]);
                }
            }
            const float dot =
                d0 * s.da[slot][kb] * static_cast<float>(is[0]) + d1 * s.da[slot][kb + 1] * static_cast<float>(is[1]);
            s.part[slot][j5][n] = dot;
        });
    }
}

// The routed items' walk at dn_threads (K5): thread r of group g = tid / 5 R keeps item (j5, n) = (r / R, r % R) and
// runs units g, g + 3, ..., its address the thread's fixed code and d offsets plus e[u] x the expert's bytes; the
// threads past the three groups run none. The shared expert's item (j5, n) goes to group n_u % 3, one with the fewest
// routed units.
template <fmt F, int DEPTH, int C> struct dn_fixed_walk {
    const meta_c<C> & m;
    int32_t g, sh_j5, sh_n, n_mine, u_last;
    const uint8_t * pc;
    const uint8_t * pd;
    bool has_sh;
    __device__ __forceinline__ dn_fixed_walk(const down_args & a, const meta_c<C> & m_, int32_t n0, int32_t R,
                                             int32_t tid)
        : m(m_) {
        const int32_t per_u = n_item_k * R;
        g = tid / per_u; // 3: past the groups
        const int32_t r = tid - g * per_u;
        sh_j5 = r / R;
        sh_n = r - sh_j5 * R;
        n_mine = g < 3 ? (m.n_u - g + 2) / 3 : 0;
        u_last = g + 3 * (n_mine - 1);
        pc = a.w + dn_code_off<F>(sh_j5, n0 + sh_n);
        pd = a.w + dn_d_off<F>(sh_j5, n0 + sh_n);
        has_sh = g == m.n_u % 3;
    }
    __device__ __forceinline__ dn_slot<F> load(int32_t u, uint32_t z) const {
        const int64_t eo = int64_t(m.e[u]) * dn_expert_bytes<F>();
        return dn_load<F>(pc + eo, pd + eo, z);
    }
    // the ring's first DEPTH units; a load past the last unit reads the last unit again (never predicated, so never a
    // select of data in flight)
    __device__ __forceinline__ void prime(dn_slot<F> (&rg)[DEPTH]) const {
        if (n_mine > 0) {
#pragma unroll
            for (int32_t q = 0; q < DEPTH; ++q) {
                rg[q] = load(min(g + 3 * q, u_last), 0u);
            }
        }
    }
    // DEPTH items a trip, so the ring slot of each is a constant (EX1 v1: moving slots waited for their loads)
    template <int B>
    __device__ __forceinline__ void run(dn_smem<C> & s, dn_slot<F> (&rg)[DEPTH], const uint32_t (&kv)[4],
                                        uint32_t mz) const {
        int32_t ul = g + 3 * DEPTH; // next unit to load
        for (int32_t k0 = 0; k0 < n_mine; k0 += DEPTH) {
#pragma unroll
            for (int32_t q = 0; q < DEPTH; ++q) {
                if (k0 + q >= n_mine) {
                    break;
                }
                const int32_t u = g + 3 * (k0 + q);
                const dn_slot<F> v = rg[q];
                rg[q] = load(min(ul, u_last), order_after(v.c[0].x, mz));
                ul += 3;
                dn_item<F, B>(s, v, sh_j5, sh_n, m.mask[u], m.slot_base[u], kv);
            }
        }
    }
};

// The routed items' walk at dn_threads_strided (71b92a9's, for Q8_0): items (u, j5, n), u-major, thread-strided by
// NT, each advanced by NT with carries. The shared expert's 5 R items continue the same numbering after the routed
// ones, so they land on the threads with the fewer routed items.
template <fmt F, int DEPTH, int NT, int C> struct dn_strided_walk {
    struct pos {
        int32_t u, j5, n;
    };
    const meta_c<C> & m;
    const uint8_t * w;
    int32_t tid, n0, R, per_u, n_items, sh_j5, sh_n;
    pos step, last;
    bool has_sh;
    __device__ __forceinline__ dn_strided_walk(const down_args & a, const meta_c<C> & m_, int32_t n0_, int32_t R_,
                                               int32_t tid_)
        : m(m_), w(a.w), tid(tid_), n0(n0_), R(R_) {
        per_u = n_item_k * R;
        n_items = m.n_u * per_u;
        step = pos_of(NT); // NT < per_u * 40 always; u may be > 0
        last = pos_of(n_items - 1);
        // this thread's shared item: the first index of its stream at or past n_items, if it is one of the 5 R
        const int32_t i_sh = n_items > tid ? tid + (n_items - tid + NT - 1) / NT * NT : tid;
        const int32_t j_sh = i_sh - n_items;
        has_sh = j_sh < per_u;
        sh_j5 = has_sh ? j_sh / R : 0;
        sh_n = has_sh ? j_sh - sh_j5 * R : 0;
    }
    __device__ __forceinline__ pos pos_of(int32_t i) const {
        const int32_t u = i / per_u;
        const int32_t r = i - u * per_u;
        const int32_t j5 = r / R;
        return {u, j5, r - j5 * R};
    }
    __device__ __forceinline__ pos advance(pos p) const {
        p.n += step.n;
        p.j5 += step.j5;
        p.u += step.u;
        if (p.n >= R) {
            p.n -= R;
            ++p.j5;
        }
        if (p.j5 >= n_item_k) {
            p.j5 -= n_item_k;
            ++p.u;
        }
        return p;
    }
    __device__ __forceinline__ dn_slot<F> load(const pos p, uint32_t z) const {
        const uint8_t * e = w + int64_t(m.e[p.u]) * dn_expert_bytes<F>();
        return dn_load<F>(e + dn_code_off<F>(p.j5, n0 + p.n), e + dn_d_off<F>(p.j5, n0 + p.n), z);
    }
    // a load past the last item reads the last item again (never predicated)
    __device__ __forceinline__ void prime(dn_slot<F> (&rg)[DEPTH]) const {
        int32_t il = tid;
        pos pl = pos_of(tid);
#pragma unroll
        for (int32_t q = 0; q < DEPTH; ++q) {
            rg[q] = load(il < n_items ? pl : last, 0u);
            il += NT;
            pl = advance(pl);
        }
    }
    template <int B>
    __device__ __forceinline__ void run(dn_smem<C> & s, dn_slot<F> (&rg)[DEPTH], const uint32_t (&kv)[4],
                                        uint32_t mz) const {
        pos p = pos_of(tid);
        int32_t il = tid + DEPTH * NT; // next item to load
        pos pl = p;
#pragma unroll
        for (int32_t q = 0; q < DEPTH; ++q) {
            pl = advance(pl);
        }
        for (int32_t i0 = tid; i0 < n_items; i0 += DEPTH * NT) {
#pragma unroll
            for (int32_t q = 0; q < DEPTH; ++q) {
                if (i0 + q * NT >= n_items) {
                    break;
                }
                const dn_slot<F> v = rg[q];
                rg[q] = load(il < n_items ? pl : last, order_after(v.c[0].x, mz));
                il += NT;
                pl = advance(pl);
                dn_item<F, B>(s, v, p.j5, p.n, m.mask[p.u], m.slot_base[p.u], kv);
                p = advance(p);
            }
        }
    }
};

// ME3: the cache's walks (down_kernel's X), the resident ones' with each expert through the cache's table
// (down_args::wt; ME1's offsets from the expert's first byte) and a unit whose first slot is in ext (the CPU's slots)
// loading nothing and running no items. The resident walks above are 412ecbd's text.
template <fmt F, int DEPTH, int C> struct dn_fixed_walk_x {
    const meta_c<C> & m;
    int32_t g, sh_j5, sh_n, n_mine, u_last;
    int32_t co, dof; // the item's code and d offsets in an expert
    const uint8_t * const * wt;
    uint64_t ext;
    bool has_sh;
    __device__ __forceinline__ dn_fixed_walk_x(const down_args & a, const meta_c<C> & m_, int32_t n0, int32_t R,
                                               int32_t tid, uint64_t ext_)
        : m(m_), wt(a.wt), ext(ext_) {
        const int32_t per_u = n_item_k * R;
        g = tid / per_u; // 3: past the groups
        const int32_t r = tid - g * per_u;
        sh_j5 = r / R;
        sh_n = r - sh_j5 * R;
        n_mine = g < 3 ? (m.n_u - g + 2) / 3 : 0;
        u_last = g + 3 * (n_mine - 1);
        co = dn_code_off<F>(sh_j5, n0 + sh_n);
        dof = dn_d_off<F>(sh_j5, n0 + sh_n);
        has_sh = g == m.n_u % 3;
    }
    __device__ __forceinline__ bool on_cpu(int32_t u) const {
        return ((ext >> m.slot_base[u]) & 1u) != 0;
    }
    __device__ __forceinline__ dn_slot<F> load(int32_t u, uint32_t z) const {
        if (on_cpu(u)) {
            return {};
        }
        const uint8_t * e = wt[m.e[u]];
        return dn_load<F>(e + co, e + dof, z);
    }
    __device__ __forceinline__ void prime(dn_slot<F> (&rg)[DEPTH]) const {
        if (n_mine > 0) {
#pragma unroll
            for (int32_t q = 0; q < DEPTH; ++q) {
                rg[q] = load(min(g + 3 * q, u_last), 0u);
            }
        }
    }
    template <int B>
    __device__ __forceinline__ void run(dn_smem<C> & s, dn_slot<F> (&rg)[DEPTH], const uint32_t (&kv)[4],
                                        uint32_t mz) const {
        int32_t ul = g + 3 * DEPTH; // next unit to load
        for (int32_t k0 = 0; k0 < n_mine; k0 += DEPTH) {
#pragma unroll
            for (int32_t q = 0; q < DEPTH; ++q) {
                if (k0 + q >= n_mine) {
                    break;
                }
                const int32_t u = g + 3 * (k0 + q);
                const dn_slot<F> v = rg[q];
                rg[q] = load(min(ul, u_last), order_after(v.c[0].x, mz));
                ul += 3;
                if (!on_cpu(u)) {
                    dn_item<F, B>(s, v, sh_j5, sh_n, m.mask[u], m.slot_base[u], kv);
                }
            }
        }
    }
};

template <fmt F, int DEPTH, int NT, int C> struct dn_strided_walk_x : dn_strided_walk<F, DEPTH, NT, C> {
    using base = dn_strided_walk<F, DEPTH, NT, C>;
    using pos = typename base::pos;
    const uint8_t * const * wt;
    uint64_t ext;
    __device__ __forceinline__ dn_strided_walk_x(const down_args & a, const meta_c<C> & m_, int32_t n0_, int32_t R_,
                                                 int32_t tid_, uint64_t ext_)
        : base(a, m_, n0_, R_, tid_), wt(a.wt), ext(ext_) {}
    __device__ __forceinline__ bool on_cpu(int32_t u) const {
        return ((ext >> this->m.slot_base[u]) & 1u) != 0;
    }
    __device__ __forceinline__ dn_slot<F> load(const pos p, uint32_t z) const {
        if (on_cpu(p.u)) {
            return {};
        }
        const uint8_t * e = wt[this->m.e[p.u]];
        return dn_load<F>(e + dn_code_off<F>(p.j5, this->n0 + p.n), e + dn_d_off<F>(p.j5, this->n0 + p.n), z);
    }
    __device__ __forceinline__ void prime(dn_slot<F> (&rg)[DEPTH]) const {
        int32_t il = this->tid;
        pos pl = this->pos_of(this->tid);
#pragma unroll
        for (int32_t q = 0; q < DEPTH; ++q) {
            rg[q] = load(il < this->n_items ? pl : this->last, 0u);
            il += NT;
            pl = this->advance(pl);
        }
    }
    template <int B>
    __device__ __forceinline__ void run(dn_smem<C> & s, dn_slot<F> (&rg)[DEPTH], const uint32_t (&kv)[4],
                                        uint32_t mz) const {
        pos p = this->pos_of(this->tid);
        int32_t il = this->tid + DEPTH * NT; // next item to load
        pos pl = p;
#pragma unroll
        for (int32_t q = 0; q < DEPTH; ++q) {
            pl = this->advance(pl);
        }
        for (int32_t i0 = this->tid; i0 < this->n_items; i0 += DEPTH * NT) {
#pragma unroll
            for (int32_t q = 0; q < DEPTH; ++q) {
                if (i0 + q * NT >= this->n_items) {
                    break;
                }
                const dn_slot<F> v = rg[q];
                rg[q] = load(il < this->n_items ? pl : this->last, order_after(v.c[0].x, mz));
                il += NT;
                pl = this->advance(pl);
                if (!on_cpu(p.u)) {
                    dn_item<F, B>(s, v, p.j5, p.n, this->m.mask[p.u], this->m.slot_base[p.u], kv);
                }
                p = this->advance(p);
            }
        }
    }
};

// Down and the weighted sum. One block an SM (72); block i owns output rows n0 .. n0 + R - 1 (n0 = 2560 i / 72, R 35
// or 36) for every routed pair and the shared expert. The routed items (u, j5, n) are dealt by the walk: at
// dn_threads, dn_fixed_walk (K5: a fixed item a thread, units g, g + 3, ...); at dn_threads_strided (Q8_0),
// dn_strided_walk (thread-strided). Each thread streams its items through a ring of DEPTH items refilled as soon as an
// item's data is taken; its one shared item (format FS, every row), if any, is loaded right after the ring is primed.
// The routed pairs' and the shared rows' GLU codes are staged once in shared memory (Q2_0: transposed), read by
// broadcast, with each run's d and its codes' exact sum. Each item writes one partial per row it serves; after a
// barrier, thread (b, n) sums each slot's 5 partials and the 10 routed experts in route order, then adds the shared
// expert's (sg[b]). Which thread runs an item changes no result.
// Routing (XB2): a.meta holds the meta of a.ids (experts_meta.cuh), written before the launch (by gate_up's block 0,
// XB3); warp 0 loads it at entry, and a.ids is not read.
// Budget: __launch_bounds__(NT, 1) (544: 96 registers; 512: 128); shared memory sizeof(dn_smem) (dynamic).
// ME3: X, the cache's instantiation: the CPU's slots (ext_slots, or the word ext_slots_d names, written before the
// launch) run no items and take no term (the combine adds their products after this launch); the resident one (X
// false) is 412ecbd's kernel.
template <bool X, fmt F, fmt FS, int B, int DEPTH, int NT>
__global__ void __launch_bounds__(NT, 1) down_kernel(down_args a) {
    static_assert(NT == dn_threads_strided ||
                      (3 * n_item_k * max_rows_block <= NT && 4 * n_item_k * (max_rows_block - 1) > NT),
                  "three groups of 5 R item threads fit, and the threads past them are fewer than a group");
    constexpr int32_t C = cap_of(B); // SP2: the routing capacity (experts.h)
    extern __shared__ uint4 smem_u4[];
    dn_smem<C> & s = *reinterpret_cast<dn_smem<C> *>(smem_u4);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const uint32_t mz = gridDim.x == n_sm ? 0u : ~0u;

    // the GLU slots (at most 11 B: the routed pairs, then the shared rows, contiguous from slot 0), codes then scales:
    // coalesced 16-byte loads through L1 (R3), stored to shared memory once the first weight items are issued
    constexpr int32_t n_stage = (n_used + 1) * B;
    constexpr int32_t n_c16 = n_stage * n_ff_card / 16;
    constexpr int32_t n_raw16 = n_c16 + n_stage * n_group * 8 / 16;
    constexpr int32_t n_raw_per = (n_raw16 + NT - 1) / NT;
    static_assert(n_ff_card % 16 == 0 && (n_group * 8) % 16 == 0, "the slots of a row tile 16-byte loads");
    uint4 raw_v[n_raw_per];
#pragma unroll
    for (int32_t k = 0; k < n_raw_per; ++k) {
        const int32_t i = tid + k * NT;
        raw_v[k] = i < n_c16     ? __ldg(reinterpret_cast<const uint4 *>(a.glu.codes) + i)
                   : i < n_raw16 ? __ldg(reinterpret_cast<const uint4 *>(a.glu.scales) + (i - n_c16))
                                 : make_uint4(0, 0, 0, 0);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if (F == fmt::iq4_nl || FS == fmt::iq4_nl) {
        const uint4 t = __ldg(reinterpret_cast<const uint4 *>(a.grids + kvalues_off));
        kv[0] = t.x;
        kv[1] = t.y;
        kv[2] = t.z;
        kv[3] = t.w;
    }
    if (tid < 32) {
        load_meta(a.meta, s.m, tid, 32); // XB2: the meta gate_up wrote (experts_meta.cuh)
    } else if (tid < 32 + B * n_used) {
        s.wts[tid - 32] = __ldg(a.wts + tid - 32);
    } else if (tid < 32 + B * n_used + B) {
        s.sg[tid - 32 - B * n_used] = __ldg(a.sg + tid - 32 - B * n_used);
    }
    __syncthreads();
    const meta_c<C> & m = s.m;
    if (m.bad != 0) {
        if (tid == 0 && blockIdx.x == 0 && a.status != nullptr) {
            *a.status = 1u;
        }
        return;
    }
    const int32_t n0 = static_cast<int32_t>((int64_t(D) * blockIdx.x) / n_sm);
    const int32_t R = static_cast<int32_t>((int64_t(D) * (blockIdx.x + 1)) / n_sm) - n0;

    const uint64_t ext = X ? (a.ext_slots_d != nullptr ? __ldg(a.ext_slots_d) : a.ext_slots) : 0u;
    using walk_r = std::conditional_t<NT == dn_threads, dn_fixed_walk<F, DEPTH, C>, dn_strided_walk<F, DEPTH, NT, C>>;
    using walk_c =
        std::conditional_t<NT == dn_threads, dn_fixed_walk_x<F, DEPTH, C>, dn_strided_walk_x<F, DEPTH, NT, C>>;
    using walk_t = std::conditional_t<X, walk_c, walk_r>;
    const walk_t walk = [&]() {
        if constexpr (X) {
            return walk_t(a, m, n0, R, tid, ext);
        } else {
            return walk_t(a, m, n0, R, tid);
        }
    }();
    dn_slot<F> rg[DEPTH] = {};
    walk.prime(rg);
    dn_slot<FS> shv = {};
    if (walk.has_sh) {
        shv = dn_load<FS>(a.sh_w + dn_code_off<FS>(walk.sh_j5, n0 + walk.sh_n),
                          a.sh_w + dn_d_off<FS>(walk.sh_j5, n0 + walk.sh_n), 0u);
    }

#pragma unroll
    for (int32_t k = 0; k < n_raw_per; ++k) {
        const int32_t i = tid + k * NT;
        if (i < n_c16) {
            reinterpret_cast<uint4 *>(s.raw)[i] = raw_v[k];
        } else if (i < n_raw16) {
            reinterpret_cast<uint4 *>(s.raw + glu_codes_c<C>)[i - n_c16] = raw_v[k];
        }
    }
    __syncthreads();
    // convert the staged slots (the routed pairs, then the shared rows): thread t < 10 x slots takes run t % 10 of slot
    // t / 10 (codes t x 32 bytes in: slots are 320 codes); a quarter-warp's 16-byte reads are 32 bytes apart, two-way
    // conflicts, once a launch. Q2_0 weights read their codes transposed: the slot's expert's format decides.
    for (int32_t t = tid; t < (m.n_slots + B) * 10; t += NT) {
        const int32_t slot = t / 10;
        const int32_t blk = t - slot * 10;
        const uint4 * src = reinterpret_cast<const uint4 *>(s.raw + t * 32);
        const uint4 c0 = src[0], c1 = src[1];
        const uint32_t qw[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        int32_t sum = 0;
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            sum = __dp4a(static_cast<int32_t>(qw[i]), 0x01010101, sum);
        }
        s.da[slot][blk] = reinterpret_cast<const float *>(s.raw + glu_codes_c<C>)[2 * t]; // d of {d, s}
        s.asum[slot][blk] = sum; // exact; s (d x sum, rounded) is not used: see the header
        uint4 * dst = reinterpret_cast<uint4 *>(s.act + slot * act_stride + blk * 32);
        if (slot < m.n_slots ? F == fmt::q2_0 : FS == fmt::q2_0) {
            dst[0] = transpose16(qw[0], qw[1], qw[2], qw[3]);
            dst[1] = transpose16(qw[4], qw[5], qw[6], qw[7]);
        } else {
            dst[0] = make_uint4(qw[0], qw[1], qw[2], qw[3]);
            dst[1] = make_uint4(qw[4], qw[5], qw[6], qw[7]);
        }
    }
    __syncthreads();

    walk.template run<B>(s, rg, kv, mz);
    if (walk.has_sh) {
        dn_item<FS, B, true>(s, shv, walk.sh_j5, walk.sh_n, (1u << B) - 1u, m.n_slots, kv);
    }
    __syncthreads();

    // the weighted sum, route order, then the shared expert's term; ME3: without the CPU's slots (X)
    for (int32_t t = tid; t < B * R; t += NT) {
        const int32_t b = t / R;
        const int32_t n = t - b * R;
        float acc = 0.0f;
#pragma unroll
        for (int32_t k = 0; k < n_used; ++k) {
            const int32_t slot = slot_of(m, m.u_of[b * n_used + k], b);
            if (X && ((ext >> slot) & 1u) != 0) {
                continue;
            }
            float ds = s.part[slot][0][n];
#pragma unroll
            for (int32_t j5 = 1; j5 < n_item_k; ++j5) {
                ds += s.part[slot][j5][n];
            }
            acc = fmaf(s.wts[b * n_used + k], ds, acc);
        }
        const int32_t ss = m.n_slots + b;
        float dsh = s.part[ss][0][n];
#pragma unroll
        for (int32_t j5 = 1; j5 < n_item_k; ++j5) {
            dsh += s.part[ss][j5][n];
        }
        a.out[b * D + n0 + n] = fmaf(s.sg[b], dsh, acc);
    }
}

// down_kernel_lanes's arguments, a lane: its own, and FMS2 item 6's X-F peer rows (experts.h down_lanes_send): the
// lane's 5 joint frame rows in the peer's slot, null for none. Beside the lane's arguments, the store's address shares
// their per-lane offset (a separate array's offset spilled the iq4_nl + q2_0 instance at the 96-register cap)
struct down_lane_karg {
    down_args a;
    uint32_t * peer;
};

struct down_lanes_kargs {
    down_lane_karg s[max_lanes];
    int32_t peer_half;  // the half the peer rows hold
    uint32_t * x16_err; // FMS2 item 7: non-null, the peer rows take FP16 halves (x16_bits) and this the range flag
};

// FMU2 (build item 1, experts.h down_lanes): a joint round's lanes in one launch, the resident walk (dn_threads) at
// B = 5. Block i runs lane k = i / NB on its arguments p.s[k] (read in place: __grid_constant__) and, in it, the lone
// launch's blocks vb = L (i % NB) .. L (i % NB) + L - 1 (L = 72 / NB, the lanes) in turn: each vb's output rows (n0 =
// 2560 vb / 72, R 35 or 36), walk, shared item and weighted sum are down_kernel's, after one staging of the lane's GLU
// slots, so the lane's out is bitwise its own down launch's (which block runs an item changes no result). A barrier
// ends each vb before the next one's items write the partials.
template <fmt F, fmt FS, int B, int DEPTH, int NB>
__global__ void __launch_bounds__(dn_threads, 1) down_kernel_lanes(const __grid_constant__ down_lanes_kargs p) {
    constexpr int32_t NT = dn_threads;
    constexpr int32_t L = n_sm / NB;
    static_assert(L * NB == n_sm && L >= 2 && L <= max_lanes, "the lanes share the 72 blocks evenly");
    constexpr int32_t C = cap_of(B);
    const int32_t ln = static_cast<int32_t>(blockIdx.x) / NB;
    const down_lane_karg & l = p.s[ln];
    const down_args & a = l.a;
    const int32_t vb0 = (static_cast<int32_t>(blockIdx.x) - ln * NB) * L;
    extern __shared__ uint4 smem_u4[];
    dn_smem<C> & s = *reinterpret_cast<dn_smem<C> *>(smem_u4);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const uint32_t mz = gridDim.x == n_sm ? 0u : ~0u;

    // the GLU slots, codes then scales (down_kernel's staging)
    constexpr int32_t n_stage = (n_used + 1) * B;
    constexpr int32_t n_c16 = n_stage * n_ff_card / 16;
    constexpr int32_t n_raw16 = n_c16 + n_stage * n_group * 8 / 16;
    constexpr int32_t n_raw_per = (n_raw16 + NT - 1) / NT;
    uint4 raw_v[n_raw_per];
#pragma unroll
    for (int32_t k = 0; k < n_raw_per; ++k) {
        const int32_t i = tid + k * NT;
        raw_v[k] = i < n_c16     ? __ldg(reinterpret_cast<const uint4 *>(a.glu.codes) + i)
                   : i < n_raw16 ? __ldg(reinterpret_cast<const uint4 *>(a.glu.scales) + (i - n_c16))
                                 : make_uint4(0, 0, 0, 0);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if (F == fmt::iq4_nl || FS == fmt::iq4_nl) {
        const uint4 t = __ldg(reinterpret_cast<const uint4 *>(a.grids + kvalues_off));
        kv[0] = t.x;
        kv[1] = t.y;
        kv[2] = t.z;
        kv[3] = t.w;
    }
    if (tid < 32) {
        load_meta(a.meta, s.m, tid, 32);
    } else if (tid < 32 + B * n_used) {
        s.wts[tid - 32] = __ldg(a.wts + tid - 32);
    } else if (tid < 32 + B * n_used + B) {
        s.sg[tid - 32 - B * n_used] = __ldg(a.sg + tid - 32 - B * n_used);
    }
    __syncthreads();
    const meta_c<C> & m = s.m;
    constexpr int32_t hw = fn::xchg::words_half;
    static_assert(D == 2 * hw && (int64_t(D) * (n_sm / 2)) / n_sm == hw, "the halves split at a whole vb");
    if (m.bad != 0) {
        if (tid == 0 && vb0 == 0 && a.status != nullptr) {
            *a.status = 1u;
        }
        // FMS2 item 6: the peer's chunks as zeros, so its receive does not wait out the poll (the status fails the
        // round)
#pragma unroll 1
        for (int32_t v = 0; l.peer != nullptr && v < L; ++v) {
            const int32_t vb = vb0 + v;
            const int32_t n0 = static_cast<int32_t>((int64_t(D) * vb) / n_sm);
            const int32_t R = static_cast<int32_t>((int64_t(D) * (vb + 1)) / n_sm) - n0;
            if (n0 / hw == p.peer_half) {
                for (int32_t t = tid; t < B * R; t += NT) {
                    const int32_t b = t / R;
                    const int32_t q = n0 - p.peer_half * hw + t - b * R;
                    if (p.x16_err != nullptr) {
                        reinterpret_cast<uint16_t *>(l.peer)[b * fn::xchg::x16_row_halves + q] = 0u;
                    } else {
                        l.peer[b * hw + q] = 0u;
                    }
                }
            }
        }
        return;
    }
    using walk_r = dn_fixed_walk<F, DEPTH, C>;
#pragma unroll 1
    for (int32_t v = 0; v < L; ++v) {
        const int32_t vb = vb0 + v;
        const int32_t n0 = static_cast<int32_t>((int64_t(D) * vb) / n_sm);
        const int32_t R = static_cast<int32_t>((int64_t(D) * (vb + 1)) / n_sm) - n0;
        const walk_r walk(a, m, n0, R, tid);
        dn_slot<F> rg[DEPTH] = {};
        walk.prime(rg);
        dn_slot<FS> shv = {};
        if (walk.has_sh) {
            shv = dn_load<FS>(a.sh_w + dn_code_off<FS>(walk.sh_j5, n0 + walk.sh_n),
                              a.sh_w + dn_d_off<FS>(walk.sh_j5, n0 + walk.sh_n), 0u);
        }
        if (v == 0) {
            // the staged slots, as down_kernel stores and converts them, once for the lane's vbs
#pragma unroll
            for (int32_t k = 0; k < n_raw_per; ++k) {
                const int32_t i = tid + k * NT;
                if (i < n_c16) {
                    reinterpret_cast<uint4 *>(s.raw)[i] = raw_v[k];
                } else if (i < n_raw16) {
                    reinterpret_cast<uint4 *>(s.raw + glu_codes_c<C>)[i - n_c16] = raw_v[k];
                }
            }
            __syncthreads();
            for (int32_t t = tid; t < (m.n_slots + B) * 10; t += NT) {
                const int32_t slot = t / 10;
                const int32_t blk = t - slot * 10;
                const uint4 * src = reinterpret_cast<const uint4 *>(s.raw + t * 32);
                const uint4 c0 = src[0], c1 = src[1];
                const uint32_t qw[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
                int32_t sum = 0;
#pragma unroll
                for (int32_t i = 0; i < 8; ++i) {
                    sum = __dp4a(static_cast<int32_t>(qw[i]), 0x01010101, sum);
                }
                s.da[slot][blk] = reinterpret_cast<const float *>(s.raw + glu_codes_c<C>)[2 * t];
                s.asum[slot][blk] = sum;
                uint4 * dst = reinterpret_cast<uint4 *>(s.act + slot * act_stride + blk * 32);
                if (slot < m.n_slots ? F == fmt::q2_0 : FS == fmt::q2_0) {
                    dst[0] = transpose16(qw[0], qw[1], qw[2], qw[3]);
                    dst[1] = transpose16(qw[4], qw[5], qw[6], qw[7]);
                } else {
                    dst[0] = make_uint4(qw[0], qw[1], qw[2], qw[3]);
                    dst[1] = make_uint4(qw[4], qw[5], qw[6], qw[7]);
                }
            }
            __syncthreads();
        }
        walk.template run<B>(s, rg, kv, mz);
        if (walk.has_sh) {
            dn_item<FS, B, true>(s, shv, walk.sh_j5, walk.sh_n, (1u << B) - 1u, m.n_slots, kv);
        }
        __syncthreads();
        // the weighted sum, route order, then the shared expert's term (down_kernel's); FMS2 item 6: a chunk of the
        // peer's half also to the peer
        const bool send = l.peer != nullptr && n0 / hw == p.peer_half;
        for (int32_t t = tid; t < B * R; t += NT) {
            const int32_t b = t / R;
            const int32_t n = t - b * R;
            float acc = 0.0f;
#pragma unroll
            for (int32_t k = 0; k < n_used; ++k) {
                const int32_t slot = slot_of(m, m.u_of[b * n_used + k], b);
                float ds = s.part[slot][0][n];
#pragma unroll
                for (int32_t j5 = 1; j5 < n_item_k; ++j5) {
                    ds += s.part[slot][j5][n];
                }
                acc = fmaf(s.wts[b * n_used + k], ds, acc);
            }
            const int32_t ss = m.n_slots + b;
            float dsh = s.part[ss][0][n];
#pragma unroll
            for (int32_t j5 = 1; j5 < n_item_k; ++j5) {
                dsh += s.part[ss][j5][n];
            }
            const float y = fmaf(s.sg[b], dsh, acc);
            a.out[b * D + n0 + n] = y;
            if (send) {
                const int32_t q = n0 - p.peer_half * hw + n;
                if (p.x16_err != nullptr) { // FMS2 item 7: an FP16 half, its own flag (xchg.h)
                    bool over = false;
                    uint16_t * const p16 = reinterpret_cast<uint16_t *>(l.peer);
                    p16[b * fn::xchg::x16_row_halves + q] = fn::xchg::x16_bits(y, over);
                    if (over) {
                        atomicOr(p.x16_err, fn::xchg::err_x16_range);
                    }
                } else {
                    l.peer[b * hw + q] = fn::xchg::canonical(y);
                }
            }
        }
        __syncthreads();
    }
}

// XB2: write_meta's launch, one warp: the IDs to shared memory, then the builder (route_meta)
template <int B>
__global__ void __launch_bounds__(32, 1) meta_kernel(const int32_t * ids, uint8_t * out, class_table cw) {
    constexpr int32_t C = cap_of(B); // SP2: the routing capacity
    __shared__ int32_t ids_sm[meta_c<C>::slots];
    __shared__ uint4 scratch[(meta_order_of(C) + 2 * max_units_of(C) + 15) / 16];
    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    for (int32_t i = lane; i < n_used * B; i += 32) {
        ids_sm[i] = __ldg(ids + i);
    }
    __syncwarp();
    route_meta<B>(ids_sm, reinterpret_cast<uint8_t *>(scratch), out, cw);
}

// ME1: glu_out's epilogue on given sums (glu_finish): warp w of a block takes (row, group) 8 blockIdx.x + w
__global__ void __launch_bounds__(256, 1) glu_finish_kernel(const float * gs, const float * us, int32_t n,
                                                              int8_t * codes, float * scales) {
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    const int32_t item = static_cast<int32_t>(blockIdx.x) * 8 + static_cast<int32_t>(threadIdx.x) / 32;
    if (item >= n * n_group) {
        return;
    }
    const int32_t c = item * 32 + lane; // = row x 320 + 32 g + lane
    float v, d, sv;
    codes[c] = glu_run(__ldg(gs + c), __ldg(us + c), v, d, sv);
    if (lane == 0) {
        reinterpret_cast<float2 *>(scales)[item] = make_float2(d, sv);
    }
}

// ---- launch -----------------------------------------------------------------------------------------------------
bool aligned16(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}
bool aligned4(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 3u) == 0;
}
// ME1's optional pointers: null, or aligned to a
bool null_or_aligned(const void * p, uintptr_t a) {
    return p == nullptr || (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0;
}

// a view of FNFL's activation format with the given shape, codes and scales each 16-byte aligned (both kernels read
// them as 16-byte loads)
template <typename V> bool act_ok(const V & v, int32_t rows, int32_t k) {
    return v.rows == rows && v.k == k && aligned16(v.codes) && aligned16(v.scales);
}

template <bool X, fmt F, fmt FS, int B> bool launch_down_x(const down_args & a, cudaStream_t st) {
    constexpr int32_t depth = F == fmt::q2_0 ? dn_depth_q2 : dn_depth_nl;
    constexpr int32_t smem = static_cast<int32_t>(sizeof(dn_smem<cap_of(B)>));
    static_assert(smem <= 96 * 1024, "one block's shared memory");
    static std::atomic<uint32_t> attr_done{0u};
    constexpr int32_t nt = dn_threads_of<F>();
    if (!detail::smem_attr_once(attr_done, reinterpret_cast<const void *>(down_kernel<X, F, FS, B, depth, nt>),
                                smem)) {
        return false;
    }
    down_kernel<X, F, FS, B, depth, nt><<<n_sm, nt, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

// ME3: the cache's instantiation when the arguments carry its table (down_args::wt), else the resident one
template <fmt F, fmt FS, int B> bool launch_down(const down_args & a, cudaStream_t st) {
    return a.wt != nullptr ? launch_down_x<true, F, FS, B>(a, st) : launch_down_x<false, F, FS, B>(a, st);
}

template <fmt F, fmt FS> bool down_rows(int32_t rows, const down_args & a, cudaStream_t st) {
    switch (rows) {
        case 1:
            return launch_down<F, FS, 1>(a, st);
        case 2:
            return launch_down<F, FS, 2>(a, st);
        case 3:
            return launch_down<F, FS, 3>(a, st);
        case 4:
            return launch_down<F, FS, 4>(a, st);
        case 5: // SP2: the target's verification rows; ME1: Q8_0 too (the Q4 candidate's down in five layers)
            return launch_down<F, FS, 5>(a, st);
        default:
            return false;
    }
}

struct gu_entry {
    fmt f;
    fn::dense::fmt fg, fu;
    int32_t layer;
};
constexpr gu_entry gu_table[] = {
#define FN_EXPERTS_GU_ENTRY(rf, sgf, suf, layer) {fmt::rf, fn::dense::fmt::sgf, fn::dense::fmt::suf, layer},
    FN_EXPERTS_GATE_UP_INSTANCES(FN_EXPERTS_GU_ENTRY)
#undef FN_EXPERTS_GU_ENTRY
};
static_assert(sizeof(gu_table) / sizeof(gu_table[0]) == n_gate_up_inst, "the gate/up instance count");

struct dn_entry {
    fmt f, fs;
    int32_t layer;
};
constexpr dn_entry dn_table[] = {
#define FN_EXPERTS_DN_ENTRY(rf, sf, layer) {fmt::rf, fmt::sf, layer},
    FN_EXPERTS_DOWN_INSTANCES(FN_EXPERTS_DN_ENTRY)
#undef FN_EXPERTS_DN_ENTRY
};
static_assert(sizeof(dn_table) / sizeof(dn_table[0]) == n_down_inst, "the down instance count");

// ME1: the Q4 candidate's lists
constexpr gu_entry q4_gu_table[] = {
#define FN_EXPERTS_GU_ENTRY(rf, sgf, suf, layer) {fmt::rf, fn::dense::fmt::sgf, fn::dense::fmt::suf, layer},
    FN_EXPERTS_Q4_GATE_UP_INSTANCES(FN_EXPERTS_GU_ENTRY)
#undef FN_EXPERTS_GU_ENTRY
};
static_assert(sizeof(q4_gu_table) / sizeof(q4_gu_table[0]) == n_q4_gate_up_inst, "the Q4 gate/up instance count");
constexpr dn_entry q4_dn_table[] = {
#define FN_EXPERTS_DN_ENTRY(rf, sf, layer) {fmt::rf, fmt::sf, layer},
    FN_EXPERTS_Q4_DOWN_INSTANCES(FN_EXPERTS_DN_ENTRY)
#undef FN_EXPERTS_DN_ENTRY
};
static_assert(sizeof(q4_dn_table) / sizeof(q4_dn_table[0]) == n_q4_down_inst, "the Q4 down instance count");
// ME7: the Q2 draft's lists
constexpr gu_entry q2d_gu_table[] = {
#define FN_EXPERTS_GU_ENTRY(rf, sgf, suf, layer) {fmt::rf, fn::dense::fmt::sgf, fn::dense::fmt::suf, layer},
    FN_EXPERTS_DRAFT_Q2_GATE_UP_INSTANCES(FN_EXPERTS_GU_ENTRY)
#undef FN_EXPERTS_GU_ENTRY
};
static_assert(sizeof(q2d_gu_table) / sizeof(q2d_gu_table[0]) == n_draft_q2_gate_up_inst, "the Q2 draft's gate/up");
constexpr dn_entry q2d_dn_table[] = {
#define FN_EXPERTS_DN_ENTRY(rf, sf, layer) {fmt::rf, fmt::sf, layer},
    FN_EXPERTS_DRAFT_Q2_DOWN_INSTANCES(FN_EXPERTS_DN_ENTRY)
#undef FN_EXPERTS_DN_ENTRY
};
static_assert(sizeof(q2d_dn_table) / sizeof(q2d_dn_table[0]) == n_draft_q2_down_inst, "the Q2 draft's down");

// GGUF's bytes of 256 weights (the packed layout's but for the K-quants, whose scales are unpacked)
int32_t gguf_block_bytes(fmt f) {
    return f == fmt::q4_k ? 144 : f == fmt::q5_k ? 176 : gu_layout_k(f).block_bytes;
}
// get_scale_min_k4 (ggml-quants.c): sub-block j's 6-bit scale and min from a K-quant superblock's 12 scale bytes
void scale_min_k4(int32_t j, const uint8_t * q, uint8_t & sc, uint8_t & mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = static_cast<uint8_t>((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4));
        mn = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ---- host packing helpers ---------------------------------------------------------------------------------------
void put(uint8_t * dst, const uint8_t * src, size_t n) {
    std::memcpy(dst, src, n);
}

} // namespace

const char * fmt_name(fmt f) {
    switch (f) {
        case fmt::iq2_xxs:
            return "IQ2_XXS";
        case fmt::iq2_xs:
            return "IQ2_XS";
        case fmt::iq2_s:
            return "IQ2_S";
        case fmt::iq3_xxs:
            return "IQ3_XXS";
        case fmt::iq3_s:
            return "IQ3_S";
        case fmt::q2_0:
            return "Q2_0";
        case fmt::iq4_nl:
            return "IQ4_NL";
        case fmt::q8_0:
            return "Q8_0";
        case fmt::q4_k:
            return "Q4_K";
        case fmt::q5_k:
            return "Q5_K";
        case fmt::q5_1:
            return "Q5_1";
    }
    return "?";
}

void workspace_offsets(size_t (&off)[n_ws + 1]) {
    const size_t b = max_rows;
    const size_t bytes[n_ws] = {b * x_codes,       b * x_scales, size_t(glu_codes), size_t(glu_scales),
                                b * 10 * 4,        b * 10 * 4,   b * 2560 * 4,      4,
                                size_t(meta_bytes)};
    off[0] = 0;
    for (int32_t i = 0; i < n_ws; ++i) {
        off[i + 1] = off[i] + (bytes[i] + 255) / 256 * 256;
    }
}

size_t workspace_bytes() {
    size_t off[n_ws + 1];
    workspace_offsets(off);
    return off[n_ws];
}

int64_t gate_up_bytes(fmt f) {
    return gu_layout_k(f).block_bytes == 0 ? 0 : tile_bytes(f) * n_group;
}

int64_t down_bytes(fmt f) {
    return f == fmt::q2_0     ? 230400
           : f == fmt::iq4_nl ? 460800
           : f == fmt::q8_0   ? dn_expert_bytes<fmt::q8_0>()
           : f == fmt::q5_1   ? dn_expert_bytes<fmt::q5_1>()
                              : 0;
}

bool grid_table(const void * const src[6], const size_t src_bytes[6], uint8_t * dst) {
    static const size_t want[6] = {2048, 4096, 8192, 1024, 2048, 16};
    size_t off = 0;
    for (int32_t i = 0; i < 6; ++i) {
        if (src[i] == nullptr || src_bytes[i] != want[i]) {
            return false;
        }
        std::memcpy(dst + off, src[i], want[i]);
        off += want[i];
    }
    // FMS4 item 2: the IQ2 grids' selector tables, entry e's nibble k the level code of byte k of entry e (weight k),
    // the index of its magnitude in the decoders' PRMT palette (experts_dev.cuh iq2_signed)
    for (int32_t i = 0; i < 3; ++i) {
        const uint8_t * g = static_cast<const uint8_t *>(src[i]);
        for (size_t e = 0; e < want[i] / 8; ++e) {
            uint32_t w = 0;
            for (int32_t k = 0; k < 8; ++k) {
                const uint8_t b = g[8 * e + static_cast<size_t>(k)];
                if (b != 8 && b != 25 && b != 43) {
                    return false;
                }
                w |= static_cast<uint32_t>(b == 8 ? 0 : b == 25 ? 1 : 2) << (4 * k);
            }
            std::memcpy(dst + off, &w, 4); // little-endian, as the kernels read it
            off += 4;
        }
    }
    return off == grid_bytes;
}

bool pack_gate_up(fmt f, const uint8_t * gate, const uint8_t * up, size_t row_bytes, uint8_t * dst) {
    const gu_layout L = gu_layout_k(f);
    const int32_t bb = gguf_block_bytes(f);
    if (L.block_bytes == 0 || gate == nullptr || up == nullptr || dst == nullptr || row_bytes < size_t(10) * bb) {
        return false;
    }
    const int64_t T = tile_bytes(f);
    for (int32_t g = 0; g < n_group; ++g) {
        uint8_t * tile = dst + g * T;
        for (int32_t rec = 0; rec < gu_threads; ++rec) {
            const int32_t j = rec >> 3, rs = rec & 7;
            const int32_t sbk = j >> 3, ib = j & 7;
            for (int32_t mrow = 0; mrow < 8; ++mrow) {
                const int32_t r = rs + 8 * mrow;
                const uint8_t * row =
                    (r < 32 ? gate + size_t(32 * g + r) * row_bytes : up + size_t(32 * g + r - 32) * row_bytes);
                const uint8_t * blk = row + size_t(sbk) * bb; // one 256-weight superblock
                const auto mc = [&](int32_t c, int32_t byte) {
                    return tile + (int64_t(c) * gu_threads + rec) * 16 + byte;
                };
                switch (f) {
                    case fmt::iq2_xxs:
                    case fmt::iq2_xs:
                        put(mc(mrow / 2, 8 * (mrow & 1)), blk + 2 + 8 * ib, 8);
                        if (f == fmt::iq2_xs) {
                            put(tile + aux8_off(f) + rec * 8 + mrow, blk + 2 + 64 + ib, 1);
                        }
                        break;
                    case fmt::iq2_s:
                        put(mc(0, mrow), blk + 2 + 64 + ib, 1);                              // qh
                        put(mc(0, 8 + mrow), blk + 2 + 64 + 8 + ib, 1);                      // scales
                        put(mc(1 + mrow / 2, 8 * (mrow & 1)), blk + 2 + 4 * ib, 4);          // grid low bits
                        put(mc(1 + mrow / 2, 8 * (mrow & 1) + 4), blk + 2 + 32 + 4 * ib, 4); // signs
                        break;
                    case fmt::iq3_xxs:
                    case fmt::iq3_s: {
                        const int32_t hq = mrow / 4, i4 = mrow & 3;
                        // the kernel's chunk order: [q r0][sg r0][sg r1], [q r1][sg r2][sg r3], [q r2][q r3]
                        static const int32_t q_chunk[4] = {0, 1, 2, 2}, q_byte[4] = {0, 0, 0, 8};
                        static const int32_t s_chunk[4] = {0, 0, 1, 1}, s_byte[4] = {8, 12, 8, 12};
                        const int32_t qc = q_chunk[i4], qb = q_byte[i4], sc_ = s_chunk[i4], sbb = s_byte[i4];
                        put(mc(3 * hq + qc, qb), blk + 2 + 8 * ib, 8);
                        if (f == fmt::iq3_xxs) {
                            put(mc(3 * hq + sc_, sbb), blk + 2 + 64 + 4 * ib, 4);
                        } else {
                            put(mc(3 * hq + sc_, sbb), blk + 2 + 64 + 8 + 4 * ib, 4);       // signs
                            put(tile + aux8_off(f) + rec * 8 + mrow, blk + 2 + 64 + ib, 1); // qh
                            if ((j & 1) == 0) {
                                put(tile + sc8_off(f) + ((j >> 1) * 8 + rs) * 8 + mrow, blk + 2 + 64 + 8 + 32 + ib / 2,
                                    1); // scales byte, shared by sub-blocks j and j + 1
                            }
                        }
                        break;
                    }
                    case fmt::q8_0: {
                        // sub-block j is GGUF block j (34 bytes: FP16 d, 32 int8), ib of the 8 in blk
                        const uint8_t * b8 = blk + 34 * ib;
                        put(mc(2 * mrow, 0), b8 + 2, 16);
                        put(mc(2 * mrow + 1, 0), b8 + 2 + 16, 16);
                        put(tile + d_off(f) + int64_t(rec) * 16 + 2 * mrow, b8, 2); // the record's own d
                        break;
                    }
                    case fmt::q2_0: {
                        // ME7: sub-block ib is half ib & 1 of GGUF block ib / 2 (18 bytes: FP16 d, 16 code bytes of
                        // 64 weights, weight t in byte 2 + t / 4): its 8 code bytes, then the pair's d once
                        const uint8_t * b2 = blk + 18 * (ib >> 1);
                        put(mc(mrow / 2, 8 * (mrow & 1)), b2 + 2 + 8 * (ib & 1), 8);
                        if ((ib & 1) == 0) {
                            put(tile + d_off(f) + ((j >> 1) * 8 + rs) * 16 + 2 * mrow, b2, 2);
                        }
                        break;
                    }
                    case fmt::q4_k:
                    case fmt::q5_k: {
                        // ME1: GGUF's block_q4_K (d, dmin, 12 scale bytes, 128 qs bytes) and block_q5_K (the same
                        // with 32 qh bytes before qs). Sub-block ib's weight l (0..31) is qs[32 (ib / 2) + l]'s low
                        // nibble (ib even) or high nibble (odd), plus 16 x bit ib of qh[l] (Q5_K)
                        const bool q5 = f == fmt::q5_k;
                        const uint8_t * qh = blk + 16;
                        const uint8_t * qs = blk + (q5 ? 48 : 16) + 32 * (ib / 2);
                        uint8_t packed[16];
                        uint32_t hw = 0;
                        for (int32_t t = 0; t < 32; ++t) {
                            const uint8_t wv = (ib & 1) ? qs[t] >> 4 : qs[t] & 0xf;
                            if (t < 16) {
                                packed[t] = wv;
                            } else {
                                packed[t - 16] = static_cast<uint8_t>(packed[t - 16] | (wv << 4));
                            }
                            if (q5) {
                                hw |= static_cast<uint32_t>((qh[t] >> ib) & 1) << t;
                            }
                        }
                        const int32_t chunk = q5 ? (mrow < 4 ? 1 + mrow : 2 + mrow) : mrow;
                        put(mc(chunk, 0), packed, 16);
                        if (q5) {
                            uint8_t hb[4];
                            std::memcpy(hb, &hw, 4); // the little-endian word the kernel reads
                            put(mc(mrow < 4 ? 0 : 5, 4 * (mrow & 3)), hb, 4);
                        }
                        uint8_t sm[2];
                        scale_min_k4(ib, blk + 4, sm[0], sm[1]);
                        put(tile + sm_off(f) + int64_t(rec) * 16 + 2 * mrow, sm, 2);
                        if (ib == 0) {
                            put(tile + dmin_off(f) + (sbk * 8 + rs) * 16 + 2 * mrow, blk + 2, 2); // dmin
                        }
                        break;
                    }
                    default:
                        return false;
                }
                if (ib == 0 && f != fmt::q8_0 && f != fmt::q2_0) {
                    put(tile + d_off(f) + (sbk * 8 + rs) * 16 + 2 * mrow, blk, 2); // d
                }
            }
        }
    }
    return true;
}

bool pack_down(fmt f, const uint8_t * down_rows_, size_t row_bytes, uint8_t * dst) {
    if (f == fmt::q5_1) { // ME1: 10 blocks of 24 bytes (FP16 d, FP16 m, qh word, 16 qs bytes) a row
        if (down_rows_ == nullptr || dst == nullptr || row_bytes < size_t(2 * n_item_k) * 24) {
            return false;
        }
        const int64_t d_plane = int64_t(n_item_k) * 2 * D * 16;
        for (int32_t n = 0; n < D; ++n) {
            const uint8_t * row = down_rows_ + size_t(n) * row_bytes;
            for (int32_t b = 0; b < 2 * n_item_k; ++b) {
                const uint8_t * blk = row + size_t(b) * 24;
                uint8_t * de = dst + d_plane + (int64_t(b / 2) * D + n) * 16;
                put(dst + (int64_t(b) * D + n) * 16, blk + 8, 16); // qs
                put(de + 4 * (b & 1), blk + 4, 4);                  // qh
                put(de + 8 + 4 * (b & 1), blk, 4);                  // d, m
            }
        }
        return true;
    }
    if (f == fmt::q8_0) { // DR1: 10 blocks of 34 bytes (FP16 d, 32 int8) a row
        constexpr int32_t nc8 = down_chunks(fmt::q8_0);
        if (down_rows_ == nullptr || dst == nullptr || row_bytes < size_t(2 * n_item_k) * 34) {
            return false;
        }
        const int64_t d_plane8 = int64_t(n_item_k) * nc8 * D * 16;
        for (int32_t n = 0; n < D; ++n) {
            const uint8_t * row = down_rows_ + size_t(n) * row_bytes;
            for (int32_t b = 0; b < 2 * n_item_k; ++b) {
                const uint8_t * blk = row + size_t(b) * 34;
                const int32_t j5 = b / 2;
                for (int32_t h = 0; h < 2; ++h) {
                    put(dst + ((int64_t(j5) * nc8 + 2 * (b & 1) + h) * D + n) * 16, blk + 2 + 16 * h, 16);
                }
                put(dst + d_plane8 + (int64_t(j5) * D + n) * 4 + 2 * (b & 1), blk, 2);
            }
        }
        return true;
    }
    const int32_t nc = f == fmt::q2_0 ? 1 : 2;
    if ((f != fmt::q2_0 && f != fmt::iq4_nl) || down_rows_ == nullptr || dst == nullptr ||
        row_bytes < size_t(n_item_k) * nc * 18) { // the card's 320 columns: Q2_0 5 blocks, IQ4_NL 10
        return false;
    }
    const int64_t d_plane = int64_t(n_item_k) * nc * D * 16;
    for (int32_t n = 0; n < D; ++n) {
        const uint8_t * row = down_rows_ + size_t(n) * row_bytes;
        for (int32_t b = 0; b < n_item_k * nc; ++b) { // 18-byte blocks: FP16 d, 16 qs bytes
            const uint8_t * blk = row + size_t(b) * 18;
            put(dst + (int64_t(b) * D + n) * 16, blk + 2, 16);
            if (f == fmt::q2_0) {
                put(dst + d_plane + (int64_t(b) * D + n) * 2, blk, 2);
            } else {
                put(dst + d_plane + (int64_t(b / 2) * D + n) * 4 + 2 * (b & 1), blk, 2);
            }
        }
    }
    return true;
}

namespace detail {

// cudaFuncSetAttribute sets the current device's attribute only, and the kernels run on both cards (R1): each launch
// site keeps one bit per device. reason for the casts at the call sites: the C API takes the kernel as const void *.
bool smem_attr_once(std::atomic<uint32_t> & done, const void * kernel, int32_t smem) {
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

} // namespace detail

namespace {

// a gate/up launch's arguments checked, X-HMIX's regions taken as x (xchg.h hmix_layout) into a
bool gate_up_prepared(const gate_up_args & in, int32_t rows, gate_up_args & a) {
    a = in;
    if (in.hmix != nullptr) { // X-HMIX: the slot's codes and scales regions (xchg.h hmix_layout)
        using hl = fn::xchg::hmix_layout;
        if (in.x.codes != nullptr || in.x.scales != nullptr || !aligned16(in.hmix) || in.err == nullptr ||
            (in.hmix_peer_half != 0 && in.hmix_peer_half != 1)) {
            return false;
        }
        a.x = {reinterpret_cast<const int8_t *>(in.hmix + hl::code_at), in.hmix + hl::scale_at, rows, hl::width};
    }
    return rows >= 1 && rows <= max_rows && aligned16(a.w) && aligned16(a.sh_gate) && aligned16(a.sh_up) &&
           aligned16(a.grids) && act_ok(a.x, rows, D) && aligned4(a.ids) && aligned16(a.meta) && class_ok(a.cw) &&
           act_ok(a.glu, glu_slots, n_ff_card) && (a.glu_f32 == nullptr || aligned4(a.glu_f32)) &&
           (a.status == nullptr || aligned4(a.status)) && a.map >= 0 && a.map <= 3 && null_or_aligned(a.wt, 8) &&
           a.skip >= 0 && a.skip <= 2 && (a.skip == 0 || a.wt != nullptr) && null_or_aligned(a.run_if, 4);
}

bool deployed_gu(fmt f, fn::dense::fmt fg, fn::dense::fmt fu) {
    for (const gu_entry & e : gu_table) {
        if (e.f == f && e.fg == fg && e.fu == fu) {
            return true;
        }
    }
    return false;
}

} // namespace

bool gate_up_pair_supported(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_pair_args & a) {
    const bool iq = f == fmt::iq2_xxs || f == fmt::iq2_xs || f == fmt::iq2_s || f == fmt::iq3_xxs || f == fmt::iq3_s;
    if (!iq || (rows != 4 && rows != 5) || !deployed_gu(f, fg, fu)) {
        return false;
    }
    gate_up_args p[2];
    for (int32_t k = 0; k < 2; ++k) {
        if (!gate_up_prepared(a.s[k], rows, p[k]) || p[k].map != 3 || p[k].wt == nullptr) {
            return false;
        }
    }
    const gate_up_args & x = a.s[0];
    const gate_up_args & y = a.s[1];
    const bool same_cw = std::memcmp(&x.cw, &y.cw, sizeof(class_table)) == 0;
    return same_cw && x.ids == y.ids && x.skip == y.skip && x.run_if == y.run_if &&
           (x.hmix == nullptr) == (y.hmix == nullptr) && x.meta != y.meta && x.glu.codes != y.glu.codes &&
           x.glu.scales != y.glu.scales && (x.glu_f32 == nullptr || x.glu_f32 != y.glu_f32) &&
           (x.status == nullptr || x.status != y.status);
}

bool gate_up_pair(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_pair_args & in,
                  void * stream) {
    gate_up_pair_args a;
    if (stream == nullptr || !gate_up_pair_supported(f, fg, fu, rows, in) || !gate_up_prepared(in.s[0], rows, a.s[0]) ||
        !gate_up_prepared(in.s[1], rows, a.s[1])) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const bool iq3 = f == fmt::iq3_xxs || f == fmt::iq3_s;
    return iq3 ? detail::gate_up_pair_iq3(f, fg, fu, rows, a, st) : detail::gate_up_pair_iq2(f, fg, fu, rows, a, st);
}

bool gate_up_lanes_supported(fmt                        f,
                             fn::dense::fmt             fg,
                             fn::dense::fmt             fu,
                             int32_t                    lanes,
                             int32_t                    rows,
                             const gate_up_lanes_args & a) {
    const bool iq2 = f == fmt::iq2_xxs || f == fmt::iq2_xs || f == fmt::iq2_s;
    const bool iq3 = f == fmt::iq3_xxs || f == fmt::iq3_s;
    // FMS13: rows 2 .. 5; the IQ3 body is B = 4 and 5 only (the lone IQ3 launch at B <= 3 is the per-triple kernel)
    if ((!iq2 && !iq3) || lanes < 2 || lanes > max_lanes || rows < (iq3 ? 4 : 2) || rows > max_rows ||
        !deployed_gu(f, fg, fu)) {
        return false;
    }
    for (int32_t k = 0; k < lanes; ++k) {
        gate_up_args p;
        const gate_up_args & y = a.s[k];
        // the lone launch's body at B = rows (gate_up_deployed): IQ2 under map 2 or 3, IQ3 under map 3; resident
        if (!gate_up_prepared(y, rows, p) || !(p.map == 3 || (iq2 && p.map == 2)) || p.wt != nullptr || p.skip != 0 ||
            p.run_if != nullptr) {
            return false;
        }
        for (int32_t q = 0; q < k; ++q) {
            const gate_up_args & x = a.s[q];
            if (std::memcmp(&x.cw, &y.cw, sizeof(class_table)) != 0 || (x.hmix == nullptr) != (y.hmix == nullptr) ||
                x.meta == y.meta || x.glu.codes == y.glu.codes || x.glu.scales == y.glu.scales ||
                (x.glu_f32 != nullptr && x.glu_f32 == y.glu_f32)) {
                return false;
            }
        }
    }
    return true;
}

bool gate_up_lanes(fmt                        f,
                   fn::dense::fmt             fg,
                   fn::dense::fmt             fu,
                   int32_t                    lanes,
                   int32_t                    rows,
                   const gate_up_lanes_args & in,
                   void *                     stream) {
    if (stream == nullptr || !gate_up_lanes_supported(f, fg, fu, lanes, rows, in)) {
        return false;
    }
    gate_up_lanes_args a = in;
    for (int32_t k = 0; k < lanes; ++k) {
        if (!gate_up_prepared(in.s[k], rows, a.s[k])) {
            return false;
        }
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const bool iq3 = f == fmt::iq3_xxs || f == fmt::iq3_s;
    return iq3 ? detail::gate_up_lanes_iq3(f, fg, fu, lanes, rows, a, st) :
                 detail::gate_up_lanes_iq2(f, fg, fu, lanes, rows, a, st);
}

bool gate_up(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & in, void * stream) {
    gate_up_args a;
    if (!gate_up_prepared(in, rows, a) || stream == nullptr) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (f) {
        case fmt::iq2_xxs:
            return detail::gate_up_iq2_xxs(fg, fu, rows, a, st);
        case fmt::iq2_xs:
            return detail::gate_up_iq2_xs(fg, fu, rows, a, st);
        case fmt::iq2_s:
            return detail::gate_up_iq2_s(fg, fu, rows, a, st);
        case fmt::iq3_xxs:
            return detail::gate_up_iq3_xxs(fg, fu, rows, a, st);
        case fmt::iq3_s:
            return detail::gate_up_iq3_s(fg, fu, rows, a, st);
        case fmt::q8_0:
            return detail::gate_up_q8_0(fg, fu, rows, a, st);
        case fmt::q2_0: // ME7: the Q2 draft's routed gate/up
            return detail::gate_up_q2_0(fg, fu, rows, a, st);
        case fmt::q4_k:
            return detail::gate_up_q4_k(fg, fu, rows, a, st);
        case fmt::q5_k:
            return detail::gate_up_q5_k(fg, fu, rows, a, st);
        default:
            return false;
    }
}

bool write_meta(int32_t rows, const int32_t * ids, const class_table & cw, uint8_t * meta, void * stream) {
    if (rows < 1 || rows > max_rows || !aligned4(ids) || !aligned16(meta) || stream == nullptr || !class_ok(cw)) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1:
            meta_kernel<1><<<1, 32, 0, st>>>(ids, meta, cw);
            break;
        case 2:
            meta_kernel<2><<<1, 32, 0, st>>>(ids, meta, cw);
            break;
        case 3:
            meta_kernel<3><<<1, 32, 0, st>>>(ids, meta, cw);
            break;
        case 4:
            meta_kernel<4><<<1, 32, 0, st>>>(ids, meta, cw);
            break;
        default:
            meta_kernel<5><<<1, 32, 0, st>>>(ids, meta, cw);
            break;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

// EL1's fits (work/EL1/raw/el1.txt, class A), routed: 1 / 1.15 / 1.43 for 1 / 2 / 3 rows at IQ2_XXS and 1 / 1.15 /
// 1.32 at IQ3_S [A], IQ2_XXS's for the other routed formats [A], 4 rows extended linearly [H] and (SP2) 5 rows
// likewise [H]; shared: the fit's shared-unit term over its routed-row term at its point, 1.3 where EL1 has no fit [H]
class_table class_weights(fmt f, fn::dense::fmt fg, fn::dense::fmt fu) {
    using df = fn::dense::fmt;
    class_table cw = {{1.0f, 1.15f, 1.43f, 1.71f, 1.3f}};
    if (f == fmt::iq3_s) {
        cw.w[2] = 1.32f;
        cw.w[3] = 1.49f;
    }
    if (f == fmt::iq2_xxs && fg == df::iq4_xs && fu == df::iq3_s) {
        cw.w[4] = 1.398f; // 7.383 / 5.283
    } else if (f == fmt::iq2_xxs && fg == df::q4_k && fu == df::q4_k) {
        cw.w[4] = 1.033f; // 6.327 / 6.132
    } else if (f == fmt::iq2_xxs && fg == df::iq3_s && fu == df::iq3_s) {
        cw.w[4] = 1.610f; // 8.584 / 5.325
    } else if (f == fmt::iq3_s && fg == df::iq3_s && fu == df::iq3_s) {
        cw.w[4] = 1.334f; // 8.743 / 6.545
    }
    return cw;
}

void unit_order(const uint32_t * mask, int32_t n_u, const class_table & cw, uint16_t * order, int32_t rows) {
    const int32_t cap = cap_of(rows), nc = cap + 1; // SP2: the capacity's classes, the shared class last
    const float w5 = class_w5(cw);
    const auto weight = [&](int32_t a) { // class_w (experts_meta.cuh)
        return cap == 5 && a == cap ? cw.w[4] : cap == 5 && a == 4 ? w5 : cw.w[a];
    };
    const auto before = [&](int32_t a, int32_t b) { // class_before (experts_meta.cuh)
        const int32_t ta = a == nc - 1 ? 0 : a + 1, tb = b == nc - 1 ? 0 : b + 1;
        return weight(a) > weight(b) || (weight(a) == weight(b) && ta > tb);
    };
    int32_t cls[6] = {0, 1, 2, 3, 4, 5};
    std::stable_sort(cls, cls + nc, before);
    int32_t r = 0;
    for (int32_t ci = 0; ci < nc; ++ci) {
        const int32_t c = cls[ci];
        if (c == nc - 1) {
            for (int32_t g = 0; g < unit_groups; ++g) {
                order[r++] = static_cast<uint16_t>(order_shared_of(cap) + g);
            }
            continue;
        }
        for (int32_t u = 0; u < n_u; ++u) {
            if (__builtin_popcount(mask[u]) == c + 1) {
                for (int32_t g = 0; g < unit_groups; ++g) {
                    order[r++] = static_cast<uint16_t>(unit_groups * u + g);
                }
            }
        }
    }
}

bool down(fmt f, fmt fs, int32_t rows, const down_args & a, void * stream) {
    if (rows < 1 || rows > max_rows || !aligned16(a.w) || !aligned16(a.sh_w) || !aligned16(a.grids) ||
        !act_ok(a.glu, glu_slots, n_ff_card) || !aligned4(a.ids) || !aligned16(a.meta) || !aligned4(a.wts) ||
        !aligned4(a.sg) || !aligned4(a.out) || stream == nullptr || (a.status != nullptr && !aligned4(a.status)) ||
        !null_or_aligned(a.wt, 8) || (a.wt == nullptr && (a.ext_slots != 0 || a.ext_slots_d != nullptr)) ||
        (a.ext_slots >> (n_used * rows)) != 0 || !null_or_aligned(a.ext_slots_d, 8)) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
#define FN_EXPERTS_DN_CASE(rf, sf, layer)                                                                              \
    if (f == fmt::rf && fs == fmt::sf) {                                                                               \
        return down_rows<fmt::rf, fmt::sf>(rows, a, st);                                                               \
    }
    FN_EXPERTS_DOWN_INSTANCES(FN_EXPERTS_DN_CASE)
    FN_EXPERTS_Q4_DOWN_INSTANCES(FN_EXPERTS_DN_CASE) // ME1: the Q4 candidate's (Q8_0's entry is the deployed one's)
    FN_EXPERTS_DRAFT_Q2_DOWN_INSTANCES(FN_EXPERTS_DN_CASE) // ME7: the Q2 draft's
#undef FN_EXPERTS_DN_CASE
    return false;
}

namespace {

template <fmt F, fmt FS, int B, int NB> bool launch_down_lanes(const down_lanes_kargs & a, cudaStream_t st) {
    constexpr int32_t depth = F == fmt::q2_0 ? dn_depth_q2 : dn_depth_nl;
    constexpr int32_t smem = static_cast<int32_t>(sizeof(dn_smem<cap_of(B)>));
    static std::atomic<uint32_t> attr_done{0u};
    if (!detail::smem_attr_once(attr_done, reinterpret_cast<const void *>(down_kernel_lanes<F, FS, B, depth, NB>),
                                smem)) {
        return false;
    }
    down_kernel_lanes<F, FS, B, depth, NB><<<n_sm, dn_threads, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <fmt F, fmt FS, int B> bool down_lanes_b(int32_t lanes, const down_lanes_kargs & a, cudaStream_t st) {
    switch (lanes) {
        case 2:
            return launch_down_lanes<F, FS, B, n_sm / 2>(a, st);
        case 3:
            return launch_down_lanes<F, FS, B, n_sm / 3>(a, st);
        case 4:
            return launch_down_lanes<F, FS, B, n_sm / 4>(a, st);
        default:
            return false;
    }
}

// FMS13: the lone down launch's body at B = rows, the round's R
template <fmt F, fmt FS> bool down_lanes_of(int32_t lanes, int32_t rows, const down_lanes_kargs & a, cudaStream_t st) {
    switch (rows) {
        case 2:
            return down_lanes_b<F, FS, 2>(lanes, a, st);
        case 3:
            return down_lanes_b<F, FS, 3>(lanes, a, st);
        case 4:
            return down_lanes_b<F, FS, 4>(lanes, a, st);
        case 5:
            return down_lanes_b<F, FS, max_rows>(lanes, a, st);
        default:
            return false;
    }
}

// a lane's arguments as down takes them at B = rows on the resident walk: no table, no CPU slots
bool down_lane_ok(const down_args & a) {
    return aligned16(a.w) && aligned16(a.sh_w) && aligned16(a.grids) && act_ok(a.glu, glu_slots, n_ff_card) &&
           aligned4(a.ids) && aligned16(a.meta) && aligned4(a.wts) && aligned4(a.sg) && aligned4(a.out) &&
           (a.status == nullptr || aligned4(a.status)) && a.wt == nullptr && a.ext_slots == 0 &&
           a.ext_slots_d == nullptr;
}

bool resident_down(fmt f, fmt fs) {
    for (const dn_entry & e : dn_table) {
        if (e.f == f && e.fs == fs) {
            return f == fmt::q2_0 || f == fmt::iq4_nl;
        }
    }
    return false;
}

} // namespace

bool down_lanes_supported(fmt f, fmt fs, int32_t lanes, int32_t rows, const down_lanes_args & a) {
    if (lanes < 2 || lanes > max_lanes || rows < 2 || rows > max_rows || !resident_down(f, fs) ||
        dn_threads_of<fmt::q2_0>() != dn_threads || dn_threads_of<fmt::iq4_nl>() != dn_threads) {
        return false;
    }
    for (int32_t k = 0; k < lanes; ++k) {
        if (!down_lane_ok(a.s[k])) {
            return false;
        }
        for (int32_t q = 0; q < k; ++q) {
            const down_args & x = a.s[q];
            const down_args & y = a.s[k];
            if (x.out == y.out || x.meta == y.meta || x.glu.codes == y.glu.codes) {
                return false;
            }
        }
    }
    return true;
}

bool down_lanes(fmt f, fmt fs, int32_t lanes, int32_t rows, const down_lanes_args & a, void * stream) {
    return down_lanes_send(f, fs, lanes, rows, a, nullptr, 0, nullptr, stream);
}

bool down_lanes_send(fmt                     f,
                     fmt                     fs,
                     int32_t                 lanes,
                     int32_t                 rows,
                     const down_lanes_args & a,
                     uint32_t *              peer,
                     int32_t                 peer_half,
                     uint32_t *              x16_err,
                     void *                  stream) {
    if (stream == nullptr || !down_lanes_supported(f, fs, lanes, rows, a) ||
        (peer != nullptr && (!aligned16(peer) || (peer_half != 0 && peer_half != 1)))) {
        return false;
    }
    down_lanes_kargs k = {};
    // lane i's rows: joint rows 5 i .. 5 i + rows - 1 (the five-row lane stride whatever rows), F32 words_half words
    // a row, or 16-bit x16_row_halves halves
    const int32_t lane_words = max_rows * (x16_err != nullptr ? fn::xchg::x16_row_halves / 2 : fn::xchg::words_half);
    for (int32_t i = 0; i < max_lanes; ++i) {
        k.s[i] = {a.s[i], peer != nullptr && i < lanes ? peer + i * lane_words : nullptr};
    }
    k.peer_half = peer_half;
    k.x16_err = peer != nullptr ? x16_err : nullptr;
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
#define FN_EXPERTS_DN_CASE(rf, sf, layer)                               \
    if constexpr (fmt::rf == fmt::q2_0 || fmt::rf == fmt::iq4_nl) {     \
        if (f == fmt::rf && fs == fmt::sf) {                            \
            return down_lanes_of<fmt::rf, fmt::sf>(lanes, rows, k, st); \
        }                                                               \
    }
    FN_EXPERTS_DOWN_INSTANCES(FN_EXPERTS_DN_CASE)
#undef FN_EXPERTS_DN_CASE
    return false;
}

bool glu_finish(const float * gs, const float * us, int32_t n, int8_t * codes, float * scales, void * stream) {
    if (n < 1 || n > max_slots || !aligned4(gs) || !aligned4(us) || !aligned4(codes) || !aligned16(scales) ||
        stream == nullptr) {
        return false;
    }
    glu_finish_kernel<<<(n * n_group + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(gs, us, n, codes, scales);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool gate_up_inst(int32_t i, fmt & f, fn::dense::fmt & fg, fn::dense::fmt & fu, int32_t & first_layer) {
    if (i < 0 || i >= n_gate_up_inst) {
        return false;
    }
    f = gu_table[i].f;
    fg = gu_table[i].fg;
    fu = gu_table[i].fu;
    first_layer = gu_table[i].layer;
    return true;
}

bool down_inst(int32_t i, fmt & f, fmt & fs, int32_t & first_layer) {
    if (i < 0 || i >= n_down_inst) {
        return false;
    }
    f = dn_table[i].f;
    fs = dn_table[i].fs;
    first_layer = dn_table[i].layer;
    return true;
}

bool q4_gate_up_inst(int32_t i, fmt & f, fn::dense::fmt & fg, fn::dense::fmt & fu, int32_t & first_layer) {
    if (i < 0 || i >= n_q4_gate_up_inst) {
        return false;
    }
    f = q4_gu_table[i].f;
    fg = q4_gu_table[i].fg;
    fu = q4_gu_table[i].fu;
    first_layer = q4_gu_table[i].layer;
    return true;
}

bool q4_down_inst(int32_t i, fmt & f, fmt & fs, int32_t & first_layer) {
    if (i < 0 || i >= n_q4_down_inst) {
        return false;
    }
    f = q4_dn_table[i].f;
    fs = q4_dn_table[i].fs;
    first_layer = q4_dn_table[i].layer;
    return true;
}

bool draft_q2_gate_up_inst(int32_t i, fmt & f, fn::dense::fmt & fg, fn::dense::fmt & fu, int32_t & first_layer) {
    if (i < 0 || i >= n_draft_q2_gate_up_inst) {
        return false;
    }
    f = q2d_gu_table[i].f;
    fg = q2d_gu_table[i].fg;
    fu = q2d_gu_table[i].fu;
    first_layer = q2d_gu_table[i].layer;
    return true;
}

bool draft_q2_down_inst(int32_t i, fmt & f, fmt & fs, int32_t & first_layer) {
    if (i < 0 || i >= n_draft_q2_down_inst) {
        return false;
    }
    f = q2d_dn_table[i].f;
    fs = q2d_dn_table[i].fs;
    first_layer = q2d_dn_table[i].layer;
    return true;
}

} // namespace fn::experts
