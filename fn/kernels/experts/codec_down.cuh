#pragma once

// TTTS3 item 2: the experts' routed and shared down formats, one codec a format (Amendment 1: plain structs of static
// constexpr constants and static __forceinline__ device functions, one explicit specialization a format, written out
// in full). The gate/up codecs are codec.cuh's; this file is the down kernels' (experts.cu) alone: in codec.cuh, read by
// every gate/up translation unit, it changed one IQ3 gate/up instance's SASS (TTTS3 sass/c1).

#include "fn/kernels/experts/experts_dev.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace fn::experts {
namespace {

// ---- down codecs ------------------------------------------------------------------------------------------------
// dn_codec<F>, a routed (or shared) down format: an item's chunks and d (dn_slot: chunks 16-byte loads and the d
// entry), the expert's packed bytes, load (an item's codes at pc, chunk c at pc + c x D x 16, an immediate offset, and
// its d at pd), item (one output row n's dot with columns 64 j5 .. 64 j5 + 63 of one expert, for each row b in mask,
// their slots consecutive from slot, into part[slot][j5][n]; ALL: mask is every row's, the shared expert) and the walk
// down_kernel runs it on: strided (71b92a9's thread-strided walk at threads = 512) or the fixed walk at 544, with
// depth items in flight a thread. transposed: the staged codes are read transposed (Q2_0's 2-bit fields); kvalues: the
// item reads kvalues_iq4nl.
constexpr int32_t act_stride = 336;  // bytes per staged slot: 320 + 16, 21 x 16 B (odd), so two slots never share a

                                     // bank group in one 16-byte read

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
    uint4                                               c[down_chunks(F)];
    // FP16 d (Q2_0: low half; IQ4_NL and Q8_0: both blocks); ME1, Q5_1: the blocks' qh words, then d0 m0, d1 m1
    std::conditional_t<F == fmt::q5_1, uint4, uint32_t> d;
};

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

template <fmt F> struct dn_codec;

// IQ4_NL (the deployed file's down): block j10's 16 nibble bytes a chunk, two blocks an item, FP16 d of both
template <> struct dn_codec<fmt::iq4_nl> {
    static constexpr fmt     f      = fmt::iq4_nl;
    static constexpr int32_t chunks = 2, d_bytes = 4, threads = 544, depth = 2;
    static constexpr int64_t expert_bytes = 460800;
    static constexpr bool    strided = false, transposed = false, kvalues = true;
    static constexpr bool    lanes = true;  // down_kernel_lanes takes it (down_lanes, down_lanes_send)
    using slot_t                   = dn_slot<f>;

    static __device__ __forceinline__ slot_t load(const uint8_t * pc, const uint8_t * pd, uint32_t z) {
        slot_t v;
#pragma unroll
        for (int32_t c = 0; c < chunks; ++c) {
            v.c[c] = __ldcs(reinterpret_cast<const uint4 *>(pc + c * D * 16 + z));
        }
        v.d = __ldcs(reinterpret_cast<const uint32_t *>(pd + z));
        return v;
    }

    template <int B, bool ALL, typename S>
    static __device__ __forceinline__ void
    item(S & s, const slot_t & v, int32_t j5, int32_t n, uint32_t mask, int32_t slot, const uint32_t (&kv)[4]) {
        const int32_t kb = 2 * j5;  // the 32-value runs of this item
        // block kb + c's weights 0..15 (lo) and 16..31 (hi): the table values of its nibbles
        uint32_t      lo[8], hi[8];
#pragma unroll
        for (int32_t c = 0; c < 2; ++c) {
#pragma unroll
            for (int32_t wi = 0; wi < 4; ++wi) {
                table16(word_of(v.c[c], wi), kv, lo[4 * c + wi], hi[4 * c + wi]);
            }
        }
        const float d0 = half_lo(v.d), d1 = half_hi(v.d);
        dn_rows<B, ALL>(mask, slot, [&](int32_t slot) {
            const uint4 * ap    = reinterpret_cast<const uint4 *>(s.act + slot * act_stride + j5 * 64);
            int32_t       is[2] = { 0, 0 };
#pragma unroll
            for (int32_t c = 0; c < 2; ++c) {
                const uint4 a0 = ap[2 * c], a1 = ap[2 * c + 1];  // elements 0..15, 16..31 of block kb + c
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
};

// Q2_0 (the deployed file's and the Q2 draft's): one chunk, the block's 2-bit codes; FP16 d (the low half of the
// loaded word)
template <> struct dn_codec<fmt::q2_0> {
    static constexpr fmt     f      = fmt::q2_0;
    static constexpr int32_t chunks = 1, d_bytes = 2, threads = 544, depth = 3;  // 540 x 3 x 18 B = 28 KB an SM
    static constexpr int64_t expert_bytes = 230400;
    static constexpr bool    strided = false, transposed = true, kvalues = false;
    static constexpr bool    lanes = true;  // down_kernel_lanes takes it (down_lanes, down_lanes_send)
    using slot_t                   = dn_slot<f>;

    static __device__ __forceinline__ slot_t load(const uint8_t * pc, const uint8_t * pd, uint32_t z) {
        slot_t v;
#pragma unroll
        for (int32_t c = 0; c < chunks; ++c) {
            v.c[c] = __ldcs(reinterpret_cast<const uint4 *>(pc + c * D * 16 + z));
        }
        v.d = __ldcs(reinterpret_cast<const unsigned short *>(pd + z));
        return v;
    }

    template <int B, bool ALL, typename S>
    static __device__ __forceinline__ void item(S &            s,
                                                const slot_t & v,
                                                int32_t        j5,
                                                int32_t        n,
                                                uint32_t       mask,
                                                int32_t        slot,
                                                [[maybe_unused]] const uint32_t (&kv)[4]) {
        const int32_t kb = 2 * j5;  // the 32-value runs of this item
        uint32_t      x[16];        // x[4 i + s] = 2-bit field s of each byte of word i
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
            int32_t       i0 = 0, i1 = 0;
#pragma unroll
            for (int32_t wi = 0; wi < 4; ++wi) {
                const uint4 av  = ap[wi];
                int32_t &   acc = wi < 2 ? i0 : i1;
                acc             = __dp4a(static_cast<int32_t>(x[4 * wi + 0]), static_cast<int32_t>(av.x), acc);
                acc             = __dp4a(static_cast<int32_t>(x[4 * wi + 1]), static_cast<int32_t>(av.y), acc);
                acc             = __dp4a(static_cast<int32_t>(x[4 * wi + 2]), static_cast<int32_t>(av.z), acc);
                acc             = __dp4a(static_cast<int32_t>(x[4 * wi + 3]), static_cast<int32_t>(av.w), acc);
            }
            const float dot     = dw * (s.da[slot][kb] * static_cast<float>(i0 - s.asum[slot][kb]) +
                                        s.da[slot][kb + 1] * static_cast<float>(i1 - s.asum[slot][kb + 1]));
            s.part[slot][j5][n] = dot;
        });
    }
};

// Q8_0 (DR1, the draft's; ME1, the Q4 file's layers 2, 4, 30, 46, 47): four chunks, block 2 j5 + c / 2's int8
// weights 16 (c & 1) ..; FP16 d of both blocks. 71b92a9's thread-strided walk at 512 threads: at 544 the cap is 96
// registers (5 warps on one of the SM's four 16K-register partitions), and K5 spills 48 bytes there (XB4)
template <> struct dn_codec<fmt::q8_0> {
    static constexpr fmt     f      = fmt::q8_0;
    static constexpr int32_t chunks = 4, d_bytes = 4, threads = 512, depth = 2;  // 512 x 2 x 68 B = 68 KB an SM
    static constexpr int64_t expert_bytes = 870400;
    static constexpr bool    strided = true, transposed = false, kvalues = false;
    static constexpr bool    lanes = true;  // down_kernel_lanes takes it (down_lanes, down_lanes_send; TTTS3 item 4)
    using slot_t                   = dn_slot<f>;

    static __device__ __forceinline__ slot_t load(const uint8_t * pc, const uint8_t * pd, uint32_t z) {
        slot_t v;
#pragma unroll
        for (int32_t c = 0; c < chunks; ++c) {
            v.c[c] = __ldcs(reinterpret_cast<const uint4 *>(pc + c * D * 16 + z));
        }
        v.d = __ldcs(reinterpret_cast<const uint32_t *>(pd + z));
        return v;
    }

    template <int B, bool ALL, typename S>
    static __device__ __forceinline__ void item(S &            s,
                                                const slot_t & v,
                                                int32_t        j5,
                                                int32_t        n,
                                                uint32_t       mask,
                                                int32_t        slot,
                                                [[maybe_unused]] const uint32_t (&kv)[4]) {
        const int32_t kb = 2 * j5;  // the 32-value runs of this item
        // block kb + c's weights 0..15 (lo) and 16..31 (hi): its own int8
        uint32_t      lo[8], hi[8];
#pragma unroll
        for (int32_t c = 0; c < 2; ++c) {
#pragma unroll
            for (int32_t wi = 0; wi < 4; ++wi) {
                lo[4 * c + wi] = word_of(v.c[2 * c], wi);
                hi[4 * c + wi] = word_of(v.c[2 * c + 1], wi);
            }
        }
        const float d0 = half_lo(v.d), d1 = half_hi(v.d);
        dn_rows<B, ALL>(mask, slot, [&](int32_t slot) {
            const uint4 * ap    = reinterpret_cast<const uint4 *>(s.act + slot * act_stride + j5 * 64);
            int32_t       is[2] = { 0, 0 };
#pragma unroll
            for (int32_t c = 0; c < 2; ++c) {
                const uint4 a0 = ap[2 * c], a1 = ap[2 * c + 1];  // elements 0..15, 16..31 of block kb + c
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
};

// Q5_1 (ME1, the Q4 file's down in 43 layers): chunks as IQ4_NL's (byte t = weight t low and t + 16 high), then 16 B
// a row: the two blocks' qh words, FP16 d0, m0, d1, m1
template <> struct dn_codec<fmt::q5_1> {
    static constexpr fmt     f      = fmt::q5_1;
    static constexpr int32_t chunks = 2, d_bytes = 16, threads = 544, depth = 2;
    static constexpr int64_t expert_bytes = 614400;
    static constexpr bool    strided = false, transposed = false, kvalues = false;
    static constexpr bool    lanes = true;  // down_kernel_lanes takes it (down_lanes, down_lanes_send; TTTS3 item 4)
    using slot_t                   = dn_slot<f>;

    static __device__ __forceinline__ slot_t load(const uint8_t * pc, const uint8_t * pd, uint32_t z) {
        slot_t v;
#pragma unroll
        for (int32_t c = 0; c < chunks; ++c) {
            v.c[c] = __ldcs(reinterpret_cast<const uint4 *>(pc + c * D * 16 + z));
        }
        v.d = __ldcs(reinterpret_cast<const uint4 *>(pd + z));
        return v;
    }

    template <int B, bool ALL, typename S>
    static __device__ __forceinline__ void item(S &            s,
                                                const slot_t & v,
                                                int32_t        j5,
                                                int32_t        n,
                                                uint32_t       mask,
                                                int32_t        slot,
                                                [[maybe_unused]] const uint32_t (&kv)[4]) {
        const int32_t kb = 2 * j5;  // the 32-value runs of this item
        // ME1: block kb + c's weights 0..15 (lo) and 16..31 (hi), 0..31 with bit 4 from the block's qh word; row b's
        // partial is (d_c da_c) is_c + (m_c da_c) asum_c a block (asum: the run's exact code sum), one fused rounding
        // each, then the two blocks' sum
        uint32_t      lo[8], hi[8];
#pragma unroll
        for (int32_t c = 0; c < 2; ++c) {
            const uint32_t qh = c == 0 ? v.d.x : v.d.y;
#pragma unroll
            for (int32_t wi = 0; wi < 4; ++wi) {
                const uint32_t x = word_of(v.c[c], wi);
                lo[4 * c + wi]   = (x & 0x0f0f0f0fu) | (spread4(qh >> (4 * wi)) << 4);
                hi[4 * c + wi]   = ((x >> 4) & 0x0f0f0f0fu) | (spread4(qh >> (16 + 4 * wi)) << 4);
            }
        }
        const float d0 = half_lo(v.d.z), m0 = half_hi(v.d.z), d1 = half_lo(v.d.w), m1 = half_hi(v.d.w);
        dn_rows<B, ALL>(mask, slot, [&](int32_t slot) {
            const uint4 * ap    = reinterpret_cast<const uint4 *>(s.act + slot * act_stride + j5 * 64);
            int32_t       is[2] = { 0, 0 };
#pragma unroll
            for (int32_t c = 0; c < 2; ++c) {
                const uint4 a0 = ap[2 * c], a1 = ap[2 * c + 1];  // elements 0..15, 16..31 of block kb + c
#pragma unroll
                for (int32_t wi = 0; wi < 4; ++wi) {
                    is[c] = __dp4a(static_cast<int32_t>(lo[4 * c + wi]), static_cast<int32_t>(word_of(a0, wi)), is[c]);
                    is[c] = __dp4a(static_cast<int32_t>(hi[4 * c + wi]), static_cast<int32_t>(word_of(a1, wi)), is[c]);
                }
            }
            const float da0 = s.da[slot][kb], da1 = s.da[slot][kb + 1];
            const float t0      = __fmaf_rn(__fmul_rn(d0, da0), static_cast<float>(is[0]),
                                            __fmul_rn(__fmul_rn(m0, da0), static_cast<float>(s.asum[slot][kb])));
            const float t1      = __fmaf_rn(__fmul_rn(d1, da1), static_cast<float>(is[1]),
                                            __fmul_rn(__fmul_rn(m1, da1), static_cast<float>(s.asum[slot][kb + 1])));
            s.part[slot][j5][n] = __fadd_rn(t0, t1);
        });
    }
};

template <fmt F> constexpr bool dn_codec_ok() {
    using C = dn_codec<F>;
    return C::chunks == down_chunks(F) && C::d_bytes == down_d_bytes(F) &&
           C::expert_bytes == int64_t(n_item_k) * (C::chunks * 16 + C::d_bytes) * D;
}

static_assert(dn_codec_ok<fmt::iq4_nl>() && dn_codec_ok<fmt::q2_0>() && dn_codec_ok<fmt::q8_0>() &&
                  dn_codec_ok<fmt::q5_1>(),
              "the down codecs' bytes are the packed layout's");

// routed down format f has a lanes form (its codec's lanes), for the host's checks: one switch over the codecs
constexpr bool dn_lanes(fmt f) {
    switch (f) {
        case fmt::iq4_nl:
            return dn_codec<fmt::iq4_nl>::lanes;
        case fmt::q2_0:
            return dn_codec<fmt::q2_0>::lanes;
        case fmt::q8_0:
            return dn_codec<fmt::q8_0>::lanes;
        case fmt::q5_1:
            return dn_codec<fmt::q5_1>::lanes;
        default:
            return false;
    }
}

}  // namespace
}  // namespace fn::experts
