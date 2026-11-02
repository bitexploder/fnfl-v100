#pragma once

// APF2: the dense ladder's kernel template and tile loaders (device and the launch helpers; dense_ladder.h has the
// contract). Instantiated by dense_ladder.cu (Q4's inventory: Q8_0 DP2, Q8_0 EX1 down, BF16 rows) and
// dense_ladder_iq.cu (the IQ3 inventory's other formats).

#include "fn/kernels/dense/dense_ladder.h"
#include "fn/kernels/dense/dense_prompt_dec.cuh"
#include "fn/kernels/tc/tc_ladder.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace fn::dense {
namespace pk {

using namespace pdc;  // the decode helpers and pd_tc's epilogue (dense_prompt_dec.cuh)

// the loader of a ladder kernel's B operand
enum class pl_src : int32_t { bf16, q8_0, q8_0_ex1, q4_k, q5_k, q6_k, iq4_xs, iq3_s, iq4_nl_ex1, q2_0_ex1, count };

struct pl_kargs {
    pt_kargs         e;  // the activation operand, rows, n, k and pd_tc's epilogue
    const uint8_t *  w[3];
    int32_t          col0[4];  // source s's first column; col0[n_src] = n
    int32_t          n_src;
    int32_t          kb;       // DP2: 256-weight steps a row (K / 256)
    int32_t          rb8;      // DP2: a tile's bytes a step
    const uint8_t *  grids;
    const uint16_t * b0;
    const uint16_t * b1;
    int32_t          n0, n1;
    const float *    b_scale;
    int32_t          splits;
    fn::tc::lsplit   ws;
};

// the packed source of column n: its pointer and the column within it, by selects (no dynamic parameter index)
__device__ __forceinline__ const uint8_t * pl_source(const pl_kargs & k, int32_t n, int32_t & nl) {
    const bool s1 = k.n_src > 1 && n >= k.col0[1];
    const bool s2 = k.n_src > 2 && n >= k.col0[2];
    nl            = n - (s2 ? k.col0[2] : s1 ? k.col0[1] : 0);
    return s2 ? k.w[2] : s1 ? k.w[1] : k.w[0];
}

__device__ __forceinline__ uint32_t pl_hmul2(uint32_t a, uint32_t b) {
    const half2 r = __hmul2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r);  // reinterpret: half2 is the word's two halves
}

// FP16 code pairs of a word's bytes (biased by 128: signed codes), times d2 (FP16 d in both halves): one rounding of
// the exact product, so pd_convert's weights16 value (f x code in F32, then FP16) for every Q8_0 weight
__device__ __forceinline__ uint2 q8_halves(uint32_t w, uint32_t d2) {
    uint32_t lo, hi;
    codes16(w ^ 0x80808080u, bias_s, lo, hi);
    return make_uint2(pl_hmul2(lo, d2), pl_hmul2(hi, d2));
}

// Q8_0, DP2's layout (dense_dec.cuh): column n's K k .. k + 7 at step k / 256 of its 8-column tile: lane
// li = 4 (n % 8) + q (q the 64-weight quarter) holds blocks 2q, 2q + 1 as four 16-byte chunks j at 512 j + 16 li, and
// their FP16 d at 2048 + 4 li + 2 (j / 2). The stage's chunks as tc_ladder.cuh's b_chunks maps them (thread t: chunk
// t + i T, column ci / 4, 8-value chunk ci % 4 of the 32-K stage); codes loaded at fetch, decoded at store
template <int32_t BN, int32_t T> struct pl_b_q8 {
    static constexpr int32_t IT = BN * 4 / T;
    const pl_kargs &         k;

    struct raw {
        uint2    c[IT];
        uint32_t d[IT];
    };

    __device__ __forceinline__ void fetch(raw & r, int32_t col0, int32_t k0) const {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
        for (int32_t i = 0; i < IT; ++i) {
            const int32_t ci = tid + i * T, n = col0 + (ci >> 2), kk = k0 + 8 * (ci & 3);
            if (n < k.e.p.n) {
                int32_t         nl  = 0;
                const uint8_t * w   = pl_source(k, n, nl);
                const int32_t   in  = kk & 255, q = in >> 6, j = (in >> 4) & 3, li = 4 * (nl & 7) + q;
                const uint8_t * stp = w + (int64_t(nl >> 3) * k.kb + (kk >> 8)) * k.rb8;
                r.c[i]              = __ldg(reinterpret_cast<const uint2 *>(stp + 512 * j + 16 * li + 8 * ((in >> 3) & 1)));
                r.d[i] = __ldg(reinterpret_cast<const unsigned short *>(stp + 2048 + 4 * li + 2 * (j >> 1)));
            } else {
                r.c[i] = make_uint2(0u, 0u);
                r.d[i] = 0u;
            }
        }
    }

    __device__ __forceinline__ void store(uint32_t sb, const raw & r) const {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
        for (int32_t i = 0; i < IT; ++i) {
            const int32_t  ci = tid + i * T;
            const uint32_t d2 = r.d[i] | (r.d[i] << 16);
            const uint2    a = q8_halves(r.c[i].x, d2), b = q8_halves(r.c[i].y, d2);
            fn::tc::detail::sts128(sb + fn::tc::lsw(ci >> 2, ci & 3), make_uint4(a.x, a.y, b.x, b.y));
        }
    }
};

// Q8_0, EX1's down layout (experts.h pack_down, the shared expert's down: K 320, N 2560): stage s (64 weights) of
// column n as four 16-weight chunks at ((4 s + c) 2560 + n) 16, the two blocks' FP16 d at the d plane's (s 2560 + n) 4
template <int32_t BN, int32_t T> struct pl_b_q8x {
    static constexpr int32_t IT = BN * 4 / T;
    const pl_kargs &         k;

    struct raw {
        uint2    c[IT];
        uint32_t d[IT];
    };

    __device__ __forceinline__ void fetch(raw & r, int32_t col0, int32_t k0) const {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
        for (int32_t i = 0; i < IT; ++i) {
            const int32_t ci = tid + i * T, n = col0 + (ci >> 2), kk = k0 + 8 * (ci & 3);
            if (n < k.e.p.n) {
                const int32_t     s = kk >> 6, in = kk & 63;
                constexpr int64_t d_plane = int64_t(20) * ex1_n * 16;
                r.c[i] = __ldg(reinterpret_cast<const uint2 *>(k.w[0] + (int64_t(4 * s + (in >> 4)) * ex1_n + n) * 16 +
                                                               8 * ((in >> 3) & 1)));
                r.d[i] = __ldg(reinterpret_cast<const unsigned short *>(k.w[0] + d_plane +
                                                                        (int64_t(s) * ex1_n + n) * 4 + 2 * (in >> 5)));
            } else {
                r.c[i] = make_uint2(0u, 0u);
                r.d[i] = 0u;
            }
        }
    }

    __device__ __forceinline__ void store(uint32_t sb, const raw & r) const {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
        for (int32_t i = 0; i < IT; ++i) {
            const int32_t  ci = tid + i * T;
            const uint32_t d2 = r.d[i] | (r.d[i] << 16);
            const uint2    a = q8_halves(r.c[i].x, d2), b = q8_halves(r.c[i].y, d2);
            fn::tc::detail::sts128(sb + fn::tc::lsw(ci >> 2, ci & 3), make_uint4(a.x, a.y, b.x, b.y));
        }
    }
};

// BF16 rows from two bases (rows [0, n0) of b0, [n0, n0 + n1) of b1, zero past them), each value times its row's
// 2^e (the inverse of the prepared 2^-e, exact) rounded once: pd_convert_bf16's FP16 value
template <int32_t BN, int32_t T> struct pl_b_bf16 {
    static constexpr int32_t IT = BN * 4 / T;
    const pl_kargs &         k;

    struct raw {
        uint4 v[IT];
        float sc[IT];
    };

    __device__ __forceinline__ void fetch(raw & r, int32_t col0, int32_t k0) const {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
        for (int32_t i = 0; i < IT; ++i) {
            const int32_t    ci  = tid + i * T, n = col0 + (ci >> 2), kk = k0 + 8 * (ci & 3);
            const uint16_t * row = n < k.n0        ? k.b0 + int64_t(n) * k.e.p.k :
                                   n < k.n0 + k.n1 ? k.b1 + int64_t(n - k.n0) * k.e.p.k :
                                                     nullptr;
            r.v[i]  = row != nullptr ? __ldg(reinterpret_cast<const uint4 *>(row + kk)) : make_uint4(0u, 0u, 0u, 0u);
            r.sc[i] = row != nullptr ? 1.0f / __ldg(k.b_scale + n) : 0.0f;
        }
    }

    __device__ __forceinline__ void store(uint32_t sb, const raw & r) const {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
        const auto    f   = [](uint32_t h) {
            return __uint_as_float(h << 16);
        };
#pragma unroll
        for (int32_t i = 0; i < IT; ++i) {
            const int32_t ci = tid + i * T;
            const float   sc = r.sc[i];
            const uint4 & v  = r.v[i];
            uint4         o;
            o.x = fn::tc::pack2(f(v.x & 0xffffu) * sc, f(v.x >> 16) * sc);
            o.y = fn::tc::pack2(f(v.y & 0xffffu) * sc, f(v.y >> 16) * sc);
            o.z = fn::tc::pack2(f(v.z & 0xffffu) * sc, f(v.z >> 16) * sc);
            o.w = fn::tc::pack2(f(v.w & 0xffffu) * sc, f(v.w >> 16) * sc);
            fn::tc::detail::sts128(sb + fn::tc::lsw(ci >> 2, ci & 3), o);
        }
    }
};

// One weight's FP16 value from a decoded word's byte: the code (signed formats biased by 128 in the word) times f,
// minus m, rounded once (weights16's arithmetic for one value)
template <fmt F> __device__ __forceinline__ uint32_t pl_one(uint32_t word, int32_t byte, float f, float m) {
    const int32_t u    = static_cast<int32_t>((word >> (8 * byte)) & 0xffu);
    const float   code = static_cast<float>(is_signed_code<F>() ? u - 128 : u);
    return __half_as_ushort(__float2half_rn(__fsub_rn(__fmul_rn(f, code), m)));
}

// The other formats (the IQ3 inventory): the 8 values at K k of column n, decoded by the device decoder's quarter
// (DP2) or stage (EX1) walk and picked at their K places (k_of). Simple and correct, not fast: each chunk decodes its
// whole quarter (stage); the decode is in fetch, so no load overlaps the products
template <fmt F, pd_layout L> struct pl_chunk_dec {
    const pl_kargs &   k;
    const uint32_t *   grid_s;  // IQ3_S's grid in shared memory
    uint4              kv;      // IQ4's values

    __device__ __forceinline__ uint4 operator()(int32_t n, int32_t kk) const {
        if (n >= k.e.p.n) {
            return make_uint4(0u, 0u, 0u, 0u);
        }
        quarter_codes qc;
        uint32_t      h[8];
        if constexpr (L == pd_layout::dp2) {
            int32_t         nl  = 0;
            const uint8_t * w   = pl_source(k, n, nl);
            const uint8_t * stp = w + (int64_t(nl >> 3) * k.kb + (kk >> 8)) * k.rb8;
            const int32_t   in  = kk & 255;
            int32_t         q = 0, x0 = 0;
            if constexpr (F == fmt::q6_k) {
                const int32_t p = in & 127;  // the 128-weight stage: quarters 2 (in / 128) + qq
                q               = 2 * (in >> 7) + ((p >> 4) & 1);
                x0              = 4 * (p >> 5) + ((p & 15) >> 2);
            } else {
                q  = in >> 6;
                x0 = (in & 63) >> 2;
            }
            dp2_quarter<F>(stp, nl & 7, q, kv, grid_s, qc);
            constexpr int32_t GW = group_words<F>();
#pragma unroll
            for (int32_t e = 0; e < 8; ++e) {
                const int32_t x = x0 + (e >> 2);
                uint32_t      wv = 0u;
                float         f = 0.0f, m = 0.0f;
#pragma unroll
                for (int32_t xx = 0; xx < 16; ++xx) {  // by selects: a dynamic index would spill qc to the stack
                    if (xx == x) {
                        wv = qc.w[xx];
                        f  = qc.f[xx / GW];
                        m  = qc.m[xx / GW];
                    }
                }
                h[e] = pl_one<F>(wv, e & 3, f, m);
            }
        } else {
            const int32_t s = kk >> 6, in = kk & 63;
            ex1_stage_codes<F>(k.w[0], n, s, kv, qc);
#pragma unroll
            for (int32_t e = 0; e < 8; ++e) {
                const int32_t ko  = in + e;  // the original K in the stage, to the decoder's place (stage_a's map)
                int32_t       pos = ko;
                if constexpr (F == fmt::iq4_nl) {
                    const int32_t bb = ko >> 5, kq = ko & 31, lo = kq >= 16 ? 1 : 0, k1 = kq & 15;
                    const int32_t c = k1 >> 2, rem = k1 & 3;
                    pos = 32 * bb + 4 * (2 * c + (rem >> 1)) + 2 * (rem & 1) + lo;
                }
                const int32_t x  = pos >> 2;
                uint32_t      wv = 0u;
                float         f = 0.0f, m = 0.0f;
#pragma unroll
                for (int32_t xx = 0; xx < 16; ++xx) {
                    if (xx == x) {
                        wv = qc.w[xx];
                        f  = qc.f[xx / 8];
                        m  = qc.m[xx / 8];
                    }
                }
                h[e] = pl_one<F>(wv, pos & 3, f, m);
            }
        }
        return make_uint4(h[0] | (h[1] << 16), h[2] | (h[3] << 16), h[4] | (h[5] << 16), h[6] | (h[7] << 16));
    }
};

template <int32_t BM, int32_t BN, pl_src S> __device__ __forceinline__ void pl_body(const pl_kargs & k) {
    using sh = fn::tc::ltile<BM, BN>;
    extern __shared__ uint4 pl_sm[];
    fn::tc::a_plain al;
    al.a          = k.e.p.a;
    al.rows       = k.e.p.rows;
    const auto epi = [&](int32_t row, int32_t c0, const float (&v)[4]) { pt_epi(k.e, row, c0, v); };
    uint8_t * const sm = reinterpret_cast<uint8_t *>(pl_sm);
    if constexpr (S == pl_src::q8_0) {
        fn::tc::lgemm<BM, BN>(k.e.p.rows, k.e.p.k, k.splits, k.ws, sm, al, pl_b_q8<BN, sh::threads>{ k }, epi);
    } else if constexpr (S == pl_src::q8_0_ex1) {
        fn::tc::lgemm<BM, BN>(k.e.p.rows, k.e.p.k, k.splits, k.ws, sm, al, pl_b_q8x<BN, sh::threads>{ k }, epi);
    } else if constexpr (S == pl_src::bf16) {
        fn::tc::lgemm<BM, BN>(k.e.p.rows, k.e.p.k, k.splits, k.ws, sm, al, pl_b_bf16<BN, sh::threads>{ k }, epi);
    } else {
        constexpr fmt F = S == pl_src::q4_k       ? fmt::q4_k :
                          S == pl_src::q5_k       ? fmt::q5_k :
                          S == pl_src::q6_k       ? fmt::q6_k :
                          S == pl_src::iq4_xs     ? fmt::iq4_xs :
                          S == pl_src::iq3_s      ? fmt::iq3_s :
                          S == pl_src::iq4_nl_ex1 ? fmt::iq4_nl :
                                                    fmt::q2_0;
        constexpr pd_layout L =
            S == pl_src::iq4_nl_ex1 || S == pl_src::q2_0_ex1 ? pd_layout::ex1_down : pd_layout::dp2;
        __shared__ uint32_t grid_s[F == fmt::iq3_s ? 512 : 1];
        if constexpr (F == fmt::iq3_s) {
            for (int32_t i = static_cast<int32_t>(threadIdx.x); i < 512; i += sh::threads) {
                grid_s[i] = __ldg(reinterpret_cast<const uint32_t *>(k.grids) + i);
            }
            __syncthreads();
        }
        uint4 kv = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (F == fmt::iq4_xs || F == fmt::iq4_nl) {
            kv = __ldg(reinterpret_cast<const uint4 *>(k.grids + 2048));
        }
        const fn::tc::b_chunks<BN, sh::threads, pl_chunk_dec<F, L>> bl{ { k, grid_s, kv } };
        fn::tc::lgemm<BM, BN>(k.e.p.rows, k.e.p.k, k.splits, k.ws, sm, al, bl, epi);
    }
}

template <int32_t BM, int32_t BN, pl_src S>
__global__ void __launch_bounds__(fn::tc::ltile<BM, BN>::threads) pl_kernel(const __grid_constant__ pl_kargs k) {
    pl_body<BM, BN, S>(k);
}

// run(kernel, ltile) for tile index t (pl_tiles' order) with the kernel instance of source S
template <pl_src S, typename Fn> bool pl_with_tile(int32_t t, Fn && run) {
    return fn::tc::lad_with_tile(t, [&](auto sh) {
        using T = decltype(sh);
        return run(pl_kernel<T::bm, T::bn, S>, sh);
    });
}

// launch / opt-in / resources of source S's instances on tile t (dense_ladder.cu defines them for Q4's sources,
// dense_ladder_iq.cu for the others; each returns false for a source it does not hold)
bool pl_launch_q4(const pl_kargs & k, pl_src src, int32_t t, cudaStream_t s);
bool pl_launch_iq(const pl_kargs & k, pl_src src, int32_t t, cudaStream_t s);
bool pl_attr_q4();
bool pl_attr_iq();
bool pl_res_q4(pl_src src, int32_t t, pd_resources & r);
bool pl_res_iq(pl_src src, int32_t t, pd_resources & r);

}  // namespace pk
}  // namespace fn::dense
