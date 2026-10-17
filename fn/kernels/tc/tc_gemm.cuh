#pragma once

// PB3: the prompt path's Volta tensor-core product core (device only), shared by the dense, HC and router suppliers.
//   math:     out[r][n] = sum_k a[r][k] b[n][k], a and b FP16, handed to an epilogue functor in F32.
//   numerics: mma.m8n8k4 (HMMA.884): FP16 products, F32 accumulators (PB3 Amendment 2: FP16 operands with F32
//             accumulation are the prompt products' default). K in order: a rerun is bitwise; rows are independent
//             (no split K).
//   shapes:   a CTA 32 MI rows x BN columns, two warps down and BN / (16 NI) across, a warp 16 MI rows x 16 NI
//             columns: MI m-tiles of 16 rows x NI n-tiles of 16 columns (NI 2: BN 64 is 4 warps, 128 is 8; PB3's NI 4,
//             the 64 x 64 warp tile: BN 128 is 4 warps), quadpair q taking rows 8 (q & 1) .. + 7 and columns 8 (q >> 1)
//             .. + 7 of each, so no two quadpairs compute from the same pair of fragments. K a stage 32 (K % 32 == 0),
//             two stages in shared memory, the next stage's global reads held in registers across the current
//             stage's products.
//   layout:   shared rows of 32 halves (four 16-byte chunks), chunk c of row x stored at chunk c ^ f(x),
//             f(x) = ((x >> 1) & 1) | ((x >> 2) & 2). A fragment read is 16 bytes (two k4 steps); each 8-lane phase
//             reads rows {0..3, 8..11} or {4..7, 12..15} of a 16-row tile (A), or 4 columns (B), which f places in
//             distinct bank groups.
//   budget:   accumulators 8 MI NI F32 a thread, fragments MI + NI uint4, staging uint4s; checked by ptxas in each
//             instantiating family.
// Rows past `rows` read zeros and are not handed to the epilogue; columns past `n` read zeros.

#include <cuda_fp16.h>

#include <cstdint>

namespace fn::tc {

constexpr int32_t bk = 32;  // K a stage

// FP16 range handling (PB3 Amendment 1, used by the HC and BF16 conversions): a row scaled by the power of two that
// brings its largest magnitude into [2^14, 2^15), so no value overflows FP16; the epilogue multiplies by 2^-e.
// The exponent e for magnitude m >= 0, held in [-100, 100] so 2^e and 2^-e stay normal F32 (0 for m = 0).
__device__ __forceinline__ int32_t range_exp(float m) {
    int32_t x = 0;
    (void) frexpf(m, &x);
    return m > 0.0f ? max(-100, min(100, 15 - x)) : 0;
}

// 2^e for e in [-126, 127], exactly
__device__ __forceinline__ float pow2(int32_t e) {
    return __int_as_float((e + 127) << 23);
}

// two F32 values rounded to FP16 (nearest even), a in the low half
__device__ __forceinline__ uint32_t pack2(float a, float b) {
    return static_cast<uint32_t>(__half_as_ushort(__float2half_rn(a))) |
           (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(b))) << 16);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    return v;
}

// An FP16 activation operand: row r, value k at a0[(r >> shift0) * ld0 + k] for k < k_split, else
// a1[(r >> shift1) * ld1 + k - k_split] (eh_proj's [e | h] row without materializing it). One segment: k_split = K.
// Pointers 16-byte aligned, ld0, ld1 and k_split multiples of 8 (k_split of 32 when it splits).
struct a_operand {
    const uint16_t * a0      = nullptr;
    int64_t          ld0     = 0;
    int32_t          shift0  = 0;
    const uint16_t * a1      = nullptr;
    int64_t          ld1     = 0;
    int32_t          shift1  = 0;
    int32_t          k_split = 0;
};

struct problem {
    a_operand        a;
    const uint16_t * b    = nullptr;  // FP16 [n][ldb], K contiguous
    int64_t          ldb  = 0;
    int32_t          rows = 0, n = 0, k = 0;
};

// a configuration's CTA rows, threads and shared memory
template <int32_t BN, int32_t MI, int32_t NI = 2> struct shape {
    static constexpr int32_t rows    = 32 * MI;
    static constexpr int32_t threads = 2 * (BN / (16 * NI)) * 32;
    static constexpr int32_t stage   = (rows + BN) * bk * 2;
    static constexpr int32_t smem    = 2 * stage;
};

namespace detail {

__device__ __forceinline__ int32_t swz(int32_t x) {
    return ((x >> 1) & 1) | ((x >> 2) & 2);
}

// D += A B: one m8n8k4 product per quadpair, FP16 inputs, F32 accumulators (HMMA.884.F32)
__device__ __forceinline__ void mma884(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// a fragment pair (16 bytes: two k4 steps) from shared memory
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ void sts128(uint32_t addr, const uint4 & v) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

// the 16-byte chunk (row r, chunk c of the stage starting at K k0) of an activation operand; zero past the rows
__device__ __forceinline__ uint4 load_a(const a_operand & a, int32_t rows, int32_t r, int32_t k) {
    if (r >= rows) {
        return make_uint4(0u, 0u, 0u, 0u);
    }
    const uint16_t * src = k < a.k_split ? a.a0 + (int64_t(r) >> a.shift0) * a.ld0 + k :
                                           a.a1 + (int64_t(r) >> a.shift1) * a.ld1 + (k - a.k_split);
    return __ldg(reinterpret_cast<const uint4 *>(src));
}

}  // namespace detail

// The C element i (0..7) of a lane in its quadpair's 8 x 8 tile: row (i & 2) | (lane & 16 ? 4 : 0) | (lane & 1),
// column (i & 1) | ((lane >> 1) & 1) << 1 | (i >> 2) << 2. The epilogue receives, per row, the four columns
// c0, c0 + 1, c0 + 4, c0 + 5 that the lane holds.
//
// Epi: __device__ void operator()(int32_t row, int32_t col0, const float (&v)[4]) const, called only for row < rows;
// it masks the columns itself (col0 + {0, 1, 4, 5} against n).
template <int32_t BN, int32_t MI, int32_t NI = 2, typename Epi>
__device__ __forceinline__ void gemm_cta(const problem & p, int32_t row0, int32_t col0, uint8_t * smem,
                                         const Epi & epi) {
    static_assert(BN == 64 || BN == 128, "BN 64 or 128");
    static_assert(MI == 2 || MI == 4, "2 or 4 m-tiles a warp");
    static_assert(NI == 2 || (NI == 4 && BN == 128), "2 n-tiles a warp, or 4 at BN 128");
    using sh = shape<BN, MI, NI>;
    constexpr int32_t bm     = sh::rows;
    constexpr int32_t NT = NI * MI; // quadpair tiles a lane
    constexpr int32_t T      = sh::threads;
    constexpr int32_t A_ITER = bm * 4 / T;  // 16-byte chunks a thread stages
    constexpr int32_t B_ITER = BN * 4 / T;
    constexpr int32_t SB     = sh::stage;
    static_assert(A_ITER * T == bm * 4 && B_ITER * T == BN * 4, "even staging");

    const int32_t  tid  = static_cast<int32_t>(threadIdx.x);
    const int32_t  warp = tid >> 5, lane = tid & 31;
    const int32_t  wm = warp & 1, wn = warp >> 1;
    const int32_t  q  = (lane >> 2) & 3;
    const int32_t  ra = (lane & 3) | ((lane >> 2) & 4);  // the lane's A row / B column in its quadpair's 8
    const uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    // shared offsets of the lane's fragments (stage 0; the chunk index is XORed in per read)
    uint32_t a_off[MI], b_off[NI];
    int32_t a_swz[MI], b_swz[NI];
#pragma unroll
    for (int32_t mi = 0; mi < MI; ++mi) {
        const int32_t row = 16 * MI * wm + 16 * mi + 8 * (q & 1) + ra;
        a_off[mi]         = static_cast<uint32_t>(row * 64);
        a_swz[mi]         = detail::swz(row);
    }
#pragma unroll
    for (int32_t ni = 0; ni < NI; ++ni) {
        const int32_t col = 16 * NI * wn + 16 * ni + 8 * (q >> 1) + ra;
        b_off[ni]         = static_cast<uint32_t>(bm * 64 + col * 64);
        b_swz[ni]         = detail::swz(col);
    }

    float acc[NT][8]; // tile mi * NI + ni
#pragma unroll
    for (int32_t t = 0; t < NT; ++t) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            acc[t][i] = 0.0f;
        }
    }

    uint4      sa[A_ITER], sb[B_ITER];
    // the global reads of stage s into registers
    const auto fetch = [&](int32_t s) {
        const int32_t k0 = s * bk;
#pragma unroll
        for (int32_t i = 0; i < A_ITER; ++i) {
            const int32_t ci  = tid + i * T;
            const int32_t row = ci >> 2, c = ci & 3;
            sa[i] = detail::load_a(p.a, p.rows, row0 + row, k0 + 8 * c);
        }
#pragma unroll
        for (int32_t i = 0; i < B_ITER; ++i) {
            const int32_t ci  = tid + i * T;
            const int32_t col = ci >> 2, c = ci & 3;
            sb[i] = col0 + col < p.n ?
                        __ldg(reinterpret_cast<const uint4 *>(p.b + int64_t(col0 + col) * p.ldb + k0 + 8 * c)) :
                        make_uint4(0u, 0u, 0u, 0u);
        }
    };
    const auto store = [&](int32_t buf) {
        const uint32_t base = s0 + static_cast<uint32_t>(buf * SB);
#pragma unroll
        for (int32_t i = 0; i < A_ITER; ++i) {
            const int32_t ci  = tid + i * T;
            const int32_t row = ci >> 2, c = ci & 3;
            detail::sts128(base + static_cast<uint32_t>(row * 64 + 16 * (c ^ detail::swz(row))), sa[i]);
        }
#pragma unroll
        for (int32_t i = 0; i < B_ITER; ++i) {
            const int32_t ci  = tid + i * T;
            const int32_t col = ci >> 2, c = ci & 3;
            detail::sts128(base + static_cast<uint32_t>(bm * 64 + col * 64 + 16 * (c ^ detail::swz(col))), sb[i]);
        }
    };

    const int32_t steps = p.k / bk;
    fetch(0);
    store(0);
    __syncthreads();
#pragma unroll 1
    for (int32_t s = 0; s < steps; ++s) {
        const int32_t buf = s & 1;
        if (s + 1 < steps) {
            fetch(s + 1);
        }
        const uint32_t base = s0 + static_cast<uint32_t>(buf * SB);
#pragma unroll
        for (int32_t c = 0; c < 4; ++c) {  // a 16-byte chunk: k4 steps 2c, 2c + 1
            uint4 bf[NI], af[MI];
#pragma unroll
            for (int32_t ni = 0; ni < NI; ++ni) {
                bf[ni] = detail::lds128(base + b_off[ni] + static_cast<uint32_t>(16 * (c ^ b_swz[ni])));
            }
#pragma unroll
            for (int32_t mi = 0; mi < MI; ++mi) {
                af[mi] = detail::lds128(base + a_off[mi] + static_cast<uint32_t>(16 * (c ^ a_swz[mi])));
            }
#pragma unroll
            for (int32_t mi = 0; mi < MI; ++mi) {
#pragma unroll
                for (int32_t ni = 0; ni < NI; ++ni) {
                    detail::mma884(acc[NI * mi + ni], af[mi].x, af[mi].y, bf[ni].x, bf[ni].y);
                }
            }
#pragma unroll
            for (int32_t mi = 0; mi < MI; ++mi) {
#pragma unroll
                for (int32_t ni = 0; ni < NI; ++ni) {
                    detail::mma884(acc[NI * mi + ni], af[mi].z, af[mi].w, bf[ni].z, bf[ni].w);
                }
            }
        }
        if (s + 1 < steps) {
            store(buf ^ 1);
        }
        __syncthreads();
    }

    // the lane's rows in a quadpair tile: lr + {0, 2}; columns cc + {0, 1, 4, 5}
    const int32_t lr = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int32_t cc = 2 * ((lane >> 1) & 1);
#pragma unroll
    for (int32_t mi = 0; mi < MI; ++mi) {
#pragma unroll
        for (int32_t rh = 0; rh < 2; ++rh) {
            const int32_t row = row0 + 16 * MI * wm + 16 * mi + 8 * (q & 1) + lr + 2 * rh;
            if (row < p.rows) {
#pragma unroll
                for (int32_t ni = 0; ni < NI; ++ni) {
                    const float * t = acc[NI * mi + ni];
                    const float   v[4] = { t[2 * rh], t[2 * rh + 1], t[2 * rh + 4], t[2 * rh + 5] };
                    epi(row, col0 + 16 * NI * wn + 16 * ni + 8 * (q >> 1) + cc, v);
                }
            }
        }
    }
}

}  // namespace fn::tc
