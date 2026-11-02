#pragma once

// APF2: the kernel-size ladder's tensor-core CTA (device only), shared by the dense, HC and grouped-expert families.
// The tile list is TurboMind's sm70 list (1Cat-vLLM 1d7321bf2d, csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/
// gemm/kernel/sm70_884_4.cu:160-240: Type<CTA_M, CTA_N, 32, TG_M, TG_N, ...>, two stages), adapted to FNFL's core
// (fn/kernels/tc/tc_gemm.cuh: its swizzled shared layout, fragment reads and m8n8k4 products); no TurboMind code.
//   tiles:    BM x BN of 8, 16, 32, 64 x 128 (warps 1 x 4, a warp BM x 32), 128 x 128 (2 x 2 warps of 64 x 64) and
//             128 x 256 (2 x 4 warps of 64 x 64). A warp's m8n8k4 quadpairs take 8 x 8 blocks: 2 x 2 of them (a 16 x 16
//             tile, tc_gemm.cuh's arrangement) when the warp has 16 rows or more, 1 x 4 (8 x 32) at 8 rows, so no
//             quadpair multiplies padding rows of an 8-row tile.
//   stages:   K 32 a stage, two stages in shared memory ((BM + BN) x 64 bytes each); the next stage's global reads in
//             registers across the current stage's products, as tc_gemm.cuh.
//   operands: A FP16 rows from an A loader (plain rows, or rows gathered by index), B FP16 columns from a B loader
//             (fetch: a stage's global reads into registers; store: its FP16 values into the stage), so a packed weight
//             format is decoded in the tile loader and no FP16 copy of the weights exists.
//   split K:  splits CTAs a tile (grid z), split s over K stages [s q + min(s, r), ..) of q = stages / splits,
//             r = stages % splits (TurboMind's even partition). Each split writes its F32 accumulators to the tile's
//             partials; the CTA that arrives last (an integer atomic on the tile's counter, no float atomics) sums the
//             splits' partials in split order 0, 1, .. and runs the epilogue, then resets the counter. The sum's
//             order is fixed, so a rerun is bitwise; another split count is another order (APF2's numerical contract: each split count inside the family's FP64 bound).
//   numerics: FP16 products, F32 accumulators (HMMA.884.F32), each split's K in order.
// The epilogue: void operator()(int32_t row, int32_t col0, const float (&v)[4]) for row < rows, columns col0, col0 + 1,
// col0 + 4, col0 + 5 (tc_gemm.cuh's), masked by the caller.

#include "fn/kernels/tc/ladder.h"
#include "fn/kernels/tc/tc_gemm.cuh"

#include <cstdint>

namespace fn::tc {

// a ladder tile's warp layout, threads and shared memory
template <int32_t BM, int32_t BN> struct ltile {
    static_assert(BM == 8 || BM == 16 || BM == 32 || BM == 64 || BM == 128, "BM 8 .. 128");
    static_assert(BN == 128 || (BN == 256 && BM == 128), "BN 128, or 256 at 128 rows");
    static constexpr int32_t bm      = BM;
    static constexpr int32_t bn      = BN;
    static constexpr int32_t WM      = BM <= 64 ? BM : 64;
    static constexpr int32_t WN      = BM <= 64 ? 32 : 64;
    static constexpr int32_t WARPS_M = BM / WM;
    static constexpr int32_t WARPS_N = BN / WN;
    static constexpr int32_t threads = 32 * WARPS_M * WARPS_N;
    static constexpr int32_t QM      = WM == 8 ? 1 : 2;  // quadpair rows of 8 a warp product
    static constexpr int32_t QN      = 4 / QM;
    static constexpr int32_t MT      = WM / (8 * QM);  // m8n8k4 tiles a lane, down and across
    static constexpr int32_t NT      = WN / (8 * QN);
    static constexpr int32_t NACC    = 8 * MT * NT;  // F32 accumulators a thread
    static constexpr int32_t stage   = (BM + BN) * bk * 2;
    static constexpr int32_t smem    = 2 * stage;
    static constexpr int32_t A_CH    = BM * 4;  // 16-byte chunks of a stage
    static constexpr int32_t B_CH    = BN * 4;
    static constexpr int32_t A_IT    = (A_CH + threads - 1) / threads;
    static_assert(NACC * threads == BM * BN, "the accumulators cover the tile");
};

// run(ltile<BM, BN>{}) for tile t of the list (fn/kernels/tc/ladder.h's lad_tiles order); false for another t
template <typename Fn> bool lad_with_tile(int32_t t, Fn && run) {
    switch (t) {
        case 0:
            return run(ltile<8, 128>{});
        case 1:
            return run(ltile<16, 128>{});
        case 2:
            return run(ltile<32, 128>{});
        case 3:
            return run(ltile<64, 128>{});
        case 4:
            return run(ltile<128, 128>{});
        case 5:
            return run(ltile<128, 256>{});
        default:
            return false;
    }
}

// the byte offset of chunk c of row x in a stage region (tc_gemm.cuh's swizzle)
__device__ __forceinline__ uint32_t lsw(int32_t x, int32_t c) {
    return static_cast<uint32_t>(x * 64 + 16 * (c ^ detail::swz(x)));
}

// the plain A loader: tc_gemm.cuh's operand (rows past rows read zero)
struct a_plain {
    a_operand a;
    int32_t   rows = 0;

    __device__ __forceinline__ uint4 operator()(int32_t r, int32_t k) const { return detail::load_a(a, rows, r, k); }
};

// a B loader from a chunk functor (uint4 operator()(col, k): the 8 FP16 values of column col at K k .. k + 7), the
// chunks a thread takes as gemm_cta_b's (thread t: chunks t + i threads, column ci / 4, chunk ci % 4)
template <int32_t BN, int32_t T, typename CH> struct b_chunks {
    static constexpr int32_t IT = BN * 4 / T;
    static_assert(IT * T == BN * 4, "even B staging");
    CH ch;

    struct raw {
        uint4 v[IT];
    };

    __device__ __forceinline__ void fetch(raw & r, int32_t col0, int32_t k0) const {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
        for (int32_t i = 0; i < IT; ++i) {
            const int32_t ci = tid + i * T;
            r.v[i]           = ch(col0 + (ci >> 2), k0 + 8 * (ci & 3));
        }
    }

    __device__ __forceinline__ void store(uint32_t sb, const raw & r) const {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
#pragma unroll
        for (int32_t i = 0; i < IT; ++i) {
            const int32_t ci = tid + i * T;
            detail::sts128(sb + lsw(ci >> 2, ci & 3), r.v[i]);
        }
    }
};

// the split-K reduction's workspace: partials F32 [tiles][splits][BM x BN] (a thread's accumulator i at
// i x threads + thread) and one I32 counter a tile (zero between launches)
struct lsplit {
    float *   part  = nullptr;
    int32_t * count = nullptr;
};

// One CTA: rows row0 .. row0 + BM - 1 (those below rows; the loaders read zero past their own ends) by columns
// col0 .. col0 + BN - 1, split blockIdx.z of splits over K k (a multiple of 32), the split counter and partials of
// output tile tile. AL the A loader (uint4 operator()(row, k)), BL a B loader (fetch / store as b_chunks), Epi the
// epilogue (rows below rows).
template <int32_t BM, int32_t BN, typename AL, typename BL, typename Epi>
__device__ __forceinline__ void lgemm_at(int32_t        row0,
                                         int32_t        rows,
                                         int32_t        col0,
                                         int32_t        tile,
                                         int32_t        k,
                                         int32_t        splits,
                                         const lsplit & ws,
                                         uint8_t *      smem,
                                         const AL &     al,
                                         const BL &     bl,
                                         const Epi &    epi) {
    using sh                = ltile<BM, BN>;
    constexpr int32_t T     = sh::threads;
    constexpr int32_t SB    = sh::stage;
    const int32_t     tid   = static_cast<int32_t>(threadIdx.x);
    const int32_t     warp  = tid >> 5, lane = tid & 31;
    const int32_t     wm    = warp % sh::WARPS_M, wn = warp / sh::WARPS_M;
    const int32_t     q     = (lane >> 2) & 3;
    const int32_t     qm    = q % sh::QM, qn = q / sh::QM;
    const int32_t     ra    = (lane & 3) | ((lane >> 2) & 4);
    const uint32_t    s0    = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int32_t     split = static_cast<int32_t>(blockIdx.z);

    uint32_t a_off[sh::MT], b_off[sh::NT];
    int32_t  a_sw[sh::MT], b_sw[sh::NT];
#pragma unroll
    for (int32_t mi = 0; mi < sh::MT; ++mi) {
        const int32_t r = sh::WM * wm + 8 * sh::QM * mi + 8 * qm + ra;
        a_off[mi]       = static_cast<uint32_t>(r * 64);
        a_sw[mi]        = detail::swz(r);
    }
#pragma unroll
    for (int32_t ni = 0; ni < sh::NT; ++ni) {
        const int32_t c = sh::WN * wn + 8 * sh::QN * ni + 8 * qn + ra;
        b_off[ni]       = static_cast<uint32_t>(BM * 64 + c * 64);
        b_sw[ni]        = detail::swz(c);
    }

    float acc[sh::MT * sh::NT][8];
#pragma unroll
    for (int32_t t = 0; t < sh::MT * sh::NT; ++t) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            acc[t][i] = 0.0f;
        }
    }

    // this split's stages
    const int32_t steps = k / bk;
    const int32_t qs = steps / splits, rs = steps % splits;
    const int32_t st0 = split * qs + min(split, rs);
    const int32_t nst = qs + (split < rs ? 1 : 0);

    uint4               sa[sh::A_IT];
    typename BL::raw    sbr;
    const auto          fetch = [&](int32_t s) {
        const int32_t k0 = s * bk;
#pragma unroll
        for (int32_t i = 0; i < sh::A_IT; ++i) {
            const int32_t ci = tid + i * T;
            if (ci < sh::A_CH) {
                sa[i] = al(row0 + (ci >> 2), k0 + 8 * (ci & 3));
            }
        }
        bl.fetch(sbr, col0, k0);
    };
    const auto store = [&](int32_t buf) {
        const uint32_t base = s0 + static_cast<uint32_t>(buf * SB);
#pragma unroll
        for (int32_t i = 0; i < sh::A_IT; ++i) {
            const int32_t ci = tid + i * T;
            if (ci < sh::A_CH) {
                detail::sts128(base + lsw(ci >> 2, ci & 3), sa[i]);
            }
        }
        bl.store(base + static_cast<uint32_t>(BM * 64), sbr);
    };

    if (nst > 0) {
        fetch(st0);
        store(0);
    }
    __syncthreads();
#pragma unroll 1
    for (int32_t s = 0; s < nst; ++s) {
        const int32_t buf = s & 1;
        if (s + 1 < nst) {
            fetch(st0 + s + 1);
        }
        const uint32_t base = s0 + static_cast<uint32_t>(buf * SB);
#pragma unroll
        for (int32_t c = 0; c < 4; ++c) {  // a 16-byte chunk: k4 steps 2c, 2c + 1
            // the B fragments for the chunk, then one A fragment at a time (a 64 x 64 warp's 128 accumulators leave
            // no room for all of A's); each accumulator still takes k4 step 2c, then 2c + 1
            uint4 bf[sh::NT];
#pragma unroll
            for (int32_t ni = 0; ni < sh::NT; ++ni) {
                bf[ni] = detail::lds128(base + b_off[ni] + static_cast<uint32_t>(16 * (c ^ b_sw[ni])));
            }
#pragma unroll
            for (int32_t mi = 0; mi < sh::MT; ++mi) {
                const uint4 af = detail::lds128(base + a_off[mi] + static_cast<uint32_t>(16 * (c ^ a_sw[mi])));
#pragma unroll
                for (int32_t ni = 0; ni < sh::NT; ++ni) {
                    detail::mma884(acc[sh::NT * mi + ni], af.x, af.y, bf[ni].x, bf[ni].y);
                }
#pragma unroll
                for (int32_t ni = 0; ni < sh::NT; ++ni) {
                    detail::mma884(acc[sh::NT * mi + ni], af.z, af.w, bf[ni].z, bf[ni].w);
                }
            }
        }
        if (s + 1 < nst) {
            store(buf ^ 1);
        }
        __syncthreads();
    }

    if (splits > 1) {
        // the tile's partials: this split's, then the last arrival sums them in split order
        __shared__ int32_t last;
        float * const      part = ws.part + int64_t(tile) * splits * (BM * BN);
#pragma unroll
        for (int32_t t = 0; t < sh::MT * sh::NT; ++t) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                __stcg(part + int64_t(split) * (BM * BN) + (8 * t + i) * T + tid, acc[t][i]);
            }
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            last = atomicAdd(ws.count + tile, 1) == splits - 1 ? 1 : 0;
        }
        __syncthreads();
        if (last == 0) {
            return;
        }
        __threadfence();
        // every split's partial from the workspace (this split's too: the values it wrote), summed in split order
#pragma unroll 1
        for (int32_t s = 0; s < splits; ++s) {
            const float * ps = part + int64_t(s) * (BM * BN) + tid;
#pragma unroll
            for (int32_t t = 0; t < sh::MT * sh::NT; ++t) {
#pragma unroll
                for (int32_t i = 0; i < 8; ++i) {
                    const float x = __ldcg(ps + (8 * t + i) * T);
                    acc[t][i]     = s == 0 ? x : acc[t][i] + x;
                }
            }
        }
        if (tid == 0) {
            ws.count[tile] = 0;  // the next launch's (stream order publishes it)
        }
    }

    // the lane's rows in a quadpair tile: lr + {0, 2}; columns cc + {0, 1, 4, 5}
    const int32_t lr = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int32_t cc = 2 * ((lane >> 1) & 1);
#pragma unroll
    for (int32_t mi = 0; mi < sh::MT; ++mi) {
#pragma unroll
        for (int32_t rh = 0; rh < 2; ++rh) {
            const int32_t row = row0 + sh::WM * wm + 8 * sh::QM * mi + 8 * qm + lr + 2 * rh;
            if (row < rows) {
#pragma unroll
                for (int32_t ni = 0; ni < sh::NT; ++ni) {
                    const float * t    = acc[sh::NT * mi + ni];
                    const float   v[4] = { t[2 * rh], t[2 * rh + 1], t[2 * rh + 4], t[2 * rh + 5] };
                    epi(row, col0 + sh::WN * wn + 8 * sh::QN * ni + 8 * qn + cc, v);
                }
            }
        }
    }
}

// lgemm_at for tile (blockIdx.x row tile, blockIdx.y column tile, blockIdx.z split) of a product of rows. ctiles > 0:
// a batch of products, blockIdx.y = batch x ctiles + column tile (the loaders and the epilogue read the batch from
// blockIdx.y themselves); columns col0 = (blockIdx.y % ctiles) BN of the batch's product
template <int32_t BM, int32_t BN, typename AL, typename BL, typename Epi>
__device__ __forceinline__ void lgemm(int32_t        rows,
                                      int32_t        k,
                                      int32_t        splits,
                                      const lsplit & ws,
                                      uint8_t *      smem,
                                      const AL &     al,
                                      const BL &     bl,
                                      const Epi &    epi,
                                      int32_t        ctiles = 0) {
    const int32_t by   = static_cast<int32_t>(blockIdx.y);
    const int32_t tile = by * static_cast<int32_t>(gridDim.x) + static_cast<int32_t>(blockIdx.x);
    lgemm_at<BM, BN>(static_cast<int32_t>(blockIdx.x) * BM, rows, (ctiles > 0 ? by % ctiles : by) * BN, tile, k, splits,
                     ws, smem, al, bl, epi);
}

}  // namespace fn::tc
