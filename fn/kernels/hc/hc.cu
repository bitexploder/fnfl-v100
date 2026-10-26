#include "fn/kernels/hc/hc.h"

#include "fn/kernels/act/act_device.cuh"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.cuh"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>

namespace fn::hc {
namespace {

constexpr int32_t n_stream = 4;
constexpr int32_t n_col = 5;         // hc_down: 256-feature columns of a stream's own half, one 16-byte load a lane
constexpr int32_t n_oct = 5;         // hc_up: 64-wide octets of the low rank, eight lanes of 16 bytes
constexpr int32_t n_rgrp = 3;        // row groups of a block
constexpr int32_t rows_per_warp = 6; // weight loads a lane issues up front
constexpr int32_t n_warp = n_col * n_rgrp;
constexpr int32_t n_thread = 32 * n_warp;
constexpr int32_t n_block = 72;      // one per SM
constexpr int32_t shipped_depth_down = 3; // HC1c: the stream depth down() and up() ship with (the fastest)
constexpr int32_t shipped_depth_up = 3;
constexpr int32_t down_rows = n_rgrp * rows_per_warp; // hc_down rows of one stream per block
constexpr int32_t down_blocks_per_stream = n_rows_down / down_rows;
constexpr int32_t up_feat_max = n_rgrp * rows_per_warp; // hc_up features per block: 56 blocks of 18, 16 of 17
constexpr int32_t up_big_blocks = 56;
constexpr int32_t n_stage_warp = n_own / 4 / 32; // hc_down: warps that stage, one float4 of each row a lane

static_assert(fn::n_hc == n_stream && 2 * n_own == fn::n_embd && fn::n_hc_lowrank == n_down, "HC shapes");
static_assert(n_col * 256 == n_own && n_oct * 64 == n_down, "one 16-byte load a lane must tile a row");
static_assert(n_stream * down_blocks_per_stream * down_rows == n_stream * n_rows_down, "hc_down rows tile the grid");
static_assert(down_blocks_per_stream * n_stream == n_block, "hc_down: one block per SM");
static_assert(n_stage_warp <= n_warp && n_stage_warp <= down_blocks_per_stream, "hc_down staging warps");
static_assert(up_big_blocks * up_feat_max + (n_block - up_big_blocks) * (up_feat_max - 1) == n_own,
              "hc_up features tile the grid");
static_assert(n_mom == 1284 + 4 * n_inject && (n_mom * 4) % 16 == 0, "the moment rows stay 16-byte aligned");
static_assert(n_mom_head == 1284 && (n_mom_head * 4) % 16 == 0 && n_mom_head <= fn::xchg::words_hc,
              "head-mode moment rows: 16-byte aligned, within an X-HC row");
static_assert(n_mom == fn::xchg::words_hc && n_own == fn::xchg::words_hmix && 2 * n_own == fn::xchg::hmix_layout::width,
              "the X-HC and X-HMIX payload rows");
constexpr int32_t hc_slot_u4 = fn::xchg::slot_words(fn::xchg::msg::hc) / 4; // X-HC's slot, both regions (HC4)
static_assert(fn::xchg::hmix_slot_u4 <= n_block * n_thread && hc_slot_u4 <= n_block * n_thread,
              "down2's threads reset an X-HC and an X-HMIX slot in one pass");

// BF16 widened exactly to F32 by one byte permute each (ALU pipe, not IMAD). z is an opaque zero (see order_after) that
// supplies the zero bytes; a single PRMT keeps ptxas from splitting the dependency off.
__device__ __forceinline__ float bf16_lo(uint32_t w, uint32_t z) {
    return __uint_as_float(__byte_perm(w, z, 0x1044));
}

__device__ __forceinline__ float bf16_hi(uint32_t w, uint32_t z) {
    return __uint_as_float(__byte_perm(w, z, 0x3244));
}

// acc += the 8 BF16 weights of w times the 8 activations in a0 (weights 0..3) and a1 (weights 4..7)
__device__ __forceinline__ float dot8(uint4 w, float4 a0, float4 a1, float acc, uint32_t z) {
    acc = fmaf(bf16_lo(w.x, z), a0.x, acc);
    acc = fmaf(bf16_hi(w.x, z), a0.y, acc);
    acc = fmaf(bf16_lo(w.y, z), a0.z, acc);
    acc = fmaf(bf16_hi(w.y, z), a0.w, acc);
    acc = fmaf(bf16_lo(w.z, z), a1.x, acc);
    acc = fmaf(bf16_hi(w.z, z), a1.y, acc);
    acc = fmaf(bf16_lo(w.w, z), a1.z, acc);
    acc = fmaf(bf16_hi(w.w, z), a1.w, acc);
    return acc;
}

// DR1: a lane's 8 Q8_0 weights (one block's, consecutive) and that block's FP16 scale bits
struct q8w {
    uint2 q;
    unsigned short d;
};

// int8 byte k of t ^ 0x80808080 (the byte plus 128) as F32, exactly: the permute builds 2^23 + q + 128 and the
// subtraction of 2^23 + 128 is exact (both are integers below 2^24). m is 0x4b000000 with an opaque zero merged in.
template <int K> __device__ __forceinline__ float q8_val(uint32_t t, uint32_t m) {
    static_assert(K >= 0 && K < 4, "a byte of the word");
    return __uint_as_float(__byte_perm(t, m, 0x7440 | K)) - 8388736.0f;
}

// acc += the 8 Q8_0 weights of w, each d * q exactly (int8 times an FP16 significand fits F32's 24 bits), times the 8
// activations, in dot8's order: with weights BF16 also holds, the result is dot8's bitwise
__device__ __forceinline__ float dot8q(q8w w, float4 a0, float4 a1, float acc, uint32_t z) {
    const float d = __half2float(__ushort_as_half(w.d));
    const uint32_t m = 0x4b000000u | z;
    const uint32_t t0 = w.q.x ^ 0x80808080u, t1 = w.q.y ^ 0x80808080u;
    acc = fmaf(q8_val<0>(t0, m) * d, a0.x, acc);
    acc = fmaf(q8_val<1>(t0, m) * d, a0.y, acc);
    acc = fmaf(q8_val<2>(t0, m) * d, a0.z, acc);
    acc = fmaf(q8_val<3>(t0, m) * d, a0.w, acc);
    acc = fmaf(q8_val<0>(t1, m) * d, a1.x, acc);
    acc = fmaf(q8_val<1>(t1, m) * d, a1.y, acc);
    acc = fmaf(q8_val<2>(t1, m) * d, a1.z, acc);
    acc = fmaf(q8_val<3>(t1, m) * d, a1.w, acc);
    return acc;
}

__device__ __forceinline__ float dot8w(uint4 w, float4 a0, float4 a1, float acc, uint32_t z) {
    return dot8(w, a0, a1, acc, z);
}

__device__ __forceinline__ float dot8w(q8w w, float4 a0, float4 a1, float acc, uint32_t z) {
    return dot8q(w, a0, a1, acc, z);
}

// A zero that depends on v. Unpacking the next chunk with it keeps that unpack after this row's FMAs: otherwise ptxas
// unpacks every chunk first and the warp waits for its last chunk (HX13). mz is zero for every valid launch.
__device__ __forceinline__ uint32_t order_after(float v, uint32_t mz) {
    return __float_as_uint(v) & mz;
}

// the payload words of a ring are F32 bits
__device__ __forceinline__ float4 as_f4(uint4 v) {
    return make_float4(__uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z), __uint_as_float(v.w));
}

__device__ __forceinline__ float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

} // namespace

// Block (s, g): stream s, rows 18g .. 18g+17 over all 1,280 own features, so every moment is finished in one block
// and no cross-block reduction is needed. The block stages its stream's x*gamma itself (18 blocks a stream repeat it,
// one round of 16-byte loads from L2) and a barrier puts those loads ahead of the weights. Warp (c, rg): column c
// (lane l owns features c*256 + 8l .. 8l+7, read from shared slots l and l+32, so a quarter-warp reads 128 contiguous
// bytes) and rows 6rg .. 6rg+5, one weight load each, consumed in the order issued.
// HC2 (X): the same kernel, and its blocks then send their moments to the peer (X-HC); see the finish below.
// V1 (H, head mode, with X): no injection rows, so a stream has 320 weight rows and the moment rows 1,284 words; block
// g = 17 of a stream has rows 306..319 only: its row group 2 reloads row 319 for its last four rows (L2 hits) and the
// block sends 14 rows.
// DR1 (Q, with X): Q8_0 weights (hc.h); a lane's row is 8 q and its block's d (q8w), decoded by dot8q in dot8's order.
template <bool X, bool Q = false> struct down_param {
    using type = down_args;
};

template <> struct down_param<true, false> {
    using type = down2_args;
};

template <> struct down_param<true, true> {
    using type = down2_q8_args;
};

__device__ __forceinline__ const down_args & base_args(const down_args & a) {
    return a;
}

__device__ __forceinline__ const down_args & base_args(const down2_args & a) {
    return a.d;
}

__device__ __forceinline__ const down_args & base_args(const down2_q8_args & a) {
    return a.a.d;
}

__device__ __forceinline__ const down2_args & x_args(const down2_args & a) {
    return a;
}

__device__ __forceinline__ const down2_args & x_args(const down2_q8_args & a) {
    return a.a;
}

// a weight row's lane slice: BF16 (16 bytes) or Q8_0 (q8w)
template <bool Q> struct w_slot {
    using type = uint4;
};

template <> struct w_slot<true> {
    using type = q8w;
};

// The lane's weight pointers: row stride rs, the slice at row r (+ an opaque zero z): BF16 one uint4, Q8_0 the q
// words and the scale of the lane's block (blk_row scales a row, the lane's at blk_lane)
template <bool Q> struct w_lane;

template <> struct w_lane<false> {
    const uint4 * p;
    int32_t rs;
    __device__ __forceinline__ uint4 at(int32_t r, uint32_t z = 0u) const { return __ldcs(p + r * rs + z); }
};

template <> struct w_lane<true> {
    const uint2 * q;
    const unsigned short * d;
    int32_t rs, ds;
    __device__ __forceinline__ q8w at(int32_t r, uint32_t z = 0u) const {
        q8w w;
        w.q = __ldcs(q + r * rs + z);
        w.d = __ldcs(d + r * ds + z);
        return w;
    }
};

// HC4 (B3): the body is down_phase, which hc_down_kernel and the fused sublayer launch (hc_sub_kernel, FUSED) share.
// FUSED: the grid's blocks have NT threads (the up phase's 640), and warps n_warp .. take no part beyond the barriers;
// this card's moments go to the own region of its X-HC slot (own_mom) instead of a.mom; and the resets of the previous
// sublayer's slots move before the block's second barrier, with warps 0..B-1 fencing after it, so every reset store is
// performed before any moment of the block is published (xchg.h, HC4).
// ME12: vb is the block's (virtual) index, blockIdx.x except in the paired launch, which runs two down tasks a block.
// U (the paired launch): act is in the dynamic region at every B, so the up phase's arrays reuse it (one arena).
template <int B, int D, bool X, bool H, bool Q, bool FUSED, int NT, bool U = false>
__device__ __forceinline__ void down_phase(const typename down_param<X, Q>::type & p, float * own_mom, int32_t vb) {
    static_assert(D >= 1 && D <= rows_per_warp, "stream depth");
    static_assert(X || !H, "head mode is an HC2 mode");
    static_assert(X || !Q, "Q8_0 weights are an HC2 mode (DR1)");
    static_assert(X || !FUSED, "the fused launch is an HC2 mode");
    // FUSED: all NT / 32 = 20 warps take rows, 4 row groups of 5 (rows 15..19 of group 3 past the block's 18 reload
    // its last row and are not sent), so no shuffle sits in a branch on the warp index; a row's arithmetic is unchanged
    constexpr int32_t n_rg = FUSED ? 4 : n_rgrp;       // row groups
    constexpr int32_t rpw = FUSED ? 5 : rows_per_warp; // rows a warp
    constexpr int32_t n_red = n_rg * rpw;              // row slots of red (18, or 20 fused)
    static_assert(D >= 1 && D <= rpw && n_red >= down_rows && n_col * n_rg * 32 == NT, "down's warps fill the block");
    constexpr int32_t n_rows_w = H ? n_down : n_rows_down; // weight rows a stream
    constexpr int32_t mom_w = H ? n_mom_head : n_mom;      // words a moment row
    const down_args & a = base_args(p);
    // SP2: the fused launch at B = 5 takes act from the dynamic region (sub_dyn_bytes), which the up phase reuses:
    // both phases' static arrays would pass 48 KB. act's last read precedes the down phase's last barrier, and the up
    // phase writes its arrays only after its first barrier.
    constexpr bool dyn = FUSED && (B > 4 || U);
    __shared__ float4 act_st[dyn ? 1 : B][n_col][64];
    extern __shared__ float4 sub_dyn[];
    float4(*act)[n_col][64] = dyn ? reinterpret_cast<float4(*)[n_col][64]>(sub_dyn) : act_st;
    __shared__ float red[n_red][B][n_col];
    __shared__ float ssp[B][n_stage_warp];
    // HC2's finish reads row b's sums with lane = row: red laid out [b][c][row] instead (same storage)
    float (*redx)[n_col][n_red] = reinterpret_cast<float (*)[n_col][n_red]>(&red[0][0][0]);

    const int32_t s = vb % n_stream; // the stream in the low bits keeps the index math to shifts
    const int32_t g = vb / n_stream;
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;

    // warps 0..9 stage float4 f4 of every row: lanes 0..15 take the even float4s of the warp's 32 and lanes 16..31 the
    // odd ones, so the shared stores below are conflict-free
    const bool stager = warp < n_stage_warp;
    const int32_t f4 = 32 * warp + 2 * (lane & 15) + (lane >> 4);
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 hr[B], hv[B], gm = zero4;
    float hp[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        hr[b] = hv[b] = zero4;
        hp[b] = 0.0f;
    }
    if (stager) {
        // 16-byte aligned, checked by the launcher
        const float4 * hres4 = reinterpret_cast<const float4 *>(a.hres) + s * (n_own / 4) + f4;
        const float4 * hx4 = reinterpret_cast<const float4 *>(a.hx) + f4;
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            hr[b] = __ldcg(hres4 + b * n_stream * (n_own / 4));
            hv[b] = __ldcg(hx4 + b * (n_own / 4));
            hp[b] = __ldcg(a.hpost + b * n_stream + s);
        }
        gm = __ldcg(reinterpret_cast<const float4 *>(a.gamma) + s * (n_own / 4) + f4);
    }

    // Every staging load goes out before any weight load (HX13: staging queued behind the weights waited for the
    // stream). ptxas reorders loads freely but cannot move coherent loads across a barrier; .nc loads would cross it.
    __syncthreads();
    const int32_t c = warp % n_col;
    const int32_t rg = warp / n_col;
    const int32_t row0 = s * n_rows_w + g * down_rows + rg * rpw;
    w_lane<Q> wrow;
    if constexpr (Q) {
        // the q plane, then the d plane (hc.h); a lane's 8 q are one 8-byte load and its block is c * 8 + lane / 4
        const uint8_t * wb = p.w;
        wrow.q = reinterpret_cast<const uint2 *>(wb) + (row0 * n_own + c * 256) / 8 + lane;
        wrow.d = reinterpret_cast<const unsigned short *>(wb + n_stream * n_rows_w * n_own) + row0 * (n_own / 32) +
                 c * 8 + lane / 4;
        wrow.rs = n_own / 8;
        wrow.ds = n_own / 32;
    } else {
        wrow.p = reinterpret_cast<const uint4 *>(a.w_down) + (row0 * n_own + c * 256) / 8 + lane;
        wrow.rs = n_own / 8;
    }
    // the warp's last row (rows past it reload it): head mode, the stream's last; FUSED, also the block's last
    const int32_t qmax = FUSED ? min(n_rows_w - g * down_rows, down_rows) - 1 - rg * rpw
                               : n_down - 1 - (g * down_rows + rg * rows_per_warp);
    const bool clamp = H || FUSED;
    // D rows a lane in flight; row q + D goes out once row q is consumed (HC1c)
    typename w_slot<Q>::type w[D];
#pragma unroll
    for (int32_t q = 0; q < D; ++q) {
        w[q] = wrow.at(clamp ? min(q, qmax) : q);
    }
    if constexpr (FUSED) {
        // release of the previous sublayer's slots on this card (xchg.h step 4, HC4): every thread of the grid takes
        // at most one 16-byte group of each, before the second barrier
        const int32_t u = vb * NT + threadIdx.x;
        if (x_args(p).hc_reset != nullptr && u < hc_slot_u4) {
            fn::xchg::reset4(reinterpret_cast<uint4 *>(x_args(p).hc_reset) + u);
        }
        if (x_args(p).hmix_reset != nullptr) {
            fn::xchg::hmix_reset_slot(x_args(p).hmix_reset, u);
        }
    }

    if (stager) {
        const int32_t slot = 16 * (warp & 1) + (lane & 15) + 32 * (lane >> 4);
        float ss[B];
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            float4 x;
            x.x = fmaf(hp[b], hv[b].x, hr[b].x);
            x.y = fmaf(hp[b], hv[b].y, hr[b].y);
            x.z = fmaf(hp[b], hv[b].z, hr[b].z);
            x.w = fmaf(hp[b], hv[b].w, hr[b].w);
            if (warp == g) { // blocks 0..9 of the stream each write 32 float4s of every row
                reinterpret_cast<float4 *>(a.x)[(b * n_stream + s) * (n_own / 4) + f4] = x;
            }
            act[b][warp / 2][slot] = make_float4(x.x * gm.x, x.y * gm.y, x.z * gm.z, x.w * gm.w);
            ss[b] = x.x * x.x;
            ss[b] = fmaf(x.y, x.y, ss[b]);
            ss[b] = fmaf(x.z, x.z, ss[b]);
            ss[b] = fmaf(x.w, x.w, ss[b]);
        }
        if (g == 0) {
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
#pragma unroll
                for (int32_t o = 16; o > 0; o >>= 1) {
                    ss[b] += __shfl_xor_sync(0xffffffffu, ss[b], o);
                }
            }
            if (lane == 0) {
#pragma unroll
                for (int32_t b = 0; b < B; ++b) {
                    ssp[b][warp] = ss[b];
                }
            }
        }
    }
    __syncthreads();
    if constexpr (FUSED) {
        if (warp < B) { // the block's moment senders: the resets above (every thread's) before any moment store
            __threadfence();
        }
    }

    float4 a0[B], a1[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        a0[b] = act[b][c][lane];
        a1[b] = act[b][c][lane + 32];
    }

    const uint32_t mz = gridDim.x == n_block ? 0u : ~0u;
    uint32_t z = mz;
#pragma unroll
    for (int32_t q = 0; q < rpw; ++q) {
        float acc[B];
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            acc[b] = dot8w(w[q % D], a0[b], a1[b], 0.0f, z);
        }
        z = order_after(acc[B - 1], mz);
        if (q + D < rpw) { // the refill's address carries z, so it waits for this row's FMAs
            w[q % D] = wrow.at(clamp ? min(q + D, qmax) : q + D, z);
        }
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
#pragma unroll
            for (int32_t o = 16; o > 0; o >>= 1) {
                acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], o);
            }
        }
        float v = acc[0];
#pragma unroll
        for (int32_t b = 1; b < B; ++b) {
            v = lane == b ? acc[b] : v;
        }
        if (lane < B) {
            if constexpr (X) {
                redx[lane][c][rg * rpw + q] = v; // banks 0, 26, 20, 14 (fused 0, 4, 8, 12; + c, row): conflict-free
            } else {
                red[rg * rows_per_warp + q][lane][c] = v;
            }
        }
    }
    __syncthreads();

    if constexpr (X) {
        // Warp b sums row b's moments in HC1c's order (lane r: row 18g + r) and sends them as each block finishes
        // them: 16-byte stores to the peer's X-HC slot and to this card's own copy (xchg::put_hc_rows). The moments
        // are stored as their canonical bits (xchg.h: NaNs canonicalized), so both cards hold the same words.
        if (warp < B) {
            const int32_t b = warp;
            const int32_t r = lane < down_rows ? lane : 0;
            float v = redx[b][0][r];
#pragma unroll
            for (int32_t cc = 1; cc < n_col; ++cc) {
                v += redx[b][cc][r];
            }
            // F32 moments handled as their bit patterns from here on
            uint32_t * own = reinterpret_cast<uint32_t *>(FUSED ? own_mom : a.mom) + b * mom_w;
            uint32_t * peer = reinterpret_cast<uint32_t *>(x_args(p).hc_peer) + b * mom_w;
            const int32_t n = H ? min(down_rows, n_down - g * down_rows) : down_rows;
            fn::xchg::put_hc_rows(own, peer, s, g * down_rows, n, fn::xchg::canonical(v));
        }
        if (g == 0 && threadIdx.x < B) {
            const int32_t b = threadIdx.x;
            float v = ssp[b][0];
#pragma unroll
            for (int32_t i = 1; i < n_stage_warp; ++i) {
                v += ssp[b][i];
            }
            const uint32_t bits = fn::xchg::canonical(v);
            reinterpret_cast<uint32_t *>(FUSED ? own_mom : a.mom)[b * mom_w + s] = bits; // F32 moment as its bits
            reinterpret_cast<uint32_t *>(x_args(p).hc_peer)[b * mom_w + s] = bits;
        }
        if constexpr (!FUSED) {
            // release: this card's X-HC slot of the previous sublayer, whose readers (hc_up) have finished (xchg.h);
            // every row of it and both regions (R2, HC4): the previous sublayer may have had more rows than this one,
            // and a row left unreset would be read as landed by a later sublayer of that many rows
            const int32_t u = vb * NT + threadIdx.x;
            if (x_args(p).hc_reset != nullptr && u < hc_slot_u4) {
                fn::xchg::reset4(reinterpret_cast<uint4 *>(x_args(p).hc_reset) + u);
            }
            // HC3: this card's X-HMIX slot of the previous sublayer, after every reader of it (xchg.h step 4)
            if (x_args(p).hmix_reset != nullptr) {
                fn::xchg::hmix_reset_slot(x_args(p).hmix_reset, u);
            }
        }
        return;
    }

    if (threadIdx.x < down_rows * B) {
        const int32_t r = threadIdx.x / B;
        const int32_t b = threadIdx.x % B;
        float v = red[r][b][0];
#pragma unroll
        for (int32_t cc = 1; cc < n_col; ++cc) {
            v += red[r][b][cc];
        }
        const int32_t j = g * down_rows + r;
        const int32_t at = j < n_down ? 4 + n_down * s + j : 4 + n_down * n_stream + n_inject * s + (j - n_down);
        a.mom[b * n_mom + at] = v;
    }
    if (g == 0 && threadIdx.x < B) {
        const int32_t b = threadIdx.x;
        float v = ssp[b][0];
#pragma unroll
        for (int32_t i = 1; i < n_stage_warp; ++i) {
            v += ssp[b][i];
        }
        a.mom[b * n_mom + s] = v;
    }
}

template <int B, int D, bool X = false, bool H = false, bool Q = false>
__global__ void __launch_bounds__(n_thread, 1) hc_down_kernel(const typename down_param<X, Q>::type p) {
    down_phase<B, D, X, H, Q, false, n_thread>(p, nullptr, blockIdx.x);
}

// Block g: features c0 .. c0+n_f-1 in all four streams, so the mix needs no other block. Every block forms low from
// the two cards' moments (one round of 16-byte loads from L2) after its first D weight loads. Warp (o, rg): low
// octet o (lane l = 8s + k owns low[64o + 8k .. 8k+7], read from shared slots k and k+8) and features 6rg .. 6rg+5;
// quarter-warp s is stream s. Each lane loads its activations once and reuses them for all six features.
template <int B, int D>
__global__ void __launch_bounds__(n_thread, 1) hc_up_kernel(const up_args a) {
    static_assert(D >= 1 && D <= rows_per_warp, "stream depth");
    constexpr int32_t n_low_warp = 3 * B;               // warps 3b .. 3b+2 form row b's low, a float4 a lane (80 lanes)
    constexpr int32_t n_mix_warp = n_warp - n_low_warp; // the others load gamma and x for the mix
    constexpr int32_t n_mix_iter = (n_stream * B + n_mix_warp - 1) / n_mix_warp;
    static_assert(n_mix_warp >= 1 && up_feat_max <= 32, "hc_up warp roles");
    static_assert(up_feat_max - 1 > (n_rgrp - 1) * rows_per_warp + rows_per_warp - 2, "rows 0..4 of each group exist");

    __shared__ float4 lows[B][n_oct][16];
    // feature innermost and a row stride of 18 words: the finish's reads of two rows in one warp are conflict-free
    __shared__ float red[n_stream][n_oct][B][up_feat_max];
    __shared__ float gx[n_stream][B][up_feat_max];
    __shared__ float rinv_s[B][n_stream];

    const int32_t g = blockIdx.x;
    const int32_t n_f = g < up_big_blocks ? up_feat_max : up_feat_max - 1;
    const int32_t c0 =
        g < up_big_blocks ? up_feat_max * g : up_big_blocks * up_feat_max + (up_feat_max - 1) * (g - up_big_blocks);
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;

    // The weights first (the order HC2 needs), D rows a lane; then the moment and mix reads. With the flood bounded
    // to D loads a lane, those reads wait about D x 0.53 us, not the whole stream (HC1c H5).
    const int32_t o = warp % n_oct;
    const int32_t rg = warp / n_oct;
    const int32_t s = lane / 8;
    const int32_t k8 = lane % 8;
    const int32_t row0 = (c0 + rg * rows_per_warp) * n_stream + s; // weight row (feature, stream)
    const uint4 * wrow = reinterpret_cast<const uint4 *>(a.w_up) + row0 * (n_down / 8) + o * 8 + k8;
    // in a 17-feature block, row group 2 reloads feature 16 as its sixth row (an L2 hit) and skips it below; rows 0..4
    // always exist (n_f >= 17), so only the last load needs the clamp
    const int32_t q5 = min(rows_per_warp - 1, n_f - 1 - rg * rows_per_warp);
    uint4 w[D];
#pragma unroll
    for (int32_t q = 0; q < D; ++q) {
        w[q] = __ldcs(wrow + (q < rows_per_warp - 1 ? q : q5) * n_stream * (n_down / 8));
    }
    __syncthreads(); // coherent loads cannot cross it: every weight load goes out before the reads below

    const float4 * mo4 = reinterpret_cast<const float4 *>(a.mom_own); // 16-byte aligned, checked by the launcher
    const float4 * mp4 = reinterpret_cast<const float4 *>(a.mom_peer);
    constexpr int32_t mrow4 = n_mom / 4;
    constexpr int32_t q4 = (4 + n_stream * n_down) / 4; // q[s][0..3] is float4 q4 + s of a moment row

    // Lane tt < 80 of row lb's 96 forms one float4 of low (tt % 16 is its shared slot: conflict-free stores). In the
    // last block lane 80 forms the four injection coefficients the same way, from q instead of u.
    const int32_t lb = warp / 3;
    const int32_t tt = 32 * (warp - 3 * lb) + lane;
    const bool lower = warp < n_low_warp && tt < 80;
    const bool injector = warp < n_low_warp && tt == 80 && g == n_block - 1;
    const int32_t j4 = 16 * (tt / 16) + 2 * (tt & 7) + ((tt >> 3) & 1);
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 vo = zero4, vp = zero4, uo[n_stream], up4[n_stream];
#pragma unroll
    for (int32_t si = 0; si < n_stream; ++si) {
        uo[si] = up4[si] = zero4;
    }
    if (lower || injector) {
        vo = __ldcg(mo4 + lb * mrow4);
        vp = __ldcg(mp4 + lb * mrow4);
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            const int32_t at = lb * mrow4 + (lower ? 1 + (n_down / 4) * si + j4 : q4 + si);
            uo[si] = __ldcg(mo4 + at);
            up4[si] = __ldcg(mp4 + at);
        }
    }
    // mix warp m: gamma and x of the block's features (lane = feature) for (row, stream) pairs m, m + n_mix_warp, ..
    const int32_t m = warp - n_low_warp;
    float xm[n_mix_iter], gmm[n_mix_iter];
#pragma unroll
    for (int32_t it = 0; it < n_mix_iter; ++it) {
        xm[it] = gmm[it] = 0.0f;
    }
    if (m >= 0) {
#pragma unroll
        for (int32_t it = 0; it < n_mix_iter; ++it) {
            const int32_t p = m + n_mix_warp * it;
            if (p < n_stream * B && lane < n_f) {
                xm[it] = __ldcg(a.x + p * n_own + c0 + lane); // pair p = b*4 + s is row p of x
                gmm[it] = __ldcg(a.gamma + (p % n_stream) * n_own + c0 + lane);
            }
        }
    }

    if (lower || injector) {
        float ri[n_stream];
        ri[0] = 1.0f / sqrtf((vo.x + vp.x) / float(fn::n_embd) + fn::rms_eps);
        ri[1] = 1.0f / sqrtf((vo.y + vp.y) / float(fn::n_embd) + fn::rms_eps);
        ri[2] = 1.0f / sqrtf((vo.z + vp.z) / float(fn::n_embd) + fn::rms_eps);
        ri[3] = 1.0f / sqrtf((vo.w + vp.w) / float(fn::n_embd) + fn::rms_eps);
        float4 z = make_float4((uo[0].x + up4[0].x) * ri[0], (uo[0].y + up4[0].y) * ri[0],
                               (uo[0].z + up4[0].z) * ri[0], (uo[0].w + up4[0].w) * ri[0]);
#pragma unroll
        for (int32_t si = 1; si < n_stream; ++si) {
            z.x = fmaf(uo[si].x + up4[si].x, ri[si], z.x);
            z.y = fmaf(uo[si].y + up4[si].y, ri[si], z.y);
            z.z = fmaf(uo[si].z + up4[si].z, ri[si], z.z);
            z.w = fmaf(uo[si].w + up4[si].w, ri[si], z.w);
        }
        z = make_float4(0.25f * z.x, 0.25f * z.y, 0.25f * z.z, 0.25f * z.w);
        if (lower) {
            lows[lb][tt / 16][tt % 16] =
                make_float4(z.x * sigmoid(z.x), z.y * sigmoid(z.y), z.z * sigmoid(z.z), z.w * sigmoid(z.w));
            if (tt == 0) {
#pragma unroll
                for (int32_t si = 0; si < n_stream; ++si) {
                    rinv_s[lb][si] = ri[si];
                }
            }
        } else {
            reinterpret_cast<float4 *>(a.inject)[lb] =
                make_float4(2.0f * sigmoid(z.x), 2.0f * sigmoid(z.y), 2.0f * sigmoid(z.z), 2.0f * sigmoid(z.w));
        }
    }
#pragma unroll
    for (int32_t it = 0; it < n_mix_iter; ++it) {
        const int32_t p = m + n_mix_warp * it;
        if (m >= 0 && p < n_stream * B && lane < n_f) {
            gx[p % n_stream][p / n_stream][lane] = gmm[it] * xm[it];
        }
    }
    __syncthreads();

    float4 a0[B], a1[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        a0[b] = lows[b][o][k8];
        a1[b] = lows[b][o][k8 + 8];
    }

    const uint32_t mz = gridDim.x == n_block ? 0u : ~0u;
    uint32_t z = mz;
#pragma unroll
    for (int32_t q = 0; q < rows_per_warp; ++q) {
        const int32_t f = rg * rows_per_warp + q;
        if (f < n_f) {
            float acc[B];
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
                acc[b] = dot8(w[q % D], a0[b], a1[b], 0.0f, z);
            }
            z = order_after(acc[B - 1], mz);
            if (q + D < rows_per_warp) { // the refill's address carries z, so it waits for this row's FMAs
                const int32_t qn = q + D < rows_per_warp - 1 ? q + D : q5;
                w[q % D] = __ldcs(wrow + qn * n_stream * (n_down / 8) + z);
            }
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
#pragma unroll
                for (int32_t mm = 1; mm < 8; mm <<= 1) {
                    acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], mm);
                }
            }
            float v = acc[0];
#pragma unroll
            for (int32_t b = 1; b < B; ++b) {
                v = k8 == b ? acc[b] : v;
            }
            if (k8 < B) {
                red[s][o][k8][f] = v;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < B * up_feat_max) {
        const int32_t b = threadIdx.x / up_feat_max;
        const int32_t f = threadIdx.x % up_feat_max;
        if (f < n_f) {
            float mixed = 0.0f;
#pragma unroll
            for (int32_t si = 0; si < n_stream; ++si) {
                float zs = red[si][0][b][f];
#pragma unroll
                for (int32_t oo = 1; oo < n_oct; ++oo) {
                    zs += red[si][oo][b][f];
                }
                mixed = fmaf(gx[si][b][f] * rinv_s[b][si], sigmoid(zs), mixed);
            }
            a.mixed[b * n_own + c0 + f] = 0.25f * mixed;
        }
    }
}

// HC2's hc_up on one card of the pair: block g, warps and lanes as hc_up_kernel, with the peer's moments polled from
// this card's X-HC slot and the block's mixed sent to the peer as it finishes (X-HMIX).
//   ARM 0 (up_arm::ring): HC1c's order: D weight rows a lane, a barrier, the moment and mix reads (the poll among
//          them), low, then the FMAs with refills.
//   ARM 1 (up_arm::staged): the block streams its whole weight slice (n_f x 4 x 320 BF16, contiguous) into shared
//          memory first, D 16-byte loads a lane in flight and each refilled as its data arrives (R1); its own moments
//          and the mix inputs are read during that stream; then it polls X-HC, forms low and runs the same FMAs in the
//          same order from shared memory, so the outputs are bitwise those of ARM 0.
// Both: the mix inputs (written before the kernel, read once) are 16-byte loads through L1 (R3); the moments of the
// two cards are summed in card order (card 0 + card 1) on both cards; red is [oct][stream][row][feature], so the dot
// product's stores are conflict-free (HC1c-L5).
// V1 (H, head mode): moment rows of 1,284 words and no injector lane; everything else unchanged.
// DR1 (Q, ARM 0 only): Q8_0 weights (hc.h); a lane's row is 8 q of one block and the block's d, decoded by dot8q.
template <bool Q> struct up2_param {
    using type = up2_args;
};

template <> struct up2_param<true> {
    using type = up2_q8_args;
};

__device__ __forceinline__ const up2_args & up_base(const up2_args & a) {
    return a;
}

__device__ __forceinline__ const up2_args & up_base(const up2_q8_args & a) {
    return a.a;
}

// FMS2 (build item 2): the body of hc_up2_kernel with its block index g, so the joint catch-up's lanes launch
// (hc_up2_lanes_kernel, blockIdx.y the lane) runs the same code on each lane's arguments
template <int B, int ARM, bool H, bool Q>
__device__ __forceinline__ void up2_body(const typename up2_param<Q>::type & p, int32_t g) {
    static_assert(!Q || ARM == 0, "Q8_0 weights take arm 0 (DR1)");
    const up2_args & a = up_base(p);
    constexpr int32_t D = shipped_depth_up;
    constexpr int32_t n_low_warp = 3 * B;
    constexpr int32_t n_mix_warp = n_warp - n_low_warp;
    // SP2: at B = 5 every warp lowers, so the mix loads go to the block's first lanes (a thread may do both)
    constexpr bool mix_all = n_mix_warp == 0;
    constexpr int32_t n_mix_lane = mix_all ? n_thread : 32 * n_mix_warp;
    constexpr int32_t mix_warp0 = mix_all ? 0 : n_low_warp; // the first mix warp
    constexpr int32_t n_grp = 6; // 16-byte groups that cover any 17 or 18 consecutive features of a row
    constexpr int32_t row_u4 = n_stream * n_down / 8;           // 160 uint4: one feature's weights, all streams
    constexpr int32_t n_stage = up_feat_max * row_u4 / n_thread; // ARM 1: uint4 loads a lane (18 features)
    static_assert(n_stage * n_thread == up_feat_max * row_u4 && n_stage >= D, "the staged slice tiles the block");
    static_assert(n_stream * B * n_grp <= n_mix_lane, "one 16-byte mix load a mix lane");
    static_assert(up_feat_max - 1 > (n_rgrp - 1) * rows_per_warp + rows_per_warp - 2, "rows 0..4 of each group exist");

    extern __shared__ uint4 wsm[]; // ARM 1: the block's weights as in memory, [feature][stream][40] (46,080 bytes)
    __shared__ float4 lows[B][n_oct][16];
    __shared__ float red[n_oct][n_stream][B][up_feat_max];
    __shared__ float gx[n_stream][B][up_feat_max];
    __shared__ float rinv_s[B][n_stream];
    __shared__ int32_t completes[2]; // HC3: this block completed its first / last run

    const int32_t n_f = g < up_big_blocks ? up_feat_max : up_feat_max - 1;
    const int32_t c0 =
        g < up_big_blocks ? up_feat_max * g : up_big_blocks * up_feat_max + (up_feat_max - 1) * (g - up_big_blocks);
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    const int32_t o = warp % n_oct;
    const int32_t rg = warp / n_oct;
    const int32_t s = lane / 8;
    const int32_t k8 = lane % 8;
    const uint32_t mz = gridDim.x == n_block ? 0u : ~0u;
    const uint4 zero_u4 = make_uint4(0u, 0u, 0u, 0u);

    const uint4 * wblk = reinterpret_cast<const uint4 *>(a.w_up) + c0 * row_u4; // 16-byte aligned (launcher)
    const int32_t n_u = n_f * row_u4;
    const int32_t row0 = (c0 + rg * rows_per_warp) * n_stream + s;
    w_lane<Q> wrow;
    if constexpr (Q) {
        // the q plane, then the d plane (hc.h); a lane's 8 q are one 8-byte load and its block is o * 2 + k8 / 4
        const uint8_t * wb = p.w;
        wrow.q = reinterpret_cast<const uint2 *>(wb) + row0 * (n_down / 8) + o * 8 + k8;
        wrow.d = reinterpret_cast<const unsigned short *>(wb + n_own * n_stream * n_down) + row0 * (n_down / 32) +
                 o * 2 + k8 / 4;
        wrow.rs = n_stream * (n_down / 8);
        wrow.ds = n_stream * (n_down / 32);
    } else {
        wrow.p = reinterpret_cast<const uint4 *>(a.w_up) + row0 * (n_down / 8) + o * 8 + k8;
        wrow.rs = n_stream * (n_down / 8);
    }
    const int32_t q5 = min(rows_per_warp - 1, n_f - 1 - rg * rows_per_warp);
    typename w_slot<Q>::type w[D];
    if constexpr (ARM == 0) {
#pragma unroll
        for (int32_t q = 0; q < D; ++q) {
            w[q] = wrow.at(q < rows_per_warp - 1 ? q : q5);
        }
        __syncthreads(); // as HC1c: every weight load goes out before the reads below
    } else {
#pragma unroll
        for (int32_t q = 0; q < D; ++q) {
            const int32_t u = threadIdx.x + q * n_thread;
            w[q] = u < n_u ? __ldcs(wblk + u) : zero_u4;
        }
    }

    // own moments (lowering lanes) and the mix inputs (mix lanes): written before the kernel, plain 16-byte loads
    const float4 * mo4 = reinterpret_cast<const float4 *>(a.mom_own); // 16-byte aligned, checked by the launcher
    const uint4 * mp4 = reinterpret_cast<const uint4 *>(a.hc_ring);
    constexpr int32_t mrow4 = (H ? n_mom_head : n_mom) / 4;
    constexpr int32_t q4 = (4 + n_stream * n_down) / 4;
    const int32_t lb = warp / 3;
    const int32_t tt = 32 * (warp - 3 * lb) + lane;
    const bool lower = warp < n_low_warp && tt < 80;
    const bool injector = !H && warp < n_low_warp && tt == 80 && g == n_block - 1;
    const int32_t j4 = 16 * (tt / 16) + 2 * (tt & 7) + ((tt >> 3) & 1);
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 vo = zero4, uo[n_stream];
#pragma unroll
    for (int32_t si = 0; si < n_stream; ++si) {
        uo[si] = zero4;
    }
    int32_t at4[n_stream]; // the float4 of each stream's u (or q) this lane forms low (or inject) from
#pragma unroll
    for (int32_t si = 0; si < n_stream; ++si) {
        at4[si] = lb * mrow4 + (lower ? 1 + (n_down / 4) * si + j4 : q4 + si);
    }
    if (lower || injector) {
        vo = mo4[lb * mrow4];
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            uo[si] = mo4[at4[si]];
        }
    }
    // mix lane ml loads 16-byte group gq of pair mp (row mp / 4, stream mp % 4) of x and gamma
    const int32_t off = c0 & 3;
    const int32_t ml = 32 * (warp - mix_warp0) + lane;
    const int32_t mpair = ml / n_grp;
    const int32_t gq = ml % n_grp;
    const bool mixer = warp >= mix_warp0 && mpair < n_stream * B && 4 * gq < off + n_f;
    float4 xm4 = zero4, gm4 = zero4;
    if (mixer) {
        const int32_t w0 = c0 - off + 4 * gq; // a multiple of 4: 16-byte aligned
        xm4 = *reinterpret_cast<const float4 *>(a.x + mpair * n_own + w0);
        gm4 = *reinterpret_cast<const float4 *>(a.gamma + (mpair % n_stream) * n_own + w0);
    }

    if constexpr (ARM == 1) {
        // the stream into shared memory: slot k % D is stored, then refilled at an address that carries an opaque zero
        // from its data, so the refill waits for the data's arrival and no more than D loads a lane are in flight
#pragma unroll
        for (int32_t k = 0; k < n_stage; ++k) {
            const int32_t u = threadIdx.x + k * n_thread;
            if (u < n_u) {
                wsm[u] = w[k % D];
            }
            if (k + D < n_stage) {
                const int32_t un = u + D * n_thread + static_cast<int32_t>(w[k % D].x & mz);
                w[k % D] = un < n_u ? __ldcs(wblk + un) : zero_u4;
            }
        }
    }

    if (lower || injector) {
        // acquire X-HC: poll this lane's five 16-byte words of the peer's moments (xchg.h)
        const uint4 * pp[1 + n_stream] = {mp4 + lb * mrow4, mp4 + at4[0], mp4 + at4[1], mp4 + at4[2], mp4 + at4[3]};
        uint4 pv[1 + n_stream];
        fn::xchg::poll<1 + n_stream>(pp, pv, a.err, fn::xchg::err_poll_hc);
        // card order: card 0's moments, then card 1's
        const bool first = a.card == 0;
        const float4 vp = as_f4(pv[0]);
        const float4 v0 = first ? vo : vp, v1 = first ? vp : vo;
        float4 u0[n_stream], u1[n_stream];
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            const float4 up4 = as_f4(pv[1 + si]);
            u0[si] = first ? uo[si] : up4;
            u1[si] = first ? up4 : uo[si];
        }
        float ri[n_stream];
        ri[0] = 1.0f / sqrtf((v0.x + v1.x) / float(fn::n_embd) + fn::rms_eps);
        ri[1] = 1.0f / sqrtf((v0.y + v1.y) / float(fn::n_embd) + fn::rms_eps);
        ri[2] = 1.0f / sqrtf((v0.z + v1.z) / float(fn::n_embd) + fn::rms_eps);
        ri[3] = 1.0f / sqrtf((v0.w + v1.w) / float(fn::n_embd) + fn::rms_eps);
        float4 z = make_float4((u0[0].x + u1[0].x) * ri[0], (u0[0].y + u1[0].y) * ri[0],
                               (u0[0].z + u1[0].z) * ri[0], (u0[0].w + u1[0].w) * ri[0]);
#pragma unroll
        for (int32_t si = 1; si < n_stream; ++si) {
            z.x = fmaf(u0[si].x + u1[si].x, ri[si], z.x);
            z.y = fmaf(u0[si].y + u1[si].y, ri[si], z.y);
            z.z = fmaf(u0[si].z + u1[si].z, ri[si], z.z);
            z.w = fmaf(u0[si].w + u1[si].w, ri[si], z.w);
        }
        z = make_float4(0.25f * z.x, 0.25f * z.y, 0.25f * z.z, 0.25f * z.w);
        if (lower) {
            lows[lb][tt / 16][tt % 16] =
                make_float4(z.x * sigmoid(z.x), z.y * sigmoid(z.y), z.z * sigmoid(z.z), z.w * sigmoid(z.w));
            if (tt == 0) {
#pragma unroll
                for (int32_t si = 0; si < n_stream; ++si) {
                    rinv_s[lb][si] = ri[si];
                }
            }
        } else {
            reinterpret_cast<float4 *>(a.inject)[lb] =
                make_float4(2.0f * sigmoid(z.x), 2.0f * sigmoid(z.y), 2.0f * sigmoid(z.z), 2.0f * sigmoid(z.w));
        }
    }
    if (mixer) {
        const float xs[4] = {xm4.x, xm4.y, xm4.z, xm4.w};
        const float gs[4] = {gm4.x, gm4.y, gm4.z, gm4.w};
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            const int32_t f = 4 * gq + i - off;
            if (f >= 0 && f < n_f) {
                gx[mpair % n_stream][mpair / n_stream][f] = gs[i] * xs[i];
            }
        }
    }
    __syncthreads();

    float4 a0[B], a1[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        a0[b] = lows[b][o][k8];
        a1[b] = lows[b][o][k8 + 8];
    }

    uint32_t z = mz;
#pragma unroll
    for (int32_t q = 0; q < rows_per_warp; ++q) {
        const int32_t f = rg * rows_per_warp + q;
        if (f < n_f) {
            float acc[B];
            if constexpr (ARM == 0) {
#pragma unroll
                for (int32_t b = 0; b < B; ++b) {
                    acc[b] = dot8w(w[q % D], a0[b], a1[b], 0.0f, z);
                }
                z = order_after(acc[B - 1], mz);
                if (q + D < rows_per_warp) {
                    const int32_t qn = q + D < rows_per_warp - 1 ? q + D : q5;
                    w[q % D] = wrow.at(qn, z);
                }
            } else {
                const uint4 wq = wsm[(f * n_stream + s) * (n_down / 8) + o * 8 + k8]; // 8 lanes: 128 contiguous bytes
#pragma unroll
                for (int32_t b = 0; b < B; ++b) {
                    acc[b] = dot8(wq, a0[b], a1[b], 0.0f, mz);
                }
            }
#pragma unroll
            for (int32_t b = 0; b < B; ++b) {
#pragma unroll
                for (int32_t mm = 1; mm < 8; mm <<= 1) {
                    acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], mm);
                }
            }
            float v = acc[0];
#pragma unroll
            for (int32_t b = 1; b < B; ++b) {
                v = k8 == b ? acc[b] : v;
            }
            if (k8 < B) {
                red[o][s][k8][f] = v; // lanes (s, b): word offsets 18 (s B + b), distinct banks for 4B <= 16
            }
        }
    }
    __syncthreads();

    // Warp b finishes row b's mixed (lane f: feature c0 + f, HC1c's order) and sends it (X-HMIX) into the peer's slot
    // and this card's, as canonical bits (xchg.h), at the row's own half of the F32 region.
    using hl = fn::xchg::hmix_layout;
    if (warp < B) {
        const int32_t b = warp;
        const int32_t f = lane < n_f ? lane : 0;
        float mixed = 0.0f;
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            float zs = red[0][si][b][f];
#pragma unroll
            for (int32_t oo = 1; oo < n_oct; ++oo) {
                zs += red[oo][si][b][f];
            }
            mixed = fmaf(gx[si][b][f] * rinv_s[b][si], sigmoid(zs), mixed);
        }
        // F32 mixed handled as bits from here on
        const int32_t row0 = b * hl::width + a.card * n_own;
        uint32_t * own = reinterpret_cast<uint32_t *>(a.hmix_own) + row0;
        uint32_t * peer = reinterpret_cast<uint32_t *>(a.hmix_peer) + row0;
        fn::xchg::put_run(own, peer, c0, n_f, 0, fn::xchg::canonical(0.25f * mixed));
        __threadfence(); // publication of the F32 values to the block that completes their run
    }
    // HC3: the activation format, a run of 32 at a time. The block's 17 or 18 features touch runs j_lo and j_hi (at
    // most two); thread 0 and 1 add the block's count of each to its counter, and the block whose add completes 32
    // quantizes that run (hc.h, HC3's epilogue).
    const int32_t j_lo = c0 / 32, j_hi = (c0 + n_f - 1) / 32;
    __syncthreads();
    if (threadIdx.x < 2) {
        const int32_t j = threadIdx.x == 0 ? j_lo : j_hi;
        int32_t last = 0;
        if (threadIdx.x == 0 || j_hi != j_lo) {
            const uint32_t n_in = static_cast<uint32_t>(min(c0 + n_f, 32 * (j + 1)) - max(c0, 32 * j));
            if (atomicAdd(a.runs + j, n_in) + n_in == 32u) {
                __threadfence(); // acquire: the other blocks' F32 values of the run
                a.runs[j] = 0u;  // no other block adds to it again in this launch
                last = 1;
            }
        }
        completes[threadIdx.x] = last;
    }
    __syncthreads();
    if (warp < 2 && completes[warp] != 0) {
        // lane 8 b + i: values 4 i .. 4 i + 3 of the run in row b0 + b (rows past B pass zeros and store nothing);
        // SP2: four rows a pass, so B = 5 takes two
        const int32_t j = warp == 0 ? j_lo : j_hi;
#pragma unroll
        for (int32_t b0 = 0; b0 < B; b0 += 4) {
            const int32_t b = b0 + lane / 8;
            const bool real = b < B;
            float4 v = zero4;
            if (real) {
                v = __ldcg(reinterpret_cast<const float4 *>(a.hmix_own + b * hl::width + a.card * n_own + 32 * j) +
                           lane % 8);
            }
            uint32_t packed = 0u;
            float2 ds = make_float2(0.0f, 0.0f);
            fn::act::quantize_run8(v, packed, ds);
            fn::xchg::hmix_put_run(a.hmix_own, a.hmix_peer, b, a.card * n_runs_card + j, packed, ds, real);
        }
    }
}

template <int B, int ARM, bool H = false, bool Q = false>
__global__ void __launch_bounds__(n_thread, 1) hc_up2_kernel(const typename up2_param<Q>::type p) {
    up2_body<B, ARM, H, Q>(p, static_cast<int32_t>(blockIdx.x));
}

// FMS2 (build item 2, the joint catch-up): the draft's Q8_0 sublayer of up to four lanes in one launch, block
// (g, u) the lone launch's block g on lane u's arguments: the same body, so each lane's outputs are bitwise its lone
// launch's at B. gridDim.x stays n_block, so the lone order masks (mz) hold.
constexpr int32_t max_lanes = 4;

struct down2_q8_lanes_args {
    down2_q8_args a[max_lanes];
};

struct up2_q8_lanes_args {
    up2_q8_args a[max_lanes];
};

template <int B>
__global__ void __launch_bounds__(n_thread, 1) hc_down_lanes_kernel(const __grid_constant__ down2_q8_lanes_args p) {
    down_phase<B, shipped_depth_down, true, false, true, false, n_thread>(p.a[blockIdx.y], nullptr, blockIdx.x);
}

template <int B>
__global__ void __launch_bounds__(n_thread, 1) hc_up2_lanes_kernel(const __grid_constant__ up2_q8_lanes_args p) {
    up2_body<B, 0, false, true>(p.a[blockIdx.y], static_cast<int32_t>(blockIdx.x));
}

// HC4 (B2): hc_up's run-aligned shape, arm 0's math. Block g owns run g of the card's 40 runs of 32 features
// (features 32g .. 32g+31 in all four streams), so the mix and the activation quantize finish inside the block: no run
// counters, fences, atomics or rereads (HC3's 72-block shape split each run over two or three blocks). Warp (o, rg):
// low octet o (lane l = 8s + k8 owns low[64o + 8k8 .. 8k8+7] of stream s) and features 8rg .. 8rg+7, one 16-byte
// weight load a lane each, D in flight, consumed in order (HC1c); low and the mix inputs as arm 0 forms them. Every
// value goes through arm 0's operations in arm 0's order, so the outputs are bitwise arm 0's.
// FUSED (B3, inside hc_sub_kernel): the own moments are polled from this card's own region of the X-HC slot (own_ring)
// like the peer's, and x is recomputed from hres, hx and hpost (down's fmaf), so the phase reads nothing another
// block of its grid writes except through the polls.
constexpr int32_t n_run_feat = 32;                  // features of a run (act.h)
constexpr int32_t n_up_block = n_own / n_run_feat;  // 40
constexpr int32_t n_rgrp_up = 4;                    // feature groups of an up block
constexpr int32_t up_rows = n_run_feat / n_rgrp_up; // 8 features a warp
constexpr int32_t n_warp_up = n_oct * n_rgrp_up;    // 20
constexpr int32_t n_thread_up = 32 * n_warp_up;     // 640
static_assert(n_up_block * n_run_feat == n_own && n_up_block == n_runs_card, "an up block a run");

// SP2: the fused launch's dynamic region at B = 5 (down_phase, up_phase): the down phase's act, then the up phase's
// arrays over it
template <int B> struct up_dyn {
    float4 lows[B][n_oct][16];
    float red[n_oct][n_stream][B][n_run_feat + 1];
    float gx[n_stream][B][n_run_feat];
    float rinv_s[B][n_stream];
};
template <int B> constexpr int32_t sub_dyn_bytes() {
    constexpr int32_t act = static_cast<int32_t>(sizeof(float4)) * B * n_col * 64;
    constexpr int32_t up = static_cast<int32_t>(sizeof(up_dyn<B>));
    return B > 4 ? (act > up ? act : up) : 0;
}
static_assert(sub_dyn_bytes<5>() == 25600 && sizeof(up_dyn<5>) == 22240, "SP2: the B = 5 region [counted]");

// ME12, COMMON (the paired launch, FUSED): low and the inverse RMS come formed from common ([B][80] float4 of low in
// the stored [row][octet][float4] order, then [B][4] inverse RMS; pair_common wrote them before the grid barrier), so
// the phase polls no moment and has no injector; the arrays are in the dynamic region at every B (down_phase's U).
template <int B, bool H, bool FUSED, bool COMMON = false>
__device__ __forceinline__ void up_phase(const up2_args & a, const float * hres, const float * hx, const float * hpost,
                                         const float * own_ring, int32_t g, uint32_t mz,
                                         const float * common = nullptr) {
    constexpr int32_t D = shipped_depth_up;
    constexpr int32_t n_low_warp = 3 * B;
    constexpr int32_t n_mix_warp = n_warp_up - n_low_warp;
    constexpr int32_t n_grp = n_run_feat / 4;           // 16-byte groups of a (row, stream) of the run
    constexpr int32_t row_u4 = n_stream * (n_down / 8); // uint4 of one feature's weights, all streams
    static_assert(n_stream * B * n_grp <= 32 * n_mix_warp, "one 16-byte mix load a mix lane");
    static_assert(up_rows >= D, "the stream depth fits a warp's rows");

    // SP2: the fused launch at B = 5 takes these from the dynamic region, over the down phase's act (down_phase)
    static_assert(FUSED || !COMMON, "the common intermediate is the paired launch's");
    constexpr bool dyn = FUSED && (B > 4 || COMMON);
    constexpr int32_t bs = dyn ? 1 : B; // the static arrays' rows
    __shared__ float4 lows_st[bs][n_oct][16];
    // feature innermost, stride 33: the dot product's stores (lanes s, b) and the finish's reads are conflict-free
    __shared__ float red_st[n_oct][n_stream][bs][n_run_feat + 1];
    __shared__ float gx_st[n_stream][bs][n_run_feat];
    __shared__ float rinv_st[bs][n_stream];
    extern __shared__ float4 sub_dyn[];
    up_dyn<B> * const ud = reinterpret_cast<up_dyn<B> *>(sub_dyn);
    float4(*lows)[n_oct][16] = reinterpret_cast<float4(*)[n_oct][16]>(dyn ? &ud->lows[0][0][0] : &lows_st[0][0][0]);
    float(*red)[n_stream][B][n_run_feat + 1] =
        reinterpret_cast<float(*)[n_stream][B][n_run_feat + 1]>(dyn ? &ud->red[0][0][0][0] : &red_st[0][0][0][0]);
    float(*gx)[B][n_run_feat] = reinterpret_cast<float(*)[B][n_run_feat]>(dyn ? &ud->gx[0][0][0] : &gx_st[0][0][0]);
    float(*rinv_s)[n_stream] = reinterpret_cast<float(*)[n_stream]>(dyn ? &ud->rinv_s[0][0] : &rinv_st[0][0]);

    const int32_t c0 = n_run_feat * g;
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    const int32_t o = warp % n_oct;
    const int32_t rg = warp / n_oct;
    const int32_t s = lane / 8;
    const int32_t k8 = lane % 8;
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // the weights first (HC-6), D rows a lane, then a barrier: coherent loads cannot cross it (HC1-L5)
    const uint4 * wrow =
        reinterpret_cast<const uint4 *>(a.w_up) + ((c0 + rg * up_rows) * n_stream + s) * (n_down / 8) + o * 8 + k8;
    uint4 w[D];
#pragma unroll
    for (int32_t q = 0; q < D; ++q) {
        w[q] = __ldcs(wrow + q * row_u4);
    }
    __syncthreads();

    constexpr int32_t mrow4 = (H ? n_mom_head : n_mom) / 4;
    constexpr int32_t q4 = (4 + n_stream * n_down) / 4;
    const int32_t lb = warp / 3;
    const int32_t tt = 32 * (warp - 3 * lb) + lane;
    const bool lower = warp < n_low_warp && tt < 80;
    const bool injector = !H && !COMMON && warp < n_low_warp && tt == 80 && g == n_up_block - 1;
    const int32_t j4 = 16 * (tt / 16) + 2 * (tt & 7) + ((tt >> 3) & 1);
    int32_t at4[n_stream]; // the float4 of each stream's u (or q) this lane forms low (or inject) from
#pragma unroll
    for (int32_t si = 0; si < n_stream; ++si) {
        at4[si] = lb * mrow4 + (lower ? 1 + (n_down / 4) * si + j4 : q4 + si);
    }
    float4 vo = zero4, uo[n_stream];
#pragma unroll
    for (int32_t si = 0; si < n_stream; ++si) {
        uo[si] = zero4;
    }
    if (!FUSED && (lower || injector)) {
        const float4 * mo4 = reinterpret_cast<const float4 *>(a.mom_own); // 16-byte aligned, checked by the launcher
        vo = mo4[lb * mrow4];
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            uo[si] = mo4[at4[si]];
        }
    }
    // mix lane ml loads 16-byte group gq of pair mp (row mp / 4, stream mp % 4): x (or, fused, x's inputs) and gamma
    const int32_t ml = 32 * (warp - n_low_warp) + lane;
    const int32_t mpair = ml / n_grp;
    const int32_t gq = ml % n_grp;
    const bool mixer = warp >= n_low_warp && mpair < n_stream * B;
    float4 xm4 = zero4, gm4 = zero4, hv4 = zero4;
    float hp = 0.0f;
    if (mixer) {
        const int32_t w0 = c0 + 4 * gq; // a multiple of 4: 16-byte aligned
        gm4 = *reinterpret_cast<const float4 *>(a.gamma + (mpair % n_stream) * n_own + w0);
        if constexpr (FUSED) {
            xm4 = *reinterpret_cast<const float4 *>(hres + mpair * n_own + w0); // [b][s] rows, as x
            hv4 = *reinterpret_cast<const float4 *>(hx + (mpair / n_stream) * n_own + w0);
            hp = hpost[mpair];
        } else {
            xm4 = *reinterpret_cast<const float4 *>(a.x + mpair * n_own + w0);
        }
    }

    if constexpr (COMMON) {
        if (lower) { // written by another block before the grid barrier: L2, not L1
            lows[lb][tt / 16][tt % 16] = __ldcg(reinterpret_cast<const float4 *>(common) + lb * 80 + tt);
            if (tt == 0) {
#pragma unroll
                for (int32_t si = 0; si < n_stream; ++si) {
                    rinv_s[lb][si] = __ldcg(common + B * n_down + lb * n_stream + si);
                }
            }
        }
    } else if (lower || injector) {
        // acquire X-HC: poll this lane's five 16-byte words of the peer's moments (and, fused, of this card's own)
        const uint4 * mp4 = reinterpret_cast<const uint4 *>(a.hc_ring);
        uint4 pv[1 + n_stream];
        if constexpr (FUSED) {
            const uint4 * op4 = reinterpret_cast<const uint4 *>(own_ring);
            const uint4 * pp[2 * (1 + n_stream)] = {mp4 + lb * mrow4, mp4 + at4[0],     mp4 + at4[1], mp4 + at4[2],
                                                    mp4 + at4[3],     op4 + lb * mrow4, op4 + at4[0], op4 + at4[1],
                                                    op4 + at4[2],     op4 + at4[3]};
            uint4 v[2 * (1 + n_stream)];
            fn::xchg::poll<2 * (1 + n_stream)>(pp, v, a.err, fn::xchg::err_poll_hc);
#pragma unroll
            for (int32_t i = 0; i < 1 + n_stream; ++i) {
                pv[i] = v[i];
            }
            vo = as_f4(v[1 + n_stream]);
#pragma unroll
            for (int32_t si = 0; si < n_stream; ++si) {
                uo[si] = as_f4(v[2 + n_stream + si]);
            }
        } else {
            const uint4 * pp[1 + n_stream] = {mp4 + lb * mrow4, mp4 + at4[0], mp4 + at4[1], mp4 + at4[2], mp4 + at4[3]};
            fn::xchg::poll<1 + n_stream>(pp, pv, a.err, fn::xchg::err_poll_hc);
        }
        // card order: card 0's moments, then card 1's
        const bool first = a.card == 0;
        const float4 vp = as_f4(pv[0]);
        const float4 v0 = first ? vo : vp, v1 = first ? vp : vo;
        float4 u0[n_stream], u1[n_stream];
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            const float4 up4 = as_f4(pv[1 + si]);
            u0[si] = first ? uo[si] : up4;
            u1[si] = first ? up4 : uo[si];
        }
        float ri[n_stream];
        ri[0] = 1.0f / sqrtf((v0.x + v1.x) / float(fn::n_embd) + fn::rms_eps);
        ri[1] = 1.0f / sqrtf((v0.y + v1.y) / float(fn::n_embd) + fn::rms_eps);
        ri[2] = 1.0f / sqrtf((v0.z + v1.z) / float(fn::n_embd) + fn::rms_eps);
        ri[3] = 1.0f / sqrtf((v0.w + v1.w) / float(fn::n_embd) + fn::rms_eps);
        float4 z = make_float4((u0[0].x + u1[0].x) * ri[0], (u0[0].y + u1[0].y) * ri[0], (u0[0].z + u1[0].z) * ri[0],
                               (u0[0].w + u1[0].w) * ri[0]);
#pragma unroll
        for (int32_t si = 1; si < n_stream; ++si) {
            z.x = fmaf(u0[si].x + u1[si].x, ri[si], z.x);
            z.y = fmaf(u0[si].y + u1[si].y, ri[si], z.y);
            z.z = fmaf(u0[si].z + u1[si].z, ri[si], z.z);
            z.w = fmaf(u0[si].w + u1[si].w, ri[si], z.w);
        }
        z = make_float4(0.25f * z.x, 0.25f * z.y, 0.25f * z.z, 0.25f * z.w);
        if (lower) {
            lows[lb][tt / 16][tt % 16] =
                make_float4(z.x * sigmoid(z.x), z.y * sigmoid(z.y), z.z * sigmoid(z.z), z.w * sigmoid(z.w));
            if (tt == 0) {
#pragma unroll
                for (int32_t si = 0; si < n_stream; ++si) {
                    rinv_s[lb][si] = ri[si];
                }
            }
        } else {
            reinterpret_cast<float4 *>(a.inject)[lb] =
                make_float4(2.0f * sigmoid(z.x), 2.0f * sigmoid(z.y), 2.0f * sigmoid(z.z), 2.0f * sigmoid(z.w));
        }
    }
    if (mixer) {
        if constexpr (FUSED) { // down's x, the same fmaf
            xm4 = make_float4(fmaf(hp, hv4.x, xm4.x), fmaf(hp, hv4.y, xm4.y), fmaf(hp, hv4.z, xm4.z),
                              fmaf(hp, hv4.w, xm4.w));
        }
        const float xs[4] = {xm4.x, xm4.y, xm4.z, xm4.w};
        const float gs[4] = {gm4.x, gm4.y, gm4.z, gm4.w};
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            gx[mpair % n_stream][mpair / n_stream][4 * gq + i] = gs[i] * xs[i];
        }
    }
    __syncthreads();

    float4 a0[B], a1[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        a0[b] = lows[b][o][k8];
        a1[b] = lows[b][o][k8 + 8];
    }
    uint32_t z = mz;
#pragma unroll
    for (int32_t q = 0; q < up_rows; ++q) {
        float acc[B];
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            acc[b] = dot8(w[q % D], a0[b], a1[b], 0.0f, z);
        }
        z = order_after(acc[B - 1], mz);
        if (q + D < up_rows) { // the refill's address carries z, so it waits for this row's FMAs
            w[q % D] = __ldcs(wrow + (q + D) * row_u4 + z);
        }
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
#pragma unroll
            for (int32_t mm = 1; mm < 8; mm <<= 1) {
                acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], mm);
            }
        }
        float v = acc[0];
#pragma unroll
        for (int32_t b = 1; b < B; ++b) {
            v = k8 == b ? acc[b] : v;
        }
        if (k8 < B) {
            red[o][s][k8][rg * up_rows + q] = v;
        }
    }
    __syncthreads();

    // Warp b finishes row b (lane f: feature c0 + f, arm 0's order), sends the run's F32 values (X-HMIX, canonical
    // bits, xchg.h) into the peer's slot and this card's, then quantizes the run (act.h, lanes 0..7) and sends its
    // codes and {d, s}: the run is the block's own, so nothing waits on another block.
    using hl = fn::xchg::hmix_layout;
    if (warp < B) {
        const int32_t b = warp;
        const int32_t f = lane;
        float mixed = 0.0f;
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            float zs = red[0][si][b][f];
#pragma unroll
            for (int32_t oo = 1; oo < n_oct; ++oo) {
                zs += red[oo][si][b][f];
            }
            mixed = fmaf(gx[si][b][f] * rinv_s[b][si], sigmoid(zs), mixed);
        }
        const uint32_t bits = fn::xchg::canonical(0.25f * mixed); // F32 mixed handled as bits from here on
        const int32_t row0 = b * hl::width + a.card * n_own;
        fn::xchg::put_run(reinterpret_cast<uint32_t *>(a.hmix_own) + row0,
                          reinterpret_cast<uint32_t *>(a.hmix_peer) + row0, c0, n_run_feat, 0, bits);
        // quantize_run8's layout: lane i < 8 holds values 4i .. 4i+3 of the run (the stored bits)
        float4 v;
        v.x = __uint_as_float(__shfl_sync(0xffffffffu, bits, 4 * (lane & 7)));
        v.y = __uint_as_float(__shfl_sync(0xffffffffu, bits, 4 * (lane & 7) + 1));
        v.z = __uint_as_float(__shfl_sync(0xffffffffu, bits, 4 * (lane & 7) + 2));
        v.w = __uint_as_float(__shfl_sync(0xffffffffu, bits, 4 * (lane & 7) + 3));
        const bool real = lane < 8;
        uint32_t packed = 0u;
        float2 ds = make_float2(0.0f, 0.0f);
        fn::act::quantize_run8(real ? v : zero4, packed, ds);
        fn::xchg::hmix_put_run(a.hmix_own, a.hmix_peer, b, a.card * n_runs_card + g, packed, ds, real);
    }
}

template <int B, bool H> __global__ void __launch_bounds__(n_thread_up, 1) hc_up3_kernel(const up2_args a) {
    up_phase<B, H, false>(a, nullptr, nullptr, nullptr, nullptr, blockIdx.x, gridDim.x == n_up_block ? 0u : ~0u);
}

// HC4 (B3): the sublayer in one launch (hc.h sub). One cooperative grid of 72 blocks of 640 threads, one block an SM:
// every block runs the down phase (down2's work and sends, its own moments into the X-HC slot's own region), then blocks
// 0..39 run the up phase in its run-aligned shape, polling both cards' moments; blocks 40..71 end after their down
// phase. Mode bits: 1 the down phase, 2 the up phase (3 in the model's graphs; 1 then 2 in the eager pair order).
struct sub_params {
    down2_args d;       // d.d.mom is not used: the own moments go to own
    up2_args u;         // u.mom_own, u.x and u.runs are not used
    const float * hres; // x's inputs, read again by the up phase
    const float * hx;
    const float * hpost;
    float * own; // this card's own region of the X-HC slot (xchg.h hc_own_at)
};

template <int B, bool H>
__global__ void __launch_bounds__(n_thread_up, 1) hc_sub_kernel(const sub_params p, int32_t mode) {
    if ((mode & 1) != 0) {
        down_phase<B, shipped_depth_down, true, H, false, true, n_thread_up>(p.d, p.own, blockIdx.x);
    }
    if ((mode & 2) != 0 && blockIdx.x < n_up_block) {
        up_phase<B, H, true>(p.u, p.hres, p.hx, p.hpost, p.own, blockIdx.x, gridDim.x == n_block ? 0u : ~0u);
    }
}

// FMU2 (hc.h sub_joint; build order item 1, Fable review 2 section 1), reshaped by FMS2 items 5 and 5b: a joint round's
// sublayer, U lanes of max_rows rows on the lanes' one weight set, one weight stream. Every lane's rows go through
// hc_sub_kernel<5>'s per-row arithmetic, so a lane's outputs are bitwise its own sub launch's:
// - down: warp (c, rg) loads its 5 packed weight rows (FUSED's rows, the clamp included) and widens them to F32 each
//   pass; each activation row of each lane takes dot8's 8 fmaf from 0.0f with each row (dot8f), then the 32-lane sum
//   in the butterfly's xor order 16, 8, 4, 2, 1 as a reduce-scatter over the 5 rows (rs5): every partial a lane forms
//   is the butterfly's partial at that lane (a + b is b + a exactly, so a butterfly's lanes hold equal bits at each
//   step), so each row's sum is the butterfly's. The finish sums the 5 columns in order, as down_phase's. A pass
//   stages two lanes (10 rows; 20 would pass the 96 KB of shared memory) at once, warps 10 h .. 10 h + 9 lane
//   k0 + h; the row loop is rolled and the (at most two) passes unrolled. Each lane's moments go out as the block's
//   one 90-word run of the joint X-HC layout (xchg.h hcj_run_at), 16-byte stores with 8-byte ends.
// - up: on all 72 blocks, in two lane groups (A: down pass 0's lanes, B: the rest), each group's (run, row) pairs
//   split evenly over the blocks (up_seg_of). A group polls its slots' moments and v words and places them in the
//   lone row order (each card-ordered sum of two moments formed as peer + own: the same bits, as F32 addition
//   commutes), forms up_phase's lows, inverse RMS and gamma x, then warp (o, rg) takes each slot's dot8f with its 8
//   packed feature rows (widened at each use), the 8-lane sum in the butterfly's order 1, 2, 4 as a reduce-scatter
//   (rs8), and warp i finishes slot i in up_phase's order, sending its X-HMIX before group B's dots. The group loop
//   is rolled; a run's packed weights stay in registers across the groups.
// Each lane writes its own X-HC and X-HMIX slots, x and inject; each lane's previous slots are reset before the
// block's second barrier and the moment senders fence after it (down_phase FUSED's order). The lane count is a
// template parameter, so no lane parameter is indexed at run time (the spill of the per-lane form).
// FMS13: LR rows a lane (the round's R = d + 1, 2 .. 5; max_rows before), a template parameter beside U: each lane's
// rows 0 .. LR - 1 are staged, dotted, finished and sent (a run of 18 LR words, then its LR v words) and polled, and
// up's groups follow U LR (group A one lane at up to 10 rows, as U = 2's, else two). Rows LR .. 4 of a lane's slots
// and buffers are neither written nor polled; every row's arithmetic is hc_sub_kernel<LR>'s.
struct sub_joint_params {
    sub_params p[fn::xchg::max_lanes];
};

// acc's 8 fmaf from 0.0f of dot8 with the row's widened weights w (bf16_lo(x), bf16_hi(x), .. of the 16 bytes)
__device__ __forceinline__ float dot8f(const float (&w)[8], float4 a0, float4 a1) {
    float acc = 0.0f;
    acc = fmaf(w[0], a0.x, acc);
    acc = fmaf(w[1], a0.y, acc);
    acc = fmaf(w[2], a0.z, acc);
    acc = fmaf(w[3], a0.w, acc);
    acc = fmaf(w[4], a1.x, acc);
    acc = fmaf(w[5], a1.y, acc);
    acc = fmaf(w[6], a1.z, acc);
    acc = fmaf(w[7], a1.w, acc);
    return acc;
}

__device__ __forceinline__ void widen8(uint4 w, float (&f)[8]) {
    f[0] = bf16_lo(w.x, 0u);
    f[1] = bf16_hi(w.x, 0u);
    f[2] = bf16_lo(w.y, 0u);
    f[3] = bf16_hi(w.y, 0u);
    f[4] = bf16_lo(w.z, 0u);
    f[5] = bf16_hi(w.z, 0u);
    f[6] = bf16_lo(w.w, 0u);
    f[7] = bf16_hi(w.w, 0u);
}

// one step of a reduce-scatter: the lane keeps k and sends the partner its s; returns k + the partner's send
template <int O> __device__ __forceinline__ float rs_step(float k, float s) {
    return k + __shfl_xor_sync(0xffffffffu, s, O);
}

// the 32-lane sums of v[0..4] in the butterfly order 16, 8, 4, 2, 1 as a reduce-scatter: returns value r's sum, r by
// the lane's bits 4, 3, 2: 0 (lanes 0-3), 1 (4-7), 2 (8-15), 3 (16-23), 4 (24-31)
__device__ __forceinline__ float rs5(const float (&v)[5], int32_t lane, int32_t & r) {
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
    // offset 16: bit 4 clear keeps 0, 1, 2 (h0, h1, h2); set keeps 3, 4 (h0, h1)
    const float h0 = rs_step<16>(b4 ? v[3] : v[0], b4 ? v[0] : v[3]);
    const float h1 = rs_step<16>(b4 ? v[4] : v[1], b4 ? v[1] : v[4]);
    const float h2 = rs_step<16>(v[2], v[2]);
    // offset 8: of 0, 1, 2 bit 3 clear keeps 0, 1 (g0, g1), set keeps 2; of 3, 4 clear keeps 3, set 4 (g0)
    const float other = b4 ? h1 : h2;
    const float g0 = rs_step<8>(b3 ? other : h0, b3 ? h0 : other);
    const float g1 = rs_step<8>(h1, h1);
    // offset 4: of 0, 1 bit 2 clear keeps 0, set keeps 1; every other lane's one value goes on as a butterfly
    const bool two = !b4 && !b3;
    float f = rs_step<4>(two && b2 ? g1 : g0, two && !b2 ? g1 : g0);
    f += __shfl_xor_sync(0xffffffffu, f, 2);
    f += __shfl_xor_sync(0xffffffffu, f, 1);
    r = !b4 ? (!b3 ? (!b2 ? 0 : 1) : 2) : (!b3 ? 3 : 4);
    return f;
}

// the 8-lane sums (lane bits 0, 1, 2) of v[0..7] in the butterfly order 1, 2, 4 as a reduce-scatter: returns value q's
// sum, q = 4 bit0 + 2 bit1 + bit2
__device__ __forceinline__ float rs8(const float (&v)[8], int32_t lane, int32_t & q) {
    const bool b0 = (lane & 1) != 0, b1 = (lane & 2) != 0, b2 = (lane & 4) != 0;
    float h[4], g[2];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) { // offset 1: bit 0 clear keeps 0..3, set keeps 4..7
        h[i] = rs_step<1>(b0 ? v[4 + i] : v[i], b0 ? v[i] : v[4 + i]);
    }
#pragma unroll
    for (int32_t i = 0; i < 2; ++i) { // offset 2: bit 1 clear keeps the first two of the four, set the last two
        g[i] = rs_step<2>(b1 ? h[2 + i] : h[i], b1 ? h[i] : h[2 + i]);
    }
    q = (b0 ? 4 : 0) + (b1 ? 2 : 0) + (b2 ? 1 : 0);
    return rs_step<4>(b2 ? g[1] : g[0], b2 ? g[0] : g[1]); // offset 4: bit 2 clear keeps the first, set the second
}

// the thread and block index read afresh (asm volatile), so the values formed from them are not held across the
// down passes and up's groups (they spilled)
__device__ __forceinline__ int32_t fresh_tid() {
    int32_t t;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
    return t;
}
__device__ __forceinline__ int32_t fresh_block() {
    int32_t b;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(b));
    return b;
}

// the joint launch's dynamic region at U lanes: the down phase's act (a pass's two lanes' rows), sums, squares and the
// moment runs it sends; the up phase's slots over it (the down phase ends with a barrier first)
constexpr int32_t wide_pass = 2;                      // lanes staged a down pass
constexpr int32_t stage_half = n_warp_up / wide_pass; // FMS2 item 5: a pass's lane k0 + h is staged by warps 10 h ..
static_assert(stage_half == n_stage_warp && down_rows == fn::xchg::hcj_rows_block && n_block == fn::xchg::hcj_blocks,
              "a half of the block stages a lane (one float4 of each row a lane); the joint X-HC layout's blocks");
template <int U> struct wide_down {
    float4 act[wide_pass * max_rows][n_col][64];
    float red[U][max_rows][n_col][20]; // [lane][row][column][the block's 20 row slots]: down_phase's redx a lane
    float ssp[U][max_rows][n_stage_warp];
    uint32_t img[wide_pass][fn::xchg::hcj_block_words]; // FMS2 item 5b: a pass's lanes' moment runs, as sent
};

// FMS2 items 5 and 5b: up's lane groups and slots. Group 0 (A) is the lanes of down's first pass (at U = 2 lane 0),
// group 1 (B) the rest. A group's rows (lane-major, row LR k + b) and the card's 40 runs make 40 x rows (run, row)
// pairs, run-major; block i takes pairs 40 rows i / 72 .. 40 rows (i + 1) / 72 of each group, its slots. FMS13: keyed
// on the round's rows U LR, U = 2's groups (A one lane) at up to 10 rows, else two lanes in A.
template <int U, int LR> __host__ __device__ constexpr int32_t group_rows(int32_t grp) {
    return grp == 0 ? LR * (U * LR <= 10 ? 1 : wide_pass) : LR * U - LR * (U * LR <= 10 ? 1 : wide_pass);
}

template <int U, int LR> __host__ __device__ constexpr int32_t up_slots() {
    constexpr int32_t r = group_rows<U, LR>(0) > group_rows<U, LR>(1) ? group_rows<U, LR>(0) : group_rows<U, LR>(1);
    return (n_runs_card * r + n_block - 1) / n_block;
}

template <int U, int LR> struct wide_up {
    float  mom[up_slots<U, LR>()][n_mom];  // a slot's row of both cards' moments (peer + own), in the lone row layout
    float4 lows[up_slots<U, LR>()][n_oct][16];
    // by group: the next group's mixers run during this one's finish
    float  gx[2][up_slots<U, LR>()][n_stream][n_run_feat];
    float  rinv[2][up_slots<U, LR>()][n_stream];  // by group, as gx (formed during the polls)
    float  red[up_slots<U, LR>()][n_oct][n_stream][n_run_feat + 1];
};

template <int U, int LR> constexpr int32_t wide_dyn_bytes() {
    constexpr int32_t d = static_cast<int32_t>(sizeof(wide_down<U>));
    constexpr int32_t u = static_cast<int32_t>(sizeof(wide_up<U, LR>));
    return d > u ? d : u;
}

static_assert(wide_dyn_bytes<4, max_rows>() <= 96 * 1024 && sizeof(wide_down<4>) == 60720 &&
                  sizeof(wide_up<4, max_rows>) == 61056 && up_slots<4, max_rows>() == 6 &&
                  up_slots<3, max_rows>() == 6 && up_slots<2, max_rows>() == 3 && (n_mom * 4) % 16 == 0,
              "the joint launch's region [counted]");

// down: stager loads of lane q's rows (down_phase's), held until stage_store
struct wide_stage {
    float4 hr[max_rows], hv[max_rows];
    float hp[max_rows];
};

template <int LR>
__device__ __forceinline__ void stage_load(const sub_params & q, int32_t s, int32_t f4, wide_stage & st) {
    const float4 * hres4 = reinterpret_cast<const float4 *>(q.d.d.hres) + s * (n_own / 4) + f4;
    const float4 * hx4 = reinterpret_cast<const float4 *>(q.d.d.hx) + f4;
#pragma unroll
    for (int32_t b = 0; b < LR; ++b) {
        st.hr[b] = __ldcg(hres4 + b * n_stream * (n_own / 4));
        st.hv[b] = __ldcg(hx4 + b * (n_own / 4));
        st.hp[b] = __ldcg(q.d.d.hpost + b * n_stream + s);
    }
}

// down_phase's staging of one lane's rows into act rows a0 .., its x and (blocks g = 0) its squares' partials, by warp
// w of the lane's ten
template <int U, int LR>
__device__ __forceinline__ void stage_store(const sub_params & q,
                                            int32_t            k,
                                            const wide_stage & st,
                                            float4             gm,
                                            int32_t            s,
                                            int32_t            g,
                                            int32_t            w,
                                            int32_t            lane,
                                            int32_t            f4,
                                            wide_down<U> &     sd,
                                            int32_t            a0) {
    const int32_t slot = 16 * (w & 1) + (lane & 15) + 32 * (lane >> 4);
    float         ss[LR];
#pragma unroll
    for (int32_t b = 0; b < LR; ++b) {
        float4 x;
        x.x = fmaf(st.hp[b], st.hv[b].x, st.hr[b].x);
        x.y = fmaf(st.hp[b], st.hv[b].y, st.hr[b].y);
        x.z = fmaf(st.hp[b], st.hv[b].z, st.hr[b].z);
        x.w = fmaf(st.hp[b], st.hv[b].w, st.hr[b].w);
        if (w == g) {
            reinterpret_cast<float4 *>(q.d.d.x)[(b * n_stream + s) * (n_own / 4) + f4] = x;
        }
        sd.act[a0 + b][w / 2][slot] = make_float4(x.x * gm.x, x.y * gm.y, x.z * gm.z, x.w * gm.w);
        ss[b] = x.x * x.x;
        ss[b] = fmaf(x.y, x.y, ss[b]);
        ss[b] = fmaf(x.z, x.z, ss[b]);
        ss[b] = fmaf(x.w, x.w, ss[b]);
    }
    if (g == 0) {
#pragma unroll
        for (int32_t b = 0; b < LR; ++b) {
#pragma unroll
            for (int32_t o = 16; o > 0; o >>= 1) {
                ss[b] += __shfl_xor_sync(0xffffffffu, ss[b], o);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int32_t b = 0; b < LR; ++b) {
                sd.ssp[k][b][w] = ss[b];
            }
        }
    }
}

template <int U, int LR, bool H> __device__ __forceinline__ void down_wide(const sub_joint_params & jp) {
    constexpr int32_t        B         = LR;
    constexpr int32_t        run_words = LR * fn::xchg::hcj_rows_block;  // the lane's run: row b's words at 18 b
    constexpr int32_t rpw = 5;                             // FUSED: 4 row groups of 5 rows a warp
    constexpr int32_t n_rows_w = H ? n_down : n_rows_down; // weight rows a stream
    constexpr int32_t n_pass = (U + wide_pass - 1) / wide_pass;
    extern __shared__ float4 sub_dyn[];
    wide_down<U> & sd = *reinterpret_cast<wide_down<U> *>(sub_dyn);
    const int32_t vb = static_cast<int32_t>(blockIdx.x);
    const int32_t s = vb % n_stream;
    const int32_t g = vb / n_stream;
    const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
    // item 5: warps 10 h .. 10 h + 9 stage lane k0 + h of a pass, warp w of them float4 f4 of each row (read afresh
    // each pass: held across a pass they spilled)
    const auto stage_ix = [](int32_t & half, int32_t & w, int32_t & f4, int32_t & sv, int32_t & gv) {
        const int32_t t = fresh_tid();
        half = t / 32 / stage_half;
        w = t / 32 % stage_half;
        f4 = 32 * w + 2 * (t % 32 & 15) + (t % 32 >> 4);
        sv = fresh_block() % n_stream;
        gv = fresh_block() / n_stream;
    };
    // every lane's release of its previous sublayer's slots (xchg.h step 4, HC4), before the second barrier: first,
    // so the sentinels are not live with the staging and the weights
    const int32_t u = vb * n_thread_up + static_cast<int32_t>(threadIdx.x);
#pragma unroll
    for (int32_t k = 0; k < U; ++k) {
        if (jp.p[k].d.hc_reset != nullptr && u < hc_slot_u4) {
            fn::xchg::reset4(reinterpret_cast<uint4 *>(jp.p[k].d.hc_reset) + u);
        }
        if (jp.p[k].d.hmix_reset != nullptr) {
            fn::xchg::hmix_reset_slot(jp.p[k].d.hmix_reset, u);
        }
    }
    wide_stage st;
    float4 gm;
    {
        int32_t half, w, f4, sv, gv;
        stage_ix(half, w, f4, sv, gv);
        stage_load<LR>(jp.p[half], sv, f4, st);  // pass 0's lanes 0 and 1 (U >= 2)
        gm = __ldcg(reinterpret_cast<const float4 *>(jp.p[0].d.d.gamma) + sv * (n_own / 4) + f4);
    }
    // every staging load before any weight load (down_phase)
    __syncthreads();
    const int32_t c = warp % n_col;
    const int32_t rg = warp / n_col;
    const int32_t row0 = s * n_rows_w + g * down_rows + rg * rpw;
    const uint4 * wp = reinterpret_cast<const uint4 *>(jp.p[0].d.d.w_down) + (row0 * n_own + c * 256) / 8 + lane;
    const int32_t qmax = min(n_rows_w - g * down_rows, down_rows) - 1 - rg * rpw;
    // one pass: streamed (down_phase); two: kept in L2 and loaded again after the second pass's staging (held across
    // it they spill)
    uint4 w4[rpw];
#pragma unroll
    for (int32_t q = 0; q < rpw; ++q) {
        w4[q] = n_pass > 1 ? __ldcg(wp + min(q, qmax) * (n_own / 8)) : __ldcs(wp + min(q, qmax) * (n_own / 8));
    }
    // item 5: the rows rolled (one copy of the dot body), the passes (at most two) unrolled; a pass's two lanes staged
    // at once, one a half of the warps
#pragma unroll
    for (int32_t pass = 0; pass < n_pass; ++pass) {
        const int32_t k0 = pass * wide_pass;
        {
            int32_t half, w, f4, sv, gv;
            stage_ix(half, w, f4, sv, gv);
            const int32_t kh = k0 + half;
            if (kh < U) {
                if (pass > 0) {
                    stage_load<LR>(jp.p[kh], sv, f4, st);
                }
                stage_store<U, LR>(jp.p[kh], kh, st, gm, sv, gv, w, fresh_tid() % 32, f4, sd, half * B);
            }
        }
        __syncthreads();
        if (pass == 0 && warp < wide_pass) { // the moment senders: every reset above before any moment store
            __threadfence();
        }
        if (pass > 0) {
#pragma unroll
            for (int32_t q = 0; q < rpw; ++q) {
                w4[q] = __ldcg(wp + min(q, qmax) * (n_own / 8));
            }
        }
        float wf[rpw][8];
#pragma unroll
        for (int32_t q = 0; q < rpw; ++q) {
            widen8(w4[q], wf[q]);
        }
        const int32_t n_j = min(U - k0, wide_pass) * B; // the pass's staged rows
#pragma unroll 1
        for (int32_t j = 0; j < n_j; ++j) {
            const float4 a0 = sd.act[j][c][lane];
            const float4 a1 = sd.act[j][c][lane + 32];
            float v[rpw];
#pragma unroll
            for (int32_t q = 0; q < rpw; ++q) {
                v[q] = dot8f(wf[q], a0, a1);
            }
            int32_t r = 0;
            const float sum = rs5(v, lane, r);
            if ((lane & 3) == 0 && (lane < 8 || (lane & 7) == 0)) { // lanes 0, 4, 8, 16, 24: rows 0..4
                sd.red[k0 + j / B][j % B][c][rg * rpw + r] = sum;
            }
        }
        __syncthreads();
        // down_phase's finish of the pass's lanes, its sums in its order; item 5b: warp h sends lane k0 + h's run of
        // the joint X-HC layout (row b's down_rows words at 18 b, a short head block's last words zero)
        if (warp < wide_pass && k0 + warp < U) {
            const int32_t k = k0 + warp;
            const int32_t n_w = H ? min(down_rows, n_down - g * down_rows) : down_rows;
#pragma unroll
            for (int32_t i0 = 0; i0 < run_words; i0 += 32) {
                const int32_t i = i0 + lane;
                if (i < run_words) {
                    const int32_t b = i / down_rows;
                    const int32_t r = i - b * down_rows;
                    float v = sd.red[k][b][0][r];
#pragma unroll
                    for (int32_t cc = 1; cc < n_col; ++cc) {
                        v += sd.red[k][b][cc][r];
                    }
                    sd.img[warp][i] = r < n_w ? fn::xchg::canonical(v) : 0u;
                }
            }
            __syncwarp();
            fn::xchg::put_words(reinterpret_cast<uint32_t *>(jp.p[k].own),
                                reinterpret_cast<uint32_t *>(jp.p[k].d.hc_peer), fn::xchg::hcj_run_at(vb, 0), run_words,
                                sd.img[warp]);
        }
        if (g == 0 && static_cast<int32_t>(threadIdx.x) < wide_pass * B &&
            k0 + static_cast<int32_t>(threadIdx.x) / B < U) {
            const int32_t k = k0 + static_cast<int32_t>(threadIdx.x) / B;
            const int32_t b = static_cast<int32_t>(threadIdx.x) % B;
            float v = sd.ssp[k][b][0];
#pragma unroll
            for (int32_t i = 1; i < n_stage_warp; ++i) {
                v += sd.ssp[k][b][i];
            }
            const uint32_t bits = fn::xchg::canonical(v);
            const int32_t at = 4 * b + fresh_block() % n_stream;  // item 5b: row b's v words at 4 b
            reinterpret_cast<uint32_t *>(jp.p[k].own)[at] = bits; // F32 moment as its bits
            reinterpret_cast<uint32_t *>(jp.p[k].d.hc_peer)[at] = bits;
        }
    }
}

// up's slots of one lane group in one block: slot i is the group's pair lo + i, run (lo + i) / rows, row
// row0 + (lo + i) % rows (lane-major)
struct up_seg {
    int32_t lo, n, rows, row0;
    __device__ __forceinline__ int32_t run(int32_t i) const { return (lo + i) / rows; }
    __device__ __forceinline__ int32_t row(int32_t i) const { return row0 + (lo + i) % rows; }
};

template <int U, int LR> __device__ __forceinline__ up_seg up_seg_of(int32_t grp, int32_t blk) {
    const int32_t rows  = grp == 0 ? group_rows<U, LR>(0) : group_rows<U, LR>(1);
    const int32_t pairs = n_runs_card * rows;
    const int32_t lo = blk * pairs / n_block;
    return { lo, (blk + 1) * pairs / n_block - lo, rows, grp == 0 ? 0 : group_rows<U, LR>(0) };
}

// item 5b: the moment polls of a slot, task j of 361: j < 360 the 16-byte group j % 5 of down block j / 5's run of the
// slot's row (5 groups cover each 18-word run: it starts at an even word), j = 360 the row's v group. Gives the
// group's two pointers (this card's ring: the peer's region and the own region) and the run's words in it (mask).
constexpr int32_t mom_groups = 5;
constexpr int32_t mom_tasks = n_block * mom_groups + 1;
struct mom_task {
    int32_t i, w4, a, vb; // slot, the group (16-byte index in the region), the run's first word, its block (-1: v)
    uint32_t mask;
    const uint4 * peer;
    const uint4 * own;
    uint32_t * err; // the row's lane's error word
};

template <int U, int LR, bool H>
__device__ __forceinline__ mom_task mom_task_of(const sub_joint_params & jp, const up_seg & seg, int32_t task) {
    mom_task t = {task / mom_tasks, 0, 0, -1, 0u, nullptr, nullptr, nullptr};
    if (t.i >= seg.n) {
        return t;
    }
    const int32_t j = task - t.i * mom_tasks;
    const int32_t row = seg.row(t.i);
    const int32_t k   = row / LR;
    const int32_t b   = row - k * LR;
    if (j == n_block * mom_groups) {
        t.w4 = b; // v: words 4 b .. 4 b + 3
        t.mask = 0xfu;
    } else {
        t.vb = j / mom_groups;
        const int32_t gv = t.vb / n_stream;
        const int32_t n_w = H ? min(down_rows, n_down - gv * down_rows) : down_rows;
        t.a = fn::xchg::hcj_run_at(t.vb, b);
        t.w4 = (t.a >> 2) + (j - t.vb * mom_groups);
        const int32_t lo = max(4 * t.w4, t.a) - 4 * t.w4;
        const int32_t hi = min(4 * t.w4 + 4, t.a + n_w) - 4 * t.w4;
        t.mask = lo < hi ? ((1u << hi) - 1u) & ~((1u << lo) - 1u) : 0u;
    }
    t.peer = reinterpret_cast<const uint4 *>(jp.p[k].u.hc_ring) + t.w4;
    t.own = reinterpret_cast<const uint4 *>(jp.p[k].own) + t.w4;
    t.err = jp.p[k].u.err;
    return t;
}

// a task's polled words into its slot's moment row (lone layout), each the sum peer + own (up_phase's card-ordered
// sum: F32 addition commutes bit for bit)
template <int U, int LR>
__device__ __forceinline__ void mom_place(const mom_task & t, uint4 pv, uint4 ov, wide_up<U, LR> & su, int32_t gb) {
    const uint32_t pw[4] = {pv.x, pv.y, pv.z, pv.w};
    const uint32_t ow[4] = {ov.x, ov.y, ov.z, ov.w};
#pragma unroll
    for (int32_t e = 0; e < 4; ++e) {
        if ((t.mask >> e & 1u) != 0u) {
            int32_t at = e; // v: stream e at word e
            if (t.vb >= 0) {
                const int32_t sv = t.vb % n_stream;
                const int32_t jr = (t.vb / n_stream) * down_rows + 4 * t.w4 + e - t.a; // the stream's moment row
                at = jr < n_down ? 4 + n_down * sv + jr : 4 + n_stream * n_down + n_inject * sv + (jr - n_down);
            }
            su.mom[t.i][at] = __uint_as_float(pw[e]) + __uint_as_float(ow[e]);
        }
    }
    if (t.vb < 0 && t.mask != 0u) { // the v group: up_phase's inverse RMS of the slot's row, one a stream
#pragma unroll
        for (int32_t e = 0; e < n_stream; ++e) {
            const float m = __uint_as_float(pw[e]) + __uint_as_float(ow[e]);
            su.rinv[gb][t.i][e] = 1.0f / sqrtf(m / float(fn::n_embd) + fn::rms_eps);
        }
    }
}

// up_phase's gamma x (the mixers, threads past the moment tasks' first) and moment polls of a group's slots
template <int U, int LR, bool H>
__device__ __forceinline__ void up_moments(const sub_joint_params & jp,
                                           const up_seg &           seg,
                                           wide_up<U, LR> &         su,
                                           int32_t                  gb) {
    constexpr int32_t n_grp = n_run_feat / 4;
    constexpr int32_t n_mix      = up_slots<U, LR>() * n_stream * n_grp;
    constexpr int32_t n_task     = up_slots<U, LR>() * mom_tasks;
    // FMS13: rounded up to the pairs (tasks past the slots poll nothing: mom_task_of's mask 0)
    constexpr int32_t per_thread = ((n_task + n_thread_up - 1) / n_thread_up + 1) / 2 * 2;
    static_assert(n_mix <= n_thread_up && per_thread % 2 == 0, "the mixers and the polls' pairs");
    const int32_t t = fresh_tid();
    const int32_t mt = t - (n_thread_up - n_mix);
    if (mt >= 0 && mt / (n_stream * n_grp) < seg.n) { // up_phase's mixer of (slot, stream, 4 features)
        const int32_t i = mt / (n_stream * n_grp);
        const int32_t si = (mt / n_grp) % n_stream;
        const int32_t gq = mt % n_grp;
        const int32_t row = seg.row(i);
        const int32_t      k     = row / LR;
        const int32_t      b     = row - k * LR;
        const sub_params & q = jp.p[k];
        const int32_t mpair = b * n_stream + si;
        const int32_t w0 = n_run_feat * seg.run(i) + 4 * gq;
        const float4 gm4 = *reinterpret_cast<const float4 *>(q.u.gamma + si * n_own + w0);
        float4 xm4 = *reinterpret_cast<const float4 *>(q.hres + mpair * n_own + w0);
        const float4 hv4 = *reinterpret_cast<const float4 *>(q.hx + b * n_own + w0);
        const float hp = q.hpost[mpair];
        xm4 = make_float4(fmaf(hp, hv4.x, xm4.x), fmaf(hp, hv4.y, xm4.y), fmaf(hp, hv4.z, xm4.z),
                          fmaf(hp, hv4.w, xm4.w));
        const float xs[4] = {xm4.x, xm4.y, xm4.z, xm4.w};
        const float gs[4] = {gm4.x, gm4.y, gm4.z, gm4.w};
#pragma unroll
        for (int32_t e = 0; e < 4; ++e) {
            su.gx[gb][i][si][4 * gq + e] = gs[e] * xs[e];
        }
    }
    // the polls, two tasks at a time (both regions: four groups in flight)
#pragma unroll 1
    for (int32_t r = 0; r < per_thread; r += 2) {
        const mom_task      t0   = mom_task_of<U, LR, H>(jp, seg, t + n_thread_up * r);
        const mom_task      t1   = mom_task_of<U, LR, H>(jp, seg, t + n_thread_up * (r + 1));
        const uint4 * const p[4] = {t0.peer, t0.own, t1.peer, t1.own};
        const uint32_t m[4] = {t0.mask, t0.mask, t1.mask, t1.mask};
        uint4 v[4];
        const uint32_t late = fn::xchg::poll_masked<4>(p, v, m);
        if ((late & 3u) != 0u) {
            atomicOr(t0.err, fn::xchg::err_poll_hc);
        }
        if ((late & 12u) != 0u) {
            atomicOr(t1.err, fn::xchg::err_poll_hc);
        }
        mom_place<U, LR>(t0, v[0], v[1], su, gb);
        mom_place<U, LR>(t1, v[2], v[3], su, gb);
    }
}

// up_phase's lows, inverse RMS and injection of a group's slots from their moment rows: thread 81 i + tt, tt < 80
// the lows' float4 tt of slot i's row, tt = 80 its injection (by the block holding the row's run 0)
template <int U, int LR, bool H>
__device__ __forceinline__ void up_lows(const sub_joint_params & jp,
                                        const up_seg &           seg,
                                        wide_up<U, LR> &         su,
                                        int32_t                  gb) {
    constexpr int32_t q4 = (4 + n_stream * n_down) / 4;
    const int32_t t = fresh_tid();
    const int32_t i = t / 81;
    const int32_t tt = t - 81 * i;
    if (i >= seg.n) {
        return;
    }
    const bool lower = tt < 80;
    if (!lower && (H || seg.run(i) != 0)) {
        return;
    }
    const int32_t j4 = 16 * (tt / 16) + 2 * (tt & 7) + ((tt >> 3) & 1);
    const float4 * m4 = reinterpret_cast<const float4 *>(su.mom[i]);
    const float * ri = su.rinv[gb][i]; // formed by the v group's poller (mom_place)
    // up_phase's sum over the streams in its order, a stream's float4 read as it is used
    const float4 u0 = m4[lower ? 1 + j4 : q4];
    float4 z = make_float4(u0.x * ri[0], u0.y * ri[0], u0.z * ri[0], u0.w * ri[0]);
#pragma unroll
    for (int32_t si = 1; si < n_stream; ++si) {
        const float4 us = m4[lower ? 1 + (n_down / 4) * si + j4 : q4 + si];
        z.x = fmaf(us.x, ri[si], z.x);
        z.y = fmaf(us.y, ri[si], z.y);
        z.z = fmaf(us.z, ri[si], z.z);
        z.w = fmaf(us.w, ri[si], z.w);
    }
    z = make_float4(0.25f * z.x, 0.25f * z.y, 0.25f * z.z, 0.25f * z.w);
    if (lower) {
        su.lows[i][tt / 16][tt % 16] =
            make_float4(z.x * sigmoid(z.x), z.y * sigmoid(z.y), z.z * sigmoid(z.z), z.w * sigmoid(z.w));
    } else {
        const int32_t row = seg.row(i);
        const int32_t k   = row / LR;
        reinterpret_cast<float4 *>(jp.p[k].u.inject)[row - k * LR] =
            make_float4(2.0f * sigmoid(z.x), 2.0f * sigmoid(z.y), 2.0f * sigmoid(z.z), 2.0f * sigmoid(z.w));
    }
}

// up_phase's finish of slot i (warp i): lane k's row b, run `run`
template <int U, int LR>
__device__ __forceinline__ void up_finish(const sub_params &     q,
                                          int32_t                b,
                                          int32_t                run,
                                          int32_t                i,
                                          int32_t                gb,
                                          const wide_up<U, LR> & su) {
    using hl = fn::xchg::hmix_layout;
    const up2_args & a = q.u;
    const int32_t lane = fresh_tid() % 32;
    const int32_t c0 = n_run_feat * run;
    const int32_t f = lane;
    float mixed = 0.0f;
#pragma unroll
    for (int32_t si = 0; si < n_stream; ++si) {
        float zs = su.red[i][0][si][f];
#pragma unroll
        for (int32_t oo = 1; oo < n_oct; ++oo) {
            zs += su.red[i][oo][si][f];
        }
        mixed = fmaf(su.gx[gb][i][si][f] * su.rinv[gb][i][si], sigmoid(zs), mixed);
    }
    const uint32_t bits = fn::xchg::canonical(0.25f * mixed);
    const int32_t row0 = b * hl::width + a.card * n_own;
    fn::xchg::put_run(reinterpret_cast<uint32_t *>(a.hmix_own) + row0, reinterpret_cast<uint32_t *>(a.hmix_peer) + row0,
                      c0, n_run_feat, 0, bits);
    float4 v;
    v.x = __uint_as_float(__shfl_sync(0xffffffffu, bits, 4 * (lane & 7)));
    v.y = __uint_as_float(__shfl_sync(0xffffffffu, bits, 4 * (lane & 7) + 1));
    v.z = __uint_as_float(__shfl_sync(0xffffffffu, bits, 4 * (lane & 7) + 2));
    v.w = __uint_as_float(__shfl_sync(0xffffffffu, bits, 4 * (lane & 7) + 3));
    const bool real = lane < 8;
    uint32_t packed = 0u;
    float2 ds = make_float2(0.0f, 0.0f);
    fn::act::quantize_run8(real ? v : make_float4(0.0f, 0.0f, 0.0f, 0.0f), packed, ds);
    fn::xchg::hmix_put_run(a.hmix_own, a.hmix_peer, b, a.card * n_runs_card + run, packed, ds, real);
}

// items 5 and 5b: up on all 72 blocks, by lane group: each group's moments (polls), lows, dots and finish (X-HMIX
// sent) before the next group's, the block's packed weights held across the groups (a run's, loaded again only when
// a slot's run differs). Warp (o, rg) holds features 8 rg .. 8 rg + 7 of the run, low octet o, stream s = lane / 8.
template <int U, int LR, bool H> __device__ __forceinline__ void up_wide(const sub_joint_params & jp) {
    constexpr int32_t row_u4 = n_stream * (n_down / 8);
    extern __shared__ float4 sub_dyn[];
    wide_up<U, LR> &         su    = *reinterpret_cast<wide_up<U, LR> *>(sub_dyn);
    // thread (o, rg, s, k8)'s rows of run r: wbase(r) + q row_u4
    const auto wbase = [&jp](int32_t r) {
        const int32_t warp = fresh_tid() / 32;
        const int32_t lane = fresh_tid() % 32;
        return reinterpret_cast<const uint4 *>(jp.p[0].u.w_up) +
               ((r * n_run_feat + (warp / n_oct) * up_rows) * n_stream + lane / 8) * (n_down / 8) + (warp % n_oct) * 8 +
               lane % 8;
    };
    {
        // group A's runs to L2 while the moments are polled
        const up_seg  sa = up_seg_of<U, LR>(0, fresh_block());
        const uint4 * w0 = wbase(sa.run(0));
        const uint4 * w1 = wbase(sa.run(sa.n - 1));
#pragma unroll
        for (int32_t q = 0; q < up_rows; ++q) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(w0 + q * row_u4));
            asm volatile("prefetch.global.L2 [%0];" ::"l"(w1 + q * row_u4));
        }
    }
    uint4 wq[up_rows] = {}; // the held run's packed weights
    int32_t held = -1;
#pragma unroll 1
    for (int32_t grp = 0; grp < 2; ++grp) {
        const up_seg seg = up_seg_of<U, LR>(grp, fresh_block());
        up_moments<U, LR, H>(jp, seg, su, grp);
        __syncthreads();
        up_lows<U, LR, H>(jp, seg, su, grp);
        __syncthreads();
        // the dots, a piece of slots a run: the held run's piece first
        const int32_t r0 = seg.run(0);
        const int32_t r1 = seg.run(seg.n - 1);
        const int32_t split = r0 == r1 ? seg.n : r1 * seg.rows - seg.lo; // the first slot of run r1
#pragma unroll 1
        for (int32_t pc = 0; pc < (r0 == r1 ? 1 : 2); ++pc) {
            const bool second = (pc == 1) != (held == r1 && r0 != r1);
            const int32_t run = second ? r1 : r0;
            if (run != held) {
                const uint4 * wr = wbase(run);
#pragma unroll
                for (int32_t q = 0; q < up_rows; ++q) {
                    wq[q] = __ldcg(wr + q * row_u4);
                }
                held = run;
            }
            const int32_t warp = fresh_tid() / 32;
            const int32_t lane = fresh_tid() % 32;
            const int32_t o = warp % n_oct;
            const int32_t k8 = lane % 8;
            const int32_t i1 = second ? seg.n : split;
#pragma unroll 1
            for (int32_t i = second ? split : 0; i < i1; ++i) {
                const float4 a0 = su.lows[i][o][k8];
                const float4 a1 = su.lows[i][o][k8 + 8];
                // the weights widened at each use (dot8: dot8f's fmaf in its order, bitwise), so only the packed run is
                // held across the groups; z, an opaque zero each slot, keeps the permutes in the loop
                uint32_t z;
                asm volatile("mov.u32 %0, 0;" : "=r"(z));
                float v[up_rows];
#pragma unroll
                for (int32_t q = 0; q < up_rows; ++q) {
                    v[q] = dot8(wq[q], a0, a1, 0.0f, z);
                }
                int32_t qq = 0;
                const float sum = rs8(v, lane, qq);
                su.red[i][o][lane / 8][(warp / n_oct) * up_rows + qq] = sum;
            }
        }
        __syncthreads();
        const int32_t fw = fresh_tid() / 32;
        if (fw < seg.n) { // warp i: slot i's row
            const int32_t row = seg.row(fw);
            const int32_t k   = row / LR;
            up_finish<U, LR>(jp.p[k], row - k * LR, seg.run(fw), fw, grp, su);
        }
    }
}

template <int U, int LR, bool H>
__global__ void __launch_bounds__(n_thread_up, 1) hc_sub_joint_kernel(const __grid_constant__ sub_joint_params p,
                                                                      int32_t                                  mode) {
    if ((mode & 1) != 0) {
        down_wide<U, LR, H>(p);
    }
    if ((mode & 2) != 0) {
        if ((mode & 1) != 0) {
            __syncthreads(); // the down phase's finish reads of the region before the up phase's stores
        }
        up_wide<U, LR, H>(p);
    }
}

// ME12: both logical shards of one sublayer on one device, one cooperative grid (hc.h pair). p[k] is shard (card) k's
// sub_params; common is the [B][320] low and [B][4] inverse RMS the up tasks read.
struct pair_params {
    sub_params p[2];
    float * common;
};

// the dynamic region of the paired launch at every B: down's act, then the up phase's arrays (and the common stage's
// inverse RMS) over it
template <int B> constexpr int32_t pair_dyn_bytes() {
    constexpr int32_t act = static_cast<int32_t>(sizeof(float4)) * B * n_col * 64;
    constexpr int32_t up = static_cast<int32_t>(sizeof(up_dyn<B>));
    return act > up ? act : up;
}
static_assert(pair_dyn_bytes<5>() == 25600 && pair_dyn_bytes<1>() == 5120, "ME12: the union arena [counted]");

// Block 0 of the paired launch, after the first grid barrier (every moment of both shards landed): forms what the 80
// up blocks of the four-launch order each formed again, once. Threads b < B: row b's four inverse RMS; then thread
// 81 lb + tt < 81 B: low float4 tt of row lb from u float4 j4 (the up phase's permutation), or (tt = 80, normal mode)
// row lb's injection, written to both shards. Every value takes up_phase's operations in its order on the same
// canonical moment words, card 0's then card 1's (both read from shard 0's slot: card 0's own region, card 1's in the
// peer region), so the bits are the four-launch order's. The words are polled as up_phase polls them; a poll that
// gives up sets err_poll_hc in both shards' error words and the block still reaches the next grid barrier.
template <int B, bool H> __device__ __forceinline__ void pair_common(const pair_params & pp) {
    constexpr int32_t mrow4 = (H ? n_mom_head : n_mom) / 4;
    constexpr int32_t q4 = (4 + n_stream * n_down) / 4;
    extern __shared__ float4 sub_dyn[];
    float(*rinv_s)[n_stream] = reinterpret_cast<float(*)[n_stream]>(sub_dyn); // [B][4], over the down arena
    const uint4 * c0 = reinterpret_cast<const uint4 *>(pp.p[0].own);       // card 0's moments
    const uint4 * c1 = reinterpret_cast<const uint4 *>(pp.p[0].u.hc_ring); // card 1's moments
    uint32_t * const err0 = pp.p[0].u.err;
    uint32_t * const err1 = pp.p[1].u.err;
    const int32_t t = threadIdx.x;
    if (t < B) {
        const uint4 * pw[2] = {c0 + t * mrow4, c1 + t * mrow4};
        uint4 w[2];
        fn::xchg::poll<2>(pw, w, err0, fn::xchg::err_poll_hc);
        if (!fn::xchg::landed(w[0]) || !fn::xchg::landed(w[1])) {
            atomicOr(err1, fn::xchg::err_poll_hc);
        }
        const float4 v0 = as_f4(w[0]), v1 = as_f4(w[1]);
        rinv_s[t][0] = 1.0f / sqrtf((v0.x + v1.x) / float(fn::n_embd) + fn::rms_eps);
        rinv_s[t][1] = 1.0f / sqrtf((v0.y + v1.y) / float(fn::n_embd) + fn::rms_eps);
        rinv_s[t][2] = 1.0f / sqrtf((v0.z + v1.z) / float(fn::n_embd) + fn::rms_eps);
        rinv_s[t][3] = 1.0f / sqrtf((v0.w + v1.w) / float(fn::n_embd) + fn::rms_eps);
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            pp.common[B * n_down + t * n_stream + si] = rinv_s[t][si];
        }
    }
    __syncthreads();
    const int32_t lb = t / 81;
    const int32_t tt = t - 81 * lb;
    const bool lower = lb < B && tt < 80;
    const bool injector = !H && lb < B && tt == 80;
    if (lower || injector) {
        const int32_t j4 = 16 * (tt / 16) + 2 * (tt & 7) + ((tt >> 3) & 1);
        int32_t at4[n_stream];
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            at4[si] = lb * mrow4 + (lower ? 1 + (n_down / 4) * si + j4 : q4 + si);
        }
        const uint4 * pw[2 * n_stream] = {c0 + at4[0], c0 + at4[1], c0 + at4[2], c0 + at4[3],
                                          c1 + at4[0], c1 + at4[1], c1 + at4[2], c1 + at4[3]};
        uint4 w[2 * n_stream];
        fn::xchg::poll<2 * n_stream>(pw, w, err0, fn::xchg::err_poll_hc);
        bool all = true;
#pragma unroll
        for (int32_t i = 0; i < 2 * n_stream; ++i) {
            all = all && fn::xchg::landed(w[i]);
        }
        if (!all) {
            atomicOr(err1, fn::xchg::err_poll_hc);
        }
        float4 u0[n_stream], u1[n_stream];
        float ri[n_stream];
#pragma unroll
        for (int32_t si = 0; si < n_stream; ++si) {
            u0[si] = as_f4(w[si]);
            u1[si] = as_f4(w[n_stream + si]);
            ri[si] = rinv_s[lb][si];
        }
        float4 z = make_float4((u0[0].x + u1[0].x) * ri[0], (u0[0].y + u1[0].y) * ri[0], (u0[0].z + u1[0].z) * ri[0],
                               (u0[0].w + u1[0].w) * ri[0]);
#pragma unroll
        for (int32_t si = 1; si < n_stream; ++si) {
            z.x = fmaf(u0[si].x + u1[si].x, ri[si], z.x);
            z.y = fmaf(u0[si].y + u1[si].y, ri[si], z.y);
            z.z = fmaf(u0[si].z + u1[si].z, ri[si], z.z);
            z.w = fmaf(u0[si].w + u1[si].w, ri[si], z.w);
        }
        z = make_float4(0.25f * z.x, 0.25f * z.y, 0.25f * z.z, 0.25f * z.w);
        if (lower) {
            reinterpret_cast<float4 *>(pp.common)[lb * 80 + tt] =
                make_float4(z.x * sigmoid(z.x), z.y * sigmoid(z.y), z.z * sigmoid(z.z), z.w * sigmoid(z.w));
        } else {
            const float4 inj =
                make_float4(2.0f * sigmoid(z.x), 2.0f * sigmoid(z.y), 2.0f * sigmoid(z.z), 2.0f * sigmoid(z.w));
            reinterpret_cast<float4 *>(pp.p[0].u.inject)[lb] = inj;
            reinterpret_cast<float4 *>(pp.p[1].u.inject)[lb] = inj;
        }
    }
}

// ME12 (hc.h pair): one cooperative grid of 72 blocks of 640 threads, one an SM. Block b runs down tasks b (shard 0's
// block b) and b + 72 (shard 1's block b): every down block, moment send and reset index of both shards' sub launches
// runs once. A grid barrier publishes both shards' moments; block 0 forms the common intermediate (pair_common); a
// second grid barrier publishes it; then block b runs up tasks b and b + 72 (< 80): task v is shard v / 40's up block
// v % 40, the four-launch order's up phase with low and the inverse RMS read instead of formed. The tasks are written
// out in sequence, not looped (a task loop carries its index and the shard's pointers across the grid barriers and
// spills at B = 5): one task's weights and shared arrays at a time, and a block barrier between a block's two tasks
// covers the first's last shared reads. No block leaves before either grid barrier (every poll is bounded).
template <int B, bool H>
__global__ void __launch_bounds__(n_thread_up, 1) hc_pair_kernel(const __grid_constant__ pair_params pp) {
    const int32_t b = blockIdx.x;
    // each task with its shard's arguments as constant-bank operands (a runtime-indexed shard would hold its pointers
    // in registers across the task and spill)
    down_phase<B, shipped_depth_down, true, H, false, true, n_thread_up, true>(pp.p[0].d, pp.p[0].own, b);
    __syncthreads();
    down_phase<B, shipped_depth_down, true, H, false, true, n_thread_up, true>(pp.p[1].d, pp.p[1].own, b);
    cooperative_groups::this_grid().sync();
    if (b == 0) {
        pair_common<B, H>(pp);
    }
    cooperative_groups::this_grid().sync();
    const uint32_t mz = gridDim.x == n_block ? 0u : ~0u;
    if (b < n_up_block) {
        up_phase<B, H, true, true>(pp.p[0].u, pp.p[0].hres, pp.p[0].hx, pp.p[0].hpost, pp.p[0].own, b, mz, pp.common);
    } else {
        up_phase<B, H, true, true>(pp.p[1].u, pp.p[1].hres, pp.p[1].hx, pp.p[1].hpost, pp.p[1].own, b - n_up_block, mz,
                                   pp.common);
    }
    if (b < 2 * n_up_block - n_block) {
        __syncthreads();
        up_phase<B, H, true, true>(pp.p[1].u, pp.p[1].hres, pp.p[1].hx, pp.p[1].hpost, pp.p[1].own,
                                   b + n_block - n_up_block, mz, pp.common);
    }
}

// The X-HMIX stand-in consumer (hc.h ready_args): block b, thread t polls 16-byte group t of the peer half of row b's
// F32 values (t < 80 also its codes, t < 20 its scales), reads the same groups of the own half, and writes the full
// row: card 0's half, then card 1's.
__global__ void __launch_bounds__(n_own / 4) hc_ready_kernel(const ready_args a) {
    using hl = fn::xchg::hmix_layout;
    const int32_t b = blockIdx.x, t = threadIdx.x, ph = 1 - a.card;
    uint4 * s4 = reinterpret_cast<uint4 *>(a.slot);
    const auto take = [&](int32_t own_at, int32_t peer_at, uint32_t sent, uint4 & own, uint4 & peer) {
        own = s4[own_at];
        const uint4 * pp[1] = {s4 + peer_at};
        uint4 pv[1];
        fn::xchg::poll<1>(pp, pv, a.err, fn::xchg::err_poll_hmix, sent);
        peer = pv[0];
        if (a.reset != 0) {
            fn::xchg::reset4(s4 + peer_at, sent);
        }
    };
    uint4 own, peer;
    take((b * hl::width + a.card * n_own) / 4 + t, (b * hl::width + ph * n_own) / 4 + t, fn::xchg::sentinel, own, peer);
    float4 * full = reinterpret_cast<float4 *>(a.full) + b * (2 * n_own / 4);
    full[a.card * (n_own / 4) + t] = as_f4(own);
    full[ph * (n_own / 4) + t] = as_f4(peer);
    if (t < n_own / 16) { // codes: 80 groups of 16 a half row
        const int32_t base = hl::code_at / 4 + b * hl::width / 16;
        take(base + a.card * (n_own / 16) + t, base + ph * (n_own / 16) + t, fn::xchg::code_sentinel, own, peer);
        if (a.codes != nullptr) {
            uint4 * c4 = reinterpret_cast<uint4 *>(a.codes) + b * (2 * n_own / 16);
            c4[a.card * (n_own / 16) + t] = own;
            c4[ph * (n_own / 16) + t] = peer;
        }
    }
    if (t < n_runs_card / 2) { // scales: 20 groups of two {d, s} pairs a half row
        const int32_t base = hl::scale_at / 4 + b * hl::runs / 2;
        take(base + a.card * (n_runs_card / 2) + t, base + ph * (n_runs_card / 2) + t, fn::xchg::sentinel, own, peer);
        if (a.codes != nullptr) {
            uint4 * s = reinterpret_cast<uint4 *>(a.scales) + b * (hl::runs / 2);
            s[a.card * (n_runs_card / 2) + t] = own;
            s[ph * (n_runs_card / 2) + t] = peer;
        }
    }
}

// The stream-only probe (bench only): hc_down's grid and weight pattern with D loads a lane in flight and no staging
// or arithmetic. Each row's load feeds an XOR that lane 0 writes, so every load stays live.
template <int D>
__global__ void __launch_bounds__(n_thread, 1) hc_probe_kernel(const uint16_t * w_down, uint32_t * out) {
    static_assert(D >= 1 && D <= rows_per_warp, "stream depth");
    const int32_t s = blockIdx.x % n_stream;
    const int32_t g = blockIdx.x / n_stream;
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    const int32_t c = warp % n_col;
    const int32_t rg = warp / n_col;
    const int32_t row0 = s * n_rows_down + g * down_rows + rg * rows_per_warp;
    const uint4 * wrow = reinterpret_cast<const uint4 *>(w_down) + (row0 * n_own + c * 256) / 8 + lane;
    const uint32_t mz = gridDim.x == n_block ? 0u : ~0u;
    uint4 w[D];
#pragma unroll
    for (int32_t q = 0; q < D; ++q) {
        w[q] = __ldcs(wrow + q * (n_own / 8));
    }
    // every word is first read by a PRMT with the opaque zero of the row before (the kernels' unpack control), so a row
    // is consumed in order and its refill waits for it
    uint32_t x = 0u, z = mz;
#pragma unroll
    for (int32_t q = 0; q < rows_per_warp; ++q) {
        const uint4 v = w[q % D];
        const uint32_t xr = __byte_perm(v.x, z, 0x1044) ^ __byte_perm(v.x, z, 0x3244) ^ __byte_perm(v.y, z, 0x1044) ^
                            __byte_perm(v.y, z, 0x3244) ^ __byte_perm(v.z, z, 0x1044) ^ __byte_perm(v.z, z, 0x3244) ^
                            __byte_perm(v.w, z, 0x1044) ^ __byte_perm(v.w, z, 0x3244);
        z = xr & mz;
        if (q + D < rows_per_warp) {
            w[q % D] = __ldcs(wrow + (q + D) * (n_own / 8) + z);
        }
        x ^= xr;
    }
    if (lane == 0) {
        out[blockIdx.x * n_warp + warp] = x;
    }
}

namespace {

bool aligned16(const void * p) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

// w: the weights the kernel reads (a.w_down, or DR1's Q8_0 weights)
bool down_ok(const down_args & a, const void * stream, const void * w) {
    return stream != nullptr && aligned16(w) && aligned16(a.hres) && aligned16(a.hx) && aligned16(a.hpost) &&
           aligned16(a.gamma) && aligned16(a.x) && aligned16(a.mom);
}

bool down_ok(const down_args & a, const void * stream) {
    return down_ok(a, stream, a.w_down);
}

bool up_ok(const up_args & a, const void * stream) {
    return stream != nullptr && aligned16(a.w_up) && aligned16(a.mom_own) && aligned16(a.mom_peer) && aligned16(a.x) &&
           aligned16(a.gamma) && aligned16(a.inject) && aligned16(a.mixed);
}

template <int B, int D> bool launch_down(const down_args & a, cudaStream_t stream) {
    hc_down_kernel<B, D><<<n_block, n_thread, 0, stream>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <int B, int D> bool launch_up(const up_args & a, cudaStream_t stream) {
    hc_up_kernel<B, D><<<n_block, n_thread, 0, stream>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

// dispatch on the row count and the depth (1, 2, 3 or 6); anything else is rejected
template <template <int, int> class L, typename... Args> bool dispatch(int32_t rows, int32_t depth, Args &&... args) {
    switch (rows * 10 + depth) {
        case 11: return L<1, 1>::run(args...);
        case 12: return L<1, 2>::run(args...);
        case 13: return L<1, 3>::run(args...);
        case 16: return L<1, 6>::run(args...);
        case 21: return L<2, 1>::run(args...);
        case 22: return L<2, 2>::run(args...);
        case 23: return L<2, 3>::run(args...);
        case 26: return L<2, 6>::run(args...);
        case 31: return L<3, 1>::run(args...);
        case 32: return L<3, 2>::run(args...);
        case 33: return L<3, 3>::run(args...);
        case 36: return L<3, 6>::run(args...);
        case 41: return L<4, 1>::run(args...);
        case 42: return L<4, 2>::run(args...);
        case 43: return L<4, 3>::run(args...);
        case 46: return L<4, 6>::run(args...);
        default: return false;
    }
}

template <int B, int D> struct run_down_a {
    static bool run(const down_args & a, cudaStream_t st) { return launch_down<B, D>(a, st); }
};

template <int B, int D> struct run_up_a {
    static bool run(const up_args & a, cudaStream_t st) { return launch_up<B, D>(a, st); }
};

constexpr int32_t up2_smem = up_feat_max * n_stream * n_down * 2; // arm A's staged weights: 46,080 bytes

template <int B, bool H> bool launch_down2(const down2_args & a, cudaStream_t stream) {
    hc_down_kernel<B, shipped_depth_down, true, H><<<n_block, n_thread, 0, stream>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <int B> bool launch_head_up2(const up2_args & a, cudaStream_t stream) {
    hc_up3_kernel<B, true><<<n_up_block, n_thread_up, 0, stream>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <int B, bool H> bool launch_down2_q8(const down2_q8_args & a, cudaStream_t stream) {
    hc_down_kernel<B, shipped_depth_down, true, H, true><<<n_block, n_thread, 0, stream>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <int B, bool H> bool launch_up2_q8(const up2_q8_args & a, cudaStream_t stream) {
    hc_up2_kernel<B, 0, H, true><<<n_block, n_thread, 0, stream>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <int B> bool launch_up2(up_arm arm, const up2_args & a, cudaStream_t stream) {
    if (arm == up_arm::ring) {
        hc_up3_kernel<B, false><<<n_up_block, n_thread_up, 0, stream>>>(a);
    } else {
        hc_up2_kernel<B, 1><<<n_block, n_thread, up2_smem, stream>>>(a);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

// the cooperative launch: every block co-resident (72 blocks, one an SM by __launch_bounds__(640, 1)); the runtime
// rejects a grid that cannot be (cudaErrorCooperativeLaunchTooLarge)
template <int B, bool H> bool launch_sub(const sub_params & p, int32_t mode, cudaStream_t stream) {
    sub_params q = p;
    int32_t m = mode;
    void * args[] = {&q, &m};
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(hc_sub_kernel<B, H>), dim3(n_block),
                                       dim3(n_thread_up), args, sub_dyn_bytes<B>(), stream) == cudaSuccess;
}

bool sub_ok(int32_t rows, int32_t mode, const sub_args & a, const void * stream, bool head) {
    return stream != nullptr && rows >= 1 && rows <= max_rows && mode >= 1 && mode <= 3 && aligned16(a.w_down) &&
           aligned16(a.w_up) && aligned16(a.hres) && aligned16(a.hx) && aligned16(a.hpost) && aligned16(a.gamma) &&
           aligned16(a.x) && (head || aligned16(a.inject)) && aligned16(a.hc_ring) && aligned16(a.hc_peer) &&
           (a.hc_reset == nullptr || aligned16(a.hc_reset)) && (a.hmix_reset == nullptr || aligned16(a.hmix_reset)) &&
           aligned16(a.hmix_own) && aligned16(a.hmix_peer) && a.err != nullptr && (a.card == 0 || a.card == 1);
}

sub_params sub_of(const sub_args & a) {
    sub_params p = {};
    p.d.d = {a.w_down, a.hres, a.hx, a.hpost, a.gamma, a.x, nullptr};
    p.d.hc_peer = a.hc_peer;
    p.d.hc_reset = a.hc_reset;
    p.d.hmix_reset = a.hmix_reset;
    p.u = {a.w_up, nullptr, a.hc_ring, nullptr, a.gamma, a.inject, a.hmix_own, a.hmix_peer, nullptr, a.err, a.card};
    p.hres = a.hres;
    p.hx = a.hx;
    p.hpost = a.hpost;
    p.own = a.hc_ring + fn::xchg::hc_own_at;
    return p;
}

// ME12: the paired launch, through the same cooperative launch (and so the same captured node kind) as launch_sub
template <int B, bool H> bool launch_pair(const pair_params & p, cudaStream_t stream) {
    pair_params q = p;
    void * args[] = {&q};
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(hc_pair_kernel<B, H>), dim3(n_block),
                                       dim3(n_thread_up), args, pair_dyn_bytes<B>(), stream) == cudaSuccess;
}

// shard s's resets touch neither shard's current slot (the X-HC and X-HMIX slots the grid writes; null: no reset)
bool resets_apart(const sub_args & s, const sub_args & s0, const sub_args & s1) {
    return s.hc_reset != s0.hc_ring && s.hc_reset != s1.hc_ring && s.hmix_reset != s0.hmix_own &&
           s.hmix_reset != s1.hmix_own;
}

// both shards' arguments valid as hc::sub's (mode 3), cards 0 and 1, and one pair: each shard's peer slots are the
// other's own slots, no reset of either shard is a slot the grid writes (the resets and the sends are one grid's),
// and the scratch is aligned
bool pair_ok(int32_t rows, const pair_args & a, const void * stream, bool head) {
    const sub_args & s0 = a.s[0];
    const sub_args & s1 = a.s[1];
    return sub_ok(rows, 3, s0, stream, head) && sub_ok(rows, 3, s1, stream, head) && s0.card == 0 && s1.card == 1 &&
           s0.hc_peer == s1.hc_ring && s1.hc_peer == s0.hc_ring && s0.hmix_peer == s1.hmix_own &&
           s1.hmix_peer == s0.hmix_own && s0.hc_ring != s1.hc_ring && s0.hmix_own != s1.hmix_own &&
           resets_apart(s0, s0, s1) && resets_apart(s1, s0, s1) && aligned16(a.scratch);
}

pair_params pair_of(const pair_args & a, bool head) {
    pair_params p = {};
    for (int32_t k = 0; k < 2; ++k) {
        p.p[k] = sub_of(a.s[k]);
        if (head) {
            p.p[k].u.inject = nullptr;
        }
    }
    p.common = a.scratch;
    return p;
}

template <bool H> bool launch_pair_rows(int32_t rows, const pair_params & p, cudaStream_t st) {
    switch (rows) {
        case 1:
            return launch_pair<1, H>(p, st);
        case 2:
            return launch_pair<2, H>(p, st);
        case 3:
            return launch_pair<3, H>(p, st);
        case 4:
            return launch_pair<4, H>(p, st);
        case 5:
            return launch_pair<5, H>(p, st);
        default:
            return false;
    }
}

// one instance co-resident at n_block blocks of n_thread_up on this device (the occupancy API with its dynamic
// shared memory), and its resources within the launch's (no local memory)
template <int B, bool H> bool pair_fits(int32_t sms) {
    int32_t per_sm = 0;
    cudaFuncAttributes fa = {};
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, hc_pair_kernel<B, H>, n_thread_up,
                                                         pair_dyn_bytes<B>()) == cudaSuccess &&
           cudaFuncGetAttributes(&fa, hc_pair_kernel<B, H>) == cudaSuccess && per_sm >= 1 &&
           int64_t(per_sm) * sms >= n_block && fa.localSizeBytes == 0 && fa.maxThreadsPerBlock >= n_thread_up;
}

bool down2_ok(const down2_args & a, const void * stream, const void * w) {
    return down_ok(a.d, stream, w) && aligned16(a.hc_peer) && (a.hc_reset == nullptr || aligned16(a.hc_reset)) &&
           (a.hmix_reset == nullptr || aligned16(a.hmix_reset));
}

bool down2_ok(const down2_args & a, const void * stream) {
    return down2_ok(a, stream, a.d.w_down);
}

bool up2_ok(const up2_args & a, const void * stream, const void * w) {
    return stream != nullptr && aligned16(w) && aligned16(a.mom_own) && aligned16(a.hc_ring) && aligned16(a.x) &&
           aligned16(a.gamma) && aligned16(a.inject) && aligned16(a.hmix_own) && aligned16(a.hmix_peer) &&
           a.runs != nullptr && a.err != nullptr && (a.card == 0 || a.card == 1);
}

bool up2_ok(const up2_args & a, const void * stream) {
    return up2_ok(a, stream, a.w_up);
}

bool head_up2_ok(const up2_args & a, const void * stream, const void * w) {
    return stream != nullptr && aligned16(w) && aligned16(a.mom_own) && aligned16(a.hc_ring) && aligned16(a.x) &&
           aligned16(a.gamma) && aligned16(a.hmix_own) && aligned16(a.hmix_peer) && a.runs != nullptr &&
           a.err != nullptr && (a.card == 0 || a.card == 1);
}

bool head_up2_ok(const up2_args & a, const void * stream) {
    return head_up2_ok(a, stream, a.w_up);
}

template <int D> bool launch_probe(const uint16_t * w_down, uint32_t * out, cudaStream_t stream) {
    hc_probe_kernel<D><<<n_block, n_thread, 0, stream>>>(w_down, out);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

variant_id shipped() {
    return {shipped_depth_down, shipped_depth_up};
}

size_t workspace_bytes() {
    const size_t b = max_rows;
    const size_t floats[] = {b * 4 * n_own, b * 4 * n_own, b * n_own, b * 4, b * n_mom, b * n_inject, b * n_own,
                             b * 2 * n_own};
    size_t total = 0;
    for (const size_t n : floats) {
        total += (n * sizeof(float) + 255) / 256 * 256;
    }
    return total;
}

bool down(int32_t rows, const down_args & a, void * stream) {
    if (!down_ok(a, stream)) {
        return false;
    }
    return dispatch<run_down_a>(rows, shipped_depth_down, a, static_cast<cudaStream_t>(stream));
}

bool up(int32_t rows, const up_args & a, void * stream) {
    if (!up_ok(a, stream)) {
        return false;
    }
    return dispatch<run_up_a>(rows, shipped_depth_up, a, static_cast<cudaStream_t>(stream));
}

bool down_variant(int32_t depth, int32_t rows, const down_args & a, void * stream) {
    if (!down_ok(a, stream)) {
        return false;
    }
    return dispatch<run_down_a>(rows, depth, a, static_cast<cudaStream_t>(stream));
}

bool up_variant(int32_t depth, int32_t rows, const up_args & a, void * stream) {
    if (!up_ok(a, stream)) {
        return false;
    }
    return dispatch<run_up_a>(rows, depth, a, static_cast<cudaStream_t>(stream));
}

bool prepare2() {
    bool ok = true;
    ok = ok && cudaFuncSetAttribute(hc_up2_kernel<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, up2_smem) ==
                   cudaSuccess;
    ok = ok && cudaFuncSetAttribute(hc_up2_kernel<2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, up2_smem) ==
                   cudaSuccess;
    ok = ok && cudaFuncSetAttribute(hc_up2_kernel<3, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, up2_smem) ==
                   cudaSuccess;
    ok = ok && cudaFuncSetAttribute(hc_up2_kernel<4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, up2_smem) ==
                   cudaSuccess;
    ok = ok && cudaFuncSetAttribute(hc_up2_kernel<5, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, up2_smem) ==
                   cudaSuccess;
    return ok;
}

bool down2(int32_t rows, const down2_args & a, void * stream) {
    if (!down2_ok(a, stream)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return launch_down2<1, false>(a, st);
        case 2: return launch_down2<2, false>(a, st);
        case 3: return launch_down2<3, false>(a, st);
        case 4: return launch_down2<4, false>(a, st);
        case 5: return launch_down2<5, false>(a, st); // SP2
        default: return false;
    }
}

bool up2(up_arm arm, int32_t rows, const up2_args & a, void * stream) {
    if (!up2_ok(a, stream) || (arm != up_arm::ring && arm != up_arm::staged)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return launch_up2<1>(arm, a, st);
        case 2: return launch_up2<2>(arm, a, st);
        case 3: return launch_up2<3>(arm, a, st);
        case 4: return launch_up2<4>(arm, a, st);
        case 5: return launch_up2<5>(arm, a, st); // SP2
        default: return false;
    }
}

bool head_down2(int32_t rows, const down2_args & a, void * stream) {
    if (!down2_ok(a, stream)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return launch_down2<1, true>(a, st);
        case 2: return launch_down2<2, true>(a, st);
        case 3: return launch_down2<3, true>(a, st);
        case 4: return launch_down2<4, true>(a, st);
        case 5: return launch_down2<5, true>(a, st); // SP2
        default: return false;
    }
}

bool head_up2(int32_t rows, const up2_args & a, void * stream) {
    if (!head_up2_ok(a, stream)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return launch_head_up2<1>(a, st);
        case 2: return launch_head_up2<2>(a, st);
        case 3: return launch_head_up2<3>(a, st);
        case 4: return launch_head_up2<4>(a, st);
        case 5: return launch_head_up2<5>(a, st); // SP2
        default: return false;
    }
}

bool down2_q8(int32_t rows, const down2_q8_args & a, void * stream) {
    if (!down2_ok(a.a, stream, a.w)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return launch_down2_q8<1, false>(a, st);
        case 2: return launch_down2_q8<2, false>(a, st);
        case 3: return launch_down2_q8<3, false>(a, st);
        case 4: return launch_down2_q8<4, false>(a, st);
        case 5: return launch_down2_q8<5, false>(a, st); // SP2
        default: return false;
    }
}

bool up2_q8(int32_t rows, const up2_q8_args & a, void * stream) {
    if (!up2_ok(a.a, stream, a.w)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return launch_up2_q8<1, false>(a, st);
        case 2: return launch_up2_q8<2, false>(a, st);
        case 3: return launch_up2_q8<3, false>(a, st);
        case 4: return launch_up2_q8<4, false>(a, st);
        case 5: return launch_up2_q8<5, false>(a, st); // SP2
        default: return false;
    }
}

bool down2_q8_lanes(int32_t lanes, const down2_q8_args * a, void * stream) {
    if (a == nullptr || lanes < 2 || lanes > fn::xchg::max_lanes || stream == nullptr) {
        return false;
    }
    down2_q8_lanes_args p = {};
    for (int32_t u = 0; u < lanes; ++u) {
        if (!down2_ok(a[u].a, stream, a[u].w)) {
            return false;
        }
        p.a[u] = a[u];
    }
    hc_down_lanes_kernel<max_rows>
        <<<dim3(n_block, static_cast<uint32_t>(lanes)), n_thread, 0, static_cast<cudaStream_t>(stream)>>>(p);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool up2_q8_lanes(int32_t lanes, const up2_q8_args * a, void * stream) {
    if (a == nullptr || lanes < 2 || lanes > fn::xchg::max_lanes || stream == nullptr) {
        return false;
    }
    up2_q8_lanes_args p = {};
    for (int32_t u = 0; u < lanes; ++u) {
        if (!up2_ok(a[u].a, stream, a[u].w)) {
            return false;
        }
        for (int32_t q = 0; q < u; ++q) {
            if (a[q].a.runs == a[u].a.runs) {
                return false; // each lane's own run counters: the lanes' blocks run concurrently
            }
        }
        p.a[u] = a[u];
    }
    hc_up2_lanes_kernel<max_rows>
        <<<dim3(n_block, static_cast<uint32_t>(lanes)), n_thread, 0, static_cast<cudaStream_t>(stream)>>>(p);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool head_down2_q8(int32_t rows, const down2_q8_args & a, void * stream) {
    if (!down2_ok(a.a, stream, a.w)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return launch_down2_q8<1, true>(a, st);
        case 2: return launch_down2_q8<2, true>(a, st);
        case 3: return launch_down2_q8<3, true>(a, st);
        case 4: return launch_down2_q8<4, true>(a, st);
        case 5: return launch_down2_q8<5, true>(a, st); // SP2
        default: return false;
    }
}

bool head_up2_q8(int32_t rows, const up2_q8_args & a, void * stream) {
    if (!head_up2_ok(a.a, stream, a.w)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return launch_up2_q8<1, true>(a, st);
        case 2: return launch_up2_q8<2, true>(a, st);
        case 3: return launch_up2_q8<3, true>(a, st);
        case 4: return launch_up2_q8<4, true>(a, st);
        case 5: return launch_up2_q8<5, true>(a, st); // SP2
        default: return false;
    }
}

bool ready(int32_t rows, const ready_args & a, void * stream) {
    if (stream == nullptr || rows < 1 || rows > max_rows || !aligned16(a.slot) || !aligned16(a.full) ||
        (a.codes != nullptr && (!aligned16(a.codes) || !aligned16(a.scales))) || a.err == nullptr ||
        (a.card != 0 && a.card != 1)) {
        return false;
    }
    hc_ready_kernel<<<rows, n_own / 4, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool sub(int32_t rows, int32_t mode, const sub_args & a, void * stream) {
    if (!sub_ok(rows, mode, a, stream, false)) {
        return false;
    }
    const sub_params p = sub_of(a);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1:
            return launch_sub<1, false>(p, mode, st);
        case 2:
            return launch_sub<2, false>(p, mode, st);
        case 3:
            return launch_sub<3, false>(p, mode, st);
        case 4:
            return launch_sub<4, false>(p, mode, st);
        case 5:
            return launch_sub<5, false>(p, mode, st); // SP2
        default:
            return false;
    }
}

bool head_sub(int32_t rows, int32_t mode, const sub_args & a, void * stream) {
    if (!sub_ok(rows, mode, a, stream, true)) {
        return false;
    }
    sub_params p = sub_of(a);
    p.u.inject = nullptr;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1:
            return launch_sub<1, true>(p, mode, st);
        case 2:
            return launch_sub<2, true>(p, mode, st);
        case 3:
            return launch_sub<3, true>(p, mode, st);
        case 4:
            return launch_sub<4, true>(p, mode, st);
        case 5:
            return launch_sub<5, true>(p, mode, st); // SP2
        default:
            return false;
    }
}

// FMU2: every lane valid as sub's (head_sub's) at rows (FMS13: 2 .. max_rows), one card, distinct X-HC and X-HMIX
// slots (each lane its own ring), and no lane's reset a slot another lane writes
bool sub_joint_ok(int32_t lanes, int32_t rows, int32_t mode, const sub_args * a, const void * stream, bool head) {
    if (a == nullptr || lanes < 2 || lanes > fn::xchg::max_lanes || rows < 2 || rows > max_rows) {
        return false;
    }
    for (int32_t k = 0; k < lanes; ++k) {
        if (!sub_ok(rows, mode, a[k], stream, head) || a[k].card != a[0].card || a[k].w_down != a[0].w_down ||
            a[k].w_up != a[0].w_up || a[k].gamma != a[0].gamma) {
            return false;
        }
        for (int32_t q = 0; q < lanes; ++q) {
            if (q != k && (a[q].hc_ring == a[k].hc_ring || a[q].hmix_own == a[k].hmix_own ||
                           a[q].hc_reset == a[k].hc_ring || a[q].hmix_reset == a[k].hmix_own)) {
                return false;
            }
        }
    }
    return true;
}

template <int U, int LR, bool H> bool launch_sub_joint_ur(int32_t mode, const sub_args * a, cudaStream_t stream) {
    sub_joint_params q = {};
    for (int32_t k = 0; k < U; ++k) {
        q.p[k] = sub_of(a[k]);
        if (H) {
            q.p[k].u.inject = nullptr;
        }
    }
    int32_t m = mode;
    void * args[] = {&q, &m};
    const void *                 kern   = reinterpret_cast<const void *>(hc_sub_joint_kernel<U, LR, H>);
    static std::atomic<uint32_t> attr_done{0u};
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        return false;
    }
    const uint32_t bit = 1u << dev;
    if ((attr_done.load() & bit) == 0u) {
        if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, wide_dyn_bytes<U, LR>()) !=
            cudaSuccess) {
            return false;
        }
        attr_done.fetch_or(bit);
    }
    return cudaLaunchCooperativeKernel(kern, dim3(n_block), dim3(n_thread_up), args, wide_dyn_bytes<U, LR>(), stream) ==
           cudaSuccess;
}

template <int U, bool H> bool launch_sub_joint_u(int32_t rows, int32_t mode, const sub_args * a, cudaStream_t stream) {
    switch (rows) {
        case 2:
            return launch_sub_joint_ur<U, 2, H>(mode, a, stream);
        case 3:
            return launch_sub_joint_ur<U, 3, H>(mode, a, stream);
        case 4:
            return launch_sub_joint_ur<U, 4, H>(mode, a, stream);
        case 5:
            return launch_sub_joint_ur<U, max_rows, H>(mode, a, stream);
        default:
            return false;
    }
}

template <bool H>
bool launch_sub_joint(int32_t lanes, int32_t rows, int32_t mode, const sub_args * a, cudaStream_t stream) {
    switch (lanes) {
        case 2:
            return launch_sub_joint_u<2, H>(rows, mode, a, stream);
        case 3:
            return launch_sub_joint_u<3, H>(rows, mode, a, stream);
        case 4:
            return launch_sub_joint_u<4, H>(rows, mode, a, stream);
        default:
            return false;
    }
}

bool sub_joint(int32_t lanes, int32_t rows, int32_t mode, const sub_args * a, void * stream) {
    return sub_joint_ok(lanes, rows, mode, a, stream, false) &&
           launch_sub_joint<false>(lanes, rows, mode, a, static_cast<cudaStream_t>(stream));
}

bool head_sub_joint(int32_t lanes, int32_t rows, int32_t mode, const sub_args * a, void * stream) {
    return sub_joint_ok(lanes, rows, mode, a, stream, true) &&
           launch_sub_joint<true>(lanes, rows, mode, a, static_cast<cudaStream_t>(stream));
}

bool pair_prepare() {
    int32_t dev = -1, coop = 0, sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || coop == 0) {
        return false;
    }
    return pair_fits<1, false>(sms) && pair_fits<2, false>(sms) && pair_fits<3, false>(sms) &&
           pair_fits<4, false>(sms) && pair_fits<5, false>(sms) && pair_fits<1, true>(sms) && pair_fits<2, true>(sms) &&
           pair_fits<3, true>(sms) && pair_fits<4, true>(sms) && pair_fits<5, true>(sms);
}

bool pair(int32_t rows, const pair_args & a, void * stream) {
    if (!pair_ok(rows, a, stream, false)) {
        return false;
    }
    return launch_pair_rows<false>(rows, pair_of(a, false), static_cast<cudaStream_t>(stream));
}

bool head_pair(int32_t rows, const pair_args & a, void * stream) {
    if (!pair_ok(rows, a, stream, true)) {
        return false;
    }
    return launch_pair_rows<true>(rows, pair_of(a, true), static_cast<cudaStream_t>(stream));
}

bool probe(int32_t depth, const uint16_t * w_down, uint32_t * out, void * stream) {
    if (stream == nullptr || !aligned16(w_down) || out == nullptr) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (depth) {
        case 1: return launch_probe<1>(w_down, out, st);
        case 2: return launch_probe<2>(w_down, out, st);
        case 3: return launch_probe<3>(w_down, out, st);
        case 6: return launch_probe<6>(w_down, out, st);
        default: return false;
    }
}

} // namespace fn::hc
