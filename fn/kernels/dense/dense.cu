#include "fn/kernels/dense/dense.h"

#include "fn/kernels/act/act_warp.cuh"
#include "fn/kernels/dense/dense_dec.cuh"
#include "fn/kernels/dense/iq3s.cuh"
#include "fn/kernels/gdn/gdn_conv.cuh"
#include "fn/kernels/round/version.h"
#include "fn/xchg/xchg.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>

namespace fn::dense {
namespace {

constexpr int32_t n_sm = 72;       // the PG500-216 (GUIDE section 1): one block an SM
constexpr int32_t n_threads = 512; // 16 warps an SM
constexpr int32_t n_warps = n_threads / 32;
constexpr int32_t n_grid_warps = n_sm * n_warps;
constexpr int32_t tile_bf = 4;    // rows of a BF16 or F32 tile (32 lanes a row)
constexpr int32_t part_slot = 32; // floats a unit's partial sums take: [8 rows][4 B]
constexpr int32_t part4_slot = 8; // SP2: a unit's row-4 partial sums: [8 rows]
constexpr int32_t iq3s_grid_bytes = 2048;
constexpr float route_floor = 6.103515625e-5f; // the weights' sum is clamped below at F16's smallest normal (ggml)

int32_t n_steps(int32_t k) {
    return (k + step_k - 1) / step_k;
}

// ---- device helpers ---------------------------------------------------------------------------------------------
// a line into L2 ahead of its use (PTX prefetch.global.L2: no register, no wait)
__device__ __forceinline__ void prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// The lane index read afresh (volatile), so per-unit address terms built from it are not hoisted out of the unit loop
// and held live across every format's path (that pushed the kernel past 128 registers)
__device__ __forceinline__ int32_t lane_id() {
    int32_t l;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(l));
    return l;
}
// v, hidden from the optimizer, so a value derived from it is computed where it is used, not hoisted and held
__device__ __forceinline__ int32_t opaque(int32_t v) {
    asm volatile("" : "+r"(v));
    return v;
}

// ---- units -----------------------------------------------------------------------------------------------------
struct mat_desc {
    const uint8_t * w;
    float * out;
    const float * scale;
    int32_t m, f, e, pair, split;
    int32_t tiles;     // row tiles
    int32_t unit_base; // first unit
    int32_t steps;     // steps of a row (the last may be partial)
    int32_t sp;        // steps a K part
    int32_t part_off;  // floats into the partials (SP2: row 4's at part_off / 4 into L.part4)
    int32_t cnt_off;   // counters (glu_up: its gate's)
};

struct launch_desc {
    mat_desc mats[max_mats];
    int32_t n_mats, k, n_units;
    int32_t need_xq, need_grid, need_kv;
    const int8_t * xc;
    const float2 * xs;
    const float * xf;
    const uint8_t * grids;
    int8_t * glu_codes;
    float * glu_scales;
    float * glu_f32;
    float * part;
    uint32_t * cnt;
    // the router: its two matrices, the blocks' arrival counter (after the others), the outputs
    int32_t route_logits, route_gate;
    uint32_t * route_cnt;
    int32_t * ids;
    float * wts;
    float * sg;
    int32_t smem_route; // byte offset of the route epilogue's shared memory
    // the GDN convolution epilogue (epi conv_silu): taps, history (fn::gdn layouts) and the committed slot (SR1)
    const float * conv_w;
    float * conv_hist;
    const uint32_t * conv_cur;
    // HC3: X-HMIX's acquire (dense.h); hmix null: none
    const float * hmix;
    uint32_t * err;
    int32_t hmix_peer, need_xf;
    float * part4; // SP2: row 4's partials (B = 5 launches; work_layout::part4_floats), last so the fields above keep
                   // their places
};

// a finished output value, scaled by row b's factor when the matrix has one (the shared down's gate)
__device__ __forceinline__ float scaled(const mat_desc & md, int32_t b, float v) {
    return md.scale != nullptr ? __fmul_rn(v, md.scale[b]) : v;
}

// The sum of n split parts p[0], p[stride], ... in part order, as a serial loop adds them (bitwise), with every part's
// L2 load issued before the first add (DP2 H3): the last arriver waits one round trip, not one a part. n <= max_split
// (12, the target's most K steps; describe and choose_splits keep a split within it at the draft's K 5,120).
constexpr int32_t max_parts = max_split;
__device__ __forceinline__ float sum_parts(const float * p, int32_t n, int32_t stride) {
    float v[max_parts];
#pragma unroll
    for (int32_t i = 0; i < max_parts; ++i) {
        v[i] = i < n ? __ldcg(p + i * stride) : 0.0f;
    }
    float s = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < max_parts; ++i) {
        s = i < n ? s + v[i] : s;
    }
    return s;
}

// The GLU's last arriver: per row b, the gate and up parts' sums (in part order), SiLU(g) u, its quantize and stores.
// Out of line, one copy a kernel, the rows in a loop (R17: inlined and unrolled in every format path it was about 16
// of a path's 29 KB at B = 3, and a gate/up launch, two formats, ran two copies). Every value comes by value: through
// a descriptor reference each field was a generic load in a dependent chain (DP2 R17a, reverted).
template <int B>
__device__ __noinline__ void glu_finish(const float * gp, const float * up, int32_t split, int32_t m, int32_t group,
                                        int32_t lane, int8_t * codes, float * scales, float * f32) {
    const int32_t t = 4 * group + (lane >> 3), slot = (lane & 7) * 4;
#pragma unroll 1
    for (int32_t b = 0; b < B; ++b) {
        const float g = sum_parts(gp + t * split * part_slot + slot + b, split, part_slot);
        const float u = sum_parts(up + t * split * part_slot + slot + b, split, part_slot);
        const float y = g / (1.0f + expf(-g)) * u;
        fn::act::put_run_warp(y, lane, codes + b * m + 32 * group, scales + 2 * (b * (m / 32) + group));
        if (f32 != nullptr) {
            f32[b * m + 32 * group + lane] = y;
        }
    }
}

// SP2: glu_finish at B = 5, a function of its own so the B <= 4 copies keep their code: rows 0..3 as glu_finish, then
// row 4 from the row-4 regions gp4 and up4 ([8 rows] a unit, the same part order)
__device__ __noinline__ void glu_finish5(const float * gp, const float * up, const float * gp4, const float * up4,
                                         int32_t split, int32_t m, int32_t group, int32_t lane, int8_t * codes,
                                         float * scales, float * f32) {
    const int32_t t = 4 * group + (lane >> 3), slot = (lane & 7) * 4;
#pragma unroll 1
    for (int32_t b = 0; b < 5; ++b) {
        const bool r4 = b == 4;
        const float * gb = r4 ? gp4 + t * split * part4_slot + (lane & 7) : gp + t * split * part_slot + slot + b;
        const float * ub = r4 ? up4 + t * split * part4_slot + (lane & 7) : up + t * split * part_slot + slot + b;
        const int32_t stride = r4 ? part4_slot : part_slot;
        const float g = sum_parts(gb, split, stride);
        const float u = sum_parts(ub, split, stride);
        const float y = g / (1.0f + expf(-g)) * u;
        fn::act::put_run_warp(y, lane, codes + b * m + 32 * group, scales + 2 * (b * (m / 32) + group));
        if (f32 != nullptr) {
            f32[b * m + 32 * group + lane] = y;
        }
    }
}

// The GDN convolution epilogue of one finished column c (LP1; gdn_conv.cuh): xs holds its rows' products (rows
// B.. unused), out [B][5120] the post-SiLU values.
template <int B, int N>
__device__ __forceinline__ void conv_column_finish(const float * conv_w, float * hist, const uint32_t * cur_word,
                                                   float * out, int32_t c, const float (&xs)[N]) {
    const uint32_t cur = *cur_word; // the committed slot (SR1); this lane read it for the unit's prefetch
    const fn::gdn::conv_column cc = fn::gdn::conv_column_load(hist, conv_w, c, cur);
    float x[B], y[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        x[b] = xs[b];
    }
    fn::gdn::conv_silu_epilogue<B>(cc, c, x, hist, cur, y);
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        out[b * fn::gdn::n_channel + c] = y[b];
    }
}

// Out of line, one copy a kernel, as glu_finish: only the conv_silu matrix's lanes call it, and inlined it would sit in
// every format path. x0..x3: rows 0..3 (rows B..3 unused); SP2: conv_finish5 adds row 4, so B <= 4 keep this code.
template <int B>
__device__ __noinline__ void conv_finish(const float * conv_w, float * hist, const uint32_t * cur_word, float * out,
                                         int32_t c, float x0, float x1, float x2, float x3) {
    const float xs[4] = {x0, x1, x2, x3};
    conv_column_finish<B>(conv_w, hist, cur_word, out, c, xs);
}
__device__ __noinline__ void conv_finish5(const float * conv_w, float * hist, const uint32_t * cur_word, float * out,
                                          int32_t c, float x0, float x1, float x2, float x3, float x4) {
    const float xs[5] = {x0, x1, x2, x3, x4};
    conv_column_finish<5>(conv_w, hist, cur_word, out, c, xs);
}

// Ends a unit: (r, q) holds row r's sum for activation row q (valid lanes only). Unsplit plain products write it. Else
// it goes to the unit's partial slot, then the arrival counter of its group: the last unit of a group sums the parts
// in part order and runs the epilogue. Hand-off: partial stores; __threadfence (publication) before the counter's
// atomicAdd; the last arriver's __threadfence (acquire) before L2 reads (ld.cg) of the parts; it resets the counter,
// which frees the slots for the next launch (ordered by the kernel boundary).
// SP2, B = 5: v4 is row r's sum for activation row 4 (every lane of the row's group holds it); lane (r, 0) publishes
// it, into its own partial region (part4), before the same fence and count, so the last arriver sees all five rows.
template <int B>
__device__ __forceinline__ void finish(const launch_desc & L, const mat_desc & md, int32_t tile, int32_t kp,
                                       int32_t rows_tile, int32_t r, int32_t q, float v, float v4, int32_t lane) {
    const int32_t row = rows_tile * tile + r;
    const bool valid = r < rows_tile && q < B && row < md.m;
    const bool valid4 = B == 5 && r < rows_tile && q == 0 && row < md.m;
    if (md.e == static_cast<int32_t>(epi::conv_silu)) { // unsplit (describe): lane 4 r + b holds row r's x_b
        const int32_t l0 = lane & ~3;
        const float x0 = __shfl_sync(0xffffffffu, v, l0);
        const float x1 = __shfl_sync(0xffffffffu, v, l0 + 1);
        const float x2 = __shfl_sync(0xffffffffu, v, l0 + 2);
        const float x3 = __shfl_sync(0xffffffffu, v, l0 + 3);
        if (q == 0 && r < rows_tile && row < md.m) {
            if constexpr (B == 5) {
                conv_finish5(L.conv_w, L.conv_hist, L.conv_cur, md.out, row, x0, x1, x2, x3, v4);
            } else {
                conv_finish<B>(L.conv_w, L.conv_hist, L.conv_cur, md.out, row, x0, x1, x2, x3);
            }
        }
        return;
    }
    if (md.split == 1 && md.e == static_cast<int32_t>(epi::f32)) {
        if (valid) {
            md.out[q * md.m + row] = scaled(md, q, v);
        }
        if (valid4) {
            md.out[4 * md.m + row] = scaled(md, 4, v4);
        }
        return;
    }
    float * part = L.part + md.part_off;
    if (valid) {
        part[(tile * md.split + kp) * part_slot + r * 4 + q] = v;
    }
    if (valid4) {
        L.part4[md.part_off / 4 + (tile * md.split + kp) * part4_slot + r] = v4;
    }
    __threadfence();
    __syncwarp();
    const bool glu = md.e != static_cast<int32_t>(epi::f32);
    const int32_t group = glu ? tile / 4 : tile;
    const uint32_t need = static_cast<uint32_t>(glu ? 8 * md.split : md.split);
    uint32_t * cnt = L.cnt + md.cnt_off + group;
    uint32_t old = 0;
    if (lane == 0) {
        old = atomicAdd(cnt, 1u);
    }
    old = __shfl_sync(0xffffffffu, old, 0);
    if (old != need - 1) {
        return;
    }
    __threadfence();
    if (!glu) {
        const int32_t rr = lane >> 2, bb = lane & 3, row2 = rows_tile * tile + rr;
        if (rr < rows_tile && bb < B && row2 < md.m) {
            const float s = sum_parts(part + tile * md.split * part_slot + lane, md.split, part_slot);
            md.out[bb * md.m + row2] = scaled(md, bb, s);
        }
        if constexpr (B == 5) { // row 4: lane rr' = lane < 8 sums its row's parts
            const int32_t row4 = rows_tile * tile + lane;
            if (lane < rows_tile && row4 < md.m) {
                const float s4 = sum_parts(L.part4 + md.part_off / 4 + tile * md.split * part4_slot + lane,
                                           md.split, part4_slot);
                md.out[4 * md.m + row4] = scaled(md, 4, s4);
            }
        }
    } else {
        const mat_desc & gm = md.e == static_cast<int32_t>(epi::glu_gate) ? md : L.mats[md.pair];
        const mat_desc & um = md.e == static_cast<int32_t>(epi::glu_up) ? md : L.mats[md.pair];
        if constexpr (B == 5) {
            glu_finish5(L.part + gm.part_off, L.part + um.part_off, L.part4 + gm.part_off / 4,
                        L.part4 + um.part_off / 4, md.split, md.m, group, lane, L.glu_codes, L.glu_scales, L.glu_f32);
        } else {
            glu_finish<B>(L.part + gm.part_off, L.part + um.part_off, md.split, md.m, group, lane, L.glu_codes,
                          L.glu_scales, L.glu_f32);
        }
    }
    if (lane == 0) {
        *cnt = 0u;
    }
}

// ---- the router's epilogue (DP2 Amendment 3) ---------------------------------------------------------------------
// One warp a row, all rows at once, in the launch's last block; per row, ggml's rule (dense.h). Lane l takes experts
// l + 32 i (i < 16) of the row's e = expf(l - max) values in shared memory. The 10 largest p (p descending, ties to the
// lower ID) by threshold: T, a lower bound of the 10th largest p, keeps the candidates p >= T (at least 10, every one
// of the 10 among them), which a bitonic network ranks.
constexpr int32_t route_cand = 64; // candidates a row ranks; a row with more takes the narrower pass
constexpr int32_t route_row_bytes = route_experts * 4 + route_cand * 8; // a row's e values, then its candidates

__device__ __forceinline__ uint32_t lanes_below(int32_t lane) {
    return (1u << lane) - 1u;
}

// x before y in the routing order: p descending, ties to the lower ID
__device__ __forceinline__ bool route_before(float px, int32_t ix, float py, int32_t iy) {
    return px > py || (px == py && ix < iy);
}

// One compare-exchange of the bitonic network: (p, id) against the pair j lanes away. flip: this element's half keeps
// the later of the two (the ascending halves and the upper lane of a descending pair).
__device__ __forceinline__ void route_cex(float & p, int32_t & id, int32_t j, bool flip) {
    const float po = __shfl_xor_sync(0xffffffffu, p, j);
    const int32_t io = __shfl_xor_sync(0xffffffffu, id, j);
    if (route_before(po, io, p, id) != flip) {
        p = po;
        id = io;
    }
}

// p of expert lane + 32 i: e x inv with a NaN sanitized to -FLT_MAX (ggml's rule, so a NaN cannot be selected twice;
// fmaxf drops a NaN)
__device__ __forceinline__ float route_p(const float * e, float inv, int32_t lane, int32_t i) {
    return fmaxf(e[lane + 32 * i] * inv, -FLT_MAX);
}

// 1 / x, correctly rounded, for a normal x whose exponent field is 1 .. 252 (a NaN stays NaN): the precise division's
// fast path (the reciprocal estimate and one Newton step, bitwise its result there) without the slow-path call and
// range test it has for other x
__device__ __forceinline__ float rcp_normal(float x) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    const float t = fmaf(x, r, -1.0f);
    return fmaf(r, -t, r);
}

// The order-preserving key of p (p >= 0, or -FLT_MAX for a NaN row)
__device__ __forceinline__ uint32_t route_key(float p) {
    const uint32_t u = __float_as_uint(p);
    return u ^ ((u >> 31) != 0 ? 0xffffffffu : 0x80000000u);
}

// Candidate c of expert lane + 32 i to slot n + (c's below this lane), when in bounds; returns the count of c
__device__ __forceinline__ int32_t route_put(float2 * cand, bool c, float p, int32_t n, int32_t lane, int32_t i) {
    const uint32_t cb = __ballot_sync(0xffffffffu, c);
    const int32_t at = n + __popc(cb & lanes_below(lane));
    if (c && at < route_cand) {
        cand[at] = make_float2(p, __int_as_float(lane + 32 * i)); // the ID's bits carried in a float slot
    }
    return __popc(cb);
}

// The narrower pass, for a row with more than route_cand candidates (ties beyond the bound: equal or NaN logits): T =
// the exact 10th largest p by ballot counts over its key's 32 bits; the candidates are p > T and the first 10 p == T
// in ID order (at most 19). Rolled: it runs only for such rows.
__device__ __forceinline__ int32_t route_narrow(const float * e, float2 * cand, float inv, int32_t lane) {
    uint32_t k = 0;
#pragma unroll 1
    for (int32_t bit = 31; bit >= 0; --bit) {
        const uint32_t probe = k | (1u << bit);
        int32_t c = 0;
#pragma unroll 1
        for (int32_t i = 0; i < route_experts / 32; ++i) {
            c += __popc(__ballot_sync(0xffffffffu, route_key(route_p(e, inv, lane, i)) >= probe));
        }
        k = c >= route_used ? probe : k;
    }
    const float thr = __uint_as_float(k ^ ((k >> 31) != 0 ? 0x80000000u : 0xffffffffu));
    int32_t n = 0, ties = 0;
#pragma unroll 1
    for (int32_t i = 0; i < route_experts / 32; ++i) {
        const float p = route_p(e, inv, lane, i);
        const uint32_t tb = __ballot_sync(0xffffffffu, p == thr);
        const bool c = p > thr || (p == thr && ties + __popc(tb & lanes_below(lane)) < route_used);
        n += route_put(cand, c, p, n, lane, i);
        ties += __popc(tb);
    }
    return n;
}

// The router's epilogue in the launch's last block: warp b < rows runs row b and its shared gate's sigmoid (after the
// row: the division's slow path is a call, and a call in a branch before the row's shuffles made ptxas give each of
// them a divergent fallback path). Every value comes by value (a descriptor reference made each field a generic load,
// DP2 R17a). sm: the block's route shared memory, route_row_bytes a row. Inlined at its one call site: out of line,
// ptxas could not prove the warps converged either.
__device__ __forceinline__ void route_epilogue(const float * logits, const float * gate, int32_t * ids, float * wts,
                                               float * sg, uint8_t * sm, int32_t rows) {
    const int32_t lane = lane_id(), w = static_cast<int32_t>(threadIdx.x >> 5);
    if (w >= rows) {
        return;
    }
    const float g = __ldcg(gate + w);
    float * e = reinterpret_cast<float *>(sm + w * route_row_bytes);
    float2 * cand = reinterpret_cast<float2 *>(e + route_experts);
    // the row's logits to shared memory, and their maximum
    const float4 * src = reinterpret_cast<const float4 *>(logits + w * route_experts);
    float4 v[4];
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
        v[j] = __ldcg(src + lane + 32 * j);
    }
    float mx = -INFINITY;
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
        reinterpret_cast<float4 *>(e)[lane + 32 * j] = v[j];
        mx = fmaxf(mx, fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w)));
    }
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    __syncwarp();
    // e = expf(l - max), the lane's sum in i order then a fixed tree, and the lane's largest e
    float sum = 0.0f, lmax = 0.0f;
#pragma unroll 4
    for (int32_t i = 0; i < route_experts / 32; ++i) {
        const float x = expf(e[lane + 32 * i] - mx);
        e[lane + 32 * i] = x;
        sum += x;
        lmax = fmaxf(lmax, x);
    }
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    const float inv = rcp_normal(sum); // sum is in [1, 512] (the maximum's e is 1) or NaN
    // T: the 10th largest of the 32 lane maxima of e cut to its top 14 bits (e <= 1: bits 30 and 31 are clear), by
    // ballot counts, then scaled as p is. Any T at or below the 10th largest p is exact; the cut costs candidates only.
    // Two bits a step: the three probes of a step are independent.
    const uint32_t lkey = __float_as_uint(lmax);
    uint32_t t = 0;
#pragma unroll 1
    for (uint32_t one = 1u << 28; one >= (1u << 16); one >>= 2) {
        const auto at_least = [&](uint32_t probe) {
            return __popc(__ballot_sync(0xffffffffu, lkey >= probe)) >= route_used;
        };
        const bool c3 = at_least(t + 3u * one), c2 = at_least(t + 2u * one), c1 = at_least(t + one);
        t += c3 ? 3u * one : c2 ? 2u * one : c1 ? one : 0u;
    }
    const float thr = fmaxf(__uint_as_float(t) * inv, -FLT_MAX);
    // the candidates, four experts a step, their loads ahead of the step's stores (which could alias them)
    int32_t n = 0;
#pragma unroll 1
    for (int32_t i = 0; i < route_experts / 32; i += 4) {
        float p[4];
#pragma unroll
        for (int32_t u = 0; u < 4; ++u) {
            p[u] = route_p(e, inv, lane, i + u);
        }
#pragma unroll
        for (int32_t u = 0; u < 4; ++u) {
            n += route_put(cand, p[u] >= thr, p[u], n, lane, i + u);
        }
    }
    if (n > route_cand) {
        n = route_narrow(e, cand, inv, lane);
    }
    __syncwarp();
    // rank: a bitonic sort of the n (10 .. 64) candidates, padded to 16, 32 or 64 with -inf (pads' IDs above 511);
    // element i = lane (and lane + 32): its block of k sorts descending when (i & k) == 0
    float pa = -INFINITY, pb = -INFINITY;
    int32_t ia = route_experts + lane, ib = route_experts + 32 + lane;
    if (lane < n) {
        const float2 c = cand[lane];
        pa = c.x;
        ia = __float_as_int(c.y);
    }
    if (lane + 32 < n) {
        const float2 c = cand[lane + 32];
        pb = c.x;
        ib = __float_as_int(c.y);
    }
    const int32_t size = n <= 16 ? 16 : n <= 32 ? 32 : 64;
    if (size < 64) {
        // the stages (k, j) in one loop: slot a only
        int32_t k = 2, j = 1;
#pragma unroll 1
        for (;;) {
            route_cex(pa, ia, j, ((lane & j) != 0) != ((lane & k) != 0));
            if (j > 1) {
                j >>= 1;
            } else if (k < size) {
                k <<= 1;
                j = k >> 1;
            } else {
                break;
            }
        }
    } else {
#pragma unroll 1
        for (int32_t k = 2; k <= 64; k <<= 1) {
#pragma unroll 1
            for (int32_t j = k >> 1; j > 0; j >>= 1) {
                if (j == 32) {
                    // within the lane: the 64-block is descending, so slot a keeps the earlier
                    if (route_before(pb, ib, pa, ia)) {
                        const float tp = pa;
                        const int32_t ti = ia;
                        pa = pb;
                        ia = ib;
                        pb = tp;
                        ib = ti;
                    }
                    continue;
                }
                const bool upper = (lane & j) != 0;
                route_cex(pa, ia, j, upper != ((lane & k) != 0));
                route_cex(pb, ib, j, upper != (((lane + 32) & k) != 0));
            }
        }
    }
    // lanes 0 .. 9 hold the 10 in order; the weights' sum in rank order
    float ws = 0.0f;
#pragma unroll
    for (int32_t r = 0; r < route_used; ++r) {
        ws += __shfl_sync(0xffffffffu, pa, r);
    }
    if (lane < route_used) {
        ids[w * route_used + lane] = ia;
        wts[w * route_used + lane] = pa * rcp_normal(fmaxf(ws, route_floor)); // in [route_floor, 10]: p <= 1
    } else if (lane == 31) {
        sg[w] = 1.0f / (1.0f + expf(-g));
    }
}

template <fmt F, int B>
__device__ __forceinline__ void run_q(const launch_desc & L, const mat_desc & md, int32_t tile, int32_t kp,
                                      const act & x, const uint32_t * grid) {
    const int32_t lane = lane_id();
    const int32_t r = lane >> 2, q = lane & 3;
    // the IQ4 table, loaded per unit (an L1 hit) rather than held across units
    constexpr bool iq4 = F == fmt::iq4_xs || F == fmt::iq4_nl;
    const uint4 kv = iq4 ? __ldg(reinterpret_cast<const uint4 *>(L.grids + iq3s_grid_bytes)) : make_uint4(0, 0, 0, 0);
    constexpr int32_t full = step_bytes(F, 4);
    const int64_t tile_bytes = int64_t(tile_q) * sb_bytes(F) * L.k / step_k;
    const uint8_t * base = md.w + tile * tile_bytes;
    const int32_t s0 = kp * md.sp, s1 = s0 + md.sp;
    const int32_t n_full = min(s1, L.k / step_k) - s0; // full steps of this part (a tail step can only be last)
    float acc[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        acc[b] = 0.0f;
    }
    // full steps with one load slot: each step's bytes are copied to registers and the slot is refilled with the next
    // step's at once, before the step's decode and dot products (DP1 step 8, change 1: refilling after the decode left
    // each lane with nothing in flight during it, and the launches ran latency-bound)
    cursor c = make_cursor<F>(base + int64_t(s0) * full, r, q);
    raw<F> v;
    dec64 o;
    // (the refill is unconditional inside the loop and the last step is peeled: a conditional refill made ptxas load
    // into new registers and copy them back, which spilled)
    if (n_full > 0) {
        load_full<F, 0>(v, c);
#pragma unroll 1
        for (int32_t i = 0; i + 1 < n_full; ++i) {
            const raw<F> cur = v;
            c.advance(full);
            load_full<F, 0>(v, c);
            decode<F>(cur, q, kv, grid, o);
            dots<F, B>(o, s0 + i, q, x, acc);
        }
        decode<F>(v, q, kv, grid, o);
        dots<F, B>(o, s0 + n_full - 1, q, x, acc);
    }
    if constexpr (has_tail(F)) { // K % 256 = 64 nq: a last step with nq lanes a row
        const int32_t nq = (L.k % step_k) / 64;
        if (nq > 0 && s1 * step_k > L.k && q < nq) {
            raw<F> rt;
            load_tail<F>(rt, base + int64_t(s0 + n_full) * full, r, q, nq);
            decode<F>(rt, q, kv, grid, o);
            dots<F, B>(o, s0 + n_full, q, x, acc);
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], 1);
        acc[b] += __shfl_xor_sync(0xffffffffu, acc[b], 2);
        sum = q == b ? acc[b] : sum;
    }
    // SP2: row 4's sum is on every lane of the group after the same two shuffles
    finish<B>(L, md, tile, kp, tile_q, r, q, sum, B == 5 ? acc[B - 1] : 0.0f, lane);
}

// BF16 or F32 weights against F32 activations: a tile of 4 rows, one 16-byte load a row per lane per sub-step. BF16:
// lane l takes weights 8l .. 8l + 7 of each 256-weight step. F32 (the draft router's, DR1): each step is two sub-steps
// of 128 weights, lane l taking 4l .. 4l + 3 of each, so the load slots stay the BF16 path's size (a 32-byte lane
// slice spilled at B = 3 and 4).
template <fmt F> struct wide_w {
    static constexpr int32_t bytes = F == fmt::f32 ? 4 : 2;        // a weight's
    static constexpr int32_t sub = F == fmt::f32 ? 2 : 1;          // sub-steps a step
    static constexpr int32_t sub_k = step_k / sub;                 // weights a sub-step
    static constexpr int32_t lane_w = sub_k / 32;                  // weights a lane takes a sub-step
    static_assert(lane_w * bytes == 16, "one 16-byte load a row per lane per sub-step");
};
template <fmt F, int B>
__device__ __forceinline__ void run_wide(const launch_desc & L, const mat_desc & md, int32_t tile, int32_t kp) {
    static_assert(F == fmt::bf16 || F == fmt::f32, "a wide format");
    constexpr int32_t WB = wide_w<F>::bytes, SK = wide_w<F>::sub_k, LW = wide_w<F>::lane_w;
    const int32_t lane = lane_id();
    const int32_t k = L.k;
    const uint8_t * base = md.w + int64_t(tile_bf) * tile * k * WB + 16 * lane;
    const int32_t s0 = kp * md.sp * wide_w<F>::sub, s1 = s0 + md.sp * wide_w<F>::sub; // sub-steps
    const int32_t n_rows = min(tile_bf, md.m - tile_bf * tile);
    // two load slots used in turn (no register copies): a step's loads are issued one step ahead
    uint4 ra[tile_bf], rb[tile_bf];
    const auto ld = [&](uint4(&v)[tile_bf], int32_t s) {
#pragma unroll
        for (int32_t r = 0; r < tile_bf; ++r) {
            v[r] = r < n_rows ? ldw4(base + (int64_t(r) * k + int64_t(s) * SK) * WB) : make_uint4(0, 0, 0, 0);
        }
    };
    float acc[tile_bf][B];
#pragma unroll
    for (int32_t r = 0; r < tile_bf; ++r) {
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            acc[r][b] = 0.0f;
        }
    }
    const auto fma_step = [&](const uint4(&v)[tile_bf], int32_t s) {
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            const float4 * xp = reinterpret_cast<const float4 *>(L.xf + b * k + SK * s + LW * lane);
            if constexpr (F == fmt::bf16) {
                const float4 x0 = __ldg(xp), x1 = __ldg(xp + 1);
                const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                for (int32_t r = 0; r < tile_bf; ++r) {
#pragma unroll
                    for (int32_t i = 0; i < 4; ++i) {
                        const uint32_t wd = word(v[r], i);
                        acc[r][b] = fmaf(__uint_as_float(wd << 16), xv[2 * i], acc[r][b]);
                        acc[r][b] = fmaf(__uint_as_float(wd & 0xffff0000u), xv[2 * i + 1], acc[r][b]);
                    }
                }
            } else {
                const float4 x0 = __ldg(xp);
                const float xv[4] = {x0.x, x0.y, x0.z, x0.w};
#pragma unroll
                for (int32_t r = 0; r < tile_bf; ++r) {
#pragma unroll
                    for (int32_t i = 0; i < 4; ++i) {
                        acc[r][b] = fmaf(__uint_as_float(word(v[r], i)), xv[i], acc[r][b]);
                    }
                }
            }
        }
    };
    // (unconditional refills inside the loop, the last steps peeled, as in run_q)
    ld(ra, s0);
    int32_t s = s0;
#pragma unroll 1
    for (; s + 3 <= s1; s += 2) {
        ld(rb, s + 1);
        fma_step(ra, s);
        ld(ra, s + 2);
        fma_step(rb, s + 1);
    }
    if (s + 2 == s1) {
        ld(rb, s + 1);
        fma_step(ra, s);
        fma_step(rb, s + 1);
    } else {
        fma_step(ra, s);
    }
    const int32_t rl = lane >> 2, ql = lane & 3;
    float v = 0.0f, v4 = 0.0f; // SP2: v4, row rl's sum for activation row 4
#pragma unroll
    for (int32_t r = 0; r < tile_bf; ++r) {
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            float a = acc[r][b];
#pragma unroll
            for (int32_t off = 16; off > 0; off >>= 1) {
                a += __shfl_xor_sync(0xffffffffu, a, off);
            }
            v = (rl == r && ql == b) ? a : v;
            if (B == 5 && b == 4) {
                v4 = rl == r ? a : v4;
            }
        }
    }
    finish<B>(L, md, tile, kp, tile_bf, rl, ql, v, v4, lane);
}

// this warp's units of matrix mi: u = gw + 1,152 j in [unit_base, the next matrix's), gw = warp * 72 + block (so
// consecutive units go to different SMs). gw is read afresh here, not held across the format switch.
template <fmt F, int B>
__device__ __forceinline__ void run_units(const launch_desc & L, int32_t mi, const act & x, const uint32_t * grid) {
    const mat_desc & md = L.mats[mi];
    const int32_t gw = static_cast<int32_t>(threadIdx.x >> 5) * n_sm + static_cast<int32_t>(blockIdx.x);
    const int32_t u0 = md.unit_base, first = u0 + (gw - u0 % n_grid_warps + n_grid_warps) % n_grid_warps;
    // the end, read from the parameter bank each unit in the BF16 path (held across it, B = 4 spilled it)
    const auto end = [&](int32_t m) { return m + 1 < L.n_mats ? L.mats[m + 1].unit_base : L.n_units; };
    const int32_t u1 = end(mi);
    constexpr bool wide = F == fmt::bf16 || F == fmt::f32;
#pragma unroll 1
    for (int32_t u = first; u < (wide ? end(opaque(mi)) : u1); u += n_grid_warps) {
        const int32_t rel = u - md.unit_base;
        const int32_t tile = rel / md.split, kp = rel - tile * md.split;
        if constexpr (!wide) {
            if (md.e == static_cast<int32_t>(epi::conv_silu) && (threadIdx.x & 3) == 0) {
                // the epilogue's history and taps do not depend on the product: into L2 at the unit's start (GD1 2b)
                const int32_t c = tile_q * tile + static_cast<int32_t>((threadIdx.x & 31) >> 2);
                prefetch_l2(L.conv_hist + fn::round::version_slot(*L.conv_cur, 0) * fn::gdn::hist_version_floats +
                            c * fn::gdn::n_hist);
                prefetch_l2(L.conv_w + c * fn::gdn::n_tap);
            }
        }
        if constexpr (wide) {
            run_wide<F, B>(L, md, tile, kp);
        } else {
            run_q<F, B>(L, md, tile, kp, x, grid);
        }
    }
}

// HC3: the warp's first unit of matrix 0 (its first unit of the launch), up to 2 KB of it into L2 before X-HMIX's
// acquire, so the weight stream starts while the peer's half lands (HC-9). Lane l < 16 prefetches line l: a quantized
// unit's steps are contiguous bytes; a BF16 unit's four rows each get four lines (one step; F32: half a step).
template <bool W32> __device__ __forceinline__ void prefetch_first_unit(const launch_desc & L) {
    const mat_desc & md = L.mats[0];
    const int32_t gw = static_cast<int32_t>(threadIdx.x >> 5) * n_sm + static_cast<int32_t>(blockIdx.x);
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    if (gw >= md.tiles * md.split || lane >= 16) {
        return;
    }
    const int32_t tile = gw / md.split, s0 = (gw - tile * md.split) * md.sp;
    const fmt f = static_cast<fmt>(md.f);
    const bool f32 = W32 && f == fmt::f32;
    if (f == fmt::bf16 || f32) {
        const int32_t row = tile_bf * tile + (lane >> 2);
        if (row < md.m) {
            prefetch_l2(md.w + (int64_t(row) * L.k + int64_t(s0) * step_k) * (f32 ? 4 : 2) + 128 * (lane & 3));
        }
    } else {
        const int64_t tile_bytes = int64_t(tile_q) * sb_bytes(f) * L.k / step_k;
        const int64_t unit_bytes = int64_t(md.sp) * step_bytes(f, 4);
        if (128 * lane < unit_bytes) {
            prefetch_l2(md.w + tile * tile_bytes + int64_t(s0) * step_bytes(f, 4) + 128 * lane);
        }
    }
}

// W32 (DR1): the instantiation with the F32 path, launched only when a matrix is F32 (the draft's router), so the
// target's launches keep the kernels without it (their code and registers as before DR1)
template <int B, bool W32>
__global__ void __launch_bounds__(n_threads, 1) product_kernel(const __grid_constant__ launch_desc L) {
    extern __shared__ uint4 smem4[];
    uint8_t * sm = reinterpret_cast<uint8_t *>(smem4);
    const int32_t nblk = L.k / 32;
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    float4 * meta = reinterpret_cast<float4 *>(sm + B * L.k);
    uint32_t * grid = reinterpret_cast<uint32_t *>(sm + (L.need_xq ? B * L.k + B * nblk * 16 : 0));
    if (L.hmix != nullptr) {
        prefetch_first_unit<W32>(L);
    }
    // stage: row b's codes into the swizzled chunks, and per run {d, s} with the exact code sums of its halves; with
    // X-HMIX, the peer half's runs are polled (xchg.cuh hmix_run)
    if (L.need_xq) {
        for (int32_t i = tid; i < B * nblk; i += n_threads) {
            const int32_t b = i / nblk, j = i - b * nblk;
            uint4 c0, c1;
            float2 sc;
            if (L.hmix != nullptr) {
                fn::xchg::hmix_run(L.hmix, L.hmix_peer, b, j, c0, c1, sc, L.err);
            } else {
                const uint4 * src = reinterpret_cast<const uint4 *>(L.xc + b * L.k + 32 * j);
                c0 = __ldg(src);
                c1 = __ldg(src + 1);
                sc = __ldg(L.xs + i);
            }
            int32_t slo = 0, shi = 0;
#pragma unroll
            for (int32_t t = 0; t < 4; ++t) {
                slo = dp4a(word(c0, t), 0x01010101u, slo);
                shi = dp4a(word(c1, t), 0x01010101u, shi);
            }
            uint4 * xs = reinterpret_cast<uint4 *>(sm + b * L.k);
            xs[swz(2 * j)] = c0;
            xs[swz(2 * j + 1)] = c1;
            meta[i] =
                make_float4(sc.x, sc.y, __int_as_float(slo), __int_as_float(shi)); // the sums' bits in float slots
        }
    }
    if (L.hmix != nullptr && L.need_xf) { // the F32 row's peer half, read after the barrier through L1 (xchg.h)
        fn::xchg::hmix_acquire_f32(L.hmix, L.hmix_peer, B, tid, n_threads, L.err);
    }
    if (L.need_grid) {
        for (int32_t i = tid; i < iq3s_grid_bytes / 16; i += n_threads) {
            reinterpret_cast<uint4 *>(grid)[i] = __ldg(reinterpret_cast<const uint4 *>(L.grids) + i);
        }
    }
    __syncthreads();
    const act x = {sm, meta, L.k};
    // matrix by matrix: the format switch is taken once per matrix; each format's path loops over its own units
    for (int32_t mi = 0; mi < L.n_mats; ++mi) {
        switch (static_cast<fmt>(L.mats[mi].f)) {
            case fmt::q4_k:
                run_units<fmt::q4_k, B>(L, mi, x, grid);
                break;
            case fmt::q5_k:
                run_units<fmt::q5_k, B>(L, mi, x, grid);
                break;
            case fmt::q6_k:
                run_units<fmt::q6_k, B>(L, mi, x, grid);
                break;
            case fmt::iq4_xs:
                run_units<fmt::iq4_xs, B>(L, mi, x, grid);
                break;
            case fmt::iq3_s:
                run_units<fmt::iq3_s, B>(L, mi, x, grid);
                break;
            case fmt::q8_0:
                run_units<fmt::q8_0, B>(L, mi, x, grid);
                break;
            case fmt::iq4_nl:
                run_units<fmt::iq4_nl, B>(L, mi, x, grid);
                break;
            case fmt::q2_0:
                run_units<fmt::q2_0, B>(L, mi, x, grid);
                break;
            case fmt::bf16:
                run_units<fmt::bf16, B>(L, mi, x, grid);
                break;
            case fmt::f32:
                if constexpr (W32) {
                    run_units<fmt::f32, B>(L, mi, x, grid);
                }
                break;
        }
    }
    // The router (DP2 Amendment 3): the launch's last block to finish runs the epilogue. Hand-off: every thread's
    // output stores, its __threadfence (publication), the block barrier, then thread 0's atomicAdd on the counter; in
    // the block that arrives last, thread 0's __threadfence (acquire) before the barrier that releases its warps to
    // read the logits through L2 (ld.cg); thread 0 resets the counter there (no other block touches it again in this
    // launch; the next launch's increments are ordered by the kernel boundary).
    if (L.route_cnt != nullptr) {
        __threadfence();
        __syncthreads();
        int32_t last = 0;
        if (threadIdx.x == 0) { // read afresh: tid held across the matrix loop spilled at B = 4
            last = atomicAdd(L.route_cnt, 1u) == static_cast<uint32_t>(n_sm) - 1 ? 1 : 0;
            if (last != 0) {
                __threadfence();
                *L.route_cnt = 0u;
            }
        }
        if (__syncthreads_or(last) != 0) {
            route_epilogue(L.mats[L.route_logits].out, L.mats[L.route_gate].out, L.ids, L.wts, L.sg, sm + L.smem_route,
                           B);
        }
    }
}

// ---- host ------------------------------------------------------------------------------------------------------
bool is_q(fmt f) {
    return f != fmt::bf16 && f != fmt::f32;
}

bool known(fmt f) {
    return static_cast<int32_t>(f) >= 0 && static_cast<int32_t>(f) <= static_cast<int32_t>(fmt::f32);
}

// the route epilogue's shared memory: a row's e values and candidates for each of B rows
size_t route_smem(int32_t rows) {
    return size_t(rows) * route_row_bytes;
}

bool has_route(const product_args & a) {
    return a.route.logits >= 0 || a.route.gate >= 0;
}

int32_t tile_rows(fmt f) {
    return is_q(f) ? tile_q : tile_bf;
}

bool shape_ok(fmt f, int32_t m, int32_t k) {
    if (!known(f) || m <= 0 || k <= 0 || k > max_k) {
        return false;
    }
    if (is_q(f) && m % tile_q != 0) {
        return false;
    }
    return has_tail(f) ? k % 64 == 0 : k % step_k == 0;
}

bool aligned(const void * p, size_t a) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % a == 0;
}

// the largest launch: K 5,120 staged at B = 4, the IQ3_S grid and the router's rows, within the 48 KB default (no
// opt-in attribute is set). SP2: B = 5 launches stage at most K 3,072 (the eh_proj's K 5,120 runs at B <= 4,
// dense.h); describe rejects a B = 5 launch with a larger K.
constexpr int32_t max_k_b5 = 3072;

bool reject(const char * why, int32_t i) {
    fprintf(stderr, "dense::product: %s (matrix %d)\n", why, i);
    return false;
}

// the descriptor of a launch; false (with the reason printed) on anything the contract rejects
bool describe(int32_t rows, const product_args & a, launch_desc & L) {
    memset(&L, 0, sizeof(L));
    if (rows < 1 || rows > max_rows) {
        return reject("rows outside 1..5", -1);
    }
    if (a.n_mats < 1 || a.n_mats > max_mats || a.k <= 0 || a.k > max_k) {
        return reject("matrix count or K", -1);
    }
    if (rows == 5 && a.k > max_k_b5) {
        return reject("K above 3,072 at B = 5 (shared memory)", -1);
    }
    int32_t units = 0, n_conv = 0;
    bool any_q = false;
    for (int32_t i = 0; i < a.n_mats; ++i) {
        const mat & m = a.mats[i];
        if (!known(m.f)) {
            fprintf(stderr, "dense::product: format %d is not a dense format of the model\n", static_cast<int>(m.f));
            return false;
        }
        if (!shape_ok(m.f, m.m, a.k)) {
            fprintf(stderr, "dense::product: %s [%d -> %d] is not a supported shape\n", fmt_name(m.f), a.k, m.m);
            return false;
        }
        if (!aligned(m.w, 16)) {
            return reject("weights null or not 16-byte aligned", i);
        }
        const int32_t steps = n_steps(a.k);
        if (m.split < 1 || m.split > max_split || steps % m.split != 0 || (m.split > 1 && a.k % step_k != 0)) {
            return reject("split must divide the K steps, at most 12 parts (and K % 256 == 0 when split)", i);
        }
        const bool glu = m.e == epi::glu_gate || m.e == epi::glu_up;
        if (m.e == epi::conv_silu) {
            if (n_conv++ > 0 || m.m != fn::gdn::n_channel || m.split != 1 || !is_q(m.f) || !aligned(m.out, 16) ||
                m.scale != nullptr || !aligned(a.conv.w, 16) || !aligned(a.conv.hist, 16) || a.conv.cur == nullptr) {
                return reject(
                    "conv_silu: one quantized matrix of 5,120 rows, unsplit, unscaled, with taps, history and "
                    "output 16-byte aligned and the committed slot's word",
                    i);
            }
        } else if (glu) {
            const int32_t p = m.pair;
            if (p < 0 || p >= a.n_mats || p == i || a.mats[p].pair != i || a.mats[p].m != m.m ||
                a.mats[p].split != m.split || a.mats[p].e == m.e || a.mats[p].e == epi::f32 || m.m % 32 != 0 ||
                !is_q(m.f) || !aligned(a.glu_codes, 16) || !aligned(a.glu_scales, 8) || m.scale != nullptr) {
                return reject("glu pair: gate and up with equal M % 32 == 0 and split, glu codes and scales", i);
            }
        } else if (!aligned(m.out, 4) || (m.scale != nullptr && !aligned(m.scale, 4))) {
            return reject("output null or misaligned, or scale misaligned", i);
        }
        any_q = any_q || is_q(m.f);
        mat_desc & d = L.mats[i];
        d.w = m.w;
        d.out = m.out;
        d.scale = m.scale;
        d.m = m.m;
        d.f = static_cast<int32_t>(m.f);
        d.e = static_cast<int32_t>(m.e);
        d.pair = m.pair;
        d.split = m.split;
        d.tiles = (m.m + tile_rows(m.f) - 1) / tile_rows(m.f);
        d.unit_base = units;
        d.steps = steps;
        d.sp = steps / m.split;
        units += d.tiles * m.split;
        L.need_grid |= m.f == fmt::iq3_s ? 1 : 0;
        L.need_kv |= (m.f == fmt::iq4_xs || m.f == fmt::iq4_nl) ? 1 : 0;
    }
    work_layout wl; // the partials' and counters' offsets (one rule with work_bytes)
    if (!layout_of(a, wl)) {
        return reject("work layout", -1);
    }
    for (int32_t i = 0; i < a.n_mats; ++i) {
        L.mats[i].part_off = std::max(wl.mats[i].part_off, 0);
        L.mats[i].cnt_off = std::max(wl.mats[i].cnt_off, 0);
    }
    bool need_work = wl.part_floats > 0 || wl.cnt_words > 0;
    const int8_t * x_codes = a.x_codes;
    const float * x_scales = a.x_scales;
    const float * x_f32 = a.x_f32;
    if (a.hmix != nullptr) { // X-HMIX: the activations are the slot's regions (xchg.h hmix_layout), K the full row
        using hl = fn::xchg::hmix_layout;
        if (a.k != hl::width || a.x_codes != nullptr || a.x_scales != nullptr || a.x_f32 != nullptr ||
            !aligned(a.hmix, 16) || a.err == nullptr || (a.hmix_peer_half != 0 && a.hmix_peer_half != 1)) {
            return reject("X-HMIX: K 2,560, a 16-byte aligned slot, an error word, the peer's half 0 or 1, and no "
                          "separate activation pointers",
                          -1);
        }
        x_codes = reinterpret_cast<const int8_t *>(a.hmix + hl::code_at); // the slot's int8 codes region
        x_scales = a.hmix + hl::scale_at;
        x_f32 = a.hmix;
    }
    if (any_q && (!aligned(x_codes, 16) || !aligned(x_scales, 8))) {
        return reject("x codes (16-byte) or scales (8-byte) null or misaligned", -1);
    }
    if (has_route(a)) {
        const int32_t li = a.route.logits, gi = a.route.gate;
        const auto route_mat = [&](int32_t i, int32_t m) {
            return i >= 0 && i < a.n_mats && (a.mats[i].f == fmt::bf16 || a.mats[i].f == fmt::f32) &&
                   a.mats[i].m == m && a.mats[i].e == epi::f32 && a.mats[i].scale == nullptr;
        };
        if (!route_mat(li, route_experts) || !route_mat(gi, 1) || li == gi || !aligned(a.mats[li].out, 16) ||
            !aligned(a.route.ids, 4) || !aligned(a.route.wts, 4) || !aligned(a.route.sg, 4)) {
            return reject("router: BF16 or F32 logits [K -> 512] (16-byte aligned) and gate [K -> 1], unscaled, and "
                          "outputs",
                          -1);
        }
        L.route_logits = li;
        L.route_gate = gi;
        L.ids = a.route.ids;
        L.wts = a.route.wts;
        L.sg = a.route.sg;
        need_work = true;
    }
    int32_t any_wide = 0; // a BF16 or F32 matrix: F32 activations
    for (int32_t i = 0; i < a.n_mats; ++i) {
        if (!is_q(a.mats[i].f) && !aligned(x_f32, 16)) {
            return reject("x_f32 null or not 16-byte aligned", i);
        }
        any_wide |= is_q(a.mats[i].f) ? 0 : 1;
    }
    if ((L.need_grid || L.need_kv) && !aligned(a.grids, 16)) {
        return reject("grids null or not 16-byte aligned", -1);
    }
    if (need_work && !aligned(a.work, 16)) {
        return reject("work null or not 16-byte aligned", -1);
    }
    L.n_mats = a.n_mats;
    L.k = a.k;
    L.n_units = units;
    L.need_xq = any_q ? 1 : 0;
    L.xc = x_codes;
    L.xs = reinterpret_cast<const float2 *>(x_scales); // {d, s} pairs, 8-byte aligned (checked)
    L.xf = x_f32;
    L.hmix = a.hmix;
    L.err = a.err;
    L.hmix_peer = a.hmix_peer_half;
    L.need_xf = any_wide;
    L.grids = a.grids;
    L.glu_codes = a.glu_codes;
    L.glu_scales = a.glu_scales;
    L.glu_f32 = a.glu_f32;
    L.part = static_cast<float *>(a.work);
    L.cnt = need_work ? reinterpret_cast<uint32_t *>(static_cast<float *>(a.work) + wl.part_floats) : nullptr;
    L.route_cnt = has_route(a) ? L.cnt + wl.route_cnt : nullptr;
    L.part4 = need_work ? static_cast<float *>(a.work) + wl.part_floats + wl.cnt_words + (has_route(a) ? 1 : 0)
                        : nullptr;
    L.conv_w = n_conv > 0 ? a.conv.w : nullptr;
    L.conv_hist = n_conv > 0 ? a.conv.hist : nullptr;
    L.conv_cur = n_conv > 0 ? a.conv.cur : nullptr;
    return true;
}

static_assert(size_t(4) * max_k * 3 / 2 + iq3s_grid_bytes + size_t(4) * route_row_bytes <= 48 * 1024 &&
                  size_t(max_rows) * max_k_b5 * 3 / 2 + iq3s_grid_bytes + size_t(max_rows) * route_row_bytes <=
                      48 * 1024,
              "the product's dynamic shared memory");

template <int B> bool launch(launch_desc & L, cudaStream_t st) {
    size_t smem = (L.need_xq ? size_t(B) * L.k * 3 / 2 : 0) + (L.need_grid ? iq3s_grid_bytes : 0);
    L.smem_route = static_cast<int32_t>(smem); // a multiple of 16: K % 32 == 0
    smem += L.route_cnt != nullptr ? route_smem(B) : 0;
    bool w32 = false;
    for (int32_t i = 0; i < L.n_mats; ++i) {
        w32 = w32 || static_cast<fmt>(L.mats[i].f) == fmt::f32;
    }
    if (w32) {
        product_kernel<B, true><<<n_sm, n_threads, smem, st>>>(L);
    } else {
        product_kernel<B, false><<<n_sm, n_threads, smem, st>>>(L);
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

const char * fmt_name(fmt f) {
    switch (f) {
        case fmt::q4_k:
            return "Q4_K";
        case fmt::q5_k:
            return "Q5_K";
        case fmt::q6_k:
            return "Q6_K";
        case fmt::iq4_xs:
            return "IQ4_XS";
        case fmt::iq3_s:
            return "IQ3_S";
        case fmt::q8_0:
            return "Q8_0";
        case fmt::iq4_nl:
            return "IQ4_NL";
        case fmt::q2_0:
            return "Q2_0";
        case fmt::bf16:
            return "BF16";
        case fmt::f32:
            return "F32";
    }
    return "?";
}

int64_t row_bytes(fmt f, int32_t k) {
    if (!known(f) || k <= 0) {
        return 0;
    }
    if (f == fmt::bf16) {
        return int64_t(2) * k;
    }
    if (f == fmt::f32) {
        return int64_t(4) * k;
    }
    if (f == fmt::q2_0) {
        return k % 64 == 0 ? int64_t(k / 64) * 18 : 0;
    }
    if (has_tail(f)) {
        return k % 32 == 0 ? int64_t(k / 32) * (f == fmt::q8_0 ? 34 : 18) : 0;
    }
    return k % step_k == 0 ? int64_t(k / step_k) * sb_bytes(f) : 0;
}

namespace {

// 32 IQ4 weights: GGUF byte j holds weights j (low nibble) and 16 + j; the packed byte b holds 2b and 2b + 1, so the
// kernel's table lookup yields weights in K order
void iq4_chunk(uint8_t * dst, const uint8_t * src) {
    const auto w = [&](int32_t i) { return i < 16 ? src[i] & 0xf : src[i - 16] >> 4; };
    for (int32_t b = 0; b < 16; ++b) {
        dst[b] = static_cast<uint8_t>(w(2 * b) | (w(2 * b + 1) << 4));
    }
}

// 64 Q2_0 codes (16 bytes): GGUF weight w sits in bits 2 (w % 4) of byte w / 4; the packed byte j of word c holds
// weight 16 c + 4 i + j in bits 2 i, so the kernel's shift-and-mask by i yields four weights in K order
void q2_chunk(uint8_t * dst, const uint8_t * src) {
    for (int32_t c = 0; c < 4; ++c) {
        for (int32_t j = 0; j < 4; ++j) {
            uint32_t v = 0;
            for (int32_t i = 0; i < 4; ++i) {
                const int32_t w = 16 * c + 4 * i + j;
                v |= ((static_cast<uint32_t>(src[w / 4]) >> (2 * (w % 4))) & 3u) << (2 * i);
            }
            dst[4 * c + j] = static_cast<uint8_t>(v);
        }
    }
}

} // namespace

bool pack(fmt f, int32_t m, int32_t k, const uint8_t * src, size_t src_row_bytes, uint8_t * dst) {
    if (!shape_ok(f, m, k) || src == nullptr || dst == nullptr || src_row_bytes < size_t(row_bytes(f, k))) {
        return false;
    }
    const int64_t rb = row_bytes(f, k);
    if (!is_q(f)) {
        for (int32_t i = 0; i < m; ++i) {
            memcpy(dst + i * rb, src + i * src_row_bytes, size_t(rb));
        }
        return true;
    }
    const int32_t steps = n_steps(k);
    for (int32_t t = 0; t < m / tile_q; ++t) {
        uint8_t * tb = dst + int64_t(t) * tile_q * rb;
        for (int32_t s = 0; s < steps; ++s) {
            const int32_t nq = std::min(4, (k - step_k * s) / 64);
            uint8_t * p = tb + int64_t(s) * step_bytes(f, 4);
            for (int32_t r = 0; r < tile_q; ++r) {
                const uint8_t * row = src + (int64_t(t) * tile_q + r) * src_row_bytes;
                for (int32_t q = 0; q < nq; ++q) {
                    const int32_t li = r * nq + q;
                    if (f == fmt::q2_0) { // block 4 s + q of the row
                        const uint8_t * blk = row + int64_t(4 * s + q) * 18;
                        q2_chunk(p + 16 * li, blk + 2);
                        memcpy(p + 128 * nq + 2 * li, blk, 2);
                        continue;
                    }
                    if (f == fmt::q8_0 || f == fmt::iq4_nl) {
                        const int32_t bb = f == fmt::q8_0 ? 34 : 18, nqs = bb - 2; // block bytes: d, then qs
                        const uint8_t * b0 = row + int64_t(8 * s + 2 * q) * bb;
                        const uint8_t * b1 = b0 + bb;
                        const int32_t c16 = nqs / 16; // 16-byte chunks a block
                        for (int32_t c = 0; c < c16; ++c) {
                            if (f == fmt::iq4_nl) {
                                iq4_chunk(p + 16 * li, b0 + 2);
                                iq4_chunk(p + 128 * nq + 16 * li, b1 + 2);
                            } else {
                                memcpy(p + c * 128 * nq + 16 * li, b0 + 2 + 16 * c, 16);
                                memcpy(p + (c16 + c) * 128 * nq + 16 * li, b1 + 2 + 16 * c, 16);
                            }
                        }
                        uint8_t * dp = p + 2 * c16 * 128 * nq + 4 * li;
                        memcpy(dp, b0, 2);
                        memcpy(dp + 2, b1, 2);
                        continue;
                    }
                    const uint8_t * sb = row + int64_t(s) * sb_bytes(f);
                    switch (f) {
                        case fmt::q4_k: // {d, dmin, scales[12]} qs[128]
                            memcpy(p + 16 * li, sb + 16 + 32 * q, 16);
                            memcpy(p + 512 + 16 * li, sb + 16 + 32 * q + 16, 16);
                            break;
                        case fmt::q5_k: // {d, dmin, scales[12]} qh[32] qs[128]
                            memcpy(p + 16 * li, sb + 48 + 32 * q, 16);
                            memcpy(p + 512 + 16 * li, sb + 48 + 32 * q + 16, 16);
                            break;
                        case fmt::q6_k: { // ql[128] qh[64] scales[16] d
                            const int32_t n = q >> 1, h = q & 1;
                            memcpy(p + 16 * li, sb + 64 * n + 16 * h, 16);
                            memcpy(p + 512 + 16 * li, sb + 64 * n + 32 + 16 * h, 16);
                            memcpy(p + 1024 + 16 * li, sb + 128 + 32 * n + 16 * h, 16);
                            break;
                        }
                        case fmt::iq4_xs: // d scales_h scales_l[4] qs[128]
                            iq4_chunk(p + 16 * li, sb + 8 + 32 * q);
                            iq4_chunk(p + 512 + 16 * li, sb + 8 + 32 * q + 16);
                            break;
                        case fmt::iq3_s: // d qs[64] qh[8] signs[32] scales[4]
                            memcpy(p + 16 * li, sb + 2 + 16 * q, 16);
                            memcpy(p + 512 + 8 * li, sb + 74 + 8 * q, 8);
                            memcpy(p + 768 + 2 * li, sb + 66 + 2 * q, 2);
                            p[832 + li] = sb[106 + q];
                            break;
                        default:
                            break;
                    }
                }
                if (has_tail(f)) {
                    continue;
                }
                const uint8_t * sb = row + int64_t(s) * sb_bytes(f);
                switch (f) { // the row segments
                    case fmt::q4_k:
                        memcpy(p + 1024 + 16 * r, sb, 16);
                        break;
                    case fmt::q5_k:
                        memcpy(p + 1024 + 16 * r, sb + 16, 16);
                        memcpy(p + 1152 + 16 * r, sb + 32, 16);
                        memcpy(p + 1280 + 16 * r, sb, 16);
                        break;
                    case fmt::q6_k:
                        memcpy(p + 1536 + 16 * r, sb + 192, 16);
                        memcpy(p + 1664 + 2 * r, sb + 208, 2);
                        break;
                    case fmt::iq4_xs:
                        memcpy(p + 1024 + 8 * r, sb, 8);
                        break;
                    case fmt::iq3_s:
                        memcpy(p + 864 + 2 * r, sb, 2);
                        break;
                    default:
                        break;
                }
            }
        }
    }
    return true;
}

bool grid_table(const void * iq3s_grid, size_t iq3s_bytes, const void * kvalues, size_t kvalues_bytes, uint8_t * dst) {
    if (iq3s_grid == nullptr || kvalues == nullptr || dst == nullptr || iq3s_bytes != iq3s_grid_bytes ||
        kvalues_bytes != 16) {
        return false;
    }
    memcpy(dst, iq3s_grid, iq3s_bytes);
    memcpy(dst + iq3s_grid_bytes, kvalues, kvalues_bytes);
    return true;
}

void choose_splits(product_args & a) {
    if (a.n_mats < 1 || a.n_mats > max_mats || a.k <= 0) {
        return;
    }
    const int32_t steps = n_steps(a.k);
    bool unset[max_mats] = {};
    int32_t tiles[max_mats] = {};
    int32_t units = 0;
    for (int32_t i = 0; i < a.n_mats; ++i) {
        mat & m = a.mats[i];
        tiles[i] = known(m.f) && m.m > 0 ? (m.m + tile_rows(m.f) - 1) / tile_rows(m.f) : 0;
        unset[i] = m.split == 0 && m.e != epi::conv_silu;
        m.split = m.split == 0 ? 1 : m.split;
        units += tiles[i] * m.split;
    }
    if (units > n_grid_warps || a.k % step_k != 0) {
        return;
    }
    // grow each unset split (a glu pair together, through its gate) while the launch stays within one round
    for (int32_t i = 0; i < a.n_mats; ++i) {
        mat & m = a.mats[i];
        const bool pair = m.e == epi::glu_gate && m.pair >= 0 && m.pair < a.n_mats;
        if (!unset[i] || m.e == epi::glu_up || (pair && !unset[m.pair])) {
            continue;
        }
        const int32_t t = tiles[i] + (pair ? tiles[m.pair] : 0);
        int32_t best = 1;
        for (int32_t d = 2; d <= steps; ++d) {
            if (steps % d == 0 && d <= max_split && units + t * (d - 1) <= n_grid_warps) {
                best = d;
            }
        }
        units += t * (best - 1);
        m.split = best;
        if (pair) {
            a.mats[m.pair].split = best;
        }
    }
}

bool layout_of(const product_args & a, work_layout & out) {
    out = work_layout();
    if (a.n_mats < 1 || a.n_mats > max_mats) {
        return false;
    }
    work_layout l;
    l.k = a.k;
    l.n_mats = a.n_mats;
    int64_t part = 0, cnt = 0, part4 = 0;
    for (int32_t i = 0; i < a.n_mats; ++i) {
        const mat & m = a.mats[i];
        if (!known(m.f) || m.m <= 0 || m.split < 1) {
            return false;
        }
        mat_layout & d = l.mats[i];
        d.m = m.m;
        d.tile_rows = tile_rows(m.f);
        d.tiles = (m.m + d.tile_rows - 1) / d.tile_rows;
        d.split = m.split;
        d.e = m.e;
        d.pair = m.pair;
        if (m.split > 1 || m.e == epi::glu_gate || m.e == epi::glu_up) {
            d.part_off = static_cast<int32_t>(part);
            part += int64_t(d.tiles) * m.split * part_slot;
            part4 += int64_t(d.tiles) * m.split * part4_slot;
        }
        if (part > INT32_MAX || part4 > INT32_MAX) {
            return false;
        }
    }
    for (int32_t i = 0; i < a.n_mats; ++i) { // counters: a split plain product's tiles, a glu pair's 32-row groups
        mat_layout & d = l.mats[i];
        if (d.e == epi::f32 && d.split > 1) {
            d.cnt_off = static_cast<int32_t>(cnt);
            cnt += d.tiles;
        } else if (d.e == epi::glu_gate) {
            d.cnt_off = static_cast<int32_t>(cnt);
            if (d.pair >= 0 && d.pair < a.n_mats) {
                l.mats[d.pair].cnt_off = d.cnt_off;
            }
            cnt += d.m / 32;
        }
    }
    l.part_floats = static_cast<int32_t>(part);
    l.cnt_words = static_cast<int32_t>(cnt);
    l.route_cnt = has_route(a) ? l.cnt_words : -1;
    l.part4_floats = static_cast<int32_t>(part4);
    out = l;
    return true;
}

bool same_layout(const work_layout & x, const work_layout & y) {
    if (x.k != y.k || x.n_mats != y.n_mats || x.part_floats != y.part_floats || x.cnt_words != y.cnt_words ||
        x.route_cnt != y.route_cnt || x.part4_floats != y.part4_floats) {
        return false;
    }
    for (int32_t i = 0; i < x.n_mats && i < max_mats; ++i) {
        const mat_layout & p = x.mats[i];
        const mat_layout & q = y.mats[i];
        if (p.m != q.m || p.tile_rows != q.tile_rows || p.tiles != q.tiles || p.split != q.split || p.e != q.e ||
            p.pair != q.pair || p.part_off != q.part_off || p.cnt_off != q.cnt_off) {
            return false;
        }
    }
    return true;
}

size_t work_bytes(const product_args & a) {
    work_layout l;
    if (!layout_of(a, l)) {
        return 0;
    }
    return (size_t(l.part_floats) + size_t(l.cnt_words) + (l.route_cnt >= 0 ? 1 : 0) + size_t(l.part4_floats)) * 4;
}

bool product(int32_t rows, const product_args & a, void * stream) {
    launch_desc L;
    if (!describe(rows, a, L)) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1:
            return launch<1>(L, st);
        case 2:
            return launch<2>(L, st);
        case 3:
            return launch<3>(L, st);
        case 4:
            return launch<4>(L, st);
        default:
            return launch<5>(L, st);
    }
}

} // namespace fn::dense
