#include "fn/kernels/experts/experts_prompt.h"
#include "fn/kernels/experts/experts_prompt_dev.cuh"
#include "fn/model/shapes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <type_traits>

namespace fn::experts {
namespace {

using namespace epd;

// ---- gate/up kernel ----------------------------------------------------------------------------------------------
// A block a task (grid ep_max_tasks; blocks past the task count exit at once), one resident an SM: 640 threads.
// Thread (w, lane) owns column c = 32 w + 8 qp + r of the warp's 32: channel 16 w + 8 (qp & 1) + r of this card, gate
// for qp < 2, up otherwise; its lane is that column's B fragment holder (quadpair qp, row r of m8n8k4), so it unpacks
// its own column's sub-block, chunk by chunk, straight into the products: each weight is unpacked once a task and no
// weight passes through shared memory. Threads below 192 stage the activation tiles (PB3: a ring of four, step j's
// tile staged during step j - 2, one barrier every second step):
// row t / 4 of the task, 8 values (t % 4) of the step's 32 FP16 values, as the producer wrote them.
// ME3: X, the cache's instantiation (ep_gate_up_args::wt); the resident one (X false) reads the bank as 412ecbd's
// APF2: Part 1's (0f47b150) kernel, instruction for instruction (APF1 Part 4's grouped skip is gone with its grouped
// kernel; the ladder's grouped route, experts_ladder.cu, takes every expert of a layer it runs)
template <bool X, fmt F> __global__ void __launch_bounds__(nt_threads, 1) ep_gate_up_kernel(ep_gate_up_args a) {
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
    constexpr int32_t rep = G.rep > 0 ? G.rep : 1; // ME1: the K-quants have no table
    const uint32_t lo = static_cast<uint32_t>((lane % rep) * G.eb);
    const int32_t srow = tid >> 2, sq = tid & 3; // staging
    const uint32_t srow_a = tile_row(at + srow * 64, asw(srow));
    constexpr int32_t n_steps = D / ks;
    constexpr int32_t TB = ep_tile_bytes(F);
    const int32_t task = block_task<X>(a);
    if constexpr (X) {
        // ME4: a ring group's tasks only
        if (a.task_off != nullptr && task >= __ldg(a.task_off + a.e_hi)) {
            return;
        }
    }
    if (task >= n_tasks_of(a.route)) {
        return;
    }
    if constexpr (G.entries > 0) { // ME1: the K-quants have no table
        build_tables<F>(tb, a.grids);
    }
    {
        const task_t tk = task_of(a.route.tasks, task);
        const int32_t nt = (tk.n + 7) >> 3;
        // ME1: the task's expert from the cache's table (offsets from its first byte), else from the resident bank
        const uint8_t * const wb = X ? a.wt[tk.e] : a.w;
        const int32_t e_tiles = X ? 0 : tk.e * n_group;
        // this column's load offsets, kept in shared memory and re-read each step (registers are the limit here)
        reinterpret_cast<int4 *>(ofs)[tid] = gu_offsets<F>(col, (e_tiles + g) * TB);
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
            if (F != fmt::iq2_s && F != fmt::iq3_s && !is_kq(F) && sreal && sq == 0) {
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
            raw_of<F> raw[2];
            {
                const int4 o = ld_ofs(ofs + 16 * tid);
                raw[0] = raw_load<F>(wb, o, 0);
                raw_prefetch<F>(wb, o, 1);
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
                if constexpr (is_kq(F)) {
                    // ME1: the next sub-block is loaded once chunks 0 and 2 have consumed words x, y of this one (both
                    // slots whole spill at 96 registers); chunks in the order 0, 2, 1, 3
                    if (j + 2 < n_steps) {
                        x_stage(x_load(j + 2), srow_a + ((T + 2) & 3) * a_tile);
                    }
                    const a_rows ar = {arow.even + T * a_tile, arow.odd + T * a_tile};
                    const gu_scale sc = kq_scale(raw[P]);
                    mma_chunk<NT>(acc, kq_chunk<F>(raw[P], sc, 0), ar, 0);
                    mma_chunk<NT>(acc, kq_chunk<F>(raw[P], sc, 2), ar, 2);
                    if (j + 1 < n_steps) {
                        const int4 o = ld_ofs(ofs + 16 * tid);
                        raw[P ^ 1] = raw_load<F>(wb, o, j + 1);
                        if (j + 2 < n_steps) {
                            raw_prefetch<F>(wb, o, j + 2);
                        }
                    }
                    mma_chunk<NT>(acc, kq_chunk<F>(raw[P], sc, 1), ar, 1);
                    mma_chunk<NT>(acc, kq_chunk<F>(raw[P], sc, 3), ar, 3);
                    if constexpr ((T & 1) != 0) {
                        __syncthreads();
                    }
                } else {
                    if (j + 3 < n_steps) {
                        x_prefetch(j + 3);
                    }
                    if (j + 1 < n_steps) {
                        const int4 o = ld_ofs(ofs + 16 * tid);
                        raw[P ^ 1] = raw_load<F>(wb, o, j + 1);
                        if (j + 2 < n_steps) {
                            raw_prefetch<F>(wb, o, j + 2);
                        }
                    }
                    if (j + 2 < n_steps) {
                        // rows hit L1 (prefetched): x is a few MB
                        x_stage(x_load(j + 2), srow_a + ((T + 2) & 3) * a_tile);
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
            const int2 fr = task_rows(a.route.tasks, block_task_fresh<X>(a));
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
// ME3: X as ep_gate_up_kernel's
template <bool X, fmt F> __global__ void __launch_bounds__(dn_threads, 1) ep_down_kernel(ep_down_args a) {
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
    const int32_t task = block_task<X>(a);
    if constexpr (X) {
        // ME4: a ring group's tasks only
        if (a.task_off != nullptr && task >= __ldg(a.task_off + a.e_hi)) {
            return;
        }
    }
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
        // ME1: the task's expert from the cache's table, else from the resident bank
        const uint8_t * const wb = X ? a.wt[tk.e] : a.w;
        const int32_t eo = X ? 0 : tk.e * dn_bytes(F);
        // two raw slots with compile-time indices (R12); step 0's weights load beside the staging
        dn_raw_of<F> raw[2];
        raw[0] = dn_load<F>(wb, eo, n16, 0);
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
                    raw[P ^ 1] = dn_load<F>(wb, eo, n16 + g * g16, j + 1);
                } else if (g + 1 < ep_n_col_groups) {
                    raw[P ^ 1] = dn_load<F>(wb, eo, n16 + (g + 1) * g16, 0);
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
                const int2 fr = task_rows(a.route.tasks, block_task<X>(a));
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

__global__ void __launch_bounds__(grp_threads, 1) ep_group_kernel(const int32_t * ids,
                                                                  ep_route        rt,
                                                                  uint32_t *      status,
                                                                  int32_t         rows,
                                                                  int32_t         task_rows,
                                                                  ep_lists        lists) {
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
            rt.counters[2] = 0;
            rt.counters[3] = 0;
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
    // APF2: the ladder's grouped task lists (lists.tasks[i] non-null): every expert's pairs in ceil(n / rows[i]) runs
    // of near-equal size, expert-major; counters[2 + i] their count (0 without the list)
    int32_t n_ltasks[2] = { 0, 0 };
#pragma unroll
    for (int32_t li = 0; li < 2; ++li) {
        const int32_t lr   = lists.rows[li];
        const int32_t ng   = lists.tasks[li] != nullptr && tid < n_expert ? (tot_e + lr - 1) / lr : 0;
        const int32_t goff = block_scan(ng, s.wsum, &n_ltasks[li]);
        for (int32_t k = 0; k < ng; ++k) {
            const int32_t q = tot_e / ng, rem = tot_e - q * ng;
            const int32_t first = off_e + k * q + (k < rem ? k : rem);
            *reinterpret_cast<int4 *>(lists.tasks[li] + 4 * (goff + k)) =
                make_int4(tid, first, q + (k < rem ? 1 : 0), 0);
        }
    }
    if (tid == 0) {
        rt.offsets[n_expert] = all;
        rt.counters[0] = n_tasks;
        rt.counters[1] = 0;
        rt.counters[2] = n_ltasks[0];
        rt.counters[3] = n_ltasks[1];
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
bool lists_ok(const ep_lists & l) {
    for (int32_t i = 0; i < 2; ++i) {
        if (l.tasks[i] != nullptr && (!aligned16(l.tasks[i]) || l.rows[i] < 8 || l.rows[i] > ep_ladder_max_rows ||
                                      l.rows[i] % 8 != 0)) {
            return false;
        }
    }
    return true;
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

// the resident and cache instances' opt-ins of format F's gate/up on the current device (one bit a device)
template <bool X, fmt F> bool gate_up_attr() {
    static std::atomic<uint32_t> done{ 0u };
    return smem_per_device(done, reinterpret_cast<const void *>(ep_gate_up_kernel<X, F>), gu_smem(F));
}

template <bool X, fmt F> bool launch_gate_up_x(const ep_gate_up_args & a, cudaStream_t st) {
    constexpr int32_t smem = gu_smem(F);
    if (!gate_up_attr<X, F>()) {
        return false;
    }
    const int32_t grid = X && a.task_off != nullptr && a.grid > 0 ? std::min(a.grid, ep_max_tasks) : ep_max_tasks;
    ep_gate_up_kernel<X, F><<<grid, nt_threads, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

// ME3: the cache's instantiation when the arguments carry its table, else the resident one
template <fmt F> bool launch_gate_up(const ep_gate_up_args & a, cudaStream_t st) {
    return a.wt != nullptr ? launch_gate_up_x<true, F>(a, st) : launch_gate_up_x<false, F>(a, st);
}

template <bool X, fmt F> bool down_attr() {
    static std::atomic<uint32_t> done{ 0u };  // PB1: per device, as above
    return smem_per_device(done, reinterpret_cast<const void *>(ep_down_kernel<X, F>), (n_ff_card / ks) * a_tile);
}

template <bool X, fmt F> bool launch_down_x(const ep_down_args & a, cudaStream_t st) {
    constexpr int32_t smem = (n_ff_card / ks) * a_tile; // every step's GLU tile
    if (!down_attr<X, F>()) {
        return false;
    }
    const int32_t grid = X && a.task_off != nullptr && a.grid > 0 ? std::min(a.grid, ep_max_tasks) : ep_max_tasks;
    ep_down_kernel<X, F><<<grid, dn_threads, smem, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

// ME3: as launch_gate_up
template <fmt F> bool launch_down(const ep_down_args & a, cudaStream_t st) {
    return a.wt != nullptr ? launch_down_x<true, F>(a, st) : launch_down_x<false, F>(a, st);
}

} // namespace

bool ep_prompt_attributes() {
    return gate_up_attr<false, fmt::iq2_xxs>() && gate_up_attr<false, fmt::iq2_xs>() &&
           gate_up_attr<false, fmt::iq2_s>() && gate_up_attr<false, fmt::iq3_xxs>() &&
           gate_up_attr<false, fmt::iq3_s>() && gate_up_attr<false, fmt::q4_k>() && gate_up_attr<false, fmt::q5_k>() &&
           down_attr<false, fmt::q2_0>() && down_attr<false, fmt::iq4_nl>() && down_attr<false, fmt::q5_1>() &&
           down_attr<false, fmt::q8_0>();
}

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

// ME4: each expert's first task by a binary search of ep_group's tasks (sorted by expert); off[n_expert] the count
__global__ void __launch_bounds__(544)
    ep_task_off_kernel(const int32_t * tasks, const int32_t * counters, int32_t * off) {
    const int32_t e = static_cast<int32_t>(threadIdx.x);
    if (e > n_expert) {
        return;
    }
    int32_t lo = 0, hi = *reinterpret_cast<const volatile int32_t *>(counters);
    while (lo < hi) {
        const int32_t mid = (lo + hi) >> 1;
        if (tasks[4 * mid] < e) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    off[e] = lo;
}

bool ep_task_offsets(const ep_route & r, int32_t * task_off, void * stream) {
    if (!route_ok(r) || !aligned4(task_off) || stream == nullptr) {
        return false;
    }
    ep_task_off_kernel<<<1, 544, 0, static_cast<cudaStream_t>(stream)>>>(r.tasks, r.counters, task_off);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool ep_group(int32_t          rows,
              int32_t          task_rows,
              const int32_t *  ids,
              const ep_route & r,
              uint32_t *       status,
              void *           stream,
              const ep_lists & lists) {
    if (rows < 1 || rows > ep_max_rows || task_rows < 8 || task_rows > ep_pass_rows || task_rows % 8 != 0 ||
        !aligned4(ids) || !route_ok(r) || stream == nullptr || (status != nullptr && !aligned4(status)) ||
        !lists_ok(lists)) {
        return false;
    }
    ep_group_kernel<<<1, grp_threads, 0, static_cast<cudaStream_t>(stream)>>>(ids, r, status, rows, task_rows,
                                                                              lists);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool ep_gate_up(fmt f, int32_t rows, const ep_gate_up_args & a, void * stream) {
    if (rows < 1 || rows > ep_max_rows || !aligned16(a.w) || !aligned16(a.grids) || !aligned16(a.x) ||
        !route_ok(a.route) || !aligned4(a.glu) || stream == nullptr ||
        (a.glu_f32 != nullptr && !aligned16(a.glu_f32)) || (reinterpret_cast<uintptr_t>(a.wt) & 7u) != 0 ||
        (a.task_off != nullptr && (a.wt == nullptr || !aligned4(a.task_off) || a.e_lo < 0 || a.e_lo > a.e_hi ||
                                   a.e_hi > n_expert || a.grid < 0))) {
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
        case fmt::q4_k:
            return launch_gate_up<fmt::q4_k>(a, st);
        case fmt::q5_k:
            return launch_gate_up<fmt::q5_k>(a, st);
        default:
            return false;
    }
}

bool ep_down(fmt f, int32_t rows, const ep_down_args & a, void * stream) {
    if (rows < 1 || rows > ep_max_rows || !aligned16(a.w) || !aligned16(a.grids) || !aligned16(a.glu) ||
        !route_ok(a.route) || (a.part16 != nullptr ? !aligned16(a.part16) || a.err == nullptr : !aligned16(a.part)) ||
        stream == nullptr || (reinterpret_cast<uintptr_t>(a.wt) & 7u) != 0 ||
        (a.task_off != nullptr && (a.wt == nullptr || !aligned4(a.task_off) || a.e_lo < 0 || a.e_lo > a.e_hi ||
                                   a.e_hi > n_expert || a.grid < 0))) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (f) {
        case fmt::q2_0:
            return launch_down<fmt::q2_0>(a, st);
        case fmt::iq4_nl:
            return launch_down<fmt::iq4_nl>(a, st);
        case fmt::q5_1:
            return launch_down<fmt::q5_1>(a, st);
        case fmt::q8_0:
            return launch_down<fmt::q8_0>(a, st);
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
