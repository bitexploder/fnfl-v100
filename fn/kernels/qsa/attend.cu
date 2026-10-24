#include "fn/kernels/qsa/attend.h"

#include "fn/kernels/act/act_warp.cuh"
#include "fn/kernels/qsa/kv_q8.cuh"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/shapes.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace fn::qsa::attn {
namespace {

constexpr int32_t threads = 256;
constexpr int32_t n_warp = threads / 32;
constexpr int32_t n_cg = n_warp;           // cell groups a CTA: one warp each, all 6 of the unit's heads
constexpr int32_t in_flight = 4;           // K or V cells a warp has in flight (a 16-byte load a lane each)
constexpr int32_t chunk_pad = 88;          // chunk rounded up to a multiple of 8 (one column a cell group)
constexpr int32_t n_task_row = n_head + 1; // a row's prepare tasks: 12 query heads, then its new cell's k and v
constexpr int32_t n_stamp = 10;            // diagnostic stamps a CTA
constexpr int32_t n_sm = 72;               // the PG500-216 (GUIDE section 1)
constexpr uint32_t poll_limit = 1u << 22;  // about 0.3 s of 64 ns sleeps: a timeout, never a wait in a valid launch

static_assert(head_dim == 32 * 8, "a lane holds 8 dims of a K, V or q row (16 bytes of BF16, 32 of F32)");
static_assert(in_flight == 4, "a pass 1 batch is the 4 cells in flight: a cell's slot k % 4 is a compile-time index");
static_assert(n_head % group == 0 && chunk <= chunk_pad && chunk_pad % n_cg == 0, "unit shape");
static_assert(max_chunks <= 32, "the finish holds a row's chunk statistics one a lane");
static_assert(fn::rms_eps == 1e-6f, "the pinned eps (attention.layer_norm_rms_epsilon)");

struct kargs {
    const float * qfull;
    const float * kv_k;
    const float * kv_v;
    const float * q_norm;
    const float * k_norm;
    const uint32_t * body;
    const int32_t * pos;
    uint16_t * k_cache;
    uint16_t * v_cache;
    float * k_f32;
    uint64_t * stamps;
    float * q;
    float * acc;
    float * ml;
    uint32_t * ready;
    uint32_t * error;
    int32_t rows;
    const int32_t * depth;   // device: cells before the rows (FC-RD1)
    const int32_t * offsets; // device: the body's byte offsets [B + 1]
    int32_t max_cells;       // the caches' cells
    float theta_scale;
    uint32_t mz; // an opaque zero (take)
    float * out;
    int8_t * codes;
    float * scales;
    const int64_t * shift; // ME3: attend_kernel<true>'s device word (args::shift), after every field the resident reads
    // KV8 (attend_kernel<false, true>): the layer's FP16 scales [cell][kv8::scale_stride] beside the int8 codes at
    // k_cache and v_cache (kv_q8.cuh), after every field the BF16 launch reads
    uint16_t * k_scale;
    uint16_t * v_scale;
};

// the launch's work at the round's depth, formed by every thread at the start (FC-RD1: the depth is read on the device)
struct plan {
    int32_t depth;
    int32_t n_units;
    int32_t nch[max_rows];       // chunks of row t
    int32_t body_word[max_rows]; // row t's body, in words from body (its offset, as select or X-SELECT wrote it)
    int32_t cum[max_rows + 1];   // units before row t's, in chunks
};

__host__ __device__ __forceinline__ int32_t sel_cells(int32_t q) {
    const int32_t n_vis = (q + 1) / 4;
    return 4 * (n_vis < 512 ? n_vis : 512) + (q + 1) % 4;
}

// thread 0 forms the plan in shared memory (rows are indexed at run time there, so no stack frame); the CTA's barrier
// follows in the caller
__device__ __forceinline__ void plan_of(const kargs & a, plan & p) {
    p.depth = *a.depth;
    int32_t c = 0;
    for (int32_t t = 0; t < a.rows; ++t) {
        p.body_word[t] = a.offsets[t] / 4;
        p.cum[t] = c;
        p.nch[t] = (sel_cells(p.depth + t) + chunk - 1) / chunk;
        c += p.nch[t];
    }
    p.cum[a.rows] = c;
    p.n_units = 2 * c;
}

struct smem_t {
    int32_t cell[chunk_pad];              // the unit's cells in selection order
    float q[group][head_dim];             // the unit's 6 prepared queries (scaled by 1/16 when read)
    float s[group][chunk_pad];            // scores (scaled by 1/16)
    float pt[chunk_pad][8];               // probabilities [cell][head], heads 6 and 7 unused
    float ml[group][2];                   // the unit's (max, sum) a head
    float red[n_cg / 2][group][head_dim]; // the cell groups' accumulators, summed as a tree
};

// ---- small helpers ----

__device__ __forceinline__ void widen(const uint4 w, float (&x)[8]) {
    const uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[2 * i] = __uint_as_float(__byte_perm(u[i], 0u, 0x1054)); // low half << 16, a PRMT
        x[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
}

// round to nearest even (finite inputs)
__device__ __forceinline__ uint32_t bf16_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ uint4 pack_bf16(const float (&x)[8]) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = bf16_bits(x[2 * i]) | (bf16_bits(x[2 * i + 1]) << 16);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// The KV-union read: read once a round, so evict-first in L1 and L2 (__ldcs; XB1, from CR3) and the round's code stays
// in L2. A cell at or past depth (written by this launch's prepare) is loaded only after the acquire, and no earlier load
// of this launch touches it, so no stale L1 line can hold it, as with the .cg form before.
__device__ __forceinline__ uint4 ld_kv(const uint16_t * p) {
    return __ldcs(reinterpret_cast<const uint4 *>(p));
}

// a diagnostic stamp (qa2-bench diag): thread 0's globaltimer at phase k of the CTA's first unit
__device__ __forceinline__ void stamp(const kargs & a, int32_t k, bool first) {
    if (a.stamps != nullptr && first && threadIdx.x == 0) {
        uint64_t t = 0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        a.stamps[blockIdx.x * n_stamp + k] = t;
    }
}

// ---- prepare: one warp, one task ----

// The F32 rms norm and IMROPE of one 256-wide row, a lane holding dims 8 lane .. 8 lane + 7 (pinned: build_norm
// LLM_NORM_RMS, then ggml_rope_multi with GGML_ROPE_TYPE_IMROPE: ggml_mrope_cache_init and rotate_pairs).
__device__ void norm_rope(float (&y)[8], const float * src, const float * w, const int32_t * pos, int32_t rows,
                          int32_t t, float theta_scale, int32_t lane) {
    const float4 a = __ldg(reinterpret_cast<const float4 *>(src) + 2 * lane);
    const float4 b = __ldg(reinterpret_cast<const float4 *>(src) + 2 * lane + 1);
    const float x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float ss = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        ss = fmaf(x[e], x[e], ss);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    const float r = 1.0f / sqrtf(ss * (1.0f / head_dim) + fn::rms_eps);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        y[e] = x[e] * r * __ldg(w + 8 * lane + e);
    }
    // pair i = lane: the sector rule picks a plane; its theta is multiplied i times by theta_scale (the F32 chain; each
    // plane's chain is independent, so only the chosen one is formed), predicated and unrolled
    const int sector = lane; // (i0 / 2) % 32 for the 32 pairs of n_rot = 64
    const int plane = sector % 3 == 1 && sector < 3 * 11   ? 1
                      : sector % 3 == 2 && sector < 3 * 10 ? 2
                      : sector % 3 == 0 && sector < 3 * 11 ? 0
                                                           : 3;
    float theta = static_cast<float>(__ldg(pos + plane * rows + t));
#pragma unroll
    for (int i = 0; i < 31; ++i) {
        theta = i < lane ? theta * theta_scale : theta;
    }
    float s = 0.0f, c = 0.0f;
    sincosf(theta, &s, &c);
    // dims 0..31 are in lanes 0..3, their partners 32..63 in lanes 4..7
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float ce = __shfl_sync(0xffffffffu, c, 8 * (lane & 3) + e);
        const float se = __shfl_sync(0xffffffffu, s, 8 * (lane & 3) + e);
        const float other = __shfl_xor_sync(0xffffffffu, y[e], 4);
        if (lane < 4) {
            y[e] = y[e] * ce - other * se; // x0 cos - x1 sin
        } else if (lane < 8) {
            y[e] = other * se + y[e] * ce; // x0 sin + x1 cos
        }
    }
}

// KV8: a row's int8 codes, cell c at byte c * 256 of the K or V array (kv_q8.cuh)
__device__ __forceinline__ int8_t * codes_row(uint16_t * base, int64_t cell) {
    // reason: in q8 the cache array holds int8 codes; its pointer is the class's untyped part
    return reinterpret_cast<int8_t *>(base) + cell * head_dim;
}

// KV8: the lane's 8 values of the new cell's K or V row, quantized (kv8::quantize8: every lane takes part)
__device__ __forceinline__ void put_q8(const float (&x)[8], uint16_t * cache, uint16_t * scale, int64_t cell,
                                       int32_t lane) {
    uint2 codes;
    uint16_t sbits = 0;
    fn::qsa::kv8::quantize8(x, codes, sbits);
    reinterpret_cast<uint2 *>(codes_row(cache, cell))[lane] = codes;
    if ((lane & 7) == 0) {
        scale[cell * fn::qsa::kv8::scale_stride + lane / 8] = sbits;
    }
}

template <bool Q8> __device__ void prepare(const kargs & a, const plan & pl, int32_t task, int32_t lane) {
    const int32_t t = task / n_task_row, r = task % n_task_row;
    const bool is_q = r < n_head;
    float y[8];
    norm_rope(y, is_q ? a.qfull + (t * n_head + r) * q_stride : a.kv_k + t * head_dim, is_q ? a.q_norm : a.k_norm,
              a.pos, a.rows, t, a.theta_scale, lane);
    if (is_q) {
        float4 * dst = reinterpret_cast<float4 *>(a.q + (t * n_head + r) * head_dim) + 2 * lane;
        dst[0] = make_float4(y[0], y[1], y[2], y[3]);
        dst[1] = make_float4(y[4], y[5], y[6], y[7]);
    } else {
        const int64_t cell = static_cast<int64_t>(pl.depth) + t;
        if constexpr (Q8) {
            put_q8(y, a.k_cache, a.k_scale, cell, lane);
        } else {
            reinterpret_cast<uint4 *>(a.k_cache + cell * head_dim)[lane] = pack_bf16(y);
        }
        if (a.k_f32 != nullptr) {
            float4 * dst = reinterpret_cast<float4 *>(a.k_f32 + t * head_dim) + 2 * lane;
            dst[0] = make_float4(y[0], y[1], y[2], y[3]);
            dst[1] = make_float4(y[4], y[5], y[6], y[7]);
        }
        const float4 va = __ldg(reinterpret_cast<const float4 *>(a.kv_v + t * head_dim) + 2 * lane);
        const float4 vb = __ldg(reinterpret_cast<const float4 *>(a.kv_v + t * head_dim) + 2 * lane + 1);
        const float v[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
        if constexpr (Q8) {
            put_q8(v, a.v_cache, a.v_scale, cell, lane);
        } else {
            reinterpret_cast<uint4 *>(a.v_cache + cell * head_dim)[lane] = pack_bf16(v);
        }
    }
    // publication: each writing lane fences its own stores, then one lane counts the task
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        atomicAdd(a.ready, 1u);
        if (a.stamps != nullptr) {
            uint64_t t = 0;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            a.stamps[gridDim.x * n_stamp + task] = t;
        }
    }
}

// thread 0 polls the counter with acquire loads (gpu scope) until every prepare task is counted, and the CTA passes a
// barrier (the acquire). An acquire load, not a fence: a fence would wait for thread 0's own K loads in flight.
__device__ void acquire(const kargs & a) {
    if (threadIdx.x == 0) {
        const uint32_t target = static_cast<uint32_t>(n_task_row * a.rows);
        uint32_t n = 0, v = 0;
        for (;;) {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a.ready) : "memory");
            if (v >= target) {
                break;
            }
            if (++n == poll_limit) {
                atomicOr(a.error, 1u);
                break;
            }
            __nanosleep(64);
        }
    }
    __syncthreads();
}

// ---- attend: one unit ----

// a warp's cell loads: cell k of its group is the unit's cell w + 8 k; a lane loads its 16 bytes (8 dims)
struct cells {
    const uint16_t * cache;
    const int32_t * cell; // smem: the unit's cells
    int32_t w, lane, n_w, depth;

    // defer: before the acquire, a cell at or past depth is not loaded yet (deferred = true)
    // dep: an opaque zero carried into the address, so the load is not issued before dep's value exists
    __device__ __forceinline__ uint4 load(int32_t k, bool defer, bool & deferred, uint32_t dep = 0) const {
        const int32_t c = k < n_w ? cell[w + n_cg * k] : -1;
        deferred = defer && c >= depth;
        if (c < 0 || deferred) {
            return make_uint4(0, 0, 0, 0);
        }
        return ld_kv(cache + static_cast<int64_t>(c) * head_dim + 8 * lane + dep);
    }
};

// take (FINDINGS R4): a slot's words leave it through an OR with the opaque zero mz, so its registers are free for the
// refill, whose address carries a zero made from the taken words (issued once they have arrived). Without it ptxas
// lands the refill in new registers and copies it back at the loop end, and the copy waits for the refill to arrive.
__device__ __forceinline__ uint4 take(const uint4 r, uint32_t mz, uint32_t & dep) {
    const uint4 v = make_uint4(r.x | mz, r.y | mz, r.z | mz, r.w | mz);
    dep = (v.x ^ v.y ^ v.z ^ v.w) & mz;
    return v;
}

// KV8: a lane's slot of one cell in q8: its 8 codes (two words) and its group's scale bits
struct kv8w {
    uint32_t c0, c1, s;
};

// KV8: the q8 form of cells: a lane loads its 8 codes (8 bytes) and its group's FP16 scale, evict-first as ld_kv
struct cells8 {
    const int8_t * codes;
    const uint16_t * scale;
    const int32_t * cell; // smem: the unit's cells
    int32_t w, lane, n_w, depth;

    __device__ __forceinline__ kv8w load(int32_t k, bool defer, bool & deferred, uint32_t dep = 0) const {
        const int32_t c = k < n_w ? cell[w + n_cg * k] : -1;
        deferred = defer && c >= depth;
        if (c < 0 || deferred) {
            return kv8w{0u, 0u, 0u};
        }
        const uint2 v =
            __ldcs(reinterpret_cast<const uint2 *>(codes + static_cast<int64_t>(c) * head_dim + 8 * lane + dep));
        const uint32_t s = __ldcs(scale + static_cast<int64_t>(c) * fn::qsa::kv8::scale_stride + lane / 8 + dep);
        return kv8w{v.x, v.y, s};
    }
};

// take (R4) in q8: the codes and the scale leave the slot through mz, and the zero dep made from all three words is
// carried into both refill loads' addresses
__device__ __forceinline__ kv8w take(const kv8w r, uint32_t mz, uint32_t & dep) {
    const kv8w v = {r.c0 | mz, r.c1 | mz, r.s | mz};
    dep = (v.c0 ^ v.c1 ^ v.s) & mz;
    return v;
}

// q8 widened: float(code) * s, exact in F32 (kv_q8.cuh decode4)
__device__ __forceinline__ void widen(const kv8w w, float (&x)[8]) {
    const float s = fn::qsa::kv8::scale_value(w.s);
    fn::qsa::kv8::decode4(w.c0, s, x);
    fn::qsa::kv8::decode4(w.c1, s, x + 4);
}

// the K or V cell loads of a unit in the launch's format
template <bool Q8> struct cells_of {
    using type = cells;

    __device__ __forceinline__ static cells make(const kargs & a, bool v, const int32_t * cell, int32_t w, int32_t lane,
                                                 int32_t n_w, int32_t depth) {
        return {v ? a.v_cache : a.k_cache, cell, w, lane, n_w, depth};
    }
};

template <> struct cells_of<true> {
    using type = cells8;

    __device__ __forceinline__ static cells8 make(const kargs & a, bool v, const int32_t * cell, int32_t w,
                                                  int32_t lane, int32_t n_w, int32_t depth) {
        // reason: in q8 the cache array holds int8 codes
        return {reinterpret_cast<const int8_t *>(v ? a.v_cache : a.k_cache),
                v ? a.v_scale : a.k_scale,
                cell,
                w,
                lane,
                n_w,
                depth};
    }
};

// the 24 partial dots of 4 cells x 6 heads summed over the warp: a reduce-scatter (xor 16, 8, 4), then a butterfly
// (xor 2, 1); lanes 4 o .. 4 o + 3 end with entries 3 o .. 3 o + 2 (entry = cell 6 + head), o = 4 b4 + 2 b3 + b2
__device__ __forceinline__ void reduce24(const float (&p)[24], float (&q3)[3], int32_t lane) {
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float q12[12], q6[6];
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        const float send = b4 ? p[i] : p[i + 12];
        q12[i] = (b4 ? p[i + 12] : p[i]) + __shfl_xor_sync(0xffffffffu, send, 16);
    }
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const float send = b3 ? q12[i] : q12[i + 6];
        q6[i] = (b3 ? q12[i + 6] : q12[i]) + __shfl_xor_sync(0xffffffffu, send, 8);
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float send = b2 ? q6[i] : q6[i + 3];
        q3[i] = (b2 ? q6[i + 3] : q6[i]) + __shfl_xor_sync(0xffffffffu, send, 4);
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        q3[i] += __shfl_xor_sync(0xffffffffu, q3[i], 2);
        q3[i] += __shfl_xor_sync(0xffffffffu, q3[i], 1);
    }
}

template <bool Q8> __device__ void run_unit(const kargs & a, const plan & pl, smem_t & sm, int32_t u, bool first) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x), w = tid / 32, lane = tid % 32;
    // unit u: row t, chunk j, head group G (units ordered by row, then chunk, then group)
    int32_t t = 0;
    while (t + 1 < a.rows && 2 * pl.cum[t + 1] <= u) {
        ++t;
    }
    const int32_t j = u / 2 - pl.cum[t], G = u % 2;
    const uint32_t * body = a.body + pl.body_word[t];
    const int32_t K = static_cast<int32_t>(__ldg(body)), tail = static_cast<int32_t>(__ldg(body + 1));
    const int32_t n_sel = 4 * K + tail, c0 = j * chunk, nc = min(chunk, n_sel - c0);
    if (tid < nc) {
        const int32_t s = c0 + tid;
        sm.cell[tid] = s < 4 * K ? 4 * static_cast<int32_t>(__ldg(body + 2 + s / 4)) + s % 4
                                 : static_cast<int32_t>(__ldg(body + 2 + K + (s - 4 * K)));
    }
    __syncthreads();
    stamp(a, 1, first);
    const int32_t n_w = nc > w ? (nc - w + n_cg - 1) / n_cg : 0;
    using cells_t = typename cells_of<Q8>::type;
    const cells_t ks = cells_of<Q8>::make(a, false, sm.cell, w, lane, n_w, pl.depth);
    const cells_t vs = cells_of<Q8>::make(a, true, sm.cell, w, lane, n_w, pl.depth);
    using slot_t = decltype(ks.load(0, false, std::declval<bool &>()));

    // pass 1: scores. Cells k + 1 .. k + 4 in flight (slot k % 4, a compile-time index: R12); before the first
    // unit's acquire only cells below depth are loaded.
    slot_t r[in_flight];
    bool deferred[in_flight];
#pragma unroll
    for (int i = 0; i < in_flight; ++i) {
        r[i] = ks.load(i, first, deferred[i]);
    }
    if (first) {
        acquire(a);
        stamp(a, 2, first);
#pragma unroll
        for (int i = 0; i < in_flight; ++i) {
            if (deferred[i]) {
                bool unused = false;
                r[i] = ks.load(i, false, unused);
            }
        }
    }
    // the unit's 6 queries, read once a CTA (coalesced: 6 KB as 384 16-byte loads), then by every warp from shared memory
    const float4 * qsrc = reinterpret_cast<const float4 *>(a.q + (t * n_head + G * group) * head_dim);
    for (int32_t i = tid; i < group * head_dim / 4; i += threads) {
        reinterpret_cast<float4 *>(&sm.q[0][0])[i] = __ldcg(qsrc + i);
    }
    __syncthreads();
    float q[group][8];
#pragma unroll
    for (int h = 0; h < group; ++h) {
        const float4 x0 = *reinterpret_cast<const float4 *>(&sm.q[h][8 * lane]);
        const float4 x1 = *reinterpret_cast<const float4 *>(&sm.q[h][8 * lane + 4]);
        const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            q[h][e] = x[e] * 0.0625f; // 1 / sqrt(256), exact
        }
    }
    if (a.stamps != nullptr && first) {
        asm volatile("" ::"f"(q[0][0]), "f"(q[group - 1][7])); // the stamp waits for the q loads
        stamp(a, 7, first);
    }
    bool unused = false;
    for (int32_t kb = 0; kb < n_w; kb += in_flight) {
        float p[in_flight * group]; // [cell of the batch][head]
#pragma unroll
        for (int cc = 0; cc < in_flight; ++cc) {
            const int32_t k = kb + cc;
            if (k < n_w) {
                uint32_t dep = 0;
                const slot_t v = take(r[cc], a.mz, dep);
                r[cc] = ks.load(k + in_flight, false, unused, dep);
                float x[8];
                widen(v, x);
#pragma unroll
                for (int h = 0; h < group; ++h) {
                    float d = 0.0f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        d = fmaf(q[h][e], x[e], d);
                    }
                    p[cc * group + h] = d;
                }
            } else {
#pragma unroll
                for (int h = 0; h < group; ++h) {
                    p[cc * group + h] = 0.0f;
                }
            }
        }
        float q3[3];
        reduce24(p, q3, lane);
        const int32_t o = lane / 4, cb = (3 * o) / group, h0 = (3 * o) % group, kg = kb + cb;
        if ((lane & 3) == 0 && kg < n_w) {
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                sm.s[h0 + i][w + n_cg * kg] = q3[i];
            }
        }
    }
    stamp(a, 3, first);
    // V's first cells go out before the softmax
#pragma unroll
    for (int i = 0; i < in_flight; ++i) {
        r[i] = vs.load(i, false, unused);
    }
    __syncthreads();
    if (w < group) {
        const int32_t h = w;
        float m = -INFINITY;
        for (int32_t i = lane; i < nc; i += 32) {
            m = fmaxf(m, sm.s[h][i]);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        }
        float l = 0.0f;
        for (int32_t i = lane; i < nc; i += 32) {
            const float e = expf(sm.s[h][i] - m);
            sm.pt[i][h] = e;
            l += e;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            l += __shfl_xor_sync(0xffffffffu, l, o);
        }
        if (lane == 0) {
            sm.ml[h][0] = m;
            sm.ml[h][1] = l;
        }
    }
    __syncthreads();
    stamp(a, 4, first);

    // pass 2: P V, a cell group's cells in order
    float acc[group][8];
#pragma unroll
    for (int h = 0; h < group; ++h) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            acc[h][e] = 0.0f;
        }
    }
    for (int32_t kb = 0; kb < n_w; kb += in_flight) {
#pragma unroll
        for (int cc = 0; cc < in_flight; ++cc) {
            const int32_t k = kb + cc;
            if (k < n_w) {
                uint32_t dep = 0;
                const slot_t tv = take(r[cc], a.mz, dep);
                r[cc] = vs.load(k + in_flight, false, unused, dep);
                float v[8];
                widen(tv, v);
                const float4 p0 = *reinterpret_cast<const float4 *>(&sm.pt[w + n_cg * k][0]);
                const float2 p1 = *reinterpret_cast<const float2 *>(&sm.pt[w + n_cg * k][4]);
                const float pp[group] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y};
#pragma unroll
                for (int h = 0; h < group; ++h) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        acc[h][e] = fmaf(pp[h], v[e], acc[h][e]);
                    }
                }
            }
        }
    }
    stamp(a, 5, first);
    // the cell groups' sums as a fixed tree: ((g0 + g4) + (g2 + g6)) + ((g1 + g5) + (g3 + g7))
#pragma unroll
    for (int half = n_cg / 2; half > 0; half /= 2) {
        if (w >= half && w < 2 * half) {
#pragma unroll
            for (int h = 0; h < group; ++h) {
                float4 * dst = reinterpret_cast<float4 *>(&sm.red[w - half][h][8 * lane]);
                dst[0] = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
                dst[1] = make_float4(acc[h][4], acc[h][5], acc[h][6], acc[h][7]);
            }
        }
        __syncthreads();
        if (w < half) {
#pragma unroll
            for (int h = 0; h < group; ++h) {
                const float4 * src = reinterpret_cast<const float4 *>(&sm.red[w][h][8 * lane]);
                const float4 x0 = src[0], x1 = src[1];
                const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[h][e] += x[e];
                }
            }
        }
        __syncthreads();
    }
    if (w == 0) {
#pragma unroll
        for (int h = 0; h < group; ++h) {
            float4 * dst =
                reinterpret_cast<float4 *>(a.acc + ((t * max_chunks + j) * n_head + G * group + h) * head_dim) +
                2 * lane;
            dst[0] = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
            dst[1] = make_float4(acc[h][4], acc[h][5], acc[h][6], acc[h][7]);
        }
        if (lane < group) {
            float2 * dst = reinterpret_cast<float2 *>(a.ml + ((t * max_chunks + j) * n_head + G * group + lane) * 2);
            *dst = make_float2(sm.ml[lane][0], sm.ml[lane][1]);
        }
    }
    __syncthreads(); // the next unit reuses the shared memory
    stamp(a, 6, first);
}

// the finish of one (row t, head h, values 32 q .. 32 q + 31), a warp: the row's chunks folded in chunk order, the gate,
// the F32 output and its activation-format run. The partials were written in this launch by other CTAs: L2 loads.
__device__ void finish(const kargs & a, const plan & pl, int32_t fu, int32_t lane) {
    const int32_t t = fu / (n_head * 8), h = fu / 8 % n_head, q = fu % 8;
    const int32_t n = pl.nch[t];
    float mj = -INFINITY, lj = 0.0f;
    if (lane < n) {
        const float2 v = __ldcg(reinterpret_cast<const float2 *>(a.ml + ((t * max_chunks + lane) * n_head + h) * 2));
        mj = v.x;
        lj = v.y;
    }
    float av[max_chunks];
#pragma unroll
    for (int jj = 0; jj < max_chunks; ++jj) {
        av[jj] = jj < n ? __ldcg(a.acc + ((t * max_chunks + jj) * n_head + h) * head_dim + 32 * q + lane) : 0.0f;
    }
    float M = mj;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    }
    const float f = lane < n ? expf(mj - M) : 0.0f;
    float L = 0.0f, A = 0.0f;
#pragma unroll
    for (int jj = 0; jj < max_chunks; ++jj) {
        const float fj = __shfl_sync(0xffffffffu, f, jj), lv = __shfl_sync(0xffffffffu, lj, jj);
        if (jj < n) {
            L = fmaf(lv, fj, L);
            A = fmaf(av[jj], fj, A);
        }
    }
    const float g = __ldg(a.qfull + (t * n_head + h) * q_stride + head_dim + 32 * q + lane);
    const float y = A / L * (1.0f / (1.0f + expf(-g)));
    a.out[t * out_width + h * head_dim + 32 * q + lane] = y;
    fn::act::put_run_warp(y, lane, a.codes + t * out_width + h * head_dim + 32 * q,
                          a.scales + (t * (out_width / 32) + h * 8 + q) * 2);
}

template <bool Q8> __device__ __forceinline__ void attend_body(const kargs & a) {
    __shared__ smem_t sm;
    const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32, lane = static_cast<int32_t>(threadIdx.x) % 32;
    const int32_t gw = static_cast<int32_t>(blockIdx.x) + static_cast<int32_t>(gridDim.x) * warp;
    stamp(a, 0, true);
    __shared__ plan pl;
    if (threadIdx.x == 0) {
        plan_of(a, pl);
    }
    __syncthreads();
    if (pl.depth < 0 || pl.depth > a.max_cells - a.rows) {
        // a depth the runtime refuses (FC-RD1): every CTA leaves before the grid barrier, nothing is written
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            atomicOr(a.error, 2u);
        }
        return;
    }
    if (gw < n_task_row * a.rows) {
        prepare<Q8>(a, pl, gw, lane);
    }
    bool first = true;
    for (int32_t u = static_cast<int32_t>(blockIdx.x); u < pl.n_units; u += static_cast<int32_t>(gridDim.x)) {
        run_unit<Q8>(a, pl, sm, u, first);
        first = false;
    }
    // every unit's partials are written and every CTA has acquired (the grid barrier orders both); the counter is free
    cooperative_groups::this_grid().sync();
    stamp(a, 8, true);
    if (gw == 0 && lane == 0) {
        *a.ready = 0;
    }
    for (int32_t fu = gw; fu < a.rows * n_head * 8; fu += static_cast<int32_t>(gridDim.x) * n_warp) {
        finish(a, pl, fu, lane);
    }
    stamp(a, 9, true);
}

// H false: the resident caches (QA2's launch, its body unchanged). ME3, H true: the host-backed KV's graphs: the K
// and V pointers are the window's base and the device word the cells it starts at (kv_tier.h), read once a thread.
// KV8, Q8 true: the resident caches in the 8-bit format (kv_q8.cuh; not with H)
template <bool H, bool Q8> __global__ void __launch_bounds__(threads, 2) attend_kernel(const kargs ka) {
    static_assert(!(H && Q8), "KV8: the host-backed KV stays BF16");
    if constexpr (H) {
        kargs a = ka;
        const int64_t s = *ka.shift * head_dim;
        a.k_cache -= s;
        a.v_cache -= s;
        attend_body<false>(a);
    } else {
        attend_body<Q8>(ka);
    }
}

// the launch's instantiation: 0 resident BF16, 1 host-backed (ME3), 2 resident q8 (KV8)
const void * kernel_of(int32_t variant) {
    return variant == 1   ? reinterpret_cast<const void *>(attend_kernel<true, false>)
           : variant == 2 ? reinterpret_cast<const void *>(attend_kernel<false, true>)
                          : reinterpret_cast<const void *>(attend_kernel<false, false>);
}

bool aligned(const void * p, uintptr_t n) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % n == 0;
}

int32_t ctas_per_sm(int32_t device, int32_t variant = 0) {
    static int32_t cache[3][16] = {}; // per kernel per device (R14)
    if (device < 0 || device >= 16 || variant < 0 || variant > 2) {
        return 0;
    }
    if (cache[variant][device] == 0) {
        int n = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel_of(variant), threads, 0) != cudaSuccess) {
            return 0;
        }
        cache[variant][device] = n;
    }
    return cache[variant][device];
}

} // namespace

size_t workspace_bytes() {
    return sizeof(float) * (size_t(max_rows) * n_head * head_dim + size_t(max_rows) * max_chunks * n_head * head_dim +
                            size_t(max_rows) * max_chunks * n_head * 2) +
           256;
}

workspace carve(void * base) {
    auto * p = static_cast<uint8_t *>(base);
    workspace w = {};
    w.q = reinterpret_cast<float *>(p);
    p += sizeof(float) * max_rows * n_head * head_dim;
    w.acc = reinterpret_cast<float *>(p);
    p += sizeof(float) * max_rows * max_chunks * n_head * head_dim;
    w.ml = reinterpret_cast<float *>(p);
    p += sizeof(float) * max_rows * max_chunks * n_head * 2;
    w.ready = reinterpret_cast<uint32_t *>(p);
    w.error = reinterpret_cast<uint32_t *>(p + 128);
    return w;
}

float rope_theta_scale() {
    return powf(10000000.0f, -2.0f / static_cast<float>(fn::qsa::n_rot));
}

int32_t selected_cells(int32_t q) {
    return sel_cells(q);
}

int32_t units(int32_t rows, int32_t depth) {
    int32_t n = 0;
    for (int32_t t = 0; t < rows; ++t) {
        n += 2 * ((selected_cells(depth + t) + chunk - 1) / chunk);
    }
    return n;
}

int32_t grid(int32_t rows, int32_t device) {
    const int32_t n = 2 * max_chunks * rows, cap = ctas_per_sm(device) * n_sm; // the units of the deepest rows
    return n < cap ? n : cap;
}

namespace {

// ME3, KV8: an instantiation's grid, from its own residency (a unit's result does not depend on its CTA)
int32_t grid_of(int32_t rows, int32_t device, int32_t variant) {
    const int32_t n = 2 * max_chunks * rows, cap = ctas_per_sm(device, variant) * n_sm;
    return n < cap ? n : cap;
}

} // namespace

bool attend(int32_t rows, const args & a, void * stream) {
    if (rows < 1 || rows > max_rows || stream == nullptr || a.depth == nullptr || a.max_cells < rows ||
        a.max_cells > fn::qsa::max_cells || !aligned(a.qfull, 16) || !aligned(a.kv_k, 16) || !aligned(a.kv_v, 16) ||
        a.q_norm == nullptr || a.k_norm == nullptr || a.body == nullptr || a.offsets == nullptr || a.pos == nullptr ||
        !aligned(a.k_cache, 16) || !aligned(a.v_cache, 16) || !aligned(a.out, 16) || !aligned(a.codes, 16) ||
        !aligned(a.scales, 8) || (a.k_f32 != nullptr && !aligned(a.k_f32, 16)) || !aligned(a.ws.q, 16) ||
        !aligned(a.ws.acc, 16) || !aligned(a.ws.ml, 8) || a.ws.ready == nullptr || a.ws.error == nullptr ||
        (a.k_scale == nullptr) != (a.v_scale == nullptr) || (a.k_scale != nullptr && a.shift != nullptr) ||
        (a.k_scale != nullptr && (!aligned(a.k_scale, 2) || !aligned(a.v_scale, 2)))) {
        return false;
    }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    kargs k = {};
    k.qfull = a.qfull;
    k.kv_k = a.kv_k;
    k.kv_v = a.kv_v;
    k.q_norm = a.q_norm;
    k.k_norm = a.k_norm;
    k.body = a.body;
    k.pos = a.pos;
    k.k_cache = a.k_cache;
    k.v_cache = a.v_cache;
    k.k_f32 = a.k_f32;
    k.stamps = a.stamps;
    k.q = a.ws.q;
    k.acc = a.ws.acc;
    k.ml = a.ws.ml;
    k.ready = a.ws.ready;
    k.error = a.ws.error;
    k.rows = rows;
    k.depth = a.depth;
    k.offsets = a.offsets;
    k.max_cells = a.max_cells;
    k.theta_scale = a.theta_scale;
    k.mz = 0u;
    k.out = a.out;
    k.codes = a.codes;
    k.scales = a.scales;
    k.shift = a.shift;
    k.k_scale = a.k_scale;
    k.v_scale = a.v_scale;
    const int32_t variant = a.shift != nullptr ? 1 : a.k_scale != nullptr ? 2 : 0;
    const int32_t g = variant == 0 ? grid(rows, device) : grid_of(rows, device, variant);
    if (g < 1 || n_task_row * rows > n_warp * g) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    void * params[] = {&k};
    if (cudaLaunchCooperativeKernel(kernel_of(variant), dim3(g), dim3(threads), params, 0, st) != cudaSuccess) {
        return false;
    }
    return true;
}

bool kernel_resources(resources & r, bool q8) {
    cudaFuncAttributes fa = {};
    int device = -1;
    if (cudaFuncGetAttributes(&fa, kernel_of(q8 ? 2 : 0)) != cudaSuccess || cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    r.regs = fa.numRegs;
    r.local_bytes = static_cast<int32_t>(fa.localSizeBytes);
    r.shared_bytes = static_cast<int32_t>(fa.sharedSizeBytes);
    r.ctas_per_sm = ctas_per_sm(device, q8 ? 2 : 0);
    return true;
}

} // namespace fn::qsa::attn
