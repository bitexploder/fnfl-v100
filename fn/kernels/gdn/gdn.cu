#include "fn/kernels/gdn/gdn.h"

#include "fn/kernels/act/act_device.cuh"
#include "fn/kernels/gdn/gdn_conv.cuh"
#include "fn/kernels/round/version.h"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::gdn {
namespace {

constexpr int32_t warps = 22; // one warp a value row; a split holds 22 or 21 rows
constexpr int32_t threads = warps * 32;
constexpr int32_t n_block = n_head * n_split;
constexpr int32_t qk_threads = 2 * head_dim;           // the group's q and k values, one thread a channel
constexpr int32_t zw_first = qk_threads;               // threads 256..383: z and ssm_norm, one thread a value channel
constexpr int32_t v_warp = (zw_first + head_dim) / 32; // warp 12: the block's v values
constexpr int32_t gate_warp = v_warp + 1;              // warp 13: g and beta

static_assert(n_split == 6 && 2 * 22 + 4 * 21 == head_dim, "splits own 22, 22, 21, 21, 21, 21 rows");
static_assert(gate_warp < warps, "the prologue's warps exist");

__host__ __device__ constexpr int32_t split_first(int32_t s) {
    return s < 2 ? 22 * s : 44 + 21 * (s - 2);
}

__host__ __device__ constexpr int32_t split_rows(int32_t s) {
    return s < 2 ? 22 : 21;
}

// The MUFU forms, without the intrinsics' range handling, where the operands cannot reach it (code size, R17):
// sigmoid(x) = 1 / (1 + 2^(-x log2 e)): ex2.approx.ftz and rcp.approx.ftz, about 2 ulp each; the denominator is at least
// 1, so the flush to zero never applies to it (2^(...) underflowing to 0 gives 1, overflowing to inf gives 0).
__device__ __forceinline__ float sigmoid_fast(float x) {
    float e = 0.0f, r = 0.0f;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(x * -1.44269504088896341f));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.0f + e));
    return r;
}

// 1 / sqrt(x) for x >= fn::rms_eps (never a denormal): rsqrt.approx.ftz, about 2 ulp
__device__ __forceinline__ float rsqrt_fast(float x) {
    float r = 0.0f;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// the hand-off's fences: acquire-release at device scope (fence.acq_rel.gpu), the order the protocol needs, lighter than
// __threadfence()'s sequentially consistent fence
__device__ __forceinline__ void fence_acq_rel_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

// gdn_core<B>: one card's GDN math after the convolution (GD1, Amendment 2b), one launch, an instance a B.
//
// Contract:
//   shapes:   rows B in 1..5 (SP2), a template instance each (compile-time trip counts: the short loops over rows are
//             unrolled with immediate offsets, R12; the row-step loop is rolled, B trips, for code size, R17); the
//             card-local shapes of gdn.h. core() rejects anything else.
//   grid:     24 heads x 6 splits = 144 blocks of 704 threads (22 warps), __launch_bounds__(704, 2): two blocks an SM
//             on 72 SMs, at most 40 registers a thread (46 would allocate 48, and two 22-warp blocks would need 67,584
//             of 65,536). Split s of head h owns value rows split_first(s) .. + split_rows(s); warp w < split_rows owns
//             row j = split_first(s) + w.
//   math (head h, group g = h % 8, row b, per the pinned lines named in gdn.h; the convolution is the producer's
//             epilogue, gdn_conv.cuh):
//             q, k:  x / sqrt(sum x^2 + eps) over the group's 128 (the pinned rms_norm(x, eps / n) / sqrt(n), eps =
//                    fn::rms_eps), computed as x rsqrtf(sum x^2 + eps);
//             gates: g = softplus(alpha + dt) a with softplus(x) = x > 20 ? x : logf(1 + expf(x)), decay = expf(g),
//                    beta = sigmoid(beta logit);
//             recurrence, per row j of the head's state M (M[j][i], i the key channel): M = decay M; delta =
//                    beta (v_j - sum_i M[j][i] k_i); M[j][i] += delta k_i; o_j = (sum_i M[j][i] q_i) / sqrt(128);
//             gated: out_j = (o_j / sqrt(sum_j o_j^2 / 128 + eps)) ssm_norm[j] sigmoid(z_j), then the activation
//                    format (quantize_run8, act_device.cuh).
//   order:    1. the small loads first: threads 0..255 the group's q and k (post-SiLU), threads 256..383 the head's z
//                and ssm_norm (for the finish), warp 12 the block's v, warp 13 alpha, beta, dt and a; then every row
//                warp its 512-byte state row (logical version 0, one float4 a lane), the dominant stream.
//             2. while the state streams: q, k, z, ssm_norm and v into shared memory and the gates; one barrier; the
//                q and k norms, warp 2b + w (w = 0 for q, 1 for k) normalizing row b's 128 values; one barrier.
//             3. ME11: the committed matrix materialized in registers: each row warp replays the selector's k records
//                onto its row (per step: the decay, the group's normalized k, the row's delta; m <- fmaf(k, delta, m *
//                decay)) and stores the result, the pre-step state, into the other bank. Then the recurrence, the rows
//                in order, in registers; each step records its operands into the other set (one writer an element:
//                the normalized k by split 0 of head g < 8 (the group's first head), the decay by split 0 of each
//                head, delta by lane 0 of each real row warp). Every warp runs it (warp 21 of a 21-row split on a
//                copy of row 20, storing nothing), so no branch encloses its shuffles.
//             4. warps 0..B-1 write the block's o to gated (F32); the head's last block (see hand-off) reads its rows
//                of gated (all 6 blocks' o) in one round, sums o^2 over each row's 128, applies the norm, ssm_norm and
//                sigmoid(z), and writes gated and its activation format.
//   reductions: dots: each lane its 4 products in order (FMUL then FMAs), then an xor butterfly over offsets 16, 8, 4,
//             2, 1 (every lane ends with the same sum); the q and k norms' and the gated norm's sums of squares: each
//             lane its 4 values in order (FMUL then FMAs), then the same butterfly.
//   hand-off (the head's 6 blocks to its last block; STANDARDS "Memory and publication"):
//             writes:  each block writes its o rows to gated (plain stores);
//             publish: each writing thread then runs fence.acq_rel.gpu, the block synchronizes, and thread 0 takes a
//                      ticket with atomicAdd on the head's counter (device scope): the release pattern;
//             acquire: the block whose ticket is 5 (the last) synchronizes, runs fence.acq_rel.gpu and reads the other
//                      blocks' data with __ldcg (L2, never a stale L1 line): the acquire pattern;
//             release: the last block resets the head's counter to 0 (atomicExch). The buffers are reused only by the
//                      next launch, which the stream orders after this one. No block waits: there is no spin.
//   numerics: F32 storage, compute and accumulation. Built without fast math (--ftz=false --prec-div=true
//             --prec-sqrt=true --fmad=true): the decay's expf and logf (the factor on the carried state) are the precise
//             forms. Fast forms where Amendment 1's bounds allow (Amendment 2b), as inline PTX without the intrinsics'
//             range handling (sigmoid_fast, rsqrt_fast): rsqrt.approx.ftz for the q and k norms and the finish's RMS,
//             ex2.approx.ftz and rcp.approx.ftz for sigmoid(beta) and sigmoid(z). Other intrinsics: __ldg/__ldcg/__ldcs/__stcs (cache hints),
//             __shfl_xor_sync, and quantize_run8's __fdiv_rn and __float2int_rn (bitwise to act_ref). Every reduction
//             has a fixed order and no float atomics, and every row's arithmetic is the same in each instance: reruns
//             are bitwise identical and a row's results do not depend on B (NC1 section 2). Reference: FP64 on the CPU
//             from the pinned lines (gdn_test.cpp); bound max |err| / max |ref| <= 1e-4 per head for every state
//             version, and per row and head for gated <= 1e-4 max(1, 1e-3 / cond), cond = rms(o) / (max_j |M_j| /
//             sqrt(128)) from the reference (GD1 Amendment 1: o = Mq can cancel, and the norm rescales it), with every
//             output exactly 0 where cond is 0; the activation format bitwise.
//   versions: (ME11) the selector read once from its device word at the start (before the small loads); bank, set and
//             k through fn::round::gdn_bank, gdn_set and gdn_steps (version.h). The round never writes the committed
//             bank or set. The replay is the recurrence's own per-cell sequence on the stored operands (FMUL by the
//             decay, then the FFMA with that product as the addend), so version k is reproduced bit for bit.
//   floor:    (GD1, before ME11) 1.949 us + bytes / 1,021.6 GB/s; bytes a launch: reads 1,572,864 + B 45,248 + 704
//             (state; conv, z, alpha, beta, the finish's re-read; dt, a, ssm_norm), writes B 1,601,280 (versions; o,
//             the finish's rewrite, codes and scales). ME11 [counted]: the state reads 1,572,864 + k 16,512 (the
//             bank and k records), writes 1,572,864 + B 16,512 (the pre-step bank and B records) in place of
//             B 1,572,864 versions; unmeasured.
//   evidence: GD1 (gd-test, gd-bench); simulator node gdn.recur (family gdn_recur).
// FMS2 (build item 3, W3): the kernel's body on a's head and split blockIdx.x (core_kernel's, and a lane's of
// core_lanes_kernel)
template <int32_t B> __device__ __forceinline__ void core_body(const core_args & a) {
    // the small values, one array (32-bit shared offsets): q [B][128], k [B][128], z [B][128], ssm_norm [128], v [B][22]
    constexpr int32_t o_q = 0, o_k = B * head_dim, o_z = 2 * B * head_dim, o_w = 3 * B * head_dim;
    constexpr int32_t o_v = o_w + head_dim, n_small = o_v + B * warps;
    __shared__ __align__(16) float s_f[n_small];
    __shared__ float s_o[B][warps];
    __shared__ float s_decay[B];
    __shared__ float s_beta[B];
    __shared__ int32_t s_last;
    static_assert(2 * B <= warps && B <= v_warp, "warps 0..2B-1 run the norms, 0..B-1 write o and finish");

    const int32_t h = static_cast<int32_t>(blockIdx.x) / n_split;
    const int32_t s = static_cast<int32_t>(blockIdx.x) % n_split;
    const int32_t g = h % n_group;
    const int32_t row0 = split_first(s);
    const int32_t n_rows = split_rows(s);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t warp = tid / 32, lane = tid % 32;
    const uint32_t gs = *a.gsel; // ME11: the committed bank, set and steps (version.h)

    // 1. the small loads, then the state row: threads 0..255 the group's q and k, 256..383 the head's z (and
    // ssm_norm), warp 12's lanes the block's v (B values each, into s_f), warp 13 the gates' inputs
    const bool qk = tid < qk_threads;
    const bool zw = tid >= zw_first && tid < zw_first + head_dim;
    const bool vt = warp == v_warp && lane < n_rows;
    const bool gt = warp == gate_warp && lane < B;
    const bool small = qk || zw || vt;
    const int32_t src_off = qk   ? (tid < head_dim ? q_first : k_first - head_dim) + g * head_dim + tid
                            : zw ? h * head_dim + tid - zw_first
                                 : v_first + h * head_dim + row0 + lane;
    const int32_t src_stride = zw ? n_value : n_channel;
    const int32_t dst = qk ? (tid < head_dim ? o_q : o_k - head_dim) + tid : zw ? o_z + tid - zw_first : o_v + lane;
    const int32_t dst_stride = vt ? warps : head_dim;
    float x[B];
    float wn = 0.0f, ga = 0.0f, gb = 0.0f, gdt = 0.0f, gaa = 0.0f;
    if (small) {
        const float * src = (zw ? a.z : a.conv) + src_off;
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            x[b] = src[b * src_stride];
        }
        if (zw) {
            wn = a.norm_w[tid - zw_first];
        }
    } else if (gt) {
        ga = a.alpha[lane * n_head + h];
        gb = a.beta[lane * n_head + h];
        gdt = a.dt[h];
        gaa = a.a[h];
    }
    // every warp runs the row steps, so no branch encloses their shuffles: warp 21 of a 21-row split works on a copy
    // of row 20 and stores nothing
    const bool has_row = warp < n_rows;
    const int32_t row_off = (h * head_dim + row0 + (has_row ? warp : n_rows - 1)) * head_dim; // floats, in a version
    constexpr int32_t sv = static_cast<int32_t>(state_version_floats);
    // the state is read next a pass later: evict-first (__ldcs, __stcs) keeps the round's code in L2 (XB1, from CR3)
    float4 m = __ldcs(
        reinterpret_cast<const float4 *>(a.state + fn::round::gdn_bank_of(gs, 0) * sv + row_off) + lane);

    // 2. while the state streams: the small values into shared memory, the gates; then the q and k norms, one warp a
    // (q or k, row) vector
    if (small) {
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            s_f[dst + b * dst_stride] = x[b];
        }
        if (zw) {
            s_f[o_w + tid - zw_first] = wn;
        }
    } else if (gt) {
        const float xg = ga + gdt;
        const float lp = logf(1.0f + expf(xg)); // computed for every x and selected (no branch; inf above about 88)
        const float sp = xg > 20.0f ? xg : lp;
        const float dec = expf(sp * gaa);
        s_decay[lane] = dec;
        s_beta[lane] = sigmoid_fast(gb);
        if (s == 0) { // ME11: the head's decay, one writer (split 0)
            a.rec[(1 - fn::round::gdn_set(gs)) * rec_set_floats + lane * rec_step_floats + rec_decay + h] = dec;
        }
    }
    __syncthreads();
    const float4 * s_q = reinterpret_cast<const float4 *>(s_f + o_q); // [B][32]
    const float4 * s_k = reinterpret_cast<const float4 *>(s_f + o_k);
    if (warp < 2 * B) {
        float4 * vec = reinterpret_cast<float4 *>(s_f + (warp % 2 == 0 ? o_q : o_k) + (warp / 2) * head_dim) + lane;
        float4 v = *vec;
        float ss = v.x * v.x;
        ss = fmaf(v.y, v.y, ss);
        ss = fmaf(v.z, v.z, ss);
        ss = fmaf(v.w, v.w, ss);
        const float r = rsqrt_fast(warp_sum(ss) + fn::rms_eps);
        v.x *= r;
        v.y *= r;
        v.z *= r;
        v.w *= r;
        *vec = v;
    }
    __syncthreads();

    // 3. ME11: the group's normalized k into the round's record set (split 0 of the group's first head, h = g)
    float * rec_out = a.rec + (1 - fn::round::gdn_set(gs)) * rec_set_floats;
    if (h < n_group && s == 0 && tid < B * head_dim) {
        rec_out[(tid / head_dim) * rec_step_floats + rec_k + h * head_dim + tid % head_dim] = s_f[o_k + tid];
    }
    // the committed matrix: the selector's k records replayed onto the bank's row, the recurrence's own operations
    const int32_t j_row = row0 + (has_row ? warp : n_rows - 1);
    {
        const float * rec_in = a.rec + fn::round::gdn_set(gs) * rec_set_floats;
        const int32_t steps = fn::round::gdn_steps(gs);
#pragma unroll 1
        for (int32_t t = 0; t < steps; ++t) {
            const float * r = rec_in + t * rec_step_floats;
            const float dec = __ldg(r + rec_decay + h);
            const float4 kv = __ldg(reinterpret_cast<const float4 *>(r + rec_k + g * head_dim) + lane);
            const float delta = __ldg(r + rec_delta + h * head_dim + j_row);
            m.x *= dec;
            m.y *= dec;
            m.z *= dec;
            m.w *= dec;
            m.x = fmaf(kv.x, delta, m.x);
            m.y = fmaf(kv.y, delta, m.y);
            m.z = fmaf(kv.z, delta, m.z);
            m.w = fmaf(kv.w, delta, m.w);
        }
    }
    if (has_row) { // the pre-step state into the other bank (logical version 1's bank)
        __stcs(reinterpret_cast<float4 *>(a.state + fn::round::gdn_bank_of(gs, 1) * sv + row_off) + lane, m);
    }

    // 4. the recurrence: one warp a row, the step's rows in order (a rolled loop of B trips: code size, R17)
    {
        const float scale = 1.0f / sqrtf(float(head_dim));
#pragma unroll 1
        for (int32_t b = 0; b < B; ++b) {
            const float4 kv = s_k[b * (head_dim / 4) + lane];
            const float4 qv = s_q[b * (head_dim / 4) + lane];
            const float dec = s_decay[b];
            m.x *= dec;
            m.y *= dec;
            m.z *= dec;
            m.w *= dec;
            float p = m.x * kv.x;
            p = fmaf(m.y, kv.y, p);
            p = fmaf(m.z, kv.z, p);
            p = fmaf(m.w, kv.w, p);
            p = warp_sum(p);
            const float delta = (s_f[o_v + b * warps + warp] - p) * s_beta[b];
            m.x = fmaf(kv.x, delta, m.x);
            m.y = fmaf(kv.y, delta, m.y);
            m.z = fmaf(kv.z, delta, m.z);
            m.w = fmaf(kv.w, delta, m.w);
            float o = m.x * qv.x;
            o = fmaf(m.y, qv.y, o);
            o = fmaf(m.z, qv.z, o);
            o = fmaf(m.w, qv.w, o);
            o = warp_sum(o) * scale;
            if (has_row && lane == 0) { // ME11: the step's delta, the only per-step state write
                rec_out[b * rec_step_floats + rec_delta + h * head_dim + j_row] = delta;
            }
            if (lane == 0) {
                s_o[b][warp] = o;
            }
        }
    }
    __syncthreads();

    // 5. publish o, then the head's last block finishes the norm from its 128 values a row
    uint32_t * counter = static_cast<uint32_t *>(a.ws) + h;
    if (warp < B && lane < n_rows) {
        a.gated[warp * n_value + h * head_dim + row0 + lane] = s_o[warp][lane];
        fence_acq_rel_gpu();
    }
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(counter, 1u) == n_split - 1 ? 1 : 0;
    }
    __syncthreads();
    if (s_last == 0) {
        return;
    }
    fence_acq_rel_gpu();
    if (warp < B) {
        const int32_t b = warp;
        float4 * gp = reinterpret_cast<float4 *>(a.gated + b * n_value + h * head_dim) + lane;
        const float4 o = __ldcg(gp);
        float ss = o.x * o.x;
        ss = fmaf(o.y, o.y, ss);
        ss = fmaf(o.z, o.z, ss);
        ss = fmaf(o.w, o.w, ss);
        ss = warp_sum(ss);
        const float rs = rsqrt_fast(ss / float(head_dim) + fn::rms_eps);
        const float4 w = reinterpret_cast<const float4 *>(s_f + o_w)[lane];
        const float4 zv = reinterpret_cast<const float4 *>(s_f + o_z + b * head_dim)[lane];
        float4 y;
        y.x = o.x * rs * w.x * sigmoid_fast(zv.x);
        y.y = o.y * rs * w.y * sigmoid_fast(zv.y);
        y.z = o.z * rs * w.z * sigmoid_fast(zv.z);
        y.w = o.w * rs * w.w * sigmoid_fast(zv.w);
        *gp = y;
        uint32_t packed = 0u;
        float2 ds = make_float2(0.0f, 0.0f);
        act::quantize_run8(y, packed, ds);
        reinterpret_cast<uint32_t *>(a.gated_q.codes + b * n_value + h * head_dim)[lane] = packed;
        if (lane % 8 == 0) {
            reinterpret_cast<float2 *>(
                a.gated_q.scales)[b * (n_value / act::run) + h * (head_dim / act::run) + lane / 8] = ds;
        }
    }
    if (tid == 0) {
        atomicExch(counter, 0u);
    }
}

template <int32_t B> __global__ void __launch_bounds__(threads, 2) core_kernel(const core_args a) {
    core_body<B>(a);
}

// FMS2 (build item 3, W3): the joint verify round's lanes in one launch, grid.y = U: lane blockIdx.y's 144 blocks are
// core_kernel's on its own arguments (its rows, state, records, selector, outputs and workspace counters)
struct core_lanes_kargs {
    core_args a[fn::xchg::max_lanes];
};

template <int32_t B>
__global__ void __launch_bounds__(threads, 2) core_lanes_kernel(const __grid_constant__ core_lanes_kargs p) {
    core_body<B>(p.a[blockIdx.y]);
}

// conv_silu: the epilogue alone, one thread a channel (gdn.h)
template <int32_t B> __global__ void __launch_bounds__(256) conv_kernel(const conv_args a) {
    const int32_t c = static_cast<int32_t>(blockIdx.x) * 256 + static_cast<int32_t>(threadIdx.x);
    const uint32_t cur = *a.cur;
    const conv_column cc = conv_column_load(a.hist, a.conv_w, c, cur);
    float x[B], y[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        x[b] = a.qkv[b * n_channel + c];
    }
    conv_silu_epilogue<B>(cc, c, x, a.hist, cur, y);
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        a.out[b * n_channel + c] = y[b];
    }
}

bool aligned16(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

const void * core_of(int32_t rows) {
    switch (rows) {
        case 1:
            return reinterpret_cast<const void *>(&core_kernel<1>);
        case 2:
            return reinterpret_cast<const void *>(&core_kernel<2>);
        case 3:
            return reinterpret_cast<const void *>(&core_kernel<3>);
        case 4:
            return reinterpret_cast<const void *>(&core_kernel<4>);
        case 5:
            return reinterpret_cast<const void *>(&core_kernel<5>);
        default:
            return nullptr;
    }
}

} // namespace

size_t workspace_bytes() {
    return size_t(n_head) * 4;
}

namespace {

bool core_ok(const core_args & a) {
    return !(a.rows < 1 || a.rows > max_rows || !aligned16(a.conv) || !aligned16(a.z) || !aligned16(a.alpha) ||
             !aligned16(a.beta) || !aligned16(a.dt) || !aligned16(a.a) || !aligned16(a.norm_w) || !aligned16(a.state) ||
             !aligned16(a.rec) || a.gsel == nullptr ||
             (a.rec < a.state + n_bank * state_version_floats && a.state < a.rec + n_rec_set * rec_set_floats) ||
             !aligned16(a.gated) || !aligned16(a.gated_q.codes) || !aligned16(a.gated_q.scales) || !aligned16(a.ws) ||
             a.gated_q.rows != a.rows || a.gated_q.k != n_value);
}

} // namespace

bool core(const core_args & a, void * stream) {
    if (stream == nullptr || !core_ok(a)) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (a.rows) {
        case 1:
            core_kernel<1><<<n_block, threads, 0, st>>>(a);
            break;
        case 2:
            core_kernel<2><<<n_block, threads, 0, st>>>(a);
            break;
        case 3:
            core_kernel<3><<<n_block, threads, 0, st>>>(a);
            break;
        case 4:
            core_kernel<4><<<n_block, threads, 0, st>>>(a);
            break;
        default:
            core_kernel<5><<<n_block, threads, 0, st>>>(a);
            break;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool core_lanes(int32_t lanes, const core_args * a, void * stream) {
    if (a == nullptr || lanes < 2 || lanes > fn::xchg::max_lanes || stream == nullptr) {
        return false;
    }
    core_lanes_kargs p = {};
    const int32_t    rows = a[0].rows;
    for (int32_t u = 0; u < lanes; ++u) {
        if (!core_ok(a[u]) || a[u].rows != rows || rows < 2) {  // FMS13: every lane at the round's R = d + 1
            return false;
        }
        p.a[u] = a[u];
    }
    const dim3         grid(n_block, lanes);
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 2:
            core_lanes_kernel<2><<<grid, threads, 0, st>>>(p);
            break;
        case 3:
            core_lanes_kernel<3><<<grid, threads, 0, st>>>(p);
            break;
        case 4:
            core_lanes_kernel<4><<<grid, threads, 0, st>>>(p);
            break;
        default:
            core_lanes_kernel<max_rows><<<grid, threads, 0, st>>>(p);
            break;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool conv_silu(const conv_args & a, void * stream) {
    if (a.rows < 1 || a.rows > max_rows || !aligned16(a.qkv) || !aligned16(a.conv_w) || !aligned16(a.hist) ||
        a.cur == nullptr || !aligned16(a.out) || stream == nullptr) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    static_assert(n_channel % 256 == 0, "whole blocks of channels");
    switch (a.rows) {
        case 1:
            conv_kernel<1><<<n_channel / 256, 256, 0, st>>>(a);
            break;
        case 2:
            conv_kernel<2><<<n_channel / 256, 256, 0, st>>>(a);
            break;
        case 3:
            conv_kernel<3><<<n_channel / 256, 256, 0, st>>>(a);
            break;
        case 4:
            conv_kernel<4><<<n_channel / 256, 256, 0, st>>>(a);
            break;
        default:
            conv_kernel<5><<<n_channel / 256, 256, 0, st>>>(a);
            break;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool resources(int32_t rows, core_resources & out) {
    const void * fn = core_of(rows);
    cudaFuncAttributes fa = {};
    int32_t per_sm = 0;
    if (fn == nullptr || cudaFuncGetAttributes(&fa, fn) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, threads, 0) != cudaSuccess) {
        return false;
    }
    out.threads = threads;
    out.blocks = n_block;
    out.blocks_per_sm = per_sm;
    out.regs = fa.numRegs;
    out.local_bytes = static_cast<int32_t>(fa.localSizeBytes);
    out.shared_bytes = fa.sharedSizeBytes;
    return true;
}

bool lanes_resources(int32_t lanes, core_resources & out) {
    const void * kf = reinterpret_cast<const void *>(&core_lanes_kernel<max_rows>);
    cudaFuncAttributes fa = {};
    int32_t per_sm = 0;
    if (lanes < 2 || lanes > fn::xchg::max_lanes || cudaFuncGetAttributes(&fa, kf) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kf, threads, 0) != cudaSuccess) {
        return false;
    }
    out.threads = threads;
    out.blocks = n_block * lanes;
    out.blocks_per_sm = per_sm;
    out.regs = fa.numRegs;
    out.local_bytes = static_cast<int32_t>(fa.localSizeBytes);
    out.shared_bytes = fa.sharedSizeBytes;
    return true;
}

} // namespace fn::gdn
