#include "fn/kernels/gdn/gdn_ref.h"

#include "fn/kernels/act/act_device.cuh"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/round/version.h"
#include "fn/model/shapes.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

// ME11 TEST REFERENCE: the kernel below is eee8f38's gdn_core, copied unchanged but for its argument struct's name

namespace fn::gdn::ref {
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
//             3. the recurrence, the rows in order; each row step stores logical version b + 1. Every warp runs it
//                (warp 21 of a 21-row split on a copy of row 20, storing nothing), so no branch encloses its shuffles.
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
//   versions: (SR1) logical version v at slot fn::round::version_slot(cur, v), cur read once from its device word at
//             the start (before the small loads); the state row's address waits for it.
//   floor:    1.949 us + bytes / 1,021.6 GB/s; bytes a launch: reads 1,572,864 + B 45,248 + 704 (state; conv, z, alpha,
//             beta, the finish's re-read; dt, a, ssm_norm), writes B 1,601,280 (versions; o, the finish's rewrite,
//             codes and scales): 1.709 / 4.804 MB at B = 3, 8.32 us. Forecast T(B) with the measured write rate W =
//             1,318.9 GB/s (Amendment 2b step 0): 8.665 us at B = 3, 6.349 at B = 1 [A, GD1 results.json a2b].
//   evidence: GD1 (gd-test, gd-bench); simulator node gdn.recur (family gdn_recur).
template <int32_t B> __global__ void __launch_bounds__(threads, 2) core_kernel(const core_ref_args a) {
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
    const uint32_t cur = *a.cur; // the committed slot (SR1)

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
    // the state and its versions are read next a pass later: evict-first (__ldcs, __stcs) keeps the round's code in L2
    // (XB1, from CR3)
    float4 m =
        __ldcs(reinterpret_cast<const float4 *>(a.state + fn::round::version_slot(cur, 0) * sv + row_off) + lane);

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
        s_decay[lane] = expf(sp * gaa);
        s_beta[lane] = sigmoid_fast(gb);
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

    // 3. the recurrence: one warp a row, the step's rows in order (a rolled loop of B trips: code size, R17)
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
            if (has_row) {
                __stcs(reinterpret_cast<float4 *>(a.state + fn::round::version_slot(cur, b + 1) * sv + row_off) + lane,
                       m);
            }
            if (lane == 0) {
                s_o[b][warp] = o;
            }
        }
    }
    __syncthreads();

    // 4. publish o, then the head's last block finishes the norm from its 128 values a row
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

} // namespace

bool core_ref(const core_ref_args & a, void * stream) {
    if (a.rows < 1 || a.rows > max_rows || a.conv == nullptr || a.z == nullptr || a.alpha == nullptr ||
        a.beta == nullptr || a.dt == nullptr || a.a == nullptr || a.norm_w == nullptr || a.state == nullptr ||
        a.cur == nullptr || a.gated == nullptr || a.ws == nullptr || stream == nullptr) {
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

void replay_host(const float * bank, const float * rec_set, int32_t steps, float * out) {
    for (int32_t h = 0; h < n_head; ++h) {
        for (int32_t j = 0; j < head_dim; ++j) {
            for (int32_t i = 0; i < head_dim; ++i) {
                const int64_t at = (int64_t(h) * head_dim + j) * head_dim + i;
                float m = bank[at];
                for (int32_t t = 0; t < steps; ++t) {
                    const float * r = rec_set + int64_t(t) * rec_step_floats;
                    const float dec = r[rec_decay + h];
                    m = m * dec;
                    m = std::fmaf(r[rec_k + (h % n_group) * head_dim + i], r[rec_delta + h * head_dim + j], m);
                }
                out[at] = m;
            }
        }
    }
}

bool read_committed(const float * banks, const float * recs, uint32_t gsel, std::vector<float> & out) {
    if (!fn::round::gdn_valid(gsel)) {
        return false;
    }
    std::vector<float> bank(static_cast<size_t>(state_version_floats)), rec(static_cast<size_t>(rec_set_floats));
    const int32_t steps = fn::round::gdn_steps(gsel);
    if (cudaMemcpy(bank.data(), banks + fn::round::gdn_bank(gsel) * state_version_floats, bank.size() * 4,
                   cudaMemcpyDefault) != cudaSuccess ||
        (steps > 0 && cudaMemcpy(rec.data(), recs + fn::round::gdn_set(gsel) * rec_set_floats,
                                 size_t(steps) * rec_step_floats * 4, cudaMemcpyDefault) != cudaSuccess)) {
        return false;
    }
    out.resize(bank.size());
    replay_host(bank.data(), rec.data(), steps, out.data());
    return true;
}

} // namespace fn::gdn::ref
