#include "fn/kernels/act/act_warp.cuh"
#include "fn/kernels/draft/draft.h"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.cuh"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace fn::draft {
namespace detail {
bool in_resources(int32_t which, resources & r); // draft_in.cu
} // namespace detail

namespace {

constexpr int32_t prep_threads = (n_head + 1) * 32; // 12 query warps, then the K/V warp
constexpr int32_t threads = 256;                    // attend
constexpr int32_t n_warp = threads / 32;
constexpr int32_t m_rows = 16;                     // two mma M-blocks of 8: heads 0..11, rows 12..15 zero
constexpr int32_t q_ld = 272;                      // Q16 row stride (halves): conflict-free A fragments
constexpr int32_t s_ld = 68;                       // S row stride (floats): conflict-free quadpair-0 stores
constexpr int32_t p_ld = 72;                       // P row stride (halves): conflict-free A fragments
constexpr int32_t n_kc = head_dim / 4;             // S: k-chunks of 4 dims (16 a quadpair)
constexpr int32_t n_fold = n_head * head_dim / 32; // fold warps: (head, 32 dims)
constexpr int32_t hc_slot_u4 = fn::xchg::slot_words(fn::xchg::msg::hc) / 4; // X-HC's slot in 16-byte groups

static_assert(head_dim == 32 * 8, "a prepare lane holds 8 dims");
static_assert(tile == 8 * n_warp, "S: a warp takes 8 cells of the tile");
static_assert(head_dim == 32 * n_warp, "P V: a warp takes 32 dims, a quadpair 8");
static_assert(tile / 4 == 16 && n_kc == 64, "the k-chunk splits below");
static_assert(m_rows * tile == 4 * threads, "softmax: 4 scores a thread");
static_assert(fn::rms_eps == 1e-6f, "the pinned eps (attention.layer_norm_rms_epsilon)");

// ---- prepare ----

// The F32 rms norm and IMROPE of one 256-wide row, a lane holding dims 8 lane .. 8 lane + 7. Copied unchanged from
// QA2's fn/kernels/qsa/attend.cu (norm_rope, lines 141-186 at db9a443), which states the pinned source (build_norm
// LLM_NORM_RMS, then ggml_rope_multi with GGML_ROPE_TYPE_IMROPE: ggml_mrope_cache_init and rotate_pairs) and whose
// qa2-test checks it; the draft's attention uses the same rope (qwen4exp.cpp build_mtp, lines 893-900).
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

__device__ __forceinline__ uint4 pack_f16(const float (&x)[8], float scale) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(x[2 * i] * scale))) |
               static_cast<uint32_t>(__half_as_ushort(__float2half_rn(x[2 * i + 1] * scale))) << 16;
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ void put8(float * dst, const float (&y)[8], int32_t lane) {
    float4 * d = reinterpret_cast<float4 *>(dst) + 2 * lane;
    d[0] = make_float4(y[0], y[1], y[2], y[3]);
    d[1] = make_float4(y[4], y[5], y[6], y[7]);
}

__global__ void __launch_bounds__(prep_threads) prepare_kernel(const prepare_args a, int32_t rows) {
    const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32, lane = static_cast<int32_t>(threadIdx.x) % 32;
    const int32_t t = static_cast<int32_t>(blockIdx.x);
    if (t == rows) {
        // the release (draft.h): this card's own X-HC and X-HMIX slot of the round, after their last readers
        // (stream order: the K/V product that read X-HMIX ran before this launch)
        for (int32_t u = static_cast<int32_t>(threadIdx.x); u < fn::xchg::hmix_slot_u4; u += prep_threads) {
            if (a.release_hc != nullptr && u < hc_slot_u4) {
                fn::xchg::reset4(reinterpret_cast<uint4 *>(a.release_hc) + u);
            }
            if (a.release_hmix != nullptr) {
                fn::xchg::hmix_reset_slot(a.release_hmix, u);
            }
        }
        return;
    }
    const int32_t nb = *a.n_before, stride = *a.stride;
    const int64_t cell = static_cast<int64_t>(nb) + a.row0 + t;
    if (nb < 0 || static_cast<int64_t>(nb) + a.row0 + rows > a.max_cells || a.row0 + rows > stride) {
        if (threadIdx.x == 0) {
            atomicOr(a.err, fn::xchg::err_draft_depth);
        }
        return;
    }
    const int32_t * pos = a.pos + a.row0;
    float y[8];
    if (warp < n_head) {
        if (a.qfull == nullptr) {
            return;
        }
        norm_rope(y, a.qfull + (t * n_head + warp) * q_stride, a.q_norm, pos, stride, t, a.theta_scale, lane);
        if (a.q_f32 != nullptr) {
            put8(a.q_f32 + (t * n_head + warp) * head_dim, y, lane);
        }
        reinterpret_cast<uint4 *>(a.q16 + (t * n_head + warp) * head_dim)[lane] = pack_f16(y, 1.0f / 16.0f);
        return;
    }
    norm_rope(y, a.kv_k + t * head_dim, a.k_norm, pos, stride, t, a.theta_scale, lane);
    if (a.k_f32 != nullptr) {
        put8(a.k_f32 + t * head_dim, y, lane);
    }
    reinterpret_cast<uint4 *>(a.k_cache + cell * head_dim)[lane] = pack_f16(y, 1.0f);
    const float4 va = __ldg(reinterpret_cast<const float4 *>(a.kv_v + t * head_dim) + 2 * lane);
    const float4 vb = __ldg(reinterpret_cast<const float4 *>(a.kv_v + t * head_dim) + 2 * lane + 1);
    const float v[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
    reinterpret_cast<uint4 *>(a.v_cache + cell * head_dim)[lane] = pack_f16(v, 1.0f);
}

// ---- attend ----

// D = A B + D, m8n8k4, F16 inputs, F32 accumulators (PTX ISA 9.7.14.5.14; fragments 9.7.14.5.1): a quadpair's
// independent 8x8x4 product. A row-major: a lane's 4 halves are row lane % 4 (+4 for lanes >= 16), columns 0..3.
__device__ __forceinline__ void mma_rc(float (&c)[8], uint2 a, uint2 b) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

// as mma_rc with B row-major: a lane's 4 halves are row lane % 4, columns 0..3 (+4 for lanes >= 16)
__device__ __forceinline__ void mma_rr(float (&c)[8], uint2 a, uint2 b) {
    asm("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

// accumulator i of a lane: row (lane & 1) + (i & 2) (+4 for lanes >= 16), column (i & 4) + (lane & 2) + (i & 1)
__device__ __forceinline__ int32_t c_row(int32_t lane, int32_t i) {
    return (lane & 1) + (i & 2) + (lane >= 16 ? 4 : 0);
}
__device__ __forceinline__ int32_t c_col(int32_t lane, int32_t i) {
    return (i & 4) + (lane & 2) + (i & 1);
}

struct smem_t {
    uint16_t q[m_rows][q_ld]; // F16 q / 16, rows 12..15 zero
    float s[m_rows][s_ld];    // the tile's scores (-inf past the last cell)
    uint16_t p[m_rows][p_ld]; // F16 probabilities
    float m[m_rows];          // running max a row
    float l[m_rows];          // running sum of the rounded P a row
    float alpha[m_rows];      // the tile's rescale of the accumulators
};

// SP2 (sparse): the tile's physical cells (0 past the last logical cell, whose loads are skipped)
struct smem_sp : smem_t {
    int32_t cells[tile];
};

struct kargs {
    attend_args a;
    int32_t grid;
};

// SP (SP2, sparse): the logical cells i < 4 K are cells 4 body[2 + i / 4] + i % 4 (the selected blocks), then the
// prefix's tail and the chain's cells 4 (N / 4) .. N + row0 in order; n counts logical cells and the tile's physical
// cells are staged in shared memory. SP false: cell i is cell i (DR1's dense form, unchanged).
template <bool SP> __global__ void __launch_bounds__(threads, 2) attend_kernel(const kargs ka) {
    const attend_args & a = ka.a;
    __shared__ std::conditional_t<SP, smem_sp, smem_t> sm;
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t warp = tid / 32, lane = tid % 32;
    const int32_t qp = (lane >> 2) & 3;                   // the lane's quadpair
    const int32_t fr = (lane & 3) + (lane >= 16 ? 4 : 0); // A row / B .col column / B .row column offset
    const int32_t nb = *a.n_before;
    const int64_t n64 = static_cast<int64_t>(nb) + a.row0 + 1;
    // sparse: K selected blocks of the prefix's N / 4 whole ones (the body of select at depth N - 1)
    const int32_t n_vis = nb / sel_ratio;
    const int32_t K = SP ? static_cast<int32_t>(__ldg(a.body)) : 0;
    if (nb < 0 || n64 > a.max_cells || (SP && K != min(n_vis, sel_blocks))) {
        // the runtime refuses such a round (FC-RD1); sparse: a body not made at this depth. Every CTA leaves before
        // the grid barrier, nothing is written
        if (blockIdx.x == 0 && tid == 0) {
            atomicOr(a.err, fn::xchg::err_draft_depth);
        }
        return;
    }
    const int32_t n = SP ? sel_ratio * K + static_cast<int32_t>(n64) - sel_ratio * n_vis : static_cast<int32_t>(n64);
    for (int32_t u = tid; u < m_rows * head_dim / 8; u += threads) {
        const int32_t r = u / (head_dim / 8), c = u % (head_dim / 8);
        const uint4 v =
            r < n_head ? __ldg(reinterpret_cast<const uint4 *>(a.q16 + r * head_dim) + c) : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4 *>(&sm.q[r][8 * c]) = v;
    }
    if (tid < m_rows) {
        sm.m[tid] = -INFINITY;
        sm.l[tid] = 0.0f;
    }
    float o[2][8];
#pragma unroll
    for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            o[mb][i] = 0.0f;
        }
    }
    const int32_t n_tiles = (n + tile - 1) / tile;
    const int32_t cta = static_cast<int32_t>(blockIdx.x);
    const int32_t t0 = cta * n_tiles / ka.grid, t1 = (cta + 1) * n_tiles / ka.grid;
    const int32_t d0 = 32 * warp + 8 * qp + (lane >= 16 ? 4 : 0); // P V: the lane's B .row dims d0 .. d0 + 3
    __syncthreads();
    for (int32_t tt = t0; tt < t1; ++tt) {
        const int32_t base = tt * tile;
        if constexpr (SP) {
            // the tile's physical cells; the previous tile's readers passed its two barriers since
            if (tid < tile) {
                const int32_t i = base + tid;
                int32_t c = 0;
                if (i < sel_ratio * K) {
                    c = sel_ratio * static_cast<int32_t>(__ldg(a.body + 2 + i / sel_ratio)) + i % sel_ratio;
                } else if (i < n) {
                    c = sel_ratio * n_vis + i - sel_ratio * K;
                }
                sm.cells[tid] = c;
            }
            __syncthreads();
        }
        // the tile's K fragments (S: cell base + 8 warp + fr, dims 16 j + 4 qp ..) and V fragments (P V: cell
        // base + 4 kc + lane % 4, dims d0 ..), zero past the last cell (stale cells may hold anything)
        const int32_t kl = base + 8 * warp + fr;
        int32_t kcell = kl;
        if constexpr (SP) {
            kcell = sm.cells[kl - base];
        }
        const uint2 * kp = reinterpret_cast<const uint2 *>(a.k_cache + static_cast<int64_t>(kcell) * head_dim);
        uint2 kb[16], vb[16];
#pragma unroll
        for (int32_t j = 0; j < 16; ++j) {
            kb[j] = kl < n ? __ldg(kp + 4 * j + qp) : make_uint2(0u, 0u);
        }
#pragma unroll
        for (int32_t kc = 0; kc < 16; ++kc) {
            const int32_t vl = base + 4 * kc + (lane & 3);
            int32_t vcell = vl;
            if constexpr (SP) {
                vcell = sm.cells[vl - base];
            }
            vb[kc] =
                vl < n
                    ? __ldg(reinterpret_cast<const uint2 *>(a.v_cache + static_cast<int64_t>(vcell) * head_dim + d0))
                    : make_uint2(0u, 0u);
        }
        // S = Q K^T: quadpair qp sums its 16 k-chunks for both M-blocks, then the quadpairs are added (xor 4, 8)
        float s[2][8];
#pragma unroll
        for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                s[mb][i] = 0.0f;
            }
        }
#pragma unroll
        for (int32_t j = 0; j < 16; ++j) {
#pragma unroll
            for (int32_t mb = 0; mb < 2; ++mb) {
                const uint2 af = *reinterpret_cast<const uint2 *>(&sm.q[8 * mb + fr][4 * (4 * j + qp)]);
                mma_rc(s[mb], af, kb[j]);
            }
        }
#pragma unroll
        for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                s[mb][i] += __shfl_xor_sync(0xffffffffu, s[mb][i], 4);
                s[mb][i] += __shfl_xor_sync(0xffffffffu, s[mb][i], 8);
            }
        }
        if (qp == 0) {
#pragma unroll
            for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
                for (int32_t i = 0; i < 8; ++i) {
                    const int32_t col = 8 * warp + c_col(lane, i);
                    sm.s[8 * mb + c_row(lane, i)][col] = base + col < n ? s[mb][i] : -INFINITY;
                }
            }
        }
        __syncthreads();
        // the online softmax: thread tid takes row tid / 16, columns 4 (tid % 16) .. + 3; the 16 lanes of a row
        // reduce by xor 8, 4, 2, 1
        {
            const int32_t r = tid / 16, c0 = 4 * (tid % 16);
            const float4 sv = *reinterpret_cast<const float4 *>(&sm.s[r][c0]);
            float mx = fmaxf(fmaxf(sv.x, sv.y), fmaxf(sv.z, sv.w));
#pragma unroll
            for (int32_t off = 8; off > 0; off >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            }
            const float m_old = sm.m[r], l_old = sm.l[r];
            const float m_new = fmaxf(m_old, mx);
            const __half h0 = __float2half_rn(expf(sv.x - m_new)), h1 = __float2half_rn(expf(sv.y - m_new));
            const __half h2 = __float2half_rn(expf(sv.z - m_new)), h3 = __float2half_rn(expf(sv.w - m_new));
            float sum = __half2float(h0) + __half2float(h1);
            sum += __half2float(h2);
            sum += __half2float(h3);
#pragma unroll
            for (int32_t off = 8; off > 0; off >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            }
            *reinterpret_cast<uint2 *>(&sm.p[r][c0]) = make_uint2(
                static_cast<uint32_t>(__half_as_ushort(h0)) | static_cast<uint32_t>(__half_as_ushort(h1)) << 16,
                static_cast<uint32_t>(__half_as_ushort(h2)) | static_cast<uint32_t>(__half_as_ushort(h3)) << 16);
            __syncwarp(); // every lane of the row has read m and l before lane 0 of the row rewrites them
            if (tid % 16 == 0) {
                const float alpha = expf(m_old - m_new); // 0 for the first tile (m_old = -inf)
                sm.m[r] = m_new;
                sm.l[r] = fmaf(l_old, alpha, sum);
                sm.alpha[r] = alpha;
            }
        }
        __syncthreads();
        // O = O alpha + P V: warp w's dims 32 w .., quadpair qp's 8 of them, all 16 k-chunks of 4 cells
#pragma unroll
        for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                o[mb][i] *= sm.alpha[8 * mb + c_row(lane, i)];
            }
        }
#pragma unroll
        for (int32_t kc = 0; kc < 16; ++kc) {
#pragma unroll
            for (int32_t mb = 0; mb < 2; ++mb) {
                const uint2 af = *reinterpret_cast<const uint2 *>(&sm.p[8 * mb + fr][4 * kc]);
                mma_rr(o[mb], af, vb[kc]);
            }
        }
        // the next tile writes s after this barrier-free stretch only; p, m, l and alpha only after its first
        // barrier, which every thread reaches after this tile's P V
    }
    // the CTA's partial: O of heads 0..11, then m and l (an empty CTA: O = 0, m = -inf, l = 0)
    float * part = a.part + static_cast<int64_t>(cta) * part_floats;
#pragma unroll
    for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t r = 8 * mb + c_row(lane, i);
            if (r < n_head) {
                part[r * head_dim + 32 * warp + 8 * qp + c_col(lane, i)] = o[mb][i];
            }
        }
    }
    __syncthreads();
    if (tid < n_head) {
        part[n_head * head_dim + tid] = sm.m[tid];
        part[n_head * head_dim + n_head + tid] = sm.l[tid];
    }
    cooperative_groups::this_grid().sync();
    // the fold of warp (head h, dims 32 blk ..): every CTA's partial in CTA order, through L2 (written this launch)
    const int32_t gw = cta + ka.grid * warp;
    if (gw >= n_fold) {
        return;
    }
    const int32_t h = gw / 8, blk = gw % 8, d = 32 * blk + lane;
    float M = -INFINITY;
    for (int32_t c = lane; c < ka.grid; c += 32) {
        M = fmaxf(M, __ldcg(a.part + static_cast<int64_t>(c) * part_floats + n_head * head_dim + h));
    }
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, off));
    }
    float L = 0.0f, A = 0.0f;
#pragma unroll 8
    for (int32_t c = 0; c < ka.grid; ++c) {
        const float * pc = a.part + static_cast<int64_t>(c) * part_floats;
        const float w = expf(__ldcg(pc + n_head * head_dim + h) - M); // 0 for an empty CTA
        L = fmaf(__ldcg(pc + n_head * head_dim + n_head + h), w, L);
        A = fmaf(__ldcg(pc + h * head_dim + d), w, A);
    }
    const float g = __ldg(a.qfull + h * q_stride + head_dim + d);
    const float y = A / L * (1.0f / (1.0f + expf(-g)));
    a.out[h * head_dim + d] = y;
    fn::act::put_run_warp(y, lane, a.codes + h * head_dim + 32 * blk, a.scales + (h * 8 + blk) * 2);
}

__global__ void select_depth_kernel(const int32_t * n_before, int32_t * depth) {
    *depth = *n_before - 1;
}

bool aligned(const void * p, uintptr_t n) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % n == 0;
}

int32_t ctas_per_sm(int32_t device) {
    static int32_t cache[16] = {}; // per device (R14)
    if (device < 0 || device >= 16) {
        return 0;
    }
    if (cache[device] == 0) {
        int nb = 0;
        int ns = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, attend_kernel<false>, threads, 0) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ns, attend_kernel<true>, threads, 0) != cudaSuccess) {
            return 0;
        }
        cache[device] = nb < ns ? nb : ns;
    }
    return cache[device];
}

int32_t sm_count(int32_t device) {
    int v = 0;
    return cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, device) == cudaSuccess ? v : 0;
}

} // namespace

size_t part_bytes() {
    return sizeof(float) * size_t(n_cta) * part_floats;
}

bool datc_prepare(int32_t rows, const prepare_args & a, void * stream) {
    if (rows < 1 || rows > max_rows || a.row0 < 0 || a.row0 + rows > max_rows || stream == nullptr || a.max_cells < 1 ||
        !aligned(a.kv_k, 16) || !aligned(a.kv_v, 16) || a.k_norm == nullptr || a.n_before == nullptr ||
        a.pos == nullptr || a.stride == nullptr || !aligned(a.k_cache, 16) || !aligned(a.v_cache, 16) ||
        a.err == nullptr || (a.k_f32 != nullptr && !aligned(a.k_f32, 16)) ||
        (a.qfull != nullptr && (!aligned(a.qfull, 16) || a.q_norm == nullptr || !aligned(a.q16, 16) ||
                                (a.q_f32 != nullptr && !aligned(a.q_f32, 16)))) ||
        (a.release_hc != nullptr && !aligned(a.release_hc, 16)) ||
        (a.release_hmix != nullptr && !aligned(a.release_hmix, 16))) {
        return false;
    }
    const int32_t blocks = rows + (a.release_hc != nullptr || a.release_hmix != nullptr ? 1 : 0);
    prepare_kernel<<<blocks, prep_threads, 0, static_cast<cudaStream_t>(stream)>>>(a, rows);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool datc_attend(const attend_args & a, void * stream) {
    const bool sp = a.body != nullptr;
    if (stream == nullptr || a.row0 < 0 || a.row0 >= (sp ? max_sparse_steps : max_rows) || a.max_cells < 1 ||
        !aligned(a.q16, 16) ||
        !aligned(a.qfull, 16) || !aligned(a.k_cache, 16) || !aligned(a.v_cache, 16) || a.n_before == nullptr ||
        !aligned(a.part, 16) || !aligned(a.out, 16) || a.codes == nullptr || !aligned(a.scales, 8) ||
        a.err == nullptr) {
        return false;
    }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    const int32_t grid = sp ? sparse_cta : n_cta;
    if (ctas_per_sm(device) * sm_count(device) < grid || grid * n_warp < n_fold) {
        return false; // the grid barrier needs every CTA co-resident
    }
    kargs k = {a, grid};
    void * params[] = {&k};
    const void * f = sp ? reinterpret_cast<const void *>(attend_kernel<true>)
                        : reinterpret_cast<const void *>(attend_kernel<false>);
    return cudaLaunchCooperativeKernel(f, dim3(grid), dim3(threads), params, 0, static_cast<cudaStream_t>(stream)) ==
           cudaSuccess;
}

bool select_depth(const int32_t * n_before, int32_t * depth, void * stream) {
    if (n_before == nullptr || depth == nullptr || stream == nullptr) {
        return false;
    }
    select_depth_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(n_before, depth);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool kernel_resources(int32_t which, resources & r) {
    if (which == 0 || which == 1) {
        return detail::in_resources(which, r);
    }
    if (which < 2 || which > 4) {
        return false;
    }
    cudaFuncAttributes fa = {};
    int device = -1;
    const void * f = which == 2   ? reinterpret_cast<const void *>(prepare_kernel)
                     : which == 3 ? reinterpret_cast<const void *>(attend_kernel<false>)
                                  : reinterpret_cast<const void *>(attend_kernel<true>);
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess || cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    r.regs = fa.numRegs;
    r.local_bytes = static_cast<int32_t>(fa.localSizeBytes);
    r.shared_bytes = static_cast<int32_t>(fa.sharedSizeBytes);
    if (which >= 3) {
        r.ctas_per_sm = ctas_per_sm(device);
    } else {
        int nb = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, prepare_kernel, prep_threads, 0) != cudaSuccess) {
            return false;
        }
        r.ctas_per_sm = nb;
    }
    return true;
}

} // namespace fn::draft
