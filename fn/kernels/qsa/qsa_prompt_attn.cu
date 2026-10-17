#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa_prompt.h"
#include "fn/kernels/tc/tc_gemm.cuh"
#include "fn/model/shapes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace fn::qsa {
namespace {

constexpr int32_t HD         = attn::head_dim;  // 256
constexpr int32_t NH         = attn::n_head;    // 12
constexpr int32_t QS         = attn::q_stride;  // 512
constexpr int32_t at_threads = 256;
constexpr float   inv16      = 1.0f / 16.0f;

// round to nearest even (finite inputs; attend.cu's bf16_bits)
__device__ __forceinline__ uint32_t bf16_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

// attend.cu's norm_rope with every rope plane the row's position: a lane holds dims 8 lane .. 8 lane + 7
__device__ void norm_rope_pos(float (&y)[8],
                              const float * src,
                              const float * w,
                              int32_t       pos,
                              float         theta_scale,
                              int32_t       lane) {
    const float4 a    = __ldg(reinterpret_cast<const float4 *>(src) + 2 * lane);
    const float4 b    = __ldg(reinterpret_cast<const float4 *>(src) + 2 * lane + 1);
    const float  x[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
    float        ss   = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        ss = fmaf(x[e], x[e], ss);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    const float r = 1.0f / sqrtf(ss * (1.0f / HD) + fn::rms_eps);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        y[e] = x[e] * r * __ldg(w + 8 * lane + e);
    }
    float theta = static_cast<float>(pos);
#pragma unroll
    for (int i = 0; i < 31; ++i) {
        theta = i < lane ? theta * theta_scale : theta;
    }
    float s = 0.0f, c = 0.0f;
    sincosf(theta, &s, &c);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float ce    = __shfl_sync(0xffffffffu, c, 8 * (lane & 3) + e);
        const float se    = __shfl_sync(0xffffffffu, s, 8 * (lane & 3) + e);
        const float other = __shfl_xor_sync(0xffffffffu, y[e], 4);
        if (lane < 4) {
            y[e] = y[e] * ce - other * se;
        } else if (lane < 8) {
            y[e] = other * se + y[e] * ce;
        }
    }
}

// a warp a row: k normed and roped and v, BF16, at the row's cell
__global__ void __launch_bounds__(256) kv_kernel(const qsp_kv_args a, float theta_scale) {
    const int32_t lane = threadIdx.x & 31;
    const int32_t t    = static_cast<int32_t>(blockIdx.x) * 8 + static_cast<int32_t>(threadIdx.x >> 5);
    if (t >= a.rows) {
        return;
    }
    const int64_t cell = int64_t(a.pos0) + t;
    float         y[8];
    norm_rope_pos(y, a.kv_k + int64_t(t) * HD, a.k_norm, a.pos0 + t, theta_scale, lane);
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = bf16_bits(y[2 * i]) | (bf16_bits(y[2 * i + 1]) << 16);
    }
    reinterpret_cast<uint4 *>(a.k_cache + cell * HD)[lane] = make_uint4(w[0], w[1], w[2], w[3]);
    const float4 v0    = __ldg(reinterpret_cast<const float4 *>(a.kv_v + int64_t(t) * HD) + 2 * lane);
    const float4 v1    = __ldg(reinterpret_cast<const float4 *>(a.kv_v + int64_t(t) * HD) + 2 * lane + 1);
    const float  vv[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = bf16_bits(vv[2 * i]) | (bf16_bits(vv[2 * i + 1]) << 16);
    }
    reinterpret_cast<uint4 *>(a.v_cache + cell * HD)[lane] = make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ float bf_lo(uint32_t w) {
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bf_hi(uint32_t w) {
    return __uint_as_float(w & 0xffff0000u);
}

// ---- the attention on tensor cores (PB3; Amendment 2: FP16 operands, F32 accumulation and softmax statistics) ----
// The tile structure, fragment layouts and online softmax are datc.cu's (DA-TC, the draft's attention), with the
// prompt's per-row cell list and the cache's BF16 K and V rounded once to FP16 as their fragments load.

constexpr int32_t tc_tile = 64;   // cells a tile
constexpr int32_t m_rows  = 16;   // two mma M-blocks of 8: heads 0..11, rows 12..15 zero
constexpr int32_t q_ld    = 272;  // Q16 row stride (halves): conflict-free A fragments
constexpr int32_t s_ld    = 68;   // S row stride (floats)
constexpr int32_t p_ld    = 72;   // P row stride (halves)
static_assert(tc_tile == 8 * (at_threads / 32), "S: a warp takes 8 cells of the tile");
static_assert(HD == 32 * (at_threads / 32), "P V: a warp takes 32 dims, a quadpair 8");
static_assert(m_rows * tc_tile == 4 * at_threads, "softmax: 4 scores a thread");

// D = A B + D, m8n8k4, F16 inputs, F32 accumulators (PTX ISA 9.7.14.5.14): a quadpair's independent 8x8x4 product.
// A row-major: a lane's 4 halves are row lane % 4 (+4 for lanes >= 16), columns 0..3. B .col: column lane % 4 (+4),
// rows 0..3; B .row: row lane % 4, columns 0..3 (+4 for lanes >= 16)
__device__ __forceinline__ void mma_rc(float (&c)[8], uint2 a, uint2 b) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
        "{%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
}

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

// 4 BF16 values (8 bytes of the cache) to 4 FP16: widened exactly, rounded once to nearest even
__device__ __forceinline__ uint2 bf4_h4(uint2 w) {
    return make_uint2(fn::tc::pack2(bf_lo(w.x), bf_hi(w.x)), fn::tc::pack2(bf_lo(w.y), bf_hi(w.y)));
}

struct tc_smem {
    uint16_t q[m_rows][q_ld];  // F16 q / 16, rows 12..15 zero
    float    s[m_rows][s_ld];  // the tile's scores (-inf past the last cell)
    uint16_t p[m_rows][p_ld];  // F16 probabilities
    float    m[m_rows];        // running max a row
    float    l[m_rows];        // running sum of the rounded P a row
    float    alpha[m_rows];    // the tile's rescale of the accumulators
    int32_t  cell[tc_tile];    // the tile's cells (-1 past the row's last)
};

// a CTA a row: the row's cells in body order (its blocks' 4 cells each, then its tail) in tiles of 64. S = Q K^T: warp
// w takes cells 8 w .. 8 w + 7, quadpair qp the k-chunks 4 j + qp, summed across quadpairs (xor 4, 8). The online
// softmax: thread tid takes row tid / 16, columns 4 (tid % 16) .. + 3. O = O alpha + P V: warp w takes dims 32 w ..,
// quadpair qp 8 of them. The finish: out = O / l x sigmoid(gate), F32 and FP16.
__global__ void __launch_bounds__(at_threads, 2) attend_kernel(const qsp_attend_args a, float theta_scale) {
    __shared__ tc_smem sm;
    const int32_t      t   = static_cast<int32_t>(blockIdx.x);
    const int32_t      tid = static_cast<int32_t>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int32_t      qp  = (lane >> 2) & 3;
    const int32_t      fr  = (lane & 3) + (lane >= 16 ? 4 : 0);
    const int32_t      pos = a.pos0 + t;
    const int32_t      slab = t / prompt_slab, rr = t % prompt_slab;
    const uint32_t * body = a.body + int64_t(slab) * prompt_slab_words + a.offsets[slab * (prompt_slab + 1) + rr] / 4;
    const int32_t    nb = static_cast<int32_t>(body[0]), ntail = static_cast<int32_t>(body[1]);
    const int32_t    n_cells = 4 * nb + ntail;
    for (int32_t h = warp; h < NH; h += at_threads / 32) {
        float y[8];
        norm_rope_pos(y, a.qfull + (int64_t(t) * NH + h) * QS, a.q_norm, pos, theta_scale, lane);
        *reinterpret_cast<uint4 *>(&sm.q[h][8 * lane]) =
            make_uint4(fn::tc::pack2(y[0] * inv16, y[1] * inv16), fn::tc::pack2(y[2] * inv16, y[3] * inv16),
                       fn::tc::pack2(y[4] * inv16, y[5] * inv16), fn::tc::pack2(y[6] * inv16, y[7] * inv16));
    }
    for (int32_t u = tid; u < (m_rows - NH) * (HD / 8); u += at_threads) {
        *reinterpret_cast<uint4 *>(&sm.q[NH + u / (HD / 8)][8 * (u % (HD / 8))]) = make_uint4(0u, 0u, 0u, 0u);
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
    const int32_t d0 = 32 * warp + 8 * qp + (lane >= 16 ? 4 : 0);  // P V: the lane's B .row dims d0 .. d0 + 3
    for (int32_t c0 = 0; c0 < n_cells; c0 += tc_tile) {
        const int32_t nc = min(tc_tile, n_cells - c0);
        if (tid < tc_tile) {
            const int32_t c    = c0 + tid;
            int32_t       cell = -1;
            if (c < 4 * nb) {
                cell = 4 * static_cast<int32_t>(body[2 + c / 4]) + (c % 4);
            } else if (c < n_cells) {
                cell = static_cast<int32_t>(body[2 + nb + (c - 4 * nb)]);
            }
            sm.cell[tid] = cell;
        }
        __syncthreads();
        // the tile's fragments: K (S: cell 8 warp + fr, dims 16 j + 4 qp ..) and V (P V: cell 4 kc + lane % 4, dims
        // d0 ..), zero past the row's last cell
        const int32_t kcell = sm.cell[8 * warp + fr];
        const uint2 * kp    = reinterpret_cast<const uint2 *>(a.k_cache + int64_t(max(kcell, 0)) * HD);
        uint2         kb[16], vb[16];
#pragma unroll
        for (int32_t j = 0; j < 16; ++j) {
            kb[j] = kcell >= 0 ? bf4_h4(__ldg(kp + 4 * j + qp)) : make_uint2(0u, 0u);
        }
#pragma unroll
        for (int32_t kc = 0; kc < 16; ++kc) {
            const int32_t vcell = sm.cell[4 * kc + (lane & 3)];
            vb[kc] = vcell >= 0 ? bf4_h4(__ldg(reinterpret_cast<const uint2 *>(a.v_cache + int64_t(vcell) * HD + d0))) :
                                  make_uint2(0u, 0u);
        }
        float sc[2][8];
#pragma unroll
        for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                sc[mb][i] = 0.0f;
            }
        }
#pragma unroll
        for (int32_t j = 0; j < 16; ++j) {
#pragma unroll
            for (int32_t mb = 0; mb < 2; ++mb) {
                const uint2 af = *reinterpret_cast<const uint2 *>(&sm.q[8 * mb + fr][4 * (4 * j + qp)]);
                mma_rc(sc[mb], af, kb[j]);
            }
        }
#pragma unroll
        for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                sc[mb][i] += __shfl_xor_sync(0xffffffffu, sc[mb][i], 4);
                sc[mb][i] += __shfl_xor_sync(0xffffffffu, sc[mb][i], 8);
            }
        }
        if (qp == 0) {
#pragma unroll
            for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
                for (int32_t i = 0; i < 8; ++i) {
                    const int32_t col                     = 8 * warp + c_col(lane, i);
                    sm.s[8 * mb + c_row(lane, i)][col] = col < nc ? sc[mb][i] : -INFINITY;
                }
            }
        }
        __syncthreads();
        {
            const int32_t r = tid / 16, cc = 4 * (tid % 16);
            const float4  sv = *reinterpret_cast<const float4 *>(&sm.s[r][cc]);
            float         mx = fmaxf(fmaxf(sv.x, sv.y), fmaxf(sv.z, sv.w));
#pragma unroll
            for (int32_t off = 8; off > 0; off >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            }
            const float  m_old = sm.m[r], l_old = sm.l[r];
            const float  m_new = fmaxf(m_old, mx);
            const __half h0 = __float2half_rn(expf(sv.x - m_new)), h1 = __float2half_rn(expf(sv.y - m_new));
            const __half h2 = __float2half_rn(expf(sv.z - m_new)), h3 = __float2half_rn(expf(sv.w - m_new));
            float        sum = __half2float(h0) + __half2float(h1);
            sum += __half2float(h2);
            sum += __half2float(h3);
#pragma unroll
            for (int32_t off = 8; off > 0; off >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            }
            *reinterpret_cast<uint2 *>(&sm.p[r][cc]) = make_uint2(
                static_cast<uint32_t>(__half_as_ushort(h0)) | static_cast<uint32_t>(__half_as_ushort(h1)) << 16,
                static_cast<uint32_t>(__half_as_ushort(h2)) | static_cast<uint32_t>(__half_as_ushort(h3)) << 16);
            __syncwarp();  // every lane of the row has read m and l before lane 0 of the row rewrites them
            if (tid % 16 == 0) {
                const float alpha = expf(m_old - m_new);  // 0 for the first tile (m_old = -inf)
                sm.m[r]           = m_new;
                sm.l[r]           = fmaf(l_old, alpha, sum);
                sm.alpha[r]       = alpha;
            }
        }
        __syncthreads();
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
        // the next tile writes cell before its first barrier: every thread read this tile's cells before this tile's
        // second barrier; s, p, m, l and alpha are rewritten only after the next tile's first barrier
    }
    bool bad = false;
#pragma unroll
    for (int32_t mb = 0; mb < 2; ++mb) {
#pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t h = 8 * mb + c_row(lane, i);
            if (h < NH) {
                const int32_t d   = 32 * warp + 8 * qp + c_col(lane, i);
                const float   g   = __ldg(a.qfull + (int64_t(t) * NH + h) * QS + HD + d);
                const float   v   = o[mb][i] / sm.l[h] * (1.0f / (1.0f + expf(-g)));
                const int64_t off = int64_t(t) * (NH * HD) + h * HD + d;
                if (a.out != nullptr) {
                    a.out[off] = v;
                }
                bad          = bad || !(fabsf(v) <= 65504.0f);
                a.out16[off] = __half_as_ushort(__float2half_rn(v));
            }
        }
    }
    if (bad && a.err != nullptr) {
        atomicOr(a.err, 1u);
    }
}

}  // namespace

bool qsp_kv(const qsp_kv_args & a, void * stream) {
    if (a.rows < 1 || a.rows > prompt_max_rows || a.pos0 < 0 || a.pos0 + a.rows > a.max_cells || a.kv_k == nullptr ||
        a.kv_v == nullptr || a.k_norm == nullptr || a.k_cache == nullptr || a.v_cache == nullptr || stream == nullptr) {
        fprintf(stderr, "qsp_kv: rejected\n");
        return false;
    }
    kv_kernel<<<(a.rows + 7) / 8, 256, 0, static_cast<cudaStream_t>(stream)>>>(a, attn::rope_theta_scale());
    return cudaPeekAtLastError() == cudaSuccess;
}

bool qsp_attend(const qsp_attend_args & a, void * stream) {
    if (a.rows < 1 || a.rows > prompt_max_rows || a.pos0 < 0 || a.pos0 + a.rows > a.max_cells || a.qfull == nullptr ||
        a.q_norm == nullptr || a.body == nullptr || a.offsets == nullptr || a.k_cache == nullptr ||
        a.v_cache == nullptr || a.out16 == nullptr || stream == nullptr) {
        fprintf(stderr, "qsp_attend: rejected\n");
        return false;
    }
    attend_kernel<<<a.rows, at_threads, 0, static_cast<cudaStream_t>(stream)>>>(a, attn::rope_theta_scale());
    return cudaPeekAtLastError() == cudaSuccess;
}

bool qsp_kernel_resources(int32_t which, qsp_resources & r) {
    cudaFuncAttributes at = {};
    cudaError_t        e  = cudaErrorInvalidValue;
    if (which == 1) {
        e = cudaFuncGetAttributes(&at, attend_kernel);
    } else if (which == 2) {
        e = cudaFuncGetAttributes(&at, kv_kernel);
    }
    if (e != cudaSuccess) {
        return false;
    }
    r.regs         = at.numRegs;
    r.local_bytes  = static_cast<int32_t>(at.localSizeBytes);
    r.shared_bytes = static_cast<int32_t>(at.sharedSizeBytes);
    return true;
}

}  // namespace fn::qsa
