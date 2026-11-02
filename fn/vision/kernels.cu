#include "fn/vision/kernels.h"
#include "fn/vision/image.h"
#include "fn/vision/profile.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace fn::vision {

namespace {

bool launched(const char * what) {
    const cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fnfl vision: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool reject(const char * what) {
    std::fprintf(stderr, "fnfl vision: %s: shapes or pointers rejected\n", what);
    return false;
}

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    return static_cast<uint32_t>(__half_as_ushort(__float2half_rn(a))) |
           (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(b))) << 16);
}

// the patch's grid position from its merge-order row
__device__ __forceinline__ void grid_of(int32_t p, int32_t gw, int32_t & gx, int32_t & gy) {
    const int32_t g = p >> 2, s = p & 3, half_w = gw >> 1;
    gx              = 2 * (g % half_w) + (s & 1);
    gy              = 2 * (g / half_w) + (s >> 1);
}

// ---- pos_resize ------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256) pos_kernel(const float * __restrict__ t, int32_t gw, int32_t gh,
                                                  float * __restrict__ x) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(gw) * gh * width) {
        return;
    }
    const int32_t p = int32_t(i / width), c = int32_t(i % width);
    int32_t       gx, gy;
    grid_of(p, gw, gx, gy);
    // ggml's upscale with GGML_SCALE_FLAG_ALIGN_CORNERS: sf = (out - 1) / (in - 1), source = out index / sf
    const float sfx = gw > 1 ? float(gw - 1) / float(pos_side - 1) : float(gw) / float(pos_side);
    const float sfy = gh > 1 ? float(gh - 1) / float(pos_side - 1) : float(gh) / float(pos_side);
    const float sx = float(gx) / sfx, sy = float(gy) / sfy;
    int32_t     x0 = int32_t(floorf(sx)), y0 = int32_t(floorf(sy));
    int32_t     x1 = x0 + 1, y1 = y0 + 1;
    x0             = max(0, min(x0, pos_side - 1));
    x1             = max(0, min(x1, pos_side - 1));
    y0             = max(0, min(y0, pos_side - 1));
    y1             = max(0, min(y1, pos_side - 1));
    const float dx = fmaxf(0.0f, fminf(sx - float(x0), 1.0f));
    const float dy = fmaxf(0.0f, fminf(sy - float(y0), 1.0f));
    const float a  = __ldg(t + int64_t(y0 * pos_side + x0) * width + c);
    const float b  = __ldg(t + int64_t(y0 * pos_side + x1) * width + c);
    const float cc = __ldg(t + int64_t(y1 * pos_side + x0) * width + c);
    const float d  = __ldg(t + int64_t(y1 * pos_side + x1) * width + c);
    x[i]           = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + cc * (1 - dx) * dy + d * dx * dy;
}

// ---- patch_embed ------------------------------------------------------------------------------------------------

constexpr int32_t pe_t = 64, pe_k = 16;

__global__ void __launch_bounds__(256) patch_kernel(const float * __restrict__ rows, int32_t P,
                                                    const float * __restrict__ w, const float * __restrict__ b,
                                                    float * __restrict__ x) {
    __shared__ float as[pe_k][pe_t + 4], bs[pe_k][pe_t + 4];
    const int32_t    tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int32_t    o0 = blockIdx.x * pe_t, p0 = blockIdx.y * pe_t;
    float            acc[4][4] = {};
    const int32_t    lr = tid >> 2, lk = (tid & 3) * 4;  // the thread's staging row and K quad
    for (int32_t k0 = 0; k0 < patch_vals; k0 += pe_k) {
        float4 av = make_float4(0, 0, 0, 0);
        if (p0 + lr < P) {
            av = __ldg(reinterpret_cast<const float4 *>(rows + int64_t(p0 + lr) * patch_vals + k0 + lk));
        }
        const float4 bv = __ldg(reinterpret_cast<const float4 *>(w + int64_t(o0 + lr) * patch_vals + k0 + lk));
        as[lk][lr] = av.x;
        as[lk + 1][lr] = av.y;
        as[lk + 2][lr] = av.z;
        as[lk + 3][lr] = av.w;
        bs[lk][lr] = bv.x;
        bs[lk + 1][lr] = bv.y;
        bs[lk + 2][lr] = bv.z;
        bs[lk + 3][lr] = bv.w;
        __syncthreads();
#pragma unroll
        for (int32_t kk = 0; kk < pe_k; ++kk) {
            float ar[4], br[4];
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                ar[i] = as[kk][ty * 4 + i];
                br[i] = bs[kk][tx * 4 + i];
            }
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        const int32_t p = p0 + ty * 4 + i;
        if (p >= P) {
            continue;
        }
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            const int32_t o = o0 + tx * 4 + j;
            float &       d = x[int64_t(p) * width + o];
            d               = (acc[i][j] + __ldg(b + o)) + d;
        }
    }
}

// ---- layernorm --------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256) ln_kernel(const float * __restrict__ x, int32_t rows,
                                                 const float * __restrict__ g, const float * __restrict__ b, float eps,
                                                 uint16_t * __restrict__ y) {
    constexpr int32_t per = width / 32;  // 36
    const int32_t     r   = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int32_t     l   = threadIdx.x & 31;
    if (r >= rows) {
        return;
    }
    const float * xr = x + int64_t(r) * width;
    float         v[per];
    float         s = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < per; ++j) {
        v[j] = xr[l + 32 * j];
        s += v[j];
    }
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
    }
    const float mean = s / float(width);
    float       q    = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < per; ++j) {
        v[j] -= mean;
        q = fmaf(v[j], v[j], q);
    }
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        q += __shfl_xor_sync(0xffffffffu, q, o);
    }
    const float rstd = 1.0f / sqrtf(q / float(width) + eps);
    uint16_t *  yr   = y + int64_t(r) * width;
#pragma unroll
    for (int32_t j = 0; j < per; ++j) {
        const int32_t c = l + 32 * j;
        yr[c]           = __half_as_ushort(__float2half_rn(fmaf(v[j] * rstd, __ldg(g + c), __ldg(b + c))));
    }
}

// ---- qkv_rope ---------------------------------------------------------------------------------------------------

// 10000^(-k / 18) rounded once to F32 (from F64; vision-test kernels checks it against make_rope_freqs)
#define FN_VISION_ROPE_FREQS 0x1.0000000000000p+0f, 0x1.32ef9a0000000p-1f, 0x1.7001ac0000000p-2f, 0x1.b93a6c0000000p-3f, 0x1.0882660000000p-3f, 0x1.3d236e0000000p-4f, 0x1.7c3d2c0000000p-5f, 0x1.c7e5000000000p-6f, 0x1.114d360000000p-6f, 0x1.47ae140000000p-7f, 0x1.88e0c40000000p-8f, 0x1.d70c600000000p-9f, 0x1.1a62d60000000p-9f, 0x1.52926e0000000p-10f, 0x1.95efea0000000p-11f, 0x1.e6b4b40000000p-12f, 0x1.23c5c20000000p-12f, 0x1.5dd3780000000p-13f
__constant__ float rope_freq[rope_pairs] = { FN_VISION_ROPE_FREQS };
constexpr float    rope_freq_host[rope_pairs] = { FN_VISION_ROPE_FREQS };

__global__ void __launch_bounds__(256) qkv_kernel(const float * __restrict__ qkv, int32_t r0, int32_t rows, int32_t P,
                                                  int32_t Pp, int32_t gw, const float * __restrict__ bias,
                                                  uint16_t * __restrict__ Q,
                                                  uint16_t * __restrict__ K, uint16_t * __restrict__ Vt) {
    __shared__ float vs[head_dim][65];
    const int32_t    h  = blockIdx.y;
    const int32_t    t0 = blockIdx.x * 64;  // the tile's first row within the chunk
    // q and k: 64 rows x 36 pairs x 2
    for (int32_t i = threadIdx.x; i < 64 * 36 * 2; i += blockDim.x) {
        const int32_t pr = i % 36, wk = (i / 36) & 1, lr = i / 72;
        const int32_t rc = t0 + lr, p = r0 + rc;
        if (rc >= rows) {
            continue;
        }
        uint16_t * dst = (wk ? K : Q) + (int64_t(h) * Pp + p) * head_dim;
        if (p >= P) {
            dst[pr]      = 0;
            dst[pr + 36] = 0;
            continue;
        }
        const int32_t col = wk * width + h * head_dim + pr;
        const float   x0  = qkv[int64_t(rc) * qkv_width + col] + __ldg(bias + col);
        const float   x1  = qkv[int64_t(rc) * qkv_width + col + 36] + __ldg(bias + col + 36);
        int32_t       gx, gy;
        grid_of(p, gw, gx, gy);
        const float th = float(pr < rope_pairs ? gy : gx) * rope_freq[pr % rope_pairs];
        float       sn, cs;
        sincosf(th, &sn, &cs);
        dst[pr]      = __half_as_ushort(__float2half_rn(x0 * cs - x1 * sn));
        dst[pr + 36] = __half_as_ushort(__float2half_rn(x0 * sn + x1 * cs));
    }
    // v: 64 rows x 72 into shared memory, then transposed
    for (int32_t i = threadIdx.x; i < 64 * head_dim; i += blockDim.x) {
        const int32_t d = i % head_dim, lr = i / head_dim;
        const int32_t rc = t0 + lr, p = r0 + rc;
        float         v  = 0.0f;
        if (rc < rows && p < P) {
            const int32_t col = 2 * width + h * head_dim + d;
            v                 = qkv[int64_t(rc) * qkv_width + col] + __ldg(bias + col);
        }
        vs[d][lr] = v;
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < 64 * head_dim; i += blockDim.x) {
        const int32_t lr = i % 64, d = i / 64;
        const int32_t rc = t0 + lr;
        if (rc < rows) {
            Vt[(int64_t(h) * head_pad + d) * Pp + r0 + rc] = __half_as_ushort(__float2half_rn(vs[d][lr]));
        }
    }
}

// ---- attention --------------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma884(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

constexpr int32_t at_q   = 64;               // queries a block
constexpr int32_t at_k   = 64;               // keys a tile
constexpr int32_t k_ch   = head_dim / 8;     // 9 16-byte chunks a K row
constexpr int32_t v_ch   = at_k / 8;         // 8 chunks a Vt row
constexpr int32_t k_tile = at_k * k_ch;      // 576 chunks
constexpr int32_t v_tile = head_pad * v_ch;  // 640 chunks
constexpr int32_t at_ld  = (k_tile + 127) / 128;  // 5 chunks a thread for each

__device__ __forceinline__ int32_t p_swz(int32_t row) {
    return (row & 3) | (((row >> 3) & 1) << 2);
}

__global__ void __launch_bounds__(128) attn_kernel(const uint16_t * __restrict__ Q, const uint16_t * __restrict__ K,
                                                   const uint16_t * __restrict__ Vt, int32_t P, int32_t Pp,
                                                   float sl2, uint16_t * __restrict__ O) {
    __shared__ uint4 sk[2][k_tile];
    __shared__ uint4 sv[2][v_tile];
    __shared__ uint4 sp[4][16 * v_ch];
    const int32_t    h = blockIdx.y, q0 = blockIdx.x * at_q;
    const int32_t    tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int32_t    qp = (lane >> 2) & 3;
    const int32_t    ra = (lane & 3) | ((lane >> 2) & 4);
    const int32_t    lr = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int32_t    cc = 2 * ((lane >> 1) & 1);

    // the lane's Q fragments: row q0 + 16 warp + 8 (qp & 1) + ra, 18 k4 steps
    uint32_t qa[18][2];
    {
        const uint4 * src = reinterpret_cast<const uint4 *>(
            Q + (int64_t(h) * Pp + q0 + 16 * warp + 8 * (qp & 1) + ra) * head_dim);
#pragma unroll
        for (int32_t c = 0; c < k_ch; ++c) {
            const uint4 v = __ldg(src + c);
            qa[2 * c][0]     = v.x;
            qa[2 * c][1]     = v.y;
            qa[2 * c + 1][0] = v.z;
            qa[2 * c + 1][1] = v.w;
        }
    }
    const uint4 * kg = reinterpret_cast<const uint4 *>(K + int64_t(h) * Pp * head_dim);
    const uint16_t * vg = Vt + int64_t(h) * head_pad * Pp;
    uint4         pk[at_ld], pv[at_ld];
    const auto    fetch = [&](int32_t t) {
#pragma unroll
        for (int32_t i = 0; i < at_ld; ++i) {
            const int32_t j = tid + 128 * i;
            if (j < k_tile) {
                pk[i] = __ldg(kg + int64_t(t) * k_tile + j);
            }
            const int32_t d = j / v_ch, c = j % v_ch;
            pv[i] = __ldg(reinterpret_cast<const uint4 *>(vg + int64_t(d) * Pp + int64_t(t) * at_k) + c);
        }
    };
    const auto store = [&](int32_t buf) {
#pragma unroll
        for (int32_t i = 0; i < at_ld; ++i) {
            const int32_t j = tid + 128 * i;
            if (j < k_tile) {
                sk[buf][j] = pk[i];
            }
            const int32_t d = j / v_ch, c = j % v_ch;
            sv[buf][d * v_ch + (c ^ (d & 7))] = pv[i];
        }
    };

    float o[5][8] = {};
    float m[2] = { -INFINITY, -INFINITY }, l[2] = { 0.0f, 0.0f };
    const int32_t nt = Pp / at_k;
    const int32_t prow = 8 * (qp & 1) + ra;  // the lane's A row in the warp's P tile
    uint32_t *    spw  = reinterpret_cast<uint32_t *>(sp[warp]);
    fetch(0);
    store(0);
    __syncthreads();
#pragma unroll 1
    for (int32_t t = 0; t < nt; ++t) {
        const int32_t buf = t & 1;
        if (t + 1 < nt) {
            fetch(t + 1);
        }
        float s[4][8] = {};
#pragma unroll
        for (int32_t c = 0; c < k_ch; ++c) {
#pragma unroll
            for (int32_t ni = 0; ni < 4; ++ni) {
                const uint4 b = sk[buf][(16 * ni + 8 * (qp >> 1) + ra) * k_ch + c];
                mma884(s[ni], qa[2 * c][0], qa[2 * c][1], b.x, b.y);
                mma884(s[ni], qa[2 * c + 1][0], qa[2 * c + 1][1], b.z, b.w);
            }
        }
        const int32_t key0 = t * at_k + 8 * (qp >> 1) + cc;
        float         mx[2] = { -INFINITY, -INFINITY };
#pragma unroll
        for (int32_t ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                const int32_t key = key0 + 16 * ni + (i & 1) + 4 * (i >> 2);
                s[ni][i]          = key < P ? s[ni][i] * sl2 : -INFINITY;
                mx[(i >> 1) & 1]  = fmaxf(mx[(i >> 1) & 1], s[ni][i]);
            }
        }
        float alpha[2];
#pragma unroll
        for (int32_t rh = 0; rh < 2; ++rh) {
            mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 2));
            mx[rh] = fmaxf(mx[rh], __shfl_xor_sync(0xffffffffu, mx[rh], 8));
            const float mn = fmaxf(m[rh], mx[rh]);
            alpha[rh]      = mn == -INFINITY ? 1.0f : exp2f(m[rh] - mn);
            m[rh]          = mn;
            l[rh] *= alpha[rh];
        }
#pragma unroll
        for (int32_t nj = 0; nj < 5; ++nj) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                o[nj][i] *= alpha[(i >> 1) & 1];
            }
        }
        // P = exp2(s - m) to the warp's tile: row 8 (qp & 1) + lr + 2 rh, columns 16 ni + 8 (qp >> 1) + cc + {0, 1, 4, 5}
#pragma unroll
        for (int32_t ni = 0; ni < 4; ++ni) {
            float e[8];
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                const int32_t rh = (i >> 1) & 1;
                e[i]             = m[rh] == -INFINITY ? 0.0f : exp2f(s[ni][i] - m[rh]);
                l[rh] += e[i];
            }
            const int32_t chunk = 2 * ni + (qp >> 1);
#pragma unroll
            for (int32_t rh = 0; rh < 2; ++rh) {
                const int32_t row = 8 * (qp & 1) + lr + 2 * rh;
                uint32_t *    dst = spw + (row * v_ch + (chunk ^ p_swz(row))) * 4;
                dst[cc >> 1]       = pack_h2(e[2 * rh], e[2 * rh + 1]);
                dst[2 + (cc >> 1)] = pack_h2(e[2 * rh + 4], e[2 * rh + 5]);
            }
        }
        __syncwarp();
#pragma unroll
        for (int32_t c = 0; c < v_ch; ++c) {
            const uint4 a = sp[warp][prow * v_ch + (c ^ p_swz(prow))];
#pragma unroll
            for (int32_t nj = 0; nj < 5; ++nj) {
                const int32_t vr = 16 * nj + 8 * (qp >> 1) + ra;
                const uint4   b  = sv[buf][vr * v_ch + (c ^ (vr & 7))];
                mma884(o[nj], a.x, a.y, b.x, b.y);
                mma884(o[nj], a.z, a.w, b.z, b.w);
            }
        }
        __syncwarp();
        if (t + 1 < nt) {
            store(buf ^ 1);
        }
        __syncthreads();
    }
#pragma unroll
    for (int32_t rh = 0; rh < 2; ++rh) {
        l[rh] += __shfl_xor_sync(0xffffffffu, l[rh], 2);
        l[rh] += __shfl_xor_sync(0xffffffffu, l[rh], 8);
    }
#pragma unroll
    for (int32_t rh = 0; rh < 2; ++rh) {
        const int32_t p = q0 + 16 * warp + 8 * (qp & 1) + lr + 2 * rh;
        if (p >= P) {
            continue;
        }
        const float inv = 1.0f / l[rh];
        uint16_t *  dst = O + int64_t(p) * width + h * head_dim;
#pragma unroll
        for (int32_t nj = 0; nj < 5; ++nj) {
            const int32_t d0 = 16 * nj + 8 * (qp >> 1) + cc;
            if (d0 < head_dim) {
                *reinterpret_cast<uint32_t *>(dst + d0) = pack_h2(o[nj][2 * rh] * inv, o[nj][2 * rh + 1] * inv);
            }
            if (d0 + 4 < head_dim) {
                *reinterpret_cast<uint32_t *>(dst + d0 + 4) =
                    pack_h2(o[nj][2 * rh + 4] * inv, o[nj][2 * rh + 5] * inv);
            }
        }
    }
}

// ---- the elementwise steps ----------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256) bias_res_kernel(const float * __restrict__ y, int64_t n_all, int32_t n,
                                                       const float * __restrict__ b, float * __restrict__ x) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_all) {
        x[i] = (y[i] + __ldg(b + i % n)) + x[i];
    }
}

__device__ __forceinline__ float gelu_tanh(float v) {
    return 0.5f * v * (1.0f + tanhf(0.79788456080286535587989211986876f * v * (1.0f + 0.044715f * v * v)));
}

__global__ void __launch_bounds__(256) bias_gelu_kernel(const float * __restrict__ y, int64_t n_all, int32_t n,
                                                        const float * __restrict__ b, uint16_t * __restrict__ g) {
    const int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (i < n_all) {
        const float v0 = y[i] + __ldg(b + i % n);
        const float v1 = y[i + 1] + __ldg(b + (i + 1) % n);
        *reinterpret_cast<uint32_t *>(g + i) = pack_h2(gelu_tanh(v0), gelu_tanh(v1));
    }
}

__global__ void __launch_bounds__(256) bias_add_kernel(float * __restrict__ y, int64_t n_all, int32_t n,
                                                       const float * __restrict__ b) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_all) {
        y[i] += __ldg(b + i % n);
    }
}

int32_t blocks_for(int64_t n, int32_t per) {
    return int32_t((n + per - 1) / per);
}

}  // namespace

rope_freqs make_rope_freqs() {
    rope_freqs r;
    for (int32_t k = 0; k < rope_pairs; ++k) {
        r.f[k] = rope_freq_host[k];
    }
    return r;
}

bool rope_table_exact() {
    for (int32_t k = 0; k < rope_pairs; ++k) {
        if (rope_freq_host[k] != static_cast<float>(std::pow(10000.0, -double(k) / double(rope_pairs)))) {
            return false;
        }
    }
    return true;
}

bool k_pos_resize(const float * table, int32_t gw, int32_t gh, float * x, void * stream) {
    if (table == nullptr || x == nullptr || gw < 2 || gh < 2 || gw % 2 || gh % 2 || int64_t(gw) * gh > max_patches) {
        return reject("pos_resize");
    }
    const int64_t n = int64_t(gw) * gh * width;
    pos_kernel<<<blocks_for(n, 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(table, gw, gh, x);
    return launched("pos_resize");
}

bool k_patch_embed(const float * rows, int32_t P, const float * w, const float * b, float * x, void * stream) {
    if (rows == nullptr || w == nullptr || b == nullptr || x == nullptr || P < 1 || P > max_patches) {
        return reject("patch_embed");
    }
    const dim3 grid(width / pe_t, blocks_for(P, pe_t));
    patch_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(rows, P, w, b, x);
    return launched("patch_embed");
}

bool k_layernorm(const float * x, int32_t rows, const float * g, const float * b, float eps, uint16_t * y,
                 void * stream) {
    if (x == nullptr || g == nullptr || b == nullptr || y == nullptr || rows < 1) {
        return reject("layernorm");
    }
    ln_kernel<<<blocks_for(rows, 8), 256, 0, static_cast<cudaStream_t>(stream)>>>(x, rows, g, b, eps, y);
    return launched("layernorm");
}

bool k_qkv_rope(const float * qkv, int32_t r0, int32_t rows, int32_t P, int32_t Pp, int32_t gw, const float * bias,
                uint16_t * q, uint16_t * k, uint16_t * vt, void * stream) {
    if (qkv == nullptr || bias == nullptr || q == nullptr || k == nullptr || vt == nullptr || rows < 1 || r0 < 0 ||
        r0 % 64 != 0 || Pp % 64 != 0 || r0 + rows > Pp || P < 1 || P > Pp || gw < 2) {
        return reject("qkv_rope");
    }
    const dim3 grid(blocks_for(rows, 64), heads);
    qkv_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(qkv, r0, rows, P, Pp, gw, bias, q, k, vt);
    return launched("qkv_rope");
}

bool k_attention(const uint16_t * q, const uint16_t * k, const uint16_t * vt, int32_t P, int32_t Pp, uint16_t * o,
                 void * stream) {
    if (q == nullptr || k == nullptr || vt == nullptr || o == nullptr || P < 1 || Pp % at_k != 0 || P > Pp ||
        Pp - P >= at_k || P > max_patches) {
        return reject("attention");
    }
    const float sl2 = static_cast<float>(1.4426950408889634 / std::sqrt(double(head_dim)));
    const dim3  grid(Pp / at_q, heads);
    attn_kernel<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(q, k, vt, P, Pp, sl2, o);
    return launched("attention");
}

bool k_bias_residual(const float * y, int32_t rows, int32_t n, const float * b, float * x, void * stream) {
    if (y == nullptr || b == nullptr || x == nullptr || rows < 1 || n < 1) {
        return reject("bias_residual");
    }
    const int64_t all = int64_t(rows) * n;
    bias_res_kernel<<<blocks_for(all, 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(y, all, n, b, x);
    return launched("bias_residual");
}

bool k_bias_gelu(const float * y, int32_t rows, int32_t n, const float * b, uint16_t * g, void * stream) {
    if (y == nullptr || b == nullptr || g == nullptr || rows < 1 || n < 2 || n % 2 != 0) {
        return reject("bias_gelu");
    }
    const int64_t all = int64_t(rows) * n;
    bias_gelu_kernel<<<blocks_for(all / 2, 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(y, all, n, b, g);
    return launched("bias_gelu");
}

bool k_bias_add(float * y, int32_t rows, int32_t n, const float * b, void * stream) {
    if (y == nullptr || b == nullptr || rows < 1 || n < 1) {
        return reject("bias_add");
    }
    const int64_t all = int64_t(rows) * n;
    bias_add_kernel<<<blocks_for(all, 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(y, all, n, b);
    return launched("bias_add");
}

int32_t vision_kernel_resources(k_resources * out, int32_t max) {
    struct entry {
        const char * name;
        const void * fn;
        int32_t      threads;
    };
    const entry e[] = {
        { "pos_resize", reinterpret_cast<const void *>(pos_kernel), 256 },
        { "patch_embed", reinterpret_cast<const void *>(patch_kernel), 256 },
        { "layernorm", reinterpret_cast<const void *>(ln_kernel), 256 },
        { "qkv_rope", reinterpret_cast<const void *>(qkv_kernel), 256 },
        { "attention", reinterpret_cast<const void *>(attn_kernel), 128 },
        { "bias_residual", reinterpret_cast<const void *>(bias_res_kernel), 256 },
        { "bias_gelu", reinterpret_cast<const void *>(bias_gelu_kernel), 256 },
        { "bias_add", reinterpret_cast<const void *>(bias_add_kernel), 256 },
    };
    int32_t n = 0;
    for (const entry & x : e) {
        if (n >= max) {
            break;
        }
        cudaFuncAttributes a{};
        int                ctas = 0;
        if (cudaFuncGetAttributes(&a, x.fn) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, x.fn, x.threads, 0) != cudaSuccess) {
            return -1;
        }
        out[n++] = { x.name, a.numRegs, int32_t(a.localSizeBytes), int32_t(a.sharedSizeBytes), ctas };
    }
    return n;
}

}  // namespace fn::vision
