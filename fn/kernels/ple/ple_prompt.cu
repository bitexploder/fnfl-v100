#include "fn/kernels/ple/ple.h"
#include "fn/kernels/ple/ple_prompt.h"
#include "fn/kernels/round/version.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace fn::ple {
namespace {

constexpr int32_t D          = 2560;
constexpr int32_t CH         = prompt_channels;
constexpr float   n_d        = 2560.0f;
constexpr float   eps        = 1e-6f;
constexpr int32_t st_threads = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

// feature f's coefficient array of card f / 1280 at its channel 1280 s + f % 1280
__device__ __forceinline__ int32_t own_ch(int32_t s, int32_t f) {
    return n_own * s + (f % n_own);
}

// block: row b. Thread t sums features t + 256 j in order, then the warp tree, then the warps in order
__global__ void __launch_bounds__(st_threads) stats_kernel(const prompt_args a) {
    __shared__ float part[st_threads / 32][13];
    const int32_t    b = static_cast<int32_t>(blockIdx.x), t = static_cast<int32_t>(threadIdx.x);
    float            acc[13];
#pragma unroll
    for (int32_t j = 0; j < 13; ++j) {
        acc[j] = 0.0f;
    }
    for (int32_t f = t; f < D; f += st_threads) {
        const int32_t h = f / n_own;
        const float   v = a.value[int64_t(b) * D + f];
        acc[12]         = fmaf(v, v, acc[12]);
#pragma unroll
        for (int32_t s = 0; s < 4; ++s) {
            const float k  = a.key[int64_t(b) * CH + s * D + f];
            const float hv = a.streams[(int64_t(b) * 4 + s) * D + f];
            acc[s]         = fmaf(k, k, acc[s]);
            acc[4 + s]     = fmaf(hv, hv, acc[4 + s]);
            acc[8 + s] = fmaf(k * __ldg(a.w.norm_key[h] + own_ch(s, f)), hv * __ldg(a.w.norm_query[h] + own_ch(s, f)),
                              acc[8 + s]);
        }
    }
    const int32_t warp = t / 32, lane = t % 32;
#pragma unroll
    for (int32_t j = 0; j < 13; ++j) {
        acc[j] = warp_sum(acc[j]);
    }
    if (lane == 0) {
#pragma unroll
        for (int32_t j = 0; j < 13; ++j) {
            part[warp][j] = acc[j];
        }
    }
    __syncthreads();
    if (t < 4) {
        const int32_t s = t;
        float         tot[4];
        const int32_t idx[4] = { s, 4 + s, 8 + s, 12 };
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            float sum = part[0][idx[j]];
            for (int32_t w = 1; w < st_threads / 32; ++w) {
                sum += part[w][idx[j]];
            }
            tot[j] = sum;
        }
        // E1's scalars (ple.cu scalars), on the full-width sums
        const float rk                 = 1.0f / sqrtf(tot[0] / n_d + eps);
        const float rq                 = 1.0f / sqrtf(tot[1] / n_d + eps);
        const float score              = tot[2] * rk * rq * (1.0f / sqrtf(n_d));
        const float mag                = sqrtf(fmaxf(fabsf(score), 1e-6f));
        const float z                  = score > 0.0f ? mag : score < 0.0f ? -mag : 0.0f;
        const float gate               = 1.0f / (1.0f + expf(-z));
        const float xinv               = 1.0f / sqrtf(gate * gate * (tot[3] / n_d) + eps);
        a.scal[int64_t(b) * 8 + s]     = gate;
        a.scal[int64_t(b) * 8 + 4 + s] = xinv;
    }
}

__global__ void x_kernel(const prompt_args a) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(a.rows) * CH) {
        return;
    }
    const int64_t b  = i / CH;
    const int32_t ch = static_cast<int32_t>(i % CH), s = ch / D, f = ch % D, h = f / n_own;
    const float   gated = a.value[b * D + f] * a.scal[b * 8 + s];
    a.x[i]              = gated * a.scal[b * 8 + 4 + s] * __ldg(a.w.norm_conv[h] + own_ch(s, f));
}

// x at row t - k of the half: the half's own x, or the history (oldest first: hist[9 + (t - k)] for t - k < 0)
__device__ __forceinline__ float x_at(const prompt_args & a, int64_t t, int32_t ch, int32_t back) {
    const int64_t r = t - back;
    return r >= 0 ? a.x[r * CH + ch] : a.hist_in[int64_t(ch) * n_hist + n_hist + r];
}

__global__ void update_kernel(const prompt_args a) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(a.rows) * CH) {
        return;
    }
    const int64_t b  = i / CH;
    const int32_t ch = static_cast<int32_t>(i % CH), s = ch / D, f = ch % D, h = f / n_own;
    const uint2   wraw = __ldg(reinterpret_cast<const uint2 *>(a.w.conv[h]) + own_ch(s, f));
    const float   w0   = __half2float(__ushort_as_half(static_cast<unsigned short>(wraw.x & 0xffffu)));
    const float   w1   = __half2float(__ushort_as_half(static_cast<unsigned short>(wraw.x >> 16)));
    const float   w2   = __half2float(__ushort_as_half(static_cast<unsigned short>(wraw.y & 0xffffu)));
    const float   w3   = __half2float(__ushort_as_half(static_cast<unsigned short>(wraw.y >> 16)));
    float         acc  = x_at(a, b, ch, 9) * w0;
    acc                = fmaf(x_at(a, b, ch, 6), w1, acc);
    acc                = fmaf(x_at(a, b, ch, 3), w2, acc);
    acc                = fmaf(a.x[i], w3, acc);
    const float silu   = acc / (1.0f + expf(-acc));
    const float gated  = a.value[b * D + f] * a.scal[b * 8 + s];
    float *     hp     = a.streams + (b * 4 + s) * D + f;
    *hp                = *hp + (gated + silu);
}

// thread: (channel, history position j): the last 9 x of the channel after the half, oldest first
__global__ void hist_kernel(const prompt_args a) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(CH) * n_hist) {
        return;
    }
    const int32_t ch = static_cast<int32_t>(i / n_hist), j = static_cast<int32_t>(i % n_hist);
    const int64_t r = int64_t(a.rows) - n_hist + j;  // the row this history value is x of (negative: older history)
    const float   v = r >= 0 ? a.x[r * CH + ch] : a.hist_in[int64_t(ch) * n_hist + n_hist + r];
    a.hist_out[i]   = v;
    const int32_t s = ch / D, f = ch % D;
    if (f / n_own == a.card) {
        const uint32_t cur = *a.cur;
        a.state[fn::round::version_slot(cur, a.v_out) * hist_version_floats + int64_t(own_ch(s, f)) * n_hist + j] = v;
    }
}

__global__ void hist_in_kernel(const prompt_args a, const float * peer) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(CH) * n_hist) {
        return;
    }
    const int32_t ch = static_cast<int32_t>(i / n_hist), j = static_cast<int32_t>(i % n_hist);
    const int32_t s = ch / D, f = ch % D;
    const int64_t c = int64_t(own_ch(s, f)) * n_hist + j;
    if (f / n_own == a.card) {
        const uint32_t cur = *a.cur;
        a.hist_out[i]      = a.state[fn::round::version_slot(cur, 0) * hist_version_floats + c];
    } else {
        a.hist_out[i] = peer[c];
    }
}

uint32_t blocks(int64_t n) {
    return static_cast<uint32_t>((n + 255) / 256);
}

bool ok_common(const prompt_args & a, void * stream) {
    for (int32_t h = 0; h < 2; ++h) {
        if (a.w.norm_key[h] == nullptr || a.w.norm_query[h] == nullptr || a.w.norm_conv[h] == nullptr ||
            a.w.conv[h] == nullptr) {
            return false;
        }
    }
    return a.rows >= 1 && a.rows <= prompt_max_rows && stream != nullptr && (a.card == 0 || a.card == 1);
}

bool fail(const char * what) {
    fprintf(stderr, "ple_prompt: %s rejected\n", what);
    return false;
}

}  // namespace

bool ple_prompt_stats(const prompt_args & a, void * stream) {
    if (!ok_common(a, stream) || a.key == nullptr || a.value == nullptr || a.streams == nullptr || a.scal == nullptr) {
        return fail("stats");
    }
    stats_kernel<<<a.rows, st_threads, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool ple_prompt_x(const prompt_args & a, void * stream) {
    if (!ok_common(a, stream) || a.value == nullptr || a.scal == nullptr || a.x == nullptr) {
        return fail("x");
    }
    x_kernel<<<blocks(int64_t(a.rows) * CH), 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool ple_prompt_update(const prompt_args & a, void * stream) {
    if (!ok_common(a, stream) || a.value == nullptr || a.scal == nullptr || a.x == nullptr || a.hist_in == nullptr ||
        a.streams == nullptr) {
        return fail("update");
    }
    update_kernel<<<blocks(int64_t(a.rows) * CH), 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool ple_prompt_hist(const prompt_args & a, void * stream) {
    if (!ok_common(a, stream) || a.x == nullptr || a.hist_in == nullptr || a.hist_out == nullptr ||
        a.hist_in == a.hist_out || a.state == nullptr || a.cur == nullptr || a.v_out < 1 || a.v_out > 4) {
        return fail("hist");
    }
    hist_kernel<<<blocks(int64_t(CH) * n_hist), 256, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool ple_prompt_hist_in(const prompt_args & a, const float * peer, void * stream) {
    if (a.hist_out == nullptr || a.state == nullptr || a.cur == nullptr || peer == nullptr || stream == nullptr ||
        (a.card != 0 && a.card != 1)) {
        return fail("hist_in");
    }
    hist_in_kernel<<<blocks(int64_t(CH) * n_hist), 256, 0, static_cast<cudaStream_t>(stream)>>>(a, peer);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace fn::ple
