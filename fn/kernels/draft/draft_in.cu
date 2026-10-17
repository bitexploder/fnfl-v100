#include "fn/kernels/act/act_warp.cuh"
#include "fn/kernels/draft/draft.h"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::draft {
namespace {

constexpr int32_t send_threads = half / 4; // a float4 a thread: 320
constexpr int32_t send_warps = send_threads / 32;
constexpr int32_t recv_threads = 512;
constexpr int32_t recv_warps = recv_threads / 32;
constexpr int32_t row_u4 = fn::xchg::words_draft / 4; // 1,602
constexpr int32_t ss_at = n_seg * half;               // 6,400

static_assert(fn::xchg::words_draft == ss_at + 8, "X-DRAFT-IN's row: the values, 5 sums, 3 zeros");
static_assert(n_stream * in_runs % recv_warps == 0, "recv: whole runs a warp");
static_assert(fn::rms_eps == 1e-6f, "the pinned eps (attention.layer_norm_rms_epsilon)");

// the hidden input of segment k >= 1 of row i (draft.h: chained, else hidden row hsrc + i - 1)
__device__ __forceinline__ const float * hidden(const send_args & a, int32_t i) {
    if (a.chained) {
        return a.h_nextn;
    }
    const int32_t j = *a.hsrc + i - 1;
    return j < 0 ? a.carry_prev : a.carry + j * n_stream * half;
}

__global__ void __launch_bounds__(send_threads) send_kernel(const send_args a) {
    __shared__ float red[send_warps];
    const int32_t i = static_cast<int32_t>(blockIdx.x) / n_seg, k = static_cast<int32_t>(blockIdx.x) % n_seg;
    const int32_t t = static_cast<int32_t>(threadIdx.x), lane = t % 32, warp = t / 32;
    const float * src = k == 0 ? a.emb + i * a.emb_stride : hidden(a, i) + (k - 1) * half;
    const float * g = k == 0 ? a.g_e : a.g_h + (k - 1) * half;
    const float4 x = reinterpret_cast<const float4 *>(src)[t];
    const float4 w = __ldg(reinterpret_cast<const float4 *>(g) + t);
    float ss = x.x * x.x;
    ss = fmaf(x.y, x.y, ss);
    ss = fmaf(x.z, x.z, ss);
    ss = fmaf(x.w, x.w, ss);
    const uint4 v = make_uint4(fn::xchg::canonical(x.x * w.x), fn::xchg::canonical(x.y * w.y),
                               fn::xchg::canonical(x.z * w.z), fn::xchg::canonical(x.w * w.w));
    const int32_t at = i * fn::xchg::words_draft + k * half + 4 * t;
    reinterpret_cast<uint4 *>(a.peer + at)[0] = v; // 16-byte aligned: rows and segments are multiples of 4 words
    reinterpret_cast<uint4 *>(a.own + at)[0] = v;
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
        red[warp] = ss;
    }
    __syncthreads();
    if (t == 0) {
        float s = red[0];
#pragma unroll
        for (int32_t q = 1; q < send_warps; ++q) {
            s += red[q];
        }
        uint32_t * peer = reinterpret_cast<uint32_t *>(a.peer) + i * fn::xchg::words_draft;
        uint32_t * own = reinterpret_cast<uint32_t *>(a.own) + i * fn::xchg::words_draft;
        const uint32_t bits = fn::xchg::canonical(s);
        peer[ss_at + k] = bits;
        own[ss_at + k] = bits;
        if (k == 0) {
#pragma unroll
            for (int32_t z = n_seg; z < 8; ++z) {
                peer[ss_at + z] = 0u;
                own[ss_at + z] = 0u;
            }
        }
    }
}

__global__ void __launch_bounds__(recv_threads) recv_kernel(const recv_args a) {
    __shared__ uint4 pv[row_u4]; // the peer's row as polled
    __shared__ float rs[n_seg];
    const int32_t i = static_cast<int32_t>(blockIdx.x);
    const int32_t t = static_cast<int32_t>(threadIdx.x), lane = t % 32, warp = t / 32;
    uint4 * slot = reinterpret_cast<uint4 *>(a.slot) + i * row_u4;
    for (int32_t u = t; u < row_u4; u += recv_threads) {
        const uint4 * const p[1] = {slot + u};
        uint4 v[1];
        fn::xchg::poll<1>(p, v, a.err, fn::xchg::err_poll_draft);
        fn::xchg::reset4(slot + u);
        pv[u] = v[0];
    }
    if (i == 0 && a.carry_prev != nullptr) {
        // the retained carry for the next round (draft.h): the round's last hidden row; the send kernel that read
        // the old one finished before this launch (stream order)
        const int32_t j = *a.hsrc + static_cast<int32_t>(gridDim.x) - 1;
        if (j >= 0) {
            for (int32_t u = t; u < n_stream * half / 4; u += recv_threads) {
                reinterpret_cast<float4 *>(a.carry_prev)[u] =
                    reinterpret_cast<const float4 *>(a.carry + j * n_stream * half)[u];
            }
        }
    }
    __syncthreads();
    const float * own = a.own + i * fn::xchg::words_draft;
    const float * peer = reinterpret_cast<const float *>(pv);
    if (t < n_seg) {
        const float so = own[ss_at + t], sp = peer[ss_at + t];
        const float ss = a.card == 0 ? so + sp : sp + so;
        rs[t] = 1.0f / sqrtf(ss / static_cast<float>(fn::n_embd) + fn::rms_eps);
    }
    __syncthreads();
    // run r of the row's 4 x 160: stream s = r / 160, value v = 32 (r % 160) + lane of [e | h_s], half c of each
    for (int32_t r = warp; r < n_stream * in_runs; r += recv_warps) {
        const int32_t s = r / in_runs, col = (r % in_runs) * 32 + lane;
        const int32_t seg = col < 2 * half ? 0 : 1 + s;
        const int32_t c = (col % (2 * half)) / half, idx = col % half;
        const float * src = c == a.card ? own : peer;
        const float y = src[seg * half + idx] * rs[seg];
        const int32_t orow = i * n_stream + s;
        if (a.f32 != nullptr) {
            a.f32[orow * in_width + col] = y;
        }
        fn::act::put_run_warp(y, lane, a.codes + orow * in_width + (r % in_runs) * 32,
                              a.scales + (orow * in_runs + r % in_runs) * 2);
    }
}

bool aligned(const void * p, uintptr_t n) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % n == 0;
}

} // namespace

bool draft_in_send(int32_t rows, const send_args & a, void * stream) {
    if (rows < 1 || rows > max_rows || stream == nullptr || !aligned(a.emb, 16) || a.emb_stride % 4 != 0 ||
        a.emb_stride < half || !aligned(a.g_e, 16) || !aligned(a.g_h, 16) || !aligned(a.peer, 16) ||
        !aligned(a.own, 16) || (a.chained && (rows != 1 || !aligned(a.h_nextn, 16))) ||
        (!a.chained && (!aligned(a.carry, 16) || !aligned(a.carry_prev, 16) || a.hsrc == nullptr))) {
        return false;
    }
    send_kernel<<<rows * n_seg, send_threads, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool draft_in_recv(int32_t rows, const recv_args & a, void * stream) {
    if (rows < 1 || rows > max_rows || stream == nullptr || !aligned(a.own, 16) || !aligned(a.slot, 16) ||
        a.codes == nullptr || !aligned(a.scales, 8) || (a.f32 != nullptr && !aligned(a.f32, 4)) || a.err == nullptr ||
        (a.card != 0 && a.card != 1) ||
        (a.carry_prev != nullptr && (!aligned(a.carry_prev, 16) || !aligned(a.carry, 16) || a.hsrc == nullptr))) {
        return false;
    }
    recv_kernel<<<rows, recv_threads, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

namespace detail {

bool in_resources(int32_t which, resources & r) {
    cudaFuncAttributes fa = {};
    if (cudaFuncGetAttributes(&fa, which == 0 ? reinterpret_cast<const void *>(send_kernel)
                                              : reinterpret_cast<const void *>(recv_kernel)) != cudaSuccess) {
        return false;
    }
    int n = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &n, which == 0 ? reinterpret_cast<const void *>(send_kernel) : reinterpret_cast<const void *>(recv_kernel),
            which == 0 ? send_threads : recv_threads, 0) != cudaSuccess) {
        return false;
    }
    r.regs = fa.numRegs;
    r.local_bytes = static_cast<int32_t>(fa.localSizeBytes);
    r.shared_bytes = static_cast<int32_t>(fa.sharedSizeBytes);
    r.ctas_per_sm = n;
    return true;
}

} // namespace detail
} // namespace fn::draft
