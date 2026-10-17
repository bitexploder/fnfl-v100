#include "fn/kernels/ple/ple.h"

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/round/version.h"
#include "fn/xchg/xchg.cuh"
#include "fn/xchg/xchg.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace fn::ple {
namespace {

constexpr int32_t stats_words = fn::xchg::words_ple; // 16: 13 statistics and 3 zero pad words a row
constexpr int64_t slot_floats = fn::xchg::slot_words(fn::xchg::msg::ple); // rings::slot spacing of X-PLE-STATS
constexpr int32_t stats_threads = 256;               // 8 warps; 5 features a thread
constexpr int32_t stats_warps = stats_threads / 32;
constexpr int32_t stats_per_thread = n_own / stats_threads;
constexpr int32_t update_threads = 64;
constexpr int32_t update_blocks = n_channel / update_threads; // 80
constexpr float eps = 1e-6f;                                  // the model's rms epsilon (fn::rms_eps)
constexpr float n_d = 2560.0f;                                // the norms' width: one stream, D

static_assert(n_own % stats_threads == 0 && n_channel % update_threads == 0, "whole blocks");
static_assert(n_stats <= stats_words && stats_words % 4 == 0, "a row is whole 16-byte words");
static_assert(max_rows * 4 <= update_threads, "one polling lane a 16-byte word");

// the workspace's parts, each 256-byte aligned
constexpr size_t al256(size_t n) {
    return (n + 255) / 256 * 256;
}
constexpr size_t off_key = 0;
constexpr size_t off_value = off_key + al256(size_t(max_rows) * n_channel * 4);
constexpr size_t off_stats = off_value + al256(size_t(max_rows) * n_own * 4);
constexpr size_t off_scal = off_stats + al256(size_t(max_rows) * stats_words * 4);
constexpr size_t off_work = off_scal + al256(size_t(max_rows) * n_stream * 4 * 4);

bool al16(const void * p) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

fn::dense::product_args product_args_of(const step_args * a, uint8_t * work) {
    fn::dense::product_args p;
    p.k = n_in;
    p.n_mats = 2;
    p.mats[0].f = fn::dense::fmt::bf16;
    p.mats[0].m = n_channel;
    p.mats[1].f = fn::dense::fmt::bf16;
    p.mats[1].m = n_own;
    if (a != nullptr) {
        const ws_view v = carve(a->ws);
        p.mats[0].w = a->w.key;
        p.mats[0].out = v.key;
        p.mats[1].w = a->w.value;
        p.mats[1].out = v.value;
        p.x_f32 = a->e;
    }
    fn::dense::choose_splits(p);
    p.work = work;
    return p;
}

size_t product_work_bytes() {
    return fn::dense::work_bytes(product_args_of(nullptr, nullptr));
}

// ---- stats --------------------------------------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

// block b: row b. Thread t sums features t + 256 j (j = 0..4) in order, then the warp's shuffle-xor tree, then
// thread w < 13 of warp 0 sums the 8 warps in order. Warp 0 sends the row (16 words) to the peer and its own copy.
__global__ void __launch_bounds__(stats_threads)
stats_kernel(const float * __restrict__ key, const float * __restrict__ value, const float * __restrict__ hres,
             const float * __restrict__ hpost, const float * __restrict__ hx, const float * __restrict__ norm_key,
             const float * __restrict__ norm_query, float * own, float * peer, float * reset, float * own_ring,
             const uint64_t * use) {
    __shared__ float part[stats_warps][n_stats];
    const int32_t b = blockIdx.x;
    const int32_t t = threadIdx.x;
    if (use != nullptr) {
        // the slots of use u from the device counter (TG1): send into slot u % 3, reset this card's slot of use u - 1
        const uint64_t u = *use;
        peer += (u % fn::xchg::n_slot) * slot_floats;
        reset = own_ring + ((u + fn::xchg::n_slot - 1) % fn::xchg::n_slot) * slot_floats;
    }
    if (reset != nullptr && b == 0 && t < max_rows * stats_words / 4) {
        fn::xchg::reset4(reinterpret_cast<uint4 *>(reset) + t); // the previous round's slot, after its readers
    }
    float acc[n_stats];
#pragma unroll
    for (int32_t j = 0; j < n_stats; ++j) {
        acc[j] = 0.0f;
    }
    float hp[n_stream];
#pragma unroll
    for (int32_t s = 0; s < n_stream; ++s) {
        hp[s] = hpost[b * n_stream + s];
    }
#pragma unroll
    for (int32_t j = 0; j < stats_per_thread; ++j) {
        const int32_t i = t + stats_threads * j;
        const float xv = hx[b * n_own + i];
        const float v = value[b * n_own + i];
        acc[12] = fmaf(v, v, acc[12]);
#pragma unroll
        for (int32_t s = 0; s < n_stream; ++s) {
            const float k = key[b * n_channel + s * n_own + i];
            const float h = fmaf(hp[s], xv, hres[(b * n_stream + s) * n_own + i]); // FC-LAY1's combine
            acc[s] = fmaf(k, k, acc[s]);
            acc[4 + s] = fmaf(h, h, acc[4 + s]);
            acc[8 + s] = fmaf(k * norm_key[s * n_own + i], h * norm_query[s * n_own + i], acc[8 + s]);
        }
    }
    const int32_t warp = t / 32, lane = t % 32;
#pragma unroll
    for (int32_t j = 0; j < n_stats; ++j) {
        acc[j] = warp_sum(acc[j]);
    }
    if (lane == 0) {
#pragma unroll
        for (int32_t j = 0; j < n_stats; ++j) {
            part[warp][j] = acc[j];
        }
    }
    __syncthreads();
    if (warp == 0) {
        float sum = 0.0f;
        if (lane < n_stats) {
            sum = part[0][lane];
#pragma unroll
            for (int32_t w = 1; w < stats_warps; ++w) {
                sum += part[w][lane];
            }
        }
        // lanes 13..15 send the zero pad words
        fn::xchg::put_run(reinterpret_cast<uint32_t *>(own) + b * stats_words,
                          reinterpret_cast<uint32_t *>(peer) + b * stats_words, 0, stats_words, 0,
                          fn::xchg::canonical(sum));
    }
}

// ---- update -------------------------------------------------------------------------------------------------------

struct row_scalars {
    float gate, xinv;
};

// The per-row and stream scalars from the 13 statistics summed in card order (card 0's partial first).
__device__ __forceinline__ row_scalars scalars(const float * own, const float * peer, int32_t card, int32_t s,
                                               float * diag) {
    float tot[4];
    const int32_t idx[4] = {s, 4 + s, 8 + s, 12};
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
        const float a = own[idx[j]], p = peer[idx[j]];
        tot[j] = card == 0 ? a + p : p + a;
    }
    const float rk = 1.0f / sqrtf(tot[0] / n_d + eps);
    const float rq = 1.0f / sqrtf(tot[1] / n_d + eps);
    const float score = tot[2] * rk * rq * (1.0f / sqrtf(n_d));
    const float mag = sqrtf(fmaxf(fabsf(score), 1e-6f));
    const float z = score > 0.0f ? mag : score < 0.0f ? -mag : 0.0f; // sign(score) mag; sign(0) = 0
    const float gate = 1.0f / (1.0f + expf(-z));
    const float xinv = 1.0f / sqrtf(gate * gate * (tot[3] / n_d) + eps);
    if (diag != nullptr) {
        diag[0] = score;
        diag[1] = gate;
        diag[2] = xinv;
        diag[3] = 0.0f;
    }
    return {gate, xinv};
}

// thread: channel c = 1280 s + i. hres and h_out may be the same buffer (no __restrict__ on either). The block polls the peer's B rows (one lane a 16-byte word), shares them and its own
// rows through shared memory, and computes the B x 4 scalars once.
template <int B>
__global__ void __launch_bounds__(update_threads)
update_kernel(const float * __restrict__ value, const float * hres, const float * __restrict__ hpost,
              const float * __restrict__ hx, const float * __restrict__ norm_conv, const uint16_t * __restrict__ conv,
              const float * __restrict__ own_stats, float * slot, float * h_out, float * __restrict__ hist,
              const uint32_t * cur_word, float * __restrict__ scal, uint32_t * err, int32_t card,
              const uint64_t * use) {
    if (use != nullptr) {
        slot += (*use % fn::xchg::n_slot) * slot_floats; // this card's slot of use u (TG1)
    }
    __shared__ __align__(16) float own_s[B][stats_words];
    __shared__ __align__(16) float peer_s[B][stats_words];
    __shared__ row_scalars rs[B][n_stream];
    const int32_t t = threadIdx.x;
    const int32_t c = blockIdx.x * update_threads + t;
    const int32_t s = c / n_own, i = c % n_own;

    // independent of the exchange: the taps, the history and the norm, read before the poll
    const uint2 wraw = __ldg(reinterpret_cast<const uint2 *>(conv) + c);
    const float w0 = __half2float(__ushort_as_half(static_cast<unsigned short>(wraw.x & 0xffffu)));
    const float w1 = __half2float(__ushort_as_half(static_cast<unsigned short>(wraw.x >> 16)));
    const float w2 = __half2float(__ushort_as_half(static_cast<unsigned short>(wraw.y & 0xffffu)));
    const float w3 = __half2float(__ushort_as_half(static_cast<unsigned short>(wraw.y >> 16)));
    const uint32_t cur = *cur_word; // the committed slot (SR1)
    const float * h0 = hist + fn::round::version_slot(cur, 0) * hist_version_floats + int64_t(c) * n_hist;
    float hv[n_hist];
#pragma unroll
    for (int32_t j = 0; j < n_hist; ++j) {
        hv[j] = h0[j];
    }
    const float nc = __ldg(norm_conv + c);

    if (t < B * 4) {
        const uint4 * p[1] = {reinterpret_cast<const uint4 *>(slot) + t};
        uint4 v[1];
        fn::xchg::poll<1>(p, v, err, fn::xchg::err_poll_ple);
        reinterpret_cast<uint4 *>(&peer_s[0][0])[t] = v[0];
        reinterpret_cast<float4 *>(&own_s[0][0])[t] = reinterpret_cast<const float4 *>(own_stats)[t];
    }
    __syncthreads();
    if (t < B * n_stream) {
        const int32_t b = t / n_stream, st = t % n_stream;
        rs[b][st] = scalars(own_s[b], peer_s[b], card, st, blockIdx.x == 0 ? scal + 4 * t : nullptr);
    }
    __syncthreads();

#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        const row_scalars r = rs[b][s];
        const float gated = value[b * n_own + i] * r.gate;
        const float x = gated * r.xinv * nc;
        float acc = hv[0] * w0;
        acc = fmaf(hv[3], w1, acc);
        acc = fmaf(hv[6], w2, acc);
        acc = fmaf(x, w3, acc);
        const float silu = acc / (1.0f + expf(-acc));
        const int64_t hi = (int64_t(b) * n_stream + s) * n_own + i;
        const float h = fmaf(hpost[b * n_stream + s], hx[b * n_own + i], hres[hi]); // FC-LAY1's combine
        h_out[hi] = h + (gated + silu);
#pragma unroll
        for (int32_t j = 0; j < n_hist - 1; ++j) {
            hv[j] = hv[j + 1];
        }
        hv[n_hist - 1] = x;
        float * dst = hist + fn::round::version_slot(cur, b + 1) * hist_version_floats + int64_t(c) * n_hist;
#pragma unroll
        for (int32_t j = 0; j < n_hist; ++j) {
            dst[j] = hv[j];
        }
    }
}

bool common_ok(int32_t rows, const step_args & a, void * stream) {
    return rows >= 1 && rows <= max_rows && al16(a.ws) && stream != nullptr;
}

bool launched() {
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

size_t workspace_bytes() {
    return off_work + al256(product_work_bytes());
}

bool prepare(uint8_t * ws) {
    return ws != nullptr && cudaMemset(ws, 0, workspace_bytes()) == cudaSuccess;
}

ws_view carve(uint8_t * ws) {
    ws_view v;
    v.key = reinterpret_cast<float *>(ws + off_key);
    v.value = reinterpret_cast<float *>(ws + off_value);
    v.stats = reinterpret_cast<float *>(ws + off_stats);
    v.scal = reinterpret_cast<float *>(ws + off_scal);
    v.work = ws + off_work;
    return v;
}

bool products(int32_t rows, const step_args & a, void * stream) {
    if (!common_ok(rows, a, stream) || !al16(a.w.key) || !al16(a.w.value) || !al16(a.e)) {
        fprintf(stderr, "ple products: rejected (rows %d, or a null or misaligned pointer)\n", rows);
        return false;
    }
    const fn::dense::product_args p = product_args_of(&a, carve(a.ws).work);
    return fn::dense::product(rows, p, stream);
}

bool stats(int32_t rows, const step_args & a, void * stream) {
    if (!common_ok(rows, a, stream) || !al16(a.hres) || !al16(a.hpost) || !al16(a.hx) || !al16(a.w.norm_key) ||
        !al16(a.w.norm_query) || !al16(a.stats_peer) || (a.stats_reset != nullptr && !al16(a.stats_reset)) ||
        (a.use != nullptr && !al16(a.stats_slot))) {
        fprintf(stderr, "ple stats: rejected (rows %d, or a null or misaligned pointer)\n", rows);
        return false;
    }
    const ws_view v = carve(a.ws);
    stats_kernel<<<rows, stats_threads, 0, static_cast<cudaStream_t>(stream)>>>(
        v.key, v.value, a.hres, a.hpost, a.hx, a.w.norm_key, a.w.norm_query, v.stats, a.stats_peer, a.stats_reset,
        a.stats_slot, a.use);
    return launched();
}

bool update(int32_t rows, const step_args & a, void * stream) {
    if (!common_ok(rows, a, stream) || !al16(a.hres) || !al16(a.hpost) || !al16(a.hx) || !al16(a.h_out) ||
        !al16(a.hist) || a.cur == nullptr || !al16(a.w.norm_conv) || !al16(a.w.conv) || !al16(a.stats_slot) ||
        a.err == nullptr ||
        (a.card != 0 && a.card != 1)) {
        fprintf(stderr, "ple update: rejected (rows %d, card %d, or a null or misaligned pointer)\n", rows, a.card);
        return false;
    }
    const ws_view v = carve(a.ws);
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
#define FN_PLE_UPDATE(R)                                                                                               \
    update_kernel<R><<<update_blocks, update_threads, 0, st>>>(v.value, a.hres, a.hpost, a.hx, a.w.norm_conv,          \
                                                               a.w.conv, v.stats, a.stats_slot, a.h_out, a.hist,       \
                                                               a.cur, v.scal, a.err, a.card, a.use)
    switch (rows) {
        case 1:
            FN_PLE_UPDATE(1);
            break;
        case 2:
            FN_PLE_UPDATE(2);
            break;
        case 3:
            FN_PLE_UPDATE(3);
            break;
        case 4:
            FN_PLE_UPDATE(4);
            break;
        default:
            FN_PLE_UPDATE(5); // SP2
            break;
    }
#undef FN_PLE_UPDATE
    return launched();
}

bool step(int32_t rows, const step_args & a, void * stream) {
    return products(rows, a, stream) && stats(rows, a, stream) && update(rows, a, stream);
}

} // namespace fn::ple
