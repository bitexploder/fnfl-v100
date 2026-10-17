// R2: the layer exchanges (X-A, X-F, X-ROUTE, X-SELECT). The protocol, layouts and payload rule are in xchg.h.
//
// Contract (every kernel): rows B in 1..max_rows; X-SELECT's body 2 B .. B * select_row_words words, its length read on
// the device from the offsets (TG1; a length outside that range is reported, err_select_body); every slot,
// partial and body pointer 16-byte aligned device memory (slots on this card, peer slots in the peer's memory with
// peer access on); anything else is rejected (false, nothing enqueued). They move bits only: no arithmetic except
// recv_add's one F32 add a word (the sum of the two partials; commutative, so card order does not change it). Built
// without fast math (fn/CMakeLists.txt); a NaN payload word is canonicalized, nothing else changes. Polls are
// bounded (poll_budget) and report through the card's error word.

#include "fn/xchg/xchg.cuh"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::xchg {
namespace {

constexpr int32_t half_words = words_half;    // 1280
constexpr int32_t row_words = 2 * half_words; // a partial row, 2560
constexpr int32_t select_recv_threads = (select_words(max_rows * select_row_words) / 4 + 31) / 32 * 32; // 672
// the header's 8 words hold offsets[0 .. rows]; word 7 is offsets[rows] (the receiver's size) for rows <= 7
static_assert(max_rows + 1 <= select_header_words, "X-SELECT's header holds the offsets");

// select_words on the device: the header, then the body rounded up to a 16-byte group
__device__ __forceinline__ int32_t message_words(int32_t body_words) {
    return select_header_words + (body_words + 3) / 4 * 4;
}

bool al16(const void * p) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

// one thread a 16-byte word of the B x 1280 payload
__global__ void __launch_bounds__(128) send_half_kernel(const float * __restrict__ partial, int32_t peer_card,
                                                        uint32_t * peer, int32_t n4) {
    const int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n4) {
        return;
    }
    const int32_t b = u / (half_words / 4);
    const int32_t q = u % (half_words / 4);
    const float4 v = *reinterpret_cast<const float4 *>(partial + b * row_words + peer_card * half_words + 4 * q);
    const uint4 w = make_uint4(canonical(v.x), canonical(v.y), canonical(v.z), canonical(v.w));
    *reinterpret_cast<uint4 *>(peer + b * half_words + 4 * q) = w;
}

__global__ void __launch_bounds__(128) recv_add_kernel(const float * __restrict__ partial, int32_t card,
                                                       uint32_t * slot, float * __restrict__ out, uint32_t * err,
                                                       uint32_t code, int32_t n4) {
    const int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n4) {
        return;
    }
    const int32_t b = u / (half_words / 4);
    const int32_t q = u % (half_words / 4);
    const uint4 * p[1] = {reinterpret_cast<const uint4 *>(slot) + u};
    uint4 v[1];
    poll<1>(p, v, err, code);
    reset4(reinterpret_cast<uint4 *>(slot) + u);
    const float4 own = *reinterpret_cast<const float4 *>(partial + b * row_words + card * half_words + 4 * q);
    float4 o;
    o.x = own.x + __uint_as_float(v[0].x);
    o.y = own.y + __uint_as_float(v[0].y);
    o.z = own.z + __uint_as_float(v[0].z);
    o.w = own.w + __uint_as_float(v[0].w);
    *reinterpret_cast<float4 *>(out + b * half_words + 4 * q) = o;
}

// one warp a row, lane l word l (l < 24)
__global__ void __launch_bounds__(32 * max_rows) send_route_kernel(const int32_t * __restrict__ ids_full,
                                                                   int32_t ids_stride, const float * __restrict__ wts,
                                                                   const float * __restrict__ sg,
                                                                   int32_t * __restrict__ ids_local, uint32_t * peer) {
    const int32_t b = threadIdx.x / 32;
    const int32_t l = threadIdx.x % 32;
    if (l >= words_route) {
        return;
    }
    uint32_t w = 0u;
    if (l < 10) {
        const int32_t id = ids_full[b * ids_stride + l];
        ids_local[b * 10 + l] = id;
        w = static_cast<uint32_t>(id);
    } else if (l < 20) {
        w = canonical(wts[b * 10 + (l - 10)]);
    } else if (l == 20) {
        w = canonical(sg[b]);
    }
    peer[b * words_route + l] = w;
}

// one thread a 16-byte word of the B x 24 payload
__global__ void __launch_bounds__(32) recv_route_kernel(uint32_t * slot, int32_t * __restrict__ ids,
                                                        float * __restrict__ wts, float * __restrict__ sg,
                                                        uint32_t * err, int32_t n4) {
    const int32_t u = threadIdx.x;
    if (u >= n4) {
        return;
    }
    const uint4 * p[1] = {reinterpret_cast<const uint4 *>(slot) + u};
    uint4 v[1];
    poll<1>(p, v, err, err_poll_route);
    reset4(reinterpret_cast<uint4 *>(slot) + u);
    const uint32_t x[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
    const int32_t b = u / (words_route / 4);
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
        const int32_t w = 4 * (u % (words_route / 4)) + j;
        if (w < 10) {
            ids[b * 10 + w] = static_cast<int32_t>(x[j]);
        } else if (w < 20) {
            wts[b * 10 + (w - 10)] = __uint_as_float(x[j]);
        } else if (w == 20) {
            sg[b] = __uint_as_float(x[j]);
        }
    }
}

// one thread a 16-byte word of the message: the header (u 0 and 1), then the body; the body's last group is read a word
// at a time, so nothing past the body is read. The body's length is offsets[rows] (bytes), read on the device (TG1): the
// grid covers the longest body of B rows and the threads past this body's message send nothing.
__global__ void __launch_bounds__(128) send_select_kernel(int32_t rows, const int32_t * __restrict__ offsets,
                                                          const uint32_t * __restrict__ body, uint32_t * peer) {
    const int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total = offsets[rows];
    // an out-of-contract size is sent as it is in the header (the receiver reports it) with no body
    const int32_t body_words = total % 4 == 0 && total >= 8 * rows && total <= 4 * rows * select_row_words ? total / 4
                                                                                                           : 0;
    if (u >= message_words(body_words) / 4) {
        return;
    }
    uint32_t x[4];
    const int32_t k = 4 * (u - select_header_words / 4);
    if (u < select_header_words / 4) {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            x[j] = static_cast<uint32_t>(offsets[min(4 * u + j, rows)]);
        }
    } else if (k + 4 <= body_words) {
        const uint4 v = *reinterpret_cast<const uint4 *>(body + k);
        x[0] = v.x;
        x[1] = v.y;
        x[2] = v.z;
        x[3] = v.w;
    } else {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            x[j] = k + j < body_words ? body[k + j] : 0u;
        }
    }
    *reinterpret_cast<uint4 *>(peer + 4 * u) = make_uint4(x[0], x[1], x[2], x[3]);
}

// One block, one thread a 16-byte word of the longest message of B rows. Thread 1 polls header group 1 first: its words
// are offsets[min(t, rows)], t = 4 .. 7, so word 7 holds the body's total bytes (SP2: at B = 5 word 4 is row 4's
// offset; at B <= 4 all four hold the total). It resets the group and shares the size; then each thread whose word is
// in this message polls it, resets it and writes the offsets or body words. Threads past the message poll nothing
// (their words were not written and stay the sentinel).
__global__ void __launch_bounds__(select_recv_threads) recv_select_kernel(int32_t rows, uint32_t * slot,
                                                                          int32_t * __restrict__ offsets,
                                                                          uint32_t * __restrict__ body,
                                                                          uint32_t * err) {
    __shared__ int32_t s_body_words;
    const int32_t u = threadIdx.x;
    uint4 v[1];
    if (u == 1) {
        const uint4 * p[1] = {reinterpret_cast<const uint4 *>(slot) + 1};
        poll<1>(p, v, err, err_poll_select);
        reset4(reinterpret_cast<uint4 *>(slot) + 1);
        const int32_t total = static_cast<int32_t>(v[0].w); // offsets[min(7, rows)] = offsets[rows]
        const bool ok = total % 4 == 0 && total >= 8 * rows && total <= 4 * rows * select_row_words;
        if (!ok) {
            atomicOr(err, err_select_body);
        }
        s_body_words = ok ? total / 4 : -1;
        const uint32_t x[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            if (ok && 4 + j <= rows) {
                offsets[4 + j] = static_cast<int32_t>(x[j]);
            }
        }
    }
    __syncthreads();
    const int32_t body_words = s_body_words;
    if (body_words < 0 || u == 1 || u >= message_words(body_words) / 4) {
        return;
    }
    const uint4 * p[1] = {reinterpret_cast<const uint4 *>(slot) + u};
    poll<1>(p, v, err, err_poll_select);
    reset4(reinterpret_cast<uint4 *>(slot) + u);
    const uint32_t x[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
    const int32_t k = 4 * (u - select_header_words / 4);
    if (u < select_header_words / 4) {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            if (4 * u + j <= rows) {
                offsets[4 * u + j] = static_cast<int32_t>(x[j]);
            }
        }
    } else if (k + 4 <= body_words) {
        *reinterpret_cast<uint4 *>(body + k) = v[0];
    } else {
#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            if (k + j < body_words) {
                body[k + j] = x[j];
            }
        }
    }
}

bool rows_ok(int32_t rows) {
    return rows >= 1 && rows <= max_rows;
}

bool launched() {
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

bool send_half(int32_t rows, int32_t peer_card, const float * partial, float * peer_slot, void * stream) {
    if (!rows_ok(rows) || peer_card < 0 || peer_card > 1 || !al16(partial) || !al16(peer_slot) || stream == nullptr) {
        return false;
    }
    const int32_t n4 = rows * half_words / 4;
    send_half_kernel<<<(n4 + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        partial, peer_card, reinterpret_cast<uint32_t *>(peer_slot), n4); // F32 payload words handled as bits
    return launched();
}

bool recv_add(int32_t rows, int32_t card, const float * partial, float * slot, float * out, uint32_t * err,
              uint32_t code, void * stream) {
    if (!rows_ok(rows) || card < 0 || card > 1 || !al16(partial) || !al16(slot) || !al16(out) || err == nullptr ||
        stream == nullptr) {
        return false;
    }
    const int32_t n4 = rows * half_words / 4;
    recv_add_kernel<<<(n4 + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        partial, card, reinterpret_cast<uint32_t *>(slot), out, err, code, n4); // F32 payload words handled as bits
    return launched();
}

bool send_route(int32_t rows, const int32_t * ids_full, int32_t ids_stride, const float * wts, const float * sg,
                int32_t * ids_local, float * peer_slot, void * stream) {
    if (!rows_ok(rows) || ids_full == nullptr || ids_stride < 10 || wts == nullptr || sg == nullptr ||
        ids_local == nullptr || !al16(peer_slot) || stream == nullptr) {
        return false;
    }
    send_route_kernel<<<1, 32 * rows, 0, static_cast<cudaStream_t>(stream)>>>(
        ids_full, ids_stride, wts, sg, ids_local, reinterpret_cast<uint32_t *>(peer_slot)); // words handled as bits
    return launched();
}

bool recv_route(int32_t rows, float * slot, int32_t * ids, float * wts, float * sg, uint32_t * err, void * stream) {
    if (!rows_ok(rows) || !al16(slot) || ids == nullptr || wts == nullptr || sg == nullptr || err == nullptr ||
        stream == nullptr) {
        return false;
    }
    recv_route_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<uint32_t *>(slot), ids, wts, sg, err, rows * words_route / 4); // words handled as bits
    return launched();
}

bool send_select(int32_t rows, const int32_t * offsets, const uint32_t * body, float * peer_slot, void * stream) {
    if (!rows_ok(rows) || offsets == nullptr || !al16(body) || !al16(peer_slot) || stream == nullptr) {
        return false;
    }
    const int32_t n4 = select_words(rows * select_row_words) / 4; // the longest message of B rows
    send_select_kernel<<<(n4 + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        rows, offsets, body, reinterpret_cast<uint32_t *>(peer_slot)); // words handled as bits
    return launched();
}

bool recv_select(int32_t rows, float * slot, int32_t * offsets, uint32_t * body, uint32_t * err, void * stream) {
    if (!rows_ok(rows) || !al16(slot) || offsets == nullptr || !al16(body) || err == nullptr || stream == nullptr) {
        return false;
    }
    const int32_t n4 = select_words(rows * select_row_words) / 4; // the longest message of B rows
    recv_select_kernel<<<1, (n4 + 31) / 32 * 32, 0, static_cast<cudaStream_t>(stream)>>>(
        rows, reinterpret_cast<uint32_t *>(slot), offsets, body, err); // words handled as bits
    return launched();
}

} // namespace fn::xchg
