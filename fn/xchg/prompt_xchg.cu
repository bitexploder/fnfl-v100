#include "fn/xchg/prompt_xchg.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace fn::xchg {
namespace {

constexpr int32_t D  = 2560;
constexpr int32_t v8 = D / 8;  // 8-value groups a row

__device__ __forceinline__ bool header_ok(const uint8_t * slot, uint64_t gen) {
    const volatile uint32_t * h = reinterpret_cast<const volatile uint32_t *>(slot);
    return h[0] == static_cast<uint32_t>(gen) && h[1] == static_cast<uint32_t>(gen >> 32);
}

__device__ __forceinline__ void put_header(uint8_t * slot, uint64_t gen, int32_t rows, int32_t kind) {
    *reinterpret_cast<uint4 *>(slot) = make_uint4(static_cast<uint32_t>(gen), static_cast<uint32_t>(gen >> 32),
                                                  static_cast<uint32_t>(rows), static_cast<uint32_t>(kind));
}

// a thread per 8 values; the header follows in a second launch (pstamp's kernel), so every payload store is complete
// at the kernel boundary before the header exists
__global__ void ppack_kernel(int32_t rows, const float * __restrict__ partial, uint8_t * slot, uint32_t * err) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(rows) * v8) {
        return;
    }
    const float4 a    = __ldg(reinterpret_cast<const float4 *>(partial) + 2 * i);
    const float4 b    = __ldg(reinterpret_cast<const float4 *>(partial) + 2 * i + 1);
    const float  v[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
    uint32_t     h[4];
    bool         bad = false;
#pragma unroll
    for (int32_t k = 0; k < 4; ++k) {
        bad  = bad || !(fabsf(v[2 * k]) <= 65504.0f) || !(fabsf(v[2 * k + 1]) <= 65504.0f);
        h[k] = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(v[2 * k]))) |
               (static_cast<uint32_t>(__half_as_ushort(__float2half_rn(v[2 * k + 1]))) << 16);
    }
    if (bad) {
        atomicOr(err, err_prange);
    }
    reinterpret_cast<uint4 *>(slot + pmsg_payload_at)[i] = make_uint4(h[0], h[1], h[2], h[3]);
}

__global__ void pstamp_kernel(uint8_t * slot, uint64_t gen, int32_t rows, int32_t kind) {
    put_header(slot, gen, rows, kind);
}

__global__ void pcheck_kernel(const uint8_t * slot, uint64_t gen, uint32_t * err) {
    if (!header_ok(slot, gen)) {
        atomicOr(err, err_pmsg);
    }
}

__device__ __forceinline__ void h8(const uint4 & u, float (&o)[8]) {
    const uint32_t w[4] = { u.x, u.y, u.z, u.w };
#pragma unroll
    for (int32_t k = 0; k < 4; ++k) {
        o[2 * k]     = __half2float(__ushort_as_half(static_cast<unsigned short>(w[k] & 0xffffu)));
        o[2 * k + 1] = __half2float(__ushort_as_half(static_cast<unsigned short>(w[k] >> 16)));
    }
}

__global__ void pcombine_kernel(int32_t         rows,
                                int32_t         card,
                                const uint8_t * own,
                                const uint8_t * peer,
                                uint64_t        gen,
                                const float * __restrict__ inject,
                                float *    streams,
                                uint32_t * err) {
    const bool    ok = header_ok(own, gen) && header_ok(peer, gen);
    const int64_t i  = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (!ok) {
        if (threadIdx.x == 0) {
            atomicOr(err, err_pmsg);
        }
        return;
    }
    if (i >= int64_t(rows) * v8) {
        return;
    }
    const int64_t r = i / v8, g = i % v8;
    float         o[8], p[8];
    h8(__ldg(reinterpret_cast<const uint4 *>(own + pmsg_payload_at) + i), o);
    h8(__ldg(reinterpret_cast<const uint4 *>(peer + pmsg_payload_at) + i), p);
    float out[8];
#pragma unroll
    for (int32_t k = 0; k < 8; ++k) {
        out[k] = card == 0 ? o[k] + p[k] : p[k] + o[k];
    }
    const float4 inj   = __ldg(reinterpret_cast<const float4 *>(inject) + r);
    const float  iv[4] = { inj.x, inj.y, inj.z, inj.w };
#pragma unroll
    for (int32_t s = 0; s < 4; ++s) {
        float4 * sp = reinterpret_cast<float4 *>(streams + (r * 4 + s) * D) + 2 * g;
        float4   a = sp[0], b = sp[1];
        a.x   = fmaf(iv[s], out[0], a.x);
        a.y   = fmaf(iv[s], out[1], a.y);
        a.z   = fmaf(iv[s], out[2], a.z);
        a.w   = fmaf(iv[s], out[3], a.w);
        b.x   = fmaf(iv[s], out[4], b.x);
        b.y   = fmaf(iv[s], out[5], b.y);
        b.z   = fmaf(iv[s], out[6], b.z);
        b.w   = fmaf(iv[s], out[7], b.w);
        sp[0] = a;
        sp[1] = b;
    }
}

bool al16(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

uint32_t blocks_of(int64_t n) {
    return static_cast<uint32_t>((n + 255) / 256);
}

}  // namespace

uint64_t pgen(int64_t chunk, int32_t layer, int32_t sub, int32_t half, pkind kind) {
    return (static_cast<uint64_t>(chunk & 0xffffff) << 24) | (static_cast<uint64_t>(layer & 0x3f) << 16) |
           (static_cast<uint64_t>(sub & 1) << 12) | (static_cast<uint64_t>(half & 1) << 8) |
           static_cast<uint64_t>(static_cast<int32_t>(kind) & 0xff) | (uint64_t(1) << 63);
}

bool ppack(int32_t rows, const float * partial, uint8_t * slot, uint64_t gen, uint32_t * err, void * stream) {
    if (rows < 1 || !al16(partial) || !al16(slot) || err == nullptr || stream == nullptr) {
        fprintf(stderr, "ppack: rejected\n");
        return false;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    ppack_kernel<<<blocks_of(int64_t(rows) * v8), 256, 0, s>>>(rows, partial, slot, err);
    pstamp_kernel<<<1, 1, 0, s>>>(slot, gen, rows, 0);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pseal(int32_t rows, uint8_t * slot, uint64_t gen, void * stream) {
    if (rows < 1 || !al16(slot) || stream == nullptr) {
        fprintf(stderr, "pseal: rejected\n");
        return false;
    }
    pstamp_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(slot, gen, rows, 0);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pcombine(int32_t         rows,
              int32_t         card,
              const uint8_t * own_slot,
              const uint8_t * peer_slot,
              uint64_t        gen,
              const float *   inject,
              float *         streams,
              uint32_t *      err,
              void *          stream) {
    if (rows < 1 || (card != 0 && card != 1) || !al16(own_slot) || !al16(peer_slot) || !al16(inject) ||
        !al16(streams) || err == nullptr || stream == nullptr) {
        fprintf(stderr, "pcombine: rejected\n");
        return false;
    }
    pcombine_kernel<<<blocks_of(int64_t(rows) * v8), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        rows, card, own_slot, peer_slot, gen, inject, streams, err);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pstamp(uint8_t * slot, uint64_t gen, int32_t rows, pkind kind, void * stream) {
    if (!al16(slot) || stream == nullptr) {
        return false;
    }
    pstamp_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(slot, gen, rows, static_cast<int32_t>(kind));
    return cudaPeekAtLastError() == cudaSuccess;
}

bool pcheck(const uint8_t * slot, uint64_t gen, uint32_t * err, void * stream) {
    if (!al16(slot) || err == nullptr || stream == nullptr) {
        return false;
    }
    pcheck_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(slot, gen, err);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace fn::xchg
