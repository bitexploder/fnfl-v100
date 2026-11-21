#pragma once

// IQ3S: the IQ4_XS token_embd table (the Swift IQ3_S file's), decoded as ggml's dequantize_row_iq4_xs does:
// block_iq4_xs = { FP16 d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128] } (136 bytes, 256 values). Sub-block ib
// (32 values) has the 6-bit scale ls = (scales_l[ib/2] >> 4*(ib%2) & 0xf) | ((scales_h >> 2*ib & 3) << 4), dl =
// d * (ls - 32) (exact: an FP16 times an integer below 2^6), and value j < 16 is dl * kvalues_iq4nl[qs[16 ib + j] &
// 0xf], value 16 + j is dl * kvalues_iq4nl[qs[16 ib + j] >> 4], each one rounded product. A card's row is 5 blocks
// (680 bytes).

#include <cuda_fp16.h>

#include <cstdint>

namespace fn::embed {

constexpr int32_t iq4xs_super_bytes = 136;
constexpr int32_t iq4xs_row_bytes   = 5 * iq4xs_super_bytes;  // a card's 1,280 features

static __constant__ float iq4xs_kvalues[16] = { -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
                                                1.0f,    13.0f,   25.0f,  38.0f,  53.0f,  69.0f,  89.0f,  113.0f };

// quarter q (0..3) of the block at blk: sub-blocks 2q and 2q + 1, 64 values in order, as 16 float4. blk is 2-byte
// aligned (GGUF rows of 136-byte blocks), so every load is 16-bit
__device__ __forceinline__ void iq4xs_quarter(const uint8_t * blk, int32_t q, float4 (&o)[16]) {
    const auto  ld = [](const uint8_t * p) { return uint32_t(__ldg(reinterpret_cast<const unsigned short *>(p))); };
    const float d  = __half2float(__ushort_as_half(static_cast<unsigned short>(ld(blk))));
    const uint32_t sh = ld(blk + 2);
    const uint32_t sl = __ldg(blk + 4 + q);  // scales_l[ib / 2], ib = 2q + h
#pragma unroll
    for (int32_t h = 0; h < 2; ++h) {
        const int32_t   ib = 2 * q + h;
        const int32_t   ls = int32_t((sl >> (4 * h)) & 0xfu) | int32_t(((sh >> (2 * ib)) & 3u) << 4);
        const float     dl = __fmul_rn(d, static_cast<float>(ls - 32));
        const uint8_t * qs = blk + 8 + 16 * ib;
        float           v[32];
#pragma unroll
        for (int32_t j = 0; j < 16; j += 2) {
            const uint32_t w = ld(qs + j);  // bytes j, j + 1
#pragma unroll
            for (int32_t i = 0; i < 2; ++i) {
                const uint32_t b = (w >> (8 * i)) & 0xffu;
                v[j + i]      = __fmul_rn(dl, iq4xs_kvalues[b & 0xfu]);
                v[16 + j + i] = __fmul_rn(dl, iq4xs_kvalues[b >> 4]);
            }
        }
#pragma unroll
        for (int32_t k = 0; k < 8; ++k) {
            o[8 * h + k] = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
        }
    }
}

} // namespace fn::embed
