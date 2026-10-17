#pragma once

// IQ3_S decode of one lane's quarter superblock: DP2's decode (H5's one-PRMT grid index), moved here from dense.cu so the
// token embedding (E1, fn/kernels/embed) runs the same device code. A quarter q of a 256-weight superblock is its
// sub-blocks 2q and 2q + 1 (weights 64q .. 64q + 63); its GGUF bytes are qs[16q, +16), signs[8q, +8), qh[2q, +2),
// scales[q] and the superblock's d (block_iq3_s: d at 0, qs at 2, qh at 66, signs at 74, scales at 106).
//
// Output: w[4c + i] holds the int8 weights 64q + 16c + 4i .. + 3 (byte j: weight + j), the grid value with its sign;
// f0 and f1 are sub-blocks 2q and 2q + 1's scales d (1 + 2 scale nibble). A weight's F32 value is f x float(int8), one
// rounded product, bitwise fn/quant's decode_iq3_s (db x grid x sign: the sign flip is exact and round to nearest is
// symmetric).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace fn::dense {

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}

// 0xff in each byte whose MSB is set (PRMT's sign-replicate selector)
__device__ __forceinline__ uint32_t msb_mask(uint32_t x) {
    return prmt(x, 0, 0xBA98u);
}
// 0xff in byte i where bit i (lo) or bit 4 + i (hi) of the sign byte s is set (as the experts' kernels)
__device__ __forceinline__ uint32_t sign_lo(uint32_t s) {
    return msb_mask((s & 0x0fu) * 0x10204080u);
}
__device__ __forceinline__ uint32_t sign_hi(uint32_t s) {
    return msb_mask((s & 0xf0u) * 0x01020408u);
}

__device__ __forceinline__ float half_lo(uint32_t h2) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(h2 & 0xffffu)));
}
__device__ __forceinline__ uint32_t word(const uint4 & v, int32_t i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// a quarter's raw bytes: qs[16q, +16), signs[8q, +8), qh[2q, +2) (low 16 bits), scales[q] (low 8 bits), d (low 16 bits)
struct iq3s_quarter {
    uint4 qs;
    uint2 sg;
    uint32_t qh, sc, d;
};

// grid: iq3s_grid's 512 words
__device__ __forceinline__ void iq3s_decode(const iq3s_quarter & v, const uint32_t * grid, uint32_t (&w)[16],
                                            float & f0, float & f1) {
    // sub-block j: qs bytes 8j + t give weights 4t .. 4t + 3 (grid index qs | qh bit t << 8); signs byte t / 2,
    // nibble t % 2. The index is one PRMT (DP2 H5): qh bit t is spread to byte t % 4 of hb[t / 4] (0 or 1), and
    // the PRMT takes qs byte t % 4 to byte 0, that hb byte to byte 1, and its sign (0) to bytes 2 and 3. The shift
    // and merge it replaces cost about 5 instructions a lookup.
    uint32_t hb[4];
#pragma unroll
    for (int32_t k = 0; k < 4; ++k) {
        hb[k] = ((v.qh >> (4 * k)) & 0xfu) * 0x00204081u & 0x01010101u; // bit k to 8k; no cross term lands there
    }
#pragma unroll
    for (int32_t t = 0; t < 16; ++t) {
        const uint32_t i = t & 3; // selectors i, 4 + i, then 0xc + i twice (sign mode)
        const uint32_t idx = prmt(word(v.qs, t >> 2), hb[t >> 2], 0xcc40u + 0x1111u * i);
        const uint32_t g = grid[idx];
        const uint32_t sgw = t < 8 ? v.sg.x : v.sg.y;
        const uint32_t sb = (sgw >> (8 * ((t & 7) >> 1))) & 0xffu;
        const uint32_t m = (t & 1) ? sign_hi(sb) : sign_lo(sb);
        w[t] = (g ^ m) + (m & 0x01010101u); // grid bytes are 1..15, so no byte carries
    }
    const float d = half_lo(v.d);
    f0 = d * static_cast<float>(1 + 2 * static_cast<int32_t>(v.sc & 0xfu));
    f1 = d * static_cast<float>(1 + 2 * static_cast<int32_t>((v.sc >> 4) & 0xfu));
}

} // namespace fn::dense
