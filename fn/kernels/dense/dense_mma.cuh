#pragma once

// FMS2 build item 4 (W2): the m8n8k4 tensor-core helpers of the joint quantized product (dense.cu). One mma.sync
// m8n8k4 is four independent 8 x 8 x 4 products, one a quadpair: quadpair p is lanes 4p .. 4p + 3 and 4p + 16 .. 4p
// + 19. A lane holds row ra of its quadpair's A (row-major, k 0..3) and column ra of its B (column-major, k 0..3),
// ra = (lane & 3) | ((lane >> 2) & 4); its eight F32 accumulators are elements (c_row(lane, i), c_col(lane, i)).
// Integer operands up to 2,048 in magnitude are exact in FP16, their products exact in F32, and integer sums below
// 2^24 exact.

#include <cuda_fp16.h>

#include <cstdint>

namespace fn::dense::mma {

// a - b on the word's two FP16 halves
__device__ __forceinline__ uint32_t hsub2(uint32_t a, uint32_t b) {
    const half2 r = __hsub2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r); // reinterpret: half2 is the word's two halves
}

// a b and a b + c on the words' two FP16 halves (FMS6: one rounding each, subnormals kept)
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    const half2 r = __hmul2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b));
    return *reinterpret_cast<const uint32_t *>(&r); // reinterpret: half2 is the word's two halves
}
__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
    const half2 r = __hfma2(*reinterpret_cast<const half2 *>(&a), *reinterpret_cast<const half2 *>(&b),
                            *reinterpret_cast<const half2 *>(&c));
    return *reinterpret_cast<const uint32_t *>(&r); // reinterpret: half2 is the word's two halves
}

// f rounded to nearest FP16, in both halves of the word
__device__ __forceinline__ uint32_t f16x2(float f) {
    const half2 r = __float2half2_rn(f);
    return *reinterpret_cast<const uint32_t *>(&r); // reinterpret: half2 is the word's two halves
}

// D += A B, one m8n8k4 product a quadpair (HMMA.884, F32 accumulators)
__device__ __forceinline__ void mma884(float (&d)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// 16 bytes of shared memory, asm volatile so the load stays beside its products (ptxas otherwise reads every n-tile's
// fragments ahead and runs out of registers)
__device__ __forceinline__ uint4 lds128(const void * p) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
    return v;
}

// 4 bytes of shared memory, asm volatile for the same reason (the products' F32 tails)
__device__ __forceinline__ uint32_t lds32(const void * p) {
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
    return v;
}

// 8 bytes of shared memory, asm volatile for the same reason (the products' F32 tails)
__device__ __forceinline__ float2 lds64f(const void * p) {
    float2 v;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];"
                 : "=f"(v.x), "=f"(v.y)
                 : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
    return v;
}

// the A row (and B column) a lane holds in its quadpair
__device__ __forceinline__ int32_t frag_row(int32_t lane) {
    return (lane & 3) | ((lane >> 2) & 4);
}

// the lane of this lane's quadpair holding A row (and B column) r
__device__ __forceinline__ int32_t frag_lane(int32_t lane, int32_t r) {
    return (r & 3) | (lane & 12) | ((r & 4) << 2);
}

// accumulator i's row and column in the quadpair's 8 x 8 block
__device__ __forceinline__ int32_t c_row(int32_t lane, int32_t i) {
    return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}
__device__ __forceinline__ int32_t c_col(int32_t lane, int32_t i) {
    return (i & 1) | (lane & 2) | (i & 4);
}

// the bytes of w (k order) as two FP16 pairs, 1,024 + byte: k 0, 1 in lo, k 2, 3 in hi
__device__ __forceinline__ void biased16(uint32_t w, uint32_t & lo, uint32_t & hi) {
    lo = __byte_perm(w, 0x64646464u, 0x5140u);
    hi = __byte_perm(w, 0x64646464u, 0x5342u);
}

// the same less bias2 (each half 1,024 + the bias of the codes), exact
__device__ __forceinline__ void codes16(uint32_t w, uint32_t bias2, uint32_t & lo, uint32_t & hi) {
    biased16(w, lo, hi);
    lo = hsub2(lo, bias2);
    hi = hsub2(hi, bias2);
}

} // namespace fn::dense::mma
