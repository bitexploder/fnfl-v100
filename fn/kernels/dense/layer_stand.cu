#include "fn/kernels/dense/layer_stand.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::dense {
namespace {

// stand_stream's kernel: a grid-stride read of n 16-byte words with op's cache operator (sm_70 PTX); 144 x 512
// threads, as cold_flush
template <stand_op OP>
__global__ void __launch_bounds__(512) stand_kernel(const uint4 * p, int64_t n, uint32_t * sink) {
    uint32_t x = 0u;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        uint4 v;
        if constexpr (OP == stand_op::ef) {
            asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "l"(p + i));
        } else {
            asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "l"(p + i));
        }
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x9e3779b9u) {
        sink[blockIdx.x] = x; // practically never taken; keeps the loads
    }
}

} // namespace

bool stand_stream(const void * p, int64_t bytes, stand_op op, uint32_t * sink, void * stream) {
    if (p == nullptr || bytes <= 0 || bytes % 16 != 0 || (reinterpret_cast<uintptr_t>(p) & 15u) != 0 ||
        sink == nullptr || stream == nullptr) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const uint4 * q = static_cast<const uint4 *>(p);
    if (op == stand_op::ef) {
        stand_kernel<stand_op::ef><<<144, 512, 0, st>>>(q, bytes / 16, sink);
    } else {
        stand_kernel<stand_op::normal><<<144, 512, 0, st>>>(q, bytes / 16, sink);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::dense
