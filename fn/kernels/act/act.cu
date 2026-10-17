#include "fn/kernels/act/act.h"

#include "fn/kernels/act/act_device.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::act {
namespace {

constexpr int32_t lanes = 8;    // lanes a run: 4 values each
constexpr int32_t threads = 64; // runs a block: 8

// thread t: run t / 8, values 4 (t % 8) .. 4 (t % 8) + 3 of it (x, codes and scales flat: rows are dense); the rule is
// quantize_run8 (act_device.cuh), shared with the producers' epilogues
__global__ void __launch_bounds__(threads) quantize_kernel(const float * __restrict__ x, int8_t * __restrict__ codes,
                                                           float2 * __restrict__ scales, int32_t n_runs) {
    const int32_t t = static_cast<int32_t>(blockIdx.x) * threads + static_cast<int32_t>(threadIdx.x);
    const int32_t r = t / lanes;
    const bool real = r < n_runs; // a whole run of lanes is real or none is, so the shuffles stay within a run
    const float4 v = real ? __ldg(reinterpret_cast<const float4 *>(x) + t) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    uint32_t packed = 0u;
    float2 ds = make_float2(0.0f, 0.0f);
    quantize_run8(v, packed, ds);
    if (real) {
        reinterpret_cast<uint32_t *>(codes)[t] = packed;
        if ((t & (lanes - 1)) == 0) {
            scales[r] = ds;
        }
    }
}

bool aligned(const void * p, uintptr_t a) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0;
}

} // namespace

bool quantize(const float * x, const out_view & o, void * stream) {
    if (!valid(o.rows, o.k) || !aligned(x, 16) || !aligned(o.codes, 16) || !aligned(o.scales, 8) || stream == nullptr) {
        return false;
    }
    const int32_t n_runs = o.rows * (o.k / run); // at most 2^27
    const int32_t blocks = (n_runs * lanes + threads - 1) / threads;
    quantize_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        x, o.codes, reinterpret_cast<float2 *>(o.scales), n_runs);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::act
