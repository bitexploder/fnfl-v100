#pragma once

// APF2: the dense ladder's per-file launch, opt-in and resource helpers over a list of sources (dense_ladder.cu,
// dense_ladder_iq.cu instantiate them; host side of those files).

#include "fn/kernels/dense/dense_ladder.cuh"

#include <atomic>

namespace fn::dense::pk {

// the shared memory opt-in of every tile of source S on the current device (one bit a device: the attribute is per
// device and the prompt path runs on both cards)
template <pl_src S> bool pl_attr_src() {
    static std::atomic<uint32_t> done{ 0u };
    int                          dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        return false;
    }
    const uint32_t bit = 1u << dev;
    if ((done.load(std::memory_order_acquire) & bit) != 0u) {
        return true;
    }
    for (int32_t t = 0; t < pl_n_tiles; ++t) {
        const bool ok = pl_with_tile<S>(t, [](auto kern, auto sh) {
            return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, decltype(sh)::smem) ==
                   cudaSuccess;
        });
        if (!ok) {
            return false;
        }
    }
    done.fetch_or(bit, std::memory_order_release);
    return true;
}

template <pl_src S> bool pl_launch_src(const pl_kargs & k, int32_t t, cudaStream_t s) {
    if (!pl_attr_src<S>()) {
        return false;
    }
    return pl_with_tile<S>(t, [&](auto kern, auto sh) {
        using T = decltype(sh);
        const dim3 grid((k.e.p.rows + T::bm - 1) / T::bm, (k.e.p.n + T::bn - 1) / T::bn, k.splits);
        kern<<<grid, T::threads, T::smem, s>>>(k);
        return cudaPeekAtLastError() == cudaSuccess;
    });
}

template <pl_src S> bool pl_res_src(int32_t t, pd_resources & r) {
    if (!pl_attr_src<S>()) {
        return false;
    }
    return pl_with_tile<S>(t, [&](auto kern, auto sh) {
        using T                = decltype(sh);
        cudaFuncAttributes at  = {};
        int32_t            cta = 0;
        if (cudaFuncGetAttributes(&at, kern) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cta, kern, T::threads, T::smem) != cudaSuccess) {
            return false;
        }
        r.regs         = at.numRegs;
        r.local_bytes  = static_cast<int32_t>(at.localSizeBytes);
        r.shared_bytes = static_cast<int32_t>(at.sharedSizeBytes) + T::smem;
        r.ctas_per_sm  = cta;
        return true;
    });
}

}  // namespace fn::dense::pk
