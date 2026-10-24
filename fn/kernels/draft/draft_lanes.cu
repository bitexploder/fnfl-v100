#include "fn/kernels/draft/draft.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::draft {
namespace {

constexpr int32_t copy_threads = 256;

struct copy_list {
    copy_seg s[max_copy_segs];
};

// block i copies segment i: dst[w] = src[w * src_stride] for w < words
__global__ void __launch_bounds__(copy_threads) copy_kernel(const copy_list l) {
    const copy_seg & g = l.s[blockIdx.x];
    for (int32_t w = static_cast<int32_t>(threadIdx.x); w < g.words; w += copy_threads) {
        g.dst[w] = g.src[int64_t(w) * g.src_stride];
    }
}

} // namespace

bool lanes_copy(int32_t n, const copy_seg * segs, void * stream) {
    if (n < 1 || n > max_copy_segs || segs == nullptr || stream == nullptr) {
        return false;
    }
    copy_list l = {};
    for (int32_t i = 0; i < n; ++i) {
        if (segs[i].src == nullptr || segs[i].dst == nullptr || segs[i].words < 1 || segs[i].src_stride < 1) {
            return false;
        }
        l.s[i] = segs[i];
    }
    copy_kernel<<<n, copy_threads, 0, static_cast<cudaStream_t>(stream)>>>(l);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::draft
