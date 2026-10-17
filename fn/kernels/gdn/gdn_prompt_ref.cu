// gdp-test's FP64 reference of the gated delta rule (test support, fnfl-drive only): the token-serial recurrence of
// gdn.h (core's per-row math, without the chunking) in double precision, from the stored F32 inputs. A warp a (head,
// value row j): lanes hold M[j][4 lane .. 4 lane + 3]; per row, M = exp(g) M; delta = beta (v_j - M k); M += delta k;
// o_j = (M q) / sqrt(128). Rows in order; the state persists in the caller's FP64 buffer across launches (a chain).
// Built without fast math.

#include "fn/kernels/gdn/gdn_prompt_test.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int32_t NH = 24, HD = 128, NCH = 5120, NV = 3072, NG = 8;

__device__ __forceinline__ double wsum(double v) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__global__ void ref_kernel(int32_t       rows,
                           const float * conv,
                           const float * g,
                           const float * beta,
                           double *      state,
                           double *      o) {
    const int32_t h = blockIdx.x / HD, j = blockIdx.x % HD, lane = threadIdx.x;
    const int32_t grp   = h % NG;
    double *      m     = state + (int64_t(h) * HD + j) * HD + 4 * lane;
    double        mm[4] = { m[0], m[1], m[2], m[3] };
    const double  scale = 1.0 / sqrt(double(HD));
    for (int32_t t = 0; t < rows; ++t) {
        const float * row = conv + int64_t(t) * NCH;
        const double  dec = exp(double(g[int64_t(t) * NH + h]));
        const double  b   = beta[int64_t(t) * NH + h];
        double        k[4], q[4];
#pragma unroll
        for (int32_t e = 0; e < 4; ++e) {
            k[e] = row[1024 + grp * HD + 4 * lane + e];
            q[e] = row[grp * HD + 4 * lane + e];
            mm[e] *= dec;
        }
        double p = 0.0;
#pragma unroll
        for (int32_t e = 0; e < 4; ++e) {
            p += mm[e] * k[e];
        }
        p                  = wsum(p);
        const double delta = b * (double(row[2048 + h * HD + j]) - p);
        double       oo    = 0.0;
#pragma unroll
        for (int32_t e = 0; e < 4; ++e) {
            mm[e] += delta * k[e];
            oo += mm[e] * q[e];
        }
        oo = wsum(oo);
        if (lane == 0) {
            o[int64_t(t) * NV + h * HD + j] = oo * scale;
        }
    }
#pragma unroll
    for (int32_t e = 0; e < 4; ++e) {
        m[e] = mm[e];
    }
}

}  // namespace

bool gdp_reference(int32_t       rows,
                   const float * conv,
                   const float * g,
                   const float * beta,
                   double *      state,
                   double *      o,
                   void *        stream) {
    ref_kernel<<<NH * HD, 32, 0, static_cast<cudaStream_t>(stream)>>>(rows, conv, g, beta, state, o);
    return cudaPeekAtLastError() == cudaSuccess;
}
