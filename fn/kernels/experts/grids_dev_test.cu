// grid-test's device part (FMS4 item 2): the decoders' IQ2 selector recipe (experts_dev.cuh sign8_of7, iq2_signed)
// on the device for every entry of the three selector tables and every sign pattern, against the 8-byte tables'
// signed bytes. The device form of fnfl/evidence/fms4-decode-recipes/final.py.

#include "fn/kernels/experts/grids_test.h"

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_dev.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using fn::experts::fmt;
using fn::experts::gu_layout_of;

// the three grids in grid_table()'s order: entries, sign patterns an entry (7 explicit bits for IQ2_XXS and IQ2_XS,
// the eighth their parity; 8 for IQ2_S), the 8-byte table's offset in the grid buffer
struct sel_grid {
    const char * name;
    int32_t entries, signs, table_off;
    fmt f;
};
constexpr sel_grid sel_grids[3] = {{"iq2xxs", 256, 128, 0, fmt::iq2_xxs}, {"iq2xs", 512, 128, 2048, fmt::iq2_xs},
                                   {"iq2s", 1024, 256, 6144, fmt::iq2_s}};
constexpr int64_t n_cases = 256 * 128 + 512 * 128 + 1024 * 256; // 360,448

struct sel_args {
    const uint32_t * sel[3]; // each grid's selector table on the device
    int64_t first[4];        // each grid's first case, then n_cases
    uint2 * out;             // [n_cases]: the signed bytes, weights 0..3 then 4..7
};

// one case a thread: case c of grid r is entry c / signs, sign pattern c % signs, through the decoders' own helpers
__global__ void sel_kernel(sel_args a) {
    const int64_t c = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= a.first[3]) {
        return;
    }
    const int32_t r = c < a.first[1] ? 0 : c < a.first[2] ? 1 : 2;
    const int64_t i = c - a.first[r];
    const int32_t signs = r == 2 ? 256 : 128;
    const uint32_t s = static_cast<uint32_t>(i % signs);
    const uint32_t g32 = a.sel[r][i / signs];
    uint2 w;
    fn::experts::iq2_signed(g32, r == 2 ? s : fn::experts::sign8_of7(s), w.x, w.y);
    a.out[c] = w;
}

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "grid-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

} // namespace

int64_t fnfl_grid_sel_device(const uint8_t * buf, int dev) {
    if (!cuda_ok(cudaSetDevice(dev), "cudaSetDevice")) {
        return -1;
    }
    sel_args a{};
    uint8_t * d_sel = nullptr;
    const size_t sel_bytes = fn::experts::grid_bytes - static_cast<size_t>(gu_layout_of(fmt::iq2_xxs).grid_off);
    const int32_t sel0 = gu_layout_of(fmt::iq2_xxs).grid_off;
    if (!cuda_ok(cudaMalloc(&d_sel, sel_bytes), "cudaMalloc") ||
        !cuda_ok(cudaMalloc(&a.out, size_t(n_cases) * sizeof(uint2)), "cudaMalloc")) {
        cudaFree(d_sel);
        return -1;
    }
    int64_t first = 0;
    for (int32_t r = 0; r < 3; ++r) {
        a.sel[r] = reinterpret_cast<const uint32_t *>(d_sel + (gu_layout_of(sel_grids[r].f).grid_off - sel0));
        a.first[r] = first;
        first += int64_t(sel_grids[r].entries) * sel_grids[r].signs;
    }
    a.first[3] = first;
    std::vector<uint2> got(static_cast<size_t>(n_cases));
    bool ran = first == n_cases && cuda_ok(cudaMemcpy(d_sel, buf + sel0, sel_bytes, cudaMemcpyHostToDevice), "upload");
    if (ran) {
        sel_kernel<<<static_cast<uint32_t>((n_cases + 255) / 256), 256>>>(a);
        ran = cuda_ok(cudaGetLastError(), "sel_kernel") && cuda_ok(cudaDeviceSynchronize(), "sel_kernel") &&
              cuda_ok(cudaMemcpy(got.data(), a.out, got.size() * sizeof(uint2), cudaMemcpyDeviceToHost), "download");
    }
    cudaFree(d_sel);
    cudaFree(a.out);
    if (!ran) {
        return -1;
    }
    // the reference: weight k of the entry is the 8-byte table's byte k, negated when sign bit k is set (ggml's
    // ksigns_iq2xs for the 7-bit groups: the eighth bit the parity of the seven)
    int64_t wrong = 0;
    for (int32_t r = 0; r < 3; ++r) {
        const sel_grid & g = sel_grids[r];
        int64_t bad = 0;
        for (int32_t e = 0; e < g.entries; ++e) {
            for (int32_t s = 0; s < g.signs; ++s) {
                uint32_t bits = static_cast<uint32_t>(s);
                if (g.signs == 128) {
                    bits |= (static_cast<uint32_t>(__builtin_popcount(bits)) & 1u) << 7;
                }
                const uint2 w = got[static_cast<size_t>(a.first[r] + int64_t(e) * g.signs + s)];
                for (int32_t k = 0; k < 8; ++k) {
                    const int32_t mag = buf[g.table_off + 8 * e + k];
                    const int32_t want = (bits >> k) & 1u ? -mag : mag;
                    const int32_t have = static_cast<int8_t>(((k < 4 ? w.x : w.y) >> (8 * (k & 3))) & 0xffu);
                    if (have != want) {
                        ++bad;
                        break;
                    }
                }
            }
        }
        printf("grid-test: %-6s selector recipe on device %d: %4d entries x %3d sign patterns, %lld wrong\n", g.name,
               dev, g.entries, g.signs, static_cast<long long>(bad));
        wrong += bad;
    }
    return wrong;
}
