// APF2: the ladder's grouped experts (experts_ladder.h). Built with relocatable device code (the launcher's device
// launch), in its own library, so experts_prompt.cu's per-task kernels keep their code.

#include "fn/kernels/experts/experts_ladder.h"
#include "fn/kernels/experts/experts_prompt_dev.cuh"
#include "fn/kernels/tc/tc_ladder.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace fn::experts {
namespace {

using namespace epd;

constexpr int32_t el_bn      = 128;
constexpr int32_t el_threads = 128;
constexpr int32_t el_gu_ct   = 2 * n_ff_card / el_bn;  // 5
constexpr int32_t el_dn_ct   = D / el_bn;              // 20
static_assert(el_gu_ct * el_bn == 2 * n_ff_card && el_dn_ct * el_bn == D, "the column tiles cover the products");

template <int32_t BM> constexpr int32_t el_smem = fn::tc::ltile<BM, el_bn>::smem;

// ---- gate/up ------------------------------------------------------------------------------------------------------
// A: the task's activation rows (sorted pair first + r: row sorted / 10), zero past the task's pairs
struct el_gu_a {
    const uint16_t * x;
    const int32_t *  sorted;
    int32_t          first, n;

    __device__ __forceinline__ uint4 operator()(int32_t r, int32_t k) const {
        if (r >= n) {
            return make_uint4(0u, 0u, 0u, 0u);
        }
        const int32_t row = __ldg(sorted + first + r) / n_used;
        return __ldg(reinterpret_cast<const uint4 *>(x + int64_t(row) * D + k));
    }
};

// B: thread t holds tile column t (its K-quant sub-block raw at fetch, its four FP16 chunks kq_chunk's at store)
template <fmt F> struct el_gu_b {
    const uint8_t * w;
    int4            o;  // gu_offsets of the column (sub-block 0)

    struct raw {
        kq_raw r;
    };

    __device__ __forceinline__ void fetch(raw & x, int32_t, int32_t k0) const { x.r = kq_load<F>(w, o, k0 / ks); }

    __device__ __forceinline__ void store(uint32_t sb, const raw & x) const {
        const int32_t  c  = static_cast<int32_t>(threadIdx.x);
        const gu_scale sc = kq_scale(x.r);
#pragma unroll
        for (int32_t l = 0; l < 4; ++l) {
            fn::tc::detail::sts128(sb + fn::tc::lsw(c, l), kq_chunk<F>(x.r, sc, l));
        }
    }
};

// a lane's columns c0, c0 + 1 (gate) and c0 + 4, c0 + 5 (up) are channels ch, ch + 1: ep_gate_up's GLU arithmetic
struct el_gu_epi {
    uint16_t * glu;
    float *    glu_f32;
    int32_t    first, col0, ch0;

    __device__ __forceinline__ void operator()(int32_t row, int32_t c0, const float (&v)[4]) const {
        const int32_t  lc = c0 - col0;
        const int32_t  ch = ch0 + 4 * (lc >> 3) + (lc & 3);
        const float    v0 = v[0] / (1.0f + expf(-v[0])) * v[2];
        const float    v1 = v[1] / (1.0f + expf(-v[1])) * v[3];
        const int64_t  oo = int64_t(first + row) * n_ff_card + ch;
        *reinterpret_cast<uint32_t *>(glu + oo) = h2_pack(v0, v1);
        if (glu_f32 != nullptr) {
            *reinterpret_cast<float2 *>(glu_f32 + oo) = make_float2(v0, v1);
        }
    }
};

template <fmt F, int32_t BM>
__global__ void __launch_bounds__(el_threads) el_gate_up_kernel(const ep_gate_up_args a, const int32_t * tasks) {
    static_assert(fn::tc::ltile<BM, el_bn>::threads == el_threads, "one column a thread");
    extern __shared__ uint4 el_sm[];
    const task_t  tk = task_of(tasks, static_cast<int32_t>(blockIdx.x));
    const int32_t y  = static_cast<int32_t>(blockIdx.y);
    const int32_t t  = static_cast<int32_t>(threadIdx.x);
    const int32_t ch = 64 * y + 4 * (t >> 3) + (t & 3);
    const bool    up = (t & 4) != 0;
    const el_gu_a al{ a.x, a.route.sorted, tk.first, tk.n };
    el_gu_b<F>    bl;
    bl.w = a.w;
    bl.o = gu_offsets<F>(gu_col_of<F>(ch & 7, (up ? 4 : 0) + ((ch & 31) >> 3)),
                         (tk.e * n_group + (ch >> 5)) * ep_tile_bytes(F));
    const el_gu_epi epi{ a.glu, a.glu_f32, tk.first, y * el_bn, 64 * y };
    fn::tc::lgemm_at<BM, el_bn>(0, tk.n, y * el_bn, 0, D, 1, fn::tc::lsplit{}, reinterpret_cast<uint8_t *>(el_sm), al,
                                bl, epi);
}

// ---- down ---------------------------------------------------------------------------------------------------------
// A: the task's GLU rows, chunk c of a step in IQ4_NL's K order (K 4 c .. 4 c + 3, 16 + 4 c .. 16 + 4 c + 3)
struct el_dn_a {
    const uint16_t * glu;
    int32_t          first, n;

    __device__ __forceinline__ uint4 operator()(int32_t r, int32_t k) const {
        if (r >= n) {
            return make_uint4(0u, 0u, 0u, 0u);
        }
        const uint16_t * p  = glu + int64_t(first + r) * n_ff_card + (k & ~(ks - 1));
        const int32_t    c  = (k >> 3) & 3;
        const uint2      lo = __ldg(reinterpret_cast<const uint2 *>(p + 4 * c));
        const uint2      hi = __ldg(reinterpret_cast<const uint2 *>(p + 16 + 4 * c));
        return make_uint4(lo.x, lo.y, hi.x, hi.y);
    }
};

// B: thread t holds output column col0 + t (dn_load at fetch, dn_chunk's four chunks at store)
template <fmt F> struct el_dn_b {
    const uint8_t * w;
    int32_t         eo, n16;

    struct raw {
        dn_raw_of<F> r;
    };

    __device__ __forceinline__ void fetch(raw & x, int32_t, int32_t k0) const {
        x.r = dn_load<F>(w, eo, n16, k0 / ks);
    }

    __device__ __forceinline__ void store(uint32_t sb, const raw & x) const {
        const int32_t  c     = static_cast<int32_t>(threadIdx.x);
        const uint32_t kv[4] = { 0u, 0u, 0u, 0u };  // IQ4_NL's table: unused by Q5_1 and Q8_0
#pragma unroll
        for (int32_t l = 0; l < 4; ++l) {
            fn::tc::detail::sts128(sb + fn::tc::lsw(c, l), dn_chunk<F>(x.r, kv, l));
        }
    }
};

// ep_down's stores: pairs of adjacent columns, F32 or FP16 rounded once (err_bit past 65,504 or NaN)
struct el_dn_epi {
    float *    part;
    uint16_t * part16;
    uint32_t * err;
    uint32_t   err_bit;
    int32_t    first;

    __device__ __forceinline__ void operator()(int32_t row, int32_t c0, const float (&v)[4]) const {
        const int64_t o = int64_t(first + row) * D + c0;
        if (part16 != nullptr) {
            bool big = false;
#pragma unroll
            for (int32_t i = 0; i < 4; ++i) {
                big = big || !(fabsf(v[i]) <= 65504.0f);
            }
            *reinterpret_cast<uint32_t *>(part16 + o)     = h2_pack(v[0], v[1]);
            *reinterpret_cast<uint32_t *>(part16 + o + 4) = h2_pack(v[2], v[3]);
            if (big) {
                atomicOr(err, err_bit);
            }
        } else {
            *reinterpret_cast<float2 *>(part + o)     = make_float2(v[0], v[1]);
            *reinterpret_cast<float2 *>(part + o + 4) = make_float2(v[2], v[3]);
        }
    }
};

template <fmt F, int32_t BM>
__global__ void __launch_bounds__(el_threads) el_down_kernel(const ep_down_args a, const int32_t * tasks) {
    static_assert(fn::tc::ltile<BM, el_bn>::threads == el_threads, "one column a thread");
    extern __shared__ uint4 el_sm[];
    const task_t    tk   = task_of(tasks, static_cast<int32_t>(blockIdx.x));
    const int32_t   col0 = static_cast<int32_t>(blockIdx.y) * el_bn;
    const el_dn_a   al{ a.glu, tk.first, tk.n };
    const el_dn_b<F> bl{ a.w, tk.e * dn_bytes(F), 16 * (col0 + static_cast<int32_t>(threadIdx.x)) };
    const el_dn_epi epi{ a.part, a.part16, a.err, a.err_bit, tk.first };
    fn::tc::lgemm_at<BM, el_bn>(0, tk.n, col0, 0, n_ff_card, 1, fn::tc::lsplit{}, reinterpret_cast<uint8_t *>(el_sm),
                                al, bl, epi);
}

// ---- the launchers: the grid from the device's task count -------------------------------------------------------
template <fmt F, int32_t BM>
__global__ void __launch_bounds__(1) el_gate_up_launch(const ep_gate_up_args a, const int32_t * tasks,
                                                      const int32_t * count) {
    const int32_t n = *reinterpret_cast<const volatile int32_t *>(count);
    if (n > 0) {
        el_gate_up_kernel<F, BM><<<dim3(n, el_gu_ct), el_threads, el_smem<BM>, cudaStreamTailLaunch>>>(a, tasks);
    }
}

template <fmt F, int32_t BM>
__global__ void __launch_bounds__(1) el_down_launch(const ep_down_args a, const int32_t * tasks,
                                                   const int32_t * count) {
    const int32_t n = *reinterpret_cast<const volatile int32_t *>(count);
    if (n > 0) {
        el_down_kernel<F, BM><<<dim3(n, el_dn_ct), el_threads, el_smem<BM>, cudaStreamTailLaunch>>>(a, tasks);
    }
}

// run(std::integral_constant<int32_t, BM>) for tile t (lad_tiles 0 .. 4)
template <typename Fn> bool el_with_tile(int32_t t, Fn && run) {
    switch (t) {
        case 0:
            return run(std::integral_constant<int32_t, 8>{});
        case 1:
            return run(std::integral_constant<int32_t, 16>{});
        case 2:
            return run(std::integral_constant<int32_t, 32>{});
        case 3:
            return run(std::integral_constant<int32_t, 64>{});
        case 4:
            return run(std::integral_constant<int32_t, 128>{});
        default:
            return false;
    }
}

bool al16(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

bool al4(const void * p) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & 3u) == 0;
}

bool route_ok(const ep_route & r) {
    return al4(r.offsets) && al4(r.sorted) && al4(r.pos) && al16(r.tasks) && al4(r.counters);
}

}  // namespace

int32_t el_rows(int32_t tile) {
    return tile >= 0 && tile < el_n_tiles ? fn::tc::lad_tiles[tile].bm : 0;
}

bool el_takes_gate_up(fmt f) {
    return f == fmt::q4_k || f == fmt::q5_k;
}

bool el_takes_down(fmt f) {
    return f == fmt::q5_1 || f == fmt::q8_0;
}

bool ep_gate_up_ladder(fmt                     f,
                       int32_t                 t,
                       int32_t                 rows,
                       const ep_gate_up_args & a,
                       const int32_t *         tasks,
                       const int32_t *         count,
                       void *                  stream) {
    if (!el_takes_gate_up(f) || t < 0 || t >= el_n_tiles || rows < 1 || rows > ep_max_rows || !al16(a.w) ||
        !al16(a.x) || !route_ok(a.route) || !al4(a.glu) || (a.glu_f32 != nullptr && !al16(a.glu_f32)) ||
        a.wt != nullptr || a.task_off != nullptr || !al16(tasks) || !al4(count) || stream == nullptr) {
        return false;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    return el_with_tile(t, [&](auto bm) {
        constexpr int32_t BM = decltype(bm)::value;
        if (f == fmt::q4_k) {
            el_gate_up_launch<fmt::q4_k, BM><<<1, 1, 0, s>>>(a, tasks, count);
        } else {
            el_gate_up_launch<fmt::q5_k, BM><<<1, 1, 0, s>>>(a, tasks, count);
        }
        return cudaPeekAtLastError() == cudaSuccess;
    });
}

bool ep_down_ladder(fmt                  f,
                    int32_t              t,
                    int32_t              rows,
                    const ep_down_args & a,
                    const int32_t *      tasks,
                    const int32_t *      count,
                    void *               stream) {
    if (!el_takes_down(f) || t < 0 || t >= el_n_tiles || rows < 1 || rows > ep_max_rows || !al16(a.w) ||
        !al16(a.glu) || !route_ok(a.route) ||
        (a.part16 != nullptr ? !al16(a.part16) || a.err == nullptr : !al16(a.part)) || a.wt != nullptr ||
        a.task_off != nullptr || !al16(tasks) || !al4(count) || stream == nullptr) {
        return false;
    }
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    return el_with_tile(t, [&](auto bm) {
        constexpr int32_t BM = decltype(bm)::value;
        if (f == fmt::q5_1) {
            el_down_launch<fmt::q5_1, BM><<<1, 1, 0, s>>>(a, tasks, count);
        } else {
            el_down_launch<fmt::q8_0, BM><<<1, 1, 0, s>>>(a, tasks, count);
        }
        return cudaPeekAtLastError() == cudaSuccess;
    });
}

bool el_resources(int32_t product, fmt f, int32_t t, fn::tc::lad_resources & r) {
    if (product == 0 ? !el_takes_gate_up(f) : product == 1 ? !el_takes_down(f) : true) {
        return false;
    }
    return el_with_tile(t, [&](auto bm) {
        constexpr int32_t BM   = decltype(bm)::value;
        const void *      kern = nullptr;
        if (product == 0) {
            kern = f == fmt::q4_k ? reinterpret_cast<const void *>(el_gate_up_kernel<fmt::q4_k, BM>) :
                                    reinterpret_cast<const void *>(el_gate_up_kernel<fmt::q5_k, BM>);
        } else {
            kern = f == fmt::q5_1 ? reinterpret_cast<const void *>(el_down_kernel<fmt::q5_1, BM>) :
                                    reinterpret_cast<const void *>(el_down_kernel<fmt::q8_0, BM>);
        }
        cudaFuncAttributes at = {};
        int32_t            n  = 0;
        if (cudaFuncGetAttributes(&at, kern) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, el_threads, el_smem<BM>) != cudaSuccess) {
            return false;
        }
        r.regs         = at.numRegs;
        r.local_bytes  = static_cast<int32_t>(at.localSizeBytes);
        r.shared_bytes = static_cast<int32_t>(at.sharedSizeBytes) + el_smem<BM>;
        r.ctas_per_sm  = n;
        return true;
    });
}

}  // namespace fn::experts
