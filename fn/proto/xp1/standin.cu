// XP1's load stand-ins (the contract is in standin.h). Each piece below is one instance of standin_kernel, so a trace
// names it; their per-piece contracts:
//
// LOAD STAND-IN (XP1): streams gdn_proj_conv's bytes so coordination is timed at a streaming kernel's speed; not
//   gdn_proj_conv's math; replaced by DP1 (the qkv and z products) and the GDN kernel ticket (not yet written, GD-3).
// LOAD STAND-IN (XP1): streams gdn_recur's bytes ...; not gdn_recur's math; replaced by the GDN kernel ticket (GD-1).
// LOAD STAND-IN (XP1): streams gdn_out's bytes ...; not gdn_out's math; replaced by DP1.
// LOAD STAND-IN (XP1): streams index_q_proj's bytes ...; not index_q_proj's math; replaced by DP1.
// LOAD STAND-IN (XP1): streams index_k_proj's bytes ...; not index_k_proj's math; replaced by DP1.
// LOAD STAND-IN (XP1): streams qsa_qkv's bytes ...; not qsa_qkv's math; replaced by DP1 and the QSA prep with QA2.
// LOAD STAND-IN (XP1): streams qsa_attention's bytes ...; not qsa_attention's math; replaced by QA2.
// LOAD STAND-IN (XP1): streams qsa_out's bytes ...; not qsa_out's math; replaced by DP1.
// LOAD STAND-IN (XP1): streams router's bytes ...; not the router's math; replaced by EX2.
// LOAD STAND-IN (XP1): streams the shared expert's bytes (gate/up, down) ...; not its math; replaced by EX2.

#include "fn/proto/xp1/standin.h"

#include "fn/xchg/xchg.cuh"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::xp1 {

const piece_info piece_table[n_piece] = {
    {"gdn_proj_conv", 10, "gdn_proj_conv", "DP1 and the GDN kernel ticket (GD-3)"},
    {"gdn_recur", 10, "gdn_recur", "the GDN kernel ticket (GD-1)"},
    {"gdn_out", 10, "gdn_out", "DP1"},
    {"index_q_proj", 11, "index_q_proj", "DP1"},
    {"index_k_proj", 11, "index_k_proj", "DP1"},
    {"qsa_qkv_norm_rope_kv", 11, "qsa_qkv_norm_rope_kv", "DP1 and the QSA prep (QA2)"},
    {"qsa_attention", 11, "qsa_attention", "QA2"},
    {"qsa_out", 11, "qsa_out", "DP1"},
    {"router", 10, "router", "EX2"},
    {"shared_expert gate_up", 10, "shared_expert", "EX2"},
    {"shared_expert down", 10, "shared_expert", "EX2"},
    {"router", 11, "router", "EX2"},
    {"shared_expert gate_up", 11, "shared_expert", "EX2"},
    {"shared_expert down", 11, "shared_expert", "EX2"},
};

namespace {

constexpr int32_t half = fn::xchg::words_half; // 1280
constexpr int32_t run_len = 28;                // put_run's longest run
constexpr int32_t runs_half = (half + run_len - 1) / run_len;
constexpr int32_t kv_u4 = 32; // 16-byte words of a 512-byte K or V row (256 BF16)
constexpr int32_t warps = si_threads / 32;
static_assert(n_piece == 14, "one launch case a piece");

__device__ __forceinline__ void xor4(uint4 & a, const uint4 b) {
    a.x ^= b.x;
    a.y ^= b.y;
    a.z ^= b.z;
    a.w ^= b.w;
}

__device__ __forceinline__ uint16_t bf16_rne(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u); // round to nearest even; the values are finite
    return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float out_value(const standin_args & a, int32_t b, int32_t j, float c) {
    return __fadd_rn(__fmul_rn(0.5f, a.in[b * a.in_width + j % a.in_width]), c);
}

// XOR of a block's words: warp butterflies, then one warp over the warps' results
__device__ __forceinline__ uint4 block_xor(uint4 v, uint4 * sm) {
#pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        v.x ^= __shfl_xor_sync(0xffffffffu, v.x, off);
        v.y ^= __shfl_xor_sync(0xffffffffu, v.y, off);
        v.z ^= __shfl_xor_sync(0xffffffffu, v.z, off);
        v.w ^= __shfl_xor_sync(0xffffffffu, v.w, off);
    }
    const int32_t lane = threadIdx.x % 32, w = threadIdx.x / 32;
    if (lane == 0) {
        sm[w] = v;
    }
    __syncthreads();
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (w == 0) {
        r = lane < warps ? sm[lane] : r;
#pragma unroll
        for (int32_t off = 16; off > 0; off >>= 1) {
            r.x ^= __shfl_xor_sync(0xffffffffu, r.x, off);
            r.y ^= __shfl_xor_sync(0xffffffffu, r.y, off);
            r.z ^= __shfl_xor_sync(0xffffffffu, r.z, off);
            r.w ^= __shfl_xor_sync(0xffffffffu, r.w, off);
        }
    }
    __syncthreads();
    return r;
}

// One instance a piece (ID), so traces name each stand-in.
template <int ID> __global__ void __launch_bounds__(si_threads) standin_kernel(const standin_args a) {
    __shared__ uint4 sm[warps];
    const int64_t G = int64_t(gridDim.x) * blockDim.x;
    const int64_t t0 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint4 xw = make_uint4(0u, 0u, 0u, 0u); // weights
    uint4 xo = make_uint4(0u, 0u, 0u, 0u); // everything else

    // the weights: a plain coalesced stream, four loads a thread in flight
    {
        const uint4 * w = static_cast<const uint4 *>(a.w);
        const int64_t n = a.w_bytes / 16;
        int64_t i = t0;
        for (; i + 3 * G < n; i += 4 * G) {
            const uint4 v0 = w[i], v1 = w[i + G], v2 = w[i + 2 * G], v3 = w[i + 3 * G];
            xor4(xw, v0);
            xor4(xw, v1);
            xor4(xw, v2);
            xor4(xw, v3);
        }
        for (; i < n; i += G) {
            xor4(xw, w[i]);
        }
    }
    // the state: read once, written as st_copies versions
    if (a.st != nullptr) {
        const uint4 * s = static_cast<const uint4 *>(a.st);
        uint4 * d = static_cast<uint4 *>(a.st_dst);
        const int64_t n = a.st_bytes / 16;
        for (int64_t i = t0; i < n; i += G) {
            const uint4 v = s[i];
            xor4(xo, v);
            for (int32_t k = 0; k < a.st_copies; ++k) {
                d[k * n + i] = v;
            }
        }
    }
    // the gather: each row's own cells, K then V, 64 16-byte words a cell
    if (a.body != nullptr) {
        const uint4 * kk = static_cast<const uint4 *>(a.kv_k);
        const uint4 * vv = static_cast<const uint4 *>(a.kv_v);
        const int64_t per_row = int64_t(a.cell_width) * 2 * kv_u4;
        const int64_t n = per_row * a.rows;
        for (int64_t e = t0; e < n; e += G) {
            const int32_t b = static_cast<int32_t>(e / per_row);
            const int32_t r = static_cast<int32_t>(e % per_row);
            const uint32_t * row = a.body + a.body_off[b] / 4;
            const int32_t nk = static_cast<int32_t>(row[0]), nt = static_cast<int32_t>(row[1]);
            const int32_t i = r / (2 * kv_u4);
            if (i >= 4 * nk + nt) {
                continue;
            }
            const int64_t cell = i < 4 * nk ? int64_t(row[2 + i / 4]) * 4 + i % 4 : int64_t(row[2 + nk + i - 4 * nk]);
            const int32_t p = r % (2 * kv_u4);
            xor4(xo, p < kv_u4 ? kk[cell * kv_u4 + p] : vv[cell * kv_u4 + p - kv_u4]);
        }
    } else if (a.cells != nullptr) {
        const int32_t * cells = a.cells + (a.decode != nullptr ? int64_t(*a.decode) * a.cells_decode_stride : 0);
        const uint4 * kk = static_cast<const uint4 *>(a.kv_k);
        const uint4 * vv = static_cast<const uint4 *>(a.kv_v);
        const int64_t per_row = int64_t(a.cell_width) * 2 * kv_u4;
        const int64_t n = per_row * a.rows;
        for (int64_t e = t0; e < n; e += G) {
            const int32_t b = static_cast<int32_t>(e / per_row);
            const int32_t r = static_cast<int32_t>(e % per_row);
            const int64_t cell = cells[int64_t(b) * a.cell_stride + r / (2 * kv_u4)];
            const int32_t p = r % (2 * kv_u4);
            xor4(xo, p < kv_u4 ? kk[cell * kv_u4 + p] : vv[cell * kv_u4 + p - kv_u4]);
        }
    }
    // the padding: four independent FMA chains on registers
    if (a.pad > 0) {
        float f0 = __uint_as_float((xw.x & 0x007fffffu) | 0x3f800000u), f1 = f0 + 1.0f, f2 = f0 + 2.0f, f3 = f0 + 3.0f;
        for (int32_t k = 0; k < a.pad; ++k) {
            f0 = fmaf(f0, 0.9999f, 1e-4f);
            f1 = fmaf(f1, 0.9999f, 1e-4f);
            f2 = fmaf(f2, 0.9999f, 1e-4f);
            f3 = fmaf(f3, 0.9999f, 1e-4f);
        }
        xo.x ^= __float_as_uint(f0);
        xo.y ^= __float_as_uint(f1);
        xo.z ^= __float_as_uint(f2);
        xo.w ^= __float_as_uint(f3);
    }
    if (a.aux != nullptr && t0 < a.aux_n) {
        xo.x ^= __float_as_uint(a.aux[t0]);
    }

    // the output: the defined function of the real input
    const int32_t gw = static_cast<int32_t>(t0 / 32), lane = threadIdx.x % 32;
    const int32_t GW = static_cast<int32_t>(G / 32);
    if (a.out != nullptr && a.peer_slot != nullptr) {
        // runs of 28 words inside each half: the peer's half goes out through put_run, ours is stored
        for (int32_t R = gw; R < a.rows * 2 * runs_half; R += GW) {
            const int32_t b = R / (2 * runs_half), h = R / runs_half % 2, base = run_len * (R % runs_half);
            const int32_t len = min(run_len, half - base);
            const int32_t j = h * half + base + lane;
            float v = 0.0f;
            if (lane < len) {
                v = out_value(a, b, j, a.c);
                if (a.add != nullptr) {
                    v = __fadd_rn(a.add[b * a.out_width + j], v);
                }
            }
            float * row = a.out + int64_t(b) * a.out_width + h * half;
            if (h == a.peer_card) {
                fn::xchg::put_run(reinterpret_cast<uint32_t *>(row), // F32 payload words handled as bits
                                  reinterpret_cast<uint32_t *>(a.peer_slot) + b * half, base, len, 0,
                                  fn::xchg::canonical(v));
            } else if (lane < len) {
                row[base + lane] = v;
            }
        }
    } else if (a.out != nullptr) {
        const int32_t n = a.rows * a.out_width;
        for (int32_t e = static_cast<int32_t>(t0); e < n; e += static_cast<int32_t>(G)) {
            const int32_t b = e / a.out_width, j = e % a.out_width;
            float v = out_value(a, b, j, a.c);
            if (a.add != nullptr) {
                v = __fadd_rn(a.add[e], v);
            }
            a.out[e] = v;
        }
    }
#pragma unroll
    for (int32_t i = 0; i < 2; ++i) {
        if (a.bf_width == 0 || a.bf[i] == nullptr) {
            continue;
        }
        const int32_t n = (a.bf_hi - a.bf_lo) * a.bf_width;
        for (int32_t e = static_cast<int32_t>(t0); e < n; e += static_cast<int32_t>(G)) {
            const int32_t b = a.bf_lo + e / a.bf_width, j = e % a.bf_width;
            a.bf[i][(a.bf_cell0 + (b - a.bf_lo)) * a.bf_width + j] = bf16_rne(out_value(a, b, j, a.bf_c[i]));
        }
    }
    if (a.ids != nullptr) {
        // the router reads its input as a dense product would; its outputs do not depend on it
        for (int32_t e = static_cast<int32_t>(t0); e < a.rows * a.in_width; e += static_cast<int32_t>(G)) {
            xo.y ^= __float_as_uint(a.in[e]);
        }
        if (t0 < a.rows * 10) {
            const int32_t b = static_cast<int32_t>(t0) / 10, k = static_cast<int32_t>(t0) % 10;
            a.ids[t0] = a.route_src != nullptr ? a.route_src[(int64_t(*a.decode) * 3 + b) * 10 + k]
                                               : (37 * k + 13 * b + 7 * a.layer) % 512;
            a.wts[t0] = 0.1f;
        }
        if (t0 < a.rows) {
            a.sg[t0] = 0.5f;
        }
    }

    const uint4 rw = block_xor(xw, sm);
    const uint4 ro = block_xor(xo, sm);
    if (threadIdx.x == 0) {
        uint4 * s = reinterpret_cast<uint4 *>(a.sink + int64_t(blockIdx.x) * sink_block_words);
        s[0] = rw;
        s[1] = ro;
    }
}

bool al16(const void * p) {
    return reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

template <int ID> bool launch(const standin_args & a, cudaStream_t st) {
    standin_kernel<ID><<<si_blocks, si_threads, 0, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

bool standin(piece p, const standin_args & a, void * stream) {
    const int32_t id = static_cast<int32_t>(p);
    const bool sizes = a.w_bytes % 16 == 0 && a.st_bytes % 16 == 0 && (a.w == nullptr) == (a.w_bytes == 0) &&
                       (a.st == nullptr) == (a.st_bytes == 0) && (a.st_copies == 0 || a.st_dst != nullptr);
    const bool shape = a.rows >= 1 && a.rows <= 4 && (a.out == nullptr || (a.in != nullptr && a.in_width > 0 &&
                                                                          a.out_width > 0)) &&
                       (a.peer_slot == nullptr ||
                        (a.out_width == 2 * half && (a.peer_card == 0 || a.peer_card == 1))) &&
                       ((a.cells == nullptr && a.body == nullptr) ||
                        (a.kv_k != nullptr && a.kv_v != nullptr && a.cell_width > 0 &&
                         (a.body == nullptr || (a.cells == nullptr && a.body_off != nullptr)))) &&
                       (a.ids == nullptr || (a.wts != nullptr && a.sg != nullptr && a.in != nullptr &&
                                             (a.route_src == nullptr || a.decode != nullptr))) &&
                       (a.bf_width == 0 || (a.bf_lo >= 0 && a.bf_hi <= a.rows && a.bf_lo <= a.bf_hi));
    if (id < 0 || id >= n_piece || !sizes || !shape || a.sink == nullptr || a.pad < 0 || !al16(a.w) || !al16(a.st) ||
        !al16(a.st_dst) || !al16(a.kv_k) || !al16(a.kv_v) || !al16(a.peer_slot) || !al16(a.out) || stream == nullptr) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (id) {
        case 0: return launch<0>(a, st);
        case 1: return launch<1>(a, st);
        case 2: return launch<2>(a, st);
        case 3: return launch<3>(a, st);
        case 4: return launch<4>(a, st);
        case 5: return launch<5>(a, st);
        case 6: return launch<6>(a, st);
        case 7: return launch<7>(a, st);
        case 8: return launch<8>(a, st);
        case 9: return launch<9>(a, st);
        case 10: return launch<10>(a, st);
        case 11: return launch<11>(a, st);
        case 12: return launch<12>(a, st);
        case 13: return launch<13>(a, st);
        default: return false;
    }
}

} // namespace fn::xp1
