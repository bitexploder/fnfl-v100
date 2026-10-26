#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/qsa/qsa_prompt.h"
#include "fn/kernels/tc/tc_gemm.cuh"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>

namespace fn::qsa {
namespace {

constexpr int32_t n_sm = 72; // the PG500-216 (GUIDE section 1)
constexpr int32_t max_tiles = max_blocks / tile_blocks;
constexpr int32_t chunk_u4 = tile_blocks; // uint4 stride between a tile's chunks
constexpr int32_t tile_u4 = n_chunk * chunk_u4;
constexpr int32_t sel_threads = 1024; // launch 2
// bytes in flight an SM that cover HX12's loaded DRAM latency at its p90, 2,843 cycles, at 1,036 GB/s over 80 SMs
// [arithmetic, QA1b]
constexpr int32_t flight_bytes = 27000;

static_assert(head_dim == 4 * 32, "a warp holds a row as 4 values a lane");
static_assert(n_rot == 64, "rope pairs (l, l + 32) sit in one lane");
static_assert(max_blocks <= 65536, "block IDs fit 16 bits in the tie select");
static_assert(fn::rms_eps == 1e-6f, "indexer norms use the model's rms eps");

// launch 1's threads a CTA by stream depth: deeper rings hold more registers, so fewer warps
template <int D> constexpr int score_threads() {
    return D <= 4 ? 704 : D == 8 ? 384 : 256;
}

int score_threads_of(int32_t depth) {
    return depth <= 4 ? 704 : depth == 8 ? 384 : 256;
}

// A zero that depends on v; mz is zero for every valid launch. A refill address that carries it cannot be issued
// before v exists, so ptxas keeps D loads in flight instead of hoisting all of a tile's (HC1c).
__device__ __forceinline__ uint32_t order_after(uint32_t v, uint32_t mz) {
    return v & mz;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, off, 32);
    }
    return x;
}

__device__ __forceinline__ uint32_t warp_max_u32(uint32_t x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x = max(x, __shfl_xor_sync(0xffffffffu, x, off, 32));
    }
    return x;
}

// ggml_qsa_select's key: larger float, larger key; subnormals count as +0; 0 means not selectable
__device__ __forceinline__ uint32_t select_key(float v) {
    if (!(v > -INFINITY)) {
        return 0;
    }
    uint32_t b = __float_as_uint(v);
    if ((b & 0x7f800000u) == 0) {
        b = 0;
    }
    return b ^ ((b >> 31) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ float relu(float d) {
    return d > 0.0f ? d : 0.0f;
}

// RMS norm times w, then the indexer rope at pos, of one 128-wide row held as x[m] = column 32m + lane. The operations
// and their order are qsa_pool_rope_f32's (ggml-cuda rope.cu, built with -use_fast_math, as this file is): the
// rms_norm_f32<256> butterfly per 32 columns and over the four warp sums, mean + eps as one FFMA with MUFU.RCP,
// MUFU.RSQ, (rscale * x) * w, theta = freq_scale * (pos * __powf(theta_scale, i)), __cosf and __sinf times attn_factor,
// and the two products of each rotation in the FFMAs that kernel's SASS uses. Columns 64..127 pass through.
__device__ __forceinline__ void norm_rope(float (&x)[4], const float * __restrict__ w, const float eps, const int ncols,
                                          const int pos, const rope_params rp, const int lane) {
    float ws[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        float tmp = 0.0f;
        tmp += x[m] * x[m];
        ws[m] = warp_sum(tmp);
    }
    float tmp = 0.0f;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (lane == m) {
            tmp = ws[m];
        }
    }
    tmp = warp_sum(tmp);
    const float mean = tmp / ncols;
    const float rscale = rsqrtf(mean + eps);
    float y[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        y[m] = rscale * x[m] * w[m * 32 + lane];
    }
    const int iw = 2 * lane;
    const float theta_base = pos * powf(rp.theta_scale, iw / 2.0f);
    const float theta = rp.freq_scale * theta_base;
    const float c = cosf(theta) * rp.attn_factor;
    const float s = sinf(theta) * rp.attn_factor;
    x[0] = __fmaf_rn(y[0], c, -__fmul_rn(y[1], s));
    x[1] = __fmaf_rn(y[0], s, __fmul_rn(y[1], c));
    x[2] = y[2];
    x[3] = y[3];
}

// ---- summary update ----

struct update_kargs {
    const uint16_t * keys;
    const float * k_norm;
    float * summaries;
    int32_t first, count;
    float scale, bias, eps;
    int32_t ncols;
    rope_params rope;
    const float * ik;   // the new cells' raw keys (QA2), F32
    uint16_t * key_out; // where they are stored
    int32_t new_first;  // the first new cell
    int32_t n_new;
    const int32_t * depth; // the round form: first, count, new_first and n_new from the device depth (n_new = rows)
    int32_t cap;           // the round form: the key cache's cells
    const int64_t * shift; // ME6, the round form: optional device word w; cell c at keys + (c - *w) * head_dim
};

// round to nearest even (finite keys), as the cache stores them
__device__ __forceinline__ uint16_t bf16_rn(float f) {
    const uint32_t u = __float_as_uint(f);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// One warp a block: the mean of its 4 cells' raw keys, norm and rope at position 4b, as qsa_pool_rope_f32 computes it
// (members summed in cell order, then scale * x + bias), stored as F32 in the tiled layout.
// FMS2 (build item 2): summary_update_kernel's warp; L, the joint catch-up's lanes launch: the new cells *rows (the
// lane's descriptor's) instead of a.n_new
template <bool L> __device__ __forceinline__ void update_warp(const update_kargs & a, const int32_t * rows) {
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int32_t n_new = L && rows != nullptr ? *rows : a.n_new;
    int first = a.first, count = a.count, new_first = a.new_first;
    if (a.depth != nullptr) {
        new_first = *a.depth;
        if (new_first < 0 || new_first > a.cap - n_new) {
            return; // refused by the runtime before launch (FC-RD1); nothing is written
        }
        first = new_first / ratio;
        count = (new_first + n_new) / ratio - first;
    }
    if (i >= count + n_new) {
        return;
    }
    // ME6: the host-backed tier's window start (graph mode); the window holds every cell the launch touches
    const int64_t shift = a.shift != nullptr ? *a.shift : 0;
    if (i >= count) {
        // a new cell's raw key: the warp stores its 128 values rounded to BF16
        const int n = i - count;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            a.key_out[(static_cast<int64_t>(new_first) + n - shift) * head_dim + m * 32 + lane] =
                bf16_rn(a.ik[n * head_dim + m * 32 + lane]);
        }
        return;
    }
    const int b = first + i;
    float x[4];
#pragma unroll
    for (int r = 0; r < ratio; ++r) {
        const int64_t c = static_cast<int64_t>(b) * ratio + r;
        const int64_t n = c - new_first;
        const uint16_t * k = a.keys + (c - shift) * head_dim;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            // a cell this launch writes is taken from ik, never read back from the cache
            const uint16_t h = n >= 0 && n < n_new ? bf16_rn(a.ik[n * head_dim + m * 32 + lane]) : k[m * 32 + lane];
            const float v = __uint_as_float(static_cast<uint32_t>(h) << 16);
            x[m] = r == 0 ? v : x[m] + v;
        }
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        x[m] = a.scale * x[m] + a.bias;
    }
    norm_rope(x, a.k_norm, a.eps, a.ncols, b * ratio, a.rope, lane);
    float * tile =
        a.summaries + static_cast<int64_t>(b / tile_blocks) * (n_chunk * tile_blocks * 4) + (b % tile_blocks) * 4;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int e = m * 32 + lane;
        tile[(e / 4) * (tile_blocks * 4) + e % 4] = x[m];
    }
}

__global__ void __launch_bounds__(128) summary_update_kernel(const update_kargs a) {
    update_warp<false>(a, nullptr);
}

// FMS2 (build item 2, the joint catch-up): lanes 2 .. 4 in one launch, blockIdx.y the lane, each lane's new cells
// *rows[u] (read on the device: its descriptor's rows); build item 3: rows[u] null, a[u].n_new
struct update_lanes_kargs {
    update_kargs a[fn::xchg::max_lanes];
    const int32_t * rows[fn::xchg::max_lanes];
};

__global__ void __launch_bounds__(128) summary_update_lanes_kernel(const __grid_constant__ update_lanes_kargs p) {
    update_warp<true>(p.a[blockIdx.y], p.rows[blockIdx.y]);
}

// ---- launch 1: score ----

struct score_kargs {
    const uint4 * summaries;
    const float * q;
    const float * q_norm;
    rope_params rope;
    float eps;
    int32_t ncols;
    const int32_t * depth; // device (FC-RD1); the tiles and blocks loaded follow from it
    int32_t max_cells;     // the grid's capacity: depth + B beyond it is refused
    int32_t tpc;           // tiles a CTA: warps 0 .. tpc - 1 own tiles; the others only help with the queries
    uint32_t * keys;
    uint32_t * best1;
    uint32_t * best2;
    uint16_t * lanes;
    uint32_t mz; // an opaque zero (order_after)
};

// acc += the 4 summary elements s times the query elements a, in element order
__device__ __forceinline__ float dot4(const float (&s)[4], float4 a, float acc) {
    acc = fmaf(s[0], a.x, acc);
    acc = fmaf(s[1], a.y, acc);
    acc = fmaf(s[2], a.z, acc);
    acc = fmaf(s[3], a.w, acc);
    return acc;
}

// 16 bytes from a shared-window byte address
__device__ __forceinline__ float4 lds128(uint32_t addr) {
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ uint4 load_chunk(const uint4 * p, bool ok) {
    return ok ? __ldcs(p) : make_uint4(0, 0, 0, 0);
}

// Warp = one 32-block tile (lane = block), tiles g, g + G, ... for global tile warp g (warps past tpc in a CTA only
// normalize queries). Each lane streams its block's 32 chunks with D loads in flight: chunk c's 16 bytes are taken
// into registers when they arrive and its slot is reissued at once for chunk c + D (in this tile or the next), before
// chunk c's FMAs, so D loads stay in flight through the FMAs. The B*4 normalized, rotated queries sit in shared memory
// and every lane reads the same address (broadcast). At the end of a tile each row's 32 keys are written and the
// tile's best two blocks a row (key, then the lower lane).
// QA (SP2, the sparse draft): the query of row t sits at position depth + t + QA; the draft's one query is at its own
// cell N with depth N - 1, so it sees the N / 4 whole blocks of the processed prefix. The target's launches are QA = 0.
// FMS2 (build item 3, W3): the kernel's body, a's tiles over blockIdx.x; L false: score_kernel's, L true: a lane's of
// score_lanes_kernel (its own instantiation, so the lone kernel's code is its own)
template <int B, int D, int QA, bool L> __device__ __forceinline__ void score_body(const score_kargs & a) {
    static_assert(n_chunk % D == 0, "D must divide the chunks");
    constexpr int NV = B * n_idx_head;
    constexpr int LAG = B >= 3 && D == 4 ? 1 : 2;
    __shared__ float4 qs[NV][head_dim / 4];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int W = blockDim.x >> 5;
    const int G = gridDim.x * a.tpc;
    const int depth = *a.depth;
    const int n_vis_max = (depth + B) / ratio; // blocks loaded: 0 .. n_vis_max - 1
    if (depth < 0 || depth > a.max_cells - B || n_vis_max <= top_blocks) {
        return; // no row selects (launch 2 takes every visible block), or a refused depth (launch 2 flags it)
    }
    const int n_tiles = (n_vis_max + tile_blocks - 1) / tile_blocks;
    int tile = warp < a.tpc ? blockIdx.x * a.tpc + warp : n_tiles;

    // this warp's first query first: its reads are not to wait behind the summary stream
    float x0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (warp < NV) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            x0[m] = a.q[warp * head_dim + m * 32 + lane];
        }
    }

    // the first D chunks of this warp's first tile
    uint4 ring[D];
    const uint4 * cur = a.summaries + static_cast<int64_t>(tile) * tile_u4 + lane;
    bool cur_ok = tile < n_tiles && tile * tile_blocks + lane < n_vis_max;
#pragma unroll
    for (int c = 0; c < D; ++c) {
        ring[c] = load_chunk(cur + c * chunk_u4, cur_ok);
    }

    // normalized, rotated queries into shared memory; query row t sits at position depth + t
    for (int v = warp; v < NV; v += W) {
        float x[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            x[m] = v == warp ? x0[m] : a.q[v * head_dim + m * 32 + lane];
        }
        norm_rope(x, a.q_norm, a.eps, a.ncols, depth + v / n_idx_head + QA, a.rope, lane);
        float * row = reinterpret_cast<float *>(qs[v]);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            row[m * 32 + lane] = x[m];
        }
    }
    __syncthreads();

    const uint32_t qs_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&qs[0][0]));
    for (; tile < n_tiles; tile += G) {
        const int next = tile + G;
        const uint4 * nxt = a.summaries + static_cast<int64_t>(next) * tile_u4 + lane;
        const bool next_ok = next < n_tiles && next * tile_blocks + lane < n_vis_max;

        // the queries do not change from tile to tile; an opaque zero keeps ptxas from hoisting their shared reads
        // out of the loop into registers
        const uint32_t qrow = qs_addr + order_after(static_cast<uint32_t>(tile), a.mz);
        float acc[NV];
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            acc[v] = 0.0f;
        }
        // chunks in groups of D: the group loop is not unrolled, so ptxas cannot read later groups' query values
        // ahead into registers (it spills when it does). Chunk c's slot is reissued as soon as chunk c has arrived:
        // the refill address carries an opaque zero of the arrived bytes, so it issues on arrival and not earlier.
        uint32_t z = 0u; // chunk c - 1's last FMA result (raw bits), masked to a zero where it orders
#pragma unroll 1
        for (int c0 = 0; c0 < n_chunk; c0 += D) {
            const bool same = c0 + D < n_chunk;
            const uint4 * src = same ? cur + (c0 + D) * chunk_u4 : nxt;
            const bool src_ok = same ? cur_ok : next_ok;
            const uint32_t qc = qrow + 16u * c0;
#pragma unroll
            for (int j = 0; j < D; ++j) {
                // take: chunk c's bytes leave the ring through an opaque-zero OR (after chunk c - 1's last FMA), so
                // the slot's registers are free and the refill lands in them; without it ptxas keeps the FMAs on the
                // ring registers, puts the refill in new ones and copies them back at the loop end, which waits for
                // the refills to arrive
                const uint4 r = ring[j];
                const uint32_t zz = order_after(z, a.mz);
                const uint32_t w[4] = {r.x | zz, r.y | zz, r.z | zz, r.w | zz};
                ring[j] = load_chunk(src + j * chunk_u4 + order_after(w[0] ^ w[1] ^ w[2] ^ w[3], a.mz), src_ok);
                __syncwarp();
                const float s[4] = {__uint_as_float(w[0]), __uint_as_float(w[1]), __uint_as_float(w[2]),
                                    __uint_as_float(w[3])};
                // vector v's query reads wait for vector v - LAG's FMAs: at most LAG vectors' query values in registers
                // (LAG 1 at B >= 3 keeps D = 4 within 704 threads' registers). The shared byte address ORs in the opaque
                // zero: (x & mz) | qj is one LOP3 (ALU pipe) a vector, and the vector's offset is the load's immediate.
                const uint32_t qj = qc + 16u * j;
#pragma unroll
                for (int v = 0; v < NV; ++v) {
                    const uint32_t x = v < LAG ? z : __float_as_uint(acc[v - LAG]);
                    const uint32_t qa = (x & a.mz) | qj;
                    acc[v] = dot4(s, lds128(qa + static_cast<uint32_t>(v * head_dim * 4)), acc[v]);
                }
                z = __float_as_uint(acc[NV - 1]);
            }
        }

        // keys and the tile's best two blocks, per row
        const int b = tile * tile_blocks + lane;
#pragma unroll
        for (int t = 0; t < B; ++t) {
            const float sc = ((relu(acc[4 * t]) + relu(acc[4 * t + 1])) + relu(acc[4 * t + 2])) + relu(acc[4 * t + 3]);
            const int n_vis = (depth + t + 1) / ratio;
            const uint32_t key = b < n_vis ? select_key(sc) : 0u;
            a.keys[t * max_blocks + b] = key;
            const uint32_t m1 = warp_max_u32(key);
            const uint32_t bal1 = __ballot_sync(0xffffffffu, m1 != 0 && key == m1);
            const int l1 = bal1 != 0 ? __ffs(bal1) - 1 : 0xff;
            const uint32_t k2 = lane == l1 ? 0u : key;
            const uint32_t m2 = warp_max_u32(k2);
            const uint32_t bal2 = __ballot_sync(0xffffffffu, m2 != 0 && k2 == m2);
            const int l2 = bal2 != 0 ? __ffs(bal2) - 1 : 0xff;
            if (lane == 0) {
                a.best1[t * max_tiles + tile] = m1;
                a.best2[t * max_tiles + tile] = m2;
                a.lanes[t * max_tiles + tile] = static_cast<uint16_t>(l1 | (l2 << 8));
            }
        }
        cur = nxt;
        cur_ok = next_ok;
    }
}

template <int B, int D, int QA = 0>
__global__ void __launch_bounds__(score_threads<D>(), 1) score_kernel(const score_kargs a) {
    score_body<B, D, QA, false>(a);
}

// FMS2 (build item 3, W3): the joint verify round's lanes in one launch, blockIdx.y the lane (target form, QA 0)
struct score_lanes_kargs {
    score_kargs a[fn::xchg::max_lanes];
};

template <int B, int D>
__global__ void __launch_bounds__(score_threads<D>(), 1)
score_lanes_kernel(const __grid_constant__ score_lanes_kargs p) {
    score_body<B, D, 0, true>(p.a[blockIdx.y]);
}

// ---- launch 2: select ----

struct select_kargs {
    const uint32_t * keys;
    const uint32_t * best1;
    const uint32_t * best2;
    const uint16_t * lanes;
    uint32_t * error;
    uint32_t * stats;
    uint32_t * body;
    int32_t * offsets;
    const int32_t * depth; // device (FC-RD1)
    int32_t max_cells, rows;
};

constexpr int32_t top_bits = 15;          // the first digit: a key's top 15 bits
constexpr int32_t n_bins = 1 << top_bits; // 16-bit counts, two a word: 64 KB of shared memory
constexpr int32_t bins_per_thread = n_bins / sel_threads;
constexpr int32_t fine_bits = 9;          // the digits after it, only for a bin over fin_cap (ties)
constexpr int32_t fin_cap = 128;          // a bin one warp finishes: four order values a lane
constexpr int32_t fail_regs = 8;          // failing tiles a warp keeps in registers (256 a CTA)
constexpr int32_t fail_packed_max = 2000; // up to this many failing tiles no 16-bit count can wrap
constexpr int32_t order_bits = 48;        // an order value: the key, then the inverted block ID
static_assert(bins_per_thread == 32, "find_top walks 16 count words a thread");
static_assert(2 * sel_threads == max_tiles, "a thread owns two tiles");
static_assert(2 * max_tiles + (tile_blocks - 2) * fail_packed_max < 65536, "16-bit counts");
static_assert((1 << fine_bits) <= sel_threads, "find_small: a bin a thread");

// the body of the row at position q: its count words, min(n_vis, 512) blocks and (q + 1) % 4 tail cells
__host__ __device__ __forceinline__ int32_t row_words_of(int32_t q) {
    const int32_t n_vis = (q + 1) / ratio;
    return 2 + (n_vis < top_blocks ? n_vis : top_blocks) + (q + 1 - n_vis * ratio);
}

// larger first: key descending, then block ascending (block IDs are below 65,536)
__device__ __forceinline__ uint64_t order_value(uint32_t key, int32_t blk) {
    return (static_cast<uint64_t>(key) << 16) | static_cast<uint64_t>(0xffff - blk);
}

// dynamic shared memory of launch 2
struct sel_smem {
    uint32_t hist[n_bins / 2]; // the top digit's counts: bin 2i in the low half of word i, 2i + 1 in the high
    uint32_t words[max_tiles]; // a failing tile's taken blocks, a bit a block
    uint32_t fails[max_tiles]; // failing tiles: the tile, and its best two lanes << 16
    uint64_t fin[fin_cap];     // the threshold bin's order values
    uint16_t sel[top_blocks];  // a row's taken blocks, ascending
    int s_w[2][32];            // warp totals of the CTA scans (two, so a scan never waits on the last one)
    int n_fail, n_fin;
    int bin, rank, cnt, total; // find_top and find_small
    uint64_t tv;               // the 512th order value
};

__device__ __forceinline__ int warp_incl_scan(int v, const int lane) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, v, o, 32);
        if (lane >= o) {
            v += y;
        }
    }
    return v;
}

// Exclusive scan of v over the CTA with one barrier: every warp scans the 32 warp totals itself. sw is read after the
// barrier, so it is not written again before the next barrier (the callers alternate s_w[0] and s_w[1]).
__device__ __forceinline__ int cta_scan(int v, int * total, int * sw) {
    static_assert(sel_threads == 32 * 32, "a warp total a lane");
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int incl = warp_incl_scan(v, lane);
    if (lane == 31) {
        sw[warp] = incl;
    }
    __syncthreads();
    const int wi = warp_incl_scan(sw[lane], lane);
    *total = __shfl_sync(0xffffffffu, wi, 31, 32);
    const int below = __shfl_sync(0xffffffffu, wi, (warp + 31) & 31, 32);
    return (warp > 0 ? below : 0) + incl - v;
}

__device__ __forceinline__ uint32_t word_of(const uint4 & v, int k) {
    return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w;
}

// The top digit's bin that holds the rank-th count from the top (rank >= 1): s.bin, s.rank (the rank within it) and
// s.cnt (its count); s.bin = -1 when fewer than rank were counted. Thread i sums the 32 bins below n_bins - 32 i, one
// scan places the runs, and the thread whose run holds rank walks it. Ends with a barrier.
__device__ void find_top(sel_smem & s, const int rank, int * sw) {
    const int tid = threadIdx.x;
    const int lo = n_bins - bins_per_thread * (tid + 1);
    // the counts as 16-byte vectors; lo / 2 is a multiple of 16 words
    const uint4 * h4 = reinterpret_cast<const uint4 *>(s.hist + lo / 2);
    uint4 v[4];
    int sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = h4[k];
        sum += (v[k].x & 0xffff) + (v[k].x >> 16) + (v[k].y & 0xffff) + (v[k].y >> 16) + (v[k].z & 0xffff) +
               (v[k].z >> 16) + (v[k].w & 0xffff) + (v[k].w >> 16);
    }
    int total = 0;
    int c = cta_scan(sum, &total, sw);
    if (tid == 0) {
        s.total = total;
        if (total < rank) {
            s.bin = -1;
        }
    }
    if (c < rank && rank <= c + sum) {
        // the owner reads its run again rather than keep 16 registers live across the scan
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = h4[k];
        }
#pragma unroll
        for (int k = 15; k >= 0; --k) {
            const uint32_t w = word_of(v[k >> 2], k & 3);
            const int hi = static_cast<int>(w >> 16);
            const int lw = static_cast<int>(w & 0xffff);
            if (c < rank && rank <= c + hi) {
                s.bin = lo + 2 * k + 1;
                s.rank = rank - c;
                s.cnt = hi;
            }
            c += hi;
            if (c < rank && rank <= c + lw) {
                s.bin = lo + 2 * k;
                s.rank = rank - c;
                s.cnt = lw;
            }
            c += lw;
        }
    }
    __syncthreads();
}

// find_top for nb <= 1,024 32-bit counts at s.hist[0 .. nb - 1], a bin a thread. Ends with a barrier.
__device__ void find_small(sel_smem & s, const int nb, const int rank, int * sw) {
    const int tid = threadIdx.x;
    const int bin = nb - 1 - tid;
    const int h = tid < nb ? static_cast<int>(s.hist[bin]) : 0;
    int total = 0;
    const int c = cta_scan(h, &total, sw);
    if (tid == 0) {
        s.total = total;
        if (total < rank) {
            s.bin = -1;
        }
    }
    if (h > 0 && c < rank && rank <= c + h) {
        s.bin = bin;
        s.rank = rank - c;
        s.cnt = h;
    }
    __syncthreads();
}

// Row t's 512 blocks. Each tile lists its best two blocks. T1 is the lower edge of the top-digit bin that holds the
// 512th listed block, so at most the true 512th: a tile whose second best is below T1 has no other block that can be
// taken, and the rest fail. The candidates C (the listed blocks of passing tiles and every block of failing tiles)
// hold the true top 512; their counts at or above T1's bin are the listed counts plus the failing tiles' unlisted
// blocks, so one more scan of the same histogram finds the bin of C's 512th, and one warp finishes inside it (a bin
// over fin_cap, which takes ties, is narrowed by 9-bit passes first). A thread owns tiles 2 tid and 2 tid + 1.
__device__ void select_row(const select_kargs & a, const int t, sel_smem & s) {
    constexpr int NT = sel_threads;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int depth = *a.depth;
    const int q = depth + t;
    const int n_vis = (q + 1) / ratio;
    const int nt = (n_vis + tile_blocks - 1) / tile_blocks;
    const int tail_start = n_vis * ratio;
    const int n_t = q + 1 - tail_start;
    const int n_sel = min(n_vis, top_blocks);
    const bool big = n_vis > top_blocks;
    const int w0 = 2 * tid;
    const uint32_t * keys_row = a.keys + static_cast<int64_t>(t) * max_blocks;
    int off = 0; // the row's word offset in the body: the rows before it, packed
    for (int r = 0; r < t; ++r) {
        off += row_words_of(depth + r);
    }
    uint32_t * out = a.body + off;

    // the tiles' lists first, so their L2 latency overlaps the writes below; two tiles' entries are 8-byte aligned
    // (max_tiles and w0 are even) and their lanes 4-byte aligned. A tile at or past nt is not this row's: launch 1 may
    // not have written its list this time (an odd tile count), so its entries are cleared.
    uint2 k1 = make_uint2(0u, 0u);
    uint2 k2 = make_uint2(0u, 0u);
    uint32_t ln = 0xffffffffu;
    if (big && w0 < nt) {
        const int64_t i = static_cast<int64_t>(t) * max_tiles + w0;
        k1 = __ldcg(reinterpret_cast<const uint2 *>(a.best1 + i));
        k2 = __ldcg(reinterpret_cast<const uint2 *>(a.best2 + i));
        ln = __ldcg(reinterpret_cast<const uint32_t *>(a.lanes + i));
        if (w0 + 1 >= nt) {
            k1.y = 0u;
            k2.y = 0u;
        }
    }
    // the counts, the tail and the offsets do not depend on the selection
    if (tid == 0) {
        out[0] = static_cast<uint32_t>(n_sel);
        out[1] = static_cast<uint32_t>(n_t);
        a.offsets[t] = 4 * off;
        if (t == a.rows - 1) {
            a.offsets[a.rows] = 4 * (off + 2 + n_sel + n_t);
        }
    }
    if (tid < n_t) {
        out[2 + n_sel + tid] = static_cast<uint32_t>(tail_start + tid);
    }
    if (big) {
        uint4 * h4 = reinterpret_cast<uint4 *>(s.hist); // n_bins / 2 words, 16-byte aligned (first member)
        for (int i = tid; i < n_bins / 8; i += NT) {
            h4[i] = make_uint4(0u, 0u, 0u, 0u);
        }
        if (tid == 0) {
            s.n_fail = 0;
            s.n_fin = 0;
        }
    }
    __syncthreads();

    uint32_t word[2] = {0u, 0u};
    if (!big) {
        if (tid < n_stats) {
            a.stats[t * n_stats + tid] = 0;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int w = w0 + i;
            if (w < nt) {
                const int nb = min(tile_blocks, n_vis - w * tile_blocks);
                word[i] = nb == 32 ? 0xffffffffu : (1u << nb) - 1u;
            }
        }
    } else {
        // the listed entries: tile w0's best and second, then tile w0 + 1's (a key is 0 where its lane is 0xff)
        const uint32_t ek[4] = {k1.x, k2.x, k1.y, k2.y};
        // entry i's block: lanes hold l1 and l2 of tile w0, then of w0 + 1, a byte each (recomputed, not kept)
        const auto eb = [&](int i) { return (w0 + (i >> 1)) * tile_blocks + static_cast<int>((ln >> (8 * i)) & 0xff); };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (ek[i] != 0) {
                const uint32_t d = ek[i] >> (32 - top_bits);
                atomicAdd(&s.hist[d >> 1], (d & 1) ? 0x10000u : 1u);
            }
        }
        __syncthreads();
        find_top(s, top_blocks, s.s_w[0]);
        // T1's bin; fewer than 512 listed: every tile with more than two blocks fails
        const int b1 = max(s.bin, 0);
        const int listed = s.total;
        const int bin1 = s.bin;

        bool fail[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int w = w0 + i;
            const int nb = w < nt ? min(tile_blocks, n_vis - w * tile_blocks) : 0;
            const uint32_t z = ek[2 * i + 1];
            fail[i] = nb > 2 && z != 0 && static_cast<int>(z >> (32 - top_bits)) >= b1;
            if (fail[i]) {
                const int f = atomicAdd(&s.n_fail, 1);
                s.fails[f] = static_cast<uint32_t>(w) | (((ln >> (16 * i)) & 0xffffu) << 16);
            }
        }
        __syncthreads();
        const int n_fail = s.n_fail;

        // a warp takes failing tiles warp, warp + 32, ..., lane = block: their keys, read once into registers
        uint32_t fk[fail_regs];
#pragma unroll
        for (int j = 0; j < fail_regs; ++j) {
            const int f = warp + 32 * j;
            fk[j] = 0u;
            if (f < n_fail) {
                const int blk = static_cast<int>(s.fails[f] & 0xffff) * tile_blocks + lane;
                fk[j] = blk < n_vis ? __ldg(keys_row + blk) : 0u;
            }
        }
        // every candidate's order value, to f (the failing tiles past fail_regs a warp are read again each time)
        const auto visit = [&](auto && fn) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (ek[i] != 0 && !fail[i >> 1]) {
                    fn(order_value(ek[i], eb(i)));
                }
            }
#pragma unroll
            for (int j = 0; j < fail_regs; ++j) {
                const int f = warp + 32 * j;
                if (f < n_fail && fk[j] != 0) {
                    fn(order_value(fk[j], static_cast<int>(s.fails[f] & 0xffff) * tile_blocks + lane));
                }
            }
            for (int f = warp + 32 * fail_regs; f < n_fail; f += 32) {
                const int blk = static_cast<int>(s.fails[f] & 0xffff) * tile_blocks + lane;
                const uint32_t key = blk < n_vis ? __ldg(keys_row + blk) : 0u;
                if (key != 0) {
                    fn(order_value(key, blk));
                }
            }
        };

        // T2: the failing tiles' unlisted blocks at or above T1's bin join the counts
        const bool packed = n_fail <= fail_packed_max;
        if (packed) {
            const auto add = [&](uint32_t key, int f) {
                const uint32_t info = s.fails[f];
                const int l1 = static_cast<int>((info >> 16) & 0xff);
                const int l2 = static_cast<int>(info >> 24);
                const uint32_t d = key >> (32 - top_bits);
                if (key != 0 && lane != l1 && lane != l2 && static_cast<int>(d) >= b1) {
                    atomicAdd(&s.hist[d >> 1], (d & 1) ? 0x10000u : 1u);
                }
            };
#pragma unroll
            for (int j = 0; j < fail_regs; ++j) {
                if (warp + 32 * j < n_fail) {
                    add(fk[j], warp + 32 * j);
                }
            }
            for (int f = warp + 32 * fail_regs; f < n_fail; f += 32) {
                const int blk = static_cast<int>(s.fails[f] & 0xffff) * tile_blocks + lane;
                add(blk < n_vis ? __ldg(keys_row + blk) : 0u, f);
            }
            __syncthreads();
            find_top(s, top_blocks, s.s_w[1]);
            if (s.bin < 0 && tid == 0) {
                *a.error = 2u; // cannot happen: C holds every block at or above T1, at least 512
            }
        }
        // the threshold bin: order values whose top order_bits - sh bits are prefix
        uint64_t prefix = packed ? static_cast<uint64_t>(max(s.bin, 0)) : 0u;
        int sh = packed ? order_bits - top_bits : order_bits;
        int rank = packed ? s.rank : top_blocks;
        int cnt = packed ? s.cnt : INT_MAX;
        const int cnt2 = cnt;
        int passes = 0;
        while (cnt > fin_cap && sh > 0) {
            ++passes;
            const int bits = min(fine_bits, sh);
            const int nb = 1 << bits;
            const int nsh = sh - bits;
            for (int i = tid; i < nb; i += NT) {
                s.hist[i] = 0u;
            }
            __syncthreads();
            visit([&](uint64_t v) {
                if ((v >> sh) == prefix) {
                    atomicAdd(&s.hist[static_cast<int>(v >> nsh) & (nb - 1)], 1u);
                }
            });
            __syncthreads();
            find_small(s, nb, rank, s.s_w[0]);
            prefix = (prefix << bits) | static_cast<uint64_t>(max(s.bin, 0));
            sh = nsh;
            rank = s.rank;
            cnt = s.cnt;
        }
        uint64_t tv = prefix; // sh == 0: the bin is one order value
        if (sh > 0) {
            visit([&](uint64_t v) {
                if ((v >> sh) == prefix) {
                    const int i = atomicAdd(&s.n_fin, 1);
                    if (i < fin_cap) {
                        s.fin[i] = v;
                    }
                }
            });
            __syncthreads();
            // one warp: the rank-th of the bin's m values is the one with rank - 1 values above it
            if (warp == 0) {
                const int m = min(s.n_fin, fin_cap);
                uint64_t mine[fin_cap / 32];
                int above[fin_cap / 32];
#pragma unroll
                for (int j = 0; j < fin_cap / 32; ++j) {
                    mine[j] = lane + 32 * j < m ? s.fin[lane + 32 * j] : 0u;
                    above[j] = 0;
                }
                const int nj = (m + 31) >> 5;
#pragma unroll 4
                for (int i = 0; i < m; ++i) {
                    const uint64_t o = s.fin[i];
#pragma unroll
                    for (int j = 0; j < fin_cap / 32; ++j) {
                        if (j < nj) {
                            above[j] += o > mine[j];
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < fin_cap / 32; ++j) {
                    if (lane + 32 * j < m && above[j] == rank - 1) {
                        s.tv = mine[j];
                    }
                }
                if (lane == 0 && s.n_fin != cnt) {
                    *a.error = 3u; // the gather missed the bin's count
                }
            }
            __syncthreads();
            tv = s.tv;
        }

        // the taken blocks: passing tiles from their two listed entries, failing tiles by their warp's ballot
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (w0 + i < nt && !fail[i]) {
#pragma unroll
                for (int e = 2 * i; e < 2 * i + 2; ++e) {
                    if (ek[e] != 0 && order_value(ek[e], eb(e)) >= tv) {
                        word[i] |= 1u << (eb(e) & 31);
                    }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < fail_regs; ++j) {
            const int f = warp + 32 * j;
            if (f < n_fail) {
                const int w = static_cast<int>(s.fails[f] & 0xffff);
                const uint32_t bits =
                    __ballot_sync(0xffffffffu, fk[j] != 0 && order_value(fk[j], w * tile_blocks + lane) >= tv);
                if (lane == 0) {
                    s.words[w] = bits;
                }
            }
        }
        for (int f = warp + 32 * fail_regs; f < n_fail; f += 32) {
            const int w = static_cast<int>(s.fails[f] & 0xffff);
            const int blk = w * tile_blocks + lane;
            const uint32_t key = blk < n_vis ? __ldg(keys_row + blk) : 0u;
            const uint32_t bits = __ballot_sync(0xffffffffu, key != 0 && order_value(key, blk) >= tv);
            if (lane == 0) {
                s.words[w] = bits;
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (fail[i]) {
                word[i] = s.words[w0 + i];
            }
        }
        if (tid == 0) {
            uint32_t * st = a.stats + t * n_stats;
            st[0] = listed;
            st[1] = static_cast<uint32_t>(bin1);
            st[2] = n_fail;
            st[3] = packed;
            st[4] = static_cast<uint32_t>(cnt2);
            st[5] = passes;
            st[6] = sh > 0 ? s.n_fin : 0;
            st[7] = 1;
        }
    }

    int n_got = 0;
    int r = cta_scan(__popc(word[0]) + __popc(word[1]), &n_got, s.s_w[1]);
    if (tid == 0 && n_got != n_sel) {
        *a.error = 1u; // the selection's self-check
    }
    // the taken blocks into a list in ascending order, then into the body, coalesced
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        uint32_t bits = word[i];
        while (bits != 0) {
            if (r < top_blocks) {
                s.sel[r] = static_cast<uint16_t>((w0 + i) * tile_blocks + __ffs(bits) - 1);
            }
            bits &= bits - 1;
            ++r;
        }
    }
    __syncthreads();
    for (int j = tid; j < n_sel; j += NT) {
        out[2 + j] = s.sel[j];
    }
}

__global__ void __launch_bounds__(sel_threads, 1) select_kernel(const select_kargs a) {
    extern __shared__ uint4 dyn[];
    sel_smem & s = *reinterpret_cast<sel_smem *>(dyn); // the dynamic shared memory is one sel_smem
    const int depth = *a.depth;
    if (depth < 0 || depth > a.max_cells - a.rows) {
        if (threadIdx.x == 0) {
            *a.error = 4u; // a depth the runtime refuses (FC-RD1): nothing is selected
        }
        return;
    }
    select_row(a, blockIdx.x, s);
}

// FMS2 (build item 3, W3): the joint verify round's lanes in one launch, blockIdx.y the lane: select_kernel's CTA
struct select_lanes_kargs {
    select_kargs a[fn::xchg::max_lanes];
};

__global__ void __launch_bounds__(sel_threads, 1) select_lanes_kernel(const __grid_constant__ select_lanes_kargs p) {
    extern __shared__ uint4 dyn[];
    sel_smem & s = *reinterpret_cast<sel_smem *>(dyn);
    const select_kargs & a = p.a[blockIdx.y];
    const int depth = *a.depth;
    if (depth < 0 || depth > a.max_cells - a.rows) {
        if (threadIdx.x == 0) {
            *a.error = 4u;
        }
        return;
    }
    select_row(a, blockIdx.x, s);
}

// ---- test support ----

__global__ void oracle_kernel(const int op, const float * x, const float * y, float * out, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float a = x[i];
    const float b = y[i];
    float r = 0.0f;
    switch (op) {
        case 0:
            r = rsqrtf(a);
            break;
        case 1:
            r = a / b;
            break;
        case 2:
            r = powf(a, b);
            break;
        case 3:
            r = cosf(a);
            break;
        case 4:
            r = sinf(a);
            break;
        default:
            break;
    }
    out[i] = r;
}

bool aligned16(const void * p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <int B, int D, int QA> bool launch_score(const score_grid & g, const score_kargs & k, cudaStream_t st) {
    score_kernel<B, D, QA><<<g.ctas, g.warps * 32, 0, st>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <int B, int QA = 0> bool launch_score_b(const score_grid & g, const score_kargs & k, cudaStream_t st) {
    switch (g.depth) {
        case 4:
            return launch_score<B, 4, QA>(g, k, st);
        case 8:
            return launch_score<B, 8, QA>(g, k, st);
        case 16:
            return launch_score<B, 16, QA>(g, k, st);
        default:
            return false;
    }
}

} // namespace

rope_params model_rope() {
    // ggml_cuda_op_rope / qsa_pool_rope: theta_scale = powf(freq_base, -2.0f / n_dims) on the host
    return {std::pow(rope_freq_base, -2.0f / static_cast<float>(n_rot)), 1.0f, 1.0f};
}

int32_t blocks_complete(int32_t n_cells) {
    return n_cells / ratio;
}

int32_t row_words(int32_t q) {
    return row_words_of(q);
}

int32_t body_offset(int32_t depth, int32_t t) {
    int32_t off = 0;
    for (int32_t r = 0; r < t; ++r) {
        off += 4 * row_words_of(depth + r);
    }
    return off;
}

int32_t body_words(int32_t depth, int32_t rows) {
    return body_offset(depth, rows) / 4;
}

size_t summary_bytes(int32_t n_blocks) {
    const size_t tiles = (static_cast<size_t>(n_blocks) + tile_blocks - 1) / tile_blocks;
    return tiles * static_cast<size_t>(tile_u4) * 16;
}

void invalidate_from(summary_index & idx, int32_t first) {
    if (first < idx.n_valid) {
        idx.n_valid = first < 0 ? 0 : first;
    }
}

bool mark_updated(summary_index & idx, int32_t first, int32_t count) {
    if (first < 0 || count < 1 || first > idx.n_valid || first + count > max_blocks) {
        return false;
    }
    idx.n_valid = std::max(idx.n_valid, first + count);
    return true;
}

bool summary_update(const update_args & a, void * stream) {
    if (stream == nullptr || a.keys == nullptr || a.k_norm == nullptr || a.summaries == nullptr || a.first < 0 ||
        a.count < 1 || a.first + a.count > max_blocks || !aligned16(a.summaries)) {
        return false;
    }
    const update_kargs k = {a.keys,  a.k_norm, a.summaries, a.first, a.count, 1.0f / static_cast<float>(ratio),
                            0.0f,    fn::rms_eps, head_dim, a.rope,  nullptr, nullptr,
                            0,       0,           nullptr,  0,           nullptr};
    const int warps = 4;
    summary_update_kernel<<<(a.count + warps - 1) / warps, warps * 32, 0, static_cast<cudaStream_t>(stream)>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool summary_update(const round_update_args & a, void * stream) {
    if (stream == nullptr || a.keys == nullptr || a.k_norm == nullptr || a.summaries == nullptr ||
        !aligned16(a.summaries) || a.ik == nullptr || a.depth == nullptr || a.rows < 1 || a.rows > max_rows ||
        a.max_cells < a.rows || a.max_cells > max_cells) {
        return false;
    }
    const update_kargs k = {a.keys, a.k_norm,    a.summaries, 0,      0,       1.0f / static_cast<float>(ratio),
                            0.0f,   fn::rms_eps, head_dim,    a.rope, a.ik,    a.keys,
                            0,      a.rows,      a.depth,     a.max_cells, a.shift};
    // the most warps any depth needs: the blocks rows cells can complete (one for rows <= 4, SP2: two at 5) pooled and
    // rows cells written
    const int warps = 4;
    const int n = (a.rows + ratio - 1) / ratio + a.rows;
    summary_update_kernel<<<(n + warps - 1) / warps, warps * 32, 0, static_cast<cudaStream_t>(stream)>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool summary_update_lanes(int32_t lanes, const round_update_args * a, const int32_t * const * rows, void * stream) {
    if (a == nullptr || lanes < 2 || lanes > fn::xchg::max_lanes || stream == nullptr) {
        return false;
    }
    update_lanes_kargs p = {};
    for (int32_t u = 0; u < lanes; ++u) {
        const round_update_args & x = a[u];
        if ((rows != nullptr && rows[u] == nullptr) || x.keys == nullptr || x.k_norm == nullptr ||
            x.summaries == nullptr || !aligned16(x.summaries) || x.ik == nullptr || x.depth == nullptr || x.rows < 1 ||
            x.rows > max_rows || (rows != nullptr && x.rows != max_rows) || x.max_cells < x.rows ||
            x.max_cells > max_cells || x.shift != nullptr) {
            return false;
        }
        p.a[u] = {x.keys,   x.k_norm, x.summaries, 0,      0, 1.0f / static_cast<float>(ratio), 0.0f,    fn::rms_eps,
                  head_dim, x.rope,   x.ik,        x.keys, 0, rows == nullptr ? x.rows : 0,     x.depth, x.max_cells,
                  nullptr};
        p.rows[u] = rows == nullptr ? nullptr : rows[u];
    }
    const int warps = 4;
    const int n = (max_rows + ratio - 1) / ratio + max_rows; // summary_update's grid at its most rows
    summary_update_lanes_kernel<<<dim3((n + warps - 1) / warps, static_cast<uint32_t>(lanes)), warps * 32, 0,
                                  static_cast<cudaStream_t>(stream)>>>(p);
    return cudaPeekAtLastError() == cudaSuccess;
}

size_t workspace_bytes() {
    const size_t rb = static_cast<size_t>(max_rows) * max_blocks * 4;
    const size_t rt = static_cast<size_t>(max_rows) * max_tiles;
    return rb + rt * 4 * 2 + rt * 2 + 64 + static_cast<size_t>(max_rows) * n_stats * 4;
}

workspace carve(void * base) {
    uint8_t * p = static_cast<uint8_t *>(base);
    const size_t rt = static_cast<size_t>(max_rows) * max_tiles;
    workspace w = {};
    w.keys = reinterpret_cast<uint32_t *>(p);
    p += static_cast<size_t>(max_rows) * max_blocks * 4;
    w.best1 = reinterpret_cast<uint32_t *>(p);
    p += rt * 4;
    w.best2 = reinterpret_cast<uint32_t *>(p);
    p += rt * 4;
    w.lanes = reinterpret_cast<uint16_t *>(p);
    p += rt * 2;
    w.error = reinterpret_cast<uint32_t *>(p);
    p += 64;
    w.stats = reinterpret_cast<uint32_t *>(p);
    return w;
}

score_grid grid_of(int32_t rows, int32_t depth) {
    score_grid g = {};
    const int32_t n_vis_max = (depth + rows) / ratio;
    g.tiles = (n_vis_max + tile_blocks - 1) / tile_blocks;
    const int32_t tps = std::max((g.tiles + n_sm - 1) / n_sm, 1);
    // the shallowest stream depth whose loads in flight through the FMAs (a tile warp holds D chunks of 512 bytes)
    // cover flight_bytes an SM, and tiles a CTA
    for (const int32_t d : {4, 8, 16}) {
        g.depth = d;
        g.tpc = std::min(tps, score_threads_of(d) / 32);
        if (g.tpc * d * tile_blocks * 16 >= flight_bytes) {
            break;
        }
    }
    // extra warps take the queries' norm and rope when a CTA owns fewer tiles than there are query vectors
    g.warps = std::max(g.tpc, std::min(rows * n_idx_head, score_threads_of(g.depth) / 32));
    g.ctas = std::min(n_sm, (g.tiles + g.tpc - 1) / g.tpc);
    return g;
}

int32_t select_launches(int32_t rows, int32_t depth) {
    return (depth + rows) / ratio > top_blocks ? 2 : 1;
}

bool select(int32_t rows, const select_args & a, void * stream) {
    if (rows < 1 || rows > max_rows || (a.q_ahead != 0 && (a.q_ahead != 1 || rows != 1)) || stream == nullptr ||
        a.summaries == nullptr || a.q == nullptr || a.q_norm == nullptr || a.body == nullptr || a.offsets == nullptr ||
        a.ws.keys == nullptr || a.depth == nullptr || a.max_cells < rows || a.max_cells > max_cells ||
        !aligned16(a.summaries) ||
        !aligned16(a.q) || !aligned16(a.body) || !aligned16(a.offsets)) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    // launch 1 at the grid of the deepest depth the launch may see; at a shallower depth its CTAs past the depth's tiles
    // exit, and with no row past 512 blocks every CTA returns at once
    const score_grid g = grid_of(rows, a.max_cells - rows);
    const score_kargs sk = {reinterpret_cast<const uint4 *>(a.summaries),
                            a.q,
                            a.q_norm,
                            a.rope,
                            fn::rms_eps,
                            head_dim,
                            a.depth,
                            a.max_cells,
                            g.tpc,
                            a.ws.keys,
                            a.ws.best1,
                            a.ws.best2,
                            a.ws.lanes,
                            0u};
    bool ok = false;
    switch (rows) {
        case 1:
            ok = a.q_ahead != 0 ? launch_score_b<1, 1>(g, sk, st) : launch_score_b<1>(g, sk, st);
            break;
        case 2:
            ok = launch_score_b<2>(g, sk, st);
            break;
        case 3:
            ok = launch_score_b<3>(g, sk, st);
            break;
        case 4:
            ok = launch_score_b<4>(g, sk, st);
            break;
        case 5:
            ok = launch_score_b<5>(g, sk, st); // SP2
            break;
        default:
            break;
    }
    if (!ok) {
        return false;
    }
    const select_kargs k = {a.ws.keys,  a.ws.best1, a.ws.best2, a.ws.lanes, a.ws.error, a.ws.stats,
                            a.body,     a.offsets,  a.depth,    a.max_cells, rows};
    // more than 48 KB of dynamic shared memory needs the opt-in; cudaFuncSetAttribute sets it for the current device
    // only, so once a device (R14: a process that runs select on both cards)
    static std::atomic<uint32_t> smem_done{0u};
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        return false;
    }
    if ((smem_done.load(std::memory_order_acquire) >> dev & 1u) == 0u) {
        if (cudaFuncSetAttribute(select_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(sizeof(sel_smem))) != cudaSuccess) {
            return false;
        }
        smem_done.fetch_or(1u << dev, std::memory_order_release);
    }
    select_kernel<<<rows, sel_threads, sizeof(sel_smem), st>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

namespace {

template <int B, int D>
bool launch_score_lanes(const score_grid & g, int32_t lanes, const score_lanes_kargs & k, cudaStream_t st) {
    score_lanes_kernel<B, D><<<dim3(g.ctas, lanes), g.warps * 32, 0, st>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

// FMS13: the score launch at the round's rows B (2 .. max_rows) and depth D
template <int B>
bool launch_score_lanes_b(const score_grid & g, int32_t lanes, const score_lanes_kargs & k, cudaStream_t st) {
    switch (g.depth) {
        case 4:
            return launch_score_lanes<B, 4>(g, lanes, k, st);
        case 8:
            return launch_score_lanes<B, 8>(g, lanes, k, st);
        case 16:
            return launch_score_lanes<B, 16>(g, lanes, k, st);
        default:
            return false;
    }
}

bool select_args_ok(int32_t rows, const select_args & a) {
    return rows >= 1 && rows <= max_rows && (a.q_ahead == 0 || (a.q_ahead == 1 && rows == 1)) &&
           a.summaries != nullptr && a.q != nullptr && a.q_norm != nullptr && a.body != nullptr &&
           a.offsets != nullptr && a.ws.keys != nullptr && a.depth != nullptr && a.max_cells >= rows &&
           a.max_cells <= max_cells && aligned16(a.summaries) && aligned16(a.q) && aligned16(a.body) &&
           aligned16(a.offsets);
}

} // namespace

score_grid grid_lanes(int32_t rows, int32_t depth, int32_t lanes, int32_t test_depth) {
    score_grid g = {};
    const int32_t n_vis_max = (depth + rows) / ratio;
    g.tiles = (n_vis_max + tile_blocks - 1) / tile_blocks;
    // grid_of's choice over every lane's tiles: tiles a CTA when the lanes share the SMs
    const int32_t tps = std::max((lanes * g.tiles + n_sm - 1) / n_sm, 1);
    for (const int32_t d : {4, 8, 16}) {
        g.depth = d;
        g.tpc = std::min(tps, score_threads_of(d) / 32);
        if (test_depth == 0 ? g.tpc * d * tile_blocks * 16 >= flight_bytes : d == test_depth) {
            break;
        }
    }
    g.warps = std::max(g.tpc, std::min(rows * n_idx_head, score_threads_of(g.depth) / 32));
    // a lane's CTAs: its tiles' share of the SMs (one CTA an SM), at least one
    g.ctas = std::max(1, std::min(std::max(n_sm / std::max(lanes, 1), 1), (g.tiles + g.tpc - 1) / g.tpc));
    return g;
}

bool select_lanes(int32_t lanes, int32_t rows, const select_args * a, void * stream, int32_t test_depth) {
    if (a == nullptr || lanes < 2 || lanes > fn::xchg::max_lanes || rows < 2 || rows > max_rows || stream == nullptr ||
        (test_depth != 0 && test_depth != 4 && test_depth != 8 && test_depth != 16)) {
        return false;
    }
    int32_t deepest = 0;
    for (int32_t u = 0; u < lanes; ++u) {
        if (!select_args_ok(rows, a[u]) || a[u].q_ahead != 0) {
            return false;
        }
        deepest = std::max(deepest, a[u].max_cells - rows);
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const score_grid   g  = grid_lanes(rows, deepest, lanes, test_depth);
    score_lanes_kargs sk = {};
    select_lanes_kargs k = {};
    for (int32_t u = 0; u < lanes; ++u) {
        const select_args & x = a[u];
        sk.a[u] = {reinterpret_cast<const uint4 *>(x.summaries),
                   x.q,
                   x.q_norm,
                   x.rope,
                   fn::rms_eps,
                   head_dim,
                   x.depth,
                   x.max_cells,
                   g.tpc,
                   x.ws.keys,
                   x.ws.best1,
                   x.ws.best2,
                   x.ws.lanes,
                   0u};
        k.a[u]                = { x.ws.keys, x.ws.best1, x.ws.best2, x.ws.lanes,  x.ws.error, x.ws.stats,
                                  x.body,    x.offsets,  x.depth,    x.max_cells, rows };
    }
    bool ok = false;
    switch (rows) {
        case 2:
            ok = launch_score_lanes_b<2>(g, lanes, sk, st);
            break;
        case 3:
            ok = launch_score_lanes_b<3>(g, lanes, sk, st);
            break;
        case 4:
            ok = launch_score_lanes_b<4>(g, lanes, sk, st);
            break;
        default:
            ok = launch_score_lanes_b<max_rows>(g, lanes, sk, st);
            break;
    }
    if (!ok) {
        return false;
    }
    static std::atomic<uint32_t> smem_done{0u}; // select()'s opt-in, for this kernel (once a device)
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        return false;
    }
    if ((smem_done.load(std::memory_order_acquire) >> dev & 1u) == 0u) {
        if (cudaFuncSetAttribute(select_lanes_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(sizeof(sel_smem))) != cudaSuccess) {
            return false;
        }
        smem_done.fetch_or(1u << dev, std::memory_order_release);
    }
    select_lanes_kernel<<<dim3(rows, lanes), sel_threads, sizeof(sel_smem), st>>>(k);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool fast_math_oracle(int32_t op, const float * x, const float * y, float * out, int32_t n, void * stream) {
    if (op < 0 || op > 4 || x == nullptr || y == nullptr || out == nullptr || n < 1 || stream == nullptr) {
        return false;
    }
    oracle_kernel<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(op, x, y, out, n);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::qsa

// ---- PB1: the index over prompt rows (qsa_prompt.h) ----
// New kernels only: the decode kernels above are unchanged; the prompt scorer repeats their per-row arithmetic
// (norm_rope, the heads' sum in order, select_key, the tile lists) with its products on tensor cores (PB3), and the
// selection is select_kernel.

namespace fn::qsa {
namespace {

// a warp a new cell: its 128 raw keys rounded to BF16 at cell pos0 + i (summary_update_kernel's store)
__global__ void __launch_bounds__(128) prompt_keys_kernel(int32_t       rows,
                                                          int32_t       pos0,
                                                          const float * ik,
                                                          uint16_t *    keys) {
    const int32_t lane = threadIdx.x & 31;
    const int32_t i    = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (i >= rows) {
        return;
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        keys[(static_cast<int64_t>(pos0) + i) * head_dim + m * 32 + lane] =
            bf16_rn(ik[int64_t(i) * head_dim + m * 32 + lane]);
    }
}

// a warp a query vector v = 4 row + head, at position pos0 + row (score_kernel's query preparation), then (PB3) its
// FP16 copy for the tensor scorer: scaled by the power of two that brings its largest magnitude into [2^14, 2^15)
// (fn::tc::range_exp), rounded once; qs[v] the scale's inverse
__global__ void __launch_bounds__(256) prompt_queries_kernel(int32_t       rows,
                                                             int32_t       pos0,
                                                             const float * iq,
                                                             const float * q_norm,
                                                             rope_params   rope,
                                                             uint16_t *    q16,
                                                             float *       qs) {
    const int lane = threadIdx.x & 31;
    const int v    = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (v >= rows * n_idx_head) {
        return;
    }
    float x[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        x[m] = iq[int64_t(v) * head_dim + m * 32 + lane];
    }
    norm_rope(x, q_norm, fn::rms_eps, head_dim, pos0 + v / n_idx_head, rope, lane);
    const int32_t e = fn::tc::range_exp(
        fn::tc::warp_max(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])))));
    const float sc = fn::tc::pow2(e);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        q16[int64_t(v) * head_dim + m * 32 + lane] = __half_as_ushort(__float2half_rn(x[m] * sc));
    }
    if (lane == 0) {
        qs[v] = fn::tc::pow2(-e);
    }
}

// ---- PB3: the prompt scorer on tensor cores (Amendment 2: FP16 operands, F32 accumulation) ----
// A CTA: 32 slab rows (row group blockIdx.x) and up to sc_tpc consecutive 32-block tiles (blockIdx.y). Warp w owns rows
// 4 w .. 4 w + 3 of the group, all four heads: its 16 query vectors (m = 4 row + head) are the A operand, held in
// registers for the whole K = 128 (m8n8k4: a lane holds row 8 (qp & 1) + ra of its quadpair, 4 values a k-step). Each
// tile's 32 summaries are read once (the tiled F32 layout: chunk c of block b at [c][b]), each block scaled by the
// power of two that brings its largest magnitude into [2^14, 2^15), rounded once to FP16 into shared memory as
// [block][dim] (B .col: a lane reads its column's 8 values of two k-steps, one LDS.128). S = Q K^T per head, then the
// F32 epilogue: dot = acc x qs x ks, the heads' relu sum in prompt_score_kernel's order (h0 + h1, + h2, + h3; a lane
// holds heads (lane & 1) and (lane & 1) + 2 of its row, the pair lane ^ 1 the other two), the key, and the tile's best
// two by (key, then the lower block) over the 8 lanes that hold the row. Rows that take every block, and tiles past a
// row's visibility, write nothing (select reads neither), as before.
constexpr int32_t sc_tpc    = 8;    // tiles a CTA
constexpr int32_t sc_rows   = 32;   // slab rows a CTA
constexpr int32_t sc_stride = 136;  // halves a shared block row (128 + 8: 272 bytes, LDS.128 aligned)

__device__ __forceinline__ bool sc_better(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib) {
    return a > b || (a == b && ia < ib);
}

// (k1, i1) >= (k2, i2) a lane's best two; merge the candidate (k, i)
__device__ __forceinline__ void sc_push(uint32_t & k1, uint32_t & i1, uint32_t & k2, uint32_t & i2, uint32_t k, uint32_t i) {
    if (sc_better(k, i, k1, i1)) {
        k2 = k1;
        i2 = i1;
        k1 = k;
        i1 = i;
    } else if (sc_better(k, i, k2, i2)) {
        k2 = k;
        i2 = i;
    }
}

__device__ __forceinline__ void sc_merge(uint32_t & k1, uint32_t & i1, uint32_t & k2, uint32_t & i2, int off) {
    const uint32_t pk1 = __shfl_xor_sync(0xffffffffu, k1, off), pi1 = __shfl_xor_sync(0xffffffffu, i1, off);
    const uint32_t pk2 = __shfl_xor_sync(0xffffffffu, k2, off), pi2 = __shfl_xor_sync(0xffffffffu, i2, off);
    sc_push(k1, i1, k2, i2, pk1, pi1);
    sc_push(k1, i1, k2, i2, pk2, pi2);
}

__global__ void __launch_bounds__(256, 2) prompt_score_tc_kernel(int32_t          rows,
                                                                 int32_t          pos0,
                                                                 int32_t          tiles,
                                                                 const uint4 *    summaries,
                                                                 const uint16_t * q16,
                                                                 const float *    qs,
                                                                 uint32_t *       keys,
                                                                 uint32_t *       best1,
                                                                 uint32_t *       best2,
                                                                 uint16_t *       lanes) {
    __shared__ __align__(16) uint16_t kh[tile_blocks * sc_stride];
    __shared__ float                  bmax[8][tile_blocks];
    __shared__ float                  ks[tile_blocks];
    const int32_t tid = static_cast<int32_t>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int32_t qp = (lane >> 2) & 3;
    const int32_t ra = (lane & 3) | ((lane >> 2) & 4);  // the lane's A row / B column in its quadpair's 8
    const int32_t r_grp = blockIdx.x * sc_rows;
    const int32_t t0 = blockIdx.y * sc_tpc, t1 = min(tiles, t0 + sc_tpc);
    // the A operand: vector m_a = 8 (qp & 1) + ra of the warp's 16 (zero past the slab's rows)
    const int32_t m_a   = 8 * (qp & 1) + ra;
    const int32_t row_a = r_grp + 4 * warp + (m_a >> 2);
    uint4         af[16];
#pragma unroll
    for (int32_t j = 0; j < 16; ++j) {
        af[j] = row_a < rows ? __ldg(reinterpret_cast<const uint4 *>(
                                   q16 + (int64_t(row_a) * n_idx_head + (m_a & 3)) * head_dim + 8 * j)) :
                               make_uint4(0u, 0u, 0u, 0u);
    }
    // the epilogue's rows: vectors 8 (qp & 1) + lr + 2 rh, so heads (lane & 1) + 2 rh of row rsub
    const int32_t lr    = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int32_t rsub  = 2 * (qp & 1) + ((lane >> 4) & 1);
    const int32_t row   = r_grp + 4 * warp + rsub;
    const int32_t cc    = 2 * ((lane >> 1) & 1);
    const int32_t n_vis = (pos0 + row + 1) / ratio;
    float         qsc[2];
#pragma unroll
    for (int32_t rh = 0; rh < 2; ++rh) {
        const int32_t m = 8 * (qp & 1) + lr + 2 * rh;
        qsc[rh]         = row < rows ? qs[(int64_t(r_grp) + 4 * warp) * n_idx_head + m] : 0.0f;
    }
    const bool    row_on = row < rows && n_vis > top_blocks;
    // the staging role: block lb = lane of the tile, chunks 4 warp .. 4 warp + 3 (dims 16 warp .. + 15)
    float4        st[4];
    const auto    fetch = [&](int32_t tile) {
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            const uint4 u = __ldcs(summaries + int64_t(tile) * tile_u4 + (4 * warp + i) * chunk_u4 + lane);
            st[i]         = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z),
                                        __uint_as_float(u.w));
        }
    };
    if (t0 < t1) {
        fetch(t0);
    }
    for (int32_t tile = t0; tile < t1; ++tile) {
        float m = 0.0f;
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            m = fmaxf(m, fmaxf(fmaxf(fabsf(st[i].x), fabsf(st[i].y)), fmaxf(fabsf(st[i].z), fabsf(st[i].w))));
        }
        bmax[warp][lane] = m;
        __syncthreads();  // also: every warp's products of the previous tile are done with kh and ks
        float bm = bmax[0][lane];
#pragma unroll
        for (int32_t w = 1; w < 8; ++w) {
            bm = fmaxf(bm, bmax[w][lane]);
        }
        const int32_t e  = fn::tc::range_exp(bm);
        const float   sc = fn::tc::pow2(e);
        uint4         h[2];
        h[0] = make_uint4(fn::tc::pack2(st[0].x * sc, st[0].y * sc), fn::tc::pack2(st[0].z * sc, st[0].w * sc),
                          fn::tc::pack2(st[1].x * sc, st[1].y * sc), fn::tc::pack2(st[1].z * sc, st[1].w * sc));
        h[1] = make_uint4(fn::tc::pack2(st[2].x * sc, st[2].y * sc), fn::tc::pack2(st[2].z * sc, st[2].w * sc),
                          fn::tc::pack2(st[3].x * sc, st[3].y * sc), fn::tc::pack2(st[3].z * sc, st[3].w * sc));
        *reinterpret_cast<uint4 *>(kh + lane * sc_stride + 16 * warp)     = h[0];
        *reinterpret_cast<uint4 *>(kh + lane * sc_stride + 16 * warp + 8) = h[1];
        if (warp == 0) {
            ks[lane] = fn::tc::pow2(-e);
        }
        __syncthreads();
        if (tile + 1 < t1) {
            fetch(tile + 1);  // in flight under the products
        }
        float acc[2][8];
#pragma unroll
        for (int32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int32_t i = 0; i < 8; ++i) {
                acc[ni][i] = 0.0f;
            }
        }
#pragma unroll
        for (int32_t j = 0; j < 16; ++j) {  // k-steps 2 j, 2 j + 1
#pragma unroll
            for (int32_t ni = 0; ni < 2; ++ni) {
                const uint4 bf = *reinterpret_cast<const uint4 *>(kh + (16 * ni + 8 * (qp >> 1) + ra) * sc_stride + 8 * j);
                fn::tc::detail::mma884(acc[ni], af[j].x, af[j].y, bf.x, bf.y);
                fn::tc::detail::mma884(acc[ni], af[j].z, af[j].w, bf.z, bf.w);
            }
        }
        // the epilogue: columns 16 ni + 8 (qp >> 1) + cc + {0, 1, 4, 5}
        const bool on = row_on && tile * tile_blocks < n_vis;
        uint32_t   k1 = 0u, i1 = 0xffu, k2 = 0u, i2 = 0xffu;
#pragma unroll
        for (int32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int32_t c = 0; c < 4; ++c) {
                const int32_t i   = (c & 1) | ((c & 2) << 1);  // accumulators 0, 1, 4, 5 (rh 0); + 2: rh 1
                const int32_t col = 16 * ni + 8 * (qp >> 1) + cc + i;
                const float   kb  = ks[col];
                const float   r0  = relu((acc[ni][i] * qsc[0]) * kb);
                const float   r2  = relu((acc[ni][i + 2] * qsc[1]) * kb);
                const float   p0  = __shfl_xor_sync(0xffffffffu, r0, 1);
                const float   p2  = __shfl_xor_sync(0xffffffffu, r2, 1);
                const float   s   = (lane & 1) ? ((p0 + r0) + p2) + r2 : ((r0 + p0) + r2) + p2;
                const int32_t b   = tile * tile_blocks + col;
                const uint32_t key = b < n_vis ? select_key(s) : 0u;
                if (on && (lane & 1) == ni) {
                    keys[static_cast<int64_t>(row) * max_blocks + b] = key;
                }
                sc_push(k1, i1, k2, i2, key == 0u ? 0u : key, key == 0u ? 0xffu : static_cast<uint32_t>(col));
            }
        }
        // the row's 32 blocks sit in the 8 lanes that differ in bits 0, 1 and 3 (bit 0: the same values)
        sc_merge(k1, i1, k2, i2, 2);
        sc_merge(k1, i1, k2, i2, 8);
        if (on && (lane & 11) == 0) {
            best1[row * max_tiles + tile] = k1;
            best2[row * max_tiles + tile] = k2;
            lanes[row * max_tiles + tile] = static_cast<uint16_t>((k1 != 0u ? i1 : 0xffu) | ((k2 != 0u ? i2 : 0xffu) << 8));
        }
    }
}

__global__ void set_int_kernel(int32_t * p, int32_t v) {
    *p = v;
}

std::atomic<uint32_t> prompt_select_smem{ 0u };

}  // namespace

size_t prompt_ws_bytes(int32_t /*max_cells*/) {
    const size_t rt = static_cast<size_t>(prompt_slab) * max_tiles;
    return static_cast<size_t>(prompt_slab) * max_blocks * 4 + rt * 4 * 2 + rt * 2 +
           static_cast<size_t>(prompt_slab) * n_idx_head * head_dim * 2 + static_cast<size_t>(prompt_slab) * n_idx_head * 4 +
           256 + static_cast<size_t>(prompt_slab) * n_stats * 4 + 256;
}

prompt_ws prompt_carve(void * base, int32_t /*max_cells*/) {
    uint8_t *    p  = static_cast<uint8_t *>(base);
    const size_t rt = static_cast<size_t>(prompt_slab) * max_tiles;
    prompt_ws    w;
    w.keys = reinterpret_cast<uint32_t *>(p);
    p += static_cast<size_t>(prompt_slab) * max_blocks * 4;
    w.best1 = reinterpret_cast<uint32_t *>(p);
    p += rt * 4;
    w.best2 = reinterpret_cast<uint32_t *>(p);
    p += rt * 4;
    w.lanes = reinterpret_cast<uint16_t *>(p);
    p += rt * 2;
    w.q16 = reinterpret_cast<uint16_t *>(p);
    p += static_cast<size_t>(prompt_slab) * n_idx_head * head_dim * 2;
    w.qs = reinterpret_cast<float *>(p);
    p += static_cast<size_t>(prompt_slab) * n_idx_head * 4;
    w.depth = reinterpret_cast<int32_t *>(p);
    p += 256;
    w.stats = reinterpret_cast<uint32_t *>(p);
    p += static_cast<size_t>(prompt_slab) * n_stats * 4;
    w.error = reinterpret_cast<uint32_t *>(p);
    return w;
}

bool qsp_keys(int32_t rows, int32_t pos0, const float * ik, uint16_t * keys, int32_t max_cells, void * stream) {
    if (rows < 1 || rows > prompt_max_rows || pos0 < 0 || pos0 + rows > max_cells || max_cells > fn::qsa::max_cells ||
        ik == nullptr || keys == nullptr || stream == nullptr) {
        return false;
    }
    prompt_keys_kernel<<<(rows + 3) / 4, 128, 0, static_cast<cudaStream_t>(stream)>>>(rows, pos0, ik, keys);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool qsp_select(const qsp_index_args & a, void * stream) {
    if (a.rows < 1 || a.rows > prompt_max_rows || a.pos0 < 0 || a.pos0 + a.rows > a.max_cells ||
        a.max_cells > max_cells || stream == nullptr || a.summaries == nullptr || a.iq == nullptr ||
        a.q_norm == nullptr || a.body == nullptr || a.offsets == nullptr || a.ws.keys == nullptr ||
        !aligned16(a.summaries) || !aligned16(a.body)) {
        return false;
    }
    cudaStream_t st  = static_cast<cudaStream_t>(stream);
    int          dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        return false;
    }
    if ((prompt_select_smem.load(std::memory_order_acquire) >> dev & 1u) == 0u) {
        if (cudaFuncSetAttribute(select_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(sizeof(sel_smem))) != cudaSuccess) {
            return false;
        }
        prompt_select_smem.fetch_or(1u << dev, std::memory_order_release);
    }
    for (int32_t s = 0; s * prompt_slab < a.rows; ++s) {
        const int32_t r0 = s * prompt_slab, n = std::min(prompt_slab, a.rows - r0), pos = a.pos0 + r0;
        const int32_t tiles = ((pos + n) / ratio + tile_blocks - 1) / tile_blocks;
        prompt_queries_kernel<<<(n * n_idx_head + 7) / 8, 256, 0, st>>>(
            n, pos, a.iq + int64_t(r0) * n_idx_head * head_dim, a.q_norm, a.rope, a.ws.q16, a.ws.qs);
        if ((pos + n) / ratio > top_blocks && tiles > 0) {
            const dim3 grid((n + sc_rows - 1) / sc_rows, (tiles + sc_tpc - 1) / sc_tpc);
            prompt_score_tc_kernel<<<grid, 256, 0, st>>>(n, pos, tiles, reinterpret_cast<const uint4 *>(a.summaries),
                                                         a.ws.q16, a.ws.qs, a.ws.keys, a.ws.best1, a.ws.best2,
                                                         a.ws.lanes);
        }
        set_int_kernel<<<1, 1, 0, st>>>(a.ws.depth + s, pos);
        const select_kargs k = { a.ws.keys,
                                 a.ws.best1,
                                 a.ws.best2,
                                 a.ws.lanes,
                                 a.ws.error,
                                 a.ws.stats,
                                 a.body + int64_t(s) * prompt_slab_words,
                                 a.offsets + s * (prompt_slab + 1),
                                 a.ws.depth + s,
                                 a.max_cells,
                                 n };
        select_kernel<<<n, sel_threads, sizeof(sel_smem), st>>>(k);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace fn::qsa
