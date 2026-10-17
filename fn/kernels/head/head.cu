#include "fn/kernels/head/head.h"
#include "fn/kernels/embed/embed.h"

#include "fn/kernels/dense/dense.h"
#include "fn/xchg/xchg.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

static_assert(fn::embed::err_token != fn::head::err_all_nan, "embedding and head errors must be distinct");
static_assert((fn::embed::err_token & 0xffffu) == 0u && (fn::head::err_all_nan & 0xffffu) == 0u,
              "kernel data errors must not overlap poll bits");

namespace fn::head {
namespace {

constexpr int32_t n_block = 72;   // one per SM
constexpr int32_t n_thread = 448; // 14 warps: at n = 124,160 a block's 432 float4s of a row are one load a lane
constexpr int32_t n_warp = n_thread / 32;
constexpr int32_t work_counter_at = max_rows * 8; // bytes: max_rows U64 bests, then the arrival counter
constexpr size_t work_size = 64;

static_assert(n_block * n_thread >= n_vocab_card / 4, "the model's shard: one float4 a lane a row");
static_assert(2 * n_vocab_card == n_vocab && n_vocab_card % 8 == 0, "the vocabulary splits in two 8-row tiled halves");
static_assert(fn::xchg::words_greedy == 2 && fn::xchg::slot_words(fn::xchg::msg::greedy) >= 2 * max_rows &&
                  fn::xchg::slot_words(fn::xchg::msg::greedy) >= fn::xchg::greedy_hash_at + 4,
              "T-GREEDY: one (value, ID) pair a row, then the decision record's four words");

// The order key: U32 order equals F32 order, with every NaN at 0 (below -inf, whose key is 0x007fffff) and -0 equal
// to +0. Bit manipulation of F32 bit patterns only; no arithmetic.
__device__ __forceinline__ uint32_t order_key(uint32_t u) {
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
        return 0u;
    }
    if ((u & 0x7fffffffu) == 0u) {
        return 0x80000000u;
    }
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

// the canonical F32 bits of a key's value (+0 for either zero; the canonical NaN for key 0)
__device__ __forceinline__ uint32_t key_value(uint32_t key) {
    if (key == 0u) {
        return fn::xchg::canonical_nan;
    }
    return (key & 0x80000000u) != 0u ? (key & 0x7fffffffu) : ~key;
}

// a candidate as one U64: the larger key first, then the lower ID (IDs are below 2^18, with or without an ID map)
__device__ __forceinline__ unsigned long long pack(uint32_t key, uint32_t id) {
    return (static_cast<unsigned long long>(key) << 32) | (0xffffffffu - id);
}

__device__ __forceinline__ unsigned long long max_u64(unsigned long long a, unsigned long long b) {
    return b > a ? b : a;
}

} // namespace

// Block g: float4s g*per .. g*per+per-1 of every row (per = ceil(n / 4 / 72)), each lane one float4 a row a step,
// every row's loads issued before any compare; a column's ID is id0 + column, or the ID map's entry (DR1). Each block's
// best a row goes to the work buffer's U64 by atomicMax (a max over a total order: any arrival order gives the same
// bits); the last block to arrive reads and clears the bests and the counter, and sends each row's pair (T-GREEDY step
// 1) to the peer and to this card's own copy.
template <int B> __global__ void __launch_bounds__(n_thread, 1) greedy_reduce_kernel(const reduce_args a) {
    __shared__ unsigned long long part[B][n_warp];
    __shared__ int32_t is_last;
    auto * best = reinterpret_cast<unsigned long long *>(a.work); // 16-byte aligned, checked by the launcher
    auto * counter = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(a.work) + work_counter_at);

    const int32_t n4 = a.n / 4;
    const int32_t per = (n4 + n_block - 1) / n_block;
    const int32_t f0 = blockIdx.x * per;
    const int32_t f1 = min(f0 + per, n4);
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;

    unsigned long long cand[B];
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
        cand[b] = 0ull;
    }
    for (int32_t f = f0 + threadIdx.x; f < f1; f += n_thread) {
        uint4 v[B];
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            // n % 4 == 0 and the logits are 16-byte aligned (launcher); F32 logits compared as their bits
            v[b] = __ldcg(reinterpret_cast<const uint4 *>(a.logits + static_cast<int64_t>(b) * a.n) + f);
        }
        uint4 id;
        if (a.ids != nullptr) {
            id = __ldg(reinterpret_cast<const uint4 *>(a.ids) + f); // 16-byte aligned (launcher)
        } else {
            const uint32_t i0 = static_cast<uint32_t>(a.id0 + 4 * f);
            id = make_uint4(i0, i0 + 1u, i0 + 2u, i0 + 3u);
        }
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            cand[b] = max_u64(cand[b], pack(order_key(v[b].x), id.x));
            cand[b] = max_u64(cand[b], pack(order_key(v[b].y), id.y));
            cand[b] = max_u64(cand[b], pack(order_key(v[b].z), id.z));
            cand[b] = max_u64(cand[b], pack(order_key(v[b].w), id.w));
        }
    }
#pragma unroll
    for (int32_t b = 0; b < B; ++b) {
#pragma unroll
        for (int32_t o = 16; o > 0; o >>= 1) {
            cand[b] = max_u64(cand[b], __shfl_xor_sync(0xffffffffu, cand[b], o));
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int32_t b = 0; b < B; ++b) {
            part[b][warp] = cand[b];
        }
    }
    __syncthreads();
    if (threadIdx.x < B) {
        const int32_t b = threadIdx.x;
        unsigned long long v = part[b][0];
#pragma unroll
        for (int32_t w = 1; w < n_warp; ++w) {
            v = max_u64(v, part[b][w]);
        }
        atomicMax(best + b, v);
        __threadfence(); // this block's maxima before its arrival
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        is_last = atomicAdd(counter, 1u) == gridDim.x - 1 ? 1 : 0;
    }
    __syncthreads();
    if (is_last == 0) {
        return;
    }
    __threadfence(); // every other block's maxima, which preceded its arrival
    if (threadIdx.x < B) {
        const int32_t b = threadIdx.x;
        const unsigned long long v = atomicExch(best + b, 0ull); // read and clear for the next launch
        const uint32_t key = static_cast<uint32_t>(v >> 32);
        const uint2 pair = make_uint2(key_value(key), 0xffffffffu - static_cast<uint32_t>(v));
        reinterpret_cast<uint2 *>(a.pair_own)[b] = pair;
        reinterpret_cast<uint2 *>(a.pair_peer)[b] = pair; // T-GREEDY: one 8-byte store to peer memory
    }
    if (threadIdx.x == 0) {
        *counter = 0u;
        if (a.hash_words != nullptr) {
            // SM1: the decision record's words after the rows (one 16-byte store)
            *reinterpret_cast<uint4 *>(a.pair_peer + fn::xchg::greedy_hash_at) =
                *reinterpret_cast<const uint4 *>(a.hash_words);
        }
    }
}

// Lane b: polls the peer's pair of row b (T-GREEDY step 3), resets it (step 4), and takes the larger of the two cards'
// values, the lower ID on an exact tie (card 0's in the vocabulary shard; either card's with the draft's ID map).
__global__ void __launch_bounds__(32) greedy_decide_kernel(const decide_args a, int32_t rows) {
    const int32_t b = threadIdx.x;
    if (b == 31 && a.hash_words != nullptr) {
        // SM1: the peer's record words, polled and reset by this lane, against this card's
        const uint4 * const p[1] = {reinterpret_cast<const uint4 *>(a.ring + fn::xchg::greedy_hash_at)};
        uint4 v[1];
        fn::xchg::poll<1>(p, v, a.err, fn::xchg::err_poll_greedy);
        fn::xchg::reset4(reinterpret_cast<uint4 *>(a.ring + fn::xchg::greedy_hash_at));
        const uint4 own = *reinterpret_cast<const uint4 *>(a.hash_words);
        if (v[0].x != own.x || v[0].y != own.y || v[0].z != own.z || v[0].w != own.w) {
            atomicOr(a.err, fn::xchg::err_hash);
        } else if (a.checked != nullptr &&
                   !(own.x == 0x7fffffffu && own.y == 0x7fffffffu && own.z == 0x7fffffffu && own.w == 0x7fffffffu)) {
            *a.checked = 1u;
        }
    }
    if (b >= rows) {
        return;
    }
    uint2 * ring = reinterpret_cast<uint2 *>(a.ring) + b; // the slot's words are U32 payload bits
    const uint2 peer = fn::xchg::poll2(ring, a.err, fn::xchg::err_poll_greedy);
    fn::xchg::reset2(ring);
    const uint2 own = reinterpret_cast<const uint2 *>(a.pair_own)[b];
    const uint2 p0 = a.card == 0 ? own : peer;
    const uint2 p1 = a.card == 0 ? peer : own;
    const uint32_t k0 = order_key(p0.x);
    const uint32_t k1 = order_key(p1.x);
    if (k0 == 0u && k1 == 0u) {
        a.token[b] = no_token;
        atomicOr(a.err, err_all_nan);
    } else {
        a.token[b] = static_cast<int32_t>(k1 > k0 || (k1 == k0 && p1.y < p0.y) ? p1.y : p0.y);
    }
    if (a.pair_seen != nullptr) {
        reinterpret_cast<uint2 *>(a.pair_seen)[b] = peer;
    }
}

namespace {

bool aligned(const void * p, uintptr_t n) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % n == 0;
}

fn::dense::product_args vocab_args(const logits_args & a) {
    fn::dense::product_args p;
    p.k = k_in;
    p.n_mats = 1;
    p.mats[0].f = fn::dense::fmt::q5_k;
    p.mats[0].m = a.n;
    p.mats[0].w = a.w;
    p.mats[0].out = a.logits;
    p.hmix = a.hmix;
    p.hmix_peer_half = 1 - a.card;
    p.err = a.err;
    fn::dense::choose_splits(p);
    return p;
}

template <int B> bool launch_reduce(const reduce_args & a, cudaStream_t st) {
    greedy_reduce_kernel<B><<<n_block, n_thread, 0, st>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

size_t logits_work_bytes(int32_t n) {
    if (n < 8 || n % 8 != 0 || n > n_vocab_card) {
        return 0;
    }
    logits_args a = {};
    a.n = n;
    return fn::dense::work_bytes(vocab_args(a));
}

size_t greedy_work_bytes() {
    return work_size;
}

bool draft_ids(const int32_t * subset, int32_t n, int32_t card, int32_t * ids) {
    if (subset == nullptr || ids == nullptr || (card != 0 && card != 1) || n < 16 || n % 2 != 0 || n / 2 % 8 != 0 ||
        n / 2 > n_vocab_card) {
        return false;
    }
    std::vector<int32_t> s(subset, subset + n);
    std::sort(s.begin(), s.end());
    if (s.front() < 0 || s.back() >= n_vocab || std::adjacent_find(s.begin(), s.end()) != s.end()) {
        return false;
    }
    for (int32_t i = 0; i < n / 2; ++i) {
        ids[i] = s[static_cast<size_t>(2 * i + card)];
    }
    return true;
}

int64_t draft_head_bytes(int32_t m) {
    if (m < 0) {
        return 0;
    }
    return fn::dense::row_bytes(fn::dense::fmt::q5_k, k_in) * m;
}

// one 8-row tile at a time: the tile's rows gathered contiguously, then dense::pack's layout of those 8 rows (pack
// lays out each 8-row tile on its own, so packing tile by tile equals packing the gathered m rows at once)
bool pack_draft_head(const uint8_t * vocab, size_t row_bytes, const int32_t * ids, int32_t m, uint8_t * dst) {
    const size_t rb = static_cast<size_t>(fn::dense::row_bytes(fn::dense::fmt::q5_k, k_in));
    if (vocab == nullptr || ids == nullptr || dst == nullptr || m < 8 || m % 8 != 0 || m > n_vocab_card ||
        row_bytes < rb) {
        return false;
    }
    std::vector<uint8_t> tile(8 * rb);
    for (int32_t t = 0; t < m / 8; ++t) {
        for (int32_t i = 0; i < 8; ++i) {
            const int32_t id = ids[8 * t + i];
            if (id < 0 || id >= n_vocab) {
                return false;
            }
            std::copy_n(vocab + static_cast<size_t>(id) * row_bytes, rb, tile.data() + static_cast<size_t>(i) * rb);
        }
        if (!fn::dense::pack(fn::dense::fmt::q5_k, 8, k_in, tile.data(), rb, dst + static_cast<size_t>(t) * 8 * rb)) {
            return false;
        }
    }
    return true;
}

bool logits(int32_t rows, const logits_args & a, void * stream) {
    if (stream == nullptr || rows < 1 || rows > max_rows || a.n < 8 || a.n % 8 != 0 || a.n > n_vocab_card ||
        !aligned(a.hmix, 16) || a.err == nullptr || (a.card != 0 && a.card != 1) || !aligned(a.w, 16) ||
        !aligned(a.logits, 16)) {
        return false;
    }
    fn::dense::product_args p = vocab_args(a);
    if (fn::dense::work_bytes(p) > 0 && !aligned(a.work, 16)) {
        return false;
    }
    p.work = a.work;
    return fn::dense::product(rows, p, stream);
}

bool greedy_reduce(int32_t rows, const reduce_args & a, void * stream) {
    const bool ids_ok = a.ids != nullptr ? aligned(a.ids, 16) : a.id0 >= 0 && a.id0 <= n_vocab - a.n;
    if (stream == nullptr || a.n < 4 || a.n % 4 != 0 || a.n > n_vocab_card || !ids_ok || !aligned(a.logits, 16) ||
        !aligned(a.pair_own, 8) || !aligned(a.pair_peer, 16) || !aligned(a.work, 16) ||
        (a.hash_words != nullptr && !aligned(a.hash_words, 16))) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1:
            return launch_reduce<1>(a, st);
        case 2:
            return launch_reduce<2>(a, st);
        case 3:
            return launch_reduce<3>(a, st);
        case 4:
            return launch_reduce<4>(a, st);
        case 5:
            return launch_reduce<5>(a, st); // SP2
        default:
            return false;
    }
}

bool greedy_decide(int32_t rows, const decide_args & a, void * stream) {
    if (stream == nullptr || rows < 1 || rows > max_rows || !aligned(a.pair_own, 8) || !aligned(a.ring, 16) ||
        !aligned(a.token, 4) || a.err == nullptr || (a.pair_seen != nullptr && !aligned(a.pair_seen, 8)) ||
        (a.card != 0 && a.card != 1) || (a.hash_words != nullptr && !aligned(a.hash_words, 16))) {
        return false;
    }
    greedy_decide_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(a, rows);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::head
