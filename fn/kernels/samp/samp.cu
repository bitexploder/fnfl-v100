#include "fn/kernels/samp/samp.h"

#include "fn/kernels/samp/philox.h"
#include "fn/xchg/xchg.cuh"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Message layouts (words; every word its own flag, xchg.h's A3 protocol; F32 words canonical, 64-bit values as (low,
// high) with a low word of 0xffffffff sent as 0xfffffffe, so no payload word is the sentinel):
//   samp_a  header (4 words: the decision record's round and hash, 31 bits each), then row r at 4 + r * 8,200:
//             top-k:   [0] n, [1..3] 0, then n pairs (logit, ID), sorted by (logit, lower ID)
//             max:     [0] flags (1: the card has a finite logit), [1] M_r, [2..3] 0
//             hist:    [0] flags, [1] M_r, [2] W, [3] 0, then W + 1 bins of (count, sum low, sum high, 0)
//   samp_b  row r at r * 248,320: the cut bin's members, (logit, ID) pairs in arrival order (resolve sorts by key)
//   samp_c  header, then row r at 4 + r * 8: [0] flags (1: mass, 2: owns the row's draft), [1] Mt_r, [2..3] Z_r
//           (double), [4] the local candidate, [5..6] the draft's numerator (double), [7] q at the draft (F32)
//   samp_d  [0] flags (1: mass), [1] the local residual candidate, [2..3] R_r (U64), [4..7] 0
//   samp_x  the record's four words

namespace fn::samp {
namespace {

constexpr uint32_t sentinel = fn::xchg::sentinel;
constexpr uint32_t no_cand = 0x7fffffffu; // a card without a candidate (no kept mass): never an ID, never the sentinel
constexpr int32_t n_warp = n_thread / 32;
constexpr int32_t hdr = fn::xchg::samp_hdr_words;
constexpr int32_t a_row = fn::xchg::samp_a_row_words;
constexpr int32_t b_row = fn::xchg::samp_b_row_words;
constexpr int32_t c_row = fn::xchg::samp_c_row_words;
static_assert(a_row >= 4 + 4 * (hist_bins + 1) && a_row >= 4 + 2 * max_k, "samp_a holds a histogram or a list a row");
static_assert(b_row == 2 * n_card, "samp_b holds a whole shard's members a row");
static_assert(max_chunk >= (n_card + n_chunk * 4 - 1) / (n_chunk * 4) * 4, "a chunk fits the shared key array");

constexpr double two50 = 1125899906842624.0;    // 2^50: chunk, list and member weights
constexpr float two46f = 70368744177664.0f;     // 2^46: a card's histogram
constexpr double two60 = 1152921504606846976.0; // 2^60: residual probabilities
constexpr float two50f = 1125899906842624.0f;

// ---- workspace ----
struct part_t {
    float mt; // the chunk's tempered maximum over its kept part (-inf: none)
    uint32_t pad;
    uint64_t s; // its integer mass (settle: 2^50 relative to mt; resid: 2^60 absolute)
};
struct plan_t {
    uint64_t kcut;       // top-p: the cut member's composite key (0: the whole cut bin kept)
    uint64_t target;     // top-p: ceil(top_p * Z) at the global scale
    uint64_t cum_before; // the mass before the cut bin
    double e[2];         // exp(M_r - M), card r's histogram scale to the global one
    double z;            // decide: the kept mass at the global tempered maximum (verify: resid's normalizer)
    double e_own;        // decide: exp(Mt_own - Mt)
    float m_card[2];     // each card's maximum M_r (as exchanged)
    float m;             // the global maximum
    float thr;           // min-p's threshold
    float mt_own;        // settle: this card's tempered maximum over the kept set
    int32_t c;           // the cut bin (W: the tail)
    int32_t tg;          // floor(64 M)
    int32_t w;           // the window's bins
    uint32_t flags;      // pf_*
    int32_t n_mem[2];    // the cut bin's members on each card
    int32_t pad[2];
};
constexpr uint32_t pf_thr = 1u << 0, pf_topp = 1u << 1, pf_members = 1u << 2, pf_empty = 1u << 3;
struct rowres_t {
    int32_t accept;
    int32_t draw;
    int32_t resid;
    int32_t pad;
};
struct vstate_t {
    int32_t need; // decide2 runs: a verify row was rejected on the full path
    int32_t row;  // the first rejection a
    int32_t pad[2];
};
struct cardres_t {
    uint32_t mkey; // the card's maximum composite key's order key
    uint32_t has;  // a finite logit
    uint32_t pad[2];
};

constexpr int32_t ctr_select = 0, ctr_max = 1, ctr_hist = 2, ctr_members = 3, ctr_settle = 4, ctr_resid = 5,
                  ctr_mempos = 6, ctr_verdict = 7, n_ctr = 16;

constexpr int64_t align256(int64_t n) {
    return (n + 255) / 256 * 256;
}

struct ws_off {
    int64_t cand, cand_n, ctr, maxkey, ghc, ghs, own_a, plan, own_b, mem_peer, part, own_c, own_d, rowres, vstate,
        cardres, total;
};
constexpr ws_off ws_layout() {
    ws_off o = {};
    int64_t at = 0;
    o.cand = at;
    at += align256(int64_t(max_rows) * n_chunk * max_k * 8);
    o.cand_n = at;
    at += align256(int64_t(max_rows) * n_chunk * 4);
    o.ctr = at;
    at += align256(int64_t(max_rows) * n_ctr * 4);
    o.maxkey = at;
    at += align256(int64_t(max_rows) * 8);
    o.ghc = at;
    at += align256(int64_t(max_rows) * (hist_bins + 1) * 4);
    o.ghs = at;
    at += align256(int64_t(max_rows) * (hist_bins + 1) * 8);
    o.own_a = at;
    at += align256(int64_t(fn::xchg::slot_words(fn::xchg::msg::samp_a)) * 4);
    o.plan = at;
    at += align256(int64_t(max_rows) * int64_t(sizeof(plan_t)));
    o.own_b = at;
    at += align256(int64_t(max_rows) * b_row * 4);
    o.mem_peer = at;
    at += align256(int64_t(max_rows) * b_row * 4);
    o.part = at;
    at += align256(int64_t(max_rows) * n_chunk * int64_t(sizeof(part_t)));
    o.own_c = at;
    at += align256(int64_t(fn::xchg::slot_words(fn::xchg::msg::samp_c)) * 4);
    o.own_d = at;
    at += align256(int64_t(fn::xchg::samp_d_words) * 4);
    o.rowres = at;
    at += align256(int64_t(max_rows) * int64_t(sizeof(rowres_t)));
    o.vstate = at;
    at += align256(int64_t(sizeof(vstate_t)));
    o.cardres = at;
    at += align256(int64_t(max_rows) * int64_t(sizeof(cardres_t)));
    o.total = at;
    return o;
}
constexpr ws_off wso = ws_layout();

// the kernels' arguments: the card's buffers, the round's rows and the mode
struct kargs {
    card_view v;
    const float * logits;
    int32_t n;
    int32_t id0;
    const int32_t * ids;
    fn::round::desc * desc; // decisions; null for the draft's pick
    int32_t rows;
    int32_t draft; // the draft's pick (pick_io)
    uint32_t pick_pos;
    const fn::round::desc * pick_desc; // SP1: the pick's live position, pick_desc->n_before + pick_off (null: pick_pos)
    int32_t pick_off;
    int32_t pick_t;
    const int32_t * dtok;
    int32_t n_dtok;
    int32_t * proposal;
    int32_t header; // this launch sends the decision record's words (the decision's first exchange)
};

template <typename T> __device__ __forceinline__ T * ws(const kargs & a, int64_t off) {
    return reinterpret_cast<T *>(a.v.work + off);
}

// ---- keys, words and numbers ----

// the order key (as head.cu): U32 order equals F32 order, every NaN at 0, -0 equal to +0
__host__ __device__ __forceinline__ uint32_t order_key(uint32_t u) {
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) {
        return 0u;
    }
    if ((u & 0x7fffffffu) == 0u) {
        return 0x80000000u;
    }
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

// the canonical F32 bits of a key's value
__host__ __device__ __forceinline__ uint32_t key_bits(uint32_t key) {
    if (key == 0u) {
        return fn::xchg::canonical_nan;
    }
    return (key & 0x80000000u) != 0u ? (key & 0x7fffffffu) : ~key;
}

__device__ __forceinline__ uint64_t composite(uint32_t key, int32_t id) {
    return (static_cast<uint64_t>(key) << 32) | (0xffffffffu - static_cast<uint32_t>(id));
}
__device__ __forceinline__ int32_t comp_id(uint64_t k) {
    return static_cast<int32_t>(0xffffffffu - static_cast<uint32_t>(k));
}
__device__ __forceinline__ float comp_logit(uint64_t k) {
    return __uint_as_float(key_bits(static_cast<uint32_t>(k >> 32)));
}

// a 64-bit payload as two words, the low one never the sentinel (the value moves by one unit when it would be)
__device__ __forceinline__ uint64_t canon64(uint64_t v) {
    return (v & 0xffffffffull) == 0xffffffffull ? v - 1 : v;
}
__device__ __forceinline__ double canon_d(double d) {
    return __longlong_as_double(static_cast<long long>(canon64(static_cast<uint64_t>(__double_as_longlong(d)))));
}
__device__ __forceinline__ uint64_t join64(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
__device__ __forceinline__ double join_d(uint32_t lo, uint32_t hi) {
    return __longlong_as_double(static_cast<long long>(join64(lo, hi)));
}
__device__ __forceinline__ uint32_t lo32(uint64_t v) {
    return static_cast<uint32_t>(v);
}
__device__ __forceinline__ uint32_t hi32(uint64_t v) {
    return static_cast<uint32_t>(v >> 32);
}
__device__ __forceinline__ uint32_t dlo(double d) {
    return lo32(static_cast<uint64_t>(__double_as_longlong(d)));
}
__device__ __forceinline__ uint32_t dhi(double d) {
    return hi32(static_cast<uint64_t>(__double_as_longlong(d)));
}

// a weight as an integer: round(w * 2^s), 0 for anything not positive (NaN included)
__device__ __forceinline__ uint64_t fixed_w(float w, float scale) {
    return w > 0.0f ? __float2ull_rn(__fmul_rn(w, scale)) : 0ull;
}

__device__ __forceinline__ uint4 poll4(const uint32_t * p, uint32_t * err) {
    const uint4 * const q[1] = {reinterpret_cast<const uint4 *>(p)};
    uint4 v[1];
    fn::xchg::poll<1>(q, v, err, fn::xchg::err_poll_samp);
    return v[0];
}
__device__ __forceinline__ uint2 poll2w(const uint32_t * p, uint32_t * err) {
    return fn::xchg::poll2(reinterpret_cast<const uint2 *>(p), err, fn::xchg::err_poll_samp);
}

// ---- block utilities (256 threads; every thread calls them) ----

template <typename T> __device__ __forceinline__ T warp_incl_scan(T v) {
    const int32_t lane = threadIdx.x % 32;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1) {
        const T n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) {
            v += n;
        }
    }
    return v;
}

// inclusive scan of one value a thread, in thread order; sh: n_warp T of shared memory
template <typename T> __device__ T block_incl_scan(T v, T * sh) {
    const int32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_incl_scan(v);
    if (lane == 31) {
        sh[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        T w = lane < n_warp ? sh[lane] : T(0);
        w = warp_incl_scan(w);
        if (lane < n_warp) {
            sh[lane] = w;
        }
    }
    __syncthreads();
    if (warp > 0) {
        v += sh[warp - 1];
    }
    __syncthreads();
    return v;
}

template <typename T> __device__ T block_sum(T v, T * sh) {
    const T s = block_incl_scan(v, sh);
    __shared__ T last;
    if (threadIdx.x == n_thread - 1) {
        last = s;
    }
    __syncthreads();
    const T r = last;
    __syncthreads();
    return r;
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t v) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        const uint64_t n = __shfl_xor_sync(0xffffffffu, v, o);
        v = n > v ? n : v;
    }
    return v;
}

// The k-th largest of n U32 keys (1 <= k <= n) by a radix select over 11, 11 and 10 bits: thr, and above, the number of
// keys greater than thr. get(i) returns key i. hist: 2,048 U32 of shared memory; sh: 2 x 256 U32 of shared memory.
template <typename Get>
__device__ void radix_kth(const Get & get, int32_t n, int32_t k, uint32_t * hist, uint32_t * sh, uint32_t & thr,
                          int32_t & above) {
    __shared__ uint32_t s_digit, s_above;
    uint32_t prefix = 0, pmask = 0;
    int32_t krem = k, abv = 0;
    const int32_t tid = threadIdx.x;
    for (int32_t p = 0; p < 3; ++p) {
        const int32_t shift = p == 0 ? 21 : p == 1 ? 10 : 0;
        const int32_t nb = p == 2 ? 1024 : 2048;
        const uint32_t dmask = static_cast<uint32_t>(nb - 1);
        for (int32_t i = tid; i < nb; i += n_thread) {
            hist[i] = 0;
        }
        __syncthreads();
        for (int32_t i = tid; i < n; i += n_thread) {
            const uint32_t key = get(i);
            if ((key & pmask) == prefix) {
                atomicAdd(&hist[(key >> shift) & dmask], 1u);
            }
        }
        __syncthreads();
        const int32_t per = nb / n_thread;
        uint32_t local = 0;
        for (int32_t b = 0; b < per; ++b) {
            local += hist[tid * per + b];
        }
        sh[tid] = local;
        __syncthreads();
        // the keys above thread t's bins: a scan over the threads in reverse order
        const uint32_t rv = sh[n_thread - 1 - tid];
        __syncthreads();
        const uint32_t incl = block_incl_scan(rv, sh + n_thread);
        sh[n_thread - 1 - tid] = incl; // the inclusive suffix of thread n_thread - 1 - tid
        __syncthreads();
        uint32_t acc = sh[tid] - local; // keys in bins above thread tid's
        for (int32_t b = per - 1; b >= 0; --b) {
            const uint32_t h = hist[tid * per + b];
            if (acc < static_cast<uint32_t>(krem) && acc + h >= static_cast<uint32_t>(krem)) {
                s_digit = static_cast<uint32_t>(tid * per + b);
                s_above = acc;
            }
            acc += h;
        }
        __syncthreads();
        const uint32_t d = s_digit;
        const uint32_t a = s_above;
        __syncthreads();
        abv += static_cast<int32_t>(a);
        krem -= static_cast<int32_t>(a);
        prefix |= d << shift;
        pmask |= dmask << shift;
    }
    thr = prefix;
    above = abv;
}

// ---- the transform: bias and penalties (F32, rounded per operation) ----

struct xform {
    uint32_t flags;
    float rep, freq, present;
    int32_t nd;
    int32_t did[2 * max_drafts];
    int32_t dv[2 * max_drafts];
};

__device__ __forceinline__ uint32_t row_position(const kargs & a, int32_t t) {
    if (a.desc == nullptr) {
        return a.pick_desc != nullptr ? static_cast<uint32_t>(a.pick_desc->n_before + a.pick_off) : a.pick_pos;
    }
    const int32_t nb = a.desc->n_before;
    return static_cast<uint32_t>(a.desc->kind == static_cast<int32_t>(fn::round::kind::bootstrap) ? nb : nb + t + 1);
}
// the slot of the sampler's messages a decision uses: its round's parity (a draft pick: its position's). Two slots
// make back-to-back decisions safe without other exchanges between them (xchg.h's chain for samp_a .. samp_d)
__device__ __forceinline__ int32_t par(const kargs & a) {
    return a.desc != nullptr ? static_cast<int32_t>(a.desc->round & 1u) : static_cast<int32_t>(row_position(a, 0) & 1u);
}

// forced rounds decide nothing (FC-RD3): every decision kernel returns at once, on both cards alike
__device__ __forceinline__ bool skip(const kargs & a) {
    return a.desc != nullptr && (a.desc->kind == static_cast<int32_t>(fn::round::kind::forced) ||
                                 a.desc->kind == static_cast<int32_t>(fn::round::kind::forced_k));
}

__device__ __forceinline__ bool is_verify(const kargs & a) {
    return a.desc != nullptr && a.desc->kind == static_cast<int32_t>(fn::round::kind::verify);
}

// thread 0: the row's deltas (the drafts before it as accepted, and the tokens they evict from the window)
__device__ void make_xform(const kargs & a, int32_t t, xform & x) {
    const params & P = *a.v.prm;
    x.flags = P.flags;
    x.rep = P.rep;
    x.freq = P.freq;
    x.present = P.present;
    x.nd = 0;
    if ((P.flags & f_pen) == 0u) {
        return;
    }
    const int32_t * toks = a.desc != nullptr ? a.desc->tok + 1 : a.dtok;
    const int32_t m = a.desc != nullptr ? (is_verify(a) ? t : 0) : a.n_dtok;
    const int64_t L = a.v.rec->window;
    const int32_t cap = P.cap;
    for (int32_t i = 0; i < m && i < max_drafts; ++i) {
        x.did[x.nd] = toks[i];
        x.dv[x.nd++] = 1;
        const int64_t pos = L + i;
        if (pos >= cap) {
            const int64_t ev = pos - cap;
            x.did[x.nd] = ev >= L ? toks[ev - L] : a.v.ring[ev % cap];
            x.dv[x.nd++] = -1;
        }
    }
}

__device__ __forceinline__ float transform(float l, int32_t id, const xform & x, const card_view & v) {
    if ((x.flags & f_bias) != 0u) {
        l = __fadd_rn(l, v.bias[id]);
    }
    if ((x.flags & f_pen) != 0u) {
        int32_t c = static_cast<int32_t>(v.counts[id]);
        for (int32_t i = 0; i < x.nd; ++i) {
            c += x.did[i] == id ? x.dv[i] : 0;
        }
        if (c > 0) {
            l = l <= 0.0f ? __fmul_rn(l, x.rep) : __fdiv_rn(l, x.rep);
            l = __fsub_rn(l, __fadd_rn(__fmul_rn(static_cast<float>(c), x.freq), x.present));
        }
    }
    return l;
}

__device__ __forceinline__ int32_t id_of(const kargs & a, int32_t j) {
    return a.ids != nullptr ? a.ids[j] : a.id0 + j;
}

// chunk c of a row: columns [j0, j0 + m)
__device__ __forceinline__ void chunk_bounds(int32_t n, int32_t c, int32_t & j0, int32_t & m) {
    const int32_t len = (n + n_chunk * 4 - 1) / (n_chunk * 4) * 4;
    j0 = c * len;
    m = max(0, min(len, n - j0));
}

// a row's chunk of transformed logits into shared memory (the columns are 16-byte aligned: n % 4 == 0)
__device__ void load_chunk(const kargs & a, int32_t t, int32_t j0, int32_t m, const xform & x, float * out) {
    const float * row = a.logits + static_cast<int64_t>(t) * a.n + j0;
    for (int32_t i = threadIdx.x; i < m / 4; i += n_thread) {
        const float4 q = __ldcg(reinterpret_cast<const float4 *>(row) + i);
        const int32_t j = 4 * i;
        out[j] = transform(q.x, id_of(a, j0 + j), x, a.v);
        out[j + 1] = transform(q.y, id_of(a, j0 + j + 1), x, a.v);
        out[j + 2] = transform(q.z, id_of(a, j0 + j + 2), x, a.v);
        out[j + 3] = transform(q.w, id_of(a, j0 + j + 3), x, a.v);
    }
}

// the last block of a row's grid: true for exactly one block, after every block's writes (fenced before arrival)
__device__ bool last_of_row(uint32_t * ctr) {
    __shared__ int32_t last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(ctr, 1u) == gridDim.x - 1 ? 1 : 0;
    }
    __syncthreads();
    if (last != 0) {
        __threadfence();
        if (threadIdx.x == 0) {
            *ctr = 0u;
        }
    }
    return last != 0;
}

// ---- the decision record and its hash ----

__host__ __device__ inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}
__host__ __device__ inline uint64_t hash_of(uint64_t round, int32_t n, const int32_t * tok, int32_t a) {
    uint64_t h = mix64(round ^ 0x5341'4d50'4c45'5231ull);
    h = mix64(
        h ^ (static_cast<uint64_t>(static_cast<uint32_t>(n)) | static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32));
    for (int32_t i = 0; i < n; ++i) {
        h = mix64(h ^ static_cast<uint32_t>(tok[i]));
    }
    h &= 0x3fffffffffffffffull;
    return h == 0x3fffffffffffffffull ? h ^ 1ull : h; // never the invalid record's words
}

// the record's four words as sent: round and hash, 31 bits each; an invalid record sends all 0x7fffffff
__host__ __device__ inline void set_words(record & r) {
    if (r.valid == 0u) {
        r.words[0] = r.words[1] = r.words[2] = r.words[3] = 0x7fffffffu;
        return;
    }
    r.words[0] = static_cast<uint32_t>(r.round) & 0x7fffffffu;
    r.words[1] = static_cast<uint32_t>(r.round >> 31) & 0x7fffffffu;
    r.words[2] = static_cast<uint32_t>(r.hash) & 0x7fffffffu;
    r.words[3] = static_cast<uint32_t>(r.hash >> 31) & 0x7fffffffu;
}

__device__ __forceinline__ uint4 record_words(const record & r) {
    return *reinterpret_cast<const uint4 *>(r.words);
}

// the sender's header: the record's words (one 16-byte store to the peer's slot)
__device__ __forceinline__ void send_header(const card_view & v, float * peer_slot) {
    *reinterpret_cast<uint4 *>(peer_slot) = record_words(*v.rec);
}

// the first reader of a decision: polls and resets the peer's header, compares it with this card's record (the record
// of the last decision, not yet overwritten), sets err_hash on a mismatch and marks the record checked
__device__ void check_header(const card_view & v, float * own_slot) {
    uint32_t * p = reinterpret_cast<uint32_t *>(own_slot);
    const uint4 peer = poll4(p, v.err);
    fn::xchg::reset4(reinterpret_cast<uint4 *>(p));
    const uint4 own = record_words(*v.rec);
    if (peer.x != own.x || peer.y != own.y || peer.z != own.z || peer.w != own.w) {
        atomicOr(v.err, fn::xchg::err_hash);
    } else if (v.rec->valid != 0u) {
        v.rec->checked = 1u;
    }
}

// the verdict's record (both cards: the same tokens)
__device__ void write_record(const kargs & a, int32_t n, const int32_t * tok, int32_t acc) {
    record & r = *a.v.rec;
    r.round = a.desc->round;
    r.n = n;
    for (int32_t i = 0; i < max_rows; ++i) {
        r.tok[i] = i < n ? tok[i] : -1;
    }
    r.a = acc;
    r.hash = hash_of(r.round, n, tok, acc);
    r.valid = 1u;
    r.checked = 0u;
    r.applied = 0u;
    set_words(r);
}

// the verdict of a decision from each row's results: generate and bootstrap emit row 0's draw; verify emits the
// leading accepted drafts, then the residual draw at the first rejection or the bonus draw (FC-DEC3)
__device__ void verdict(const kargs & a, const rowres_t * rr, bool resid_later) {
    fn::round::desc & d = *a.desc;
    if (!is_verify(a)) {
        d.argmax[0] = rr[0].draw;
        write_record(a, 1, &rr[0].draw, 0);
        return;
    }
    const int32_t nd = a.rows - 1;
    int32_t acc = nd;
    for (int32_t t = 0; t < nd; ++t) {
        if (rr[t].accept == 0) {
            acc = t;
            break;
        }
    }
    int32_t out[max_rows];
    for (int32_t t = 0; t < acc; ++t) {
        out[t] = d.tok[t + 1];
    }
    for (int32_t t = 0; t < a.rows; ++t) {
        d.argmax[t] = rr[t].draw;
    }
    if (acc < nd && resid_later) {
        vstate_t & vs = *ws<vstate_t>(a, wso.vstate);
        vs.need = 1;
        vs.row = acc;
        d.a = acc;
        return;
    }
    out[acc] = acc < nd ? rr[acc].resid : rr[nd].draw;
    // FC-DEC3's end and budget (SP1): an EOS at out[j], or a budget of only j + 1 tokens, for some j < a lowers a to
    // j, so the record, the penalty tables and the commit (a + 1) follow the cut list
    const int32_t eos = a.v.prm->eos;
    for (int32_t j = 0; j < acc; ++j) {
        if ((eos >= 0 && out[j] == eos) || j + 1 >= d.budget) {
            acc = j;
            break;
        }
    }
    d.a = acc;
    d.n_out = acc + 1;
    for (int32_t t = 0; t < max_rows; ++t) {
        d.out[t] = t <= acc ? out[t] : -1;
    }
    write_record(a, acc + 1, out, acc);
}

// each row's block publishes its result; the last block to arrive reads them all (through L2, after the fences) and
// writes the verdict. One block a row.
__device__ void publish_and_verdict(const kargs & a, const rowres_t & res, bool resid_later) {
    rowres_t * rr = ws<rowres_t>(a, wso.rowres);
    uint32_t * ctr = ws<uint32_t>(a, wso.ctr) + ctr_verdict;
    __shared__ int32_t last;
    if (threadIdx.x == 0) {
        rr[blockIdx.x] = res;
        __threadfence();
        last = atomicAdd(ctr, 1u) == gridDim.x - 1 ? 1 : 0;
    }
    __syncthreads();
    if (last != 0 && threadIdx.x == 0) {
        *ctr = 0u;
        __threadfence();
        rowres_t v[max_rows] = {};
        for (int32_t q = 0; q < a.rows; ++q) {
            v[q].accept = __ldcg(&rr[q].accept);
            v[q].draw = __ldcg(&rr[q].draw);
            v[q].resid = __ldcg(&rr[q].resid);
        }
        verdict(a, v, resid_later);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// apply: the record's tokens join the penalty tables (one thread)
__global__ void apply_kernel(const kargs a) {
    if (threadIdx.x != 0 || (a.v.prm->flags & f_pen) == 0u) {
        return;
    }
    record & r = *a.v.rec;
    if (r.valid == 0u || r.applied != 0u) {
        return;
    }
    const int32_t cap = a.v.prm->cap;
    for (int32_t i = 0; i < r.n; ++i) {
        const int64_t pos = r.window;
        int32_t * slot = a.v.ring + pos % cap;
        if (pos >= cap) {
            a.v.counts[*slot] -= 1u;
        }
        *slot = r.tok[i];
        a.v.counts[r.tok[i]] += 1u;
        r.window = pos + 1;
    }
    r.applied = 1u;
}

// ---------------------------------------------------------------------------------------------------------------
// top-k path, select: chunk c of row t keeps its top k (ascending ID order); the row's last block selects the card's
// top k of the chunks' lists, sorts them by (logit, lower ID) and sends them (samp_a), with its own copy.
struct select_smem {
    float l[max_chunk];
    uint32_t hist[2048];
    uint32_t sh[2 * n_thread];
    uint64_t sel[max_k];
    int32_t off[n_chunk + 1];
    xform x;
    int32_t gt, eq_take;
    uint32_t thr;
};

__global__ void __launch_bounds__(n_thread) select_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ select_smem s;
    const int32_t c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const int32_t k = a.v.prm->top_k;
    int32_t j0 = 0, m = 0;
    chunk_bounds(a.n, c, j0, m);
    if (tid == 0) {
        make_xform(a, t, s.x);
    }
    __syncthreads();
    load_chunk(a, t, j0, m, s.x, s.l);
    __syncthreads();
    const int32_t kc = min(k, m);
    uint64_t * cand = ws<uint64_t>(a, wso.cand) + (static_cast<int64_t>(t) * n_chunk + c) * max_k;
    // the per-thread segment of the chunk, in column order (ordered compaction)
    const int32_t per = (m + n_thread - 1) / n_thread;
    const int32_t s0 = min(m, tid * per), s1 = min(m, s0 + per);
    if (kc > 0) {
        uint32_t thr = 0;
        int32_t above = 0;
        const float * lv = s.l;
        radix_kth([lv](int32_t i) { return order_key(__float_as_uint(lv[i])); }, m, kc, s.hist, s.sh, thr, above);
        const int32_t take = kc - above; // keys equal to thr taken, lowest columns first
        uint32_t eq = 0;
        for (int32_t i = s0; i < s1; ++i) {
            eq += order_key(__float_as_uint(s.l[i])) == thr ? 1u : 0u;
        }
        const uint32_t eq_before = block_incl_scan(eq, s.sh) - eq;
        uint32_t sel = 0;
        {
            uint32_t e = eq_before;
            for (int32_t i = s0; i < s1; ++i) {
                const uint32_t key = order_key(__float_as_uint(s.l[i]));
                const bool is_eq = key == thr;
                sel += (key > thr || (is_eq && e < static_cast<uint32_t>(take))) ? 1u : 0u;
                e += is_eq ? 1u : 0u;
            }
        }
        uint32_t pos = block_incl_scan(sel, s.sh) - sel;
        uint32_t e = eq_before;
        for (int32_t i = s0; i < s1; ++i) {
            const uint32_t key = order_key(__float_as_uint(s.l[i]));
            const bool is_eq = key == thr;
            if (key > thr || (is_eq && e < static_cast<uint32_t>(take))) {
                cand[pos++] = composite(key, id_of(a, j0 + i));
            }
            e += is_eq ? 1u : 0u;
        }
    }
    if (tid == 0) {
        ws<int32_t>(a, wso.cand_n)[t * n_chunk + c] = kc;
    }
    if (!last_of_row(ws<uint32_t>(a, wso.ctr) + t * n_ctr + ctr_select)) {
        return;
    }
    // the row's last block: the card's top k of the chunks' lists (chunk-major is ascending ID order)
    if (tid == 0) {
        int32_t o = 0;
        for (int32_t q = 0; q < n_chunk; ++q) {
            s.off[q] = o;
            o += ws<int32_t>(a, wso.cand_n)[t * n_chunk + q];
        }
        s.off[n_chunk] = o;
    }
    __syncthreads();
    const int32_t n_all = s.off[n_chunk];
    const int32_t kk = min(k, n_all);
    const uint64_t * call = ws<uint64_t>(a, wso.cand) + static_cast<int64_t>(t) * n_chunk * max_k;
    const int32_t * off = s.off;
    // candidate i (0 .. n_all - 1) in chunk-major order
    const auto at = [call, off](int32_t i) {
        int32_t q = 0;
        while (off[q + 1] <= i) {
            ++q;
        }
        return call[static_cast<int64_t>(q) * max_k + (i - off[q])];
    };
    for (int32_t i = tid; i < max_k; i += n_thread) {
        s.sel[i] = 0ull;
    }
    __syncthreads();
    if (kk > 0) {
        uint32_t thr = 0;
        int32_t above = 0;
        radix_kth([&at](int32_t i) { return static_cast<uint32_t>(at(i) >> 32); }, n_all, kk, s.hist, s.sh, thr, above);
        const int32_t take = kk - above;
        const int32_t pa = (n_all + n_thread - 1) / n_thread;
        const int32_t a0 = min(n_all, tid * pa), a1 = min(n_all, a0 + pa);
        uint32_t eq = 0;
        for (int32_t i = a0; i < a1; ++i) {
            eq += static_cast<uint32_t>(at(i) >> 32) == thr ? 1u : 0u;
        }
        const uint32_t eq_before = block_incl_scan(eq, s.sh) - eq;
        uint32_t sel = 0;
        {
            uint32_t e = eq_before;
            for (int32_t i = a0; i < a1; ++i) {
                const uint32_t key = static_cast<uint32_t>(at(i) >> 32);
                sel += (key > thr || (key == thr && e < static_cast<uint32_t>(take))) ? 1u : 0u;
                e += key == thr ? 1u : 0u;
            }
        }
        uint32_t pos = block_incl_scan(sel, s.sh) - sel;
        uint32_t e = eq_before;
        for (int32_t i = a0; i < a1; ++i) {
            const uint64_t ck = at(i);
            const uint32_t key = static_cast<uint32_t>(ck >> 32);
            if (key > thr || (key == thr && e < static_cast<uint32_t>(take))) {
                s.sel[pos++] = ck;
            }
            e += key == thr ? 1u : 0u;
        }
    }
    __syncthreads();
    // bitonic sort of the 256 slots, descending (empty slots are 0, below every candidate)
    for (int32_t size = 2; size <= max_k; size <<= 1) {
        for (int32_t stride = size / 2; stride > 0; stride >>= 1) {
            const int32_t i = tid;
            const int32_t j = i ^ stride;
            if (j > i) {
                const bool desc = (i & size) == 0;
                const uint64_t x = s.sel[i], y = s.sel[j];
                if ((x < y) == desc) {
                    s.sel[i] = y;
                    s.sel[j] = x;
                }
            }
            __syncthreads();
        }
    }
    // the card's list: n, then (logit, ID) pairs, to the peer's slot and this card's own copy
    uint32_t * own = ws<uint32_t>(a, wso.own_a) + hdr + t * a_row;
    uint32_t * peer = reinterpret_cast<uint32_t *>(a.v.a_peer[par(a)]) + hdr + t * a_row;
    for (int32_t i = tid; i < (kk + 1) / 2; i += n_thread) {
        const uint64_t x = s.sel[2 * i];
        const uint64_t y = 2 * i + 1 < kk ? s.sel[2 * i + 1] : 0ull;
        const uint32_t ybits = 2 * i + 1 < kk ? key_bits(static_cast<uint32_t>(y >> 32)) : 0u;
        const uint32_t yid = 2 * i + 1 < kk ? static_cast<uint32_t>(comp_id(y)) : 0u;
        const uint4 w =
            make_uint4(key_bits(static_cast<uint32_t>(x >> 32)), static_cast<uint32_t>(comp_id(x)), ybits, yid);
        if (2 * i + 1 < kk) {
            *reinterpret_cast<uint4 *>(own + 4 + 4 * i) = w;
            *reinterpret_cast<uint4 *>(peer + 4 + 4 * i) = w;
        } else {
            *reinterpret_cast<uint2 *>(own + 4 + 4 * i) = make_uint2(w.x, w.y);
            *reinterpret_cast<uint2 *>(peer + 4 + 4 * i) = make_uint2(w.x, w.y);
        }
    }
    if (tid == 0) {
        const uint4 h = make_uint4(static_cast<uint32_t>(kk), 0u, 0u, 0u);
        *reinterpret_cast<uint4 *>(own) = h;
        *reinterpret_cast<uint4 *>(peer) = h;
        if (t == 0 && a.header != 0) {
            send_header(a.v, a.v.a_peer[par(a)]);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// top-k path, tk_decide: block t merges row t's two lists, applies top-p, min-p and temperature, and draws (or, for a
// verify row, takes the accept step); the last block writes the verdict. In the draft's pick it writes q_list[t] and
// the proposal instead.
struct tk_smem {
    uint64_t own[max_k], peer[max_k], L[max_k];
    uint64_t w1[max_k], wt[max_k], wr[max_k];
    int32_t qid[max_k];
    float qp[max_k];
    uint64_t sh64[n_warp];
    int32_t n_own, n_peer, kk, n1, n2, pick, rpick, qn;
    int32_t x_idx;
    float qx;
};

__device__ int32_t count_greater(const uint64_t * list, int32_t n, uint64_t key) {
    int32_t lo = 0, hi = n; // list sorted descending: the first index whose entry is not greater
    while (lo < hi) {
        const int32_t mid = (lo + hi) / 2;
        if (list[mid] > key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// the first index i < n with cum[i] > target (inclusive prefix sums in w), -1 when none; every thread calls it
__device__ int32_t first_above(const uint64_t * cum, int32_t n, uint64_t target, int32_t * out) {
    if (threadIdx.x == 0) {
        *out = -1;
    }
    __syncthreads();
    for (int32_t i = threadIdx.x; i < n; i += n_thread) {
        if (cum[i] > target && (i == 0 || cum[i - 1] <= target)) {
            *out = i;
        }
    }
    __syncthreads();
    return *out;
}

__global__ void __launch_bounds__(n_thread) tk_decide_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ tk_smem s;
    const int32_t t = blockIdx.x, tid = threadIdx.x;
    const params & P = *a.v.prm;
    const int32_t k = P.top_k;
    const uint32_t * own_row = ws<uint32_t>(a, wso.own_a) + hdr + t * a_row;
    uint32_t * slot_row = reinterpret_cast<uint32_t *>(a.v.a_own[par(a)]) + hdr + t * a_row;
    if (t == 0 && tid == 0 && a.header != 0) {
        check_header(a.v, a.v.a_own[par(a)]);
    }
    if (tid == 0) {
        s.n_own = static_cast<int32_t>(own_row[0]);
        const uint4 h = poll4(slot_row, a.v.err); // the peer's count word, polled as its 16-byte group
        fn::xchg::reset4(reinterpret_cast<uint4 *>(slot_row));
        s.n_peer = h.x == sentinel ? 0 : min(static_cast<int32_t>(h.x), max_k);
    }
    __syncthreads();
    for (int32_t i = tid; i < max_k; i += n_thread) {
        s.own[i] = 0ull;
        s.peer[i] = 0ull;
        s.L[i] = 0ull;
        if (i < s.n_own) {
            const uint32_t * w = own_row + 4 + 2 * i;
            s.own[i] = composite(order_key(w[0]), static_cast<int32_t>(w[1]));
        }
        if (i < s.n_peer) {
            uint32_t * w = slot_row + 4 + 2 * i;
            const uint2 v = poll2w(w, a.v.err);
            fn::xchg::reset2(reinterpret_cast<uint2 *>(w));
            s.peer[i] = composite(order_key(v.x), static_cast<int32_t>(v.y));
        }
    }
    __syncthreads();
    // the merged top k: a candidate's rank is its index plus the other list's greater entries (keys are unique)
    for (int32_t i = tid; i < s.n_own; i += n_thread) {
        const int32_t r = i + count_greater(s.peer, s.n_peer, s.own[i]);
        if (r < k) {
            s.L[r] = s.own[i];
        }
    }
    for (int32_t i = tid; i < s.n_peer; i += n_thread) {
        const int32_t r = i + count_greater(s.own, s.n_own, s.peer[i]);
        if (r < k) {
            s.L[r] = s.peer[i];
        }
    }
    if (tid == 0) {
        s.kk = min(k, s.n_own + s.n_peer);
    }
    __syncthreads();
    const int32_t kk = s.kk;
    rowres_t res = {0, -1, -1, 0};
    const float M = kk > 0 ? comp_logit(s.L[0]) : 0.0f;
    const bool greedy = (P.flags & f_greedy) != 0u;
    const bool verify_row = is_verify(a) && t < a.rows - 1;
    const int32_t x = verify_row ? a.desc->tok[t + 1] : -1;
    const uint32_t pos = row_position(a, t);
    uint64_t u0 = 0, u1 = 0;
    if (kk == 0 || !(M == M)) {
        if (tid == 0) {
            atomicOr(a.v.err, fn::xchg::err_samp_empty);
        }
    } else if (greedy) {
        res.draw = comp_id(s.L[0]);
        res.accept = x == res.draw ? 1 : 0;
    } else {
        // top-p on the merged list (T = 1, at the list's maximum), then min-p, then temperature
        const int32_t i = tid;
        const float li = i < kk ? comp_logit(s.L[i]) : 0.0f;
        const uint64_t w1 = i < kk ? fixed_w(expf(__fsub_rn(li, M)), two50f) : 0ull;
        s.w1[i] = block_incl_scan(w1, s.sh64);
        __syncthreads(); // every entry stored before any thread reads another's
        if (tid == 0) {
            s.n1 = kk;
            s.n2 = kk;
        }
        __syncthreads();
        const uint64_t z1 = s.w1[kk - 1];
        if (P.top_p < 1.0f) {
            const double tgt = static_cast<double>(P.top_p) * static_cast<double>(z1);
            if (i < kk && static_cast<double>(s.w1[i]) >= tgt && (i == 0 || static_cast<double>(s.w1[i - 1]) < tgt)) {
                s.n1 = i + 1;
            }
        }
        if (P.min_p > 0.0f) {
            const float thr = __fadd_rn(M, logf(P.min_p));
            if (i < kk && i > 0 && li < thr && comp_logit(s.L[i - 1]) >= thr) {
                s.n2 = i;
            }
        }
        __syncthreads();
        const int32_t nk = min(s.n1, s.n2);
        const float Mt = __fdiv_rn(M, P.temp);
        const uint64_t wt = i < nk ? fixed_w(expf(__fsub_rn(__fdiv_rn(li, P.temp), Mt)), two50f) : 0ull;
        s.wt[i] = block_incl_scan(wt, s.sh64);
        __syncthreads();
        const uint64_t zt = s.wt[nk - 1];
        if (zt == 0ull) {
            if (tid == 0) {
                atomicOr(a.v.err, fn::xchg::err_samp_empty);
            }
        } else {
            const uint32_t pur = a.draft != 0 ? p_draft : p_shard;
            uniforms((static_cast<uint64_t>(P.seed_hi) << 32) | P.seed_lo,
                     (static_cast<uint64_t>(P.req_hi) << 32) | P.req_lo, pos, pur, u0, u1);
            const int32_t pk = first_above(s.wt, nk, scaled_target(u0, zt), &s.pick);
            res.draw = pk >= 0 ? comp_id(s.L[pk]) : comp_id(s.L[nk - 1]);
            if (a.draft != 0) {
                // the draft's kept list: q at each ID, the same on both cards
                qlist & q = a.v.q_list[a.pick_t];
                if (i < nk) {
                    q.id[i] = comp_id(s.L[i]);
                    q.q[i] = static_cast<float>(static_cast<double>(wt) / static_cast<double>(zt));
                }
                if (tid == 0) {
                    q.n = nk;
                    *a.proposal = res.draw;
                }
            }
            if (verify_row) {
                // p(x) and q(x); the accept test u * q(x) < p(x) (common_sampler_rejection_step)
                const qlist & q = a.v.q_list[t];
                if (tid == 0) {
                    s.qn = min(q.n, max_k);
                    s.x_idx = -1;
                    s.qx = 0.0f;
                }
                __syncthreads();
                if (i < s.qn) {
                    s.qid[i] = q.id[i];
                    s.qp[i] = q.q[i];
                    if (q.id[i] == x) {
                        s.qx = q.q[i];
                    }
                }
                if (i < nk && comp_id(s.L[i]) == x) {
                    s.x_idx = i;
                }
                __syncthreads();
                // x's integer weight is the difference of the inclusive sums at its index
                const int32_t xi = s.x_idx;
                const uint64_t wx = xi < 0 ? 0ull : xi == 0 ? s.wt[0] : s.wt[xi] - s.wt[xi - 1];
                const double px = static_cast<double>(wx) / static_cast<double>(zt);
                uint64_t ua = 0, ub = 0;
                uniforms((static_cast<uint64_t>(P.seed_hi) << 32) | P.seed_lo,
                         (static_cast<uint64_t>(P.req_hi) << 32) | P.req_lo, pos, p_accept, ua, ub);
                res.accept = unit_double(ua) * static_cast<double>(s.qx) < px ? 1 : 0;
                // the residual max(0, p - q) over the kept list, or p when it has no mass
                double qi = 0.0;
                if (i < nk) {
                    const int32_t id = comp_id(s.L[i]);
                    for (int32_t j = 0; j < s.qn; ++j) {
                        if (s.qid[j] == id) {
                            qi = s.qp[j];
                        }
                    }
                }
                const double pi = static_cast<double>(wt) / static_cast<double>(zt);
                const double r = pi - qi;
                const uint64_t wr = i < nk && r > 0.0 ? __double2ull_rn(r * two60) : 0ull;
                s.wr[i] = block_incl_scan(wr, s.sh64);
                __syncthreads();
                const uint64_t rt = s.wr[nk - 1];
                uint64_t ur = 0, us = 0;
                uniforms((static_cast<uint64_t>(P.seed_hi) << 32) | P.seed_lo,
                         (static_cast<uint64_t>(P.req_hi) << 32) | P.req_lo, pos, p_resid_shard, ur, us);
                int32_t rk = -1;
                if (rt > 0ull) {
                    rk = first_above(s.wr, nk, scaled_target(ur, rt), &s.rpick);
                } else {
                    rk = first_above(s.wt, nk, scaled_target(ur, zt), &s.rpick);
                }
                res.resid = rk >= 0 ? comp_id(s.L[rk]) : comp_id(s.L[nk - 1]);
            }
        }
    }
    if (a.draft != 0) {
        if (greedy && kk > 0 && tid == 0) {
            qlist & q = a.v.q_list[a.pick_t];
            q.n = 1;
            q.id[0] = res.draw;
            q.q[0] = 1.0f;
            *a.proposal = res.draw;
        }
        return;
    }
    publish_and_verdict(a, res, false);
}

// ---------------------------------------------------------------------------------------------------------------
// full path, max: each card's maximum (composite key, so a tie keeps the lower ID); min-p alone sends it (samp_a)
__global__ void __launch_bounds__(n_thread) max_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ xform x;
    __shared__ uint64_t wm[n_warp];
    const int32_t c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    int32_t j0 = 0, m = 0;
    chunk_bounds(a.n, c, j0, m);
    if (tid == 0) {
        make_xform(a, t, x);
    }
    __syncthreads();
    const float * row = a.logits + static_cast<int64_t>(t) * a.n + j0;
    uint64_t best = 0ull;
    for (int32_t i = tid; i < m / 4; i += n_thread) {
        const float4 q = __ldcg(reinterpret_cast<const float4 *>(row) + i);
        const float v4[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int32_t e = 0; e < 4; ++e) {
            const int32_t id = id_of(a, j0 + 4 * i + e);
            const uint64_t k = composite(order_key(__float_as_uint(transform(v4[e], id, x, a.v))), id);
            best = k > best ? k : best;
        }
    }
    best = warp_max_u64(best);
    if (tid % 32 == 0) {
        wm[tid / 32] = best;
    }
    __syncthreads();
    uint64_t * mk = ws<uint64_t>(a, wso.maxkey) + t;
    if (tid == 0) {
        uint64_t b = wm[0];
        for (int32_t w = 1; w < n_warp; ++w) {
            b = wm[w] > b ? wm[w] : b;
        }
        atomicMax(reinterpret_cast<unsigned long long *>(mk), static_cast<unsigned long long>(b));
    }
    if (!last_of_row(ws<uint32_t>(a, wso.ctr) + t * n_ctr + ctr_max)) {
        return;
    }
    if (tid == 0) {
        const uint64_t k = atomicExch(reinterpret_cast<unsigned long long *>(mk), 0ull);
        const uint32_t key = static_cast<uint32_t>(k >> 32);
        cardres_t & cr = ws<cardres_t>(a, wso.cardres)[t];
        cr.mkey = key;
        cr.has = key > 0x007fffffu && key < 0xff800000u ? 1u : 0u; // a finite maximum (above -inf, below +inf)
        if ((a.v.prm->flags & f_topp) == 0u) {
            const uint4 w = make_uint4(cr.has, key_bits(key), 0u, 0u);
            *reinterpret_cast<uint4 *>(ws<uint32_t>(a, wso.own_a) + hdr + t * a_row) = w;
            *reinterpret_cast<uint4 *>(reinterpret_cast<uint32_t *>(a.v.a_peer[par(a)]) + hdr + t * a_row) = w;
            if (t == 0 && a.header != 0) {
                send_header(a.v, a.v.a_peer[par(a)]);
            }
        }
    }
}

// full path, hist: the absolute-grid histogram (w = 1/64) from the card's top bin, W bins and the tail, counts and
// integer sums at 2^46 relative to M_r (T = 1); the row's last block sends it (samp_a) and clears the accumulators
struct hist_smem {
    uint32_t cnt[hist_bins + 1];
    unsigned long long sum[hist_bins + 1];
    xform x;
};

__global__ void __launch_bounds__(n_thread) hist_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ hist_smem s;
    const int32_t c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const int32_t W = a.v.prm->window_bins;
    int32_t j0 = 0, m = 0;
    chunk_bounds(a.n, c, j0, m);
    if (tid == 0) {
        make_xform(a, t, s.x);
    }
    for (int32_t b = tid; b <= W; b += n_thread) {
        s.cnt[b] = 0u;
        s.sum[b] = 0ull;
    }
    __syncthreads();
    const cardres_t cr = ws<cardres_t>(a, wso.cardres)[t];
    uint32_t * ghc = ws<uint32_t>(a, wso.ghc) + t * (hist_bins + 1);
    unsigned long long * ghs = ws<unsigned long long>(a, wso.ghs) + t * (hist_bins + 1);
    if (cr.has != 0u) {
        const float M = __uint_as_float(key_bits(cr.mkey));
        const int32_t tb = static_cast<int32_t>(floorf(__fmul_rn(M, 64.0f)));
        const float * row = a.logits + static_cast<int64_t>(t) * a.n + j0;
        for (int32_t i = tid; i < m / 4; i += n_thread) {
            const float4 q = __ldcg(reinterpret_cast<const float4 *>(row) + i);
            const float v4[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int32_t e = 0; e < 4; ++e) {
                const float l = transform(v4[e], id_of(a, j0 + 4 * i + e), s.x, a.v);
                const uint64_t f = fixed_w(expf(__fsub_rn(l, M)), two46f);
                if (f == 0ull) {
                    continue;
                }
                const int32_t j = tb - static_cast<int32_t>(floorf(__fmul_rn(l, 64.0f)));
                const int32_t b = j < W ? j : W;
                atomicAdd(&s.cnt[b], 1u);
                atomicAdd(&s.sum[b], static_cast<unsigned long long>(f));
            }
        }
        __syncthreads();
        for (int32_t b = tid; b <= W; b += n_thread) {
            if (s.cnt[b] != 0u) {
                atomicAdd(ghc + b, s.cnt[b]);
                atomicAdd(ghs + b, s.sum[b]);
            }
        }
    }
    if (!last_of_row(ws<uint32_t>(a, wso.ctr) + t * n_ctr + ctr_hist)) {
        return;
    }
    uint32_t * own = ws<uint32_t>(a, wso.own_a) + hdr + t * a_row;
    uint32_t * peer = reinterpret_cast<uint32_t *>(a.v.a_peer[par(a)]) + hdr + t * a_row;
    for (int32_t b = tid; b <= W; b += n_thread) {
        const uint32_t n = __ldcg(ghc + b);
        const uint64_t sm = canon64(__ldcg(ghs + b));
        ghc[b] = 0u;
        ghs[b] = 0ull;
        const uint4 w = make_uint4(n, lo32(sm), hi32(sm), 0u);
        *reinterpret_cast<uint4 *>(own + 4 + 4 * b) = w;
        *reinterpret_cast<uint4 *>(peer + 4 + 4 * b) = w;
    }
    if (tid == 0) {
        const uint4 w = make_uint4(cr.has, key_bits(cr.mkey), static_cast<uint32_t>(W), 0u);
        *reinterpret_cast<uint4 *>(own) = w;
        *reinterpret_cast<uint4 *>(peer) = w;
        if (t == 0 && a.header != 0) {
            send_header(a.v, a.v.a_peer[par(a)]);
        }
    }
}

// full path, plan (block t): both cards' histograms at the global maximum's scale, the cut bin c (the first bin whose
// running mass reaches top_p * Z), min-p's threshold, and whether the cut bin's members are needed
struct plan_smem {
    uint32_t has[2], mbits[2];
    int32_t tb[2], o[2];
    uint64_t sh64[n_warp];
    unsigned long long tail_s[2];
    uint32_t tail_c[2];
    int32_t c_first, c_last;
};

__global__ void __launch_bounds__(n_thread) plan_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ plan_smem s;
    const int32_t t = blockIdx.x, tid = threadIdx.x, card = a.v.card, peer = 1 - card;
    const params & P = *a.v.prm;
    const int32_t W = P.window_bins;
    const uint32_t * own_row = ws<uint32_t>(a, wso.own_a) + hdr + t * a_row;
    uint32_t * slot_row = reinterpret_cast<uint32_t *>(a.v.a_own[par(a)]) + hdr + t * a_row;
    uint32_t * pbins = ws<uint32_t>(a, wso.mem_peer) + static_cast<int64_t>(t) * b_row; // the peer's bins, copied
    if (t == 0 && tid == 0 && a.header != 0) {
        check_header(a.v, a.v.a_own[par(a)]);
    }
    if (tid == 0) {
        const uint4 h = poll4(slot_row, a.v.err);
        fn::xchg::reset4(reinterpret_cast<uint4 *>(slot_row));
        s.has[peer] = h.x == 1u ? 1u : 0u;
        s.mbits[peer] = h.y;
        s.has[card] = own_row[0];
        s.mbits[card] = own_row[1];
        s.tail_s[0] = s.tail_s[1] = 0ull;
        s.tail_c[0] = s.tail_c[1] = 0u;
        s.c_first = 0x7fffffff;
        s.c_last = -1;
    }
    __syncthreads();
    for (int32_t b = tid; b <= W; b += n_thread) {
        uint32_t * w = slot_row + 4 + 4 * b;
        const uint4 v = poll4(w, a.v.err);
        fn::xchg::reset4(reinterpret_cast<uint4 *>(w));
        *reinterpret_cast<uint4 *>(pbins + 4 * b) = v;
    }
    __syncthreads();
    plan_t & pl = ws<plan_t>(a, wso.plan)[t];
    float M = 0.0f;
    bool any = false;
    for (int32_t r = 0; r < 2; ++r) {
        if (s.has[r] != 0u) {
            const float mr = __uint_as_float(s.mbits[r]);
            M = any ? fmaxf(M, mr) : mr;
            any = true;
        }
    }
    if (!any) {
        if (tid == 0) {
            pl.flags = pf_empty;
            atomicOr(a.v.err, fn::xchg::err_samp_empty);
        }
        return;
    }
    const int32_t tg = static_cast<int32_t>(floorf(__fmul_rn(M, 64.0f)));
    double e[2] = {0.0, 0.0};
    for (int32_t r = 0; r < 2; ++r) {
        if (s.has[r] != 0u) {
            const float mr = __uint_as_float(s.mbits[r]);
            e[r] = exp(static_cast<double>(mr) - static_cast<double>(M));
            if (tid == 0) {
                s.tb[r] = static_cast<int32_t>(floorf(__fmul_rn(mr, 64.0f)));
                s.o[r] = tg - s.tb[r];
            }
        }
    }
    __syncthreads();
    // card r's card-local bin j (0 .. W-1) and its tail bin W
    const auto cnt_of = [&](int32_t r, int32_t j) -> uint32_t { return r == card ? own_row[4 + 4 * j] : pbins[4 * j]; };
    const auto sum_of = [&](int32_t r, int32_t j) -> uint64_t {
        return r == card ? join64(own_row[5 + 4 * j], own_row[6 + 4 * j]) : join64(pbins[1 + 4 * j], pbins[2 + 4 * j]);
    };
    // each card's mass below the global window: its bins j >= W - o_r and its own tail (integer sums)
    for (int32_t r = 0; r < 2; ++r) {
        if (s.has[r] == 0u) {
            continue;
        }
        unsigned long long ts = 0ull;
        uint32_t tc = 0u;
        for (int32_t j = tid; j <= W; j += n_thread) {
            if (j == W || j + s.o[r] >= W) {
                ts += sum_of(r, j);
                tc += cnt_of(r, j);
            }
        }
        atomicAdd(&s.tail_s[r], ts);
        atomicAdd(&s.tail_c[r], tc);
    }
    __syncthreads();
    // global bin g's mass (integers at the global scale; card 0 first) and count; g == W is the tail
    const auto gmass = [&](int32_t g, uint32_t & cnt) -> uint64_t {
        uint64_t mm = 0ull;
        cnt = 0u;
        for (int32_t r = 0; r < 2; ++r) {
            if (s.has[r] == 0u) {
                continue;
            }
            if (g == W) {
                mm += __double2ull_rn(static_cast<double>(s.tail_s[r]) * e[r]);
                cnt += s.tail_c[r];
            } else {
                const int32_t j = g - s.o[r];
                if (j >= 0 && j < W) {
                    mm += __double2ull_rn(static_cast<double>(sum_of(r, j)) * e[r]);
                    cnt += cnt_of(r, j);
                }
            }
        }
        return mm;
    };
    // the running mass from the top bin: each thread a contiguous run of bins, then a block scan
    const int32_t nb = W + 1;
    const int32_t per = (nb + n_thread - 1) / n_thread;
    const int32_t g0 = min(nb, tid * per), g1 = min(nb, g0 + per);
    uint64_t local = 0ull;
    for (int32_t g = g0; g < g1; ++g) {
        uint32_t cnt = 0u;
        local += gmass(g, cnt);
    }
    const uint64_t incl = block_incl_scan(local, s.sh64);
    __shared__ uint64_t z_all;
    if (tid == n_thread - 1) {
        z_all = incl;
    }
    __syncthreads();
    const uint64_t Z = z_all;
    const bool topp = P.top_p < 1.0f;
    const uint64_t target = !topp ? 0ull
                            : P.top_p <= 0.0f
                                ? 0ull
                                : static_cast<uint64_t>(ceil(static_cast<double>(P.top_p) * static_cast<double>(Z)));
    {
        uint64_t cum = incl - local;
        for (int32_t g = g0; g < g1; ++g) {
            uint32_t cnt = 0u;
            const uint64_t mm = gmass(g, cnt);
            cum += mm;
            if (cnt != 0u) {
                atomicMax(&s.c_last, g);
                if (cum >= target) {
                    atomicMin(&s.c_first, g);
                }
            }
        }
    }
    __syncthreads();
    if (tid != 0) {
        return;
    }
    const int32_t c = s.c_first != 0x7fffffff ? s.c_first : s.c_last;
    // the mass before bin c, recounted in bin order (one thread)
    uint64_t before = 0ull;
    for (int32_t g = 0; g < c; ++g) {
        uint32_t cnt = 0u;
        before += gmass(g, cnt);
    }
    uint32_t flags = topp ? pf_topp : 0u;
    float thr = 0.0f;
    if ((P.flags & f_minp) != 0u) {
        thr = __fadd_rn(M, logf(P.min_p));
        flags |= pf_thr;
    }
    const float upper = static_cast<float>(tg - c + 1) * 0.015625f; // bin c's upper edge
    const bool dominated = (flags & pf_thr) != 0u && thr >= upper;
    int32_t nm[2] = {0, 0};
    for (int32_t r = 0; r < 2; ++r) {
        if (s.has[r] == 0u) {
            continue;
        }
        if (c == W) {
            nm[r] = static_cast<int32_t>(s.tail_c[r]);
        } else {
            const int32_t j = c - s.o[r];
            nm[r] = j >= 0 && j < W ? static_cast<int32_t>(cnt_of(r, j)) : 0;
        }
    }
    if (topp && !dominated && nm[0] + nm[1] > 1) {
        flags |= pf_members;
    }
    pl.kcut = 0ull;
    pl.target = target;
    pl.cum_before = before;
    pl.e[0] = e[0];
    pl.e[1] = e[1];
    pl.m_card[0] = __uint_as_float(s.mbits[0]);
    pl.m_card[1] = __uint_as_float(s.mbits[1]);
    pl.m = M;
    pl.thr = thr;
    pl.c = topp ? c : W + 1; // without top-p every bin is kept
    pl.tg = tg;
    pl.w = W;
    pl.flags = flags;
    pl.n_mem[0] = nm[0];
    pl.n_mem[1] = nm[1];
}

// full path, members: each card's members of the cut bin, (logit, ID) in arrival order, to the peer and its own copy
__global__ void __launch_bounds__(n_thread) members_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ xform x;
    const int32_t c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const plan_t & pl = ws<plan_t>(a, wso.plan)[t];
    if ((pl.flags & pf_members) == 0u) {
        return;
    }
    int32_t j0 = 0, m = 0;
    chunk_bounds(a.n, c, j0, m);
    if (tid == 0) {
        make_xform(a, t, x);
    }
    __syncthreads();
    const float M = pl.m_card[a.v.card];
    uint32_t * pos = ws<uint32_t>(a, wso.ctr) + t * n_ctr + ctr_mempos;
    uint32_t * own = ws<uint32_t>(a, wso.own_b) + static_cast<int64_t>(t) * b_row;
    uint32_t * peer = reinterpret_cast<uint32_t *>(a.v.b_peer[par(a)]) + static_cast<int64_t>(t) * b_row;
    const float * row = a.logits + static_cast<int64_t>(t) * a.n + j0;
    for (int32_t i = tid; i < m / 4; i += n_thread) {
        const float4 q = __ldcg(reinterpret_cast<const float4 *>(row) + i);
        const float v4[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int32_t e = 0; e < 4; ++e) {
            const int32_t id = id_of(a, j0 + 4 * i + e);
            const float l = transform(v4[e], id, x, a.v);
            if (fixed_w(expf(__fsub_rn(l, M)), two46f) == 0ull) {
                continue;
            }
            int32_t g = pl.tg - static_cast<int32_t>(floorf(__fmul_rn(l, 64.0f)));
            g = g > pl.w ? pl.w : g;
            if (g == pl.c) {
                const uint32_t p = atomicAdd(pos, 1u);
                const uint2 w = make_uint2(fn::xchg::canonical(l), static_cast<uint32_t>(id));
                *reinterpret_cast<uint2 *>(own + 2 * p) = w;
                *reinterpret_cast<uint2 *>(peer + 2 * p) = w;
            }
        }
    }
    if (last_of_row(ws<uint32_t>(a, wso.ctr) + t * n_ctr + ctr_members) && tid == 0) {
        // the members sent must be the plan's count of this card (the peer polls exactly that many)
        if (static_cast<int32_t>(__ldcg(pos)) != pl.n_mem[a.v.card]) {
            atomicOr(a.v.err, fn::xchg::err_samp_count);
        }
        *pos = 0u;
    }
}

// full path, resolve (block t): the cut member K*, by a radix select over the composite key weighted by integer mass
// at the global scale: the first member, in (logit, lower ID) order, whose running mass reaches the target
__global__ void __launch_bounds__(n_thread) resolve_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ unsigned long long hm[256];
    __shared__ uint32_t hc[256];
    __shared__ uint64_t s_prefix;
    __shared__ uint64_t s_mask;
    __shared__ uint64_t s_rem;
    const int32_t t = blockIdx.x, tid = threadIdx.x, card = a.v.card;
    plan_t & pl = ws<plan_t>(a, wso.plan)[t];
    if ((pl.flags & pf_members) == 0u) {
        return;
    }
    const int32_t n_own = pl.n_mem[card], n_peer = pl.n_mem[1 - card];
    const uint32_t * own = ws<uint32_t>(a, wso.own_b) + static_cast<int64_t>(t) * b_row;
    uint32_t * cp = ws<uint32_t>(a, wso.mem_peer) + static_cast<int64_t>(t) * b_row;
    uint32_t * slot = reinterpret_cast<uint32_t *>(a.v.b_own[par(a)]) + static_cast<int64_t>(t) * b_row;
    for (int32_t i = tid; i < n_peer; i += n_thread) {
        const uint2 v = poll2w(slot + 2 * i, a.v.err);
        fn::xchg::reset2(reinterpret_cast<uint2 *>(slot + 2 * i));
        *reinterpret_cast<uint2 *>(cp + 2 * i) = v;
    }
    __syncthreads();
    const int32_t n = n_own + n_peer;
    const auto member = [&](int32_t i, uint64_t & key, uint64_t & mass) {
        const bool mine = i < n_own;
        const uint32_t * w = mine ? own + 2 * i : cp + 2 * (i - n_own);
        const int32_t r = mine ? card : 1 - card;
        const float l = __uint_as_float(w[0]);
        key = composite(order_key(w[0]), static_cast<int32_t>(w[1]));
        mass = __double2ull_rn(static_cast<double>(fixed_w(expf(__fsub_rn(l, pl.m_card[r])), two46f)) * pl.e[r]);
    };
    if (tid == 0) {
        s_prefix = 0ull;
        s_mask = 0ull;
        s_rem = pl.target > pl.cum_before ? pl.target - pl.cum_before : 0ull;
    }
    __syncthreads();
    for (int32_t p = 0; p < 8; ++p) {
        const int32_t shift = 56 - 8 * p;
        hm[tid] = 0ull;
        hc[tid] = 0u;
        __syncthreads();
        const uint64_t prefix = s_prefix, mask = s_mask;
        // the members of a warp step that share a digit are summed in the warp first (integer, order-free), and one
        // lane adds them: sm_70's 64-bit shared atomic is a CAS loop, and members of one bin share their top bytes
        const int32_t lane = tid % 32;
        for (int32_t i0 = 0; i0 < n; i0 += n_thread) {
            const int32_t i = i0 + tid;
            uint64_t key = 0ull, mass = 0ull;
            bool in = false;
            if (i < n) {
                member(i, key, mass);
                in = (key & mask) == prefix;
            }
            const uint32_t d = in ? static_cast<uint32_t>(key >> shift) & 255u : 256u + static_cast<uint32_t>(lane);
            const uint32_t peers = __match_any_sync(0xffffffffu, d);
            uint64_t tot = 0ull;
            for (int32_t j = 0; j < 32; ++j) {
                const uint64_t mj = __shfl_sync(0xffffffffu, in ? mass : 0ull, j);
                tot += ((peers >> j) & 1u) != 0u ? mj : 0ull;
            }
            if (in && lane == __ffs(static_cast<int32_t>(peers)) - 1) {
                atomicAdd(&hm[d], static_cast<unsigned long long>(tot));
                atomicAdd(&hc[d], static_cast<uint32_t>(__popc(peers)));
            }
        }
        __syncthreads();
        if (tid == 0) {
            const uint64_t rem = s_rem;
            uint64_t acc = 0ull, above = 0ull;
            int32_t chosen = -1, lowest = -1;
            for (int32_t d = 255; d >= 0; --d) {
                if (hc[d] == 0u) {
                    continue;
                }
                lowest = d;
                if (chosen < 0 && acc + hm[d] >= rem) {
                    chosen = d;
                    above = acc;
                }
                acc += hm[d];
            }
            if (chosen < 0) { // rounding left the bin short of the target: the whole bin is kept
                chosen = lowest;
                s_rem = ~0ull;
            } else {
                s_rem = rem - above;
            }
            s_prefix = prefix | (static_cast<uint64_t>(chosen) << shift);
            s_mask = mask | (255ull << shift);
        }
        __syncthreads();
    }
    if (tid == 0) {
        pl.kcut = s_prefix;
    }
}

// the kept set of a full-path row (top-p's cut and min-p's threshold; no filter keeps every finite logit)
struct keep_t {
    uint32_t flags;
    float thr;
    int32_t c, tg, w;
    uint64_t kcut;
};

__device__ __forceinline__ bool kept(float l, int32_t id, const keep_t & k) {
    if (!(l == l) || l == -INFINITY) {
        return false;
    }
    if ((k.flags & pf_thr) != 0u && !(l >= k.thr)) {
        return false;
    }
    if ((k.flags & pf_topp) != 0u) {
        int32_t g = k.tg - static_cast<int32_t>(floorf(__fmul_rn(l, 64.0f)));
        g = g > k.w ? k.w : g;
        if (g > k.c) {
            return false;
        }
        if (g == k.c && k.kcut != 0ull && composite(order_key(__float_as_uint(l)), id) < k.kcut) {
            return false;
        }
    }
    return true;
}

// full path, settle: each card's kept mass Z_r at its tempered maximum Mt_r and a local candidate (the chunk by its
// mass, then an ascending-ID scan of integer weights); verify rows add the owner's numerator and q at the draft.
// Sends samp_c. Min-p alone: every block reads both cards' maxima from samp_a (decide resets them).
struct settle_smem {
    float l[max_chunk];
    uint64_t sh64[n_warp];
    float shf[n_warp];
    xform x;
    keep_t k;
    int32_t chosen, cand;
    uint64_t t_int;
    float mt_c;
};

__device__ __forceinline__ float block_max_f(float v, float * sh) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    }
    if (threadIdx.x % 32 == 0) {
        sh[threadIdx.x / 32] = v;
    }
    __syncthreads();
    float r = sh[0];
    for (int32_t w = 1; w < n_warp; ++w) {
        r = fmaxf(r, sh[w]);
    }
    __syncthreads();
    return r;
}

// the uniforms of a row's position and purpose
__device__ __forceinline__ uint64_t row_uniform(const params & P, uint32_t pos, uint32_t pur) {
    uint64_t u0 = 0, u1 = 0;
    uniforms((static_cast<uint64_t>(P.seed_hi) << 32) | P.seed_lo, (static_cast<uint64_t>(P.req_hi) << 32) | P.req_lo,
             pos, pur, u0, u1);
    return u0;
}

// the first index of a chunk whose running integer weight exceeds target (block; the chunk's weights by wfn)
template <typename Wfn>
__device__ int32_t chunk_pick(int32_t m, const Wfn & wfn, uint64_t target, uint64_t * sh64, int32_t * out) {
    const int32_t tid = threadIdx.x;
    const int32_t per = (m + n_thread - 1) / n_thread;
    const int32_t s0 = min(m, tid * per), s1 = min(m, s0 + per);
    uint64_t local = 0ull;
    for (int32_t i = s0; i < s1; ++i) {
        local += wfn(i);
    }
    if (tid == 0) {
        *out = -1;
    }
    uint64_t cum = block_incl_scan(local, sh64) - local;
    int32_t last_pos = -1;
    for (int32_t i = s0; i < s1; ++i) {
        const uint64_t w = wfn(i);
        if (w == 0ull) {
            continue;
        }
        last_pos = i;
        if (cum <= target && cum + w > target) {
            *out = i;
        }
        cum += w;
    }
    __syncthreads();
    // none (only an out-of-range target can cause it): the chunk's last positive weight
    __shared__ int32_t s_last;
    if (*out < 0) {
        if (tid == 0) {
            s_last = -1;
        }
        __syncthreads();
        if (last_pos >= 0) {
            atomicMax(&s_last, last_pos);
        }
        __syncthreads();
        if (tid == 0) {
            *out = s_last;
        }
        __syncthreads();
    }
    return *out;
}

__global__ void __launch_bounds__(n_thread) settle_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ settle_smem s;
    const int32_t c = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const params & P = *a.v.prm;
    const bool topp = (P.flags & f_topp) != 0u, minp = (P.flags & f_minp) != 0u;
    plan_t & pl = ws<plan_t>(a, wso.plan)[t];
    int32_t j0 = 0, m = 0;
    chunk_bounds(a.n, c, j0, m);
    if (tid == 0) {
        make_xform(a, t, s.x);
        keep_t k = {};
        if (topp) {
            k.flags = pl.flags & (pf_thr | pf_topp);
            k.thr = pl.thr;
            k.c = pl.c;
            k.tg = pl.tg;
            k.w = pl.w;
            k.kcut = pl.kcut;
        } else if (minp) {
            // min-p alone: both cards' maxima from samp_a (read, not reset: decide resets them)
            if (c == 0 && t == 0 && a.header != 0) {
                uint32_t * h = reinterpret_cast<uint32_t *>(a.v.a_own[par(a)]);
                const uint4 peer = poll4(h, a.v.err);
                const uint4 own = record_words(*a.v.rec);
                if (peer.x != own.x || peer.y != own.y || peer.z != own.z || peer.w != own.w) {
                    atomicOr(a.v.err, fn::xchg::err_hash);
                } else if (a.v.rec->valid != 0u) {
                    a.v.rec->checked = 1u;
                }
            }
            const uint32_t * own_row = ws<uint32_t>(a, wso.own_a) + hdr + t * a_row;
            const uint4 pr = poll4(reinterpret_cast<uint32_t *>(a.v.a_own[par(a)]) + hdr + t * a_row, a.v.err);
            const uint32_t has[2] = {a.v.card == 0 ? own_row[0] : pr.x, a.v.card == 0 ? pr.x : own_row[0]};
            const uint32_t mb[2] = {a.v.card == 0 ? own_row[1] : pr.y, a.v.card == 0 ? pr.y : own_row[1]};
            float M = 0.0f;
            bool any = false;
            for (int32_t r = 0; r < 2; ++r) {
                if (has[r] == 1u) {
                    M = any ? fmaxf(M, __uint_as_float(mb[r])) : __uint_as_float(mb[r]);
                    any = true;
                }
            }
            k.flags = pf_thr;
            k.thr = any ? __fadd_rn(M, logf(P.min_p)) : INFINITY;
        }
        s.k = k;
    }
    __syncthreads();
    load_chunk(a, t, j0, m, s.x, s.l);
    __syncthreads();
    // the chunk's tempered maximum over its kept part, then its integer mass relative to it
    float mt = -INFINITY;
    for (int32_t i = tid; i < m; i += n_thread) {
        if (kept(s.l[i], id_of(a, j0 + i), s.k)) {
            mt = fmaxf(mt, __fdiv_rn(s.l[i], P.temp));
        }
    }
    mt = block_max_f(mt, s.shf);
    uint64_t sum = 0ull;
    if (mt > -INFINITY) {
        for (int32_t i = tid; i < m; i += n_thread) {
            if (kept(s.l[i], id_of(a, j0 + i), s.k)) {
                sum += fixed_w(expf(__fsub_rn(__fdiv_rn(s.l[i], P.temp), mt)), two50f);
            }
        }
    }
    sum = block_sum(sum, s.sh64);
    part_t * part = ws<part_t>(a, wso.part) + t * n_chunk;
    if (tid == 0) {
        part[c].mt = mt;
        part[c].s = sum;
    }
    if (!last_of_row(ws<uint32_t>(a, wso.ctr) + t * n_ctr + ctr_settle)) {
        return;
    }
    // the row's last block: Mt_r, Z_r, and the local candidate
    __shared__ float s_mtr;
    __shared__ double s_z;
    if (tid == 0) {
        float mtr = -INFINITY;
        for (int32_t q = 0; q < n_chunk; ++q) {
            if (__ldcg(&part[q].s) != 0ull) {
                mtr = fmaxf(mtr, __ldcg(&part[q].mt));
            }
        }
        double z = 0.0;
        for (int32_t q = 0; q < n_chunk; ++q) {
            const uint64_t sq = __ldcg(&part[q].s);
            if (sq != 0ull) {
                z += static_cast<double>(sq) / two50 * exp(static_cast<double>(__ldcg(&part[q].mt)) - mtr);
            }
        }
        s_mtr = mtr;
        s_z = canon_d(z);
        s.chosen = -1;
        if (z > 0.0) {
            const double target = unit_double(row_uniform(P, row_position(a, t), p_local)) * s_z;
            double acc = 0.0;
            int32_t last = -1;
            double before_last = 0.0;
            double before = 0.0;
            for (int32_t q = 0; q < n_chunk; ++q) {
                const uint64_t sq = __ldcg(&part[q].s);
                if (sq == 0ull) {
                    continue;
                }
                const double zq = static_cast<double>(sq) / two50 * exp(static_cast<double>(__ldcg(&part[q].mt)) - mtr);
                last = q;
                before_last = acc;
                if (s.chosen < 0 && acc + zq > target) {
                    s.chosen = q;
                    before = acc;
                }
                acc += zq;
            }
            if (s.chosen < 0) {
                s.chosen = last;
                before = before_last;
            }
            const int32_t q = s.chosen;
            const float mq = __ldcg(&part[q].mt);
            const uint64_t sq = __ldcg(&part[q].s);
            const double units = (target - before) / exp(static_cast<double>(mq) - mtr) * two50;
            s.t_int = units <= 0.0 ? 0ull : units >= static_cast<double>(sq) ? sq - 1 : static_cast<uint64_t>(units);
            s.mt_c = mq;
        }
    }
    __syncthreads();
    int32_t cand = -1;
    if (s.chosen >= 0) {
        int32_t q0 = 0, qm = 0;
        chunk_bounds(a.n, s.chosen, q0, qm);
        load_chunk(a, t, q0, qm, s.x, s.l);
        __syncthreads();
        const float mq = s.mt_c;
        const float * lv = s.l;
        const keep_t & kk = s.k;
        const int32_t pk = chunk_pick(
            qm,
            [&](int32_t i) {
                return kept(lv[i], id_of(a, q0 + i), kk)
                           ? fixed_w(expf(__fsub_rn(__fdiv_rn(lv[i], P.temp), mq)), two50f)
                           : 0ull;
            },
            s.t_int, s.sh64, &s.cand);
        cand = pk >= 0 ? id_of(a, q0 + pk) : -1;
    }
    if (tid != 0) {
        return;
    }
    const float mtr = s_mtr;
    const double z = s_z;
    uint32_t flags = z > 0.0 ? 1u : 0u;
    double wx = 0.0;
    float qx = 0.0f;
    if (is_verify(a) && t < a.rows - 1 && a.ids == nullptr) {
        const int32_t x = a.desc->tok[t + 1];
        if (x >= a.id0 && x < a.id0 + a.n) {
            flags |= 2u;
            const float l = transform(a.logits[static_cast<int64_t>(t) * a.n + (x - a.id0)], x, s.x, a.v);
            if (z > 0.0 && kept(l, x, s.k)) {
                wx = canon_d(static_cast<double>(expf(__fsub_rn(__fdiv_rn(l, P.temp), mtr))));
            }
            qx = a.v.q_shard[static_cast<int64_t>(t) * n_card + (x - a.id0)];
        }
    }
    const uint4 w0 = make_uint4(flags, fn::xchg::canonical(mtr), dlo(z), dhi(z));
    const uint4 w1 =
        make_uint4(cand >= 0 ? static_cast<uint32_t>(cand) : no_cand, dlo(wx), dhi(wx), fn::xchg::canonical(qx));
    uint32_t * own = ws<uint32_t>(a, wso.own_c) + hdr + t * c_row;
    uint32_t * peer = reinterpret_cast<uint32_t *>(a.v.c_peer[par(a)]) + hdr + t * c_row;
    *reinterpret_cast<uint4 *>(own) = w0;
    *reinterpret_cast<uint4 *>(own + 4) = w1;
    *reinterpret_cast<uint4 *>(peer) = w0;
    *reinterpret_cast<uint4 *>(peer + 4) = w1;
    if (!topp) {
        pl.flags = s.k.flags;
        pl.thr = s.k.thr;
    }
    pl.mt_own = mtr;
    if (!topp && !minp && t == 0 && a.header != 0) {
        send_header(a.v, a.v.c_peer[par(a)]);
    }
}

// a samp_c row as words
struct crow {
    uint32_t flags;
    float mt;
    double z;
    int32_t cand;
    double wx;
    float qx;
};
__device__ __forceinline__ crow parse_c(uint4 w0, uint4 w1) {
    crow r;
    r.flags = w0.x;
    r.mt = __uint_as_float(w0.y);
    r.z = join_d(w0.z, w0.w);
    r.cand = static_cast<int32_t>(w1.x);
    r.wx = join_d(w1.y, w1.z);
    r.qx = __uint_as_float(w1.w);
    return r;
}

// full path, decide (one block; thread 0 a row each in turn): the shard by its mass, the token from that card's
// candidate; verify rows: the accept bits. The verdict, unless a verify row was rejected (then resid and decide2).
__global__ void __launch_bounds__(32) decide_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }
    const params & P = *a.v.prm;
    const int32_t card = a.v.card;
    const bool topp = (P.flags & f_topp) != 0u, minp = (P.flags & f_minp) != 0u;
    if (!topp && !minp && a.header != 0) {
        check_header(a.v, a.v.c_own[par(a)]);
    }
    if (!topp && minp && a.header != 0) {
        // the min-p exchange's words, read by every settle block, are reset here (after all of them, stream order)
        fn::xchg::reset4(reinterpret_cast<uint4 *>(a.v.a_own[par(a)]));
        for (int32_t t = 0; t < a.rows; ++t) {
            fn::xchg::reset4(
                reinterpret_cast<uint4 *>(reinterpret_cast<uint32_t *>(a.v.a_own[par(a)]) + hdr + t * a_row));
        }
    }
    ws<vstate_t>(a, wso.vstate)->need = 0;
    rowres_t rr[max_rows] = {};
    for (int32_t t = 0; t < a.rows; ++t) {
        uint32_t * slot = reinterpret_cast<uint32_t *>(a.v.c_own[par(a)]) + hdr + t * c_row;
        const uint4 p0 = poll4(slot, a.v.err), p1 = poll4(slot + 4, a.v.err);
        fn::xchg::reset4(reinterpret_cast<uint4 *>(slot));
        fn::xchg::reset4(reinterpret_cast<uint4 *>(slot + 4));
        const uint32_t * ow = ws<uint32_t>(a, wso.own_c) + hdr + t * c_row;
        const crow own = parse_c(make_uint4(ow[0], ow[1], ow[2], ow[3]), make_uint4(ow[4], ow[5], ow[6], ow[7]));
        const crow peer = parse_c(p0, p1);
        const crow & r0 = card == 0 ? own : peer;
        const crow & r1 = card == 0 ? peer : own;
        const bool h0 = (r0.flags & 1u) != 0u, h1 = (r1.flags & 1u) != 0u;
        const float mt = h0 && h1 ? fmaxf(r0.mt, r1.mt) : h0 ? r0.mt : r1.mt;
        const double e0 = h0 ? exp(static_cast<double>(r0.mt) - mt) : 0.0;
        const double e1 = h1 ? exp(static_cast<double>(r1.mt) - mt) : 0.0;
        const double z0 = r0.z * e0, z1 = r1.z * e1;
        const double z = z0 + z1;
        plan_t & pl = ws<plan_t>(a, wso.plan)[t];
        pl.z = z;
        pl.e_own = card == 0 ? e0 : e1;
        rr[t].draw = -1;
        if (!(z > 0.0)) {
            atomicOr(a.v.err, fn::xchg::err_samp_empty);
            continue;
        }
        const uint32_t pos = row_position(a, t);
        const double u = unit_double(row_uniform(P, pos, p_shard));
        rr[t].draw = u * z < z0 ? r0.cand : r1.cand;
        if (is_verify(a) && t < a.rows - 1) {
            const crow & ow2 = (r0.flags & 2u) != 0u ? r0 : r1;
            const double eo = (r0.flags & 2u) != 0u ? e0 : e1;
            const double px = ow2.wx * eo / z;
            const double ua = unit_double(row_uniform(P, pos, p_accept));
            rr[t].accept = ua * static_cast<double>(ow2.qx) < px ? 1 : 0;
        }
    }
    rowres_t * rs = ws<rowres_t>(a, wso.rowres);
    for (int32_t t = 0; t < a.rows; ++t) {
        rs[t] = rr[t];
    }
    verdict(a, rr, true);
}

// full path, resid (on a rejection at row a): the residual max(0, p - q) on each card as integers at 2^60, its mass
// R_r and a local candidate; sends samp_d
__global__ void __launch_bounds__(n_thread) resid_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    __shared__ settle_smem s;
    const vstate_t vs = *ws<vstate_t>(a, wso.vstate);
    if (vs.need == 0) {
        return;
    }
    const int32_t c = blockIdx.x, t = vs.row, tid = threadIdx.x;
    const params & P = *a.v.prm;
    const plan_t & pl = ws<plan_t>(a, wso.plan)[t];
    int32_t j0 = 0, m = 0;
    chunk_bounds(a.n, c, j0, m);
    if (tid == 0) {
        make_xform(a, t, s.x);
        keep_t k = {};
        k.flags = pl.flags & (pf_thr | pf_topp);
        k.thr = pl.thr;
        k.c = pl.c;
        k.tg = pl.tg;
        k.w = pl.w;
        k.kcut = pl.kcut;
        s.k = k;
    }
    __syncthreads();
    load_chunk(a, t, j0, m, s.x, s.l);
    __syncthreads();
    const float * qrow = a.v.q_shard + static_cast<int64_t>(t) * n_card;
    const float mtr = pl.mt_own;
    const double scale = pl.e_own / pl.z;
    const float * lv = s.l;
    const keep_t & kk = s.k;
    const auto rw = [&](int32_t i) -> uint64_t {
        const double p = kept(lv[i], id_of(a, j0 + i), kk)
                             ? static_cast<double>(expf(__fsub_rn(__fdiv_rn(lv[i], P.temp), mtr))) * scale
                             : 0.0;
        const double r = p - static_cast<double>(qrow[j0 + i]);
        return r > 0.0 ? __double2ull_rn(r * two60) : 0ull;
    };
    uint64_t sum = 0ull;
    for (int32_t i = tid; i < m; i += n_thread) {
        sum += rw(i);
    }
    sum = block_sum(sum, s.sh64);
    part_t * part = ws<part_t>(a, wso.part) + t * n_chunk;
    if (tid == 0) {
        part[c].s = sum;
    }
    if (!last_of_row(ws<uint32_t>(a, wso.ctr) + t * n_ctr + ctr_resid)) {
        return;
    }
    __shared__ uint64_t s_r;
    if (tid == 0) {
        uint64_t r = 0ull;
        for (int32_t q = 0; q < n_chunk; ++q) {
            r += __ldcg(&part[q].s);
        }
        s_r = r;
        s.chosen = -1;
        if (r > 0ull) {
            const uint64_t target = scaled_target(row_uniform(P, row_position(a, t), p_resid_local), r);
            uint64_t acc = 0ull;
            for (int32_t q = 0; q < n_chunk; ++q) {
                const uint64_t sq = __ldcg(&part[q].s);
                if (s.chosen < 0 && acc + sq > target) {
                    s.chosen = q;
                    s.t_int = target - acc;
                }
                acc += sq;
            }
        }
    }
    __syncthreads();
    int32_t cand = -1;
    if (s.chosen >= 0) {
        int32_t q0 = 0, qm = 0;
        chunk_bounds(a.n, s.chosen, q0, qm);
        load_chunk(a, t, q0, qm, s.x, s.l);
        __syncthreads();
        const auto rw2 = [&](int32_t i) -> uint64_t {
            const double p = kept(lv[i], id_of(a, q0 + i), kk)
                                 ? static_cast<double>(expf(__fsub_rn(__fdiv_rn(lv[i], P.temp), mtr))) * scale
                                 : 0.0;
            const double r = p - static_cast<double>(qrow[q0 + i]);
            return r > 0.0 ? __double2ull_rn(r * two60) : 0ull;
        };
        const int32_t pk = chunk_pick(qm, rw2, s.t_int, s.sh64, &s.cand);
        cand = pk >= 0 ? id_of(a, q0 + pk) : -1;
    }
    if (tid == 0) {
        const uint64_t r = canon64(s_r);
        const uint4 w0 =
            make_uint4(r > 0ull ? 1u : 0u, cand >= 0 ? static_cast<uint32_t>(cand) : no_cand, lo32(r), hi32(r));
        const uint4 w1 = make_uint4(0u, 0u, 0u, 0u);
        uint32_t * own = ws<uint32_t>(a, wso.own_d);
        uint32_t * peer = reinterpret_cast<uint32_t *>(a.v.d_peer[par(a)]);
        *reinterpret_cast<uint4 *>(own) = w0;
        *reinterpret_cast<uint4 *>(own + 4) = w1;
        *reinterpret_cast<uint4 *>(peer) = w0;
        *reinterpret_cast<uint4 *>(peer + 4) = w1;
    }
}

// full path, decide2 (on a rejection): the residual's shard by R, or p itself when R is 0; the verdict
__global__ void __launch_bounds__(32) decide2_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    vstate_t & vs = *ws<vstate_t>(a, wso.vstate);
    if (threadIdx.x != 0 || vs.need == 0) {
        return;
    }
    const int32_t t = vs.row, card = a.v.card;
    uint32_t * slot = reinterpret_cast<uint32_t *>(a.v.d_own[par(a)]);
    const uint4 p0 = poll4(slot, a.v.err), p1 = poll4(slot + 4, a.v.err);
    (void)p1;
    fn::xchg::reset4(reinterpret_cast<uint4 *>(slot));
    fn::xchg::reset4(reinterpret_cast<uint4 *>(slot + 4));
    const uint32_t * ow = ws<uint32_t>(a, wso.own_d);
    const uint4 own = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    const uint4 r0 = card == 0 ? own : p0;
    const uint4 r1 = card == 0 ? p0 : own;
    const uint64_t R0 = join64(r0.z, r0.w), R1 = join64(r1.z, r1.w);
    rowres_t rr[max_rows] = {};
    const rowres_t * rs = ws<rowres_t>(a, wso.rowres);
    for (int32_t q = 0; q < a.rows; ++q) {
        rr[q] = rs[q];
    }
    if (R0 + R1 > 0ull) {
        const uint64_t target = scaled_target(row_uniform(*a.v.prm, row_position(a, t), p_resid_shard), R0 + R1);
        rr[t].resid = target < R0 ? static_cast<int32_t>(r0.y) : static_cast<int32_t>(r1.y);
    } else {
        rr[t].resid = rr[t].draw; // the residual has no mass (rounding): p itself
    }
    vs.need = 0;
    verdict(a, rr, false);
}

// greedy path, after V1's greedy_decide (which wrote desc.argmax and compared the records): verify's accepted run
// (a draft is accepted iff it equals the previous row's argmax), the round's tokens and the record
__global__ void __launch_bounds__(32) greedy_verdict_kernel(const kargs a) {
    if (skip(a)) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }
    rowres_t rr[max_rows] = {};
    fn::round::desc & d = *a.desc;
    for (int32_t t = 0; t < a.rows; ++t) {
        rr[t].draw = d.argmax[t];
        rr[t].accept = is_verify(a) && t < a.rows - 1 && d.tok[t + 1] == d.argmax[t] ? 1 : 0;
        rr[t].resid = d.argmax[t];
    }
    verdict(a, rr, false);
}

// the closing check: the record's words to the peer; then the comparison
__global__ void close_send_kernel(const card_view v) {
    if (threadIdx.x == 0) {
        send_header(v, v.x_peer);
    }
}
__global__ void close_recv_kernel(const card_view v) {
    if (threadIdx.x == 0) {
        check_header(v, v.x_own);
    }
}

// the host's operations on the window: a list of (token removed, ring slot, token restored or -1), applied in order
struct rewind_op {
    int32_t removed, slot, restored, pad;
};
__global__ void rewind_kernel(const card_view v, const rewind_op * ops, int32_t n_ops, int64_t window) {
    if (threadIdx.x != 0) {
        return;
    }
    for (int32_t i = 0; i < n_ops; ++i) {
        v.counts[ops[i].removed] -= 1u;
        if (ops[i].restored >= 0) {
            v.ring[ops[i].slot] = ops[i].restored;
            v.counts[ops[i].restored] += 1u;
        }
    }
    v.rec->window = window;
    v.rec->valid = 0u;
    set_words(*v.rec);
}

__global__ void discard_kernel(record * r) {
    if (threadIdx.x == 0) {
        r->valid = 0u;
        set_words(*r);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// launchers

bool al16(const void * p) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

bool launched() {
    return cudaPeekAtLastError() == cudaSuccess;
}

bool view_ok(const card_view & v) {
    return (v.card == 0 || v.card == 1) && al16(v.prm) && al16(v.bias) && al16(v.counts) && al16(v.ring) &&
           al16(v.rec) && al16(v.q_list) && al16(v.q_shard) && al16(v.work) && v.err != nullptr && al16(v.a_own[0]) &&
           al16(v.a_own[1]) && al16(v.a_peer[0]) && al16(v.a_peer[1]) && al16(v.b_own[0]) && al16(v.b_own[1]) &&
           al16(v.b_peer[0]) && al16(v.b_peer[1]) && al16(v.c_own[0]) && al16(v.c_own[1]) && al16(v.c_peer[0]) &&
           al16(v.c_peer[1]) && al16(v.d_own[0]) && al16(v.d_own[1]) && al16(v.d_peer[0]) && al16(v.d_peer[1]) &&
           al16(v.x_own) && al16(v.x_peer);
}

kargs make_args(const card_view & v, const decide_io & io, int32_t rows) {
    kargs a = {};
    a.v = v;
    a.logits = io.logits;
    a.n = io.n;
    a.id0 = io.id0;
    a.ids = io.ids;
    a.desc = io.desc;
    a.rows = rows;
    a.header = 1;
    return a;
}

bool io_ok(const decide_io & io, int32_t rows) {
    return rows >= 1 && rows <= max_rows && al16(io.logits) && io.n >= 128 && io.n <= n_card && io.n % 4 == 0 &&
           (io.ids != nullptr || (io.id0 >= 0 && io.id0 + io.n <= n_vocab)) && io.desc != nullptr;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// host

bool options::penalties() const {
    return penalty_last_n != 0 && !(penalty_repeat == 1.0f && penalty_freq == 0.0f && penalty_present == 0.0f);
}

bool options::check(int32_t n_ctx, std::string & err) const {
    std::string e;
    if (dry || top_n_sigma || typical || xtc || mirostat || grammar) {
        e += "DRY, top-n-sigma, typical-p, XTC, mirostat and grammar are not supported (SM0 section 9); ";
    }
    if (!(temp == temp) || !std::isfinite(temp)) {
        e += "temperature must be finite; ";
    }
    if (top_k < 0 || top_k > max_k) {
        e += "top-k must be 0 (off) or 1 .. 256; ";
    }
    if (!(top_p == top_p) || top_p > 1.0f) {
        e += "top-p must be at most 1; ";
    }
    if (!(min_p == min_p) || min_p > 1.0f) {
        e += "min-p must be at most 1; ";
    }
    if (!std::isfinite(penalty_repeat) || penalty_repeat <= 0.0f || !std::isfinite(1.0f / penalty_repeat)) {
        e += "penalty_repeat must be finite and greater than 0; ";
    }
    if (!std::isfinite(penalty_freq) || !std::isfinite(penalty_present)) {
        e += "penalty_freq and penalty_present must be finite; ";
    }
    if (penalty_last_n < -1 || (penalty_last_n == -1 && n_ctx <= 0) || penalty_last_n > n_ctx) {
        e += "penalty_last_n must be -1, 0 or 1 .. the context; ";
    }
    for (const auto & b : bias) {
        if (b.first < 0 || b.first >= n_vocab || b.second != b.second) {
            e += "a logit bias names a token outside the vocabulary, or is NaN; ";
            break;
        }
    }
    if (!e.empty()) {
        err = "sampling: " + e.substr(0, e.size() - 2);
        return false;
    }
    return true;
}

path path_of(const options & o) {
    if (o.temp <= 0.0f) {
        return o.transform() ? path::topk : path::greedy;
    }
    return o.top_k > 0 ? path::topk : path::full;
}

params make_params(const options & o, int32_t n_ctx) {
    params p = {};
    const bool greedy = o.temp <= 0.0f;
    p.temp = greedy ? 1.0f : o.temp;
    p.top_p = o.top_p;
    p.min_p = o.min_p;
    p.top_k = greedy ? 1 : o.top_k;
    p.rep = o.penalty_repeat;
    p.freq = o.penalty_freq;
    p.present = o.penalty_present;
    p.cap = o.penalties() ? (o.penalty_last_n < 0 ? n_ctx : o.penalty_last_n) : 0;
    p.flags = (o.bias.empty() ? 0u : f_bias) | (o.penalties() ? f_pen : 0u) | (greedy ? f_greedy : 0u) |
              (o.top_p < 1.0f ? f_topp : 0u) | (o.min_p > 0.0f ? f_minp : 0u);
    p.window_bins = hist_bins;
    p.seed_lo = static_cast<uint32_t>(o.seed);
    p.seed_hi = static_cast<uint32_t>(o.seed >> 32);
    p.req_lo = static_cast<uint32_t>(o.request);
    p.req_hi = static_cast<uint32_t>(o.request >> 32);
    p.eos = o.eos;
    return p;
}

size_t workspace_bytes() {
    return static_cast<size_t>(wso.total);
}

bool prepare(const card_view & v, void * stream) {
    if (!view_ok(v) || stream == nullptr) {
        return false;
    }
    return cudaMemsetAsync(v.work, 0, workspace_bytes(), static_cast<cudaStream_t>(stream)) == cudaSuccess;
}

std::vector<stage> stages_of(const options & o, bool verify) {
    std::vector<stage> s;
    if (o.penalties()) {
        s.push_back(stage::apply);
    }
    switch (path_of(o)) {
        case path::greedy:
            s.push_back(stage::greedy_verdict); // after V1's greedy_reduce and greedy_decide
            break;
        case path::topk:
            s.push_back(stage::select);
            s.push_back(stage::tk_decide);
            break;
        case path::full:
            if (o.top_p < 1.0f) {
                s.push_back(stage::max);
                s.push_back(stage::hist);
                s.push_back(stage::plan);
                s.push_back(stage::members);
                s.push_back(stage::resolve);
            } else if (o.min_p > 0.0f) {
                s.push_back(stage::max);
            }
            s.push_back(stage::settle);
            s.push_back(stage::decide);
            if (verify) {
                s.push_back(stage::resid);
                s.push_back(stage::decide2);
            }
            break;
    }
    return s;
}

bool enqueue(stage s, int32_t rows, const card_view & v, const decide_io & io, void * stream) {
    if (!view_ok(v) || stream == nullptr || rows < 1 || rows > max_rows || io.desc == nullptr) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    const kargs a = make_args(v, io, rows);
    const dim3 grid(n_chunk, static_cast<uint32_t>(rows));
    if (s != stage::apply && s != stage::greedy_verdict && s != stage::decide && s != stage::decide2 &&
        !io_ok(io, rows)) {
        return false;
    }
    switch (s) {
        case stage::apply:
            apply_kernel<<<1, 32, 0, st>>>(a);
            break;
        case stage::select:
            select_kernel<<<grid, n_thread, 0, st>>>(a);
            break;
        case stage::tk_decide:
            tk_decide_kernel<<<rows, n_thread, 0, st>>>(a);
            break;
        case stage::max:
            max_kernel<<<grid, n_thread, 0, st>>>(a);
            break;
        case stage::hist:
            hist_kernel<<<grid, n_thread, 0, st>>>(a);
            break;
        case stage::plan:
            plan_kernel<<<rows, n_thread, 0, st>>>(a);
            break;
        case stage::members:
            members_kernel<<<grid, n_thread, 0, st>>>(a);
            break;
        case stage::resolve:
            resolve_kernel<<<rows, n_thread, 0, st>>>(a);
            break;
        case stage::settle:
            settle_kernel<<<grid, n_thread, 0, st>>>(a);
            break;
        case stage::decide:
            decide_kernel<<<1, 32, 0, st>>>(a);
            break;
        case stage::resid:
            resid_kernel<<<n_chunk, n_thread, 0, st>>>(a);
            break;
        case stage::decide2:
            decide2_kernel<<<1, 32, 0, st>>>(a);
            break;
        case stage::greedy_verdict:
            greedy_verdict_kernel<<<1, 32, 0, st>>>(a);
            break;
    }
    return launched();
}

const uint32_t * greedy_hash_words(const card_view & v) {
    return v.rec->words;
}

uint32_t * greedy_checked(const card_view & v) {
    return &v.rec->checked;
}

bool close_send(const card_view & v, void * stream) {
    if (!view_ok(v) || stream == nullptr) {
        return false;
    }
    close_send_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(v);
    return launched();
}

bool close_recv(const card_view & v, void * stream) {
    if (!view_ok(v) || stream == nullptr) {
        return false;
    }
    close_recv_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(v);
    return launched();
}

bool pick_stage(const card_view & v, const pick_io & io, int32_t stage, void * stream) {
    if (!view_ok(v) || stream == nullptr || !al16(io.logits) || io.n < 128 || io.n > n_card || io.n % 4 != 0 ||
        (io.ids == nullptr && (io.id0 < 0 || io.id0 + io.n > n_vocab)) || io.t < 0 || io.t >= max_drafts ||
        io.proposal == nullptr || io.n_dtok < 0 || io.n_dtok > max_drafts || (io.n_dtok > 0 && io.dtok == nullptr) ||
        (stage != 0 && stage != 1)) {
        return false;
    }
    kargs a = {};
    a.v = v;
    a.logits = io.logits;
    a.n = io.n;
    a.id0 = io.id0;
    a.ids = io.ids;
    a.desc = nullptr;
    a.rows = 1;
    a.draft = 1;
    a.pick_pos = io.position;
    a.pick_desc = io.live;
    a.pick_off = io.live_offset;
    a.pick_t = io.t;
    a.dtok = io.dtok;
    a.n_dtok = io.n_dtok;
    a.proposal = io.proposal;
    a.header = 0;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    if (stage == 0) {
        select_kernel<<<dim3(n_chunk, 1), n_thread, 0, st>>>(a);
    } else {
        tk_decide_kernel<<<1, n_thread, 0, st>>>(a);
    }
    return launched();
}

bool pick(const card_view & v, const pick_io & io, void * stream) {
    return pick_stage(v, io, 0, stream) && pick_stage(v, io, 1, stream);
}

bool pick_lanes_stage(const card_view * const v[], const pick_io io[], int32_t lanes, int32_t stage, void * stream) {
    if (v == nullptr || io == nullptr || lanes < 1 || lanes > max_rows) {
        return false;
    }
    for (int32_t k = 0; k < lanes; ++k) {
        if (v[k] == nullptr) {
            return false;
        }
        card_view w = *v[k];
        for (int32_t q = 0; q < 2; ++q) {
            w.a_own[q] = v[k]->a_own[q] + int64_t(k) * a_row;
            w.a_peer[q] = v[k]->a_peer[q] + int64_t(k) * a_row;
        }
        if (!pick_stage(w, io[k], stage, stream)) {
            return false;
        }
    }
    return true;
}

bool read_list(const card_view & v, int32_t row, std::vector<uint32_t> & words, void * stream) {
    if (!view_ok(v) || stream == nullptr || row < 0 || row >= max_rows) {
        return false;
    }
    words.assign(4 + 2 * max_k, 0u);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    return cudaMemcpyAsync(words.data(), v.work + wso.own_a + 4 * (hdr + int64_t(row) * a_row), words.size() * 4,
                           cudaMemcpyDeviceToHost, st) == cudaSuccess &&
           cudaStreamSynchronize(st) == cudaSuccess;
}

uint64_t decision_hash(uint64_t round, int32_t n, const int32_t * tok, int32_t a) {
    return hash_of(round, n, tok, a);
}

bool write_params(const card_view & v, const params & p, const options & o, void * stream) {
    if (!view_ok(v) || stream == nullptr) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    std::vector<float> table(n_vocab, 0.0f);
    for (const auto & b : o.bias) {
        table[size_t(b.first)] += b.second;
    }
    return cudaMemcpyAsync(v.prm, &p, sizeof(p), cudaMemcpyHostToDevice, st) == cudaSuccess &&
           cudaMemcpyAsync(v.bias, table.data(), table.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
           cudaStreamSynchronize(st) == cudaSuccess;
}

bool load_window(const card_view & v, int32_t cap, const std::vector<int32_t> & ledger, void * stream) {
    if (!view_ok(v) || stream == nullptr || cap < 0) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    std::vector<uint32_t> counts(n_vocab, 0u);
    const int64_t L = static_cast<int64_t>(ledger.size());
    std::vector<int32_t> ring(size_t(cap > 0 ? cap : 1), 0);
    for (int64_t pos = cap > 0 ? std::max<int64_t>(0, L - cap) : L; pos < L; ++pos) {
        const int32_t tk = ledger[size_t(pos)];
        if (tk < 0 || tk >= n_vocab) {
            return false;
        }
        counts[size_t(tk)] += 1u;
        ring[size_t(pos % cap)] = tk;
    }
    record r = {};
    r.window = cap > 0 ? L : 0;
    r.valid = 0u;
    r.applied = 1u;
    for (int32_t i = 0; i < max_rows; ++i) {
        r.tok[i] = -1;
    }
    set_words(r);
    return cudaMemcpyAsync(v.counts, counts.data(), counts.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
           (cap == 0 ||
            cudaMemcpyAsync(v.ring, ring.data(), size_t(cap) * 4, cudaMemcpyHostToDevice, st) == cudaSuccess) &&
           cudaMemcpyAsync(v.rec, &r, sizeof(r), cudaMemcpyHostToDevice, st) == cudaSuccess &&
           cudaStreamSynchronize(st) == cudaSuccess;
}

bool discard_record(const card_view & v, void * stream) {
    if (!view_ok(v) || stream == nullptr) {
        return false;
    }
    discard_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(v.rec);
    return launched();
}

bool read_record(const card_view & v, record & out, void * stream) {
    if (!view_ok(v) || stream == nullptr) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    return cudaMemcpyAsync(&out, v.rec, sizeof(out), cudaMemcpyDeviceToHost, st) == cudaSuccess &&
           cudaStreamSynchronize(st) == cudaSuccess;
}

bool rewind(const card_view & v, int32_t cap, const std::vector<int32_t> & ledger_before, int64_t n_keep,
            void * stream) {
    if (!view_ok(v) || stream == nullptr || cap <= 0 || n_keep < 0) {
        return false;
    }
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    // the record's tokens join the tables first (apply), so the window is the ledger's first L tokens
    kargs a = {};
    a.v = v;
    apply_kernel<<<1, 32, 0, st>>>(a);
    record r = {};
    if (!launched() || !read_record(v, r, stream)) {
        return false;
    }
    const int64_t L = r.window;
    if (n_keep > L || L > static_cast<int64_t>(ledger_before.size())) {
        return false;
    }
    std::vector<rewind_op> ops;
    for (int64_t pos = L - 1; pos >= n_keep; --pos) {
        rewind_op op = {ledger_before[size_t(pos)], static_cast<int32_t>(pos % cap), -1, 0};
        if (pos >= cap) {
            op.restored = ledger_before[size_t(pos - cap)];
        }
        ops.push_back(op);
    }
    rewind_op * d_ops = nullptr;
    const size_t nb = std::max<size_t>(1, ops.size()) * sizeof(rewind_op);
    bool ok = cudaMalloc(&d_ops, nb) == cudaSuccess &&
              (ops.empty() || cudaMemcpyAsync(d_ops, ops.data(), ops.size() * sizeof(rewind_op), cudaMemcpyHostToDevice,
                                              st) == cudaSuccess);
    if (ok) {
        rewind_kernel<<<1, 32, 0, st>>>(v, d_ops, static_cast<int32_t>(ops.size()), n_keep);
        ok = launched() && cudaStreamSynchronize(st) == cudaSuccess;
    }
    cudaFree(d_ops);
    return ok;
}

} // namespace fn::samp
