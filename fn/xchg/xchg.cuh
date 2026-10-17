#pragma once

// fn/xchg device side: the A3 sends, polls and resets (the protocol and its four steps are in xchg.h).

#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::xchg {

// polls of one spin-wait before it gives up and reports (at about 0.2-1 us a poll, 0.2-1 s)
constexpr uint32_t poll_budget = 1u << 20;

// The payload rule: every NaN is sent as the canonical NaN, so no payload word equals the sentinel.
__device__ __forceinline__ uint32_t canonical(float v) {
    return v != v ? canonical_nan : __float_as_uint(v);
}

__device__ __forceinline__ bool landed(uint4 v, uint32_t sent = sentinel) {
    return v.x != sent && v.y != sent && v.z != sent && v.w != sent;
}

// one poll: a volatile 16-byte load (bypasses L1, so a later poll sees the peer's store)
__device__ __forceinline__ uint4 load_poll(const uint4 * p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

// Polls the N 16-byte words p[i] until none holds the sentinel sent; v[i] receives them. Each pass reloads only the
// words still pending. After poll_budget passes it ORs code into *err and returns with v holding what it last saw.
template <int N>
__device__ __forceinline__ void poll(const uint4 * const (&p)[N], uint4 (&v)[N], uint32_t * err, uint32_t code,
                                     uint32_t sent = sentinel) {
    uint32_t pending = (1u << N) - 1u;
    for (uint32_t n = 0;; ++n) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (pending >> i & 1u) {
                v[i] = load_poll(p[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (landed(v[i], sent)) {
                pending &= ~(1u << i);
            }
        }
        if (pending == 0u) {
            return;
        }
        if (n >= poll_budget) {
            atomicOr(err, code);
            return;
        }
    }
}

// one poll of an 8-byte word pair (T-GREEDY), as load_poll
__device__ __forceinline__ uint2 load_poll2(const uint2 * p) {
    uint2 v;
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
    return v;
}

// Polls the 8-byte pair *p until neither word holds the sentinel, as poll; returns what it last saw.
__device__ __forceinline__ uint2 poll2(const uint2 * p, uint32_t * err, uint32_t code) {
    for (uint32_t n = 0;; ++n) {
        const uint2 v = load_poll2(p);
        if (v.x != sentinel && v.y != sentinel) {
            return v;
        }
        if (n >= poll_budget) {
            atomicOr(err, code);
            return v;
        }
    }
}

__device__ __forceinline__ void reset2(uint2 * p) {
    *p = make_uint2(sentinel, sentinel);
}

__device__ __forceinline__ void reset4(uint4 * p, uint32_t sent = sentinel) {
    *p = make_uint4(sent, sent, sent, sent);
}

// Warp-collective send of a run of len words (len <= 28) starting at word index base of a 16-byte aligned payload:
// lane lane0 + i holds word base + i (as canonical bits). Lane q stores the words of the payload's 16-byte group q of
// the run: one 16-byte store for a whole group, an 8- or 4-byte store at the run's unaligned ends, to the peer slot and
// (if local is not null) to this card's own copy. Every lane of the warp must call it with the same base, len and lane0.
__device__ __forceinline__ void put_run(uint32_t * local, uint32_t * peer, int32_t base, int32_t len, int32_t lane0,
                                        uint32_t bits) {
    const int32_t q = threadIdx.x % 32;
    const int32_t off = base & 3;
    uint32_t v[4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        const int32_t src = lane0 + 4 * q + i - off;
        v[i] = __shfl_sync(0xffffffffu, bits, src < 0 ? 0 : src > 31 ? 31 : src);
    }
    const int32_t lo = max(4 * q, off) - 4 * q; // the run's words in group q: lo .. hi-1
    const int32_t hi = min(4 * q + 4, off + len) - 4 * q;
    if (lo >= hi) {
        return;
    }
    const int32_t at = base - off + 4 * q; // the group's first word
    if (lo == 0 && hi == 4) {
        const uint4 w = make_uint4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<uint4 *>(peer + at) = w; // 16-byte aligned: at % 4 == 0 and the payload is aligned
        if (local != nullptr) {
            *reinterpret_cast<uint4 *>(local + at) = w;
        }
    } else if (hi - lo == 2 && (lo & 1) == 0) {
        const uint2 w = lo == 0 ? make_uint2(v[0], v[1]) : make_uint2(v[2], v[3]); // constant indices: no stack
        *reinterpret_cast<uint2 *>(peer + at + lo) = w; // 8-byte aligned: at + lo is even
        if (local != nullptr) {
            *reinterpret_cast<uint2 *>(local + at + lo) = w;
        }
    } else {
#pragma unroll
        for (int32_t i = 0; i < 4; ++i) {
            if (i >= lo && i < hi) {
                peer[at + i] = v[i];
                if (local != nullptr) {
                    local[at + i] = v[i];
                }
            }
        }
    }
}

// X-HC: a block's finished moments of stream s, rows j0 .. j0+n-1 of the 324 (u rows 0..319, then q rows 320..323),
// lane i holding row j0 + i, for one payload row (local_row and peer_row point at its word 0). Warp-collective.
__device__ __forceinline__ void put_hc_rows(uint32_t * local_row, uint32_t * peer_row, int32_t s, int32_t j0, int32_t n,
                                            uint32_t bits) {
    constexpr int32_t n_u = 320, u0 = 4, q0 = 4 + 4 * n_u; // the payload layout (xchg.h)
    const int32_t n_lo = j0 >= n_u ? 0 : min(n, n_u - j0);  // rows in the u part
    if (n_lo > 0) {
        put_run(local_row, peer_row, u0 + n_u * s + j0, n_lo, 0, bits);
    }
    if (n_lo < n) {
        put_run(local_row, peer_row, q0 + 4 * s + (j0 + n_lo - n_u), n - n_lo, n_lo, bits);
    }
}

// ---- X-HMIX (HC3): the mixed row in both forms; the slot's regions are xchg.h's hmix_layout ----

// Release (step 4): resets 16-byte group u (0 .. hmix_slot_u4 - 1) of a slot, both halves, to the sentinels (code
// words to code_sentinel). The resetter (hc_down) spreads u over its threads. The own half needs no reset for the
// protocol (it is never polled); resetting the whole slot keeps the resetter free of the card index.
constexpr int32_t hmix_slot_u4 = hmix_layout::slot_words / 4; // 3,360
__device__ __forceinline__ void hmix_reset_slot(float * slot, int32_t u) {
    uint4 * s4 = reinterpret_cast<uint4 *>(slot);
    if (u < hmix_layout::code_at / 4) {
        reset4(s4 + u);
    } else if (u < hmix_layout::scale_at / 4) {
        reset4(s4 + u, code_sentinel);
    } else if (u < hmix_slot_u4) {
        reset4(s4 + u);
    }
}

// Writes (step 1) of one run's codes and scales, row b, run j (0 .. 79) of the row, from the 8 lanes 8m .. 8m + 7 that
// hold it as act::quantize_run8 left it (lane i: codes 4i .. 4i + 3 packed, value 0 in the low byte; ds the same in
// all 8): lanes i % 4 == 0 gather four packed words and store 16 bytes; lane i == 0 stores {d, s} as canonical bits.
// To the peer's slot and (own not null) this card's own. Warp-collective: every lane of the warp calls it (the shuffles name the full
// mask); lanes whose run is not real pass real == false and store nothing.
__device__ __forceinline__ void hmix_put_run(float * own, float * peer, int32_t b, int32_t j, uint32_t packed,
                                             float2 ds, bool real) {
    const int32_t lane = static_cast<int32_t>(threadIdx.x % 32);
    uint32_t w[4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        w[i] = __shfl_sync(0xffffffffu, packed, (lane & ~3) + i);
    }
    if (!real) {
        return;
    }
    const int32_t i = lane & 7;
    if ((i & 3) == 0) {
        const int32_t at = hmix_layout::code_at + (b * hmix_layout::width + 32 * j) / 4 + i; // a code word, 16-aligned
        const uint4 v = make_uint4(w[0], w[1], w[2], w[3]);
        *reinterpret_cast<uint4 *>(reinterpret_cast<uint32_t *>(peer) + at) = v;
        if (own != nullptr) {
            *reinterpret_cast<uint4 *>(reinterpret_cast<uint32_t *>(own) + at) = v;
        }
    }
    if (i == 0) {
        const int32_t at = hmix_layout::scale_at + 2 * (b * hmix_layout::runs + j);
        const uint2 v = make_uint2(canonical(ds.x), canonical(ds.y));
        *reinterpret_cast<uint2 *>(reinterpret_cast<uint32_t *>(peer) + at) = v;
        if (own != nullptr) {
            *reinterpret_cast<uint2 *>(reinterpret_cast<uint32_t *>(own) + at) = v;
        }
    }
}

// Acquire (step 3) of run j of row b's codes and scales from this card's slot: c0, c1 the run's 32 codes, sc its
// {d, s}. Polls (volatile) when the run is in the peer's half, else plain loads of this card's own half (written by
// this card's hc_up, which stream order completed). err: err_poll_hmix after poll_budget passes.
__device__ __forceinline__ void hmix_run(const float * slot, int32_t peer_half, int32_t b, int32_t j, uint4 & c0,
                                         uint4 & c1, float2 & sc, uint32_t * err) {
    const uint4 * cp =
        reinterpret_cast<const uint4 *>(slot + hmix_layout::code_at) + (b * hmix_layout::width + 32 * j) / 16;
    const uint2 * sp = reinterpret_cast<const uint2 *>(slot + hmix_layout::scale_at) + b * hmix_layout::runs + j;
    if (j / (words_hmix / 32) == peer_half) {
        const uint4 * const pc[2] = {cp, cp + 1};
        uint4 v[2];
        poll<2>(pc, v, err, err_poll_hmix, code_sentinel);
        c0 = v[0];
        c1 = v[1];
        const uint2 s = poll2(sp, err, err_poll_hmix);
        sc = make_float2(__uint_as_float(s.x), __uint_as_float(s.y));
    } else {
        c0 = __ldg(cp);
        c1 = __ldg(cp + 1);
        const uint2 s = __ldg(sp);
        sc = make_float2(__uint_as_float(s.x), __uint_as_float(s.y));
    }
}

// Acquire (step 3) of the peer half's F32 words of rows 0 .. rows-1, for a launch that reads the F32 row through L1
// after its block barrier: thread t of n polls 16-byte groups t, t + n, ... and discards them.
__device__ __forceinline__ void hmix_acquire_f32(const float * slot, int32_t peer_half, int32_t rows, int32_t t,
                                                 int32_t n, uint32_t * err) {
    const uint4 * s4 = reinterpret_cast<const uint4 *>(slot);
    for (int32_t u = t; u < rows * (words_hmix / 4); u += n) {
        const int32_t b = u / (words_hmix / 4), i = u - b * (words_hmix / 4);
        const uint4 * const p[1] = {s4 + (b * hmix_layout::width + peer_half * words_hmix) / 4 + i};
        uint4 v[1];
        poll<1>(p, v, err, err_poll_hmix);
    }
}

} // namespace fn::xchg
