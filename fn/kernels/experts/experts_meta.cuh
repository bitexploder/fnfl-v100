#pragma once

// XB2: the decode experts' routing meta and its unit order (XB2 (d)). XB3 part 1: every gate_up block builds the meta
// of its launch's IDs in its prologue (build_meta_of, 71b92a9's build_meta) and its own ranks of the order
// (experts_gu.cuh block_units); block 0 writes the struct to the region and each block its ranks' order entries, and
// down reads the struct at entry instead of building it (experts.cu). write_meta (experts.cu) runs route_meta, the same
// builders in a one-block launch, for the tests and benches.
//
// The region (meta_bytes, 16-byte aligned, rewritten every layer): the meta struct, then the order from meta_order.
// Its publication is gate_up's launch end; the down launch that follows it on the stream reads it with plain loads.

#include "fn/kernels/experts/experts.h"
#include "fn/model/shapes.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fn::experts {

// the distinct experts of a launch's B rows, first-appearance order; 16-byte aligned and sized, so it copies as words.
// SP2: C is the routing capacity (experts.h cap_of): a launch of B rows uses meta_c<cap_of(B)>, 40 slots for B <= 4
// (the struct those launches had before SP2), 50 at B = 5.
template <int C> struct alignas(16) meta_c {
    static constexpr int32_t slots = fn::n_expert_used * C;
    int32_t n_u;          // U
    int32_t n_slots;      // routed (expert, row) pairs
    uint32_t bad;         // an ID outside [0, 512)
    int32_t e[slots];     // expert of u
    uint32_t mask[slots]; // rows of u (bit b)
    int32_t slot_base[slots];
    int32_t u_of[slots]; // u of entry p = 10 b + k
};
using meta = meta_c<4>;
template <int B> using meta_of = meta_c<cap_of(B)>;

// 16-byte words of the struct
template <int C> constexpr int32_t meta_words_c = static_cast<int32_t>(sizeof(meta_c<C>) / 16);
constexpr int32_t meta_words = meta_words_c<4>;
template <int C> constexpr bool meta_view_ok() {
    constexpr meta_view v = meta_view_of(C);
    return offsetof(meta_c<C>, n_u) == v.n_u * 4 && offsetof(meta_c<C>, n_slots) == v.n_slots * 4 &&
           offsetof(meta_c<C>, bad) == v.bad * 4 && offsetof(meta_c<C>, e) == v.e * 4 &&
           offsetof(meta_c<C>, mask) == v.mask * 4 && offsetof(meta_c<C>, slot_base) == v.slot_base * 4 &&
           offsetof(meta_c<C>, u_of) == v.u_of * 4 && sizeof(meta_c<C>) == size_t(v.order) &&
           meta_c<C>::slots == v.slots;
}
static_assert(meta_words_c<5> * 16 <= meta_bytes && meta_bytes % 16 == 0, "the meta region holds the struct");
static_assert(meta_view_ok<4>() && meta_view_ok<5>(), "experts.h's host view of the region");

// Warp 0 builds the meta of 10 B IDs (B <= 5, so at most 50 entries, two a lane); id(i) reads ID i. First-appearance
// order: entry p is the first of its ID when no entry q < p has it; u is the rank among firsts. __match_any_sync finds
// equal IDs. m is in shared memory (its masks are built by atomicOr).
template <int B, typename ID> __device__ __forceinline__ void build_meta_of(const ID & id, meta_of<B> & m) {
    constexpr int32_t max_slots = meta_of<B>::slots; // the capacity's (SP2), so B <= 4 keep their code
    constexpr int32_t n = fn::n_expert_used * B;
    const uint32_t full = 0xffffffffu;
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    const uint32_t lt = (1u << lane) - 1u;
    const int32_t v0 = lane < n ? id(lane) : -1 - lane;
    const int32_t v1 = lane + 32 < n ? id(lane + 32) : -33 - lane;
    const bool bad0 = lane < n && (v0 < 0 || v0 >= fn::n_expert);
    const bool bad1 = lane + 32 < n && (v1 < 0 || v1 >= fn::n_expert);
    const uint32_t bad = __ballot_sync(full, bad0 || bad1);
    const uint32_t mm0 = __match_any_sync(full, v0);
    const uint32_t mm1 = __match_any_sync(full, v1);
    const bool first0 = lane < n && (mm0 & lt) == 0;
    bool first1 = lane + 32 < n && (mm1 & lt) == 0;
    int32_t cross = -1; // entry 32 + lane equals entry cross < 32
    if (n > 32) {
#pragma unroll
        for (int32_t l = 0; l < n - 32; ++l) {
            const int32_t x = __shfl_sync(full, v1, l);
            const uint32_t hit = __ballot_sync(full, lane < n && v0 == x);
            if (lane == l && hit != 0) {
                first1 = false;
                cross = __ffs(hit) - 1;
            }
        }
    }
    const uint32_t f0 = __ballot_sync(full, first0);
    const uint32_t f1 = __ballot_sync(full, first1);
    const int32_t r0 = __popc(f0 & lt);
    const int32_t r1 = __popc(f0) + __popc(f1 & lt);
    const int32_t u0 = __shfl_sync(full, r0, __ffs(mm0) - 1);
    const int32_t u1a = __shfl_sync(full, r0, cross < 0 ? 0 : cross);
    const int32_t u1b = __shfl_sync(full, r1, __ffs(mm1) - 1);
    const int32_t u1 = cross >= 0 ? u1a : u1b;
    if (lane < max_slots) {
        m.mask[lane] = 0;
    }
    if (lane + 32 < max_slots) {
        m.mask[lane + 32] = 0;
    }
    __syncwarp();
    if (first0) {
        m.e[r0] = v0;
    }
    if (first1) {
        m.e[r1] = v1;
    }
    if (lane < n) {
        m.u_of[lane] = u0;
        atomicOr(&m.mask[u0], 1u << (lane / fn::n_expert_used)); // integer OR: the result does not depend on the order
    }
    if (lane + 32 < n) {
        m.u_of[lane + 32] = u1;
        atomicOr(&m.mask[u1], 1u << ((lane + 32) / fn::n_expert_used));
    }
    __syncwarp();
    const int32_t c0 = lane < max_slots ? __popc(m.mask[lane]) : 0;
    const int32_t c1 = lane + 32 < max_slots ? __popc(m.mask[lane + 32]) : 0;
    int32_t s0 = c0, s1 = c1; // inclusive scans
#pragma unroll
    for (int32_t off = 1; off < 32; off <<= 1) {
        const int32_t t0 = __shfl_up_sync(full, s0, off);
        const int32_t t1 = __shfl_up_sync(full, s1, off);
        if (lane >= off) {
            s0 += t0;
            s1 += t1;
        }
    }
    const int32_t tot0 = __shfl_sync(full, s0, 31);
    if (lane < max_slots) {
        m.slot_base[lane] = s0 - c0;
    }
    if (lane + 32 < max_slots) {
        m.slot_base[lane + 32] = tot0 + s1 - c1;
    }
    const int32_t tot1 = __shfl_sync(full, s1, 31);
    if (lane == 0) {
        m.n_u = __popc(f0) + __popc(f1);
        m.n_slots = tot0 + tot1;
        m.bad = bad != 0 ? 1u : 0u;
    }
}

static_assert(sizeof(meta) == meta_order, "the order follows the struct (experts.h)");

// SP2: the classes at capacity C: C routed classes (class a < C: a unit of a + 1 active rows) and the shared class C,
// so capacity 4 has experts.h's five (shared at 4) and capacity 5 six; a class's weight is cw.w[a] for a < 4, the
// shared class's cw.w[4], the 5-row class's class_w5 (experts.h)
template <int C> __device__ __forceinline__ float class_w(const class_table & cw, int32_t a) {
    return C == 5 && a == C ? cw.w[4] : C == 5 && a == 4 ? 2.0f * cw.w[3] - cw.w[2] : cw.w[a]; // class_w5
}

// XB2 (d): class a (0..C-1: a routed unit of a + 1 active rows, C: shared) ranks before class b: the heavier first,
// ties to the class with more rows, the shared class after a routed one (experts.h)
template <int C> __device__ __forceinline__ bool class_before(const class_table & cw, int32_t a, int32_t b) {
    const int32_t ta = a == C ? 0 : a + 1, tb = b == C ? 0 : b + 1;
    const float wa = class_w<C>(cw, a), wb = class_w<C>(cw, b);
    return wa > wb || (wa == wb && ta > tb);
}

// Warp 0 writes the unit order of m (built) into order (shared memory, U16 [10 U + 10]): each class's units
// contiguous, the classes by class_before, u-major then g within a class. Each expert u (lane, lane + 32) gets its
// class's start plus 10 times its rank among the class's experts below it (a ballot), then writes its 10 entries.
template <int C>
__device__ __forceinline__ void build_order(const meta_c<C> & m, const class_table & cw, uint16_t * order) {
    constexpr int32_t n_classes = C + 1;                 // SP2: the capacity's
    constexpr int32_t order_shared = order_shared_of(C); // SP2: the capacity's
    const uint32_t full = 0xffffffffu;
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    const uint32_t lt = (1u << lane) - 1u;
    const int32_t n_u = m.n_u;
    const int32_t c0 = lane < n_u ? __popc(m.mask[lane]) - 1 : -1;
    const int32_t c1 = lane + 32 < n_u ? __popc(m.mask[lane + 32]) - 1 : -1;
    int32_t size[n_classes]; // units of each class
    int32_t r0 = 0, r1 = 0;  // the experts' ranks in their classes
#pragma unroll
    for (int32_t c = 0; c < n_classes - 1; ++c) {
        const uint32_t b0 = __ballot_sync(full, c0 == c);
        const uint32_t b1 = __ballot_sync(full, c1 == c);
        size[c] = unit_groups * (__popc(b0) + __popc(b1));
        r0 = c0 == c ? __popc(b0 & lt) : r0;
        r1 = c1 == c ? __popc(b0) + __popc(b1 & lt) : r1;
    }
    size[n_classes - 1] = unit_groups;
    int32_t start[n_classes];
#pragma unroll
    for (int32_t a = 0; a < n_classes; ++a) {
        start[a] = 0;
#pragma unroll
        for (int32_t b = 0; b < n_classes; ++b) {
            start[a] += b != a && class_before<C>(cw, b, a) ? size[b] : 0;
        }
    }
    int32_t s0 = 0, s1 = 0; // class starts of the lane's experts (dynamic indexing would put start in local memory)
#pragma unroll
    for (int32_t c = 0; c < n_classes - 1; ++c) {
        s0 = c0 == c ? start[c] : s0;
        s1 = c1 == c ? start[c] : s1;
    }
    if (c0 >= 0) {
#pragma unroll
        for (int32_t g = 0; g < unit_groups; ++g) {
            order[s0 + unit_groups * r0 + g] = static_cast<uint16_t>(unit_groups * lane + g);
        }
    }
    if (c1 >= 0) {
#pragma unroll
        for (int32_t g = 0; g < unit_groups; ++g) {
            order[s1 + unit_groups * r1 + g] = static_cast<uint16_t>(unit_groups * (lane + 32) + g);
        }
    }
    if (lane < unit_groups) {
        order[start[n_classes - 1] + lane] = static_cast<uint16_t>(order_shared + lane);
    }
}

// write_meta's builder (XB2's router form), warp 0 of a block: the meta of ids_sm ([B][10] in shared memory) and its
// unit order by cw, built in scratch (shared memory, meta_bytes, 16-byte aligned), then written to out (global,
// meta_bytes, 16-byte aligned) as 16-byte stores, the struct and the order's 10 U + 10 entries. Out of line: one copy
// a kernel, its registers apart from the caller's (R17).
template <int B>
__device__ __noinline__ void route_meta(const int32_t * ids_sm, uint8_t * scratch, uint8_t * out, class_table cw) {
    constexpr int32_t C = cap_of(B), meta_order = meta_order_of(C);
    meta_c<C> & m = *reinterpret_cast<meta_c<C> *>(scratch);
    build_meta_of<B>([&](int32_t i) { return ids_sm[i]; }, m);
    __syncwarp();
    build_order<C>(m, cw, reinterpret_cast<uint16_t *>(scratch + meta_order));
    __syncwarp();
    const int32_t lane = static_cast<int32_t>(threadIdx.x & 31);
    const int32_t n_words = (meta_order + 2 * (unit_groups * m.n_u + unit_groups) + 15) / 16;
    const uint4 * src = reinterpret_cast<const uint4 *>(scratch);
    uint4 * dst = reinterpret_cast<uint4 *>(out);
#pragma unroll 1
    for (int32_t i = lane; i < n_words; i += 32) {
        dst[i] = src[i];
    }
}

// a reader's copy of the region's struct: words i = lane, lane + step, ... of meta_words_c<C>, global to shared memory
template <int C>
__device__ __forceinline__ void load_meta(const uint8_t * src, meta_c<C> & m, int32_t i, int32_t step) {
    const uint4 * s = reinterpret_cast<const uint4 *>(src);
    uint4 * d = reinterpret_cast<uint4 *>(&m);
#pragma unroll 1
    for (; i < meta_words_c<C>; i += step) {
        d[i] = __ldg(s + i);
    }
}

} // namespace fn::experts
