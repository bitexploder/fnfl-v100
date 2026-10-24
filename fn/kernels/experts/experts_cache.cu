#include "fn/kernels/experts/experts_cache.h"

#include "fn/kernels/experts/experts.h"

#include <cuda_runtime.h>

namespace fn::experts {

namespace {

constexpr int32_t n_ex = 512;
constexpr int32_t n_used = 10;
constexpr int32_t resolve_threads = 64;

// resolve: thread p < 10 B takes routing entry p. Every thread of an expert writes the same values to its table
// entries, so repeated experts store equal words. The flags and the two words follow a block barrier; the any-miss
// test is __syncthreads_or over every entry. Registers: a handful; no shared memory.
__global__ void __launch_bounds__(resolve_threads, 1) resolve_kernel(resolve_args a) {
    const int32_t p = static_cast<int32_t>(threadIdx.x);
    bool miss = false;
    if (p < n_used * a.rows) {
        const int32_t e = a.ids[p];
        if (e >= 0 && e < n_ex) {
            const uint8_t * const r = a.res[e];
            miss = r == nullptr;
#pragma unroll
            for (int32_t k = 0; k < 4; ++k) { // unrolled: off[k] stays in registers
                if (k >= a.parts) {
                    break;
                }
                const bool down = (k & 1) != 0;
                a.tab[k * n_ex + e] = r != nullptr ? r + a.off[k] : down ? a.zero : nullptr;
                a.tab2[k * n_ex + e] = nullptr;
            }
        } else {
            miss = true; // gate_up flags the bad ID; the host's plan skips it
        }
    }
    const int32_t any = __syncthreads_or(miss ? 1 : 0);
    if (p == 0) {
        *a.ext_slots = 0;
        *a.run2 = 0;
        // ME3: the flags' one value (set) when every expert is resident, else 0 for the service to set: the same
        // words each replay of a captured window
        const uint32_t f = any == 0 ? a.set : 0u;
        a.flags[0] = f;
        a.flags[1] = f;
    }
}

// apply: one block; thread i takes pairs i, i + 256, ... (indices are distinct within a call, so the order of the
// stores does not matter)
constexpr int32_t apply_threads = 256;
__global__ void __launch_bounds__(apply_threads, 1) apply_kernel(uint64_t * words, const int32_t * idx,
                                                                 const uint64_t * val, int32_t n) {
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < n; i += apply_threads) {
        words[idx[i]] = val[i];
    }
}

// combine: thread n of the 2,560 columns (ten blocks of 256) takes every row's column n; the slot word and the meta's
// words are the same for every thread (uniform loads). Per (b, k) in route order: the slot of the entry's expert u for
// row b (experts_dev.cuh slot_of: slot_base[u] + the rows of u before b), then one fmaf when its bit is set; the row
// is written only when it took a term. Contract: card 0's stream, after down (out complete) and the down flag (ext_ds
// and the word written by the service before it). Registers: a handful; no shared memory.
constexpr int32_t combine_threads = 256;
constexpr int32_t combine_d = 2560;
static_assert(combine_d % combine_threads == 0, "whole blocks over the columns");
__global__ void __launch_bounds__(combine_threads, 1) combine_kernel(combine_args a) {
    const uint64_t ext = *a.ext_slots;
    const int32_t * const mw = reinterpret_cast<const int32_t *>(a.meta);
    const meta_view v = meta_view_of(a.rows);
    if (ext == 0 || mw[v.bad] != 0) {
        return;
    }
    const int32_t n = static_cast<int32_t>(blockIdx.x) * combine_threads + static_cast<int32_t>(threadIdx.x);
    for (int32_t b = 0; b < a.rows; ++b) {
        float acc = a.out[b * combine_d + n];
        bool took = false;
        for (int32_t k = 0; k < n_used; ++k) {
            const int32_t u = mw[v.u_of + b * n_used + k];
            const int32_t slot = mw[v.slot_base + u] + __popc(static_cast<uint32_t>(mw[v.mask + u]) & ((1u << b) - 1u));
            if (((ext >> slot) & 1u) != 0) {
                acc = fmaf(a.wts[b * n_used + k], a.ext_ds[int64_t(slot) * combine_d + n], acc);
                took = true;
            }
        }
        if (took) {
            a.out[b * combine_d + n] = acc;
        }
    }
}

// ME10: combine's terms, then each half as the exchange's recv_add makes it (own + received, the received operand
// canonicalized as send_half sends it). Contract: card 0's stream on the one device, after both downs and the down
// flag; no other reader of p0 after it. Registers: a handful; no shared memory.
constexpr int32_t finish_half = combine_d / 2;
__device__ __forceinline__ float finish_canon(float v) {
    return v != v ? __uint_as_float(0x7fc00000u) : v; // xchg.h canonical_nan
}
__global__ void __launch_bounds__(combine_threads, 1) finish_kernel(combine_args a, const float * p1, float * h0,
                                                                     float * h1) {
    const uint64_t ext = *a.ext_slots;
    const int32_t * const mw = reinterpret_cast<const int32_t *>(a.meta);
    const meta_view v = meta_view_of(a.rows);
    const bool terms = ext != 0 && mw[v.bad] == 0;
    const int32_t n = static_cast<int32_t>(blockIdx.x) * combine_threads + static_cast<int32_t>(threadIdx.x);
    for (int32_t b = 0; b < a.rows; ++b) {
        float acc = a.out[b * combine_d + n];
        for (int32_t k = 0; terms && k < n_used; ++k) {
            const int32_t u = mw[v.u_of + b * n_used + k];
            const int32_t slot = mw[v.slot_base + u] + __popc(static_cast<uint32_t>(mw[v.mask + u]) & ((1u << b) - 1u));
            if (((ext >> slot) & 1u) != 0) {
                acc = fmaf(a.wts[b * n_used + k], a.ext_ds[int64_t(slot) * combine_d + n], acc);
            }
        }
        const float q = p1[b * combine_d + n];
        if (n < finish_half) {
            h0[b * finish_half + n] = __fadd_rn(acc, finish_canon(q));
        } else {
            h1[b * finish_half + n - finish_half] = __fadd_rn(q, finish_canon(acc));
        }
    }
}

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void __launch_bounds__(32, 1) wait_kernel(wait_args a) {
    if (threadIdx.x != 0) {
        return;
    }
    const uint64_t t0 = globaltimer();
    const uint32_t target = a.need != nullptr ? *reinterpret_cast<const volatile uint32_t *>(a.need) : a.value;
    uint32_t ns = 32;
    while (static_cast<int32_t>(*reinterpret_cast<const volatile uint32_t *>(a.word) - target) < 0) {
        __nanosleep(ns);
        ns = ns < 256 ? ns * 2 : ns;
    }
    __threadfence();
    if (a.t != nullptr) {
        volatile uint64_t * const t = a.t;
        t[0] = t0;
        t[1] = globaltimer();
        __threadfence_system();
    }
}

__global__ void __launch_bounds__(32, 1) post_kernel(uint32_t * word, uint32_t value) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t *>(word) = value;
        __threadfence_system();
    }
}

bool aligned8(const void * p) {
    return (reinterpret_cast<uintptr_t>(p) & 7u) == 0;
}

} // namespace

bool resolve(const resolve_args & a, void * stream) {
    if (a.ids == nullptr || a.rows < 1 || a.rows > 5 || a.res == nullptr || !aligned8(a.res) ||
        (a.parts != 2 && a.parts != 4) || a.zero == nullptr || a.tab == nullptr || !aligned8(a.tab) ||
        a.tab2 == nullptr || !aligned8(a.tab2) || a.flags == nullptr || a.ext_slots == nullptr ||
        !aligned8(a.ext_slots) || a.run2 == nullptr || !aligned8(a.run2) || stream == nullptr) {
        return false;
    }
    resolve_kernel<<<1, resolve_threads, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool apply(uint64_t * words, const int32_t * idx, const uint64_t * val, int32_t n, void * stream) {
    if (words == nullptr || !aligned8(words) || idx == nullptr || val == nullptr || !aligned8(val) || n < 0 ||
        n > max_apply || stream == nullptr) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    apply_kernel<<<1, apply_threads, 0, static_cast<cudaStream_t>(stream)>>>(words, idx, val, n);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool combine(const combine_args & a, void * stream) {
    if (a.meta == nullptr || (reinterpret_cast<uintptr_t>(a.meta) & 15u) != 0 || a.wts == nullptr ||
        a.ext_ds == nullptr || a.ext_slots == nullptr || !aligned8(a.ext_slots) || a.out == nullptr || a.rows < 1 ||
        a.rows > 5 || stream == nullptr) {
        return false;
    }
    combine_kernel<<<combine_d / combine_threads, combine_threads, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool combine_finish(const combine_args & a, const float * p1, float * const half[2], void * stream) {
    if (a.meta == nullptr || (reinterpret_cast<uintptr_t>(a.meta) & 15u) != 0 || a.wts == nullptr ||
        a.ext_ds == nullptr || a.ext_slots == nullptr || !aligned8(a.ext_slots) || a.out == nullptr || a.rows < 1 ||
        a.rows > 5 || p1 == nullptr || half == nullptr || half[0] == nullptr || half[1] == nullptr ||
        half[0] == half[1] || stream == nullptr) {
        return false;
    }
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    finish_kernel<<<combine_d / combine_threads, combine_threads, 0, st>>>(a, p1, half[0], half[1]);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool wait_word(const wait_args & a, void * stream) {
    if (a.word == nullptr || (reinterpret_cast<uintptr_t>(a.word) & 3u) != 0 ||
        (a.need != nullptr && (reinterpret_cast<uintptr_t>(a.need) & 3u) != 0) || (a.t != nullptr && !aligned8(a.t)) ||
        stream == nullptr) {
        return false;
    }
    wait_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool post_word(uint32_t * word, uint32_t value, void * stream) {
    if (word == nullptr || (reinterpret_cast<uintptr_t>(word) & 3u) != 0 || stream == nullptr) {
        return false;
    }
    post_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(word, value);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace fn::experts
