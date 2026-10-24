#include "fn/kernels/round/round.h"

#include "fn/kernels/round/version.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fn::round {
namespace {

static_assert(desc_words <= 64, "begin copies the descriptor with two words a lane");

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint64_t load_acquire_sys(const uint64_t * p) {
    uint64_t v = 0;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t load_volatile(const uint32_t * p) {
    uint32_t v = 0;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// One warp. Every lane acquires the start flag itself (so each lane's slot reads are ordered after the host's slot
// writes), then reads two words of the slot; lane 0 checks the slot's round and advances the counter.
__global__ void __launch_bounds__(32) begin_kernel(const begin_args a) {
    const int32_t lane = threadIdx.x;
    const uint64_t r = a.st->next_round;
    const uint64_t t0 = globaltimer();
    bool late = false;
    while (load_acquire_sys(a.flag) < r) {
        if (globaltimer() - t0 > start_timeout_ns) {
            late = true;
            break;
        }
        __nanosleep(32);
    }
    const uint32_t * src = reinterpret_cast<const uint32_t *>(a.host_ring + r % ring);
    uint32_t * dst = reinterpret_cast<uint32_t *>(a.dev);
    for (int32_t w = lane; w < desc_words; w += 32) {
        dst[w] = load_volatile(src + w);
    }
    __syncwarp();
    if (lane == 0) {
        if (late) {
            atomicOr(a.err, fn::xchg::err_poll_start);
        }
        if (a.dev->round != r) {
            atomicOr(a.err, fn::xchg::err_round);
        }
        a.st->t_begin = t0;
        a.st->next_round = r + 1;
    }
}

__global__ void stamp_kernel(state * st) {
    st->t_start = globaltimer();
}

__global__ void stamp_logits_kernel(state * st) {
    st->t_logits = globaltimer();
}

__device__ __forceinline__ int32_t versions_of(const desc & d) {
    switch (static_cast<kind>(d.kind)) {
        case kind::forced:
            return d.rows;
        case kind::generate:
            return 1;
        case kind::verify:
            return d.a + 1;
        case kind::forced_k:
            return d.test_k;
        default:
            return 0;
    }
}

// One warp. Lane i < n_err reads error word i; nothing is committed when any is set (FC-RD2), or when k is 0 or out of
// range. The test hook (desc::test_fail's bit for this card) sets err_test_commit in the card's exchange word first, so
// a test sees exactly what a bounded poll failing on this card alone leaves; ME11: a selector outside the contract sets
// err_gdn_sel there the same way. Otherwise lane 0 writes cur <- version_slot(cur, k) and gdn <- gdn_after_round(gdn,
// k).
__global__ void __launch_bounds__(32) commit_kernel(const commit_args a) {
    const int32_t lane = threadIdx.x;
    const desc & d = *a.dev;
    if (lane == 0 && ((d.test_fail >> a.card) & 1) != 0) {
        // the list is read-only to the frame; only this test hook writes word 0 (the card's exchange word)
        atomicOr(const_cast<uint32_t *>(a.errs[0]), fn::xchg::err_test_commit);
    }
    if (lane == 0 && !gdn_valid(a.st->gdn)) {
        atomicOr(const_cast<uint32_t *>(a.errs[0]), fn::xchg::err_gdn_sel);
    }
    __syncwarp();
    const uint32_t e = lane < a.n_err ? *reinterpret_cast<const volatile uint32_t *>(a.errs[lane]) : 0u;
    if (__any_sync(0xffffffffu, e != 0u)) {
        return;
    }
    const int32_t k = versions_of(d);
    if (lane == 0 && k >= 1 && k <= max_rows) {
        a.st->cur = static_cast<uint32_t>(version_slot(a.st->cur, k));
        a.st->gdn = gdn_after_round(a.st->gdn, k);
    }
}

__global__ void advance_kernel(uint32_t * cur, int32_t k) {
    *cur = static_cast<uint32_t>(version_slot(*cur, k));
}

__global__ void advance_gdn_kernel(uint32_t * cur, uint32_t * gdn, int32_t k, bool prompt) {
    *cur = static_cast<uint32_t>(version_slot(*cur, k));
    const uint32_t g = *gdn;
    if (gdn_valid(g)) {
        *gdn = prompt ? gdn_after_prompt(g) : gdn_after_round(g, k);
    }
}

// one warp: the emitted tokens by kind, then the completion record; the round number last, after a system fence.
// SP1: a verify round's out[0 .. a] and n_out = a + 1 are the decision's (FC-DEC3, the EOS and budget cut applied),
// passed through unchanged
__global__ void __launch_bounds__(32) finish_kernel(const finish_args a) {
    const int32_t lane = threadIdx.x;
    desc & d = *a.dev;
    const kind k = static_cast<kind>(d.kind);
    const bool verify = k == kind::verify;
    const int32_t n_out = verify ? d.n_out : k == kind::generate || k == kind::bootstrap ? 1 : 0;
    if (lane < max_rows) {
        if (!verify) {
            d.out[lane] = lane < n_out ? d.argmax[lane] : -1;
        }
        a.host->out[lane] = d.out[lane];
        a.host->argmax[lane] = lane < d.rows ? d.argmax[lane] : -1;
    }
    if (lane < max_err) {
        a.host->err[lane] = lane < a.n_err ? *a.errs[lane] : 0u;
    }
    if (lane == 0) {
        d.n_out = n_out;
        a.host->cur = a.st->cur;
        a.host->gdn = a.st->gdn;
        const uint64_t t = globaltimer();
        a.st->t_end = t;
        a.host->n_out = n_out;
        a.host->t_begin = a.st->t_begin;
        a.host->t_start = a.st->t_start;
        a.host->t_end = t;
        a.host->t_logits = a.st->t_logits;
    }
    __syncwarp();
    __threadfence_system();
    __syncwarp();
    if (lane == 0) {
        *reinterpret_cast<volatile uint64_t *>(&a.host->round) = d.round;
    }
}

bool al16(const void * p) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

bool launched() {
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

bool begin(const begin_args & a, void * stream) {
    if (!al16(a.host_ring) || a.flag == nullptr || !al16(a.dev) || a.st == nullptr || a.err == nullptr ||
        stream == nullptr) {
        return false;
    }
    begin_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return launched();
}

bool stamp(state * st, void * stream) {
    if (st == nullptr || stream == nullptr) {
        return false;
    }
    stamp_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(st);
    return launched();
}

bool stamp_logits(state * st, void * stream) {
    if (st == nullptr || stream == nullptr) {
        return false;
    }
    stamp_logits_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(st);
    return launched();
}

bool commit(const commit_args & a, void * stream) {
    if (a.dev == nullptr || a.st == nullptr || a.n_err < 1 || a.n_err > max_err || a.errs == nullptr ||
        (a.card != 0 && a.card != 1) || stream == nullptr) {
        return false;
    }
    commit_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return launched();
}

bool advance(uint32_t * cur, int32_t k, void * stream) {
    if (cur == nullptr || k < 0 || k > max_rows || stream == nullptr) {
        return false;
    }
    if (k == 0) {
        return true;
    }
    advance_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(cur, k);
    return launched();
}

bool advance(uint32_t * cur, uint32_t * gdn, int32_t k, bool prompt, void * stream) {
    if (cur == nullptr || gdn == nullptr || k < 0 || k > max_rows || stream == nullptr) {
        return false;
    }
    if (k == 0) {
        return true;
    }
    advance_gdn_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(cur, gdn, k, prompt);
    return launched();
}

bool finish(const finish_args & a, void * stream) {
    if (a.dev == nullptr || a.st == nullptr || a.host == nullptr || a.n_err < 1 || a.n_err > max_err ||
        a.errs == nullptr || stream == nullptr) {
        return false;
    }
    finish_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(a);
    return launched();
}

int32_t commit_versions(int32_t k, int32_t rows, int32_t a, int32_t test_k) {
    switch (static_cast<kind>(k)) {
        case kind::forced:
            return rows;
        case kind::generate:
            return 1;
        case kind::verify:
            return a + 1;
        case kind::forced_k:
            return test_k;
        default:
            return 0;
    }
}

} // namespace fn::round
