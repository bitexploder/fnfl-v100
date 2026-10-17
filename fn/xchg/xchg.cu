#include "fn/xchg/xchg.h"

#include "fn/xchg/xchg.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace fn::xchg {
namespace {

// every message's slots_of(m) slots, in msg order
constexpr int64_t words_before(int32_t m) {
    int64_t at = 0;
    for (int32_t i = 0; i < m; ++i) {
        at += int64_t(slots_of(static_cast<msg>(i))) * slot_words(static_cast<msg>(i));
    }
    return at;
}
constexpr int64_t card_words = words_before(n_msg);

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "xchg: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// sets the current device and restores the previous one when it goes out of scope
struct device_scope {
    int prev = 0;
    bool ok = false;

    explicit device_scope(int32_t dev) { ok = cudaGetDevice(&prev) == cudaSuccess && cudaSetDevice(dev) == cudaSuccess; }

    ~device_scope() { cudaSetDevice(prev); }

    device_scope(const device_scope &) = delete;
    device_scope & operator=(const device_scope &) = delete;
};

} // namespace

bool enable_peer_access(int32_t dev0, int32_t dev1) {
    const int32_t dev[2] = {dev0, dev1};
    for (int32_t c = 0; c < 2; ++c) {
        int can = 0;
        if (!cuda_ok(cudaDeviceCanAccessPeer(&can, dev[c], dev[1 - c]), "cudaDeviceCanAccessPeer")) {
            return false;
        }
        if (can == 0) {
            fprintf(stderr, "xchg: device %d cannot access device %d\n", dev[c], dev[1 - c]);
            return false;
        }
        device_scope ds(dev[c]);
        if (!ds.ok) {
            return cuda_ok(cudaGetLastError(), "cudaSetDevice");
        }
        const cudaError_t e = cudaDeviceEnablePeerAccess(dev[1 - c], 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError(); // clears the sticky-free error state; peer access is on
        } else if (!cuda_ok(e, "cudaDeviceEnablePeerAccess")) {
            return false;
        }
    }
    return true;
}

int64_t rings_card_bytes() {
    return card_words * int64_t(sizeof(float)) + int64_t(sizeof(uint32_t));
}

rings::rings(int32_t dev0, int32_t dev1) {
    dev_[0] = dev0;
    dev_[1] = dev1;
    bool ok = true;
    for (int32_t c = 0; c < 2 && ok; ++c) {
        device_scope ds(dev_[c]);
        ok = ds.ok && cuda_ok(cudaMalloc(&buf_[c], size_t(card_words) * sizeof(float)), "cudaMalloc rings") &&
             cuda_ok(cudaMalloc(&err_[c], sizeof(uint32_t)), "cudaMalloc err");
    }
    ok_ = ok && reset_all();
}

rings::~rings() {
    for (int32_t c = 0; c < 2; ++c) {
        if (buf_[c] == nullptr && err_[c] == nullptr) {
            continue;
        }
        device_scope ds(dev_[c]);
        if (buf_[c] != nullptr) {
            cudaFree(buf_[c]);
        }
        if (err_[c] != nullptr) {
            cudaFree(err_[c]);
        }
    }
}

float * rings::slot(int32_t card, msg m, int32_t s) const {
    if (card < 0 || card > 1 || s < 0 || s >= slots_of(m) || buf_[card] == nullptr) {
        return nullptr;
    }
    return buf_[card] + words_before(static_cast<int32_t>(m)) + int64_t(s) * slot_words(m);
}

bool rings::reset_all() const {
    for (int32_t c = 0; c < 2; ++c) {
        device_scope ds(dev_[c]);
        if (!ds.ok || buf_[c] == nullptr || err_[c] == nullptr ||
            !cuda_ok(cudaMemset(buf_[c], 0xff, size_t(card_words) * sizeof(float)), "prefill") ||
            !cuda_ok(cudaMemset(err_[c], 0, sizeof(uint32_t)), "err clear")) {
            return false;
        }
        // X-HMIX's code words have their own sentinel (HC3): 0x80 bytes
        for (int32_t s = 0; s < n_slot; ++s) {
            if (!cuda_ok(cudaMemset(slot(c, msg::hmix, s) + hmix_layout::code_at, 0x80,
                                    size_t(hmix_layout::code_words) * sizeof(float)),
                         "prefill codes")) {
                return false;
            }
        }
        if (!cuda_ok(cudaDeviceSynchronize(), "prefill sync")) {
            return false;
        }
    }
    return true;
}

bool rings::read_err(int32_t card, uint32_t & out) const {
    out = ~0u;
    if (card < 0 || card > 1 || err_[card] == nullptr) {
        return false;
    }
    device_scope ds(dev_[card]);
    return ds.ok && cuda_ok(cudaMemcpy(&out, err_[card], sizeof(out), cudaMemcpyDeviceToHost), "read err");
}

// ---- xchg-test stand-ins ----

namespace {

constexpr int32_t n_block = 72; // both senders' and the X-HC readers' grids (hc_down, hc_up)
constexpr int32_t hc_rows_per_block = 18;
constexpr int32_t hmix_big_blocks = 56; // hc_up: 56 blocks of 18 features, then 16 of 17
constexpr int32_t hmix_feat_max = 18;

__host__ __device__ __forceinline__ uint32_t test_word(int32_t c, int32_t m, uint32_t k, int32_t w) {
    if (k % 7u == 3u) {
        if (w % 97 == 5) {
            const int32_t i = (w / 97) % 3; // the sentinel pattern itself, then two other NaNs
            return i == 0 ? sentinel : i == 1 ? 0x7fc00001u : 0xffc00000u;
        }
        if (w % 89 == 7) {
            return (w / 89) % 2 != 0 ? 0x7f800000u : 0xff800000u;
        }
    }
    const uint32_t h = (uint32_t(w) * 2654435761u) ^ (k * 0x9e3779b9u) ^ (c != 0 ? 0xa5a5a5a5u : 0x3c3c3c3cu) ^
                       (m != 0 ? 0x5a5a0000u : 0u);
    return (h & 0x807fffffu) | ((107u + (h >> 23 & 31u)) << 23); // a finite normal value
}

__host__ __device__ __forceinline__ uint32_t expected_word(int32_t c, int32_t m, uint32_t k, int32_t w) {
    const uint32_t u = test_word(c, m, k, w);
    const bool nan = (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u;
    return nan ? canonical_nan : u;
}

__device__ __forceinline__ uint32_t canon_bits(uint32_t u) {
    return canonical(__uint_as_float(u));
}

// X-HMIX's code and scale test words (HC3), at word indices of their own (code_w0, scale_w0 + ...): code words are
// test_word with every 0x80 byte made 0x81 (act.h never produces -128) and, in the non-finite rounds, every 13th word
// four -1 codes (the F32 sentinel's bits); scale words are F32 test words, canonicalized like the F32 values
constexpr int32_t code_w0 = 100000, scale_w0 = 200000;
__host__ __device__ __forceinline__ uint32_t code_word(int32_t c, uint32_t k, int32_t w) {
    if (k % 7u == 3u && w % 13 == 4) {
        return sentinel;
    }
    uint32_t u = test_word(c, 2, k, code_w0 + w);
    for (int32_t i = 0; i < 4; ++i) {
        u ^= ((u >> (8 * i)) & 0xffu) == 0x80u ? 1u << (8 * i) : 0u;
    }
    return u;
}

// X-HC sender: block (s, g) as hc_down's, rows 18g .. 18g+17 of stream s for every payload row, then g = 0 sends v[s]
template <int B>
__global__ void __launch_bounds__(128) send_hc_kernel(int32_t card, uint32_t k, uint32_t * peer, uint32_t * reset,
                                                      float * hmix_reset) {
    const int32_t s = blockIdx.x % 4;
    const int32_t g = blockIdx.x / 4;
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    if (warp < B) {
        const int32_t b = warp;
        const int32_t j = g * hc_rows_per_block + (lane < hc_rows_per_block ? lane : 0);
        const int32_t at = j < 320 ? 4 + 320 * s + j : 1284 + 4 * s + (j - 320);
        const uint32_t bits = canon_bits(test_word(card, 0, k, b * words_hc + at));
        put_hc_rows(nullptr, peer + b * words_hc, s, g * hc_rows_per_block, hc_rows_per_block, bits);
    }
    if (g == 0 && threadIdx.x < B) {
        const int32_t b = threadIdx.x;
        peer[b * words_hc + s] = canon_bits(test_word(card, 0, k, b * words_hc + s));
    }
    if (reset != nullptr) {
        const int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u < B * words_hc / 4) {
            reset4(reinterpret_cast<uint4 *>(reset) + u);
        }
    }
    // HC3: this card's X-HMIX slot of the previous round, after its readers, as hc_down resets it (xchg.h)
    if (hmix_reset != nullptr) {
        for (int32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < hmix_slot_u4; u += gridDim.x * blockDim.x) {
            hmix_reset_slot(hmix_reset, u);
        }
    }
}

// X-HMIX sender (HC3): block g as hc_up's, features c0 .. c0+n_f-1 of every payload row's F32 values; block g < 40
// also sends run g of the card's half (codes and {d, s}) as the block that completes it would (hmix_put_run)
template <int B> __global__ void __launch_bounds__(128) send_hmix_kernel(int32_t card, uint32_t k, float * peer) {
    const int32_t g = blockIdx.x;
    const int32_t n_f = g < hmix_big_blocks ? hmix_feat_max : hmix_feat_max - 1;
    const int32_t c0 = g < hmix_big_blocks ? hmix_feat_max * g
                                           : hmix_big_blocks * hmix_feat_max + (hmix_feat_max - 1) * (g - hmix_big_blocks);
    const int32_t warp = threadIdx.x / 32;
    const int32_t lane = threadIdx.x % 32;
    if (warp < B) {
        const int32_t b = warp;
        const uint32_t bits = canon_bits(test_word(card, 1, k, b * words_hmix + c0 + (lane < n_f ? lane : 0)));
        put_run(nullptr, reinterpret_cast<uint32_t *>(peer) + b * hmix_layout::width + card * words_hmix, c0, n_f, 0,
                bits);
    }
    if (warp == 3 && g < words_hmix / 32) { // lane 8 b + i: code word i of run j of row b
        const int32_t b = lane / 8, i = lane % 8, j = card * (words_hmix / 32) + g;
        const bool real = b < B;
        const uint32_t packed = code_word(card, k, b * (hmix_layout::width / 4) + 8 * j + i);
        const int32_t sw = b * 2 * hmix_layout::runs + 2 * j;
        const float2 ds = make_float2(__uint_as_float(test_word(card, 3, k, scale_w0 + sw)),
                                      __uint_as_float(test_word(card, 3, k, scale_w0 + sw + 1)));
        hmix_put_run(nullptr, peer, b, j, packed, ds, real);
    }
}

__device__ __forceinline__ void count(unsigned long long * mism, uint32_t bad, uint32_t checked) {
#pragma unroll
    for (int32_t o = 16; o > 0; o >>= 1) {
        bad += __shfl_xor_sync(0xffffffffu, bad, o);
        checked += __shfl_xor_sync(0xffffffffu, checked, o);
    }
    if (threadIdx.x % 32 == 0) {
        if (bad != 0u) {
            atomicAdd(mism, static_cast<unsigned long long>(bad));
        }
        atomicAdd(mism + 1, static_cast<unsigned long long>(checked));
    }
}

__device__ __forceinline__ uint32_t bad4(uint4 v, int32_t sender, int32_t m, uint32_t k, int32_t w) {
    return (v.x != expected_word(sender, m, k, w)) + (v.y != expected_word(sender, m, k, w + 1)) +
           (v.z != expected_word(sender, m, k, w + 2)) + (v.w != expected_word(sender, m, k, w + 3));
}

// X-HC reader: every block polls every word of the slot, as each block of hc_up reads all the moments
template <int B> __global__ void __launch_bounds__(480) recv_hc_kernel(int32_t sender, uint32_t k, const uint32_t * slot,
                                                                       uint32_t * err, unsigned long long * mism) {
    uint32_t bad = 0, checked = 0;
    for (int32_t u = threadIdx.x; u < B * words_hc / 4; u += blockDim.x) {
        const uint4 * p[1] = {reinterpret_cast<const uint4 *>(slot) + u};
        uint4 v[1];
        poll<1>(p, v, err, err_poll_test);
        bad += bad4(v[0], sender, 0, k, 4 * u);
        checked += 4;
    }
    count(mism, bad, checked);
}

// X-HMIX reader (HC3): every block polls every word of the sender's half, as each block of a consumer reads it: F32
// (320 groups a row), codes (80) and scales (20); no reset (the next round's X-HC sender resets the slot)
template <int B>
__global__ void __launch_bounds__(480) recv_hmix_kernel(int32_t sender, uint32_t k, const float * slot, uint32_t * err,
                                                        unsigned long long * mism) {
    constexpr int32_t g_f = words_hmix / 4, g_c = words_hmix / 16, g_s = words_hmix / 32 / 2; // groups a row
    const uint4 * s4 = reinterpret_cast<const uint4 *>(slot);
    uint32_t bad = 0, checked = 0;
    for (int32_t u = threadIdx.x; u < B * (g_f + g_c + g_s); u += blockDim.x) {
        const uint4 * p[1];
        uint4 v[1];
        if (u < B * g_f) {
            const int32_t b = u / g_f, i = u - b * g_f;
            p[0] = s4 + (b * hmix_layout::width + sender * words_hmix) / 4 + i;
            poll<1>(p, v, err, err_poll_test);
            bad += bad4(v[0], sender, 1, k, b * words_hmix + 4 * i);
        } else if (u < B * (g_f + g_c)) {
            const int32_t b = (u - B * g_f) / g_c, i = u - B * g_f - b * g_c;
            const int32_t w = b * (hmix_layout::width / 4) + sender * (words_hmix / 4) + 4 * i; // code word index
            p[0] = s4 + hmix_layout::code_at / 4 + w / 4;
            poll<1>(p, v, err, err_poll_test, code_sentinel);
            bad += (v[0].x != code_word(sender, k, w)) + (v[0].y != code_word(sender, k, w + 1)) +
                   (v[0].z != code_word(sender, k, w + 2)) + (v[0].w != code_word(sender, k, w + 3));
        } else {
            const int32_t b = (u - B * (g_f + g_c)) / g_s, i = u - B * (g_f + g_c) - b * g_s;
            const int32_t w = b * 2 * hmix_layout::runs + sender * (words_hmix / 32) * 2 + 4 * i; // scale word index
            p[0] = s4 + hmix_layout::scale_at / 4 + w / 4;
            poll<1>(p, v, err, err_poll_test);
            bad += bad4(v[0], sender, 3, k, scale_w0 + w);
        }
        checked += 4;
    }
    count(mism, bad, checked);
}

template <int B>
bool send(msg m, int32_t card, uint32_t k, uint32_t * peer, uint32_t * reset, float * hmix_reset, cudaStream_t st) {
    if (m == msg::hc) {
        send_hc_kernel<B><<<n_block, 128, 0, st>>>(card, k, peer, reset, hmix_reset);
    } else {
        send_hmix_kernel<B><<<n_block, 128, 0, st>>>(card, k, reinterpret_cast<float *>(peer));
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

template <int B> bool recv(msg m, int32_t sender, uint32_t k, uint32_t * slot, uint32_t * err,
                           unsigned long long * mism, cudaStream_t st) {
    if (m == msg::hc) {
        recv_hc_kernel<B><<<n_block, 480, 0, st>>>(sender, k, slot, err, mism);
    } else {
        recv_hmix_kernel<B><<<n_block, 480, 0, st>>>(sender, k, reinterpret_cast<const float *>(slot), err, mism);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

bool aligned16(const void * p) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

} // namespace

uint32_t test_word_host(int32_t c, msg m, uint32_t k, int32_t w) {
    return test_word(c, static_cast<int32_t>(m), k, w);
}

uint32_t test_code_word_host(int32_t c, uint32_t k, int32_t w) {
    return code_word(c, k, w);
}

bool test_send(msg m, int32_t rows, int32_t card, uint32_t k, float * peer_slot, float * reset_slot, void * stream,
               float * hmix_reset) {
    if (stream == nullptr || !aligned16(peer_slot) || (reset_slot != nullptr && !aligned16(reset_slot)) ||
        (hmix_reset != nullptr && (m != msg::hc || !aligned16(hmix_reset)))) {
        return false;
    }
    uint32_t * peer = reinterpret_cast<uint32_t *>(peer_slot); // F32 payload words handled as bits
    uint32_t * reset = reinterpret_cast<uint32_t *>(reset_slot);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1:
            return send<1>(m, card, k, peer, reset, hmix_reset, st);
        case 3:
            return send<3>(m, card, k, peer, reset, hmix_reset, st);
        default: return false;
    }
}

bool test_recv(msg m, int32_t rows, int32_t sender, uint32_t k, float * slot, uint32_t * err,
               unsigned long long * mism, void * stream) {
    if (stream == nullptr || !aligned16(slot) || err == nullptr || mism == nullptr) {
        return false;
    }
    uint32_t * s = reinterpret_cast<uint32_t *>(slot); // F32 payload words handled as bits
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    switch (rows) {
        case 1: return recv<1>(m, sender, k, s, err, mism, st);
        case 3: return recv<3>(m, sender, k, s, err, mism, st);
        default: return false;
    }
}

} // namespace fn::xchg
