#include "fn/xchg/xchg_test.h"

#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr double timeout_s = 120.0; // the host's bound on one row count's rounds (10,000 rounds take about 1 s)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "xchg-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// per card: a stream and the two 64-bit counters (mismatched words, checked words)
struct card_ctx {
    int32_t dev = -1;
    cudaStream_t st = nullptr;
    unsigned long long * mism = nullptr;

    card_ctx() = default;
    card_ctx(const card_ctx &) = delete;
    card_ctx & operator=(const card_ctx &) = delete;

    ~card_ctx() {
        if (dev < 0) {
            return;
        }
        cudaSetDevice(dev);
        if (mism != nullptr) {
            cudaFree(mism);
        }
        if (st != nullptr) {
            cudaStreamDestroy(st);
        }
    }

    bool init(int32_t d) {
        dev = d;
        return cuda_ok(cudaSetDevice(d), "cudaSetDevice") &&
               cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream") &&
               cuda_ok(cudaMalloc(&mism, 2 * sizeof(unsigned long long)), "cudaMalloc counters");
    }
};

// waits for both streams with a host timeout; false on an error or the timeout
bool wait_both(card_ctx (&cards)[2], double limit_s) {
    const auto t0 = std::chrono::steady_clock::now();
    bool done[2] = {false, false};
    while (!(done[0] && done[1])) {
        for (int32_t c = 0; c < 2; ++c) {
            if (done[c]) {
                continue;
            }
            const cudaError_t e = cudaStreamQuery(cards[c].st);
            if (e == cudaSuccess) {
                done[c] = true;
            } else if (e != cudaErrorNotReady) {
                return cuda_ok(e, "stream");
            }
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > limit_s) {
            fprintf(stderr, "xchg-test: host timeout after %.0f s\n", limit_s);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

bool is_nan_bits(uint32_t u) {
    return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u;
}

bool is_inf_bits(uint32_t u) {
    return (u & 0x7fffffffu) == 0x7f800000u;
}

} // namespace

int fnfl_xchg_test(int dev0, int dev1, int32_t rounds, uint32_t rows_mask) {
    using fn::xchg::msg;
    if (rounds < 1) {
        fprintf(stderr, "xchg-test: ROUNDS must be at least 1\n");
        return 2;
    }
    // loopback (dev0 == dev1, for Compute Sanitizer): both cards' rings on one device and every kernel on one stream in
    // dependency order (both sends before either read), so nothing waits on another device
    const bool loopback = dev0 == dev1;
    if (!loopback && !fn::xchg::enable_peer_access(dev0, dev1)) {
        printf("xchg-test: FAIL (no peer access)\n");
        return 1;
    }
    if (loopback) {
        printf("xchg-test: loopback on device %d: both cards' kernels on one stream, sends before reads\n", dev0);
    }
    bool ok = true;
    {
        card_ctx cards[2];
        ok = cards[0].init(dev0) && cards[1].init(dev1);
        fn::xchg::rings rg(dev0, dev1);
        ok = ok && rg.ok();

        // what the payloads carry: count the non-finite words of the rounds that are checked (host, same function)
        int64_t n_sent_sentinel = 0, n_nan = 0, n_inf = 0;
        int64_t n_code_ff = 0; // X-HMIX code words of four -1 codes (the F32 sentinel's bits), HC3
        for (int32_t w = 0; w < 3 * fn::xchg::hmix_layout::width / 4; ++w) {
            n_code_ff += fn::xchg::test_code_word_host(0, 3u, w) == fn::xchg::sentinel ? 1 : 0;
        }
        for (const msg m : {msg::hc, msg::hmix}) {
            for (int32_t w = 0; w < 3 * (m == msg::hc ? fn::xchg::words_hc : fn::xchg::words_hmix); ++w) {
                const uint32_t u = fn::xchg::test_word_host(0, m, 3u, w); // round 3 is a non-finite round
                n_sent_sentinel += u == fn::xchg::sentinel ? 1 : 0;
                n_nan += is_nan_bits(u) ? 1 : 0;
                n_inf += is_inf_bits(u) ? 1 : 0;
            }
        }
        printf("xchg-test: a non-finite round at B = 3 carries %lld NaN words (%lld of them the sentinel pattern) and "
               "%lld infinities from each card\n",
               static_cast<long long>(n_nan), static_cast<long long>(n_sent_sentinel), static_cast<long long>(n_inf));
        printf("xchg-test: and X-HMIX code words of four -1 codes (the F32 sentinel's bits): %lld\n",
               static_cast<long long>(n_code_ff));
        ok = ok && n_sent_sentinel > 0 && n_inf > 0 && n_code_ff > 0;

        for (const int32_t rows : {1, 3}) {
            if (!ok || (rows_mask >> (rows - 1) & 1u) == 0) {
                continue;
            }
            ok = rg.reset_all();
            for (int32_t c = 0; c < 2 && ok; ++c) {
                ok = cuda_ok(cudaSetDevice(cards[c].dev), "cudaSetDevice") &&
                     cuda_ok(cudaMemset(cards[c].mism, 0, 2 * sizeof(unsigned long long)), "counters") &&
                     cuda_ok(cudaDeviceSynchronize(), "counters"); // legacy stream: not ordered with cards[c].st
            }
            const auto t0 = std::chrono::steady_clock::now();
            // each card per round: send X-HC (resetting its previous X-HC slot), read X-HC, send X-HMIX, read X-HMIX;
            // phase by phase over the cards, so in loopback every send precedes its read on the one stream
            for (int32_t k = 0; k < rounds && ok; ++k) {
                const int32_t s = k % fn::xchg::n_slot;
                const int32_t prev = (k + fn::xchg::n_slot - 1) % fn::xchg::n_slot;
                const uint32_t ku = static_cast<uint32_t>(k);
                for (int32_t ph = 0; ph < 4 && ok; ++ph) {
                    for (int32_t c = 0; c < 2 && ok; ++c) {
                        cudaStream_t st = loopback ? cards[0].st : cards[c].st;
                        ok = cuda_ok(cudaSetDevice(cards[c].dev), "cudaSetDevice");
                        if (ph == 0) {
                            ok = ok && fn::xchg::test_send(msg::hc, rows, c, ku, rg.slot(1 - c, msg::hc, s),
                                                           k > 0 ? rg.slot(c, msg::hc, prev) : nullptr, st,
                                                           k > 0 ? rg.slot(c, msg::hmix, prev) : nullptr);
                        } else if (ph == 1) {
                            ok = ok && fn::xchg::test_recv(msg::hc, rows, 1 - c, ku, rg.slot(c, msg::hc, s), rg.err(c),
                                                           cards[c].mism, st);
                        } else if (ph == 2) {
                            ok = ok &&
                                 fn::xchg::test_send(msg::hmix, rows, c, ku, rg.slot(1 - c, msg::hmix, s), nullptr, st);
                        } else {
                            ok = ok && fn::xchg::test_recv(msg::hmix, rows, 1 - c, ku, rg.slot(c, msg::hmix, s),
                                                           rg.err(c), cards[c].mism, st);
                        }
                        if (!ok) {
                            fprintf(stderr, "xchg-test: round %d card %d phase %d not enqueued: %s\n", k, c, ph,
                                    cudaGetErrorString(cudaGetLastError()));
                        }
                    }
                }
            }
            ok = ok && wait_both(cards, timeout_s);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // per round each card checks X-HC once in each of hc_up's 72 blocks, and the peer's half of X-HMIX (F32,
            // codes, scales: 1,680 words a row) once in each of the consumer's 72 blocks
            const unsigned long long hmix_half_words =
                fn::xchg::words_hmix + fn::xchg::words_hmix / 4 + fn::xchg::words_hmix / 32 * 2;
            const unsigned long long want =
                static_cast<unsigned long long>(rounds) * rows * (72ull * fn::xchg::words_hc + 72ull * hmix_half_words);
            for (int32_t c = 0; c < 2 && ok; ++c) {
                unsigned long long h[2] = {~0ull, 0ull};
                uint32_t err = ~0u;
                ok = cuda_ok(cudaSetDevice(cards[c].dev), "cudaSetDevice") &&
                     cuda_ok(cudaMemcpy(h, cards[c].mism, sizeof(h), cudaMemcpyDeviceToHost), "counters") &&
                     rg.read_err(c, err);
                // every X-HMIX slot but the last round's is back at the sentinels: the next round's X-HC sender
                // reset it after its readers (HC3; code words to code_sentinel)
                int64_t not_reset = 0;
                for (int32_t s = 0; s < fn::xchg::n_slot && ok; ++s) {
                    if (s == (rounds - 1) % fn::xchg::n_slot || (rounds < fn::xchg::n_slot && s >= rounds - 1)) {
                        continue;
                    }
                    std::vector<uint32_t> w(size_t(fn::xchg::slot_bytes(msg::hmix) / 4));
                    ok = cuda_ok(cudaMemcpy(w.data(), rg.slot(c, msg::hmix, s), w.size() * 4, cudaMemcpyDeviceToHost),
                                 "slot");
                    for (size_t i = 0; i < w.size(); ++i) {
                        const bool code =
                            i >= size_t(fn::xchg::hmix_layout::code_at) && i < size_t(fn::xchg::hmix_layout::scale_at);
                        not_reset += w[i] != (code ? fn::xchg::code_sentinel : fn::xchg::sentinel) ? 1 : 0;
                    }
                }
                const bool pass = ok && h[0] == 0 && h[1] == want && err == 0 && not_reset == 0;
                printf("xchg-test B%d card %d: %d rounds, %llu words checked (want %llu), %llu mismatched, error word "
                       "0x%x, X-HMIX words not reset %lld, %.2f s%s\n",
                       rows, c, rounds, h[1], want, h[0], err, static_cast<long long>(not_reset), secs,
                       pass ? "" : "  FAIL");
                ok = pass;
            }
        }
        for (int32_t c = 0; c < 2; ++c) {
            cudaSetDevice(cards[c].dev);
            ok = cuda_ok(cudaDeviceSynchronize(), "final sync") && ok;
        }
    }
    printf("xchg-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
