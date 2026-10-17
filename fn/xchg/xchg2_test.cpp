#include "fn/xchg/xchg2_test.h"

#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace x = fn::xchg;

constexpr int32_t D = 2560;
constexpr int32_t F = 1280;
constexpr int32_t R = x::max_rows; // SP2: the rows a test buffer holds (5)
constexpr int32_t max_body = x::max_rows * x::select_row_words;
constexpr uint32_t untouched = 0xcdcdcdcdu; // the receive buffers' fill: words past the body must keep it

bool ok_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "xchg2-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

uint32_t hash(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 2654435761u ^ (b + 0x9e3779b9u + (a << 6) + (a >> 2)) ^ (c * 0x85ebca6bu);
    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    h ^= h >> 12;
    return h;
}

// a finite normal value, except some words NaN (one the sentinel pattern) in rounds k % 5 == 2
float partial_word(int32_t card, uint32_t k, int32_t w) {
    const uint32_t h = hash(uint32_t(card) + 7u, k, uint32_t(w));
    uint32_t u = (h & 0x807fffffu) | ((110u + (h >> 23 & 15u)) << 23);
    if (k % 5u == 2u && w % 251 == 3) {
        u = w % 2 ? x::sentinel : 0x7fc00001u;
    }
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct card_bufs {
    float * partial = nullptr;      // [R][2560]
    float * out = nullptr;          // [R][1280]
    int32_t * ids_full = nullptr;   // [R][512]
    float * wts = nullptr;          // [R][10]
    float * sg = nullptr;           // [R]
    int32_t * ids = nullptr;        // [R][10]
    uint32_t * body = nullptr;      // [max_body]: the owner's selection body
    int32_t * offsets = nullptr;    // [8]
    uint32_t * body_in = nullptr;   // [max_body]: the peer's received body
    int32_t * offsets_in = nullptr; // [8]
    cudaStream_t st = nullptr;
};

bool alloc(card_bufs & b, int32_t dev) {
    return ok_cuda(cudaSetDevice(dev), "device") && ok_cuda(cudaMalloc(&b.partial, int64_t{R} * D * 4), "partial") &&
           ok_cuda(cudaMalloc(&b.out, int64_t{R} * F * 4), "out") &&
           ok_cuda(cudaMalloc(&b.ids_full, int64_t{R} * 512 * 4), "ids_full") &&
           ok_cuda(cudaMalloc(&b.wts, int64_t{R} * 10 * 4), "wts") &&
           ok_cuda(cudaMalloc(&b.sg, int64_t{R} * 4), "sg") &&
           ok_cuda(cudaMalloc(&b.ids, int64_t{R} * 10 * 4), "ids") &&
           ok_cuda(cudaMalloc(&b.body, int64_t(max_body) * 4), "body") &&
           ok_cuda(cudaMalloc(&b.offsets, 32), "offsets") &&
           ok_cuda(cudaMalloc(&b.body_in, int64_t(max_body) * 4), "body_in") &&
           ok_cuda(cudaMalloc(&b.offsets_in, 32), "offsets_in") &&
           ok_cuda(cudaStreamCreateWithFlags(&b.st, cudaStreamNonBlocking), "stream");
}

void release(card_bufs & b, int32_t dev) {
    cudaSetDevice(dev);
    cudaFree(b.partial);
    cudaFree(b.out);
    cudaFree(b.ids_full);
    cudaFree(b.wts);
    cudaFree(b.sg);
    cudaFree(b.ids);
    cudaFree(b.body);
    cudaFree(b.offsets);
    cudaFree(b.body_in);
    cudaFree(b.offsets_in);
    if (b.st != nullptr) {
        cudaStreamDestroy(b.st);
    }
}

bool same_f32(float got, float want) {
    if (std::isnan(want)) {
        return std::isnan(got);
    }
    return std::memcmp(&got, &want, 4) == 0;
}

} // namespace

int fnfl_xchg2_test(int32_t dev0, int32_t dev1, int32_t rounds) {
    const int32_t dev[2] = {dev0, dev1};
    // loopback (dev0 == dev1: one card, or Compute Sanitizer): both cards' buffers and rings on one device and every
    // kernel on card 0's stream, so each send precedes its read
    const bool loopback = dev0 == dev1;
    if (loopback) {
        printf("xchg2-test: loopback on device %d: both cards' kernels on one stream, sends before reads\n", dev0);
    } else if (!x::enable_peer_access(dev0, dev1)) {
        return 1;
    }
    x::rings rg(dev0, dev1);
    card_bufs cb[2];
    bool ok = rg.ok() && alloc(cb[0], dev0) && alloc(cb[1], dev1);
    if (ok && loopback) {
        cudaStreamDestroy(cb[1].st);
        cb[1].st = cb[0].st;
    }
    int64_t words = 0, bad = 0;
    std::vector<float> part[2], out(int64_t{R} * F);
    std::vector<int32_t> ids_full(int64_t{R} * 512), ids(int64_t{R} * 10), offs(8), got_offs(8);
    std::vector<float> wts(R * 10), sg(R), got_w(R * 10), got_sg(R);
    std::vector<uint32_t> body, got_body(max_body);
    int64_t uses_half = 0, uses_route = 0, uses_select = 0;
    for (uint32_t k = 0; ok && k < uint32_t(rounds); ++k) {
        for (int32_t B = 1; ok && B <= x::max_rows; ++B) { // SP2: 1..5
            const int32_t owner = int32_t((k + uint32_t(B)) % 2u);
            // host payloads
            for (int32_t c = 0; c < 2; ++c) {
                part[c].resize(int64_t{R} * D);
                for (int32_t w = 0; w < B * D; ++w) {
                    part[c][w] = partial_word(c, k * 4u + uint32_t(B), w);
                }
            }
            for (int32_t b = 0; b < B; ++b) {
                for (int32_t j = 0; j < 512; ++j) {
                    ids_full[b * 512 + j] = int32_t(hash(k, uint32_t(b), uint32_t(j)) % 512u);
                }
                for (int32_t j = 0; j < 10; ++j) {
                    wts[b * 10 + j] = partial_word(owner, k + 100u, b * 10 + j);
                }
                sg[b] = partial_word(owner, k + 200u, b);
            }
            // a selection body in fn::qsa's layout at a depth (shallow in a quarter of the rounds, so rows hold fewer
            // than 512 blocks and the body's last 16-byte group is partial): counts, block IDs, tail cells
            const uint32_t hd = hash(k, uint32_t(B), 9u);
            const int32_t depth = int32_t(hd % 4u == 0u ? hd / 4u % 3000u : hd / 4u % 262140u);
            body.clear();
            for (int32_t b = 0; b < B; ++b) {
                const int32_t q = depth + b, n_vis = (q + 1) / 4, kb = std::min(n_vis, 512);
                offs[b] = int32_t(body.size() * 4);
                body.push_back(uint32_t(kb));
                body.push_back(uint32_t(q + 1 - 4 * n_vis));
                for (int32_t j = 0; j < kb; ++j) {
                    body.push_back(hash(k, uint32_t(b), uint32_t(j) + 13u) % uint32_t(n_vis));
                }
                for (int32_t c = 4 * n_vis; c <= q; ++c) {
                    body.push_back(uint32_t(c));
                }
            }
            const int32_t n_body = int32_t(body.size());
            for (int32_t t = B; t < 8; ++t) {
                offs[t] = n_body * 4;
            }
            const int32_t s_half = int32_t(uses_half++ % x::n_slot);
            const int32_t s_route = int32_t(uses_route++ % x::n_slot);
            const int32_t s_sel = int32_t(uses_select++ % x::n_slot);
            for (int32_t c = 0; c < 2 && ok; ++c) {
                card_bufs & q = cb[c];
                ok =
                    ok_cuda(cudaSetDevice(dev[c]), "device") &&
                    ok_cuda(cudaMemcpyAsync(q.partial, part[c].data(), size_t(B) * D * 4, cudaMemcpyHostToDevice, q.st),
                            "h2d") &&
                    ok_cuda(cudaStreamSynchronize(q.st), "h2d");
                if (ok && c == owner) {
                    ok =
                        ok_cuda(cudaMemcpy(q.ids_full, ids_full.data(), size_t(B) * 512 * 4, cudaMemcpyHostToDevice),
                                "h2d") &&
                        ok_cuda(cudaMemcpy(q.wts, wts.data(), size_t(B) * 40, cudaMemcpyHostToDevice), "h2d") &&
                        ok_cuda(cudaMemcpy(q.sg, sg.data(), size_t(B) * 4, cudaMemcpyHostToDevice), "h2d") &&
                        ok_cuda(cudaMemcpy(q.body, body.data(), size_t(n_body) * 4, cudaMemcpyHostToDevice), "h2d") &&
                        ok_cuda(cudaMemcpy(q.offsets, offs.data(), size_t(B + 1) * 4, cudaMemcpyHostToDevice), "h2d") &&
                        ok_cuda(cudaDeviceSynchronize(), "h2d"); // pageable copies may return before the DMA lands
                } else if (ok) {
                    ok = ok_cuda(cudaMemsetAsync(q.body_in, 0xcd, int64_t(max_body) * 4, q.st), "fill") &&
                         ok_cuda(cudaMemsetAsync(q.offsets_in, 0xcd, 32, q.st), "fill") &&
                         ok_cuda(cudaStreamSynchronize(q.st), "fill");
                }
            }
            // the layer's phase order: the owners' sends, the peer's receives; both half sends, then both receives
            const int32_t peer = 1 - owner;
            cudaSetDevice(dev[owner]);
            ok = ok &&
                 x::send_route(B, cb[owner].ids_full, 512, cb[owner].wts, cb[owner].sg, cb[owner].ids,
                               rg.slot(peer, x::msg::route, s_route), cb[owner].st) &&
                 x::send_select(B, cb[owner].offsets, cb[owner].body, rg.slot(peer, x::msg::select, s_sel),
                                cb[owner].st);
            cudaSetDevice(dev[peer]);
            ok = ok &&
                 x::recv_route(B, rg.slot(peer, x::msg::route, s_route), cb[peer].ids, cb[peer].wts, cb[peer].sg,
                               rg.err(peer), cb[peer].st) &&
                 x::recv_select(B, rg.slot(peer, x::msg::select, s_sel), cb[peer].offsets_in, cb[peer].body_in,
                                rg.err(peer), cb[peer].st);
            for (int32_t c = 0; c < 2 && ok; ++c) {
                cudaSetDevice(dev[c]);
                ok = x::send_half(B, 1 - c, cb[c].partial, rg.slot(1 - c, x::msg::a, s_half), cb[c].st);
            }
            for (int32_t c = 0; c < 2 && ok; ++c) {
                cudaSetDevice(dev[c]);
                ok = x::recv_add(B, c, cb[c].partial, rg.slot(c, x::msg::a, s_half), cb[c].out, rg.err(c),
                                 x::err_poll_a, cb[c].st);
            }
            for (int32_t c = 0; c < 2 && ok; ++c) {
                cudaSetDevice(dev[c]);
                ok = ok_cuda(cudaStreamSynchronize(cb[c].st), "round");
            }
            // checks
            for (int32_t c = 0; c < 2 && ok; ++c) {
                cudaSetDevice(dev[c]);
                ok = ok_cuda(cudaMemcpy(out.data(), cb[c].out, size_t(B) * F * 4, cudaMemcpyDeviceToHost), "d2h");
                for (int32_t b = 0; b < B; ++b) {
                    for (int32_t i = 0; i < F; ++i) {
                        const float want = part[c][b * D + c * F + i] + part[1 - c][b * D + c * F + i];
                        bad += same_f32(out[b * F + i], want) ? 0 : 1;
                        words++;
                    }
                }
            }
            if (ok) {
                std::vector<int32_t> got_ids(R * 10), own_ids(R * 10);
                cudaSetDevice(dev[peer]);
                ok = ok_cuda(cudaMemcpy(got_ids.data(), cb[peer].ids, size_t(B) * 40, cudaMemcpyDeviceToHost), "d2h") &&
                     ok_cuda(cudaMemcpy(got_w.data(), cb[peer].wts, size_t(B) * 40, cudaMemcpyDeviceToHost), "d2h") &&
                     ok_cuda(cudaMemcpy(got_sg.data(), cb[peer].sg, size_t(B) * 4, cudaMemcpyDeviceToHost), "d2h") &&
                     ok_cuda(
                         cudaMemcpy(got_body.data(), cb[peer].body_in, int64_t(max_body) * 4, cudaMemcpyDeviceToHost),
                         "d2h") &&
                     ok_cuda(cudaMemcpy(got_offs.data(), cb[peer].offsets_in, 32, cudaMemcpyDeviceToHost), "d2h");
                cudaSetDevice(dev[owner]);
                ok = ok &&
                     ok_cuda(cudaMemcpy(own_ids.data(), cb[owner].ids, size_t(B) * 40, cudaMemcpyDeviceToHost), "d2h");
                for (int32_t b = 0; b < B; ++b) {
                    for (int32_t j = 0; j < 10; ++j) {
                        bad += got_ids[b * 10 + j] != ids_full[b * 512 + j];
                        bad += own_ids[b * 10 + j] != ids_full[b * 512 + j];
                        bad += same_f32(got_w[b * 10 + j], wts[b * 10 + j]) ? 0 : 1;
                        words += 3;
                    }
                    bad += same_f32(got_sg[b], sg[b]) ? 0 : 1;
                    words++;
                }
                // the body and offsets unchanged, and nothing written past them
                for (int32_t i = 0; i < max_body; ++i) {
                    bad += got_body[i] != (i < n_body ? body[i] : untouched);
                    words++;
                }
                for (int32_t t = 0; t < 8; ++t) {
                    bad += uint32_t(got_offs[t]) != (t <= B ? uint32_t(offs[t]) : untouched);
                    words++;
                }
            }
        }
    }
    uint32_t e0 = ~0u, e1 = ~0u;
    ok = ok && rg.read_err(0, e0) && rg.read_err(1, e1);
    if (loopback) {
        cb[1].st = nullptr; // card 0's, released with it
    }
    release(cb[0], dev0);
    release(cb[1], dev1);
    printf("xchg2-test: %d rounds x B = 1..5 on devices %d and %d: %lld words checked, %lld wrong; error words 0x%x "
           "0x%x\n",
           rounds, dev0, dev1, (long long)words, (long long)bad, e0, e1);
    const bool pass = ok && bad == 0 && e0 == 0 && e1 == 0;
    printf("xchg2-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
