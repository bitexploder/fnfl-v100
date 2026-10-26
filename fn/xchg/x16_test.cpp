#include "fn/xchg/x16_test.h"

#include "fn/xchg/x16_ref.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace x = fn::xchg;

constexpr int32_t D = 2 * x::words_half;                               // a partial row, 2560
constexpr int32_t H = x::words_half;                                   // a half, 1280
constexpr int64_t slot_words = int64_t{x::joint_rows} * x::words_half; // a joint slot (the F32 form's size)

bool ok_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "x16-test: %s: %s\n", what, cudaGetErrorString(e));
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

float from_bits(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

// the edge values, all within FP16's range or at it: +-65,504, 65,519 (rounds to 65,504), the smallest normal, the
// largest subnormal, the smallest subnormal and the ties around it, F32 subnormals, the ties of the normal range,
// NaNs (quiet, signalling, negative), +-inf, +-0
const std::vector<float> & edge_values() {
    static const std::vector<float> v = {65504.0f,
                                         -65504.0f,
                                         65519.0f,
                                         -65519.0f,
                                         65519.996f,
                                         std::ldexp(1.0f, -14),
                                         -std::ldexp(1.0f, -14),
                                         std::ldexp(1023.0f, -24),
                                         std::ldexp(1.0f, -24),
                                         -std::ldexp(1.0f, -24),
                                         std::ldexp(1.0f, -25),
                                         std::ldexp(3.0f, -25),
                                         std::ldexp(5.0f, -25),
                                         std::ldexp(1.5f, -25),
                                         std::ldexp(1.0f, -26),
                                         from_bits(0x00000001u),
                                         from_bits(0x807fffffu),
                                         1.0f + std::ldexp(1.0f, -11),
                                         1.0f + std::ldexp(3.0f, -11),
                                         -(1.0f + std::ldexp(1.0f, -11)),
                                         2047.5f,
                                         2048.5f,
                                         2049.0f,
                                         std::ldexp(2047.0f, -25),
                                         from_bits(0x7fc00000u),
                                         from_bits(0x7f800001u),
                                         from_bits(0xffc00000u),
                                         from_bits(0x7fffffffu),
                                         INFINITY,
                                         -INFINITY,
                                         0.0f,
                                         -0.0f,
                                         std::ldexp(1.0f, -14) - std::ldexp(1.0f, -25),
                                         65503.0f,
                                         32768.0f + 16.0f,
                                         1.0f / 3.0f};
    return v;
}

// beyond FP16's range: rounds to an infinity, sets err_x16_range
const std::vector<float> & over_values() {
    static const std::vector<float> v = {65520.0f, -65520.0f, 65536.0f, 1.0e6f, -3.0e38f, FLT_MAX, -FLT_MAX};
    return v;
}

// a random value: random sign, mantissa, exponent 2^-27 .. 2^15 (FP16 zero, subnormal and normal range below
// 65,504), or every fourth an exact tie between two FP16 neighbours
float random_value(uint32_t h, uint32_t g) {
    const uint32_t sign = (g & 1u) << 31;
    if ((g & 6u) == 0u) {
        const uint16_t lo = static_cast<uint16_t>(h % 0x7bffu); // a finite half below 65,504, its next above it
        const float a = x::x16_to_f32(lo), b = x::x16_to_f32(static_cast<uint16_t>(lo + 1u));
        const float t = a + (b - a) * 0.5f; // exact: two neighbours' midpoint needs 12 significant bits
        return sign != 0u ? -t : t;
    }
    const uint32_t e = 100u + (h >> 23) % 42u; // biased F32 exponent 100 .. 141
    return from_bits(sign | (e << 23) | (h & 0x7fffffu));
}

struct card {
    int32_t dev = 0;
    float * partial = nullptr; // [rows][2560]
    float * out = nullptr;     // [rows][1280]
    uint32_t * slot = nullptr; // the joint slot this card receives on
    uint32_t * err = nullptr;  // its joint word
    cudaStream_t st = nullptr;
};

bool alloc(card & c, int32_t dev) {
    c.dev = dev;
    return ok_cuda(cudaSetDevice(dev), "device") &&
           ok_cuda(cudaMalloc(&c.partial, int64_t{x::joint_rows} * D * 4), "partial") &&
           ok_cuda(cudaMalloc(&c.out, int64_t{x::joint_rows} * H * 4), "out") &&
           ok_cuda(cudaMalloc(&c.slot, slot_words * 4), "slot") &&
           ok_cuda(cudaMemset(c.slot, 0xff, slot_words * 4), "slot fill") && ok_cuda(cudaMalloc(&c.err, 4), "err") &&
           ok_cuda(cudaMemset(c.err, 0, 4), "err fill") &&
           ok_cuda(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking), "stream") &&
           ok_cuda(cudaDeviceSynchronize(), "alloc sync");
}

void release(card & c) {
    cudaSetDevice(c.dev);
    cudaFree(c.partial);
    cudaFree(c.out);
    cudaFree(c.slot);
    cudaFree(c.err);
    if (c.st != nullptr) {
        cudaStreamDestroy(c.st);
    }
}

bool same_f32(float got, float want) {
    if (std::isnan(want)) {
        return std::isnan(got);
    }
    return std::memcmp(&got, &want, 4) == 0;
}

// FMS13: the F32 joint exchange of lanes of R rows (send_half_joint, recv_add_joint), at U = 2 .. 4 and R = 2 .. 5.
// Card c's partial is random with NaNs among the sent values and NaN in every row not sent (lane u's rows R .. 4, and
// the lanes past U); every slot word and out word not sent starts as a marker. After both sends each slot's sent rows
// (memory row 5 u + i, i < R) hold the peer's canonical bits and every other word the marker; after both receives the
// sent rows' out words are own + peer, the sent slot words the sentinel, every other word still the marker, both error
// words 0. False only when a CUDA call fails; bad counts the words not as expected.
constexpr uint32_t marker = 0x7b7b7b7bu;

bool joint_f32_lanes(card (&cd)[2], int64_t & words, int64_t & bad) {
    const int32_t         R5 = x::max_rows;
    std::vector<float>    part[2];
    std::vector<uint32_t> fill(slot_words), got(slot_words);
    std::vector<float>    out_fill(size_t(x::joint_rows) * H), got_out(size_t(x::joint_rows) * H);
    std::memcpy(out_fill.data(), std::vector<uint32_t>(out_fill.size(), marker).data(), out_fill.size() * 4);
    bool ok = true;
    for (int32_t U = 2; ok && U <= x::max_lanes; ++U) {
        for (int32_t R = 2; ok && R <= R5; ++R) {
            const auto sent = [&](int64_t row) {
                return row / R5 < U && row % R5 < R;
            };
            for (int64_t w = 0; w < slot_words; ++w) {
                fill[size_t(w)] = sent(w / H) ? x::sentinel : marker;
            }
            for (int32_t c = 0; c < 2; ++c) {
                part[c].assign(size_t(x::joint_rows) * D, 0.0f);
                for (int64_t i = 0; i < int64_t(x::joint_rows) * D; ++i) {
                    const uint32_t h   = hash(uint32_t(c * 31 + U * 7 + R), 5u, uint32_t(i));
                    part[c][size_t(i)] = !sent(i / D) || h % 997u == 0u ? std::nanf("") : random_value(h, h >> 7);
                }
                ok = ok && ok_cuda(cudaSetDevice(cd[c].dev), "device") &&
                     ok_cuda(cudaMemcpyAsync(cd[c].partial, part[c].data(), part[c].size() * 4, cudaMemcpyHostToDevice,
                                             cd[c].st),
                             "partial") &&
                     ok_cuda(cudaMemcpyAsync(cd[c].slot, fill.data(), size_t(slot_words) * 4, cudaMemcpyHostToDevice,
                                             cd[c].st),
                             "slot") &&
                     ok_cuda(cudaMemcpyAsync(cd[c].out, out_fill.data(), out_fill.size() * 4, cudaMemcpyHostToDevice,
                                             cd[c].st),
                             "out") &&
                     ok_cuda(cudaMemsetAsync(cd[c].err, 0, 4, cd[c].st), "err") &&
                     ok_cuda(cudaStreamSynchronize(cd[c].st), "fill");
            }
            for (int32_t c = 0; ok && c < 2; ++c) {
                ok = ok_cuda(cudaSetDevice(cd[c].dev), "device") &&
                     x::send_half_joint(U, R, 1 - c, cd[c].partial, reinterpret_cast<float *>(cd[1 - c].slot),
                                        cd[c].st) &&
                     ok_cuda(cudaStreamSynchronize(cd[c].st), "send");
            }
            int64_t b0 = bad;
            for (int32_t r = 0; ok && r < 2; ++r) {  // card r's slot: card 1 - r's half r
                ok =
                    ok_cuda(cudaSetDevice(cd[r].dev), "device") &&
                    ok_cuda(cudaMemcpy(got.data(), cd[r].slot, size_t(slot_words) * 4, cudaMemcpyDeviceToHost), "slot");
                for (int64_t w = 0; ok && w < slot_words; ++w) {
                    const int64_t row = w / H, q = w % H;
                    uint32_t      want = marker;
                    if (sent(row)) {
                        const float v = part[1 - r][size_t(row * D + r * H + q)];
                        std::memcpy(&want, &v, 4);
                        want = std::isnan(v) ? x::canonical_nan : want;
                    }
                    bad += got[size_t(w)] != want ? 1 : 0;
                    ++words;
                }
            }
            for (int32_t r = 0; ok && r < 2; ++r) {
                ok = ok_cuda(cudaSetDevice(cd[r].dev), "device") &&
                     x::recv_add_joint(U, R, r, cd[r].partial, reinterpret_cast<float *>(cd[r].slot), cd[r].out,
                                       cd[r].err, x::err_poll_a, cd[r].st) &&
                     ok_cuda(cudaStreamSynchronize(cd[r].st), "receive");
            }
            for (int32_t r = 0; ok && r < 2; ++r) {
                uint32_t e = 0;
                ok =
                    ok_cuda(cudaSetDevice(cd[r].dev), "device") &&
                    ok_cuda(cudaMemcpy(got_out.data(), cd[r].out, got_out.size() * 4, cudaMemcpyDeviceToHost), "out") &&
                    ok_cuda(cudaMemcpy(got.data(), cd[r].slot, size_t(slot_words) * 4, cudaMemcpyDeviceToHost),
                            "slot") &&
                    ok_cuda(cudaMemcpy(&e, cd[r].err, 4, cudaMemcpyDeviceToHost), "err");
                for (int64_t i = 0; ok && i < int64_t(got_out.size()); ++i) {
                    const int64_t row = i / H, q = i % H;
                    uint32_t      gb = 0;
                    std::memcpy(&gb, &got_out[size_t(i)], 4);
                    if (!sent(row)) {
                        bad += gb != marker ? 1 : 0;
                    } else {
                        const size_t at   = size_t(row * D + r * H + q);
                        const float  want = part[r][at] + part[1 - r][at];  // own, then the peer's (F32)
                        bad += same_f32(got_out[size_t(i)], want) ? 0 : 1;
                    }
                    bad += got[size_t(i)] != (sent(row) ? x::sentinel : marker) ? 1 : 0;  // the slot after its receive
                    words += 2;
                }
                bad += e != 0u ? 1 : 0;
            }
            printf("x16-test FMS13 F32 joint exchange U %d R %d: %lld words not as expected\n", U, R,
                   static_cast<long long>(bad - b0));
        }
    }
    return ok;
}

} // namespace

int fnfl_x16_test(int32_t dev0, int32_t dev1) {
    if (dev0 == dev1) {
        fprintf(stderr, "x16-test: DEV0 and DEV1 must differ\n");
        return 2;
    }
    if (!x::enable_peer_access(dev0, dev1)) {
        return 1;
    }
    card cd[2];
    bool ok = alloc(cd[0], dev0) && alloc(cd[1], dev1);
    int64_t halves = 0, bad_sent = 0, bad_past = 0, bad_out = 0, bad_reset = 0, bad_err = 0, n_edge = 0, n_over = 0;
    std::vector<float> part[2];
    std::vector<uint32_t> got_slot(slot_words);
    std::vector<float> got_out;
    const std::vector<float> & edge = edge_values();
    const std::vector<float> & over = over_values();
    for (int32_t U = 2; ok && U <= x::max_lanes; ++U) {
        const int32_t rows = U * x::max_rows;
        // case 0: no value beyond FP16's range (both error words 0); case 1: card s's payload holds the overflowing
        // values (only card s's word has err_x16_range), s = U % 2
        for (int32_t cs = 0; ok && cs < 2; ++cs) {
            const int32_t over_card = cs == 1 ? U % 2 : -1;
            for (int32_t c = 0; c < 2; ++c) {
                part[c].assign(int64_t(rows) * D, 0.0f);
                for (int32_t i = 0; i < rows * D; ++i) {
                    const uint32_t h = hash(uint32_t(c * 7 + U), uint32_t(cs), uint32_t(i));
                    part[c][i] = random_value(h, hash(h, 3u, uint32_t(i)));
                }
                // the sent half (the peer's, 1 - c): the edge values at the start of each row and at row ends
                const int32_t ph = 1 - c;
                for (int32_t b = 0; b < rows; ++b) {
                    for (size_t k = 0; k < edge.size(); ++k) {
                        const int32_t q = int32_t((size_t(b) * 7u + k * 37u) % size_t(H));
                        part[c][int64_t(b) * D + ph * H + q] = edge[k];
                        ++n_edge;
                    }
                    if (c == over_card) {
                        for (size_t k = 0; k < over.size(); ++k) {
                            const int32_t q = int32_t((size_t(b) * 13u + k * 101u + 5u) % size_t(H));
                            part[c][int64_t(b) * D + ph * H + q] = over[k];
                            ++n_over;
                        }
                    }
                }
                // on the card's stream, then synchronised: a pageable cudaMemcpy may return before its copy lands,
                // and the non-blocking stream's send would not wait for it
                ok = ok && ok_cuda(cudaSetDevice(cd[c].dev), "device") &&
                     ok_cuda(cudaMemcpyAsync(cd[c].partial, part[c].data(), part[c].size() * 4, cudaMemcpyHostToDevice,
                                             cd[c].st),
                             "partial") &&
                     ok_cuda(cudaStreamSynchronize(cd[c].st), "partial");
            }
            // both sends (each to the peer's slot), then the slots read back before any receive
            for (int32_t c = 0; ok && c < 2; ++c) {
                ok = ok_cuda(cudaSetDevice(cd[c].dev), "device") &&
                     x::send_half_joint16(rows, 1 - c, cd[c].partial, reinterpret_cast<float *>(cd[1 - c].slot),
                                          cd[c].err, cd[c].st) &&
                     ok_cuda(cudaStreamSynchronize(cd[c].st), "send");
            }
            for (int32_t r = 0; ok && r < 2; ++r) { // card r's slot holds card 1 - r's payload of half r
                ok = ok_cuda(cudaSetDevice(cd[r].dev), "device") &&
                     ok_cuda(cudaMemcpy(got_slot.data(), cd[r].slot, slot_words * 4, cudaMemcpyDeviceToHost), "slot");
                const std::vector<float> & src = part[1 - r];
                for (int64_t i = 0; ok && i < slot_words * 2; ++i) {
                    const uint16_t got = static_cast<uint16_t>(got_slot[size_t(i / 2)] >> (16 * (i % 2)));
                    if (i >= int64_t(rows) * x::x16_row_halves) {
                        bad_past += got != x::x16_sentinel ? 1 : 0;
                        continue;
                    }
                    const int64_t b = i / x::x16_row_halves, q = i % x::x16_row_halves;
                    bool o = false;
                    const uint16_t want = x::x16_ref(src[size_t(b * D + r * H + q)], o);
                    ++halves;
                    if (got != want) {
                        if (bad_sent < 8) {
                            fprintf(stderr,
                                    "x16-test: U %d card %d row %lld col %lld: value %a sent 0x%04x, ref "
                                    "0x%04x\n",
                                    U, 1 - r, static_cast<long long>(b), static_cast<long long>(q),
                                    double(src[size_t(b * D + r * H + q)]), got, want);
                        }
                        ++bad_sent;
                    }
                }
            }
            for (int32_t r = 0; ok && r < 2; ++r) {
                ok = ok_cuda(cudaSetDevice(cd[r].dev), "device") &&
                     x::recv_add_joint16(rows, r, cd[r].partial, reinterpret_cast<float *>(cd[r].slot), cd[r].out,
                                         cd[r].err, x::err_poll_a, cd[r].st) &&
                     ok_cuda(cudaStreamSynchronize(cd[r].st), "receive");
            }
            for (int32_t r = 0; ok && r < 2; ++r) {
                got_out.assign(int64_t(rows) * H, 0.0f);
                uint32_t e = 0;
                ok =
                    ok_cuda(cudaSetDevice(cd[r].dev), "device") &&
                    ok_cuda(cudaMemcpy(got_out.data(), cd[r].out, got_out.size() * 4, cudaMemcpyDeviceToHost), "out") &&
                    ok_cuda(cudaMemcpy(got_slot.data(), cd[r].slot, slot_words * 4, cudaMemcpyDeviceToHost), "slot") &&
                    ok_cuda(cudaMemcpy(&e, cd[r].err, 4, cudaMemcpyDeviceToHost), "err");
                for (int64_t i = 0; ok && i < int64_t(rows) * H; ++i) {
                    const int64_t b = i / H, q = i % H;
                    bool o = false;
                    const float peer = x::x16_to_f32(x::x16_ref(part[1 - r][size_t(b * D + r * H + q)], o));
                    const float want = part[r][size_t(b * D + r * H + q)] + peer; // own, then the peer's (F32)
                    bad_out += same_f32(got_out[size_t(i)], want) ? 0 : 1;
                }
                for (int64_t w = 0; ok && w < slot_words; ++w) {
                    bad_reset += got_slot[size_t(w)] != x::sentinel ? 1 : 0;
                }
                // card r's word: err_x16_range when it sent the overflowing values, else 0 (no poll code)
                const uint32_t want_err = r == over_card ? x::err_x16_range : 0u;
                if (e != want_err) {
                    fprintf(stderr, "x16-test: U %d case %d card %d: error word 0x%08x, expected 0x%08x\n", U, cs, r, e,
                            want_err);
                    ++bad_err;
                }
                ok = ok && ok_cuda(cudaMemset(cd[r].err, 0, 4), "err reset");
            }
            printf("x16-test: U %d rows %d case %s: sent halves %lld bad so far, outputs %lld, slot resets %lld, "
                   "error words %lld\n",
                   U, rows, cs == 0 ? "in range" : "overflow", static_cast<long long>(bad_sent),
                   static_cast<long long>(bad_out), static_cast<long long>(bad_reset), static_cast<long long>(bad_err));
        }
    }
    int64_t lane_words = 0, lane_bad = 0;
    ok = ok && joint_f32_lanes(cd, lane_words, lane_bad);
    printf("x16-test FMS13: the F32 joint exchange at U 2 .. 4, R 2 .. 5: %lld words checked, %lld not as expected\n",
           static_cast<long long>(lane_words), static_cast<long long>(lane_bad));
    release(cd[0]);
    release(cd[1]);
    const bool pass = ok && halves > 0 && bad_sent == 0 && bad_past == 0 && bad_out == 0 && bad_reset == 0 &&
                      bad_err == 0 && lane_words > 0 && lane_bad == 0;
    printf("x16-test: %lld halves checked (%lld edge, %lld overflowing placements); sent %lld bad, rows past %lld "
           "not the sentinel, outputs %lld bad, slot words not reset %lld, error words %lld bad\n",
           static_cast<long long>(halves), static_cast<long long>(n_edge), static_cast<long long>(n_over),
           static_cast<long long>(bad_sent), static_cast<long long>(bad_past), static_cast<long long>(bad_out),
           static_cast<long long>(bad_reset), static_cast<long long>(bad_err));
    printf("x16-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
