#include "fn/kernels/gdn/gdnj_test.h"

#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/round/version.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace {

namespace gd = fn::gdn;

constexpr int32_t R             = gd::max_rows;  // the buffers' rows (a lane's five-row stride)
constexpr int32_t U_MAX = fn::xchg::max_lanes;
// FMS13's sentinel test: at rr < 5 rows, rows rr .. 4 of each input are NaN and of each output this byte pattern
constexpr uint8_t sentinel_byte = 0x7b;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "gdnj-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// one device allocation and its initial bytes (uploaded by fresh)
struct buf {
    void * p = nullptr;
    size_t n = 0;
    std::vector<uint8_t> init;

    ~buf() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }
};

template <typename T>
T * add(std::vector<std::unique_ptr<buf>> & bs, size_t count, const std::vector<T> & init, bool & ok) {
    auto x = std::make_unique<buf>();
    x->n = count * sizeof(T);
    x->init.assign(x->n, 0);
    if (!init.empty()) {
        std::memcpy(x->init.data(), init.data(), std::min(x->n, init.size() * sizeof(T)));
    }
    ok = ok && cuda_ok(cudaMalloc(&x->p, x->n), "cudaMalloc");
    T * p = static_cast<T *>(x->p);
    bs.push_back(std::move(x));
    return p;
}

std::vector<float> uniform(std::mt19937 & g, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float & x : v) {
        x = d(g);
    }
    return v;
}

// a lane: every buffer core() reads or writes, in b, so a snapshot holds every byte its launch can write
struct lane {
    std::vector<std::unique_ptr<buf>> b;
    gd::core_args a;
};

struct shared {
    std::vector<std::unique_ptr<buf>> b;
    float * dt = nullptr;
    float * a = nullptr;
    float * norm_w = nullptr;
};

bool make_shared(shared & w) {
    std::mt19937 g(11);
    bool ok = true;
    w.dt = add(w.b, gd::n_head, uniform(g, gd::n_head, -1.0f, 1.0f), ok);
    w.a = add(w.b, gd::n_head, uniform(g, gd::n_head, -2.0f, -0.1f), ok); // -exp(A_log)
    w.norm_w = add(w.b, gd::head_dim, uniform(g, gd::head_dim, 0.5f, 1.5f), ok);
    return ok;
}

// rows rr .. R - 1 of a [R][width] input set to NaN (FMS13: a launch reading them spreads NaN into rows < rr)
std::vector<float> nan_tail(std::vector<float> v, int32_t rr, size_t width) {
    std::fill(v.begin() + ptrdiff_t(size_t(rr) * width), v.end(), std::numeric_limits<float>::quiet_NaN());
    return v;
}

// lane k at rr rows (FMS13: 2 .. 5): random inputs, banks and both record sets (each step's decay in 0.5 .. 1), its
// selector gsel; rows rr .. 4 of the inputs NaN, every output byte the sentinel
bool make_lane(lane & l, const shared & w, int32_t k, uint32_t gsel, int32_t rr) {
    std::mt19937 g(100u + 17u * static_cast<uint32_t>(k));
    bool ok = true;
    std::vector<float> rec = uniform(g, size_t(gd::n_rec_set * gd::rec_set_floats), -0.5f, 0.5f);
    for (int64_t s = 0; s < int64_t(gd::n_rec_set) * R; ++s) {
        float * d = rec.data() + s * gd::rec_step_floats + gd::rec_decay;
        const std::vector<float> dv = uniform(g, gd::n_head, 0.5f, 1.0f);
        std::copy(dv.begin(), dv.end(), d);
    }
    const size_t banks = size_t(gd::n_bank * gd::state_version_floats);
    gd::core_args & a = l.a;
    a.conv                = add(l.b, size_t(R) * gd::n_channel,
                                nan_tail(uniform(g, size_t(R) * gd::n_channel, -1.0f, 1.0f), rr, gd::n_channel), ok);
    a.z = add(l.b, size_t(R) * gd::n_value, nan_tail(uniform(g, size_t(R) * gd::n_value, -2.0f, 2.0f), rr, gd::n_value),
              ok);
    a.alpha =
        add(l.b, size_t(R) * gd::n_head, nan_tail(uniform(g, size_t(R) * gd::n_head, -2.0f, 2.0f), rr, gd::n_head), ok);
    a.beta =
        add(l.b, size_t(R) * gd::n_head, nan_tail(uniform(g, size_t(R) * gd::n_head, -2.0f, 2.0f), rr, gd::n_head), ok);
    a.dt = w.dt;
    a.a = w.a;
    a.norm_w = w.norm_w;
    a.state = add(l.b, banks, uniform(g, banks, -0.2f, 0.2f), ok);
    a.rec = add(l.b, rec.size(), rec, ok);
    a.gsel = add(l.b, 4, std::vector<uint32_t>{gsel}, ok);
    a.gated = add(l.b, size_t(R) * gd::n_value, std::vector<float>(), ok);
    l.b.back()->init.assign(l.b.back()->n, sentinel_byte);
    a.gated_q.codes = add(l.b, size_t(R) * gd::n_value, std::vector<int8_t>(), ok);
    l.b.back()->init.assign(l.b.back()->n, sentinel_byte);
    a.gated_q.scales = add(l.b, size_t(R) * (gd::n_value / 32) * 2, std::vector<float>(), ok); // {d, s} pairs
    l.b.back()->init.assign(l.b.back()->n, sentinel_byte);
    a.gated_q.rows = rr;
    a.gated_q.k = gd::n_value;
    a.ws = add(l.b, gd::workspace_bytes(), std::vector<uint8_t>(), ok);
    a.rows         = rr;
    return ok;
}

// FMS13: rows rr .. 4 of the lane's outputs (gated, its codes and scales) still the sentinel, its NaN input rows
// unchanged, in a snapshot (make_lane's buffer order: conv, z, alpha, beta, state, rec, gsel, gated, codes, scales)
bool rows_untouched(const lane & l, const std::vector<uint8_t> & snap, int32_t rr) {
    size_t at = 0;
    bool   ok = true;
    for (size_t i = 0; i < l.b.size(); ++i) {
        const size_t n   = l.b[i]->n;
        const bool   out = i >= 7 && i <= 9, in = i <= 3;
        if (out || in) {
            const size_t tail = n / size_t(R) * size_t(rr);  // each of these is [R][width]
            ok                = ok && std::equal(snap.begin() + ptrdiff_t(at + tail), snap.begin() + ptrdiff_t(at + n),
                                                 l.b[i]->init.begin() + ptrdiff_t(tail));
        }
        at += n;
    }
    return ok;
}

bool fresh(const std::vector<std::unique_ptr<lane>> & ln, int32_t n) {
    bool ok = cuda_ok(cudaDeviceSynchronize(), "sync");
    for (int32_t k = 0; k < n && ok; ++k) {
        for (const auto & x : ln[size_t(k)]->b) {
            ok = ok && cuda_ok(cudaMemcpy(x->p, x->init.data(), x->n, cudaMemcpyHostToDevice), "fresh");
        }
    }
    return ok && cuda_ok(cudaDeviceSynchronize(), "sync");
}

bool snapshot(const lane & l, std::vector<uint8_t> & out) {
    out.clear();
    bool ok = cuda_ok(cudaDeviceSynchronize(), "sync") && cuda_ok(cudaGetLastError(), "launch");
    for (const auto & x : l.b) {
        const size_t at = out.size();
        out.resize(at + x->n);
        ok = ok && cuda_ok(cudaMemcpy(out.data() + at, x->p, x->n, cudaMemcpyDeviceToHost), "snapshot");
    }
    return ok;
}

// on a mismatch: each differing buffer of the lane (its index in make_lane's order) and its differing bytes
void diff(const lane & l, const std::vector<uint8_t> & got, const std::vector<uint8_t> & want, int32_t k) {
    size_t at = 0;
    for (size_t i = 0; i < l.b.size(); ++i) {
        const size_t n = l.b[i]->n;
        size_t bad = 0;
        for (size_t j = 0; j < n && at + j < got.size() && at + j < want.size(); ++j) {
            bad += got[at + j] != want[at + j] ? 1 : 0;
        }
        if (bad != 0) {
            printf("gdnj-test   lane %d buffer %zu: %zu of %zu bytes differ\n", k, i, bad, n);
        }
        at += n;
    }
}

} // namespace

int fnfl_gdnj_test(int dev) {
    bool ok = cuda_ok(cudaSetDevice(dev), "cudaSetDevice");
    cudaStream_t st = nullptr;
    ok = ok && cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    bool pass = ok;
    {
        shared w;
        ok = ok && make_shared(w);
        // each lane a different bank, record set and replay count
        const uint32_t     gsel[U_MAX] = { fn::round::gdn_word(0, 0, 0), fn::round::gdn_word(1, 0, 3),
                                           fn::round::gdn_word(0, 1, 5), fn::round::gdn_word(1, 1, 2) };
        gd::core_resources r;
        for (int32_t U = 2; U <= U_MAX && ok; ++U) {
            if (gd::lanes_resources(U, r)) {
                printf("gdnj-test core_lanes_kernel<5> U%d: %d threads, %d blocks, %d blocks an SM, %d registers, %d "
                       "local bytes, %zu shared bytes\n",
                       U, r.threads, r.blocks, r.blocks_per_sm, r.regs, r.local_bytes, r.shared_bytes);
            }
        }
        // FMS13: every case at R = 2 .. 5 rows a lane against the lone core() at B = R; below 5 the sentinel test
        std::vector<std::unique_ptr<lane>> ln;
        gd::core_args a[U_MAX];
        for (int32_t rr = 2; rr <= R && ok; ++rr) {
            ln.clear();
            for (int32_t k = 0; k < U_MAX && ok; ++k) {
                ln.push_back(std::make_unique<lane>());
                ok = make_lane(*ln.back(), w, k, gsel[k], rr);
            }
            std::vector<uint8_t> want[U_MAX];
            for (int32_t k = 0; k < U_MAX && ok; ++k) {
                ok = fresh(ln, U_MAX) && gd::core(ln[size_t(k)]->a, st) && snapshot(*ln[size_t(k)], want[k]);
            }
            // the lone runs repeat bitwise (the reference is deterministic)
            int32_t lone_same = 0;
            for (int32_t k = 0; k < U_MAX && ok; ++k) {
                std::vector<uint8_t> again;
                ok = fresh(ln, U_MAX) && gd::core(ln[size_t(k)]->a, st) && snapshot(*ln[size_t(k)], again);
                lone_same += ok && again == want[k] ? 1 : 0;
                if (ok && again != want[k]) {
                    diff(*ln[size_t(k)], again, want[k], k);
                }
            }
            printf("gdnj-test R%d lone core() runs repeated: %d of %d lanes bitwise: %s\n", rr, lone_same, U_MAX,
                   ok && lone_same == U_MAX ? "yes" : "NO  FAIL");
            pass = pass && ok && lone_same == U_MAX;
            for (int32_t k = 0; k < U_MAX; ++k) {
                a[k] = ln[size_t(k)]->a;
            }
            for (int32_t U = 2; U <= U_MAX && ok; ++U) {
                for (int32_t rep = 0; rep < (U == U_MAX ? 2 : 1) && ok; ++rep) {
                    ok           = fresh(ln, U) && gd::core_lanes(U, a, st);
                    int32_t same = 0, kept = 0;
                    size_t  bytes = 0;
                    for (int32_t k = 0; k < U && ok; ++k) {
                        std::vector<uint8_t> got;
                        ok = snapshot(*ln[size_t(k)], got);
                        same += ok && got == want[k] ? 1 : 0;
                        kept += ok && rows_untouched(*ln[size_t(k)], got, rr) ? 1 : 0;
                        bytes += got.size();
                        if (ok && got != want[k]) {
                            diff(*ln[size_t(k)], got, want[k], k);
                        }
                    }
                    const bool good = ok && same == U && kept == U;
                    printf(
                        "gdnj-test R%d U%d%s: %d of %d lanes bitwise their core() runs (%zu bytes), rows %d .. 4 "
                        "untouched in %d: %s\n",
                        rr, U, rep == 1 ? ", second joint run" : "", same, U, bytes, rr, kept,
                        good ? "yes" : "NO  FAIL");
                    pass = pass && good;
                }
            }
        }
        // rejects (the lanes at R = 5): lanes 1 and 5, a lane at fewer rows, a null pointer, a null stream, every lane
        // at one row (FMS13); nothing enqueued
        std::vector<uint8_t> before[U_MAX];
        ok = ok && fresh(ln, U_MAX);
        for (int32_t k = 0; k < U_MAX && ok; ++k) {
            ok = snapshot(*ln[size_t(k)], before[k]);
        }
        gd::core_args b[U_MAX];
        std::copy_n(a, U_MAX, b);
        b[1].rows = 3;
        b[1].gated_q.rows = 3;
        gd::core_args c[U_MAX];
        std::copy_n(a, U_MAX, c);
        c[2].gsel = nullptr;
        gd::core_args o[U_MAX];
        std::copy_n(a, U_MAX, o);
        for (gd::core_args & x : o) {
            x.rows         = 1;
            x.gated_q.rows = 1;
        }
        const bool launched = gd::core_lanes(1, a, st) || gd::core_lanes(5, a, st) || gd::core_lanes(2, b, st) ||
                              gd::core_lanes(3, c, st) || gd::core_lanes(2, a, nullptr) ||
                              gd::core_lanes(2, nullptr, st) || gd::core_lanes(2, o, st);
        int32_t same = 0;
        for (int32_t k = 0; k < U_MAX && ok; ++k) {
            std::vector<uint8_t> after;
            ok = snapshot(*ln[size_t(k)], after);
            same += ok && after == before[k] ? 1 : 0;
        }
        printf("gdnj-test rejects: 7 refused, every lane untouched: %s\n",
               ok && !launched && same == U_MAX ? "yes" : "NO  FAIL");
        pass = pass && ok && !launched && same == U_MAX;
        cudaDeviceSynchronize();
    }
    if (st != nullptr) {
        cudaStreamDestroy(st);
    }
    printf("gdnj-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
