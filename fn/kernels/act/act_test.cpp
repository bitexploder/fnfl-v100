#include "fn/kernels/act/act_test.h"

#include "fn/kernels/act/act.h"
#include "fn/quant/act_ref.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "act-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// owns one device allocation
struct dev_buf {
    void * p = nullptr;

    explicit dev_buf(size_t n) {
        if (!cuda_ok(cudaMalloc(&p, n), "cudaMalloc")) {
            p = nullptr;
        }
    }

    ~dev_buf() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dev_buf(const dev_buf &) = delete;
    dev_buf & operator=(const dev_buf &) = delete;
};

// one case: the kernel on x against the reference, codes and scales bitwise; prints the first mismatch
bool run_case(const std::string & name, const std::vector<float> & x, int32_t rows, int32_t k, cudaStream_t st) {
    const size_t cb = fn::act::codes_bytes(rows, k), sb = fn::act::scales_bytes(rows, k);
    std::vector<int8_t> rc(cb), gc(cb);
    std::vector<float> rs(sb / 4), gs(sb / 4);
    fn::quant::act_quantize_ref(x.data(), rows, k, rc.data(), rs.data());
    dev_buf dx(x.size() * 4), dc(cb), ds(sb);
    if (dx.p == nullptr || dc.p == nullptr || ds.p == nullptr) {
        return false;
    }
    // poison the outputs so a value the kernel does not write shows; the upload and the fills go on the kernel's
    // stream and finish before it (st is non-blocking, so it does not wait for work on the legacy stream)
    if (!cuda_ok(cudaMemcpyAsync(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice, st), "put x") ||
        !cuda_ok(cudaMemsetAsync(dc.p, 0x5a, cb, st), "memset") ||
        !cuda_ok(cudaMemsetAsync(ds.p, 0xff, sb, st), "memset") || !cuda_ok(cudaStreamSynchronize(st), "sync")) {
        return false;
    }
    const fn::act::out_view o = {static_cast<int8_t *>(dc.p), static_cast<float *>(ds.p), rows, k};
    if (!fn::act::quantize(static_cast<const float *>(dx.p), o, st)) {
        fprintf(stderr, "act-test: %s: quantize not enqueued: %s\n", name.c_str(),
                cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    if (!cuda_ok(cudaStreamSynchronize(st), "sync") ||
        !cuda_ok(cudaMemcpyAsync(gc.data(), dc.p, cb, cudaMemcpyDeviceToHost, st), "get codes") ||
        !cuda_ok(cudaMemcpyAsync(gs.data(), ds.p, sb, cudaMemcpyDeviceToHost, st), "get scales") ||
        !cuda_ok(cudaStreamSynchronize(st), "sync")) {
        return false;
    }
    for (size_t i = 0; i < cb; ++i) {
        if (gc[i] != rc[i]) {
            fprintf(stderr, "act-test: %s: FAIL: code %zu (row %zu, value %zu): kernel %d, reference %d, x %.9g\n",
                    name.c_str(), i, i / size_t(k), i % size_t(k), gc[i], rc[i], static_cast<double>(x[i]));
            return false;
        }
    }
    if (memcmp(gs.data(), rs.data(), sb) != 0) {
        for (size_t i = 0; i < gs.size(); ++i) {
            if (memcmp(&gs[i], &rs[i], 4) != 0) {
                fprintf(stderr, "act-test: %s: FAIL: run %zu %s: kernel %.9g, reference %.9g\n", name.c_str(), i / 2,
                        i % 2 == 0 ? "d" : "s", static_cast<double>(gs[i]), static_cast<double>(rs[i]));
                break;
            }
        }
        return false;
    }
    printf("act-test: %-34s B %d K %5d: %zu codes and %zu scale pairs bitwise\n", name.c_str(), rows, k, cb, sb / 8);
    return true;
}

} // namespace

int fnfl_act_test(int device) {
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaStream_t st = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) {
        return 1;
    }
    bool ok = true;
    std::mt19937 rng(0x43324143u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(-6.0f, 3.0f);
    const int32_t ks[] = {320, 2560, 3072, 6144};

    // random rows: each run its own magnitude (10^-6 .. 10^3), with an occasional outlier
    for (const int32_t k : ks) {
        for (int32_t rows = 1; rows <= 4; ++rows) {
            std::vector<float> x(size_t(rows) * size_t(k));
            for (size_t r = 0; r < x.size() / 32; ++r) {
                const float m = std::pow(10.0f, ud(rng));
                for (size_t i = 0; i < 32; ++i) {
                    x[32 * r + i] = m * nd(rng) * (rng() % 64 == 0 ? 20.0f : 1.0f);
                }
            }
            ok = run_case("random", x, rows, k, st) && ok;
        }
    }
    // all zeros: d = s = 0 and every code 0
    for (const int32_t k : ks) {
        ok = run_case("zeros", std::vector<float>(size_t(3) * size_t(k), 0.0f), 3, k, st) && ok;
    }
    // one large value a run (alternating sign), the rest small: the small values round to 0 or +-1
    for (const int32_t k : ks) {
        std::vector<float> x(size_t(4) * size_t(k));
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = 1e-3f * nd(rng);
        }
        for (size_t r = 0; r < x.size() / 32; ++r) {
            x[32 * r + (r * 7) % 32] = (r % 2 == 0 ? 3.0e4f : -2.5e4f) * (1.0f + 0.01f * float(r % 13));
        }
        ok = run_case("one large value", x, 4, k, st) && ok;
    }
    // midpoints: amax 127 gives id = 1 exactly, so the other values j + 0.5 are ties (to even)
    for (const int32_t k : ks) {
        std::vector<float> x(size_t(2) * size_t(k));
        for (size_t r = 0; r < x.size() / 32; ++r) {
            x[32 * r] = (r % 2 == 0) ? 127.0f : -127.0f;
            for (size_t i = 1; i < 32; ++i) {
                const float j = static_cast<float>((r * 31 + i * 5) % 254) - 127.0f; // -127 .. 126
                x[32 * r + i] = j + 0.5f;
            }
        }
        ok = run_case("midpoints", x, 2, k, st) && ok;
    }
    // midpoints at another scale: amax 254 gives id = 0.5 exactly, values 2 j + 1 land on j + 0.5
    for (const int32_t k : ks) {
        std::vector<float> x(size_t(1) * size_t(k));
        for (size_t r = 0; r < x.size() / 32; ++r) {
            x[32 * r + 31] = 254.0f;
            for (size_t i = 0; i < 31; ++i) {
                x[32 * r + i] = 2.0f * static_cast<float>((r * 17 + i * 3) % 253) - 252.0f + 1.0f;
            }
        }
        ok = run_case("midpoints, id 0.5", x, 1, k, st) && ok;
    }
    // the arguments outside the contract are rejected with nothing enqueued
    {
        dev_buf b(1 << 16);
        int8_t * c = static_cast<int8_t *>(b.p);
        float * f = static_cast<float *>(b.p);
        const bool rej = !fn::act::quantize(f, {c, f, 0, 32}, st) && !fn::act::quantize(f, {c, f, 1, 48}, st) &&
                         !fn::act::quantize(f, {c, f, 1, 32}, nullptr) &&
                         !fn::act::quantize(f + 1, {c, f, 1, 32}, st) && !fn::act::quantize(f, {c + 8, f, 1, 32}, st) &&
                         !fn::act::quantize(nullptr, {c, f, 1, 32}, st);
        printf("act-test: rejected arguments: %s\n", rej ? "all six rejected" : "FAIL");
        ok = ok && rej && cuda_ok(cudaStreamSynchronize(st), "sync");
    }
    cudaStreamDestroy(st);
    printf("act-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
