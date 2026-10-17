#pragma once

// PB1's prompt-kernel tests (fnfl-drive pd-test, hcp-test, gdp-test, qsp-test, pe-test, dcp-test): the small pieces
// they share. Host code only (test support, not part of the engine).

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace fn::ptest {

inline bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "prompt test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// one device allocation (poisoned to NaN bytes on request, so a missed output cannot pass)
struct dbuf {
    uint8_t * p = nullptr;
    size_t    n = 0;

    explicit dbuf(size_t bytes) : n(std::max<size_t>(bytes, 256)) {
        if (!cuda_ok(cudaMalloc(&p, n), "cudaMalloc")) {
            p = nullptr;
        }
    }

    ~dbuf() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dbuf(const dbuf &)             = delete;
    dbuf & operator=(const dbuf &) = delete;

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); }

    template <typename T> bool put(const std::vector<T> & h) const {
        return p != nullptr && h.size() * sizeof(T) <= n &&
               cuda_ok(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "put");
    }

    template <typename T> bool get(std::vector<T> & h, size_t count) const {
        h.resize(count);
        return p != nullptr && count * sizeof(T) <= n &&
               cuda_ok(cudaMemcpy(h.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "get");
    }

    bool poison() const { return p != nullptr && cuda_ok(cudaMemset(p, 0xff, n), "memset"); }

    bool zero() const { return p != nullptr && cuda_ok(cudaMemset(p, 0, n), "memset"); }
};

// splitmix64, Box-Muller: the same values on every host
struct rng {
    uint64_t s;

    explicit rng(uint64_t seed) : s(seed) {}

    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }

    double normal() {
        const double u1 = std::max(uniform(), 1e-300), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

template <typename F> void parallel_for(int64_t n, const F & fn) {
    const int32_t            nt = static_cast<int32_t>(std::max<int64_t>(1, std::min<int64_t>(n, 12)));
    std::atomic<int64_t>     next{ 0 };
    std::vector<std::thread> th;
    for (int32_t t = 0; t < nt; ++t) {
        th.emplace_back([&]() {
            for (int64_t i = next++; i < n; i = next++) {
                fn(i);
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
}

struct verdict {
    const char * tag;
    bool         all = true;
    int32_t      n   = 0;

    explicit verdict(const char * t) : tag(t) {}

    void put(const std::string & name, bool ok, const std::string & what) {
        printf("%s: %s: %s: %s\n", tag, name.c_str(), ok ? "PASS" : "FAIL", what.c_str());
        fflush(stdout);
        all = all && ok;
        ++n;
    }

    int finish() {
        printf("%s: %s (%d checks)\n", tag, all ? "PASS" : "FAIL", n);
        return all ? 0 : 1;
    }
};

inline std::string e3(double v) {
    char b[32];
    snprintf(b, sizeof(b), "%.3e", v);
    return b;
}

// max over units of max |got - ref| / max |ref| over the unit (unit = consecutive blocks of `width`); a non-finite got
// is infinite error
inline double unit_error(const float * got, const double * ref, int64_t units, int64_t width) {
    double worst = 0.0;
    for (int64_t u = 0; u < units; ++u) {
        double e = 0.0, m = 0.0;
        for (int64_t i = 0; i < width; ++i) {
            const double g = got[u * width + i], r = ref[u * width + i];
            if (!std::isfinite(g)) {
                return INFINITY;
            }
            e = std::max(e, std::fabs(g - r));
            m = std::max(m, std::fabs(r));
        }
        worst = std::max(worst, m > 0 ? e / m : e);
    }
    return worst;
}

// the launches' stream: the prompt kernels' launchers refuse the null stream (fn/model/prompt.cpp passes the compute
// streams). A blocking stream on the current device, created once; callers synchronize with cudaDeviceSynchronize
inline void * test_stream() {
    static cudaStream_t s = nullptr;
    if (s == nullptr && cudaStreamCreate(&s) != cudaSuccess) {
        s = nullptr;
    }
    return s;
}

}  // namespace fn::ptest
