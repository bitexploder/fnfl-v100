#include "fn/kernels/experts/experts_calib.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

// Calibration kernels for the experts sheet's C1-C3. Each launches 72 blocks of 1,024 threads (32 warps, one block an
// SM by the 50 KiB of dynamic shared memory each requests) and times its loop inside the block with clock64(), so the
// rates come out per clock whatever the SM clock was. Built with the experts kernels' flags (no fast math).

namespace {

constexpr int32_t n_sm = 72;
constexpr int32_t n_threads = 1024;
constexpr int32_t n_warps = n_threads / 32;
constexpr int32_t dyn_smem = 50 * 1024;
constexpr int32_t n_words = 8;   // C1: index words a thread, 4 lookups each, so 32 lookups an iteration
constexpr int32_t n_chain = 8;   // C2/C3: independent dependency chains a thread and opcode
constexpr int32_t per_it = 512;  // C2/C3: instructions an iteration (both opcodes together for a pair)
constexpr int32_t n_launch = 20; // timed launches a point, after 3 untimed
constexpr double warm_s = 2.0;   // HX12 Later 15: each process starts at 1,260 MHz for 0.3-1 s

struct block_time {
    long long cycles;
    unsigned long long ns;
    uint32_t smid;
    uint32_t pad;
};

__device__ __forceinline__ uint32_t smid() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long r;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
    return r;
}

// Three-input add as two PTX adds, so ptxas emits IADD3 (HX12 M5: with two inputs it moves some to IMAD.IADD).
__device__ __forceinline__ uint32_t iadd3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm volatile("{\n\t.reg .u32 t;\n\tadd.u32 t, %1, %2;\n\tadd.u32 %0, t, %3;\n\t}"
                 : "=r"(r)
                 : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ uint32_t popc(uint32_t a) {
    uint32_t r;
    asm volatile("popc.b32 %0, %1;" : "=r"(r) : "r"(a));
    return r;
}

__device__ __forceinline__ uint32_t idp4a(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ void time_begin(long long & c0, unsigned long long & t0) {
    __syncthreads();
    c0 = clock64();
    t0 = gtime();
}

__device__ __forceinline__ void time_end(long long c0, unsigned long long t0, block_time * bt) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const long long c1 = clock64();
        const unsigned long long t1 = gtime();
        bt[blockIdx.x] = {c1 - c0, t1 - t0, smid(), 0};
    }
}

template <typename E> __device__ __forceinline__ uint32_t fold(E v);
template <> __device__ __forceinline__ uint32_t fold<uint32_t>(uint32_t v) {
    return v;
}
template <> __device__ __forceinline__ uint32_t fold<uint2>(uint2 v) {
    return v.x ^ v.y;
}
template <> __device__ __forceinline__ uint32_t fold<uint4>(uint4 v) {
    return v.x ^ v.y ^ v.z ^ v.w;
}

// C1. Contract: 72 x 1,024 threads, dyn_smem bytes of dynamic shared memory; the first table_bytes hold the table
// (copied from tab, entries of sizeof(E) bytes, a power of two in count). words: n_words a thread. Each iteration
// unpacks 4 indices from each word (shifts 0, 7, 14, 21 under the entry-count mask) and folds the entry (LOOKUP) or
// the index itself; a warp-uniform add to each word an iteration keeps every lookup loop-variant (HX12 M6).
template <typename E, bool LOOKUP>
__global__ void __launch_bounds__(n_threads, 1) lookup_kernel(const uint32_t * words, const uint4 * tab,
                                                              int32_t table_bytes, int32_t iters, uint32_t step,
                                                              uint32_t * out, block_time * bt) {
    extern __shared__ uint4 tab_s[];
    for (int32_t i = threadIdx.x; i < table_bytes / 16; i += n_threads) {
        tab_s[i] = tab[i];
    }
    const E * t = reinterpret_cast<const E *>(tab_s); // the table's entries, E bytes each
    const uint32_t mask = static_cast<uint32_t>(table_bytes / static_cast<int32_t>(sizeof(E))) - 1u;
    const uint32_t tid = blockIdx.x * n_threads + threadIdx.x;
    uint32_t w[n_words];
#pragma unroll
    for (int32_t i = 0; i < n_words; ++i) {
        w[i] = words[static_cast<size_t>(tid) * n_words + i];
    }
    long long c0;
    unsigned long long t0;
    time_begin(c0, t0);
    uint32_t acc = 0;
    for (int32_t it = 0; it < iters; ++it) {
#pragma unroll
        for (int32_t i = 0; i < n_words; ++i) {
#pragma unroll
            for (int32_t b = 0; b < 4; ++b) {
                const uint32_t idx = (w[i] >> (7 * b)) & mask;
                acc ^= LOOKUP ? fold<E>(t[idx]) : idx;
            }
            w[i] += step;
        }
    }
    out[tid] = acc;
    time_end(c0, t0, bt);
}

enum class op { popc, popc_iadd3, idp4a_ffma, idp4a_iadd3 };

// C2/C3. Contract: 72 x 1,024 threads. n_chain independent chains of each opcode a thread; per_it instructions an
// iteration, the two opcodes of a pair interleaved 1:1. k and kf are runtime constants so nothing folds.
template <op OP>
__global__ void __launch_bounds__(n_threads, 1) issue_kernel(int32_t iters, uint32_t k, float kf, uint32_t * out,
                                                             block_time * bt) {
    const uint32_t tid = blockIdx.x * n_threads + threadIdx.x;
    uint32_t p[n_chain], q[n_chain];
    float f[n_chain];
#pragma unroll
    for (int32_t c = 0; c < n_chain; ++c) {
        p[c] = tid * 2654435761u + static_cast<uint32_t>(c);
        q[c] = tid ^ (0x9e3779b9u * static_cast<uint32_t>(c + 1));
        f[c] = 1.0f + 1e-3f * static_cast<float>(c);
    }
    long long c0;
    unsigned long long t0;
    time_begin(c0, t0);
    constexpr int32_t steps = OP == op::popc ? per_it / n_chain : per_it / (2 * n_chain);
    for (int32_t it = 0; it < iters; ++it) {
#pragma unroll
        for (int32_t s = 0; s < steps; ++s) {
#pragma unroll
            for (int32_t c = 0; c < n_chain; ++c) {
                if constexpr (OP == op::popc) {
                    p[c] = popc(p[c]);
                } else if constexpr (OP == op::popc_iadd3) {
                    p[c] = popc(p[c]);
                    q[c] = iadd3(q[c], p[c], k);
                } else if constexpr (OP == op::idp4a_ffma) {
                    p[c] = idp4a(q[c], k, p[c]);
                    f[c] = fmaf(kf, f[c], kf);
                } else {
                    p[c] = idp4a(q[c], k, p[c]);
                    q[c] = iadd3(q[c], p[c], k);
                }
            }
        }
    }
    uint32_t r = 0;
#pragma unroll
    for (int32_t c = 0; c < n_chain; ++c) {
        r ^= p[c] ^ q[c] ^ __float_as_uint(f[c]);
    }
    out[tid] = r;
    time_end(c0, t0, bt);
}

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "ex-calib: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

template <typename T> struct dev_buf {
    T * p = nullptr;

    explicit dev_buf(size_t n) {
        if (!cuda_ok(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc")) {
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

struct stats {
    double cycles_median;
    double cycles_p90;
    double mhz;
    int32_t distinct_sm;
};

// Runs launch() 3 times untimed and n_launch times timed; the statistics are over every block of the timed launches.
template <typename F> bool measure(cudaStream_t st, block_time * bt_dev, const F & launch, stats & s) {
    std::vector<double> cyc;
    std::vector<double> mhz;
    std::vector<int32_t> sms;
    std::vector<block_time> h(n_sm);
    for (int32_t i = 0; i < 3 + n_launch; ++i) {
        launch();
        if (!cuda_ok(cudaGetLastError(), "launch") || !cuda_ok(cudaStreamSynchronize(st), "sync")) {
            return false;
        }
        if (i < 3) {
            continue;
        }
        if (!cuda_ok(cudaMemcpy(h.data(), bt_dev, sizeof(block_time) * n_sm, cudaMemcpyDeviceToHost), "copy")) {
            return false;
        }
        for (const block_time & b : h) {
            cyc.push_back(static_cast<double>(b.cycles));
            mhz.push_back(1e3 * static_cast<double>(b.cycles) / static_cast<double>(b.ns));
            sms.push_back(static_cast<int32_t>(b.smid));
        }
    }
    std::sort(cyc.begin(), cyc.end());
    std::sort(mhz.begin(), mhz.end());
    std::sort(sms.begin(), sms.end());
    s.cycles_median = cyc[cyc.size() / 2];
    s.cycles_p90 = cyc[(cyc.size() * 9) / 10];
    s.mhz = mhz[mhz.size() / 2];
    s.distinct_sm = static_cast<int32_t>(std::unique(sms.begin(), sms.end()) - sms.begin());
    return true;
}

template <typename E, bool LOOKUP> bool set_smem() {
    return cuda_ok(
        cudaFuncSetAttribute(lookup_kernel<E, LOOKUP>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem),
        "smem attribute");
}

struct ctx {
    cudaStream_t st = nullptr;
    dev_buf<uint32_t> words{size_t(n_sm) * n_threads * n_words};
    dev_buf<uint4> tab{8192 / 16};
    dev_buf<uint32_t> out{size_t(n_sm) * n_threads};
    dev_buf<block_time> bt{size_t(n_sm)};
};

// C1 prediction (experts sheet): an 8-byte entry at about half and a 16-byte entry at about a quarter of the 4-byte
// rate; the 4-byte rate is HX12 M6's 0.311 warp lookups a clock an SM (whole loop, 1 KB table)
template <typename E> bool c1_point(ctx & c, int32_t table_bytes) {
    constexpr int32_t iters = 128;
    const uint32_t step = 0x01010101u * 37u;
    stats with = {}, without = {};
    bool ok = set_smem<E, true>() && set_smem<E, false>();
    ok = ok && measure(
                   c.st, c.bt.p,
                   [&]() {
                       lookup_kernel<E, true><<<n_sm, n_threads, dyn_smem, c.st>>>(c.words.p, c.tab.p, table_bytes,
                                                                                   iters, step, c.out.p, c.bt.p);
                   },
                   with);
    ok = ok && measure(
                   c.st, c.bt.p,
                   [&]() {
                       lookup_kernel<E, false><<<n_sm, n_threads, dyn_smem, c.st>>>(c.words.p, c.tab.p, table_bytes,
                                                                                    iters, step, c.out.p, c.bt.p);
                   },
                   without);
    if (!ok) {
        return false;
    }
    const double lookups = static_cast<double>(n_warps) * iters * n_words * 4; // warp lookups an SM
    const double rate = lookups / with.cycles_median;
    const double diff = (with.cycles_median - without.cycles_median) / lookups;
    const double pred = sizeof(E) == 4 ? 0.311 : sizeof(E) == 8 ? 0.311 / 2 : 0.311 / 4;
    printf("{\"ticket\": \"EX1\", \"point\": \"C1_e%d_t%d\", \"what\": \"random shared-memory lookup\", "
           "\"entry_bytes\": %d, \"table_bytes\": %d, \"warps_per_sm\": %d, \"warp_lookups_per_sm\": %.0f, "
           "\"cycles_median\": %.0f, \"cycles_p90\": %.0f, \"cycles_nolookup_median\": %.0f, "
           "\"lookups_per_clk_sm\": %.4f, \"cycles_per_lookup_by_difference\": %.3f, "
           "\"prediction_lookups_per_clk_sm\": %.4f, \"ratio\": %.3f, \"sm_mhz\": %.1f, \"distinct_sm\": %d, "
           "\"label\": \"measured\"}\n",
           static_cast<int>(sizeof(E)), table_bytes, static_cast<int>(sizeof(E)), table_bytes, n_warps, lookups,
           with.cycles_median, with.cycles_p90, without.cycles_median, rate, diff, pred, rate / pred, with.mhz,
           with.distinct_sm);
    fflush(stdout);
    return true;
}

template <op OP> bool issue_point(ctx & c, const char * name, double pred, const char * pred_note) {
    constexpr int32_t iters = 64;
    stats s = {};
    if (!measure(
            c.st, c.bt.p,
            [&]() { issue_kernel<OP><<<n_sm, n_threads, 0, c.st>>>(iters, 0x01020304u, 1e-3f, c.out.p, c.bt.p); }, s)) {
        return false;
    }
    const double instr = static_cast<double>(n_warps) * iters * per_it; // warp instructions an SM, both opcodes
    const double rate = instr / s.cycles_median;
    printf("{\"ticket\": \"EX1\", \"point\": \"%s\", \"what\": \"issue rate\", \"warps_per_sm\": %d, "
           "\"warp_instr_per_sm\": %.0f, \"cycles_median\": %.0f, \"cycles_p90\": %.0f, "
           "\"warp_instr_per_clk_sm\": %.4f, \"prediction\": %.3f, \"prediction_note\": \"%s\", \"ratio\": %.3f, "
           "\"sm_mhz\": %.1f, \"distinct_sm\": %d, \"label\": \"measured\"}\n",
           name, n_warps, instr, s.cycles_median, s.cycles_p90, rate, pred, pred_note, rate / pred, s.mhz,
           s.distinct_sm);
    fflush(stdout);
    return true;
}

} // namespace

int fnfl_ex_calib(int device) {
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaDeviceProp prop = {};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "properties")) {
        return 1;
    }
    printf("# ex-calib: %s, sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    if (prop.multiProcessorCount != n_sm) {
        fprintf(stderr, "ex-calib: expected %d SMs\n", n_sm);
        return 1;
    }
    bool ok = true;
    {
        ctx c;
        ok = c.words.p != nullptr && c.tab.p != nullptr && c.out.p != nullptr && c.bt.p != nullptr;
        ok = ok && cuda_ok(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking), "stream");
        if (ok) {
            std::mt19937 rng(0x45583143u);
            std::vector<uint32_t> w(size_t(n_sm) * n_threads * n_words);
            for (auto & v : w) {
                v = rng();
            }
            std::vector<uint32_t> t(8192 / 4);
            for (auto & v : t) {
                v = rng();
            }
            ok = cuda_ok(cudaMemcpy(c.words.p, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "words") &&
                 cuda_ok(cudaMemcpy(c.tab.p, t.data(), t.size() * 4, cudaMemcpyHostToDevice), "table") &&
                 cuda_ok(cudaDeviceSynchronize(), "table"); // pageable copies may return before the DMA lands
        }
        // keep the card busy for warm_s first, with the C1 kernel itself (8 KB, 4-byte entries)
        if (ok) {
            ok = set_smem<uint32_t, true>();
            const auto t0 = std::chrono::steady_clock::now();
            int64_t n = 0;
            while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
                for (int32_t i = 0; i < 16; ++i, ++n) {
                    lookup_kernel<uint32_t, true>
                        <<<n_sm, n_threads, dyn_smem, c.st>>>(c.words.p, c.tab.p, 8192, 1024, 37u, c.out.p, c.bt.p);
                }
                ok = cuda_ok(cudaGetLastError(), "warm launch") && cuda_ok(cudaStreamSynchronize(c.st), "warm sync");
            }
            printf("# ex-calib: busy for %.2f s (%lld launches) before timing\n",
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                   static_cast<long long>(n));
        }
        for (const int32_t tb : {2048, 4096, 8192}) {
            ok = ok && c1_point<uint32_t>(c, tb) && c1_point<uint2>(c, tb) && c1_point<uint4>(c, tb);
        }
        ok = ok && issue_point<op::popc>(c, "C2_popc", 0.5, "0.5 a clock an SM, own pipe (experts sheet C2)");
        ok = ok && issue_point<op::popc_iadd3>(c, "C2_popc_iadd3", 1.0,
                                               "both opcodes together: POPC at 0.5 with IADD3 unslowed, 1:1 mix");
        ok = ok && issue_point<op::idp4a_ffma>(c, "C3_idp4a_ffma", 2.0,
                                               "both together: 2.0 if IDP4A shares the FP32 pipe (experts sheet C3)");
        ok = ok && issue_point<op::idp4a_iadd3>(c, "C3_idp4a_iadd3", 4.0,
                                                "both together: 4.0 if IDP4A and IADD3 are on different pipes");
        ok = cuda_ok(cudaStreamSynchronize(c.st), "final sync") && ok;
        if (c.st != nullptr) {
            cudaStreamDestroy(c.st);
        }
    }
    printf("# ex-calib: %s\n", ok ? "done" : "FAILED");
    return ok ? 0 : 1;
}
