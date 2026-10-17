#include "fn/kernels/gdn/gdn_bench.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/proto/xp1/timing.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace {

namespace gd = fn::gdn;

constexpr int32_t n_set = 136;            // layer-sized sets rotated over (each its weights, state, history, I/O)
constexpr int32_t n_cold = 64;            // launches a cold replay, each after a flush and the evictor
constexpr int32_t n_warm = 3;             // untimed replays
constexpr int32_t n_timed = 20;           // timed replays
constexpr double warm_s = 2.0;            // HX12 Later 15: a process starts at 1,260 MHz for 0.3-1 s
constexpr double hw_gbs = 1021.6;         // HX2 B1
constexpr double ramp_us = 1.949;         // HX2 B1's fixed term
constexpr int64_t flush_bytes = 12 << 20; // the L2 flush (XP1 Amendment 4)
constexpr int32_t n_block_probe = gd::n_head * gd::n_split;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "gd-bench: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm = {};
    gmtime_r(&t, &tm);
    char buf[64];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms));
    return buf;
}

// owns one device allocation
struct dev_buf {
    uint8_t * p = nullptr;

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

// Copied from XP1's icache_evict (fn/proto/xp1/timing.cu on branch xp1-attr, 3530eae; XP1 Amendment 4): about 12,288
// distinct FFMA instructions in a row (about 192 KB of code), one warp on every SM, so each SM's instruction caches
// hold this code and not the previous kernel's. Measurement scaffolding only.
template <int N> __device__ __forceinline__ float chain(float a) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        a = fmaf(a, 1.0f + float(i % 97) * 1e-7f, float(i) * 1e-9f);
    }
    return a;
}

__global__ void __launch_bounds__(32) icache_kernel(float * out) {
    const float a = chain<12288>(float(threadIdx.x));
    if (a == 123.0f) {
        out[0] = a; // practically never taken; keeps the chain
    }
}

bool icache_evict(float * out, cudaStream_t st) {
    icache_kernel<<<160, 32, 0, st>>>(out);
    return cudaPeekAtLastError() == cudaSuccess;
}

// bytes a launch moves at B rows (gdn.cu's floor, Amendment 2b): reads 1,572,864 + B 45,248 + 704, writes
// B 1,601,280
int64_t read_bytes(int32_t rows) {
    return gd::state_version_floats * 4 + int64_t(rows) * 45248 + 704;
}

int64_t write_bytes(int32_t rows) {
    return int64_t(rows) * 1601280;
}

int64_t launch_bytes(int32_t rows) {
    return read_bytes(rows) + write_bytes(rows);
}

double floor_us(int32_t rows) {
    return ramp_us + double(launch_bytes(rows)) / (hw_gbs * 1e3);
}

// the sets: one allocation, each set's buffers 256-byte aligned inside it
struct set_layout {
    size_t qkv, z, alpha, beta, conv, dt, a, norm, state, hist, gated, codes, scales, bytes;
};

set_layout layout() {
    const auto up = [](size_t n) { return (n + 255) / 256 * 256; };
    set_layout l = {};
    size_t o = 0;
    const auto take = [&](size_t n) {
        const size_t at = o;
        o += up(n);
        return at;
    };
    l.qkv = take(size_t(gd::max_rows) * gd::n_channel * 4);
    l.z = take(size_t(gd::max_rows) * gd::n_value * 4);
    l.alpha = take(size_t(gd::max_rows) * gd::n_head * 4);
    l.beta = take(size_t(gd::max_rows) * gd::n_head * 4);
    l.conv = take(size_t(gd::n_channel) * gd::n_tap * 4);
    l.dt = take(gd::n_head * 4);
    l.a = take(gd::n_head * 4);
    l.norm = take(gd::head_dim * 4);
    l.state = take(size_t(gd::n_version) * gd::state_version_floats * 4);
    l.hist = take(size_t(gd::n_version) * gd::hist_version_floats * 4);
    l.gated = take(size_t(gd::max_rows) * gd::n_value * 4);
    l.codes = take(fn::act::codes_bytes(gd::max_rows, gd::n_value));
    l.scales = take(fn::act::scales_bytes(gd::max_rows, gd::n_value));
    l.bytes = o;
    return l;
}

gd::core_args args_of(uint8_t * base, const set_layout & l, void * ws, const uint32_t * cur, int32_t rows) {
    gd::core_args a;
    a.conv = reinterpret_cast<const float *>(base + l.qkv); // the convolution's post-SiLU output
    a.z = reinterpret_cast<const float *>(base + l.z);
    a.alpha = reinterpret_cast<const float *>(base + l.alpha);
    a.beta = reinterpret_cast<const float *>(base + l.beta);
    a.dt = reinterpret_cast<const float *>(base + l.dt);
    a.a = reinterpret_cast<const float *>(base + l.a);
    a.norm_w = reinterpret_cast<const float *>(base + l.norm);
    a.state = reinterpret_cast<float *>(base + l.state);
    a.cur = cur; // slot 0 throughout (SR1): the bench never commits
    a.gated = reinterpret_cast<float *>(base + l.gated);
    a.gated_q = {reinterpret_cast<int8_t *>(base + l.codes), reinterpret_cast<float *>(base + l.scales), rows,
                 gd::n_value};
    a.ws = ws;
    a.rows = rows;
    return a;
}

// values for every set: a fixed pattern in [-1, 1) (the time does not depend on them), with the softplus input below
// its threshold and finite decays
__global__ void fill_kernel(float * p, int64_t n, float scale) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        p[i] = scale * (float(int32_t((uint64_t(i) * 2654435761u) % 2001u) - 1000) * 1e-3f);
    }
}

// GD1 Amendment 2b step 0, the write-rate probe (scaffolding, never a kernel result): gdn_core's grid, residency and
// access shape at B = 3. Block (h, s), warp w < its split's rows owns state row j (512 bytes a version, one float4 a
// lane). stores: versions 1..3 of the row get values made from the lane; reads then stores: the row's version 0 is
// read and written to versions 1..3, each store waiting for the read, as gdn_core's do. 4.72 MB of stores a launch,
// and 1.57 MB of reads with READ.
template <bool READ> __global__ void __launch_bounds__(704, 2) probe_kernel(float * state) {
    const int32_t h = static_cast<int32_t>(blockIdx.x) / gd::n_split,
                  s = static_cast<int32_t>(blockIdx.x) % gd::n_split;
    const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32, lane = static_cast<int32_t>(threadIdx.x) % 32;
    const int32_t n_rows = s < 2 ? 22 : 21, row0 = s < 2 ? 22 * s : 44 + 21 * (s - 2);
    if (warp >= n_rows) {
        return;
    }
    float4 * row = reinterpret_cast<float4 *>(state + (int64_t(h) * gd::head_dim + row0 + warp) * gd::head_dim) + lane;
    float4 v = make_float4(float(lane), 1.0f, 2.0f, 3.0f);
    if (READ) {
        v = __ldg(row);
    }
#pragma unroll
    for (int32_t r = 1; r <= 3; ++r) {
        row[r * gd::state_version_floats / 4] = v;
    }
}

struct timing {
    double median = 0.0, p90 = 0.0;
};

timing stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    timing t;
    const size_t n = v.size();
    t.median = n % 2 == 0 ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
    t.p90 = v[(9 * n + 9) / 10 - 1];
    return t;
}

// the SM clock (MHz) from XP1's probe
bool sm_mhz(uint64_t * probe, cudaStream_t st, double & mhz) {
    uint64_t h[2] = {0, 0};
    if (!fn::xp1::clock_probe(probe, st) ||
        !cuda_ok(cudaMemcpyAsync(h, probe, 16, cudaMemcpyDeviceToHost, st), "probe") ||
        !cuda_ok(cudaStreamSynchronize(st), "probe")) {
        return false;
    }
    mhz = h[1] > 0 ? double(h[0]) * 1e3 / double(h[1]) : 0.0;
    return true;
}

bool replay_us(cudaGraphExec_t exec, cudaStream_t st, cudaEvent_t e0, cudaEvent_t e1, double & us) {
    float ms = 0.0f;
    const bool ok = cuda_ok(cudaEventRecord(e0, st), "record") && cuda_ok(cudaGraphLaunch(exec, st), "replay") &&
                    cuda_ok(cudaEventRecord(e1, st), "record") && cuda_ok(cudaEventSynchronize(e1), "sync") &&
                    cuda_ok(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    us = 1000.0 * double(ms);
    return ok;
}

} // namespace

int fnfl_gd_bench(int device, const char * mode) {
    const bool ncu = strcmp(mode, "ncu") == 0, probe = strcmp(mode, "probe") == 0, diag = strcmp(mode, "diag") == 0;
    if (!ncu && !probe && !diag && strcmp(mode, "all") != 0) {
        fprintf(stderr, "gd-bench: mode is all, ncu, probe or diag\n");
        return 2;
    }
    cudaDeviceProp prop = {};
    int driver = 0, runtime = 0;
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice") || !cuda_ok(cudaGetDeviceProperties(&prop, device), "props") ||
        !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
        return 1;
    }
    printf("# gd-bench: %s, sm_%d%d, %d SMs, driver API %d, runtime %d, mode %s, start %s\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, driver, runtime, mode, utc_now().c_str());
    for (int32_t rows = 1; rows <= gd::max_rows; ++rows) {
        gd::core_resources res;
        if (!gd::resources(rows, res)) {
            return 1;
        }
        printf("# gd-bench: gdn_core<%d> %d blocks x %d threads, %d registers, %zu B shared, %d B local, %d blocks an "
               "SM\n",
               rows, res.blocks, res.threads, res.regs, res.shared_bytes, res.local_bytes, res.blocks_per_sm);
    }
    const set_layout l = layout();
    dev_buf sets(l.bytes * n_set), ws(gd::workspace_bytes()), flush(flush_bytes), evict_out(256), probe_buf(16),
        cur(16);
    cudaStream_t st = nullptr;
    bool ok = sets.p != nullptr && ws.p != nullptr && flush.p != nullptr && evict_out.p != nullptr &&
              probe_buf.p != nullptr && cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    ok = ok && cuda_ok(cudaMemsetAsync(ws.p, 0, gd::workspace_bytes(), st), "zero ws") &&
         cuda_ok(cudaMemsetAsync(flush.p, 0, flush_bytes, st), "zero flush") &&
         cur.p != nullptr && cuda_ok(cudaMemsetAsync(cur.p, 0, 16, st), "zero cur");
    // the committed slot's word (SR1), cudaMalloc storage
    const uint32_t * cur_word = reinterpret_cast<const uint32_t *>(cur.p);
    for (int32_t s = 0; s < n_set && ok; ++s) {
        uint8_t * b = sets.p + size_t(s) * l.bytes;
        fill_kernel<<<144, 256, 0, st>>>(reinterpret_cast<float *>(b), int64_t(l.bytes / 4), 1.0f);
        fill_kernel<<<144, 256, 0, st>>>(reinterpret_cast<float *>(b + l.state),
                                         int64_t(gd::n_version) * gd::state_version_floats, 0.2f);
        ok = cuda_ok(cudaPeekAtLastError(), "fill");
    }
    ok = ok && cuda_ok(cudaStreamSynchronize(st), "fill");
    printf("# gd-bench: %d sets of %zu bytes (%.1f MB rotated), %d launches a warm replay, %d a cold replay\n", n_set,
           l.bytes, double(l.bytes) * n_set / 1e6, n_set, n_cold);
    // keep the card busy for warm_s before timing
    if (ok) {
        const auto t0 = std::chrono::steady_clock::now();
        int64_t n = 0;
        while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
            for (int32_t i = 0; i < 16 && ok; ++i, ++n) {
                ok = cuda_ok(cudaMemsetAsync(sets.p, static_cast<int>(n & 0xff), size_t(64) << 20, st), "memset");
            }
            ok = ok && cuda_ok(cudaStreamSynchronize(st), "warm sync");
        }
        // the busy loop overwrote the first sets: refill them
        for (int32_t s = 0; s < n_set && ok && size_t(s) * l.bytes < (size_t(64) << 20); ++s) {
            uint8_t * b = sets.p + size_t(s) * l.bytes;
            fill_kernel<<<144, 256, 0, st>>>(reinterpret_cast<float *>(b), int64_t(l.bytes / 4), 1.0f);
            fill_kernel<<<144, 256, 0, st>>>(reinterpret_cast<float *>(b + l.state),
                                             int64_t(gd::n_version) * gd::state_version_floats, 0.2f);
            ok = cuda_ok(cudaPeekAtLastError(), "fill");
        }
        ok = ok && cuda_ok(cudaStreamSynchronize(st), "fill");
        printf("# gd-bench: busy for %.2f s, timing from %s\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), utc_now().c_str());
    }
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    ok = ok && cuda_ok(cudaEventCreate(&e0), "event") && cuda_ok(cudaEventCreate(&e1), "event");

    if (ncu) {
        // ncu: a graph of 8 launches at B = 3 on rotated sets, 3 replays, then one more: profile launch 25
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        ok = ok && cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        for (int32_t i = 0; i < 8 && ok; ++i) {
            ok = gd::core(args_of(sets.p + size_t(i) * l.bytes, l, ws.p, cur_word, 3), st);
        }
        ok = cuda_ok(cudaStreamEndCapture(st, &graph), "end capture") && ok;
        ok = ok && cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int32_t r = 0; r < 4 && ok; ++r) {
            ok = cuda_ok(cudaGraphLaunch(exec, st), "replay") && cuda_ok(cudaStreamSynchronize(st), "sync");
        }
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
    }

    if (probe) {
        // Amendment 2b step 0: n_set launches back to back on rotated sets (data cold, code warm), stores only and
        // reads then stores; W = store bytes / (time - 1.949 us); the reads-then-stores time against the additive model
        const double st_bytes = 3.0 * gd::state_version_floats * 4, rd_bytes = double(gd::state_version_floats) * 4;
        double med[2] = {0.0, 0.0};
        for (int32_t k = 0; k < 2 && ok; ++k) {
            cudaGraph_t gr = nullptr;
            cudaGraphExec_t ex = nullptr;
            ok = cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            for (int32_t i = 0; i < n_set && ok; ++i) {
                float * stp = reinterpret_cast<float *>(sets.p + size_t(i) * l.bytes + l.state);
                if (k == 0) {
                    probe_kernel<false><<<n_block_probe, 704, 0, st>>>(stp);
                } else {
                    probe_kernel<true><<<n_block_probe, 704, 0, st>>>(stp);
                }
                ok = cuda_ok(cudaPeekAtLastError(), "probe launch");
            }
            ok = cuda_ok(cudaStreamEndCapture(st, &gr), "end capture") && ok;
            ok = ok && cuda_ok(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
            double mhz0 = 0.0, mhz1 = 0.0;
            ok = ok && sm_mhz(reinterpret_cast<uint64_t *>(probe_buf.p), st, mhz0);
            for (int32_t i = 0; i < n_warm && ok; ++i) {
                ok = cuda_ok(cudaGraphLaunch(ex, st), "warm") && cuda_ok(cudaStreamSynchronize(st), "warm");
            }
            std::vector<double> per;
            for (int32_t i = 0; i < n_timed && ok; ++i) {
                double us = 0.0;
                ok = replay_us(ex, st, e0, e1, us);
                per.push_back(us / n_set);
            }
            ok = ok && sm_mhz(reinterpret_cast<uint64_t *>(probe_buf.p), st, mhz1);
            if (ex != nullptr) {
                cudaGraphExecDestroy(ex);
            }
            if (gr != nullptr) {
                cudaGraphDestroy(gr);
            }
            if (!ok) {
                break;
            }
            const timing t = stats(per);
            med[k] = t.median;
            printf("{\"ticket\": \"GD1 A2b\", \"point\": \"probe %s\", \"store_bytes\": %.0f, \"read_bytes\": %.0f, "
                   "\"us\": %.3f, \"p90_us\": %.3f, \"launches\": %d, \"timed\": %d, \"sm_mhz_before\": %.0f, "
                   "\"sm_mhz_after\": %.0f}\n",
                   k == 0 ? "stores" : "reads then stores", st_bytes, k == 0 ? 0.0 : rd_bytes, t.median, t.p90, n_set,
                   n_timed, mhz0, mhz1);
        }
        if (ok) {
            const double w_gbs = st_bytes / ((med[0] - ramp_us) * 1e3);
            const double model = ramp_us + rd_bytes / (hw_gbs * 1e3) + st_bytes / (w_gbs * 1e3);
            printf("{\"ticket\": \"GD1 A2b\", \"point\": \"probe W\", \"W_gbs\": %.1f, \"mixed_us\": %.3f, "
                   "\"additive_model_us\": %.3f, \"mixed_over_model\": %.4f}\n",
                   w_gbs, med[1], model, med[1] / model);
        }
    }

    if (diag) {
        // Amendment 2b step 4's check: warm, the kernel as built against the same kernel with its finish disabled (the
        // head counters preset to 1,000, so no block takes ticket 5; publish, fence and ticket still run)
        for (const int32_t rows : {1, 3}) {
            for (int32_t variant = 0; variant < 2 && ok; ++variant) {
                std::vector<uint32_t> cnt(gd::n_head, variant == 0 ? 0u : 1000u);
                ok = cuda_ok(cudaMemcpyAsync(ws.p, cnt.data(), cnt.size() * 4, cudaMemcpyHostToDevice, st),
                             "counters") &&
                     cuda_ok(cudaStreamSynchronize(st), "counters");
                cudaGraph_t gr = nullptr;
                cudaGraphExec_t ex = nullptr;
                ok = ok && cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
                for (int32_t i = 0; i < n_set && ok; ++i) {
                    ok = gd::core(args_of(sets.p + size_t(i) * l.bytes, l, ws.p, cur_word, rows), st);
                }
                ok = cuda_ok(cudaStreamEndCapture(st, &gr), "end capture") && ok;
                ok = ok && cuda_ok(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
                double mhz0 = 0.0, mhz1 = 0.0;
                ok = ok && sm_mhz(reinterpret_cast<uint64_t *>(probe_buf.p), st, mhz0);
                for (int32_t i = 0; i < n_warm && ok; ++i) {
                    ok = cuda_ok(cudaGraphLaunch(ex, st), "warm") && cuda_ok(cudaStreamSynchronize(st), "warm");
                }
                std::vector<double> per;
                for (int32_t i = 0; i < n_timed && ok; ++i) {
                    double us = 0.0;
                    ok = replay_us(ex, st, e0, e1, us);
                    per.push_back(us / n_set);
                }
                ok = ok && sm_mhz(reinterpret_cast<uint64_t *>(probe_buf.p), st, mhz1);
                if (ex != nullptr) {
                    cudaGraphExecDestroy(ex);
                }
                if (gr != nullptr) {
                    cudaGraphDestroy(gr);
                }
                if (!ok) {
                    break;
                }
                const timing t = stats(per);
                printf(
                    "{\"ticket\": \"GD1 A2b\", \"point\": \"diag B%d %s\", \"warm_us\": %.3f, \"warm_p90_us\": %.3f, "
                    "\"launches\": %d, \"timed\": %d, \"sm_mhz_before\": %.0f, \"sm_mhz_after\": %.0f}\n",
                    rows, variant == 0 ? "as built" : "finish disabled", t.median, t.p90, n_set, n_timed, mhz0, mhz1);
            }
        }
        ok = ok && cuda_ok(cudaMemsetAsync(ws.p, 0, gd::workspace_bytes(), st), "zero ws");
    }

    for (int32_t rows = 1; rows <= gd::max_rows && ok && !ncu && !probe && !diag; ++rows) {
        // warm: n_set launches back to back on rotated sets; cold: n_cold x (flush, evictor, launch) against the same
        // graph without the launches
        cudaGraph_t gr[3] = {nullptr, nullptr, nullptr};
        cudaGraphExec_t ex[3] = {nullptr, nullptr, nullptr};
        for (int32_t k = 0; k < 3 && ok; ++k) {
            ok = cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            const int32_t n = k == 0 ? n_set : n_cold;
            for (int32_t i = 0; i < n && ok; ++i) {
                if (k > 0) {
                    ok = fn::xp1::flush_l2(flush.p, flush_bytes, st) &&
                         icache_evict(reinterpret_cast<float *>(evict_out.p), st);
                }
                if (ok && k < 2) {
                    ok = gd::core(args_of(sets.p + size_t(i % n_set) * l.bytes, l, ws.p, cur_word, rows), st);
                }
            }
            ok = cuda_ok(cudaStreamEndCapture(st, &gr[k]), "end capture") && ok;
            ok = ok && cuda_ok(cudaGraphInstantiate(&ex[k], gr[k], 0), "instantiate");
        }
        double mhz0 = 0.0, mhz1 = 0.0;
        ok = ok && sm_mhz(reinterpret_cast<uint64_t *>(probe_buf.p), st, mhz0);
        for (int32_t i = 0; i < n_warm && ok; ++i) {
            for (int32_t k = 0; k < 3 && ok; ++k) {
                ok = cuda_ok(cudaGraphLaunch(ex[k], st), "warm replay") && cuda_ok(cudaStreamSynchronize(st), "warm");
            }
        }
        std::vector<double> warm, cold, with, without;
        const std::string t0 = utc_now();
        for (int32_t i = 0; i < n_timed && ok; ++i) {
            double us[3] = {0.0, 0.0, 0.0};
            for (int32_t k = 0; k < 3 && ok; ++k) {
                ok = replay_us(ex[k], st, e0, e1, us[k]);
            }
            warm.push_back(us[0] / n_set);
            cold.push_back((us[1] - us[2]) / n_cold);
            with.push_back(us[1]);
            without.push_back(us[2]);
        }
        const std::string t1 = utc_now();
        ok = ok && sm_mhz(reinterpret_cast<uint64_t *>(probe_buf.p), st, mhz1);
        for (int32_t k = 0; k < 3; ++k) {
            if (ex[k] != nullptr) {
                cudaGraphExecDestroy(ex[k]);
            }
            if (gr[k] != nullptr) {
                cudaGraphDestroy(gr[k]);
            }
        }
        if (!ok) {
            break;
        }
        const timing w = stats(warm), c = stats(cold), cw = stats(with), cn = stats(without);
        const double fl = floor_us(rows);
        printf(
            "{\"ticket\": \"GD1 A2b\", \"point\": \"gdn_core B%d\", \"rows\": %d, \"bytes\": %lld, \"floor_us\": %.3f, "
            "\"warm_us\": %.3f, \"warm_p90_us\": %.3f, \"warm_pct_hw\": %.1f, \"cold_us\": %.3f, \"cold_p90_us\": "
            "%.3f, \"cold_pct_hw\": %.1f, \"gate_us\": %.3f, \"cold_with_replay_us\": %.2f, "
            "\"cold_without_replay_us\": %.2f, \"sets\": %d, \"launches_warm\": %d, \"launches_cold\": %d, "
            "\"timed\": %d, \"warm\": %d, \"sm_mhz_before\": %.0f, \"sm_mhz_after\": %.0f, \"t_start\": \"%s\", "
            "\"t_end\": \"%s\"}\n",
            rows, rows, static_cast<long long>(launch_bytes(rows)), fl, w.median, w.p90, 100.0 * fl / w.median,
            c.median, c.p90, 100.0 * fl / c.median, fl / 0.9, cw.median, cn.median, n_set, n_set, n_cold, n_timed,
            n_warm, mhz0, mhz1, t0.c_str(), t1.c_str());
        fflush(stdout);
    }
    if (e0 != nullptr) {
        cudaEventDestroy(e0);
    }
    if (e1 != nullptr) {
        cudaEventDestroy(e1);
    }
    ok = ok && cuda_ok(cudaStreamSynchronize(st), "end");
    if (st != nullptr) {
        cudaStreamDestroy(st);
    }
    printf("# gd-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
