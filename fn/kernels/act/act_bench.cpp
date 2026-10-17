#include "fn/kernels/act/act.h"
#include "fn/kernels/act/act_test.h"

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

constexpr int32_t n_unit = 512;   // launches a replay (a round), each on its own slice (cold) or all on slice 0 (hot)
constexpr int32_t n_warm = 3;     // untimed replays
constexpr int32_t n_timed = 20;   // timed replays
constexpr double warm_s = 2.0;    // HX12 Later 15: a process starts at 1,260 MHz for 0.3-1 s
constexpr double hw_gbs = 1021.6; // the % of hardware speed's bandwidth (HX2 B1)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "act-bench: %s: %s\n", what, cudaGetErrorString(e));
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

// one point: n_unit launches captured in a graph, timed over n_timed replays; prints a JSON line
bool time_point(int32_t rows, int32_t k, bool cold, cudaStream_t st) {
    const size_t xb = size_t(rows) * size_t(k) * 4, cb = fn::act::codes_bytes(rows, k),
                 sb = fn::act::scales_bytes(rows, k);
    const auto up = [](size_t n) { return (n + 255) / 256 * 256; };
    const size_t slices = cold ? size_t(n_unit) : 1;
    const size_t xs = up(xb), cs = up(cb), ss = up(sb);
    dev_buf x(slices * xs), c(slices * cs), s(slices * ss);
    if (x.p == nullptr || c.p == nullptr || s.p == nullptr) {
        return false;
    }
    // inputs: a fixed pattern of magnitudes (the time does not depend on the values)
    {
        std::vector<float> h(slices * xs / 4);
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] = static_cast<float>(static_cast<int32_t>((i * 2654435761u) % 2001u) - 1000) * 1e-3f;
        }
        if (!cuda_ok(cudaMemcpy(x.p, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "put x") ||
            !cuda_ok(cudaDeviceSynchronize(), "put x")) {
            return false;
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    bool ok = cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
    for (int32_t i = 0; i < n_unit && ok; ++i) {
        const size_t j = cold ? size_t(i) : 0;
        const fn::act::out_view o = {reinterpret_cast<int8_t *>(c.p + j * cs), reinterpret_cast<float *>(s.p + j * ss),
                                     rows, k};
        ok = fn::act::quantize(reinterpret_cast<const float *>(x.p + j * xs), o, st);
    }
    ok = cuda_ok(cudaStreamEndCapture(st, &graph), "end capture") && ok;
    ok = ok && cuda_ok(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    ok = ok && cuda_ok(cudaEventCreate(&e0), "event") && cuda_ok(cudaEventCreate(&e1), "event");
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = cuda_ok(cudaGraphLaunch(exec, st), "warm replay") && cuda_ok(cudaStreamSynchronize(st), "warm");
    }
    std::vector<double> round_us;
    const std::string t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        float ms = 0.0f;
        ok = cuda_ok(cudaEventRecord(e0, st), "record") && cuda_ok(cudaGraphLaunch(exec, st), "replay") &&
             cuda_ok(cudaEventRecord(e1, st), "record") && cuda_ok(cudaEventSynchronize(e1), "sync") &&
             cuda_ok(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        round_us.push_back(1000.0 * ms);
    }
    const std::string t1 = utc_now();
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
    }
    if (e0 != nullptr) {
        cudaEventDestroy(e0);
    }
    if (e1 != nullptr) {
        cudaEventDestroy(e1);
    }
    if (!ok) {
        fprintf(stderr, "act-bench: B %d K %d failed\n", rows, k);
        return false;
    }
    std::vector<double> sorted = round_us;
    std::sort(sorted.begin(), sorted.end());
    const double median = 0.5 * (sorted[n_timed / 2 - 1] + sorted[n_timed / 2]);
    const double p90 = sorted[(9 * n_timed + 9) / 10 - 1];
    const double launch_us = median / n_unit;
    const size_t bytes = xb + cb + sb;
    const double pct = 100.0 * double(bytes) / (launch_us * 1e-6) / (hw_gbs * 1e9);
    printf("{\"ticket\": \"C2\", \"point\": \"act B%d K%d %s\", \"rows\": %d, \"k\": %d, \"round_ms\": %.4f, "
           "\"round_p90_ms\": %.4f, \"launches_per_round\": %d, \"launch_us\": %.3f, \"bytes_per_launch\": %zu, "
           "\"pct_hw\": %.2f, \"timed\": %d, \"warm\": %d, \"t_start\": \"%s\", \"t_end\": \"%s\", \"round_us\": [",
           rows, k, cold ? "cold" : "hot", rows, k, median / 1000.0, p90 / 1000.0, n_unit, launch_us, bytes, pct,
           n_timed, n_warm, t0.c_str(), t1.c_str());
    for (size_t i = 0; i < round_us.size(); ++i) {
        printf("%s%.2f", i == 0 ? "" : ", ", round_us[i]);
    }
    printf("]}\n");
    return true;
}

} // namespace

int fnfl_act_bench(int device, const char * mode) {
    const bool acceptance = strcmp(mode, "acceptance") == 0;
    if (!acceptance && strcmp(mode, "all") != 0) {
        fprintf(stderr, "act-bench: mode is all or acceptance\n");
        return 2;
    }
    cudaDeviceProp prop = {};
    int driver = 0, runtime = 0;
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice") || !cuda_ok(cudaGetDeviceProperties(&prop, device), "props") ||
        !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
        return 1;
    }
    printf("# act-bench: %s, sm_%d%d, %d SMs, driver API %d, runtime %d, mode %s, start %s\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, driver, runtime, mode, utc_now().c_str());
    cudaStream_t st = nullptr;
    bool ok = cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    // keep the card busy for warm_s before timing: memsets of a 64 MiB scratch buffer
    if (ok) {
        dev_buf scratch(size_t(64) << 20);
        ok = scratch.p != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        int64_t n = 0;
        while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
            for (int32_t i = 0; i < 32 && ok; ++i, ++n) {
                ok = cuda_ok(cudaMemsetAsync(scratch.p, static_cast<int>(n & 0xff), size_t(64) << 20, st), "memset");
            }
            ok = ok && cuda_ok(cudaStreamSynchronize(st), "warm sync");
        }
        printf("# act-bench: busy for %.2f s (%lld memsets of 64 MiB), timing from %s\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), static_cast<long long>(n),
               utc_now().c_str());
    }
    std::vector<int32_t> rows_list = acceptance ? std::vector<int32_t>{3} : std::vector<int32_t>{1, 2, 3, 4};
    std::vector<int32_t> k_list =
        acceptance ? std::vector<int32_t>{2560, 6144} : std::vector<int32_t>{320, 2560, 3072, 6144};
    for (const int32_t k : k_list) {
        for (const int32_t rows : rows_list) {
            ok = ok && time_point(rows, k, true, st) && time_point(rows, k, false, st);
        }
    }
    if (st != nullptr) {
        cudaStreamDestroy(st);
    }
    printf("# act-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
