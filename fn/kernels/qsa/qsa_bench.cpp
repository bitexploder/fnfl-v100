#include "fn/kernels/qsa/qsa_bench.h"

#include "fn/kernels/qsa/qsa.h"
#include "fn/model/shapes.h"
#include "fn/quant/formats.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

namespace q = fn::qsa;

constexpr int32_t n_unit = 64;                     // units (one launch, or one sequence) captured in a graph
constexpr int32_t n_warm = 3;                      // untimed replays before timing
constexpr int32_t n_timed = 20;                    // timed replays
constexpr double warm_s = 2.0;                     // HX12: a process starts at 1,260 MHz for 0.3-1 s
constexpr size_t rotate_bytes = size_t(200) << 20; // cold data: at least this much rotated over a graph
constexpr int32_t max_depth = 196608;              // the deepest timed point
constexpr int32_t HD = q::head_dim;
constexpr int32_t R = q::ratio;
constexpr int32_t fixture_width = 2051; // the HX7 fixture's rows: the pinned source's top_k width (2,048 + 3)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "qa-bench: %s: %s\n", what, cudaGetErrorString(e));
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

// owns one device allocation of n elements of T
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

// owns a captured graph's executable and the two timing events
struct timed_graph {
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;

    timed_graph() = default;

    ~timed_graph() {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
        }
        if (e0 != nullptr) {
            cudaEventDestroy(e0);
        }
        if (e1 != nullptr) {
            cudaEventDestroy(e1);
        }
    }

    timed_graph(const timed_graph &) = delete;
    timed_graph & operator=(const timed_graph &) = delete;
};

// rotation sets of `bytes` each in a pool of pool_bytes: as many as reach rotate_bytes, at least min_sets, at most
// n_unit and what the pool holds
int32_t sets_for(size_t bytes, int32_t min_sets, size_t pool_bytes) {
    const size_t n = (rotate_bytes + bytes - 1) / bytes;
    const size_t fit = pool_bytes / bytes;
    return static_cast<int32_t>(std::min<size_t>({std::max<size_t>(n, static_cast<size_t>(min_sets)), n_unit, fit}));
}

struct bench {
    cudaStream_t st = nullptr;
    // summaries: room for 17 layers at 196K (16 of 25.2 MB are rotated); keys: the raw indexer cache, room for 5 sets
    // at 196K (4 of 50.3 MB are rotated). Shallower points rotate more, smaller sets in the same pools.
    size_t sum_pool_elems = q::summary_bytes((max_depth + q::max_rows) / R) / 4 * 17;
    size_t key_pool_elems = static_cast<size_t>(max_depth + q::max_rows) * HD * 5;
    dev_buf<float> sums{sum_pool_elems};
    dev_buf<uint16_t> keys{key_pool_elems};
    dev_buf<float> qv{static_cast<size_t>(q::max_rows) * q::n_idx_head * HD};
    dev_buf<float> k_norm{HD}, q_norm{HD};
    dev_buf<uint8_t> ws_mem{q::workspace_bytes()};
    q::workspace ws = q::carve(ws_mem.p);
    dev_buf<uint32_t> body{static_cast<size_t>(q::max_rows) * q::max_row_words};
    dev_buf<int32_t> offsets{static_cast<size_t>(q::max_rows) + 1};
    // the timed depths on the device (TG1: select reads the depth there), written once before any capture
    static constexpr int32_t bench_depths[3] = {8192, 65536, 196608};
    dev_buf<int32_t> depths{3};
    bool depths_ok = false;

    const int32_t * depth_ptr(int32_t depth) const {
        for (int32_t i = 0; i < 3; ++i) {
            if (bench_depths[i] == depth) {
                return depths.p + i;
            }
        }
        return nullptr;
    }
    // launches of each kernel so far, so a profiler can skip to a timed launch
    std::map<std::string, int64_t> launches;

    bool ok() {
        if (!depths_ok && depths.p != nullptr) {
            depths_ok =
                cuda_ok(cudaMemcpy(depths.p, bench_depths, sizeof(bench_depths), cudaMemcpyHostToDevice), "depths");
        }
        return sums.p && keys.p && qv.p && k_norm.p && q_norm.p && ws_mem.p && body.p && offsets.p && depths_ok;
    }

    static size_t sum_set_elems(int32_t depth) { return q::summary_bytes((depth + q::max_rows) / R) / 4; }

    static size_t key_set_elems(int32_t depth) { return static_cast<size_t>(depth + q::max_rows) * HD; }

    bool select(int32_t rows, int32_t depth, int32_t set) {
        // max_cells = depth + rows: launch 1's grid as QA1b sized it for this depth
        const q::select_args a = {sums.p + static_cast<size_t>(set) * sum_set_elems(depth),
                                  qv.p,
                                  q_norm.p,
                                  depth_ptr(depth),
                                  depth + rows,
                                  q::model_rope(),
                                  ws,
                                  body.p,
                                  offsets.p};
        if (a.depth == nullptr) {
            fprintf(stderr, "qa-bench: depth %d is not one of the bench's\n", depth);
            return false;
        }
        {
            // launch 1 is always enqueued (it returns at once below 513 blocks)
            const q::score_grid g = q::grid_of(rows, depth);
            ++launches["score_kernel B" + std::to_string(rows) + " D" + std::to_string(g.depth)];
        }
        ++launches["select_kernel"];
        if (!q::select(rows, a, st)) {
            fprintf(stderr, "qa-bench: select not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        return true;
    }

    // the round's summary update: the last count complete blocks at this depth
    bool update(int32_t rows, int32_t depth, int32_t count, int32_t sum_set, int32_t key_set) {
        const int32_t nb = (depth + rows) / R;
        const q::update_args u = {keys.p + static_cast<size_t>(key_set) * key_set_elems(depth),
                                  k_norm.p,
                                  sums.p + static_cast<size_t>(sum_set) * sum_set_elems(depth),
                                  nb - count,
                                  count,
                                  q::model_rope()};
        ++launches["summary_update_kernel"];
        if (!q::summary_update(u, st)) {
            fprintf(stderr, "qa-bench: update not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        return true;
    }
};

// Captures n_unit units of work, replays n_warm untimed and n_timed timed, and prints one JSON line. The time of a
// unit is a replay's event time over n_unit; median and p90 (nearest rank) over the timed replays.
bool time_point(bench & bn, const std::string & point, const std::string & what, int32_t rows, int32_t depth,
                int32_t launches_per_unit, const std::string & rotation, const std::function<bool(int32_t)> & unit) {
    cudaGraph_t graph = nullptr;
    timed_graph tg;
    if (!cuda_ok(cudaStreamBeginCapture(bn.st, cudaStreamCaptureModeThreadLocal), "begin capture")) {
        return false;
    }
    bool ok = true;
    const std::map<std::string, int64_t> before = bn.launches;
    for (int32_t i = 0; i < n_unit && ok; ++i) {
        ok = unit(i);
    }
    // capture enqueues nothing: undo the count and add it back per replay
    std::map<std::string, int64_t> per_replay;
    for (const auto & kv : bn.launches) {
        const auto it = before.find(kv.first);
        const int64_t n = kv.second - (it == before.end() ? 0 : it->second);
        if (n > 0) {
            per_replay[kv.first] = n;
        }
    }
    bn.launches = before;
    const auto replayed = [&]() {
        for (const auto & kv : per_replay) {
            bn.launches[kv.first] += kv.second;
        }
    };
    ok = cuda_ok(cudaStreamEndCapture(bn.st, &graph), "end capture") && ok;
    if (!ok) {
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        return false;
    }
    ok = cuda_ok(cudaGraphInstantiate(&tg.exec, graph, 0), "instantiate");
    cudaGraphDestroy(graph);
    ok = ok && cuda_ok(cudaEventCreate(&tg.e0), "event") && cuda_ok(cudaEventCreate(&tg.e1), "event");
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = cuda_ok(cudaGraphLaunch(tg.exec, bn.st), "warm replay") &&
             cuda_ok(cudaStreamSynchronize(bn.st), "warm sync");
        replayed();
    }
    std::map<std::string, int64_t> first_timed;
    for (const auto & kv : per_replay) {
        first_timed[kv.first] = bn.launches[kv.first];
    }
    std::vector<double> us;
    const std::string t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        float ms = 0.0f;
        ok = cuda_ok(cudaEventRecord(tg.e0, bn.st), "record") && cuda_ok(cudaGraphLaunch(tg.exec, bn.st), "replay") &&
             cuda_ok(cudaEventRecord(tg.e1, bn.st), "record") && cuda_ok(cudaEventSynchronize(tg.e1), "sync") &&
             cuda_ok(cudaEventElapsedTime(&ms, tg.e0, tg.e1), "elapsed");
        replayed();
        us.push_back(1000.0 * ms / n_unit);
    }
    const std::string t1 = utc_now();
    uint32_t err = 0;
    ok = ok && cuda_ok(cudaMemcpy(&err, bn.ws.error, 4, cudaMemcpyDeviceToHost), "error word");
    if (!ok || err != 0) {
        fprintf(stderr, "qa-bench: %s failed (device error word %u)\n", point.c_str(), err);
        return false;
    }
    std::vector<double> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    const double median = 0.5 * (sorted[n_timed / 2 - 1] + sorted[n_timed / 2]);
    const double p90 = sorted[(9 * n_timed + 9) / 10 - 1];
    printf("{\"ticket\": \"QA1b\", \"point\": \"%s\", \"what\": \"%s\", \"rows\": %d, \"depth\": %d, "
           "\"median_us\": %.3f, \"p90_us\": %.3f, \"n\": %d, \"units_per_replay\": %d, \"launches_per_unit\": %d, "
           "\"warm_replays\": %d, \"rotation\": \"%s\", \"first_timed_launch\": {",
           point.c_str(), what.c_str(), rows, depth, median, p90, n_timed, n_unit, launches_per_unit, n_warm,
           rotation.c_str());
    bool first = true;
    for (const auto & kv : first_timed) {
        printf("%s\"%s\": %lld", first ? "" : ", ", kv.first.c_str(), static_cast<long long>(kv.second));
        first = false;
    }
    printf("}, \"t_start\": \"%s\", \"t_end\": \"%s\", \"us\": [", t0.c_str(), t1.c_str());
    for (int32_t i = 0; i < n_timed; ++i) {
        printf("%s%.3f", i == 0 ? "" : ", ", us[static_cast<size_t>(i)]);
    }
    printf("]}\n");
    fflush(stdout);
    return true;
}

// The HX7 selection pattern at 196,608 cells (fixture from work/QA1/src/make_fixtures.py): the real layer-3 index
// norms, and keys shaped so each row's best blocks are the ones HX7 captured at 128K (blocks past 32,000 are noise), as
// qa-test's HX7 case builds them. Written to key set 0 and query buffer; returns false if the fixture cannot be read.
bool load_clustered(const char * path, bench & bn, int32_t depth, int32_t rows) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        fprintf(stderr, "qa-bench: cannot open %s\n", path);
        return false;
    }
    int32_t hdr[4] = {};
    std::vector<int32_t> cells(static_cast<size_t>(3) * fixture_width);
    std::vector<float> kn(HD), qn(HD);
    const bool rd = fread(hdr, 4, 4, f) == 4 && hdr[0] == 0x31415141 && hdr[2] == 3 && hdr[3] == fixture_width &&
                    fread(cells.data(), 4, cells.size(), f) == cells.size() && fread(kn.data(), 4, HD, f) == HD &&
                    fread(qn.data(), 4, HD, f) == HD;
    fclose(f);
    if (!rd || rows != 3) {
        fprintf(stderr, "qa-bench: %s is not a QA1 HX7 fixture\n", path);
        return false;
    }
    std::vector<std::vector<char>> in(3, std::vector<char>(static_cast<size_t>(depth / R + 1), 0));
    for (int32_t t = 0; t < 3; ++t) {
        for (int32_t j = 0; j < q::top_blocks * R; j += R) {
            in[static_cast<size_t>(t)][static_cast<size_t>(cells[static_cast<size_t>(t) * fixture_width + j] / R)] = 1;
        }
    }
    std::mt19937 rng(128000);
    std::uniform_real_distribution<float> u11(-1.0f, 1.0f), u01(0.0f, 1.0f);
    std::vector<std::vector<float>> P(3, std::vector<float>(HD, 0.0f));
    for (auto & p : P) {
        for (int e = 64; e < HD; ++e) {
            p[e] = (u01(rng) < 0.5f ? -1.0f : 1.0f) * (0.5f + u01(rng));
        }
    }
    std::vector<float> qv(static_cast<size_t>(q::max_rows) * q::n_idx_head * HD, 0.0f);
    for (int32_t t = 0; t < 3; ++t) {
        for (int i = 0; i < q::n_idx_head * HD; ++i) {
            qv[static_cast<size_t>(t) * q::n_idx_head * HD + i] = P[static_cast<size_t>(t)][i % HD] + 0.2f * u11(rng);
        }
    }
    const size_t n = bench::key_set_elems(depth);
    std::vector<uint16_t> k(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t c = i / HD;
        const int e = static_cast<int>(i % HD);
        const size_t b = c / R;
        float v = 0.3f * u11(rng);
        for (int32_t t = 0; t < 3 && e >= 64; ++t) {
            if (b < in[static_cast<size_t>(t)].size() && in[static_cast<size_t>(t)][b]) {
                v += P[static_cast<size_t>(t)][e];
            }
        }
        k[i] = fn::quant::f32_to_bf16(v);
    }
    return cuda_ok(cudaMemcpyAsync(bn.keys.p, k.data(), n * 2, cudaMemcpyHostToDevice, bn.st), "keys") &&
           cuda_ok(cudaMemcpyAsync(bn.k_norm.p, kn.data(), int64_t(HD) * 4, cudaMemcpyHostToDevice, bn.st), "k_norm") &&
           cuda_ok(cudaMemcpyAsync(bn.q_norm.p, qn.data(), int64_t(HD) * 4, cudaMemcpyHostToDevice, bn.st), "q_norm") &&
           cuda_ok(cudaMemcpyAsync(bn.qv.p, qv.data(), qv.size() * 4, cudaMemcpyHostToDevice, bn.st), "q") &&
           cuda_ok(cudaStreamSynchronize(bn.st), "clustered upload");
}

// "dump=DIR": no timing. One select() of each bench point on rotation set 0 (the same data the timed graph's first
// unit reads), then every row's keys, the tiles' lists and the path stats to DIR/<point>.bin for the CPU replay of
// launch 2 (QA1b's predictions). Header: 8 int32 (magic 'QA1b', depth, rows, max_blocks, max_tiles, n_stats, 0, 0),
// then keys u32 [rows][max_blocks], best1 and best2 u32 [rows][max_tiles], lanes u16 [rows][max_tiles], stats u32
// [rows][n_stats], the error word u32.
bool dump_point(bench & bn, const std::string & dir, const std::string & point, int32_t rows, int32_t depth) {
    const size_t nt = q::max_blocks / q::tile_blocks;
    std::vector<uint32_t> keys(static_cast<size_t>(rows) * q::max_blocks), b1(rows * nt), b2(rows * nt),
        stats(static_cast<size_t>(rows) * q::n_stats), err(1);
    std::vector<uint16_t> lanes(rows * nt);
    bool ok = bn.select(rows, depth, 0) && cuda_ok(cudaStreamSynchronize(bn.st), "dump select") &&
              cuda_ok(cudaMemcpy(keys.data(), bn.ws.keys, keys.size() * 4, cudaMemcpyDeviceToHost), "dump keys") &&
              cuda_ok(cudaMemcpy(b1.data(), bn.ws.best1, b1.size() * 4, cudaMemcpyDeviceToHost), "dump best1") &&
              cuda_ok(cudaMemcpy(b2.data(), bn.ws.best2, b2.size() * 4, cudaMemcpyDeviceToHost), "dump best2") &&
              cuda_ok(cudaMemcpy(lanes.data(), bn.ws.lanes, lanes.size() * 2, cudaMemcpyDeviceToHost), "dump lanes") &&
              cuda_ok(cudaMemcpy(stats.data(), bn.ws.stats, stats.size() * 4, cudaMemcpyDeviceToHost), "dump stats") &&
              cuda_ok(cudaMemcpy(err.data(), bn.ws.error, 4, cudaMemcpyDeviceToHost), "dump error");
    if (!ok) {
        return false;
    }
    const std::string path = dir + "/" + point + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "qa-bench: cannot write %s\n", path.c_str());
        return false;
    }
    const int32_t hdr[8] = {0x62314151, depth, rows, q::max_blocks, static_cast<int32_t>(nt), q::n_stats, 0, 0};
    ok = fwrite(hdr, 4, 8, f) == 8 && fwrite(keys.data(), 4, keys.size(), f) == keys.size() &&
         fwrite(b1.data(), 4, b1.size(), f) == b1.size() && fwrite(b2.data(), 4, b2.size(), f) == b2.size() &&
         fwrite(lanes.data(), 2, lanes.size(), f) == lanes.size() &&
         fwrite(stats.data(), 4, stats.size(), f) == stats.size() && fwrite(err.data(), 4, 1, f) == 1;
    ok = fclose(f) == 0 && ok;
    printf("# qa-bench: dumped %s (error word %u)\n", path.c_str(), err[0]);
    return ok && err[0] == 0;
}

std::string mb(size_t bytes) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / 1e6);
    return buf;
}

} // namespace

int fnfl_qsa_bench(int device, const char * mode, const char * fixture) {
    const bool acceptance = mode != nullptr && std::strcmp(mode, "acceptance") == 0;
    const std::string dump_dir = mode != nullptr && std::strncmp(mode, "dump=", 5) == 0 ? std::string(mode + 5) : "";
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaDeviceProp prop = {};
    int driver = 0, runtime = 0;
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "properties") ||
        !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
        return 1;
    }
    printf("# qa-bench: %s, sm_%d%d, %d SMs, driver API %d, runtime %d, mode %s, start %s\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, driver, runtime, acceptance ? "acceptance" : "all", utc_now().c_str());

    bool ok = true;
    {
        bench bn;
        ok = bn.ok() && cuda_ok(cudaStreamCreateWithFlags(&bn.st, cudaStreamNonBlocking), "stream");
        std::mt19937 rng(11);
        // inputs: raw keys (set 0 from the host, copied to the rest of the pool), norms, queries
        if (ok) {
            std::uniform_real_distribution<float> u11(-1.0f, 1.0f), u01(0.5f, 1.5f);
            const size_t n0 = bench::key_set_elems(max_depth);
            std::vector<uint16_t> k(n0);
            for (auto & v : k) {
                v = fn::quant::f32_to_bf16(u11(rng));
            }
            std::vector<float> kn(HD), qn(HD), qq(static_cast<size_t>(q::max_rows) * q::n_idx_head * HD);
            for (auto & v : kn) {
                v = u01(rng);
            }
            for (auto & v : qn) {
                v = u01(rng);
            }
            for (auto & v : qq) {
                v = u11(rng);
            }
            ok = cuda_ok(cudaMemcpyAsync(bn.keys.p, k.data(), n0 * 2, cudaMemcpyHostToDevice, bn.st), "keys") &&
                 cuda_ok(cudaMemcpyAsync(bn.k_norm.p, kn.data(), int64_t(HD) * 4, cudaMemcpyHostToDevice, bn.st),
                         "k_norm") &&
                 cuda_ok(cudaMemcpyAsync(bn.q_norm.p, qn.data(), int64_t(HD) * 4, cudaMemcpyHostToDevice, bn.st),
                         "q_norm") &&
                 cuda_ok(cudaMemcpyAsync(bn.qv.p, qq.data(), qq.size() * 4, cudaMemcpyHostToDevice, bn.st), "q") &&
                 cuda_ok(cudaMemsetAsync(bn.ws_mem.p, 0, q::workspace_bytes(), bn.st), "workspace");
            for (int32_t s = 1; ok && (s + 1) * n0 <= bn.key_pool_elems; ++s) {
                ok = cuda_ok(cudaMemcpyAsync(bn.keys.p + s * n0, bn.keys.p, n0 * 2, cudaMemcpyDeviceToDevice, bn.st),
                             "key sets");
            }
            // summaries: set 0 by the kernel, copied to the other 15 layers
            const int32_t nb = (max_depth + q::max_rows) / R;
            const q::update_args u = {bn.keys.p, bn.k_norm.p, bn.sums.p, 0, nb, q::model_rope()};
            ok = ok && q::summary_update(u, bn.st);
            const size_t s0 = bench::sum_set_elems(max_depth);
            for (int32_t s = 1; ok && (s + 1) * s0 <= bn.sum_pool_elems; ++s) {
                ok = cuda_ok(cudaMemcpyAsync(bn.sums.p + s * s0, bn.sums.p, s0 * 4, cudaMemcpyDeviceToDevice, bn.st),
                             "sum sets");
            }
            ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "prefill");
        }

        if (ok && !dump_dir.empty()) {
            for (const int32_t d : {8192, 65536, 196608}) {
                for (const int32_t r : {1, 3}) {
                    ok = ok &&
                         dump_point(bn, dump_dir, "iso_ssu_d" + std::to_string(d) + "_B" + std::to_string(r), r, d);
                }
            }
            if (ok && fixture != nullptr && fixture[0] != '\0') {
                const int32_t d = 196608, r = 3;
                const q::update_args u = {bn.keys.p, bn.k_norm.p, bn.sums.p, 0, (d + q::max_rows) / R, q::model_rope()};
                ok = load_clustered(fixture, bn, d, r) && q::summary_update(u, bn.st) &&
                     dump_point(bn, dump_dir, "iso_ssu_d196608_B3_clustered", r, d);
            }
            ok = cuda_ok(cudaStreamSynchronize(bn.st), "dump sync") && ok;
            cudaStreamDestroy(bn.st);
            printf("# qa-bench: dump %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
            return ok ? 0 : 1;
        }

        // keep the card busy for warm_s before timing: memsets of a 64 MiB scratch buffer, no kernel of ours
        if (ok) {
            dev_buf<uint8_t> scratch(size_t(64) << 20);
            const auto t0 = std::chrono::steady_clock::now();
            int64_t n_memset = 0;
            while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
                for (int32_t i = 0; i < 32 && ok; ++i, ++n_memset) {
                    ok = cuda_ok(cudaMemsetAsync(scratch.p, static_cast<int>(n_memset & 0xff), size_t(64) << 20, bn.st),
                                 "memset");
                }
                ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "warm sync");
            }
            printf("# qa-bench: busy for %.2f s (%lld memsets of 64 MiB), timing from %s\n",
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                   static_cast<long long>(n_memset), utc_now().c_str());
        }

        // QA1 step 6: score + select alone (the union before C2), summaries rotated over at least 200 MB (or 64 sets)
        const std::vector<int32_t> depths =
            acceptance ? std::vector<int32_t>{196608} : std::vector<int32_t>{8192, 65536, 196608};
        const std::vector<int32_t> rowsv = acceptance ? std::vector<int32_t>{3} : std::vector<int32_t>{1, 3};
        for (const int32_t d : depths) {
            for (const int32_t r : rowsv) {
                const size_t set_b = bench::sum_set_elems(d) * 4;
                const int32_t ns = sets_for(set_b, 16, bn.sum_pool_elems * 4);
                const q::score_grid g = q::grid_of(r, d);
                printf("# qa-bench: depth %d B %d: score grid %d tiles, %d CTAs x %d warps (%d with a tile), depth %d; "
                       "%d summary sets "
                       "of %s (%s)\n",
                       d, r, g.tiles, g.ctas, g.warps, g.tpc, g.depth, ns, mb(set_b).c_str(),
                       mb(set_b * static_cast<size_t>(ns)).c_str());
                const std::string rot = std::to_string(ns) + " summary sets, " + mb(set_b * static_cast<size_t>(ns));
                ok = ok &&
                     time_point(bn, "iso_ssu_d" + std::to_string(d) + "_B" + std::to_string(r), "score + select (QA1b)",
                                r, d, 2, rot, [&](int32_t i) { return bn.select(r, d, i % ns); });
            }
        }
        if (!acceptance) {
            // the summary update alone (1 and 2 blocks), and with score + select, at 196K and B = 3
            const int32_t d = 196608, r = 3;
            const int32_t ns = sets_for(bench::sum_set_elems(d) * 4, 16, bn.sum_pool_elems * 4);
            const int32_t nk = sets_for(bench::key_set_elems(d) * 2, 1, bn.key_pool_elems * 2);
            const std::string rot = std::to_string(ns) + " summary sets, " + std::to_string(nk) + " key sets";
            for (const int32_t cnt : {1, 2}) {
                ok = ok && time_point(bn, "iso_update_" + std::to_string(cnt) + "blk_d196608", "summary update", r, d,
                                      1, rot, [&](int32_t i) { return bn.update(r, d, cnt, i % ns, i % nk); });
            }
            ok = ok && time_point(bn, "ctx_update_ssu_d196608_B3", "summary update (2 blocks) + score + select", r, d,
                                  1 + 2, rot, [&](int32_t i) {
                                      return bn.update(r, d, 2, i % ns, i % nk) && bn.select(r, d, i % ns);
                                  });
        }
        // the HX7 selection pattern at the acceptance point: realistic clustering of the winners (many failing tiles)
        if (ok && fixture != nullptr && fixture[0] != '\0') {
            const int32_t d = 196608, r = 3;
            ok = load_clustered(fixture, bn, d, r);
            const int32_t nb = (d + q::max_rows) / R;
            const q::update_args u = {bn.keys.p, bn.k_norm.p, bn.sums.p, 0, nb, q::model_rope()};
            ok = ok && q::summary_update(u, bn.st);
            const size_t s0 = bench::sum_set_elems(d);
            const int32_t ns = sets_for(s0 * 4, 16, bn.sum_pool_elems * 4);
            for (int32_t s = 1; ok && s < ns; ++s) {
                ok = cuda_ok(cudaMemcpyAsync(bn.sums.p + s * s0, bn.sums.p, s0 * 4, cudaMemcpyDeviceToDevice, bn.st),
                             "sum sets");
            }
            ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "clustered prefill");
            const std::string rot = std::to_string(ns) + " summary sets, " + mb(s0 * 4 * static_cast<size_t>(ns));
            ok = ok && time_point(bn, "iso_ssu_d196608_B3_clustered", "score + select (QA1b), HX7 selection pattern", r,
                                  d, 2, rot, [&](int32_t i) { return bn.select(r, d, i % ns); });
        }
        ok = cuda_ok(cudaStreamSynchronize(bn.st), "final sync") && ok;
        if (bn.st != nullptr) {
            cudaStreamDestroy(bn.st);
        }
    }
    printf("# qa-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
