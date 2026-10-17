#include "fn/kernels/experts/experts_prompt_bench.h"

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_prompt.h"
#include "fn/kernels/experts/experts_prompt_test.h"
#include "fn/kernels/experts/experts_test.h"
#include "fn/model/shapes.h"

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
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace fn::experts;

constexpr int32_t NU = fn::n_expert_used;
constexpr int32_t NE = fn::n_expert;
constexpr int32_t D = fn::n_embd;
constexpr int32_t NFF = fn::n_ff_expert;
constexpr int32_t n_layer = fn::n_layer;
constexpr int32_t n_unit = 16;               // layers captured in a graph, each with its own routing and weights
constexpr int32_t n_warm = 3;                // untimed replays
constexpr int32_t n_timed = 20;              // timed replays
constexpr double warm_s = 2.0;               // HX12 Later 15: a process starts at 1,260 MHz for 0.3-1 s
constexpr size_t min_cold = size_t(1) << 30; // weights rotated over at least this many bytes (EP1 step 7)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "ep-bench: %s: %s\n", what, cudaGetErrorString(e));
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

// owns one device allocation of n bytes
struct dev_mem {
    uint8_t * p = nullptr;
    size_t n = 0;

    explicit dev_mem(size_t bytes) : n(bytes) {
        if (!cuda_ok(cudaMalloc(&p, n), "cudaMalloc")) {
            p = nullptr;
        }
    }

    ~dev_mem() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dev_mem(const dev_mem &) = delete;
    dev_mem & operator=(const dev_mem &) = delete;

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); } // the allocation is 256-byte aligned

    // a pageable copy may return before the DMA lands, and the bench's stream is non-blocking: wait for the device
    bool put(const void * h, size_t bytes, size_t off = 0) const {
        return p != nullptr && off + bytes <= n &&
               cuda_ok(cudaMemcpy(p + off, h, bytes, cudaMemcpyHostToDevice), "put") &&
               cuda_ok(cudaDeviceSynchronize(), "put");
    }
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

// The 16 graph units of one row count: unit s takes R consecutive delivered rows of layer (7 s) mod 48 from row
// (997 s) mod (n - R + 1) of FN4's routes; router weights uniform (FN4 recorded IDs only); activations synthetic.
struct units {
    int32_t rows = 0;
    std::vector<std::vector<int32_t>> ids;
    std::unique_ptr<dev_mem> ids_d, wts_d, x_d; // [16][R x 10], [16][R x 10], F16 [16][R][2560]
    double touched_mean = 0.0;
    int32_t max_rows = 0;
};

// synthetic FP16 rows: n rows of k values, normal FP16 numbers of magnitude 2^-2 .. 2 with signs and mantissas from a
// fixed pattern (values do not change any kernel's work)
std::vector<uint16_t> synth_f16(int32_t n, int32_t k, uint32_t seed) {
    std::vector<uint16_t> h(size_t(n) * size_t(k));
    uint32_t x = seed * 2654435761u + 12345u;
    for (uint16_t & v : h) {
        x = x * 1664525u + 1013904223u;
        v = static_cast<uint16_t>(((x >> 31) << 15) | ((13u + (x >> 20) % 3u) << 10) | ((x >> 8) & 0x3ffu));
    }
    return h;
}

bool make_units(const ep_fn4 & fr, int32_t rows, units & u) {
    const int32_t pairs = rows * NU;
    u.rows = rows;
    u.ids_d = std::make_unique<dev_mem>(size_t(n_unit) * pairs * 4);
    u.wts_d = std::make_unique<dev_mem>(size_t(n_unit) * pairs * 4);
    u.x_d = std::make_unique<dev_mem>(size_t(n_unit) * rows * D * 2);
    const std::vector<float> wts(size_t(pairs), 1.0f / NU);
    bool ok = true;
    double t = 0.0;
    for (int32_t s = 0; s < n_unit && ok; ++s) {
        const int32_t layer = (7 * s) % fn::n_layer, start = (997 * s) % (fr.n_rows - rows + 1);
        u.ids.push_back(ep_window(fr, layer, start, rows));
        const std::vector<int32_t> & ids = u.ids.back();
        std::vector<int32_t> cnt(NE, 0);
        for (const int32_t e : ids) {
            ++cnt[size_t(e)];
        }
        for (const int32_t n : cnt) {
            t += n > 0 ? 1 : 0;
            u.max_rows = std::max(u.max_rows, n);
        }
        const std::vector<uint16_t> x = synth_f16(rows, D, static_cast<uint32_t>(s));
        ok = u.ids_d->put(ids.data(), size_t(pairs) * 4, size_t(s) * pairs * 4) &&
             u.wts_d->put(wts.data(), size_t(pairs) * 4, size_t(s) * pairs * 4) &&
             u.x_d->put(x.data(), x.size() * 2, size_t(s) * rows * D * 2);
    }
    u.touched_mean = t / n_unit;
    return ok;
}

// one format's weights, rotated: n_copies x 512 experts (expert e = synthetic e % n_synth), unit s reads copy
// s % n_copies, in EX1's packed layout of card 0.
struct weights {
    fmt f = fmt::iq3_xxs;
    std::unique_ptr<dev_mem> mem;
    size_t expert_bytes = 0;
    int32_t n_copies = 0;
    const uint8_t * copy(int32_t s) const { return mem->p + size_t(s % n_copies) * NE * expert_bytes; }
};

bool load_weights(fmt f, const char * dir, weights & w) {
    synth s;
    if (!load_synth(f, dir, s)) {
        return false;
    }
    w.f = f;
    const bool dn = f == fmt::q2_0 || f == fmt::iq4_nl;
    w.expert_bytes = static_cast<size_t>(dn ? down_bytes(f) : gate_up_bytes(f));
    std::vector<int32_t> map(NE);
    for (int32_t e = 0; e < NE; ++e) {
        map[size_t(e)] = e < n_synth ? e : n_synth;
    }
    std::vector<uint8_t> h;
    if (!pack_card(s, 0, map, 1, h)) {
        return false;
    }
    const std::vector<uint8_t> one(h.begin(), h.begin() + std::ptrdiff_t(w.expert_bytes) * n_synth); // n_synth experts
    w.n_copies = static_cast<int32_t>((min_cold + w.expert_bytes * NE - 1) / (w.expert_bytes * NE));
    w.mem = std::make_unique<dev_mem>(size_t(w.n_copies) * NE * w.expert_bytes);
    if (w.mem->p == nullptr) {
        return false;
    }
    std::vector<uint8_t> copy(size_t(NE) * w.expert_bytes);
    for (int32_t e = 0; e < NE; ++e) {
        std::memcpy(copy.data() + size_t(e) * w.expert_bytes, one.data() + size_t(e % n_synth) * w.expert_bytes,
                    w.expert_bytes);
    }
    bool ok = true;
    for (int32_t c = 0; c < w.n_copies && ok; ++c) {
        ok = w.mem->put(copy.data(), copy.size(), size_t(c) * copy.size());
    }
    return ok;
}

struct route_set { // one per unit
    dev_mem offsets{size_t(NE + 1) * 4}, sorted{size_t(ep_max_pairs) * 4}, pos{size_t(ep_max_pairs) * 4},
        tasks{size_t(ep_max_tasks) * 16}, counters{32};
    ep_route view() const {
        return {offsets.as<int32_t>(), sorted.as<int32_t>(), pos.as<int32_t>(), tasks.as<int32_t>(),
                counters.as<int32_t>()};
    }
};

struct bench {
    cudaStream_t st = nullptr;
    std::unique_ptr<dev_mem> grids, glu, part, out;
    std::vector<std::unique_ptr<route_set>> routes; // per unit
    std::map<std::string, int64_t> launches;        // product launches of each kernel so far (ncu --launch-skip)

    static std::string key(const char * k, fmt f) { return std::string(k) + " " + fmt_name(f); }

    bool group(const units & u, int32_t s, int32_t task_rows) {
        ++launches["group"];
        return ep_group(u.rows, task_rows, u.ids_d->as<int32_t>() + size_t(s) * u.rows * NU, routes[size_t(s)]->view(),
                        nullptr, st);
    }
    uint16_t * glu_of(const units & u, int32_t s) const {
        return glu->as<uint16_t>() + size_t(s) * u.rows * NU * n_ff_card;
    }
    bool gate_up(const weights & w, const units & u, int32_t s) {
        ++launches[key("gate_up", w.f)];
        const ep_gate_up_args a = {
            w.copy(s),    grids->p, u.x_d->as<uint16_t>() + size_t(s) * u.rows * D, routes[size_t(s)]->view(),
            glu_of(u, s), nullptr};
        return ep_gate_up(w.f, u.rows, a, st);
    }
    bool down(const weights & w, const units & u, int32_t s) {
        ++launches[key("down", w.f)];
        const ep_down_args a = {w.copy(s), grids->p, glu_of(u, s), routes[size_t(s)]->view(), part->as<float>()};
        return ep_down(w.f, u.rows, a, st);
    }
    bool sum(const units & u, int32_t s) {
        ++launches["sum"];
        const ep_sum_args a = {part->as<float>(), routes[size_t(s)]->view(),
                               reinterpret_cast<const float *>(u.wts_d->p) + size_t(s) * u.rows * NU, nullptr,
                               out->as<float>()};
        return ep_sum(u.rows, a, st);
    }
    bool layer(const weights & wg, const weights & wd, const units & u, int32_t s) {
        return group(u, s, ep_task_rows(wg.f)) && gate_up(wg, u, s) && down(wd, u, s) && sum(u, s);
    }
};

struct point_info {
    std::string point, what, path, pair;
    int32_t rows = 0;
    double touched_mean = 0.0, tasks_mean = 0.0, tiles_mean = 0.0;
    int32_t max_rows = 0, task_rows = 0;
    double weight_bytes_per_unit = 0.0;
};

// Captures n_unit units, replays n_warm untimed and n_timed timed, and prints one JSON line. The time of a unit is a
// replay's event time over n_unit; median and p90 (nearest rank) over the timed replays.
bool time_point(bench & bn, const point_info & pi, const std::function<bool(int32_t)> & unit) {
    // one eager pass first: every unit's inputs exist
    bool ok = true;
    for (int32_t i = 0; i < n_unit && ok; ++i) {
        ok = unit(i);
    }
    ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "eager pass");
    if (!ok) {
        return false;
    }
    cudaGraph_t graph = nullptr;
    timed_graph tg;
    if (!cuda_ok(cudaStreamBeginCapture(bn.st, cudaStreamCaptureModeThreadLocal), "begin capture")) {
        return false;
    }
    const std::map<std::string, int64_t> before = bn.launches;
    for (int32_t i = 0; i < n_unit && ok; ++i) {
        ok = unit(i);
    }
    std::map<std::string, int64_t> per_replay;
    for (const auto & kv : bn.launches) {
        const auto it = before.find(kv.first);
        const int64_t n = kv.second - (it == before.end() ? 0 : it->second);
        if (n > 0) {
            per_replay[kv.first] = n;
        }
    }
    bn.launches = before; // capture enqueues nothing: count per replay instead
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
        ok = cuda_ok(cudaGraphLaunch(tg.exec, bn.st), "warm replay") && cuda_ok(cudaStreamSynchronize(bn.st), "warm");
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
    if (!ok) {
        fprintf(stderr, "ep-bench: %s failed\n", pi.point.c_str());
        return false;
    }
    std::vector<double> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    const double median = 0.5 * (sorted[n_timed / 2 - 1] + sorted[n_timed / 2]);
    const double p90 = sorted[(9 * n_timed + 9) / 10 - 1];
    printf("{\"ticket\": \"EP1\", \"point\": \"%s\", \"path\": \"%s\", \"pair\": \"%s\", \"what\": \"%s\", "
           "\"rows\": %d, \"median_us\": %.3f, \"p90_us\": %.3f, \"n\": %d, \"units_per_replay\": %d, "
           "\"warm_replays\": %d, \"touched_mean\": %.2f, \"max_rows_one_expert\": %d, \"task_rows\": %d, "
           "\"tasks_mean\": %.2f, \"row_tiles_mean\": %.2f, \"weight_bytes_per_unit\": %.0f, "
           "\"cache\": \"cold weights (rotated over >= 1 GB), activations and routing per unit\", "
           "\"first_timed_launch\": {",
           pi.point.c_str(), pi.path.c_str(), pi.pair.c_str(), pi.what.c_str(), pi.rows, median, p90, n_timed, n_unit,
           n_warm, pi.touched_mean, pi.max_rows, pi.task_rows, pi.tasks_mean, pi.tiles_mean, pi.weight_bytes_per_unit);
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

bool busy(bench & bn) {
    // keep the card busy for warm_s before timing: memsets of a 64 MiB scratch buffer
    dev_mem scratch(size_t(64) << 20);
    const auto t0 = std::chrono::steady_clock::now();
    int64_t n = 0;
    bool ok = scratch.p != nullptr;
    while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
        for (int32_t i = 0; i < 32 && ok; ++i, ++n) {
            ok = cuda_ok(cudaMemsetAsync(scratch.p, static_cast<int>(n & 0xff), scratch.n, bn.st), "memset");
        }
        ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "warm sync");
    }
    printf("# ep-bench: busy for %.2f s (%lld memsets of 64 MiB), timing from %s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), static_cast<long long>(n),
           utc_now().c_str());
    return ok;
}

} // namespace

int fnfl_ep_bench(int device, const char * mode, const char * cache_dir, const char * routes_path) {
    const bool ncu = strcmp(mode, "ncu") == 0;
    if (!ncu && strcmp(mode, "all") != 0) {
        fprintf(stderr, "ep-bench: mode is all or ncu\n");
        return 2;
    }
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    cudaDeviceProp prop = {};
    int driver = 0, runtime = 0;
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "properties") ||
        !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
        return 1;
    }
    printf("# ep-bench: %s, sm_%d%d, %d SMs, driver API %d, runtime %d, mode %s, start %s\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, driver, runtime, mode, utc_now().c_str());
    ep_fn4 fr;
    if (!ep_read_fn4(routes_path, fr)) {
        return 1;
    }
    printf("# ep-bench: routes %s: %d rows a layer\n", routes_path, fr.n_rows);

    struct pair_t {
        fmt gu, dn;
    };
    const std::vector<pair_t> pairs_all = {
        {fmt::iq3_xxs, fmt::iq4_nl}, {fmt::iq2_xxs, fmt::q2_0}, {fmt::iq3_s, fmt::iq4_nl}};
    const std::vector<pair_t> pairs = ncu ? std::vector<pair_t>{pairs_all[0]} : pairs_all;
    const std::vector<int32_t> row_counts = ncu ? std::vector<int32_t>{1536} : std::vector<int32_t>{128, 512, 1536};
    bool ok = true;
    {
        bench bn;
        ok = cuda_ok(cudaStreamCreateWithFlags(&bn.st, cudaStreamNonBlocking), "stream");
        std::vector<uint8_t> g;
        bn.grids = std::make_unique<dev_mem>(grid_bytes);
        ok = ok && grid_buffer(g) && bn.grids->put(g.data(), g.size());
        bn.glu = std::make_unique<dev_mem>(size_t(n_unit) * ep_max_pairs * n_ff_card * 2);
        bn.part = std::make_unique<dev_mem>(size_t(ep_max_pairs) * D * 4);
        bn.out = std::make_unique<dev_mem>(size_t(ep_max_rows) * D * 4);
        for (int32_t s = 0; s < n_unit; ++s) {
            bn.routes.push_back(std::make_unique<route_set>());
        }
        for (const pair_t & pr : pairs) {
            const std::string pname = std::string(fmt_name(pr.gu)) + "/" + fmt_name(pr.dn);
            weights wg, wd;
            ok = ok && load_weights(pr.gu, cache_dir, wg) && load_weights(pr.dn, cache_dir, wd);
            if (!ok) {
                break;
            }
            printf("# ep-bench: %s: weights %s %d x 512 x %zu B, %s %d x 512 x %zu B\n", pname.c_str(), fmt_name(pr.gu),
                   wg.n_copies, wg.expert_bytes, fmt_name(pr.dn), wd.n_copies, wd.expert_bytes);
            for (const int32_t R : row_counts) {
                units u;
                ok = ok && make_units(fr, R, u);
                if (!ok) {
                    break;
                }
                const int32_t tr = ep_task_rows(pr.gu);
                // the routes and every unit's GLU, once (the isolated down and sum read them)
                double tasks = 0.0, tiles = 0.0;
                for (int32_t s = 0; s < n_unit && ok; ++s) {
                    ok = bn.group(u, s, tr) && bn.gate_up(wg, u, s);
                }
                ok = ok && cuda_ok(cudaStreamSynchronize(bn.st), "setup");
                for (int32_t s = 0; s < n_unit && ok; ++s) {
                    int32_t cnt[2] = {0, 0};
                    std::vector<int32_t> tk(size_t(ep_max_tasks) * 4);
                    ok = cuda_ok(cudaMemcpy(cnt, bn.routes[size_t(s)]->counters.p, 8, cudaMemcpyDeviceToHost),
                                 "counters") &&
                         cuda_ok(cudaMemcpy(tk.data(), bn.routes[size_t(s)]->tasks.p, tk.size() * 4,
                                            cudaMemcpyDeviceToHost),
                                 "tasks");
                    tasks += cnt[0];
                    for (int32_t i = 0; i < cnt[0]; ++i) {
                        tiles += (tk[size_t(i) * 4 + 2] + 7) / 8;
                    }
                }
                ok = ok && busy(bn);
                point_info pi;
                pi.rows = R;
                pi.pair = pname;
                pi.touched_mean = u.touched_mean;
                pi.max_rows = u.max_rows;
                pi.task_rows = tr;
                pi.tasks_mean = tasks / n_unit;
                pi.tiles_mean = tiles / n_unit;
                const std::string sfx =
                    "_" + std::string(fmt_name(pr.gu)) + "_" + fmt_name(pr.dn) + "_R" + std::to_string(R);
                const auto point = [&](const char * name, const char * what, const char * path, double bytes,
                                       const std::function<bool(int32_t)> & fn) {
                    pi.point = std::string(name) + sfx;
                    pi.what = what;
                    pi.path = path;
                    pi.weight_bytes_per_unit = bytes;
                    ok = ok && time_point(bn, pi, fn);
                };
                const double gb = u.touched_mean * double(wg.expert_bytes),
                             db = u.touched_mean * double(wd.expert_bytes);
                if (ncu) {
                    point("ncu_gu", "ep_gate_up alone", "ours", gb, [&](int32_t s) { return bn.gate_up(wg, u, s); });
                    point("ncu_dn", "ep_down alone", "ours", db, [&](int32_t s) { return bn.down(wd, u, s); });
                    continue;
                }
                point("group", "ep_group alone", "ours", 0.0, [&](int32_t s) { return bn.group(u, s, tr); });
                point("gate_up", "ep_gate_up alone", "ours", gb, [&](int32_t s) { return bn.gate_up(wg, u, s); });
                point("down", "ep_down alone", "ours", db, [&](int32_t s) { return bn.down(wd, u, s); });
                point("sum", "ep_sum alone", "ours", 0.0, [&](int32_t s) { return bn.sum(u, s); });
                point("layer", "group + gate_up + down + sum", "ours", gb + db,
                      [&](int32_t s) { return bn.layer(wg, wd, u, s); });
            }
        }
        ok = cuda_ok(cudaStreamSynchronize(bn.st), "final sync") && ok;
        if (bn.st != nullptr) {
            cudaStreamDestroy(bn.st);
        }
    }
    printf("# ep-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
