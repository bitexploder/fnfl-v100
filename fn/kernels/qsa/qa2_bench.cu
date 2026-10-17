#include "fn/kernels/qsa/qa2_bench.h"

#include "fn/kernels/dense/dense_cold.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/proto/xp1/timing.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

namespace at = fn::qsa::attn;
namespace qs = fn::qsa;

constexpr int32_t HD = at::head_dim;
constexpr int32_t R = qs::ratio;
constexpr int32_t n_cold = 32;                      // launches a cold replay, each after a flush and the evictor
constexpr int32_t n_warm = 3;                       // untimed replays
constexpr int32_t n_timed = 20;                     // timed replays
constexpr double warm_s = 2.0;                      // HX12 Later 15: a process starts at 1,260 MHz for 0.3-1 s
constexpr double rotate_bytes = 200e6;              // K and V read over a warm replay
constexpr double hw_gbs = 1021.6;                   // HX2 B1
constexpr double ramp_us = 1.949;                   // HX2 B1's fixed term
constexpr double tail_us = 0.83;                    // HX12's cross-block tail (the ticket's hardware speed)
constexpr double fma_per_us = 72.0 * 64.0 * 1380.0; // FP32 lanes x clock
constexpr int64_t flush_bytes = 12 << 20;           // the L2 flush (XP1 Amendment 4)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "qa2-bench: %s: %s\n", what, cudaGetErrorString(e));
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

struct dev_buf {
    uint8_t * p = nullptr;

    explicit dev_buf(size_t n) {
        if (!cuda_ok(cudaMalloc(&p, n > 0 ? n : 1), "cudaMalloc")) {
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

// values from a hash: BF16 in [-2, 2) for the caches, F32 in [lo, hi) for the inputs (the time does not depend on them)
__global__ void fill_bf16(uint16_t * p, int64_t n, uint32_t seed) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const uint32_t h = (uint32_t(i) ^ seed) * 2654435761u;
        const float f = -2.0f + 4.0f * float(h >> 8) * (1.0f / 16777216.0f);
        p[i] = static_cast<uint16_t>(__float_as_uint(f) >> 16);
    }
}

__global__ void fill_f32(float * p, int64_t n, uint32_t seed, float lo, float hi) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const uint32_t h = (uint32_t(i) ^ seed) * 2654435761u;
        p[i] = lo + (hi - lo) * float(h >> 8) * (1.0f / 16777216.0f);
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

struct point {
    std::string name;
    int32_t depth, rows;
    bool hx7;
    double forecast_us; // the frozen forecast (work/QA2/results.json), or 0 where none was frozen
};

// the B rows' bodies: each row 512 whole blocks and its tail; rows (and sets in a copy) take disjoint random blocks,
// or HX7's three selections
std::vector<uint32_t> make_body(int32_t depth, int32_t rows, const std::vector<int32_t> & blocks, size_t first) {
    std::vector<uint32_t> b;
    for (int32_t t = 0; t < rows; ++t) {
        const int32_t q = depth + t, tail = (q + 1) % R;
        std::vector<int32_t> sel(blocks.begin() + int64_t(first) + int64_t(t) * 512,
                                 blocks.begin() + int64_t(first) + int64_t(t + 1) * 512);
        std::sort(sel.begin(), sel.end());
        b.push_back(512);
        b.push_back(uint32_t(tail));
        for (const int32_t x : sel) {
            b.push_back(uint32_t(x));
        }
        for (int32_t i = 0; i < tail; ++i) {
            b.push_back(uint32_t((q + 1) / R * R + i));
        }
    }
    return b;
}

bool hx7_blocks(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path, std::ios::binary);
    int32_t hdr[4] = {};
    std::vector<int32_t> cells(size_t(3) * 2051);
    if (!f.read(reinterpret_cast<char *>(hdr), sizeof(hdr)) || hdr[0] != 0x31415141 || hdr[2] != 3 || hdr[3] != 2051 ||
        !f.read(reinterpret_cast<char *>(cells.data()), static_cast<std::streamsize>(cells.size() * 4))) {
        fprintf(stderr, "qa2-bench: %s is not QA1's HX7 fixture\n", path.c_str());
        return false;
    }
    out.clear();
    for (int32_t t = 0; t < 3; ++t) {
        for (int32_t i = 0; i < 2048; i += R) {
            out.push_back(cells[size_t(t) * 2051 + i] / R);
        }
    }
    return true;
}

} // namespace

int fnfl_qa2_bench(int device, const std::string & mode, const std::string & fixture) {
    const bool ncu = mode == "ncu", diag = mode == "diag";
    if (!ncu && !diag && mode != "all") {
        fprintf(stderr, "qa2-bench: mode is all, ncu or diag\n");
        return 2;
    }
    cudaDeviceProp prop = {};
    int driver = 0, runtime = 0;
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice") || !cuda_ok(cudaGetDeviceProperties(&prop, device), "props") ||
        !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
        return 1;
    }
    at::resources res = {};
    if (!at::kernel_resources(res)) {
        return 1;
    }
    printf("# qa2-bench: %s, sm_%d%d, %d SMs, driver API %d, runtime %d, mode %s, start %s\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount, driver, runtime, mode.c_str(), utc_now().c_str());
    printf("# qa2-bench: attend %d registers, %d B local, %d B shared, %d CTAs an SM\n", res.regs, res.local_bytes,
           res.shared_bytes, res.ctas_per_sm);
    std::vector<int32_t> hx7;
    if (!hx7_blocks(fixture, hx7)) {
        return 1;
    }
    std::vector<point> points;
    if (ncu) {
        points.push_back({"hx7 d196608 B3", 196608, 3, true, 11.46});
    } else {
        for (const int32_t d : {8192, 65536, 196608}) {
            for (const int32_t B : {1, 3}) {
                points.push_back({"d" + std::to_string(d) + " B" + std::to_string(B) + " disjoint", d, B, false,
                                  B == 1 ? 7.96 : 11.46});
            }
        }
        points.push_back({"hx7 d196608 B3", 196608, 3, true, 11.46});
    }
    cudaStream_t st = nullptr;
    dev_buf io(size_t(at::max_rows) * (at::n_head * at::q_stride + 2 * HD + at::out_width + HD) * 4 +
               size_t(at::max_rows) * (at::out_width + at::out_width / 4) + 4096),
        aws(at::workspace_bytes()), flush(flush_bytes), evict_out(256), probe_buf(16), stamps_buf(8 * 8192);
    bool ok = io.p && aws.p && flush.p && evict_out.p && probe_buf.p &&
              cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    // the inputs, shared by every set (the producers' outputs, warm in L2 as in a layer)
    float * qfull = reinterpret_cast<float *>(io.p);
    float * kvk = qfull + at::max_rows * at::n_head * at::q_stride;
    float * kvv = kvk + at::max_rows * HD;
    float * out = kvv + at::max_rows * HD;
    float * norms = out + at::max_rows * at::out_width; // q_norm, k_norm
    int8_t * codes = reinterpret_cast<int8_t *>(norms + 2 * HD);
    float * scales = reinterpret_cast<float *>(codes + at::max_rows * at::out_width);
    int32_t * pos = reinterpret_cast<int32_t *>(scales + at::max_rows * at::out_width / 16);
    const at::workspace ws = at::carve(aws.p);
    ok = ok && cuda_ok(cudaMemsetAsync(aws.p, 0, at::workspace_bytes(), st), "workspace") &&
         cuda_ok(cudaMemsetAsync(flush.p, 0, flush_bytes, st), "flush");
    if (ok) {
        fill_f32<<<64, 256, 0, st>>>(qfull, at::max_rows * at::n_head * at::q_stride, 1u, -1.5f, 1.5f);
        fill_f32<<<16, 256, 0, st>>>(kvk, 2 * at::max_rows * HD, 2u, -2.0f, 2.0f);
        fill_f32<<<2, 256, 0, st>>>(norms, 2 * HD, 3u, 0.5f, 1.5f);
        ok = cuda_ok(cudaPeekAtLastError(), "fill");
    }
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    ok = ok && cuda_ok(cudaEventCreate(&e0), "event") && cuda_ok(cudaEventCreate(&e1), "event");
    bool busy = false;
    for (const point & pt : points) {
        if (!ok) {
            break;
        }
        const int32_t B = pt.rows, d = pt.depth, n_blocks = d / R;
        std::vector<int32_t> pos_h(size_t(4) * B);
        for (int p = 0; p < 4; ++p) {
            for (int32_t t = 0; t < B; ++t) {
                pos_h[size_t(p) * B + t] = d + t;
            }
        }
        // sets: (copy, body); a copy is a whole cache, a body a disjoint choice of blocks in it
        int64_t union_cells = 0;
        int32_t bodies_per_copy = 1;
        std::vector<int32_t> perm(static_cast<size_t>(n_blocks));
        for (int32_t i = 0; i < n_blocks; ++i) {
            perm[size_t(i)] = i;
        }
        std::mt19937 g(uint32_t(d * 7 + B));
        std::shuffle(perm.begin(), perm.end(), g);
        if (pt.hx7) {
            std::set<int32_t> u(hx7.begin(), hx7.end());
            union_cells = int64_t(u.size()) * R + B;
        } else {
            union_cells = int64_t(B) * 2048 + B;
            bodies_per_copy = n_blocks / (512 * B);
        }
        const int32_t n_set = ncu ? 4 : std::max(2, int32_t(std::ceil(rotate_bytes / (double(union_cells) * 1024.0))));
        const int32_t n_copy = (n_set + bodies_per_copy - 1) / bodies_per_copy;
        const size_t cache_elems = (size_t(d) + at::max_rows) * HD;
        std::unique_ptr<dev_buf> caches(new dev_buf(size_t(n_copy) * cache_elems * 2 * 2));
        const int32_t words = qs::body_words(d, B);
        std::unique_ptr<dev_buf> bodies(new dev_buf(size_t(n_set) * (size_t(words) + at::max_rows + 1) * 4));
        if (caches->p == nullptr || bodies->p == nullptr) {
            ok = false;
            break;
        }
        uint16_t * kc0 = reinterpret_cast<uint16_t *>(caches->p);
        fill_bf16<<<1024, 256, 0, st>>>(kc0, int64_t(n_copy) * cache_elems * 2, uint32_t(d + B));
        std::vector<uint32_t> bod;
        std::vector<int32_t> offs;
        for (int32_t t = 0; t <= B; ++t) {
            offs.push_back(qs::body_offset(d, t));
        }
        offs.push_back(d); // the depth on the device, after the offsets (TG1: attend reads it there)
        for (int32_t s = 0; s < n_set; ++s) {
            const std::vector<uint32_t> b =
                pt.hx7 ? make_body(d, B, hx7, 0) : make_body(d, B, perm, size_t(s % bodies_per_copy) * 512 * B);
            bod.insert(bod.end(), b.begin(), b.end());
        }
        const size_t off_at = size_t(n_set) * size_t(words);
        ok = ok && bod.size() == off_at &&
             cuda_ok(cudaMemcpyAsync(bodies->p, bod.data(), off_at * 4, cudaMemcpyHostToDevice, st), "bodies") &&
             cuda_ok(cudaMemcpyAsync(bodies->p + off_at * 4, offs.data(), offs.size() * 4, cudaMemcpyHostToDevice, st),
                     "offsets") &&
             cuda_ok(cudaMemcpyAsync(pos, pos_h.data(), pos_h.size() * 4, cudaMemcpyHostToDevice, st), "pos") &&
             cuda_ok(cudaStreamSynchronize(st), "setup");
        auto args_of = [&](int32_t s, bool with_stamps = false) {
            at::args a = {};
            const int32_t copy = s / bodies_per_copy;
            a.qfull = qfull;
            a.kv_k = kvk;
            a.kv_v = kvv;
            a.q_norm = norms;
            a.k_norm = norms + HD;
            a.body = reinterpret_cast<const uint32_t *>(bodies->p) + size_t(s) * size_t(words);
            a.offsets = reinterpret_cast<const int32_t *>(bodies->p + off_at * 4);
            a.pos = pos;
            a.k_cache = kc0 + size_t(copy) * cache_elems * 2;
            a.v_cache = a.k_cache + cache_elems;
            a.depth = reinterpret_cast<const int32_t *>(bodies->p + off_at * 4) + B + 1;
            a.max_cells = d + at::max_rows;
            a.theta_scale = at::rope_theta_scale();
            a.out = out;
            a.codes = codes;
            a.scales = scales;
            a.k_f32 = nullptr;
            a.stamps = with_stamps ? reinterpret_cast<uint64_t *>(stamps_buf.p) : nullptr;
            a.ws = ws;
            return a;
        };
        // keep the card busy for warm_s before the first timing
        if (ok && !busy) {
            const auto t0 = std::chrono::steady_clock::now();
            while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
                for (int32_t i = 0; i < 16 && ok; ++i) {
                    ok = cuda_ok(cudaMemsetAsync(flush.p, i, flush_bytes, st), "memset");
                }
                ok = ok && cuda_ok(cudaStreamSynchronize(st), "busy");
            }
            busy = true;
        }
        if (ncu) {
            // 3 untimed replays of 4 sets, then one more: profile launches 25 and 26 (attend, finish)
            cudaGraph_t gr = nullptr;
            cudaGraphExec_t ex = nullptr;
            ok = ok && cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            for (int32_t s = 0; s < n_set && ok; ++s) {
                ok = at::attend(B, args_of(s), st);
            }
            ok = cuda_ok(cudaStreamEndCapture(st, &gr), "end capture") && ok;
            ok = ok && cuda_ok(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
            for (int32_t i = 0; i < 4 && ok; ++i) {
                ok = cuda_ok(cudaGraphLaunch(ex, st), "replay") && cuda_ok(cudaStreamSynchronize(st), "replay");
            }
            if (ex != nullptr) {
                cudaGraphExecDestroy(ex);
            }
            if (gr != nullptr) {
                cudaGraphDestroy(gr);
            }
            printf("# qa2-bench: ncu point %s, %d sets, %s\n", pt.name.c_str(), n_set, ok ? "done" : "FAILED");
            continue;
        }
        if (diag) {
            // the warm graph with stamps on its last step: each CTA's phase times from its own entry, and the prepare
            // tasks' publications from the earliest entry
            cudaGraph_t gr = nullptr;
            cudaGraphExec_t ex = nullptr;
            ok = ok && cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            for (int32_t s = 0; s < n_set && ok; ++s) {
                ok = at::attend(B, args_of(s, s == n_set - 1), st);
            }
            ok = cuda_ok(cudaStreamEndCapture(st, &gr), "end capture") && ok;
            ok = ok && cuda_ok(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
            for (int32_t i = 0; i < 4 && ok; ++i) {
                ok = cuda_ok(cudaGraphLaunch(ex, st), "replay") && cuda_ok(cudaStreamSynchronize(st), "replay");
            }
            const int32_t g = at::grid(B, device), n_task = 13 * B;
            std::vector<uint64_t> t(size_t(g) * 10 + size_t(n_task));
            ok = ok && cuda_ok(cudaMemcpy(t.data(), stamps_buf.p, t.size() * 8, cudaMemcpyDeviceToHost), "stamps");
            if (ex != nullptr) {
                cudaGraphExecDestroy(ex);
            }
            if (gr != nullptr) {
                cudaGraphDestroy(gr);
            }
            if (!ok) {
                break;
            }
            uint64_t t0 = ~0ull;
            for (int32_t c = 0; c < g; ++c) {
                t0 = std::min(t0, t[size_t(c) * 10]);
            }
            const char * names[10] = {"entry",  "cell ids", "acquired", "pass 1",    "softmax",
                                      "pass 2", "unit end", "q ready",  "grid sync", "finish"};
            printf("{\"ticket\": \"QA2\", \"diag\": \"%s\", \"ctas\": %d", pt.name.c_str(), g);
            for (int32_t k = 0; k < 10; ++k) {
                std::vector<double> v;
                for (int32_t c = 0; c < g; ++c) {
                    v.push_back(double(t[size_t(c) * 10 + k] - t0) / 1e3);
                }
                std::sort(v.begin(), v.end());
                printf(", \"%s_us\": [%.2f, %.2f, %.2f]", names[k], v.front(), v[v.size() / 2], v.back());
            }
            std::vector<double> p;
            for (int32_t i = 0; i < n_task; ++i) {
                p.push_back(double(t[size_t(g) * 10 + size_t(i)] - t0) / 1e3);
            }
            std::sort(p.begin(), p.end());
            printf(", \"tasks_published_us\": [%.2f, %.2f, %.2f], \"note\": \"[min, median, max] over CTAs from the "
                   "earliest entry\"}\n",
                   p.front(), p[p.size() / 2], p.back());
            fflush(stdout);
            continue;
        }
        // warm: n_set steps back to back on rotated sets; cold: n_cold x (flush, evictor, step) against the same graph
        // without the steps
        cudaGraph_t gr[3] = {nullptr, nullptr, nullptr};
        cudaGraphExec_t ex[3] = {nullptr, nullptr, nullptr};
        for (int32_t k = 0; k < 3 && ok; ++k) {
            ok = cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            const int32_t n = k == 0 ? n_set : n_cold;
            for (int32_t i = 0; i < n && ok; ++i) {
                if (k > 0) {
                    ok = fn::dense::cold_flush(flush.p, flush_bytes, st) &&
                         fn::dense::cold_icache(reinterpret_cast<float *>(evict_out.p), st);
                }
                if (ok && k < 2) {
                    ok = at::attend(B, args_of(i % n_set), st);
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
        std::vector<double> warm, cold;
        const std::string t0 = utc_now();
        for (int32_t i = 0; i < n_timed && ok; ++i) {
            double us[3] = {0.0, 0.0, 0.0};
            for (int32_t k = 0; k < 3 && ok; ++k) {
                ok = replay_us(ex[k], st, e0, e1, us[k]);
            }
            warm.push_back(us[0] / n_set);
            cold.push_back((us[1] - us[2]) / n_cold);
        }
        const std::string t1 = utc_now();
        ok = ok && sm_mhz(reinterpret_cast<uint64_t *>(probe_buf.p), st, mhz1);
        uint32_t err = 0;
        ok = ok && cuda_ok(cudaMemcpyAsync(&err, ws.error, 4, cudaMemcpyDeviceToHost, st), "error") &&
             cuda_ok(cudaStreamSynchronize(st), "error");
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
        const timing w = stats(warm), c = stats(cold);
        const double bytes = double(union_cells) * 1024.0 + 0.1e6 * B / 3.0;
        const double fmas = double(B) * at::max_sel * 2.0 * at::n_head * HD;
        const double hw = ramp_us + std::max(bytes / (hw_gbs * 1e3), fmas / fma_per_us) + tail_us;
        printf(
            "{\"ticket\": \"QA2\", \"point\": \"%s\", \"rows\": %d, \"depth\": %d, \"union_cells\": %lld, "
            "\"union_mb\": %.2f, \"hw_us\": %.3f, \"forecast_us\": %.2f, \"gate_us\": %.2f, \"warm_us\": %.3f, "
            "\"warm_p90_us\": %.3f, \"warm_pct_hw\": %.1f, \"warm_vs_forecast\": %.3f, \"cold_us\": %.3f, "
            "\"cold_p90_us\": %.3f, \"cold_minus_warm_us\": %.3f, \"sets\": %d, \"copies\": %d, \"rotated_mb\": %.0f, "
            "\"launches_cold\": %d, \"timed\": %d, \"error_word\": %u, \"sm_mhz_before\": %.0f, \"sm_mhz_after\": "
            "%.0f, "
            "\"t_start\": \"%s\", \"t_end\": \"%s\"}\n",
            pt.name.c_str(), B, d, static_cast<long long>(union_cells), bytes / 1e6, hw, pt.forecast_us,
            1.15 * pt.forecast_us, w.median, w.p90, 100.0 * hw / w.median, w.median / pt.forecast_us, c.median, c.p90,
            c.median - w.median, n_set, n_copy, double(n_set) * double(union_cells) * 1024.0 / 1e6, n_cold, n_timed,
            err, mhz0, mhz1, t0.c_str(), t1.c_str());
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
    printf("# qa2-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
