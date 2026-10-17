#include "fn/kernels/draft/draft_bench.h"

#include "fn/kernels/dense/dense_cold.h"
#include "fn/kernels/draft/draft.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/proto/xp1/timing.h"
#include "fn/xchg/xchg.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace dr = fn::draft;
namespace xc = fn::xchg;

constexpr int32_t HD = dr::head_dim;
constexpr int32_t n_cold = 32;            // launches a cold replay, each after a flush and the evictor
constexpr int32_t n_warm = 3;             // untimed replays
constexpr int32_t n_timed = 20;           // timed replays
constexpr int32_t n_xchg = 32;            // exchanges a draft-in graph
constexpr double warm_s = 2.0;            // HX12 Later 15: a process starts at 1,260 MHz for 0.3-1 s
constexpr double rotate_bytes = 200e6;    // K and V read over a warm replay
constexpr double hw_gbs = 1021.6;         // HX2 B1
constexpr double ramp_us = 1.949;         // HX2 B1's fixed term
constexpr int64_t flush_bytes = 12 << 20; // the L2 flush (XP1 Amendment 4)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "draft-kernel-bench: %s: %s\n", what, cudaGetErrorString(e));
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

// device memory on the current device
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

    template <typename T> T * as(size_t at_bytes = 0) const { return reinterpret_cast<T *>(p + at_bytes); }
};

// values from a hash (the time does not depend on them): F16 in [-2, 2) for the caches, F32 in [lo, hi) for inputs
__global__ void fill_f16(uint16_t * p, int64_t n, uint32_t seed) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        const uint32_t h = (uint32_t(i) ^ seed) * 2654435761u;
        p[i] = __half_as_ushort(__float2half_rn(-2.0f + 4.0f * float(h >> 8) * (1.0f / 16777216.0f)));
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

// keeps the card busy for warm_s before its first timing
bool busy_wait(void * scratch, cudaStream_t st) {
    bool ok = true;
    const auto t0 = std::chrono::steady_clock::now();
    while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
        for (int32_t i = 0; i < 16 && ok; ++i) {
            ok = cuda_ok(cudaMemsetAsync(scratch, i, flush_bytes, st), "memset");
        }
        ok = ok && cuda_ok(cudaStreamSynchronize(st), "busy");
    }
    return ok;
}

void destroy(cudaGraph_t & g, cudaGraphExec_t & e) {
    if (e != nullptr) {
        cudaGraphExecDestroy(e);
    }
    if (g != nullptr) {
        cudaGraphDestroy(g);
    }
    e = nullptr;
    g = nullptr;
}

// one card's DA-TC inputs (the producers' outputs, warm in L2 as in a step), its scratch and the error word
struct datc_io {
    dev_buf io{size_t(dr::max_rows) * (dr::n_head * dr::q_stride + 2 * HD) * 4 + 2 * HD * 4 +
               size_t(dr::max_rows) * dr::n_head * HD * 2 + dr::out_width * 4 + dr::out_width + dr::out_width / 8 * 4 +
               64 * 4};
    dev_buf part{dr::part_bytes()};
    dev_buf rel{size_t(xc::slot_bytes(xc::msg::hc) + xc::slot_bytes(xc::msg::hmix))};
    float *qfull = nullptr, *kvk = nullptr, *kvv = nullptr, *norms = nullptr, *out = nullptr, *scales = nullptr;
    uint16_t * q16 = nullptr;
    int8_t * codes = nullptr;
    int32_t * ints = nullptr; // [0] n_before, [1] stride, [8 .. 8 + 16) the rope planes [4][4]
    uint32_t * err = nullptr;

    bool carve() {
        if (io.p == nullptr || part.p == nullptr || rel.p == nullptr) {
            return false;
        }
        qfull = io.as<float>();
        kvk = qfull + dr::max_rows * dr::n_head * dr::q_stride;
        kvv = kvk + dr::max_rows * HD;
        norms = kvv + dr::max_rows * HD;
        out = norms + 2 * HD;
        scales = out + dr::out_width;
        q16 = reinterpret_cast<uint16_t *>(scales + dr::out_width / 8);
        codes = reinterpret_cast<int8_t *>(q16 + dr::max_rows * dr::n_head * HD);
        ints = reinterpret_cast<int32_t *>(codes + dr::out_width);
        err = reinterpret_cast<uint32_t *>(ints + 32);
        return true;
    }
};

dr::prepare_args prepare_of(const datc_io & d, bool step, uint16_t * kc, uint16_t * vc, int32_t max_cells) {
    dr::prepare_args p = {};
    p.qfull = step ? d.qfull : nullptr;
    p.kv_k = d.kvk;
    p.kv_v = d.kvv;
    p.q_norm = d.norms;
    p.k_norm = d.norms + HD;
    p.n_before = d.ints;
    p.pos = d.ints + 8;
    p.stride = d.ints + 1;
    p.row0 = 0;
    p.theta_scale = fn::qsa::attn::rope_theta_scale();
    p.k_cache = kc;
    p.v_cache = vc;
    p.max_cells = max_cells;
    p.q16 = d.q16;
    p.err = d.err;
    p.release_hc = step ? nullptr : d.rel.as<float>();
    p.release_hmix = step ? nullptr : d.rel.as<float>(size_t(xc::slot_bytes(xc::msg::hc)));
    return p;
}

dr::attend_args attend_of(const datc_io & d, const uint16_t * kc, const uint16_t * vc, int32_t max_cells) {
    dr::attend_args a = {};
    a.q16 = d.q16;
    a.qfull = d.qfull;
    a.k_cache = kc;
    a.v_cache = vc;
    a.n_before = d.ints;
    a.row0 = 0;
    a.max_cells = max_cells;
    a.part = d.part.as<float>();
    a.out = d.out;
    a.codes = d.codes;
    a.scales = d.scales;
    a.err = d.err;
    return a;
}

// the inputs of a point: n_before = depth, every rope plane the position (stride: the rows)
bool set_point(const datc_io & d, int32_t depth, int32_t rows, cudaStream_t st) {
    int32_t h[32] = {};
    h[0] = depth;
    h[1] = rows;
    for (int32_t pl = 0; pl < 4; ++pl) {
        for (int32_t t = 0; t < rows; ++t) {
            h[8 + pl * rows + t] = depth + t;
        }
    }
    return cuda_ok(cudaMemcpyAsync(d.ints, h, sizeof(h), cudaMemcpyHostToDevice, st), "point") &&
           cuda_ok(cudaMemsetAsync(d.err, 0, 4, st), "error word") && cuda_ok(cudaStreamSynchronize(st), "point");
}

struct datc_point {
    int32_t depth;
    double lo, mid, hi; // Stage A section 3's frozen DA-TC forecast, us
};

// the step's row (prepare with q, attend) warm and cold at one depth, and prepare alone; one JSON line each
bool bench_datc(int32_t dev, const datc_point & pt, datc_io & d, void * flush, float * evict, uint64_t * probe,
                cudaStream_t st, cudaEvent_t e0, cudaEvent_t e1) {
    const int32_t max_cells = pt.depth + dr::max_rows;
    const double copy_bytes = double(max_cells) * HD * 2 * 2;
    const int32_t n_copy = std::max(2, int32_t(std::ceil(rotate_bytes / copy_bytes)));
    const size_t cache_elems = size_t(max_cells) * HD;
    std::unique_ptr<dev_buf> caches(new dev_buf(size_t(n_copy) * cache_elems * 2 * 2));
    if (caches->p == nullptr || !set_point(d, pt.depth, 1, st)) {
        return false;
    }
    uint16_t * kc0 = caches->as<uint16_t>();
    fill_f16<<<1024, 256, 0, st>>>(kc0, int64_t(n_copy) * int64_t(cache_elems) * 2, uint32_t(pt.depth));
    bool ok = cuda_ok(cudaPeekAtLastError(), "fill") && cuda_ok(cudaStreamSynchronize(st), "fill");
    auto kc = [&](int32_t c) { return kc0 + size_t(c) * cache_elems * 2; };
    auto vc = [&](int32_t c) { return kc0 + size_t(c) * cache_elems * 2 + cache_elems; };
    // 0: warm (n_copy rows back to back), 1: cold (n_cold x flush, evictor, row), 2: the cold graph's scaffolding,
    // 3: prepare alone warm (n_copy)
    cudaGraph_t gr[4] = {};
    cudaGraphExec_t ex[4] = {};
    for (int32_t k = 0; k < 4 && ok; ++k) {
        ok = cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        const int32_t n = k == 1 || k == 2 ? n_cold : n_copy;
        for (int32_t i = 0; i < n && ok; ++i) {
            const int32_t c = i % n_copy;
            if (k == 1 || k == 2) {
                ok = fn::dense::cold_flush(flush, flush_bytes, st) && fn::dense::cold_icache(evict, st);
            }
            if (ok && k != 2) {
                ok = dr::datc_prepare(1, prepare_of(d, true, kc(c), vc(c), max_cells), st);
            }
            if (ok && k < 2) {
                ok = dr::datc_attend(attend_of(d, kc(c), vc(c), max_cells), st);
            }
        }
        ok = cuda_ok(cudaStreamEndCapture(st, &gr[k]), "end capture") && ok;
        ok = ok && cuda_ok(cudaGraphInstantiate(&ex[k], gr[k], 0), "instantiate");
    }
    double mhz0 = 0.0, mhz1 = 0.0;
    ok = ok && sm_mhz(probe, st, mhz0);
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        for (int32_t k = 0; k < 4 && ok; ++k) {
            ok = cuda_ok(cudaGraphLaunch(ex[k], st), "warm replay") && cuda_ok(cudaStreamSynchronize(st), "warm");
        }
    }
    std::vector<double> warm, cold, prep;
    const std::string t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        double us[4] = {};
        for (int32_t k = 0; k < 4 && ok; ++k) {
            ok = replay_us(ex[k], st, e0, e1, us[k]);
        }
        warm.push_back(us[0] / n_copy);
        cold.push_back((us[1] - us[2]) / n_cold);
        prep.push_back(us[3] / n_copy);
    }
    const std::string t1 = utc_now();
    ok = ok && sm_mhz(probe, st, mhz1);
    uint32_t err = 0;
    ok = ok && cuda_ok(cudaMemcpyAsync(&err, d.err, 4, cudaMemcpyDeviceToHost, st), "error") &&
         cuda_ok(cudaStreamSynchronize(st), "error");
    for (int32_t k = 0; k < 4; ++k) {
        destroy(gr[k], ex[k]);
    }
    if (!ok) {
        return false;
    }
    const timing w = stats(warm), c = stats(cold), p = stats(prep);
    const double kv_mb = double(pt.depth + 1) * 1024.0 / 1e6;
    const double floor_us = ramp_us + kv_mb * 1e6 / (hw_gbs * 1e3);
    printf("{\"ticket\": \"DR1\", \"point\": \"DA-TC step row d%d\", \"device\": %d, \"depth\": %d, \"kv_mb\": %.2f, "
           "\"floor_us\": %.2f, \"forecast_lo_mid_hi_us\": [%.1f, %.1f, %.1f], \"warm_us\": %.3f, \"warm_p90_us\": "
           "%.3f, \"warm_pct_floor\": %.1f, \"warm_vs_mid\": %.3f, \"cold_us\": %.3f, \"cold_p90_us\": %.3f, "
           "\"prepare_us\": %.3f, \"prepare_p90_us\": %.3f, \"copies\": %d, \"rotated_mb\": %.0f, \"launches_cold\": "
           "%d, \"timed\": %d, \"error_word\": %u, \"sm_mhz_before\": %.0f, \"sm_mhz_after\": %.0f, \"cache\": \"warm: "
           "code warm, K/V rotated (DRAM); cold: L2 flush + icache evictor a launch\", \"t_start\": \"%s\", \"t_end\": "
           "\"%s\"}\n",
           pt.depth, dev, pt.depth, kv_mb, floor_us, pt.lo, pt.mid, pt.hi, w.median, w.p90, 100.0 * floor_us / w.median,
           w.median / pt.mid, c.median, c.p90, p.median, p.p90, n_copy, double(n_copy) * copy_bytes / 1e6, n_cold,
           n_timed, err, mhz0, mhz1, t0.c_str(), t1.c_str());
    fflush(stdout);
    return true;
}

// the catch-up's prepare (K/V only, the slot release) at B_c = 1..4, warm, at 4,096 cells
bool bench_prepare_catchup(int32_t dev, datc_io & d, uint64_t * probe, cudaStream_t st, cudaEvent_t e0,
                           cudaEvent_t e1) {
    const int32_t depth = 4096, max_cells = depth + dr::max_rows;
    dev_buf cache(size_t(max_cells) * HD * 2 * 2);
    if (cache.p == nullptr) {
        return false;
    }
    uint16_t * kc = cache.as<uint16_t>();
    uint16_t * vc = kc + size_t(max_cells) * HD;
    bool ok = true;
    for (int32_t rows = 1; rows <= dr::max_rows && ok; ++rows) {
        ok = set_point(d, depth, rows, st);
        cudaGraph_t gr = nullptr;
        cudaGraphExec_t ex = nullptr;
        constexpr int32_t n = 32;
        ok = ok && cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        for (int32_t i = 0; i < n && ok; ++i) {
            ok = dr::datc_prepare(rows, prepare_of(d, false, kc, vc, max_cells), st);
        }
        ok = cuda_ok(cudaStreamEndCapture(st, &gr), "end capture") && ok;
        ok = ok && cuda_ok(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
        double mhz = 0.0;
        ok = ok && sm_mhz(probe, st, mhz);
        for (int32_t i = 0; i < n_warm && ok; ++i) {
            ok = cuda_ok(cudaGraphLaunch(ex, st), "warm replay") && cuda_ok(cudaStreamSynchronize(st), "warm");
        }
        std::vector<double> v;
        for (int32_t i = 0; i < n_timed && ok; ++i) {
            double us = 0.0;
            ok = replay_us(ex, st, e0, e1, us);
            v.push_back(us / n);
        }
        destroy(gr, ex);
        if (ok) {
            const timing t = stats(v);
            printf("{\"ticket\": \"DR1\", \"point\": \"DA-TC catch-up prepare B_c %d\", \"device\": %d, \"warm_us\": "
                   "%.3f, \"warm_p90_us\": %.3f, \"launches_a_replay\": %d, \"sm_mhz_before\": %.0f, \"cache\": "
                   "\"back to back (inputs and slots warm)\"}\n",
                   rows, dev, t.median, t.p90, n, mhz);
            fflush(stdout);
        }
    }
    return ok;
}

// X-DRAFT-IN on both cards at rows 1 and 4
bool bench_draft_in(int32_t dev0, int32_t dev1) {
    if (!xc::enable_peer_access(dev0, dev1)) {
        return false;
    }
    xc::rings rg(dev0, dev1);
    if (!rg.ok()) {
        fprintf(stderr, "draft-kernel-bench: rings\n");
        return false;
    }
    constexpr int32_t H = dr::half, W = xc::words_draft;
    const size_t in_floats = size_t(dr::max_rows) * dr::n_stream * H * 2 + size_t(dr::n_stream) * H * 3 + H +
                             size_t(dr::max_rows) * W + size_t(dr::max_rows) * dr::n_stream * dr::in_runs * 2;
    const size_t code_bytes = size_t(dr::max_rows) * dr::n_stream * dr::in_width;
    std::unique_ptr<dev_buf> buf[2], scratch[2], probe[2];
    cudaStream_t st[2] = {};
    cudaEvent_t ev[2][2] = {};
    bool ok = true;
    for (int32_t c = 0; c < 2 && ok; ++c) {
        ok = cuda_ok(cudaSetDevice(rg.device(c)), "set device") &&
             cuda_ok(cudaStreamCreateWithFlags(&st[c], cudaStreamNonBlocking), "stream") &&
             cuda_ok(cudaEventCreate(&ev[c][0]), "event") && cuda_ok(cudaEventCreate(&ev[c][1]), "event");
        buf[c].reset(new dev_buf(in_floats * 4 + code_bytes + 64));
        scratch[c].reset(new dev_buf(flush_bytes));
        probe[c].reset(new dev_buf(16));
        ok = ok && buf[c]->p != nullptr && scratch[c]->p != nullptr && probe[c]->p != nullptr;
        if (ok) {
            fill_f32<<<64, 256, 0, st[c]>>>(buf[c]->as<float>(), int64_t(in_floats), uint32_t(7 + c), 0.1f, 1.5f);
            ok = cuda_ok(cudaMemsetAsync(buf[c]->p + in_floats * 4 + code_bytes, 0, 64, st[c]), "hsrc") &&
                 cuda_ok(cudaPeekAtLastError(), "fill") && busy_wait(scratch[c]->p, st[c]);
        }
    }
    auto args = [&](int32_t c, int32_t s, dr::send_args & a, dr::recv_args & b) {
        float * f = buf[c]->as<float>();
        float * emb = f;
        float * carry = emb + size_t(dr::max_rows) * dr::n_stream * H;
        float * carry_prev = carry + size_t(dr::max_rows) * dr::n_stream * H;
        float * h_nextn = carry_prev + size_t(dr::n_stream) * H;
        float * g_h = h_nextn + size_t(dr::n_stream) * H;
        float * g_e = g_h + size_t(dr::n_stream) * H;
        float * own = g_e + H;
        float * scales = own + size_t(dr::max_rows) * W;
        int8_t * codes = reinterpret_cast<int8_t *>(f + in_floats);
        const int32_t * hsrc = reinterpret_cast<const int32_t *>(codes + code_bytes);
        a = {};
        a.emb = emb;
        a.emb_stride = dr::n_stream * H;
        a.carry = carry;
        a.carry_prev = carry_prev;
        a.h_nextn = h_nextn;
        a.hsrc = hsrc;
        a.chained = false;
        a.g_e = g_e;
        a.g_h = g_h;
        a.peer = rg.slot(1 - c, xc::msg::draft, s);
        a.own = own;
        b = {};
        b.own = own;
        b.slot = rg.slot(c, xc::msg::draft, s);
        b.codes = codes;
        b.scales = scales;
        b.err = rg.err(c);
        b.card = c;
    };
    for (const int32_t rows : {1, 4}) {
        if (!ok) {
            break;
        }
        cudaGraph_t gr[2] = {};
        cudaGraphExec_t ex[2] = {};
        for (int32_t c = 0; c < 2 && ok; ++c) {
            ok = cuda_ok(cudaSetDevice(rg.device(c)), "set device") &&
                 cuda_ok(cudaStreamBeginCapture(st[c], cudaStreamCaptureModeThreadLocal), "capture");
            for (int32_t i = 0; i < n_xchg && ok; ++i) {
                dr::send_args a;
                dr::recv_args b;
                args(c, i % 3, a, b);
                ok = dr::draft_in_send(rows, a, st[c]) && dr::draft_in_recv(rows, b, st[c]);
            }
            ok = cuda_ok(cudaStreamEndCapture(st[c], &gr[c]), "end capture") && ok;
            ok = ok && cuda_ok(cudaGraphInstantiate(&ex[c], gr[c], 0), "instantiate");
        }
        double mhz[2] = {};
        for (int32_t c = 0; c < 2 && ok; ++c) {
            ok = cuda_ok(cudaSetDevice(rg.device(c)), "set device") && sm_mhz(probe[c]->as<uint64_t>(), st[c], mhz[c]);
        }
        std::vector<double> span[2];
        for (int32_t i = 0; i < n_warm + n_timed && ok; ++i) {
            for (int32_t c = 0; c < 2 && ok; ++c) {
                ok = cuda_ok(cudaSetDevice(rg.device(c)), "set device") &&
                     cuda_ok(cudaEventRecord(ev[c][0], st[c]), "record") &&
                     cuda_ok(cudaGraphLaunch(ex[c], st[c]), "replay") &&
                     cuda_ok(cudaEventRecord(ev[c][1], st[c]), "record");
            }
            for (int32_t c = 0; c < 2 && ok; ++c) {
                float ms = 0.0f;
                ok = cuda_ok(cudaSetDevice(rg.device(c)), "set device") &&
                     cuda_ok(cudaEventSynchronize(ev[c][1]), "sync") &&
                     cuda_ok(cudaEventElapsedTime(&ms, ev[c][0], ev[c][1]), "elapsed");
                if (i >= n_warm) {
                    span[c].push_back(1000.0 * double(ms) / n_xchg);
                }
            }
        }
        uint32_t err[2] = {};
        for (int32_t c = 0; c < 2; ++c) {
            ok = ok && rg.read_err(c, err[c]);
            cudaSetDevice(rg.device(c));
            destroy(gr[c], ex[c]);
        }
        if (!ok) {
            break;
        }
        const timing t0 = stats(span[0]), t1 = stats(span[1]);
        printf("{\"ticket\": \"DR1\", \"point\": \"X-DRAFT-IN send + recv rows %d\", \"devices\": [%d, %d], "
               "\"forecast_us\": [6.0, 9.0], \"forecast_note\": \"Stage A section 5's draft-input line, embed "
               "included\", \"card0_us\": %.3f, \"card0_p90_us\": %.3f, \"card1_us\": %.3f, \"card1_p90_us\": %.3f, "
               "\"exchanges_a_replay\": %d, \"timed\": %d, \"error_words\": [%u, %u], \"sm_mhz_before\": [%.0f, %.0f], "
               "\"cache\": \"back to back (inputs warm)\"}\n",
               rows, dev0, dev1, t0.median, t0.p90, t1.median, t1.p90, n_xchg, n_timed, err[0], err[1], mhz[0], mhz[1]);
        fflush(stdout);
    }
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(rg.device(c));
        if (st[c] != nullptr) {
            cudaStreamSynchronize(st[c]);
            cudaStreamDestroy(st[c]);
        }
        for (cudaEvent_t e : ev[c]) {
            if (e != nullptr) {
                cudaEventDestroy(e);
            }
        }
    }
    return ok;
}

} // namespace

int fnfl_draft_kernel_bench(int dev0, int dev1) {
    cudaDeviceProp prop = {};
    int driver = 0, runtime = 0;
    if (!cuda_ok(cudaSetDevice(dev0), "cudaSetDevice") || !cuda_ok(cudaGetDeviceProperties(&prop, dev0), "props") ||
        !cuda_ok(cudaDriverGetVersion(&driver), "driver") || !cuda_ok(cudaRuntimeGetVersion(&runtime), "runtime")) {
        return 1;
    }
    printf("# draft-kernel-bench: %s, sm_%d%d, %d SMs, driver API %d, runtime %d, devices %d and %d, start %s\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, driver, runtime, dev0, dev1, utc_now().c_str());
    static const char * names[] = {"draft_in send", "draft_in recv", "DA-TC prepare", "DA-TC attend"};
    for (int32_t which = 0; which < 4; ++which) {
        dr::resources rs = {};
        if (dr::kernel_resources(which, rs)) {
            printf("# draft-kernel-bench: %s %d registers, %d B local, %d B shared, %d CTAs an SM\n", names[which],
                   rs.regs, rs.local_bytes, rs.shared_bytes, rs.ctas_per_sm);
        }
    }
    cudaStream_t st = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    datc_io d;
    dev_buf flush(flush_bytes), evict(256), probe(16);
    bool ok =
        d.carve() && flush.p && evict.p && probe.p &&
        cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream") &&
        cuda_ok(cudaEventCreate(&e0), "event") && cuda_ok(cudaEventCreate(&e1), "event") &&
        cuda_ok(cudaMemsetAsync(flush.p, 0, flush_bytes, st), "flush") &&
        cuda_ok(cudaMemsetAsync(d.part.p, 0, dr::part_bytes(), st), "part") &&
        cuda_ok(cudaMemsetAsync(d.rel.p, 0, xc::slot_bytes(xc::msg::hc) + xc::slot_bytes(xc::msg::hmix), st), "slots");
    if (ok) {
        fill_f32<<<64, 256, 0, st>>>(d.qfull, int64_t(dr::max_rows) * (dr::n_head * dr::q_stride + 2 * HD), 1u, -1.5f,
                                     1.5f);
        fill_f32<<<2, 256, 0, st>>>(d.norms, 2 * HD, 3u, 0.5f, 1.5f);
        ok = cuda_ok(cudaPeekAtLastError(), "fill") && busy_wait(flush.p, st);
    }
    const datc_point points[] = {
        {4096, 10.3, 12.7, 17.5},
        {58581, 67.8, 74.7, 90.3},
        {131072, 144.3, 157.3, 187.2},
        {196608, 213.4, 231.9, 274.8},
    };
    for (const datc_point & pt : points) {
        ok = ok && bench_datc(dev0, pt, d, flush.p, evict.as<float>(), probe.as<uint64_t>(), st, e0, e1);
    }
    ok = ok && bench_prepare_catchup(dev0, d, probe.as<uint64_t>(), st, e0, e1);
    if (st != nullptr) {
        cudaStreamSynchronize(st);
        cudaStreamDestroy(st);
    }
    if (e0 != nullptr) {
        cudaEventDestroy(e0);
    }
    if (e1 != nullptr) {
        cudaEventDestroy(e1);
    }
    ok = ok && bench_draft_in(dev0, dev1);
    printf("# draft-kernel-bench: %s, end %s\n", ok ? "done" : "FAILED", utc_now().c_str());
    return ok ? 0 : 1;
}
