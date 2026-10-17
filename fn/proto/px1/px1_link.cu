#include "fn/proto/px1/px1_link.h"

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_prompt.h"
#include "fn/kernels/experts/experts_prompt_test.h"
#include "fn/kernels/experts/experts_test.h"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.cuh"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

// Contract (every kernel here; prototype for PX1, never timed as a result of its own):
//   a3_send_kernel:  shapes: n_words a multiple of 4 in [4, 4 Mi] (16 MiB), src and peer 16-byte aligned; grid
//                    a3_blocks x a3_threads (fn/xchg's X-HC and X-HMIX sender grid, xchg.cu n_block). Each warp sends
//                    runs of 28 words (put_run's largest, so every store is a whole 16-byte group) in a grid-stride
//                    loop.
//                    Moves bits only: the payload is finite normal F32 words, never the sentinel.
//   a3_recv_kernel:  the same grid; each lane polls its 16-byte words in a grid-stride loop with fn/xchg's poll
//                    (volatile, bounded) and resets each to the sentinel after reading it (X-HMIX's acquire and
//                    release). Counts words that differ from the sender's payload.
//   read_kernel:     HX6's read load (HX2's B1 read set, adapted from work/HX8/src/hx8_x.cu read_set at HX8's Result):
//                    72 blocks x 384 threads, 20 16-byte __ldg loads a thread a set, sets rotated over a 256 MiB pool.
//   gate_kernel:     one thread; reports arrival in mapped host memory, then spins (bounded) on a mapped host flag.
//   Budget: no shared memory; registers as ptxas reports (printed at start). Floor: the link (HX8 A3/A5 one-way
//   ~5.7-6.5 GB/s [M]); this bench measures it. Evidence: PX1. Built without fast math (CMake), no float arithmetic.

namespace {

using namespace fn::experts;

constexpr int32_t a3_blocks = 72;   // xchg.cu n_block: the X-HC and X-HMIX senders' grid
constexpr int32_t a3_threads = 128; // the X-HC and X-HMIX senders' block
constexpr int32_t run_words = 28;   // put_run's largest run
constexpr int32_t rd_threads = 384, rd_loads = 20, rd_blocks = 72;
constexpr int64_t rd_set_16 = int64_t(rd_loads) * rd_blocks * rd_threads; // 16-byte words a set (8,847,360 B)
constexpr int32_t rd_sets_launch = 8;                                     // ~71 MB a launch
constexpr size_t rd_pool_bytes = size_t(256) << 20;
constexpr int64_t max_bytes = int64_t(16) << 20;
constexpr uint32_t err_poll_px1 = 1u << 8;
constexpr uint32_t err_gate = 1u << 9;
constexpr uint32_t gate_budget = 1u << 22; // polls of the gate flag (~100 ns sleep each: ~0.4 s)
constexpr double host_timeout_s = 10.0;
constexpr int32_t n_unit = 16; // EP1 units (routing windows) the gate/up graph cycles through
constexpr size_t x_row_bytes = fn::n_embd * 2; // F16 [R][2560]
constexpr size_t min_cold = size_t(1) << 30;

// stamps, %globaltimer ns, per card: gate release, sender blocks' first and last start and last end, receiver blocks'
// first and last start and last end
enum stamp : int32_t { s_gate, s_send0, s_send1, s_send_end, s_recv0, s_recv1, s_recv_end, n_stamp };

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// payload word w from card c: a finite normal F32 value, so never the sentinel or a NaN
__host__ __device__ __forceinline__ uint32_t payload(int32_t c, uint32_t w) {
    const uint32_t h = (w * 2654435761u) ^ (c != 0 ? 0xa5a5a5a5u : 0x3c3c3c3cu);
    return (h & 0x807fffffu) | ((107u + (h >> 23 & 31u)) << 23);
}

__device__ __forceinline__ void stamp_start(unsigned long long * st, int32_t first) {
    if (threadIdx.x == 0) {
        const unsigned long long t = gtime();
        atomicMin(st + first, t);
        atomicMax(st + first + 1, t);
    }
}

__device__ __forceinline__ void stamp_end(unsigned long long * st, int32_t at) {
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(st + at, gtime());
    }
}

__global__ void fill_kernel(uint32_t * dst, int64_t n, int32_t card) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        dst[i] = payload(card, static_cast<uint32_t>(i));
    }
}

__global__ void stamps_init_kernel(unsigned long long * st) {
    if (threadIdx.x < n_stamp) {
        const bool lo = threadIdx.x == s_send0 || threadIdx.x == s_recv0;
        st[threadIdx.x] = lo ? ~0ull : 0ull;
    }
}

__global__ void __launch_bounds__(32) gate_kernel(volatile uint32_t * arrive, const volatile uint32_t * flag,
                                                  uint32_t want, uint32_t * err, unsigned long long * st) {
    if (threadIdx.x != 0) {
        return;
    }
    *arrive = want;
    __threadfence_system();
    for (uint32_t n = 0; n < gate_budget; ++n) {
        if (*flag == want) {
            st[s_gate] = gtime();
            return;
        }
        __nanosleep(100);
    }
    atomicOr(err, err_gate);
}

__global__ void __launch_bounds__(a3_threads) a3_send_kernel(const uint32_t * __restrict__ src, uint32_t * peer,
                                                             int32_t n_words, unsigned long long * st) {
    stamp_start(st, s_send0);
    const int32_t lane = threadIdx.x % 32;
    const int32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int32_t n_warp = gridDim.x * blockDim.x / 32;
    for (int32_t base = warp * run_words; base < n_words; base += n_warp * run_words) {
        const int32_t len = min(run_words, n_words - base);
        const uint32_t bits = lane < len ? src[base + lane] : 0u;
        fn::xchg::put_run(nullptr, peer + base, 0, len, 0, bits); // peer + base is 16-byte aligned: base % 4 == 0
    }
    stamp_end(st, s_send_end);
}

__global__ void __launch_bounds__(a3_threads) a3_recv_kernel(uint32_t * ring, int32_t n4, int32_t sender,
                                                             uint32_t * err, unsigned long long * mism,
                                                             unsigned long long * st) {
    stamp_start(st, s_recv0);
    uint32_t bad = 0;
    for (int32_t u = blockIdx.x * blockDim.x + threadIdx.x; u < n4; u += gridDim.x * blockDim.x) {
        uint4 * w = reinterpret_cast<uint4 *>(ring) + u; // ring holds F32 payload words handled as bits
        const uint4 * p[1] = {w};
        uint4 v[1];
        fn::xchg::poll<1>(p, v, err, err_poll_px1);
        fn::xchg::reset4(w);
        const uint32_t k = 4u * static_cast<uint32_t>(u);
        bad += (v[0].x != payload(sender, k)) + (v[0].y != payload(sender, k + 1)) +
               (v[0].z != payload(sender, k + 2)) + (v[0].w != payload(sender, k + 3));
    }
    if (bad != 0u) {
        atomicAdd(mism, static_cast<unsigned long long>(bad));
    }
    stamp_end(st, s_recv_end);
}

// A5's destination after a point: words that differ from the sender's payload
__global__ void check_kernel(const uint32_t * dst, int64_t n, int32_t sender, unsigned long long * mism) {
    uint32_t bad = 0;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
        bad += dst[i] != payload(sender, static_cast<uint32_t>(i));
    }
    if (bad != 0u) {
        atomicAdd(mism, static_cast<unsigned long long>(bad));
    }
}

__global__ void __launch_bounds__(rd_threads, 3) read_kernel(const uint4 * __restrict__ pool, int32_t set0,
                                                             int32_t n_sets_pool, uint32_t * sink) {
    const int64_t nth = int64_t(gridDim.x) * rd_threads;
    const int64_t i = int64_t(blockIdx.x) * rd_threads + threadIdx.x;
    uint32_t s = 0;
#pragma unroll 1
    for (int32_t j = 0; j < rd_sets_launch; ++j) {
        const uint4 * p = pool + int64_t((set0 + j) % n_sets_pool) * rd_set_16;
        uint4 w[rd_loads];
#pragma unroll
        for (int32_t c = 0; c < rd_loads; ++c) {
            w[c] = __ldg(p + i + c * nth);
        }
#pragma unroll
        for (int32_t c = 0; c < rd_loads; ++c) {
            s ^= w[c].x ^ w[c].y ^ w[c].z ^ w[c].w;
        }
    }
    if (s == 0x9e3779b9u) { // the pool is 0x11 bytes, so the XOR is 0: never taken, keeps the loads
        sink[0] = s;
    }
}

bool cuda_ok(cudaError_t e, const char * what, int32_t dev = -1) {
    if (e != cudaSuccess) {
        fprintf(stderr, "px1-link: %s (device %d): %s\n", what, dev, cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool launched(const char * what, int32_t dev) {
    return cuda_ok(cudaPeekAtLastError(), what, dev);
}

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
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

// one device allocation on dev
struct dev_buf {
    int32_t dev = -1;
    uint8_t * p = nullptr;
    size_t n = 0;

    dev_buf(int32_t d, size_t bytes) : dev(d), n(bytes) {
        if (!cuda_ok(cudaSetDevice(dev), "cudaSetDevice", dev) || !cuda_ok(cudaMalloc(&p, n), "cudaMalloc", dev)) {
            p = nullptr;
        }
    }

    ~dev_buf() {
        if (p != nullptr) {
            cudaSetDevice(dev);
            cudaFree(p);
        }
    }

    dev_buf(const dev_buf &) = delete;
    dev_buf & operator=(const dev_buf &) = delete;

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); } // cudaMalloc: 256-byte aligned
};

// one card's EP1 gate/up co-runner: rotated synthetic IQ3_XXS weights, n_unit routing windows of 1,536 rows
struct ep_card {
    std::unique_ptr<dev_buf> w, grids, x, ids, glu;
    std::vector<std::unique_ptr<dev_buf>> route_mem; // per unit: offsets, sorted, pos, tasks, counters
    std::vector<ep_route> routes;
    size_t expert_bytes = 0;
    int32_t n_copies = 0;
};

// per-card state
struct card {
    int32_t dev = -1;
    cudaStream_t s_msg = nullptr, s_rcv = nullptr, s_cor = nullptr;
    cudaEvent_t cor0 = nullptr, cor1 = nullptr, start = nullptr, send_end = nullptr, recv_end = nullptr;
    std::unique_ptr<dev_buf> src, ring, a5_dst, pool, misc; // misc: err word, mismatch counter, stamps, sink
    ep_card ep;
    uint32_t * err() const { return misc->as<uint32_t>(); }
    unsigned long long * mism() const { return misc->as<unsigned long long>() + 1; }
    unsigned long long * stamps() const { return misc->as<unsigned long long>() + 2; }
    uint32_t * sink() const { return misc->as<uint32_t>() + 2 * (2 + n_stamp); }
};

constexpr size_t misc_bytes = 256;

struct graph_exec {
    cudaGraphExec_t g = nullptr;
    int32_t dev = -1;

    graph_exec() = default;

    ~graph_exec() {
        if (g != nullptr) {
            cudaSetDevice(dev);
            cudaGraphExecDestroy(g);
        }
    }

    graph_exec(const graph_exec &) = delete;
    graph_exec & operator=(const graph_exec &) = delete;
};

enum class cond : int32_t { idle, stream, ep };
enum class proto : int32_t { a3, a5 };

const char * cond_name(cond c) {
    return c == cond::idle ? "idle" : c == cond::stream ? "stream" : "ep_gate_up";
}

struct bench {
    card c[2];
    uint32_t * host_flags = nullptr; // mapped: [0] the gate flag, [1], [2] each card's arrival
    uint32_t seq = 0;
    int32_t prio_hi = 0, prio_lo = 0;
    bool ok = true;

    ~bench() {
        for (card & k : c) {
            if (k.dev < 0) {
                continue;
            }
            cudaSetDevice(k.dev);
            cudaDeviceSynchronize();
            for (cudaStream_t s : {k.s_msg, k.s_rcv, k.s_cor}) {
                if (s != nullptr) {
                    cudaStreamDestroy(s);
                }
            }
            for (cudaEvent_t e : {k.cor0, k.cor1, k.start, k.send_end, k.recv_end}) {
                if (e != nullptr) {
                    cudaEventDestroy(e);
                }
            }
            k.src.reset();
            k.ring.reset();
            k.a5_dst.reset();
            k.pool.reset();
            k.misc.reset();
            k.ep = ep_card();
        }
        if (host_flags != nullptr) {
            cudaFreeHost(host_flags);
        }
    }
};

bool setup_card(bench & b, int32_t i, int32_t dev) {
    card & k = b.c[i];
    k.dev = dev;
    if (!cuda_ok(cudaSetDevice(dev), "cudaSetDevice", dev)) {
        return false;
    }
    bool ok = cuda_ok(cudaStreamCreateWithPriority(&k.s_msg, cudaStreamNonBlocking, b.prio_hi), "stream", dev) &&
              cuda_ok(cudaStreamCreateWithPriority(&k.s_rcv, cudaStreamNonBlocking, b.prio_hi), "stream", dev) &&
              cuda_ok(cudaStreamCreateWithPriority(&k.s_cor, cudaStreamNonBlocking, b.prio_lo), "stream", dev);
    for (cudaEvent_t * e : {&k.cor0, &k.cor1, &k.start, &k.send_end, &k.recv_end}) {
        ok = ok && cuda_ok(cudaEventCreate(e), "event", dev);
    }
    k.src = std::make_unique<dev_buf>(dev, size_t(max_bytes));
    k.ring = std::make_unique<dev_buf>(dev, size_t(max_bytes));
    k.a5_dst = std::make_unique<dev_buf>(dev, size_t(max_bytes));
    k.misc = std::make_unique<dev_buf>(dev, misc_bytes);
    ok = ok && k.src->p != nullptr && k.ring->p != nullptr && k.a5_dst->p != nullptr && k.misc->p != nullptr;
    if (!ok) {
        return false;
    }
    fill_kernel<<<288, 256, 0, k.s_msg>>>(k.src->as<uint32_t>(), max_bytes / 4, i);
    return launched("fill", dev) && cuda_ok(cudaMemsetAsync(k.ring->p, 0xff, k.ring->n, k.s_msg), "ring", dev) &&
           cuda_ok(cudaMemsetAsync(k.a5_dst->p, 0, k.a5_dst->n, k.s_msg), "a5 dst", dev) &&
           cuda_ok(cudaMemsetAsync(k.misc->p, 0, misc_bytes, k.s_msg), "misc", dev) &&
           cuda_ok(cudaStreamSynchronize(k.s_msg), "setup sync", dev);
}

bool setup_stream_load(card & k) {
    cudaSetDevice(k.dev);
    k.pool = std::make_unique<dev_buf>(k.dev, rd_pool_bytes);
    return k.pool->p != nullptr && cuda_ok(cudaMemset(k.pool->p, 0x11, k.pool->n), "pool", k.dev);
}

// as experts_prompt_bench.cpp's load_weights, make_units and route_set, for card i's half of every expert
bool setup_ep(card & k, int32_t i, const synth & s, const ep_fn4 & fr) {
    constexpr int32_t R = ep_max_rows, NU = fn::n_expert_used, NE = fn::n_expert;
    ep_card & e = k.ep;
    cudaSetDevice(k.dev);
    std::vector<int32_t> map(NE);
    for (int32_t x = 0; x < NE; ++x) {
        map[size_t(x)] = x < n_synth ? x : n_synth;
    }
    std::vector<uint8_t> h;
    if (!pack_card(s, i, map, 1, h)) {
        return false;
    }
    e.expert_bytes = static_cast<size_t>(gate_up_bytes(fmt::iq3_xxs));
    std::vector<uint8_t> copy(size_t(NE) * e.expert_bytes);
    for (int32_t x = 0; x < NE; ++x) {
        std::memcpy(copy.data() + size_t(x) * e.expert_bytes, h.data() + size_t(x % n_synth) * e.expert_bytes,
                    e.expert_bytes);
    }
    e.n_copies = static_cast<int32_t>((min_cold + copy.size() - 1) / copy.size());
    e.w = std::make_unique<dev_buf>(k.dev, size_t(e.n_copies) * copy.size());
    e.grids = std::make_unique<dev_buf>(k.dev, grid_bytes);
    e.x = std::make_unique<dev_buf>(k.dev, size_t(n_unit) * R * x_row_bytes);
    e.ids = std::make_unique<dev_buf>(k.dev, size_t(n_unit) * R * NU * 4);
    e.glu = std::make_unique<dev_buf>(k.dev, size_t(ep_max_pairs) * n_ff_card * 2);
    if (e.w->p == nullptr || e.grids->p == nullptr || e.x->p == nullptr || e.ids->p == nullptr || e.glu->p == nullptr) {
        return false;
    }
    bool ok = true;
    for (int32_t c = 0; c < e.n_copies && ok; ++c) {
        ok = cuda_ok(cudaMemcpy(e.w->p + size_t(c) * copy.size(), copy.data(), copy.size(), cudaMemcpyHostToDevice),
                     "weights", k.dev);
    }
    std::vector<uint8_t> g;
    ok = ok && grid_buffer(g) && cuda_ok(cudaMemcpy(e.grids->p, g.data(), g.size(), cudaMemcpyHostToDevice), "grids");
    for (int32_t u = 0; u < n_unit && ok; ++u) {
        const int32_t layer = (7 * u) % fn::n_layer, start = (997 * u) % (fr.n_rows - R + 1);
        const std::vector<int32_t> ids = ep_window(fr, layer, start, R);
        std::vector<uint16_t> xq(size_t(R) * fn::n_embd);
        uint32_t r = static_cast<uint32_t>(u) * 2654435761u + 12345u;
        for (uint16_t & v : xq) {
            r = r * 1664525u + 1013904223u;
            v = static_cast<uint16_t>(((r >> 31) << 15) | ((13u + (r >> 20) % 3u) << 10) | ((r >> 8) & 0x3ffu));
        }
        ok = cuda_ok(cudaMemcpy(e.ids->p + size_t(u) * R * NU * 4, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice),
                     "ids") &&
             cuda_ok(cudaMemcpy(e.x->p + size_t(u) * R * x_row_bytes, xq.data(), xq.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
                     "x");
        auto rm = std::make_unique<dev_buf>(k.dev, size_t(NE + 1) * 4 + 2 * size_t(ep_max_pairs) * 4 +
                                                       size_t(ep_max_tasks) * 16 + 32 + 4 * 256);
        ok = ok && rm->p != nullptr;
        if (!ok) {
            break;
        }
        // 256-byte aligned pieces of one allocation
        size_t at = 0;
        const auto take = [&](size_t n) {
            int32_t * p = reinterpret_cast<int32_t *>(rm->p + at); // the allocation holds I32 arrays
            at += (n + 255) / 256 * 256;
            return p;
        };
        ep_route rt;
        rt.offsets = take(size_t(NE + 1) * 4);
        rt.sorted = take(size_t(ep_max_pairs) * 4);
        rt.pos = take(size_t(ep_max_pairs) * 4);
        rt.tasks = take(size_t(ep_max_tasks) * 16);
        rt.counters = take(32);
        ok = ep_group(R, ep_task_rows(fmt::iq3_xxs), e.ids->as<int32_t>() + size_t(u) * R * NU, rt, nullptr, k.s_cor);
        e.routes.push_back(rt);
        e.route_mem.push_back(std::move(rm));
    }
    return ok && cuda_ok(cudaStreamSynchronize(k.s_cor), "ep setup", k.dev);
}

bool enqueue_ep(card & k, int32_t unit) {
    ep_card & e = k.ep;
    const ep_gate_up_args a = {e.w->p + size_t(unit % e.n_copies) * fn::n_expert * e.expert_bytes, e.grids->p,
                               e.x->as<uint16_t>() + size_t(unit) * ep_max_rows * fn::n_embd, e.routes[size_t(unit)],
                               e.glu->as<uint16_t>(), nullptr};
    return ep_gate_up(fmt::iq3_xxs, ep_max_rows, a, k.s_cor);
}

bool enqueue_read(card & k, int32_t launch) {
    const int32_t n_sets_pool = static_cast<int32_t>(rd_pool_bytes / (size_t(rd_set_16) * 16));
    read_kernel<<<rd_blocks, rd_threads, 0, k.s_cor>>>(k.pool->as<uint4>(), (launch * rd_sets_launch) % n_sets_pool,
                                                       n_sets_pool, k.sink());
    return launched("read", k.dev);
}

// the co-runner: n launches of the condition's kernel, captured once
bool make_graph(card & k, cond cd, int32_t n, graph_exec & ge) {
    cudaSetDevice(k.dev);
    cudaGraph_t g = nullptr;
    if (!cuda_ok(cudaStreamBeginCapture(k.s_cor, cudaStreamCaptureModeThreadLocal), "capture", k.dev)) {
        return false;
    }
    bool ok = true;
    for (int32_t i = 0; i < n && ok; ++i) {
        ok = cd == cond::stream ? enqueue_read(k, i) : enqueue_ep(k, i % n_unit);
    }
    ok = cuda_ok(cudaStreamEndCapture(k.s_cor, &g), "end capture", k.dev) && ok;
    if (ok) {
        ge.dev = k.dev;
        ok = cuda_ok(cudaGraphInstantiate(&ge.g, g, 0), "instantiate", k.dev);
    }
    if (g != nullptr) {
        cudaGraphDestroy(g);
    }
    return ok;
}

struct iter_out {
    float msg_in_us[2] = {0, 0};  // A3: into card r (r's start to r's receiver end); A5: out of card r (copy event)
    float send_us[2] = {0, 0};    // A3: card r's sender kernel (start to sender end)
    float span_us[2] = {0, 0};    // A3: card r's window, start to max(sender end, receiver end); A5: = msg
    float cor_us[2] = {0, 0};     // co-runner graph span
    float overlap_us[2] = {0, 0}; // message window inside the co-runner span
    float lead_us[2] = {0, 0};    // co-runner start to message start
    double wait_first_us[2] = {0, 0}, wait_last_us[2] = {0, 0}; // gate release to first / last sender block start
    double recv_wait_last_us[2] = {0, 0};
};

bool wait_event(cudaEvent_t e, int32_t dev, double deadline, bool & timeout) {
    cudaSetDevice(dev);
    for (;;) {
        const cudaError_t r = cudaEventQuery(e);
        if (r == cudaSuccess) {
            return true;
        }
        if (r != cudaErrorNotReady) {
            return cuda_ok(r, "event query", dev);
        }
        if (now_s() > deadline) {
            timeout = true;
            fprintf(stderr, "px1-link: host timeout waiting on device %d\n", dev);
            return false;
        }
    }
}

float elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    return 1000.0f * ms;
}

// one iteration: the co-runner graphs (if any) on both cards, then the message (if any) both ways, released together
// by the gate once both cards' gates have arrived and (with a co-runner) both co-runners have started plus 100 us
bool run_iter(bench & b, proto pr, int64_t bytes, const graph_exec * cor, bool msg, iter_out & o, bool & timeout) {
    const uint32_t want = ++b.seq;
    volatile uint32_t * hf = b.host_flags; // mapped host memory shared with both cards
    for (int32_t i = 0; i < 2; ++i) {
        card & k = b.c[i];
        card & p = b.c[1 - i];
        cudaSetDevice(k.dev);
        if (cor != nullptr) {
            if (!cuda_ok(cudaEventRecord(k.cor0, k.s_cor), "record", k.dev) ||
                !cuda_ok(cudaGraphLaunch(cor[i].g, k.s_cor), "graph launch", k.dev) ||
                !cuda_ok(cudaEventRecord(k.cor1, k.s_cor), "record", k.dev)) {
                return false;
            }
        }
        if (!msg) {
            continue;
        }
        stamps_init_kernel<<<1, 32, 0, k.s_msg>>>(k.stamps());
        gate_kernel<<<1, 32, 0, k.s_msg>>>(hf + 1 + i, hf, want, k.err(), k.stamps());
        if (!launched("gate", k.dev) || !cuda_ok(cudaEventRecord(k.start, k.s_msg), "record", k.dev)) {
            return false;
        }
        if (pr == proto::a3) {
            a3_send_kernel<<<a3_blocks, a3_threads, 0, k.s_msg>>>(k.src->as<uint32_t>(), p.ring->as<uint32_t>(),
                                                                  static_cast<int32_t>(bytes / 4), k.stamps());
            if (!launched("a3 send", k.dev) || !cuda_ok(cudaEventRecord(k.send_end, k.s_msg), "record", k.dev) ||
                !cuda_ok(cudaStreamWaitEvent(k.s_rcv, k.start, 0), "wait", k.dev)) {
                return false;
            }
            a3_recv_kernel<<<a3_blocks, a3_threads, 0, k.s_rcv>>>(
                k.ring->as<uint32_t>(), static_cast<int32_t>(bytes / 16), 1 - i, k.err(), k.mism(), k.stamps());
            if (!launched("a3 recv", k.dev) || !cuda_ok(cudaEventRecord(k.recv_end, k.s_rcv), "record", k.dev)) {
                return false;
            }
        } else {
            if (!cuda_ok(cudaMemcpyPeerAsync(p.a5_dst->p, p.dev, k.src->p, k.dev, size_t(bytes), k.s_msg), "peer copy",
                         k.dev) ||
                !cuda_ok(cudaEventRecord(k.send_end, k.s_msg), "record", k.dev)) {
                return false;
            }
        }
    }
    const double deadline = now_s() + host_timeout_s;
    if (msg) {
        while (hf[1] != want || hf[2] != want) {
            if (now_s() > deadline) {
                timeout = true;
                fprintf(stderr, "px1-link: host timeout waiting for the gates\n");
                return false;
            }
        }
        if (cor != nullptr) {
            for (int32_t i = 0; i < 2; ++i) {
                if (!wait_event(b.c[i].cor0, b.c[i].dev, deadline, timeout)) {
                    return false;
                }
            }
            const double t = now_s() + 100e-6;
            while (now_s() < t) {
            }
        }
        hf[0] = want;
    }
    for (int32_t i = 0; i < 2; ++i) {
        card & k = b.c[i];
        if (msg && !wait_event(pr == proto::a3 ? k.recv_end : k.send_end, k.dev, deadline, timeout)) {
            return false;
        }
        if (msg && !wait_event(k.send_end, k.dev, deadline, timeout)) {
            return false;
        }
        if (cor != nullptr && !wait_event(k.cor1, k.dev, deadline, timeout)) {
            return false;
        }
    }
    for (int32_t i = 0; i < 2; ++i) {
        card & k = b.c[i];
        cudaSetDevice(k.dev);
        if (cor != nullptr) {
            o.cor_us[i] = elapsed(k.cor0, k.cor1);
        }
        if (!msg) {
            continue;
        }
        const float send = elapsed(k.start, k.send_end);
        const float in = pr == proto::a3 ? elapsed(k.start, k.recv_end) : send;
        o.send_us[i] = send;
        o.span_us[i] = std::max(send, in);
        o.msg_in_us[i] = pr == proto::a3 ? in : send;
        if (cor != nullptr) {
            const float m0 = elapsed(k.cor0, k.start), m1 = m0 + o.span_us[i];
            o.lead_us[i] = m0;
            o.overlap_us[i] = std::max(0.0f, std::min(m1, o.cor_us[i]) - std::max(m0, 0.0f));
        }
        unsigned long long st[n_stamp];
        if (!cuda_ok(cudaMemcpy(st, k.stamps(), sizeof(st), cudaMemcpyDeviceToHost), "stamps", k.dev)) {
            return false;
        }
        if (pr == proto::a3) {
            o.wait_first_us[i] = 1e-3 * double(int64_t(st[s_send0] - st[s_gate]));
            o.wait_last_us[i] = 1e-3 * double(int64_t(st[s_send1] - st[s_gate]));
            o.recv_wait_last_us[i] = 1e-3 * double(int64_t(st[s_recv1] - st[s_gate]));
        }
    }
    return true;
}

struct stats {
    double med = 0, p90 = 0, min = 0, max = 0;
};

stats stat_of(std::vector<double> v) {
    stats s;
    if (v.empty()) {
        return s;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    s.med = n % 2 != 0 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.p90 = v[(9 * n + 9) / 10 - 1];
    s.min = v.front();
    s.max = v.back();
    return s;
}

void put_stats(const char * name, const std::vector<double> & v, bool comma = true) {
    const stats s = stat_of(v);
    printf("\"%s\": {\"med\": %.3f, \"p90\": %.3f, \"min\": %.3f, \"max\": %.3f}%s", name, s.med, s.p90, s.min, s.max,
           comma ? ", " : "");
}

bool read_err(bench & b, uint32_t (&err)[2], unsigned long long (&mism)[2]) {
    for (int32_t i = 0; i < 2; ++i) {
        cudaSetDevice(b.c[i].dev);
        if (!cuda_ok(cudaMemcpy(&err[i], b.c[i].err(), 4, cudaMemcpyDeviceToHost), "err", b.c[i].dev) ||
            !cuda_ok(cudaMemcpy(&mism[i], b.c[i].mism(), 8, cudaMemcpyDeviceToHost), "mism", b.c[i].dev)) {
            return false;
        }
    }
    return true;
}

// A5: the destination of each card's copy holds the sender's payload
bool check_a5(bench & b, int64_t bytes) {
    for (int32_t i = 0; i < 2; ++i) {
        card & k = b.c[i];
        cudaSetDevice(k.dev);
        check_kernel<<<288, 256, 0, k.s_msg>>>(k.a5_dst->as<uint32_t>(), bytes / 4, 1 - i, k.mism());
        if (!launched("check", k.dev) || !cuda_ok(cudaStreamSynchronize(k.s_msg), "check", k.dev)) {
            return false;
        }
    }
    return true;
}

// co-runner launches for a size: at least 100 us lead plus 4x the message at 4 GB/s plus 200 us, rounded up
int32_t cor_launches(cond cd, int64_t bytes, double one_us) {
    const double want = 100.0 + 4.0 * double(bytes) / 4000.0 + 200.0;
    return static_cast<int32_t>(want / one_us) + 1 + (cd == cond::ep ? 1 : 0);
}

struct point_cfg {
    proto pr;
    cond cd;
    int64_t bytes;
    int32_t n_warm, n_timed;
};

// one point: idle, n_timed message iterations; with a co-runner, alternating co-runner alone and co-runner beside the
// message (pairs). One JSON line.
bool run_point(bench & b, const point_cfg & pc, const graph_exec * cor, int32_t n_cor, bool & timeout) {
    std::vector<iter_out> alone, beside;
    const bool loaded = pc.cd != cond::idle;
    const std::string t0 = utc_now();
    for (int32_t i = 0; i < pc.n_warm + pc.n_timed; ++i) {
        iter_out a, m;
        if (loaded && !run_iter(b, pc.pr, pc.bytes, cor, false, a, timeout)) {
            return false;
        }
        if (!run_iter(b, pc.pr, pc.bytes, loaded ? cor : nullptr, true, m, timeout)) {
            return false;
        }
        if (i >= pc.n_warm) {
            alone.push_back(a);
            beside.push_back(m);
        }
    }
    const std::string t1 = utc_now();
    if (pc.pr == proto::a5 && !check_a5(b, pc.bytes)) {
        return false;
    }
    uint32_t err[2];
    unsigned long long mism[2];
    if (!read_err(b, err, mism)) {
        return false;
    }
    const double mib = double(pc.bytes) / double(1 << 20);
    printf("{\"ticket\": \"PX1\", \"kind\": \"point\", \"proto\": \"%s\", \"cond\": \"%s\", \"bytes\": %lld, "
           "\"mib\": %.0f, \"n\": %d, \"warm\": %d, \"cor_launches\": %d, ",
           pc.pr == proto::a3 ? "A3" : "A5", cond_name(pc.cd), static_cast<long long>(pc.bytes), mib, pc.n_timed,
           pc.n_warm, n_cor);
    // each way: 0->1 and 1->0. A3: the receiver's card; A5: the sender's card
    for (int32_t dir = 0; dir < 2; ++dir) {
        const int32_t at = pc.pr == proto::a3 ? 1 - dir : dir;
        std::vector<double> us, gbs, send, span;
        for (const iter_out & m : beside) {
            us.push_back(m.msg_in_us[at]);
            gbs.push_back(double(pc.bytes) / (1000.0 * m.msg_in_us[at]));
            send.push_back(m.send_us[dir]);
            span.push_back(m.span_us[dir]);
        }
        printf("\"dir%d%d\": {", dir, 1 - dir);
        put_stats("us", us);
        put_stats("gbs", gbs);
        put_stats("sender_us", send);
        put_stats("card_window_us", span, false);
        printf("}, ");
    }
    for (int32_t i = 0; i < 2; ++i) {
        std::vector<double> wf, wl, rw;
        for (const iter_out & m : beside) {
            wf.push_back(m.wait_first_us[i]);
            wl.push_back(m.wait_last_us[i]);
            rw.push_back(m.recv_wait_last_us[i]);
        }
        printf("\"card%d_blocks\": {", i);
        put_stats("gate_to_first_sender_us", wf);
        put_stats("gate_to_last_sender_us", wl);
        put_stats("gate_to_last_receiver_us", rw, false);
        printf("}, ");
        if (!loaded) {
            continue;
        }
        std::vector<double> ca, cb, d, ov, lead, per_us, rate;
        for (size_t j = 0; j < beside.size(); ++j) {
            ca.push_back(alone[j].cor_us[i]);
            cb.push_back(beside[j].cor_us[i]);
            const double dd = double(beside[j].cor_us[i]) - double(alone[j].cor_us[i]);
            const double o = beside[j].overlap_us[i];
            d.push_back(dd);
            ov.push_back(o);
            lead.push_back(beside[j].lead_us[i]);
            per_us.push_back(o > 0 ? dd / o : 0.0);
            rate.push_back(o > 0 ? 1.0 - dd / o : 0.0); // the co-runner lost dd of work time inside o
        }
        printf("\"card%d_cor\": {", i);
        put_stats("alone_us", ca);
        put_stats("beside_us", cb);
        put_stats("delta_us", d);
        put_stats("overlap_us", ov);
        put_stats("lead_us", lead);
        put_stats("delta_per_overlap_us", per_us);
        put_stats("rate_during_overlap", rate, false);
        printf("}, ");
    }
    printf("\"err\": [%u, %u], \"mism\": [%llu, %llu], \"t_start\": \"%s\", \"t_end\": \"%s\"}\n", err[0], err[1],
           mism[0], mism[1], t0.c_str(), t1.c_str());
    fflush(stdout);
    if (err[0] != 0u || err[1] != 0u || mism[0] != 0u || mism[1] != 0u) {
        fprintf(stderr, "px1-link: error words %u %u, mismatches %llu %llu: stop\n", err[0], err[1], mism[0], mism[1]);
        return false;
    }
    return true;
}

// the co-runner's single-launch time, for sizing the graphs: one launch alone, after one untimed
bool one_launch_us(card & k, cond cd, double & us) {
    cudaSetDevice(k.dev);
    graph_exec ge;
    if (!make_graph(k, cd, 1, ge)) {
        return false;
    }
    bool ok = true;
    for (int32_t i = 0; i < 4 && ok; ++i) {
        ok = cuda_ok(cudaEventRecord(k.cor0, k.s_cor), "record") && cuda_ok(cudaGraphLaunch(ge.g, k.s_cor), "launch") &&
             cuda_ok(cudaEventRecord(k.cor1, k.s_cor), "record") && cuda_ok(cudaEventSynchronize(k.cor1), "sync");
        us = elapsed(k.cor0, k.cor1);
    }
    return ok;
}

bool busy(bench & b, double seconds) {
    const double t_end = now_s() + seconds;
    int64_t n = 0;
    bool ok = true;
    while (ok && now_s() < t_end) {
        for (int32_t i = 0; i < 2 && ok; ++i) {
            cudaSetDevice(b.c[i].dev);
            for (int32_t j = 0; j < 8 && ok; ++j, ++n) {
                ok = cuda_ok(cudaMemsetAsync(b.c[i].ring->p, 0xff, b.c[i].ring->n, b.c[i].s_cor), "busy");
            }
        }
        for (int32_t i = 0; i < 2 && ok; ++i) {
            cudaSetDevice(b.c[i].dev);
            ok = cuda_ok(cudaStreamSynchronize(b.c[i].s_cor), "busy sync");
        }
    }
    printf("# px1-link: busy %.1f s on both cards (%lld memsets of 16 MiB), timing from %s\n", seconds,
           static_cast<long long>(n), utc_now().c_str());
    return ok;
}

void print_attrs(const char * name, const void * fn, int32_t threads) {
    cudaFuncAttributes a = {};
    int32_t occ = 0;
    cudaFuncGetAttributes(&a, fn);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, 0);
    printf("# px1-link: %s: %d registers, %zu B local, %zu B static shared, %d blocks of %d an SM\n", name, a.numRegs,
           a.localSizeBytes, a.sharedSizeBytes, occ, threads);
}

// Compute Sanitizer: A3 at 1 MiB, card by card in stream order (sends complete before receives start), two rounds
int run_san(bench & b) {
    const int64_t bytes = int64_t(1) << 20;
    bool ok = true;
    for (int32_t round = 0; round < 2 && ok; ++round) {
        for (int32_t i = 0; i < 2 && ok; ++i) {
            card & k = b.c[i];
            cudaSetDevice(k.dev);
            stamps_init_kernel<<<1, 32, 0, k.s_msg>>>(k.stamps());
            a3_send_kernel<<<a3_blocks, a3_threads, 0, k.s_msg>>>(
                k.src->as<uint32_t>(), b.c[1 - i].ring->as<uint32_t>(), static_cast<int32_t>(bytes / 4), k.stamps());
            ok = launched("a3 send", k.dev) && cuda_ok(cudaStreamSynchronize(k.s_msg), "send sync", k.dev);
        }
        for (int32_t i = 0; i < 2 && ok; ++i) {
            card & k = b.c[i];
            cudaSetDevice(k.dev);
            a3_recv_kernel<<<a3_blocks, a3_threads, 0, k.s_msg>>>(
                k.ring->as<uint32_t>(), static_cast<int32_t>(bytes / 16), 1 - i, k.err(), k.mism(), k.stamps());
            ok = launched("a3 recv", k.dev) && cuda_ok(cudaStreamSynchronize(k.s_msg), "recv sync", k.dev);
        }
    }
    uint32_t err[2] = {1, 1};
    unsigned long long mism[2] = {1, 1};
    ok = ok && read_err(b, err, mism);
    printf("{\"ticket\": \"PX1\", \"kind\": \"san\", \"bytes\": %lld, \"rounds\": 2, \"err\": [%u, %u], "
           "\"mism\": [%llu, %llu], \"ok\": %s}\n",
           static_cast<long long>(bytes), err[0], err[1], mism[0], mism[1], ok ? "true" : "false");
    return ok && err[0] == 0u && err[1] == 0u && mism[0] == 0u && mism[1] == 0u ? 0 : 1;
}

} // namespace

int fnfl_px1_link(int32_t dev0, int32_t dev1, const char * mode, const char * cache, const char * routes) {
    const bool all = strcmp(mode, "all") == 0, nsys = strcmp(mode, "nsys") == 0, san = strcmp(mode, "san") == 0;
    if (!all && !nsys && !san) {
        fprintf(stderr, "px1-link: mode is all, nsys or san\n");
        return 2;
    }
    if (dev0 == dev1) {
        fprintf(stderr, "px1-link: DEV0 and DEV1 must differ\n");
        return 2;
    }
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    for (const int32_t d : {dev0, dev1}) {
        cudaDeviceProp prop = {};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, d), "properties", d)) {
            return 1;
        }
        printf("# px1-link: device %d: %s, sm_%d%d, %d SMs, bus %02x:%02x\n", d, prop.name, prop.major, prop.minor,
               prop.multiProcessorCount, prop.pciBusID, prop.pciDeviceID);
    }
    printf("# px1-link: mode %s, driver API %d, runtime %d, start %s\n", mode, driver, runtime, utc_now().c_str());
    if (!fn::xchg::enable_peer_access(dev0, dev1)) {
        return 1;
    }
    int status = 0;
    {
        bench b;
        cudaDeviceGetStreamPriorityRange(&b.prio_lo, &b.prio_hi);
        printf("# px1-link: message streams at priority %d (greatest), co-runner at %d (least)\n", b.prio_hi,
               b.prio_lo);
        if (!cuda_ok(cudaHostAlloc(&b.host_flags, 64, cudaHostAllocMapped | cudaHostAllocPortable), "host flags")) {
            return 1;
        }
        std::memset(b.host_flags, 0, 64);
        if (!setup_card(b, 0, dev0) || !setup_card(b, 1, dev1)) {
            return 1;
        }
        print_attrs("a3_send_kernel", reinterpret_cast<const void *>(a3_send_kernel), a3_threads);
        print_attrs("a3_recv_kernel", reinterpret_cast<const void *>(a3_recv_kernel), a3_threads);
        print_attrs("read_kernel", reinterpret_cast<const void *>(read_kernel), rd_threads);
        print_attrs("gate_kernel", reinterpret_cast<const void *>(gate_kernel), 32);
        if (san) {
            return run_san(b);
        }
        synth s;
        ep_fn4 fr;
        if (!load_synth(fmt::iq3_xxs, cache, s) || !ep_read_fn4(routes, fr)) {
            return 1;
        }
        for (int32_t i = 0; i < 2; ++i) {
            if (!setup_stream_load(b.c[i]) || !setup_ep(b.c[i], i, s, fr)) {
                return 1;
            }
        }
        printf("# px1-link: EP1 co-runner: IQ3_XXS gate/up, R = %d, %d units (FN4 windows), weights %d x 512 x %zu B a "
               "card (cold); read co-runner: %d x %d threads, %d sets of %lld B a launch over %zu B\n",
               ep_max_rows, n_unit, b.c[0].ep.n_copies, b.c[0].ep.expert_bytes, rd_blocks, rd_threads, rd_sets_launch,
               static_cast<long long>(rd_set_16 * 16), rd_pool_bytes);
        if (!busy(b, 2.0)) {
            return 1;
        }
        double one[2][3] = {};
        for (int32_t i = 0; i < 2; ++i) {
            if (!one_launch_us(b.c[i], cond::stream, one[i][1]) || !one_launch_us(b.c[i], cond::ep, one[i][2])) {
                return 1;
            }
        }
        printf("# px1-link: one co-runner launch alone: read %.1f / %.1f us, ep_gate_up %.1f / %.1f us (card 0 / 1)\n",
               one[0][1], one[1][1], one[0][2], one[1][2]);
        const std::vector<int64_t> sizes = nsys ? std::vector<int64_t>{int64_t(8) << 20}
                                                : std::vector<int64_t>{int64_t(1) << 20, int64_t(2) << 20,
                                                                       int64_t(4) << 20, int64_t(8) << 20,
                                                                       int64_t(16) << 20};
        const int32_t n_warm = nsys ? 1 : 3, n_timed = nsys ? 5 : 20;
        bool ok = true, timeout = false;
        for (const cond cd : {cond::idle, cond::stream, cond::ep}) {
            for (const int64_t bytes : sizes) {
                int32_t n_cor = 0;
                graph_exec cor[2];
                if (cd != cond::idle) {
                    const int32_t ci = cd == cond::stream ? 1 : 2;
                    n_cor = cor_launches(cd, bytes, std::max(one[0][ci], one[1][ci]));
                    ok = make_graph(b.c[0], cd, n_cor, cor[0]) && make_graph(b.c[1], cd, n_cor, cor[1]);
                }
                for (const proto pr : {proto::a3, proto::a5}) {
                    if (!ok) {
                        break;
                    }
                    const point_cfg pc = {pr, cd, bytes, n_warm, n_timed};
                    ok = run_point(b, pc, cd == cond::idle ? nullptr : cor, n_cor, timeout);
                }
                if (!ok) {
                    break;
                }
            }
            if (!ok) {
                break;
            }
        }
        status = ok ? 0 : timeout ? 4 : 1;
    }
    printf("# px1-link: %s, end %s\n", status == 0 ? "done" : "FAILED", utc_now().c_str());
    return status;
}
