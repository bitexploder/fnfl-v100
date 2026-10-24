#include "fn/runtime/state_test.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/gdn/gdn_ref.h"
#include "fn/kernels/ple/ple.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/round/version.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/formats.h"
#include "fn/quant/synth.h"
#include "fn/runtime/versions.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

namespace rd = fn::round;
namespace x = fn::xchg;
using fn::runtime::ledger;
using fn::runtime::slot_versions;

constexpr int32_t D = 2560;                 // the HC row (GDN input K; PLE input)
constexpr int32_t GC = fn::gdn::n_channel;  // 5,120
constexpr int32_t GV = fn::gdn::n_value;    // 3,072
constexpr int32_t GH = fn::gdn::n_head;     // 24
constexpr int32_t PF = fn::ple::n_own;      // 1,280
constexpr int32_t PCH = fn::ple::n_channel; // 5,120
constexpr int32_t IK = fn::qsa::head_dim;   // 128
constexpr int32_t cap = 1024;               // the index's cells
constexpr int32_t max_rows = rd::max_rows; // SP2: 5
constexpr int64_t sv = fn::gdn::state_version_floats;
constexpr int64_t hv = fn::gdn::hist_version_floats;
constexpr int64_t pv = fn::ple::hist_version_floats;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "state-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// a value in [-scale, scale) from (seed, i)
float val(uint64_t seed, uint64_t i, float scale) {
    const uint64_t h = mix(seed * 0x100000001b3ull ^ mix(i));
    return scale * (float(h >> 40) / float(1u << 24) * 2.0f - 1.0f);
}

uint16_t bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); // round to nearest even (finite values)
}

uint16_t f16(float f) {
    // small finite values only (|f| < 1, normal): sign, exponent and a truncated mantissa
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int32_t e = static_cast<int32_t>((u >> 23) & 0xff) - 127 + 15;
    if (e <= 0) {
        return static_cast<uint16_t>(s);
    }
    return static_cast<uint16_t>(s | (uint32_t(e) << 10) | ((u >> 13) & 0x3ffu));
}

uint64_t hash_bytes(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, b + i, 8);
        h = mix(h ^ w);
    }
    for (; i < n; ++i) {
        h = mix(h ^ b[i]);
    }
    return mix(h ^ n);
}

// one card's buffers (device dev), owned here and freed at the end
struct card {
    int32_t r = 0, dev = 0;
    cudaStream_t st = nullptr;
    std::vector<void *> owned;
    // the frame
    rd::desc * desc = nullptr;
    rd::state * rst = nullptr;
    const uint32_t ** errs = nullptr;
    // GDN
    uint8_t *wqkv = nullptr, *wz = nullptr;
    uint16_t *walpha = nullptr, *wbeta = nullptr;
    float *conv_w = nullptr, *dt = nullptr, *ga = nullptr, *norm = nullptr;
    float * x_f32 = nullptr;
    int8_t * x_codes = nullptr;
    float * x_scales = nullptr;
    float *conv_out = nullptr, *z = nullptr, *alpha = nullptr, *beta = nullptr, *gated = nullptr;
    int8_t * mid_codes = nullptr;
    float * mid_scales = nullptr;
    void *dense_work = nullptr, *gdn_ws = nullptr;
    float *gdn_state = nullptr, *gdn_rec = nullptr, *gdn_hist = nullptr; // ME11: two banks, two record sets
    fn::dense::product_args gin;
    // PLE
    fn::ple::weights pw;
    float *e = nullptr, *hres = nullptr, *hpost = nullptr, *hx = nullptr, *h_out = nullptr, *ple_hist = nullptr;
    uint8_t * ple_ws = nullptr;
    // the index (card 0)
    uint16_t * keys = nullptr;
    float *k_norm = nullptr, *summ = nullptr, *ik = nullptr;
    // pinned staging of a round's inputs
    float * stage = nullptr;
    cudaGraph_t graph[max_rows + 1] = {};
    cudaGraphExec_t exec[max_rows + 1] = {};

    template <typename T> bool alloc(T *& p, size_t bytes, const char * what) {
        void * q = nullptr;
        if (!cuda_ok(cudaMalloc(&q, bytes), what) || !cuda_ok(cudaMemset(q, 0, bytes), what)) {
            return false;
        }
        owned.push_back(q);
        p = static_cast<T *>(q);
        return true;
    }
    template <typename T> bool put(T * dst, const std::vector<T> & v, const char * what) {
        return cuda_ok(cudaMemcpy(dst, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    }
};

// the staging layout of one round's inputs (floats; R = max_rows, SP2: 5): x [R][2560], codes [R][2560] bytes and
// scales [R][80][2] (as floats' storage), e [R][2560], hres [R][4][1280], hpost [R][4], hx [R][1280], ik [R][128]
constexpr size_t R_ = size_t(max_rows);
constexpr size_t st_x = 0, st_codes = st_x + R_ * D, st_scales = st_codes + R_ * D / 4,
                 st_e = st_scales + R_ * (D / 32) * 2, st_hres = st_e + R_ * D, st_hpost = st_hres + R_ * 4 * PF,
                 st_hx = st_hpost + R_ * 4, st_ik = st_hx + R_ * PF, st_floats = st_ik + R_ * IK;

// what a check compares: each object's hash
struct snap {
    uint64_t h[8] = {}; // per card: GDN state, GDN history, PLE history; card 0: keys, summaries
    int64_t n = 0;
    int32_t n_valid = 0;
};
const char * const snap_names[8] = {"card 0 GDN state",  "card 0 GDN history", "card 0 PLE history",
                                    "card 1 GDN state",  "card 1 GDN history", "card 1 PLE history",
                                    "card 0 index keys", "card 0 summaries"};

// a round's output rows, hashed per position: card r's GDN gated row and PLE h_out row
struct row_hash {
    uint64_t h[2] = {};
};

class rig {
  public:
    rig(int32_t dev0, int32_t dev1, bool eager) : rg_(dev0, dev1), eager_(eager) {
        c_[0].dev = dev0;
        c_[1].dev = dev1;
        c_[0].r = 0;
        c_[1].r = 1;
    }
    ~rig();
    bool init();

    // a forced (kind forced, test_k unused) or forced_k round of B rows; out: 0 committed, 1 recovered, -1 failed.
    // rows: when non-null, each row's output hashes by position
    int32_t round(ledger & l, slot_versions & v, int32_t kind, int32_t rows, int32_t test_k, int32_t test_fail,
                  std::map<int64_t, row_hash> * out_rows, uint32_t reported[2]);
    bool snapshot(const ledger & l, snap & s);
    bool zero_state();
    bool save(std::vector<std::vector<uint8_t>> & blob);
    bool restore(const std::vector<std::vector<uint8_t>> & blob);
    // a clean replay of l's first N tokens at B = 1 from zero state, its snapshot and rows; the device state restored
    bool replay(const ledger & l, slot_versions & v, snap & s, std::map<int64_t, row_hash> & rows);
    bool write_cur(uint32_t cur, uint32_t gdn);
    slot_versions bind_versions();

  private:
    bool enqueue_phase(card & c, int32_t rows, int32_t phase);
    bool upload(const ledger & l, int32_t rows);
    bool rows_out(int32_t rows, int64_t n_before, std::map<int64_t, row_hash> & out);

    card c_[2];
    x::rings rg_;
    bool eager_ = false;
    rd::desc * ring_ = nullptr; // mapped host memory
    uint64_t * flag_ = nullptr;
    rd::done * done_ = nullptr;
    rd::desc * ring_dev_ = nullptr;
    uint64_t * flag_dev_ = nullptr;
    rd::done * done_dev_ = nullptr;
    uint64_t next_round_ = 1;
    uint64_t ple_use_ = 0;

  public:
    uint32_t reported_gdn_[2] = {0, 0}; // ME11: each card's GDN selector in the last round's completion record
};

rig::~rig() {
    for (card & c : c_) {
        cudaSetDevice(c.dev);
        if (c.st != nullptr) {
            cudaStreamSynchronize(c.st);
        }
        for (int32_t b = 0; b <= max_rows; ++b) {
            if (c.exec[b] != nullptr) {
                cudaGraphExecDestroy(c.exec[b]);
            }
            if (c.graph[b] != nullptr) {
                cudaGraphDestroy(c.graph[b]);
            }
        }
        for (void * p : c.owned) {
            cudaFree(p);
        }
        cudaFreeHost(c.stage);
        if (c.st != nullptr) {
            cudaStreamDestroy(c.st);
        }
    }
    cudaFreeHost(ring_);
    cudaFreeHost(flag_);
    cudaFreeHost(done_);
}

slot_versions rig::bind_versions() {
    slot_versions v;
    for (card & c : c_) {
        v.bind(c.r, c.dev, c.st, &c.rst->cur, &c.rst->gdn);
    }
    return v;
}

bool rig::init() {
    if (!rg_.ok()) {
        return false;
    }
    const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
    if (!cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&ring_), sizeof(rd::desc) * rd::ring, fl), "ring") ||
        !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&flag_), 64, fl), "flag") ||
        !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&done_), sizeof(rd::done) * 2, fl), "done") ||
        !cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void **>(&ring_dev_), ring_, 0), "ring") ||
        !cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void **>(&flag_dev_), flag_, 0), "flag") ||
        !cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void **>(&done_dev_), done_, 0), "done")) {
        return false;
    }
    std::memset(ring_, 0, sizeof(rd::desc) * rd::ring);
    std::memset(flag_, 0, 64);
    std::memset(done_, 0, sizeof(rd::done) * 2);
    const int64_t qkv_row = fn::dense::row_bytes(fn::dense::fmt::q6_k, D);
    for (card & c : c_) {
        const uint64_t cs = 0x53523100ull + uint64_t(c.r) * 0x1000; // "SR1", per card
        bool ok = cuda_ok(cudaSetDevice(c.dev), "device") &&
                  cuda_ok(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking), "stream") &&
                  cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&c.stage), st_floats * 4, cudaHostAllocPortable),
                          "staging") &&
                  c.alloc(c.desc, sizeof(rd::desc), "desc") && c.alloc(c.rst, sizeof(rd::state), "round state") &&
                  c.alloc(c.errs, 64, "error list");
        // GDN weights: qkv and z Q6_K (valid random blocks, C1b's generator), alpha and beta BF16 at 0.01, taps, gates
        std::vector<uint8_t> q(size_t(GC) * qkv_row), qp(q.size()), zq(size_t(GV) * qkv_row), zp(zq.size());
        const int64_t bpr = qkv_row / fn::quant::info(fn::quant::wfmt::q6_k).block_bytes;
        ok = ok && fn::quant::synth_blocks(fn::quant::wfmt::q6_k, cs + 1, GC * bpr, q.data()) &&
             fn::quant::synth_blocks(fn::quant::wfmt::q6_k, cs + 2, GV * bpr, zq.data()) &&
             fn::dense::pack(fn::dense::fmt::q6_k, GC, D, q.data(), size_t(qkv_row), qp.data()) &&
             fn::dense::pack(fn::dense::fmt::q6_k, GV, D, zq.data(), size_t(qkv_row), zp.data()) &&
             c.alloc(c.wqkv, qp.size(), "qkv") && c.alloc(c.wz, zp.size(), "z") && c.put(c.wqkv, qp, "qkv") &&
             c.put(c.wz, zp, "z");
        std::vector<uint16_t> ab(size_t(GH) * D), bb(size_t(GH) * D);
        for (size_t i = 0; i < ab.size(); ++i) {
            ab[i] = bf16(val(cs + 3, i, 0.01f));
            bb[i] = bf16(val(cs + 4, i, 0.01f));
        }
        std::vector<float> taps(size_t(GC) * 4), dt(GH), ga(GH), norm(fn::gdn::head_dim);
        for (size_t i = 0; i < taps.size(); ++i) {
            taps[i] = val(cs + 5, i, 0.5f);
        }
        for (int32_t h = 0; h < GH; ++h) {
            dt[size_t(h)] = -3.0f + val(cs + 6, uint64_t(h), 0.5f);  // softplus about 0.05: decays near 0.97
            ga[size_t(h)] = -0.5f + val(cs + 7, uint64_t(h), 0.25f); // -exp(A_log)
        }
        for (size_t i = 0; i < norm.size(); ++i) {
            norm[i] = 1.0f + val(cs + 8, i, 0.2f);
        }
        ok = ok && c.alloc(c.walpha, ab.size() * 2, "alpha w") && c.alloc(c.wbeta, bb.size() * 2, "beta w") &&
             c.put(c.walpha, ab, "alpha w") && c.put(c.wbeta, bb, "beta w") &&
             c.alloc(c.conv_w, taps.size() * 4, "taps") && c.put(c.conv_w, taps, "taps") &&
             c.alloc(c.dt, size_t{64} * 4, "dt") && c.put(c.dt, dt, "dt") && c.alloc(c.ga, size_t{64} * 4, "a") &&
             c.put(c.ga, ga, "a") && c.alloc(c.norm, norm.size() * 4, "norm") && c.put(c.norm, norm, "norm");
        ok = ok && c.alloc(c.x_f32, size_t(max_rows) * D * 4, "x") &&
             c.alloc(c.x_codes, fn::act::codes_bytes(max_rows, D), "x codes") &&
             c.alloc(c.x_scales, fn::act::scales_bytes(max_rows, D), "x scales") &&
             c.alloc(c.conv_out, size_t(max_rows) * GC * 4, "conv out") &&
             c.alloc(c.z, size_t(max_rows) * GV * 4, "z out") && c.alloc(c.alpha, size_t(max_rows) * GH * 4, "alpha") &&
             c.alloc(c.beta, size_t(max_rows) * GH * 4, "beta") &&
             c.alloc(c.gated, size_t(max_rows) * GV * 4, "gated") &&
             c.alloc(c.mid_codes, fn::act::codes_bytes(max_rows, GV), "mid codes") &&
             c.alloc(c.mid_scales, fn::act::scales_bytes(max_rows, GV), "mid scales") &&
             c.alloc(c.gdn_ws, fn::gdn::workspace_bytes(), "gdn ws") &&
             c.alloc(c.gdn_state, size_t(fn::gdn::n_bank) * sv * 4, "gdn banks") &&
             c.alloc(c.gdn_rec, size_t(fn::gdn::records_bytes), "gdn records") &&
             c.alloc(c.gdn_hist, size_t(rd::n_version) * hv * 4, "gdn history");
        if (!ok) {
            return false;
        }
        // DP2's GDN input launch, as the layer's (fn/model/layer.cpp gdn_body), with written activations (no X-HMIX)
        fn::dense::product_args & a = c.gin;
        a.k = D;
        a.n_mats = 4;
        a.mats[0].f = fn::dense::fmt::q6_k;
        a.mats[0].m = GC;
        a.mats[0].w = c.wqkv;
        a.mats[0].out = c.conv_out;
        a.mats[0].e = fn::dense::epi::conv_silu;
        a.mats[1].f = fn::dense::fmt::q6_k;
        a.mats[1].m = GV;
        a.mats[1].w = c.wz;
        a.mats[1].out = c.z;
        a.mats[2].f = fn::dense::fmt::bf16;
        a.mats[2].m = GH;
        a.mats[2].w = reinterpret_cast<const uint8_t *>(c.walpha); // BF16 bits
        a.mats[2].out = c.alpha;
        a.mats[3].f = fn::dense::fmt::bf16;
        a.mats[3].m = GH;
        a.mats[3].w = reinterpret_cast<const uint8_t *>(c.wbeta); // BF16 bits
        a.mats[3].out = c.beta;
        a.x_codes = c.x_codes;
        a.x_scales = c.x_scales;
        a.x_f32 = c.x_f32;
        a.conv.w = c.conv_w;
        a.conv.hist = c.gdn_hist;
        a.conv.cur = &c.rst->cur;
        fn::dense::choose_splits(a);
        const size_t wb = fn::dense::work_bytes(a);
        if (!c.alloc(c.dense_work, wb > 0 ? wb : 16, "dense work")) {
            return false;
        }
        a.work = c.dense_work;
        // PLE weights: key and value BF16 at 0.02, norms near 1, F16 taps
        std::vector<uint16_t> pk(size_t(PCH) * D), pvv(size_t(PF) * D), pc(size_t(PCH) * 4);
        for (size_t i = 0; i < pk.size(); ++i) {
            pk[i] = bf16(val(cs + 9, i, 0.02f));
        }
        for (size_t i = 0; i < pvv.size(); ++i) {
            pvv[i] = bf16(val(cs + 10, i, 0.02f));
        }
        for (size_t i = 0; i < pc.size(); ++i) {
            pc[i] = f16(val(cs + 11, i, 0.5f));
        }
        std::vector<float> nk(PCH), nq(PCH), nc(PCH);
        for (size_t i = 0; i < nk.size(); ++i) {
            nk[i] = 1.0f + val(cs + 12, i, 0.2f);
            nq[i] = 1.0f + val(cs + 13, i, 0.2f);
            nc[i] = 1.0f + val(cs + 14, i, 0.2f);
        }
        uint8_t *kw = nullptr, *vw = nullptr;
        uint16_t * cw = nullptr;
        float *nkd = nullptr, *nqd = nullptr, *ncd = nullptr;
        ok = c.alloc(kw, pk.size() * 2, "ple key") && c.alloc(vw, pvv.size() * 2, "ple value") &&
             c.alloc(cw, pc.size() * 2, "ple taps") && c.alloc(nkd, size_t(PCH) * 4, "nk") &&
             c.alloc(nqd, size_t(PCH) * 4, "nq") && c.alloc(ncd, size_t(PCH) * 4, "nc") &&
             cuda_ok(cudaMemcpy(kw, pk.data(), pk.size() * 2, cudaMemcpyHostToDevice), "ple key") &&
             cuda_ok(cudaMemcpy(vw, pvv.data(), pvv.size() * 2, cudaMemcpyHostToDevice), "ple value") &&
             c.put(cw, pc, "ple taps") && c.put(nkd, nk, "nk") && c.put(nqd, nq, "nq") && c.put(ncd, nc, "nc") &&
             c.alloc(c.e, size_t(max_rows) * D * 4, "e") && c.alloc(c.hres, size_t(max_rows) * 4 * PF * 4, "hres") &&
             c.alloc(c.hpost, size_t(max_rows) * 16, "hpost") && c.alloc(c.hx, size_t(max_rows) * PF * 4, "hx") &&
             c.alloc(c.h_out, size_t(max_rows) * 4 * PF * 4, "h_out") &&
             c.alloc(c.ple_hist, size_t(rd::n_version) * pv * 4, "ple history") &&
             c.alloc(c.ple_ws, fn::ple::workspace_bytes(), "ple ws") && fn::ple::prepare(c.ple_ws);
        c.pw = {kw, vw, nkd, nqd, ncd, cw};
        // the index (card 0): keys, k_norm, summaries, ik
        if (ok && c.r == 0) {
            std::vector<float> kn(IK);
            for (size_t i = 0; i < kn.size(); ++i) {
                kn[i] = 1.0f + val(cs + 15, i, 0.2f);
            }
            ok = c.alloc(c.keys, size_t(cap) * IK * 2, "keys") && c.alloc(c.k_norm, size_t(IK) * 4, "k_norm") &&
                 c.put(c.k_norm, kn, "k_norm") &&
                 c.alloc(c.summ, fn::qsa::summary_bytes(cap / fn::qsa::ratio), "summaries") &&
                 c.alloc(c.ik, size_t(max_rows) * IK * 4, "ik");
        }
        rd::state s0 = {};
        s0.next_round = next_round_;
        s0.cur = 0;
        s0.gdn = 0;
        const uint32_t * err_list[1] = {rg_.err(c.r)};
        ok = ok && cuda_ok(cudaMemcpy(c.rst, &s0, sizeof(s0), cudaMemcpyHostToDevice), "round state") &&
             cuda_ok(cudaMemcpy(c.errs, err_list, sizeof(err_list), cudaMemcpyHostToDevice), "error list") &&
             cuda_ok(cudaDeviceSynchronize(), "init");
        if (!ok) {
            return false;
        }
    }
    if (eager_) {
        return true;
    }
    for (int32_t b = 1; b <= max_rows; ++b) {
        for (card & c : c_) {
            bool ok = cuda_ok(cudaSetDevice(c.dev), "device") &&
                      cuda_ok(cudaStreamBeginCapture(c.st, cudaStreamCaptureModeThreadLocal), "capture");
            for (int32_t ph = 0; ok && ph < 8; ++ph) {
                ok = enqueue_phase(c, b, ph);
            }
            const cudaError_t e = cudaStreamEndCapture(c.st, &c.graph[b]);
            if (!ok || !cuda_ok(e, "end capture") ||
                !cuda_ok(cudaGraphInstantiate(&c.exec[b], c.graph[b], 0), "inst") ||
                !cuda_ok(cudaGraphUpload(c.exec[b], c.st), "upload") || !cuda_ok(cudaStreamSynchronize(c.st), "sync")) {
                return false;
            }
        }
    }
    return true;
}

// phase 0 begin and stamp, 1 the GDN input launch, 2 gdn_core, 3 PLE products, 4 PLE stats (sends X-PLE-STATS),
// 5 PLE update (reads it), 6 the index's round update (card 0), 7 commit and finish
bool rig::enqueue_phase(card & c, int32_t rows, int32_t ph) {
    if (!cuda_ok(cudaSetDevice(c.dev), "device")) {
        return false;
    }
    fn::ple::step_args pa;
    pa.w = c.pw;
    pa.e = c.e;
    pa.hres = c.hres;
    pa.hpost = c.hpost;
    pa.hx = c.hx;
    pa.h_out = c.h_out;
    pa.hist = c.ple_hist;
    pa.cur = &c.rst->cur;
    pa.ws = c.ple_ws;
    pa.stats_peer = rg_.slot(1 - c.r, x::msg::ple, 0);
    pa.stats_slot = rg_.slot(c.r, x::msg::ple, 0);
    pa.use = &c.desc->ple_use;
    pa.err = rg_.err(c.r);
    pa.card = c.r;
    bool ok = true;
    switch (ph) {
        case 0: {
            const rd::begin_args b = {ring_dev_, flag_dev_, c.desc, c.rst, rg_.err(c.r)};
            ok = rd::begin(b, c.st) && rd::stamp(c.rst, c.st);
            break;
        }
        case 1:
            ok = fn::dense::product(rows, c.gin, c.st);
            break;
        case 2: {
            fn::gdn::core_args g;
            g.conv = c.conv_out;
            g.z = c.z;
            g.alpha = c.alpha;
            g.beta = c.beta;
            g.dt = c.dt;
            g.a = c.ga;
            g.norm_w = c.norm;
            g.state = c.gdn_state;
            g.rec = c.gdn_rec;
            g.gsel = &c.rst->gdn;
            g.gated = c.gated;
            g.gated_q = {c.mid_codes, c.mid_scales, rows, GV};
            g.ws = c.gdn_ws;
            g.rows = rows;
            ok = fn::gdn::core(g, c.st);
            break;
        }
        case 3:
            ok = fn::ple::products(rows, pa, c.st);
            break;
        case 4:
            ok = fn::ple::stats(rows, pa, c.st);
            break;
        case 5:
            ok = fn::ple::update(rows, pa, c.st);
            break;
        case 6:
            if (c.r == 0) {
                fn::qsa::round_update_args u;
                u.keys = c.keys;
                u.k_norm = c.k_norm;
                u.summaries = c.summ;
                u.rope = fn::qsa::model_rope();
                u.ik = c.ik;
                u.depth = &c.desc->n_before;
                u.rows = rows;
                u.max_cells = cap;
                ok = fn::qsa::summary_update(u, c.st);
            }
            break;
        default: {
            const rd::commit_args ca = {c.desc, c.rst, c.errs, 1, c.r};
            const rd::finish_args fa = {c.desc, c.rst, c.errs, 1, done_dev_ + c.r};
            ok = rd::commit(ca, c.st) && rd::finish(fa, c.st);
            break;
        }
    }
    if (!ok) {
        fprintf(stderr, "state-test: card %d phase %d not enqueued: %s\n", c.r, ph,
                cudaGetErrorString(cudaGetLastError()));
    }
    return ok;
}

// each row's inputs from its position and token
bool rig::upload(const ledger & l, int32_t rows) {
    std::vector<float> xs(size_t(rows) * D);
    for (int32_t t = 0; t < rows; ++t) {
        const int64_t p = l.n + t;
        const uint64_t seed = mix(uint64_t(p) * 0x10001ull ^ uint64_t(uint32_t(l.tok[size_t(p)])) << 20);
        for (int32_t i = 0; i < D; ++i) {
            xs[size_t(t) * D + size_t(i)] = val(seed, uint64_t(i), 1.0f);
        }
        for (card & c : c_) {
            float * s = c.stage;
            for (int32_t i = 0; i < D; ++i) {
                s[st_x + size_t(t) * D + size_t(i)] = xs[size_t(t) * D + size_t(i)];
                s[st_e + size_t(t) * D + size_t(i)] = val(seed + 1, uint64_t(i), 1.0f);
            }
            for (int32_t st = 0; st < 4; ++st) {
                for (int32_t i = 0; i < PF; ++i) {
                    const uint64_t full = uint64_t(st) * D + uint64_t(c.r) * PF + uint64_t(i);
                    s[st_hres + (size_t(t) * 4 + size_t(st)) * PF + size_t(i)] = val(seed + 2, full, 2.0f);
                }
                s[st_hpost + size_t(t) * 4 + size_t(st)] = val(seed + 3, uint64_t(st), 0.5f);
            }
            for (int32_t i = 0; i < PF; ++i) {
                s[st_hx + size_t(t) * PF + size_t(i)] = val(seed + 4, uint64_t(c.r) * PF + uint64_t(i), 1.0f);
            }
            for (int32_t i = 0; i < IK; ++i) {
                s[st_ik + size_t(t) * IK + size_t(i)] = val(seed + 5, uint64_t(i), 1.0f);
            }
        }
    }
    for (card & c : c_) {
        int8_t * codes = reinterpret_cast<int8_t *>(c.stage + st_codes); // pinned staging bytes
        fn::quant::act_quantize_ref(xs.data(), rows, D, codes, c.stage + st_scales);
        float * s = c.stage;
        const bool ok =
            cuda_ok(cudaSetDevice(c.dev), "device") &&
            cuda_ok(cudaMemcpyAsync(c.x_f32, s + st_x, size_t(rows) * D * 4, cudaMemcpyHostToDevice, c.st), "x") &&
            cuda_ok(cudaMemcpyAsync(c.x_codes, codes, fn::act::codes_bytes(rows, D), cudaMemcpyHostToDevice, c.st),
                    "codes") &&
            cuda_ok(cudaMemcpyAsync(c.x_scales, s + st_scales, fn::act::scales_bytes(rows, D), cudaMemcpyHostToDevice,
                                    c.st),
                    "scales") &&
            cuda_ok(cudaMemcpyAsync(c.e, s + st_e, size_t(rows) * D * 4, cudaMemcpyHostToDevice, c.st), "e") &&
            cuda_ok(cudaMemcpyAsync(c.hres, s + st_hres, size_t(rows) * 4 * PF * 4, cudaMemcpyHostToDevice, c.st),
                    "hres") &&
            cuda_ok(cudaMemcpyAsync(c.hpost, s + st_hpost, size_t(rows) * 16, cudaMemcpyHostToDevice, c.st), "hpost") &&
            cuda_ok(cudaMemcpyAsync(c.hx, s + st_hx, size_t(rows) * PF * 4, cudaMemcpyHostToDevice, c.st), "hx") &&
            (c.r != 0 ||
             cuda_ok(cudaMemcpyAsync(c.ik, s + st_ik, size_t(rows) * IK * 4, cudaMemcpyHostToDevice, c.st), "ik"));
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool rig::rows_out(int32_t rows, int64_t n_before, std::map<int64_t, row_hash> & out) {
    std::vector<float> g(size_t(rows) * GV), h(size_t(rows) * 4 * PF);
    for (card & c : c_) {
        if (!cuda_ok(cudaSetDevice(c.dev), "device") ||
            !cuda_ok(cudaMemcpy(g.data(), c.gated, g.size() * 4, cudaMemcpyDeviceToHost), "gated") ||
            !cuda_ok(cudaMemcpy(h.data(), c.h_out, h.size() * 4, cudaMemcpyDeviceToHost), "h_out")) {
            return false;
        }
        for (int32_t t = 0; t < rows; ++t) {
            uint64_t x = hash_bytes(0x4744u, &g[size_t(t) * GV], size_t(GV) * 4);
            x = hash_bytes(x, &h[size_t(t) * 4 * PF], size_t(4) * PF * 4);
            out[n_before + t].h[c.r] = x;
        }
    }
    return true;
}

int32_t rig::round(ledger & l, slot_versions & v, int32_t kind, int32_t rows, int32_t test_k, int32_t test_fail,
                   std::map<int64_t, row_hash> * out_rows, uint32_t reported[2]) {
    if (rows < 1 || rows > max_rows || rows > l.u || l.n + rows > cap || test_k < 0 || test_k > rows) {
        fprintf(stderr, "state-test: round refused (B %d, U %lld, N %lld)\n", rows, static_cast<long long>(l.u),
                static_cast<long long>(l.n));
        return -1;
    }
    const uint64_t r = next_round_;
    const int64_t n_before = l.n;
    rd::desc d = {};
    d.round = r;
    d.ple_use = ple_use_;
    d.kind = kind;
    d.rows = rows;
    d.n_before = static_cast<int32_t>(n_before);
    d.budget = rows;
    d.test_k = test_k;
    d.test_fail = test_fail;
    for (int32_t p = 0; p < 4; ++p) {
        for (int32_t t = 0; t < rows; ++t) {
            d.pos[p * rows + t] = static_cast<int32_t>(n_before + t);
        }
    }
    for (int32_t t = 0; t < max_rows; ++t) {
        d.tok[t] = t < rows ? l.tok[size_t(n_before + t)] : 0;
        d.out[t] = d.argmax[t] = -1;
    }
    if (!upload(l, rows)) {
        return -1;
    }
    v.begin_step(l, true); // forced rounds (this rig has no head; nothing is retained)
    std::memcpy(&ring_[r % rd::ring], &d, sizeof(d));
    std::atomic_thread_fence(std::memory_order_release);
    if (eager_) {
        reinterpret_cast<volatile uint64_t *>(flag_)[0] = r;
        for (int32_t ph = 0; ph < 8; ++ph) {
            for (card & c : c_) {
                if (!enqueue_phase(c, rows, ph)) {
                    return -1;
                }
            }
        }
    } else {
        for (card & c : c_) {
            if (!cuda_ok(cudaSetDevice(c.dev), "device") || !cuda_ok(cudaGraphLaunch(c.exec[rows], c.st), "launch")) {
                return -1;
            }
        }
        std::atomic_thread_fence(std::memory_order_release);
        reinterpret_cast<volatile uint64_t *>(flag_)[0] = r;
    }
    next_round_ = r + 1;
    ++ple_use_;
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const uint64_t a = reinterpret_cast<volatile uint64_t *>(&done_[0].round)[0];
        const uint64_t b = reinterpret_cast<volatile uint64_t *>(&done_[1].round)[0];
        if (a == r && b == r) {
            std::atomic_thread_fence(std::memory_order_acquire);
            break;
        }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 600.0) {
            fprintf(stderr, "state-test: round %llu: no completion record\n", static_cast<unsigned long long>(r));
            return -1;
        }
    }
    for (card & c : c_) {
        if (!cuda_ok(cudaSetDevice(c.dev), "device") || !cuda_ok(cudaStreamSynchronize(c.st), "round")) {
            return -1;
        }
        reported[c.r] = done_[c.r].cur;
        reported_gdn_[c.r] = done_[c.r].gdn;
    }
    if (out_rows != nullptr && !rows_out(rows, n_before, *out_rows)) {
        return -1;
    }
    bool any = false;
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t i = 0; i < rd::max_err; ++i) {
            any = any || done_[c].err[i] != 0u;
        }
    }
    std::string err;
    if (any) {
        // FC-RD2: both cards back at the pre-step slot, the error words cleared, the ledger unchanged
        for (card & c : c_) {
            if (!cuda_ok(cudaSetDevice(c.dev), "device") || !cuda_ok(cudaMemset(rg_.err(c.r), 0, 4), "clear error")) {
                return -1;
            }
        }
        if (!v.restore(l, err) || !slot_versions::check(l, err)) {
            fprintf(stderr, "state-test: restore: %s\n", err.c_str());
            return -1;
        }
        return 1;
    }
    const int32_t k = rd::commit_versions(kind, rows, 0, test_k);
    if (!v.committed(l, k, reported, reported_gdn_, err)) {
        fprintf(stderr, "state-test: %s\n", err.c_str());
        return -1;
    }
    l.n += k;
    l.u -= k;
    slot_versions::settle(l);
    if (!slot_versions::check(l, err)) {
        fprintf(stderr, "state-test: %s\n", err.c_str());
        return -1;
    }
    return 0;
}

bool rig::write_cur(uint32_t cur, uint32_t gdn) {
    for (card & c : c_) {
        if (!cuda_ok(cudaSetDevice(c.dev), "device") ||
            !cuda_ok(cudaMemcpy(&c.rst->cur, &cur, 4, cudaMemcpyHostToDevice), "cur") ||
            !cuda_ok(cudaMemcpy(&c.rst->gdn, &gdn, 4, cudaMemcpyHostToDevice), "GDN selector")) {
            return false;
        }
    }
    return true;
}

// the logical state: each object's version 0 (slot cur, read back from each card's round state; ME11: the GDN matrix
// its selector names, bank and records replayed, gdn_ref.h), keys of cells < N,
// summaries of blocks < N / 4 (one block's 128 values gathered from the tiled layout)
bool rig::snapshot(const ledger & l, snap & s) {
    s = snap();
    s.n = l.n;
    s.n_valid = l.summaries.n_valid;
    for (card & c : c_) {
        uint32_t cur = 0, gsel = 0;
        std::vector<float> a(static_cast<size_t>(sv)), b(static_cast<size_t>(hv)), p(static_cast<size_t>(pv));
        if (!cuda_ok(cudaSetDevice(c.dev), "device") ||
            !cuda_ok(cudaMemcpy(&cur, &c.rst->cur, 4, cudaMemcpyDeviceToHost), "cur") || cur != l.cur ||
            !cuda_ok(cudaMemcpy(&gsel, &c.rst->gdn, 4, cudaMemcpyDeviceToHost), "GDN selector") || gsel != l.gdn ||
            !fn::gdn::ref::read_committed(c.gdn_state, c.gdn_rec, gsel, a) ||
            !cuda_ok(
                cudaMemcpy(b.data(), c.gdn_hist + rd::version_slot(cur, 0) * hv, b.size() * 4, cudaMemcpyDeviceToHost),
                "history") ||
            !cuda_ok(
                cudaMemcpy(p.data(), c.ple_hist + rd::version_slot(cur, 0) * pv, p.size() * 4, cudaMemcpyDeviceToHost),
                "ple history")) {
            if (cur != l.cur || gsel != l.gdn) {
                fprintf(stderr, "state-test: card %d's cur %u / GDN selector %u are not the ledger's %u / %u\n", c.r,
                        cur, gsel, l.cur, l.gdn);
            }
            return false;
        }
        s.h[size_t{3} * size_t(c.r)] = hash_bytes(1, a.data(), a.size() * 4);
        s.h[size_t{3} * size_t(c.r) + 1] = hash_bytes(2, b.data(), b.size() * 4);
        s.h[size_t{3} * size_t(c.r) + 2] = hash_bytes(3, p.data(), p.size() * 4);
    }
    card & c = c_[0];
    std::vector<uint16_t> k(size_t(l.n) * IK);
    std::vector<float> sm(fn::qsa::summary_bytes(cap / fn::qsa::ratio) / 4);
    if (!cuda_ok(cudaSetDevice(c.dev), "device") ||
        (l.n > 0 && !cuda_ok(cudaMemcpy(k.data(), c.keys, k.size() * 2, cudaMemcpyDeviceToHost), "keys")) ||
        !cuda_ok(cudaMemcpy(sm.data(), c.summ, sm.size() * 4, cudaMemcpyDeviceToHost), "summaries")) {
        return false;
    }
    s.h[6] = hash_bytes(6, k.data(), k.size() * 2);
    uint64_t h = 7;
    const int32_t blocks = fn::qsa::blocks_complete(static_cast<int32_t>(l.n));
    std::vector<float> one(IK);
    for (int32_t bk = 0; bk < blocks; ++bk) {
        for (int32_t e = 0; e < IK; ++e) {
            one[size_t(e)] = sm[size_t(bk / 32) * 4096 + size_t(e / 4) * 128 + size_t(bk % 32) * 4 + size_t(e % 4)];
        }
        h = hash_bytes(h, one.data(), one.size() * 4);
    }
    s.h[7] = h;
    return true;
}

bool rig::zero_state() {
    for (card & c : c_) {
        if (!cuda_ok(cudaSetDevice(c.dev), "device") ||
            !cuda_ok(cudaMemset(c.gdn_state, 0, size_t(fn::gdn::n_bank) * sv * 4), "zero") ||
            !cuda_ok(cudaMemset(c.gdn_rec, 0, size_t(fn::gdn::records_bytes)), "zero") ||
            !cuda_ok(cudaMemset(c.gdn_hist, 0, size_t(rd::n_version) * hv * 4), "zero") ||
            !cuda_ok(cudaMemset(c.ple_hist, 0, size_t(rd::n_version) * pv * 4), "zero") ||
            (c.r == 0 && (!cuda_ok(cudaMemset(c.keys, 0, size_t(cap) * IK * 2), "zero") ||
                          !cuda_ok(cudaMemset(c.summ, 0, fn::qsa::summary_bytes(cap / fn::qsa::ratio)), "zero")))) {
            return false;
        }
    }
    return true;
}

// every slot of every object, cur, the index; in a fixed order
bool rig::save(std::vector<std::vector<uint8_t>> & blob) {
    blob.clear();
    for (card & c : c_) {
        const std::pair<const void *, size_t> parts[] = {
            {c.gdn_state, size_t(fn::gdn::n_bank) * sv * 4},
            {c.gdn_rec, size_t(fn::gdn::records_bytes)},
            {c.gdn_hist, size_t(rd::n_version) * hv * 4},
            {c.ple_hist, size_t(rd::n_version) * pv * 4},
            {&c.rst->cur, 8}, // cur, then the GDN selector (rd::state)
            {c.keys, c.r == 0 ? size_t(cap) * IK * 2 : 0},
            {c.summ, c.r == 0 ? fn::qsa::summary_bytes(cap / fn::qsa::ratio) : 0}};
        for (const auto & [p, n] : parts) {
            blob.emplace_back(n);
            if (n > 0 && (!cuda_ok(cudaSetDevice(c.dev), "device") ||
                          !cuda_ok(cudaMemcpy(blob.back().data(), p, n, cudaMemcpyDeviceToHost), "save"))) {
                return false;
            }
        }
    }
    return true;
}

bool rig::restore(const std::vector<std::vector<uint8_t>> & blob) {
    size_t i = 0;
    for (card & c : c_) {
        void * parts[] = {c.gdn_state, c.gdn_rec, c.gdn_hist, c.ple_hist, &c.rst->cur, c.keys, c.summ};
        for (void * p : parts) {
            const std::vector<uint8_t> & b = blob[i++];
            if (!b.empty() && (!cuda_ok(cudaSetDevice(c.dev), "device") ||
                               !cuda_ok(cudaMemcpy(p, b.data(), b.size(), cudaMemcpyHostToDevice), "restore"))) {
                return false;
            }
        }
    }
    return true;
}

bool rig::replay(const ledger & l, slot_versions & v, snap & s, std::map<int64_t, row_hash> & rows) {
    std::vector<std::vector<uint8_t>> blob;
    slot_versions v_saved = v;
    if (!save(blob) || !zero_state()) {
        return false;
    }
    ledger r;
    std::string err;
    slot_versions rv = bind_versions();
    if (!rv.reset(r, err)) {
        fprintf(stderr, "state-test: %s\n", err.c_str());
        return false;
    }
    r.tok.assign(l.tok.begin(), l.tok.begin() + l.n);
    r.u = l.n;
    r.prompt_len = l.n;
    uint32_t rep[2] = {0, 0};
    while (r.u > 0) {
        if (round(r, rv, static_cast<int32_t>(rd::kind::forced), 1, 0, 0, &rows, rep) != 0) {
            return false;
        }
    }
    if (!snapshot(r, s)) {
        return false;
    }
    v = v_saved;
    return restore(blob);
}

// ---- the script ----

struct tally {
    int64_t checks = 0, mismatches = 0, row_checks = 0, row_mismatches = 0;
    int32_t k_seen[max_rows + 1][max_rows + 1] = {};                // [B][k]
    int32_t j_seen[max_rows + 1][max_rows + 1][max_rows + 1] = {}; // [B][k][j]
    uint32_t slots = 1;           // bit s: the committed slot was s
    uint32_t orient = 1;          // ME11: bit bank + 2 set of the committed GDN selector
    int32_t gdn_steps_seen = 1;   // ME11: bit k: a check ran with k live records
    int32_t errors_ok = 0, errors_bad = 0, healthy_committed = 0;
    bool control_detected = false, control_run = false, summary_case = false;
};

class script {
  public:
    script(rig & g, tally & t, uint64_t seed) : g_(g), t_(t), seed_(seed), v_(g.bind_versions()) {}

    bool start() {
        std::string err;
        return v_.reset(l_, err);
    }

    void append(int32_t n) {
        for (int32_t i = 0; i < n; ++i) {
            l_.tok.push_back(static_cast<int32_t>(mix(seed_ ^ uint64_t(++draws_)) % 248320u));
        }
        l_.u += n;
        l_.prompt_len = static_cast<int64_t>(l_.tok.size()); // forced input, no outputs
    }

    // the state against a clean replay; rows: a round's output rows to compare at their positions
    bool check(const char * what, const std::map<int64_t, row_hash> * rows, bool expect_equal = true) {
        snap a, b;
        std::map<int64_t, row_hash> rr;
        if (!g_.snapshot(l_, a) || !g_.replay(l_, v_, b, rr)) {
            return false;
        }
        bool same = a.n == b.n && a.n_valid == b.n_valid;
        std::string diff;
        for (int32_t i = 0; i < 8; ++i) {
            if (a.h[i] != b.h[i]) {
                same = false;
                diff += std::string(diff.empty() ? "" : ", ") + snap_names[i];
            }
        }
        if (!expect_equal) {
            return !same;
        }
        ++t_.checks;
        if (!same) {
            ++t_.mismatches;
            printf("state-test: MISMATCH after %s at N %lld: %s\n", what, static_cast<long long>(l_.n), diff.c_str());
        }
        if (rows != nullptr) {
            for (const auto & [p, h] : *rows) {
                ++t_.row_checks;
                const auto it = rr.find(p);
                if (it == rr.end() || it->second.h[0] != h.h[0] || it->second.h[1] != h.h[1]) {
                    ++t_.row_mismatches;
                    printf("state-test: ROW MISMATCH after %s at position %lld\n", what, static_cast<long long>(p));
                }
            }
        }
        t_.slots |= 1u << l_.cur;
        t_.orient |= 1u << (rd::gdn_bank(l_.gdn) + 2 * rd::gdn_set(l_.gdn));
        t_.gdn_steps_seen |= 1 << rd::gdn_steps(l_.gdn);
        return true;
    }

    // one case: B fresh tokens, a forced_k round committing k, a cut to j kept rows (the ledger to N_before + j),
    // then a forced round of b2 fresh tokens
    bool run_case(int32_t B, int32_t k, int32_t j, int32_t b2, bool control) {
        uint32_t rep[2];
        append(B);
        const int64_t n0 = l_.n;
        if (g_.round(l_, v_, static_cast<int32_t>(rd::kind::forced_k), B, k, 0, nullptr, rep) != 0 ||
            !check("forced_k", nullptr)) {
            return false;
        }
        t_.k_seen[B][k]++;
        std::string err;
        if (!v_.cut(l_, j, n0 + j, err) || !slot_versions::check(l_, err)) {
            fprintf(stderr, "state-test: cut: %s\n", err.c_str());
            return false;
        }
        if (!check("cut", nullptr)) {
            return false;
        }
        t_.j_seen[B][k][j]++;
        if (control && j < k) {
            // the negative control: cur on the uncut version k must not match the replay of N_before + j
            t_.control_run = true;
            const uint32_t good = l_.cur, good_gdn = l_.gdn;
            const uint32_t bad = static_cast<uint32_t>(rd::version_slot(v_.last().cur_pre, k));
            const uint32_t bad_gdn = rd::gdn_after_round(v_.last().gdn_pre, k);
            if (!g_.write_cur(bad, bad_gdn)) {
                return false;
            }
            ledger saved = l_;
            l_.cur = bad;
            l_.gdn = bad_gdn;
            t_.control_detected = check("control", nullptr, false);
            l_ = saved;
            if (!g_.write_cur(good, good_gdn)) {
                return false;
            }
        }
        return cont(b2);
    }

    bool cont(int32_t b2) {
        uint32_t rep[2];
        std::map<int64_t, row_hash> rows;
        append(b2);
        return g_.round(l_, v_, static_cast<int32_t>(rd::kind::forced), b2, 0, 0, &rows, rep) == 0 &&
               check("continue", &rows);
    }

    // a recoverable error injected before commit on the cards in mask: both cards must stay at cur_pre
    bool error_case(int32_t B, int32_t k, int32_t mask) {
        uint32_t rep[2];
        append(B);
        const uint32_t pre = l_.cur, pre_gdn = l_.gdn;
        const int64_t n0 = l_.n;
        if (g_.round(l_, v_, static_cast<int32_t>(rd::kind::forced_k), B, k, mask, nullptr, rep) != 1) {
            fprintf(stderr, "state-test: the injected error was not seen\n");
            return false;
        }
        uint32_t now[2] = {0, 0}, now_gdn[2] = {0, 0};
        std::string err;
        if (!v_.read_words(now, now_gdn, err)) {
            return false;
        }
        // ME11: the GDN selectors back at the pre-step word too
        const bool back = now[0] == pre && now[1] == pre && l_.cur == pre && l_.n == n0 && now_gdn[0] == pre_gdn &&
                          now_gdn[1] == pre_gdn && l_.gdn == pre_gdn;
        // the card whose words were clear did commit before the host restored it (the hazard FC-RD2 closes), its GDN
        // selector moved with cur
        for (int32_t c = 0; c < 2; ++c) {
            if (((mask >> c) & 1) == 0 && k > 0 && rep[c] == static_cast<uint32_t>(rd::version_slot(pre, k)) &&
                g_.reported_gdn_[c] == rd::gdn_after_round(pre_gdn, k)) {
                t_.healthy_committed++;
            }
        }
        (back ? t_.errors_ok : t_.errors_bad)++;
        printf("state-test: error on card mask %d (B %d, k %d): reported cur %u / %u, GDN selector %u / %u; both cards "
               "back at cur_pre %u and GDN selector %u: %s\n",
               mask, B, k, rep[0], rep[1], g_.reported_gdn_[0], g_.reported_gdn_[1], pre, pre_gdn,
               back ? "yes" : "NO");
        if (!check("error", nullptr)) {
            return false;
        }
        // the unprocessed rows run again as a forced round
        std::map<int64_t, row_hash> rows;
        return g_.round(l_, v_, static_cast<int32_t>(rd::kind::forced), B, 0, 0, &rows, rep) == 0 &&
               check("after error", &rows);
    }

    // FC-ST1's summaries: a block completed by a round, cut inside the block, then completed again by other tokens
    bool summary_case() {
        uint32_t rep[2];
        while (l_.n % 4 != 1) {
            append(1);
            if (g_.round(l_, v_, static_cast<int32_t>(rd::kind::forced), 1, 0, 0, nullptr, rep) != 0) {
                return false;
            }
        }
        append(3);
        const int64_t n0 = l_.n; // 4m + 1: three rows complete block m
        std::string err;
        if (g_.round(l_, v_, static_cast<int32_t>(rd::kind::forced_k), 3, 3, 0, nullptr, rep) != 0 ||
            !check("summary: block completed", nullptr) || !v_.cut(l_, 1, n0 + 1, err) ||
            !check("summary: cut inside the block", nullptr)) {
            fprintf(stderr, "state-test: summary case: %s\n", err.c_str());
            return false;
        }
        printf("state-test: summary case: block %lld pooled at N %lld, cut to N %lld (valid blocks %d), continued\n",
               static_cast<long long>(n0) / 4, static_cast<long long>(n0) + 3, static_cast<long long>(l_.n),
               l_.summaries.n_valid);
        t_.summary_case = true;
        return cont(4);
    }

    const ledger & led() const { return l_; }

  private:
    rig & g_;
    tally & t_;
    uint64_t seed_;
    uint64_t draws_ = 0;
    ledger l_;
    slot_versions v_;
};

} // namespace

int fnfl_state_test_mu1(const std::string & run, int32_t dev0, int32_t dev1, bool kv_q8); // slots_gpu_test.cpp
int fnfl_state_test_mxmu1(const std::string & run, int32_t dev);                          // slots_gpu_test.cpp (MXMU1)

int fnfl_state_test(int argc, char ** argv, bool kv_q8) {
    const std::string mode = argc >= 3 ? argv[2] : "all";
    const std::string run = argc >= 4 ? argv[3] : "graph";
    const int32_t dev0 = argc >= 5 ? atoi(argv[4]) : 0;
    const int32_t dev1 = argc >= 6 ? atoi(argv[5]) : 1;
    const bool san = mode == "san";
    if (mode == "mu1-one") {
        // MXMU1: two slots on one card (the expert cache, the host-backed KV; slots_gpu_test.cpp part C)
        if (run != "graph" && run != "eager" && run != "c0") {
            fprintf(stderr, "state-test: mu1-one [graph|eager|c0] [DEV]\n");
            return 2;
        }
        return fnfl_state_test_mxmu1(run, dev0);
    }
    if (mode == "mu1") {
        // MU1: the multi-slot engine's pool and frames (slots_gpu_test.cpp)
        // peer access is enabled by part B's load (part A's peer copy needs none)
        if ((run != "graph" && run != "eager" && run != "read" && run != "joint" && run != "joint-eager" &&
             run != "joint-san") ||
            dev0 == dev1) {
            fprintf(stderr, "state-test: mu1 [graph|eager|read|joint|joint-eager|joint-san] [DEV0] [DEV1] "
                            "(two distinct cards)\n");
            return 2;
        }
        return fnfl_state_test_mu1(run, dev0, dev1, kv_q8);
    }
    if ((mode != "all" && !san) || (run != "graph" && run != "eager") || dev0 == dev1) {
        fprintf(stderr, "state-test: [all|san] [graph|eager] [DEV0] [DEV1] (two distinct cards)\n");
        return 2;
    }
    if (!x::enable_peer_access(dev0, dev1)) {
        return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    tally t;
    {
        rig g(dev0, dev1, run == "eager");
        if (!g.init()) {
            fprintf(stderr, "state-test: setup failed\n");
            return 1;
        }
        script s(g, t, 0x5352310000ull);
        bool ok = s.start() && s.check("start", nullptr);
        int32_t n_case = 0;
        if (san) {
            ok = ok && s.run_case(3, 2, 1, 2, true) && s.run_case(3, 3, 0, 1, false) && s.run_case(5, 5, 3, 5, false) &&
                 s.error_case(2, 2, 1) && s.error_case(2, 1, 3) && s.error_case(5, 4, 2) && s.summary_case();
        } else {
            for (int32_t B = 1; ok && B <= max_rows; ++B) {
                for (int32_t k = 0; ok && k <= B; ++k) {
                    for (int32_t j = 0; ok && j <= k; ++j) {
                        ok = s.run_case(B, k, j, 1 + n_case % max_rows, n_case == 4); // case 4: B 2, k 1, j 0
                        ++n_case;
                    }
                }
            }
            for (const int32_t mask : {3, 1, 2}) {
                for (const int32_t k : {3, 2}) {
                    ok = ok && s.error_case(3, k, mask);
                }
            }
            ok = ok && s.error_case(4, 4, 1) && s.error_case(5, 5, 3) && s.error_case(5, 2, 2) &&
                 s.error_case(1, 1, 2) && s.summary_case();
        }
        if (!ok) {
            fprintf(stderr, "state-test: the script stopped\n");
            printf("state-test: FAIL\n");
            return 1;
        }
        printf("state-test: final ledger L %zu, N %lld, U %lld, cur %u, checks %lld\n", s.led().tok.size(),
               static_cast<long long>(s.led().n), static_cast<long long>(s.led().u), s.led().cur,
               static_cast<long long>(s.led().checks));
    }
    bool covered = true;
    std::string ks;
    for (int32_t B = 1; B <= max_rows; ++B) {
        for (int32_t k = 0; k <= B; ++k) {
            for (int32_t j = 0; j <= k; ++j) {
                covered = covered && (san || t.j_seen[B][k][j] > 0);
            }
            covered = covered && (san || t.k_seen[B][k] > 0);
        }
    }
    const int32_t slots = __builtin_popcount(t.slots);
    printf("state-test: %lld state checks against a clean replay at B = 1, %lld differ; %lld continuation rows, %lld "
           "differ\n",
           static_cast<long long>(t.checks), static_cast<long long>(t.mismatches), static_cast<long long>(t.row_checks),
           static_cast<long long>(t.row_mismatches));
    printf("state-test: every k <= B and every cut j <= k at B = 1..%d covered: %s; committed slots visited: %d of "
           "%d\n",
           max_rows,
           san       ? "n/a (san)"
           : covered ? "yes"
                     : "NO",
           slots, rd::n_version);
    printf("state-test: GDN selector (ME11): bank and set orientations committed %d (of the 2 a target round's flip "
           "reaches), live record counts seen 0..5: %s\n",
           __builtin_popcount(t.orient), t.gdn_steps_seen == 63 ? "all" : san ? "some (san)" : "NOT ALL");
    printf("state-test: injected errors: %d with both cards back at cur_pre, %d not; a clear card had committed before "
           "the restore in %d of them\n",
           t.errors_ok, t.errors_bad, t.healthy_committed);
    printf("state-test: summary case (cut inside a block, continued past its end): %s; negative control (cur on the "
           "uncut version) detected: %s\n",
           t.summary_case ? "run" : "NOT RUN", t.control_run ? (t.control_detected ? "yes" : "NO") : "n/a");
    printf("state-test: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const bool pass = t.mismatches == 0 && t.row_mismatches == 0 && t.checks > 0 && covered &&
                      (san || slots == rd::n_version) && t.errors_bad == 0 && t.errors_ok > 0 &&
                      (san || (t.gdn_steps_seen == 63 && __builtin_popcount(t.orient) >= 2)) &&
                      t.healthy_committed > 0 && t.summary_case &&
                      (!t.control_run || t.control_detected) && (san || t.control_run);
    printf("state-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
