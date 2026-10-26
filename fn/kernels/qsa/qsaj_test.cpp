#include "fn/kernels/qsa/qsaj_test.h"

#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/state.h"
#include "fn/quant/formats.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int32_t R = fn::qsa::max_rows;              // 5 rows a lane
constexpr int32_t IH = fn::qsa::head_dim;             // 128: the index key and query width
constexpr int32_t IQ = fn::qsa::n_idx_head * IH;      // 512: a row's index queries
constexpr int32_t AH = fn::qsa::attn::head_dim;       // 256
constexpr int32_t NH = fn::qsa::attn::n_head;         // 12
constexpr int32_t QS = fn::qsa::attn::q_stride;       // 512
constexpr int32_t OW = fn::qsa::attn::out_width;      // 3,072
constexpr int32_t SS = fn::model::kv_q8_scale_stride; // KV8: FP16 scales a cell
constexpr int32_t U_MAX = fn::xchg::max_lanes;
constexpr int32_t sel_slot = 0; // the X-SELECT slot every run uses
constexpr uint8_t sentinel_byte = 0x7b;  // FMS13: rows rr .. 4 of each output before a launch at rr rows

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "qsaj-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// one device allocation and its initial bytes (uploaded by fresh)
struct buf {
    int32_t dev = -1;
    void * p = nullptr;
    size_t n = 0;
    std::vector<uint8_t> init;

    ~buf() {
        if (p != nullptr) {
            cudaSetDevice(dev);
            cudaFree(p);
        }
    }
};

// One lane: its index pieces on the owner card (dev0) and its attention pieces on the peer card (dev1), its own ring
// set. Every buffer is in b, so a snapshot holds every byte the lane's launches can write.
struct lane {
    int32_t cells = 0, depth = 0, rows = R;  // FMS13: rows, the round's R (2 .. 5) in [R]-row buffers
    bool    q8 = false;

    // FMS13's sentinel test: byte ranges (buffer, offset, length) a launch at rows must leave at their initial bytes:
    // rows rows .. 4 of each input (NaN) and output (the sentinel), and the cache cells past depth + rows
    struct span {
        size_t buf, at, n;
    };

    std::vector<span>                 kept;
    fn::xchg::rings rg;
    std::vector<std::unique_ptr<buf>> b;
    // owner (dev0)
    uint16_t * keys = nullptr; // BF16 [cells][128]
    float * sums = nullptr;    // tiled F32 summaries of cells / 4 blocks
    float * ik = nullptr;      // [R][128]
    float * iq = nullptr;      // [R][4][128]
    uint8_t * qws = nullptr;   // fn::qsa::workspace_bytes()
    uint32_t * body0 = nullptr;
    int32_t * off0 = nullptr;
    int32_t * depth0 = nullptr;
    // peer (dev1)
    uint32_t * body1 = nullptr;
    int32_t * off1 = nullptr;
    float * qfull = nullptr; // [R][12][512]
    float * kv_k = nullptr;  // [R][256]
    float * kv_v = nullptr;
    int32_t * pos = nullptr; // [4][R]
    uint16_t * kc = nullptr; // BF16 [cells][256], or KV8 codes
    uint16_t * vc = nullptr;
    uint16_t * ks = nullptr; // KV8: FP16 [cells][SS]
    uint16_t * vs = nullptr;
    float * out = nullptr;    // [R][3072]
    int8_t * codes = nullptr; // [R][3072]
    float * scales = nullptr; // [R][96][2]
    uint8_t * aws = nullptr;  // fn::qsa::attn::workspace_bytes()
    int32_t * depth1 = nullptr;

    lane(int32_t d0, int32_t d1) : rg(d0, d1) {}
};

// the layer's weights every lane reads: the index norms on the owner, the attention norms on the peer
struct shared {
    std::vector<std::unique_ptr<buf>> b;
    float * iq_norm = nullptr; // [128]
    float * ik_norm = nullptr;
    float * q_norm = nullptr; // [256]
    float * k_norm = nullptr;
};

struct env {
    int32_t dev0 = 0, dev1 = 1;
    cudaStream_t s0 = nullptr, s1 = nullptr; // s1 == s0 in loopback
    shared w;
    std::vector<std::unique_ptr<lane>> ln;
};

template <typename T>
T * add(std::vector<std::unique_ptr<buf>> & bs, int32_t dev, size_t count, const std::vector<T> & init, bool & ok) {
    auto x = std::make_unique<buf>();
    x->dev = dev;
    x->n = count * sizeof(T);
    x->init.assign(x->n, 0);
    if (!init.empty()) {
        std::memcpy(x->init.data(), init.data(), std::min(x->n, init.size() * sizeof(T)));
    }
    ok = ok && cuda_ok(cudaSetDevice(dev), "cudaSetDevice") && cuda_ok(cudaMalloc(&x->p, x->n), "cudaMalloc");
    T * p = static_cast<T *>(x->p);
    bs.push_back(std::move(x));
    return p;
}

std::vector<float> uniform(std::mt19937 & g, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float & x : v) {
        x = d(g);
    }
    return v;
}

std::vector<uint16_t> bf16s(std::mt19937 & g, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<uint16_t> v(n);
    for (uint16_t & x : v) {
        x = fn::quant::f32_to_bf16(d(g));
    }
    return v;
}

bool make_shared(env & e) {
    std::mt19937 g(7);
    bool ok = true;
    e.w.iq_norm = add(e.w.b, e.dev0, IH, uniform(g, IH, 0.5f, 1.5f), ok);
    e.w.ik_norm = add(e.w.b, e.dev0, IH, uniform(g, IH, 0.5f, 1.5f), ok);
    e.w.q_norm = add(e.w.b, e.dev1, AH, uniform(g, AH, 0.5f, 1.5f), ok);
    e.w.k_norm = add(e.w.b, e.dev1, AH, uniform(g, AH, 0.5f, 1.5f), ok);
    return ok;
}

// rows rr .. R - 1 of a [R][width] input NaN (FMS13: a launch reading them spreads NaN into rows < rr)
std::vector<float> nan_tail(std::vector<float> v, int32_t rr, size_t width) {
    std::fill(v.begin() + ptrdiff_t(size_t(rr) * width), v.end(), std::numeric_limits<float>::quiet_NaN());
    return v;
}

// lane k's buffers and synthetic inputs at cells, depth and rr rows (FMS13: rows rr .. 4 NaN in, the sentinel out; the
// selection bodies and offsets start zero)
bool make_lane(env & e, int32_t k, int32_t cells, int32_t depth, bool q8, uint32_t seed, int32_t rr) {
    e.ln.push_back(std::make_unique<lane>(e.dev0, e.dev1));
    lane & l = *e.ln.back();
    l.cells = cells;
    l.depth = depth;
    l.rows   = rr;
    l.q8 = q8;
    std::mt19937 g(seed + 1000u * static_cast<uint32_t>(k));
    bool ok = l.rg.ok();
    const size_t c = static_cast<size_t>(cells);
    const size_t sum_floats = fn::qsa::summary_bytes(cells / fn::qsa::ratio) / sizeof(float);
    const size_t body_words = size_t(R) * fn::qsa::max_row_words + 4;
    const std::vector<int32_t> dv = {depth};
    // a [R][width] buffer's rows rr .. 4, kept; a [cells][bytes] cache's cells past depth + rr (to depth + 5), kept
    const auto                 tail       = [&](size_t width_bytes) {
        l.kept.push_back({ l.b.size() - 1, size_t(rr) * width_bytes, size_t(R - rr) * width_bytes });
    };
    const auto cells_after = [&](size_t cell_bytes) {
        const int64_t c0 = std::clamp<int64_t>(int64_t(depth) + rr, 0, cells);
        const int64_t c1 = std::clamp<int64_t>(int64_t(depth) + R, 0, cells);
        l.kept.push_back({ l.b.size() - 1, size_t(c0) * cell_bytes, size_t(c1 - c0) * cell_bytes });
    };
    // an output buffer: every byte the sentinel before the launch
    const auto sentinel = [&]() {
        l.b.back()->init.assign(l.b.back()->n, sentinel_byte);
    };
    l.keys = add(l.b, e.dev0, c * IH, bf16s(g, c * IH, -2.0f, 2.0f), ok);
    cells_after(IH * 2);
    l.sums = add(l.b, e.dev0, sum_floats, uniform(g, sum_floats, -1.0f, 1.0f), ok);
    l.ik   = add(l.b, e.dev0, size_t(R) * IH, nan_tail(uniform(g, size_t(R) * IH, -2.0f, 2.0f), rr, IH), ok);
    tail(IH * 4);
    l.iq = add(l.b, e.dev0, size_t(R) * IQ, nan_tail(uniform(g, size_t(R) * IQ, -2.0f, 2.0f), rr, IQ), ok);
    tail(IQ * 4);
    l.qws = add(l.b, e.dev0, fn::qsa::workspace_bytes(), std::vector<uint8_t>(), ok);
    l.body0 = add(l.b, e.dev0, body_words, std::vector<uint32_t>(), ok);
    l.off0 = add(l.b, e.dev0, 8, std::vector<int32_t>(), ok);
    l.depth0 = add(l.b, e.dev0, 1, dv, ok);
    l.body1 = add(l.b, e.dev1, body_words, std::vector<uint32_t>(), ok);
    l.off1 = add(l.b, e.dev1, 8, std::vector<int32_t>(), ok);
    l.qfull  = add(l.b, e.dev1, size_t(R) * NH * QS,
                   nan_tail(uniform(g, size_t(R) * NH * QS, -2.0f, 2.0f), rr, size_t(NH) * QS), ok);
    tail(size_t(NH) * QS * 4);
    l.kv_k = add(l.b, e.dev1, size_t(R) * AH, nan_tail(uniform(g, size_t(R) * AH, -2.0f, 2.0f), rr, AH), ok);
    tail(AH * 4);
    l.kv_v = add(l.b, e.dev1, size_t(R) * AH, nan_tail(uniform(g, size_t(R) * AH, -2.0f, 2.0f), rr, AH), ok);
    tail(AH * 4);
    std::vector<int32_t> pv(4 * R, 0);                    // [4][rr]: the planes at the lane's row stride
    for (int32_t t = 0; t < rr; ++t) {
        pv[t] = pv[rr + t] = pv[2 * rr + t] = depth + t;  // text: t, h and w at the position, e 0
    }
    l.pos = add(l.b, e.dev1, 4 * R, pv, ok);
    if (q8) {
        std::uniform_int_distribution<int32_t> cd(-127, 127);
        std::vector<uint16_t> kv(c * AH / 2), vv(c * AH / 2);
        for (size_t i = 0; i < kv.size(); ++i) {
            kv[i] = uint16_t(uint8_t(int8_t(cd(g)))) | uint16_t(uint16_t(uint8_t(int8_t(cd(g)))) << 8);
            vv[i] = uint16_t(uint8_t(int8_t(cd(g)))) | uint16_t(uint16_t(uint8_t(int8_t(cd(g)))) << 8);
        }
        std::uniform_real_distribution<float> sd(0.001f, 0.05f);
        std::vector<uint16_t> ksv(c * SS), vsv(c * SS);
        for (size_t i = 0; i < ksv.size(); ++i) {
            ksv[i] = fn::quant::f32_to_f16(sd(g));
            vsv[i] = fn::quant::f32_to_f16(sd(g));
        }
        l.kc = add(l.b, e.dev1, c * AH / 2, kv, ok); // reason: int8 codes [cells][256] in the u16 cache pointer
        cells_after(AH);
        l.vc = add(l.b, e.dev1, c * AH / 2, vv, ok);
        cells_after(AH);
        l.ks = add(l.b, e.dev1, c * SS, ksv, ok);
        cells_after(SS * 2);
        l.vs = add(l.b, e.dev1, c * SS, vsv, ok);
        cells_after(SS * 2);
    } else {
        l.kc = add(l.b, e.dev1, c * AH, bf16s(g, c * AH, -2.0f, 2.0f), ok);
        cells_after(AH * 2);
        l.vc = add(l.b, e.dev1, c * AH, bf16s(g, c * AH, -2.0f, 2.0f), ok);
        cells_after(AH * 2);
    }
    l.out = add(l.b, e.dev1, size_t(R) * OW, std::vector<float>(), ok);
    sentinel();
    tail(OW * 4);
    l.codes = add(l.b, e.dev1, size_t(R) * OW, std::vector<int8_t>(), ok);
    sentinel();
    tail(OW);
    l.scales = add(l.b, e.dev1, size_t(R) * (OW / 32) * 2, std::vector<float>(), ok);
    sentinel();
    tail((OW / 32) * 2 * 4);
    l.aws = add(l.b, e.dev1, fn::qsa::attn::workspace_bytes(), std::vector<uint8_t>(), ok);
    l.depth1 = add(l.b, e.dev1, 1, dv, ok);
    return ok;
}

// FMS13: every kept span of lane l (rows rr .. 4, cells past depth + rr) at its initial bytes in a snapshot
bool rows_untouched(const lane & l, const std::vector<uint8_t> & snap) {
    std::vector<size_t> base;
    size_t              at = 0;
    for (const auto & x : l.b) {
        base.push_back(at);
        at += x->n;
    }
    bool ok = true;
    for (const lane::span & s : l.kept) {
        const auto & init = l.b[s.buf]->init;
        ok = ok && std::equal(snap.begin() + ptrdiff_t(base[s.buf] + s.at),
                              snap.begin() + ptrdiff_t(base[s.buf] + s.at + s.n), init.begin() + ptrdiff_t(s.at));
    }
    return ok;
}

bool idle(const env & e) {
    return cuda_ok(cudaSetDevice(e.dev0), "cudaSetDevice") && cuda_ok(cudaDeviceSynchronize(), "idle 0") &&
           cuda_ok(cudaSetDevice(e.dev1), "cudaSetDevice") && cuda_ok(cudaDeviceSynchronize(), "idle 1") &&
           cuda_ok(cudaGetLastError(), "idle");
}

// lanes 0 .. n - 1 at their initial bytes, their rings at the sentinels with zero error words
bool fresh(const env & e, int32_t n) {
    bool ok = idle(e);
    for (int32_t k = 0; k < n && ok; ++k) {
        for (const auto & x : e.ln[size_t(k)]->b) {
            ok = ok && cuda_ok(cudaSetDevice(x->dev), "cudaSetDevice") &&
                 cuda_ok(cudaMemcpy(x->p, x->init.data(), x->n, cudaMemcpyHostToDevice), "fresh");
        }
        ok = ok && e.ln[size_t(k)]->rg.reset_all();
    }
    return ok && idle(e);
}

// every byte of lane k's buffers, its X-SELECT slot on the peer and both error words
bool snapshot(const env & e, int32_t k, std::vector<uint8_t> & out) {
    const lane & l = *e.ln[size_t(k)];
    out.clear();
    bool ok = idle(e);
    for (const auto & x : l.b) {
        const size_t at = out.size();
        out.resize(at + x->n);
        ok = ok && cuda_ok(cudaSetDevice(x->dev), "cudaSetDevice") &&
             cuda_ok(cudaMemcpy(out.data() + at, x->p, x->n, cudaMemcpyDeviceToHost), "snapshot");
    }
    const size_t sb = size_t(fn::xchg::slot_words(fn::xchg::msg::select)) * sizeof(float);
    const size_t at = out.size();
    out.resize(at + sb + 8);
    uint32_t err[2] = {0, 0};
    ok = ok && cuda_ok(cudaSetDevice(e.dev1), "cudaSetDevice") &&
         cuda_ok(cudaMemcpy(out.data() + at, l.rg.slot(1, fn::xchg::msg::select, sel_slot), sb, cudaMemcpyDeviceToHost),
                 "slot") &&
         l.rg.read_err(0, err[0]) && l.rg.read_err(1, err[1]);
    std::memcpy(out.data() + at + sb, err, 8);
    return ok;
}

// on a mismatch: each differing piece of lane k's snapshot (its index in make_lane's order; then the slot, the
// error words) and its differing bytes
void diff(const lane & l, const std::vector<uint8_t> & got, const std::vector<uint8_t> & want, int32_t k) {
    size_t at = 0;
    std::vector<size_t> n;
    for (const auto & x : l.b) {
        n.push_back(x->n);
    }
    n.push_back(size_t(fn::xchg::slot_words(fn::xchg::msg::select)) * sizeof(float));
    n.push_back(8);
    for (size_t i = 0; i < n.size(); ++i) {
        size_t bad = 0;
        for (size_t j = 0; j < n[i] && at + j < got.size() && at + j < want.size(); ++j) {
            bad += got[at + j] != want[at + j] ? 1 : 0;
        }
        if (bad != 0) {
            printf("qsaj-test   lane %d piece %zu: %zu of %zu bytes differ\n", k, i, bad, n[i]);
        }
        at += n[i];
    }
}

fn::qsa::round_update_args update_args(const env & e, const lane & l) {
    fn::qsa::round_update_args a;
    a.keys = l.keys;
    a.k_norm = e.w.ik_norm;
    a.summaries = l.sums;
    a.rope = fn::qsa::model_rope();
    a.ik = l.ik;
    a.depth = l.depth0;
    a.rows      = l.rows;
    a.max_cells = l.cells;
    return a;
}

fn::qsa::select_args select_args_of(const env & e, const lane & l) {
    fn::qsa::select_args a;
    a.summaries = l.sums;
    a.q = l.iq;
    a.q_norm = e.w.iq_norm;
    a.depth = l.depth0;
    a.max_cells = l.cells;
    a.rope = fn::qsa::model_rope();
    a.ws = fn::qsa::carve(l.qws);
    a.body = l.body0;
    a.offsets = l.off0;
    return a;
}

fn::qsa::attn::args attend_args(const env & e, const lane & l) {
    fn::qsa::attn::args a = {};
    a.qfull = l.qfull;
    a.kv_k = l.kv_k;
    a.kv_v = l.kv_v;
    a.q_norm = e.w.q_norm;
    a.k_norm = e.w.k_norm;
    a.body = l.body1;
    a.offsets = l.off1;
    a.pos = l.pos;
    a.k_cache = l.kc;
    a.v_cache = l.vc;
    a.k_scale = l.ks;
    a.v_scale = l.vs;
    a.depth = l.depth1;
    a.max_cells = l.cells;
    a.theta_scale = fn::qsa::attn::rope_theta_scale();
    a.out = l.out;
    a.codes = l.codes;
    a.scales = l.scales;
    a.ws = fn::qsa::attn::carve(l.aws);
    return a;
}

// lane k's launches alone: summary, select and send on the owner, receive and attend on the peer
bool lone(const env & e, int32_t k) {
    const lane & l = *e.ln[size_t(k)];
    const fn::qsa::round_update_args ua = update_args(e, l);
    const fn::qsa::select_args sa = select_args_of(e, l);
    const fn::qsa::attn::args aa = attend_args(e, l);
    const int32_t                    rr = l.rows;
    return cuda_ok(cudaSetDevice(e.dev0), "cudaSetDevice") && fn::qsa::summary_update(ua, e.s0) &&
           fn::qsa::select(rr, sa, e.s0) &&
           fn::xchg::send_select(rr, l.off0, l.body0, l.rg.slot(1, fn::xchg::msg::select, sel_slot), e.s0) &&
           cuda_ok(cudaSetDevice(e.dev1), "cudaSetDevice") &&
           fn::xchg::recv_select(rr, l.rg.slot(1, fn::xchg::msg::select, sel_slot), l.off1, l.body1, l.rg.err(1),
                                 e.s1) &&
           fn::qsa::attn::attend(rr, aa, e.s1) && idle(e);
}

// lanes 0 .. U - 1 in one launch a piece; test_depth: select_lanes' D (0: grid_lanes')
bool joint(const env & e, int32_t U, int32_t test_depth) {
    fn::qsa::round_update_args ua[U_MAX];
    fn::qsa::select_args sa[U_MAX];
    fn::qsa::attn::args aa[U_MAX];
    const int32_t * off0[U_MAX] = {};
    const uint32_t * body0[U_MAX] = {};
    float * peer[U_MAX] = {};
    float * slot[U_MAX] = {};
    int32_t * off1[U_MAX] = {};
    uint32_t * body1[U_MAX] = {};
    uint32_t * err[U_MAX] = {};
    for (int32_t k = 0; k < U; ++k) {
        const lane & l = *e.ln[size_t(k)];
        ua[k] = update_args(e, l);
        sa[k] = select_args_of(e, l);
        aa[k] = attend_args(e, l);
        off0[k] = l.off0;
        body0[k] = l.body0;
        peer[k] = l.rg.slot(1, fn::xchg::msg::select, sel_slot);
        slot[k] = peer[k];
        off1[k] = l.off1;
        body1[k] = l.body1;
        err[k] = l.rg.err(1);
    }
    const int32_t rr = e.ln[0]->rows;
    return cuda_ok(cudaSetDevice(e.dev0), "cudaSetDevice") && fn::qsa::summary_update_lanes(U, ua, nullptr, e.s0) &&
           fn::qsa::select_lanes(U, rr, sa, e.s0, test_depth) &&
           fn::xchg::send_select_lanes(U, rr, off0, body0, peer, e.s0) &&
           cuda_ok(cudaSetDevice(e.dev1), "cudaSetDevice") &&
           fn::xchg::recv_select_lanes(U, rr, slot, off1, body1, err, e.s1) &&
           fn::qsa::attn::attend_lanes(U, rr, aa, e.s1) && idle(e);
}

struct case_cfg {
    const char * name;
    int32_t cells[U_MAX];
    int32_t depth[U_MAX];
    bool q8;
    bool sweep; // score at D 4, 8 and 16 too
};

// a case at rr rows a lane (FMS13: 2 .. 5): each lane's lone snapshot at B = rr, then every U and D jointly, each lane
// held bitwise its lone run's and (below 5) its rows rr .. 4 and cells past depth + rr untouched; U = 4 at grid_lanes'
// D twice
bool run_case(env & e, const case_cfg & cf, uint32_t seed, int32_t rr) {
    e.ln.clear();
    bool ok = true;
    for (int32_t k = 0; k < U_MAX && ok; ++k) {
        ok = make_lane(e, k, cf.cells[k], cf.depth[k], cf.q8, seed, rr);
    }
    if (!ok) {
        printf("qsaj-test R%d %s: setup FAIL\n", rr, cf.name);
        return false;
    }
    std::vector<uint8_t> want[U_MAX];
    int32_t deepest = 0;
    std::string lone_d;
    for (int32_t k = 0; k < U_MAX && ok; ++k) {
        ok = fresh(e, U_MAX) && lone(e, k) && snapshot(e, k, want[k]);
        lone_d += (k ? ", " : "") + std::to_string(fn::qsa::grid_of(rr, cf.cells[k] - rr).depth);
        deepest = std::max(deepest, cf.cells[k] - rr);
    }
    if (!ok) {
        printf("qsaj-test R%d %s: lone runs FAIL\n", rr, cf.name);
        return false;
    }
    printf("qsaj-test R%d %s: depths %d, %d, %d, %d; cells %d, %d, %d, %d; the lone score D a lane %s\n", rr, cf.name,
           cf.depth[0], cf.depth[1], cf.depth[2], cf.depth[3], cf.cells[0], cf.cells[1], cf.cells[2], cf.cells[3],
           lone_d.c_str());
    bool pass = true;
    for (int32_t U = 2; U <= U_MAX && ok; ++U) {
        const std::vector<int32_t> ds = cf.sweep ? std::vector<int32_t>{0, 4, 8, 16} : std::vector<int32_t>{0};
        int32_t lanes_deepest = 0;
        for (int32_t k = 0; k < U; ++k) {
            lanes_deepest = std::max(lanes_deepest, cf.cells[k] - rr);
        }
        for (const int32_t td : ds) {
            for (int32_t rep = 0; rep < (U == U_MAX && td == 0 ? 2 : 1) && ok; ++rep) {
                ok = fresh(e, U) && joint(e, U, td);
                int32_t same = 0, kept = 0;
                size_t bytes = 0;
                for (int32_t k = 0; k < U && ok; ++k) {
                    std::vector<uint8_t> got;
                    ok = snapshot(e, k, got);
                    same += ok && got == want[k] ? 1 : 0;
                    kept += ok && rows_untouched(*e.ln[size_t(k)], got) ? 1 : 0;
                    bytes += got.size();
                    if (ok && got != want[k]) {
                        diff(*e.ln[size_t(k)], got, want[k], k);
                    }
                }
                const fn::qsa::score_grid g    = fn::qsa::grid_lanes(rr, lanes_deepest, U, td);
                const bool                good = ok && same == U && kept == U;
                printf(
                    "qsaj-test R%d %s U%d score D %d (%s), %d CTAs a lane x %d warps%s: %d of %d lanes bitwise "
                    "their lone runs (%zu bytes), rows %d .. 4 and cells past depth + %d untouched in %d: %s\n",
                    rr, cf.name, U, g.depth, td == 0 ? "grid_lanes'" : "forced", g.ctas, g.warps,
                    rep == 1 ? ", second joint run" : "", same, U, bytes, rr, rr, kept, good ? "yes" : "NO  FAIL");
                pass = pass && good;
            }
        }
    }
    return ok && pass;
}

// out-of-contract calls refused with nothing enqueued (every lane's bytes, slots and words still fresh)
bool rejects(env & e) {
    std::vector<uint8_t> before[U_MAX];
    bool ok = fresh(e, U_MAX);
    for (int32_t k = 0; k < U_MAX && ok; ++k) {
        ok = snapshot(e, k, before[k]);
    }
    fn::qsa::round_update_args ua[U_MAX];
    fn::qsa::select_args sa[U_MAX];
    fn::qsa::attn::args aa[U_MAX];
    const int32_t * off0[U_MAX] = {};
    const uint32_t * body0[U_MAX] = {};
    float * slot[U_MAX] = {};
    int32_t * off1[U_MAX] = {};
    uint32_t * body1[U_MAX] = {};
    uint32_t * err[U_MAX] = {};
    for (int32_t k = 0; k < U_MAX; ++k) {
        const lane & l = *e.ln[size_t(k)];
        ua[k] = update_args(e, l);
        sa[k] = select_args_of(e, l);
        aa[k] = attend_args(e, l);
        off0[k] = l.off0;
        body0[k] = l.body0;
        slot[k] = l.rg.slot(1, fn::xchg::msg::select, sel_slot);
        off1[k] = l.off1;
        body1[k] = l.body1;
        err[k] = l.rg.err(1);
    }
    int32_t refused = 0, cases = 0;
    const auto refuse = [&](bool launched) {
        ++cases;
        refused += launched ? 0 : 1;
    };
    ok = ok && cuda_ok(cudaSetDevice(e.dev0), "cudaSetDevice");
    refuse(fn::qsa::summary_update_lanes(1, ua, nullptr, e.s0));
    refuse(fn::qsa::summary_update_lanes(5, ua, nullptr, e.s0));
    refuse(fn::qsa::select_lanes(1, R, sa, e.s0));
    refuse(fn::qsa::select_lanes(5, R, sa, e.s0));
    refuse(fn::qsa::select_lanes(2, R, sa, e.s0, 12));  // a D outside 4, 8, 16
    refuse(fn::qsa::select_lanes(2, 1, sa, e.s0));      // FMS13: rows outside 2 .. 5
    refuse(fn::qsa::select_lanes(2, R + 1, sa, e.s0));
    fn::qsa::select_args sb[U_MAX];
    std::copy_n(sa, U_MAX, sb);
    sb[1].q_ahead = 1; // the draft's form
    refuse(fn::qsa::select_lanes(2, R, sb, e.s0));
    std::copy_n(sa, U_MAX, sb);
    sb[1].max_cells = R - 1;
    refuse(fn::qsa::select_lanes(2, R, sb, e.s0));
    refuse(fn::xchg::send_select_lanes(1, R, off0, body0, slot, e.s0));
    refuse(fn::xchg::send_select_lanes(2, R + 1, off0, body0, slot, e.s0));
    ok = ok && cuda_ok(cudaSetDevice(e.dev1), "cudaSetDevice");
    refuse(fn::xchg::recv_select_lanes(5, R, slot, off1, body1, err, e.s1));
    uint32_t * err_b[U_MAX] = {err[0], nullptr, err[2], err[3]};
    refuse(fn::xchg::recv_select_lanes(2, R, slot, off1, body1, err_b, e.s1));
    refuse(fn::qsa::attn::attend_lanes(1, R, aa, e.s1));
    refuse(fn::qsa::attn::attend_lanes(2, R + 1, aa, e.s1));
    fn::qsa::attn::args ab[U_MAX];
    std::copy_n(aa, U_MAX, ab);
    ab[1].stamps = reinterpret_cast<uint64_t *>(ab[1].out); // the bench's diagnostic stamps
    refuse(fn::qsa::attn::attend_lanes(2, R, ab, e.s1));
    std::copy_n(aa, U_MAX, ab);
    ab[1].k_scale = ab[1].v_scale = reinterpret_cast<uint16_t *>(ab[1].out); // lane 1 KV8 beside BF16 lane 0
    refuse(fn::qsa::attn::attend_lanes(2, R, ab, e.s1));
    std::copy_n(aa, U_MAX, ab);
    ab[1].shift = reinterpret_cast<const int64_t *>(ab[1].depth); // the host-backed form
    refuse(fn::qsa::attn::attend_lanes(2, R, ab, e.s1));
    ok = ok && idle(e);
    int32_t same = 0;
    for (int32_t k = 0; k < U_MAX && ok; ++k) {
        std::vector<uint8_t> after;
        ok = snapshot(e, k, after);
        same += ok && after == before[k] ? 1 : 0;
    }
    printf("qsaj-test rejects: %d of %d refused, every lane untouched: %s\n", refused, cases,
           ok && same == U_MAX ? "yes" : "NO  FAIL");
    return ok && refused == cases && same == U_MAX;
}

void resources() {
    fn::qsa::attn::resources r = {};
    int32_t grid = 0;
    for (const bool q8 : {false, true}) {
        for (int32_t U = 2; U <= U_MAX; ++U) {
            if (fn::qsa::attn::lanes_resources(r, q8, R, U, grid)) {
                printf("qsaj-test attend_lanes_kernel<%s> U%d: %d registers, %d local bytes, %d shared bytes, %d CTAs "
                       "an SM, grid %d\n",
                       q8 ? "q8" : "bf16", U, r.regs, r.local_bytes, r.shared_bytes, r.ctas_per_sm, grid);
            }
        }
    }
}

} // namespace

int fnfl_qsaj_test(int dev0, int dev1) {
    if (dev0 == dev1) {
        printf("qsaj-test: loopback on device %d: every launch on one stream\n", dev0);
    } else if (!fn::xchg::enable_peer_access(dev0, dev1)) {
        printf("qsaj-test: FAIL (no peer access)\n");
        return 1;
    }
    bool ok = true;
    {
        env e;
        e.dev0 = dev0;
        e.dev1 = dev1;
        ok = cuda_ok(cudaSetDevice(dev0), "cudaSetDevice") &&
             cuda_ok(cudaStreamCreateWithFlags(&e.s0, cudaStreamNonBlocking), "stream");
        if (ok && dev1 != dev0) {
            ok = cuda_ok(cudaSetDevice(dev1), "cudaSetDevice") &&
                 cuda_ok(cudaStreamCreateWithFlags(&e.s1, cudaStreamNonBlocking), "stream");
        } else {
            e.s1 = e.s0;
        }
        ok = ok && make_shared(e);
        resources();
        // about 2K cells (lane 1 below 512 blocks: no score work; lane 3 a smaller capacity) and about 64K (lane 2 a
        // smaller capacity, so the deepest lane sizes the grid), BF16 with the D sweep, KV8 at grid_lanes' D; then a
        // lane at a refused depth (cells - 4)
        const case_cfg cases[] = {
            {"2K bf16", {4096, 4096, 4096, 3072}, {2047, 2040, 2100, 2300}, false, true},
            {"64K bf16", {69632, 69632, 66560, 69632}, {65531, 64000, 66000, 60001}, false, true},
            {"2K q8", {4096, 4096, 4096, 3072}, {2047, 2040, 2100, 2300}, true, false},
            {"64K q8", {69632, 69632, 66560, 69632}, {65531, 64000, 66000, 60001}, true, false},
            {"refused", {4096, 4096, 4096, 4096}, {2100, 4092, 2051, 2049}, false, false},
        };
        // FMS13: every case at R = 2 .. 5 rows a lane; the rejects on the 2K BF16 lanes at R = 5
        for (int32_t rr = 2; rr <= R && ok; ++rr) {
            uint32_t seed = 1;
            for (const case_cfg & cf : cases) {
                if (!ok) {
                    break;
                }
                ok = run_case(e, cf, seed++, rr) && ok;
                if (rr == R && std::strcmp(cf.name, "2K bf16") == 0) {
                    ok = rejects(e) && ok;
                }
            }
        }
        e.ln.clear();
        idle(e);
        cudaSetDevice(dev0);
        cudaStreamDestroy(e.s0);
        if (e.s1 != e.s0) {
            cudaSetDevice(dev1);
            cudaStreamDestroy(e.s1);
        }
    }
    printf("qsaj-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
