#include "fn/proto/xp1/pair.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_test.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/placement.h"
#include "fn/proto/xp1/timing.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>

namespace fn::xp1 {

namespace {

constexpr int32_t D = 2560;
constexpr int32_t F = 1280;
constexpr int32_t S = 4;      // HC streams
constexpr int32_t K = S * D;  // the GGUF row length of HC down and inject
constexpr int32_t L = 320;    // HC low rank
constexpr int32_t J = 4;      // HC injection rows
constexpr int32_t RD = L + J; // HC down rows
constexpr int32_t W = 2051;   // the most cells a row selects: 512 blocks of 4 and a tail of 3
constexpr int32_t BW = fn::qsa::max_row_words; // 517: the longest row body
constexpr int32_t GS = fn::experts::glu_slots; // 44 GLU slots (EX2: the routed pairs and the shared rows)
constexpr int64_t flush_bytes = 32ll << 20;    // > 6 MiB L2
constexpr int32_t index_owner = (11 / 4) % 2;  // layer 11's (the frame's (layer / full_attn_interval) % 2)

namespace x = fn::xchg;

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

cudaStream_t cs(void * s) {
    return static_cast<cudaStream_t>(s);
}

uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

uint16_t to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u); // round to nearest even; the synthetic values are finite
    return static_cast<uint16_t>(u >> 16);
}

// a uniform value in [-a, a) from a hash
float hash_unit(uint64_t seed, uint64_t i, float a) {
    const uint64_t h = mix64(seed * 0x100000001b3ull ^ i);
    return a * (static_cast<float>(h >> 40) / 8388608.0f - 1.0f);
}

int64_t tensor_bytes(const char * name, int32_t card) {
    for (int32_t t = 0; t < fn::model::n_tensors; ++t) {
        if (std::strcmp(fn::model::tensors[t].name, name) == 0) {
            return fn::model::tensors[t].card[card].bytes;
        }
    }
    return -1;
}

// the tensors each stand-in streams on a card (R1's placement)
std::vector<std::string> piece_tensors(piece p) {
    switch (p) {
        case piece::gdn_proj_conv:
            return {"blk.10.attn_qkv.weight",   "blk.10.attn_gate.weight", "blk.10.ssm_alpha.weight",
                    "blk.10.ssm_beta.weight",   "blk.10.ssm_conv1d.weight", "blk.10.ssm_a",
                    "blk.10.ssm_dt.bias"};
        case piece::gdn_recur: return {"blk.10.ssm_norm.weight"};
        case piece::gdn_out: return {"blk.10.ssm_out.weight"};
        case piece::index_q_proj: return {"blk.11.indexer.q_proj.weight", "blk.11.indexer.q_norm.weight"};
        case piece::index_k_proj: return {"blk.11.indexer.k_proj.weight", "blk.11.indexer.k_norm.weight"};
        case piece::qsa_qkv:
            return {"blk.11.attn_q.weight", "blk.11.attn_k.weight", "blk.11.attn_v.weight", "blk.11.attn_q_norm.weight",
                    "blk.11.attn_k_norm.weight"};
        case piece::qsa_attention: return {};
        case piece::qsa_out: return {"blk.11.attn_output.weight"};
        case piece::router10: return {"blk.10.ffn_gate_inp.weight", "blk.10.ffn_gate_inp_shexp.weight"};
        case piece::sh_gate_up10: return {"blk.10.ffn_gate_shexp.weight", "blk.10.ffn_up_shexp.weight"};
        case piece::sh_down10: return {"blk.10.ffn_down_shexp.weight"};
        case piece::router11: return {"blk.11.ffn_gate_inp.weight", "blk.11.ffn_gate_inp_shexp.weight"};
        case piece::sh_gate_up11: return {"blk.11.ffn_gate_shexp.weight", "blk.11.ffn_up_shexp.weight"};
        case piece::sh_down11: return {"blk.11.ffn_down_shexp.weight"};
    }
    return {};
}

// DP2's dense mode: what a dense_rest stand-in still streams (its piece's tensors outside DP2's input launch)
std::vector<std::string> rest_tensors(piece p) {
    switch (p) {
        case piece::gdn_proj_conv:
            return {"blk.10.ssm_conv1d.weight", "blk.10.ssm_a", "blk.10.ssm_dt.bias"};
        case piece::qsa_qkv:
            return {"blk.11.attn_q_norm.weight", "blk.11.attn_k_norm.weight"};
        default:
            return {}; // index_k_proj: the raw key write only (its norm is QA1b's)
    }
}

int32_t hc_slot_graph(int32_t k) { // HC2's slot_of for a graph of 4 sublayers: 0, 1, 2, 1
    return k == 3 ? 1 : k % x::n_slot;
}

// owns device allocations, freed on its card
struct dev_pool {
    int32_t dev = -1;
    std::vector<void *> ptrs;
    ~dev_pool() {
        if (dev >= 0) {
            cudaSetDevice(dev);
        }
        for (void * p : ptrs) {
            cudaFree(p);
        }
    }
    template <typename T> T * get(size_t bytes, std::string & err, const char * what) {
        void * p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, bytes), what, err) || !cuda_ok(cudaMemset(p, 0, bytes), what, err)) {
            return nullptr;
        }
        ptrs.push_back(p);
        return static_cast<T *>(p);
    }
};

struct card {
    int32_t dev = -1;
    cudaStream_t st = nullptr;
    dev_pool pool;
    // HC: weights per sublayer (L10 attention, L10 FFN, L11 attention, L11 FFN) and two activation sets
    uint16_t * w_down[4] = {};
    uint16_t * w_up[4] = {};
    float * gamma[4] = {};
    float *x[2] = {}, *mom[2] = {}, *inject[2] = {}, *full[2] = {}, *out[2] = {};
    uint32_t * runs = nullptr;      // HC up's run counters (HC3), zeroed once
    float * in0[max_rows + 1] = {}; // the teacher-forced input per B: own half [B][4][1280]
    float * zero = nullptr;          // [4][1280]
    float * partial = nullptr;       // [4][2560]
    float * routed = nullptr;        // [4][2560] V1+: the routed experts' partial
    // the stand-ins
    void * w[n_piece] = {};
    int64_t wb[n_piece] = {};
    uint32_t * sink = nullptr; // [n_piece][sink_words]
    int32_t pad[n_piece] = {};
    float * proj_out = nullptr;  // [4][8240]
    float * recur_out = nullptr; // [4][3072]
    float * gdn_state = nullptr; // [5][24 x 128 x 128] versions
    float * conv_hist = nullptr; // [5][5120 x 3]
    // GD1 (gdn != 0): gdn_core's inputs (the projections' stand-in writes qkv; z, alpha and beta are fixed synthetic
    // values), weights, outputs in FNFL's activation format and workspace
    float * gd_qkv = nullptr;    // [4][5120]
    float * gd_z = nullptr;      // [4][3072]
    float * gd_ab = nullptr;     // [2][4][24] alpha, then beta
    float * gd_w = nullptr;      // conv1d [5120][4], dt [24], a [24], ssm_norm [128]
    int8_t * gd_codes = nullptr; // [4][3072]
    float * gd_scales = nullptr; // [4][96][2]
    void * gd_ws = nullptr;      // fn::gdn::workspace_bytes()
    uint32_t * gd_cur = nullptr; // the committed slot (SR1): 0 throughout, this prototype keeps its own state copies
    float * qkv_out = nullptr;   // [4][6144]
    float * attn_out = nullptr;  // [4][3072]
    uint16_t * k_cache = nullptr; // [cells][256] BF16, this card's KV head
    uint16_t * v_cache = nullptr;
    // the QSA index: the whole history (the S2 owner's)
    uint16_t * keys = nullptr; // [cells][128] BF16
    float * summ = nullptr;    // tiled F32
    float * ik = nullptr; // [4][128]
    float * iq = nullptr; // [4][512]
    float * q_norm = nullptr;
    float * k_norm = nullptr;
    uint8_t * qsa_ws = nullptr;
    uint32_t * body = nullptr;     // [4][517] QA1b's body (the owner's select writes it)
    int32_t * off = nullptr;       // [8] its byte offsets
    uint32_t * sel_body = nullptr; // [4][517] the received X-SELECT (the peer's)
    int32_t * sel_off = nullptr;   // [8]
    uint32_t * sel2_body = nullptr; // the isolated X-SELECT's receive on the owner
    int32_t * sel2_off = nullptr;
    // FFN
    int32_t * ids_full = nullptr; // [4][10] the router owner's (X-ROUTE's source)
    int32_t * ids = nullptr;
    uint8_t * meta = nullptr; // the routing meta of the experts' IDs (XB3: gate_up's, for down)
    float * wts = nullptr;
    float * sg = nullptr;
    int8_t * xc = nullptr;   // EX1b's input in FNFL's format: codes [4][2560], scales [4][80] {d, s}
    float * xs = nullptr;
    int8_t * gluc = nullptr; // the GLU: codes [44][320], scales [44][10] {d, s}
    float * glus = nullptr;
    float * glu_f32 = nullptr;
    uint32_t * status = nullptr;
    float * sh_glu = nullptr;  // [4][320]
    float * sh_part = nullptr; // [4][2560]
    uint8_t * ex_gu[2] = {};
    uint8_t * ex_dn[2] = {};
    uint8_t * ex_sg[2] = {}; // EX2's shared expert a layer: gate and up in DP2's packed layout, down in pack_down's
    uint8_t * ex_su[2] = {};
    uint8_t * ex_sd[2] = {};
    uint8_t * grids = nullptr;
    // DP2's dense mode: each launch's arguments (weights, outputs, work; inputs set by build) and its buffers
    fn::dense::product_args d_gdn, d_ssm, d_qsa, d_att, d_rt[2]; // (the shared expert's left with EX2)
    int8_t * d_xc = nullptr; // full[0] in FNFL's format: [4][2560], scales [4][80] {d, s}
    float * d_xs = nullptr;
    int8_t * d_mc = nullptr; // recur_out and attn_out in FNFL's format: [4][3072], scales [4][96]
    float * d_ms = nullptr;
    float * d_qkv = nullptr;   // GDN input's outputs [4][5120], ...
    float * d_q = nullptr;     // QSA input's [4][6144], ...
    float * d_iq = nullptr;    // [4][512]
    float * d_ik = nullptr;    // [4][128]
    float * d_k = nullptr;     // QSA input's k and v: [4][256] each
    float * d_v = nullptr;
    float * qa_norm = nullptr;   // QA2: attn_q_norm, attn_k_norm [256] each (synthetic)
    int32_t * qa_pos = nullptr;  // QA2: the rope positions [4][B] for B = 1, 2, 3, 4 back to back
    int32_t * d_depth = nullptr; // the pair's depth on the device (TG1: select, the key write and attend read it)
    void * qa_ws = nullptr;      // QA2: fn::qsa::attn::workspace_bytes()
    int32_t * d_ids = nullptr;   // DP2's router: [4][10]
    uint8_t * d_meta = nullptr;  // and the routing meta of d_ids (XB3: gate_up's)
    int32_t * ids_hx7[2] = {};   // HX7's captured IDs of the round's decode, layers 10 and 11: [3][10]
    // rounds and traffic
    uint64_t * stamp = nullptr;
    int32_t * counter = nullptr;
    int32_t * decode = nullptr;
    int32_t * routes[2] = {}; // [1024][3][10]
    int32_t * cells = nullptr; // [1024][3][2051]
    void * flush = nullptr;
    uint32_t * flags = nullptr;
    uint64_t * pings = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t ev[2] = {nullptr, nullptr};
};

struct use_counts {
    int64_t sub = 0, a = 0, f = 0, route = 0, select = 0;
};

} // namespace

const char * variant_name(variant v) {
    static const char * n[] = {"V0", "V1", "V2"};
    return n[static_cast<int32_t>(v)];
}

bool parse_variant(const std::string & s, variant & v) {
    for (int32_t i = 0; i < n_variant; ++i) {
        if (s == variant_name(static_cast<variant>(i))) {
            v = static_cast<variant>(i);
            return true;
        }
    }
    return false;
}

int64_t piece_weight_bytes(piece p, int32_t c) {
    int64_t n = 0;
    for (const std::string & t : piece_tensors(p)) {
        int64_t b = tensor_bytes(t.c_str(), c);
        if (b == 0) {
            b = tensor_bytes(t.c_str(), 1 - c); // not resident on this card in S2: the owner's (V2)
        }
        if (b < 0) {
            return -1;
        }
        n += b;
    }
    return n;
}

bool dense_replaced(piece p) {
    switch (p) {
        case piece::gdn_out:
        case piece::index_q_proj:
        case piece::qsa_out:
        case piece::router10:
        case piece::sh_gate_up10:
        case piece::sh_down10:
        case piece::router11:
        case piece::sh_gate_up11:
        case piece::sh_down11:
            return true;
        default:
            return false;
    }
}

bool dense_rest(piece p) {
    return p == piece::gdn_proj_conv || p == piece::index_k_proj || p == piece::qsa_qkv;
}

struct pair::impl {
    card c[2];
    std::unique_ptr<x::rings> rg;
    use_counts u;
    facts fx;
    fn::experts::synth syn_gu[2], syn_dn;
    fn::dense::synth sh_g[2], sh_u[2], sh_d; // EX2's shared expert (facts::sh_gate)
    bool have_graphs = false;
    bool loopback = false;
};

pair::pair() : m_(std::make_unique<impl>()) {}

pair::~pair() {
    drop_graphs();
    for (int32_t k = 0; k < 2; ++k) {
        card & c = m_->c[k];
        if (c.dev < 0) {
            continue;
        }
        cudaSetDevice(c.dev);
        if (c.st != nullptr && !(k == 1 && m_->loopback)) {
            cudaStreamSynchronize(c.st);
            cudaStreamDestroy(c.st);
        }
        for (cudaEvent_t & e : c.ev) {
            if (e != nullptr) {
                cudaEventDestroy(e);
            }
        }
    }
    m_->rg.reset();
}

int32_t pair::device(int32_t card) const {
    return m_->c[card].dev;
}

void * pair::stream(int32_t card) const {
    return m_->c[card].st;
}

uint64_t * pair::stamps(int32_t card) const {
    return m_->c[card].stamp;
}

void * pair::flush_buffer(int32_t card) const {
    return m_->c[card].flush;
}

int64_t pair::flush_size() const {
    return flush_bytes;
}

bool pair::clock_offset(int32_t n, double & offset_ns, double & rtt_ns, std::string & err) {
    if (n < 2 || n > 64 || m_->loopback || !sync(err)) {
        return fail(err, "clock offset");
    }
    card & c0 = m_->c[0];
    card & c1 = m_->c[1];
    if (!fn::xp1::clock_offset(c0.dev, c1.dev, c0.flags, c1.flags, c0.pings, c1.pings, n, offset_ns, rtt_ns)) {
        return fail(err, "clock offset: a round trip failed");
    }
    return true;
}

const pair::facts & pair::info() const {
    return m_->fx;
}

void pair::set_pad(piece p, int32_t card, int32_t pad) {
    m_->c[card].pad[static_cast<int32_t>(p)] = pad;
}

int32_t pair::pad(piece p, int32_t card) const {
    return m_->c[card].pad[static_cast<int32_t>(p)];
}

namespace {

template <typename T> bool put(T * dst, const std::vector<T> & src, std::string & err, const char * what) {
    return cuda_ok(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), what, err);
}

// hc_test's weights for one sublayer (uniform ranges), seeded by the sublayer
void make_hc(int32_t k, std::vector<uint16_t> & down, std::vector<uint16_t> & inject, std::vector<uint16_t> & up,
             std::vector<float> & gamma) {
    std::mt19937 rng(20261011u + 31u * static_cast<uint32_t>(k));
    std::uniform_real_distribution<float> d_down(-0.07f, 0.07f), d_up(-0.25f, 0.25f), d_gamma(0.5f, 1.5f);
    down.resize(size_t(L) * K);
    inject.resize(size_t(J) * K);
    up.resize(size_t(K) * L);
    gamma.resize(K);
    for (auto & v : down) {
        v = to_bf16(d_down(rng));
    }
    for (auto & v : inject) {
        v = to_bf16(d_down(rng));
    }
    for (auto & v : up) {
        v = to_bf16(d_up(rng));
    }
    for (auto & v : gamma) {
        v = d_gamma(rng);
    }
}

// one DP2 launch on a card: synthetic weights per matrix (dp-bench's rows), outputs, splits, work; build sets inputs
struct dense_mat {
    fn::dense::fmt f;
    int32_t m;
    fn::dense::epi e;
    int32_t pair;
    float * out;
};

bool make_dense(card & c, int32_t k, const std::vector<dense_mat> & mats, const uint8_t * grids,
                std::map<int64_t, fn::dense::synth> & dsyn, fn::dense::product_args & a, std::string & err) {
    a = fn::dense::product_args();
    a.k = k;
    a.n_mats = static_cast<int32_t>(mats.size());
    a.grids = grids;
    for (size_t i = 0; i < mats.size(); ++i) {
        const dense_mat & d = mats[i];
        const int64_t key = int64_t(d.f) * 65536 + k;
        if (dsyn.count(key) == 0 && !fn::dense::make_synth(d.f, k, dsyn[key])) {
            return fail(err, "dense synthetic rows");
        }
        std::vector<uint8_t> h;
        if (!fn::dense::pack_synth(dsyn[key], d.m, int32_t(37 * i + 11) % fn::dense::n_synth_rows,
                                   fn::dense::n_synth_rows, h)) {
            return fail(err, "dense pack");
        }
        uint8_t * w = c.pool.get<uint8_t>(h.size(), err, "dense weights");
        if (w == nullptr || !put(w, h, err, "dense weights")) {
            return false;
        }
        fn::dense::mat & m = a.mats[i];
        m.f = d.f;
        m.m = d.m;
        m.w = w;
        m.out = d.out;
        m.e = d.e;
        m.pair = d.pair;
    }
    fn::dense::choose_splits(a);
    const size_t wb = fn::dense::work_bytes(a);
    a.work = wb > 0 ? c.pool.get<uint8_t>(wb, err, "dense work") : nullptr; // zeroed by the pool
    return wb == 0 || a.work != nullptr;
}

// owner: the card that runs layer 11's indexer (S2), whose QSA input launch also projects the index query and key
bool init_dense(card & c, bool owner, std::map<int64_t, fn::dense::synth> & dsyn, const std::vector<uint8_t> & dgrids,
                std::string & err) {
    namespace d = fn::dense;
    dev_pool & P = c.pool;
    bool ok = true;
    const auto f32 = [&](size_t n, const char * what) {
        float * p = P.get<float>(n * 4, err, what);
        ok = ok && p != nullptr;
        return p;
    };
    uint8_t * grids = P.get<uint8_t>(dgrids.size(), err, "dense grids");
    ok = grids != nullptr && put(grids, dgrids, err, "dense grids");
    c.d_xc = P.get<int8_t>(fn::act::codes_bytes(max_rows, D), err, "dense x codes");
    c.d_xs = P.get<float>(fn::act::scales_bytes(max_rows, D), err, "dense x scales");
    c.d_mc = P.get<int8_t>(fn::act::codes_bytes(max_rows, 3072), err, "dense mid codes");
    c.d_ms = P.get<float>(fn::act::scales_bytes(max_rows, 3072), err, "dense mid scales");
    c.d_ids = P.get<int32_t>(size_t(max_rows) * 10 * 4, err, "dense ids");
    c.d_meta = P.get<uint8_t>(fn::experts::meta_bytes, err, "dense routing meta");
    c.ids_hx7[0] = P.get<int32_t>(size_t(max_rows) * 10 * 4, err, "hx7 ids");
    c.ids_hx7[1] = P.get<int32_t>(size_t(max_rows) * 10 * 4, err, "hx7 ids");
    ok = ok && c.d_xc != nullptr && c.d_xs != nullptr && c.d_mc != nullptr && c.d_ms != nullptr && c.d_ids != nullptr &&
         c.d_meta != nullptr && c.ids_hx7[0] != nullptr && c.ids_hx7[1] != nullptr;
    c.d_qkv = f32(size_t(max_rows) * 5120, "dense qkv");
    float * z = f32(size_t(max_rows) * 3072, "dense z");
    float * alpha = f32(size_t(max_rows) * 32, "dense alpha");
    float * beta = f32(size_t(max_rows) * 32, "dense beta");
    c.d_q = f32(size_t(max_rows) * 6144, "dense q");
    float * kk = f32(size_t(max_rows) * 256, "dense k");
    float * vv = f32(size_t(max_rows) * 256, "dense v");
    c.d_k = kk;
    c.d_v = vv;
    c.d_iq = f32(size_t(max_rows) * 512, "dense iq");
    c.d_ik = f32(size_t(max_rows) * 128, "dense ik");
    float * logits = f32(size_t(max_rows) * 512, "dense logits");
    float * gate = f32(size_t(max_rows) * 4, "dense gate");
    if (!ok) {
        return false;
    }
    // the deployed formats of layers 10 and 11 (fn/model/placement), each card's shapes (dense.h)
    const d::fmt q6 = d::fmt::q6_k, q5 = d::fmt::q5_k, q4 = d::fmt::q4_k, bf = d::fmt::bf16;
    const d::epi e = d::epi::f32;
    ok =
        make_dense(c, D,
                   {{q6, 5120, e, -1, c.d_qkv}, {q6, 3072, e, -1, z}, {bf, 24, e, -1, alpha}, {bf, 24, e, -1, beta}},
                   grids, dsyn, c.d_gdn, err) &&
        make_dense(c, 3072, {{q5, D, e, -1, c.partial}}, grids, dsyn, c.d_ssm, err) &&
        make_dense(c, D,
                   owner ? std::vector<dense_mat>{{q4, 6144, e, -1, c.d_q},
                                                  {q4, 256, e, -1, kk},
                                                  {q4, 256, e, -1, vv},
                                                  {bf, 512, e, -1, c.d_iq},
                                                  {bf, 128, e, -1, c.d_ik}}
                         : std::vector<dense_mat>{{q4, 6144, e, -1, c.d_q}, {q4, 256, e, -1, kk}, {q4, 256, e, -1, vv}},
                   grids, dsyn, c.d_qsa, err) &&
        make_dense(c, 3072, {{q6, D, e, -1, c.partial}}, grids, dsyn, c.d_att, err);
    for (int32_t l = 0; l < 2 && ok; ++l) {
        ok = make_dense(c, D, {{bf, 512, e, -1, logits}, {bf, 1, e, -1, gate}}, grids, dsyn, c.d_rt[l], err);
        if (ok) {
            d::route_args & ra = c.d_rt[l].route;
            ra.logits = 0;
            ra.gate = 1;
            ra.ids = c.d_ids;
            ra.wts = c.wts;
            ra.sg = c.sg;
        }
    }
    return ok;
}

} // namespace

bool pair::init(const pair_config & cfg, std::string & err) {
    cfg_ = cfg;
    impl & m = *m_;
    facts & fx = m.fx;
    fx.dev[0] = cfg.dev[0];
    fx.dev[1] = cfg.dev[1];
    for (int32_t p = 0; p < n_piece; ++p) {
        fx.c_piece[p] = 0.0625f * static_cast<float>(p + 1);
    }
    // loopback (dev0 == dev1, for Compute Sanitizer): both cards' buffers and rings on one device and every op on card
    // 0's stream in program order, so each send precedes its read; eager steps only
    m.loopback = cfg.dev[0] == cfg.dev[1];
    if (!m.loopback && !x::enable_peer_access(cfg.dev[0], cfg.dev[1])) {
        return fail(err, "two cards with peer access are needed");
    }
    m.rg = std::make_unique<x::rings>(cfg.dev[0], cfg.dev[1]);
    if (!m.rg->ok()) {
        return fail(err, "rings");
    }
    if (!cfg.hx7_dir.empty() && !load_hx7(cfg.hx7_dir, fx.hx7, err)) {
        return false;
    }
    // the synthetic experts: EX1b's generator (cached in synth_dir), every expert ID on a synthetic expert
    if (!fn::experts::load_synth(fn::experts::fmt::iq2_xs, cfg.synth_dir, m.syn_gu[0]) ||
        !fn::experts::load_synth(fn::experts::fmt::iq2_xxs, cfg.synth_dir, m.syn_gu[1]) ||
        !fn::experts::load_synth(fn::experts::fmt::q2_0, cfg.synth_dir, m.syn_dn)) {
        return fail(err, "synthetic experts in " + cfg.synth_dir);
    }
    fx.gate_up[0] = &m.syn_gu[0];
    fx.gate_up[1] = &m.syn_gu[1];
    fx.down = &m.syn_dn;
    // EX2's shared expert in the deployed formats of layers 10 and 11 (fn/model/placement): DP2's synthetic rows
    for (int32_t l = 0; l < 2; ++l) {
        const fn::dense::fmt g = l == 0 ? fn::dense::fmt::iq3_s : fn::dense::fmt::q4_k;
        const fn::dense::fmt u = l == 0 ? fn::dense::fmt::iq4_xs : fn::dense::fmt::q4_k;
        if (!fn::dense::make_synth(g, D, m.sh_g[l]) || !fn::dense::make_synth(u, D, m.sh_u[l])) {
            return fail(err, "shared expert synthetic rows");
        }
        fx.sh_gate[l] = &m.sh_g[l];
        fx.sh_up[l] = &m.sh_u[l];
        for (int32_t r = 0; r < 2; ++r) {
            for (int32_t t = 0; t < 3; ++t) {
                fx.sh_off[l][r][t] = (97 * (4 * l + 2 * r + t) + 13) % fn::dense::n_synth_rows;
            }
        }
    }
    if (!fn::dense::make_synth(fn::dense::fmt::iq4_nl, L, m.sh_d)) {
        return fail(err, "shared down synthetic rows");
    }
    fx.sh_down = &m.sh_d;
    fx.map.resize(512);
    for (int32_t e = 0; e < 512; ++e) {
        fx.map[e] = e % fn::experts::n_synth;
    }
    std::vector<uint8_t> grids;
    if (!fn::experts::grid_buffer(grids)) {
        return fail(err, "grid table");
    }
    for (int32_t k = 0; k < 4; ++k) {
        make_hc(k, fx.hc_down[k], fx.hc_inject[k], fx.hc_up[k], fx.hc_gamma[k]);
    }
    // DP2's dense mode: its synthetic rows per (format, K), shared by the cards, and its tables
    std::map<int64_t, fn::dense::synth> dsyn;
    std::vector<uint8_t> dgrids;
    if (cfg.dense && !fn::dense::grid_buffer(dgrids)) {
        return fail(err, "dense grid table");
    }
    for (int32_t B = 1; B <= max_rows; ++B) {
        std::mt19937 rng(1000u * static_cast<uint32_t>(B) + 7u);
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        fx.in0[B].resize(size_t(B) * S * D);
        for (auto & v : fx.in0[B]) {
            v = d(rng);
        }
    }

    for (int32_t r = 0; r < 2; ++r) {
        card & c = m.c[r];
        c.dev = cfg.dev[r];
        c.pool.dev = c.dev;
        dev_pool & P = c.pool;
        if (r == 1 && m.loopback) {
            c.st = m.c[0].st;
        }
        if (!cuda_ok(cudaSetDevice(c.dev), "device", err) ||
            (c.st == nullptr && !cuda_ok(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking), "stream", err)) ||
            !cuda_ok(cudaEventCreate(&c.ev[0]), "event", err) || !cuda_ok(cudaEventCreate(&c.ev[1]), "event", err)) {
            return false;
        }
        bool ok = true;
        const auto f32 = [&](size_t n, const char * what) {
            float * p = P.get<float>(n * 4, err, what);
            ok = ok && p != nullptr;
            return p;
        };
        // HC
        for (int32_t k = 0; k < 4 && ok; ++k) {
            std::vector<uint16_t> wd(fn::hc::w_down_elems), wu(fn::hc::w_up_elems);
            std::vector<float> g(size_t(S) * F);
            for (int32_t s = 0; s < S; ++s) {
                for (int32_t j = 0; j < RD; ++j) {
                    const uint16_t * src = j < L ? &fx.hc_down[k][size_t(j) * K] : &fx.hc_inject[k][size_t(j - L) * K];
                    for (int32_t i = 0; i < F; ++i) {
                        wd[(size_t(s) * RD + j) * F + i] = src[s * D + r * F + i];
                    }
                }
                for (int32_t i = 0; i < F; ++i) {
                    g[size_t(s) * F + i] = fx.hc_gamma[k][s * D + r * F + i];
                }
            }
            for (int32_t cc = 0; cc < F; ++cc) {
                for (int32_t s = 0; s < S; ++s) {
                    const uint16_t * src = &fx.hc_up[k][size_t(s * D + r * F + cc) * L];
                    std::copy(src, src + L, &wu[(size_t(cc) * S + s) * L]);
                }
            }
            c.w_down[k] = P.get<uint16_t>(wd.size() * 2, err, "hc w_down");
            c.w_up[k] = P.get<uint16_t>(wu.size() * 2, err, "hc w_up");
            c.gamma[k] = f32(g.size(), "hc gamma");
            ok = ok && c.w_down[k] != nullptr && c.w_up[k] != nullptr && put(c.w_down[k], wd, err, "w_down") &&
                 put(c.w_up[k], wu, err, "w_up") && put(c.gamma[k], g, err, "gamma");
        }
        for (int32_t s = 0; s < 2 && ok; ++s) {
            c.x[s] = f32(size_t(max_rows) * S * F, "x");
            c.mom[s] = f32(size_t(max_rows) * fn::hc::n_mom, "mom");
            c.inject[s] = f32(size_t(max_rows) * S, "inject");
            c.full[s] = f32(size_t(max_rows) * D, "full");
            c.out[s] = f32(size_t(max_rows) * F, "out");
        }
        for (int32_t B = 1; B <= max_rows && ok; ++B) {
            std::vector<float> h(size_t(B) * S * F);
            for (int32_t b = 0; b < B; ++b) {
                for (int32_t s = 0; s < S; ++s) {
                    for (int32_t i = 0; i < F; ++i) {
                        h[(size_t(b) * S + s) * F + i] = fx.in0[B][(size_t(b) * S + s) * D + r * F + i];
                    }
                }
            }
            c.in0[B] = f32(h.size(), "in0");
            ok = ok && put(c.in0[B], h, err, "in0");
        }
        c.zero = f32(size_t(max_rows) * F, "zero");
        c.runs = P.get<uint32_t>(size_t(fn::hc::n_runs_card) * 4, err, "hc run counters");
        ok = ok && c.runs != nullptr &&
             cuda_ok(cudaMemset(c.runs, 0, size_t(fn::hc::n_runs_card) * 4), "hc run counters", err);
        c.partial = f32(size_t(max_rows) * D, "partial");
        c.routed = f32(size_t(max_rows) * D, "routed");
        // the stand-ins' weights: hashed words, their XOR kept for the sink check
        for (int32_t p = 0; p < n_piece && ok; ++p) {
            int64_t nb = piece_weight_bytes(static_cast<piece>(p), r);
            if (cfg.dense && dense_rest(static_cast<piece>(p))) {
                nb = 0;
                for (const std::string & t : rest_tensors(static_cast<piece>(p))) {
                    const int64_t b = tensor_bytes(t.c_str(), r);
                    nb += b > 0 ? b : tensor_bytes(t.c_str(), 1 - r);
                }
                nb = (nb + 15) / 16 * 16; // the stand-in streams 16-byte words
            }
            if (nb < 0 || nb % 16 != 0) {
                return fail(err, std::string("weight bytes of ") + piece_table[p].name);
            }
            c.wb[p] = nb;
            uint32_t xr[4] = {0u, 0u, 0u, 0u};
            if (nb > 0) {
                std::vector<uint32_t> h(static_cast<size_t>(nb / 4));
                for (size_t i = 0; i < h.size(); ++i) {
                    h[i] = static_cast<uint32_t>(mix64((uint64_t(r) << 40) ^ (uint64_t(p) << 32) ^ i));
                    xr[i % 4] ^= h[i];
                }
                c.w[p] = P.get<uint8_t>(static_cast<size_t>(nb), err, "stand-in weights");
                ok = ok && c.w[p] != nullptr && put(static_cast<uint32_t *>(c.w[p]), h, err, "stand-in weights");
            }
            std::memcpy(fx.w_xor[p][r], xr, sizeof(xr));
        }
        c.sink = P.get<uint32_t>(size_t(n_piece) * sink_words() * 4, err, "sinks");
        c.proj_out = f32(size_t(max_rows) * proj_width, "proj_out");
        c.recur_out = f32(size_t(max_rows) * 3072, "recur_out");
        c.qkv_out = f32(size_t(max_rows) * 6144, "qkv_out");
        c.attn_out = f32(size_t(max_rows) * 3072, "attn_out");
        c.gdn_state = f32(size_t(max_rows + 1) * gdn_state_floats, "gdn state");
        c.conv_hist = f32(size_t(max_rows + 1) * conv_floats, "conv history");
        if (cfg.gdn != 0) {
            c.gd_qkv = f32(size_t(max_rows) * 5120, "gdn qkv");
            c.gd_z = f32(size_t(max_rows) * 3072, "gdn z");
            c.gd_ab = f32(size_t(2) * max_rows * 24, "gdn alpha, beta");
            c.gd_w = f32(size_t(5120) * 4 + 24 + 24 + 128, "gdn weights");
            c.gd_codes = P.get<int8_t>(size_t(max_rows) * 3072, err, "gdn codes");
            c.gd_scales = f32(size_t(max_rows) * 96 * 2, "gdn scales");
            c.gd_ws = P.get<uint8_t>(fn::gdn::workspace_bytes(), err, "gdn workspace");
            c.gd_cur = P.get<uint32_t>(4, err, "gdn committed slot");
            std::vector<float> z(size_t(max_rows) * 3072), ab(size_t(2) * max_rows * 24), w(size_t(5120) * 4 + 176);
            for (size_t i = 0; i < z.size(); ++i) {
                z[i] = hash_unit(0x6d1 + r, i, 3.0f);
            }
            for (size_t i = 0; i < ab.size(); ++i) {
                ab[i] = hash_unit(0x6d2 + r, i, 2.0f);
            }
            for (size_t i = 0; i < w.size(); ++i) {
                w[i] = hash_unit(0x6d3 + r, i, 0.5f);
            }
            for (size_t i = 0; i < 24; ++i) {
                w[5120 * 4 + 24 + i] = -0.05f - std::fabs(w[5120 * 4 + 24 + i]); // a = -exp(A_log) < 0
            }
            ok = ok && c.gd_codes != nullptr && c.gd_ws != nullptr && put(c.gd_z, z, err, "gdn z") &&
                 put(c.gd_ab, ab, err, "gdn alpha, beta") && put(c.gd_w, w, err, "gdn weights") &&
                 cudaMemset(c.gd_ws, 0, fn::gdn::workspace_bytes()) == cudaSuccess && c.gd_cur != nullptr &&
                 cudaMemset(c.gd_cur, 0, 4) == cudaSuccess;
        }
        if (cfg.qsa != 0) {
            c.qa_norm = f32(size_t(512), "qsa norms");
            c.qa_pos = P.get<int32_t>(size_t(4) * (1 + 2 + 3 + 4) * 4, err, "qsa positions");
            c.qa_ws = P.get<uint8_t>(fn::qsa::attn::workspace_bytes(), err, "qsa attend workspace");
            std::vector<float> nw(512);
            for (size_t i = 0; i < nw.size(); ++i) {
                nw[i] = 1.0f + hash_unit(0x6e1 + r, i, 0.5f);
            }
            std::vector<int32_t> pos;
            for (int32_t b = 1; b <= max_rows; ++b) {
                for (int32_t pl = 0; pl < 4; ++pl) {
                    for (int32_t t = 0; t < b; ++t) {
                        pos.push_back(pair_depth + t);
                    }
                }
            }
            ok = ok && c.qa_norm != nullptr && c.qa_pos != nullptr && c.qa_ws != nullptr &&
                 put(c.qa_norm, nw, err, "qsa norms") && put(c.qa_pos, pos, err, "qsa positions");
        }
        {
            std::vector<float> h(static_cast<size_t>(gdn_state_floats));
            for (size_t i = 0; i < h.size(); ++i) {
                h[i] = hash_unit(0x6d0 + r, i, 0.1f);
            }
            std::vector<float> hc(static_cast<size_t>(conv_floats));
            for (size_t i = 0; i < hc.size(); ++i) {
                hc[i] = hash_unit(0xc0 + r, i, 1.0f);
            }
            ok = ok && put(c.gdn_state, h, err, "gdn state") && put(c.conv_hist, hc, err, "conv history");
        }
        // KV of this card's head and the index history: synthetic keys for cells 0 .. depth - 1
        c.k_cache = P.get<uint16_t>(size_t(pair_cells) * 256 * 2, err, "k cache");
        c.v_cache = P.get<uint16_t>(size_t(pair_cells) * 256 * 2, err, "v cache");
        c.keys = P.get<uint16_t>(size_t(pair_cells) * 128 * 2, err, "index keys");
        c.summ = f32(fn::qsa::summary_bytes(pair_cells / 4) / 4, "summaries");
        ok = ok && c.k_cache != nullptr && c.v_cache != nullptr && c.keys != nullptr;
        if (ok) {
            std::vector<uint16_t> kv(size_t(pair_cells) * 256, 0);
            for (int32_t i = 0; i < 2 && ok; ++i) {
                for (size_t e = 0; e < size_t(pair_depth) * 256; ++e) {
                    kv[e] = to_bf16(hash_unit(0x4b56 + 2 * r + i, e, 1.0f));
                }
                ok = put(i == 0 ? c.k_cache : c.v_cache, kv, err, "kv");
            }
            // the index keys are the same on both cards (one history), N(0, 1)-like values
            std::vector<uint16_t> ks(size_t(pair_cells) * 128, 0);
            for (size_t e = 0; e < size_t(pair_depth) * 128; ++e) {
                ks[e] = to_bf16(hash_unit(0x1d8, e, 1.7f));
            }
            ok = ok && put(c.keys, ks, err, "index keys");
        }
        c.ik = f32(size_t(max_rows) * 128, "ik");
        c.iq = f32(size_t(max_rows) * 512, "iq");
        c.q_norm = f32(128, "q_norm");
        c.k_norm = f32(128, "k_norm");
        {
            std::vector<float> qn(128), kn(128);
            for (int32_t i = 0; i < 128; ++i) {
                qn[i] = 1.0f + hash_unit(0x91, uint64_t(i), 0.5f);
                kn[i] = 1.0f + hash_unit(0x92, uint64_t(i), 0.5f);
            }
            ok = ok && put(c.q_norm, qn, err, "q_norm") && put(c.k_norm, kn, err, "k_norm");
        }
        c.qsa_ws = P.get<uint8_t>(fn::qsa::workspace_bytes(), err, "qsa workspace");
        c.body = P.get<uint32_t>(size_t(max_rows) * BW * 4, err, "body");
        c.off = P.get<int32_t>(32, err, "offsets");
        c.sel_body = P.get<uint32_t>(size_t(max_rows) * BW * 4, err, "sel body");
        c.sel_off = P.get<int32_t>(32, err, "sel offsets");
        c.sel2_body = P.get<uint32_t>(size_t(max_rows) * BW * 4, err, "sel2 body");
        c.sel2_off = P.get<int32_t>(32, err, "sel2 offsets");
        c.ids_full = P.get<int32_t>(size_t(max_rows) * 10 * 4, err, "ids_full");
        c.ids = P.get<int32_t>(size_t(max_rows) * 10 * 4, err, "ids");
        c.meta = P.get<uint8_t>(fn::experts::meta_bytes, err, "routing meta");
        c.wts = f32(size_t(max_rows) * 10, "wts");
        c.sg = f32(16, "sg");
        c.xc = P.get<int8_t>(fn::act::codes_bytes(max_rows, D), err, "x codes");
        c.xs = P.get<float>(fn::act::scales_bytes(max_rows, D), err, "x scales");
        c.gluc = P.get<int8_t>(fn::act::codes_bytes(GS, L), err, "glu codes");
        c.glus = P.get<float>(fn::act::scales_bytes(GS, L), err, "glu scales");
        c.glu_f32 = f32(size_t(fn::experts::glu_slots) * fn::experts::n_ff_card, "glu_f32");
        c.status = P.get<uint32_t>(16, err, "status");
        c.sh_glu = f32(size_t(max_rows) * 320, "sh_glu");
        c.sh_part = f32(size_t(max_rows) * D, "sh_part");
        c.grids = P.get<uint8_t>(fn::experts::grid_bytes, err, "grids");
        ok = ok && c.grids != nullptr && put(c.grids, grids, err, "grids");
        for (int32_t l = 0; l < 2 && ok; ++l) {
            std::vector<uint8_t> h;
            ok = fn::experts::pack_card(m.syn_gu[l], r, fx.map, 1, h) || fail(err, "pack gate_up");
            c.ex_gu[l] = ok ? P.get<uint8_t>(h.size(), err, "experts gate_up") : nullptr;
            ok = ok && c.ex_gu[l] != nullptr && put(c.ex_gu[l], h, err, "experts gate_up");
            ok = ok && (fn::experts::pack_card(m.syn_dn, r, fx.map, 1, h) || fail(err, "pack down"));
            c.ex_dn[l] = ok ? P.get<uint8_t>(h.size(), err, "experts down") : nullptr;
            ok = ok && c.ex_dn[l] != nullptr && put(c.ex_dn[l], h, err, "experts down");
            // EX2's shared expert: card r's 320 gate and up rows (DP2's pack), its down's 2,560 rows of its 320 columns
            const int32_t n_sr = fn::dense::n_synth_rows;
            ok = ok && (fn::dense::pack_synth(m.sh_g[l], L, fx.sh_off[l][r][0], n_sr, h) || fail(err, "pack sh gate"));
            c.ex_sg[l] = ok ? P.get<uint8_t>(h.size(), err, "shared gate") : nullptr;
            ok = ok && c.ex_sg[l] != nullptr && put(c.ex_sg[l], h, err, "shared gate");
            ok = ok && (fn::dense::pack_synth(m.sh_u[l], L, fx.sh_off[l][r][1], n_sr, h) || fail(err, "pack sh up"));
            c.ex_su[l] = ok ? P.get<uint8_t>(h.size(), err, "shared up") : nullptr;
            ok = ok && c.ex_su[l] != nullptr && put(c.ex_su[l], h, err, "shared up");
            if (ok) {
                std::vector<uint8_t> rows(m.sh_d.row_bytes * D);
                for (int32_t n = 0; n < D; ++n) {
                    std::memcpy(rows.data() + m.sh_d.row_bytes * n, m.sh_d.row((n + fx.sh_off[l][r][2]) % n_sr),
                                m.sh_d.row_bytes);
                }
                h.assign(size_t(fn::experts::down_bytes(fn::experts::fmt::iq4_nl)), 0);
                ok = fn::experts::pack_down(fn::experts::fmt::iq4_nl, rows.data(), m.sh_d.row_bytes, h.data()) ||
                     fail(err, "pack sh down");
            }
            c.ex_sd[l] = ok ? P.get<uint8_t>(h.size(), err, "shared down") : nullptr;
            ok = ok && c.ex_sd[l] != nullptr && put(c.ex_sd[l], h, err, "shared down");
        }
        if (cfg.dense) {
            ok = ok && init_dense(c, r == index_owner, dsyn, dgrids, err);
        }
        c.stamp = P.get<uint64_t>(64 * 8, err, "stamp");
        c.counter = P.get<int32_t>(16, err, "counter");
        c.decode = P.get<int32_t>(16, err, "decode");
        c.flags = P.get<uint32_t>(16, err, "flags");
        c.pings = P.get<uint64_t>(size_t(2) * 64 * 8, err, "pings");
        c.flush = P.get<uint8_t>(static_cast<size_t>(flush_bytes), err, "flush");
        if (fx.hx7.n_decode > 0) {
            for (int32_t l = 0; l < 2 && ok; ++l) {
                c.routes[l] = P.get<int32_t>(fx.hx7.routes[l].size() * 4, err, "routes");
                ok = c.routes[l] != nullptr && put(c.routes[l], fx.hx7.routes[l], err, "routes");
            }
            c.cells = P.get<int32_t>(fx.hx7.cells.size() * 4, err, "cells");
            ok = ok && c.cells != nullptr && put(c.cells, fx.hx7.cells, err, "cells");
        }
        if (!ok || c.sink == nullptr || c.qsa_ws == nullptr || c.flush == nullptr || c.body == nullptr ||
            c.sel_body == nullptr || c.sel2_body == nullptr || c.xc == nullptr || c.xs == nullptr ||
            c.gluc == nullptr || c.glus == nullptr) {
            return err.empty() ? fail(err, "allocation") : false;
        }
        // the depth on the device (TG1)
        c.d_depth = P.get<int32_t>(16, err, "depth");
        if (c.d_depth == nullptr || !put(c.d_depth, std::vector<int32_t>{pair_depth}, err, "depth")) {
            return err.empty() ? fail(err, "depth") : false;
        }
        // the summaries of the history's complete blocks (QA1b)
        fn::qsa::update_args ua;
        ua.keys = c.keys;
        ua.k_norm = c.k_norm;
        ua.summaries = c.summ;
        ua.first = 0;
        ua.count = pair_depth / 4;
        ua.rope = fn::qsa::model_rope();
        if (!fn::qsa::summary_update(ua, c.st) || !cuda_ok(cudaStreamSynchronize(c.st), "summaries", err)) {
            return err.empty() ? fail(err, "summary_update") : false;
        }
        if (!ok || !cuda_ok(cudaDeviceSynchronize(), "setup", err)) {
            return false;
        }
    }
    return reset_exchange(err) && reset_round(err) && reset_state(err);
}

bool pair::sync(std::string & err) {
    for (int32_t k = 0; k < 2; ++k) {
        if (!cuda_ok(cudaSetDevice(m_->c[k].dev), "device", err) ||
            !cuda_ok(cudaStreamSynchronize(m_->c[k].st), ("card " + std::to_string(k)).c_str(), err)) {
            return false;
        }
    }
    return true;
}

bool pair::reset_exchange(std::string & err) {
    if (!sync(err) || !m_->rg->reset_all()) {
        return err.empty() ? fail(err, "rings reset") : false;
    }
    m_->u = use_counts();
    return true;
}

bool pair::reset_round(std::string & err) {
    for (int32_t k = 0; k < 2; ++k) {
        if (!cuda_ok(cudaSetDevice(m_->c[k].dev), "device", err) ||
            !cuda_ok(cudaMemsetAsync(m_->c[k].counter, 0, 16, m_->c[k].st), "counter", err)) {
            return false;
        }
    }
    return sync(err);
}

bool pair::set_round(int32_t r, std::string & err) {
    for (int32_t k = 0; k < 2; ++k) {
        if (!cuda_ok(cudaSetDevice(m_->c[k].dev), "device", err) ||
            !cuda_ok(cudaMemcpyAsync(m_->c[k].counter, &r, 4, cudaMemcpyHostToDevice, m_->c[k].st), "counter", err) ||
            !cuda_ok(cudaStreamSynchronize(m_->c[k].st), "counter", err)) {
            return false;
        }
    }
    return sync(err);
}

bool pair::read_final(int32_t rows, card_record * const rec[2], std::string & err) {
    if (!sync(err)) {
        return false;
    }
    for (int32_t r = 0; r < 2; ++r) {
        const card & c = m_->c[r];
        card_record & R = *rec[r];
        if (!cuda_ok(cudaSetDevice(c.dev), "device", err) ||
            !cuda_ok(cudaMemcpy(R.x[3], c.x[1], size_t(rows) * S * F * 4, cudaMemcpyDeviceToHost), "x", err) ||
            !cuda_ok(cudaMemcpy(R.inject[3], c.inject[1], size_t(rows) * S * 4, cudaMemcpyDeviceToHost), "inject",
                     err) ||
            !cuda_ok(cudaMemcpy(R.out[3], c.out[1], size_t(rows) * F * 4, cudaMemcpyDeviceToHost), "out", err) ||
            !cuda_ok(cudaMemcpy(R.body, r == index_owner ? c.body : c.sel_body, size_t(rows) * BW * 4,
                                cudaMemcpyDeviceToHost),
                     "body", err) ||
            !cuda_ok(cudaMemcpy(R.off, r == index_owner ? c.off : c.sel_off, size_t(rows + 1) * 4,
                                cudaMemcpyDeviceToHost),
                     "offsets", err) ||
            !cuda_ok(cudaMemcpy(&R.err, m_->rg->err(r), 4, cudaMemcpyDeviceToHost), "err", err)) {
            return false;
        }
    }
    return true;
}

bool pair::reset_state(std::string & err) {
    if (!sync(err)) {
        return false;
    }
    for (int32_t r = 0; r < 2; ++r) {
        card & c = m_->c[r];
        const size_t tail = size_t(pair_cells - pair_depth);
        bool ok = cuda_ok(cudaSetDevice(c.dev), "device", err) &&
                  cuda_ok(cudaMemset(c.k_cache + size_t(pair_depth) * 256, 0, tail * 512), "k tail", err) &&
                  cuda_ok(cudaMemset(c.v_cache + size_t(pair_depth) * 256, 0, tail * 512), "v tail", err) &&
                  cuda_ok(cudaMemset(c.keys + size_t(pair_depth) * 128, 0, tail * 256), "keys tail", err);
        ok = ok && cuda_ok(cudaDeviceSynchronize(), "reset", err);
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool pair::piece_args(piece p, int32_t r, int32_t rows, standin_args & a) const {
    const card & c = m_->c[r];
    const facts & fx = m_->fx;
    const int32_t id = static_cast<int32_t>(p);
    a = standin_args();
    a.rows = rows;
    a.c = fx.c_piece[id];
    a.w = c.w[id];
    a.w_bytes = c.wb[id];
    a.sink = c.sink + size_t(id) * sink_words();
    a.pad = c.pad[id];
    const bool hx7 = cfg_.tr == traffic::hx7 && fx.hx7.n_decode > 0;
    switch (p) {
        case piece::gdn_proj_conv:
            a.in = c.full[0], a.in_width = D, a.out = c.proj_out, a.out_width = proj_width;
            a.st = c.conv_hist, a.st_bytes = int64_t(conv_floats) * 4, a.st_dst = c.conv_hist + conv_floats;
            a.st_copies = rows;
            if (cfg_.gdn != 0) {
                // GD1: the products only (qkv, z, alpha, beta): gdn_core reads the history and conv1d, a, dt
                a.out = c.gd_qkv, a.out_width = 5120;
                a.st = nullptr, a.st_bytes = 0, a.st_dst = nullptr, a.st_copies = 0;
                for (const char * t : {"blk.10.ssm_conv1d.weight", "blk.10.ssm_a", "blk.10.ssm_dt.bias"}) {
                    a.w_bytes -= tensor_bytes(t, r);
                }
            }
            break;
        case piece::gdn_recur:
            a.in = c.proj_out, a.in_width = proj_width, a.out = c.recur_out, a.out_width = 3072;
            a.st = c.gdn_state, a.st_bytes = int64_t(gdn_state_floats) * 4, a.st_dst = c.gdn_state + gdn_state_floats;
            a.st_copies = rows;
            break;
        case piece::gdn_out: a.in = c.recur_out, a.in_width = 3072, a.out = c.partial, a.out_width = D; break;
        case piece::index_q_proj: a.in = c.full[0], a.in_width = D, a.out = c.iq, a.out_width = 512; break;
        case piece::index_k_proj:
            a.in = c.full[0], a.in_width = D, a.out = c.ik, a.out_width = 128;
            a.bf[0] = c.keys, a.bf_c[0] = a.c, a.bf_width = 128, a.bf_cell0 = pair_depth, a.bf_lo = 0, a.bf_hi = rows;
            break;
        case piece::qsa_qkv:
            a.in = c.full[0], a.in_width = D, a.out = c.qkv_out, a.out_width = 6144;
            a.bf[0] = c.k_cache, a.bf[1] = c.v_cache, a.bf_c[0] = a.c + 0.015625f, a.bf_c[1] = a.c + 0.03125f;
            a.bf_width = 256, a.bf_cell0 = pair_depth, a.bf_lo = 0, a.bf_hi = rows;
            break;
        case piece::qsa_attention:
            a.in = c.qkv_out, a.in_width = 6144, a.out = c.attn_out, a.out_width = 3072;
            a.kv_k = c.k_cache, a.kv_v = c.v_cache, a.cell_width = W, a.cell_stride = W;
            if (hx7) {
                a.cells = c.cells, a.cells_decode_stride = int64_t(hx7_rows) * W, a.decode = c.decode;
            } else if (r == index_owner) {
                a.body = c.body, a.body_off = c.off; // QA1b's own selection
            } else {
                a.body = c.sel_body, a.body_off = c.sel_off; // the received X-SELECT
            }
            break;
        case piece::qsa_out: a.in = c.attn_out, a.in_width = 3072, a.out = c.partial, a.out_width = D; break;
        case piece::router10:
        case piece::router11:
            a.in = c.full[1], a.in_width = D, a.ids = c.ids_full, a.wts = c.wts, a.sg = c.sg;
            a.layer = p == piece::router10 ? 10 : 11;
            if (hx7) {
                a.route_src = c.routes[a.layer - 10], a.decode = c.decode;
            }
            break;
        case piece::sh_gate_up10:
        case piece::sh_gate_up11: a.in = c.full[1], a.in_width = D, a.out = c.sh_glu, a.out_width = 320; break;
        case piece::sh_down10:
        case piece::sh_down11:
            a.in = c.sh_glu, a.in_width = 320, a.out = c.sh_part, a.out_width = D, a.aux = c.sg, a.aux_n = rows;
            break;
    }
    return true;
}

bool pair::piece_op(piece p, int32_t r, int32_t rows, op & o, std::string & err) {
    standin_args a;
    if (r < 0 || r > 1 || !piece_args(p, r, rows, a)) {
        return fail(err, "piece");
    }
    const int32_t id = static_cast<int32_t>(p);
    o.card = r;
    o.layer = piece_table[id].layer;
    o.piece = piece_table[id].name;
    o.role.clear();
    o.launches = 1;
    o.fn = [p, a](void * s) { return standin(p, a, s); };
    return true;
}

bool pair::build(variant v, int32_t rows, bool graph, std::vector<op> & prog, std::string & err,
                 card_record * const rec[2]) {
    prog.clear();
    impl & m = *m_;
    const int32_t B = rows;
    const int32_t depth = pair_depth;
    if (B < 1 || B > max_rows || (graph && rec != nullptr)) {
        return fail(err, "rows, or records in a graph");
    }
    const bool dense = cfg_.dense && v == variant::v2; // DP2's launches (no producer sends, as V0)
    const bool send_in_producer = v != variant::v0 && !dense;
    const bool router_both = v == variant::v2;
    const bool hx7 = cfg_.tr == traffic::hx7 && m.fx.hx7.n_decode > 0;
    if (cfg_.dense && hx7 && B > hx7_rows) {
        return fail(err, "dense mode with HX7 traffic takes rows 1..3");
    }
    const bool glu_f32 = cfg_.glu_f32;
    x::rings & rg = *m.rg;
    use_counts & u = m.u;
    bool ok = true;
    const auto add = [&](int32_t r, int32_t layer, const std::string & piece_name, const std::string & role,
                         int32_t launches, std::function<bool(void *)> fn) {
        prog.push_back({r, layer, piece_name, role, launches, std::move(fn)});
    };
    const auto stand = [&](piece p, int32_t r, const std::string & role,
                           const std::function<void(standin_args &)> & edit) {
        standin_args a;
        ok = ok && piece_args(p, r, B, a);
        if (edit) {
            edit(a);
        }
        const int32_t id = static_cast<int32_t>(p);
        add(r, piece_table[id].layer, piece_table[id].name, role, 1, [p, a](void * st) { return standin(p, a, st); });
    };
    const auto rec_copy = [&](int32_t r, void * dst, const void * src, size_t bytes) {
        if (rec != nullptr) {
            add(r, 0, "record", "", 0, [dst, src, bytes](void * st) {
                return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, cs(st)) == cudaSuccess;
            });
        }
    };

    // DP2's launch with its inputs: codes and scales for the quantized matrices, F32 for BF16
    const auto dense_op = [&](int32_t r, int32_t layer, const char * name, fn::dense::product_args a,
                              const int8_t * codes, const float * scales, const float * f32) {
        bool any_q = false, any_bf = false;
        for (int32_t i = 0; i < a.n_mats; ++i) {
            any_q = any_q || a.mats[i].f != fn::dense::fmt::bf16;
            any_bf = any_bf || a.mats[i].f == fn::dense::fmt::bf16;
        }
        a.x_codes = any_q ? codes : nullptr;
        a.x_scales = any_q ? scales : nullptr;
        a.x_f32 = any_bf ? f32 : nullptr;
        add(r, layer, name, "", 1, [a, B](void * st) { return fn::dense::product(B, a, st); });
    };
    // C2's quantize: the stand-in for the producer's epilogue writing FNFL's format
    const auto quant_op = [&](int32_t r, int32_t layer, const char * name, const float * xin, int8_t * codes,
                              float * scales, int32_t k) {
        const fn::act::out_view xo = {codes, scales, B, k};
        add(r, layer, name, "", 1, [xin, xo](void * st) { return fn::act::quantize(xin, xo, st); });
    };

    for (int32_t r = 0; r < 2; ++r) {
        card * c = &m.c[r];
        const int32_t nd = hx7 ? m.fx.hx7.n_decode : 1;
        if (cfg_.dense && hx7) { // EX1b reads HX7's IDs; DP2's router (V2) writes its own
            add(r, 10, "round", "", 1, [c, nd, B](void * st) {
                const int32_t * const src[2] = {c->routes[0], c->routes[1]};
                int32_t * const dst[2] = {c->ids_hx7[0], c->ids_hx7[1]};
                return round_start_routes(c->stamp, c->counter, c->decode, nd, src, dst, B, st);
            });
        } else {
            add(r, 10, "round", "", 1,
                [c, nd](void * st) { return round_start(c->stamp, c->counter, c->decode, nd, st); });
        }
    }
    const int32_t nb0 = depth / 4, nb1 = (depth + B) / 4; // blocks complete before and after the rows
    for (int32_t k = 0; k < 4 && ok; ++k) {
        const int32_t layer = k < 2 ? 10 : 11;
        const int32_t set = k % 2;
        const bool ffn = set == 1;
        const int32_t s = graph ? hc_slot_graph(k) : static_cast<int32_t>((u.sub + k) % x::n_slot);
        const int32_t s_prev = graph ? hc_slot_graph((k + 3) % 4)
                                     : (u.sub + k == 0 ? -1 : static_cast<int32_t>((u.sub + k - 1) % x::n_slot));
        // the other messages' slots: by position in the graph, else by running count
        const int32_t s_half = graph ? k / 2 : static_cast<int32_t>(((ffn ? u.f : u.a) + k / 2) % x::n_slot);
        const int32_t s_route = graph ? k / 2 : static_cast<int32_t>((u.route + k / 2) % x::n_slot);
        const int32_t s_sel = graph ? 0 : static_cast<int32_t>(u.select % x::n_slot);
        const x::msg half_msg = ffn ? x::msg::f : x::msg::a;
        const uint32_t half_code = ffn ? x::err_poll_f : x::err_poll_a;
        const std::string half_name = ffn ? "X-F" : "X-A";
        // V1+: the producer sends the peer's half
        const auto send_edit = [&](int32_t r) -> std::function<void(standin_args &)> {
            return [&rg, r, half_msg, s_half](standin_args & a) {
                a.peer_slot = rg.slot(1 - r, half_msg, s_half);
                a.peer_card = 1 - r;
            };
        };

        // HC2: down2, up2 and ready on both cards
        for (int32_t ph = 0; ph < 3; ++ph) {
            for (int32_t r = 0; r < 2; ++r) {
                card & c = m.c[r];
                const float * hres = k == 0 ? c.in0[B] : c.x[1 - set];
                const float * hpost = k == 0 ? c.zero : c.inject[1 - set];
                const float * hx = k == 0 ? c.zero : c.out[1 - set];
                if (ph == 0) {
                    fn::hc::down2_args d2;
                    d2.d = {c.w_down[k], hres, hx, hpost, c.gamma[k], c.x[set], c.mom[set]};
                    d2.hc_peer = rg.slot(1 - r, x::msg::hc, s);
                    d2.hc_reset = s_prev >= 0 ? rg.slot(r, x::msg::hc, s_prev) : nullptr;
                    add(r, layer, "HC", "send X-HC", 1, [d2, B](void * st) { return fn::hc::down2(B, d2, st); });
                } else if (ph == 1) {
                    const fn::hc::up2_args u2 = {c.w_up[k],
                                                 c.mom[set],
                                                 rg.slot(r, x::msg::hc, s),
                                                 c.x[set],
                                                 c.gamma[k],
                                                 c.inject[set],
                                                 rg.slot(r, x::msg::hmix, s),
                                                 rg.slot(1 - r, x::msg::hmix, s),
                                                 c.runs,
                                                 rg.err(r),
                                                 r};
                    add(r, layer, "HC", "recv X-HC; send X-HMIX", 1,
                        [u2, B](void * st) { return fn::hc::up2(fn::hc::up_arm::ring, B, u2, st); });
                } else {
                    // the stand-in consumer (HC3): the full F32 row from X-HMIX, the peer half reset as read
                    const fn::hc::ready_args ra = {
                        rg.slot(r, x::msg::hmix, s), c.full[set], nullptr, nullptr, rg.err(r), r, 1};
                    add(r, layer, "HC", "recv X-HMIX", 1, [ra, B](void * st) { return fn::hc::ready(B, ra, st); });
                }
            }
        }

        if (k == 0) {
            // GDN (layer 10): no exchange inside the body
            for (int32_t r = 0; r < 2; ++r) {
                card & c = m.c[r];
                if (dense) {
                    quant_op(r, 10, "act quantize GDN input", c.full[0], c.d_xc, c.d_xs, D);
                    dense_op(r, 10, "dense GDN input", c.d_gdn, c.d_xc, c.d_xs, c.full[0]);
                    float * qkv = c.d_qkv;
                    stand(piece::gdn_proj_conv, r, "", [qkv](standin_args & a) {
                        a.in = qkv;
                        a.in_width = 5120;
                    });
                } else {
                    stand(piece::gdn_proj_conv, r, "", nullptr);
                }
                if (cfg_.gdn == 0) {
                    stand(piece::gdn_recur, r, "", nullptr);
                } else if (cfg_.gdn == 1) {
                    card & cr = m.c[r];
                    fn::gdn::core_args g;
                    // the projections' stand-in output stands in for the epilogue's post-SiLU values (Amendment
                    // 2b: the convolution is the producer's epilogue, priced in LP1)
                    g.conv = cr.gd_qkv;
                    g.z = cr.gd_z;
                    g.alpha = cr.gd_ab;
                    g.beta = cr.gd_ab + std::ptrdiff_t{max_rows} * 24;
                    g.dt = cr.gd_w + std::ptrdiff_t{5120} * 4;
                    g.a = g.dt + 24;
                    g.norm_w = g.dt + 48;
                    g.state = cr.gdn_state;
                    g.cur = cr.gd_cur;
                    g.gated = cr.recur_out;
                    g.gated_q = {cr.gd_codes, cr.gd_scales, B, 3072};
                    g.ws = cr.gd_ws;
                    g.rows = B;
                    add(r, 10, "gdn_core", "", 1, [g](void * st) { return fn::gdn::core(g, st); });
                }
                rec_copy(r, rec != nullptr ? rec[r]->recur_out : nullptr, c.recur_out, size_t(B) * 3072 * 4);
                if (dense) {
                    quant_op(r, 10, "act quantize ssm_out", c.recur_out, c.d_mc, c.d_ms, 3072);
                    dense_op(r, 10, "dense ssm_out", c.d_ssm, c.d_mc, c.d_ms, nullptr);
                } else if (send_in_producer) {
                    stand(piece::gdn_out, r, "send X-A", send_edit(r));
                } else {
                    stand(piece::gdn_out, r, "", nullptr);
                }
            }
        } else if (k == 2) {
            // QSA (layer 11), S2's indexer on its owner; X-SELECT carries QA1b's body (C2)
            const int32_t o = index_owner, p = 1 - o;
            card & co = m.c[o];
            card & cp = m.c[p];
            const int32_t bw = fn::qsa::body_words(depth, B);
            // dense: DP2's QSA input with the index query and key, then the rest of each stand-in's piece
            const auto qsa_input = [&](int32_t r) {
                card & c = m.c[r];
                quant_op(r, 11, "act quantize QSA input", c.full[0], c.d_xc, c.d_xs, D);
                dense_op(r, 11, "dense QSA input", c.d_qsa, c.d_xc, c.d_xs, c.full[0]);
            };
            const auto qkv_rest = [&](int32_t r) {
                float * q = m.c[r].d_q;
                stand(piece::qsa_qkv, r, "", [q](standin_args & a) {
                    a.in = q;
                    a.in_width = 6144;
                });
            };
            const auto attn_output = [&](int32_t r) {
                card & c = m.c[r];
                if (cfg_.qsa == 0) {
                    quant_op(r, 11, "act quantize attn_output", c.attn_out, c.d_mc, c.d_ms, 3072);
                }
                dense_op(r, 11, "dense attn_output", c.d_att, c.d_mc, c.d_ms, nullptr);
            };
            // QA2: attend in place of the stand-in (or neither, the knock-out)
            const auto attention = [&](int32_t r, const uint32_t * body, const int32_t * off) {
                card & c = m.c[r];
                if (cfg_.qsa == 0) {
                    stand(piece::qsa_attention, r, "", nullptr);
                    return;
                }
                if (cfg_.qsa == 2) {
                    return;
                }
                fn::qsa::attn::args aa = {};
                aa.qfull = c.d_q;
                aa.kv_k = c.d_k;
                aa.kv_v = c.d_v;
                aa.q_norm = c.qa_norm;
                aa.k_norm = c.qa_norm + 256;
                aa.body = body;
                aa.offsets = off;
                aa.pos = c.qa_pos + std::ptrdiff_t{2} * (B - 1) * B;
                aa.k_cache = c.k_cache;
                aa.v_cache = c.v_cache;
                aa.depth = c.d_depth;
                aa.max_cells = depth + max_rows;
                aa.theta_scale = fn::qsa::attn::rope_theta_scale();
                aa.out = c.attn_out;
                aa.codes = c.d_mc;
                aa.scales = c.d_ms;
                aa.ws = fn::qsa::attn::carve(c.qa_ws);
                add(r, 11, "QSA attend", "", 1, [aa, B](void * st) { return fn::qsa::attn::attend(B, aa, st); });
            };
            if (dense) {
                qsa_input(o);
                float * ik = co.d_ik;
                stand(piece::index_k_proj, o, "", [ik](standin_args & a) {
                    a.in = ik;
                    a.in_width = 128;
                });
            } else {
                stand(piece::index_q_proj, o, "", nullptr);
                stand(piece::index_k_proj, o, "", nullptr);
            }
            if (dense && cfg_.qsa == 1) {
                // QA2: the raw index-key write in summary_update's launch, every round
                fn::qsa::round_update_args ua;
                ua.keys = co.keys;
                ua.k_norm = co.k_norm;
                ua.summaries = co.summ;
                ua.rope = fn::qsa::model_rope();
                ua.ik = co.d_ik;
                ua.depth = co.d_depth;
                ua.rows = B;
                ua.max_cells = depth + max_rows;
                add(o, 11, "QSA index build", "", 1, [ua](void * st) { return fn::qsa::summary_update(ua, st); });
            } else if (nb1 > nb0) {
                fn::qsa::update_args ua;
                ua.keys = co.keys;
                ua.k_norm = co.k_norm;
                ua.summaries = co.summ;
                ua.first = nb0;
                ua.count = nb1 - nb0;
                ua.rope = fn::qsa::model_rope();
                add(o, 11, "QSA index build", "", 1, [ua](void * st) { return fn::qsa::summary_update(ua, st); });
            }
            fn::qsa::select_args sa;
            sa.summaries = co.summ;
            sa.q = dense ? co.d_iq : co.iq;
            sa.q_norm = co.q_norm;
            sa.depth = co.d_depth;
            sa.max_cells = depth + B;
            sa.rope = fn::qsa::model_rope();
            sa.ws = fn::qsa::carve(co.qsa_ws);
            sa.body = co.body;
            sa.offsets = co.off;
            add(o, 11, "QSA select", "", 2, [sa, B](void * st) { return fn::qsa::select(B, sa, st); });
            {
                const int32_t * off = co.off;
                const uint32_t * body = co.body;
                float * slot = rg.slot(p, x::msg::select, s_sel);
                add(o, 11, "X-SELECT send", "send X-SELECT", 1,
                    [B, off, body, slot](void * st) { return x::send_select(B, off, body, slot, st); });
            }
            if (dense) {
                qkv_rest(o);
            } else {
                stand(piece::qsa_qkv, o, "", nullptr);
            }
            if (dense) {
                attention(o, co.body, co.off);
            } else {
                stand(piece::qsa_attention, o, "", nullptr);
            }
            rec_copy(o, rec != nullptr ? rec[o]->attn_out : nullptr, co.attn_out, size_t(B) * 3072 * 4);
            rec_copy(o, rec != nullptr ? rec[o]->body : nullptr, co.body, size_t(bw) * 4);
            rec_copy(o, rec != nullptr ? rec[o]->off : nullptr, co.off, size_t(B + 1) * 4);
            if (dense) {
                attn_output(o);
                qsa_input(p);
                qkv_rest(p);
            } else {
                stand(piece::qsa_out, o, send_in_producer ? "send X-A" : "", send_in_producer ? send_edit(o) : nullptr);
                stand(piece::qsa_qkv, p, "", nullptr);
            }
            {
                float * slot = rg.slot(p, x::msg::select, s_sel);
                int32_t * off = cp.sel_off;
                uint32_t * body = cp.sel_body;
                uint32_t * e = rg.err(p);
                add(p, 11, "X-SELECT recv", "recv X-SELECT", 1,
                    [B, slot, off, body, e](void * st) { return x::recv_select(B, slot, off, body, e, st); });
            }
            if (dense) {
                attention(p, cp.sel_body, cp.sel_off);
            } else {
                stand(piece::qsa_attention, p, "", nullptr);
            }
            rec_copy(p, rec != nullptr ? rec[p]->attn_out : nullptr, cp.attn_out, size_t(B) * 3072 * 4);
            rec_copy(p, rec != nullptr ? rec[p]->body : nullptr, cp.sel_body, size_t(bw) * 4);
            rec_copy(p, rec != nullptr ? rec[p]->off : nullptr, cp.sel_off, size_t(B + 1) * 4);
            if (dense) {
                attn_output(p);
            } else {
                stand(piece::qsa_out, p, send_in_producer ? "send X-A" : "", send_in_producer ? send_edit(p) : nullptr);
            }
        } else if (dense) {
            // the FFN with DP2's router on each card, then EX2's experts, which carry the shared expert (its term,
            // scaled by the router's sg, added last into the partial)
            const int32_t l = layer - 10;
            const fn::experts::fmt gu_fmt = l == 0 ? fn::experts::fmt::iq2_xs : fn::experts::fmt::iq2_xxs;
            const fn::dense::fmt sgf = l == 0 ? fn::dense::fmt::iq3_s : fn::dense::fmt::q4_k;
            const fn::dense::fmt suf = l == 0 ? fn::dense::fmt::iq4_xs : fn::dense::fmt::q4_k;
            const fn::experts::class_table cw = fn::experts::class_weights(gu_fmt, sgf, suf);
            for (int32_t r = 0; r < 2; ++r) {
                card & c = m.c[r];
                const int32_t * ids = hx7 ? c.ids_hx7[l] : c.d_ids;
                quant_op(r, layer, "act quantize", c.full[1], c.xc, c.xs, D);
                dense_op(r, layer, "dense router", c.d_rt[l], nullptr, nullptr, c.full[1]);
                uint8_t * meta = hx7 ? c.meta : c.d_meta; // gate_up writes the meta of ids for down (XB3)
                fn::experts::gate_up_args ga = {c.ex_gu[l],
                                                c.ex_sg[l],
                                                c.ex_su[l],
                                                c.grids,
                                                {c.xc, c.xs, B, D},
                                                ids,
                                                meta,
                                                {c.gluc, c.glus, GS, L},
                                                glu_f32 ? c.glu_f32 : nullptr,
                                                c.status};
                ga.cw = cw;
                add(r, layer, "experts gate_up", "", 1,
                    [gu_fmt, sgf, suf, ga, B](void * st) { return fn::experts::gate_up(gu_fmt, sgf, suf, B, ga, st); });
                const fn::experts::down_args da = {c.ex_dn[l], c.ex_sd[l], c.grids, {c.gluc, c.glus, GS, L},
                                                   ids,        meta,       c.wts,   c.sg,
                                                   c.partial,  c.status};
                add(r, layer, "experts down", "", 1, [da, B](void * st) {
                    return fn::experts::down(fn::experts::fmt::q2_0, fn::experts::fmt::iq4_nl, B, da, st);
                });
            }
        } else {
            // the FFN: router, shared expert, EX1b's routed experts
            const int32_t l = layer - 10;
            const int32_t ro = layer % 2;
            const piece p_router = l == 0 ? piece::router10 : piece::router11;
            const piece p_gu = l == 0 ? piece::sh_gate_up10 : piece::sh_gate_up11;
            const piece p_dn = l == 0 ? piece::sh_down10 : piece::sh_down11;
            const fn::experts::fmt gu_fmt = l == 0 ? fn::experts::fmt::iq2_xs : fn::experts::fmt::iq2_xxs;
            const fn::dense::fmt sgf = l == 0 ? fn::dense::fmt::iq3_s : fn::dense::fmt::q4_k;
            const fn::dense::fmt suf = l == 0 ? fn::dense::fmt::iq4_xs : fn::dense::fmt::q4_k;
            const fn::experts::class_table cw = fn::experts::class_weights(gu_fmt, sgf, suf);
            for (int32_t i = 0; i < 2 && ok; ++i) {
                const int32_t r = i == 0 ? ro : 1 - ro; // the router owner's body first (it sends X-ROUTE)
                card & c = m.c[r];
                {
                    // C2's quantize: the stand-in producer of EX1b's input until HC2's epilogue writes the format
                    const float * xin = c.full[1];
                    const fn::act::out_view xo = {c.xc, c.xs, B, D};
                    add(r, layer, "act quantize", "", 1,
                        [xin, xo](void * st) { return fn::act::quantize(xin, xo, st); });
                }
                if (router_both) {
                    int32_t * ids = c.ids;
                    stand(p_router, r, "", [ids](standin_args & a) { a.ids = ids; });
                } else if (r == ro) {
                    stand(p_router, r, "", nullptr);
                    const int32_t * idf = c.ids_full;
                    const float * wts = c.wts;
                    const float * sg = c.sg;
                    int32_t * ids = c.ids;
                    float * slot = rg.slot(1 - r, x::msg::route, s_route);
                    add(r, layer, "X-ROUTE send", "send X-ROUTE", 1, [B, idf, wts, sg, ids, slot](void * st) {
                        return x::send_route(B, idf, 10, wts, sg, ids, slot, st);
                    });
                }
                stand(p_gu, r, "", nullptr);
                if (!router_both && r != ro) {
                    float * slot = rg.slot(r, x::msg::route, s_route);
                    int32_t * ids = c.ids;
                    float * wts = c.wts;
                    float * sg = c.sg;
                    uint32_t * e = rg.err(r);
                    add(r, layer, "X-ROUTE recv", "recv X-ROUTE", 1, [B, slot, ids, wts, sg, e](void * st) {
                        return x::recv_route(B, slot, ids, wts, sg, e, st);
                    });
                }
                fn::experts::gate_up_args ga = {c.ex_gu[l],
                                                c.ex_sg[l],
                                                c.ex_su[l],
                                                c.grids,
                                                {c.xc, c.xs, B, D},
                                                c.ids,
                                                c.meta,
                                                {c.gluc, c.glus, GS, L},
                                                glu_f32 ? c.glu_f32 : nullptr,
                                                c.status};
                ga.cw = cw; // XB3: gate_up writes the meta of c.ids for down
                // EX2: the shared stand-ins still stream their bytes (load stand-ins), but their output (sh_part) no
                // longer reaches X-F: EX2's launches compute the shared expert, and its down writes the partial, which
                // the separate X-F send below takes in every variant (no real producer sends X-F from its epilogue)
                stand(p_dn, r, "", nullptr);
                add(r, layer, "experts gate_up", "", 1,
                    [gu_fmt, sgf, suf, ga, B](void * st) { return fn::experts::gate_up(gu_fmt, sgf, suf, B, ga, st); });
                const fn::experts::down_args da = {c.ex_dn[l], c.ex_sd[l], c.grids, {c.gluc, c.glus, GS, L},
                                                   c.ids,      c.meta,     c.wts,   c.sg,
                                                   c.partial,  c.status};
                add(r, layer, "experts down", "", 1, [da, B](void * st) {
                    return fn::experts::down(fn::experts::fmt::q2_0, fn::experts::fmt::iq4_nl, B, da, st);
                });
                if (rec != nullptr) {
                    card_record & R = *rec[r];
                    rec_copy(r, R.xc[l], c.xc, fn::act::codes_bytes(B, D));
                    rec_copy(r, R.xs[l], c.xs, fn::act::scales_bytes(B, D));
                    rec_copy(r, R.ids[l], c.ids, size_t(B) * 40);
                    rec_copy(r, R.wts[l], c.wts, size_t(B) * 40);
                    rec_copy(r, R.sg[l], c.sg, size_t(B) * 4);
                    rec_copy(r, R.gluc[l], c.gluc, sizeof(R.gluc[l]));
                    rec_copy(r, R.glus[l], c.glus, sizeof(R.glus[l]));
                    rec_copy(r, R.glu_f32[l], c.glu_f32, sizeof(R.glu_f32[l]));
                    rec_copy(r, R.sh_glu[l], c.sh_glu, size_t(B) * 320 * 4);
                }
            }
        }

        // the sublayer's destination-half exchange
        for (int32_t r = 0; r < 2 && (ffn || !send_in_producer); ++r) {
            const float * part = m.c[r].partial;
            float * slot = rg.slot(1 - r, half_msg, s_half);
            add(r, layer, half_name + " send", "send " + half_name, 1,
                [B, r, part, slot](void * st) { return x::send_half(B, 1 - r, part, slot, st); });
        }
        for (int32_t r = 0; r < 2; ++r) {
            card & c = m.c[r];
            const float * part = c.partial;
            float * slot = rg.slot(r, half_msg, s_half);
            float * out = c.out[set];
            uint32_t * e = rg.err(r);
            add(r, layer, half_name + " recv", "recv " + half_name, 1,
                [B, r, part, slot, out, e, half_code](void * st) {
                    return x::recv_add(B, r, part, slot, out, e, half_code, st);
                });
            if (rec != nullptr) {
                card_record & R = *rec[r];
                rec_copy(r, R.x[k], c.x[set], size_t(B) * S * F * 4);
                rec_copy(r, R.inject[k], c.inject[set], size_t(B) * S * 4);
                rec_copy(r, R.full[k], c.full[set], size_t(B) * D * 4);
                rec_copy(r, R.out[k], c.out[set], size_t(B) * F * 4);
            }
        }
    }
    if (rec != nullptr) {
        for (int32_t r = 0; r < 2; ++r) {
            rec_copy(r, rec[r]->sink, m.c[r].sink, sizeof(rec[r]->sink));
            rec_copy(r, &rec[r]->err, rg.err(r), 4);
        }
    }
    if (!ok) {
        return err.empty() ? fail(err, "build") : false;
    }
    if (!graph) {
        u.sub += 4;
        u.a += 2;
        u.f += 2;
        u.route += 2;
        u.select += 1;
    }
    return true;
}

bool pair::exchange_ops(const std::string & name, int32_t rows, int32_t rep, std::vector<op> & ops, std::string & err) {
    ops.clear();
    impl & m = *m_;
    x::rings & rg = *m.rg;
    const int32_t B = rows;
    const int32_t owner = index_owner;
    if (B < 1 || B > max_rows || rep < 1 || rep % x::n_slot != 0) {
        return fail(err, "exchange_ops: rows, or rep not a multiple of 3");
    }
    const auto add = [&](int32_t r, const std::string & piece_name, std::function<bool(void *)> fn) {
        ops.push_back({r, 0, piece_name, "", 1, std::move(fn)});
    };
    for (int32_t i = 0; i < rep; ++i) {
        const int32_t s = i % x::n_slot;
        for (int32_t r = 0; r < 2; ++r) {
            card & c = m.c[r];
            uint32_t * e = rg.err(r);
            if (name == "X-A" || name == "X-F") {
                const x::msg mm = name == "X-A" ? x::msg::a : x::msg::f;
                const uint32_t code = name == "X-A" ? x::err_poll_a : x::err_poll_f;
                const float * part = c.partial;
                float * peer = rg.slot(1 - r, mm, s);
                float * slot = rg.slot(r, mm, s);
                float * out = c.out[0];
                add(r, name + " send",
                    [B, r, part, peer](void * st) { return x::send_half(B, 1 - r, part, peer, st); });
                add(r, name + " recv", [B, r, part, slot, out, e, code](void * st) {
                    return x::recv_add(B, r, part, slot, out, e, code, st);
                });
            } else if (name == "X-ROUTE") {
                const int32_t * idf = c.ids_full;
                int32_t * ids = c.ids;
                float * wts = c.wts;
                float * sg = c.sg;
                float * peer = rg.slot(1 - r, x::msg::route, s);
                float * slot = rg.slot(r, x::msg::route, s);
                add(r, name + " send", [B, idf, wts, sg, ids, peer](void * st) {
                    return x::send_route(B, idf, 10, wts, sg, ids, peer, st);
                });
                add(r, name + " recv",
                    [B, slot, ids, wts, sg, e](void * st) { return x::recv_route(B, slot, ids, wts, sg, e, st); });
            } else if (name == "X-SELECT") {
                // the owner sends QA1b's body, the peer the body it received (the same words)
                const int32_t * off = r == owner ? c.off : c.sel_off;
                const uint32_t * body = r == owner ? c.body : c.sel_body;
                float * peer = rg.slot(1 - r, x::msg::select, s);
                float * slot = rg.slot(r, x::msg::select, s);
                int32_t * off_in = r == owner ? c.sel2_off : c.sel_off; // the owner's receive lands in scratch
                uint32_t * body_in = r == owner ? c.sel2_body : c.sel_body;
                add(r, name + " send",
                    [B, off, body, peer](void * st) { return x::send_select(B, off, body, peer, st); });
                add(r, name + " recv", [B, slot, off_in, body_in, e](void * st) {
                    return x::recv_select(B, slot, off_in, body_in, e, st);
                });
            } else if (name == "X-A V1") {
                standin_args a;
                piece_args(piece::qsa_out, r, B, a);
                a.peer_slot = rg.slot(1 - r, x::msg::a, s);
                a.peer_card = 1 - r;
                const float * part = c.partial;
                float * slot = rg.slot(r, x::msg::a, s);
                float * out = c.out[0];
                add(r, "qsa_out (send X-A)", [a](void * st) { return standin(piece::qsa_out, a, st); });
                add(r, "X-A recv", [B, r, part, slot, out, e](void * st) {
                    return x::recv_add(B, r, part, slot, out, e, x::err_poll_a, st);
                });
            } else if (name == "HC") {
                for (int32_t h = 0; h < 2; ++h) {
                    const int32_t n = 2 * i + h, set = h;
                    const int32_t sh = n % x::n_slot, sp = (n + 2 * rep - 1) % x::n_slot;
                    fn::hc::down2_args d2;
                    d2.d = {c.w_down[1], c.x[1 - set], c.out[1 - set], c.inject[1 - set],
                            c.gamma[1],  c.x[set],     c.mom[set]};
                    d2.hc_peer = rg.slot(1 - r, x::msg::hc, sh);
                    d2.hc_reset = rg.slot(r, x::msg::hc, sp);
                    const fn::hc::up2_args u2 = {c.w_up[1],
                                                 c.mom[set],
                                                 rg.slot(r, x::msg::hc, sh),
                                                 c.x[set],
                                                 c.gamma[1],
                                                 c.inject[set],
                                                 rg.slot(r, x::msg::hmix, sh),
                                                 rg.slot(1 - r, x::msg::hmix, sh),
                                                 c.runs,
                                                 e,
                                                 r};
                    const fn::hc::ready_args ra = {
                        rg.slot(r, x::msg::hmix, sh), c.full[set], nullptr, nullptr, e, r, 1};
                    add(r, "hc down2", [d2, B](void * st) { return fn::hc::down2(B, d2, st); });
                    add(r, "hc up2", [u2, B](void * st) { return fn::hc::up2(fn::hc::up_arm::ring, B, u2, st); });
                    add(r, "hc ready", [ra, B](void * st) { return fn::hc::ready(B, ra, st); });
                }
            } else {
                return fail(err, "exchange_ops: unknown exchange " + name);
            }
        }
    }
    return true;
}

bool pair::run_eager(const std::vector<op> & prog, std::string & err) {
    for (const op & o : prog) {
        card & c = m_->c[o.card];
        if (!cuda_ok(cudaSetDevice(c.dev), "device", err) || !o.fn(c.st)) {
            return err.empty() ? fail(err, "card " + std::to_string(o.card) + ": " + o.piece + " failed to enqueue")
                               : false;
        }
    }
    return sync(err);
}

bool pair::capture(const std::vector<op> & prog, std::string & err) {
    drop_graphs();
    if (m_->loopback) {
        return fail(err, "no graphs in loopback (one device: the cards' graphs would wait on each other)");
    }
    for (int32_t r = 0; r < 2; ++r) {
        card & c = m_->c[r];
        if (!cuda_ok(cudaSetDevice(c.dev), "device", err) || !cuda_ok(cudaStreamSynchronize(c.st), "sync", err) ||
            !cuda_ok(cudaStreamBeginCapture(c.st, cudaStreamCaptureModeThreadLocal), "begin capture", err)) {
            return false;
        }
        bool ok = true;
        for (const op & o : prog) {
            if (o.card == r && ok) {
                ok = o.launches > 0 && o.fn(c.st);
                if (!ok) {
                    err = "card " + std::to_string(r) + ": " + o.piece + " failed in capture";
                }
            }
        }
        cudaGraph_t g = nullptr;
        const cudaError_t e = cudaStreamEndCapture(c.st, &g);
        if (!ok || !cuda_ok(e, "end capture", err)) {
            if (g != nullptr) {
                cudaGraphDestroy(g);
            }
            return false;
        }
        c.graph = g;
        if (!cuda_ok(cudaGraphInstantiate(&c.exec, g, 0), "instantiate", err) ||
            !cuda_ok(cudaGraphUpload(c.exec, c.st), "upload", err) ||
            !cuda_ok(cudaStreamSynchronize(c.st), "upload", err)) {
            return false;
        }
    }
    m_->have_graphs = true;
    return true;
}

bool pair::replay(float ms[2], uint64_t start_ns[2], std::string & err) {
    for (int32_t r = 0; r < 2; ++r) {
        card & c = m_->c[r];
        if (c.exec == nullptr || !cuda_ok(cudaSetDevice(c.dev), "device", err) ||
            !cuda_ok(cudaEventRecord(c.ev[0], c.st), "record", err) ||
            !cuda_ok(cudaGraphLaunch(c.exec, c.st), "launch", err) ||
            !cuda_ok(cudaEventRecord(c.ev[1], c.st), "record", err)) {
            return err.empty() ? fail(err, "no graph") : false;
        }
    }
    if (!sync(err)) {
        return false;
    }
    for (int32_t r = 0; r < 2; ++r) {
        card & c = m_->c[r];
        cudaSetDevice(c.dev);
        if (!cuda_ok(cudaEventElapsedTime(&ms[r], c.ev[0], c.ev[1]), "elapsed", err) ||
            !cuda_ok(cudaMemcpy(&start_ns[r], c.stamp, 8, cudaMemcpyDeviceToHost), "stamp", err)) {
            return false;
        }
    }
    return true;
}

void pair::drop_graphs() {
    for (card & c : m_->c) {
        if (c.dev < 0) {
            continue;
        }
        cudaSetDevice(c.dev);
        if (c.exec != nullptr) {
            cudaGraphExecDestroy(c.exec);
            c.exec = nullptr;
        }
        if (c.graph != nullptr) {
            cudaGraphDestroy(c.graph);
            c.graph = nullptr;
        }
    }
    m_->have_graphs = false;
}

} // namespace fn::xp1
