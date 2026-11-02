#include "fn/tools/prompt_tune.h"

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_ladder.h"
#include "fn/kernels/dense/dense_prompt.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_ladder.h"
#include "fn/kernels/experts/experts_prompt.h"
#include "fn/kernels/experts/grids.h"
#include "fn/kernels/hc/hc_prompt.h"
#include "fn/loader/loader.h"
#include "fn/model/ladder.h"
#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/model/prompt.h"
#include "fn/model/shapes.h"
#include "fn/quant/formats.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace fn::tools {
namespace {

namespace md = fn::model;

constexpr int32_t D         = 2560;
constexpr int32_t NE        = 512;
constexpr int32_t NU        = 10;
constexpr int32_t warm      = 5;
constexpr int32_t timed     = 20;
constexpr int32_t max_rows  = 4096;  // a piece's rows (prompt_half_rows)
constexpr int32_t max_erung = 64;    // routed rows an expert of a 4,096-row piece: 80, rung 64

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "prompt-tune: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// a device buffer
struct dmem {
    uint8_t * p = nullptr;
    size_t    n = 0;

    dmem() = default;

    bool reset(size_t bytes) {
        if (p != nullptr) {
            cudaFree(p);
            p = nullptr;
            n = 0;
        }
        if (bytes > 0 && cuda_ok(cudaMalloc(&p, bytes), "cudaMalloc")) {
            n = bytes;
        }
        return bytes == 0 || p != nullptr;
    }

    ~dmem() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dmem(const dmem &)             = delete;
    dmem & operator=(const dmem &) = delete;

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); }
};

// one key's product, as the prompt path's call site forms it (fn/model/prompt.cpp layer_half), on card 0's weights
struct product {
    md::lad_key                   key;
    // dense packed (dprod): the run's sources, one format and layout; BF16 (bprod): rows b0 [n0] then b1 [n1]
    fn::dense::fmt                f   = fn::dense::fmt::q8_0;
    fn::dense::pd_layout          lay = fn::dense::pd_layout::dp2;
    std::vector<md::dense_weight> src;
    const uint16_t *              b0 = nullptr, *b1 = nullptr;
    int32_t                       n0 = 0, n1 = 0;
    // HC: both cards' halves
    fn::hc::hcp_weights           hw;
    // experts: the layer's packed experts and formats
    const uint8_t *               ex_gu = nullptr, *ex_dn = nullptr;
    fn::experts::fmt              gu_f = fn::experts::fmt::q4_k, dn_f = fn::experts::fmt::q5_1;
};

const uint16_t * hc_down_of(const md::hc_module & h) {
    return h.q8_down != nullptr ? reinterpret_cast<const uint16_t *>(h.q8_down) : h.w_down;
}

const uint16_t * hc_up_of(const md::hc_module & h) {
    return h.q8_up != nullptr ? reinterpret_cast<const uint16_t *>(h.q8_up) : h.w_up;
}

fn::dense::fmt dense_of(int32_t ef) {
    const auto f = static_cast<fn::experts::fmt>(ef);
    return f == fn::experts::fmt::q2_0 ? fn::dense::fmt::q2_0 :
           f == fn::experts::fmt::q8_0 ? fn::dense::fmt::q8_0 :
                                         fn::dense::fmt::iq4_nl;
}

void add(std::vector<product> & ps, product p) {
    for (const product & q : ps) {
        if (q.key == p.key) {
            return;
        }
    }
    ps.push_back(std::move(p));
}

// a dense call site's runs (prompt.cpp dense_run_keys: one format, at most three sources a run)
void dense_runs(std::vector<product> &                          ps,
                std::initializer_list<const md::dense_weight *> ws,
                int32_t                                         kk,
                fn::dense::pd_layout                            lay) {
    const md::dense_weight * const * w  = ws.begin();
    const int32_t                    nw = static_cast<int32_t>(ws.size());
    for (int32_t i = 0; i < nw;) {
        product p;
        p.f        = static_cast<fn::dense::fmt>(w[i]->fmt);
        p.lay      = lay;
        int32_t nn = 0;
        for (int32_t m = 0; i < nw && m < 3 && w[i]->fmt == static_cast<int32_t>(p.f); ++i, ++m) {
            p.src.push_back(*w[i]);
            nn += static_cast<int32_t>(w[i]->n_out);
        }
        p.key = md::prompt_dense_key(static_cast<int32_t>(p.f), static_cast<int32_t>(lay), kk, nn);
        add(ps, std::move(p));
    }
}

void bf16_prod(std::vector<product> & ps,
               const uint16_t *       b0,
               int32_t                n0,
               const uint16_t *       b1,
               int32_t                n1,
               int32_t                nn) {
    product p;
    p.key = md::prompt_dense_key(static_cast<int32_t>(fn::dense::fmt::bf16), 0, D, nn);
    p.b0  = b0;
    p.n0  = n0;
    p.b1  = b1;
    p.n1  = n1;
    add(ps, std::move(p));
}

void hc_prods(std::vector<product> & ps, const md::hc_module & h0, const md::hc_module & h1) {
    for (const bool up : { false, true }) {
        product p;
        p.key         = md::prompt_hc_key(up, h0.q8_down != nullptr);
        p.hw.form     = fn::hc::hcp_form::replicated;
        p.hw.q8       = h0.q8_down != nullptr;
        p.hw.down[0]  = hc_down_of(h0);
        p.hw.down[1]  = hc_down_of(h1);
        p.hw.up[0]    = hc_up_of(h0);
        p.hw.up[1]    = hc_up_of(h1);
        p.hw.gamma[0] = h0.gamma;
        p.hw.gamma[1] = h1.gamma;
        add(ps, std::move(p));
    }
}

// every key of the window's layers (both cards: the index owner's products) and the draft, prompt_ladder_keys' order
std::vector<product> products_of(const md::layer_weights (&w)[2][4],
                                 const md::draft_weights &       dw,
                                 const md::draft_index_weights & di) {
    std::vector<product> ps;
    for (int32_t l = 0; l < 4; ++l) {
        const md::layer_weights & a   = w[0][l];
        const bool                gdn = l % fn::full_attn_interval != fn::full_attn_interval - 1;
        hc_prods(ps, w[0][l].hc_attn, w[1][l].hc_attn);
        if (gdn) {
            dense_runs(ps, { &a.qkv, &a.z }, D, fn::dense::pd_layout::dp2);
            bf16_prod(ps, a.alpha, 24, a.beta, 24, 48);
            dense_runs(ps, { &a.ssm_out }, 3072, fn::dense::pd_layout::dp2);
        } else {
            dense_runs(ps, { &a.wq, &a.wk, &a.wv }, D, fn::dense::pd_layout::dp2);
            dense_runs(ps, { &a.wo }, 3072, fn::dense::pd_layout::dp2);
            const md::layer_weights & o = w[a.index_owner][l];
            bf16_prod(ps, o.idx_q, 512, o.idx_k, 128, 640);
        }
        hc_prods(ps, w[0][l].hc_ffn, w[1][l].hc_ffn);
        bf16_prod(ps, a.router, NE, a.sh_gate_w, 1, NE + 8);
        dense_runs(ps, { &a.sh_gate, &a.sh_up }, D, fn::dense::pd_layout::dp2);
        md::dense_weight sd;
        sd.w     = a.ex_sh_down;
        sd.fmt   = static_cast<int32_t>(dense_of(a.ex_sh_down_fmt));
        sd.k     = 320;
        sd.n_out = D;
        dense_runs(ps, { &sd }, 320, fn::dense::pd_layout::ex1_down);
        for (const bool down : { false, true }) {
            const auto f = static_cast<fn::experts::fmt>(down ? a.ex_down_fmt : a.ex_gu_fmt);
            if (down ? fn::experts::el_takes_down(f) : fn::experts::el_takes_gate_up(f)) {
                product p;
                p.key   = md::prompt_expert_key(down, down ? a.ex_down_fmt : a.ex_gu_fmt);
                p.ex_gu = a.ex_gate_up;
                p.ex_dn = a.ex_down;
                p.gu_f  = static_cast<fn::experts::fmt>(a.ex_gu_fmt);
                p.dn_f  = static_cast<fn::experts::fmt>(a.ex_down_fmt);
                add(ps, std::move(p));
            }
        }
    }
    dense_runs(ps, { &dw.eh }, 2 * D, fn::dense::pd_layout::dp2);
    dense_runs(ps, { &dw.wk }, D, fn::dense::pd_layout::dp2);
    if (di.k_proj != nullptr) {
        bf16_prod(ps, di.k_proj, 128, nullptr, 0, 128);
    }
    return ps;
}

// the timings of one candidate: 5 warm-up launches, then 20 each between two events; the median
struct timing {
    std::vector<float> us;
    float              med = -1.0f;
};

struct timer {
    cudaStream_t s = nullptr;
    cudaEvent_t  e[2 * timed]{};

    bool init() {
        if (!cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream")) {
            return false;
        }
        for (cudaEvent_t & x : e) {
            if (!cuda_ok(cudaEventCreate(&x), "event")) {
                return false;
            }
        }
        return true;
    }

    bool run(const std::function<bool()> & launch, timing & t) {
        for (int32_t i = 0; i < warm; ++i) {
            if (!launch()) {
                return false;
            }
        }
        for (int32_t i = 0; i < timed; ++i) {
            if (!cuda_ok(cudaEventRecord(e[size_t(2) * i], s), "event") || !launch() ||
                !cuda_ok(cudaEventRecord(e[size_t(2) * i + 1], s), "event")) {
                return false;
            }
        }
        if (!cuda_ok(cudaStreamSynchronize(s), "the timed launches")) {
            return false;
        }
        t.us.clear();
        for (int32_t i = 0; i < timed; ++i) {
            float ms = 0.0f;
            if (!cuda_ok(cudaEventElapsedTime(&ms, e[size_t(2) * i], e[size_t(2) * i + 1]), "elapsed")) {
                return false;
            }
            t.us.push_back(ms * 1000.0f);
        }
        std::vector<float> v = t.us;
        std::sort(v.begin(), v.end());
        t.med = 0.5f * (v[timed / 2 - 1] + v[timed / 2]);
        return true;
    }
};

// a candidate's measurement at a rung
struct meas {
    md::lad_cfg cfg;
    int32_t     rank  = 0;  // the cost model's (1 ..), 0 the retained route
    float       score = 0.0f;
    timing      t[2];       // experts: uniform, skewed; else t[0]
    float       us   = 0.0f;
    bool        pick = false;
};

std::string fam_enum(md::lad_fam f) {
    switch (f) {
        case md::lad_fam::dense:
            return "lad_fam::dense";
        case md::lad_fam::hc_down:
            return "lad_fam::hc_down";
        case md::lad_fam::hc_up:
            return "lad_fam::hc_up";
        case md::lad_fam::ex_gate_up:
            return "lad_fam::ex_gate_up";
        default:
            return "lad_fam::ex_down";
    }
}

// the shared buffers the timings run on (card 0's device)
struct bench {
    timer   t;
    dmem    x16, out, w16, prep, hx, mom, low, rinv, inj, m16, err, hct, hcs, scr, ws_part, ws_count, grids, dgrids;
    dmem    ids, offsets, sorted, pos, tasks, counters, lists, glu, part, ex_x;
    int64_t ws_floats    = 0;
    int32_t ws_count_len = 0;

    fn::tc::lad_ws ws() const {
        fn::tc::lad_ws w;
        w.part        = ws_part.as<float>();
        w.part_floats = ws_floats;
        w.count       = ws_count.as<int32_t>();
        w.count_len   = ws_count_len;
        return w;
    }
};

bool fill_random16(dmem & d, size_t count, uint64_t seed) {
    std::vector<uint16_t>           h(count);
    std::mt19937_64                 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t & v : h) {
        v = fn::quant::f32_to_f16(nd(g));
    }
    return cuda_ok(cudaMemcpy(d.p, h.data(), count * 2, cudaMemcpyHostToDevice), "random activations");
}

bool setup(bench & b, const md::lad_model & dm) {
    const size_t R = max_rows;
    b.ws_floats    = std::max<int64_t>(dm.ws_floats, 1);
    b.ws_count_len = std::max(dm.ws_count, 1);
    bool ok = b.t.init() && b.x16.reset(R * 2 * D * 2) && b.out.reset(R * 10240 * 4) &&
              b.w16.reset(size_t(10240) * 2 * D * 2) && b.prep.reset(size_t(1024) * 4) && b.hx.reset(R * 4 * D * 4) &&
              b.mom.reset(R * fn::hc::hcp_mom * 4) && b.low.reset(R * 320 * 4) && b.rinv.reset(R * 16) &&
              b.inj.reset(R * 16) && b.m16.reset(R * D * 2) && b.err.reset(256) &&
              b.hct.reset(size_t(fn::hc::hct_weight_bytes())) && b.hcs.reset(size_t(fn::hc::hct_scale_bytes())) &&
              b.scr.reset(size_t(fn::hc::hct_scratch_bytes(max_rows))) && b.ws_part.reset(size_t(b.ws_floats) * 4) &&
              b.ws_count.reset(size_t(b.ws_count_len) * 4) && b.grids.reset(fn::experts::grid_bytes) &&
              b.dgrids.reset(fn::dense::grid_bytes);
    // the experts: ids [R][10], the route, the grouped lists, the GLU and the partials of every pair
    const size_t pairs = size_t(fn::experts::ep_max_pairs);
    ok = ok && b.ids.reset(pairs * 4) && b.offsets.reset(size_t(513) * 4) && b.sorted.reset(pairs * 4) &&
         b.pos.reset(pairs * 4) && b.tasks.reset(size_t(fn::experts::ep_max_tasks) * 16) && b.counters.reset(64) &&
         b.lists.reset(size_t(2) * fn::experts::ep_max_ltasks * 16) && b.glu.reset(pairs * 320 * 2) &&
         b.part.reset(pairs * D * 4) && b.ex_x.reset(R * D * 2);
    if (!ok) {
        return false;
    }
    std::vector<uint8_t> g(fn::experts::grid_bytes), dg(fn::dense::grid_bytes);
    if (!fn::experts::grid_table(fn::experts::grid_src, fn::experts::grid_src_bytes, g.data()) ||
        !fn::dense::grid_table(fn::experts::iq3s_grid, sizeof(fn::experts::iq3s_grid), fn::experts::kvalues_iq4nl,
                               sizeof(fn::experts::kvalues_iq4nl), dg.data())) {
        fprintf(stderr, "prompt-tune: the grid tables\n");
        return false;
    }
    std::vector<float>                    hx(R * 4 * D);
    std::mt19937_64                       gen(0xa9f2);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
    for (float & v : hx) {
        v = ud(gen);
    }
    return cuda_ok(cudaMemcpy(b.grids.p, g.data(), g.size(), cudaMemcpyHostToDevice), "grids") &&
           cuda_ok(cudaMemcpy(b.dgrids.p, dg.data(), dg.size(), cudaMemcpyHostToDevice), "dense grids") &&
           cuda_ok(cudaMemcpy(b.hx.p, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "HC streams") &&
           fill_random16(b.x16, R * 2 * D, 0x51) && fill_random16(b.ex_x, R * D, 0x52) &&
           cuda_ok(cudaMemset(b.ws_count.p, 0, b.ws_count.n), "counters") &&
           cuda_ok(cudaMemset(b.err.p, 0, b.err.n), "error words");
}

// ---- dense ----

bool dense_launch(bench & b, const product & p, int32_t rows, const md::lad_cfg & c) {
    const int32_t     n = p.key.n, k = p.key.k;
    fn::dense::pd_act x;
    x.a0      = b.x16.as<uint16_t>();
    x.ld0     = k;
    x.k_split = k;
    const fn::dense::pt_seg seg{ b.out.as<float>(), n, n };
    if (c.retained()) {
        // pd_convert (BF16: pd_convert_bf16) of every source, then pd_tc
        fn::dense::pt_args t;
        t.w      = b.w16.as<uint16_t>();
        t.n      = n;
        t.k      = k;
        t.rows   = rows;
        t.x      = x;
        t.seg[0] = seg;
        t.n_seg  = 1;
        if (p.src.empty()) {
            float * cs  = b.prep.as<float>();
            t.col_scale = cs;
            return fn::dense::pd_convert_bf16(p.b0, p.n0, D, b.w16.as<uint16_t>(), cs, b.t.s) &&
                   (p.b1 == nullptr || fn::dense::pd_convert_bf16(
                                           p.b1, p.n1, D, b.w16.as<uint16_t>() + size_t(p.n0) * D, cs + p.n0, b.t.s)) &&
                   fn::dense::pd_tc(t, b.t.s);
        }
        int64_t at = 0;
        for (const md::dense_weight & w : p.src) {
            fn::dense::pd_args a;
            a.f     = p.f;
            a.lay   = p.lay;
            a.k     = k;
            a.n     = static_cast<int32_t>(w.n_out);
            a.w     = w.w;
            a.grids = b.dgrids.as<uint8_t>();
            if (!fn::dense::pd_convert(a, b.w16.as<uint16_t>() + at, b.t.s)) {
                return false;
            }
            at += int64_t(w.n_out) * k;
        }
        return fn::dense::pd_tc(t, b.t.s);
    }
    fn::dense::pq_args q;
    q.f     = p.f;
    q.lay   = p.lay;
    q.grids = b.dgrids.as<uint8_t>();
    for (const md::dense_weight & w : p.src) {
        q.src[q.n_src++] = { w.w, static_cast<int32_t>(w.n_out) };
    }
    if (p.src.empty()) {
        q.b0      = p.b0;
        q.n0      = p.n0;
        q.b1      = p.b1;
        q.n1      = p.n1;
        q.b_scale = b.prep.as<float>();
    }
    q.t.n      = n;
    q.t.k      = k;
    q.t.rows   = rows;
    q.t.x      = x;
    q.t.seg[0] = seg;
    q.t.n_seg  = 1;
    if (p.src.empty()) {
        q.t.col_scale = b.prep.as<float>();
    }
    return fn::dense::pl_tc(q, c.tile, c.splits, b.ws(), b.t.s);
}

// the BF16 product's prepared scales (prompt_path::prepare's pd_bf16_scales), for the ladder's tiles
bool dense_prepare(bench & b, const product & p) {
    if (!p.src.empty()) {
        return true;
    }
    float * cs = b.prep.as<float>();
    return cuda_ok(cudaMemset(cs, 0, size_t(p.key.n) * 4), "scales") &&
           fn::dense::pd_bf16_scales(p.b0, p.n0, D, cs, b.t.s) &&
           (p.b1 == nullptr || fn::dense::pd_bf16_scales(p.b1, p.n1, D, cs + p.n0, b.t.s)) &&
           cuda_ok(cudaStreamSynchronize(b.t.s), "scales");
}

// ---- HC ----

fn::hc::hcp_args hc_args(bench & b, const product & p, int32_t rows) {
    fn::hc::hcp_args a;
    a.w          = p.hw;
    a.rows       = rows;
    a.x          = b.hx.as<float>();
    a.mom        = b.mom.as<float>();
    a.low        = b.low.as<float>();
    a.rinv       = b.rinv.as<float>();
    a.inject     = b.inj.as<float>();
    a.mixed16    = b.m16.as<uint16_t>();
    a.ld_mixed16 = D;
    a.err        = b.err.as<uint32_t>();
    return a;
}

bool hc_launch(bench & b, const product & p, int32_t rows, const fn::hc::hct_route & r, bool retained) {
    const fn::hc::hcp_args    a = hc_args(b, p, rows);
    const fn::hc::hct_combine none;
    if (retained) {
        const fn::hc::hct_weights hw = fn::hc::hct_carve(b.hct.as<uint8_t>());
        return fn::hc::hct_convert(a.w, hw, b.t.s) && fn::hc::hct_mix(a, none, hw, b.scr.as<uint8_t>(), b.t.s);
    }
    return fn::hc::hct_mix_ladder(a, none, fn::hc::hct_carve_scales(b.hcs.as<uint8_t>()), b.scr.as<uint8_t>(), r,
                                  b.ws(), b.t.s);
}

// ---- experts ----

// SYNTHETIC routing for rows rows: uniform (pair 10 j + s to expert (10 j + s) mod 512: every expert rows x 10 / 512
// pairs, within one) or skewed (each row's first five slots on 32 hot experts, the rest uniform over the others)
std::vector<int32_t> routing(int32_t rows, bool skewed) {
    std::vector<int32_t> ids(size_t(rows) * NU);
    for (int32_t j = 0; j < rows; ++j) {
        for (int32_t s = 0; s < NU; ++s) {
            const int32_t i = j * NU + s;
            ids[size_t(i)]  = !skewed ? i % NE : s < 5 ? (j * 5 + s) % 32 : 32 + (j * 5 + s - 5) % (NE - 32);
        }
    }
    return ids;
}

bool ex_route(bench & b, const product & p, int32_t rows, int32_t tile, bool down) {
    fn::experts::ep_route r;
    r.offsets  = b.offsets.as<int32_t>();
    r.sorted   = b.sorted.as<int32_t>();
    r.pos      = b.pos.as<int32_t>();
    r.tasks    = b.tasks.as<int32_t>();
    r.counters = b.counters.as<int32_t>();
    fn::experts::ep_lists ls;
    if (tile >= 0) {
        ls.tasks[down ? 1 : 0] = b.lists.as<int32_t>() + (down ? size_t(fn::experts::ep_max_ltasks) * 4 : 0);
        ls.rows[down ? 1 : 0]  = fn::experts::el_rows(tile);
    }
    return fn::experts::ep_group(rows, fn::experts::ep_task_rows(p.gu_f), b.ids.as<int32_t>(), r, nullptr, b.t.s, ls) &&
           cuda_ok(cudaStreamSynchronize(b.t.s), "ep_group");
}

bool ex_launch(bench & b, const product & p, int32_t rows, const md::lad_cfg & c, bool down) {
    fn::experts::ep_route r;
    r.offsets  = b.offsets.as<int32_t>();
    r.sorted   = b.sorted.as<int32_t>();
    r.pos      = b.pos.as<int32_t>();
    r.tasks    = b.tasks.as<int32_t>();
    r.counters = b.counters.as<int32_t>();
    if (!down) {
        const fn::experts::ep_gate_up_args a = { p.ex_gu, b.grids.as<uint8_t>(), b.ex_x.as<uint16_t>(),
                                                 r,       b.glu.as<uint16_t>(),  nullptr };
        return c.retained() ? fn::experts::ep_gate_up(p.gu_f, rows, a, b.t.s) :
                              fn::experts::ep_gate_up_ladder(p.gu_f, c.tile, rows, a, b.lists.as<int32_t>(),
                                                             b.counters.as<int32_t>() + 2, b.t.s);
    }
    fn::experts::ep_down_args a = { p.ex_dn, b.grids.as<uint8_t>(), b.glu.as<uint16_t>(), r, b.part.as<float>() };
    return c.retained() ? fn::experts::ep_down(p.dn_f, rows, a, b.t.s) :
                          fn::experts::ep_down_ladder(p.dn_f, c.tile, rows, a,
                                                      b.lists.as<int32_t>() + size_t(fn::experts::ep_max_ltasks) * 4,
                                                      b.counters.as<int32_t>() + 3, b.t.s);
}

// ---- the tuning ----

struct key_result {
    const product * p = nullptr;

    struct rung_res {
        int32_t           rung = 0, rows = 0;
        std::vector<meas> c;
    };

    std::vector<rung_res> rungs;
};

void pick(std::vector<meas> & c) {
    // the fastest median; the retained route on a tie
    int32_t best = -1;
    for (int32_t i = 0; i < int32_t(c.size()); ++i) {
        if (best < 0 || c[size_t(i)].us < c[size_t(best)].us ||
            (c[size_t(i)].us == c[size_t(best)].us && c[size_t(i)].cfg.retained())) {
            best = i;
        }
    }
    for (int32_t i = 0; i < int32_t(c.size()); ++i) {
        c[size_t(i)].pick = i == best;
    }
}

const meas * picked(const std::vector<meas> & c) {
    for (const meas & m : c) {
        if (m.pick) {
            return &m;
        }
    }
    return nullptr;
}

bool tune_key(bench & b, const product & p, int32_t device, key_result & kr, FILE * tsv) {
    md::lad_model m;
    if (!md::prompt_lad_model(p.key, device, m)) {
        fprintf(stderr, "prompt-tune: %s: the cost model\n", p.key.text().c_str());
        return false;
    }
    const bool experts = p.key.fam == md::lad_fam::ex_gate_up || p.key.fam == md::lad_fam::ex_down;
    const bool hc      = p.key.fam == md::lad_fam::hc_down || p.key.fam == md::lad_fam::hc_up;
    const bool down    = p.key.fam == md::lad_fam::ex_down;
    kr.p               = &p;
    if (!experts && !hc && !dense_prepare(b, p)) {
        return false;
    }
    if (hc && !(fn::hc::hct_scales(p.hw, fn::hc::hct_carve_scales(b.hcs.as<uint8_t>()), b.t.s) &&
                cuda_ok(cudaStreamSynchronize(b.t.s), "HC scales"))) {
        return false;
    }
    for (int32_t ri = 0; ri < md::lad_n_rungs; ++ri) {
        const int32_t rung = md::lad_rungs[ri];
        if (rung > (experts ? max_erung : max_rows)) {
            break;
        }
        // the piece's rows: the rung's (experts: the rows whose routed rows an expert are the rung's, at least)
        const int32_t         rows = experts ? std::min(max_rows, (rung * NE + NU - 1) / NU) : rung;
        const md::lad_ranking rk   = md::lad_rank(p.key.fam, rung, p.key.k, p.key.n, m);
        key_result::rung_res  rr;
        rr.rung      = rung;
        rr.rows      = rows;
        int32_t rank = 0;
        for (const md::lad_cfg & c : md::lad_candidates(rk)) {
            meas mc;
            mc.cfg  = c;
            mc.rank = c.retained() ? 0 : ++rank;
            for (const md::lad_cand & x : rk.ranked) {
                if (x.cfg == c) {
                    mc.score = x.score;
                }
            }
            for (int32_t layout = 0; layout < (experts ? 2 : 1); ++layout) {
                bool ok = true;
                if (experts) {
                    const std::vector<int32_t> ids = routing(rows, layout == 1);
                    ok = cuda_ok(cudaMemcpy(b.ids.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids") &&
                         ex_route(b, p, rows, c.tile, down) &&
                         // the down reads the GLU: one gate/up first (its contents do not matter to the timing)
                         (!down || ex_launch(b, p, rows, md::lad_cfg{}, false));
                    ok = ok && b.t.run([&]() { return ex_launch(b, p, rows, c, down); }, mc.t[layout]);
                } else if (hc) {
                    fn::hc::hct_route r;
                    if (p.key.fam == md::lad_fam::hc_down) {
                        r.down_tile   = c.tile;
                        r.down_splits = c.splits;
                    } else {
                        r.up_tile   = c.tile;
                        r.up_splits = c.splits;
                    }
                    ok = b.t.run([&]() { return hc_launch(b, p, rows, r, c.retained()); }, mc.t[0]);
                } else {
                    ok = b.t.run([&]() { return dense_launch(b, p, rows, c); }, mc.t[0]);
                }
                if (!ok) {
                    fprintf(stderr, "prompt-tune: %s rung %d: %s failed\n", p.key.text().c_str(), rung,
                            c.text().c_str());
                    return false;
                }
            }
            mc.us = experts ? 0.5f * (mc.t[0].med + mc.t[1].med) : mc.t[0].med;
            rr.c.push_back(mc);
        }
        pick(rr.c);
        for (const meas & mc : rr.c) {
            for (int32_t layout = 0; layout < (experts ? 2 : 1); ++layout) {
                fprintf(tsv, "%s\t%d\t%d\t%s\t%s\t%d\t%.4f\t%.3f\t%s", p.key.text().c_str(), rung, rows,
                        experts ? (layout == 0 ? "uniform (synthetic)" : "skewed (synthetic)") : "-",
                        mc.cfg.text().c_str(), mc.rank, static_cast<double>(mc.score),
                        static_cast<double>(mc.t[layout].med), mc.pick ? "pick" : "-");
                for (const float u : mc.t[layout].us) {
                    fprintf(tsv, "\t%.3f", static_cast<double>(u));
                }
                fprintf(tsv, "\n");
            }
        }
        fflush(tsv);
        const meas * w = picked(rr.c);
        printf("prompt-tune: %s rung %d (%d rows): %s %.2f us (retained %.2f us)\n", p.key.text().c_str(), rung, rows,
               w->cfg.text().c_str(), static_cast<double>(w->us), static_cast<double>(rr.c.back().us));
        fflush(stdout);
        kr.rungs.push_back(std::move(rr));
    }
    return true;
}

// HC: each rung's two picks together against the retained route; the retained route kept for both when the pair is
// not faster (the prompt path takes the ladder only when both products do)
bool hc_pairs(bench & b, std::vector<key_result> & res, FILE * tsv) {
    for (key_result & kd : res) {
        if (kd.p->key.fam != md::lad_fam::hc_down) {
            continue;
        }
        key_result * ku = nullptr;
        for (key_result & x : res) {
            if (x.p->key.fam == md::lad_fam::hc_up && x.p->key.fmt == kd.p->key.fmt) {
                ku = &x;
            }
        }
        if (ku == nullptr) {
            continue;
        }
        for (size_t i = 0; i < kd.rungs.size() && i < ku->rungs.size(); ++i) {
            const meas *d = picked(kd.rungs[i].c), *u = picked(ku->rungs[i].c);
            if (d->cfg.retained() || u->cfg.retained()) {
                continue;
            }
            const int32_t     rows = kd.rungs[i].rows;
            fn::hc::hct_route r{ d->cfg.tile, d->cfg.splits, u->cfg.tile, u->cfg.splits };
            timing            tp;
            if (!(fn::hc::hct_scales(kd.p->hw, fn::hc::hct_carve_scales(b.hcs.as<uint8_t>()), b.t.s) &&
                  b.t.run([&]() { return hc_launch(b, *kd.p, rows, r, false); }, tp))) {
                return false;
            }
            const float ret  = kd.rungs[i].c.back().us;
            const bool  keep = tp.med < ret;
            fprintf(tsv, "%s pair\t%d\t%d\t-\tdown %s, up %s\t0\t0\t%.3f\t%s", kd.p->key.fmt.c_str(), kd.rungs[i].rung,
                    rows, d->cfg.text().c_str(), u->cfg.text().c_str(), static_cast<double>(tp.med),
                    keep ? "kept" : "retained instead");
            for (const float x : tp.us) {
                fprintf(tsv, "\t%.3f", static_cast<double>(x));
            }
            fprintf(tsv, "\n");
            printf("prompt-tune: HC %s rung %d: the pair %.2f us against the retained route %.2f us: %s\n",
                   kd.p->key.fmt.c_str(), kd.rungs[i].rung, static_cast<double>(tp.med), static_cast<double>(ret),
                   keep ? "kept" : "the retained route instead");
            if (!keep) {
                for (std::vector<meas> * c : { &kd.rungs[i].c, &ku->rungs[i].c }) {
                    for (meas & x : *c) {
                        x.pick = x.cfg.retained();
                    }
                }
            }
        }
    }
    return true;
}

bool write_table(const std::string &             path,
                 const std::vector<key_result> & res,
                 const std::string &             model,
                 const char *                    dev) {
    FILE * f = fopen(path.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "prompt-tune: cannot write %s\n", path.c_str());
        return false;
    }
    char        when[32];
    std::time_t now = std::time(nullptr);
    std::strftime(when, sizeof when, "%Y-%m-%d", std::gmtime(&now));
    fprintf(
        f,
        "// APF2: the kernel-size ladder's measured table, written by fnfl-drive prompt-tune (fn/tools/prompt_tune.h)\n"
        "// from %s on %s, %s: each candidate's median of %d timed launches after %d warm-up launches\n"
        "// (experts: the mean of the uniform and skewed SYNTHETIC routings' medians); pick marks each (key, rung)'s\n"
        "// winner, the retained route on a tie. { family, format, K, N, rung, tile (-1 retained), splits, us, pick "
        "}\n",
        model.substr(model.find_last_of('/') + 1).c_str(), dev, when, timed, warm);
    for (const key_result & k : res) {
        for (const auto & r : k.rungs) {
            for (const meas & m : r.c) {
                fprintf(f, "{ %s, \"%s\", %d, %d, %d, %d, %d, %.3ff, %s },\n", fam_enum(k.p->key.fam).c_str(),
                        k.p->key.fmt.c_str(), k.p->key.k, k.p->key.n, r.rung, m.cfg.tile, m.cfg.splits,
                        static_cast<double>(m.us), m.pick ? "true" : "false");
            }
        }
    }
    fclose(f);
    return true;
}

}  // namespace

int fnfl_prompt_tune(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "prompt-tune: MODEL DRAFT OUT [DEVICE]\n");
        return 2;
    }
    const std::string model = argv[2], draft = argv[3], out = argv[4];
    const int32_t     device = argc >= 6 ? atoi(argv[5]) : 0;
    std::error_code   ec;
    std::filesystem::create_directories(out, ec);
    if (ec) {
        fprintf(stderr, "prompt-tune: %s: %s\n", out.c_str(), ec.message().c_str());
        return 1;
    }
    if (!cuda_ok(cudaSetDevice(device), "device") || !cuda_ok(cudaFree(nullptr), "context")) {
        return 1;
    }
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "properties")) {
        return 1;
    }
    // one load: layers 0 .. 3 and the head with the draft, both cards' halves on the one device
    fn::loader::load_params lp;
    lp.model_path            = model;
    lp.draft_path            = draft;
    lp.device[0]             = device;
    lp.device[1]             = device;
    lp.expert_cache          = false;
    lp.win.lo                = 0;
    lp.win.hi                = 3;
    lp.win.head              = true;
    lp.weights_only          = true;  // both cards' halves on the one device, no KV or exchange
    auto                   m = std::make_unique<md::model>();
    fn::loader::load_times lt;
    std::string            err;
    if (!fn::loader::open_model(lp, *m, lt, err) || !fn::loader::allocate_weights(*m, err) ||
        !fn::loader::fill_weights(lp, *m, lt, err)) {
        fprintf(stderr, "prompt-tune: load: %s\n", err.c_str());
        return 1;
    }
    md::layer_weights       w[2][4];
    md::draft_weights       dw;
    md::draft_index_weights di;
    for (int32_t l = 0; l < 4; ++l) {
        if (!md::pack_dense(*m, l, err)) {
            fprintf(stderr, "prompt-tune: pack layer %d: %s\n", l, err.c_str());
            return 1;
        }
    }
    if (!md::pack_draft(*m, err)) {
        fprintf(stderr, "prompt-tune: pack the draft: %s\n", err.c_str());
        return 1;
    }
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t l = 0; l < 4; ++l) {
            if (!md::resolve_weights(*m, l, c, w[c][l], err)) {
                fprintf(stderr, "prompt-tune: layer %d card %d: %s\n", l, c, err.c_str());
                return 1;
            }
        }
    }
    if (!md::resolve_draft_weights(*m, 0, dw, err) || !md::resolve_draft_index_weights(*m, di, err)) {
        fprintf(stderr, "prompt-tune: the draft: %s\n", err.c_str());
        return 1;
    }
    if (!md::dense_q8_of(*m)) {
        fprintf(stderr, "prompt-tune: the table is UD-Q4_K_XL's (dense Q8_0); %s is another file\n", model.c_str());
        return 1;
    }
    if (!fn::dense::pl_attributes() || !fn::hc::hcl_attributes() || !fn::experts::ep_prompt_attributes() ||
        !fn::dense::pq_tc_attributes()) {
        fprintf(stderr, "prompt-tune: the kernels' shared memory opt-ins\n");
        return 1;
    }
    const std::vector<product> ps = products_of(w, dw, di);
    printf("prompt-tune: %s (draft %s) on device %d (%s): %zu ladder keys\n", model.c_str(), draft.c_str(), device,
           prop.name, ps.size());
    md::lad_model dm;
    if (!md::prompt_lad_model(ps.front().key, device, dm)) {
        return 1;
    }
    for (const product & p : ps) {
        md::lad_model km;
        if (md::prompt_lad_model(p.key, device, km)) {
            dm.ws_floats = std::max(dm.ws_floats, km.ws_floats);
            dm.ws_count  = std::max(dm.ws_count, km.ws_count);
        }
    }
    bench b;
    if (!setup(b, dm)) {
        return 1;
    }
    FILE * tsv = fopen((out + "/prompt-tune.tsv").c_str(), "w");
    if (tsv == nullptr) {
        fprintf(stderr, "prompt-tune: cannot write %s/prompt-tune.tsv\n", out.c_str());
        return 1;
    }
    fprintf(tsv, "key\trung\trows\trouting\tcandidate\trank\tscore\tmedian_us\tpick");
    for (int32_t i = 0; i < timed; ++i) {
        fprintf(tsv, "\tus%d", i + 1);
    }
    fprintf(tsv, "\n");
    std::vector<key_result> res(ps.size());
    bool                    ok = true;
    for (size_t i = 0; i < ps.size() && ok; ++i) {
        printf("prompt-tune: %s\n", ps[i].key.text().c_str());
        ok = tune_key(b, ps[i], device, res[i], tsv);
    }
    ok = ok && hc_pairs(b, res, tsv);
    fclose(tsv);
    if (!ok || !write_table(out + "/ladder_table.inc", res, model, prop.name)) {
        printf("prompt-tune: FAIL\n");
        return 1;
    }
    int32_t entries = 0, ladder = 0;
    for (const key_result & k : res) {
        for (const auto & r : k.rungs) {
            entries += int32_t(r.c.size());
            const meas * x = picked(r.c);
            ladder += x != nullptr && !x->cfg.retained() ? 1 : 0;
        }
    }
    printf(
        "prompt-tune: wrote %s/ladder_table.inc (%d entries; %d (key, rung) picks a ladder configuration) and "
        "%s/prompt-tune.tsv\n",
        out.c_str(), entries, ladder, out.c_str());
    printf("prompt-tune: PASS\n");
    return 0;
}

}  // namespace fn::tools
