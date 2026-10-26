#include "fn/model/layer.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/grids.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/ple/ple.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/kv_tier.h"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace fn::model {

namespace {

constexpr int32_t D = fn::n_embd;                               // 2560
constexpr int32_t F = fn::hc::n_own;                            // 1280
constexpr int32_t GH = gdn_heads_card;                          // 24 value heads
constexpr int32_t GS = gdn_head_dim;                            // 128
constexpr int32_t GC = gdn_conv_channels_card;                  // 5120
constexpr int32_t GV = GH * GS;                                 // 3072
constexpr int32_t QH = 12;                                      // query heads a card
constexpr int32_t HD = fn::n_embd_head;                         // 256
constexpr int32_t QF = QH * 2 * HD;                             // 6144: [q | gate] a head
constexpr int32_t FFS = fn::experts::n_ff_card;                 // 320
constexpr int32_t NE = fn::n_expert;                            // 512
constexpr int32_t NU = fn::n_expert_used;                       // 10
constexpr int32_t IQ = fn::qsa::n_idx_head * fn::qsa::head_dim; // 512 index query values a row
constexpr int64_t gdn_state_floats = int64_t(GH) * GS * GS;
constexpr int64_t conv_floats = int64_t(GC) * gdn_conv_history;
static_assert(GC == fn::gdn::n_channel && GV == fn::gdn::n_value && GH == fn::gdn::n_head, "gdn.h's card shapes");
static_assert(GV == fn::qsa::attn::out_width && QF == QH * fn::qsa::attn::q_stride, "attend.h's card shapes");
static_assert(NE == fn::dense::route_experts && NU == fn::dense::route_used, "the router epilogue's shapes");

int64_t align256(int64_t n) {
    return (n + 255) / 256 * 256;
}

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

std::string tname(int32_t layer, const char * t) {
    return "blk." + std::to_string(layer) + "." + t;
}

// rows and columns a card holds of a tensor, from the placement's ranges
void card_extent(int32_t t, int32_t card, int64_t & rows, int64_t & cols) {
    const card_place & cp = tensors[t].card[card];
    rows = cols = 0;
    for (int32_t i = 0; i < cp.row_count; ++i) {
        rows += ranges[cp.row_first + i].count;
    }
    for (int32_t i = 0; i < cp.col_count; ++i) {
        cols += ranges[cp.col_first + i].count;
    }
}

// the quantized formats DP2's products take (BF16 matrices are resolved as raw pointers)
bool dense_fmt(const char * f, int32_t & out) {
    using fn::dense::fmt;
    const struct {
        const char * name;
        fmt q;
    } map[] = {{"Q4_K", fmt::q4_k},   {"Q5_K", fmt::q5_k}, {"Q6_K", fmt::q6_k},     {"IQ4_XS", fmt::iq4_xs},
               {"IQ3_S", fmt::iq3_s}, {"Q8_0", fmt::q8_0}, {"IQ4_NL", fmt::iq4_nl}, {"Q2_0", fmt::q2_0}};
    for (const auto & e : map) {
        if (std::strcmp(f, e.name) == 0) {
            out = static_cast<int32_t>(e.q);
            return true;
        }
    }
    return false;
}

bool expert_fmt(const char * f, int32_t & out) {
    const struct {
        const char * name;
        fn::experts::fmt e;
    } map[] = {{"IQ2_XXS", fn::experts::fmt::iq2_xxs}, {"IQ2_XS", fn::experts::fmt::iq2_xs},
               {"IQ2_S", fn::experts::fmt::iq2_s},     {"IQ3_XXS", fn::experts::fmt::iq3_xxs},
               {"IQ3_S", fn::experts::fmt::iq3_s},     {"Q2_0", fn::experts::fmt::q2_0},
               {"IQ4_NL", fn::experts::fmt::iq4_nl},   {"Q8_0", fn::experts::fmt::q8_0}};
    for (const auto & e : map) {
        if (std::strcmp(f, e.name) == 0) {
            out = static_cast<int32_t>(e.e);
            return true;
        }
    }
    return false;
}

struct finder {
    const model & m;
    int32_t layer, card;
    std::string & err;

    // the region of a tensor on this card; nullptr (and err) when the card holds none of it
    const uint8_t * ptr(const char * t, int32_t & idx) const {
        idx = model::find(tname(layer, t).c_str());
        const uint8_t * p = idx < 0 ? nullptr : m.region_ptr(idx, card);
        if (p == nullptr) {
            err = tname(layer, t) + ": not on card " + std::to_string(card);
        }
        return p;
    }
    const uint8_t * ptr(const char * t) const {
        int32_t idx = -1;
        return ptr(t, idx);
    }
    bool dense(const char * t, dense_weight & d) const {
        int32_t idx = -1;
        d.w = ptr(t, idx);
        if (d.w == nullptr) {
            return false;
        }
        const tensor_place & tp = tensors[idx];
        if (tp.lay != layout::gguf || !dense_fmt(tp.format, d.fmt)) {
            err = tname(layer, t) + ": format " + tp.format + " is not a dense format the step runs";
            return false;
        }
        if (idx >= (int32_t)m.dense_packed.size() || m.dense_packed[idx] == 0) {
            err = tname(layer, t) + ": not in DP2's packed layout (pack_dense)";
            return false;
        }
        int64_t rows = 0, cols = 0;
        card_extent(idx, card, rows, cols);
        d.k = cols;
        d.n_out = rows;
        d.row_bytes = cols / tp.block_elems * tp.block_bytes;
        return true;
    }
    // the shared expert's down: Q2_0 or IQ4_NL, repacked by pack_dense into the experts' layout (one expert)
    bool shared_down(const char * t, const uint8_t *& w, int32_t & f) const {
        int32_t idx = -1;
        w = ptr(t, idx);
        if (w == nullptr) {
            return false;
        }
        const tensor_place & tp = tensors[idx];
        int64_t rows = 0, cols = 0;
        card_extent(idx, card, rows, cols);
        if (tp.lay != layout::gguf || !expert_fmt(tp.format, f) ||
            (f != static_cast<int32_t>(fn::experts::fmt::q2_0) &&
             f != static_cast<int32_t>(fn::experts::fmt::iq4_nl)) ||
            rows != D || cols != FFS) {
            err = tname(layer, t) + ": format " + tp.format + " or its card slice is not the shared down EX2 runs";
            return false;
        }
        if (idx >= (int32_t)m.dense_packed.size() || m.dense_packed[idx] != 2) {
            err = tname(layer, t) + ": not in the experts' packed layout (pack_dense)";
            return false;
        }
        return true;
    }
    bool hc(const char * down, const char * up, const char * norm, hc_module & h) const {
        h.w_down = reinterpret_cast<const uint16_t *>(ptr(down)); // BF16 bits (fn::hc layout)
        h.w_up = reinterpret_cast<const uint16_t *>(ptr(up));
        h.gamma = reinterpret_cast<const float *>(ptr(norm)); // F32 GGUF slice: [4][1280]
        return h.w_down != nullptr && h.w_up != nullptr && h.gamma != nullptr;
    }
    // DR1: the draft's Q8_0 modules (the loader's planar hc_down / hc_up fill, hc.h's Q8_0 layout)
    bool hc_q8(const char * down, const char * up, const char * norm, int64_t down_bytes, hc_module & h) const {
        int32_t di = -1, ui = -1;
        h.q8_down = ptr(down, di);
        h.q8_up = ptr(up, ui);
        h.gamma = reinterpret_cast<const float *>(ptr(norm));
        if (h.q8_down == nullptr || h.q8_up == nullptr || h.gamma == nullptr) {
            return false;
        }
        if (std::strcmp(tensors[di].format, "Q8_0") != 0 || tensors[di].lay != layout::hc_down ||
            std::strcmp(tensors[ui].format, "Q8_0") != 0 || tensors[ui].lay != layout::hc_up ||
            m.regions[card][size_t(m.region_of[size_t(di)])].bytes < down_bytes ||
            m.regions[card][size_t(m.region_of[size_t(ui)])].bytes < fn::hc::w_up_q8_bytes) {
            err = tname(layer, down) + ": not in hc.h's Q8_0 layout";
            return false;
        }
        return true;
    }
};

// the card-local shapes the step assumes, checked against the placement
bool check_dense(const dense_weight & d, int64_t k, int64_t n_out, const char * what, std::string & err) {
    if (d.k != k || d.n_out != n_out) {
        err = std::string(what) + ": card slice " + std::to_string(d.n_out) + " x " + std::to_string(d.k) +
              ", the step needs " + std::to_string(n_out) + " x " + std::to_string(k);
        return false;
    }
    return true;
}

int32_t find_part(const reserved_class * rc, const std::string & name) {
    if (rc == nullptr) {
        return -1;
    }
    for (int32_t p = 0; p < (int32_t)rc->plan.parts.size(); ++p) {
        if (rc->plan.parts[p].name == name) {
            return p;
        }
    }
    return -1;
}

void note(const card_ctx & c, const char * piece) {
    if (c.observe != nullptr) {
        c.observe(c.observe_user, c.card, piece);
    }
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// the X-HC / X-HMIX slot of HC sublayer n: n % 3 eagerly; in a graph of graph_subs sublayers (not a multiple of 3),
// HC2's slot_of: the last takes slot 1, so consecutive sublayers differ across replays too
// DR1's draft rounds (hc_rule): a step round of G = 3 d sublayers takes {1, 2, 0}[(G - 1 - k) % 3] (its last slot 1,
// consecutive sublayers differ), a catch-up's one sublayer slot 2
int32_t hc_slot(const use_counters & u, int64_t n) {
    if (!u.graph) {
        return static_cast<int32_t>(n % fn::xchg::n_slot);
    }
    const int64_t k = n % u.graph_subs;
    switch (u.rule) {
        case hc_rule::target:
            break;
        case hc_rule::draft_step: {
            static constexpr int32_t from_end[3] = {1, 2, 0};
            return from_end[(u.graph_subs - 1 - k) % fn::xchg::n_slot];
        }
        case hc_rule::catchup:
            return 2;
        case hc_rule::stage:
            if ((u.graph_subs - 2) % fn::xchg::n_slot == 1) {
                static constexpr int32_t from_end[3] = { 1, 2, 0 };
                return from_end[(u.graph_subs - 1 - k) % fn::xchg::n_slot];
            }
            break;
    }
    return k == u.graph_subs - 1 ? 1 : static_cast<int32_t>(k % fn::xchg::n_slot);
}

// the previous sublayer's slot, which down2 resets: eagerly (n - 1) % 3; a graph's first sublayer resets slot 1, the
// slot every round's graph leaves dirty (hc_rule)
int32_t hc_prev_slot(const use_counters & u, int64_t n) {
    if (!u.graph) {
        return n == 0 ? -1 : static_cast<int32_t>((n - 1) % fn::xchg::n_slot);
    }
    return n % u.graph_subs == 0 || u.rule == hc_rule::catchup ? 1 : hc_slot(u, n - 1);
}

int32_t use_slot(const use_counters & u, int64_t n, int64_t per_graph) {
    return static_cast<int32_t>((u.graph ? n % per_graph : n) % fn::xchg::n_slot);
}

// a BF16 HC sublayer's arguments on card c (hc::sub; ME12's hc::pair takes both cards')
fn::hc::sub_args hc_sub_args(const card_ctx & c, const hc_module & h, int32_t set, const float * hres,
                             const float * hpost, const float * hx, const use_counters & u) {
    const int32_t r = c.card;
    const int32_t s = hc_slot(u, u.sub);
    const int32_t s_prev = hc_prev_slot(u, u.sub);
    const fn::xchg::rings & rg = *c.rings;
    fn::hc::sub_args a = {};
    a.w_down = h.w_down;
    a.w_up = h.w_up;
    a.hres = hres;
    a.hx = hx;
    a.hpost = hpost;
    a.gamma = h.gamma;
    a.x = c.ws.x[set];
    a.inject = c.ws.inject[set];
    a.hc_ring = rg.slot(r, fn::xchg::msg::hc, s);
    a.hc_peer = rg.slot(1 - r, fn::xchg::msg::hc, s);
    a.hc_reset = s_prev >= 0 ? rg.slot(r, fn::xchg::msg::hc, s_prev) : nullptr;
    a.hmix_reset = s_prev >= 0 ? rg.slot(r, fn::xchg::msg::hmix, s_prev) : nullptr;
    a.hmix_own = rg.slot(r, fn::xchg::msg::hmix, s);
    a.hmix_peer = rg.slot(1 - r, fn::xchg::msg::hmix, s);
    a.err = rg.err(r);
    a.card = r;
    return a;
}

// one HC sublayer's kernels, part 0 or 1: x = hres + hpost * hx, then the mixed input in both forms into the
// sublayer's X-HMIX slot on both cards (HC3); the body's first launches acquire it (hmix_in). BF16 weights (HC4):
// fused, one launch of hc::sub at part 0 (the graphs); otherwise its down phase at part 0 and its up phase at part 1
// (the pair order, eager or one device's graphs: every send precedes its read). Q8_0 weights (DR1): down2_q8, up2_q8.
bool hc_phase(card_ctx & c, const hc_module & h, int32_t set, const float * hres, const float * hpost, const float * hx,
              int32_t rows, const use_counters & u, int32_t part, bool fused, std::string & err) {
    const int32_t r = c.card;
    const int64_t n = u.sub;
    const int32_t s = hc_slot(u, n);
    const fn::xchg::rings & rg = *c.rings;
    if (h.q8_down == nullptr) {
        if (fused && part == 1) {
            return true;
        }
        const fn::hc::sub_args a = hc_sub_args(c, h, set, hres, hpost, hx, u);
        if (!fn::hc::sub(rows, fused ? 3 : part + 1, a, c.stream)) {
            return fail(err, "hc sub");
        }
        note(c, fused ? "hc.sub" : part == 0 ? "hc.sub.down" : "hc.sub.up");
        return true;
    }
    if (part == 0) {
        const int32_t s_prev = hc_prev_slot(u, n);
        fn::hc::down2_args d2;
        d2.d = {h.w_down, hres, hx, hpost, h.gamma, c.ws.x[set], c.ws.mom[set]};
        d2.hc_peer = rg.slot(1 - r, fn::xchg::msg::hc, s);
        d2.hc_reset = s_prev >= 0 ? rg.slot(r, fn::xchg::msg::hc, s_prev) : nullptr;
        d2.hmix_reset = s_prev >= 0 ? rg.slot(r, fn::xchg::msg::hmix, s_prev) : nullptr;
        const bool ok = fn::hc::down2_q8(rows, {d2, h.q8_down}, c.stream);
        if (!ok) {
            return fail(err, "hc down2");
        }
        note(c, "hc.down2");
        return true;
    }
    const fn::hc::up2_args u2 = {h.w_up,
                                 c.ws.mom[set],
                                 rg.slot(r, fn::xchg::msg::hc, s),
                                 c.ws.x[set],
                                 h.gamma,
                                 c.ws.inject[set],
                                 rg.slot(r, fn::xchg::msg::hmix, s),
                                 rg.slot(1 - r, fn::xchg::msg::hmix, s),
                                 c.ws.hc_runs,
                                 rg.err(r),
                                 r};
    const bool ok = fn::hc::up2_q8(rows, {u2, h.q8_up}, c.stream);
    if (!ok) {
        return fail(err, "hc up2");
    }
    note(c, "hc.up2");
    return true;
}

// ME12: both logical cards' BF16 HC sublayer in one launch (hc::pair), when they share one device and one stream:
// it replaces their down and up phases (the pair order's four launches). The common intermediate's scratch is card 0's
// ws.mom[set], which the BF16 producer does not otherwise use (the Q8_0 down2 writes it); the launch forms and reads it
// on the stream, so nothing before or after it on any stream sees it.
static_assert(fn::hc::pair_scratch_bytes <= int64_t(layer_max_rows) * fn::hc::n_mom * 4, "ME12: the scratch fits mom");

bool hc_paired(card_ctx * const c[2], const hc_module & h0, const hc_module & h1) {
    return c[0]->device == c[1]->device && c[0]->stream != nullptr && c[0]->stream == c[1]->stream &&
           h0.q8_down == nullptr && h1.q8_down == nullptr;
}

// sublayer set's HC input on card c: the layer's input (attention), or the attention sublayer's streams, injection and
// output half (FFN)
layer_input hc_input(const card_ctx & c, const layer_input & in, int32_t set) {
    return set == 0 ? in : layer_input{c.ws.x[0], c.ws.inject[0], c.ws.out[0]};
}

bool hc_pair_phase(card_ctx * const c[2], const hc_module * const h[2], int32_t set, const layer_input in[2],
                   int32_t rows, const use_counters u[2], std::string & err) {
    fn::hc::pair_args a = {};
    for (int32_t k = 0; k < 2; ++k) {
        const layer_input hi = hc_input(*c[k], in[k], set);
        a.s[k] = hc_sub_args(*c[k], *h[k], set, hi.hres, hi.hpost, hi.hx, u[k]);
    }
    a.scratch = c[0]->ws.mom[set];
    if (!cuda_ok(cudaSetDevice(c[0]->device), "cudaSetDevice", err)) {
        return false;
    }
    if (!fn::hc::pair(rows, a, c[0]->stream)) {
        return fail(err, "hc pair");
    }
    note(*c[0], "hc.pair");
    return true;
}

// the current sublayer's X-HMIX slot on this card: the body's input (xchg.h; HC3)
float * hmix_in(const card_ctx & c, const use_counters & u) {
    return c.rings->slot(c.card, fn::xchg::msg::hmix, hc_slot(u, u.sub));
}

// model-test's record of a sublayer's F32 mixed input, after its body (card_ctx::record_mixed)
bool record_mixed(card_ctx & c, int32_t set, int32_t rows, const use_counters & u, std::string & err) {
    if (c.record_mixed[set] == nullptr) {
        return true;
    }
    return cuda_ok(cudaMemcpyAsync(c.record_mixed[set], hmix_in(c, u), size_t(rows) * D * 4, cudaMemcpyDefault,
                                   static_cast<cudaStream_t>(c.stream)),
                   "record mixed", err);
}

// the sublayer's destination-half exchange: send the peer's half of ws.partial, or receive and add ours into
// ws.out[set]
bool exchange_half(card_ctx & c, int32_t set, int32_t rows, int32_t slot, bool send, std::string & err) {
    const int32_t r = c.card;
    const fn::xchg::msg m = set == 0 ? fn::xchg::msg::a : fn::xchg::msg::f;
    const uint32_t code = set == 0 ? fn::xchg::err_poll_a : fn::xchg::err_poll_f;
    if (send) {
        if (!fn::xchg::send_half(rows, 1 - r, c.ws.partial, c.rings->slot(1 - r, m, slot), c.stream)) {
            return fail(err, "send_half");
        }
        note(c, set == 0 ? "x.a.send" : "x.f.send");
        return true;
    }
    if (!fn::xchg::recv_add(rows, r, c.ws.partial, c.rings->slot(r, m, slot), c.ws.out[set], c.rings->err(r), code,
                            c.stream)) {
        return fail(err, "recv_add");
    }
    note(c, set == 0 ? "x.a.recv" : "x.f.recv");
    return true;
}

// The pieces of each body FNFL has not built, in the body's order (AUD1 Amendment 1 review, section 2, has the design
// interface of each). A body runs nothing while one is listed: each feeds the next. LP1 built the last of them.
constexpr std::array<const char *, 0> gdn_unbuilt{};
constexpr std::array<const char *, 0> qsa_unbuilt{};
constexpr std::array<const char *, 0> ffn_unbuilt{};

enum class body : int32_t { gdn, qsa, ffn };

std::vector<std::string> body_unbuilt(body b) {
    const auto list = [](const auto & pieces) {
        return std::vector<std::string>(std::begin(pieces), std::end(pieces));
    };
    switch (b) {
        case body::gdn:
            return list(gdn_unbuilt);
        case body::qsa:
            return list(qsa_unbuilt);
        case body::ffn:
            return list(ffn_unbuilt);
    }
    return {};
}

// ---- the bodies (LP1) ----

// a quantized matrix of a product launch, F32 output
fn::dense::mat qmat(const dense_weight & d, float * out) {
    fn::dense::mat m;
    m.f = static_cast<fn::dense::fmt>(d.fmt);
    m.m = static_cast<int32_t>(d.n_out);
    m.w = d.w;
    m.out = out;
    return m;
}

// a BF16 matrix (GGUF rows, read by DP2 unchanged), F32 output
fn::dense::mat bmat(const uint16_t * w, int32_t rows, float * out) {
    fn::dense::mat m;
    m.f = fn::dense::fmt::bf16;
    m.m = rows;
    m.w = reinterpret_cast<const uint8_t *>(w); // BF16 bits
    m.out = out;
    return m;
}

// DP2's launch with the card's grid table in its site's work region, its layout checked against the site's
bool dense_launch(card_ctx & c, fn::dense::product_args a, int32_t rows, dense_site site, const char * piece,
                  std::string & err) {
    a.grids = c.dense_grids;
    std::string why;
    if (!dense_site_launch(c.dense_work, c.dense_sites, site, a, rows, c.stream, why)) {
        return fail(err, std::string(piece) + ": " + why);
    }
    note(c, piece);
    return true;
}

// a body's first launch reads the mixed input from X-HMIX, acquiring the peer's half (dense.h, HC3)
void acquire_input(fn::dense::product_args & a, const card_ctx & c, const use_counters & u) {
    a.hmix = hmix_in(c, u);
    a.hmix_peer_half = 1 - c.card;
    a.err = c.rings->err(c.card);
}

// GDN: DP2's input launch (qkv with the convolution epilogue, z, alpha, beta; it acquires X-HMIX), GD1's gdn_core,
// DP2's ssm_out into the X-A partial
bool gdn_body(card_ctx & c, const layer_weights & w, const layer_state & st, int32_t rows, const use_counters & u,
              std::string & err) {
    layer_ws & ws = c.ws;
    fn::dense::product_args in;
    in.k = D;
    in.n_mats = 4;
    in.mats[0] = qmat(w.qkv, ws.conv);
    in.mats[0].e = fn::dense::epi::conv_silu;
    in.mats[1] = qmat(w.z, ws.z);
    in.mats[2] = bmat(w.alpha, GH, ws.alpha);
    in.mats[3] = bmat(w.beta, GH, ws.beta);
    acquire_input(in, c, u);
    in.conv.w = w.conv1d;
    in.conv.hist = st.conv;
    in.conv.cur = c.cur;
    if (!dense_launch(c, in, rows, dense_site::gdn_input, "gdn.input", err)) {
        return false;
    }
    fn::gdn::core_args g;
    g.conv = ws.conv;
    g.z = ws.z;
    g.alpha = ws.alpha;
    g.beta = ws.beta;
    g.dt = w.dt;
    g.a = w.a;
    g.norm_w = w.ssm_norm;
    g.state = st.gdn;
    g.rec = st.rec;
    g.gsel = c.gsel;
    g.gated = ws.gated;
    g.gated_q = {ws.mid_codes, ws.mid_scales, rows, GV};
    g.ws = c.gdn_ws;
    g.rows = rows;
    if (!fn::gdn::core(g, c.stream)) {
        return fail(err, "gdn core");
    }
    note(c, "gdn.core");
    fn::dense::product_args out;
    out.k = GV;
    out.n_mats = 1;
    out.mats[0] = qmat(w.ssm_out, ws.partial);
    out.x_codes = ws.mid_codes;
    out.x_scales = ws.mid_scales;
    return dense_launch(c, out, rows, dense_site::gdn_out, "gdn.out", err);
}

// QSA: DP2's input launch (q with its gate, k, v; the index query and key on the index owner); on the owner
// summary_update with the raw index-key write, select and X-SELECT to the peer, on the peer the X-SELECT receive; QA2's
// attend on both cards; DP2's wo into the X-A partial
bool qsa_body(card_ctx & c, const layer_weights & w, const layer_state & st, int32_t rows, const use_counters & u,
              std::string & err) {
    layer_ws & ws = c.ws;
    const int32_t r = c.card;
    const bool owner = r == w.index_owner;
    const int32_t sel_slot = use_slot(u, u.select, u.graph_select);
    fn::dense::product_args in;
    in.k = D;
    in.n_mats = owner ? 5 : 3;
    in.mats[0] = qmat(w.wq, ws.qfull);
    in.mats[1] = qmat(w.wk, ws.kv_k);
    in.mats[2] = qmat(w.wv, ws.kv_v);
    if (owner) {
        in.mats[3] = bmat(w.idx_q, IQ, ws.iq);
        in.mats[4] = bmat(w.idx_k, fn::qsa::head_dim, ws.ik);
    }
    acquire_input(in, c, u);
    if (!dense_launch(c, in, rows, owner ? dense_site::qsa_input_owner : dense_site::qsa_input_peer, "qsa.input",
                      err)) {
        return false;
    }
    if (owner) {
        fn::qsa::round_update_args ua;
        ua.keys = st.idx_keys;
        ua.k_norm = w.idx_k_norm;
        ua.summaries = st.idx_summaries;
        ua.rope = fn::qsa::model_rope();
        ua.ik = ws.ik;
        ua.depth = c.n_before;
        ua.rows = rows;
        ua.max_cells = st.n_cells;
        if (c.kv != nullptr) { // ME6: the host-backed tier's key window (kv_tier.h)
            ua.keys = c.kv->keys(r, st.kv_attn);
            ua.shift = c.kv->shift(r);
        }
        if (!fn::qsa::summary_update(ua, c.stream)) {
            return fail(err, "summary_update");
        }
        note(c, "qsa.summary");
        fn::qsa::select_args sa;
        sa.summaries = st.idx_summaries;
        sa.q = ws.iq;
        sa.q_norm = w.idx_q_norm;
        sa.depth = c.n_before;
        sa.max_cells = st.n_cells;
        sa.rope = fn::qsa::model_rope();
        sa.ws = fn::qsa::carve(c.qsa_ws);
        sa.body = ws.sel_body;
        sa.offsets = ws.sel_off;
        if (!fn::qsa::select(rows, sa, c.stream)) {
            return fail(err, "qsa select");
        }
        note(c, "qsa.select");
        if (!fn::xchg::send_select(rows, ws.sel_off, ws.sel_body, c.rings->slot(1 - r, fn::xchg::msg::select, sel_slot),
                                   c.stream)) {
            return fail(err, "send_select");
        }
        note(c, "x.select.send");
    } else {
        if (!fn::xchg::recv_select(rows, c.rings->slot(r, fn::xchg::msg::select, sel_slot), ws.sel_off, ws.sel_body,
                                   c.rings->err(r), c.stream)) {
            return fail(err, "recv_select");
        }
        note(c, "x.select.recv");
    }
    fn::qsa::attn::args aa = {};
    aa.qfull = ws.qfull;
    aa.kv_k = ws.kv_k;
    aa.kv_v = ws.kv_v;
    aa.q_norm = w.q_norm;
    aa.k_norm = w.k_norm;
    aa.body = ws.sel_body;
    aa.offsets = ws.sel_off;
    aa.pos = c.pos;
    aa.k_cache = st.k_cache;
    aa.v_cache = st.v_cache;
    aa.k_scale = st.k_scale; // KV8: set in q8
    aa.v_scale = st.v_scale;
    if (c.kv != nullptr) {
        // ME1: the host-backed KV: the window's pointers and the body with the older blocks gathered (kv_tier.h)
        if (!c.kv->select(r, st.kv_attn, rows, ws.sel_body, ws.sel_off, c.rings->err(r), c.stream, aa.body, err)) {
            return false;
        }
        note(c, "kv.gather");
        aa.k_cache = c.kv->k(r, st.kv_attn);
        aa.v_cache = c.kv->v(r, st.kv_attn);
        aa.shift = c.kv->shift(r); // ME3: the graphs' window start (null for eager)
    }
    aa.depth = c.n_before;
    aa.max_cells = st.n_cells;
    aa.theta_scale = fn::qsa::attn::rope_theta_scale();
    aa.out = ws.attn;
    aa.codes = ws.mid_codes;
    aa.scales = ws.mid_scales;
    aa.k_f32 = nullptr;
    aa.stamps = nullptr;
    aa.ws = fn::qsa::attn::carve(c.attn_ws);
    if (!fn::qsa::attn::attend(rows, aa, c.stream)) {
        return fail(err, "qsa attend");
    }
    note(c, "qsa.attend");
    fn::dense::product_args out;
    out.k = GV;
    out.n_mats = 1;
    out.mats[0] = qmat(w.wo, ws.partial);
    out.x_codes = ws.mid_codes;
    out.x_scales = ws.mid_scales;
    return dense_launch(c, out, rows, dense_site::qsa_out, "qsa.out", err);
}

// FFN on each card (XP1's V2): DP2's router with its top-10 epilogue and the shared gate's sigmoid, then EX2's experts:
// gate_up runs the routed experts and the shared expert's units, down their weighted sum with the shared term (sg)
// added last into the X-F partial (R2's F7). The router (F32 row) and gate_up (codes and scales, the slot's last
// reader) each acquire X-HMIX (HC3).
bool ffn_router(card_ctx & c, const layer_weights & w, int32_t rows, const use_counters & u, std::string & err) {
    layer_ws & ws = c.ws;
    fn::dense::product_args rt;
    rt.k = D;
    rt.n_mats = 2;
    rt.mats[0] = bmat(w.router, NE, ws.logits);
    rt.mats[1] = bmat(w.sh_gate_w, 1, ws.sg_logit);
    acquire_input(rt, c, u);
    rt.route.logits = 0;
    rt.route.gate = 1;
    rt.route.ids = c.ex.ids;
    rt.route.wts = c.ex.wts;
    rt.route.sg = ws.sg;
    return dense_launch(c, rt, rows, dense_site::router, "ffn.router", err);
}

// the experts' launches of the FFN as the resident banks run them (ME1's cache sets their tables)
// ME8: rv, another logical card's routing on this device and stream (route_view), in place of c's own
void expert_args(const card_ctx & c, const layer_weights & w, int32_t rows, const use_counters & u,
                 fn::experts::gate_up_args & ga, fn::experts::down_args & da, const route_view * rv = nullptr) {

    const fn::act::out_view glu = {c.ex.glu_codes, c.ex.glu_scales, fn::experts::glu_slots, FFS};
    const fn::act::view glu_in = {c.ex.glu_codes, c.ex.glu_scales, fn::experts::glu_slots, FFS};
    ga = {w.ex_gate_up, w.sh_gate.w, w.sh_up.w, c.grids, {nullptr, nullptr, rows, D},
          c.ex.ids,     c.ex.meta,   glu,       nullptr, c.ex.status};
    ga.cw = fn::experts::class_weights(static_cast<fn::experts::fmt>(w.ex_gu_fmt),
                                       static_cast<fn::dense::fmt>(w.sh_gate.fmt),
                                       static_cast<fn::dense::fmt>(w.sh_up.fmt)); // XB3: the unit order's weights
    ga.hmix = hmix_in(c, u);
    ga.hmix_peer_half = 1 - c.card;
    ga.err = c.rings->err(c.card);
    ga.map = c.ex_map; // SP4
    da = {w.ex_down, w.ex_sh_down, c.grids, glu_in, c.ex.ids, c.ex.meta, c.ex.wts, c.ws.sg, c.ws.partial, c.ex.status};
    if (rv != nullptr) {
        fn::experts::use_route(*rv, ga, da);
    }
}

// ME8: both logical cards on one device and one compute stream (the single-device executor's placement), so one
// card's routing, written earlier in that stream, is readable by the other's experts
bool same_stream(const card_ctx & a, const card_ctx & b) {
    return a.device == b.device && a.stream == b.stream && a.stream != nullptr;
}

// the card's routing as a view (ME8)
route_view route_of(const card_ctx & c) {
    return {c.ex.ids, c.ex.wts, c.ws.sg};
}

// FFN on each card (XP1's V2): DP2's router with its top-10 epilogue and the shared gate's sigmoid, then EX2's experts:
// gate_up runs the routed experts and the shared expert's units, down their weighted sum with the shared term (sg)
// added last into the X-F partial (R2's F7). The router (F32 row) and gate_up (codes and scales, the slot's last
// reader) each acquire X-HMIX (HC3).
// ME8: the experts of the body alone, on rv's routing (null: c's own router's)
bool ffn_experts(card_ctx & c, const layer_weights & w, int32_t rows, const use_counters & u, const route_view * rv,
                 std::string & err) {
    fn::experts::gate_up_args ga;
    fn::experts::down_args da;
    expert_args(c, w, rows, u, ga, da, rv);
    if (c.ex_service != nullptr) { // ME3: a two-device graph capture, this card's part of the cache's window
        return c.ex_service->ffn_card(c, w, rows, ga, da, err); // (it notes ffn.ex_gate_up and ffn.ex_down)
    }
    if (!fn::experts::gate_up(static_cast<fn::experts::fmt>(w.ex_gu_fmt), static_cast<fn::dense::fmt>(w.sh_gate.fmt),
                              static_cast<fn::dense::fmt>(w.sh_up.fmt), rows, ga, c.stream)) {
        return fail(err, "experts gate_up");
    }
    note(c, "ffn.ex_gate_up");
    if (!fn::experts::down(static_cast<fn::experts::fmt>(w.ex_down_fmt),
                           static_cast<fn::experts::fmt>(w.ex_sh_down_fmt), rows, da, c.stream)) {
        return fail(err, "experts down");
    }
    note(c, "ffn.ex_down");
    return true;
}

bool ffn_body(card_ctx & c, const layer_weights & w, int32_t rows, const use_counters & u, std::string & err) {
    return ffn_router(c, w, rows, u, err) && ffn_experts(c, w, rows, u, nullptr, err);
}

// ME1: both cards' FFN bodies through the expert cache: both routers, then the service's launches. ME8: on one device
// and stream, card 0's router alone; card 1's experts read its routing (route_view)
step_status ffn_pair(card_ctx * const c[2], const layer_weights * const w[2], int32_t rows, const use_counters u[2],
                     std::string & err) {
    fn::experts::gate_up_args ga[2];
    fn::experts::down_args da[2];
    const bool one = same_stream(*c[0], *c[1]);
    const route_view r0 = route_of(*c[0]);
    for (int32_t k = 0; k < 2; ++k) {
        if (!cuda_ok(cudaSetDevice(c[k]->device), "cudaSetDevice", err) ||
            ((k == 0 || !one) && !ffn_router(*c[k], *w[k], rows, u[k], err))) {
            err = "card " + std::to_string(k) + ": " + err;
            return step_status::failed;
        }
        expert_args(*c[k], *w[k], rows, u[k], ga[k], da[k], k == 1 && one ? &r0 : nullptr);
    }
    if (!c[0]->ex_service->ffn_pair(c, w, rows, ga, da, err) || !record_mixed(*c[0], 1, rows, u[0], err) ||
        !record_mixed(*c[1], 1, rows, u[1], err)) {
        return step_status::failed;
    }
    return step_status::ok;
}

// ME8: both cards' resident FFN bodies on one device and stream: card 0's router, then card 0's experts and card 1's
// on its routing (route_view), each body's mixed record as run_body's
step_status ffn_pair_resident(card_ctx * const c[2], const layer_weights * const w[2], int32_t rows,
                              const use_counters u[2], std::string & err) {
    const route_view r0 = route_of(*c[0]);
    for (int32_t k = 0; k < 2; ++k) {
        const bool ok = cuda_ok(cudaSetDevice(c[k]->device), "cudaSetDevice", err) &&
                        (k == 1 || ffn_router(*c[0], *w[0], rows, u[0], err)) &&
                        ffn_experts(*c[k], *w[k], rows, u[k], k == 1 ? &r0 : nullptr, err) &&
                        record_mixed(*c[k], 1, rows, u[k], err);
        if (!ok) {
            err = "card " + std::to_string(k) + ": " + err;
            return step_status::failed;
        }
    }
    return step_status::ok;
}

// a sublayer's body; not_built while a piece is unbuilt
step_status run_body(body b, card_ctx & c, const layer_weights & w, const layer_state & st, int32_t rows,
                     const use_counters & u, std::string & err) {
    const std::vector<std::string> missing = body_unbuilt(b);
    if (!missing.empty()) {
        err = std::string(b == body::gdn ? "gdn" : b == body::qsa ? "qsa" : "ffn") + " body not built:";
        for (const std::string & m : missing) {
            err += "\n  " + m;
        }
        return step_status::not_built;
    }
    const bool ok = (b == body::gdn   ? gdn_body(c, w, st, rows, u, err)
                     : b == body::qsa ? qsa_body(c, w, st, rows, u, err)
                                      : ffn_body(c, w, rows, u, err)) &&
                    record_mixed(c, b == body::ffn ? 1 : 0, rows, u, err);
    return ok ? step_status::ok : step_status::failed;
}

} // namespace

bool one_router(const card_ctx & c0, const card_ctx & c1) {
    return same_stream(c0, c1);
}

int32_t hc_slot_of(const use_counters & u, int64_t n) {
    return hc_slot(u, n);
}

int32_t hc_prev_slot_of(const use_counters & u, int64_t n) {
    return hc_prev_slot(u, n);
}

int32_t use_slot_of(const use_counters & u, int64_t n, int32_t per_graph) {
    return use_slot(u, n, per_graph);
}

bool hc_sublayer(card_ctx & c, const hc_module & h, int32_t set, const layer_input & in, int32_t rows,
                 const use_counters & u, int32_t part, std::string & err) {
    return hc_phase(c, h, set, in.hres, in.hpost, in.hx, rows, u, part, false, err);
}

float * hmix_slot(const card_ctx & c, const use_counters & u) {
    return hmix_in(c, u);
}

bool dense_run(card_ctx & c, fn::dense::product_args a, int32_t rows, dense_site site, bool acquire,
               const use_counters & u, const char * piece, std::string & err) {
    if (acquire) {
        acquire_input(a, c, u);
    }
    return dense_launch(c, a, rows, site, piece, err);
}

bool hc_sublayer_lanes(card_ctx * const c[], int32_t lanes, const hc_module & h, int32_t set, const layer_input in[],
                       const use_counters & u, int32_t part, uint32_t * runs, std::string & err) {
    if (lanes < 2 || lanes > fn::xchg::max_lanes || h.q8_down == nullptr || runs == nullptr) {
        return fail(err, "hc lanes: 2 .. 4 lanes of a Q8_0 sublayer with run counters");
    }
    const int64_t n = u.sub;
    const int32_t s = hc_slot(u, n);
    const int32_t s_prev = hc_prev_slot(u, n);
    fn::hc::down2_q8_args d[fn::xchg::max_lanes] = {};
    fn::hc::up2_q8_args up[fn::xchg::max_lanes] = {};
    for (int32_t k = 0; k < lanes; ++k) {
        const card_ctx & ck = *c[k];
        const int32_t r = ck.card;
        const fn::xchg::rings & rg = *ck.rings;
        if (r != c[0]->card || ck.stream != c[0]->stream) {
            return fail(err, "hc lanes: the lanes on one card and stream");
        }
        fn::hc::down2_args d2;
        d2.d = {h.w_down, in[k].hres, in[k].hx, in[k].hpost, h.gamma, ck.ws.x[set], ck.ws.mom[set]};
        d2.hc_peer = rg.slot(1 - r, fn::xchg::msg::hc, s);
        d2.hc_reset = s_prev >= 0 ? rg.slot(r, fn::xchg::msg::hc, s_prev) : nullptr;
        d2.hmix_reset = s_prev >= 0 ? rg.slot(r, fn::xchg::msg::hmix, s_prev) : nullptr;
        d[k] = {d2, h.q8_down};
        const fn::hc::up2_args u2 = {h.w_up,
                                     ck.ws.mom[set],
                                     rg.slot(r, fn::xchg::msg::hc, s),
                                     ck.ws.x[set],
                                     h.gamma,
                                     ck.ws.inject[set],
                                     rg.slot(r, fn::xchg::msg::hmix, s),
                                     rg.slot(1 - r, fn::xchg::msg::hmix, s),
                                     runs + int64_t(k) * fn::hc::n_runs_card,
                                     rg.err(r),
                                     r};
        up[k] = {u2, h.q8_up};
    }
    const bool ok =
        part == 0 ? fn::hc::down2_q8_lanes(lanes, d, c[0]->stream) : fn::hc::up2_q8_lanes(lanes, up, c[0]->stream);
    if (!ok) {
        return fail(err, part == 0 ? "hc down2 lanes" : "hc up2 lanes");
    }
    note(*c[0], part == 0 ? "hc.down2.lanes" : "hc.up2.lanes");
    return true;
}

bool dense_joint_run(card_ctx * const c[], int32_t lanes, fn::dense::product_args a, const fn::xchg::joint_rings & jr,
                     void * work, const use_counters & u, const char * piece, std::string & err) {
    if (lanes < 2 || lanes > fn::xchg::max_lanes || work == nullptr) {
        return fail(err, std::string(piece) + ": 2 .. 4 lanes with the joint work");
    }
    const card_ctx & c0 = *c[0];
    fn::dense::joint_args ja;
    ja.lanes = lanes;
    ja.hmix_lane_floats = static_cast<int32_t>(fn::xchg::joint_rings::lane_floats());
    a.hmix = hmix_in(c0, u);
    a.hmix_peer_half = 1 - c0.card;
    a.err = jr.err(c0.card);
    for (int32_t k = 1; k < lanes; ++k) {
        if (hmix_in(*c[k], u) != a.hmix + int64_t(k) * ja.hmix_lane_floats || c[k]->card != c0.card) {
            return fail(err, std::string(piece) + ": the lanes' X-HMIX slots are not the joint rings' lane slots");
        }
    }
    a.grids = c0.dense_grids;
    fn::dense::choose_splits(a);
    a.work = work;
    if (!fn::dense::product_joint(a, ja, c0.stream)) {
        return fail(err, std::string(piece) + ": joint product");
    }
    note(c0, piece);
    return true;
}

bool half_exchange(card_ctx & c, int32_t set, int32_t rows, int32_t slot, bool send, std::string & err) {
    return exchange_half(c, set, rows, slot, send, err);
}

void note_piece(const card_ctx & c, const char * piece) {
    note(c, piece);
}

std::vector<std::string> unbuilt_pieces(int32_t layer) {
    const bool gdn = layer % fn::full_attn_interval != fn::full_attn_interval - 1;
    std::vector<std::string> out = body_unbuilt(gdn ? body::gdn : body::qsa);
    const std::vector<std::string> ffn = body_unbuilt(body::ffn);
    out.insert(out.end(), ffn.begin(), ffn.end());
    return out;
}

bool resolve_weights(const model & m, int32_t layer, int32_t card, layer_weights & out, std::string & err) {
    out = layer_weights();
    out.layer = layer;
    out.gdn = layer % fn::full_attn_interval != fn::full_attn_interval - 1;
    out.index_owner = (layer / fn::full_attn_interval) % 2;
    out.router_owner = layer % 2;
    const finder f{m, layer, card, err};
    if (!f.hc("hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_norm.weight", out.hc_attn) ||
        !f.hc("hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_norm.weight", out.hc_ffn)) {
        return false;
    }
    if (out.gdn) {
        if (!f.dense("attn_qkv.weight", out.qkv) || !f.dense("attn_gate.weight", out.z) ||
            !f.dense("ssm_out.weight", out.ssm_out) || !check_dense(out.qkv, D, GC, "gdn.qkv", err) ||
            !check_dense(out.z, D, GV, "gdn.z", err) || !check_dense(out.ssm_out, GV, D, "gdn.out", err)) {
            return false;
        }
        out.alpha = reinterpret_cast<const uint16_t *>(f.ptr("ssm_alpha.weight")); // BF16 bits
        out.beta = reinterpret_cast<const uint16_t *>(f.ptr("ssm_beta.weight"));
        out.conv1d = reinterpret_cast<const float *>(f.ptr("ssm_conv1d.weight"));
        out.dt = reinterpret_cast<const float *>(f.ptr("ssm_dt.bias"));
        out.a = reinterpret_cast<const float *>(f.ptr("ssm_a"));
        out.ssm_norm = reinterpret_cast<const float *>(f.ptr("ssm_norm.weight"));
        if (out.alpha == nullptr || out.beta == nullptr || out.conv1d == nullptr || out.dt == nullptr ||
            out.a == nullptr || out.ssm_norm == nullptr) {
            return false;
        }
    } else {
        if (!f.dense("attn_q.weight", out.wq) || !f.dense("attn_k.weight", out.wk) ||
            !f.dense("attn_v.weight", out.wv) || !f.dense("attn_output.weight", out.wo) ||
            !check_dense(out.wq, D, QF, "qsa.q", err) || !check_dense(out.wk, D, HD, "qsa.k", err) ||
            !check_dense(out.wv, D, HD, "qsa.v", err) || !check_dense(out.wo, int64_t(QH) * HD, D, "qsa.out", err)) {
            return false;
        }
        out.q_norm = reinterpret_cast<const float *>(f.ptr("attn_q_norm.weight"));
        out.k_norm = reinterpret_cast<const float *>(f.ptr("attn_k_norm.weight"));
        if (out.q_norm == nullptr || out.k_norm == nullptr) {
            return false;
        }
        if (card == out.index_owner) {
            out.idx_q = reinterpret_cast<const uint16_t *>(f.ptr("indexer.q_proj.weight")); // BF16 bits
            out.idx_k = reinterpret_cast<const uint16_t *>(f.ptr("indexer.k_proj.weight"));
            out.idx_q_norm = reinterpret_cast<const float *>(f.ptr("indexer.q_norm.weight"));
            out.idx_k_norm = reinterpret_cast<const float *>(f.ptr("indexer.k_norm.weight"));
            if (out.idx_q == nullptr || out.idx_k == nullptr || out.idx_q_norm == nullptr ||
                out.idx_k_norm == nullptr) {
                return false;
            }
        }
    }
    // the router and the shared gate on both cards (XP1's V2; the placement holds them on both since LP1)
    out.router = reinterpret_cast<const uint16_t *>(f.ptr("ffn_gate_inp.weight")); // BF16 bits
    out.sh_gate_w = reinterpret_cast<const uint16_t *>(f.ptr("ffn_gate_inp_shexp.weight"));
    if (out.router == nullptr || out.sh_gate_w == nullptr) {
        return false;
    }
    if (!f.dense("ffn_gate_shexp.weight", out.sh_gate) || !f.dense("ffn_up_shexp.weight", out.sh_up) ||
        !check_dense(out.sh_gate, D, FFS, "ffn.sh_gate", err) || !check_dense(out.sh_up, D, FFS, "ffn.sh_up", err) ||
        !f.shared_down("ffn_down_shexp.weight", out.ex_sh_down, out.ex_sh_down_fmt)) {
        return false;
    }
    int32_t gi = -1, di = -1;
    out.ex_gate_up = f.ptr("ffn_gate_exps.weight", gi);
    out.ex_down = f.ptr("ffn_down_exps.weight", di);
    // ME1: the expert cache holds the routed experts (null banks; its service supplies their addresses)
    const auto cached = [&](int32_t t) {
        return t >= 0 && m.region_of[size_t(t)] >= 0 && m.regions[card][m.region_of[size_t(t)]].cached;
    };
    if (m.expert_cache && cached(gi) && cached(di)) {
        err.clear();
    } else if (out.ex_gate_up == nullptr || out.ex_down == nullptr) {
        return false;
    }
    if (!expert_fmt(tensors[gi].format, out.ex_gu_fmt) || !expert_fmt(tensors[di].format, out.ex_down_fmt) ||
        tensors[gi].lay != layout::ex1_gate || tensors[di].lay != layout::ex1_down) {
        return fail(err, tname(layer, "ffn experts") + ": not in EX1's packed layout and formats");
    }
    return true;
}

bool pack_dense_tensor(model & m, int32_t idx, std::string & err) {
    if (idx < 0 || idx >= n_tensors) {
        return fail(err, "pack: no such tensor");
    }
    if (m.dense_packed.size() != size_t(n_tensors)) {
        m.dense_packed.assign(size_t(n_tensors), 0);
    }
    if (m.dense_packed[size_t(idx)] != 0) {
        return true;
    }
    const tensor_place & tp = tensors[idx];
    int32_t f = -1;
    if (tp.lay != layout::gguf || !dense_fmt(tp.format, f)) {
        return fail(err, std::string(tp.name) + ": format " + tp.format + " is not a dense format the step runs");
    }
    std::vector<uint8_t> src, dst;
    for (int32_t card = 0; card < 2; ++card) {
        uint8_t * p = m.region_ptr(idx, card);
        int64_t rows = 0, cols = 0;
        card_extent(idx, card, rows, cols);
        const int64_t rb = cols / tp.block_elems * tp.block_bytes;
        if (p == nullptr || rows <= 0 || rows > INT32_MAX || cols > fn::dense::max_k ||
            rb != fn::dense::row_bytes(static_cast<fn::dense::fmt>(f), static_cast<int32_t>(cols)) ||
            rows * rb != tp.card[card].bytes) {
            return fail(err, std::string(tp.name) + ": card " + std::to_string(card) + " slice is not DP2's shape");
        }
        const size_t n = static_cast<size_t>(rows * rb);
        src.resize(n);
        dst.resize(n);
        if (!cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemcpy(src.data(), p, n, cudaMemcpyDeviceToHost), "pack read", err)) {
            return false;
        }
        if (!fn::dense::pack(static_cast<fn::dense::fmt>(f), static_cast<int32_t>(rows), static_cast<int32_t>(cols),
                             src.data(), static_cast<size_t>(rb), dst.data())) {
            return fail(err, std::string(tp.name) + ": pack");
        }
        if (!cuda_ok(cudaMemcpy(p, dst.data(), n, cudaMemcpyHostToDevice), "pack write", err)) {
            return false;
        }
    }
    m.dense_packed[size_t(idx)] = 1;
    return true;
}

namespace {

// EX2: the shared expert's down on each card ([2560 rows][its 320 columns], GGUF blocks) into fn::experts::pack_down's
// layout in place, as one routed expert's; m.dense_packed records 2
bool pack_shared_down(model & m, int32_t idx, std::string & err) {
    if (m.dense_packed.size() != size_t(n_tensors)) {
        m.dense_packed.assign(size_t(n_tensors), 0);
    }
    if (m.dense_packed[size_t(idx)] == 2) {
        return true;
    }
    const tensor_place & tp = tensors[idx];
    int32_t f = -1;
    if (m.dense_packed[size_t(idx)] != 0 || tp.lay != layout::gguf || !expert_fmt(tp.format, f)) {
        return fail(err, std::string(tp.name) + ": not a GGUF slice in a shared down format");
    }
    const auto ef = static_cast<fn::experts::fmt>(f);
    std::vector<uint8_t> src, dst;
    for (int32_t card = 0; card < 2; ++card) {
        uint8_t * p = m.region_ptr(idx, card);
        int64_t rows = 0, cols = 0;
        card_extent(idx, card, rows, cols);
        const int64_t rb = cols / tp.block_elems * tp.block_bytes;
        const int64_t n = fn::experts::down_bytes(ef);
        if (p == nullptr || rows != D || cols != FFS || rows * rb != n || tp.card[card].bytes != n) {
            return fail(err, std::string(tp.name) + ": card " + std::to_string(card) + " slice is not EX2's shape");
        }
        src.resize(static_cast<size_t>(n));
        dst.resize(static_cast<size_t>(n));
        if (!cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemcpy(src.data(), p, src.size(), cudaMemcpyDeviceToHost), "pack read", err)) {
            return false;
        }
        if (!fn::experts::pack_down(ef, src.data(), static_cast<size_t>(rb), dst.data())) {
            return fail(err, std::string(tp.name) + ": pack_down");
        }
        if (!cuda_ok(cudaMemcpy(p, dst.data(), dst.size(), cudaMemcpyHostToDevice), "pack write", err)) {
            return false;
        }
    }
    m.dense_packed[size_t(idx)] = 2;
    return true;
}

} // namespace

bool pack_dense(model & m, int32_t layer, std::string & err) {
    const bool gdn = layer % fn::full_attn_interval != fn::full_attn_interval - 1;
    std::vector<const char *> names;
    if (gdn) {
        names = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"};
    } else {
        names = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"};
    }
    names.insert(names.end(), {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"});
    for (const char * t : names) {
        const int32_t idx = model::find(tname(layer, t).c_str());
        if (idx < 0) {
            return fail(err, tname(layer, t) + ": not in the placement");
        }
        if (!pack_dense_tensor(m, idx, err)) {
            return false;
        }
    }
    const int32_t sd = model::find(tname(layer, "ffn_down_shexp.weight").c_str());
    return sd >= 0 ? pack_shared_down(m, sd, err) : fail(err, tname(layer, "ffn_down_shexp.weight") + ": not placed");
}

bool pack_draft(model & m, std::string & err) {
    const char * const names[] = {"nextn.eh_proj.weight", "attn_q.weight",      "attn_k.weight",
                                  "attn_v.weight",        "attn_output.weight", "ffn_gate_shexp.weight",
                                  "ffn_up_shexp.weight"};
    for (const char * t : names) {
        const int32_t idx = model::find(tname(fn::n_layer, t).c_str());
        if (idx < 0) {
            return fail(err, tname(fn::n_layer, t) + ": not in the placement");
        }
        if (!pack_dense_tensor(m, idx, err)) {
            return false;
        }
    }
    const int32_t sd = model::find(tname(fn::n_layer, "ffn_down_shexp.weight").c_str());
    return sd >= 0 ? pack_shared_down(m, sd, err)
                   : fail(err, tname(fn::n_layer, "ffn_down_shexp.weight") + ": not placed");
}

bool resolve_draft_weights(const model & m, int32_t card, draft_weights & out, std::string & err) {
    out = draft_weights();
    const int32_t L = fn::n_layer; // blk.48
    const finder f{m, L, card, err};
    if (!f.hc_q8("hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_norm.weight", fn::hc::w_down_q8_bytes,
                 out.hc_attn) ||
        !f.hc_q8("hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_norm.weight", fn::hc::w_down_q8_bytes, out.hc_ffn) ||
        !f.hc_q8("nextn.hc_head_down.weight", "nextn.hc_head_up.weight", "nextn.hc_head_norm.weight",
                 fn::hc::w_down_head_q8_bytes, out.head)) {
        return false;
    }
    if (!f.dense("nextn.eh_proj.weight", out.eh) || !f.dense("attn_q.weight", out.wq) ||
        !f.dense("attn_k.weight", out.wk) || !f.dense("attn_v.weight", out.wv) ||
        !f.dense("attn_output.weight", out.wo) || !check_dense(out.eh, 2 * D, F, "draft.eh_proj", err) ||
        !check_dense(out.wq, D, QF, "draft.q", err) || !check_dense(out.wk, D, HD, "draft.k", err) ||
        !check_dense(out.wv, D, HD, "draft.v", err) || !check_dense(out.wo, int64_t(QH) * HD, D, "draft.out", err)) {
        return false;
    }
    // F32 vectors: the norms whole, enorm and hnorm this card's halves (the placement's column ranges)
    const auto f32 = [&](const char * t, int64_t elems) -> const float * {
        int32_t idx = -1;
        const float * p = reinterpret_cast<const float *>(f.ptr(t, idx));
        if (p == nullptr) {
            return nullptr;
        }
        int64_t rows = 0, cols = 0;
        card_extent(idx, card, rows, cols);
        if (std::strcmp(tensors[idx].format, "F32") != 0 || tensors[idx].lay != layout::gguf || rows * cols != elems) {
            err = tname(L, t) + ": card slice is not " + std::to_string(elems) + " F32 values";
            return nullptr;
        }
        return p;
    };
    out.q_norm = f32("attn_q_norm.weight", HD);
    out.k_norm = f32("attn_k_norm.weight", HD);
    out.enorm = f32("nextn.enorm.weight", F);
    out.hnorm = f32("nextn.hnorm.weight", int64_t(4) * F);
    out.router = f32("ffn_gate_inp.weight", int64_t(NE) * D);
    out.sh_gate_w = f32("ffn_gate_inp_shexp.weight", D);
    if (out.q_norm == nullptr || out.k_norm == nullptr || out.enorm == nullptr || out.hnorm == nullptr ||
        out.router == nullptr || out.sh_gate_w == nullptr) {
        return false;
    }
    if (!f.dense("ffn_gate_shexp.weight", out.sh_gate) || !f.dense("ffn_up_shexp.weight", out.sh_up) ||
        !check_dense(out.sh_gate, D, FFS, "draft.sh_gate", err) ||
        !check_dense(out.sh_up, D, FFS, "draft.sh_up", err)) {
        return false;
    }
    int32_t si = -1, gi = -1, di = -1;
    out.ex_sh_down = f.ptr("ffn_down_shexp.weight", si);
    out.ex_gate_up = f.ptr("ffn_gate_exps.weight", gi);
    out.ex_down = f.ptr("ffn_down_exps.weight", di);
    if (out.ex_sh_down == nullptr || out.ex_gate_up == nullptr || out.ex_down == nullptr) {
        return false;
    }
    int64_t rows = 0, cols = 0;
    card_extent(si, card, rows, cols);
    // ME7: the routed experts as this load resolved them (model::tp: Q8_0, or the opt-in Q2_0), one format for gate/up
    // and down; the shared expert Q8_0 either way
    const char * rf = m.tp(gi).format;
    const bool routed = (std::strcmp(rf, "Q8_0") == 0 || std::strcmp(rf, "Q2_0") == 0) &&
                        std::strcmp(m.tp(di).format, rf) == 0;
    if (!routed || std::strcmp(tensors[si].format, "Q8_0") != 0 || rows != D || cols != FFS ||
        si >= (int32_t)m.dense_packed.size() || m.dense_packed[size_t(si)] != 2 ||
        tensors[gi].lay != layout::ex1_gate || tensors[di].lay != layout::ex1_down) {
        return fail(err, tname(L, "ffn experts") + ": not Q8_0 or Q2_0 routed (one format), Q8_0 shared, in EX1's "
                                                    "packed layouts (pack_draft)");
    }
    out.ex_fmt = int32_t(std::strcmp(rf, "Q2_0") == 0 ? fn::experts::fmt::q2_0 : fn::experts::fmt::q8_0);
    return true;
}

bool resolve_draft_index_weights(const model & m, draft_index_weights & out, std::string & err) {
    out = draft_index_weights();
    const finder f{m, fn::n_layer, 0, err}; // blk.48, on card 0 (the placement)
    out.q_proj = reinterpret_cast<const uint16_t *>(f.ptr("indexer.q_proj.weight")); // BF16 bits, GGUF rows
    out.k_proj = reinterpret_cast<const uint16_t *>(f.ptr("indexer.k_proj.weight"));
    out.q_norm = reinterpret_cast<const float *>(f.ptr("indexer.q_norm.weight"));
    out.k_norm = reinterpret_cast<const float *>(f.ptr("indexer.k_norm.weight"));
    return out.q_proj != nullptr && out.k_proj != nullptr && out.q_norm != nullptr && out.k_norm != nullptr;
}

bool resolve_state(const model & m, int32_t layer, int32_t card, layer_state & out, std::string & err) {
    out = layer_state();
    const std::string L = "L" + std::to_string(layer);
    const bool gdn = layer % fn::full_attn_interval != fn::full_attn_interval - 1;
    const auto part = [&](const char * cls, const std::string & name) -> uint8_t * {
        const reserved_class * rc = m.state_class(cls);
        const int32_t p = find_part(rc, name);
        uint8_t * ptr = p < 0 ? nullptr : rc->part_ptr(p, card);
        if (ptr == nullptr) {
            err = std::string(cls) + " / " + name + ": not reserved on card " + std::to_string(card);
        }
        return ptr;
    };
    const reserved_class * kv = m.state_class("target KV (K0)");
    const int32_t pk = find_part(kv, L + " K");
    // KV8: the class's scale parts mark the 8-bit format (a byte a value)
    const bool q8 = find_part(kv, "K scales") >= 0;
    out.n_cells = pk < 0 ? 0 : static_cast<int32_t>(kv->plan.parts[pk].bytes[card] / (int64_t(HD) * (q8 ? 1 : 2)));
    if (gdn) {
        out.gdn = reinterpret_cast<float *>(part("GDN state", L));
        out.rec = reinterpret_cast<float *>(part("GDN state", L + " records"));
        out.conv = reinterpret_cast<float *>(part("convolution histories", L + " GDN"));
        return out.gdn != nullptr && out.rec != nullptr && out.conv != nullptr;
    }
    out.kv_attn = layer / fn::full_attn_interval;
    out.k_cache = reinterpret_cast<uint16_t *>(part("target KV (K0)", L + " K")); // BF16 bits (KV8: codes)
    out.v_cache = reinterpret_cast<uint16_t *>(part("target KV (K0)", L + " V"));
    if (out.k_cache == nullptr || out.v_cache == nullptr) {
        return false;
    }
    if (q8) {
        // KV8: this layer's 4 scales a cell inside the cell-major scale rows
        const int32_t col = 4 * (layer / fn::full_attn_interval);
        uint16_t * ks = reinterpret_cast<uint16_t *>(part("target KV (K0)", "K scales"));
        uint16_t * vs = reinterpret_cast<uint16_t *>(part("target KV (K0)", "V scales"));
        if (ks == nullptr || vs == nullptr) {
            return false;
        }
        out.k_scale = ks + col;
        out.v_scale = vs + col;
    }
    if (card == (layer / fn::full_attn_interval) % 2) {
        // ME6: with the host-backed tier the keys' class is their host history
        const char * kcls = m.state_class("target index keys (ME6)") != nullptr ? "target index keys (ME6)"
                                                                                : "target index keys and summaries";
        out.idx_keys = reinterpret_cast<uint16_t *>(part(kcls, L + " keys"));
        out.idx_summaries = reinterpret_cast<float *>(part("target index keys and summaries", L + " summaries"));
        return out.idx_keys != nullptr && out.idx_summaries != nullptr;
    }
    return true;
}

namespace {

// FMU2: a lane's selection body and offsets in a joint layer_ws: its layer_max_rows rows' words and layer_max_rows + 1
// offsets, each rounded up to 16 bytes, the alignment qsa::select and the attention require of each lane's view
constexpr int64_t lane_sel_body_words = (int64_t(layer_max_rows) * fn::qsa::max_row_words + 3) / 4 * 4;
constexpr int64_t lane_sel_off_words = (layer_max_rows + 1 + 3) / 4 * 4;

// the workspace's buffers in layer_ws's order, each 256-byte aligned (FMU2: at rows rows; above layer_max_rows, the
// selection body and offsets keep lane_sel_body_words and lane_sel_off_words for each five rows, so every lane view
// has its own)
template <typename Fn> void ws_items(layer_ws & w, int32_t rows, Fn && f) {
    const int64_t B = rows;
    for (int32_t s = 0; s < 2; ++s) {
        f(reinterpret_cast<void **>(&w.x[s]), B * 4 * F * 4);
        f(reinterpret_cast<void **>(&w.mom[s]), B * fn::hc::n_mom * 4);
        f(reinterpret_cast<void **>(&w.inject[s]), B * 4 * 4);
        f(reinterpret_cast<void **>(&w.out[s]), B * F * 4);
    }
    f(reinterpret_cast<void **>(&w.hc_runs), int64_t(fn::hc::n_runs_card) * 4);
    f(reinterpret_cast<void **>(&w.zero), B * F * 4);
    f(reinterpret_cast<void **>(&w.partial), B * D * 4);
    f(reinterpret_cast<void **>(&w.mid_codes), (int64_t)fn::act::codes_bytes(B, GV));
    f(reinterpret_cast<void **>(&w.mid_scales), (int64_t)fn::act::scales_bytes(B, GV));
    f(reinterpret_cast<void **>(&w.conv), B * GC * 4);
    f(reinterpret_cast<void **>(&w.z), B * GV * 4);
    f(reinterpret_cast<void **>(&w.alpha), B * GH * 4);
    f(reinterpret_cast<void **>(&w.beta), B * GH * 4);
    f(reinterpret_cast<void **>(&w.gated), B * GV * 4);
    f(reinterpret_cast<void **>(&w.qfull), B * QF * 4);
    f(reinterpret_cast<void **>(&w.kv_k), B * HD * 4);
    f(reinterpret_cast<void **>(&w.kv_v), B * HD * 4);
    f(reinterpret_cast<void **>(&w.attn), B * GV * 4);
    f(reinterpret_cast<void **>(&w.ik), B * fn::qsa::head_dim * 4);
    f(reinterpret_cast<void **>(&w.iq), B * IQ * 4);
    const int64_t lanes = (B + layer_max_rows - 1) / layer_max_rows;
    f(reinterpret_cast<void **>(&w.sel_body),
      (B <= layer_max_rows ? B * fn::qsa::max_row_words : lanes * lane_sel_body_words) * 4);
    f(reinterpret_cast<void **>(&w.sel_off), (B <= layer_max_rows ? B + 1 : lanes * lane_sel_off_words) * 4);
    f(reinterpret_cast<void **>(&w.pos), 4 * B * 4);
    f(reinterpret_cast<void **>(&w.depth), 4);
    f(reinterpret_cast<void **>(&w.cur), 8); // ME11: cur, then the GDN selector
    f(reinterpret_cast<void **>(&w.logits), B * NE * 4);
    f(reinterpret_cast<void **>(&w.sg_logit), B * 4);
    f(reinterpret_cast<void **>(&w.sg), B * 4);
}

// the product launches of each site at their per-card shapes (formats: any quantized one tiles and splits alike;
// dense.h): each site's layout and region size. The QSA input's two shapes, the index owner's and the peer's, have
// a site each.
std::vector<std::pair<dense_site, fn::dense::product_args>> launch_shapes() {
    using fn::dense::epi;
    using fn::dense::fmt;
    const auto launch = [](int32_t k, std::vector<std::pair<fmt, int32_t>> mats) {
        fn::dense::product_args a;
        a.k = k;
        a.n_mats = static_cast<int32_t>(mats.size());
        for (size_t i = 0; i < mats.size(); ++i) {
            a.mats[i].f = mats[i].first;
            a.mats[i].m = mats[i].second;
        }
        return a;
    };
    std::vector<fn::dense::product_args> v = {
        launch(D, {{fmt::q6_k, GC}, {fmt::q6_k, GV}, {fmt::bf16, GH}, {fmt::bf16, GH}}),
        launch(GV, {{fmt::q4_k, D}}),
        launch(D, {{fmt::q4_k, QF}, {fmt::q4_k, HD}, {fmt::q4_k, HD}, {fmt::bf16, IQ}, {fmt::bf16, fn::qsa::head_dim}}),
        launch(D, {{fmt::q4_k, QF}, {fmt::q4_k, HD}, {fmt::q4_k, HD}}),
        launch(GV, {{fmt::q4_k, D}}),
        launch(D, {{fmt::bf16, NE}, {fmt::bf16, 1}}),
        // DR1's draft block (FC-DR1 "Cards"): eh_proj [5120 -> 1280 a card] on each stream row, the attention input
        // (the step: q, k, v; the catch-up: k, v), the output [3072 -> 2560] and the F32 router
        launch(2 * D, {{fmt::q8_0, F}}),
        launch(D, {{fmt::q8_0, QF}, {fmt::q8_0, HD}, {fmt::q8_0, HD}}),
        launch(D, {{fmt::q8_0, HD}, {fmt::q8_0, HD}}),
        launch(QH * HD, {{fmt::q8_0, D}}),
        launch(D, {{fmt::f32, NE}, {fmt::f32, 1}}),
        // SP2 (sparse draft): the step-0 input with the index query, the catch-up's k, v with the index key
        launch(D, {{fmt::q8_0, QF}, {fmt::q8_0, HD}, {fmt::q8_0, HD}, {fmt::bf16, IQ}}),
        launch(D, {{fmt::q8_0, HD}, {fmt::q8_0, HD}, {fmt::bf16, fn::qsa::head_dim}}),
    };
    v[0].mats[0].e = epi::conv_silu;
    v[5].route.logits = 0;
    v[5].route.gate = 1;
    v[10].route.logits = 0;
    v[10].route.gate = 1;
    const dense_site site[] = {dense_site::gdn_input,      dense_site::gdn_out,     dense_site::qsa_input_owner,
                               dense_site::qsa_input_peer, dense_site::qsa_out,     dense_site::router,
                               dense_site::draft_eh,       dense_site::draft_qkv,   dense_site::draft_kv,
                               dense_site::draft_out,      dense_site::draft_router, dense_site::draft_qkv_idx,
                               dense_site::draft_kv_idx};
    std::vector<std::pair<dense_site, fn::dense::product_args>> out;
    for (size_t i = 0; i < v.size(); ++i) {
        out.emplace_back(site[i], v[i]);
    }
    return out;
}

int64_t align256w(size_t n) {
    return (static_cast<int64_t>(n) + 255) / 256 * 256;
}

} // namespace

size_t layer_ws_bytes(int32_t rows) {
    layer_ws w;
    int64_t total = 0;
    ws_items(w, rows, [&](void **, int64_t bytes) { total += align256(bytes); });
    return static_cast<size_t>(total);
}

const char * dense_site_name(dense_site site) {
    switch (site) {
        case dense_site::gdn_input:
            return "gdn_input";
        case dense_site::gdn_out:
            return "gdn_out";
        case dense_site::qsa_input_owner:
            return "qsa_input_owner";
        case dense_site::qsa_input_peer:
            return "qsa_input_peer";
        case dense_site::qsa_out:
            return "qsa_out";
        case dense_site::router:
            return "router";
        case dense_site::draft_eh:
            return "draft_eh";
        case dense_site::draft_qkv:
            return "draft_qkv";
        case dense_site::draft_kv:
            return "draft_kv";
        case dense_site::draft_out:
            return "draft_out";
        case dense_site::draft_router:
            return "draft_router";
        case dense_site::draft_qkv_idx:
            return "draft_qkv_idx";
        case dense_site::draft_kv_idx:
            return "draft_kv_idx";
        case dense_site::count:
            break;
    }
    return "?";
}

std::string dense_layout_text(const fn::dense::work_layout & l) {
    static const char * const epi_name[] = {"f32", "glu_gate", "glu_up", "conv_silu"};
    std::string t = "K " + std::to_string(l.k) + ":";
    for (int32_t i = 0; i < l.n_mats && i < fn::dense::max_mats; ++i) {
        const fn::dense::mat_layout & m = l.mats[i];
        t += std::string(i == 0 ? " " : " | ") + std::to_string(m.m) + " rows = " + std::to_string(m.tiles) + " x " +
             std::to_string(m.tile_rows) + ", split " + std::to_string(m.split) + ", " +
             epi_name[static_cast<int32_t>(m.e) & 3];
        if (m.pair >= 0) {
            t += ", pair " + std::to_string(m.pair);
        }
        if (m.part_off >= 0) {
            t += ", part @" + std::to_string(m.part_off);
        }
        if (m.cnt_off >= 0) {
            t += ", cnt @" + std::to_string(m.cnt_off);
        }
    }
    t += "; partials " + std::to_string(l.part_floats) + " floats, counters " + std::to_string(l.cnt_words) +
         " words, router counter " + (l.route_cnt >= 0 ? "@" + std::to_string(l.route_cnt) : std::string("none"));
    return t;
}

bool dense_site_layouts(dense_layouts & out, std::string & err) {
    out = dense_layouts{};
    std::array<bool, n_dense_sites> seen{};
    for (auto & [s, a] : launch_shapes()) {
        const int32_t i = static_cast<int32_t>(s);
        fn::dense::choose_splits(a);
        fn::dense::work_layout l;
        if (!fn::dense::layout_of(a, l)) {
            return fail(err, std::string("DP2 site ") + dense_site_name(s) + ": no layout");
        }
        if (seen[i] && !fn::dense::same_layout(l, out[i])) {
            return fail(err, std::string("DP2 site ") + dense_site_name(s) +
                                 " serves two layouts: " + dense_layout_text(out[i]) + " and " + dense_layout_text(l));
        }
        seen[i] = true;
        out[i] = l;
    }
    for (int32_t i = 0; i < n_dense_sites; ++i) {
        if (!seen[i]) {
            return fail(err, std::string("DP2 site ") + dense_site_name(static_cast<dense_site>(i)) + ": no launch");
        }
    }
    return true;
}

bool dense_site_launch(void * work, const dense_layouts & layouts, dense_site site, fn::dense::product_args a,
                       int32_t rows, void * stream, std::string & err) {
    const int32_t i = static_cast<int32_t>(site);
    if (work == nullptr || i < 0 || i >= n_dense_sites) {
        return fail(err, "DP2 work or site missing");
    }
    fn::dense::choose_splits(a);
    fn::dense::work_layout l;
    if (!fn::dense::layout_of(a, l) || !fn::dense::same_layout(l, layouts[i])) {
        return fail(err, std::string("DP2 work layout is not site ") + dense_site_name(site) + "'s: launch " +
                             dense_layout_text(l) + "; site " + dense_layout_text(layouts[i]));
    }
    a.work = static_cast<uint8_t *>(work) + dense_work_offset(site);
    if (!fn::dense::product(rows, a, stream)) {
        return fail(err, "DP2 product");
    }
    return true;
}

size_t dense_site_bytes(dense_site site) {
    size_t n = 16;
    for (auto & [s, a] : launch_shapes()) {
        if (s == site) {
            fn::dense::choose_splits(a);
            n = std::max(n, fn::dense::work_bytes(a));
        }
    }
    return n;
}

size_t dense_work_offset(dense_site site) {
    int64_t off = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(site); ++i) {
        off += align256w(dense_site_bytes(static_cast<dense_site>(i)));
    }
    return static_cast<size_t>(off);
}

size_t dense_work_bytes() {
    return dense_work_offset(dense_site::count);
}

layer_ws carve_layer_ws(void * base, int32_t rows) {
    layer_ws w;
    uint8_t * p = static_cast<uint8_t *>(base);
    ws_items(w, rows, [&](void ** dst, int64_t bytes) {
        *dst = p;
        p += align256(bytes);
    });
    return w;
}

layer_ws lane_ws(const layer_ws & joint, int32_t u) {
    const int64_t b = int64_t(u) * layer_max_rows; // the lane's first row
    layer_ws w = joint;
    for (int32_t s = 0; s < 2; ++s) {
        w.x[s] += b * 4 * F;
        w.mom[s] += b * fn::hc::n_mom;
        w.inject[s] += b * 4;
        w.out[s] += b * F;
    }
    w.zero += b * F;
    w.partial += b * D;
    w.mid_codes += fn::act::codes_bytes(static_cast<int32_t>(b), GV);
    w.mid_scales += fn::act::scales_bytes(static_cast<int32_t>(b), GV) / sizeof(float);
    w.conv += b * GC;
    w.z += b * GV;
    w.alpha += b * GH;
    w.beta += b * GH;
    w.gated += b * GV;
    w.qfull += b * QF;
    w.kv_k += b * HD;
    w.kv_v += b * HD;
    w.attn += b * GV;
    w.ik += b * fn::qsa::head_dim;
    w.iq += b * IQ;
    w.sel_body += int64_t(u) * lane_sel_body_words;
    w.sel_off += int64_t(u) * lane_sel_off_words;
    w.logits += b * NE;
    w.sg_logit += b;
    w.sg += b;
    return w;
}

bool resolve_card(const model & m, int32_t card, void * stream, card_ctx & out, std::string & err) {
    out = card_ctx();
    out.card = card;
    out.device = m.device[card];
    out.stream = stream;
    out.rings = m.rings.get();
    const reserved_class * ws = m.state_class("kernel workspaces and constants");
    const int32_t p_ex = find_part(ws, "EX1 decode (B <= 5)");
    const int32_t p_qsa = find_part(ws, "QSA select scratch");
    const int32_t p_io = find_part(ws, "QSA select I/O (B <= 5)");
    const int32_t p_grid = find_part(ws, "EX1 grid table");
    const int32_t p_layer = find_part(ws, "R2 layer activations (B <= 5)");
    const int32_t p_dw = find_part(ws, "DP2 work (B <= 5)");
    const int32_t p_dg = find_part(ws, "DP2 grid table");
    const int32_t p_gdn = find_part(ws, "GDN core workspace");
    const int32_t p_attn = find_part(ws, "QSA attend workspace");
    if (out.rings == nullptr || p_ex < 0 || p_qsa < 0 || p_io < 0 || p_grid < 0 || p_layer < 0 || p_dw < 0 ||
        p_dg < 0 || p_gdn < 0 || p_attn < 0) {
        return fail(err, "the rings or a workspace part is not reserved");
    }
    uint8_t * ex = ws->part_ptr(p_ex, card);
    size_t off[fn::experts::n_ws + 1];
    fn::experts::workspace_offsets(off);
    out.ex.x_codes = reinterpret_cast<int8_t *>(ex + off[0]);
    out.ex.x_scales = reinterpret_cast<float *>(ex + off[1]);
    out.ex.glu_codes = reinterpret_cast<int8_t *>(ex + off[2]);
    out.ex.glu_scales = reinterpret_cast<float *>(ex + off[3]);
    out.ex.ids = reinterpret_cast<int32_t *>(ex + off[4]);
    out.ex.wts = reinterpret_cast<float *>(ex + off[5]);
    out.ex.meta = ex + off[8];
    out.ex.status =
        reinterpret_cast<uint32_t *>(ex + off[7]); // off[6]: the kernels' out, unused (the X-F partial is ours)
    out.qsa_ws = ws->part_ptr(p_qsa, card);
    out.qsa_io = ws->part_ptr(p_io, card);
    out.grids = ws->part_ptr(p_grid, card);
    out.dense_work = ws->part_ptr(p_dw, card);
    uint8_t * dense_grids = ws->part_ptr(p_dg, card);
    out.dense_grids = dense_grids;
    out.gdn_ws = ws->part_ptr(p_gdn, card);
    out.attn_ws = ws->part_ptr(p_attn, card);
    uint8_t * layer = ws->part_ptr(p_layer, card);
    if (layer == nullptr || (size_t)ws->plan.parts[p_layer].bytes[card] < layer_ws_bytes() ||
        (size_t)ws->plan.parts[p_dw].bytes[card] < dense_work_bytes()) {
        return fail(err, "the layer activations or DP2 work part is missing or too small");
    }
    out.ws = carve_layer_ws(layer);
    std::string why;
    if (!dense_site_layouts(out.dense_sites, why)) {
        return fail(err, why);
    }
    // DP2's tables from FNFL's constants (grids.h): the IQ3_S grid and the IQ4_NL values
    std::vector<uint8_t> table(fn::dense::grid_bytes);
    if (!fn::dense::grid_table(fn::experts::iq3s_grid, sizeof(fn::experts::iq3s_grid), fn::experts::kvalues_iq4nl,
                               sizeof(fn::experts::kvalues_iq4nl), table.data())) {
        return fail(err, "DP2 grid table");
    }
    return cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", err) &&
           cuda_ok(cudaMemcpy(dense_grids, table.data(), table.size(), cudaMemcpyHostToDevice), "dense grids", err) &&
           clear_workspaces(m, card, err);
}

bool clear_workspaces(const model & m, int32_t card, std::string & err) {
    const reserved_class * ws = m.state_class("kernel workspaces and constants");
    for (const char * name :
         {"QSA select scratch", "DP2 work (B <= 5)", "GDN core workspace", "QSA attend workspace"}) {
        const int32_t p = find_part(ws, name);
        if (p < 0 || !cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemset(ws->part_ptr(p, card), 0, (size_t)ws->plan.parts[p].bytes[card]), name, err)) {
            return false;
        }
    }
    // HC up's run counters (HC3), which every complete launch leaves at zero
    const int32_t pl = find_part(ws, "R2 layer activations (B <= 5)");
    if (pl < 0 || !cuda_ok(cudaMemset(carve_layer_ws(ws->part_ptr(pl, card)).hc_runs, 0,
                                      size_t(fn::hc::n_runs_card) * sizeof(uint32_t)),
                           "hc run counters", err)) {
        return pl < 0 ? fail(err, "the layer activations part is not reserved") : false;
    }
    return cuda_ok(cudaDeviceSynchronize(), "clear workspaces", err);
}

layer_input layer_output(const card_ctx & c) {
    layer_input o;
    o.hres = c.ws.x[1];
    o.hpost = c.ws.inject[1];
    o.hx = c.ws.out[1];
    return o;
}

namespace {

enum class phase : int32_t { down, up, body, send, recv };

step_status of(bool ok) {
    return ok ? step_status::ok : step_status::failed;
}

// one phase of sublayer sub (0 attention, 1 FFN) on one card
// fused: the per-card graphs' single HC launch (layer_card_step); the pair order runs HC's two phases (HC4, ME3)
step_status run_phase(card_ctx & c, const layer_weights & w, const layer_state & st, const layer_input & in,
                      int32_t sub, phase ph, int32_t rows, const use_counters & u, bool fused, std::string & err) {
    if (!cuda_ok(cudaSetDevice(c.device), "cudaSetDevice", err)) {
        return step_status::failed;
    }
    const hc_module & hm = sub == 0 ? w.hc_attn : w.hc_ffn;
    const layer_input hi = hc_input(c, in, sub);
    switch (ph) {
        case phase::down:
        case phase::up:
            return of(hc_phase(c, hm, sub, hi.hres, hi.hpost, hi.hx, rows, u, static_cast<int32_t>(ph), fused, err));
        case phase::body:
            return run_body(sub == 1 ? body::ffn : w.gdn ? body::gdn : body::qsa, c, w, st, rows, u, err);
        case phase::send:
        case phase::recv:
            return of(exchange_half(c, sub, rows, sub == 0 ? use_slot(u, u.a, u.graph_a) : use_slot(u, u.f, u.graph_f),
                                    ph == phase::send, err));
    }
    return of(fail(err, "phase"));
}

// the card whose body sends to its peer (X-SELECT on a QSA layer); it runs first
int32_t body_first(const layer_weights & w, int32_t sub) {
    return sub == 0 && !w.gdn ? w.index_owner : 0;
}

void advance(use_counters & u, const layer_weights & w, int32_t sub) {
    u.sub++;
    if (sub == 0) {
        u.a++;
        if (!w.gdn) {
            u.select++;
        }
    } else {
        u.f++;
    }
}

bool check_step(const card_ctx & c, const layer_weights & w, const layer_state & st, int32_t rows, std::string & err) {
    if (rows < 1 || rows > layer_max_rows || c.n_before == nullptr || c.pos == nullptr ||
        (w.gdn && (c.cur == nullptr || c.gsel == nullptr)) || (!w.gdn && st.n_cells < rows)) {
        return fail(err, "rows out of range, or no device depth, positions, committed slot or GDN selector");
    }
    return true;
}

} // namespace

step_status layer_pair_step(card_ctx * const c[2], const layer_weights * const w[2], const layer_state * const st[2],
                            const layer_input in[2], int32_t rows, use_counters u[2], std::string & err) {
    if (!check_step(*c[0], *w[0], *st[0], rows, err) || !check_step(*c[1], *w[1], *st[1], rows, err)) {
        return step_status::failed;
    }
    for (int32_t sub = 0; sub < 2; ++sub) {
        const int32_t first = body_first(*w[0], sub);
        const struct {
            phase ph;
            int32_t card;
        } order[] = {{phase::down, 0},     {phase::down, 1},         {phase::up, 0},   {phase::up, 1},
                     {phase::body, first}, {phase::body, 1 - first}, {phase::send, 0}, {phase::send, 1},
                     {phase::recv, 0},     {phase::recv, 1}};
        // ME1: with the expert cache the FFN's two bodies run as one (ffn_pair); ME8: so do the resident ones on one
        // device and stream (ffn_pair_resident), one router for both
        const bool pair = sub == 1 && c[0]->ex_service != nullptr;
        const bool shared = sub == 1 && !pair && same_stream(*c[0], *c[1]);
        if (pair && c[1]->ex_service != c[0]->ex_service) {
            err = "the expert cache's cards share one service";
            return step_status::failed;
        }
        c[0]->ffn_finished = c[1]->ffn_finished = false;
        // ME10: X-F's exchange is skipped only when the cached window's finish wrote both cards' outputs, which it
        // does on one device and stream with both shards in the window; its slots stay at the sentinel
        bool finished = false;
        // ME12: on one device and stream both cards' HC phases are one launch, at {down, 0}
        const hc_module * const hm[2] = {sub == 0 ? &w[0]->hc_attn : &w[0]->hc_ffn,
                                         sub == 0 ? &w[1]->hc_attn : &w[1]->hc_ffn};
        const bool hc_pair = hc_paired(c, *hm[0], *hm[1]);
        for (const auto & o : order) {
            const int32_t k = o.card;
            if (hc_pair && (o.ph == phase::down || o.ph == phase::up)) {
                if (o.ph == phase::down && k == 0 && !hc_pair_phase(c, hm, sub, in, rows, u, err)) {
                    return step_status::failed;
                }
                continue;
            }
            if ((pair || shared) && o.ph == phase::body) {
                if (k == first) {
                    const step_status ps = pair ? ffn_pair(c, w, rows, u, err) : ffn_pair_resident(c, w, rows, u, err);
                    if (ps != step_status::ok) {
                        return ps;
                    }
                    finished = pair && c[0]->ffn_finished && c[1]->ffn_finished;
                    if ((c[0]->ffn_finished || c[1]->ffn_finished) && (!finished || !same_stream(*c[0], *c[1]))) {
                        err = "the FFN finish wrote one card's output, or ran across devices";
                        return step_status::failed;
                    }
                }
                continue;
            }
            if (finished && (o.ph == phase::send || o.ph == phase::recv)) {
                continue;
            }
            const step_status ps = run_phase(*c[k], *w[k], *st[k], in[k], sub, o.ph, rows, u[k], false, err);
            if (ps != step_status::ok) {
                err = "card " + std::to_string(k) + ": " + err;
                return ps;
            }
        }
        advance(u[0], *w[0], sub);
        advance(u[1], *w[1], sub);
    }
    return step_status::ok;
}

step_status layer_card_step(card_ctx & c, const layer_weights & w, const layer_state & st, const layer_input & in,
                            int32_t rows, use_counters & u, std::string & err) {
    if (!check_step(c, w, st, rows, err)) {
        return step_status::failed;
    }
    for (int32_t sub = 0; sub < 2; ++sub) {
        for (const phase ph : {phase::down, phase::up, phase::body, phase::send, phase::recv}) {
            const step_status ps = run_phase(c, w, st, in, sub, ph, rows, u, true, err);
            if (ps != step_status::ok) {
                return ps;
            }
        }
        advance(u, w, sub);
    }
    return step_status::ok;
}

// ---- FMU2: the joint round ----

namespace {

constexpr int32_t joint_sites = static_cast<int32_t>(dense_site::router) + 1; // the target's six sites

size_t joint_site_bytes(dense_site site) {
    size_t n = 16;
    for (auto & [s, a] : launch_shapes()) {
        if (s == site) {
            fn::dense::choose_splits(a);
            n = std::max(n, fn::dense::joint_work_bytes(a));
        }
    }
    return n;
}

size_t joint_work_offset(dense_site site) {
    int64_t off = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(site); ++i) {
        off += align256w(joint_site_bytes(static_cast<dense_site>(i)));
    }
    return static_cast<size_t>(off);
}

// the x16 exchange's rows: every lane's five (joint_x16 runs at R = 5 only: joint.cpp refuses it at R < 5)
int32_t joint_rows_of(const joint_lanes & j) {
    return j.lanes * layer_max_rows;
}

// a body's first joint launch acquires every lane's X-HMIX slot of sublayer u.sub (lane 0's, then lane_floats apart);
// a failed acquire raises the joint error word
void acquire_joint(fn::dense::product_args & a, fn::dense::joint_args & ja, const joint_lanes & j,
                   const use_counters & u) {
    const card_ctx & c0 = *j.lane[0];
    a.hmix = hmix_in(c0, u);
    a.hmix_peer_half = 1 - c0.card;
    a.err = j.jr->err(c0.card);
    ja.hmix_lane_floats = static_cast<int32_t>(fn::xchg::joint_rings::lane_floats());
}

// fn::dense::product_joint in site's joint region, its layout checked against the site's
int32_t exchange_slot(const use_counters & u, int32_t sub) {
    return sub == 0 ? use_slot(u, u.a, u.graph_a) : use_slot(u, u.f, u.graph_f);
}

// FMS2 item 6 (XC-1): the peer's joint X-A (set 0) or X-F (1) slot of this use, which the producer stores its half to
uint32_t * peer_slot_joint(const joint_lanes & j, int32_t set, const use_counters & u) {
    const fn::xchg::msg m = set == 0 ? fn::xchg::msg::a : fn::xchg::msg::f;
    return reinterpret_cast<uint32_t *>(j.jr->slot(1 - j.lane[0]->card, m, exchange_slot(u, set)));
}

// sent (non-null): whether the product sent X-A itself (ja.peer kept: a split product)
bool joint_launch(joint_lanes & j, fn::dense::product_args a, fn::dense::joint_args ja, dense_site site,
                  const char * piece, std::string & err, bool * sent = nullptr) {
    card_ctx & c0 = *j.lane[0];
    const int32_t i = static_cast<int32_t>(site);
    if (j.dense_work == nullptr || i < 0 || i >= joint_sites) {
        return fail(err, std::string(piece) + ": joint work or site missing");
    }
    a.grids = c0.dense_grids;
    fn::dense::choose_splits(a);
    fn::dense::work_layout l;
    if (!fn::dense::joint_layout_of(a, l) || !fn::dense::same_layout(l, j.dense_sites[size_t(i)])) {
        return fail(err, std::string(piece) + ": the joint work layout is not site " + dense_site_name(site) + "'s");
    }
    a.work = static_cast<uint8_t *>(j.dense_work) + joint_work_offset(site);
    ja.lanes = j.lanes;
    ja.rows      = j.rows;      // FMS13
    ja.dense_f16 = j.dense_f16; // FMS6
    if (ja.peer != nullptr && a.mats[0].split < 2) {
        ja.peer = nullptr; // an unsplit product leaves X-A to send_half_joint (or send_half_joint16)
        ja.x16_err = nullptr;
    }
    if (!fn::dense::product_joint(a, ja, c0.stream)) {
        return fail(err, std::string(piece) + ": joint product");
    }
    if (sent != nullptr) {
        *sent = ja.peer != nullptr;
    }
    note(c0, piece);
    return true;
}

// FMS2 item 6: the X-A producer's arguments (matrix 0's peer half to the peer's slot of this use)
fn::dense::joint_args xa_joint_args(const joint_lanes & j, const use_counters & u) {
    fn::dense::joint_args ja;
    ja.peer = peer_slot_joint(j, 0, u);
    ja.peer_half = 1 - j.lane[0]->card;
    ja.x16_err = j.x16 ? j.jr->err(j.lane[0]->card) : nullptr; // FMS2 item 7
    return ja;
}

bool gdn_joint(joint_lanes & j, const layer_weights & w, const layer_state * const st[], const use_counters & u,
               bool & sent, std::string & err) {
    layer_ws & ws = j.lane[0]->ws; // the joint frame's base: every lane's rows
    fn::dense::product_args in;
    fn::dense::joint_args ja;
    in.k = D;
    in.n_mats = 4;
    in.mats[0] = qmat(w.qkv, ws.conv);
    in.mats[0].e = fn::dense::epi::conv_silu;
    in.mats[1] = qmat(w.z, ws.z);
    in.mats[2] = bmat(w.alpha, GH, ws.alpha);
    in.mats[3] = bmat(w.beta, GH, ws.beta);
    acquire_joint(in, ja, j, u);
    for (int32_t k = 0; k < j.lanes; ++k) {
        ja.conv[k] = {w.conv1d, st[k]->conv, j.lane[k]->cur};
    }
    in.conv = ja.conv[0];
    if (!joint_launch(j, in, ja, dense_site::gdn_input, "gdn.input", err)) {
        return false;
    }
    // FMS2 (build item 3, W3): every lane's gdn_core in one launch, grid.y = U, each lane on its own arguments and
    // workspace (joint_lane_kernel_ws)
    fn::gdn::core_args ga[fn::xchg::max_lanes];
    for (int32_t k = 0; k < j.lanes; ++k) {
        card_ctx & c = *j.lane[k];
        fn::gdn::core_args & g = ga[k];
        g.conv = c.ws.conv;
        g.z = c.ws.z;
        g.alpha = c.ws.alpha;
        g.beta = c.ws.beta;
        g.dt = w.dt;
        g.a = w.a;
        g.norm_w = w.ssm_norm;
        g.state = st[k]->gdn;
        g.rec = st[k]->rec;
        g.gsel = c.gsel;
        g.gated = c.ws.gated;
        g.gated_q              = { c.ws.mid_codes, c.ws.mid_scales, j.rows, GV };
        g.ws = c.gdn_ws;
        g.rows                 = j.rows;
    }
    if (!fn::gdn::core_lanes(j.lanes, ga, j.lane[0]->stream)) {
        return fail(err, "gdn core (lanes)");
    }
    note(*j.lane[0], "gdn.core");
    fn::dense::product_args out;
    out.k = GV;
    out.n_mats = 1;
    out.mats[0] = qmat(w.ssm_out, ws.partial);
    out.x_codes = ws.mid_codes;
    out.x_scales = ws.mid_scales;
    return joint_launch(j, out, xa_joint_args(j, u), dense_site::gdn_out, "gdn.out", err, &sent);
}

bool qsa_joint(joint_lanes & j, const layer_weights & w, const layer_state * const st[], const use_counters & u,
               bool & sent, std::string & err) {
    layer_ws & ws = j.lane[0]->ws;
    const int32_t r = j.lane[0]->card;
    const bool owner = r == w.index_owner;
    const int32_t sel_slot = use_slot(u, u.select, u.graph_select);
    fn::dense::product_args in;
    fn::dense::joint_args ja;
    in.k = D;
    in.n_mats = owner ? 5 : 3;
    in.mats[0] = qmat(w.wq, ws.qfull);
    in.mats[1] = qmat(w.wk, ws.kv_k);
    in.mats[2] = qmat(w.wv, ws.kv_v);
    if (owner) {
        in.mats[3] = bmat(w.idx_q, IQ, ws.iq);
        in.mats[4] = bmat(w.idx_k, fn::qsa::head_dim, ws.ik);
    }
    acquire_joint(in, ja, j, u);
    if (!joint_launch(j, in, ja, owner ? dense_site::qsa_input_owner : dense_site::qsa_input_peer, "qsa.input", err)) {
        return false;
    }
    // FMS2 (build item 3, W3): every lane's selection in one launch a piece: the owner's summary, score, select and
    // send of all lanes (one X-SELECT send a layer), or the peer's one receive; then every lane's attend in one
    // cooperative launch. Each lane's arguments and buffers are FMU2's (its own rows of the joint ws, its slot's state,
    // its lane ring) with its own kernel workspaces (joint_lane_kernel_ws), so every lane's bits are its own launches'
    card_ctx & c0 = *j.lane[0];
    const int32_t L = j.lanes;
    if (owner) {
        fn::qsa::round_update_args ua[fn::xchg::max_lanes];
        fn::qsa::select_args sa[fn::xchg::max_lanes];
        const int32_t * off[fn::xchg::max_lanes] = {};
        const uint32_t * body[fn::xchg::max_lanes] = {};
        float * peer[fn::xchg::max_lanes] = {};
        for (int32_t k = 0; k < L; ++k) {
            card_ctx & c = *j.lane[k];
            const layer_state & sk = *st[k];
            ua[k].keys = sk.idx_keys;
            ua[k].k_norm = w.idx_k_norm;
            ua[k].summaries = sk.idx_summaries;
            ua[k].rope = fn::qsa::model_rope();
            ua[k].ik = c.ws.ik;
            ua[k].depth = c.n_before;
            ua[k].rows             = j.rows;
            ua[k].max_cells = sk.n_cells;
            sa[k].summaries = sk.idx_summaries;
            sa[k].q = c.ws.iq;
            sa[k].q_norm = w.idx_q_norm;
            sa[k].depth = c.n_before;
            sa[k].max_cells = sk.n_cells;
            sa[k].rope = fn::qsa::model_rope();
            sa[k].ws = fn::qsa::carve(c.qsa_ws);
            sa[k].body = c.ws.sel_body;
            sa[k].offsets = c.ws.sel_off;
            off[k] = c.ws.sel_off;
            body[k] = c.ws.sel_body;
            peer[k] = c.rings->slot(1 - r, fn::xchg::msg::select, sel_slot);
        }
        if (!fn::qsa::summary_update_lanes(L, ua, nullptr, c0.stream)) {
            return fail(err, "summary_update (lanes)");
        }
        note(c0, "qsa.summary");
        if (!fn::qsa::select_lanes(L, j.rows, sa, c0.stream)) {
            return fail(err, "qsa select (lanes)");
        }
        note(c0, "qsa.select");
        if (!fn::xchg::send_select_lanes(L, j.rows, off, body, peer, c0.stream)) {
            return fail(err, "send_select (lanes)");
        }
        note(c0, "x.select.send");
    } else {
        float * slot[fn::xchg::max_lanes] = {};
        int32_t * off[fn::xchg::max_lanes] = {};
        uint32_t * body[fn::xchg::max_lanes] = {};
        uint32_t * werr[fn::xchg::max_lanes] = {};
        for (int32_t k = 0; k < L; ++k) {
            card_ctx & c = *j.lane[k];
            slot[k] = c.rings->slot(r, fn::xchg::msg::select, sel_slot);
            off[k] = c.ws.sel_off;
            body[k] = c.ws.sel_body;
            werr[k] = c.rings->err(r);
        }
        if (!fn::xchg::recv_select_lanes(L, j.rows, slot, off, body, werr, c0.stream)) {
            return fail(err, "recv_select (lanes)");
        }
        note(c0, "x.select.recv");
    }
    fn::qsa::attn::args aa[fn::xchg::max_lanes] = {};
    for (int32_t k = 0; k < L; ++k) {
        card_ctx & c = *j.lane[k];
        const layer_state & sk = *st[k];
        fn::qsa::attn::args & a = aa[k];
        a.qfull = c.ws.qfull;
        a.kv_k = c.ws.kv_k;
        a.kv_v = c.ws.kv_v;
        a.q_norm = w.q_norm;
        a.k_norm = w.k_norm;
        a.body = c.ws.sel_body;
        a.offsets = c.ws.sel_off;
        a.pos = c.pos;
        a.k_cache = sk.k_cache;
        a.v_cache = sk.v_cache;
        a.k_scale = sk.k_scale;
        a.v_scale = sk.v_scale;
        a.depth = c.n_before;
        a.max_cells = sk.n_cells;
        a.theta_scale = fn::qsa::attn::rope_theta_scale();
        a.out = c.ws.attn;
        a.codes = c.ws.mid_codes;
        a.scales = c.ws.mid_scales;
        a.k_f32 = nullptr;
        a.stamps = nullptr;
        a.ws = fn::qsa::attn::carve(c.attn_ws);
    }
    if (!fn::qsa::attn::attend_lanes(L, j.rows, aa, c0.stream)) {
        return fail(err, "qsa attend (lanes)");
    }
    note(c0, "qsa.attend");
    fn::dense::product_args out;
    out.k = GV;
    out.n_mats = 1;
    out.mats[0] = qmat(w.wo, ws.partial);
    out.x_codes = ws.mid_codes;
    out.x_scales = ws.mid_scales;
    return joint_launch(j, out, xa_joint_args(j, u), dense_site::qsa_out, "qsa.out", err, &sent);
}

bool ffn_joint(joint_lanes & j, const layer_weights & w, const use_counters & u, bool & sent, std::string & err) {
    card_ctx & c0 = *j.lane[0];
    layer_ws & ws = c0.ws;
    fn::dense::product_args rt;
    fn::dense::joint_args ja;
    rt.k = D;
    rt.n_mats = 2;
    rt.mats[0] = bmat(w.router, NE, ws.logits);
    rt.mats[1] = bmat(w.sh_gate_w, 1, ws.sg_logit);
    acquire_joint(rt, ja, j, u);
    rt.route.logits = 0;
    rt.route.gate = 1;
    rt.route.ids = c0.ex.ids; // lane 0's view is the joint routing's base
    rt.route.wts = c0.ex.wts;
    rt.route.sg = ws.sg;
    ja.router_tc = j.router_tc; // FMS5
    if (!joint_launch(j, rt, ja, dense_site::router, "ffn.router", err)) {
        return false;
    }
    // FMU2 (build item 1): every lane's gate/up in one launch, then every lane's down in one, each lane's units its own
    // (fn::experts::gate_up_lanes); a form the lanes launch does not take runs each lane's experts alone. FMS13: at
    // R < 5 a gate/up the lanes launch does not take (IQ3 at R <= 3: no lanes form of its lone kernel) runs each lane's
    // lone gate/up at B = R, then the lanes' down (both bitwise the lone launches)
    const auto f = static_cast<fn::experts::fmt>(w.ex_gu_fmt);
    const auto fg = static_cast<fn::dense::fmt>(w.sh_gate.fmt);
    const auto fu = static_cast<fn::dense::fmt>(w.sh_up.fmt);
    const auto fd = static_cast<fn::experts::fmt>(w.ex_down_fmt);
    const auto fs = static_cast<fn::experts::fmt>(w.ex_sh_down_fmt);
    fn::experts::gate_up_lanes_args ga;
    fn::experts::down_lanes_args da;
    bool lanes = true;
    const int32_t                   R     = j.rows;
    for (int32_t k = 0; k < j.lanes; ++k) {
        lanes = lanes && j.lane[k]->ex_service == nullptr;
        expert_args(*j.lane[k], w, R, u, ga.s[k], da.s[k]);
    }
    const bool gu_lanes = lanes && fn::experts::gate_up_lanes_supported(f, fg, fu, j.lanes, R, ga);
    if (lanes && (gu_lanes || R < layer_max_rows) && fn::experts::down_lanes_supported(fd, fs, j.lanes, R, da)) {
        if (gu_lanes) {
            if (!fn::experts::gate_up_lanes(f, fg, fu, j.lanes, R, ga, c0.stream)) {
                return fail(err, "experts gate_up_lanes");
            }
        } else {
            for (int32_t k = 0; k < j.lanes; ++k) {
                if (!fn::experts::gate_up(f, fg, fu, R, ga.s[k], c0.stream)) {
                    return fail(err, "experts gate_up (a lane alone)");
                }
            }
        }
        note(c0, "ffn.ex_gate_up");
        // FMS2 item 6: X-F sent by the down launch
        if (!fn::experts::down_lanes_send(fd, fs, j.lanes, R, da, peer_slot_joint(j, 1, u), 1 - c0.card,
                                          j.x16 ? j.jr->err(c0.card) : nullptr, c0.stream)) {
            return fail(err, "experts down_lanes");
        }
        sent = true;
        note(c0, "ffn.ex_down");
        return true;
    }
    for (int32_t k = 0; k < j.lanes; ++k) {
        if (!ffn_experts(*j.lane[k], w, R, u, nullptr, err)) {
            return false;
        }
    }
    return true;
}

// the joint destination-half exchange of the joint partial (set 0: X-A, 1: X-F) on joint slot `slot`
bool exchange_joint(joint_lanes & j, int32_t set, int32_t slot, bool send, std::string & err) {
    card_ctx & c0 = *j.lane[0];
    const int32_t r = c0.card;
    const int32_t       rows = joint_rows_of(j);  // x16 only (FMS13: the F32 forms move each lane's R rows)
    const fn::xchg::msg m = set == 0 ? fn::xchg::msg::a : fn::xchg::msg::f;
    if (send) {
        float * const peer = j.jr->slot(1 - r, m, slot);
        if (!(j.x16 ? fn::xchg::send_half_joint16(rows, 1 - r, c0.ws.partial, peer, j.jr->err(r), c0.stream) :
                      fn::xchg::send_half_joint(j.lanes, j.rows, 1 - r, c0.ws.partial, peer, c0.stream))) {
            return fail(err, "send_half_joint");
        }
        note(c0, set == 0 ? "x.a.send" : "x.f.send");
        return true;
    }
    const uint32_t code = set == 0 ? fn::xchg::err_poll_a : fn::xchg::err_poll_f;
    float * const own = j.jr->slot(r, m, slot);
    if (!(j.x16 ?
              fn::xchg::recv_add_joint16(rows, r, c0.ws.partial, own, c0.ws.out[set], j.jr->err(r), code, c0.stream) :
              fn::xchg::recv_add_joint(j.lanes, j.rows, r, c0.ws.partial, own, c0.ws.out[set], j.jr->err(r), code,
                                       c0.stream))) {
        return fail(err, "recv_add_joint");
    }
    note(c0, set == 0 ? "x.a.recv" : "x.f.recv");
    return true;
}

} // namespace

bool joint_dense_layouts(dense_layouts & out, std::string & err) {
    out = dense_layouts{};
    std::array<bool, n_dense_sites> seen{};
    for (auto & [s, a] : launch_shapes()) {
        const int32_t i = static_cast<int32_t>(s);
        if (i >= joint_sites) {
            continue;
        }
        fn::dense::choose_splits(a);
        fn::dense::work_layout l;
        if (!fn::dense::joint_layout_of(a, l)) {
            return fail(err, std::string("joint site ") + dense_site_name(s) + ": no layout");
        }
        if (seen[size_t(i)] && !fn::dense::same_layout(l, out[size_t(i)])) {
            return fail(err, std::string("joint site ") + dense_site_name(s) + " serves two layouts");
        }
        seen[size_t(i)] = true;
        out[size_t(i)] = l;
    }
    for (int32_t i = 0; i < joint_sites; ++i) {
        if (!seen[size_t(i)]) {
            return fail(err, std::string("joint site ") + dense_site_name(static_cast<dense_site>(i)) + ": no launch");
        }
    }
    return true;
}

size_t joint_ple_work_offset() {
    return joint_work_offset(dense_site::router) + align256w(joint_site_bytes(dense_site::router));
}

size_t joint_dense_work_bytes() {
    return joint_ple_work_offset() + align256w(std::max<size_t>(fn::ple::joint_products_work_bytes(), 16));
}

namespace {

// a lane's experts' buffers in the joint routing region: GLU codes, GLU scales ({d, s} a run), meta
constexpr int64_t lane_ex_codes = int64_t(fn::experts::glu_slots) * FFS;
constexpr int64_t lane_ex_scales = int64_t(fn::experts::glu_slots) * (FFS / 32) * 2 * 4;
int64_t lane_ex_bytes() {
    return align256(lane_ex_codes) + align256(lane_ex_scales) + align256(fn::experts::meta_bytes);
}

} // namespace

size_t joint_route_bytes() {
    return static_cast<size_t>(2 * align256(int64_t(fn::xchg::joint_rows) * NU * 4) +
                               fn::xchg::max_lanes * lane_ex_bytes());
}

namespace {

int64_t lane_kernel_ws_bytes() {
    return align256(int64_t(fn::qsa::workspace_bytes())) + align256(int64_t(fn::qsa::attn::workspace_bytes())) +
           align256(int64_t(fn::gdn::workspace_bytes()));
}

} // namespace

size_t joint_kernel_ws_bytes() {
    return static_cast<size_t>(fn::xchg::max_lanes * lane_kernel_ws_bytes());
}

void joint_lane_kernel_ws(void * base, int32_t u, card_ctx & c) {
    uint8_t * p = static_cast<uint8_t *>(base) + int64_t(u) * lane_kernel_ws_bytes();
    c.qsa_ws = p;
    p += align256(int64_t(fn::qsa::workspace_bytes()));
    c.attn_ws = p;
    p += align256(int64_t(fn::qsa::attn::workspace_bytes()));
    c.gdn_ws = p;
}

void joint_lane_ex(void * base, int32_t u, ex_bufs & ex) {
    uint8_t * p = static_cast<uint8_t *>(base) + 2 * align256(int64_t(fn::xchg::joint_rows) * NU * 4) +
                  int64_t(u) * lane_ex_bytes();
    ex.glu_codes = reinterpret_cast<int8_t *>(p);
    ex.glu_scales = reinterpret_cast<float *>(p + align256(lane_ex_codes));
    ex.meta = p + align256(lane_ex_codes) + align256(lane_ex_scales);
}

void joint_route_carve(void * base, int32_t *& ids, float *& wts) {
    uint8_t * p = static_cast<uint8_t *>(base);
    ids = reinterpret_cast<int32_t *>(p);
    wts = reinterpret_cast<float *>(p + align256(int64_t(fn::xchg::joint_rows) * NU * 4));
}

namespace {

// the joint step's checks: lanes 2..4 on one card and stream, resident banks, every lane's step (check_step)
bool joint_check(const joint_lanes & j, const layer_weights & w, const layer_state * const st[], std::string & err) {
    if (j.lanes < 2 || j.lanes > fn::xchg::max_lanes || j.jr == nullptr || j.dense_work == nullptr || j.rows < 2 ||
        j.rows > layer_max_rows || (j.x16 && j.rows != layer_max_rows)) {
        return fail(err,
                    "joint step: lanes outside 2 .. 4, rows outside 2 .. 5 (x16 at 5 only), or no joint "
                    "rings or work");
    }
    for (int32_t k = 0; k < j.lanes; ++k) {
        const card_ctx & c = *j.lane[k];
        if (!check_step(c, w, *st[k], j.rows, err)) {
            return false;
        }
        if (c.ex_service != nullptr || c.kv != nullptr || c.card != j.lane[0]->card || c.stream != j.lane[0]->stream) {
            return fail(err, "joint step: the expert cache, the host-backed KV, or lanes on different cards");
        }
    }
    return true;
}

// every lane's HC sublayer sub in one launch (hc::sub_joint: each lane's the lone B = R code on its ring); mode as
// sub's (3 both phases; 1 and 2 the pair order's)
bool hc_joint(const joint_lanes & j, const layer_weights & w, const layer_input in[], int32_t sub, int32_t mode,
              const use_counters & u, std::string & err) {
    const hc_module & hm = sub == 0 ? w.hc_attn : w.hc_ffn;
    if (hm.q8_down != nullptr) {
        return fail(err, "joint step: Q8_0 HC weights");
    }
    fn::hc::sub_args ha[fn::xchg::max_lanes];
    for (int32_t k = 0; k < j.lanes; ++k) {
        const card_ctx & c = *j.lane[k];
        const layer_input hi = hc_input(c, in[k], sub);
        ha[k] = hc_sub_args(c, hm, sub, hi.hres, hi.hpost, hi.hx, u);
    }
    if (!cuda_ok(cudaSetDevice(j.lane[0]->device), "cudaSetDevice", err) ||
        !fn::hc::sub_joint(j.lanes, j.rows, mode, ha, j.lane[0]->stream)) {
        return err.empty() ? fail(err, "hc sub_joint") : false;
    }
    note(*j.lane[0], "hc.sub_joint");
    return true;
}

// the sublayer's body for every lane
// sent: the body's producer sent the sublayer's X-A or X-F itself (FMS2 item 6), so no send_half_joint
bool body_joint(joint_lanes & j, const layer_weights & w, const layer_state * const st[], const use_counters & u,
                int32_t sub, bool & sent, std::string & err) {
    sent = false;
    const std::vector<std::string> missing = body_unbuilt(sub == 1 ? body::ffn : w.gdn ? body::gdn : body::qsa);
    if (!missing.empty()) {
        return fail(err, "joint step: a body piece is not built");
    }
    if (!cuda_ok(cudaSetDevice(j.lane[0]->device), "cudaSetDevice", err)) {
        return false;
    }
    if (sub == 1) {
        return ffn_joint(j, w, u, sent, err);
    }
    return w.gdn ? gdn_joint(j, w, st, u, sent, err) : qsa_joint(j, w, st, u, sent, err);
}

} // namespace

step_status layer_joint_step(joint_lanes & j, const layer_weights & w, const layer_state * const st[],
                             const layer_input in[], use_counters & u, std::string & err) {
    if (!joint_check(j, w, st, err)) {
        return step_status::failed;
    }
    for (int32_t sub = 0; sub < 2; ++sub) {
        bool sent = false;
        if (!hc_joint(j, w, in, sub, 3, u, err) || !body_joint(j, w, st, u, sub, sent, err)) {
            return step_status::failed;
        }
        const int32_t slot = exchange_slot(u, sub);
        if (!(sent || exchange_joint(j, sub, slot, true, err)) || !exchange_joint(j, sub, slot, false, err)) {
            return step_status::failed;
        }
        advance(u, w, sub);
    }
    return step_status::ok;
}

step_status layer_joint_pair_step(joint_lanes * const j[2], const layer_weights * const w[2],
                                  const layer_state * const * const st[2], const layer_input * const in[2],
                                  use_counters u[2], std::string & err) {
    for (int32_t k = 0; k < 2; ++k) {
        if (!joint_check(*j[k], *w[k], st[k], err)) {
            err = "card " + std::to_string(k) + ": " + err;
            return step_status::failed;
        }
    }
    if (j[0]->lanes != j[1]->lanes || j[0]->lane[0]->card != 0 || j[1]->lane[0]->card != 1) {
        return of(fail(err, "joint pair step: the same lanes on cards 0 and 1"));
    }
    for (int32_t sub = 0; sub < 2; ++sub) {
        // layer_pair_step's order: both cards' down phases, both up phases, the bodies (the index owner's first),
        // both sends, both receives
        const int32_t first = body_first(*w[0], sub);
        const int32_t slot[2] = {exchange_slot(u[0], sub), exchange_slot(u[1], sub)};
        bool ok = true;
        for (int32_t mode = 1; mode <= 2 && ok; ++mode) {
            for (int32_t k = 0; k < 2 && ok; ++k) {
                ok = hc_joint(*j[k], *w[k], in[k], sub, mode, u[k], err);
            }
        }
        bool sent[2] = {};
        for (const int32_t k : {first, 1 - first}) {
            ok = ok && body_joint(*j[k], *w[k], st[k], u[k], sub, sent[k], err);
        }
        for (const bool send : {true, false}) {
            for (int32_t k = 0; k < 2 && ok; ++k) {
                if (send && sent[k]) {
                    continue; // the producer sent it
                }
                ok = cuda_ok(cudaSetDevice(j[k]->lane[0]->device), "cudaSetDevice", err) &&
                     exchange_joint(*j[k], sub, slot[k], send, err);
            }
        }
        if (!ok) {
            return step_status::failed;
        }
        advance(u[0], *w[0], sub);
        advance(u[1], *w[1], sub);
    }
    return step_status::ok;
}

} // namespace fn::model
