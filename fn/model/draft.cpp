#include "fn/model/draft.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/draft/draft.h"
#include "fn/kernels/embed/embed.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/head/head.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/samp/samp.h"
#include "fn/model/kv_tier.h"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace fn::model {

namespace {

constexpr int32_t D = fn::n_embd;                // 2560
constexpr int32_t F = fn::hc::n_own;             // 1280
constexpr int32_t QH = fn::draft::n_head;        // 12
constexpr int32_t HD = fn::draft::head_dim;      // 256
constexpr int32_t QF = QH * fn::draft::q_stride; // 6144
constexpr int32_t NE = fn::n_expert;             // 512
constexpr int32_t FFS = fn::experts::n_ff_card;  // 320
constexpr int32_t NV = fn::head::n_draft_card;   // 32768
constexpr int32_t step_phases = 11;              // a draft step's phases (phase())
constexpr int32_t catchup_phases = 4;

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

int32_t part_of(const reserved_class * rc, const std::string & name) {
    for (int32_t i = 0; rc != nullptr && i < (int32_t)rc->plan.parts.size(); ++i) {
        if (rc->plan.parts[i].name == name) {
            return i;
        }
    }
    return -1;
}

fn::dense::mat qmat(const dense_weight & w, float * out) {
    fn::dense::mat m;
    m.f = static_cast<fn::dense::fmt>(w.fmt);
    m.m = static_cast<int32_t>(w.n_out);
    m.w = w.w;
    m.out = out;
    return m;
}

// a BF16 matrix (GGUF rows, read by DP2 unchanged), F32 output (SP2: the draft's index products)
fn::dense::mat bmat16(const uint16_t * w, int32_t rows, float * out) {
    fn::dense::mat m;
    m.f = fn::dense::fmt::bf16;
    m.m = rows;
    m.w = reinterpret_cast<const uint8_t *>(w);
    m.out = out;
    return m;
}

// an F32 matrix (GGUF rows, read by DP2 unchanged)
fn::dense::mat fmat(const float * w, int32_t rows, float * out) {
    fn::dense::mat m;
    m.f = fn::dense::fmt::f32;
    m.m = rows;
    m.w = reinterpret_cast<const uint8_t *>(w);
    m.out = out;
    return m;
}

bool is_step(int32_t kind) {
    return kind == static_cast<int32_t>(fn::round::kind::draft_step);
}

bool check_round(int32_t kind, int32_t rows, std::string & err) {
    const bool step = is_step(kind);
    if ((!step && kind != static_cast<int32_t>(fn::round::kind::draft_catchup)) || rows < 1 ||
        rows > (step ? draft_max_steps : fn::draft::max_rows)) {
        return fail(err, "draft round: kind or rows out of range");
    }
    return true;
}

// ---- the pieces, one card each ----

bool begin(pass_card & p, std::string & err) {
    const fn::round::begin_args a = {p.host_ring, p.flag, p.desc, p.rstate, p.c.rings->err(p.c.card)};
    // ME3: the failure names the pending CUDA error (the launch check peeks at it)
    return fn::round::begin(a, p.c.stream) && fn::round::stamp(p.rstate, p.c.stream)
               ? true
               : fail(err, std::string("draft round begin: ") + cudaGetErrorString(cudaPeekAtLastError()));
}

bool finish(pass_card & p, std::string & err) {
    const fn::round::commit_args ca = {p.desc, p.rstate, p.errs, p.n_err, p.c.card};
    const fn::round::finish_args fa = {p.desc, p.rstate, p.errs, p.n_err, p.done};
    return fn::round::commit(ca, p.c.stream) && fn::round::finish(fa, p.c.stream) ? true
                                                                                  : fail(err, "draft round commit");
}

// the embedding of the rows' tokens into the boundary's hres (only stream 0's own half, e, is read)
bool embed(pass_card & p, const int32_t * tok, int32_t rows, std::string & err) {
    fn::embed::args a;
    a.table = p.token_embd;
    a.grids = p.c.dense_grids;
    a.tok = tok;
    a.hres = p.emb_hres;
    a.hpost = p.emb_hpost;
    a.hx = p.emb_hx;
    a.err = p.c.rings->err(p.c.card);
    return fn::embed::embed(rows, a, p.c.stream) ? true : fail(err, "draft embed");
}

bool draft_in(pass_card & p, draft_card & d, int32_t rows, bool chained, bool catchup, bool send,
              const use_counters & u, std::string & err) {
    const int32_t r = p.c.card;
    const fn::xchg::rings & rg = *p.c.rings;
    const int32_t s = use_slot_of(u, u.draft, u.graph_draft);
    if (send) {
        fn::draft::send_args a;
        a.emb = p.emb_hres;
        a.emb_stride = 4 * F;
        a.carry = p.carry;
        a.carry_prev = d.carry_prev;
        a.h_nextn = d.h_nextn;
        a.hsrc = &p.desc->hsrc;
        a.chained = chained;
        a.g_e = d.w.enorm;
        a.g_h = d.w.hnorm;
        a.peer = rg.slot(1 - r, fn::xchg::msg::draft, s);
        a.own = d.in_own;
        if (!fn::draft::draft_in_send(rows, a, p.c.stream)) {
            return fail(err, "draft_in_send");
        }
        note_piece(p.c, "draft.in.send");
        return true;
    }
    fn::draft::recv_args a;
    a.own = d.in_own;
    a.slot = rg.slot(r, fn::xchg::msg::draft, s);
    a.codes = d.codes;
    a.scales = d.scales;
    a.f32 = nullptr;
    a.err = rg.err(r);
    a.card = r;
    a.carry_prev = catchup ? d.carry_prev : nullptr;
    a.carry = p.carry;
    a.hsrc = &p.desc->hsrc;
    if (!fn::draft::draft_in_recv(rows, a, p.c.stream)) {
        return fail(err, "draft_in_recv");
    }
    note_piece(p.c, "draft.in.recv");
    return true;
}

// eh_proj on the rows' 4 streams each (Stage C amendment 1's adapter), in the draft_eh site's work region
bool eh_proj(pass_card & p, draft_card & d, int32_t rows, std::string & err) {
    fn::dense::product_args a;
    a.k = 2 * D;
    a.n_mats = 1;
    a.mats[0] = qmat(d.w.eh, d.hres_d);
    a.x_codes = d.codes;
    a.x_scales = d.scales;
    a.grids = p.c.dense_grids;
    fn::dense::choose_splits(a);
    fn::dense::work_layout l;
    const int32_t site = static_cast<int32_t>(dense_site::draft_eh);
    if (!fn::dense::layout_of(a, l) || !fn::dense::same_layout(l, p.c.dense_sites[site])) {
        return fail(err, "draft eh_proj: not the draft_eh site's layout");
    }
    a.work = static_cast<uint8_t *>(p.c.dense_work) + dense_work_offset(dense_site::draft_eh);
    if (!fn::draft::eh_proj(rows, a, p.c.stream)) {
        return fail(err, "draft eh_proj");
    }
    note_piece(p.c, "draft.eh_proj");
    return true;
}

// the attention sublayer's input: eh_proj's streams, with hpost = hx = 0
layer_input attn_input(const pass_card & p, const draft_card & d) {
    layer_input in;
    in.hres = d.hres_d;
    in.hpost = p.c.ws.zero;
    in.hx = p.c.ws.zero;
    return in;
}

// the FFN sublayer's input: the attention sublayer's streams, injection and output half
layer_input ffn_input(const pass_card & p) {
    layer_input in;
    in.hres = p.c.ws.x[0];
    in.hpost = p.c.ws.inject[0];
    in.hx = p.c.ws.out[0];
    return in;
}

bool prepare(pass_card & p, draft_card & d, int32_t rows, int32_t row0, bool catchup, const use_counters & u,
             std::string & err) {
    const int32_t r = p.c.card;
    const fn::xchg::rings & rg = *p.c.rings;
    fn::draft::prepare_args a;
    a.qfull = catchup ? nullptr : p.c.ws.qfull;
    a.kv_k = p.c.ws.kv_k;
    a.kv_v = p.c.ws.kv_v;
    a.q_norm = d.w.q_norm;
    a.k_norm = d.w.k_norm;
    a.n_before = &p.desc->n_before;
    a.pos = p.desc->pos;
    a.stride = &p.desc->rows;
    a.row0 = row0;
    a.theta_scale = fn::qsa::attn::rope_theta_scale();
    a.k_cache = d.k_cache;
    a.v_cache = d.v_cache;
    const int64_t * shift = nullptr;
    if (p.c.kv != nullptr) { // ME1: the host-backed KV's decode window (kv_tier.h)
        a.k_cache = p.c.kv->k(r, kv_tier::draft_attn);
        a.v_cache = p.c.kv->v(r, kv_tier::draft_attn);
        shift = p.c.kv->shift(r); // ME3: the graphs' window start (null for eager)
    }
    a.max_cells = d.max_cells;
    a.q16 = d.q16;
    a.q_f32 = nullptr;
    a.k_f32 = nullptr;
    a.err = rg.err(r);
    // the catch-up's sublayer slot has no later sublayer in its round to reset it (hc_rule catchup): its last
    // readers are done (up2 read X-HC, the K/V product acquired X-HMIX), so it is released here
    const int32_t s = hc_slot_of(u, u.sub);
    a.release_hc = catchup ? rg.slot(r, fn::xchg::msg::hc, s) : nullptr;
    a.release_hmix = catchup ? rg.slot(r, fn::xchg::msg::hmix, s) : nullptr;
    if (!fn::draft::datc_prepare(rows, a, p.c.stream, shift)) {
        return fail(err, "datc_prepare");
    }
    note_piece(p.c, "draft.datc.prepare");
    return true;
}

// SP2 (sparse, step 0): the select of the step's index query over the draft index's summaries into the layer
// workspace's body (free between target rounds; steps 1 .. d - 1 reuse it)
bool draft_select(pass_card & p, draft_card & d, std::string & err) {
    layer_ws & ws = p.c.ws;
    if (!fn::draft::select_depth(&p.desc->n_before, d.sel_depth, p.c.stream)) {
        return fail(err, "draft select depth");
    }
    fn::qsa::select_args sa;
    sa.summaries = d.idx_summ;
    sa.q = ws.iq;
    sa.q_norm = d.iw.q_norm;
    sa.depth = d.sel_depth;
    sa.max_cells = d.max_cells;
    sa.rope = fn::qsa::model_rope();
    sa.ws = fn::qsa::carve(p.c.qsa_ws);
    sa.body = ws.sel_body;
    sa.offsets = ws.sel_off;
    sa.q_ahead = 1;
    if (!fn::qsa::select(1, sa, p.c.stream)) {
        return fail(err, "draft select");
    }
    note_piece(p.c, "draft.select");
    return true;
}

// the step's attention body after the HC sublayer: q/k/v (acquiring X-HMIX), DA-TC, the output product, X-A's send;
// SP2's sparse step 0 adds the index query to the product and selects after the prepare
bool attn_body(pass_card & p, draft_card & d, int32_t j, const use_counters & u, std::string & err) {
    layer_ws & ws = p.c.ws;
    const bool sel = d.sparse && j == 0;
    fn::dense::product_args in;
    in.k = D;
    in.n_mats = sel ? 4 : 3;
    in.mats[0] = qmat(d.w.wq, ws.qfull);
    in.mats[1] = qmat(d.w.wk, ws.kv_k);
    in.mats[2] = qmat(d.w.wv, ws.kv_v);
    if (sel) {
        in.mats[3] = bmat16(d.iw.q_proj, fn::qsa::n_idx_head * fn::qsa::head_dim, ws.iq);
    }
    if (!dense_run(p.c, in, 1, sel ? dense_site::draft_qkv_idx : dense_site::draft_qkv, true, u, "draft.qkv", err) ||
        !prepare(p, d, 1, j, false, u, err) || (sel && !draft_select(p, d, err))) {
        return false;
    }
    fn::draft::attend_args a;
    a.q16 = d.q16;
    a.qfull = ws.qfull;
    a.k_cache = d.k_cache;
    a.v_cache = d.v_cache;
    a.n_before = &p.desc->n_before;
    a.row0 = j;
    a.max_cells = d.max_cells;
    a.part = d.part;
    a.out = ws.attn;
    a.codes = ws.mid_codes;
    a.scales = ws.mid_scales;
    a.err = p.c.rings->err(p.c.card);
    a.body = d.sparse ? ws.sel_body : nullptr;
    const int64_t * shift = nullptr;
    if (p.c.kv != nullptr) {
        // ME1: the host-backed KV: step 0 gathers its selection's older blocks, the chain's later steps reuse them
        a.k_cache = p.c.kv->k(p.c.card, kv_tier::draft_attn);
        a.v_cache = p.c.kv->v(p.c.card, kv_tier::draft_attn);
        if (!d.sparse) {
            return fail(err, "the host-backed KV tier serves the sparse draft only");
        }
        if (sel && !p.c.kv->select(p.c.card, kv_tier::draft_attn, 1, ws.sel_body, ws.sel_off, a.err, p.c.stream,
                                   a.body, err)) {
            return false;
        }
        a.body = sel ? a.body : p.c.kv->draft_body(p.c.card);
        shift = p.c.kv->shift(p.c.card);
        note_piece(p.c, "draft.kv.gather");
    }
    if (!fn::draft::datc_attend(a, p.c.stream, shift)) {
        return fail(err, "datc_attend");
    }
    note_piece(p.c, "draft.datc.attend");
    fn::dense::product_args out;
    out.k = QH * HD;
    out.n_mats = 1;
    out.mats[0] = qmat(d.w.wo, ws.partial);
    out.x_codes = ws.mid_codes;
    out.x_scales = ws.mid_scales;
    return dense_run(p.c, out, 1, dense_site::draft_out, false, u, "draft.out", err) &&
           half_exchange(p.c, 0, 1, use_slot_of(u, u.a, u.graph_a), true, err);
}

// the FFN body after its HC sublayer: the F32 router and shared gate (acquiring X-HMIX), the routed experts (Q8_0, or
// ME7's opt-in Q2_0: the draft file's, draft_weights::ex_fmt) with the Q8_0 shared expert, X-F's send
bool ffn_body(pass_card & p, draft_card & d, int32_t rows, const use_counters & u, std::string & err) {
    card_ctx & c = p.c;
    layer_ws & ws = c.ws;
    // ME8: on one device and stream, card 1 reads card 0's routing (d.shared_route) and runs no router of its own
    const route_view own = {c.ex.ids, c.ex.wts, ws.sg};
    const route_view & rv = d.shared_route != nullptr ? *d.shared_route : own;
    if (d.shared_route == nullptr) {
        fn::dense::product_args rt;
        rt.k = D;
        rt.n_mats = 2;
        rt.mats[0] = fmat(d.w.router, NE, ws.logits);
        rt.mats[1] = fmat(d.w.sh_gate_w, 1, ws.sg_logit);
        rt.route.logits = 0;
        rt.route.gate = 1;
        rt.route.ids = c.ex.ids;
        rt.route.wts = c.ex.wts;
        rt.route.sg = ws.sg;
        if (!dense_run(c, rt, rows, dense_site::draft_router, true, u, "draft.router", err)) {
            return false;
        }
    }
    const fn::act::out_view glu = {c.ex.glu_codes, c.ex.glu_scales, fn::experts::glu_slots, FFS};
    const fn::act::view glu_in = {c.ex.glu_codes, c.ex.glu_scales, fn::experts::glu_slots, FFS};
    fn::experts::gate_up_args ga = {d.w.ex_gate_up, d.w.sh_gate.w, d.w.sh_up.w, c.grids, {nullptr, nullptr, rows, D},
                                    c.ex.ids,       c.ex.meta,     glu,         nullptr, c.ex.status};
    const auto rf = static_cast<fn::experts::fmt>(d.w.ex_fmt); // ME7: the routed format
    ga.cw = fn::experts::class_weights(rf, static_cast<fn::dense::fmt>(d.w.sh_gate.fmt),
                                       static_cast<fn::dense::fmt>(d.w.sh_up.fmt)); // XB3: the unit order's weights
    ga.hmix = hmix_slot(c, u);
    ga.hmix_peer_half = 1 - c.card;
    ga.err = c.rings->err(c.card);
    fn::experts::down_args da = {d.w.ex_down, d.w.ex_sh_down, c.grids, glu_in,     c.ex.ids,
                                 c.ex.meta,   c.ex.wts,       ws.sg,   ws.partial, c.ex.status};
    fn::experts::use_route(rv, ga, da); // ME8: card 0's routing on one device and stream, else this card's own
    const auto q8 = fn::experts::fmt::q8_0;
    if (!fn::experts::gate_up(rf, static_cast<fn::dense::fmt>(d.w.sh_gate.fmt),
                              static_cast<fn::dense::fmt>(d.w.sh_up.fmt), rows, ga, c.stream)) {
        return fail(err, "draft experts gate_up");
    }
    note_piece(c, "draft.ex_gate_up");
    if (!fn::experts::down(rf, q8, rows, da, c.stream)) {
        return fail(err, "draft experts down");
    }
    note_piece(c, "draft.ex_down");
    return half_exchange(c, 1, rows, use_slot_of(u, u.f, u.graph_f), true, err);
}

// the draft's HC head (sublayer u.sub): part 0 down2 (its streams into h: h_nextn, or the joint step's rows; sends
// X-HC), 1 up2 (the mixed rows into X-HMIX on both cards)
bool head(pass_card & p, draft_card & d, int32_t part, int32_t rows, float * h, const use_counters & u,
          std::string & err) {
    const int32_t r = p.c.card;
    const fn::xchg::rings & rg = *p.c.rings;
    const layer_ws & ws = p.c.ws;
    const int32_t s = hc_slot_of(u, u.sub);
    if (part == 0) {
        const int32_t s_prev = hc_prev_slot_of(u, u.sub);
        fn::hc::down2_q8_args a;
        a.a.d = {nullptr, ws.x[1], ws.out[1], ws.inject[1], d.w.head.gamma, h, p.head_mom};
        a.a.hc_peer = rg.slot(1 - r, fn::xchg::msg::hc, s);
        a.a.hc_reset = s_prev >= 0 ? rg.slot(r, fn::xchg::msg::hc, s_prev) : nullptr;
        a.a.hmix_reset = s_prev >= 0 ? rg.slot(r, fn::xchg::msg::hmix, s_prev) : nullptr;
        a.w = d.w.head.q8_down;
        if (!fn::hc::head_down2_q8(rows, a, p.c.stream)) {
            return fail(err, "draft head down2");
        }
        note_piece(p.c, "draft.head.down2");
        return true;
    }
    fn::hc::up2_q8_args a;
    a.a = {nullptr,
           p.head_mom,
           rg.slot(r, fn::xchg::msg::hc, s),
           h,
           d.w.head.gamma,
           nullptr,
           rg.slot(r, fn::xchg::msg::hmix, s),
           rg.slot(1 - r, fn::xchg::msg::hmix, s),
           p.c.ws.hc_runs,
           rg.err(r),
           r};
    a.w = d.w.head.q8_up;
    if (!fn::hc::head_up2_q8(rows, a, p.c.stream)) {
        return fail(err, "draft head up2");
    }
    note_piece(p.c, "draft.head.up2");
    return true;
}

// the logits of step j on this card's half of the subset (reading the head's X-HMIX) and greedy's reduce (sends
// T-GREEDY, slot 0); decide reads it (the subset's greedy token, lower ID on an exact tie) into desc argmax[j]
bool pick(pass_card & p, draft_card & d, int32_t j, bool decide, const use_counters & u, std::string & err) {
    const int32_t r = p.c.card;
    const fn::xchg::rings & rg = *p.c.rings;
    float * lg = d.logits + int64_t(j) * NV;
    if (d.sampled) {
        // SP1: SM1's pick (select sends samp_a; tk_decide reads it), position n_before + j + 1 read at each replay
        fn::samp::pick_io io;
        io.logits = lg;
        io.n = NV;
        io.ids = d.head_ids;
        io.live = p.desc;
        io.live_offset = j + 1;
        io.t = j;
        io.dtok = p.desc->argmax;
        io.n_dtok = j;
        io.proposal = p.desc->argmax + j;
        if (!decide) {
            const fn::head::logits_args la = {
                rg.slot(r, fn::xchg::msg::hmix, hc_slot_of(u, u.sub)), rg.err(r), r, d.head_w, NV, lg, d.logits_work};
            if (!fn::head::logits(1, la, p.c.stream) || !fn::samp::pick_stage(p.samp, io, 0, p.c.stream)) {
                return fail(err, "draft logits / sampled pick select");
            }
            note_piece(p.c, "draft.pick.select");
            return true;
        }
        if (!fn::samp::pick_stage(p.samp, io, 1, p.c.stream)) {
            return fail(err, "draft sampled pick decide");
        }
        note_piece(p.c, "draft.pick.tk_decide");
        return true;
    }
    if (!decide) {
        const fn::head::logits_args la = {
            rg.slot(r, fn::xchg::msg::hmix, hc_slot_of(u, u.sub)), rg.err(r), r, d.head_w, NV, lg, d.logits_work};
        fn::head::reduce_args ra = {lg, NV, 0, p.pair_own, rg.slot(1 - r, fn::xchg::msg::greedy, 0), p.greedy_work};
        ra.ids = d.head_ids;
        if (!fn::head::logits(1, la, p.c.stream) || !fn::head::greedy_reduce(1, ra, p.c.stream)) {
            return fail(err, "draft logits / reduce");
        }
        note_piece(p.c, "draft.pick.reduce");
        return true;
    }
    const fn::head::decide_args da = {
        p.pair_own, rg.slot(r, fn::xchg::msg::greedy, 0), p.desc->argmax + j, rg.err(r), nullptr, r};
    if (!fn::head::greedy_decide(1, da, p.c.stream)) {
        return fail(err, "draft decide");
    }
    note_piece(p.c, "draft.pick.decide");
    return true;
}

// SP2 (sparse catch-up): the rows' index keys into the draft index at cells n_before + t and the blocks they
// complete pooled (the target's round form)
bool index_update(pass_card & p, draft_card & d, int32_t rows, std::string & err) {
    fn::qsa::round_update_args ua;
    ua.keys = d.idx_keys;
    ua.k_norm = d.iw.k_norm;
    ua.summaries = d.idx_summ;
    ua.rope = fn::qsa::model_rope();
    ua.ik = p.c.ws.ik;
    ua.depth = &p.desc->n_before;
    ua.rows = rows;
    ua.max_cells = d.max_cells;
    if (p.c.kv != nullptr) { // ME6: the host-backed tier's key window (kv_tier.h)
        ua.keys = p.c.kv->keys(p.c.card, kv_tier::draft_attn);
        ua.shift = p.c.kv->shift(p.c.card);
    }
    if (!fn::qsa::summary_update(ua, p.c.stream)) {
        return fail(err, "draft summary_update");
    }
    note_piece(p.c, "draft.summary");
    return true;
}

// Phase ph of the round on one card. Every phase that receives a message follows, in phase order, the phase in which
// both cards sent it, so the pair form (phase by phase, card 0 then card 1) never waits on an unsent message.
bool phase(pass_card & p, draft_card & d, int32_t kind, int32_t rows, int32_t ph, use_counters & u, std::string & err) {
    if (!cuda_ok(cudaSetDevice(p.c.device), "cudaSetDevice", err)) {
        return false;
    }
    if (!is_step(kind)) {
        switch (ph) {
            case 0:
                return begin(p, err) && embed(p, p.desc->tok, rows, err) &&
                       draft_in(p, d, rows, false, true, true, u, err);
            case 1:
                if (!draft_in(p, d, rows, false, true, false, u, err)) {
                    return false;
                }
                u.draft++;
                return eh_proj(p, d, rows, err) && hc_sublayer(p.c, d.w.hc_attn, 0, attn_input(p, d), rows, u, 0, err);
            case 2:
                return hc_sublayer(p.c, d.w.hc_attn, 0, attn_input(p, d), rows, u, 1, err);
            default: {
                fn::dense::product_args kv;
                kv.k = D;
                kv.n_mats = d.sparse ? 3 : 2;
                kv.mats[0] = qmat(d.w.wk, p.c.ws.kv_k);
                kv.mats[1] = qmat(d.w.wv, p.c.ws.kv_v);
                if (d.sparse) {
                    kv.mats[2] = bmat16(d.iw.k_proj, fn::qsa::head_dim, p.c.ws.ik); // SP2: the index key
                }
                if (!dense_run(p.c, kv, rows, d.sparse ? dense_site::draft_kv_idx : dense_site::draft_kv, true, u,
                               "draft.kv", err) ||
                    !prepare(p, d, rows, 0, true, u, err) || (d.sparse && !index_update(p, d, rows, err))) {
                    return false;
                }
                u.sub++;
                return finish(p, err);
            }
        }
    }
    const int32_t j = ph / step_phases;
    switch (ph % step_phases) {
        case 0:
            return (j > 0 || begin(p, err)) && embed(p, j == 0 ? p.desc->tok : p.desc->argmax + (j - 1), 1, err) &&
                   draft_in(p, d, 1, j > 0, false, true, u, err);
        case 1:
            if (!draft_in(p, d, 1, j > 0, false, false, u, err)) {
                return false;
            }
            u.draft++;
            return eh_proj(p, d, 1, err) && hc_sublayer(p.c, d.w.hc_attn, 0, attn_input(p, d), 1, u, 0, err);
        case 2:
            return hc_sublayer(p.c, d.w.hc_attn, 0, attn_input(p, d), 1, u, 1, err);
        case 3:
            return attn_body(p, d, j, u, err);
        case 4:
            if (!half_exchange(p.c, 0, 1, use_slot_of(u, u.a, u.graph_a), false, err)) {
                return false;
            }
            u.sub++;
            u.a++;
            return hc_sublayer(p.c, d.w.hc_ffn, 1, ffn_input(p), 1, u, 0, err);
        case 5:
            return hc_sublayer(p.c, d.w.hc_ffn, 1, ffn_input(p), 1, u, 1, err);
        case 6:
            return ffn_body(p, d, 1, u, err);
        case 7:
            if (!half_exchange(p.c, 1, 1, use_slot_of(u, u.f, u.graph_f), false, err)) {
                return false;
            }
            u.sub++;
            u.f++;
            return head(p, d, 0, 1, d.h_nextn, u, err);
        case 8:
            return head(p, d, 1, 1, d.h_nextn, u, err);
        case 9:
            return pick(p, d, j, false, u, err);
        default:
            if (!pick(p, d, j, true, u, err)) {
                return false;
            }
            u.sub++;
            return j < rows - 1 || finish(p, err);
    }
}

int32_t n_phases(int32_t kind, int32_t rows) {
    return is_step(kind) ? rows * step_phases : catchup_phases;
}

// ---- FMU2 (build item 2): the joint draft step (draft.h draft_joint) ----

constexpr int32_t L = fn::xchg::max_lanes;
constexpr int32_t IQ = fn::qsa::n_idx_head * fn::qsa::head_dim; // the index query's floats a row

bool joint_check(const draft_joint & j, int32_t rows, std::string & err) {
    if (j.lanes < 2 || j.lanes > L || rows < 1 || rows > draft_max_steps || j.tok == nullptr || j.carry == nullptr ||
        j.h == nullptr || j.one == nullptr) {
        return fail(err, "joint draft: lanes outside 2 .. 4, rows outside 1 .. 4, or its buffers missing");
    }
    for (int32_t k = 0; k < j.lanes; ++k) {
        const pass_card * p = j.p[k];
        const draft_card * d = j.d[k];
        if (p == nullptr || d == nullptr || j.p[0] == nullptr || j.d[0] == nullptr || d->sampled != j.d[0]->sampled ||
            d->sparse != j.d[0]->sparse || d->shared_route != nullptr || p->c.kv != nullptr ||
            p->c.rings != j.p[0]->c.rings || p->c.card != j.p[0]->c.card || p->c.device != j.p[0]->c.device ||
            p->c.stream != j.p[0]->c.stream || (d->sparse && (j.sel_body[k] == nullptr || j.sel_off[k] == nullptr))) {
            return fail(err, "joint draft: lane " + std::to_string(k) +
                                 " has another pick kind or draft form, is on the host-backed KV, or not on lane 0's "
                                 "frame");
        }
    }
    return true;
}

// step j's X-DRAFT-IN send of the U rows: row k's hidden input is carry row k (lane k's retained carry) at step 0,
// then its head output row (hsrc = 1, not chained)
bool joint_in_send(draft_joint & J, int32_t j, const use_counters & u, std::string & err) {
    pass_card & p = *J.p[0];
    draft_card & d = *J.d[0];
    const int32_t r = p.c.card;
    fn::draft::send_args a;
    a.emb = p.emb_hres;
    a.emb_stride = 4 * F;
    a.carry = j == 0 ? J.carry : J.h;
    a.carry_prev = d.carry_prev; // not read: hidden row hsrc + i - 1 >= 0
    a.h_nextn = d.h_nextn;
    a.hsrc = J.one;
    a.chained = false;
    a.g_e = d.w.enorm;
    a.g_h = d.w.hnorm;
    a.peer = p.c.rings->slot(1 - r, fn::xchg::msg::draft, use_slot_of(u, u.draft, u.graph_draft));
    a.own = d.in_own;
    if (!fn::draft::draft_in_send(J.lanes, a, p.c.stream)) {
        return fail(err, "joint draft_in_send");
    }
    note_piece(p.c, "draft.in.send");
    return true;
}

// step j's attention body of the U rows: q/k/v (acquiring X-HMIX), then each lane's DA-TC prepare, SP2 select (sparse
// step 0) and attend on its own cells and history, the output product and X-A's send
bool joint_attn(draft_joint & J, int32_t j, const use_counters & u, std::string & err) {
    pass_card & p = *J.p[0];
    draft_card & d = *J.d[0];
    layer_ws & ws = p.c.ws;
    const int32_t U = J.lanes;
    const bool sel = d.sparse && j == 0;
    fn::dense::product_args in;
    in.k = D;
    in.n_mats = sel ? 4 : 3;
    in.mats[0] = qmat(d.w.wq, ws.qfull);
    in.mats[1] = qmat(d.w.wk, ws.kv_k);
    in.mats[2] = qmat(d.w.wv, ws.kv_v);
    if (sel) {
        in.mats[3] = bmat16(d.iw.q_proj, IQ, ws.iq);
    }
    if (!dense_run(p.c, in, U, sel ? dense_site::draft_qkv_idx : dense_site::draft_qkv, true, u, "draft.qkv", err)) {
        return false;
    }
    for (int32_t k = 0; k < U; ++k) {
        pass_card & pk = *J.p[k];
        draft_card & dk = *J.d[k];
        uint32_t * e = pk.c.rings->err(pk.c.card);
        fn::draft::prepare_args a;
        a.qfull = ws.qfull + int64_t(k) * QF;
        a.kv_k = ws.kv_k + int64_t(k) * HD;
        a.kv_v = ws.kv_v + int64_t(k) * HD;
        a.q_norm = d.w.q_norm;
        a.k_norm = d.w.k_norm;
        a.n_before = &pk.desc->n_before;
        a.pos = pk.desc->pos;
        a.stride = &pk.desc->rows;
        a.row0 = j;
        a.theta_scale = fn::qsa::attn::rope_theta_scale();
        a.k_cache = dk.k_cache;
        a.v_cache = dk.v_cache;
        a.max_cells = dk.max_cells;
        a.q16 = dk.q16;
        a.q_f32 = nullptr;
        a.k_f32 = nullptr;
        a.err = e;
        a.release_hc = nullptr;
        a.release_hmix = nullptr;
        if (!fn::draft::datc_prepare(1, a, p.c.stream)) {
            return fail(err, "joint datc_prepare");
        }
        note_piece(p.c, "draft.datc.prepare");
        if (sel) {
            if (!fn::draft::select_depth(&pk.desc->n_before, dk.sel_depth, p.c.stream)) {
                return fail(err, "joint draft select depth");
            }
            fn::qsa::select_args sa;
            sa.summaries = dk.idx_summ;
            sa.q = ws.iq + int64_t(k) * IQ;
            sa.q_norm = d.iw.q_norm;
            sa.depth = dk.sel_depth;
            sa.max_cells = dk.max_cells;
            sa.rope = fn::qsa::model_rope();
            sa.ws = fn::qsa::carve(p.c.qsa_ws);
            sa.body = J.sel_body[k];
            sa.offsets = J.sel_off[k];
            sa.q_ahead = 1;
            if (!fn::qsa::select(1, sa, p.c.stream)) {
                return fail(err, "joint draft select");
            }
            note_piece(p.c, "draft.select");
        }
        fn::draft::attend_args t;
        t.q16 = dk.q16;
        t.qfull = ws.qfull + int64_t(k) * QF;
        t.k_cache = dk.k_cache;
        t.v_cache = dk.v_cache;
        t.n_before = &pk.desc->n_before;
        t.row0 = j;
        t.max_cells = dk.max_cells;
        t.part = dk.part;
        t.out = ws.attn + int64_t(k) * fn::draft::out_width;
        t.codes = ws.mid_codes + fn::act::codes_bytes(k, fn::draft::out_width);
        t.scales = ws.mid_scales + fn::act::scales_bytes(k, fn::draft::out_width) / sizeof(float);
        t.err = e;
        t.body = d.sparse ? J.sel_body[k] : nullptr;
        if (!fn::draft::datc_attend(t, p.c.stream)) {
            return fail(err, "joint datc_attend");
        }
        note_piece(p.c, "draft.datc.attend");
    }
    fn::dense::product_args out;
    out.k = QH * HD;
    out.n_mats = 1;
    out.mats[0] = qmat(d.w.wo, ws.partial);
    out.x_codes = ws.mid_codes;
    out.x_scales = ws.mid_scales;
    return dense_run(p.c, out, U, dense_site::draft_out, false, u, "draft.out", err) &&
           half_exchange(p.c, 0, U, use_slot_of(u, u.a, u.graph_a), true, err);
}

// FMU4: step j's sampled pick of the U rows (every lane sampled): stage 0 (select) or 1 (tk_decide) of lane k's lone
// pick (draft.cpp pick) on logits row k, with its own sampler view, q_list row j, live position, deltas and proposal
// slot desc argmax[j]; its list in row k of samp_a's slot (samp::pick_lanes_stage), as the lanes share lane 0's rings.
// After the decides the proposals are gathered into tok row j + 1 for the next step's embedding.
bool joint_pick_sampled(draft_joint & J, int32_t j, int32_t stage, int32_t rows, std::string & err) {
    pass_card & p = *J.p[0];
    draft_card & d = *J.d[0];
    const int32_t U = J.lanes;
    const fn::samp::card_view * v[L] = {};
    fn::samp::pick_io io[L];
    for (int32_t k = 0; k < U; ++k) {
        v[k] = &J.p[k]->samp;
        io[k].logits = d.logits + int64_t(k) * NV;
        io[k].n = NV;
        io[k].ids = d.head_ids;
        io[k].live = J.p[k]->desc;
        io[k].live_offset = j + 1;
        io[k].t = j;
        io[k].dtok = J.p[k]->desc->argmax;
        io[k].n_dtok = j;
        io[k].proposal = J.p[k]->desc->argmax + j;
    }
    if (!fn::samp::pick_lanes_stage(v, io, U, stage, p.c.stream)) {
        return fail(err, stage == 0 ? "joint draft sampled pick select" : "joint draft sampled pick decide");
    }
    note_piece(p.c, stage == 0 ? "draft.pick.select" : "draft.pick.tk_decide");
    if (stage == 0 || j == rows - 1) {
        return true;
    }
    fn::draft::copy_seg g[fn::draft::max_copy_segs];
    for (int32_t k = 0; k < U; ++k) {
        g[k] = {reinterpret_cast<const uint32_t *>(J.p[k]->desc->argmax + j),
                reinterpret_cast<uint32_t *>(J.tok + (j + 1) * L + k), 1, 1};
    }
    if (!fn::draft::lanes_copy(U, g, p.c.stream)) {
        return fail(err, "joint draft proposals gather");
    }
    note_piece(p.c, "draft.joint.gather");
    return true;
}

// step j's logits of the U rows and greedy's reduce (T-GREEDY slot 0), or its decide into tok row j + 1; sampled
// lanes: the logits, then each lane's select, or each lane's decide (joint_pick_sampled)
bool joint_pick(draft_joint & J, int32_t j, bool decide, int32_t rows, const use_counters & u, std::string & err) {
    pass_card & p = *J.p[0];
    draft_card & d = *J.d[0];
    const int32_t r = p.c.card;
    const fn::xchg::rings & rg = *p.c.rings;
    if (d.sampled) {
        if (!decide) {
            const fn::head::logits_args la = {rg.slot(r, fn::xchg::msg::hmix, hc_slot_of(u, u.sub)),
                                              rg.err(r),
                                              r,
                                              d.head_w,
                                              NV,
                                              d.logits,
                                              d.logits_work};
            if (!fn::head::logits(J.lanes, la, p.c.stream)) {
                return fail(err, "joint draft logits");
            }
        }
        return joint_pick_sampled(J, j, decide ? 1 : 0, rows, err);
    }
    if (!decide) {
        const fn::head::logits_args la = {
            rg.slot(r, fn::xchg::msg::hmix, hc_slot_of(u, u.sub)), rg.err(r), r, d.head_w, NV, d.logits, d.logits_work};
        fn::head::reduce_args ra = {d.logits,     NV, 0, p.pair_own, rg.slot(1 - r, fn::xchg::msg::greedy, 0),
                                    p.greedy_work};
        ra.ids = d.head_ids;
        if (!fn::head::logits(J.lanes, la, p.c.stream) || !fn::head::greedy_reduce(J.lanes, ra, p.c.stream)) {
            return fail(err, "joint draft logits / reduce");
        }
        note_piece(p.c, "draft.pick.reduce");
        return true;
    }
    const fn::head::decide_args da = {
        p.pair_own, rg.slot(r, fn::xchg::msg::greedy, 0), J.tok + (j + 1) * L, rg.err(r), nullptr, r};
    if (!fn::head::greedy_decide(J.lanes, da, p.c.stream)) {
        return fail(err, "joint draft decide");
    }
    note_piece(p.c, "draft.pick.decide");
    return true;
}

// Phase ph of the joint step on one card, in the lone step's phase order (draft.cpp phase)
bool joint_phase(draft_joint & J, int32_t rows, int32_t ph, use_counters & u, std::string & err) {
    pass_card & p = *J.p[0];
    draft_card & d = *J.d[0];
    const int32_t U = J.lanes;
    if (!cuda_ok(cudaSetDevice(p.c.device), "cudaSetDevice", err)) {
        return false;
    }
    const int32_t j = ph / step_phases;
    switch (ph % step_phases) {
        case 0:
            if (j == 0) {
                // every lane's round begun; its token and retained carry gathered into row k
                fn::draft::copy_seg g[fn::draft::max_copy_segs];
                for (int32_t k = 0; k < U; ++k) {
                    if (!begin(*J.p[k], err)) {
                        return false;
                    }
                    g[2 * k] = {reinterpret_cast<const uint32_t *>(J.p[k]->desc->tok),
                                reinterpret_cast<uint32_t *>(J.tok + k), 1, 1};
                    g[2 * k + 1] = {reinterpret_cast<const uint32_t *>(J.d[k]->carry_prev),
                                    reinterpret_cast<uint32_t *>(J.carry + int64_t(k) * 4 * F), 4 * F, 1};
                }
                if (!fn::draft::lanes_copy(2 * U, g, p.c.stream)) {
                    return fail(err, "joint draft gather");
                }
                note_piece(p.c, "draft.joint.gather");
            }
            return embed(p, J.tok + j * L, U, err) && joint_in_send(J, j, u, err);
        case 1:
            if (!draft_in(p, d, U, false, false, false, u, err)) {
                return false;
            }
            u.draft++;
            return eh_proj(p, d, U, err) && hc_sublayer(p.c, d.w.hc_attn, 0, attn_input(p, d), U, u, 0, err);
        case 2:
            return hc_sublayer(p.c, d.w.hc_attn, 0, attn_input(p, d), U, u, 1, err);
        case 3:
            return joint_attn(J, j, u, err);
        case 4:
            if (!half_exchange(p.c, 0, U, use_slot_of(u, u.a, u.graph_a), false, err)) {
                return false;
            }
            u.sub++;
            u.a++;
            return hc_sublayer(p.c, d.w.hc_ffn, 1, ffn_input(p), U, u, 0, err);
        case 5:
            return hc_sublayer(p.c, d.w.hc_ffn, 1, ffn_input(p), U, u, 1, err);
        case 6:
            return ffn_body(p, d, U, u, err);
        case 7:
            if (!half_exchange(p.c, 1, U, use_slot_of(u, u.f, u.graph_f), false, err)) {
                return false;
            }
            u.sub++;
            u.f++;
            return head(p, d, 0, U, J.h, u, err);
        case 8:
            return head(p, d, 1, U, J.h, u, err);
        case 9:
            return joint_pick(J, j, false, rows, u, err);
        default: {
            if (!joint_pick(J, j, true, rows, u, err)) {
                return false;
            }
            u.sub++;
            if (j < rows - 1) {
                return true;
            }
            // each lane's picks into its descriptor's argmax[0 .. d - 1] (sampled picks are there already), then its
            // commit and finish
            if (!d.sampled) {
                fn::draft::copy_seg g[fn::draft::max_copy_segs];
                for (int32_t k = 0; k < U; ++k) {
                    g[k] = {reinterpret_cast<const uint32_t *>(J.tok + L + k),
                            reinterpret_cast<uint32_t *>(J.p[k]->desc->argmax), rows, L};
                }
                if (!fn::draft::lanes_copy(U, g, p.c.stream)) {
                    return fail(err, "joint draft scatter");
                }
                note_piece(p.c, "draft.joint.scatter");
            }
            for (int32_t k = 0; k < U; ++k) {
                if (!finish(*J.p[k], err)) {
                    return false;
                }
            }
            return true;
        }
    }
}

uint8_t * class_part(const model & m, const char * cls, const char * name, int32_t card, std::string & err) {
    const reserved_class * rc = m.state_class(cls);
    const int32_t i = part_of(rc, name);
    uint8_t * q = i < 0 ? nullptr : rc->part_ptr(i, card);
    if (q == nullptr) {
        err = std::string(cls) + " / " + name + ": not reserved on card " + std::to_string(card);
    }
    return q;
}

int64_t class_part_bytes(const model & m, const char * cls, const char * name, int32_t card) {
    const reserved_class * rc = m.state_class(cls);
    const int32_t i = part_of(rc, name);
    return i < 0 ? 0 : rc->plan.parts[size_t(i)].bytes[card];
}

// SP2: the draft index's buffers on this card; card 1 copies the indexer's weights from card 0's regions
bool draft_index_init(model & m, int32_t card, draft_card & d, std::string & err) {
    const char * X = "draft index (SP2)";
    // ME6: with the host-backed tier the keys' class is their host history
    const char * XK = m.state_class("draft index keys (ME6)") != nullptr ? "draft index keys (ME6)" : X;
    d.idx_keys = reinterpret_cast<uint16_t *>(class_part(m, XK, "L48 keys", card, err));
    d.idx_summ = reinterpret_cast<float *>(class_part(m, X, "L48 summaries", card, err));
    d.sel_depth = reinterpret_cast<int32_t *>(class_part(m, X, "select depth", card, err));
    draft_index_weights w0;
    if (d.idx_keys == nullptr || d.idx_summ == nullptr || d.sel_depth == nullptr ||
        !resolve_draft_index_weights(m, w0, err)) {
        return false;
    }
    if (class_part_bytes(m, XK, "L48 keys", card) < int64_t(d.max_cells) * fn::qsa::head_dim * 2 ||
        class_part_bytes(m, X, "L48 summaries", card) <
            static_cast<int64_t>(fn::qsa::summary_bytes(fn::qsa::blocks_complete(d.max_cells)))) {
        return fail(err, "draft index: keys or summaries smaller than the draft KV's cells");
    }
    if (card == 0) {
        d.iw = w0;
        return true;
    }
    const void * src[4] = {w0.q_proj, w0.k_proj, w0.q_norm, w0.k_norm};
    const char * names[4] = {"indexer q_proj copy", "indexer k_proj copy", "indexer q_norm copy",
                             "indexer k_norm copy"};
    const int64_t bytes[4] = {int64_t(fn::qsa::n_idx_head) * fn::qsa::head_dim * D * 2,
                              int64_t(fn::qsa::head_dim) * D * 2, int64_t(fn::qsa::head_dim) * 4,
                              int64_t(fn::qsa::head_dim) * 4};
    uint8_t * dst[4] = {};
    for (int32_t i = 0; i < 4; ++i) {
        dst[i] = class_part(m, X, names[i], card, err);
        if (dst[i] == nullptr || class_part_bytes(m, X, names[i], card) != bytes[i]) {
            return err.empty() ? fail(err, std::string("draft index: ") + names[i] + " is not its weight's size")
                               : false;
        }
        if (!cuda_ok(cudaMemcpy(dst[i], src[i], size_t(bytes[i]), cudaMemcpyDefault), "draft index weight copy",
                     err)) {
            return false;
        }
    }
    d.iw.q_proj = reinterpret_cast<const uint16_t *>(dst[0]);
    d.iw.k_proj = reinterpret_cast<const uint16_t *>(dst[1]);
    d.iw.q_norm = reinterpret_cast<const float *>(dst[2]);
    d.iw.k_norm = reinterpret_cast<const float *>(dst[3]);
    return true;
}

} // namespace

bool read_draft_subset(const std::string & path, std::vector<int32_t> & ids, std::string & err) {
    ids.clear();
    std::ifstream f(path);
    if (!f) {
        return fail(err, "cannot read the draft subset " + path);
    }
    std::vector<uint8_t> seen(size_t(fn::head::n_vocab), 0);
    long long v = 0;
    while (f >> v) {
        if (v < 0 || v >= fn::head::n_vocab || seen[size_t(v)] != 0) {
            return fail(err, path + ": ID " + std::to_string(v) + " out of range or repeated");
        }
        seen[size_t(v)] = 1;
        ids.push_back(static_cast<int32_t>(v));
    }
    if (!f.eof() || (int32_t)ids.size() != fn::head::n_draft_vocab) {
        return fail(err,
                    path + ": " + std::to_string(ids.size()) + " IDs, need " + std::to_string(fn::head::n_draft_vocab));
    }
    return true;
}

bool draft_head_fill(model & m, const std::vector<int32_t> & subset, std::string & err) {
    if ((int32_t)subset.size() != fn::head::n_draft_vocab) {
        return fail(err, "draft head: the subset is not 65,536 IDs");
    }
    const int32_t oi = model::find("output.weight");
    if (oi < 0 || (size_t)oi >= m.sources.size()) {
        return fail(err, "draft head: output.weight not placed");
    }
    const tensor_source & src = m.sources[size_t(oi)];
    const uint8_t * vocab = src.file != nullptr ? src.file->data() + src.offset : nullptr;
    const int64_t hb = fn::head::draft_head_bytes(NV);
    if (vocab != nullptr && (src.row_bytes != fn::dense::row_bytes(fn::dense::fmt::q5_k, D) ||
                             src.offset + uint64_t(src.row_bytes) * fn::head::n_vocab > src.file->size())) {
        return fail(err, "draft head: output.weight is not Q5_K rows of 2,560 in its file");
    }
    std::vector<int32_t> ids(size_t(NV), 0);
    std::vector<uint8_t> rows(size_t(hb), 0);
    for (int32_t card = 0; card < 2; ++card) {
        if (!fn::head::draft_ids(subset.data(), fn::head::n_draft_vocab, card, ids.data())) {
            return fail(err, "draft head: draft_ids");
        }
        std::fill(rows.begin(), rows.end(), uint8_t(0));
        if (vocab != nullptr && !fn::head::pack_draft_head(vocab, size_t(src.row_bytes), ids.data(), NV, rows.data())) {
            return fail(err, "draft head: pack_draft_head");
        }
        uint8_t * w = class_part(m, "draft head copy (DR1)", "draft head rows (Q5_K, packed)", card, err);
        uint8_t * map = class_part(m, "draft head copy (DR1)", "draft head ID map", card, err);
        if (w == nullptr || map == nullptr) {
            return false;
        }
        if (class_part_bytes(m, "draft head copy (DR1)", "draft head rows (Q5_K, packed)", card) != hb) {
            return fail(err, "draft head: the reservation's copy is not draft_head_bytes(32,768)");
        }
        if (!cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemcpy(w, rows.data(), rows.size(), cudaMemcpyHostToDevice), "draft head upload", err) ||
            !cuda_ok(cudaMemcpy(map, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "draft head IDs", err)) {
            return false;
        }
    }
    return true;
}

bool draft_init(model & m, const pass_card & p, draft_card & d, std::string & err) {
    d = draft_card();
    const int32_t card = p.c.card;
    if (!resolve_draft_weights(m, card, d.w, err)) {
        return false;
    }
    const char * R = "draft round (DR1)";
    d.head_w = class_part(m, "draft head copy (DR1)", "draft head rows (Q5_K, packed)", card, err);
    d.head_ids =
        reinterpret_cast<const int32_t *>(class_part(m, "draft head copy (DR1)", "draft head ID map", card, err));
    d.k_cache = reinterpret_cast<uint16_t *>(class_part(m, "draft KV (F16)", "L48 K", card, err));
    d.v_cache = reinterpret_cast<uint16_t *>(class_part(m, "draft KV (F16)", "L48 V", card, err));
    d.carry_prev = reinterpret_cast<float *>(class_part(m, R, "retained carry", card, err));
    d.h_nextn = reinterpret_cast<float *>(class_part(m, R, "draft hidden output", card, err));
    d.in_own = reinterpret_cast<float *>(class_part(m, R, "X-DRAFT-IN own payload", card, err));
    d.codes = reinterpret_cast<int8_t *>(class_part(m, R, "eh_proj input codes", card, err));
    d.scales = reinterpret_cast<float *>(class_part(m, R, "eh_proj input scales", card, err));
    d.hres_d = reinterpret_cast<float *>(class_part(m, R, "eh_proj output", card, err));
    d.q16 = reinterpret_cast<uint16_t *>(class_part(m, R, "DA-TC q16", card, err));
    d.part = reinterpret_cast<float *>(class_part(m, R, "DA-TC partials", card, err));
    d.logits = reinterpret_cast<float *>(class_part(m, R, "draft logits", card, err));
    uint8_t * work = class_part(m, R, "draft head work", card, err);
    if (d.head_w == nullptr || d.head_ids == nullptr || d.k_cache == nullptr || d.v_cache == nullptr ||
        d.carry_prev == nullptr || d.h_nextn == nullptr || d.in_own == nullptr || d.codes == nullptr ||
        d.scales == nullptr || d.hres_d == nullptr || d.q16 == nullptr || d.part == nullptr || d.logits == nullptr ||
        work == nullptr) {
        return false;
    }
    d.logits_work = fn::head::logits_work_bytes(NV) > 0 ? work : nullptr;
    if (!draft_index_init(m, card, d, err)) {
        return false;
    }
    const int64_t kb = class_part_bytes(m, "draft KV (F16)", "L48 K", card);
    d.max_cells = static_cast<int32_t>(kb / (int64_t(HD) * 2));
    if (d.max_cells < 1 || class_part_bytes(m, "draft KV (F16)", "L48 V", card) != kb) {
        return fail(err, "draft KV: K and V parts differ or are empty");
    }
    const reserved_class * rc = m.state_class(R);
    if (!cuda_ok(cudaSetDevice(p.c.device), "cudaSetDevice", err)) {
        return false;
    }
    for (int32_t i = 0; i < (int32_t)rc->plan.parts.size(); ++i) {
        const int64_t n = rc->plan.parts[size_t(i)].bytes[card];
        if (n > 0 && !cuda_ok(cudaMemset(rc->part_ptr(i, card), 0, size_t(n)), "draft round buffers", err)) {
            return false;
        }
    }
    return cuda_ok(cudaDeviceSynchronize(), "draft init", err);
}

use_counters draft_graph_counters(int32_t kind, int32_t rows) {
    use_counters u;
    u.graph = true;
    if (is_step(kind)) {
        u.rule = hc_rule::draft_step;
        u.graph_subs = 3 * rows;
        u.graph_a = u.graph_f = u.graph_draft = rows;
    } else {
        u.rule = hc_rule::catchup;
        u.graph_subs = 1;
        u.graph_a = u.graph_f = u.graph_draft = 1;
    }
    return u;
}

step_status draft_card_enqueue(pass_card & p, draft_card & d, int32_t kind, int32_t rows, use_counters & u,
                               std::string & err) {
    if (!check_round(kind, rows, err)) {
        return step_status::failed;
    }
    for (int32_t ph = 0; ph < n_phases(kind, rows); ++ph) {
        if (!phase(p, d, kind, rows, ph, u, err)) {
            return step_status::failed;
        }
    }
    return step_status::ok;
}

step_status draft_pair_enqueue(pass_card * const p[2], draft_card * const d[2], int32_t kind, int32_t rows,
                               use_counters u[2], std::string & err) {
    if (!check_round(kind, rows, err)) {
        return step_status::failed;
    }
    // ME8: on one device and stream card 1's FFN reads card 0's routing; each phase runs card 0 before card 1, so
    // card 0's router precedes both cards' experts in the stream. Set for this call only, cleared on every exit
    const card_ctx & c0 = p[0]->c, & c1 = p[1]->c;
    const bool one = c0.device == c1.device && c0.stream == c1.stream && c0.stream != nullptr;
    const route_view r0 = {c0.ex.ids, c0.ex.wts, c0.ws.sg};
    struct clear_route {
        draft_card & d;
        ~clear_route() { d.shared_route = nullptr; }
    } clear{*d[1]};
    d[1]->shared_route = one ? &r0 : nullptr;
    for (int32_t ph = 0; ph < n_phases(kind, rows); ++ph) {
        for (int32_t k = 0; k < 2; ++k) {
            if (!phase(*p[k], *d[k], kind, rows, ph, u[k], err)) {
                return step_status::failed;
            }
        }
    }
    return step_status::ok;
}

step_status draft_joint_enqueue(draft_joint & j, int32_t rows, use_counters & u, std::string & err) {
    if (!joint_check(j, rows, err)) {
        return step_status::failed;
    }
    for (int32_t ph = 0; ph < rows * step_phases; ++ph) {
        if (!joint_phase(j, rows, ph, u, err)) {
            return step_status::failed;
        }
    }
    return step_status::ok;
}

step_status draft_joint_pair_enqueue(draft_joint * const j[2], int32_t rows, use_counters u[2], std::string & err) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!joint_check(*j[c], rows, err)) {
            return step_status::failed;
        }
    }
    if (j[0]->lanes != j[1]->lanes || j[0]->p[0]->c.card != 0 || j[1]->p[0]->c.card != 1 ||
        j[0]->p[0]->c.device == j[1]->p[0]->c.device) {
        fail(err, "joint draft: the cards' lanes differ, or both cards on one device");
        return step_status::failed;
    }
    for (int32_t ph = 0; ph < rows * step_phases; ++ph) {
        for (int32_t c = 0; c < 2; ++c) {
            if (!joint_phase(*j[c], rows, ph, u[c], err)) {
                return step_status::failed;
            }
        }
    }
    return step_status::ok;
}

} // namespace fn::model
