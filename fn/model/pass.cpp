#include "fn/model/pass.h"

#include "fn/kernels/embed/embed.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/hc/hc.h"
#include "fn/kernels/head/head.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/shapes.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace fn::model {

namespace {

constexpr int32_t F = fn::hc::n_own; // 1280

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

int64_t align256(int64_t n) {
    return (n + 255) / 256 * 256;
}

int32_t part_of(const reserved_class * rc, const std::string & name) {
    for (int32_t i = 0; rc != nullptr && i < (int32_t)rc->plan.parts.size(); ++i) {
        if (rc->plan.parts[i].name == name) {
            return i;
        }
    }
    return -1;
}

const uint8_t * tensor_ptr(const model & m, const char * name, int32_t card, std::string & err) {
    const int32_t idx = model::find(name);
    const uint8_t * p = idx < 0 ? nullptr : m.region_ptr(idx, card);
    if (p == nullptr) {
        err = std::string(name) + ": not on card " + std::to_string(card);
    }
    return p;
}

bool set_device(const pass_card & p, std::string & err) {
    return cuda_ok(cudaSetDevice(p.c.device), "cudaSetDevice", err);
}

// ---- the pieces outside the layers, one card each ----

bool round_begin(pass_card & p, std::string & err) {
    const fn::round::begin_args a = {p.host_ring, p.flag, p.desc, p.rstate, p.c.rings->err(p.c.card)};
    // ME3: the failure names the pending CUDA error (the launch check peeks at it)
    return set_device(p, err) &&
           (fn::round::begin(a, p.c.stream) && fn::round::stamp(p.rstate, p.c.stream)
                ? true
                : fail(err, std::string("round begin: ") + cudaGetErrorString(cudaPeekAtLastError())));
}

bool embed(pass_card & p, int32_t rows, std::string & err) {
    fn::embed::args a;
    a.table = p.token_embd;
    a.grids = p.c.dense_grids;
    a.tok = p.desc->tok;
    a.hres = p.emb_hres;
    a.hpost = p.emb_hpost;
    a.hx = p.emb_hx;
    a.err = p.c.rings->err(p.c.card);
    return set_device(p, err) && (fn::embed::embed(rows, a, p.c.stream) ? true : fail(err, "embed"));
}

// TTTS1: the received boundary of a back stage (pass_stage::in) into the embedding boundary's buffers, layer lo's
// input, host to device in stream order after the round's begin (the host launched the round after the receive)
bool stage_in(pass_card & p, int32_t rows, const pass_options & o, std::string & err) {
    const uint8_t * src = o.stage.recv[p.c.card];
    cudaStream_t    s   = static_cast<cudaStream_t>(p.c.stream);
    return src != nullptr && set_device(p, err) &&
           cuda_ok(cudaMemcpyAsync(p.emb_hres, src, size_t(rows) * 4 * F * 4, cudaMemcpyHostToDevice, s),
                   "stage in (hres)", err) &&
           cuda_ok(cudaMemcpyAsync(p.emb_hpost, src + stage_hpost_at(rows), size_t(rows) * 4 * 4,
                                   cudaMemcpyHostToDevice, s),
                   "stage in (hpost)", err) &&
           cuda_ok(cudaMemcpyAsync(p.emb_hx, src + stage_hx_at(rows), size_t(rows) * F * 4, cudaMemcpyHostToDevice, s),
                   "stage in (hx)", err);
}

// TTTS1: a front stage's (pass_stage::out) layer hi output boundary into its send buffer, device to host in stream
// order before the round's finish, so the completion record publishes it
bool stage_out(pass_card & p, int32_t rows, const pass_options & o, std::string & err) {
    uint8_t *         dst = o.stage.send[p.c.card];
    const layer_input in  = layer_output(p.c);
    cudaStream_t      s   = static_cast<cudaStream_t>(p.c.stream);
    return dst != nullptr && set_device(p, err) &&
           cuda_ok(cudaMemcpyAsync(dst, in.hres, size_t(rows) * 4 * F * 4, cudaMemcpyDeviceToHost, s),
                   "stage out (hres)", err) &&
           cuda_ok(
               cudaMemcpyAsync(dst + stage_hpost_at(rows), in.hpost, size_t(rows) * 4 * 4, cudaMemcpyDeviceToHost, s),
               "stage out (hpost)", err) &&
           cuda_ok(cudaMemcpyAsync(dst + stage_hx_at(rows), in.hx, size_t(rows) * F * 4, cudaMemcpyDeviceToHost, s),
                   "stage out (hx)", err);
}

// TTTS1: a front stage's round end: the completion record only (no commit: the host's, after the back's verdict)
bool stage_finish(pass_card & p, std::string & err) {
    const fn::round::finish_args fa = { p.desc, p.rstate, p.errs, p.n_err, p.done };
    return set_device(p, err) && (fn::round::finish(fa, p.c.stream) ? true : fail(err, "stage finish"));
}

// the round's input: the embedding, or (a back stage) the received boundary
bool round_input(pass_card & p, int32_t rows, const pass_options & o, std::string & err) {
    return o.stage.in ? stage_in(p, rows, o, err) : embed(p, rows, err);
}

fn::ple::step_args ple_args(pass_card & p) {
    const int32_t r = p.c.card;
    const layer_input in = layer_output(p.c); // layer 0's output boundary
    fn::ple::step_args a;
    a.w = p.ple_w;
    a.e = p.ple_e;
    a.hres = in.hres;
    a.hpost = in.hpost;
    a.hx = in.hx;
    a.h_out = p.c.ws.x[1]; // in place: layer 1 reads it with hpost = hx = 0
    a.hist = p.ple_hist;
    a.cur = &p.rstate->cur;
    a.ws = p.ple_ws;
    a.stats_peer = p.c.rings->slot(1 - r, fn::xchg::msg::ple, 0);
    a.stats_slot = p.c.rings->slot(r, fn::xchg::msg::ple, 0);
    a.stats_reset = nullptr;
    a.use = &p.desc->ple_use;
    a.err = p.c.rings->err(r);
    a.card = r;
    return a;
}

// PLE part 0 products, 1 stats (sends X-PLE-STATS), 2 update (reads it)
bool ple_part(pass_card & p, int32_t rows, int32_t part, std::string & err) {
    if (!set_device(p, err)) {
        return false;
    }
    const fn::ple::step_args a = ple_args(p);
    const bool ok = part == 0   ? fn::ple::products(rows, a, p.c.stream)
                    : part == 1 ? fn::ple::stats(rows, a, p.c.stream)
                                : fn::ple::update(rows, a, p.c.stream);
    return ok ? true : fail(err, "ple part " + std::to_string(part));
}

// the layer after layer lo's input: layer 1's input is PLE's output with hpost = hx = 0
layer_input layer_in(const pass_card & p, int32_t layer, const pass_options & o) {
    layer_input in;
    if (layer == o.layer_lo) {
        in.hres = p.emb_hres;
        in.hpost = p.emb_hpost;
        in.hx = p.emb_hx;
    } else if (layer == 1) {
        in.hres = p.c.ws.x[1];
        in.hpost = p.c.ws.zero;
        in.hx = p.c.ws.zero;
    } else {
        in = layer_output(p.c);
    }
    return in;
}

bool records_after(pass_card & p, int32_t layer, int32_t rows, const pass_options & o, std::string & err) {
    for (int32_t i = 0; i < o.n_records; ++i) {
        const pass_record_layer & rec = o.records[p.c.card][i];
        if (rec.layer != layer) {
            continue;
        }
        cudaStream_t s = static_cast<cudaStream_t>(p.c.stream);
        const layer_ws & ws = p.c.ws;
        const auto cp = [&](void * dst, const void * src, size_t bytes) {
            return dst == nullptr ||
                   cuda_ok(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s), "record", err);
        };
        if (!set_device(p, err) || !cp(rec.x, ws.x[1], size_t(rows) * 4 * F * 4) ||
            !cp(rec.inject, ws.inject[1], size_t(rows) * 4 * 4) || !cp(rec.out, ws.out[1], size_t(rows) * F * 4)) {
            return false;
        }
    }
    return true;
}

// the mixed inputs are recorded by the layer after each body (card_ctx::record_mixed), before the next HC sublayer
// resets their X-HMIX slot (HC3): set before a recorded layer, cleared after it
void records_before(pass_card & p, int32_t layer, const pass_options & o) {
    p.c.record_mixed[0] = p.c.record_mixed[1] = nullptr;
    for (int32_t i = 0; i < o.n_records; ++i) {
        const pass_record_layer & rec = o.records[p.c.card][i];
        if (rec.layer == layer) {
            p.c.record_mixed[0] = rec.mixed_attn;
            p.c.record_mixed[1] = rec.mixed_ffn;
        }
    }
}

// the decision after the logits (FC-DEC1, FC-DEC2): V1's greedy reduce and decide; with a sampler (o.sampling) its
// stages (samp.h), or on its greedy path V1's greedy carrying the decision hash, then the sampler's verdict
struct decision_part {
    int32_t kind; // 0: greedy reduce, 1: greedy decide, 2: a sampler stage
    fn::samp::stage s;
};

std::vector<decision_part> decision_parts(const pass_options & o) {
    std::vector<decision_part> v;
    if (o.sampling == nullptr || fn::samp::path_of(*o.sampling) == fn::samp::path::greedy) {
        v.push_back({0, fn::samp::stage::apply});
        v.push_back({1, fn::samp::stage::apply});
    }
    if (o.sampling != nullptr) {
        for (fn::samp::stage st : fn::samp::stages_of(*o.sampling, false)) {
            v.push_back({2, st});
        }
    }
    return v;
}

int32_t n_head_parts(const pass_options & o) {
    return 3 + static_cast<int32_t>(decision_parts(o).size()) + 1;
}

// one decision part on card p for the rows of logits (the round's, or the bootstrap's retained row)
bool decision_part_enqueue(pass_card & p, int32_t rows, const float * logits, const decision_part & d,
                           const pass_options & o) {
    const int32_t r = p.c.card;
    const fn::xchg::rings & rg = *p.c.rings;
    const bool hash = o.sampling != nullptr;
    if (d.kind == 0) {
        // T-GREEDY's slot 0 every round: one slot suffices (xchg.h's chain for T-GREEDY)
        fn::head::reduce_args ra = {logits,
                                    fn::head::n_vocab_card,
                                    r * fn::head::n_vocab_card,
                                    p.pair_own,
                                    rg.slot(1 - r, fn::xchg::msg::greedy, 0),
                                    p.greedy_work};
        ra.hash_words = hash ? fn::samp::greedy_hash_words(p.samp) : nullptr;
        return fn::head::greedy_reduce(rows, ra, p.c.stream);
    }
    if (d.kind == 1) {
        fn::head::decide_args da = {
            p.pair_own, rg.slot(r, fn::xchg::msg::greedy, 0), p.desc->argmax, rg.err(r), nullptr, r};
        da.hash_words = hash ? fn::samp::greedy_hash_words(p.samp) : nullptr;
        da.checked = hash ? fn::samp::greedy_checked(p.samp) : nullptr;
        return fn::head::greedy_decide(rows, da, p.c.stream);
    }
    fn::samp::decide_io io;
    io.logits = logits;
    io.n = fn::head::n_vocab_card;
    io.id0 = r * fn::head::n_vocab_card;
    io.desc = p.desc;
    return fn::samp::enqueue(d.s, rows, p.samp, io, p.c.stream);
}

// the head HC's arguments on card p (hc::head_sub; ME12's hc::head_pair takes both cards')
fn::hc::sub_args head_sub_args(const pass_card & p, const use_counters & u) {
    const int32_t r = p.c.card;
    const fn::xchg::rings & rg = *p.c.rings;
    const int32_t s = hc_slot_of(u, u.sub); // the head is HC sublayer u.sub (96 of a whole round in a graph)
    const layer_input in = layer_output(p.c); // layer 47's output boundary, only read (FC-HEAD3)
    const bool first = !u.graph && u.sub == 0;
    fn::hc::sub_args a = {};
    a.w_down = p.head.w_down;
    a.w_up = p.head.w_up;
    a.hres = in.hres;
    a.hx = in.hx;
    a.hpost = in.hpost;
    a.gamma = p.head.gamma;
    a.x = p.carry;
    a.hc_ring = rg.slot(r, fn::xchg::msg::hc, s);
    a.hc_peer = rg.slot(1 - r, fn::xchg::msg::hc, s);
    a.hc_reset = first ? nullptr : rg.slot(r, fn::xchg::msg::hc, hc_slot_of(u, u.sub - 1));
    a.hmix_reset = first ? nullptr : rg.slot(r, fn::xchg::msg::hmix, hc_slot_of(u, u.sub - 1));
    a.hmix_own = rg.slot(r, fn::xchg::msg::hmix, s);
    a.hmix_peer = rg.slot(1 - r, fn::xchg::msg::hmix, s);
    a.err = rg.err(r);
    a.card = r;
    return a;
}

// the head: part 0 HC's down phase (sends X-HC), 1 its up phase (reads X-HC, sends X-HMIX in both forms), 2 logits
// (reads X-HMIX), then the decision's parts (decision_parts: each send before its read), and last commit and finish.
// fused (the graphs, HC4): parts 0 and 1 are one launch of hc::head_sub at part 0
bool head_part(pass_card & p, int32_t rows, int32_t part, const use_counters & u, const pass_options & o, bool fused,
               std::string & err) {
    if (!set_device(p, err)) {
        return false;
    }
    const int32_t r = p.c.card;
    const fn::xchg::rings & rg = *p.c.rings;
    const int32_t s = hc_slot_of(u, u.sub);
    const std::vector<decision_part> dp = decision_parts(o);
    bool ok = false;
    if (part == 0 || part == 1) {
        if (fused && part == 1) {
            return true;
        }
        ok = fn::hc::head_sub(rows, fused ? 3 : part + 1, head_sub_args(p, u), p.c.stream);
    } else if (part == 2) {
        const fn::head::logits_args la = {
            rg.slot(r, fn::xchg::msg::hmix, s), rg.err(r), r, p.out_w, fn::head::n_vocab_card, p.logits, nullptr};
        // SM1: the decision's start in context (stamp_logits), only with a sampler, so TG1's graphs are unchanged
        ok = fn::head::logits(rows, la, p.c.stream) &&
             (o.sampling == nullptr || fn::round::stamp_logits(p.rstate, p.c.stream));
    } else if (part < 3 + static_cast<int32_t>(dp.size())) {
        ok = decision_part_enqueue(p, rows, p.logits, dp[size_t(part - 3)], o);
    } else {
        const fn::round::commit_args ca = {p.desc, p.rstate, p.errs, p.n_err, p.c.card};
        const fn::round::finish_args fa = {p.desc, p.rstate, p.errs, p.n_err, p.done};
        ok = fn::round::commit(ca, p.c.stream) && fn::round::finish(fa, p.c.stream);
    }
    return ok ? true : fail(err, "head part " + std::to_string(part));
}

// ME12: the head HC of both cards on one device and stream, one launch
bool head_pair_part(pass_card * const p[2], int32_t rows, const use_counters u[2], std::string & err) {
    if (!set_device(*p[0], err)) {
        return false;
    }
    const fn::hc::pair_args a = {{head_sub_args(*p[0], u[0]), head_sub_args(*p[1], u[1])}, p[0]->c.ws.mom[1]};
    return fn::hc::head_pair(rows, a, p[0]->c.stream) ? true : fail(err, "head hc pair");
}

bool check_options(const pass_options & o, int32_t rows, std::string & err) {
    if (rows < 1 || rows > layer_max_rows || o.layer_lo < 0 || o.layer_hi >= fn::n_layer || o.layer_lo > o.layer_hi ||
        o.n_records < 0 || o.n_records > pass_max_records ||
        (o.n_records > 0 && (o.records[0] == nullptr || o.records[1] == nullptr))) {
        return fail(err, "pass: rows or options out of range");
    }
    if ((o.stage.in && (o.layer_lo == 0 || o.stage.recv[0] == nullptr || o.stage.recv[1] == nullptr)) ||
        (o.stage.out && (o.stage.send[0] == nullptr || o.stage.send[1] == nullptr))) {
        return fail(err, "pass: a stage without its boundary buffers, or a received boundary at layer 0");
    }
    return true;
}

bool ple_runs(const pass_options & o) {
    return o.layer_lo == 0 && o.layer_hi >= 1;
}

} // namespace

bool pack_dense_all(model & m, std::string & err) {
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (m.win.layer(l) && !pack_dense(m, l, err)) {  // TTTS1: the box's layers
            return false;
        }
    }
    if (!m.win.head) {
        return true;  // TTTS1: output.weight is the back's
    }
    const int32_t out = model::find("output.weight");
    return out >= 0 ? pack_dense_tensor(m, out, err) : fail(err, "output.weight: not in the placement");
}

bool pass_init(model & m, int32_t card, void * stream, const fn::round::desc * host_ring, const uint64_t * flag,
               fn::round::done * done, uint64_t next_round, pass_card & p, std::string & err) {
    p = pass_card();
    if (!resolve_card(m, card, stream, p.c, err)) {
        return false;
    }
    // TTTS1: the box's window (fn/model/state.h): its layers, the PLE layer with layer 1, the head with the back
    const window & win = m.win;
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (win.layer(l) && (!resolve_weights(m, l, card, p.w[l], err) || !resolve_state(m, l, card, p.s[l], err))) {
            return false;
        }
    }
    p.token_embd = tensor_ptr(m, "token_embd.weight", card, err);
    if (p.token_embd == nullptr) {
        return false;
    }
    if (win.ple()) {
        p.ple_w.key        = tensor_ptr(m, "blk.1.ple_key.weight", card, err);
        p.ple_w.value      = tensor_ptr(m, "blk.1.ple_value.weight", card, err);
        p.ple_w.norm_key   = reinterpret_cast<const float *>(tensor_ptr(m, "blk.1.ple_norm_key.weight", card, err));
        p.ple_w.norm_query = reinterpret_cast<const float *>(tensor_ptr(m, "blk.1.ple_norm_query.weight", card, err));
        p.ple_w.norm_conv  = reinterpret_cast<const float *>(tensor_ptr(m, "blk.1.ple_norm_conv.weight", card, err));
        p.ple_w.conv = reinterpret_cast<const uint16_t *>(tensor_ptr(m, "blk.1.ple_conv1d.weight", card, err));  // F16
        if (p.ple_w.key == nullptr || p.ple_w.value == nullptr || p.ple_w.norm_key == nullptr ||
            p.ple_w.norm_query == nullptr || p.ple_w.norm_conv == nullptr || p.ple_w.conv == nullptr) {
            return false;
        }
    }
    if (win.head) {
        p.head.w_down = reinterpret_cast<const uint16_t *>(tensor_ptr(m, "output_hc_down.weight", card, err));  // BF16
        p.head.w_up   = reinterpret_cast<const uint16_t *>(tensor_ptr(m, "output_hc_up.weight", card, err));
        p.head.gamma  = reinterpret_cast<const float *>(tensor_ptr(m, "output_hc_norm.weight", card, err));
        p.out_w       = tensor_ptr(m, "output.weight", card, err);
        if (p.head.w_down == nullptr || p.head.w_up == nullptr || p.head.gamma == nullptr || p.out_w == nullptr) {
            return false;
        }
        const int32_t oi = model::find("output.weight");
        if (oi < 0 || oi >= (int32_t) m.dense_packed.size() || m.dense_packed[size_t(oi)] == 0) {
            return fail(err, "output.weight: not in DP2's packed layout (pack_dense_all)");
        }
    }
    // the reservation's buffers
    const reserved_class * rc = m.state_class("round and pass (TG1)");
    const reserved_class * conv = m.state_class("convolution histories");
    const auto part = [&](const reserved_class * c, const char * name) -> uint8_t * {
        const int32_t i = part_of(c, name);
        uint8_t * q = i < 0 ? nullptr : c->part_ptr(i, card);
        if (q == nullptr) {
            err = std::string("reservation part ") + name + ": missing";
        }
        return q;
    };
    uint8_t * round = part(rc, "round descriptor and state");
    uint8_t * commit = part(rc, "commit error list");
    uint8_t * ple_e = part(rc, "PLE upload slot (B <= 5)");
    uint8_t * ple_ws = part(rc, "PLE workspace");
    uint8_t * emb = part(rc, "embedding boundary (B <= 5)");
    uint8_t * carry = part(rc, "hidden carry (FC-HEAD3, B <= 5)");
    uint8_t * head = part(rc, "head activations (B <= 5)");
    uint8_t * logits = part(rc, "logits (B <= 5)");
    uint8_t * gw = part(rc, "greedy work");
    uint8_t * hist   = win.ple() ? part(conv, "L1 PLE") : nullptr;  // TTTS1: the PLE history with layer 1
    if (round == nullptr || commit == nullptr || ple_e == nullptr || ple_ws == nullptr || emb == nullptr ||
        carry == nullptr || head == nullptr || logits == nullptr || gw == nullptr || (win.ple() && hist == nullptr)) {
        return false;
    }
    p.desc = reinterpret_cast<fn::round::desc *>(round);
    p.rstate = reinterpret_cast<fn::round::state *>(round + align256(sizeof(fn::round::desc)));
    p.errs = reinterpret_cast<const uint32_t **>(commit);
    p.ple_e = reinterpret_cast<float *>(ple_e);
    p.ple_ws = ple_ws;
    p.ple_hist = reinterpret_cast<float *>(hist);
    p.emb_hres = reinterpret_cast<float *>(emb);
    p.emb_hpost = reinterpret_cast<float *>(emb + align256(int64_t(round_rows) * 4 * F * 4));
    p.emb_hx = reinterpret_cast<float *>(emb + align256(int64_t(round_rows) * 4 * F * 4) +
                                         align256(int64_t(round_rows) * 4 * 4));
    p.carry = reinterpret_cast<float *>(carry);
    p.head_mom = reinterpret_cast<float *>(head);
    // the reservation's two buffers after the moments ([5][1280] and [5][2560] F32) held the head's mixed row until
    // HC3, which writes it into X-HMIX; they stay reserved, unused, until the reservation is revised
    p.pair_own = reinterpret_cast<uint32_t *>(head + align256(int64_t(round_rows) * fn::hc::n_mom * 4) +
                                              align256(int64_t(round_rows) * F * 4) +
                                              align256(int64_t(round_rows) * 2 * F * 4));
    p.logits = reinterpret_cast<float *>(logits);
    p.greedy_work = gw;
    // SM1: the sampler's buffers and slots
    const reserved_class * sc = m.state_class("sampler (SM1)");
    uint8_t * sp = part(sc, "sampler params and record");
    uint8_t * sb = part(sc, "logit bias table");
    uint8_t * scn = part(sc, "penalty counts");
    uint8_t * sr = part(sc, "penalty ring");
    uint8_t * sql = part(sc, "retained q lists");
    uint8_t * sqs = part(sc, "retained q shards");
    uint8_t * sw = part(sc, "sampler workspace");
    if (sp == nullptr || sb == nullptr || scn == nullptr || sr == nullptr || sql == nullptr || sqs == nullptr ||
        sw == nullptr) {
        return false;
    }
    fn::samp::card_view & sv = p.samp;
    sv.card = card;
    sv.prm = reinterpret_cast<fn::samp::params *>(sp);
    sv.rec = reinterpret_cast<fn::samp::record *>(sp + align256(sizeof(fn::samp::params)));
    sv.bias = reinterpret_cast<float *>(sb);
    sv.counts = reinterpret_cast<uint32_t *>(scn);
    sv.ring = reinterpret_cast<int32_t *>(sr);
    sv.q_list = reinterpret_cast<fn::samp::qlist *>(sql);
    sv.q_shard = reinterpret_cast<float *>(sqs);
    sv.work = sw;
    sv.err = p.c.rings->err(card);
    for (int32_t q = 0; q < 2; ++q) {
        sv.a_own[q] = p.c.rings->slot(card, fn::xchg::msg::samp_a, q);
        sv.a_peer[q] = p.c.rings->slot(1 - card, fn::xchg::msg::samp_a, q);
        sv.b_own[q] = p.c.rings->slot(card, fn::xchg::msg::samp_b, q);
        sv.b_peer[q] = p.c.rings->slot(1 - card, fn::xchg::msg::samp_b, q);
        sv.c_own[q] = p.c.rings->slot(card, fn::xchg::msg::samp_c, q);
        sv.c_peer[q] = p.c.rings->slot(1 - card, fn::xchg::msg::samp_c, q);
        sv.d_own[q] = p.c.rings->slot(card, fn::xchg::msg::samp_d, q);
        sv.d_peer[q] = p.c.rings->slot(1 - card, fn::xchg::msg::samp_d, q);
    }
    sv.x_own = p.c.rings->slot(card, fn::xchg::msg::samp_x, 0);
    sv.x_peer = p.c.rings->slot(1 - card, fn::xchg::msg::samp_x, 0);
    p.host_ring = host_ring;
    p.flag = flag;
    p.done = done;
    // the layers read N_before and the rope planes from the descriptor (FC-RD1)
    p.c.n_before = &p.desc->n_before;
    p.c.pos = p.desc->pos;
    // SR1: every versioned object (each GDN layer's state and convolution history, the PLE history; FC-ST1) reads and
    // writes its versions through the round state's one committed-slot word, which the commit node moves
    p.c.cur = &p.rstate->cur;
    p.c.gsel = &p.rstate->gdn; // ME11: the GDN state's selector beside it, moved by the same commit
    // the error words: the card's exchange word, attend's and select's, EX1b's status
    const fn::qsa::attn::workspace aw = fn::qsa::attn::carve(p.c.attn_ws);
    const fn::qsa::workspace qw = fn::qsa::carve(p.c.qsa_ws);
    const uint32_t * errs[] = {p.c.rings->err(card), aw.error, qw.error, p.c.ex.status};
    p.n_err = 4;
    p.err_words[0] = p.c.rings->err(card);
    p.err_words[1] = aw.error;
    p.err_words[2] = qw.error;
    p.err_words[3] = p.c.ex.status;
    fn::round::state st = {};
    st.next_round = next_round;
    st.cur = 0;
    st.gdn = 0;
    return cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", err) &&
           cuda_ok(cudaMemcpy(p.errs, errs, sizeof(errs), cudaMemcpyHostToDevice), "error list", err) &&
           cuda_ok(cudaMemcpy(p.rstate, &st, sizeof(st), cudaMemcpyHostToDevice), "round state", err) &&
           cuda_ok(cudaMemset(p.desc, 0, sizeof(fn::round::desc)), "descriptor", err) &&
           cuda_ok(cudaMemset(p.c.ws.zero, 0, size_t(layer_max_rows) * F * 4), "zero", err) &&
           cuda_ok(cudaMemset(p.greedy_work, 0, fn::head::greedy_work_bytes()), "greedy work", err) &&
           (fn::ple::prepare(p.ple_ws) ? true : fail(err, "ple prepare")) &&
           cuda_ok(cudaMemset(sql, 0, size_t(fn::samp::max_drafts) * sizeof(fn::samp::qlist)), "q lists", err) &&
           cuda_ok(cudaMemset(sqs, 0, size_t(fn::samp::max_drafts) * fn::samp::n_card * 4), "q shards", err) &&
           (fn::samp::prepare(sv, stream) ? true : fail(err, "sampler prepare")) &&
           (fn::samp::load_window(sv, 0, {}, stream) ? true : fail(err, "sampler record")) &&
           cuda_ok(cudaDeviceSynchronize(), "pass init", err);
}

use_counters pass_graph_counters(const pass_options & o) {
    use_counters u;
    u.graph = true;
    const int32_t n = o.layer_hi - o.layer_lo + 1;
    int32_t qsa = 0;
    for (int32_t l = o.layer_lo; l <= o.layer_hi; ++l) {
        qsa += l % fn::full_attn_interval == fn::full_attn_interval - 1 ? 1 : 0;
    }
    u.graph_subs = 2 * n + (o.stage.out ? 0 : 1);  // the layers' sublayers and the head (a front stage has none)
    if (o.stage.in || o.stage.out) {
        u.rule = hc_rule::stage;  // TTTS1: a stage graph's sublayer count is any; the rule keeps slot 1 last
    }
    u.graph_a = n;
    u.graph_f = n;
    u.graph_select = qsa > 0 ? qsa : 1;
    return u;
}

step_status pass_card_enqueue(pass_card & p, int32_t rows, const pass_options & o, use_counters & u,
                              std::string & err) {
    if (!check_options(o, rows, err) || !round_begin(p, err) || !round_input(p, rows, o, err)) {
        return step_status::failed;
    }
    for (int32_t l = o.layer_lo; l <= o.layer_hi; ++l) {
        if (l == 1 && ple_runs(o)) {
            for (int32_t part = 0; part < 3; ++part) {
                if (!ple_part(p, rows, part, err)) {
                    return step_status::failed;
                }
            }
        }
        records_before(p, l, o);
        const step_status ss = layer_card_step(p.c, p.w[l], p.s[l], layer_in(p, l, o), rows, u, err);
        p.c.record_mixed[0] = p.c.record_mixed[1] = nullptr;
        if (ss != step_status::ok) {
            err = "layer " + std::to_string(l) + ": " + err;
            return ss;
        }
        if (!records_after(p, l, rows, o, err)) {
            return step_status::failed;
        }
    }
    if (o.stage.out) {
        return stage_out(p, rows, o, err) && stage_finish(p, err) ? step_status::ok : step_status::failed;
    }
    for (int32_t part = 0; part < n_head_parts(o); ++part) {
        if (!head_part(p, rows, part, u, o, true, err)) {
            return step_status::failed;
        }
    }
    u.sub++;
    return step_status::ok;
}

step_status pass_pair_enqueue(pass_card * const p[2], int32_t rows, const pass_options & o, use_counters u[2],
                              std::string & err) {
    if (!check_options(o, rows, err)) {
        return step_status::failed;
    }
    for (int32_t k = 0; k < 2; ++k) {
        if (!round_begin(*p[k], err) || !round_input(*p[k], rows, o, err)) {
            return step_status::failed;
        }
    }
    card_ctx * cp[2] = {&p[0]->c, &p[1]->c};
    for (int32_t l = o.layer_lo; l <= o.layer_hi; ++l) {
        if (l == 1 && ple_runs(o)) {
            for (int32_t part = 0; part < 3; ++part) {
                for (int32_t k = 0; k < 2; ++k) {
                    if (!ple_part(*p[k], rows, part, err)) {
                        return step_status::failed;
                    }
                }
            }
        }
        const layer_weights * wp[2] = {&p[0]->w[l], &p[1]->w[l]};
        const layer_state * sp[2] = {&p[0]->s[l], &p[1]->s[l]};
        const layer_input in[2] = {layer_in(*p[0], l, o), layer_in(*p[1], l, o)};
        records_before(*p[0], l, o);
        records_before(*p[1], l, o);
        const step_status ss = layer_pair_step(cp, wp, sp, in, rows, u, err);
        for (int32_t k = 0; k < 2; ++k) {
            p[k]->c.record_mixed[0] = p[k]->c.record_mixed[1] = nullptr;
        }
        if (ss != step_status::ok) {
            err = "layer " + std::to_string(l) + ": " + err;
            return ss;
        }
        for (int32_t k = 0; k < 2; ++k) {
            if (!records_after(*p[k], l, rows, o, err)) {
                return step_status::failed;
            }
        }
    }
    if (o.stage.out) {
        for (int32_t k = 0; k < 2; ++k) {
            if (!stage_out(*p[k], rows, o, err) || !stage_finish(*p[k], err)) {
                return step_status::failed;
            }
        }
        return step_status::ok;
    }
    // ME12: on one device and stream the head HC's four phase launches are one (hc::head_pair) at part 0; the scratch
    // is card 0's ws.mom[1], which the target's BF16 layers leave unused (layer.cpp hc_pair_phase)
    const bool hc_pair =
        p[0]->c.device == p[1]->c.device && p[0]->c.stream != nullptr && p[0]->c.stream == p[1]->c.stream;
    for (int32_t part = 0; part < n_head_parts(o); ++part) {
        if (hc_pair && part < 2) {
            if (part == 0 && !head_pair_part(p, rows, u, err)) {
                return step_status::failed;
            }
            continue;
        }
        for (int32_t k = 0; k < 2; ++k) {
            if (!head_part(*p[k], rows, part, u[k], o, false, err)) {
                return step_status::failed;
            }
        }
    }
    u[0].sub++;
    u[1].sub++;
    return step_status::ok;
}

bool pass_bootstrap_enqueue(pass_card * const p[2], int32_t row, const pass_options & o, std::string & err) {
    if (row < 0 || row >= layer_max_rows) {
        return fail(err, "bootstrap: row out of range");
    }
    for (int32_t k = 0; k < 2; ++k) {
        if (!round_begin(*p[k], err)) {
            return false;
        }
    }
    // the decision's parts on the retained row, each on both cards before the next (sends before reads)
    for (const decision_part & d : decision_parts(o)) {
        for (int32_t k = 0; k < 2; ++k) {
            pass_card & q = *p[k];
            if (!set_device(q, err) ||
                !decision_part_enqueue(q, 1, q.logits + int64_t(row) * fn::head::n_vocab_card, d, o)) {
                return err.empty() ? fail(err, "bootstrap decision") : false;
            }
        }
    }
    for (int32_t k = 0; k < 2; ++k) {
        pass_card & q = *p[k];
        // the commit of k = 0 (FC-ST2) changes nothing; it carries the test hook (desc::test_fail) like every round's
        const fn::round::commit_args ca = {q.desc, q.rstate, q.errs, q.n_err, q.c.card};
        const fn::round::finish_args fa = {q.desc, q.rstate, q.errs, q.n_err, q.done};
        if (!set_device(q, err) || !fn::round::commit(ca, q.c.stream) || !fn::round::finish(fa, q.c.stream)) {
            return err.empty() ? fail(err, "bootstrap commit") : false;
        }
    }
    return true;
}

// ---- FMU2: the joint verify round ----

bool pass_lane_init(const pass_card & slot, const fn::xchg::joint_rings & jr, int32_t u, const layer_ws & joint_ws,
                    void * route, void * kernel_ws, const uint32_t ** errs, pass_card & out, std::string & err) {
    if (u < 0 || u >= fn::xchg::max_lanes || route == nullptr || kernel_ws == nullptr || errs == nullptr) {
        return fail(err, "pass_lane_init: lane, routing, kernel workspaces or error list missing");
    }
    out = slot;
    const int32_t card = slot.c.card;
    const fn::xchg::rings & rg = jr.lane(u);
    out.c.rings = &rg;
    out.c.ws = lane_ws(joint_ws, u);
    int32_t * ids = nullptr;
    float * wts = nullptr;
    joint_route_carve(route, ids, wts);
    out.c.ex.ids = ids + int64_t(u) * layer_max_rows * fn::n_expert_used;
    out.c.ex.wts = wts + int64_t(u) * layer_max_rows * fn::n_expert_used;
    joint_lane_ex(route, u, out.c.ex); // FMU2 (build item 1): the lane's GLU slots and meta (the status word shared)
    joint_lane_kernel_ws(kernel_ws, u, out.c); // FMS2 (build item 3): the lane's QSA, attend and GDN workspaces
    out.c.record_mixed[0] = out.c.record_mixed[1] = nullptr;
    out.c.observe = nullptr;
    fn::samp::card_view & sv = out.samp;
    sv.err = rg.err(card);
    for (int32_t q = 0; q < 2; ++q) {
        sv.a_own[q] = rg.slot(card, fn::xchg::msg::samp_a, q);
        sv.a_peer[q] = rg.slot(1 - card, fn::xchg::msg::samp_a, q);
        sv.b_own[q] = rg.slot(card, fn::xchg::msg::samp_b, q);
        sv.b_peer[q] = rg.slot(1 - card, fn::xchg::msg::samp_b, q);
        sv.c_own[q] = rg.slot(card, fn::xchg::msg::samp_c, q);
        sv.c_peer[q] = rg.slot(1 - card, fn::xchg::msg::samp_c, q);
        sv.d_own[q] = rg.slot(card, fn::xchg::msg::samp_d, q);
        sv.d_peer[q] = rg.slot(1 - card, fn::xchg::msg::samp_d, q);
    }
    sv.x_own = rg.slot(card, fn::xchg::msg::samp_x, 0);
    sv.x_peer = rg.slot(1 - card, fn::xchg::msg::samp_x, 0);
    const fn::qsa::attn::workspace aw = fn::qsa::attn::carve(out.c.attn_ws);
    const fn::qsa::workspace qw = fn::qsa::carve(out.c.qsa_ws);
    uint32_t * const words[] = {rg.err(card), aw.error, qw.error, out.c.ex.status, jr.err(card)};
    const uint32_t * list[fn::round::max_err] = {};
    out.n_err = 5;
    for (int32_t i = 0; i < fn::round::max_err; ++i) {
        out.err_words[i] = i < out.n_err ? words[i] : nullptr;
        list[i] = out.err_words[i];
    }
    out.errs = errs;
    return cuda_ok(cudaSetDevice(out.c.device), "cudaSetDevice", err) &&
           cuda_ok(cudaMemcpy(errs, list, sizeof(list), cudaMemcpyHostToDevice), "lane error list", err);
}

namespace {

// the joint pass's checks (lanes 2..4, each lane's options the whole model without records), j's lanes at the passes'
bool joint_begin(joint_pass & jp, std::string & err) {
    if (jp.lanes < 2 || jp.lanes > fn::xchg::max_lanes || jp.j.rows < 2 || jp.j.rows > layer_max_rows) {
        return fail(err, "joint pass: lanes outside 2 .. 4 or rows (R = d + 1) outside 2 .. 5");
    }
    jp.j.lanes = jp.lanes;
    const pass_options & o0    = *jp.o[0];
    // TTTS1: the whole model (or split-test's one-box window from layer 0), or a split stage's window; every lane the
    // same layers and stage
    const bool           whole = o0.layer_lo == 0 && !o0.stage.in && !o0.stage.out;
    const bool           stage = o0.stage.in || o0.stage.out;
    for (int32_t k = 0; k < jp.lanes; ++k) {
        const pass_options & o = *jp.o[k];
        if (!check_options(o, jp.j.rows, err) || o.layer_lo != o0.layer_lo || o.layer_hi != o0.layer_hi ||
            o.stage.in != o0.stage.in || o.stage.out != o0.stage.out || o.n_records != 0 || !(whole || stage)) {
            return fail(err, "joint pass: lane " + std::to_string(k) +
                                 ": the whole model or one split stage for every lane, without records only");
        }
        jp.j.lane[k] = &jp.lane[k]->c;
    }
    return true;
}

// PLE's part 0 of every lane in one joint launch on lane 0's stream (FMU2 build item 4: each weight loaded once; each
// lane's key and value bitwise its own part 0), in the joint frame's PLE region
bool joint_ple_products(joint_pass & jp, std::string & err) {
    if (jp.j.dense_work == nullptr) {
        return fail(err, "ple joint products: no joint work");
    }
    if (!set_device(*jp.lane[0], err)) {
        return false;
    }
    fn::ple::step_args a[fn::xchg::max_lanes];
    const fn::ple::step_args * ap[fn::xchg::max_lanes] = {};
    for (int32_t k = 0; k < jp.lanes; ++k) {
        a[k] = ple_args(*jp.lane[k]);
        ap[k] = &a[k];
    }
    uint8_t * work = static_cast<uint8_t *>(jp.j.dense_work) + joint_ple_work_offset();
    return fn::ple::products_joint(jp.lanes, jp.j.rows, ap, work, jp.lane[0]->c.stream) ?
               true :
               fail(err, "ple joint products");
}

// layer l's states and inputs of every lane
void joint_layer_in(joint_pass & jp, int32_t l, const layer_state * st[], layer_input in[]) {
    for (int32_t k = 0; k < jp.lanes; ++k) {
        st[k] = &jp.lane[k]->s[l];
        in[k] = layer_in(*jp.lane[k], l, *jp.o[k]);
    }
}

// the head HC of every lane in one launch (hc::head_sub_joint; mode as sub's)
bool joint_head_hc(joint_pass & jp, int32_t mode, const use_counters & u, std::string & err) {
    fn::hc::sub_args ha[fn::xchg::max_lanes];
    for (int32_t k = 0; k < jp.lanes; ++k) {
        ha[k] = head_sub_args(*jp.lane[k], u);
    }
    if (!set_device(*jp.lane[0], err)) {
        return false;
    }
    return fn::hc::head_sub_joint(jp.lanes, jp.j.rows, mode, ha, jp.lane[0]->c.stream) ? true :
                                                                                         fail(err, "head hc sub_joint");
}

// every lane's logits in one launch into its own buffer (head::logits_joint)
bool joint_logits(joint_pass & jp, const use_counters & u, std::string & err) {
    fn::head::logits_args la[fn::xchg::max_lanes];
    const int32_t s = hc_slot_of(u, u.sub);
    for (int32_t k = 0; k < jp.lanes; ++k) {
        pass_card & p = *jp.lane[k];
        la[k] = {p.c.rings->slot(p.c.card, fn::xchg::msg::hmix, s),
                 p.c.rings->err(p.c.card),
                 p.c.card,
                 p.out_w,
                 fn::head::n_vocab_card,
                 p.logits,
                 nullptr};
    }
    pass_card & p0 = *jp.lane[0];
    if (!set_device(p0, err)) {
        return false;
    }
    if (!fn::head::logits_joint(jp.lanes, jp.j.rows, la, static_cast<int32_t>(fn::xchg::joint_rings::lane_floats()),
                                jp.j.jr->err(p0.c.card), jp.j.dense_f16, p0.c.stream)) {
        return fail(err, "head logits_joint");
    }
    return true;
}

// lane k's head part (3: its SM1 stamp, then its decision, commit and finish onward)
bool joint_head_part(joint_pass & jp, int32_t k, int32_t part, const use_counters & u, std::string & err) {
    // SM1: the decision's start in context, as head_part's part 2 stamps it
    if (part == 3 && jp.o[k]->sampling != nullptr &&
        (!set_device(*jp.lane[k], err) || !fn::round::stamp_logits(jp.lane[k]->rstate, jp.lane[k]->c.stream))) {
        return fail(err, "lane " + std::to_string(k) + ": stamp_logits");
    }
    if (part >= n_head_parts(*jp.o[k])) {
        return true;
    }
    if (!head_part(*jp.lane[k], jp.j.rows, part, u, *jp.o[k], true, err)) {
        err = "lane " + std::to_string(k) + ": " + err;
        return false;
    }
    return true;
}

int32_t joint_head_parts(const joint_pass & jp) {
    int32_t n = 0;
    for (int32_t k = 0; k < jp.lanes; ++k) {
        n = std::max(n, n_head_parts(*jp.o[k]));
    }
    return n;
}

} // namespace

step_status pass_joint_enqueue(joint_pass & jp, use_counters & u, std::string & err) {
    const int32_t rows = jp.j.rows;  // FMS13: R = d + 1
    if (!joint_begin(jp, err)) {
        return step_status::failed;
    }
    for (int32_t k = 0; k < jp.lanes; ++k) {
        if (!round_begin(*jp.lane[k], err)) {
            return step_status::failed;
        }
    }
    for (int32_t k = 0; k < jp.lanes; ++k) {
        if (!round_input(*jp.lane[k], rows, *jp.o[k], err)) {
            return step_status::failed;
        }
    }
    const pass_options & o0 = *jp.o[0];
    for (int32_t l = o0.layer_lo; l <= o0.layer_hi; ++l) {
        if (l == 1 && ple_runs(o0)) {
            // FMU2: PLE's products in one joint launch (build item 4), then its parts 1 and 2 part-outer, lane-inner
            // (build item 3: each lane's kernels and buffers unchanged)
            if (!joint_ple_products(jp, err)) {
                return step_status::failed;
            }
            for (int32_t part = 1; part < 3; ++part) {
                for (int32_t k = 0; k < jp.lanes; ++k) {
                    if (!ple_part(*jp.lane[k], rows, part, err)) {
                        return step_status::failed;
                    }
                }
            }
        }
        const layer_state * st[fn::xchg::max_lanes] = {};
        layer_input in[fn::xchg::max_lanes];
        joint_layer_in(jp, l, st, in);
        const step_status ss = layer_joint_step(jp.j, jp.lane[0]->w[l], st, in, u, err);
        if (ss != step_status::ok) {
            err = "layer " + std::to_string(l) + ": " + err;
            return ss;
        }
    }
    if (o0.stage.out) {
        for (int32_t k = 0; k < jp.lanes; ++k) {
            if (!stage_out(*jp.lane[k], rows, *jp.o[k], err) || !stage_finish(*jp.lane[k], err)) {
                return step_status::failed;
            }
        }
        return step_status::ok;
    }
    // the head HC of every lane in one launch, every lane's logits in one launch, then the lanes' decision, commit and
    // finish parts, part-outer and lane-inner (FMU2 build item 3: a lane's cross-card part waits once for its peer,
    // while the other lanes' parts of the same stage follow it; each lane's kernels and buffers are unchanged)
    if (!joint_head_hc(jp, 3, u, err) || !joint_logits(jp, u, err)) {
        return step_status::failed;
    }
    const int32_t parts = std::max(joint_head_parts(jp), 4);
    for (int32_t part = 3; part < parts; ++part) {
        for (int32_t k = 0; k < jp.lanes; ++k) {
            if (!joint_head_part(jp, k, part, u, err)) {
                return step_status::failed;
            }
        }
    }
    u.sub++;
    return step_status::ok;
}

step_status pass_joint_pair_enqueue(joint_pass * const jp[2], use_counters u[2], std::string & err) {
    const int32_t rows = jp[0]->j.rows;  // FMS13: R = d + 1
    for (int32_t c = 0; c < 2; ++c) {
        if (!joint_begin(*jp[c], err)) {
            return step_status::failed;
        }
    }
    const int32_t lanes = jp[0]->lanes;
    if (jp[1]->lanes != lanes || jp[1]->j.rows != rows) {
        err = "joint pair pass: the same lanes and rows on both cards";
        return step_status::failed;
    }
    // pass_pair_enqueue's order: each card's begins and embeddings, then every phase on card 0 before card 1's
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t k = 0; k < lanes; ++k) {
            if (!round_begin(*jp[c]->lane[k], err) || !round_input(*jp[c]->lane[k], rows, *jp[c]->o[k], err)) {
                return step_status::failed;
            }
        }
    }
    const pass_options & o0 = *jp[0]->o[0];
    for (int32_t l = o0.layer_lo; l <= o0.layer_hi; ++l) {
        if (l == 1 && ple_runs(o0)) {
            for (int32_t c = 0; c < 2; ++c) {
                if (!joint_ple_products(*jp[c], err)) {
                    return step_status::failed;
                }
            }
            for (int32_t part = 1; part < 3; ++part) {
                for (int32_t c = 0; c < 2; ++c) {
                    for (int32_t k = 0; k < lanes; ++k) {
                        if (!ple_part(*jp[c]->lane[k], rows, part, err)) {
                            return step_status::failed;
                        }
                    }
                }
            }
        }
        const layer_state * st[2][fn::xchg::max_lanes] = {};
        layer_input in[2][fn::xchg::max_lanes];
        joint_layer_in(*jp[0], l, st[0], in[0]);
        joint_layer_in(*jp[1], l, st[1], in[1]);
        joint_lanes * jj[2] = {&jp[0]->j, &jp[1]->j};
        const layer_weights * w[2] = {&jp[0]->lane[0]->w[l], &jp[1]->lane[0]->w[l]};
        const layer_state * const * sp[2] = {st[0], st[1]};
        const layer_input * ip[2] = {in[0], in[1]};
        const step_status ss = layer_joint_pair_step(jj, w, sp, ip, u, err);
        if (ss != step_status::ok) {
            err = "layer " + std::to_string(l) + ": " + err;
            return ss;
        }
    }
    if (o0.stage.out) {
        for (int32_t c = 0; c < 2; ++c) {
            for (int32_t k = 0; k < lanes; ++k) {
                if (!stage_out(*jp[c]->lane[k], rows, *jp[c]->o[k], err) || !stage_finish(*jp[c]->lane[k], err)) {
                    return step_status::failed;
                }
            }
        }
        return step_status::ok;
    }
    for (int32_t mode = 1; mode <= 2; ++mode) {
        for (int32_t c = 0; c < 2; ++c) {
            if (!joint_head_hc(*jp[c], mode, u[c], err)) {
                return step_status::failed;
            }
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!joint_logits(*jp[c], u[c], err)) {
            return step_status::failed;
        }
    }
    const int32_t parts = std::max({joint_head_parts(*jp[0]), joint_head_parts(*jp[1]), 4});
    for (int32_t part = 3; part < parts; ++part) {
        for (int32_t c = 0; c < 2; ++c) {
            for (int32_t k = 0; k < lanes; ++k) {
                if (!joint_head_part(*jp[c], k, part, u[c], err)) {
                    return step_status::failed;
                }
            }
        }
    }
    u[0].sub++;
    u[1].sub++;
    return step_status::ok;
}

} // namespace fn::model
