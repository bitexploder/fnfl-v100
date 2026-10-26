#include "fn/loader/load_check.h"

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/hc/hc.h"
#include "fn/loader/loader.h"
#include "fn/loader/run_load.h"
#include "fn/loader/sha256.h"
#include "fn/model/model.h"
#include "fn/model/state.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace fn;

constexpr int32_t n_output = 2048; // the output tokens of R1's capacity case
constexpr int32_t n_spare = 64;
constexpr int32_t graph_copies = 100; // instances of each capture, to measure above the allocator's 2 MiB pages

// The round's kernel nodes per card [E], counted from the design's node lists (runtime-design.md R3 with the borrowed
// families): a GDN layer 20 (HC 6, GDN products, convolution, recurrence, norm and output 7, router, shared expert and
// routed experts 5, activation quantization 2), a QSA layer 25 (HC 6, QSA 12, experts 5, quantization 2), a draft step
// 30 at up to 3 steps, embedding, PLE and the head 8.
constexpr int32_t round_nodes_estimate = 36 * 20 + 12 * 25 + 3 * 30 + 8;
constexpr int32_t round_shapes = 4; // one graph per B in 1..4

double gib(int64_t b) {
    return (double)b / (1024.0 * 1024.0 * 1024.0);
}

bool cuda_check(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "load-check: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

struct mem_point {
    int64_t free[2] = {0, 0};
    int64_t total[2] = {0, 0};
};

bool mem_now(const model::model & m, mem_point & p) {
    for (int32_t c = 0; c < 2; ++c) {
        size_t f = 0, t = 0;
        if (!cuda_check(cudaSetDevice(m.device[c]), "cudaSetDevice") ||
            !cuda_check(cudaMemGetInfo(&f, &t), "cudaMemGetInfo")) {
            return false;
        }
        p.free[c] = (int64_t)f;
        p.total[c] = (int64_t)t;
    }
    return true;
}

struct account_row {
    std::string name;
    std::string label; // M measured, A arithmetic
    std::string source;
    int64_t requested[2] = {0, 0};
    int64_t allocation[2] = {0, 0};
    int64_t free_before[2] = {0, 0};
    int64_t free_after[2] = {0, 0};
    int64_t wp2[2] = {-1, -1}; // memory-account.json's term at this capacity (its formula), -1: none
};

int64_t measured(const account_row & r, int32_t c) {
    return r.free_before[c] - r.free_after[c];
}

// WP2 memory-account.json terms, linear in capacity at its 262,144 case (B = 3: base + 3 GDN slots)
void wp2_terms(account_row & r, int64_t cells) {
    const auto lin = [&](int64_t at262k) { return at262k / 262144 * cells; };
    if (r.name == "target KV (K0)") {
        r.wp2[0] = r.wp2[1] = lin(3221225472);
    } else if (r.name == "draft KV (F16)") {
        r.wp2[0] = r.wp2[1] = lin(268435456);
    } else if (r.name == "target index keys and summaries") {
        r.wp2[0] = r.wp2[1] = lin(402653184) + lin(201326592);
    } else if (r.name == "draft index (SP2)") {
        r.wp2[0] = lin(67108864) + lin(33554432); // WP2 provisioned card 0 only; SP2 adds card 1 (and its weights)
        r.wp2[1] = 0;
    } else if (r.name == "GDN state") {
        r.wp2[0] = r.wp2[1] = 226492416;
    } else if (r.name == "convolution histories") {
        r.wp2[0] = r.wp2[1] = 8847360;
    } else if (r.name == "weights") {
        r.wp2[0] = 24906516736;
        r.wp2[1] = 24897985792;
    }
}

bool reserve(model::model & m, const model::buffer_class & plan, account_row & row,
             std::unique_ptr<model::reserved_class> & out) {
    row.name = plan.name;
    row.source = plan.source;
    row.label = "A (requested); M (cudaMemGetInfo)";
    mem_point a, b;
    if (!mem_now(m, a)) {
        return false;
    }
    out = std::make_unique<model::reserved_class>();
    out->plan = plan;
    for (int32_t c = 0; c < 2; ++c) {
        row.requested[c] = plan.requested(c);
        row.allocation[c] = plan.allocation(c);
        std::string err;
        if (!out->buf[c].allocate(m.device[c], (size_t)row.allocation[c], err)) {
            fprintf(stderr, "load-check: %s: %s\n", plan.name.c_str(), err.c_str());
            return false;
        }
        // initialize all state (STANDARDS): the QSA flags and every valid length start at zero
        if (row.allocation[c] == 0) {
            continue;
        }
        if (!cuda_check(cudaSetDevice(m.device[c]), "cudaSetDevice") ||
            !cuda_check(cudaMemset(out->buf[c].data(), 0, (size_t)row.allocation[c]), "cudaMemset") ||
            !cuda_check(cudaDeviceSynchronize(), "cudaMemset")) {
            return false;
        }
    }
    if (!mem_now(m, b)) {
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        row.free_before[c] = a.free[c];
        row.free_after[c] = b.free[c];
    }
    return true;
}

// the HC workspace's buffers in hc::workspace_bytes order
struct hc_ws {
    float *hres, *x, *hx, *hpost, *mom, *inject, *mixed, *full;
    uint32_t * runs; // HC3: up2's run counters, in the mixed buffer (the capture runs no kernel)
};

hc_ws carve_hc(uint8_t * p) {
    const size_t b = hc::max_rows;
    const size_t floats[] = {b * 4 * hc::n_own, b * 4 * hc::n_own, b * hc::n_own, b * 4,
                             b * hc::n_mom,     b * hc::n_inject,  b * hc::n_own, b * 2 * hc::n_own};
    float * out[8];
    for (int32_t i = 0; i < 8; ++i) {
        out[i] = reinterpret_cast<float *>(p); // reason: carving typed views out of one byte allocation
        p += (floats[i] * sizeof(float) + 255) / 256 * 256;
    }
    // reason: the run counters are words of the [B][1280] mixed buffer, which up2 no longer writes (HC3)
    return {out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], reinterpret_cast<uint32_t *>(out[6])};
}

struct ex_ws {
    act::out_view x, glu; // FNFL's activation format (C2): x 4 rows of 2560, glu 44 slots of 320
    int32_t * ids;
    float *wts, *out;
    uint32_t * status;
    uint8_t * meta; // the routing meta of ids (XB2)
};

ex_ws carve_ex(uint8_t * p) {
    size_t off[experts::n_ws + 1];
    experts::workspace_offsets(off);
    // reason: carving typed views out of one byte allocation
    return {{reinterpret_cast<int8_t *>(p + off[0]), reinterpret_cast<float *>(p + off[1]), experts::max_rows, 2560},
            {reinterpret_cast<int8_t *>(p + off[2]), reinterpret_cast<float *>(p + off[3]), experts::glu_slots,
             experts::n_ff_card},
            reinterpret_cast<int32_t *>(p + off[4]),
            reinterpret_cast<float *>(p + off[5]),
            reinterpret_cast<float *>(p + off[6]),
            reinterpret_cast<uint32_t *>(p + off[7]),
            p + off[8]};
}

bool ex1_fmt_of(const char * name, experts::fmt & f) {
    for (int32_t i = 0; i <= (int32_t)experts::fmt::iq4_nl; ++i) {
        if (std::strcmp(experts::fmt_name((experts::fmt)i), name) == 0) {
            f = (experts::fmt)i;
            return true;
        }
    }
    return false;
}

// the shared expert's gate and up formats (EX2 takes them as fn::dense formats)
bool dense_fmt_of(const char * name, dense::fmt & f) {
    for (int32_t i = 0; i < (int32_t)dense::fmt::bf16; ++i) {
        if (std::strcmp(dense::fmt_name((dense::fmt)i), name) == 0) {
            f = (dense::fmt)i;
            return true;
        }
    }
    return false;
}

struct graph_row {
    std::string name;
    int32_t card = 0;
    int64_t nodes = 0;
    int64_t free_before = 0, free_after = 0, free_released = 0;
    double per_exec = 0, per_node = 0;
};

// one capture on card c: HC2's sublayer (down2, up2, ready) or EX1b's layer (gate_up, down), layer 0, B = 3; then
// graph_copies instances, each uploaded; no kernel runs
bool capture(model::model & m, int32_t c, bool hc_graph, graph_row & row) {
    row.name = hc_graph ? "HC2 sublayer (down2, up2, ready)" : "EX1b layer (gate_up, down)";
    row.card = c;
    if (!cuda_check(cudaSetDevice(m.device[c]), "cudaSetDevice")) {
        return false;
    }
    const model::reserved_class * ws = m.state_class("kernel workspaces and constants");
    if (ws == nullptr || m.rings == nullptr) {
        return false;
    }
    cudaStream_t st = nullptr;
    if (!cuda_check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) {
        return false;
    }
    bool ok = cuda_check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
    if (hc_graph) {
        const hc_ws h = carve_hc(ws->part_ptr(0, c));
        // reason: the regions hold BF16 bits and F32 gamma, as fn::hc reads them
        const auto region = [&](const char * name) { return m.region_ptr(model::model::find(name), c); };
        const auto * w_down = reinterpret_cast<const uint16_t *>(region("blk.0.hc_attn_down.weight"));
        const auto * w_up = reinterpret_cast<const uint16_t *>(region("blk.0.hc_attn_up.weight"));
        const auto * gamma = reinterpret_cast<const float *>(region("blk.0.hc_attn_norm.weight"));
        hc::down2_args d2{};
        d2.d.w_down = w_down;
        d2.d.hres = h.hres;
        d2.d.hx = h.hx;
        d2.d.hpost = h.hpost;
        d2.d.gamma = gamma;
        d2.d.x = h.x;
        d2.d.mom = h.mom;
        d2.hc_peer = m.rings->slot(1 - c, xchg::msg::hc, 0);
        d2.hc_reset = nullptr;
        hc::up2_args u2{};
        u2.w_up = w_up;
        u2.mom_own = h.mom;
        u2.hc_ring = m.rings->slot(c, xchg::msg::hc, 0);
        u2.x = h.x;
        u2.gamma = gamma;
        u2.inject = h.inject;
        u2.hmix_own = m.rings->slot(c, xchg::msg::hmix, 0);
        u2.hmix_peer = m.rings->slot(1 - c, xchg::msg::hmix, 0);
        u2.runs = h.runs;
        u2.err = m.rings->err(c);
        u2.card = c;
        hc::ready_args ra{};
        ra.slot = m.rings->slot(c, xchg::msg::hmix, 0);
        ra.full = h.full;
        ra.err = m.rings->err(c);
        ra.card = c;
        ra.reset = 1;
        ok = ok && hc::down2(3, d2, st) && hc::up2(hc::up_arm::ring, 3, u2, st) && hc::ready(3, ra, st);
    } else {
        const ex_ws e = carve_ex(ws->part_ptr(1, c));
        const int32_t t_gate = model::model::find("blk.0.ffn_gate_exps.weight");
        const int32_t t_down = model::model::find("blk.0.ffn_down_exps.weight");
        const int32_t t_sg = model::model::find("blk.0.ffn_gate_shexp.weight");
        const int32_t t_su = model::model::find("blk.0.ffn_up_shexp.weight");
        const int32_t t_sd = model::model::find("blk.0.ffn_down_shexp.weight");
        experts::fmt fg{}, fd{}, fsd{};
        dense::fmt sgf{}, suf{};
        ok = ok && ex1_fmt_of(model::tensors[t_gate].format, fg) && ex1_fmt_of(model::tensors[t_down].format, fd) &&
             ex1_fmt_of(model::tensors[t_sd].format, fsd) && dense_fmt_of(model::tensors[t_sg].format, sgf) &&
             dense_fmt_of(model::tensors[t_su].format, suf);
        uint8_t * grids = ws->part_ptr(4, c);
        experts::gate_up_args ga{};
        ga.w = m.region_ptr(t_gate, c);
        ga.sh_gate = m.region_ptr(t_sg, c); // a capture only: the bytes' layout does not matter here
        ga.sh_up = m.region_ptr(t_su, c);
        ga.grids = grids;
        ga.x = {e.x.codes, e.x.scales, 3, e.x.k};
        ga.ids = e.ids;
        ga.meta = e.meta;
        ga.glu = e.glu;
        ga.glu_f32 = nullptr;
        ga.status = e.status;
        ga.cw = experts::class_weights(fg, sgf, suf); // XB3: gate_up writes the meta for down
        experts::down_args da{};
        da.w = m.region_ptr(t_down, c);
        da.sh_w = m.region_ptr(t_sd, c);
        da.grids = grids;
        da.glu = {e.glu.codes, e.glu.scales, e.glu.rows, e.glu.k};
        da.ids = e.ids;
        da.meta = e.meta;
        da.wts = e.wts;
        da.sg = e.wts; // a capture only: any [B] floats
        da.out = e.out;
        da.status = e.status;
        ok = ok && experts::gate_up(fg, sgf, suf, 3, ga, st) && experts::down(fd, fsd, 3, da, st);
    }
    cudaGraph_t g = nullptr;
    ok = cuda_check(cudaStreamEndCapture(st, &g), "end capture") && ok;
    size_t n_nodes = 0;
    ok = ok && cuda_check(cudaGraphGetNodes(g, nullptr, &n_nodes), "graph nodes");
    row.nodes = (int64_t)n_nodes;
    std::vector<cudaGraphExec_t> ex;
    mem_point a, b, r;
    ok = ok && mem_now(m, a) && cuda_check(cudaSetDevice(m.device[c]), "cudaSetDevice");
    for (int32_t k = 0; ok && k < graph_copies; ++k) {
        cudaGraphExec_t x = nullptr;
        ok = cuda_check(cudaGraphInstantiate(&x, g, 0), "instantiate") && cuda_check(cudaGraphUpload(x, st), "upload");
        if (x != nullptr) {
            ex.push_back(x);
        }
    }
    ok = ok && cuda_check(cudaStreamSynchronize(st), "upload sync") && mem_now(m, b);
    cudaSetDevice(m.device[c]);
    for (cudaGraphExec_t x : ex) {
        cudaGraphExecDestroy(x);
    }
    if (g != nullptr) {
        cudaGraphDestroy(g);
    }
    cudaStreamSynchronize(st);
    cudaStreamDestroy(st);
    ok = ok && mem_now(m, r);
    if (ok) {
        row.free_before = a.free[c];
        row.free_after = b.free[c];
        row.free_released = r.free[c];
        row.per_exec = (double)(a.free[c] - b.free[c]) / graph_copies;
        row.per_node = row.nodes > 0 ? row.per_exec / (double)row.nodes : 0;
    }
    return ok;
}

bool component_sums(std::string & err) {
    std::vector<int64_t> sums(int64_t{2} * model::n_components, 0);
    for (int32_t t = 0; t < model::n_tensors; ++t) {
        for (int32_t c = 0; c < 2; ++c) {
            sums[2 * model::tensors[t].component + c] += model::tensors[t].card[c].bytes;
        }
    }
    for (int32_t i = 0; i < model::n_components; ++i) {
        for (int32_t c = 0; c < 2; ++c) {
            if (sums[2 * i + c] != model::components[i].card_bytes[c]) {
                err = std::string("component ") + model::components[i].key + " card " + std::to_string(c) +
                      ": tensors sum to " + std::to_string(sums[2 * i + c]) + ", S2_card_unique_bytes " +
                      std::to_string(model::components[i].card_bytes[c]);
                return false;
            }
        }
    }
    return true;
}

bool open_and_hash(const fnfl_load_check_args & a, loader::load_params & lp, model::model & m, loader::load_times & lt,
                   std::vector<loader::card_digests> & want, bool allow_cache) {
    lp.model_path = a.model_path;
    lp.draft_path = a.draft_path;
    lp.threads_per_card = a.threads_per_card;
    if (!loader::sha256_self_test()) {
        return false;
    }
    printf("sha256: self-test PASS (%s)\n", loader::sha256_hardware() ? "SHA extensions" : "portable");
    std::string err;
    if (!component_sums(err) || !loader::open_model(lp, m, lt, err)) {
        fprintf(stderr, "load: %s\n", err.c_str());
        return false;
    }
    printf("placement: %d tensors, %d components, %d regions; every component's tensors sum to its "
           "S2_card_unique_bytes; card bytes %lld / %lld (%.6f / %.6f GiB) = memory-account weight_card_bytes\n",
           model::n_tensors, model::n_components, model::n_regions, (long long)m.weight_bytes[0],
           (long long)m.weight_bytes[1], gib(m.weight_bytes[0]), gib(m.weight_bytes[1]));
    printf("open: %zu files mapped, directory checked against the placement in %.3f s; PLE table %llu bytes mapped on "
           "the host; ple.conv_kernel %d, ple.ngram_size %d\n",
           m.files.size(), lt.open_s, (unsigned long long)m.ple_bytes, m.ple_conv_kernel, m.ple_ngram);
    const std::string cache = a.out_dir + "/gguf-slices.sha256";
    const std::string fp = loader::source_fingerprint(m);
    if (allow_cache && loader::read_hash_cache(cache, fp, want)) {
        printf("gguf hashes: read from %s (fingerprint %s)\n", cache.c_str(), fp.c_str());
        return true;
    }
    double s = 0;
    loader::hash_gguf_slices(m, 12, want, s);
    printf("gguf hashes: %d tensors' card slices hashed from the mapping in %.2f s (wall, 12 threads)\n",
           model::n_tensors, s);
    if (!loader::write_hash_cache(cache, fp, want)) {
        fprintf(stderr, "load: cannot write %s\n", cache.c_str());
        return false;
    }
    printf("gguf hashes: written to %s (fingerprint %s)\n", cache.c_str(), fp.c_str());
    return true;
}

void print_verify(const char * what, const loader::verify_report & r) {
    printf("%s: card 0 %d/%d tensors equal, card 1 %d/%d; %lld + %lld bytes; %d EX1 pack permutations (bijective); "
           "%.2f s\n",
           what, r.equal[0], r.placed[0], r.equal[1], r.placed[1], (long long)r.bytes_read[0],
           (long long)r.bytes_read[1], r.ex1_permutations, r.seconds);
    for (size_t i = 0; i < r.failures.size() && i < 20; ++i) {
        printf("  MISMATCH %s\n", r.failures[i].c_str());
    }
}

int32_t expected_placed(int32_t c) {
    int32_t n = 0;
    for (int32_t t = 0; t < model::n_tensors; ++t) {
        n += model::tensors[t].lay != model::layout::host && model::tensors[t].card[c].bytes > 0;
    }
    return n;
}

bool verify_ok(const loader::verify_report & r) {
    return r.failures.empty() && r.equal[0] == r.placed[0] && r.equal[1] == r.placed[1] &&
           r.placed[0] == expected_placed(0) && r.placed[1] == expected_placed(1);
}

} // namespace

int fnfl_load_plan(const fnfl_load_check_args & a) {
    loader::load_params lp;
    model::model m;
    loader::load_times lt;
    std::vector<loader::card_digests> want;
    if (!open_and_hash(a, lp, m, lt, want, false)) {
        return 1;
    }
    loader::verify_report rep;
    std::string err;
    if (!loader::verify_host(lp, m, want, rep, err)) {
        fprintf(stderr, "load-plan: %s\n", err.c_str());
        return 1;
    }
    print_verify("dry run (host fill, no GPU)", rep);
    const bool ok = verify_ok(rep);
    printf("load-plan: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int fnfl_load_check(const fnfl_load_check_args & a) {
    using clk = std::chrono::steady_clock;
    const auto t_open = clk::now();
    loader::load_params lp;
    auto mp = std::make_unique<model::model>();
    model::model & m = *mp;
    loader::load_times lt;
    std::vector<loader::card_digests> want;
    if (!open_and_hash(a, lp, m, lt, want, true)) {
        return 1;
    }
    int n_dev = 0;
    if (!cuda_check(cudaGetDeviceCount(&n_dev), "device count") || n_dev < 2) {
        fprintf(stderr, "load-check: needs two devices, found %d\n", n_dev);
        return 1;
    }
    std::vector<account_row> rows;
    mem_point ctx;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_check(cudaSetDevice(m.device[c]), "cudaSetDevice") || !cuda_check(cudaFree(nullptr), "context")) {
            return 1;
        }
    }
    if (!mem_now(m, ctx)) {
        return 1;
    }
    account_row ctx_row;
    ctx_row.name = "CUDA context";
    ctx_row.label = "M (total - free after the context exists)";
    ctx_row.source = "cudaMemGetInfo";
    for (int32_t c = 0; c < 2; ++c) {
        ctx_row.free_before[c] = ctx.total[c];
        ctx_row.free_after[c] = ctx.free[c];
    }
    rows.push_back(ctx_row);

    // weights
    std::string err;
    account_row wr;
    wr.name = "weights";
    wr.label = "A (placement bytes); M (cudaMemGetInfo)";
    wr.source = "fn/model/placement.h (S2): node-costs.json S2_card_unique_bytes";
    mem_point a0, a1;
    const auto t_alloc = clk::now();
    if (!mem_now(m, a0) || !loader::allocate_weights(m, err) || !mem_now(m, a1)) {
        fprintf(stderr, "load-check: weights: %s\n", err.c_str());
        return 1;
    }
    const double alloc_s = std::chrono::duration<double>(clk::now() - t_alloc).count();
    for (int32_t c = 0; c < 2; ++c) {
        wr.requested[c] = m.weight_bytes[c];
        wr.allocation[c] = m.weight_alloc[c];
        wr.free_before[c] = a0.free[c];
        wr.free_after[c] = a1.free[c];
    }
    wp2_terms(wr, a.n_cells);
    rows.push_back(wr);
    if (!loader::fill_weights(lp, m, lt, err)) {
        fprintf(stderr, "load-check: fill: %s\n", err.c_str());
        return 1;
    }
    const double ready_s = std::chrono::duration<double>(clk::now() - t_open).count();
    printf("load: open %.3f s, allocate %.3f s, fill and copy %.3f s (wall); open to ready %.3f s (includes the GGUF "
           "hash cache read)\n",
           lt.open_s, alloc_s, lt.fill_wall_s, ready_s);
    printf("load: host read %.2f thread-s, repack %.2f thread-s, buffer wait %.2f thread-s (%d threads a card); H2D "
           "card 0 %.3f s for %lld bytes (%.2f GB/s), card 1 %.3f s for %lld bytes (%.2f GB/s), %d + %d pieces\n",
           lt.read_busy_s, lt.repack_busy_s, lt.wait_busy_s, lp.threads_per_card, lt.h2d_s[0],
           (long long)lt.h2d_bytes[0], double(lt.h2d_bytes[0]) / lt.h2d_s[0] / 1e9, lt.h2d_s[1],
           (long long)lt.h2d_bytes[1], double(lt.h2d_bytes[1]) / lt.h2d_s[1] / 1e9, lt.pieces[0], lt.pieces[1]);

    loader::verify_report rep;
    if (!loader::verify_device(lp, m, want, rep, err)) {
        fprintf(stderr, "load-check: verify: %s\n", err.c_str());
        return 1;
    }
    print_verify("placement check (read back from the cards)", rep);
    const bool placement_ok = verify_ok(rep);

    // state and workspaces
    model::state_params sp;
    sp.n_cells = a.n_cells;
    sp.ple_conv_kernel = m.ple_conv_kernel;
    sp.ple_ngram = m.ple_ngram;
    for (const model::buffer_class & plan : model::state_classes(sp)) {
        account_row r;
        std::unique_ptr<model::reserved_class> rc;
        if (!reserve(m, plan, r, rc)) {
            return 1;
        }
        wp2_terms(r, a.n_cells);
        rows.push_back(r);
        m.state.push_back(std::move(rc));
    }
    if (std::string gerr; !loader::fill_ex1_grids(m, gerr)) {
        fprintf(stderr, "load-check: the EX1 grid table: %s\n", gerr.c_str());
        return 1;
    }
    {
        account_row r;
        r.name = "exchange rings (fn/xchg)";
        r.label = "A (rings_card_bytes); M (cudaMemGetInfo)";
        r.source = "xchg.h: X-HC and X-HMIX, 3 slots of 4 rows each, and the error word";
        mem_point b0, b1;
        if (!mem_now(m, b0)) {
            return 1;
        }
        m.rings = std::make_unique<xchg::rings>(m.device[0], m.device[1]);
        if (!m.rings->ok() || !mem_now(m, b1)) {
            fprintf(stderr, "load-check: exchange rings\n");
            return 1;
        }
        for (int32_t c = 0; c < 2; ++c) {
            r.requested[c] = r.allocation[c] = xchg::rings_card_bytes();
            r.free_before[c] = b0.free[c];
            r.free_after[c] = b1.free[c];
        }
        rows.push_back(r);
    }

    // prefill chunk buffers: 1,536 rows measured and released, then 3,072 rows kept
    account_row ch[2];
    int64_t chunk_delta[2][2] = {{0, 0}, {0, 0}};
    for (int32_t k = 0; k < 2; ++k) {
        std::unique_ptr<model::reserved_class> rc;
        if (!reserve(m, model::chunk_class(k == 0 ? 1536 : 3072), ch[k], rc)) {
            return 1;
        }
        for (int32_t c = 0; c < 2; ++c) {
            chunk_delta[k][c] = measured(ch[k], c);
        }
        if (k == 0) {
            rc.reset();
        } else {
            m.state.push_back(std::move(rc));
        }
    }
    rows.push_back(ch[0]);
    rows.back().name += " [measured, then released]";
    rows.push_back(ch[1]);

    // graph instances
    std::vector<graph_row> graphs;
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t k = 0; k < 2; ++k) {
            graph_row g;
            if (!capture(m, c, k == 0, g)) {
                fprintf(stderr, "load-check: graph capture failed\n");
                return 1;
            }
            graphs.push_back(g);
        }
    }
    mem_point end;
    if (!mem_now(m, end)) {
        return 1;
    }

    // the account
    printf("\naccount at capacity %d cells (196,608 + %d output + %d), per card; A arithmetic, M measured "
           "(cudaMemGetInfo free before - after)\n",
           a.n_cells, n_output, n_spare);
    printf("%-52s %15s %15s %15s %15s %13s %13s\n", "class", "requested c0", "requested c1", "measured c0",
           "measured c1", "WP2 c0 GiB", "WP2 c1 GiB");
    int64_t sum_req[2] = {0, 0}, sum_meas[2] = {0, 0};
    for (const account_row & r : rows) {
        const bool released = r.name.find("released") != std::string::npos;
        printf("%-52s %15lld %15lld %15lld %15lld %13s %13s\n", r.name.c_str(), (long long)r.requested[0],
               (long long)r.requested[1], (long long)measured(r, 0), (long long)measured(r, 1),
               r.wp2[0] >= 0 ? std::to_string(gib(r.wp2[0])).substr(0, 8).c_str() : "-",
               r.wp2[1] >= 0 ? std::to_string(gib(r.wp2[1])).substr(0, 8).c_str() : "-");
        if (!released) {
            for (int32_t c = 0; c < 2; ++c) {
                sum_req[c] += r.requested[c];
                sum_meas[c] += measured(r, c);
            }
        }
    }
    double per_node = 0;
    for (const graph_row & g : graphs) {
        printf("graph %-46s card %d: %lld nodes; %d instances took %lld bytes (M): %.0f a graph, %.0f a node; released "
               "to %lld free\n",
               g.name.c_str(), g.card, (long long)g.nodes, graph_copies, (long long)(g.free_before - g.free_after),
               g.per_exec, g.per_node, (long long)g.free_released);
        per_node = std::max(per_node, g.per_node);
    }
    const int64_t graph_allow = (int64_t)(per_node * round_nodes_estimate * round_shapes + 0.5);
    printf("graph allowance: %.0f bytes a node (M, the larger capture) x %d round nodes (E) x %d round graphs = %lld "
           "bytes a card\n",
           per_node, round_nodes_estimate, round_shapes, (long long)graph_allow);
    printf("total reserved (3,072-row chunks): requested %lld / %lld bytes (%.3f / %.3f GiB), measured %lld / %lld "
           "(%.3f / %.3f GiB) with the context\n",
           (long long)sum_req[0], (long long)sum_req[1], gib(sum_req[0]), gib(sum_req[1]), (long long)sum_meas[0],
           (long long)sum_meas[1], gib(sum_meas[0]), gib(sum_meas[1]));

    // the 192K decision
    int64_t head[2][2]; // [chunk 1536 / 3072][card]
    int64_t cap[2] = {INT64_MAX, INT64_MAX};
    for (int32_t c = 0; c < 2; ++c) {
        head[1][c] = end.free[c] - graph_allow;
        head[0][c] = head[1][c] + chunk_delta[1][c] - chunk_delta[0][c];
        for (int32_t k = 0; k < 2; ++k) {
            const int64_t per = model::bytes_per_cell(c);
            const int64_t extra = head[k][c] >= 0 ? head[k][c] / per : -((-head[k][c] + per - 1) / per);
            cap[k] = std::min(cap[k], (int64_t)a.n_cells + extra);
        }
    }
    printf("\nfree at the end (M): %lld / %lld bytes (%.3f / %.3f GiB) of %lld / %lld total\n", (long long)end.free[0],
           (long long)end.free[1], gib(end.free[0]), gib(end.free[1]), (long long)end.total[0],
           (long long)end.total[1]);
    for (int32_t k = 1; k >= 0; --k) {
        printf("chunks of %d rows: headroom after the graph allowance %lld / %lld bytes (%.3f / %.3f GiB): %s at %d "
               "cells; largest capacity by memory %lld cells (%lld + 2,112), by %lld / %lld bytes a cell (A)\n",
               k == 1 ? 3072 : 1536, (long long)head[k][0], (long long)head[k][1], gib(head[k][0]), gib(head[k][1]),
               std::min(head[k][0], head[k][1]) >= 0 ? "FITS" : "DOES NOT FIT", a.n_cells, (long long)cap[k],
               (long long)(cap[k] - n_output - n_spare), (long long)model::bytes_per_cell(0),
               (long long)model::bytes_per_cell(1));
    }

    // SV1: serving's engine (fn::runtime::engine::init) takes the model with its target; load-check keeps it

    // results
    const std::string path = a.out_dir + "/load-check.json";
    FILE * f = fopen(path.c_str(), "w");
    if (f != nullptr) {
        fprintf(f, "{\n \"capacity_cells\": %d,\n \"placement_ok\": %s,\n", a.n_cells, placement_ok ? "true" : "false");
        fprintf(f,
                " \"placement\": {\"tensors\": %d, \"components\": %d, \"regions\": %d, \"card_bytes\": [%lld, %lld], "
                "\"verified\": [[%d, %d], [%d, %d]], \"bytes_read_back\": [%lld, %lld], \"ex1_permutations\": %d, "
                "\"verify_s\": %.3f},\n",
                model::n_tensors, model::n_components, model::n_regions, (long long)m.weight_bytes[0],
                (long long)m.weight_bytes[1], rep.equal[0], rep.placed[0], rep.equal[1], rep.placed[1],
                (long long)rep.bytes_read[0], (long long)rep.bytes_read[1], rep.ex1_permutations, rep.seconds);
        fprintf(f,
                " \"load_s\": {\"open\": %.4f, \"allocate\": %.4f, \"fill_wall\": %.4f, \"open_to_ready\": %.4f, "
                "\"host_read_thread_s\": %.4f, \"repack_thread_s\": %.4f, \"buffer_wait_thread_s\": %.4f, "
                "\"h2d_s\": [%.4f, %.4f], \"h2d_bytes\": [%lld, %lld], \"threads_per_card\": %d, "
                "\"pieces\": [%d, %d]},\n",
                lt.open_s, alloc_s, lt.fill_wall_s, ready_s, lt.read_busy_s, lt.repack_busy_s, lt.wait_busy_s,
                lt.h2d_s[0], lt.h2d_s[1], (long long)lt.h2d_bytes[0], (long long)lt.h2d_bytes[1], lp.threads_per_card,
                lt.pieces[0], lt.pieces[1]);
        fprintf(f, " \"account\": [\n");
        for (size_t i = 0; i < rows.size(); ++i) {
            const account_row & r = rows[i];
            fprintf(f,
                    "  {\"class\": \"%s\", \"label\": \"%s\", \"requested\": [%lld, %lld], "
                    "\"allocation\": [%lld, %lld], \"free_before\": [%lld, %lld], \"free_after\": [%lld, %lld], "
                    "\"measured\": [%lld, %lld], \"wp2_at_capacity\": [%lld, %lld]}%s\n",
                    r.name.c_str(), r.label.c_str(), (long long)r.requested[0], (long long)r.requested[1],
                    (long long)r.allocation[0], (long long)r.allocation[1], (long long)r.free_before[0],
                    (long long)r.free_before[1], (long long)r.free_after[0], (long long)r.free_after[1],
                    (long long)measured(r, 0), (long long)measured(r, 1), (long long)r.wp2[0], (long long)r.wp2[1],
                    i + 1 < rows.size() ? "," : "");
        }
        fprintf(f, " ],\n \"graphs\": [\n");
        for (size_t i = 0; i < graphs.size(); ++i) {
            const graph_row & g = graphs[i];
            fprintf(f,
                    "  {\"graph\": \"%s\", \"card\": %d, \"nodes\": %lld, \"instances\": %d, \"free_before\": %lld, "
                    "\"free_after\": %lld, \"free_released\": %lld, \"bytes_a_graph\": %.1f, "
                    "\"bytes_a_node\": %.1f}%s\n",
                    g.name.c_str(), g.card, (long long)g.nodes, graph_copies, (long long)g.free_before,
                    (long long)g.free_after, (long long)g.free_released, g.per_exec, g.per_node,
                    i + 1 < graphs.size() ? "," : "");
        }
        fprintf(f,
                " ],\n \"graph_allowance\": {\"bytes_a_node\": %.1f, \"round_nodes_estimate\": %d, "
                "\"round_graphs\": %d, \"bytes\": %lld},\n",
                per_node, round_nodes_estimate, round_shapes, (long long)graph_allow);
        fprintf(f, " \"free_end\": [%lld, %lld],\n \"total\": [%lld, %lld],\n", (long long)end.free[0],
                (long long)end.free[1], (long long)end.total[0], (long long)end.total[1]);
        fprintf(f, " \"chunk_measured\": {\"1536\": [%lld, %lld], \"3072\": [%lld, %lld]},\n",
                (long long)chunk_delta[0][0], (long long)chunk_delta[0][1], (long long)chunk_delta[1][0],
                (long long)chunk_delta[1][1]);
        fprintf(f,
                " \"decision\": {\"headroom_3072\": [%lld, %lld], \"headroom_1536\": [%lld, %lld], "
                "\"max_cells_3072\": %lld, \"max_cells_1536\": %lld, \"bytes_per_cell\": [%lld, %lld]}\n}\n",
                (long long)head[1][0], (long long)head[1][1], (long long)head[0][0], (long long)head[0][1],
                (long long)cap[1], (long long)cap[0], (long long)model::bytes_per_cell(0),
                (long long)model::bytes_per_cell(1));
        fclose(f);
        printf("written: %s\n", path.c_str());
    }
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(c);
        cudaDeviceSynchronize();
    }
    const bool ok = placement_ok;
    printf("load-check: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---- TTTS1: the split's per-box plan ----

namespace {

struct stage_plan {
    int64_t weights[2] = { 0, 0 };
    int64_t slots[2] = { 0, 0 }, pool[2] = { 0, 0 }, engine[2] = { 0, 0 }, lanes[2] = { 0, 0 };

    int64_t total(int32_t c) const { return weights[c] + slots[c] + pool[c] + engine[c] + lanes[c]; }
};

bool plan_stage(const std::string & model_path,
                const std::string & draft_path,
                int32_t             k,
                bool                front,
                int32_t             n_slots,
                int32_t             slot_cells,
                int32_t             pool_cells,
                bool                kv_q8,
                stage_plan &        p,
                std::string &       err) {
    loader::load_params lp;
    lp.model_path = model_path;
    lp.draft_path = draft_path;
    lp.n_slots    = n_slots;
    lp.kv_q8      = kv_q8;
    lp.win.lo     = front ? 0 : k;
    lp.win.hi     = front ? k - 1 : n_layer - 1;
    lp.win.head   = !front;
    model::model       m;
    loader::load_times lt;
    if (!loader::open_model(lp, m, lt, err)) {
        return false;
    }
    model::state_params sp;
    sp.n_cells         = slot_cells;
    sp.ple_conv_kernel = m.ple_conv_kernel;
    sp.ple_ngram       = m.ple_ngram;
    sp.kv_q8           = kv_q8;
    sp.win             = m.win;
    p                  = stage_plan();
    for (int32_t c = 0; c < 2; ++c) {
        p.weights[c] = m.weight_alloc[c];
        p.pool[c]    = model::bytes_per_cell(c, kv_q8, m.win) * int64_t(pool_cells);
    }
    for (const model::buffer_class & b : model::state_classes(sp)) {
        if (model::pooled_class(b.name)) {
            continue;  // the pool's, at pool_cells
        }
        for (int32_t c = 0; c < 2; ++c) {
            (model::slot_class(b.name) ? p.slots[c] : p.engine[c]) +=
                b.allocation(c) * (model::slot_class(b.name) ? n_slots : 1);
        }
    }
    for (const model::buffer_class & b : n_slots > 1 ? model::lane_classes() : std::vector<model::buffer_class>()) {
        for (int32_t c = 0; c < 2; ++c) {
            p.lanes[c] += b.allocation(c);
        }
    }
    return true;
}

}  // namespace

int fnfl_split_plan(const fnfl_split_plan_args & a) {
    std::string err;
    printf(
        "split-plan: %s, k %d: %d slots of %d cells, the KV pool %d cells (CPU only: the placement and the state "
        "classes; no GPU)\n",
        a.model_path.c_str(), a.k, a.slots, a.slot_cells, a.pool_cells);
    for (int32_t role = 0; role < 2; ++role) {
        const bool     front = role == 0;
        const char *   box   = front ? "the front (Carlos: layers 0 .. k - 1, the embeddings)" :
                                       "the back (Carlin: layers k .. 47, the head and the draft)";
        const double * freem = front ? a.free_front : a.free_back;
        // the calibration: the measured config's plan (its free figure measured with it)
        stage_plan     cal;
        const bool     have_cal = freem[0] >= 0 && !a.cal_model.empty();
        if (have_cal && !plan_stage(a.cal_model, a.cal_draft, a.cal_k, front, a.cal_slots, a.cal_slot_cells,
                                    a.cal_pool_cells, a.cal_q8, cal, err)) {
            fprintf(stderr, "split-plan: the calibration's plan: %s\n", err.c_str());
            return 1;
        }
        for (int32_t q8 = 0; q8 < 2; ++q8) {
            stage_plan p;
            if (!plan_stage(a.model_path, a.draft_path, a.k, front, a.slots, a.slot_cells, a.pool_cells, q8 != 0, p,
                            err)) {
                fprintf(stderr, "split-plan: %s\n", err.c_str());
                return 1;
            }
            for (int32_t c = 0; c < 2; ++c) {
                printf(
                    "split-plan: %s, card %d, %s KV: weights %.3f GiB; state %.3f GiB (slot classes x %d %.3f, the "
                    "KV pool %.3f, the engine's %.3f, the other-reader lane %.3f); planned %.3f GiB",
                    box, c, q8 ? "q8" : "BF16", gib(p.weights[c]), gib(p.total(c) - p.weights[c]), a.slots,
                    gib(p.slots[c]), gib(p.pool[c]), gib(p.engine[c]), gib(p.lanes[c]), gib(p.total(c)));
                if (have_cal) {
                    const double head = freem[c] - (gib(p.total(c)) - gib(cal.total(c)));
                    printf(
                        "; headroom %.3f GiB (measured free %.3f GiB at the calibration config, planned %.3f GiB "
                        "there)%s",
                        head, freem[c], gib(cal.total(c)),
                        q8 && head < 1.0 ? ": UNDER 1 GiB (recorded for TTTS2)" : "");
                }
                printf("\n");
            }
        }
    }
    return 0;
}
