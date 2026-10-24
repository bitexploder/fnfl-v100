#include "fn/kernels/experts/experts_bench.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_cold.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/dense/layer_stand.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_test.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/shapes.h"
#include "fn/quant/act_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace fn::experts;
using dfmt = fn::dense::fmt;

constexpr int32_t NU = fn::n_expert_used;
constexpr int32_t NE = fn::n_expert;
constexpr int32_t D = fn::n_embd;
constexpr int32_t n_unit = 64;             // launches in a warm graph, each with the next decode's routing
constexpr int32_t n_warm = 3;              // untimed replays (warm) and trials (cold)
constexpr int32_t n_timed = 20;            // timed replays (warm) and trials (cold)
constexpr int32_t n_cold_rep = 20;         // launches in a cold graph (XP1's rep, dp-bench's)
constexpr double warm_s = 2.0;             // HX12 Later 15: a process starts at 1,260 MHz for 0.3-1 s
constexpr int64_t flush_used = 12ll << 20; // > 6 MiB of L2 (XP1)
constexpr int32_t d_first = 3, n_dec = 20; // XP1's --attr set: decodes 3 .. 22 of HX7's capture, replayed in turn
constexpr int32_t n_sh_copies = 8;         // shared-expert copies rotated (8 x 1.2-1.8 MB, past the 6 MiB L2)
constexpr double ramp_us = 1.949;          // HX12 card 0 at 1,380 MHz
constexpr double dram_gbps = 1021.6;       // HX12
constexpr double tail_us = 0.83;           // HX12: the GLU and weighted sums finish across blocks (S26)

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "ex-bench: %s: %s\n", what, cudaGetErrorString(e));
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

// owns one device allocation of n bytes
struct dev_mem {
    uint8_t * p = nullptr;
    size_t n = 0;

    explicit dev_mem(size_t bytes) : n(std::max<size_t>(bytes, 16)) {
        if (!cuda_ok(cudaMalloc(&p, n), "cudaMalloc")) {
            p = nullptr;
        }
    }
    ~dev_mem() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }
    dev_mem(const dev_mem &) = delete;
    dev_mem & operator=(const dev_mem &) = delete;

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); } // allocations are 256-byte aligned
    bool put(const void * h, size_t bytes, size_t at = 0) const {
        return p != nullptr && at + bytes <= n && cuda_ok(cudaMemcpy(p + at, h, bytes, cudaMemcpyHostToDevice), "put");
    }
};

struct graph_exec {
    cudaGraphExec_t x = nullptr;
    graph_exec() = default;
    ~graph_exec() {
        if (x != nullptr) {
            cudaGraphExecDestroy(x);
        }
    }
    graph_exec(const graph_exec &) = delete;
    graph_exec & operator=(const graph_exec &) = delete;
};

struct event_pair {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    event_pair() = default;
    ~event_pair() {
        if (e0 != nullptr) {
            cudaEventDestroy(e0);
        }
        if (e1 != nullptr) {
            cudaEventDestroy(e1);
        }
    }
    event_pair(const event_pair &) = delete;
    event_pair & operator=(const event_pair &) = delete;
    bool make() { return cuda_ok(cudaEventCreate(&e0), "event") && cuda_ok(cudaEventCreate(&e1), "event"); }
};

struct stats {
    double median = 0.0, p90 = 0.0;
    std::vector<double> us;
    std::string t0, t1;
    void finish() {
        std::vector<double> s = us;
        std::sort(s.begin(), s.end());
        median = 0.5 * (s[n_timed / 2 - 1] + s[n_timed / 2]);
        p90 = s[(9 * n_timed + 9) / 10 - 1];
    }
};

// HX7's captured routing of one layer (capture-58581: lines "decode row layer e0 .. e9", rows 3 d .. 3 d + 2 of decode
// d), decodes d_first .. d_first + n_dec (and the next one's row 0 for B = 4): ids[d][row][10]
bool load_routes(const std::string & dir, int32_t layer, std::vector<int32_t> & ids) {
    const std::string cmd = "gzip -dc '" + dir + "/routes.txt.gz'";
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) {
        return false;
    }
    ids.assign(size_t(n_dec + 1) * 3 * NU, -1);
    char line[512];
    int32_t found = 0;
    while (fgets(line, sizeof(line), p) != nullptr) {
        int32_t d = 0, row = 0, l = 0, e[NU] = {};
        if (sscanf(line, "%d %d %d %d %d %d %d %d %d %d %d %d %d", &d, &row, &l, &e[0], &e[1], &e[2], &e[3], &e[4],
                   &e[5], &e[6], &e[7], &e[8], &e[9]) != 13 ||
            l != layer || d < d_first || d > d_first + n_dec) {
            continue;
        }
        const int32_t r = row - 3 * d;
        if (r < 0 || r > 2) {
            continue;
        }
        for (int32_t k = 0; k < NU; ++k) {
            ids[(size_t(d - d_first) * 3 + r) * NU + k] = e[k];
        }
        ++found;
    }
    pclose(p);
    return found == (n_dec + 1) * 3;
}

// the B rows of decode i of the set: rows 0..2 of the decode, then row 0 of the next for B = 4 and rows 0, 1 of the
// next for B = 5 (SP4 Amendment 1)
std::vector<int32_t> rows_of(const std::vector<int32_t> & ids, int32_t i, int32_t B) {
    std::vector<int32_t> out;
    for (int32_t b = 0; b < B; ++b) {
        const int32_t d = b < 3 ? i : i + 1, r = b < 3 ? b : b - 3;
        const std::ptrdiff_t at = (std::ptrdiff_t(d) * 3 + r) * NU;
        out.insert(out.end(), ids.begin() + at, ids.begin() + at + NU);
    }
    return out;
}

// the captured layer whose routing a point replays: HX7 captured the target's layers only, so the MTP draft layer
// (DR1, layer 48) replays layer 47's
int32_t route_layer(int32_t layer) {
    return std::min(layer, fn::n_layer - 1);
}

// a routed format's key in bench::routed: Q8_0 is a gate/up and a down format (DR1)
int32_t rkey(fmt f, bool down) {
    return int32_t(f) * 2 + (down ? 1 : 0);
}

int32_t distinct(const std::vector<int32_t> & ids) {
    std::vector<int32_t> s = ids;
    std::sort(s.begin(), s.end());
    return static_cast<int32_t>(std::unique(s.begin(), s.end()) - s.begin());
}

struct bench {
    cudaStream_t st = nullptr;
    std::string cache, hx7;
    std::vector<int32_t> map; // expert ID -> synthetic expert (every ID routed somewhere real)
    std::map<int32_t, std::unique_ptr<dev_mem>> routed; // rkey(format, down) -> packed experts, card 0
    std::map<int32_t, std::unique_ptr<dev_mem>> sh_gu;  // dense format -> n_sh_copies DP2-packed 320 x 2560
    std::map<int32_t, std::unique_ptr<dev_mem>> sh_dn;  // dense format (q2_0, iq4_nl, q8_0) -> n_sh_copies pack_down
    std::map<int32_t, size_t> sh_gu_bytes, sh_dn_bytes;
    std::map<int32_t, std::vector<int32_t>> routes; // layer -> HX7 ids
    int32_t ex_map = 0;                             // SP4 Amendment 1 (sp4prof): gate_up_args::map of the launches
    dev_mem grids{grid_bytes};
    dev_mem x{size_t(fn::act::codes_bytes(max_rows, D) + fn::act::scales_bytes(max_rows, D))};
    dev_mem ids{size_t(n_dec) * max_rows * NU * 4}; // [decode][B][10] at the current point's B
    dev_mem glu{size_t(n_dec) * (fn::act::codes_bytes(glu_slots, n_ff_card) +
                                 fn::act::scales_bytes(glu_slots, n_ff_card))}; // [decode]: down's inputs
    dev_mem meta{size_t(n_dec) * meta_bytes}; // [decode]: the routing meta of its ids (XB2)
    dev_mem wts{size_t(max_rows) * NU * 4};
    dev_mem sg{size_t(max_rows) * 4};
    dev_mem out{size_t(max_rows) * D * 4};
    dev_mem status{16};
    dev_mem flush{size_t(flush_used)};
    dev_mem evict{16};
    // CR1 (r4 mode): the evict-first flush's buffer, and the pre-touch's own copies of the routed and shared weights,
    // the activations, routing, GLU, router weights, shared gate and output, so a pre-touch shares only the kernel's
    // code, its parameters' layout, the grid tables and the status word (written only on a bad expert ID) with the unit
    std::unique_ptr<dev_mem> flush_ef;
    std::map<int32_t, std::unique_ptr<dev_mem>> routed_p, sh_gu_p, sh_dn_p;
    std::unique_ptr<dev_mem> x_p, ids_p, glu_p, wts_p, sg_p, out_p;
    static constexpr size_t glu_bytes =
        fn::act::codes_bytes(glu_slots, n_ff_card) + fn::act::scales_bytes(glu_slots, n_ff_card);

    bool cold() {
        return fn::dense::cold_flush(flush.p, flush_used, st) && fn::dense::cold_icache(evict.as<float>(), st);
    }

    // pre (CR1): the pre-touch's copies; base (CR2): a [decode][max_rows][10] routing buffer (CR2's per-layer ones),
    // or null for ids (or the pre-touch's ids_p)
    const int32_t * ids_of(int32_t i, int32_t /*B: the block's first B rows*/, bool pre = false,
                           const int32_t * base = nullptr) const {
        return (base != nullptr ? base : (pre ? *ids_p : ids).as<int32_t>()) + size_t(i % n_dec) * max_rows * NU;
    }
    fn::act::view x_in(int32_t B, bool pre = false) const {
        const dev_mem & m = pre ? *x_p : x;
        return {m.as<int8_t>(), reinterpret_cast<const float *>(m.p + fn::act::codes_bytes(max_rows, D)), B, D};
    }
    fn::act::out_view glu_out(int32_t i, bool pre = false) const {
        uint8_t * g = (pre ? *glu_p : glu).p + size_t(i % n_dec) * glu_bytes;
        return {reinterpret_cast<int8_t *>(g),
                reinterpret_cast<float *>(g + fn::act::codes_bytes(glu_slots, n_ff_card)), glu_slots, n_ff_card};
    }

    uint8_t * meta_of(int32_t i) const { return meta.p + size_t(i % n_dec) * meta_bytes; }
    const class_table dn_class_w = class_weights(fmt::iq2_xxs, dfmt::q4_k, dfmt::q4_k); // down's meta: any table

    // XB2: decode i's routing meta by write_meta's builder in a one-block launch, ahead of each down launch (XB3: in
    // the model gate/up writes it, and the bench's gate/up launches build their own); its own time is a point of its
    // own (<point>_meta). cw: the unit order's class weights (down reads no order)
    bool meta_unit(int32_t B, int32_t i, const class_table & cw, const int32_t * ids_base = nullptr, bool pre = false) {
        return write_meta(B, ids_of(i, B, pre, ids_base), cw, meta_of(i), st) || err("write_meta");
    }

    // EX2's launches (the base branch puts EX1b's and DP2's here)
    // ids_base (CR2): the sequence's per-layer routing, or null; pre (CR1): the pre-touch's copies. The meta is built
    // from the same IDs the unit reads (XB3 part 0)
    bool gate_up_unit(fmt f, dfmt fg, dfmt fu, int32_t B, int32_t i, const int32_t * ids_base = nullptr,
                      bool pre = false) {
        const int32_t c = i % n_sh_copies;
        const fn::act::out_view g = glu_out(i, pre);
        const auto & rw = pre ? routed_p : routed;
        const auto & sw = pre ? sh_gu_p : sh_gu;
        gate_up_args a = {rw.at(rkey(f, false))->p,
                          sw.at(int32_t(fg))->p + size_t(c) * sh_gu_bytes.at(int32_t(fg)),
                          sw.at(int32_t(fu))->p + size_t(c) * sh_gu_bytes.at(int32_t(fu)),
                          grids.p,
                          x_in(B, pre),
                          ids_of(i, B, pre, ids_base),
                          meta_of(i),
                          g,
                          nullptr,
                          status.as<uint32_t>()};
        a.cw = class_weights(f, fg, fu); // XB3: gate_up builds the meta and order, and writes the meta for down
        a.map = ex_map;
        return gate_up(f, fg, fu, B, a, st) || err("gate_up");
    }
    bool down_unit(fmt f, fmt fs, int32_t B, int32_t i, const int32_t * ids_base = nullptr, bool pre = false) {
        if (!meta_unit(B, i, dn_class_w, ids_base, pre)) {
            return false;
        }
        const int32_t c = i % n_sh_copies;
        const dfmt fsd = fs == fmt::q2_0 ? dfmt::q2_0 : fs == fmt::q8_0 ? dfmt::q8_0 : dfmt::iq4_nl;
        const fn::act::out_view g = glu_out(i, pre);
        const down_args a = {(pre ? routed_p : routed).at(rkey(f, true))->p,
                             (pre ? sh_dn_p : sh_dn).at(int32_t(fsd))->p + size_t(c) * sh_dn_bytes.at(int32_t(fsd)),
                             grids.p,
                             {g.codes, g.scales, g.rows, g.k},
                             ids_of(i, B, pre, ids_base),
                             meta_of(i),
                             (pre ? *wts_p : wts).as<float>(),
                             (pre ? *sg_p : sg).as<float>(),
                             (pre ? *out_p : out).as<float>(),
                             status.as<uint32_t>()};
        return down(f, fs, B, a, st) || err("down");
    }
    static bool err(const char * what) {
        fprintf(stderr, "ex-bench: %s not enqueued: %s\n", what, cudaGetErrorString(cudaGetLastError()));
        return false;
    }

    // the decodes' routing at B into ids (each decode's B x 10 block at [decode][max_rows][10])
    bool set_routes(int32_t point_layer, int32_t B) {
        const int32_t layer = route_layer(point_layer);
        if (routes.count(layer) == 0 && !load_routes(hx7, layer, routes[layer])) {
            fprintf(stderr, "ex-bench: HX7 routes of layer %d not found in %s\n", layer, hx7.c_str());
            return false;
        }
        std::vector<int32_t> h(size_t(n_dec) * max_rows * NU, 0);
        for (int32_t i = 0; i < n_dec; ++i) {
            const std::vector<int32_t> r = rows_of(routes[layer], i, B);
            std::copy(r.begin(), r.end(), h.begin() + std::ptrdiff_t(i) * max_rows * NU);
        }
        return ids.put(h.data(), h.size() * 4);
    }
    // CR2: layer's routing at B into its own buffer (ids' layout), kept for the sequence
    std::map<int32_t, std::unique_ptr<dev_mem>> layer_ids;
    bool set_layer_routes(int32_t layer, int32_t B) {
        if (routes.count(layer) == 0 && !load_routes(hx7, layer, routes[layer])) {
            fprintf(stderr, "ex-bench: HX7 routes of layer %d not found in %s\n", layer, hx7.c_str());
            return false;
        }
        std::vector<int32_t> h(size_t(n_dec) * max_rows * NU, 0);
        for (int32_t i = 0; i < n_dec; ++i) {
            const std::vector<int32_t> r = rows_of(routes[layer], i, B);
            std::copy(r.begin(), r.end(), h.begin() + std::ptrdiff_t(i) * max_rows * NU);
        }
        layer_ids[layer] = std::make_unique<dev_mem>(h.size() * 4);
        return layer_ids[layer]->put(h.data(), h.size() * 4);
    }
    double mean_u(int32_t point_layer, int32_t B) {
        const int32_t layer = route_layer(point_layer);
        double u = 0.0;
        for (int32_t i = 0; i < n_dec; ++i) {
            u += distinct(rows_of(routes[layer], i, B));
        }
        return u / n_dec;
    }

    bool prepare() {
        if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) {
            return false;
        }
        std::vector<uint8_t> g;
        if (!grid_buffer(g) || !grids.put(g.data(), g.size())) {
            return false;
        }
        map.resize(NE);
        for (int32_t e = 0; e < NE; ++e) {
            map[size_t(e)] = e % n_synth;
        }
        // the routed formats in fmt's order (Q8_0's gate/up shape), then Q8_0's down shape (DR1)
        for (int32_t i = 0; i < 9; ++i) {
            const fmt f = i < 8 ? static_cast<fmt>(i) : fmt::q8_0;
            const bool down = f == fmt::q2_0 || f == fmt::iq4_nl || i == 8;
            synth s;
            std::vector<uint8_t> h;
            if (!load_synth(f, cache, s, false, down) || !pack_card(s, 0, map, 1, h)) {
                fprintf(stderr, "ex-bench: synthetic %s%s failed\n", fmt_name(f), down ? " down" : "");
                return false;
            }
            routed[rkey(f, down)] = std::make_unique<dev_mem>(h.size());
            if (!routed[rkey(f, down)]->put(h.data(), h.size())) {
                return false;
            }
        }
        int32_t salt = 0;
        for (const dfmt f : {dfmt::iq3_s, dfmt::iq4_xs, dfmt::q4_k, dfmt::q5_k, dfmt::q6_k, dfmt::q8_0}) {
            fn::dense::synth s;
            std::vector<uint8_t> h;
            if (!fn::dense::make_synth(f, D, s) ||
                !fn::dense::pack_synth(s, n_ff_card, 37 * ++salt % fn::dense::n_synth_rows, fn::dense::n_synth_rows,
                                       h)) {
                return false;
            }
            sh_gu_bytes[int32_t(f)] = h.size();
            sh_gu[int32_t(f)] = std::make_unique<dev_mem>(h.size() * n_sh_copies);
            for (int32_t c = 0; c < n_sh_copies; ++c) {
                if (!sh_gu[int32_t(f)]->put(h.data(), h.size(), size_t(c) * h.size())) {
                    return false;
                }
            }
        }
        for (const dfmt f : {dfmt::q2_0, dfmt::iq4_nl, dfmt::q8_0}) {
            fn::dense::synth s;
            if (!fn::dense::make_synth(f, n_ff_card, s)) {
                return false;
            }
            std::vector<uint8_t> rows(s.row_bytes * D);
            for (int32_t n = 0; n < D; ++n) {
                std::memcpy(rows.data() + s.row_bytes * n, s.row((n + 101) % fn::dense::n_synth_rows), s.row_bytes);
            }
            const fmt ef = f == dfmt::q2_0 ? fmt::q2_0 : f == dfmt::q8_0 ? fmt::q8_0 : fmt::iq4_nl;
            std::vector<uint8_t> h(size_t(down_bytes(ef)));
            if (!pack_down(ef, rows.data(), s.row_bytes, h.data())) {
                return false;
            }
            sh_dn_bytes[int32_t(f)] = h.size();
            sh_dn[int32_t(f)] = std::make_unique<dev_mem>(h.size() * n_sh_copies);
            for (int32_t c = 0; c < n_sh_copies; ++c) {
                if (!sh_dn[int32_t(f)]->put(h.data(), h.size(), size_t(c) * h.size())) {
                    return false;
                }
            }
        }
        // the rows' activations (random, in FNFL's format), router weights 0.1, shared gate 0.5
        std::mt19937 rng(0x45583242u);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xf(size_t(max_rows) * D);
        for (auto & v : xf) {
            v = nd(rng);
        }
        std::vector<uint8_t> xq(x.n, 0);
        fn::quant::act_quantize_ref(xf.data(), max_rows, D, reinterpret_cast<int8_t *>(xq.data()),
                                    reinterpret_cast<float *>(xq.data() + fn::act::codes_bytes(max_rows, D)));
        std::vector<float> w(size_t(max_rows) * NU, 0.1f), s(max_rows, 0.5f);
        return x.put(xq.data(), xq.size()) && wts.put(w.data(), w.size() * 4) && sg.put(s.data(), s.size() * 4) &&
               cuda_ok(cudaMemset(glu.p, 0, glu.n), "glu");
    }

    // CR1: the pre-touch's copies (device to device) and the evict-first flush's buffer
    static bool copy_of(const dev_mem & src, std::unique_ptr<dev_mem> & dst) {
        dst = std::make_unique<dev_mem>(src.n);
        return dst->p != nullptr && cuda_ok(cudaMemcpy(dst->p, src.p, src.n, cudaMemcpyDeviceToDevice), "pre copy");
    }
    bool prepare_r4() {
        bool ok = true;
        for (const auto * m : {&routed, &sh_gu, &sh_dn}) {
            auto & d = m == &routed ? routed_p : m == &sh_gu ? sh_gu_p : sh_dn_p;
            for (const auto & kv : *m) {
                ok = ok && copy_of(*kv.second, d[kv.first]);
            }
        }
        flush_ef = std::make_unique<dev_mem>(size_t(flush_used));
        return ok && copy_of(x, x_p) && copy_of(ids, ids_p) && copy_of(glu, glu_p) && copy_of(wts, wts_p) &&
               copy_of(sg, sg_p) && copy_of(out, out_p) && flush_ef->p != nullptr &&
               cuda_ok(cudaMemset(flush_ef->p, 0x69, flush_ef->n), "flush_ef") &&
               cuda_ok(cudaDeviceSynchronize(), "pre copies"); // st does not wait on the legacy stream's copies
    }
    // CR1: refresh the pre-touch's routing and GLU from the unit's (after set_routes and the down inputs)
    bool sync_pre() const {
        return cuda_ok(cudaMemcpy(ids_p->p, ids.p, ids.n, cudaMemcpyDeviceToDevice), "pre ids") &&
               cuda_ok(cudaMemcpy(glu_p->p, glu.p, glu.n, cudaMemcpyDeviceToDevice), "pre glu") &&
               cuda_ok(cudaDeviceSynchronize(), "pre sync");
    }
    bool cold_r4(int32_t v, const std::function<bool()> & pre) {
        return fn::dense::cold_r4(v, {flush.p, flush_ef->p, flush_used, evict.as<float>()}, st, pre);
    }

    // raise the clock: 2 s of device work before the first timed point (dp-bench's rule)
    bool warm_up() {
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s) {
            for (int32_t k = 0; k < 64; ++k) {
                if (!cuda_ok(cudaMemsetAsync(flush.p, k, flush.n, st), "memset")) {
                    return false;
                }
            }
            if (!cuda_ok(cudaStreamSynchronize(st), "sync")) {
                return false;
            }
        }
        return true;
    }
};

bool capture(bench & bn, int32_t n, const std::function<bool(int32_t)> & unit, graph_exec & ge) {
    cudaGraph_t graph = nullptr;
    if (!cuda_ok(cudaStreamBeginCapture(bn.st, cudaStreamCaptureModeThreadLocal), "begin capture")) {
        return false;
    }
    bool ok = true;
    for (int32_t i = 0; i < n && ok; ++i) {
        ok = unit(i);
    }
    ok = cuda_ok(cudaStreamEndCapture(bn.st, &graph), "end capture") && ok;
    ok = ok && cuda_ok(cudaGraphInstantiate(&ge.x, graph, 0), "instantiate");
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    return ok;
}

bool replay_us(bench & bn, const graph_exec & ge, event_pair & ev, double & us) {
    float ms = 0.0f;
    const bool ok = cuda_ok(cudaEventRecord(ev.e0, bn.st), "record") &&
                    cuda_ok(cudaGraphLaunch(ge.x, bn.st), "replay") &&
                    cuda_ok(cudaEventRecord(ev.e1, bn.st), "record") && cuda_ok(cudaEventSynchronize(ev.e1), "sync") &&
                    cuda_ok(cudaEventElapsedTime(&ms, ev.e0, ev.e1), "elapsed");
    us = 1e3 * double(ms);
    return ok;
}

// Warm (dp-bench's): n_unit launches back to back in one graph, each with the next decode's routing and shared copy
// (the routed weights stream from DRAM, the code stays in the SMs' caches); 3 untimed and 20 timed replays
bool time_warm(bench & bn, const std::function<bool(int32_t)> & unit, stats & s) {
    graph_exec ge;
    event_pair ev;
    bool ok = capture(bn, n_unit, unit, ge) && ev.make();
    double us = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = replay_us(bn, ge, ev, us);
    }
    s.us.clear();
    s.t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        ok = replay_us(bn, ge, ev, us);
        s.us.push_back(us / n_unit);
    }
    s.t1 = utc_now();
    if (ok) {
        s.finish();
    }
    return ok;
}

// Cold (R17, XP1's alone_timer as dp-bench): A = 20 x [flush, eviction, launch], B = 20 x [flush, eviction]; a trial
// is (A - B) / 20: a launch with its weights, activations and code cold, plus one node boundary
bool time_cold(bench & bn, const std::function<bool(int32_t)> & unit, stats & s) {
    graph_exec ga, gb;
    event_pair ev;
    bool ok = capture(
                  bn, n_cold_rep, [&](int32_t c) { return bn.cold() && unit(c); }, ga) &&
              capture(bn, n_cold_rep, [&](int32_t) { return bn.cold(); }, gb) && ev.make();
    double ta = 0.0, tb = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = replay_us(bn, ga, ev, ta) && replay_us(bn, gb, ev, tb);
    }
    s.us.clear();
    s.t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        ok = replay_us(bn, gb, ev, tb) && replay_us(bn, ga, ev, ta);
        s.us.push_back((ta - tb) / n_cold_rep);
    }
    s.t1 = utc_now();
    if (ok) {
        s.finish();
    }
    return ok;
}

// CR1 (R4 on the bench): time_cold's (A - B) / n_cold_rep for each of the four cold states of dense_cold.h's cold_r4,
// graph A = n_cold_rep x [state v, unit c], B = n_cold_rep x [state v]; pre(c) is unit c's pre-touch. A trial replays
// B then A of V0, V1, V2 and V3 in turn.
bool time_r4(bench & bn, const std::function<bool(int32_t)> & unit, const std::function<bool(int32_t)> & pre,
             stats s[4]) {
    graph_exec ga[4], gb[4];
    event_pair ev;
    bool ok = ev.make();
    for (int32_t v = 0; v < 4 && ok; ++v) {
        ok =
            capture(
                bn, n_cold_rep, [&](int32_t c) { return bn.cold_r4(v, [&]() { return pre(c); }) && unit(c); }, ga[v]) &&
            capture(bn, n_cold_rep, [&](int32_t c) { return bn.cold_r4(v, [&]() { return pre(c); }); }, gb[v]);
    }
    double ta = 0.0, tb = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        for (int32_t v = 0; v < 4 && ok; ++v) {
            ok = replay_us(bn, gb[v], ev, tb) && replay_us(bn, ga[v], ev, ta);
        }
    }
    for (int32_t v = 0; v < 4; ++v) {
        s[v].us.clear();
        s[v].t0 = utc_now();
    }
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        for (int32_t v = 0; v < 4 && ok; ++v) {
            ok = replay_us(bn, gb[v], ev, tb) && replay_us(bn, ga[v], ev, ta);
            s[v].us.push_back((ta - tb) / n_cold_rep);
        }
    }
    for (int32_t v = 0; v < 4 && ok; ++v) {
        s[v].t1 = utc_now();
        s[v].finish();
    }
    return ok;
}

double p10_of(const stats & s) {
    std::vector<double> v = s.us;
    std::sort(v.begin(), v.end());
    return v[size_t(n_timed / 10)];
}

void print_us(const char * key, const std::vector<double> & us) {
    printf(", \"%s\": [", key);
    for (size_t i = 0; i < us.size(); ++i) {
        printf("%s%.3f", i ? ", " : "", us[i]);
    }
    printf("]");
}

void print_point(const std::string & point, const std::string & formats, int32_t layer, int32_t B, double u,
                 double bytes, const stats & warm, const stats & cold) {
    const double floor = ramp_us + bytes / (dram_gbps * 1e3) + tail_us;
    const std::string stand_in = route_layer(layer) == layer ? "" : " (stand-in for " + std::to_string(layer) + ")";
    printf("{\"ticket\": \"EX2\", \"point\": \"%s\", \"formats\": \"%s\", \"routing\": \"HX7 capture-58581 layer %d%s, "
           "decodes %d..%d\", \"rows\": %d, \"u_mean\": %.2f, \"weight_bytes\": %.0f, \"floor_us\": %.3f, "
           "\"warm_median_us\": %.3f, \"warm_p90_us\": %.3f, \"hw_pct_warm\": %.1f, \"cold_median_us\": %.3f, "
           "\"cold_p90_us\": %.3f, \"hw_pct_cold\": %.1f, \"warm\": \"graph of %d launches, weights cold (each "
           "decode's experts, %d shared copies), code warm\", \"cold\": \"(A - B) / %d: 12 MB L2 flush and "
           "instruction-cache eviction before each launch\", \"t_start\": \"%s\", \"t_end\": \"%s\"",
           point.c_str(), formats.c_str(), route_layer(layer), stand_in.c_str(), d_first, d_first + n_dec - 1, B, u,
           bytes, floor, warm.median, warm.p90, 100.0 * floor / warm.median, cold.median, cold.p90,
           100.0 * floor / cold.median, n_unit, n_sh_copies, n_cold_rep, warm.t0.c_str(), cold.t1.c_str());
    print_us("warm_us", warm.us);
    print_us("cold_us", cold.us);
    printf("}\n");
    fflush(stdout);
}

// CR1: one r4 point's line: warm, then V0..V3
void print_r4(const std::string & point, const std::string & formats, int32_t layer, int32_t B, double u, double bytes,
              const stats & warm, const stats v[4]) {
    printf("{\"ticket\": \"CR1\", \"point\": \"%s\", \"formats\": \"%s\", \"routing\": \"HX7 capture-58581 layer "
           "%d, decodes %d..%d\", \"rows\": %d, \"u_mean\": %.2f, \"weight_bytes\": %.0f, \"warm_median_us\": %.3f, "
           "\"warm_p10_us\": %.3f, \"warm_p90_us\": %.3f",
           point.c_str(), formats.c_str(), route_layer(layer), d_first, d_first + n_dec - 1, B, u, bytes, warm.median,
           p10_of(warm), warm.p90);
    for (int32_t k = 0; k < 4; ++k) {
        printf(", \"V%d_median_us\": %.3f, \"V%d_p10_us\": %.3f, \"V%d_p90_us\": %.3f", k, v[k].median, k, p10_of(v[k]),
               k, v[k].p90);
    }
    printf(", \"cold\": \"(A - B) / %d; V0: flush, evict | V1: flush, evict, pre-touch, ef flush, evict | V2: flush, "
           "evict, ef flush, evict | V3: flush, evict, pre-touch, flush, evict; flushes 12 MB\", \"warm\": \"graph of "
           "%d launches, weights cold (each decode's experts, %d shared copies), code warm\", \"t_start\": \"%s\", "
           "\"t_end\": \"%s\"",
           n_cold_rep, n_unit, n_sh_copies, warm.t0.c_str(), v[3].t1.c_str());
    print_us("warm_us", warm.us);
    for (int32_t k = 0; k < 4; ++k) {
        const std::string key = "V" + std::to_string(k) + "_us";
        print_us(key.c_str(), v[k].us);
    }
    printf("}\n");
    fflush(stdout);
}

// XB2: the meta launches alone at a down point's routing (<point>_meta), the time its figure includes and the Result
// subtracts (XB3: gate/up points have none; their figure includes their own meta and order)
bool time_meta(bench & bn, const std::string & point, int32_t layer, int32_t B, double u, const class_table & cw) {
    const auto unit = [&](int32_t k) { return bn.meta_unit(B, k, cw); };
    stats warm, cold;
    if (!time_warm(bn, unit, warm) || !time_cold(bn, unit, cold)) {
        fprintf(stderr, "ex-bench: %s_meta failed\n", point.c_str());
        return false;
    }
    print_point(point + "_meta", "write_meta (one block)", layer, B, u, 0.0, warm, cold);
    return true;
}

std::string lower(const char * s) {
    std::string r(s);
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return r;
}

// ---- CR2: the sequence (R3 on the bench) ---------------------------------------------------------------------
// The experts' launches of four consecutive target layers in the round's order: per layer the instruction-cache
// evictor (standing for the other families' code between two expert launches), then gate/up, then down, B = 3 rows of
// each layer's own HX7 routing. Layers 36-39 (experts_inst.h): the routed gate/up format changes at every layer.
struct seq_layer {
    int32_t layer;
    fmt f;
    dfmt fg, fu; // gate/up
    fmt fd, fs;  // down
};
constexpr seq_layer seq_layers[4] = {
    {36, fmt::iq3_xxs, dfmt::iq3_s, dfmt::iq3_s, fmt::q2_0, fmt::iq4_nl},
    {37, fmt::iq3_s, dfmt::q4_k, dfmt::iq3_s, fmt::iq4_nl, fmt::iq4_nl},
    {38, fmt::iq2_s, dfmt::iq3_s, dfmt::iq3_s, fmt::iq4_nl, fmt::iq4_nl},
    {39, fmt::iq2_xs, dfmt::iq4_xs, dfmt::iq4_xs, fmt::iq4_nl, fmt::iq4_nl},
};
constexpr int32_t seq_B = 3;
constexpr int32_t n_seq = 8; // launches a pass: k = 2 L (gate/up of layer L), 2 L + 1 (its down)

// true when the layer's formats are a deployed instantiation (experts_inst.h)
bool seq_layer_deployed(const seq_layer & sl) {
    bool gu_ok = false, dn_ok = false;
    for (int32_t i = 0; i < n_gate_up_inst; ++i) {
        fmt f{};
        dfmt fg{}, fu{};
        int32_t l = 0;
        gate_up_inst(i, f, fg, fu, l);
        gu_ok = gu_ok || (f == sl.f && fg == sl.fg && fu == sl.fu);
    }
    for (int32_t i = 0; i < n_down_inst; ++i) {
        fmt f{}, fs{};
        int32_t l = 0;
        down_inst(i, f, fs, l);
        dn_ok = dn_ok || (f == sl.fd && fs == sl.fs);
    }
    return gu_ok && dn_ok;
}

std::string seq_point(int32_t k) {
    const seq_layer & sl = seq_layers[k / 2];
    if (k % 2 == 0) {
        return "gu_" + lower(fmt_name(sl.f)) + "_" + lower(fn::dense::fmt_name(sl.fg)) + "_" +
               lower(fn::dense::fmt_name(sl.fu)) + "_B" + std::to_string(seq_B);
    }
    return "dn_" + lower(fmt_name(sl.fd)) + "_" + lower(fmt_name(sl.fs)) + "_B" + std::to_string(seq_B);
}

// launch k of the sequence at decode d
bool seq_launch(bench & bn, int32_t k, int32_t d) {
    const seq_layer & sl = seq_layers[k / 2];
    const int32_t * ids = bn.layer_ids.at(sl.layer)->as<int32_t>();
    return k % 2 == 0 ? bn.gate_up_unit(sl.f, sl.fg, sl.fu, seq_B, d, ids) : bn.down_unit(sl.fd, sl.fs, seq_B, d, ids);
}

// one pass at decode d: per layer the evictor, gate/up, down; launch omit (-1: none) is left out, and with
// launches false only the evictors run
bool seq_pass(bench & bn, int32_t d, int32_t omit, bool launches) {
    for (int32_t L = 0; L < 4; ++L) {
        if (!fn::dense::cold_icache(bn.evict.as<float>(), bn.st)) {
            return bench::err("cold_icache");
        }
        for (int32_t k = 2 * L; k < 2 * L + 2 && launches; ++k) {
            if (k != omit && !seq_launch(bn, k, d)) {
                return false;
            }
        }
    }
    return true;
}

struct dist {
    double median = 0.0, p10 = 0.0, p90 = 0.0, lo = 0.0, hi = 0.0;
};
dist dist_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    dist r;
    r.median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    r.p10 = v[(n + 9) / 10 - 1];
    r.p90 = v[(9 * n + 9) / 10 - 1];
    r.lo = v.front();
    r.hi = v.back();
    return r;
}

int32_t run_seq(bench & bn, const std::string & mode) {
    for (const seq_layer & sl : seq_layers) {
        if (!seq_layer_deployed(sl) || !bn.set_layer_routes(sl.layer, seq_B)) {
            fprintf(stderr, "ex-bench: seq layer %d is not a deployed instantiation or has no routing\n", sl.layer);
            return 1;
        }
    }
    // the routed weights a pass streams, and the distinct routed bytes over the n_dec decodes (the rotation)
    double pass_bytes = 0.0, rot_bytes = 0.0;
    for (const seq_layer & sl : seq_layers) {
        std::vector<int32_t> all;
        for (int32_t i = 0; i < n_dec; ++i) {
            const std::vector<int32_t> r = rows_of(bn.routes[sl.layer], i, seq_B);
            all.insert(all.end(), r.begin(), r.end());
        }
        const double per = double(gate_up_bytes(sl.f) + down_bytes(sl.fd));
        pass_bytes += bn.mean_u(sl.layer, seq_B) * per;
        rot_bytes += distinct(all) * per;
    }
    std::vector<std::unique_ptr<graph_exec>> keep;
    const auto make = [&](int32_t passes, int32_t first_d, int32_t omit, bool launches) -> graph_exec * {
        keep.push_back(std::make_unique<graph_exec>());
        if (!capture(bn, passes, [&](int32_t i) { return seq_pass(bn, first_d + i, omit, launches); }, *keep.back())) {
            return nullptr;
        }
        return keep.back().get();
    };
    graph_exec * full = make(n_dec, 0, -1, true);
    if (full == nullptr) {
        return 1;
    }
    event_pair ev;
    if (!ev.make()) {
        return 1;
    }
    double us = 0.0;
    if (mode == "seq:once") {
        // ncu: one replay of the whole rotation (n_seq x n_dec = 160 experts launches, ncu's --launch-skip 160), then
        // one pass at decode 0, which follows decode 19's pass as it does in the rotation
        graph_exec * once = make(1, 0, -1, true);
        if (once == nullptr || !replay_us(bn, *full, ev, us) || !replay_us(bn, *once, ev, us)) {
            return 1;
        }
        printf(
            "{\"ticket\": \"CR2\", \"mode\": \"seq:once\", \"layers\": [36, 37, 38, 39], \"rows\": %d, \"skip\": %d, "
            "\"once_pass_us\": %.3f}\n",
            seq_B, n_seq * n_dec, us);
        return 0;
    }
    // A - B: the full rotation against the same graph without launch k (in every pass); the evictors alone give the
    // pass's experts total. Paired: each trial replays A then B, the pairs' order rotating trial to trial.
    std::vector<graph_exec *> minus(n_seq);
    for (int32_t k = 0; k < n_seq; ++k) {
        if ((minus[size_t(k)] = make(n_dec, 0, k, true)) == nullptr) {
            return 1;
        }
    }
    graph_exec * evs = make(n_dec, 0, -1, false);
    if (evs == nullptr) {
        return 1;
    }
    const int32_t n_pair = n_seq + 1; // pair n_seq: full against the evictors alone
    std::vector<std::vector<double>> diff(n_pair), a_us(n_pair);
    for (int32_t i = 0; i < n_warm; ++i) {
        for (int32_t p = 0; p < n_pair; ++p) {
            if (!replay_us(bn, *full, ev, us) || !replay_us(bn, p < n_seq ? *minus[size_t(p)] : *evs, ev, us)) {
                return 1;
            }
        }
    }
    const std::string t0 = utc_now();
    for (int32_t t = 0; t < n_timed; ++t) {
        for (int32_t pp = 0; pp < n_pair; ++pp) {
            const int32_t p = (pp + t) % n_pair;
            double ta = 0.0, tb = 0.0;
            if (!replay_us(bn, *full, ev, ta) || !replay_us(bn, p < n_seq ? *minus[size_t(p)] : *evs, ev, tb)) {
                return 1;
            }
            diff[size_t(p)].push_back((ta - tb) / n_dec);
            a_us[size_t(p)].push_back(ta / n_dec);
        }
    }
    const std::string t1 = utc_now();
    std::vector<double> all_a;
    for (const auto & v : a_us) {
        all_a.insert(all_a.end(), v.begin(), v.end());
    }
    const dist pass = dist_of(all_a), experts = dist_of(diff[size_t(n_seq)]);
    double sum_k = 0.0;
    std::vector<dist> seq_k(n_seq);
    for (int32_t k = 0; k < n_seq; ++k) {
        seq_k[size_t(k)] = dist_of(diff[size_t(k)]);
        sum_k += seq_k[size_t(k)].median;
    }
    printf(
        "{\"ticket\": \"CR2\", \"mode\": \"%s\", \"what\": \"pass\", \"layers\": [36, 37, 38, 39], \"rows\": %d, "
        "\"passes_a_graph\": %d, \"trials\": %d, \"pass_median_us\": %.3f, \"pass_p90_us\": %.3f, "
        "\"experts_in_pass_median_us\": %.3f, \"experts_in_pass_p10_us\": %.3f, \"experts_in_pass_p90_us\": %.3f, "
        "\"sum_of_launch_medians_us\": %.3f, \"routed_bytes_a_pass\": %.0f, \"routed_bytes_distinct_over_rotation\": "
        "%.0f, \"method\": \"per launch: (A - B_k) / %d, A the rotation of %d passes, B_k the same graph without "
        "launch k; experts in pass: (A - evictors alone) / %d; paired replays, order rotating\", \"t_start\": "
        "\"%s\", \"t_end\": \"%s\"}\n",
        mode.c_str(), seq_B, n_dec, n_timed, pass.median, pass.p90, experts.median, experts.p10, experts.p90, sum_k,
        pass_bytes, rot_bytes, n_dec, n_dec, n_dec, t0.c_str(), t1.c_str());
    // each launch alone at the same routing: warm (ex-bench's graph of 64, weights cold, code warm) and cold (flush and
    // eviction before each launch)
    for (int32_t k = 0; k < n_seq; ++k) {
        const seq_layer & sl = seq_layers[k / 2];
        const int32_t * ids = bn.layer_ids.at(sl.layer)->as<int32_t>();
        if (k % 2 == 1) {
            // down's inputs: its layer's gate/up GLU of every decode
            bool ok = true;
            for (int32_t d = 0; d < n_dec && ok; ++d) {
                ok = bn.gate_up_unit(sl.f, sl.fg, sl.fu, seq_B, d, ids);
            }
            if (!ok || !cuda_ok(cudaStreamSynchronize(bn.st), "down inputs")) {
                return 1;
            }
        }
        const auto unit = [&](int32_t i) { return seq_launch(bn, k, i); };
        stats warm, cold;
        if (!time_warm(bn, unit, warm) || !time_cold(bn, unit, cold)) {
            fprintf(stderr, "ex-bench: seq launch %d alone failed\n", k);
            return 1;
        }
        const dist & d = seq_k[size_t(k)];
        printf("{\"ticket\": \"CR2\", \"mode\": \"%s\", \"what\": \"launch\", \"k\": %d, \"layer\": %d, \"kernel\": "
               "\"%s\", \"point\": \"%s\", \"u_mean\": %.2f, \"seq_median_us\": %.3f, \"seq_p10_us\": %.3f, "
               "\"seq_p90_us\": %.3f, \"seq_min_us\": %.3f, \"seq_max_us\": %.3f, \"warm_median_us\": %.3f, "
               "\"warm_p90_us\": %.3f, \"cold_median_us\": %.3f, \"cold_p90_us\": %.3f, \"excess_median_us\": %.3f, "
               "\"excess_over_cold_price\": %.3f, \"t_alone_start\": \"%s\", \"t_alone_end\": \"%s\"",
               mode.c_str(), k, sl.layer, k % 2 ? "down" : "gate_up", seq_point(k).c_str(), bn.mean_u(sl.layer, seq_B),
               d.median, d.p10, d.p90, d.lo, d.hi, warm.median, warm.p90, cold.median, cold.p90, d.median - warm.median,
               (d.median - warm.median) / (cold.median - warm.median), warm.t0.c_str(), cold.t1.c_str());
        print_us("seq_us", diff[size_t(k)]);
        print_us("warm_us", warm.us);
        print_us("cold_us", cold.us);
        printf("}\n");
        fflush(stdout);
    }
    return 0;
}

// ---- CR3: the expert code across one whole layer's traffic ------------------------------------------------------
// Between two launches of the same expert code: the layer's real attention kernel (gdn_core at B = 3, or attend at
// 131,072 cells and B = 3) and a LOAD STAND-IN (layer_stand.h) for the layer's other launches: HC's two sublayers'
// weights (evict-first in both), the layer's dense weights (attention input and output, router and shared gate, the
// shared expert's gate/up: evict-first, the round's form since XB1) and 0.75 MB of normal reads standing for the other
// kernels' code. Layers: GDN 36 (the gate/up unit's own instantiation layer) and QSA 35, at their deployed formats
// (inventory), per card.
struct cr3_mat {
    const char * name;
    dfmt f;
    int32_t k, m;
};
struct cr3_body {
    const char * name;
    int32_t layer;
    std::vector<cr3_mat> input, output, shexp;
};
const cr3_body cr3_gdn = {"GDN",
                          36,
                          {{"attn_qkv", dfmt::q6_k, 2560, 5120},
                           {"attn_gate", dfmt::iq4_xs, 2560, 3072},
                           {"ssm_alpha", dfmt::bf16, 2560, 24},
                           {"ssm_beta", dfmt::bf16, 2560, 24}},
                          {{"ssm_out", dfmt::q6_k, 3072, 2560}},
                          {{"ffn_gate_shexp", dfmt::iq3_s, 2560, 320}, {"ffn_up_shexp", dfmt::iq3_s, 2560, 320}}};
const cr3_body cr3_qsa = {"QSA",
                          35,
                          {{"attn_q", dfmt::q4_k, 2560, 6144},
                           {"attn_k", dfmt::q6_k, 2560, 256},
                           {"attn_v", dfmt::q6_k, 2560, 256},
                           {"indexer.q_proj", dfmt::bf16, 2560, 512},
                           {"indexer.k_proj", dfmt::bf16, 2560, 128}},
                          {{"attn_output", dfmt::q6_k, 3072, 2560}},
                          {{"ffn_gate_shexp", dfmt::iq4_xs, 2560, 320}, {"ffn_up_shexp", dfmt::q6_k, 2560, 320}}};
const std::vector<cr3_mat> cr3_router = {{"ffn_gate_inp", dfmt::bf16, 2560, 512},
                                         {"ffn_gate_inp_shexp", dfmt::bf16, 2560, 1}};
constexpr int64_t cr3_hc_bytes =
    6594560; // an HC sublayer's weights a card: down and up BF16 [10240 x 320] / 2 + inject
constexpr int64_t cr3_code_bytes = 786432; // 0.75 MB: the other kernels' code (the count's 0.47-0.74 MB, upper end)
constexpr int32_t cr3_depth = 131072, cr3_B = 3;

int64_t mats_bytes(const std::vector<cr3_mat> & ms) {
    int64_t b = 0;
    for (const cr3_mat & m : ms) {
        b += fn::dense::row_bytes(m.f, m.k) * m.m;
    }
    return (b + 15) / 16 * 16;
}

struct cr3_layer {
    cudaStream_t st = nullptr;
    // the stand-in's buffers (one each: the arm's leading 12 MB flush leaves them cold)
    std::unique_ptr<dev_mem> hc, input, output, router, shexp, code, sink;
    // gdn_core's: committed state (5 versions), inputs, outputs, workspace, committed slot word
    std::unique_ptr<dev_mem> g_state, g_in, g_out, g_ws, g_cur;
    // attend's: K and V caches, inputs, outputs, workspace, body and offsets, positions
    std::unique_ptr<dev_mem> a_kv, a_in, a_out, a_ws, a_body, a_pos;
    size_t cache_elems = 0;
    int32_t body_words = 0;
    const cr3_body * body = nullptr;
    int64_t in_bytes = 0, out_bytes = 0, router_bytes = 0, shexp_bytes = 0;

    bool make_stand(const cr3_body & b) {
        body = &b;
        in_bytes = mats_bytes(b.input);
        out_bytes = mats_bytes(b.output);
        router_bytes = mats_bytes(cr3_router);
        shexp_bytes = mats_bytes(b.shexp);
        hc = std::make_unique<dev_mem>(size_t(cr3_hc_bytes));
        input = std::make_unique<dev_mem>(size_t(in_bytes));
        output = std::make_unique<dev_mem>(size_t(out_bytes));
        router = std::make_unique<dev_mem>(size_t(router_bytes));
        shexp = std::make_unique<dev_mem>(size_t(shexp_bytes));
        code = std::make_unique<dev_mem>(size_t(cr3_code_bytes));
        sink = std::make_unique<dev_mem>(144 * 4);
        bool ok = true;
        for (const auto * m : {&hc, &input, &output, &router, &shexp, &code}) {
            ok = ok && (*m)->p != nullptr && cuda_ok(cudaMemset((*m)->p, 0x3c, (*m)->n), "stand-in");
        }
        return ok && sink->p != nullptr && cuda_ok(cudaMemset(sink->p, 0, sink->n), "sink");
    }

    bool make_gdn() {
        namespace gd = fn::gdn;
        // ME11: gdn_core's two banks and two record sets
        g_state = std::make_unique<dev_mem>(size_t(gd::banks_bytes + gd::records_bytes));
        // conv [B][5120], z [B][3072], alpha, beta [B][24], dt, a [24], ssm_norm [128] (F32, 256-byte aligned each)
        g_in = std::make_unique<dev_mem>(size_t(1) << 20);
        // gated [B][3072], its codes and scales
        g_out = std::make_unique<dev_mem>(size_t(1) << 20);
        g_ws = std::make_unique<dev_mem>(gd::workspace_bytes());
        g_cur = std::make_unique<dev_mem>(16);
        // values: 0x3c bytes (F32 0.0115 everywhere: finite state, decays and gates)
        return cuda_ok(cudaMemset(g_state->p, 0x3c, g_state->n), "state") &&
               cuda_ok(cudaMemset(g_in->p, 0x3c, g_in->n), "gdn in") &&
               cuda_ok(cudaMemset(g_out->p, 0, g_out->n), "out") && cuda_ok(cudaMemset(g_ws->p, 0, g_ws->n), "ws") &&
               cuda_ok(cudaMemset(g_cur->p, 0, g_cur->n), "cur");
    }
    fn::gdn::core_args gdn_args() const {
        namespace gd = fn::gdn;
        const auto f = [&](size_t off) { return reinterpret_cast<float *>(g_in->p + off); };
        gd::core_args a;
        a.conv = f(0);
        a.z = f(size_t(1) << 17);
        a.alpha = f(size_t(2) << 17);
        a.beta = f((size_t(2) << 17) + 4096);
        a.dt = f((size_t(2) << 17) + 8192);
        a.a = f((size_t(2) << 17) + 12288);
        a.norm_w = f((size_t(2) << 17) + 16384);
        a.state = g_state->as<float>();
        a.rec = reinterpret_cast<float *>(g_state->p + gd::banks_bytes);
        a.gsel = g_cur->as<uint32_t>(); // ME11: selector 0 (bank 0, no records)
        a.gated = g_out->as<float>();
        a.gated_q = {reinterpret_cast<int8_t *>(g_out->p + (size_t(1) << 17)),
                     reinterpret_cast<float *>(g_out->p + (size_t(3) << 17)), cr3_B, gd::n_value};
        a.ws = g_ws->p;
        a.rows = cr3_B;
        return a;
    }

    bool make_qsa() {
        namespace at = fn::qsa::attn;
        namespace qs = fn::qsa;
        cache_elems = (size_t(cr3_depth) + at::max_rows) * at::head_dim;
        a_kv = std::make_unique<dev_mem>(cache_elems * 2 * 2);
        a_in = std::make_unique<dev_mem>(size_t(1) << 20);
        a_out = std::make_unique<dev_mem>(size_t(1) << 17);
        a_ws = std::make_unique<dev_mem>(at::workspace_bytes());
        a_pos = std::make_unique<dev_mem>(size_t(4) * at::max_rows * 4);
        body_words = qs::body_words(cr3_depth, cr3_B);
        a_body = std::make_unique<dev_mem>((size_t(body_words) + at::max_rows + 2) * 4);
        // the B rows' bodies (qa2-bench's disjoint form): each row 512 whole blocks drawn without repeats, its tail
        const int32_t n_blocks = cr3_depth / qs::ratio;
        std::vector<int32_t> perm(static_cast<size_t>(n_blocks));
        for (int32_t i = 0; i < n_blocks; ++i) {
            perm[size_t(i)] = i;
        }
        std::mt19937 g(0x43523353u);
        std::shuffle(perm.begin(), perm.end(), g);
        std::vector<uint32_t> b;
        for (int32_t t = 0; t < cr3_B; ++t) {
            const int32_t q = cr3_depth + t, tail = (q + 1) % qs::ratio;
            std::vector<int32_t> sel(perm.begin() + int64_t(t) * 512, perm.begin() + int64_t(t + 1) * 512);
            std::sort(sel.begin(), sel.end());
            b.push_back(512);
            b.push_back(uint32_t(tail));
            for (const int32_t x : sel) {
                b.push_back(uint32_t(x));
            }
            for (int32_t i = 0; i < tail; ++i) {
                b.push_back(uint32_t((q + 1) / qs::ratio * qs::ratio + i));
            }
        }
        if (int32_t(b.size()) != body_words) {
            fprintf(stderr, "ex-bench: cr3 body has %zu words, not %d\n", b.size(), body_words);
            return false;
        }
        for (int32_t t = 0; t <= cr3_B; ++t) {
            b.push_back(uint32_t(qs::body_offset(cr3_depth, t)));
        }
        b.push_back(uint32_t(cr3_depth)); // the depth on the device, after the offsets (TG1)
        std::vector<int32_t> pos(size_t(4) * cr3_B);
        for (int32_t p = 0; p < 4; ++p) {
            for (int32_t t = 0; t < cr3_B; ++t) {
                pos[size_t(p) * cr3_B + t] = cr3_depth + t;
            }
        }
        // values: 0x3c bytes (BF16 0.0115 in the caches, F32 0.0115 in the inputs and norms)
        return a_kv->p != nullptr && cuda_ok(cudaMemset(a_kv->p, 0x3c, a_kv->n), "kv") &&
               cuda_ok(cudaMemset(a_in->p, 0x3c, a_in->n), "attend in") &&
               cuda_ok(cudaMemset(a_ws->p, 0, a_ws->n), "attend ws") && a_body->put(b.data(), b.size() * 4) &&
               a_pos->put(pos.data(), pos.size() * 4);
    }
    fn::qsa::attn::args qsa_args() const {
        namespace at = fn::qsa::attn;
        at::args a = {};
        const auto f = [&](size_t off) { return reinterpret_cast<float *>(a_in->p + off); };
        a.qfull = f(0);                                 // [B][12][512]: 72 KB at B = 3
        a.kv_k = f(size_t(1) << 18);                    // [B][256]
        a.kv_v = f((size_t(1) << 18) + 8192);           // [B][256]
        a.q_norm = f((size_t(1) << 18) + 16384);        // [256]
        a.k_norm = f((size_t(1) << 18) + 16384 + 1024); // [256]
        a.body = a_body->as<uint32_t>();
        a.offsets = reinterpret_cast<const int32_t *>(a_body->p + size_t(body_words) * 4);
        a.depth = a.offsets + cr3_B + 1;
        a.pos = a_pos->as<int32_t>();
        a.k_cache = reinterpret_cast<uint16_t *>(a_kv->p);
        a.v_cache = a.k_cache + cache_elems;
        a.max_cells = cr3_depth + at::max_rows;
        a.theta_scale = at::rope_theta_scale();
        a.out = a_out->as<float>();
        a.codes = reinterpret_cast<int8_t *>(a_out->p + (size_t(1) << 16));
        a.scales = reinterpret_cast<float *>(a_out->p + (size_t(1) << 16) + 16384);
        a.k_f32 = nullptr;
        a.stamps = nullptr;
        a.ws = at::carve(a_ws->p);
        return a;
    }

    // the layer's traffic in the round's order: HC (attention side), the attention input, the real attention kernel,
    // its output, HC (FFN side), the router and shared gate, the shared expert's gate/up, then the code stand-in. The
    // round's operators (XB1): the dense weights evict-first, and the attention kernels as the round runs them
    // (core_kernel's state and versions and attend's KV union evict-first)
    bool run() const {
        using fn::dense::stand_op;
        uint32_t * sk = sink->as<uint32_t>();
        bool ok = fn::dense::stand_stream(hc->p, cr3_hc_bytes, stand_op::ef, sk, st) &&
                  fn::dense::stand_stream(input->p, in_bytes, stand_op::ef, sk, st);
        if (body == &cr3_gdn) {
            ok = ok && fn::gdn::core(gdn_args(), st);
        } else {
            ok = ok && fn::qsa::attn::attend(cr3_B, qsa_args(), st);
        }
        return ok && fn::dense::stand_stream(output->p, out_bytes, stand_op::ef, sk, st) &&
               fn::dense::stand_stream(hc->p, cr3_hc_bytes, stand_op::ef, sk, st) &&
               fn::dense::stand_stream(router->p, router_bytes, stand_op::ef, sk, st) &&
               fn::dense::stand_stream(shexp->p, shexp_bytes, stand_op::ef, sk, st) &&
               fn::dense::stand_stream(code->p, cr3_code_bytes, stand_op::normal, sk, st);
    }
};

// one timed cold state before a unit: V0 (flush, evictor) or a layer arm (flush, evictor, [pre-touch], the layer's
// traffic, evictor); pre_other: the pre-touch is another unit (XB1's pair arm: A before B), not the unit's own
struct cr3_arm {
    std::string name;
    const cr3_layer * layer; // null: V0
    bool pre, pre_other;
};

// (A - B) / n_cold_rep per arm, A = n_cold_rep x [state, unit c], B = n_cold_rep x [state]; a trial replays every
// arm's B then A, the arms' order rotating trial to trial
bool time_cr3(bench & bn, const std::vector<cr3_arm> & arms, const std::function<bool(int32_t)> & unit,
              const std::function<bool(int32_t)> & pre, const std::function<bool(int32_t)> & pre_other,
              std::vector<stats> & out) {
    const size_t n = arms.size();
    std::vector<std::unique_ptr<graph_exec>> ga, gb;
    event_pair ev;
    bool ok = ev.make();
    for (size_t k = 0; k < n && ok; ++k) {
        const cr3_arm & r = arms[k];
        const auto state = [&](int32_t c) {
            bool s = bn.cold();
            if (r.layer != nullptr) {
                s = s && (!r.pre || (r.pre_other ? pre_other(c) : pre(c))) && r.layer->run() &&
                    fn::dense::cold_icache(bn.evict.as<float>(), bn.st);
            }
            return s;
        };
        ga.push_back(std::make_unique<graph_exec>());
        gb.push_back(std::make_unique<graph_exec>());
        ok = capture(
                 bn, n_cold_rep, [&](int32_t c) { return state(c) && unit(c); }, *ga.back()) &&
             capture(bn, n_cold_rep, state, *gb.back());
    }
    out.assign(n, stats());
    double ta = 0.0, tb = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        for (size_t k = 0; k < n && ok; ++k) {
            ok = replay_us(bn, *gb[k], ev, tb) && replay_us(bn, *ga[k], ev, ta);
        }
    }
    for (size_t k = 0; k < n; ++k) {
        out[k].t0 = utc_now();
    }
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        for (size_t kk = 0; kk < n && ok; ++kk) {
            const size_t k = (kk + size_t(i)) % n;
            ok = replay_us(bn, *gb[k], ev, tb) && replay_us(bn, *ga[k], ev, ta);
            out[k].us.push_back((ta - tb) / n_cold_rep);
        }
    }
    for (size_t k = 0; k < n && ok; ++k) {
        out[k].t1 = utc_now();
        out[k].finish();
    }
    return ok;
}

// down point name -> its formats and layer (the first deployed instantiation of that pair)
bool down_point(const std::string & point, fmt & f, fmt & fs, int32_t & layer) {
    for (int32_t i = 0; i < n_down_inst; ++i) {
        down_inst(i, f, fs, layer);
        if (point == "dn_" + lower(fmt_name(f)) + "_" + lower(fmt_name(fs)) + "_B3") {
            return true;
        }
    }
    return false;
}

// gate/up point name -> its formats and layer (the deployed instantiation)
bool gu_point(const std::string & point, fmt & f, dfmt & fg, dfmt & fu, int32_t & layer) {
    for (int32_t i = 0; i < n_gate_up_inst; ++i) {
        gate_up_inst(i, f, fg, fu, layer);
        if (point == "gu_" + lower(fmt_name(f)) + "_" + lower(fn::dense::fmt_name(fg)) + "_" +
                         lower(fn::dense::fmt_name(fu)) + "_B3") {
            return true;
        }
    }
    return false;
}

// cr3seq:POINTS: each named B = 3 unit (CR1's: gu_iq3_xxs_iq3_s_iq3_s_B3, dn_q2_0_iq4_nl_B3) warm, at V0, and after
// one layer's traffic with the round's operators (XB1: CR3's fix is the round's form): GDN and QSA, with and without
// the pre-touch (the same kernel on CR1's pre-touch copies). XB1's pair arms, a point A>B (two down points): B warm,
// at V0, after GDN's traffic with A as the pre-touch (A's down, today's per-pair instantiation, on its own pre-touch
// weight copies), and after GDN's traffic without a pre-touch.
int32_t run_cr3seq(bench & bn, const std::string & points) {
    cr3_layer lg, lq;
    lg.st = lq.st = bn.st;
    if (!lg.make_stand(cr3_gdn) || !lg.make_gdn() || !lq.make_stand(cr3_qsa) || !lq.make_qsa() ||
        !cuda_ok(cudaDeviceSynchronize(), "cr3 setup")) {
        return 1;
    }
    for (const cr3_layer * l : {&lg, &lq}) {
        printf("{\"ticket\": \"XB1\", \"what\": \"layer\", \"body\": \"%s\", \"layer\": %d, \"stand_in\": \"LOAD "
               "STAND-IN (layer_stand.h)\", \"order\": \"HC attn (ef), input (ef), real %s, output (ef), HC ffn (ef), "
               "router + shared gate (ef), shexp gate/up (ef), code (normal), then cold_icache\", \"hc_bytes_each\": "
               "%lld, \"input_bytes\": %lld, \"output_bytes\": %lld, \"router_bytes\": %lld, \"shexp_bytes\": %lld, "
               "\"code_bytes\": %lld, \"real_kernel\": \"%s (the round's operators)\", \"rows\": %d, \"depth\": %d}\n",
               l->body->name, l->body->layer, l->body == &cr3_gdn ? "gdn_core" : "attend",
               static_cast<long long>(cr3_hc_bytes), static_cast<long long>(l->in_bytes),
               static_cast<long long>(l->out_bytes), static_cast<long long>(l->router_bytes),
               static_cast<long long>(l->shexp_bytes), static_cast<long long>(cr3_code_bytes),
               l->body == &cr3_gdn ? "gdn_core" : "attend", cr3_B, l->body == &cr3_gdn ? 0 : cr3_depth);
    }
    fflush(stdout);
    int32_t found = 0;
    const auto run_unit = [&](const std::string & point, int32_t layer, const std::vector<cr3_arm> & arms,
                              const std::function<bool(int32_t)> & unit, const std::function<bool(int32_t)> & pre,
                              const std::function<bool(int32_t)> & pre_other) {
        stats warm;
        std::vector<stats> v;
        if (!time_warm(bn, unit, warm) || !time_cr3(bn, arms, unit, pre, pre_other, v)) {
            fprintf(stderr, "ex-bench: cr3seq %s failed\n", point.c_str());
            return false;
        }
        const auto line = [&](const std::string & what, const stats & s) {
            printf("{\"ticket\": \"XB1\", \"what\": \"arm\", \"point\": \"%s\", \"layer\": %d, \"arm\": \"%s\", "
                   "\"median_us\": %.3f, \"p10_us\": %.3f, \"p90_us\": %.3f, \"t_start\": \"%s\", \"t_end\": \"%s\"",
                   point.c_str(), layer, what.c_str(), s.median, p10_of(s), s.p90, s.t0.c_str(), s.t1.c_str());
            print_us("us", s.us);
            printf("}\n");
        };
        line("warm", warm);
        for (size_t k = 0; k < arms.size(); ++k) {
            line(arms[k].name, v[k]);
        }
        fflush(stdout);
        ++found;
        return true;
    };
    const std::vector<cr3_arm> layer_arms = {{"V0", nullptr, false, false},
                                             {"GDN pre-touch", &lg, true, false},
                                             {"GDN no pre-touch", &lg, false, false},
                                             {"QSA pre-touch", &lq, true, false},
                                             {"QSA no pre-touch", &lq, false, false}};
    const auto down_inputs = [&](int32_t layer) {
        // the first gate/up instantiation deployed at the same layer writes each decode's GLU (down's input)
        fmt gf{};
        dfmt gg{}, gu{};
        for (int32_t j = 0; j < n_gate_up_inst; ++j) {
            int32_t lj = 0;
            fmt a{};
            dfmt b{}, c{};
            gate_up_inst(j, a, b, c, lj);
            if (j == 0 || lj == layer) {
                gf = a, gg = b, gu = c;
            }
            if (lj == layer) {
                break;
            }
        }
        bool ok = bn.set_routes(layer, 3);
        for (int32_t d = 0; d < n_dec && ok; ++d) {
            ok = bn.gate_up_unit(gf, gg, gu, 3, d);
        }
        return ok && cuda_ok(cudaStreamSynchronize(bn.st), "down inputs") && bn.sync_pre();
    };
    size_t pos = 0;
    while (pos <= points.size()) {
        const size_t end = std::min(points.find(',', pos), points.size());
        const std::string point = points.substr(pos, end - pos);
        pos = end + 1;
        if (point.empty()) {
            continue;
        }
        const size_t gt = point.find('>');
        if (gt != std::string::npos) { // XB1's pair arm A>B
            fmt ga{}, gb{};
            dfmt gga{}, gua{}, ggb{}, gub{};
            int32_t lga = 0, lgb = 0;
            if (gu_point(point.substr(0, gt), ga, gga, gua, lga) && gu_point(point.substr(gt + 1), gb, ggb, gub, lgb)) {
                // XB3 part 6's hit pair: two gate/up points, B after GDN's traffic with A as the pre-touch (A's kernel
                // on its own pre-touch copies, B's routing), set against B's own repeat in the same run
                const std::vector<cr3_arm> arms = {{"V0", nullptr, false, false},
                                                   {"GDN pre-touch " + point.substr(0, gt), &lg, true, true},
                                                   {"GDN pre-touch own", &lg, true, false},
                                                   {"GDN no pre-touch", &lg, false, false}};
                if (!bn.set_routes(lgb, 3) || !bn.sync_pre() ||
                    !run_unit(
                        point, lgb, arms, [&](int32_t k) { return bn.gate_up_unit(gb, ggb, gub, 3, k); },
                        [&](int32_t k) { return bn.gate_up_unit(gb, ggb, gub, 3, k, nullptr, true); },
                        [&](int32_t k) { return bn.gate_up_unit(ga, gga, gua, 3, k, nullptr, true); })) {
                    return 1;
                }
                continue;
            }
            fmt fa{}, fsa{}, f{}, fs{};
            int32_t la = 0, layer = 0;
            if (!down_point(point.substr(0, gt), fa, fsa, la) || !down_point(point.substr(gt + 1), f, fs, layer)) {
                fprintf(stderr, "ex-bench: cr3seq pair %s: two down points or two gate/up points\n", point.c_str());
                return 1;
            }
            const std::vector<cr3_arm> arms = {{"V0", nullptr, false, false},
                                               {"GDN pre-touch " + point.substr(0, gt), &lg, true, true},
                                               {"GDN no pre-touch", &lg, false, false}};
            if (!down_inputs(layer) || !run_unit(
                                           point, layer, arms, [&](int32_t k) { return bn.down_unit(f, fs, 3, k); },
                                           [&](int32_t k) { return bn.down_unit(f, fs, 3, k, nullptr, true); },
                                           [&](int32_t k) { return bn.down_unit(fa, fsa, 3, k, nullptr, true); })) {
                return 1;
            }
            continue;
        }
        bool done = false;
        for (int32_t i = 0; i < n_gate_up_inst && !done; ++i) {
            fmt f{};
            dfmt fg{}, fu{};
            int32_t layer = 0;
            gate_up_inst(i, f, fg, fu, layer);
            if (point != "gu_" + lower(fmt_name(f)) + "_" + lower(fn::dense::fmt_name(fg)) + "_" +
                             lower(fn::dense::fmt_name(fu)) + "_B3") {
                continue;
            }
            done = true;
            const auto u = [&](int32_t k) { return bn.gate_up_unit(f, fg, fu, 3, k); };
            const auto up = [&](int32_t k) { return bn.gate_up_unit(f, fg, fu, 3, k, nullptr, true); };
            if (!bn.set_routes(layer, 3) || !bn.sync_pre() || !run_unit(point, layer, layer_arms, u, up, up)) {
                return 1;
            }
        }
        fmt f{}, fs{};
        int32_t layer = 0;
        if (!done && down_point(point, f, fs, layer)) {
            done = true;
            const auto u = [&](int32_t k) { return bn.down_unit(f, fs, 3, k); };
            const auto up = [&](int32_t k) { return bn.down_unit(f, fs, 3, k, nullptr, true); };
            if (!down_inputs(layer) || !run_unit(point, layer, layer_arms, u, up, up)) {
                return 1;
            }
        }
        if (!done) {
            fprintf(stderr, "ex-bench: cr3seq: %s is not a B = 3 point\n", point.c_str());
            return 1;
        }
    }
    if (found == 0) {
        fprintf(stderr, "ex-bench: cr3seq: no B = 3 point in %s\n", points.c_str());
        return 1;
    }
    return 0;
}

// XB2 step 5 (EL1's el1prof): xb2prof:POINTS (B = 3 names), each listed point's gate/up once a decode, decodes d_first
// .. d_first + n_dec - 1, each after its meta launch, point after point in experts_inst.h's order, no warm-up; for
// ncu's count. A "# xb2prof:" line a launch names its point, decode and the routing's 30 IDs.
int xb2_prof(bench & bn, const std::string & points) {
    const std::string list = "," + points + ",";
    int32_t n_launch = 0;
    for (int32_t i = 0; i < n_gate_up_inst; ++i) {
        fmt f{};
        dfmt fg{}, fu{};
        int32_t layer = 0;
        gate_up_inst(i, f, fg, fu, layer);
        const std::string point = "gu_" + lower(fmt_name(f)) + "_" + lower(fn::dense::fmt_name(fg)) + "_" +
                                  lower(fn::dense::fmt_name(fu)) + "_B3";
        if (list.find("," + point + ",") == std::string::npos) {
            continue;
        }
        if (!bn.set_routes(layer, 3)) {
            return 1;
        }
        bool ok = true;
        for (int32_t d = 0; d < n_dec && ok; ++d) {
            ok = bn.gate_up_unit(f, fg, fu, 3, d);
            printf("# xb2prof: launch %d point %s layer %d decode %d ids", n_launch++, point.c_str(),
                   route_layer(layer), d_first + d);
            for (const int32_t id : rows_of(bn.routes[route_layer(layer)], d, 3)) {
                printf(" %d", id);
            }
            printf("\n");
        }
        if (!ok || !cuda_ok(cudaStreamSynchronize(bn.st), "xb2prof")) {
            return 1;
        }
    }
    printf("# ex-bench: xb2prof, %d launches, end %s\n", n_launch, utc_now().c_str());
    return n_launch > 0 ? 0 : 1;
}

// SP4 Amendment 1: sp4prof:POINTS:MAP (point names without _B), each listed point's B = 5 gate/up once a decode,
// decodes d_first .. d_first + 3 (n_dec 4 for this mode), point after point in experts_inst.h's order, each launch
// after cold() (the data flush and cold_icache: the round fetches expert code cold at every layer, FINDINGS EX-11), no
// warm-up; MAP 0 SP2's 640-thread kernels, 1 SP4's paired 320-thread ones (gate_up_args::map), on identical inputs; for
// ncu. Routing: HX7's measured teacher-forced rows (rows 0-2 of decode i, rows 0-1 of decode i + 1), not five-row
// verify routing. A "# sp4prof:" line a launch names its point, decode, U, the active-row histogram (units with 1..5
// routed rows) and the routing's 50 IDs.
int sp4_prof(bench & bn, const std::string & arg) {
    constexpr int32_t B = 5, n_prof = 4;
    const size_t colon = arg.rfind(':');
    const std::string points = colon == std::string::npos ? "" : arg.substr(0, colon);
    const std::string m = colon == std::string::npos ? "" : arg.substr(colon + 1);
    if (points.empty() || (m != "0" && m != "1")) {
        fprintf(stderr, "ex-bench: sp4prof:POINTS:MAP, MAP 0 (SP2) or 1 (SP4)\n");
        return 2;
    }
    bn.ex_map = m == "1" ? 1 : 0;
    printf("# sp4prof: map %d (%s), B = %d, decodes %d .. %d, routing HX7 capture-58581 measured teacher-forced rows "
           "(rows 0-2 of decode i, rows 0-1 of decode i + 1), not five-row verify routing; cold() before each launch\n",
           bn.ex_map, bn.ex_map == 1 ? "SP4 paired 320-thread" : "SP2 640-thread", B, d_first, d_first + n_prof - 1);
    const std::string list = "," + points + ",";
    int32_t n_launch = 0;
    for (int32_t i = 0; i < n_gate_up_inst; ++i) {
        fmt f{};
        dfmt fg{}, fu{};
        int32_t layer = 0;
        gate_up_inst(i, f, fg, fu, layer);
        const std::string point = "gu_" + lower(fmt_name(f)) + "_" + lower(fn::dense::fmt_name(fg)) + "_" +
                                  lower(fn::dense::fmt_name(fu));
        if (list.find("," + point + ",") == std::string::npos) {
            continue;
        }
        if (f == fmt::q8_0 || !bn.set_routes(layer, B)) {
            return 1;
        }
        bool ok = true;
        for (int32_t d = 0; d < n_prof && ok; ++d) {
            ok = bn.cold() && bn.gate_up_unit(f, fg, fu, B, d);
            const std::vector<int32_t> r = rows_of(bn.routes[route_layer(layer)], d, B);
            std::map<int32_t, int32_t> rows; // ID -> its routed rows
            for (const int32_t id : r) {
                ++rows[id];
            }
            int32_t h[B + 1] = {};
            for (const auto & kv : rows) {
                ++h[std::min(kv.second, B)];
            }
            printf("# sp4prof: launch %d point %s map %d layer %d decode %d U %d rows1-5 %d,%d,%d,%d,%d ids",
                   n_launch++, point.c_str(), bn.ex_map, route_layer(layer), d_first + d, int32_t(rows.size()), h[1],
                   h[2], h[3], h[4], h[5]);
            for (const int32_t id : r) {
                printf(" %d", id);
            }
            printf("\n");
        }
        if (!ok || !cuda_ok(cudaStreamSynchronize(bn.st), "sp4prof")) {
            return 1;
        }
    }
    uint32_t st = 0;
    if (!cuda_ok(cudaMemcpy(&st, bn.status.p, 4, cudaMemcpyDeviceToHost), "status") || st != 0) {
        fprintf(stderr, "ex-bench: status %u\n", st);
        return 1;
    }
    printf("# ex-bench: sp4prof, %d launches, end %s\n", n_launch, utc_now().c_str());
    return n_launch > 0 ? 0 : 1;
}

} // namespace

int fnfl_ex_bench(int device, const char * mode, const char * cache_dir, const char * routes_dir) {
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    bench bn;
    bn.cache = cache_dir;
    bn.hx7 = routes_dir;
    if (!bn.prepare()) {
        return 1;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    // mode: "all", or a comma list of point names or name prefixes (gu_<routed>_<gate>_<up>_B<rows>,
    // dn_<routed>_<shared>_B<rows>; for example gu_iq3_s,dn_q2_0) timed warm and cold; CR1: "r4:" and such a list,
    // the matching B = 3 points timed warm and in R4's four cold states
    const bool r4 = mode != nullptr && strncmp(mode, "r4:", 3) == 0;
    const std::string sel = mode == nullptr || std::string(mode) == "all" ? "" : r4 ? mode + 3 : mode;
    if (r4 && (sel.empty() || !bn.prepare_r4())) {
        return 1;
    }
    const auto skip = [&](const std::string & point) {
        if (r4 && point.compare(point.size() - 3, 3, "_B3") != 0) {
            return true;
        }
        if (sel.empty()) {
            return false;
        }
        size_t pos = 0;
        while (pos <= sel.size()) {
            const size_t end = std::min(sel.find(',', pos), sel.size());
            const std::string pre = sel.substr(pos, end - pos);
            if (!pre.empty() && point.compare(0, pre.size(), pre) == 0) {
                return false;
            }
            pos = end + 1;
        }
        return true;
    };
    const std::string & only = sel;
    printf("# ex-bench: %s, sm_%d%d, %d SMs, mode %s, start %s\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, only.empty() ? "all" : only.c_str(), utc_now().c_str());
    if (sel.compare(0, 8, "xb2prof:") == 0) {
        return xb2_prof(bn, sel.substr(8));
    }
    if (sel.compare(0, 8, "sp4prof:") == 0) {
        return sp4_prof(bn, sel.substr(8));
    }
    if (!bn.warm_up()) {
        return 1;
    }
    if (only.compare(0, 7, "cr3seq:") == 0) {
        // CR3: the units after one layer's traffic (run_cr3seq)
        const int32_t rc = bn.prepare_r4() ? run_cr3seq(bn, only.substr(7)) : 1;
        uint32_t st = 0;
        if (rc == 0 && (!cuda_ok(cudaMemcpy(&st, bn.status.p, 4, cudaMemcpyDeviceToHost), "status") || st != 0)) {
            fprintf(stderr, "ex-bench: status %u\n", st);
            return 1;
        }
        printf("# ex-bench: %s, end %s\n", only.c_str(), utc_now().c_str());
        return rc;
    }
    if (only.compare(0, 3, "seq") == 0) {
        // CR2: seq (the sequence timed, and each launch alone) or seq:once (ncu's capture)
        if (only != "seq" && only != "seq:once") {
            fprintf(stderr, "ex-bench: seq modes are seq and seq:once\n");
            return 2;
        }
        const int32_t rc = run_seq(bn, only);
        uint32_t st = 0;
        if (rc == 0 && (!cuda_ok(cudaMemcpy(&st, bn.status.p, 4, cudaMemcpyDeviceToHost), "status") || st != 0)) {
            fprintf(stderr, "ex-bench: status %u\n", st);
            return 1;
        }
        printf("# ex-bench: %s, end %s\n", only.c_str(), utc_now().c_str());
        return rc;
    }
    int32_t n_points = 0;
    for (int32_t i = 0; i < n_gate_up_inst; ++i) {
        fmt f{};
        dfmt fg{}, fu{};
        int32_t layer = 0;
        gate_up_inst(i, f, fg, fu, layer);
        for (const int32_t B : {1, 3}) {
            const std::string point = "gu_" + lower(fmt_name(f)) + "_" + lower(fn::dense::fmt_name(fg)) + "_" +
                                      lower(fn::dense::fmt_name(fu)) + "_B" + std::to_string(B);
            if (skip(point)) {
                continue;
            }
            if (!bn.set_routes(layer, B)) {
                return 1;
            }
            const double u = bn.mean_u(layer, B);
            const double bytes = u * double(gate_up_bytes(f)) + double(bn.sh_gu_bytes[int32_t(fg)]) +
                                 double(bn.sh_gu_bytes[int32_t(fu)]);
            const auto unit = [&](int32_t k) { return bn.gate_up_unit(f, fg, fu, B, k); };
            const std::string formats =
                std::string(fmt_name(f)) + " + shared " + fn::dense::fmt_name(fg) + "/" + fn::dense::fmt_name(fu);
            if (r4) {
                const auto pre = [&](int32_t k) { return bn.gate_up_unit(f, fg, fu, B, k, nullptr, true); };
                stats warm, v[4];
                if (!bn.sync_pre() || !time_warm(bn, unit, warm) || !time_r4(bn, unit, pre, v)) {
                    fprintf(stderr, "ex-bench: %s failed\n", point.c_str());
                    return 1;
                }
                print_r4(point, formats, layer, B, u, bytes, warm, v);
                ++n_points;
                continue;
            }
            stats warm, cold;
            if (!time_warm(bn, unit, warm) || !time_cold(bn, unit, cold)) {
                fprintf(stderr, "ex-bench: %s failed\n", point.c_str());
                return 1;
            }
            print_point(point, formats, layer, B, u, bytes, warm, cold);
            ++n_points;
        }
    }
    for (int32_t i = 0; i < n_down_inst; ++i) {
        fmt f{}, fs{};
        int32_t layer = 0;
        down_inst(i, f, fs, layer);
        // the first gate/up instantiation deployed at the same layer writes each decode's GLU (down's input)
        fmt gf{};
        dfmt gg{}, gu{};
        for (int32_t j = 0; j < n_gate_up_inst; ++j) {
            int32_t lj = 0;
            fmt a{};
            dfmt b{}, c{};
            gate_up_inst(j, a, b, c, lj);
            if (j == 0 || lj == layer) {
                gf = a, gg = b, gu = c;
            }
            if (lj == layer) {
                break;
            }
        }
        for (const int32_t B : {1, 3}) {
            const std::string point = "dn_" + lower(fmt_name(f)) + "_" + lower(fmt_name(fs)) + "_B" + std::to_string(B);
            if (skip(point)) {
                continue;
            }
            bool ok = bn.set_routes(layer, B);
            for (int32_t d = 0; d < n_dec && ok; ++d) {
                ok = bn.gate_up_unit(gf, gg, gu, B, d);
            }
            if (!ok || !cuda_ok(cudaStreamSynchronize(bn.st), "down inputs")) {
                return 1;
            }
            const double u = bn.mean_u(layer, B);
            const double bytes = u * double(down_bytes(f)) + double(down_bytes(fs));
            const auto unit = [&](int32_t k) { return bn.down_unit(f, fs, B, k); };
            const std::string formats = std::string(fmt_name(f)) + " + shared " + fmt_name(fs);
            if (r4) {
                const auto pre = [&](int32_t k) { return bn.down_unit(f, fs, B, k, nullptr, true); };
                stats warm, v[4];
                if (!bn.sync_pre() || !time_warm(bn, unit, warm) || !time_r4(bn, unit, pre, v)) {
                    fprintf(stderr, "ex-bench: %s failed\n", point.c_str());
                    return 1;
                }
                print_r4(point, formats, layer, B, u, bytes, warm, v);
                ++n_points;
                continue;
            }
            stats warm, cold;
            if (!time_warm(bn, unit, warm) || !time_cold(bn, unit, cold)) {
                fprintf(stderr, "ex-bench: %s failed\n", point.c_str());
                return 1;
            }
            print_point(point, formats, layer, B, u, bytes, warm, cold);
            ++n_points;
            if (!time_meta(bn, point, layer, B, u, bn.dn_class_w)) {
                return 1;
            }
        }
    }
    uint32_t st = 0;
    if (!cuda_ok(cudaMemcpy(&st, bn.status.p, 4, cudaMemcpyDeviceToHost), "status") || st != 0) {
        fprintf(stderr, "ex-bench: status %u\n", st);
        return 1;
    }
    printf("# ex-bench: %d points, end %s\n", n_points, utc_now().c_str());
    return n_points > 0 ? 0 : 1;
}
