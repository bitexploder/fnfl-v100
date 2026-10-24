#include "fn/model/kv_tier_test.h"

#include "fn/kernels/draft/draft.h"
#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/qsa/qsa_prompt.h"
#include "fn/model/kv_tier.h"
#include "fn/model/model.h"
#include "fn/model/state.h"
#include "fn/quant/formats.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

namespace md = fn::model;
namespace at = fn::qsa::attn;
namespace dr = fn::draft;
namespace qs = fn::qsa;

constexpr int32_t HD = 256;
constexpr int32_t NH = 12;
constexpr int32_t QS = 512;
constexpr int32_t a_target = 5; // the tracked QSA layer (layer 23)
constexpr int32_t a_draft = md::kv_tier::draft_attn;
constexpr int32_t KD = qs::head_dim; // ME6: a raw index key's values

// ME6: the tracked target key layer of card c (the keys live on the owner): a_target on its owner, a_target - 1 on
// the other card; slot 1 the draft (both cards)
int32_t key_layer(int32_t c, int32_t k) {
    return k == 1 ? a_draft : (c == a_target % 2 ? a_target : a_target - 1);
}

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "kv-tier-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

struct rng {
    uint64_t s;

    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * float(next() >> 40) * (1.0f / 16777216.0f); }
    int64_t below(int64_t n) { return n <= 0 ? 0 : int64_t(next() % uint64_t(n)); }
};

template <typename T> struct dev {
    T * p = nullptr;
    size_t n = 0;
    int device = -1;
    explicit dev(size_t count, bool shared = false) : n(count) {
        cuda_ok(cudaGetDevice(&device), "allocation device");
        if (!cuda_ok(shared ? cudaMallocManaged(&p, n * sizeof(T)) : cudaMalloc(&p, n * sizeof(T)), "allocate")) {
            p = nullptr;
        }
    }
    ~dev() {
        if (p != nullptr) {
            int previous = -1;
            cuda_ok(cudaGetDevice(&previous), "free device");
            cuda_ok(cudaSetDevice(device), "free device");
            cuda_ok(cudaFree(p), "free");
            cuda_ok(cudaSetDevice(previous), "restore device");
        }
    }
    dev(const dev &) = delete;
    dev & operator=(const dev &) = delete;
};

// MEKV: common launch inputs/scratch migrate between cards only after the previous stream drains.
// Reference caches, tier windows and summaries remain ordinary allocations on their owning device.
template <typename T> struct shared_dev : dev<T> {
    explicit shared_dev(size_t count) : dev<T>(count, true) {}
};

template <typename T> bool put(T * d, const std::vector<T> & h, cudaStream_t st) {
    return cuda_ok(cudaMemcpyAsync(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice, st), "upload");
}

template <typename T> bool get(std::vector<T> & h, const T * d, size_t n, cudaStream_t st) {
    h.resize(n);
    return cuda_ok(cudaMemcpyAsync(h.data(), d, n * sizeof(T), cudaMemcpyDefault, st), "download") &&
           cuda_ok(cudaStreamSynchronize(st), "download");
}

struct checks {
    int64_t n = 0, failed = 0;
    void check(bool ok, const std::string & what) {
        ++n;
        if (!ok && ++failed <= 20) {
            printf("  FAIL %s\n", what.c_str());
        }
    }
};

// cells [first, first + n) of 256 values: BF16 (target) or F16 (draft) of uniform(-2, 2)
std::vector<uint16_t> random_cells(rng & r, int64_t n, bool f16) {
    std::vector<uint16_t> v(size_t(n) * HD);
    for (uint16_t & x : v) {
        const float f = r.uniform(-2.0f, 2.0f);
        x = f16 ? fn::quant::f32_to_f16(f) : fn::quant::f32_to_bf16(f);
    }
    return v;
}

// one row's selected blocks at position q (qsa.h body: K = min(n_vis, 512) distinct ascending blocks of n_vis), mixing
// the window's edge (lo / 4 - 2 .. lo / 4 + 1), the oldest and newest blocks, a pool shared by the round's rows (the
// gather's union) and fresh ones
std::vector<int32_t> pick_blocks(rng & r, int32_t n_vis, int32_t lo, const std::vector<int32_t> & shared) {
    const int32_t K = std::min(n_vis, qs::top_blocks);
    std::vector<int32_t> out;
    if (K == n_vis) {
        for (int32_t b = 0; b < n_vis; ++b) {
            out.push_back(b);
        }
        return out;
    }
    std::set<int32_t> s;
    const int32_t e = lo / qs::ratio;
    for (int32_t b : {0, 1, e - 2, e - 1, e, e + 1, n_vis - 1, n_vis - 2}) {
        if (b >= 0 && b < n_vis && int32_t(s.size()) < K) {
            s.insert(b);
        }
    }
    for (int32_t b : shared) {
        if (int32_t(s.size()) >= K) {
            break;
        }
        if (b < n_vis && (r.next() & 1u) != 0) {
            s.insert(b);
        }
    }
    while (int32_t(s.size()) < K) {
        s.insert(int32_t(r.below(n_vis)));
    }
    out.assign(s.begin(), s.end());
    return out;
}

// the B rows' body and byte offsets at depth (qsa.h layout)
void make_body(rng & r, int32_t depth, int32_t rows, int32_t lo, std::vector<uint32_t> & body,
               std::vector<int32_t> & offsets) {
    body.clear();
    offsets.clear();
    std::vector<int32_t> shared;
    const int32_t n_vis_max = (depth + rows) / qs::ratio;
    for (int32_t i = 0; i < 400 && n_vis_max > 0; ++i) {
        shared.push_back(int32_t(r.below(n_vis_max)));
    }
    for (int32_t t = 0; t < rows; ++t) {
        const int32_t q = depth + t, n_vis = (q + 1) / qs::ratio, tail = (q + 1) % qs::ratio;
        offsets.push_back(int32_t(body.size()) * 4);
        const std::vector<int32_t> b = pick_blocks(r, n_vis, lo, shared);
        body.push_back(uint32_t(b.size()));
        body.push_back(uint32_t(tail));
        for (int32_t x : b) {
            body.push_back(uint32_t(x));
        }
        for (int32_t c = qs::ratio * n_vis; c <= q; ++c) {
            body.push_back(uint32_t(c));
        }
    }
    offsets.push_back(int32_t(body.size()) * 4);
}

struct world {
    int32_t dev_id = 0;
    int32_t n_cells = 0, window = 0;
    cudaStream_t st = nullptr;
    cudaStream_t streams[2] = {};
    bool two_devices = false;
    bool on(int32_t c) {
        if (st != nullptr && !cuda_ok(cudaStreamSynchronize(st), "scratch handoff")) {
            return false;
        }
        if (!cuda_ok(cudaSetDevice(m.device[c]), "shard device")) {
            return false;
        }
        st = streams[c];
        return true;
    }
    md::model m;
    md::kv_tier tier;
    // the resident reference of the tracked layers: [card][0 target, 1 draft] K and V
    std::unique_ptr<dev<uint16_t>> rk[2][2], rv[2][2];
    // launch buffers
    shared_dev<float> qfull{size_t(qs::max_rows) * NH * QS};
    shared_dev<float> kvk{size_t(qs::max_rows) * HD}, kvv{size_t(qs::max_rows) * HD};
    shared_dev<float> qn{HD}, kn{HD};
    shared_dev<int32_t> depth{1}, pos{size_t(4) * qs::max_rows}, offsets{size_t(qs::max_rows) + 1}, stride{1};
    shared_dev<uint32_t> body{size_t(qs::max_rows) * qs::max_row_words}, err{2};
    shared_dev<uint8_t> aws{at::workspace_bytes()};
    shared_dev<float> out{size_t(qs::max_rows) * at::out_width}, scales{size_t(qs::max_rows) * at::out_width / 16};
    shared_dev<int8_t> codes{size_t(qs::max_rows) * at::out_width};
    shared_dev<uint16_t> q16{size_t(dr::max_rows) * NH * HD};
    shared_dev<float> part{dr::part_bytes() / 4};
    int64_t n = 0; // the slot's cells (target and draft alike)
    checks ck;
    int64_t gathered_rounds = 0, rebuilds_seen = 0;
    // ME3: the graph form, a third arm in the window-move case's rounds: the tier in graph mode (set_graph: each
    // layer's window base, the device word's shift in the launches, select's remap always), on the round's inputs
    // again, against the resident arm
    bool graph_arm = false;
    checks ckg;
    int64_t graph_rounds = 0, graph_gathered_rounds = 0;
    // ME6: the raw index keys: the resident key cache of the tracked key layers [card][0 target, 1 draft], the
    // summaries each arm pools [card][layer][arm] (0 resident, 1 tier, 2 tier in graph mode), the new cells' F32 keys
    std::unique_ptr<dev<uint16_t>> rkey[2][2];
    std::unique_ptr<dev<float>> sums[2][2][3];
    shared_dev<float> ik{size_t(qs::prompt_max_rows) * KD}, ikn{KD};
    int64_t key_rounds = 0, key_blocks = 0, key_prompt_blocks = 0, key_moved_rounds = 0;
    bool key_control_run = false;
    // ME6: arm 2's captured graphs [card][layer][rows] (one summary_update each, captured at first use with the
    // window's base and the device word), the window start at capture, and the replays at another window start
    cudaGraphExec_t key_exec[2][2][qs::max_rows + 1] = {};
    int64_t key_exec_lo[2][2][qs::max_rows + 1] = {};
    int64_t key_graph_replays = 0, key_graph_moved = 0;

    uint16_t * ref(int32_t c, int32_t which, bool v) const { return (v ? rv : rk)[c][which]->p; }
};

bool build(world & w, int32_t dev_id, int32_t n_cells, int32_t window, bool two_devices) {
    w.dev_id = dev_id;
    w.n_cells = n_cells;
    w.window = window;
    w.two_devices = two_devices;
    w.m.device[0] = dev_id;
    w.m.device[1] = two_devices ? 1 : dev_id;
    w.m.single = !two_devices;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(w.m.device[c]), "device") ||
            !cuda_ok(cudaStreamCreateWithFlags(&w.streams[c], cudaStreamNonBlocking), "stream")) {
            return false;
        }
    }
    w.st = w.streams[0];
    w.m.kv_host = true;
    md::state_params sp;
    sp.n_cells = n_cells;
    sp.kv_window = window;
    std::string err;
    for (const md::buffer_class & plan : md::state_classes(sp)) {
        if (plan.name != "target KV (K0)" && plan.name != "draft KV (F16)" && plan.name != "KV tier (ME1)" &&
            plan.name != "index key windows (ME6)" && plan.name != "target index keys (ME6)" &&
            plan.name != "draft index keys (ME6)") {
            continue;
        }
        auto rc = std::make_unique<md::reserved_class>();
        rc->plan = plan;
        const bool host = md::host_class(plan.name, !two_devices, true);
        for (int32_t c = 0; c < 2; ++c) {
            if (!w.on(c)) { return false; }
            const size_t bytes = size_t(plan.allocation(c));
            const bool made =
                host ? rc->buf[c].allocate_host(w.m.device[c], bytes, err, true)
                     : rc->buf[c].allocate(w.m.device[c], bytes, err);
            if (!made) {
                fprintf(stderr, "kv-tier-test: %s: %s\n", plan.name.c_str(), err.c_str());
                return false;
            }
            if (host) {
                std::memset(rc->buf[c].data(), 0, bytes);
            } else if (!cuda_ok(cudaMemset(rc->buf[c].data(), 0, bytes), "zero")) {
                return false;
            }
        }
        w.m.state.push_back(std::move(rc));
    }
    void * streams[2] = {w.streams[0], w.streams[1]};
    if (!w.tier.init(w.m, window, err, streams)) {
        fprintf(stderr, "kv-tier-test: %s\n", err.c_str());
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!w.on(c)) { return false; }
        for (int32_t k = 0; k < 2; ++k) {
            w.rk[c][k] = std::make_unique<dev<uint16_t>>(size_t(n_cells) * HD);
            w.rv[c][k] = std::make_unique<dev<uint16_t>>(size_t(n_cells) * HD);
            if (w.rk[c][k]->p == nullptr || w.rv[c][k]->p == nullptr ||
                !cuda_ok(cudaMemset(w.rk[c][k]->p, 0, size_t(n_cells) * HD * 2), "zero") ||
                !cuda_ok(cudaMemset(w.rv[c][k]->p, 0, size_t(n_cells) * HD * 2), "zero")) {
                return false;
            }
        }
    }
    const size_t sb = qs::summary_bytes(qs::blocks_complete(n_cells));
    for (int32_t c = 0; c < 2; ++c) {
        if (!w.on(c)) { return false; }
        for (int32_t k = 0; k < 2; ++k) {
            w.rkey[c][k] = std::make_unique<dev<uint16_t>>(size_t(n_cells) * KD);
            if (w.rkey[c][k]->p == nullptr ||
                !cuda_ok(cudaMemset(w.rkey[c][k]->p, 0, size_t(n_cells) * KD * 2), "zero")) {
                return false;
            }
            for (auto & a : w.sums[c][k]) {
                a = std::make_unique<dev<float>>(sb / 4);
                if (a->p == nullptr || !cuda_ok(cudaMemset(a->p, 0, sb), "zero")) {
                    return false;
                }
            }
        }
    }
    rng r{77};
    std::vector<float> qn(HD), kn(HD), ikn(KD);
    for (int32_t i = 0; i < HD; ++i) {
        qn[size_t(i)] = r.uniform(0.5f, 1.5f);
        kn[size_t(i)] = r.uniform(0.5f, 1.5f);
    }
    for (float & x : ikn) {
        x = r.uniform(0.5f, 1.5f);
    }
    return w.qfull.p && w.aws.p && w.part.p && w.ik.p && put(w.qn.p, qn, w.st) && put(w.kn.p, kn, w.st) &&
           put(w.ikn.p, ikn, w.st) &&
           cuda_ok(cudaMemsetAsync(w.aws.p, 0, at::workspace_bytes(), w.st), "workspace") &&
           cuda_ok(cudaStreamSynchronize(w.st), "setup");
}

struct launch_out {
    std::vector<float> out, scales;
    std::vector<int8_t> codes;
    std::vector<uint32_t> err;
};

bool read_out(world & w, int32_t rows, int32_t width, launch_out & o) {
    return get(o.out, w.out.p, size_t(rows) * width, w.st) && get(o.codes, w.codes.p, size_t(rows) * width, w.st) &&
           get(o.scales, w.scales.p, size_t(rows) * width / 16, w.st) && get(o.err, w.err.p, 2, w.st);
}

bool same(const launch_out & a, const launch_out & b) {
    return a.out.size() == b.out.size() && std::memcmp(a.out.data(), b.out.data(), a.out.size() * 4) == 0 &&
           a.codes == b.codes && std::memcmp(a.scales.data(), b.scales.data(), a.scales.size() * 4) == 0 &&
           a.err == b.err && a.err[0] == 0 && a.err[1] == 0;
}

void random_inputs(world & w, rng & r, int32_t rows, std::vector<float> & qfull, std::vector<float> & kv) {
    qfull.resize(size_t(rows) * NH * QS);
    for (float & x : qfull) {
        x = r.uniform(-1.5f, 1.5f);
    }
    kv.resize(size_t(rows) * HD * 2);
    for (float & x : kv) {
        x = r.uniform(-2.0f, 2.0f);
    }
    (void) w;
}

// ME6: blocks b0 .. b1 - 1 of two tiled summary buffers, bitwise
bool same_blocks(world & w, const float * x, const float * y, int32_t b0, int32_t b1) {
    if (b1 <= b0) {
        return true;
    }
    const size_t tile = size_t(qs::n_chunk) * qs::tile_blocks * 4; // floats a tile
    const size_t t0 = size_t(b0 / qs::tile_blocks), t1 = size_t((b1 - 1) / qs::tile_blocks) + 1;
    std::vector<float> a, b;
    if (!get(a, x + t0 * tile, (t1 - t0) * tile, w.st) || !get(b, y + t0 * tile, (t1 - t0) * tile, w.st)) {
        return false;
    }
    for (int32_t blk = b0; blk < b1; ++blk) {
        const size_t base = (size_t(blk / qs::tile_blocks) - t0) * tile + size_t(blk % qs::tile_blocks) * 4;
        for (int32_t e = 0; e < KD; ++e) {
            const size_t i = base + size_t(e / 4) * qs::tile_blocks * 4 + size_t(e % 4);
            if (std::memcmp(&a[i], &b[i], 4) != 0) {
                return false;
            }
        }
    }
    return true;
}

std::vector<float> random_ik(rng & r, int64_t rows) {
    std::vector<float> v(size_t(rows) * KD);
    for (float & x : v) {
        x = r.uniform(-2.0f, 2.0f);
    }
    return v;
}

// ME6: arm 2's update replayed from its captured graph (captured on first use: the window's base and the device word,
// so a replay after a window move reads the new window start from the word)
bool key_graph(world & w, int32_t c, int32_t k, int32_t rows, const qs::round_update_args & ua) {
    cudaGraphExec_t & ex = w.key_exec[c][k][rows];
    if (ex == nullptr) {
        cudaGraph_t g = nullptr;
        if (!cuda_ok(cudaStreamBeginCapture(w.st, cudaStreamCaptureModeThreadLocal), "capture")) {
            return false;
        }
        const bool launched = qs::summary_update(ua, w.st);
        if (!cuda_ok(cudaStreamEndCapture(w.st, &g), "capture") || !launched ||
            !cuda_ok(cudaGraphInstantiate(&ex, g, 0), "instantiate")) {
            if (g != nullptr) {
                cudaGraphDestroy(g);
            }
            return false;
        }
        cudaGraphDestroy(g);
        w.key_exec_lo[c][k][rows] = w.tier.lo();
    }
    ++w.key_graph_replays;
    w.key_graph_moved += w.tier.lo() != w.key_exec_lo[c][k][rows] ? 1 : 0;
    return cuda_ok(cudaGraphLaunch(ex, w.st), "replay");
}

// ME6: the round form of summary_update on card c's tracked key layer k (0 target, 1 draft) at the slot's n (the
// device depth word set by the caller): the new cells' keys written and the blocks they complete pooled, on the
// resident keys (arm 0), through the tier's window (arm 1, eager) and in graph mode with the window's word (arm 2, a
// replay of a captured graph: key_graph), each into its own summaries; the pooled blocks compared bitwise against
// arm 0. control: the window's copy of the first cell of the block the round completes is changed for the tier's arms
// (and restored), and the summaries must differ
bool key_round(world & w, rng & r, int32_t k, int32_t rows, bool control = false) {
    const int64_t N = w.n;
    const int32_t b0 = int32_t(N / qs::ratio), b1 = int32_t((N + rows) / qs::ratio);
    const bool ctl = control && b1 > b0 && N % qs::ratio != 0;
    bool ok = true;
    for (int32_t c = 0; c < 2 && ok; ++c) {
        if (!w.on(c)) { return false; }
        const int32_t a = key_layer(c, k);
        ok = put(w.ik.p, random_ik(r, rows), w.st);
        for (int32_t arm = 0; arm < (w.graph_arm ? 3 : 2) && ok; ++arm) {
            w.tier.set_graph(arm == 2);
            qs::round_update_args ua;
            ua.keys = arm == 0 ? w.rkey[c][k]->p : w.tier.keys(c, a);
            ua.shift = arm == 0 ? nullptr : w.tier.shift(c);
            ua.k_norm = w.ikn.p;
            ua.summaries = w.sums[c][k][arm]->p;
            ua.rope = qs::model_rope();
            ua.ik = w.ik.p;
            ua.depth = w.depth.p;
            ua.rows = rows;
            ua.max_cells = w.n_cells;
            w.tier.set_graph(false);
            uint16_t * cell = w.tier.keys(c, a) + (N / qs::ratio * qs::ratio) * KD; // eager: the window's cell
            uint16_t kept = 0;
            if (arm >= 1 && ctl) {
                const uint16_t bad = fn::quant::f32_to_bf16(64.0f);
                ok = cuda_ok(cudaMemcpyAsync(&kept, cell, 2, cudaMemcpyDeviceToHost, w.st), "control") &&
                     cuda_ok(cudaStreamSynchronize(w.st), "control") &&
                     cuda_ok(cudaMemcpyAsync(cell, &bad, 2, cudaMemcpyHostToDevice, w.st), "control");
            }
            ok = ok && (arm == 2 ? key_graph(w, c, k, rows, ua) : qs::summary_update(ua, w.st)) &&
                 cuda_ok(cudaStreamSynchronize(w.st), "key round");
            if (arm >= 1 && ctl) {
                ok = ok && cuda_ok(cudaMemcpyAsync(cell, &kept, 2, cudaMemcpyHostToDevice, w.st), "control") &&
                     cuda_ok(cudaStreamSynchronize(w.st), "control");
            }
        }
        const std::string at = " at " + std::to_string(N) + " B " + std::to_string(rows) + " card " +
                               std::to_string(c) + (k == 0 ? " target" : " draft") + " lo " +
                               std::to_string(w.tier.lo());
        if (control) {
            if (ctl) {
                w.ck.check(ok && !same_blocks(w, w.sums[c][k][0]->p, w.sums[c][k][1]->p, b0, b1),
                           "key control round (a changed window key detected)" + at);
                if (w.graph_arm) {
                    w.ckg.check(ok && !same_blocks(w, w.sums[c][k][0]->p, w.sums[c][k][2]->p, b0, b1),
                                "graph form: key control round" + at);
                }
                w.key_control_run = true;
            }
            continue;
        }
        w.ck.check(ok && same_blocks(w, w.sums[c][k][0]->p, w.sums[c][k][1]->p, b0, b1), "key round" + at);
        if (w.graph_arm) {
            w.ckg.check(ok && same_blocks(w, w.sums[c][k][0]->p, w.sums[c][k][2]->p, b0, b1),
                        "graph form: key round" + at);
        }
    }
    if (!control) {
        ++w.key_rounds;
        w.key_blocks += b1 - b0;
        w.key_moved_rounds += w.tier.lo() > 0 && b1 > b0 ? 1 : 0;
    }
    return ok;
}

// ME6: a prompt piece's keys (as the prompt path: at most one chunk, staged once, in two halves) for card c's tracked
// key layer k at cells start .. start + rows - 1: resident (qsp_keys and the cache-form pooling on the resident keys)
// and through the tier's key staging and write-back; the pooled blocks compared bitwise
bool key_prompt(world & w, rng & r, int32_t c, int32_t k, int64_t start, int64_t rows, std::string & err) {
    const int32_t a = key_layer(c, k);
    bool ok = w.tier.stage_keys(c, a, start, rows, w.st, err);
    const int64_t h0 = std::min<int64_t>(rows, qs::prompt_max_rows);
    const int64_t half[2] = {h0, rows - h0};
    int64_t at = start;
    for (int32_t h = 0; h < 2 && ok; ++h) {
        if (half[h] == 0) {
            continue;
        }
        const int32_t n = int32_t(half[h]), b0 = int32_t(at / qs::ratio), b1 = int32_t((at + n) / qs::ratio);
        ok = put(w.ik.p, random_ik(r, n), w.st);
        for (int32_t arm = 0; arm < 2 && ok; ++arm) {
            uint16_t * keys = arm == 0 ? w.rkey[c][k]->p : w.tier.stage_keys_ptr(c);
            qs::update_args ua = {keys, w.ikn.p, w.sums[c][k][arm]->p, b0, b1 - b0, qs::model_rope()};
            ok = qs::qsp_keys(n, int32_t(at), w.ik.p, keys, w.n_cells, w.st) &&
                 (b1 == b0 || qs::summary_update(ua, w.st)) &&
                 (arm == 0 || w.tier.write_back_keys(c, a, at, n, w.st, err)) &&
                 cuda_ok(cudaStreamSynchronize(w.st), "key prompt");
        }
        w.ck.check(ok && same_blocks(w, w.sums[c][k][0]->p, w.sums[c][k][1]->p, b0, b1),
                   "key prompt at " + std::to_string(at) + " rows " + std::to_string(n) + " card " +
                       std::to_string(c) + (k == 0 ? " target" : " draft"));
        w.key_prompt_blocks += b1 - b0;
        at += n;
    }
    return ok;
}

// a target verify round of rows at the slot's n: attend on the resident cache and through the tier, each card.
// control: the comparison's sensitivity; with the window above cell 0 (block 0 is always selected, gathered from host
// memory), cell 0's first K value in host memory is changed for the tier's launch and restored, and the outputs must
// differ
bool target_round(world & w, rng & r, int32_t rows, bool control = false) {
    std::string err;
    const int64_t N = w.n;
    if (!w.tier.decode(N, N + rows, N, w.st, err)) {
        fprintf(stderr, "kv-tier-test: %s\n", err.c_str());
        return false;
    }
    const int32_t lo = int32_t(w.tier.lo());
    std::vector<uint32_t> body;
    std::vector<int32_t> offs;
    make_body(r, int32_t(N), rows, lo, body, offs);
    std::vector<int32_t> pos(size_t(4) * rows);
    for (int32_t p = 0; p < 4; ++p) {
        for (int32_t t = 0; t < rows; ++t) {
            pos[size_t(p) * rows + t] = int32_t(N) + t;
        }
    }
    const int32_t d32 = int32_t(N);
    bool ok = put(w.body.p, body, w.st) && put(w.offsets.p, offs, w.st) && put(w.pos.p, pos, w.st) &&
              cuda_ok(cudaMemcpyAsync(w.depth.p, &d32, 4, cudaMemcpyHostToDevice, w.st), "depth");
    for (int32_t c = 0; c < 2 && ok; ++c) {
        if (!w.on(c)) { return false; }
        std::vector<float> qfull, kv;
        random_inputs(w, r, rows, qfull, kv);
        const std::vector<float> kk(kv.begin(), kv.begin() + size_t(rows) * HD), vv(kv.begin() + size_t(rows) * HD,
                                                                                   kv.end());
        launch_out o[3];
        for (int32_t arm = 0; arm < (w.graph_arm ? 3 : 2) && ok; ++arm) {
            ok = put(w.qfull.p, qfull, w.st) && put(w.kvk.p, kk, w.st) && put(w.kvv.p, vv, w.st) &&
                 cuda_ok(cudaMemsetAsync(w.out.p, 0xff, w.out.n * 4, w.st), "junk") &&
                 cuda_ok(cudaMemsetAsync(w.err.p, 0, 8, w.st), "error words");
            at::args a = {};
            a.qfull = w.qfull.p;
            a.kv_k = w.kvk.p;
            a.kv_v = w.kvv.p;
            a.q_norm = w.qn.p;
            a.k_norm = w.kn.p;
            a.body = w.body.p;
            a.offsets = w.offsets.p;
            a.pos = w.pos.p;
            a.k_cache = w.ref(c, 0, false);
            a.v_cache = w.ref(c, 0, true);
            a.depth = w.depth.p;
            a.max_cells = w.n_cells;
            a.theta_scale = at::rope_theta_scale();
            a.out = w.out.p;
            a.codes = w.codes.p;
            a.scales = w.scales.p;
            a.ws = at::carve(w.aws.p);
            uint16_t * const cell0 = w.tier.host_k(c, a_target);
            const uint16_t kept = *cell0;
            if (arm >= 1 && control) {
                *cell0 = fn::quant::f32_to_bf16(64.0f);
            }
            w.tier.set_graph(arm == 2);
            if (arm >= 1) {
                ok = w.tier.select(c, a_target, rows, w.body.p, w.offsets.p, w.err.p + 1, w.st, a.body, err);
                a.k_cache = w.tier.k(c, a_target);
                a.v_cache = w.tier.v(c, a_target);
                a.shift = w.tier.shift(c);
            }
            ok = ok && at::attend(rows, a, w.st) &&
                 cuda_ok(cudaMemcpyAsync(w.err.p, a.ws.error, 4, cudaMemcpyDeviceToDevice, w.st), "error") &&
                 read_out(w, rows, at::out_width, o[arm]);
            *cell0 = kept; // read_out synchronized the stream
            w.tier.set_graph(false);
        }
        if (control) {
            w.ck.check(ok && lo > 0 && !same(o[0], o[1]), "control round (a changed host cell detected) at " +
                                                                std::to_string(N) + " card " + std::to_string(c));
            if (w.graph_arm) {
                w.ckg.check(ok && lo > 0 && !same(o[0], o[2]), "graph form: control round at " + std::to_string(N) +
                                                                   " card " + std::to_string(c));
            }
            continue;
        }
        w.ck.check(ok && same(o[0], o[1]), "target round at " + std::to_string(N) + " B " + std::to_string(rows) +
                                               " card " + std::to_string(c) + " lo " + std::to_string(lo));
        if (w.graph_arm) {
            w.ckg.check(ok && same(o[0], o[2]), "graph form: target round at " + std::to_string(N) + " B " +
                                                    std::to_string(rows) + " card " + std::to_string(c) + " lo " +
                                                    std::to_string(lo));
        }
    }
    ok = ok && key_round(w, r, 0, rows, control); // ME6: the target's index keys of the same round
    if (lo > 0) {
        ++w.gathered_rounds;
    }
    if (w.graph_arm) {
        ++w.graph_rounds;
        w.graph_gathered_rounds += lo > 0 ? 1 : 0;
    }
    return ok;
}

// the draft's prepare (rows at row0, qfull null: a catch-up) and, for a step, attend; on the resident cache (arm 0)
// and through the tier (arm 1; arm 2: in graph mode, with the shift); step 0 selects through the tier, the chain's
// later steps reuse its body
bool draft_launch(world & w, int32_t c, int32_t arm, int32_t rows, int32_t row0, bool step,
                  launch_out & o, std::string & err) {
    w.tier.set_graph(arm == 2);
    struct off {
        md::kv_tier & t;
        ~off() { t.set_graph(false); }
    } restore{w.tier};
    const int64_t * shift = arm == 2 ? w.tier.shift(c) : nullptr;
    dr::prepare_args p = {};
    p.qfull = step ? w.qfull.p : nullptr;
    p.kv_k = w.kvk.p;
    p.kv_v = w.kvv.p;
    p.q_norm = w.qn.p;
    p.k_norm = w.kn.p;
    p.n_before = w.depth.p;
    p.pos = w.pos.p;
    p.stride = w.stride.p;
    p.row0 = row0;
    p.theta_scale = at::rope_theta_scale();
    p.k_cache = arm == 0 ? w.ref(c, 1, false) : w.tier.k(c, a_draft);
    p.v_cache = arm == 0 ? w.ref(c, 1, true) : w.tier.v(c, a_draft);
    p.max_cells = w.n_cells;
    p.q16 = w.q16.p;
    p.err = w.err.p;
    bool ok = cuda_ok(cudaMemsetAsync(w.err.p, 0, 8, w.st), "error words") && dr::datc_prepare(rows, p, w.st, shift);
    if (!step) {
        o.err.clear();
        return ok && get(o.err, w.err.p, 2, w.st);
    }
    dr::attend_args a = {};
    a.q16 = w.q16.p;
    a.qfull = w.qfull.p;
    a.k_cache = p.k_cache;
    a.v_cache = p.v_cache;
    a.n_before = w.depth.p;
    a.row0 = row0;
    a.max_cells = w.n_cells;
    a.part = w.part.p;
    a.out = w.out.p;
    a.codes = w.codes.p;
    a.scales = w.scales.p;
    a.err = w.err.p;
    a.body = w.body.p;
    if (arm >= 1) {
        if (row0 == 0) {
            ok = ok && w.tier.select(c, a_draft, 1, w.body.p, w.offsets.p, w.err.p + 1, w.st, a.body, err);
        } else {
            a.body = w.tier.draft_body(c);
        }
    }
    return ok && cuda_ok(cudaMemsetAsync(w.out.p, 0xff, w.out.n * 4, w.st), "junk") &&
           dr::datc_attend(a, w.st, shift) &&
           read_out(w, 1, dr::out_width, o);
}

// a draft chain of d steps at the slot's n (the target's pending token at n), then nothing kept
bool draft_chain(world & w, rng & r, int32_t d) {
    std::string err;
    const int64_t N = w.n;
    if (N < 4) {
        return true;
    }
    if (!w.tier.decode(N, N + d, N, w.st, err)) {
        fprintf(stderr, "kv-tier-test: %s\n", err.c_str());
        return false;
    }
    const int32_t lo = int32_t(w.tier.lo());
    // the select at depth N - 1 with the query at N: K = min(N / 4, 512) blocks of the N / 4 whole ones
    const int32_t n_vis = int32_t(N) / qs::ratio;
    std::vector<int32_t> shared;
    for (int32_t i = 0; i < 200; ++i) {
        shared.push_back(int32_t(r.below(n_vis)));
    }
    const std::vector<int32_t> b = pick_blocks(r, n_vis, lo, shared);
    // select's q_ahead form: the row at depth N - 1, K blocks, then its tail cells 4 (N / 4) .. N - 1
    std::vector<uint32_t> body = {uint32_t(b.size()), uint32_t(N % qs::ratio)};
    for (int32_t x : b) {
        body.push_back(uint32_t(x));
    }
    for (int64_t c = qs::ratio * n_vis; c < N; ++c) {
        body.push_back(uint32_t(c));
    }
    const std::vector<int32_t> offs = {0, int32_t(body.size()) * 4};
    std::vector<int32_t> pos(size_t(4) * d);
    for (int32_t p = 0; p < 4; ++p) {
        for (int32_t t = 0; t < d; ++t) {
            pos[size_t(p) * d + t] = int32_t(N) + t;
        }
    }
    const int32_t d32 = int32_t(N);
    bool ok = put(w.body.p, body, w.st) && put(w.offsets.p, offs, w.st) && put(w.pos.p, pos, w.st) &&
              cuda_ok(cudaMemcpyAsync(w.depth.p, &d32, 4, cudaMemcpyHostToDevice, w.st), "depth") &&
              cuda_ok(cudaMemcpyAsync(w.stride.p, &d, 4, cudaMemcpyHostToDevice, w.st), "stride");
    for (int32_t c = 0; c < 2 && ok; ++c) {
        if (!w.on(c)) { return false; }
        // both arms run the whole chain, the resident one first: the inputs of each step again for the tier
        std::vector<std::vector<float>> qf(static_cast<size_t>(d)), kv(static_cast<size_t>(d));
        for (int32_t j = 0; j < d; ++j) {
            random_inputs(w, r, 1, qf[size_t(j)], kv[size_t(j)]);
        }
        std::vector<launch_out> o[3];
        for (int32_t arm = 0; arm < (w.graph_arm ? 3 : 2) && ok; ++arm) {
            o[arm].resize(size_t(d));
            for (int32_t j = 0; j < d && ok; ++j) {
                const std::vector<float> & k2 = kv[size_t(j)];
                ok = put(w.qfull.p, qf[size_t(j)], w.st) &&
                     put(w.kvk.p, std::vector<float>(k2.begin(), k2.begin() + HD), w.st) &&
                     put(w.kvv.p, std::vector<float>(k2.begin() + HD, k2.end()), w.st) &&
                     draft_launch(w, c, arm, 1, j, true, o[arm][size_t(j)], err);
            }
        }
        for (int32_t j = 0; j < d && ok; ++j) {
            w.ck.check(same(o[0][size_t(j)], o[1][size_t(j)]), "draft step " + std::to_string(j) + " at " +
                                                                   std::to_string(N) + " card " +
                                                                   std::to_string(c) + " lo " + std::to_string(lo));
            if (w.graph_arm) {
                w.ckg.check(same(o[0][size_t(j)], o[2][size_t(j)]), "graph form: draft step " + std::to_string(j) +
                                                                        " at " + std::to_string(N) + " card " +
                                                                        std::to_string(c) + " lo " +
                                                                        std::to_string(lo));
            }
        }
    }
    if (!ok && !err.empty()) {
        fprintf(stderr, "kv-tier-test: %s\n", err.c_str());
    }
    return ok;
}

// the draft's catch-up of rows at the slot's n (K and V only), both arms
bool catchup(world & w, rng & r, int32_t rows) {
    std::string err;
    const int64_t N = w.n;
    if (!w.tier.decode(N, N + rows, N, w.st, err)) {
        fprintf(stderr, "kv-tier-test: %s\n", err.c_str());
        return false;
    }
    std::vector<int32_t> pos(size_t(4) * rows);
    for (int32_t p = 0; p < 4; ++p) {
        for (int32_t t = 0; t < rows; ++t) {
            pos[size_t(p) * rows + t] = int32_t(N) + t;
        }
    }
    const int32_t d32 = int32_t(N);
    bool ok = put(w.pos.p, pos, w.st) &&
              cuda_ok(cudaMemcpyAsync(w.depth.p, &d32, 4, cudaMemcpyHostToDevice, w.st), "depth") &&
              cuda_ok(cudaMemcpyAsync(w.stride.p, &rows, 4, cudaMemcpyHostToDevice, w.st), "stride");
    for (int32_t c = 0; c < 2 && ok; ++c) {
        if (!w.on(c)) { return false; }
        std::vector<float> qf, kv;
        random_inputs(w, r, rows, qf, kv);
        ok = put(w.kvk.p, std::vector<float>(kv.begin(), kv.begin() + size_t(rows) * HD), w.st) &&
             put(w.kvv.p, std::vector<float>(kv.begin() + size_t(rows) * HD, kv.end()), w.st);
        launch_out o[3];
        for (int32_t arm = 0; arm < (w.graph_arm ? 3 : 2) && ok; ++arm) {
            ok = draft_launch(w, c, arm, rows, 0, false, o[arm], err);
        }
        w.ck.check(ok && o[0].err == o[1].err && o[0].err[0] == 0, "catch-up at " + std::to_string(N));
        if (w.graph_arm) {
            w.ckg.check(ok && o[0].err == o[2].err, "graph form: catch-up at " + std::to_string(N));
        }
    }
    return ok && key_round(w, r, 1, rows); // ME6: the draft index's keys of the catch-up
}

// a prompt chunk of n rows at the slot's n in two halves: the target layer through the staging (checked against the
// resident cells first), the draft's K and V straight into host memory
bool prompt_chunk(world & w, rng & r, int64_t rows) {
    std::string err;
    const int64_t N = w.n;
    if (!w.tier.prompt(w.st, err)) {
        fprintf(stderr, "kv-tier-test: %s\n", err.c_str());
        return false;
    }
    bool ok = true;
    const int64_t half[2] = {(rows + 1) / 2, rows / 2};
    for (int32_t c = 0; c < 2 && ok; ++c) {
        if (!w.on(c)) { return false; }
        ok = w.tier.stage(c, a_target, N, w.st, err);
        std::vector<uint16_t> sk, rk, sv, rv;
        const size_t cells = size_t(N) * HD;
        ok = ok && get(sk, w.tier.stage_k(c), cells, w.st) && get(rk, w.ref(c, 0, false), cells, w.st) &&
             get(sv, w.tier.stage_v(c), cells, w.st) && get(rv, w.ref(c, 0, true), cells, w.st);
        w.ck.check(ok && sk == rk && sv == rv, "staged history below " + std::to_string(N) + " card " +
                                                   std::to_string(c));
        int64_t at = N;
        for (int32_t h = 0; h < 2 && ok; ++h) {
            if (half[h] == 0) {
                continue;
            }
            const std::vector<uint16_t> k = random_cells(r, half[h], false), v = random_cells(r, half[h], false);
            const size_t b = size_t(half[h]) * HD * 2;
            ok = put(w.tier.stage_k(c) + at * HD, k, w.st) && put(w.tier.stage_v(c) + at * HD, v, w.st) &&
                 put(w.ref(c, 0, false) + at * HD, k, w.st) && put(w.ref(c, 0, true) + at * HD, v, w.st) &&
                 w.tier.write_back(c, a_target, at, half[h], w.st, err) &&
                 cuda_ok(cudaStreamSynchronize(w.st), "half");
            // the draft's prompt K and V: written straight into host memory (the draft prompt's kernel does so)
            const std::vector<uint16_t> dk = random_cells(r, half[h], true), dv = random_cells(r, half[h], true);
            std::memcpy(w.tier.host_k(c, a_draft) + at * HD, dk.data(), b);
            std::memcpy(w.tier.host_v(c, a_draft) + at * HD, dv.data(), b);
            ok = ok && put(w.ref(c, 1, false) + at * HD, dk, w.st) && put(w.ref(c, 1, true) + at * HD, dv, w.st) &&
                 cuda_ok(cudaStreamSynchronize(w.st), "half");
            at += half[h];
        }
        // ME6: the index keys, target and draft, in the prompt path's chunks
        for (int64_t p0 = N; p0 < N + rows && ok; p0 += md::prefill_stage_rows) {
            const int64_t n = std::min<int64_t>(N + rows - p0, md::prefill_stage_rows);
            ok = key_prompt(w, r, c, 0, p0, n, err) && key_prompt(w, r, c, 1, p0, n, err);
        }
    }
    if (!ok && !err.empty()) {
        fprintf(stderr, "kv-tier-test: %s\n", err.c_str());
    }
    w.n = N + rows;
    return ok;
}

// a failed comparison's first and last differing cells (width values a cell), printed for diagnosis
void print_diff(const char * what, const uint16_t * x, const uint16_t * y, int64_t cells, int32_t width) {
    int64_t first = -1, last = -1, n = 0;
    for (int64_t i = 0; i < cells; ++i) {
        if (std::memcmp(x + i * width, y + i * width, size_t(width) * 2) != 0) {
            first = first < 0 ? i : first;
            last = i;
            ++n;
        }
    }
    if (n > 0) {
        fprintf(stderr, "kv-tier-test:   %s: %lld cells differ, %lld .. %lld\n", what, (long long)n, (long long)first,
                (long long)last);
    }
}

// the host history after a flush against the resident cells below the slot's n, both tracked layers and cards
bool state_check(world & w, const std::string & when) {
    std::string err;
    if (!w.tier.flush(w.st, err)) {
        fprintf(stderr, "kv-tier-test: %s\n", err.c_str());
        return false;
    }
    bool ok = true;
    for (int32_t c = 0; c < 2 && ok; ++c) {
        if (!w.on(c)) { return false; }
        for (int32_t k = 0; k < 2 && ok; ++k) {
            const int32_t a = k == 0 ? a_target : a_draft;
            std::vector<uint16_t> rk, rv;
            const size_t cells = size_t(w.n) * HD;
            ok = get(rk, w.ref(c, k, false), cells, w.st) && get(rv, w.ref(c, k, true), cells, w.st);
            const bool eq = ok && std::memcmp(rk.data(), w.tier.host_k(c, a), rk.size() * 2) == 0 &&
                            std::memcmp(rv.data(), w.tier.host_v(c, a), rv.size() * 2) == 0;
            w.ck.check(eq, "host history below " + std::to_string(w.n) + " (" + when + ") card " + std::to_string(c) +
                               (k == 0 ? " target" : " draft"));
            if (ok && !eq) {
                print_diff("K", rk.data(), w.tier.host_k(c, a), w.n, HD);
                print_diff("V", rv.data(), w.tier.host_v(c, a), w.n, HD);
            }
            // ME6: the tracked key layer's host keys
            std::vector<uint16_t> rkey;
            ok = ok && get(rkey, w.rkey[c][k]->p, size_t(w.n) * KD, w.st);
            const uint16_t * hk = w.tier.host_keys(c, key_layer(c, k));
            const bool keq = ok && hk != nullptr && std::memcmp(rkey.data(), hk, rkey.size() * 2) == 0;
            w.ck.check(keq, "host keys below " + std::to_string(w.n) + " (" + when + ") card " + std::to_string(c) +
                                (k == 0 ? " target" : " draft"));
            if (ok && hk != nullptr && !keq) {
                print_diff("keys", rkey.data(), hk, w.n, KD);
            }
        }
    }
    return ok;
}

// one speculative step: a verify of d + 1 rows, a of them accepted, the catch-up, a draft chain
bool spec_step(world & w, rng & r) {
    const int32_t d = 1 + int32_t(r.below(4));
    const int32_t B = d + 1;
    if (w.n + B + 4 > w.n_cells) {
        return true;
    }
    if (!target_round(w, r, B)) {
        return false;
    }
    const int32_t a = int32_t(r.below(B)); // rows a + 1 kept, the rest rejected (rewritten by the next round)
    if (!catchup(w, r, a + 1)) {
        return false;
    }
    // the kept rows' cells: the target's as the round wrote them (its K/V in both arms), the draft's as the catch-up
    w.n += a + 1;
    return draft_chain(w, r, d);
}

} // namespace

int fnfl_kv_tier_test(int device, const std::string & mode) {
    const bool two_devices = mode == "tp" || mode == "tp-san";
    const bool san = mode == "san" || mode == "tp-san";
    if (!san && mode != "all" && mode != "tp") {
        fprintf(stderr, "kv-tier-test: mode all, san, tp or tp-san\n");
        return 2;
    }
    if (!cuda_ok(cudaSetDevice(device), "device")) { return 1; }
    world w;
    const int32_t n_cells = san ? 32768 : qs::max_cells, window = san ? 1024 : 4096;
    if (!build(w, device, n_cells, window, two_devices)) {
        return 1;
    }
    printf("kv-tier-test: device %d, %d cells, window %d cells (tracked: QSA layer %d and the draft, both cards)\n",
           device, n_cells, window, a_target);
    printf("kv-tier-test: physical devices %d, %d; both eager and graph forms in every case\n",
           w.m.device[0], w.m.device[1]);
    rng r{20261017};
    bool ok = true;
    w.graph_arm = true; // MEKV: every case runs eager and graph-window forms
    // 1. from an empty slot: a short prompt, decode inside the first window (no gathers), then past it
    ok = ok && prompt_chunk(w, r, 37) && state_check(w, "first prompt");
    for (int32_t i = 0; i < 40 && ok; ++i) {
        ok = spec_step(w, r);
    }
    ok = ok && state_check(w, "early decode");
    // 2. prompts to several windows' depth, then decode across window moves (rebuilds) with gathers
    while (ok && w.n < 5 * window) {
        ok = prompt_chunk(w, r, 1 + r.below(window));
    }
    w.graph_arm = true; // ME3: the window-move case's graph form as well
    const int64_t rebuilds0 = w.tier.count().rebuilds;
    w.n -= w.n % qs::ratio == 0 ? 1 : 0; // ME6: the key control needs a block the round completes from older cells
    ok = ok && target_round(w, r, 3, true);
    for (int32_t i = 0; i < (san ? 400 : 1600) && ok; ++i) {
        ok = spec_step(w, r);
        if (ok && i % 200 == 199) {
            ok = state_check(w, "decode " + std::to_string(i));
        }
    }
    w.graph_arm = true; // MEKV: cover graph form in restores, limit and corrupt-cell cases too
    const int64_t graph_rebuilds = w.tier.count().rebuilds - rebuilds0;
    printf("kv-tier-test: the window-move case's graph form: %lld target rounds (%lld with gathers), %lld window "
           "rebuilds; %lld checks against the resident launches, %lld failed\n",
           (long long)w.graph_rounds, (long long)w.graph_gathered_rounds, (long long)graph_rebuilds,
           (long long)w.ckg.n, (long long)w.ckg.failed);
    // 3. checkpoint restores: inside the window, then far below it, each followed by decode; then a divergent prefix
    // (a restore and a different prompt continuation)
    for (int32_t k = 0; k < 6 && ok; ++k) {
        const int64_t lo = w.tier.lo();
        const int64_t to = k % 2 == 0 ? std::max<int64_t>(lo + 1, w.n - 7) : 1 + r.below(std::max<int64_t>(lo, 2));
        w.n = std::min(w.n, to);
        for (int32_t i = 0; i < 30 && ok; ++i) {
            ok = spec_step(w, r);
        }
        ok = ok && state_check(w, "restore " + std::to_string(k));
        if (ok && k == 3) {
            w.n = std::max<int64_t>(4, w.n / 3);
            ok = prompt_chunk(w, r, 1 + r.below(2 * window)) && state_check(w, "divergent prefix");
            for (int32_t i = 0; i < 30 && ok; ++i) {
                ok = spec_step(w, r);
            }
        }
    }
    // 4. the 262,144-cell limit: prompts to its last rows, decode until no round fits
    while (ok && w.n < n_cells - 64) {
        ok = prompt_chunk(w, r, std::min<int64_t>(n_cells - 64 - w.n, 1 + r.below(san ? 4096 : 32768)));
    }
    for (int32_t i = 0; i < 60 && ok && w.n + 9 <= n_cells; ++i) {
        ok = spec_step(w, r);
    }
    const int64_t tail_n = w.n, tail_lo = w.tier.lo(), tail_rebuilds = w.tier.count().rebuilds;
    while (ok && w.n + 5 <= n_cells) {
        // ME6 (Shell, lab 7f7f7239): the draft caught up on the kept row before the common length advances, as in
        // spec_step, so state_check's draft history below w.n is history the draft wrote
        ok = target_round(w, r, int32_t(std::min<int64_t>(5, n_cells - w.n))) && catchup(w, r, 1);
        w.n += 1;
    }
    printf("kv-tier-test: limit tail: target rounds from %lld to %lld, window %lld -> %lld, %lld rebuilds\n",
           (long long)tail_n, (long long)w.n, (long long)tail_lo, (long long)w.tier.lo(),
           (long long)(w.tier.count().rebuilds - tail_rebuilds));
    ok = ok && state_check(w, "limit");
    std::string err;
    ok = ok && w.tier.read_counters(err);
    printf("kv-tier-test: slot at %lld cells; %lld target rounds with gathers; %s\n", (long long)w.n,
           (long long)w.gathered_rounds, w.tier.counters_text().c_str());
    printf("kv-tier-test: index keys (ME6): %lld rounds (%lld pooling with the window above cell 0), %lld blocks "
           "pooled in rounds, %lld in prompts; key control %s; captured key graphs replayed %lld times, %lld at "
           "another window start than their capture's\n",
           (long long)w.key_rounds, (long long)w.key_moved_rounds, (long long)w.key_blocks,
           (long long)w.key_prompt_blocks, w.key_control_run ? "run" : "NOT RUN", (long long)w.key_graph_replays,
           (long long)w.key_graph_moved);
    printf("kv-tier-test: %lld checks, %lld failed\n", (long long)w.ck.n, (long long)w.ck.failed);
    const bool pass = ok && w.ck.failed == 0 && w.gathered_rounds > 0 && w.tier.count().rebuilds > 2 &&
                      w.ckg.n > 0 && w.ckg.failed == 0 && w.graph_gathered_rounds > 0 && w.key_control_run &&
                      w.key_moved_rounds > 0 && w.key_graph_moved > 0;
    printf("kv-tier-test: %s\n", pass ? "PASS" : "FAILED");
    for (int32_t c = 0; c < 2; ++c) {
        cuda_ok(cudaSetDevice(w.m.device[c]), "destroy graph device");
        for (auto & layer : w.key_exec[c]) {
            for (cudaGraphExec_t ex : layer) {
                if (ex != nullptr) {
                    cudaGraphExecDestroy(ex);
                }
            }
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        cuda_ok(cudaSetDevice(w.m.device[c]), "destroy stream device");
        cuda_ok(cudaStreamDestroy(w.streams[c]), "destroy stream");
    }
    return pass ? 0 : 1;
}
