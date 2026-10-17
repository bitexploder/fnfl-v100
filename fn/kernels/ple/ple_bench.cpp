#include "fn/kernels/ple/ple_bench.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_cold.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/embed/embed.h"
#include "fn/kernels/ple/ple.h"
#include "fn/model/model.h"
#include "fn/runtime/page_cache.h"
#include "fn/runtime/ple_host.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace pl = fn::ple;
namespace rt = fn::runtime;
namespace x = fn::xchg;

constexpr int32_t D = 2560;
constexpr int32_t F = 1280;
constexpr int32_t CH = pl::n_channel;
constexpr int32_t V = fn::embed::n_vocab;
constexpr int32_t n_unit = 64;     // launches in a warm graph (products: two rotations of its copies)
constexpr int32_t n_warm = 3;      // untimed replays (warm) and trials (cold)
constexpr int32_t n_timed = 20;    // timed replays (warm) and trials (cold)
constexpr int32_t n_cold_rep = 20; // launches in a cold graph (XP1's rep)
constexpr int32_t n_rounds = 3;    // rounds in a combined graph: one per X-PLE-STATS slot
constexpr size_t min_cold = size_t(200) << 20;
constexpr int64_t flush_used = 12ll << 20; // > 6 MiB of L2 (XP1)
constexpr double ramp_us = 1.949;          // HX12 card 0 at 1,380 MHz
constexpr double dram_gbps = 1021.6;       // HX12
constexpr double tail_single_us = 0.31;    // HX12: a single-pass launch's tail
constexpr int64_t key_bytes = int64_t(CH) * D * 2;
constexpr int64_t value_bytes = int64_t(F) * D * 2;
constexpr int64_t pw_bytes = key_bytes + value_bytes; // 32,768,000

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "ple-bench: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    char buf[64];
    std::tm tm = {};
    gmtime_r(&t, &tm);
    const size_t n = strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    snprintf(buf + n, sizeof(buf) - n, ".%03dZ", static_cast<int>(ms));
    return buf;
}

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    float uni(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() >> 40) * (1.0f / 16777216.0f); }
};

// one device allocation, freed on its device
struct dev_mem {
    int32_t dev = 0;
    uint8_t * p = nullptr;
    size_t n = 0;
    dev_mem(int32_t d, size_t bytes) : dev(d), n(bytes) {
        if (cudaSetDevice(d) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&p), std::max<size_t>(n, 16)) != cudaSuccess) {
            p = nullptr;
        }
    }
    ~dev_mem() {
        if (p != nullptr) {
            cudaSetDevice(dev);
            cudaFree(p);
        }
    }
    dev_mem(const dev_mem &) = delete;
    dev_mem & operator=(const dev_mem &) = delete;
    template <typename T> T * as() const { return reinterpret_cast<T *>(p); }
    bool put(const void * src, size_t bytes, size_t off = 0) const {
        return p != nullptr && cudaSetDevice(dev) == cudaSuccess &&
               cudaMemcpy(p + off, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
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

// one card's bench buffers
struct card {
    int32_t dev = 0;
    int32_t r = 0;
    cudaStream_t st = nullptr;
    int32_t n_copies = 0;
    std::unique_ptr<dev_mem> pw, norms, conv, e, hres, hpost, hx, h_out, hist, cur, ws, table, grids, tok, ehres,
        ehpost, ehx, flush, evict;

    bool make(int32_t device, int32_t card_index) {
        dev = device;
        r = card_index;
        n_copies = static_cast<int32_t>((min_cold + pw_bytes - 1) / pw_bytes);
        auto mk = [&](std::unique_ptr<dev_mem> & m, size_t bytes) {
            m = std::make_unique<dev_mem>(dev, bytes);
            return m->p != nullptr;
        };
        if (!mk(pw, size_t(pw_bytes) * n_copies) || !mk(norms, size_t(3) * CH * 4) || !mk(conv, size_t(CH) * 8) ||
            !mk(e, size_t(4) * D * 4) || !mk(hres, size_t(4) * CH * 4) || !mk(hpost, 64) ||
            !mk(hx, size_t(4) * F * 4) || !mk(h_out, size_t(4) * CH * 4) || !mk(hist, size_t(pl::hist_bytes)) ||
            !mk(cur, 4) || !mk(ws, pl::workspace_bytes()) || !mk(table, size_t(V) * fn::embed::row_bytes) ||
            !mk(grids, fn::dense::grid_bytes) || !mk(tok, size_t(n_unit) * 16) || !mk(ehres, size_t(4) * CH * 4) ||
            !mk(ehpost, 64) || !mk(ehx, size_t(4) * F * 4) || !mk(flush, size_t(flush_used)) || !mk(evict, 16)) {
            fprintf(stderr, "ple-bench: allocation failed on device %d\n", dev);
            return false;
        }
        rng g(0xbe7c4ull + uint64_t(r));
        // weights: small BF16 values (timing does not depend on them); one copy filled, the rest copied
        std::vector<uint16_t> w(size_t(pw_bytes / 2));
        for (uint16_t & v : w) {
            const float f = g.uni(-0.05f, 0.05f);
            uint32_t u = 0;
            std::memcpy(&u, &f, 4);
            v = static_cast<uint16_t>(u >> 16);
        }
        bool ok = pw->put(w.data(), size_t(pw_bytes));
        for (int32_t c = 1; ok && c < n_copies; ++c) {
            ok = cuda_ok(cudaMemcpy(pw->p + size_t(c) * pw_bytes, pw->p, size_t(pw_bytes), cudaMemcpyDeviceToDevice),
                         "copy weights");
        }
        std::vector<float> nv(size_t(3) * CH), act(size_t(4) * CH);
        for (float & v : nv) {
            v = g.uni(0.5f, 1.5f);
        }
        for (float & v : act) {
            v = g.uni(-1.0f, 1.0f);
        }
        std::vector<uint16_t> cv(size_t(CH) * 4, 0x2e66u); // 0.1 in F16
        std::vector<uint8_t> tb(size_t(V) * fn::embed::row_bytes);
        for (size_t i = 0; i < tb.size(); i += 8) {
            const uint64_t z = g.next();
            std::memcpy(&tb[i], &z, std::min<size_t>(8, tb.size() - i));
        }
        std::vector<int32_t> toks(size_t(n_unit) * 4);
        for (int32_t & t : toks) {
            t = static_cast<int32_t>(g.next() % uint64_t(V));
        }
        std::vector<uint8_t> gb;
        ok = ok && fn::dense::grid_buffer(gb) && norms->put(nv.data(), nv.size() * 4) &&
             conv->put(cv.data(), cv.size() * 2) && e->put(act.data(), size_t(4) * D * 4) &&
             hres->put(act.data(), size_t(4) * CH * 4) && hpost->put(act.data(), 64) &&
             hx->put(act.data(), size_t(4) * F * 4) && table->put(tb.data(), tb.size()) &&
             grids->put(gb.data(), gb.size()) && tok->put(toks.data(), toks.size() * 4) &&
             cuda_ok(cudaMemset(hist->p, 0, hist->n), "hist") && cuda_ok(cudaMemset(cur->p, 0, 4), "cur") &&
             pl::prepare(ws->p) && cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        return ok;
    }

    ~card() {
        if (st != nullptr) {
            cudaSetDevice(dev);
            cudaStreamDestroy(st);
        }
    }

    pl::step_args args(const x::rings & rg, int32_t copy, int32_t slot, int32_t reset_slot) const {
        pl::step_args a;
        a.w.key = pw->p + size_t(copy) * pw_bytes;
        a.w.value = pw->p + size_t(copy) * pw_bytes + key_bytes;
        a.w.norm_key = norms->as<float>();
        a.w.norm_query = norms->as<float>() + CH;
        a.w.norm_conv = norms->as<float>() + int64_t{2} * CH;
        a.w.conv = conv->as<uint16_t>();
        a.e = e->as<float>();
        a.hres = hres->as<float>();
        a.hpost = hpost->as<float>();
        a.hx = hx->as<float>();
        a.h_out = h_out->as<float>();
        a.hist = hist->as<float>();
        // cudaMalloc storage, aligned for a word; slot 0 throughout: the bench never commits
        a.cur = reinterpret_cast<const uint32_t *>(cur->p);
        a.ws = ws->p;
        a.stats_peer = rg.slot(1 - r, x::msg::ple, slot);
        a.stats_slot = rg.slot(r, x::msg::ple, slot);
        a.stats_reset = reset_slot >= 0 ? rg.slot(r, x::msg::ple, reset_slot) : nullptr;
        a.err = rg.err(r);
        a.card = r;
        return a;
    }

    fn::embed::args eargs(const x::rings & rg, int32_t unit) const {
        fn::embed::args a;
        a.table = table->p;
        a.grids = grids->p;
        a.tok = tok->as<int32_t>() + int64_t{4} * (unit % n_unit);
        a.hres = ehres->as<float>();
        a.hpost = ehpost->as<float>();
        a.hx = ehx->as<float>();
        a.err = rg.err(r);
        return a;
    }

    bool cold() {
        return fn::dense::cold_flush(flush->p, flush_used, st) && fn::dense::cold_icache(evict->as<float>(), st);
    }
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

bool capture(cudaStream_t st, int32_t n, const std::function<bool(int32_t)> & unit, graph_exec & ge) {
    cudaGraph_t graph = nullptr;
    if (!cuda_ok(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture")) {
        return false;
    }
    bool ok = true;
    for (int32_t i = 0; i < n && ok; ++i) {
        ok = unit(i);
    }
    ok = cuda_ok(cudaStreamEndCapture(st, &graph), "end capture") && ok;
    ok = ok && cuda_ok(cudaGraphInstantiate(&ge.x, graph, 0), "instantiate");
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    return ok;
}

bool replay_us(cudaStream_t st, const graph_exec & ge, event_pair & ev, double & us) {
    float ms = 0.0f;
    const bool ok = cuda_ok(cudaEventRecord(ev.e0, st), "record") && cuda_ok(cudaGraphLaunch(ge.x, st), "replay") &&
                    cuda_ok(cudaEventRecord(ev.e1, st), "record") && cuda_ok(cudaEventSynchronize(ev.e1), "sync") &&
                    cuda_ok(cudaEventElapsedTime(&ms, ev.e0, ev.e1), "elapsed");
    us = 1e3 * double(ms);
    return ok;
}

bool time_warm(card & c, int32_t units, const std::function<bool(int32_t)> & unit, stats & s) {
    graph_exec ge;
    event_pair ev;
    bool ok = capture(c.st, units, unit, ge) && ev.make();
    double us = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = replay_us(c.st, ge, ev, us);
    }
    s.us.clear();
    s.t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        ok = replay_us(c.st, ge, ev, us);
        s.us.push_back(us / units);
    }
    s.t1 = utc_now();
    if (ok) {
        s.finish();
    }
    return ok;
}

// R17 (XP1's alone_timer): graph A = n x [flush, eviction, launch], graph B = n x [flush, eviction]; (A - B) / n
bool time_cold(card & c, const std::function<bool(int32_t)> & unit, stats & s) {
    graph_exec ga, gb;
    event_pair ev;
    bool ok = capture(
                  c.st, n_cold_rep, [&](int32_t u) { return c.cold() && unit(u); }, ga) &&
              capture(c.st, n_cold_rep, [&](int32_t) { return c.cold(); }, gb) && ev.make();
    double ta = 0.0, tb = 0.0;
    for (int32_t i = 0; i < n_warm && ok; ++i) {
        ok = replay_us(c.st, ga, ev, ta) && replay_us(c.st, gb, ev, tb);
    }
    s.us.clear();
    s.t0 = utc_now();
    for (int32_t i = 0; i < n_timed && ok; ++i) {
        ok = replay_us(c.st, gb, ev, tb) && replay_us(c.st, ga, ev, ta);
        s.us.push_back((ta - tb) / n_cold_rep);
    }
    s.t1 = utc_now();
    if (ok) {
        s.finish();
    }
    return ok;
}

void print_us(const char * key, const std::vector<double> & v) {
    printf(", \"%s\": [", key);
    for (size_t i = 0; i < v.size(); ++i) {
        printf("%s%.3f", i ? ", " : "", v[i]);
    }
    printf("]");
}

double floor_us(double bytes, bool tail) {
    return ramp_us + bytes / (dram_gbps * 1e3) + (tail ? tail_single_us : 0.0);
}

// the bytes each launch moves at B rows (counted from the source: reads + writes)
double launch_bytes(const std::string & l, int32_t B) {
    if (l == "embed") {
        return B * (4.0 + fn::embed::row_bytes) + 2048.0 + B * (double(CH) + F + 4) * 4;
    }
    if (l == "products") {
        return double(pw_bytes) + B * double(D) * 4 + B * double(CH + F) * 4;
    }
    if (l == "stats") {
        return B * (double(CH) + F + CH + F + 4) * 4 + 2.0 * CH * 4 + B * 64.0 * 2;
    }
    return double(pl::hist_version_floats) * 4 + double(CH) * 12 + B * (double(F) + CH + F + 4 + 32) * 4 +
           B * (double(pl::hist_version_floats) + CH) * 4;
}

void print_point(const std::string & l, int32_t B, const stats & w, const stats * c, const char * warm_what) {
    const double bytes = launch_bytes(l, B);
    const double fl = floor_us(bytes, l == "products");
    printf("{\"point\": \"%s\", \"rows\": %d, \"bytes\": %.0f, \"floor_us\": %.3f, \"median_us\": %.3f, "
           "\"p90_us\": %.3f, \"hw_pct\": %.1f, \"warm\": \"%s\", \"t_start\": \"%s\", \"t_end\": \"%s\"",
           l.c_str(), B, bytes, fl, w.median, w.p90, 100.0 * fl / w.median, warm_what, w.t0.c_str(), w.t1.c_str());
    if (c != nullptr) {
        printf(", \"cold_median_us\": %.3f, \"cold_p90_us\": %.3f, \"hw_pct_cold\": %.1f, \"cold\": \"(A - B) / %d: 12 "
               "MB L2 flush and instruction-cache eviction before each launch\", \"cold_t_start\": \"%s\", "
               "\"cold_t_end\": \"%s\"",
               c->median, c->p90, 100.0 * fl / c->median, n_cold_rep, c->t0.c_str(), c->t1.c_str());
    }
    print_us("us", w.us);
    if (c != nullptr) {
        print_us("cold_us", c->us);
    }
    printf("}\n");
    fflush(stdout);
}

// one launch of point l on card c at B rows, unit u
bool launch(card & c, const x::rings & rg, const std::string & l, int32_t B, int32_t u) {
    if (l == "embed") {
        return fn::embed::embed(B, c.eargs(rg, u), c.st);
    }
    // stats sends to the peer's slot 0 with no reader and no reset; update reads this card's slot 0, prefilled
    const pl::step_args a = c.args(rg, u % c.n_copies, 0, -1);
    if (l == "products") {
        return pl::products(B, a, c.st);
    }
    if (l == "stats") {
        return pl::stats(B, a, c.st);
    }
    return pl::update(B, a, c.st); // "update"
}

const char * warm_what(const std::string & l) {
    if (l == "products") {
        return "weights rotated over >= 200 MB (7 copies), activations and code warm";
    }
    if (l == "embed") {
        return "64 token sets of random IDs over the 136.6 MB table (rows cold), code and grid warm";
    }
    return "activations, state and code warm (producer-consumer data in a round)";
}

// ---- the combined graph: the embedding and the PLE layer, n_rounds rounds a replay, both cards -----------------

bool combined(card (&c)[2], x::rings & rg, int32_t B) {
    graph_exec ge[2];
    event_pair ev[2];
    bool ok = rg.reset_all();
    for (int32_t r = 0; r < 2 && ok; ++r) {
        ok = cuda_ok(cudaSetDevice(c[r].dev), "device") && ev[r].make() &&
             capture(
                 c[r].st, n_rounds,
                 [&](int32_t j) {
                     // round j: slot j, resetting slot j - 1 (mod 3) of the previous round
                     const pl::step_args a = c[r].args(rg, j % c[r].n_copies, j, (j + n_rounds - 1) % n_rounds);
                     return fn::embed::embed(B, c[r].eargs(rg, j), c[r].st) && pl::step(B, a, c[r].st);
                 },
                 ge[r]);
    }
    std::vector<double> us[2];
    // clocks up on both cards (card 1 idles through the alone points and starts at 1,260 MHz): 2 s of replays, then
    // n_warm untimed and n_timed timed
    int32_t warm_replays = 0;
    for (const auto w0 = std::chrono::steady_clock::now();
         ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 2.0; ++warm_replays) {
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = cuda_ok(cudaSetDevice(c[r].dev), "device") && cuda_ok(cudaGraphLaunch(ge[r].x, c[r].st), "replay");
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = cuda_ok(cudaSetDevice(c[r].dev), "device") && cuda_ok(cudaStreamSynchronize(c[r].st), "sync");
        }
    }
    const std::string t0 = utc_now();
    for (int32_t i = 0; i < n_warm + n_timed && ok; ++i) {
        for (int32_t r = 0; r < 2 && ok; ++r) {
            ok = cuda_ok(cudaSetDevice(c[r].dev), "device") && cuda_ok(cudaEventRecord(ev[r].e0, c[r].st), "record") &&
                 cuda_ok(cudaGraphLaunch(ge[r].x, c[r].st), "replay") &&
                 cuda_ok(cudaEventRecord(ev[r].e1, c[r].st), "record");
        }
        for (int32_t r = 0; r < 2 && ok; ++r) {
            float ms = 0.0f;
            ok = cuda_ok(cudaSetDevice(c[r].dev), "device") && cuda_ok(cudaEventSynchronize(ev[r].e1), "sync") &&
                 cuda_ok(cudaEventElapsedTime(&ms, ev[r].e0, ev[r].e1), "elapsed");
            if (i >= n_warm) {
                us[r].push_back(1e3 * double(ms) / n_rounds);
            }
        }
    }
    const std::string t1 = utc_now();
    uint32_t err0 = 1, err1 = 1;
    ok = ok && rg.read_err(0, err0) && rg.read_err(1, err1) && err0 == 0 && err1 == 0;
    if (!ok) {
        fprintf(stderr, "ple-bench: combined B %d failed (error words 0x%x 0x%x)\n", B, err0, err1);
        return false;
    }
    for (int32_t r = 0; r < 2; ++r) {
        stats s;
        s.us = us[r];
        s.finish();
        const double fl = floor_us(launch_bytes("embed", B), false) + floor_us(launch_bytes("products", B), true) +
                          floor_us(launch_bytes("stats", B), false) + floor_us(launch_bytes("update", B), false);
        printf("{\"point\": \"embed+ple (one graph, both cards)\", \"card\": %d, \"rows\": %d, \"median_us\": %.3f, "
               "\"p90_us\": %.3f, \"sum_of_floors_us\": %.3f, \"what\": \"a round: embed, products, stats (X-PLE-STATS "
               "send), update (poll); %d rounds a replay (slots 0..2), both cards' graphs launched together; weights "
               "32.8 MB a round (cold in L2 by size)\", \"clock_warm_replays\": %d, \"t_start\": \"%s\", \"t_end\": "
               "\"%s\"",
               r, B, s.median, s.p90, fl, n_rounds, warm_replays, t0.c_str(), t1.c_str());
        print_us("us", s.us);
        printf("}\n");
    }
    fflush(stdout);
    return true;
}

// ---- the host gather --------------------------------------------------------------------------------------------

bool read_tokens(const char * path, std::vector<int32_t> & out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    long v = -1;
    for (const char ch : s) {
        if (ch >= '0' && ch <= '9') {
            v = (v < 0 ? 0 : v) * 10 + (ch - '0');
        } else if (v >= 0) {
            out.push_back(static_cast<int32_t>(v));
            v = -1;
        }
    }
    return !out.empty();
}

// the PLE table mapped fresh in this process, and the tokens
struct host_table {
    rt::ple_params pp;
    std::vector<int32_t> toks;
    fn::model::mapped_file map;
    const uint8_t * table = nullptr;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
};

bool open_table(const char * model, const char * tokens, host_table & h) {
    std::string err;
    if (!rt::read_ple_params(model, h.pp, err) || !read_tokens(tokens, h.toks)) {
        fprintf(stderr, "ple-bench: host gather: %s\n", err.empty() ? "cannot read the tokens" : err.c_str());
        return false;
    }
    std::vector<std::string> paths;
    if (!fnfl_gguf_shard_paths(model, paths, err)) {
        fprintf(stderr, "ple-bench: host gather: %s\n", err.c_str());
        return false;
    }
    for (const std::string & p : paths) {
        fnfl_gguf_dir dir;
        if (!fnfl_gguf_read_dir(p, dir, err)) {
            fprintf(stderr, "ple-bench: host gather: %s\n", err.c_str());
            return false;
        }
        for (const fnfl_gguf_tensor & t : dir.tensors) {
            if (t.name == "per_layer_token_embd.weight") {
                if (!h.map.open(p, err)) {
                    fprintf(stderr, "ple-bench: host gather: %s\n", err.c_str());
                    return false;
                }
                h.table = h.map.data() + t.offset;
                h.file_offset = t.offset;
                h.bytes = t.bytes;
            }
        }
    }
    if (h.table == nullptr) {
        fprintf(stderr, "ple-bench: host gather: no table\n");
        return false;
    }
    return true;
}

// one pass of steps at B rows: indices, the prefetch (PL1, when on) and the gather a step, timed; keep (when given)
// receives every step's e
bool host_pass(const host_table & h, int32_t B, int32_t steps, bool prefetch, const char * label,
               std::vector<float> * keep) {
    constexpr int32_t first = 64;
    if (int64_t(h.toks.size()) < first + int64_t(steps) * B + B) {
        fprintf(stderr, "ple-bench: host gather: too few tokens (%zu)\n", h.toks.size());
        return false;
    }
    std::vector<float> e(size_t(B) * D);
    if (keep != nullptr) {
        keep->assign(size_t(steps) * size_t(B) * D, 0.0f);
    }
    std::vector<double> us, pf_us;
    int64_t ranges = 0;
    std::string err;
    rusage r0 = {}, r1 = {};
    const int64_t io0 = rt::storage_read_bytes();
    getrusage(RUSAGE_SELF, &r0);
    const auto w0 = std::chrono::steady_clock::now();
    for (int32_t s = 0; s < steps; ++s) {
        const int64_t pos = first + int64_t(s) * B;
        int32_t idx[4 * 16];
        int32_t n_ranges = 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (!rt::ple_indices(h.pp, h.toks.data(), pos, h.toks.data() + pos, B, idx)) {
            return false;
        }
        const auto tp = std::chrono::steady_clock::now();
        if (prefetch && !rt::ple_prefetch(h.pp, h.table, idx, B, n_ranges, err)) {
            fprintf(stderr, "ple-bench: %s\n", err.c_str());
            return false;
        }
        const auto tg = std::chrono::steady_clock::now();
        if (!rt::ple_gather(h.pp, h.table, idx, B, e.data())) {
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        pf_us.push_back(std::chrono::duration<double, std::micro>(tg - tp).count());
        ranges += n_ranges;
        if (keep != nullptr) {
            std::memcpy(keep->data() + size_t(s) * size_t(B) * D, e.data(), e.size() * sizeof(float));
        }
    }
    const double wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();
    getrusage(RUSAGE_SELF, &r1);
    const int64_t io1 = rt::storage_read_bytes();
    std::sort(us.begin(), us.end());
    std::sort(pf_us.begin(), pf_us.end());
    printf("{\"point\": \"host gather\", \"pass\": \"%s\", \"prefetch\": \"%s\", \"rows\": %d, \"steps\": %d, "
           "\"median_us\": %.2f, \"p90_us\": %.2f, \"max_us\": %.2f, \"mean_us\": %.2f, \"prefetch_median_us\": %.2f, "
           "\"prefetch_p90_us\": %.2f, \"madvise_calls_per_step\": %.2f, \"minor_faults\": %ld, \"major_faults\": %ld, "
           "\"storage_read_mb\": %.3f, \"what\": \"indices + prefetch (when on) + %d rows of 90 B gathered and decoded "
           "(IQ4_NL) a step, one thread; real tokens as the ledger and rows; the table mapped fresh in this "
           "process\"}\n",
           label, prefetch ? "on" : "off", B, steps, us[us.size() / 2], us[(9 * us.size()) / 10], us.back(),
           wall / steps, pf_us[pf_us.size() / 2], pf_us[(9 * pf_us.size()) / 10], double(ranges) / steps,
           r1.ru_minflt - r0.ru_minflt, r1.ru_majflt - r0.ru_majflt,
           io0 >= 0 && io1 >= 0 ? double(io1 - io0) / 1e6 : -1.0, 16 * B);
    fflush(stdout);
    return true;
}

bool print_counts(const host_table & h, const char * when, rt::page_counts & c) {
    std::string err;
    if (!rt::page_cache_counts(h.table, h.bytes, h.map.path(), h.file_offset, c, err)) {
        fprintf(stderr, "ple-bench: %s\n", err.c_str());
        return false;
    }
    printf("{\"point\": \"table pages\", \"when\": \"%s\", \"pages\": %lld, \"mincore_resident\": %lld, "
           "\"cachestat_cached\": %lld}\n",
           when, static_cast<long long>(c.pages), static_cast<long long>(c.mapping_res),
           static_cast<long long>(c.file_cached));
    fflush(stdout);
    return true;
}

// E1's host point (mode all): first touch and warm at B = 3, the prefetch on (PL1's default)
bool host_gather(const char * model, const char * tokens) {
    host_table h;
    return open_table(model, tokens, h) && host_pass(h, 3, 1000, true, "first touch", nullptr) &&
           host_pass(h, 3, 1000, true, "warm", nullptr);
}

// PL1 (mode host:<off|on>:<B>): the table evicted (page_cache.h; void when more than 1,024 of its pages stay cached),
// then cold, page-cached first touch (the process's page-table entries dropped, the pages cached) and warm passes of
// 1,000 steps, then every cold step's e checked bitwise against a gather without the prefetch on the same indices
constexpr int64_t void_pages = 1024;

bool host_cold(const char * model, const char * tokens, bool prefetch, int32_t B) {
    constexpr int32_t steps = 1000;
    host_table h;
    rt::page_counts c;
    std::string err;
    if (!open_table(model, tokens, h) || !print_counts(h, "before eviction", c)) {
        return false;
    }
    if (!rt::page_cache_evict(h.table, h.bytes, h.map.path(), h.file_offset, err)) {
        fprintf(stderr, "ple-bench: %s\n", err.c_str());
        return false;
    }
    if (!print_counts(h, "after eviction", c)) {
        return false;
    }
    if (c.mapping_res > void_pages || c.file_cached > void_pages) {
        printf("{\"point\": \"VOID\", \"why\": \"the table starts warm: more than %lld pages cached\"}\n",
               static_cast<long long>(void_pages));
        return false;
    }
    std::vector<float> cold;
    if (!host_pass(h, B, steps, prefetch, "cold", &cold) || !print_counts(h, "after cold", c) ||
        !rt::page_cache_unmap(h.table, h.bytes, err) || !print_counts(h, "after unmap", c) ||
        !host_pass(h, B, steps, prefetch, "page-cached first touch", nullptr) ||
        !host_pass(h, B, steps, prefetch, "warm", nullptr)) {
        if (!err.empty()) {
            fprintf(stderr, "ple-bench: %s\n", err.c_str());
        }
        return false;
    }
    // the check: the current gather (no prefetch) on the same indices, bitwise
    constexpr int32_t first = 64;
    std::vector<float> e(size_t(B) * D);
    int32_t differ = 0;
    for (int32_t s = 0; s < steps; ++s) {
        const int64_t pos = first + int64_t(s) * B;
        int32_t idx[4 * 16];
        if (!rt::ple_indices(h.pp, h.toks.data(), pos, h.toks.data() + pos, B, idx) ||
            !rt::ple_gather(h.pp, h.table, idx, B, e.data())) {
            return false;
        }
        differ += std::memcmp(e.data(), cold.data() + size_t(s) * size_t(B) * D, e.size() * sizeof(float)) != 0;
    }
    printf("{\"point\": \"bitwise\", \"prefetch\": \"%s\", \"rows\": %d, \"steps\": %d, \"steps_differing\": %d, "
           "\"what\": \"each cold step's e (indices, prefetch when on, gather) against the gather without the prefetch "
           "on the same indices afterwards\"}\n",
           prefetch ? "on" : "off", B, steps, differ);
    fflush(stdout);
    return differ == 0;
}

} // namespace

int fnfl_ple_bench(int dev0, int dev1, const char * mode, const char * model, const char * tokens) {
    const std::string m = mode;
    if (m.rfind("host:", 0) == 0) {
        // PL1: host only, no device
        const bool on = m.rfind("host:on:", 0) == 0;
        const int32_t B = atoi(m.c_str() + (on ? 8 : 9));
        if ((!on && m.rfind("host:off:", 0) != 0) || (B != 1 && B != 3)) {
            fprintf(stderr, "ple-bench: host mode is host:<off|on>:<1|3>\n");
            return 2;
        }
        printf("# ple-bench: %s, %s\n", m.c_str(), utc_now().c_str());
        return host_cold(model, tokens, on, B) ? 0 : 1;
    }
    std::string only;
    int32_t only_b = 0;
    if (m.rfind("ncu:", 0) == 0) {
        const size_t c = m.find(':', 4);
        only = m.substr(4, c == std::string::npos ? std::string::npos : c - 4);
        only_b = c == std::string::npos ? 3 : atoi(m.c_str() + c + 1);
    } else if (m != "all") {
        fprintf(stderr, "ple-bench: mode is all or ncu:<launch>:<B>\n");
        return 2;
    }
    const bool same = dev0 == dev1;
    if (!same && !x::enable_peer_access(dev0, dev1)) {
        return 1;
    }
    x::rings rg(dev0, dev1);
    card c[2];
    if (!rg.ok() || !c[0].make(dev0, 0) || !c[1].make(dev1, 1)) {
        return 1;
    }
    printf("# ple-bench: devices %d %d, %d weight copies (%.1f MB rotated), %s\n", dev0, dev1, c[0].n_copies,
           double(pw_bytes) * c[0].n_copies / 1e6, utc_now().c_str());
    // update's slot 0 on each card holds a payload (zeros), so update alone never waits
    bool ok = cuda_ok(cudaSetDevice(dev0), "device") &&
              cuda_ok(cudaMemset(rg.slot(0, x::msg::ple, 0), 0, size_t(x::slot_bytes(x::msg::ple))), "prefill") &&
              cuda_ok(cudaSetDevice(dev1), "device") &&
              cuda_ok(cudaMemset(rg.slot(1, x::msg::ple, 0), 0, size_t(x::slot_bytes(x::msg::ple))), "prefill") &&
              cuda_ok(cudaDeviceSynchronize(), "sync") && cuda_ok(cudaSetDevice(dev0), "device");
    const std::vector<std::string> launches = {"embed", "products", "stats", "update"};
    if (!only.empty()) {
        stats w;
        const int32_t units = only == "products" ? 2 * c[0].n_copies : n_unit;
        ok = ok && time_warm(c[0], units, [&](int32_t u) { return launch(c[0], rg, only, only_b, u); }, w);
        if (ok) {
            print_point(only, only_b, w, nullptr, warm_what(only));
        }
        return ok ? 0 : 1;
    }
    // clocks up (HX12 Later 15: a process starts at 1,260 MHz for 0.3-1 s): 2 s of product launches
    {
        stats w;
        const auto t0 = std::chrono::steady_clock::now();
        while (ok && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 2.0) {
            ok = time_warm(c[0], c[0].n_copies, [&](int32_t u) { return launch(c[0], rg, "products", 3, u); }, w);
        }
    }
    for (const int32_t B : {1, 3}) {
        for (const std::string & l : launches) {
            stats w, cd;
            const int32_t units = l == "products" ? 2 * c[0].n_copies : n_unit;
            ok = ok &&
                 time_warm(
                     c[0], units, [&](int32_t u) { return launch(c[0], rg, l, B, u); }, w) &&
                 time_cold(c[0], [&](int32_t u) { return launch(c[0], rg, l, B, u); }, cd);
            if (!ok) {
                fprintf(stderr, "ple-bench: %s at B %d failed\n", l.c_str(), B);
                return 1;
            }
            print_point(l, B, w, &cd, warm_what(l));
        }
    }
    for (const int32_t B : {1, 3}) {
        if (same) {
            printf("# ple-bench: one device: the combined two-card graph is skipped\n");
            break;
        }
        ok = ok && combined(c, rg, B);
    }
    ok = ok && host_gather(model, tokens);
    return ok ? 0 : 1;
}
