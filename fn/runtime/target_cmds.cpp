#include "fn/runtime/target_cmds.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/head/head.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/round/round.h"
#include "fn/loader/loader.h"
#include "fn/loader/run_load.h"
#include "fn/model/state.h"
#include "fn/proto/xp1/timing.h"
#include "fn/xchg/xchg.h"
#include "fn/model/model.h"
#include "fn/model/pass.h"
#include "fn/model/shapes.h"
#include "fn/runtime/page_cache.h"
#include "fn/runtime/target.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace rt = fn::runtime;
namespace md = fn::model;

constexpr int32_t V = fn::head::n_vocab;       // 248,320
constexpr int32_t VC = fn::head::n_vocab_card; // 124,160
constexpr int32_t F = 1280;
constexpr int32_t D = 2560;
const int32_t rec_layers[md::pass_max_records] = {11, 23, 35, 47};

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool read_ids(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    out.clear();
    long v = -1;
    for (const char ch : s) {
        if (ch >= '0' && ch <= '9') {
            v = (v < 0 ? 0 : v) * 10 + (ch - '0');
        } else {
            if (v >= 0) {
                out.push_back(static_cast<int32_t>(v));
            }
            v = -1;
        }
    }
    if (v >= 0) {
        out.push_back(static_cast<int32_t>(v));
    }
    return !out.empty();
}

bool read_file(const std::string & path, std::vector<uint8_t> & out, size_t want) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path.c_str());
        return false;
    }
    out.resize(want);
    if (!f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(want))) {
        fprintf(stderr, "%s: shorter than %zu bytes\n", path.c_str(), want);
        return false;
    }
    return true;
}

// a number after "key": inside the object after "obj": in a JSON text (ref-spread3.json's flat form)
bool json_number(const std::string & js, const std::string & obj, const std::string & key, double & out) {
    const size_t o = js.find("\"" + obj + "\"");
    if (o == std::string::npos) {
        return false;
    }
    const size_t k = js.find("\"" + key + "\"", o);
    if (k == std::string::npos) {
        return false;
    }
    const size_t c = js.find(':', k);
    if (c == std::string::npos) {
        return false;
    }
    out = std::strtod(js.c_str() + c + 1, nullptr);
    return true;
}

// numpy's default (linear) percentile
double percentile(std::vector<double> v, double q) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const double pos = q / 100.0 * double(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

uint64_t hash_bytes(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, b + i, 8);
        h = (h ^ w) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    for (; i < n; ++i) {
        h = (h ^ b[i]) * 0x100000001b3ull;
    }
    return h;
}

// KL(p || q) over the full vocabulary from F32 logits, in FP64
double kl_div(const float * p, const float * q0, const float * q1) {
    double mp = -HUGE_VAL, mq = -HUGE_VAL;
    for (int32_t i = 0; i < V; ++i) {
        mp = std::max(mp, double(p[i]));
        mq = std::max(mq, double(i < VC ? q0[i] : q1[i - VC]));
    }
    double sp = 0.0, sq = 0.0;
    for (int32_t i = 0; i < V; ++i) {
        sp += std::exp(double(p[i]) - mp);
        sq += std::exp(double(i < VC ? q0[i] : q1[i - VC]) - mq);
    }
    const double lsp = std::log(sp), lsq = std::log(sq);
    double kl = 0.0;
    for (int32_t i = 0; i < V; ++i) {
        const double lp = double(p[i]) - mp - lsp;
        const double lq = double(i < VC ? q0[i] : q1[i - VC]) - mq - lsq;
        kl += std::exp(lp) * (lp - lq);
    }
    return kl;
}

bool load(const fnfl_target_args & a, int32_t n_cells, md::model & m, const char * tool) {
    fn::loader::load_params lp;
    lp.model_path = a.model_path;
    lp.draft_path = a.draft_path;
    lp.ple_table = a.ple_table;
    fn::loader::run_load_report lr;
    std::string err;
    if (!fn::loader::load_for_run(lp, n_cells, m, lr, err)) {
        fprintf(stderr, "%s: load: %s\n", tool, err.c_str());
        return false;
    }
    printf("%s: model loaded in %.2f s (open %.2f, fill %.2f, reserve %.2f) at %d cells; free %.3f / %.3f GiB\n", tool,
           lr.total_s, lr.open_s, lr.fill_s, lr.reserve_s, n_cells, double(lr.free_after[0]) / 1073741824.0,
           double(lr.free_after[1]) / 1073741824.0);
    for (const std::string & line : fn::loader::ple_table_lines(lr)) {
        printf("%s: %s\n", tool, line.c_str());
    }
    return true;
}

// model-dry's load without model data: the loader's open (the mappings and the region plan, CPU) and allocation, every
// weight byte zeroed instead of filled (no model data reaches the cards), then load_for_run's reservation, grids,
// peer access and rings. Zero weights decode to zero products, so every value stays finite and greedy takes ID 0.
// The PLE table keeps the plain mapping whatever --ple-table says (PN1): pinning 28.8 GB is model data the dry load
// exists to avoid.
bool load_dry(const fnfl_target_args & a, int32_t n_cells, md::model & m) {
    fn::loader::load_params lp;
    lp.model_path = a.model_path;
    lp.draft_path = a.draft_path;
    fn::loader::load_times lt;
    std::string err;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(lp.device[c]), "device") || !cuda_ok(cudaFree(nullptr), "context")) {
            return false;
        }
    }
    if (!fn::loader::open_model(lp, m, lt, err) || !fn::loader::allocate_weights(m, err)) {
        fprintf(stderr, "model-dry: %s\n", err.c_str());
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaMemset(m.weights[c].data(), 0, m.weights[c].size()), "zero weights")) {
            return false;
        }
    }
    md::state_params sp;
    sp.n_cells = n_cells;
    sp.ple_conv_kernel = m.ple_conv_kernel;
    sp.ple_ngram = m.ple_ngram;
    for (const md::buffer_class & plan : md::state_classes(sp)) {
        auto rc = std::make_unique<md::reserved_class>();
        rc->plan = plan;
        for (int32_t c = 0; c < 2; ++c) {
            const int64_t bytes = plan.allocation(c);
            if (!rc->buf[c].allocate(m.device[c], size_t(bytes), err) ||
                (bytes > 0 && (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
                               !cuda_ok(cudaMemset(rc->buf[c].data(), 0, size_t(bytes)), "zero state")))) {
                fprintf(stderr, "model-dry: %s %s\n", plan.name.c_str(), err.c_str());
                return false;
            }
        }
        m.state.push_back(std::move(rc));
    }
    if (!fn::loader::fill_ex1_grids(m, err) || !fn::xchg::enable_peer_access(m.device[0], m.device[1])) {
        fprintf(stderr, "model-dry: grids or peer access %s\n", err.c_str());
        return false;
    }
    m.rings = std::make_unique<fn::xchg::rings>(m.device[0], m.device[1]);
    return m.rings->ok() && cuda_ok(cudaDeviceSynchronize(), "dry load");
}

// card 1's %globaltimer minus card 0's (XP1's ping-pong, fn/proto/xp1/timing.h): the start skew is t_start[1] -
// offset - t_start[0]
bool clock_offset(md::model & m, double & offset_ns, double & rtt_ns) {
    void * f[2] = {nullptr, nullptr};
    void * o[2] = {nullptr, nullptr};
    const int32_t n = 64;
    bool ok = true;
    for (int32_t c = 0; c < 2 && ok; ++c) {
        ok = cuda_ok(cudaSetDevice(m.device[c]), "device") && cuda_ok(cudaMalloc(&f[c], 16), "flags") &&
             cuda_ok(cudaMalloc(&o[c], size_t(2) * n * 8), "pings");
    }
    ok = ok &&
         fn::xp1::clock_offset(m.device[0], m.device[1], static_cast<uint32_t *>(f[0]), static_cast<uint32_t *>(f[1]),
                               static_cast<uint64_t *>(o[0]), static_cast<uint64_t *>(o[1]), n, offset_ns, rtt_ns);
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m.device[c]);
        cudaFree(f[c]);
        cudaFree(o[c]);
    }
    if (ok) {
        printf("clock offset card 1 - card 0 %.0f ns, round trip %.0f ns (XP1's ping-pong, median of 63)\n", offset_ns,
               rtt_ns);
    }
    return ok;
}

double g_offset_ns = 0.0; // the process's measured clock offset, for the skew

// ---- model-test ----

struct card_records {
    md::pass_record_layer rec[md::pass_max_records];
    void * mem = nullptr;
};

bool alloc_records(md::model & m, int32_t card, card_records & r) {
    // SP2: five rows ([5][4][1280], [5][4] in 128 bytes, [5][1280], [5][2560])
    const size_t xb = size_t(5) * 4 * F * 4, ib = 128, ob = size_t(5) * F * 4, mb = size_t(5) * D * 4;
    const size_t per = xb + ib + ob + 2 * mb;
    if (!cuda_ok(cudaSetDevice(m.device[card]), "device") ||
        !cuda_ok(cudaMalloc(&r.mem, per * md::pass_max_records), "records")) {
        return false;
    }
    uint8_t * p = static_cast<uint8_t *>(r.mem);
    for (int32_t i = 0; i < md::pass_max_records; ++i) {
        md::pass_record_layer & q = r.rec[i];
        q.layer = rec_layers[i];
        q.x = reinterpret_cast<float *>(p);
        q.inject = reinterpret_cast<float *>(p + xb);
        q.out = reinterpret_cast<float *>(p + xb + ib);
        const bool mixed = q.layer == 11 || q.layer == 35;
        q.mixed_attn = mixed ? reinterpret_cast<float *>(p + xb + ib + ob) : nullptr;
        q.mixed_ffn = mixed ? reinterpret_cast<float *>(p + xb + ib + ob + mb) : nullptr;
        p += per;
    }
    return true;
}

struct pass_out {
    std::vector<uint64_t> hash;  // per position: both cards' logits
    std::vector<int32_t> argmax; // per position
    std::vector<double> kl;      // per p % 8 == 7 position (first pass only)
    int64_t rounds = 0;
};

// one forced pass over n positions from empty state; B: rows of each round in turn
bool run_pass(rt::target & t, const std::vector<int32_t> & ids, int32_t n, const std::vector<int32_t> & cycle,
              const std::vector<uint8_t> * ref_logits, card_records * rec, FILE * const * rec_files, pass_out & o,
              float * h_logits) {
    std::string err;
    if (!t.reset(err)) {
        fprintf(stderr, "model-test: reset: %s\n", err.c_str());
        return false;
    }
    t.set_prompt(std::vector<int32_t>(ids.begin(), ids.begin() + n));
    o.hash.assign(size_t(n), 0);
    o.argmax.assign(size_t(n), -1);
    o.kl.clear();
    constexpr size_t R = size_t(fn::round::max_rows); // SP2: five rows (h_logits [2][R][VC])
    std::vector<float> hrec(R * 4 * F + R * 4 + R * F + 2 * R * D);
    std::vector<float> rows_out(R * 4 * D), mixed_out(R * D);
    int32_t pos = 0;
    size_t ci = 0;
    while (pos < n) {
        const int32_t B = std::min(cycle[ci % cycle.size()], n - pos);
        ++ci;
        rt::round_result r;
        if (!t.forced(B, r, err)) {
            fprintf(stderr, "model-test: round at position %d (B %d): %s\n", pos, B, err.c_str());
            return false;
        }
        ++o.rounds;
        // the logits (valid until the next round's head, FC-HEAD2): both cards' shards
        for (int32_t c = 0; c < 2; ++c) {
            if (!cuda_ok(cudaSetDevice(t.card(c).c.device), "device") ||
                !cuda_ok(cudaMemcpy(h_logits + size_t(c) * R * VC, t.card(c).logits, size_t(B) * VC * 4,
                                    cudaMemcpyDeviceToHost),
                         "logits")) {
                return false;
            }
        }
        for (int32_t b = 0; b < B; ++b) {
            const int32_t p = pos + b;
            const float * l0 = h_logits + size_t(b) * VC;
            const float * l1 = h_logits + R * VC + size_t(b) * VC;
            o.hash[size_t(p)] = hash_bytes(l1, size_t(VC) * 4, hash_bytes(l0, size_t(VC) * 4, 0xcbf29ce484222325ull));
            o.argmax[size_t(p)] = r.argmax[0][b];
            if (ref_logits != nullptr && p % 8 == 7) {
                const float * ref = reinterpret_cast<const float *>(ref_logits->data()) + size_t(p / 8) * V;
                o.kl.push_back(kl_div(ref, l0, l1));
            }
        }
        // the records: l_last as [4][2560] a position (stream s, card r's features at r * 1280), hc_mixed [2560]
        if (rec != nullptr) {
            for (int32_t i = 0; i < md::pass_max_records; ++i) {
                for (int32_t c = 0; c < 2; ++c) {
                    const md::pass_record_layer & q = rec[c].rec[i];
                    float * h = hrec.data();
                    if (!cuda_ok(cudaSetDevice(t.card(c).c.device), "device") ||
                        !cuda_ok(cudaMemcpy(h, q.x, size_t(B) * 4 * F * 4, cudaMemcpyDeviceToHost), "rec x") ||
                        !cuda_ok(cudaMemcpy(h + R * 4 * F, q.inject, size_t(B) * 4 * 4, cudaMemcpyDeviceToHost),
                                 "rec inject") ||
                        !cuda_ok(
                            cudaMemcpy(h + R * 4 * F + R * 4, q.out, size_t(B) * F * 4, cudaMemcpyDeviceToHost),
                            "rec out")) {
                        return false;
                    }
                    for (int32_t b = 0; b < B; ++b) {
                        for (int32_t s = 0; s < 4; ++s) {
                            const float inj = h[R * 4 * F + size_t(b) * 4 + size_t(s)];
                            for (int32_t j = 0; j < F; ++j) {
                                const float x = h[(size_t(b) * 4 + s) * F + j];
                                const float ox = h[R * 4 * F + R * 4 + size_t(b) * F + j];
                                rows_out[(size_t(b) * 4 + s) * D + size_t(c) * F + j] = std::fma(inj, ox, x);
                            }
                        }
                    }
                }
                if (fwrite(rows_out.data(), 4, size_t(B) * 4 * D, rec_files[i]) != size_t(B) * 4 * D) {
                    return false;
                }
                const md::pass_record_layer & q0 = rec[0].rec[i];
                if (q0.mixed_attn != nullptr) {
                    const int32_t k = rec_layers[i] == 11 ? 0 : 1;
                    for (int32_t side = 0; side < 2; ++side) {
                        std::vector<float> m1(size_t(B) * D);
                        if (!cuda_ok(cudaSetDevice(t.card(0).c.device), "device") ||
                            !cuda_ok(cudaMemcpy(mixed_out.data(), side == 0 ? q0.mixed_attn : q0.mixed_ffn,
                                                size_t(B) * D * 4, cudaMemcpyDeviceToHost),
                                     "rec mixed") ||
                            !cuda_ok(cudaSetDevice(t.card(1).c.device), "device") ||
                            !cuda_ok(cudaMemcpy(m1.data(),
                                                side == 0 ? rec[1].rec[i].mixed_attn : rec[1].rec[i].mixed_ffn,
                                                size_t(B) * D * 4, cudaMemcpyDeviceToHost),
                                     "rec mixed")) {
                            return false;
                        }
                        if (std::memcmp(m1.data(), mixed_out.data(), size_t(B) * D * 4) != 0) {
                            fprintf(stderr, "model-test: the cards' hc_mixed rows differ at layer %d\n", rec_layers[i]);
                            return false;
                        }
                        if (fwrite(mixed_out.data(), 4, size_t(B) * D, rec_files[4 + 2 * k + side]) != size_t(B) * D) {
                            return false;
                        }
                    }
                }
            }
        }
        pos += B;
    }
    return true;
}

// ---- model-bench ----

// a synthetic history (R3's: random K, V and index keys in their stored formats, summaries from QA1b's summary_update,
// random GDN and PLE state) over the whole reservation; the values need only be finite
bool fill_synthetic(md::model & m, rt::target & t, int32_t depth) {
    std::vector<uint16_t> h16(size_t(1) << 25);
    std::vector<float> h32(size_t(1) << 24);
    uint64_t x = 0x9e3779b97f4a7c15ull;
    const auto next = [&]() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>(static_cast<double>(x >> 11) * (2.0 / 9007199254740992.0) - 1.0);
    };
    for (uint16_t & v : h16) {
        const float f = next();
        uint32_t u = 0;
        std::memcpy(&u, &f, 4);
        v = static_cast<uint16_t>(u >> 16);
    }
    for (float & v : h32) {
        v = next();
    }
    size_t at16 = 0, at32 = 0;
    const auto fill = [&](uint8_t * dst, size_t bytes, bool bf16, float scale) {
        size_t done = 0;
        std::vector<float> scaled;
        while (done < bytes) {
            if (bf16) {
                const size_t n = std::min(bytes - done, (h16.size() - at16) * 2);
                if (!cuda_ok(cudaMemcpy(dst + done, h16.data() + at16, n, cudaMemcpyHostToDevice), "fill")) {
                    return false;
                }
                done += n;
                at16 = (at16 + n / 2 + 977) % (h16.size() / 2);
            } else {
                const size_t n = std::min(bytes - done, (h32.size() - at32) * 4);
                scaled.assign(h32.begin() + static_cast<std::ptrdiff_t>(at32),
                              h32.begin() + static_cast<std::ptrdiff_t>(at32 + n / 4));
                for (float & v : scaled) {
                    v *= scale;
                }
                if (!cuda_ok(cudaMemcpy(dst + done, scaled.data(), n, cudaMemcpyHostToDevice), "fill")) {
                    return false;
                }
                done += n;
                at32 = (at32 + n / 4 + 613) % (h32.size() / 2);
            }
        }
        return true;
    };
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device")) {
            return false;
        }
        for (const char * cls : {"target KV (K0)", "target index keys and summaries", "GDN state",
                                 "convolution histories", "draft KV (F16)"}) {
            const md::reserved_class * rc = m.state_class(cls);
            for (int32_t p = 0; rc != nullptr && p < (int32_t)rc->plan.parts.size(); ++p) {
                const std::string & name = rc->plan.parts[p].name;
                const int64_t n = rc->plan.parts[p].bytes[c];
                if (n <= 0 || name.find("summaries") != std::string::npos) {
                    continue;
                }
                // the draft's F16 cells take the same bits: BF16 values in [-1, 1] read as F16 are finite
                const bool bf16 = std::string(cls) == "target KV (K0)" || std::string(cls) == "draft KV (F16)" ||
                                  name.find("keys") != std::string::npos;
                const float scale = std::string(cls) == "GDN state" ? 0.05f : 1.0f;
                if (!fill(rc->part_ptr(p, c), size_t(n), bf16, scale)) {
                    return false;
                }
            }
        }
        // the summaries of every block the deepest timed rows see, pooled from the random keys by QA1b's kernel
        md::pass_card & pc = t.card(c);
        for (int32_t l = 3; l < fn::n_layer; l += 4) {
            if (pc.w[l].index_owner != c) {
                continue;
            }
            fn::qsa::update_args ua;
            ua.keys = pc.s[l].idx_keys;
            ua.k_norm = pc.w[l].idx_k_norm;
            ua.summaries = pc.s[l].idx_summaries;
            ua.first = 0;
            ua.count = fn::qsa::blocks_complete(depth + 4);
            ua.rope = fn::qsa::model_rope();
            if (!fn::qsa::summary_update(ua, t.stream(c))) {
                fprintf(stderr, "model-bench: summary_update\n");
                return false;
            }
        }
        if (!cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "summaries")) {
            return false;
        }
    }
    return true;
}

struct sample {
    double span_ms[2];
    double pass_us[2]; // stamp after the start to the finish node
    double skew_us;    // card 1's start stamp minus card 0's
};

double med(const std::vector<double> & v) {
    return percentile(v, 50.0);
}

// nearest-rank p90 as earlier benches report it
double p90(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t k = static_cast<size_t>(std::ceil(0.9 * double(v.size()))) - 1;
    return v[std::min(k, v.size() - 1)];
}

bool clock_offset(md::model & m, double & offset_ns, double & rtt_ns);

bool bench_point(md::model & m, rt::target & t, FILE * jf, int32_t depth, int32_t B, bool history_real, int32_t n_warm,
                 int32_t n_timed, const char * label) {
    std::string err;
    // the cards' timers drift apart (about 0.3 us a second, model-bench-1): the offset again before each point
    double rtt_ns = 0.0;
    if (!clock_offset(m, g_offset_ns, rtt_ns)) {
        fprintf(stderr, "model-bench: clock offset\n");
        return false;
    }
    std::vector<sample> s;
    for (int32_t i = 0; i < n_warm + n_timed; ++i) {
        rt::round_result r;
        if (!t.set_depth(depth, err) || !t.forced(B, r, err, false, true)) {
            fprintf(stderr, "model-bench: depth %d B %d replay %d: %s\n", depth, B, i, err.c_str());
            return false;
        }
        if (i < n_warm) {
            continue;
        }
        sample q;
        for (int32_t c = 0; c < 2; ++c) {
            q.span_ms[c] = r.span_ms[c];
            q.pass_us[c] = double(r.t_end[c] - r.t_start[c]) / 1000.0;
        }
        q.skew_us = (double(r.t_start[1]) - g_offset_ns - double(r.t_start[0])) / 1000.0;
        s.push_back(q);
        fprintf(jf,
                "{\"label\": \"%s\", \"depth\": %d, \"B\": %d, \"history\": \"%s\", \"replay\": %d, \"round\": %llu, "
                "\"span_ms\": [%.4f, %.4f], \"pass_us\": [%.2f, %.2f], \"skew_us\": %.3f, \"begin_wait_us\": [%.2f, "
                "%.2f]}\n",
                label, depth, B, history_real ? "real" : "synthetic", i - n_warm,
                static_cast<unsigned long long>(r.round), q.span_ms[0], q.span_ms[1], q.pass_us[0], q.pass_us[1],
                q.skew_us, double(r.t_start[0] - r.t_begin[0]) / 1000.0, double(r.t_start[1] - r.t_begin[1]) / 1000.0);
    }
    std::vector<double> sp[2], pa[2], sk;
    for (const sample & q : s) {
        for (int32_t c = 0; c < 2; ++c) {
            sp[c].push_back(q.span_ms[c] * 1000.0);
            pa[c].push_back(q.pass_us[c]);
        }
        sk.push_back(q.skew_us);
    }
    std::vector<double> ask;
    for (double v : sk) {
        ask.push_back(std::fabs(v));
    }
    printf("model-bench: %-9s depth %6d B %d  pass us card0 %9.2f (p90 %9.2f) card1 %9.2f (p90 %9.2f)  span us %9.2f / "
           "%9.2f  skew us med %.2f p90 %.2f max %.2f\n",
           label, depth, B, med(pa[0]), p90(pa[0]), med(pa[1]), p90(pa[1]), med(sp[0]), med(sp[1]), med(sk), p90(ask),
           *std::max_element(ask.begin(), ask.end()));
    fflush(stdout);
    fflush(jf);
    return true;
}

// rounds for about `seconds` (the clocks up before timing), untimed
bool warm(rt::target & t, int32_t depth, double seconds) {
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        rt::round_result r;
        if (!t.set_depth(depth, err) || !t.forced(1, r, err, false, false)) {
            fprintf(stderr, "model-bench: warm-up: %s\n", err.c_str());
            return false;
        }
    }
    return true;
}

} // namespace

int fnfl_model_test(const fnfl_target_args & a) {
    const bool full = a.mode == "full";
    if (!full && a.mode != "san" && a.mode != "race") {
        fprintf(stderr, "model-test: mode is full, san or race\n");
        return 2;
    }
    const int32_t N = full ? a.n_pos : 16;
    std::vector<int32_t> ids;
    if (!read_ids(a.ids, ids) || int32_t(ids.size()) < N) {
        fprintf(stderr, "model-test: %s holds fewer than %d tokens\n", a.ids.c_str(), N);
        return 1;
    }
    // the reference and the yardstick (full mode)
    std::vector<uint8_t> ref_top, ref_logits;
    double y_p50 = 0, y_p99 = 0, y_max = 0, y_top1 = 0;
    if (full) {
        std::string js;
        {
            std::ifstream f(a.spread_json);
            std::stringstream ss;
            ss << f.rdbuf();
            js = ss.str();
        }
        if (!read_file(a.ref_dir + "/top.bin", ref_top, size_t(N) * 16) ||
            !read_file(a.ref_dir + "/logits.bin", ref_logits, size_t(N / 8) * V * 4) ||
            !json_number(js, "kl", "p50", y_p50) || !json_number(js, "kl", "p99", y_p99) ||
            !json_number(js, "kl", "max", y_max) || !json_number(js, "top1", "differ", y_top1)) {
            fprintf(stderr, "model-test: the reference or the yardstick (%s) is missing\n", a.spread_json.c_str());
            return 1;
        }
        printf("model-test: yardstick KL(decode || prompt) p50 %.6g p99 %.6g max %.6g; top-1 differs at %.0f "
               "positions\n",
               y_p50, y_p99, y_max, y_top1);
    }
    auto m = std::make_unique<md::model>();
    if (!load(a, full ? a.n_cells : 8192, *m, "model-test")) {
        return 1;
    }
    rt::target_options o;
    o.model_path = a.model_path;
    o.eager = !full;
    o.timeout_s = full ? 60.0 : 1800.0;
    if (a.mode == "race") {
        o.pass.layer_lo = 8;
        o.pass.layer_hi = 11;
    }
    card_records rec[2];
    if (full) {
        if (!alloc_records(*m, 0, rec[0]) || !alloc_records(*m, 1, rec[1])) {
            return 1;
        }
        o.pass.records[0] = rec[0].rec;
        o.pass.records[1] = rec[1].rec;
        o.pass.n_records = md::pass_max_records;
    }
    rt::target t;
    std::string err;
    const auto ti = std::chrono::steady_clock::now();
    if (!t.init(*m, o, err)) {
        fprintf(stderr, "model-test: init: %s\n", err.c_str());
        return 1;
    }
    printf("model-test: init (dense packing, passes, %d graph captures) in %.2f s\n", t.captures(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count());
    float * h_logits = nullptr;
    if (!cuda_ok(cudaMallocHost(&h_logits, size_t(2) * fn::round::max_rows * VC * 4), "pinned logits")) {
        return 1;
    }
    std::unique_ptr<float, decltype(&cudaFreeHost)> hold(h_logits, cudaFreeHost);
    const std::vector<int32_t> cycle = {1, 2, 3, 4};
    if (!full) {
        pass_out p;
        if (!run_pass(t, ids, N, cycle, nullptr, nullptr, nullptr, p, h_logits)) {
            return 1;
        }
        printf("model-test %s: %lld rounds over %d positions (eager, both cards in phases; layers %d..%d), error words "
               "0\n",
               a.mode.c_str(), static_cast<long long>(p.rounds), N, o.pass.layer_lo, o.pass.layer_hi);
        return 0;
    }
    // pass 1, with the records
    const char * rec_names[8] = {"l_last-11",        "l_last-23",       "l_last-35",        "l_last-47",
                                 "hc_mixed-11.attn", "hc_mixed-11.ffn", "hc_mixed-35.attn", "hc_mixed-35.ffn"};
    FILE * rf[8] = {};
    for (int32_t i = 0; i < 8; ++i) {
        rf[i] = fopen((a.out_dir + "/fnfl-" + rec_names[i] + ".bin").c_str(), "wb");
        if (rf[i] == nullptr) {
            fprintf(stderr, "model-test: cannot write %s\n", a.out_dir.c_str());
            return 1;
        }
    }
    pass_out p1, p2, p3;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok1 = run_pass(t, ids, N, cycle, &ref_logits, rec, rf, p1, h_logits);
    for (FILE * f : rf) {
        fclose(f);
    }
    if (!ok1) {
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (!run_pass(t, ids, N, cycle, nullptr, nullptr, nullptr, p2, h_logits)) {
        return 1;
    }
    const auto t2 = std::chrono::steady_clock::now();
    const int32_t n_bi = std::min(N, 512);
    if (!run_pass(t, ids, n_bi, {1}, nullptr, nullptr, nullptr, p3, h_logits)) {
        return 1;
    }
    const auto t3 = std::chrono::steady_clock::now();
    printf("model-test: pass 1 %lld rounds %.1f s, pass 2 %.1f s, B = 1 pass %lld rounds %.1f s\n",
           static_cast<long long>(p1.rounds), std::chrono::duration<double>(t1 - t0).count(),
           std::chrono::duration<double>(t2 - t1).count(), static_cast<long long>(p3.rounds),
           std::chrono::duration<double>(t3 - t2).count());
    // the checks
    const double k50 = percentile(p1.kl, 50), k99 = percentile(p1.kl, 99),
                 kmax = *std::max_element(p1.kl.begin(), p1.kl.end());
    const bool kl_ok = k50 <= 2 * y_p50 && k99 <= 2 * y_p99 && kmax <= 2 * y_max;
    const uint32_t * top = reinterpret_cast<const uint32_t *>(ref_top.data());
    std::vector<int32_t> diff;
    for (int32_t p = 0; p < N; ++p) {
        if (p1.argmax[size_t(p)] != static_cast<int32_t>(top[size_t(4) * p])) {
            diff.push_back(p);
        }
    }
    const bool top_ok = double(diff.size()) <= 2 * y_top1;
    int32_t det_bad = 0, bi_bad = 0;
    for (int32_t p = 0; p < N; ++p) {
        det_bad += p1.hash[size_t(p)] != p2.hash[size_t(p)] || p1.argmax[size_t(p)] != p2.argmax[size_t(p)] ? 1 : 0;
    }
    for (int32_t p = 0; p < n_bi; ++p) {
        bi_bad += p1.hash[size_t(p)] != p3.hash[size_t(p)] || p1.argmax[size_t(p)] != p3.argmax[size_t(p)] ? 1 : 0;
    }
    const rt::ledger & L = t.slot();
    const bool led_ok = L.failures == 0;
    // the outputs
    FILE * jf = fopen((a.out_dir + "/model-test.json").c_str(), "w");
    FILE * tf = fopen((a.out_dir + "/kl-per-position.tsv").c_str(), "w");
    FILE * df = fopen((a.out_dir + "/top1-differences.tsv").c_str(), "w");
    FILE * hf = fopen((a.out_dir + "/logit-hashes.tsv").c_str(), "w");
    if (jf == nullptr || tf == nullptr || df == nullptr || hf == nullptr) {
        fprintf(stderr, "model-test: cannot write the outputs\n");
        return 1;
    }
    fprintf(tf, "pos\tkl_ref_fnfl\n");
    for (size_t i = 0; i < p1.kl.size(); ++i) {
        fprintf(tf, "%zu\t%.9g\n", i * 8 + 7, p1.kl[i]);
    }
    fprintf(df, "pos\tfnfl\tref\tref_margin\tref_top2\n");
    for (const int32_t p : diff) {
        float v1 = 0, v2 = 0;
        std::memcpy(&v1, &top[size_t(4) * p + 1], 4);
        std::memcpy(&v2, &top[size_t(4) * p + 3], 4);
        fprintf(df, "%d\t%d\t%u\t%.6g\t%u\n", p, p1.argmax[size_t(p)], top[size_t(4) * p], double(v1) - double(v2),
                top[size_t(4) * p + 2]);
    }
    fprintf(hf, "pos\tpass1\tpass2\tb1\targmax\n");
    for (int32_t p = 0; p < N; ++p) {
        char b1[24] = "-";
        if (p < n_bi) {
            snprintf(b1, sizeof(b1), "%016llx", static_cast<unsigned long long>(p3.hash[size_t(p)]));
        }
        fprintf(hf, "%d\t%016llx\t%016llx\t%s\t%d\n", p, static_cast<unsigned long long>(p1.hash[size_t(p)]),
                static_cast<unsigned long long>(p2.hash[size_t(p)]), b1, p1.argmax[size_t(p)]);
    }
    fprintf(jf,
            "{\"positions\": %d, \"rounds\": [%lld, %lld, %lld], \"captures\": %d,\n"
            " \"kl\": {\"p50\": %.9g, \"p99\": %.9g, \"max\": %.9g},\n"
            " \"yardstick\": {\"kl_p50\": %.9g, \"kl_p99\": %.9g, \"kl_max\": %.9g, \"top1_differ\": %.0f},\n"
            " \"top1_differ\": %zu, \"determinism_positions_differ\": %d, \"batch_invariance_positions_differ\": %d,\n"
            " \"ledger_checks\": %lld, \"ledger_failures\": %lld,\n"
            " \"pass\": {\"kl\": %s, \"top1\": %s, \"determinism\": %s, \"batch_invariance\": %s, \"ledger\": %s}}\n",
            N, static_cast<long long>(p1.rounds), static_cast<long long>(p2.rounds), static_cast<long long>(p3.rounds),
            t.captures(), k50, k99, kmax, y_p50, y_p99, y_max, y_top1, diff.size(), det_bad, bi_bad,
            static_cast<long long>(L.checks), static_cast<long long>(L.failures), kl_ok ? "true" : "false",
            top_ok ? "true" : "false", det_bad == 0 ? "true" : "false", bi_bad == 0 ? "true" : "false",
            led_ok ? "true" : "false");
    fclose(jf);
    fclose(tf);
    fclose(df);
    fclose(hf);
    printf("model-test: KL(reference || FNFL) p50 %.6g p99 %.6g max %.6g  (2x yardstick %.6g / %.6g / %.6g)  %s\n", k50,
           k99, kmax, 2 * y_p50, 2 * y_p99, 2 * y_max, kl_ok ? "ok" : "FAIL");
    printf("model-test: top-1 differs at %zu positions (limit %.0f)  %s\n", diff.size(), 2 * y_top1,
           top_ok ? "ok" : "FAIL");
    printf("model-test: determinism: %d of %d positions' logits differ between pass 1 and pass 2  %s\n", det_bad, N,
           det_bad == 0 ? "ok" : "FAIL");
    printf("model-test: batch invariance: %d of %d positions' logits at B = 1 differ from the cycled pass  %s\n",
           bi_bad, n_bi, bi_bad == 0 ? "ok" : "FAIL");
    printf("model-test: ledger: %lld checks, %lld failures (last pass); error words 0 in every round; graph captures "
           "%d\n",
           static_cast<long long>(L.checks), static_cast<long long>(L.failures), t.captures());
    const bool pass = kl_ok && top_ok && det_bad == 0 && bi_bad == 0 && led_ok;
    printf("model-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

namespace {

// model-dry's draft rounds (DR1 Stage C): forced rounds, B cycling 1..5, each batch caught up whole or as 1 + (B - 1)
// with draft steps of d = 1..4 between (SP2); zero weights make every logit 0, so every proposal is the subset's
// lowest ID (the greedy tie rule). Then the ledger's refusals and (graphs) the draft's last cells.
bool dry_draft(rt::target & t, const rt::draft_options & o, const std::vector<int32_t> & ids, bool eager,
               int32_t cells) {
    std::string err;
    std::vector<int32_t> subset;
    if (!md::read_draft_subset(o.subset, subset, err)) {
        fprintf(stderr, "model-dry: %s\n", err.c_str());
        return false;
    }
    const int32_t low = *std::min_element(subset.begin(), subset.end());
    const int32_t n = eager ? 12 : 40;
    t.set_prompt(std::vector<int32_t>(ids.begin(), ids.begin() + n + 8));
    if (!t.reset(err)) {
        fprintf(stderr, "model-dry: %s\n", err.c_str());
        return false;
    }
    t.set_prompt(std::vector<int32_t>(ids.begin(), ids.begin() + n + 8));
    int64_t steps = 0, catchups = 0, wrong = 0;
    bool ok = true;
    rt::round_result r;
    const auto step = [&](int32_t d) {
        ok = ok && t.draft_step(d, r, err);
        for (int32_t j = 0; ok && j < d; ++j) {
            wrong += r.argmax[0][j] != low || r.argmax[1][j] != low ? 1 : 0;
        }
        steps += ok ? 1 : 0;
    };
    const auto catchup = [&](int32_t rows) {
        ok = ok && t.draft_catchup(rows, r, err);
        catchups += ok ? 1 : 0;
    };
    for (int32_t pos = 0, ci = 0; ok && pos < n; ++ci) {
        const int32_t B = std::min(1 + ci % 5, n - pos);
        ok = t.forced(B, r, err);
        if (ci % 10 < 5 || B == 1) {
            catchup(B);
        } else {
            catchup(1);
            step(1 + ci % 4);
            catchup(B - 1);
        }
        step(1 + ci % 4);
        pos += B;
    }
    printf("model-dry: draft: %d positions, %lld catch-ups and %lld steps (%s); %lld proposals not ID %d; draft N %lld"
           "\n",
           n, static_cast<long long>(catchups), static_cast<long long>(steps), ok ? "ok" : err.c_str(),
           static_cast<long long>(wrong), low, static_cast<long long>(t.draft_n()));
    ok = ok && wrong == 0 && t.draft_n() == n;
    // refusals: a catch-up beyond N, a step of d = 5 (SP2), and a catch-up after the target moved past the draft
    rt::round_result rr;
    const bool r1 = !t.draft_catchup(1, rr, err);
    const bool r2 = !t.draft_step(5, rr, err);
    const bool r3 = ok && t.forced(1, rr, err) && t.forced(1, rr, err) && !t.draft_catchup(1, rr, err);
    printf("model-dry: draft refusals: beyond N %s, d = 5 %s, behind the last batch %s (%s)\n", r1 ? "yes" : "NO",
           r2 ? "yes" : "NO", r3 ? "yes" : "NO", err.c_str());
    ok = ok && r1 && r2 && r3;
    if (ok && !eager) {
        // the draft's last cells: a catch-up of 5 ending at the last cell, a step of 4 ending there, one past refused
        const int32_t mc = t.draft(0).max_cells;
        const bool fits = mc <= cells && size_t(mc) <= ids.size();
        t.set_prompt(ids);
        ok = fits && t.set_depth(mc, err) && t.draft_set_depth(mc - 5, err) && t.draft_catchup(5, rr, err) &&
             t.draft_set_depth(mc - 4, err) && t.draft_step(4, rr, err);
        const bool past = ok && t.draft_set_depth(mc - 3, err) && !t.draft_step(4, rr, err);
        printf("model-dry: draft's last cells (%d): catch-up 5 and step 4 %s; a step past them refused %s\n", mc,
               ok ? "ok" : err.c_str(), past ? "yes" : "NO");
        ok = ok && past;
    }
    return ok;
}

} // namespace

bool fnfl_load_dry(const fnfl_target_args & a, int32_t n_cells, md::model & m) {
    return load_dry(a, n_cells, m);
}

bool fnfl_load(const fnfl_target_args & a, int32_t n_cells, md::model & m, const char * tool) {
    return load(a, n_cells, m, tool);
}

bool fnfl_fill_synthetic(md::model & m, rt::target & t, int32_t depth) {
    return fill_synthetic(m, t, depth);
}

bool fnfl_draft_settings(const fnfl_target_args & a, bool required, const char * tool, rt::draft_options & o) {
    o = rt::draft_options();
    o.on = required || a.draft_attention_given;
    o.attention = a.draft_attention;
    o.subset = a.draft_subset;
    std::string err;
    if (!rt::check_draft_options(o, err)) {
        fprintf(stderr, "%s: %s\n", tool, err.c_str());
        return false;
    }
    if (o.on) {
        printf("%s: settings: draft on; draft attention %s (DR1's switch, until QA-2 is decided; sparse: SP2's, until "
               "its disposition); subset %s\n",
               tool, o.attention.c_str(), o.subset.c_str());
    } else {
        printf("%s: settings: draft off\n", tool);
    }
    return true;
}

namespace {

// model-dry's sample mode (SM1): the sampler in the round frame with zero weights (every logit 0)
int model_dry_sample(const fnfl_target_args & a, md::model & m, const std::vector<int32_t> & ids) {
    rt::target_options o;
    o.model_path = a.model_path;
    o.timeout_s = 30.0;
    o.sample = true;
    o.sampling = a.sampling; // TK1: the command line's --temp, --top-k, --top-p and --min-p
    o.sampling.seed = 4242;
    o.sampling.penalty_repeat = 1.3f; // the apply node and the window in the graph
    o.sampling.penalty_last_n = 16;
    rt::target t;
    std::string err;
    if (!t.init(m, o, err)) {
        fprintf(stderr, "model-dry: sample init: %s\n", err.c_str());
        return 1;
    }
    const std::vector<int32_t> prompt(ids.begin(), ids.begin() + 6);
    // a prompt, the bootstrap and 12 sampled generate rounds, twice from reset: the tokens bitwise equal
    std::vector<int32_t> toks[2];
    rt::round_result r;
    bool ok = true;
    for (int32_t run = 0; run < 2 && ok; ++run) {
        ok = t.reset(err);
        t.set_prompt(prompt);
        ok = ok && t.forced(4, r, err) && t.forced(2, r, err) && t.bootstrap(r, err);
        toks[run].push_back(r.out[0]);
        for (int32_t i = 0; ok && i < 12; ++i) {
            ok = t.generate(r, err);
            toks[run].push_back(r.out[0]);
        }
    }
    printf("model-dry: sampled (top-k %d, top-p %.2f, min-p %.2f, T %.1f, repeat %.1f over %d): %s; tokens:",
           o.sampling.top_k, o.sampling.top_p, o.sampling.min_p, o.sampling.temp, o.sampling.penalty_repeat,
           o.sampling.penalty_last_n, ok ? "ok" : err.c_str());
    for (int32_t x : toks[0]) {
        printf(" %d", x);
    }
    const bool same = ok && toks[0] == toks[1];
    printf("\nmodel-dry: two runs from reset bitwise equal: %s\n", same ? "yes" : "NO");
    // the closing check of the last decision
    bool checked = false;
    ok = ok && t.close_check(checked, err);
    printf("model-dry: closing check: %s\n", ok && checked ? "checked" : "NOT CHECKED");
    ok = ok && checked;
    // a forced mismatch: card 0's record flipped; the next round fails with err_hash, commits nothing, and the slot
    // is invalidated (FC-RD2)
    const size_t L0 = t.slot().tok.size();
    const int64_t N0 = t.slot().n;
    const uint32_t cur0 = t.slot().cur;
    ok = ok && t.test_flip_record(0, err);
    rt::round_result rf;
    const bool failed = ok && !t.generate(rf, err);
    const bool hash_bit = ((rf.err[0][0] | rf.err[1][0]) & fn::xchg::err_hash) != 0u;
    const bool nothing = t.slot().tok.size() == L0 && t.slot().n == N0 && t.slot().cur == cur0 && rf.recovered;
    printf("model-dry: flipped hash on card 0: round fails %s (%s), err_hash on card 0 %s card 1 %s, nothing "
           "committed (L %zu N %lld cur %u) %s, slot invalidated %s\n",
           failed ? "yes" : "NO", err.c_str(), (rf.err[0][0] & fn::xchg::err_hash) != 0u ? "yes" : "no",
           (rf.err[1][0] & fn::xchg::err_hash) != 0u ? "yes" : "no", t.slot().tok.size(),
           static_cast<long long>(t.slot().n), t.slot().cur, nothing ? "yes" : "NO", t.invalid() ? "yes" : "NO");
    rt::round_result rn;
    const bool refused = !t.generate(rn, err);
    printf("model-dry: the next round refused: %s (%s)\n", refused ? "yes" : "NO", err.c_str());
    ok = ok && failed && hash_bit && nothing && t.invalid() && refused;
    // after reset: a fresh prompt, then a flipped hash caught by the closing check
    ok = ok && t.reset(err);
    t.set_prompt(prompt);
    ok = ok && t.forced(4, r, err) && t.forced(2, r, err) && t.bootstrap(r, err) && t.generate(r, err) &&
         t.test_flip_record(1, err);
    const bool caught = ok && !t.close_check(checked, err) && t.invalid();
    printf("model-dry: flipped hash on card 1 caught by the closing check: %s (%s)\n", caught ? "yes" : "NO",
           err.c_str());
    ok = ok && caught && same;
    printf("model-dry: sample: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}

} // namespace

int fnfl_model_dry(const fnfl_target_args & a) {
    if (a.mode == "spec") {
        // SP1's C1: spec's segments with zero weights (verify rounds, the T > 0 draft step, both targets' graphs)
        fnfl_target_args b = a;
        b.mode = "dry";
        if (b.run_settings.empty()) {
            // spec takes its T = 1 settings from the first run setting (TK1); give it the spec command's defaults
            fnfl_target_args::run_setting s;
            s.sampling.temp = 1.0f;
            s.sampling.penalty_last_n = 0;
            b.run_settings.push_back(s);
        }
        return fnfl_spec(b);
    }
    const bool eager = a.mode == "eager";
    const bool sample = a.mode == "sample";
    std::vector<int32_t> ids;
    if (!read_ids(a.ids, ids)) {
        return 1;
    }
    // the deepest rows' ledger beyond the file repeats it from its start (synthetic positions)
    for (size_t i = 0; ids.size() < size_t(198720) + 8; ++i) {
        ids.push_back(ids[i]);
    }
    const int32_t cells = eager ? 8192 : a.n_cells;
    auto m = std::make_unique<md::model>();
    if (!load_dry(a, cells, *m)) {
        return 1;
    }
    if (sample) {
        return model_dry_sample(a, *m, ids);
    }
    rt::target_options o;
    o.model_path = a.model_path;
    o.eager = eager;
    o.timeout_s = eager ? 1800.0 : 30.0;
    o.graph_nodes = eager ? std::string() : a.out_dir; // SR1: each captured graph's node list
    if (!fnfl_draft_settings(a, false, "model-dry", o.draft)) {
        return 1;
    }
    rt::target t;
    std::string err;
    if (!t.init(*m, o, err)) {
        fprintf(stderr, "model-dry: init: %s\n", err.c_str());
        return 1;
    }
    printf("model-dry: zero weights, %d cells, %s, %d graph captures\n", cells, eager ? "eager" : "graphs",
           t.captures());
    double rtt_ns = 0.0;
    if (!clock_offset(*m, g_offset_ns, rtt_ns)) {
        fprintf(stderr, "model-dry: clock offset\n");
        return 1;
    }
    // forced rounds, B cycling, twice: the logits' hashes equal
    const int32_t N = eager ? 16 : 64;
    float * h = nullptr;
    if (!cuda_ok(cudaMallocHost(&h, size_t(2) * fn::round::max_rows * VC * 4), "pinned")) {
        return 1;
    }
    std::unique_ptr<float, decltype(&cudaFreeHost)> hold(h, cudaFreeHost);
    pass_out p1, p2;
    if (!run_pass(t, ids, N, {1, 2, 3, 4, 5}, nullptr, nullptr, nullptr, p1, h) ||
        !run_pass(t, ids, N, {1, 2, 3, 4, 5}, nullptr, nullptr, nullptr, p2, h)) {
        return 1;
    }
    int32_t bad = 0;
    for (int32_t i = 0; i < N; ++i) {
        bad += p1.hash[size_t(i)] != p2.hash[size_t(i)] || p1.argmax[size_t(i)] != 0 ? 1 : 0;
    }
    printf("model-dry: %d positions twice, %lld rounds each; %d differ or are not token 0\n", N,
           static_cast<long long>(p1.rounds), bad);
    // the bootstrap and generate rounds after a 6-token prompt
    t.set_prompt(std::vector<int32_t>(ids.begin(), ids.begin() + 6));
    if (!t.reset(err)) {
        fprintf(stderr, "model-dry: %s\n", err.c_str());
        return 1;
    }
    t.set_prompt(std::vector<int32_t>(ids.begin(), ids.begin() + 6));
    rt::round_result r;
    bool ok = t.forced(4, r, err) && t.forced(2, r, err) && t.bootstrap(r, err);
    for (int32_t i = 0; ok && i < 5; ++i) {
        ok = t.generate(r, err, true);
        printf("model-dry: generate round %llu: pass %.1f / %.1f us, skew %.2f us, begin to start %.1f / %.1f us, host "
               "wait %.1f us\n",
               static_cast<unsigned long long>(r.round), double(r.t_end[0] - r.t_start[0]) / 1000.0,
               double(r.t_end[1] - r.t_start[1]) / 1000.0,
               (double(r.t_start[1]) - g_offset_ns - double(r.t_start[0])) / 1000.0,
               double(r.t_start[0] - r.t_begin[0]) / 1000.0, double(r.t_start[1] - r.t_begin[1]) / 1000.0,
               r.host_wait_s * 1e6);
    }
    if (!ok) {
        fprintf(stderr, "model-dry: generate: %s\n", err.c_str());
        return 1;
    }
    printf("model-dry: prompt 6, bootstrap and 5 generate rounds: L %zu N %lld U %lld; ledger checks %lld failures "
           "%lld\n",
           t.slot().tok.size(), static_cast<long long>(t.slot().n), static_cast<long long>(t.slot().u),
           static_cast<long long>(t.slot().checks), static_cast<long long>(t.slot().failures));
    if (ok && o.draft.on) {
        ok = dry_draft(t, o.draft, ids, eager, cells);
    }
    // the deepest depth the slot holds: every B at its end
    if (!eager) {
        t.set_prompt(ids);
        for (int32_t B = 1; ok && B <= fn::round::max_rows; ++B) {
            ok = t.set_depth(cells - B, err) && t.forced(B, r, err, true, true);
            printf("model-dry: depth %d B %d: %s; pass %.1f / %.1f us, skew %.2f us, begin to start %.1f / %.1f us, "
                   "errors %u %u %u %u / %u %u %u %u\n",
                   cells - B, B, ok ? "ok" : err.c_str(), double(r.t_end[0] - r.t_start[0]) / 1000.0,
                   double(r.t_end[1] - r.t_start[1]) / 1000.0,
                   (double(r.t_start[1]) - g_offset_ns - double(r.t_start[0])) / 1000.0,
                   double(r.t_start[0] - r.t_begin[0]) / 1000.0, double(r.t_start[1] - r.t_begin[1]) / 1000.0,
                   r.err[0][0], r.err[0][1], r.err[0][2], r.err[0][3], r.err[1][0], r.err[1][1], r.err[1][2],
                   r.err[1][3]);
        }
        // a round past the capacity is refused before launch
        rt::round_result rr;
        const bool refused = t.set_depth(cells - 1, err) && !t.forced(2, rr, err);
        printf("model-dry: a round past the capacity refused before launch: %s (%s)\n", refused ? "yes" : "NO",
               err.c_str());
        ok = ok && refused;
    }
    printf("model-dry: %s\n", ok && bad == 0 ? "ok" : "FAILED");
    return ok && bad == 0 ? 0 : 1;
}

int fnfl_model_bench(const fnfl_target_args & a) {
    const bool nsys = a.mode == "nsys";
    if (!nsys && a.mode != "all") {
        fprintf(stderr, "model-bench: mode is all or nsys\n");
        return 2;
    }
    std::vector<int32_t> ids;
    const int32_t deepest = 196608;
    if (!read_ids(a.ids, ids) || int32_t(ids.size()) < 4096) {
        fprintf(stderr, "model-bench: %s holds too few tokens\n", a.ids.c_str());
        return 1;
    }
    // TR1: a real point's history and replay rows are the file's own tokens, never the repetition below
    if (nsys && a.bench_real && int64_t(ids.size()) < int64_t(a.bench_depth) + a.bench_b) {
        fprintf(stderr, "model-bench nsys real: %s holds %zu tokens; depth %d at B %d needs %lld\n", a.ids.c_str(),
                ids.size(), a.bench_depth, a.bench_b, static_cast<long long>(a.bench_depth) + a.bench_b);
        return 1;
    }
    // the synthetic depths' rows beyond the file take its tokens from its start
    for (size_t i = 0; ids.size() < size_t(deepest) + 8; ++i) {
        ids.push_back(ids[i]);
    }
    ids.resize(size_t(deepest) + 8);
    auto m = std::make_unique<md::model>();
    if (!load(a, a.n_cells, *m, "model-bench")) {
        return 1;
    }
    rt::target_options o;
    o.model_path = a.model_path;
    rt::target t;
    std::string err;
    if (!t.init(*m, o, err)) {
        fprintf(stderr, "model-bench: init: %s\n", err.c_str());
        return 1;
    }
    printf("model-bench: %d graph captures (one per card per B); no capture after this line\n", t.captures());
    double rtt_ns = 0.0;
    if (!clock_offset(*m, g_offset_ns, rtt_ns)) {
        fprintf(stderr, "model-bench: clock offset\n");
        return 1;
    }
    FILE * jf = fopen((a.out_dir + "/model-bench.jsonl").c_str(), "w");
    if (jf == nullptr) {
        return 1;
    }
    bool ok = true;
    // nsys (SF1): one point; a real history at depths up to all's 4,096 real positions, a synthetic one above;
    // real (TR1): a real history at any depth
    const bool nsys_real = nsys && (a.bench_real || a.bench_depth <= 4096);
    const int32_t real_depth = nsys_real ? a.bench_depth : 4096;
    if (!nsys || nsys_real) {
        // the real history: a forced pass over the first real_depth tokens, B cycling 1..4
        t.set_prompt(ids);
        int32_t pos = 0, ci = 0;
        const auto th = std::chrono::steady_clock::now();
        while (ok && pos < real_depth) {
            const int32_t B = std::min(1 + ci % 4, real_depth - pos);
            ++ci;
            rt::round_result r;
            ok = t.forced(B, r, err);
            pos += B;
        }
        if (!ok) {
            fprintf(stderr, "model-bench: history: %s\n", err.c_str());
            return 1;
        }
        printf("model-bench: real history of %d positions in %.1f s\n", real_depth,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - th).count());
        ok = warm(t, real_depth, 2.0);
        if (nsys) {
            ok = ok && cuda_ok(cudaProfilerStart(), "profiler start") &&
                 bench_point(*m, t, jf, real_depth, a.bench_b, true, 3, 20, "nsys") &&
                 cuda_ok(cudaProfilerStop(), "profiler stop");
        }
        for (int32_t B = 1; ok && !nsys && B <= 4; ++B) {
            ok = bench_point(*m, t, jf, 4096, B, true, 3, 20, "real");
        }
    }
    if (ok && !nsys_real) {
        const auto tf = std::chrono::steady_clock::now();
        ok = t.reset(err) && fill_synthetic(*m, t, deepest);
        t.set_prompt(ids);
        printf("model-bench: synthetic history in %.1f s\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - tf).count());
    }
    const std::vector<int32_t> depths = nsys_real ? std::vector<int32_t>{}
                                        : nsys    ? std::vector<int32_t>{a.bench_depth}
                                                  : std::vector<int32_t>{58581, 131072, 196608};
    for (const int32_t d : depths) {
        ok = ok && warm(t, d, 2.0);
        if (nsys) {
            ok = ok && cuda_ok(cudaProfilerStart(), "profiler start") &&
                 bench_point(*m, t, jf, d, a.bench_b, false, 3, 20, "nsys") &&
                 cuda_ok(cudaProfilerStop(), "profiler stop");
            continue;
        }
        for (int32_t B = 1; ok && B <= 4; ++B) {
            ok = bench_point(*m, t, jf, d, B, false, 3, 20, "synthetic");
        }
    }
    fclose(jf);
    double end_offset_ns = 0.0;
    if (ok && clock_offset(*m, end_offset_ns, rtt_ns)) {
        printf("model-bench: clock offset drift over the run %.0f ns\n", end_offset_ns - g_offset_ns);
    }
    printf("model-bench: %s; graph captures %d (none after the first round)\n", ok ? "done" : "FAILED", t.captures());
    return ok ? 0 : 1;
}

namespace {

bool readable(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    char c = 0;
    return f && f.read(&c, 1);
}

// PL1: the PLE table's file and offset, from the mapping that holds it
bool ple_table_file(const md::model & m, std::string & path, uint64_t & offset) {
    for (const auto & f : m.files) {
        if (m.ple_table >= f->data() && m.ple_table + m.ple_bytes <= f->data() + f->size()) {
            path = f->path();
            offset = static_cast<uint64_t>(m.ple_table - f->data());
            return true;
        }
    }
    return false;
}

// PL1: the table evicted when asked, and its page counts before and after, printed; false when the table stays warm
bool condition_table(const fnfl_target_args & a, const md::model & m) {
    std::string path, err;
    uint64_t off = 0;
    if (!ple_table_file(m, path, off)) {
        fprintf(stderr, "generate: the PLE table is in no mapped file\n");
        return false;
    }
    rt::page_counts c0, c1;
    if (!rt::page_cache_counts(m.ple_table, m.ple_bytes, path, off, c0, err) ||
        (a.evict_ple && !rt::page_cache_evict(m.ple_table, m.ple_bytes, path, off, err)) ||
        !rt::page_cache_counts(m.ple_table, m.ple_bytes, path, off, c1, err)) {
        fprintf(stderr, "generate: %s\n", err.c_str());
        return false;
    }
    printf("generate: table pages %lld: before %s mincore %lld cachestat %lld; after: mincore %lld cachestat "
           "%lld\n",
           static_cast<long long>(c0.pages), a.evict_ple ? "eviction" : "the run (no eviction)",
           static_cast<long long>(c0.mapping_res), static_cast<long long>(c0.file_cached),
           static_cast<long long>(c1.mapping_res), static_cast<long long>(c1.file_cached));
    constexpr int64_t void_pages = 1024; // PL1: a run that starts warm is void
    if (a.evict_ple && (c1.mapping_res > void_pages || c1.file_cached > void_pages)) {
        fprintf(stderr, "generate: VOID: the table starts warm (more than %lld pages cached)\n",
                static_cast<long long>(void_pages));
        return false;
    }
    return true;
}

} // namespace

int fnfl_generate(const fnfl_target_args & a) {
    if (a.evict_ple && a.ple_table == rt::ple_residency::full) {
        // PN1: a pinned table cannot be evicted, so PL1's cold start cannot be made
        fprintf(stderr, "generate: --evict-ple needs --ple-table lazy (the full table is pinned and stays resident)\n");
        return 2;
    }
    struct rusage ru0 = {};
    getrusage(RUSAGE_SELF, &ru0);
    std::vector<int32_t> prompt;
    if (!read_ids(a.ids, prompt)) {
        return 1;
    }
    printf("generate: settings: prompt %zu tokens, %d tokens; evict_ple %s (default no)\n", prompt.size(), a.n_gen,
           a.evict_ple ? "yes" : "no");
    if (a.sample) {
        const fn::samp::options & so = a.sampling;
        printf("generate: sampling (SM1): temp %g top-k %d top-p %g min-p %g repeat %g last-n %d presence %g frequency "
               "%g seed %llu request %llu bias %zu; path %s\n",
               double(so.temp), so.top_k, double(so.top_p), double(so.min_p), double(so.penalty_repeat),
               so.penalty_last_n, double(so.penalty_present), double(so.penalty_freq),
               static_cast<unsigned long long>(so.seed), static_cast<unsigned long long>(so.request), so.bias.size(),
               fn::samp::path_of(so) == fn::samp::path::greedy ? "greedy"
               : fn::samp::path_of(so) == fn::samp::path::topk ? "top-k"
                                                               : "full");
    } else {
        printf("generate: greedy (V1), no sampler\n");
    }
    if (a.dry_paths) {
        // PL1 step 3: every input readable and the output directory writable, before a load is spent
        const std::string probe = a.out_dir + "/.dry-paths";
        FILE * pf = fopen(probe.c_str(), "w");
        const bool out_ok = pf != nullptr;
        if (pf != nullptr) {
            fclose(pf);
            std::remove(probe.c_str());
        }
        std::vector<std::string> shards;
        std::string err;
        bool ok = out_ok && fnfl_gguf_shard_paths(a.model_path, shards, err) && readable(a.draft_path);
        for (const std::string & sh : shards) {
            ok = ok && readable(sh);
            printf("generate: dry-paths: shard %s %s\n", sh.c_str(), readable(sh) ? "readable" : "NOT READABLE");
        }
        printf("generate: dry-paths: prompt %s readable (%zu tokens); draft %s %s; out %s %s\n", a.ids.c_str(),
               prompt.size(), a.draft_path.c_str(), readable(a.draft_path) ? "readable" : "NOT READABLE",
               a.out_dir.c_str(), out_ok ? "writable" : "NOT WRITABLE");
        printf("generate: dry-paths: --ple-table %s\n", rt::ple_residency_name(a.ple_table));
        printf("generate: dry-paths: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    auto m = std::make_unique<md::model>();
    if (!load(a, a.n_cells, *m, "generate")) {
        return 1;
    }
    rt::target_options o;
    o.model_path = a.model_path;
    o.sample = a.sample;
    o.sampling = a.sampling;
    rt::target t;
    std::string err;
    if (!t.init(*m, o, err)) {
        fprintf(stderr, "generate: init: %s\n", err.c_str());
        return 1;
    }
    double rtt_ns = 0.0;
    if (!clock_offset(*m, g_offset_ns, rtt_ns)) {
        fprintf(stderr, "generate: clock offset\n");
        return 1;
    }
    struct rusage ru_ready = {};
    getrusage(RUSAGE_SELF, &ru_ready);
    if (!condition_table(a, *m)) {
        return 1;
    }
    t.set_prompt(prompt);
    struct rusage rr0 = {}, rr1 = {}, rg0 = {};
    getrusage(RUSAGE_SELF, &rr0);
    const int64_t io0 = rt::storage_read_bytes();
    FILE * tr = fopen((a.out_dir + "/generate-rounds.tsv").c_str(), "w");
    if (tr == nullptr) {
        return 1;
    }
    fprintf(tr, "round\tkind\tB\tN_before\tN_after\tk\temitted\tgather_us\tupload_enqueue_us\tlaunch_us\twait_us\t"
                "upload_gpu_us0\tupload_gpu_us1\tpass_us0\tpass_us1\tskew_us\tdone_s\tdecide_us0\tdecide_us1\n");
    int32_t printed = 0;
    const auto log = [&](const rt::round_result & r, int32_t n_after, int32_t k) {
        const char * kinds[] = {"forced", "generate", "verify", "bootstrap"};
        // SM1: the decision in context, from after the logits to the finish node's timer (commit and finish included;
        // -1 without a sampler or in a bootstrap round)
        const auto dec = [&](int32_t c) {
            return r.t_logits[c] != 0 && r.t_logits[c] < r.t_end[c] && r.t_logits[c] > r.t_start[c]
                       ? double(r.t_end[c] - r.t_logits[c]) / 1000.0
                       : -1.0;
        };
        fprintf(
            tr,
            "%llu\t%s\t%d\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.3f\t%.6f\t%.2f\t%.2f\n",
            static_cast<unsigned long long>(r.round), kinds[r.kind], r.rows, r.n_before, n_after, k,
            r.n_out > 0 ? r.out[0] : -1, r.host_gather_s * 1e6, r.host_upload_s * 1e6, r.host_launch_s * 1e6,
            r.host_wait_s * 1e6, double(r.upload_ms[0]) * 1000, double(r.upload_ms[1]) * 1000,
            double(r.t_end[0] - r.t_start[0]) / 1000.0, double(r.t_end[1] - r.t_start[1]) / 1000.0,
            (double(r.t_start[1]) - g_offset_ns - double(r.t_start[0])) / 1000.0, r.t_done_s, dec(0), dec(1));
        if (printed < 8) {
            printf("generate: round %llu kind %s B %d N_before %d N_after %d k %d emitted %d\n",
                   static_cast<unsigned long long>(r.round), kinds[r.kind], r.rows, r.n_before, n_after, k,
                   r.n_out > 0 ? r.out[0] : -1);
            ++printed;
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    // the prompt through the decode path at B = 4, the last round taking the remainder
    while (t.slot().u > 0) {
        const int32_t B = static_cast<int32_t>(std::min<int64_t>(4, t.slot().u));
        rt::round_result r;
        if (!t.forced(B, r, err, true, true)) {
            fprintf(stderr, "generate: prompt round at %lld: %s\n", static_cast<long long>(t.slot().n), err.c_str());
            return 1;
        }
        log(r, r.n_before + B, B);
    }
    const auto t1 = std::chrono::steady_clock::now();
    getrusage(RUSAGE_SELF, &rg0);
    std::vector<int32_t> out;
    std::vector<double> done_at, gather_us;
    std::vector<double> decide_us[2], pass_us[2]; // SM1: the decision in context, and the whole pass, per card
    rt::round_result rb;
    if (!t.bootstrap(rb, err)) {
        fprintf(stderr, "generate: bootstrap: %s\n", err.c_str());
        return 1;
    }
    printed = std::min(printed, 6); // the bootstrap and the first generate round are printed
    log(rb, rb.n_before, 0);
    out.push_back(rb.out[0]);
    while (int32_t(out.size()) < a.n_gen) {
        rt::round_result r;
        if (!t.generate(r, err, true)) {
            fprintf(stderr, "generate: round %zu: %s\n", out.size(), err.c_str());
            return 1;
        }
        log(r, r.n_before + 1, 1);
        out.push_back(r.out[0]);
        done_at.push_back(r.t_done_s);
        gather_us.push_back(r.host_gather_s * 1e6);
        for (int32_t c = 0; c < 2 && a.sample; ++c) {
            if (r.t_logits[c] > r.t_start[c] && r.t_logits[c] < r.t_end[c]) {
                decide_us[c].push_back(double(r.t_end[c] - r.t_logits[c]) / 1000.0);
                pass_us[c].push_back(double(r.t_end[c] - r.t_start[c]) / 1000.0);
            }
        }
    }
    const auto t2 = std::chrono::steady_clock::now();
    // SM1 (FC-RD2): the last decision's hash, compared by the closing exchange before the run returns
    bool checked = false;
    if (a.sample) {
        if (!t.close_check(checked, err)) {
            fprintf(stderr, "generate: %s\n", err.c_str());
            return 1;
        }
        printf("generate: decision hash: every round's record compared by the next decision; the last (round %llu) by "
               "the closing check: %s\n",
               static_cast<unsigned long long>(t.slot().n), checked ? "checked" : "NOT CHECKED");
    }
    getrusage(RUSAGE_SELF, &rr1);
    const int64_t io1 = rt::storage_read_bytes();
    fclose(tr);
    FILE * of = fopen((a.out_dir + "/generate-tokens.txt").c_str(), "w");
    if (of == nullptr) {
        return 1;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        fprintf(of, "%s%d", i == 0 ? "" : ",", out[i]);
    }
    fprintf(of, "\n");
    fclose(of);
    std::vector<double> itl_ms;
    for (size_t i = 1; i < done_at.size(); ++i) {
        itl_ms.push_back((done_at[i] - done_at[i - 1]) * 1e3);
    }
    const rt::ledger & L = t.slot();
    printf("generate: prompt %lld tokens in %.2f s; %zu tokens (bootstrap + %zu generate rounds) in %.2f s\n",
           static_cast<long long>(L.prompt_len), std::chrono::duration<double>(t1 - t0).count(), out.size(),
           out.size() - 1, std::chrono::duration<double>(t2 - t1).count());
    printf("generate: ledger L %zu N %lld U %lld pending %d O %lld A %lld; checks %lld, failures %lld\n", L.tok.size(),
           static_cast<long long>(L.n), static_cast<long long>(L.u), L.pending, static_cast<long long>(L.o),
           static_cast<long long>(L.a), static_cast<long long>(L.checks), static_cast<long long>(L.failures));
    printf("generate: inter-token ms (%zu intervals) median %.3f p90 %.3f max %.3f; host gather us (%zu generate "
           "rounds) median %.1f p90 %.1f max %.1f\n",
           itl_ms.size(), percentile(itl_ms, 50), percentile(itl_ms, 90), percentile(itl_ms, 100), gather_us.size(),
           percentile(gather_us, 50), percentile(gather_us, 90), percentile(gather_us, 100));
    printf("generate: page faults: load %ld minor / %ld major; prompt and generation %ld minor / %ld major (generation "
           "alone %ld / %ld); storage reads %.1f MB\n",
           ru_ready.ru_minflt - ru0.ru_minflt, ru_ready.ru_majflt - ru0.ru_majflt, rr1.ru_minflt - rr0.ru_minflt,
           rr1.ru_majflt - rr0.ru_majflt, rr1.ru_minflt - rg0.ru_minflt, rr1.ru_majflt - rg0.ru_majflt,
           io0 >= 0 && io1 >= 0 ? double(io1 - io0) / 1e6 : -1.0);
    if (a.sample) {
        printf("generate: decision in context (after the logits to finish, commit and finish included), %zu generate "
               "rounds: card 0 median %.2f p90 %.2f us, card 1 median %.2f p90 %.2f us; the pass (start to finish): "
               "card 0 median %.1f us, card 1 median %.1f us\n",
               decide_us[0].size(), percentile(decide_us[0], 50), percentile(decide_us[0], 90),
               percentile(decide_us[1], 50), percentile(decide_us[1], 90), percentile(pass_us[0], 50),
               percentile(pass_us[1], 50));
    }
    printf("generate: graph captures %d; tokens: ", t.captures());
    for (const int32_t x : out) {
        printf("%d ", x);
    }
    const bool good = L.failures == 0 && (!a.sample || checked);
    printf("\ngenerate: %s\n", good ? "done" : "FAILED");
    return good ? 0 : 1;
}
