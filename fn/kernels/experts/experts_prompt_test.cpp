#include "fn/kernels/experts/experts_prompt_test.h"

#include "fn/kernels/experts/experts_prompt.h"
#include "fn/kernels/experts/experts_share.h"
#include "fn/model/shapes.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/tools/test_dump.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fn::experts {
namespace {

constexpr int32_t D = fn::n_embd;         // 2560
constexpr int32_t NFF = fn::n_ff_expert;  // 640: both cards
constexpr int32_t NU = fn::n_expert_used; // 10
constexpr int32_t NE = fn::n_expert;      // 512

fn::quant::wfmt to_quant(fmt f) {
    using fn::quant::wfmt;
    static const wfmt q[11] = {wfmt::iq2_xxs, wfmt::iq2_xs, wfmt::iq2_s, wfmt::iq3_xxs, wfmt::iq3_s, wfmt::q2_0,
                               wfmt::iq4_nl,  wfmt::q8_0,   wfmt::q4_k,  wfmt::q5_k,    wfmt::q5_1}; // fmt's order
    return q[static_cast<int32_t>(f)];
}

// ME1: Q5_1 and Q8_0 are the Q4 candidate's down formats (Q8_0 in its down shape, load_synth's down)
bool is_down(fmt f) {
    return f == fmt::q2_0 || f == fmt::iq4_nl || f == fmt::q5_1 || f == fmt::q8_0;
}

// runs fn(i) for i in [0, n) on up to 12 threads
template <typename F> void parallel_for(int32_t n, const F & fn) {
    const int32_t nt = std::max<int32_t>(1, std::min<int32_t>(n, 12));
    std::atomic<int32_t> next{0};
    std::vector<std::thread> th;
    th.reserve(static_cast<size_t>(nt));
    for (int32_t t = 0; t < nt; ++t) {
        th.emplace_back([&]() {
            for (int32_t i = next++; i < n; i = next++) {
                fn(i);
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
}

double f16_to_double(uint16_t h) {
    return static_cast<double>(fn::quant::f16_to_f32(h));
}

uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

} // namespace

bool ep_read_fn4(const char * path, ep_fn4 & r) {
    FILE * fp = fopen(path, "r");
    if (fp == nullptr) {
        fprintf(stderr, "ep: cannot read %s\n", path);
        return false;
    }
    std::vector<std::map<int32_t, std::vector<int32_t>>> rows(fn::n_layer);
    int32_t d = 0, row = 0, layer = 0;
    while (fscanf(fp, "%d %d %d", &d, &row, &layer) == 3) {
        if (layer < 0 || layer >= fn::n_layer) {
            fclose(fp);
            return false;
        }
        std::vector<int32_t> & v = rows[size_t(layer)][row];
        for (int32_t k = 0; k < NU; ++k) {
            int32_t e = 0;
            if (fscanf(fp, "%d", &e) != 1 || e < 0 || e >= NE) {
                fclose(fp);
                fprintf(stderr, "ep: bad expert ID in routes\n");
                return false;
            }
            v.push_back(e);
        }
    }
    fclose(fp);
    r.by_layer.assign(fn::n_layer, {});
    r.n_rows = 1 << 30;
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        for (const auto & kv : rows[size_t(l)]) {
            r.by_layer[size_t(l)].insert(r.by_layer[size_t(l)].end(), kv.second.begin(), kv.second.end());
        }
        r.n_rows = std::min(r.n_rows, static_cast<int32_t>(r.by_layer[size_t(l)].size() / NU));
    }
    return r.n_rows >= ep_max_rows;
}

std::vector<int32_t> ep_window(const ep_fn4 & r, int32_t layer, int32_t start, int32_t rows) {
    const std::vector<int32_t> & v = r.by_layer[size_t(layer)];
    return std::vector<int32_t>(v.begin() + int64_t(start) * NU, v.begin() + (int64_t(start) + rows) * NU);
}

ep_sort ep_sort_pairs(const std::vector<int32_t> & ids) {
    ep_sort h;
    h.bounds.assign(NE + 1, 0);
    for (const int32_t e : ids) {
        ++h.bounds[size_t(e) + 1];
    }
    for (int32_t e = 0; e < NE; ++e) {
        h.bounds[size_t(e) + 1] += h.bounds[size_t(e)];
    }
    std::vector<int32_t> cur(h.bounds.begin(), h.bounds.end() - 1);
    h.sorted.assign(ids.size(), 0);
    for (size_t p = 0; p < ids.size(); ++p) {
        h.sorted[size_t(cur[size_t(ids[p])]++)] = static_cast<int32_t>(p);
    }
    return h;
}

namespace {

// ---- the cases ---------------------------------------------------------------------------------------------------
// FN4 windows as prompt rows; rows b with b % every == 0 are sampled (all 10 of their pairs) for the references.
struct case_cfg {
    int32_t rows, layer, start, every, card;
};
const case_cfg k_cases[] = {{128, 5, 100, 1, 0}, {128, 5, 100, 1, 1}, {512, 17, 1000, 4, 0}, {1536, 33, 2000, 8, 0}};

// the deployed formats ("all"), then ME1's Q4 candidate's ("q4": gate/up Q4_K, Q5_K; down Q5_1, Q8_0)
const fmt k_gu[7] = {fmt::iq2_xxs, fmt::iq2_xs, fmt::iq2_s, fmt::iq3_xxs, fmt::iq3_s, fmt::q4_k, fmt::q5_k};
const fmt k_dn[4] = {fmt::q2_0, fmt::iq4_nl, fmt::q5_1, fmt::q8_0};
constexpr int32_t n_fmt = 11;

bool is_q4(fmt f) {
    return f == fmt::q4_k || f == fmt::q5_k || f == fmt::q5_1 || f == fmt::q8_0;
}
// f's index in k_dn
int32_t dn_index(fmt f) {
    return f == fmt::q2_0 ? 0 : f == fmt::iq4_nl ? 1 : f == fmt::q5_1 ? 2 : 3;
}

const char * lname(fmt f) {
    static const char * names[n_fmt] = {"iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "q2_0",
                                        "iq4_nl",  "q8_0",   "q4_k",  "q5_k",    "q5_1"};
    return names[static_cast<int32_t>(f)];
}

struct case_in {
    case_cfg cfg;
    std::vector<int32_t> ids;   // [R][10]
    std::vector<float> wts;     // [R][10]
    std::vector<uint16_t> xh;   // F16 [R][2560]: the producer's tile
    std::vector<double> xd;     // [R][2560]
    std::vector<int32_t> srows; // sampled rows
    int32_t touched = 0, max_rows = 0;
    int32_t n_sp() const { return static_cast<int32_t>(srows.size()) * NU; }
    int32_t sp_pair(int32_t i) const { return srows[size_t(i / NU)] * NU + i % NU; }
};

case_in make_inputs(const ep_fn4 & fr, const case_cfg & cfg, bool zero_x) {
    case_in c;
    c.cfg = cfg;
    c.ids = ep_window(fr, cfg.layer, cfg.start, cfg.rows);
    std::mt19937 rng(0x45503100u + 7u * static_cast<uint32_t>(cfg.rows));
    std::uniform_real_distribution<float> ud(0.05f, 1.0f);
    for (int32_t b = 0; b < cfg.rows; ++b) {
        float w[NU], sum = 0.0f;
        for (float & v : w) {
            v = ud(rng);
            sum += v;
        }
        for (const float v : w) {
            c.wts.push_back(v / sum);
        }
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(size_t(cfg.rows) * D);
    for (auto & v : x) {
        v = zero_x ? 0.0f : nd(rng);
    }
    // the FP16 tile the kernel reads, and the reference's input: the same FP16 values, exactly (in FP64)
    c.xh.assign(x.size(), 0);
    c.xd.assign(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        c.xh[i] = fn::quant::f32_to_f16(x[i]);
        c.xd[i] = f16_to_double(c.xh[i]);
    }
    for (int32_t b = 0; b < cfg.rows; b += cfg.every) {
        c.srows.push_back(b);
    }
    std::vector<int32_t> cnt(NE, 0);
    for (const int32_t e : c.ids) {
        ++cnt[size_t(e)];
    }
    for (const int32_t n : cnt) {
        c.touched += n > 0 ? 1 : 0;
        c.max_rows = std::max(c.max_rows, n);
    }
    return c;
}

// the down input: F16 GLU per pair (pair order); sampled pairs carry the GLU reference rounded to FP16, the others
// fixed pseudo-random values of the same size
std::vector<uint16_t> down_input(const case_in & c, const std::vector<double> & glu_ref) {
    const int32_t pairs = c.cfg.rows * NU;
    std::vector<uint16_t> g(size_t(pairs) * n_ff_card);
    for (int32_t p = 0; p < pairs; ++p) {
        for (int32_t ch = 0; ch < n_ff_card; ++ch) {
            const uint32_t h = mix(static_cast<uint32_t>(p) * 977u + static_cast<uint32_t>(ch));
            g[size_t(p) * n_ff_card + ch] = fn::quant::f32_to_f16((static_cast<float>(h % 2001u) - 1000.0f) * 0.0005f);
        }
    }
    for (int32_t i = 0; i < c.n_sp(); ++i) {
        const int32_t p = c.sp_pair(i);
        for (int32_t ch = 0; ch < n_ff_card; ++ch) {
            g[size_t(p) * n_ff_card + ch] =
                fn::quant::f32_to_f16(static_cast<float>(glu_ref[size_t(i) * n_ff_card + ch]));
        }
    }
    return g;
}

std::string ref_name(const std::string & dir, const case_cfg & c, fmt gu, int32_t dn) {
    return dir + "/ep-f16-R" + std::to_string(c.rows) + "-c" + std::to_string(c.card) + "-" + lname(gu) +
           (dn < 0 ? std::string(".glu") : std::string("-") + lname(k_dn[dn]) + ".ref");
}

template <typename T> bool write_vec(const std::string & path, const std::vector<T> & v) {
    FILE * fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        return false;
    }
    const bool ok = fwrite(v.data(), sizeof(T), v.size(), fp) == v.size();
    return fclose(fp) == 0 && ok;
}
template <typename T> bool read_vec(const std::string & path, std::vector<T> & v, size_t n) {
    v.assign(n, T());
    FILE * fp = fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        fprintf(stderr, "ep-test: missing %s (run ep-prep)\n", path.c_str());
        return false;
    }
    const bool ok = fread(v.data(), sizeof(T), n, fp) == n && fgetc(fp) == EOF;
    fclose(fp);
    return ok;
}

// dequantized weights of the n_synth experts for one card: gate/up [16][640][2560] (gate 0..319, up 320..639), down
// [16][2560][320]
std::vector<float> dequant_card(const synth & s, int32_t card) {
    const fn::quant::wfmt q = to_quant(s.f);
    const bool dn = is_down(s.f);
    std::vector<float> out(size_t(n_synth) * (dn ? size_t(D) * n_ff_card : size_t(NFF) * D));
    parallel_for(n_synth, [&](int32_t i) {
        const uint8_t * e = s.expert(i);
        if (dn) {
            std::vector<float> row(NFF);
            for (int32_t n = 0; n < D; ++n) {
                fn::quant::decode_row(q, e + size_t(n) * s.row_bytes, row.data(), NFF);
                std::memcpy(out.data() + (size_t(i) * D + n) * n_ff_card, row.data() + size_t(card) * n_ff_card,
                            sizeof(float) * n_ff_card);
            }
        } else {
            for (int32_t ch = 0; ch < NFF; ++ch) {
                const int32_t src = ch < n_ff_card ? card * n_ff_card + ch : NFF + card * n_ff_card + ch - n_ff_card;
                fn::quant::decode_row(q, e + size_t(src) * s.row_bytes, out.data() + (size_t(i) * NFF + ch) * D, D);
            }
        }
    });
    return out;
}

// the down references of the sampled pairs (inputs in, FP64): partials at columns 4 i (n_sp x 640, F32) and the
// sampled rows' weighted sums (FP64, route order)
void down_refs(const case_in & c, const std::vector<float> & wdn, const std::vector<double> & in /* [pairs][320] */,
               std::vector<float> & part4, std::vector<double> & sum) {
    const int32_t nsp = c.n_sp();
    std::vector<double> full(size_t(nsp) * D);
    parallel_for(nsp, [&](int32_t i) {
        const int32_t p = c.sp_pair(i);
        const float * w = wdn.data() + size_t(c.ids[size_t(p)] % n_synth) * D * n_ff_card;
        const double * x = in.data() + size_t(p) * n_ff_card;
        for (int32_t n = 0; n < D; ++n) {
            double acc = 0.0;
            for (int32_t k = 0; k < n_ff_card; ++k) {
                acc += static_cast<double>(w[size_t(n) * n_ff_card + k]) * x[k];
            }
            full[size_t(i) * D + n] = acc;
        }
    });
    part4.assign(size_t(nsp) * (D / 4), 0.0f);
    for (int32_t i = 0; i < nsp; ++i) {
        for (int32_t n = 0; n < D; n += 4) {
            part4[size_t(i) * (D / 4) + n / 4] = static_cast<float>(full[size_t(i) * D + n]);
        }
    }
    const int32_t nr = static_cast<int32_t>(c.srows.size());
    sum.assign(size_t(nr) * D, 0.0);
    for (int32_t s = 0; s < nr; ++s) {
        const int32_t b = c.srows[size_t(s)];
        for (int32_t n = 0; n < D; ++n) {
            double acc = 0.0;
            for (int32_t k = 0; k < NU; ++k) {
                acc += static_cast<double>(c.wts[size_t(b) * NU + k]) * full[(size_t(s) * NU + k) * D + n];
            }
            sum[size_t(s) * D + n] = acc;
        }
    }
}

} // namespace
} // namespace fn::experts

int fnfl_ep_prep(const char * dir, const char * syn_dir, const char * routes, const char * formats) {
    using namespace fn::experts;
    ep_fn4 fr;
    const bool q4 = strcmp(formats, "q4") == 0;
    if ((!q4 && strcmp(formats, "all") != 0) || !ep_read_fn4(routes, fr)) {
        fprintf(stderr, "ep-prep: FORMATS is all (the deployed formats) or q4 (ME1's)\n");
        return 1;
    }
    std::map<int32_t, synth> syn;
    for (int32_t i = 0; i < n_fmt; ++i) {
        const fmt f = static_cast<fmt>(i);
        if (is_q4(f) == q4 && !load_synth(f, syn_dir, syn[i], true, f == fmt::q8_0)) {
            fprintf(stderr, "ep-prep: synthetic %s failed\n", fmt_name(f));
            return 1;
        }
    }
    for (const case_cfg & cfg : k_cases) {
        const case_in c = make_inputs(fr, cfg, false);
        printf("ep-prep: R %d card %d: layer %d rows %d.., %d experts touched (most rows %d), %d sampled pairs\n",
               cfg.rows, cfg.card, cfg.layer, cfg.start, c.touched, c.max_rows, c.n_sp());
        std::vector<float> wdn[4];
        for (int32_t d = 0; d < 4; ++d) {
            if (is_q4(k_dn[d]) == q4) {
                wdn[d] = dequant_card(syn[static_cast<int32_t>(k_dn[d])], cfg.card);
            }
        }
        for (const fmt gu : k_gu) {
            if (is_q4(gu) != q4) {
                continue;
            }
            const std::vector<float> wg = dequant_card(syn[static_cast<int32_t>(gu)], cfg.card);
            std::vector<double> glu(size_t(c.n_sp()) * n_ff_card);
            parallel_for(c.n_sp(), [&](int32_t i) {
                const int32_t p = c.sp_pair(i), b = p / NU;
                const float * w = wg.data() + size_t(c.ids[size_t(p)] % n_synth) * NFF * D;
                const double * x = c.xd.data() + size_t(b) * D;
                for (int32_t ch = 0; ch < n_ff_card; ++ch) {
                    double g = 0.0, u = 0.0;
                    for (int32_t k = 0; k < D; ++k) {
                        g += static_cast<double>(w[size_t(ch) * D + k]) * x[k];
                        u += static_cast<double>(w[size_t(n_ff_card + ch) * D + k]) * x[k];
                    }
                    glu[size_t(i) * n_ff_card + ch] = g / (1.0 + std::exp(-g)) * u;
                }
            });
            if (!write_vec(ref_name(dir, cfg, gu, -1), glu)) {
                return 1;
            }
            const std::vector<uint16_t> g16 = down_input(c, glu);
            std::vector<double> in(g16.size());
            for (size_t i = 0; i < g16.size(); ++i) {
                in[i] = f16_to_double(g16[i]);
            }
            for (int32_t d = 0; d < 4; ++d) {
                if (is_q4(k_dn[d]) != q4) {
                    continue;
                }
                std::vector<float> p4;
                std::vector<double> sum;
                down_refs(c, wdn[d], in, p4, sum);
                FILE * fp = fopen(ref_name(dir, cfg, gu, d).c_str(), "wb");
                bool ok = fp != nullptr;
                ok = ok && fwrite(p4.data(), 4, p4.size(), fp) == p4.size() &&
                     fwrite(sum.data(), 8, sum.size(), fp) == sum.size();
                ok = (fp != nullptr && fclose(fp) == 0) && ok;
                if (!ok) {
                    fprintf(stderr, "ep-prep: cannot write %s\n", ref_name(dir, cfg, gu, d).c_str());
                    return 1;
                }
            }
            printf("ep-prep:   %s: GLU and down references written\n", fmt_name(gu));
            fflush(stdout);
        }
    }
    return 0;
}

namespace fn::experts {
namespace {

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "ep-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// owns one device allocation of n bytes
struct dev_mem {
    uint8_t * p = nullptr;
    size_t n = 0;

    explicit dev_mem(size_t bytes) : n(bytes) {
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

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); } // the allocation is 256-byte aligned

    // copies and fills run on the kernels' own stream st and wait for it: st is non-blocking, so a legacy-stream
    // cudaMemcpy or cudaMemset is not ordered with its work
    bool put(const void * h, size_t bytes, cudaStream_t st) const {
        return p != nullptr && bytes <= n && cuda_ok(cudaMemcpyAsync(p, h, bytes, cudaMemcpyHostToDevice, st), "put") &&
               cuda_ok(cudaStreamSynchronize(st), "put");
    }
    bool get(void * h, size_t bytes, cudaStream_t st) const {
        return p != nullptr && bytes <= n && cuda_ok(cudaMemcpyAsync(h, p, bytes, cudaMemcpyDeviceToHost, st), "get") &&
               cuda_ok(cudaStreamSynchronize(st), "get");
    }
    bool fill(int v, cudaStream_t st) const {
        return p != nullptr && cuda_ok(cudaMemsetAsync(p, v, n, st), "memset") &&
               cuda_ok(cudaStreamSynchronize(st), "memset");
    }
};

struct err_acc {
    double max_err = 0.0, max_ref = 0.0;
    void add(double got, double ref) {
        const double e = std::fabs(got - ref);
        max_err = std::isnan(e) ? INFINITY : std::max(max_err, e);
        max_ref = std::max(max_ref, std::fabs(ref));
    }
    double rel() const { return max_ref > 0.0 ? max_err / max_ref : max_err; }
};

struct ctx {
    bool ok = true;
    int32_t n_checks = 0;
    bool full = true;
    std::string dir;
    std::map<int32_t, synth> syn;
    std::unique_ptr<dev_mem> grids;
    cudaStream_t st = nullptr;

    bool check(bool pass, const std::string & what) {
        ++n_checks;
        if (!pass) {
            fprintf(stderr, "ep-test: FAIL %s\n", what.c_str());
            ok = false;
        }
        return pass;
    }
    bool sync(const char * what) { return cuda_ok(cudaStreamSynchronize(st), what); }
};

struct route_dev {
    dev_mem offsets{size_t(NE + 1) * 4}, sorted{size_t(ep_max_pairs) * 4}, pos{size_t(ep_max_pairs) * 4},
        tasks{size_t(ep_max_tasks) * 16}, counters{32};
    ep_route view() const {
        return {offsets.as<int32_t>(), sorted.as<int32_t>(), pos.as<int32_t>(), tasks.as<int32_t>(),
                counters.as<int32_t>()};
    }
};

// expert e = synthetic e % n_synth, one copy of 512
bool packed(const synth & s, int32_t card, std::vector<uint8_t> & out) {
    std::vector<int32_t> map(NE);
    for (int32_t e = 0; e < NE; ++e) {
        map[size_t(e)] = e % n_synth;
    }
    return pack_card(s, card, map, 1, out);
}

std::string tag(const case_in & c) {
    return "R " + std::to_string(c.cfg.rows) + " card " + std::to_string(c.cfg.card);
}

// group, checked against the host sort; returns the downloaded pos (sorted position of each pair)
bool run_group(ctx & x, const case_in & c, int32_t task_rows, const dev_mem & ids, route_dev & rd,
               std::vector<int32_t> & pos) {
    const int32_t pairs = c.cfg.rows * NU;
    if (!x.check(ep_group(c.cfg.rows, task_rows, ids.as<int32_t>(), rd.view(), nullptr, x.st), "ep_group enqueue") ||
        !x.sync("group")) {
        return false;
    }
    std::vector<int32_t> off(NE + 1), sorted(static_cast<size_t>(pairs)), cnt(8),
        tasks(static_cast<size_t>(ep_max_tasks) * 4);
    pos.assign(size_t(pairs), 0);
    if (!rd.offsets.get(off.data(), off.size() * 4, x.st) || !rd.sorted.get(sorted.data(), sorted.size() * 4, x.st) ||
        !rd.pos.get(pos.data(), pos.size() * 4, x.st) || !rd.counters.get(cnt.data(), 32, x.st) ||
        !rd.tasks.get(tasks.data(), tasks.size() * 4, x.st)) {
        return false;
    }
    const ep_sort hs = ep_sort_pairs(c.ids);
    bool same = off == hs.bounds && sorted == hs.sorted;
    for (int32_t i = 0; i < pairs && same; ++i) {
        same = pos[size_t(sorted[size_t(i)])] == i;
    }
    // tasks: expert-major, each expert's pairs split into ceil(n / task_rows) contiguous runs of near-equal size
    int32_t want = 0, at = 0;
    bool tasks_ok = cnt[1] == 0;
    for (int32_t e = 0; e < NE && tasks_ok; ++e) {
        const int32_t n = hs.bounds[size_t(e) + 1] - hs.bounds[size_t(e)];
        const int32_t np = (n + task_rows - 1) / task_rows;
        int32_t first = hs.bounds[size_t(e)];
        for (int32_t k = 0; k < np && tasks_ok; ++k, ++at) {
            const int32_t * t = tasks.data() + size_t(at) * 4;
            tasks_ok = t[0] == e && t[1] == first && t[2] >= 1 && t[2] <= task_rows && t[2] >= n / np;
            first += t[2];
        }
        tasks_ok = tasks_ok && first == hs.bounds[size_t(e) + 1];
        want += np;
    }
    tasks_ok = tasks_ok && cnt[0] == want;
    x.check(same, tag(c) + ": group order, offsets and positions equal the host's stable sort");
    x.check(tasks_ok, tag(c) + ": tasks cover each expert's pairs in runs of at most " + std::to_string(task_rows));
    return same && tasks_ok;
}

struct dn_out {
    std::vector<float> part; // [pairs][2560]
    std::vector<float> out;  // [R][2560]
};

// ep_down and ep_sum (all five column groups in one launch each), input gin (sorted F16 GLU rows)
bool run_down(ctx & x, const case_in & c, fmt f, const dev_mem & w, const dev_mem & gin, const route_dev & rd,
              const dev_mem & wts, dev_mem & part, dev_mem & out, dn_out & res) {
    const int32_t pairs = c.cfg.rows * NU;
    res.part.assign(size_t(pairs) * D, 0.0f);
    res.out.assign(size_t(c.cfg.rows) * D, 0.0f);
    if (!out.fill(0xff, x.st) || !part.fill(0xff, x.st)) {
        return false;
    }
    const ep_down_args da = {w.p, x.grids->p, gin.as<uint16_t>(), rd.view(), part.as<float>()};
    const ep_sum_args sa = {part.as<float>(), rd.view(), wts.as<float>(), nullptr, out.as<float>()};
    if (!x.check(ep_down(f, c.cfg.rows, da, x.st), tag(c) + ": ep_down enqueued") ||
        !x.check(ep_sum(c.cfg.rows, sa, x.st), tag(c) + ": ep_sum enqueued") || !x.sync("down/sum") ||
        !part.get(res.part.data(), size_t(pairs) * D * 4, x.st)) {
        return false;
    }
    const std::string t = "ep-dn-" + tag(c) + "-" + fmt_name(f);
    if (!out.get(res.out.data(), res.out.size() * 4, x.st)) {
        return false;
    }
    // PB3: the sum written as FP16 (out16, the P payload): bitwise the F32 sum rounded to nearest even, the error
    // word clear
    dev_mem o16(size_t(c.cfg.rows) * D * 2), ew(4);
    ep_sum_args s16 = sa;
    s16.out = nullptr;
    s16.out16 = o16.as<uint16_t>();
    s16.err = ew.as<uint32_t>();
    s16.err_bit = 1u << 26;
    std::vector<uint16_t> got(size_t(c.cfg.rows) * D);
    uint32_t ev_word = 1;
    if (!o16.fill(0xff, x.st) || !ew.fill(0, x.st) ||
        !x.check(ep_sum(c.cfg.rows, s16, x.st), tag(c) + ": ep_sum out16") || !x.sync("sum out16") ||
        !o16.get(got.data(), got.size() * 2, x.st) || !ew.get(&ev_word, 4, x.st)) {
        return false;
    }
    int64_t diff = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        diff += got[i] != fn::quant::f32_to_f16(res.out[i]);
    }
    x.check(diff == 0 && ev_word == 0, t + ": ep_sum's FP16 output is the F32 sum rounded (" + std::to_string(diff) +
                                           " differ), the error word clear");
    // PB3: the partials written as FP16 (part16): bitwise the F32 partials rounded to nearest even, the error word
    // clear; the sum read from them is the fmaf chain over those values in route order (replayed on the host, bitwise)
    dev_mem p16(size_t(pairs) * D * 2), o2(size_t(c.cfg.rows) * D * 4);
    ep_down_args d16 = da;
    d16.part = nullptr;
    d16.part16 = p16.as<uint16_t>();
    d16.err = ew.as<uint32_t>();
    d16.err_bit = 1u << 26;
    ep_sum_args s2 = sa;
    s2.part = nullptr;
    s2.part16 = p16.as<uint16_t>();
    s2.out = o2.as<float>();
    std::vector<uint16_t> ph(size_t(pairs) * D);
    std::vector<float> o2h(size_t(c.cfg.rows) * D), wh(static_cast<size_t>(pairs));
    std::vector<int32_t> posh(static_cast<size_t>(pairs));
    ev_word = 1;
    if (!p16.fill(0xff, x.st) || !o2.fill(0xff, x.st) || !ew.fill(0, x.st) ||
        !x.check(ep_down(f, c.cfg.rows, d16, x.st), tag(c) + ": ep_down part16") ||
        !x.check(ep_sum(c.cfg.rows, s2, x.st), tag(c) + ": ep_sum part16") || !x.sync("down/sum part16") ||
        !p16.get(ph.data(), ph.size() * 2, x.st) || !o2.get(o2h.data(), o2h.size() * 4, x.st) ||
        !ew.get(&ev_word, 4, x.st) || !rd.pos.get(posh.data(), posh.size() * 4, x.st) ||
        !wts.get(wh.data(), wh.size() * 4, x.st)) {
        return false;
    }
    int64_t pdiff = 0, sdiff = 0;
    for (size_t i = 0; i < ph.size(); ++i) {
        pdiff += ph[i] != fn::quant::f32_to_f16(res.part[i]);
    }
    for (int32_t b = 0; b < c.cfg.rows; ++b) {
        for (int32_t n = 0; n < D; ++n) {
            float acc = 0.0f;
            for (int32_t k = 0; k < NU; ++k) {
                const size_t q = size_t(b) * NU + size_t(k);
                acc = std::fmaf(wh[q], fn::quant::f16_to_f32(ph[size_t(posh[q]) * D + size_t(n)]), acc);
            }
            sdiff += std::memcmp(&acc, &o2h[size_t(b) * D + size_t(n)], 4) != 0;
        }
    }
    x.check(pdiff == 0 && sdiff == 0 && ev_word == 0, t + ": ep_down's FP16 partials are the F32 partials rounded (" +
                                                          std::to_string(pdiff) +
                                                          " differ), the sum from them the host's fmaf chain (" +
                                                          std::to_string(sdiff) + " differ), the error word clear");
    return fn::tools::test_dump(t + "-part", res.part.data(), res.part.size() * 4) &&
           fn::tools::test_dump(t + "-out", res.out.data(), res.out.size() * 4);
}

void report(const char * what, const std::string & t, const err_acc & ours, ctx & x) {
    printf("ep-test: %s %-40s ours %.6e (max|ref| %.3e)", t.c_str(), what, ours.rel(), ours.max_ref);
    const bool pass = ours.rel() <= 1e-3;
    printf("%s\n", pass ? "" : "  OVER");
    x.check(pass, t + " " + what + " within the bound");
}

void run_case(ctx & x, const case_in & c, const std::vector<fmt> & gus, const std::vector<fmt> & dns) {
    const int32_t R = c.cfg.rows, pairs = R * NU, card = c.cfg.card;
    const std::string t = tag(c);
    printf("ep-test: %s: layer %d rows %d.., %d experts touched (%d with no rows), most rows on one expert %d\n",
           t.c_str(), c.cfg.layer, c.cfg.start, c.touched, NE - c.touched, c.max_rows);
    dev_mem ids(c.ids.size() * 4), wts(c.wts.size() * 4), xh(c.xh.size() * 2), glu(size_t(pairs) * n_ff_card * 2),
        glu_f32(size_t(pairs) * n_ff_card * 4), gin(size_t(pairs) * n_ff_card * 2), part(size_t(pairs) * D * 4),
        out(size_t(R) * D * 4);
    route_dev rd;
    if (!ids.put(c.ids.data(), ids.n, x.st) || !wts.put(c.wts.data(), wts.n, x.st) ||
        !xh.put(c.xh.data(), xh.n, x.st)) {
        x.ok = false;
        return;
    }
    for (const fmt gu : gus) {
        const std::string tg = t + " " + fmt_name(gu);
        std::vector<int32_t> pos;
        if (!run_group(x, c, ep_task_rows(gu), ids, rd, pos)) {
            return;
        }
        std::vector<uint8_t> wh;
        if (!packed(x.syn[static_cast<int32_t>(gu)], card, wh)) {
            x.ok = false;
            return;
        }
        dev_mem w(wh.size());
        if (!w.put(wh.data(), wh.size(), x.st)) {
            x.ok = false;
            return;
        }
        // gate/up, twice
        std::vector<uint16_t> g16[2];
        std::vector<float> g32[2];
        for (int32_t run = 0; run < (x.full ? 2 : 1); ++run) {
            if (!glu.fill(0xff, x.st) || !glu_f32.fill(0xff, x.st)) {
                x.ok = false;
                return;
            }
            const ep_gate_up_args a = {w.p,       x.grids->p,         xh.as<uint16_t>(),
                                       rd.view(), glu.as<uint16_t>(), glu_f32.as<float>()};
            if (!x.check(ep_gate_up(gu, R, a, x.st), tg + ": ep_gate_up enqueued") || !x.sync("gate_up")) {
                return;
            }
            g16[run].assign(size_t(pairs) * n_ff_card, 0);
            g32[run].assign(size_t(pairs) * n_ff_card, 0.0f);
            if (!glu.get(g16[run].data(), glu.n, x.st) || !glu_f32.get(g32[run].data(), glu_f32.n, x.st)) {
                x.ok = false;
                return;
            }
        }
        if (x.full) {
            x.check(g16[0] == g16[1] && std::memcmp(g32[0].data(), g32[1].data(), glu_f32.n) == 0,
                    tg + ": second gate_up run bitwise identical");
        }
        bool f16_same = true;
        int32_t n_bad = 0, first_bad = -1;
        float glu_max = 0.0f; // the largest |GLU| of every pair: the F16 GLU must stay finite (C1b Amendment 4)
        for (size_t i = 0; i < g16[0].size(); ++i) {
            f16_same = f16_same && g16[0][i] == fn::quant::f32_to_f16(g32[0][i]);
            glu_max = std::isnan(g32[0][i]) ? INFINITY : std::max(glu_max, std::fabs(g32[0][i]));
            if (!std::isfinite(g32[0][i])) {
                first_bad = first_bad < 0 ? static_cast<int32_t>(i) : first_bad;
                ++n_bad;
            }
        }
        if (n_bad > 0) {
            const int32_t ps = first_bad / n_ff_card;
            int32_t e = -1;
            for (int32_t p = 0; p < pairs; ++p) {
                e = pos[size_t(p)] == ps ? c.ids[size_t(p)] : e;
            }
            fprintf(stderr, "ep-test: %s: %d non-finite F32 GLU values, first at position %d channel %d (expert %d)\n",
                    tg.c_str(), n_bad, ps, first_bad % n_ff_card, e);
        }
        x.check(f16_same, tg + ": F16 GLU = the F32 GLU rounded to nearest, every pair");
        printf("ep-test: %s largest |GLU| %.4e (F16 largest finite 65504)%s\n", tg.c_str(), glu_max,
               glu_max <= 65504.0f ? "" : "  OVER");
        x.check(glu_max <= 65504.0f, tg + ": largest |GLU| within F16's range");
        std::vector<double> ref;
        if (!read_vec(ref_name(x.dir, c.cfg, gu, -1), ref, size_t(c.n_sp()) * n_ff_card)) {
            x.ok = false;
            return;
        }
        err_acc eg;
        for (int32_t i = 0; i < c.n_sp(); ++i) {
            const int32_t p = c.sp_pair(i);
            for (int32_t ch = 0; ch < n_ff_card; ++ch) {
                eg.add(g32[0][size_t(pos[size_t(p)]) * n_ff_card + ch], ref[size_t(i) * n_ff_card + ch]);
            }
        }
        report("GLU (F32 before rounding)", tg, eg, x);

        // down: the reference's F16 GLU as input (sorted order)
        const std::vector<uint16_t> din = down_input(c, ref);
        std::vector<uint16_t> din_sorted(din.size());
        for (int32_t p = 0; p < pairs; ++p) {
            std::memcpy(din_sorted.data() + size_t(pos[size_t(p)]) * n_ff_card, din.data() + size_t(p) * n_ff_card,
                        int64_t(n_ff_card) * 2);
        }
        if (!gin.put(din_sorted.data(), gin.n, x.st)) {
            x.ok = false;
            return;
        }
        for (const fmt dn : dns) {
            const std::string td = tg + "/" + fmt_name(dn);
            const int32_t di = dn_index(dn);
            std::vector<uint8_t> dh;
            if (!packed(x.syn[static_cast<int32_t>(dn)], card, dh)) {
                x.ok = false;
                return;
            }
            dev_mem wd(dh.size());
            if (!wd.put(dh.data(), dh.size(), x.st)) {
                x.ok = false;
                return;
            }
            const int32_t nsp = c.n_sp(), nr = static_cast<int32_t>(c.srows.size());
            std::vector<float> p4;
            std::vector<double> sum;
            {
                // the FP64 partials and sums (older ep-prep files hold two more parts after these, unread)
                FILE * fp = fopen(ref_name(x.dir, c.cfg, gu, di).c_str(), "rb");
                p4.assign(size_t(nsp) * (D / 4), 0.0f);
                sum.assign(size_t(nr) * D, 0.0);
                const bool rd_ok = fp != nullptr && fread(p4.data(), 4, p4.size(), fp) == p4.size() &&
                                   fread(sum.data(), 8, sum.size(), fp) == sum.size();
                if (fp != nullptr) {
                    fclose(fp);
                }
                if (!x.check(rd_ok, td + ": reference file (run ep-prep)")) {
                    return;
                }
            }
            dn_out r1, r2;
            if (!run_down(x, c, dn, wd, gin, rd, wts, part, out, r1)) {
                x.ok = false;
                return;
            }
            err_acc ep_, es;
            for (int32_t i = 0; i < nsp; ++i) {
                const int32_t p = c.sp_pair(i);
                for (int32_t n = 0; n < D; n += 4) {
                    ep_.add(r1.part[size_t(pos[size_t(p)]) * D + size_t(n)], p4[size_t(i) * (D / 4) + n / 4]);
                }
            }
            for (int32_t s = 0; s < nr; ++s) {
                for (int32_t n = 0; n < D; ++n) {
                    es.add(r1.out[size_t(c.srows[size_t(s)]) * D + n], sum[size_t(s) * D + n]);
                }
            }
            if (x.full) {
                if (!run_down(x, c, dn, wd, gin, rd, wts, part, out, r2)) {
                    x.ok = false;
                    return;
                }
                x.check(std::memcmp(r1.part.data(), r2.part.data(), r1.part.size() * 4) == 0 &&
                            std::memcmp(r1.out.data(), r2.out.data(), r1.out.size() * 4) == 0,
                        td + ": second down + sum run bitwise identical");
                // the chain: down + sum on this kernel's own GLU, against the reference's (it differs by the GLU's
                // own error): bound 1e-3 only
                dn_out rc;
                if (!run_down(x, c, dn, wd, glu, rd, wts, part, out, rc)) {
                    x.ok = false;
                    return;
                }
                err_acc ec;
                for (int32_t s = 0; s < nr; ++s) {
                    for (int32_t n = 0; n < D; ++n) {
                        ec.add(rc.out[size_t(c.srows[size_t(s)]) * D + n], sum[size_t(s) * D + n]);
                    }
                }
                report("chain sum (gate_up's own GLU)", td, ec, x);
            }
            report("down partials (every 4th column)", td, ep_, x);
            report("weighted sum", td, es, x);
        }
    }
}

// all-zero activations: GLU, partials and sum exactly zero
void zero_case(ctx & x, const ep_fn4 & fr, fmt gu, fmt dn) {
    const case_in c = make_inputs(fr, k_cases[0], true);
    const int32_t R = c.cfg.rows, pairs = R * NU;
    dev_mem ids(c.ids.size() * 4), wts(c.wts.size() * 4), xh(c.xh.size() * 2), glu(size_t(pairs) * n_ff_card * 2),
        part(size_t(pairs) * D * 4), out(size_t(R) * D * 4);
    route_dev rd;
    std::vector<int32_t> pos;
    std::vector<uint8_t> wh, dh;
    if (!ids.put(c.ids.data(), ids.n, x.st) || !wts.put(c.wts.data(), wts.n, x.st) ||
        !xh.put(c.xh.data(), xh.n, x.st) || !run_group(x, c, ep_task_rows(gu), ids, rd, pos) ||
        !packed(x.syn[static_cast<int32_t>(gu)], 0, wh) || !packed(x.syn[static_cast<int32_t>(dn)], 0, dh) ||
        !glu.fill(0xff, x.st)) {
        x.ok = false;
        return;
    }
    dev_mem w(wh.size()), wd(dh.size());
    const ep_gate_up_args a = {w.p, x.grids->p, xh.as<uint16_t>(), rd.view(), glu.as<uint16_t>(), nullptr};
    dn_out r;
    if (!w.put(wh.data(), wh.size(), x.st) || !wd.put(dh.data(), dh.size(), x.st) || !ep_gate_up(gu, R, a, x.st) ||
        !x.sync("zero gate_up") || !run_down(x, c, dn, wd, glu, rd, wts, part, out, r)) {
        x.ok = false;
        return;
    }
    std::vector<uint16_t> g(size_t(pairs) * n_ff_card);
    bool zero = glu.get(g.data(), glu.n, x.st);
    for (const uint16_t v : g) {
        zero = zero && v == 0;
    }
    for (const float v : r.part) {
        zero = zero && v == 0.0f;
    }
    for (const float v : r.out) {
        zero = zero && v == 0.0f;
    }
    printf("ep-test: zero input %s/%s R %d: GLU, partials and sum %s\n", fmt_name(gu), fmt_name(dn), R,
           zero ? "exactly zero" : "NOT ZERO");
    x.check(zero, "zero input gives zero outputs");
}

// an expert ID outside [0, 512): status 1, no task, nothing written
void bad_id_case(ctx & x, const ep_fn4 & fr, fmt gu) {
    case_in c = make_inputs(fr, k_cases[0], false);
    c.ids[size_t(NU * c.cfg.rows - 1)] = NE + 7;
    const int32_t R = c.cfg.rows, pairs = R * NU;
    dev_mem ids(c.ids.size() * 4), wts(c.wts.size() * 4), xh(c.xh.size() * 2), glu(size_t(pairs) * n_ff_card * 2),
        part(size_t(pairs) * D * 4), out(size_t(R) * D * 4), status(4);
    route_dev rd;
    std::vector<uint8_t> wh;
    if (!ids.put(c.ids.data(), ids.n, x.st) || !wts.put(c.wts.data(), wts.n, x.st) ||
        !xh.put(c.xh.data(), xh.n, x.st) || !packed(x.syn[static_cast<int32_t>(gu)], 0, wh) || !glu.fill(0xa5, x.st) ||
        !out.fill(0xa5, x.st) || !status.fill(0, x.st)) {
        x.ok = false;
        return;
    }
    dev_mem w(wh.size());
    const ep_gate_up_args a = {w.p, x.grids->p, xh.as<uint16_t>(), rd.view(), glu.as<uint16_t>(), nullptr};
    const ep_sum_args sa = {part.as<float>(), rd.view(), wts.as<float>(), nullptr, out.as<float>()};
    if (!w.put(wh.data(), wh.size(), x.st) ||
        !ep_group(R, ep_task_rows(gu), ids.as<int32_t>(), rd.view(), status.as<uint32_t>(), x.st) ||
        !ep_gate_up(gu, R, a, x.st) || !ep_sum(R, sa, x.st) || !x.sync("bad id")) {
        x.ok = false;
        return;
    }
    uint32_t st = 0;
    std::vector<uint8_t> g(glu.n), o(out.n);
    bool untouched = status.get(&st, 4, x.st) && glu.get(g.data(), g.size(), x.st) && out.get(o.data(), o.size(), x.st);
    for (const uint8_t v : g) {
        untouched = untouched && v == 0xa5;
    }
    for (const uint8_t v : o) {
        untouched = untouched && v == 0xa5;
    }
    printf("ep-test: bad ID %d: status %u, outputs %s\n", NE + 7, st, untouched ? "untouched" : "WRITTEN");
    x.check(st == 1 && untouched, "bad expert ID");
}

// ME4: the cache instantiation's ring groups. One routed input three ways: the resident bank (w), the cache's table
// (wt) in one launch, and the table in four uneven expert groups (one a single expert) over ep_task_offsets with each
// group's tight grid. Every task reads the same weights, so the GLU and the partials must be bitwise equal across the
// three, and every pair written (the buffers start 0xff)
void grouped_case(ctx & x, const ep_fn4 & fr, fmt gu, fmt dn, const case_cfg & cfg) {
    const case_in c = make_inputs(fr, cfg, false);
    const int32_t R = c.cfg.rows, pairs = R * NU, tr = ep_task_rows(gu);
    dev_mem ids(c.ids.size() * 4), xh(c.xh.size() * 2), toff(size_t(NE + 1) * 4), tab_gu(size_t(NE) * 8),
        tab_dn(size_t(NE) * 8);
    route_dev rd;
    std::vector<int32_t> pos;
    std::vector<uint8_t> wh, dh;
    if (!ids.put(c.ids.data(), ids.n, x.st) || !xh.put(c.xh.data(), xh.n, x.st) ||
        !run_group(x, c, tr, ids, rd, pos) || !packed(x.syn[static_cast<int32_t>(gu)], c.cfg.card, wh) ||
        !packed(x.syn[static_cast<int32_t>(dn)], c.cfg.card, dh)) {
        x.ok = false;
        return;
    }
    dev_mem w(wh.size()), wd(dh.size());
    std::vector<const uint8_t *> tg(NE), td(NE);
    for (int32_t e = 0; e < NE; ++e) {
        tg[size_t(e)] = w.p + int64_t(e) * gate_up_bytes(gu);
        td[size_t(e)] = wd.p + int64_t(e) * down_bytes(dn);
    }
    if (!w.put(wh.data(), wh.size(), x.st) || !wd.put(dh.data(), dh.size(), x.st) ||
        !tab_gu.put(tg.data(), tab_gu.n, x.st) || !tab_dn.put(td.data(), tab_dn.n, x.st) ||
        !ep_task_offsets(rd.view(), toff.as<int32_t>(), x.st) || !x.sync("task offsets")) {
        x.ok = false;
        return;
    }
    // the experts' pairs (the group grids' bound, as the caller computes it from the route)
    std::vector<int64_t> pe(NE, 0);
    for (const int32_t id : c.ids) {
        ++pe[size_t(id)];
    }
    const int32_t cuts[5] = {0, 100, 101, 400, NE};
    std::vector<uint16_t> glu_h[3];
    std::vector<float> part_h[3];
    for (int32_t way = 0; way < 3; ++way) {
        dev_mem glu(size_t(pairs) * n_ff_card * 2), part(size_t(pairs) * D * 4);
        if (!glu.fill(0xff, x.st) || !part.fill(0xff, x.st)) {
            x.ok = false;
            return;
        }
        for (int32_t g = 0; g < (way == 2 ? 4 : 1); ++g) {
            ep_gate_up_args a = {w.p, x.grids->p, xh.as<uint16_t>(), rd.view(), glu.as<uint16_t>(), nullptr};
            ep_down_args d = {wd.p, x.grids->p, glu.as<uint16_t>(), rd.view(), part.as<float>()};
            if (way > 0) {
                a.wt = tab_gu.as<const uint8_t *>();
                d.wt = tab_dn.as<const uint8_t *>();
            }
            if (way == 2) {
                int64_t gp = 0;
                for (int32_t e = cuts[g]; e < cuts[g + 1]; ++e) {
                    gp += pe[size_t(e)];
                }
                a.task_off = d.task_off = toff.as<int32_t>();
                a.e_lo = d.e_lo = cuts[g];
                a.e_hi = d.e_hi = cuts[g + 1];
                a.grid = d.grid = int32_t((cuts[g + 1] - cuts[g]) + (gp + tr - 1) / tr);
            }
            if (!x.check(ep_gate_up(gu, R, a, x.st), tag(c) + ": grouped gate_up enqueued") ||
                !x.check(ep_down(dn, R, d, x.st), tag(c) + ": grouped down enqueued") || !x.sync("grouped")) {
                return;
            }
        }
        glu_h[way].resize(size_t(pairs) * n_ff_card);
        part_h[way].resize(size_t(pairs) * D);
        if (!glu.get(glu_h[way].data(), glu.n, x.st) || !part.get(part_h[way].data(), part.n, x.st)) {
            x.ok = false;
            return;
        }
    }
    bool written = true;
    for (const uint16_t v : glu_h[2]) {
        written = written && v != 0xffff;
    }
    const bool same = glu_h[0] == glu_h[1] && glu_h[1] == glu_h[2] &&
                      std::memcmp(part_h[0].data(), part_h[1].data(), part_h[0].size() * 4) == 0 &&
                      std::memcmp(part_h[1].data(), part_h[2].data(), part_h[1].size() * 4) == 0;
    printf("ep-test: %s/%s %s: grouped (4 groups, task offsets, tight grids) against one table launch and the "
           "resident bank: GLU and partials %s, every pair written %s\n",
           fmt_name(gu), fmt_name(dn), tag(c).c_str(), same ? "bitwise equal" : "DIFFER", written ? "yes" : "NO");
    x.check(same && written, tag(c) + ": ME4 grouped launches equal the single launch bitwise");
}

// MX7: the prompt CPU share's kernels (experts_share.h) on one routed input. es_drop against the host's in-order
// filter of ep_group's tasks (the CPU's experts: every third expert of at most 8 pairs, and two unrouted), then
// ep_task_offsets over the kept tasks; after a bad ID it changes nothing. es_gather: the CPU experts' rows bitwise.
// es_add: their pairs' rows (or zeros) at pos, every other partial untouched, the error bit only when flagged
void share_case(ctx & x, const ep_fn4 & fr, fmt gu, const case_cfg & cfg) {
    const case_in c = make_inputs(fr, cfg, false);
    const int32_t R = c.cfg.rows, pairs = R * NU, tr = ep_task_rows(gu);
    dev_mem ids(c.ids.size() * 4), xh(c.xh.size() * 2), toff(size_t(NE + 1) * 4), skip(NE);
    route_dev rd;
    std::vector<int32_t> pos;
    if (!ids.put(c.ids.data(), ids.n, x.st) || !xh.put(c.xh.data(), xh.n, x.st) || !run_group(x, c, tr, ids, rd, pos)) {
        x.ok = false;
        return;
    }
    std::vector<int32_t> cnt(NE, 0);
    for (const int32_t id : c.ids) {
        ++cnt[size_t(id)];
    }
    std::vector<uint8_t> sk(NE, 0);
    int32_t n_cpu = 0, unrouted = 0;
    for (int32_t e = 0; e < NE; ++e) {
        if (cnt[size_t(e)] >= 1 && cnt[size_t(e)] <= 8 && e % 3 == 0) {
            sk[size_t(e)] = 1;
            ++n_cpu;
        } else if (cnt[size_t(e)] == 0 && unrouted < 2) {
            sk[size_t(e)] = 1;
            ++unrouted;
        }
    }
    std::vector<int32_t> t0(size_t(ep_max_tasks) * 4), t1(t0.size()), c0(8), c1(8), off(NE + 1);
    if (!rd.tasks.get(t0.data(), t0.size() * 4, x.st) || !rd.counters.get(c0.data(), 32, x.st) ||
        !skip.put(sk.data(), sk.size(), x.st) || !x.check(es_drop(rd.view(), skip.as<uint8_t>(), x.st), "es_drop") ||
        !x.check(ep_task_offsets(rd.view(), toff.as<int32_t>(), x.st), "task offsets") || !x.sync("es_drop") ||
        !rd.tasks.get(t1.data(), t1.size() * 4, x.st) || !rd.counters.get(c1.data(), 32, x.st) ||
        !toff.get(off.data(), off.size() * 4, x.st)) {
        x.ok = false;
        return;
    }
    std::vector<int32_t> kept;
    for (int32_t i = 0; i < c0[0]; ++i) {
        if (sk[size_t(t0[size_t(i) * 4])] == 0) {
            kept.insert(kept.end(), t0.begin() + i * 4, t0.begin() + i * 4 + 4);
        }
    }
    const int32_t nk = int32_t(kept.size() / 4);
    bool drop =
        c1[0] == nk && c1[1] == 0 && n_cpu > 0 && nk < c0[0] && std::equal(kept.begin(), kept.end(), t1.begin());
    for (int32_t e = 0, i = 0; e <= NE && drop; ++e) {
        while (i < nk && kept[size_t(i) * 4] < e) {
            ++i;
        }
        drop = off[size_t(e)] == (e == NE ? nk : i);
    }
    // a bad ID flagged: nothing changes
    std::vector<int32_t> bad = c1, t2(t1.size()), c2(8);
    bad[1] = 1;
    const bool held = rd.counters.put(bad.data(), 32, x.st) && es_drop(rd.view(), skip.as<uint8_t>(), x.st) &&
                      x.sync("es_drop bad") && rd.tasks.get(t2.data(), t2.size() * 4, x.st) &&
                      rd.counters.get(c2.data(), 32, x.st) && t2 == t1 && c2 == bad;
    x.check(drop && held, tag(c) + ": MX7 es_drop keeps the GPU's tasks in order, task offsets over them, a bad ID "
                                   "changes nothing");
    // the CPU experts' rows (ascending) and pairs (by expert, then pair)
    std::vector<int32_t> rows, cp;
    std::vector<uint8_t> in(size_t(R), 0);
    for (int32_t e = 0; e < NE; ++e) {
        for (int32_t p = 0; p < pairs && sk[size_t(e)] != 0; ++p) {
            if (c.ids[size_t(p)] == e) {
                cp.push_back(p);
                in[size_t(p / NU)] = 1;
            }
        }
    }
    for (int32_t r = 0; r < R; ++r) {
        if (in[size_t(r)] != 0) {
            rows.push_back(r);
        }
    }
    const int32_t nr = int32_t(rows.size()), np = int32_t(cp.size());
    dev_mem rows_d(size_t(nr) * 4), out(size_t(nr) * D * 2), cp_d(size_t(np) * 4), r16(size_t(np) * D * 2),
        part(size_t(pairs) * D * 2), err(4);
    std::vector<uint16_t> got(size_t(nr) * D), rh(size_t(np) * D);
    std::mt19937 rng(17);
    for (uint16_t & v : rh) {
        v = uint16_t(rng() & 0x7bff); // finite F16
    }
    if (!rows_d.put(rows.data(), rows_d.n, x.st) || !cp_d.put(cp.data(), cp_d.n, x.st) ||
        !r16.put(rh.data(), r16.n, x.st) || !part.fill(0xff, x.st) || !err.fill(0, x.st) ||
        !x.check(es_gather(xh.as<uint16_t>(), rows_d.as<int32_t>(), nr, out.as<uint16_t>(), x.st), "es_gather") ||
        !x.sync("es_gather") || !out.get(got.data(), out.n, x.st)) {
        x.ok = false;
        return;
    }
    bool gather = true;
    for (int32_t i = 0; i < nr && gather; ++i) {
        gather = std::memcmp(got.data() + int64_t(i) * D, c.xh.data() + int64_t(rows[size_t(i)]) * D, D * 2) == 0;
    }
    std::vector<uint16_t> ph(size_t(pairs) * D);
    uint32_t e0 = 1, e1 = 0;
    bool add = x.check(es_add(cp_d.as<int32_t>(), r16.as<uint16_t>(), np, rd.pos.as<int32_t>(), part.as<uint16_t>(), 0,
                              err.as<uint32_t>(), 4u, x.st),
                       "es_add") &&
               x.sync("es_add") && part.get(ph.data(), part.n, x.st) && err.get(&e0, 4, x.st) && e0 == 0;
    std::vector<uint8_t> at(size_t(pairs), 0);
    for (int32_t i = 0; i < np && add; ++i) {
        const int32_t q = pos[size_t(cp[size_t(i)])];
        at[size_t(q)] = 1;
        add = std::memcmp(ph.data() + int64_t(q) * D, rh.data() + int64_t(i) * D, D * 2) == 0;
    }
    for (int32_t q = 0; q < pairs && add; ++q) {
        for (int32_t k = 0; k < D && at[size_t(q)] == 0 && add; ++k) {
            add = ph[size_t(q) * D + size_t(k)] == 0xffff;
        }
    }
    add = add &&
          es_add(cp_d.as<int32_t>(), nullptr, np, rd.pos.as<int32_t>(), part.as<uint16_t>(), 1, err.as<uint32_t>(), 4u,
                 x.st) &&
          x.sync("es_add zeros") && part.get(ph.data(), part.n, x.st) && err.get(&e1, 4, x.st) && e1 == 4u;
    for (int32_t q = 0; q < pairs && add; ++q) {
        for (int32_t k = 0; k < D && add; ++k) {
            add = ph[size_t(q) * D + size_t(k)] == (at[size_t(q)] != 0 ? 0 : 0xffff);
        }
    }
    printf("ep-test: %s %s: MX7 share kernels: %d CPU experts (+%d unrouted), tasks %d -> %d, %d rows gathered %s, "
           "%d pairs added %s\n",
           fmt_name(gu), tag(c).c_str(), n_cpu, unrouted, c0[0], c1[0], nr, gather ? "bitwise" : "DIFFER", np,
           add ? "at their positions (zeros, the error bit)" : "WRONG");
    x.check(gather && add, tag(c) + ": MX7 es_gather and es_add");
}

bool parse_formats(const char * s, std::vector<fmt> & gus, std::vector<fmt> & dns) {
    const std::string str(s);
    for (int32_t i = 0; i < n_fmt; ++i) {
        const fmt f = static_cast<fmt>(i);
        bool want = (str == "all" && !is_q4(f)) || (str == "q4" && is_q4(f));
        size_t p = 0;
        while (!want && p <= str.size()) {
            const size_t e = std::min(str.find(',', p), str.size());
            want = str.substr(p, e - p) == lname(f);
            p = e + 1;
        }
        if (want) {
            (is_down(f) ? dns : gus).push_back(f);
        }
    }
    return !gus.empty() && !dns.empty();
}

} // namespace
} // namespace fn::experts

int fnfl_ep_test(int device, const char * rows, const char * formats, const char * dir, const char * syn_dir,
                 const char * routes, const char * mode) {
    using namespace fn::experts;
    ctx x;
    x.full = strcmp(mode, "san") != 0;
    x.dir = dir;
    std::vector<fmt> gus, dns;
    if (!parse_formats(formats, gus, dns) || (strcmp(mode, "full") != 0 && strcmp(mode, "san") != 0)) {
        fprintf(stderr, "ep-test: FORMATS must list a gate/up and a down format; MODE is full or san\n");
        return 2;
    }
    std::vector<int32_t> want;
    for (const char * p = rows; *p != '\0';) {
        want.push_back(atoi(p));
        while (*p != '\0' && *p != ',') {
            ++p;
        }
        if (*p == ',') {
            ++p;
        }
    }
    ep_fn4 fr;
    if (!ep_read_fn4(routes, fr) || !cuda_ok(cudaSetDevice(device), "cudaSetDevice") ||
        !cuda_ok(cudaStreamCreateWithFlags(&x.st, cudaStreamNonBlocking), "stream")) {
        return 1;
    }
    for (const std::vector<fmt> * v : {&gus, &dns}) {
        for (const fmt f : *v) {
            if (!load_synth(f, syn_dir, x.syn[static_cast<int32_t>(f)], true, f == fmt::q8_0)) {
                return 1;
            }
        }
    }
    std::vector<uint8_t> g;
    x.grids = std::make_unique<dev_mem>(grid_bytes);
    if (!grid_buffer(g) || !x.grids->put(g.data(), g.size(), x.st)) {
        return 1;
    }
    for (const case_cfg & cfg : k_cases) {
        const bool listed = std::find(want.begin(), want.end(), cfg.rows) != want.end();
        if (!listed || (cfg.card != 0 && !x.full)) {
            continue;
        }
        run_case(x, make_inputs(fr, cfg, false), gus, dns);
    }
    if (x.full) {
        zero_case(x, fr, gus[0], dns[0]);
        bad_id_case(x, fr, gus[0]);
    }
    // ME4: grouped launches, every listed gate/up format with each down format, at every listed row count
    for (const case_cfg & cfg : k_cases) {
        if (std::find(want.begin(), want.end(), cfg.rows) == want.end() || cfg.card != 0) {
            continue;
        }
        for (const fmt g : gus) {
            for (const fmt d : dns) {
                grouped_case(x, fr, g, d, cfg);
            }
        }
    }
    // MX7: the share's kernels at every listed row count
    for (const case_cfg & cfg : k_cases) {
        if (std::find(want.begin(), want.end(), cfg.rows) != want.end() && cfg.card == 0) {
            share_case(x, fr, gus[0], cfg);
        }
    }
    cudaStreamDestroy(x.st);
    if (x.ok) {
        printf("ep-test: PASS (%d checks)\n", x.n_checks);
        return 0;
    }
    printf("ep-test: FAIL\n");
    return 1;
}
