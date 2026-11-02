#include "fn/runtime/vis_kl.h"

#include "fn/kernels/head/head.h"
#include "fn/kernels/round/round.h"
#include "fn/loader/run_load.h"
#include "fn/model/model.h"
#include "fn/runtime/image_rows.h"
#include "fn/runtime/runtime.h"
#include "fn/vision/image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fn::runtime {

namespace {

constexpr int32_t V  = fn::head::n_vocab;       // 248,320
constexpr int32_t VC = fn::head::n_vocab_card;  // 124,160: a card's shard

struct ref_prompt {
    std::string                                          id, kind;
    int64_t                                              prompt_n = 0;
    std::vector<std::string>                             files;
    std::vector<std::vector<int32_t>>                    pieces;
    std::vector<int32_t>                                 gen;
    std::vector<std::vector<std::pair<int32_t, double>>> top;  // each step's top 20: ID, logprob
};

bool read_ref(const char * path, int32_t & vs, int32_t & ve, std::vector<ref_prompt> & out) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::string line;
    vs = ve = -1;
    while (std::getline(f, line)) {
        std::istringstream in(line);
        std::string        tag;
        in >> tag;
        if (tag == "B") {
            in >> vs >> ve;
        } else if (tag == "P") {
            ref_prompt p;
            int32_t    n_images = 0;
            in >> p.id >> p.kind >> n_images >> p.prompt_n;
            out.push_back(std::move(p));
        } else if (out.empty()) {
            return false;
        } else if (tag == "I") {
            std::string s;
            in >> s;
            out.back().files.push_back(s);
        } else if (tag == "S" || tag == "G") {
            int32_t              n = 0;
            std::vector<int32_t> v;
            in >> n;
            for (int32_t i = 0, x = 0; i < n && in >> x; ++i) {
                v.push_back(x);
            }
            if (int32_t(v.size()) != n) {
                return false;
            }
            (tag == "S" ? out.back().pieces.emplace_back() : out.back().gen) = std::move(v);
        } else if (tag == "K") {
            int32_t                                 n = 0;
            std::vector<std::pair<int32_t, double>> v;
            in >> n;
            int32_t id = 0;
            double  lp = 0;
            for (int32_t i = 0; i < n && in >> id >> lp; ++i) {
                v.emplace_back(id, lp);
            }
            if (int32_t(v.size()) != n || n < 1) {
                return false;
            }
            out.back().top.push_back(std::move(v));
        }
    }
    return vs >= 0 && ve >= 0 && !out.empty();
}

// KL of the reference's top-20 distribution plus one outcome for the rest, against FNFL's full softmax (FP64)
double kl_top(const float * l, const std::vector<std::pair<int32_t, double>> & top) {
    double m = -HUGE_VAL;
    for (int32_t i = 0; i < V; ++i) {
        m = std::max(m, double(l[i]));
    }
    double s = 0.0;
    for (int32_t i = 0; i < V; ++i) {
        s += std::exp(double(l[i]) - m);
    }
    const double lse = m + std::log(s);
    double       k = 0.0, sp = 0.0, sq = 0.0;
    for (const auto & t : top) {
        const double lq = double(l[t.first]) - lse;
        k += std::exp(t.second) * (t.second - lq);
        sp += std::exp(t.second);
        sq += std::exp(lq);
    }
    const double pr = 1.0 - sp, qr = std::max(1.0 - sq, 1e-300);
    if (pr > 0.0) {
        k += pr * (std::log(pr) - std::log(qr));
    }
    return k;
}

int32_t argmax(const float * l) {  // the lower ID on a tie
    int32_t b = 0;
    for (int32_t i = 1; i < V; ++i) {
        b = l[i] > l[b] ? i : b;
    }
    return b;
}

// rows 0 .. rows - 1 of the slot's logits (card 0's shard, then card 1's) into out [rows][V]
bool take(target & t, const model::model & m, int32_t rows, std::vector<float> & out) {
    out.resize(size_t(rows) * V);
    std::vector<float> h(size_t(rows) * VC);
    for (int32_t c = 0; c < 2; ++c) {
        if (cudaSetDevice(m.device[c]) != cudaSuccess ||
            cudaMemcpy(h.data(), t.card(c).logits, h.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        for (int32_t b = 0; b < rows; ++b) {
            std::copy_n(h.data() + size_t(b) * VC, VC, out.data() + size_t(b) * V + size_t(c) * VC);
        }
    }
    return true;
}

bool clean(const round_result & r) {
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t i = 0; i < fn::round::max_err; ++i) {
            if (r.err[c][i] != 0u) {
                return false;
            }
        }
    }
    return true;
}

bool read_ids(const std::string & path, std::vector<int32_t> & v) {
    std::ifstream f(path);
    v.clear();
    for (int32_t x = 0; f >> x;) {
        v.push_back(x);
    }
    return !v.empty();
}

// the a0 request (fmu_scen.py's sampler: T 0, top-k 20, top-p 0.95, min-p 0.05, seed 20261013, 2,048 tokens) on
// slot 0 through the speculative path; true when its tokens are the reference's, bitwise
bool run_a0(engine & e, const std::vector<int32_t> & prompt, const std::vector<int32_t> & ref) {
    request_options o;
    o.sampling.temp           = 0.0f;
    o.sampling.top_k          = 20;
    o.sampling.top_p          = 0.95f;
    o.sampling.min_p          = 0.05f;
    o.sampling.penalty_last_n = 0;
    o.sampling.seed           = 20261013;
    o.max_tokens              = int32_t(ref.size());
    int32_t np                = 0;
    if (e.set_sampling(0, o) != run_status::ok || e.truncate(0, 0) != run_status::ok ||
        e.prefill(0, prompt.data(), int32_t(prompt.size()), 0, np) != run_status::ok) {
        std::printf("a0: the read: %s\n", e.error().c_str());
        return false;
    }
    std::vector<int32_t> got;
    int32_t              steps = 0;
    for (bool end = false; !end && got.size() < ref.size();) {
        int32_t out[max_step_tokens] = {};
        int32_t n                    = 0;
        if (e.step(0, out, n, end) != run_status::ok) {
            std::printf("a0: step %d: %s\n", steps, e.error().c_str());
            return false;
        }
        got.insert(got.end(), out, out + n);
        ++steps;
    }
    size_t i = 0;
    while (i < got.size() && i < ref.size() && got[i] == ref[i]) {
        ++i;
    }
    const bool same = i == ref.size() && got.size() == ref.size();
    std::printf("a0: %zu prompt IDs, %zu tokens in %d steps (%.3f a step): %s", prompt.size(), got.size(), steps,
                steps > 0 ? double(got.size()) / steps : 0.0,
                same ? "bitwise the reference's 2,048 (FMU4 A-cand-t1 four-t0 a0)\n" : "");
    if (!same) {
        std::printf("FIRST DIVERGENCE at token %zu: got %d, reference %d\n", i, i < got.size() ? got[i] : -1,
                    i < ref.size() ? ref[i] : -1);
    }
    return same;
}

}  // namespace

int fnfl_vis_kl(const vis_kl_args & a) {
    int32_t                 vs = -1, ve = -1;
    std::vector<ref_prompt> ps;
    std::vector<int32_t>    a0p, a0r;
    if (!read_ref(a.ref.c_str(), vs, ve, ps)) {
        std::fprintf(stderr, "vis-kl: cannot read the reference file %s\n", a.ref.c_str());
        return 2;
    }
    if (!a.a0_prompt.empty() && (!read_ids(a.a0_prompt, a0p) || !read_ids(a.a0_ref, a0r))) {
        std::fprintf(stderr, "vis-kl: cannot read a0's IDs (%s, %s)\n", a.a0_prompt.c_str(), a.a0_ref.c_str());
        return 2;
    }
    // as fnfl_load loads the served IQ3_XXS one-box configuration (fmu_scen.py's a0 server), with the projector
    constexpr int32_t           cells = 262144;
    auto                        m     = std::make_unique<model::model>();
    fn::loader::load_params     lp;
    fn::loader::run_load_report lr;
    std::string                 err;
    lp.model_path = a.model;
    lp.draft_path = a.draft;
    lp.mmproj     = a.mmproj;
    lp.n_slots    = 4;
    lp.kv_q8      = true;
    lp.pool_cells = a.pool_cells;
    if (!fn::loader::load_for_run(lp, cells, *m, lr, err)) {
        std::fprintf(stderr, "vis-kl: load: %s\n", err.c_str());
        return 1;
    }
    std::printf(
        "vis-kl: %s, draft %s, projector %s: loaded in %.2f s; four slots of %d cells, KV pool %d cells (q8); "
        "free after the load %.3f / %.3f GiB\n",
        a.model.c_str(), a.draft.c_str(), a.mmproj.c_str(), lr.total_s, cells, a.pool_cells,
        double(lr.free_after[0]) / 1073741824.0, double(lr.free_after[1]) / 1073741824.0);
    engine        e;
    engine_config c;
    c.model_path      = a.model;
    c.mmproj          = a.mmproj;
    c.n_ctx_slot      = cells;
    c.draft.on        = true;
    c.draft.attention = "sparse";
    c.draft.subset    = a.subset;
    c.spec_d          = 4;
    c.eog             = { 248044, 248046, 248063, 248064, 248065 };
    c.temp            = 1.0f;
    c.top_k           = 20;
    c.top_p           = 0.95f;
    c.min_p           = 0.05f;
    if (!e.init(std::move(m), c, err)) {
        std::fprintf(stderr, "vis-kl: engine: %s\n", err.c_str());
        return 1;
    }
    std::printf("vis-kl: %s\n", e.settings().c_str());
    const bool a0_ok = a0p.empty() || run_a0(e, a0p, a0r);
    std::fflush(stdout);
    request_options ro;
    ro.sampling.temp          = 0.0f;
    ro.max_tokens             = 100000;
    int                failed = a0_ok ? 0 : 1;
    std::vector<float> lg;
    for (const ref_prompt & p : ps) {
        const auto bad = [&](const std::string & why) {
            std::printf("vis-kl: %s: %s\n", p.id.c_str(), why.c_str());
            ++failed;
        };
        if (p.pieces.size() != p.files.size() + 1 || p.gen.empty() || p.top.size() != p.gen.size()) {
            bad("a malformed reference record");
            continue;
        }
        // the prompt as the server builds it: each piece, then each image's start token, image rows and end token
        std::vector<int32_t>      tok;
        std::vector<prompt_image> im;
        bool                      ok = true;
        for (size_t i = 0; i < p.pieces.size() && ok; ++i) {
            tok.insert(tok.end(), p.pieces[i].begin(), p.pieces[i].end());
            if (i == p.files.size()) {
                break;
            }
            std::ifstream        f(p.files[i], std::ios::binary);
            std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            auto                 img = std::make_shared<vision::prepared>();
            if (b.empty() || vision::prepare(b.data(), b.size(), vision::limits(), *img) != vision::status::ok) {
                bad("cannot prepare " + p.files[i]);
                ok = false;
                break;
            }
            tok.push_back(vs);
            prompt_image pi;
            pi.span = { int64_t(tok.size()), img->nx, img->ny, img->digest };
            pi.img  = img;
            tok.insert(tok.end(), size_t(img->nx) * img->ny, image_token);
            tok.push_back(ve);
            im.push_back(std::move(pi));
        }
        if (!ok) {
            continue;
        }
        std::printf("N %s %zu\n", p.id.c_str(), tok.size());
        std::printf("vis-kl: %s: prompt %zu tokens (the reference's %lld: %s), %zu image%s\n", p.id.c_str(), tok.size(),
                    static_cast<long long>(p.prompt_n), int64_t(tok.size()) == p.prompt_n ? "equal" : "DIFFERENT",
                    im.size(), im.size() == 1 ? "" : "s");
        int32_t np = 0;
        if (e.set_sampling(0, ro) != run_status::ok || e.truncate(0, 0) != run_status::ok ||
            e.prefill(0, tok.data(), int32_t(tok.size()), 0, np, {}, im.empty() ? nullptr : &im) != run_status::ok) {
            bad("the read: " + e.error());
            continue;
        }
        target &       t  = e.test_target(0);
        model::model & mm = e.test_model();
        // step 0 from the read's retained row, then the reference's tokens forced, each row predicting the next step
        const int32_t  G  = int32_t(p.gen.size());
        if (!take(t, mm, 1, lg)) {
            bad("the retained logits");
            continue;
        }
        const auto report = [&](int32_t step, const float * l) {
            const int32_t a = argmax(l), r = p.top[size_t(step)][0].first;
            std::printf("R %s %d %.9g %d %d %d\n", p.id.c_str(), step, kl_top(l, p.top[size_t(step)]), a == r ? 1 : 0,
                        a, r);
        };
        report(0, lg.data());
        if (G > 1) {
            t.append(std::vector<int32_t>(p.gen.begin(), p.gen.end() - 1));
        }
        for (int32_t done = 0; done < G - 1 && ok;) {
            const int32_t rows = std::min(fn::round::max_rows, G - 1 - done);
            round_result  r;
            if (!t.forced(rows, r, err) || !clean(r) || !take(t, mm, rows, lg)) {
                bad("forced round at step " + std::to_string(done + 1) + ": " + err);
                ok = false;
                break;
            }
            for (int32_t b = 0; b < rows; ++b) {
                report(done + b + 1, lg.data() + size_t(b) * V);
            }
            done += rows;
        }
        std::fflush(stdout);
    }
    std::printf("vis-kl: a0 %s; %d of %zu prompts failed\n",
                a0p.empty() ? "not run" :
                a0_ok       ? "bitwise" :
                              "DIFFERS",
                failed - (a0_ok ? 0 : 1), ps.size());
    return failed == 0 ? 0 : 1;
}

}  // namespace fn::runtime
