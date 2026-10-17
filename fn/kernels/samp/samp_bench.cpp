#include "fn/kernels/samp/samp_bench.h"

#include "fn/kernels/samp/samp.h"
#include "fn/kernels/samp/samp_harness.h"
#include "fn/kernels/samp/samp_ref.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <thread>
#include <vector>

// SM1 step 6: each path's decision alone, the logits of TG1's recorded rows (the pinned engine's decode-mode logits)
// static in device memory, so they are L2-warm after the first decision (1.5 MB a card at B = 3, below V100's 6 MB L2)
// as they are when the head has just written them. Both cards' graphs (16 decisions each, back to back, every record
// checked by the next decision) are launched together; a decision's time is a card's event span over 16.

namespace {

using namespace fn::samp;
using namespace fn::samp::harness;

constexpr int32_t n_dec = 16;   // decisions a graph
constexpr int32_t n_warm = 30;  // untimed replays (HX12: a process starts at 1,260 MHz for 0.3-1 s)
constexpr int32_t n_timed = 20; // timed replays

struct point {
    std::string name;
    options o;
    int32_t rows = 1;    // 1: generate; 3: verify with 2 drafts
    bool reject = false; // verify: drafts the target rejects (the residual stage runs)
};

bool read_rows(const char * path, int32_t n, std::vector<std::vector<float>> & rows) {
    const std::string cmd = std::string("gzip -dc '") + path + "'";
    FILE * f = popen(cmd.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    rows.assign(static_cast<size_t>(n), std::vector<float>(static_cast<size_t>(V)));
    bool ok = true;
    for (int32_t i = 0; i < n && ok; ++i) {
        ok = fread(rows[size_t(i)].data(), 4, size_t(V), f) == size_t(V);
    }
    pclose(f);
    return ok;
}

double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 0 : n % 2 == 0 ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
}

double p90(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[(9 * v.size() + 9) / 10 - 1];
}

} // namespace

int fnfl_samp_bench(int dev0, int dev1, const char * out_path) {
    if (dev0 == dev1 || !fn::xchg::enable_peer_access(dev0, dev1)) {
        fprintf(stderr, "samp-bench: two cards with peer access\n");
        return 1;
    }
    std::vector<std::vector<float>> rows;
    if (!read_rows("/home/jeremy/src/flashnext/work/TG1/raw/ref-decode/logits.bin.gz", 4, rows)) {
        fprintf(stderr, "samp-bench: TG1's recorded logits\n");
        return 1;
    }
    FILE * jf = out_path != nullptr && out_path[0] != 0 ? fopen(out_path, "w") : nullptr;
    std::vector<point> pts;
    {
        options base;
        base.seed = 99;
        options g = base;
        g.temp = 0.0f;
        pts.push_back({"greedy (V1 + verdict)", g, 1, false});
        pts.push_back({"greedy (V1 + verdict)", g, 3, false});
        pts.push_back({"top-k 20, top-p 0.95, min-p 0.05 (defaults)", base, 1, false});
        pts.push_back({"top-k 20, top-p 0.95, min-p 0.05 (defaults)", base, 3, false});
        options f = base;
        f.top_k = 0;
        f.top_p = 1.0f;
        f.min_p = 0.0f;
        pts.push_back({"no filter", f, 1, false});
        pts.push_back({"no filter", f, 3, false});
        pts.push_back({"no filter", f, 3, true});
        f.min_p = 0.05f;
        pts.push_back({"full min-p 0.05", f, 1, false});
        pts.push_back({"full min-p 0.05", f, 3, false});
        f.top_p = 0.95f;
        pts.push_back({"full top-p 0.95 and min-p 0.05", f, 1, false});
        pts.push_back({"full top-p 0.95 and min-p 0.05", f, 3, false});
        pts.push_back({"full top-p 0.95 and min-p 0.05", f, 3, true});
        f.min_p = 0.0f;
        pts.push_back({"full top-p 0.95 alone", f, 1, false});
        pts.push_back({"full top-p 0.95 alone", f, 3, false});
    }
    int rc = 0;
    {
        card_s k0(dev0), k1(dev1);
        fn::xchg::rings rg(dev0, dev1);
        tester T;
        T.c[0] = &k0;
        T.c[1] = &k1;
        T.rg = &rg;
        cudaEvent_t ev[2][2] = {};
        bool ok = rg.ok() && T.init() && T.put_logits(rows);
        for (int32_t r = 0; r < 2 && ok; ++r) {
            cudaSetDevice(T.c[r]->dev);
            ok = cuda_ok(cudaEventCreate(&ev[r][0]), "event") && cuda_ok(cudaEventCreate(&ev[r][1]), "event");
        }
        for (const point & pt : pts) {
            if (!ok) {
                break;
            }
            ok = T.set_options(pt.o, {});
            // q = p for the drafts' rows (accepted unless reject: then the drafts are tokens q holds at 1 and p near 0)
            std::string err;
            std::vector<ref_dist> p(3);
            for (int32_t t = 0; t < 3 && ok; ++t) {
                ok = ref_distribution(pt.o, rows[size_t(t)].data(), V, NC, p[size_t(t)], err);
            }
            std::vector<std::vector<int32_t>> qid(3);
            std::vector<std::vector<float>> qp(3), qs(3, std::vector<float>(static_cast<size_t>(V), 0.0f));
            int32_t drafts[2] = {0, 0};
            for (int32_t t = 0; t < 2 && ok; ++t) {
                const ref_dist & d = p[size_t(t)];
                size_t best = 0;
                for (size_t i = 0; i < d.id.size(); ++i) {
                    best = d.prob[i] > d.prob[best] ? i : best;
                }
                drafts[t] = pt.reject ? 3 + t : d.id[best]; // IDs 3, 4: outside every kept set of these rows
                for (size_t i = 0; i < d.id.size(); ++i) {
                    qid[size_t(t)].push_back(d.id[i]);
                    qp[size_t(t)].push_back(static_cast<float>(d.prob[i]));
                    qs[size_t(t)][size_t(d.id[i])] = static_cast<float>(d.prob[i]);
                }
                if (pt.reject) {
                    qid[size_t(t)].push_back(drafts[t]);
                    qp[size_t(t)].push_back(1.0f);
                    qs[size_t(t)][size_t(drafts[t])] = 1.0f;
                }
            }
            ok = ok && T.put_q(qid, qp, qs);
            std::vector<dec_in> in(n_dec);
            for (int32_t i = 0; i < n_dec; ++i) {
                in[size_t(i)].kind =
                    static_cast<int32_t>(pt.rows == 1 ? fn::round::kind::generate : fn::round::kind::verify);
                in[size_t(i)].rows = pt.rows;
                in[size_t(i)].n_before = 64 * i;
                in[size_t(i)].tok[0] = 1;
                in[size_t(i)].tok[1] = drafts[0];
                in[size_t(i)].tok[2] = drafts[1];
            }
            ok = ok && T.put_desc(in);
            cudaGraphExec_t ge[2] = {nullptr, nullptr};
            for (int32_t r = 0; r < 2 && ok; ++r) {
                cudaSetDevice(T.c[r]->dev);
                cudaGraph_t g = nullptr;
                ok = cuda_ok(cudaStreamBeginCapture(T.c[r]->st, cudaStreamCaptureModeThreadLocal), "capture");
                const bool enq = ok && T.enqueue_batch(in, 0, false, r);
                ok = cuda_ok(cudaStreamEndCapture(T.c[r]->st, &g), "end capture") && enq;
                ok = ok && cuda_ok(cudaGraphInstantiate(&ge[r], g, 0), "instantiate") &&
                     cuda_ok(cudaGraphUpload(ge[r], T.c[r]->st), "upload") &&
                     cuda_ok(cudaStreamSynchronize(T.c[r]->st), "upload");
                if (g != nullptr) {
                    cudaGraphDestroy(g);
                }
            }
            std::vector<double> span[2];
            for (int32_t rep = 0; rep < n_warm + n_timed && ok; ++rep) {
                for (int32_t r = 0; r < 2 && ok; ++r) {
                    cudaSetDevice(T.c[r]->dev);
                    ok = cuda_ok(cudaEventRecord(ev[r][0], T.c[r]->st), "record") &&
                         cuda_ok(cudaGraphLaunch(ge[r], T.c[r]->st), "launch") &&
                         cuda_ok(cudaEventRecord(ev[r][1], T.c[r]->st), "record");
                }
                ok = ok && T.sync();
                for (int32_t r = 0; r < 2 && ok && rep >= n_warm; ++r) {
                    float ms = 0;
                    ok = cuda_ok(cudaEventElapsedTime(&ms, ev[r][0], ev[r][1]), "elapsed");
                    span[r].push_back(1e3 * ms / n_dec);
                }
            }
            uint32_t e[2] = {0, 0};
            ok = ok && T.errs(e);
            for (int32_t r = 0; r < 2; ++r) {
                if (ge[r] != nullptr) {
                    cudaSetDevice(T.c[r]->dev);
                    cudaGraphExecDestroy(ge[r]);
                }
            }
            if (!ok || e[0] != 0 || e[1] != 0) {
                fprintf(stderr, "samp-bench: %s B %d: failed (errors 0x%x 0x%x)\n", pt.name.c_str(), pt.rows, e[0],
                        e[1]);
                rc = 1;
                ok = false;
                break;
            }
            const std::vector<stage> ss = stages_of(pt.o, pt.rows > 1);
            printf(
                "samp-bench: %-46s B %d%s: card 0 median %.2f p90 %.2f us, card 1 median %.2f p90 %.2f us a decision "
                "(%zu sampler launches a card)\n",
                pt.name.c_str(), pt.rows, pt.reject ? " rejected" : (pt.rows > 1 ? " accepted" : ""), med(span[0]),
                p90(span[0]), med(span[1]), p90(span[1]), ss.size());
            if (jf != nullptr) {
                fprintf(
                    jf,
                    "{\"path\": \"%s\", \"B\": %d, \"verify\": \"%s\", \"card0_median_us\": %.3f, \"card0_p90_us\": "
                    "%.3f, \"card1_median_us\": %.3f, \"card1_p90_us\": %.3f, \"decisions_a_graph\": %d, "
                    "\"timed_replays\": %d, \"logits\": \"TG1 ref-decode rows 0-2, L2-warm\"}\n",
                    pt.name.c_str(), pt.rows,
                    pt.rows == 1 ? "none"
                    : pt.reject  ? "rejected"
                                 : "accepted",
                    med(span[0]), p90(span[0]), med(span[1]), p90(span[1]), n_dec, n_timed);
            }
        }
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(T.c[r]->dev);
            for (cudaEvent_t x : ev[r]) {
                if (x != nullptr) {
                    cudaEventDestroy(x);
                }
            }
        }
        rc = ok ? rc : 1;
    }
    if (jf != nullptr) {
        fclose(jf);
    }
    printf("samp-bench: %s\n", rc == 0 ? "done" : "FAILED");
    return rc;
}
