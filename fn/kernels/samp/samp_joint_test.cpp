#include "fn/kernels/samp/samp_joint_test.h"

#include "fn/kernels/round/round.h"
#include "fn/kernels/samp/samp.h"
#include "fn/kernels/samp/samp_harness.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace fn::samp;
using namespace fn::samp::harness;

constexpr int32_t nd = 32768;         // a card's draft head rows (the joint draft's logits row)
constexpr int32_t steps = max_drafts; // d = 4 chained picks
constexpr int32_t lanes_max = 4;      // fn::xchg::max_lanes
constexpr int32_t not_picked = -7;    // a proposal slot before its pick
static_assert(lanes_max <= max_rows, "a lane's list is its own row of samp_a's slot");

struct gen {
    uint64_t s;
    explicit gen(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    double u() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (static_cast<double>(s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
    float gauss() { return static_cast<float>(std::sqrt(-2.0 * std::log(u())) * std::cos(6.283185307179586 * u())); }
};

// every lane's picks of one run: per card, its four proposals and q_list rows
struct lane_out {
    int32_t prop[2][steps];
    qlist q[2][steps];
};

struct fixture {
    bool loop = false;
    int32_t dev[2] = {0, 0};
    std::unique_ptr<fn::xchg::rings> rg;
    std::unique_ptr<card_s> cs[lanes_max][2];
    tester T[lanes_max];
    int32_t * ids[2] = {nullptr, nullptr}; // the draft head's ID map, a card
    float * jl[2] = {nullptr, nullptr};    // the joint step's logits rows [steps][lanes_max][nd], a card

    ~fixture() {
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(dev[r]);
            cudaFree(ids[r]);
            cudaFree(jl[r]);
        }
    }

    // every kernel on lane 0's streams (one stream a card, as the joint draft's frame)
    cudaStream_t st(int32_t r) const { return T[0].st(r); }
    int32_t sdev(int32_t r) const { return loop ? dev[0] : dev[r]; }

    pick_io io(int32_t k, int32_t r, int32_t j, bool joint) const {
        card_s & c = *T[k].c[r];
        pick_io o;
        o.logits = joint ? jl[r] + (int64_t(j) * lanes_max + k) * nd : c.logits.p + int64_t(j) * nd;
        o.n = nd;
        o.ids = ids[r];
        o.live = c.desc.p;
        o.live_offset = j + 1;
        o.t = j;
        o.dtok = c.desc.p->argmax;
        o.n_dtok = j;
        o.proposal = c.desc.p->argmax + j;
        return o;
    }
};

// the host's copies and sets (default stream) complete before the lanes' non-blocking streams read them
bool sync_devices(const fixture & F) {
    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(F.dev[r]);
        if (!cuda_ok(cudaDeviceSynchronize(), "device sync")) {
            return false;
        }
    }
    return true;
}

bool setup(fixture & F, int32_t dev0, int32_t dev1) {
    F.loop = dev0 == dev1;
    F.dev[0] = dev0;
    F.dev[1] = dev1;
    F.rg = std::make_unique<fn::xchg::rings>(dev0, dev1);
    if (!F.rg->ok()) {
        return false;
    }
    for (int32_t k = 0; k < lanes_max; ++k) {
        for (int32_t r = 0; r < 2; ++r) {
            F.cs[k][r] = std::make_unique<card_s>(F.dev[r]);
            F.T[k].c[r] = F.cs[k][r].get();
        }
        F.T[k].rg = F.rg.get(); // every lane's views on the one pair of rings (the joint draft shares lane 0's)
        F.T[k].loop = F.loop;
        if (!F.T[k].init()) {
            return false;
        }
    }
    // the draft subset as samp-test's draft pick: card r takes the sorted subset's ranks r, r + 2, ...
    for (int32_t r = 0; r < 2; ++r) {
        std::vector<int32_t> ids(nd);
        for (int32_t i = 0; i < nd; ++i) {
            const int32_t g = 2 * i + r;
            ids[size_t(i)] = g * 3 + (g % 3);
        }
        cudaSetDevice(F.dev[r]);
        if (!cuda_ok(cudaMalloc(&F.ids[r], size_t(nd) * 4), "ids") ||
            !cuda_ok(cudaMemcpy(F.ids[r], ids.data(), size_t(nd) * 4, cudaMemcpyHostToDevice), "ids") ||
            !cuda_ok(cudaMalloc(&F.jl[r], size_t(steps) * lanes_max * nd * 4), "joint logits")) {
            return false;
        }
    }
    // each lane's own options; lane 1 with penalties over a window
    for (int32_t k = 0; k < lanes_max; ++k) {
        options o;
        o.seed = 101 + uint64_t(k);
        o.request = 7 + uint64_t(k);
        std::vector<int32_t> led;
        switch (k) {
            case 0:
                o.temp = 1.0f;
                break;
            case 1:
                o.temp = 0.8f;
                o.top_k = 40;
                o.penalty_last_n = 64;
                o.penalty_repeat = 1.25f;
                o.penalty_freq = 0.05f;
                o.penalty_present = 0.1f;
                for (int32_t i = 0; i < 80; ++i) {
                    led.push_back(((i * 37) % (2 * nd)) * 3 + (((i * 37) % (2 * nd)) % 3));
                }
                break;
            case 2:
                o.temp = 1.0f;
                o.top_k = 256;
                o.top_p = 1.0f;
                o.min_p = 0.0f;
                break;
            default:
                o.temp = 1.5f;
                o.top_k = 5;
                o.top_p = 0.5f;
                break;
        }
        if (!F.T[k].set_options(o, led)) {
            return false;
        }
    }
    return sync_devices(F);
}

// one round's logits (lane k, step j: a lone row j, and the joint row (j, k)) and descriptors (n_before)
bool put_round(fixture & F, int32_t round, const int32_t nb[lanes_max]) {
    for (int32_t k = 0; k < lanes_max; ++k) {
        for (int32_t j = 0; j < steps; ++j) {
            gen g(uint64_t(round) * 1000 + uint64_t(k) * 10 + uint64_t(j));
            std::vector<float> half[2] = {std::vector<float>(nd), std::vector<float>(nd)};
            for (int32_t i = 0; i < 2 * nd; ++i) {
                float x = 2.5f * g.gauss();
                if (i % 997 == 13) {
                    x += 6.0f; // a few strong candidates
                }
                half[i % 2][size_t(i / 2)] = x;
            }
            for (int32_t r = 0; r < 2; ++r) {
                cudaSetDevice(F.dev[r]);
                if (!cuda_ok(cudaMemcpy(F.T[k].c[r]->logits.p + int64_t(j) * nd, half[r].data(), size_t(nd) * 4,
                                        cudaMemcpyHostToDevice),
                             "lone logits") ||
                    !cuda_ok(cudaMemcpy(F.jl[r] + (int64_t(j) * lanes_max + k) * nd, half[r].data(), size_t(nd) * 4,
                                        cudaMemcpyHostToDevice),
                             "joint logits")) {
                    return false;
                }
            }
        }
        fn::round::desc d;
        memset(&d, 0, sizeof(d));
        d.n_before = nb[k];
        for (int32_t r = 0; r < 2; ++r) {
            cudaSetDevice(F.dev[r]);
            if (!cuda_ok(cudaMemcpy(F.T[k].c[r]->desc.p, &d, sizeof(d), cudaMemcpyHostToDevice), "desc")) {
                return false;
            }
        }
    }
    return sync_devices(F);
}

// before a run: every lane's proposal slots unpicked and its q lists zero
bool clear(fixture & F, int32_t U) {
    for (int32_t k = 0; k < U; ++k) {
        for (int32_t r = 0; r < 2; ++r) {
            card_s & c = *F.T[k].c[r];
            cudaSetDevice(F.dev[r]);
            const int32_t np[fn::round::max_rows] = {not_picked, not_picked, not_picked, not_picked, not_picked};
            if (!cuda_ok(cudaMemcpy(c.desc.p->argmax, np, sizeof(np), cudaMemcpyHostToDevice), "argmax") ||
                !cuda_ok(cudaMemset(c.ql.p, 0, c.ql.n * sizeof(qlist)), "q lists")) {
                return false;
            }
        }
    }
    return sync_devices(F);
}

bool collect(fixture & F, int32_t U, lane_out out[lanes_max]) {
    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(F.sdev(r));
        if (!cuda_ok(cudaStreamSynchronize(F.st(r)), "sync")) {
            return false;
        }
    }
    for (int32_t k = 0; k < U; ++k) {
        for (int32_t r = 0; r < 2; ++r) {
            card_s & c = *F.T[k].c[r];
            cudaSetDevice(F.dev[r]);
            if (!cuda_ok(cudaMemcpy(out[k].prop[r], c.desc.p->argmax, sizeof(out[k].prop[r]), cudaMemcpyDeviceToHost),
                         "proposals") ||
                !cuda_ok(cudaMemcpy(out[k].q[r], c.ql.p, sizeof(out[k].q[r]), cudaMemcpyDeviceToHost), "q lists")) {
                return false;
            }
        }
    }
    return true;
}

// each lane alone, d = 4 picks, every lane's list in row 0 (the lone draft step's pick)
bool run_lone(fixture & F, int32_t U, lane_out out[lanes_max]) {
    bool ok = clear(F, U);
    for (int32_t k = 0; k < U && ok; ++k) {
        for (int32_t j = 0; j < steps && ok; ++j) {
            for (int32_t stg = 0; stg < 2 && ok; ++stg) {
                for (int32_t r = 0; r < 2 && ok; ++r) {
                    cudaSetDevice(F.sdev(r));
                    ok = pick_stage(F.T[k].c[r]->v, F.io(k, r, j, false), stg, F.st(r));
                }
            }
        }
    }
    return ok && collect(F, U, out);
}

// the joint pick of step j's stage stg on card r (all U lanes, pick_lanes_stage)
bool joint_stage(fixture & F, int32_t U, int32_t j, int32_t stg, int32_t r) {
    const card_view * v[lanes_max] = {};
    pick_io io[lanes_max];
    for (int32_t k = 0; k < U; ++k) {
        v[k] = &F.T[k].c[r]->v;
        io[k] = F.io(k, r, j, true);
    }
    cudaSetDevice(F.sdev(r));
    return pick_lanes_stage(v, io, U, stg, F.st(r));
}

bool run_joint_eager(fixture & F, int32_t U, lane_out out[lanes_max]) {
    bool ok = clear(F, U);
    for (int32_t j = 0; j < steps && ok; ++j) {
        for (int32_t stg = 0; stg < 2 && ok; ++stg) {
            for (int32_t r = 0; r < 2 && ok; ++r) {
                ok = joint_stage(F, U, j, stg, r);
            }
        }
    }
    return ok && collect(F, U, out);
}

// captured: a graph a card (two devices, both launched, the cards' kernels exchanging as the joint draft's graphs do),
// or one graph of both cards' launches in order (loopback)
bool run_joint_graphs(fixture & F, int32_t U, lane_out out[lanes_max]) {
    cudaGraphExec_t x[2] = {nullptr, nullptr};
    const int32_t n_graph = F.loop ? 1 : 2;
    bool ok = clear(F, U);
    for (int32_t gi = 0; gi < n_graph && ok; ++gi) {
        cudaSetDevice(F.sdev(gi));
        cudaStream_t s = F.st(gi);
        ok = cuda_ok(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
        bool enq = ok;
        for (int32_t j = 0; j < steps && enq; ++j) {
            for (int32_t stg = 0; stg < 2 && enq; ++stg) {
                for (int32_t r = 0; r < 2 && enq; ++r) {
                    if (F.loop || r == gi) {
                        enq = joint_stage(F, U, j, stg, r);
                    }
                }
            }
        }
        cudaGraph_t g = nullptr;
        cudaSetDevice(F.sdev(gi));
        ok = ok && cuda_ok(cudaStreamEndCapture(s, &g), "end capture") && enq &&
             cuda_ok(cudaGraphInstantiate(&x[gi], g, 0), "instantiate");
        if (g != nullptr) {
            cudaGraphDestroy(g);
        }
    }
    for (int32_t gi = 0; gi < n_graph && ok; ++gi) {
        cudaSetDevice(F.sdev(gi));
        ok = cuda_ok(cudaGraphLaunch(x[gi], F.st(gi)), "graph launch");
    }
    ok = ok && collect(F, U, out);
    for (int32_t gi = 0; gi < n_graph; ++gi) {
        if (x[gi] != nullptr) {
            cudaSetDevice(F.sdev(gi));
            cudaGraphExecDestroy(x[gi]);
        }
    }
    return ok;
}

// lanes bitwise against the lone run (both cards); the lone run's cards equal and every pick made
std::string compare(const lane_out ref[lanes_max], const lane_out got[lanes_max], int32_t U, bool & ok) {
    int32_t bad_prop = 0, bad_q = 0;
    for (int32_t k = 0; k < U; ++k) {
        for (int32_t r = 0; r < 2; ++r) {
            for (int32_t j = 0; j < steps; ++j) {
                bad_prop += got[k].prop[r][j] != ref[k].prop[r][j] ? 1 : 0;
                bad_q += memcmp(&got[k].q[r][j], &ref[k].q[r][j], sizeof(qlist)) != 0 ? 1 : 0;
            }
        }
    }
    ok = bad_prop == 0 && bad_q == 0;
    return "proposals differing " + std::to_string(bad_prop) + ", q_list rows differing " + std::to_string(bad_q) +
           " of " + std::to_string(U * 2 * steps);
}

bool lone_sane(const lane_out ref[lanes_max], int32_t U, std::string & why) {
    int32_t unpicked = 0, cards = 0, empty = 0;
    for (int32_t k = 0; k < U; ++k) {
        for (int32_t j = 0; j < steps; ++j) {
            unpicked += ref[k].prop[0][j] == not_picked ? 1 : 0;
            cards +=
                ref[k].prop[0][j] != ref[k].prop[1][j] || memcmp(&ref[k].q[0][j], &ref[k].q[1][j], sizeof(qlist)) != 0
                    ? 1
                    : 0;
            empty += ref[k].q[0][j].n < 1 ? 1 : 0;
        }
    }
    why = "lone: unpicked " + std::to_string(unpicked) + ", cards differing " + std::to_string(cards) +
          ", empty q lists " + std::to_string(empty);
    return unpicked == 0 && cards == 0 && empty == 0;
}

} // namespace

int fnfl_dpj_pick_test(int dev0, int dev1, bool san) {
    if (dev0 != dev1 && !fn::xchg::enable_peer_access(dev0, dev1)) {
        printf("dpj-test: sampled pick: FAIL (no peer access between devices %d and %d)\n", dev0, dev1);
        return 1;
    }
    auto F = std::make_unique<fixture>();
    if (!setup(*F, dev0, dev1)) {
        printf("dpj-test: sampled pick: FAIL (setup)\n");
        return 1;
    }
    printf("dpj-test: sampled pick (FMU4): the joint draft step's sampled pick (pick_lanes_stage) against each lane's "
           "lone pick, d = 4, %s; lanes' options: T 1.0 top-k 20 / T 0.8 top-k 40 penalties / T 1.0 top-k 256 no "
           "top-p or min-p / T 1.5 top-k 5 top-p 0.5\n",
           F->loop ? ("loopback on device " + std::to_string(dev0)).c_str()
                   : ("devices " + std::to_string(dev0) + " and " + std::to_string(dev1)).c_str());
    const std::vector<int32_t> lane_set = san ? std::vector<int32_t>{2, 4} : std::vector<int32_t>{2, 3, 4};
    const int32_t rounds = san ? 1 : 3;
    int32_t n_fail = 0;
    for (const int32_t U : lane_set) {
        for (int32_t round = 0; round < rounds; ++round) {
            int32_t nb[lanes_max];
            for (int32_t k = 0; k < lanes_max; ++k) {
                nb[k] = 3000 + 977 * k + 13 * round + 5 * U; // positions of both parities across lanes and rounds
            }
            lane_out ref[lanes_max], eg[lanes_max], gr[lanes_max];
            std::string sane;
            const bool ran = put_round(*F, round, nb) && run_lone(*F, U, ref) && run_joint_eager(*F, U, eg) &&
                             run_joint_graphs(*F, U, gr);
            bool ok_e = false, ok_g = false;
            const bool ok_s = ran && lone_sane(ref, U, sane);
            const std::string ce = ran ? compare(ref, eg, U, ok_e) : std::string("not run");
            const std::string cg = ran ? compare(ref, gr, U, ok_g) : std::string("not run");
            uint32_t e[2] = {0, 0};
            const bool ok_w = F->T[0].errs(e) && e[0] == 0 && e[1] == 0;
            const bool ok = ran && ok_s && ok_e && ok_g && ok_w;
            printf("dpj-test: sampled pick U%d round %d: %s; eager: %s; graphs: %s; error words %08x %08x\n  %s\n", U,
                   round, sane.c_str(), ce.c_str(), cg.c_str(), e[0], e[1], ok ? "PASS" : "FAIL");
            n_fail += ok ? 0 : 1;
        }
    }
    return n_fail;
}
