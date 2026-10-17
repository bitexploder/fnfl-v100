// fnfl-drive rollback-test (SR1 step 6): one model load. The real target pass on both cards (fn/runtime/target.h),
// forced rounds over HX7's reference tokens with the test-only commits of k of B rows (forced_k), cuts within the last
// step to every j <= k (versions.h's primitive, the ledger to N_before + j), recoverable errors injected before commit
// on one card and on both, each followed by a forced round of fresh reference tokens; filler forced rounds of 4 between
// the cases so the script spans 512 positions.
//
// After every cut and every continuation the whole state is hashed: on each card every GDN layer's state and
// convolution history (logical version 0), the PLE history, the K and V of cells < N of every QSA layer, and on the
// index owners the raw index keys of cells < N and the summaries of blocks < N / 4. These states lie on the final
// ledger (a later cut never reaches below a later step's N_before), so one clean replay of the final ledger from zero
// state in forced rounds at B = 1 checks them all: its state at each recorded N, and its logits (both cards' shards)
// at the continuation rounds' positions, bitwise.

#include "fn/kernels/head/head.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/round/version.h"
#include "fn/loader/run_load.h"
#include "fn/model/model.h"
#include "fn/model/pass.h"
#include "fn/runtime/target.h"
#include "fn/runtime/target_cmds.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace rt = fn::runtime;
namespace md = fn::model;
namespace rd = fn::round;

constexpr int32_t VC = fn::head::n_vocab_card;
constexpr int32_t n_positions = 512;
constexpr int64_t gdn_floats = int64_t(md::gdn_heads_card) * md::gdn_head_dim * md::gdn_head_dim;
constexpr int64_t conv_floats = int64_t(md::gdn_conv_channels_card) * md::gdn_conv_history;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "rollback-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

uint64_t hash_bytes(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, b + i, 8);
        h = mix(h ^ w);
    }
    for (; i < n; ++i) {
        h = mix(h ^ b[i]);
    }
    return mix(h ^ n);
}

// the IDs as TG1's model-test reads them: every run of digits (deep.ids is comma-separated)
bool read_ids(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "rollback-test: cannot read %s\n", path.c_str());
        return false;
    }
    out.clear();
    long v = -1;
    char ch = 0;
    while (f.get(ch)) {
        if (ch >= '0' && ch <= '9') {
            v = (v < 0 ? 0 : v) * 10 + (ch - '0');
        } else if (v >= 0) {
            out.push_back(static_cast<int32_t>(v));
            v = -1;
        }
    }
    if (v >= 0) {
        out.push_back(static_cast<int32_t>(v));
    }
    return !out.empty();
}

// the state's hashes by object class, both cards
constexpr int32_t n_class = 7;
const char * const class_names[n_class] = {"GDN states", "GDN histories", "PLE history", "K cache",
                                           "V cache",    "index keys",    "summaries"};
struct state_hash {
    int64_t n = 0;
    uint64_t h[n_class] = {};
};

template <typename T> bool get(std::vector<T> & v, const T * src, size_t n) {
    v.resize(n);
    return n == 0 || cuda_ok(cudaMemcpy(v.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
}

bool snapshot(rt::target & t, md::model & m, state_hash & s) {
    const rt::ledger & l = t.slot();
    s = state_hash();
    s.n = l.n;
    for (int32_t i = 0; i < n_class; ++i) {
        s.h[i] = uint64_t(i) + 1;
    }
    const int64_t n = l.n;
    const int32_t blocks = fn::qsa::blocks_complete(static_cast<int32_t>(n));
    std::vector<float> f;
    std::vector<uint16_t> u;
    std::vector<float> one(fn::qsa::head_dim);
    for (int32_t c = 0; c < 2; ++c) {
        md::pass_card & p = t.card(c);
        uint32_t cur = 0;
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "sync") ||
            !cuda_ok(cudaMemcpy(&cur, &p.rstate->cur, 4, cudaMemcpyDeviceToHost), "cur")) {
            return false;
        }
        if (cur != l.cur) {
            fprintf(stderr, "rollback-test: card %d's committed slot %u is not the ledger's %u\n", c, cur, l.cur);
            return false;
        }
        const int64_t s0 = rd::version_slot(cur, 0);
        for (int32_t layer = 0; layer < 48; ++layer) {
            const md::layer_state & st = p.s[layer];
            const uint64_t tag = uint64_t(c) << 8 | uint64_t(layer);
            if (p.w[layer].gdn) {
                if (!get(f, st.gdn + s0 * gdn_floats, size_t(gdn_floats))) {
                    return false;
                }
                s.h[0] = hash_bytes(s.h[0] ^ tag, f.data(), f.size() * 4);
                if (!get(f, st.conv + s0 * conv_floats, size_t(conv_floats))) {
                    return false;
                }
                s.h[1] = hash_bytes(s.h[1] ^ tag, f.data(), f.size() * 4);
                continue;
            }
            if (!get(u, st.k_cache, size_t(n) * md::kv_head_dim)) {
                return false;
            }
            s.h[3] = hash_bytes(s.h[3] ^ tag, u.data(), u.size() * 2);
            if (!get(u, st.v_cache, size_t(n) * md::kv_head_dim)) {
                return false;
            }
            s.h[4] = hash_bytes(s.h[4] ^ tag, u.data(), u.size() * 2);
            if (st.idx_keys != nullptr) {
                if (!get(u, st.idx_keys, size_t(n) * fn::qsa::head_dim) ||
                    !get(f, st.idx_summaries, fn::qsa::summary_bytes(blocks) / 4)) {
                    return false;
                }
                s.h[5] = hash_bytes(s.h[5] ^ tag, u.data(), u.size() * 2);
                uint64_t h = s.h[6] ^ tag;
                for (int32_t b = 0; b < blocks; ++b) {
                    for (int32_t e = 0; e < fn::qsa::head_dim; ++e) {
                        one[size_t(e)] =
                            f[size_t(b / 32) * 4096 + size_t(e / 4) * 128 + size_t(b % 32) * 4 + size_t(e % 4)];
                    }
                    h = hash_bytes(h, one.data(), one.size() * 4);
                }
                s.h[6] = h;
            }
        }
        if (!get(f, p.ple_hist + s0 * fn::ple::hist_version_floats, size_t(fn::ple::hist_version_floats))) {
            return false;
        }
        s.h[2] = hash_bytes(s.h[2] ^ uint64_t(c), f.data(), f.size() * 4);
    }
    return true;
}

// each row's logits (both cards' shards) by position
bool logit_rows(rt::target & t, md::model & m, int32_t rows, int64_t n_before, std::map<int64_t, uint64_t> & out) {
    std::vector<float> lg;
    uint64_t h[4] = {0, 0, 0, 0};
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "sync") ||
            !get(lg, t.card(c).logits, size_t(rows) * VC)) {
            return false;
        }
        for (int32_t r = 0; r < rows; ++r) {
            h[r] = hash_bytes(h[r] ^ uint64_t(c), &lg[size_t(r) * VC], size_t(VC) * 4);
        }
    }
    for (int32_t r = 0; r < rows; ++r) {
        out[n_before + r] = h[r];
    }
    return true;
}

struct record {
    std::string what;
    int32_t B = 0, k = 0, j = 0, fail = 0;
    state_hash s;
};

// Mode retain (zero weights, no model data): the retained row (FC-PF2) after cuts. (1) prompt [a, b], forced(1),
// forced(1), cut(0, 1): the bootstrap refuses. (2) prompt [a, b], forced(1), forced_k(1, k = 0), cut(0, 1): refuses.
// (c) prompt [a, b], forced(2), cut(1, 1): refuses (SR1 keeps nothing retained for 1 <= j < k, no reselection);
// cut(2, 2): the bootstrap runs. (3) prompt [a, b], forced(2), the retained row's logits overwritten with seeded random
// values on both cards; bootstrap, then cut back to the prompt (j = 0, n_keep 2), then bootstrap again: the same token,
// equal to the CPU argmax of those logits (ties: the lower ID), at the same output position (O = 1 both times).
bool retained_checks(rt::target & t, md::model & m, const std::vector<int32_t> & pool) {
    std::string err;
    rt::round_result r;
    const std::vector<int32_t> ab = {pool[0], pool[1]};
    const auto fresh = [&]() {
        if (!t.reset(err)) {
            return false;
        }
        t.set_prompt(ab);
        return true;
    };
    const auto refuses = [&](const char * what) {
        std::string e;
        const bool ran = t.bootstrap(r, e);
        printf("rollback-test retain: %s: bootstrap %s (%s); retained %d, U %lld\n", what, ran ? "RAN" : "refused",
               ran ? "-" : e.c_str(), t.slot().retained ? 1 : 0, static_cast<long long>(t.slot().u));
        return !ran;
    };
    bool ok = fresh() && t.forced(1, r, err) && t.forced(1, r, err) && t.cut(0, 1, err) &&
              refuses("(1) forced(1), forced(1), cut(0, 1)");
    ok = ok && fresh() && t.forced(1, r, err) && t.forced_k(1, 0, 0, r, err) && t.cut(0, 1, err) &&
         refuses("(2) forced(1), forced_k(1, k 0), cut(0, 1)");
    ok = ok && fresh() && t.forced(2, r, err) && t.cut(1, 1, err) && refuses("(c) forced(2), cut(1, 1)");
    bool ran_k = false;
    if (ok && fresh() && t.forced(2, r, err) && t.cut(2, 2, err)) {
        ran_k = t.bootstrap(r, err);
        printf("rollback-test retain: (c) forced(2), cut(2, 2) (j = k, nothing cut): bootstrap %s, token %d\n",
               ran_k ? "ran" : "REFUSED", r.out[0]);
    }
    ok = ok && ran_k;
    // (3) synthetic retained logits, both shards
    int32_t tok1 = -1, tok2 = -1, want = -1;
    int64_t o1 = -1, o2 = -1;
    if (ok && fresh() && t.forced(2, r, err)) {
        const int32_t row = t.slot().retained_row;
        float best = -1e30f;
        for (int32_t c = 0; c < 2 && ok; ++c) {
            std::vector<float> v(static_cast<size_t>(VC));
            uint64_t z = 0x5352315245ull + uint64_t(c);
            for (size_t i = 0; i < v.size(); ++i) {
                z = mix(z);
                v[i] = float(z >> 40) / float(1u << 24); // [0, 1): ties possible, the lower ID wins
                if (v[i] > best) {
                    best = v[i];
                    want = c * VC + static_cast<int32_t>(i);
                }
            }
            ok = cuda_ok(cudaSetDevice(m.device[c]), "device") &&
                 cuda_ok(
                     cudaMemcpy(t.card(c).logits + int64_t(row) * VC, v.data(), v.size() * 4, cudaMemcpyHostToDevice),
                     "logits");
        }
        ok = ok && t.bootstrap(r, err);
        tok1 = r.out[0];
        o1 = t.slot().o;
        ok = ok && t.cut(0, 2, err) && t.bootstrap(r, err);
        tok2 = r.out[0];
        o2 = t.slot().o;
        printf("rollback-test retain: (3) bootstrap %d at output %lld; cut back to the prompt; bootstrap again %d at "
               "output %lld; CPU argmax of the retained logits %d\n",
               tok1, static_cast<long long>(o1), tok2, static_cast<long long>(o2), want);
        ok = ok && tok1 == want && tok2 == want && o1 == 1 && o2 == 1;
    }
    // (a) FC-RD2 after a head-running round: prompt [a, b], forced(1) (retained row 0), then forced_k(1, k = 1) with an
    // error injected on card 0 alone: the restore keeps the ledger but leaves nothing retained; the bootstrap refuses
    if (ok && fresh() && t.forced(1, r, err)) {
        const bool before = t.slot().retained;
        const bool failed = !t.forced_k(1, 1, 1, r, err);
        err.clear();
        printf("rollback-test retain: (a) forced(1) (retained %d), forced_k(1, k 1) with an error on card 0: %s, "
               "recovered %d, N %lld, cur %u / %u reported\n",
               before ? 1 : 0, failed ? "failed" : "COMMITTED", r.recovered ? 1 : 0, static_cast<long long>(t.slot().n),
               r.cur[0], r.cur[1]);
        ok = before && failed && r.recovered && !t.slot().retained && t.slot().n == 1 &&
             refuses("(a) after the restore");
    }
    // (b) FC-RD2 in a bootstrap: prompt [a, b], forced(2), synthetic retained logits, a bootstrap with an error injected
    // on card 1 alone: the restore keeps the ledger and the retained row; the next bootstrap decides the same token
    if (ok && fresh() && t.forced(2, r, err)) {
        const int32_t row = t.slot().retained_row;
        float best = -1e30f;
        int32_t want_b = -1;
        for (int32_t c = 0; c < 2 && ok; ++c) {
            std::vector<float> v(static_cast<size_t>(VC));
            uint64_t z = 0x5352315246ull + uint64_t(c);
            for (size_t i = 0; i < v.size(); ++i) {
                z = mix(z);
                v[i] = float(z >> 40) / float(1u << 24);
                if (v[i] > best) {
                    best = v[i];
                    want_b = c * VC + static_cast<int32_t>(i);
                }
            }
            ok = cuda_ok(cudaSetDevice(m.device[c]), "device") &&
                 cuda_ok(
                     cudaMemcpy(t.card(c).logits + int64_t(row) * VC, v.data(), v.size() * 4, cudaMemcpyHostToDevice),
                     "logits");
        }
        const bool failed = ok && !t.bootstrap(r, err, 2);
        err.clear();
        const int32_t decided = r.out[0];
        const bool kept = t.slot().retained && t.slot().retained_row == row && t.slot().u == 0 && t.slot().o == 0;
        printf("rollback-test retain: (b) bootstrap with an error on card 1: %s, recovered %d, its decision %d; after "
               "the restore: retained %d (row %d), U %lld, O %lld\n",
               failed ? "failed" : "COMMITTED", r.recovered ? 1 : 0, decided, t.slot().retained ? 1 : 0,
               t.slot().retained_row, static_cast<long long>(t.slot().u), static_cast<long long>(t.slot().o));
        ok = ok && failed && r.recovered && kept && t.bootstrap(r, err);
        printf("rollback-test retain: (b) bootstrap again: %d at output %lld; CPU argmax of the retained logits %d\n",
               r.out[0], static_cast<long long>(t.slot().o), want_b);
        ok = ok && r.out[0] == want_b && decided == want_b && t.slot().o == 1;
    }
    if (!ok && !err.empty()) {
        fprintf(stderr, "rollback-test retain: %s\n", err.c_str());
    }
    return ok;
}

} // namespace

bool fnfl_state_hashes(fn::runtime::target & t, fn::model::model & m, uint64_t h[7]) {
    state_hash s;
    if (!snapshot(t, m, s)) {
        return false;
    }
    for (int32_t i = 0; i < n_class; ++i) {
        h[i] = s.h[i];
    }
    return true;
}

int fnfl_rollback_test(const fnfl_target_args & a) {
    std::vector<int32_t> pool;
    if (!read_ids(a.ids, pool) || pool.size() < size_t{4} * n_positions) {
        fprintf(stderr, "rollback-test: %s holds too few tokens\n", a.ids.c_str());
        return 1;
    }
    FILE * tf = fopen((a.out_dir + "/rollback-ops.tsv").c_str(), "w");
    if (tf == nullptr) {
        fprintf(stderr, "rollback-test: cannot write %s/rollback-ops.tsv\n", a.out_dir.c_str());
        return 1;
    }
    struct closer {
        FILE * f;
        ~closer() { fclose(f); }
    } tf_hold{tf};
    fprintf(tf, "op\twhat\tB\tk\tj\tfail\tN\tcur\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", class_names[0], class_names[1],
            class_names[2], class_names[3], class_names[4], class_names[5], class_names[6]);
    printf("rollback-test: ids %s (%zu tokens), model %s, out %s, 8,192 cells\n", a.ids.c_str(), pool.size(),
           a.model_path.c_str(), a.out_dir.c_str());

    auto m = std::make_unique<md::model>();
    const bool dry = a.mode == "dry" || a.mode == "retain";
    if (dry) {
        if (!fnfl_load_dry(a, 8192, *m)) {
            return 1;
        }
        printf("rollback-test: DRY: zero weights, no model data on the cards (plumbing only; every state stays 0)\n");
    } else {
        fn::loader::load_params lp;
        lp.model_path = a.model_path;
        lp.draft_path = a.draft_path;
        lp.ple_table = a.ple_table;
        fn::loader::run_load_report lr;
        std::string err;
        if (!fn::loader::load_for_run(lp, 8192, *m, lr, err)) {
            fprintf(stderr, "rollback-test: load: %s\n", err.c_str());
            return 1;
        }
        printf("rollback-test: model loaded in %.2f s\n", lr.total_s);
        for (const std::string & line : fn::loader::ple_table_lines(lr)) {
            printf("rollback-test: %s\n", line.c_str());
        }
    }
    rt::target_options o;
    o.model_path = a.model_path;
    o.graph_nodes = a.out_dir; // the real pass's captured graphs: one node list a card a B
    rt::target t;
    std::string err;
    if (!t.init(*m, o, err) || !t.reset(err)) {
        fprintf(stderr, "rollback-test: init: %s\n", err.c_str());
        return 1;
    }
    if (a.mode == "retain") {
        const bool pass = retained_checks(t, *m, pool);
        printf("rollback-test retain: %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    t.set_prompt({});
    const auto t0 = std::chrono::steady_clock::now();

    // the script
    size_t next = 0;
    const auto append = [&](int32_t n) {
        t.append(std::vector<int32_t>(pool.begin() + std::ptrdiff_t(next), pool.begin() + std::ptrdiff_t(next + n)));
        next += size_t(n);
    };
    std::vector<record> recs;
    std::map<int64_t, uint64_t> cont_logits; // the continuation rounds' rows
    int64_t ops = 0;
    rt::round_result r;
    bool ok = true;
    int32_t errors_ok = 0, errors_bad = 0, healthy_committed = 0;
    uint32_t slots = 1;
    const auto rec = [&](const char * what, int32_t B, int32_t k, int32_t j, int32_t fail) {
        record x;
        x.what = what;
        x.B = B;
        x.k = k;
        x.j = j;
        x.fail = fail;
        ok = ok && snapshot(t, *m, x.s);
        if (ok) {
            fprintf(tf, "%lld\t%s\t%d\t%d\t%d\t%d\t%lld\t%u", static_cast<long long>(ops++), what, B, k, j, fail,
                    static_cast<long long>(x.s.n), t.slot().cur);
            for (const uint64_t h : x.s.h) {
                fprintf(tf, "\t%016llx", static_cast<unsigned long long>(h));
            }
            fprintf(tf, "\n");
            recs.push_back(x);
            slots |= 1u << t.slot().cur;
        }
    };
    const auto forced = [&](int32_t rows, bool keep_logits) {
        append(rows);
        const int64_t n0 = t.slot().n;
        ok = ok && t.forced(rows, r, err) && (!keep_logits || logit_rows(t, *m, rows, n0, cont_logits));
    };
    struct case_t {
        int32_t B, k, j, fail;
    };
    std::vector<case_t> cases;
    for (int32_t B = 1; B <= 4; ++B) {
        for (int32_t k = 0; k <= B; ++k) {
            for (int32_t j = 0; j <= k; ++j) {
                cases.push_back({B, k, j, 0});
            }
        }
    }
    cases.push_back({3, 3, 0, 1}); // the one-card errors, then both cards
    cases.push_back({3, 2, 0, 2});
    cases.push_back({2, 2, 0, 3});
    const int64_t spacing = (n_positions - 24) / int64_t(cases.size());
    for (size_t i = 0; ok && i < cases.size(); ++i) {
        const case_t & c = cases[i];
        while (ok && t.slot().n + 4 <= int64_t(i) * spacing) {
            forced(4, false);
        }
        append(c.B);
        const int64_t n0 = t.slot().n;
        const uint32_t pre = t.slot().cur;
        if (c.fail != 0) {
            // FC-RD2: the injected error on c.fail's cards; both cards must end at the pre-step slot
            const bool committed = t.forced_k(c.B, c.k, c.fail, r, err);
            uint32_t now[2] = {0, 0};
            ok = ok && !committed && r.recovered && t.versions().read_cur(now, err);
            const bool back = ok && now[0] == pre && now[1] == pre && t.slot().cur == pre && t.slot().n == n0;
            for (int32_t card = 0; card < 2; ++card) {
                if (((c.fail >> card) & 1) == 0 && r.cur[card] == static_cast<uint32_t>(rd::version_slot(pre, c.k))) {
                    ++healthy_committed;
                }
            }
            (back ? errors_ok : errors_bad)++;
            printf("rollback-test: error on card mask %d (B %d, k %d) at N %lld: reported cur %u / %u, both cards back "
                   "at cur_pre %u: %s\n",
                   c.fail, c.B, c.k, static_cast<long long>(n0), r.cur[0], r.cur[1], pre, back ? "yes" : "NO");
            err.clear();
            rec("after error", c.B, c.k, 0, c.fail);
            // the rows run again as a forced round
            ok = ok && t.forced(c.B, r, err) && logit_rows(t, *m, c.B, n0, cont_logits);
            rec("continue", c.B, c.k, 0, c.fail);
            continue;
        }
        ok = ok && t.forced_k(c.B, c.k, 0, r, err) && t.cut(c.j, n0 + c.j, err);
        rec("cut", c.B, c.k, c.j, 0);
        forced(1 + int32_t(i % 4), true);
        rec("continue", c.B, c.k, c.j, 0);
    }
    while (ok && t.slot().n < n_positions) {
        forced(static_cast<int32_t>(std::min<int64_t>(4, n_positions - t.slot().n)), false);
    }
    if (!ok) {
        fprintf(stderr, "rollback-test: the script stopped: %s\n", err.c_str());
        printf("rollback-test: FAIL\n");
        return 1;
    }
    const std::vector<int32_t> final_ledger(t.slot().tok.begin(), t.slot().tok.begin() + t.slot().n);
    printf("rollback-test: script: %zu cases, %zu records, ledger N %lld (L %zu); %.1f s\n", cases.size(), recs.size(),
           static_cast<long long>(t.slot().n), t.slot().tok.size(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // the clean replay: zero state, the final ledger at B = 1
    std::map<int64_t, state_hash> want;
    std::map<int64_t, uint64_t> rep_logits;
    for (const record & x : recs) {
        want[x.s.n] = state_hash();
    }
    if (!t.reset(err)) {
        fprintf(stderr, "rollback-test: reset: %s\n", err.c_str());
        return 1;
    }
    t.set_prompt(final_ledger);
    if (want.count(0) != 0) {
        ok = snapshot(t, *m, want[0]);
    }
    for (int64_t p = 0; ok && p < int64_t(final_ledger.size()); ++p) {
        ok = t.forced(1, r, err) && logit_rows(t, *m, 1, p, rep_logits) &&
             (want.count(p + 1) == 0 || snapshot(t, *m, want[p + 1]));
    }
    if (!ok) {
        fprintf(stderr, "rollback-test: the replay stopped: %s\n", err.c_str());
        printf("rollback-test: FAIL\n");
        return 1;
    }
    int64_t state_bad = 0, logit_bad = 0;
    for (const record & x : recs) {
        const state_hash & w = want[x.s.n];
        std::string diff;
        for (int32_t i = 0; i < n_class; ++i) {
            if (w.h[i] != x.s.h[i]) {
                diff += std::string(diff.empty() ? "" : ", ") + class_names[i];
            }
        }
        if (!diff.empty()) {
            ++state_bad;
            printf("rollback-test: MISMATCH after %s (B %d, k %d, j %d, fail %d) at N %lld: %s\n", x.what.c_str(), x.B,
                   x.k, x.j, x.fail, static_cast<long long>(x.s.n), diff.c_str());
        }
    }
    for (const auto & [p, h] : cont_logits) {
        if (rep_logits.count(p) == 0 || rep_logits[p] != h) {
            ++logit_bad;
            printf("rollback-test: LOGITS MISMATCH at position %lld\n", static_cast<long long>(p));
        }
    }
    printf("rollback-test: %zu states (after each cut and each continuation) against the clean replay at B = 1: %lld "
           "differ; the next rounds' logits at %zu positions: %lld differ\n",
           recs.size(), static_cast<long long>(state_bad), cont_logits.size(), static_cast<long long>(logit_bad));
    printf("rollback-test: injected errors: %d with both cards back at cur_pre, %d not; a clear card had committed "
           "before the restore in %d; committed slots visited: %d of 6 (SP2)\n",
           errors_ok, errors_bad, healthy_committed, __builtin_popcount(slots));
    printf("rollback-test: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const bool pass = state_bad == 0 && logit_bad == 0 && errors_bad == 0 && errors_ok == 3 && healthy_committed >= 2 &&
                      !recs.empty() && !cont_logits.empty();
    printf("rollback-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
