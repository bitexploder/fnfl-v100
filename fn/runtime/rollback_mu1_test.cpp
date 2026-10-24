// fnfl-drive rollback-test IDS OUT mu1 (MU1, one model load: the real weights, so every state is nonzero): a slot's
// read run in layer groups, paused at every layer boundary with other slots' work between the groups, against the same
// read run straight through on another slot, from the same restore point; four slots of 65,536 cells on both cards.
//
// The states compared (each slot's frame, content only): every GDN layer's committed matrix and history, the PLE
// history, the K and V of cells < N, the index keys and summaries (rollback-test's state hash), and the retained
// logits, the hidden carry, the draft's retained carry and its K and V of cells < its length. Token identity of the
// following greedy steps is recorded apart from the states.
//
// The script, on slots A (0, straight through), B (1, paused), W (2, a writer stepping between B's groups) and S (3,
// a short reader TU serves out of turn):
//   1. A and B each read P0 (1,500 tokens) and take 3 greedy steps (their GDN selectors and committed versions leave the
//      initial bank), straight through; their states compared.
//   2. A reads the pure extension P1 (9,000 tokens: two chunks, the second from a noninitial bank) in one step a chunk;
//      B reads it one layer a group (every layer boundary, the pending FFN exchange live at each), with one of W's
//      steps after every group, and at layer 20 of its first chunk S's whole read (P3, 300 tokens) in the other-reader
//      lane while B's chunk is suspended. Compared: A against B; S against S's own lone read (step 5).
//   3. A and B take 8 greedy steps (tokens compared); then each truncates inside its last step and reads a 300-token
//      extension, then 4 steps: states and tokens compared.
//   4. Cancellation on B: before any chunk (nothing new kept); during a chunk (it finishes: BG1, the first chunk kept),
//      then the read resumed from its checkpoint; while suspended (cancelled with S's out-of-turn read in progress,
//      B's chunk finished after it), then resumed. Each resumed read's state against A's of the same prompt.
//   5. S's lone read of P3 (after a reset), against its out-of-turn read in step 2.
//   6. FMU1's presence-based read paths, each against A's straight read of Q in step 4: A reads Q lone (each chunk one
//      step of one 48-layer group, PF1's kernel order); B reads Q lone for its first chunk and with a writer present
//      for its second (a peer arriving at a chunk boundary: 1-layer groups, the chunk's end and its finish steps of
//      their own, W's step between every two, the chunk staged ahead by P); B reads Q with a writer present for 10
//      groups of its first chunk, then lone (the writer leaving mid-chunk), its second chunk staged by P. Then 4
//      greedy steps on A and B: tokens compared.
//   7. FMU2: joint verify rounds against lone steps (I3), each slot from the same state: slots 0..3 each read their
//      own prompt R0..R3 (different lengths) and take 12 lone greedy steps, the reference (tokens and states); each
//      reads its prompt again from token 0 (step 4's equality) and takes 12 steps as the members of one step_joint
//      call each step (four lanes); then again with the lane orders rotated every step and every second step as two
//      two-member calls. Each slot's tokens and states against its reference; the joint rounds counted.
//
// mu1-joint (one model load): the four slots read R0..R3 and take two step_joint calls, the bootstraps and one
// four-lane joint round (its capture and launch), then exit. mu1-joint-eager (the sanitizers' run): the same on the
// eager engine, the joint round enqueued in phases (Compute Sanitizer runs one kernel at a time: the graphs'
// cross-card polling cannot progress under it).
//
// IDS: a token file (rollback-test's form); when it is missing or short, a seeded sequence of real vocabulary IDs (the
// states need nonzero values, not a meaningful text). PASS (exit 0) needs every comparison equal.

#include "fn/kernels/head/head.h"
#include "fn/loader/run_load.h"
#include "fn/model/model.h"
#include "fn/runtime/joint.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/target_cmds.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace md = fn::model;
namespace rt = fn::runtime;

int g_bad = 0;

bool put(const std::string & what, bool ok, const std::string & detail = std::string()) {
    printf("rollback-test mu1: %s: %s%s%s\n", what.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : " (",
           detail.empty() ? "" : (detail + ")").c_str());
    g_bad += ok ? 0 : 1;
    return ok;
}

uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

uint64_t hash_dev(uint64_t h, int32_t dev, const void * p, size_t n) {
    std::vector<uint8_t> v(n);
    if (n > 0 && (cudaSetDevice(dev) != cudaSuccess || cudaMemcpy(v.data(), p, n, cudaMemcpyDefault) != cudaSuccess)) {
        return 0;
    }
    for (size_t i = 0; i + 8 <= n; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, v.data() + i, 8);
        h = mix64(h ^ w);
    }
    return mix64(h ^ n);
}

// one slot's state: rollback-test's seven classes, then the retained logits row, the hidden carry, the draft's
// retained carry and its K and V of cells < its length
struct slot_hash {
    int64_t n = 0, draft_n = 0;
    uint64_t h[11] = {};
    bool ok = false;
};
const char * const hash_names[11] = {"GDN states",   "GDN histories", "PLE history",  "K cache",
                                     "V cache",      "index keys",    "summaries",    "retained logits",
                                     "hidden carry", "draft carry",   "draft K and V"};

slot_hash take(rt::engine & e, int32_t slot) {
    slot_hash s;
    rt::target & t = e.test_target(slot);
    md::model & m = e.test_model();
    if (!fnfl_state_hashes(t, m, s.h)) {
        return s;
    }
    s.n = t.slot().n;
    s.draft_n = t.draft_n();
    for (int32_t c = 0; c < 2; ++c) {
        const int32_t dev = m.device[c];
        s.h[7] = hash_dev(s.h[7] ^ uint64_t(c), dev, t.card(c).logits, size_t(fn::head::n_vocab_card) * 4);
        s.h[8] = hash_dev(s.h[8] ^ uint64_t(c), dev, t.card(c).carry, size_t(5) * 4 * 1280 * 4);
        s.h[9] = hash_dev(s.h[9] ^ uint64_t(c), dev, t.draft(c).carry_prev, size_t(4) * 1280 * 4);
        const size_t kv = size_t(std::max<int64_t>(0, s.draft_n)) * 256 * 2;
        s.h[10] = hash_dev(hash_dev(s.h[10] ^ uint64_t(c), dev, t.draft(c).k_cache, kv), dev, t.draft(c).v_cache, kv);
    }
    s.ok = true;
    return s;
}

std::string diff(const slot_hash & a, const slot_hash & b) {
    if (!a.ok || !b.ok) {
        return "not read";
    }
    std::string d;
    if (a.n != b.n || a.draft_n != b.draft_n) {
        d = "N " + std::to_string(a.n) + " / " + std::to_string(b.n) + ", draft " + std::to_string(a.draft_n) + " / " +
            std::to_string(b.draft_n);
    }
    for (int32_t i = 0; i < 11; ++i) {
        if (a.h[i] != b.h[i]) {
            d += (d.empty() ? "" : ", ") + std::string(hash_names[i]);
        }
    }
    return d;
}

bool read_ids_file(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path);
    out.clear();
    if (!f) {
        return false;
    }
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

} // namespace

int fnfl_rollback_mu1(const fnfl_target_args & a) {
    const int32_t cells = 65536;
    // the tokens: the file's, repeated to length, or a seeded sequence of vocabulary IDs
    std::vector<int32_t> src;
    const bool from_file = read_ids_file(a.ids, src) && src.size() >= 512;
    if (!from_file) {
        src.clear();
        uint64_t z = 0x4d5531;
        for (int32_t i = 0; i < 4096; ++i) {
            z = mix64(z);
            src.push_back(int32_t(1000 + z % 150000));
        }
    }
    const auto take_n = [&](size_t off, size_t n) {
        std::vector<int32_t> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = src[(off + i) % src.size()];
        }
        return v;
    };
    const std::vector<int32_t> P0 = take_n(0, 1500), P1 = take_n(1500, 9000), P2 = take_n(10500, 300),
                               P3 = take_n(20000, 300);
    printf("rollback-test mu1: tokens %s (%zu); four slots of %d cells; P0 %zu, P1 %zu (two chunks), P2 %zu, P3 %zu\n",
           from_file ? a.ids.c_str() : "a seeded sequence of vocabulary IDs (no usable IDS file)", src.size(), cells,
           P0.size(), P1.size(), P2.size(), P3.size());

    auto m = std::make_unique<md::model>();
    const bool joint_eager = a.mode == "mu1-joint-eager";
    const bool joint_only = a.mode == "mu1-joint" || joint_eager;
    if (a.mode == "mu1-dry") {
        fnfl_target_args aa = a;
        aa.n_slots = 4;
        if (!fnfl_load_dry(aa, cells, *m)) {
            return 1;
        }
        printf("rollback-test mu1: DRY: zero weights (plumbing only; the states stay 0)\n");
    } else {
        fn::loader::load_params lp;
        lp.model_path = a.model_path;
        lp.draft_path = a.draft_path;
        lp.ple_table = a.ple_table;
        lp.n_slots = 4;
        fn::loader::run_load_report lr;
        std::string err;
        if (!fn::loader::load_for_run(lp, cells, *m, lr, err)) {
            fprintf(stderr, "rollback-test mu1: load: %s\n", err.c_str());
            return 1;
        }
        printf("rollback-test mu1: model loaded in %.2f s; free %.3f / %.3f GiB\n", lr.total_s,
               double(lr.free_after[0]) / 1073741824.0, double(lr.free_after[1]) / 1073741824.0);
    }
    fnfl_target_args da = a;
    da.draft_attention = "sparse"; // the serving setting
    da.draft_attention_given = true;
    rt::draft_options dopt;
    if (!fnfl_draft_settings(da, true, "rollback-test mu1", dopt)) {
        return 1;
    }
    rt::engine_config c;
    c.model_path = a.model_path;
    c.n_ctx_slot = cells;
    c.draft = dopt;
    c.eog = {248046};
    c.spec_d = 4; // the serving d (MX0's recipe)
    c.timeout_s = joint_eager ? 1800.0 : 60.0; // the sanitizers' run: a round under racecheck outlasts 60 s
    c.eager = joint_eager;
    rt::engine e;
    std::string err;
    if (!e.init(std::move(m), c, err)) {
        fprintf(stderr, "rollback-test mu1: engine: %s\n", err.c_str());
        return 1;
    }
    printf("rollback-test mu1: %s\n", e.settings().c_str());
    rt::request_options ro;
    ro.sampling.temp = 0.0f;
    ro.max_tokens = 100000;
    for (int32_t s = 0; s < 4; ++s) {
        if (e.set_sampling(s, ro) != rt::run_status::ok) {
            fprintf(stderr, "rollback-test mu1: set_sampling: %s\n", e.error().c_str());
            return 1;
        }
    }
    // 7's prompts and its joint steps (also the sanitizers' run)
    std::vector<int32_t> R[4];
    for (int32_t s = 0; s < 4; ++s) {
        R[s] = take_n(40000 + size_t(s) * 3000, 1200 + size_t(s) * 200);
    }
    int32_t joint_steps = 0;
    const auto joint_call = [&](const std::vector<int32_t> & sl, std::vector<int32_t> * tok) {
        rt::joint_step_out res[4];
        e.step_joint(sl.data(), int32_t(sl.size()), res);
        for (size_t k = 0; k < sl.size(); ++k) {
            if (res[k].st != rt::run_status::ok) {
                fprintf(stderr, "rollback-test mu1: step_joint on slot %d: %s\n", sl[k], res[k].why.c_str());
                return false;
            }
            joint_steps += res[k].joint ? 1 : 0;
            tok[sl[k]].insert(tok[sl[k]].end(), res[k].out, res[k].out + res[k].n_out);
        }
        return true;
    };
    if (joint_only) {
        std::vector<int32_t> tok[4];
        int32_t np = 0;
        bool ok = true;
        for (int32_t s = 0; s < 4 && ok; ++s) {
            ok = e.read_begin(s, R[s].data(), int32_t(R[s].size()), 0, np) == 0;
            rt::read_status st = rt::read_status::yielded;
            while (ok && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
                st = e.read_step(s, fn::n_layer);
            }
            ok = ok && st == rt::read_status::done;
        }
        ok = ok && joint_call({0, 1, 2, 3}, tok) && joint_call({0, 1, 2, 3}, tok);
        put("mu1-joint: the bootstraps and one four-lane joint round", ok && joint_steps == 4,
            std::to_string(joint_steps) + " joint steps");
        printf("rollback-test mu1: %s\n", g_bad == 0 ? "PASS" : "FAIL");
        return g_bad == 0 ? 0 : 1;
    }
    const int32_t A = 0, Bs = 1, W = 2, S = 3;
    std::vector<int32_t> tokens[4];
    const auto steps = [&](int32_t s, int32_t n) {
        int32_t out[rt::max_step_tokens];
        int32_t n_out = 0;
        bool end = false;
        for (int32_t i = 0; i < n; ++i) {
            if (e.step(s, out, n_out, end) != rt::run_status::ok) {
                fprintf(stderr, "rollback-test mu1: step on slot %d: %s\n", s, e.error().c_str());
                return false;
            }
            tokens[s].insert(tokens[s].end(), out, out + n_out);
        }
        return true;
    };
    // a whole read on a slot, its chunks one step each (the layers in one group)
    const auto read_straight = [&](int32_t s, const std::vector<int32_t> & tok, int32_t n_reuse) {
        int32_t np = 0;
        if (e.read_begin(s, tok.data() + n_reuse, int32_t(tok.size()) - n_reuse, n_reuse, np) != 0) {
            fprintf(stderr, "rollback-test mu1: read_begin on slot %d: %s\n", s, e.error().c_str());
            return false;
        }
        rt::read_status st = rt::read_status::yielded;
        while (st == rt::read_status::yielded || st == rt::read_status::chunk_done) {
            st = e.read_step(s, fn::n_layer);
        }
        return st == rt::read_status::done;
    };
    // the slot's whole ledger tokens (the read's input when it is extended from them)
    const auto ledger_of = [&](int32_t s) {
        std::vector<int32_t> h;
        int32_t pend = -1;
        e.history(s, h, pend);
        if (pend >= 0) {
            h.push_back(pend);
        }
        return h;
    };
    bool ok = true;
    // 1. the common history: P0 and 3 steps, straight through on A and B
    ok = read_straight(A, P0, 0) && steps(A, 3) && read_straight(Bs, P0, 0) && steps(Bs, 3);
    const slot_hash a1 = take(e, A), b1 = take(e, Bs);
    const std::string d1 = diff(a1, b1);
    put("1. P0 and 3 steps on two slots, straight through: states", ok && d1.empty(), d1);
    put("1. their tokens", ok && tokens[A] == tokens[Bs], std::to_string(tokens[A].size()) + " tokens");
    {
        rt::target & t = e.test_target(Bs);
        put("1. the slot's GDN selector and committed slot left the initial bank before the extension",
            t.slot().gdn != 0 || t.slot().cur != 0,
            "cur " + std::to_string(t.slot().cur) + ", gdn " + std::to_string(t.slot().gdn));
    }
    // the writer: a short prompt, generating through step 2
    ok = ok && read_straight(W, P2, 0) && steps(W, 1);
    // 2. the extension P1: A straight through, B one layer a group with W's steps and S's out-of-turn read between
    std::vector<int32_t> ext_a = ledger_of(A), ext_b = ledger_of(Bs);
    ext_a.insert(ext_a.end(), P1.begin(), P1.end());
    ext_b.insert(ext_b.end(), P1.begin(), P1.end());
    // the kept prefix is the processed tokens; the pending token is read again with the extension
    const auto processed = [&](int32_t s) {
        std::vector<int32_t> h;
        int32_t pend = -1;
        e.history(s, h, pend);
        return int32_t(h.size());
    };
    const int32_t keep_a = processed(A), keep_b = processed(Bs);
    ok = ok && read_straight(A, ext_a, keep_a);
    int32_t np = 0;
    ok = ok && e.set_read_group(1) &&
         e.read_begin(Bs, ext_b.data() + keep_b, int32_t(ext_b.size()) - keep_b, keep_b, np) == 0;
    int32_t groups = 0, w_steps = 0, chunk = 0;
    bool s_done = false, nested_ok = true;
    rt::read_status st = rt::read_status::yielded;
    while (ok && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
        st = e.read_step(Bs, 1);
        ++groups;
        if (st == rt::read_status::chunk_done) {
            ++chunk;
        }
        if (st == rt::read_status::yielded || st == rt::read_status::chunk_done) {
            ok = ok && steps(W, 1);
            ++w_steps;
        }
        rt::read_info ri;
        e.get_read_info(Bs, ri);
        if (!s_done && chunk == 0 && ri.in_chunk && ri.group == 20) {
            // S's whole read in the other-reader lane while B's chunk is suspended at layer 20
            nested_ok = e.read_begin(S, P3.data(), int32_t(P3.size()), 0, np) == 0;
            rt::read_status ss = rt::read_status::yielded;
            int32_t lane_seen = -1;
            while (nested_ok && ss == rt::read_status::yielded) {
                ss = e.read_step(S, 4);
                rt::read_info si;
                e.get_read_info(S, si);
                lane_seen = si.in_chunk ? si.lane : lane_seen;
                nested_ok = nested_ok && steps(W, 1);
            }
            nested_ok = nested_ok && ss == rt::read_status::done && e.lane_owner(0) == Bs;
            s_done = true;
            printf("rollback-test mu1: S's out-of-turn read while B's chunk was suspended at layer 20: %s (lane %d)\n",
                   nested_ok ? "done" : "FAILED", lane_seen);
        }
    }
    ok = ok && st == rt::read_status::done && e.set_read_group(4);
    const slot_hash a2 = take(e, A), b2 = take(e, Bs), s2 = take(e, S);
    const std::string d2 = diff(a2, b2);
    put("2. P1 paused at every layer boundary (one layer a group, " + std::to_string(groups) +
            " steps, a writer's step " +
            "between each, a short read out of turn at layer 20) against straight through: states",
        ok && nested_ok && s_done && d2.empty(), d2.empty() ? std::to_string(w_steps) + " writer steps" : d2);
    // 3. 8 greedy steps; then a truncate inside the last step and a 300-token extension, 4 steps
    tokens[A].clear();
    tokens[Bs].clear();
    ok = ok && steps(A, 8) && steps(Bs, 8);
    put("3. the 8 steps after the paused read: tokens", ok && tokens[A] == tokens[Bs],
        std::to_string(tokens[A].size()) + " tokens");
    std::vector<int32_t> la = ledger_of(A), lb = ledger_of(Bs);
    const int32_t cut = int32_t(la.size()) - 2; // inside the last step
    ok = ok && e.truncate(A, cut) == rt::run_status::ok && e.truncate(Bs, cut) == rt::run_status::ok;
    la.resize(size_t(cut));
    la.insert(la.end(), P2.begin(), P2.end());
    ok = ok && read_straight(A, la, cut) && e.set_read_group(2) && read_straight(Bs, la, cut) && e.set_read_group(4);
    tokens[A].clear();
    tokens[Bs].clear();
    ok = ok && steps(A, 4) && steps(Bs, 4);
    const slot_hash a3 = take(e, A), b3 = take(e, Bs);
    const std::string d3 = diff(a3, b3);
    put("3. after a truncate inside the last step, a 300-token extension and 4 steps: states", ok && d3.empty(), d3);
    put("3. those steps' tokens", ok && tokens[A] == tokens[Bs], std::to_string(tokens[A].size()) + " tokens");
    // 4. cancellation on B, each resumed read against A's same prompt read straight through (A from token 0 of Q)
    const std::vector<int32_t> Q = take_n(30000, 9000);
    ok = ok && e.truncate(A, 0) == rt::run_status::ok && read_straight(A, Q, 0);
    const slot_hash aq = take(e, A);
    // 4a: before any chunk
    ok = ok && e.truncate(Bs, 0) == rt::run_status::ok && e.read_begin(Bs, Q.data(), int32_t(Q.size()), 0, np) == 0;
    e.read_cancel(Bs);
    const bool c0 = ok && e.read_step(Bs, 1) == rt::read_status::cancelled && e.held(Bs) == 0;
    // 4b: during a chunk: it finishes (8,192 kept), then the read resumes from its checkpoint
    ok = ok && e.read_begin(Bs, Q.data(), int32_t(Q.size()), 0, np) == 0 &&
         e.read_step(Bs, 5) == rt::read_status::yielded;
    e.read_cancel(Bs);
    st = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded) {
        st = e.read_step(Bs, 3);
    }
    const bool c1 = ok && st == rt::read_status::cancelled && e.held(Bs) == 8192;
    ok = ok && read_straight(Bs, Q, 8192);
    const slot_hash bq1 = take(e, Bs);
    const std::string dq1 = diff(aq, bq1);
    // 4c: while suspended: B's chunk suspended at layer 12 (3 groups of 4), S's out-of-turn read in progress, B cancelled; S finishes,
    // B's chunk finishes and cancels, then B resumes
    ok = ok && e.truncate(Bs, 0) == rt::run_status::ok && e.truncate(S, 0) == rt::run_status::ok &&
         e.read_begin(Bs, Q.data(), int32_t(Q.size()), 0, np) == 0 && e.read_step(Bs, 3) == rt::read_status::yielded &&
         e.read_begin(S, P3.data(), int32_t(P3.size()), 0, np) == 0 && e.read_step(S, 1) == rt::read_status::yielded;
    e.read_cancel(Bs);
    st = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded) {
        st = e.read_step(S, 4);
    }
    const bool s_ok = ok && st == rt::read_status::done;
    st = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded) {
        st = e.read_step(Bs, 6);
    }
    const bool c2 = s_ok && st == rt::read_status::cancelled && e.held(Bs) == 8192;
    ok = ok && read_straight(Bs, Q, 8192);
    const slot_hash bq2 = take(e, Bs);
    const std::string dq2 = diff(aq, bq2);
    put("4. cancel before any chunk: nothing new kept", c0);
    put("4. cancel during a chunk: it finishes and is kept (BG1); the resumed read's states against straight through",
        c1 && dq1.empty(), dq1);
    put("4. cancel while suspended (a short read out of turn in progress): the chunk finishes after it, kept; the "
        "resumed read's states against straight through",
        c2 && dq2.empty(), dq2);
    // 5. S's lone read of P3 against its out-of-turn read in step 2
    const slot_hash s4 = take(e, S); // S's out-of-turn read of 4c, the same P3 from token 0
    ok = ok && e.truncate(S, 0) == rt::run_status::ok && read_straight(S, P3, 0);
    const slot_hash s5 = take(e, S);
    const std::string d5 = diff(s5, s2), d5b = diff(s5, s4);
    put("5. a short read out of turn (twice, beside suspended chunks) against its lone read: states",
        ok && d5.empty() && d5b.empty(), d5 + (d5b.empty() ? "" : "; " + d5b));
    // 6. FMU1: the lone bulk geometry, a peer arriving at a chunk boundary, a writer leaving mid-chunk
    const auto read_lone = [&](int32_t s, const std::vector<int32_t> & tok) {
        rt::read_status rs = rt::read_status::yielded;
        while (rs == rt::read_status::yielded || rs == rt::read_status::chunk_done) {
            if (!e.read_prefetch(s, s)) {
                return false;
            }
            rs = e.read_step(s, 1, rt::read_mode::lone);
        }
        return rs == rt::read_status::done && ledger_of(s).size() == tok.size();
    };
    ok = e.truncate(A, 0) == rt::run_status::ok && e.read_begin(A, Q.data(), int32_t(Q.size()), 0, np) == 0 &&
         read_lone(A, Q);
    rt::read_info la6;
    e.get_read_info(A, la6);
    const slot_hash a6 = take(e, A);
    const std::string d6a = diff(aq, a6);
    put("6. FMU1 a lone read (one 48-layer group a chunk) against straight through: states",
        ok && la6.layers == fn::n_layer && la6.steps == 2 && d6a.empty(),
        d6a.empty() ? "steps " + std::to_string(la6.steps) + ", layers " + std::to_string(la6.layers) : d6a);
    // 6b: the first chunk lone; a writer present from the second chunk's start
    ok = e.truncate(Bs, 0) == rt::run_status::ok && e.read_begin(Bs, Q.data(), int32_t(Q.size()), 0, np) == 0 &&
         e.read_prefetch(Bs, Bs) && e.read_step(Bs, 1, rt::read_mode::lone) == rt::read_status::chunk_done;
    int32_t wsteps = 0;
    st = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded) {
        ok = e.read_prefetch(Bs, Bs);
        st = e.read_step(Bs, 1, rt::read_mode::writers);
        ok = ok && steps(W, 1);
        ++wsteps;
    }
    ok = ok && st == rt::read_status::done;
    const slot_hash b6 = take(e, Bs);
    const std::string d6b = diff(aq, b6);
    put("6. FMU1 a writer arriving at a chunk boundary (the second chunk in 1-layer groups, its end and finish their "
        "own steps, a writer's step between each): states against straight through",
        ok && d6b.empty() && wsteps == 50, d6b.empty() ? std::to_string(wsteps) + " steps" : d6b);
    // 6c: a writer present for 10 groups of the first chunk, then lone (the writer left)
    ok = e.truncate(Bs, 0) == rt::run_status::ok && e.read_begin(Bs, Q.data(), int32_t(Q.size()), 0, np) == 0;
    for (int32_t g = 0; g < 10 && ok; ++g) {
        ok = e.read_prefetch(Bs, Bs) && e.read_step(Bs, 1, rt::read_mode::writers) == rt::read_status::yielded &&
             steps(W, 1);
    }
    ok = ok && read_lone(Bs, Q);
    const slot_hash b6c = take(e, Bs);
    const std::string d6c = diff(aq, b6c);
    put("6. FMU1 a writer leaving mid-chunk (10 1-layer groups, then lone): states against straight through",
        ok && d6c.empty(), d6c);
    tokens[A].clear();
    tokens[Bs].clear();
    ok = ok && steps(A, 4) && steps(Bs, 4);
    put("6. FMU1 the steps after the lone and mixed reads: tokens", ok && tokens[A] == tokens[Bs],
        std::to_string(tokens[A].size()) + " tokens");
    // 7. FMU2: joint rounds against lone steps
    for (int32_t s = 0; s < 4; ++s) {
        e.release_slot(s);
        tokens[s].clear();
    }
    const auto read_all4 = [&]() {
        bool r = true;
        for (int32_t s = 0; s < 4 && r; ++s) {
            r = e.set_sampling(s, ro) == rt::run_status::ok && e.truncate(s, 0) == rt::run_status::ok &&
                read_straight(s, R[s], 0);
        }
        return r;
    };
    ok = read_all4();
    for (int32_t s = 0; s < 4 && ok; ++s) {
        ok = steps(s, 12);
    }
    slot_hash ref[4];
    std::vector<int32_t> ref_tok[4];
    for (int32_t s = 0; s < 4; ++s) {
        ref[s] = take(e, s);
        ref_tok[s] = tokens[s];
    }
    const rt::joint_frame * jf = e.test_joint();
    put("7. FMU2 the engine has a joint frame; the lone references", ok && jf != nullptr,
        std::to_string(ref_tok[0].size()) + ", " + std::to_string(ref_tok[1].size()) + ", " +
            std::to_string(ref_tok[2].size()) + ", " + std::to_string(ref_tok[3].size()) + " tokens");
    const auto compare = [&](const std::string & what, std::vector<int32_t> * tok) {
        for (int32_t s = 0; s < 4; ++s) {
            const slot_hash h = take(e, s);
            const std::string d = diff(ref[s], h);
            put(what + ", slot " + std::to_string(s) + ": tokens and states against its lone steps",
                ok && d.empty() && tok[s] == ref_tok[s],
                d.empty() ? std::to_string(tok[s].size()) + " tokens" + (tok[s] == ref_tok[s] ? "" : " (differ)") : d);
        }
    };
    // 7a: four lanes, one call a step
    std::vector<int32_t> jt[4];
    ok = ok && read_all4();
    joint_steps = 0;
    for (int32_t i = 0; i < 12 && ok; ++i) {
        ok = joint_call({0, 1, 2, 3}, jt);
    }
    const int32_t ja = joint_steps;
    compare("7. FMU2 four lanes (" + std::to_string(ja) + " joint lane steps)", jt);
    // 7b: lane orders rotated every step; every second step two two-member calls
    for (int32_t s = 0; s < 4; ++s) {
        jt[s].clear();
    }
    ok = ok && read_all4();
    joint_steps = 0;
    const int32_t rot[4] = {3, 1, 0, 2};
    for (int32_t i = 0; i < 12 && ok; ++i) {
        if (i % 2 == 0) {
            std::vector<int32_t> sl;
            for (int32_t k = 0; k < 4; ++k) {
                sl.push_back(rot[(k + i / 2) % 4]);
            }
            ok = joint_call(sl, jt);
        } else {
            ok = joint_call({1, 3}, jt) && joint_call({2, 0}, jt);
        }
    }
    const int32_t jb = joint_steps;
    compare("7. FMU2 rotated lanes and two-member rounds (" + std::to_string(jb) + " joint lane steps)", jt);
    put("7. FMU2 the joint rounds ran", ja >= 4 * 10 && jb >= 4 * 10,
        std::to_string(jf != nullptr ? jf->captures() : 0) + " captures, " +
            std::to_string(jf != nullptr ? jf->rounds() : 0) + " rounds");
    if (jf != nullptr) {
        printf("rollback-test mu1: 7. the joint frame: %.1f / %.1f MiB; graphs kept %.1f / %.1f MiB; free after the "
               "last capture %.3f / %.3f GiB\n",
               double(jf->bytes(0)) / 1048576.0, double(jf->bytes(1)) / 1048576.0,
               double(jf->graph_bytes(0)) / 1048576.0, double(jf->graph_bytes(1)) / 1048576.0,
               double(jf->free_after_capture(0)) / 1073741824.0, double(jf->free_after_capture(1)) / 1073741824.0);
        printf("rollback-test mu1: 7. the last joint capture (%d lanes): %lld / %lld graph nodes; slot 0's lone "
               "five-row graph %lld / %lld\n",
               jf->last_lanes(), static_cast<long long>(jf->last_nodes(0)), static_cast<long long>(jf->last_nodes(1)),
               static_cast<long long>(e.test_target(0).graph_node_count(5, 0)),
               static_cast<long long>(e.test_target(0).graph_node_count(5, 1)));
    }
    printf("rollback-test mu1: %s\n", g_bad == 0 ? "PASS" : "FAIL");
    return g_bad == 0 ? 0 : 1;
}
