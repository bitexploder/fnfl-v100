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
//   7c. TTTS3 (one box): four-lane joint rounds at d = 4 with the experts forced to their per-lane fallback (the path a
//      format without a joint form takes), against the lone references.
//   8. MUP1 piece 5, the slot's rotary position offset (rotary positions = cell + offset), d = 4: (a) the offset set to
//      0 explicitly on both boxes against the never-set path: four slots' reads and 12 lone steps, tokens and states
//      bitwise; (b) the API's refusals (out of range, during a read); (c) offsets 30,000, 0, 12,345 and -777 on the four
//      slots: the lone references (reads and 24 steps); the roped classes (K cache, summaries, the draft's K) differ
//      from offset 0's on every nonzero slot, and the tokens agree with offset 0's (a rotary site left on the cell
//      would put the query and key X apart: garbage from the first token and the draft's acceptance gone; a bug
//      check, not a numerical bound: each nonzero slot's first 4 tokens equal and the tokens a step at least 3/4 of
//      offset 0's); (d) joint rounds with those mixed offsets (four lanes, 24 steps) against the lone references,
//      tokens and states bitwise; (e) a rollback at each offset (truncated 3 tokens past its prompt, the 4th read back,
//      12 steps) regenerating its own continuation as far as the same slot's rollback at offset 0 does (up to 8
//      tokens); (f) -777 on slots 0 and 2 and +777 on slot 3 (whether (c)'s early difference at -777 follows the sign
//      or the text): (c)'s rule against offset 0's tokens, and (e)'s rollback. Then every offset back to 0.
//   9. MUP2 (the split only: eight slots, a fifth for the reader): reads beside four writers in units and pieces,
//      d = 2: (a) a 64,000-token read at level 1, in pieces of 2,048 and 1,024 rows, and under a 50 ms budget (the
//      pieces chosen), each against its lone read (states), the writers against their lone steps (tokens and states:
//      I3, KL 0), and a rollback after each read against the lone read's; (b) a 3,000-token append at cell 1,537 (off
//      the 64-cell grid) at level 1 and in 1,024-row pieces against its lone append; (c) a cancel mid-piece: the
//      chunk kept (8,192), the resumed read against the lone read.
//  10. APF2, the kernel-size ladder: fresh reads of 2,048 and 4,000 tokens and 64-, 256-, 300- and 1,024-token
//      appends at cell 3,000 on slot 0, each under the retained routes, the table's, the new routes and the table's
//      again, from the same state: the rerun bitwise (states and 4 greedy steps); a file without a table bitwise the
//      retained routes; the executed variants on both boxes as each arm says. The arms' KL is joint-kl --route-kl's.
//
// mu1-ladder (APF2, one model load): section 10 alone.
// mu1-ladder (APF2, one model load): section 10 alone.
// mu1-joint (one model load): the four slots read R0..R3 and take two step_joint calls, the bootstraps and one
// four-lane joint round (its capture and launch), then exit. mu1-joint-eager (the sanitizers' run): the same on the
// eager engine, the joint round enqueued in phases (Compute Sanitizer runs one kernel at a time: the graphs'
// cross-card polling cannot progress under it).
//
// IDS: a token file (rollback-test's form); when it is missing or short, a seeded sequence of real vocabulary IDs (the
// states need nonzero values, not a meaningful text). PASS (exit 0) needs every comparison equal.

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/hc/hc_prompt.h"
#include "fn/kernels/head/head.h"
#include "fn/loader/run_load.h"
#include "fn/model/model.h"
#include "fn/model/prompt.h"
#include "fn/runtime/joint.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/stage_cmds.h"
#include "fn/runtime/target_cmds.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
// retained carry and its K and V of cells < its length; MUP2: the hidden carry's row 0 apart (a read's prompt head
// writes that row only, B = 1; rows 1 .. 4 are the slot's earlier rounds', untouched: hc.h)
constexpr int32_t n_hash = 12;
struct slot_hash {
    int64_t n = 0, draft_n = 0;
    uint64_t h[n_hash] = {};
    bool ok = false;
};

const char * const hash_names[n_hash] = { "GDN states",   "GDN histories", "PLE history",   "K cache",
                                          "V cache",      "index keys",    "summaries",     "retained logits",
                                          "hidden carry", "draft carry",   "draft K and V", "hidden carry row 0" };
constexpr int32_t  hash_carry         = 8;

slot_hash take_one(rt::engine & e, int32_t slot) {
    slot_hash s;
    rt::target & t = e.test_target(slot);
    md::model & m = e.test_model();
    if (!fnfl_state_hashes(t, m, s.h)) {
        return s;
    }
    s.n = t.slot().n;
    s.draft_n = t.draft_on() ? t.draft_n() : 0;
    for (int32_t c = 0; c < 2; ++c) {
        const int32_t dev = m.device[c];
        if (m.win.head) {  // TTTS1: the head's buffers and the draft's live on the back stage
            s.h[7] = hash_dev(s.h[7] ^ uint64_t(c), dev, t.card(c).logits, size_t(fn::head::n_vocab_card) * 4);
            s.h[8] = hash_dev(s.h[8] ^ uint64_t(c), dev, t.card(c).carry, size_t(5) * 4 * 1280 * 4);
            s.h[11] = hash_dev(s.h[11] ^ uint64_t(c), dev, t.card(c).carry, size_t(4) * 1280 * 4);
        }
        if (t.draft_on()) {
            s.h[9]          = hash_dev(s.h[9] ^ uint64_t(c), dev, t.draft(c).carry_prev, size_t(4) * 1280 * 4);
            const size_t kv = size_t(std::max<int64_t>(0, s.draft_n)) * 256 * 2;
            s.h[10] =
                hash_dev(hash_dev(s.h[10] ^ uint64_t(c), dev, t.draft(c).k_cache, kv), dev, t.draft(c).v_cache, kv);
        }
    }
    s.ok = true;
    return s;
}

// TTTS1: the split's leader, when the test runs on the split (every slot's state is the front's and the back's)
fn::runtime::stage::leader * g_lead = nullptr;

slot_hash take(rt::engine & e, int32_t slot) {
    slot_hash s = take_one(e, slot);
    if (g_lead == nullptr || !s.ok) {
        return s;
    }
    // the back's part (fnfl_mu1_slot_hash on the follower), each class's two hashes mixed
    std::vector<uint8_t> b;
    slot_hash            bs;
    if (!g_lead->probe(slot, 0, b) || b.size() != sizeof(slot_hash)) {
        s.ok = false;
        return s;
    }
    std::memcpy(&bs, b.data(), sizeof bs);
    if (!bs.ok || bs.n != s.n) {
        s.ok = false;
        return s;
    }
    s.draft_n = bs.draft_n;
    for (int32_t i = 0; i < n_hash; ++i) {
        s.h[i] = mix64(s.h[i] ^ mix64(bs.h[i] + uint64_t(i)));
    }
    return s;
}

// the engine's calls the test makes, on one box or through the split's leader (the front's own state for the rest)
struct box {
    rt::engine *                 e = nullptr;
    fn::runtime::stage::leader * L = nullptr;

    rt::run_status set_sampling(int32_t s, const rt::request_options & ro) {
        return L != nullptr ? L->set_sampling(s, ro) : e->set_sampling(s, ro);
    }

    rt::run_status set_draft(int32_t d, bool sparse) {
        return L != nullptr ? L->set_draft(d, sparse) : e->set_draft(d, sparse);
    }

    rt::run_status truncate(int32_t s, int32_t n) { return L != nullptr ? L->truncate(s, n) : e->truncate(s, n); }

    rt::run_status release_slot(int32_t s) { return L != nullptr ? L->release_slot(s) : e->release_slot(s); }

    // MUP1 piece 5: on the split, both boxes
    rt::run_status set_pos_offset(int32_t s, int32_t off) {
        return L != nullptr ? L->set_pos_offset(s, off) : e->set_pos_offset(s, off);
    }

    int32_t pos_offset(int32_t s) const { return e->pos_offset(s); }

    // APF2: the forced prompt routes, on the split both boxes; back: the follower process's totals (the split)
    rt::run_status force_route(const md::prompt_route_force & f, md::prompt_path::route_counts * back = nullptr) {
        if (L != nullptr) {
            return L->test_force_route(f, back);
        }
        e->prefill_driver().test_force_route(f);
        if (back != nullptr) {
            *back = md::prompt_path::route_counts{};
        }
        return rt::run_status::ok;
    }

    int32_t read_begin(int32_t s, const int32_t * tok, int32_t n, int32_t reuse, int32_t & np) {
        return L != nullptr ? L->read_begin(s, tok, n, reuse, np) : e->read_begin(s, tok, n, reuse, np);
    }

    rt::read_status read_step(int32_t s, int32_t groups, rt::read_mode mode = rt::read_mode::shared) {
        return L != nullptr ? L->read_step(s, groups, mode) : e->read_step(s, groups, mode);
    }

    // MUP2: a read step beside writers under a budget (the front's units; APF1: the back budgets its own by its
    // writer-pause deadline)
    rt::read_status read_step_paced(int32_t s, const rt::read_budget & b) {
        return L != nullptr ? L->read_step(s, 1, rt::read_mode::writers, &b) :
                              e->read_step(s, 1, rt::read_mode::writers, &b);
    }

    void read_cancel(int32_t s) {
        if (L != nullptr) {
            L->read_cancel(s);
        } else {
            e->read_cancel(s);
        }
    }

    rt::run_status step(int32_t s, int32_t * out, int32_t & n_out, bool & end) {
        if (L == nullptr) {
            return e->step(s, out, n_out, end);
        }
        rt::joint_step_out r;
        L->step_group(&s, 1, &r);
        std::copy(r.out, r.out + r.n_out, out);
        n_out = r.n_out;
        end   = r.end;
        return r.st;
    }

    void step_joint(const int32_t * sl, int32_t n, rt::joint_step_out * res) {
        if (L != nullptr) {
            L->step_group(sl, n, res);
        } else {
            e->step_joint(sl, n, res);
        }
    }

    std::string error() const { return L != nullptr ? L->error() : e->error(); }

    // PRD1 item 5: the slot's processed tokens kept and toks appended (both boxes on the split), then forced rounds of
    // rows rows over them; each row's logits over the whole vocabulary (card 0's shard, then card 1's) appended to out
    bool forced_logits(int32_t                      s,
                       const std::vector<int32_t> & toks,
                       int32_t                      rows,
                       std::vector<float> &         out,
                       std::string &                err) {
        constexpr int32_t VC       = fn::head::n_vocab_card;
        const auto        put_rows = [&](const float * card0, const float * card1) {
            for (int32_t b = 0; b < rows; ++b) {
                out.insert(out.end(), card0 + size_t(b) * VC, card0 + size_t(b + 1) * VC);
                out.insert(out.end(), card1 + size_t(b) * VC, card1 + size_t(b + 1) * VC);
            }
        };
        if (L != nullptr) {
            if (L->test_prompt(s, toks, true) != rt::run_status::ok) {
                err = L->error();
                return false;
            }
            for (size_t i = 0; i + size_t(rows) <= toks.size(); i += size_t(rows)) {
                rt::joint_step_out r;
                const int32_t      one = 1;
                L->forced_groups(&s, &one, 1, rows, &r);
                if (r.st != rt::run_status::ok) {
                    err = r.why;
                    return false;
                }
                const std::vector<float> & lg = L->forced_logits(s);
                put_rows(lg.data(), lg.data() + size_t(rows) * VC);
            }
            return true;
        }
        rt::target & t = e->test_target(s);
        if (!t.keep_processed(err)) {
            return false;
        }
        t.append(toks);
        md::model &        m = e->test_model();
        std::vector<float> host(size_t(rows) * VC * 2);
        for (size_t i = 0; i + size_t(rows) <= toks.size(); i += size_t(rows)) {
            rt::round_result r;
            if (!t.forced(rows, r, err)) {
                return false;
            }
            for (int32_t c = 0; c < 2; ++c) {
                if (cudaSetDevice(m.device[c]) != cudaSuccess ||
                    cudaMemcpy(host.data() + size_t(c) * rows * VC, t.card(c).logits, size_t(rows) * VC * 4,
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
                    err = "the forced round's logits";
                    return false;
                }
            }
            put_rows(host.data(), host.data() + size_t(rows) * VC);
        }
        return true;
    }

    // the front's own (one box: the engine's)
    rt::run_status history(int32_t s, std::vector<int32_t> & h, int32_t & pend) const { return e->history(s, h, pend); }

    int64_t held(int32_t s) const { return e->held(s); }

    bool get_read_info(int32_t s, rt::read_info & ri) const { return e->get_read_info(s, ri); }

    int32_t lane_owner(int32_t lane) const { return e->lane_owner(lane); }

    bool set_read_group(int32_t g) { return e->set_read_group(g); }

    bool read_prefetch(int32_t a, int32_t b) { return e->read_prefetch(a, b); }

    rt::target & test_target(int32_t s) { return e->test_target(s); }

    const rt::joint_frame * test_joint() const { return e->test_joint(); }

    rt::joint_frame * test_joint_frame() { return e->test_joint_frame(); }  // TTTS3: section 7c

    std::unique_ptr<md::model> test_take_model() { return e->test_take_model(); }

    std::string settings() const { return e->settings(); }
};

slot_hash take(box & b, int32_t slot) {
    return take(*b.e, slot);
}

// PRD1 item 5: KL(p || q) over the whole vocabulary (F32 logits, FP64 sums; joint-kl's), the top token (the lower ID
// on a tie), and a percentile (linear between the ranks)
double kl_full(const float * p, const float * q) {
    constexpr int32_t V  = fn::head::n_vocab;
    double            mp = -HUGE_VAL, mq = -HUGE_VAL;
    for (int32_t i = 0; i < V; ++i) {
        mp = std::max(mp, double(p[i]));
        mq = std::max(mq, double(q[i]));
    }
    double sp = 0.0, sq = 0.0;
    for (int32_t i = 0; i < V; ++i) {
        sp += std::exp(double(p[i]) - mp);
        sq += std::exp(double(q[i]) - mq);
    }
    const double lsp = std::log(sp), lsq = std::log(sq);
    double       k = 0.0;
    for (int32_t i = 0; i < V; ++i) {
        const double lp = double(p[i]) - mp - lsp;
        const double lq = double(q[i]) - mq - lsq;
        k += std::exp(lp) * (lp - lq);
    }
    return k;
}

int32_t top1_full(const float * p) {
    int32_t best = 0;
    for (int32_t i = 1; i < fn::head::n_vocab; ++i) {
        best = p[i] > p[best] ? i : best;
    }
    return best;
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const double pos = q / 100.0 * double(v.size() - 1);
    const size_t lo  = static_cast<size_t>(std::floor(pos));
    const size_t hi  = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - double(lo));
}

// read: a read's state against another read's of the same tokens (MUP2 section 9): the hidden carry's row 0 only
std::string diff(const slot_hash & a, const slot_hash & b, bool read = false) {
    if (!a.ok || !b.ok) {
        return "not read";
    }
    std::string d;
    if (a.n != b.n || a.draft_n != b.draft_n) {
        d = "N " + std::to_string(a.n) + " / " + std::to_string(b.n) + ", draft " + std::to_string(a.draft_n) + " / " +
            std::to_string(b.draft_n);
    }
    for (int32_t i = 0; i < n_hash; ++i) {
        if (a.h[i] != b.h[i] && !(read && i == hash_carry)) {
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

bool fnfl_mu1_slot_hash(fn::runtime::engine & e, int32_t slot, std::vector<uint8_t> & out) {
    const slot_hash s = take_one(e, slot);
    out.resize(sizeof s);
    std::memcpy(out.data(), &s, sizeof s);
    return s.ok;
}

int fnfl_rollback_mu1(const fnfl_target_args & a_in) {
    const int32_t cells = 65536;
    // MUP2: on the split, section 9's read (four pages) beside four writers (a page each) needs more than one slot's
    // pages in the shared q8 pool (0: a slot's cells, 4 pages): four whole slots (16 pages), as fnfl_lay_out lays the
    // follower's out on reconfigure; the follower's --pool-cells alike (X-HELLO)
    fnfl_target_args a_pool = a_in;
    if (a_in.split_role == 1 && a_pool.pool_cells == 0) {
        a_pool.pool_cells = 4 * cells;
    }
    const fnfl_target_args & a = a_pool;
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
    const bool ladder_only = a.mode == "mu1-ladder";  // APF2: section 10 alone (Load 3, IQ3_XXS resident TP)
    if (a.mode == "mu1-dry") {
        fnfl_target_args aa = a;
        aa.n_slots          = a.split_role == 1 ? 8 : 4;  // MUP2: section 9's fifth slot on the split
        if (!fnfl_load_dry(aa, cells, *m)) {
            return 1;
        }
        printf("rollback-test mu1: DRY: zero weights (plumbing only; the states stay 0)\n");
    } else {
        fn::loader::load_params lp;
        lp.model_path = a.model_path;
        lp.draft_path = a.draft_path;
        lp.ple_table = a.ple_table;
        lp.n_slots    = a.split_role == 1 ? 8 : 4;  // MUP2: section 9's fifth slot on the split
        lp.kv_q8      = a.kv_q8;
        lp.pool_cells = a.pool_cells;
        if (a.split_role == 1) {
            // TTTS1: the split's front (layers 0 .. k - 1); the follower holds the back
            lp.win.lo   = 0;
            lp.win.hi   = a.split_k - 1;
            lp.win.head = false;
        }
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
    c.joint_x16 = false;       // FMS2 item 7: section 7 is I3, every lane bitwise its lone step
    c.joint_router_tc = false; // FMS5: likewise (the tensor-core router is judged by joint-kl)
    c.joint_dense_f16 = false; // FMS6: likewise (the FP16 dense products are judged by dpj-test f16)
    // TTTS1 (--split): this box the front; the follower (fnfl-drive stage) the back, rebuilt with these switches
    fnfl_split  sp;
    std::string err;
    const bool  split = a.split_role == 1;
    if (split && !fnfl_split_prepare(a, 8, sp, c, err)) {
        fprintf(stderr, "rollback-test mu1: the split: %s\n", err.c_str());
        return 1;
    }
    rt::engine eng;
    if (!eng.init(std::move(m), c, err)) {
        fprintf(stderr, "rollback-test mu1: engine: %s\n", err.c_str());
        return 1;
    }
    fn::runtime::stage::leader * L = nullptr;
    if (split) {
        if (!fnfl_split_attach(a, eng, cells, sp, err)) {
            fprintf(stderr, "rollback-test mu1: the split: %s\n", err.c_str());
            return 1;
        }
        L = sp.lead.get();
        L->set_block_reads(true);
        g_lead = L;
        fn::runtime::stage::ctrl rc;
        rc.a    = a.kv_q8 ? 1 : 0;
        rc.b    = c.spec_d;
        rc.c    = (c.joint_x16 ? fn::runtime::stage::sw_joint_x16 : 0) |
                  (c.joint_router_tc ? fn::runtime::stage::sw_router_tc : 0) |
                  (c.joint_dense_f16 ? fn::runtime::stage::sw_dense_f16 : 0) |
                  (c.eager ? fn::runtime::stage::sw_eager : 0);
        rc.base = cells;
        if (L->reconfigure(rc) != rt::run_status::ok) {
            fprintf(stderr, "rollback-test mu1: the follower's switches: %s\n", L->error().c_str());
            return 1;
        }
        printf(
            "rollback-test mu1: TTTS1 split at k %d: this box the front, the follower the back with the same "
            "switches (16-bit X-A/X-F %s, the router on the tensor cores %s, dense FP16 %s)\n",
            a.split_k, c.joint_x16 ? "on" : "off", c.joint_router_tc ? "on" : "off", c.joint_dense_f16 ? "on" : "off");
    }
    box e{ &eng, L };
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
        // FMS13: at each d = 1 .. 4 in turn (set_draft finishes every slot, so each d reads its prompts again)
        for (int32_t d = 1; d <= 4; ++d) {
            std::vector<int32_t> tok[4];
            int32_t              np = 0;
            bool                 ok = e.set_draft(d, true) == rt::run_status::ok;
            for (int32_t s = 0; s < 4 && ok; ++s) {
                ok                 = e.truncate(s, 0) == rt::run_status::ok &&
                                     e.read_begin(s, R[s].data(), int32_t(R[s].size()), 0, np) == 0;
                rt::read_status st = rt::read_status::yielded;
                while (ok && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
                    st = e.read_step(s, fn::n_layer);
                }
                ok = ok && st == rt::read_status::done;
            }
            joint_steps = 0;
            ok          = ok && joint_call({ 0, 1, 2, 3 }, tok) && joint_call({ 0, 1, 2, 3 }, tok);
            put("mu1-joint: d = " + std::to_string(d) + ": the bootstraps and one four-lane joint round",
                ok && joint_steps == 4, std::to_string(joint_steps) + " joint steps");
        }
        printf("rollback-test mu1: %s\n", g_bad == 0 ? "PASS" : "FAIL");
        return g_bad == 0 ? 0 : 1;
    }
    const int32_t A = 0, Bs = 1, W = 2, S = 3;
    std::vector<int32_t> tokens[8];  // MUP2: section 9 steps slot 4

    // up to n steps: a sequence ends with its request (eog or budget), since main refuses a step after it (PRD1,
    // TTTS8); the slots a check compares stop at the same step when their tokens agree
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
            if (end) {
                break;
            }
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
    // APF2: section 10 (the ladder), run in its place, or alone (mu1-ladder)
    const auto ladder_section = [&]() {
        // 10. APF2: the kernel-size ladder on the real weights. Each read from the same state on slot 0, four arms: every
        // route retained (dense converted, HC converted, per-task experts, eager), the table's routes (serving's), the new
        // routes (each product the cost model's first rank, the grouped experts wherever the format takes them, graphs
        // on), and the table's routes again. Split K sums in another order, so the arms are compared with the retained
        // one by joint-kl's KL line (--route-kl at depth, APF2 section 5), not here; here: the rerun bitwise its first
        // run (states and 4 greedy steps: determinism); a file the table is not for (IQ3_XXS) bitwise the retained arm
        // under the table's routes (every product retained); the executed variants (the front's process totals; on the
        // split the back's from the follower) as each arm says. Each arm's state difference from the retained arm is
        // reported.
        {
            md::model & m        = e.e->test_model();
            const bool  dense_q8 = md::dense_q8_of(m);
            bool        kq       = false;  // a routed gate/up of this box is Q4_K or Q5_K (the grouped route's formats)
            for (int32_t l = m.win.lo; l <= m.win.hi; ++l) {
                const auto f = static_cast<fn::experts::fmt>(e.e->test_target(0).card(0).w[l].ex_gu_fmt);
                kq           = kq || f == fn::experts::fmt::q4_k || f == fn::experts::fmt::q5_k;
            }
            const md::prompt_route_force arm_force[4] = {
                { 2, 2, 2, 2 },
                {},
                { 1, 1, 1, 1 },
                {}
            };
            const char * const arm_name[4] = { "retained", "table", "new routes", "table again" };

            struct lad_case {
                const char * name;
                int32_t      prefix, rows;
            };

            const lad_case cases[] = {
                { "a 2,048-token fresh read",           0,    2048 },
                { "a 64-token append at cell 3,000",    3000, 64   },
                { "a 256-token append at cell 3,000",   3000, 256  },
                { "a 1,024-token append at cell 3,000", 3000, 1024 },
                { "a 300-token append at cell 3,000",   3000, 300  },
                { "a 4,000-token fresh read",           0,    4000 },
            };
            const std::vector<int32_t> pre = take_n(70000, 3000);
            e.release_slot(0);
            ok = e.set_sampling(0, ro) == rt::run_status::ok;
            put("10. APF2: the ladder on slot 0", ok, ok ? std::string() : e.error());
            md::prompt_route_collect(true);  // the test's totals (off in serving)
            for (const lad_case & c : cases) {
                std::vector<int32_t>       tok(pre.begin(), pre.begin() + c.prefix);
                const std::vector<int32_t> ext = take_n(80000, size_t(c.rows));
                tok.insert(tok.end(), ext.begin(), ext.end());
                slot_hash                     h[4];
                std::vector<int32_t>          tk[4];
                md::prompt_path::route_counts rf[4], rb[4];
                for (int32_t arm = 0; arm < 4 && ok; ++arm) {
                    tokens[0].clear();
                    ok = e.truncate(0, 0) == rt::run_status::ok &&
                         (c.prefix == 0 ||
                          read_straight(0, std::vector<int32_t>(pre.begin(), pre.begin() + c.prefix), 0));
                    md::prompt_path::route_counts r0, r1, b0, b1;
                    uint32_t                      g0 = 0, g1 = 0;
                    ok = ok && e.force_route(arm_force[arm], &b0) == rt::run_status::ok;
                    md::prompt_route_totals(r0, g0, true);
                    ok = ok && read_straight(0, tok, c.prefix);
                    md::prompt_route_totals(r1, g1);
                    ok = ok && e.force_route(md::prompt_route_force{}, &b1) == rt::run_status::ok;
                    rf[arm].add_since(r1, r0);
                    rb[arm].add_since(b1, b0);
                    h[arm]  = take(e, 0);
                    ok      = ok && steps(0, 4);
                    tk[arm] = tokens[0];
                }
                const bool split = e.L != nullptr;
                // the retained arm: no ladder product, grouped expert or graph on either box
                const auto none  = [](const md::prompt_path::route_counts & x) {
                    return x.dense_ladder == 0 && x.hc_ladder == 0 && x.ex_grouped == 0 && x.exd_grouped == 0 &&
                           x.graph_units == 0;
                };
                const bool             ret_ok  = none(rf[0]) && (!split || none(rb[0])) && rf[0].dense_retained > 0 &&
                                                 rf[0].hc_retained > 0 && rf[0].qsa > 0 && rf[0].gdn > 0;
                // the table arm: UD-Q4_K_XL ladder products where the table picks them (at least one dense on the front);
                // another file every product retained
                const bool             tab_ok  = dense_q8 ? (md::lad_table().empty() || rf[1].dense_ladder > 0) :
                                                            none(rf[1]) && (!split || none(rb[1]));
                // the new routes: ladder dense and HC products, grouped experts where the format takes them, graphs as
                // prompt_route_for says
                const md::prompt_route e1      = md::prompt_route_for(c.rows, true, arm_force[2]);
                const bool             new_ok  = rf[2].dense_ladder > 0 && (!dense_q8 || rf[2].hc_ladder > 0) &&
                                                 (kq ? rf[2].ex_grouped > 0 : rf[2].ex_grouped == 0) &&
                                                 (e1.graph ? rf[2].graph_units > 0 && rf[2].eager_units == 0 :
                                                             rf[2].graph_units == 0 && rf[2].eager_units > 0);
                const std::string      d_rerun = diff(h[1], h[3], true);
                const std::string      d_ret   = diff(h[0], h[1], true);
                const std::string      d_new   = diff(h[0], h[2], true);
                const bool             rerun   = d_rerun.empty() && tk[1] == tk[3];
                const bool             iq_same = dense_q8 || (d_ret.empty() && tk[0] == tk[1]);
                std::string            counts;
                for (int32_t arm = 0; arm < 4; ++arm) {
                    counts += std::string("; ") + arm_name[arm] + ": front " + rf[arm].text() +
                              (split ? "; back " + rb[arm].text() : std::string());
                }
                put(std::string("10. ") + c.name + ": the table's routes rerun bitwise (states and 4 steps)" +
                        (dense_q8 ? "" : ", and bitwise the retained routes (no table for this file)") +
                        "; the routes each arm says",
                    ok && rerun && iq_same && ret_ok && tab_ok && new_ok,
                    std::string(rerun ? "rerun bitwise" : "RERUN DIFFERS: " + d_rerun) +
                        "; table against retained: " + (d_ret.empty() ? std::string("bitwise") : "differs: " + d_ret) +
                        (tk[0] == tk[1] ? "" : " (tokens differ)") + "; new routes against retained: " +
                        (d_new.empty() ? std::string("bitwise") : "differs: " + d_new) +
                        (tk[0] == tk[2] ? "" : " (tokens differ)") + counts +
                        (ret_ok ? "" : "; UNEXPECTED ROUTE (retained arm)") +
                        (tab_ok ? "" : "; UNEXPECTED ROUTE (table arm)") +
                        (new_ok ? "" : "; UNEXPECTED ROUTE (new routes arm)"));
            }
            md::prompt_path::route_counts now;
            uint32_t                      now_rungs = 0;
            md::prompt_route_totals(now, now_rungs);
            printf(
                "rollback-test mu1: 10. the executed variants of section 10 (every read, every arm; the front): %s\n",
                now.text().c_str());
            e.release_slot(0);
        }
    };
    if (ladder_only) {
        ladder_section();
        printf("rollback-test mu1: %s\n", g_bad == 0 ? "PASS" : "FAIL");
        return g_bad == 0 ? 0 : 1;
    }
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
    // TTTS1: on the split the front's lone group is its window (layers 0 .. k - 1), and a chunk's 1-layer groups are
    // its layers plus the chunk's end and finish
    const int32_t     win_layers = split ? a.split_k : fn::n_layer;
    put("6. FMU1 a lone read (one 48-layer group a chunk; the front's window on the split) against straight through: "
        "states",
        ok && la6.layers == win_layers && la6.steps == 2 && d6a.empty(),
        d6a.empty() ? "steps " + std::to_string(la6.steps) + ", layers " + std::to_string(la6.layers) : d6a);
    // 6b: the first chunk lone; a writer present from the second chunk's start
    ok = e.truncate(Bs, 0) == rt::run_status::ok && e.read_begin(Bs, Q.data(), int32_t(Q.size()), 0, np) == 0 &&
         e.read_prefetch(Bs, Bs) && e.read_step(Bs, 1, rt::read_mode::lone) == rt::read_status::chunk_done;
    // MUP2: writers mode without a budget keeps FMU1's 1-layer groups (units need a budget): the window's layers, the
    // end and the finish, each its own step
    int32_t wsteps = 0, units6 = 0;
    st = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded) {
        ok = e.read_prefetch(Bs, Bs);
        st = e.read_step(Bs, 1, rt::read_mode::writers);
        if (wsteps == 0) {
            rt::read_info i6;
            e.get_read_info(Bs, i6);
            units6 = i6.groups;
        }
        ok = ok && steps(W, 1);
        ++wsteps;
    }
    ok = ok && st == rt::read_status::done;
    const slot_hash b6 = take(e, Bs);
    const std::string d6b = diff(aq, b6);
    put("6. FMU1 a writer arriving at a chunk boundary (the second chunk in 1-layer groups, its end and finish their "
        "own steps, a writer's step between each): states against straight through",
        ok && d6b.empty() && units6 == win_layers + 1 && wsteps == win_layers + 2,
        d6b.empty() ? std::to_string(wsteps) + " steps, " + std::to_string(units6) + " groups" : d6b);
    // 6c (PRD1 item 8): the server's path when the writers leave mid-chunk: the first chunk begun in units under a
    // budget of 12 ms (MUP2's served T 12 ms at F 0.5; APF1: a known 12 ms budget), one step and a writer's step, then
    // read_step(shared, 8) calls,
    // each running min(8, the units left) units of the started chunk (never repartitioned), then the read's rest
    // shared; states against straight through
    ok = e.truncate(Bs, 0) == rt::run_status::ok && e.read_begin(Bs, Q.data(), int32_t(Q.size()), 0, np) == 0;
    rt::read_budget b6c;
    b6c.budget_ms = 12.0;
    b6c.known     = true;
    ok = ok && e.read_prefetch(Bs, Bs) && e.read_step_paced(Bs, b6c) == rt::read_status::yielded && steps(W, 1);
    rt::read_info i6c;
    e.get_read_info(Bs, i6c);
    const int32_t groups6c = i6c.groups;
    const int64_t begun6c  = i6c.units_run;
    int32_t       shared6c = 0;
    std::string   bad6c    = i6c.units_run > 0 ? "" : "the chunk did not begin in units";
    st                     = rt::read_status::yielded;
    while (ok && st == rt::read_status::yielded && i6c.group >= 0 && i6c.group < i6c.groups) {
        const int32_t left = i6c.groups - i6c.group;
        const int64_t u0   = i6c.units_run;
        ok                 = e.read_prefetch(Bs, Bs);
        st                 = e.read_step(Bs, 8, rt::read_mode::shared);
        e.get_read_info(Bs, i6c);
        ++shared6c;
        const int64_t ran = i6c.units_run - u0;
        if (ran != std::min(8, left) && bad6c.empty()) {
            bad6c = "a shared step ran " + std::to_string(ran) + " units with " + std::to_string(left) + " left";
        }
        if (i6c.groups != groups6c && i6c.group >= 0 && bad6c.empty()) {
            bad6c = "the chunk was repartitioned";
        }
    }
    while (ok && (st == rt::read_status::yielded || st == rt::read_status::chunk_done)) {
        ok = e.read_prefetch(Bs, Bs);
        st = e.read_step(Bs, 8, rt::read_mode::shared);
    }
    ok                      = ok && st == rt::read_status::done;
    const slot_hash   b6c_h = take(e, Bs);
    const std::string d6c   = diff(aq, b6c_h);
    put("6. PRD1 writers leaving mid-chunk: a chunk begun in units under a budget, then read_step(shared, 8) runs "
        "min(8, the units left) units a step (never repartitioned): states against straight through",
        ok && d6c.empty() && bad6c.empty() && shared6c > 1,
        !d6c.empty()   ? d6c :
        !bad6c.empty() ? bad6c :
                         std::to_string(groups6c) + " units, " + std::to_string(begun6c) + " in the budgeted step, " +
                             std::to_string(shared6c) + " shared steps");
    tokens[A].clear();
    tokens[Bs].clear();
    ok = ok && steps(A, 4) && steps(Bs, 4);
    put("6. FMU1 the steps after the lone and mixed reads: tokens", ok && tokens[A] == tokens[Bs],
        std::to_string(tokens[A].size()) + " tokens");
    // 7. FMU2: joint rounds against lone steps. FMS13: at each d = 1 .. 4 in turn, the draft sparse (as serving):
    // set_draft finishes every slot, so each d takes its lone references again, then 7a, 7b and the rounds' count
    const auto read_all4 = [&]() {
        bool r = true;
        for (int32_t s = 0; s < 4 && r; ++s) {
            r = e.set_sampling(s, ro) == rt::run_status::ok && e.truncate(s, 0) == rt::run_status::ok &&
                read_straight(s, R[s], 0);
        }
        return r;
    };
    const rt::joint_frame * jf = e.test_joint();
    for (int32_t d = 1; d <= 4; ++d) {
        const std::string at = "7. FMU2 d = " + std::to_string(d) + ": ";
        for (int32_t s = 0; s < 4; ++s) {
            e.release_slot(s);
            tokens[s].clear();
        }
        ok = e.set_draft(d, true) == rt::run_status::ok;
        ok = ok && read_all4();
        for (int32_t s = 0; s < 4 && ok; ++s) {
            ok = steps(s, 12);
        }
        slot_hash            ref[4];
        std::vector<int32_t> ref_tok[4];
        for (int32_t s = 0; s < 4; ++s) {
            ref[s]     = take(e, s);
            ref_tok[s] = tokens[s];
        }
        put(at + "the engine has a joint frame; the lone references", ok && jf != nullptr,
            std::to_string(ref_tok[0].size()) + ", " + std::to_string(ref_tok[1].size()) + ", " +
                std::to_string(ref_tok[2].size()) + ", " + std::to_string(ref_tok[3].size()) + " tokens");
        const auto compare = [&](const std::string & what, std::vector<int32_t> * tok) {
            for (int32_t s = 0; s < 4; ++s) {
                const slot_hash   h  = take(e, s);
                const std::string df = diff(ref[s], h);
                put(what + ", slot " + std::to_string(s) + ": tokens and states against its lone steps",
                    ok && df.empty() && tok[s] == ref_tok[s],
                    df.empty() ? std::to_string(tok[s].size()) + " tokens" + (tok[s] == ref_tok[s] ? "" : " (differ)") :
                                 df);
            }
        };
        const int32_t        c0 = jf != nullptr ? jf->captures() : 0;
        const int64_t        r0 = jf != nullptr ? jf->rounds() : 0;
        // 7a: four lanes, one call a step
        std::vector<int32_t> jt[4];
        ok          = ok && read_all4();
        joint_steps = 0;
        for (int32_t i = 0; i < 12 && ok; ++i) {
            ok = joint_call({ 0, 1, 2, 3 }, jt);
        }
        const int32_t ja = joint_steps;
        compare(at + "four lanes (" + std::to_string(ja) + " joint lane steps)", jt);
        // 7b: lane orders rotated every step; every second step two two-member calls
        for (int32_t s = 0; s < 4; ++s) {
            jt[s].clear();
        }
        ok                   = ok && read_all4();
        joint_steps          = 0;
        const int32_t rot[4] = { 3, 1, 0, 2 };
        for (int32_t i = 0; i < 12 && ok; ++i) {
            if (i % 2 == 0) {
                std::vector<int32_t> sl;
                for (int32_t k = 0; k < 4; ++k) {
                    sl.push_back(rot[(k + i / 2) % 4]);
                }
                ok = joint_call(sl, jt);
            } else {
                ok = joint_call({ 1, 3 }, jt) && joint_call({ 2, 0 }, jt);
            }
        }
        const int32_t jb = joint_steps;
        compare(at + "rotated lanes and two-member rounds (" + std::to_string(jb) + " joint lane steps)", jt);
        put(at + "the joint rounds ran", ja >= 4 * 10 && jb >= 4 * 10,
            std::to_string(jf != nullptr ? jf->captures() - c0 : 0) + " captures, " +
                std::to_string(jf != nullptr ? jf->rounds() - r0 : 0) + " rounds at this d");
        if (jf != nullptr) {
            printf(
                "rollback-test mu1: %sthe joint frame: %.1f / %.1f MiB; graphs kept %.1f / %.1f MiB; free after "
                "the last capture %.3f / %.3f GiB\n",
                at.c_str(), double(jf->bytes(0)) / 1048576.0, double(jf->bytes(1)) / 1048576.0,
                double(jf->graph_bytes(0)) / 1048576.0, double(jf->graph_bytes(1)) / 1048576.0,
                double(jf->free_after_capture(0)) / 1073741824.0, double(jf->free_after_capture(1)) / 1073741824.0);
            printf(
                "rollback-test mu1: %sthe last joint capture (%d lanes): %lld / %lld graph nodes; slot 0's lone "
                "%d-row graph %lld / %lld\n",
                at.c_str(), jf->last_lanes(), static_cast<long long>(jf->last_nodes(0)),
                static_cast<long long>(jf->last_nodes(1)), d + 1,
                static_cast<long long>(e.test_target(0).graph_node_count(d + 1, 0)),
                static_cast<long long>(e.test_target(0).graph_node_count(d + 1, 1)));
        }
    }
    // 7c. TTTS3: four-lane joint rounds at d = 4 with the experts forced to their per-lane fallback (each lane's lone
    // gate/up and down on lane 0's stream, as a format without a joint form runs them), against the lone references.
    // One box only: on the split the back's joint frame is the follower's.
    if (!split && jf != nullptr && ok) {
        rt::joint_frame * jw = e.test_joint_frame();
        const std::string at = "7c. TTTS3 experts per lane, d = 4: ";
        for (int32_t s = 0; s < 4; ++s) {
            e.release_slot(s);
            tokens[s].clear();
        }
        ok = e.set_draft(4, true) == rt::run_status::ok && read_all4();
        for (int32_t s = 0; s < 4 && ok; ++s) {
            ok = steps(s, 12);
        }
        slot_hash            ref[4];
        std::vector<int32_t> ref_tok[4], jt[4];
        for (int32_t s = 0; s < 4; ++s) {
            ref[s]     = take(e, s);
            ref_tok[s] = tokens[s];
        }
        jw->test_set_lone(md::joint_lone_experts);
        ok          = ok && read_all4();
        joint_steps = 0;
        for (int32_t i = 0; i < 12 && ok; ++i) {
            ok = joint_call({ 0, 1, 2, 3 }, jt);
        }
        jw->test_set_lone(0);
        for (int32_t s = 0; s < 4; ++s) {
            const slot_hash   h  = take(e, s);
            const std::string df = diff(ref[s], h);
            put(at + "slot " + std::to_string(s) + ": tokens and states against its lone steps",
                ok && df.empty() && jt[s] == ref_tok[s],
                df.empty() ? std::to_string(jt[s].size()) + " tokens" + (jt[s] == ref_tok[s] ? "" : " (differ)") : df);
        }
        put(at + "the joint rounds ran", joint_steps >= 4 * 10, std::to_string(joint_steps) + " joint lane steps");
    }
    // 8. MUP1 piece 5: the slot's rotary position offset
    if (ok && jf != nullptr) {
        const bool        dry   = a.mode == "mu1-dry";
        const std::string at    = "8. MUP1 offset, d = 4: ";
        const auto        fresh = [&]() {
            for (int32_t s = 0; s < 4; ++s) {
                e.release_slot(s);
                tokens[s].clear();
            }
        };
        const auto lone_refs = [&](int32_t n, slot_hash * h, std::vector<int32_t> * tk) {
            bool r = read_all4();
            for (int32_t s = 0; s < 4 && r; ++s) {
                tokens[s].clear();
                r = steps(s, n);
            }
            for (int32_t s = 0; s < 4; ++s) {
                h[s]  = take(e, s);
                tk[s] = tokens[s];
            }
            return r;
        };
        fresh();
        ok = e.set_draft(4, true) == rt::run_status::ok;
        // 8a: never set, then set to 0 explicitly on both boxes
        slot_hash            z_h[4], z0_h[4];
        std::vector<int32_t> z_t[4], z0_t[4];
        ok = ok && lone_refs(12, z_h, z_t);
        fresh();
        for (int32_t s = 0; s < 4 && ok; ++s) {
            ok = e.set_pos_offset(s, 0) == rt::run_status::ok;
        }
        ok = ok && lone_refs(12, z0_h, z0_t);
        for (int32_t s = 0; s < 4; ++s) {
            const std::string df = diff(z_h[s], z0_h[s]);
            put(at + "a. slot " + std::to_string(s) + " offset set to 0 against never set: tokens and states",
                ok && df.empty() && z_t[s] == z0_t[s],
                df.empty() ? std::to_string(z0_t[s].size()) + " tokens" + (z_t[s] == z0_t[s] ? "" : " (differ)") : df);
        }
        // 8b: the refusals
        {
            const bool range = e.set_pos_offset(0, rt::engine::max_pos_offset + 1) == rt::run_status::bad_request &&
                               e.set_pos_offset(0, -rt::engine::max_pos_offset - 1) == rt::run_status::bad_request;
            int32_t    np8   = 0;
            const bool began = e.truncate(0, 0) == rt::run_status::ok &&
                               e.read_begin(0, R[0].data(), int32_t(R[0].size()), 0, np8) == 0;
            const bool live  = began && e.set_pos_offset(0, 5) == rt::run_status::bad_request;
            e.read_cancel(0);
            // section 4a's form: a read cancelled before its first chunk ends at its next step
            const bool ended = !began || e.read_step(0, 1) == rt::read_status::cancelled;
            put(at + "b. refused out of range and during a read; the offset unchanged",
                range && live && ended && e.pos_offset(0) == 0, "");
            fresh();
        }
        // 8c: mixed offsets, the lone references against offset 0's
        const int32_t        off[4] = { 30000, 0, 12345, -777 };
        slot_hash            x_h[4], z24_h[4];
        std::vector<int32_t> x_t[4], z24_t[4];
        ok = ok && lone_refs(24, z24_h, z24_t);  // offset 0 (every slot set to 0 in 8a), 24 steps
        fresh();
        for (int32_t s = 0; s < 4 && ok; ++s) {
            ok = e.set_pos_offset(s, off[s]) == rt::run_status::ok && e.pos_offset(s) == off[s];
        }
        ok = ok && lone_refs(24, x_h, x_t);
        for (int32_t s = 0; s < 4; ++s) {
            size_t same = 0;
            while (same < x_t[s].size() && same < z24_t[s].size() && x_t[s][same] == z24_t[s][same]) {
                ++same;
            }
            const bool roped =
                x_h[s].h[3] != z24_h[s].h[3] && x_h[s].h[6] != z24_h[s].h[6] && x_h[s].h[10] != z24_h[s].h[10];
            const bool agree = same >= 4 && 4 * x_t[s].size() >= 3 * z24_t[s].size();
            const bool pass =
                ok && (off[s] == 0 ? diff(x_h[s], z24_h[s]).empty() && x_t[s] == z24_t[s] : (dry || roped) && agree);
            put(at + "c. slot " + std::to_string(s) + " offset " + std::to_string(off[s]) +
                    (off[s] == 0 ? ": tokens and states against offset 0's (its neighbours' offsets set)" :
                                   ": against offset 0's lone steps"),
                pass,
                std::to_string(x_t[s].size()) + " / " + std::to_string(z24_t[s].size()) + " tokens in 24 steps, " +
                    std::to_string(same) + " equal before the first difference; K cache, summaries, draft K and V " +
                    (roped ? "differ" : "EQUAL") + (dry ? " (dry: zero states)" : ""));
        }
        // 8d: joint rounds with the mixed offsets against the lone references
        {
            std::vector<int32_t> jt[4];
            ok          = ok && read_all4();
            joint_steps = 0;
            for (int32_t i = 0; i < 24 && ok; ++i) {
                ok = joint_call({ 0, 1, 2, 3 }, jt);
            }
            for (int32_t s = 0; s < 4; ++s) {
                const std::string df = diff(x_h[s], take(e, s));
                put(at + "d. joint rounds (" + std::to_string(joint_steps) + " joint lane steps), slot " +
                        std::to_string(s) + " offset " + std::to_string(off[s]) +
                        ": tokens and states against its lone steps",
                    ok && joint_steps >= 4 * 20 && df.empty() && jt[s] == x_t[s],
                    df.empty() ? std::to_string(jt[s].size()) + " tokens" + (jt[s] == x_t[s] ? "" : " (differ)") : df);
            }
        }
        // 8e and 8f (PRD1 item 5): the shifted rows against offset 0 by the forced rows' KL (the joint path's KL line
        // form), not a greedy run's length. C: a slot's continuation, the first 20 tokens of its offset 0 lone steps
        // (8c's z24). Z: offset 0, the prompt read, then forced rounds of 4 rows over C. Zr: offset 0, the prompt and
        // C's first 4 tokens read back (a rollback's next read), then forced rounds over C's last 16. Xr: Zr's at the
        // slot's offset. Over C's last 16 positions, KL(Zr || Xr) against the yardstick KL(Z || Zr), pooled over the
        // shifted slots: p50 and top-1 differences at or under the yardstick's; p99 and max reported, a p99 above 1.15
        // x the yardstick's flagged for the engineer. 8e: a rollback at offsets 30,000 and 12,345 (slots 0 and 2); 8f:
        // -777 and +777 on those texts (the sign). Slot 3's text (a fragile greedy run in the first form) is dropped
        {
            constexpr int32_t VV = fn::head::n_vocab, rows8 = 4, r8 = 4, n8 = 20;
            const int32_t     shifted[2] = { 0, 2 };
            std::string       why8;
            const auto        run8 = [&](int32_t s, int32_t x, bool rolled, std::vector<float> & lg) {
                if (z24_t[s].size() < size_t(n8)) {
                    why8 = "slot " + std::to_string(s) + "'s continuation has " + std::to_string(z24_t[s].size()) +
                           " tokens";
                    return false;
                }
                std::vector<int32_t> pr = R[s];
                std::vector<int32_t> c(z24_t[s].begin() + (rolled ? r8 : 0), z24_t[s].begin() + n8);
                if (rolled) {
                    pr.insert(pr.end(), z24_t[s].begin(), z24_t[s].begin() + r8);
                }
                e.release_slot(s);
                tokens[s].clear();
                lg.clear();
                std::string fe;
                const bool  r = e.set_pos_offset(s, x) == rt::run_status::ok &&
                                e.set_sampling(s, ro) == rt::run_status::ok && e.truncate(s, 0) == rt::run_status::ok &&
                                read_straight(s, pr, 0) && e.forced_logits(s, c, rows8, lg, fe);
                if (!r && why8.empty()) {
                    why8 = "slot " + std::to_string(s) + " at offset " + std::to_string(x) + ": " + fe + e.error();
                }
                e.release_slot(s);
                return r;
            };
            std::vector<float> Z[4], Zr[4];
            for (const int32_t s : shifted) {
                ok = ok && run8(s, 0, false, Z[s]) && run8(s, 0, true, Zr[s]);
            }
            // the yardstick: offset 0's own read-back against its forced rows, at C's last 16 positions
            std::vector<double> ky;
            int32_t             ty = 0;
            for (const int32_t s : shifted) {
                for (int32_t q = 0; ok && q < n8 - r8; ++q) {
                    const float * a = Z[s].data() + size_t(q + r8) * VV;
                    const float * b = Zr[s].data() + size_t(q) * VV;
                    ky.push_back(kl_full(a, b));
                    ty += top1_full(a) != top1_full(b) ? 1 : 0;
                }
            }
            const double y50 = pct(ky, 50), y99 = pct(ky, 99);
            const double ymax = ky.empty() ? 0.0 : *std::max_element(ky.begin(), ky.end());
            const auto   arm8 = [&](const char * name, const int32_t (&xo)[2]) {
                std::vector<double> kj;
                int32_t             tj = 0;
                std::vector<float>  X;
                for (int32_t i = 0; i < 2 && ok; ++i) {
                    const int32_t s = shifted[i];
                    ok              = run8(s, xo[i], true, X);
                    for (int32_t q = 0; ok && q < n8 - r8; ++q) {
                        const float * a = Zr[s].data() + size_t(q) * VV;
                        const float * b = X.data() + size_t(q) * VV;
                        kj.push_back(kl_full(a, b));
                        tj += top1_full(a) != top1_full(b) ? 1 : 0;
                    }
                }
                const double j50 = pct(kj, 50), j99 = pct(kj, 99);
                const double jmax = kj.empty() ? 0.0 : *std::max_element(kj.begin(), kj.end());
                const bool   gate = ok && !kj.empty() && j50 <= y50 && tj <= ty;
                char         line[400];
                snprintf(line, sizeof line,
                         "KL(Zr || Xr) over %zu positions: p50 %.3g, p99 %.3g, max %.3g, top-1 differ %d; the "
                         "yardstick KL(Z || Zr): p50 %.3g, p99 %.3g, max %.3g, top-1 differ %d%s",
                         kj.size(), j50, j99, jmax, tj, y50, y99, ymax, ty,
                         j99 > 1.15 * y99 ? "; p99 ABOVE 1.15 x the yardstick's: to the engineer" : "");
                put(at + name, gate, ok ? std::string(line) + (dry ? " (dry: zero logits)" : "") : why8);
            };
            const int32_t xe[2] = { 30000, 12345 }, xf[2] = { -777, 777 };
            arm8(
                "e. a rollback at offsets 30,000 and 12,345 (slots 0 and 2): the forced rows against offset 0's "
                "(p50 and top-1 at or under the yardstick)",
                xe);
            arm8(
                "f. offsets -777 and +777 (slots 0 and 2's texts): the forced rows against offset 0's (p50 and top-1 "
                "at or under the yardstick)",
                xf);
        }
        fresh();
        for (int32_t s = 0; s < 4; ++s) {
            ok = e.set_pos_offset(s, 0) == rt::run_status::ok && ok;
        }
        put(at + "every offset back to 0", ok, "");
    }

    // 9. MUP2 (the split): reads beside four writers in units and pieces against their lone reads; the writers against
    // their lone steps. Slots 0 .. 3 the writers (each its prompt, then joint steps of the four between the reader's
    // steps, at most 300), slot 4 the reader, d = 2 (serving's). The front's budget is unknown (one unit a step:
    // the host pauses between every two units) or B = 50 ms (MUP2's T = 50 ms at F = 0.5; APF1: a known 50 ms budget:
    // as many units a step as its predictions fit, the pieces chosen by the budget); the back budgets its own units by
    // its writer-pause deadline (APF1).
    //   a. Q64 (64,000 tokens, eight chunks) at level 1, in pieces of 2,048 and 1,024 rows, and under B = 50 ms:
    //      states against its lone read; the writers' tokens and states against their lone steps (joint rounds with the
    //      switches off are I3: bitwise, KL 0); then 8 steps, a truncate inside the last, a 300-token extension and 4
    //      steps against the lone read's same sequence (a rollback after a pieced read).
    //   b. a 3,000-token append at cell 1,537 (off the 64-cell grid: a QSA block and GDN's blocks cross the piece cuts)
    //      at level 1 and in 1,024-row pieces: states against its lone append.
    //   c. a cancel mid-piece (1,024-row pieces, after three units of the first chunk's first sublayer): it ends at
    //      the chunk's end with the chunk kept (8,192 held); the read resumed from there against the lone read.
    if (split) {
        const int32_t              Rd   = 4;
        const int32_t              n_ww = 300;
        const std::vector<int32_t> Q64  = take_n(60000, 64000);
        std::vector<int32_t>       Wp[4];
        for (int32_t s = 0; s < 4; ++s) {
            Wp[s] = take_n(52000 + size_t(s) * 2500, 1500);
        }
        ok = e.set_draft(2, true) == rt::run_status::ok;
        for (int32_t s = 0; s < 5 && ok; ++s) {
            e.release_slot(s);
            ok = e.set_sampling(s, ro) == rt::run_status::ok;
        }
        put("9. MUP2: d = 2, five slots", ok, e.error());
        // the writers' lone references: each its prompt and n_ww lone steps
        slot_hash            w_ref[4];
        std::vector<int32_t> w_tok[4];
        for (int32_t s = 0; s < 4 && ok; ++s) {
            tokens[s].clear();
            ok       = e.truncate(s, 0) == rt::run_status::ok && read_straight(s, Wp[s], 0) && steps(s, n_ww);
            w_ref[s] = take(e, s);
            w_tok[s] = tokens[s];
        }
        put("9. the writers' lone references", ok,
            std::to_string(w_tok[0].size()) + ", " + std::to_string(w_tok[1].size()) + ", " +
                std::to_string(w_tok[2].size()) + ", " + std::to_string(w_tok[3].size()) + " tokens in " +
                std::to_string(n_ww) + " steps");
        // the reader's rollback sequence after a read: 8 steps, a truncate inside the last, P2 read, 4 steps (each
        // stopping at the request's end: after P2 the reader's request ends at eog on its third step)
        const auto rollback_seq = [&](int32_t s, slot_hash & h, std::vector<int32_t> & tk) {
            tokens[s].clear();
            bool                 r   = steps(s, 8);
            std::vector<int32_t> lr  = ledger_of(s);
            const int32_t        cut = int32_t(lr.size()) - 2;
            r                        = r && e.truncate(s, cut) == rt::run_status::ok;
            lr.resize(size_t(cut));
            lr.insert(lr.end(), P2.begin(), P2.end());
            r  = r && read_straight(s, lr, cut) && steps(s, 4);
            h  = take(e, s);
            tk = tokens[s];
            return r;
        };
        // the lone reference
        ok                         = ok && e.truncate(Rd, 0) == rt::run_status::ok && read_straight(Rd, Q64, 0);
        const slot_hash      q_ref = take(e, Rd);
        slot_hash            q_roll;
        std::vector<int32_t> q_roll_tok;
        ok = ok && rollback_seq(Rd, q_roll, q_roll_tok);
        put("9. Q64's lone read and its rollback sequence", ok, std::to_string(q_roll_tok.size()) + " tokens");

        // a read beside the writers: the writers from their prompts, a joint step of the four after each read step
        // (at most n_ww, the rest after the read)
        struct arm_out {
            int32_t       read_steps = 0, w_steps = 0;
            rt::read_info ri;
        };

        const auto beside = [&](const std::vector<int32_t> & tok, int32_t reuse, int32_t rows, double budget_ms,
                                int32_t cancel_after, std::vector<int32_t> * wt, arm_out & ao) {
            bool r = true;
            for (int32_t s = 0; s < 4 && r; ++s) {
                wt[s].clear();
                r = e.truncate(s, 0) == rt::run_status::ok && read_straight(s, Wp[s], 0);
            }
            eng.test_piece_rows(rows);
            int32_t np2 = 0;
            r           = r && e.read_begin(Rd, tok.data() + reuse, int32_t(tok.size()) - reuse, reuse, np2) == 0;
            rt::read_budget b;
            b.budget_ms        = budget_ms;  // APF1: a fixed budget (< 0: unknown, one unit a step)
            b.known            = budget_ms >= 0;
            rt::read_status rs = rt::read_status::yielded;
            while (r && (rs == rt::read_status::yielded || rs == rt::read_status::chunk_done)) {
                if (cancel_after >= 0 && ao.read_steps == cancel_after) {
                    e.read_cancel(Rd);
                }
                rs = e.read_step_paced(Rd, b);
                ++ao.read_steps;
                if (ao.w_steps < n_ww && (rs == rt::read_status::yielded || rs == rt::read_status::chunk_done)) {
                    r = joint_call({ 0, 1, 2, 3 }, wt);
                    ++ao.w_steps;
                }
            }
            eng.test_piece_rows(0);
            e.get_read_info(Rd, ao.ri);
            r = r && rs == (cancel_after >= 0 ? rt::read_status::cancelled : rt::read_status::done);
            while (r && ao.w_steps < n_ww) {
                r = joint_call({ 0, 1, 2, 3 }, wt);
                ++ao.w_steps;
            }
            return r;
        };
        const auto units_of = [](const rt::read_info & ri) {
            return std::to_string(ri.units_run) + " units in " + std::to_string(ri.unit_steps) + " steps (most " +
                   std::to_string(ri.most_units) + "), pieces " + std::to_string(ri.piece_min) + ".." +
                   std::to_string(ri.piece_max) + " rows";
        };

        struct arm {
            const char * name;
            int32_t      rows;
            double       budget_ms;
        };

        const arm arms[] = {
            { "level 1",                  0,    -1.0 },
            { "2,048-row pieces",         2048, -1.0 },
            { "1,024-row pieces",         1024, -1.0 },
            { "B = 50 ms, pieces chosen", 0,    50.0 }
        };
        // each read against a lone read of the same tokens taken just before it, with no round on the reader between
        // them, so every class is compared whole, the hidden carry's five rows included: a read's prompt head writes
        // the carry's row 0 (B = 1) and rows 1 .. 4 stay the slot's earlier rounds' (hc.h), the same for both
        const auto lone_now = [&](const char * what, slot_hash & h) {
            const bool r = e.truncate(Rd, 0) == rt::run_status::ok && read_straight(Rd, Q64, 0);
            h            = take(e, Rd);
            // the control (reported, not a check): this lone read against the first, rounds between them
            printf("rollback-test mu1: 9 (control). %s: Q64's lone read again against its first: %s\n", what,
                   diff(q_ref, h).empty() ? "every class equal" : ("differ in " + diff(q_ref, h)).c_str());
            return r;
        };
        for (const arm & am : arms) {
            const std::string    at = std::string("9a. Q64 beside four writers, ") + am.name;
            std::vector<int32_t> wt[4];
            arm_out              ao;
            slot_hash            q_now;
            ok                   = ok && lone_now(at.c_str(), q_now) && e.truncate(Rd, 0) == rt::run_status::ok &&
                                   beside(Q64, 0, am.rows, am.budget_ms, -1, wt, ao);
            const slot_hash   h  = take(e, Rd);
            const std::string dq = diff(q_now, h);
            put(at + ": states against its lone read (taken just before it; every class whole)", ok && dq.empty(),
                dq.empty() ? std::to_string(ao.read_steps) + " read steps, " + units_of(ao.ri) + ", " +
                                 std::to_string(ao.w_steps) + " writer steps" :
                             dq);
            for (int32_t s = 0; s < 4; ++s) {
                const slot_hash   hw = take(e, s);
                const std::string dw = diff(w_ref[s], hw);
                put(at + ": writer " + std::to_string(s) + "'s tokens and states against its lone steps (KL 0)",
                    ok && dw.empty() && wt[s] == w_tok[s],
                    dw.empty() ? std::to_string(wt[s].size()) + " tokens" + (wt[s] == w_tok[s] ? "" : " (differ)") :
                                 dw);
            }
            slot_hash            hr;
            std::vector<int32_t> hr_tok;
            ok                   = ok && rollback_seq(Rd, hr, hr_tok);
            const std::string dr = diff(q_roll, hr);
            put(at + ": a rollback after the read: states and tokens against the lone read's",
                ok && dr.empty() && hr_tok == q_roll_tok, dr.empty() ? std::to_string(hr_tok.size()) + " tokens" : dr);
        }
        // 9b: the append at cell 1,537
        const std::vector<int32_t> pre = take_n(90000, 1537);
        std::vector<int32_t>       app = pre;
        {
            const std::vector<int32_t> t3 = take_n(92000, 3000);
            app.insert(app.end(), t3.begin(), t3.end());
        }
        ok                     = ok && e.truncate(Rd, 0) == rt::run_status::ok && read_straight(Rd, pre, 0) &&
                                 read_straight(Rd, app, int32_t(pre.size()));
        const slot_hash ap_ref = take(e, Rd);
        for (const int32_t rows : { 0, 1024 }) {
            std::vector<int32_t> wt[4];
            arm_out              ao;
            ok                   = ok && e.truncate(Rd, 0) == rt::run_status::ok && read_straight(Rd, pre, 0) &&
                                   beside(app, int32_t(pre.size()), rows, -1.0, -1, wt, ao);
            const slot_hash   h  = take(e, Rd);
            const std::string da = diff(ap_ref, h);
            put(std::string("9b. 3,000 tokens appended at cell 1,537 beside four writers, ") +
                    (rows == 0 ? "level 1" : "1,024-row pieces") + ": states against its lone append",
                ok && da.empty(), da.empty() ? units_of(ao.ri) : da);
        }
        // 9c: a cancel mid-piece
        {
            std::vector<int32_t> wt[4];
            arm_out              ao;
            slot_hash            q_now;
            ok                   = ok && lone_now("9c", q_now) && e.truncate(Rd, 0) == rt::run_status::ok &&
                                   beside(Q64, 0, 1024, -1.0, 3, wt, ao);
            const int64_t kept   = e.held(Rd);
            ok                   = ok && read_straight(Rd, Q64, 8192);
            const slot_hash   h  = take(e, Rd);
            const std::string dc = diff(q_now, h);
            put("9c. a cancel mid-piece (1,024-row pieces, after 3 units): it ends at the chunk's end with the chunk "
                "kept; the resumed read's states against a lone read taken just before it (every class whole)",
                ok && kept == 8192 && dc.empty(),
                "held " + std::to_string(kept) + " after " + std::to_string(ao.read_steps) + " read steps" +
                    (dc.empty() ? "" : "; " + dc));
        }
        for (int32_t s = 0; s < 5; ++s) {
            e.release_slot(s);
        }
    } else {
        printf("rollback-test mu1: 9. MUP2's reads beside four writers run on the split (a fifth slot): skipped\n");
    }
    ladder_section();
    printf("rollback-test mu1: %s\n", g_bad == 0 ? "PASS" : "FAIL");
    if (a.joint_kl.empty()) {
        if (L != nullptr) {
            L->bye();
        }
        return g_bad == 0 ? 0 : 1;
    }
    // FMS2 item 7 (--joint-kl DEEP, load 1): joint-kl on this load's model, its state laid out again (no reload).
    // MUP2: it runs whatever sections 1-9 found (independent checks; a failed section must not cost the load its
    // joint-kl), and the run fails if either failed
    fnfl_target_args ka = a;
    ka.mode = a.mode == "mu1-dry" ? "dry" : "real";
    const int jk        = fnfl_joint_kl_on(ka, e.test_take_model(), cells, split ? &sp : nullptr);
    return g_bad == 0 && jk == 0 ? 0 : 1;
}
