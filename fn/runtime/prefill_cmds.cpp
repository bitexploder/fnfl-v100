// PF1: fnfl-drive prefill-test (fn/runtime/target_cmds.h): bulk prefill's driver (fn/runtime/prefill.h). PB1: the bulk
// path (fn/model/prompt.h) is the driver's default; PF1 Stage 1's forced slices stay selectable as the reference.
// Modes:
//   plan: no device. The chunk rules on synthetic lengths (contiguous, 1 .. 8,192 rows, no chunk across a multiple of
//         8,192, one half up to 1,536 rows and two near-equal halves of at most 4,096 above (PB3), a short chunk only at
//         a multiple of 8,192 or the prompt's end, the plan from
//         any multiple of 8,192 the uninterrupted plan's suffix) and the checkpoint bookkeeping's scenarios.
//   dry, dry-eager: zero weights (model-dry's load), 16,384 cells, the draft on (the plumbing; the numbers are the model
//         mode's): a fresh prompt by the bulk path and the forced reference against the decode path, every declared
//         call site active, a pure extension, a checkpoint's bytes restored exactly, a restore at a prompt end, a
//         mismatched prefix and n_reuse = 0; then (dry only) 9,000-token prompts: a tail, the 8,192 boundary and
//         successive chunks against the forced reference, an append and a restore each across 8,192, cancellation at
//         a chunk end and the resend, a failed chunk publishing nothing; the engine's prefill, truncate, history,
//         phase line, cancellation and failure as the server calls them; the headroom after the reservation at
//         202,816 and 262,144 cells. dry-eager: the decode passes in phases (Compute Sanitizer).
//   model (PB1 Amendment 1): one load at the given cells, the draft on, T = 0. TG1's pinned dumps checked against
//         their manifests first (no load otherwise); NC1 section 6 on three cases (fresh at 2,048, fresh at 3,584, the
//         append 2,048 -> 3,584), each teacher-forced to 4,095; then on the first 20,000 tokens: the bulk prefill
//         (the HC dump for PB2), determinism, restores from 8,192 and 16,384 bitwise with the checkpoint's bytes,
//         cancellation and the resend, the draft's caches and a 512-token continuation against the forced reference;
//         then, only when all pass and --deep is given, the deep stage (deep_stage below). PB3 Amendment 3: NC1's
//         scores are printed as a diagnostic, not counted; the functional checks count.
//   vcheck (the PB1 review's V check): no model file read (a dry load); the draft's catch-up by the forced rounds and
//         by the bulk span on identical random draft inputs and hidden rows, against each other and FP64 references
//         (vcheck_mode below).

#include "fn/kernels/qsa/attend.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/model/draft.h"
#include "fn/model/model.h"
#include "fn/model/placement.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/quant/synth.h"
#include "fn/runtime/prefill.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/target.h"
#include "fn/runtime/target_cmds.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace rt = fn::runtime;
namespace md = fn::model;

constexpr int32_t VC = 124160; // a card's logits row (pass.h)
constexpr int32_t eos_tokenizer = 248046;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "prefill-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

bool read_ids(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "prefill-test: cannot read %s\n", path.c_str());
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

template <typename T> bool get(std::vector<T> & v, const T * src, size_t n) {
    v.resize(n);
    return n == 0 || cuda_ok(cudaMemcpy(v.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
}

bool sync_cards(rt::target & t, md::model & m) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))), "sync")) {
            return false;
        }
    }
    return true;
}

// the compared state: rollback-test's seven target classes, the draft's K and V of cells < draft_n and its retained
// carry, the retained logits row and the target's carry row, both cards
constexpr int32_t n_hash = 12;
const char * const hash_names[n_hash] = {"GDN states", "GDN histories", "PLE history",     "K cache",
                                         "V cache",    "index keys",    "summaries",       "draft K",
                                         "draft V",    "draft carry",   "retained logits", "retained carry"};
struct state_hash {
    int64_t n = 0, draft_n = 0;
    int32_t retained_row = -1;
    uint64_t h[n_hash] = {};
};

bool snapshot(rt::target & t, md::model & m, state_hash & s) {
    s = state_hash();
    const rt::ledger & l = t.slot();
    s.n = l.n;
    s.draft_n = t.draft_n();
    s.retained_row = l.retained ? l.retained_row : -1;
    if (!sync_cards(t, m) || !fnfl_state_hashes(t, m, s.h)) {
        return false;
    }
    std::vector<uint16_t> u;
    std::vector<float> f;
    for (int32_t i = 7; i < n_hash; ++i) {
        s.h[i] = uint64_t(i) + 1;
    }
    for (int32_t c = 0; c < 2; ++c) {
        md::draft_card & d = t.draft(c);
        md::pass_card & p = t.card(c);
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device")) {
            return false;
        }
        if (t.draft_on()) {
            if (s.draft_n < 0 || !get(u, d.k_cache, size_t(s.draft_n) * 256)) {
                return false;
            }
            s.h[7] = hash_bytes(s.h[7] ^ uint64_t(c), u.data(), u.size() * 2);
            if (!get(u, d.v_cache, size_t(s.draft_n) * 256)) {
                return false;
            }
            s.h[8] = hash_bytes(s.h[8] ^ uint64_t(c), u.data(), u.size() * 2);
            if (!get(f, d.carry_prev, size_t(4) * 1280)) {
                return false;
            }
            s.h[9] = hash_bytes(s.h[9] ^ uint64_t(c), f.data(), f.size() * 4);
        }
        if (s.retained_row >= 0) {
            if (!get(f, p.logits + size_t(s.retained_row) * VC, size_t(VC))) {
                return false;
            }
            s.h[10] = hash_bytes(s.h[10] ^ uint64_t(c), f.data(), f.size() * 4);
            if (!get(f, p.carry + size_t(s.retained_row) * 4 * 1280, size_t(4) * 1280)) {
                return false;
            }
            s.h[11] = hash_bytes(s.h[11] ^ uint64_t(c), f.data(), f.size() * 4);
        }
    }
    return true;
}

// "equal" or the differing classes
std::string compare(const state_hash & a, const state_hash & b) {
    std::string d;
    if (a.n != b.n || a.draft_n != b.draft_n) {
        d = "N " + std::to_string(a.n) + " / " + std::to_string(b.n) + ", draft N " + std::to_string(a.draft_n) +
            " / " + std::to_string(b.draft_n);
    }
    for (int32_t i = 0; i < n_hash; ++i) {
        if (a.h[i] != b.h[i]) {
            d += (d.empty() ? "" : ", ") + std::string(hash_names[i]);
        }
    }
    return d.empty() ? "equal" : "DIFFERENT: " + d;
}

std::string chunk_list(const rt::pf_run_stats & s) {
    std::string out;
    for (const rt::pf_chunk_stats & c : s.chunks) {
        out += (out.empty() ? "" : " ") + std::to_string(c.c.start) + "+" + std::to_string(c.c.rows) +
               (c.checkpoint >= 0 ? "*" : "");
    }
    return out;
}

struct verdicts {
    FILE * log = nullptr;
    bool all = true;
    int32_t n = 0;

    void put(const std::string & name, bool ok, const std::string & what) {
        printf("prefill-test: %s: %s: %s\n", name.c_str(), ok ? "PASS" : "FAIL", what.c_str());
        if (log != nullptr) {
            fprintf(log, "%s: %s: %s\n", name.c_str(), ok ? "PASS" : "FAIL", what.c_str());
            fflush(log);
        }
        fflush(stdout);
        all = all && ok;
        ++n;
    }
};

// ---- plan: the chunk rules and the bookkeeping, no device ----

std::string check_plan(int64_t from, int64_t to) {
    const std::vector<rt::pf_chunk> p = rt::plan_chunks(from, to);
    if (p.empty() || p.front().start != from) {
        return "does not start at from";
    }
    int64_t pos = from;
    for (size_t i = 0; i < p.size(); ++i) {
        const rt::pf_chunk & c = p[i];
        const int64_t end = c.start + c.rows;
        if (c.start != pos) {
            return "not contiguous at " + std::to_string(pos);
        }
        if (c.rows < 1 || c.rows > rt::pf_chunk_rows) {
            return std::to_string(c.rows) + " rows at " + std::to_string(c.start);
        }
        if (c.start / rt::pf_span != (end - 1) / rt::pf_span) {
            return "crosses a multiple of 8,192 at " + std::to_string(c.start);
        }
        const bool split = c.rows > rt::pf_split_rows;
        if (c.half[0] + c.half[1] != c.rows || (!split && c.half[1] != 0) ||
            (split && (c.half[0] % 64 != 0 || c.half[0] < c.half[1] || c.half[0] - c.half[1] > 128 ||
                       c.half[0] > rt::pf_half_rows || c.half[1] < 1))) {
            return "halves " + std::to_string(c.half[0]) + " + " + std::to_string(c.half[1]);
        }
        // the tiling restarts only at from and at multiples of 8,192: a short chunk ends at one or at the end
        if (c.rows < rt::pf_chunk_rows && end % rt::pf_span != 0 && end != to) {
            return "a short chunk " + std::to_string(c.start) + "+" + std::to_string(c.rows) + " inside a span";
        }
        pos = end;
    }
    if (pos != to) {
        return "ends at " + std::to_string(pos);
    }
    // the plan from each multiple of 8,192 inside is this plan's suffix (a restore repeats the partition)
    for (int64_t k = (from / rt::pf_span + 1) * rt::pf_span; k < to; k += rt::pf_span) {
        const std::vector<rt::pf_chunk> q = rt::plan_chunks(k, to);
        size_t j = 0;
        while (j < p.size() && p[j].start != k) {
            ++j;
        }
        if (j == p.size() || p.size() - j != q.size()) {
            return "the plan from " + std::to_string(k) + " is not a suffix";
        }
        for (size_t i = 0; i < q.size(); ++i) {
            if (q[i].start != p[j + i].start || q[i].rows != p[j + i].rows) {
                return "the plan from " + std::to_string(k) + " differs at " + std::to_string(q[i].start);
            }
        }
    }
    return "";
}

int plan_mode(verdicts & v) {
    // the chunk rules: every from with every length
    const int64_t froms[] = {0,    1,    3,    4,    1000,  3071,  3072,  3073,  6143,  6144,
                             7001, 8191, 8192, 8193, 12000, 16383, 16384, 20000, 196607};
    const int64_t lens[] = {1,    2,    3,    4,    5,    1535, 1536, 1537, 1538,  2048,  3071,  3072,
                            3073, 3391, 4096, 6144, 6145, 8191, 8192, 8193, 10000, 16385, 24577, 50000};
    int32_t cases = 0;
    std::string bad;
    for (const int64_t f : froms) {
        for (const int64_t n : lens) {
            const std::string e = check_plan(f, f + n);
            ++cases;
            if (!e.empty() && bad.empty()) {
                bad = std::to_string(f) + " .. " + std::to_string(f + n) + ": " + e;
            }
        }
    }
    for (const int64_t to : {int64_t(8192), int64_t(131072), int64_t(198720)}) {
        const std::string e = check_plan(0, to);
        ++cases;
        if (!e.empty() && bad.empty()) {
            bad = "0 .. " + std::to_string(to) + ": " + e;
        }
    }
    v.put("chunk rules", bad.empty(),
          bad.empty() ? std::to_string(cases) + " (from, to) cases: contiguous, 1 .. 8,192 rows, none across a "
                                                "multiple of 8,192, one half up to 1,536 rows, above it two halves "
                                                "within 128 rows (the first a multiple of 64, <= 4,096), short chunks "
                                                "only at a multiple or the end, every plan from a multiple its suffix"
                      : bad);
    // the chunk shapes of a full slot and of 131,072 (information)
    for (const int64_t to : {int64_t(131072), int64_t(198720)}) {
        std::map<int32_t, int32_t> hist;
        int32_t ck = 0;
        for (const rt::pf_chunk & c : rt::plan_chunks(0, to)) {
            ++hist[c.rows];
            ck += rt::checkpoint_due(c.start + c.rows, to) ? 1 : 0;
        }
        std::string h;
        for (const auto & e : hist) {
            h += (h.empty() ? "" : ", ") + std::to_string(e.second) + " x " + std::to_string(e.first);
        }
        printf("prefill-test: plan 0 .. %lld: chunks %s; checkpoints due %d\n", static_cast<long long>(to), h.c_str(),
               ck);
        if (v.log != nullptr) {
            fprintf(v.log, "plan 0 .. %lld: chunks %s; checkpoints due %d\n", static_cast<long long>(to), h.c_str(),
                    ck);
        }
    }
    {
        const bool ok = rt::checkpoint_due(8192, 9000) && rt::checkpoint_due(9000, 9000) &&
                        !rt::checkpoint_due(8191, 9000) && !rt::checkpoint_due(0, 9000) &&
                        rt::checkpoint_due(16384, 16384) && !rt::checkpoint_due(3072, 9000);
        v.put("checkpoint positions", ok, "multiples of 8,192 and the prompt's end only");
    }

    // the bookkeeping
    std::vector<int32_t> T(110000);
    for (size_t i = 0; i < T.size(); ++i) {
        T[i] = static_cast<int32_t>((i * 2654435761u) % 248000u);
    }
    const auto key = [&](const std::vector<int32_t> & t, int64_t n) { return rt::token_key(t.data(), n); };
    const auto prefix = [&](int64_t n) { return std::vector<int32_t>(T.begin(), T.begin() + n); };
    const auto rec = [&](rt::checkpoint_book & b, const std::vector<int32_t> & t, int64_t pos) {
        b.record(b.slot_for(pos), pos, key(t, pos));
    };
    // A: a prompt of 20,000; then one that keeps 12,000 and differs after: 8,192's checkpoint
    {
        rt::checkpoint_book b;
        for (const int64_t p : {int64_t(8192), int64_t(16384), int64_t(20000)}) {
            rec(b, T, p);
        }
        std::vector<int32_t> a = prefix(12000);
        a.push_back(7);
        a.push_back(8);
        const int32_t s = rt::choose_checkpoint(b, a, 12000);
        const bool ok = s >= 0 && b.at(s).pos == 8192 && b.count() == 3;
        v.put("book A (restore at 8,192 for n_reuse 12,000)", ok,
              "chose " + std::to_string(s >= 0 ? b.at(s).pos : -1) + ", " + std::to_string(b.count()) + " held");
    }
    // B: a token below 8,192 differs: 8,192's prefix does not match; it is dropped, none fits
    {
        rt::checkpoint_book b;
        for (const int64_t p : {int64_t(8192), int64_t(16384), int64_t(20000)}) {
            rec(b, T, p);
        }
        std::vector<int32_t> a = prefix(12000);
        a[100] ^= 1;
        const int32_t s = rt::choose_checkpoint(b, a, 12000);
        const bool ok = s < 0 && b.count() == 2 && b.latest(20000) >= 0 && b.at(b.latest(20000)).pos == 20000;
        v.put("book B (a mismatched prefix: none)", ok,
              "chose " + std::to_string(s) + ", " + std::to_string(b.count()) + " held (8,192 dropped)");
    }
    // C: the cap: 12 multiples then 100,000; the last 8 written stay
    {
        rt::checkpoint_book b;
        for (int64_t k = 1; k <= 12; ++k) {
            rec(b, T, k * 8192);
        }
        rec(b, T, 100000);
        std::vector<int64_t> held;
        for (int32_t i = 0; i < rt::pf_max_checkpoints; ++i) {
            if (b.at(i).pos >= 0) {
                held.push_back(b.at(i).pos);
            }
        }
        std::sort(held.begin(), held.end());
        const std::vector<int64_t> want = {49152, 57344, 65536, 73728, 81920, 90112, 98304, 100000};
        std::string h;
        for (const int64_t p : held) {
            h += (h.empty() ? "" : " ") + std::to_string(p);
        }
        const bool latest_ok =
            b.latest(49151) < 0 && b.at(b.latest(70000)).pos == 65536 && b.at(b.latest(200000)).pos == 100000;
        v.put("book C (8 a slot, the oldest written dropped)", held == want && latest_ok, "held " + h);
    }
    // D: a checkpoint written again at its position reuses its slot (a re-prefill through it)
    {
        rt::checkpoint_book b;
        rec(b, T, 8192);
        rec(b, T, 16384);
        const int32_t s0 = b.latest(8192);
        const int32_t s1 = b.slot_for(8192);
        rec(b, T, 8192);
        v.put("book D (the same position, the same slot)", s0 == s1 && b.count() == 2 && b.latest(8192) == s0,
              std::to_string(b.count()) + " held");
    }
    // E: append: 7,001 (a prompt's end), then the extension to 17,000; a later prompt keeping 7,500 restores 7,001
    {
        rt::checkpoint_book b;
        for (const int64_t p : {int64_t(7001), int64_t(8192), int64_t(16384), int64_t(17000)}) {
            rec(b, T, p);
        }
        std::vector<int32_t> a = prefix(7500);
        a.push_back(1);
        const int32_t s = rt::choose_checkpoint(b, a, 7500);
        const bool ok = s >= 0 && b.at(s).pos == 7001;
        b.drop_after(7001);
        v.put("book E (append, then restore at the prompt end 7,001)", ok && b.count() == 1,
              "chose " + std::to_string(s >= 0 ? b.at(s).pos : -1) + ", " + std::to_string(b.count()) +
                  " held after drop_after");
    }
    // F: n_reuse 0 resets; G: the same prompt again (n_reuse = L) re-processes its last token: the prompt end's
    // checkpoint does not fit, the one before it does
    {
        rt::checkpoint_book b;
        for (const int64_t p : {int64_t(8192), int64_t(16384), int64_t(17000)}) {
            rec(b, T, p);
        }
        const std::vector<int32_t> a = prefix(17000);
        const int32_t s0 = rt::choose_checkpoint(b, a, 0);
        const int32_t s1 = rt::choose_checkpoint(b, a, 17000);
        v.put("book F (n_reuse 0: none)", s0 < 0 && b.count() == 3, "chose " + std::to_string(s0));
        v.put("book G (the same prompt: 16,384, not its end)", s1 >= 0 && b.at(s1).pos == 16384,
              "chose " + std::to_string(s1 >= 0 ? b.at(s1).pos : -1));
    }
    // H: a restore at a prompt end below a multiple; the re-prefill's plan starts there and its next chunk ends at
    // 8,192
    {
        const std::vector<rt::pf_chunk> p = rt::plan_chunks(7001, 9000);
        const bool ok =
            p.size() == 2 && p[0].start == 7001 && p[0].rows == 1191 && p[1].start == 8192 && p[1].rows == 808;
        v.put("plan H (7,001 .. 9,000)", ok, "7001+1191 8192+808");
    }
    return v.all ? 0 : 1;
}

// ---- dry: zero weights ----

// version cur's GDN states and histories, the PLE history and the draft's retained carry, both cards
bool slot_bytes(rt::target & t, md::model & m, uint32_t cur, std::vector<uint8_t> & out) {
    out.clear();
    std::vector<uint8_t> b;
    const auto add = [&](const void * p, size_t n) {
        b.resize(n);
        if (!cuda_ok(cudaMemcpy(b.data(), p, n, cudaMemcpyDeviceToHost), "read state")) {
            return false;
        }
        out.insert(out.end(), b.begin(), b.end());
        return true;
    };
    constexpr size_t gdn = size_t(24) * 128 * 128 * 4, conv = size_t(5120) * 3 * 4, ple = size_t(5120) * 9 * 4;
    for (int32_t c = 0; c < 2; ++c) {
        md::pass_card & p = t.card(c);
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device")) {
            return false;
        }
        for (const md::layer_state & st : p.s) {
            if (st.gdn != nullptr && (!add(reinterpret_cast<const uint8_t *>(st.gdn) + cur * gdn, gdn) ||
                                      !add(reinterpret_cast<const uint8_t *>(st.conv) + cur * conv, conv))) {
                return false;
            }
        }
        if (!add(reinterpret_cast<const uint8_t *>(p.ple_hist) + cur * ple, ple) ||
            (t.draft_on() && !add(t.draft(c).carry_prev, size_t(4) * 1280 * 4))) {
            return false;
        }
    }
    return true;
}

// random F32 values in [-1, 1) into the same regions
bool slot_scribble(rt::target & t, md::model & m, uint32_t cur, uint64_t seed) {
    uint64_t x = seed;
    std::vector<float> h;
    const auto put = [&](void * p, size_t n) {
        h.resize(n / 4);
        for (float & v : h) {
            x = x * 6364136223846793005ull + 1442695040888963407ull;
            v = static_cast<float>(static_cast<double>(x >> 11) * (2.0 / 9007199254740992.0) - 1.0);
        }
        return cuda_ok(cudaMemcpy(p, h.data(), n, cudaMemcpyHostToDevice), "write state");
    };
    constexpr size_t gdn = size_t(24) * 128 * 128 * 4, conv = size_t(5120) * 3 * 4, ple = size_t(5120) * 9 * 4;
    for (int32_t c = 0; c < 2; ++c) {
        md::pass_card & p = t.card(c);
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") || !sync_cards(t, m)) {
            return false;
        }
        for (const md::layer_state & st : p.s) {
            if (st.gdn != nullptr && (!put(reinterpret_cast<uint8_t *>(st.gdn) + cur * gdn, gdn) ||
                                      !put(reinterpret_cast<uint8_t *>(st.conv) + cur * conv, conv))) {
                return false;
            }
        }
        if (!put(reinterpret_cast<uint8_t *>(p.ple_hist) + cur * ple, ple) ||
            (t.draft_on() && !put(t.draft(c).carry_prev, size_t(4) * 1280 * 4))) {
            return false;
        }
    }
    return true;
}

// the target's ledger and the draft at the handoff
std::string handoff(const rt::target & t, int64_t L) {
    const rt::ledger & l = t.slot();
    const bool ok = l.n == L && l.u == 0 && static_cast<int64_t>(l.tok.size()) == L && l.retained && l.o == 0 &&
                    (!t.draft_on() || t.draft_n() == L);
    return ok ? "" : "the handoff state is not N = L = " + std::to_string(L) + ", U = 0, retained, O = 0, draft at N";
}

// PB1: the prompt path's declared call sites after a bulk run, each enqueued at least once (no adapter exists: every
// site is its bulk supplier)
std::string site_line(rt::prefiller & pf, bool & active) {
    const auto & n = pf.prompt()->site_counts();
    active         = true;
    std::string s;
    for (int32_t i = 0; i < md::n_psites; ++i) {
        active = active && n[size_t(i)] > 0;
        s += (s.empty() ? "" : ", ") + std::string(md::psite_name(static_cast<md::psite>(i))) + " " +
             std::to_string(n[size_t(i)]);
    }
    return s;
}

// PB1 item 6: the phase line's order (entry <= draft <= decode-ready, the target marked unavailable) and its row and
// checkpoint fields; restored -1: none. Empty when it holds.
std::string check_phase(const std::string & line, int64_t rows, int64_t restored) {
    const auto num = [&](const char * key, double & out) {
        const size_t p = line.find(key);
        return p != std::string::npos && sscanf(line.c_str() + p + strlen(key), "%lf", &out) == 1;
    };
    double e = 0, d = 0, r = 0, n = 0, ck = -1;
    if (!num("entry ", e) || !num("; draft ", d) || !num("; decode-ready ", r) || !num("; rows ", n)) {
        return "the phase line lacks a field: " + line;
    }
    const size_t pc = line.find("restored checkpoint ");
    if (pc == std::string::npos) {
        return "no restored checkpoint field: " + line;
    }
    if (line.compare(pc + 20, 4, "none") != 0 && sscanf(line.c_str() + pc + 20, "%lf", &ck) != 1) {
        return "a bad restored checkpoint field: " + line;
    }
    if (line.find("target unavailable") == std::string::npos) {
        return "the target's marker is not marked unavailable: " + line;
    }
    if (!(e <= d && d <= r)) {
        return "out of order: " + line;
    }
    if (int64_t(n) != rows || int64_t(ck) != restored) {
        return "rows " + std::to_string(int64_t(n)) + " (expected " + std::to_string(rows) + "), checkpoint " +
               std::to_string(int64_t(ck)) + " (expected " + std::to_string(restored) + ")";
    }
    return "";
}

// the bulk run's chunk list and its checkpoints
bool chunks_are(const rt::pf_run_stats & s, const std::string & want) {
    return chunk_list(s) == want;
}

int dry_mode(const fnfl_target_args & a, const std::vector<int32_t> & pool, verdicts & v) {
    const bool eager = a.mode == "dry-eager";
    const int32_t n = std::min<int32_t>(a.n_pos, 7000);
    constexpr int32_t cells = 16384, L9 = 9000;
    if (static_cast<int32_t>(pool.size()) < L9 + 1600 || n < 64) {
        fprintf(stderr, "prefill-test: dry needs %d tokens and 64 .. 7,000 positions\n", L9 + 1600);
        return 1;
    }
    rt::draft_options dopt;
    if (!fnfl_draft_settings(a, true, "prefill-test", dopt)) {
        return 1;
    }
    auto m = std::make_unique<md::model>();
    if (!fnfl_load_dry(a, cells, *m)) {
        return 1;
    }
    printf("prefill-test: DRY (%s): zero weights, %d cells, a prompt of %d tokens%s\n", eager ? "eager" : "graphs",
           cells, n, eager ? "" : "; then 9,000-token prompts (a tail, the 8,192 boundary, successive chunks)");
    std::string err;
    {
        rt::target_options o;
        o.model_path = a.model_path;
        o.eager = eager;
        o.sample = true;
        o.sampling.temp = 0.0f;
        o.draft = dopt;
        o.timeout_s = 120.0;
        rt::target t;
        rt::prefiller pf;
        if (!t.init(*m, o, err) || !pf.init(t, *m, err)) {
            fprintf(stderr, "prefill-test: init: %s\n", err.c_str());
            return 1;
        }
        const std::vector<int32_t> P(pool.begin(), pool.begin() + n);
        const int32_t              n_chunks = static_cast<int32_t>(rt::plan_chunks(0, n).size());
        int64_t from = -1;
        rt::pf_run_stats st;
        state_hash                 hd, hp, hf;
        // D1: the decode path, then the bulk path from token 0 (zero weights: the plumbing; the numbers are the model
        // mode's)
        // brief (dry-eager above 512 positions, the sanitizers' two-chunk run): the bulk path alone, D1 and D2; the
        // decode-path and forced references are the graphs run's and dry-eager 64's
        const bool                 brief = eager && n > 512;
        pf.prompt()->clear_site_counts();
        bool ok = (brief || (rt::prefill(t, P, err) && snapshot(t, *m, hd))) && pf.begin(t, P, 0, from, err) &&
                  pf.run(t, &st, err) && snapshot(t, *m, hp);
        if (brief) {
            hd = hp;
        }
        std::string h = ok ? handoff(t, n) : err;
        v.put(brief ? "dry fresh prompt (bulk; brief: no decode-path reference)" : "dry fresh prompt (bulk)",
              ok && st.bulk && from == 0 && h.empty() && compare(hd, hp) == "equal" && pf.book().count() == 1 &&
                  st.rounds == n_chunks,
              ok ? "chunks " + chunk_list(st) + "; " + std::to_string(st.rounds) +
                       " chunks; against the decode path: " + compare(hd, hp) + (h.empty() ? "" : "; " + h) :
                   err);
        bool              active = false;
        const std::string sites  = site_line(pf, active);
        v.put("dry declared call sites", ok && active, "every site enqueued by the bulk run: " + sites);
        // D1b: the forced reference (PF1 Stage 1's slices), selectable
        if (!brief) {
            pf.set_bulk(false);
            ok = ok && pf.begin(t, P, 0, from, err) && pf.run(t, &st, err) && snapshot(t, *m, hf);
            pf.set_bulk(true);
            v.put("dry fresh prompt (forced reference)",
                  ok && !st.bulk && compare(hd, hf) == "equal" && st.rounds == (n + 3) / 4 && handoff(t, n).empty(),
                  ok ? std::to_string(st.rounds) + " rounds; against the decode path: " + compare(hd, hf) : err);
        }
        // D2: a pure extension of 8 tokens
        const std::vector<int32_t> X(pool.begin() + n, pool.begin() + n + 8);
        ok = ok && pf.begin(t, P, 0, from, err) && pf.run(t, &st, err) && rt::extend_ledger(t, X.data(), 8, err) &&
             pf.run(t, &st, err);
        h = ok ? handoff(t, n + 8) : err;
        v.put("dry pure extension", ok && h.empty() && pf.book().count() == 2 && st.from == n,
              ok ? "chunks " + chunk_list(st) + (h.empty() ? "" : "; " + h) : err);
        if (brief) {
            return v.all ? 0 : 1;
        }
        // D3: a checkpoint's bytes restored exactly: random state, the checkpoint, other random state, the restore
        const uint32_t cur0 = t.slot().cur;
        std::vector<uint8_t> want, got;
        ok = ok && slot_scribble(t, *m, cur0, 0x1234) && pf.checkpoint_now(t, err) && slot_bytes(t, *m, cur0, want) &&
             slot_scribble(t, *m, cur0, 0x5678) && slot_bytes(t, *m, cur0, got);
        const bool scribbled = ok && got != want;
        std::vector<int32_t> all(t.slot().tok.begin(), t.slot().tok.begin() + n + 8);
        all.push_back(pool[size_t(n) + 8]);
        all.push_back(pool[size_t(n) + 9]);
        ok = ok && pf.begin(t, all, n + 8, from, err) && sync_cards(t, *m) && slot_bytes(t, *m, t.slot().cur, got);
        const bool exact = ok && got == want && t.slot().cur == 0;
        ok = ok && pf.run(t, &st, err);
        h = ok ? handoff(t, n + 10) : err;
        v.put("dry checkpoint bytes restored", ok && scribbled && exact && from == n + 8 && h.empty(),
              ok ? std::to_string(want.size()) + " bytes (version " + std::to_string(cur0) +
                       " saved, version 0 restored): " + (exact ? "equal" : "DIFFERENT") + "; restored at " +
                       std::to_string(from) + ", chunks " + chunk_list(st)
                 : err);
        // D4: a prompt keeping n + 2 tokens restores the prompt end n and drops the later checkpoints
        std::vector<int32_t> b(P);
        b.push_back(pool[size_t(n)]);
        b.push_back(pool[size_t(n) + 1]);
        b.push_back(11);
        b.push_back(12);
        ok = ok && pf.begin(t, b, n + 2, from, err) && pf.run(t, &st, err);
        h = ok ? handoff(t, n + 4) : err;
        v.put("dry restore at the prompt end", ok && from == n && h.empty() && pf.book().count() == 2,
              ok ? "restored at " + std::to_string(from) + " (n_reuse " + std::to_string(n + 2) + "), chunks " +
                       chunk_list(st) + ", " + std::to_string(pf.book().count()) + " held"
                 : err);
        // D5: a token below every checkpoint differs: from token 0, the book cleared
        std::vector<int32_t> c(b);
        c[5] ^= 1;
        ok = ok && pf.begin(t, c, n + 2, from, err) && pf.run(t, &st, err);
        v.put("dry mismatched prefix", ok && from == 0 && pf.book().count() == 1 && handoff(t, n + 4).empty(),
              ok ? "from " + std::to_string(from) + ", " + std::to_string(pf.book().count()) + " held" : err);
        // D6: n_reuse 0
        ok = ok && pf.begin(t, c, 0, from, err) && pf.run(t, &st, err);
        v.put("dry n_reuse 0", ok && from == 0 && handoff(t, n + 4).empty(), ok ? "from 0" : err);
        if (!ok) {
            printf("prefill-test: dry: %s\n", err.c_str());
        }
        // PB1's 9,000-token cases (graphs mode; dry-eager is the sanitizers' small run)
        if (!eager && ok) {
            const std::vector<int32_t> P9(pool.begin(), pool.begin() + L9);
            state_hash                 h9, h9f, hx, hr, hrf, hc;
            // B1: a tail, the 8,192 boundary and successive chunks; against the forced reference
            ok                    = pf.begin(t, P9, 0, from, err) && pf.run(t, &st, err) && snapshot(t, *m, h9);
            const std::string ch9 = ok ? chunk_list(st) : "";
            pf.set_bulk(false);
            ok = ok && pf.begin(t, P9, 0, from, err) && pf.run(t, &st, err) && snapshot(t, *m, h9f);
            pf.set_bulk(true);
            v.put("dry 9,000: tail, 8,192 boundary, successive chunks",
                  ok && ch9 == "0+8192* 8192+808*" && compare(h9, h9f) == "equal" && handoff(t, L9).empty(),
                  ok ? "chunks " + ch9 + "; against the forced reference: " + compare(h9, h9f) : err);
            // B1b (PB1's determinism follow-up): every arena byte poisoned with NaN, then the same prompt, fresh and
            // restored from 8,192: a run that read what an earlier run left in the arena would carry the NaN
            state_hash hz, hzr;
            ok = ok && pf.begin(t, P9, 0, from, err) && pf.prompt()->test_poison(err) && pf.run(t, &st, err) &&
                 snapshot(t, *m, hz) && pf.prompt()->test_poison(err) && pf.begin(t, P9, 8192, from, err) &&
                 pf.run(t, &st, err) && snapshot(t, *m, hzr);
            v.put(
                "dry 9,000 after the arena poisoned (fresh, and restored at 8,192)",
                ok && compare(h9, hz) == "equal" && compare(h9, hzr) == "equal" && from == 8192,
                ok ? "fresh: " + compare(h9, hz) + "; restored from " + std::to_string(from) + ": " + compare(h9, hzr) :
                     err);
            // B2: append across 8,192: 7,001, then a pure extension to 9,000
            rt::pf_run_stats s0;
            ok = ok && pf.begin(t, std::vector<int32_t>(P9.begin(), P9.begin() + 7001), 0, from, err) &&
                 pf.run(t, &s0, err) && rt::extend_ledger(t, P9.data() + 7001, L9 - 7001, err) && pf.run(t, &st, err) &&
                 snapshot(t, *m, hx);
            v.put("dry append 7,001 -> 9,000 across 8,192",
                  ok && chunk_list(s0) == "0+7001*" && chunks_are(st, "7001+1191* 8192+808*") &&
                      compare(h9, hx) == "equal" && handoff(t, L9).empty(),
                  ok ? "chunks " + chunk_list(s0) + " | " + chunk_list(st) + "; against fresh: " + compare(h9, hx)
                     : err);
            // B3: restore at 7,001 and re-prefill across 8,192 (A keeps 7,500 of P9, then 1,500 others)
            std::vector<int32_t> A(P9.begin(), P9.begin() + 7500);
            A.insert(A.end(), pool.begin() + L9, pool.begin() + L9 + 1500);
            ok = ok && pf.begin(t, A, 7500, from, err) && pf.run(t, &st, err) && snapshot(t, *m, hr);
            const int64_t     from_r = from;
            const std::string chr    = ok ? chunk_list(st) : "";
            ok = ok && pf.begin(t, A, 0, from, err) && pf.run(t, &st, err) && snapshot(t, *m, hrf);
            v.put("dry restore at 7,001 across 8,192",
                  ok && from_r == 7001 && chr == "7001+1191* 8192+808*" && compare(hr, hrf) == "equal",
                  ok ? "from " + std::to_string(from_r) + ", chunks " + chr +
                           "; against A from token 0: " + compare(hr, hrf) :
                       err);
            // B4: cancellation at a chunk end through the callback (8,192), then the resend from its checkpoint
            const rt::prefill_progress stop_8192 = [](int64_t, int64_t done, int64_t) {
                return done < 8192;
            };
            ok                 = ok && pf.begin(t, P9, 0, from, err);
            const bool stopped = ok && !pf.run(t, &st, err, stop_8192);
            const bool at      = stopped && st.stopped == 8192 && t.slot().n == 8192 && t.draft_n() == 8192 &&
                                 pf.book().latest(8192) >= 0 && pf.book().at(pf.book().latest(8192)).pos == 8192;
            ok = ok && stopped && pf.begin(t, P9, 8192, from, err) && pf.run(t, &st, err) && snapshot(t, *m, hc);
            v.put("dry cancellation at 8,192 and the resend",
                  ok && at && from == 8192 && compare(h9, hc) == "equal" && handoff(t, L9).empty(),
                  ok ? std::string("stopped at ") + std::to_string(8192) +
                           (at ? " (N, draft and checkpoint at it)" : " WRONG STATE") + "; resend from " +
                           std::to_string(from) + ", chunks " + chunk_list(st) +
                           "; against uninterrupted: " + compare(h9, hc) :
                       err);
            // B5: a failed chunk (card 1's error word set during chunk 1) publishes nothing: N and the committed slot
            // stay at chunk 0's, no step is recorded; then a fresh prompt runs clean
            uint32_t                   cur_after0 = 0;
            const rt::prefill_progress fail_next = [&](int64_t, int64_t done, int64_t) {
                if (done == 8192) {
                    cur_after0 = t.slot().cur;
                    pf.prompt()->test_fail_next(1);
                }
                return true;
            };
            ok = ok && pf.begin(t, P9, 0, from, err);
            std::string ferr;
            const bool  failed = ok && !pf.run(t, &st, ferr, fail_next);
            const bool kept = failed && st.stopped < 0 && t.slot().n == 8192 && t.slot().cur == cur_after0 &&
                              !t.versions().last().valid && st.chunks.size() == 1;
            pf.drop_all();  // the engine's response to a failed prefill
            ok = ok && failed && pf.begin(t, P9, 0, from, err) && pf.run(t, &st, err) && snapshot(t, *m, hc);
            v.put("dry failed chunk publishes nothing", ok && kept && compare(h9, hc) == "equal",
                  ok ? "the run failed (" + ferr + "); N " + std::to_string(kept ? 8192 : -1) +
                           ", committed slot and step record " + (kept ? "kept at chunk 0's" : "WRONG") +
                           "; a fresh run after it: " + compare(h9, hc)
                     : err);
            if (!ok) {
                printf("prefill-test: dry 9,000: %s\n", err.c_str());
            }
        }
    }
    if (eager) {
        return v.all ? 0 : 1;
    }
    // the engine as the server drives it
    {
        rt::engine_config c;
        c.model_path = a.model_path;
        c.n_ctx_slot = cells;
        c.draft      = dopt;
        c.eog        = { eos_tokenizer };
        c.timeout_s  = 30.0;
        rt::engine e;
        if (!e.init(std::move(m), c, err)) {
            fprintf(stderr, "prefill-test: engine: %s\n", err.c_str());
            return 1;
        }
        rt::request_options ro;
        ro.sampling.temp = 0.0f;
        ro.max_tokens    = 64;
        std::vector<int32_t> held, hist;
        int32_t              pending = -1, np = 0, n_out = 0;
        int32_t              out[rt::max_step_tokens] = {};
        bool                 end                      = false;
        const auto           expect = [&](const std::string & what, bool cond, const std::string & more) {
            e.history(0, hist, pending);
            const bool same = hist == held && pending == -1;
            v.put("engine " + what, cond && same,
                  more + "; history " + std::to_string(hist.size()) + " tokens (expected " +
                      std::to_string(held.size()) + ")" + (same ? "" : " MISMATCH") +
                      (e.error().empty() ? "" : " [" + e.error() + "]"));
        };
        const auto phase = [&](const std::string & what, int64_t rows, int64_t restored) {
            const std::string why = check_phase(e.last_phase_line(), rows, restored);
            v.put("engine phase line: " + what, why.empty(), why.empty() ? e.last_phase_line() : why);
        };
        const auto steps = [&](int32_t k) {
            bool ok = true;
            for (int32_t i = 0; i < k && ok; ++i) {
                ok = e.step(0, out, n_out, end) == rt::run_status::ok && n_out >= 1;
            }
            return ok;
        };
        const int32_t h  = n / 2;
        // E1: a fresh prompt of h tokens; 3 steps; a deep truncate to h; the next prefill restores h's checkpoint
        bool          ok = e.set_sampling(0, ro) == rt::run_status::ok && e.truncate(0, 0) == rt::run_status::ok &&
                           e.prefill(0, pool.data(), h, 0, np) == rt::run_status::ok;
        held.assign(pool.begin(), pool.begin() + h);
        expect("fresh prompt", ok && np == h, "n_processed " + std::to_string(np));
        phase("fresh prompt", h, -1);
        ok = ok && steps(3) && e.truncate(0, h) == rt::run_status::ok;
        ok = ok && e.prefill(0, pool.data() + h, n - h, h, np) == rt::run_status::ok;
        held.assign(pool.begin(), pool.begin() + n);
        expect("restore after a deep truncate", ok && np == n - h && e.last_prefill().from == h,
               "n_processed " + std::to_string(np) + " (expected " + std::to_string(n - h) + "), from " +
                   std::to_string(e.last_prefill().from) + ", chunks " + chunk_list(e.last_prefill()));
        phase("restore", n - h, h);
        // E2: 2 steps; a truncate to h - 5 with other tokens: no checkpoint at or below it, from token 0
        const int32_t other[3] = { 21, 22, 23 };
        ok                     = ok && steps(2) && e.truncate(0, h - 5) == rt::run_status::ok &&
                                 e.prefill(0, other, 3, h - 5, np) == rt::run_status::ok;
        held.resize(size_t(h - 5));
        held.insert(held.end(), other, other + 3);
        expect("no checkpoint fits: from token 0", ok && np == h - 2 && e.last_prefill().from == 0,
               "n_processed " + std::to_string(np) + " (expected " + std::to_string(h - 2) + ")");
        // E3: nothing new (n_tokens = 0): the last kept token processed again; its prompt end's checkpoint does not fit
        ok = ok && e.prefill(0, nullptr, 0, h - 2, np) == rt::run_status::ok;
        expect("nothing new", ok && np == h - 2 && e.last_prefill().from == 0,
               "n_processed " + std::to_string(np) + " (expected " + std::to_string(h - 2) + ")");
        // E4: steps, then a truncate back to the prompt (the bootstrap's step and one more: within the last step or a
        // restore at the prompt end, both from h - 2) and a next turn
        ok = ok && steps(2) && e.truncate(0, h - 2) == rt::run_status::ok;
        ok = ok && e.prefill(0, other, 2, h - 2, np) == rt::run_status::ok;
        held.insert(held.end(), other, other + 2);
        expect("next turn after steps", ok && np == 2 && e.last_prefill().from == h - 2,
               "n_processed " + std::to_string(np) + ", from " + std::to_string(e.last_prefill().from));
        // E5: a pure extension (the kept length is the processed length)
        ok = ok && e.prefill(0, other + 2, 1, h, np) == rt::run_status::ok;
        held.push_back(other[2]);
        expect("pure extension", ok && np == 1 && e.last_prefill().from == h,
               "n_processed " + std::to_string(np) + ", from " + std::to_string(e.last_prefill().from));
        phase("pure extension", 1, -1);
        // E6 (PB1): a truncate inside the last prompt chunk is not cut within it (its versions are half ends): the
        // next prefill restores 8,192's checkpoint
        ok = ok && e.truncate(0, 0) == rt::run_status::ok &&
             e.prefill(0, pool.data(), L9, 0, np) == rt::run_status::ok &&
             e.truncate(0, L9 - 1) == rt::run_status::ok &&
             e.prefill(0, pool.data() + L9 - 1, 1, L9 - 1, np) == rt::run_status::ok;
        held.assign(pool.begin(), pool.begin() + L9);
        expect("truncate inside the last prompt chunk", ok && np == L9 - 8192 && e.last_prefill().from == 8192,
               "n_processed " + std::to_string(np) + ", from " + std::to_string(e.last_prefill().from));
        phase("restore at 8,192", L9 - 8192, 8192);
        // E7 (PB1, BG1's seam): cancellation at a chunk end through the callback; the resend restores its checkpoint
        const rt::prefill_progress stop_8192 = [](int64_t, int64_t done, int64_t) {
            return done < 8192;
        };
        ok                      = ok && e.truncate(0, 0) == rt::run_status::ok;
        const rt::run_status cs = ok ? e.prefill(0, pool.data(), L9, 0, np, stop_8192) : rt::run_status::error;
        held.assign(pool.begin(), pool.begin() + 8192);
        expect("cancelled at 8,192", cs == rt::run_status::cancelled && np == 8192,
               std::string("status ") + (cs == rt::run_status::cancelled ? "cancelled" : "NOT cancelled") +
                   ", n_processed " + std::to_string(np));
        const std::string & pl = e.last_phase_line();
        v.put("engine phase line: cancelled",
              pl.find("decode-ready not reached") != std::string::npos && pl.find("; rows 0;") != std::string::npos &&
                  pl.find("restored checkpoint none") != std::string::npos,
              pl);
        ok = cs == rt::run_status::cancelled &&
             e.prefill(0, pool.data() + 8192, L9 - 8192, 8192, np) == rt::run_status::ok;
        held.assign(pool.begin(), pool.begin() + L9);
        expect("the resend after the cancel", ok && np == L9 - 8192 && e.last_prefill().from == 8192,
               "n_processed " + std::to_string(np) + ", from " + std::to_string(e.last_prefill().from));
        // E8 (PB1): a failed chunk: the engine reports the error and forgets every checkpoint; the next prefill
        // starts from token 0
        e.prefill_driver().prompt()->test_fail_next(0);
        const rt::run_status fs =
            e.truncate(0, 0) == rt::run_status::ok ? e.prefill(0, pool.data(), L9, 0, np) : rt::run_status::ok;
        const int32_t held_after = e.prefill_driver().book().count();
        ok = fs == rt::run_status::error && e.prefill(0, pool.data(), L9, 0, np) == rt::run_status::ok;
        expect("a failed chunk, then a fresh prefill", ok && held_after == 0 && e.last_prefill().from == 0,
               std::string("status ") + (fs == rt::run_status::error ? "error" : "NOT error") + ", " +
                   std::to_string(held_after) + " checkpoints held after it; next from " +
                   std::to_string(e.last_prefill().from));
    }
    // headroom after the reservation at 202,816 and 262,144 cells (zero weights; the engine as the server builds it)
    for (const int32_t big : { 202816, 262144 }) {
        auto          mb = std::make_unique<md::model>();
        std::string   line;
        bool          fit    = fnfl_load_dry(a, big, *mb);
        const int32_t dev[2] = { mb->device[0], mb->device[1] };
        if (fit) {
            rt::engine_config c;
            c.model_path = a.model_path;
            c.n_ctx_slot = big;
            c.draft      = dopt;
            c.eog        = { eos_tokenizer };
            rt::engine e;
            fit = e.init(std::move(mb), c, err);
            for (int32_t card = 0; card < 2 && fit; ++card) {
                size_t fr = 0, tot = 0;
                fit = cuda_ok(cudaSetDevice(dev[card]), "device") && cuda_ok(cudaMemGetInfo(&fr, &tot), "memory");
                line += (line.empty() ? "" : "; ") + std::string("card ") + std::to_string(card) + " free " +
                        std::to_string(fr >> 20) + " MiB of " + std::to_string(tot >> 20);
            }
            if (!fit && line.empty()) {
                line = "the engine did not fit: " + err;
            }
        } else {
            line = "the reservation did not fit";
        }
        if (big == 202816) {
            v.put("headroom at 202,816 cells", fit, line);
        } else {
            printf("prefill-test: headroom at 262,144 cells (reported): %s\n", line.c_str());
            if (v.log != nullptr) {
                fprintf(v.log, "headroom at 262,144 cells (reported): %s\n", line.c_str());
            }
        }
    }
    return v.all ? 0 : 1;
}

// ---- model (PB1 Amendment 1: one load; NC1 section 6 on TG1's pinned dumps, then the 20,000-token checks, then the
// deep stage) ----

constexpr int32_t V     = 2 * VC;  // 248,320
constexpr int32_t ref_n = 4096;

// CRC-32 (zlib's polynomial, as the manifests' crc32)
uint32_t crc32_of(const uint8_t * p, size_t n) {
    static uint32_t table[256] = {};
    if (table[1] == 0) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int32_t k = 0; k < 8; ++k) {
                c = (c & 1) != 0 ? 0xedb88320u ^ (c >> 1) : c >> 1;
            }
            table[i] = c;
        }
    }
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        c = table[(c ^ p[i]) & 0xff] ^ (c >> 8);
    }
    return c ^ 0xffffffffu;
}

// a gzip file decompressed into memory through gzip -dc (nothing written decompressed); exactly want bytes
bool read_gz(const std::string & path, std::vector<uint8_t> & out, size_t want) {
    FILE * f = popen(("gzip -dc '" + path + "'").c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    out.resize(want + 1);
    size_t got = 0, k = 0;
    while ((k = fread(out.data() + got, 1, out.size() - got, f)) > 0) {
        got += k;
        if (got == out.size()) {
            break;
        }
    }
    const int rc = pclose(f);
    out.resize(std::min(got, want));
    return rc == 0 && got == want;
}

// "key": "value" (the first after from)
std::string json_str(const std::string & js, const std::string & key, size_t from = 0) {
    const std::string k = "\"" + key + "\": \"";
    const size_t      p = js.find(k, from);
    if (p == std::string::npos) {
        return "";
    }
    const size_t q = js.find('"', p + k.size());
    return q == std::string::npos ? "" : js.substr(p + k.size(), q - p - k.size());
}

// TG1's pinned-engine dump (ref_dump3): F32 logits at the 512 positions p % 8 == 7 and the top two at all 4,096
struct pinned {
    std::string          dir, mode;
    std::vector<uint8_t> logits, top;

    const float * row(int64_t p) const { return reinterpret_cast<const float *>(logits.data()) + size_t(p / 8) * V; }

    int32_t top1(int64_t p) const {
        uint32_t x = 0;
        std::memcpy(&x, top.data() + size_t(p) * 16, 4);
        return static_cast<int32_t>(x);
    }

    double margin(int64_t p) const {
        float a = 0, b = 0;
        std::memcpy(&a, top.data() + size_t(p) * 16 + 4, 4);
        std::memcpy(&b, top.data() + size_t(p) * 16 + 12, 4);
        return double(a) - double(b);
    }
};

// the manifest must name this model file and these IDs (Amendment 1), 4,096 tokens; each file's CRC its manifest's
bool load_pinned(const std::string & dir, const fnfl_target_args & a, pinned & r, verdicts & v) {
    r.dir = dir;
    std::ifstream     f(dir + "/manifest.json");
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string js    = ss.str();
    r.mode                  = json_str(js, "mode");
    const std::string model = json_str(js, "model"), ids = json_str(js, "ids");
    const bool        names =
        !js.empty() && model == a.model_path && ids == a.ids && js.find("\"tokens\": 4096") != std::string::npos;
    const std::string crc_l     = json_str(js, "crc32", js.find("\"logits\": {"));
    const std::string crc_t     = json_str(js, "crc32", js.find("\"top\": {"));
    bool              ok        = names && read_gz(dir + "/logits.bin.gz", r.logits, size_t(ref_n / 8) * V * 4) &&
                                  read_gz(dir + "/top.bin.gz", r.top, size_t(ref_n) * 16);
    char              got_l[16] = "-", got_t[16] = "-";
    if (ok) {
        snprintf(got_l, sizeof(got_l), "%08x", crc32_of(r.logits.data(), r.logits.size()));
        snprintf(got_t, sizeof(got_t), "%08x", crc32_of(r.top.data(), r.top.size()));
        ok = crc_l == got_l && crc_t == got_t;
    }
    v.put("pinned reference " + dir, ok,
          "manifest mode " + r.mode + ", model " + (model == a.model_path ? "this file" : "OTHER: " + model) +
              ", IDs " + (ids == a.ids ? "these" : "OTHER: " + ids) + "; logits CRC " + got_l + " (manifest " + crc_l +
              "), top CRC " + got_t + " (manifest " + crc_t + ")");
    return ok;
}

// KL(p || q) over the vocabulary in FP64 from F32 logits (ref_spread3.py's method)
double kl_full(const float * p, const float * q) {
    double mp = -HUGE_VAL, mq = -HUGE_VAL;
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
    double       kl = 0.0;
    for (int32_t i = 0; i < V; ++i) {
        const double lp = double(p[i]) - mp - lsp;
        kl += std::exp(lp) * (lp - (double(q[i]) - mq - lsq));
    }
    return kl;
}

double pct(std::vector<double> x, double q) {
    if (x.empty()) {
        return 0.0;
    }
    std::sort(x.begin(), x.end());
    const double at = q / 100.0 * double(x.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(at));
    const size_t hi = std::min(lo + 1, x.size() - 1);
    return x[lo] + (x[hi] - x[lo]) * (at - double(lo));
}

int32_t argmax_of(const float * x) {
    int32_t b = 0;
    for (int32_t i = 1; i < V; ++i) {
        b = x[i] > x[b] ? i : b;  // the lower ID on a tie
    }
    return b;
}

// both cards' logits row into V values (card 0's vocabulary half first)
bool read_full(rt::target & t, md::model & m, int32_t row, float * out) {
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaMemcpy(out + size_t(c) * VC, t.card(c).logits + size_t(row) * VC, size_t(VC) * 4,
                                cudaMemcpyDeviceToHost),
                     "logits")) {
            return false;
        }
    }
    return true;
}

// FNFL's outputs after a prefill to L: the handoff's retained row (position L - 1), then the fixture's tokens
// teacher-forced to end in forced rounds cycling B 1, 2, 3, 4 (as model-test runs them), each followed by the draft's
// catch-up; the top-1 at every position in [L, end) and the full logits at its p % 8 == 7 positions
struct fnfl_cont {
    int64_t              L = 0, end = 0;
    std::vector<int32_t> top;
    std::vector<float>   rows;
    std::vector<int64_t> row_pos;
    std::vector<float>   head;
    int32_t              head_top = -1;

    const float * row_at(int64_t p) const {
        for (size_t i = 0; i < row_pos.size(); ++i) {
            if (row_pos[i] == p) {
                return rows.data() + i * V;
            }
        }
        return nullptr;
    }
};

bool continue_forced(rt::target &                 t,
                     md::model &                  m,
                     const std::vector<int32_t> & ids,
                     int64_t                      end,
                     fnfl_cont &                  o,
                     std::string &                err) {
    o                    = fnfl_cont();
    const rt::ledger & l = t.slot();
    o.L                  = l.n;
    o.end                = end;
    if (!l.retained || l.u != 0 || end <= o.L || end > static_cast<int64_t>(ids.size())) {
        err = "the continuation needs the handoff state and tokens to " + std::to_string(end);
        return false;
    }
    o.head.resize(V);
    if (!read_full(t, m, l.retained_row, o.head.data())) {
        return false;
    }
    o.head_top = argmax_of(o.head.data());
    if (!rt::extend_ledger(t, ids.data() + o.L, static_cast<int32_t>(end - o.L), err)) {
        return false;
    }
    std::vector<float> row(V);
    int64_t            pos = o.L;
    for (size_t ci = 0; pos < end; ++ci) {
        const int32_t    B = static_cast<int32_t>(std::min<int64_t>(int64_t(ci % 4) + 1, end - pos));
        rt::round_result r;
        if (!t.forced(B, r, err) || (t.draft_on() && !t.draft_catchup(B, r, err))) {
            return false;
        }
        for (int32_t b = 0; b < B; ++b) {
            if (!read_full(t, m, b, row.data())) {
                return false;
            }
            o.top.push_back(argmax_of(row.data()));
            if ((pos + b) % 8 == 7) {
                o.row_pos.push_back(pos + b);
                o.rows.insert(o.rows.end(), row.begin(), row.end());
            }
        }
        pos += B;
    }
    return true;
}

// NC1 section 5 on a case's decoded positions [L, 4,096): KL(pinned decode || FNFL) p50, p99 and max over p % 8 == 7,
// each <= 2x the pinned engine's own KL(decode || prompt) there; top-1 differences from pinned decode over every
// decoded position <= 2x the pinned modes' own there. KL against pinned prompt mode and the handoff's own row
// (position L - 1, the prompt head's) are diagnostics.
bool nc1_case(const std::string & name,
              const pinned &      dec,
              const pinned &      pro,
              const fnfl_cont &   f,
              FILE *              tsv,
              verdicts &          v) {
    std::vector<double> k_f, k_p, k_s;
    int32_t             top_s = 0;
    std::string         diffs;
    int32_t             top_f = 0;
    for (int64_t p = f.L; p < ref_n; ++p) {
        top_s += dec.top1(p) != pro.top1(p) ? 1 : 0;
        if (f.top[size_t(p - f.L)] != dec.top1(p)) {
            ++top_f;
            char b[128];
            snprintf(b, sizeof(b), "%s%lld (FNFL %d, decode %d, its margin %.4g)", diffs.empty() ? "" : "; ",
                     static_cast<long long>(p), f.top[size_t(p - f.L)], dec.top1(p), dec.margin(p));
            diffs += b;
        }
        if (p % 8 == 7) {
            const float * q = f.row_at(p);
            if (q == nullptr) {
                v.put("NC1 " + name, false, "no FNFL row at " + std::to_string(p));
                return false;
            }
            k_f.push_back(kl_full(dec.row(p), q));
            k_p.push_back(kl_full(pro.row(p), q));
            k_s.push_back(kl_full(dec.row(p), pro.row(p)));
            if (tsv != nullptr) {
                fprintf(tsv, "%s\t%lld\t%.9g\t%.9g\t%.9g\n", name.c_str(), static_cast<long long>(p), k_f.back(),
                        k_p.back(), k_s.back());
            }
        }
    }
    const double f50 = pct(k_f, 50), f99 = pct(k_f, 99), fmx = pct(k_f, 100);
    const double s50 = pct(k_s, 50), s99 = pct(k_s, 99), smx = pct(k_s, 100);
    const bool   kl_ok  = f50 <= 2 * s50 && f99 <= 2 * s99 && fmx <= 2 * smx;
    const bool   top_ok = top_f <= 2 * top_s;
    char         b[640];
    snprintf(b, sizeof(b),
             "positions %lld .. 4,095 (%zu at p %% 8 == 7): KL(decode || FNFL) p50 %.4g p99 %.4g max %.4g against 2x "
             "the pinned spread %.4g / %.4g / %.4g %s; top-1 differs at %d (limit 2 x %d) %s; KL(prompt || FNFL) p50 "
             "%.4g p99 %.4g max %.4g (diagnostic)",
             static_cast<long long>(f.L), k_f.size(), f50, f99, fmx, 2 * s50, 2 * s99, 2 * smx, kl_ok ? "ok" : "OVER",
             top_f, top_s, top_ok ? "ok" : "OVER", pct(k_p, 50), pct(k_p, 99), pct(k_p, 100));
    v.put("NC1 " + name, kl_ok && top_ok, std::string(b) + (diffs.empty() ? "" : "; differences: " + diffs));
    // the handoff's own row (the prompt head), a diagnostic
    const int64_t hp = f.L - 1;
    std::string   hl = "position " + std::to_string(hp) + ": FNFL top-1 " + std::to_string(f.head_top) + ", decode " +
                       std::to_string(dec.top1(hp)) + ", prompt " + std::to_string(pro.top1(hp));
    if (hp % 8 == 7) {
        char k[160];
        snprintf(k, sizeof(k),
                 "; KL(decode || FNFL) %.4g, KL(prompt || FNFL) %.4g, the pinned KL(decode || prompt) %.4g",
                 kl_full(dec.row(hp), f.head.data()), kl_full(pro.row(hp), f.head.data()),
                 kl_full(dec.row(hp), pro.row(hp)));
        hl += k;
    }
    printf("prefill-test: NC1 %s: the handoff row (diagnostic): %s\n", name.c_str(), hl.c_str());
    if (v.log != nullptr) {
        fprintf(v.log, "NC1 %s: the handoff row (diagnostic): %s\n", name.c_str(), hl.c_str());
    }
    return kl_ok && top_ok;
}

// FNFL against FNFL on a common continuation: KL(a || b) over the shared p % 8 == 7 rows and top-1 differences
std::string mutual(const fnfl_cont & a, const fnfl_cont & b) {
    std::vector<double> k;
    int32_t             d = 0, n = 0;
    for (int64_t p = std::max(a.L, b.L); p < std::min(a.end, b.end); ++p) {
        d += a.top[size_t(p - a.L)] != b.top[size_t(p - b.L)] ? 1 : 0;
        ++n;
        const float * x = a.row_at(p);
        const float * y = b.row_at(p);
        if (x != nullptr && y != nullptr) {
            k.push_back(kl_full(x, y));
        }
    }
    char s[256];
    snprintf(s, sizeof(s), "KL p50 %.4g p99 %.4g max %.4g over %zu rows; top-1 differs at %d of %d positions",
             pct(k, 50), pct(k, 99), pct(k, 100), k.size(), d, n);
    return s;
}

// the draft's caches and retained carry on both cards (cells < draft_n)
struct draft_arrays {
    int64_t               n = 0;
    std::vector<uint16_t> k[2], v[2];
    std::vector<float>    carry[2];
    // SP2 (the sparse form only): the draft index's raw keys of cells < n and the summaries of its complete blocks
    int64_t               index_n = -1;
    std::vector<uint16_t> ik[2];
    std::vector<float>    sm[2];
};

bool read_draft(rt::target & t, md::model & m, draft_arrays & d) {
    d.n = t.draft_n();
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") || d.n < 0 ||
            !get(d.k[c], t.draft(c).k_cache, size_t(d.n) * 256) ||
            !get(d.v[c], t.draft(c).v_cache, size_t(d.n) * 256) ||
            !get(d.carry[c], t.draft(c).carry_prev, size_t(4) * 1280)) {
            return false;
        }
        if (t.draft_sparse()) {
            const int32_t nb = fn::qsa::blocks_complete(static_cast<int32_t>(d.n));
            if (!get(d.ik[c], static_cast<const uint16_t *>(t.draft(c).idx_keys), size_t(d.n) * fn::qsa::head_dim) ||
                !get(d.sm[c], static_cast<const float *>(t.draft(c).idx_summ), fn::qsa::summary_bytes(nb) / 4)) {
                return false;
            }
        }
    }
    d.index_n = t.draft_sparse() ? t.index_n() : -1;
    return true;
}

// per cell, max |a - b| / max |b| over its 256 values; p50, p99 and max over the cells and both cards
std::string cache_error(const draft_arrays & a, const draft_arrays & b, bool & finite) {
    std::vector<double> ek, ev;
    finite = true;
    for (int32_t c = 0; c < 2; ++c) {
        for (int64_t i = 0; i < std::min(a.n, b.n); ++i) {
            for (int32_t w = 0; w < 2; ++w) {
                const uint16_t * x  = (w == 0 ? a.k[c] : a.v[c]).data() + size_t(i) * 256;
                const uint16_t * y  = (w == 0 ? b.k[c] : b.v[c]).data() + size_t(i) * 256;
                double           dm = 0, ym = 0;
                for (int32_t e = 0; e < 256; ++e) {
                    const double fx = fn::quant::f16_to_f32(x[e]), fy = fn::quant::f16_to_f32(y[e]);
                    finite = finite && std::isfinite(fx);
                    dm     = std::max(dm, std::fabs(fx - fy));
                    ym     = std::max(ym, std::fabs(fy));
                }
                (w == 0 ? ek : ev).push_back(ym > 0 ? dm / ym : dm);
            }
        }
    }
    double ec = 0;
    for (int32_t c = 0; c < 2; ++c) {
        double dm = 0, ym = 0;
        for (size_t i = 0; i < a.carry[c].size(); ++i) {
            dm = std::max(dm, std::fabs(double(a.carry[c][i]) - b.carry[c][i]));
            ym = std::max(ym, std::fabs(double(b.carry[c][i])));
        }
        ec = std::max(ec, ym > 0 ? dm / ym : dm);
    }
    char s[320];
    snprintf(s, sizeof(s),
             "K per-cell relative p50 %.3g p99 %.3g max %.3g; V p50 %.3g p99 %.3g max %.3g; retained carry %.3g",
             pct(ek, 50), pct(ek, 99), pct(ek, 100), pct(ev, 50), pct(ev, 99), pct(ev, 100), ec);
    return s;
}

// the deep stage (Amendment 1): the engine on this process's model after a complete slot reset (the target and the
// prefiller above destroyed); the prompt bulk-prefilled, then the full speculative loop once with spec run's request
// (fnfl_spec_request). restore > 0: first a request of the prompt's first `restore` tokens, then the prompt with that
// many kept, so the engine restores its prompt-end checkpoint and appends. Reports the acceptance (sum a / sum d over
// the speculative steps), the phase line and the runtime's FNFL request: summary line; the prefill time is incidental.
// PB3's closure cycle: with --append, the same engine then runs the append point (--append-restore, the prefix
// request untimed, its own full reset by truncating to 0), after the fresh point, each with its own lines and tokens.
bool deep_stage(std::unique_ptr<md::model> m,
                const fnfl_target_args &   a,
                const rt::draft_options &  dopt,
                verdicts &                 v) {
    const fnfl_target_args::run_setting st =
        a.run_settings.empty() ? fnfl_target_args::run_setting() : a.run_settings.front();
    rt::engine_config c;
    c.model_path = a.model_path;
    c.n_ctx_slot = a.n_cells;
    c.draft = dopt;
    c.eog = {eos_tokenizer};
    c.timeout_s  = 60.0;
    rt::engine e;
    std::string err;
    if (!e.init(std::move(m), c, err)) {
        v.put("deep stage", false, "engine: " + err);
        return false;
    }
    rt::request_options o;
    o.sampling   = st.sampling;
    o.max_tokens = st.max_tokens;
    char what[256];
    snprintf(what, sizeof(what), "T %.2f top-k %d top-p %.2f min-p %.2f seed %llu, at most %d tokens",
             double(st.sampling.temp), st.sampling.top_k, double(st.sampling.top_p), double(st.sampling.min_p),
             static_cast<unsigned long long>(st.sampling.seed), st.max_tokens);
    // one point: its prompt, restore point, verdict name and token file
    const auto point = [&](const std::string & ids, int32_t restore, const std::string & name, const char * tok_file) {
        std::vector<int32_t> P;
        if (!read_ids(ids, P)) {
            v.put(name, false, "cannot read " + ids);
            return false;
        }
        if (restore < 0 || restore >= static_cast<int32_t>(P.size()) ||
            static_cast<int64_t>(P.size()) + st.max_tokens > a.n_cells) {
            v.put(name, false, "the prompt, its restore point or its tokens do not fit the cells");
            return false;
        }
        printf("prefill-test: %s: %s (%zu tokens), %s%s\n", name.c_str(), ids.c_str(), P.size(), what,
               restore > 0 ? (", restoring at " + std::to_string(restore)).c_str() : "");
        bool ok = true;
        fnfl_request_out ro;
        std::string pre;
        if (restore > 0) {
            ok = fnfl_spec_request(e, std::vector<int32_t>(P.begin(), P.begin() + restore), 0, o, ro);
            pre = e.last_phase_line() + " | " + e.last_summary_line() + " || ";
        }
        ok = ok && fnfl_spec_request(e, P, restore, o, ro);
        if (!ok) {
            v.put(name, false, e.error());
            return false;
        }
        int64_t sa = 0, sd = 0;
        for (const rt::step_info & x : e.last_steps()) {
            if (x.kind == 2) {
                sa += x.a;
                sd += x.d;
            }
        }
        const std::string why =
            check_phase(e.last_phase_line(), static_cast<int64_t>(P.size()) - restore, restore > 0 ? restore : -1);
        char b[256];
        snprintf(b, sizeof(b),
                 "%zu tokens; acceptance %lld / %lld = %.4f (sum a / sum d over the speculative steps); "
                 "prefill %.3f s (incidental)",
                 ro.toks.size(), static_cast<long long>(sa), static_cast<long long>(sd),
                 sd > 0 ? double(sa) / double(sd) : 0.0, ro.pre_s);
        v.put(name + " completes", static_cast<int32_t>(ro.toks.size()) >= 1, b);
        v.put(name + " phase line", why.empty(), why.empty() ? e.last_phase_line() : why);
        printf("prefill-test: %s: %s%s\n", name.c_str(), pre.c_str(), e.last_summary_line().c_str());
        if (v.log != nullptr) {
            fprintf(v.log, "%s: %s%s\n", name.c_str(), pre.c_str(), e.last_summary_line().c_str());
        }
        if (!a.out_dir.empty()) {
            FILE * f = fopen((a.out_dir + "/" + tok_file).c_str(), "w");
            if (f != nullptr) {
                for (const int32_t t : ro.toks) {
                    fprintf(f, "%d\n", t);
                }
                fclose(f);
            }
            // the timed request's chunks as the prefiller recorded them (host clock, no added synchronization):
            // the depth each starts at, its rows, its rounds and checkpoint, and its wait for the next gather
            f = fopen((a.out_dir + "/" + name.substr(0, name.find(' ')) + "-chunks.tsv").c_str(), "w");
            if (f != nullptr) {
                fprintf(f, "start\trows\tcompute_s\tstall_s\tgather_s\tcheckpoint\n");
                for (const rt::pf_chunk_stats & x : e.last_prefill().chunks) {
                    fprintf(f, "%lld\t%d\t%.6f\t%.6f\t%.6f\t%d\n", static_cast<long long>(x.c.start), x.c.rows,
                            x.compute_s, x.stall_s, x.gather_s, x.checkpoint);
                }
                fclose(f);
            }
        }
        return true;
    };
    const bool ok = point(a.deep_ids, a.deep_restore, "deep stage", "deep-tokens.txt");
    return a.append_ids.empty() ? ok : point(a.append_ids, a.append_restore, "append stage", "append-tokens.txt") && ok;
}

// plumb (mode model-plumb, no load): the same flow on zero weights (model-dry's load) before the one real load: the
// references read and checked, every case, dump and the deep stage run; the NC1 verdicts are printed but not counted
// (zero weights cannot meet them), every other check counts
int model_mode(const fnfl_target_args & a, const std::vector<int32_t> & pool, verdicts & v, bool plumb) {
    constexpr int32_t L20 = 20000, n_cont = 512;
    if (pool.size() < size_t(L20) + n_cont || a.ref_dir.empty() || a.ref_prompt.empty()) {
        fprintf(stderr, "prefill-test: model needs %d tokens, --ref-decode and --ref-prompt\n", L20 + n_cont);
        return 1;
    }
    rt::draft_options dopt;
    if (!fnfl_draft_settings(a, true, "prefill-test", dopt)) {
        return 1;
    }
    // the references first (no load if they do not match)
    pinned dec, pro;
    if (!load_pinned(a.ref_dir, a, dec, v) || !load_pinned(a.ref_prompt, a, pro, v) || dec.mode != "decode" ||
        pro.mode != "prompt") {
        fprintf(stderr, "prefill-test: the pinned references do not match this model, these IDs and their modes\n");
        return 1;
    }
    auto m = std::make_unique<md::model>();
    if (!(plumb ? fnfl_load_dry(a, a.n_cells, *m) : fnfl_load(a, a.n_cells, *m, "prefill-test"))) {
        return 1;
    }
    printf("prefill-test: %s, %d cells, the draft on, T = 0; the fixture %s\n",
           plumb ? "model-plumb (ZERO WEIGHTS: the flow only; NC1 not counted)" : "model", a.n_cells, a.ids.c_str());
    // PB3 Amendment 3: NC1 section 6's scores are diagnostic (printed, not counted, the deep stage not gated on them);
    // the functional checks below count
    verdicts   scratch;
    verdicts & vn  = scratch;
    bool       nc  = true;
    FILE *     tsv = a.out_dir.empty() ? nullptr : fopen((a.out_dir + "/nc1-kl.tsv").c_str(), "w");
    if (tsv != nullptr) {
        fprintf(tsv, "case\tpos\tkl_decode_fnfl\tkl_prompt_fnfl\tkl_decode_prompt\n");
    }
    {
        rt::target_options o;
        o.model_path    = a.model_path;
        o.sample        = true;
        o.sampling.temp = 0.0f;
        o.draft         = dopt;
        o.timeout_s     = 60.0;
        rt::target    t;
        rt::prefiller pf;
        std::string   err;
        if (!t.init(*m, o, err) || !pf.init(t, *m, err)) {
            fprintf(stderr, "prefill-test: init: %s\n", err.c_str());
            return 1;
        }
        const auto prefix = [&](int64_t n) {
            return std::vector<int32_t>(pool.begin(), pool.begin() + n);
        };
        int64_t           from = -1;
        rt::pf_run_stats  st, s0;
        // NC1 section 6: three cases, each bulk-prefilled, then teacher-forced to 4,095
        fnfl_cont         f2, f3, fa;
        bool              ok = pf.begin(t, prefix(2048), 0, from, err) && pf.run(t, &st, err);
        const std::string c2 = ok ? chunk_list(st) : "";
        ok                   = ok && continue_forced(t, *m, pool, ref_n, f2, err);
        printf("prefill-test: fresh at 2,048: chunks %s (halves %d + %d)\n", c2.c_str(),
               st.chunks.empty() ? 0 : st.chunks[0].c.half[0], st.chunks.empty() ? 0 : st.chunks[0].c.half[1]);
        nc                   = ok && nc1_case("fresh at 2,048", dec, pro, f2, tsv, vn) && nc;
        ok                   = ok && pf.begin(t, prefix(3584), 0, from, err) && pf.run(t, &st, err);
        const std::string c3 = ok ? chunk_list(st) : "";
        ok                   = ok && continue_forced(t, *m, pool, ref_n, f3, err);
        printf("prefill-test: fresh at 3,584: chunks %s (896 QSA blocks)\n", c3.c_str());
        nc = ok && nc1_case("fresh at 3,584", dec, pro, f3, tsv, vn) && nc;
        // append: 2,048, its prompt-end checkpoint restored, then to 3,584
        ok = ok && pf.begin(t, prefix(2048), 0, from, err) && pf.run(t, &s0, err) &&
             continue_forced(t, *m, pool, 2100, fa, err) && pf.begin(t, prefix(3584), 2048, from, err);
        const int64_t from_a = from;
        ok                   = ok && pf.run(t, &st, err);
        const std::string ca =
            ok ? chunk_list(s0) + " | restored at " + std::to_string(from_a) + ": " + chunk_list(st) : "";
        ok = ok && from_a == 2048 && continue_forced(t, *m, pool, ref_n, fa, err);
        printf("prefill-test: append: chunks %s\n", ca.c_str());
        nc = ok && nc1_case("append 2,048 -> 3,584", dec, pro, fa, tsv, vn) && nc;
        if (ok) {
            const std::string mu = mutual(f3, fa);
            printf("prefill-test: append against fresh at 3,584 (diagnostic): %s\n", mu.c_str());
            if (v.log != nullptr) {
                fprintf(v.log, "append against fresh at 3,584 (diagnostic): %s\n", mu.c_str());
            }
        }
        if (tsv != nullptr) {
            fclose(tsv);
        }
        if (!ok || !nc) {
            if (!err.empty()) {
                printf("prefill-test: NC1: %s\n", err.c_str());
            }
        }
        printf("prefill-test: NC1 section 6 (diagnostic, PB3 Amendment 3): %s\n",
               plumb ? "zero weights, not judged" :
               nc    ? "every case within the old thresholds" :
                       "a case outside the old thresholds (reported, not gating)");
        if (v.log != nullptr) {
            fprintf(v.log, "NC1 section 6 (diagnostic, PB3 Amendment 3): %s\n", nc ? "within" : "outside");
        }
        if (!ok) {
            v.put("NC1 section 6 cases run", false, "a case did not run");
            return 1;
        }
        // the 20,000-token checks (FNFL-internal)
        const std::vector<int32_t> P20 = prefix(L20);
        md::hc_dump_spec           dump;
        dump.layers     = { 1, 24, 47 };
        dump.chunks = {0}; // the full chunk 0+8192 (PB3)
        dump.row_stride = 16;
        dump.dir        = a.out_dir + (plumb ? "/hc-inputs" : "/../hc-inputs");
        std::error_code ec;
        std::filesystem::create_directories(dump.dir, ec);
        if (ec) {
            v.put("HC dump", false, "cannot create " + dump.dir);
            return 1;
        }
        pf.prompt()->set_hc_dump(dump);
        std::vector<uint8_t>       bytes8192, restored;
        const rt::prefill_progress at_8192 = [&](int64_t, int64_t done, int64_t) {
            return done != 8192 || (sync_cards(t, *m) && slot_bytes(t, *m, t.slot().cur, bytes8192));
        };
        state_hash hb, hb2, hb3, hb4, hr, hr2, hc;
        pf.prompt()->clear_site_counts();
        ok                       = pf.begin(t, P20, 0, from, err) && pf.run(t, &st, err) && snapshot(t, *m, hb);
        const std::string c20    = ok ? chunk_list(st) : "";
        bool              active = false;
        const std::string sites  = site_line(pf, active);
        pf.prompt()->set_hc_dump(md::hc_dump_spec());
        v.put("20,000: bulk prefill",
              ok && c20 == "0+8192* 8192+8192* 16384+3616*" && handoff(t, L20).empty() && active,
              ok ? "chunks " + c20 + "; call sites: " + sites : err);
        fnfl_cont    cb, cf;
        draft_arrays db, dfz;
        ok = ok && continue_forced(t, *m, pool, L20 + n_cont, cb, err);
        // determinism: two runs without the dump (the second capturing the live state at 8,192); the dump run
        // against them is reported (the dump synchronizes card 0 between kernels), then a run after every arena byte
        // was poisoned with NaN (a run must not read what an earlier one left)
        ok = ok && pf.begin(t, P20, 0, from, err) && pf.run(t, &st, err) && snapshot(t, *m, hb2);
        ok = ok && pf.begin(t, P20, 0, from, err) && pf.run(t, &st, err, at_8192) && snapshot(t, *m, hb3);
        v.put("20,000: determinism", ok && compare(hb2, hb3) == "equal",
              ok ? "two bulk prefills: " + compare(hb2, hb3) +
                       "; the HC dump's run against them (diagnostic): " + compare(hb, hb2) :
                   err);
        ok = ok && pf.begin(t, P20, 0, from, err) && pf.prompt()->test_poison(err) && pf.run(t, &st, err) &&
             snapshot(t, *m, hb4);
        v.put("20,000: a run after the arena poisoned", ok && compare(hb2, hb4) == "equal",
              ok ? "every arena byte 0xFF before the run: " + compare(hb2, hb4) : err);
        ok                   = ok && read_draft(t, *m, db);
        // restore and re-prefill: from 8,192 (n_reuse 12,000; byte-exact restore) and from 16,384 (n_reuse 17,000)
        ok                   = ok && pf.begin(t, P20, 12000, from, err);
        const int64_t from_8 = from;
        ok = ok && sync_cards(t, *m) && slot_bytes(t, *m, t.slot().cur, restored) && pf.run(t, &st, err) &&
             snapshot(t, *m, hr);
        const bool exact      = ok && !bytes8192.empty() && restored == bytes8192;
        ok                    = ok && pf.begin(t, P20, 17000, from, err);
        const int64_t from_16 = from;
        ok                    = ok && pf.run(t, &st, err) && snapshot(t, *m, hr2);
        v.put("20,000: restore and re-prefill against uninterrupted",
              ok && from_8 == 8192 && from_16 == 16384 && exact && compare(hb2, hr) == "equal" &&
                  compare(hb2, hr2) == "equal",
              ok ? "from " + std::to_string(from_8) + ": " + compare(hb2, hr) + " (the checkpoint's " +
                       std::to_string(bytes8192.size()) + " bytes " + (exact ? "equal" : "DIFFERENT") +
                       " to the live state at 8,192); from " + std::to_string(from_16) + ": " + compare(hb2, hr2) :
                   err);
        // cancellation at 16,384 and the resend from its checkpoint
        const rt::prefill_progress stop_16384 = [](int64_t, int64_t done, int64_t) {
            return done < 16384;
        };
        ok = ok && pf.begin(t, P20, 0, from, err);
        std::string serr;
        const bool  stopped = ok && !pf.run(t, &st, serr, stop_16384) && st.stopped == 16384 && t.slot().n == 16384 &&
                              t.draft_n() == 16384;
        ok = ok && stopped && pf.begin(t, P20, 16384, from, err) && pf.run(t, &st, err) && snapshot(t, *m, hc);
        v.put("20,000: cancellation at 16,384 and the resend", ok && from == 16384 && compare(hb2, hc) == "equal",
              ok ? "stopped at 16,384 (" + serr + "); resend from " + std::to_string(from) + ": " + compare(hb2, hc) :
                   err);
        // the forced reference: the draft's caches and the 512-token continuation
        pf.set_bulk(false);
        ok = ok && pf.begin(t, P20, 0, from, err) && pf.run(t, &st, err) && handoff(t, L20).empty() &&
             read_draft(t, *m, dfz) && continue_forced(t, *m, pool, L20 + n_cont, cf, err);
        pf.set_bulk(true);
        bool              finite = false;
        const std::string ce     = ok ? cache_error(db, dfz, finite) : "";
        v.put("20,000: draft caches against the forced catch-up", ok && db.n == L20 && dfz.n == L20 && finite,
              ok ? "validity lengths " + std::to_string(db.n) + " / " + std::to_string(dfz.n) +
                       ", pending 0 at both handoffs; " + ce +
                       " (the draft's inputs differ by the target paths' own difference; the kernels' bounds are "
                       "dcp-test's, pd-test's and hcp-test's)" :
                   err);
        const std::string bf = ok ? mutual(cf, cb) : "";
        printf("prefill-test: 20,000: bulk against forced on positions 20,000 .. 20,511 (diagnostic): %s\n",
               bf.c_str());
        if (v.log != nullptr) {
            fprintf(v.log, "20,000: bulk against forced on positions 20,000 .. 20,511 (diagnostic): %s\n", bf.c_str());
        }
        // the HC dump's manifest (PB2)
        if (ok) {
            FILE * mf = fopen((dump.dir + "/manifest.tsv").c_str(), "w");
            ok        = mf != nullptr;
            if (ok) {
                fprintf(mf,
                        "# PB1 HC inputs for PB2: card 0, F32 little-endian, row-major; every 16th row of each half "
                        "(first row the half's first); prompt %s first 20,000 tokens, bulk path, %s\n",
                        a.ids.c_str(), a.model_path.c_str());
                fprintf(mf, "file\tchunk\tchunk_start\tlayer\tcall\thalf\tfirst_pos\tstride\trows\twidth\twhat\n");
                const std::vector<rt::pf_chunk> plan = rt::plan_chunks(0, L20);
                for (const int64_t ci : dump.chunks) {
                    const rt::pf_chunk & c = plan[size_t(ci)];
                    for (const int32_t l : dump.layers) {
                        for (const char * call : { "attn", "ffn" }) {
                            for (int32_t h = 0; h < 2 && c.half[h] > 0; ++h) {
                                const int32_t     rows = (c.half[h] + 15) / 16;
                                const int64_t     p0   = c.start + (h == 0 ? 0 : c.half[0]);
                                const std::string tag  = "c" + std::to_string(ci) + "-L" + std::to_string(l) + "-" +
                                                         call + "-h" + std::to_string(h);

                                const struct {
                                    const char * suf;
                                    int32_t      width;
                                    const char * what;
                                } files[3] = {
                                    { "-x.f32",     4 * 2560, "the HC call's input streams [4][2560]" },
                                    { "-low.f32",   320,      "its coefficients after the mid step"   },
                                    { "-mixed.f32", 2560,     "its mixed output"                      }
                                };

                                for (const auto & fl : files) {
                                    fprintf(mf, "%s%s\t%lld\t%lld\t%d\t%s\t%d\t%lld\t16\t%d\t%d\t%s\n", tag.c_str(),
                                            fl.suf, static_cast<long long>(ci), static_cast<long long>(c.start), l,
                                            call, h, static_cast<long long>(p0), rows, fl.width, fl.what);
                                }
                            }
                        }
                    }
                }
                fclose(mf);
            }
            v.put("HC dump for PB2", ok, dump.dir + "/manifest.tsv");
        }
        if (!ok && !err.empty()) {
            printf("prefill-test: 20,000: %s\n", err.c_str());
        }
    }
    // the deep stage, only after every check passed, after a complete slot reset (the target and prefiller are gone)
    if (!v.all) {
        printf("prefill-test: a check failed: the deep stage is not run\n");
        return 1;
    }
    if (a.deep_ids.empty()) {
        return 0;
    }
    return deep_stage(std::move(m), a, dopt, v) && v.all ? 0 : 1;
}

// ---- vcheck (the PB1 review's V check, no model file): the draft's catch-up by the forced rounds
// (target::draft_catchup, B cycling 1 .. 4) and by the bulk span (prompt_path's draft halves) on identical inputs ----
// A dry load (zero weights) whose draft attention inputs are then given valid random values on both cards: eh_proj,
// attn_k, attn_v (Q8_0, packed by the target's init), attn_k_norm, enorm, hnorm and the attention HC's norm (F32),
// its Q8_0 down, inject and up (hc.h's planar layouts), and token_embd's rows 0 .. 4,095 (IQ3_S). The prompt is 8,736
// random tokens below 4,096 (PB3: chunks 8,192 in halves of 4,096, and 544, so position 8,192's predecessor crosses a
// chunk boundary and position 4,096's a half boundary). Both runs
// read the same hidden rows H[p] [4][2560] (random, with per-stream scales and outlier features): the forced run's
// target carry rows overwritten after each target round, the bulk run's final streams overwritten by
// prompt_path::test_hidden before each draft half. Both start from a zero retained carry and the draft's K and V cells
// set to NaN. FP64 references on sampled positions: R_b with the bulk span's declared roundings (FP16 [e | h] operands,
// FP16 mixed row; since SP2 Amendment 1 item 7 also the FP16 eh_proj, wk and wv that PB3 candidate 5 (79c2f42) gave the
// bulk span: pd_convert's decoded weight rounded once to FP16, fn/kernels/dense/dense_prompt.h, held bitwise by
// pd-test) and R_f with the forced path's (act_ref's codes of [e | h] and of the mixed row; the decoded weights).
// Bounds, per cell max |x - y| / max |y| over its 256 values, set before the run (the PB1 Result's review follow-up;
// SP2 item 7 corrects R_b and keeps the bound):
//   bulk against R_b <= 2e-3 (one FP16 step of the cell's largest value, 9.8e-4, for the final rounding, plus
//     dcp-test's one-FP16-step operands, pd-test's 1e-5 and hcp-test's split 1e-5 through the products);
//   forced against R_f <= 5e-3 (the same final step, draft-kernel-test's 1e-6 input and 1e-5 eh_proj, hc2-test's 1e-5,
//     plus act_ref code flips where the kernel's F32 value and the FP64 one straddle a rounding boundary);
//   bulk against forced over all 8,736 cells of both cards: p50, p99 and max each <= 2 x the same statistic of
//     R_b against R_f on the sampled cells + 4e-3; the per-cell V ratio ||V_bulk|| / ||V_forced|| with median in
//     [0.998, 1.002] and p1, p99 in [0.99, 1.01] (zero-mean quantization differences move it by about 1e-3);
//   the retained carries bitwise equal to each other and to H[8,735]'s own halves; every written cell finite.

constexpr int32_t vc_D = 2560, vc_F = 1280, vc_HD = 256, vc_N = 8736, vc_tok = 4096;

struct vc_rng {
    uint64_t s;

    uint64_t next() {
        s += 0x9e3779b97f4a7c15ull;
        uint64_t z = s;
        z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    double uniform() { return double(next() >> 11) * 0x1.0p-53; }

    double normal() {
        const double u1 = std::max(uniform(), 1e-300), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

// the draft attention's inputs as full logical tensors (decoded values)
struct vc_weights {
    std::vector<double> eh;      // [2560][5120]
    std::vector<double> wk, wv;  // [512][2560]: rows 256 c .. of card c's head
    // SP2 item 7: the bulk span's operands of the same three (PB3 candidate 5, 79c2f42: pd_convert's FP16 rounding)
    std::vector<double> eh16, wk16, wv16;
    std::vector<double> kn;      // [256]
    std::vector<double> enorm;   // [2560]
    std::vector<double> hnorm;   // [4][2560]
    std::vector<double> gam;     // [4][2560]
    std::vector<double> wd;      // [324][4][2560]: j < 320 hc_attn_down, 320 .. 323 hc_attn_inject
    std::vector<double> wu;      // [2560][4][320]
    std::vector<double> emb;     // [4096][2560]
    // SP2 (the sparse form): the draft indexer's k_proj, BF16 [128][2560] (as stored, and its values), and k_norm
    // [128];
    // the bulk path's operand: each row as FP16 after its 2^e scaling (pd_convert_bf16), and col_s = 2^-e
    std::vector<uint16_t> ik16;
    std::vector<double>   ik, ikb, ik_cs, ikn;
};

bool vc_put(md::model & m, const char * name, int32_t c, const void * src, size_t bytes, std::string & err) {
    const int32_t idx = md::model::find(name);
    uint8_t *     dst = idx < 0 ? nullptr : m.region_ptr(idx, c);
    if (dst == nullptr) {
        err = std::string(name) + ": no region on card " + std::to_string(c);
        return false;
    }
    if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
        !cuda_ok(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), name)) {
        err = std::string(name) + ": upload";
        return false;
    }
    return true;
}

// a gguf-layout Q8_0 matrix [rows][k] split by rows between the cards (eh_proj, attn_k, attn_v)
bool vc_dense(md::model &           m,
              const char *          name,
              int32_t               rows,
              int32_t               k,
              uint64_t              seed,
              std::vector<double> & out,
              std::string &         err) {
    const fn::quant::wfmt f   = fn::quant::wfmt::q8_0;
    const int32_t   idx = md::model::find(name);
    if (idx < 0 || std::strcmp(md::tensors[idx].format, "Q8_0") != 0 || md::tensors[idx].lay != md::layout::gguf) {
        err = std::string(name) + ": not a gguf-layout Q8_0 tensor";
        return false;
    }
    const size_t         rb = fn::quant::row_bytes(f, k);
    std::vector<uint8_t> q(rb * static_cast<size_t>(rows));
    std::vector<float>   row(static_cast<size_t>(k));
    if (!fn::quant::synth_blocks(f, seed, int64_t(rows) * k / 32, q.data())) {
        err = std::string(name) + ": synth";
        return false;
    }
    out.resize(size_t(rows) * size_t(k));
    for (int32_t r = 0; r < rows; ++r) {
        if (!fn::quant::decode_row(f, q.data() + size_t(r) * rb, row.data(), k)) {
            err = std::string(name) + ": decode";
            return false;
        }
        std::copy(row.begin(), row.end(), out.begin() + ptrdiff_t(r) * k);
    }
    for (int32_t c = 0; c < 2; ++c) {
        const size_t half = rb * size_t(rows / 2);
        if (md::tensors[idx].card[c].bytes != int64_t(half)) {
            err = std::string(name) + ": card slice is not half the rows";
            return false;
        }
        if (!vc_put(m, name, c, q.data() + size_t(c) * half, half, err)) {
            return false;
        }
    }
    return true;
}

// an F32 vector of `streams` x 2560 values (streams 0: 256, replicated) as the placement slices it: card c holds
// features c*1280 .. of each stream, or (replicated) the whole vector
bool vc_f32(md::model &           m,
            const char *          name,
            int32_t               streams,
            vc_rng &              r,
            std::vector<double> & out,
            std::string &         err) {
    const bool    whole = streams == 0;
    const int32_t n     = whole ? vc_HD : streams * vc_D;
    out.resize(size_t(n));
    for (double & x : out) {
        x = static_cast<float>(0.25 + 1.5 * r.uniform());
    }
    for (int32_t c = 0; c < 2; ++c) {
        std::vector<float> slice;
        if (whole) {
            slice.assign(out.begin(), out.end());
        } else {
            for (int32_t s = 0; s < streams; ++s) {
                slice.insert(slice.end(), out.begin() + ptrdiff_t(s) * vc_D + ptrdiff_t(c) * vc_F,
                             out.begin() + ptrdiff_t(s) * vc_D + ptrdiff_t(c + 1) * vc_F);
            }
        }
        const int32_t idx = md::model::find(name);
        if (idx < 0 || md::tensors[idx].card[c].bytes != int64_t(slice.size()) * 4) {
            err = std::string(name) + ": card slice size";
            return false;
        }
        if (!vc_put(m, name, c, slice.data(), slice.size() * 4, err)) {
            return false;
        }
    }
    return true;
}

// Q8_0 blocks in hc.h's planar layout: n blocks' codes in element order, then their FP16 scales (synth's blocks
// scattered); val(b, i) receives block b's value i
template <typename Fn>
bool vc_planar(md::model & m, const char * name, int32_t c, int64_t n, uint64_t seed, Fn val, std::string & err) {
    const fn::quant::wfmt f = fn::quant::wfmt::q8_0;
    std::vector<uint8_t> blk(size_t(n) * 34), planar(size_t(n) * 34);
    if (!fn::quant::synth_blocks(f, seed, n, blk.data())) {
        err = std::string(name) + ": synth";
        return false;
    }
    for (int64_t b = 0; b < n; ++b) {
        const uint8_t * p = blk.data() + size_t(b) * 34;
        uint16_t        d = 0;
        std::memcpy(&d, p, 2);
        std::memcpy(planar.data() + size_t(b) * 32, p + 2, 32);
        std::memcpy(planar.data() + size_t(n) * 32 + size_t(b) * 2, &d, 2);
        const double dd = fn::quant::f16_to_f32(d);
        for (int32_t i = 0; i < 32; ++i) {
            val(b, i, dd * double(static_cast<int8_t>(p[2 + i])));
        }
    }
    return vc_put(m, name, c, planar.data(), planar.size(), err);
}

double vc_f16r(double x) {
    return fn::quant::f16_to_f32(fn::quant::f32_to_f16(static_cast<float>(x)));
}

bool vc_fill(md::model & m, vc_weights & w, std::string & err) {
    vc_rng r{ 0x56434b30u };
    if (!vc_dense(m, "blk.48.nextn.eh_proj.weight", vc_D, 2 * vc_D, fn::quant::synth_seed(9001), w.eh, err) ||
        !vc_dense(m, "blk.48.attn_k.weight", 2 * vc_HD, vc_D, fn::quant::synth_seed(9002), w.wk, err) ||
        !vc_dense(m, "blk.48.attn_v.weight", 2 * vc_HD, vc_D, fn::quant::synth_seed(9003), w.wv, err) ||
        !vc_f32(m, "blk.48.attn_k_norm.weight", 0, r, w.kn, err) ||
        !vc_f32(m, "blk.48.nextn.enorm.weight", 1, r, w.enorm, err) ||
        !vc_f32(m, "blk.48.nextn.hnorm.weight", 4, r, w.hnorm, err) ||
        !vc_f32(m, "blk.48.hc_attn_norm.weight", 4, r, w.gam, err)) {
        return false;
    }
    for (const auto & [src, dst] :
         { std::pair(&w.eh, &w.eh16), std::pair(&w.wk, &w.wk16), std::pair(&w.wv, &w.wv16) }) {
        dst->resize(src->size());
        for (size_t i = 0; i < src->size(); ++i) {
            (*dst)[i] = vc_f16r((*src)[i]);
        }
    }
    w.wd.assign(size_t(324) * 4 * vc_D, 0.0);
    w.wu.assign(size_t(vc_D) * 4 * 320, 0.0);
    for (int32_t c = 0; c < 2; ++c) {
        // down + inject: codes [s][j][i], blocks [s][j][i / 32] (40 a row)
        const auto down = [&](int64_t b, int32_t i, double x) {
            const int64_t s = b / int64_t(324 * 40), j = b / 40 % 324, col = (b % 40) * 32 + i;
            w.wd[(size_t(j) * 4 + size_t(s)) * vc_D + size_t(c) * vc_F + size_t(col)] = x;
        };
        // up: codes [cc][s][j], blocks [cc][s][j / 32] (10 a row)
        const auto up = [&](int64_t b, int32_t i, double x) {
            const int64_t cc = b / int64_t(4 * 10), s = b / 10 % 4, j = (b % 10) * 32 + i;
            w.wu[((size_t(c) * vc_F + size_t(cc)) * 4 + size_t(s)) * 320 + size_t(j)] = x;
        };
        if (!vc_planar(m, "blk.48.hc_attn_down.weight", c, int64_t(4) * 324 * 40, fn::quant::synth_seed(9010 + c), down,
                       err) ||
            !vc_planar(m, "blk.48.hc_attn_up.weight", c, int64_t(vc_F) * 4 * 10, fn::quant::synth_seed(9012 + c), up,
                       err)) {
            return false;
        }
    }
    // token_embd rows 0 .. 4,095 (IQ3_S, card c's 1,280 columns: 5 blocks a row)
    const fn::quant::wfmt f = fn::quant::wfmt::iq3_s;
    if (std::strcmp(md::tensors[md::model::find("token_embd.weight")].format, "IQ3_S") != 0) {
        err = "token_embd: not IQ3_S";
        return false;
    }
    const size_t rb = fn::quant::row_bytes(f, vc_F);
    w.emb.assign(size_t(vc_tok) * vc_D, 0.0);
    std::vector<uint8_t> q(rb * vc_tok);
    std::vector<float>   row(vc_F);
    for (int32_t c = 0; c < 2; ++c) {
        if (!fn::quant::synth_blocks(f, fn::quant::synth_seed(9020 + c), int64_t(vc_tok) * 5, q.data())) {
            err = "token_embd: synth";
            return false;
        }
        for (int32_t t = 0; t < vc_tok; ++t) {
            if (!fn::quant::decode_row(f, q.data() + size_t(t) * rb, row.data(), vc_F)) {
                err = "token_embd: decode";
                return false;
            }
            std::copy(row.begin(), row.end(), w.emb.begin() + ptrdiff_t(t) * vc_D + ptrdiff_t(c) * vc_F);
        }
        if (!vc_put(m, "token_embd.weight", c, q.data(), q.size(), err)) {
            return false;
        }
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") || !cuda_ok(cudaDeviceSynchronize(), "fill")) {
            return false;
        }
    }
    return true;
}

// SP2 (the sparse form): the draft indexer's k_proj (BF16, normal x 0.02) and k_norm (F32 in [0.25, 1.75]) on card 0
// (the placement's only copy; the draft's init copies both to card 1), and the bulk path's scaled FP16 rows
bool vc_fill_index(md::model & m, vc_weights & w, std::string & err) {
    constexpr int32_t KD = fn::qsa::head_dim;
    vc_rng            r{ 0x56434b32u };
    w.ik16.resize(size_t(KD) * vc_D);
    w.ik.resize(w.ik16.size());
    w.ikb.resize(w.ik16.size());
    w.ik_cs.resize(KD);
    w.ikn.resize(KD);
    for (size_t i = 0; i < w.ik16.size(); ++i) {
        w.ik16[i] = fn::quant::f32_to_bf16(static_cast<float>(0.02 * r.normal()));
        w.ik[i]   = fn::quant::bf16_to_f32(w.ik16[i]);
    }
    for (int32_t e = 0; e < KD; ++e) {
        double mx = 0.0;
        for (int32_t f = 0; f < vc_D; ++f) {
            mx = std::max(mx, std::fabs(w.ik[size_t(e) * vc_D + size_t(f)]));
        }
        int32_t ex = 0;
        if (mx > 0.0) {
            int x = 0;
            std::frexp(mx, &x);
            ex = 15 - x; // mx 2^ex in [2^14, 2^15) (pd_convert_bf16)
        }
        w.ik_cs[size_t(e)] = std::ldexp(1.0, -ex);
        for (int32_t f = 0; f < vc_D; ++f) {
            const size_t i = size_t(e) * vc_D + size_t(f);
            w.ikb[i]       = vc_f16r(std::ldexp(w.ik[i], ex));
        }
    }
    std::vector<float> kn(KD);
    for (int32_t e = 0; e < KD; ++e) {
        kn[size_t(e)]    = static_cast<float>(0.25 + 1.5 * r.uniform());
        w.ikn[size_t(e)] = kn[size_t(e)];
    }
    const int32_t ik = md::model::find("blk.48.indexer.k_proj.weight");
    const int32_t in = md::model::find("blk.48.indexer.k_norm.weight");
    if (ik < 0 || in < 0 || md::tensors[ik].card[0].bytes != int64_t(w.ik16.size()) * 2 ||
        md::tensors[in].card[0].bytes != int64_t(KD) * 4) {
        err = "blk.48.indexer: not BF16 [128][2560] and F32 [128] on card 0";
        return false;
    }
    return vc_put(m, "blk.48.indexer.k_proj.weight", 0, w.ik16.data(), w.ik16.size() * 2, err) &&
           vc_put(m, "blk.48.indexer.k_norm.weight", 0, kn.data(), kn.size() * 4, err) &&
           cuda_ok(cudaSetDevice(m.device[0]), "device") && cuda_ok(cudaDeviceSynchronize(), "fill");
}

// SP2: a cell's stored index key from the mixed row x [2560]: the bulk form (the scaled FP16 rows, then col_s) or the
// forced form (the BF16 rows), each sum in FP64, then F32, then BF16 to nearest even (the stored key)
void vc_ikey(const vc_weights & w, const double * x, bool bulk, uint16_t * key) {
    for (int32_t e = 0; e < fn::qsa::head_dim; ++e) {
        const double * row = &(bulk ? w.ikb : w.ik)[size_t(e) * vc_D];
        double         a   = 0.0;
        for (int32_t f = 0; f < vc_D; ++f) {
            a += row[f] * x[f];
        }
        key[e] = fn::quant::f32_to_bf16(static_cast<float>(bulk ? a * w.ik_cs[size_t(e)] : a));
    }
}

// SP2: block b's summary in FP64 from its four stored keys (4b .. 4b + 3, BF16 [4][128]): the mean, rms_norm times
// k_norm, the indexer rope at position 4b (pairs (i, i + 32), i < 32, angle 4b ts^i)
void vc_pool(const vc_weights & w, const uint16_t * k4, int64_t b, double ts, double * y) {
    constexpr int32_t KD = fn::qsa::head_dim;
    double            ss = 0.0;
    for (int32_t e = 0; e < KD; ++e) {
        double x = 0.0;
        for (int32_t r = 0; r < 4; ++r) {
            x += fn::quant::bf16_to_f32(k4[r * KD + e]);
        }
        y[e] = 0.25 * x;
        ss += y[e] * y[e];
    }
    const double rn = 1.0 / std::sqrt(ss / KD + 1e-6);
    for (int32_t e = 0; e < KD; ++e) {
        y[e] *= rn * w.ikn[size_t(e)];
    }
    for (int32_t i = 0; i < 32; ++i) {
        const double th = double(4 * b) * std::pow(ts, double(i)), cs = std::cos(th), sn = std::sin(th);
        const double a0 = y[i], b0 = y[i + 32];
        y[i]            = a0 * cs - b0 * sn;
        y[i + 32]       = a0 * sn + b0 * cs;
    }
}

// x [k] F32-cast, act_ref's codes and their values
void vc_act(const double * x, int32_t k, double * out) {
    std::vector<float>  f(static_cast<size_t>(k)), vals(static_cast<size_t>(k)), sc(static_cast<size_t>(k) / 16);
    std::vector<int8_t> codes(static_cast<size_t>(k));
    for (int32_t i = 0; i < k; ++i) {
        f[size_t(i)] = static_cast<float>(x[i]);
    }
    fn::quant::act_quantize_ref(f.data(), 1, k, codes.data(), sc.data());
    fn::quant::act_values_ref(codes.data(), sc.data(), k, vals.data());
    for (int32_t i = 0; i < k; ++i) {
        out[i] = vals[size_t(i)];
    }
}

// R_b's and R_f's stages (split by SP2 Amendment 1 item 7, the arithmetic unchanged; R_b's eh_proj FP16). vc_eh_in:
// position p's four eh_proj input rows in [4][2 D], e and h_s normalized, then the bulk span's FP16 rounding or the
// forced path's activation codes
void vc_eh_in(const vc_weights &           w,
              const float *                H,
              const std::vector<int32_t> & tok,
              int64_t                      p,
              bool                         forced,
              double *                     in) {
    constexpr int32_t D = vc_D, K2 = 2 * vc_D;
    const double *      e  = &w.emb[size_t(tok[size_t(p)]) * D];
    double              ss = 0.0;
    for (int32_t i = 0; i < D; ++i) {
        ss += e[i] * e[i];
    }
    const double rse = 1.0 / std::sqrt(ss / D + 1e-6);
    for (int32_t s = 0; s < 4; ++s) {
        double hs[D];
        double sh = 0.0;
        for (int32_t i = 0; i < D; ++i) {
            hs[i] = p == 0 ? 0.0 : double(H[(size_t(p - 1) * 4 + size_t(s)) * D + size_t(i)]);
            sh += hs[i] * hs[i];
        }
        const double rsh = 1.0 / std::sqrt(sh / D + 1e-6);
        double *     row = &in[size_t(s) * K2];
        for (int32_t i = 0; i < D; ++i) {
            row[i]     = e[i] * w.enorm[size_t(i)] * rse;
            row[D + i] = hs[i] * w.hnorm[size_t(s) * D + size_t(i)] * rsh;
        }
        if (forced) {
            vc_act(row, K2, row);
        } else {
            for (int32_t k = 0; k < K2; ++k) {
                row[k] = vc_f16r(row[k]);
            }
        }
    }
}

// vc_eh: x [4][D] = eh_proj times in, FP64: the Q8_0 values decoded exactly (forced), or their FP16 rounding (bulk)
void vc_eh(const vc_weights & w, const double * in, bool bulk, double * x) {
    constexpr int32_t D = vc_D, K2 = 2 * vc_D;
    for (int32_t o = 0; o < D; ++o) {
        const double * wr   = &(bulk ? w.eh16 : w.eh)[size_t(o) * K2];
        double         a[4] = { 0, 0, 0, 0 };
        for (int32_t k = 0; k < K2; ++k) {
            for (int32_t s = 0; s < 4; ++s) {
                a[s] += wr[k] * in[size_t(s) * K2 + size_t(k)];
            }
        }
        for (int32_t s = 0; s < 4; ++s) {
            x[size_t(s) * D + size_t(o)] = a[s];
        }
    }
}

// vc_hc: the split HC's formulas (hc_prompt.h) in FP64 on x [4][D]: low [320] and the mixed row [D]
void vc_hc(const vc_weights & w, const double * x, double * low, double * mixed) {
    constexpr int32_t D = vc_D;
    double rms[4];
    for (int32_t s = 0; s < 4; ++s) {
        double v = 0.0;
        for (int32_t f = 0; f < D; ++f) {
            v += x[size_t(s) * D + size_t(f)] * x[size_t(s) * D + size_t(f)];
        }
        rms[s] = std::sqrt(v / D + 1e-6);
    }
    for (int32_t j = 0; j < 320; ++j) {
        double z = 0.0;
        for (int32_t s = 0; s < 4; ++s) {
            double         u  = 0.0;
            const double * wd = &w.wd[(size_t(j) * 4 + size_t(s)) * D];
            for (int32_t f = 0; f < D; ++f) {
                u += wd[f] * w.gam[size_t(s) * D + size_t(f)] * x[size_t(s) * D + size_t(f)];
            }
            z += u / rms[s];
        }
        z *= 0.25;
        low[j] = z / (1.0 + std::exp(-z));
    }
    for (int32_t f = 0; f < D; ++f) {
        double m = 0.0;
        for (int32_t s = 0; s < 4; ++s) {
            const double * wu = &w.wu[(size_t(f) * 4 + size_t(s)) * 320];
            double         g  = 0.0;
            for (int32_t j = 0; j < 320; ++j) {
                g += wu[j] * low[j];
            }
            g = 1.0 / (1.0 + std::exp(-g));
            m += w.gam[size_t(s) * D + size_t(f)] * x[size_t(s) * D + size_t(f)] / rms[s] * g;
        }
        mixed[f] = 0.25 * m;
    }
}

// the FP64 catch-up of position p, with the bulk span's roundings (forced = false) or the forced path's; both cards'
// K and V cells (F16). SP2 Amendment 1's localization (bulk only): mq_out receives the modelled FP16 mixed row, and
// mix_in (the bulk span's captured FP16 mixed row) replaces it as the K and V products' operand
void vc_reference(const vc_weights &           w,
                  const float *                H,
                  const std::vector<int32_t> & tok,
                  int64_t                      p,
                  bool                         forced,
                  float                        ts,
                  uint16_t (&K)[2][vc_HD],
                  uint16_t (&V)[2][vc_HD],
                  uint16_t *                   ikey   = nullptr,
                  const uint16_t *             mix_in = nullptr,
                  double *                     mq_out = nullptr) {
    constexpr int32_t   D = vc_D, K2 = 2 * vc_D;
    std::vector<double> in(size_t(4) * K2), x(size_t(4) * D), mixed(D), mq(D);
    double              low[320];
    vc_eh_in(w, H, tok, p, forced, in.data());
    vc_eh(w, in.data(), !forced, x.data());
    vc_hc(w, x.data(), low, mixed.data());
    if (forced) {
        vc_act(mixed.data(), D, mq.data());
    } else {
        for (int32_t f = 0; f < D; ++f) {
            mq[size_t(f)] = vc_f16r(mixed[size_t(f)]);
            if (mq_out != nullptr) {
                mq_out[f] = mq[size_t(f)];
            }
            if (mix_in != nullptr) {
                mq[size_t(f)] = fn::quant::f16_to_f32(mix_in[f]);
            }
        }
    }
    if (ikey != nullptr) {
        // SP2: the index key's operand, the forced path's F32 mixed row (the BF16 product reads F32 activations) or
        // the bulk span's FP16 one
        std::vector<double> xf(D);
        for (int32_t f = 0; f < D; ++f) {
            xf[size_t(f)] = forced ? double(static_cast<float>(mixed[size_t(f)])) : mq[size_t(f)];
        }
        vc_ikey(w, xf.data(), !forced, ikey);
    }
    for (int32_t c = 0; c < 2; ++c) {
        double kk[vc_HD], vv[vc_HD], sk = 0.0;
        for (int32_t e2 = 0; e2 < vc_HD; ++e2) {
            const double * rk = &(forced ? w.wk : w.wk16)[(size_t(c) * vc_HD + size_t(e2)) * D];
            const double * rv = &(forced ? w.wv : w.wv16)[(size_t(c) * vc_HD + size_t(e2)) * D];
            double         ak = 0.0, av = 0.0;
            for (int32_t f = 0; f < D; ++f) {
                ak += rk[f] * mq[size_t(f)];
                av += rv[f] * mq[size_t(f)];
            }
            kk[e2] = ak;
            vv[e2] = av;
            sk += ak * ak;
        }
        const double rn = 1.0 / std::sqrt(sk / vc_HD + 1e-6);
        double       y[vc_HD];
        for (int32_t e2 = 0; e2 < vc_HD; ++e2) {
            y[e2] = kk[e2] * rn * w.kn[size_t(e2)];
        }
        for (int32_t i = 0; i < 32; ++i) {  // dcp-test's rope: pair (i, i + 32), the F32 angle chain
            float theta = static_cast<float>(p);
            for (int32_t q = 0; q < i; ++q) {
                theta = theta * ts;
            }
            const double cs = std::cos(double(theta)), sn = std::sin(double(theta)), a0 = y[i], b0 = y[i + 32];
            y[i]      = a0 * cs - b0 * sn;
            y[i + 32] = a0 * sn + b0 * cs;
        }
        for (int32_t e2 = 0; e2 < vc_HD; ++e2) {
            K[c][e2] = fn::quant::f32_to_f16(static_cast<float>(y[e2]));
            V[c][e2] = fn::quant::f32_to_f16(static_cast<float>(vv[e2]));
        }
    }
}

// per cell: max |x - y| / max |y| over its 256 values, and ||x|| / ||y||
double vc_rel(const uint16_t * x, const uint16_t * y) {
    double dm = 0.0, ym = 0.0;
    for (int32_t e = 0; e < vc_HD; ++e) {
        const double fx = fn::quant::f16_to_f32(x[e]), fy = fn::quant::f16_to_f32(y[e]);
        dm = std::max(dm, std::fabs(fx - fy));
        ym = std::max(ym, std::fabs(fy));
    }
    return ym > 0 ? dm / ym : dm;
}

double vc_ratio(const uint16_t * x, const uint16_t * y) {
    double nx = 0.0, ny = 0.0;
    for (int32_t e = 0; e < vc_HD; ++e) {
        const double fx = fn::quant::f16_to_f32(x[e]), fy = fn::quant::f16_to_f32(y[e]);
        nx += fx * fx;
        ny += fy * fy;
    }
    return ny > 0 ? std::sqrt(nx / ny) : 0.0;
}

std::string vc_stats(const std::vector<double> & x) {
    char s[96];
    snprintf(s, sizeof(s), "p50 %.3g p99 %.3g max %.3g", pct(x, 50), pct(x, 99), pct(x, 100));
    return s;
}

// SP2 Amendment 1, vcheck's sparse form: the draft index of the forced run (catch-ups, B 1 .. 5: the round form's
// key write and pooling) and of the bulk span (the BF16 product on tensor cores, qsp_keys and the cache form's pooling)
// on identical inputs. Per cell, max |x - y| / max |y| over its 128 key values (per block, over the summary's 128).
// Bounds set before the run:
//   both runs: index_n = 8,736; every key of cells 0 .. 8,735 and summary of blocks 0 .. 2,183 finite (the cells and
//     summaries were poisoned NaN); both cards' keys and summaries bitwise equal, and the bulk span's FP16 mixed rows;
//   each run's summaries bitwise the cache form's pooling (summary_update, QA1b's arithmetic, held bitwise to its
//     reference by qa-test) of that run's own stored keys: the right keys of the right cells pooled at the right
//     position;
//   bulk keys against R_b' over all cells: the FP64 product of the bulk span's own captured FP16 mixed row with the
//     2^e-scaled FP16 rows, times 2^-e, F32, then BF16 to nearest even, <= 8e-3 every cell (one BF16 step of the
//     cell's largest value, 2^-7 = 7.8e-3, for the stored key's rounding, plus pd-test's 1e-5 accumulation);
//   forced keys against R_f on the sampled positions' whole blocks: the FP64 mixed row in F32 with the BF16 rows,
//     <= 8e-3 every cell (the same BF16 step, plus draft-kernel-test's 1e-6, the eh_proj's 1e-5 and hc2-test's 1e-5);
//   bulk against forced over all cells (keys) and all blocks (summaries): p50, p99 and max each <= 2 x the same
//     statistic of R_b against R_f (R_b: vc_reference's bulk roundings, the FP16 mixed row; keys on the sampled
//     cells, summaries pooled in FP64 on the sampled blocks) + 1.6e-2 (both stored keys' final BF16 roundings);
//   the forced run has at least one five-row catch-up that completes two blocks.
double vc_reld(const double * x, const double * y, int32_t n) {
    double dm = 0.0, ym = 0.0;
    for (int32_t e = 0; e < n; ++e) {
        dm = std::max(dm, std::fabs(x[e] - y[e]));
        ym = std::max(ym, std::fabs(y[e]));
    }
    return ym > 0 ? dm / ym : dm;
}

double vc_relbf(const uint16_t * x, const uint16_t * y) {
    double fx[fn::qsa::head_dim], fy[fn::qsa::head_dim];
    for (int32_t e = 0; e < fn::qsa::head_dim; ++e) {
        fx[e] = fn::quant::bf16_to_f32(x[e]);
        fy[e] = fn::quant::bf16_to_f32(y[e]);
    }
    return vc_reld(fx, fy, fn::qsa::head_dim);
}

template <typename Fn> void vc_parallel(size_t n, Fn fn) {
    std::atomic<size_t>      next{ 0 };
    std::vector<std::thread> pool;
    const int32_t            nt = std::max(1, std::min(12, int32_t(std::thread::hardware_concurrency())));
    for (int32_t i = 0; i < nt; ++i) {
        pool.emplace_back([&]() {
            for (size_t k = next++; k < n; k = next++) {
                fn(k);
            }
        });
    }
    for (std::thread & th : pool) {
        th.join();
    }
}

void vc_index_checks(rt::target &                 t,
                     md::model &                  m,
                     const vc_weights &           w,
                     const float *                H,
                     const std::vector<int32_t> & P,
                     const std::vector<int64_t> & S0,
                     int64_t                      two_blocks,
                     const draft_arrays &         df,
                     const draft_arrays &         db,
                     uint16_t * const             mix[2],
                     verdicts &                   v) {
    constexpr int32_t KD = fn::qsa::head_dim, N = vc_N, NB = vc_N / 4, D = vc_D;
    const auto sm_at = [](const std::vector<float> & f, int64_t b, int32_t e) {
        return f[size_t(b / 32) * 4096 + size_t(e / 4) * 128 + size_t(b % 32) * 4 + size_t(e % 4)];
    };
    const draft_arrays * run[2] = { &df, &db };
    // written and finite
    bool sizes = true, finite = true;
    for (const draft_arrays * r : run) {
        for (int32_t c = 0; c < 2; ++c) {
            sizes = sizes && r->ik[c].size() == size_t(N) * KD && r->sm[c].size() >= fn::qsa::summary_bytes(NB) / 4;
        }
    }
    for (int32_t i = 0; i < 2 && sizes; ++i) {
        for (int32_t c = 0; c < 2; ++c) {
            for (uint16_t k : run[i]->ik[c]) {
                finite = finite && std::isfinite(fn::quant::bf16_to_f32(k));
            }
            for (int64_t b = 0; b < NB; ++b) {
                for (int32_t e = 0; e < KD; ++e) {
                    finite = finite && std::isfinite(sm_at(run[i]->sm[c], b, e));
                }
            }
        }
    }
    const bool written = sizes && df.index_n == N && db.index_n == N;
    v.put("vcheck index written", written && finite,
          "index_n forced " + std::to_string(df.index_n) + ", bulk " + std::to_string(db.index_n) + " (N " +
              std::to_string(N) + "); keys of cells 0 .. 8,735 and summaries of blocks 0 .. 2,183 " +
              (finite ? "finite" : "NOT all finite (unwritten, or a range error)") + " in both runs, both cards");
    if (!written) {
        return;
    }
    // both cards equal
    bool cards = df.ik[0] == df.ik[1] && db.ik[0] == db.ik[1] &&
                 std::memcmp(mix[0], mix[1], size_t(N) * D * 2) == 0;
    for (const draft_arrays * r : run) {
        for (int64_t b = 0; b < NB && cards; ++b) {
            for (int32_t e = 0; e < KD; ++e) {
                const float x = sm_at(r->sm[0], b, e), y = sm_at(r->sm[1], b, e);
                cards         = cards && std::memcmp(&x, &y, 4) == 0;
            }
        }
    }
    v.put("vcheck index both cards", cards,
          cards ? "keys and summaries bitwise equal on the two cards in both runs; the bulk span's FP16 mixed rows "
                  "bitwise equal on the two cards"
                : "the cards differ");
    // each run's summaries against the cache form's pooling of its own stored keys, on each card
    int64_t differ[2] = { 0, 0 };
    bool    pool_ok   = true;
    for (int32_t i = 0; i < 2; ++i) {
        for (int32_t c = 0; c < 2 && pool_ok; ++c) {
            uint16_t *         dk = nullptr;
            float *            ds = nullptr;
            cudaStream_t       st = nullptr;
            std::vector<float> got;
            pool_ok = cuda_ok(cudaSetDevice(m.device[c]), "device") &&
                      cuda_ok(cudaMalloc(&dk, size_t(N) * KD * 2), "keys") &&
                      cuda_ok(cudaMalloc(&ds, fn::qsa::summary_bytes(NB)), "summaries") &&
                      cuda_ok(cudaStreamCreate(&st), "stream") &&
                      cuda_ok(cudaMemcpy(dk, run[i]->ik[c].data(), size_t(N) * KD * 2, cudaMemcpyHostToDevice),
                              "keys") &&
                      cuda_ok(cudaMemset(ds, 0, fn::qsa::summary_bytes(NB)), "summaries");
            if (pool_ok) {
                fn::qsa::update_args ua;
                ua.keys      = dk;
                ua.k_norm    = t.draft(c).iw.k_norm;
                ua.summaries = ds;
                ua.first     = 0;
                ua.count     = NB;
                ua.rope      = fn::qsa::model_rope();
                pool_ok      = fn::qsa::summary_update(ua, st) && cuda_ok(cudaStreamSynchronize(st), "pool") &&
                          get(got, static_cast<const float *>(ds), fn::qsa::summary_bytes(NB) / 4);
            }
            for (int64_t b = 0; b < NB && pool_ok; ++b) {
                bool same = true;
                for (int32_t e = 0; e < KD; ++e) {
                    const float x = sm_at(run[i]->sm[c], b, e), y = sm_at(got, b, e);
                    same          = same && std::memcmp(&x, &y, 4) == 0;
                }
                differ[i] += same ? 0 : 1;
            }
            if (st != nullptr) {
                cudaStreamDestroy(st);
            }
            cudaFree(dk);
            cudaFree(ds);
        }
    }
    v.put("vcheck summaries re-pooled", pool_ok && differ[0] == 0 && differ[1] == 0,
          pool_ok ? "each run's stored summaries against the cache form's pooling of its own stored keys, bitwise: "
                    "blocks differing of 2,184 x 2 cards: forced " + std::to_string(differ[0]) + ", bulk " +
                        std::to_string(differ[1])
                  : "the re-pooling did not run");
    // bulk keys against R_b' (the bulk span's own FP16 mixed row) over all cells
    std::vector<double> eb(static_cast<size_t>(N));
    vc_parallel(size_t(N), [&](size_t p) {
        std::vector<double> x(static_cast<size_t>(D));
        uint16_t            key[KD];
        for (int32_t f = 0; f < D; ++f) {
            x[size_t(f)] = fn::quant::f16_to_f32(mix[0][p * D + size_t(f)]);
        }
        vc_ikey(w, x.data(), true, key);
        eb[p] = vc_relbf(&db.ik[0][p * KD], key);
    });
    const int64_t wb = std::max_element(eb.begin(), eb.end()) - eb.begin();
    int64_t       over_b = 0;
    for (double x : eb) {
        over_b += x > 8e-3 ? 1 : 0;
    }
    v.put("vcheck bulk keys against FP64 (own mixed row)", over_b == 0,
          "all 8,736 cells: " + vc_stats(eb) + " (bound 8e-3 every cell; worst at cell " + std::to_string(wb) +
              "; cells over " + std::to_string(over_b) + ")");
    // the sampled positions' whole blocks: R_b and R_f keys (vc_reference), forced against R_f, the design spread
    std::vector<int64_t> SB;
    for (int64_t p : S0) {
        SB.push_back(p / 4);
    }
    std::sort(SB.begin(), SB.end());
    SB.erase(std::unique(SB.begin(), SB.end()), SB.end());
    std::vector<std::array<uint16_t, KD>> rbk(SB.size() * 4), rfk(SB.size() * 4);
    const float                           ts = fn::qsa::attn::rope_theta_scale();
    vc_parallel(SB.size() * 4, [&](size_t k) {
        const int64_t p = SB[k / 4] * 4 + int64_t(k % 4);
        uint16_t      K[2][vc_HD], V[2][vc_HD];
        vc_reference(w, H, P, p, false, ts, K, V, rbk[k].data());
        vc_reference(w, H, P, p, true, ts, K, V, rfk[k].data());
    });
    std::vector<double> ef, dk, ds, ak, as;
    int64_t             wf = -1;
    double              wfv = 0.0;
    for (size_t k = 0; k < SB.size() * 4; ++k) {
        const int64_t p = SB[k / 4] * 4 + int64_t(k % 4);
        ef.push_back(vc_relbf(&df.ik[0][size_t(p) * KD], rfk[k].data()));
        dk.push_back(vc_relbf(rbk[k].data(), rfk[k].data()));
        if (ef.back() > wfv) {
            wfv = ef.back();
            wf  = p;
        }
    }
    const double idx_ts = fn::qsa::model_rope().theta_scale;
    for (size_t j = 0; j < SB.size(); ++j) {
        uint16_t kb[4 * KD], kf[4 * KD];
        for (int32_t r = 0; r < 4; ++r) {
            std::memcpy(kb + r * KD, rbk[j * 4 + size_t(r)].data(), KD * 2);
            std::memcpy(kf + r * KD, rfk[j * 4 + size_t(r)].data(), KD * 2);
        }
        double yb[KD], yf[KD];
        vc_pool(w, kb, SB[j], idx_ts, yb);
        vc_pool(w, kf, SB[j], idx_ts, yf);
        ds.push_back(vc_reld(yb, yf, KD));
    }
    v.put("vcheck forced keys against FP64", pct(ef, 100) <= 8e-3,
          std::to_string(SB.size()) + " sampled blocks' " + std::to_string(SB.size() * 4) + " cells: " + vc_stats(ef) +
              " (bound 8e-3 every cell; worst at cell " + std::to_string(wf) + ")");
    // bulk against forced: every cell's keys, every block's summaries
    for (int64_t p = 0; p < N; ++p) {
        ak.push_back(vc_relbf(&db.ik[0][size_t(p) * KD], &df.ik[0][size_t(p) * KD]));
    }
    for (int64_t b = 0; b < NB; ++b) {
        double xb[KD], xf[KD];
        for (int32_t e = 0; e < KD; ++e) {
            xb[e] = sm_at(db.sm[0], b, e);
            xf[e] = sm_at(df.sm[0], b, e);
        }
        as.push_back(vc_reld(xb, xf, KD));
    }
    bool pk = true, ps = true;
    for (const double q : { 50.0, 99.0, 100.0 }) {
        pk = pk && pct(ak, q) <= 2.0 * pct(dk, q) + 1.6e-2;
        ps = ps && pct(as, q) <= 2.0 * pct(ds, q) + 1.6e-2;
    }
    v.put("vcheck index keys bulk against forced", pk,
          "all 8,736 cells: " + vc_stats(ak) + "; bound 2 x R_b against R_f on the sampled cells (" + vc_stats(dk) +
              ") + 1.6e-2");
    v.put("vcheck index summaries bulk against forced", ps,
          "all 2,184 blocks: " + vc_stats(as) + "; bound 2 x R_b against R_f pooled in FP64 on the sampled blocks (" +
              vc_stats(ds) + ") + 1.6e-2");
    v.put("vcheck five-row catch-ups", two_blocks > 0,
          std::to_string(two_blocks) + " forced catch-ups of five rows that complete two blocks each");
}

int vcheck_mode(const fnfl_target_args & a, verdicts & v) {
    constexpr int32_t cells = 16384, D = vc_D, F = vc_F, N = vc_N;
    rt::draft_options dopt;
    if (!fnfl_draft_settings(a, true, "prefill-test", dopt)) {
        return 1;
    }
    // SP2 Amendment 1: the sparse form (--draft-attention sparse) also fills the draft indexer's k_proj and k_norm,
    // cycles the forced run's B through 1 .. 5 and checks the draft index (vc_index_checks)
    const bool    sparse = dopt.attention == "sparse";
    const int32_t b_max  = sparse ? 5 : 4;
    auto          m      = std::make_unique<md::model>();
    if (!fnfl_load_dry(a, cells, *m)) {
        return 1;
    }
    printf(
        "prefill-test: VCHECK: a dry load (no weights read), the draft attention's inputs random; %d positions, "
        "forced catch-up (B 1..%d) against the bulk span%s\n",
        N, b_max, sparse ? "; the sparse form: the draft index checked too (SP2)" : "");
    std::string err;
    vc_weights  w;
    if (!vc_fill(*m, w, err) || (sparse && !vc_fill_index(*m, w, err))) {
        fprintf(stderr, "prefill-test: vcheck fill: %s\n", err.c_str());
        return 1;
    }
    vc_rng               r{ 0x56434b31u };
    std::vector<int32_t> P(N);
    for (int32_t & t : P) {
        t = static_cast<int32_t>(r.next() % vc_tok);
    }
    float *    H      = nullptr;
    uint16_t * mix[2] = { nullptr, nullptr }; // SP2: the bulk span's FP16 mixed rows, each card's
    float *    ehh[2] = { nullptr, nullptr }; // SP2 item 7: its eh_proj outputs [N][4][1280] and HC low [N][320]
    float *    lowh[2] = { nullptr, nullptr };
    if (!cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&H), size_t(N) * 4 * D * 4, cudaHostAllocPortable),
                 "hidden rows")) {
        return 1;
    }
    for (int32_t c = 0; c < 2 && sparse; ++c) {
        if (!cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&mix[c]), size_t(N) * D * 2, cudaHostAllocPortable),
                     "mixed rows") ||
            !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&ehh[c]), size_t(N) * 4 * vc_F * 4,
                                   cudaHostAllocPortable),
                     "eh rows") ||
            !cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&lowh[c]), size_t(N) * 320 * 4, cudaHostAllocPortable),
                     "low rows")) {
            for (void * x : { static_cast<void *>(H), static_cast<void *>(mix[0]), static_cast<void *>(mix[1]),
                              static_cast<void *>(ehh[0]), static_cast<void *>(ehh[1]), static_cast<void *>(lowh[0]),
                              static_cast<void *>(lowh[1]) }) {
                if (x != nullptr) {
                    cudaFreeHost(x);
                }
            }
            return 1;
        }
    }
    const auto free_host = [&]() {
        cudaFreeHost(H);
        for (void * x : { static_cast<void *>(mix[0]), static_cast<void *>(mix[1]), static_cast<void *>(ehh[0]),
                          static_cast<void *>(ehh[1]), static_cast<void *>(lowh[0]), static_cast<void *>(lowh[1]) }) {
            if (x != nullptr) {
                cudaFreeHost(x);
            }
        }
    };
    const double sigma[4] = { 1.0, 3.0, 0.5, 8.0 };
    for (int64_t p = 0; p < N; ++p) {
        for (int32_t s = 0; s < 4; ++s) {
            for (int32_t f = 0; f < D; ++f) {
                const bool outlier = f == 7 || f == 300 || f == 1299 || f == 1500 || f == 2047 || f == 2559;
                H[(size_t(p) * 4 + size_t(s)) * D + size_t(f)] =
                    static_cast<float>(sigma[s] * r.normal() * (outlier ? 12.0 : 1.0));
            }
        }
    }
    int rc = 1;
    {
        rt::target_options o;
        o.model_path    = a.model_path;
        o.sample        = true;
        o.sampling.temp = 0.0f;
        o.draft         = dopt;
        o.timeout_s     = 120.0;
        rt::target    t;
        rt::prefiller pf;
        if (!t.init(*m, o, err) || !pf.init(t, *m, err)) {
            fprintf(stderr, "prefill-test: init: %s\n", err.c_str());
            free_host();
            return 1;
        }
        // a zero retained carry and NaN draft cells
        const auto prepare = [&]() {
            for (int32_t c = 0; c < 2; ++c) {
                md::draft_card & d = t.draft(c);
                if (!cuda_ok(cudaSetDevice(m->device[c]), "device") ||
                    !cuda_ok(cudaMemset(d.carry_prev, 0, size_t(4) * F * 4), "carry") ||
                    !cuda_ok(cudaMemset(d.k_cache, 0xff, size_t(N) * vc_HD * 2), "poison K") ||
                    !cuda_ok(cudaMemset(d.v_cache, 0xff, size_t(N) * vc_HD * 2), "poison V") ||
                    (sparse &&
                     !cuda_ok(cudaMemset(d.idx_keys, 0xff, size_t(N) * fn::qsa::head_dim * 2), "poison keys")) ||
                    (sparse &&
                     !cuda_ok(cudaMemset(d.idx_summ, 0xff, fn::qsa::summary_bytes(N / 4)), "poison summaries")) ||
                    !cuda_ok(cudaDeviceSynchronize(), "prepare")) {
                    return false;
                }
            }
            return true;
        };
        draft_arrays df, db;
        // the forced catch-up: target rounds of B = 1, 2, 3, 4, ..., each followed by the catch-up of its rows on the
        // given hidden rows
        bool         ok = t.reset(err) && prepare();
        t.set_prompt(P);
        std::vector<float> stage(size_t(b_max) * 4 * F);
        int64_t            pos = 0, rounds = 0, two_blocks = 0;
        for (size_t ci = 0; ok && pos < N; ++ci) {
            const int32_t B = static_cast<int32_t>(std::min<int64_t>(int64_t(ci % size_t(b_max)) + 1, N - pos));
            two_blocks += B == 5 && pos % 4 == 3 ? 1 : 0; // SP2: a five-row catch-up that completes two blocks
            rt::round_result rr;
            ok = t.forced(B, rr, err);
            for (int32_t c = 0; c < 2 && ok; ++c) {
                for (int32_t b = 0; b < B; ++b) {
                    for (int32_t s = 0; s < 4; ++s) {
                        std::memcpy(&stage[(size_t(b) * 4 + size_t(s)) * F],
                                    &H[(size_t(pos + b) * 4 + size_t(s)) * D + size_t(c) * F], size_t(F) * 4);
                    }
                }
                ok = cuda_ok(cudaSetDevice(m->device[c]), "device") &&
                     cuda_ok(cudaMemcpy(t.card(c).carry, stage.data(), size_t(B) * 4 * F * 4, cudaMemcpyHostToDevice),
                             "carry rows");
            }
            ok = ok && t.draft_catchup(B, rr, err);
            pos += B;
            ++rounds;
        }
        ok              = ok && read_draft(t, *m, df);
        const bool f_ok = ok;
        v.put("vcheck forced catch-up", ok && df.n == N,
              ok ? std::to_string(rounds) + " target rounds and catch-ups (B 1.." + std::to_string(b_max) +
                       "); draft_n " + std::to_string(df.n) +
                       (sparse ? "; five-row catch-ups completing two blocks " + std::to_string(two_blocks) : "") :
                   err);
        // the bulk span
        int64_t          from = -1;
        rt::pf_run_stats st;
        pf.prompt()->test_hidden(H, N);
        if (sparse) {
            pf.prompt()->test_mixed(mix[0], mix[1], N);
            pf.prompt()->test_stages(ehh[0], ehh[1], lowh[0], lowh[1]);
        }
        ok = pf.begin(t, P, 0, from, err) && prepare() && pf.run(t, &st, err) && sync_cards(t, *m) &&
             read_draft(t, *m, db);
        pf.prompt()->test_hidden(nullptr, 0);
        pf.prompt()->test_mixed(nullptr, nullptr, 0);
        pf.prompt()->test_stages(nullptr, nullptr, nullptr, nullptr);
        const bool b_ok = ok;
        v.put("vcheck bulk span", ok && st.bulk && db.n == N && st.rounds == 2,
              ok ? "chunks " + chunk_list(st) + "; draft_n " + std::to_string(db.n) : err);
        if (!f_ok || !b_ok || df.n != N || db.n != N) {
            free_host();
            return 1;
        }
        bool finite = true;
        for (int32_t c = 0; c < 2; ++c) {
            for (const std::vector<uint16_t> * x : { &df.k[c], &df.v[c], &db.k[c], &db.v[c] }) {
                for (uint16_t h : *x) {
                    finite = finite && std::isfinite(fn::quant::f16_to_f32(h));
                }
            }
        }
        v.put("vcheck every cell written", finite,
              finite ? "K and V finite in both runs, both cards, cells 0 .. 8,735"
                     : "a NaN or infinite cell (unwritten, or a range error)");
        // the sampled positions: the first, both halves' and the chunk boundary's neighbours, the last, every 31st
        std::vector<int64_t> S = {0,    1,    2,    3,    4,    5,    6,    7,    4094, 4095, 4096, 4097, 8189,
                                  8190, 8191, 8192, 8193, 8194, 8462, 8463, 8464, 8465, 8732, 8733, 8734, 8735};
        for (int64_t p = 17; p < N; p += 31) {
            S.push_back(p);
        }
        std::sort(S.begin(), S.end());
        S.erase(std::unique(S.begin(), S.end()), S.end());

        struct ref_cell {
            uint16_t K[2][vc_HD], V[2][vc_HD];
        };

        std::vector<ref_cell> rb(S.size()), rf(S.size());
        const float           ts = fn::qsa::attn::rope_theta_scale();
        {
            std::atomic<size_t>      next{ 0 };
            std::vector<std::thread> pool;
            const int32_t            nt = std::max(1, std::min(12, int32_t(std::thread::hardware_concurrency())));
            for (int32_t i = 0; i < nt; ++i) {
                pool.emplace_back([&]() {
                    for (size_t k = next++; k < S.size(); k = next++) {
                        vc_reference(w, H, P, S[k], false, ts, rb[k].K, rb[k].V);
                        vc_reference(w, H, P, S[k], true, ts, rf[k].K, rf[k].V);
                    }
                });
            }
            for (std::thread & th : pool) {
                th.join();
            }
        }
        std::vector<double> bk, bv, fk, fv, dk, dv;  // bulk-R_b, forced-R_f, R_b-R_f (design)
        int64_t             worst_b = -1, worst_f = -1;
        double              wb = 0.0, wf = 0.0;
        for (size_t k = 0; k < S.size(); ++k) {
            const size_t at = size_t(S[k]) * vc_HD;
            for (int32_t c = 0; c < 2; ++c) {
                const double xbk = vc_rel(&db.k[c][at], rb[k].K[c]), xbv = vc_rel(&db.v[c][at], rb[k].V[c]);
                const double xfk = vc_rel(&df.k[c][at], rf[k].K[c]), xfv = vc_rel(&df.v[c][at], rf[k].V[c]);
                bk.push_back(xbk);
                bv.push_back(xbv);
                fk.push_back(xfk);
                fv.push_back(xfv);
                dk.push_back(vc_rel(rb[k].K[c], rf[k].K[c]));
                dv.push_back(vc_rel(rb[k].V[c], rf[k].V[c]));
                if (std::max(xbk, xbv) > wb) {
                    wb      = std::max(xbk, xbv);
                    worst_b = S[k];
                }
                if (std::max(xfk, xfv) > wf) {
                    wf      = std::max(xfk, xfv);
                    worst_f = S[k];
                }
            }
        }
        const bool pa = pct(bk, 100) <= 2e-3 && pct(bv, 100) <= 2e-3;
        v.put("vcheck bulk against FP64 (bulk numerics)", pa,
              std::to_string(S.size()) + " sampled positions x 2 cards; K " + vc_stats(bk) + "; V " + vc_stats(bv) +
                  " (bound 2e-3 every cell; worst at position " + std::to_string(worst_b) + ")");
        if (sparse) {
            // SP2 Amendment 1, localizing the check above (information, no verdict): on the sampled positions, the
            // bulk span's captured FP16 mixed row (card 0's; the cards' are bitwise equal, checked below) against
            // R_b's modelled one, per position max |captured - modelled| / max |modelled|; then the bulk cells
            // against R_b' (R_b with the captured row as the K and V products' operand)
            std::vector<ref_cell> rm(S.size());
            std::vector<double>   em(S.size()), mk, mv;
            vc_parallel(S.size(), [&](size_t k) {
                std::vector<double> mo(static_cast<size_t>(vc_D));
                vc_reference(w, H, P, S[k], false, ts, rm[k].K, rm[k].V, nullptr, mix[0] + size_t(S[k]) * vc_D,
                             mo.data());
                double dm = 0.0, ym = 0.0;
                for (int32_t f = 0; f < vc_D; ++f) {
                    const double x = fn::quant::f16_to_f32(mix[0][size_t(S[k]) * vc_D + size_t(f)]);
                    dm             = std::max(dm, std::fabs(x - mo[size_t(f)]));
                    ym             = std::max(ym, std::fabs(mo[size_t(f)]));
                }
                em[k] = ym > 0 ? dm / ym : dm;
            });
            for (size_t k = 0; k < S.size(); ++k) {
                const size_t at = size_t(S[k]) * vc_HD;
                for (int32_t c = 0; c < 2; ++c) {
                    mk.push_back(vc_rel(&db.k[c][at], rm[k].K[c]));
                    mv.push_back(vc_rel(&db.v[c][at], rm[k].V[c]));
                }
            }
            // item 7, per stage on the same positions, each against FP64 on the bulk span's own captured input:
            // eh_proj (captured streams against R_b's FP64 product of its FP16-rounded inputs; per stream too), then
            // the split HC on the captured streams (low against captured low, both cards; the mixed row against the
            // captured FP16 row)
            std::vector<double> e_eh(S.size()), e_low(S.size() * 2), e_mix(S.size()), e_st[4];
            for (std::vector<double> & x : e_st) {
                x.resize(S.size());
            }
            vc_parallel(S.size(), [&](size_t k) {
                const int64_t       p = S[k];
                std::vector<double> in(size_t(4) * 2 * vc_D), xr(size_t(4) * vc_D), xc(size_t(4) * vc_D), lo(320),
                    mx(static_cast<size_t>(vc_D));
                vc_eh_in(w, H, P, p, false, in.data());
                vc_eh(w, in.data(), true, xr.data());
                double dm = 0.0, ym = 0.0;
                for (int32_t st = 0; st < 4; ++st) {
                    double dms = 0.0, yms = 0.0;
                    for (int32_t o = 0; o < vc_D; ++o) {
                        const int32_t c  = o / vc_F;
                        const size_t  at = size_t(st) * vc_D + size_t(o);
                        xc[at] = ehh[c][(size_t(p) * 4 + size_t(st)) * vc_F + size_t(o - c * vc_F)];
                        dms    = std::max(dms, std::fabs(xc[at] - xr[at]));
                        yms    = std::max(yms, std::fabs(xr[at]));
                    }
                    e_st[st][k] = yms > 0 ? dms / yms : dms;
                    dm          = std::max(dm, dms);
                    ym          = std::max(ym, yms);
                }
                e_eh[k] = ym > 0 ? dm / ym : dm;
                vc_hc(w, xc.data(), lo.data(), mx.data());
                for (int32_t c = 0; c < 2; ++c) {
                    double dl = 0.0, yl = 0.0;
                    for (int32_t j = 0; j < 320; ++j) {
                        dl = std::max(dl, std::fabs(double(lowh[c][size_t(p) * 320 + size_t(j)]) - lo[size_t(j)]));
                        yl = std::max(yl, std::fabs(lo[size_t(j)]));
                    }
                    e_low[k * 2 + size_t(c)] = yl > 0 ? dl / yl : dl;
                }
                double dx = 0.0, yx = 0.0;
                for (int32_t f = 0; f < vc_D; ++f) {
                    const double y = vc_f16r(mx[size_t(f)]);
                    const double got = fn::quant::f16_to_f32(mix[0][size_t(p) * vc_D + size_t(f)]);
                    dx               = std::max(dx, std::fabs(got - y));
                    yx = std::max(yx, std::fabs(y));
                }
                e_mix[k] = yx > 0 ? dx / yx : dx;
            });
            const std::string line =
                "vcheck bulk localization: INFO: captured FP16 mixed row against R_b's on " + std::to_string(S.size()) +
                " positions: " + vc_stats(em) + "; bulk against R_b' (the captured row as operand): K " +
                vc_stats(mk) + "; V " + vc_stats(mv) + "; per stage on its own captured input: eh_proj " +
                vc_stats(e_eh) + " (streams 0..3 max " + std::to_string(pct(e_st[0], 100)) + " " +
                std::to_string(pct(e_st[1], 100)) + " " + std::to_string(pct(e_st[2], 100)) + " " +
                std::to_string(pct(e_st[3], 100)) + "); HC low (both cards) " + vc_stats(e_low) +
                "; HC mixed (FP16) " + vc_stats(e_mix);
            printf("prefill-test: %s\n", line.c_str());
            if (v.log != nullptr) {
                fprintf(v.log, "%s\n", line.c_str());
            }
        }
        const bool pb = pct(fk, 100) <= 5e-3 && pct(fv, 100) <= 5e-3;
        v.put("vcheck forced against FP64 (forced numerics)", pb,
              "K " + vc_stats(fk) + "; V " + vc_stats(fv) + " (bound 5e-3 every cell; worst at position " +
                  std::to_string(worst_f) + ")");
        std::vector<double> ak, av, ratio_v, ratio_k, design_ratio;
        for (int32_t c = 0; c < 2; ++c) {
            for (int64_t p = 0; p < N; ++p) {
                const size_t at = size_t(p) * vc_HD;
                ak.push_back(vc_rel(&db.k[c][at], &df.k[c][at]));
                av.push_back(vc_rel(&db.v[c][at], &df.v[c][at]));
                ratio_v.push_back(vc_ratio(&db.v[c][at], &df.v[c][at]));
                ratio_k.push_back(vc_ratio(&db.k[c][at], &df.k[c][at]));
            }
        }
        for (size_t k = 0; k < S.size(); ++k) {
            for (int32_t c = 0; c < 2; ++c) {
                design_ratio.push_back(vc_ratio(rb[k].V[c], rf[k].V[c]));
            }
        }
        bool pc = true;
        for (const double q : { 50.0, 99.0, 100.0 }) {
            pc = pc && pct(ak, q) <= 2.0 * pct(dk, q) + 4e-3 && pct(av, q) <= 2.0 * pct(dv, q) + 4e-3;
        }
        v.put("vcheck bulk against forced", pc,
              "all 8,736 cells x 2 cards: K " + vc_stats(ak) + "; V " + vc_stats(av) +
                  "; bound 2 x R_b against R_f on the sampled cells (K " + vc_stats(dk) + "; V " + vc_stats(dv) +
                  ") + 4e-3");
        char rs[400];
        snprintf(rs, sizeof(rs),
                 "||V_bulk|| / ||V_forced|| per cell: min %.5f p1 %.5f p50 %.5f p99 %.5f max %.5f (bound p50 in "
                 "[0.998, 1.002], p1 and p99 in [0.99, 1.01]); K's p1 %.5f p50 %.5f p99 %.5f; R_b / R_f V on the "
                 "sampled cells p1 %.5f p50 %.5f p99 %.5f",
                 pct(ratio_v, 0), pct(ratio_v, 1), pct(ratio_v, 50), pct(ratio_v, 99), pct(ratio_v, 100),
                 pct(ratio_k, 1), pct(ratio_k, 50), pct(ratio_k, 99), pct(design_ratio, 1), pct(design_ratio, 50),
                 pct(design_ratio, 99));
        const bool pr =
            std::fabs(pct(ratio_v, 50) - 1.0) <= 2e-3 && pct(ratio_v, 1) >= 0.99 && pct(ratio_v, 99) <= 1.01;
        v.put("vcheck V ratio", pr, rs);
        bool carry = true;
        for (int32_t c = 0; c < 2; ++c) {
            for (int32_t s = 0; s < 4; ++s) {
                carry =
                    carry && std::memcmp(&df.carry[c][size_t(s) * F],
                                         &H[(size_t(N - 1) * 4 + size_t(s)) * D + size_t(c) * F], size_t(F) * 4) == 0;
            }
            carry = carry && df.carry[c] == db.carry[c];
        }
        v.put("vcheck retained carry", carry,
              carry ? "both runs' retained carries bitwise H[8,735]'s own halves" : "a retained carry differs");
        if (sparse) {
            vc_index_checks(t, *m, w, H, P, S, two_blocks, df, db, mix, v);
        }
        rc = 0;
    }
    free_host();
    return rc;
}

} // namespace

int fnfl_prefill_test(const fnfl_target_args & a) {
    verdicts v;
    if (!a.out_dir.empty()) {
        v.log = fopen((a.out_dir + "/prefill-test.log").c_str(), "w");
        if (v.log == nullptr) {
            fprintf(stderr, "prefill-test: cannot write %s/prefill-test.log\n", a.out_dir.c_str());
            return 1;
        }
    }
    int rc = 2;
    if (a.mode == "plan") {
        rc = plan_mode(v);
    } else {
        std::vector<int32_t> pool;
        if (!read_ids(a.ids, pool)) {
            rc = 1;
        } else if (a.mode == "dry" || a.mode == "dry-eager") {
            rc = dry_mode(a, pool, v);
        } else if (a.mode == "model" || a.mode == "model-plumb") {
            rc = model_mode(a, pool, v, a.mode == "model-plumb");
        } else if (a.mode == "vcheck") {
            rc = vcheck_mode(a, v);
        } else {
            fprintf(stderr, "prefill-test: mode %s: plan, dry, dry-eager, vcheck or model\n", a.mode.c_str());
        }
    }
    printf("prefill-test: %s (%d checks)\n", rc == 0 && v.all ? "PASS" : "FAIL", v.n);
    if (v.log != nullptr) {
        fprintf(v.log, "%s (%d checks)\n", rc == 0 && v.all ? "PASS" : "FAIL", v.n);
        fclose(v.log);
    }
    return rc == 0 && v.all ? 0 : 1;
}
