// ME11 CPU check (no GPU, no model): the GDN selector protocol through the real host ledger (fn/runtime/versions.cpp)
// and the one mapping (fn/kernels/round/version.h), against a symbolic model of what each card's banks, record sets
// and history slots hold. The CUDA copies versions.cpp makes are stubbed below onto host memory, so no device is
// touched; the device side (gdn_core, the scan, the commit kernel) is modelled by calling the same version.h functions
// the kernels call.
//
// The model: a bank holds "the state after p tokens"; a record set holds the steps from position q (w of them, step t
// taking q + t to q + t + 1); a history slot holds the history after p tokens. A card's committed GDN state is position
// bank(b) + k when k = 0 or set r starts at bank(b) and holds at least k steps, else invalid. After every operation
// both cards' committed GDN state and history must be the ledger's N, and no round may have written the bank or set its
// selector committed.
//
// Also the record writers: gdn_core's write conditions and indices (gdn.cu, restated here) over its whole grid at
// B = 1..5, every element the replay reads written exactly once and nothing else in the set.
//
// Build and run (Carlin, CPU only):
//   g++-14 -std=c++17 -O2 -Wall -Wextra -Werror -I<engine> -I/usr/local/cuda-12.8/include
//       fn/tools/me11_ledger_check.cpp fn/runtime/versions.cpp -o me11-ledger-check
//   ./me11-ledger-check

#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/qsa/qsa.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/round/version.h"
#include "fn/runtime/versions.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---- stubs: the host copies versions.cpp makes, onto host memory; the qsa ledger helpers (qsa.cu's host code) ----

extern "C" {
cudaError_t cudaSetDevice(int) {
    return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t n, cudaMemcpyKind, cudaStream_t) {
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return cudaSuccess;
}
const char * cudaGetErrorString(cudaError_t) {
    return "stub";
}
}

namespace fn::qsa {
int32_t blocks_complete(int32_t n_cells) {
    return n_cells / ratio;
}
void invalidate_from(summary_index & idx, int32_t first) {
    if (first < idx.n_valid) {
        idx.n_valid = first < 0 ? 0 : first;
    }
}
bool mark_updated(summary_index & idx, int32_t first, int32_t count) {
    if (first < 0 || count < 1 || first > idx.n_valid) {
        return false;
    }
    idx.n_valid = std::max(idx.n_valid, first + count);
    return true;
}
} // namespace fn::qsa

namespace {

namespace rd = fn::round;
using fn::runtime::ledger;
using fn::runtime::slot_versions;

constexpr int64_t bad = -1000000;

struct rec_set {
    int64_t q = bad;
    int32_t w = 0;
};

struct card_model {
    uint32_t words[2] = {0, 0}; // the "device" cur and GDN selector (rd::state's adjacent words)
    int64_t bank[2] = {0, 0};
    rec_set set[2];
    int64_t hist[rd::n_version] = {0, bad, bad, bad, bad, bad};

    int64_t committed_gdn() const {
        const uint32_t g = words[1];
        if (!rd::gdn_valid(g)) {
            return bad;
        }
        const int64_t p = bank[rd::gdn_bank(g)];
        const int32_t k = rd::gdn_steps(g);
        const rec_set & s = set[rd::gdn_set(g)];
        if (p == bad || (k > 0 && (s.q != p || s.w < k))) {
            return bad;
        }
        return p + k;
    }
    int64_t committed_hist() const { return hist[rd::version_slot(words[0], 0)]; }
};

struct harness {
    card_model c[2];
    ledger l;
    slot_versions v;
    int64_t ops = 0, checks = 0, failures = 0;
    int32_t k_seen = 0, j_seen = 0, orient = 0, fails_seen = 0, prompt_seen = 0, ck_seen = 0;

    harness() {
        for (int32_t r = 0; r < 2; ++r) {
            v.bind(r, 0, nullptr, &c[r].words[0], &c[r].words[1]);
        }
    }

    void fail(const std::string & what) {
        ++failures;
        if (failures <= 20) {
            printf("me11-ledger-check: FAIL at op %lld: %s (N %lld, ledger cur %u gdn %u; card 0 words %u %u, "
                   "card 1 %u %u)\n",
                   static_cast<long long>(ops), what.c_str(), static_cast<long long>(l.n), l.cur, l.gdn,
                   c[0].words[0], c[0].words[1], c[1].words[0], c[1].words[1]);
        }
    }

    void check(const char * what) {
        ++checks;
        std::string err;
        if (!slot_versions::check(l, err)) {
            fail(std::string(what) + ": ledger check: " + err);
        }
        for (int32_t r = 0; r < 2; ++r) {
            if (c[r].words[0] != l.cur || c[r].words[1] != l.gdn) {
                fail(std::string(what) + ": card " + std::to_string(r) + "'s words are not the ledger's");
            }
            if (c[r].committed_gdn() != l.n) {
                fail(std::string(what) + ": card " + std::to_string(r) + "'s committed GDN state is at " +
                     std::to_string(c[r].committed_gdn()) + ", not N");
            }
            if (c[r].committed_hist() != l.n) {
                fail(std::string(what) + ": card " + std::to_string(r) + "'s committed history is not N");
            }
        }
        orient |= 1 << (rd::gdn_bank(l.gdn) + 2 * rd::gdn_set(l.gdn));
    }

    void append(int32_t n) {
        for (int32_t i = 0; i < n; ++i) {
            l.tok.push_back(static_cast<int32_t>(l.tok.size() % 248320));
        }
        l.u += n;
        l.prompt_len = static_cast<int64_t>(l.tok.size());
    }

    // a target round of B rows (gdn_core and the convolution epilogue on each card, the device's selector reads),
    // committing k on each card whose words are clear (the commit kernel's rule); fail_mask: the cards whose commit
    // sees an error. Returns false when the round failed (the host restored).
    bool round(int32_t B, int32_t k, int32_t fail_mask) {
        ++ops;
        v.begin_step(l, true);
        const int64_t p = l.n;
        for (int32_t r = 0; r < 2; ++r) {
            card_model & m = c[r];
            const uint32_t g = m.words[1];
            const int32_t b = rd::gdn_bank(g), s = rd::gdn_set(g);
            if (m.committed_gdn() != p) {
                fail("round: card " + std::to_string(r) + " reads a GDN state not at N_before");
            }
            // the kernel writes the other bank (logical version 1's) and the other set, never the committed ones
            const int32_t wb = rd::gdn_bank_of(g, 1), ws = 1 - s;
            if (wb == b || ws == s) {
                fail("round: writes the committed bank or set");
            }
            m.bank[wb] = p;
            m.set[ws] = {p, B};
            for (int32_t t = 1; t <= B; ++t) {
                m.hist[rd::version_slot(m.words[0], t)] = p + t;
            }
        }
        uint32_t reported[2], reported_gdn[2];
        for (int32_t r = 0; r < 2; ++r) {
            card_model & m = c[r];
            if (((fail_mask >> r) & 1) == 0 && k >= 1) { // commit_kernel: only with clear error words
                m.words[0] = static_cast<uint32_t>(rd::version_slot(m.words[0], k));
                m.words[1] = rd::gdn_after_round(m.words[1], k);
            }
            reported[r] = m.words[0];
            reported_gdn[r] = m.words[1];
        }
        std::string err;
        if (fail_mask != 0) {
            ++fails_seen;
            if (!v.restore(l, err)) {
                fail("restore: " + err);
            }
            check("after a failed round");
            return false;
        }
        if (!v.committed(l, k, reported, reported_gdn, err)) {
            fail("committed: " + err);
            return false;
        }
        l.n += k;
        l.u -= k;
        slot_versions::settle(l);
        k_seen |= 1 << k;
        check("after a commit");
        return true;
    }

    // a bulk prompt chunk of rows in two halves (the scan: half 0 from version 0 into the other bank, half 1 in place)
    bool prompt(int32_t rows, bool fail_it) {
        ++ops;
        v.begin_step(l, true);
        const int64_t p = l.n;
        const int32_t h0 = rows / 2;
        for (int32_t r = 0; r < 2; ++r) {
            card_model & m = c[r];
            const uint32_t g = m.words[1];
            if (m.committed_gdn() != p) {
                fail("prompt: reads a GDN state not at N_before");
            }
            const int32_t out = rd::gdn_bank_of(g, 1);
            if (out == rd::gdn_bank(g)) {
                fail("prompt: writes the committed bank");
            }
            m.bank[out] = p + h0;   // half 0
            m.bank[rd::gdn_bank_of(g, 2)] = p + rows; // half 1, in place (version 1's bank is version 2's)
            m.hist[rd::version_slot(m.words[0], 1)] = p + h0;
            m.hist[rd::version_slot(m.words[0], 2)] = p + rows;
        }
        std::string err;
        if (fail_it) { // a chunk that fails publishes nothing (no advance); the step is restored
            ++fails_seen;
            if (!v.restore(l, err)) {
                fail("prompt restore: " + err);
            }
            check("after a failed chunk");
            return false;
        }
        uint32_t reported[2], reported_gdn[2];
        for (int32_t r = 0; r < 2; ++r) { // advance(cur, gdn, 2, prompt) on each card
            card_model & m = c[r];
            m.words[0] = static_cast<uint32_t>(rd::version_slot(m.words[0], 2));
            m.words[1] = rd::gdn_after_prompt(m.words[1]);
            reported[r] = m.words[0];
            reported_gdn[r] = m.words[1];
        }
        if (!v.committed_prompt(l, 2, reported, reported_gdn, err)) {
            fail("committed_prompt: " + err);
            return false;
        }
        l.n += rows;
        l.u -= rows;
        slot_versions::settle(l);
        ++prompt_seen;
        check("after a chunk");
        return true;
    }

    void cut(int32_t j, int64_t n0) {
        ++ops;
        std::string err;
        if (!v.cut(l, j, n0 + j, err)) {
            fail("cut: " + err);
            return;
        }
        j_seen |= 1 << j;
        check("after a cut");
    }

    // a checkpoint (prefiller::save: the selector's bank, its live records and k) and its restore after a rewind
    // (prefiller::load into bank 0 and set 0, target::restored: the selector (0, 0, k)); the history's version 0
    struct checkpoint {
        int64_t bank = bad, hist = bad, n = 0;
        rec_set set;
        int32_t k = 0;
        std::vector<int32_t> tok;
    };
    checkpoint save() const {
        checkpoint s;
        const card_model & m = c[0];
        s.bank = m.bank[rd::gdn_bank(m.words[1])];
        s.set = m.set[rd::gdn_set(m.words[1])];
        s.k = rd::gdn_steps(m.words[1]);
        s.hist = m.committed_hist();
        s.n = l.n;
        s.tok.assign(l.tok.begin(), l.tok.begin() + l.n);
        return s;
    }
    void restore_checkpoint(const checkpoint & s) {
        ++ops;
        std::string err;
        if (!v.reset(l, err)) { // rewind: cur 0, the selector 0, the ledger emptied; the state objects kept
            fail("reset: " + err);
        }
        for (card_model & m : c) {
            for (int64_t & h : m.hist) {
                h = bad;
            }
            m.bank[1] = bad;
            m.set[1] = {};
            m.bank[0] = s.bank;
            m.set[0] = s.set;
            m.hist[0] = s.hist;
        }
        if (!v.set_gdn(l, rd::gdn_word(0, 0, s.k), err)) {
            fail("set_gdn: " + err);
        }
        l.tok = s.tok;
        l.n = s.n;
        l.prompt_len = s.n;
        slot_versions::settle(l);
        ++ck_seen;
        check("after a checkpoint restore");
    }

    void reset() {
        ++ops;
        std::string err;
        if (!v.reset(l, err)) {
            fail("reset: " + err);
        }
        for (card_model & m : c) {
            m = card_model();
        }
        check("after a reset");
    }
};

// gdn_core's record writes at B (gdn.cu): block (h, s), thread tid; returns the writes per float of one set
std::vector<int32_t> record_writes(int32_t B) {
    namespace gd = fn::gdn;
    std::vector<int32_t> n(size_t(gd::rec_set_floats), 0);
    for (int32_t blk = 0; blk < gd::n_head * gd::n_split; ++blk) {
        const int32_t h = blk / gd::n_split, s = blk % gd::n_split;
        const int32_t row0 = s < 2 ? 22 * s : 44 + 21 * (s - 2), n_rows = s < 2 ? 22 : 21;
        for (int32_t tid = 0; tid < 22 * 32; ++tid) {
            const int32_t warp = tid / 32, lane = tid % 32;
            // the decay: warp 13 (gate_warp), lane < B, split 0
            if (warp == 13 && lane < B && s == 0) {
                ++n[size_t(lane) * gd::rec_step_floats + gd::rec_decay + h];
            }
            // the normalized k: split 0 of head h < 8, tid < B * 128
            if (h < gd::n_group && s == 0 && tid < B * gd::head_dim) {
                ++n[size_t(tid / gd::head_dim) * gd::rec_step_floats + gd::rec_k + h * gd::head_dim +
                    tid % gd::head_dim];
            }
            // delta: lane 0 of each real row warp, every step
            const bool has_row = warp < n_rows;
            const int32_t j_row = row0 + (has_row ? warp : n_rows - 1);
            for (int32_t b = 0; b < B; ++b) {
                if (has_row && lane == 0) {
                    ++n[size_t(b) * gd::rec_step_floats + gd::rec_delta + h * gd::head_dim + j_row];
                }
            }
        }
    }
    return n;
}

// every element the replay reads (steps < B) written once, nothing else
bool record_writers_ok(int32_t & bad_elems) {
    namespace gd = fn::gdn;
    bad_elems = 0;
    for (int32_t B = 1; B <= gd::max_rows; ++B) {
        const std::vector<int32_t> n = record_writes(B);
        std::vector<int32_t> want(n.size(), 0);
        for (int32_t t = 0; t < B; ++t) {
            int32_t * st = want.data() + int64_t(t) * gd::rec_step_floats;
            for (int32_t i = 0; i < gd::n_group * gd::head_dim; ++i) {
                st[gd::rec_k + i] = 1;
            }
            for (int32_t h = 0; h < gd::n_head; ++h) {
                st[gd::rec_decay + h] = 1;
            }
            for (int32_t i = 0; i < gd::n_head * gd::head_dim; ++i) {
                st[gd::rec_delta + i] = 1;
            }
        }
        for (size_t i = 0; i < n.size(); ++i) {
            bad_elems += n[i] != want[i];
        }
    }
    return bad_elems == 0;
}

} // namespace

int main() {
    harness h;
    h.check("start");
    // every k <= B and every cut j <= k (state-test's sweep), with a continuation round after each
    for (int32_t B = 1; B <= rd::max_rows; ++B) {
        for (int32_t k = 0; k <= B; ++k) {
            for (int32_t j = 0; j <= k; ++j) {
                h.append(B);
                const int64_t n0 = h.l.n;
                h.round(B, k, 0);
                h.cut(j, n0);
                if (j > 0) {
                    h.cut(j - 1, n0); // a repeat cut, lower
                }
                const int32_t b2 = 1 + (B + k + j) % rd::max_rows;
                h.l.tok.resize(size_t(h.l.n)); // the cut tokens dropped; fresh ones for the continuation
                h.l.u = 0;
                h.l.prompt_len = h.l.n;
                h.append(b2);
                h.round(b2, b2, 0);
            }
        }
    }
    // one-card and both-card failures after the other card committed, then the rows again
    for (const int32_t mask : {1, 2, 3}) {
        for (int32_t k = 0; k <= 4; ++k) {
            h.append(4);
            h.round(4, k, mask);
            h.round(4, 4, 0);
        }
    }
    // prompt chunks (two halves), a failed chunk, a cut to the chunk's start, then decode rounds
    h.append(300);
    h.prompt(300, false);
    h.append(100);
    h.prompt(100, true);
    h.prompt(100, false);
    {
        const int64_t n0 = h.l.n - 100;
        h.cut(0, n0); // the chunk keeps no rows: j = 0 only
        h.l.tok.resize(size_t(h.l.n));
        h.l.u = 0;
        h.l.prompt_len = h.l.n;
    }
    // a short read (forced rounds), a checkpoint with live records, more work, its restore, a bulk chunk from it
    h.append(5);
    h.round(5, 5, 0);
    h.append(3);
    h.round(3, 3, 0);
    const harness::checkpoint ck = h.save();
    const int32_t ck_k = ck.k;
    h.append(40);
    h.prompt(40, false);
    h.append(2);
    h.round(2, 2, 0);
    h.restore_checkpoint(ck);
    h.append(64);
    h.prompt(64, false);
    h.append(4);
    h.round(4, 2, 0);
    h.l.tok.resize(size_t(h.l.n));
    h.l.u = 0;
    h.l.prompt_len = h.l.n;
    // the ready state (save_ready/restore_ready: the same normalization)
    const harness::checkpoint ready = h.save();
    h.append(3);
    h.round(3, 3, 0);
    h.restore_checkpoint(ready);
    // a long random sequence: rounds with random k, cuts, failures, chunks, checkpoints, resets
    std::mt19937_64 rng(0x4d4531ull);
    std::vector<harness::checkpoint> book;
    for (int32_t i = 0; i < 20000; ++i) {
        const int32_t op = static_cast<int32_t>(rng() % 100);
        if (op < 55) {
            const int32_t B = 1 + static_cast<int32_t>(rng() % rd::max_rows);
            const int32_t k = static_cast<int32_t>(rng() % (B + 1));
            const int32_t mask = rng() % 10 == 0 ? 1 + static_cast<int32_t>(rng() % 3) : 0;
            h.append(B);
            const int64_t n0 = h.l.n;
            if (h.round(B, k, mask) && rng() % 3 == 0) {
                h.cut(static_cast<int32_t>(rng() % (k + 1)), n0);
            }
            h.l.tok.resize(size_t(h.l.n));
            h.l.u = 0;
            h.l.prompt_len = h.l.n;
        } else if (op < 70) {
            const int32_t rows = 2 + static_cast<int32_t>(rng() % 200);
            h.append(rows);
            h.prompt(rows, rng() % 8 == 0);
            h.l.tok.resize(size_t(h.l.n));
            h.l.u = 0;
            h.l.prompt_len = h.l.n;
        } else if (op < 82) {
            book.push_back(h.save());
        } else if (op < 92 && !book.empty()) {
            h.restore_checkpoint(book[rng() % book.size()]);
            h.l.u = 0;
        } else if (op < 93) {
            h.reset();
            book.clear();
        } else {
            // bootstrap and draft rounds: no GDN work, k = 0, the words unchanged
            uint32_t rep[2] = {h.c[0].words[0], h.c[1].words[0]}, rep_g[2] = {h.c[0].words[1], h.c[1].words[1]};
            std::string err;
            h.v.begin_step(h.l, false);
            if (!h.v.committed(h.l, 0, rep, rep_g, err)) {
                h.fail("bootstrap commit: " + err);
            }
            h.check("after a k = 0 round");
        }
    }
    // the selector's word range: every valid word round-trips and stays valid under each transition
    int32_t words_bad = 0;
    for (uint32_t g = 0; g < 64; ++g) {
        const bool valid = rd::gdn_valid(g);
        if (valid != (g < 24) || rd::gdn_steps(g) != (valid ? int32_t(g >> 2) : 0)) {
            ++words_bad;
        }
        if (valid) {
            for (int32_t k = 1; k <= rd::gdn_max_steps; ++k) {
                const uint32_t a = rd::gdn_after_round(g, k);
                words_bad += !rd::gdn_valid(a) || rd::gdn_bank(a) == rd::gdn_bank(g) ||
                             rd::gdn_set(a) == rd::gdn_set(g) || rd::gdn_steps(a) != k;
            }
            const uint32_t p = rd::gdn_after_prompt(g);
            words_bad += !rd::gdn_valid(p) || rd::gdn_bank(p) == rd::gdn_bank(g) || rd::gdn_set(p) != rd::gdn_set(g) ||
                         rd::gdn_steps(p) != 0;
            words_bad += rd::gdn_word(rd::gdn_bank(g), rd::gdn_set(g), rd::gdn_steps(g)) != g;
        }
    }
    std::string err;
    ledger bogus;
    bogus.gdn = rd::gdn_word(0, 0, 6);
    const bool refuses = !slot_versions::check(bogus, err) && !h.v.set_gdn(h.l, 24u, err);
    printf("me11-ledger-check: %lld operations, %lld checks, %lld failures\n", static_cast<long long>(h.ops),
           static_cast<long long>(h.checks), static_cast<long long>(h.failures));
    printf("me11-ledger-check: commits k seen 0..5: %s; cuts j seen 0..5: %s; bank/set orientations %d of 4; failed "
           "rounds and chunks %d; chunks %d; checkpoint restores %d (the first with %d live records)\n",
           h.k_seen == 63 ? "all" : "NOT ALL", h.j_seen == 63 ? "all" : "NOT ALL", __builtin_popcount(h.orient),
           h.fails_seen, h.prompt_seen, h.ck_seen, ck_k);
    printf("me11-ledger-check: selector words 0..63: %d wrong; out-of-contract word refused by check and set_gdn: %s\n",
           words_bad, refuses ? "yes" : "NO");
    int32_t bad_elems = 0;
    const bool writers = record_writers_ok(bad_elems);
    printf("me11-ledger-check: record writers at B = 1..5 (gdn_core's grid): every replayed element written once, "
           "nothing else: %s (%d elements wrong)\n",
           writers ? "yes" : "NO", bad_elems);
    const bool pass = writers && h.failures == 0 && h.k_seen == 63 && h.j_seen == 63 &&
                      __builtin_popcount(h.orient) == 4 && words_bad == 0 && refuses && ck_k > 0;
    printf("me11-ledger-check: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
