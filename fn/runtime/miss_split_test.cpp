#include "fn/runtime/miss_split_test.h"

#include "fn/kernels/experts/experts_cpu.h"
#include "fn/runtime/miss_split.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fn::runtime {

namespace {

namespace sp = fn::runtime::split;

int32_t n_pass = 0, n_fail = 0;

void expect(bool ok, const char * what) {
    (ok ? n_pass : n_fail) += 1;
    if (!ok) {
        std::printf("ex-cpu-test: me5 %s: FAIL\n", what);
    }
}

// a window: its routing meta (window_meta, as the service builds it) and its misses (route order) under a resident set
struct window {
    int32_t rows = 0, n_u = 0, n_slots = 0, nm = 0;
    int32_t ids[5 * 10] = {};
    int32_t e[50] = {}, mask[50] = {}, base[50] = {};
    int32_t misses[50] = {};
};

void make(std::mt19937 & g, int32_t rows, int32_t span, double p_res, window & w) {
    w.rows = rows;
    std::uniform_int_distribution<int32_t> id(0, span - 1);
    for (int32_t b = 0; b < rows; ++b) {
        for (int32_t k = 0; k < 10; ++k) { // ten distinct IDs a row, as the router's top-10
            int32_t x = 0;
            bool dup = true;
            while (dup) {
                x = id(g);
                dup = std::find(w.ids + b * 10, w.ids + b * 10 + k, x) != w.ids + b * 10 + k;
            }
            w.ids[b * 10 + k] = x;
        }
    }
    w.n_u = fn::experts::cpu::window_meta(w.ids, rows, w.e, w.mask, w.base, w.n_slots);
    std::bernoulli_distribution res(p_res);
    w.nm = 0;
    for (int32_t u = 0; u < w.n_u; ++u) {
        if (!res(g)) {
            w.misses[w.nm++] = u;
        }
    }
}

// the CPU's row incidences under a choice
int32_t cpu_rows(const window & w, const uint8_t * to_gpu) {
    int32_t s = 0;
    for (int32_t i = 0; i < w.nm; ++i) {
        s += to_gpu[i] != 0 ? 0 : sp::rows_of(w.mask[w.misses[i]]);
    }
    return s;
}

// the fewest CPU row incidences any choice of n_gpu misses gives, by enumeration (nm <= 16)
int32_t brute_min(const window & w, int32_t n_gpu) {
    int32_t best = 1 << 30;
    for (uint32_t m = 0; m < (1u << w.nm); ++m) {
        if (__builtin_popcount(m) != n_gpu) {
            continue;
        }
        int32_t s = 0;
        for (int32_t i = 0; i < w.nm; ++i) {
            s += (m >> i & 1) != 0 ? 0 : sp::rows_of(w.mask[w.misses[i]]);
        }
        best = std::min(best, s);
    }
    return best;
}

// the same bound without the planner's sort: rows histogram, the lane taking the largest counts first
int32_t hist_min(const window & w, int32_t n_gpu) {
    int32_t h[6] = {}, total = 0;
    for (int32_t i = 0; i < w.nm; ++i) {
        const int32_t r = sp::rows_of(w.mask[w.misses[i]]);
        ++h[r];
        total += r;
    }
    int32_t left = n_gpu;
    for (int32_t r = 5; r >= 1 && left > 0; --r) {
        const int32_t t = std::min(left, h[r]);
        total -= t * r;
        left -= t;
    }
    return total;
}

// the rule before ME5: the last n_gpu misses in route order
void old_rule(const window & w, int32_t n_gpu, uint8_t * to_gpu) {
    for (int32_t i = 0; i < w.nm; ++i) {
        to_gpu[i] = i >= w.nm - n_gpu ? 1 : 0;
    }
}

void property() {
    std::mt19937 g(20261018u);
    int32_t windows = 0, brute = 0, changed = 0, moved_rows = 0;
    bool minimal = true, count = true, det = true, inputs = true, b1_same = true, ties_later = true;
    for (int32_t it = 0; it < 4000; ++it) {
        window w;
        const int32_t rows = 1 + it % 5;
        // small ID spans make duplicates (an expert on several rows); high residency keeps nm small for enumeration
        const int32_t span = it % 3 == 0 ? 16 : it % 3 == 1 ? 40 : 512;
        make(g, rows, span, it % 2 == 0 ? 0.7 : 0.2, w);
        ++windows;
        for (int32_t n_gpu = 0; n_gpu <= w.nm; ++n_gpu) {
            uint8_t a[50], b[50], o[50];
            int32_t msk[50], mis[50];
            std::memcpy(msk, w.mask, sizeof(msk));
            std::memcpy(mis, w.misses, sizeof(mis));
            const int32_t got = sp::choose_gpu(w.misses, w.nm, w.mask, n_gpu, a);
            sp::choose_gpu(w.misses, w.nm, w.mask, n_gpu, b);
            inputs = inputs && std::memcmp(msk, w.mask, sizeof(msk)) == 0 &&
                     std::memcmp(mis, w.misses, sizeof(mis)) == 0;
            int32_t chosen = 0;
            for (int32_t i = 0; i < w.nm; ++i) {
                chosen += a[i];
            }
            count = count && got == n_gpu && chosen == n_gpu;
            det = det && std::memcmp(a, b, size_t(w.nm)) == 0;
            const int32_t c = cpu_rows(w, a);
            minimal = minimal && c == hist_min(w, n_gpu);
            if (w.nm <= 16) {
                minimal = minimal && c == brute_min(w, n_gpu);
                ++brute;
            }
            old_rule(w, n_gpu, o);
            const bool same = std::memcmp(a, o, size_t(w.nm)) == 0;
            if (rows == 1) {
                b1_same = b1_same && same;
            }
            if (!same) {
                ++changed;
                moved_rows += cpu_rows(w, o) - c;
            }
            // ties: among chosen and unchosen of equal rows, every chosen one is later in route order
            for (int32_t i = 0; i < w.nm; ++i) {
                for (int32_t j = 0; j < w.nm; ++j) {
                    const bool tie = sp::rows_of(w.mask[w.misses[i]]) == sp::rows_of(w.mask[w.misses[j]]);
                    if (a[i] != 0 && a[j] == 0 && tie) {
                        ties_later = ties_later && i > j;
                    }
                }
            }
        }
    }
    expect(count, "split: exactly n_gpu chosen");
    expect(minimal, "split: the CPU's row incidences the fewest for the count (enumeration and histogram)");
    expect(det, "split: deterministic");
    expect(inputs, "split: inputs (masks, misses) unchanged");
    expect(b1_same, "split: B = 1 windows assigned as before ME5 (the last n_gpu in route order)");
    expect(ties_later, "split: ties to the later in route order");
    std::printf("ex-cpu-test: me5 split property: %d windows (B 1..5), %d (window, count) cases enumerated; %d cases "
                "assigned differently from the old rule, %d CPU row incidences fewer in all\n",
                windows, brute, changed, moved_rows);
}

void edges() {
    uint8_t t[50];
    int32_t mask[50] = {}, misses[50] = {};
    expect(sp::choose_gpu(misses, 0, mask, 3, t) == 0, "split: an empty window chooses none");
    for (int32_t i = 0; i < 6; ++i) {
        misses[i] = i;
        mask[i] = i == 2 ? 0x1f : i == 4 ? 0x3 : 0x1; // rows 1, 1, 5, 1, 2, 1
    }
    expect(sp::choose_gpu(misses, 6, mask, 0, t) == 0 && std::count(t, t + 6, 1) == 0, "split: all-CPU (n_gpu 0)");
    expect(sp::choose_gpu(misses, 6, mask, 6, t) == 6 && std::count(t, t + 6, 1) == 6, "split: all-GPU (n_gpu nm)");
    expect(sp::choose_gpu(misses, 6, mask, 9, t) == 6, "split: n_gpu above nm clamped");
    expect(sp::choose_gpu(misses, 6, mask, -2, t) == 0, "split: a negative n_gpu clamped");
    sp::choose_gpu(misses, 6, mask, 2, t);
    expect(t[2] == 1 && t[4] == 1 && std::count(t, t + 6, 1) == 2, "split: the 5-row and 2-row misses to the lane");
    sp::choose_gpu(misses, 6, mask, 3, t);
    expect(t[2] == 1 && t[4] == 1 && t[5] == 1, "split: then the latest 1-row miss");
    // misses indexing a subset of the window's experts (hits between them)
    int32_t sub[3] = {1, 2, 5};
    sp::choose_gpu(sub, 3, mask, 1, t);
    expect(t[0] == 0 && t[1] == 1 && t[2] == 0, "split: misses as indices into the window's experts");
    // bounded: a count above the window's maximum is clamped to 50
    int32_t big[60], bmask[60];
    uint8_t bt[60] = {};
    for (int32_t i = 0; i < 60; ++i) {
        big[i] = i < 50 ? i : 0;
        bmask[i] = 1;
    }
    expect(sp::choose_gpu(big, 60, bmask, 60, bt) == sp::max_misses && bt[50] == 0, "split: bounded at 50 misses");
    // the routing meta it reads: duplicate IDs across rows give one expert with several rows; invalid IDs skipped
    int32_t ids[20], e[50], m[50], base[50], n_slots = 0;
    for (int32_t k = 0; k < 10; ++k) {
        ids[k] = k;
        ids[10 + k] = k < 5 ? k : 100 + k;
    }
    ids[19] = 600; // outside [0, 512)
    const int32_t n_u = fn::experts::cpu::window_meta(ids, 2, e, m, base, n_slots);
    expect(n_u == 14 && sp::rows_of(m[0]) == 2 && sp::rows_of(m[9]) == 1, "split: duplicate IDs one expert, rows 2");
}

} // namespace

bool miss_split_test() {
    n_pass = n_fail = 0;
    property();
    edges();
    std::printf("ex-cpu-test: me5 miss split: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0;
}

} // namespace fn::runtime
