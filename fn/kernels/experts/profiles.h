#pragma once

// TTTS3 item 2: the model files' expert profiles. A profile is one model file's expert instances: each (routed
// gate/up, shared gate, shared up) triple and each (routed down, shared down) pair its layers use, built from the
// X-macro lists of experts_inst.h, the one place the instances are written. The IQ3_XXS two-card file and the Q8_0
// draft share EX2's deployed lists (the draft is its Q8_0 entries); UD-Q4_K_XL and the Q2_0 draft have their own.
//
// The loader resolves each layer's formats against the compiled profiles at load (profile_of) and refuses a
// combination none of them holds, by name. The gate/up families take their shared pairs from the profiles
// (family_pairs: the pairs of the triples whose routed format is the family's, in the order the profiles list them),
// so the family body's pair switch is the profiles' and not a hand-written list. Adding a format is one codec
// (codec.cuh, codec_down.cuh) and its profile entries.

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_inst.h"

#include <cstdint>

namespace fn::experts {

struct gu_inst {
    fmt            f;     // routed gate/up
    fn::dense::fmt g, u;  // shared gate, shared up
    int32_t        layer;
};

struct dn_inst {
    fmt     f, s;  // routed down, shared down
    int32_t layer;
};

constexpr int32_t profile_max_gu = 32;  // the IQ3_XXS file's 30 triples, with room
constexpr int32_t profile_max_dn = 8;

struct gu_list {
    int32_t n                 = 0;
    gu_inst e[profile_max_gu] = {};
};

struct dn_list {
    int32_t n                 = 0;
    dn_inst e[profile_max_dn] = {};
};

struct expert_profile {
    const char * name;
    gu_list      gu;
    dn_list      dn;
};

namespace profile_src {
#define FN_EXPERTS_PROFILE_GU(rf, sgf, suf, layer) { fmt::rf, fn::dense::fmt::sgf, fn::dense::fmt::suf, layer },
#define FN_EXPERTS_PROFILE_DN(rf, sf, layer)       { fmt::rf, fmt::sf, layer },
inline constexpr gu_inst deployed_gu[] = { FN_EXPERTS_GATE_UP_INSTANCES(FN_EXPERTS_PROFILE_GU) };
inline constexpr dn_inst deployed_dn[] = { FN_EXPERTS_DOWN_INSTANCES(FN_EXPERTS_PROFILE_DN) };
inline constexpr gu_inst q4_gu[]       = { FN_EXPERTS_Q4_GATE_UP_INSTANCES(FN_EXPERTS_PROFILE_GU) };
inline constexpr dn_inst q4_dn[]       = { FN_EXPERTS_Q4_DOWN_INSTANCES(FN_EXPERTS_PROFILE_DN) };
inline constexpr gu_inst q2d_gu[]      = { FN_EXPERTS_DRAFT_Q2_GATE_UP_INSTANCES(FN_EXPERTS_PROFILE_GU) };
inline constexpr dn_inst q2d_dn[]      = { FN_EXPERTS_DRAFT_Q2_DOWN_INSTANCES(FN_EXPERTS_PROFILE_DN) };
#undef FN_EXPERTS_PROFILE_GU
#undef FN_EXPERTS_PROFILE_DN

// a source list's entries whose routed format is (draft) or is not (!draft) Q8_0: EX2's deployed lists hold the
// IQ3_XXS file's layers 0-47 and the Q8_0 draft's layer 48
template <int N> constexpr gu_list gu_of(const gu_inst (&src)[N], int32_t draft) {
    gu_list l;
    for (int32_t i = 0; i < N; ++i) {
        if (draft < 0 || (src[i].f == fmt::q8_0) == (draft == 1)) {
            l.e[l.n++] = src[i];
        }
    }
    return l;
}

template <int N> constexpr dn_list dn_of(const dn_inst (&src)[N], int32_t draft) {
    dn_list l;
    for (int32_t i = 0; i < N; ++i) {
        if (draft < 0 || (src[i].f == fmt::q8_0) == (draft == 1)) {
            l.e[l.n++] = src[i];
        }
    }
    return l;
}
}  // namespace profile_src

// the compiled profiles; draft: 1 the Q8_0 entries, 0 the others, -1 all
inline constexpr expert_profile expert_profiles[] = {
    { "IQ3_XXS",    profile_src::gu_of(profile_src::deployed_gu, 0),  profile_src::dn_of(profile_src::deployed_dn, 0)  },
    { "UD-Q4_K_XL", profile_src::gu_of(profile_src::q4_gu,       -1), profile_src::dn_of(profile_src::q4_dn,       -1) },
    { "Q8_0 draft", profile_src::gu_of(profile_src::deployed_gu, 1),  profile_src::dn_of(profile_src::deployed_dn, 1)  },
    { "Q2_0 draft", profile_src::gu_of(profile_src::q2d_gu,      -1), profile_src::dn_of(profile_src::q2d_dn,      -1) },
};
inline constexpr int32_t n_expert_profiles = int32_t(sizeof(expert_profiles) / sizeof(expert_profiles[0]));
static_assert(expert_profiles[0].gu.n + expert_profiles[2].gu.n == n_gate_up_inst &&
                  expert_profiles[0].dn.n + expert_profiles[2].dn.n == n_down_inst &&
                  expert_profiles[1].gu.n == n_q4_gate_up_inst && expert_profiles[1].dn.n == n_q4_down_inst &&
                  expert_profiles[3].gu.n == n_draft_q2_gate_up_inst && expert_profiles[3].dn.n == n_draft_q2_down_inst,
              "the profiles hold every listed instance");

constexpr bool profile_has_gu(const expert_profile & p, fmt f, fn::dense::fmt g, fn::dense::fmt u) {
    for (int32_t i = 0; i < p.gu.n; ++i) {
        if (p.gu.e[i].f == f && p.gu.e[i].g == g && p.gu.e[i].u == u) {
            return true;
        }
    }
    return false;
}

constexpr bool profile_has_dn(const expert_profile & p, fmt f, fmt s) {
    for (int32_t i = 0; i < p.dn.n; ++i) {
        if (p.dn.e[i].f == f && p.dn.e[i].s == s) {
            return true;
        }
    }
    return false;
}

// the first compiled profile holding a layer's gate/up triple and down pair, or -1 (the loader's check, once a layer)
constexpr int32_t profile_of(fmt f, fn::dense::fmt g, fn::dense::fmt u, fmt d, fmt s) {
    for (int32_t i = 0; i < n_expert_profiles; ++i) {
        if (profile_has_gu(expert_profiles[i], f, g, u) && profile_has_dn(expert_profiles[i], d, s)) {
            return i;
        }
    }
    return -1;
}

// every compiled profile's gate/up triples and down pairs as flat sets, built at compile time: the launches' checks
// are one indexed read (TTTS3 Amendment 1 line 3: no search over the profiles a launch)
constexpr int32_t n_fmts  = int32_t(fmt::q5_1) + 1;
constexpr int32_t n_dfmts = int32_t(fn::dense::fmt::f32) + 1;
static_assert(n_fmts == 11 && n_dfmts == 10, "the format enums (experts.h, dense.h)");

struct compiled_sets {
    bool gu[n_fmts][n_dfmts][n_dfmts] = {};
    bool dn[n_fmts][n_fmts]           = {};
};

constexpr compiled_sets compiled_of() {
    compiled_sets c;
    for (int32_t i = 0; i < n_expert_profiles; ++i) {
        const expert_profile & p = expert_profiles[i];
        for (int32_t k = 0; k < p.gu.n; ++k) {
            c.gu[int32_t(p.gu.e[k].f)][int32_t(p.gu.e[k].g)][int32_t(p.gu.e[k].u)] = true;
        }
        for (int32_t k = 0; k < p.dn.n; ++k) {
            c.dn[int32_t(p.dn.e[k].f)][int32_t(p.dn.e[k].s)] = true;
        }
    }
    return c;
}

inline constexpr compiled_sets compiled = compiled_of();

// a compiled profile's gate/up triple or down pair (the launches' check)
constexpr bool compiled_gu(fmt f, fn::dense::fmt g, fn::dense::fmt u) {
    return int32_t(f) >= 0 && int32_t(f) < n_fmts && int32_t(g) >= 0 && int32_t(g) < n_dfmts && int32_t(u) >= 0 &&
           int32_t(u) < n_dfmts && compiled.gu[int32_t(f)][int32_t(g)][int32_t(u)];
}

constexpr bool compiled_dn(fmt f, fmt s) {
    return int32_t(f) >= 0 && int32_t(f) < n_fmts && int32_t(s) >= 0 && int32_t(s) < n_fmts &&
           compiled.dn[int32_t(f)][int32_t(s)];
}

// ---- a gate/up family's shared pairs --------------------------------------------------------------------------------
// The (shared gate, shared up) pairs of every profiled triple whose routed format is one of a family's, each once, in
// the order the profiles list them; key = gate x 16 + up (the family body's switch), unused slots -1 - i (no pair's)
constexpr int32_t gu_max_pairs = 12;

struct gu_pairs {
    int32_t        n                 = 0;
    fn::dense::fmt g[gu_max_pairs]   = {};
    fn::dense::fmt u[gu_max_pairs]   = {};
    int32_t        key[gu_max_pairs] = {};
};

constexpr int32_t pair_key(fn::dense::fmt g, fn::dense::fmt u) {
    return int32_t(g) * 16 + int32_t(u);
}

template <fmt... Fs> constexpr gu_pairs family_pairs() {
    gu_pairs  p;
    const fmt fs[] = { Fs... };
    for (int32_t i = 0; i < n_expert_profiles; ++i) {
        const gu_list & l = expert_profiles[i].gu;
        for (int32_t k = 0; k < l.n; ++k) {
            bool mine = false, seen = false;
            for (const fmt f : fs) {
                mine = mine || l.e[k].f == f;
            }
            const int32_t key = pair_key(l.e[k].g, l.e[k].u);
            for (int32_t q = 0; q < p.n; ++q) {
                seen = seen || p.key[q] == key;
            }
            if (mine && !seen) {
                p.g[p.n]   = l.e[k].g;
                p.u[p.n]   = l.e[k].u;
                p.key[p.n] = key;
                ++p.n;
            }
        }
    }
    for (int32_t q = p.n; q < gu_max_pairs; ++q) {
        p.key[q] = -1 - q;
    }
    return p;
}

// p's pairs are exactly keys, in order (the IQ bodies' written pair lists against their profiles)
template <int N> constexpr bool pairs_are(const gu_pairs & p, const int32_t (&keys)[N]) {
    bool ok = p.n == N;
    for (int32_t q = 0; q < N && ok; ++q) {
        ok = p.key[q] == keys[q];
    }
    return ok;
}

constexpr bool pairs_have(const gu_pairs & p, fn::dense::fmt f) {
    for (int32_t q = 0; q < p.n; ++q) {
        if (p.g[q] == f || p.u[q] == f) {
            return true;
        }
    }
    return false;
}

}  // namespace fn::experts
