#include "fn/model/ladder_test.h"

#include "fn/model/ladder.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fn::model {
namespace {

int32_t g_bad = 0;

void put(const std::string & what, bool ok, const std::string & detail = std::string()) {
    printf("ladder-test: %s: %s%s%s\n", what.c_str(), ok ? "PASS" : "FAIL", detail.empty() ? "" : ": ", detail.c_str());
    g_bad += ok ? 0 : 1;
}

// a card of 80 SMs whose six tiles hold 4, 4, 2, 2, 1 and 1 CTAs an SM; Q8_0 weights (34 bytes a 32)
lad_model card(int64_t ws_floats = 0, int32_t ws_count = 0) {
    lad_model m;
    m.sm_count        = 80;
    const int32_t c[] = { 4, 4, 2, 2, 1, 1 };
    for (int32_t t = 0; t < fn::tc::lad_n_tiles; ++t) {
        m.ctas[t] = c[t];
    }
    m.b_bytes   = 34.0 / 32;
    m.ws_floats = ws_floats;
    m.ws_count  = ws_count;
    return m;
}

const lad_cand * find(const lad_ranking & r, int32_t t, int32_t s) {
    for (const lad_cand & c : r.ranked) {
        if (c.cfg.tile == t && c.cfg.splits == s) {
            return &c;
        }
    }
    return nullptr;
}

bool rejected(const lad_ranking & r, int32_t t, int32_t s, const char * why) {
    for (const auto & [c, w] : r.rejected) {
        if (c.tile == t && c.splits == s && w.find(why) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// 8 rows, K 2,560, N 2,560: the filter keeps the 8-row tile only; 20 output tiles on 320 resident CTAs: 1 split a
// quarter wave (ceil(20 / 80) / 4), 5 splits half a wave (ceil(100 / 80) / 4), 16 splits one full wave; at 1 split
// mma 8 x 2,560 x 2,560 x 16 (splits a wave) x 0.25 = 209,715,200 and mio 2 x 20 x 8 x 2,560 + 34 / 32 x 2,560 x
// 2,560 = 7,782,400; the first rank 4 splits (mma 52,428,800, mio 8,273,920)
void small_rows() {
    const lad_ranking r    = lad_rank(lad_fam::dense, 8, 2560, 2560, card());
    bool              filt = true;
    for (int32_t t = 1; t < fn::tc::lad_n_tiles; ++t) {
        filt = filt && rejected(r, t, 1, "filtered");
    }
    put("8 x 2,560 -> 2,560: the filter keeps the 8-row tile", filt && r.ranked.size() == 16,
        std::to_string(r.ranked.size()) + " candidates, " + std::to_string(r.rejected.size()) + " rejected");
    const lad_cand *s1 = find(r, 0, 1), *s5 = find(r, 0, 5), *s16 = find(r, 0, 16);
    put("8 x 2,560 -> 2,560: waves at 1, 5 and 16 splits (0.25, 0.5, 1)", s1 != nullptr && s5 != nullptr &&
                                                                              s16 != nullptr && s1->waves == 0.25f &&
                                                                              s5->waves == 0.5f && s16->waves == 1.0f);
    put("8 x 2,560 -> 2,560: 1 split's mma 209,715,200 and mio 7,782,400",
        s1 != nullptr && s1->mma == 209715200 && s1->mio == 7782400,
        s1 != nullptr ? "mma " + std::to_string(s1->mma) + ", mio " + std::to_string(s1->mio) : "missing");
    put("8 x 2,560 -> 2,560: the first rank t8x128 s4 (mma 52,428,800, mio 8,273,920)",
        !r.ranked.empty() && r.ranked[0].cfg == lad_cfg{ 0, 4 } && r.ranked[0].mma == 52428800 &&
            r.ranked[0].mio == 8273920,
        r.ranked.empty() ? "none" : r.ranked[0].cfg.text());
    const std::vector<lad_cfg> c = lad_candidates(r);
    put("8 x 2,560 -> 2,560: the candidates the first 8 then the retained route",
        c.size() == size_t(lad_top) + 1 && c.back().retained() && c[0] == r.ranked[0].cfg);
}

// 300 rows: no tile covers them, so the filter keeps every tile (96 candidates: 6 tiles x 16 splits); the first rank
// t64x128 s3 (5 x 5 tiles x 3 = 75 CTAs on 160: half a wave; mma 566,231,040, mio 19,660,800)
void wide_rows() {
    const lad_ranking r = lad_rank(lad_fam::dense, 300, 2560, 640, card());
    put("300 x 2,560 -> 640: no tile covers the rows, every tile kept", r.ranked.size() == 96 && r.rejected.empty(),
        std::to_string(r.ranked.size()) + " candidates");
    put("300 x 2,560 -> 640: the first rank t64x128 s3 (0.5 waves, mma 566,231,040, mio 19,660,800)",
        !r.ranked.empty() && r.ranked[0].cfg == lad_cfg{ 3, 3 } && r.ranked[0].waves == 0.5f &&
            r.ranked[0].mma == 566231040 && r.ranked[0].mio == 19660800,
        r.ranked.empty() ? "none" : r.ranked[0].cfg.text());
}

// 8,192 rows, K 320, N 10,240: every tile past 32 waves at 1 split, so 2 splits are rejected (max_waves) and only the
// six single-split candidates rank; the first t128x256 s1 (32 waves)
void many_waves() {
    lad_model m         = card();
    m.b_bytes           = 2.0;
    const lad_ranking r = lad_rank(lad_fam::dense, 8192, 320, 10240, m);
    bool              w = true;
    for (int32_t t = 0; t < fn::tc::lad_n_tiles; ++t) {
        w = w && rejected(r, t, 2, "waves past 32");
    }
    put("8,192 x 320 -> 10,240: 2 splits past 32 waves on every tile", w && r.ranked.size() == 6);
    put("8,192 x 320 -> 10,240: the first rank t128x256 s1 (32 waves)",
        !r.ranked.empty() && r.ranked[0].cfg == lad_cfg{ 5, 1 } && r.ranked[0].waves == 32.0f,
        r.ranked.empty() ? "none" : r.ranked[0].cfg.text());
}

// 64 rows on a 100,000-float workspace: every split count above 1 rejected (t8: 8 x 20 tiles x 2 x 1,024 floats past
// it); the filter drops the 128-row tiles; the first rank t16x128 s1
void workspace() {
    const lad_ranking r = lad_rank(lad_fam::dense, 64, 2560, 2560, card(100000, 65536));
    bool              w = true;
    for (int32_t t = 0; t < 4; ++t) {
        w = w && rejected(r, t, 2, "workspace");
    }
    put("64 x 2,560 -> 2,560, a 100,000-float workspace: split 2 rejected on every kept tile, 128-row tiles filtered",
        w && rejected(r, 4, 1, "filtered") && rejected(r, 5, 1, "filtered") && r.ranked.size() == 4);
    put("64 x 2,560 -> 2,560, a 100,000-float workspace: the first rank t16x128 s1",
        !r.ranked.empty() && r.ranked[0].cfg == lad_cfg{ 1, 1 }, r.ranked.empty() ? "none" : r.ranked[0].cfg.text());
}

// the grouped experts: one split only, the five 128-column tiles
void experts() {
    lad_model m           = card();
    m.num                 = 512;
    m.ctas[5]             = 0;
    const lad_ranking r   = lad_rank(lad_fam::ex_gate_up, 20, 2560, 640, m);
    bool              one = !r.ranked.empty();
    for (const lad_cand & c : r.ranked) {
        one = one && c.cfg.splits == 1 && c.cfg.tile < 5;
    }
    put("grouped experts 20 rows: one split, the 32-row tile and smaller", one && r.ranked.size() == 3,
        std::to_string(r.ranked.size()) + " candidates");
}

// the lookup: an entry at rungs 64 and 512 of one key; rows 64 .. 95 take rung 64's pick, 96 .. 511 none (retained),
// 512 and above rung 512's; below 8 rows, another key, another format: retained
void lookup() {
    const std::vector<lad_entry> t = {
        { lad_fam::dense, "Q8_0", 2560, 2560, 64,  2,  3, 10.0f, true  },
        { lad_fam::dense, "Q8_0", 2560, 2560, 64,  -1, 1, 12.0f, false },
        { lad_fam::dense, "Q8_0", 2560, 2560, 512, 4,  1, 30.0f, true  },
        { lad_fam::dense, "Q8_0", 2560, 2560, 768, -1, 1, 40.0f, true  },
    };
    lad_set_table(&t);
    const lad_key k{ lad_fam::dense, "Q8_0", 2560, 2560 };
    put("lookup: rows 64 and 95 at rung 64 (t32x128 s3)",
        lad_pick(k, 64) == lad_cfg{ 2, 3 } && lad_pick(k, 95) == lad_cfg{ 2, 3 });
    put("lookup: rows 96 and 511 (rungs 96 and 384, no entry) retained",
        lad_pick(k, 96).retained() && lad_pick(k, 511).retained());
    put("lookup: rows 512 and 767 at rung 512 (t128x128 s1)",
        lad_pick(k, 512) == lad_cfg{ 4, 1 } && lad_pick(k, 767) == lad_cfg{ 4, 1 });
    put("lookup: rows 768 (rung 768, the retained route picked) retained", lad_pick(k, 768).retained());
    put("lookup: rows 7 (below every rung) retained", lad_pick(k, 7).retained());
    put("lookup: another N and another format retained",
        lad_pick({ lad_fam::dense, "Q8_0", 2560, 6656 }, 64).retained() &&
            lad_pick({ lad_fam::dense, "BF16", 2560, 2560 }, 64).retained());
    put("lookup: coverage", lad_covers(k) && !lad_covers({ lad_fam::hc_up, "Q8_0", 320, 10240 }));
    // the workspace at the rows, not the rung's: t32x128 s3 at 64 rows needs 2 x 20 tiles x 3 x 4,096 = 491,520
    // floats, at 95 rows 3 x 20 x 3 x 4,096 = 737,280
    lad_model ws;
    ws.ws_floats = 600000;
    ws.ws_count  = 65536;
    put("lookup: the split's workspace checked at the rows (64 fits 600,000 floats, 95 does not: retained; one split "
        "always fits)",
        lad_lookup(k, 64, ws) == lad_cfg{ 2, 3 } && lad_lookup(k, 95, ws).retained() &&
            lad_lookup(k, 767, ws) == lad_cfg{ 4, 1 } && !lad_fits(lad_cfg{ 2, 3 }, 95, 2560, ws) &&
            lad_fits(lad_cfg{ 2, 3 }, 64, 2560, ws));
    put("rungs: 7 -> none, 8 -> 8, 47 -> 32, 8,191 -> 6,144, 100,000 -> 8,192",
        lad_rung_index(7) < 0 && lad_rungs[lad_rung_index(8)] == 8 && lad_rungs[lad_rung_index(47)] == 32 &&
            lad_rungs[lad_rung_index(8191)] == 6144 && lad_rungs[lad_rung_index(100000)] == 8192);
    lad_set_table(nullptr);
}

}  // namespace

int fnfl_ladder_test() {
    small_rows();
    wide_rows();
    many_waves();
    workspace();
    experts();
    lookup();
    printf("ladder-test: built-in table %zu entries\n", lad_table().size());
    printf("ladder-test: %s\n", g_bad == 0 ? "PASS" : "FAIL");
    return g_bad == 0 ? 0 : 1;
}

}  // namespace fn::model
