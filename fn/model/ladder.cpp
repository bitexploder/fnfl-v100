#include "fn/model/ladder.h"

#include <algorithm>
#include <cstdio>

namespace fn::model {
namespace {

// prompt-tune's table (fn/model/ladder_table.inc: lad_entry initializers; the first entry a sentinel, rung 0)
const lad_entry builtin_entries[] = {
    { lad_fam::dense, "", 0, 0, 0, -1, 1, 0.0f, false },
#include "fn/model/ladder_table.inc"
};

const std::vector<lad_entry> & builtin() {
    static const std::vector<lad_entry> t(std::begin(builtin_entries) + 1, std::end(builtin_entries));
    return t;
}

const std::vector<lad_entry> * table_override = nullptr;

int64_t cdiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

bool same_key(const lad_entry & e, const lad_key & k) {
    return e.fam == k.fam && k.fmt == e.fmt && e.k == k.k && e.n == k.n;
}

std::string us_text(float us) {
    char b[32];
    snprintf(b, sizeof b, "%.2f us", static_cast<double>(us));
    return b;
}

}  // namespace

int32_t lad_rung_index(int32_t rows) {
    int32_t i = -1;
    while (i + 1 < lad_n_rungs && lad_rungs[i + 1] <= rows) {
        ++i;
    }
    return i;
}

const char * lad_fam_name(lad_fam f) {
    switch (f) {
        case lad_fam::dense:
            return "dense";
        case lad_fam::hc_down:
            return "hc_down";
        case lad_fam::hc_up:
            return "hc_up";
        case lad_fam::ex_gate_up:
            return "ex_gate_up";
        case lad_fam::ex_down:
            return "ex_down";
        default:
            return "?";
    }
}

std::string lad_key::text() const {
    return std::string(lad_fam_name(fam)) + " " + fmt + " K " + std::to_string(k) + " N " + std::to_string(n);
}

std::string lad_cfg::text() const {
    if (tile < 0) {
        return "retained";
    }
    const fn::tc::lad_tile & t = fn::tc::lad_tiles[tile];
    return "t" + std::to_string(t.bm) + "x" + std::to_string(t.bn) + " s" + std::to_string(splits);
}

lad_ranking lad_rank(lad_fam fam, int32_t rows, int32_t k, int32_t n, const lad_model & m) {
    lad_ranking   out;
    const int32_t nt = lad_fam_tiles(fam);
    if (rows < 1 || k < 1 || n < 1 || m.sm_count < 1) {
        return out;
    }
    // Context::Filter: the smallest tile rows covering rows bound the tiles kept
    int32_t max_bm = 0;
    for (int32_t t = 0; t < nt; ++t) {
        if (m.ctas[t] > 0) {
            max_bm = std::max(max_bm, fn::tc::lad_tiles[t].bm);
        }
    }
    for (int32_t t = 0; t < nt; ++t) {
        if (m.ctas[t] > 0 && fn::tc::lad_tiles[t].bm >= rows) {
            max_bm = std::min(max_bm, fn::tc::lad_tiles[t].bm);
        }
    }
    const int32_t chunks = static_cast<int32_t>(cdiv(k, fn::tc::lad_bk));
    const int32_t num    = std::max(1, m.num);
    for (int32_t t = 0; t < nt; ++t) {
        const fn::tc::lad_tile & tl = fn::tc::lad_tiles[t];
        if (m.ctas[t] < 1) {
            out.rejected.push_back({
                lad_cfg{ t, 1 },
                "not compiled for the format (or no resident CTA)"
            });
            continue;
        }
        if (tl.bm > max_bm) {
            out.rejected.push_back({
                lad_cfg{ t, 1 },
                "filtered: " + std::to_string(tl.bm) + " rows past the smallest covering tile (" +
                                     std::to_string(max_bm) + ")"
            });
            continue;
        }
        // Context::Populate
        const int64_t tiled_m         = cdiv(rows, tl.bm);
        const int64_t tiled_n         = cdiv(n, tl.bn);
        const int32_t concurrency     = m.sm_count * m.ctas[t];
        const float   waves_per_split = float(tiled_m * tiled_n) / float(concurrency);
        const float   splits_per_wave = 1.0f / waves_per_split;
        const int64_t ceil_m          = tiled_m * tl.bm;
        const int64_t ceil_n          = tiled_n * tl.bn;
        const int32_t max_splits      = lad_fam_splits(fam) ? std::min(m.max_splits, chunks) : 1;
        for (int32_t splits = 1; splits <= max_splits; ++splits) {
            const int64_t split_ceil_k   = cdiv(chunks, splits) * fn::tc::lad_bk;
            const int64_t split_mma_cost = ceil_m * ceil_n * split_ceil_k;
            const auto    wave_mma_cost  = static_cast<int64_t>(static_cast<float>(split_mma_cost) * splits_per_wave);
            const int64_t grid           = tiled_m * tiled_n * splits * num;
            const int64_t full_waves     = grid / concurrency;
            const int64_t residue        = grid % concurrency;
            const float   partial_wave   = float(cdiv(residue, m.sm_count)) / float(m.ctas[t]);
            const float   waves          = float(full_waves) + partial_wave;
            if (splits > 1 && m.ws_floats > 0 &&
                (tiled_m * tiled_n * num > m.ws_count ||
                 tiled_m * tiled_n * num * splits * tl.bm * tl.bn > m.ws_floats)) {
                out.rejected.push_back({
                    lad_cfg{ t, splits },
                    "the split workspace (and every larger split count)"
                });
                break;
            }
            if (splits > 1 && waves > float(m.max_waves)) {
                out.rejected.push_back({
                    lad_cfg{ t, splits },
                    "waves past " + std::to_string(m.max_waves) + " (and every larger split count)"
                });
                break;
            }
            lad_cand c;
            c.cfg               = { t, splits };
            c.waves             = waves;
            c.mma               = static_cast<int64_t>(static_cast<float>(wave_mma_cost) * waves);
            const auto    mio_a = static_cast<int64_t>(2.0 * double(tiled_n * rows * split_ceil_k) * splits * num);
            const auto    mio_b = static_cast<int64_t>(m.b_bytes * double(tiled_m * n * split_ceil_k) * splits * num);
            const int64_t mio_c = int64_t(4) * rows * n * (splits - 1) * 2 * num;
            c.mio               = mio_a + mio_b + mio_c;
            out.ranked.push_back(c);
        }
    }
    // Find: the mean of the normalized costs, ascending (stable: ties keep tile and split order)
    int64_t mio_max = 1, mma_max = 1;
    for (const lad_cand & c : out.ranked) {
        mio_max = std::max(mio_max, c.mio);
        mma_max = std::max(mma_max, c.mma);
    }
    for (lad_cand & c : out.ranked) {
        c.score = 0.5f * (float(c.mio) / float(mio_max) + float(c.mma) / float(mma_max));
    }
    std::stable_sort(out.ranked.begin(), out.ranked.end(),
                     [](const lad_cand & a, const lad_cand & b) { return a.score < b.score; });
    return out;
}

std::vector<lad_cfg> lad_candidates(const lad_ranking & r) {
    std::vector<lad_cfg> c;
    for (size_t i = 0; i < r.ranked.size() && i < size_t(lad_top); ++i) {
        c.push_back(r.ranked[i].cfg);
    }
    c.push_back(lad_cfg{});
    return c;
}

const std::vector<lad_entry> & lad_table() {
    return table_override != nullptr ? *table_override : builtin();
}

void lad_set_table(const std::vector<lad_entry> * t) {
    table_override = t;
}

lad_cfg lad_pick(const lad_key & key, int32_t rows) {
    const int32_t ri = lad_rung_index(rows);
    if (ri < 0) {
        return {};
    }
    for (const lad_entry & e : lad_table()) {
        if (e.pick && e.rung == lad_rungs[ri] && same_key(e, key)) {
            return { e.tile, e.splits };
        }
    }
    return {};
}

bool lad_fits(const lad_cfg & c, int32_t rows, int32_t n, const lad_model & m) {
    if (c.retained() || c.splits <= 1 || m.ws_floats <= 0) {
        return true;
    }
    const fn::tc::lad_tile & tl    = fn::tc::lad_tiles[c.tile];
    const int64_t            tiles = cdiv(rows, tl.bm) * cdiv(n, tl.bn) * std::max(1, m.num);
    return tiles <= m.ws_count && tiles * c.splits * tl.bm * tl.bn <= m.ws_floats;
}

lad_cfg lad_lookup(const lad_key & key, int32_t rows, const lad_model & m) {
    const lad_cfg c = lad_pick(key, rows);
    return lad_fits(c, rows, key.n, m) ? c : lad_cfg{};
}

bool lad_covers(const lad_key & key) {
    for (const lad_entry & e : lad_table()) {
        if (same_key(e, key)) {
            return true;
        }
    }
    return false;
}

std::string lad_report(const lad_key & key, const lad_model & m) {
    std::string out =
        "ladder: " + key.text() + (lad_covers(key) ? "" : " (no table entry: retained at every rung)") + "\n";
    for (int32_t ri = 0; ri < lad_n_rungs; ++ri) {
        const int32_t rung = lad_rungs[ri];
        const lad_cfg pick = lad_pick(key, rung);
        float         pus  = -1.0f;
        std::string   meas;
        for (const lad_entry & e : lad_table()) {
            if (e.rung != rung || !same_key(e, key)) {
                continue;
            }
            const lad_cfg c{ e.tile, e.splits };
            if (e.pick) {
                pus = e.us;
            } else {
                meas += "; " + c.text() + " " + us_text(e.us) + " (measured slower)";
            }
        }
        out += "  rung " + std::to_string(rung) + ": " + pick.text() + (pus >= 0.0f ? " " + us_text(pus) : "") + meas;
        const lad_ranking r = lad_rank(key.fam, rung, key.k, key.n, m);
        for (size_t i = size_t(lad_top); i < r.ranked.size(); ++i) {
            out += "; " + r.ranked[i].cfg.text() + " rank " + std::to_string(i + 1) + " (not in the top " +
                   std::to_string(lad_top) + ")";
        }
        for (const auto & [c, why] : r.rejected) {
            out += "; " + c.text() + " " + why;
        }
        out += "\n";
    }
    return out;
}

}  // namespace fn::model
