#pragma once

// APF2: the kernel-size ladder's selector, TurboMind's algorithm in FNFL code (1Cat-vLLM 1d7321bf2d,
// csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/context.cu Context::Filter / Populate, gemm.cu Find and its
// sm70 tuning: max_splits 16, max_waves 32), over the registry of compiled tiles (fn/kernels/tc/ladder.h) of each
// family, and the measured table prompt-tune writes (fn/model/ladder_table.inc).
//   filter:   the tiles whose rows cover rows no more than the smallest tile that covers them (all, past 128 rows)
//   populate: per tile and split count: tiles = ceil(rows / BM) ceil(N / BN) (x num, grouped); concurrency = the
//             SM count x the tile's resident CTAs; waves = full waves + ceil(residue / SMs) / resident CTAs; past
//             max_waves no more splits; mma = ceil_m ceil_n ceil_k(split) / waves a split x waves; mio = A's bytes
//             (column tiles x rows x K) + B's (row tiles x N x K) per split, + C's partials 2 (splits - 1) x rows x N
//   find:     ascending by the mean of mio / max mio and mma / max mma; the first 8 and the retained route are the
//             candidates prompt-tune times
//   lookup:   a product of rows takes its key's table entry at the largest rung at or below rows; no entry (a
//             shape, rung or format the table does not hold, or rows below 8) runs the retained route.
// Rungs: TurboMind's tuning ladder, 8 .. 8,192 rows (for the grouped experts, routed rows an expert).

#include "fn/kernels/tc/ladder.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fn::model {

constexpr int32_t lad_n_rungs            = 19;
constexpr int32_t lad_rungs[lad_n_rungs] = { 8,   16,  32,   48,   64,   96,   128,  192,  256, 384,
                                             512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192 };
constexpr int32_t lad_top                = 8;   // the candidates timed besides the retained route
constexpr int32_t lad_max_waves          = 32;  // TurboMind's sm70 tuning (gemm.cu: max_waves 32)

// the index of the largest rung at or below rows; -1 below 8
int32_t lad_rung_index(int32_t rows);

enum class lad_fam : int32_t { dense = 0, hc_down, hc_up, ex_gate_up, ex_down, count };
const char * lad_fam_name(lad_fam f);

// the families with split K (the grouped experts run one split)
constexpr bool lad_fam_splits(lad_fam f) {
    return f == lad_fam::dense || f == lad_fam::hc_down || f == lad_fam::hc_up;
}

// the tiles a family's registry holds (the grouped experts: 8 .. 128 rows by 128 columns)
constexpr int32_t lad_fam_tiles(lad_fam f) {
    return f == lad_fam::ex_gate_up || f == lad_fam::ex_down ? 5 : fn::tc::lad_n_tiles;
}

// a product: family, the weights' format name (fn::dense::fmt_name, "bf16"; HC "q8_0" / "bf16"; experts
// fn::experts::fmt_name), K and N. HC down: N 320 a stream (num 4); experts: num 512
struct lad_key {
    lad_fam     fam = lad_fam::dense;
    std::string fmt;
    int32_t     k = 0, n = 0;

    bool operator==(const lad_key & o) const { return fam == o.fam && fmt == o.fmt && k == o.k && n == o.n; }

    std::string text() const;  // "dense q8_0 K 2560 N 6656"
};

// tile: fn::tc::lad_tiles' index, -1 the retained route (pd_convert + pt_kernel, HC converted, per-task experts)
struct lad_cfg {
    int32_t tile = -1, splits = 1;

    bool retained() const { return tile < 0; }

    bool operator==(const lad_cfg & o) const { return tile == o.tile && splits == o.splits; }

    std::string text() const;  // "retained", "t32x128 s4"
};

// what the cost model knows of a family's compiled tiles on this card
struct lad_model {
    int32_t sm_count                  = 0;
    int32_t ctas[fn::tc::lad_n_tiles] = {};   // resident CTAs an SM (0: not compiled for the key)
    double  b_bytes                   = 2.0;  // the weights' bytes a value (TurboMind's type_b)
    int32_t num                       = 1;    // products in the launch (HC down 4, experts 512)
    int32_t max_splits                = fn::tc::lad_max_splits;
    int32_t max_waves                 = lad_max_waves;
    int64_t ws_floats                 = 0;  // the split workspace (fn::tc::lad_ws): partials and counters
    int32_t ws_count                  = 0;
};

struct lad_cand {
    lad_cfg cfg;
    int64_t mio = 0, mma = 0;
    float   waves = 0.0f, score = 0.0f;
};

struct lad_ranking {
    std::vector<lad_cand>                        ranked;    // ascending score
    std::vector<std::pair<lad_cfg, std::string>> rejected;  // filtered, or past max_waves
};

// Filter + Populate + Find for a product of rows (routed rows an expert, grouped)
lad_ranking          lad_rank(lad_fam fam, int32_t rows, int32_t k, int32_t n, const lad_model & m);
// the candidates prompt-tune times: the ranking's first lad_top, then the retained route
std::vector<lad_cfg> lad_candidates(const lad_ranking & r);

// a measured table entry (prompt-tune's output): a candidate's median at a rung; pick marks the winner (the retained
// route on a tie)
struct lad_entry {
    lad_fam      fam;
    const char * fmt;
    int32_t      k, n, rung;
    int32_t      tile, splits;
    float        us;
    bool         pick;
};

// the built-in table (ladder_table.inc), or a replacement (TEST ONLY: the selector host test's)
const std::vector<lad_entry> & lad_table();
void                           lad_set_table(const std::vector<lad_entry> * t);  // nullptr: the built-in

// the route for key at rows (the table's pick at the rung, else the retained route)
lad_cfg     lad_pick(const lad_key & key, int32_t rows);
// whether m's split-K workspace holds c for a product of rows x n (always for one split, or with no workspace limit)
bool        lad_fits(const lad_cfg & c, int32_t rows, int32_t n, const lad_model & m);
// the route the prompt path takes for key at rows: lad_pick's, or the retained route when the pick's split does not fit
// m's workspace at rows (the cost model checked the fit at the rung's rows, and rows run up to the next rung)
lad_cfg     lad_lookup(const lad_key & key, int32_t rows, const lad_model & m);
// whether the table holds any entry for key (the startup report's coverage)
bool        lad_covers(const lad_key & key);
// the startup report of key: each rung's pick, every measured candidate's median, and every candidate the cost
// model filtered or ranked out with its reason (m as the selector's at this card)
std::string lad_report(const lad_key & key, const lad_model & m);

}  // namespace fn::model
