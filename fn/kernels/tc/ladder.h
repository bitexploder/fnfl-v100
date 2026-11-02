#pragma once

// APF2: the kernel-size ladder's tile list and split-K workspace, shared by the families that launch ladder tiles
// (dense, HC, grouped experts; host and device). tc_ladder.cuh holds the CTA.
//   tiles:   TurboMind's sm70 list (1Cat-vLLM 1d7321bf2d, kernels/gemm/kernel/sm70_884_4.cu:160-240): 8, 16, 32, 64
//            and 128 rows by 128 columns, and 128 by 256, K 32 a stage.
//   split K: 1 .. lad_max_splits CTAs a tile; partials F32 [tile][split][BM x BN] and one I32 counter a tile, zero
//            between launches (each launch leaves them zero).

#include <cstdint>

namespace fn::tc {

constexpr int32_t lad_n_tiles    = 6;
constexpr int32_t lad_max_splits = 16;  // TurboMind's sm70 tuning (gemm.cu: max_splits 16)
constexpr int32_t lad_bk         = 32;  // K a stage

struct lad_tile {
    int32_t bm, bn;
};

constexpr lad_tile lad_tiles[lad_n_tiles] = {
    { 8, 128 }, { 16, 128 }, { 32, 128 }, { 64, 128 }, { 128, 128 }, { 128, 256 }
};

// the split-K workspace on one card
struct lad_ws {
    float *   part        = nullptr;
    int64_t   part_floats = 0;
    int32_t * count       = nullptr;
    int32_t   count_len   = 0;
};

// a compiled instance's budget (the ptxas report) and resident CTAs an SM at its shared memory (the cost model's)
struct lad_resources {
    int32_t regs = 0, local_bytes = 0, shared_bytes = 0, ctas_per_sm = 0;
};

// the output tiles of a product of rows x n on tile t (0 for a bad argument)
constexpr int32_t lad_tile_count(int32_t rows, int32_t n, int32_t t) {
    return t < 0 || t >= lad_n_tiles || rows < 1 || n < 1 ?
               0 :
               ((rows + lad_tiles[t].bm - 1) / lad_tiles[t].bm) * ((n + lad_tiles[t].bn - 1) / lad_tiles[t].bn);
}

// the partial floats splits > 1 needs for those tiles
constexpr int64_t lad_split_floats(int32_t rows, int32_t n, int32_t t, int32_t splits) {
    return splits > 1 ? int64_t(lad_tile_count(rows, n, t)) * splits * lad_tiles[t].bm * lad_tiles[t].bn : 0;
}

// whether ws holds a launch of tiles output tiles with splits partials of BM x BN each
constexpr bool lad_ws_fits(const lad_ws & ws, int32_t tiles, int32_t splits, int32_t bm, int32_t bn) {
    return splits == 1 || (ws.part != nullptr && ws.count != nullptr && tiles <= ws.count_len &&
                           int64_t(tiles) * splits * bm * bn <= ws.part_floats);
}

}  // namespace fn::tc
