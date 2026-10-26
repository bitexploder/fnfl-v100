#pragma once

// QSA indexer summaries, scoring and block selection for one QSA layer on its index-owner card (QA1, QA1b). Simulator
// graph Q: the indexer and selection nodes on the index owner; the selected IDs feed X-SELECT and attention (QA2).
//
// Contract:
//   shapes:  indexer keys and queries are 128 wide (fn::qsa::head_dim); 4 query heads a row; blocks of 4 cells; the
//            top 512 blocks a row. Rows B in {1, 2, 3, 4, 5} (SP2: 5) at consecutive positions depth ..
//            depth + B - 1 of one sequence, so n_kv = depth + B cells and cell c holds position c. History up to
//            262,144 cells (65,536 blocks). Anything else is rejected (the call returns false, nothing enqueued).
//   math, one QSA layer (source: qwen4exp.cpp build_qsa_top_k, 1124-1360 at 66e11c3, and ggml_qsa_select):
//            summary[b] = rope(rms_norm(mean(k[4b..4b+3])) * k_norm, position 4b), the F32 pooled key, stored when
//                         block b completes (summary_update)
//            q'[t][h]   = rope(rms_norm(q[t][h]) * q_norm, position depth + t)
//            score[t][b] = sum_h relu(q'[t][h] . summary[b])
//            row t sees block b whole when 4b + 3 <= depth + t; it takes the 512 whole blocks with the highest
//            score (equal scores: the lower block first), in ascending block order, and its tail cells (the
//            incomplete block up to its own position). The output is that selection as WP3's compact body (QI-10):
//            no expansion to cells, no padding and no union; attention reads a row's blocks through the logical map.
//   numerics: QI-11: the selection is held to within 2x the pinned engine's own prompt-against-decode selection
//            spread, and is deterministic with a fixed tie rule (equal scores: the lower block first); it is not
//            held bitwise to the borrowed arithmetic. The summaries' arithmetic stays as QA1b wrote it, after the
//            pooling it came from (provenance: ggml-cuda qsa_pool_rope_f32, built with -use_fast_math): F32, FTZ,
//            the members summed in order, scale by an FFMA, a 256-thread butterfly for the mean square, mean + eps
//            as one FFMA with MUFU.RCP, MUFU.RSQ, __powf, __sinf and __cosf, stored as F32. So this file is built
//            with -use_fast_math (declared in fn/CMakeLists.txt). Queries go through the same arithmetic. Scores
//            are F32: each head's 128-term dot product is one fmaf chain in element order from 0, relu is
//            (d > 0 ? d : 0), and the heads are summed ((h0 + h1) + h2) + h3; the selection key orders the scores
//            with subnormals as +0. The CPU reference (qsa_test.cpp) computes the same scores bit for bit, so the
//            body is bitwise equal to it. BF16 or F16 storage and other scoring arithmetic are design choices
//            measured against the spread rule (qa-test reports the blocks that move against the fixture).
//   floor:   (B = 3, 1,380 MHz, HX2's 1.95 us ramp plus the stream at 1,021.6 GB/s) score + select at
//            196,608 cells: 25.2 MB of summaries, 26.6 us, target <= 38 us; 65,536 cells: 8.4 MB, 10.2 us, target
//            <= 20 us; 8,192 cells: 1.0 MB, 3.0 us, target <= 12 us. Select alone <= 8 us at every depth;
//            score alone <= 1.15x the floor at 196K. summary_update: <= 2 blocks a round, target < 4 us.
//   budget:  select() is two launches. Launch 1 (score) streams the summaries, one warp a 32-block tile, lane = block,
//            and keeps D 16-byte loads a lane in flight through the FMAs: each chunk is taken into registers when it
//            arrives and its slot reissued at once. D is the shallowest of 4, 8, 16 whose loads in flight cover 27,000
//            bytes an SM (HX12's loaded-latency p90); a CTA has up to 22 tile warps (704 threads at D = 4) plus warps
//            that only normalize queries. It writes every key and each tile's two best blocks a row. Launch 2 (select):
//            1,024-thread CTAs, one a row, no global atomics, no waits between CTAs, deterministic; each writes its
//            row's body (header and tail first, the blocks after the selection) and its byte offset.
//            A row takes one histogram pass on the keys' top 15 bits (16-bit counts, 64 KB of shared memory) over
//            the tiles' listed blocks for T1, reads back the tiles whose second best is in or above T1's bin (a warp
//            a tile, keys in registers), adds their other blocks to the same counts for T2, and one warp finishes
//            inside T2's bin (at most 128 keys; a larger bin, from ties, is narrowed by 9-bit passes first): about 13
//            CTA barriers a row. Registers, spills and shared memory: work/QA1b/provenance/ptxas-v.txt.
//            Measured [M, QA1b, before C2's body]: 196K B = 3 takes 59.4 us (score 34.3, select and union 24.9):
//            both launches are latency-bound, not stream-bound; see work/QA1b/RESULT.md. C2's figures for the body
//            (an edge change): tickets/C2-fnfl-edges.md, Result.
//   evidence: QA1 (the first build; work/QA1/RESULT.md), QA1b (this one; work/QA1b/RESULT.md), HX7 (selection
//            captures), HX10 (today's 654.8 us layer at 196K, B = 3), HX12 (machine sheet: loaded DRAM latency, group
//            reductions), HC1c (stream depth), fnfl/design/qsa-requirements.md, graph Q.
//
// Layouts (device memory owned by the caller; every pointer 16-byte aligned, no overlaps):
//   keys:      the raw indexer key cache, BF16 [cell][128] (as the pinned source caches it: before pooling, norm and
//              rope)
//   summaries: F32 in tiles of 32 blocks: element e of block b at (b / 32) * 4096 + (e / 4) * 128 + (b % 32) * 4 +
//              e % 4, so one 16-byte load a lane reads one chunk of 32 consecutive blocks. summary_bytes(n_blocks).
//   q:         F32 [B][4][128], the index query projection of each row (before norm and rope)
//   body:      WP3's selected-block body (fnfl/design/history/wp3/transfers.md), U32 words. Row t, at position
//              q = depth + t with n_vis = (q + 1) / 4: U32 K = min(n_vis, 512), U32 tail count (q + 1) % 4, then K
//              I32 logical block IDs ascending, then the tail's I32 cell IDs ascending (4 n_vis .. q): 8 + 4 K +
//              4 tail bytes. Rows are packed in row order with no padding: body_words(depth, B) words in all.
//   offsets:   I32 [B + 1]: row t's byte offset in the body, then the body's total bytes. Both depend on depth and
//              B only (body_offset gives them on the host).

#include <cstddef>
#include <cstdint>

namespace fn::qsa {

constexpr int32_t head_dim = 128;   // indexer key and query width
constexpr int32_t n_idx_head = 4;   // index query heads a row
constexpr int32_t ratio = 4;        // cells a block (compress ratio of the QSA layers)
constexpr int32_t top_blocks = 512; // indexer_top_k / ratio
constexpr int32_t n_tail = ratio - 1;
constexpr int32_t max_rows = 5; // SP2: the five-row verification pass
constexpr int32_t max_cells = 262144;
constexpr int32_t max_blocks = max_cells / ratio;
constexpr int32_t tile_blocks = 32;            // blocks a tile: one warp, one lane a block
constexpr int32_t n_chunk = head_dim * 4 / 16; // 16-byte chunks of a summary (F32)
constexpr int32_t n_rot = 64;                  // rope.dimension_count; the rest of a row passes through
constexpr float rope_freq_base = 10000000.0f;

// The rope of the indexer (IMROPE, sections 11, 11, 10, 0, all positions equal for text). theta_scale is
// powf(freq_base, -2 / n_rot) on the host, as ggml-cuda computes it; freq_scale and attn_factor are 1 for this model.
struct rope_params {
    float theta_scale;
    float freq_scale;
    float attn_factor;
};
rope_params model_rope();

// sizes
int32_t blocks_complete(int32_t n_cells);                  // n_cells / 4
constexpr int32_t max_row_words = 2 + top_blocks + n_tail; // the longest row body: 517 words, 2,068 bytes
int32_t row_words(int32_t q);                              // the body of the row at position q: 2 + K + tail words
int32_t body_offset(int32_t depth, int32_t t);             // row t's byte offset (t = B: the total bytes)
int32_t body_words(int32_t depth, int32_t rows);           // the whole body, at most max_rows * max_row_words
size_t summary_bytes(int32_t n_blocks);                    // whole tiles

// Which blocks hold a valid summary: blocks 0 .. n_valid - 1. A rollback marks blocks from first on stale; the runtime
// then recomputes them with summary_update before they are scored again.
struct summary_index {
    int32_t n_valid = 0;
};
void invalidate_from(summary_index & idx, int32_t first);
// after summary_update(first, count) has been enqueued; first must be <= n_valid (no gap)
bool mark_updated(summary_index & idx, int32_t first, int32_t count);

// Pools blocks first .. first + count - 1 (each complete: its 4 cells' keys written) into summaries. One warp a block.
struct update_args {
    const uint16_t * keys; // BF16 [cell][128]
    const float * k_norm;  // [128] index_k_norm
    float * summaries;     // tiled F32
    int32_t first;
    int32_t count; // 1 .. max_blocks
    rope_params rope;
};
bool summary_update(const update_args & a, void * stream);

// The round form (QA2's raw index-key write with the pooling, TG1: the depth from the round descriptor, FC-RD1): one
// launch, every round. One warp a new cell stores bf16(ik) (round to nearest even) at its cell; each block the round's
// rows complete (blocks depth / 4 .. (depth + rows) / 4 - 1: none or one for rows <= 4, up to two at SP2's 5) is
// pooled by one warp, which takes a new cell's key from ik rounded the same way, never from the cache, so the pooled
// summaries stay bitwise what the cache path computes. The depth is read on the device, so one captured launch serves
// every depth; a depth outside 0 .. max_cells - rows writes nothing (the runtime refuses such a round before launch,
// FC-RD1). Its grid is sized for the most warps any depth needs (the blocks pooled and rows cells written).
struct round_update_args {
    uint16_t * keys = nullptr;      // BF16 [cell][128], the raw key cache (read and written)
    const float * k_norm = nullptr; // [128] index_k_norm
    float * summaries = nullptr;    // tiled F32
    rope_params rope = {};
    const float * ik = nullptr;      // F32 [rows][128], the index key projection of the new cells
    const int32_t * depth = nullptr; // device: cells before the rows (FC-RD1's N_before)
    int32_t rows = 0;                // 1 .. max_rows
    int32_t max_cells = 0;           // host: the key cache's cells (the slot's capacity); depth + rows beyond it writes
                                     // nothing
    // ME6 (the host-backed tier's graphs, fn/model/kv_tier.h): optional device word w; when set, cell c is read and
    // written at keys + (c - *w) * 128, *w read on the device (null: the resident form)
    const int64_t * shift = nullptr;
};
bool summary_update(const round_update_args & a, void * stream);
// FMS2 (build item 2, the joint catch-up): the round form of lanes 2 .. 4 in one launch, lane u on a[u] with *rows[u]
// new cells (1 .. max_rows, read on the device: its descriptor's rows; a[u].rows must be max_rows, which sizes the
// grid): lane u's warps are summary_update's at *rows[u], the same device code. Resident only (shift null). False,
// with nothing enqueued, otherwise. FMS2 (build item 3, W3): rows null, every lane's new cells are its a[u].rows (the
// joint verify round's R = d + 1, 1 .. max_rows; FMS13), on the grid of max_rows (warps past a lane's cells exit).
bool summary_update_lanes(int32_t lanes, const round_update_args * a, const int32_t * const * rows, void * stream);

// Scratch of select(), reused across layers and rounds; carve() lays it out in one allocation of workspace_bytes().
struct workspace {
    uint32_t * keys;  // [max_rows][max_blocks] the selection key of every block
    uint32_t * best1; // [max_rows][tiles] each tile's best key
    uint32_t * best2; // [max_rows][tiles] its second
    uint16_t * lanes; // [max_rows][tiles] their lanes (low byte best, high byte second; 0xff: none)
    uint32_t * error; // [1] nonzero after a failed self-check; cleared by the caller
    uint32_t * stats; // [max_rows][n_stats] each row's selection path in the last select() (tests and benches)
};
// stats: listed blocks counted, T1's bin (or -1), failing tiles, 16-bit counts used (1/0), T2's bin count, 9-bit
// passes, the bin one warp finished (0 when a pass left one value), 1 when the row selected (more than 512 blocks)
constexpr int32_t n_stats = 8;
size_t workspace_bytes();
workspace carve(void * base);

struct select_args {
    const float * summaries; // tiled F32: every block a row sees whole, (depth + B) / 4 of them, valid (the runtime's
                             // summary_index covers them before the round is launched)
    const float * q;         // F32 [B][4][128]
    const float * q_norm;    // [128] index_q_norm
    const int32_t * depth;   // device: cells before the rows (FC-RD1's N_before); n_kv = depth + B
    int32_t max_cells;       // host: the largest n_kv the launch may see (the slot's capacity); sizes launch 1's grid
    rope_params rope;
    workspace ws;
    uint32_t * body;   // [B * max_row_words]: body_words(depth, B) of them written
    int32_t * offsets; // [B + 1]
    // SP2 (the sparse draft, temporary at SP2's disposition): 1 rotates the one row's query at position depth + 1
    // instead of depth (the draft's step-0 query at its cell N, scoring the N / 4 whole blocks of the prefix with
    // depth = N - 1); rows must be 1. 0: the target's form.
    int32_t q_ahead = 0;
};
// Both launches are always enqueued, so one captured pair serves every depth: launch 1 (score) returns at once when no
// row sees more than 512 whole blocks, and its grid is sized for max_cells (a CTA past the depth's tiles exits). A
// depth outside 0 .. max_cells - B writes nothing, and select's error word is set.
bool select(int32_t rows, const select_args & a, void * stream);
// whether launch 1 does work at these rows and depth (it is always enqueued), for a profiler's counts
int32_t select_launches(int32_t rows, int32_t depth);

// launch 1's grid for a depth, as select() picks it (bench and SASS checks)
struct score_grid {
    int32_t tiles; // 32-block tiles scored
    int32_t tpc;   // tiles a CTA (one warp each)
    int32_t warps; // warps a CTA: tpc, or more to normalize the queries
    int32_t ctas;
    int32_t depth; // 16-byte loads a lane in flight
};
score_grid grid_of(int32_t rows, int32_t depth);

// FMS2 (build item 3, W3): select() of lanes 2 .. 4 of the joint verify round, each at rows (FMS13: the round's
// R = d + 1, 2 .. max_rows, every lane alike) on its own
// arguments a[u] (target form, q_ahead 0; its own workspace, body and offsets), in two launches for all lanes:
//   score:  blockIdx.y the lane, grid_lanes' CTAs a lane. D and the tiles a CTA come from every lane's tiles together
//           (grid_lanes), so a lane's D and grid may differ from its select() launch's; a block's key and a tile's
//           best two depend on neither (each lane's chunks are summed in chunk order whatever D, and every tile's
//           warp is the same arithmetic wherever it runs), so each lane's keys are its select()'s bitwise.
//   select: blockIdx.y the lane, one CTA a row, select()'s.
// test_depth: 0, grid_lanes' D; 4, 8 or 16 forces it (qsaj-test). False, with nothing enqueued, for an argument
// outside select()'s contract in any lane, lanes outside 2 .. 4 or rows outside 2 .. max_rows.
bool       select_lanes(int32_t lanes, int32_t rows, const select_args * a, void * stream, int32_t test_depth = 0);
// the score grid of select_lanes at a depth (the deepest lane's max_cells - rows): tiles and ctas are a lane's
score_grid grid_lanes(int32_t rows, int32_t depth, int32_t lanes, int32_t test_depth = 0);

// ---- test support: the device's fast-math results for arguments the CPU reference computed ----
// op 0: rsqrtf(x); 1: x / y as -use_fast_math divides; 2: __powf(x, y); 3: __cosf(x); 4: __sinf(x). out[i] = op(x[i], y[i]).
bool fast_math_oracle(int32_t op, const float * x, const float * y, float * out, int32_t n, void * stream);

} // namespace fn::qsa
