#pragma once

// QSA attention at decode rows on one card (QA2): the q/k RMS norms with IMROPE, the K and V cache writes, attention
// over each row's selected cells read in place, and the sigmoid output gate. Simulator graph Q: qsa.attend and
// qsa.combine; the raw index-key write is summary_update's (qsa.h, round_update_args).
//
// Contract:
//   shapes:  one KV head a card (K and V BF16 [cell][256]) and its 12 query heads; rows B in {1, 2, 3, 4, 5} at
//            positions depth .. depth + B - 1; history up to fn::qsa::max_cells; each row attends to its selection
//            in qsa.h's body format (its K whole blocks, then its tail cells), at most 2,051 cells. Anything else is
//            rejected (false, nothing enqueued).
//   math (pinned source 66e11c3: qwen4exp.cpp build_layer_attn 1607-1692, build_attn_qsa 1364-1605; ggml-cpu ops.cpp
//            rope_yarn 5984-5999, ggml_mrope_cache_init 6017-6087, rotate_pairs 6089-6106):
//            q[t][h]   = imrope(rms_norm(qfull[t][h][0..255]) * q_norm, pos[.][t]); gate[t][h] = qfull[t][h][256..511]
//            k_new[t]  = imrope(rms_norm(kv_k[t]) * k_norm, pos[.][t]); v_new[t] = kv_v[t] (no norm)
//            cache[depth + t] = (bf16(k_new[t]), bf16(v_new[t])), round to nearest even
//            o[t][h]   = softmax_c(q[t][h] . K[c] / 16) . V[c] over row t's selected cells c (16 = sqrt(256); the
//                        GGUF has no attention.scale), out[t][h] = o[t][h] * sigmoid(gate[t][h])
//            rms_norm: x / sqrt(mean(x^2) + 1e-6). imrope: 64 of 256 dims in pairs (i, i + 32), i < 32; pair i's angle
//            is plane p's position times theta_scale^i as the F32 product chain (theta multiplied by theta_scale after
//            each pair, every plane), theta_scale = powf(1e7, -2 / 64); the plane is h when i % 3 == 1 and i < 33, w
//            when i % 3 == 2 and i < 30, t when i % 3 == 0 and i < 33, else e (sections 11, 11, 10, 0); the GGUF has
//            no rope scaling, so freq_scale = 1, ext_factor = 0 and attn_factor = 1 (qa2-test checks the keys).
//   numerics: F32 throughout (norms, rope, scores, softmax statistics, accumulators); K and V widened exactly from
//            BF16; no fast math (--ftz=false --prec-div=true --prec-sqrt=true --fmad=true, declared in
//            fn/CMakeLists.txt): sqrtf, the divisions, expf and sincosf are the precise forms. q is pre-scaled by the
//            exact 1/16. Reduction orders, all fixed and independent of B: the RMS sum of squares is a lane's 8 values
//            by fmaf then a 32-lane xor butterfly; a score is a lane's 8 products by fmaf chain, then a reduce-scatter
//            over 4 cells x 6 heads (xor 16, 8, 4) and a butterfly (xor 2, 1); a chunk's softmax takes its max, then
//            expf(s - max), summed lane-strided then by butterfly; P.V accumulates a cell group's cells (chunk cells
//            w, w + 8, ...) in order by fmaf, then the 8 groups as ((g0 + g4) + (g2 + g6)) + ((g1 + g5) + (g3 + g7));
//            the finish folds a row's chunks in chunk order: M = max m_j, L = sum l_j e^(m_j - M), A = sum acc_j
//            e^(m_j - M), o = A / L. Bound (qa2-test, against an FP64 reference written from the pinned lines):
//            max |error| / max |reference| <= 1e-5 per row and head; K cache within one BF16 ulp of the FP64 k; V
//            cache bitwise bf16(kv_v); the activation format bitwise act_ref of the F32 output; bitwise reruns; each
//            row's outputs at B = 1 bitwise equal to the same row's at B = 3 and 4.
//   floor:   B = 3, 2,051 cells a row, rows disjoint: 1.949 us ramp + max(6.3 MB at 1,021.6 GB/s = 6.17 us, 37.8 M
//            FMAs at 6.36 T a second = 5.94 us) + 0.83 us cross-block tail = 9.04 us. Frozen forecast (QA2 step 0,
//            work/QA2/results.json): 11.46 us at B = 3, 7.96 us at B = 1, warm.
//   budget:  one cooperative launch of persistent 256-thread CTAs, 2 an SM: 119 registers, no spills, 36,048 B of
//            static shared memory, 65.5 KB of SASS. Prepare: 13 B one-warp tasks (each row's 12 query heads and its
//            new cell's k and v) on the grid's warps, published by fences and a counter. Attend: unit (row, chunk of
//            86 selected cells, 6 heads); a CTA's 8 warps are 8 cell groups, each warp all 6 heads, 4 cells of K (then
//            V) in flight straight from L2/DRAM, q read once a CTA into shared memory; the unit writes (m, l,
//            acc[256]) a head. Finish, behind a grid barrier: a warp a (row, head, 32 values) folds the chunks,
//            gates, writes F32 and the activation format.
//   measured: QA2 Result (work/QA2/RESULT.md): warm 29.2 us at B = 3 and 22.5 us at B = 1 (196,608 cells, rows
//            disjoint), cold 39.3 / 34.4 us; latency-bound (issue active 36%), over the frozen forecast.
//   evidence: QA2, qsa-requirements.md, HX12 (machine sheet), graph Q.
//
// KV8 (--fnfl-kv-format q8, an opt-in format; NC1 section 8 amended by the operator, 2026-10-22): with k_scale and
// v_scale set, the caches hold int8 codes [cell][256] and FP16 scales [cell][kv8::scale_stride] (fn/kernels/qsa/
// kv_q8.cuh; CPU reference fn/quant/kv_q8_ref.h). prepare quantizes the F32 k (after the norm and IMROPE) and v it would
// round to BF16: per 64 values amax by a 3-step xor butterfly within each 8-lane quarter, scale = fp16_rn(amax / 127),
// code = clamp(rint(x / s), -127, 127) (0 when s is 0); a lane stores its 8 codes (8 bytes) and lane 8 g the scale.
// The attend loads a cell's codes (8 bytes a lane) and its group's scale in place and widens them to float(code) * s,
// exact in F32, where the BF16 form widens exactly; every other operation and reduction order is unchanged. Bound
// (qa2-test q8): the same 1e-5 against FP64 over the decoded cache; the new cells' codes and scales bitwise the CPU
// quantizer of the launch's own F32 k (k_f32) and of v; bitwise reruns and B invariance. A separate instantiation
// (attend_kernel<false, true>; the BF16 launch is unchanged); not with the host-backed KV (shift). Resources:
// kernel_resources(r, true).
//
// Publication (one launch): the producer warps write the prepared q and the new cells' K and V; each writing lane
// fences (__threadfence) and lane 0 of the warp adds 1 to the counter after a __syncwarp; every consumer CTA's thread 0
// polls the counter with acquire loads (gpu scope) until it reaches 13 B (bounded; a timeout sets the error word), and
// the CTA passes a barrier before reading q and any cell >= depth with L2 (.cg) loads. Before that acquire a CTA
// issues only loads of cells below depth, written before the launch. The partials are published to the finish by the
// grid barrier; after it, every consumer has acquired, so CTA 0 resets the counter for the next launch.

#include <cstddef>
#include <cstdint>

namespace fn::qsa::attn {

constexpr int32_t n_head = 12; // query heads a card
constexpr int32_t head_dim = 256;
constexpr int32_t q_stride = 512; // qfull: [q | gate] a head
constexpr int32_t chunk = 86;     // selected cells a unit (C: one constant for every B and depth)
constexpr int32_t group = 6;      // heads a unit (g)
constexpr int32_t max_sel = 4 * 512 + 3;
constexpr int32_t max_chunks = (max_sel + chunk - 1) / chunk; // 24
constexpr int32_t max_rows = 5; // SP2: the five-row verification pass
constexpr int32_t out_width = n_head * head_dim; // 3,072

// Scratch reused across layers and rounds; carve() lays it out in one allocation of workspace_bytes(), which the
// caller zeroes once (the counter starts at 0 and every finish leaves it at 0).
struct workspace {
    float * q;        // [max_rows][12][256] the prepared queries
    float * acc;      // [max_rows][max_chunks][12][256] each unit's unnormalized output
    float * ml;       // [max_rows][max_chunks][12][2] its (max, sum)
    uint32_t * ready; // [1] prepare tasks published this launch
    uint32_t * error; // [1] nonzero after a timed-out acquire; cleared by the caller
};
size_t workspace_bytes();
workspace carve(void * base);

struct args {
    const float * qfull;     // F32 [B][12][512]: each head's q, then its gate
    const float * kv_k;      // F32 [B][256], before the norm
    const float * kv_v;      // F32 [B][256]
    const float * q_norm;    // [256] attn_q_norm
    const float * k_norm;    // [256] attn_k_norm
    const uint32_t * body;   // the selection body (qsa.h layout) of the B rows
    const int32_t * offsets; // [B + 1] its byte offsets
    const int32_t * pos;     // [4][B] the rope planes t, h, w, e of each row
    uint16_t * k_cache;      // BF16 [cells][256] (KV8: int8 codes); cells depth .. depth + B - 1 are written
    uint16_t * v_cache;
    const int32_t * depth; // device: cells before the rows (FC-RD1's N_before), read by the launch (TG1)
    int32_t max_cells;     // host: the caches' cells; a depth with depth + B beyond them writes nothing and sets
                           // the error word (2)
    float theta_scale; // powf(1e7, -2 / 64) on the host (rope_theta_scale())
    float * out;       // F32 [B][3072], gated
    int8_t * codes;    // [B][3072] the activation format (fn::act)
    float * scales;    // [B][96][2]
    float * k_f32;     // optional [B][256]: the new cells' F32 k before rounding (qa2-test)
    uint64_t * stamps; // optional [grid][10], then [13 B]: globaltimer stamps of each CTA's first unit's phases and of
                       // each prepare task's publication (qa2-bench diag)
    workspace ws;
    // ME3 (the host-backed KV's graphs, fn/model/kv_tier.h): optional device word w; when set, the launch reads and
    // writes cell c at k_cache + (c - *w) * 256 (and v_cache alike), *w read on the device (a separate instantiation;
    // null: the resident launch)
    const int64_t * shift = nullptr;
    // KV8: both set, the caches are q8: k_cache and v_cache hold int8 codes [cells][256] and these the layer's FP16
    // scales, cell c's group g at [c * kv8::scale_stride + g] (cells depth .. depth + B - 1 written); both null: BF16
    uint16_t * k_scale = nullptr;
    uint16_t * v_scale = nullptr;
};

float rope_theta_scale();

// the selected cells of the row at position q (qsa.h body): 4 min((q + 1) / 4, 512) + (q + 1) % 4
int32_t selected_cells(int32_t q);
// units of the attend launch at these rows and depth; its grid (persistent CTAs, all co-resident) is the deepest
// rows' units or the co-resident limit, the same at every depth, so one captured launch serves every depth (TG1); a
// CTA without a unit takes part in the prepare tasks, the grid barrier and the finish only
int32_t units(int32_t rows, int32_t depth);
int32_t grid(int32_t rows, int32_t device);

// Enqueues the launch on stream. False, with nothing enqueued, for an argument outside the contract; false after
// a failed submission (the CUDA error is left for the caller).
bool attend(int32_t rows, const args & a, void * stream);

// FMS2 (build item 3, W3): attend() of lanes 2 .. 4 of the joint verify round in one cooperative launch, lane u on its
// own arguments a[u] (its rows, caches, selection and workspace; every lane BF16 or every lane KV8, resident, no
// stamps). The grid is the co-resident limit or the lanes' deepest units. Each lane's prepare tasks, units and finish
// are attend()'s on its arguments, lane after lane in one index space each; a CTA acquires a lane's counter before
// its first unit of that lane, and the grid barrier orders every lane's units before every finish. A unit's result
// does not depend on its CTA, so each lane's outputs and cache cells are its attend()'s bitwise. A lane at a refused
// depth sets its error word (2) and writes nothing; the other lanes run. False, with nothing enqueued, for an argument
// outside attend()'s contract in any lane, or lanes outside 2 .. 4.
bool attend_lanes(int32_t lanes, int32_t rows, const args * a, void * stream);

// the launch's resources on the current device, for the bench and the Result
struct resources {
    int32_t regs;
    int32_t local_bytes;
    int32_t shared_bytes;
    int32_t ctas_per_sm;
};
// q8: the KV8 instantiation's (attend_kernel<false, true>)
bool kernel_resources(resources & r, bool q8 = false);
// FMS2: attend_lanes' kernel (attend_lanes_kernel<q8>) and its grid at rows and lanes
bool lanes_resources(resources & r, bool q8, int32_t rows, int32_t lanes, int32_t & grid_out);

} // namespace fn::qsa::attn
