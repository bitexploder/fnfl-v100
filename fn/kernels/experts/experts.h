#pragma once

// Routed-expert kernels for one card of the two-card split (EX1), with the shared expert folded in (EX2): card r owns
// intermediate channels 320*r .. 320*r+319 of every routed expert and of the shared expert, so gate and up are
// [2560 -> 320] and down is [320 -> 2560] per expert per card. Simulator graph F, nodes FRun0/FRun1 (gate_up) and
// FOut0/FOut1 (down); fnfl/design/experts-requirements.md (v0).
//
// Contract (both kernels):
//   shapes:  D = 2560, 512 experts, 10 selected per row, 320 channels per card; rows B in {1, 2, 3, 4, 5} (one
//            template instance each; SP2 adds B = 5 with the capacity-5 routing layout, cap_of below, so the
//            B <= 4 instances keep the four-row layout). Any other B, a null or misaligned pointer, or a format
//            combination outside the deployed list (experts_inst.h) is rejected (returns false, nothing enqueued).
//            Expert IDs must be in [0, 512); a launch that sees any other ID sets status[0] = 1 (when status is not
//            null) and writes no output.
//   formats: routed gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; routed down Q2_0, IQ4_NL; shared gate and up
//            IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K (fn::dense::fmt, each its own); shared down Q2_0, IQ4_NL. DR1: Q8_0 for
//            all six (the MTP draft layer 48: routed and shared gate, up and down), at B <= 4 only (SP2: B = 5 is
//            rejected; the draft's FFN runs at B = 1, its catch-up has none). One kernel instance per deployed
//            (routed, shared gate, shared up) triple or (routed down, shared down) pair and B.
//   work:    the distinct experts of the B rows in first-appearance order (row 0's ten, then row 1's new ones, ...),
//            U <= 10 B. Each distinct expert's weights are read once and serve every row routed to it; rows are not
//            grouped otherwise (sheet E3). Slot s(u, b) numbers the routed (expert, row) pairs: slot_base(u) + the rank
//            of b among u's rows, slot_base(u) = the routed pairs of experts 0 .. u-1; n_slots <= 10 B of them. The
//            shared expert serves every row: its GLU of row b is slot n_slots + b (so at most 11 B slots, 55 at B = 5).
//            gate_up runs U x 10 routed units (expert, 32-channel group) and then the shared expert's 10 units on the
//            blocks with the fewer routed units; down runs U + 1 experts' items, the shared expert's last.
//   math:    gate_up, per routed pair (u, b) and per shared row b, channel c of this card:
//                g = sum_k Wgate[c][k] x_b[k],   v = sum_k Wup[c][k] x_b[k]   (x_b: row b's values, code x d)
//                glu[c] = SiLU(g) * v = g / (1 + exp(-g)) * v,  then FNFL's activation format per 32 channels
//                (fn/kernels/act/act.h: d = amax / 127, codes round to nearest even, s = d sum q)
//            down, per row b, output n: out[b][n] = sum_{k = 0..9 in route order} w[b][k] * dot_n(e_bk, glu(s(e,b)))
//                + sg[b] * dot_n(shared, glu(n_slots + b))   (the shared term added last; sg = sigmoid of the shared
//                gate, DP2's router epilogue)
//   numerics: weights as stored (the packed layouts keep every decoded value), activations in FNFL's decode format
//            (C2; F32 d). Integer dot products (IDP4A) are exact (Q8_0: int8 weights x codes, 32 a block, so |sum| <=
//            32 x 128 x 127 < 2^24 and its F32 conversion is exact too); scales, sums and the GLU are F32, built
//            without fast math (--ftz=false --prec-div=true --prec-sqrt=true --fmad=true); expf and the format's
//            divisions are the precise forms (__fdiv_rn), the codes __float2int_rn. Q2_0's offset uses the exact
//            integer sum of the staged codes, not s (d x the sum, rounded): the difference of two sums in F32 would
//            lose the bits s rounded. The shared gate/up decode and dot products are DP2's
//            (fn/kernels/dense/dense_dec.cuh) on DP2's packed layout. Reduction orders are fixed and there are no
//            atomics, so two runs are bitwise identical; no split, tile or order touching a row's values depends on B
//            (NC1 section 2).
//            Reference: FP64 on fn/quant's decoded weights and the same activation values (code x d;
//            experts_test.cpp); bound max |err| / max |ref| <= 1e-5 for the GLU output (before quantization) and the
//            down partial, and the GLU codes and scales bitwise equal to fn/quant/act_ref applied to the kernel's own
//            F32 GLU output.
//   floor:   cold weights, 1,380 MHz: 1.949 us + (U x routed bytes + shared bytes) / 1,021.6 GB/s [M, HX2 B1] + HX12's
//            0.83 us tail (EX2's hardware speed).
//   budget:  gate_up: 72 blocks of 640 threads, __launch_bounds__(640, 1), so at most 96 registers; dynamic shared
//            memory: the routed grid table (1-8 KB), the IQ3_S grid when a shared matrix needs it and the routed format
//            does not carry it, B x (2,560 staged codes + 1,280 run meta), 2 x B x 64 x 21 floats of partials, meta.
//            down: 72 blocks of 544 threads (XB4: three groups of 5 R item threads), __launch_bounds__(544, 1), so
//            at most 96 registers (5 warps share one SM partition's 16K); Q8_0 keeps the 512-thread walk (128
//            registers). 68.4 KB dynamic shared memory (B <= 4), 85.5 KB at B = 5 [C]. SP2, B = 5 gate_up: the first
//            routed row's activations in registers and each further row's read from shared memory at each record, so
//            the live registers do not grow with B (SP2 design, "Experts"). Counts, times and the frozen forecast:
//            work/EX2 (results.json); ptxas reports: work/EX2/provenance/ and work/XB4/.
//   evidence: HX4, HX5, HX12, EX1 C1-C3, EX1, EX1b and C2 (the routed recipe), DP2 (the shared expert's recipe), EX2
//            (the fold and its measurements).
//
// Packed weight layout (FNFL owns loading; pack_gate_up / pack_down build it from the GGUF bytes, a permutation of
// bytes that keeps every decoded value):
//   gate/up, per expert: 10 tiles (one per group of 32 channels, gate and up rows together; experts_dev.cuh has the
//   record layout), gate_up_bytes(f) bytes per expert, the GGUF rows' byte count; down, per expert: down_bytes(f), the
//   GGUF bytes of the card's 320 columns. The shared expert's gate and up are fn::dense::pack's layout of its card's
//   320 rows each; its down is pack_down's layout (one expert).
// Activations, in FNFL's decode format (fn/kernels/act/act.h), codes and scales each 16-byte aligned: x is B rows of
// 2,560 (codes [B][2560], scales [B][80]); glu is 55 slots of 320 (codes [55][320], scales [55][10]; down reads its
// first 11 B slots as 16-byte loads); ids int32 and wts F32 [B][10]; sg F32 [B]; out F32 [B][2560]; glu_f32 (test only,
// nullable) F32 [55 slots][320].
// grids: the device copy of grid_table() (grid_bytes bytes), 16-byte aligned.

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"

#include <cstddef>
#include <cstdint>
#include <limits>

// SP2: the capacity helpers below are constexpr on both sides; a host compiler sees plain functions (version.h's way)
#if defined(__CUDACC__)
#define FN_EXPERTS_HD __host__ __device__
#else
#define FN_EXPERTS_HD
#endif

namespace fn::experts {

// q8_0: gate/up and down. ME1: q4_k, q5_k (gate/up) and q5_1 (down), the Unsloth Q4 candidate's routed formats
enum class fmt : int32_t { iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, q2_0, iq4_nl, q8_0, q4_k, q5_k, q5_1 };

constexpr int32_t n_ff_card = 320; // channels a card owns per expert
constexpr int32_t max_rows = 5;     // SP2: the five-row verification pass
constexpr int32_t max_slots = 50;   // routed (expert, row) pairs at B = 5
constexpr int32_t glu_slots = max_slots + max_rows; // and the shared expert's rows
// SP2: the routing layout (the meta struct, the unit order, the down kernel's staging) a launch of B rows uses: its
// capacity, 4 for B <= 4 (the layout and code those launches had before SP2) and 5 for B = 5
FN_EXPERTS_HD constexpr int32_t cap_of(int32_t rows) {
    return rows <= 4 ? 4 : 5;
}

// The decode buffers of one card at max_rows, each 256-byte aligned (R1's reservation), in this order: x codes
// [5][2560] and scales [5][80], glu codes [55][320] and scales [55][10], ids I32 and wts F32 [5][10], out F32
// [5][2560], status U32 [1], the routing meta (meta_bytes, XB2). glu_f32 is test only.
constexpr int32_t n_ws = 9;
// XB2: the launch's routing meta (experts_meta.cuh): the distinct experts, their rows and slots, then the units'
// order. XB3: gate_up builds it in every block (as 71b92a9's build_meta), block 0 writes the struct here and each block
// its own ranks of the order; down reads the struct at entry instead of building it
constexpr int32_t meta_bytes = 1840; // SP2: the capacity-5 region (1,836 bytes, 16-aligned); capacity 4 uses 1,476
// the region's I32 words as a host reads them: n_u, n_slots, bad (0 or 1), e[S] (u < n_u), mask[S] (rows of u, bit
// b; 0 from n_u on), slot_base[S] (u < n_u), u_of[S] (entry p = 10 b + k < 10 B), S = 10 cap_of(B) slots; other words
// are unspecified. meta_w_*: capacity 4 (B <= 4); meta_view_of gives either capacity's.
constexpr int32_t meta_w_n_u = 0, meta_w_n_slots = 1, meta_w_bad = 2, meta_w_e = 3, meta_w_mask = 43,
                  meta_w_slot_base = 83, meta_w_u_of = 123;
// XB2 (d): gate_up's unit order, U16 entries from byte meta_order on: entry r (r < 10 U + 10) is the launch's unit at
// rank r, routed unit (u, g) as 10 u + g and shared unit g as order_shared + g. The ranks sort the units by cost
// class, heaviest first (unit_order); block i of the 72 runs ranks 72 k + i (k even) and 72 k + 71 - i (k odd).
// meta_order, order_shared and max_units: capacity 4; the _of forms give either capacity's.
constexpr int32_t meta_order = 656;
constexpr int32_t unit_groups = n_ff_card / 32;           // gate/up units of an expert on a card (32 channels)
constexpr int32_t order_shared = 40 * unit_groups;        // 400
constexpr int32_t max_units = order_shared + unit_groups; // 410
FN_EXPERTS_HD constexpr int32_t meta_order_of(int32_t cap) {
    return cap <= 4 ? 656 : 816;
}
FN_EXPERTS_HD constexpr int32_t order_shared_of(int32_t cap) {
    return 10 * cap * unit_groups; // 400, 500
}
FN_EXPERTS_HD constexpr int32_t max_units_of(int32_t cap) {
    return order_shared_of(cap) + unit_groups; // 410, 510
}
static_assert(meta_order_of(4) + 2 * max_units_of(4) <= meta_bytes &&
                  meta_order_of(5) + 2 * max_units_of(5) <= meta_bytes,
              "the order fits the region");
// SP2: the region's words at a launch of B rows (host form of experts_meta.cuh's meta_c<cap_of(B)>)
struct meta_view {
    int32_t n_u, n_slots, bad, e, mask, slot_base, u_of; // I32 words
    int32_t order;                                       // byte offset of the unit order
    int32_t slots;                                       // S
};
FN_EXPERTS_HD constexpr meta_view meta_view_of(int32_t rows) {
    const int32_t s = 10 * cap_of(rows);
    return {0, 1, 2, 3, 3 + s, 3 + 2 * s, 3 + 3 * s, meta_order_of(cap_of(rows)), s};
}
static_assert(meta_view_of(4).mask == meta_w_mask && meta_view_of(4).slot_base == meta_w_slot_base &&
                  meta_view_of(4).u_of == meta_w_u_of,
              "the capacity-4 view");
// the cost classes' weights: a routed unit's by its expert's active rows (1 to 4 in w[0..3]), then the shared unit's
// (w[4]); SP2: a 5-row unit's is class_w5 (B = 5 launches only). Finite and positive. Ties go to the class with more
// rows, the shared class after a routed one.
constexpr int32_t n_classes = 5;
struct class_table {
    float w[n_classes];
};
// SP2: a 5-row unit's weight (capacity 5), not held (the B <= 4 launches' parameters stay as they were): the
// 4-row weight plus the 3-to-4 step, w[3] + (w[3] - w[2]) [H, extrapolated; no fit at 5 rows]
inline float class_w5(const class_table & cw) {
    return 2.0f * cw.w[3] - cw.w[2];
}
// every weight finite and positive (a NaN fails both comparisons)
inline bool class_ok(const class_table & cw) {
    bool ok = class_w5(cw) > 0.0f && class_w5(cw) <= std::numeric_limits<float>::max();
    for (const float w : cw.w) {
        ok = ok && w > 0.0f && w <= std::numeric_limits<float>::max();
    }
    return ok;
}
// the build's table for a layer's formats (the routed gate/up format and the shared expert's gate and up): EL1's fits
class_table class_weights(fmt f, fn::dense::fmt fg, fn::dense::fmt fu);
// the host's form of the order of n_u experts with masks mask[u] at a launch of `rows` rows (its capacity's classes
// and order_shared_of): order[0 .. 10 n_u + 10) as above; the tests' reference for the device's
void unit_order(const uint32_t * mask, int32_t n_u, const class_table & cw, uint16_t * order, int32_t rows = 4);
size_t workspace_bytes();
// the byte offset of each buffer above in a workspace_bytes() allocation, and the end
void workspace_offsets(size_t (&off)[n_ws + 1]);

// bytes of one expert's packed weights on one card (0 for a format the kernel does not take)
int64_t gate_up_bytes(fmt f);
int64_t down_bytes(fmt f);

// the grid tables' buffer: iq2xxs, iq2xs, iq2s, iq3xxs, iq3s grids and kvalues_iq4nl at fixed offsets, then (FMS4
// item 2) the three IQ2 grids' selector tables, derived from the first three: 4 bytes an entry, nibble k the level code
// of byte k (8 -> 0, 25 -> 1, 43 -> 2)
constexpr size_t grid_bytes = 2048 + 4096 + 8192 + 1024 + 2048 + 16 + 1024 + 2048 + 4096;
// fill dst (grid_bytes) from the tables: src[i] and src_bytes[i] in the order above, then the selector tables; false
// on a size mismatch or an IQ2 grid byte other than 8, 25 and 43
bool grid_table(const void * const src[6], const size_t src_bytes[6], uint8_t * dst);

// Load-time repack of one expert's card half. gate and up: 320 GGUF rows of 2560 weights each (row_bytes apart);
// down: 2560 GGUF rows, of which this card's 320 columns start at down + (n * row_bytes) for row n.
bool pack_gate_up(fmt f, const uint8_t * gate, const uint8_t * up, size_t row_bytes, uint8_t * dst);
bool pack_down(fmt f, const uint8_t * down, size_t row_bytes, uint8_t * dst);

struct gate_up_args {
    const uint8_t * w;       // [512][gate_up_bytes(f)] packed
    const uint8_t * sh_gate; // the shared expert's gate: this card's 320 rows of 2560, fn::dense::pack's layout
    const uint8_t * sh_up;   // its up, alike
    const uint8_t * grids;   // grid_bytes
    fn::act::view x;         // B rows of 2560 (rows = B, k = 2560)
    const int32_t * ids;     // [B][10]
    uint8_t * meta;          // meta_bytes, 16-byte aligned, out (XB3): the routing meta of ids and its unit order by cw
    fn::act::out_view glu;   // 55 slots of 320 out (rows = 55, k = 320)
    float * glu_f32;         // [55][320] out, test only; nullptr in production
    uint32_t * status;       // [1] or nullptr: set to 1 on an expert ID outside [0, 512)
    // HC3, X-HMIX's acquire (fn/xchg/xchg.h): non-null hmix is this card's X-HMIX slot of the FFN sublayer, x is taken
    // from its codes and scales regions (leave x.codes and x.scales null), and every block polls the peer's half of
    // the runs it stages after priming its weight ring. Null: x was written before the launch.
    const float * hmix = nullptr;
    int32_t hmix_peer_half = 0; // the half the peer writes: 1 - card
    uint32_t * err = nullptr;   // the card's error word (xchg::err_poll_hmix)
    // XB3: the unit order's class weights (class_weights of the launch's formats; finite and positive, or the launch
    // is rejected)
    class_table cw = {};
    // XB3 part 6: the IQ2 body's run-time formats (fmt, fn::dense::fmt), set by gate_up; the caller's are not read
    int32_t u_f = -1, u_fg = -1, u_fu = -1;
    // SP4 (switch of ticket SP4): the gate/up mapping at B = 4 and 5 of the target formats, 0 the 640-thread kernels
    // (SP2's), 1 the paired 320-thread kernel (experts_gu.cuh gate_up_kernel_p), 2 (H2) the IQ2 formats on the
    // 640-thread IQ2 body with its compact optional-row loop and the IQ3 formats on SP2's, 3 (XB3 C1) the IQ2 formats
    // as 2 and the IQ3 formats on the IQ3 body (gate_up_kernel_u<gu_iq3>, the same loop); the same arithmetic and
    // order (bitwise the same GLU). B <= 3 and Q8_0 ignore it. Any other value is rejected
    int32_t map = 0;
    // ME1 (the expert cache): non-null, expert e's packed gate/up is wt[e] (a device table of 512 pointers, each
    // 16-byte aligned) instead of w + e x gate_up_bytes(f); w is then not read for routed experts (still non-null and
    // aligned). The arithmetic is unchanged: an address is the only difference
    const uint8_t * const * wt = nullptr;
    // ME1 (GPU-side hit resolution): with wt, skip 1 runs only the routed experts whose wt entry is non-null (the
    // others' GLU slots are not written) and the shared expert; skip 2 runs the same routed experts and not the shared
    // expert. 0 runs every expert. The meta is the whole launch's either way (down's slots do not change)
    int32_t skip = 0;
    // ME1: non-null, the launch does nothing when *run_if is 0 (read by every block at entry, written before the launch
    // in stream order)
    const uint32_t * run_if = nullptr;
};

struct down_args {
    const uint8_t * w;     // [512][down_bytes(f)] packed
    const uint8_t * sh_w;  // [down_bytes(fs)] the shared expert's down, packed as one expert (pack_down)
    const uint8_t * grids; // grid_bytes
    fn::act::view glu;     // 55 slots of 320 from gate_up (rows = 55, k = 320)
    const int32_t * ids;   // [B][10]
    const uint8_t * meta;  // meta_bytes, 16-byte aligned: the routing meta of ids, as gate_up writes it
    const float * wts;     // [B][10] router weights
    const float * sg;      // [B] the shared expert's weight: sigmoid of the shared gate (DP2's router epilogue)
    float * out;           // [B][2560] out
    uint32_t * status;     // [1] or nullptr
    // ME1 (the expert cache): wt as gate_up_args::wt, for down (down_bytes(f) an expert); non-null selects the cache's
    // instantiation (ME3). ME3: each routed slot s with bit s of ext_slots set is the CPU's: its unit's items neither
    // load nor run and the weighted sum leaves its term out (the remaining terms' order and arithmetic unchanged); the
    // CPU's products are added after this launch (experts_cache.h combine). Requires wt.
    const uint8_t * const * wt = nullptr;
    uint64_t ext_slots = 0;
    // ME1 (GPU-side hit resolution): non-null, the slot mask is *ext_slots_d (written before the launch in stream
    // order) and ext_slots is not read; requires wt
    const uint64_t * ext_slots_d = nullptr;
};

// ME8: one logical card's routing (its router's top-10 IDs and weights, its shared gate) read by another card's expert
// launches on the same device and stream: the router computed once. Non-owning; use_route points a launch pair's
// routing at it (fn/model's single-device pair paths and ex-test's "me8" check take this one function)
struct route_ptrs {
    const int32_t * ids = nullptr; // [B][10]
    const float * wts = nullptr;   // [B][10]
    const float * sg = nullptr;    // [B]
};
inline void use_route(const route_ptrs & r, gate_up_args & ga, down_args & da) {
    ga.ids = r.ids;
    da.ids = r.ids;
    da.wts = r.wts;
    da.sg = r.sg;
}

// ME1: glu_out's epilogue (experts_gu.cuh) for gate/up sums made elsewhere (the CPU products, experts_cpu.h): row i of
// n (n <= max_slots) takes gs[i][c], us[i][c] (F32 [n][320]) as the sums of channel c and writes the GLU in FNFL's
// activation format to codes [n][320] and scales {d, s} [n][10], bitwise as gate_up would from the same sums. One
// warp a (row, 32-channel group); false (nothing enqueued) on a bad n or a null or misaligned pointer.
bool glu_finish(const float * gs, const float * us, int32_t n, int8_t * codes, float * scales, void * stream);

// Enqueue on stream (a cudaStream_t). Return false, with nothing enqueued, for a format combination outside the
// deployed list, an unsupported row count or a null or misaligned pointer; return false after a failed launch
// submission (the CUDA error is left for the caller).
bool gate_up(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, void * stream);
bool down(fmt f, fmt fs, int32_t rows, const down_args & a, void * stream);

// ME10: one device and stream: both logical shards' gate/up of one cache pass in one launch, 72 blocks, 36 a shard.
// Block i runs shard k = i / 36's arguments s[k] as its block i % 36 of 36 (the unit deal over 36 blocks) with
// gate_up's unit arithmetic, staging and reduction order, so each shard's GLU, meta and status are bitwise gate_up's.
// Supported: an IQ2 or IQ3 routed format with (fg, fu) deployed, rows 4 or 5, each shard valid for gate_up with map 3
// and a table (wt), both shards with the same ids, skip, run_if and class table and the same activation form (x or
// hmix), and distinct meta, GLU, F32 GLU and status buffers. Each shard keeps its own table, weights, hmix slot and
// error word. Unsupported: the caller runs gate_up per shard. gate_up_pair: false, nothing enqueued, when unsupported;
// false after a failed launch submission
struct gate_up_pair_args {
    gate_up_args s[2];
};
bool gate_up_pair_supported(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_pair_args & a);
bool gate_up_pair(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_pair_args & a,
                  void * stream);

// FMU2 (build item 1): a joint round's two to four lanes (fn/runtime/joint.h) at B = rows (FMS13: the round's R,
// 2 .. 5, every lane alike; lane k's rows at the five-row lane stride) on one card in one 72-block launch each, the
// lanes' units never mixed. gate_up_lanes: lane k of n runs on blocks k NB .. k NB + NB - 1 (NB =
// 72 / n: 36, 24, 18) as its own gate_up launch's deal over NB blocks (ME10's form of the snake deal), with
// gate_up_kernel_u's body on its own arguments s[k], so each lane's GLU, meta and status are bitwise its own gate_up
// launch's (every unit's arithmetic, staging and reduction order are the block's alone). down_lanes: block i runs lane
// k = i / NB and in it the lone launch's blocks n (i % NB) .. n (i % NB) + n - 1 in turn, their output rows and walk
// unchanged, after one staging of the lane's GLU slots (which block runs an item changes no result: down_kernel), so
// each lane's out is bitwise its own down launch's. Supported: the resident target formats down's and gate_up's map 2
// or 3 bodies take at B = rows (IQ2 routed gate/up at rows 2 .. 5 and IQ3 at 4 and 5, each with a deployed triple: the
// lone IQ3 launch at B <= 3 is the per-triple kernel, which has no lanes form; Q2_0 or IQ4_NL down), no table (wt),
// no run_if or skip, each lane valid for its lone launch, the lanes' distinct GLU, meta and out buffers (the status
// word may be shared: every writer stores 1), the same class table and activation form (x or hmix). Unsupported:
// false, nothing enqueued, and the caller launches each lane alone; false after a failed launch submission.
constexpr int32_t max_lanes = 4;
struct gate_up_lanes_args {
    gate_up_args s[max_lanes];
};
struct down_lanes_args {
    down_args s[max_lanes];
};

bool gate_up_lanes_supported(fmt                        f,
                             fn::dense::fmt             fg,
                             fn::dense::fmt             fu,
                             int32_t                    lanes,
                             int32_t                    rows,
                             const gate_up_lanes_args & a);
bool gate_up_lanes(fmt                        f,
                   fn::dense::fmt             fg,
                   fn::dense::fmt             fu,
                   int32_t                    lanes,
                   int32_t                    rows,
                   const gate_up_lanes_args & a,
                   void *                     stream);
bool down_lanes_supported(fmt f, fmt fs, int32_t lanes, int32_t rows, const down_lanes_args & a);
bool down_lanes(fmt f, fmt fs, int32_t lanes, int32_t rows, const down_lanes_args & a, void * stream);

// XB2: the routing meta of ids ([rows][10]) into meta (meta_bytes, 16-byte aligned) by a one-block launch of the
// meta builder (experts_meta.cuh): the tests' and benches' reference, and the meta of a down launch that no gate_up
// launch precedes. The model's layers take gate_up's (XB3).
// cw: the order's class table (class_weights).
bool write_meta(int32_t rows, const int32_t * ids, const class_table & cw, uint8_t * meta, void * stream);

// The deployed instantiations (experts_inst.h), in that list's order; i out of range returns false
constexpr int32_t n_gate_up_inst = 31;
constexpr int32_t n_down_inst = 5;
bool gate_up_inst(int32_t i, fmt & f, fn::dense::fmt & fg, fn::dense::fmt & fu, int32_t & first_layer);
bool down_inst(int32_t i, fmt & f, fmt & fs, int32_t & first_layer);
// ME1: the Q4 candidate's instantiations (experts_inst.h FN_EXPERTS_Q4_*), not deployed: the tests' list
constexpr int32_t n_q4_gate_up_inst = 2;
constexpr int32_t n_q4_down_inst = 2;
bool q4_gate_up_inst(int32_t i, fmt & f, fn::dense::fmt & fg, fn::dense::fmt & fu, int32_t & first_layer);
bool q4_down_inst(int32_t i, fmt & f, fmt & fs, int32_t & first_layer);
// ME7: the Q2 draft's instantiations (experts_inst.h FN_EXPERTS_DRAFT_Q2_*), opt-in: the tests' list
constexpr int32_t n_draft_q2_gate_up_inst = 1;
constexpr int32_t n_draft_q2_down_inst = 1;
bool draft_q2_gate_up_inst(int32_t i, fmt & f, fn::dense::fmt & fg, fn::dense::fmt & fu, int32_t & first_layer);
bool draft_q2_down_inst(int32_t i, fmt & f, fmt & fs, int32_t & first_layer);

const char * fmt_name(fmt f);

// FMS2 item 6 (XC-1): down_lanes with X-F sent by the launch. Each finished output value of the half peer_half (lane
// k's row b < rows, joint row 5 k + b) is also stored to peer, canonical, joint row r at r words_half (the peer's joint
// slot, as send_half_joint writes it); a block whose routing meta is bad sends zeros for its chunks of that half. peer
// is 16-byte aligned, peer_half 0 or 1. FMS2 item 7: non-null x16_err, the send is 16-bit (fn/xchg/xchg.h
// send_half_joint16's form: FP16 halves, joint row r at r x16_row_halves halves), and a value beyond FP16's range sets
// err_x16_range there (the sending card's joint word).
bool down_lanes_send(fmt                     f,
                     fmt                     fs,
                     int32_t                 lanes,
                     int32_t                 rows,
                     const down_lanes_args & a,
                     uint32_t *              peer,
                     int32_t                 peer_half,
                     uint32_t *              x16_err,
                     void *                  stream);

} // namespace fn::experts

#undef FN_EXPERTS_HD
