#pragma once

// HC (hyper-connection) kernels for one card of the two-card split: card r owns features r*1280 .. r*1280+1279 of
// all four streams. Simulator nodes HDown0/HDown1 (hc_down) and HUp0/HUp1 with HMix0/HMix1 (hc_up); WP2 section 3.
//
// Contract (both kernels):
//   shapes:  D = 2560 per stream, C = 4 streams, F = 1280 own features, 320 down rows, 4 injection rows, low rank 320.
//            Rows B in {1, 2, 3, 4} (one template instance each); any other B is rejected (returns false). SP2: the
//            HC2 .. HC4 forms (down2, up2, ready, sub and their head and Q8_0 forms) also take B = 5; HC1's down, up
//            and variants stay 1..4. At B = 5 the fused launch (sub) takes 25,600 bytes of dynamic shared memory, the
//            down phase's activations and then the up phase's arrays (static arrays alone would pass 48 KB); up2's
//            72-block shape lowers with all 15 warps, so its mix loads go to the block's first 120 lanes.
//   math, per row b (s = stream, i = own feature, card r; the moments of both cards are summed by hc_up):
//            x[s,i]    = hres[s,i] + hpost[s] * hx[i]                                   (hc_down; written out)
//            v[s]      = sum_i x[s,i]^2                                                 (hc_down)
//            u[s,j]    = sum_i Wdown[j,s,i] * gamma[s,i] * x[s,i]       j = 0..319       (hc_down)
//            q[s,k]    = sum_i Winject[k,s,i] * gamma[s,i] * x[s,i]     k = 0..3         (hc_down)
//            rms[s]    = sqrt((v_own[s] + v_peer[s]) / D + eps)                         (hc_up, eps = fn::rms_eps)
//            low[j]    = SiLU((1/4) * sum_s (u_own[s,j] + u_peer[s,j]) / rms[s])
//            inject[k] = 2 * sigmoid((1/4) * sum_s (q_own[s,k] + q_peer[s,k]) / rms[s])
//            gate[s,c] = sigmoid(sum_j Wup[s,c,j] * low[j])                             c = own feature
//            mixed[c]  = (1/4) * sum_s gamma[s,c] * x[s,c] / rms[s] * gate[s,c]
//   numerics: inputs, accumulation and outputs F32, weights BF16 (widened exactly). Built without fast math, with
//            --ftz=false --prec-div=true --prec-sqrt=true --fmad=true; expf, sqrtf and the divisions are the precise
//            forms. Intrinsics: __byte_perm (the exact BF16 widening), __ldcg/__ldcs (cache hints only) and
//            __shfl_xor_sync; no math intrinsics. x is one fmaf. Every reduction has a fixed order and no atomics, so
//            two runs are bitwise identical. Reference: FP64 on the CPU from the same BF16 weights (hc_test.cpp); bound
//            max |err| / max |ref| <= 1e-6 for each output. The division by rms is a multiply by 1/rms; v_own + v_peer
//            and u_own + u_peer are commutative in F32, so both cards compute identical low and inject.
//   floor:   (B = 3, cold weights, 1,380 MHz; 1.949 us ramp + bytes / 1,021.6 GB/s [M, HX2 B1])
//            hc_down 3,317,760 weight bytes: 5.20 us; hc_up 3,276,800 weight bytes: 5.16 us. Targets <= 6.0 us.
//   budget:  both kernels: 72 blocks (one per SM) of 480 threads, __launch_bounds__(480, 1), so up to 128 registers;
//            registers, shared memory and spills in work/HC1c/provenance/ptxas-v.txt (no spills).
//   depth:   (HC1c) each lane keeps D = 3 16-byte weight loads in flight (23 KB a SM; HX12 asks for 8-16 KiB) and
//            issues row q + 3 once row q is consumed; the refill's address carries row q's opaque zero, so ptxas keeps
//            the order. hc_down issues its staging loads, then (after a barrier) its first three weight rows. hc_up
//            issues its first three weight rows first (the order HC2 needs), then its moment and mix reads. Depths 1,
//            2, 3 and 6 are bench and test candidates (down_variant, up_variant); depth 3 was fastest for both.
//            hc_down at B = 1 and 2 may fit two blocks an SM by registers; the grid relies on the block scheduler
//            spreading one block to each SM.
//   evidence: HX12 (machine sheet), HX13 (acceptance capture of the prototype), HC1, HC1b, HC1c (stream depth),
//            fnfl/design/hc-requirements.md (v2).
//
// Weight layout (FNFL owns loading; card r, BF16 bits as uint16_t):
//   w_down[s][j][i], s = 0..3, j = 0..323, i = 0..1279 (3,317,760 bytes):
//       j < 320:  GGUF blk.N.hc_{attn,ffn}_down.weight (ne0 10240, ne1 320; BF16), row j, column s*2560 + r*1280 + i
//       j >= 320: GGUF blk.N.hc_{attn,ffn}_inject.weight (ne0 10240, ne1 4; BF16), row j - 320, same column
//   w_up[c][s][j], c = 0..1279, s = 0..3, j = 0..319 (3,276,800 bytes):
//       GGUF blk.N.hc_{attn,ffn}_up.weight (ne0 320, ne1 10240; BF16) row s*2560 + r*1280 + c, column j
//   gamma[s][i]: GGUF blk.N.hc_{attn,ffn}_norm.weight (10240, F32) element s*2560 + r*1280 + i
//   (ne0 is the contiguous dimension. Recorded from the deployed GGUF in work/HC1/gguf-hc-tensors.txt.)
//
// Activation layouts (F32, contiguous, card r's own half):
//   hres[b][s][i], x[b][s][i]: B x 4 x 1280;  hx[b][i]: B x 1280;  hpost[b][s]: B x 4;  gamma[s][i]: 4 x 1280
//   mom[b][1300]: v[s] at 0..3, u[s][j] at 4 + 320*s + j, q[s][k] at 1284 + 4*s + k (the X-HC packet, 1,300 F32 a row)
//   inject[b][k]: B x 4;  mixed[b][c]: B x 1280
// Every pointer must be 16-byte aligned (cudaMalloc gives 256) and the buffers must not overlap.

#include <cstddef>
#include <cstdint>

namespace fn::hc {

constexpr int32_t n_own = 1280;   // features a card owns per stream
constexpr int32_t n_down = 320;   // down rows (the low rank)
constexpr int32_t n_inject = 4;   // injection rows
constexpr int32_t n_rows_down = n_down + n_inject;
constexpr int32_t n_mom = 4 + 4 * n_down + 4 * n_inject; // 1,300 F32 a row
constexpr int32_t n_mom_head = 4 + 4 * n_down;           // head mode: 1,284 F32 a row (no injection)
constexpr int32_t max_rows = 5;     // SP2: the five-row verification pass
constexpr int32_t hc1_max_rows = 4; // HC1's down, up and variants

constexpr int64_t w_down_elems = int64_t(4) * n_rows_down * n_own; // 1,658,880 BF16
constexpr int64_t w_up_elems = int64_t(n_own) * 4 * n_down;        // 1,638,400 BF16
constexpr int64_t w_down_head_elems = int64_t(4) * n_down * n_own; // 1,638,400 BF16

constexpr int64_t probe_out_elems = int64_t{72} * 15;

// The activations of one sublayer on one card at max_rows, each 256-byte aligned (R1's reservation): hres and x
// [B][4][1280], hx [B][1280], hpost [B][4], mom [B][1300], inject [B][4], mixed [B][1280] and full [B][2560], F32.
size_t workspace_bytes();

struct down_args {
    const uint16_t * w_down; // [4][324][1280] BF16
    const float * hres;      // [B][4][1280]
    const float * hx;        // [B][1280]
    const float * hpost;     // [B][4]
    const float * gamma;     // [4][1280]
    float * x;               // [B][4][1280] out: the new streams
    float * mom;             // [B][1300]    out: this card's moments
};

struct up_args {
    const uint16_t * w_up;  // [1280][4][320] BF16
    const float * mom_own;  // [B][1300]
    const float * mom_peer; // [B][1300] the other card's moments (X-HC)
    const float * x;        // [B][4][1280] from hc_down
    const float * gamma;    // [4][1280]
    float * inject;         // [B][4]    out
    float * mixed;          // [B][1280] out: this card's half of the mixed input
};

// Enqueue on stream (a cudaStream_t). Return false, with nothing enqueued, for an unsupported row count or a null or
// misaligned pointer; return false after a failed launch submission (the CUDA error is left for the caller to read).
bool down(int32_t rows, const down_args & a, void * stream);
bool up(int32_t rows, const up_args & a, void * stream);

// ---- HC1c bench and test only: the depths the shipped down() and up() were chosen from ----
// HC1's kernels at stream depth 1, 2, 3 or 6 (16-byte weight loads a lane in flight); down() and up() are one of them.
bool down_variant(int32_t depth, int32_t rows, const down_args & a, void * stream);
bool up_variant(int32_t depth, int32_t rows, const up_args & a, void * stream);
// the stream-only probe: hc_down's grid and weight pattern at a depth, no staging or arithmetic; out: probe_out_elems
bool probe(int32_t depth, const uint16_t * w_down, uint32_t * out, void * stream);
// the depths the shipped down() and up() use, so the bench can count their launches with the variants'
struct variant_id {
    int32_t depth_down, depth_up;
};
variant_id shipped();

// ---- HC2: the sublayer on both cards, with X-HC and X-HMIX through fn/xchg/ ----
// Per card r and sublayer n, on one stream: down2 (sends X-HC(n) to the peer as each block finishes; resets this card's
// X-HC slot of sublayer n-1 and the peer's half of its X-HMIX slot of sublayer n-1), up2 (polls X-HC(n), writes the
// mixed row's own half in both forms into both cards' X-HMIX(n) slots), then the consumers, which acquire X-HMIX(n)
// themselves (fn/xchg; HC3). Slots follow xchg.h's rule (s(n+1) != s(n)). Outputs are bitwise identical to down() and
// up() run as two halves on one card (both cards sum the moments in card order, card 0 then card 1), except that NaN
// moments and mixed values are canonicalized (xchg.h) in both copies.
//
// HC3, up2's epilogue (both arms and the head mode): mixed is written in both forms, F32 and FNFL's decode activation
// format (act.h), by the rule of act.h, bitwise fn/quant/act_ref of the F32 row (C2's standalone quantize, which this
// replaces). The codes and scales are a function of a run's 32 values alone.
// HC4 (B2), arm 0 and the head mode: up2 runs in the run-aligned shape, 40 blocks of 640 threads, block g owning run
// g's 32 features (all four streams), so the block quantizes its own run straight from its warps with
// act::quantize_run8: no run counters, fences or second completion. Every value takes arm 0's operations in arm 0's
// order, so the outputs are bitwise HC3's. runs[] is no longer touched by arm 0; arm A and the Q8_0 forms keep HC3's
// 72-block shape: a run of 32 spans two or three of their blocks (17 or 18 features each), so a block's F32 stores are
// published (__threadfence, then an atomicAdd of its feature count on the run's counter, runs[]) and the block that
// completes the run acquires (__threadfence), reads the run's 32 values of every row from this card's slot (ld.cg),
// quantizes them and writes the codes and {d, s} into both slots; it resets the counter.

struct down2_args {
    down_args d;      // as down(); d.mom is this card's own moments (the X-HC payload layout)
    float * hc_peer;  // the peer's X-HC receive slot for this sublayer (peer memory)
    float * hc_reset; // this card's X-HC slot of the previous sublayer, reset to the sentinel (all max_rows rows); null: none
    // HC3: this card's X-HMIX slot of the previous sublayer, its peer half reset to the sentinels (all max_rows rows;
    // xchg.h step 4) after every reader; null: none
    float * hmix_reset = nullptr;
};

struct up2_args {
    const uint16_t * w_up;  // [1280][4][320] BF16
    const float * mom_own;  // [B][1300] from down2
    const float * hc_ring;  // [B][1300] this card's X-HC receive slot for this sublayer (written by the peer)
    const float * x;        // [B][4][1280] from down2
    const float * gamma;    // [4][1280]
    float * inject;         // [B][4]    out
    float * hmix_own;       // this card's X-HMIX slot for this sublayer: the own half written here (xchg.h hmix_layout)
    float * hmix_peer;      // the peer's X-HMIX receive slot for this sublayer (peer memory), the same half
    uint32_t * runs;        // [40] run counters (arm A and Q8_0 only, HC4): zeroed once; every launch leaves them 0
    uint32_t * err;         // this card's error word (fn::xchg::err_poll_hc when a poll gives up)
    int32_t card;           // 0 or 1
};

constexpr int32_t n_runs_card = n_own / 32; // up2's run counters a card (40)

// The X-HMIX stand-in consumer (tests, benches and XP1's prototype; the model's consumers acquire X-HMIX themselves):
// polls the peer half of row b of the slot (every word of the three regions), reads the own half, and writes the full
// row: F32, and (when codes is not null) codes and scales in act.h's layout. reset 1 resets the peer half's words of
// the rows it read (down2's job in the model; a second reset before the slot's next write is harmless, xchg.h).
struct ready_args {
    float * slot;   // this card's X-HMIX slot for this sublayer (hmix_layout)
    float * full;   // [B][2560] out: card 0's half, then card 1's
    int8_t * codes; // [B][2560] out, or null
    float * scales; // [B][80][2] out (with codes)
    uint32_t * err; // this card's error word (fn::xchg::err_poll_hmix)
    int32_t card;   // 0 or 1
    int32_t reset;  // 1: reset the peer half's words after reading them; 0: leave the slot
};

// hc_up's two arms (HC2): arm 0 is HC1c's hc_up with the ring; arm A stages the block's weights in shared memory while
// X-HC is in flight. Both give bitwise identical outputs.
enum class up_arm : int32_t {
    ring = 0,
    staged = 1,
};

// Per device, before the first up2 of arm A on it (and outside stream capture): arm A needs 46,080 bytes of dynamic
// shared memory a block. False if the attribute cannot be set.
bool prepare2();

// Enqueue on stream; false, with nothing enqueued, for an unsupported row count or arm or a null or misaligned pointer;
// false after a failed launch submission.
bool down2(int32_t rows, const down2_args & a, void * stream);
bool up2(up_arm arm, int32_t rows, const up2_args & a, void * stream);
bool ready(int32_t rows, const ready_args & a, void * stream);

// ---- HC4 (B3): the sublayer in one launch ----
// One cooperative launch a card (cudaLaunchCooperativeKernel: 72 blocks of 640 threads, one an SM, every block
// co-resident), the model graphs' HC launch. Mode 3: every block runs the down phase as down2 (x out, its moments to
// the peer's X-HC slot and to the own region of this card's slot, xchg.h HC4; the previous sublayer's X-HC and X-HMIX
// slots reset first), then blocks 0..39 run the up phase as up2 arm 0 (the run-aligned shape) polling both cards'
// moments from this card's slot, with x recomputed from hres, hx and hpost (down's fmaf), and write the mixed row in
// both forms into both cards' X-HMIX slots. Mode 1 runs only the down phase and mode 2 only the up phase: the eager
// pair order, where every send precedes its read (Compute Sanitizer). x, inject and both forms of mixed are bitwise
// down2 + up2 arm 0's (and so HC3's); the moments are not an output. Head mode (head_sub): as head_down2 + head_up2,
// inject ignored.
struct sub_args {
    const uint16_t * w_down; // [4][324][1280] BF16 ([4][320][1280] in head mode)
    const uint16_t * w_up;   // [1280][4][320] BF16
    const float * hres;      // [B][4][1280]
    const float * hx;        // [B][1280]
    const float * hpost;     // [B][4]
    const float * gamma;     // [4][1280]
    float * x;               // [B][4][1280] out: the new streams
    float * inject;          // [B][4] out (head mode: not written, may be null)
    float * hc_ring;         // this card's X-HC slot for this sublayer: the peer's moments, then its own (hc_own_at)
    float * hc_peer;         // the peer's X-HC slot for this sublayer (peer memory)
    float * hc_reset;        // this card's X-HC slot of the previous sublayer, reset whole; null: none
    float * hmix_reset;      // this card's X-HMIX slot of the previous sublayer, reset whole; null: none
    float * hmix_own;        // this card's X-HMIX slot for this sublayer (hmix_layout)
    float * hmix_peer;       // the peer's X-HMIX slot for this sublayer (peer memory)
    uint32_t * err;          // this card's error word (fn::xchg::err_poll_hc when a poll gives up)
    int32_t card;            // 0 or 1
};

// Enqueue on stream; false, with nothing enqueued, for an unsupported row count or mode (1, 2 or 3) or a null or
// misaligned pointer; false after a failed launch submission.
bool sub(int32_t rows, int32_t mode, const sub_args & a, void * stream);
bool head_sub(int32_t rows, int32_t mode, const sub_args & a, void * stream);

// ---- FMU2: a joint round's sublayer, lanes of max_rows rows, one launch ----
// One cooperative launch a card (sub's grid and block) for 2..4 lanes on one weight set, a[k] lane k's sub arguments
// at max_rows (its own inputs, outputs, X-HC and X-HMIX slots and error word: each lane its own ring). Build order
// item 1 (Fable review 2): one weight stream. Each warp holds its weight rows (down 5, up 8) widened to F32 for every
// lane's rows; each row of each lane takes sub's B = max_rows arithmetic (dot8's fmaf chain, the lane sums in the
// butterfly's xor order as a reduce-scatter, the finishes' orders), so every output of a lane (x, inject, both forms
// of mixed in both cards' X-HMIX slots, its X-HC words, its resets, its error word) is bitwise its own sub launch's.
// Dynamic shared memory up to 88,960 bytes (four lanes). Modes as sub's (1 the down phases, 2 the up phases, 3 both).
// False, with nothing enqueued, for lanes outside 2..4, any lane sub would refuse at max_rows, lanes on different
// cards or weights (w_down, w_up, gamma), two lanes sharing an X-HC or X-HMIX slot, or a lane's reset of a slot
// another lane writes; false after a failed launch submission. head_sub_joint: head_sub's form (no inject).
bool sub_joint(int32_t lanes, int32_t mode, const sub_args * a, void * stream);
bool head_sub_joint(int32_t lanes, int32_t mode, const sub_args * a, void * stream);

// ---- ME12: both logical shards of a sublayer on one device, one launch ----
// One cooperative launch (as sub's: 72 blocks of 640 threads, one an SM) that does both cards' sub mode 3 on one device
// and stream: block b runs card 0's down block b, then card 1's; a grid barrier; block 0 forms the low-rank activation
// and the inverse RMS once (and, normal mode, the injection, written to both cards' inject) into scratch; a grid
// barrier; then the 80 up blocks of both cards (block b: up tasks b and b + 72) read them from scratch instead of each
// forming them. Every output (x, inject, both X-HC regions of both slots, both X-HMIX slots in both forms, the resets
// and the error words) is bitwise the two cards' sub launches (modes 1, 1, 2, 2 in the pair order): the same operations
// in the same order on the same canonical moment words. A moment poll that gives up sets err_poll_hc in both error
// words. Head mode (head_pair): as head_sub, no injection.
struct pair_args {
    sub_args s[2];  // card k's sub arguments (s[k].card == k); s[k].hc_peer == s[1-k].hc_ring, likewise X-HMIX
    float * scratch; // [B][320] low then [B][4] inverse RMS, F32, 16-byte aligned; read and written by this launch only
};
constexpr int64_t pair_scratch_bytes = int64_t(max_rows) * (n_down + 4) * 4; // 6,480
// Per device, before a capture: false unless the device supports cooperative launches and every instance (B = 1..5,
// both modes) is co-resident at 72 blocks of 640 threads (the occupancy API), with no local memory.
bool pair_prepare();
// Enqueue on stream; false, with nothing enqueued, for an unsupported row count, a shard argument hc::sub would refuse,
// cards other than 0 then 1, slots that are not one pair's, a reset (by either shard) of either shard's current X-HC
// or X-HMIX slot, or a null or misaligned scratch; false after a failed launch submission. Both cards' buffers must be
// on the stream's device (the caller's placement check).
bool pair(int32_t rows, const pair_args & a, void * stream);
bool head_pair(int32_t rows, const pair_args & a, void * stream);

// ---- V1: the HC head (FC-HEAD1), HC2 with injection off ----
// The pinned engine's final mixer (qwen4exp.cpp 534-546 at 66e11c3): build_hc_mix with the output_hc_* weights and no
// injection weight. It is down2, up2 (arm 0) and ready with the head-mode template argument, so HC2's layer
// instantiations are unchanged (SASS identical, V1 checks/). The contract above holds with these differences:
//   - no injection rows: w_down is [4][320][1280] (GGUF output_hc_down.weight, ne0 10240, ne1 320; BF16, row j, column
//     s*2560 + r*1280 + i); w_up and gamma as above from output_hc_up.weight and output_hc_norm.weight;
//   - the moments are mom[b][1284] (v[s] at 0..3, u[s][j] at 4 + 320*s + j): X-HC carries 1,284 words a row in the
//     same slots (the slot stays sized for 1,300; down2's reset of the previous slot still covers the whole slot);
//   - no inject output (up2_args::inject is ignored and may be null);
//   - (HC3) the mixed row in both forms into X-HMIX, as a sublayer's; its reader is the vocabulary logits launch;
//   - bound: the full mixed row max |err| / max |ref| <= 1e-5 a row (V1), reference FP64 build_hc_mix without
//     injection (head_test.cpp).
// The streams x [B][4][1280] are written to down2_args::d.x, so the inputs hres, hx and hpost (layer 47's boundary,
// FC-HEAD3) are only read. Both cards' X-HMIX slots then hold the full mixed row [B][2560], identical on both cards.
bool head_down2(int32_t rows, const down2_args & a, void * stream);
bool head_up2(int32_t rows, const up2_args & a, void * stream);

// ---- DR1 Stage B: Q8_0 weights (the MTP draft block's two sublayers and its HC head) ----
// The draft's HC tensors are Q8_0 in the GGUF (blocks of 32 along ne0, each an FP16 scale d and 32 int8 q). down2_q8,
// up2_q8 (arm 0 only) and their head forms are down2, up2 and head_down2 / head_up2 with that weight format, as a
// template argument, so the BF16 instantiations are unchanged. The contract above holds with these differences:
//   - weights: each weight is decoded exactly to F32 as d * q (an int8 times an FP16 significand: at most 19
//     significant bits) and enters the same FMAs in the same order as the BF16 weights do. With weights that BF16 also
//     holds exactly (d a power of two), the outputs are bitwise those of the BF16 kernels (hc-test checks this); in
//     general they are the same equations on the Q8_0 weights' exact values. The int8 values are widened without I2F
//     (a byte permute into 2^23 + q + 128 and one exact subtraction), and d by the exact FP16 to F32 conversion.
//   - bound: as the BF16 kernels', against FP64 from the decoded Q8_0 weights (hc2-test, the head included).
//   - layout (card r; FNFL's repack, planar so every lane's loads stay aligned):
//       w_down: qs[s][j][i] int8 (s = 0..3, j as w_down's rows, 324 or 320 in head mode, i = 0..1279), then
//               d[s][j][i / 32] FP16 bits: GGUF row j's block (s*2560 + r*1280 + i) / 32;
//       w_up:   qs[c][s][j] int8 (c = 0..1279, s = 0..3, j = 0..319), then d[c][s][j / 32] FP16 bits: GGUF row
//               s*2560 + r*1280 + c, block j / 32.
//     A lane takes 8 consecutive q of one block (one 8-byte load) and that block's d (one 2-byte load).
constexpr int64_t w_down_q8_bytes = int64_t(4) * n_rows_down * (n_own + n_own / 32 * 2); // 1,762,560
constexpr int64_t w_down_head_q8_bytes = int64_t(4) * n_down * (n_own + n_own / 32 * 2); // 1,740,800
constexpr int64_t w_up_q8_bytes = int64_t(n_own) * 4 * (n_down + n_down / 32 * 2);       // 1,740,800

struct down2_q8_args {
    down2_args a;      // as down2; a.d.w_down is not read
    const uint8_t * w; // Q8_0 w_down in the layout above, 16-byte aligned
};

struct up2_q8_args {
    up2_args a;        // as up2; a.w_up is not read
    const uint8_t * w; // Q8_0 w_up in the layout above, 16-byte aligned
};

bool down2_q8(int32_t rows, const down2_q8_args & a, void * stream);
bool up2_q8(int32_t rows, const up2_q8_args & a, void * stream);
bool head_down2_q8(int32_t rows, const down2_q8_args & a, void * stream);
bool head_up2_q8(int32_t rows, const up2_q8_args & a, void * stream);

} // namespace fn::hc
