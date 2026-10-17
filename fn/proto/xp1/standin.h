// LOAD STAND-IN (XP1): streams <piece>'s bytes so coordination is timed at a streaming kernel's speed; not <piece>'s math; replaced by <its kernel ticket> (the pieces and their tickets are listed in piece_table below).
#pragma once

// XP1's load stand-ins for the pieces of layers 10 and 11 that FNFL has not built (STANDARDS "Load stand-ins, for
// timing tests only"). Each one reads its piece's real input buffer, streams the piece's bytes on this card (weights
// from R1's memory plan, the GDN state, or the selected KV cells), writes the piece's real output buffer through a
// defined function, and pads with register FMAs to a set duration (XP1 Amendment 1). It is never a correctness
// reference, never ships in a model pass, and its time is never a kernel result.
//
// Contract (every stand-in, one kernel instance a piece so a trace names it):
//   shapes:   rows B in 1..4; in [B][in_width] F32, out [B][out_width] F32; streamed bytes a multiple of 16; every
//             pointer 16-byte aligned. Anything else is rejected (false, nothing enqueued).
//   output:   out[b][j] = 0.5 * in[b][j mod in_width] + c, as __fmul_rn then __fadd_rn (no FMA contraction); with add,
//             out[b][j] = add[b][j] + that value (__fadd_rn). The router writes IDs (37k + 13b + 7l) mod 512 for k =
//             0..9 (Amendment 3: or the captured IDs of the replay's decode), weights 0.1 and the shared gate 0.5.
//             BF16 writes (KV, raw index keys) are the same function with their own constant, rounded to nearest even.
//   sink:     each block writes the XOR of every 16-byte weight word it read to sink[block][0..3], and the XOR of every
//             other word it read (state, cells, inputs, the shared gate) and its padding FMAs to sink[block][4..7], so
//             no read is optimized away. sink holds sink_words(); the weights' XOR over the blocks is checkable.
//   padding:  after its bytes, each thread runs pad iterations of four independent register FMA chains; the count is
//             calibrated at startup so the piece's alone time lands on its setting. Padding computes; it does not
//             stream, so a padded stand-in loads memory less than a real kernel of the same duration.
//   send:     with peer_slot (V1), the peer's half of out ([B][1280] words) goes to the peer's X-A or X-F slot from the
//             epilogue through fn::xchg::put_run, the A3 protocol and payload rule of xchg.h; the receive is unchanged.
//   math:     built without fast math (--ftz=false --prec-div=true --prec-sqrt=true --fmad=true); only the stated adds,
//             multiplies, padding FMAs and integer XORs.
//   budget:   144 blocks of 512 threads, __launch_bounds__(512); plain coalesced 16-byte loads, no tuning.

#include <cstddef>
#include <cstdint>

namespace fn::xp1 {

enum class piece : int32_t {
    gdn_proj_conv = 0, // layer 10: GDN projections and convolution (qkv, z, alpha, beta, conv1d, a, dt; history)
    gdn_recur,         // layer 10: L2 norms, recurrence, gated norm (the GDN state)
    gdn_out,           // layer 10: output projection (ssm_out)
    index_q_proj,      // layer 11: index query projection (BF16 [2560, 512])
    index_k_proj,      // layer 11: index key projection (BF16 [2560, 128]) and the raw key write
    qsa_qkv,           // layer 11: q (with gate), k, v, their norms and rope, the KV write
    qsa_attention,     // layer 11: attention over each row's selected cells, with its gate
    qsa_out,           // layer 11: output projection (attn_output)
    router10,          // layer 10: router and shared-expert gate (BF16), top-k
    sh_gate_up10,      // layer 10: shared expert gate and up, GLU
    sh_down10,         // layer 10: shared expert down, gate scale
    router11,
    sh_gate_up11,
    sh_down11,
};
constexpr int32_t n_piece = 14;

struct piece_info {
    const char * name;        // as XP1's tables name it
    int32_t layer;            // 10 or 11
    const char * target_key;  // its key in tickets/XP1/targets.json (the shared expert's two halves share one)
    const char * replaced_by; // the ticket whose kernel replaces it
};
extern const piece_info piece_table[n_piece];

constexpr int32_t si_blocks = 144;
constexpr int32_t si_threads = 512;
constexpr int32_t sink_block_words = 8;
constexpr size_t sink_words() {
    return size_t(si_blocks) * sink_block_words;
}

struct standin_args {
    int32_t rows = 0;
    const float * in = nullptr; // [B][in_width] the piece's input
    int32_t in_width = 0;
    float * out = nullptr; // [B][out_width] the piece's output, or null (the router)
    int32_t out_width = 0;
    float c = 0.0f;              // the piece's constant
    const float * add = nullptr; // [B][out_width] added to out (V1's shared down: the routed partial)
    float * peer_slot = nullptr; // V1: the peer's X-A or X-F slot (peer memory); out_width must be 2560
    int32_t peer_card = 0;
    const void * w = nullptr; // the weights, w_bytes
    int64_t w_bytes = 0;
    const void * st = nullptr; // a state read (GDN state or history version 0), st_bytes
    int64_t st_bytes = 0;
    void * st_dst = nullptr; // st_copies versions of st_bytes each, written from it (the per-row versions)
    int32_t st_copies = 0;
    // the gather (attention): row b reads cells[b * cell_stride + i], i < cell_width, each cell's K and V rows of 512
    // bytes. With decode non-null, cells is offset by *decode * cells_decode_stride (Amendment 3's captured cells).
    // Or, with body non-null (and cells null), row b reads its own selection from QA1b's body (fn::qsa's layout, row b
    // at body_off[b] bytes: K, t, K block IDs, t tail cells) through the logical map, block k = cells 4k .. 4k + 3:
    // its 4 K + t cells, i < cell_width.
    const int32_t * cells = nullptr;
    const uint32_t * body = nullptr;
    const int32_t * body_off = nullptr;
    int32_t cell_width = 0;
    int32_t cell_stride = 0;
    int64_t cells_decode_stride = 0;
    const void * kv_k = nullptr;
    const void * kv_v = nullptr;
    // BF16 writes: bf[i][(bf_cell0 + b - bf_lo) * bf_width + j] for rows bf_lo .. bf_hi - 1, j < bf_width, with
    // constant bf_c[i] (the KV write: K and V; the raw index keys: one)
    uint16_t * bf[2] = {nullptr, nullptr};
    float bf_c[2] = {0.0f, 0.0f};
    int32_t bf_width = 0;
    int64_t bf_cell0 = 0;
    int32_t bf_lo = 0, bf_hi = 0;
    // the router: ids [B][10], wts [B][10], sg [B]; layer for the synthetic IDs, or route_src [n_decode][3][10] for the
    // captured ones of decode *decode
    int32_t * ids = nullptr;
    float * wts = nullptr;
    float * sg = nullptr;
    int32_t layer = 0;
    const int32_t * route_src = nullptr;
    const int32_t * decode = nullptr; // [1] the replay's decode index (Amendment 3), device memory
    const float * aux = nullptr;      // read into the sink: the shared gate the shared expert's down is scaled by
    int32_t aux_n = 0;
    uint32_t * sink = nullptr; // sink_words()
    int32_t pad = 0;           // padding iterations a thread
};

// Enqueue piece p's stand-in on stream; false (nothing enqueued) on a rejected argument or a failed submission.
bool standin(piece p, const standin_args & a, void * stream);

} // namespace fn::xp1
