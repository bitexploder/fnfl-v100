#pragma once

// The token embedding on one card (E1; FC-EMB1, simulator node Embed): each row's token_embd row, the card's 1,280
// features, decoded from IQ3_S to F32 and written into all four of the card's half-streams, the boundary layer 0's
// first HC mix reads (FC-LAY1: hres = the row, hpost = 0, hx = 0). Pinned source: qwen4exp.cpp build (lines 455-466
// at 66e11c3): the residual starts as hc identical copies of the token's embedding.
//
// Contract (embed):
//   shapes:  rows B in {1, .., 5} (SP2); anything else, a null or misaligned pointer, is rejected (false, nothing
//            enqueued; FC-G1). token_embd on card r (fn/model/placement, layout gguf): features r*1280 .. of every
//            token, 5 IQ3_S superblocks (550 bytes) a token, tokens in ID order.
//   tokens:  tok[B] is device memory (FC-RD1). check_tokens rejects an ID outside [0, V) on the host before upload;
//            the kernel checks again and, for such an ID, ORs err_token into *err and writes nothing for that row
//            (never decodes it: FC-EMB1, the round fails).
//   math:    exact: each value is the superblock quarter's scale times its signed grid byte, one rounded F32 product,
//            DP2's IQ3_S decode (fn/kernels/dense/iq3s.cuh); bitwise fn/quant's decode_iq3_s (C1b). Built without
//            fast math (--ftz=false --prec-div=true --prec-sqrt=true --fmad=true); no FMA can form (one multiply).
//            Rows are independent, so each row is bitwise independent of B (NC1 section 2).
//   budget:  B blocks of 80 threads (thread: stream, superblock, quarter), 2 KB static shared memory for the grid.
//   floor:   reads 550 B a row (+2 KB grid), writes 25.6 KB a row; ramp-bound: about 1.95 us ramp [M, HX12] plus two
//            dependent reads (token, then row) [E].
//   evidence: E1 (ple-test embedding cases, ple-bench).
//
// Layouts (card r, F32): hres [B][4][1280], hpost [B][4], hx [B][1280]; 16-byte aligned. grids: fn::dense's grid
// buffer (dense.h grid_table: iq3s_grid's 512 words first).

#include "fn/xchg/xchg.h"

#include <cstdint>

namespace fn::embed {

constexpr int32_t n_vocab = 248320;  // V (FC-K)
constexpr int32_t n_own = 1280;      // features a card owns
constexpr int32_t n_stream = 4;      // HC streams
constexpr int32_t n_super = 5;       // IQ3_S superblocks of a card's row
constexpr int32_t super_bytes = 110; // block_iq3_s
constexpr int32_t row_bytes = n_super * super_bytes;
constexpr int32_t max_rows = 5; // SP2
constexpr uint32_t err_token = fn::xchg::err_token; // an ID outside [0, V) reached the kernel

struct args {
    const uint8_t * table = nullptr; // the card's token_embd slice, [V][550] bytes
    const uint8_t * grids = nullptr; // fn::dense grid buffer
    const int32_t * tok = nullptr;   // [B] device
    float * hres = nullptr;          // [B][4][1280] out
    float * hpost = nullptr;         // [B][4] out: zero
    float * hx = nullptr;            // [B][1280] out: zero
    uint32_t * err = nullptr;        // this card's error word
};

// Host check before upload: every ID in [0, V). False (with a message naming the row and ID) otherwise.
bool check_tokens(const int32_t * tok, int32_t rows);

// Enqueue on stream (a cudaStream_t); false, with nothing enqueued, for a rejected input; false after a failed launch
// submission (the CUDA error is left for the caller).
bool embed(int32_t rows, const args & a, void * stream);

// ---- PB1: the embedding over prompt rows (FC-PF3's FC-EMB1 row) ----
// embed_rows: rows 1 .. 8,192 of tok (device), each row's card half decoded as embed does (one rounded product a value,
// bitwise fn/quant's decode_iq3_s) into own [rows][1280] (the EMB message's payload), into its columns of full
// [rows][2560] (the embeddings the draft keeps) and of all four streams of S [rows][4][2560]. embed_join: the peer's
// half (its message, received) into the other columns of full and S. An ID outside [0, V) sets err_token and writes
// nothing for its row. Exact copies otherwise.
constexpr int32_t prompt_max_rows = 8192; // PB3's chunk

struct rows_args {
    const uint8_t * table   = nullptr;  // the card's token_embd slice
    const uint8_t * grids   = nullptr;
    const int32_t * tok     = nullptr;  // [rows] device
    int32_t         rows    = 0;
    int32_t         card    = 0;
    float *         own     = nullptr;  // [rows][1280]
    float *         full    = nullptr;  // [rows][2560]
    float *         streams = nullptr;  // [rows][4][2560]
    uint32_t *      err     = nullptr;
};

bool embed_rows(const rows_args & a, void * stream);
// peer: the received [rows][1280] of card 1 - a.card
bool embed_join(const rows_args & a, const float * peer, void * stream);

} // namespace fn::embed
