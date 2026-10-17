#pragma once

// dp-bench's cold state (DP2, rule R17): before a cold launch, a read of a buffer larger than L2 and a long
// straight-line kernel on every SM, so the launch fetches its weights, activations and code from DRAM, as kernels do in
// a real round after a weight stream. Measurement scaffolding, not part of a model pass. Copied from XP1's timing.cu at
// xp1-attr (3530eae), flush_kernel and icache_kernel, unchanged but for names.

#include <cstdint>
#include <functional>

namespace fn::dense {

// L2 ends full of the buffer's clean lines (reads, not writes, so the timed kernel pays no write-back); bytes a
// multiple of 16, the buffer 16-byte aligned
bool cold_flush(void * scratch, int64_t bytes, void * stream);

// about 12,288 distinct FFMA instructions (about 192 KB of code) by one warp on each of 160 blocks, so every SM's
// instruction caches hold this code and not the previous kernel's; out: one float of device memory (practically never
// written)
bool cold_icache(float * out, void * stream);

// CR1 (R4 on the bench): cold_flush's reads through ld.global.cs (__ldcs), so L2 allocates the buffer's lines
// evict-first; scratch: a buffer of its own (not cold_flush's), bytes a multiple of 16
bool cold_flush_ef(void * scratch, int64_t bytes, void * stream);

// CR1: the buffers of the R4 cold states: cold_flush's buffer, cold_flush_ef's (a different one), their size, and
// cold_icache's out
struct r4_bufs {
    void * flush;
    void * flush_ef;
    int64_t bytes;
    float * evict;
};

// CR1: the cold state before a unit in R4 variant 0..3. Every variant starts with today's (cold_flush, cold_icache), so
// a unit's state does not depend on the unit before it and a pre-touch starts equally cold in graphs A and B:
//   0 (V0, today): flush, evict
//   1 (V1, R4): flush, evict, pre, flush_ef, evict
//   2 (V2, control): flush, evict, flush_ef, evict
//   3 (V3, control): flush, evict, pre, flush, evict
// pre: the pre-touch launch (the unit's kernel on other weights and buffers), called in variants 1 and 3 only
bool cold_r4(int32_t variant, const r4_bufs & b, void * stream, const std::function<bool()> & pre);

// CR1's retention probe (dp-bench l2keep). l2keep_stream: ef_bytes of ef read evict-first and normal_bytes of normal
// read with normal loads, interleaved evenly in one pass (normal_bytes may be 0); both multiples of 16.
bool l2keep_stream(void * ef, int64_t ef_bytes, void * normal, int64_t normal_bytes, void * stream);

// l2keep_walk: n_walks blocks of one thread, block b walking n_steps 32-byte sectors from sector b, each load's address
// from the previous load's value (word 0 of each sector: the next sector's index); cycles[b * n_steps + k]: load k's
// clock64 cycles; walk_cycles[b]: the walk's
bool l2keep_walk(const uint32_t * keep, int32_t n_walks, int32_t n_steps, uint32_t * cycles, uint32_t * walk_cycles,
                 void * stream);

} // namespace fn::dense
