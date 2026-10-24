#pragma once

// ME1: the device steps of the host-backed KV tier (fn/model/kv_tier.h runs them; the 16 GB configuration). The whole K
// and V history of every attention layer stays in mapped pinned host memory; the device holds a recent window of each
// layer and one gather area a card. An attention launch (attend.h; draft.h's DA-TC in its sparse form) reads a cell c
// at k_cache + c * 256 for the cell IDs its selection body names and writes its new cells at depth + t, so the tier
// hands it a k_cache whose cells lo .. lo + W - 1 are the window's, and a copy of the body whose block IDs below lo
// are remapped to the gathered copies of those blocks. The launches, their order of cells and their arithmetic are
// unchanged: a row reads the same K and V values in the same order as from a resident cache (kv-tier-test checks the
// outputs bitwise).
//
// Contract:
//   remap:  one CTA of 1,024 threads. In: the B rows' selection bodies (qsa.h's layout; DA-TC's sparse body is the
//           same form) and their byte offsets [B + 1], lo (a multiple of 4) and base. Out: the body with every
//           whole-block ID b < lo / 4 replaced by base + s, where s is b's slot among the distinct such blocks of
//           all B rows in ascending block order (a union for the transport only: each row keeps its own blocks, order
//           and tail), list[s] = b, count = the slots, table[b] = s; every other word copied. A tail cell below lo
//           or more than max_gather slots set err_bit in err; every cell below lo is then pointed at cell lo (block
//           lo / 4) and count is 0, so the attention reads only window cells in a round that fails on its error word.
//   gather: the count blocks of list copied from the host history into the gather area, K and V, slot s at cells
//           4 s .. 4 s + 3 (the block's cells in order). 16-byte integer copies; bitwise.
//   shapes: B in 1 .. 5; lo in 0 .. fn::qsa::max_cells; at most max_gather slots (5 rows x 512 blocks); cells of 256
//           16-bit values (BF16 for the target, F16 for the draft; the copy does not look at them).
//   budget: remap reads B x 2,068 bytes and writes as much, a 64 Kbit shared bitmap of the blocks below lo (8 KB), one
//           CTA; gather moves 4 KB a slot from host memory over PCIe (zero-copy loads, many CTAs to keep enough reads
//           in flight).

#include "fn/kernels/qsa/qsa.h"

#include <cstdint>

namespace fn::qsa::tier {

constexpr int32_t cell_values = 256;                                // a cell's 16-bit values (one KV head)
constexpr int32_t max_gather = max_rows * top_blocks;              // 2,560 blocks
constexpr int32_t max_body_words = max_rows * max_row_words;       // 2,585
constexpr int32_t gather_cells = max_gather * ratio;               // 10,240

struct remap_args {
    const uint32_t * body;   // the rows' selection body (device)
    const int32_t * offsets; // [rows + 1] its byte offsets (device)
    int32_t rows;
    int32_t lo;              // cells below lo are not in the window (lo % 4 == 0)
    int32_t base;            // the remapped ID of slot 0
    uint32_t * out_body;     // [max_body_words]
    int32_t * table;         // [max_blocks] block -> slot of the blocks below lo / 4 (scratch)
    int32_t * list;          // [max_gather] slot -> block
    int32_t * count;         // [1]
    unsigned long long * total; // optional [1]: += count (the blocks gathered, for the counters)
    uint32_t * err;          // the card's error word
    uint32_t err_bit;
    // ME3 (the graphs): optional device word; when set, lo is *lo_d (read on the device) and the remapped ID of slot 0
    // is base + *lo_d / 4 (base then the window-independent part). At lo 0 the body is copied unchanged, count 0
    const int64_t * lo_d = nullptr;
};

struct gather_args {
    const int32_t * list;
    const int32_t * count;
    const uint16_t * host_k; // [cells][256] device-visible pinned host memory
    const uint16_t * host_v;
    uint16_t * area_k;       // [gather_cells][256] device memory
    uint16_t * area_v;
};

// Enqueue on stream; false (nothing enqueued) for arguments outside the contract or a failed launch
bool remap(const remap_args & a, void * stream);
bool gather(const gather_args & a, void * stream);

} // namespace fn::qsa::tier
