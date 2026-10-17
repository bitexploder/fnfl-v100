#pragma once

// XP1's timing helpers (xp1-bench and the pair's graphs). Measurement scaffolding, not part of a model pass.
//   round:  the first node of every card's graph and eager step: stamps the card's globaltimer (ns) for the start skew,
//           and takes this round's decode index (Amendment 3: replay r reads decode r mod n_decode) from a counter it
//           advances, so graph replays index the captured traffic with no host work. With routes (DP2's --dense, where
//           the router is DP2's on synthetic weights), it also copies the decode's captured IDs of rows 0 .. rows - 1
//           for layers 10 and 11 into ids[0] and ids[1] ([rows][10] each), which EX1b then reads.
//   flush:  writes a scratch buffer larger than L2 before an "alone" measurement, so weights start cold.
//   offset: card 1's globaltimer minus card 0's, from n round trips through peer memory (the midpoint of each trip),
//           so the two cards' round stamps share one clock.

#include <cstdint>

namespace fn::xp1 {

bool round_start(uint64_t * stamp, int32_t * counter, int32_t * decode, int32_t n_decode, void * stream);
// routes[l]: [n_decode][3][10] device memory (HX7's capture); rows 1..3
bool round_start_routes(uint64_t * stamp, int32_t * counter, int32_t * decode, int32_t n_decode,
                        const int32_t * const routes[2], int32_t * const ids[2], int32_t rows, void * stream);
bool stamp(uint64_t * at, void * stream);
bool flush_l2(void * scratch, int64_t bytes, void * stream);
// the SM clock now: one thread counts clock64 over 200 us of globaltimer; out[0] = cycles, out[1] = ns. Run between
// timed replays, never inside them.
bool clock_probe(uint64_t * out, void * stream);

// Peer access must be on both ways. flags0/flags1: 16 bytes of device memory on each card; out0 [n][2] and out1 [n]
// on each card. Returns false on a failed launch or a timed-out trip.
bool clock_offset(int32_t dev0, int32_t dev1, uint32_t * flags0, uint32_t * flags1, uint64_t * out0, uint64_t * out1,
                  int32_t n, double & offset_ns, double & rtt_ns);

} // namespace fn::xp1
