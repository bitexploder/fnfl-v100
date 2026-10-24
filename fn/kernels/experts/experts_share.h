#pragma once

// MX7: the prompt CPU share's device side (Strata-V100 69713b61's set_cpu_pool, src/prefill/prefill.cpp 3333-3720 and
// 4395-4410: the activation gather, and the CPU's rows added before the combine; rewritten as FNFL code, nothing
// borrowed). On a route-first ring layer the CPU takes some routed nonresident experts whole (fn/runtime/
// prompt_share.h); the GPU then neither stages nor computes them:
//   es_gather: the token rows the CPU's experts read, gathered from the layer's F16 input into a compact buffer (the
//              share's own stream downloads it);
//   es_drop:   ep_group's tasks of the CPU's experts removed in place (counters[0] lowered), the rest in their order,
//              so ep_task_offsets and the ring groups' launches see only the GPU's experts;
//   es_add:    the CPU's whole unweighted products (F16, 2,560 a pair) written into a logical card's partials at
//              their pairs' sorted positions (pos), zero for card 1, so each card's ep_sum weights them in route order
//              with the GPU's: card 0's partial carries the whole product, card 1's nothing.
// Contract: D = 2560, 512 experts, 10 slots a row; rows R in [1, 4096]; at most ep_max_tasks tasks. A null or
// misaligned pointer, or a count outside its range, is rejected (false, nothing enqueued). The kernels move bits only
// (no arithmetic on values): es_gather copies rows, es_add copies the CPU's F16 rows or writes zeros; bad != 0 sets
// err_bit in *err (the host found a CPU output beyond 65,504 or NaN, ep_down's rule for a partial). es_drop keeps the
// surviving tasks' order (sorted by expert, as ep_task_offsets needs); after a bad ID (counters[1] != 0) it changes
// nothing.

#include "fn/kernels/experts/experts_prompt.h"

#include <cstdint>

namespace fn::experts {

// out [n][2560] F16 = x [rows[i]][2560] F16, i < n (rows: device I32 [n], each in [0, R)); 16-byte aligned rows
bool es_gather(const uint16_t * x, const int32_t * rows, int32_t n, uint16_t * out, void * stream);

// skip: device U8 [512], nonzero for an expert whose tasks go
bool es_drop(const ep_route & r, const uint8_t * skip, void * stream);

// pairs: device I32 [n] (pair p = 10 row + slot); rows16: device F16 [n][2560], or null for zeros; part16 [R x 10]
// [2560] at pos[p]. n in [1, ep_max_pairs]
bool es_add(const int32_t * pairs, const uint16_t * rows16, int32_t n, const int32_t * pos, uint16_t * part16,
            int32_t bad, uint32_t * err, uint32_t err_bit, void * stream);

} // namespace fn::experts
