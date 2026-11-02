#pragma once

// APF2: the kernel-size ladder's grouped experts (the resident Q4 inventory: Q4_K / Q5_K gate/up, Q5_1 / Q8_0 down).
// A task is one expert's run of at most BM sorted pairs (ep_group's list for tile t, ep_lists); a CTA a (task, column
// tile) on the ladder's tensor-core CTA (fn/kernels/tc/tc_ladder.cuh), tiles 8, 16, 32, 64 and 128 routed rows by 128
// columns (TurboMind's grouped list, 1Cat-vLLM 1d7321bf2d, kernels/gemm/kernel/sm70_884_4.cu:160-240). The grid is
// the actual task count: a one-thread launcher reads ep_group's count on the device and tail-launches the grid
// (cudaStreamTailLaunch: it runs when the launcher ends, and the stream's next work waits for it), so no CTA exits
// for want of a task. The activation tile is staged once a CTA, into shared memory, for its warps.
//   gate/up: columns 128 c .. of an expert's 640 (gate and up interleaved: column 8 g + i is channel 64 c + 4 g + i % 4,
//            gate for i < 4, up otherwise), K 2,560, 5 column tiles; GLU in the epilogue (each lane holds a channel's
//            gate and up), stored F16 [pairs][320] in sorted order (and F32 with glu_f32), as ep_gate_up.
//   down:    columns 128 c .. of 2,560, K 320 (the GLU rows; IQ4_NL's K order inside each step, both operands alike),
//            20 column tiles; partials F32 or FP16 [pairs][2560] in sorted order, as ep_down.
//   weights: EX1's packed layouts, decoded in the tile loader to the FP16 values ep_gate_up / ep_down unpack (the same
//            kq_chunk / dn_chunk), one column a thread a K step.
//   numerics: FP16 operands, F32 accumulators (HMMA.884.F32), K in order: a rerun is bitwise; against the per-task
//            kernels, the same FP16 operands summed in another order (ep-test's FP64 bound holds both).
//   budget:  128 threads, 2 x (BM + 128) x 64 bytes of shared memory; registers in el_resources.

#include "fn/kernels/experts/experts_prompt.h"
#include "fn/kernels/tc/ladder.h"

#include <cstdint>

namespace fn::experts {

constexpr int32_t el_n_tiles = 5;  // fn::tc::lad_tiles 0 .. 4

// the tile's routed rows (8 .. 128), and whether the route takes the format
int32_t el_rows(int32_t tile);
bool    el_takes_gate_up(fmt f);
bool    el_takes_down(fmt f);

// gate/up of a layer's rows on tile t: tasks the list ep_group wrote with rows el_rows(t), count its counter
// (route.counters + 2 + list). a as ep_gate_up's, resident weights only (wt and task_off null)
bool ep_gate_up_ladder(fmt                     f,
                       int32_t                 t,
                       int32_t                 rows,
                       const ep_gate_up_args & a,
                       const int32_t *         tasks,
                       const int32_t *         count,
                       void *                  stream);
// down, as ep_down's, on tile t over that list
bool ep_down_ladder(fmt                  f,
                    int32_t              t,
                    int32_t              rows,
                    const ep_down_args & a,
                    const int32_t *      tasks,
                    const int32_t *      count,
                    void *               stream);
// product 0 gate/up, 1 down, format f, tile t: the kernel's budget (ctas_per_sm for the cost model)
bool el_resources(int32_t product, fmt f, int32_t t, fn::tc::lad_resources & r);

}  // namespace fn::experts
