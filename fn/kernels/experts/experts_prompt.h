#pragma once

// Routed-expert kernels for prompt rows on one card (EP1): card r owns intermediate channels 320 r .. 320 r + 319 of
// every expert (the decode layout), so gate and up are [2560 -> 320] and down is [320 -> 2560] per expert per card.
// Simulator node: the prompt expert node of fnfl/sim (S19's design budget); fnfl/design/history/fnfl-prefill-design.md
// section 6. Five launches a layer:
//   ep_group:   counting sort of the R x 10 (row, slot) pairs by expert, stable by row then slot; the tasks.
//   ep_gate_up: per task (an expert's next <= 40 sorted pairs), each weight unpacked to FP16 once and multiplied against
//               the task's rows on the tensor cores (mma.sync m8n8k4), gate and up together, SiLU(gate) x up in the
//               epilogue: F16 GLU per pair, 320 channels. Each lane holds one weight column's B fragments and unpacks
//               that column itself, so the unpacked weights stay in registers (a shared-memory tile would only be
//               read back by the lane that wrote it); the activations are staged in shared memory.
//   ep_down:    per task, [320 -> 2560] against the task's GLU rows in five column groups of 512 (PB3: one launch,
//               the GLU rows staged once): F32 partials.
//   ep_sum:     per row and output column, the 10 slots' partials times the router weights in route order, plus the
//               shared expert's partial last (zero when null; the shared expert is EX2).
//
// Contract:
//   shapes:  D = 2560, 512 experts, 10 slots a row, 320 channels a card; rows R in [1, 4096]. Tasks of at most 40 pairs
//            (ep_task_rows; ep_down takes up to 48): one expert's sorted pairs, split evenly; at most 40,960 / 40 + 512 =
//            1,536. A null or misaligned pointer, R outside [1, 4096] or a format outside the lists below is
//            rejected (false, nothing enqueued). An expert ID outside [0, 512) sets status[0] = 1 (when
//            status is not null); then no task runs and ep_sum writes nothing.
//   formats: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; down Q2_0, IQ4_NL, in EX1's packed layout (experts.h:
//            pack_gate_up, pack_down): one copy of the weights serves decode and prompt rows.
//   math:    weights FP16 (FNFL's choice for Volta's tensor cores, which take FP16 only): code x scale rounded once
//            to FP16, the scale itself rounded to FP16 first; activations FP16 as the producer wrote them (EP-5: no
//            in-kernel conversion); FP32 accumulation (HMMA.884.F32); GLU
//            g / (1 + expf(-g)) x u in F32 (precise expf), stored F16 (round to nearest); down reads the F16 GLU as is;
//            partials and the sum F32 (fmaf in route order; PB3's prompt path stores the partials FP16, each rounded
//            once, and sums them in F32). No fast math (--ftz=false --prec-div=true
//            --prec-sqrt=true --fmad=true). The K order inside a 32-weight step is fixed per format (IQ4_NL and Q2_0
//            permute it, both operands alike), reductions have a fixed order and there are no floating-point atomics,
//            so two runs are bitwise identical. Reference (R16): FP64 on the dequantized weights and the same FP16
//            activation values (experts_prompt_test.cpp); bound per output tensor: max |err| / max |ref| at most 1e-3
//            (measured, EP1 with 8-bit input: GLU 2.8-5.2e-4; partials and sums 0.4e-6-3.1e-4; C2's Result has the
//            figures with FP16 input).
//   floor:   a 1,536-row half per card per layer (gate/up IQ3_XXS, down IQ4_NL, 72 SMs, 1,380 MHz): weights 557 MB,
//            547 us at 1,021.6 GB/s; tensor cores 75.5 GFLOP, 742 us at 101.7 TFLOPS. Budget (S19): 1,484 us.
//   unpack:  counted from SASS (EP1 step 2; per 32-weight sub-block of one lane, loads included): IQ3_XXS 133 (0.34 warp
//            instructions a weight byte), IQ2_XXS 117, IQ2_XS 125, IQ2_S 126, IQ3_S 159; down IQ4_NL 116, Q2_0 73.
//   budget:  a block a task (grid ep_max_tasks, blocks past the task count exit), one block an SM. ep_gate_up: 640 threads,
//            __launch_bounds__(640, 1), 96 registers, no spills; dynamic shared memory: two activation tiles [48][32]
//            FP16 (6 KB), each column's load offsets (10 KB) and the grid table as bank-private copies (32-64 KB).
//            ep_down: 512 threads, 110-120 registers, every step's GLU tile 30 KB. ep_group: one block of 1024 threads, 37 KB static.
//   measured: EP1 Result (card 0, 1,380 MHz, FN4 routes as prompt rows, cold weights): IQ3_XXS/IQ4_NL at R = 1536, a
//            layer 3.01 ms (gate_up 1.64, down x 5 1.08, sum x 5 0.25, group 0.05); the 1.48 ms budget is missed.
//            C2 (FP16 input tiles): its Result has the bench point before and after.
//   evidence: HX8 (borrowed costs), S19 (budget), EX1 and EX1b (the packed layout, the decode recipes), EP1.
//
// Buffers (device): ids I32 [R][10]; route: offsets I32 [513], sorted I32 [R x 10] (pair p = 10 row + slot, by
// expert), pos I32 [R x 10] (the sorted position of pair p), tasks I32 [ep_max_tasks][4] (expert, first position, pairs, 0),
// counters I32 [8] (counters[0] the task count, counters[1] 1 after a bad ID). x: F16 [R][2560] (16-byte aligned), the
// producer's FP16 tile. glu: F16 [R x 10][320] in sorted order. part: F32 [R x 10][512] in sorted order (31.5 MB at
// R = 1536). wts F32 [R][10]; out and shared F32 [R][2560].

#include "fn/kernels/experts/experts.h"

#include <cstddef>
#include <cstdint>

namespace fn::experts {

constexpr int32_t ep_max_rows = 4096; // PB3's half (was 1,536)
constexpr int32_t ep_max_pairs = ep_max_rows * 10;
constexpr int32_t ep_pass_rows = 48; // the largest task
constexpr int32_t ep_min_task_rows = 40;
constexpr int32_t ep_max_tasks = ep_max_pairs / ep_min_task_rows + 512; // 1,536

// the task size ep_group must use for a layer whose gate/up format is f: 40 pairs (48 rows of accumulators spill at 96
// registers once the K loop is specialized on the task's row tiles, EP1 change 1); 0 for a format ep_gate_up does not
// take
int32_t ep_task_rows(fmt gate_up);
constexpr int32_t ep_col_group = 512;
constexpr int32_t ep_n_col_groups = 5;

// The prompt buffers of one card for R rows, each 256-byte aligned (R1's reservation): ids I32 [R][10]; the route
// (offsets [513], sorted and pos [R x 10], tasks [ep_max_tasks][4], counters [8]); x F16 [R][2560]; glu F16
// [R x 10][320]; part F32 [R x 10][2560]; wts F32 [R][10]; out and shared F32 [R][2560]. 0 for R outside 1..ep_max_rows.
size_t ep_workspace_bytes(int32_t rows);

struct ep_route {
    int32_t * offsets;
    int32_t * sorted;
    int32_t * pos;
    int32_t * tasks;
    int32_t * counters;
};

// task_rows: ep_task_rows of the layer's gate/up format (any multiple of 8 in [8, 48] for ep_down alone)
bool ep_group(int32_t rows, int32_t task_rows, const int32_t * ids, const ep_route & r, uint32_t * status,
              void * stream);

struct ep_gate_up_args {
    const uint8_t * w;     // [512][gate_up_bytes(f)] packed
    const uint8_t * grids; // grid_bytes
    const uint16_t * x;    // F16 [R][2560], 16-byte aligned
    ep_route route;        // as ep_group wrote it
    uint16_t * glu;        // F16 [R x 10][320] out
    float * glu_f32;       // F32 [R x 10][320] out, test only; nullptr in production
};

struct ep_down_args {
    const uint8_t * w;     // [512][down_bytes(f)] packed
    const uint8_t * grids; // grid_bytes
    const uint16_t * glu;  // F16 [R x 10][320] from ep_gate_up
    ep_route route;
    float * part; // F32 [R x 10][2560] out
    // PB3: with part16, the partials written as FP16 [R x 10][2560] there instead of part (which may be null), each
    // rounded once to nearest from its F32 accumulator; a value beyond 65,504 or NaN sets err_bit in *err
    uint16_t * part16 = nullptr;
    uint32_t * err = nullptr;
    uint32_t err_bit = 0;
};

struct ep_sum_args {
    const float * part; // F32 [R x 10][2560] from ep_down
    ep_route route;
    const float * wts;    // F32 [R][10]
    const float * shared; // F32 [R][2560] or nullptr
    float * out;          // F32 [R][2560]
    // PB3: with out16, the sum written as FP16 [R][2560] there instead of out (which may be null), rounded to nearest;
    // a value beyond 65,504 or NaN sets err_bit in *err (fn::xchg::ppack's arithmetic: a P message's payload)
    uint16_t * out16 = nullptr;
    uint32_t * err = nullptr;
    uint32_t err_bit = 0;
    // PB3: the partials as ep_down's part16 wrote them (part may then be null); the sum stays F32 (fmaf in route order)
    const uint16_t * part16 = nullptr;
};

// Enqueue on stream (a cudaStream_t). Return false, with nothing enqueued, for an unsupported format or argument;
// false after a failed launch submission (the CUDA error is left for the caller).
bool ep_gate_up(fmt f, int32_t rows, const ep_gate_up_args & a, void * stream);
bool ep_down(fmt f, int32_t rows, const ep_down_args & a, void * stream);
bool ep_sum(int32_t rows, const ep_sum_args & a, void * stream);

} // namespace fn::experts
