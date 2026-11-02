#pragma once

// TTTS6 item 2: the staging steps of fn::dense's Q8_0 instances (dense.h product_args::steps). A header of its own, so
// the sources that include dense.h for its types only (the experts kernels) compile as before: adding these
// declarations to dense.h changed cicc's register numbering in an IQ3_S experts kernel (the same instructions, other
// registers and order).

#include "fn/kernels/dense/dense.h"

#include <cstdint>

namespace fn::dense {

// TTTS6 item 2: a launch's staging steps. Block b stages the K steps [lo[b], hi[b]) of every activation row: the
// hull of the steps its quantized units read (unit u runs on block u mod 72; none when hi[b] is 0). They depend only on
// the launch's shape (K, its matrices' tile sizes, tiles and splits), so a site computes them once (stage_steps_of) and
// its launches point at them; k and n_units name the shape, and a launch whose own differ is rejected.
struct stage_steps {
    uint8_t lo[72] = {}, hi[72] = {};
    int32_t k = 0, n_units = 0;
};

// a's staging steps (its splits set, as choose_splits leaves them); false for a launch layout_of rejects
bool stage_steps_of(const product_args & a, stage_steps & out);

}  // namespace fn::dense
