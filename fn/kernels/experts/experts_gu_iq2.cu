// EX2: the gate/up launches of the IQ2 routed formats (experts_gu.cuh). XB3 part 6: one translation unit for the three,
// so the IQ2 body (gate_up_kernel's U form) is one symbol a B for every IQ2 layer
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_iq2_xxs(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::iq2_xxs>(fg, fu, rows, a, st);
}

bool gate_up_iq2_xs(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::iq2_xs>(fg, fu, rows, a, st);
}

bool gate_up_iq2_s(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::iq2_s>(fg, fu, rows, a, st);
}

// ME10: both shards of one cache pass in one launch (the IQ2 body's pair family), B = 4 and 5
bool gate_up_pair_iq2(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_pair_args & a,
                      cudaStream_t st) {
    return gate_up_pair_u<gu_iq2>(f, fg, fu, rows, a, st);
}

} // namespace fn::experts::detail
