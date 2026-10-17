// EX2: the gate/up launches of the IQ3 routed formats (experts_gu.cuh). XB3 C1: one translation unit for the two, so
// the IQ3 body (gate_up_kernel_u<gu_iq3>, map 3) is one symbol a B for every IQ3 layer
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_iq3_xxs(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::iq3_xxs>(fg, fu, rows, a, st);
}

bool gate_up_iq3_s(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::iq3_s>(fg, fu, rows, a, st);
}

} // namespace fn::experts::detail
