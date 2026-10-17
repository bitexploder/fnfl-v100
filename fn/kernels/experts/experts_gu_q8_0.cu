// DR1: the gate/up launches of routed format Q8_0 (experts_gu.cuh), the MTP draft layer's, one file a routed format so
// the instantiations compile in parallel
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_q8_0(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::q8_0>(fg, fu, rows, a, st);
}

} // namespace fn::experts::detail
