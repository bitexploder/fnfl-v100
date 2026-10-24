// ME1: the gate/up launches of routed format Q4_K (experts_gu.cuh), the Unsloth Q4 candidate's, one file a routed
// format so the instantiations compile in parallel
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_q4_k(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::q4_k>(fg, fu, rows, a, st);
}

} // namespace fn::experts::detail
