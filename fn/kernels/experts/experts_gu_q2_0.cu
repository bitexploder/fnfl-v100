// ME7: the gate/up launches of routed format Q2_0 (experts_gu.cuh), the Q2 MTP draft's (opt-in: the draft file's
// formats choose it), one file a routed format so the instantiations compile in parallel
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_q2_0(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_for<fmt::q2_0>(fg, fu, rows, a, st);
}

} // namespace fn::experts::detail
