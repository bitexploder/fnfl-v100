// ME1: the gate/up launches of routed format Q5_K (experts_gu.cuh), the Unsloth Q4 candidate's, one file a routed
// format so the instantiations compile in parallel
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_q5_k(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    // TTTS3 item 3: map 3 (the default) runs the family body (experts_gu_body_q5_k.cu); maps 0-2 ME1's gate_up_kernel
    return a.map == 3 ? gate_up_body_q5_k(fg, fu, rows, a, st) : gate_up_for<fmt::q5_k>(fg, fu, rows, a, st);
}

} // namespace fn::experts::detail
