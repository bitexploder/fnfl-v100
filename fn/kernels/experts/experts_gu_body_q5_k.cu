// TTTS3 item 3: routed format Q5_K's family body (gu_q5_k, experts_gu.cuh): its lone launch under map 3 at B = 1-5 and
// its lanes launch (gate_up_lanes) at B = 2-5. Its own translation unit, so experts_gu_q5_k.cu (gate_up_kernel, map 0)
// compiles as before
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_body_q5_k(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_body<gu_q5_k>(fmt::q5_k, fg, fu, rows, a, st);
}

bool gate_up_lanes_q5_k(fmt                        f,
                        fn::dense::fmt             fg,
                        fn::dense::fmt             fu,
                        int32_t                    lanes,
                        int32_t                    rows,
                        const gate_up_lanes_args & a,
                        cudaStream_t               st) {
    return f == fmt::q5_k && gate_up_lanes_u<gu_q5_k>(f, fg, fu, lanes, rows, a, st);
}

}  // namespace fn::experts::detail
