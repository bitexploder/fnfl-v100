// TTTS3 item 3: routed format IQ4_XS's family body (gu_iq4_xs, experts_gu.cuh): its lone launch under map 3 at B = 1-5 and
// its lanes launch (gate_up_lanes) at B = 2-5. Its own translation unit, so experts_gu_iq4_xs.cu (gate_up_kernel, map 0)
// compiles as before
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_body_iq4_xs(fn::dense::fmt fg, fn::dense::fmt fu, int32_t rows, const gate_up_args & a, cudaStream_t st) {
    return gate_up_body<gu_iq4_xs>(fmt::iq4_xs, fg, fu, rows, a, st);
}

bool gate_up_lanes_iq4_xs(fmt                        f,
                        fn::dense::fmt             fg,
                        fn::dense::fmt             fu,
                        int32_t                    lanes,
                        int32_t                    rows,
                        const gate_up_lanes_args & a,
                        cudaStream_t               st) {
    return f == fmt::iq4_xs && gate_up_lanes_u<gu_iq4_xs>(f, fg, fu, lanes, rows, a, st);
}

}  // namespace fn::experts::detail
