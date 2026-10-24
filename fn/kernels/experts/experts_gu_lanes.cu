// FMU2 (build item 1): a joint round's lanes' gate/up in one launch (experts.h gate_up_lanes), the IQ2 and IQ3 bodies
// at B = 5 over 36, 24 or 18 blocks a lane. Its own translation unit, so the lone launches' units (experts_gu_iq2.cu,
// experts_gu_iq3.cu) compile as before
#include "fn/kernels/experts/experts_gu.cuh"

namespace fn::experts::detail {

bool gate_up_lanes_iq2(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t lanes, const gate_up_lanes_args & a,
                       cudaStream_t st) {
    return gate_up_lanes_u<gu_iq2>(f, fg, fu, lanes, a, st);
}

bool gate_up_lanes_iq3(fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t lanes, const gate_up_lanes_args & a,
                       cudaStream_t st) {
    return gate_up_lanes_u<gu_iq3>(f, fg, fu, lanes, a, st);
}

} // namespace fn::experts::detail
