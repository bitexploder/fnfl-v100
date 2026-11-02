// APF2: the dense ladder's instances for Q4's (UD-Q4_K_XL) dense inventory: Q8_0 in DP2's layout, Q8_0 in EX1's down
// layout (the shared expert's down) and BF16 rows (dense_ladder.h has the contract)

#include "fn/kernels/dense/dense_ladder_inst.cuh"

namespace fn::dense::pk {

bool pl_launch_q4(const pl_kargs & k, pl_src src, int32_t t, cudaStream_t s) {
    switch (src) {
        case pl_src::q8_0:
            return pl_launch_src<pl_src::q8_0>(k, t, s);
        case pl_src::q8_0_ex1:
            return pl_launch_src<pl_src::q8_0_ex1>(k, t, s);
        case pl_src::bf16:
            return pl_launch_src<pl_src::bf16>(k, t, s);
        default:
            return false;
    }
}

bool pl_attr_q4() {
    return pl_attr_src<pl_src::q8_0>() && pl_attr_src<pl_src::q8_0_ex1>() && pl_attr_src<pl_src::bf16>();
}

bool pl_res_q4(pl_src src, int32_t t, pd_resources & r) {
    switch (src) {
        case pl_src::q8_0:
            return pl_res_src<pl_src::q8_0>(t, r);
        case pl_src::q8_0_ex1:
            return pl_res_src<pl_src::q8_0_ex1>(t, r);
        case pl_src::bf16:
            return pl_res_src<pl_src::bf16>(t, r);
        default:
            return false;
    }
}

}  // namespace fn::dense::pk
