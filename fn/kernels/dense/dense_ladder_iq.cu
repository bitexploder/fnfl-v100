// APF2: the dense ladder's instances for the IQ3_XXS deployment's other dense formats (Q4_K, Q5_K, Q6_K, IQ4_XS and
// IQ3_S in DP2's layout; IQ4_NL and Q2_0 in EX1's down layout), through the device decoder's quarter (stage) walk
// (dense_ladder.cuh pl_chunk_dec). The IQ3 file has no table, so serving runs its retained routes; these entries are
// the registry's candidates for those formats and are checked by pd-test

#include "fn/kernels/dense/dense_ladder_inst.cuh"

namespace fn::dense::pk {

template <typename Fn> bool pl_iq_switch(pl_src src, Fn && fn) {
    switch (src) {
        case pl_src::q4_k:
            return fn(std::integral_constant<pl_src, pl_src::q4_k>{});
        case pl_src::q5_k:
            return fn(std::integral_constant<pl_src, pl_src::q5_k>{});
        case pl_src::q6_k:
            return fn(std::integral_constant<pl_src, pl_src::q6_k>{});
        case pl_src::iq4_xs:
            return fn(std::integral_constant<pl_src, pl_src::iq4_xs>{});
        case pl_src::iq3_s:
            return fn(std::integral_constant<pl_src, pl_src::iq3_s>{});
        case pl_src::iq4_nl_ex1:
            return fn(std::integral_constant<pl_src, pl_src::iq4_nl_ex1>{});
        case pl_src::q2_0_ex1:
            return fn(std::integral_constant<pl_src, pl_src::q2_0_ex1>{});
        default:
            return false;
    }
}

bool pl_launch_iq(const pl_kargs & k, pl_src src, int32_t t, cudaStream_t s) {
    return pl_iq_switch(src, [&](auto S) { return pl_launch_src<decltype(S)::value>(k, t, s); });
}

bool pl_attr_iq() {
    bool ok = true;
    for (const pl_src s : { pl_src::q4_k, pl_src::q5_k, pl_src::q6_k, pl_src::iq4_xs, pl_src::iq3_s,
                            pl_src::iq4_nl_ex1, pl_src::q2_0_ex1 }) {
        ok = ok && pl_iq_switch(s, [](auto S) { return pl_attr_src<decltype(S)::value>(); });
    }
    return ok;
}

bool pl_res_iq(pl_src src, int32_t t, pd_resources & r) {
    return pl_iq_switch(src, [&](auto S) { return pl_res_src<decltype(S)::value>(t, r); });
}

}  // namespace fn::dense::pk
