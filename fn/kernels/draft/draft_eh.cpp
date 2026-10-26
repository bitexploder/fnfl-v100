#include "fn/kernels/dense/dense.h"
#include "fn/kernels/draft/draft.h"

#include <cstdint>

namespace fn::draft {

namespace {

// the calls of positions 0 .. positions - 1, each with the same work layout
bool eh_calls(int32_t positions, const fn::dense::product_args & base, fn::dense::product_args (&call)[max_rows]) {
    if (positions < 1 || positions > max_rows || base.k != in_width || base.n_mats != 1 || base.hmix != nullptr ||
        base.x_codes == nullptr || base.x_scales == nullptr || base.mats[0].out == nullptr) {
        return false;
    }
    fn::dense::work_layout first = {};
    for (int32_t b = 0; b < positions; ++b) {
        call[b] = base;
        call[b].x_codes = base.x_codes + int64_t(b) * n_stream * in_width;
        call[b].x_scales = base.x_scales + int64_t(b) * n_stream * in_runs * 2;
        call[b].mats[0].out = base.mats[0].out + int64_t(b) * n_stream * base.mats[0].m;
        fn::dense::work_layout l = {};
        if (!fn::dense::layout_of(call[b], l) || (b > 0 && !fn::dense::same_layout(first, l))) {
            return false;
        }
        if (b == 0) {
            first = l;
        }
    }
    return true;
}

} // namespace

bool eh_proj(int32_t positions, const fn::dense::product_args & base, void * stream) {
    fn::dense::product_args call[max_rows];
    if (!eh_calls(positions, base, call)) {
        return false;
    }
    for (int32_t b = 0; b < positions; ++b) {
        if (!fn::dense::product(n_stream, call[b], stream)) {
            return false;
        }
    }
    return true;
}

bool eh_proj_gated(const fn::dense::product_args & base, const int32_t * positions, void * stream) {
    fn::dense::product_args call[max_rows];
    if (positions == nullptr || !eh_calls(max_rows, base, call)) {
        return false;
    }
    for (int32_t b = 0; b < max_rows; ++b) {
        if (!fn::dense::product_gated(n_stream, call[b], positions, b, stream)) {
            return false;
        }
    }
    return true;
}

} // namespace fn::draft
