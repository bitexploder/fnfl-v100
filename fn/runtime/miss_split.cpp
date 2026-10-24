#include "fn/runtime/miss_split.h"

#include <algorithm>

namespace fn::runtime::split {

int32_t choose_gpu(const int32_t * misses, int32_t nm, const int32_t * mask, int32_t n_gpu, uint8_t * to_gpu) {
    nm = std::clamp(nm, 0, max_misses);
    n_gpu = std::clamp(n_gpu, 0, nm);
    // the misses' positions ranked: more rows first, then later in route order first (the existing rule's order)
    int32_t rank[max_misses];
    for (int32_t i = 0; i < nm; ++i) {
        rank[i] = i;
        to_gpu[i] = 0;
    }
    std::sort(rank, rank + nm, [&](int32_t a, int32_t b) {
        const int32_t ra = rows_of(mask[misses[a]]), rb = rows_of(mask[misses[b]]);
        return ra != rb ? ra > rb : a > b;
    });
    for (int32_t k = 0; k < n_gpu; ++k) {
        to_gpu[rank[k]] = 1;
    }
    return n_gpu;
}

} // namespace fn::runtime::split
