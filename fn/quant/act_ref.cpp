#include "fn/quant/act_ref.h"

#include <cmath>
#include <cstdint>

namespace fn::quant {

void act_quantize_ref(const float * x, int32_t rows, int32_t k, int8_t * codes, float * scales) {
    const int32_t n_runs = rows * (k / 32);
    for (int32_t r = 0; r < n_runs; ++r) {
        const float * v = x + int64_t(r) * 32;
        float amax = 0.0f;
        for (int32_t i = 0; i < 32; ++i) {
            amax = std::fmax(amax, std::fabs(v[i]));
        }
        const float d = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int32_t sum = 0;
        for (int32_t i = 0; i < 32; ++i) {
            const float e = v[i] * id;
            // nearbyintf in the default mode (FE_TONEAREST): ties to even, as __float2int_rn
            const int32_t q =
                std::isnan(e) ? 0 : static_cast<int32_t>(std::fmin(std::fmax(std::nearbyint(e), -127.0f), 127.0f));
            codes[int64_t(r) * 32 + i] = static_cast<int8_t>(q);
            sum += q;
        }
        scales[2 * int64_t(r)] = d;
        scales[2 * int64_t(r) + 1] = d * static_cast<float>(sum);
    }
}

void act_values_ref(const int8_t * codes, const float * scales, int32_t k, float * out) {
    for (int32_t i = 0; i < k; ++i) {
        out[i] = static_cast<float>(codes[i]) * scales[int64_t{2} * (i / 32)];
    }
}

} // namespace fn::quant
