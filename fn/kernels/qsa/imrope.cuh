#pragma once

// MUP1 piece 5: IMROPE's plane for a rotation pair, and a prompt row's rotary position. The sector rule is attend.cu's
// norm_rope (pair i = sector, sections 11, 11, 10, 0 of n_rot = 64): plane 1 for i % 3 == 1 below 33, plane 2 for
// i % 3 == 2 below 30, plane 0 for i % 3 == 0 below 33, else plane 3.
//
// A prompt row's position: with supplied planes (planes [rows][4], row t's plane p at planes[4 t + p], so a half's rows
// are a pointer offset into its chunk's; VIS1's embedding rows) the pair's plane; else every plane rope0 + t (text:
// rope0 = the rows' first cell plus the slot's offset), so text reads the same integer in every plane as before.

#include <cstdint>

namespace fn::qsa {

__device__ __forceinline__ int32_t imrope_plane(int32_t sector) {
    return sector % 3 == 1 && sector < 3 * 11   ? 1
           : sector % 3 == 2 && sector < 3 * 10 ? 2
           : sector % 3 == 0 && sector < 3 * 11 ? 0
                                                : 3;
}

__device__ __forceinline__ int32_t prompt_rope_pos(const int32_t * planes, int32_t rope0, int32_t t, int32_t sector) {
    return planes != nullptr ? __ldg(planes + 4 * t + imrope_plane(sector)) : rope0 + t;
}

}  // namespace fn::qsa
