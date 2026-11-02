#pragma once

// APF2: the kernel-size ladder's selector host checks (fn/model/ladder.h), no model and no device: the cost model
// against hand-computed tile, wave and split counts and costs for a few shapes, its filter and its wave and workspace
// rejections, and the table lookup (the largest rung at or below rows; the retained route without an entry). Each check
// prints "ladder-test: ...: PASS" or FAIL. Correctness of the selection only, never performance evidence.

namespace fn::model {

// 0 when every check passes
int fnfl_ladder_test();

}  // namespace fn::model
