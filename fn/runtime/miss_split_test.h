#pragma once

// ME5: the row-aware miss split's host checks (fn/runtime/miss_split.h): no model, no device. ex-cpu-test runs them
// after ME4's planner checks; each prints "ex-cpu-test: me5 ...: PASS" or FAIL. Correctness and the counted CPU-row
// property only, never latency.

namespace fn::runtime {

// true when every check passes
bool miss_split_test();

} // namespace fn::runtime
