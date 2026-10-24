#pragma once

// ME4: the prompt planner's host checks (fn/runtime/prompt_plan.h, loan.h): no model, no device. ex-cpu-test runs them
// after ME3's arms; work/ME4/planner/check.cpp adds the evidence pins. Each check prints "ex-cpu-test: me4 ...: PASS"
// or FAIL. Correctness boundaries only, never performance evidence.

namespace fn::runtime {

// true when every check passes
bool prompt_plan_test();

} // namespace fn::runtime
