#pragma once

// ME5 rider (Fable, 2026-10-18): the miss split's estimator (fn/runtime/expert_policy.h pcie_auto) on recorded-style
// leg times, CPU only: refinement past three readings, the deadband on the unrounded reading, readings alternating
// either side of a rounding boundary, constant inputs, the reset (per arm and at boot), the deferrals, k and the share
// split. ex-test runs it; so does fn/tools/me5_split_check.cpp on a machine without a GPU.

#include <string>

namespace fn::runtime {

// true when every case holds; report gets one line a case
bool split_estimator_check(std::string & report);

} // namespace fn::runtime
