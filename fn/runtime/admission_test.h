#pragma once

// ME9: the admission records' host checks (fn/runtime/admission.h): no model, no device. ex-cpu-test runs them after
// ME5's; each prints "ex-cpu-test: me9 ...: PASS" or FAIL. The state machine and its choices only, never latency.

namespace fn::runtime {

// true when every check passes
bool admission_test();

} // namespace fn::runtime
