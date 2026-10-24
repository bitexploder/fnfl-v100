#pragma once

// MU1: the multi-slot engine's host checks: the KV pool's page bookkeeping (kv_pages: rounding, a failed reservation
// leaving nothing taken, release order), its reclaim choice (kv_reclaim_victim), the shared checkpoint book's owners
// and the read plan's partition. No model, no device. ex-cpu-test runs them; each prints "ex-cpu-test: mu1 ...: PASS"
// or FAIL. Correctness boundaries only, never performance evidence.

namespace fn::runtime {

// true when every check passes
bool slots_cpu_test();

} // namespace fn::runtime
