#pragma once

// ME2: the CPU miss path (experts_cpu.h) on the CPU alone, from the synthetic experts (experts_test.h load_synth in
// dir; missing files are generated there):
//   round trip: pack_gate_up / pack_down then unpack_gate_up_tile / unpack_down_rows return the GGUF rows bitwise, for
//               the seven gate/up formats (both cards, every tile) and the four down formats (every row);
//   FP64:       whole experts through the pool (both cards' gate/up with SwiGLU, the down activation, both cards'
//               down) at B = 1..5 against FP64 on the same quantized operands (the GGUF values, the Q8_K activation;
//               down: the pool's own down activation), each output within the bound stated in experts_cpu_test.cpp;
//               max and p99 of |error| / max |reference| reported;
//   throughput: per deployed class (gate/up, down), 64 windows of 8 experts at B = 1 from a 512-expert layer backing,
//               the pool on CPUs 1..5 (a CPU-only diagnostic line, not a measurement).
// Prints "ex-cpu-test: FMT B=1..5 PASS" per format and "ex-cpu-test: PASS"; returns 0.
int fnfl_ex_cpu_test(const char * dir);
