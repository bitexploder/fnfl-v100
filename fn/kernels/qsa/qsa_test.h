#pragma once

#include <cstdint>
#include <string>
#include <vector>

// The QSA kernels against the CPU reference (QA1 step 3): summaries, selection, bias and union bitwise, and a bitwise
// second run. depths and rows pick the synthetic cases (empty: all of QA1's);
// fixture is the HX7 real-selection case's file (empty: skipped, and said so). Prints "qa-test: PASS" and returns 0
// when every case passes.
int fnfl_qsa_test(int device, const std::vector<int32_t> & depths, const std::vector<int32_t> & rows,
                  const std::string & fixture);
