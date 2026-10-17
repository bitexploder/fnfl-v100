#pragma once

// XB1: test outputs written for a byte comparison across two binaries (work/XB1's src/xb1_bitwise.py), so a kernel
// change with no reference copy left in the binary can still be checked bitwise against the base revision's binary.
// When the environment sets FNFL_TEST_DUMP to a directory, test_dump writes n bytes to <dir>/<label>.<k>.bin, k
// counting this label's earlier dumps in the process; otherwise it does nothing. Test code only, never the engine.

#include <cstddef>
#include <string>

namespace fn::tools {

// false only when a dump was asked for and the write failed
bool test_dump(const std::string & label, const void * p, size_t n);

} // namespace fn::tools
