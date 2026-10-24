#pragma once

// ME9 (Shell 0a1b4eb5): a strict JSON well-formedness check (RFC 8259: one value, objects, arrays, strings with their
// escapes, numbers, true, false, null, nothing after it) for the machine-readable records the engine prints (the expert
// cache's request and counter lines). CPU only; it validates, it does not build a tree.

#include <cstddef>
#include <string>

namespace fn::runtime {

// true when s is exactly one JSON value; false with at the byte offset where parsing stopped
bool json_valid(const std::string & s, size_t & at);

} // namespace fn::runtime
