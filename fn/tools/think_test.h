#pragma once

// GLFL16 thinking budget's CPU checks (fn/runtime/think_budget.h): a scripted model generated token by token and
// speculatively under every budget, the closing sequence at exactly the budget, and speculation never changing the
// tokens. IFR1 R3: the UTF-8 rule on synthetic byte pieces, then each fixture an engine's chat check wrote with its
// model's tokenizer (fixtures: their paths). Returns the failures (0: PASS); prints a line a check
#include <string>
#include <vector>

namespace fn::tools {

int think_test(const std::vector<std::string> & fixtures);

} // namespace fn::tools
