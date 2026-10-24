// ME5 rider: the miss split's estimator check (fn/runtime/split_test.h) without a GPU. Build and run:
//   g++-14 -std=c++17 -O2 -Wall -Wextra -Werror -I<engine> fn/tools/me5_split_check.cpp fn/runtime/split_test.cpp
//       fn/runtime/expert_policy.cpp -o me5-split-check
//   ./me5-split-check

#include "fn/runtime/split_test.h"

#include <cstdio>
#include <string>

int main() {
    std::string report;
    const bool ok = fn::runtime::split_estimator_check(report);
    std::fputs(report.c_str(), stdout);
    std::printf("me5-split-check: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
