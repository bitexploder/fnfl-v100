#pragma once

// ME3 (the cleanup): the expert cache's units as the facade holds them (expert_cache.cpp, cache_report.cpp). Each unit
// is the one owner of its state; the devices (and their copy streams) are fixed at init and read by all. Declared in
// this order, so a unit outlives every unit that uses it.

#include "fn/runtime/cache_loan.h"
#include "fn/runtime/cache_residency.h"
#include "fn/runtime/cache_service.h"
#include "fn/runtime/cache_util.h"
#include "fn/runtime/expert_cache.h"
#include "fn/runtime/prompt_share.h"

namespace fn::runtime {

struct expert_cache::parts {
    cache::devices dv;
    residency res;
    prompt_loan loan;
    window_service win;
    prompt_share share; // MX7: its thread runs win's pool
};

} // namespace fn::runtime
