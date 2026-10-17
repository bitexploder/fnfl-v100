#pragma once

// fnfl-drive xchg2-test (R2): the layer exchanges (X-A, X-F, X-ROUTE, X-SELECT) on two cards with synthetic payloads,
// no model. Each round and row count B in 1..4: both cards send a partial [B][2560] (some words NaN, one the sentinel
// pattern itself) and receive the sum of their half; the round's router owner sends IDs, weights and a gate; the round's
// index owner sends a union (cells, row bits, count) and both build the attention mask. Sends are issued before the
// receives that read them (the layer's phase order), so it also runs under Compute Sanitizer. dev0 == dev1: loopback,
// both cards on one device and one stream. Checks every word bitwise against the host's expectation (NaN as any NaN);
// prints "xchg2-test: PASS" and returns 0.

#include <cstdint>

int fnfl_xchg2_test(int32_t dev0, int32_t dev1, int32_t rounds);
