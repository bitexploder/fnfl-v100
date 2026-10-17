#pragma once

// grid-test (C1, CPU only): FNFL's grid tables (grids.h) against the GGUF formats' definition, ggml/src/ggml-common.h
// at 66e11c3. header is that file; its SHA-256 must be the pinned revision's. Each table is parsed from it and compared
// byte for byte (little-endian), and grid_table() must pack them into grid_bytes. Prints "grid-test: PASS" and returns
// 0 when every table is identical.
int fnfl_grid_test(const char * header);
