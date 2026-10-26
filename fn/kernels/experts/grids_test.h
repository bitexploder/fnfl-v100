#pragma once

#include <cstdint>

// grid-test (C1): FNFL's grid tables (grids.h) against the GGUF formats' definition, ggml/src/ggml-common.h at
// 66e11c3. header is that file; its SHA-256 must be the pinned revision's. Each table is parsed from it and compared
// byte for byte (little-endian), and grid_table() must pack them into grid_bytes. FMS4 item 2: each IQ2 selector
// region of the packed buffer byte for byte against its derivation from the 8-byte table, then the decoders' selector
// recipe on device 0 (fnfl_grid_sel_device). Prints "grid-test: PASS" and returns 0 when every check passes.
int fnfl_grid_test(const char * header);

// FMS4 item 2 (grids_dev_test.cu): the decoders' IQ2 selector recipe on device dev for every entry of the three
// selector tables in buf (grid_table()'s buffer, grid_bytes) and every sign pattern (128 for IQ2_XXS and IQ2_XS, 256
// for IQ2_S: 360,448 cases), against the 8-byte tables' signed bytes in buf. The wrong cases, or -1 on a CUDA error.
int64_t fnfl_grid_sel_device(const uint8_t * buf, int dev);
