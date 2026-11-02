#pragma once

#include <cstdint>

// grid-test (C1; host only, no device): FNFL's grid tables (grids.h) against the GGUF formats' definition,
// ggml/src/ggml-common.h at 66e11c3. header is that file; its SHA-256 must be the pinned revision's. Each table is
// parsed from it and compared byte for byte (little-endian), and grid_table() must pack them into grid_bytes. FMS4
// item 2: each IQ2 selector region of the packed buffer byte for byte against its derivation from the 8-byte table.
// Prints "grid-test: PASS" and returns 0 when every check passes. TFC: the device recipe is grid-gpu-test.
int fnfl_grid_test(const char * header);

// grid-gpu-test (FMS4 item 2; TFC: split out of grid-test, needs device dev): grid_table() packed on the host, then the
// decoders' selector recipe on the device (fnfl_grid_sel_device). Prints "grid-gpu-test: PASS" and returns 0 when
// every case is right; 1 otherwise.
int fnfl_grid_gpu_test(int dev);

// FMS4 item 2 (grids_dev_test.cu): the decoders' IQ2 selector recipe on device dev for every entry of the three
// selector tables in buf (grid_table()'s buffer, grid_bytes) and every sign pattern (128 for IQ2_XXS and IQ2_XS, 256
// for IQ2_S: 360,448 cases), against the 8-byte tables' signed bytes in buf. The wrong cases, or -1 on a CUDA error.
int64_t fnfl_grid_sel_device(const uint8_t * buf, int dev);
