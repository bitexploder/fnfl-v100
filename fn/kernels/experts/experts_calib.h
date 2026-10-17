#pragma once

// EX1 step 2 (experts sheet C1-C3): the machine rates the routed-expert kernels are designed against, measured before
// the kernels. Prints one JSON line a point and returns 0 when every launch succeeded.
//   C1: random lookups from a shared-memory table of 2, 4 and 8 KB with 4-, 8- and 16-byte entries (HX12 M6's method:
//       the whole loop with and without the lookup);
//   C2: POPC alone and beside IADD3;
//   C3: IDP4A beside FFMA and beside IADD3.
// Rates are warp instructions per clock per SM, from clock64() inside each block (one block of 32 warps an SM).
int fnfl_ex_calib(int device);
