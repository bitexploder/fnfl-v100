#pragma once

// ple-pin-test (PN1): the PLE table's residency setting on a temporary file, no model load.
//   probe: a 64 MiB file mapped PROT_READ, MAP_SHARED: cudaDevAttrHostRegisterReadOnlySupported on both cards; the
//          mapping populated (MADV_POPULATE_READ) and registered with cudaHostRegister, without and then with
//          cudaHostRegisterReadOnly; page residency (mincore), VmLck and VmPin (/proc/self/status) and the kernel's
//          FOLL_PIN counters (/proc/vmstat) before and after; then the same counters around a 64 MiB cudaHostAlloc.
//          Prints what it sees; exits 0 when every call it makes returns, whatever the driver answers.
//   unit:  ple_host's residency on a file larger than RLIMIT_MEMLOCK's 8 MiB: full makes every page resident, keeps
//          the bytes, and ple_prefetch reports 0 ranges; an invalid range fails with the named error; lazy pins
//          nothing and still hints. Prints "ple-pin-test: PASS" and returns 0 when every check passes.
// dir: where the temporary file goes (removed before return); device: the card whose context registers.
int fnfl_ple_pin_test(const char * mode, const char * dir, int device);
