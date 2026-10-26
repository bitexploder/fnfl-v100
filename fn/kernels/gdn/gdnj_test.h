#pragma once

// gdnj-test (FMS2 build item 3, W3): core_lanes, the joint verify round's one GDN core launch, against each lane's own
// core() bitwise. Lanes of max_rows rows on DEV, each its own synthetic inputs (no model), banks, record sets,
// selector (a different bank, set and replay count a lane), outputs and workspace; the layer's dt, a and norm weights
// shared. At U = 2..4 each lane's banks, records, gated F32, codes, scales and workspace counters are held bitwise its
// lone run's, and a second joint run at U = 4 bitwise the first. Out-of-contract calls are refused with nothing
// enqueued. Prints "gdnj-test: PASS" and returns 0 when every case passes.
int fnfl_gdnj_test(int dev);
