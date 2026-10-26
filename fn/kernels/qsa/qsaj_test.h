#pragma once

// qsaj-test (FMS2 build item 3, W3): the joint verify round's merged QSA launches against each lane's own launches,
// bitwise. Lanes of max_rows rows, each its own synthetic inputs (no model), buffers and ring set (as the joint round's
// lane rings), the index owner on DEV0 and the peer on DEV1: summary_update_lanes, select_lanes (score at grid_lanes'
// D, then at D forced to 4, 8 and 16), send_select_lanes on the owner, recv_select_lanes on the peer and attend_lanes
// there, at U = 2..4, at about 2K and 64K cells (one lane a different capacity), BF16 and KV8 caches, and a lane at a
// refused depth. After the sequence each lane's snapshot (key cache, summaries, select workspace, body and offsets
// on both cards, X-SELECT slot, attend's outputs, caches and workspace, both error words) is held bitwise its lone
// run's, and a second joint run bitwise the first. Out-of-contract calls are refused with nothing enqueued.
// DEV0 == DEV1: loopback on one stream (Compute Sanitizer). Prints "qsaj-test: PASS" and returns 0 when every case
// passes.
int fnfl_qsaj_test(int dev0, int dev1);
