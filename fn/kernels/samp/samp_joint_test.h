#pragma once

// FMU4 (dpj-test's sampled lanes): the joint draft step's sampled pick (samp.h pick_lanes_stage) at U = 2, 3 and 4
// lanes on one pair of rings against each lane's lone pick (pick_stage) on its own logits: for each lane its own
// options (temperature, top-k, top-p, min-p, one lane with penalties and a window), its own live descriptor (Philox
// position n_before + j + 1, the proposals so far as penalty deltas) and d = 4 chained picks; every lane's four
// proposals and q_list rows bitwise its lone run's on both cards, eagerly (each stage on card 0 then card 1) and from
// captured graphs (a graph a card on two devices, one graph on a loopback device), and no error word. dev0 == dev1:
// loopback on one device and one stream. san: U = 2 and 4, one round each. Returns the number of failed checks.
int fnfl_dpj_pick_test(int dev0, int dev1, bool san);
