#pragma once

// fnfl-drive samp-test [all|quick|san] [DEV0] [DEV1] [RECORDED] [OUT] (SM1): the sampler against the CPU reference on
// two cards (DEV0 == DEV1: loopback on one device and one stream, sends before reads, for Compute Sanitizer).
// RECORDED: TG1's decode-mode logits.bin.gz (whole rows), read in place; OUT: the per-case lines and listed cases.
int fnfl_samp_test(int dev0, int dev1, const char * mode, const char * recorded, const char * out_path);
