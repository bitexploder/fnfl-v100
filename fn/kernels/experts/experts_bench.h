#pragma once

// EX2: each deployed experts launch (the routed experts with the shared expert folded in) timed alone at B = 1 and 3,
// warm (a graph of 64 launches, weights streaming cold, code warm) and cold (dp-bench's (A - B) / 20 with a 12 MB L2
// flush and instruction-cache eviction before each launch), routing each instantiation's first deployed layer through
// HX7's captured decodes 3..22 (XP1's --attr set). Prints one JSON line a point (hardware speed: 1.949 us + bytes /
// 1,021.6 GB/s + 0.83 us, on the launch's own bytes) and returns 0 when every launch succeeded. mode "all", or one
// point's name for ncu (gu_<routed>_<gate>_<up>_B<rows>, dn_<routed>_<shared>_B<rows>).
int fnfl_ex_bench(int device, const char * mode, const char * cache_dir, const char * hx7_dir);
