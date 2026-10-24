#pragma once

// ME1: the expert cache's small device steps (fn/runtime/expert_cache.h): resolve, apply and (ME2b) combine.
//
// resolve (the compute stream, before a window's first gate_up): each routed expert of the window's B rows looks up
// its layer's residency row (a device table of entry addresses, null when the expert is not resident) and writes the
// window's address tables: tab's gate/up parts get the resident address or null (gate_up_args::skip leaves a null
// expert out), its down parts the resident address or the zero expert; tab2's parts of the window's experts are
// cleared to null (the second gate_up's table, which the host fills for the misses it serves on the GPU). It zeroes
// the CPU slot word and the second launch's run word, and writes both of the window's flags: set when every expert of
// the window is resident, so the stream's two waits pass without the host, else 0 (ME3: the flags take one value, set,
// which the service writes; resolve owns their reset, so a captured window replays with the same arguments).
//
// apply (the cache's copy stream, after a window's uploads): n (index, value) pairs into the cache's word table
// (residency rows, tab, tab2, the CPU slot word, the run word: one array of 8-byte words), in order; one block.
//
// combine (ME2b; card 0's compute stream, after the window's down and its wait on the down flag): the CPU-served
// slots' terms added into card 0's down output after the GPU's route sum and the shared expert's term. Down (the
// cache's instantiation, ME3) left those slots out, so the CPU's products enter here, in route order k, exactly once.
// Card 1's output never takes them, as in ME1 and ME2.

#include <cstdint>

namespace fn::experts {

struct resolve_args {
    const int32_t * ids;               // [rows][10] the window's routing
    int32_t rows;                      // 1..5
    const uint8_t * const * res;       // [512] the layer's resident entries (null: not resident)
    int64_t off[4];                    // the parts' byte offsets in an entry (gate/up, down, gate/up, down)
    int32_t parts;                     // 2 (one card's gate/up and down) or 4 (both cards on one device)
    const uint8_t * zero;              // the zero expert (down's entry of an expert that is not resident)
    const uint8_t ** tab;              // [parts][512] out
    const uint8_t ** tab2;             // [parts][512] out: the window's experts' entries null
    uint32_t * flags;                  // [2] the window's gate/up and down flags: out, set or 0 (ME3)
    uint32_t set;                      // the flags' set value
    uint64_t * ext_slots;              // out: 0
    uint64_t * run2;                   // out: 0 (its low word is gate_up_args::run_if)
};

// one block of 64 threads; false (nothing enqueued) on a bad argument
bool resolve(const resolve_args & a, void * stream);

// words[idx[i]] = val[i] for i < n (n <= max_apply), in order of i; idx and val are device arrays
constexpr int32_t max_apply = 4096;
bool apply(uint64_t * words, const int32_t * idx, const uint64_t * val, int32_t n, void * stream);

struct combine_args {
    const uint8_t * meta;       // meta_bytes, 16-byte aligned: card 0's routing meta of the window (down_args::meta)
    const float * wts;          // [rows][10] router weights (down_args::wts)
    const float * ext_ds;       // [50][2560] the CPU's products, a row of 2,560 a slot
    const uint64_t * ext_slots; // the slot word: bit s set for slot s served on the CPU
    float * out;                // [rows][2560] card 0's down output, read and written
    int32_t rows;               // 1..5
};
// for each row b and k = 0..9 in order whose slot (the meta's, as down numbers it) has its bit set:
// out[b][n] = fmaf(wts[b][k], ext_ds[slot][n], out[b][n]). The word 0, or the meta flagging a bad ID: nothing written.
// Ten blocks of 256 threads; false (nothing enqueued) on a bad argument
bool combine(const combine_args & a, void * stream);

// ME10: one device's finish of a cached FFN window, in place of combine and X-F's exchange (send_half and recv_add on
// both cards). a.out is card 0's down output p0 (read only), p1 card 1's [rows][2560], half[k] card k's ws.out[1]
// [rows][1280]. Per row b and column n: acc = p0[b][n] plus combine's CPU terms in its order (none when the slot word
// is 0 or the meta flags a bad ID); half[0][b][n] = acc + canon(p1[b][n]) for n < 1280, half[1][b][n - 1280] =
// p1[b][n] + canon(acc) above, canon the exchange's NaN canonicalization of the received operand. So every output is
// bitwise combine then the exchange, NaN payloads included. Writes every output; p0 and p1 are not written. Ten
// blocks of 256 threads; false (nothing enqueued) on a bad argument
bool combine_finish(const combine_args & a, const float * p1, float * const half[2], void * stream);

// ME3 (the graph executor): a replay's waits and window stamps as kernels, so a captured window holds kernels and
// copies only. A graph that holds stream memory operations or event nodes can block its launch in the driver until the
// GPU has run part of it, holding a driver lock the service's copies need while the GPU waits for those copies (load
// B's hang). wait_word: one thread polls word (volatile, a nanosleep backoff of 32 to 256 ns) until
// (int32_t)(*word - target) >= 0, the target *need or, need null, value; then, t not null, t[0] and t[1] the
// globaltimer (ns) at the wait's start and end, written to mapped host memory (system fence). post_word: *word = value
// after a system fence (a mapped host word: the stream's earlier copies have landed). False (nothing enqueued) on a
// bad argument.
struct wait_args {
    const uint32_t * word;
    const uint32_t * need; // or null: value
    uint32_t value;
    uint64_t * t; // [2] or null
};
bool wait_word(const wait_args & a, void * stream);
bool post_word(uint32_t * word, uint32_t value, void * stream);

} // namespace fn::experts
