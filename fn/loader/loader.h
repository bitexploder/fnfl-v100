#pragma once

// fn/loader: FNFL's model loader (R1). Reads the deployed GGUF (the target shards and the MTP draft file) through the
// shell's reader (fn/borrowed/gguf_tensors), maps the files read only, places every weight on the two cards as
// fn/model/placement.h says, and checks the placement byte for byte.
//
// Each card's weights are one allocation: the placement's regions in order, each 256-byte aligned. A region's layout:
//   gguf:               the tensor's card slice in slice order (placement.h).
//   hc_down (+ inject): fn::hc w_down [s][j][i], j < 320 from the down tensor, j = 320..323 from its inject tensor.
//   hc_up:              fn::hc w_up [c][s][j].
//                       (DR1: Q8_0 HC tensors are held planar, hc.h's Q8_0 layout: every code in that element order,
//                       then every FP16 scale [..][i / 32]; the check rebuilds the GGUF blocks.)
//   ex1_gate (+ up):    [512][gate_up_bytes(f)]: expert e is pack_gate_up of its 320 gate and 320 up rows.
//   ex1_down:           [512][down_bytes(f)]: expert e is pack_down of its 2,560 rows' 320 card columns.
// The PLE table stays in its mapping on the host; load_for_run pins it there when the residency is full (PN1).
// Load: worker threads fill pinned pieces (up to 64 MiB of a card's allocation each) from the mapping, repacking
// expert regions on the host, and each card's copies go through one stream, so the copy events time that card's
// host-to-device engine exactly.
//
// The check (verify_device): every region is read back and every tensor's card bytes, in slice order (inverting the
// HC layouts and, for EX1, the pack permutation derived by index probes of pack_gate_up and pack_down), are hashed
// with SHA-256 and compared with the hash of the GGUF slice read from the mapping.

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/experts_cpu.h"
#include "fn/loader/sha256.h"
#include "fn/model/model.h"
#include "fn/runtime/ple_host.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace fn::loader {

struct load_params {
    std::string model_path; // the first target shard
    std::string draft_path; // the MTP draft GGUF
    int32_t device[2] = {0, 1};
    int32_t threads_per_card = 5;
    int64_t piece_bytes = int64_t(64) << 20;
    // PN1: the PLE table's residency (load_for_run): full populates and pins it, lazy leaves the plain mapping
    runtime::ple_residency ple_table = runtime::ple_residency::lazy;
    // ME1: the target's routed experts are left to the expert cache (fn/runtime/expert_cache.h, which packs them into
    // host memory); their regions get no device bytes. With device[0] == device[1] both logical cards share the one
    // device, and card 1's copies of the tensors replicated whole on both cards read card 0's (model::single)
    bool expert_cache = false;
    // MEKV (--kv-host, one or two devices): the host-backed KV tier (fn/model/kv_tier.h): the target and draft KV
    // classes
    // in mapped pinned host memory and the tier's device class (state_params::kv_window); ME3 (--kv-window): its window
    bool kv_host = false;
    int32_t kv_window = 32768;
    // MU1 (--parallel 2..4, resident TP): the slots; above one, each slot's own classes (fn/model/state.h slot_class)
    // and the shared KV pool (fn/model/kv_pool.h) at n_cells a slot, and the other-reader lane's classes. MXMU1: or 2
    // on one card with the expert cache and the host-backed tier: each slot's classes of its own, no pool or lane
    int32_t n_slots = 1;
    // KV8 (--fnfl-kv-format q8, resident TP on two devices): the target KV in the 8-bit format (state_params::kv_q8);
    // pool_cells (more than one slot): the KV pool's physical cells, at least a slot's (0: a slot's)
    bool kv_q8 = false;
    int32_t pool_cells = 0;
    // TTTS1: this box's window (fn/model/state.h): the tensors outside it are not placed (model::win)
    model::window          win;
};

// Wall times are measured with steady_clock on the host; busy times are summed over threads (thread-seconds).
struct load_times {
    double open_s = 0;         // directory reads, mappings, the table checks and the region plan
    double fill_wall_s = 0;    // the fill and copy phase: first piece to both devices drained
    double read_busy_s = 0;    // thread-seconds copying GGUF slices into pinned pieces (page-cache reads included)
    double repack_busy_s = 0;  // thread-seconds in pack_gate_up and pack_down (their reads included)
    double wait_busy_s = 0;    // thread-seconds waiting for a pinned buffer to come back
    double h2d_s[2] = {0, 0};  // per card: the sum of its copies' event times (one stream: no overlap)
    int64_t h2d_bytes[2] = {0, 0};
    int32_t pieces[2] = {0, 0};
    // PN1 (full): the PLE table's populate (on its own thread beside the fill) and registration, wall seconds; their
    // sum is ple_pin_s. 0 when lazy
    double ple_populate_s = 0;
    double ple_register_s = 0;
    double ple_pin_s = 0;
};

// Map the files, check every GGUF tensor against the placement (name, format, shape, bytes; no tensor left out),
// read the PLE keys and plan both cards' regions. CPU only.
bool open_model(const load_params & p, model::model & m, load_times & t, std::string & err);

// cudaMalloc each card's weight allocation (m.weight_alloc bytes)
bool allocate_weights(model::model & m, std::string & err);

// fill both cards' weights from the mappings
bool fill_weights(const load_params & p, model::model & m, load_times & t, std::string & err);

// TTTS1 (split-test: random weights, no model data): every region the window places filled from synthetic GGUF bytes
// in place of the mapping, through the loader's own layouts (the same slices, HC layouts and EX1 packs as
// fill_weights): region by region, each member tensor's whole GGUF bytes written by synth(t, dst, stored_bytes) into
// host memory, its card units laid out and copied to the card. The same synth gives every window the same bytes of
// each tensor. Synchronous.
using synth_fn = std::function<bool(int32_t tensor, uint8_t * dst, int64_t bytes)>;
bool fill_weights_synth(model::model & m, const synth_fn & synth, std::string & err);

using card_digests = std::array<digest, 2>;

// SHA-256 of every tensor's card slice, read from the mapping in slice order (CPU); tensors a card does not hold get
// an all-zero digest. seconds: wall time.
bool hash_gguf_slices(const model::model & m, int32_t threads, std::vector<card_digests> & out, double & seconds);

// identifies the placement table and the files (path, size, mtime) for a cache of hash_gguf_slices
std::string source_fingerprint(const model::model & m);
bool write_hash_cache(const std::string & path, const std::string & fingerprint, const std::vector<card_digests> & d);
// false when the file is absent or its fingerprint differs
bool read_hash_cache(const std::string & path, const std::string & fingerprint, std::vector<card_digests> & d);

struct verify_report {
    int32_t placed[2] = {0, 0}; // tensors a card holds
    int32_t equal[2] = {0, 0};  // of them, read-back hash equal to the GGUF slice's
    int64_t bytes_read[2] = {0, 0};
    int32_t ex1_permutations = 0; // pack permutations derived and checked bijective
    double seconds = 0;
    std::vector<std::string> failures;
};

// Derive the EX1 pack permutations (CPU), then read both cards back and compare every tensor's hash with want.
bool verify_device(const load_params & p, const model::model & m, const std::vector<card_digests> & want,
                   verify_report & rep, std::string & err);

// The CPU dry run of the same check: each unit filled on the host by the loader's fill instead of read back (no GPU).
bool verify_host(const load_params & p, const model::model & m, const std::vector<card_digests> & want,
                 verify_report & rep, std::string & err);

// ME1: one target layer's routed experts as the expert cache holds them (fn/runtime/expert_cache.h). An entry is one
// expert of the layer with both cards' halves: [gate/up card 0][down card 0][gate/up card 1][down card 1], each the
// bytes the device region of that card would hold for the expert (pack_gate_up of its 320 gate and 320 up rows,
// pack_down of its 2,560 rows' 320 columns). Every part is a multiple of 256 bytes.
struct expert_layer {
    int32_t layer = -1;
    int32_t gu_region = -1, dn_region = -1; // placement regions (the gate region holds the up tensor too)
    experts::fmt gu_fmt = experts::fmt::iq3_xxs, dn_fmt = experts::fmt::q2_0;
    int64_t gu_bytes = 0, dn_bytes = 0; // one card's half of one expert
    int64_t entry_bytes() const { return 2 * (gu_bytes + dn_bytes); }
    int64_t gu_at(int32_t card) const { return card * (gu_bytes + dn_bytes); }
    int64_t dn_at(int32_t card) const { return card * (gu_bytes + dn_bytes) + gu_bytes; }
};
// the layer's regions and formats from the opened model (open_model); false (err set) for a layer outside the target
// or regions that are not EX1's
bool expert_layer_of(const model::model & m, int32_t layer, expert_layer & out, std::string & err);
// expert e's entry into dst (entry_bytes), packed from the mapping (CPU); false on a failed pack
bool pack_expert_entry(const model::model & m, const expert_layer & l, int32_t e, uint8_t * dst);
// ME4: the layer's routed experts as immutable native views into the model's GGUF mappings (the CPU miss path's
// operands, experts_cpu.h native_layer): the gate, up and down tensors' data, their row and expert strides. Checked:
// each tensor placed and mapped, the EX1 shapes ([512][640][2,560] gate and up, [512][2,560][640] down), gate and up
// of the layer's gate/up format and down of its down format, the row bytes the CPU kernels read (gu_row_bytes,
// dn_row_bytes), and every expert's bytes inside its file's mapping. The model owns the mappings; the views live as
// long as it. False (err) on any difference
// ME7: the startup account's line on the MTP draft's routed experts: the format resolved from the draft file (Q8_0, or
// the opt-in Q2_0), their packed bytes on each card and the placement's Q8_0 bytes, and the cards' weight totals that
// follow (open_model's; the target cache's pool is sized after them)
std::string draft_expert_line(const model::model & m);
// FMS5: the startup line on the target's BF16 router weights (every layer's ffn_gate_inp and ffn_gate_inp_shexp, from
// the file mappings): how many the tensor-core router's BF16 to FP16 conversion does not keep exactly, below FP16's
// normal range (2^-14) and beyond its range (65,504). Reported, not a gate (fn/kernels/dense/dense.h router_tc)
std::string router_f16_line(const model::model & m);
bool native_expert_layer(const model::model & m, const expert_layer & l, experts::cpu::native_layer & out,
                         std::string & err);

} // namespace fn::loader
