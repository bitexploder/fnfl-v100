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

#include "fn/loader/sha256.h"
#include "fn/model/model.h"
#include "fn/runtime/ple_host.h"

#include <array>
#include <cstdint>
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

} // namespace fn::loader
