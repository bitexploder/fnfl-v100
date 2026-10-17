#pragma once

// The routed-expert kernels against an FP64 CPU reference (EX1 step 4), and the synthetic weights the test and the
// bench share: n_synth distinct experts per format, valid random blocks from fn/quant's seeded generators (C1b), plus
// one poison expert that every unrouted ID points at, cached in a directory so the generation runs outside the locks.

#include "fn/kernels/experts/experts.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fn::experts {

constexpr int32_t n_synth = 16;

// One format's synthetic experts at full width (both cards): gate/up formats hold 640 gate rows then 640 up rows of
// 2560 weights per expert; down formats hold 2560 rows of 640 weights. Expert n_synth is the poison expert. Q8_0 (DR1)
// is both: down says which.
struct synth {
    fmt f = fmt::iq3_xxs;
    bool down = false;       // down's shape (Q2_0 and IQ4_NL always)
    size_t row_bytes = 0;    // GGUF bytes of one row
    size_t expert_bytes = 0; // GGUF bytes of one expert (both cards)
    std::vector<uint8_t> bytes;
    const uint8_t * expert(int32_t s) const { return bytes.data() + size_t(s) * expert_bytes; }
};

// Loads dir/<fmt>-fnfl.bin, generating and writing it first if it is missing (CPU only). False on failure. With
// deployed (ep-test only, C1b Amendment 4): dir/<fmt>-fnfl-deployed.bin, each block's d from the format's deployed
// range on ep-test's own seed stream (fn/quant/synth.h). down (DR1): Q8_0's down shape, dir/Q8_0-down-fnfl.bin (the
// other formats' shape follows the format).
bool load_synth(fmt f, const std::string & dir, synth & out, bool deployed = false, bool down = false);

// One card's packed buffer of n_copies x 512 experts: expert e of copy c is synthetic expert map[e] (n_synth: poison).
bool pack_card(const synth & s, int32_t card, const std::vector<int32_t> & map, int32_t n_copies,
               std::vector<uint8_t> & out);

// The grid buffer (grid_bytes) built from FNFL's tables (grids.h)
bool grid_buffer(std::vector<uint8_t> & out);

} // namespace fn::experts

// ex-prep: writes the synthetic weights of all seven formats and Q8_0's two shapes (DR1) to dir (CPU only). Returns 0
// on success.
int fnfl_ex_prep(const char * dir);

// ex-test (EX2): every deployed gate/up instantiation (routed format, shared gate format, shared up format) and every
// deployed down instantiation (routed down, shared down) at each row count in rows_mask (bit B-1), the order of B
// rotating between instantiations; both cards' halves on device against the FP64 reference (the routed experts and
// the shared expert on DP2's synthetic rows); each launch run twice and compared bitwise; each row at B < 4 against the
// same row at B = 4, bitwise (NC1 section 2); all-zero input and a bad expert ID. formats: "all", "san" (Compute
// Sanitizer's covering set: each routed and each shared format once, every down pair) or a list of routed gate/up
// formats (iq2_xxs, ...). XB2: the routing meta by write_meta against the host's, and routings that place the shared
// expert's units across the snake's turn (Amendment 1). With base_dir (71b92a9's experts cubins), also the
// xb3-bitwise arm: every launch's outputs byte-identical to 71b92a9's kernels' on the same inputs; and the
// xb4-bitwise arm: every down instantiation against 71b92a9's at the cases and at U cases of each n_u mod 3. Prints
// "ex-test: PASS" and returns 0 when every check passes.
int fnfl_ex_test(int device, uint32_t rows_mask, const char * formats, const char * dir,
                 const char * base_dir = nullptr);
