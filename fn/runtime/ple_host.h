#pragma once

// The PLE n-gram gather on the host (E1; FC-PLE1): each round row's 16 hashed rows of per_layer_token_embd, read from
// the mapped table, decoded to F32 by fn/quant's IQ4_NL decoder (C1b) and laid out head-major as the 2,560-value input
// e that both cards' PLE layers read. Pinned source (66e11c3): qwen4exp.cpp llm_graph_input_ple::set_input (lines
// 1899-1964), the hash and the window rule, and build_inp_ple (2022-2041), the head-major layout; the table and keys
// as qwen4exp.cpp's loader reads them (lines 97-155). The hash is adapted from set_input (integer, exact).
//
// Window rule: row t at position N_before + t has predecessors one and two back, from the round's own earlier rows
// (accepted tokens and proposals alike) or the slot's ledger. A missing predecessor (before position 0), or any
// predecessor at or before an EOS (ple.eos_token_id), reads as EOS; the row's own token is never cut. An image position
// carries ple.image_token_id in the ledger and is hashed as that ID (the pinned KV cells store it, ext.tok).
//
// Residency of the 28.8 GB table is the operator's setting (PN1 Amendment 1, 2026-10-15: lazy by default, the plain
// mapping and PL1's hints, as before PN1; ple_residency::full pins the whole table at load). A page fault costs time,
// not correctness.
// The output buffer is the round's upload slot: the caller refills it only after the round completes (FC-PLE1, FC-RD2).

#include <cstdint>
#include <string>

namespace fn::runtime {

constexpr int32_t ple_ngram = 3;
constexpr int32_t ple_heads_per_ngram = 8;
constexpr int32_t ple_heads = (ple_ngram - 1) * ple_heads_per_ngram; // 16
constexpr int32_t ple_head_dim = 160;                                // n_embd_per_layer
constexpr int32_t ple_row_bytes = ple_head_dim / 32 * 18;            // 5 IQ4_NL blocks: 90 bytes
constexpr int32_t ple_input = ple_heads * ple_head_dim;              // 2,560 = D
constexpr int32_t ple_conv_kernel = 4;
constexpr int32_t ple_layer = 1;
constexpr int32_t ple_eos = 248044; // FC-K: ple.eos_token_id (not the tokenizer's 248,046)
constexpr int32_t ple_max_rows = 5; // SP2: fn::ple::max_rows

struct ple_params {
    int32_t eos = -1;
    int32_t image = -1;              // ple.image_token_id
    uint64_t mult[ple_ngram] = {};   // ple.layer_multipliers
    uint32_t offset[ple_heads] = {}; // ple.head_offsets
    uint32_t vocab[ple_heads] = {};  // ple.head_vocab_sizes
    int64_t table_rows = 0;          // per_layer_token_embd's rows
};

// Reads the ple.* keys and the table's shape from the model's GGUF files (first shard; the table may be in any
// shard) and checks them against FNFL's constants: layers [1], ngram 3, 8 heads an n-gram, conv kernel 4, EOS
// 248,044, 16 offsets and vocabularies whose rows lie inside the table, the table IQ4_NL with 160 values a row.
// False, with err set, on any difference.
bool read_ple_params(const std::string & first_shard, ple_params & p, std::string & err);

// PN1: the table's residency (--fnfl-ple-table, fnfl-drive --ple-table). lazy (the default, Amendment 1): the plain
// mapping, nothing pinned, PL1's hints before each gather. full: populated and pinned at load by ple_pin.
enum class ple_residency : int32_t { lazy = 0, full = 1 };

// "lazy" or "full"; false for anything else
bool ple_residency_parse(const std::string & s, ple_residency & r);
const char * ple_residency_name(ple_residency r);

// PN1: the PLE table pinned in place. populate() reads the page-aligned range around [table, table + bytes) of a
// read-only file mapping in (madvise MADV_POPULATE_READ: every page read into the page cache and mapped, an I/O error
// reported); pin() then registers it with the CUDA driver (cudaHostRegister, Portable | ReadOnly; PN1's probe: a
// read-only mapping registers only with ReadOnly, and neither the 8 MiB RLIMIT_MEMLOCK nor VmLck/VmPin is charged).
// The pages stay resident and unevictable until release. While pinned, ple_prefetch on this table issues no hint
// (n_ranges 0). The current device's context registers (Portable: every context sees it); release unregisters. One
// pinned table at a time (fn::runtime holds one model). The owner outlives every gather that reads the table.
class ple_pin {
  public:
    ple_pin() = default;
    ~ple_pin();
    ple_pin(const ple_pin &) = delete;
    ple_pin & operator=(const ple_pin &) = delete;

    // the two steps, populate first; false with err naming the call (and suggesting --fnfl-ple-table lazy) on a
    // failure, which leaves nothing pinned
    bool populate(const uint8_t * table, uint64_t bytes, std::string & err);
    bool pin(std::string & err);
    void release();

    bool pinned() const { return pinned_; }
    uint64_t pinned_bytes() const { return pinned_ ? n_ : 0; } // the page-aligned range
    double populate_s = 0;                                     // the two steps' wall times
    double register_s = 0;
    // the kernel's net FOLL_PIN pages (/proc/vmstat nr_foll_pin_acquired - nr_foll_pin_released, system-wide; PN1's
    // probe: the driver's pin shows there, not in VmLck or VmPin) just before and after the registration; -1 unread.
    // Read at once, so a few pages a CPU short until the vmstat worker folds the CPUs' counts (vm.stat_interval):
    // PN1's unit test saw 8 of 16,384 missing here and none a few seconds later
    int64_t foll_pin_before = -1;
    int64_t foll_pin_after = -1;

  private:
    uint8_t * a_ = nullptr; // the page-aligned range (mutable for the system calls; nothing writes it)
    uint64_t n_ = 0;
    bool populated_ = false;
    bool pinned_ = false;
};

// /proc/vmstat's nr_foll_pin_acquired - nr_foll_pin_released (pages pinned with FOLL_PIN, system-wide); -1 unreadable
int64_t ple_foll_pinned_pages();

// The 16 row indices of one row: token t0 and its predecessors t1 (one back) and t2 (two back), -1 for a missing one.
void ple_row_indices(const ple_params & p, int32_t t0, int32_t t1, int32_t t2, int32_t (&idx)[ple_heads]);

// The round's indices idx[rows][16]: row t's predecessors from tok (rows before t) or ledger[0 .. n_before) (the
// slot's tokens before row 0's position). False for rows outside 1..5, a token outside [0, V) (fn::embed's rule) or
// null pointers.
bool ple_indices(const ple_params & p, const int32_t * ledger, int64_t n_before, const int32_t * tok, int32_t rows,
                 int32_t * idx);

// Gather and decode: e[rows][2560] from the table (the mapped per_layer_token_embd data, table_rows x 90 bytes).
// False for an index outside the table or null pointers.
bool ple_gather(const ple_params & p, const uint8_t * table, const int32_t * idx, int32_t rows, float * e);

// PL1: before ple_gather reads the rows idx[rows][16], posix_madvise(POSIX_MADV_WILLNEED) on every page they touch (a
// 90-byte row spans one or two pages), with equal and adjacent pages merged into one range each. On a cold table the
// kernel queues the reads together, so a step waits about one disk read instead of one major fault a row in turn; on
// resident pages it is a system call a range and no I/O. The pinned engine's prefetch_rows (66e11c3: qwen4exp.cpp
// 1959, llama-mmap.cpp 455-489 and 731-745). table must point into a file mapping (the hint needs a mapped range).
// The rows read are unchanged. n_ranges: the posix_madvise calls made. False, with err set, for null pointers, rows
// outside 1..5, an index outside the table, or a failed call (the ranges lie inside the mapping, so a failure is a
// defect, not a cold page). PN1: on a table pinned by ple_pin it returns true with n_ranges 0 and no system call.
bool ple_prefetch(const ple_params & p, const uint8_t * table, const int32_t * idx, int32_t rows, int32_t & n_ranges,
                  std::string & err);

} // namespace fn::runtime
