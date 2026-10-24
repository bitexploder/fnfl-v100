#pragma once

// ME2: the routed experts' CPU miss path, Strata-V100 3d4ce4e0's native-expert pool (src/kernels/cpu/pool.cpp,
// modes 5 and 6; MIT, the notice in experts_cpu.cpp) on FNFL's packed experts. A CPU-served expert is computed whole
// on the CPU: both cards' gate/up (the 640 channels) with SwiGLU in F32, the down input quantized by Strata's
// quantizer for the down format, and both cards' down (all 640 columns), so its output is the expert's whole
// 2,560-value product (the GPU's combine applies the router weight; experts.h down_args::ext_ds).
//
// ME4: a job reads its expert's native GGUF rows in place from the model's mappings (native_layer: the loader's
// immutable views, fn/loader/loader.h native_expert_layer), so no layout is rebuilt per miss. Without native views
// (the reference path, kept while ME4's decision is open: tests only, never a silent serving fallback) the row
// kernels (cpu_kernels.h), which read GGUF's native blocks, need each job to un-permute its piece of the packed
// backing (pack_gate_up / pack_down's exact inverse) into a per-worker scratch first. Both give the same operands
// bitwise, so the same products:
//   gate/up: one 32-channel tile of one card (tile g: gate rows 32 g .. 32 g + 31, then the same up rows), 64 native
//            rows of 2,560 columns (gu_scratch_bytes; Q5_K's 112,640 bytes the largest);
//   down:    down_piece rows (128 output rows) of 640 columns, card 0's 320 then card 1's (dn_scratch_bytes).
// A window's activation (FNFL's codes and {d, s} scales, act.h) is dequantized (d x code) and quantized to Q8_K once
// for all the window's CPU experts (gate_up_act).
//
// Numerical contract (ME2; bitwise CPU = GPU revoked by the operator): each product against FP64 on the same
// quantized operands (experts_cpu_test.cpp), the bound stated there from the format's arithmetic.

#include "fn/kernels/experts/cpu_kernels.h"
#include "fn/kernels/experts/experts.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fn::experts::cpu {

constexpr int32_t d_model = 2560;
constexpr int32_t n_ch = 320; // channels of one card's half
constexpr int32_t n_ff = 640; // an expert's channels
constexpr int32_t n_tiles = 2 * (n_ch / 32); // gate/up tiles of an expert (both cards)
constexpr int32_t down_piece = 128;          // output rows a down job
constexpr int32_t n_pieces = d_model / down_piece;

// true for the formats these products take: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, Q4_K, Q5_K; down Q2_0,
// IQ4_NL, Q5_1, Q8_0
bool gate_up_supported(fmt f);
bool down_supported(fmt f);

// GGUF bytes of a native row: gate/up over 2,560 columns, down over 640
int64_t gu_row_bytes(fmt f);
int64_t dn_row_bytes(fmt f);
int64_t gu_scratch_bytes(fmt f); // 64 rows
int64_t dn_scratch_bytes(fmt f); // down_piece rows
constexpr int64_t scratch_bytes = 112640; // the largest of either over the supported formats (Q5_K's tile)

// pack_gate_up's inverse for tile g (0..9) of one card's packed gate/up (w: gate_up_bytes(f)): dst row r (0..63) at
// r x gu_row_bytes(f) holds GGUF row 32 g + r of the card's gate (r < 32) or up (r >= 32), bitwise
void unpack_gate_up_tile(fmt f, const uint8_t * w, int32_t g, uint8_t * dst);
// pack_down's inverse for output rows n0 .. n0 + n - 1 of both cards' packed down (w0, w1: down_bytes(f) each): dst
// row i at i x dn_row_bytes(f) holds GGUF row n0 + i (card 0's 320 columns, then card 1's), bitwise
void unpack_down_rows(fmt f, const uint8_t * w0, const uint8_t * w1, int32_t n0, int32_t n, uint8_t * dst);

// a window row's gate/up activation: n_q8k blocks of Q8_K from FNFL's codes [2560] and {d, s} scales [80]
constexpr int32_t n_q8k = d_model / 256;
void gate_up_act(const int8_t * codes, const float * scales, block_q8_k * act);
// MX7 (the prompt CPU share): a prompt row's gate/up activation: n_q8k blocks of Q8_K from its F16 values [2560]
// (each converted exactly to F32, then quantize_q8_k)
void gate_up_act_f16(const uint16_t * x16, block_q8_k * act);
// MX7: n F32 values to F16, rounded to nearest even (F16C); false when any is beyond 65,504 or NaN (ep_down's rule
// for a partial; the values are still written)
bool to_f16(const float * x, int64_t n, uint16_t * out);
// the down activation of one row of SwiGLU output h [640] in f's activation format (dn_act_bytes(f) bytes at act)
constexpr int64_t dn_act_max = 2048; // bytes a row's down activation slot (act_q with its bit-plane image the largest)
int64_t dn_act_bytes(fmt f);
void down_act(fmt f, const float * h, void * act);

// the products on native rows: a tile's 32 gate rows (gate) and 32 up rows (up), row_bytes apart (gu_row_bytes;
// unpack_gate_up_tile's scratch: gate = tile, up = tile + 32 rows), against nt rows' Q8_K activations, ff[t][r] for
// r < 32 the SwiGLU of the tile's channel r; down rows (n of them, row_bytes apart: dn_row_bytes) against nt rows'
// down activations, out[t][i] for i < n
void gate_up_tile(fmt f, const uint8_t * gate, const uint8_t * up, int64_t row_bytes, const block_q8_k * const * act,
                  int32_t nt, float * const * ff);
void down_rows(fmt f, const uint8_t * rows, int64_t row_bytes, int32_t n, const void * const * act, int32_t nt,
               float * const * out);

// ME4: one layer's routed experts as the model's GGUF mappings hold them (the loader's native views): gate and up
// [512 experts][640 rows][2,560 columns], down [512][2,560 rows][640 columns]; a row's blocks contiguous. Bases,
// row and expert strides in bytes; immutable, the model owns the mappings and outlives every CPU job. No GGUF type
struct native_layer {
    const uint8_t * gate = nullptr;
    const uint8_t * up = nullptr;
    const uint8_t * down = nullptr;
    int64_t gu_row = 0, gu_expert = 0; // a gate/up row (gu_row_bytes), one expert's 640 rows
    int64_t dn_row = 0, dn_expert = 0; // a down row (dn_row_bytes), one expert's 2,560 rows
    bool valid() const { return gate != nullptr && up != nullptr && down != nullptr; }
};

// ME4: n experts of expert_bytes each, from offset, inside a file of size bytes (the loader's extent check; no
// overflow: the bound is divided, not the extent multiplied)
inline bool native_fits(uint64_t offset, uint64_t size, int64_t expert_bytes, int32_t n) {
    return n > 0 && expert_bytes > 0 && offset <= size && uint64_t(expert_bytes) <= (size - offset) / uint64_t(n);
}

// The routing meta of a launch's IDs ([rows][10]) on the host, as the kernels number the work (experts.h "work"): the
// distinct experts in first-appearance order over p = 10 b + k, each one's rows (bit b) and slot base; IDs outside
// [0, 512) are skipped (the launch then flags them and writes nothing). e, mask, base: 50 entries. Returns n_u and
// sets n_slots.
int32_t window_meta(const int32_t * ids, int32_t rows, int32_t * e, int32_t * mask, int32_t * base, int32_t & n_slots);

// MX7: a job's rows, up to Strata's MAXT (the kernels' max_nt): a prompt job's 1..8; a decode window's 1..max_rows (5)
constexpr int32_t job_max_rows = max_nt;
static_assert(job_max_rows >= max_rows, "a job serves a whole decode window");

// One CPU-served expert of a window (or MX7: of a prompt layer): both cards' packed parts, the rows it serves and its
// output rows
struct job {
    const uint8_t * gu[2] = {nullptr, nullptr}; // gate_up_bytes(gu fmt) each (the reference path)
    const uint8_t * dn[2] = {nullptr, nullptr}; // down_bytes(down fmt) each (the reference path)
    // ME4: the expert's native rows (native_layer, set by set_native); ngate null: the packed parts above
    const uint8_t * ngate = nullptr;
    const uint8_t * nup = nullptr;
    const uint8_t * ndown = nullptr;
    int64_t ngu_row = 0, ndn_row = 0;
    int32_t nt = 0;                 // rows (1..job_max_rows; a decode window's 1..5)
    int32_t row[job_max_rows] = {}; // the window rows (gate_up_act's index)
    float * out[job_max_rows] = {}; // its product rows, 2,560 F32 each
};

// ME4: job j reads expert e of a layer's native views (the rows of e, both cards' halves: gate/up rows 0..639, down
// rows' 640 columns)
inline void set_native(const native_layer & v, int32_t e, job & j) {
    j.ngate = v.gate + int64_t(e) * v.gu_expert;
    j.nup = v.up + int64_t(e) * v.gu_expert;
    j.ndown = v.down + int64_t(e) * v.dn_expert;
    j.ngu_row = v.gu_row;
    j.ndn_row = v.dn_row;
}

// ME4 (the native views' check: ex-cpu-test, and the residency's at cache init): job j's native rows (set_native)
// against the reference path's reconstruction of its packed parts (j.gu, j.dn), byte for byte: both cards' ten
// gate/up tiles and every down piece. scratch: scratch_bytes. False with where naming the first difference
bool native_equal(fmt gu, fmt dn, const job & j, uint8_t * scratch, std::string & where);

struct run_stats {
    double gu_ms = 0, act_ms = 0, dn_ms = 0; // phase A, the service's down activation, phase C (wall)
    int64_t bytes = 0;                       // packed expert bytes read (both cards' gate/up and down)
    // ME2b (diagnostic, pool::set_unit_timing; zero when off): thread-ms summed over the workers and the host, each
    // unit's un-permute (into scratch) apart from its product: gate/up tiles (phase A), down pieces (phase C)
    double gu_unpack_ms = 0, gu_compute_ms = 0, dn_unpack_ms = 0, dn_compute_ms = 0;
};

// Strata's ExpertPool protocol (pool.hpp, pool.cpp): one batch at a time, published by bumping an epoch; workers
// claim tasks through one packed head word (epoch << 32 | n << 16 | i) by compare-exchange, count done, and park
// between batches spinning on _mm_pause for 20 ms, then sleeping on a condition variable (seq_cst sleepers / epoch,
// pool.cpp publish). A phase is 3 x (workers + host) tasks, each an equal range of the phase's units over all its
// experts (a range may span experts): phase A's units are gate/up tiles (whole 32-channel tiles: an FNFL edit,
// Strata's are rows), phase C's are down pieces. The host (the caller: the cache's service) drains each phase with the
// workers on its own scratch, as Strata's (R2.2; ME2b), then waits for the rest; between the phases it quantizes the
// down activations. Edits against the source, in the ME2 Result: the workers are pinned one a CPU by the caller's
// list, a stall of 60 s fails the batch instead of aborting.
class pool {
  public:
    pool() = default;
    ~pool() { stop(); }
    pool(const pool &) = delete;
    pool & operator=(const pool &) = delete;

    // one worker a CPU of cpus (-1: unpinned); false and err when a pinning fails. ME3: park_ms, the workers' spin
    // before they sleep (fn/cache_settings.h park-ms; 20 as ME2b)
    bool start(const std::vector<int32_t> & cpus, std::string & err, int32_t park_ms = 20);
    void stop();
    int32_t workers() const { return n_; }
    // the window's CPU experts on the gate/up and down formats: act [rows][n_q8k] (gate_up_act), the calling thread
    // draining with the workers. False on a stall. MX7: one caller at a time (the cache service's thread or the prompt
    // CPU share's, prompt_share.h): a caller that finds the pool held fails (a debug build asserts) instead of
    // touching its batch state; the hold's acquire and release order one caller's batch before the next's
    bool run(fmt gu, fmt dn, const block_q8_k * act, const job * jobs, int32_t n, run_stats & st);
    // TEST: the tasks phase A and C were cut into last run, and the units each covered
    int32_t last_tasks() const { return tasks_; }
    // TEST (ex-cpu-test): tasks the host has claimed since start (it drains)
    int64_t host_tasks() const { return host_tasks_; }
    // TEST (ex-cpu-test; ME2b amendment 1): the per-unit clocks of run_stats' un-permute and product thread-ms. Off by
    // default and off in serving: two clock reads a unit inside the hot loop would change the path measured. Set
    // between runs only.
    void set_unit_timing(bool on) { unit_timing_ = on; }
    // TEST (ex-cpu-test, MX7): the pool held as by another caller, between runs (a release build refuses the next run)
    void test_hold(bool on) { held_.store(on, std::memory_order_release); }
    bool unit_timing() const { return unit_timing_; }
    // TEST (ex-cpu-test): the last run's SwiGLU output of job j's row t ([640]) and its down activation
    const float * test_h(int32_t j, int32_t t) const {
        return h_.data() + (size_t(j) * job_max_rows + size_t(t)) * n_ff;
    }
    const void * test_act(int32_t j, int32_t t) const { return a2_[size_t(j) * job_max_rows + size_t(t)].b; }
    // worker i's affinity mask as read back after its pinning (CPUs 0..63)
    uint64_t affinity(int32_t i) const { return masks_[size_t(i)].load(); }

  private:
    std::chrono::milliseconds spin_before_sleep{20};
    static constexpr std::chrono::seconds stall{60};
    void worker(int32_t id, int32_t cpu);
    void publish();
    int32_t claim(uint32_t epoch);
    uint32_t begin_batch(int32_t n);
    bool wait_parked();
    bool wait_done(int32_t n);
    int32_t drain(int32_t id, uint32_t epoch); // the tasks claimed
    bool run_phase(int32_t mode);

    int32_t n_ = 0;
    std::vector<std::thread> threads_;
    std::vector<uint8_t *> scratch_; // per worker, then the host's (index n_): scratch_bytes (64-byte aligned)
    // per worker, then the host's: thread-ms of the un-permutes and products, each written by its own thread only
    struct alignas(64) unit_ms {
        double gu_unpack = 0, gu_compute = 0, dn_unpack = 0, dn_compute = 0;
    };
    std::vector<unit_ms> ums_;
    int64_t host_tasks_ = 0;
    bool unit_timing_ = false; // read by the workers after the epoch's acquire
    std::atomic<uint64_t> head_{0};
    std::atomic<uint32_t> epoch_{0}, done_{0}, parked_{0}, sleepers_{0};
    std::atomic<bool> stop_{false};
    std::atomic<bool> held_{false}; // MX7: a caller inside run
    std::mutex sleep_mu_;
    std::condition_variable sleep_cv_;
    std::atomic<int32_t> pin_failed_{0};
    std::vector<std::atomic<uint64_t>> masks_;
    // the batch (written by the host before begin_batch; read by the workers after the epoch's acquire)
    int32_t mode_ = 0; // 1 gate/up, 2 down
    int32_t tasks_ = 0, units_ = 0;
    fmt gu_ = fmt::iq2_xxs, dn_ = fmt::q2_0;
    const block_q8_k * act_ = nullptr;
    const job * jobs_ = nullptr;
    std::vector<float> h_;      // [job][row][640] the SwiGLU outputs
    struct alignas(64) act_slot {
        uint8_t b[dn_act_max];
    };
    std::vector<act_slot> a2_; // [job][row] the down activations
};

} // namespace fn::experts::cpu
