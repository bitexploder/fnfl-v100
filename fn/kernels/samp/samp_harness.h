#pragma once

// SM1: the sampler's two-card harness for samp-test and samp-bench (test code): both cards' buffers and card views on
// fn::xchg's rings, and a batch of decisions enqueued stage by stage on card 0 then card 1, so every send precedes its
// read (DEV0 == DEV1: loopback on one device and one stream).

#include "fn/kernels/round/round.h"
#include "fn/kernels/samp/samp.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace fn::samp::harness {

constexpr int32_t V = n_vocab;
constexpr int32_t NC = n_card;
constexpr int32_t kmax = 512;      // decisions a batch
constexpr int32_t ring_cap = 4096; // the tests' largest window

bool cuda_ok(cudaError_t e, const char * what);

// owns one device allocation of n elements of T (on the current device)
template <typename T> struct dev_buf {
    T * p = nullptr;
    size_t n = 0;

    explicit dev_buf(size_t count) : n(count) {
        if (!cuda_ok(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc")) {
            p = nullptr;
        }
    }

    ~dev_buf() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dev_buf(const dev_buf &) = delete;
    dev_buf & operator=(const dev_buf &) = delete;
};

struct device_guard {
    explicit device_guard(int32_t d) { cudaSetDevice(d); }
};

struct card_s {
    int32_t dev;
    device_guard guard;
    cudaStream_t st = nullptr;
    dev_buf<params> prm{1};
    dev_buf<float> bias{size_t(V)};
    dev_buf<uint32_t> counts{size_t(V)};
    dev_buf<int32_t> ring{size_t(ring_cap)};
    dev_buf<record> rec{1};
    dev_buf<qlist> ql{size_t(max_drafts)};
    dev_buf<float> qs{size_t(max_drafts) * NC};
    dev_buf<uint8_t> work{workspace_bytes()};
    dev_buf<float> logits{size_t(max_rows) * NC};
    dev_buf<fn::round::desc> desc{size_t(kmax)};
    dev_buf<int32_t> prop{4};
    dev_buf<uint32_t> pair{size_t(max_rows) * 2}; // SP2: a (value, ID) pair a row
    dev_buf<uint8_t> gwork{256}; // head::greedy_work_bytes() (64) and more
    card_view v;

    explicit card_s(int32_t d);
    ~card_s();
    card_s(const card_s &) = delete;
    card_s & operator=(const card_s &) = delete;
};

// one decision's inputs (the descriptor's host fields)
struct dec_in {
    int32_t kind = static_cast<int32_t>(fn::round::kind::generate);
    int32_t rows = 1;
    int32_t n_before = 0;
    int32_t tok[max_rows] = {}; // SP2: five
};

struct tester {
    card_s * c[2] = {nullptr, nullptr};
    fn::xchg::rings * rg = nullptr;
    bool loop = false;
    options opt;
    params prm = {};
    std::vector<int32_t> ledger; // the window's ledger (penalties)

    cudaStream_t st(int32_t r) const { return loop ? c[0]->st : c[r]->st; }
    bool init();
    // the options, the bias table, and the window from the ledger (both cards); window_bins: the test hook
    bool set_options(const options & o, const std::vector<int32_t> & led, int32_t window_bins = hist_bins);
    // rows of the whole vocabulary: card r gets IDs r * 124,160 .. of each
    bool put_logits(const std::vector<std::vector<float>> & rows);
    // the draft rows' q: kept lists (both cards) and the shards (card r its half of a whole-vocabulary row)
    bool put_q(const std::vector<std::vector<int32_t>> & ids, const std::vector<std::vector<float>> & q,
               const std::vector<std::vector<float>> & shard);
    // the descriptors of a batch on both cards
    bool put_desc(const std::vector<dec_in> & in);
    // enqueue the batch's decisions (each stage on card 0 then card 1); row0: the logits row a B = 1 decision reads;
    // discard: the record dropped after each (a static window). With per_card, card r's work only (a graph capture).
    bool enqueue_batch(const std::vector<dec_in> & in, int32_t row0, bool discard, int32_t per_card = -1);
    // put_desc, enqueue_batch, sync, and both cards' descriptors back
    bool run(const std::vector<dec_in> & in, std::vector<fn::round::desc> out[2], int32_t row0 = 0,
             bool discard = false);
    bool sync() const;
    // both cards' error words, then cleared
    bool errs(uint32_t e[2]);

    // run's graphs (two cards, not loopback): one captured batch a card, replayed while the batch's shape is unchanged
    bool graphs = true;
    cudaGraphExec_t ge[2] = {nullptr, nullptr};
    std::vector<int32_t> ge_key;
    void drop_graphs();
    ~tester() { drop_graphs(); }
};

} // namespace fn::samp::harness
