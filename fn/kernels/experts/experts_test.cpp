#include "fn/kernels/experts/experts_test.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/experts/experts_cache.h"
#include "fn/kernels/experts/experts_cpu.h"
#include "fn/kernels/experts/grids.h"
#include "fn/model/shapes.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/decode.h"
#include "fn/quant/synth.h"
#include "fn/runtime/expert_cache.h"
#include "fn/runtime/expert_policy.h"
#include "fn/runtime/split_test.h"
#include "fn/tools/test_dump.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>
#include <cxxabi.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fn::experts {
namespace {

constexpr int32_t D = fn::n_embd;           // 2560
constexpr int32_t NFF = fn::n_ff_expert;    // 640: both cards
constexpr int32_t NU = fn::n_expert_used;   // 10
constexpr int32_t NE = fn::n_expert;        // 512
constexpr int32_t n_group = n_ff_card / 32; // 32-channel groups a card owns
constexpr int32_t n_sm = 72;                // the gate/up grid: the snake deal turns every 72 ranks (XB2 (d))

fn::quant::wfmt to_quant(fmt f) {
    using fn::quant::wfmt;
    static const wfmt q[11] = {wfmt::iq2_xxs, wfmt::iq2_xs, wfmt::iq2_s, wfmt::iq3_xxs, wfmt::iq3_s, wfmt::q2_0,
                               wfmt::iq4_nl,  wfmt::q8_0,   wfmt::q4_k,  wfmt::q5_k,    wfmt::q5_1}; // fmt's order
    return q[static_cast<int32_t>(f)];
}

bool is_down(fmt f) {
    return f == fmt::q2_0 || f == fmt::iq4_nl || f == fmt::q5_1;
}

// a routed down format's fn::dense format (the shared down's synthetic rows)
fn::dense::fmt dense_of(fmt f) {
    return f == fmt::q2_0 ? fn::dense::fmt::q2_0 : f == fmt::q8_0 ? fn::dense::fmt::q8_0 : fn::dense::fmt::iq4_nl;
}

// runs fn(i) for i in [0, n) on up to 12 threads
template <typename F> void parallel_for(int32_t n, const F & fn) {
    const int32_t nt = std::min<int32_t>(n, 12);
    std::atomic<int32_t> next{0};
    std::vector<std::thread> th;
    th.reserve(static_cast<size_t>(nt));
    for (int32_t t = 0; t < nt; ++t) {
        th.emplace_back([&]() {
            for (int32_t i = next++; i < n; i = next++) {
                fn(i);
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
}

} // namespace

bool load_synth(fmt f, const std::string & dir, synth & out, bool deployed, bool down, bool gu) {
    const fn::quant::wfmt q = to_quant(f);
    const bool q8_down = down && f == fmt::q8_0; // DR1: Q8_0's down shape, its own file and seeds
    const bool q2_gu = gu && f == fmt::q2_0;     // ME7: Q2_0's gate/up shape, likewise
    const bool dn = (is_down(f) && !q2_gu) || q8_down;
    const int64_t n_per_row = dn ? NFF : D;
    const int64_t n_rows = dn ? D : 2 * NFF;
    out.f = f;
    out.down = dn;
    out.row_bytes = fn::quant::row_bytes(q, n_per_row);
    out.expert_bytes = out.row_bytes * static_cast<size_t>(n_rows);
    if (out.row_bytes == 0) {
        return false;
    }
    const size_t total = out.expert_bytes * (n_synth + 1);
    // fn/quant's blocks (C1b), never a quantizer's
    const std::string path =
        dir + "/" + fmt_name(f) + (q8_down ? "-down" : q2_gu ? "-gu" : "") +
        (deployed ? "-fnfl-deployed.bin" : "-fnfl.bin");
    out.bytes.assign(total, 0);
    if (FILE * fp = fopen(path.c_str(), "rb")) {
        const size_t got = fread(out.bytes.data(), 1, total, fp);
        const bool at_end = fgetc(fp) == EOF;
        fclose(fp);
        if (got == total && at_end) {
            return true;
        }
    }
    // generate: valid random blocks per expert (fn/quant/synth.h), case index (format, expert)
    const int64_t n_blocks =
        static_cast<int64_t>(out.expert_bytes / static_cast<size_t>(fn::quant::info(q).block_bytes));
    parallel_for(n_synth + 1, [&](int32_t s) {
        const uint64_t case_index =
            uint64_t(static_cast<int32_t>(f) + (q8_down ? 8 : 0) + (q2_gu ? 32 : 0)) * (n_synth + 1) + uint64_t(s);
        uint8_t * dst = out.bytes.data() + size_t(s) * out.expert_bytes;
        if (deployed) {
            fn::quant::synth_blocks_deployed(q, fn::quant::synth_seed_ep(case_index), n_blocks, dst);
        } else {
            fn::quant::synth_blocks(q, fn::quant::synth_seed(case_index), n_blocks, dst);
        }
    });
    FILE * fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        fprintf(stderr, "ex: cannot write %s\n", path.c_str());
        return false;
    }
    const bool ok = fwrite(out.bytes.data(), 1, total, fp) == total;
    return fclose(fp) == 0 && ok;
}

bool pack_card(const synth & s, int32_t card, const std::vector<int32_t> & map, int32_t n_copies,
               std::vector<uint8_t> & out) {
    const bool dn = s.down;
    const size_t pb = static_cast<size_t>(dn ? down_bytes(s.f) : gate_up_bytes(s.f));
    if (pb == 0 || map.size() != static_cast<size_t>(NE) || n_copies < 1) {
        return false;
    }
    std::vector<uint8_t> packed(pb * (n_synth + 1));
    bool ok = true;
    for (int32_t i = 0; i <= n_synth && ok; ++i) {
        const uint8_t * e = s.expert(i);
        if (dn) {
            ok = pack_down(s.f, e + size_t(card) * (s.row_bytes / 2), s.row_bytes, packed.data() + pb * i);
        } else {
            ok = pack_gate_up(s.f, e + size_t(card) * n_ff_card * s.row_bytes,
                              e + (size_t(NFF) + size_t(card) * n_ff_card) * s.row_bytes, s.row_bytes,
                              packed.data() + pb * i);
        }
    }
    if (!ok) {
        return false;
    }
    out.assign(pb * NE * static_cast<size_t>(n_copies), 0);
    for (int32_t c = 0; c < n_copies; ++c) {
        for (int32_t e = 0; e < NE; ++e) {
            const int32_t i = map[static_cast<size_t>(e)];
            const size_t src = pb * size_t(i < 0 || i > n_synth ? n_synth : i);
            std::memcpy(out.data() + (size_t(c) * NE + e) * pb, packed.data() + src, pb);
        }
    }
    return true;
}

bool grid_buffer(std::vector<uint8_t> & out) {
    out.assign(grid_bytes, 0);
    return grid_table(grid_src, grid_src_bytes, out.data());
}

namespace {

// the pass bound: max |gpu - ref| over max |ref| per output tensor, against the FP64 reference
constexpr double rel_bound = 1e-5;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "ex-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// owns one device allocation of n bytes
struct dev_mem {
    uint8_t * p = nullptr;
    size_t n = 0;

    explicit dev_mem(size_t bytes) : n(bytes) {
        if (!cuda_ok(cudaMalloc(&p, n), "cudaMalloc")) {
            p = nullptr;
        }
    }

    ~dev_mem() {
        if (p != nullptr) {
            cudaFree(p);
        }
    }

    dev_mem(const dev_mem &) = delete;
    dev_mem & operator=(const dev_mem &) = delete;

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); } // the allocation is 256-byte aligned

    bool put(const void * h, size_t bytes) const {
        return p != nullptr && bytes <= n && cuda_ok(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice), "put");
    }

    bool get(void * h, size_t bytes) const {
        return p != nullptr && bytes <= n && cuda_ok(cudaMemcpy(h, p, bytes, cudaMemcpyDeviceToHost), "get");
    }
};

// FNFL's activation format (act.h) of rows x k values held in one buffer: the codes, then the scales (16-byte aligned
// when the buffer is: the codes' bytes are a multiple of 32)
size_t act_bytes(int32_t rows, int32_t k) {
    return fn::act::codes_bytes(rows, k) + fn::act::scales_bytes(rows, k);
}
fn::act::view act_in(const uint8_t * p, int32_t rows, int32_t k) {
    return {reinterpret_cast<const int8_t *>(p), reinterpret_cast<const float *>(p + fn::act::codes_bytes(rows, k)),
            rows, k};
}
fn::act::out_view act_out(uint8_t * p, int32_t rows, int32_t k) {
    return {reinterpret_cast<int8_t *>(p), reinterpret_cast<float *>(p + fn::act::codes_bytes(rows, k)), rows, k};
}
// the reference quantizer into such a buffer
void act_quantize(const float * x, int32_t rows, int32_t k, uint8_t * p) {
    fn::quant::act_quantize_ref(x, rows, k, reinterpret_cast<int8_t *>(p),
                                reinterpret_cast<float *>(p + fn::act::codes_bytes(rows, k)));
}
// the first n rows of such a buffer (of rows x k) to values, code x d (FP64)
void act_values(const uint8_t * p, int32_t rows, int32_t k, int32_t n, double * out) {
    const int8_t * codes = reinterpret_cast<const int8_t *>(p);
    std::vector<float> sc(fn::act::scales_bytes(rows, k) / 4);
    std::memcpy(sc.data(), p + fn::act::codes_bytes(rows, k), sc.size() * 4);
    for (size_t i = 0; i < size_t(n) * size_t(k); ++i) {
        out[i] = static_cast<double>(codes[i]) * static_cast<double>(sc[2 * (i / 32)]);
    }
}

// the kernel's distinct-expert order and slots, on the CPU
struct cpu_meta {
    int32_t n_u = 0, n_slots = 0;
    std::vector<int32_t> e, mask, slot_base, u_of;
    int32_t slot(int32_t u, int32_t b) const {
        int32_t s = slot_base[size_t(u)];
        for (int32_t bb = 0; bb < b; ++bb) {
            s += (mask[size_t(u)] >> bb) & 1;
        }
        return s;
    }
    // the slot of row b's k-th routed expert
    int32_t slot_of_route(int32_t b, int32_t k) const { return slot(u_of[size_t(b) * NU + k], b); }
};

cpu_meta make_meta(const std::vector<int32_t> & ids, int32_t rows) {
    cpu_meta m;
    for (int32_t p = 0; p < rows * NU; ++p) {
        const int32_t id = ids[size_t(p)];
        int32_t u = -1;
        for (int32_t i = 0; i < m.n_u; ++i) {
            if (m.e[size_t(i)] == id) {
                u = i;
            }
        }
        if (u < 0) {
            u = m.n_u++;
            m.e.push_back(id);
            m.mask.push_back(0);
        }
        m.mask[size_t(u)] |= 1 << (p / NU);
        m.u_of.push_back(u);
    }
    for (int32_t u = 0; u < m.n_u; ++u) {
        m.slot_base.push_back(m.n_slots);
        for (int32_t b = 0; b < rows; ++b) {
            m.n_slots += (m.mask[size_t(u)] >> b) & 1;
        }
    }
    return m;
}

struct err_acc {
    double max_err = 0.0, max_ref = 0.0;
    void add(double got, double ref) {
        max_err = std::max(max_err, std::fabs(got - ref));
        max_ref = std::max(max_ref, std::fabs(ref));
    }
    double rel() const { return max_ref > 0.0 ? max_err / max_ref : max_err; }
};

fn::quant::wfmt dense_quant(fn::dense::fmt f) {
    using fn::quant::wfmt;
    switch (f) {
        case fn::dense::fmt::q4_k:
            return wfmt::q4_k;
        case fn::dense::fmt::q5_k:
            return wfmt::q5_k;
        case fn::dense::fmt::q6_k:
            return wfmt::q6_k;
        case fn::dense::fmt::iq4_xs:
            return wfmt::iq4_xs;
        case fn::dense::fmt::iq3_s:
            return wfmt::iq3_s;
        case fn::dense::fmt::q8_0:
            return wfmt::q8_0;
        case fn::dense::fmt::iq4_nl:
            return wfmt::iq4_nl;
        default:
            return wfmt::q2_0;
    }
}

// The shared expert's synthetic weights in one format (DP2's rows, fn::dense::make_synth), on device per card: gate
// and up formats as DP2's packed 320 x 2560 matrices, down formats as pack_down's expert. Card r's matrix row i is
// synthetic row (i + off[r]) % n_synth_rows (gate/up: its 320 rows; down: its 2,560 rows of the card's 320 columns).
struct shared_w {
    fn::dense::synth s;
    int32_t off[2] = {0, 0};
    std::unique_ptr<dev_mem> d[2];
    const uint8_t * row(int32_t card, int32_t i) const { return s.row((i + off[card]) % fn::dense::n_synth_rows); }
};

bool make_shared(fn::dense::fmt f, bool down_fmt, int32_t salt, shared_w & out) {
    if (!fn::dense::make_synth(f, down_fmt ? n_ff_card : D, out.s)) {
        return false;
    }
    for (int32_t card = 0; card < 2; ++card) {
        out.off[card] = (211 * salt + 389 * card + 17) % fn::dense::n_synth_rows;
        std::vector<uint8_t> h;
        if (down_fmt) {
            std::vector<uint8_t> rows(out.s.row_bytes * D);
            for (int32_t n = 0; n < D; ++n) {
                std::memcpy(rows.data() + out.s.row_bytes * n, out.row(card, n), out.s.row_bytes);
            }
            const fmt ef = f == fn::dense::fmt::q2_0 ? fmt::q2_0 : f == fn::dense::fmt::q8_0 ? fmt::q8_0 : fmt::iq4_nl;
            h.assign(size_t(down_bytes(ef)), 0);
            if (!pack_down(ef, rows.data(), out.s.row_bytes, h.data())) {
                return false;
            }
        } else if (!fn::dense::pack_synth(out.s, n_ff_card, out.off[card], fn::dense::n_synth_rows, h)) {
            return false;
        }
        out.d[card] = std::make_unique<dev_mem>(h.size());
        if (!out.d[card]->put(h.data(), h.size())) {
            return false;
        }
    }
    return true;
}

struct test_ctx {
    std::vector<uint8_t> grids;
    dev_mem d_grids{grid_bytes};
    std::vector<int32_t> map;        // expert ID -> synthetic expert
    std::vector<int32_t> pool;       // the routed IDs
    std::vector<int32_t> pool_x;     // XB2: more routed IDs, each reading a pool ID's synthetic expert (the turn cases)
    std::vector<int32_t> pool_y;     // SP4: 16 more such IDs, so a five-row case routes 50 distinct experts
    std::map<int32_t, synth> routed; // rkey(routed format, down) -> synthetic experts
    std::map<int32_t, shared_w> sh_gu; // dense format -> the shared gate/up matrices
    std::map<int32_t, shared_w> sh_dn; // dense format (q2_0, iq4_nl, q8_0) -> the shared down
    std::map<int32_t, std::vector<uint8_t>> packed_gu, packed_dn; // routed format * 2 + card -> packed experts
    std::map<int32_t, std::unique_ptr<dev_mem>> dev_gu, dev_dn;   // XB2: the same on device, uploaded once
    bool ok = true;
    int32_t n_checks = 0;
};

bool check(test_ctx & c, bool pass, const std::string & what) {
    ++c.n_checks;
    if (!pass) {
        fprintf(stderr, "ex-test: FAIL %s\n", what.c_str());
        c.ok = false;
    }
    return pass;
}

// the routed synth map's key: Q8_0 is a gate/up and a down format (DR1)
int32_t rkey(fmt f, bool down) {
    return int32_t(f) * 2 + (down ? 1 : 0);
}

const std::vector<uint8_t> & packed(test_ctx & c, fmt f, bool down, int32_t card) {
    std::map<int32_t, std::vector<uint8_t>> & cache = down ? c.packed_dn : c.packed_gu;
    const int32_t key = int32_t(f) * 2 + card;
    if (cache.count(key) == 0 && !pack_card(c.routed.at(rkey(f, down)), card, c.map, 1, cache[key])) {
        fprintf(stderr, "ex-test: pack %s card %d failed\n", fmt_name(f), card);
        cache[key].clear();
    }
    return cache[key];
}

// XB2: packed's buffer on device, uploaded at its first use (every launch of a format and card reads the same bytes)
const dev_mem * dev_packed(test_ctx & c, fmt f, bool down, int32_t card) {
    std::map<int32_t, std::unique_ptr<dev_mem>> & cache = down ? c.dev_dn : c.dev_gu;
    const int32_t key = int32_t(f) * 2 + card;
    if (cache.count(key) == 0) {
        const std::vector<uint8_t> & pk = packed(c, f, down, card);
        auto d = std::make_unique<dev_mem>(pk.size());
        if (pk.empty() || !d->put(pk.data(), pk.size())) {
            return nullptr;
        }
        cache[key] = std::move(d);
    }
    return cache[key].get();
}

// One case: rows B of a 4-row master case (row b's IDs and activations are the master's), so a row's results at
// different B can be compared bitwise
struct row_case {
    int32_t rows = 0;
    std::vector<int32_t> ids;
    std::vector<float> wts;
    std::vector<float> sg;
    std::vector<uint8_t> xq; // FNFL's format, act_bytes(rows, D)
    std::vector<double> xd;
    cpu_meta m;
    bool turn = false; // XB2 Amendment 1: a turn_case, whose gate/up order takes turn_w
};

// XB2 (d): the turn cases' class table: the shared class between the 1-row class and the others (EL1's IQ2_XXS
// routed weights), so a turn_case's shared units rank 70-79
constexpr class_table turn_w = {{1.0f, 1.15f, 1.43f, 1.71f, 1.05f}};
// down reads no order: its meta's table
const class_table dn_w = class_weights(fmt::iq2_xxs, fn::dense::fmt::q4_k, fn::dense::fmt::q4_k);
// the class table of a case's gate/up launch: the formats' (as the model's layer passes it), or turn_w
class_table case_w(const row_case & rc, fmt f, fn::dense::fmt fg, fn::dense::fmt fu) {
    return rc.turn ? turn_w : class_weights(f, fg, fu);
}

row_case make_master(test_ctx & c, std::mt19937 & rng, bool zero_x) {
    row_case rc;
    rc.rows = max_rows;
    std::uniform_real_distribution<float> ud(0.05f, 1.0f);
    for (int32_t b = 0; b < max_rows; ++b) {
        std::vector<int32_t> p = c.pool;
        std::shuffle(p.begin(), p.end(), rng); // 10 distinct of the pool a row, so rows repeat experts
        float sum = 0.0f;
        std::vector<float> w(NU);
        for (int32_t k = 0; k < NU; ++k) {
            rc.ids.push_back(p[size_t(k)]);
            w[size_t(k)] = ud(rng);
            sum += w[size_t(k)];
        }
        for (int32_t k = 0; k < NU; ++k) {
            rc.wts.push_back(w[size_t(k)] / sum);
        }
        rc.sg.push_back(ud(rng));
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(size_t(max_rows) * D);
    for (auto & v : x) {
        v = zero_x ? 0.0f : nd(rng);
    }
    rc.xq.assign(act_bytes(max_rows, D), 0);
    act_quantize(x.data(), max_rows, D, rc.xq.data());
    return rc;
}

// rows 0 .. B - 1 of the master (the activation format is per row: codes [B][2560], then scales [B][80])
row_case prefix(const row_case & mc, int32_t B) {
    row_case rc;
    rc.rows = B;
    rc.ids.assign(mc.ids.begin(), mc.ids.begin() + std::ptrdiff_t(B) * NU);
    rc.wts.assign(mc.wts.begin(), mc.wts.begin() + std::ptrdiff_t(B) * NU);
    rc.sg.assign(mc.sg.begin(), mc.sg.begin() + B);
    rc.xq.assign(act_bytes(B, D), 0);
    std::memcpy(rc.xq.data(), mc.xq.data(), size_t(B) * D);
    std::memcpy(rc.xq.data() + fn::act::codes_bytes(B, D), mc.xq.data() + fn::act::codes_bytes(max_rows, D),
                fn::act::scales_bytes(B, D));
    rc.xd.assign(size_t(B) * D, 0.0);
    act_values(rc.xq.data(), B, D, B, rc.xd.data());
    rc.m = make_meta(rc.ids, B);
    return rc;
}

// XB2 Amendment 1: rows B of the master with routing replaced: n_many experts in each of rows 0 .. many_rows - 1,
// every other entry its own expert (so the class of many_rows-row experts holds n_many, the 1-row class the rest:
// U = n_many + B * 10 - n_many * many_rows), positions shuffled within a row. With n_many = 7, when the shared weight
// lies between the two classes' weights, the shared expert's 10 units rank 70-79 in the row-weighted order: across
// the snake's turn at 72.
row_case turn_case(const test_ctx & c, const row_case & master, int32_t B, int32_t n_many, int32_t many_rows,
                   std::mt19937 & rng) {
    row_case rc = prefix(master, B);
    std::vector<int32_t> fresh = c.pool;
    fresh.insert(fresh.end(), c.pool_x.begin(), c.pool_x.end());
    std::shuffle(fresh.begin(), fresh.end(), rng);
    size_t next = size_t(n_many);
    for (int32_t b = 0; b < B; ++b) {
        std::vector<int32_t> row;
        for (int32_t k = 0; k < NU; ++k) {
            row.push_back(b < many_rows && k < n_many ? fresh[size_t(k)] : fresh[next++]);
        }
        std::shuffle(row.begin(), row.end(), rng);
        std::copy(row.begin(), row.end(), rc.ids.begin() + std::ptrdiff_t(b) * NU);
    }
    rc.m = make_meta(rc.ids, B);
    rc.turn = true;
    return rc;
}

// XB4: rows B of the master with routing replaced by U distinct experts (10 <= U <= 10 B): row-major, each expert
// once, then each row filled with experts it lacks, positions shuffled within a row (so n_u = U, any residue mod 3)
row_case u_case(const test_ctx & c, const row_case & master, int32_t B, int32_t U, std::mt19937 & rng) {
    row_case rc = prefix(master, B);
    std::vector<int32_t> fresh = c.pool;
    fresh.insert(fresh.end(), c.pool_x.begin(), c.pool_x.end());
    std::shuffle(fresh.begin(), fresh.end(), rng);
    fresh.resize(size_t(U));
    int32_t next = 0;
    for (int32_t b = 0; b < B; ++b) {
        std::vector<int32_t> row;
        for (int32_t k = 0; k < NU && next < U; ++k) {
            row.push_back(fresh[size_t(next++)]);
        }
        for (size_t i = 0; row.size() < size_t(NU); ++i) {
            if (std::find(row.begin(), row.end(), fresh[i]) == row.end()) {
                row.push_back(fresh[i]);
            }
        }
        std::shuffle(row.begin(), row.end(), rng);
        std::copy(row.begin(), row.end(), rc.ids.begin() + std::ptrdiff_t(b) * NU);
    }
    rc.m = make_meta(rc.ids, B);
    return rc;
}

// SP4: rows B of the master with routing replaced by one expert per row mask in masks (bit b: row b routes it), the
// masks holding exactly 10 experts a row; distinct IDs from the pools (at most 64), positions shuffled within a row
row_case mask_case(const test_ctx & c, const row_case & master, int32_t B, const std::vector<uint32_t> & masks,
                   std::mt19937 & rng) {
    row_case rc = prefix(master, B);
    std::vector<int32_t> fresh = c.pool;
    fresh.insert(fresh.end(), c.pool_x.begin(), c.pool_x.end());
    fresh.insert(fresh.end(), c.pool_y.begin(), c.pool_y.end());
    std::shuffle(fresh.begin(), fresh.end(), rng);
    for (int32_t b = 0; b < B; ++b) {
        std::vector<int32_t> row;
        for (size_t e = 0; e < masks.size(); ++e) {
            if ((masks[e] >> b) & 1u) {
                row.push_back(fresh[e]);
            }
        }
        std::shuffle(row.begin(), row.end(), rng);
        std::copy(row.begin(), row.end(), rc.ids.begin() + std::ptrdiff_t(b) * NU);
    }
    rc.m = make_meta(rc.ids, B);
    return rc;
}

// SP4: every nonempty row mask of B rows, as few mask_cases as the greedy packing (widest masks first) gives, each row
// filled to 10 with one-row experts; then all rows routing the same 10 experts (U 10), then 10 B one-row experts
std::vector<row_case> sp4_route_cases(const test_ctx & c, const row_case & master, int32_t B, std::mt19937 & rng) {
    std::vector<uint32_t> left;
    for (uint32_t m = 1; m < (1u << B); ++m) {
        left.push_back(m);
    }
    std::stable_sort(left.begin(), left.end(),
                     [](uint32_t x, uint32_t y) { return __builtin_popcount(x) > __builtin_popcount(y); });
    std::vector<row_case> out;
    while (!left.empty()) {
        std::vector<uint32_t> ms;
        int32_t n[max_rows] = {};
        for (auto it = left.begin(); it != left.end();) {
            bool fits = true;
            for (int32_t b = 0; b < B; ++b) {
                fits = fits && (((*it >> b) & 1u) == 0 || n[b] < NU);
            }
            if (!fits) {
                ++it;
                continue;
            }
            for (int32_t b = 0; b < B; ++b) {
                n[b] += (*it >> b) & 1u;
            }
            ms.push_back(*it);
            it = left.erase(it);
        }
        for (int32_t b = 0; b < B; ++b) {
            for (; n[b] < NU; ++n[b]) {
                ms.push_back(1u << b);
            }
        }
        out.push_back(mask_case(c, master, B, ms, rng));
    }
    out.push_back(mask_case(c, master, B, std::vector<uint32_t>(NU, (1u << B) - 1), rng));
    std::vector<uint32_t> disjoint;
    for (int32_t b = 0; b < B; ++b) {
        disjoint.insert(disjoint.end(), NU, 1u << b);
    }
    out.push_back(mask_case(c, master, B, disjoint, rng));
    return out;
}

// SP4: rows B of the master with activations zero except the listed runs of 32 (run r: row r / 80, values 32 (r %
// 80) ..), there standard normal, quantized as the master's
row_case spot_case(const row_case & master, int32_t B, const std::vector<int32_t> & runs, std::mt19937 & rng) {
    row_case rc = prefix(master, B);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(size_t(B) * D, 0.0f);
    for (const int32_t r : runs) {
        for (int32_t i = 0; i < 32; ++i) {
            x[size_t(r) * 32 + size_t(i)] = nd(rng);
        }
    }
    act_quantize(x.data(), B, D, rc.xq.data());
    act_values(rc.xq.data(), B, D, B, rc.xd.data());
    return rc;
}

constexpr size_t glu_buf_bytes =
    fn::act::codes_bytes(glu_slots, n_ff_card) + fn::act::scales_bytes(glu_slots, n_ff_card);
constexpr size_t glu_sc = fn::act::codes_bytes(glu_slots, n_ff_card); // the scales' offset in a GLU buffer

struct gu_out {
    std::vector<uint8_t> glu[2];   // per card: 44 slots of 320 in FNFL's format
    std::vector<float> glu_f32[2]; // per card: [44][320]
};

// XB2: the routing meta of the case's IDs (ids on the device) and its unit order by cw, by write_meta, into meta
// (meta_bytes)
bool device_meta(const dev_mem & ids, int32_t rows, const class_table & cw, const dev_mem & meta) {
    if (ids.p == nullptr || meta.p == nullptr ||
        !write_meta(rows, ids.as<int32_t>(), cw, meta.p, cudaStreamPerThread)) {
        fprintf(stderr, "ex-test: write_meta not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "write_meta");
}

// XB2: a meta region's words (w, meta_bytes / 4) against make_meta's of the case's IDs, over every specified word, and
// its unit order by cw against unit_order's, every entry (experts.h). XB3: the region write_meta writes, or the one
// gate_up's blocks write (the struct by block 0, each block its ranks' order entries)
// the region's struct words (n_u, n_slots, bad, the live e and slot_base, every mask, the 10 B u_of) against
// make_meta's; bad: the case holds an ID outside [0, 512)
bool struct_matches(const std::vector<int32_t> & w, const row_case & rc, bool & bad, bool & bad_out) {
    const cpu_meta & m = rc.m;
    bad = false;
    for (const int32_t id : rc.ids) {
        bad = bad || id < 0 || id >= NE;
    }
    const meta_view v = meta_view_of(rc.rows); // SP2: the launch's capacity (S = 40 for B <= 4, 50 at B = 5)
    bad_out = w[size_t(v.bad)] != 0;
    bool same = w[size_t(v.n_u)] == m.n_u && w[size_t(v.n_slots)] == m.n_slots && w[size_t(v.bad)] == (bad ? 1 : 0);
    for (int32_t u = 0; u < v.slots; ++u) {
        const bool live = u < m.n_u;
        same = same && w[size_t(v.mask) + size_t(u)] == (live ? m.mask[size_t(u)] : 0);
        same = same && (!live || (w[size_t(v.e) + size_t(u)] == m.e[size_t(u)] &&
                                  w[size_t(v.slot_base) + size_t(u)] == m.slot_base[size_t(u)]));
    }
    for (int32_t p = 0; p < rc.rows * NU; ++p) {
        same = same && w[size_t(v.u_of) + size_t(p)] == m.u_of[size_t(p)];
    }
    return same;
}

bool region_matches(const std::vector<int32_t> & w, const row_case & rc, const class_table & cw, bool & bad_out) {
    const cpu_meta & m = rc.m;
    const meta_view v = meta_view_of(rc.rows);
    bool bad = false;
    bool same = struct_matches(w, rc, bad, bad_out);
    if (!bad) {
        std::vector<uint32_t> mask(size_t(m.n_u));
        std::copy(m.mask.begin(), m.mask.begin() + m.n_u, mask.begin());
        const int32_t units = max_units_of(cap_of(rc.rows));
        std::vector<uint16_t> want(static_cast<size_t>(units)), got(static_cast<size_t>(units));
        unit_order(mask.data(), m.n_u, cw, want.data(), rc.rows);
        std::memcpy(got.data(), reinterpret_cast<const uint8_t *>(w.data()) + v.order, got.size() * 2);
        same = same && std::equal(want.begin(), want.begin() + (m.n_u * n_group + n_group), got.begin());
    }
    return same;
}

// R18 repair: one pass of a two-pass cache launch (gate_up_args::wt with skip 1 or 2), the meta both of its forms
// write. The struct is the whole launch's (build_meta_of reads neither wt nor skip), so struct_matches applies
// unchanged. The order is the pass's (experts_gu.cuh block_units): only an expert u whose table entry wt[e[u]] is
// non-null takes a class, keeping its index (units 10 u + g), and the shared class is in it at skip 1 only. Its
// reference is unit_order over the masks with the other experts' masks zeroed (no class), the shared units dropped at
// skip 2: n_order entries.
int32_t pass_order(const row_case & rc, const class_table & cw, const std::vector<bool> & runs, bool shared,
                   std::vector<uint16_t> & want) {
    const cpu_meta & m = rc.m;
    std::vector<uint32_t> mask(size_t(m.n_u), 0u);
    int32_t n_run = 0;
    for (int32_t u = 0; u < m.n_u; ++u) {
        if (runs[size_t(u)]) {
            mask[size_t(u)] = m.mask[size_t(u)];
            ++n_run;
        }
    }
    std::vector<uint16_t> all(static_cast<size_t>(max_units_of(cap_of(rc.rows))));
    unit_order(mask.data(), m.n_u, cw, all.data(), rc.rows);
    const int32_t n_order = n_group * n_run + (shared ? n_group : 0);
    want.clear();
    for (int32_t r = 0; r < n_run * n_group + n_group; ++r) { // unit_order's entries: the running units and shared
        if (shared || all[size_t(r)] < order_shared_of(cap_of(rc.rows))) {
            want.push_back(all[size_t(r)]);
        }
    }
    return want.size() == size_t(n_order) ? n_order : -1;
}

bool pass_region_matches(const std::vector<int32_t> & w, const row_case & rc, const class_table & cw,
                         const std::vector<bool> & runs, bool shared) {
    bool bad = false, bad_out = false;
    if (!struct_matches(w, rc, bad, bad_out) || bad) {
        return false;
    }
    std::vector<uint16_t> want;
    const int32_t n = pass_order(rc, cw, runs, shared, want);
    if (n < 0) {
        return false;
    }
    std::vector<uint16_t> got(static_cast<size_t>(n));
    std::memcpy(got.data(), reinterpret_cast<const uint8_t *>(w.data()) + meta_view_of(rc.rows).order, got.size() * 2);
    return got == want;
}

// XB2: write_meta's region of the case's IDs against the host's (region_matches)
bool meta_matches(const row_case & rc, const class_table & cw, bool & bad_out) {
    dev_mem ids(rc.ids.size() * 4), meta(meta_bytes);
    std::vector<int32_t> w(meta_bytes / 4);
    // ME3 (initcheck): the buffer filled with 0xFF first, so reading it back whole reads no uninitialized bytes; the
    // region compared is still the kernel's (a field it fails to write keeps the fill and differs)
    if (!ids.put(rc.ids.data(), ids.n) || meta.p == nullptr ||
        !cuda_ok(cudaMemset(meta.p, 0xFF, meta.n), "meta fill") || !device_meta(ids, rc.rows, cw, meta) ||
        !meta.get(w.data(), meta.n)) {
        return false;
    }
    return region_matches(w, rc, cw, bad_out);
}

// XB2 Amendment 1: the first rank of the shared class in unit_order's order by cw, and whether the class straddles
// a snake turn (a multiple of 72 inside it: some blocks then hold two shared units)
int32_t shared_start(const row_case & rc, const class_table & cw, bool & straddles) {
    std::vector<uint32_t> mask(size_t(rc.m.n_u));
    std::copy(rc.m.mask.begin(), rc.m.mask.begin() + rc.m.n_u, mask.begin());
    std::vector<uint16_t> ord(size_t(max_units_of(cap_of(rc.rows))));
    unit_order(mask.data(), rc.m.n_u, cw, ord.data(), rc.rows);
    const auto it = std::find(ord.begin(), ord.end(), uint16_t(order_shared_of(cap_of(rc.rows))));
    const int32_t s0 = static_cast<int32_t>(it - ord.begin());
    straddles = s0 / n_sm != (s0 + n_group - 1) / n_sm;
    return s0;
}

// SP4: the candidate mapping the SP4 checks below run (gate_up_args::map): XB3 C1's 3, H2's IQ2 body at B = 4 and 5
// for the IQ2 formats and the IQ3 body for the IQ3 formats, both with the compact optional-row loop. H1's paired
// mapping (1) passed these checks in H1's hold and H2's (2) in H2's; they stay on the branch, not run.
constexpr int32_t sp4_map = 3;
const char * map_tag(int32_t map) {
    return map == 1 ? "sp4" : map == 2 ? "h2" : map == 3 ? "xb3" : "640";
}
bool is_iq2(fmt f) {
    return f == fmt::iq2_xxs || f == fmt::iq2_xs || f == fmt::iq2_s;
}
// the instances map changes at B 4 and 5: maps 1 and 3 every target format, map 2 the IQ2 formats (the IQ3 formats
// and Q8_0 run SP2's kernels under it)
bool map_covers(int32_t map, fmt f) { // ME1: the Q4 candidate's K-quants take gate_up_kernel under every map
    return map == 1 || map == 3 ? f != fmt::q8_0 && f != fmt::q2_0 && f != fmt::q4_k && f != fmt::q5_k
                                : map == 2 && is_iq2(f); // ME7: Q2_0 (the draft's) as Q8_0
}

// SP4: a launch's gate/up mapping (gate_up_args::map) and, when hmix is set, its X-HMIX slot (x is read from it, the
// peer half 1 - card polled; err gets the card's error word, zeroed before each run)
struct gu_opt {
    int32_t map = 0;
    const float * hmix = nullptr;
    uint32_t * err = nullptr;
    const uint8_t * const * wt = nullptr; // ME1: the routed experts' address table (device); tag names the dumps
    const char * tag = nullptr;
};

// gate_up of one card twice; out gets the first run; false if the runs differ or a launch fails
bool run_gate_up(test_ctx & c, fmt f, fn::dense::fmt fg, fn::dense::fmt fu, int32_t card, const row_case & rc,
                 std::vector<uint8_t> & glu, std::vector<float> & glu_f32, uint32_t & status_out,
                 const gu_opt & o = {}) {
    const dev_mem * w = dev_packed(c, f, false, card);
    dev_mem x(rc.xq.size()), ids(rc.ids.size() * 4), g(glu_buf_bytes), gf(size_t(glu_slots) * n_ff_card * 4), st(4),
        meta(meta_bytes);
    if (w == nullptr || !x.put(rc.xq.data(), rc.xq.size()) || !ids.put(rc.ids.data(), ids.n)) {
        return false;
    }
    std::vector<uint8_t> glu2(g.n), gfb(gf.n), gf2b(gf.n);
    for (int32_t run = 0; run < 2; ++run) {
        // fill the outputs with a pattern so an unwritten byte shows
        if (!cuda_ok(cudaMemset(g.p, 0xa5, g.n), "memset") || !cuda_ok(cudaMemset(gf.p, 0xa5, gf.n), "memset") ||
            !cuda_ok(cudaMemset(st.p, 0, 4), "memset") || !cuda_ok(cudaMemset(meta.p, 0xa5, meta.n), "memset") ||
            (o.err != nullptr && !cuda_ok(cudaMemset(o.err, 0, 4), "memset"))) {
            return false;
        }
        gate_up_args a = {w->p,
                          c.sh_gu.at(int32_t(fg)).d[card]->p,
                          c.sh_gu.at(int32_t(fu)).d[card]->p,
                          c.d_grids.p,
                          act_in(x.p, rc.rows, D),
                          ids.as<int32_t>(),
                          meta.p,
                          act_out(g.p, glu_slots, n_ff_card),
                          gf.as<float>(),
                          st.as<uint32_t>()};
        a.cw = case_w(rc, f, fg, fu);
        a.map = o.map;
        a.wt = o.wt;
        if (o.hmix != nullptr) {
            a.x = {nullptr, nullptr, rc.rows, D};
            a.hmix = o.hmix;
            a.hmix_peer_half = 1 - card;
            a.err = o.err;
        }
        if (!gate_up(f, fg, fu, rc.rows, a, cudaStreamPerThread)) {
            fprintf(stderr, "ex-test: gate_up not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        if (!cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "gate_up")) {
            return false;
        }
        if (run == 0) {
            glu.assign(g.n, 0);
            if (!g.get(glu.data(), g.n) || !gf.get(gfb.data(), gf.n) || !st.get(&status_out, 4)) {
                return false;
            }
            // XB3 part 1: the meta region the launch wrote (the struct by block 0, the order by every block)
            std::vector<int32_t> mw(meta_bytes / 4);
            bool bad = false;
            if (!meta.get(mw.data(), meta.n) || !region_matches(mw, rc, a.cw, bad)) {
                fprintf(stderr, "ex-test: %s B %d: gate_up's routing meta region differs from the host's\n",
                        fmt_name(f), rc.rows);
                return false;
            }
            const std::string t = std::string("ex-gu-") + fmt_name(f) + "-" + fn::dense::fmt_name(fg) + "-" +
                                  fn::dense::fmt_name(fu) + "-card" + std::to_string(card) + "-B" +
                                  std::to_string(rc.rows) + (o.map != 0 ? std::string("-") + map_tag(o.map) : "") +
                                  (o.hmix != nullptr ? "-hmix" : "") +
                                  (o.tag != nullptr ? std::string("-") + o.tag : "");
            if (!fn::tools::test_dump(t + "-glu", glu.data(), glu.size()) ||
                !fn::tools::test_dump(t + "-f32", gfb.data(), gfb.size()) ||
                !fn::tools::test_dump(t + "-status", &status_out, 4)) {
                return false;
            }
        } else if (!g.get(glu2.data(), g.n) || !gf.get(gf2b.data(), gf.n)) {
            return false;
        }
    }
    glu_f32.assign(size_t(glu_slots) * n_ff_card, 0.0f);
    std::memcpy(glu_f32.data(), gfb.data(), gfb.size());
    return glu == glu2 && gfb == gf2b;
}

// ME1: a down launch's address table and CPU slots (down_args::wt, ext_slots); tag names the dumps
struct dn_opt {
    const uint8_t * const * wt = nullptr;
    uint64_t ext_slots = 0;
    const char * tag = nullptr;
    const uint64_t * ext_slots_d = nullptr; // ME1 (GPU-side resolution): the slot mask from device memory
};

bool run_down(test_ctx & c, fmt f, fmt fs, int32_t card, const row_case & rc, const std::vector<uint8_t> & glu,
              std::vector<float> & out, uint32_t & status_out, const dn_opt & opt = {}) {
    const dev_mem * w = dev_packed(c, f, true, card);
    const fn::dense::fmt fsd = dense_of(fs);
    dev_mem g(glu.size()), ids(rc.ids.size() * 4), wts(rc.wts.size() * 4), sg(rc.sg.size() * 4),
        o(size_t(rc.rows) * D * 4), st(4), meta(meta_bytes);
    if (w == nullptr || !g.put(glu.data(), glu.size()) || !ids.put(rc.ids.data(), ids.n) ||
        !wts.put(rc.wts.data(), wts.n) || !sg.put(rc.sg.data(), sg.n) || !device_meta(ids, rc.rows, dn_w, meta)) {
        return false;
    }
    std::vector<float> o2(size_t(rc.rows) * D);
    out.assign(size_t(rc.rows) * D, 0.0f);
    for (int32_t run = 0; run < 2; ++run) {
        if (!cuda_ok(cudaMemset(o.p, 0xa5, o.n), "memset") || !cuda_ok(cudaMemset(st.p, 0, 4), "memset")) {
            return false;
        }
        down_args a = {w->p,
                       c.sh_dn.at(int32_t(fsd)).d[card]->p,
                       c.d_grids.p,
                       act_in(g.p, glu_slots, n_ff_card),
                       ids.as<int32_t>(),
                       meta.p,
                       wts.as<float>(),
                       sg.as<float>(),
                       o.as<float>(),
                       st.as<uint32_t>()};
        a.wt = opt.wt;
        a.ext_slots = opt.ext_slots;
        a.ext_slots_d = opt.ext_slots_d;
        if (!down(f, fs, rc.rows, a, cudaStreamPerThread)) {
            fprintf(stderr, "ex-test: down not enqueued: %s\n", cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        if (!cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "down")) {
            return false;
        }
        if (!o.get(run == 0 ? out.data() : o2.data(), o.n) || (run == 0 && !st.get(&status_out, 4))) {
            return false;
        }
        if (run == 0) {
            const std::string t = std::string("ex-dn-") + fmt_name(f) + "-" + fmt_name(fs) + "-card" +
                                  std::to_string(card) + "-B" + std::to_string(rc.rows) +
                                  (opt.tag != nullptr ? std::string("-") + opt.tag : "");
            if (!fn::tools::test_dump(t + "-out", out.data(), o.n) ||
                !fn::tools::test_dump(t + "-status", &status_out, 4)) {
                return false;
            }
        }
    }
    return std::memcmp(out.data(), o2.data(), o.n) == 0;
}

// FP64 GLU reference of the routed pairs, both cards: ref[card][slot * 320 + c]
void routed_glu_reference(const synth & s, const test_ctx & c, const row_case & rc, std::vector<double> (&ref)[2]) {
    const fn::quant::wfmt q = to_quant(s.f);
    const cpu_meta & m = rc.m;
    for (auto & r : ref) {
        r.assign(size_t(glu_slots) * n_ff_card, 0.0);
    }
    parallel_for(m.n_u * 2, [&](int32_t i) {
        const int32_t u = i / 2, card = i % 2;
        const uint8_t * e = s.expert(c.map[size_t(m.e[size_t(u)])]);
        std::vector<float> wg(D), wu(D);
        for (int32_t ch = 0; ch < n_ff_card; ++ch) {
            const int32_t row = card * n_ff_card + ch;
            fn::quant::decode_row(q, e + size_t(row) * s.row_bytes, wg.data(), D);
            fn::quant::decode_row(q, e + size_t(NFF + row) * s.row_bytes, wu.data(), D);
            for (int32_t b = 0; b < rc.rows; ++b) {
                if (((m.mask[size_t(u)] >> b) & 1) == 0) {
                    continue;
                }
                double g = 0.0, v = 0.0;
                for (int32_t k = 0; k < D; ++k) {
                    g += static_cast<double>(wg[size_t(k)]) * rc.xd[size_t(b) * D + k];
                    v += static_cast<double>(wu[size_t(k)]) * rc.xd[size_t(b) * D + k];
                }
                ref[card][size_t(m.slot(u, b)) * n_ff_card + ch] = g / (1.0 + std::exp(-g)) * v;
            }
        }
    });
}

// FP64 GLU reference of the shared expert's rows: ref[card][b * 320 + c]
void shared_glu_reference(const shared_w & sgw, const shared_w & suw, const row_case & rc,
                          std::vector<double> (&ref)[2]) {
    for (auto & r : ref) {
        r.assign(size_t(rc.rows) * n_ff_card, 0.0);
    }
    parallel_for(2 * n_ff_card, [&](int32_t i) {
        const int32_t card = i / n_ff_card, ch = i % n_ff_card;
        std::vector<float> wg(D), wu(D);
        fn::quant::decode_row(dense_quant(sgw.s.f), sgw.row(card, ch), wg.data(), D);
        fn::quant::decode_row(dense_quant(suw.s.f), suw.row(card, ch), wu.data(), D);
        for (int32_t b = 0; b < rc.rows; ++b) {
            double g = 0.0, v = 0.0;
            for (int32_t k = 0; k < D; ++k) {
                g += static_cast<double>(wg[size_t(k)]) * rc.xd[size_t(b) * D + k];
                v += static_cast<double>(wu[size_t(k)]) * rc.xd[size_t(b) * D + k];
            }
            ref[card][size_t(b) * n_ff_card + ch] = g / (1.0 + std::exp(-g)) * v;
        }
    });
}

// FP64 down reference of one card from the kernel's own GLU codes and scales: the routed weighted sum, then
// sg[b] x the shared expert's product
void down_reference(const synth & s, const shared_w & sdw, const test_ctx & c, const row_case & rc, int32_t card,
                    const std::vector<uint8_t> & glu, std::vector<double> & ref) {
    const fn::quant::wfmt q = to_quant(s.f);
    const cpu_meta & m = rc.m;
    const int32_t n_sl = m.n_slots + rc.rows;
    std::vector<double> gv(size_t(n_sl) * n_ff_card);
    act_values(glu.data(), glu_slots, n_ff_card, n_sl, gv.data());
    std::vector<double> dot(size_t(n_sl) * D, 0.0); // per slot and output
    parallel_for(m.n_u + 1, [&](int32_t u) {
        std::vector<float> wrow(NFF);
        for (int32_t n = 0; n < D; ++n) {
            if (u == m.n_u) { // the shared expert: every row, slots n_slots + b
                fn::quant::decode_row(dense_quant(sdw.s.f), sdw.row(card, n), wrow.data(), n_ff_card);
                for (int32_t b = 0; b < rc.rows; ++b) {
                    double acc = 0.0;
                    for (int32_t ch = 0; ch < n_ff_card; ++ch) {
                        acc += static_cast<double>(wrow[size_t(ch)]) * gv[size_t(m.n_slots + b) * n_ff_card + ch];
                    }
                    dot[size_t(m.n_slots + b) * D + n] = acc;
                }
                continue;
            }
            const uint8_t * e = s.expert(c.map[size_t(m.e[size_t(u)])]);
            fn::quant::decode_row(q, e + size_t(n) * s.row_bytes, wrow.data(), NFF);
            for (int32_t b = 0; b < rc.rows; ++b) {
                if (((m.mask[size_t(u)] >> b) & 1) == 0) {
                    continue;
                }
                const int32_t sl = m.slot(u, b);
                double acc = 0.0;
                for (int32_t ch = 0; ch < n_ff_card; ++ch) {
                    acc += static_cast<double>(wrow[int64_t(card) * n_ff_card + ch]) * gv[size_t(sl) * n_ff_card + ch];
                }
                dot[size_t(sl) * D + n] = acc;
            }
        }
    });
    ref.assign(size_t(rc.rows) * D, 0.0);
    for (int32_t b = 0; b < rc.rows; ++b) {
        for (int32_t n = 0; n < D; ++n) {
            double acc = 0.0;
            for (int32_t k = 0; k < NU; ++k) {
                acc += static_cast<double>(rc.wts[size_t(b) * NU + k]) * dot[size_t(m.slot_of_route(b, k)) * D + n];
            }
            ref[size_t(b) * D + n] = acc + static_cast<double>(rc.sg[size_t(b)]) * dot[size_t(m.n_slots + b) * D + n];
        }
    }
}

std::string gu_name(fmt f, fn::dense::fmt fg, fn::dense::fmt fu) {
    return std::string(fmt_name(f)) + "+" + fn::dense::fmt_name(fg) + "/" + fn::dense::fmt_name(fu);
}

// one row's GLU of one card and one expert pair (or the shared expert) at a case: 320 codes, 10 {d, s} and 320 F32
std::vector<uint8_t> row_glu(const std::vector<uint8_t> & glu, const std::vector<float> & f32, int32_t slot) {
    std::vector<uint8_t> r(size_t(n_ff_card) + size_t(n_group) * 8 + size_t(n_ff_card) * 4);
    std::memcpy(r.data(), glu.data() + size_t(slot) * n_ff_card, n_ff_card);
    constexpr size_t sc_bytes = size_t(n_group) * 8;
    std::memcpy(r.data() + n_ff_card, glu.data() + glu_sc + size_t(slot) * sc_bytes, sc_bytes);
    std::memcpy(r.data() + n_ff_card + sc_bytes, f32.data() + size_t(slot) * n_ff_card, size_t(n_ff_card) * 4);
    return r;
}

// The GLU of one gate/up instantiation at one case, both cards: checks against FP64 and act_ref; zero input: all
// zero. Returns false when the launch did not run.
bool check_gate_up(test_ctx & c, fmt f, fn::dense::fmt fg, fn::dense::fmt fu, const row_case & rc, bool zero_x,
                   const char * label, gu_out & go, int32_t map = 0) {
    const std::string name = gu_name(f, fg, fu) + (map != 0 ? std::string(" ") + map_tag(map) : "");
    bool ran = true;
    if ((f == fmt::q8_0 || f == fmt::q2_0) && rc.rows == 5) { // SP2: the draft's layer runs B <= 4 (experts.h); ME7:
                                                              // its Q2_0 too; B = 5 is rejected
        uint32_t status = 0;
        check(c, !run_gate_up(c, f, fg, fu, 0, rc, go.glu[0], go.glu_f32[0], status),
              std::string(label) + " " + name + " B 5: rejected (the draft's FFN runs B <= 4)");
        return false;
    }
    for (int32_t card = 0; card < 2 && ran; ++card) {
        uint32_t status = 0;
        ran = run_gate_up(c, f, fg, fu, card, rc, go.glu[card], go.glu_f32[card], status, {map}) && status == 0;
    }
    if (!check(c, ran,
               std::string(label) + " " + name + " B " + std::to_string(rc.rows) +
                   ": gate_up runs (two, bitwise equal)")) {
        return false;
    }
    const int32_t n_sl = rc.m.n_slots + rc.rows;
    err_acc ge, gs;
    bool act_same = true, zero_ok = true;
    if (zero_x) {
        for (int32_t card = 0; card < 2; ++card) {
            for (int32_t i = 0; i < n_sl * n_ff_card; ++i) {
                zero_ok = zero_ok && go.glu_f32[card][size_t(i)] == 0.0f && go.glu[card][size_t(i)] == 0;
            }
            for (size_t i = 0; i < fn::act::scales_bytes(n_sl, n_ff_card); ++i) {
                zero_ok = zero_ok && go.glu[card][glu_sc + i] == 0;
            }
        }
    } else {
        std::vector<double> ref[2], sref[2];
        routed_glu_reference(c.routed.at(rkey(f, false)), c, rc, ref);
        shared_glu_reference(c.sh_gu.at(int32_t(fg)), c.sh_gu.at(int32_t(fu)), rc, sref);
        for (int32_t card = 0; card < 2; ++card) {
            for (int32_t i = 0; i < rc.m.n_slots * n_ff_card; ++i) {
                ge.add(go.glu_f32[card][size_t(i)], ref[card][size_t(i)]);
            }
            for (int32_t i = 0; i < rc.rows * n_ff_card; ++i) {
                gs.add(go.glu_f32[card][size_t(rc.m.n_slots) * n_ff_card + size_t(i)], sref[card][size_t(i)]);
            }
        }
    }
    for (int32_t card = 0; card < 2; ++card) {
        const size_t gc = size_t(n_sl) * n_ff_card, gs4 = fn::act::scales_bytes(n_sl, n_ff_card);
        std::vector<int8_t> rq(gc);
        std::vector<float> rs(gs4 / 4);
        fn::quant::act_quantize_ref(go.glu_f32[card].data(), n_sl, n_ff_card, rq.data(), rs.data());
        act_same = act_same && std::memcmp(rq.data(), go.glu[card].data(), gc) == 0 &&
                   std::memcmp(rs.data(), go.glu[card].data() + glu_sc, gs4) == 0;
    }
    printf("ex-test: %s %-22s B %d U %2d slots %2d+%d: GLU routed %.3e shared %.3e (max|ref| %.3e / %.3e)%s, codes "
           "and scales %s\n",
           label, name.c_str(), rc.rows, rc.m.n_u, rc.m.n_slots, rc.rows, ge.rel(), gs.rel(), ge.max_ref, gs.max_ref,
           zero_x ? " zero input" : "", act_same ? "bitwise equal" : "DIFFER");
    const std::string what = std::string(label) + " " + name + " B " + std::to_string(rc.rows);
    check(c, zero_x ? zero_ok : (ge.rel() <= rel_bound && gs.rel() <= rel_bound), what + ": GLU bound / zero output");
    check(c, act_same, what + ": GLU codes and scales bitwise");
    return true;
}

bool check_down(test_ctx & c, fmt f, fmt fs, const row_case & rc, const gu_out & go, bool zero_x, const char * label,
                std::vector<float> (&out)[2]) {
    const std::string name = std::string(fmt_name(f)) + "+" + fmt_name(fs);
    bool ran = true;
    for (int32_t card = 0; card < 2 && ran; ++card) {
        uint32_t status = 0;
        ran = run_down(c, f, fs, card, rc, go.glu[card], out[card], status) && status == 0;
    }
    const std::string what = std::string(label) + " down " + name + " B " + std::to_string(rc.rows);
    if (!check(c, ran, what + ": runs (two, bitwise equal)")) {
        return false;
    }
    err_acc e[2], full;
    bool dzero = true;
    if (zero_x) {
        for (int32_t card = 0; card < 2; ++card) {
            for (const float v : out[card]) {
                dzero = dzero && v == 0.0f;
            }
        }
    } else {
        const fn::dense::fmt fsd = dense_of(fs);
        std::vector<double> ref[2];
        for (int32_t card = 0; card < 2; ++card) {
            down_reference(c.routed.at(rkey(f, true)), c.sh_dn.at(int32_t(fsd)), c, rc, card, go.glu[card], ref[card]);
            for (size_t i = 0; i < ref[card].size(); ++i) {
                e[card].add(out[card][i], ref[card][i]);
            }
        }
        for (size_t i = 0; i < ref[0].size(); ++i) {
            full.add(static_cast<double>(out[0][i] + out[1][i]), ref[0][i] + ref[1][i]);
        }
    }
    printf("ex-test: %s down %-14s B %d: card 0 %.3e, card 1 %.3e, full width %.3e (max|ref| %.3e)%s\n", label,
           name.c_str(), rc.rows, e[0].rel(), e[1].rel(), full.rel(), full.max_ref, zero_x ? ", zero input" : "");
    check(c, zero_x ? dzero : (e[0].rel() <= rel_bound && e[1].rel() <= rel_bound && full.rel() <= rel_bound),
          what + ": bound / zero output");
    return true;
}

// NC1 section 2: row b's GLU (each routed pair's and the shared row's) and down output at B equal the same row's at
// B = 4, bitwise
bool same_rows_gu(const gu_out & a, const row_case & ra, const gu_out & b4, const row_case & r4) {
    bool same = true;
    for (int32_t card = 0; card < 2; ++card) {
        for (int32_t b = 0; b < ra.rows; ++b) {
            for (int32_t k = 0; k < NU; ++k) {
                same = same && row_glu(a.glu[card], a.glu_f32[card], ra.m.slot_of_route(b, k)) ==
                                   row_glu(b4.glu[card], b4.glu_f32[card], r4.m.slot_of_route(b, k));
            }
            same = same && row_glu(a.glu[card], a.glu_f32[card], ra.m.n_slots + b) ==
                               row_glu(b4.glu[card], b4.glu_f32[card], r4.m.n_slots + b);
        }
    }
    return same;
}

bool same_rows_dn(const std::vector<float> (&a)[2], int32_t rows, const std::vector<float> (&b4)[2]) {
    bool same = true;
    for (int32_t card = 0; card < 2; ++card) {
        same = same && std::memcmp(a[card].data(), b4[card].data(), size_t(rows) * D * 4) == 0;
    }
    return same;
}

struct gu_inst_t {
    fmt f;
    fn::dense::fmt fg, fu;
};
struct dn_inst_t {
    fmt f, fs;
};

// ---- ME1: the expert cache's launches (fn/runtime/expert_cache.h) against the resident bank's ----------------------
// Each routed expert's packed bytes copied to its own place in a scattered buffer (reverse order, a 256-byte gap
// between experts) and named by a device table; the CPU-served experts (every second distinct expert) named as a zero
// expert, their gate/up sums from the CPU products through glu_finish and their down products added after down by
// the combine (ME3).

// a device table of 512 pointers: expert e at its scattered copy, or zero when cpu holds e
struct me1_bank {
    std::unique_ptr<dev_mem> buf, tab, zero;
};
bool me1_scatter(const std::vector<uint8_t> & packed_card, int64_t bytes, const std::set<int32_t> & cpu,
                 me1_bank & out) {
    const size_t stride = size_t(bytes) + 256;
    out.buf = std::make_unique<dev_mem>(stride * NE);
    out.tab = std::make_unique<dev_mem>(size_t(NE) * sizeof(void *));
    out.zero = std::make_unique<dev_mem>(size_t(bytes));
    if (out.buf->p == nullptr || out.tab->p == nullptr || out.zero->p == nullptr ||
        !cuda_ok(cudaMemset(out.buf->p, 0x5a, out.buf->n), "memset") ||
        !cuda_ok(cudaMemset(out.zero->p, 0, out.zero->n), "memset")) {
        return false;
    }
    std::vector<const uint8_t *> t(NE);
    for (int32_t e = 0; e < NE; ++e) {
        uint8_t * d = out.buf->p + stride * size_t(NE - 1 - e);
        if (!cuda_ok(cudaMemcpy(d, packed_card.data() + size_t(e) * size_t(bytes), size_t(bytes),
                                cudaMemcpyHostToDevice),
                     "scatter")) {
            return false;
        }
        t[size_t(e)] = cpu.count(e) != 0 ? out.zero->p : d;
    }
    return out.tab->put(t.data(), out.tab->n);
}

// the case's CPU set: every second distinct expert (u odd), and the slot mask they hold
std::set<int32_t> me1_cpu_set(const row_case & rc, uint64_t & slots) {
    std::set<int32_t> cpu;
    slots = 0;
    for (int32_t u = 1; u < rc.m.n_u; u += 2) {
        cpu.insert(rc.m.e[size_t(u)]);
        for (int32_t b = 0; b < rc.rows; ++b) {
            if ((rc.m.mask[size_t(u)] >> b) & 1) {
                slots |= uint64_t(1) << rc.m.slot(u, b);
            }
        }
    }
    return cpu;
}

// GPU-side resolution's two passes on one GLU buffer: the first (skip 1) runs the resident experts (u % 4 == 0) and the
// shared expert, the second (skip 2, its own meta, run_if a device word of 1) the GPU-served misses (u % 4 == 2); the
// CPU set (u odd) is left out of both. Every slot either pass ran is bitwise the bank's; the CPU set's slots keep the
// fill. A second pass with run_if 0 writes nothing.
void me1_two_pass(test_ctx & c, const gu_inst_t & g, const row_case & rc, int32_t card, const std::vector<uint8_t> & pk,
                  int64_t bytes, const std::vector<uint8_t> & glu_r, const std::string & name) {
    std::set<int32_t> hit, miss, rest;
    for (int32_t u = 0; u < rc.m.n_u; ++u) {
        const int32_t e = rc.m.e[size_t(u)];
        (u % 2 == 1 ? rest : u % 4 == 0 ? hit : miss).insert(e);
    }
    std::set<int32_t> not_hit = miss, not_miss = hit;
    not_hit.insert(rest.begin(), rest.end());
    not_miss.insert(rest.begin(), rest.end());
    me1_bank b1, b2;
    dev_mem x(rc.xq.size()), ids(rc.ids.size() * 4), gb(glu_buf_bytes), gf(size_t(glu_slots) * n_ff_card * 4), st(4),
        meta(meta_bytes), meta2(meta_bytes), run(8);
    bool ok = me1_scatter(pk, bytes, {}, b1) && me1_scatter(pk, bytes, {}, b2) && x.put(rc.xq.data(), rc.xq.size()) &&
              ids.put(rc.ids.data(), ids.n);
    // the tables: null where the pass skips the expert
    for (int32_t pass = 0; ok && pass < 2; ++pass) {
        me1_bank & bk = pass == 0 ? b1 : b2;
        std::vector<const uint8_t *> t(NE);
        ok = bk.tab->get(t.data(), bk.tab->n);
        for (const int32_t e : pass == 0 ? not_hit : not_miss) {
            t[size_t(e)] = nullptr;
        }
        ok = ok && bk.tab->put(t.data(), bk.tab->n);
    }
    const dev_mem * w = dev_packed(c, g.f, false, card);
    std::vector<uint8_t> out[2];
    for (int32_t arm = 0; ok && arm < 2; ++arm) { // arm 1: the second pass's run word 0
        const uint64_t one = arm == 0 ? 1 : 0;
        ok = ok && w != nullptr && cuda_ok(cudaMemset(gb.p, 0xa5, gb.n), "memset") &&
             cuda_ok(cudaMemset(st.p, 0, 4), "memset") && run.put(&one, 8);
        gate_up_args a = {w->p,
                          c.sh_gu.at(int32_t(g.fg)).d[card]->p,
                          c.sh_gu.at(int32_t(g.fu)).d[card]->p,
                          c.d_grids.p,
                          act_in(x.p, rc.rows, D),
                          ids.as<int32_t>(),
                          meta.p,
                          act_out(gb.p, glu_slots, n_ff_card),
                          gf.as<float>(),
                          st.as<uint32_t>()};
        a.cw = case_w(rc, g.f, g.fg, g.fu);
        a.wt = b1.tab->as<const uint8_t *>();
        a.skip = 1;
        gate_up_args a2 = a;
        a2.wt = b2.tab->as<const uint8_t *>();
        a2.skip = 2;
        a2.meta = meta2.p;
        a2.run_if = run.as<const uint32_t>();
        ok = ok && gate_up(g.f, g.fg, g.fu, rc.rows, a, cudaStreamPerThread) &&
             gate_up(g.f, g.fg, g.fu, rc.rows, a2, cudaStreamPerThread) &&
             cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "two passes");
        out[arm].assign(gb.n, 0);
        ok = ok && gb.get(out[arm].data(), gb.n);
        uint32_t status = 0;
        ok = ok && st.get(&status, 4) && status == 0;
    }
    if (!check(c, ok, name + " card " + std::to_string(card) + ": two-pass launches")) {
        return;
    }
    const auto slot_is = [&](const std::vector<uint8_t> & v, int32_t sl, const uint8_t * want, uint8_t fill) {
        for (int32_t i = 0; i < n_ff_card; ++i) {
            if (v[size_t(sl) * n_ff_card + size_t(i)] != (want != nullptr ? want[size_t(sl) * n_ff_card + size_t(i)]
                                                                           : fill)) {
                return false;
            }
        }
        for (int32_t i = 0; i < 80; ++i) {
            const size_t at = glu_sc + size_t(sl) * 80 + size_t(i);
            if (v[at] != (want != nullptr ? want[at] : fill)) {
                return false;
            }
        }
        return true;
    };
    bool run_same = true, rest_kept = true, off_kept = true;
    for (int32_t u = 0; u < rc.m.n_u; ++u) {
        const int32_t e = rc.m.e[size_t(u)];
        for (int32_t b = 0; b < rc.rows; ++b) {
            if (((rc.m.mask[size_t(u)] >> b) & 1) == 0) {
                continue;
            }
            const int32_t sl = rc.m.slot(u, b);
            if (rest.count(e) != 0) {
                rest_kept = rest_kept && slot_is(out[0], sl, nullptr, 0xa5);
            } else {
                run_same = run_same && slot_is(out[0], sl, glu_r.data(), 0);
            }
            off_kept = off_kept && (miss.count(e) != 0 ? slot_is(out[1], sl, nullptr, 0xa5)
                                                       : rest.count(e) != 0 || slot_is(out[1], sl, glu_r.data(), 0));
        }
    }
    for (int32_t b = 0; b < rc.rows; ++b) {
        run_same = run_same && slot_is(out[0], rc.m.n_slots + b, glu_r.data(), 0);
    }
    printf("ex-test: %s card %d: two passes (%zu resident, %zu GPU-served, %zu CPU experts): run slots %s, CPU slots "
           "%s; run_if 0: %s\n",
           name.c_str(), card, hit.size(), miss.size(), rest.size(), run_same ? "bitwise" : "DIFFER",
           rest_kept ? "unwritten" : "WRITTEN", off_kept ? "nothing written" : "WROTE");
    check(c, run_same, name + " card " + std::to_string(card) + ": two passes bitwise the bank's");
    check(c, rest_kept, name + " card " + std::to_string(card) + ": two passes leave the CPU slots");
    check(c, off_kept, name + " card " + std::to_string(card) + ": run_if 0");
}

// gate/up: the table launch bitwise the bank's GLU (all of it); the CPU split's GPU slots bitwise the bank's (ME2: a
// CPU expert's gate/up is computed whole on the CPU, experts_cpu.h, so its GPU slots are not read; ex-cpu-test checks
// the CPU products against FP64)
void me1_gate_up(test_ctx & c, const gu_inst_t & g, const row_case & rc) {
    const std::string name = "me1 " + gu_name(g.f, g.fg, g.fu) + " B " + std::to_string(rc.rows);
    // the host routing meta (the cache's) against make_meta's, which region_matches holds to the kernels'
    int32_t e[max_slots], mask[max_slots], base[max_slots], n_slots = 0;
    const int32_t n_u = cpu::window_meta(rc.ids.data(), rc.rows, e, mask, base, n_slots);
    bool meta_same = n_u == rc.m.n_u && n_slots == rc.m.n_slots;
    for (int32_t u = 0; meta_same && u < n_u; ++u) {
        meta_same = e[u] == rc.m.e[size_t(u)] && mask[u] == rc.m.mask[size_t(u)] &&
                    base[u] == rc.m.slot_base[size_t(u)];
    }
    check(c, meta_same, name + ": the host window meta");
    uint64_t slots = 0;
    const std::set<int32_t> cpu = me1_cpu_set(rc, slots);
    const int64_t bytes = gate_up_bytes(g.f);
    for (int32_t card = 0; card < 2; ++card) {
        const std::vector<uint8_t> & pk = packed(c, g.f, false, card);
        std::vector<uint8_t> glu_r, glu_t, glu_s;
        std::vector<float> f_r, f_t, f_s;
        uint32_t st = 0;
        me1_bank all, split;
        if (!check(c,
                   me1_scatter(pk, bytes, {}, all) && me1_scatter(pk, bytes, cpu, split) &&
                       run_gate_up(c, g.f, g.fg, g.fu, card, rc, glu_r, f_r, st) &&
                       run_gate_up(c, g.f, g.fg, g.fu, card, rc, glu_t, f_t, st,
                                   {0, nullptr, nullptr, all.tab->as<const uint8_t *>(), "me1-table"}) &&
                       run_gate_up(c, g.f, g.fg, g.fu, card, rc, glu_s, f_s, st,
                                   {0, nullptr, nullptr, split.tab->as<const uint8_t *>(), "me1-split"}),
                   name + ": launches")) {
            return;
        }
        check(c, glu_t == glu_r && f_t == f_r,
              name + " card " + std::to_string(card) + ": table GLU bitwise the bank's");
        bool gpu_same = true;
        for (int32_t sl = 0; sl < rc.m.n_slots + rc.rows; ++sl) { // the routed GPU slots, then the shared expert's
            if (sl < rc.m.n_slots && ((slots >> sl) & 1) != 0) {
                continue;
            }
            gpu_same = gpu_same &&
                       std::memcmp(glu_s.data() + size_t(sl) * n_ff_card, glu_r.data() + size_t(sl) * n_ff_card,
                                   n_ff_card) == 0 &&
                       std::memcmp(glu_s.data() + glu_sc + size_t(sl) * 80, glu_r.data() + glu_sc + size_t(sl) * 80,
                                   80) == 0;
        }
        printf("ex-test: %s card %d: table GLU %s; split (%zu CPU experts): GPU slots %s\n", name.c_str(), card,
               glu_t == glu_r && f_t == f_r ? "bitwise" : "DIFFER", cpu.size(), gpu_same ? "bitwise" : "DIFFER");
        check(c, gpu_same, name + " card " + std::to_string(card) + ": split GPU slots");
        me1_two_pass(c, g, rc, card, pk, bytes, glu_r, name);
    }
}

// ME2b: the combine (experts_cache.h) after a down launch that leaves the CPU slots out (ME3). CPU products: uniform
// in [-1, 1) at each CPU slot (every second distinct expert: slots at several places in route order). With the slot
// word: within the FP32 bound of the FP64 sum (the down output plus each CPU term w x ds, exact); no CPU slots (the
// word 0, the all-resident table): bitwise the table launch's output (b444839's down), the combine writing nothing.
void me2b_combine(test_ctx & c, const dn_inst_t & d, const row_case & rc, int32_t card, const std::string & name,
                  uint64_t slots, const std::vector<float> & o_d, const std::vector<float> & o_t,
                  const std::vector<float> & o_n, const std::vector<float> & prod) {
    (void)d;
    dev_mem ids(rc.ids.size() * 4), meta(meta_bytes), wts(rc.wts.size() * 4), dp(prod.size() * 4), word(8),
        out(o_d.size() * 4);
    std::vector<float> o_c(o_d.size()), o_0(o_t.size());
    const uint64_t zero = 0;
    const auto run = [&](const std::vector<float> & in, const uint64_t & w, std::vector<float> & got) {
        fn::experts::combine_args a{};
        a.meta = meta.p;
        a.wts = wts.as<float>();
        a.ext_ds = dp.as<float>();
        a.ext_slots = word.as<const uint64_t>();
        a.out = out.as<float>();
        a.rows = rc.rows;
        return word.put(&w, 8) && out.put(in.data(), out.n) && fn::experts::combine(a, cudaStreamPerThread) &&
               cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "combine") && out.get(got.data(), out.n);
    };
    const bool ran = ids.put(rc.ids.data(), ids.n) && device_meta(ids, rc.rows, dn_w, meta) &&
                     wts.put(rc.wts.data(), wts.n) && dp.put(prod.data(), dp.n) && run(o_d, slots, o_c) &&
                     run(o_t, zero, o_0);
    if (!check(c, ran, name + " card " + std::to_string(card) + ": combine launches")) {
        return;
    }
    constexpr double u = 1.0 / 16777216.0;
    double worst = 0;
    int32_t terms = 0;
    bool within = true;
    for (int32_t b = 0; b < rc.rows; ++b) {
        for (int32_t n = 0; n < D; ++n) {
            const size_t at = size_t(b) * D + size_t(n);
            double ref = double(o_d[at]), mag = std::fabs(ref);
            int32_t nt = 0;
            for (int32_t k = 0; k < NU; ++k) {
                int32_t uu = 0;
                while (rc.m.e[size_t(uu)] != rc.ids[size_t(b * NU + k)]) {
                    ++uu;
                }
                const int32_t sl = rc.m.slot(uu, b);
                if (((slots >> sl) & 1) != 0) {
                    const double t = double(rc.wts[size_t(b * NU + k)]) * double(prod[size_t(sl) * D + size_t(n)]);
                    ref += t;
                    mag += std::fabs(t);
                    ++nt;
                }
            }
            terms += n == 0 ? nt : 0;
            const double bound = double(nt) * u / (1.0 - double(nt) * u) * mag;
            const double e = std::fabs(double(o_c[at]) - ref);
            within = within && std::isfinite(double(o_c[at])) && e <= bound;
            worst = std::max(worst, bound > 0 ? e / bound : (e > 0 ? 1e30 : 0.0));
        }
    }
    const bool none = std::memcmp(o_n.data(), o_t.data(), o_t.size() * 4) == 0 &&
                      std::memcmp(o_0.data(), o_t.data(), o_t.size() * 4) == 0;
    printf("ex-test: %s card %d: combine (%d CPU terms over the rows): within the FP32 bound of FP64 %s (worst %.3f of "
           "it); no CPU slots: down and combine %s\n",
           name.c_str(), card, terms, within ? "ok" : "NOT", worst, none ? "bitwise the table output" : "DIFFER");
    check(c, within, name + " card " + std::to_string(card) + ": combine within the FP32 bound of FP64");
    check(c, none, name + " card " + std::to_string(card) + ": no CPU slots bitwise b444839's down");
}

// down: the table launch (the cache's instantiation) bitwise the bank's output (the resident one); the CPU split (ME3:
// the CPU slots run nothing and take no term): the CPU set's experts zero in the reference bank (each one's ds 0)
// against the split launch (their GLU slots and weights poisoned), equal as floats (fmaf(w, 0, acc) against no term
// differs at most in a zero's sign; the signed-zero differences are counted), with the slot mask as an argument and as
// the device slot word
void me1_down(test_ctx & c, const dn_inst_t & d, const row_case & rc, const gu_out & go) {
    const std::string name =
        std::string("me1 down ") + fmt_name(d.f) + "+" + fmt_name(d.fs) + " B " + std::to_string(rc.rows);
    uint64_t slots = 0;
    const std::set<int32_t> cpu = me1_cpu_set(rc, slots);
    const int64_t bytes = down_bytes(d.f);
    for (int32_t card = 0; card < 2; ++card) {
        const std::vector<uint8_t> & pk = packed(c, d.f, true, card);
        const std::vector<uint8_t> & glu = go.glu[card];
        std::vector<float> o_r, o_t, o_z, o_s;
        uint32_t st = 0;
        me1_bank all, split, zeroed;
        std::vector<uint8_t> zk = pk;
        for (const int32_t e : cpu) {
            std::memset(zk.data() + size_t(e) * size_t(bytes), 0, size_t(bytes));
        }
        std::vector<uint8_t> poisoned = glu;
        for (int32_t sl = 0; sl < rc.m.n_slots; ++sl) {
            if ((slots >> sl) & 1) {
                std::memset(poisoned.data() + size_t(sl) * n_ff_card, 0x7f, n_ff_card);
            }
        }
        const bool ran = me1_scatter(pk, bytes, {}, all) &&
                         me1_scatter(zk, bytes, {}, zeroed) && me1_scatter(pk, bytes, cpu, split) &&
                         run_down(c, d.f, d.fs, card, rc, glu, o_r, st) &&
                         run_down(c, d.f, d.fs, card, rc, glu, o_t, st,
                                  {all.tab->as<const uint8_t *>(), 0, "me1-table"}) &&
                         run_down(c, d.f, d.fs, card, rc, glu, o_z, st,
                                  {zeroed.tab->as<const uint8_t *>(), 0, "me2-zeroed"}) &&
                         run_down(c, d.f, d.fs, card, rc, poisoned, o_s, st,
                                  {split.tab->as<const uint8_t *>(), slots, "me1-split"});
        if (!check(c, ran, name + ": launches")) {
            return;
        }
        std::vector<float> o_d;
        dev_mem word(8);
        const bool ran_d = word.put(&slots, 8) &&
                           run_down(c, d.f, d.fs, card, rc, poisoned, o_d, st,
                                    {split.tab->as<const uint8_t *>(), 0, "me1-slot-word", word.as<const uint64_t>()});
        // ME3: equal as floats, counting the bitwise differences (each a zero's sign)
        const auto float_eq = [](const std::vector<float> & x, const std::vector<float> & y, int32_t & sz) {
            sz = 0;
            bool eq = x.size() == y.size();
            for (size_t i = 0; eq && i < x.size(); ++i) {
                eq = x[i] == y[i];
                sz += eq && std::memcmp(&x[i], &y[i], 4) != 0 ? 1 : 0;
            }
            return eq;
        };
        int32_t sz_s = 0, sz_d = 0;
        const bool t_same = o_t == o_r && std::memcmp(o_t.data(), o_r.data(), o_r.size() * 4) == 0;
        const bool s_same = float_eq(o_s, o_z, sz_s);
        const bool d_same = ran_d && float_eq(o_d, o_z, sz_d);
        printf("ex-test: %s card %d: table output %s; split (%zu CPU experts, no down work) output %s the zeroed "
               "bank's (%d signed-zero differences); with the device slot word %s (%d)\n",
               name.c_str(), card, t_same ? "bitwise" : "DIFFER", cpu.size(), s_same ? "equal to" : "DIFFER from", sz_s,
               d_same ? "equal" : "DIFFER", sz_d);
        check(c, t_same, name + " card " + std::to_string(card) + ": table output");
        check(c, s_same, name + " card " + std::to_string(card) + ": CPU split output (down skip)");
        check(c, d_same, name + " card " + std::to_string(card) + ": CPU split output, device slot word");
        // ME2b: the CPU products and the no-CPU window (the word 0, every expert in the table)
        std::vector<float> prod(size_t(max_slots) * D);
        std::mt19937 pr(0x2b0c + uint32_t(rc.rows) * 7u + uint32_t(card));
        std::uniform_real_distribution<float> pu(-1.0f, 1.0f);
        for (float & v : prod) {
            v = pu(pr);
        }
        std::vector<float> o_n;
        dev_mem word0(8);
        const uint64_t zero = 0;
        const bool ran_b = word0.put(&zero, 8) &&
                           run_down(c, d.f, d.fs, card, rc, glu, o_n, st,
                                    {all.tab->as<const uint8_t *>(), 0, "me2b-no-cpu", word0.as<const uint64_t>()});
        if (check(c, ran_b, name + " card " + std::to_string(card) + ": ME2b down launches")) {
            me2b_combine(c, d, rc, card, name, slots, o_d, o_t, o_n, prod);
        }
    }
}

// ME2: the cache's host policy (fn/runtime/expert_policy.h) without a model: the profile file round trip and its
// refusals, the learned ranking's order, the boot placement on a tiny pool (class slots, the zone's mixed slots, each
// key once), the swap rule (the usage and gain thresholds, the cap by gain, layer order), the decay, PcieAuto's share
// changes (after 48, then 96 windows, rounded and clamped, three updates at most) and the miss split
void me2_policy(test_ctx & c) {
    namespace pol = fn::runtime::policy;
    std::string err;
    // the profile file
    const std::string path =
        (std::filesystem::temp_directory_path() / ("fnfl-me2-profile-" + std::to_string(getpid()) + ".bin")).string();
    std::vector<int32_t> keys(48 * NE);
    for (int32_t i = 0; i < 48 * NE; ++i) {
        keys[size_t(i)] = i;
    }
    std::mt19937 r(0x4d453250u);
    std::shuffle(keys.begin(), keys.end(), r);
    keys.resize(3000);
    std::vector<int32_t> back;
    const bool rt = pol::write_profile(path, 48, keys, err) && pol::read_profile(path, 48, back, err) && back == keys;
    std::vector<int32_t> none;
    const bool wrong_layers = !pol::read_profile(path, 40, none, err);
    bool dup_refused = false;
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        char pair[4];
        f.seekg(24);
        f.read(pair, 4);
        f.seekp(28);
        f.write(pair, 4);
        f.close();
        dup_refused = !pol::read_profile(path, 48, none, err);
    }
    std::filesystem::remove(path);
    printf("ex-test: me2 profile: %zu keys written and read %s; a 40-layer read %s; a repeated pair %s\n", keys.size(),
           rt ? "equal" : "DIFFERENT", wrong_layers ? "refused" : "ACCEPTED", dup_refused ? "refused" : "ACCEPTED");
    check(c, rt && wrong_layers && dup_refused, "me2 profile round trip and refusals");
    // the learned ranking: residents first by heat, then heat, then the prior, then key
    {
        std::vector<uint8_t> res(NE, 0);
        std::vector<double> heat(NE, 0.0);
        res[5] = res[9] = 1;
        heat[3] = heat[7] = 10;
        heat[9] = 1;
        const std::vector<int32_t> o = pol::rank_learned(1, res, heat, {7, 3, 20, 5});
        const std::vector<int32_t> want = {9, 5, 7, 3, 20, 0, 1, 2, 4, 6, 8, 10};
        std::vector<int32_t> sorted = o;
        std::sort(sorted.begin(), sorted.end());
        bool perm = sorted.size() == size_t(NE);
        for (int32_t i = 0; perm && i < NE; ++i) {
            perm = sorted[size_t(i)] == i;
        }
        const bool head = o.size() >= want.size() && std::equal(want.begin(), want.end(), o.begin());
        printf("ex-test: me2 learned ranking: %s, head %s\n", perm ? "a permutation" : "NOT a permutation",
               head ? "as the rule" : "NOT as the rule");
        check(c, perm && head, "me2 learned ranking order");
    }
    // the boot placement: layers 0 and 2 class 0 (100 bytes, 3 slots), layer 1 class 1 (60 bytes, 2 slots), zone 250
    {
        pol::place_in in;
        in.class_of_layer = {0, 1, 0};
        in.entry = {100, 60};
        in.count = {3, 2};
        in.zone_bytes = 250;
        const auto k = [](int32_t l, int32_t e) { return l * NE + e; };
        const pol::place_out o = pol::place(
            in, {k(0, 1), k(1, 2), k(0, 3), k(2, 4), k(0, 5), k(1, 6), k(1, 7), k(0, 8), k(2, 9), k(1, 10), k(0, 1)});
        const bool ok = o.n_main == 5 && o.first == std::vector<int32_t>{0, 3} &&
                        o.slot_key == std::vector<int32_t>{k(0, 1), k(0, 3), k(2, 4), k(1, 2), k(1, 6), k(0, 5),
                                                           k(1, 7), k(1, 10)} &&
                        o.slot_cls == std::vector<int32_t>{0, 0, 0, 1, 1, 0, 1, 1} &&
                        o.zone_at == std::vector<int64_t>{0, 100, 160};
        printf("ex-test: me2 placement (2 classes, 5 slots, 250 zone bytes): %s\n", ok ? "as the rule" : "NOT as the rule");
        check(c, ok, "me2 boot placement");
    }
    // the swap rule
    {
        std::vector<float> use(size_t(2) * NE, 0.0f);
        std::vector<int32_t> slot(size_t(2) * NE, -1);
        const float u0[4] = {0, 5, 1, 3};
        for (int32_t e = 0; e < 4; ++e) {
            slot[size_t(e)] = e;
            use[size_t(e)] = u0[e];
        }
        use[10] = 9;
        use[11] = 4;
        use[12] = 2.0f;
        use[13] = 1.9f;
        slot[size_t(NE)] = 4;
        slot[size_t(NE) + 1] = 5;
        use[size_t(NE) + 5] = 1.6f;
        use[size_t(NE) + 6] = 2.0f;
        pol::adapt_options o;
        const std::vector<pol::swap> s = pol::choose_swaps(2, use.data(), slot.data(), o);
        const bool rule = s.size() == 3 && s[0].layer == 0 && s[0].in == 10 && s[0].out == 0 && s[1].layer == 0 &&
                          s[1].in == 11 && s[1].out == 2 && s[2].layer == 1 && s[2].in == 6 &&
                          (s[2].out == 0 || s[2].out == 1);
        o.max_swaps = 2;
        const std::vector<pol::swap> s2 = pol::choose_swaps(2, use.data(), slot.data(), o);
        const bool cap2 = s2.size() == 2 && s2[0].in == 10 && s2[1].in == 11;
        // 128 residents at 0 against 200 candidates 2.0 + 0.1 i: 128 pass, the 96 largest gains kept
        std::vector<float> u3(NE, 0.0f);
        std::vector<int32_t> sl3(NE, -1);
        for (int32_t e = 0; e < 128; ++e) {
            sl3[size_t(e)] = e;
        }
        for (int32_t i = 0; i < 200; ++i) {
            u3[size_t(128 + i)] = 2.0f + 0.1f * float(i);
        }
        const std::vector<pol::swap> s3 = pol::choose_swaps(1, u3.data(), sl3.data(), pol::adapt_options());
        bool cap = s3.size() == 96;
        for (const pol::swap & x : s3) {
            cap = cap && x.in >= 128 + 104 && x.out < 128;
        }
        std::vector<float> du = {1.0f, 2.0f};
        std::vector<double> dh = {0.0, 1.0};
        pol::decay_usage(du, dh, 0.7f);
        const bool decay = dh[0] == 1.0 && dh[1] == 3.0 && du[0] == 0.7f && du[1] == 1.4f;
        printf("ex-test: me2 swap rule: thresholds and order %s; cap 2 %s; cap 96 of 128 by gain %s; decay %s\n",
               rule ? "as the rule" : "NOT", cap2 ? "ok" : "NOT", cap ? "ok" : "NOT", decay ? "ok" : "NOT");
        check(c, rule && cap2 && cap && decay, "me2 swap rule");
    }
    // PcieAuto, ME5 rider (Fable 2026-10-18): each leg's own clock, refinement for the whole run, the 0.0375 deadband on
    // the unrounded reading; the cases are fn/runtime/split_test.cpp's (also run without a GPU by me5_split_check)
    {
        std::string report;
        const bool ok = fn::runtime::split_estimator_check(report);
        size_t at = 0;
        for (size_t nl = report.find('\n'); nl != std::string::npos; at = nl + 1, nl = report.find('\n', at)) {
            printf("ex-test: me5 split estimator: %s\n", report.substr(at, nl - at).c_str());
        }
        check(c, ok, "me5 split estimator (PcieAuto share and split)");
    }
}

void me2_sleep(void * ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(reinterpret_cast<intptr_t>(ms)));
}

// ME2: a pass's device ordering (fn/runtime/expert_cache.h publish_pass, wait_layer) on a tiny pool of 4 slots: the
// compute stream held (a host function) with a reader of the victim slot enqueued, then the publication: the pass
// is pending while the reader waits (its event not reached), the reader sees the old bytes (the lease), a reader after
// the layer's wait the new bytes, the residency rows patched, the layer events set for the layers with copies only; a
// pass out of layer order refused; a refill after a held copy stream's staging write lands after it
void me2_swap(test_ctx & c) {
    constexpr size_t SB = 4096;
    cudaStream_t cs = nullptr, ss = nullptr, cp = nullptr;
    cudaEvent_t pub = nullptr, lay[3] = {}, t0 = nullptr, t1 = nullptr, w0 = nullptr, w1 = nullptr, ecp = nullptr;
    bool ok = cuda_ok(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream") &&
              cuda_ok(cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking), "stream") &&
              cuda_ok(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking), "stream") &&
              cuda_ok(cudaEventCreateWithFlags(&pub, cudaEventDisableTiming), "event") &&
              cuda_ok(cudaEventCreateWithFlags(&ecp, cudaEventDisableTiming), "event") &&
              cuda_ok(cudaEventCreate(&t0), "event") && cuda_ok(cudaEventCreate(&t1), "event") &&
              cuda_ok(cudaEventCreate(&w0), "event") && cuda_ok(cudaEventCreate(&w1), "event");
    for (auto & e : lay) {
        ok = ok && cuda_ok(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
    }
    dev_mem pool(4 * SB), words(size_t(3) * NE * 8), rb(3 * SB), idx_d(64 * 4), val_d(64 * 8);
    uint8_t * host = nullptr;
    int32_t * idx = nullptr;
    uint64_t * val = nullptr;
    ok = ok && cuda_ok(cudaMallocHost(&host, 4 * SB), "pinned") && cuda_ok(cudaMallocHost(&idx, 64 * 4), "pinned") &&
         cuda_ok(cudaMallocHost(&val, 64 * 8), "pinned");
    if (!check(c, ok && pool.p != nullptr && words.p != nullptr, "me2 swap: setup")) {
        return;
    }
    std::vector<uint8_t> old(4 * SB);
    for (size_t i = 0; i < old.size(); ++i) {
        old[i] = uint8_t(0xa0 + i / SB);
        host[i] = uint8_t(0x11 * (1 + i / SB));
    }
    std::vector<uint64_t> w(size_t(3) * NE, 0);
    w[3] = uint64_t(reinterpret_cast<uintptr_t>(pool.p));                 // layer 0: expert 3 in slot 0
    w[size_t(NE) + 4] = uint64_t(reinterpret_cast<uintptr_t>(pool.p + SB)); // layer 1: expert 4 in slot 1
    ok = pool.put(old.data(), old.size()) && words.put(w.data(), words.n);
    // the held compute stream, its reader of slot 0, then the publication
    ok = ok && cuda_ok(cudaLaunchHostFunc(cs, me2_sleep, reinterpret_cast<void *>(intptr_t(300))), "hold") &&
         cuda_ok(cudaMemcpyAsync(rb.p, pool.p, SB, cudaMemcpyDeviceToDevice, cs), "reader");
    const int32_t n = 4;
    idx[0] = 3;
    val[0] = 0;
    idx[1] = 7;
    val[1] = w[3];
    idx[2] = NE + 4;
    val[2] = 0;
    idx[3] = NE + 8;
    val[3] = w[size_t(NE) + 4];
    fn::runtime::pass_dev p;
    p.words = words.as<uint64_t>();
    p.idx_d = idx_d.as<int32_t>();
    p.val_d = val_d.as<uint64_t>();
    p.idx_h = idx;
    p.val_h = val;
    p.n = n;
    p.compute = cs;
    p.swap = ss;
    p.ev_pub = pub;
    void * evl[3] = {lay[0], lay[1], lay[2]};
    p.ev_layer = evl;
    p.ev_t0 = t0;
    p.ev_t1 = t1;
    uint8_t has[3] = {};
    std::string err;
    ok = ok && fn::runtime::publish_pass(p, {{0, pool.p, host, int64_t(SB)}, {1, pool.p + SB, host + SB, int64_t(SB)}},
                                         has, err);
    const bool pending = ok && cudaEventQuery(t1) == cudaErrorNotReady;
    ok = ok && fn::runtime::wait_layer(cs, lay[0], w0, w1, err) &&
         cuda_ok(cudaMemcpyAsync(rb.p + SB, pool.p, SB, cudaMemcpyDeviceToDevice, cs), "reader") &&
         fn::runtime::wait_layer(cs, lay[1], w0, w1, err) &&
         cuda_ok(cudaMemcpyAsync(rb.p + 2 * SB, pool.p + SB, SB, cudaMemcpyDeviceToDevice, cs), "reader") &&
         cuda_ok(cudaDeviceSynchronize(), "sync");
    const bool landed = ok && cudaEventQuery(t1) == cudaSuccess;
    std::vector<uint8_t> got(3 * SB);
    std::vector<uint64_t> wg(w.size());
    ok = ok && rb.get(got.data(), got.size()) && words.get(wg.data(), words.n);
    const bool lease = ok && std::memcmp(got.data(), old.data(), SB) == 0;
    const bool fresh = ok && std::memcmp(got.data() + SB, host, SB) == 0 && std::memcmp(got.data() + 2 * SB, host + SB, SB) == 0;
    const bool rows = ok && wg[3] == 0 && wg[7] == w[3] && wg[size_t(NE) + 4] == 0 && wg[size_t(NE) + 8] == w[size_t(NE) + 4];
    const bool layers = has[0] == 1 && has[1] == 1 && has[2] == 0;
    // out of layer order: refused
    uint8_t has2[3] = {};
    p.n = 0;
    const bool order = !fn::runtime::publish_pass(
        p, {{1, pool.p + 3 * SB, host + 3 * SB, int64_t(SB)}, {0, pool.p + 3 * SB, host + 3 * SB, int64_t(SB)}}, has2,
        err);
    cudaDeviceSynchronize();
    // a refill: the copy stream held, then a staging write of slot 2; the refill's copies after it (ev_cp)
    std::vector<uint8_t> junk(SB, 0xee);
    uint8_t * junk_h = nullptr;
    ok = ok && cuda_ok(cudaMallocHost(&junk_h, SB), "pinned");
    if (ok) {
        std::memcpy(junk_h, junk.data(), SB);
    }
    idx[0] = 2 * NE + 9;
    val[0] = uint64_t(reinterpret_cast<uintptr_t>(pool.p + 2 * SB));
    p.n = 1;
    uint8_t has3[3] = {};
    ok = ok && cuda_ok(cudaLaunchHostFunc(cp, me2_sleep, reinterpret_cast<void *>(intptr_t(200))), "hold") &&
         cuda_ok(cudaMemcpyAsync(pool.p + 2 * SB, junk_h, SB, cudaMemcpyHostToDevice, cp), "staging") &&
         cuda_ok(cudaEventRecord(ecp, cp), "event") && cuda_ok(cudaStreamWaitEvent(ss, ecp, 0), "wait") &&
         fn::runtime::publish_pass(p, {{2, pool.p + 2 * SB, host + 2 * SB, int64_t(SB)}}, has3, err) &&
         fn::runtime::wait_layer(cs, lay[2], w0, w1, err) &&
         cuda_ok(cudaMemcpyAsync(rb.p, pool.p + 2 * SB, SB, cudaMemcpyDeviceToDevice, cs), "reader") &&
         cuda_ok(cudaDeviceSynchronize(), "sync") && rb.get(got.data(), SB) && words.get(wg.data(), words.n);
    const bool refill = ok && std::memcmp(got.data(), host + 2 * SB, SB) == 0 && has3[2] == 1 &&
                        wg[size_t(2) * NE + 9] == uint64_t(reinterpret_cast<uintptr_t>(pool.p + 2 * SB));
    printf("ex-test: me2 swap ordering: pending while the readers run %s; the earlier reader %s; after the layer's "
           "wait %s; rows %s; layer events %s; out of order %s; refill after the staging write %s; landed %s\n",
           pending ? "yes" : "NO", lease ? "old bytes (the lease)" : "NEW BYTES", fresh ? "new bytes" : "OLD BYTES",
           rows ? "patched" : "NOT", layers ? "0, 1 only" : "WRONG", order ? "refused" : "ACCEPTED",
           refill ? "ok" : "NOT", landed ? "yes" : "NO");
    check(c, pending && lease && fresh && rows && layers && order && refill && landed, "me2 swap ordering");
    for (cudaEvent_t e : {pub, t0, t1, w0, w1, ecp, lay[0], lay[1], lay[2]}) {
        cudaEventDestroy(e);
    }
    for (cudaStream_t s : {cs, ss, cp}) {
        cudaStreamDestroy(s);
    }
    cudaFreeHost(host);
    cudaFreeHost(idx);
    cudaFreeHost(val);
    cudaFreeHost(junk_h);
}

// ME2b: the lent region's device ordering (fn/runtime/expert_cache.h lent_sync, publish_pass, wait_layer, as
// expert_cache::prompt_hold and zone_return use them) on a tiny pool of 6 slots, slots 4 and 5 the lent region
// (slot 5 PF1's staging view, slot 4 PB1's arena view), each holding a resident. Borrow: decode readers of both
// residents held on the compute stream and a late copy into slot 5 held on the swap stream, then the residency rows
// retired on the compute stream and lent_sync; the prompt's first write is the staging upload on its own copy stream
// at once (the early staging boundary, prefill.cpp's upload before run_bulk's run_start), then the arena's
// preparation and the chunk on the compute stream. The readers see the residents' bytes, no row points into the
// region while it is held, the staging holds the upload (not the late copy), and the prompt's output is bitwise the
// output with PB1 and PF1 reserved outside the pool. Return: lent_sync, then the refill as a pass; a reader after
// the layer's wait sees the refilled bytes and the rows point at them again. Poison: the borrow fills the region
// with 0xFF (seen before the upload) and the output is unchanged. A failed prompt: borrowed, its write held on the
// compute stream, returned without its join; the refill lands after the write.
bool me2b_prompt(cudaStream_t cs, cudaStream_t cp, cudaEvent_t up, uint8_t * pf1, uint8_t * pb1, const uint8_t * in_h,
                 uint8_t * out, size_t sb) {
    return cuda_ok(cudaMemcpyAsync(pf1, in_h, sb, cudaMemcpyHostToDevice, cp), "staging upload") &&
           cuda_ok(cudaEventRecord(up, cp), "event") && cuda_ok(cudaStreamWaitEvent(cs, up, 0), "wait") &&
           cuda_ok(cudaMemsetAsync(pb1, 0, sb / 2, cs), "run start") &&
           cuda_ok(cudaMemcpyAsync(pb1 + sb / 2, pf1, sb / 2, cudaMemcpyDeviceToDevice, cs), "chunk") &&
           cuda_ok(cudaMemcpyAsync(out, pb1, sb, cudaMemcpyDeviceToDevice, cs), "chunk output");
}

void me2b_lent(test_ctx & c, int device) {
    constexpr size_t SB = 4096;
    constexpr int32_t e_pb = 10, e_pf = 11, e_main = 3;
    cudaStream_t cs = nullptr, ss = nullptr, cp = nullptr;
    cudaEvent_t pub = nullptr, lay[3] = {}, t0 = nullptr, t1 = nullptr, w0 = nullptr, w1 = nullptr, up = nullptr,
                er = nullptr;
    bool ok = cuda_ok(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream") &&
              cuda_ok(cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking), "stream") &&
              cuda_ok(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking), "stream") &&
              cuda_ok(cudaEventCreateWithFlags(&pub, cudaEventDisableTiming), "event") &&
              cuda_ok(cudaEventCreateWithFlags(&up, cudaEventDisableTiming), "event") &&
              cuda_ok(cudaEventCreateWithFlags(&er, cudaEventDisableTiming), "event") &&
              cuda_ok(cudaEventCreate(&t0), "event") && cuda_ok(cudaEventCreate(&t1), "event") &&
              cuda_ok(cudaEventCreate(&w0), "event") && cuda_ok(cudaEventCreate(&w1), "event");
    for (auto & e : lay) {
        ok = ok && cuda_ok(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
    }
    dev_mem pool(6 * SB), words(size_t(NE) * 8), rb(2 * SB), out(SB), out_ref(SB), own(2 * SB), idx_d(64 * 4),
        val_d(64 * 8);
    uint8_t * host = nullptr; // [0] the prompt's input, [1] the late copy, [2] [3] the residents' refill bytes
    int32_t * idx = nullptr;
    uint64_t * val = nullptr, * rows_h = nullptr;
    ok = ok && cuda_ok(cudaMallocHost(&host, 4 * SB), "pinned") && cuda_ok(cudaMallocHost(&idx, 64 * 4), "pinned") &&
         cuda_ok(cudaMallocHost(&val, 64 * 8), "pinned") && cuda_ok(cudaMallocHost(&rows_h, size_t(NE) * 8), "pinned");
    if (!check(c, ok && pool.p != nullptr && words.p != nullptr && own.p != nullptr, "me2b lent region: setup")) {
        return;
    }
    uint8_t * const region = pool.p + 4 * SB;
    uint8_t * const pb1 = pool.p + 4 * SB, * const pf1 = pool.p + 5 * SB;
    const auto addr = [&](int32_t slot) { return uint64_t(reinterpret_cast<uintptr_t>(pool.p + size_t(slot) * SB)); };
    std::vector<uint8_t> boot(6 * SB);
    for (size_t i = 0; i < boot.size(); ++i) {
        boot[i] = uint8_t(0xa0 + i / SB);
    }
    for (size_t i = 0; i < 4 * SB; ++i) {
        host[i] = uint8_t(i < SB ? (i * 7 + 1) & 0xff : 0x11 * (1 + i / SB));
    }
    std::vector<uint64_t> w(NE, 0);
    w[e_main] = addr(0);
    w[e_pb] = addr(4);
    w[e_pf] = addr(5);
    ok = pool.put(boot.data(), boot.size()) && words.put(w.data(), words.n);
    const auto into_region = [&](const std::vector<uint64_t> & v) {
        for (const uint64_t a : v) {
            if (a >= uint64_t(reinterpret_cast<uintptr_t>(region)) &&
                a < uint64_t(reinterpret_cast<uintptr_t>(region + 2 * SB))) {
                return true;
            }
        }
        return false;
    };
    // the borrow, as prompt_hold: the residents retired (rows on the compute stream, as zone_take), then lent_sync
    const int32_t dev[1] = {device};
    std::string err;
    const auto borrow = [&](bool poison) {
        std::memcpy(rows_h, w.data(), size_t(NE) * 8);
        rows_h[e_pb] = 0;
        rows_h[e_pf] = 0;
        return cuda_ok(cudaMemcpyAsync(words.p, rows_h, size_t(NE) * 8, cudaMemcpyHostToDevice, cs), "rows") &&
               fn::runtime::lent_sync(dev, 1, true, region, int64_t(2 * SB), poison, err);
    };
    // the refill after a return, as zone_return's pass: rows patched, copies on the swap stream, the layer's wait
    fn::runtime::pass_dev p;
    p.words = words.as<uint64_t>();
    p.idx_d = idx_d.as<int32_t>();
    p.val_d = val_d.as<uint64_t>();
    p.idx_h = idx;
    p.val_h = val;
    p.compute = cs;
    p.swap = ss;
    p.ev_pub = pub;
    void * evl[3] = {lay[0], lay[1], lay[2]};
    p.ev_layer = evl;
    p.ev_t0 = t0;
    p.ev_t1 = t1;
    const auto refill = [&]() {
        idx[0] = e_pb;
        val[0] = addr(4);
        idx[1] = e_pf;
        val[1] = addr(5);
        p.n = 2;
        uint8_t has[3] = {};
        return fn::runtime::publish_pass(
                   p, {{0, pb1, host + 2 * SB, int64_t(SB)}, {0, pf1, host + 3 * SB, int64_t(SB)}}, has, err) &&
               fn::runtime::wait_layer(cs, lay[0], w0, w1, err) &&
               cuda_ok(cudaMemcpyAsync(rb.p, region, 2 * SB, cudaMemcpyDeviceToDevice, cs), "reader") &&
               cuda_ok(cudaDeviceSynchronize(), "sync");
    };
    std::vector<uint8_t> got(2 * SB), o1(SB), o2(SB), o_ref(SB), stage(SB), pois(2 * SB);
    std::vector<uint64_t> wg(NE);
    // the reference: PB1 and PF1 their own allocation
    ok = ok && me2b_prompt(cs, cp, up, own.p + SB, own.p, host, out_ref.p, SB) &&
         cuda_ok(cudaDeviceSynchronize(), "sync") && out_ref.get(o_ref.data(), SB);
    // the borrow behind held readers and a held late copy (after the readers, as a swap's lease), then the prompt at
    // once
    ok = ok && cuda_ok(cudaLaunchHostFunc(cs, me2_sleep, reinterpret_cast<void *>(intptr_t(200))), "hold") &&
         cuda_ok(cudaMemcpyAsync(rb.p, region, 2 * SB, cudaMemcpyDeviceToDevice, cs), "reader") &&
         cuda_ok(cudaEventRecord(er, cs), "event") && cuda_ok(cudaStreamWaitEvent(ss, er, 0), "wait") &&
         cuda_ok(cudaLaunchHostFunc(ss, me2_sleep, reinterpret_cast<void *>(intptr_t(100))), "hold") &&
         cuda_ok(cudaMemcpyAsync(pf1, host + SB, SB, cudaMemcpyHostToDevice, ss), "late copy") && borrow(false) &&
         words.get(wg.data(), words.n);
    const bool retired = ok && !into_region(wg) && wg[e_main] == addr(0);
    ok = ok && me2b_prompt(cs, cp, up, pf1, pb1, host, out.p, SB) && cuda_ok(cudaDeviceSynchronize(), "sync") &&
         rb.get(got.data(), 2 * SB) && out.get(o1.data(), SB) && cudaMemcpy(stage.data(), pf1, SB,
                                                                             cudaMemcpyDeviceToHost) == cudaSuccess;
    const bool readers = ok && std::memcmp(got.data(), boot.data() + 4 * SB, 2 * SB) == 0;
    const bool staged = ok && std::memcmp(stage.data(), host, SB) == 0;
    const bool same = ok && o1 == o_ref;
    // the return and the refill
    ok = ok && fn::runtime::lent_sync(dev, 1, false, region, int64_t(2 * SB), false, err) && refill() &&
         rb.get(got.data(), 2 * SB) && words.get(wg.data(), words.n);
    const bool refilled = ok && std::memcmp(got.data(), host + 2 * SB, 2 * SB) == 0 && wg[e_pb] == addr(4) &&
                          wg[e_pf] == addr(5);
    printf("ex-test: me2b lent region: readers before the borrow %s; rows while held %s; the staging upload %s; the "
           "prompt's output %s; after the return and refill %s\n",
           readers ? "saw the residents" : "SAW PROMPT BYTES", retired ? "none into the region" : "POINT INTO IT",
           staged ? "kept (after the late copy)" : "OVERWRITTEN",
           same ? "bitwise the own-allocation output" : "DIFFERS",
           refilled ? "refilled bytes, rows restored" : "NOT REFILLED");
    check(c, readers && retired && staged && same && refilled, "me2b lent region");
    // poison: the region 0xFF at the borrow, the output unchanged
    ok = ok && borrow(true) && cudaMemcpy(pois.data(), region, 2 * SB, cudaMemcpyDeviceToHost) == cudaSuccess &&
         me2b_prompt(cs, cp, up, pf1, pb1, host, out.p, SB) && cuda_ok(cudaDeviceSynchronize(), "sync") &&
         out.get(o2.data(), SB) && fn::runtime::lent_sync(dev, 1, false, region, int64_t(2 * SB), false, err) &&
         refill() && rb.get(got.data(), 2 * SB);
    const bool poisoned = ok && std::all_of(pois.begin(), pois.end(), [](uint8_t b) { return b == 0xff; });
    const bool same2 = ok && o2 == o_ref && std::memcmp(got.data(), host + 2 * SB, 2 * SB) == 0;
    printf("ex-test: me2b poison: the region %s at the borrow; the prompt's output %s\n",
           poisoned ? "0xFF" : "NOT POISONED", same2 ? "bitwise unchanged, then refilled" : "DIFFERS");
    check(c, poisoned && same2, "me2b poison");
    // a failed prompt: its arena write held on the compute stream, the region returned without the prompt's join
    ok = ok && borrow(false) && words.get(wg.data(), words.n);
    const bool retired2 = ok && !into_region(wg);
    ok = ok && cuda_ok(cudaLaunchHostFunc(cs, me2_sleep, reinterpret_cast<void *>(intptr_t(200))), "hold") &&
         cuda_ok(cudaMemsetAsync(pb1, 0x5a, SB, cs), "failed write") &&
         fn::runtime::lent_sync(dev, 1, false, region, int64_t(2 * SB), false, err) && refill() &&
         rb.get(got.data(), 2 * SB) && words.get(wg.data(), words.n);
    const bool cleanup = ok && retired2 && std::memcmp(got.data(), host + 2 * SB, 2 * SB) == 0 &&
                         wg[e_pb] == addr(4) && wg[e_pf] == addr(5);
    printf("ex-test: me2b lent region, a failed prompt: rows while held %s; the refill %s\n",
           retired2 ? "none into the region" : "POINT INTO IT",
           cleanup ? "after the failed write, rows restored" : "BEFORE IT OR NOT RESTORED");
    check(c, cleanup, "me2b lent region failure cleanup");
    if (!err.empty()) {
        printf("ex-test: me2b lent region: %s\n", err.c_str());
    }
    for (cudaEvent_t e : {pub, t0, t1, w0, w1, up, er, lay[0], lay[1], lay[2]}) {
        cudaEventDestroy(e);
    }
    for (cudaStream_t s : {cs, ss, cp}) {
        cudaStreamDestroy(s);
    }
    cudaFreeHost(host);
    cudaFreeHost(idx);
    cudaFreeHost(val);
    cudaFreeHost(rows_h);
}

// ME1: resolve and apply (experts_cache.h) against the host's expectation: a residency row with every third expert
// resident (addresses that are never read), routings all resident, mixed, and with an invalid ID, at B 1 and 5 and with
// 2 and 4 parts; tab and tab2 entries of the routed experts written, every other entry kept; the flags only when every
// expert is resident; the slot and run words zeroed. apply: 4,096 pairs, the others kept.
void me1_resolve(test_ctx & c) {
    constexpr size_t n_words = size_t(NE) + 8 * NE + 2;
    dev_mem words(n_words * 8), flags(8), ids(size_t(max_rows) * 10 * 4);
    std::vector<uint64_t> res(NE, 0), init(n_words);
    for (int32_t e = 0; e < NE; e += 3) {
        res[size_t(e)] = 0x700000000000ull + uint64_t(e) * 0x10000ull;
    }
    std::mt19937 r(0x4d453152u);
    for (auto & v : init) {
        v = (uint64_t(r()) << 32) | r();
    }
    const uint64_t zero = 0x7f0000000000ull;
    bool all_ok = true;
    for (int32_t parts : {2, 4}) {
        for (int32_t rows : {1, 5}) {
            for (int32_t kind = 0; kind < 3; ++kind) { // all resident, mixed, an invalid ID
                std::vector<int32_t> id(size_t(rows) * 10);
                for (int32_t b = 0; b < rows; ++b) {
                    std::set<int32_t> used;
                    for (int32_t k = 0; k < 10; ++k) {
                        int32_t e = 0;
                        do {
                            e = int32_t(r() % NE);
                            if (kind == 0) {
                                e -= e % 3;
                            }
                        } while (used.count(e) != 0);
                        used.insert(e);
                        id[size_t(b) * 10 + size_t(k)] = e;
                    }
                }
                if (kind == 1) {
                    id[0] = id[0] - id[0] % 3 + 1 < NE ? id[0] - id[0] % 3 + 1 : 1; // at least one miss
                }
                if (kind == 2) {
                    id[size_t(rows) * 10 - 1] = NE + 7;
                }
                std::vector<uint64_t> w = init;
                std::copy(res.begin(), res.end(), w.begin());
                const uint32_t seq = 1000u + uint32_t(parts * 10 + rows * 3 + kind);
                const uint32_t f0[2] = {0xdead0000u, 0xdead0001u}; // ME3: resolve writes both flags in every case
                if (!check(c, words.put(w.data(), words.n) && flags.put(f0, 8) && ids.put(id.data(), id.size() * 4),
                           "me1 resolve: inputs")) {
                    return;
                }
                fn::experts::resolve_args a{};
                a.ids = ids.as<int32_t>();
                a.rows = rows;
                a.res = words.as<const uint8_t * const>();
                for (int32_t k = 0; k < 4; ++k) {
                    a.off[k] = 4096 * k + 256;
                }
                a.parts = parts;
                a.zero = reinterpret_cast<const uint8_t *>(zero);
                a.tab = reinterpret_cast<const uint8_t **>(words.as<uint64_t>() + NE);
                a.tab2 = reinterpret_cast<const uint8_t **>(words.as<uint64_t>() + NE + 4 * NE);
                a.flags = flags.as<uint32_t>();
                a.set = seq;
                a.ext_slots = words.as<uint64_t>() + NE + 8 * NE;
                a.run2 = a.ext_slots + 1;
                std::vector<uint64_t> got(n_words);
                uint32_t fl[2] = {0, 0};
                if (!check(c,
                           fn::experts::resolve(a, cudaStreamPerThread) &&
                               cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "resolve") &&
                               words.get(got.data(), words.n) && flags.get(fl, 8),
                           "me1 resolve: launch")) {
                    return;
                }
                std::vector<uint64_t> want = w;
                bool any_miss = false;
                for (const int32_t e : id) {
                    if (e < 0 || e >= NE) {
                        any_miss = true;
                        continue;
                    }
                    const uint64_t re = res[size_t(e)];
                    any_miss = any_miss || re == 0;
                    for (int32_t k = 0; k < parts; ++k) {
                        want[size_t(NE + k * NE + e)] = re != 0 ? re + uint64_t(a.off[k]) : (k & 1) != 0 ? zero : 0;
                        want[size_t(NE + 4 * NE + k * NE + e)] = 0;
                    }
                }
                want[size_t(NE + 8 * NE)] = 0;
                want[size_t(NE + 8 * NE + 1)] = 0;
                const bool same = got == want && fl[0] == (any_miss ? 0u : seq) && fl[1] == fl[0];
                all_ok = all_ok && same;
                check(c, same,
                      "me1 resolve: parts " + std::to_string(parts) + " B " + std::to_string(rows) + " kind " +
                          std::to_string(kind));
            }
        }
    }
    // apply: distinct indices, values in order
    std::vector<int32_t> idx(size_t(fn::experts::max_apply));
    std::vector<uint64_t> val(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        idx[i] = int32_t((i * 7 + 3) % n_words);
        val[i] = (uint64_t(r()) << 32) | r();
    }
    dev_mem di(idx.size() * 4), dv(val.size() * 8);
    std::vector<uint64_t> got(n_words), want = init;
    for (size_t i = 0; i < idx.size(); ++i) {
        want[size_t(idx[i])] = val[i];
    }
    const bool applied = words.put(init.data(), words.n) && di.put(idx.data(), di.n) && dv.put(val.data(), dv.n) &&
                         fn::experts::apply(words.as<uint64_t>(), di.as<int32_t>(), dv.as<uint64_t>(),
                                            int32_t(idx.size()), cudaStreamPerThread) &&
                         cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "apply") &&
                         words.get(got.data(), words.n) && got == want;
    check(c, applied, "me1 apply");
    printf("ex-test: me1 resolve (12 cases: parts 2 and 4, B 1 and 5, all resident, mixed, invalid ID) %s; apply "
           "(%zu pairs) %s\n",
           all_ok ? "as expected" : "DIFFER", idx.size(), applied ? "as expected" : "DIFFER");
}

// ME3 (the graph-replay arm): the cached graph executor's premise on the cache's own kernels. resolve, captured once
// for B 1 and once for B 5 as the executor captures it (4 parts, one card; the flag set fixed at capture), is
// replayed 8 times each with changed routing (all resident, mixed, an invalid ID) and changed residency (64 experts
// a replay moved to new addresses or evicted: new slots and generations, uploaded by apply on the same stream before
// the replay as the cache's row patches are), the outputs junk before each replay; every replay's tables and flags
// against the host's expectation for the current inputs, with no capture, instantiation or update after the first
void me3_graph_replay(test_ctx & c) {
    constexpr size_t n_words = size_t(NE) + 8 * NE + 2;
    constexpr int32_t parts = 4, n_replay = 8;
    dev_mem words(n_words * 8), flags(8), ids(size_t(max_rows) * 10 * 4), di(size_t(NE) * 4), dv(size_t(NE) * 8);
    cudaStream_t cs = nullptr;
    if (!check(c, cuda_ok(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream") && words.p != nullptr &&
                      flags.p != nullptr && ids.p != nullptr && di.p != nullptr && dv.p != nullptr,
               "me3 graph replay: setup")) {
        return;
    }
    std::mt19937 r(0x4d453352u);
    std::vector<uint64_t> res(NE, 0);
    uint64_t gen = 1;
    for (int32_t e = 0; e < NE; e += 2) {
        res[size_t(e)] = 0x700000000000ull + (gen++ << 16);
    }
    const uint64_t zero = 0x7f0000000000ull;
    const uint32_t seq = 0x5e3u;
    int32_t n_inst = 0, n_launch = 0, n_bad = 0;
    bool ok = true;
    for (int32_t rows : {1, 5}) {
        fn::experts::resolve_args a{};
        a.ids = ids.as<int32_t>();
        a.rows = rows;
        a.res = words.as<const uint8_t * const>();
        for (int32_t k = 0; k < 4; ++k) {
            a.off[k] = 4096 * k + 256;
        }
        a.parts = parts;
        a.zero = reinterpret_cast<const uint8_t *>(zero);
        a.tab = reinterpret_cast<const uint8_t **>(words.as<uint64_t>() + NE);
        a.tab2 = reinterpret_cast<const uint8_t **>(words.as<uint64_t>() + NE + 4 * NE);
        a.flags = flags.as<uint32_t>();
        a.set = seq;
        a.ext_slots = words.as<uint64_t>() + NE + 8 * NE;
        a.run2 = a.ext_slots + 1;
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        if (!check(c,
                   cuda_ok(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "capture") &&
                       fn::experts::resolve(a, cs) && cuda_ok(cudaStreamEndCapture(cs, &g), "capture") &&
                       cuda_ok(cudaGraphInstantiate(&ge, g, 0), "instantiate"),
                   "me3 graph replay: capture B " + std::to_string(rows))) {
            cudaStreamDestroy(cs);
            return;
        }
        ++n_inst;
        std::vector<uint64_t> words_h(n_words);
        for (auto & v : words_h) {
            v = (uint64_t(r()) << 32) | r();
        }
        std::copy(res.begin(), res.end(), words_h.begin());
        ok = ok && words.put(words_h.data(), words.n);
        for (int32_t q = 0; q < n_replay && ok; ++q) {
            // the residency's changes: 64 distinct experts each moved to a new address (a new generation) or evicted
            std::vector<int32_t> idx;
            std::vector<uint64_t> val;
            std::set<int32_t> moved;
            while (moved.size() < 64) {
                moved.insert(int32_t(r() % NE));
            }
            for (const int32_t e : moved) {
                res[size_t(e)] = (r() & 3) == 0 ? 0 : 0x700000000000ull + (gen++ << 16);
                idx.push_back(e);
                val.push_back(res[size_t(e)]);
            }
            const int32_t kind = q % 3; // all resident, mixed, an invalid ID
            std::vector<int32_t> id(size_t(rows) * 10);
            for (int32_t b = 0; b < rows; ++b) {
                std::set<int32_t> used;
                for (int32_t k = 0; k < 10; ++k) {
                    int32_t e = 0;
                    do {
                        e = int32_t(r() % NE);
                    } while (used.count(e) != 0 || (kind == 0 && res[size_t(e)] == 0));
                    used.insert(e);
                    id[size_t(b) * 10 + size_t(k)] = e;
                }
            }
            if (kind == 1) {
                for (int32_t e = 0; e < NE; ++e) {
                    if (res[size_t(e)] == 0 && std::find(id.begin(), id.end(), e) == id.end()) {
                        id[0] = e; // at least one miss
                        break;
                    }
                }
            }
            if (kind == 2) {
                id[size_t(rows) * 10 - 1] = NE + 7;
            }
            // junk in the outputs, so the replay must write what it reports
            std::vector<uint64_t> junk(n_words - size_t(NE));
            for (auto & v : junk) {
                v = (uint64_t(r()) << 32) | r();
            }
            const uint32_t f0[2] = {0xdead0000u, 0xdead0001u};
            ok = ok && check(c,
                             cuda_ok(cudaMemcpyAsync(words.p + size_t(NE) * 8, junk.data(), junk.size() * 8,
                                                     cudaMemcpyHostToDevice, cs),
                                     "junk") &&
                                 cuda_ok(cudaMemcpyAsync(flags.p, f0, 8, cudaMemcpyHostToDevice, cs), "flags") &&
                                 cuda_ok(cudaMemcpyAsync(ids.p, id.data(), id.size() * 4, cudaMemcpyHostToDevice, cs),
                                         "ids") &&
                                 cuda_ok(cudaMemcpyAsync(di.p, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice, cs),
                                         "idx") &&
                                 cuda_ok(cudaMemcpyAsync(dv.p, val.data(), val.size() * 8, cudaMemcpyHostToDevice, cs),
                                         "val") &&
                                 fn::experts::apply(words.as<uint64_t>(), di.as<int32_t>(), dv.as<uint64_t>(),
                                                    int32_t(idx.size()), cs) &&
                                 cuda_ok(cudaGraphLaunch(ge, cs), "replay") &&
                                 cuda_ok(cudaStreamSynchronize(cs), "replay"),
                             "me3 graph replay: replay");
            if (!ok) {
                break;
            }
            ++n_launch;
            std::vector<uint64_t> got(n_words), want(n_words);
            uint32_t fl[2] = {0, 0};
            ok = words.get(got.data(), words.n) && flags.get(fl, 8);
            std::copy(res.begin(), res.end(), want.begin());
            std::copy(junk.begin(), junk.end(), want.begin() + NE);
            bool any_miss = false;
            for (const int32_t e : id) {
                if (e < 0 || e >= NE) {
                    any_miss = true;
                    continue;
                }
                const uint64_t re = res[size_t(e)];
                any_miss = any_miss || re == 0;
                for (int32_t k = 0; k < parts; ++k) {
                    want[size_t(NE + k * NE + e)] = re != 0 ? re + uint64_t(a.off[k]) : (k & 1) != 0 ? zero : 0;
                    want[size_t(NE + 4 * NE + k * NE + e)] = 0;
                }
            }
            want[size_t(NE + 8 * NE)] = 0;
            want[size_t(NE + 8 * NE + 1)] = 0;
            const bool same = ok && got == want && fl[0] == (any_miss ? 0u : seq) && fl[1] == fl[0];
            n_bad += same ? 0 : 1;
            check(c, same, "me3 graph replay: B " + std::to_string(rows) + " replay " + std::to_string(q));
        }
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    cudaStreamDestroy(cs);
    printf("ex-test: me3 graph replay (resolve captured once a B, B 1 and 5; each replay new routes and 64 experts "
           "moved or evicted): %d instantiations, %d replays, %d differ from the current inputs' expectation: %s\n",
           n_inst, n_launch, n_bad, ok && n_bad == 0 && n_launch == 2 * n_replay ? "as expected" : "DIFFER");
}

// ME3 (the deadlock fix): a replay's waits and stamp as kernels. One graph captured once: wait_word on a mapped host
// word for the value 3 (its times kept), post_word of 7 to a mapped stamp, wait_word on the word for the target a
// second mapped word holds. Replayed twice: first with the word below both targets (the replay holds before the
// stamp, the host raises the word to 3: the stamp lands and the replay holds again; then to 5: it completes), then
// with both targets met at launch (it completes at once). The host's waits are bounded (a hang fails the check)
void me3_wait_post(test_ctx & c) {
    uint32_t * w = nullptr;
    uint64_t * t = nullptr;
    cudaStream_t cs = nullptr;
    if (!check(c,
               cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&w), 16, cudaHostAllocMapped), "mapped words") &&
                   cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&t), 16, cudaHostAllocMapped), "mapped times") &&
                   cuda_ok(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream"),
               "me3 wait/post: setup")) {
        cudaFreeHost(w);
        cudaFreeHost(t);
        return;
    }
    volatile uint32_t * const vw = w; // [0] the word, [1] the second wait's target, [2] the stamp
    volatile uint64_t * const vt = t;
    uint32_t * dw = nullptr;
    uint64_t * dt = nullptr;
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    bool ok = cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dw), w, 0), "word address") &&
              cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dt), t, 0), "times address") &&
              cuda_ok(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "capture") &&
              fn::experts::wait_word(fn::experts::wait_args{dw, nullptr, 3, dt}, cs) &&
              fn::experts::post_word(dw + 2, 7, cs) &&
              fn::experts::wait_word(fn::experts::wait_args{dw, dw + 1, 0, nullptr}, cs) &&
              cuda_ok(cudaStreamEndCapture(cs, &g), "capture") &&
              cuda_ok(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    check(c, ok, "me3 wait/post: capture");
    // the host's bounded wait for cond (true when it held within the bound)
    const auto until = [](const std::function<bool()> & cond) {
        for (int32_t i = 0; i < 20000; ++i) {
            if (cond()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        return false;
    };
    bool held = false, stamped = false, held2 = false, done = false, timed = false, done2 = false;
    if (ok) {
        vw[0] = 0;
        vw[1] = 5;
        vw[2] = 0;
        vt[0] = 0;
        vt[1] = 0;
        ok = cuda_ok(cudaGraphLaunch(ge, cs), "replay");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        held = ok && vw[2] == 0 && cudaStreamQuery(cs) == cudaErrorNotReady;
        vw[0] = 3;
        stamped = ok && until([&] { return vw[2] == 7; });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        held2 = stamped && cudaStreamQuery(cs) == cudaErrorNotReady;
        vw[0] = 5;
        done = ok && until([&] { return cudaStreamQuery(cs) != cudaErrorNotReady; }) &&
               cuda_ok(cudaStreamSynchronize(cs), "replay");
        timed = done && vt[0] != 0 && vt[1] >= vt[0];
        // both targets met at launch
        vw[2] = 0;
        done2 = done && cuda_ok(cudaGraphLaunch(ge, cs), "replay") &&
                until([&] { return cudaStreamQuery(cs) != cudaErrorNotReady; }) &&
                cuda_ok(cudaStreamSynchronize(cs), "replay") && vw[2] == 7;
    }
    check(c, held, "me3 wait/post: the replay holds before the stamp");
    check(c, stamped, "me3 wait/post: the stamp after the first wait");
    check(c, held2, "me3 wait/post: the replay holds at the second wait");
    check(c, done, "me3 wait/post: the replay completes");
    check(c, timed, "me3 wait/post: the wait's times");
    check(c, done2, "me3 wait/post: targets met at launch");
    // every wait released before the graph and the buffers go
    vw[0] = 5;
    vw[1] = 5;
    cudaStreamSynchronize(cs);
    if (ge != nullptr) {
        cudaGraphExecDestroy(ge);
    }
    if (g != nullptr) {
        cudaGraphDestroy(g);
    }
    cudaStreamDestroy(cs);
    cudaFreeHost(w);
    cudaFreeHost(t);
    printf("ex-test: me3 wait/post (one capture, two replays): holds %d, stamp %d, holds at the second wait %d, "
           "completes %d, times %d, met at launch %d: %s\n",
           held, stamped, held2, done, timed, done2,
           held && stamped && held2 && done && timed && done2 ? "as expected" : "DIFFER");
}

// ---- SP4: the candidate mapping (sp4_map: H1's 1, H2's 2, XB3 C1's 3) against SP2's 640-thread kernels -----------
bool same_gu(const gu_out & a, const gu_out & b) {
    return a.glu[0] == b.glu[0] && a.glu[1] == b.glu[1] && a.glu_f32[0] == b.glu_f32[0] &&
           a.glu_f32[1] == b.glu_f32[1];
}

// one case of an instance at B 4 or 5 in the candidate mapping: the FP64 contract and the GLU codes and scales
// (check_gate_up), then the whole GLU buffers (codes, scales, F32: every routed pair's and the shared rows) of both
// cards bitwise against SP2's 640-thread kernels' (base, or run here)
bool sp4_case(test_ctx & c, const gu_inst_t & g, const row_case & rc, const char * label, gu_out & p,
              const gu_out * base = nullptr) {
    if (!check_gate_up(c, g.f, g.fg, g.fu, rc, false, label, p, sp4_map)) {
        return false;
    }
    gu_out b;
    bool ran = true;
    if (base == nullptr) {
        for (int32_t card = 0; card < 2 && ran; ++card) {
            uint32_t st = 0;
            ran = run_gate_up(c, g.f, g.fg, g.fu, card, rc, b.glu[card], b.glu_f32[card], st) && st == 0;
        }
        base = &b;
    }
    const bool same = ran && same_gu(p, *base);
    printf("ex-test: %s %-22s B %d U %2d: %s against the 640-thread kernels: GLU buffers %s\n", label,
           gu_name(g.f, g.fg, g.fu).c_str(), rc.rows, rc.m.n_u, map_tag(sp4_map), same ? "bitwise equal" : "DIFFER");
    bool h2_same = true;
    if (sp4_map == 3 && is_iq2(g.f) && rc.rows >= 4) { // XB3 C1: map 3 leaves the IQ2 formats on H2's body
        gu_out h;
        for (int32_t card = 0; card < 2 && h2_same; ++card) {
            uint32_t st = 0;
            h2_same = run_gate_up(c, g.f, g.fg, g.fu, card, rc, h.glu[card], h.glu_f32[card], st, {2}) && st == 0;
        }
        h2_same = h2_same && same_gu(p, h);
        printf("ex-test: %s %-22s B %d U %2d: xb3 against h2: GLU buffers %s\n", label,
               gu_name(g.f, g.fg, g.fu).c_str(), rc.rows, rc.m.n_u, h2_same ? "bitwise equal" : "DIFFER");
    }
    return check(c, same && h2_same,
                 std::string(label) + " " + gu_name(g.f, g.fg, g.fu) + " B " + std::to_string(rc.rows) + ": " +
                     map_tag(sp4_map) + " bitwise against the 640-thread kernels" +
                     (sp4_map == 3 && is_iq2(g.f) && rc.rows >= 4 ? " and h2" : ""));
}

// SP4: case rc in X-HMIX form on card `card`: slot words of xchg::hmix_layout, rows < B's codes and scales from the
// case (F32 words and rows >= B the sentinels); stale: the peer half's last run of the last row left at its sentinels
std::vector<uint32_t> hmix_slot(const row_case & rc, int32_t card, bool stale) {
    using L = fn::xchg::hmix_layout;
    std::vector<uint32_t> w(size_t(L::slot_words), fn::xchg::sentinel);
    std::fill(w.begin() + L::code_at, w.begin() + L::scale_at, fn::xchg::code_sentinel);
    const size_t sc_at = fn::act::codes_bytes(rc.rows, D);
    for (int32_t b = 0; b < rc.rows; ++b) {
        std::memcpy(reinterpret_cast<uint8_t *>(w.data() + L::code_at) + size_t(b) * L::width,
                    rc.xq.data() + size_t(b) * D, size_t(D));
        std::memcpy(w.data() + L::scale_at + size_t(b) * L::runs * 2, rc.xq.data() + sc_at + size_t(b) * L::runs * 8,
                    size_t(L::runs) * 8);
    }
    if (stale) {
        const int32_t j = (1 - card) * (L::runs / 2) + L::runs / 2 - 1; // the peer half's last run
        uint32_t * cw = w.data() + L::code_at + (size_t(rc.rows - 1) * L::width + 32 * size_t(j)) / 4;
        std::fill(cw, cw + 8, fn::xchg::code_sentinel);
    }
    return w;
}

// SP4: X-HMIX's acquire in both mappings (SP2's and the candidate's) at a case of B 4 or 5. A complete slot: the GLU
// bitwise the direct input's (direct), the error word 0. stale: the poll gives up (err_poll_hmix set, the launch
// completes) in both mappings, with the same GLU. Both cards.
bool sp4_hmix(test_ctx & c, const gu_inst_t & g, const row_case & rc, const gu_out & direct, bool stale) {
    gu_out o[2];
    uint32_t errs[2][2] = {};
    bool ran = true;
    dev_mem slot(size_t(fn::xchg::hmix_layout::slot_words) * 4), err(4);
    for (int32_t k = 0; k < 2 && ran; ++k) {
        const int32_t map = k == 0 ? 0 : sp4_map;
        for (int32_t card = 0; card < 2 && ran; ++card) {
            const std::vector<uint32_t> w = hmix_slot(rc, card, stale);
            uint32_t st = 0;
            ran = slot.put(w.data(), slot.n) &&
                  run_gate_up(c, g.f, g.fg, g.fu, card, rc, o[k].glu[card], o[k].glu_f32[card], st,
                              {map, slot.as<float>(), err.as<uint32_t>()}) &&
                  err.get(&errs[k][card], 4) && st == 0;
        }
    }
    const uint32_t want = stale ? fn::xchg::err_poll_hmix : 0u;
    const bool errs_ok = errs[0][0] == want && errs[0][1] == want && errs[1][0] == want && errs[1][1] == want;
    const bool maps_same = ran && same_gu(o[0], o[1]);
    const bool direct_same = ran && (stale || same_gu(o[0], direct));
    printf("ex-test: hmix %s %-22s B %d: error words %u %u (640) %u %u (%s), want %u; GLU %s against 640 %s%s\n",
           stale ? "stale" : "complete", gu_name(g.f, g.fg, g.fu).c_str(), rc.rows, errs[0][0], errs[0][1],
           errs[1][0], errs[1][1], map_tag(sp4_map), want, map_tag(sp4_map), maps_same ? "bitwise equal" : "DIFFER",
           stale ? "" : (direct_same ? ", both the direct input's" : ", NOT the direct input's"));
    return check(c, ran && errs_ok && maps_same && direct_same,
                 std::string("hmix ") + (stale ? "stale " : "complete ") + gu_name(g.f, g.fg, g.fu) + " B " +
                     std::to_string(rc.rows));
}

// ---- XB3's bitwise arm (XB2's): 71b92a9's gate_up and down kernels on the same inputs ------------------------------
// 71b92a9's argument layouts (its experts.h): this build's without the routing meta
struct base_gate_up_args {
    const uint8_t * w;
    const uint8_t * sh_gate;
    const uint8_t * sh_up;
    const uint8_t * grids;
    fn::act::view x;
    const int32_t * ids;
    fn::act::out_view glu;
    float * glu_f32;
    uint32_t * status;
    const float * hmix;
    int32_t hmix_peer_half;
    uint32_t * err;
};
struct base_down_args {
    const uint8_t * w;
    const uint8_t * sh_w;
    const uint8_t * grids;
    fn::act::view glu;
    const int32_t * ids;
    const float * wts;
    const float * sg;
    float * out;
    uint32_t * status;
};
static_assert(sizeof(base_gate_up_args) + 8 + sizeof(class_table) + 3 * sizeof(int32_t) == offsetof(gate_up_args, map) &&
                  sizeof(base_down_args) + 8 == offsetof(down_args, wt),
              "the arguments differ from 71b92a9's by the meta pointer (and gate_up's by its class table, the IQ2 "
              "body's formats, XB3; SP4's map after them; ME1's tables and CPU slots after those)");
constexpr int32_t base_grid = 72;        // 71b92a9's n_sm (experts_dev.cuh): both kernels launch one block an SM
constexpr int32_t base_gu_threads = 640; // its gu_threads and dn_threads
constexpr int32_t base_dn_threads = 512;
constexpr uint32_t poison_word = 0xa5a5a5a5u; // the outputs' fill before a launch

// 71b92a9's kernels from that build's cubins (DIR/experts*.sm_70.cubin, cuobjdump -xelf of its fnfl-drive), keyed by
// their demangled template arguments, for example "gate_up_kernel<0, 0, 4, 4, 2>" or "down_kernel<7, 7, 4, 2, 512>".
// Every launch gets the device's opt-in maximum of dynamic shared memory (the kernels have no static shared memory,
// and none reads the size), at least what each needs.
struct base_kernels {
    std::vector<cudaLibrary_t> libs;
    std::map<std::string, cudaKernel_t> k;
    int32_t smem = 0;
    base_kernels() = default;
    base_kernels(const base_kernels &) = delete;
    base_kernels & operator=(const base_kernels &) = delete;
    ~base_kernels() {
        for (cudaLibrary_t l : libs) {
            cudaLibraryUnload(l);
        }
    }
};

// "void fn::experts::(anonymous namespace)::gate_up_kernel<(fn::experts::fmt)0, ...>(...)" -> "gate_up_kernel<0, ...>"
std::string base_key(const std::string & demangled) {
    static const std::string ns = "void fn::experts::(anonymous namespace)::";
    if (demangled.rfind(ns, 0) != 0) {
        return {};
    }
    std::string k = demangled.substr(ns.size(), demangled.find(">(") - ns.size() + 1);
    if (k.rfind("gate_up_kernel<", 0) != 0 && k.rfind("down_kernel<", 0) != 0) {
        return {};
    }
    for (const std::string cast : {"(fn::experts::fmt)", "(fn::dense::fmt)"}) {
        for (size_t at = k.find(cast); at != std::string::npos; at = k.find(cast)) {
            k.erase(at, cast.size());
        }
    }
    return k;
}

bool load_base(const std::string & dir, int device, base_kernels & bk) {
    namespace fsys = std::filesystem;
    std::error_code ec;
    for (const auto & de : fsys::directory_iterator(dir, ec)) {
        const std::string name = de.path().filename().string();
        const std::string ext = ".sm_70.cubin";
        if (name.rfind("experts", 0) != 0 || name.size() <= ext.size() ||
            name.compare(name.size() - ext.size(), ext.size(), ext) != 0) {
            continue;
        }
        std::ifstream in(de.path(), std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        cudaLibrary_t lib = nullptr;
        if (!cuda_ok(cudaLibraryLoadFromFile(&lib, de.path().c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0),
                     name.c_str())) {
            return false;
        }
        bk.libs.push_back(lib);
        // the kernels' names from the cubin's strings: each NUL-terminated run from a mangled fn::experts name
        for (size_t at = bytes.find("_ZN2fn7experts"); at != std::string::npos;
             at = bytes.find("_ZN2fn7experts", at + 1)) {
            const std::string sym(bytes.c_str() + at);
            int st = 0;
            char * dm = abi::__cxa_demangle(sym.c_str(), nullptr, nullptr, &st);
            const std::string key = st == 0 && dm != nullptr ? base_key(dm) : std::string();
            std::free(dm);
            cudaKernel_t kern = nullptr;
            if (key.empty() || bk.k.count(key) != 0) {
                continue;
            }
            if (!cuda_ok(cudaLibraryGetKernel(&kern, lib, sym.c_str()), key.c_str())) {
                return false;
            }
            bk.k[key] = kern;
        }
    }
    if (ec || bk.k.empty() ||
        !cuda_ok(cudaDeviceGetAttribute(&bk.smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), "smem")) {
        fprintf(stderr, "xb3-bitwise: no base kernels in %s\n", dir.c_str());
        return false;
    }
    for (const auto & kv : bk.k) {
        if (!cuda_ok(cudaKernelSetAttributeForDevice(kv.second, cudaFuncAttributeMaxDynamicSharedMemorySize, bk.smem,
                                                     device),
                     kv.first.c_str())) {
            return false;
        }
    }
    printf("xb3-bitwise: %zu base kernels from %s, %d B dynamic shared memory each\n", bk.k.size(), dir.c_str(),
           bk.smem);
    return true;
}

bool base_launch(const base_kernels & bk, const std::string & key, int32_t threads, void * args) {
    const auto it = bk.k.find(key);
    if (it == bk.k.end()) {
        fprintf(stderr, "xb3-bitwise: no base kernel %s\n", key.c_str());
        return false;
    }
    void * params[] = {args};
    return cuda_ok(cudaLaunchKernel(reinterpret_cast<const void *>(it->second), dim3(base_grid), dim3(threads), params,
                                    size_t(bk.smem), cudaStreamPerThread),
                   key.c_str()) &&
           cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), key.c_str());
}

// the first differing byte of two buffers as "what offset N", or empty
std::string first_diff(const char * what, const void * a, const void * b, size_t n) {
    const auto * x = static_cast<const uint8_t *>(a);
    const auto * y = static_cast<const uint8_t *>(b);
    for (size_t i = 0; i < n; ++i) {
        if (x[i] != y[i]) {
            return std::string(what) + " offset " + std::to_string(i);
        }
    }
    return {};
}

// every F32 GLU value of the shared expert's rows (slots n_slots + b) written
bool shared_written(const std::vector<float> & glu_f32, const row_case & rc) {
    bool ok = true;
    for (int32_t i = 0; i < rc.rows * n_ff_card; ++i) {
        uint32_t v = 0;
        std::memcpy(&v, &glu_f32[size_t(rc.m.n_slots) * n_ff_card + size_t(i)], 4);
        ok = ok && v != poison_word;
    }
    return ok;
}

// The base gate_up of one card on rc: the GLU buffer, its F32 values and the status
bool run_base_gate_up(test_ctx & c, const base_kernels & bk, const gu_inst_t & g, int32_t card, const row_case & rc,
                      std::vector<uint8_t> & glu, std::vector<float> & glu_f32, uint32_t & status) {
    const dev_mem * w = dev_packed(c, g.f, false, card);
    dev_mem x(rc.xq.size()), ids(rc.ids.size() * 4), gb(glu_buf_bytes), gf(size_t(glu_slots) * n_ff_card * 4), st(4);
    if (w == nullptr || !x.put(rc.xq.data(), rc.xq.size()) || !ids.put(rc.ids.data(), ids.n) ||
        !cuda_ok(cudaMemset(gb.p, 0xa5, gb.n), "memset") || !cuda_ok(cudaMemset(gf.p, 0xa5, gf.n), "memset") ||
        !cuda_ok(cudaMemset(st.p, 0, 4), "memset")) {
        return false;
    }
    base_gate_up_args a = {w->p,
                           c.sh_gu.at(int32_t(g.fg)).d[card]->p,
                           c.sh_gu.at(int32_t(g.fu)).d[card]->p,
                           c.d_grids.p,
                           act_in(x.p, rc.rows, D),
                           ids.as<int32_t>(),
                           act_out(gb.p, glu_slots, n_ff_card),
                           gf.as<float>(),
                           st.as<uint32_t>(),
                           nullptr,
                           0,
                           nullptr};
    const std::string key = "gate_up_kernel<" + std::to_string(int32_t(g.f)) + ", " + std::to_string(int32_t(g.fg)) +
                            ", " + std::to_string(int32_t(g.fu)) + ", " + std::to_string(rc.rows) + ", 2>";
    glu.assign(gb.n, 0);
    glu_f32.assign(size_t(glu_slots) * n_ff_card, 0.0f);
    return base_launch(bk, key, base_gu_threads, &a) && gb.get(glu.data(), gb.n) && gf.get(glu_f32.data(), gf.n) &&
           st.get(&status, 4);
}

bool run_base_down(test_ctx & c, const base_kernels & bk, const dn_inst_t & d, int32_t card, const row_case & rc,
                   const std::vector<uint8_t> & glu, std::vector<float> & out, uint32_t & status) {
    const dev_mem * w = dev_packed(c, d.f, true, card);
    dev_mem g(glu.size()), ids(rc.ids.size() * 4), wts(rc.wts.size() * 4), sg(rc.sg.size() * 4),
        o(size_t(rc.rows) * D * 4), st(4);
    if (w == nullptr || !g.put(glu.data(), glu.size()) || !ids.put(rc.ids.data(), ids.n) ||
        !wts.put(rc.wts.data(), wts.n) || !sg.put(rc.sg.data(), sg.n) ||
        !cuda_ok(cudaMemset(o.p, 0xa5, o.n), "memset") || !cuda_ok(cudaMemset(st.p, 0, 4), "memset")) {
        return false;
    }
    base_down_args a = {w->p,
                        c.sh_dn.at(int32_t(dense_of(d.fs))).d[card]->p,
                        c.d_grids.p,
                        act_in(g.p, glu_slots, n_ff_card),
                        ids.as<int32_t>(),
                        wts.as<float>(),
                        sg.as<float>(),
                        o.as<float>(),
                        st.as<uint32_t>()};
    const std::string key = "down_kernel<" + std::to_string(int32_t(d.f)) + ", " + std::to_string(int32_t(d.fs)) +
                            ", " + std::to_string(rc.rows) + ", " + (d.f == fmt::q2_0 ? "3" : "2") + ", 512>";
    out.assign(size_t(rc.rows) * D, 0.0f);
    return base_launch(bk, key, base_dn_threads, &a) && o.get(out.data(), o.n) && st.get(&status, 4);
}

// XB3's bitwise arm (XB2's xb2-bitwise): every instantiation under test at every case, both cards, this build's launch
// (its meta by write_meta) against 71b92a9's on the same inputs, every output byte: gate_up's GLU buffer (codes, scales
// and the unwritten slots' fill), its F32 values and status; down's output and status (on this build's GLU of the first
// gate_up instantiation). The shared expert's rows are checked written in both. Prints "xb3-bitwise: PASS" or the
// first differing buffer and offset.
bool xb2_bitwise(test_ctx & c, int device, const std::string & dir, const std::vector<gu_inst_t> & gus,
                 const std::vector<dn_inst_t> & dns, const std::vector<const row_case *> & cases) {
    base_kernels bk;
    if (!load_base(dir, device, bk)) {
        printf("xb3-bitwise: FAIL (no base kernels)\n");
        return false;
    }
    std::vector<std::vector<uint8_t>> glu_in(cases.size() * 2); // [case][card]: down's input
    int32_t n_cmp = 0;
    std::string diff;
    for (size_t gi = 0; gi < gus.size() && diff.empty(); ++gi) {
        const gu_inst_t & g = gus[gi];
        for (size_t ci = 0; ci < cases.size() && diff.empty(); ++ci) {
            const row_case & rc = *cases[ci];
            for (int32_t card = 0; card < 2 && diff.empty(); ++card) {
                std::vector<uint8_t> gn, gb;
                std::vector<float> fn32, fb32;
                uint32_t sn = 0, sb = 0;
                const std::string at = gu_name(g.f, g.fg, g.fu) + " B " + std::to_string(rc.rows) + " U " +
                                       std::to_string(rc.m.n_u) + " card " + std::to_string(card) + ": ";
                if (!run_gate_up(c, g.f, g.fg, g.fu, card, rc, gn, fn32, sn) ||
                    !run_base_gate_up(c, bk, g, card, rc, gb, fb32, sb)) {
                    diff = at + "a launch failed";
                    break;
                }
                diff = first_diff("GLU", gn.data(), gb.data(), gn.size());
                diff = diff.empty() ? first_diff("GLU F32", fn32.data(), fb32.data(), fn32.size() * 4) : diff;
                diff = diff.empty() && sn != sb ? std::string("status") : diff;
                diff = diff.empty() && !(shared_written(fn32, rc) && shared_written(fb32, rc))
                           ? std::string("shared rows unwritten")
                           : diff;
                diff = diff.empty() ? diff : at + diff;
                ++n_cmp;
                if (gi == 0) {
                    glu_in[ci * 2 + size_t(card)] = gn;
                }
            }
        }
    }
    for (size_t di = 0; di < dns.size() && diff.empty(); ++di) {
        const dn_inst_t & d = dns[di];
        for (size_t ci = 0; ci < cases.size() && diff.empty(); ++ci) {
            const row_case & rc = *cases[ci];
            for (int32_t card = 0; card < 2 && diff.empty(); ++card) {
                std::vector<float> on, ob;
                uint32_t sn = 0, sb = 0;
                const std::string at = std::string("down ") + fmt_name(d.f) + "+" + fmt_name(d.fs) + " B " +
                                       std::to_string(rc.rows) + " U " + std::to_string(rc.m.n_u) + " card " +
                                       std::to_string(card) + ": ";
                const std::vector<uint8_t> & gin = glu_in[ci * 2 + size_t(card)];
                if (!run_down(c, d.f, d.fs, card, rc, gin, on, sn) || !run_base_down(c, bk, d, card, rc, gin, ob, sb)) {
                    diff = at + "a launch failed";
                    break;
                }
                diff = first_diff("output", on.data(), ob.data(), on.size() * 4);
                diff = diff.empty() && sn != sb ? std::string("status") : diff;
                diff = diff.empty() ? diff : at + diff;
                ++n_cmp;
            }
        }
    }
    if (!diff.empty()) {
        printf("xb3-bitwise: FAIL at %s\n", diff.c_str());
        return false;
    }
    std::set<int32_t> rows; // the kernels compared: each instantiation at each B the cases hold
    for (const row_case * rc : cases) {
        rows.insert(rc->rows);
    }
    printf("xb3-bitwise: PASS (%zu gate_up and %zu down instantiations at %zu B, %zu and %zu kernels, %zu cases, both "
           "cards: %d launches each byte-identical to 71b92a9's)\n",
           gus.size(), dns.size(), rows.size(), gus.size() * rows.size(), dns.size() * rows.size(), cases.size(),
           n_cmp);
    return true;
}

// XB4's bitwise arm, on XB3's harness (XB5): every down instantiation at every case, both cards, this build's launch
// (its meta by write_meta) against 71b92a9's on the same inputs (this build's GLU of the first gate_up instantiation on
// the case), every output byte and the status. Prints "xb4-bitwise: PASS" or the first differing case and offset.
bool xb4_bitwise(test_ctx & c, int device, const std::string & dir, const gu_inst_t & g,
                 const std::vector<dn_inst_t> & dns, const std::vector<const row_case *> & cases) {
    base_kernels bk;
    if (!load_base(dir, device, bk)) {
        printf("xb4-bitwise: FAIL (no base kernels)\n");
        return false;
    }
    std::vector<std::vector<uint8_t>> glu_in(cases.size() * 2); // [case][card]
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        for (int32_t card = 0; card < 2; ++card) {
            std::vector<float> gf32;
            uint32_t status = 0;
            if (!run_gate_up(c, g.f, g.fg, g.fu, card, *cases[ci], glu_in[ci * 2 + size_t(card)], gf32, status) ||
                status != 0) {
                printf("xb4-bitwise: FAIL (gate_up B %d U %d card %d)\n", cases[ci]->rows, cases[ci]->m.n_u, card);
                return false;
            }
        }
    }
    int32_t n_cmp = 0;
    std::string diff;
    for (size_t di = 0; di < dns.size() && diff.empty(); ++di) {
        const dn_inst_t & d = dns[di];
        for (int32_t card = 0; card < 2 && diff.empty(); ++card) {
            for (size_t ci = 0; ci < cases.size() && diff.empty(); ++ci) {
                const row_case & rc = *cases[ci];
                std::vector<float> on, ob;
                uint32_t sn = 0, sb = 0;
                const std::string at = std::string("down ") + fmt_name(d.f) + "+" + fmt_name(d.fs) + " B " +
                                       std::to_string(rc.rows) + " U " + std::to_string(rc.m.n_u) + " card " +
                                       std::to_string(card) + ": ";
                const std::vector<uint8_t> & gin = glu_in[ci * 2 + size_t(card)];
                if (!run_down(c, d.f, d.fs, card, rc, gin, on, sn) || !run_base_down(c, bk, d, card, rc, gin, ob, sb)) {
                    diff = at + "a launch failed";
                    break;
                }
                diff = first_diff("output", on.data(), ob.data(), on.size() * 4);
                diff = diff.empty() && sn != sb ? std::string("status") : diff;
                diff = diff.empty() ? diff : at + diff;
                ++n_cmp;
            }
        }
    }
    if (!diff.empty()) {
        printf("xb4-bitwise: FAIL at %s\n", diff.c_str());
        return false;
    }
    printf("xb4-bitwise: PASS (%zu down instantiations, %zu cases, both cards: %d launches each byte-identical to "
           "71b92a9's)\n",
           dns.size(), cases.size(), n_cmp);
    return true;
}


// ---- ME8: one router for both logical cards on one device and stream (fn/model's ffn_pair and ffn_pair_resident, the
// draft's draft_pair_enqueue) -----------------------------------------------------------------------------------------
// Synthetic, no model. The router is fn::dense::product with its top-10 epilogue (BF16 weights for the target, F32 for
// the draft) on random weights: nonzero rows for the 64 IDs that read a synthetic expert (pool, pool_x, pool_y), zero
// rows for the others, and 16 pairs of identical rows over distinct synthetic experts (ties inside the top 10 and at
// its cut). One graph a case, captured once: per group (the target: one launch of B rows; the draft: P proposals of one
// row each, in turn, as its step graph) card 0's router, card 0's experts on its routing, card 1's experts on card 0's
// (use_route, the one function fn/model takes), each router's IDs, weights, shared gate and logits copied to a log.
// Replayed with new activations each time, and each replay held to the eager two-router form run after it (card 1's own
// router on the same weights and activations into its own buffers, its experts on those): every group's routing equal,
// both cards' outputs bitwise equal, statuses 0. Card 1's own routing buffers hold the previous replay's last route
// when the graph runs, so a launch that read them (a stale second-card buffer) differs; every group's IDs change from
// one replay to the next. cached: the experts through the cache's 512-entry address tables (wt, every routed expert a
// hit), the form ffn_pair's service launches; the service's resolve and its split passes read ga.ids, which use_route
// sets, and are covered by ME1's and ME3's checks, not here.
struct me8_stats {
    int32_t cases = 0, replays = 0, groups = 0, rows = 0, bad = 0, unchanged = 0, ties_in = 0, ties_cut = 0;
};

uint16_t me8_bf16(float v) {
    uint32_t b = 0;
    std::memcpy(&b, &v, 4);
    return uint16_t((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

void me8_shared_route(test_ctx & c, const gu_inst_t & g, const dn_inst_t & d, bool draft, bool cached, int32_t n,
                      me8_stats & s) {
    constexpr int32_t n_replay = 6;
    const int32_t rows = draft ? 1 : n, groups = draft ? n : 1;
    const std::string name = std::string("me8 ") + (draft ? "draft " : "target ") + gu_name(g.f, g.fg, g.fu) + "+" +
                             fmt_name(d.f) + "+" + fmt_name(d.fs) + (cached ? " cached" : " resident") +
                             (draft ? " P " : " B ") + std::to_string(n);
    ++s.cases;
    std::mt19937 rng(0x4d453800u + uint32_t(n) * 16 + (draft ? 8 : 0) + (cached ? 4 : 0) + uint32_t(g.f));
    // the router's weights: the mapped IDs' rows random (logits about N(0, 1) on N(0, 1) activations), 16 tied pairs
    std::vector<int32_t> mapped(c.pool);
    mapped.insert(mapped.end(), c.pool_x.begin(), c.pool_x.end());
    mapped.insert(mapped.end(), c.pool_y.begin(), c.pool_y.end());
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> rw(size_t(NE) * D, 0.0f), gw(D);
    for (const int32_t e : mapped) {
        for (int32_t k = 0; k < D; ++k) {
            rw[size_t(e) * D + size_t(k)] = 0.02f * nd(rng);
        }
    }
    for (int32_t p = 0; p < 16; ++p) { // pool[2i], pool[2i + 1] and pool_x[2i], pool_x[2i + 1]: distinct experts
        const int32_t a = p < 8 ? c.pool[size_t(2 * p)] : c.pool_x[size_t(2 * (p - 8))];
        const int32_t b = p < 8 ? c.pool[size_t(2 * p + 1)] : c.pool_x[size_t(2 * (p - 8) + 1)];
        std::copy(rw.begin() + std::ptrdiff_t(a) * D, rw.begin() + std::ptrdiff_t(a + 1) * D,
                  rw.begin() + std::ptrdiff_t(b) * D);
    }
    for (float & v : gw) {
        v = 0.02f * nd(rng);
    }
    std::vector<uint8_t> rw_h, gw_h;
    if (draft) {
        rw_h.resize(rw.size() * 4);
        std::memcpy(rw_h.data(), rw.data(), rw_h.size());
        gw_h.resize(gw.size() * 4);
        std::memcpy(gw_h.data(), gw.data(), gw_h.size());
    } else {
        std::vector<uint16_t> rb(rw.size()), gb(gw.size());
        std::transform(rw.begin(), rw.end(), rb.begin(), me8_bf16);
        std::transform(gw.begin(), gw.end(), gb.begin(), me8_bf16);
        rw_h.resize(rb.size() * 2);
        std::memcpy(rw_h.data(), rb.data(), rw_h.size());
        gw_h.resize(gb.size() * 2);
        std::memcpy(gw_h.data(), gb.data(), gw_h.size());
    }
    const size_t xf_b = size_t(rows) * D * 4, xq_b = act_bytes(rows, D), o_b = size_t(rows) * D * 4;
    dev_mem rwd(rw_h.size()), gwd(gw_h.size()), xf(xf_b * size_t(groups)), xq(xq_b * size_t(groups));
    // per card: its routing, the router's outputs and work, the experts' meta, GLU, status and outputs
    struct card_bufs {
        std::unique_ptr<dev_mem> ids, wts, sg, logits, sgl, work, meta, glu, st, out;
        std::unique_ptr<dev_mem> l_ids, l_wts, l_sg, l_lg; // the router's log, a group's routing each
    } cd[2];
    fn::dense::product_args ra;
    ra.k = D;
    ra.n_mats = 2;
    ra.mats[0].f = ra.mats[1].f = draft ? fn::dense::fmt::f32 : fn::dense::fmt::bf16;
    ra.mats[0].m = NE;
    ra.mats[1].m = 1;
    ra.route.logits = 0;
    ra.route.gate = 1;
    fn::dense::choose_splits(ra);
    const size_t work_b = fn::dense::work_bytes(ra);
    bool ok = rwd.put(rw_h.data(), rw_h.size()) && gwd.put(gw_h.data(), gw_h.size()) && work_b > 0;
    for (card_bufs & b : cd) {
        b.ids = std::make_unique<dev_mem>(size_t(rows) * NU * 4);
        b.wts = std::make_unique<dev_mem>(size_t(rows) * NU * 4);
        b.sg = std::make_unique<dev_mem>(size_t(rows) * 4);
        b.logits = std::make_unique<dev_mem>(size_t(rows) * NE * 4);
        b.sgl = std::make_unique<dev_mem>(size_t(rows) * 4);
        b.work = std::make_unique<dev_mem>(std::max<size_t>(work_b, 16));
        b.meta = std::make_unique<dev_mem>(meta_bytes);
        b.glu = std::make_unique<dev_mem>(glu_buf_bytes);
        b.st = std::make_unique<dev_mem>(4);
        b.out = std::make_unique<dev_mem>(o_b * size_t(groups));
        b.l_ids = std::make_unique<dev_mem>(size_t(groups) * rows * NU * 4);
        b.l_wts = std::make_unique<dev_mem>(size_t(groups) * rows * NU * 4);
        b.l_sg = std::make_unique<dev_mem>(size_t(groups) * rows * 4);
        b.l_lg = std::make_unique<dev_mem>(size_t(groups) * rows * NE * 4);
        for (const dev_mem * m : {b.ids.get(), b.wts.get(), b.sg.get(), b.logits.get(), b.sgl.get(), b.work.get(),
                                  b.meta.get(), b.glu.get(), b.st.get(), b.out.get(), b.l_ids.get(), b.l_wts.get(),
                                  b.l_sg.get(), b.l_lg.get()}) {
            ok = ok && m->p != nullptr && cuda_ok(cudaMemset(m->p, 0, m->n), "me8 memset");
        }
    }
    // the experts: the resident banks, or (cached) the address tables over scattered copies
    const dev_mem * wgu[2] = {dev_packed(c, g.f, false, 0), dev_packed(c, g.f, false, 1)};
    const dev_mem * wdn[2] = {dev_packed(c, d.f, true, 0), dev_packed(c, d.f, true, 1)};
    me1_bank tgu[2], tdn[2];
    for (int32_t k = 0; k < 2 && ok; ++k) {
        ok = wgu[k] != nullptr && wdn[k] != nullptr &&
             (!cached || (me1_scatter(packed(c, g.f, false, k), gate_up_bytes(g.f), {}, tgu[k]) &&
                          me1_scatter(packed(c, d.f, true, k), down_bytes(d.f), {}, tdn[k])));
    }
    cudaStream_t cs = nullptr;
    ok = ok && cuda_ok(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "me8 stream");
    if (!check(c, ok, name + ": setup")) {
        if (cs != nullptr) {
            cudaStreamDestroy(cs);
        }
        return;
    }
    const auto log_route = [&](int32_t k, int32_t grp, const card_bufs & to) {
        const card_bufs & b = cd[k];
        const size_t r = size_t(rows);
        return cuda_ok(cudaMemcpyAsync(to.l_ids->p + size_t(grp) * r * NU * 4, b.ids->p, r * NU * 4,
                                       cudaMemcpyDeviceToDevice, cs),
                       "me8 log") &&
               cuda_ok(cudaMemcpyAsync(to.l_wts->p + size_t(grp) * r * NU * 4, b.wts->p, r * NU * 4,
                                       cudaMemcpyDeviceToDevice, cs),
                       "me8 log") &&
               cuda_ok(cudaMemcpyAsync(to.l_sg->p + size_t(grp) * r * 4, b.sg->p, r * 4, cudaMemcpyDeviceToDevice, cs),
                       "me8 log") &&
               cuda_ok(cudaMemcpyAsync(to.l_lg->p + size_t(grp) * r * NE * 4, b.logits->p, r * NE * 4,
                                       cudaMemcpyDeviceToDevice, cs),
                       "me8 log");
    };
    // card k's router on group grp's activations into its own routing, logged into log's group grp
    const auto router = [&](int32_t k, int32_t grp, const card_bufs & log) {
        fn::dense::product_args a = ra;
        a.mats[0].w = rwd.p;
        a.mats[0].out = cd[k].logits->as<float>();
        a.mats[1].w = gwd.p;
        a.mats[1].out = cd[k].sgl->as<float>();
        a.x_f32 = reinterpret_cast<const float *>(xf.p + size_t(grp) * xf_b);
        a.route.ids = cd[k].ids->as<int32_t>();
        a.route.wts = cd[k].wts->as<float>();
        a.route.sg = cd[k].sg->as<float>();
        a.work = cd[k].work->p;
        return fn::dense::product(rows, a, cs) && log_route(k, grp, log);
    };
    // card k's experts on group grp, on its own routing or (shared) card 0's
    const auto experts = [&](int32_t k, int32_t grp, bool shared) {
        card_bufs & b = cd[k];
        gate_up_args ga = {wgu[k]->p,
                           c.sh_gu.at(int32_t(g.fg)).d[k]->p,
                           c.sh_gu.at(int32_t(g.fu)).d[k]->p,
                           c.d_grids.p,
                           act_in(xq.p + size_t(grp) * xq_b, rows, D),
                           b.ids->as<int32_t>(),
                           b.meta->p,
                           act_out(b.glu->p, glu_slots, n_ff_card),
                           nullptr,
                           b.st->as<uint32_t>()};
        ga.cw = class_weights(g.f, g.fg, g.fu);
        down_args da = {wdn[k]->p,
                        c.sh_dn.at(int32_t(dense_of(d.fs))).d[k]->p,
                        c.d_grids.p,
                        act_in(b.glu->p, glu_slots, n_ff_card),
                        b.ids->as<int32_t>(),
                        b.meta->p,
                        b.wts->as<float>(),
                        b.sg->as<float>(),
                        reinterpret_cast<float *>(b.out->p + size_t(grp) * o_b),
                        b.st->as<uint32_t>()};
        if (cached) {
            ga.wt = tgu[k].tab->as<const uint8_t *>();
            da.wt = tdn[k].tab->as<const uint8_t *>();
        }
        if (shared) {
            use_route(route_ptrs{cd[0].ids->as<int32_t>(), cd[0].wts->as<float>(), cd[0].sg->as<float>()}, ga, da);
        }
        if (!gate_up(g.f, g.fg, g.fu, rows, ga, cs) || !down(d.f, d.fs, rows, da, cs)) {
            fprintf(stderr, "ex-test: %s: experts not enqueued: %s\n", name.c_str(),
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        return true;
    };
    // the graph: per group card 0's router (logged into card 0's log), its experts, card 1's on card 0's routing
    cudaGraph_t gr = nullptr;
    cudaGraphExec_t ge = nullptr;
    ok = cuda_ok(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "me8 capture");
    for (int32_t grp = 0; grp < groups && ok; ++grp) {
        ok = router(0, grp, cd[0]) && experts(0, grp, false) && experts(1, grp, true);
    }
    const bool ended = cuda_ok(cudaStreamEndCapture(cs, &gr), "me8 capture");
    ok = ok && ended && cuda_ok(cudaGraphInstantiate(&ge, gr, 0), "me8 instantiate");
    // the eager reference's logs (card 0's router, card 1's own)
    card_bufs ref[2];
    for (card_bufs & b : ref) {
        b.l_ids = std::make_unique<dev_mem>(cd[0].l_ids->n);
        b.l_wts = std::make_unique<dev_mem>(cd[0].l_wts->n);
        b.l_sg = std::make_unique<dev_mem>(cd[0].l_sg->n);
        b.l_lg = std::make_unique<dev_mem>(cd[0].l_lg->n);
    }
    std::vector<int32_t> prev_ids;
    int32_t bad = 0, unchanged = 0;
    for (int32_t q = 0; q < n_replay && ok; ++q) {
        std::vector<float> x(size_t(groups) * rows * D);
        for (float & v : x) {
            v = nd(rng);
        }
        std::vector<uint8_t> xqh(xq.n);
        for (int32_t grp = 0; grp < groups; ++grp) {
            act_quantize(x.data() + size_t(grp) * rows * D, rows, D, xqh.data() + size_t(grp) * xq_b);
        }
        ok = cuda_ok(cudaMemcpyAsync(xf.p, x.data(), xf.n, cudaMemcpyHostToDevice, cs), "me8 x") &&
             cuda_ok(cudaMemcpyAsync(xq.p, xqh.data(), xq.n, cudaMemcpyHostToDevice, cs), "me8 xq") &&
             cuda_ok(cudaGraphLaunch(ge, cs), "me8 replay") && cuda_ok(cudaStreamSynchronize(cs), "me8 replay");
        std::vector<uint8_t> out_g[2], out_e[2];
        uint32_t st_g[2] = {9, 9}, st_e[2] = {9, 9};
        for (int32_t k = 0; k < 2 && ok; ++k) {
            out_g[k].resize(cd[k].out->n);
            ok = cd[k].out->get(out_g[k].data(), out_g[k].size()) && cd[k].st->get(&st_g[k], 4);
        }
        // the eager two-router form
        for (int32_t grp = 0; grp < groups && ok; ++grp) {
            ok = router(0, grp, ref[0]) && experts(0, grp, false) && router(1, grp, ref[1]) && experts(1, grp, false);
        }
        ok = ok && cuda_ok(cudaStreamSynchronize(cs), "me8 eager");
        for (int32_t k = 0; k < 2 && ok; ++k) {
            out_e[k].resize(cd[k].out->n);
            ok = cd[k].out->get(out_e[k].data(), out_e[k].size()) && cd[k].st->get(&st_e[k], 4);
        }
        // the routing logs: the graph's (card 0's router), the eager card 0's and card 1's own
        const size_t n_id = size_t(groups) * rows * NU;
        std::vector<int32_t> ids_g(n_id), ids_e0(n_id), ids_e1(n_id);
        std::vector<float> wts_g(n_id), wts_e1(n_id), sg_g(size_t(groups) * rows), sg_e1(sg_g.size()),
            lg_g(size_t(groups) * rows * NE);
        ok = ok && cd[0].l_ids->get(ids_g.data(), n_id * 4) && ref[0].l_ids->get(ids_e0.data(), n_id * 4) &&
             ref[1].l_ids->get(ids_e1.data(), n_id * 4) && cd[0].l_wts->get(wts_g.data(), n_id * 4) &&
             ref[1].l_wts->get(wts_e1.data(), n_id * 4) && cd[0].l_sg->get(sg_g.data(), sg_g.size() * 4) &&
             ref[1].l_sg->get(sg_e1.data(), sg_e1.size() * 4) && cd[0].l_lg->get(lg_g.data(), lg_g.size() * 4);
        if (!ok) {
            break;
        }
        // nontrivial routes: every ID reads a synthetic expert, the weights and shared gates in (0, 1), the set the
        // top 10 of the logits (unless a tie at the cut makes it ambiguous); the ties counted
        bool route_ok = true;
        for (int32_t r = 0; r < groups * rows; ++r) {
            const float * lg = lg_g.data() + size_t(r) * NE;
            std::vector<float> sorted(lg, lg + NE);
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            const bool cut_tie = sorted[NU - 1] == sorted[NU];
            s.ties_cut += cut_tie ? 1 : 0;
            std::set<float> sel_lg;
            for (int32_t k = 0; k < NU; ++k) {
                const int32_t e = ids_g[size_t(r) * NU + size_t(k)];
                const float w = wts_g[size_t(r) * NU + size_t(k)];
                route_ok = route_ok && e >= 0 && e < NE && c.map[size_t(e)] < n_synth && w > 0.0f && w < 1.0f &&
                           (cut_tie ? lg[e] >= sorted[NU - 1] : lg[e] > sorted[NU]);
                if (e >= 0 && e < NE) {
                    sel_lg.insert(lg[e]);
                }
            }
            s.ties_in += int32_t(sel_lg.size()) < NU ? 1 : 0;
            route_ok = route_ok && sg_g[size_t(r)] > 0.0f && sg_g[size_t(r)] < 1.0f;
        }
        // the replay's routes changed from the last one's, group by group
        bool changed = true;
        for (int32_t grp = 0; grp < groups && !prev_ids.empty(); ++grp) {
            const size_t at = size_t(grp) * rows * NU;
            changed = changed && !std::equal(ids_g.begin() + std::ptrdiff_t(at),
                                             ids_g.begin() + std::ptrdiff_t(at + size_t(rows) * NU),
                                             prev_ids.begin() + std::ptrdiff_t(at));
        }
        prev_ids = ids_g;
        float out_max = 0.0f;
        for (size_t i = 0; i < out_g[1].size() / 4; ++i) {
            float v = 0.0f;
            std::memcpy(&v, out_g[1].data() + 4 * i, 4);
            out_max = std::max(out_max, std::fabs(v));
        }
        const bool same = ids_g == ids_e0 && ids_g == ids_e1 &&
                          std::memcmp(wts_g.data(), wts_e1.data(), n_id * 4) == 0 &&
                          std::memcmp(sg_g.data(), sg_e1.data(), sg_g.size() * 4) == 0 && out_g[0] == out_e[0] &&
                          out_g[1] == out_e[1] && st_g[0] == 0 && st_g[1] == 0 && st_e[0] == 0 && st_e[1] == 0;
        const bool pass = same && route_ok && out_max > 0.0f && std::isfinite(out_max);
        bad += pass ? 0 : 1;
        unchanged += changed ? 0 : 1;
        if (!pass) {
            fprintf(stderr,
                    "ex-test: %s replay %d: routing %s, card 0 out %s, card 1 out %s, status %u %u (eager %u %u), "
                    "routes %s, card 1 max |out| %g\n",
                    name.c_str(), q, ids_g == ids_e1 && ids_g == ids_e0 ? "equal" : "DIFFERS",
                    out_g[0] == out_e[0] ? "equal" : "DIFFERS", out_g[1] == out_e[1] ? "equal" : "DIFFERS", st_g[0],
                    st_g[1], st_e[0], st_e[1], route_ok ? "nontrivial" : "TRIVIAL OR INVALID", double(out_max));
        }
        ++s.replays;
        s.groups += groups;
        s.rows += groups * rows;
    }
    if (ge != nullptr) {
        cudaGraphExecDestroy(ge);
    }
    if (gr != nullptr) {
        cudaGraphDestroy(gr);
    }
    cudaStreamDestroy(cs);
    s.bad += bad;
    s.unchanged += unchanged;
    printf("ex-test: %s: %d replays of one graph against the eager two-router form: %d differ, %d with an unchanged "
           "route\n",
           name.c_str(), ok ? n_replay : -1, bad, unchanged);
    check(c, ok && bad == 0 && unchanged == 0, name);
}

// ME8's cases: the target's B the row mask holds, resident and cached; a draft's P = 1..4 proposals of one row
void me8_routes(test_ctx & c, const std::vector<gu_inst_t> & gus, const std::vector<dn_inst_t> & dns,
                const std::vector<int32_t> & Bs, bool q2d) {
    const auto gu_if = [&](auto pred) -> const gu_inst_t * {
        const auto it = std::find_if(gus.begin(), gus.end(), pred);
        return it == gus.end() ? nullptr : &*it;
    };
    const auto dn_if = [&](auto pred) -> const dn_inst_t * {
        const auto it = std::find_if(dns.begin(), dns.end(), pred);
        return it == dns.end() ? nullptr : &*it;
    };
    const auto draft_fmt = [](fmt f) { return f == fmt::q8_0 || f == fmt::q2_0; };
    me8_stats s;
    const gu_inst_t * tg = q2d ? nullptr : gu_if([&](const gu_inst_t & g) { return !draft_fmt(g.f); });
    const dn_inst_t * td = q2d ? nullptr : dn_if([](const dn_inst_t & d) { return d.f != fmt::q8_0; });
    const fmt df = q2d ? fmt::q2_0 : fmt::q8_0; // the draft's routed format: Q8_0 in the deployed lists, ME7's Q2_0
    const gu_inst_t * dg = gu_if([&](const gu_inst_t & g) { return g.f == df && g.fg == fn::dense::fmt::q8_0; });
    const dn_inst_t * dd = dn_if([&](const dn_inst_t & d) { return d.f == df && d.fs == fmt::q8_0; });
    if (tg != nullptr && td != nullptr) {
        for (const bool cached : {false, true}) {
            for (const int32_t B : Bs) {
                me8_shared_route(c, *tg, *td, false, cached, B, s);
            }
        }
    }
    if (dg != nullptr && dd != nullptr) {
        for (int32_t P = 1; P <= 4; ++P) {
            me8_shared_route(c, *dg, *dd, true, false, P, s);
        }
    }
    printf("ex-test: me8 shared routing: %d cases (target %s, draft %s), %d replays, %d groups, %d rows: %d differ "
           "from the eager two-router form, %d with an unchanged route; ties: %d rows inside the top 10, %d at its "
           "cut: %s\n",
           s.cases, tg != nullptr && td != nullptr ? "run" : "none in this list",
           dg != nullptr && dd != nullptr ? fmt_name(df) : "none in this list", s.replays, s.groups, s.rows, s.bad,
           s.unchanged, s.ties_in, s.ties_cut,
           s.cases > 0 && s.bad == 0 && s.unchanged == 0 && s.ties_in > 0 && s.ties_cut > 0 ? "as expected"
                                                                                               : "NOT as expected");
    check(c, s.cases > 0, "me8: a case in this list");
    check(c, s.cases == 0 || (s.ties_in > 0 && s.ties_cut > 0), "me8: tied routes exercised");
}


// ---- ME10: the one-device expert window (combine_finish, gate_up_pair) ---------------------------------------------
// The finish against today's combine then X-F's exchange on one device (combine into card 0's partial; send_half of
// each card's peer half into the other's slot, prefilled with the sentinel; recv_add on both cards): both halves
// bitwise equal, NaN payloads included, the slots back at the sentinel after the reference, its error word 0, and the
// finish's inputs unwritten. Each B of the mask: CPU slots set, the slot word 0, and a bad ID (the meta flagged: no
// CPU term). The partials, the CPU products and the weights hold signed zeros, infinities, NaNs of several payloads,
// subnormals and values whose sum overflows.
void me10_finish(test_ctx & c, const row_case & master, const std::vector<int32_t> & Bs) {
    std::mt19937 rng(0x4d453130u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const uint32_t edge_bits[] = {0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00000u,
                                  0x7f800001u, 0x00000001u, 0x807fffffu, 0x7f7fffffu, 0xff7fffffu, 0x00800000u};
    const auto edge = [&](std::vector<float> & v, int32_t every) {
        for (size_t i = 0; i < v.size(); ++i) {
            if (rng() % uint32_t(every) == 0) {
                std::memcpy(&v[i], &edge_bits[rng() % (sizeof(edge_bits) / 4)], 4);
            }
        }
    };
    const size_t HW = fn::xchg::words_half;
    int32_t n_cases = 0, n_bad = 0;
    int64_t n_nan = 0, n_inf = 0, n_negzero = 0, n_sub = 0;
    for (const int32_t B : Bs) {
        for (int32_t kind = 0; kind < 3; ++kind) { // 0: CPU slots; 1: the slot word 0; 2: a bad ID
            row_case rc = prefix(master, B);
            if (kind == 2) {
                rc.ids[size_t(NU * B - 1)] = NE + 7;
                rc.m = make_meta(rc.ids, B);
            }
            uint64_t slots = 0;
            if (kind == 0) {
                me1_cpu_set(rc, slots);
            } else if (kind == 2) {
                slots = 0x3ffull;
            }
            std::vector<float> p0(size_t(B) * D), p1(size_t(B) * D), ext(size_t(max_slots) * D), wts(rc.wts);
            for (std::vector<float> * v : {&p0, &p1, &ext}) {
                for (float & x : *v) {
                    x = 64.0f * nd(rng);
                }
                edge(*v, 9);
            }
            for (size_t i = 0; i < 64 && i < p0.size(); ++i) { // overflowing sums and inf - inf
                p0[i * 37 % p0.size()] = i % 2 == 0 ? 3.0e38f : INFINITY;
                p1[i * 37 % p1.size()] = i % 2 == 0 ? 3.0e38f : -INFINITY;
            }
            edge(wts, 11);
            dev_mem ids(rc.ids.size() * 4), meta(meta_bytes), dw(wts.size() * 4), dext(ext.size() * 4), word(8),
                p0r(p0.size() * 4), p0f(p0.size() * 4), dp1(p1.size() * 4), s0(size_t(B) * HW * 4),
                s1(size_t(B) * HW * 4), o0(size_t(B) * HW * 4), o1(size_t(B) * HW * 4), h0(size_t(B) * HW * 4),
                h1(size_t(B) * HW * 4), err(4);
            const std::vector<uint32_t> sent(size_t(B) * HW, fn::xchg::sentinel);
            bool ok = ids.put(rc.ids.data(), ids.n) && device_meta(ids, B, dn_w, meta) && dw.put(wts.data(), dw.n) &&
                      dext.put(ext.data(), dext.n) && word.put(&slots, 8) && p0r.put(p0.data(), p0r.n) &&
                      p0f.put(p0.data(), p0f.n) && dp1.put(p1.data(), dp1.n) && s0.put(sent.data(), s0.n) &&
                      s1.put(sent.data(), s1.n) && cuda_ok(cudaMemset(o0.p, 0xa5, o0.n), "memset") &&
                      cuda_ok(cudaMemset(o1.p, 0xa5, o1.n), "memset") &&
                      cuda_ok(cudaMemset(h0.p, 0xa5, h0.n), "memset") &&
                      cuda_ok(cudaMemset(h1.p, 0xa5, h1.n), "memset") && cuda_ok(cudaMemset(err.p, 0, 4), "memset");
            void * const st = cudaStreamPerThread;
            combine_args cb{};
            cb.meta = meta.p;
            cb.wts = dw.as<float>();
            cb.ext_ds = dext.as<float>();
            cb.ext_slots = word.as<uint64_t>();
            cb.out = p0r.as<float>();
            cb.rows = B;
            combine_args cf = cb;
            cf.out = p0f.as<float>();
            float * const half[2] = {h0.as<float>(), h1.as<float>()};
            ok = ok && combine(cb, st) && fn::xchg::send_half(B, 1, p0r.as<float>(), s1.as<float>(), st) &&
                 fn::xchg::send_half(B, 0, dp1.as<float>(), s0.as<float>(), st) &&
                 fn::xchg::recv_add(B, 0, p0r.as<float>(), s0.as<float>(), o0.as<float>(), err.as<uint32_t>(),
                                    fn::xchg::err_poll_f, st) &&
                 fn::xchg::recv_add(B, 1, dp1.as<float>(), s1.as<float>(), o1.as<float>(), err.as<uint32_t>(),
                                    fn::xchg::err_poll_f, st) &&
                 combine_finish(cf, dp1.as<float>(), half, st) && cuda_ok(cudaStreamSynchronize(cudaStreamPerThread),
                                                                           "me10 finish");
            std::vector<uint32_t> g0(size_t(B) * HW), g1(g0.size()), f0(g0.size()), f1(g0.size()), sl0(g0.size()),
                sl1(g0.size()), q0(p0.size()), q1(p1.size());
            uint32_t ew = 9;
            ok = ok && o0.get(g0.data(), o0.n) && o1.get(g1.data(), o1.n) && h0.get(f0.data(), h0.n) &&
                 h1.get(f1.data(), h1.n) && s0.get(sl0.data(), s0.n) && s1.get(sl1.data(), s1.n) &&
                 p0f.get(q0.data(), p0f.n) && dp1.get(q1.data(), dp1.n) && err.get(&ew, 4);
            const bool same = ok && g0 == f0 && g1 == f1;
            const bool slots_back = ok && sl0 == sent && sl1 == sent && ew == 0;
            const bool inputs_kept = ok && std::memcmp(q0.data(), p0.data(), p0f.n) == 0 &&
                                     std::memcmp(q1.data(), p1.data(), dp1.n) == 0;
            for (const std::vector<uint32_t> * v : {&f0, &f1}) {
                for (const uint32_t x : *v) {
                    n_nan += (x & 0x7fffffffu) > 0x7f800000u ? 1 : 0;
                    n_inf += (x & 0x7fffffffu) == 0x7f800000u ? 1 : 0;
                    n_negzero += x == 0x80000000u ? 1 : 0;
                    n_sub += (x & 0x7f800000u) == 0 && (x & 0x007fffffu) != 0 ? 1 : 0;
                }
            }
            ++n_cases;
            n_bad += same && slots_back && inputs_kept ? 0 : 1;
            if (!(same && slots_back && inputs_kept)) {
                fprintf(stderr, "ex-test: me10 finish B %d kind %d: halves %s, reference slots %s (error word %u), "
                                "inputs %s\n",
                        B, kind, same ? "equal" : "DIFFER", slots_back ? "back at the sentinel" : "NOT RESET", ew,
                        inputs_kept ? "kept" : "WRITTEN");
            }
        }
    }
    printf("ex-test: me10 finish against combine + X-F's exchange (one device): %d cases (B of the mask x CPU slots, "
           "word 0, bad ID), %d differ; outputs compared held %lld NaN, %lld infinite, %lld -0, %lld subnormal: %s\n",
           n_cases, n_bad, (long long)n_nan, (long long)n_inf, (long long)n_negzero, (long long)n_sub,
           n_bad == 0 && n_cases > 0 && n_nan > 0 && n_inf > 0 && n_negzero > 0 && n_sub > 0 ? "as expected"
                                                                                             : "NOT as expected");
    check(c, n_cases > 0 && n_bad == 0, "me10 finish bitwise combine + exchange");
    check(c, n_nan > 0 && n_inf > 0 && n_negzero > 0 && n_sub > 0, "me10 finish: the edge values reached the output");
}

// R15 repair: the first differing byte of two buffers, -1 when equal (a size difference counts at the shorter's end)
int64_t first_diff_at(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return int64_t(i);
        }
    }
    return a.size() == b.size() ? -1 : int64_t(n);
}

// R15 repair: the meta region's defined bytes at the case's rows (experts.h meta_view and the unit order): n_u,
// n_slots, bad; e and slot_base of the live experts u < n_u; every mask; u_of of the 10 B entries; the active order's
// n_order entries (R18 repair: the pass's, pass_order). Everything else (unused e and slot_base entries, the struct's
// tail, later order entries) is unspecified.
std::vector<uint8_t> meta_defined_bytes(const row_case & rc, int32_t n_order) {
    std::vector<uint8_t> d(size_t(meta_bytes), 0);
    const meta_view v = meta_view_of(rc.rows);
    const auto word = [&](int32_t w) {
        for (int32_t b = 0; b < 4; ++b) {
            d[size_t(w) * 4 + size_t(b)] = 1;
        }
    };
    word(v.n_u);
    word(v.n_slots);
    word(v.bad);
    for (int32_t u = 0; u < v.slots; ++u) {
        word(v.mask + u);
        if (u < rc.m.n_u) {
            word(v.e + u);
            word(v.slot_base + u);
        }
    }
    for (int32_t p = 0; p < rc.rows * NU; ++p) {
        word(v.u_of + p);
    }
    const int32_t order_bytes = 2 * n_order;
    for (int32_t i = 0; i < order_bytes; ++i) {
        d[size_t(v.order) + size_t(i)] = 1;
    }
    return d;
}

// gate_up_pair against two gate_up launches (the window's two shards on one device) at B 4 and 5 of the mask, for the
// five IQ formats' first deployed triple: the master case and SP4's route cases (every nonempty row mask, one-row and
// fully shared experts, 50 distinct experts at B 5), each pass (skip 1 on the hit table with the shared expert, then
// skip 2 on the miss table, run_if 1 and 0), each shard on its own scattered table (unrelated addresses); the master
// case also through each card's X-HMIX slot. Per shard every byte of the GLU, the F32 GLU, both passes' meta (struct
// and unit order), the status and the error word bitwise equal (R15 repair: the meta by its defined words against the
// host reference and between the forms, untouched when a pass is skipped; see the comparison). Then refusals
// (gate_up_pair_supported false, nothing enqueued), and the snake deal over 36 and 72 blocks taking every rank once.
void me10_pair(test_ctx & c, const std::vector<gu_inst_t> & gus, const row_case & master,
               const std::vector<int32_t> & Bs) {
    // the deal (experts_gu.cuh block_units): rank r = 72 k or 36 k + first (k even), + NB - 1 - first (k odd)
    bool deal_ok = true;
    for (const int32_t nb : {36, 72}) {
        for (const int32_t cap : {4, 5}) {
            const int32_t units = max_units_of(cap), ranks = (units + nb - 1) / nb;
            std::vector<int32_t> seen(size_t(units), 0);
            for (int32_t first = 0; first < nb; ++first) {
                for (int32_t k = 0; k < ranks; ++k) {
                    const int32_t r = k * nb + ((k & 1) != 0 ? nb - 1 - first : first);
                    if (r < units) {
                        ++seen[size_t(r)];
                    }
                }
            }
            deal_ok = deal_ok && ranks <= 32 && std::all_of(seen.begin(), seen.end(), [](int32_t n) { return n == 1; });
        }
    }
    check(c, deal_ok, "me10 the snake deal over 36 and 72 blocks: every rank once");
    std::mt19937 rng(0x4d453131u);
    int32_t n_cmp = 0, n_bad = 0, n_formats = 0, n_live_diff = 0, n_undef_diff = 0, n_neg = 0, n_neg_missed = 0,
            n_neg_order = 0;
    for (const fmt f : {fmt::iq2_xxs, fmt::iq2_xs, fmt::iq2_s, fmt::iq3_xxs, fmt::iq3_s}) {
        const auto gi = std::find_if(gus.begin(), gus.end(), [&](const gu_inst_t & g) { return g.f == f; });
        if (gi == gus.end()) {
            continue;
        }
        const gu_inst_t g = *gi;
        ++n_formats;
        const int64_t bytes = gate_up_bytes(f);
        me1_bank bk[2];
        std::vector<const uint8_t *> t_all[2];
        bool ok = true;
        for (int32_t k = 0; k < 2 && ok; ++k) {
            t_all[k].resize(NE);
            ok = me1_scatter(packed(c, f, false, k), bytes, {}, bk[k]) && bk[k].tab->get(t_all[k].data(), bk[k].tab->n);
        }
        if (!check(c, ok, "me10 pair " + gu_name(g.f, g.fg, g.fu) + ": tables")) {
            continue;
        }
        for (const int32_t B : Bs) {
            if (B < 4) {
                continue;
            }
            std::vector<row_case> cases = {prefix(master, B)};
            for (row_case & rc : sp4_route_cases(c, master, B, rng)) {
                cases.push_back(std::move(rc));
            }
            for (size_t ci = 0; ci < cases.size(); ++ci) {
                const row_case & rc = cases[ci];
                for (const int32_t hm : {0, 1}) {
                    if (hm == 1 && ci != 0) {
                        continue;
                    }
                    for (const uint32_t run : {1u, 0u}) {
                        // the hit table (u even) and the miss table (u odd), each shard its own
                        std::vector<const uint8_t *> t1[2], t2[2];
                        for (int32_t k = 0; k < 2; ++k) {
                            t1[k].assign(NE, nullptr);
                            t2[k].assign(NE, nullptr);
                            for (int32_t u = 0; u < rc.m.n_u; ++u) {
                                const int32_t e = rc.m.e[size_t(u)];
                                (u % 2 == 0 ? t1[k] : t2[k])[size_t(e)] = t_all[k][size_t(e)];
                            }
                        }
                        dev_mem x(rc.xq.size()), ids(rc.ids.size() * 4), runw(4);
                        std::unique_ptr<dev_mem> tab1[2], tab2[2], glu[2][2], gf[2][2], m1[2][2], m2[2][2], st[2][2],
                            hs[2], er[2][2];
                        ok = x.put(rc.xq.data(), x.n) && ids.put(rc.ids.data(), ids.n) && runw.put(&run, 4);
                        for (int32_t k = 0; k < 2 && ok; ++k) {
                            tab1[k] = std::make_unique<dev_mem>(size_t(NE) * 8);
                            tab2[k] = std::make_unique<dev_mem>(size_t(NE) * 8);
                            hs[k] = std::make_unique<dev_mem>(size_t(fn::xchg::hmix_layout::slot_words) * 4);
                            const std::vector<uint32_t> hw = hmix_slot(rc, k, false);
                            ok = tab1[k]->put(t1[k].data(), tab1[k]->n) && tab2[k]->put(t2[k].data(), tab2[k]->n) &&
                                 hs[k]->put(hw.data(), hs[k]->n);
                            for (int32_t form = 0; form < 2 && ok; ++form) { // 0 separate, 1 paired
                                glu[form][k] = std::make_unique<dev_mem>(glu_buf_bytes);
                                gf[form][k] = std::make_unique<dev_mem>(size_t(glu_slots) * n_ff_card * 4);
                                m1[form][k] = std::make_unique<dev_mem>(meta_bytes);
                                m2[form][k] = std::make_unique<dev_mem>(meta_bytes);
                                st[form][k] = std::make_unique<dev_mem>(4);
                                er[form][k] = std::make_unique<dev_mem>(4);
                                for (const dev_mem * d : {glu[form][k].get(), gf[form][k].get(), m1[form][k].get(),
                                                          m2[form][k].get()}) {
                                    ok = ok && cuda_ok(cudaMemset(d->p, 0xa5, d->n), "memset");
                                }
                                ok = ok && cuda_ok(cudaMemset(st[form][k]->p, 0, 4), "memset") &&
                                     cuda_ok(cudaMemset(er[form][k]->p, 0, 4), "memset");
                            }
                        }
                        // shard k's pass p arguments in form (0 separate, 1 paired)
                        const auto args = [&](int32_t form, int32_t k, int32_t pass) {
                            gate_up_args a = {bk[k].zero->p,
                                              c.sh_gu.at(int32_t(g.fg)).d[k]->p,
                                              c.sh_gu.at(int32_t(g.fu)).d[k]->p,
                                              c.d_grids.p,
                                              act_in(x.p, rc.rows, D),
                                              ids.as<int32_t>(),
                                              (pass == 1 ? m1 : m2)[form][k]->p,
                                              act_out(glu[form][k]->p, glu_slots, n_ff_card),
                                              gf[form][k]->as<float>(),
                                              st[form][k]->as<uint32_t>()};
                            a.cw = class_weights(g.f, g.fg, g.fu);
                            a.map = 3;
                            a.wt = (pass == 1 ? tab1 : tab2)[k]->as<const uint8_t *>();
                            a.skip = pass;
                            a.run_if = pass == 2 ? runw.as<uint32_t>() : nullptr;
                            if (hm == 1) {
                                a.x = {nullptr, nullptr, rc.rows, D};
                                a.hmix = hs[k]->as<float>();
                                a.hmix_peer_half = 1 - k;
                                a.err = er[form][k]->as<uint32_t>();
                            }
                            return a;
                        };
                        void * const s = cudaStreamPerThread;
                        bool sup = true;
                        for (int32_t pass = 1; pass <= 2 && ok; ++pass) {
                            const gate_up_pair_args pa = {{args(1, 0, pass), args(1, 1, pass)}};
                            sup = sup && gate_up_pair_supported(g.f, g.fg, g.fu, rc.rows, pa);
                            ok = gate_up(g.f, g.fg, g.fu, rc.rows, args(0, 0, pass), s) &&
                                 gate_up(g.f, g.fg, g.fu, rc.rows, args(0, 1, pass), s) &&
                                 gate_up_pair(g.f, g.fg, g.fu, rc.rows, pa, s);
                        }
                        ok = ok && cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "me10 pair");
                        // R15 repair (Fable's ruling): every buffer compared, each failure named with its first
                        // differing byte. The live outputs (GLU codes and scales, F32 GLU, status, error word) stay
                        // bitwise over the whole buffer: both forms start from the same fill. The meta is compared by
                        // its contract (experts.h): each form's defined words and active unit order against the host
                        // reference, the forms' defined bytes against each other, and a skipped pass (run_if 0)
                        // leaving its region at the fill. R18 repair: the reference is the pass's
                        // (pass_region_matches): the whole launch's struct, and the order of the experts this pass's
                        // table holds, original indices kept, the shared class at pass 1 only. Bytes outside the
                        // defined words (unused e and slot_base entries, the struct's tail, order entries past the
                        // pass's extent) are unspecified: a difference there is counted, not failed. Every compared
                        // meta must also FAIL with one defined field and one active-order entry corrupted.
                        bool same = ok && sup;
                        uint32_t status_or = 0;
                        std::string why;
                        const auto note = [&](const std::string & w) { why += (why.empty() ? "" : "; ") + w; };
                        const class_table cw = class_weights(g.f, g.fg, g.fu);
                        for (int32_t k = 0; k < 2 && ok; ++k) {
                            const std::pair<const char *, const decltype(glu) *> live[] = {
                                {"GLU", &glu}, {"F32 GLU", &gf}, {"status", &st}, {"error word", &er}};
                            for (const auto & [name, arr] : live) {
                                const dev_mem & a0 = *(*arr)[0][k];
                                const dev_mem & a1 = *(*arr)[1][k];
                                std::vector<uint8_t> v0(a0.n), v1(a1.n);
                                if (!a0.get(v0.data(), a0.n) || !a1.get(v1.data(), a1.n)) {
                                    same = false;
                                    note("shard " + std::to_string(k) + " " + name + ": read back failed");
                                    continue;
                                }
                                const int64_t off = first_diff_at(v0, v1);
                                if (off >= 0) {
                                    same = false;
                                    ++n_live_diff;
                                    note("shard " + std::to_string(k) + " " + name + " (live) first differs at byte " +
                                         std::to_string(off) + " of " + std::to_string(v0.size()));
                                }
                                if (arr == &st || arr == &er) {
                                    uint32_t w0 = 0, w1 = 0;
                                    std::memcpy(&w0, v0.data(), 4);
                                    std::memcpy(&w1, v1.data(), 4);
                                    status_or |= w0 | w1;
                                }
                            }
                            for (int32_t pass = 1; pass <= 2; ++pass) {
                                const auto & arr = pass == 1 ? m1 : m2;
                                const bool skipped = pass == 2 && run == 0u;
                                // the experts this pass runs, from its own table (the hit or the miss table)
                                std::vector<bool> runs(size_t(rc.m.n_u), false);
                                for (int32_t u = 0; u < rc.m.n_u; ++u) {
                                    runs[size_t(u)] = (pass == 1 ? t1 : t2)[k][size_t(rc.m.e[size_t(u)])] != nullptr;
                                }
                                std::vector<uint16_t> want_order;
                                const int32_t n_order = pass_order(rc, cw, runs, pass == 1, want_order);
                                if (n_order < 0) {
                                    same = false;
                                    note("the pass's reference order has the wrong extent");
                                    continue;
                                }
                                const std::vector<uint8_t> defined = meta_defined_bytes(rc, n_order);
                                std::vector<uint8_t> v[2];
                                for (int32_t form = 0; form < 2; ++form) {
                                    const dev_mem & d = *arr[form][k];
                                    v[form].resize(d.n);
                                    if (!d.get(v[form].data(), d.n)) {
                                        same = false;
                                        note("meta read back failed");
                                        continue;
                                    }
                                    const std::string where = "shard " + std::to_string(k) + " pass " +
                                                              std::to_string(pass) +
                                                              (form == 0 ? " two-launch" : " paired") + " meta";
                                    if (skipped) {
                                        const bool kept = std::all_of(v[form].begin(), v[form].end(),
                                                                      [](uint8_t b) { return b == 0xa5u; });
                                        if (!kept) {
                                            same = false;
                                            note(where + ": written although the pass is skipped (run_if 0)");
                                        }
                                        continue;
                                    }
                                    std::vector<int32_t> w(meta_bytes / 4);
                                    std::memcpy(w.data(), v[form].data(), meta_bytes);
                                    if (!pass_region_matches(w, rc, cw, runs, pass == 1)) {
                                        same = false;
                                        note(where + ": defined fields or active order differ from the host reference");
                                        continue;
                                    }
                                    // the negative cases: one defined struct word, one active-order entry
                                    const meta_view mv = meta_view_of(rc.rows);
                                    std::vector<int32_t> wf = w, wo = w;
                                    wf[size_t(mv.n_slots)] += 1;
                                    // the active order's last entry, through a uint16_t temporary (no aliased store)
                                    uint8_t * const oe = reinterpret_cast<uint8_t *>(wo.data()) + mv.order +
                                                         size_t(n_order > 0 ? n_order - 1 : 0) * 2;
                                    uint16_t entry = 0;
                                    std::memcpy(&entry, oe, 2);
                                    entry ^= 1u;
                                    std::memcpy(oe, &entry, 2);
                                    ++n_neg;
                                    n_neg_order += n_order > 0 ? 1 : 0;
                                    if (pass_region_matches(wf, rc, cw, runs, pass == 1) ||
                                        (n_order > 0 && pass_region_matches(wo, rc, cw, runs, pass == 1))) {
                                        same = false;
                                        ++n_neg_missed;
                                        note(where + ": a corrupted defined field or order entry PASSED the "
                                                     "comparison");
                                    }
                                }
                                if (v[0].size() != v[1].size() || v[0].size() != defined.size()) {
                                    continue;
                                }
                                int64_t first_def = -1, first_undef = -1;
                                for (size_t i = 0; i < defined.size(); ++i) {
                                    int64_t & first = defined[i] != 0 ? first_def : first_undef;
                                    if (v[0][i] != v[1][i] && first < 0) {
                                        first = int64_t(i);
                                    }
                                }
                                if (first_def >= 0) {
                                    same = false;
                                    note("shard " + std::to_string(k) + " pass " + std::to_string(pass) +
                                         " meta: the forms' defined bytes first differ at byte " +
                                         std::to_string(first_def));
                                }
                                if (first_undef >= 0) {
                                    ++n_undef_diff; // unspecified bytes (experts.h): reported, not compared
                                    if (n_undef_diff <= 4) {
                                        fprintf(stderr,
                                                "ex-test: me10 pair %s B %d case %zu%s run_if %u shard %d pass %d: "
                                                "unspecified meta bytes differ between the forms, first at byte %lld "
                                                "(not compared)\n",
                                                gu_name(g.f, g.fg, g.fu).c_str(), rc.rows, ci,
                                                hm == 1 ? " (X-HMIX)" : "", run, k, pass, (long long)first_undef);
                                    }
                                }
                            }
                        }
                        ++n_cmp;
                        if (!same || status_or != 0) {
                            ++n_bad;
                            fprintf(stderr, "ex-test: me10 pair %s B %d case %zu%s run_if %u: %s%s%s%s\n",
                                    gu_name(g.f, g.fg, g.fu).c_str(), rc.rows, ci, hm == 1 ? " (X-HMIX)" : "", run,
                                    !sup ? "NOT SUPPORTED " : "", same ? "status or error word set" : "DIFFERS",
                                    why.empty() ? "" : ": ", why.c_str());
                        }
                    }
                }
            }
        }
    }
    printf("ex-test: me10 gate_up_pair against two gate_up launches: %d formats, %d case x pass sets at B 4/5 compared "
           "(GLU, F32 GLU, status, error word bitwise; both metas by their defined words and active order against the "
           "host reference, untouched when skipped; per shard), %d differ (%d live-output buffers): %s; unspecified "
           "meta bytes differing between the forms in %d buffers (not compared)\n",
           n_formats, n_cmp, n_bad, n_live_diff, n_bad == 0 && n_cmp > 0 ? "equal" : "NOT EQUAL", n_undef_diff);
    printf("ex-test: me10 gate_up_pair negative cases (R18 repair): %d compared metas each with a corrupted defined "
           "field (n_slots), %d of them also with a corrupted active-order entry: %d passed the comparison: %s\n",
           n_neg, n_neg_order, n_neg_missed, n_neg > 0 && n_neg_order > 0 && n_neg_missed == 0 ? "all FAIL" : "NOT");
    check(c, n_cmp > 0 && n_bad == 0, "me10 gate_up_pair bitwise two launches");
    check(c, n_neg > 0 && n_neg_order > 0 && n_neg_missed == 0, "me10 gate_up_pair: corrupted meta fails");
    // refusals: each one change from a supported pair; nothing enqueued (the GLU keeps its fill)
    const auto g0 = std::find_if(gus.begin(), gus.end(), [](const gu_inst_t & g) { return is_iq2(g.f); });
    const auto gq = std::find_if(gus.begin(), gus.end(), [](const gu_inst_t & g) { return g.f == fmt::q8_0; });
    if (g0 == gus.end() || Bs.back() < 4) {
        return;
    }
    const int32_t B = Bs.back();
    const row_case rc = prefix(master, B);
    dev_mem x(rc.xq.size()), ids(rc.ids.size() * 4), ids2(rc.ids.size() * 4), tab(size_t(NE) * 8), zero(256),
        glu0(glu_buf_bytes), glu1(glu_buf_bytes), m0(meta_bytes), m1(meta_bytes);
    const std::vector<const uint8_t *> tn(NE, nullptr);
    bool ok = x.put(rc.xq.data(), x.n) && ids.put(rc.ids.data(), ids.n) && ids2.put(rc.ids.data(), ids2.n) &&
              tab.put(tn.data(), tab.n) && cuda_ok(cudaMemset(glu0.p, 0xa5, glu0.n), "memset") &&
              cuda_ok(cudaMemset(zero.p, 0, zero.n), "memset");
    const auto base = [&](int32_t k) {
        gate_up_args a = {zero.p,
                          c.sh_gu.at(int32_t(g0->fg)).d[k]->p,
                          c.sh_gu.at(int32_t(g0->fu)).d[k]->p,
                          c.d_grids.p,
                          act_in(x.p, B, D),
                          ids.as<int32_t>(),
                          (k == 0 ? m0 : m1).p,
                          act_out((k == 0 ? glu0 : glu1).p, glu_slots, n_ff_card),
                          nullptr,
                          nullptr};
        a.cw = class_weights(g0->f, g0->fg, g0->fu);
        a.map = 3;
        a.wt = tab.as<const uint8_t *>();
        a.skip = 1;
        return a;
    };
    const gate_up_pair_args good = {{base(0), base(1)}};
    struct refusal {
        const char * what;
        gate_up_pair_args a;
        int32_t rows;
        fmt f;
    };
    std::vector<refusal> rs;
    rs.push_back({"B 3", good, 3, g0->f});
    gate_up_pair_args t = good;
    t.s[1].map = 2;
    rs.push_back({"map 2", t, B, g0->f});
    t = good;
    t.s[0].wt = nullptr;
    t.s[0].skip = 0;
    t.s[1].wt = nullptr;
    t.s[1].skip = 0;
    rs.push_back({"no tables (resident)", t, B, g0->f});
    t = good;
    t.s[1].ids = ids2.as<int32_t>();
    rs.push_back({"different IDs", t, B, g0->f});
    t = good;
    t.s[1].meta = t.s[0].meta;
    rs.push_back({"one meta buffer", t, B, g0->f});
    t = good;
    t.s[1].skip = 2;
    rs.push_back({"different skip", t, B, g0->f});
    if (gq != gus.end()) {
        rs.push_back({"Q8_0", good, B, fmt::q8_0});
    }
    bool refused = ok && gate_up_pair_supported(g0->f, g0->fg, g0->fu, B, good);
    for (const refusal & r : rs) {
        const bool sup = gate_up_pair_supported(r.f, g0->fg, g0->fu, r.rows, r.a);
        const bool ran = gate_up_pair(r.f, g0->fg, g0->fu, r.rows, r.a, cudaStreamPerThread);
        refused = refused && !sup && !ran;
        if (sup || ran) {
            fprintf(stderr, "ex-test: me10 pair refusal %s: %s\n", r.what, sup ? "SUPPORTED" : "LAUNCHED");
        }
    }
    std::vector<uint8_t> gv(glu0.n);
    const bool untouched = ok && cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "refusals") &&
                           glu0.get(gv.data(), glu0.n) &&
                           std::all_of(gv.begin(), gv.end(), [](uint8_t v) { return v == 0xa5; });
    printf("ex-test: me10 gate_up_pair refusals (%zu: B 3, map 2, resident, different IDs, one meta, different "
           "skip%s): %s, nothing enqueued: %s\n",
           rs.size(), gq != gus.end() ? ", Q8_0" : "", refused ? "all refused" : "NOT ALL REFUSED",
           untouched ? "yes" : "NO");
    check(c, refused && untouched, "me10 gate_up_pair refusals");
}

// FMU2 (build item 1): gate_up_lanes and down_lanes against each lane's own gate_up and down launches at B 5 (the joint
// round's lanes on one card), for 2, 3 and 4 lanes, the five IQ formats' first deployed triple (map 3, and map 2 for
// the IQ2 formats) and every resident down pair, the activations as x and through each lane's X-HMIX slot. Lane k
// takes its own case (the master and SP4's route cases, rotated), its own GLU, F32 GLU, meta, IDs, weights, shared
// gates and output; the status word is one for the lanes form (the joint frame's) and each launch's own for the
// separate form. Down reads each form's gate_up meta, as the layer does. Per lane every byte of the GLU, the F32 GLU,
// the error word and the output bitwise equal, the meta by its defined words (meta_defined_bytes) against the host
// reference and between the forms, every status 0. Then refusals: one lane, five lanes, a shared GLU or meta buffer,
// a table (wt), a Q8_0 down.
// FMU2 (build item 2, the joint draft step: lane k is row k of one step at B = U): row r of the master alone (B = 1)
row_case row_of(const row_case & mc, int32_t r) {
    row_case rc;
    rc.rows = 1;
    rc.ids.assign(mc.ids.begin() + std::ptrdiff_t(r) * NU, mc.ids.begin() + std::ptrdiff_t(r + 1) * NU);
    rc.wts.assign(mc.wts.begin() + std::ptrdiff_t(r) * NU, mc.wts.begin() + std::ptrdiff_t(r + 1) * NU);
    rc.sg.assign(1, mc.sg[size_t(r)]);
    rc.xq.assign(act_bytes(1, D), 0);
    std::memcpy(rc.xq.data(), mc.xq.data() + size_t(r) * D, size_t(D));
    std::memcpy(rc.xq.data() + fn::act::codes_bytes(1, D),
                mc.xq.data() + fn::act::codes_bytes(max_rows, D) + fn::act::scales_bytes(r, D),
                fn::act::scales_bytes(1, D));
    rc.xd.assign(size_t(D), 0.0);
    act_values(rc.xq.data(), 1, D, 1, rc.xd.data());
    rc.m = make_meta(rc.ids, 1);
    return rc;
}

// FMU2 (build item 2): the draft's instances (Q8_0 routed, ME7's Q2_0 routed; the Q8_0 shared expert) at B = 2 .. 4,
// each row's GLU (its routed pairs and shared row, both cards) and down output bitwise the same row alone at B = 1
void fmu2_rows_alone(test_ctx & c, const std::vector<gu_inst_t> & gus, const std::vector<dn_inst_t> & dns,
                     const row_case & master) {
    constexpr int32_t U = 4; // the draft's experts run at B <= 4 (experts.h)
    std::vector<row_case> alone;
    for (int32_t r = 0; r < U; ++r) {
        alone.push_back(row_of(master, r));
    }
    for (const gu_inst_t & g : gus) {
        if ((g.f != fmt::q8_0 && g.f != fmt::q2_0) || g.fg != fn::dense::fmt::q8_0 || g.fu != fn::dense::fmt::q8_0) {
            continue;
        }
        gu_out a1[U];
        bool ok = true;
        for (int32_t r = 0; r < U && ok; ++r) {
            ok = check_gate_up(c, g.f, g.fg, g.fu, alone[size_t(r)], false, "row alone", a1[r]);
        }
        const auto dn =
            std::find_if(dns.begin(), dns.end(), [&](const dn_inst_t & d) { return d.f == g.f && d.fs == fmt::q8_0; });
        std::vector<float> o1[U][2];
        for (int32_t r = 0; r < U && ok && dn != dns.end(); ++r) {
            ok = check_down(c, dn->f, dn->fs, alone[size_t(r)], a1[r], false, "row alone", o1[r]);
        }
        for (int32_t B = 2; B <= U && ok; ++B) {
            const row_case pc = prefix(master, B);
            gu_out ob;
            std::vector<float> outb[2];
            ok = check_gate_up(c, g.f, g.fg, g.fu, pc, false, "rows", ob) &&
                 (dn == dns.end() || check_down(c, dn->f, dn->fs, pc, ob, false, "rows", outb));
            bool same_gu = ok, same_dn = ok;
            for (int32_t card = 0; card < 2 && ok; ++card) {
                for (int32_t r = 0; r < B; ++r) {
                    const row_case & a = alone[size_t(r)];
                    for (int32_t k = 0; k < NU; ++k) {
                        same_gu = same_gu && row_glu(ob.glu[card], ob.glu_f32[card], pc.m.slot_of_route(r, k)) ==
                                                 row_glu(a1[r].glu[card], a1[r].glu_f32[card], a.m.slot_of_route(0, k));
                    }
                    same_gu = same_gu && row_glu(ob.glu[card], ob.glu_f32[card], pc.m.n_slots + r) ==
                                             row_glu(a1[r].glu[card], a1[r].glu_f32[card], a.m.n_slots);
                    same_dn = same_dn && (dn == dns.end() || std::memcmp(outb[card].data() + size_t(r) * D,
                                                                         o1[r][card].data(), size_t(D) * 4) == 0);
                }
            }
            printf("ex-test: FMU2 rows alone %-22s B %d: every row's GLU %s, down %s+%s output %s, against the row "
                   "alone at B 1\n",
                   gu_name(g.f, g.fg, g.fu).c_str(), B, same_gu ? "bitwise" : "DIFFERS",
                   dn != dns.end() ? fmt_name(dn->f) : "-", dn != dns.end() ? fmt_name(dn->fs) : "-",
                   dn == dns.end() ? "not run"
                   : same_dn       ? "bitwise"
                                   : "DIFFERS");
            check(c, same_gu && same_dn && dn != dns.end(),
                  gu_name(g.f, g.fg, g.fu) + " B " + std::to_string(B) + ": rows bitwise each row alone");
        }
        check(c, ok, gu_name(g.f, g.fg, g.fu) + ": FMU2 rows alone ran");
    }
}

void fmu2_lanes(test_ctx & c, const std::vector<gu_inst_t> & gus, const std::vector<dn_inst_t> & dns,
                const row_case & master) {
    std::mt19937 rng(0x464d5532u);
    std::vector<row_case> cases = {prefix(master, max_rows)};
    for (row_case & rc : sp4_route_cases(c, master, max_rows, rng)) {
        cases.push_back(std::move(rc));
    }
    std::vector<dn_inst_t> rdn;
    for (const dn_inst_t & d : dns) {
        if ((d.f == fmt::q2_0 || d.f == fmt::iq4_nl) && (d.fs == fmt::q2_0 || d.fs == fmt::iq4_nl)) {
            rdn.push_back(d);
        }
    }
    int32_t n_gu = 0, n_dn = 0, n_bad = 0, n_formats = 0, n_undef = 0;
    void * const s = cudaStreamPerThread;
    for (const fmt f : {fmt::iq2_xxs, fmt::iq2_xs, fmt::iq2_s, fmt::iq3_xxs, fmt::iq3_s}) {
        const auto gi = std::find_if(gus.begin(), gus.end(), [&](const gu_inst_t & g) { return g.f == f; });
        if (gi == gus.end()) {
            continue;
        }
        const gu_inst_t g = *gi;
        ++n_formats;
        const class_table cw = class_weights(g.f, g.fg, g.fu);
        for (const int32_t map : {3, 2}) {
            if (map == 2 && !is_iq2(f)) {
                continue;
            }
            for (int32_t lanes = 2; lanes <= max_lanes; ++lanes) {
                const int32_t card = lanes % 2;
                const dev_mem * w = dev_packed(c, f, false, card);
                for (const int32_t hm : {0, 1}) {
                    const int32_t first = (lanes + hm + map) % int32_t(cases.size());
                    // per lane: its inputs; per form (0 separate, 1 lanes) and lane: its outputs
                    std::vector<std::unique_ptr<dev_mem>> x, ids, wts, sg, hs;
                    std::unique_ptr<dev_mem> glu[2][max_lanes], gf[2][max_lanes], meta[2][max_lanes], er[2][max_lanes],
                        out[2][max_lanes], st[2][max_lanes];
                    bool ok = w != nullptr;
                    const auto rc_of = [&](int32_t k) -> const row_case & {
                        return cases[size_t((first + k) % int32_t(cases.size()))];
                    };
                    for (int32_t k = 0; k < lanes && ok; ++k) {
                        const row_case & rc = rc_of(k);
                        x.push_back(std::make_unique<dev_mem>(rc.xq.size()));
                        ids.push_back(std::make_unique<dev_mem>(rc.ids.size() * 4));
                        wts.push_back(std::make_unique<dev_mem>(rc.wts.size() * 4));
                        sg.push_back(std::make_unique<dev_mem>(rc.sg.size() * 4));
                        hs.push_back(std::make_unique<dev_mem>(size_t(fn::xchg::hmix_layout::slot_words) * 4));
                        const std::vector<uint32_t> hw = hmix_slot(rc, card, false);
                        ok = x.back()->put(rc.xq.data(), rc.xq.size()) &&
                             ids.back()->put(rc.ids.data(), ids.back()->n) &&
                             wts.back()->put(rc.wts.data(), wts.back()->n) &&
                             sg.back()->put(rc.sg.data(), sg.back()->n) && hs.back()->put(hw.data(), hs.back()->n);
                        for (int32_t form = 0; form < 2 && ok; ++form) {
                            glu[form][k] = std::make_unique<dev_mem>(glu_buf_bytes);
                            gf[form][k] = std::make_unique<dev_mem>(size_t(glu_slots) * n_ff_card * 4);
                            meta[form][k] = std::make_unique<dev_mem>(meta_bytes);
                            er[form][k] = std::make_unique<dev_mem>(4);
                            out[form][k] = std::make_unique<dev_mem>(size_t(max_rows) * D * 4);
                            st[form][k] = std::make_unique<dev_mem>(4);
                            for (const dev_mem * d :
                                 {glu[form][k].get(), gf[form][k].get(), meta[form][k].get(), out[form][k].get()}) {
                                ok = ok && cuda_ok(cudaMemset(d->p, 0xa5, d->n), "memset");
                            }
                            ok = ok && cuda_ok(cudaMemset(er[form][k]->p, 0, 4), "memset") &&
                                 cuda_ok(cudaMemset(st[form][k]->p, 0, 4), "memset");
                        }
                    }
                    // the lanes form's one status word: lane 0's
                    const auto status_of = [&](int32_t form, int32_t k) {
                        return st[form][form == 1 ? 0 : k]->as<uint32_t>();
                    };
                    const auto gargs = [&](int32_t form, int32_t k) {
                        const row_case & rc = rc_of(k);
                        gate_up_args a = {w->p,
                                          c.sh_gu.at(int32_t(g.fg)).d[card]->p,
                                          c.sh_gu.at(int32_t(g.fu)).d[card]->p,
                                          c.d_grids.p,
                                          act_in(x[size_t(k)]->p, rc.rows, D),
                                          ids[size_t(k)]->as<int32_t>(),
                                          meta[form][k]->p,
                                          act_out(glu[form][k]->p, glu_slots, n_ff_card),
                                          gf[form][k]->as<float>(),
                                          status_of(form, k)};
                        a.cw = cw;
                        a.map = map;
                        if (hm == 1) {
                            a.x = {nullptr, nullptr, rc.rows, D};
                            a.hmix = hs[size_t(k)]->as<float>();
                            a.hmix_peer_half = 1 - card;
                            a.err = er[form][k]->as<uint32_t>();
                        }
                        return a;
                    };
                    gate_up_lanes_args la{};
                    for (int32_t k = 0; k < lanes && ok; ++k) {
                        ok = gate_up(g.f, g.fg, g.fu, max_rows, gargs(0, k), s);
                        la.s[k] = gargs(1, k);
                    }
                    const bool gsup = ok && gate_up_lanes_supported(g.f, g.fg, g.fu, lanes, la);
                    ok = ok && gate_up_lanes(g.f, g.fg, g.fu, lanes, la, s) &&
                         cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "fmu2 lanes gate_up");
                    std::string why;
                    const auto note = [&](const std::string & m) { why += (why.empty() ? "" : "; ") + m; };
                    const auto same_buf = [&](const dev_mem & a0, const dev_mem & a1, const std::string & name) {
                        std::vector<uint8_t> v0(a0.n), v1(a1.n);
                        if (!a0.get(v0.data(), a0.n) || !a1.get(v1.data(), a1.n)) {
                            note(name + ": read back failed");
                            return false;
                        }
                        const int64_t off = first_diff_at(v0, v1);
                        if (off >= 0) {
                            note(name + " first differs at byte " + std::to_string(off));
                        }
                        return off < 0;
                    };
                    bool same = ok && gsup;
                    for (int32_t k = 0; k < lanes && ok; ++k) {
                        const row_case & rc = rc_of(k);
                        const std::string ln = "lane " + std::to_string(k) + " ";
                        same = same_buf(*glu[0][k], *glu[1][k], ln + "GLU") && same;
                        same = same_buf(*gf[0][k], *gf[1][k], ln + "F32 GLU") && same;
                        same = same_buf(*er[0][k], *er[1][k], ln + "error word") && same;
                        std::vector<bool> runs(size_t(rc.m.n_u), true);
                        std::vector<uint16_t> want;
                        const int32_t n_order = pass_order(rc, cw, runs, true, want);
                        const std::vector<uint8_t> defined = meta_defined_bytes(rc, n_order < 0 ? 0 : n_order);
                        std::vector<uint8_t> v[2];
                        for (int32_t form = 0; form < 2; ++form) {
                            v[form].resize(meta_bytes);
                            std::vector<int32_t> mw(meta_bytes / 4);
                            if (!meta[form][k]->get(v[form].data(), meta_bytes)) {
                                same = false;
                                note(ln + "meta read back failed");
                                continue;
                            }
                            std::memcpy(mw.data(), v[form].data(), meta_bytes);
                            if (n_order < 0 || !pass_region_matches(mw, rc, cw, runs, true)) {
                                same = false;
                                note(ln + (form == 0 ? "separate" : "lanes") + " meta differs from the host reference");
                            }
                        }
                        for (size_t i = 0; i < defined.size(); ++i) {
                            if (v[0][i] != v[1][i]) {
                                if (defined[i] != 0) {
                                    same = false;
                                    note(ln + "meta's defined bytes differ at byte " + std::to_string(i));
                                    break;
                                }
                                ++n_undef;
                            }
                        }
                    }
                    ++n_gu;
                    // down on each form's GLU and meta, every resident pair
                    for (const dn_inst_t & d : rdn) {
                        const dev_mem * wd = dev_packed(c, d.f, true, card);
                        bool dok = ok && wd != nullptr;
                        down_lanes_args da{};
                        const auto dargs = [&](int32_t form, int32_t k) {
                            down_args a = {wd->p,
                                           c.sh_dn.at(int32_t(dense_of(d.fs))).d[card]->p,
                                           c.d_grids.p,
                                           act_in(glu[form][k]->p, glu_slots, n_ff_card),
                                           ids[size_t(k)]->as<int32_t>(),
                                           meta[form][k]->p,
                                           wts[size_t(k)]->as<float>(),
                                           sg[size_t(k)]->as<float>(),
                                           out[form][k]->as<float>(),
                                           status_of(form, k)};
                            return a;
                        };
                        for (int32_t k = 0; k < lanes && dok; ++k) {
                            dok = cuda_ok(cudaMemset(out[0][k]->p, 0xa5, out[0][k]->n), "memset") &&
                                  cuda_ok(cudaMemset(out[1][k]->p, 0x5a, out[1][k]->n), "memset") &&
                                  down(d.f, d.fs, max_rows, dargs(0, k), s);
                            da.s[k] = dargs(1, k);
                        }
                        const bool dsup = dok && down_lanes_supported(d.f, d.fs, lanes, da);
                        dok = dok && down_lanes(d.f, d.fs, lanes, da, s) &&
                              cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "fmu2 lanes down");
                        bool dsame = dok && dsup;
                        for (int32_t k = 0; k < lanes && dok; ++k) {
                            dsame = same_buf(*out[0][k], *out[1][k],
                                             "lane " + std::to_string(k) + " down " + fmt_name(d.f) + "+" +
                                                 fmt_name(d.fs) + " out") &&
                                    dsame;
                        }
                        ++n_dn;
                        same = same && dsame;
                        if (!dsup) {
                            note(std::string("down ") + fmt_name(d.f) + "+" + fmt_name(d.fs) + " NOT SUPPORTED");
                        }
                    }
                    uint32_t status_or = 0;
                    for (int32_t form = 0; form < 2 && ok; ++form) {
                        for (int32_t k = 0; k < lanes; ++k) {
                            uint32_t v = 0;
                            ok = st[form][k]->get(&v, 4) && ok;
                            status_or |= v;
                        }
                    }
                    if (!ok || !same || status_or != 0) {
                        ++n_bad;
                        fprintf(stderr, "ex-test: fmu2 lanes %s map %d lanes %d%s: %s%s%s%s\n",
                                gu_name(g.f, g.fg, g.fu).c_str(), map, lanes, hm == 1 ? " (X-HMIX)" : "",
                                !ok ? "FAILED TO RUN " : "", !gsup ? "NOT SUPPORTED " : "",
                                status_or != 0 ? "status set " : (same ? "" : "DIFFERS"),
                                why.empty() ? "" : (": " + why).c_str());
                    }
                }
            }
        }
    }
    printf("ex-test: fmu2 gate_up_lanes + down_lanes against each lane's own launches at B 5: %d formats, %d gate_up "
           "sets (lanes 2-4, map 3 and IQ2 map 2, x and X-HMIX), %d down sets (%zu resident pairs); GLU, F32 GLU, "
           "error word and output bitwise, meta by its defined words (%d unspecified bytes differ, not compared): "
           "%d differ\n",
           n_formats, n_gu, n_dn, rdn.size(), n_undef, n_bad);
    check(c, n_formats > 0 && n_gu > 0 && n_dn > 0 && n_bad == 0, "fmu2 lanes bitwise each lane's own launches");

    // refusals: nothing enqueued (the outputs keep their fill)
    const gu_inst_t g = gus.front();
    dev_mem x(cases[0].xq.size()), ids(cases[0].ids.size() * 4), wts(cases[0].wts.size() * 4),
        sg(cases[0].sg.size() * 4), tab(size_t(NE) * 8);
    std::unique_ptr<dev_mem> glu[max_lanes + 1], meta[max_lanes + 1], out[max_lanes + 1];
    bool ok = x.put(cases[0].xq.data(), x.n) && ids.put(cases[0].ids.data(), ids.n) &&
              wts.put(cases[0].wts.data(), wts.n) && sg.put(cases[0].sg.data(), sg.n) &&
              cuda_ok(cudaMemset(tab.p, 0, tab.n), "memset");
    for (int32_t k = 0; k <= max_lanes && ok; ++k) {
        glu[k] = std::make_unique<dev_mem>(glu_buf_bytes);
        meta[k] = std::make_unique<dev_mem>(meta_bytes);
        out[k] = std::make_unique<dev_mem>(size_t(max_rows) * D * 4);
        ok = cuda_ok(cudaMemset(glu[k]->p, 0xa5, glu[k]->n), "memset") &&
             cuda_ok(cudaMemset(meta[k]->p, 0xa5, meta[k]->n), "memset") &&
             cuda_ok(cudaMemset(out[k]->p, 0xa5, out[k]->n), "memset");
    }
    const dev_mem * w = dev_packed(c, g.f, false, 0);
    const auto ga = [&](int32_t k) {
        gate_up_args a = {w->p,
                          c.sh_gu.at(int32_t(g.fg)).d[0]->p,
                          c.sh_gu.at(int32_t(g.fu)).d[0]->p,
                          c.d_grids.p,
                          act_in(x.p, max_rows, D),
                          ids.as<int32_t>(),
                          meta[k]->p,
                          act_out(glu[k]->p, glu_slots, n_ff_card),
                          nullptr,
                          nullptr};
        a.cw = class_weights(g.f, g.fg, g.fu);
        a.map = 3;
        return a;
    };
    const dn_inst_t dq = rdn.empty() ? dn_inst_t{fmt::q2_0, fmt::q2_0} : rdn.front();
    const dev_mem * wd = dev_packed(c, dq.f, true, 0);
    const auto dna = [&](int32_t k) {
        down_args a = {wd->p,
                       c.sh_dn.at(int32_t(dense_of(dq.fs))).d[0]->p,
                       c.d_grids.p,
                       act_in(glu[k]->p, glu_slots, n_ff_card),
                       ids.as<int32_t>(),
                       meta[k]->p,
                       wts.as<float>(),
                       sg.as<float>(),
                       out[k]->as<float>(),
                       nullptr};
        return a;
    };
    struct refusal {
        const char * what;
        int32_t lanes;
        int32_t kind; // 0 none, 1 shared GLU, 2 shared meta, 3 table, 4 Q8_0 down, 5 map 1
    };
    const refusal rs[] = {{"one lane", 1, 0},      {"five lanes", 5, 0},   {"a shared GLU", 2, 1},
                          {"a shared meta", 2, 2}, {"a table (wt)", 2, 3}, {"a Q8_0 down", 2, 4},
                          {"map 1", 2, 5}};
    bool refused = ok && w != nullptr && wd != nullptr, untouched = true;
    for (const refusal & r : rs) {
        if (!refused) {
            break;
        }
        gate_up_lanes_args la{};
        down_lanes_args da{};
        for (int32_t k = 0; k < std::min(r.lanes, max_lanes); ++k) {
            la.s[k] = ga(k);
            da.s[k] = dna(k);
        }
        if (r.kind == 1) {
            la.s[1].glu = la.s[0].glu;
            da.s[1].glu = da.s[0].glu;
        } else if (r.kind == 2) {
            la.s[1].meta = la.s[0].meta;
            da.s[1].meta = da.s[0].meta;
        } else if (r.kind == 3) {
            la.s[1].wt = tab.as<const uint8_t *>();
            da.s[1].wt = tab.as<const uint8_t *>();
        } else if (r.kind == 5) {
            la.s[1].map = 1;
        }
        const fmt fd = r.kind == 4 ? fmt::q8_0 : dq.f, fsd = r.kind == 4 ? fmt::q8_0 : dq.fs;
        // map 1 refuses gate_up only, a Q8_0 down down only: the other side is not called
        const bool want_g = r.kind != 4, want_d = r.kind != 5;
        const bool gs = want_g && (gate_up_lanes_supported(g.f, g.fg, g.fu, r.lanes, la) ||
                                   gate_up_lanes(g.f, g.fg, g.fu, r.lanes, la, s));
        const bool ds = want_d && (down_lanes_supported(fd, fsd, r.lanes, da) || down_lanes(fd, fsd, r.lanes, da, s));
        if ((want_g && gs) || (want_d && ds)) {
            refused = false;
            fprintf(stderr, "ex-test: fmu2 lanes refusal %s: %s%s\n", r.what, want_g && gs ? "gate_up SUPPORTED " : "",
                    want_d && ds ? "down SUPPORTED" : "");
        }
    }
    ok = cuda_ok(cudaStreamSynchronize(cudaStreamPerThread), "fmu2 refusals");
    for (int32_t k = 0; k <= max_lanes && ok; ++k) {
        for (const dev_mem * d : {glu[k].get(), meta[k].get(), out[k].get()}) {
            std::vector<uint8_t> v(d->n);
            untouched = untouched && d->get(v.data(), d->n) &&
                        std::all_of(v.begin(), v.end(), [](uint8_t b) { return b == 0xa5u; });
        }
    }
    printf("ex-test: fmu2 lanes refusals (%zu: one and five lanes, a shared GLU or meta, a table, a Q8_0 down, map 1): "
           "%s, outputs untouched: %s\n",
           sizeof(rs) / sizeof(rs[0]), refused ? "all refused" : "NOT ALL REFUSED", untouched && ok ? "yes" : "NO");
    check(c, ok && refused && untouched, "fmu2 lanes refusals");
}

} // namespace
} // namespace fn::experts

int fnfl_ex_prep(const char * dir) {
    using namespace fn::experts;
    static const fmt all[9] = {fmt::iq2_xxs, fmt::iq2_xs, fmt::iq2_s, fmt::iq3_xxs, fmt::iq3_s,
                               fmt::q2_0,    fmt::iq4_nl, fmt::q8_0,  fmt::q8_0};
    for (int32_t i = 0; i < 9; ++i) {
        const fmt f = all[i];
        const bool down = i == 8; // DR1: Q8_0 gate/up, then Q8_0 down
        synth s;
        if (!load_synth(f, dir, s, false, down)) {
            fprintf(stderr, "ex-prep: %s failed\n", fmt_name(f));
            return 1;
        }
        printf("ex-prep: %s%s: %d synthetic experts + poison, %zu bytes each (both cards)\n", fmt_name(f),
               down ? " down" : "", n_synth, s.expert_bytes);
    }
    return 0;
}

int fnfl_ex_test(int device, uint32_t rows_mask, const char * formats, const char * dir, const char * base_dir) {
    using namespace fn::experts;
    using dfmt = fn::dense::fmt;
    const std::string sel(formats);
    // the instantiations under test: every deployed one ("all"), the sanitizer's covering set ("san": each routed
    // format and each shared format once, every down pair), or those whose routed gate/up format is listed
    std::vector<gu_inst_t> gus;
    std::vector<dn_inst_t> dns;
    // ME1: "q4", the Unsloth Q4 candidate's own lists (experts_inst.h FN_EXPERTS_Q4_*), gate/up and down
    const bool q4 = sel == "q4";
    // ME7: "q2d", the Q2 draft's own lists (experts_inst.h FN_EXPERTS_DRAFT_Q2_*), gate/up and down
    const bool q2d = sel == "q2d";
    for (int32_t i = 0; q2d && i < n_draft_q2_gate_up_inst; ++i) {
        gu_inst_t g{};
        int32_t layer = 0;
        draft_q2_gate_up_inst(i, g.f, g.fg, g.fu, layer);
        gus.push_back(g);
    }
    for (int32_t i = 0; q2d && i < n_draft_q2_down_inst; ++i) {
        dn_inst_t d{};
        int32_t layer = 0;
        draft_q2_down_inst(i, d.f, d.fs, layer);
        dns.push_back(d);
    }
    for (int32_t i = 0; q4 && i < n_q4_gate_up_inst; ++i) {
        gu_inst_t g{};
        int32_t layer = 0;
        q4_gate_up_inst(i, g.f, g.fg, g.fu, layer);
        gus.push_back(g);
    }
    for (int32_t i = 0; q4 && i < n_q4_down_inst; ++i) {
        dn_inst_t d{};
        int32_t layer = 0;
        q4_down_inst(i, d.f, d.fs, layer);
        dns.push_back(d);
    }
    for (int32_t i = 0; !q4 && !q2d && i < n_gate_up_inst; ++i) {
        gu_inst_t g{};
        int32_t layer = 0;
        gate_up_inst(i, g.f, g.fg, g.fu, layer);
        std::string fl = fmt_name(g.f);
        std::transform(fl.begin(), fl.end(), fl.begin(), [](unsigned char ch) { return std::tolower(ch); });
        const bool san = (g.f == fmt::iq2_xxs && g.fg == dfmt::iq4_xs && g.fu == dfmt::q6_k) ||
                         (g.f == fmt::iq2_xs && g.fg == dfmt::q4_k && g.fu == dfmt::q5_k) ||
                         (g.f == fmt::iq2_s && g.fg == dfmt::iq3_s && g.fu == dfmt::iq3_s) ||
                         (g.f == fmt::iq3_xxs && g.fg == dfmt::q4_k && g.fu == dfmt::iq3_s) ||
                         (g.f == fmt::iq3_s && g.fg == dfmt::iq4_xs && g.fu == dfmt::iq4_xs) || g.f == fmt::q8_0;
        if (sel == "all" || (sel == "san" && san) || (sel != "san" && sel.find(fl) != std::string::npos)) {
            gus.push_back(g);
        }
    }
    for (int32_t i = 0; !q4 && !q2d && i < n_down_inst; ++i) {
        dn_inst_t d{};
        int32_t layer = 0;
        down_inst(i, d.f, d.fs, layer);
        dns.push_back(d);
    }
    if (gus.empty()) {
        fprintf(stderr, "ex-test: FORMATS is all, san, q4, q2d or a list of routed gate/up formats (iq2_xxs, ...)\n");
        return 2;
    }
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) {
        return 1;
    }
    test_ctx c;
    if (!grid_buffer(c.grids) || !c.d_grids.put(c.grids.data(), c.grids.size())) {
        fprintf(stderr, "ex-test: grid tables failed\n");
        return 1;
    }
    for (const fmt f : {fmt::iq2_xxs, fmt::iq2_xs, fmt::iq2_s, fmt::iq3_xxs, fmt::iq3_s, fmt::q2_0, fmt::iq4_nl}) {
        if (!load_synth(f, dir, c.routed[rkey(f, is_down(f))])) {
            fprintf(stderr, "ex-test: synthetic %s failed\n", fmt_name(f));
            return 1;
        }
    }
    // DR1: Q8_0 in both shapes
    if (!load_synth(fmt::q8_0, dir, c.routed[rkey(fmt::q8_0, false)]) ||
        !load_synth(fmt::q8_0, dir, c.routed[rkey(fmt::q8_0, true)], false, true)) {
        fprintf(stderr, "ex-test: synthetic Q8_0 failed\n");
        return 1;
    }
    // ME7: Q2_0 in the gate/up shape, the Q2 draft's
    if (q2d && !load_synth(fmt::q2_0, dir, c.routed[rkey(fmt::q2_0, false)], false, false, true)) {
        fprintf(stderr, "ex-test: synthetic Q2_0 gate/up failed\n");
        return 1;
    }
    for (const fmt f : {fmt::q4_k, fmt::q5_k, fmt::q5_1}) { // ME1: the Q4 candidate's routed formats
        if (q4 && !load_synth(f, dir, c.routed[rkey(f, is_down(f))])) {
            fprintf(stderr, "ex-test: synthetic %s failed\n", fmt_name(f));
            return 1;
        }
    }
    int32_t salt = 0;
    for (const dfmt f : {dfmt::iq3_s, dfmt::iq4_xs, dfmt::q4_k, dfmt::q5_k, dfmt::q6_k}) {
        if (!make_shared(f, false, ++salt, c.sh_gu[int32_t(f)])) {
            fprintf(stderr, "ex-test: shared %s failed\n", fn::dense::fmt_name(f));
            return 1;
        }
    }
    for (const dfmt f : {dfmt::q2_0, dfmt::iq4_nl}) {
        if (!make_shared(f, true, ++salt, c.sh_dn[int32_t(f)])) {
            fprintf(stderr, "ex-test: shared down %s failed\n", fn::dense::fmt_name(f));
            return 1;
        }
    }
    // DR1: the shared Q8_0 gate/up and down, salted after the others so their cases keep their data
    if (!make_shared(dfmt::q8_0, false, ++salt, c.sh_gu[int32_t(dfmt::q8_0)]) ||
        !make_shared(dfmt::q8_0, true, ++salt, c.sh_dn[int32_t(dfmt::q8_0)])) {
        fprintf(stderr, "ex-test: shared Q8_0 failed\n");
        return 1;
    }
    // the routed pool: n_synth distinct IDs over [0, 512); every other ID reads the poison expert
    std::mt19937 rng(0x45583154u);
    std::vector<int32_t> all_ids(NE);
    for (int32_t e = 0; e < NE; ++e) {
        all_ids[size_t(e)] = e;
    }
    std::shuffle(all_ids.begin(), all_ids.end(), rng);
    c.map.assign(NE, n_synth);
    for (int32_t i = 0; i < n_synth; ++i) {
        c.pool.push_back(all_ids[size_t(i)]);
        c.map[size_t(all_ids[size_t(i)])] = i;
    }
    for (int32_t i = n_synth; i < 3 * n_synth; ++i) { // XB2: 32 more IDs, on the pool's experts
        c.pool_x.push_back(all_ids[size_t(i)]);
        c.map[size_t(all_ids[size_t(i)])] = i % n_synth;
    }
    for (int32_t i = 3 * n_synth; i < 4 * n_synth; ++i) { // SP4: 16 more (no case before SP4's routes them)
        c.pool_y.push_back(all_ids[size_t(i)]);
        c.map[size_t(all_ids[size_t(i)])] = i % n_synth;
    }
    std::vector<int32_t> Bs;
    for (int32_t b = 1; b <= max_rows; ++b) {
        if ((rows_mask >> (b - 1)) & 1u) {
            Bs.push_back(b);
        }
    }
    const bool inv = std::find(Bs.begin(), Bs.end(), max_rows) != Bs.end() && Bs.size() > 1;
    const row_case master = make_master(c, rng, false);
    row_case cases[max_rows + 1];
    for (const int32_t B : Bs) {
        cases[B] = prefix(master, B);
        // the order under the deployed tables' extremes, the shared class pinned first and every weight tied
        const class_table tables[] = {class_weights(fmt::iq2_xxs, fn::dense::fmt::iq3_s, fn::dense::fmt::iq3_s),
                                      class_weights(fmt::iq2_xxs, fn::dense::fmt::q4_k, fn::dense::fmt::q4_k),
                                      class_weights(fmt::iq3_s, fn::dense::fmt::iq3_s, fn::dense::fmt::iq3_s),
                                      turn_w,
                                      {{1.0f, 1.15f, 1.43f, 1.71f, 1e30f}},
                                      {{1.0f, 1.0f, 1.0f, 1.0f, 1.0f}}};
        bool same = true;
        for (const class_table & cw : tables) {
            bool bad = false;
            same = meta_matches(cases[B], cw, bad) && same;
        }
        printf("ex-test: routing meta and unit order (6 tables) B %d (U %d): device %s the host's\n", B, cases[B].m.n_u,
               same ? "equals" : "DIFFERS from");
        check(c, same, "routing meta B " + std::to_string(B));
    }
    // XB2 Amendment 1: routings that place the shared expert's units across the snake's turn, at B = 3 and B = 4
    // (turn_case), each at the B the row mask holds
    std::vector<row_case> turns;
    {
        std::mt19937 tr(0x58423254u);
        const int32_t spec[4][3] = {{3, 7, 3}, {3, 7, 2}, {4, 7, 4}, {4, 7, 3}}; // B, n_many, many_rows
        for (const auto & t : spec) {
            if ((rows_mask >> (t[0] - 1)) & 1u) {
                turns.push_back(turn_case(c, master, t[0], t[1], t[2], tr));
            }
        }
    }
    for (const row_case & t : turns) {
        bool bad = false, straddles = false;
        const int32_t s0 = shared_start(t, turn_w, straddles);
        printf("ex-test: turn case B %d (U %d): shared ranks %d-%d, %s\n", t.rows, t.m.n_u, s0, s0 + n_group - 1,
               straddles ? "across a snake turn" : "NOT across a turn: FAIL");
        check(c, meta_matches(t, turn_w, bad) && straddles, "routing meta turn B " + std::to_string(t.rows));
    }
    // gate/up: every instantiation at each B, the order of B rotating from one instantiation to the next (a launch's
    // B differs from the one before it)
    gu_out first_glu[max_rows + 1]; // the first instantiation's GLU at each B: down's input
    std::vector<gu_out> turn_glu;   // and at each turn case
    for (size_t gi = 0; gi < gus.size(); ++gi) {
        const gu_inst_t & g = gus[gi];
        gu_out out[max_rows + 1];
        bool ran[max_rows + 1] = {};
        for (size_t k = 0; k < Bs.size(); ++k) {
            const int32_t B = Bs[(k + gi) % Bs.size()];
            ran[B] = check_gate_up(c, g.f, g.fg, g.fu, cases[B], false, "case", out[B]);
        }
        if (inv && ran[max_rows]) {
            for (const int32_t B : Bs) {
                if (B != max_rows && ran[B]) {
                    const bool same = same_rows_gu(out[B], cases[B], out[max_rows], cases[max_rows]);
                    printf("ex-test: batch invariance %-22s B %d against B 5: GLU rows %s\n",
                           gu_name(g.f, g.fg, g.fu).c_str(), B, same ? "bitwise equal" : "DIFFER");
                    check(c, same, gu_name(g.f, g.fg, g.fu) + " B " + std::to_string(B) + ": GLU rows batch invariant");
                }
            }
        }
        // SP4: the candidate mapping at B 4 and 5 of an instance it changes, on the same cases: bitwise the 640-thread
        // kernels' GLU (out), and its rows batch invariant against every other B's
        if (map_covers(sp4_map, g.f)) {
            gu_out p[max_rows + 1];
            bool pr[max_rows + 1] = {};
            for (const int32_t B : Bs) {
                if (B >= 4 && ran[B]) {
                    pr[B] = sp4_case(c, g, cases[B], "case", p[B], &out[B]);
                }
            }
            if (inv && pr[max_rows]) {
                for (const int32_t B : Bs) {
                    if (B != max_rows && (pr[B] || (B < 4 && ran[B]))) {
                        const bool same = same_rows_gu(pr[B] ? p[B] : out[B], cases[B], p[max_rows], cases[max_rows]);
                        printf("ex-test: batch invariance %-22s B %d%s%s against B 5 %s: GLU rows %s\n",
                               gu_name(g.f, g.fg, g.fu).c_str(), B, pr[B] ? " " : "", pr[B] ? map_tag(sp4_map) : "",
                               map_tag(sp4_map), same ? "bitwise equal" : "DIFFER");
                        check(c, same, gu_name(g.f, g.fg, g.fu) + " B " + std::to_string(B) +
                                           ": GLU rows batch invariant against B 5 " + map_tag(sp4_map));
                    }
                }
            }
        }
        if (gi == 0) {
            for (const int32_t B : Bs) {
                first_glu[B] = out[B];
            }
        }
        // ME1: the expert cache's tables and CPU split, every target format at each B
        if (g.f != fmt::q8_0 && cpu::gate_up_supported(g.f)) {
            for (const int32_t B : Bs) {
                if (ran[B]) {
                    me1_gate_up(c, g, cases[B]);
                }
            }
        }
        for (size_t ti = 0; ti < turns.size(); ++ti) {
            gu_out to;
            if (check_gate_up(c, g.f, g.fg, g.fu, turns[ti], false, "turn", to) && gi == 0) {
                turn_glu.push_back(to);
            }
            if (map_covers(sp4_map, g.f) && turns[ti].rows >= 4) {
                gu_out tp;
                sp4_case(c, g, turns[ti], "turn", tp);
            }
        }
    }
    me1_resolve(c); // ME1: GPU-side resolution's two small kernels
    me3_graph_replay(c); // ME3: resolve captured once and replayed with changed routes and residency
    me3_wait_post(c);    // ME3: a replay's waits and stamp as kernels
    me8_routes(c, gus, dns, Bs, q2d); // ME8: one router for both cards, captured once and replayed on new routes
    me10_finish(c, master, Bs);       // ME10: the one-device finish against combine + X-F's exchange
    if (!q2d && !q4) {
        me10_pair(c, gus, master, Bs); // ME10: both shards' gate/up in one launch against two
    }
    if (!q2d && !q4 && std::find(Bs.begin(), Bs.end(), max_rows) != Bs.end()) {
        fmu2_lanes(c, gus, dns, master); // FMU2 (build item 1): the joint round's lanes in one launch, each alone
    }
    fmu2_rows_alone(c, gus, dns, master); // FMU2 (build item 2): the draft's instances, every row each row alone
    me2_policy(c);  // ME2: the cache's host policy
    me2_swap(c);    // ME2: a pass's device ordering
    me2b_lent(c, device); // ME2b: the lent region's borrow, return, poison and failure cleanup
    // down: every instantiation at each B, on the first gate/up instantiation's GLU
    for (size_t di = 0; di < dns.size(); ++di) {
        const dn_inst_t & d = dns[di];
        std::vector<float> out[max_rows + 1][2];
        bool ran[max_rows + 1] = {};
        for (size_t k = 0; k < Bs.size(); ++k) {
            const int32_t B = Bs[(k + di) % Bs.size()];
            ran[B] =
                !first_glu[B].glu[0].empty() && check_down(c, d.f, d.fs, cases[B], first_glu[B], false, "case", out[B]);
        }
        if (inv && ran[max_rows]) {
            for (const int32_t B : Bs) {
                if (B != max_rows && ran[B]) {
                    const bool same = same_rows_dn(out[B], B, out[max_rows]);
                    printf("ex-test: batch invariance down %s+%s B %d against B 5: output rows %s\n", fmt_name(d.f),
                           fmt_name(d.fs), B, same ? "bitwise equal" : "DIFFER");
                    check(c, same,
                          std::string("down ") + fmt_name(d.f) + "+" + fmt_name(d.fs) + " B " + std::to_string(B) +
                              ": output rows batch invariant");
                }
            }
        }
        for (size_t ti = 0; ti < turn_glu.size(); ++ti) {
            std::vector<float> to[2];
            check_down(c, d.f, d.fs, turns[ti], turn_glu[ti], false, "turn", to);
        }
        if ((d.f != fmt::q8_0 || q4) && cpu::down_supported(d.f)) { // ME1, as gate/up's; Q8_0 in the Q4 lists
            for (const int32_t B : Bs) {
                if (ran[B]) {
                    me1_down(c, d, cases[B], first_glu[B]);
                }
            }
        }
    }
    // SP4: the candidate mapping's own cases at B 4 and 5, every instance it changes: every nonempty row mask, one-row
    // and fully shared experts, all-disjoint routes (50 experts at B 5), and activations only in the staged runs at the
    // stride-320 loops' edges (319, 320, 399 at B 5); then X-HMIX's acquire, complete and stale ("san": complete only)
    {
        std::mt19937 sr(0x53503455u);
        std::vector<row_case> sp4_cases[max_rows + 1];
        for (const int32_t B : Bs) {
            if (B < 4) {
                continue;
            }
            sp4_cases[B] = sp4_route_cases(c, master, B, sr);
            sp4_cases[B].push_back(spot_case(master, B,
                                             B == 5 ? std::vector<int32_t>{319, 320, 399}
                                                    : std::vector<int32_t>{79, 80, 319},
                                             sr));
            std::set<int32_t> masks; // the row masks the cases' experts hold, all nonempty ones expected
            printf("ex-test: sp4 cases B %d: %zu (U:", B, sp4_cases[B].size());
            for (const row_case & rc : sp4_cases[B]) {
                printf(" %d", rc.m.n_u);
                for (int32_t u = 0; u < rc.m.n_u; ++u) {
                    masks.insert(rc.m.mask[size_t(u)]);
                }
            }
            printf("), row masks %zu of %d\n", masks.size(), (1 << B) - 1);
            check(c, int32_t(masks.size()) == (1 << B) - 1 && masks.count(0) == 0,
                  "sp4 cases B " + std::to_string(B) + ": every nonempty row mask");
        }
        for (const gu_inst_t & g : gus) {
            if (!map_covers(sp4_map, g.f)) {
                continue;
            }
            for (const int32_t B : Bs) {
                for (const row_case & rc : sp4_cases[B]) {
                    gu_out p;
                    sp4_case(c, g, rc, "sp4", p);
                }
            }
        }
        std::set<int32_t> stale_done; // one stale instance per routed format family (IQ2 body, each IQ3 kernel)
        for (const gu_inst_t & g : gus) {
            if (!map_covers(sp4_map, g.f)) {
                continue;
            }
            for (const int32_t B : Bs) {
                if (B < 4) {
                    continue;
                }
                gu_out direct;
                bool ran = true;
                for (int32_t card = 0; card < 2 && ran; ++card) {
                    uint32_t st = 0;
                    ran = run_gate_up(c, g.f, g.fg, g.fu, card, cases[B], direct.glu[card], direct.glu_f32[card], st);
                }
                if (!check(c, ran, "hmix direct " + gu_name(g.f, g.fg, g.fu))) {
                    continue;
                }
                sp4_hmix(c, g, cases[B], direct, false);
                const int32_t fam = g.f == fmt::iq3_xxs || g.f == fmt::iq3_s ? int32_t(g.f) : -1;
                if (sel != "san" && stale_done.insert(fam * 8 + B).second) {
                    sp4_hmix(c, g, cases[B], direct, true);
                }
            }
        }
    }
    // all-zero activations (first instantiations), then an expert ID outside [0, 512): status set, nothing written.
    // ME7 (R8-01, Shell b3334d61): at widths the first instantiation runs. A draft format (Q8_0, Q2_0: B <= 4) at
    // B = 1, the serving drafter's, and the mask's largest B <= 4; its B = 5 refusal stays check_gate_up's. A target
    // format at the mask's largest B, as before
    const int32_t lb = Bs.back();
    const bool draft_first = gus[0].f == fmt::q8_0 || gus[0].f == fmt::q2_0;
    std::vector<int32_t> widths;
    if (draft_first) {
        widths.push_back(1);
        int32_t w4 = 0;
        for (const int32_t B : Bs) {
            w4 = B <= 4 ? std::max(w4, B) : w4;
        }
        if (w4 > 1) {
            widths.push_back(w4);
        }
    } else {
        widths.push_back(lb);
    }
    for (const int32_t wb : widths) {
        const std::string at = gu_name(gus[0].f, gus[0].fg, gus[0].fu) + " B " + std::to_string(wb);
        {
            std::mt19937 zr(7u);
            const row_case rz = prefix(make_master(c, zr, true), wb);
            gu_out gz;
            if (check(c, check_gate_up(c, gus[0].f, gus[0].fg, gus[0].fu, rz, true, "zero", gz),
                      "zero input " + at + ": gate_up ran")) {
                std::vector<float> oz[2];
                check_down(c, dns[0].f, dns[0].fs, rz, gz, true, "zero", oz);
            }
        }
        row_case rb = prefix(master, wb);
        rb.ids[size_t(NU * wb - 1)] = NE + 7;
        rb.m = make_meta(rb.ids, wb);
        bool meta_bad = false;
        check(c, meta_matches(rb, dn_w, meta_bad) && meta_bad, "bad expert ID " + at + ": routing meta");
        std::vector<uint8_t> glu;
        std::vector<float> gf32;
        uint32_t status = 0;
        const bool ran = run_gate_up(c, gus[0].f, gus[0].fg, gus[0].fu, 0, rb, glu, gf32, status);
        bool untouched = ran;
        for (const uint8_t v : glu) {
            untouched = untouched && v == 0xa5;
        }
        printf("ex-test: bad ID %d %s: %s, status %u, outputs %s\n", NE + 7, at.c_str(),
               ran ? "launched" : "NOT LAUNCHED", status, !ran ? "not read" : untouched ? "untouched" : "WRITTEN");
        check(c, ran && status == 1 && untouched, "bad expert ID " + at);
        // SP4: the same in the candidate mapping, on the first instance it changes, at B 4 and 5; XB3 C1: and on the
        // first IQ3 instance (the IQ3 body)
        const auto gc_iq2 =
            std::find_if(gus.begin(), gus.end(), [](const gu_inst_t & g) { return map_covers(sp4_map, g.f); });
        const auto gc_iq3 = std::find_if(gus.begin(), gus.end(), [](const gu_inst_t & g) {
            return map_covers(sp4_map, g.f) && (g.f == fmt::iq3_xxs || g.f == fmt::iq3_s);
        });
        for (const auto gc : {gc_iq2, gc_iq3}) {
            if (wb < 4 || gc == gus.end() || (gc == gc_iq3 && gc_iq3 == gc_iq2)) {
                continue;
            }
            uint32_t st = 0;
            const bool r2 = run_gate_up(c, gc->f, gc->fg, gc->fu, 0, rb, glu, gf32, st, {sp4_map});
            bool u2 = r2;
            for (const uint8_t v : glu) {
                u2 = u2 && v == 0xa5;
            }
            printf("ex-test: bad ID %d %s %s B %d: %s, status %u, outputs %s\n", NE + 7, map_tag(sp4_map),
                   gu_name(gc->f, gc->fg, gc->fu).c_str(), wb, r2 ? "launched" : "NOT LAUNCHED", st,
                   !r2 ? "not read" : u2 ? "untouched" : "WRITTEN");
            check(c, r2 && st == 1 && u2, std::string("bad expert ID ") + map_tag(sp4_map));
        }
    }
    if (base_dir != nullptr) {
        std::vector<const row_case *> all_cases;
        for (const int32_t B : Bs) {
            all_cases.push_back(&cases[B]);
        }
        for (const row_case & t : turns) {
            all_cases.push_back(&t);
        }
        check(c, xb2_bitwise(c, device, base_dir, gus, dns, all_cases), "xb3-bitwise");
        // XB4: the cases at each B, then U cases with n_u of each residue mod 3 at the Bs the row mask holds
        std::vector<const row_case *> dn_cases(all_cases.begin(), all_cases.begin() + std::ptrdiff_t(Bs.size()));
        std::vector<row_case> u_cases;
        std::mt19937 ur(0x58423455u);
        const int32_t spec[][2] = {{2, 20}, {2, 15}, {3, 30}, {3, 28}, {3, 29}, {4, 40}, {4, 39}, {4, 38}}; // B, U
        for (const auto & t : spec) {
            if ((rows_mask >> (t[0] - 1)) & 1u) {
                u_cases.push_back(u_case(c, master, t[0], t[1], ur));
            }
        }
        for (const row_case & t : u_cases) {
            dn_cases.push_back(&t);
        }
        check(c, xb4_bitwise(c, device, base_dir, gus[0], dns, dn_cases), "xb4-bitwise");
    }
    if (c.ok) {
        printf("ex-test: PASS (%d checks)\n", c.n_checks);
        return 0;
    }
    printf("ex-test: FAIL\n");
    return 1;
}
