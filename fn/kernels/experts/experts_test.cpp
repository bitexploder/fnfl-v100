#include "fn/kernels/experts/experts_test.h"

#include "fn/kernels/act/act.h"
#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/experts/grids.h"
#include "fn/model/shapes.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/decode.h"
#include "fn/quant/synth.h"
#include "fn/tools/test_dump.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>
#include <cxxabi.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    static const wfmt q[8] = {wfmt::iq2_xxs, wfmt::iq2_xs, wfmt::iq2_s,  wfmt::iq3_xxs,
                              wfmt::iq3_s,   wfmt::q2_0,   wfmt::iq4_nl, wfmt::q8_0}; // fmt's order
    return q[static_cast<int32_t>(f)];
}

bool is_down(fmt f) {
    return f == fmt::q2_0 || f == fmt::iq4_nl;
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

bool load_synth(fmt f, const std::string & dir, synth & out, bool deployed, bool down) {
    const fn::quant::wfmt q = to_quant(f);
    const bool q8_down = down && f == fmt::q8_0; // DR1: Q8_0's down shape, its own file and seeds
    const bool dn = is_down(f) || q8_down;
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
        dir + "/" + fmt_name(f) + (q8_down ? "-down" : "") + (deployed ? "-fnfl-deployed.bin" : "-fnfl.bin");
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
        const uint64_t case_index = uint64_t(static_cast<int32_t>(f) + (q8_down ? 8 : 0)) * (n_synth + 1) + uint64_t(s);
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
bool region_matches(const std::vector<int32_t> & w, const row_case & rc, const class_table & cw, bool & bad_out) {
    const cpu_meta & m = rc.m;
    bool bad = false;
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

// XB2: write_meta's region of the case's IDs against the host's (region_matches)
bool meta_matches(const row_case & rc, const class_table & cw, bool & bad_out) {
    dev_mem ids(rc.ids.size() * 4), meta(meta_bytes);
    std::vector<int32_t> w(meta_bytes / 4);
    if (!ids.put(rc.ids.data(), ids.n) || !device_meta(ids, rc.rows, cw, meta) || !meta.get(w.data(), meta.n)) {
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
bool map_covers(int32_t map, fmt f) {
    return map == 1 || map == 3 ? f != fmt::q8_0 : map == 2 && is_iq2(f);
}

// SP4: a launch's gate/up mapping (gate_up_args::map) and, when hmix is set, its X-HMIX slot (x is read from it, the
// peer half 1 - card polled; err gets the card's error word, zeroed before each run)
struct gu_opt {
    int32_t map = 0;
    const float * hmix = nullptr;
    uint32_t * err = nullptr;
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
                                  (o.hmix != nullptr ? "-hmix" : "");
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

bool run_down(test_ctx & c, fmt f, fmt fs, int32_t card, const row_case & rc, const std::vector<uint8_t> & glu,
              std::vector<float> & out, uint32_t & status_out) {
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
        const down_args a = {w->p,
                             c.sh_dn.at(int32_t(fsd)).d[card]->p,
                             c.d_grids.p,
                             act_in(g.p, glu_slots, n_ff_card),
                             ids.as<int32_t>(),
                             meta.p,
                             wts.as<float>(),
                             sg.as<float>(),
                             o.as<float>(),
                             st.as<uint32_t>()};
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
                                  std::to_string(card) + "-B" + std::to_string(rc.rows);
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
    if (f == fmt::q8_0 && rc.rows == 5) { // SP2: the draft's Q8_0 layer runs B <= 4 (experts.h); B = 5 is rejected
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
    if (f == fmt::q8_0 && rc.rows == 5) { // SP2: as check_gate_up's
        uint32_t status = 0;
        check(c, !run_down(c, f, fs, 0, rc, go.glu[0], out[0], status),
              std::string(label) + " down " + name + " B 5: rejected (the draft's FFN runs B <= 4)");
        return false;
    }
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
                  sizeof(base_down_args) + 8 == sizeof(down_args),
              "the arguments differ from 71b92a9's by the meta pointer (and gate_up's by its class table, the IQ2 "
              "body's formats, XB3; SP4's map after them)");
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
    for (int32_t i = 0; i < n_gate_up_inst; ++i) {
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
    for (int32_t i = 0; i < n_down_inst; ++i) {
        dn_inst_t d{};
        int32_t layer = 0;
        down_inst(i, d.f, d.fs, layer);
        dns.push_back(d);
    }
    if (gus.empty()) {
        fprintf(stderr, "ex-test: FORMATS is all, san or a list of routed gate/up formats (iq2_xxs, ...)\n");
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
    // all-zero activations (first instantiations), then an expert ID outside [0, 512): status set, nothing written
    const int32_t lb = Bs.back();
    {
        std::mt19937 zr(7u);
        const row_case rz = prefix(make_master(c, zr, true), lb);
        gu_out gz;
        if (check_gate_up(c, gus[0].f, gus[0].fg, gus[0].fu, rz, true, "zero", gz)) {
            std::vector<float> oz[2];
            check_down(c, dns[0].f, dns[0].fs, rz, gz, true, "zero", oz);
        }
    }
    {
        row_case rb = cases[lb];
        rb.ids[size_t(NU * lb - 1)] = NE + 7;
        rb.m = make_meta(rb.ids, lb);
        bool meta_bad = false;
        check(c, meta_matches(rb, dn_w, meta_bad) && meta_bad, "bad expert ID: routing meta");
        std::vector<uint8_t> glu;
        std::vector<float> gf32;
        uint32_t status = 0;
        const bool ran = run_gate_up(c, gus[0].f, gus[0].fg, gus[0].fu, 0, rb, glu, gf32, status);
        bool untouched = ran;
        for (const uint8_t v : glu) {
            untouched = untouched && v == 0xa5;
        }
        printf("ex-test: bad ID %d: status %u, outputs %s\n", NE + 7, status, untouched ? "untouched" : "WRITTEN");
        check(c, ran && status == 1 && untouched, "bad expert ID");
        // SP4: the same in the candidate mapping, on the first instance it changes, at B 4 and 5; XB3 C1: and on the
        // first IQ3 instance (the IQ3 body)
        const auto gc_iq2 =
            std::find_if(gus.begin(), gus.end(), [](const gu_inst_t & g) { return map_covers(sp4_map, g.f); });
        const auto gc_iq3 = std::find_if(gus.begin(), gus.end(), [](const gu_inst_t & g) {
            return map_covers(sp4_map, g.f) && (g.f == fmt::iq3_xxs || g.f == fmt::iq3_s);
        });
        for (const auto gc : {gc_iq2, gc_iq3}) {
            if (lb < 4 || gc == gus.end() || (gc == gc_iq3 && gc_iq3 == gc_iq2)) {
                continue;
            }
            uint32_t st = 0;
            const bool r2 = run_gate_up(c, gc->f, gc->fg, gc->fu, 0, rb, glu, gf32, st, {sp4_map});
            bool u2 = r2;
            for (const uint8_t v : glu) {
                u2 = u2 && v == 0xa5;
            }
            printf("ex-test: bad ID %d %s %s: status %u, outputs %s\n", NE + 7, map_tag(sp4_map),
                   gu_name(gc->f, gc->fg, gc->fu).c_str(), st, u2 ? "untouched" : "WRITTEN");
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
