#include "fn/kernels/ple/ple_test.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/embed/embed.h"
#include "fn/kernels/ple/ple.h"
#include "fn/kernels/round/round.h"
#include "fn/kernels/round/version.h"
#include "fn/model/model.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/runtime/ple_host.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace pl = fn::ple;
namespace rt = fn::runtime;
namespace x = fn::xchg;

constexpr int32_t D = 2560;
constexpr int32_t F = 1280;
constexpr int32_t S = 4;
constexpr int32_t CH = pl::n_channel;
constexpr int32_t NH = pl::n_hist;
constexpr int32_t V = fn::embed::n_vocab;
constexpr int32_t E_ROW = fn::embed::row_bytes;
constexpr int32_t EMB_FULL_ROW = 2 * E_ROW; // token_embd: 10 superblocks a token

bool ok_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "ple-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// splitmix64: the test's seeded values
struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    float uni(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() >> 40) * (1.0f / 16777216.0f); }
    int32_t below(int32_t n) { return static_cast<int32_t>(next() % uint64_t(n)); }
};

// ---- the model's files ----------------------------------------------------------------------------------------------

struct files {
    std::vector<std::string> paths;
    std::vector<fnfl_gguf_dir> dirs;
};

bool open_files(const std::string & first, files & f) {
    std::string err;
    if (!fnfl_gguf_shard_paths(first, f.paths, err)) {
        fprintf(stderr, "ple-test: %s\n", err.c_str());
        return false;
    }
    for (const std::string & p : f.paths) {
        f.dirs.emplace_back();
        if (!fnfl_gguf_read_dir(p, f.dirs.back(), err)) {
            fprintf(stderr, "ple-test: %s\n", err.c_str());
            return false;
        }
    }
    return true;
}

const fnfl_gguf_tensor * find(const files & f, const std::string & name, size_t & d) {
    for (d = 0; d < f.dirs.size(); ++d) {
        for (const fnfl_gguf_tensor & t : f.dirs[d].tensors) {
            if (t.name == name) {
                return &t;
            }
        }
    }
    return nullptr;
}

bool read_at(const std::string & path, uint64_t off, void * dst, size_t n) {
    FILE * fp = fopen(path.c_str(), "rb");
    const bool ok = fp != nullptr && fseeko(fp, static_cast<off_t>(off), SEEK_SET) == 0 && fread(dst, 1, n, fp) == n;
    if (fp != nullptr) {
        fclose(fp);
    }
    return ok;
}

// a whole tensor of the given type and byte count
bool read_tensor(const files & f, const std::string & name, const char * type, uint64_t bytes, void * dst) {
    size_t d = 0;
    const fnfl_gguf_tensor * t = find(f, name, d);
    if (t == nullptr || t->type_name != type || t->bytes != bytes || !read_at(f.dirs[d].path, t->offset, dst, bytes)) {
        fprintf(stderr, "ple-test: %s: missing, not %s, not %llu bytes or unreadable\n", name.c_str(), type,
                static_cast<unsigned long long>(bytes));
        return false;
    }
    return true;
}

// layer 1's PLE tensors, whole (both cards' slices are cut from them)
struct ple_w {
    std::vector<uint16_t> key;                 // [10240][2560] BF16
    std::vector<uint16_t> value;               // [2560][2560] BF16
    std::vector<float> nk, nq, nc;             // [10240]
    std::vector<uint16_t> conv;                // [10240][4] F16
    std::vector<float> key_f, value_f, conv_f; // widened (exact)
};

bool read_ple_w(const files & f, ple_w & w) {
    w.key.resize(size_t(4) * D * D);
    w.value.resize(size_t(D) * D);
    w.nk.resize(size_t(4) * D);
    w.nq.resize(size_t(4) * D);
    w.nc.resize(size_t(4) * D);
    w.conv.resize(size_t(4) * D * 4);
    if (!read_tensor(f, "blk.1.ple_key.weight", "bf16", w.key.size() * 2, w.key.data()) ||
        !read_tensor(f, "blk.1.ple_value.weight", "bf16", w.value.size() * 2, w.value.data()) ||
        !read_tensor(f, "blk.1.ple_norm_key.weight", "f32", w.nk.size() * 4, w.nk.data()) ||
        !read_tensor(f, "blk.1.ple_norm_query.weight", "f32", w.nq.size() * 4, w.nq.data()) ||
        !read_tensor(f, "blk.1.ple_norm_conv.weight", "f32", w.nc.size() * 4, w.nc.data()) ||
        !read_tensor(f, "blk.1.ple_conv1d.weight", "f16", w.conv.size() * 2, w.conv.data())) {
        return false;
    }
    w.key_f.resize(w.key.size());
    w.value_f.resize(w.value.size());
    w.conv_f.resize(w.conv.size());
    for (size_t i = 0; i < w.key.size(); ++i) {
        w.key_f[i] = fn::quant::bf16_to_f32(w.key[i]);
    }
    for (size_t i = 0; i < w.value.size(); ++i) {
        w.value_f[i] = fn::quant::bf16_to_f32(w.value[i]);
    }
    for (size_t i = 0; i < w.conv.size(); ++i) {
        w.conv_f[i] = fn::quant::f16_to_f32(w.conv[i]);
    }
    return true;
}

// ---- host: an independent implementation of the quoted window and hash ---------------------------------------------
// Written from the quoted lines without ple_host's code: the window as positions (p - 1, p - 2) where a predecessor
// is EOS if it is missing or it, or any position after it up to p - 1, is EOS; the hash as the XOR of 64-bit products.

void ref_indices(const rt::ple_params & p, const std::vector<int32_t> & seq, int64_t pos, int32_t (&out)[16]) {
    int64_t w[3] = {seq[size_t(pos)], 0, 0};
    for (int32_t back = 1; back <= 2; ++back) {
        bool eos = false;
        for (int32_t k = 1; k <= back; ++k) { // the predecessor itself and every predecessor nearer to pos
            const int64_t q = pos - k;
            eos = eos || q < 0 || seq[size_t(q)] == p.eos;
        }
        w[back] = eos ? p.eos : seq[size_t(pos - back)];
    }
    const uint64_t m2 = uint64_t(w[0]) * p.mult[0] ^ uint64_t(w[1]) * p.mult[1];
    const uint64_t m3 = m2 ^ uint64_t(w[2]) * p.mult[2];
    for (int32_t g = 0; g < 8; ++g) {
        out[g] = static_cast<int32_t>(m2 % p.vocab[g] + p.offset[g]);
        out[8 + g] = static_cast<int32_t>(m3 % p.vocab[8 + g] + p.offset[8 + g]);
    }
}

struct host_result {
    int64_t windows = 0, index_bad = 0, rows = 0, row_bad = 0, reject_bad = 0;
};

bool host_checks(const rt::ple_params & p, const files & f, const uint8_t * table, host_result & r) {
    // windows: every (t2, t1, t0) over a set with EOS, the tokenizer's EOS, the image token, the ends of the vocabulary
    // and missing predecessors (-1, only as a prefix: a missing predecessor has no predecessor), each as a sequence
    const std::vector<int32_t> set = {0, 1, 7, p.eos, p.eos + 2, p.image, V - 1, 123456, 248043, 248045};
    for (int32_t a = -1; a < int32_t(set.size()); ++a) {
        for (int32_t b = -1; b < int32_t(set.size()); ++b) {
            if (a >= 0 && b < 0) {
                continue;
            }
            for (int32_t c = 0; c < int32_t(set.size()); ++c) {
                std::vector<int32_t> seq;
                if (a >= 0) {
                    seq.push_back(set[size_t(a)]);
                }
                if (b >= 0) {
                    seq.push_back(set[size_t(b)]);
                }
                seq.push_back(set[size_t(c)]);
                int32_t want[16], got[16];
                ref_indices(p, seq, int64_t(seq.size()) - 1, want);
                const int32_t t1 = seq.size() >= 2 ? seq[seq.size() - 2] : -1;
                const int32_t t2 = seq.size() >= 3 ? seq[seq.size() - 3] : -1;
                rt::ple_row_indices(p, seq.back(), t1, t2, got);
                r.windows++;
                r.index_bad += std::memcmp(want, got, sizeof(want)) != 0 ? 1 : 0;
            }
        }
    }
    // the round form: ledgers of 0 .. 5 tokens and 1 .. 5 rows (SP2) over a sequence with EOS and image runs
    rng g(0x51e1u);
    std::vector<int32_t> seq(64);
    for (size_t i = 0; i < seq.size(); ++i) {
        const int32_t u = g.below(10);
        seq[i] = u == 0 ? p.eos : u == 1 ? p.image : g.below(V);
    }
    for (int64_t n0 = 0; n0 <= 5; ++n0) {
        for (int32_t B = 1; B <= rt::ple_max_rows; ++B) {
            for (int64_t start = n0; start + B <= int64_t(seq.size()); start += 7) {
                int32_t idx[rt::ple_max_rows * 16];
                if (!rt::ple_indices(p, seq.data(), start, seq.data() + start, B, idx)) {
                    r.index_bad++;
                    continue;
                }
                for (int32_t t = 0; t < B; ++t) {
                    int32_t want[16];
                    ref_indices(p, seq, start + t, want);
                    r.windows++;
                    r.index_bad += std::memcmp(want, idx + int64_t{16} * t, sizeof(want)) != 0 ? 1 : 0;
                }
            }
        }
    }
    // the gathered rows against rows read from the file and decoded independently
    size_t d = 0;
    const fnfl_gguf_tensor * t = find(f, "per_layer_token_embd.weight", d);
    if (t == nullptr) {
        return false;
    }
    for (int32_t k = 0; k < 64; ++k) {
        const int32_t tok[4] = {g.below(V), k % 3 == 0 ? p.eos : g.below(V), k % 5 == 0 ? p.image : g.below(V),
                                g.below(V)};
        int32_t idx[64];
        std::vector<float> e(size_t(4) * D);
        if (!rt::ple_indices(p, seq.data(), 2, tok, 4, idx) || !rt::ple_gather(p, table, idx, 4, e.data())) {
            r.row_bad++;
            continue;
        }
        for (int32_t j = 0; j < 64; ++j) {
            uint8_t raw[rt::ple_row_bytes];
            float want[rt::ple_head_dim];
            if (!read_at(f.dirs[d].path, t->offset + uint64_t(idx[j]) * rt::ple_row_bytes, raw, sizeof(raw))) {
                r.row_bad++;
                continue;
            }
            for (int32_t b = 0; b < rt::ple_head_dim / 32; ++b) {
                fn::quant::decode_block(fn::quant::wfmt::iq4_nl, raw + int64_t{18} * b, want + int64_t{32} * b);
            }
            r.rows++;
            r.row_bad += std::memcmp(want, e.data() + int64_t(j / 16) * D + int64_t(j % 16) * rt::ple_head_dim,
                                     sizeof(want)) != 0;
        }
    }
    // out-of-range IDs are rejected before any gather or upload
    const int32_t bad[3] = {-1, V, 1 << 30};
    for (const int32_t b : bad) {
        const int32_t tok[2] = {5, b};
        int32_t idx[32];
        r.reject_bad += rt::ple_indices(p, nullptr, 0, tok, 2, idx) ? 1 : 0;
        r.reject_bad += fn::embed::check_tokens(tok, 2) ? 1 : 0;
    }
    return true;
}

// ---- device buffers of one card ---------------------------------------------------------------------------------

constexpr int32_t R = pl::max_rows; // SP2: the device buffers' rows (5)

template <typename T> bool dalloc(T *& p, size_t n, const char * what) {
    return ok_cuda(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)), what);
}

struct card {
    int32_t dev = 0;
    cudaStream_t st = nullptr;
    bool own_stream = false;
    // PLE
    uint8_t * key = nullptr;
    uint8_t * value = nullptr;
    float * nk = nullptr;
    float * nq = nullptr;
    float * nc = nullptr;
    uint16_t * conv = nullptr;
    float * e = nullptr;
    float * hres = nullptr;
    float * hpost = nullptr;
    float * hx = nullptr;
    float * h_out = nullptr;
    float * hist = nullptr;
    uint32_t * cur = nullptr; // the committed slot (SR1), advanced by commit_both
    uint32_t cur_h = 0;       // its host mirror
    uint8_t * ws = nullptr;
    // embedding
    uint8_t * table = nullptr; // [V][550]
    uint8_t * grids = nullptr;
    int32_t * tok = nullptr;
    float * ehres = nullptr;
    float * ehpost = nullptr;
    float * ehx = nullptr;
};

bool alloc_card(card & c) {
    return ok_cuda(cudaSetDevice(c.dev), "device") && dalloc(c.key, size_t(CH) * D * 2, "key") &&
           dalloc(c.value, size_t(F) * D * 2, "value") && dalloc(c.nk, size_t(CH), "nk") &&
           dalloc(c.nq, size_t(CH), "nq") && dalloc(c.nc, size_t(CH), "nc") && dalloc(c.conv, size_t(CH) * 4, "conv") &&
           dalloc(c.e, size_t(R) * D, "e") && dalloc(c.hres, size_t(R) * CH, "hres") &&
           dalloc(c.hpost, size_t(R) * 4, "hpost") && dalloc(c.hx, size_t(R) * F, "hx") &&
           dalloc(c.h_out, size_t(R) * CH, "h_out") &&
           dalloc(c.hist, size_t(pl::n_version) * pl::hist_version_floats, "hist") && dalloc(c.cur, 1, "cur") &&
           ok_cuda(cudaMemset(c.cur, 0, 4), "cur zero") && dalloc(c.ws, pl::workspace_bytes(), "ws") &&
           pl::prepare(c.ws) && dalloc(c.table, size_t(V) * E_ROW, "table") &&
           dalloc(c.grids, fn::dense::grid_bytes, "grids") && dalloc(c.tok, size_t(R), "tok") &&
           dalloc(c.ehres, size_t(R) * CH, "ehres") && dalloc(c.ehpost, size_t(R) * 4, "ehpost") &&
           dalloc(c.ehx, size_t(R) * F, "ehx") && ok_cuda(cudaMemset(c.table, 0, size_t(V) * E_ROW), "table zero") &&
           ok_cuda(cudaMemset(c.hist, 0, size_t(pl::n_version) * pl::hist_version_floats * 4), "hist zero") &&
           ok_cuda(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking), "stream") && (c.own_stream = true);
}

void free_card(card & c) {
    cudaSetDevice(c.dev);
    void * ps[] = {c.key,   c.value, c.nk, c.nq,    c.nc,    c.conv, c.e,     c.hres,   c.hpost, c.hx,
                   c.h_out, c.hist,  c.ws, c.table, c.grids, c.tok,  c.ehres, c.ehpost, c.ehx,   c.cur};
    for (void * p : ps) {
        cudaFree(p);
    }
    if (c.own_stream && c.st != nullptr) {
        cudaStreamDestroy(c.st);
    }
}

bool upload_weights(card & c, int32_t r, const ple_w & w) {
    std::vector<uint16_t> key(size_t(CH) * D), conv(size_t(CH) * 4);
    std::vector<float> nk(CH), nq(CH), nc(CH);
    for (int32_t s = 0; s < S; ++s) {
        for (int32_t i = 0; i < F; ++i) {
            const size_t g = size_t(s) * D + size_t(r) * F + i, l = size_t(s) * F + i;
            std::memcpy(&key[l * D], &w.key[g * D], size_t(D) * 2);
            std::memcpy(&conv[l * 4], &w.conv[g * 4], 8);
            nk[l] = w.nk[g];
            nq[l] = w.nq[g];
            nc[l] = w.nc[g];
        }
    }
    std::vector<uint8_t> grids;
    return fn::dense::grid_buffer(grids) && ok_cuda(cudaSetDevice(c.dev), "device") &&
           ok_cuda(cudaMemcpy(c.key, key.data(), key.size() * 2, cudaMemcpyHostToDevice), "key") &&
           ok_cuda(cudaMemcpy(c.value, &w.value[size_t(r) * F * D], size_t(F) * D * 2, cudaMemcpyHostToDevice),
                   "value") &&
           ok_cuda(cudaMemcpy(c.conv, conv.data(), conv.size() * 2, cudaMemcpyHostToDevice), "conv") &&
           ok_cuda(cudaMemcpy(c.nk, nk.data(), size_t(CH) * 4, cudaMemcpyHostToDevice), "nk") &&
           ok_cuda(cudaMemcpy(c.nq, nq.data(), size_t(CH) * 4, cudaMemcpyHostToDevice), "nq") &&
           ok_cuda(cudaMemcpy(c.nc, nc.data(), size_t(CH) * 4, cudaMemcpyHostToDevice), "nc") &&
           ok_cuda(cudaMemcpy(c.grids, grids.data(), grids.size(), cudaMemcpyHostToDevice), "grids");
}

pl::step_args step_args_of(const card & c, const x::rings & rg, int32_t r, int64_t use) {
    pl::step_args a;
    a.w.key = c.key;
    a.w.value = c.value;
    a.w.norm_key = c.nk;
    a.w.norm_query = c.nq;
    a.w.norm_conv = c.nc;
    a.w.conv = c.conv;
    a.e = c.e;
    a.hres = c.hres;
    a.hpost = c.hpost;
    a.hx = c.hx;
    a.h_out = c.h_out;
    a.hist = c.hist;
    a.cur = c.cur;
    a.ws = c.ws;
    const int32_t s = static_cast<int32_t>(use % x::n_slot);
    a.stats_peer = rg.slot(1 - r, x::msg::ple, s);
    a.stats_slot = rg.slot(r, x::msg::ple, s);
    a.stats_reset = use > 0 ? rg.slot(r, x::msg::ple, static_cast<int32_t>((use - 1) % x::n_slot)) : nullptr;
    a.err = rg.err(r);
    a.card = r;
    return a;
}

// ---- one PLE round on both cards --------------------------------------------------------------------------------

struct round_in {
    int32_t B = 0;
    std::vector<float> e;     // [B][2560]
    std::vector<float> hres;  // [B][4][2560] full width
    std::vector<float> hpost; // [B][4]
    std::vector<float> hx;    // [B][2560]
};

struct round_out {
    std::vector<float> h_out[2]; // [B][4][1280]
    std::vector<float> scal[2];  // [B][4][4]
    std::vector<float> hist[2];  // [1 + B][5120][9], logical versions 0..B (read from slots version_slot(cur, v))
};

struct pair_ctx {
    card c[2];
    x::rings * rg = nullptr;
    int64_t use = 0; // PLE uses (rounds) so far: picks the X-PLE-STATS slots
    uint32_t slots_seen = 1; // bit s: the committed slot was s at some round (SR1; slot 0 at the start)
};

bool upload_inputs(card & c, int32_t r, const round_in & in) {
    const int32_t B = in.B;
    std::vector<float> hres(size_t(B) * CH), hx(size_t(B) * F);
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t s = 0; s < S; ++s) {
            std::memcpy(&hres[(size_t(b) * S + s) * F], &in.hres[(size_t(b) * S + s) * D + size_t(r) * F],
                        size_t(F) * 4);
        }
        std::memcpy(&hx[size_t(b) * F], &in.hx[size_t(b) * D + size_t(r) * F], size_t(F) * 4);
    }
    return ok_cuda(cudaSetDevice(c.dev), "device") &&
           ok_cuda(cudaMemcpy(c.e, in.e.data(), size_t(B) * D * 4, cudaMemcpyHostToDevice), "e") &&
           ok_cuda(cudaMemcpy(c.hres, hres.data(), hres.size() * 4, cudaMemcpyHostToDevice), "hres") &&
           ok_cuda(cudaMemcpy(c.hpost, in.hpost.data(), size_t(B) * 16, cudaMemcpyHostToDevice), "hpost") &&
           ok_cuda(cudaMemcpy(c.hx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "hx");
}

// Enqueue every send before the receive that reads it (products, stats on both cards, then update on both), so the
// issue order is a valid sequential schedule (loopback); then read the outputs back. No commit.
bool run_round(pair_ctx & p, const round_in & in, round_out & out) {
    const int32_t B = in.B;
    bool ok = upload_inputs(p.c[0], 0, in) && upload_inputs(p.c[1], 1, in);
    pl::step_args a[2] = {step_args_of(p.c[0], *p.rg, 0, p.use), step_args_of(p.c[1], *p.rg, 1, p.use)};
    for (int32_t r = 0; ok && r < 2; ++r) {
        ok = ok_cuda(cudaSetDevice(p.c[r].dev), "device") && pl::products(B, a[r], p.c[r].st);
    }
    for (int32_t r = 0; ok && r < 2; ++r) {
        ok = ok_cuda(cudaSetDevice(p.c[r].dev), "device") && pl::stats(B, a[r], p.c[r].st);
    }
    for (int32_t r = 0; ok && r < 2; ++r) {
        ok = ok_cuda(cudaSetDevice(p.c[r].dev), "device") && pl::update(B, a[r], p.c[r].st);
    }
    for (int32_t r = 0; ok && r < 2; ++r) {
        ok = ok_cuda(cudaSetDevice(p.c[r].dev), "device") && ok_cuda(cudaStreamSynchronize(p.c[r].st), "round");
    }
    p.use++;
    for (int32_t r = 0; ok && r < 2; ++r) {
        uint32_t err = 0;
        ok = p.rg->read_err(r, err) && err == 0;
        if (!ok) {
            fprintf(stderr, "ple-test: card %d error word 0x%x\n", r, err);
        }
        out.h_out[r].resize(size_t(B) * CH);
        out.scal[r].resize(size_t(B) * 16);
        out.hist[r].resize(size_t(1 + B) * pl::hist_version_floats);
        const pl::ws_view v = pl::carve(p.c[r].ws);
        ok = ok && ok_cuda(cudaSetDevice(p.c[r].dev), "device") &&
             ok_cuda(cudaMemcpy(out.h_out[r].data(), p.c[r].h_out, out.h_out[r].size() * 4, cudaMemcpyDeviceToHost),
                     "h_out") &&
             ok_cuda(cudaMemcpy(out.scal[r].data(), v.scal, out.scal[r].size() * 4, cudaMemcpyDeviceToHost), "scal");
        for (int32_t ver = 0; ok && ver <= B; ++ver) {
            const int64_t slot = fn::round::version_slot(p.c[r].cur_h, ver);
            ok = ok_cuda(cudaMemcpy(&out.hist[r][size_t(ver) * pl::hist_version_floats],
                                    p.c[r].hist + slot * pl::hist_version_floats, size_t(pl::hist_version_floats) * 4,
                                    cudaMemcpyDeviceToHost),
                         "hist");
        }
    }
    return ok;
}

// the commit (SR1): the committed slot moves by k on both cards; the device word is checked against the host mirror,
// and every slot it reaches is recorded (the test covers all five)
bool commit_both(pair_ctx & p, int32_t k) {
    bool ok = true;
    for (int32_t r = 0; ok && r < 2; ++r) {
        uint32_t got = 0;
        p.c[r].cur_h = static_cast<uint32_t>(fn::round::version_slot(p.c[r].cur_h, k));
        ok = ok_cuda(cudaSetDevice(p.c[r].dev), "device") && fn::round::advance(p.c[r].cur, k, p.c[r].st) &&
             ok_cuda(cudaStreamSynchronize(p.c[r].st), "commit") &&
             ok_cuda(cudaMemcpy(&got, p.c[r].cur, 4, cudaMemcpyDeviceToHost), "cur") && got == p.c[r].cur_h;
        p.slots_seen |= 1u << p.c[r].cur_h;
    }
    return ok;
}

bool set_hist0(pair_ctx & p, const std::vector<float> (&h0)[2]) {
    bool ok = true;
    for (int32_t r = 0; ok && r < 2; ++r) {
        const int64_t slot = fn::round::version_slot(p.c[r].cur_h, 0);
        ok = ok_cuda(cudaSetDevice(p.c[r].dev), "device") &&
             ok_cuda(cudaMemcpy(p.c[r].hist + slot * pl::hist_version_floats, h0[r].data(),
                                size_t(pl::hist_version_floats) * 4, cudaMemcpyHostToDevice),
                     "hist0");
    }
    return ok;
}

bool get_hist0(pair_ctx & p, std::vector<float> (&h0)[2]) {
    bool ok = true;
    for (int32_t r = 0; ok && r < 2; ++r) {
        h0[r].resize(size_t(pl::hist_version_floats));
        const int64_t slot = fn::round::version_slot(p.c[r].cur_h, 0);
        ok = ok_cuda(cudaSetDevice(p.c[r].dev), "device") &&
             ok_cuda(cudaMemcpy(h0[r].data(), p.c[r].hist + slot * pl::hist_version_floats, h0[r].size() * 4,
                                cudaMemcpyDeviceToHost),
                     "hist0");
    }
    return ok;
}

// ---- the FP64 reference of the pinned form ----------------------------------------------------------------------

double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}
double gate_of(double s) {
    const double m = std::sqrt(std::max(std::fabs(s), 1e-6));
    return sigmoid(s > 0 ? m : s < 0 ? -m : 0.0);
}
// d gate / d s away from the clamp (the amendment's slope)
double slope_of(double s) {
    const double m = std::sqrt(std::max(std::fabs(s), 1e-6));
    const double g = sigmoid(m);
    return g * (1.0 - g) / (2.0 * m);
}

struct ref_row {
    double key[4 * D];
    double value[D];
    double score[S];
    double h[S][D];
};

// products and scores of one row (independent of the other rows)
void ref_scores(const ple_w & w, const round_in & in, int32_t b, ref_row & rr) {
    constexpr double eps = 1e-6;
    const float * e = &in.e[size_t(b) * D];
    for (int32_t n = 0; n < 4 * D; ++n) {
        const float * kw = &w.key_f[size_t(n) * D];
        double acc = 0.0;
        for (int32_t k = 0; k < D; ++k) {
            acc += double(kw[k]) * double(e[k]);
        }
        rr.key[n] = acc;
    }
    for (int32_t n = 0; n < D; ++n) {
        const float * vw = &w.value_f[size_t(n) * D];
        double acc = 0.0;
        for (int32_t k = 0; k < D; ++k) {
            acc += double(vw[k]) * double(e[k]);
        }
        rr.value[n] = acc;
    }
    for (int32_t s = 0; s < S; ++s) {
        // h as the kernel's stored input: FC-LAY1's combine, one F32 fmaf
        for (int32_t j = 0; j < D; ++j) {
            rr.h[s][j] = double(
                std::fmaf(in.hpost[size_t(b) * S + s], in.hx[size_t(b) * D + j], in.hres[(size_t(b) * S + s) * D + j]));
        }
        double sk = 0.0, sq = 0.0;
        for (int32_t j = 0; j < D; ++j) {
            sk += rr.key[s * D + j] * rr.key[s * D + j];
            sq += rr.h[s][j] * rr.h[s][j];
        }
        const double rk = 1.0 / std::sqrt(sk / D + eps), rq = 1.0 / std::sqrt(sq / D + eps);
        double dot = 0.0;
        for (int32_t j = 0; j < D; ++j) {
            dot += (rr.key[s * D + j] * rk * double(w.nk[size_t(s) * D + j])) *
                   (rr.h[s][j] * rq * double(w.nq[size_t(s) * D + j]));
        }
        rr.score[s] = dot / std::sqrt(double(D));
    }
}

struct worst {
    double score = 0, gate_own = 0, gate_ref = 0, update = 0, xerr = 0;
    double gate_ref_ratio = 0; // |gate - FP64| / bound
    double s_min = 1e300, s_max = -1e300, g_min = 1, g_max = 0;
    int64_t zero_scores = 0;
    int64_t rows = 0, explained = 0, fails = 0, hist_bad = 0, scal_diff = 0;
    struct top {
        double bound, score, err, diff;
    };
    std::vector<top> tops;
};

// checks one round's rows against the reference; hist0: the kernel's version 0 (both cards)
void check_round(const ple_w & w, const round_in & in, const round_out & out, worst & wr, bool verbose) {
    constexpr double eps = 1e-6;
    const int32_t B = in.B;
    std::vector<ref_row> rr(static_cast<size_t>(B));
    std::vector<std::thread> th;
    for (int32_t b = 0; b < B; ++b) {
        th.emplace_back([&, b] { ref_scores(w, in, b, rr[size_t(b)]); });
    }
    for (std::thread & t : th) {
        t.join();
    }
    // the reference history: the kernel's version 0, full width [s][j][9]
    std::vector<double> hist(size_t(S) * D * NH);
    for (int32_t r = 0; r < 2; ++r) {
        for (int32_t s = 0; s < S; ++s) {
            for (int32_t i = 0; i < F; ++i) {
                for (int32_t k = 0; k < NH; ++k) {
                    hist[(size_t(s) * D + size_t(r) * F + i) * NH + k] = out.hist[r][(size_t(s) * F + i) * NH + k];
                }
            }
        }
    }
    // both cards' scalars bitwise equal
    wr.scal_diff += std::memcmp(out.scal[0].data(), out.scal[1].data(), out.scal[0].size() * 4) != 0 ? 1 : 0;
    for (int32_t b = 0; b < B; ++b) {
        const ref_row & R = rr[size_t(b)];
        double sv = 0.0;
        for (int32_t j = 0; j < D; ++j) {
            sv += R.value[j] * R.value[j];
        }
        for (int32_t s = 0; s < S; ++s) {
            const float * sc = &out.scal[0][(size_t(b) * S + s) * 4];
            const double ks = sc[0], kg = sc[1];
            const double s64 = R.score[s];
            const double own = std::fabs(ks - s64);
            const double e_score = own / std::max(1.0, std::fabs(s64));
            const double e_gown = std::fabs(kg - gate_of(ks));
            const double bound = 1e-6 + 2.0 * slope_of(s64) * own;
            const double e_gref = std::fabs(kg - gate_of(s64));
            const bool explained = std::fabs(s64) <= 2.0 * own;
            const double G = explained ? kg : gate_of(s64);
            wr.explained += explained ? 1 : 0;
            wr.tops.push_back({bound, s64, own, e_gref});
            wr.s_min = std::min(wr.s_min, s64);
            wr.s_max = std::max(wr.s_max, s64);
            wr.g_min = std::min(wr.g_min, kg);
            wr.g_max = std::max(wr.g_max, kg);
            wr.zero_scores += ks == 0.0 && s64 == 0.0 ? 1 : 0;
            // gated, x, conv and the update of this stream, full width; the kernel's x from version b + 1
            const double xinv = 1.0 / std::sqrt(G * G * (sv / D) + eps);
            double emax = 0.0, rmax = 0.0, xemax = 0.0, xrmax = 0.0;
            for (int32_t j = 0; j < D; ++j) {
                const int32_t r = j / F, i = j % F;
                const size_t c = size_t(s) * F + i;
                const double gated = R.value[j] * G;
                const double xr = gated * xinv * double(w.nc[size_t(s) * D + j]);
                double * hv = &hist[(size_t(s) * D + j) * NH];
                const float * cw = &w.conv_f[(size_t(s) * D + j) * 4];
                const double conv =
                    double(cw[0]) * hv[0] + double(cw[1]) * hv[3] + double(cw[2]) * hv[6] + double(cw[3]) * xr;
                const double ref = R.h[s][j] + (gated + conv / (1.0 + std::exp(-conv)));
                const double got = out.h_out[r][(size_t(b) * S + s) * F + i];
                emax = std::max(emax, std::fabs(got - ref));
                rmax = std::max(rmax, std::fabs(ref));
                const float * kv = &out.hist[r][size_t(b + 1) * pl::hist_version_floats + c * NH];
                const float * kp = &out.hist[r][size_t(b) * pl::hist_version_floats + c * NH];
                xemax = std::max(xemax, std::fabs(double(kv[NH - 1]) - xr));
                xrmax = std::max(xrmax, std::fabs(xr));
                // version b + 1 is version b shifted by one with the kernel's x appended, bitwise
                wr.hist_bad += std::memcmp(kv, kp + 1, size_t(NH - 1) * 4) != 0 ? 1 : 0;
                for (int32_t k = 0; k < NH - 1; ++k) {
                    hv[k] = hv[k + 1];
                }
                hv[NH - 1] = xr;
            }
            const double e_up = emax / std::max(rmax, 1e-300);
            const double e_x = xemax / std::max(xrmax, 1e-300);
            const bool fail = e_score > 1e-5 || e_gown > 1e-6 || e_gref > bound || e_up > 1e-5 || e_x > 1e-5;
            wr.fails += fail ? 1 : 0;
            if (fail && verbose) {
                printf(
                    "ple-test: FAIL row %d stream %d: score %.9g (FP64 %.9g, err %.3g), gate own %.3g, gate ref %.3g "
                    "(bound %.3g), update %.3g, x %.3g\n",
                    b, s, ks, s64, e_score, e_gown, e_gref, bound, e_up, e_x);
            }
            wr.score = std::max(wr.score, e_score);
            wr.gate_own = std::max(wr.gate_own, e_gown);
            wr.gate_ref = std::max(wr.gate_ref, e_gref);
            wr.gate_ref_ratio = std::max(wr.gate_ref_ratio, e_gref / bound);
            wr.update = std::max(wr.update, e_up);
            wr.xerr = std::max(wr.xerr, e_x);
            wr.rows++;
        }
    }
}

// ---- inputs ---------------------------------------------------------------------------------------------------------

// h of a row: per stream scale, a few large features, seeded by (position, variant)
void make_h(uint64_t seed, round_in & in, int32_t b) {
    rng g(seed);
    const float scale[S] = {1.0f, 3.0f, 0.25f, 8.0f};
    for (int32_t s = 0; s < S; ++s) {
        for (int32_t j = 0; j < D; ++j) {
            float v = g.uni(-scale[s], scale[s]);
            if (g.below(512) == 0) {
                v *= 40.0f;
            }
            in.hres[(size_t(b) * S + s) * D + j] = v;
        }
        in.hpost[size_t(b) * S + s] = g.uni(-1.5f, 1.5f);
    }
    for (int32_t j = 0; j < D; ++j) {
        in.hx[size_t(b) * D + j] = g.uni(-2.0f, 2.0f);
    }
}

void resize_in(round_in & in, int32_t B) {
    in.B = B;
    in.e.assign(size_t(B) * D, 0.0f);
    in.hres.assign(size_t(B) * S * D, 0.0f);
    in.hpost.assign(size_t(B) * S, 0.0f);
    in.hx.assign(size_t(B) * D, 0.0f);
}

// Numerical stress on some true positions (pos % 5 == 2): streams 0 and 2 are the row's own key (F32 product on the
// CPU) times nk nq, plus noise, with opposite signs, so |score| reaches tens and the gate nears 0 or 1; on pos % 7 == 3
// stream 1 is all zero (h = 0: the score is exactly 0, sign(0) = 0, gate 1/2).
void stress_h(const ple_w & w, int64_t pos, round_in & in, int32_t b) {
    if (pos % 7 == 3) {
        for (int32_t j = 0; j < D; ++j) {
            in.hres[(size_t(b) * S + 1) * D + j] = 0.0f;
        }
        in.hpost[size_t(b) * S + 1] = 0.0f;
    }
    if (pos % 5 != 2) {
        return;
    }
    rng g(0xa11ull + uint64_t(pos));
    const float * e = &in.e[size_t(b) * D];
    for (const int32_t s : {0, 2}) {
        const float sign = s == 0 ? 1.0f : -1.0f;
        for (int32_t j = 0; j < D; ++j) {
            const float * kw = &w.key_f[(size_t(s) * D + j) * D];
            float k = 0.0f;
            for (int32_t t = 0; t < D; ++t) {
                k = std::fmaf(kw[t], e[t], k);
            }
            in.hres[(size_t(b) * S + s) * D + j] =
                sign * k * w.nk[size_t(s) * D + j] * w.nq[size_t(s) * D + j] * 3.0f + g.uni(-0.01f, 0.01f);
        }
        in.hpost[size_t(b) * S + s] = 0.0f;
    }
}

// a round's rows: row 0 the pending token seq[N], rows t < k_true seq[N + t], later rows drafts; e gathered from the
// table through ple_host over the ledger seq[0 .. N)
bool make_round(const rt::ple_params & pp, const uint8_t * table, const ple_w * w, const std::vector<int32_t> & seq,
                int64_t N, int32_t B, int32_t k_true, uint64_t round_seed, round_in & in, std::vector<int32_t> & tok) {
    resize_in(in, B);
    tok.assign(size_t(B), 0);
    rng g(round_seed);
    for (int32_t t = 0; t < B; ++t) {
        const bool truth = t == 0 || t < k_true;
        tok[size_t(t)] = truth ? seq[size_t(N + t)] : g.below(V);
        make_h(truth ? 0x10000ull + uint64_t(N + t) : round_seed * 7919u + uint64_t(t), in, t);
    }
    int32_t idx[rt::ple_max_rows * 16];
    if (!rt::ple_indices(pp, seq.data(), N, tok.data(), B, idx) || !rt::ple_gather(pp, table, idx, B, in.e.data())) {
        return false;
    }
    for (int32_t t = 0; w != nullptr && t < B; ++t) {
        if (t == 0 || t < k_true) {
            stress_h(*w, N + t, in, t);
        }
    }
    return true;
}

uint64_t fnv(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 0x100000001b3ull;
    }
    return h;
}

uint64_t digest(const round_out & o) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int32_t r = 0; r < 2; ++r) {
        h = fnv(h, o.h_out[r].data(), o.h_out[r].size() * 4);
        h = fnv(h, o.scal[r].data(), o.scal[r].size() * 4);
        h = fnv(h, o.hist[r].data(), o.hist[r].size() * 4);
    }
    return h;
}

// ---- the chained run --------------------------------------------------------------------------------------------

struct chain_result {
    int64_t rounds = 0, positions = 0, k_short = 0;
    uint64_t digest = 0;
};

bool run_chain(pair_ctx & p, const rt::ple_params & pp, const uint8_t * table, const ple_w & w,
               const std::vector<int32_t> & seq, int64_t positions, const std::vector<int32_t> & rows_cycle, worst & wr,
               chain_result & cr, bool check) {
    std::vector<float> zero[2] = {std::vector<float>(size_t(pl::hist_version_floats), 0.0f),
                                  std::vector<float>(size_t(pl::hist_version_floats), 0.0f)};
    if (!set_hist0(p, zero)) {
        return false;
    }
    cr = chain_result();
    cr.digest = 0xcbf29ce484222325ull;
    int64_t N = 0;
    for (int64_t round = 0; N < positions; ++round) {
        const int32_t B = std::min<int32_t>(rows_cycle[size_t(round) % rows_cycle.size()], int32_t(positions - N));
        // commit k = B, except every fifth round k = B - 1 (k = 0 at B = 1 once in a while: nothing accepted)
        int32_t k = B;
        if (round % 5 == 4) {
            k = B - 1;
        }
        if (k == 0 && round % 15 != 4) {
            k = 1;
        }
        round_in in;
        std::vector<int32_t> tok;
        round_out out;
        if (!make_round(pp, table, &w, seq, N, B, k, 0x1000ull + uint64_t(round), in, tok) || !run_round(p, in, out)) {
            return false;
        }
        if (check) {
            check_round(w, in, out, wr, true);
        }
        cr.digest = fnv(cr.digest ^ digest(out), &round, sizeof(round));
        if (!commit_both(p, k)) {
            return false;
        }
        cr.k_short += k < B ? 1 : 0;
        N += k;
        cr.rounds++;
    }
    cr.positions = N;
    return true;
}

// a B-row round against B one-row rounds from the same state (NC1 section 2.2), and a k < B commit against k
// one-row steps; every output bitwise
bool batch_invariance(pair_ctx & p, const rt::ple_params & pp, const uint8_t * table, const std::vector<int32_t> & seq,
                      int64_t N, int32_t B, int32_t k, int64_t & diffs) {
    std::vector<float> h0[2];
    round_in in;
    std::vector<int32_t> tok;
    round_out big;
    if (!get_hist0(p, h0) || !make_round(pp, table, nullptr, seq, N, B, B, 0x77ull + uint64_t(B), in, tok) ||
        !run_round(p, in, big) || !commit_both(p, k)) {
        return false;
    }
    std::vector<float> after_big[2];
    if (!get_hist0(p, after_big) || !set_hist0(p, h0)) {
        return false;
    }
    std::vector<float> after_small[2];
    for (int32_t t = 0; t < B; ++t) {
        round_in one;
        resize_in(one, 1);
        std::memcpy(one.e.data(), &in.e[size_t(t) * D], size_t(D) * 4);
        std::memcpy(one.hres.data(), &in.hres[size_t(t) * S * D], size_t(S) * D * 4);
        std::memcpy(one.hpost.data(), &in.hpost[size_t(t) * S], size_t(S) * 4);
        std::memcpy(one.hx.data(), &in.hx[size_t(t) * D], size_t(D) * 4);
        // the one-row round's e from the ledger extended by rows 0 .. t - 1 equals the big round's row t
        std::vector<int32_t> led(seq.begin(), seq.begin() + N);
        led.insert(led.end(), tok.begin(), tok.begin() + t);
        int32_t idx[16];
        std::vector<float> e1(D);
        if (!rt::ple_indices(pp, led.data(), int64_t(led.size()), &tok[size_t(t)], 1, idx) ||
            !rt::ple_gather(pp, table, idx, 1, e1.data())) {
            return false;
        }
        diffs += std::memcmp(e1.data(), one.e.data(), size_t(D) * 4) != 0 ? 1 : 0;
        round_out o;
        if (!run_round(p, one, o) || !commit_both(p, 1)) {
            return false;
        }
        for (int32_t r = 0; r < 2; ++r) {
            diffs += std::memcmp(o.h_out[r].data(), &big.h_out[r][size_t(t) * CH], size_t(CH) * 4) != 0 ? 1 : 0;
            diffs += std::memcmp(o.scal[r].data(), &big.scal[r][size_t(t) * 16], 64) != 0 ? 1 : 0;
            diffs +=
                std::memcmp(&o.hist[r][pl::hist_version_floats], &big.hist[r][size_t(t + 1) * pl::hist_version_floats],
                            size_t(pl::hist_version_floats) * 4) != 0
                    ? 1
                    : 0;
        }
        if (t + 1 == k && !get_hist0(p, after_small)) {
            return false;
        }
    }
    for (int32_t r = 0; r < 2 && k > 0; ++r) {
        diffs += after_small[r] != after_big[r] ? 1 : 0; // the k < B commit equals k one-row steps
    }
    return true;
}

// ---- the embedding ----------------------------------------------------------------------------------------------

struct embed_result {
    int64_t rows = 0, bad = 0;
};

bool embed_checks(pair_ctx & p, const files & f, bool san, embed_result & er) {
    size_t d = 0;
    const fnfl_gguf_tensor * t = find(f, "token_embd.weight", d);
    if (t == nullptr || t->type_name != "iq3_s" || t->ne[0] != D || t->ne[1] != V) {
        fprintf(stderr, "ple-test: token_embd.weight is not IQ3_S [2560][248320]\n");
        return false;
    }
    rng g(0xe1e1u);
    std::vector<int32_t> ids = {0, 1, V - 1, rt::ple_eos, 248046, 248056, 248045};
    while (ids.size() < 16) {
        ids.push_back(g.below(V));
    }
    // each card's table holds the test tokens' rows (the rest zero)
    std::vector<std::vector<uint8_t>> rows(ids.size(), std::vector<uint8_t>(EMB_FULL_ROW));
    for (size_t k = 0; k < ids.size(); ++k) {
        if (!read_at(f.dirs[d].path, t->offset + uint64_t(ids[k]) * EMB_FULL_ROW, rows[k].data(), EMB_FULL_ROW)) {
            return false;
        }
        for (int32_t r = 0; r < 2; ++r) {
            if (!ok_cuda(cudaSetDevice(p.c[r].dev), "device") ||
                !ok_cuda(cudaMemcpy(p.c[r].table + int64_t(ids[k]) * E_ROW, rows[k].data() + int64_t(r) * E_ROW, E_ROW,
                                    cudaMemcpyHostToDevice),
                         "table row")) {
                return false;
            }
        }
    }
    auto run = [&](int32_t r, const std::vector<int32_t> & tok, std::vector<float> & hres, std::vector<float> & hpost,
                   std::vector<float> & hx) {
        card & c = p.c[r];
        const int32_t B = int32_t(tok.size());
        fn::embed::args a;
        a.table = c.table;
        a.grids = c.grids;
        a.tok = c.tok;
        a.hres = c.ehres;
        a.hpost = c.ehpost;
        a.hx = c.ehx;
        a.err = p.rg->err(r);
        hres.resize(size_t(B) * CH);
        hpost.resize(size_t(B) * S);
        hx.resize(size_t(B) * F);
        return ok_cuda(cudaSetDevice(c.dev), "device") &&
               ok_cuda(cudaMemset(c.ehres, 0xff, size_t(R) * CH * 4), "fill") &&
               ok_cuda(cudaMemset(c.ehpost, 0xff, size_t(R) * 16), "fill") &&
               ok_cuda(cudaMemset(c.ehx, 0xff, size_t(R) * F * 4), "fill") &&
               ok_cuda(cudaMemcpy(c.tok, tok.data(), size_t(B) * 4, cudaMemcpyHostToDevice), "tok") &&
               fn::embed::embed(B, a, c.st) && ok_cuda(cudaStreamSynchronize(c.st), "embed") &&
               ok_cuda(cudaMemcpy(hres.data(), c.ehres, hres.size() * 4, cudaMemcpyDeviceToHost), "hres") &&
               ok_cuda(cudaMemcpy(hpost.data(), c.ehpost, hpost.size() * 4, cudaMemcpyDeviceToHost), "hpost") &&
               ok_cuda(cudaMemcpy(hx.data(), c.ehx, hx.size() * 4, cudaMemcpyDeviceToHost), "hx");
    };
    // the expected row of token k on card r
    auto want = [&](size_t k, int32_t r) {
        std::vector<float> v(F);
        for (int32_t sb = 0; sb < fn::embed::n_super; ++sb) {
            fn::quant::decode_block(fn::quant::wfmt::iq3_s, rows[k].data() + int64_t(r) * E_ROW + int64_t(sb) * 110,
                                    v.data() + int64_t{256} * sb);
        }
        return v;
    };
    const std::vector<int32_t> Bs = san ? std::vector<int32_t>{1, 3} : std::vector<int32_t>{1, 2, 3, 4, 5};
    for (int32_t r = 0; r < 2; ++r) {
        // B = 1 for every token: the row against the CPU decode, and the reference for the batched runs
        std::vector<std::vector<float>> single(ids.size());
        for (size_t k = 0; k < ids.size(); ++k) {
            std::vector<float> hres, hpost, hx;
            if (!run(r, {ids[k]}, hres, hpost, hx)) {
                return false;
            }
            const std::vector<float> wv = want(k, r);
            for (int32_t s = 0; s < S; ++s) {
                er.bad += std::memcmp(&hres[size_t(s) * F], wv.data(), size_t(F) * 4) != 0 ? 1 : 0;
            }
            for (const float z : hpost) {
                er.bad += z != 0.0f || std::signbit(z) ? 1 : 0;
            }
            for (const float z : hx) {
                er.bad += z != 0.0f || std::signbit(z) ? 1 : 0;
            }
            single[k] = hres;
            er.rows++;
        }
        // the same tokens in every slot of B-row launches
        for (const int32_t B : Bs) {
            for (size_t k0 = 0; k0 + size_t(B) <= ids.size(); k0 += 3) {
                std::vector<int32_t> tok(ids.begin() + int64_t(k0), ids.begin() + int64_t(k0) + B);
                std::vector<float> hres, hpost, hx;
                if (!run(r, tok, hres, hpost, hx)) {
                    return false;
                }
                for (int32_t b = 0; b < B; ++b) {
                    er.bad += std::memcmp(&hres[size_t(b) * CH], single[k0 + size_t(b)].data(), size_t(CH) * 4) != 0;
                    er.rows++;
                }
            }
        }
        // an ID outside [0, V) that reached the device: the error bit, and nothing written for its row
        {
            card & c = p.c[r];
            const int32_t tok[2] = {ids[3], V};
            std::vector<float> hres(size_t(2) * CH);
            fn::embed::args a;
            a.table = c.table;
            a.grids = c.grids;
            a.tok = c.tok;
            a.hres = c.ehres;
            a.hpost = c.ehpost;
            a.hx = c.ehx;
            a.err = p.rg->err(r);
            uint32_t err = 0;
            bool ok = ok_cuda(cudaSetDevice(c.dev), "device") &&
                      ok_cuda(cudaMemset(c.ehres, 0xff, size_t(4) * CH * 4), "fill") &&
                      ok_cuda(cudaMemcpy(c.tok, tok, 8, cudaMemcpyHostToDevice), "tok") &&
                      fn::embed::embed(2, a, c.st) && ok_cuda(cudaStreamSynchronize(c.st), "embed") &&
                      ok_cuda(cudaMemcpy(hres.data(), c.ehres, hres.size() * 4, cudaMemcpyDeviceToHost), "hres") &&
                      p.rg->read_err(r, err);
            if (!ok) {
                return false;
            }
            er.bad += err != fn::embed::err_token ? 1 : 0;
            er.bad += std::memcmp(hres.data(), single[3].data(), size_t(CH) * 4) != 0 ? 1 : 0;
            for (int64_t i = CH; i < int64_t{2} * CH; ++i) {
                uint32_t u = 0;
                std::memcpy(&u, &hres[size_t(i)], 4);
                er.bad += u != 0xffffffffu ? 1 : 0;
            }
            er.rows += 2;
        }
        // row counts outside 1..5 enqueue nothing (SP2)
        fn::embed::args a;
        a.table = p.c[r].table;
        a.grids = p.c[r].grids;
        a.tok = p.c[r].tok;
        a.hres = p.c[r].ehres;
        a.hpost = p.c[r].ehpost;
        a.hx = p.c[r].ehx;
        a.err = p.rg->err(r);
        er.bad += fn::embed::embed(0, a, p.c[r].st) ? 1 : 0;
        er.bad += fn::embed::embed(6, a, p.c[r].st) ? 1 : 0;
    }
    return ok_cuda(cudaSetDevice(p.c[0].dev), "device") && ok_cuda(cudaDeviceSynchronize(), "sync") &&
           ok_cuda(cudaSetDevice(p.c[1].dev), "device") && ok_cuda(cudaDeviceSynchronize(), "sync") &&
           p.rg->reset_all();
}

} // namespace

int fnfl_ple_test(const char * mode, int dev0, int dev1, const char * model) {
    const bool san = strcmp(mode, "san") == 0;
    if (!san && strcmp(mode, "all") != 0) {
        fprintf(stderr, "ple-test: mode is all or san\n");
        return 2;
    }
    const bool loopback = dev0 == dev1;
    printf("ple-test: mode %s, devices %d %d%s, model %s\n", mode, dev0, dev1,
           loopback ? " (loopback: one device, one stream, sends before reads)" : "", model);

    // ---- host: the keys, the hash, the gather -------------------------------------------------------------------
    rt::ple_params pp;
    std::string err;
    if (!rt::read_ple_params(model, pp, err)) {
        fprintf(stderr, "ple-test: %s\n", err.c_str());
        return 1;
    }
    printf("ple-test: keys: eos %d, image %d, multipliers [%llu, %llu, %llu], table rows %lld\n", pp.eos, pp.image,
           static_cast<unsigned long long>(pp.mult[0]), static_cast<unsigned long long>(pp.mult[1]),
           static_cast<unsigned long long>(pp.mult[2]), static_cast<long long>(pp.table_rows));
    printf("ple-test: head offsets / vocabularies:");
    for (int32_t h = 0; h < rt::ple_heads; ++h) {
        printf(" %u/%u", pp.offset[h], pp.vocab[h]);
    }
    printf("\n");
    files f;
    if (!open_files(model, f)) {
        return 1;
    }
    size_t td = 0;
    const fnfl_gguf_tensor * tt = find(f, "per_layer_token_embd.weight", td);
    fn::model::mapped_file map;
    if (tt == nullptr || !map.open(f.dirs[td].path, err)) {
        fprintf(stderr, "ple-test: the PLE table: %s\n", err.c_str());
        return 1;
    }
    const uint8_t * table = map.data() + tt->offset;
    host_result hr;
    if (!host_checks(pp, f, table, hr)) {
        return 1;
    }
    printf("ple-test: host: %lld windows, %lld index mismatches; %lld gathered rows, %lld not bitwise; %lld "
           "out-of-range IDs accepted\n",
           static_cast<long long>(hr.windows), static_cast<long long>(hr.index_bad), static_cast<long long>(hr.rows),
           static_cast<long long>(hr.row_bad), static_cast<long long>(hr.reject_bad));
    bool pass = hr.index_bad == 0 && hr.row_bad == 0 && hr.reject_bad == 0;

    // ---- the cards ------------------------------------------------------------------------------------------------
    if (!loopback && !x::enable_peer_access(dev0, dev1)) {
        return 1;
    }
    ple_w w;
    if (!read_ple_w(f, w)) {
        return 1;
    }
    x::rings rg(dev0, dev1);
    pair_ctx p;
    p.rg = &rg;
    p.c[0].dev = dev0;
    p.c[1].dev = dev1;
    bool ok = rg.ok() && alloc_card(p.c[0]) && alloc_card(p.c[1]);
    if (ok && loopback) {
        cudaStreamDestroy(p.c[1].st);
        p.c[1].st = p.c[0].st;
        p.c[1].own_stream = false;
    }
    ok = ok && upload_weights(p.c[0], 0, w) && upload_weights(p.c[1], 1, w);

    embed_result er;
    ok = ok && embed_checks(p, f, san, er);
    if (ok) {
        printf("ple-test: embed: %lld rows, %lld not bitwise or not as specified (bitwise IQ3_S decode, zero hpost/hx, "
               "B-invariant, device ID guard, B outside 1..5 rejected)\n",
               static_cast<long long>(er.rows), static_cast<long long>(er.bad));
        pass = pass && er.bad == 0;
    }

    // FC-G1: row counts outside 1..5, and a commit of more than 5, enqueue nothing (SP2)
    if (ok) {
        const pl::step_args a = step_args_of(p.c[0], rg, 0, 0);
        int64_t accepted = 0;
        for (const int32_t B : {0, 6}) {
            accepted += pl::products(B, a, p.c[0].st) + pl::stats(B, a, p.c[0].st) + pl::update(B, a, p.c[0].st);
        }
        accepted += fn::round::advance(p.c[0].cur, 6, p.c[0].st) ? 1 : 0;
        printf("ple-test: rejection: %lld calls outside the contract accepted\n", static_cast<long long>(accepted));
        pass = pass && accepted == 0;
    }

    // the token sequence: random IDs with EOS, an image run, the ends of the vocabulary
    const int64_t positions = san ? 64 : 512;
    std::vector<int32_t> seq(size_t(positions + 8));
    rng g(0x5e9u);
    for (size_t i = 0; i < seq.size(); ++i) {
        seq[i] = i % 37 == 5 || i % 61 == 6 ? pp.eos : (i >= 100 && i < 104) ? pp.image : g.below(V);
    }
    seq[20] = 0;
    seq[21] = V - 1;
    seq[22] = 248046;
    const std::vector<int32_t> cycle =
        san ? std::vector<int32_t>{1, 3} : std::vector<int32_t>{1, 2, 3, 4, 5, 3, 2, 5, 4, 1}; // SP2: B = 5

    worst wr;
    chain_result c1, c2;
    ok = ok && run_chain(p, pp, table, w, seq, positions, cycle, wr, c1, true);
    if (ok) {
        printf("ple-test: chain: %lld rounds over %lld positions (%lld commits with k < B), %lld row-streams checked\n",
               static_cast<long long>(c1.rounds), static_cast<long long>(c1.positions),
               static_cast<long long>(c1.k_short), static_cast<long long>(wr.rows));
        printf("ple-test: worst: score %.3e (<= 1e-5 x max(1, |s|)), gate at own score %.3e (<= 1e-6), gate vs FP64 "
               "%.3e (%.3f of its bound), update %.3e (<= 1e-5), x %.3e (<= 1e-5); %lld explained rows (|s| <= 2 own "
               "error); %lld failing row-streams; history versions not bitwise %lld; card scalars differing %lld\n",
               wr.score, wr.gate_own, wr.gate_ref, wr.gate_ref_ratio, wr.update, wr.xerr,
               static_cast<long long>(wr.explained), static_cast<long long>(wr.fails),
               static_cast<long long>(wr.hist_bad), static_cast<long long>(wr.scal_diff));
        printf("ple-test: FP64 scores %.4g .. %.4g, kernel gates %.6g .. %.6g, %lld exact-zero scores (zero stream)\n",
               wr.s_min, wr.s_max, wr.g_min, wr.g_max, static_cast<long long>(wr.zero_scores));
        std::sort(wr.tops.begin(), wr.tops.end(),
                  [](const worst::top & a, const worst::top & b) { return a.bound > b.bound; });
        printf("ple-test: the 10 largest gate bounds (FP64 score, own score error, bound, |gate - FP64 gate|):\n");
        for (size_t i = 0; i < std::min<size_t>(10, wr.tops.size()); ++i) {
            printf("  %+.9e  %.3e  %.3e  %.3e\n", wr.tops[i].score, wr.tops[i].err, wr.tops[i].bound, wr.tops[i].diff);
        }
        pass = pass && wr.fails == 0 && wr.hist_bad == 0 && wr.scal_diff == 0;
        printf("ple-test: committed slots visited: %d of %d (SR1)\n", __builtin_popcount(p.slots_seen), pl::n_version);
        pass = pass && p.slots_seen == (1u << pl::n_version) - 1u;
    }
    // determinism: the whole chain again, bitwise
    worst unused;
    ok = ok && run_chain(p, pp, table, w, seq, positions, cycle, unused, c2, false);
    if (ok) {
        printf("ple-test: rerun: digest %016llx vs %016llx: %s\n", static_cast<unsigned long long>(c1.digest),
               static_cast<unsigned long long>(c2.digest), c1.digest == c2.digest ? "bitwise" : "DIFFERENT");
        pass = pass && c1.digest == c2.digest;
    }
    // batch invariance and k < B commits, from a state with history (after the chain)
    int64_t diffs = 0, cases = 0;
    const std::vector<std::pair<int32_t, int32_t>> bk =
        san ? std::vector<std::pair<int32_t, int32_t>>{{3, 3}, {3, 1}}
            : std::vector<std::pair<int32_t, int32_t>>{{3, 3}, {4, 4}, {2, 2}, {3, 2}, {4, 1},
                                                       {4, 3}, {3, 0}, {5, 5}, {5, 2}}; // SP2: B = 5
    for (size_t i = 0; ok && i < bk.size(); ++i) {
        ok = batch_invariance(p, pp, table, seq, int64_t(i) * 5 + 3, bk[i].first, bk[i].second, diffs);
        cases++;
    }
    if (ok) {
        printf("ple-test: batch invariance: %lld cases (B-row rounds against B one-row rounds; k < B commits "
               "against k one-row steps), %lld bitwise differences\n",
               static_cast<long long>(cases), static_cast<long long>(diffs));
        pass = pass && diffs == 0;
    }

    for (int32_t r = 0; r < 2; ++r) {
        cudaSetDevice(p.c[r].dev);
        cudaDeviceSynchronize();
    }
    free_card(p.c[1]);
    free_card(p.c[0]);
    if (!ok) {
        printf("ple-test: FAIL (a run did not complete)\n");
        return 1;
    }
    printf("ple-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
