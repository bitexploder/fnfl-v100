#include "fn/runtime/split_stage_test.h"

#include "fn/kernels/hc/hc_prompt.h"
#include "fn/kernels/round/round.h"
#include "fn/loader/loader.h"
#include "fn/loader/run_load.h"
#include "fn/model/kv_pool.h"
#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/model/pass.h"
#include "fn/model/prompt.h"
#include "fn/model/state.h"
#include "fn/quant/formats.h"
#include "fn/quant/synth.h"
#include "fn/runtime/joint.h"
#include "fn/runtime/prefill.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/stage.h"
#include "fn/runtime/target.h"
#include "fn/tools/cases.h"
#include "fn/tools/test_cuda.h"
#include "fn/vision/profile.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime_api.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fn::runtime {

namespace {

namespace md = fn::model;

constexpr int32_t n_slot     = 4;      // the joint cases' lanes; the lone cases run on slot 0
constexpr int32_t slot_cells = 16384;  // a slot's cells (one KV page)
constexpr int32_t max_b      = fn::round::max_rows;

bool cuda_ok(cudaError_t e, const char * what) {
    return fn::test::cuda_ok(e, "split-test", what);
}

uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// ---- synthetic weights: fn/quant's valid blocks for the quantized formats, small finite values for the float ones;
// generated in 1 MiB chunks, each seeded by (tensor, chunk), so the bytes do not depend on the thread count ----

uint16_t to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

uint16_t to_f16(float f) {
    // |f| < 2 (synthetic values only): normal, subnormal (truncated) or zero
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int32_t  e = static_cast<int32_t>((u >> 23) & 0xff) - 127 + 15;
    if (e <= 0) {
        const float a = std::fabs(f) * 16777216.0f;  // in units of 2^-24, the F16 subnormal step
        return static_cast<uint16_t>(s | std::min<uint32_t>(0x3ffu, static_cast<uint32_t>(a)));
    }
    return static_cast<uint16_t>(s | (uint32_t(e) << 10) | ((u >> 13) & 0x3ffu));
}

float uniform(uint64_t seed, uint64_t i) {
    const uint64_t h = mix(seed ^ mix(i));
    return float(h >> 40) / float(1u << 24) * 2.0f - 1.0f;
}

// a float tensor's value i: norms and gains near 1, every other float small
float float_value(const std::string & name, uint64_t seed, uint64_t i) {
    const float u = uniform(seed, i);
    if (name.find("norm") != std::string::npos) {
        return 1.0f + 0.1f * u;
    }
    if (name.find("ssm_a") != std::string::npos) {
        return -0.5f - 0.5f * std::fabs(u);
    }
    return 0.05f * u;
}

bool synth_tensor(const md::model & m, int32_t t, uint8_t * dst, int64_t bytes) {
    const md::tensor_place & tp  = m.tp(t);
    std::string              fmt = tp.format;
    for (char & ch : fmt) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    fn::quant::wfmt f{};
    if (!fn::quant::wfmt_of_name(fmt.c_str(), f)) {
        fprintf(stderr, "split-test: %s: format %s has no generator\n", tp.name, tp.format);
        return false;
    }
    // TTTS2 (the Q4 inventory): an F32 file tensor placed BF16 is stored F32, each value exactly a BF16
    const bool bf16_exact = m.sources[size_t(t)].conv == md::src_conv::f32_bf16;
    if (bf16_exact) {
        f = fn::quant::wfmt::f32;
    }
    const std::string name     = tp.name;
    const uint64_t    seed     = mix(0x5454535331ull + uint64_t(t) * 0x10001ull);
    const bool        is_float = f == fn::quant::wfmt::bf16 || f == fn::quant::wfmt::f16 || f == fn::quant::wfmt::f32;
    const int64_t     esize    = f == fn::quant::wfmt::f32 ? 4 : 2;
    const int64_t     unit     = is_float ? esize : tp.block_bytes;
    if (bytes % unit != 0) {
        fprintf(stderr, "split-test: %s: %lld bytes is not whole blocks\n", tp.name, (long long) bytes);
        return false;
    }
    const int64_t        n_units   = bytes / unit;
    const int64_t        per_chunk = std::max<int64_t>(1, (int64_t(1) << 20) / unit);
    const int64_t        n_chunks  = (n_units + per_chunk - 1) / per_chunk;
    std::atomic<int64_t> next{ 0 };
    std::atomic<bool>    bad{ false };
    const auto           work = [&]() {
        for (int64_t ch; (ch = next.fetch_add(1)) < n_chunks;) {
            const int64_t u0 = ch * per_chunk, u1 = std::min(n_units, u0 + per_chunk);
            uint8_t *     p = dst + u0 * unit;
            if (!is_float) {
                // the deployed scale magnitudes where fn/quant has them (the bulk prompt path's FP16 operands stay in
                // range); Q6_K and IQ4_XS: C1b's blocks with each d redrawn log-uniform in [1e-5, 2.5e-3]
                const uint64_t sd = mix(seed + uint64_t(ch));
                if (fn::quant::synth_blocks_deployed(f, sd, u1 - u0, p)) {
                    continue;
                }
                if (!fn::quant::synth_blocks(f, sd, u1 - u0, p)) {
                    bad = true;
                    continue;
                }
                const int64_t d_at = f == fn::quant::wfmt::q6_k ? 208 : f == fn::quant::wfmt::iq4_xs ? 0 : -1;
                for (int64_t i = u0; d_at >= 0 && i < u1; ++i) {
                    const double   x = (uniform(sd, uint64_t(i)) + 1.0) * 0.5;  // [0, 1)
                    const float    d = static_cast<float>(1e-5 * std::pow(250.0, x));
                    const uint16_t h = to_f16(d);
                    std::memcpy(dst + i * unit + d_at, &h, 2);
                }
                continue;
            }
            for (int64_t i = u0; i < u1; ++i) {
                const float v = float_value(name, seed, uint64_t(i));
                if (f == fn::quant::wfmt::f32) {
                    uint32_t w = 0;
                    std::memcpy(&w, &v, 4);
                    w = bf16_exact ? uint32_t(to_bf16(v)) << 16 : w;
                    std::memcpy(dst + i * 4, &w, 4);
                } else {
                    const uint16_t h = f == fn::quant::wfmt::bf16 ? to_bf16(v) : to_f16(v);
                    std::memcpy(dst + i * 2, &h, 2);
                }
            }
        }
    };
    const int32_t            nt = n_chunks > 4 ? 10 : 1;
    std::vector<std::thread> th;
    for (int32_t i = 1; i < nt; ++i) {
        th.emplace_back(work);
    }
    work();
    for (std::thread & x : th) {
        x.join();
    }
    return !bad;
}

struct options {
    std::string          model_path  = fn::cases::fixture_path("model");  // TFC: the named fixtures
    std::string          draft_path  = fn::cases::fixture_path("draft");
    bool        san         = false;  // the small shape: layers 0 .. 5 (k = 3), eager, the lone case only
    std::vector<int32_t> ks = { 3, 27, 45 };
    bool                 modes[2] = { true, true };  // graph, eager
    std::string          cases    = "lone,joint,prompt";
    int32_t              lo = 0, hi = fn::n_layer - 1;
    bool        perturb = false;  // TEST of the test: one bit of one crossing flipped, the comparison must fail
    // the pipe case's processes: 1 the leader (this box the front), 2 the follower (the back), over the link
    int32_t     role    = 0;
    std::string addr    = "10.25.0.2";
    int32_t     port    = 13372;
};

// a loaded synthetic model with its zero PLE mapping
struct loaded {
    std::unique_ptr<md::model> m;
    void *                     ple_zero = nullptr;
    size_t                     ple_len  = 0;

    ~loaded() {
        if (m != nullptr) {
            for (int32_t c = 0; c < 2; ++c) {
                cudaSetDevice(m->device[c]);
                cudaDeviceSynchronize();
            }
            m.reset();
        }
        if (ple_zero != nullptr) {
            munmap(ple_zero, ple_len);
        }
    }
};

bool load_synth(const options &           o,
                const md::window &        w,
                loaded &                  out,
                int32_t                   n_slots = n_slot,
                const md::vision_params * vision  = nullptr) {
    out.m                     = std::make_unique<md::model>();
    md::model &             m = *out.m;
    fn::loader::load_params lp;
    lp.model_path = o.model_path;
    lp.draft_path = o.draft_path;
    lp.win        = w;
    lp.n_slots    = n_slots;
    lp.kv_q8      = true;
    fn::loader::load_times lt;
    std::string            err;
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(lp.device[c]), "device") || !cuda_ok(cudaFree(nullptr), "context")) {
            return false;
        }
    }
    if (!fn::loader::open_model(lp, m, lt, err) || !fn::loader::allocate_weights(m, err) ||
        !fn::loader::fill_weights_synth(
            m, [&m](int32_t t, uint8_t * dst, int64_t n) { return synth_tensor(m, t, dst, n); }, err)) {
        fprintf(stderr, "split-test: the synthetic load: %s\n", err.c_str());
        return false;
    }
    // no PLE data: a zero mapping of the table's size in place of the file's (gathers read zero blocks)
    out.ple_len  = size_t(m.ple_bytes);
    out.ple_zero = mmap(nullptr, out.ple_len, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (out.ple_zero == MAP_FAILED) {
        out.ple_zero = nullptr;
        fprintf(stderr, "split-test: the zero PLE mapping\n");
        return false;
    }
    m.ple_table = static_cast<const uint8_t *>(out.ple_zero);
    md::state_params sp;
    sp.n_cells         = slot_cells;
    sp.ple_conv_kernel = m.ple_conv_kernel;
    sp.ple_ngram       = m.ple_ngram;
    sp.kv_q8           = true;
    sp.win             = w;
    fn::loader::file_state_params(m, sp);  // the Q4 file's head and PLE state, as the run load's
    int64_t host_bytes = 0;
    if (!fn::loader::reserve_state(m, sp, n_slots, host_bytes, err, n_slots * slot_cells, vision) ||
        !fn::loader::fill_ex1_grids(m, err) || !fn::xchg::enable_peer_access(m.device[0], m.device[1])) {
        fprintf(stderr, "split-test: the reservation: %s\n", err.c_str());
        return false;
    }
    m.rings = std::make_unique<fn::xchg::rings>(m.device[0], m.device[1]);
    for (int32_t s = 0; s < n_slots && m.pool != nullptr; ++s) {
        if (!m.pool->ensure(s, slot_cells, err)) {
            fprintf(stderr, "split-test: the pool's pages: %s\n", err.c_str());
            return false;
        }
    }
    return m.rings->ok() && cuda_ok(cudaDeviceSynchronize(), "synthetic load");
}

// the split's prompt crossing in one process: the front queues each chunk's crossing, the back takes them in order
struct local_crossing : pf_stage {
    int32_t                           k = 0;
    std::vector<std::vector<uint8_t>> q;
    std::vector<int64_t>              seq, start, rows;
    std::vector<md::prompt_chunk>     part;  // MUP2: the front's partition, which the back runs
    size_t                            next = 0;

    bool sync(target & t, std::string & err) {
        for (int32_t c = 0; c < 2; ++c) {
            if (cudaSetDevice(t.card(c).c.device) != cudaSuccess ||
                cudaStreamSynchronize(static_cast<cudaStream_t>(t.stream(c))) != cudaSuccess) {
                err = "crossing: stream sync";
                return false;
            }
        }
        return true;
    }

    bool out(target & t, md::prompt_path & pp, const md::prompt_chunk & c, std::string & err) override {
        std::vector<uint8_t> b(size_t(md::prompt_path::crossing_bytes(c)));
        if (!pp.crossing_out(c, k, b.data(), err) || !sync(t, err)) {
            return false;
        }
        q.push_back(std::move(b));
        seq.push_back(c.seq);
        start.push_back(c.start);
        rows.push_back(c.rows);
        md::prompt_chunk pc;
        pc.n_pieces = c.n_pieces;
        std::copy(c.piece, c.piece + md::prompt_max_pieces, pc.piece);
        part.push_back(pc);
        return true;
    }

    bool in(target & t, md::prompt_path & pp, md::prompt_chunk & c, std::string & err) override {
        if (next >= q.size() || start[next] != c.start || rows[next] != c.rows) {
            err = "crossing: the back's chunk at " + std::to_string(c.start) + " (" + std::to_string(c.rows) +
                  " rows) is not the front's next";
            return false;
        }
        c.seq      = seq[next];
        c.n_pieces = part[next].n_pieces;
        std::copy(part[next].piece, part[next].piece + md::prompt_max_pieces, c.piece);
        const bool ok = pp.crossing_in(c, k, q[next].data(), err) && sync(t, err);
        ++next;
        return ok;
    }
};

// the targets of one model (a slot each), on shared streams as the engine's, and its joint frame
struct stage_set {
    md::model *                  m       = nullptr;
    void *                       st[2]   = { nullptr, nullptr };
    uint64_t                     ple_use = 0;
    md::use_counters             u[2];
    std::unique_ptr<target>      t[n_slot];
    std::unique_ptr<joint_frame> joint;
    std::unique_ptr<prefiller>   pf;
    local_crossing               cross;
    uint8_t * buf[n_slot][2] = {};  // a stage's boundary buffers (send on the front, receive on the back)

    ~stage_set() {
        pf.reset();
        for (int32_t s = 0; s < n_slot; ++s) {
            t[s].reset();
        }
        joint.reset();
        for (int32_t c = 0; c < 2 && m != nullptr; ++c) {
            if (st[c] != nullptr) {
                cudaSetDevice(m->device[c]);
                cudaStreamSynchronize(static_cast<cudaStream_t>(st[c]));
                cudaStreamDestroy(static_cast<cudaStream_t>(st[c]));
            }
        }
        for (int32_t s = 0; s < n_slot; ++s) {
            for (int32_t c = 0; c < 2; ++c) {
                if (buf[s][c] != nullptr) {
                    cudaFreeHost(buf[s][c]);
                }
            }
        }
    }
};

// role: 0 the whole model, 1 the front, 2 the back
bool make_set(const options & o, md::model & m, int32_t role, bool eager, stage_set & ss) {
    ss.m = &m;
    std::string err;
    for (int32_t c = 0; c < 2; ++c) {
        cudaStream_t cs = nullptr;
        if (!cuda_ok(cudaSetDevice(m.device[c]), "device") ||
            !cuda_ok(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream")) {
            return false;
        }
        ss.st[c] = cs;
    }
    for (int32_t s = 0; s < n_slot; ++s) {
        target_options to;
        to.model_path    = o.model_path;
        to.eager         = eager;
        to.sample        = true;
        to.sampling      = fn::samp::options();
        to.sampling.temp = 0.0f;  // the greedy path: decisions every run repeats
        to.slot          = s;
        to.streams[0]    = ss.st[0];
        to.streams[1]    = ss.st[1];
        to.ple_use       = &ss.ple_use;
        to.eager_uses    = ss.u;
        to.pass.layer_lo = m.win.lo;
        to.pass.layer_hi = m.win.hi;
        if (role != 0) {
            for (int32_t c = 0; c < 2; ++c) {
                if (!cuda_ok(cudaHostAlloc(reinterpret_cast<void **>(&ss.buf[s][c]), size_t(md::stage_bytes(max_b)),
                                           cudaHostAllocPortable),
                             "boundary buffer")) {
                    return false;
                }
                std::memset(ss.buf[s][c], 0, size_t(md::stage_bytes(max_b)));
                if (role == 1) {
                    to.pass.stage.out     = true;
                    to.pass.stage.send[c] = ss.buf[s][c];
                } else {
                    to.pass.stage.in      = true;
                    to.pass.stage.recv[c] = ss.buf[s][c];
                }
            }
        }
        ss.t[s] = std::make_unique<target>();
        if (!ss.t[s]->init(m, to, err)) {
            fprintf(stderr, "split-test: slot %d's target: %s\n", s, err.c_str());
            return false;
        }
    }
    ss.joint = std::make_unique<joint_frame>();
    if (!ss.joint->init(m, ss.st, 8, eager, false, true, true, err)) {
        fprintf(stderr, "split-test: the joint frame: %s\n", err.c_str());
        return false;
    }
    ss.pf = std::make_unique<prefiller>();
    if (!ss.pf->init(*ss.t[0], m, err) || !ss.pf->init_lane(*ss.t[0], err)) {
        fprintf(stderr, "split-test: the prefiller: %s\n", err.c_str());
        return false;
    }
    ss.cross.k = role == 0 ? 0 : role == 1 ? m.win.hi + 1 : m.win.lo;
    if (role != 0) {
        ss.pf->set_stage(&ss.cross);
    }
    return true;
}

// ---- snapshots: the tokens and each layer's state, hashed ----

struct snapshot {
    std::string                     what;
    std::vector<int32_t>            ints;   // tokens, decided rows, committed words
    std::map<std::string, uint64_t> state;  // per part and card
};

uint64_t hash_bytes(const uint8_t * p, size_t n) {
    uint64_t h = 0x84222325cbf29ce4ull ^ n;
    size_t   i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w = 0;
        std::memcpy(&w, p + i, 8);
        h = mix(h ^ w);
    }
    for (; i < n; ++i) {
        h = mix(h ^ p[i]);
    }
    return h;
}

struct reader {
    std::vector<uint8_t> host;

    bool read(int32_t device, const uint8_t * dev, size_t n, uint64_t & h) {
        host.resize(n);
        if (n == 0) {
            h = hash_bytes(nullptr, 0);
            return true;
        }
        if (!cuda_ok(cudaSetDevice(device), "device") ||
            !cuda_ok(cudaMemcpy(host.data(), dev, n, cudaMemcpyDefault), "state read")) {
            return false;
        }
        h = hash_bytes(host.data(), n);
        return true;
    }
};

// slot s's window state of model m into snap (every part of the window's layers, whole; the q8 scale columns of its
// QSA layers; the PLE history; with the head, the logits and the hidden carry). MUP2 (cur >= 0, the slot's committed
// version slot): the logical state: the versioned histories (GDN convolution, PLE) at their committed slot only (a
// chunk of an odd number of pieces commits version 1, the halves 2, so the same history sits in another physical slot
// and the others hold the pieces' intermediate versions), and no hidden carry (its rows past the read's are the slot's
// earlier steps'; the steps after the read check the rows it wrote), nor the logits (written by rounds, not reads).
// PRD1: with gsel >= 0 (the slot's GDN selector, fn::round::gdn_word) the GDN matrices are logical too: the committed
// bank and the first k records of its set only (a prompt chunk's commit flips the bank and a restore normalizes it to
// bank 0, so a read resumed from a checkpoint holds the same matrix in the other bank)
bool state_of(md::model & m, int32_t s, snapshot & snap, reader & rd, int32_t cur = -1, int64_t gsel = -1) {
    const md::slot_scope bind(m, s);
    const std::string    pre       = "s" + std::to_string(s) + " ";
    const auto           part_hash = [&](const char * cls, const std::string & part, const std::string & key) -> bool {
        const md::reserved_class * rc = m.state_class(cls);
        if (rc == nullptr) {
            return false;
        }
        for (int32_t p = 0; p < (int32_t) rc->plan.parts.size(); ++p) {
            if (rc->plan.parts[size_t(p)].name != part) {
                continue;
            }
            const bool versioned = cur >= 0 && std::string(cls) == "convolution histories";
            const bool banks     = gsel >= 0 && std::string(cls) == "GDN state";
            for (int32_t c = 0; c < 2; ++c) {
                const int64_t n = rc->live_bytes(p, c);
                uint64_t      h = 0;
                if (n > 0 && !rd.read(m.device[c], rc->part_ptr(p, c), size_t(n), h)) {
                    return false;
                }
                if (n > 0 && banks) {
                    const uint32_t g    = uint32_t(gsel);
                    const bool     recs = part.size() > 8 && part.compare(part.size() - 8, 8, " records") == 0;
                    const int64_t  bank = int64_t(md::gdn_heads_card) * md::gdn_head_dim * md::gdn_head_dim * 4;
                    const int64_t  off =
                        recs ? int64_t(fn::round::gdn_set(g)) * fn::round::gdn_max_steps * md::gdn_record_step_bytes :
                               int64_t(fn::round::gdn_bank(g)) * bank;
                    const int64_t len = recs ? int64_t(fn::round::gdn_steps(g)) * md::gdn_record_step_bytes : bank;
                    if (off + len > n) {
                        return false;
                    }
                    h = hash_bytes(rd.host.data() + off, size_t(len));
                }
                if (n > 0 && versioned) {
                    if (n % fn::round::n_version != 0) {
                        return false;
                    }
                    const int64_t vb = n / fn::round::n_version;
                    h = hash_bytes(rd.host.data() + int64_t(fn::round::version_slot(uint32_t(cur), 0)) * vb,
                                   size_t(vb));
                }
                if (n > 0) {
                    snap.state[pre + key + " c" + std::to_string(c)] = h;
                }
            }
            return true;
        }
        return false;
    };
    for (int32_t l = 0; l < fn::n_layer; ++l) {
        if (!m.win.layer(l)) {
            continue;
        }
        const std::string L   = "L" + std::to_string(l);
        const bool        gdn = l % fn::full_attn_interval != fn::full_attn_interval - 1;
        if (gdn) {
            if (!part_hash("GDN state", L, L + " gdn") || !part_hash("GDN state", L + " records", L + " records") ||
                !part_hash("convolution histories", L + " GDN", L + " conv")) {
                fprintf(stderr, "split-test: %s's GDN parts\n", L.c_str());
                return false;
            }
            continue;
        }
        if (!part_hash("target KV (K0)", L + " K", L + " K") || !part_hash("target KV (K0)", L + " V", L + " V")) {
            fprintf(stderr, "split-test: %s's KV parts\n", L.c_str());
            return false;
        }
        part_hash("target index keys and summaries", L + " keys", L + " keys");
        part_hash("target index keys and summaries", L + " summaries", L + " summaries");
        // the q8 scale rows hold every QSA layer's 4 groups a cell: this layer's columns only
        const md::reserved_class * kv = m.state_class("target KV (K0)");
        for (const char * sc : { "K scales", "V scales" }) {
            for (int32_t p = 0; kv != nullptr && p < (int32_t) kv->plan.parts.size(); ++p) {
                if (kv->plan.parts[size_t(p)].name != sc) {
                    continue;
                }
                for (int32_t c = 0; c < 2; ++c) {
                    const int64_t n = kv->live_bytes(p, c);
                    uint64_t      h = 0;
                    if (n > 0 && !rd.read(m.device[c], kv->part_ptr(p, c), size_t(n), h)) {
                        return false;
                    }
                    const int64_t        cells = n / (md::kv_q8_scale_stride * 2);
                    const int32_t        col   = 4 * (l / fn::full_attn_interval);
                    std::vector<uint8_t> cut(size_t(cells) * 8);
                    for (int64_t q = 0; q < cells; ++q) {
                        std::memcpy(&cut[size_t(q) * 8], &rd.host[size_t(q * md::kv_q8_scale_stride * 2 + col * 2)], 8);
                    }
                    snap.state[pre + L + " " + sc + " c" + std::to_string(c)] = hash_bytes(cut.data(), cut.size());
                }
            }
        }
    }
    if (m.win.ple() && !part_hash("convolution histories", "L1 PLE", "L1 PLE")) {
        fprintf(stderr, "split-test: the PLE history\n");
        return false;
    }
    if (m.win.head) {
        if (cur < 0 && (!part_hash("round and pass (TG1)", "logits (B <= 5)", "logits") ||
                        !part_hash("round and pass (TG1)", "hidden carry (FC-HEAD3, B <= 5)", "carry"))) {
            fprintf(stderr, "split-test: the logits and carry\n");
            return false;
        }
    }
    return true;
}

// ---- the drivers: one box, or the split's front and back ----

struct driver {
    stage_set *           whole = nullptr;  // one box
    stage_set *           front = nullptr;  // the split
    stage_set *           back  = nullptr;
    std::vector<snapshot> snaps;
    reader                rd;
    std::string           err;
    bool                  perturb   = false;
    int64_t               crossings = 0;

    bool split() const { return front != nullptr; }

    target & lead(int32_t s) { return split() ? *back->t[s] : *whole->t[s]; }  // the deciding stage's target

    // the boundary of slot s's last front round to the back (the link's place)
    void carry(int32_t s, int32_t rows) {
        for (int32_t c = 0; c < 2; ++c) {
            std::memcpy(back->buf[s][c], front->buf[s][c], size_t(md::stage_bytes(rows)));
        }
        if (perturb && ++crossings == 5) {
            back->buf[s][1][md::stage_hx_at(rows) + 3] ^= 0x01;  // one mantissa bit of card 1's hx, once
        }
    }

    bool snap(const std::string & what, const int32_t * slots, int32_t n, const round_result * r) {
        snapshot x;
        x.what = what;
        for (int32_t i = 0; i < n; ++i) {
            const int32_t  s = slots[i];
            const ledger & l = lead(s).slot();
            x.ints.push_back(-1000 - s);
            x.ints.insert(x.ints.end(), l.tok.begin(), l.tok.end());
            x.ints.push_back(static_cast<int32_t>(l.n));
            x.ints.push_back(l.pending);
            x.ints.push_back(static_cast<int32_t>(l.cur));
            x.ints.push_back(static_cast<int32_t>(l.gdn));
            if (split()) {
                // the front's ledger must be the back's
                const ledger & f = front->t[s]->slot();
                if (f.tok != l.tok || f.n != l.n || f.u != l.u || f.pending != l.pending || f.cur != l.cur ||
                    f.gdn != l.gdn || f.o != l.o) {
                    err = what + ": slot " + std::to_string(s) + ": the front's ledger differs from the back's";
                    return false;
                }
            }
            if (r != nullptr) {
                x.ints.push_back(r[i].n_out);
                for (int32_t t = 0; t < max_b; ++t) {
                    x.ints.push_back(r[i].out[t]);
                    x.ints.push_back(r[i].argmax[0][t]);
                    x.ints.push_back(r[i].argmax[1][t]);
                }
            }
            if (split()) {
                if (!state_of(*front->m, s, x, rd) || !state_of(*back->m, s, x, rd)) {
                    err = what + ": a state read";
                    return false;
                }
            } else if (!state_of(*whole->m, s, x, rd)) {
                err = what + ": a state read";
                return false;
            }
        }
        snaps.push_back(std::move(x));
        return true;
    }

    bool reset(int32_t s) {
        return split() ? front->t[s]->reset(err) && back->t[s]->reset(err) : whole->t[s]->reset(err);
    }

    void prompt(int32_t s, const std::vector<int32_t> & tok) {
        if (split()) {
            front->t[s]->set_prompt(tok);
            back->t[s]->set_prompt(tok);
        } else {
            whole->t[s]->set_prompt(tok);
        }
    }

    bool forced(int32_t s, int32_t rows, round_result & r) {
        if (!split()) {
            return whole->t[s]->forced(rows, r, err);
        }
        target::front_round fr;
        round_result        rf;
        if (!front->t[s]->front_begin(static_cast<int32_t>(fn::round::kind::forced), rows, nullptr, rows, fr, err) ||
            !front->t[s]->front_wait(fr, rf, err)) {
            return false;
        }
        carry(s, rows);
        return back->t[s]->forced(rows, r, err) && front->t[s]->front_end(fr, rows, nullptr, 0, -1, 0, err);
    }

    bool bootstrap(int32_t s, round_result & r) {
        if (!split()) {
            return whole->t[s]->bootstrap(r, err);
        }
        return back->t[s]->bootstrap(r, err) && front->t[s]->front_bootstrap(r.out[0], err);
    }

    bool generate(int32_t s, round_result & r) {
        if (!split()) {
            return whole->t[s]->generate(r, err);
        }
        target::front_round fr;
        round_result        rf;
        if (!front->t[s]->front_begin(static_cast<int32_t>(fn::round::kind::generate), 1, nullptr, 1, fr, err) ||
            !front->t[s]->front_wait(fr, rf, err)) {
            return false;
        }
        carry(s, 1);
        return back->t[s]->generate(r, err) && front->t[s]->front_end(fr, 1, r.out, 1, -1, 0, err);
    }

    bool verify(int32_t s, int32_t d, const int32_t * prop, round_result & r) {
        const int32_t budget = 1000;
        if (!split()) {
            return whole->t[s]->verify(d, prop, budget, r, err);
        }
        target::front_round fr;
        round_result        rf;
        if (!front->t[s]->front_begin(static_cast<int32_t>(fn::round::kind::verify), d + 1, prop, budget, fr, err) ||
            !front->t[s]->front_wait(fr, rf, err)) {
            return false;
        }
        carry(s, d + 1);
        return back->t[s]->verify(d, prop, budget, r, err) &&
               front->t[s]->front_end(fr, r.n_out, r.out, r.n_out, -1, 0, err);
    }

    // PF1's bulk prefill (run_bulk) of all keeping n_reuse tokens: a restore or a reset, then the chunks; the split's
    // front runs first, queueing its crossings, then the back takes them
    bool prefill(int32_t s, const std::vector<int32_t> & all, int64_t n_reuse, int64_t & from) {
        if (!split()) {
            return whole->pf->begin(*whole->t[s], all, n_reuse, from, err) &&
                   whole->pf->run(*whole->t[s], nullptr, err);
        }
        int64_t from_b = -1;
        front->cross.q.clear();
        front->cross.seq.clear();
        front->cross.start.clear();
        front->cross.rows.clear();
        front->cross.part.clear();
        if (!front->pf->begin(*front->t[s], all, n_reuse, from, err) || !front->pf->run(*front->t[s], nullptr, err)) {
            err = "front prefill: " + err;
            return false;
        }
        back->cross.q     = std::move(front->cross.q);
        back->cross.seq   = front->cross.seq;
        back->cross.start = front->cross.start;
        back->cross.rows  = front->cross.rows;
        back->cross.part  = front->cross.part;
        back->cross.next  = 0;
        if (!back->pf->begin(*back->t[s], all, n_reuse, from_b, err) || !back->pf->run(*back->t[s], nullptr, err)) {
            err = "back prefill: " + err;
            return false;
        }
        if (from_b != from || back->cross.next != back->cross.q.size()) {
            err = "prefill: the boxes' restore points (" + std::to_string(from) + ", " + std::to_string(from_b) +
                  ") or chunks differ";
            return false;
        }
        return true;
    }

    bool cut(int32_t s, int32_t j, int64_t n_keep) {
        return split() ? front->t[s]->cut(j, n_keep, err) && back->t[s]->cut(j, n_keep, err) :
                         whole->t[s]->cut(j, n_keep, err);
    }

    bool joint_verify(int32_t n, int32_t d, const int32_t (*prop)[4], round_result * r) {
        const int32_t   budget[n_slot] = { 1000, 1000, 1000, 1000 };
        const int32_t   dd[n_slot]     = { d, d, d, d };
        const int32_t * pp[n_slot]     = { prop[0], prop[1], prop[2], prop[3] };
        bool            ok[n_slot]     = {};
        std::string     why[n_slot];
        bool            staged = false;
        if (!split()) {
            target * tt[n_slot] = { whole->t[0].get(), whole->t[1].get(), whole->t[2].get(), whole->t[3].get() };
            if (!whole->joint->verify(tt, n, dd, pp, budget, r, ok, why, staged, err)) {
                return false;
            }
        } else {
            target * tf[n_slot] = { front->t[0].get(), front->t[1].get(), front->t[2].get(), front->t[3].get() };
            target * tb[n_slot] = { back->t[0].get(), back->t[1].get(), back->t[2].get(), back->t[3].get() };
            target::front_round fr[n_slot];
            round_result        rf[n_slot];
            if (!front->joint->front_verify(tf, n, dd, pp, budget, fr, rf, ok, why, staged, err)) {
                return false;
            }
            for (int32_t k = 0; k < n; ++k) {
                if (!ok[k]) {
                    err = "joint front member " + std::to_string(k) + ": " + why[k];
                    return false;
                }
                carry(k, d + 1);
            }
            if (!back->joint->verify(tb, n, dd, pp, budget, r, ok, why, staged, err)) {
                return false;
            }
            for (int32_t k = 0; k < n && ok[k]; ++k) {
                if (!front->t[k]->front_end(fr[k], r[k].n_out, r[k].out, r[k].n_out, -1, 0, err)) {
                    return false;
                }
            }
        }
        for (int32_t k = 0; k < n; ++k) {
            if (!ok[k]) {
                err = "joint member " + std::to_string(k) + ": " + why[k];
                return false;
            }
        }
        return true;
    }
};

std::vector<int32_t> tokens_of(uint64_t seed, int32_t n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        v[size_t(i)] = static_cast<int32_t>(mix(seed + uint64_t(i)) % 200000u) + 10;
    }
    return v;
}

// a slot from reset to decode-ready: its prompt by forced rounds of B cycling 1 .. 5, the bootstrap, one generate
bool ready_slot(driver & dv, int32_t s, uint64_t seed, int32_t n, bool snaps, const char * name) {
    round_result r;
    if (!dv.reset(s)) {
        return false;
    }
    dv.prompt(s, tokens_of(seed, n));
    int32_t b = 1;
    while (dv.lead(s).slot().u > 0) {
        const int32_t rows = static_cast<int32_t>(std::min<int64_t>(b, dv.lead(s).slot().u));
        if (!dv.forced(s, rows, r) ||
            (snaps && !dv.snap(std::string(name) + " forced B" + std::to_string(rows), &s, 1, &r))) {
            return false;
        }
        b = b % max_b + 1;
    }
    return dv.bootstrap(s, r) && (!snaps || dv.snap(std::string(name) + " bootstrap", &s, 1, &r)) &&
           dv.generate(s, r) && (!snaps || dv.snap(std::string(name) + " generate", &s, 1, &r));
}

bool lone_case(driver & dv) {
    const int32_t s = 0;
    round_result  r;
    if (!ready_slot(dv, s, 101, 23, true, "lone")) {
        return false;
    }
    if (!dv.generate(s, r) || !dv.snap("lone generate 2", &s, 1, &r)) {
        return false;
    }
    for (int32_t d = 1; d <= 4; ++d) {
        // a verify of arbitrary proposals (mostly rejected), then the same position again with the first verify's
        // decided tokens (accepted at least once), then a cut within that step
        const int64_t n0 = dv.lead(s).slot().n;
        int32_t       prop[4];
        for (int32_t j = 0; j < 4; ++j) {
            prop[j] = static_cast<int32_t>(mix(uint64_t(d) * 77 + uint64_t(j)) % 200000u) + 10;
        }
        if (!dv.verify(s, d, prop, r) || !dv.snap("lone verify d" + std::to_string(d), &s, 1, &r)) {
            return false;
        }
        int32_t probe[4];
        for (int32_t j = 0; j < 4; ++j) {
            probe[j] = r.argmax[0][j];
        }
        if (!dv.cut(s, 0, n0 + 1) || !dv.snap("lone cut 0 d" + std::to_string(d), &s, 1, nullptr)) {
            return false;
        }
        if (!dv.verify(s, d, probe, r) || !dv.snap("lone verify accepted d" + std::to_string(d), &s, 1, &r)) {
            return false;
        }
        const int32_t kept = r.n_out;
        if (kept >= 2 && (!dv.cut(s, 1, n0 + 2) || !dv.snap("lone cut 1 d" + std::to_string(d), &s, 1, nullptr))) {
            return false;
        }
    }
    // a reset and a second prompt
    return ready_slot(dv, s, 202, 17, true, "lone second prompt");
}

bool joint_case(driver & dv) {
    for (int32_t s = 0; s < n_slot; ++s) {
        if (!ready_slot(dv, s, 300 + uint64_t(s) * 31, 13 + 3 * s, false, "joint")) {
            return false;
        }
    }
    const int32_t slots[n_slot] = { 0, 1, 2, 3 };
    if (!dv.snap("joint ready", slots, n_slot, nullptr)) {
        return false;
    }
    for (int32_t u = 2; u <= n_slot; ++u) {
        for (int32_t d = 1; d <= 4; ++d) {
            int32_t prop[n_slot][4];
            for (int32_t k = 0; k < n_slot; ++k) {
                for (int32_t j = 0; j < 4; ++j) {
                    prop[k][j] =
                        static_cast<int32_t>(mix(uint64_t(u * 1000 + d * 10 + k) + uint64_t(j)) % 200000u) + 10;
                }
            }
            round_result r[n_slot];
            if (!dv.joint_verify(u, d, prop, r) ||
                !dv.snap("joint U" + std::to_string(u) + " d" + std::to_string(d), slots, u, r)) {
                return false;
            }
        }
    }
    return true;
}

// PF1's bulk path: a prompt across the 8,192 boundary (two chunks, the checkpoint at 8,192), decode rounds, then a
// prompt that keeps 8,292 of its tokens: the checkpoint at 8,192 restored and the rest re-prefilled
bool prompt_case(driver & dv) {
    const int32_t        s = 0;
    round_result         r;
    std::vector<int32_t> all  = tokens_of(707, 9000);
    int64_t              from = -1;
    if (!dv.prefill(s, all, 0, from) || !dv.snap("prompt 9000", &s, 1, nullptr) || !dv.bootstrap(s, r) ||
        !dv.snap("prompt bootstrap", &s, 1, &r) || !dv.generate(s, r) || !dv.snap("prompt generate", &s, 1, &r)) {
        return false;
    }
    int32_t prop[4] = { 11, 12, 13, 14 };
    if (!dv.verify(s, 4, prop, r) || !dv.snap("prompt verify d4", &s, 1, &r)) {
        return false;
    }
    std::vector<int32_t>       more(all.begin(), all.begin() + 8292);
    const std::vector<int32_t> tail = tokens_of(808, 500);
    more.insert(more.end(), tail.begin(), tail.end());
    if (!dv.prefill(s, more, 8292, from) || from != 8192 || !dv.snap("prompt restored at 8192", &s, 1, nullptr) ||
        !dv.bootstrap(s, r) || !dv.snap("prompt restored bootstrap", &s, 1, &r) || !dv.generate(s, r) ||
        !dv.snap("prompt restored generate", &s, 1, &r)) {
        if (dv.err.empty()) {
            dv.err = "prompt: the restore started at " + std::to_string(from) + ", not the checkpoint at 8192";
        }
        return false;
    }
    return true;
}

// ---- the engine case: the split's leader and follower (fn/runtime/stage.h) over the in-process port ----

engine_config engine_cfg(const options & o) {
    engine_config c;
    c.model_path      = o.model_path;
    c.n_ctx_slot      = slot_cells;
    c.draft.on        = true;
    c.draft.attention = "sparse";
    c.draft.subset    = fn::cases::fixture_path("draft-subset");
    c.eog             = { 248046 };
    c.spec_d          = 4;
    c.timeout_s       = 120.0;
    c.record_hash     = true;
    return c;
}

// one step record: the slot, its tokens, the end, the decision hash and whether its verify was a joint lane
struct step_rec {
    int32_t  slot = -1, n = 0;
    int32_t  tok[5] = {};
    bool     end = false, joint = false;
    uint64_t hash = 0;

    bool operator==(const step_rec & b) const {
        return slot == b.slot && n == b.n && std::equal(tok, tok + 5, b.tok) && end == b.end && joint == b.joint &&
               hash == b.hash;
    }
};

// the requests as the server's shell makes them, on one box's engine or the split's leader
struct requests {
    engine *              one   = nullptr;    // one box
    engine *              front = nullptr;    // the split's front (the records)
    engine *              back  = nullptr;    // the split's back (read after a call returns; its thread waits then)
    stage::leader *       lead  = nullptr;    // the split's calls
    std::vector<step_rec> exact, functional;  // the steps that must be bitwise one box's, and the others
    std::string           err;

    // slot s's rotary offset on every box is want
    bool offsets(int32_t s, int32_t want) {
        const int32_t here  = (lead != nullptr ? front : one)->pos_offset(s);
        const int32_t there = back != nullptr ? back->pos_offset(s) : want;
        if (here != want || there != want) {
            err = "slot " + std::to_string(s) + "'s offset " + std::to_string(here) +
                  (back != nullptr ? " (the back's " + std::to_string(there) + ")" : "") + ", want " +
                  std::to_string(want);
            return false;
        }
        return true;
    }

    run_status sampling(int32_t s, float temp, uint64_t seed, int32_t max_tokens) {
        request_options o;
        o.sampling          = fn::samp::options();
        o.sampling.temp     = temp;
        o.sampling.top_k    = 20;
        o.sampling.seed     = seed;
        o.max_tokens        = max_tokens;
        const run_status rs = lead != nullptr ? lead->set_sampling(s, o) : one->set_sampling(s, o);
        if (rs != run_status::ok) {
            err = "set_sampling: " + (lead != nullptr ? lead->error() : one->error());
        }
        return rs;
    }

    run_status prefill(int32_t                           s,
                       const std::vector<int32_t> &      all,
                       int32_t                           n_reuse,
                       const std::vector<prompt_image> * images = nullptr) {
        int32_t          np = 0;
        const run_status rs =
            lead != nullptr ?
                lead->prefill(s, all.data() + n_reuse, int32_t(all.size()) - n_reuse, n_reuse, np, images) :
                one->prefill(s, all.data() + n_reuse, int32_t(all.size()) - n_reuse, n_reuse, np, {}, images);
        if (rs != run_status::ok) {
            err = "prefill: " + (lead != nullptr ? lead->error() : one->error());
        }
        return rs;
    }

    bool step(const std::vector<int32_t> & slots, int32_t times, bool strict) {
        for (int32_t t = 0; t < times; ++t) {
            joint_step_out r[stage::max_members];
            const int32_t  n = int32_t(slots.size());
            if (lead != nullptr) {
                lead->step_group(slots.data(), n, r);
            } else if (n == 1) {
                r[0].st  = one->step(slots[0], r[0].out, r[0].n_out, r[0].end);
                r[0].why = one->error();
            } else {
                one->step_joint(slots.data(), n, r);
            }
            for (int32_t k = 0; k < n; ++k) {
                if (r[k].st != run_status::ok) {
                    err = "step of slot " + std::to_string(slots[size_t(k)]) + ": " + r[k].why;
                    return false;
                }
                step_rec x;
                x.slot = slots[size_t(k)];
                x.n    = r[k].n_out;
                std::copy(r[k].out, r[k].out + 5, x.tok);
                x.end   = r[k].end;
                x.joint = r[k].joint;
                x.hash  = (lead != nullptr ? front : one)->split_last_hash(x.slot);
                (strict ? exact : functional).push_back(x);
            }
        }
        return true;
    }

    run_status truncate(int32_t s, int32_t n_keep) {
        return lead != nullptr ? lead->truncate(s, n_keep) : one->truncate(s, n_keep);
    }

    run_status release(int32_t s) { return lead != nullptr ? lead->release_slot(s) : one->release_slot(s); }

    int64_t held(int32_t s) { return (lead != nullptr ? front : one)->held(s); }
};

bool engine_script(requests & q, std::function<bool()> inject_restore_failure) {
    // one user: a prompt, steps, a truncate within the last step, steps again, a request at T = 1, a long prompt
    // and a restore of its 8,192 checkpoint (every step bitwise one box's)
    const std::vector<int32_t> a = tokens_of(901, 700);
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, a, 0) != run_status::ok ||
        !q.step({ 0 }, 12, true)) {
        return false;
    }
    // the shell's truncate (a stop string): the kept prefix, then its next prompt reads the kept tokens and new ones
    const int64_t        h = q.held(0);
    std::vector<int32_t> hist;
    int32_t              pend = -1;
    if (q.truncate(0, int32_t(h - 1)) != run_status::ok ||
        (q.lead != nullptr ? q.front : q.one)->history(0, hist, pend) != run_status::ok) {
        q.err = "truncate: " + q.err;
        return false;
    }
    const int32_t kept = int32_t(hist.size());
    hist.push_back(1234);
    hist.push_back(5678);
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, hist, kept) != run_status::ok ||
        !q.step({ 0 }, 4, true)) {
        q.err = "after the truncate: " + q.err;
        return false;
    }
    const std::vector<int32_t> b = tokens_of(902, 300);
    if (q.sampling(0, 1.0f, 20261013, 512) != run_status::ok || q.prefill(0, b, 0) != run_status::ok ||
        !q.step({ 0 }, 10, true)) {
        return false;
    }
    std::vector<int32_t> e = tokens_of(903, 9000);
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, e, 0) != run_status::ok ||
        !q.step({ 0 }, 3, true)) {
        return false;
    }
    std::vector<int32_t>       e2(e.begin(), e.begin() + 8300);
    const std::vector<int32_t> tail = tokens_of(904, 200);
    e2.insert(e2.end(), tail.begin(), tail.end());
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, e2, 8300) != run_status::ok ||
        !q.step({ 0 }, 4, true)) {
        q.err = "restore: " + q.err;
        return false;
    }
    // four users at once from their prompts: a stable joint group (bitwise one box's)
    for (int32_t s = 0; s < 4; ++s) {
        if (q.sampling(s, 0.0f, 0, 512) != run_status::ok ||
            q.prefill(s, tokens_of(910 + uint64_t(s), 200 + 37 * s), 0) != run_status::ok) {
            return false;
        }
    }
    if (!q.step({ 0, 1, 2, 3 }, 10, true)) {
        return false;
    }
    // the lifecycle (membership changes: functional, T = 0 tokens one box's): a release, three users, a reset by a new
    // prompt on slot 2, a new request bound to slot 3
    if (q.release(3) != run_status::ok || !q.step({ 0, 1, 2 }, 4, false) ||
        q.prefill(2, tokens_of(920, 150), 0) != run_status::ok || !q.step({ 0, 1, 2 }, 4, false) ||
        q.sampling(3, 0.0f, 0, 512) != run_status::ok || q.prefill(3, tokens_of(921, 90), 0) != run_status::ok ||
        !q.step({ 0, 1, 2, 3 }, 4, false)) {
        return false;
    }
    // the split only: a restore on one box failing on the other fails the request; the slot then reads again
    return inject_restore_failure == nullptr || inject_restore_failure();
}

bool engine_case(const options & o, bool eager, std::string & report) {
    // one box
    requests ref;
    {
        md::window w;
        loaded     L;
        if (!load_synth(o, w, L)) {
            report = "the one-box load";
            return false;
        }
        engine_config c = engine_cfg(o);
        c.eager         = eager;
        engine      e;
        std::string err;
        if (!e.init(std::move(L.m), c, err)) {
            report = "the one-box engine: " + err;
            return false;
        }
        ref.one = &e;
        if (!engine_script(ref, nullptr)) {
            report = "one box: " + ref.err;
            return false;
        }
    }
    // the split at 27, the follower on its own thread
    const int32_t k = 27;
    md::window    wf, wb;
    wf.hi   = k - 1;
    wf.head = false;
    wb.lo   = k;
    loaded LF, LB;
    if (!load_synth(o, wf, LF) || !load_synth(o, wb, LB)) {
        report = "the split's loads";
        return false;
    }
    std::unique_ptr<stage::port> pa, pb;
    stage::loop_pair(pa, pb);
    stage::buffers  bf, bb;
    stage::crossing xf, xb;
    std::string     err;
    if (!bf.alloc(n_slot, pa.get(), err) || !bb.alloc(n_slot, pb.get(), err) || !xf.init(*pa, true, k, err) ||
        !xb.init(*pb, false, k, err)) {
        report = "the split's buffers: " + err;
        return false;
    }
    engine_config cf = engine_cfg(o), cb = engine_cfg(o);
    cf.eager = cb.eager = eager;
    cf.stage            = 1;
    cf.stage_buf        = bf.slot;
    cf.record_hash      = false;
    cb.stage            = 2;
    cb.stage_buf        = bb.slot;
    engine ef, eb;
    if (!ef.init(std::move(LF.m), cf, err) || !eb.init(std::move(LB.m), cb, err)) {
        report = "the split's engines: " + err;
        return false;
    }
    ef.prefill_driver().set_stage(&xf);
    eb.prefill_driver().set_stage(&xb);
    stage::leader   L;
    stage::follower F;
    if (!L.init(ef, *pa, xf, err) || !F.init(eb, *pb, xb, err)) {
        report = "the split's stages: " + err;
        return false;
    }
    std::string fwhy;
    bool        fbye = false;
    std::thread fth([&]() { fbye = F.serve(fwhy); });
    requests    sp;
    sp.front = &ef;
    sp.lead  = &L;
    std::string inject;
    const bool  ok = engine_script(sp, [&]() {
        // the long prompt again on slot 1, then a restore of its checkpoint with the back's checkpoints dropped (the
        // back resets and reads from token 0): the request fails; then a fresh read works
        std::vector<int32_t> e = tokens_of(930, 9000);
        if (sp.sampling(1, 0.0f, 0, 64) != run_status::ok || sp.prefill(1, e, 0) != run_status::ok ||
            !sp.step({ 1 }, 2, false)) {
            inject = "the restore case's prompt: " + sp.err;
            return false;
        }
        eb.prefill_driver().drop_owner(1);  // the injection: the back's checkpoints of slot 1 gone
        std::vector<int32_t>       e2(e.begin(), e.begin() + 8300);
        const std::vector<int32_t> t2 = tokens_of(931, 100);
        e2.insert(e2.end(), t2.begin(), t2.end());
        const run_status rs = sp.prefill(1, e2, 8300);
        if (rs == run_status::ok) {
            inject = "the restore failing on the back did NOT fail the request";
            return false;
        }
        const std::string why = sp.err;
        if (sp.prefill(1, e2, 0) != run_status::ok || !sp.step({ 1 }, 3, false)) {
            inject = "after the failed restore, a fresh read: " + sp.err;
            return false;
        }
        inject = "the request failed as required (" + why + "), then a fresh read stepped";
        return true;
    });
    L.bye();
    fth.join();
    if (!ok) {
        report = "the split: " + sp.err + (inject.empty() ? "" : "; " + inject);
        return false;
    }
    // the comparison
    int32_t bad = 0, bad_tok = 0;
    if (ref.exact.size() != sp.exact.size()) {
        report = "the exact steps: " + std::to_string(sp.exact.size()) + " against one box's " +
                 std::to_string(ref.exact.size());
        return false;
    }
    std::string diffs;
    for (size_t i = 0; i < ref.exact.size(); ++i) {
        const bool same = ref.exact[i] == sp.exact[i];
        bad += same ? 0 : 1;
        if (!same && bad <= 6) {
            const auto str = [](const step_rec & r) {
                char buf[200];
                snprintf(buf, sizeof buf, "slot %d n %d tok %d,%d,%d,%d,%d end %d joint %d hash %016llx", r.slot, r.n,
                         r.tok[0], r.tok[1], r.tok[2], r.tok[3], r.tok[4], r.end ? 1 : 0, r.joint ? 1 : 0,
                         (unsigned long long) r.hash);
                return std::string(buf);
            };
            diffs += "\n  step " + std::to_string(i) + ": one box " + str(ref.exact[i]) + "\n  " +
                     std::string(std::to_string(i).size() + 7, ' ') + "split   " + str(sp.exact[i]);
        }
    }
    if (!diffs.empty()) {
        printf("split-test engine: the first differing exact steps:%s\n", diffs.c_str());
    }
    for (size_t i = 0; i < std::min(ref.functional.size(), sp.functional.size()); ++i) {
        const step_rec & x = ref.functional[i];
        const step_rec & y = sp.functional[i];
        bad_tok += x.slot == y.slot && x.n == y.n && std::equal(x.tok, x.tok + 5, y.tok) ? 0 : 1;
    }
    const stage::leader_stats & ls = L.stats();
    report = std::to_string(ref.exact.size()) +
             " steps bitwise one box's: " + (bad == 0 ? "PASS" : std::to_string(bad) + " differ: FAIL") + "; " +
             std::to_string(ref.functional.size()) + " lifecycle steps (" + std::to_string(sp.functional.size()) +
             " on the split): " +
             (bad_tok == 0 && ref.functional.size() <= sp.functional.size() ?
                  "the same T = 0 tokens" :
                  std::to_string(bad_tok) + " differ in their tokens") +
             "; " + inject + "; " + std::to_string(ls.groups) + " step groups, " + std::to_string(ls.boundaries) +
             " boundaries, " + std::to_string(xf.count()) + " prompt crossings; follower " +
             (fbye ? "ended at X-BYE" : "lost: " + fwhy);
    return bad == 0 && fbye;
}

// ---- MUP2 piece 1: reads in pieces and units, bitwise today's lone read ----

// Two reads on slot 0: 9,000 tokens (chunks of 8,192 and 808 rows) and a 3,000-token append at cell 100 (off the
// 64-cell grid: a QSA block and GDN's blocks cross the piece cuts). The reference is today's lone read (prefill: the
// halves). The arms read the same tokens (test_piece_rows) in units beside writers, one unit a step (writers mode, a
// budget unknown, so the host pauses between every two units), at level 1 and in pieces of 2,048, 512 and 64
// rows, and in pieces of 512 rows lone and 64 rows shared (no pauses), and as the server's lone reader (shared, no
// budget: whole chunks, no unit on either box, the back's account read from the leader). After each read every state
// part of every
// layer, the PLE history and the head's logits and carry, then four T = 0 steps' tokens and decision hashes, are
// bitwise the reference's. On one box, then on the split at k = 27 (the pieces sent with the crossing), against one
// box's reference
struct piece_arm {
    const char * name;
    int32_t      rows;  // 0: chosen (level 1 with the budget unknown)
    read_mode    mode;
    bool         budget;
};

const piece_arm piece_arms[] = {
    { "level 1, units",                    0,    read_mode::writers, true  },
    { "2,048-row units",                   2048, read_mode::writers, true  },
    { "512-row units",                     512,  read_mode::writers, true  },
    { "64-row units",                      64,   read_mode::writers, true  },
    { "512-row pieces, lone",              512,  read_mode::lone,    false },
    { "64-row pieces, shared",             64,   read_mode::shared,  false },
    // the server's lone reader (no writer step pending: its shared mode, no budget): whole chunks on both boxes, 0 units
    { "shared, no budget (a lone reader)", 0,    read_mode::shared,  false },
};

struct piece_out {
    snapshot              snap;
    std::vector<step_rec> steps;
    int64_t               front_units = 0, back_units = 0;  // the read's units on this box (the front) and the back
};

// a read's end on slot s: the slot's state (both boxes on the split), then four steps from it and the release
bool piece_tail(engine &        front,
                engine *        back,
                stage::leader * lead,
                requests &      q,
                int32_t         s,
                piece_out &     out,
                std::string &   err) {
    reader        rd;
    const int32_t cf = int32_t(front.test_target(s).slot().cur);
    const int32_t cb = back != nullptr ? int32_t(back->test_target(s).slot().cur) : cf;
    const int64_t gf = front.test_target(s).slot().gdn;
    const int64_t gb = back != nullptr ? int64_t(back->test_target(s).slot().gdn) : gf;
    if (cf != cb || !state_of(front.test_model(), s, out.snap, rd, cf, gf) ||
        (back != nullptr && !state_of(back->test_model(), s, out.snap, rd, cb, gb))) {
        err = "the state read";
        return false;
    }
    // the decision hashes carry the deciding target's round count, which every earlier run on the slot advanced: the
    // four steps' rounds counted from 1 in every run (the back's on the split)
    engine &       dec  = back != nullptr ? *back : front;
    const uint64_t base = dec.test_target(s).round_next() - 1, have = dec.test_round_bias(s);
    if (base > have) {
        dec.split_discard_rounds(s, base - have);
    }
    if (!q.step({ s }, 4, true)) {
        err = q.err;
        return false;
    }
    out.steps           = q.exact;
    const run_status rs = lead != nullptr ? lead->release_slot(s) : front.release_slot(s);
    if (rs != run_status::ok) {
        err = "release";
        return false;
    }
    return true;
}

// one read on slot 0 (a == nullptr: the reference), then its state and four steps
bool piece_run(engine &                     front,
               engine *                     back,
               stage::leader *              lead,
               const std::vector<int32_t> & all,
               int32_t                      reuse,
               const piece_arm *            a,
               piece_out &                  out,
               std::string &                err) {
    requests q;
    q.one           = lead == nullptr ? &front : nullptr;
    q.front         = &front;
    q.lead          = lead;
    const int32_t s = 0;
    if (q.sampling(s, 0.0f, 0, 512) != run_status::ok ||
        (reuse > 0 && q.prefill(s, std::vector<int32_t>(all.begin(), all.begin() + reuse), 0) != run_status::ok)) {
        err = q.err;
        return false;
    }
    if (a == nullptr) {
        if (q.prefill(s, all, reuse) != run_status::ok) {
            err = q.err;
            return false;
        }
    } else {
        front.test_piece_rows(a->rows);
        int32_t       np = 0;
        const int32_t rb = lead != nullptr ?
                               lead->read_begin(s, all.data() + reuse, int32_t(all.size()) - reuse, reuse, np) :
                               front.read_begin(s, all.data() + reuse, int32_t(all.size()) - reuse, reuse, np);
        if (rb != 0) {
            front.test_piece_rows(0);
            err = std::string("read_begin: ") + (lead != nullptr ? lead->error() : front.error());
            return false;
        }
        read_budget b;  // unknown: one unit a step
        read_status st = read_status::yielded;
        for (int64_t n = 0;
             st == read_status::yielded || st == read_status::chunk_done || st == read_status::front_done; ++n) {
            const int32_t       g  = a->mode == read_mode::shared ? 2 : 1;
            const read_budget * bp = a->budget ? &b : nullptr;
            st = lead != nullptr ? lead->read_step(s, g, a->mode, bp) : front.read_step(s, g, a->mode, bp);
        }
        front.test_piece_rows(0);
        if (st != read_status::done) {
            err = std::string("read: ") + (lead != nullptr ? lead->error() : front.error());
            return false;
        }
        read_info fi, bi;
        front.get_read_info(s, fi);
        out.front_units = fi.units_run;
        if (lead != nullptr) {
            lead->back_info(s, bi);
            out.back_units = bi.units_run;
        }
    }
    return piece_tail(front, back, lead, q, s, out, err);
}

// the arms' differences from the reference ("" when bitwise equal); a lone reader's arm (shared, no budget) also
// runs no unit on either box (whole chunks, MUP1's geometry)
std::string piece_diff(const piece_out & ref, const piece_out & x, const piece_arm * a = nullptr) {
    const bool        whole = a != nullptr && a->mode == read_mode::shared && !a->budget && a->rows == 0;
    const std::string units =
        whole && (x.front_units != 0 || x.back_units != 0) ?
            "units ran (front " + std::to_string(x.front_units) + ", back " + std::to_string(x.back_units) + ")" :
            "";
    int32_t     parts = 0;
    std::string first;
    for (const auto & kv : ref.snap.state) {
        const auto it = x.snap.state.find(kv.first);
        if (it == x.snap.state.end() || it->second != kv.second) {
            ++parts;
            first = first.empty() ? kv.first : first;
        }
    }
    int32_t     steps = ref.steps.size() == x.steps.size() ? 0 : 1;
    std::string first_step;
    for (size_t i = 0; i < std::min(ref.steps.size(), x.steps.size()); ++i) {
        const step_rec & a = ref.steps[i];
        const step_rec & b = x.steps[i];
        if (a == b) {
            continue;
        }
        ++steps;
        if (first_step.empty()) {
            first_step = " (first: step " + std::to_string(i) + ": n " + std::to_string(a.n) + "/" +
                         std::to_string(b.n) + ", tok0 " + std::to_string(a.tok[0]) + "/" + std::to_string(b.tok[0]) +
                         ", end " + std::to_string(a.end) + "/" + std::to_string(b.end) + ", joint " +
                         std::to_string(a.joint) + "/" + std::to_string(b.joint) + ", hash " +
                         (a.hash == b.hash ? "equal" : "differs") + ")";
        }
    }
    if (parts == 0 && steps == 0 && ref.snap.state.size() == x.snap.state.size()) {
        return units;
    }
    return (units.empty() ? "" : units + "; ") + std::to_string(parts) + " of " +
           std::to_string(ref.snap.state.size()) + " parts differ (first " + first + "), " + std::to_string(steps) +
           " steps" + first_step;
}

bool pieces_case(const options & o, bool eager, std::string & report) {
    const std::vector<int32_t> long_read = tokens_of(940, 9000);
    const std::vector<int32_t> append    = tokens_of(941, 3100);  // 100 kept, 3,000 appended

    struct read_kind {
        const char *                 name;
        const std::vector<int32_t> * all;
        int32_t                      reuse;
    };

    const read_kind kinds[] = {
        { "9,000 tokens",      &long_read, 0   },
        { "3,000 at cell 100", &append,    100 }
    };
    piece_out   ref[2];
    int32_t     bad = 0, runs = 0;
    std::string lines;
    const auto  note = [&](const std::string & who, const std::string & d) {
        ++runs;
        if (!d.empty()) {
            ++bad;
            lines += (lines.empty() ? "" : "; ") + who + ": " + d;
        }
    };
    {
        md::window w;  // the small shape (san, --small): layers 0 .. 5 with the head, one box only (memcheck's form)
        w.lo = o.lo;
        w.hi = o.hi;
        loaded L;
        if (!load_synth(o, w, L)) {
            report = "the one-box load";
            return false;
        }
        engine_config c = engine_cfg(o);
        c.eager         = eager;
        engine      e;
        std::string err;
        if (!e.init(std::move(L.m), c, err)) {
            report = "the one-box engine: " + err;
            return false;
        }
        for (int32_t k = 0; k < 2; ++k) {
            if (!piece_run(e, nullptr, nullptr, *kinds[k].all, kinds[k].reuse, nullptr, ref[k], err)) {
                report = std::string("one box, the reference (") + kinds[k].name + "): " + err;
                return false;
            }
            for (const piece_arm & a : piece_arms) {
                piece_out x;
                if (!piece_run(e, nullptr, nullptr, *kinds[k].all, kinds[k].reuse, &a, x, err)) {
                    report = std::string("one box, ") + a.name + " (" + kinds[k].name + "): " + err;
                    return false;
                }
                note(std::string("one box ") + a.name + " (" + kinds[k].name + ")", piece_diff(ref[k], x, &a));
            }
        }
    }
    if (o.lo != 0 || o.hi != fn::n_layer - 1) {
        report =
            std::to_string(runs) + " one-box reads (layers " + std::to_string(o.lo) + " .. " + std::to_string(o.hi) +
            ") against today's lone read: " + (bad == 0 ? "bitwise equal" : std::to_string(bad) + " differ: " + lines);
        return bad == 0;
    }
    // the split at 27 over the in-process port, the follower on its own thread
    const int32_t k = 27;
    md::window    wf, wb;
    wf.hi   = k - 1;
    wf.head = false;
    wb.lo   = k;
    loaded LF, LB;
    if (!load_synth(o, wf, LF) || !load_synth(o, wb, LB)) {
        report = "the split's loads";
        return false;
    }
    std::unique_ptr<stage::port> pa, pb;
    stage::loop_pair(pa, pb);
    stage::buffers  bf, bb;
    stage::crossing xf, xb;
    std::string     err;
    if (!bf.alloc(n_slot, pa.get(), err) || !bb.alloc(n_slot, pb.get(), err) || !xf.init(*pa, true, k, err) ||
        !xb.init(*pb, false, k, err)) {
        report = "the split's buffers: " + err;
        return false;
    }
    engine_config cf = engine_cfg(o), cb = engine_cfg(o);
    cf.eager = cb.eager = eager;
    cf.stage            = 1;
    cf.stage_buf        = bf.slot;
    cf.record_hash      = false;
    cb.stage            = 2;
    cb.stage_buf        = bb.slot;
    engine ef, eb;
    if (!ef.init(std::move(LF.m), cf, err) || !eb.init(std::move(LB.m), cb, err)) {
        report = "the split's engines: " + err;
        return false;
    }
    ef.prefill_driver().set_stage(&xf);
    eb.prefill_driver().set_stage(&xb);
    stage::leader   L;
    stage::follower F;
    if (!L.init(ef, *pa, xf, err) || !F.init(eb, *pb, xb, err)) {
        report = "the split's stages: " + err;
        return false;
    }
    std::string fwhy;
    bool        fbye = false;
    std::thread fth([&]() { fbye = F.serve(fwhy); });
    std::string fail;
    for (int32_t kk = 0; kk < 2 && fail.empty(); ++kk) {
        piece_out x;
        if (!piece_run(ef, &eb, &L, *kinds[kk].all, kinds[kk].reuse, nullptr, x, err)) {
            fail = std::string("the split, lone (") + kinds[kk].name + "): " + err;
            break;
        }
        note(std::string("split lone (") + kinds[kk].name + ")", piece_diff(ref[kk], x));
        for (const piece_arm & a : piece_arms) {
            piece_out y;
            if (!piece_run(ef, &eb, &L, *kinds[kk].all, kinds[kk].reuse, &a, y, err)) {
                fail = std::string("the split, ") + a.name + " (" + kinds[kk].name + "): " + err;
                break;
            }
            note(std::string("split ") + a.name + " (" + kinds[kk].name + ")", piece_diff(ref[kk], y, &a));
        }
    }
    L.bye();
    fth.join();
    if (!fail.empty()) {
        report = fail;
        return false;
    }
    report =
        std::to_string(runs) + " reads against today's lone read (" + std::to_string(ref[0].snap.state.size()) +
        " state parts and 4 steps each): " + (bad == 0 ? "bitwise equal" : std::to_string(bad) + " differ: " + lines) +
        "; follower " + (fbye ? "ended at X-BYE" : "lost: " + fwhy);
    return bad == 0 && fbye;
}

// ---- VIS1 Stage B: the image case: one box against the split (k = 27) with image rows ----

// an image's synthetic rows: small values from its digest and position (the embedding's scale)
void synth_image_rows(const image_span & s, float * out) {
    for (int64_t i = 0; i < s.tokens() * vision::out_width; ++i) {
        out[i] = 0.05f * uniform(s.digest, uint64_t(i));
    }
}

// a prompt of n tokens with images (nx, ny, digest) at the given cells: their cells hold image_token
struct img_prompt {
    std::vector<int32_t>      tok;
    std::vector<prompt_image> im;
};

img_prompt image_prompt(uint64_t seed, int32_t n, const std::vector<image_span> & spans) {
    img_prompt p;
    p.tok = tokens_of(seed, n);
    for (const image_span & sp : spans) {
        std::fill(p.tok.begin() + sp.cell, p.tok.begin() + sp.end(), image_token);
        prompt_image pi;
        pi.span = sp;
        p.im.push_back(pi);
    }
    return p;
}

bool image_script(requests & q) {
    // two images in a short prompt (a 13 x 9 and a 6 x 20 grid), steps; the next turn reuses them (a pure extension)
    const img_prompt a = image_prompt(940, 420,
                                      {
                                          { 41,  13, 9,  0x1111 },
                                          { 190, 6,  20, 0x2222 }
    });
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, a.tok, 0, &a.im) != run_status::ok ||
        !q.step({ 0 }, 8, true)) {
        q.err = "two images: " + q.err;
        return false;
    }
    std::vector<int32_t> hist;
    int32_t              pend = -1;
    if ((q.lead != nullptr ? q.front : q.one)->history(0, hist, pend) != run_status::ok) {
        q.err = "history";
        return false;
    }
    const int32_t kept = int32_t(hist.size());
    hist.push_back(4321);
    hist.push_back(8765);
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, hist, kept, &a.im) != run_status::ok ||
        !q.step({ 0 }, 4, true)) {
        q.err = "the next turn: " + q.err;
        return false;
    }
    // the same tokens with the second image changed: the reuse stops at its first cell (read again)
    img_prompt b        = a;
    b.im[1].span.digest = 0x3333;
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, b.tok, 400, &b.im) != run_status::ok ||
        !q.step({ 0 }, 4, true)) {
        q.err = "another second image: " + q.err;
        return false;
    }
    // a long prompt whose 64 x 40 image straddles the chunk end at 8,192 (its rows in two chunks, the checkpoint
    // there skipped), then a re-read of most of it
    const img_prompt c = image_prompt(950, 10000,
                                      {
                                          { 7000, 64, 40, 0x4444 }
    });
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, c.tok, 0, &c.im) != run_status::ok ||
        !q.step({ 0 }, 3, true)) {
        q.err = "the straddling image: " + q.err;
        return false;
    }
    std::vector<int32_t>       c2(c.tok.begin(), c.tok.begin() + 9700);
    const std::vector<int32_t> tail = tokens_of(951, 50);
    c2.insert(c2.end(), tail.begin(), tail.end());
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, c2, 9700, &c.im) != run_status::ok ||
        !q.step({ 0 }, 3, true)) {
        q.err = "the straddling image's re-read: " + q.err;
        return false;
    }
    // a second user with the first image at another cell, then both together (a joint group)
    const img_prompt d = image_prompt(960, 300,
                                      {
                                          { 17, 13, 9, 0x1111 }
    });
    if (q.sampling(1, 0.0f, 0, 512) != run_status::ok || q.prefill(1, d.tok, 0, &d.im) != run_status::ok ||
        q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, a.tok, 0, &a.im) != run_status::ok ||
        !q.step({ 0, 1 }, 6, true)) {
        q.err = "two users: " + q.err;
        return false;
    }
    // VIS1 Load 3's finding: a cut after an image turn sets the slot's offset to the kept ledger's on every box (a cut
    // keeping the first image only: its shift; the server's erase, a cut to 0: 0), so the next text request, which
    // carries no ledger, reads and steps at cell positions
    image_rows  first;
    std::string ferr;
    if (!first.set(a.tok.data(), 170, { a.im[0].span }, ferr)) {
        q.err = "the first image's ledger: " + ferr;
        return false;
    }
    if (first.offset_at(170) == 0 || q.truncate(0, 170) != run_status::ok || !q.offsets(0, first.offset_at(170)) ||
        q.truncate(0, 0) != run_status::ok || !q.offsets(0, 0)) {
        q.err = "the cut's offset: " + q.err;
        return false;
    }
    const std::vector<int32_t> e = tokens_of(970, 150);
    if (q.sampling(0, 0.0f, 0, 512) != run_status::ok || q.prefill(0, e, 0) != run_status::ok || !q.offsets(0, 0) ||
        !q.step({ 0 }, 4, true)) {
        q.err = "a text request after the cut: " + q.err;
        return false;
    }
    return true;
}

bool image_case(const options & o, bool eager, std::string & report) {
    md::vision_params rows_vp;  // the synthetic rows: no weights or scratch
    md::vision_params planes_vp;
    planes_vp.planes_only    = true;
    md::vision_params one_vp = rows_vp;  // one box drafts over its supplied rows: the image token's row
    one_vp.token_row         = true;
    requests ref;
    {
        md::window w;
        loaded     L;
        if (!load_synth(o, w, L, n_slot, &one_vp)) {
            report = "the one-box load";
            return false;
        }
        engine_config c   = engine_cfg(o);
        c.eager           = eager;
        c.test_image_rows = synth_image_rows;
        engine      e;
        std::string err;
        if (!e.init(std::move(L.m), c, err)) {
            report = "the one-box engine: " + err;
            return false;
        }
        ref.one = &e;
        if (!image_script(ref)) {
            report = "one box: " + ref.err;
            return false;
        }
    }
    const int32_t k = 27;
    md::window    wf, wb;
    wf.hi   = k - 1;
    wf.head = false;
    wb.lo   = k;
    loaded LF, LB;
    if (!load_synth(o, wf, LF, n_slot, &rows_vp) || !load_synth(o, wb, LB, n_slot, &planes_vp)) {
        report = "the split's loads";
        return false;
    }
    std::unique_ptr<stage::port> pa, pb;
    stage::loop_pair(pa, pb);
    stage::buffers  bf, bb;
    stage::crossing xf, xb;
    std::string     err;
    if (!bf.alloc(n_slot, pa.get(), err) || !bb.alloc(n_slot, pb.get(), err) || !xf.init(*pa, true, k, err) ||
        !xb.init(*pb, false, k, err)) {
        report = "the split's buffers: " + err;
        return false;
    }
    engine_config cf = engine_cfg(o), cb = engine_cfg(o);
    cf.eager = cb.eager = eager;
    cf.stage            = 1;
    cf.stage_buf        = bf.slot;
    cf.record_hash      = false;
    cf.test_image_rows  = synth_image_rows;
    cb.stage            = 2;
    cb.stage_buf        = bb.slot;
    cb.image_planes     = true;
    engine ef, eb;
    if (!ef.init(std::move(LF.m), cf, err) || !eb.init(std::move(LB.m), cb, err)) {
        report = "the split's engines: " + err;
        return false;
    }
    ef.prefill_driver().set_stage(&xf);
    eb.prefill_driver().set_stage(&xb);
    stage::leader   L;
    stage::follower F;
    if (!L.init(ef, *pa, xf, err) || !F.init(eb, *pb, xb, err)) {
        report = "the split's stages: " + err;
        return false;
    }
    std::string fwhy;
    bool        fbye = false;
    std::thread fth([&]() { fbye = F.serve(fwhy); });
    requests    sp;
    sp.front      = &ef;
    sp.back       = &eb;
    sp.lead       = &L;
    const bool ok = image_script(sp);
    L.bye();
    fth.join();
    if (!ok) {
        report = "the split: " + sp.err;
        return false;
    }
    int32_t bad = 0;
    for (size_t i = 0; i < std::min(ref.exact.size(), sp.exact.size()); ++i) {
        bad += ref.exact[i] == sp.exact[i] ? 0 : 1;
    }
    const bool same_n = ref.exact.size() == sp.exact.size();
    report = std::to_string(ref.exact.size()) + " steps (" + std::to_string(sp.exact.size()) + " on the split), " +
             (bad == 0 && same_n ? "bitwise one box's" : std::to_string(bad) + " differ") + "; " +
             std::to_string(xf.count()) + " prompt crossings; follower " + (fbye ? "ended at X-BYE" : "lost: " + fwhy);
    return bad == 0 && same_n && fbye;
}

// PRD1: the split at k = 27 over the in-process port, the follower serving on its own thread (pieces_case's form)
struct inproc_split {
    static constexpr int32_t     k = 27;
    std::unique_ptr<stage::port> pa, pb;
    stage::buffers               bf, bb;
    stage::crossing              xf, xb;
    // the synthetic loads' host parts (the zero PLE mapping) outlive the engines that read them (declared before
    // them: destroyed after)
    loaded                       lf, lb;
    engine                       ef, eb;
    stage::leader                L;
    stage::follower              F;
    std::string                  fwhy;
    bool                         fbye = false, running = false;
    std::thread                  fth;

    // idle_poll_s and test_order set on the follower before it serves
    bool up(const options & o, bool eager, std::string & report, double idle_poll_s = 1.0, bool test_order = false) {
        md::window wf, wb;
        wf.hi   = k - 1;
        wf.head = false;
        wb.lo   = k;
        if (!load_synth(o, wf, lf) || !load_synth(o, wb, lb)) {
            report = "the split's loads";
            return false;
        }
        stage::loop_pair(pa, pb);
        std::string err;
        if (!bf.alloc(n_slot, pa.get(), err) || !bb.alloc(n_slot, pb.get(), err) || !xf.init(*pa, true, k, err) ||
            !xb.init(*pb, false, k, err)) {
            report = "the split's buffers: " + err;
            return false;
        }
        engine_config cf = engine_cfg(o), cb = engine_cfg(o);
        cf.eager = cb.eager = eager;
        cf.stage            = 1;
        cf.stage_buf        = bf.slot;
        cf.record_hash      = false;
        cb.stage            = 2;
        cb.stage_buf        = bb.slot;
        if (!ef.init(std::move(lf.m), cf, err) || !eb.init(std::move(lb.m), cb, err)) {
            report = "the split's engines: " + err;
            return false;
        }
        ef.prefill_driver().set_stage(&xf);
        eb.prefill_driver().set_stage(&xb);
        if (!L.init(ef, *pa, xf, err) || !F.init(eb, *pb, xb, err)) {
            report = "the split's stages: " + err;
            return false;
        }
        F.idle_poll_s = idle_poll_s;
        F.test_order  = test_order;
        fth           = std::thread([this]() { fbye = F.serve(fwhy); });
        running       = true;
        return true;
    }

    void down() {
        if (running) {
            L.bye();
            fth.join();
            running = false;
        }
    }

    ~inproc_split() { down(); }
};

// PRD1 item 4: a forced joint-capture failure on the split's front: the latch (joint verify off from then on), the
// refusal said once, and the later groups verifying alone and succeeding; and the refusals' say-once rule itself
bool latch_case(const options & o, bool eager, std::string & report) {
    if (o.lo != 0 || o.hi != fn::n_layer - 1) {
        report = "the full shape only (skipped)";
        return true;
    }
    // the rule: a cause said once however often refused; a failed capture latches
    stage::joint_refusals jr;
    jr.refuse("TEST", "a cause", false, "alone");
    jr.refuse("TEST", "a cause", false, "alone");
    const bool once = jr.said.size() == 1 && !jr.off;
    jr.refuse("TEST", "a failed capture", true, "alone");
    const bool   latched = jr.off && jr.said.size() == 2;
    inproc_split sp;
    if (!sp.up(o, eager, report)) {
        return false;
    }
    requests q;
    q.front = &sp.ef;
    q.lead  = &sp.L;
    bool ok = true;
    for (int32_t s = 0; s < 2 && ok; ++s) {
        ok = q.sampling(s, 0.0f, 0, 512) == run_status::ok &&
             q.prefill(s, tokens_of(980 + uint64_t(s), 300), 0) == run_status::ok;
    }
    joint_frame * jf = sp.ef.split_joint();
    ok               = ok && jf != nullptr;
    if (jf != nullptr) {
        jf->test_fail_captures(1);
    }
    // the first group's joint verify fails to capture: refused, latched; then six more groups, all alone
    ok                                  = ok && q.step({ 0, 1 }, 7, false);
    const stage::joint_refusals & v     = sp.L.test_joint_verify();
    int32_t                       joint = 0;
    for (const step_rec & r : q.functional) {
        joint += r.joint ? 1 : 0;
    }
    const bool split_ok = ok && v.off && v.said.size() == 1 && joint == 0 && q.functional.size() == 14;
    for (int32_t s = 0; s < 2; ++s) {
        q.release(s);
    }
    sp.down();
    report = std::string("say once ") + (once ? "ok" : "FAILED") + ", latch " + (latched ? "ok" : "FAILED") +
             "; the split: " + (ok ? "" : "the steps failed: " + q.err + "; ") + "latched " + (v.off ? "yes" : "no") +
             ", causes said " + std::to_string(v.said.size()) + ", joint steps " + std::to_string(joint) + " of " +
             std::to_string(q.functional.size()) + "; follower " + (sp.fbye ? "ended at X-BYE" : "lost: " + sp.fwhy);
    return once && latched && split_ok && sp.fbye;
}

// ---- PRD1: the split's read path (items 7, 8, 9 and 11) over the in-process port ----

// Every read is compared bitwise with its lone read on one box (the slot's state on both boxes, then four steps). The
// follower's test hold (test_hold_reads) lets the front finish reads before the back takes any, so the orders the
// checks name happen by construction, not by timing; the drain's checks wait for its progress, not for a time
bool reads_case(const options & o, bool eager, std::string & report) {
    if (o.lo != 0 || o.hi != fn::n_layer - 1) {
        report = "the full shape only (skipped)";
        return true;
    }
    const std::vector<int32_t> small = tokens_of(970, 256);   // item 7: a small read (at most 512 rows)
    const std::vector<int32_t> two   = tokens_of(971, 9000);  // two chunks (8,192 and 808 rows)
    const std::vector<int32_t> one_a = tokens_of(972, 700);   // one chunk each, not small
    const std::vector<int32_t> one_b = tokens_of(973, 900);
    const std::vector<int32_t> wp    = tokens_of(974, 150);   // the writer's prompt (slot 1)

    struct ref_read {
        int32_t                      slot;
        const std::vector<int32_t> * tok;
        piece_out                    out;
    };

    ref_read refs[] = {
        { 0, &small, {} },
        { 0, &two,   {} },
        { 0, &one_a, {} },
        { 2, &one_b, {} },
    };
    const auto ref_of = [&](int32_t s, const std::vector<int32_t> & t) -> const piece_out & {
        for (const ref_read & r : refs) {
            if (r.slot == s && r.tok == &t) {
                return r.out;
            }
        }
        return refs[0].out;
    };
    std::string err;

    // PRD1 item 14 on one box: a cancel during the read's last chunk (recorded here, checked with the split's)
    struct cancel_rec {
        bool        sent = false, retained = true, state_ok = false, again = false;
        read_status status = read_status::error;
        int64_t     held   = -1;
        piece_out   out;
    } one14;

    // a read of t on slot s to its end through step, cancelled once its last chunk is in progress
    const auto cancel_last = [](auto && step, auto && info, auto && cancel, bool & sent) {
        read_status st = read_status::yielded;
        while (st == read_status::yielded || st == read_status::chunk_done) {
            read_info ri;
            info(ri);
            if (!sent && ri.in_chunk && ri.chunk == ri.chunks - 1) {
                cancel();
                sent = true;
            }
            st = step();
        }
        return st;
    };
    {
        md::window w;
        loaded     L1;
        if (!load_synth(o, w, L1)) {
            report = "the one-box load";
            return false;
        }
        engine_config c = engine_cfg(o);
        c.eager         = eager;
        engine e;
        if (!e.init(std::move(L1.m), c, err)) {
            report = "the one-box engine: " + err;
            return false;
        }
        for (ref_read & r : refs) {
            requests q;
            q.one   = &e;
            q.front = &e;
            if (q.sampling(r.slot, 0.0f, 0, 512) != run_status::ok || q.prefill(r.slot, *r.tok, 0) != run_status::ok ||
                !piece_tail(e, nullptr, nullptr, q, r.slot, r.out, err)) {
                report = "one box, the reference: " + (err.empty() ? q.err : err);
                return false;
            }
        }
        // ---- item 14 (one box): a cancel during the last chunk of a two-chunk read ----
        requests q1;
        q1.one     = &e;
        q1.front   = &e;
        int32_t np = 0;
        if (q1.sampling(0, 0.0f, 0, 512) == run_status::ok &&
            e.read_begin(0, two.data(), int32_t(two.size()), 0, np) == 0) {
            one14.status =
                cancel_last([&]() { return e.read_step(0, 1, read_mode::shared); },
                            [&](read_info & ri) { e.get_read_info(0, ri); }, [&]() { e.read_cancel(0); }, one14.sent);
        }
        one14.held     = e.held(0);
        one14.retained = e.test_target(0).slot().retained;
        {
            reader   rd;
            snapshot sn;
            one14.state_ok = state_of(e.test_model(), 0, sn, rd, int32_t(e.test_target(0).slot().cur),
                                      e.test_target(0).slot().gdn) &&
                             sn.state == ref_of(0, two).snap.state;
        }
        // the slot free: released, then the read again from 0
        one14.again = e.release_slot(0) == run_status::ok && q1.sampling(0, 0.0f, 0, 512) == run_status::ok &&
                      q1.prefill(0, two, 0) == run_status::ok && piece_tail(e, nullptr, nullptr, q1, 0, one14.out, err);
    }
    // the split at 27 (pieces_case's)
    inproc_split sp;
    // a held follower wakes for a release without a command (its idle poll); the order log on
    if (!sp.up(o, eager, report, 0.01, true)) {
        return false;
    }
    engine &          ef = sp.ef;
    engine &          eb = sp.eb;
    stage::leader &   L  = sp.L;
    stage::follower & F  = sp.F;
    stage::crossing & xb = sp.xb;

    std::vector<std::string> bad;
    int32_t                  checks = 0;
    const auto               check  = [&](const std::string & what, bool ok, const std::string & why = "") {
        ++checks;
        if (!ok) {
            bad.push_back(what + (why.empty() ? "" : ": " + why));
        }
    };
    requests q;
    q.front           = &ef;
    q.lead            = &L;
    // the writer on slot 1, ready (its steps are the step groups beside the reads; functional)
    const auto writer = [&]() {
        q.release(1);
        return q.sampling(1, 0.0f, 0, 512) == run_status::ok && q.prefill(1, wp, 0) == run_status::ok;
    };
    const auto wstep = [&]() {
        return q.step({ 1 }, 1, false);
    };
    // a read's front: steps until it is not yielded or chunk_done (the writer stepping between when beside writers)
    const auto front = [&](int32_t s, read_mode m, const read_budget * b, int32_t groups, bool with_writer,
                           int32_t & calls) {
        read_status st = read_status::yielded;
        calls          = 0;
        while (st == read_status::yielded || st == read_status::chunk_done) {
            st = L.read_step(s, groups, m, b);
            ++calls;
            if (with_writer && (st == read_status::yielded || st == read_status::chunk_done) && !wstep()) {
                return read_status::error;
            }
        }
        return st;
    };
    // a front-done read to its end (the back's answers; polled, never counted as a step)
    const auto finish = [&](int32_t s, read_status st) {
        const auto t0 = std::chrono::steady_clock::now();
        while (st == read_status::front_done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(600)) {
            if (L.read_ready(s)) {
                st = L.read_poll(s);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        return st;
    };
    const auto begin = [&](int32_t s, const std::vector<int32_t> & t, int32_t reuse) {
        int32_t np = 0;
        if (reuse == 0 && q.sampling(s, 0.0f, 0, 512) != run_status::ok) {
            return false;
        }
        return L.read_begin(s, t.data() + reuse, int32_t(t.size()) - reuse, reuse, np) == 0;
    };
    const auto tail = [&](const std::string & what, int32_t s, const std::vector<int32_t> & t) {
        requests qt;
        qt.front = &ef;
        qt.lead  = &L;
        piece_out x;
        if (!piece_tail(ef, &eb, &L, qt, s, x, err)) {
            check(what + ": the state and steps", false, err + " " + qt.err);
            return;
        }
        const std::string d = piece_diff(ref_of(s, t), x);
        check(what + ": bitwise its lone read", d.empty(), d);
    };
    // the drain's progress: the back launched a unit since n (false: its chunk ended first; a hang guard, not a
    // budget)
    const auto drained_past = [&](int64_t n, int64_t chunk_end) {
        const auto t0 = std::chrono::steady_clock::now();
        while (eb.read_drained() <= n && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(600)) {
            read_info bi;
            eb.get_read_info(0, bi);
            if (!bi.live || bi.committed >= chunk_end) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        return eb.read_drained() > n;
    };
    read_budget served;  // item 7's budget as MUP2 served it (T 12 ms, F 0.5): APF1, a known 12 ms budget
    served.budget_ms = 12.0;
    served.known     = true;
    read_budget big;  // a budget every unit fits (a chunk's front in a step)
    big.budget_ms = 1000.0;
    big.known     = true;

    // ---- item 7: a small read beside a writer runs whole: one front step, one back step, no units ----
    for (int32_t arm = 0; arm < 2 && writer(); ++arm) {
        read_budget b = served;
        b.budget_ms   = arm == 0 ? 12.0 : -1.0;
        b.known       = arm == 0;
        const std::string what =
            std::string("7. a 256-row read beside a writer, budget ") + (arm == 0 ? "12 ms" : "unknown");
        int32_t     calls = 0;
        read_status st    = read_status::error;
        if (begin(0, small, 0)) {
            st = finish(0, front(0, read_mode::writers, &b, 1, true, calls));
        }
        read_info fi, bi, ba;
        ef.get_read_info(0, fi);
        eb.get_read_info(0, bi);
        L.back_info(0, ba);
        check(what + ": one front step, whole on the back, 0 units",
              st == read_status::done && calls == 1 && fi.steps == 1 && bi.steps == 1 && fi.units_run == 0 &&
                  bi.units_run == 0 && ba.units_run == 0,
              "status " + std::to_string(int(st)) + ", front calls " + std::to_string(calls) + ", steps " +
                  std::to_string(fi.steps) + "/" + std::to_string(bi.steps) + ", units " +
                  std::to_string(fi.units_run) + "/" + std::to_string(bi.units_run));
        tail(what, 0, small);
    }

    // ---- item 8: writers leaving mid-chunk: a chunk begun under the budget, then shared steps of 8 units ----
    if (writer() && begin(0, two, 0)) {
        read_status st = L.read_step(0, 1, read_mode::writers, &served);
        bool        ok = st == read_status::yielded && wstep();
        read_info   fi;
        ef.get_read_info(0, fi);
        const int32_t groups = fi.groups;
        std::string   why    = fi.units_run > 0 ? "" : "the chunk did not begin in units";
        int32_t       nsh    = 0;
        while (ok && st == read_status::yielded && fi.group >= 0 && fi.group < fi.groups) {
            const int32_t left = fi.groups - fi.group;
            const int64_t u0   = fi.units_run;
            st                 = L.read_step(0, 8, read_mode::shared);
            ef.get_read_info(0, fi);
            ++nsh;
            if (fi.units_run - u0 != std::min(8, left) && why.empty()) {
                why = "a shared step ran " + std::to_string(fi.units_run - u0) + " units with " + std::to_string(left) +
                      " left";
            }
            if (fi.group >= 0 && fi.groups != groups && why.empty()) {
                why = "the chunk was repartitioned";
            }
        }
        int32_t calls = 0;
        if (ok && (st == read_status::yielded || st == read_status::chunk_done)) {
            st = front(0, read_mode::shared, nullptr, 8, false, calls);
        }
        st = finish(0, st);
        check("8. writers leaving mid-chunk: min(8, the units left) a shared step, never repartitioned",
              ok && st == read_status::done && why.empty() && nsh > 1,
              why.empty() ? "status " + std::to_string(int(st)) : why);
        tail("8. writers leaving mid-chunk", 0, two);
    }

    // ---- item 8: a writer arriving during an idle drain, and a cancel during one ----
    for (int32_t arm = 0; arm < 2 && writer(); ++arm) {
        const std::string what = arm == 0 ? "8. a writer arriving during an idle drain" : "8. a cancel during a drain";
        if (!begin(0, two, 0)) {
            check(what, false, L.error());
            continue;
        }
        // the first chunk's front in one step beside the writer: its back half then drains (no step group pending)
        read_status st = L.read_step(0, 1, read_mode::writers, &big);
        bool        ok = true;
        while (ok && st == read_status::yielded) {
            ok = wstep();
            st = L.read_step(0, 1, read_mode::writers, &big);
        }
        ok = ok && st == read_status::chunk_done;
        // step groups arriving while the back's units are in flight (the joins are checked over the whole case)
        for (int32_t w = 0; w < 3 && ok && drained_past(eb.read_drained(), 8192); ++w) {
            ok = wstep();
        }
        if (arm == 1 && ok) {
            drained_past(eb.read_drained(), 8192);
            L.read_cancel(0);
        }
        int32_t calls = 0;
        st            = ok ? finish(0, front(0, read_mode::writers, &big, 1, true, calls)) : read_status::error;
        if (arm == 0) {
            check(what + ": the read", st == read_status::done, "status " + std::to_string(int(st)));
            tail(what, 0, two);
        } else {
            // the cancel took effect at the first chunk's end: 8,192 kept; the read resumed from there, lone
            const int64_t held = q.held(0);
            check(what + ": cancelled at the chunk's end, 8,192 kept", st == read_status::cancelled && held == 8192,
                  "status " + std::to_string(int(st)) + ", held " + std::to_string(held));
            int32_t c2 = 0;
            st         = begin(0, two, int32_t(held)) ? finish(0, front(0, read_mode::lone, nullptr, 1, false, c2)) :
                                                        read_status::error;
            check(what + ": the resumed read", st == read_status::done, "status " + std::to_string(int(st)));
            tail(what + ", resumed", 0, two);
        }
    }

    // ---- item 9: the front freed at its end; item 11: the crossing received ahead ----
    // two one-chunk reads back to back on slots 0 and 2: the second's front runs before the first's back answers
    const auto held_pair = [&](const std::string & what, const std::vector<int32_t> & first, int32_t cancel) {
        if (!writer()) {
            check(what, false, q.err);
            return;
        }
        const std::string log0 = F.test_log();
        F.test_hold_reads      = true;
        int32_t     c0 = 0, c2 = 0;
        read_status s0     = begin(0, first, 0) ? front(0, read_mode::writers, &big, 1, true, c0) : read_status::error;
        const bool  ready0 = L.read_ready(0);
        if (cancel == 0) {
            L.read_cancel(0);  // the first's back still runs: its front is done, nothing to stop
        }
        read_status s2 = read_status::error;
        if (begin(2, one_b, 0)) {
            if (cancel == 2) {
                L.read_cancel(2);
            }
            s2 = front(2, read_mode::writers, &big, 1, true, c2);
        }
        check(what + ": the first's front done while its back is held, the second's front then",
              s0 == read_status::front_done && !ready0 && (s2 == read_status::front_done),
              "statuses " + std::to_string(int(s0)) + " " + std::to_string(int(s2)));
        F.test_hold_reads     = false;
        s0                    = finish(0, s0);
        s2                    = finish(2, s2);
        const std::string log = F.test_log().substr(log0.size());
        const size_t      p0 = log.find("P0"), p2 = log.find("P2");
        check(what + ": the back ran them in order", p0 != std::string::npos && p2 != std::string::npos && p0 < p2,
              log);
        check(what + ": both ended",
              s0 == read_status::done && s2 == (cancel == 2 ? read_status::cancelled : read_status::done),
              "statuses " + std::to_string(int(s0)) + " " + std::to_string(int(s2)));
        if (&first == &two) {
            // item 11: the second chunk's crossing all received before the back took it (the front sent both first)
            check(what + ": the second chunk received ahead", xb.received_ahead(0) == 1,
                  std::to_string(xb.received_ahead(0)) + " of 1");
        }
        tail(what + ", the first", 0, first);
        if (cancel == 2) {
            int32_t c3 = 0;
            s2 = begin(2, one_b, 0) ? finish(2, front(2, read_mode::lone, nullptr, 1, false, c3)) : read_status::error;
            check(what + ": the second read again", s2 == read_status::done, std::to_string(int(s2)));
        }
        tail(what + ", the second", 2, one_b);
    };
    held_pair("9. two one-chunk reads back to back", one_a, -1);
    held_pair("9. a cancel of the second while the first finishes", one_a, 2);
    held_pair("9. a cancel of the first while its back finishes", one_a, 0);
    held_pair("9. a two-chunk read, then a one-chunk read (the buffers' parity, base + 257)", two, -1);

    // ---- item 14: a cancel during a read's last chunk ends it cancelled at that chunk's end (the chunk kept, no
    // handoff, so no token), on one box and on the split (the back ends it alike after the front's chunks); the
    // slot's state is its lone read's, and the slot is free for the next read. (A cancel while the back finishes is
    // item 9's case above: the engine's read is done, and the shell releases it: fnfl-splice-test's read-end rule)
    check("14. one box: a cancel during the last chunk: cancelled, 9,000 kept, no handoff",
          one14.sent && one14.status == read_status::cancelled && one14.held == 9000 && !one14.retained,
          "status " + std::to_string(int(one14.status)) + ", held " + std::to_string(one14.held) +
              (one14.retained ? ", handed off" : ""));
    check("14. one box: the slot's state is its lone read's", one14.state_ok);
    {
        const std::string d = one14.again ? piece_diff(ref_of(0, two), one14.out) : "the read again failed: " + err;
        check("14. one box: the slot free, the read again bitwise its lone read", one14.again && d.empty(), d);
    }
    {
        bool        sent = false;
        read_status st   = read_status::error;
        if (begin(0, two, 0)) {
            st = cancel_last([&]() { return L.read_step(0, 1, read_mode::shared); },
                             [&](read_info & ri) { ef.get_read_info(0, ri); }, [&]() { L.read_cancel(0); }, sent);
        }
        st               = finish(0, st);
        const int64_t hf = ef.held(0), hb = eb.held(0);
        const bool    ret = eb.test_target(0).slot().retained;
        check("14. the split: a cancel during the last chunk: cancelled, 9,000 kept on both boxes, no handoff",
              sent && st == read_status::cancelled && hf == 9000 && hb == 9000 && !ret,
              "status " + std::to_string(int(st)) + ", held " + std::to_string(hf) + "/" + std::to_string(hb) +
                  (ret ? ", handed off" : "") + (st == read_status::error ? ": " + L.error() : ""));
        reader        rd;
        snapshot      sn;
        const int32_t cf = int32_t(ef.test_target(0).slot().cur), cb = int32_t(eb.test_target(0).slot().cur);
        const bool    sok = cf == cb && state_of(ef.test_model(), 0, sn, rd, cf, ef.test_target(0).slot().gdn) &&
                            state_of(eb.test_model(), 0, sn, rd, cb, eb.test_target(0).slot().gdn) &&
                            sn.state == ref_of(0, two).snap.state;
        check("14. the split: the slot's state is its lone read's on both boxes", sok);
        int32_t c = 0;
        st        = L.release_slot(0) == run_status::ok && begin(0, two, 0) ?
                        finish(0, front(0, read_mode::lone, nullptr, 1, false, c)) :
                        read_status::error;
        check("14. the split: the slot free, the read again", st == read_status::done,
              "status " + std::to_string(int(st)));
        tail("14. the split: the read again", 0, two);
    }
    // a rollback after: slot 0 back to its 8,192 checkpoint, the rest read again (bitwise straight through)
    if (writer() && begin(0, two, 0)) {
        int32_t     c   = 0;
        read_status st  = finish(0, front(0, read_mode::writers, &big, 1, true, c));
        const bool  cut = st == read_status::done && q.truncate(0, 8192) == run_status::ok;
        st =
            cut && begin(0, two, 8192) ? finish(0, front(0, read_mode::writers, &big, 1, true, c)) : read_status::error;
        check("9. a rollback after: the read again from 8,192", st == read_status::done,
              "status " + std::to_string(int(st)));
        tail("9. a rollback after", 0, two);
    }
    q.release(1);
    sp.down();
    // item 8's ordering over the whole case: no read unit between a step group's arrival and its back half queued,
    // and at most two drain units in flight when a command came
    const std::string log   = F.test_log();
    bool              order = true;
    for (size_t i = log.find('C'); i != std::string::npos; i = log.find('C', i + 1)) {
        const size_t qd = log.find('Q', i);
        order           = order && qd != std::string::npos && log.find('L', i) > qd;
    }
    check("8. no read unit launched between a step group's arrival and its back half queued", order);
    check("8. at most two drain units in flight when a command arrived, drains joined at commands",
          F.test_in_flight_max <= 2 && F.test_settles > 0,
          std::to_string(F.test_in_flight_max.load()) + " in flight, " + std::to_string(F.test_settles.load()) +
              " joins");
    std::string lines;
    for (const std::string & b : bad) {
        lines += (lines.empty() ? "" : "; ") + b;
    }
    report = std::to_string(checks) + " checks" +
             (bad.empty() ? "" : ", " + std::to_string(bad.size()) + " failed: " + lines) + "; follower " +
             (sp.fbye ? "ended at X-BYE" : "lost: " + sp.fwhy);
    return bad.empty() && sp.fbye;
}

// ---- the pipe case: the server's path over the in-process port (stage::pipeline, the leader's reads) ----

// Two processes over the link, the real topology (each stage alone on its box's cards: both stages pipelined on one
// box's cards starve each other's cross-card polls): the leader (--role leader, the front) and the follower
// (--role follower, the back), on synthetic weights. Eight requests: four prompts read whole first, then four reads
// (one of 9,000 tokens: two chunks; MUP2: in 512-row pieces) in the writers geometry beside the pipelined groups
// (MUP2: budgeted units, one a turn on the front, the back's from its own cycle), each
// joining the writers at its end (groups of four and four); every request steps 12 times at T = 0. One box (the
// leader's, before its front stage loads) runs each request alone with the same read geometry. Joint rounds are off
// on both stages (each member's verify a lone round: a joint round's 16-bit exchanges are not a lone round's bits;
// the engine case and joint-kl cover them), so each request's tokens must be one box's exactly
constexpr int32_t pipe_slots = 8;
constexpr int32_t pipe_first = 4;  // the requests read whole before the pipeline starts
constexpr int32_t pipe_steps = 12;

std::vector<int32_t> pipe_prompt(int32_t s) {
    return tokens_of(950 + uint64_t(s), s == 5 ? 9000 : 120 + 31 * s);
}

request_options pipe_sampling() {
    request_options ro;
    ro.sampling       = fn::samp::options();
    ro.sampling.temp  = 0.0f;
    ro.sampling.top_k = 20;
    ro.max_tokens     = 512;
    return ro;
}

// the reference: each request alone on one box, its prompt read as the split reads it
bool pipe_reference(const options & o, bool eager, std::vector<std::vector<int32_t>> & toks, std::string & err) {
    md::window w;
    loaded     L;
    if (!load_synth(o, w, L)) {  // one box serves 1 .. 4 slots: each request alone on slot s mod 4
        err = "the one-box load";
        return false;
    }
    engine_config c = engine_cfg(o);
    c.eager         = eager;
    c.record_hash   = false;
    engine e;
    if (!e.init(std::move(L.m), c, err)) {
        return false;
    }
    toks.assign(pipe_slots, {});
    for (int32_t r = 0; r < pipe_slots; ++r) {
        const int32_t              s  = r % n_slot;
        const std::vector<int32_t> p  = pipe_prompt(r);
        int32_t                    np = 0;
        if (e.set_sampling(s, pipe_sampling()) != run_status::ok) {
            err = "one box: set_sampling: " + e.error();
            return false;
        }
        if (r < pipe_first) {
            if (e.prefill(s, p.data(), int32_t(p.size()), 0, np) != run_status::ok) {
                err = "one box: prefill: " + e.error();
                return false;
            }
        } else {
            if (e.read_begin(s, p.data(), int32_t(p.size()), 0, np) != 0) {
                err = "one box: read_begin: " + e.error();
                return false;
            }
            read_status st = read_status::yielded;
            while (st == read_status::yielded || st == read_status::chunk_done) {
                st = e.read_step(s, 1, read_mode::writers);
            }
            if (st != read_status::done) {
                err = "one box: read: " + e.error();
                return false;
            }
        }
        for (int32_t t = 0; t < pipe_steps; ++t) {
            int32_t out[5] = {}, n = 0;
            bool    end = false;
            if (e.step(s, out, n, end) != run_status::ok) {
                err = "one box: step: " + e.error();
                return false;
            }
            toks[size_t(r)].insert(toks[size_t(r)].end(), out, out + n);
        }
        e.release_slot(s);
    }
    return true;
}

// a stage over the link: its synthetic window, buffers, engine, link (X-HELLO), crossing and prefiller
struct pipe_stage {
    loaded                       l;
    stage::buffers               buf;
    stage::crossing              x;
    std::unique_ptr<engine>      e;
    fn::xchg::link::link         link;
    std::unique_ptr<stage::port> port;
};

bool pipe_stage_up(const options & o, bool eager, bool leader, pipe_stage & ps, std::string & err) {
    const int32_t k = 27;
    md::window    w;
    if (leader) {
        w.hi   = k - 1;
        w.head = false;
    } else {
        w.lo = k;
    }
    if (!load_synth(o, w, ps.l, pipe_slots)) {
        err = "the synthetic load of the " + std::string(leader ? "front" : "back");
        return false;
    }
    if (!ps.buf.alloc(pipe_slots, nullptr, err)) {
        return false;
    }
    engine_config c = engine_cfg(o);
    c.eager         = eager;
    c.stage         = leader ? 1 : 2;
    c.stage_buf     = ps.buf.slot;
    c.record_hash   = false;
    ps.e            = std::make_unique<engine>();
    if (!ps.e->init(std::move(ps.l.m), c, err)) {
        return false;
    }
    ps.e->split_joint_off();  // each member's rounds lone, as the reference's
    namespace lk = fn::xchg::link;
    lk::options lo;
    lo.leader    = leader;
    lo.addr      = o.addr;
    lo.port      = static_cast<uint16_t>(o.port);
    lo.timeout_s = 120.0;
    lo.connect_s = 1800.0;
    const std::string settings =
        stage::hello_settings(k, pipe_slots, slot_cells, 0, true, c.spec_d, c.draft.attention, c.eog);
    lk::set_text(lo.me.model, sizeof lo.me.model,
                 stage::hello_files(ps.e->test_model()) + "/" + stage::hello_hash(settings));
    lo.me.split = k;
    lo.me.slots = pipe_slots;
    printf("split-test pipe: the %s, %s %s:%d; the shared settings: %s\n", leader ? "leader" : "follower",
           leader ? "connecting to" : "listening on", o.addr.c_str(), o.port, settings.c_str());
    fflush(stdout);
    if (!ps.link.start(lo, err)) {
        return false;
    }
    printf("split-test pipe: connected; transport %s\n", ps.link.transport().c_str());
    fflush(stdout);
    ps.port = stage::link_port(ps.link);
    for (auto & sl : ps.buf.slot) {
        for (uint8_t * p : sl) {
            if (!ps.port->map(p, size_t(md::stage_bytes(fn::round::max_rows)), err)) {
                return false;
            }
        }
    }
    if (!ps.x.init(*ps.port, leader, k, err)) {
        return false;
    }
    ps.e->prefill_driver().set_stage(&ps.x);
    return true;
}

bool pipe_follower(const options & o, bool eager, std::string & report) {
    pipe_stage  ps;
    std::string err;
    if (!pipe_stage_up(o, eager, false, ps, err)) {
        report = err;
        return false;
    }
    stage::follower F;
    if (!F.init(*ps.e, *ps.port, ps.x, err)) {
        report = err;
        return false;
    }
    std::string                    why;
    const bool                     bye = F.serve(why);
    const fn::xchg::link::counters t   = ps.link.totals();
    report = std::string(bye ? "the leader said X-BYE" : "the leader was lost: " + why) + "; link: sent " +
             std::to_string(t.sent_msgs) + " messages, received " + std::to_string(t.recv_msgs) +
             "; prompt crossings " + std::to_string(ps.x.count());
    return bye;
}

bool pipe_case(const options & o, bool eager, std::string & report) {
    if (o.role == 2) {
        return pipe_follower(o, eager, report);
    }
    if (o.role != 1) {
        report =
            "the pipe case runs as two processes, --role leader on the front's box and --role follower on the "
            "back's";
        return false;
    }
    std::vector<std::vector<int32_t>> ref;
    std::string                       err;
    if (!pipe_reference(o, eager, ref, err)) {
        report = err;
        return false;
    }
    pipe_stage ps;
    if (!pipe_stage_up(o, eager, true, ps, err)) {
        report = err;
        return false;
    }
    engine &          ef = *ps.e;
    stage::crossing & xf = ps.x;
    stage::leader     L;
    if (!L.init(ef, *ps.port, xf, err)) {
        report = "the leader: " + err;
        return false;
    }
    stage::pipeline                   P(L);
    std::vector<std::vector<int32_t>> got(pipe_slots);
    std::vector<int32_t>              steps(pipe_slots, 0);
    std::vector<int32_t>              writers, readers;
    bool                              ok = true;
    std::string                       why;
    int32_t                           turns = 0, read_groups = 0, overlap_turns = 0;
    for (int32_t s = 0; s < pipe_slots && ok; ++s) {
        const std::vector<int32_t> p  = pipe_prompt(s);
        int32_t                    np = 0;
        ok                            = L.set_sampling(s, pipe_sampling()) == run_status::ok;
        if (ok && s < pipe_first) {
            ok = L.prefill(s, p.data(), int32_t(p.size()), 0, np) == run_status::ok;
            writers.push_back(s);
        }
        why = ok ? "" : "setup of slot " + std::to_string(s) + ": " + L.error();
    }
    int32_t next_reader = pipe_first;
    while (ok && (!writers.empty() || !readers.empty() || next_reader < pipe_slots)) {
        // a new read every third turn
        if (next_reader < pipe_slots && (turns % 3 == 2 || (writers.empty() && readers.empty()))) {
            const std::vector<int32_t> p  = pipe_prompt(next_reader);
            int32_t                    np = 0;
            if (L.read_begin(next_reader, p.data(), int32_t(p.size()), 0, np) != 0) {
                ok  = false;
                why = "read_begin of slot " + std::to_string(next_reader) + ": " + L.error();
                break;
            }
            readers.push_back(next_reader++);
        }
        // one pipeline turn over the writers (none: the read's back chunks finish on the follower meanwhile)
        if (writers.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        } else {
            ++turns;
            joint_step_out res[pipe_slots];
            bool           done[pipe_slots] = {};
            P.step(writers.data(), int32_t(writers.size()), res, done);
            for (int32_t g = 0, inflight = 0; g < int32_t(writers.size()); ++g) {
                inflight += P.in_flight(writers[size_t(g)]) ? 1 : 0;
                if (g + 1 == int32_t(writers.size()) && inflight > 0 && inflight < int32_t(writers.size())) {
                    ++overlap_turns;
                }
            }
            std::vector<int32_t> keep;
            for (size_t q = 0; q < writers.size(); ++q) {
                const int32_t s = writers[q];
                if (done[q]) {
                    if (res[q].st != run_status::ok) {
                        ok  = false;
                        why = "the step of slot " + std::to_string(s) + ": " + res[q].why;
                        break;
                    }
                    got[size_t(s)].insert(got[size_t(s)].end(), res[q].out, res[q].out + res[q].n_out);
                    ++steps[size_t(s)];
                }
                if (steps[size_t(s)] < pipe_steps) {
                    keep.push_back(s);
                } else {
                    P.release_slot(s);
                }
            }
            writers = keep;
        }
        // one layer group of the first read (the shell's slice beside writers; on the split one read runs at a time,
        // in the main lane: the glue offers no other-reader lane)
        std::vector<int32_t> rkeep;
        for (size_t q = 0; q < readers.size(); ++q) {
            const int32_t s = readers[q];
            if (q > 0) {
                rkeep.push_back(s);
                continue;
            }
            // the writers geometry throughout (the reference's), writers present or not. MUP2: the server's path, a
            // budget (unknown on the front: one unit a step; the follower budgets its own units), the 9,000-token read
            // in 512-row pieces (the partition crosses with each chunk)
            read_budget rb;
            ps.e->test_piece_rows(s == 5 ? 512 : 0);
            const read_status st = L.read_step(s, 1, read_mode::writers, &rb);
            ps.e->test_piece_rows(0);
            ++read_groups;
            if (st == read_status::done) {
                writers.push_back(s);
            } else if (st == read_status::yielded || st == read_status::chunk_done || st == read_status::front_done) {
                rkeep.push_back(s);  // PRD1 item 9: a finishing read is polled through read_step to its end
            } else {
                ok  = false;
                why = "the read of slot " + std::to_string(s) + ": " + L.error();
            }
        }
        readers = rkeep;
    }
    P.settle_all();
    L.bye();
    if (!ok) {
        report = "the split: " + why;
        return false;
    }
    int32_t     bad = 0;
    std::string first;
    for (int32_t s = 0; s < pipe_slots; ++s) {
        if (got[size_t(s)] != ref[size_t(s)]) {
            ++bad;
            if (first.empty()) {
                first = "; slot " + std::to_string(s) + " differs first";
            }
        }
    }
    const stage::leader_stats & ls = L.stats();
    report = std::to_string(pipe_slots) + " requests of " + std::to_string(pipe_steps) +
             " steps (four reads beside the writers, one of two chunks): " +
             (bad == 0 ? "every request's T = 0 tokens one box's" : std::to_string(bad) + " differ" + first) + "; " +
             std::to_string(turns) + " pipeline turns, " + std::to_string(overlap_turns) +
             " leaving a group in flight; " + std::to_string(read_groups) + " front read calls; " +
             std::to_string(ls.groups) + " step groups, " + std::to_string(xf.count()) + " prompt crossings; " +
             (L.lost() ? "the follower was lost: " + L.lost_why() : "X-BYE answered");
    return bad == 0 && !L.lost();
}

// ---- the deliver case (TTTS8): every finished step reaches its request once, in order ----

// The server's split calls (stage::pipeline's, which fn/glue's split branches forward to) over the link, as the pipe
// case runs them (the leader here, the follower on the back's box), with the shell's bookkeeping of each slot as
// server-context.cpp keeps it: apply_step's tokens (the slot holds a step's tokens but its last, which it keeps as
// sampled) and --fnfl-check-state's comparison with the front engine's history after every delivered step. Each case
// drives one transition from the pipeline to the lone call (fnfl_step: stage::pipeline::step_one) while the remaining
// writer's step is in flight, then steps it alone; and the truncates inside a request's last step that a stop string
// and a cancel leave (the next turn's read on both boxes from the same point)
struct deliver_shell {
    struct slot_view {
        std::vector<int32_t> toks;             // the slot's tokens (server_slot::prompt.tokens)
        int32_t              sampled = -1;     // the last step's last token (server_slot::sampled)
        bool                 gen     = false;
        bool                 reading = false;  // a read begun and not ended
        size_t               before  = 0;      // the last step's: the slot's tokens before it, and its token count
        int32_t              n_last  = 0;
    };

    engine &               ef;
    stage::leader &        L;
    stage::pipeline &      P;
    std::vector<slot_view> v;
    int32_t                delivered = 0, checks = 0;
    std::string            bad;  // the first failure

    deliver_shell(engine & e, stage::leader & l, stage::pipeline & p, int32_t n) : ef(e), L(l), P(p), v(size_t(n)) {}

    bool fail(const std::string & why) {
        if (bad.empty()) {
            bad = why;
        }
        return false;
    }

    // --fnfl-check-state (fnfl_slot_tokens: the slot's group settled, then the front engine's history)
    bool check(int32_t s, const char * after) {
        P.settle(s);
        std::vector<int32_t> hist;
        int32_t              pending = -1;
        if (ef.history(s, hist, pending) != run_status::ok) {
            return fail("slot " + std::to_string(s) + ": history after " + after);
        }
        ++checks;
        size_t i = 0;
        while (i < hist.size() && i < v[size_t(s)].toks.size() && hist[i] == v[size_t(s)].toks[i]) {
            ++i;
        }
        if (i != hist.size() || i != v[size_t(s)].toks.size() ||
            (v[size_t(s)].sampled >= 0 && pending != v[size_t(s)].sampled)) {
            return fail("slot " + std::to_string(s) + ": state check MISMATCH after " + after + ": engine " +
                        std::to_string(hist.size()) + " tokens, the shell " + std::to_string(v[size_t(s)].toks.size()) +
                        ", first difference at " + std::to_string(i) + "; pending engine " + std::to_string(pending) +
                        " shell " + std::to_string(v[size_t(s)].sampled));
        }
        return true;
    }

    // the request ends (the shell's release: callback_on_release, fnfl_release_slot)
    bool end(int32_t s) {
        v[size_t(s)].gen = false;
        return P.release_slot(s) == run_status::ok || fail("release of slot " + std::to_string(s) + ": " + L.error());
    }

    // apply_step: a step's tokens into the slot, then the state check; the request ends at its end of generation
    bool apply(int32_t s, const joint_step_out & r, const char * after) {
        if (r.st != run_status::ok || r.n_out < 1) {
            return fail("slot " + std::to_string(s) + ": step after " + after + ": " +
                        (r.st != run_status::ok ? r.why : std::string("no tokens")));
        }
        slot_view & x = v[size_t(s)];
        x.before      = x.toks.size();
        x.n_last      = r.n_out;
        if (x.sampled >= 0) {
            x.toks.push_back(x.sampled);
        }
        x.toks.insert(x.toks.end(), r.out, r.out + r.n_out - 1);
        x.sampled = r.out[r.n_out - 1];
        ++delivered;
        if (!check(s, after)) {
            return false;
        }
        return !r.end || end(s);
    }

    // a read's calls until it ends (done, cancelled or an error); a hang guard of 600 s, not a budget
    read_status drain(int32_t s, read_mode mode) {
        const auto  t0 = std::chrono::steady_clock::now();
        read_status st = read_status::yielded;
        while ((st == read_status::yielded || st == read_status::chunk_done || st == read_status::front_done) &&
               std::chrono::steady_clock::now() - t0 < std::chrono::seconds(600)) {
            st = L.read_step(s, 1, mode);
        }
        v[size_t(s)].reading =
            st == read_status::yielded || st == read_status::chunk_done || st == read_status::front_done;
        return st;
    }

    // a read to its end with no writer stepping (the lone geometry)
    bool read_all(int32_t s) {
        return drain(s, read_mode::lone) == read_status::done ||
               fail("the read of slot " + std::to_string(s) + ": " + L.error());
    }

    // between cases: a read left in progress is cancelled and ended, then every slot is released
    void reset_all() {
        for (int32_t s = 0; s < int32_t(v.size()); ++s) {
            if (v[size_t(s)].reading) {
                L.read_cancel(s);
                drain(s, read_mode::lone);
            }
            P.release_slot(s);
            v[size_t(s)].gen = false;
        }
    }

    // a request's start as start_prompt and tu_admit make it: its sampling, the cached prefix kept (fnfl_truncate),
    // the rest read (fnfl_read_begin); whole now when read is true (else the caller steps the read)
    bool start(int32_t s, const std::vector<int32_t> & prompt, bool read) {
        slot_view & x = v[size_t(s)];
        if (P.set_sampling(s, pipe_sampling()) != run_status::ok) {
            return fail("set_sampling of slot " + std::to_string(s) + ": " + L.error());
        }
        int32_t n_past = 0;
        while (n_past < int32_t(x.toks.size()) && n_past < int32_t(prompt.size()) &&
               x.toks[size_t(n_past)] == prompt[size_t(n_past)]) {
            ++n_past;
        }
        if (n_past == int32_t(prompt.size())) {
            --n_past;
        }
        if (P.truncate(s, n_past) != run_status::ok) {
            return fail("truncate of slot " + std::to_string(s) + " to " + std::to_string(n_past) + ": " + L.error());
        }
        x.toks.resize(size_t(n_past));
        x.sampled = -1;
        if (!check(s, "truncate")) {
            return false;
        }
        int32_t np = 0;
        if (P.read_begin(s, prompt.data() + n_past, int32_t(prompt.size()) - n_past, n_past, np) != 0) {
            return fail("read_begin of slot " + std::to_string(s) + " (kept " + std::to_string(n_past) +
                        "): " + L.error());
        }
        x.toks    = prompt;
        x.reading = true;
        if (read && !read_all(s)) {
            return false;
        }
        x.gen = read;
        return !read || check(s, "prefill");
    }

    // one tu_decode on the split (fnfl_step_pipe): each finished step applied
    bool turn(const std::vector<int32_t> & w, std::vector<bool> * done_out = nullptr) {
        joint_step_out res[pipe_slots];
        bool           done[pipe_slots] = {};
        P.step(w.data(), int32_t(w.size()), res, done);
        if (done_out != nullptr) {
            done_out->assign(done, done + w.size());
        }
        for (size_t k = 0; k < w.size(); ++k) {
            if (done[k] && !apply(w[k], res[k], "a pipeline turn")) {
                return false;
            }
        }
        return true;
    }

    // the lone writer's step (fnfl_step)
    bool lone(int32_t s, int32_t times) {
        for (int32_t t = 0; t < times && v[size_t(s)].gen; ++t) {
            joint_step_out r;
            P.step_one(s, r);
            if (!apply(s, r, "the lone call")) {
                return false;
            }
        }
        return true;
    }

    // pipeline turns over w until the turn that finishes ender's step leaves stay's step in flight (at least three
    // turns first); false when that never happened within 64 turns
    bool until_in_flight(const std::vector<int32_t> & w, int32_t stay, int32_t ender) {
        size_t ie = 0;
        while (ie < w.size() && w[ie] != ender) {
            ++ie;
        }
        for (int32_t t = 0; t < 64; ++t) {
            std::vector<bool> done;
            if (!turn(w, &done)) {
                return false;
            }
            for (int32_t s : w) {
                if (!v[size_t(s)].gen) {
                    return fail("slot " + std::to_string(s) + " ended (end of generation) before the transition");
                }
            }
            if (t >= 3 && ie < done.size() && done[ie] && P.in_flight(stay)) {
                return true;
            }
        }
        return fail("slot " + std::to_string(stay) + "'s step was never in flight at a turn that finished slot " +
                    std::to_string(ender) + "'s");
    }
};

std::vector<int32_t> deliver_prompt(int32_t s, int32_t c, int32_t n) {
    return tokens_of(970 + 16 * uint64_t(c) + uint64_t(s), n);
}

// one case: its name, PASS or FAIL with the shell's first failure; every slot it used released after it
bool deliver_one(deliver_shell & sh, const std::string & name, const std::function<bool()> & body, std::string & out) {
    sh.bad.clear();
    const int32_t d0 = sh.delivered, c0 = sh.checks;
    const bool    ok = body() && sh.bad.empty();
    sh.reset_all();
    out += "split-test deliver: " + name + ": " + std::to_string(sh.delivered - d0) + " steps delivered, " +
           std::to_string(sh.checks - c0) + " state checks: " + (ok ? "PASS" : "FAIL (" + sh.bad + ")") + "\n";
    return ok;
}

bool deliver_case(const options & o, bool eager, std::string & report) {
    if (o.role == 2) {
        return pipe_follower(o, eager, report);
    }
    if (o.role != 1) {
        report =
            "the deliver case runs as two processes, --role leader on the front's box and --role follower on the "
            "back's";
        return false;
    }
    pipe_stage  ps;
    std::string err;
    if (!pipe_stage_up(o, eager, true, ps, err)) {
        report = err;
        return false;
    }
    stage::leader L;
    if (!L.init(*ps.e, *ps.port, ps.x, err)) {
        report = "the leader: " + err;
        return false;
    }
    stage::pipeline P(L);
    deliver_shell   sh(*ps.e, L, P, pipe_slots);
    std::string     out;
    int32_t         failed = 0;
    const auto      run    = [&](const std::string & name, const std::function<bool()> & body) {
        failed += deliver_one(sh, name, body, out) ? 0 : 1;
        printf("%s", out.c_str());
        fflush(stdout);
        report += out;
        out.clear();
    };
    // the other writer ends (its last step delivered in the turn that leaves this writer's in flight)
    run("another writer ends", [&]() {
        return sh.start(0, deliver_prompt(0, 0, 150), true) && sh.start(1, deliver_prompt(1, 0, 181), true) &&
               sh.until_in_flight({ 0, 1 }, 0, 1) && sh.end(1) && sh.lone(0, 6);
    });
    // a reader is cancelled (its read stops at its chunk's end and the slot is released), with the other writer's end
    run("a reader is cancelled", [&]() {
        if (!sh.start(0, deliver_prompt(0, 1, 140), true) || !sh.start(1, deliver_prompt(1, 1, 171), true) ||
            !sh.start(2, deliver_prompt(2, 1, 9000), false)) {
            return false;
        }
        for (int32_t t = 0; t < 8; ++t) {  // the read beside the writers, a layer group a turn
            if (!sh.turn({ 0, 1 }) || L.read_step(2, 1, read_mode::writers) == read_status::error) {
                return sh.fail("the read beside the writers: " + L.error());
            }
        }
        if (!sh.until_in_flight({ 0, 1 }, 0, 1) || !sh.end(1)) {
            return false;
        }
        L.read_cancel(2);
        const read_status st = sh.drain(2, read_mode::writers);
        if (st != read_status::cancelled && st != read_status::done) {
            return sh.fail("the cancelled read of slot 2: " + L.error());
        }
        return sh.end(2) && sh.lone(0, 6);
    });
    // a reader's job ends: its read finishes and its request ends before its first step
    run("a reader's job ends", [&]() {
        if (!sh.start(0, deliver_prompt(0, 2, 133), true) || !sh.start(1, deliver_prompt(1, 2, 164), true) ||
            !sh.start(2, deliver_prompt(2, 2, 300), false)) {
            return false;
        }
        if (!sh.until_in_flight({ 0, 1 }, 0, 1) || !sh.end(1)) {
            return false;
        }
        if (sh.drain(2, read_mode::writers) != read_status::done) {
            return sh.fail("the read of slot 2: " + L.error());
        }
        return sh.end(2) && sh.lone(0, 6);
    });
    // a started request is refused (its prompt past the slot's cells: read_begin refuses it on admission)
    run("a started request is refused", [&]() {
        if (!sh.start(0, deliver_prompt(0, 3, 129), true) || !sh.start(1, deliver_prompt(1, 3, 158), true) ||
            !sh.until_in_flight({ 0, 1 }, 0, 1) || !sh.end(1)) {
            return false;
        }
        const std::vector<int32_t> big = deliver_prompt(3, 3, slot_cells + 100);
        int32_t                    np  = 0;
        if (P.set_sampling(3, pipe_sampling()) != run_status::ok || P.truncate(3, 0) != run_status::ok ||
            P.read_begin(3, big.data(), int32_t(big.size()), 0, np) == 0) {
            return sh.fail("the oversized request was not refused");
        }
        return sh.end(3) && sh.lone(0, 6);
    });
    // co-members: slots 0 and 2 in one group (three writers: groups of two and one); slot 1 ends, slot 2 is
    // cancelled while their group is in flight, and slot 0 steps alone (its step kept by slot 2's settle)
    run("a co-member is cancelled", [&]() {
        if (!sh.start(0, deliver_prompt(0, 4, 137), true) || !sh.start(1, deliver_prompt(1, 4, 166), true) ||
            !sh.start(2, deliver_prompt(2, 4, 195), true) || !sh.until_in_flight({ 0, 1, 2 }, 0, 1)) {
            return false;
        }
        if (!P.in_flight(2)) {
            return sh.fail("slots 0 and 2 were not one group in flight");
        }
        return sh.end(1) && sh.end(2) && sh.lone(0, 6);
    });
    // a cancel while the writer's step is in flight: the request ends without that step (dropped with its request);
    // the next turn continues the history the client received, so the engine cuts inside that step
    run("a cancel mid-step, then the next turn", [&]() {
        const std::vector<int32_t> p = deliver_prompt(0, 6, 9000);
        if (!sh.start(0, p, true) || !sh.start(1, deliver_prompt(1, 6, 177), true) ||
            !sh.until_in_flight({ 0, 1 }, 0, 1) || !sh.end(0)) {
            return false;
        }
        std::vector<int32_t> next = sh.v[0].toks;
        if (sh.v[0].sampled >= 0) {
            next.push_back(sh.v[0].sampled);
        }
        const std::vector<int32_t> tail = tokens_of(991, 40);
        next.insert(next.end(), tail.begin(), tail.end());
        return sh.start(0, next, true) && sh.lone(0, 4);
    });
    P.settle_all();
    L.bye();
    report += std::to_string(failed) + " of 6 cases failed; " +
              (L.lost() ? "the follower was lost: " + L.lost_why() : "X-BYE answered");
    return failed == 0 && !L.lost();
}

// ---- the truncate case (TTTS8): a cut inside the request's last step that fails on the follower ----

// In one process over the in-process port (synchronous steps: both stages may share this box's cards), with the
// server's split calls and the shell's bookkeeping as the deliver case. The follower's cut of slot 0 inside its last
// step fails, as deep64's did (its draft catch-up refused; split-test injects that outcome: follower::test_fail_cut),
// after the two ways a client's history ends inside the engine's last step: a stop string that the step's fed token
// began and its first new token completed (the fed token's text held back as a partial stop), and a cancel of a step
// the engine finished and the client never got. The next turn's read must start at the same point on both boxes
// (read_begin's count check), then every step matches the shell's tokens. The prompt is past a checkpoint (8,192)
bool truncate_case(const options & o, bool eager, std::string & report) {
    const int32_t k = 27;
    md::window    wf, wb;
    wf.hi   = k - 1;
    wf.head = false;
    wb.lo   = k;
    loaded LF, LB;
    if (!load_synth(o, wf, LF) || !load_synth(o, wb, LB)) {
        report = "the split's loads";
        return false;
    }
    std::unique_ptr<stage::port> pa, pb;
    stage::loop_pair(pa, pb);
    stage::buffers  bf, bb;
    stage::crossing xf, xb;
    std::string     err;
    if (!bf.alloc(n_slot, pa.get(), err) || !bb.alloc(n_slot, pb.get(), err) || !xf.init(*pa, true, k, err) ||
        !xb.init(*pb, false, k, err)) {
        report = "the split's buffers: " + err;
        return false;
    }
    engine_config cf = engine_cfg(o), cb = engine_cfg(o);
    cf.eager = cb.eager = eager;
    cf.stage            = 1;
    cf.stage_buf        = bf.slot;
    cf.record_hash      = false;
    cb.stage            = 2;
    cb.stage_buf        = bb.slot;
    engine ef, eb;
    if (!ef.init(std::move(LF.m), cf, err) || !eb.init(std::move(LB.m), cb, err)) {
        report = "the split's engines: " + err;
        return false;
    }
    ef.prefill_driver().set_stage(&xf);
    eb.prefill_driver().set_stage(&xb);
    stage::leader   L;
    stage::follower F;
    if (!L.init(ef, *pa, xf, err) || !F.init(eb, *pb, xb, err)) {
        report = "the split's stages: " + err;
        return false;
    }
    std::string     fwhy;
    bool            fbye = false;
    std::thread     fth([&]() { fbye = F.serve(fwhy); });
    stage::pipeline P(L);
    deliver_shell   sh(ef, L, P, n_slot);
    int32_t         failed = 0;
    std::string     out;
    const auto      run = [&](const std::string & name, const std::function<bool()> & body) {
        failed += deliver_one(sh, name, body, out) ? 0 : 1;
    };
    run("truncate: a stop across two steps, the follower's cut fails", [&]() {
        const std::vector<int32_t> p = deliver_prompt(0, 7, 9000);
        if (!sh.start(0, p, true) || !sh.lone(0, 4)) {
            return false;
        }
        std::vector<int32_t>       next(sh.v[0].toks.begin(), sh.v[0].toks.begin() + int64_t(sh.v[0].before));
        const std::vector<int32_t> tail = tokens_of(992, 40);
        next.insert(next.end(), tail.begin(), tail.end());
        F.test_fail_cut.store(0);
        return sh.end(0) && sh.start(0, next, true) && sh.lone(0, 4);
    });
    run("truncate: a cancel of a finished step, the follower's cut fails", [&]() {
        const std::vector<int32_t> p = deliver_prompt(0, 8, 9000);
        if (!sh.start(0, p, true) || !sh.lone(0, 4)) {
            return false;
        }
        joint_step_out r;
        P.step_one(0, r);  // the cancelled request's step: finished in the engine, never applied by the shell
        if (r.st != run_status::ok) {
            return sh.fail("the cancelled step: " + r.why);
        }
        std::vector<int32_t> next = sh.v[0].toks;
        next.push_back(sh.v[0].sampled);
        const std::vector<int32_t> tail = tokens_of(993, 40);
        next.insert(next.end(), tail.begin(), tail.end());
        F.test_fail_cut.store(0);
        return sh.end(0) && sh.start(0, next, true) && sh.lone(0, 4);
    });
    P.settle_all();
    L.bye();
    fth.join();
    printf("%s", out.c_str());
    report =
        out + std::to_string(failed) + " of 2 cases failed; follower " + (fbye ? "ended at X-BYE" : "lost: " + fwhy);
    return failed == 0 && fbye;
}

// ---- APF1 Part 4: the ladder's coverage (structured reads through the real callers) ----

// the process's executed-variant counts (every prompt path of every engine here: md::prompt_route_totals) less a mark
struct route_mark {
    md::prompt_path::route_counts c;
    uint32_t                      rungs = 0;

    route_mark() { md::prompt_route_totals(c, rungs, true); }

    md::prompt_path::route_counts since() const {
        md::prompt_path::route_counts n;
        uint32_t                      r = 0;
        md::prompt_route_totals(n, r);
        md::prompt_path::route_counts d;
        d.add_since(n, c);
        n = d;
        return n;
    }
};

std::string routes_text(const md::prompt_path::route_counts & r) {
    return r.text();
}

// the tensor inventory's name from the GGUF path (the coverage table's format column)
std::string inventory_of(const options & o) {
    return o.model_path.find("Q4_K") != std::string::npos ? "UD-Q4_K_XL" :
           o.model_path.find("IQ3") != std::string::npos  ? "IQ3_XXS" :
                                                            o.model_path;
}

// a box the ladder's reads run on: one engine (resident TP), or the split's front, back and leader
struct lad_box {
    std::string     name;
    engine *        e    = nullptr;
    engine *        back = nullptr;
    stage::leader * L    = nullptr;
    bool            kq   = false;  // a window layer's routed gate/up is Q4_K or Q5_K (the grouped route's formats)

    void force(const md::prompt_route_force & f) {
        e->prefill_driver().test_force_route(f);
        if (back != nullptr) {
            back->prefill_driver().test_force_route(f);
        }
    }

    bool skew(bool on, std::string & err) {
        return e->prefill_driver().test_skew_experts(on, err) &&
               (back == nullptr || back->prefill_driver().test_skew_experts(on, err));
    }
};

bool kq_layers(engine & e) {
    const md::model & m = e.test_model();
    for (int32_t l = m.win.lo; l <= m.win.hi; ++l) {
        const auto f = static_cast<fn::experts::fmt>(e.test_target(0).card(0).w[l].ex_gu_fmt);
        if (f == fn::experts::fmt::q4_k || f == fn::experts::fmt::q5_k) {
            return true;
        }
    }
    return false;
}

// the ladder's reads, each from slot 0's fresh state (reuse > 0: its prefix read first): the read under the retained
// routes (every route forced off: pd_convert + pd_tc, hct_convert + hct_mix, per-task experts, eager units), then the
// ladder arm twice. kind 0: lone (the request's prefill); 1: beside a writer on slot 1 (read_begin, writers-mode steps
// with an unknown budget, a writer step between every two); 2: as 1, cancelled after two steps, the slot released,
// then read again lone. skew: synthetic skewed expert assignment in every arm (the grouped route reached
// predictably). force: the ladder arm's forced routes (md::prompt_route_force: 0 the table's choice, 1 the new routes,
// 2 retained). APF2: split K sums in another order, so the ladder arm is checked against itself (the rerun bitwise)
// and its routes, its difference from the retained arm reported (the KL line is joint-kl --route-kl's)
struct lad_read {
    const char *           name;
    uint64_t               seed;
    int32_t                n, reuse, kind;
    bool                   skew;
    md::prompt_route_force force;
};

constexpr md::prompt_route_force lad_new{ 1, 1, 1, 0 };

const lad_read lad_reads[] = {
    { "fresh 64 (the table's routes)",              9601, 64,   0,    0, false, {}             },
    { "fresh 65 (a partial 64-row tile)",           9602, 65,   0,    0, false, lad_new        },
    { "fresh 300",                                  9603, 300,  0,    0, false, lad_new        },
    { "fresh 512",                                  9604, 512,  0,    0, false, lad_new        },
    { "fresh 513",                                  9605, 513,  0,    0, false, lad_new        },
    { "fresh 2,048 (the table's routes)",           9606, 2048, 0,    0, false, {}             },
    { "fresh 2,049",                                9607, 2049, 0,    0, false, lad_new        },
    { "64 at cell 3,000",                           9608, 3064, 3000, 0, false, lad_new        },
    { "256 at cell 3,000 (the table's routes)",     9609, 3256, 3000, 0, false, {}             },
    { "fresh 1,024, skewed experts (synthetic)",    9610, 1024, 0,    0, true,  lad_new        },
    { "fresh 64 again (the graphs updated)",        9601, 64,   0,    0, false, lad_new        },
    { "fresh 64, graph off",                        9601, 64,   0,    0, false, { 1, 1, 1, 2 } },
    { "300 beside a writer",                        9611, 300,  0,    1, false, lad_new        },
    { "700 beside a writer, in units",              9612, 700,  0,    1, false, lad_new        },
    { "700 beside a writer, cancelled, read again", 9612, 700,  0,    2, false, lad_new        },
};

bool lad_run(lad_box &                       b,
             const lad_read &                r,
             bool                            ladder,
             piece_out &                     out,
             std::string &                   err,
             md::prompt_path::route_counts * routes = nullptr) {
    std::unique_ptr<route_mark>  mk;  // set just before the read itself (after its prefix and the writer's prompt)
    const std::vector<int32_t>   all = tokens_of(r.seed, r.n);
    const std::vector<int32_t>   wp  = tokens_of(9690, 150);
    const md::prompt_route_force retained{ 2, 2, 2, 2 };
    b.force(ladder ? r.force : retained);
    if (!b.skew(r.skew, err)) {
        return false;
    }
    requests q;
    q.one           = b.L == nullptr ? b.e : nullptr;
    q.front         = b.e;
    q.lead          = b.L;
    const int32_t s = 0;
    if (q.sampling(s, 0.0f, 0, 512) != run_status::ok ||
        (r.reuse > 0 && q.prefill(s, std::vector<int32_t>(all.begin(), all.begin() + r.reuse), 0) != run_status::ok)) {
        err = q.err;
        return false;
    }
    if (!ladder || r.kind == 0) {
        mk = std::make_unique<route_mark>();
        if (q.prefill(s, all, r.reuse) != run_status::ok) {
            err = q.err;
            return false;
        }
    } else {
        // the writer on slot 1, then slot 0's read in writers-mode steps with the writer's steps between
        q.release(1);
        if (q.sampling(1, 0.0f, 0, 512) != run_status::ok || q.prefill(1, wp, 0) != run_status::ok) {
            err = q.err;
            return false;
        }
        int32_t np       = 0;
        mk               = std::make_unique<route_mark>();
        const int32_t rb = b.L != nullptr ? b.L->read_begin(s, all.data() + r.reuse, r.n - r.reuse, r.reuse, np) :
                                            b.e->read_begin(s, all.data() + r.reuse, r.n - r.reuse, r.reuse, np);
        if (rb != 0) {
            err = "read_begin: " + (b.L != nullptr ? b.L->error() : b.e->error());
            return false;
        }
        read_budget bu;  // unknown: one unit a step
        read_status st = read_status::yielded;
        for (int32_t calls = 0; st == read_status::yielded || st == read_status::chunk_done; ++calls) {
            if (r.kind == 2 && calls == 2) {
                if (b.L != nullptr) {
                    b.L->read_cancel(s);
                } else {
                    b.e->read_cancel(s);
                }
            }
            st = b.L != nullptr ? b.L->read_step(s, 1, read_mode::writers, &bu) :
                                  b.e->read_step(s, 1, read_mode::writers, &bu);
            if ((st == read_status::yielded || st == read_status::chunk_done) && !q.step({ 1 }, 1, false)) {
                err = q.err;
                return false;
            }
        }
        for (const auto t0 = std::chrono::steady_clock::now();
             st == read_status::front_done && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(600);) {
            if (b.L->read_ready(s)) {
                st = b.L->read_poll(s);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (q.release(1) != run_status::ok) {
            err = "release the writer";
            return false;
        }
        if (r.kind == 2) {
            // cancelled: the slot released and read again lone (the slot's reuse)
            if (st != read_status::cancelled && st != read_status::done) {
                err = "the cancelled read ended " + std::to_string(int(st));
                return false;
            }
            if (q.release(s) != run_status::ok || q.sampling(s, 0.0f, 0, 512) != run_status::ok ||
                q.prefill(s, all, 0) != run_status::ok) {
                err = "the read again: " + q.err;
                return false;
            }
        } else if (st != read_status::done) {
            err = "read: " + (b.L != nullptr ? b.L->error() : b.e->error());
            return false;
        }
    }
    if (routes != nullptr) {
        *routes = mk->since();
    }
    const bool ok = piece_tail(*b.e, b.back, b.L, q, s, out, err);
    b.force({});
    std::string e2;
    return b.skew(false, e2) && ok;
}

// the ladder's checks on one box: each read's state against the retained routes', and the routes it ran against the
// selection for its rows (prompt_route_for); a table row each
// both cards' free bytes (cudaMemGetInfo)
void free_bytes(const md::model & m, int64_t (&out)[2]) {
    for (int32_t c = 0; c < 2; ++c) {
        size_t fr = 0, tot = 0;
        out[c] =
            cudaSetDevice(m.device[c]) == cudaSuccess && cudaMemGetInfo(&fr, &tot) == cudaSuccess ? int64_t(fr) : -1;
    }
}

bool lad_box_reads(lad_box & b, const std::string & inv, std::vector<std::string> & table, int32_t & bad) {
    int64_t f0[2], f1[2];
    free_bytes(b.e->test_model(), f0);
    const auto field = [](int32_t f) {
        return f == 1 ? "new" : f == 2 ? "retained" : "table";
    };
    for (const lad_read & r : lad_reads) {
        piece_out   ref, x, x2;
        std::string err;
        if (!lad_run(b, r, false, ref, err)) {
            table.push_back("| " + b.name + " | " + r.name + " | retained arm failed: " + err + " | | | | FAIL |");
            ++bad;
            continue;
        }
        md::prompt_path::route_counts rc, rc2;
        const bool                    ran  = lad_run(b, r, true, x, err, &rc) && lad_run(b, r, true, x2, err, &rc2);
        const int32_t                 rows = r.n - r.reuse;
        const md::prompt_route_force  f    = r.force;
        const md::prompt_route        e    = md::prompt_route_for(rows, true, f);
        // the executed variants against the forced fields (the table's choice: not asserted; kind 2's counts include
        // its cancelled read's chunk)
        const bool dense_ok = f.dense == 1 ? rc.dense_ladder > 0 : f.dense == 2 ? rc.dense_ladder == 0 : true;
        const bool hc_ok    = f.hc == 1 ? rc.hc_ladder > 0 : f.hc == 2 ? rc.hc_ladder == 0 : true;
        const bool ex_ok    = f.expert == 1 ? (b.kq ? rc.ex_grouped > 0 : rc.ex_grouped == 0) :
                              f.expert == 2 ? rc.ex_grouped == 0 && rc.exd_grouped == 0 :
                                              true;
        // the families on their retained routes (QSA attention, GDN core, PLE) ran in the read
        const bool fam_ok = rc.qsa > 0 && rc.gdn > 0 && rc.ple > 0;
        const bool graph_ok =
            e.graph ? rc.graph_units > 0 && rc.eager_units == 0 : rc.graph_units == 0 && rc.eager_units > 0;
        const std::string d     = ran ? piece_diff(x, x2) : "the ladder arm failed: " + err;
        const bool        state = ran && d.empty();
        const std::string dr    = ran ? piece_diff(ref, x) : std::string();
        const bool        ok    = state && dense_ok && hc_ok && ex_ok && graph_ok && fam_ok;
        bad += ok ? 0 : 1;
        const std::string expect = std::string("dense ") + field(f.dense) + ", hc " + field(f.hc) + ", experts " +
                                   field(f.expert) + (f.expert == 1 && !b.kq ? " (IQ gate/up: per-task)" : "") + ", " +
                                   md::prompt_route_name(e);
        table.push_back(
            "| " + b.name + " | " + r.name + " | " + std::to_string(rows) + " | " + inv + " | " + expect + " | " +
            routes_text(rc) + " | " + (state ? "the rerun bitwise (state and 4 steps)" : "RERUN DIFFERS: " + d) +
            "; against the retained arm: " + (dr.empty() ? std::string("bitwise") : "differs (" + dr + ")") +
            (dense_ok && hc_ok && ex_ok && graph_ok && fam_ok ?
                 "" :
                 std::string("; UNEXPECTED ROUTE:") + (dense_ok ? "" : " dense") + (hc_ok ? "" : " hc") +
                     (ex_ok ? "" : " experts") + (graph_ok ? "" : " graph") + (fam_ok ? "" : " qsa/gdn/ple")) +
            " | " + (ok ? "PASS" : "FAIL") + " |");
        printf("split-test ladder: %s\n", table.back().c_str());
        fflush(stdout);
    }
    free_bytes(b.e->test_model(), f1);
    const int64_t nodes = b.e->prefill_driver().test_graph_nodes() +
                          (b.back != nullptr ? b.back->prefill_driver().test_graph_nodes() : 0);
    printf(
        "split-test ladder memory: %s: free bytes before the reads %lld, %lld, after %lld, %lld (cards 0, 1; the "
        "change: the graph cache and any lazy allocation); unit graph nodes cached %lld\n",
        b.name.c_str(), static_cast<long long>(f0[0]), static_cast<long long>(f0[1]), static_cast<long long>(f1[0]),
        static_cast<long long>(f1[1]), static_cast<long long>(nodes));
    return bad == 0;
}

bool ladder_case(const options & o, bool eager, std::string & report) {
    const std::string        inv = inventory_of(o);
    std::vector<std::string> table;
    table.push_back(
        "| box | read | rows | format | ladder arm's routes | executed variants (ladder arm) | state check | outcome "
        "|");
    table.push_back("| --- | --- | --- | --- | --- | --- | --- | --- |");
    int32_t     bad = 0;
    std::string err;
    {
        md::window w;
        w.lo = o.lo;
        w.hi = o.hi;
        loaded L;
        if (!load_synth(o, w, L)) {
            report = "the one-box load";
            return false;
        }
        engine_config c = engine_cfg(o);
        c.eager         = eager;
        engine e;
        if (!e.init(std::move(L.m), c, err)) {
            report = "the one-box engine: " + err;
            return false;
        }
        lad_box b;
        b.name = "one box (resident TP, layers " + std::to_string(o.lo) + " .. " + std::to_string(o.hi) + ")";
        b.e    = &e;
        b.kq   = kq_layers(e);
        lad_box_reads(b, inv, table, bad);
    }
    // the split over the in-process port at the first k inside the window (the follower on its own thread)
    int32_t k = -1;
    for (const int32_t kk : o.ks) {
        k = k < 0 && kk > o.lo + 1 && kk <= o.hi ? kk : k;
    }
    if (k > 0) {
        md::window wf, wb;
        wf.lo   = o.lo;
        wf.hi   = k - 1;
        wf.head = false;
        wb.lo   = k;
        wb.hi   = o.hi;
        loaded LF, LB;
        if (!load_synth(o, wf, LF) || !load_synth(o, wb, LB)) {
            report = "the split's loads";
            return false;
        }
        std::unique_ptr<stage::port> pa, pb;
        stage::loop_pair(pa, pb);
        stage::buffers  bf, bb;
        stage::crossing xf, xb;
        if (!bf.alloc(n_slot, pa.get(), err) || !bb.alloc(n_slot, pb.get(), err) || !xf.init(*pa, true, k, err) ||
            !xb.init(*pb, false, k, err)) {
            report = "the split's buffers: " + err;
            return false;
        }
        engine_config cf = engine_cfg(o), cb = engine_cfg(o);
        cf.eager = cb.eager = eager;
        cf.stage            = 1;
        cf.stage_buf        = bf.slot;
        cf.record_hash      = false;
        cb.stage            = 2;
        cb.stage_buf        = bb.slot;
        engine ef, eb;
        if (!ef.init(std::move(LF.m), cf, err) || !eb.init(std::move(LB.m), cb, err)) {
            report = "the split's engines: " + err;
            return false;
        }
        ef.prefill_driver().set_stage(&xf);
        eb.prefill_driver().set_stage(&xb);
        stage::leader   L;
        stage::follower F;
        if (!L.init(ef, *pa, xf, err) || !F.init(eb, *pb, xb, err)) {
            report = "the split's stages: " + err;
            return false;
        }
        std::string fwhy;
        bool        fbye = false;
        std::thread fth([&]() { fbye = F.serve(fwhy); });
        lad_box     b;
        b.name = "split at k " + std::to_string(k) + " (front " + std::to_string(o.lo) + " .. " +
                 std::to_string(k - 1) + ", back " + std::to_string(k) + " .. " + std::to_string(o.hi) + ")";
        b.e    = &ef;
        b.back = &eb;
        b.L    = &L;
        b.kq   = kq_layers(ef) || kq_layers(eb);
        lad_box_reads(b, inv, table, bad);
        L.bye();
        fth.join();
        if (!fbye) {
            ++bad;
            table.push_back("| split | the follower | | | | | lost: " + fwhy + " | FAIL |");
        }
    }
    printf("split-test ladder: the coverage table (%s, %s)\n", inv.c_str(), eager ? "eager decode" : "graph decode");
    for (const std::string & t : table) {
        printf("%s\n", t.c_str());
    }
    fflush(stdout);
    report = std::to_string(table.size() - 2) + " reads (" + inv + "): " +
             (bad == 0 ? "every route as selected, every rerun bitwise (the difference from the retained routes "
                         "reported; the KL line judges it)" :
                         std::to_string(bad) + " failed");
    return bad == 0;
}

bool run_cases(const options & o, driver & dv) {
    const auto on = [&](const char * c) {
        return ("," + o.cases + ",").find("," + std::string(c) + ",") != std::string::npos;
    };
    if (on("lone") && !lone_case(dv)) {
        return false;
    }
    if (on("joint") && !joint_case(dv)) {
        return false;
    }
    if (on("prompt") && !prompt_case(dv)) {
        return false;
    }
    return true;
}

// every stage graph keeps the HC slot rule: consecutive sublayers differ, slot 1 last, slot 1 not first
bool slot_rule_check(int32_t & bad) {
    bad             = 0;
    int32_t checked = 0;
    for (int32_t k = 2; k <= fn::n_layer - 1; ++k) {
        for (int32_t role = 0; role < 2; ++role) {
            md::pass_options po;
            po.layer_lo               = role == 0 ? 0 : k;
            po.layer_hi               = role == 0 ? k - 1 : fn::n_layer - 1;
            po.stage.out              = role == 0;
            po.stage.in               = role == 1;
            const md::use_counters u  = md::pass_graph_counters(po);
            const int32_t          G  = u.graph_subs;
            bool                   ok = md::hc_slot_of(u, G - 1) == 1 && md::hc_slot_of(u, 0) != 1;
            for (int32_t n = 0; n + 1 < G; ++n) {
                ok = ok && md::hc_slot_of(u, n) != md::hc_slot_of(u, n + 1);
            }
            bad += ok ? 0 : 1;
            ++checked;
        }
    }
    printf("split-test: the HC slot rule: %d stage graphs (front and back, k = 2 .. 47), %d break it: %s\n", checked,
           bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0;
}

int32_t compare(const std::vector<snapshot> & a, const std::vector<snapshot> & b, const std::string & label) {
    int32_t bad = 0;
    if (a.size() != b.size()) {
        printf("split-test %s: %zu snapshots against the reference's %zu: FAIL\n", label.c_str(), b.size(), a.size());
        return 1;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const snapshot & x    = a[i];
        const snapshot & y    = b[i];
        bool             same = x.what == y.what && x.ints == y.ints && x.state.size() == y.state.size();
        std::string      first;
        for (const auto & [k, h] : x.state) {
            const auto it = y.state.find(k);
            if (it == y.state.end() || it->second != h) {
                same = false;
                if (first.empty()) {
                    first = k;
                }
            }
        }
        if (!same) {
            ++bad;
            if (bad <= 5) {
                printf("split-test %s: snapshot %zu (%s): differs%s%s (%zu state hashes vs %zu; ints %s)\n",
                       label.c_str(), i, x.what.c_str(), first.empty() ? "" : " first at ", first.c_str(),
                       x.state.size(), y.state.size(), x.ints == y.ints ? "equal" : "differ");
            }
        }
    }
    size_t parts = 0;
    for (const snapshot & x : a) {
        parts += x.state.size();
    }
    printf("split-test %s: %zu snapshots, %zu state hashes: %s\n", label.c_str(), a.size(), parts,
           bad == 0 ? "bitwise equal: PASS" : "FAIL");
    return bad;
}

}  // namespace

int fnfl_split_test(int argc, char ** argv) {
    options o;
    for (int32_t i = 2; i < argc; ++i) {
        const std::string a   = argv[i];
        const bool        has = i + 1 < argc;
        if (a == "san") {
            o.san      = true;
            o.ks       = { 3 };
            o.lo       = 0;
            o.hi       = 5;
            o.modes[0] = false;
            o.cases    = "lone";
        } else if (a == "all") {
        } else if (a == "graph") {
            o.modes[1] = false;
        } else if (a == "eager") {
            o.modes[0] = false;
        } else if (a == "--k" && has) {
            o.ks.clear();
            std::string v = argv[++i];
            for (size_t p = 0; p < v.size();) {
                const size_t q = v.find(',', p);
                o.ks.push_back(std::atoi(v.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str()));
                p = q == std::string::npos ? v.size() : q + 1;
            }
        } else if (a == "--cases" && has) {
            o.cases = argv[++i];
        } else if (a == "--perturb") {
            o.perturb = true;
        } else if (a == "--small") {
            o.lo = 0;
            o.hi = 5;
            o.ks = { 3 };
        } else if (a == "--role" && has) {
            const std::string r = argv[++i];
            o.role              = r == "leader" ? 1 : r == "follower" ? 2 : -1;
        } else if (a == "--addr" && has) {
            o.addr = argv[++i];
        } else if (a == "--port" && has) {
            o.port = std::atoi(argv[++i]);
        } else if (a == "--model" && has) {
            o.model_path = argv[++i];
        } else {
            fprintf(stderr,
                    "usage: fnfl-drive split-test [all|san] [graph|eager] [--k LIST] [--cases lone,joint] "
                    "[--model GGUF]\n");
            return 2;
        }
    }
    md::prompt_route_collect(true);  // the coverage case's totals (off in serving)
    int32_t failed = 0;
    int32_t bad    = 0;
    failed += slot_rule_check(bad) ? 0 : 1;
    for (int32_t mode = 0; mode < 2; ++mode) {
        if (!o.modes[mode]) {
            continue;
        }
        const bool   eager = mode == 1;
        const char * mname = eager ? "eager" : "graph";
        if (("," + o.cases + ",").find(",pipe,") != std::string::npos) {
            std::string report;
            const route_mark mk;
            const bool  ok = pipe_case(o, eager, report);
            printf("split-test %s pipe: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf("split-test %s pipe (the server's path: stage::pipeline and the leader's reads, k = 27): %s: %s\n",
                   mname, report.c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "pipe") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",deliver,") != std::string::npos) {
            std::string report;
            const route_mark mk;
            const bool  ok = deliver_case(o, eager, report);
            printf("split-test %s deliver: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf("split-test %s deliver (TTTS8: the server's split calls, k = 27): %s%s\n", mname,
                   (o.role == 2 ? report + ": " : std::string()).c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "deliver") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",truncate,") != std::string::npos) {
            std::string report;
            const route_mark mk;
            const bool  ok = truncate_case(o, eager, report);
            printf("split-test %s truncate: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf("split-test %s truncate (TTTS8: the follower's failed cut inside the last step, k = 27): %s\n",
                   mname, ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "truncate") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",pieces,") != std::string::npos) {
            std::string report;
            const route_mark mk;
            const bool  ok = pieces_case(o, eager, report);
            printf("split-test %s pieces: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf(
                "split-test %s pieces (MUP2: reads in pieces and units against today's lone read, one box and the "
                "split at k = 27): %s: %s\n",
                mname, report.c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "pieces") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",image,") != std::string::npos) {
            std::string report;
            const route_mark mk;
            const bool  ok = image_case(o, eager, report);
            printf("split-test %s image: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf("split-test %s image (VIS1: image rows, one box against the split at k = 27): %s: %s\n", mname,
                   report.c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "image") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",latch,") != std::string::npos) {
            std::string report;
            const route_mark mk;
            const bool  ok = latch_case(o, eager, report);
            printf("split-test %s latch: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf("split-test %s latch (PRD1: a forced joint-capture failure on the split's front, k = 27): %s: %s\n",
                   mname, report.c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "latch") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",ladder,") != std::string::npos) {
            std::string      report;
            const route_mark mk;
            const bool       ok = ladder_case(o, eager, report);
            printf("split-test %s ladder: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf(
                "split-test %s ladder (APF2: the kernel-size ladder's routes through the resident and split callers, "
                "each read twice, beside the retained routes): %s: %s\n",
                mname, report.c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "ladder") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",reads,") != std::string::npos) {
            std::string report;
            const route_mark mk;
            const bool  ok = reads_case(o, eager, report);
            printf("split-test %s reads: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf(
                "split-test %s reads (PRD1: small reads whole, no-writer progress, the front freed at its end, the "
                "crossing received ahead; k = 27): %s: %s\n",
                mname, report.c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "reads") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",engine,") != std::string::npos) {
            std::string report;
            const route_mark mk;
            const bool  ok = engine_case(o, eager, report);
            printf("split-test %s engine: the ladder's executed variants (every prompt path): %s\n", mname,
                   routes_text(mk.since()).c_str());
            printf("split-test %s engine (the leader and follower over the in-process port, k = 27): %s: %s\n", mname,
                   report.c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "engine") {
                continue;
            }
        }
        // the reference: one box, the whole window (the small shape: layers 0 .. 5 with the head)
        std::vector<snapshot> ref;
        {
            md::window w;
            w.lo = o.lo;
            w.hi = o.hi;
            loaded    L;
            stage_set ss;
            driver    dv;
            dv.whole = &ss;
            printf("split-test: %s: the reference, one box, layers %d .. %d\n", mname, w.lo, w.hi);
            fflush(stdout);
            if (!load_synth(o, w, L) || !make_set(o, *L.m, 0, eager, ss) || !run_cases(o, dv)) {
                printf("split-test: %s reference: FAIL: %s\n", mname, dv.err.c_str());
                return 1;
            }
            ref = std::move(dv.snaps);
            // the reference is not trivial: the decided tokens vary and the final logits are finite
            std::map<int32_t, int32_t> seen;
            for (const snapshot & x : ref) {
                for (int32_t v : x.ints) {
                    seen[v]++;
                }
            }
            int64_t nonfinite = 0, total = 0;
            {
                const md::slot_scope       bind(*L.m, 0);
                const md::reserved_class * rc = L.m->state_class("round and pass (TG1)");
                for (int32_t p = 0; rc != nullptr && p < (int32_t) rc->plan.parts.size(); ++p) {
                    if (rc->plan.parts[size_t(p)].name == "logits (B <= 5)") {
                        std::vector<float> h(size_t(rc->plan.parts[size_t(p)].bytes[0] / 4));
                        cudaSetDevice(L.m->device[0]);
                        cudaMemcpy(h.data(), rc->part_ptr(p, 0), h.size() * 4, cudaMemcpyDeviceToHost);
                        for (float v : h) {
                            nonfinite += std::isfinite(v) ? 0 : 1;
                        }
                        total = int64_t(h.size());
                    }
                }
            }
            printf(
                "split-test: %s reference: %zu snapshots, %zu distinct recorded values, logits of slot 0 card 0: "
                "%lld of %lld non-finite\n",
                mname, ref.size(), seen.size(), (long long) nonfinite, (long long) total);
        }
        for (int32_t k : o.ks) {
            if (k <= o.lo + 1 || k > o.hi) {
                printf("split-test: k %d outside %d .. %d: skipped\n", k, o.lo + 2, o.hi);
                continue;
            }
            md::window wf, wb;
            wf.lo   = o.lo;
            wf.hi   = k - 1;
            wf.head = false;
            wb.lo   = k;
            wb.hi   = o.hi;
            loaded    LF, LB;
            stage_set sf, sb;
            driver    dv;
            dv.front   = &sf;
            dv.back    = &sb;
            dv.perturb = o.perturb;
            printf("split-test: %s: the split at k = %d (front %d .. %d, back %d .. %d with the head)\n", mname, k,
                   wf.lo, wf.hi, wb.lo, wb.hi);
            fflush(stdout);
            if (!load_synth(o, wf, LF) || !load_synth(o, wb, LB) || !make_set(o, *LF.m, 1, eager, sf) ||
                !make_set(o, *LB.m, 2, eager, sb) || !run_cases(o, dv)) {
                printf("split-test: %s k %d: FAIL: %s\n", mname, k, dv.err.c_str());
                ++failed;
                continue;
            }
            const int32_t diff = compare(ref, dv.snaps, std::string(mname) + " k " + std::to_string(k));
            if (o.perturb) {
                printf("split-test %s k %d: the perturbed crossing %s\n", mname, k,
                       diff > 0 ? "was detected: PASS (the test can fail)" : "was NOT detected: FAIL");
                failed += diff > 0 ? 0 : 1;
            } else {
                failed += diff == 0 ? 0 : 1;
            }
        }
    }
    printf("split-test: %s\n", failed == 0 ? "PASS" : "FAIL");
    return failed == 0 ? 0 : 1;
}

}  // namespace fn::runtime
