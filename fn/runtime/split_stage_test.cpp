#include "fn/runtime/split_stage_test.h"

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
    if (e != cudaSuccess) {
        fprintf(stderr, "split-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
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
                    std::memcpy(dst + i * 4, &v, 4);
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
    std::string model_path =
        "/home/jeremy/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/"
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    std::string draft_path  = "/home/jeremy/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
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

bool load_synth(const options & o, const md::window & w, loaded & out, int32_t n_slots = n_slot) {
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
    int64_t host_bytes = 0;
    if (!fn::loader::reserve_state(m, sp, n_slots, host_bytes, err, n_slots * slot_cells) ||
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
        return true;
    }

    bool in(target & t, md::prompt_path & pp, md::prompt_chunk & c, std::string & err) override {
        if (next >= q.size() || start[next] != c.start || rows[next] != c.rows) {
            err = "crossing: the back's chunk at " + std::to_string(c.start) + " (" + std::to_string(c.rows) +
                  " rows) is not the front's next";
            return false;
        }
        c.seq         = seq[next];
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
// QSA layers; the PLE history; with the head, the logits and the hidden carry)
bool state_of(md::model & m, int32_t s, target & t, snapshot & snap, reader & rd) {
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
            for (int32_t c = 0; c < 2; ++c) {
                const int64_t n = rc->live_bytes(p, c);
                uint64_t      h = 0;
                if (n > 0 && !rd.read(m.device[c], rc->part_ptr(p, c), size_t(n), h)) {
                    return false;
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
        if (!part_hash("round and pass (TG1)", "logits (B <= 5)", "logits") ||
            !part_hash("round and pass (TG1)", "hidden carry (FC-HEAD3, B <= 5)", "carry")) {
            fprintf(stderr, "split-test: the logits and carry\n");
            return false;
        }
    }
    (void) t;
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
                if (!state_of(*front->m, s, *front->t[s], x, rd) || !state_of(*back->m, s, *back->t[s], x, rd)) {
                    err = what + ": a state read";
                    return false;
                }
            } else if (!state_of(*whole->m, s, *whole->t[s], x, rd)) {
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
        if (!front->pf->begin(*front->t[s], all, n_reuse, from, err) || !front->pf->run(*front->t[s], nullptr, err)) {
            err = "front prefill: " + err;
            return false;
        }
        back->cross.q     = std::move(front->cross.q);
        back->cross.seq   = front->cross.seq;
        back->cross.start = front->cross.start;
        back->cross.rows  = front->cross.rows;
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

const char * const draft_subset = "/home/jeremy/src/flashnext/work/DR1/stageC/subset/draft-subset.ids";

engine_config engine_cfg(const options & o) {
    engine_config c;
    c.model_path      = o.model_path;
    c.n_ctx_slot      = slot_cells;
    c.draft.on        = true;
    c.draft.attention = "sparse";
    c.draft.subset    = draft_subset;
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
    stage::leader *       lead  = nullptr;    // the split's calls
    std::vector<step_rec> exact, functional;  // the steps that must be bitwise one box's, and the others
    std::string           err;

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

    run_status prefill(int32_t s, const std::vector<int32_t> & all, int32_t n_reuse) {
        int32_t          np = 0;
        const run_status rs = lead != nullptr ?
                                  lead->prefill(s, all.data() + n_reuse, int32_t(all.size()) - n_reuse, n_reuse, np) :
                                  one->prefill(s, all.data() + n_reuse, int32_t(all.size()) - n_reuse, n_reuse, np);
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

// ---- the pipe case: the server's path over the in-process port (stage::pipeline, the leader's reads) ----

// Two processes over the link, the real topology (each stage alone on its box's cards: both stages pipelined on one
// box's cards starve each other's cross-card polls): the leader (--role leader, the front) and the follower
// (--role follower, the back), on synthetic weights. Eight requests: four prompts read whole first, then four reads
// (one of 9,000 tokens: two chunks) in the writers geometry a layer group a turn beside the pipelined groups, each
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
                    P.settle(s);
                    P.forget(s);
                    L.release_slot(s);
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
            // the writers geometry throughout (the reference's), writers present or not
            const read_status st = L.read_step(s, 1, read_mode::writers);
            ++read_groups;
            if (st == read_status::done) {
                writers.push_back(s);
            } else if (st == read_status::yielded || st == read_status::chunk_done) {
                rkeep.push_back(s);
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
            const bool  ok = pipe_case(o, eager, report);
            printf("split-test %s pipe (the server's path: stage::pipeline and the leader's reads, k = 27): %s: %s\n",
                   mname, report.c_str(), ok ? "PASS" : "FAIL");
            fflush(stdout);
            failed += ok ? 0 : 1;
            if (o.cases == "pipe") {
                continue;
            }
        }
        if (("," + o.cases + ",").find(",engine,") != std::string::npos) {
            std::string report;
            const bool  ok = engine_case(o, eager, report);
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
