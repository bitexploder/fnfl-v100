#include "fn/kernels/dense/dense_joint_test.h"

#include "fn/kernels/dense/dense.h"
#include "fn/kernels/dense/dense_test.h"
#include "fn/kernels/experts/grids.h"
#include "fn/kernels/gdn/gdn.h"
#include "fn/kernels/round/version.h"
#include "fn/kernels/samp/samp_joint_test.h"
#include "fn/quant/act_ref.h"
#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/xchg/x16_ref.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fn::dense {
namespace {

constexpr int32_t n_rows = max_joint_rows; // the buffers' rows
using hl = fn::xchg::hmix_layout;

bool cuda_ok(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "dpj-test: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

struct dev_mem {
    uint8_t * p = nullptr;
    size_t n = 0;

    explicit dev_mem(size_t bytes) : n(std::max<size_t>(bytes, 16)) {
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

    template <typename T> T * as() const { return reinterpret_cast<T *>(p); }
    bool put(const void * h, size_t bytes, size_t at = 0) const {
        return p != nullptr && at + bytes <= n && cuda_ok(cudaMemcpy(p + at, h, bytes, cudaMemcpyHostToDevice), "put");
    }
    bool get(void * h, size_t bytes) const {
        return p != nullptr && bytes <= n && cuda_ok(cudaMemcpy(h, p, bytes, cudaMemcpyDeviceToHost), "get");
    }
    bool poison() const { return p != nullptr && cuda_ok(cudaMemset(p, 0xff, n), "memset"); }
    bool zero() const { return p != nullptr && cuda_ok(cudaMemset(p, 0, n), "memset"); }
};

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    double next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

uint64_t name_seed(const std::string & s) {
    uint64_t h = 1469598103934665603ull ^ 0x4a4f494e54ull; // "JOINT": not dp-test's activations
    for (const char c : s) {
        h = (h ^ static_cast<uint8_t>(c)) * 1099511628211ull;
    }
    return h;
}

template <typename F> void parallel_for(int32_t n, const F & fn) {
    const int32_t nt = std::min<int32_t>(n, 12);
    std::atomic<int32_t> next{0};
    std::vector<std::thread> th;
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

quant::wfmt to_wfmt(fmt f) {
    switch (f) {
        case fmt::q4_k:
            return quant::wfmt::q4_k;
        case fmt::q5_k:
            return quant::wfmt::q5_k;
        case fmt::q6_k:
            return quant::wfmt::q6_k;
        case fmt::iq4_xs:
            return quant::wfmt::iq4_xs;
        case fmt::iq3_s:
            return quant::wfmt::iq3_s;
        case fmt::q8_0:
            return quant::wfmt::q8_0;
        case fmt::iq4_nl:
            return quant::wfmt::iq4_nl;
        case fmt::q2_0:
            return quant::wfmt::q2_0;
        case fmt::bf16:
            return quant::wfmt::bf16;
        case fmt::f32:
            return quant::wfmt::f32;
    }
    return quant::wfmt::f32;
}

struct mspec {
    const char * name;
    fmt f;
    int32_t m;
    epi e;
    int32_t pair;
    int32_t split;
    int32_t period;
    bool scaled;
};

mspec plain(const char * name, fmt f, int32_t m, int32_t split = 0) {
    return {name, f, m, epi::f32, -1, split, n_synth_rows, false};
}
mspec scaled(const char * name, fmt f, int32_t m) {
    return {name, f, m, epi::f32, -1, 0, n_synth_rows, true};
}
mspec glu(const char * name, fmt f, epi e, int32_t pair, int32_t split = 0) {
    return {name, f, 320, e, pair, split, n_synth_rows, false};
}
mspec conv(const char * name, fmt f) {
    return {name, f, fn::gdn::n_channel, epi::conv_silu, -1, 1, n_synth_rows, false};
}

struct jcase {
    std::string name;
    int32_t k;
    std::vector<mspec> mats;
    bool zero = false;
    bool route = false;
    double xscale = 1.0;
    int32_t nan_expert = -1;
    bool conv = false;
    bool hmix = false;     // the activations through per-lane X-HMIX slots (K 2,560)
    bool lane_out = false; // the joint launch writes each lane's rows to its own buffers (joint_args::out; the head)
    bool lane_x = false;   // each lane's F32 activations in its own buffer (joint_args::x_f32; PLE, FMU2 item 4)
    bool extreme = false;  // FMS2 item 4: extreme weight codes (extremize) and every activation code at +-127
    bool peer = false;     // FMS2 item 6: X-A sent by the producer (joint_args::peer), half 0 then half 1
    bool x16 = false;      // FMS2 item 7: that send in 16 bits (joint_args::x16_err)
    bool over = false;     // FMS2 item 7: the case must send a value beyond FP16's range (its flag set)
    bool tc = false;       // FMS5: the router on the tensor cores (joint_args::router_tc), judged by run_case_tc
    bool f16 = false;      // FMS6: the quantized products in FP16 (joint_args::dense_f16), judged by run_case_f16
};

// a case's device buffers for n_rows rows and the joint launch's arguments (row 0's pointers)
struct jrun {
    const jcase & tc;
    std::vector<std::unique_ptr<dev_mem>> w, out;
    std::unique_ptr<dev_mem> xc, xs, xf, grid, gc, gs, gf, ids, wts, sg, scale, taps, slots, err, work_l, work_j;
    std::unique_ptr<dev_mem> hist[max_lanes], cur[max_lanes], lane_out[max_lane_mats][max_lanes], lane_x[max_lanes];
    std::unique_ptr<dev_mem> peer; // FMS2 item 6: the peer's joint slot (n_rows rows of words_half words)
    int32_t peer_half = 0;
    size_t peer_bad = 0; // peer words not as send_half_joint would write them (or rows past the lanes' not untouched)
    std::unique_ptr<dev_mem> x16_err; // FMS2 item 7: the 16-bit send's range flag
    size_t x16_over = 0;              // the sends with a value beyond FP16's range (the flag then checked set)
    std::vector<float> hist0[max_lanes];
    uint32_t cur_h[max_lanes] = {};
    product_args a;
    int32_t glu_m = 0;
    std::vector<int32_t> off;
    std::vector<float> x, xv; // F32 rows, and the decode format's values
    std::vector<int8_t> codes;
    std::vector<float> scales;
    std::vector<float>       sc;              // a scaled matrix's per-row scale
    int32_t                  rr = lane_rows;  // FMS13: R, the rows of each lane the launches read and write

    explicit jrun(const jcase & c) : tc(c) {}

    // FMS13: buffer row b is a row of the launches (row b % 5 of lane b / 5 below R)
    bool sent(int32_t b) const { return b % lane_rows < rr; }

    // FMS13: the inputs at R = rr: rows R..4 of every lane's activations (F32, codes and scales, X-HMIX slots, lane
    // buffers) and scales NaN or code 127, so a launch that reads them taints a compared row or a sentinel
    bool inputs(int32_t r) {
        rr                     = r;
        const int32_t       k  = tc.k;
        std::vector<float>  xt = x, st = scales, sct = sc;
        std::vector<int8_t> ct = codes;
        for (int32_t b = 0; b < n_rows; ++b) {
            if (sent(b)) {
                continue;
            }
            std::fill(xt.begin() + size_t(b) * k, xt.begin() + size_t(b + 1) * k, NAN);
            std::fill(ct.begin() + size_t(b) * k, ct.begin() + size_t(b + 1) * k, int8_t(127));
            std::fill(st.begin() + size_t(b) * k / 16, st.begin() + size_t(b + 1) * k / 16, NAN);
            sct[size_t(b)] = NAN;
        }
        bool ok = xc->put(ct.data(), ct.size()) && xs->put(st.data(), st.size() * 4) &&
                  xf->put(xt.data(), xt.size() * 4) && scale->put(sct.data(), sct.size() * 4);
        for (int32_t u = 0; ok && tc.lane_x && u < max_lanes; ++u) {
            ok = lane_x[u]->put(xt.data() + size_t(lane_rows) * u * k, size_t(lane_rows) * k * 4);
        }
        for (int32_t u = 0; ok && tc.hmix && u < max_lanes; ++u) {
            const size_t base = size_t(u) * hl::slot_words * 4, r0 = size_t(lane_rows) * u;
            ok = slots->put(xt.data() + r0 * k, size_t(lane_rows) * k * 4, base) &&
                 slots->put(ct.data() + r0 * k, size_t(lane_rows) * k, base + size_t(hl::code_at) * 4) &&
                 slots->put(st.data() + r0 * k / 16, size_t(lane_rows) * k / 16 * 4, base + size_t(hl::scale_at) * 4);
        }
        return ok;
    }

    // FMS13, the sentinel test: the bytes of rows R..4 of each lane in a snapshot's per-row outputs (laid out as
    // snapshot writes them) that are not the poison 0xff
    size_t touched(int32_t lanes, const std::vector<uint8_t> & o) const {
        const int32_t rows = lane_rows * lanes;
        size_t        at = 0, n = 0;
        const auto    region = [&](size_t row_bytes) {
            for (int32_t b = 0; b < rows; ++b) {
                for (size_t i = 0; !sent(b) && i < row_bytes; ++i) {
                    n += o[at + size_t(b) * row_bytes + i] != 0xff ? 1 : 0;
                }
            }
            at += size_t(rows) * row_bytes;
        };
        for (size_t i = 0; i < out.size(); ++i) {
            region(size_t(tc.mats[i].m) * 4);
        }
        if (glu_m > 0) {
            region(size_t(glu_m));
            region(size_t(glu_m) / 16 * 4);
            region(size_t(glu_m) * 4);
        }
        if (tc.route) {
            region(size_t(route_used) * 4);
            region(size_t(route_used) * 4);
            region(4);
        }
        return n;
    }

    // the per-row outputs back to their initial state: poisoned, each lane's history version 0 at its slot
    bool reset(int32_t lanes) {
        bool ok = true;
        for (const auto & d : out) {
            ok = ok && d->poison();
        }
        ok = ok && gc->poison() && gs->poison() && gf->poison() && ids->poison() && wts->poison() && sg->poison();
        for (size_t i = 0; tc.lane_out && i < tc.mats.size(); ++i) {
            for (int32_t u = 0; ok && u < max_lanes; ++u) {
                ok = lane_out[i][u]->poison();
            }
        }
        for (int32_t u = 0; ok && tc.conv && u < lanes; ++u) {
            ok = hist[u]->poison() &&
                 hist[u]->put(hist0[u].data(), hist0[u].size() * 4,
                              size_t(fn::round::version_slot(cur_h[u], 0)) * fn::gdn::hist_version_floats * 4);
        }
        return ok;
    }

    // every output byte of rows [0, 5 lanes) and the lanes' histories
    bool snapshot(int32_t lanes, std::vector<uint8_t> & o) {
        o.clear();
        const auto add = [&](const dev_mem & d, size_t bytes) {
            const size_t at = o.size();
            o.resize(at + bytes);
            return d.get(o.data() + at, bytes);
        };
        const int32_t rows = lane_rows * lanes;
        bool ok = cuda_ok(cudaDeviceSynchronize(), "launch");
        for (size_t i = 0; ok && i < out.size(); ++i) {
            ok = add(*out[i], size_t(rows) * tc.mats[i].m * 4);
        }
        if (ok && glu_m > 0) {
            ok = add(*gc, size_t(rows) * glu_m) && add(*gs, size_t(rows) * glu_m / 16 * 4) &&
                 add(*gf, size_t(rows) * glu_m * 4);
        }
        if (ok && tc.route) {
            ok = add(*ids, size_t(rows) * route_used * 4) && add(*wts, size_t(rows) * route_used * 4) &&
                 add(*sg, size_t(rows) * 4);
        }
        for (int32_t u = 0; ok && tc.conv && u < lanes; ++u) {
            ok = add(*hist[u], hist[u]->n);
        }
        return ok;
    }

    // lane u's lone launch at B = R: row 0 of each buffer moved to row 5 u
    bool lone(int32_t u) {
        product_args l = a;
        const int32_t r0 = lane_rows * u, k = tc.k;
        l.x_codes = a.x_codes != nullptr ? a.x_codes + size_t(r0) * k : nullptr;
        l.x_scales = a.x_scales != nullptr ? a.x_scales + size_t(r0) * (k / 32) * 2 : nullptr;
        l.x_f32 = a.x_f32 != nullptr ? a.x_f32 + size_t(r0) * k : nullptr;
        l.hmix = a.hmix != nullptr ? a.hmix + size_t(u) * hl::slot_words : nullptr;
        for (int32_t i = 0; i < a.n_mats; ++i) {
            mat & m = l.mats[i];
            m.out = m.out != nullptr ? m.out + size_t(r0) * m.m : nullptr;
            m.scale = m.scale != nullptr ? m.scale + r0 : nullptr;
        }
        l.glu_codes = a.glu_codes != nullptr ? a.glu_codes + size_t(r0) * glu_m : nullptr;
        l.glu_scales = a.glu_scales != nullptr ? a.glu_scales + size_t(r0) * glu_m / 16 : nullptr;
        l.glu_f32 = a.glu_f32 != nullptr ? a.glu_f32 + size_t(r0) * glu_m : nullptr;
        l.route.ids = a.route.ids != nullptr ? a.route.ids + r0 * route_used : nullptr;
        l.route.wts = a.route.wts != nullptr ? a.route.wts + r0 * route_used : nullptr;
        l.route.sg = a.route.sg != nullptr ? a.route.sg + r0 : nullptr;
        if (tc.conv) {
            l.conv = {taps->as<float>(), hist[u]->as<float>(), cur[u]->as<uint32_t>()};
        }
        l.work = work_l->p;
        return product(rr, l, nullptr);
    }

    // the producer's peer words against send_half_joint's: row b < 5 lanes word q is the canonical bits of matrix 0's
    // output (b, peer_half words_half + q); the rows past the lanes' and rows R..4 of each lane (FMS13) stay the
    // sentinel. FMS2 item 7 (tc.x16): row b half q at b x16_row_halves + q is the host's rounding of that value
    // (fn/xchg/x16_ref.h), every other half the sentinel, and the flag set exactly when a value is beyond FP16's range
    bool check_peer(int32_t lanes) {
        constexpr int32_t hw = fn::xchg::words_half;
        const int32_t m = tc.mats[0].m;
        std::vector<float> o(size_t(n_rows) * m);
        std::vector<uint32_t> got(size_t(n_rows) * hw);
        uint32_t flag = 0;
        if (!cuda_ok(cudaDeviceSynchronize(), "launch") || !out[0]->get(o.data(), o.size() * 4) ||
            !peer->get(got.data(), got.size() * 4) || (tc.x16 && !x16_err->get(&flag, 4))) {
            return false;
        }
        if (tc.x16) {
            bool over = false;
            for (size_t i = 0; i < got.size() * 2; ++i) {
                const size_t b = i / fn::xchg::x16_row_halves, q = i % fn::xchg::x16_row_halves;
                const uint16_t h = static_cast<uint16_t>(got[i / 2] >> (16 * (i % 2)));
                const uint16_t want = b < size_t(lane_rows * lanes) && sent(int32_t(b)) ?
                                          fn::xchg::x16_ref(o[b * m + size_t(peer_half) * hw + q], over) :
                                          fn::xchg::x16_sentinel;
                peer_bad += h != want ? 1 : 0;
            }
            peer_bad += flag != (over ? fn::xchg::err_x16_range : 0u) ? 1 : 0;
            x16_over += over ? 1 : 0;
            return true;
        }
        for (int32_t b = 0; b < n_rows; ++b) {
            for (int32_t q = 0; q < hw; ++q) {
                const float v = o[size_t(b) * m + size_t(peer_half) * hw + q];
                uint32_t want = 0xffffffffu;
                if (b < lane_rows * lanes && sent(b)) {
                    memcpy(&want, &v, 4);
                    want = v != v ? fn::xchg::canonical_nan : want;
                }
                peer_bad += got[size_t(b) * hw + q] != want ? 1 : 0;
            }
        }
        return true;
    }

    bool joint(int32_t lanes) {
        joint_args j;
        j.lanes = lanes;
        j.rows             = rr;
        j.hmix_lane_floats = hl::slot_words;
        for (int32_t u = 0; tc.conv && u < lanes; ++u) {
            j.conv[u] = {taps->as<float>(), hist[u]->as<float>(), cur[u]->as<uint32_t>()};
        }
        j.router_tc = tc.tc;
        j.dense_f16 = tc.f16;
        product_args l = a;
        l.work = work_j->p;
        for (size_t i = 0; tc.lane_out && i < tc.mats.size(); ++i) {
            for (int32_t u = 0; u < lanes; ++u) {
                j.out[i][u] = lane_out[i][u]->as<float>();
            }
            l.mats[i].out = nullptr; // ignored with lane outputs
        }
        for (int32_t u = 0; tc.lane_x && u < lanes; ++u) {
            j.x_f32[u] = lane_x[u]->as<float>();
        }
        l.x_f32 = tc.lane_x ? nullptr : l.x_f32; // ignored with per-lane activations
        bool ok = true;
        if (tc.peer) {
            ok = peer->poison() && (!tc.x16 || x16_err->zero());
            j.peer = peer->as<uint32_t>();
            j.peer_half = peer_half;
            j.x16_err = tc.x16 ? x16_err->as<uint32_t>() : nullptr;
        }
        ok = ok && product_joint(l, j, nullptr);
        if (ok && tc.peer) {
            ok = check_peer(lanes);
        }
        // each lane's buffers into its rows of the matrices' outputs, which the snapshot reads
        for (size_t i = 0; ok && tc.lane_out && i < tc.mats.size(); ++i) {
            const size_t lane_bytes = size_t(lane_rows) * tc.mats[i].m * 4;
            for (int32_t u = 0; ok && u < lanes; ++u) {
                ok = cuda_ok(cudaMemcpy(out[i]->as<uint8_t>() + u * lane_bytes, lane_out[i][u]->p, lane_bytes,
                                        cudaMemcpyDeviceToDevice),
                             "lane output");
            }
        }
        return ok;
    }
};

// FMS2 build item 4: the synthetic rows with every code field at an extreme, so the tensor-core products' integer sums
// reach their largest magnitudes. Row r (block scales kept): r % 4 = 0 the largest positive decoded codes, 1 the
// largest negative (or the format's lowest code), 2 the two alternating byte by byte, 3 the random row unchanged.
// Sub-scales go to their extremes with the codes (6-bit 63, Q6_K's int8 127 and -128, IQ4_XS's 63 and 0); IQ3_S takes
// the grid entry of the largest values, with every sign bit clear or set.
void extremize(fmt f, synth & s) {
    int32_t g3 = 0, best = -1; // IQ3_S: the grid index whose four values sum largest
    for (int32_t i = 0; i < 512; ++i) {
        int32_t sum = 0;
        for (int32_t b = 0; b < 4; ++b) {
            sum += static_cast<int32_t>((fn::experts::iq3s_grid[i] >> (8 * b)) & 0xffu);
        }
        if (sum > best) {
            best = sum;
            g3 = i;
        }
    }
    const int32_t bb = quant::info(to_wfmt(f)).block_bytes;
    for (int32_t r = 0; r < n_synth_rows; ++r) {
        const int32_t mode = r % 4;
        if (mode == 3) {
            continue;
        }
        uint8_t * row = s.bytes.data() + size_t(r) * s.row_bytes;
        for (size_t at = 0; at + size_t(bb) <= s.row_bytes; at += size_t(bb)) {
            uint8_t * b = row + at;
            const auto fill = [&](int32_t from, int32_t to, uint8_t hi, uint8_t lo) {
                for (int32_t i = from; i < to; ++i) {
                    b[i] = mode == 0 ? hi : mode == 1 ? lo : ((i & 1) ? lo : hi);
                }
            };
            switch (f) {
                case fmt::q8_0:
                    fill(2, 34, 0x7f, 0x80);
                    break;
                case fmt::q4_k:
                    fill(4, 16, 0xff, 0xff);
                    fill(16, 144, 0xff, 0x00);
                    break;
                case fmt::q5_k:
                    fill(4, 16, 0xff, 0xff);
                    fill(16, 176, 0xff, 0x00);
                    break;
                case fmt::q6_k:
                    fill(0, 192, 0xff, 0x00);
                    fill(192, 208, 0x7f, 0x80);
                    break;
                case fmt::iq4_xs:
                    fill(2, 8, 0xff, 0x00);
                    fill(8, 136, 0xff, 0x00);
                    break;
                case fmt::iq4_nl:
                    fill(2, 18, 0xff, 0x00);
                    break;
                case fmt::q2_0:
                    fill(2, 18, 0xff, 0x00);
                    break;
                case fmt::iq3_s:
                    for (int32_t i = 2; i < 66; ++i) {
                        b[i] = static_cast<uint8_t>(g3 & 0xff);
                    }
                    for (int32_t i = 66; i < 74; ++i) {
                        b[i] = (g3 & 0x100) != 0 ? 0xff : 0x00;
                    }
                    fill(74, 106, 0x00, 0xff);
                    fill(106, 110, 0xff, 0xff);
                    break;
                case fmt::bf16:
                case fmt::f32:
                    break;
            }
        }
    }
}

int64_t synth_key(const jcase & tc, fmt f) {
    return int64_t(f) * 65536 + tc.k + (tc.extreme ? int64_t(1) << 40 : 0);
}

bool setup(jrun & cr, std::map<int64_t, synth> & sets, const std::vector<uint8_t> & grids) {
    const jcase & tc = cr.tc;
    const int32_t k = tc.k, n = static_cast<int32_t>(tc.mats.size());
    product_args & a = cr.a;
    a.k = k;
    a.n_mats = n;
    bool any_bf = false, any_q = false;
    cr.off.assign(size_t(n), 0);
    for (int32_t i = 0; i < n; ++i) {
        const mspec & ms = tc.mats[size_t(i)];
        const int64_t key = synth_key(tc, ms.f);
        if (sets.count(key) == 0) {
            if (!make_synth(ms.f, k, sets[key])) {
                fprintf(stderr, "dpj-test: no synthetic %s at K %d\n", fmt_name(ms.f), k);
                return false;
            }
            if (tc.extreme) {
                extremize(ms.f, sets[key]);
            }
        }
        const synth & s = sets[key];
        cr.off[size_t(i)] = (37 * i + 11) % n_synth_rows;
        std::vector<uint8_t> packed;
        if (!pack_synth(s, ms.m, cr.off[size_t(i)], ms.period, packed)) {
            return false;
        }
        if (i == 0 && tc.nan_expert >= 0) { // BF16: the row's first weight a quiet NaN
            uint8_t * w0 = packed.data() + size_t(tc.nan_expert) * s.row_bytes;
            w0[0] = 0xc0;
            w0[1] = 0x7f;
        }
        cr.w.push_back(std::make_unique<dev_mem>(packed.size()));
        cr.out.push_back(std::make_unique<dev_mem>(size_t(n_rows) * ms.m * 4));
        if (!cr.w.back()->put(packed.data(), packed.size())) {
            return false;
        }
        mat & m = a.mats[i];
        m.f = ms.f;
        m.m = ms.m;
        m.w = cr.w.back()->p;
        m.out = ms.e == epi::f32 || ms.e == epi::conv_silu ? cr.out.back()->as<float>() : nullptr;
        m.e = ms.e;
        m.pair = ms.pair;
        m.split = ms.split;
        cr.glu_m = ms.e == epi::glu_gate ? ms.m : cr.glu_m;
        const bool wide = ms.f == fmt::bf16;
        any_bf = any_bf || wide;
        any_q = any_q || !wide;
    }
    choose_splits(a);
    cr.x.assign(size_t(n_rows) * k, 0.0f);
    if (!tc.zero) {
        lcg g(name_seed(tc.name));
        for (size_t i = 0; i < cr.x.size(); ++i) {
            const double v = (g.next() + g.next() + g.next() - 1.5) * (1.0 + 3.0 * double(i % 97 == 0));
            cr.x[i] = static_cast<float>(v * tc.xscale);
        }
    }
    for (size_t i = 0; tc.extreme && i < cr.x.size(); ++i) { // every code at +-127: rows +1, -1, then alternating
        const size_t b = i / size_t(k) % 3;
        cr.x[i] = b == 0 ? 1.0f : b == 1 ? -1.0f : ((i & 1) != 0 ? -1.0f : 1.0f);
    }
    cr.codes.resize(cr.x.size());
    cr.scales.resize(cr.x.size() / 16);
    quant::act_quantize_ref(cr.x.data(), n_rows, k, cr.codes.data(), cr.scales.data());
    cr.xv.resize(cr.x.size());
    for (int32_t b = 0; b < n_rows; ++b) {
        quant::act_values_ref(cr.codes.data() + size_t(b) * k, cr.scales.data() + size_t(b) * k / 16, k,
                              cr.xv.data() + size_t(b) * k);
    }
    std::vector<float> & sc = cr.sc;
    sc.assign(n_rows, 0.0f);
    for (int32_t b = 0; b < n_rows; ++b) {
        sc[size_t(b)] = 1.0f / (1.0f + std::exp(-0.37f * float(b - 9))); // sigmoid-like values in (0, 1)
    }
    const size_t gm = size_t(std::max(cr.glu_m, 32));
    cr.xc = std::make_unique<dev_mem>(cr.codes.size());
    cr.xs = std::make_unique<dev_mem>(cr.scales.size() * 4);
    cr.xf = std::make_unique<dev_mem>(cr.x.size() * 4);
    cr.grid = std::make_unique<dev_mem>(grid_bytes);
    cr.gc = std::make_unique<dev_mem>(n_rows * gm);
    cr.gs = std::make_unique<dev_mem>(n_rows * gm / 16 * 4);
    cr.gf = std::make_unique<dev_mem>(n_rows * gm * 4);
    cr.ids = std::make_unique<dev_mem>(size_t(n_rows) * route_used * 4);
    cr.wts = std::make_unique<dev_mem>(size_t(n_rows) * route_used * 4);
    cr.sg = std::make_unique<dev_mem>(size_t(n_rows) * 4);
    cr.scale = std::make_unique<dev_mem>(size_t(n_rows) * 4);
    cr.err = std::make_unique<dev_mem>(4);
    for (size_t i = 0; tc.lane_out && i < tc.mats.size(); ++i) {
        for (int32_t u = 0; u < max_lanes; ++u) {
            cr.lane_out[i][u] = std::make_unique<dev_mem>(size_t(lane_rows) * tc.mats[i].m * 4);
        }
    }
    for (int32_t u = 0; tc.lane_x && u < max_lanes; ++u) { // lane u's five rows in a buffer of its own
        cr.lane_x[u] = std::make_unique<dev_mem>(size_t(lane_rows) * k * 4);
        if (!cr.lane_x[u]->put(cr.x.data() + size_t(lane_rows) * u * k, size_t(lane_rows) * k * 4)) {
            return false;
        }
    }
    if (!cr.xc->put(cr.codes.data(), cr.codes.size()) || !cr.xs->put(cr.scales.data(), cr.scales.size() * 4) ||
        !cr.xf->put(cr.x.data(), cr.x.size() * 4) || !cr.grid->put(grids.data(), grids.size()) ||
        !cr.scale->put(sc.data(), sc.size() * 4) || !cr.err->zero()) {
        return false;
    }
    if (tc.hmix) { // each lane's slot: its five rows' F32 values, codes and scales, both halves already written
        if (k != hl::width) {
            return false;
        }
        cr.slots = std::make_unique<dev_mem>(size_t(max_lanes) * hl::slot_words * 4);
        for (int32_t u = 0; u < max_lanes; ++u) {
            const size_t base = size_t(u) * hl::slot_words * 4, r0 = size_t(lane_rows) * u;
            if (!cr.slots->put(cr.x.data() + r0 * k, size_t(lane_rows) * k * 4, base) ||
                !cr.slots->put(cr.codes.data() + r0 * k, size_t(lane_rows) * k, base + size_t(hl::code_at) * 4) ||
                !cr.slots->put(cr.scales.data() + r0 * k / 16, size_t(lane_rows) * k / 16 * 4,
                               base + size_t(hl::scale_at) * 4)) {
                return false;
            }
        }
        a.hmix = cr.slots->as<float>();
        a.hmix_peer_half = 1;
        a.err = cr.err->as<uint32_t>();
    } else {
        a.x_codes = any_q ? cr.xc->as<int8_t>() : nullptr;
        a.x_scales = any_q ? cr.xs->as<float>() : nullptr;
        a.x_f32 = any_bf ? cr.xf->as<float>() : nullptr;
    }
    a.grids = cr.grid->p;
    a.glu_codes = cr.glu_m > 0 ? cr.gc->as<int8_t>() : nullptr;
    a.glu_scales = cr.glu_m > 0 ? cr.gs->as<float>() : nullptr;
    a.glu_f32 = cr.glu_m > 0 ? cr.gf->as<float>() : nullptr;
    for (int32_t i = 0; i < n; ++i) {
        a.mats[i].scale = tc.mats[size_t(i)].scaled ? cr.scale->as<float>() : nullptr;
    }
    if (tc.route) {
        a.route.logits = 0;
        a.route.gate = 1;
        a.route.ids = cr.ids->as<int32_t>();
        a.route.wts = cr.wts->as<float>();
        a.route.sg = cr.sg->as<float>();
    }
    if (tc.conv) { // per lane: its own history and committed slot
        lcg g(name_seed(tc.name + " conv"));
        std::vector<float> taps(size_t(fn::gdn::n_channel) * fn::gdn::n_tap);
        for (float & v : taps) {
            v = static_cast<float>(g.next() - 0.5);
        }
        cr.taps = std::make_unique<dev_mem>(taps.size() * 4);
        if (!cr.taps->put(taps.data(), taps.size() * 4)) {
            return false;
        }
        for (int32_t u = 0; u < max_lanes; ++u) {
            cr.hist0[u].resize(size_t(fn::gdn::hist_version_floats));
            for (float & v : cr.hist0[u]) {
                v = static_cast<float>((g.next() - 0.5) * 8.0);
            }
            cr.cur_h[u] = static_cast<uint32_t>((2 * u + 1) % fn::gdn::n_version);
            cr.hist[u] = std::make_unique<dev_mem>(size_t(fn::gdn::n_version) * fn::gdn::hist_version_floats * 4);
            cr.cur[u] = std::make_unique<dev_mem>(4);
            if (!cr.cur[u]->put(&cr.cur_h[u], 4)) {
                return false;
            }
        }
        a.conv = {cr.taps->as<float>(), cr.hist[0]->as<float>(), cr.cur[0]->as<uint32_t>()};
    }
    if (tc.peer) {
        cr.peer = std::make_unique<dev_mem>(size_t(n_rows) * fn::xchg::words_half * 4);
        cr.x16_err = std::make_unique<dev_mem>(4);
    }
    cr.work_l = std::make_unique<dev_mem>(work_bytes(a));
    cr.work_j = std::make_unique<dev_mem>(joint_work_bytes(a));
    return cr.work_l->zero() && cr.work_j->zero();
}

// the F32 outputs of a snapshot's rows against FP64 on the decoded synthetic rows: a vector passes when max |err| /
// max |ref| <= 1e-5 (dense.h) or max |err| <= 1e-6 x its conditioning (the largest sum |w x| of the vector's outputs:
// an F32 sum's rounding scales with it, not with |ref|, so a cancelling output, as a one-row matrix's, passes by it;
// engineer's correction, 2026-10-23). by_cond: the vectors that passed by the conditioning bound alone. worst: the
// largest ratio, where with its conditioning
bool reference(const jrun & cr, std::map<int64_t, synth> & sets, const std::vector<uint8_t> & snap, int32_t lanes,
               double & worst, std::string & where, int32_t & by_cond) {
    const jcase & tc = cr.tc;
    const int32_t k = tc.k, rows = lane_rows * lanes;
    bool ok = true;
    size_t at = 0;
    worst = 0.0;
    by_cond = 0;
    for (size_t i = 0; i < tc.mats.size(); ++i) {
        const mspec & ms = tc.mats[i];
        const float * got = reinterpret_cast<const float *>(snap.data() + at);
        at += size_t(rows) * ms.m * 4;
        if (ms.e != epi::f32) {
            continue;
        }
        const synth & s = sets[synth_key(tc, ms.f)];
        const int32_t period = std::min(ms.period, n_synth_rows), distinct = std::min(ms.m, period);
        std::vector<double> d(size_t(rows) * distinct), da(size_t(rows) * distinct);
        const std::vector<float> & xa = ms.f == fmt::bf16 ? cr.x : cr.xv;
        std::atomic<bool> dec_ok{true};
        parallel_for(distinct, [&](int32_t r) {
            std::vector<float> wr(static_cast<size_t>(k));
            if (!quant::decode_row(to_wfmt(ms.f), s.row((r + cr.off[i]) % n_synth_rows), wr.data(), k)) {
                dec_ok = false;
                return;
            }
            for (int32_t b = 0; b < rows; ++b) {
                double t = 0.0, ta = 0.0;
                for (int32_t j = 0; j < k; ++j) {
                    t += double(wr[size_t(j)]) * double(xa[size_t(b) * k + j]);
                    ta += std::fabs(double(wr[size_t(j)]) * double(xa[size_t(b) * k + j]));
                }
                d[size_t(b) * distinct + r] = t;
                da[size_t(b) * distinct + r] = ta;
            }
        });
        ok = ok && dec_ok;
        for (int32_t b = 0; b < rows; ++b) {
            if (!cr.sent(b)) {
                continue;
            }
            const double scl = ms.scaled ? double(1.0f / (1.0f + std::exp(-0.37f * float(b - 9)))) : 1.0;
            double emax = 0.0, rmax = 0.0, amax = 0.0;
            for (int32_t r = 0; r < ms.m; ++r) {
                if (i == 0 && r == tc.nan_expert) {
                    ok = ok && std::isnan(got[size_t(b) * ms.m + r]);
                    continue;
                }
                const double w = scl * d[size_t(b) * distinct + r % period];
                const double g = got[size_t(b) * ms.m + r];
                emax = std::isfinite(g) ? std::max(emax, std::fabs(g - w)) : INFINITY;
                rmax = std::max(rmax, std::fabs(w));
                amax = std::max(amax, scl * da[size_t(b) * distinct + r % period]);
            }
            const double rel = rmax > 0.0 ? emax / rmax : emax;
            const bool rel_ok = rmax > 0.0 ? rel <= 1e-5 : emax == 0.0;
            const bool cond_ok = emax <= 1e-6 * amax;
            ok = ok && (rel_ok || cond_ok);
            by_cond += !rel_ok && cond_ok ? 1 : 0;
            if (rel > worst) {
                char buf[200];
                snprintf(buf, sizeof(buf),
                         "matrix %zu (M %d) row %d: max |err| %.3e, max |ref| %.3e, max sum |w x| %.3e "
                         "(err / sum %.3e)",
                         i, ms.m, b, emax, rmax, amax, amax > 0.0 ? emax / amax : 0.0);
                where = buf;
            }
            worst = std::max(worst, rel);
        }
    }
    return ok;
}

bool run_case(const jcase &                tc,
              const std::vector<int32_t> & rr_set,
              const std::vector<int32_t> & lane_set,
              std::map<int64_t, synth> &   sets,
              const std::vector<uint8_t> & grids) {
    jrun cr(tc);
    if (!setup(cr, sets, grids)) {
        printf("dpj-test: %s: setup failed\n  FAIL\n", tc.name.c_str());
        return false;
    }
    std::string splits;
    for (int32_t i = 0; i < cr.a.n_mats; ++i) {
        splits += std::string(i ? "," : "") + std::to_string(cr.a.mats[i].split);
    }
    printf("dpj-test: %s K %d splits %s%s%s%s%s\n", tc.name.c_str(), tc.k, splits.c_str(), tc.zero ? " (zero x)" : "",
           tc.hmix ? " (X-HMIX lanes)" : "", tc.lane_out ? " (lane outputs)" : "",
           tc.lane_x ? " (lane activations)" : "");
    bool pass = true;
    for (const int32_t lr : rr_set) {
        if (!cr.inputs(lr)) {
            printf("  R%d: inputs failed\n", lr);
            pass = false;
            continue;
        }
        for (const int32_t lanes : lane_set) {
            std::vector<uint8_t> lone, j1, j2;
            bool                 ran = cr.reset(lanes);
            for (int32_t u = 0; ran && u < lanes; ++u) {
                ran = cr.lone(u);
            }
            cr.peer_half = 0;
            cr.peer_bad  = 0;
            cr.x16_over  = 0;
            ran = ran && cr.snapshot(lanes, lone) && cr.reset(lanes) && cr.joint(lanes) && cr.snapshot(lanes, j1);
            cr.peer_half = 1;
            ran          = ran && cr.reset(lanes) && cr.joint(lanes) && cr.snapshot(lanes, j2);
            uint32_t err = 0;
            ran          = ran && cr.err->get(&err, 4) && err == 0;
            if (!ran) {
                printf("  R%d U%d: launches failed (error word %#x)\n", lr, lanes, err);
                pass = false;
                continue;
            }
            const bool        same = lone == j1, again = j1 == j2;
            const size_t      stray = cr.touched(lanes, j1) + cr.touched(lanes, j2);
            double            worst = 0.0;
            std::string       where;
            int32_t           by_cond = 0;
            const bool        ref     = reference(cr, sets, j1, lanes, worst, where, by_cond);
            const std::string bound   = by_cond > 0 ?
                                            " (" + std::to_string(by_cond) +
                                                " vectors by the conditioning bound, the rest by the ratio bound)" :
                                            " (every vector by the ratio bound)";
            printf(
                "  R%d U%d: joint against each lane at B = R %s; second joint run %s; F32 against FP64 rel %.3e "
                "%s%s",
                lr, lanes, same ? "bitwise" : "DIFFERS: FAIL", again ? "bitwise" : "DIFFERS: FAIL", worst,
                ref ? "ok" : "FAIL", ref ? bound.c_str() : "");
            if (lr < lane_rows) {
                printf("; rows R..4 %s (%zu bytes changed)", stray == 0 ? "untouched" : "WRITTEN: FAIL", stray);
            }
            printf("\n");
            if (tc.peer && tc.x16) {
                printf(
                    "      X-A from the producer in 16 bits (halves 0 and 1): peer halves %s (%zu not the host's "
                    "rounding or flag); %zu of 2 sends beyond FP16's range (flag set)%s\n",
                    cr.peer_bad == 0 ? "bitwise" : "DIFFER: FAIL", cr.peer_bad, cr.x16_over,
                    tc.over && cr.x16_over != 2 ? ": FAIL (the case must overflow)" : "");
            } else if (tc.peer) {
                printf("      X-A from the producer (halves 0 and 1): peer words %s (%zu not send_half_joint's)\n",
                       cr.peer_bad == 0 ? "bitwise" : "DIFFER: FAIL", cr.peer_bad);
            }
            if (!ref || by_cond > 0) {
                printf("      worst: %s\n", where.c_str());
            }
            if (!same) {
                size_t first = 0;
                while (first < lone.size() && lone[first] == j1[first]) {
                    ++first;
                }
                printf("      first differing byte %zu of %zu\n", first, lone.size());
            }
            pass = pass && same && again && ref && stray == 0 && cr.peer_bad == 0 && (!tc.over || cr.x16_over == 2);
        }
    }
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// FMS5: a router case with the switch on, at each U: two joint runs (bitwise each other) and every logit and the gate
// against FP64 on the same BF16 weights and F32 activations, |err| <= 2^-14 x sum |w x| (dense.h router_tc; the NaN
// expert's logit NaN). Reported, not gated: the rows whose top-10 ID set differs from FP64's top 10 (the lower ID on a
// tie), and from the host's top 10 of the launch's own logits (the epilogue's input).
bool run_case_tc(const jcase &                tc,
                 const std::vector<int32_t> & rr_set,
                 const std::vector<int32_t> & lane_set,
                 std::map<int64_t, synth> &   sets,
                 const std::vector<uint8_t> & grids) {
    jrun cr(tc);
    if (!setup(cr, sets, grids)) {
        printf("dpj-test: %s: setup failed\n  FAIL\n", tc.name.c_str());
        return false;
    }
    printf("dpj-test: %s K %d, the router on the tensor cores%s%s\n", tc.name.c_str(), tc.k, tc.zero ? " (zero x)" : "",
           tc.hmix ? " (X-HMIX lanes)" : "");
    const int32_t k = tc.k, ne = tc.mats[0].m;
    const double bound = std::ldexp(1.0, -14);
    // FP64 per distinct weight row and activation row: sum w x and sum |w x|
    const int32_t max_r = lane_rows * max_lanes;
    std::vector<std::vector<double>> d(tc.mats.size()), da(tc.mats.size());
    bool ok = true;
    for (size_t i = 0; i < tc.mats.size(); ++i) {
        const mspec & ms = tc.mats[i];
        const synth & s = sets[synth_key(tc, ms.f)];
        const int32_t period = std::min(ms.period, n_synth_rows), distinct = std::min(ms.m, period);
        d[i].assign(size_t(max_r) * distinct, 0.0);
        da[i].assign(size_t(max_r) * distinct, 0.0);
        std::atomic<bool> dec_ok{true};
        parallel_for(distinct, [&](int32_t r) {
            std::vector<float> wr(static_cast<size_t>(k));
            if (!quant::decode_row(to_wfmt(ms.f), s.row((r + cr.off[i]) % n_synth_rows), wr.data(), k)) {
                dec_ok = false;
                return;
            }
            for (int32_t b = 0; b < max_r; ++b) {
                double t = 0.0, ta = 0.0;
                for (int32_t j = 0; j < k; ++j) {
                    t += double(wr[size_t(j)]) * double(cr.x[size_t(b) * k + j]);
                    ta += std::fabs(double(wr[size_t(j)]) * double(cr.x[size_t(b) * k + j]));
                }
                d[i][size_t(b) * distinct + r] = t;
                da[i][size_t(b) * distinct + r] = ta;
            }
        });
        ok = ok && dec_ok;
    }
    if (!ok) {
        printf("  decode failed\n  FAIL\n");
        return false;
    }
    // the top 10 of v (the lower ID on a tie; NaN never taken), as a sorted ID list
    const auto top10 = [&](const auto & v) {
        std::vector<int32_t> id(static_cast<size_t>(ne));
        for (int32_t e = 0; e < ne; ++e) {
            id[size_t(e)] = e;
        }
        const auto key = [&](int32_t e) { return std::isnan(double(v(e))) ? -INFINITY : double(v(e)); };
        std::partial_sort(id.begin(), id.begin() + route_used, id.end(),
                          [&](int32_t x, int32_t y) { return key(x) > key(y) || (key(x) == key(y) && x < y); });
        id.resize(route_used);
        std::sort(id.begin(), id.end());
        return id;
    };
    bool pass = true;
    for (const int32_t lr : rr_set) {
        if (!cr.inputs(lr)) {
            printf("  R%d: inputs failed\n", lr);
            pass = false;
            continue;
        }
        for (const int32_t lanes : lane_set) {
            const int32_t        rows = lane_rows * lanes;
            std::vector<uint8_t> j1, j2;
            cr.peer_half = 0;
            bool     ran = cr.reset(lanes) && cr.joint(lanes) && cr.snapshot(lanes, j1) && cr.reset(lanes) &&
                           cr.joint(lanes) && cr.snapshot(lanes, j2);
            uint32_t err = 0;
            ran          = ran && cr.err->get(&err, 4) && err == 0;
            if (!ran) {
                printf("  R%d U%d: launches failed (error word %#x)\n", lr, lanes, err);
                pass = false;
                continue;
            }
            const bool      again = j1 == j2;
            const size_t    stray = cr.touched(lanes, j1) + cr.touched(lanes, j2);
            const float *   lg    = reinterpret_cast<const float *>(j1.data());
            const float *   gt    = lg + size_t(rows) * ne;
            const int32_t * ids   = reinterpret_cast<const int32_t *>(gt + rows);
            double          worst = 0.0;
            std::string     where;
            int32_t         bad = 0, top_ref = 0, top_own = 0, n_sent = 0;
            for (int32_t b = 0; b < rows; ++b) {
                if (!cr.sent(b)) {
                    continue;
                }
                ++n_sent;
                for (size_t i = 0; i < tc.mats.size(); ++i) {
                    const mspec & ms     = tc.mats[i];
                    const int32_t period = std::min(ms.period, n_synth_rows), distinct = std::min(ms.m, period);
                    const float * got = i == 0 ? lg + size_t(b) * ne : gt + b;
                    for (int32_t r = 0; r < ms.m; ++r) {
                        const double g = got[r];
                        if (i == 0 && r == tc.nan_expert) {
                            bad += std::isnan(g) ? 0 : 1;
                            continue;
                        }
                        const double w     = d[i][size_t(b) * distinct + r % period];
                        const double lim   = bound * da[i][size_t(b) * distinct + r % period];
                        const double e     = std::isfinite(g) ? std::fabs(g - w) : INFINITY;
                        const double ratio = lim > 0.0 ? e / lim : (e == 0.0 ? 0.0 : INFINITY);
                        bad += e <= lim ? 0 : 1;
                        if (ratio > worst || (where.empty() && e > lim)) {
                            char buf[200];
                            snprintf(buf, sizeof(buf), "matrix %zu row %d logit %d: |err| %.3e, 2^-14 sum |w x| %.3e",
                                     i, b, r, e, lim);
                            where = buf;
                        }
                        worst = std::max(worst, ratio);
                    }
                }
                const int32_t              distinct0 = std::min(std::min(tc.mats[0].period, n_synth_rows), ne);
                const int32_t              period0   = std::min(tc.mats[0].period, n_synth_rows);
                const std::vector<int32_t> want      = top10(
                    [&](int32_t e) { return e == tc.nan_expert ? NAN : d[0][size_t(b) * distinct0 + e % period0]; });
                const std::vector<int32_t> own = top10([&](int32_t e) { return double(lg[size_t(b) * ne + e]); });
                std::vector<int32_t>       have(ids + size_t(b) * route_used, ids + size_t(b + 1) * route_used);
                std::sort(have.begin(), have.end());
                top_ref += have != want ? 1 : 0;
                top_own += have != own ? 1 : 0;
            }
            printf(
                "  R%d U%d: second joint run %s; logits and gate against FP64: %d past 2^-14 sum |w x| (worst "
                "%.3f of the bound) %s; top-10 ID set differs from FP64's at %d of %d rows, from the launch's own "
                "logits' at %d (reported)",
                lr, lanes, again ? "bitwise" : "DIFFERS: FAIL", bad, worst, bad == 0 ? "ok" : "FAIL", top_ref, n_sent,
                top_own);
            if (lr < lane_rows) {
                printf("; rows R..4 %s (%zu bytes changed)", stray == 0 ? "untouched" : "WRITTEN: FAIL", stray);
            }
            printf("\n");
            if (bad > 0 || worst > 0.5) {
                printf("      worst: %s\n", where.c_str());
            }
            pass = pass && again && bad == 0 && stray == 0;
        }
    }
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// FMS6: the bound's |w| of synthetic row `row` (k values): |w|, and for Q4_K and Q5_K |c f| + |m| = max(|c f - m|,
// |c f + m|), the second decoded with each block's dmin negated
bool weight_mags(fmt f, const uint8_t * row, int32_t k, std::vector<float> & w, std::vector<float> & mag) {
    const quant::wfmt q = to_wfmt(f);
    w.resize(size_t(k));
    mag.resize(size_t(k));
    if (!quant::decode_row(q, row, w.data(), k)) {
        return false;
    }
    if (f != fmt::q4_k && f != fmt::q5_k) {
        for (int32_t j = 0; j < k; ++j) {
            mag[size_t(j)] = std::fabs(w[size_t(j)]);
        }
        return true;
    }
    const quant::wfmt_info & bi = quant::info(q);
    std::vector<uint8_t> neg(row, row + size_t(k / bi.block_values) * bi.block_bytes);
    for (size_t at = 0; at < neg.size(); at += size_t(bi.block_bytes)) {
        neg[at + 3] ^= 0x80u; // dmin (FP16 at byte 2): its sign bit
    }
    if (!quant::decode_row(q, neg.data(), mag.data(), k)) {
        return false;
    }
    for (int32_t j = 0; j < k; ++j) {
        mag[size_t(j)] = std::max(std::fabs(w[size_t(j)]), std::fabs(mag[size_t(j)]));
    }
    return true;
}

// FMS6: a case with the FP16 switch on, at each U: two joint runs (bitwise each other); each quantized F32-epilogue
// output against FP64 on the same decoded weights and activations (code x d), |err| <= 2^-8 x sum |w| |x| (dense.h
// dense_f16); each BF16 matrix's outputs bitwise the lone launches' (its path unchanged). A GLU or conv matrix's
// products are judged by the case's twin with F32 epilogues (f16_cases); here its outputs must be finite where the
// lone launches' are, their largest difference from the lone launches' reported.
bool run_case_f16(const jcase &                tc,
                  const std::vector<int32_t> & rr_set,
                  const std::vector<int32_t> & lane_set,
                  std::map<int64_t, synth> &   sets,
                  const std::vector<uint8_t> & grids) {
    jrun cr(tc);
    if (!setup(cr, sets, grids)) {
        printf("dpj-test: %s: setup failed\n  FAIL\n", tc.name.c_str());
        return false;
    }
    std::string splits;
    for (int32_t i = 0; i < cr.a.n_mats; ++i) {
        splits += std::string(i ? "," : "") + std::to_string(cr.a.mats[i].split);
    }
    printf("dpj-test: %s K %d splits %s, dense FP16%s%s%s\n", tc.name.c_str(), tc.k, splits.c_str(),
           tc.zero ? " (zero x)" : "", tc.hmix ? " (X-HMIX lanes)" : "", tc.lane_out ? " (lane outputs)" : "");
    const int32_t k = tc.k, max_r = lane_rows * max_lanes;
    const double bound = std::ldexp(1.0, -8);
    // FP64 per distinct weight row and activation row: sum w x and sum |w| |x| (quantized F32-epilogue matrices)
    std::vector<std::vector<double>> d(tc.mats.size()), da(tc.mats.size());
    bool ok = true;
    for (size_t i = 0; i < tc.mats.size(); ++i) {
        const mspec & ms = tc.mats[i];
        if (ms.f == fmt::bf16 || ms.e != epi::f32) {
            continue;
        }
        const synth & s = sets[synth_key(tc, ms.f)];
        const int32_t period = std::min(ms.period, n_synth_rows), distinct = std::min(ms.m, period);
        d[i].assign(size_t(max_r) * distinct, 0.0);
        da[i].assign(size_t(max_r) * distinct, 0.0);
        std::atomic<bool> dec_ok{true};
        parallel_for(distinct, [&](int32_t r) {
            std::vector<float> wr, wm;
            if (!weight_mags(ms.f, s.row((r + cr.off[i]) % n_synth_rows), k, wr, wm)) {
                dec_ok = false;
                return;
            }
            for (int32_t b = 0; b < max_r; ++b) {
                double t = 0.0, ta = 0.0;
                for (int32_t j = 0; j < k; ++j) {
                    const double xv = cr.xv[size_t(b) * k + j];
                    t += double(wr[size_t(j)]) * xv;
                    ta += double(wm[size_t(j)]) * std::fabs(xv);
                }
                d[i][size_t(b) * distinct + r] = t;
                da[i][size_t(b) * distinct + r] = ta;
            }
        });
        ok = ok && dec_ok;
    }
    if (!ok) {
        printf("  decode failed\n  FAIL\n");
        return false;
    }
    bool pass = true;
    for (const int32_t lr : rr_set) {
        if (!cr.inputs(lr)) {
            printf("  R%d: inputs failed\n", lr);
            pass = false;
            continue;
        }
        for (const int32_t lanes : lane_set) {
            const int32_t        rows = lane_rows * lanes;
            std::vector<uint8_t> lone, j1, j2;
            bool                 ran = cr.reset(lanes);
            for (int32_t u = 0; ran && u < lanes; ++u) {
                ran = cr.lone(u);
            }
            cr.peer_half = 0;
            cr.peer_bad  = 0;
            cr.x16_over  = 0;
            ran = ran && cr.snapshot(lanes, lone) && cr.reset(lanes) && cr.joint(lanes) && cr.snapshot(lanes, j1);
            cr.peer_half = 1;
            ran          = ran && cr.reset(lanes) && cr.joint(lanes) && cr.snapshot(lanes, j2);
            uint32_t err = 0;
            ran          = ran && cr.err->get(&err, 4) && err == 0;
            if (!ran) {
                printf("  R%d U%d: launches failed (error word %#x)\n", lr, lanes, err);
                pass = false;
                continue;
            }
            const bool   again = j1 == j2;
            const size_t stray = cr.touched(lanes, j1) + cr.touched(lanes, j2);
            double       worst = 0.0, epi_diff = 0.0, epi_max = 0.0;
            std::string  where;
            int32_t      bad = 0, bf_bad = 0, epi_bad = 0, n_q = 0, n_bf = 0, n_epi = 0;
            size_t       at = 0;
            for (size_t i = 0; i < tc.mats.size(); ++i) {
                const mspec & ms    = tc.mats[i];
                const size_t  bytes = size_t(rows) * ms.m * 4;
                const float * got   = reinterpret_cast<const float *>(j1.data() + at);
                const float * ref   = reinterpret_cast<const float *>(lone.data() + at);
                if (ms.e != epi::f32) {  // GLU (no F32 rows: the GLU's own buffers below) or conv's outputs
                    for (size_t e = 0; ms.e == epi::conv_silu && e < bytes / 4; ++e) {
                        ++n_epi;
                        if (std::isfinite(ref[e])) {
                            epi_bad += std::isfinite(got[e]) ? 0 : 1;
                            epi_diff = std::max(epi_diff, double(std::fabs(got[e] - ref[e])));
                            epi_max  = std::max(epi_max, double(std::fabs(ref[e])));
                        }
                    }
                } else if (ms.f == fmt::bf16) {
                    ++n_bf;
                    bf_bad += memcmp(got, ref, bytes) == 0 ? 0 : 1;
                } else {
                    ++n_q;
                    const int32_t period = std::min(ms.period, n_synth_rows), distinct = std::min(ms.m, period);
                    for (int32_t b = 0; b < rows; ++b) {
                        if (!cr.sent(b)) {
                            continue;
                        }
                        const double scl = ms.scaled ? double(1.0f / (1.0f + std::exp(-0.37f * float(b - 9)))) : 1.0;
                        for (int32_t o = 0; o < ms.m; ++o) {
                            const double w     = scl * d[i][size_t(b) * distinct + o % period];
                            const double lim   = bound * scl * da[i][size_t(b) * distinct + o % period];
                            const double g     = got[size_t(b) * ms.m + o];
                            const double e     = std::isfinite(g) ? std::fabs(g - w) : INFINITY;
                            const double ratio = lim > 0.0 ? e / lim : (e == 0.0 ? 0.0 : INFINITY);
                            bad += e <= lim ? 0 : 1;
                            if (ratio > worst || (where.empty() && e > lim)) {
                                char buf[200];
                                snprintf(buf, sizeof(buf),
                                         "matrix %zu row %d output %d: |err| %.3e, 2^-8 sum |w| |x| %.3e", i, b, o, e,
                                         lim);
                                where = buf;
                            }
                            worst = std::max(worst, ratio);
                        }
                    }
                }
                at += bytes;
            }
            if (cr.glu_m > 0) {  // the GLU's F32 values (after its codes and scales)
                const size_t  gf_at = at + size_t(rows) * cr.glu_m + size_t(rows) * cr.glu_m / 16 * 4;
                const float * got   = reinterpret_cast<const float *>(j1.data() + gf_at);
                const float * ref   = reinterpret_cast<const float *>(lone.data() + gf_at);
                for (size_t e = 0; e < size_t(rows) * cr.glu_m; ++e) {
                    ++n_epi;
                    if (std::isfinite(ref[e])) {
                        epi_bad += std::isfinite(got[e]) ? 0 : 1;
                        epi_diff = std::max(epi_diff, double(std::fabs(got[e] - ref[e])));
                        epi_max = std::max(epi_max, double(std::fabs(ref[e])));
                    }
                }
            }
            printf(
                "  R%d U%d: second joint run %s; quantized outputs (%d matrices) against FP64: %d past 2^-8 sum |w| "
                "|x| (worst %.3f of the bound) %s",
                lr, lanes, again ? "bitwise" : "DIFFERS: FAIL", n_q, bad, worst, bad == 0 ? "ok" : "FAIL");
            if (n_bf > 0) {
                printf("; BF16 matrices (%d) against the lone launches %s", n_bf,
                       bf_bad == 0 ? "bitwise" : "DIFFER: FAIL");
            }
            if (n_epi > 0) {
                printf(
                    "; epilogue outputs (products by the twin case) %s, max |joint - lone| %.3e of max |lone| %.3e "
                    "(reported)",
                    epi_bad == 0 ? "finite" : "NOT FINITE: FAIL", epi_diff, epi_max);
            }
            if (lr < lane_rows) {
                printf("; rows R..4 %s (%zu bytes changed)", stray == 0 ? "untouched" : "WRITTEN: FAIL", stray);
            }
            printf("\n");
            if (tc.peer) {
                printf("      X-A from the producer: peer words %s (%zu not send_half_joint's)\n",
                       cr.peer_bad == 0 ? "as the outputs" : "DIFFER: FAIL", cr.peer_bad);
            }
            if (bad > 0 || worst > 0.5) {
                printf("      worst: %s\n", where.c_str());
            }
            pass = pass && again && bad == 0 && stray == 0 && bf_bad == 0 && epi_bad == 0 && cr.peer_bad == 0 &&
                   (!tc.over || cr.x16_over == 2);
        }
    }
    printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// FMS6: the full mode's cases with a quantized matrix, the switch on, except the X-A 16-bit overflow case (activations
// x 1e6, past FP16's range by design); a case with a quantized GLU or conv matrix gets a twin with every such matrix
// plain F32 (its products then judged against FP64)
std::vector<jcase> f16_cases(const std::vector<jcase> & full) {
    std::vector<jcase> out;
    for (const jcase & c : full) {
        bool q = false, q_epi = false;
        for (const mspec & m : c.mats) {
            q = q || m.f != fmt::bf16;
            q_epi = q_epi || (m.f != fmt::bf16 && m.e != epi::f32);
        }
        if (!q || c.route || c.over) {
            continue;
        }
        out.push_back(c);
        out.back().f16 = true;
        if (q_epi) {
            jcase t = c;
            t.name += " (products as F32)";
            t.conv = false;
            for (mspec & m : t.mats) {
                if (m.e != epi::f32) {
                    m = plain(m.name, m.f, m.m, m.split);
                }
            }
            t.f16 = true;
            out.push_back(t);
        }
    }
    return out;
}

} // namespace
} // namespace fn::dense

int fnfl_dpj_test(int device, const char * mode, const char * only) {
    using namespace fn::dense;
    const bool san = strcmp(mode, "san") == 0, tcm = strcmp(mode, "tc") == 0, f16m = strcmp(mode, "f16") == 0;
    if (!san && !tcm && !f16m && strcmp(mode, "full") != 0) {
        fprintf(stderr, "dpj-test: mode is full, san, tc or f16\n");
        return 2;
    }
    if (cudaSetDevice(device) != cudaSuccess) {
        fprintf(stderr, "dpj-test: cannot use device %d\n", device);
        return 1;
    }
    std::vector<uint8_t> grids;
    if (!grid_buffer(grids)) {
        return 1;
    }
    std::map<int64_t, synth> sets;
    std::vector<jcase> cases;
    const auto conv_case = [](const char * name, fmt qkv, fmt gate, bool hmix) {
        jcase c{name,
                2560,
                {conv("attn_qkv", qkv), plain("attn_gate", gate, 3072), plain("alpha", fmt::bf16, 24),
                 plain("beta", fmt::bf16, 24)}};
        c.conv = true;
        c.hmix = hmix;
        return c;
    };
    const std::vector<mspec> router = {plain("router", fmt::bf16, 512), plain("shexp_gate", fmt::bf16, 1)};
    const auto route_case = [](const char * name, std::vector<mspec> m, double xs, int32_t nan, bool hmix) {
        jcase c{name, 2560, std::move(m)};
        c.route = true;
        c.xscale = xs;
        c.nan_expert = nan;
        c.hmix = hmix;
        return c;
    };
    const std::vector<mspec> owner = {plain("attn_q", fmt::q4_k, 6144), plain("attn_k", fmt::q4_k, 256),
                                      plain("attn_v", fmt::q4_k, 256), plain("idx_q", fmt::bf16, 512),
                                      plain("idx_k", fmt::bf16, 128)};
    // PLE's products (fn::ple::products_joint): key and value, each lane's e and outputs in its own buffers
    jcase ple{"PLE products (lane activations and outputs)",
              2560,
              {plain("ple_key", fmt::bf16, 5120), plain("ple_value", fmt::bf16, 1280)}};
    ple.lane_out = ple.lane_x = true;
    jcase ple_tail{"small BF16 pair (lane activations and outputs, tail rows)",
                   512,
                   {plain("a", fmt::bf16, 23), plain("b", fmt::bf16, 9)}};
    ple_tail.lane_out = ple_tail.lane_x = true;
    const std::vector<mspec> shx0 = {glu("gate", fmt::iq4_xs, epi::glu_gate, 1), glu("up", fmt::iq3_s, epi::glu_up, 0)};
    // FMS2 item 6: an X-A producer's send (split: 0 chosen; the send needs a split product, an unsplit one sends after)
    const auto xa_case = [](const char * name, const char * mat, fmt f, int32_t split = 0) {
        jcase c{name, 3072, {plain(mat, f, 2560, split)}};
        c.peer = true;
        return c;
    };
    // FMS2 item 7: the same send in 16 bits; over: the activations scaled so outputs pass FP16's range
    const auto xa16_case = [&](const char * name, const char * mat, fmt f, int32_t split = 0, double xs = 1.0) {
        jcase c = xa_case(name, mat, f, split);
        c.x16 = true;
        c.xscale = xs;
        c.over = xs > 1.0;
        return c;
    };
    std::vector<mspec> ties = router;
    ties[0].period = 128;
    if (tcm) { // FMS5: the router cases with the switch on
        cases.push_back(route_case("router + shared gate split", router, 0.02, -1, false));
        cases.push_back(route_case("router + shared gate unsplit",
                                   {plain("router", fmt::bf16, 512, 1), plain("shexp_gate", fmt::bf16, 1, 1)}, 0.02, -1,
                                   false));
        cases.push_back(route_case("router + shared gate (wide logits)", router, 0.2, -1, false));
        cases.push_back(route_case("router + shared gate (ties)", ties, 0.02, -1, false));
        cases.push_back(route_case("router + shared gate (NaN logit)", router, 0.02, 300, false));
        cases.push_back(route_case("router + shared gate (X-HMIX)", router, 0.02, -1, true));
        jcase rz = route_case("router + shared gate", router, 1.0, -1, false);
        rz.zero = true;
        cases.push_back(rz);
        for (jcase & c : cases) {
            c.tc = true;
        }
    } else if (san) {
        cases.push_back({"small Q4_K", 512, {plain("w", fmt::q4_k, 64)}});
        cases.push_back(conv_case("GDN input + conv (layer 10)", fmt::q6_k, fmt::q6_k, true));
        cases.push_back({"QSA input owner (layer 11)", 2560, owner});
        jcase xa{"ssm_out Q5_K split (X-A from the producer)", 3072, {plain("ssm_out", fmt::q5_k, 2560)}};
        xa.peer = true;
        cases.push_back(xa);
        cases.push_back(xa16_case("ssm_out Q5_K split (X-A from the producer, 16-bit)", "ssm_out", fmt::q5_k));
        cases.push_back({"shexp gate/up GLU (layer 0) split", 2560, shx0});
        cases.push_back({"shexp down Q2_0 (layer 3)", 320, {scaled("down", fmt::q2_0, 2560)}});
        cases.push_back(route_case("router + shared gate split", router, 0.02, -1, false));
        cases.push_back(route_case("router + shared gate (X-HMIX)", router, 0.02, -1, true));
        jcase vh{"vocabulary Q5_K (8,192 rows, X-HMIX)", 2560, {plain("vocab", fmt::q5_k, 8192)}};
        vh.hmix = vh.lane_out = true;
        cases.push_back(vh);
        cases.push_back(ple);
    } else {
        for (const fmt f :
             {fmt::q4_k, fmt::q5_k, fmt::q6_k, fmt::iq4_xs, fmt::iq3_s, fmt::q8_0, fmt::iq4_nl, fmt::q2_0}) {
            cases.push_back({std::string("small ") + fmt_name(f), 512, {plain("w", f, 64)}});
        }
        cases.push_back({"small Q8_0 (tail step)", 320, {plain("w", fmt::q8_0, 64)}});
        // FMS2 item 4: the adversarial set (extremize): every format, split and unsplit, tail steps and the conv path
        for (const fmt f :
             {fmt::q4_k, fmt::q5_k, fmt::q6_k, fmt::iq4_xs, fmt::iq3_s, fmt::q8_0, fmt::iq4_nl, fmt::q2_0}) {
            jcase e{std::string("extreme ") + fmt_name(f) + " split", 3072, {plain("w", f, 256)}};
            e.extreme = true;
            cases.push_back(e);
            jcase u{std::string("extreme ") + fmt_name(f) + " unsplit", 2560, {plain("w", f, 128, 1)}};
            u.extreme = true;
            cases.push_back(u);
        }
        for (const fmt f : {fmt::q8_0, fmt::iq4_nl, fmt::q2_0}) {
            jcase e{std::string("extreme ") + fmt_name(f) + " (tail step)", 320, {plain("w", f, 64)}};
            e.extreme = true;
            cases.push_back(e);
        }
        cases.push_back({"small Q2_0 (tail step)", 320, {plain("w", fmt::q2_0, 64)}});
        cases.push_back({"small BF16 (tail rows)", 512, {plain("w", fmt::bf16, 22)}});
        cases.push_back({"small BF16 split", 3072, {plain("w", fmt::bf16, 64)}});
        cases.push_back(ple_tail);
        cases.push_back(ple);
        cases.push_back(conv_case("GDN input + conv (layer 10)", fmt::q6_k, fmt::q6_k, false));
        jcase ec = conv_case("extreme GDN input + conv (Q6_K / Q8_0)", fmt::q6_k, fmt::q8_0, false);
        ec.extreme = true;
        cases.push_back(ec);
        cases.push_back(conv_case("GDN input + conv (layer 0)", fmt::iq4_xs, fmt::iq3_s, false));
        cases.push_back(conv_case("GDN input + conv (layer 10, X-HMIX)", fmt::q6_k, fmt::q6_k, true));
        cases.push_back({"GDN input (layer 0)",
                         2560,
                         {plain("attn_qkv", fmt::iq4_xs, 5120), plain("attn_gate", fmt::iq3_s, 3072),
                          plain("alpha", fmt::bf16, 24), plain("beta", fmt::bf16, 24)}});
        cases.push_back({"QSA input owner (layer 11)", 2560, owner});
        jcase oh{"QSA input owner (layer 11, X-HMIX)", 2560, owner};
        oh.hmix = true;
        cases.push_back(oh);
        cases.push_back(
            {"QSA input peer (layer 11)",
             2560,
             {plain("attn_q", fmt::q4_k, 6144), plain("attn_k", fmt::q4_k, 256), plain("attn_v", fmt::q4_k, 256)}});
        cases.push_back(
            {"QSA input (layer 48, Q8_0)",
             2560,
             {plain("attn_q", fmt::q8_0, 6144), plain("attn_k", fmt::q8_0, 256), plain("attn_v", fmt::q8_0, 256),
              plain("idx_q", fmt::bf16, 512), plain("idx_k", fmt::bf16, 128)}});
        cases.push_back({"ssm_out Q5_K split", 3072, {plain("ssm_out", fmt::q5_k, 2560)}});
        cases.push_back(xa_case("ssm_out Q5_K split (X-A from the producer)", "ssm_out", fmt::q5_k));
        cases.push_back({"ssm_out Q5_K unsplit", 3072, {plain("ssm_out", fmt::q5_k, 2560, 1)}});
        cases.push_back({"ssm_out IQ3_S split", 3072, {plain("ssm_out", fmt::iq3_s, 2560)}});
        cases.push_back(xa_case("ssm_out IQ3_S split (X-A from the producer)", "ssm_out", fmt::iq3_s));
        cases.push_back({"ssm_out Q4_K split", 3072, {plain("ssm_out", fmt::q4_k, 2560)}});
        cases.push_back(xa_case("ssm_out Q4_K split (X-A from the producer)", "ssm_out", fmt::q4_k));
        cases.push_back({"attn_output Q6_K split", 3072, {plain("attn_output", fmt::q6_k, 2560)}});
        cases.push_back(xa_case("attn_output Q6_K split (X-A from the producer)", "attn_output", fmt::q6_k));
        cases.push_back({"attn_output Q8_0 split", 3072, {plain("attn_output", fmt::q8_0, 2560)}});
        cases.push_back(xa_case("attn_output Q8_0 split (X-A from the producer)", "attn_output", fmt::q8_0));
        cases.push_back({"attn_output IQ4_XS split", 3072, {plain("attn_output", fmt::iq4_xs, 2560)}});
        cases.push_back(xa_case("attn_output IQ4_XS split (X-A from the producer)", "attn_output", fmt::iq4_xs));
        cases.push_back(xa_case("ssm_out BF16 split 2 (X-A from the producer, 4-row tiles)", "ssm_out", fmt::bf16, 2));
        cases.push_back(xa16_case("ssm_out Q5_K split (X-A from the producer, 16-bit)", "ssm_out", fmt::q5_k));
        cases.push_back(xa16_case("ssm_out IQ3_S split (X-A from the producer, 16-bit)", "ssm_out", fmt::iq3_s));
        cases.push_back(xa16_case("ssm_out Q4_K split (X-A from the producer, 16-bit)", "ssm_out", fmt::q4_k));
        cases.push_back(xa16_case("attn_output Q6_K split (X-A from the producer, 16-bit)", "attn_output", fmt::q6_k));
        cases.push_back(xa16_case("attn_output Q8_0 split (X-A from the producer, 16-bit)", "attn_output", fmt::q8_0));
        cases.push_back(
            xa16_case("attn_output IQ4_XS split (X-A from the producer, 16-bit)", "attn_output", fmt::iq4_xs));
        cases.push_back(xa16_case("ssm_out BF16 split 2 (X-A from the producer, 16-bit)", "ssm_out", fmt::bf16, 2));
        cases.push_back(xa16_case("ssm_out Q5_K split (X-A 16-bit, beyond FP16: flag)", "ssm_out", fmt::q5_k, 0, 1e6));
        cases.push_back({"shexp gate/up GLU (layer 0) split", 2560, shx0});
        cases.push_back({"shexp gate/up GLU (layer 0) unsplit",
                         2560,
                         {glu("gate", fmt::iq4_xs, epi::glu_gate, 1, 1), glu("up", fmt::iq3_s, epi::glu_up, 0, 1)}});
        cases.push_back({"shexp gate/up GLU (layer 11)",
                         2560,
                         {glu("gate", fmt::q4_k, epi::glu_gate, 1), glu("up", fmt::q4_k, epi::glu_up, 0)}});
        cases.push_back({"shexp gate/up GLU (layer 48)",
                         2560,
                         {glu("up", fmt::q8_0, epi::glu_up, 1), glu("gate", fmt::q8_0, epi::glu_gate, 0)}});
        cases.push_back({"shexp gate/up GLU (Q5_K / Q6_K)",
                         2560,
                         {glu("gate", fmt::q5_k, epi::glu_gate, 1), glu("up", fmt::q6_k, epi::glu_up, 0)}});
        jcase gh{"shexp gate/up GLU (layer 0, X-HMIX)", 2560, shx0};
        gh.hmix = true;
        cases.push_back(gh);
        jcase gz{"shexp gate/up GLU (layer 0)", 2560, shx0};
        gz.zero = true;
        cases.push_back(gz);
        cases.push_back({"shexp down IQ4_NL (layer 0)", 320, {scaled("down", fmt::iq4_nl, 2560)}});
        cases.push_back({"shexp down Q2_0 (layer 3)", 320, {scaled("down", fmt::q2_0, 2560)}});
        cases.push_back({"shexp down Q8_0 (layer 48)", 320, {scaled("down", fmt::q8_0, 2560)}});
        cases.push_back({"shexp down Q2_0 unscaled", 320, {plain("down", fmt::q2_0, 2560)}});
        cases.push_back({"vocabulary Q5_K (8,192 rows)", 2560, {plain("vocab", fmt::q5_k, 8192)}});
        // the head's joint logits (fn::head::logits_joint): each lane's own buffer, X-HMIX lanes, a card's vocabulary
        jcase vh{"vocabulary Q5_K (124,160 rows, X-HMIX)", 2560, {plain("vocab", fmt::q5_k, 124160)}};
        vh.hmix = vh.lane_out = true;
        cases.push_back(vh);
        cases.push_back(route_case("router + shared gate split", router, 0.02, -1, false));
        cases.push_back(route_case("router + shared gate unsplit",
                                   {plain("router", fmt::bf16, 512, 1), plain("shexp_gate", fmt::bf16, 1, 1)}, 0.02, -1,
                                   false));
        cases.push_back(route_case("router + shared gate (wide logits)", router, 0.2, -1, false));
        cases.push_back(route_case("router + shared gate (ties)", ties, 0.02, -1, false));
        cases.push_back(route_case("router + shared gate (NaN logit)", router, 0.02, 300, false));
        cases.push_back(route_case("router + shared gate (X-HMIX)", router, 0.02, -1, true));
        jcase rz = route_case("router + shared gate", router, 1.0, -1, false);
        rz.zero = true;
        cases.push_back(rz);
    }
    if (f16m) { // FMS6
        cases = f16_cases(cases);
    }
    const std::vector<int32_t> lane_set = san    ? std::vector<int32_t>{2, 4}
                                          : f16m ? std::vector<int32_t>{3, 4}
                                                 : std::vector<int32_t>{2, 3, 4};
    // FMS13: every case at R = 2 .. 5 (san: R = 2, the ticket's sanitizer run)
    const std::vector<int32_t> rr_set   = san ? std::vector<int32_t>{ 2 } : std::vector<int32_t>{ 2, 3, 4, 5 };
    int32_t n_fail = 0;
    std::vector<std::string> names; // only: names separated by '|'
    for (std::string o = only != nullptr ? only : ""; only != nullptr;) {
        const size_t bar = o.find('|');
        names.push_back(o.substr(0, bar));
        if (bar == std::string::npos) {
            break;
        }
        o = o.substr(bar + 1);
    }
    const auto picked = [&](const std::string & n) {
        return names.empty() ||
               std::any_of(names.begin(), names.end(), [&](const std::string & f) { return n.find(f) != n.npos; });
    };
    size_t n_run = 0;
    for (const jcase & tc : cases) {
        if (picked(tc.name)) {
            const bool ok = tc.tc  ? run_case_tc(tc, rr_set, lane_set, sets, grids) :
                            tc.f16 ? run_case_f16(tc, rr_set, lane_set, sets, grids) :
                                     run_case(tc, rr_set, lane_set, sets, grids);
            n_fail += ok ? 0 : 1;
            ++n_run;
        }
    }
    // FMU4: the joint draft step's sampled pick (two devices: this one and the other; san: loopback on this one)
    if (!tcm && !f16m && picked("sampled pick")) {
        n_fail += fnfl_dpj_pick_test(device, san ? device : (device == 0 ? 1 : 0), san) > 0 ? 1 : 0;
        ++n_run;
    }
    n_fail += n_run == 0 ? 1 : 0; // a filter that matches nothing fails
    printf("dpj-test: %zu cases, %d failed\n", n_run, n_fail);
    printf("dpj-test: %s\n", n_fail == 0 ? "PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
