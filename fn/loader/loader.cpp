#include "fn/loader/loader.h"

#include "fn/borrowed/gguf_tensors.h"
#include "fn/kernels/experts/experts.h"
#include "fn/model/shapes.h"

#include <cuda_runtime_api.h>
#include <strings.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace fn::loader {

namespace {

using clock_type = std::chrono::steady_clock;
using model::layout;

double seconds_since(clock_type::time_point t0) {
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// One tensor's card slice: the rows (in every matrix) times the columns of placement.h, in the mapping.
struct slice_info {
    const uint8_t * base = nullptr; // the tensor's data
    int64_t row_bytes = 0;          // one GGUF row
    int64_t matrix_bytes = 0;       // ne1 rows
    int64_t n_matrix = 0;           // ne2
    int64_t n_sel = 0;              // selected rows a matrix
    int64_t row_out = 0;            // bytes of one selected row's columns
    const model::range * rows = nullptr;
    int32_t n_rows = 0;
    const model::range * cols = nullptr;
    int32_t n_cols = 0;
    int32_t blk = 1, bb = 1;

    int64_t n_slice_rows() const { return n_matrix * n_sel; }
    int64_t bytes() const { return n_slice_rows() * row_out; }
};

slice_info info_of(const model::model & m, int32_t t, int32_t card) {
    const model::tensor_place & tp = m.tp(t);
    const model::card_place & cp = tp.card[card];
    slice_info s;
    s.blk = tp.block_elems;
    s.bb = tp.block_bytes;
    s.base = m.sources[t].file->data() + m.sources[t].offset;
    s.row_bytes = tp.ne[0] / tp.block_elems * tp.block_bytes;
    s.matrix_bytes = s.row_bytes * tp.ne[1];
    s.n_matrix = tp.ne[2];
    s.rows = cp.row_count > 0 ? &model::ranges[cp.row_first] : nullptr;
    s.n_rows = cp.row_count;
    s.cols = cp.col_count > 0 ? &model::ranges[cp.col_first] : nullptr;
    s.n_cols = cp.col_count;
    for (int32_t i = 0; i < s.n_rows; ++i) {
        s.n_sel += s.rows[i].count;
    }
    for (int32_t i = 0; i < s.n_cols; ++i) {
        s.row_out += s.cols[i].count / s.blk * s.bb;
    }
    return s;
}

// Calls fn(src, len) for the slice rows [q0, q1) in slice order (matrix, row range and row, column range), merging
// runs that are adjacent in the file.
template <class F> void walk(const slice_info & s, int64_t q0, int64_t q1, F && fn) {
    if (q0 >= q1) {
        return;
    }
    int64_t m = q0 / s.n_sel, k = q0 % s.n_sel;
    int32_t ri = 0;
    while (k >= s.rows[ri].count) {
        k -= s.rows[ri].count;
        ++ri;
    }
    const uint8_t * run = nullptr;
    int64_t run_len = 0;
    for (int64_t q = q0; q < q1; ++q) {
        const uint8_t * rp = s.base + m * s.matrix_bytes + (s.rows[ri].first + k) * s.row_bytes;
        for (int32_t c = 0; c < s.n_cols; ++c) {
            const uint8_t * p = rp + s.cols[c].first / s.blk * s.bb;
            const int64_t len = s.cols[c].count / s.blk * s.bb;
            if (run != nullptr && run + run_len == p) {
                run_len += len;
            } else {
                if (run != nullptr) {
                    fn(run, run_len);
                }
                run = p;
                run_len = len;
            }
        }
        if (++k == s.rows[ri].count) {
            k = 0;
            if (++ri == s.n_rows) {
                ri = 0;
                ++m;
            }
        }
    }
    fn(run, run_len);
}

bool ex1_fmt(const char * name, experts::fmt & f) {
    static const std::pair<const char *, experts::fmt> table[] = {
        {"IQ2_XXS", experts::fmt::iq2_xxs}, {"IQ2_XS", experts::fmt::iq2_xs}, {"IQ2_S", experts::fmt::iq2_s},
        {"IQ3_XXS", experts::fmt::iq3_xxs}, {"IQ3_S", experts::fmt::iq3_s},   {"Q2_0", experts::fmt::q2_0},
        {"IQ4_NL", experts::fmt::iq4_nl},   {"Q8_0", experts::fmt::q8_0},     {"Q4_K", experts::fmt::q4_k},
        {"Q5_K", experts::fmt::q5_k},       {"Q5_1", experts::fmt::q5_1}, // ME1: the Q4 candidate's routed formats
    };
    for (const auto & e : table) {
        if (std::strcmp(e.first, name) == 0) {
            f = e.second;
            return true;
        }
    }
    return false;
}

constexpr int64_t hc_feat = fn::n_embd / 2; // 1,280 own features a stream
constexpr int32_t hc_streams = fn::n_hc;
constexpr int64_t hc_low = fn::n_hc_lowrank; // 320

// the HC layouts' row count a stream: 324 with an inject tensor, 320 without
int64_t hc_rows(const model::region_place & rg) {
    return rg.members[1] >= 0 ? hc_low + 4 : hc_low;
}

// DR1: the draft's HC tensors are Q8_0 (GGUF blocks of 32 values: an FP16 scale, then 32 int8 codes); the HC layouts
// hold them planar (hc.h): every code of the region, then every scale, each in the BF16 layout's element order
constexpr int64_t q8_block = 32, q8_block_bytes = 34;

bool hc_q8(const model::region_place & rg) {
    return std::strcmp(model::tensors[rg.members[0]].format, "Q8_0") == 0;
}

// one row of n Q8_0 blocks (GGUF bytes at src) to the planar codes at q and scales at d
void q8_split(const uint8_t * src, int64_t n, uint8_t * q, uint8_t * d) {
    for (int64_t b = 0; b < n; ++b) {
        std::memcpy(d + b * 2, src + b * q8_block_bytes, 2);
        std::memcpy(q + b * q8_block, src + b * q8_block_bytes + 2, q8_block);
    }
}

// the inverse: n GGUF blocks at dst from the planar codes q and scales d
void q8_join(const uint8_t * q, const uint8_t * d, int64_t n, uint8_t * dst) {
    for (int64_t b = 0; b < n; ++b) {
        std::memcpy(dst + b * q8_block_bytes, d + b * 2, 2);
        std::memcpy(dst + b * q8_block_bytes + 2, q + b * q8_block, q8_block);
    }
}

// A unit of a card's fill: slice rows (gguf), experts (ex1) or the whole region (hc), written at dst.
struct unit {
    int32_t region = 0;
    int64_t a = 0, b = 0;
    int64_t dst = 0;
    int64_t bytes = 0;
};

struct piece {
    int64_t lo = 0, hi = 0; // span of the card's allocation
    int32_t u0 = 0, u1 = 0; // units
};

struct card_plan {
    std::vector<unit> units;
    std::vector<piece> pieces;
};

int64_t per_expert(const model::model & m, const model::region_place & rg) {
    experts::fmt f{};
    ex1_fmt(m.tp(rg.members[0]).format, f);
    return rg.lay == layout::ex1_gate ? experts::gate_up_bytes(f) : experts::down_bytes(f);
}

card_plan plan_card(const model::model & m, int32_t card, int64_t piece_bytes) {
    card_plan cp;
    for (int32_t r = 0; r < (int32_t)m.regions[card].size(); ++r) {
        const model::region_place & rg = m.regions[card][r];
        if (rg.bytes == 0 || rg.cached || rg.alias) {
            continue;
        }
        if (rg.lay == layout::gguf) {
            const slice_info s = info_of(m, rg.members[0], card);
            const int64_t step = std::max<int64_t>(1, piece_bytes / s.row_out);
            for (int64_t q = 0; q < s.n_slice_rows(); q += step) {
                const int64_t e = std::min(s.n_slice_rows(), q + step);
                cp.units.push_back({r, q, e, rg.offset + q * s.row_out, (e - q) * s.row_out});
            }
        } else if (rg.lay == layout::ex1_gate || rg.lay == layout::ex1_down) {
            const int64_t pe = per_expert(m, rg);
            const int64_t step = std::max<int64_t>(1, piece_bytes / pe);
            for (int64_t e0 = 0; e0 < fn::n_expert; e0 += step) {
                const int64_t e1 = std::min<int64_t>(fn::n_expert, e0 + step);
                cp.units.push_back({r, e0, e1, rg.offset + e0 * pe, (e1 - e0) * pe});
            }
        } else {
            cp.units.push_back({r, 0, 1, rg.offset, rg.bytes});
        }
    }
    for (int32_t u = 0; u < (int32_t)cp.units.size(); ++u) {
        const unit & x = cp.units[u];
        if (cp.pieces.empty() || x.dst + x.bytes - cp.pieces.back().lo > piece_bytes) {
            cp.pieces.push_back({x.dst, x.dst + x.bytes, u, u + 1});
        } else {
            cp.pieces.back().hi = x.dst + x.bytes;
            cp.pieces.back().u1 = u + 1;
        }
    }
    return cp;
}

// Write one unit's bytes at h (the unit's dst in host memory). repack: set when the unit ran an EX1 pack.
bool fill_unit(const model::model & m, int32_t card, const unit & u, uint8_t * h, bool & repack) {
    const model::region_place & rg = m.regions[card][u.region];
    repack = false;
    switch (rg.lay) {
        case layout::gguf: {
            const slice_info s = info_of(m, rg.members[0], card);
            uint8_t * d = h;
            walk(s, u.a, u.b, [&](const uint8_t * p, int64_t len) {
                std::memcpy(d, p, (size_t)len);
                d += len;
            });
            return d - h == u.bytes;
        }
        case layout::hc_down: {
            const int64_t R = hc_rows(rg);
            const bool q8 = hc_q8(rg);
            const int64_t n_codes = hc_streams * R * hc_feat;
            for (int32_t mi = 0; mi < 2 && rg.members[mi] >= 0; ++mi) {
                const slice_info s = info_of(m, rg.members[mi], card);
                const int64_t row0 = mi == 0 ? 0 : hc_low;
                for (int64_t j = 0; j < m.tp(rg.members[mi]).ne[1]; ++j) {
                    for (int32_t c = 0; c < hc_streams; ++c) {
                        const int64_t row = c * R + row0 + j;
                        if (q8) {
                            q8_split(s.base + j * s.row_bytes + s.cols[c].first / q8_block * q8_block_bytes,
                                     hc_feat / q8_block, h + row * hc_feat,
                                     h + n_codes + row * (hc_feat / q8_block) * 2);
                        } else {
                            const uint8_t * src = s.base + j * s.row_bytes + s.cols[c].first * 2;
                            std::memcpy(h + row * hc_feat * 2, src, hc_feat * 2);
                        }
                    }
                }
            }
            return true;
        }
        case layout::hc_up: {
            const slice_info s = info_of(m, rg.members[0], card);
            const bool q8 = hc_q8(rg);
            const int64_t n_codes = hc_feat * hc_streams * hc_low;
            for (int32_t c = 0; c < hc_streams; ++c) {
                for (int64_t f = 0; f < hc_feat; ++f) {
                    const int64_t row = f * hc_streams + c;
                    const uint8_t * src = s.base + (s.rows[c].first + f) * s.row_bytes;
                    if (q8) {
                        q8_split(src, hc_low / q8_block, h + row * hc_low, h + n_codes + row * (hc_low / q8_block) * 2);
                    } else {
                        std::memcpy(h + row * hc_low * 2, src, hc_low * 2);
                    }
                }
            }
            return true;
        }
        case layout::ex1_gate: {
            experts::fmt f{};
            ex1_fmt(m.tp(rg.members[0]).format, f);
            const slice_info g = info_of(m, rg.members[0], card), up = info_of(m, rg.members[1], card);
            const int64_t pe = experts::gate_up_bytes(f);
            repack = true;
            for (int64_t e = u.a; e < u.b; ++e) {
                if (!experts::pack_gate_up(f, g.base + e * g.matrix_bytes + g.rows[0].first * g.row_bytes,
                                           up.base + e * up.matrix_bytes + up.rows[0].first * up.row_bytes,
                                           (size_t)g.row_bytes, h + (e - u.a) * pe)) {
                    return false;
                }
            }
            return true;
        }
        case layout::ex1_down: {
            experts::fmt f{};
            ex1_fmt(m.tp(rg.members[0]).format, f);
            const slice_info s = info_of(m, rg.members[0], card);
            const int64_t pe = experts::down_bytes(f);
            repack = true;
            for (int64_t e = u.a; e < u.b; ++e) {
                if (!experts::pack_down(f, s.base + e * s.matrix_bytes + s.cols[0].first / s.blk * s.bb,
                                        (size_t)s.row_bytes, h + (e - u.a) * pe)) {
                    return false;
                }
            }
            return true;
        }
        default: return false;
    }
}

// ---- open: the directory against the placement ----

int32_t layout_rank(layout l) {
    return l == layout::hc_inject || l == layout::ex1_up ? 1 : 0;
}

bool check_shapes(int32_t t, std::string & err) {
    const model::tensor_place & tp = model::tensors[t];
    const auto bad = [&](const char * why) {
        err = std::string(tp.name) + ": " + why;
        return false;
    };
    for (int32_t card = 0; card < 2; ++card) {
        const model::card_place & cp = tp.card[card];
        if (cp.row_count == 0) {
            continue;
        }
        const model::range * rows = &model::ranges[cp.row_first];
        const model::range * cols = &model::ranges[cp.col_first];
        switch (tp.lay) {
            case layout::hc_down:
            case layout::hc_inject:
                if ((std::strcmp(tp.format, "BF16") != 0 && std::strcmp(tp.format, "Q8_0") != 0) ||
                    tp.ne[0] != 4 * fn::n_embd || cp.col_count != hc_streams || cp.row_count != 1 ||
                    rows[0].count != tp.ne[1] || tp.ne[1] != (tp.lay == layout::hc_down ? hc_low : 4)) {
                    return bad("not the HC down/inject shape");
                }
                for (int32_t c = 0; c < hc_streams; ++c) {
                    if (cols[c].first != c * fn::n_embd + card * hc_feat || cols[c].count != hc_feat) {
                        return bad("HC columns are not the card's features of each stream");
                    }
                }
                break;
            case layout::hc_up:
                if ((std::strcmp(tp.format, "BF16") != 0 && std::strcmp(tp.format, "Q8_0") != 0) ||
                    tp.ne[0] != hc_low || tp.ne[1] != 4 * fn::n_embd || cp.row_count != hc_streams ||
                    cp.col_count != 1 || cols[0].count != tp.ne[0]) {
                    return bad("not the HC up shape");
                }
                for (int32_t c = 0; c < hc_streams; ++c) {
                    if (rows[c].first != c * fn::n_embd + card * hc_feat || rows[c].count != hc_feat) {
                        return bad("HC rows are not the card's features of each stream");
                    }
                }
                break;
            case layout::ex1_gate:
            case layout::ex1_up:
                if (tp.ne[0] != fn::n_embd || tp.ne[1] != fn::n_ff_expert || tp.ne[2] != fn::n_expert ||
                    cp.row_count != 1 || rows[0].first != card * experts::n_ff_card ||
                    rows[0].count != experts::n_ff_card || cp.col_count != 1 || cols[0].count != tp.ne[0]) {
                    return bad("not the EX1 gate/up shape");
                }
                break;
            case layout::ex1_down:
                if (tp.ne[0] != fn::n_ff_expert || tp.ne[1] != fn::n_embd || tp.ne[2] != fn::n_expert ||
                    cp.row_count != 1 || rows[0].count != tp.ne[1] || cp.col_count != 1 ||
                    cols[0].first != card * experts::n_ff_card || cols[0].count != experts::n_ff_card) {
                    return bad("not the EX1 down shape");
                }
                break;
            default: break;
        }
    }
    return true;
}

// ME1: a region of one GGUF-layout tensor that both cards hold whole, the same rows and columns (the routers, the
// shared-expert gates and the norms: never repacked in place, so one copy serves both logical cards)
bool replicated(const model::region_place & rg) {
    if (rg.lay != layout::gguf || rg.members[1] >= 0) {
        return false;
    }
    const model::tensor_place & tp = model::tensors[rg.members[0]];
    const model::card_place & a = tp.card[0];
    const model::card_place & b = tp.card[1];
    if (a.row_count == 0 || a.row_count != b.row_count || a.col_count != b.col_count || a.bytes != b.bytes) {
        return false;
    }
    const auto same = [](const model::range & x, const model::range & y) {
        return x.first == y.first && x.count == y.count;
    };
    for (int32_t i = 0; i < a.row_count; ++i) {
        if (!same(model::ranges[a.row_first + i], model::ranges[b.row_first + i])) {
            return false;
        }
    }
    for (int32_t i = 0; i < a.col_count; ++i) {
        if (!same(model::ranges[a.col_first + i], model::ranges[b.col_first + i])) {
            return false;
        }
    }
    return true;
}

bool plan_regions(model::model & m, std::string & err) {
    // region members from the table
    std::vector<std::array<int32_t, 2>> members(model::n_regions, {-1, -1});
    std::vector<layout> lay(model::n_regions, layout::gguf);
    m.region_of.assign(model::n_tensors, -1);
    for (int32_t t = 0; t < model::n_tensors; ++t) {
        const model::tensor_place & tp = model::tensors[t];
        if (tp.lay == layout::host) {
            continue;
        }
        if (tp.region < 0 || tp.region >= model::n_regions || members[tp.region][layout_rank(tp.lay)] >= 0) {
            err = std::string(tp.name) + ": bad region";
            return false;
        }
        members[tp.region][layout_rank(tp.lay)] = t;
        if (layout_rank(tp.lay) == 0) {
            lay[tp.region] = tp.lay;
        }
        m.region_of[t] = tp.region;
    }
    for (int32_t card = 0; card < 2; ++card) {
        m.regions[card].assign(model::n_regions, model::region_place());
        int64_t off = 0, sum = 0;
        for (int32_t r = 0; r < model::n_regions; ++r) {
            model::region_place & rg = m.regions[card][r];
            rg.lay = lay[r];
            rg.members[0] = members[r][0];
            rg.members[1] = members[r][1];
            if (rg.members[0] < 0) {
                err = "region " + std::to_string(r) + " has no first member";
                return false;
            }
            rg.bytes = m.tp(rg.members[0]).card[card].bytes;
            if (rg.members[1] >= 0) {
                rg.bytes += m.tp(rg.members[1]).card[card].bytes;
            }
            if ((rg.lay == layout::ex1_gate || rg.lay == layout::ex1_down) && rg.bytes > 0 &&
                rg.bytes != fn::n_expert * per_expert(m, rg)) {
                err = std::string(m.tp(rg.members[0]).name) + ": the EX1 region is not 512 packed experts";
                return false;
            }
            if (rg.lay == layout::ex1_gate &&
                (rg.members[1] < 0 ||
                 std::strcmp(m.tp(rg.members[0]).format, m.tp(rg.members[1]).format) != 0)) {
                err = std::string(m.tp(rg.members[0]).name) + ": gate and up formats differ";
                return false;
            }
            sum += rg.bytes;
            // ME1: the cache's experts and card 1's replicated copies on one device take no device bytes
            rg.cached = m.expert_cache && (rg.lay == layout::ex1_gate || rg.lay == layout::ex1_down) &&
                        model::components[m.tp(rg.members[0]).component].layer < fn::n_layer;
            rg.alias = m.single && card == 1 && replicated(rg);
            if (rg.cached || rg.alias) {
                rg.offset = rg.alias ? m.regions[0][r].offset : -1;
                continue;
            }
            rg.offset = off;
            off += (rg.bytes + model::region_align - 1) / model::region_align * model::region_align;
        }
        m.weight_bytes[card] = sum;
        m.weight_alloc[card] = off;
        if (sum != m.placed_card_bytes[card]) { // ME7: the placement's, with the draft's experts as resolved
            err = "card " + std::to_string(card) + ": regions hold " + std::to_string(sum) + " bytes, the placement " +
                  std::to_string(m.placed_card_bytes[card]);
            return false;
        }
    }
    return true;
}

// ME7: the MTP draft's routed expert tensors (layer fn::n_layer), whose format the draft file decides
bool draft_expert_tensor(const std::string & name) {
    const std::string pre = "blk." + std::to_string(fn::n_layer) + ".";
    return name == pre + "ffn_gate_exps.weight" || name == pre + "ffn_up_exps.weight" ||
           name == pre + "ffn_down_exps.weight";
}

// ME7: a draft routed expert tensor's entry for this load. Q8_0: placement.h's, unchanged. Q2_0 (the opt-in Q2 draft:
// work/ME7's splice): GGUF's Q2_0 geometry, and each card's packed bytes (512 experts of the card's half: gate and up
// each half of gate_up_bytes, down down_bytes), with the cards' totals moved by the difference. Any other format is
// refused by name.
bool resolve_draft_expert(model::model & m, int32_t t, const fnfl_gguf_tensor & g, std::string & err) {
    model::tensor_place & tp = m.place[size_t(t)];
    if (strcasecmp(g.type_name.c_str(), "Q8_0") == 0) {
        return true;
    }
    if (strcasecmp(g.type_name.c_str(), "Q2_0") != 0) {
        err = "draft tensor '" + g.name + "' is " + g.type_name + ": the draft's routed experts are Q8_0 or Q2_0";
        return false;
    }
    const experts::fmt f = experts::fmt::q2_0;
    const int64_t per = tp.lay == layout::ex1_down ? experts::down_bytes(f) : experts::gate_up_bytes(f) / 2;
    if (per <= 0 || (tp.lay != layout::ex1_down && tp.lay != layout::ex1_gate && tp.lay != layout::ex1_up)) {
        err = "draft tensor '" + g.name + "': no Q2_0 packing for its layout";
        return false;
    }
    tp.format = "Q2_0";
    tp.block_elems = 64;
    tp.block_bytes = 18;
    tp.stored_bytes = tp.ne[0] / 64 * 18 * tp.ne[1] * tp.ne[2];
    for (int32_t card = 0; card < 2; ++card) {
        const int64_t bytes = fn::n_expert * per;
        m.placed_card_bytes[card] += bytes - tp.card[card].bytes;
        tp.card[card].bytes = bytes;
    }
    return true;
}

} // namespace

bool open_model(const load_params & p, model::model & m, load_times & t, std::string & err) {
    const auto t0 = clock_type::now();
    m.device[0] = p.device[0];
    m.device[1] = p.device[1];
    m.single = p.device[0] == p.device[1];
    m.expert_cache = p.expert_cache;
    m.kv_host = p.kv_host;
    m.kv_window = p.kv_window;
    if (m.kv_host && !m.single && !m.expert_cache) {
        err = "two-device host KV needs the expert cache (--cached-tp)";
        return false;
    }
    std::vector<std::string> paths;
    if (!fnfl_gguf_shard_paths(p.model_path, paths, err)) {
        return false;
    }
    paths.push_back(p.draft_path);

    std::unordered_map<std::string, int32_t> index;
    for (int32_t t2 = 0; t2 < model::n_tensors; ++t2) {
        index[model::tensors[t2].name] = t2;
    }
    m.sources.assign(model::n_tensors, model::tensor_source());
    std::vector<bool> seen(model::n_tensors, false);
    // ME7: this load's entries, the draft's routed experts resolved from its file below
    m.place.assign(model::tensors, model::tensors + model::n_tensors);
    m.placed_card_bytes[0] = model::weight_card_bytes[0];
    m.placed_card_bytes[1] = model::weight_card_bytes[1];
    std::string draft_formats;
    for (const std::string & path : paths) {
        const bool draft_file = &path == &paths.back();
        fnfl_gguf_dir dir;
        if (!fnfl_gguf_read_dir(path, dir, err)) {
            return false;
        }
        auto f = std::make_unique<model::mapped_file>();
        if (!f->open(path, err)) {
            return false;
        }
        if (dir.file_bytes > f->size()) {
            err = "'" + path + "' is shorter than its tensor directory";
            return false;
        }
        for (const fnfl_gguf_tensor & g : dir.tensors) {
            const auto it = index.find(g.name);
            if (it == index.end()) {
                err = "GGUF tensor '" + g.name + "' in '" + path + "' has no placement";
                return false;
            }
            if (draft_file && draft_expert_tensor(g.name)) {
                if (!resolve_draft_expert(m, it->second, g, err)) {
                    return false;
                }
                draft_formats += draft_formats.empty() ? g.type_name : "," + g.type_name;
            }
            const model::tensor_place & tp = m.tp(it->second);
            if (seen[it->second] || strcasecmp(g.type_name.c_str(), tp.format) != 0 ||
                g.block_elems != tp.block_elems || g.block_bytes != tp.block_bytes || g.ne[0] != tp.ne[0] ||
                g.ne[1] != tp.ne[1] || g.ne[2] != tp.ne[2] || g.ne[3] != 1 || (int64_t)g.bytes != tp.stored_bytes) {
                err = "GGUF tensor '" + g.name + "' (" + g.type_name + ") does not match its placement";
                return false;
            }
            seen[it->second] = true;
            model::tensor_source & s = m.sources[it->second];
            s.file = f.get();
            s.offset = g.offset;
            s.row_bytes = tp.ne[0] / tp.block_elems * tp.block_bytes;
            if (tp.lay == layout::host) {
                m.ple_table = f->data() + g.offset;
                m.ple_bytes = g.bytes;
            }
        }
        m.files.push_back(std::move(f));
    }
    // ME7: the draft's routed gate, up and down in one format: Q8_0 (placed) or Q2_0 (the opt-in); a mix is refused
    {
        int32_t n_q2 = 0, n_seen = 0;
        for (int32_t t2 = 0; t2 < model::n_tensors; ++t2) {
            if (draft_expert_tensor(model::tensors[t2].name) && seen[t2]) {
                ++n_seen;
                n_q2 += std::strcmp(m.tp(t2).format, "Q2_0") == 0 ? 1 : 0;
            }
        }
        if (n_seen == 3 && n_q2 != 0 && n_q2 != 3) {
            err = "the draft's routed experts mix formats (" + draft_formats + "): gate, up and down are all Q8_0 or "
                  "all Q2_0";
            return false;
        }
        m.draft_expert_format = n_q2 == 3 ? "Q2_0" : "Q8_0";
    }
    for (int32_t t2 = 0; t2 < model::n_tensors; ++t2) {
        if (!seen[t2]) {
            err = std::string("placed tensor '") + model::tensors[t2].name + "' is in no file";
            return false;
        }
        if (!check_shapes(t2, err)) {
            return false;
        }
    }
    const int64_t kern = fnfl_gguf_read_int(paths[0], std::string(fn::arch) + ".ple.conv_kernel");
    const int64_t ngram = fnfl_gguf_read_int(paths[0], std::string(fn::arch) + ".ple.ngram_size");
    if (kern < 2 || kern > 64 || ngram < 2 || ngram > 64) {
        err = "the PLE conv_kernel and ngram_size keys are missing or out of range";
        return false;
    }
    m.ple_conv_kernel = (int32_t)kern;
    m.ple_ngram = (int32_t)ngram;
    if (!plan_regions(m, err)) {
        return false;
    }
    t.open_s = seconds_since(t0);
    return true;
}

bool expert_layer_of(const model::model & m, int32_t layer, expert_layer & out, std::string & err) {
    if (layer < 0 || layer >= fn::n_layer || m.region_of.size() != size_t(model::n_tensors)) {
        err = "expert layer " + std::to_string(layer) + ": not a target layer of an opened model";
        return false;
    }
    const std::string pre = "blk." + std::to_string(layer) + ".";
    const int32_t tg = model::model::find((pre + "ffn_gate_exps.weight").c_str());
    const int32_t td = model::model::find((pre + "ffn_down_exps.weight").c_str());
    if (tg < 0 || td < 0) {
        err = pre + ": the routed expert tensors are not placed";
        return false;
    }
    out.layer = layer;
    out.gu_region = m.region_of[size_t(tg)];
    out.dn_region = m.region_of[size_t(td)];
    if (out.gu_region < 0 || out.dn_region < 0 || m.regions[0][out.gu_region].lay != layout::ex1_gate ||
        m.regions[0][out.dn_region].lay != layout::ex1_down || !ex1_fmt(m.tp(tg).format, out.gu_fmt) ||
        !ex1_fmt(m.tp(td).format, out.dn_fmt)) {
        err = pre + ": the routed expert regions are not EX1's";
        return false;
    }
    out.gu_bytes = experts::gate_up_bytes(out.gu_fmt);
    out.dn_bytes = experts::down_bytes(out.dn_fmt);
    if (out.gu_bytes <= 0 || out.dn_bytes <= 0 || out.gu_bytes % 256 != 0 || out.dn_bytes % 256 != 0) {
        err = pre + ": an expert format the cache cannot hold";
        return false;
    }
    return true;
}

bool pack_expert_entry(const model::model & m, const expert_layer & l, int32_t e, uint8_t * dst) {
    for (int32_t card = 0; card < 2; ++card) {
        bool repack = false;
        const unit g{l.gu_region, e, e + 1, 0, l.gu_bytes};
        const unit d{l.dn_region, e, e + 1, 0, l.dn_bytes};
        if (!fill_unit(m, card, g, dst + l.gu_at(card), repack) ||
            !fill_unit(m, card, d, dst + l.dn_at(card), repack)) {
            return false;
        }
    }
    return true;
}

bool native_expert_layer(const model::model & m, const expert_layer & l, experts::cpu::native_layer & out,
                         std::string & err) {
    out = experts::cpu::native_layer();
    const std::string pre = "blk." + std::to_string(l.layer) + ".";
    const char * const names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    const uint8_t * base[3] = {nullptr, nullptr, nullptr};
    int64_t row[3] = {0, 0, 0}, mat[3] = {0, 0, 0};
    for (int32_t i = 0; i < 3; ++i) {
        const int32_t t = model::model::find((pre + names[i]).c_str());
        if (t < 0 || size_t(t) >= m.sources.size() || m.sources[size_t(t)].file == nullptr) {
            err = pre + names[i] + ": not placed or not mapped (native expert views)";
            return false;
        }
        const model::tensor_place & tp = m.tp(t);
        const model::tensor_source & src = m.sources[size_t(t)];
        experts::fmt f{};
        const bool down = i == 2;
        const int64_t ne0 = down ? experts::cpu::n_ff : experts::cpu::d_model;
        const int64_t ne1 = down ? experts::cpu::d_model : experts::cpu::n_ff;
        if (!ex1_fmt(tp.format, f) || f != (down ? l.dn_fmt : l.gu_fmt) || tp.ne[0] != ne0 || tp.ne[1] != ne1 ||
            tp.ne[2] != fn::n_expert || tp.block_elems <= 0) {
            err = pre + names[i] + ": not the EX1 shape or the layer's format (native expert views)";
            return false;
        }
        row[i] = tp.ne[0] / tp.block_elems * tp.block_bytes;
        mat[i] = row[i] * tp.ne[1];
        const int64_t want = down ? experts::cpu::dn_row_bytes(f) : experts::cpu::gu_row_bytes(f);
        if (row[i] != want || row[i] <= 0 || (src.row_bytes > 0 && src.row_bytes != row[i])) {
            err = pre + names[i] + ": a native row of " + std::to_string(row[i]) + " bytes, the CPU kernels read " +
                  std::to_string(want);
            return false;
        }
        // every expert's bytes inside the mapping
        if (!experts::cpu::native_fits(uint64_t(src.offset), uint64_t(src.file->size()), mat[i], fn::n_expert)) {
            err = pre + names[i] + ": its 512 experts run past the end of " + src.file->path();
            return false;
        }
        base[i] = src.file->data() + src.offset;
    }
    if (row[0] != row[1] || mat[0] != mat[1]) {
        err = pre + ": gate and up rows differ (native expert views)";
        return false;
    }
    out.gate = base[0];
    out.up = base[1];
    out.down = base[2];
    out.gu_row = row[0];
    out.gu_expert = mat[0];
    out.dn_row = row[2];
    out.dn_expert = mat[2];
    return true;
}

std::string draft_expert_line(const model::model & m) {
    int64_t now[2] = {0, 0}, q8[2] = {0, 0};
    for (int32_t t = 0; t < model::n_tensors; ++t) {
        if (draft_expert_tensor(model::tensors[t].name)) {
            for (int32_t c = 0; c < 2; ++c) {
                now[c] += m.tp(t).card[c].bytes;
                q8[c] += model::tensors[t].card[c].bytes;
            }
        }
    }
    const double mib = 1048576.0;
    char b[512];
    std::snprintf(b, sizeof(b),
                  "MTP draft routed experts: %s (the draft file's%s); packed %.1f / %.1f MiB on cards 0 / 1 (Q8_0: "
                  "%.1f / %.1f MiB, %+.1f MiB a card); weights %.3f / %.3f GiB a card",
                  m.draft_expert_format, std::strcmp(m.draft_expert_format, "Q2_0") == 0 ? "; ME7's opt-in" : "",
                  double(now[0]) / mib, double(now[1]) / mib, double(q8[0]) / mib, double(q8[1]) / mib,
                  double(now[0] - q8[0]) / mib, double(m.weight_bytes[0]) / 1073741824.0,
                  double(m.weight_bytes[1]) / 1073741824.0);
    return b;
}

bool allocate_weights(model::model & m, std::string & err) {
    for (int32_t card = 0; card < 2; ++card) {
        if (!m.weights[card].allocate(m.device[card], (size_t)m.weight_alloc[card], err)) {
            return false;
        }
    }
    return true;
}

bool fill_weights(const load_params & p, model::model & m, load_times & t, std::string & err) {
    card_plan plans[2] = {plan_card(m, 0, p.piece_bytes), plan_card(m, 1, p.piece_bytes)};
    std::atomic<int32_t> next[2] = {{0}, {0}};
    std::atomic<bool> failed{false};
    std::mutex err_mu, copy_mu[2];
    std::string first_err;
    cudaStream_t copy_stream[2] = {nullptr, nullptr};
    std::vector<cudaEvent_t> ev_start[2], ev_end[2];
    for (int32_t card = 0; card < 2; ++card) {
        if (!cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", err) ||
            !cuda_ok(cudaStreamCreateWithFlags(&copy_stream[card], cudaStreamNonBlocking), "stream", err)) {
            return false;
        }
        const size_t n = plans[card].pieces.size();
        ev_start[card].resize(n);
        ev_end[card].resize(n);
        for (size_t i = 0; i < n; ++i) {
            if (!cuda_ok(cudaEventCreate(&ev_start[card][i]), "event", err) ||
                !cuda_ok(cudaEventCreate(&ev_end[card][i]), "event", err)) {
                return false;
            }
        }
        t.pieces[card] = (int32_t)n;
    }
    std::vector<double> read_s(2 * p.threads_per_card, 0), repack_s(2 * p.threads_per_card, 0),
        wait_s(2 * p.threads_per_card, 0);
    const auto fail = [&](const std::string & e) {
        std::lock_guard<std::mutex> lk(err_mu);
        if (!failed.exchange(true)) {
            first_err = e;
        }
    };

    const auto worker = [&](int32_t card, int32_t w) {
        const int32_t id = card * p.threads_per_card + w;
        std::string e;
        if (!cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", e)) {
            fail(e);
            return;
        }
        uint8_t * buf[2] = {nullptr, nullptr};
        cudaEvent_t done[2] = {nullptr, nullptr};
        for (int32_t b = 0; b < 2; ++b) {
            void * h = nullptr;
            if (!cuda_ok(cudaHostAlloc(&h, (size_t)p.piece_bytes, cudaHostAllocDefault), "cudaHostAlloc", e) ||
                !cuda_ok(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming), "event", e)) {
                fail(e);
                return;
            }
            buf[b] = static_cast<uint8_t *>(h);
        }
        bool used[2] = {false, false};
        int32_t iter = 0;
        for (int32_t k; !failed.load() && (k = next[card].fetch_add(1)) < (int32_t)plans[card].pieces.size(); ++iter) {
            const int32_t b = iter & 1;
            const piece & pc = plans[card].pieces[k];
            auto tw = clock_type::now();
            if (used[b] && !cuda_ok(cudaEventSynchronize(done[b]), "piece buffer wait", e)) {
                fail(e);
                break;
            }
            wait_s[id] += seconds_since(tw);
            int64_t at = pc.lo;
            for (int32_t ui = pc.u0; ui < pc.u1; ++ui) {
                const unit & u = plans[card].units[ui];
                const auto t1 = clock_type::now();
                if (u.dst > at) {
                    std::memset(buf[b] + (at - pc.lo), 0, (size_t)(u.dst - at)); // alignment gap
                }
                bool repack = false;
                if (!fill_unit(m, card, u, buf[b] + (u.dst - pc.lo), repack)) {
                    fail("fill of region " + std::to_string(u.region) + " on card " + std::to_string(card) + " failed");
                    break;
                }
                (repack ? repack_s : read_s)[id] += seconds_since(t1);
                at = u.dst + u.bytes;
            }
            {
                std::lock_guard<std::mutex> lk(copy_mu[card]);
                const bool ok = cuda_ok(cudaEventRecord(ev_start[card][k], copy_stream[card]), "record", e) &&
                                cuda_ok(cudaMemcpyAsync(m.weights[card].data() + pc.lo, buf[b], (size_t)(pc.hi - pc.lo),
                                                        cudaMemcpyHostToDevice, copy_stream[card]),
                                        "cudaMemcpyAsync", e) &&
                                cuda_ok(cudaEventRecord(ev_end[card][k], copy_stream[card]), "record", e) &&
                                cuda_ok(cudaEventRecord(done[b], copy_stream[card]), "record", e);
                if (!ok) {
                    fail(e);
                    break;
                }
            }
            used[b] = true;
        }
        for (int32_t b = 0; b < 2; ++b) {
            if (done[b] != nullptr) {
                cudaEventSynchronize(done[b]);
                cudaEventDestroy(done[b]);
            }
            if (buf[b] != nullptr) {
                cudaFreeHost(buf[b]);
            }
        }
    };

    const auto t0 = clock_type::now();
    std::vector<std::thread> threads;
    for (int32_t card = 0; card < 2; ++card) {
        for (int32_t w = 0; w < p.threads_per_card; ++w) {
            threads.emplace_back(worker, card, w);
        }
    }
    for (std::thread & th : threads) {
        th.join();
    }
    for (int32_t card = 0; card < 2; ++card) {
        cudaSetDevice(m.device[card]);
        if (!cuda_ok(cudaStreamSynchronize(copy_stream[card]), "copy stream", err)) {
            fail(err);
        }
    }
    t.fill_wall_s = seconds_since(t0);
    for (int32_t i = 0; i < 2 * p.threads_per_card; ++i) {
        t.read_busy_s += read_s[i];
        t.repack_busy_s += repack_s[i];
        t.wait_busy_s += wait_s[i];
    }
    for (int32_t card = 0; card < 2; ++card) {
        cudaSetDevice(m.device[card]);
        for (size_t i = 0; i < ev_start[card].size(); ++i) {
            float ms = 0;
            if (!failed.load() && cudaEventElapsedTime(&ms, ev_start[card][i], ev_end[card][i]) == cudaSuccess) {
                t.h2d_s[card] += ms / 1e3;
            }
            cudaEventDestroy(ev_start[card][i]);
            cudaEventDestroy(ev_end[card][i]);
        }
        for (const piece & pc : plans[card].pieces) {
            t.h2d_bytes[card] += pc.hi - pc.lo;
        }
        cudaStreamDestroy(copy_stream[card]);
    }
    if (failed.load()) {
        err = first_err;
        return false;
    }
    return true;
}

// ---- the GGUF side of the check ----

bool hash_gguf_slices(const model::model & m, int32_t threads, std::vector<card_digests> & out, double & seconds) {
    const auto t0 = clock_type::now();
    out.assign(model::n_tensors, card_digests());
    std::vector<std::pair<int32_t, int32_t>> jobs; // (tensor, card), largest first
    for (int32_t t = 0; t < model::n_tensors; ++t) {
        for (int32_t card = 0; card < 2; ++card) {
            if (m.tp(t).lay != layout::host && m.tp(t).card[card].bytes > 0) {
                jobs.emplace_back(t, card);
            }
        }
    }
    std::sort(jobs.begin(), jobs.end(), [&m](const auto & a, const auto & b) {
        return m.tp(a.first).card[a.second].bytes > m.tp(b.first).card[b.second].bytes;
    });
    std::atomic<size_t> next{0};
    const auto worker = [&]() {
        for (size_t j; (j = next.fetch_add(1)) < jobs.size();) {
            const slice_info s = info_of(m, jobs[j].first, jobs[j].second);
            sha256 h;
            walk(s, 0, s.n_slice_rows(), [&](const uint8_t * p, int64_t len) { h.update(p, (size_t)len); });
            out[jobs[j].first][jobs[j].second] = h.finish();
        }
    };
    std::vector<std::thread> ts;
    for (int32_t i = 0; i < threads; ++i) {
        ts.emplace_back(worker);
    }
    for (std::thread & th : ts) {
        th.join();
    }
    seconds = seconds_since(t0);
    return true;
}

std::string source_fingerprint(const model::model & m) {
    sha256 h;
    for (int32_t t = 0; t < model::n_tensors; ++t) {
        const model::tensor_place & tp = m.tp(t);
        h.update(tp.name, std::strlen(tp.name) + 1);
        h.update(tp.ne, sizeof(tp.ne));
        h.update(&tp.lay, sizeof(tp.lay));
        for (int32_t card = 0; card < 2; ++card) {
            const model::card_place & cp = tp.card[card];
            h.update(&cp.bytes, sizeof(cp.bytes));
            for (int32_t i = 0; i < cp.row_count; ++i) {
                h.update(&model::ranges[cp.row_first + i], sizeof(model::range));
            }
            for (int32_t i = 0; i < cp.col_count; ++i) {
                h.update(&model::ranges[cp.col_first + i], sizeof(model::range));
            }
        }
        h.update(&m.sources[t].offset, sizeof(m.sources[t].offset));
    }
    for (const auto & f : m.files) {
        struct stat st = {};
        stat(f->path().c_str(), &st);
        const int64_t id[3] = {(int64_t)st.st_size, (int64_t)st.st_mtim.tv_sec, (int64_t)st.st_mtim.tv_nsec};
        h.update(f->path().c_str(), f->path().size() + 1);
        h.update(id, sizeof(id));
    }
    return hex(h.finish());
}

bool write_hash_cache(const std::string & path, const std::string & fingerprint, const std::vector<card_digests> & d) {
    FILE * f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        return false;
    }
    std::fprintf(f, "fingerprint %s\n", fingerprint.c_str());
    for (int32_t t = 0; t < model::n_tensors && t < (int32_t)d.size(); ++t) {
        std::fprintf(f, "%s %s %s\n", model::tensors[t].name, hex(d[t][0]).c_str(), hex(d[t][1]).c_str());
    }
    return std::fclose(f) == 0;
}

bool read_hash_cache(const std::string & path, const std::string & fingerprint, std::vector<card_digests> & d) {
    FILE * f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return false;
    }
    char word[256], a[80], b[80];
    bool ok = std::fscanf(f, "%255s %79s", word, a) == 2 && std::strcmp(word, "fingerprint") == 0 && fingerprint == a;
    d.assign(model::n_tensors, card_digests());
    for (int32_t t = 0; ok && t < model::n_tensors; ++t) {
        ok = std::fscanf(f, "%255s %79s %79s", word, a, b) == 3 && std::strcmp(word, model::tensors[t].name) == 0 &&
             std::strlen(a) == 64 && std::strlen(b) == 64;
        for (int32_t i = 0; ok && i < 32; ++i) {
            unsigned x = 0, y = 0;
            ok = std::sscanf(a + 2 * i, "%2x", &x) == 1 && std::sscanf(b + 2 * i, "%2x", &y) == 1;
            d[t][0][i] = (uint8_t)x;
            d[t][1][i] = (uint8_t)y;
        }
    }
    std::fclose(f);
    return ok;
}

// ---- the device side of the check ----

namespace {

// The inverse of one expert's pack: slice_of[d] is the slice-order index of packed byte d. Derived by index probes
// (pack is a value-independent byte permutation: every source byte i is seen as the digits of i).
struct ex1_perm {
    std::vector<uint32_t> slice_of;
};

// grb: the GGUF row bytes the pack is given (gate/up: whole rows; down: the card's columns of each 2,560 rows)
bool derive_perm(experts::fmt f, bool gate_up, int64_t grb, ex1_perm & out, std::string & err) {
    const int64_t n = gate_up ? experts::gate_up_bytes(f) : experts::down_bytes(f);
    const int64_t n_src = gate_up ? n : fn::n_embd * grb;
    const int64_t srb = gate_up ? grb : n / fn::n_embd; // slice row bytes
    if (n <= 0 || n_src >= (int64_t(1) << 24) || (gate_up && 2 * experts::n_ff_card * grb != n)) {
        err = std::string("EX1 ") + experts::fmt_name(f) + ": unexpected packed size";
        return false;
    }
    std::vector<uint8_t> src((size_t)n_src), dst((size_t)n);
    const auto pack = [&]() {
        return gate_up ? experts::pack_gate_up(f, src.data(), src.data() + n / 2, (size_t)grb, dst.data())
                       : experts::pack_down(f, src.data(), (size_t)grb, dst.data());
    };
    // every packed byte written: zero sources over a 0xAA background
    std::fill(src.begin(), src.end(), uint8_t(0));
    std::fill(dst.begin(), dst.end(), uint8_t(0xAA));
    if (!pack() || std::find(dst.begin(), dst.end(), uint8_t(0xAA)) != dst.end()) {
        err = std::string("EX1 ") + experts::fmt_name(f) + ": the pack does not write every byte";
        return false;
    }
    std::vector<uint32_t> raw((size_t)n, 0);
    for (int32_t digit = 0; digit < 3; ++digit) {
        for (int64_t i = 0; i < n_src; ++i) {
            src[i] = uint8_t(i >> (8 * digit));
        }
        if (!pack()) {
            err = std::string("EX1 ") + experts::fmt_name(f) + ": pack failed";
            return false;
        }
        for (int64_t d = 0; d < n; ++d) {
            raw[d] |= uint32_t(dst[d]) << (8 * digit);
        }
    }
    out.slice_of.assign((size_t)n, 0);
    std::vector<uint8_t> seen((size_t)n, 0);
    for (int64_t d = 0; d < n; ++d) {
        const int64_t row = raw[d] / grb, cb = raw[d] % grb;
        const int64_t s = row * srb + cb;
        if (raw[d] >= n_src || cb >= srb || s >= n || seen[s]++ != 0) {
            err = std::string("EX1 ") + experts::fmt_name(f) + ": the pack is not a permutation of the slice";
            return false;
        }
        out.slice_of[d] = (uint32_t)s;
    }
    return true;
}

using perm_map = std::map<std::tuple<int32_t, bool, int64_t>, ex1_perm>;

// the pack permutations, one per (format, gate/up or down, GGUF row bytes)
bool build_perms(const model::model & m, perm_map & perms, std::string & err) {
    for (const model::region_place & rg : m.regions[0]) {
        if (rg.bytes == 0 || (rg.lay != layout::ex1_gate && rg.lay != layout::ex1_down)) {
            continue;
        }
        experts::fmt f{};
        ex1_fmt(m.tp(rg.members[0]).format, f);
        const bool gu = rg.lay == layout::ex1_gate;
        const auto key = std::make_tuple((int32_t)f, gu, m.sources[rg.members[0]].row_bytes);
        if (perms.count(key) == 0 && !derive_perm(f, gu, std::get<2>(key), perms[key], err)) {
            return false;
        }
    }
    return true;
}

// Hash every tensor's card bytes in slice order, region by region, and compare with want. device: read the bytes
// back from the cards; otherwise fill them on the host with the loader's own fill (the CPU dry run).
bool check_all(const load_params & p, const model::model & m, const std::vector<card_digests> & want, bool device,
               verify_report & rep, std::string & err) {
    const auto t0 = clock_type::now();
    if ((int32_t)want.size() != model::n_tensors) {
        err = "no GGUF slice hashes";
        return false;
    }
    perm_map perms;
    if (!build_perms(m, perms, err)) {
        return false;
    }
    rep.ex1_permutations = (int32_t)perms.size();

    std::mutex mu;
    std::atomic<bool> failed{false};
    std::string first_err;
    const auto fail = [&](const std::string & e) {
        std::lock_guard<std::mutex> lk(mu);
        if (!failed.exchange(true)) {
            first_err = e;
        }
    };
    const auto check_card = [&](int32_t card, int32_t n_threads) {
        const card_plan plan = plan_card(m, card, p.piece_bytes);
        std::vector<std::pair<int32_t, int32_t>> spans; // units of each non-empty region
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) {
            if (spans.empty() || plan.units[spans.back().first].region != plan.units[u].region) {
                spans.emplace_back(u, u + 1);
            } else {
                spans.back().second = u + 1;
            }
        }
        std::atomic<size_t> next{0};
        const auto worker = [&]() {
            std::string e;
            void * h = nullptr;
            cudaStream_t st = nullptr;
            std::vector<uint8_t> host_buf;
            if (device) {
                if (!cuda_ok(cudaSetDevice(m.device[card]), "cudaSetDevice", e) ||
                    !cuda_ok(cudaHostAlloc(&h, (size_t)p.piece_bytes, cudaHostAllocDefault), "cudaHostAlloc", e) ||
                    !cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream", e)) {
                    fail(e);
                    if (h != nullptr) {
                        cudaFreeHost(h);
                    }
                    return;
                }
            } else {
                host_buf.resize((size_t)p.piece_bytes);
                h = host_buf.data();
            }
            uint8_t * buf = static_cast<uint8_t *>(h);
            std::vector<uint8_t> rec;
            int64_t read = 0;
            for (size_t j; !failed.load() && (j = next.fetch_add(1)) < spans.size();) {
                const model::region_place & rg = m.regions[card][plan.units[spans[j].first].region];
                sha256 hs[2];
                const ex1_perm * perm = nullptr;
                if (rg.lay == layout::ex1_gate || rg.lay == layout::ex1_down) {
                    experts::fmt f{};
                    ex1_fmt(m.tp(rg.members[0]).format, f);
                    perm = &perms.at(std::make_tuple((int32_t)f, rg.lay == layout::ex1_gate,
                                                     m.sources[rg.members[0]].row_bytes));
                    rec.resize(perm->slice_of.size());
                }
                for (int32_t ui = spans[j].first; ui < spans[j].second; ++ui) {
                    const unit & u = plan.units[ui];
                    bool repack = false;
                    if (device ? !(cuda_ok(cudaMemcpyAsync(buf, m.weights[card].data() + u.dst, (size_t)u.bytes,
                                                           cudaMemcpyDeviceToHost, st),
                                           "read back", e) &&
                                   cuda_ok(cudaStreamSynchronize(st), "read back", e))
                               : !fill_unit(m, card, u, buf, repack)) {
                        fail(device ? e : "host fill failed");
                        break;
                    }
                    read += u.bytes;
                    switch (rg.lay) {
                        case layout::gguf: hs[0].update(buf, (size_t)u.bytes); break;
                        case layout::hc_down: {
                            const int64_t R = hc_rows(rg);
                            const bool q8 = hc_q8(rg);
                            const int64_t n_codes = hc_streams * R * hc_feat, nb = hc_feat / q8_block;
                            uint8_t blk[hc_feat / q8_block * q8_block_bytes];
                            for (int32_t mi = 0; mi < 2 && rg.members[mi] >= 0; ++mi) {
                                const int64_t row0 = mi == 0 ? 0 : hc_low;
                                for (int64_t r = 0; r < m.tp(rg.members[mi]).ne[1]; ++r) {
                                    for (int32_t c = 0; c < hc_streams; ++c) {
                                        const int64_t row = c * R + row0 + r;
                                        if (q8) {
                                            q8_join(buf + row * hc_feat, buf + n_codes + row * nb * 2, nb, blk);
                                            hs[mi].update(blk, sizeof(blk));
                                        } else {
                                            hs[mi].update(buf + row * hc_feat * 2, hc_feat * 2);
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case layout::hc_up: {
                            const bool q8 = hc_q8(rg);
                            const int64_t n_codes = hc_feat * hc_streams * hc_low, nb = hc_low / q8_block;
                            uint8_t blk[hc_low / q8_block * q8_block_bytes];
                            for (int32_t c = 0; c < hc_streams; ++c) {
                                for (int64_t f = 0; f < hc_feat; ++f) {
                                    const int64_t row = f * hc_streams + c;
                                    if (q8) {
                                        q8_join(buf + row * hc_low, buf + n_codes + row * nb * 2, nb, blk);
                                        hs[0].update(blk, sizeof(blk));
                                    } else {
                                        hs[0].update(buf + row * hc_low * 2, hc_low * 2);
                                    }
                                }
                            }
                            break;
                        }
                        case layout::ex1_gate:
                        case layout::ex1_down: {
                            const int64_t n = (int64_t)perm->slice_of.size();
                            for (int64_t ex = u.a; ex < u.b; ++ex) {
                                const uint8_t * pk = buf + (ex - u.a) * n;
                                for (int64_t d = 0; d < n; ++d) {
                                    rec[perm->slice_of[d]] = pk[d];
                                }
                                if (rg.lay == layout::ex1_gate) {
                                    hs[0].update(rec.data(), (size_t)n / 2);
                                    hs[1].update(rec.data() + n / 2, (size_t)n / 2);
                                } else {
                                    hs[0].update(rec.data(), (size_t)n);
                                }
                            }
                            break;
                        }
                        default: fail("unexpected layout"); break;
                    }
                }
                for (int32_t mi = 0; mi < 2 && rg.members[mi] >= 0; ++mi) {
                    const int32_t t = rg.members[mi];
                    if (m.tp(t).card[card].bytes == 0) {
                        continue;
                    }
                    const digest got = hs[mi].finish();
                    std::lock_guard<std::mutex> lk(mu);
                    rep.placed[card]++;
                    if (got == want[t][card]) {
                        rep.equal[card]++;
                    } else {
                        rep.failures.push_back(std::string(m.tp(t).name) + " card " + std::to_string(card) +
                                               ": read back " + hex(got) + ", GGUF slice " + hex(want[t][card]));
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                rep.bytes_read[card] += read;
            }
            if (device) {
                cudaStreamDestroy(st);
                cudaFreeHost(h);
            }
        };
        std::vector<std::thread> ts;
        for (int32_t i = 0; i < n_threads; ++i) {
            ts.emplace_back(worker);
        }
        for (std::thread & th : ts) {
            th.join();
        }
    };
    std::thread c0(check_card, 0, p.threads_per_card), c1(check_card, 1, p.threads_per_card);
    c0.join();
    c1.join();
    rep.seconds = seconds_since(t0);
    if (failed.load()) {
        err = first_err;
        return false;
    }
    return true;
}

} // namespace

bool verify_device(const load_params & p, const model::model & m, const std::vector<card_digests> & want,
                   verify_report & rep, std::string & err) {
    return check_all(p, m, want, true, rep, err);
}

bool verify_host(const load_params & p, const model::model & m, const std::vector<card_digests> & want,
                 verify_report & rep, std::string & err) {
    return check_all(p, m, want, false, rep, err);
}

} // namespace fn::loader
