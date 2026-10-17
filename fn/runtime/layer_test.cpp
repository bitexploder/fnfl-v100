#include "fn/runtime/layer_test.h"

#include "fn/loader/run_load.h"
#include "fn/loader/sha256.h"
#include "fn/model/layer.h"
#include "fn/model/model.h"
#include "fn/runtime/layer_run.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using fn::runtime_layers::layer_record;
using fn::runtime_layers::pair_runner;

constexpr int32_t D = 2560;
constexpr int32_t F = 1280;
constexpr int32_t L0 = 10, L1 = 11;

// ---- the reference: ref_dump's records (index.tsv and <name>.bin) ----

struct ref_tensor {
    std::vector<uint8_t> data;
    std::vector<int64_t> at;    // per position: byte offset of its row (plane 0)
    std::vector<int64_t> len;   // per position: elements in its row
    std::vector<int64_t> plane; // per position: byte offset of plane 1 (indexer_top_k only), else -1
    bool ok = false;
};

bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    out.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    return static_cast<bool>(f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size())));
}

// rows of a record: ne[2] for the [n, 4 or heads, rows] tensors and indexer_top_k's [width, rows, 1, 2] uses ne[1]
ref_tensor load_ref(const std::string & dir, const std::string & name, int32_t n_pos) {
    ref_tensor t;
    t.at.assign(n_pos, -1);
    t.len.assign(n_pos, 0);
    t.plane.assign(n_pos, -1);
    if (!read_file(dir + "/" + name + ".bin", t.data)) {
        fprintf(stderr, "layer-test: cannot read %s/%s.bin\n", dir.c_str(), name.c_str());
        return t;
    }
    std::ifstream idx(dir + "/index.tsv");
    std::string line;
    std::getline(idx, line);         // header
    std::map<int64_t, int64_t> seen; // decode -> rows already placed (prompt tiles)
    while (std::getline(idx, line)) {
        std::istringstream ss(line);
        std::string nm, type;
        int64_t dec = 0, tok0 = 0, call = 0, rows = 0, off = 0, bytes = 0, ne[4] = {};
        ss >> nm >> dec >> tok0 >> call >> rows >> off >> bytes >> ne[0] >> ne[1] >> ne[2] >> ne[3] >> type;
        if (nm != name) {
            continue;
        }
        const int64_t t0 = tok0 + seen[dec];
        seen[dec] += rows;
        const int64_t per_row = name == "indexer_top_k-11" ? ne[0] : (bytes / 4) / rows;
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t p = t0 + r;
            if (p < 0 || p >= n_pos) {
                continue;
            }
            t.at[p] = off + r * per_row * 4;
            t.len[p] = per_row;
            if (name == "indexer_top_k-11") {
                t.plane[p] = off + (rows * ne[0] + r * ne[0]) * 4;
            }
        }
    }
    t.ok = std::all_of(t.at.begin(), t.at.end(), [](int64_t v) { return v >= 0; });
    if (!t.ok) {
        fprintf(stderr, "layer-test: %s does not cover the %d positions\n", name.c_str(), n_pos);
    }
    return t;
}

const float * f32(const ref_tensor & t, int32_t p) {
    return reinterpret_cast<const float *>(t.data.data() + t.at[p]); // the dump's F32 rows
}

const int32_t * i32(const ref_tensor & t, int32_t p, bool plane1 = false) {
    return reinterpret_cast<const int32_t *>(t.data.data() + (plane1 ? t.plane[p] : t.at[p])); // its I32 rows
}

double rel_err(const float * a, const float * ref, int64_t n) {
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        num = std::max(num, std::fabs(double(a[i]) - double(ref[i])));
        den = std::max(den, std::fabs(double(ref[i])));
    }
    return den > 0.0 ? num / den : num;
}

struct stats {
    double p50 = 0, p99 = 0, max = 0;
    int32_t argmax = -1;
};

stats stats_of(const std::vector<double> & e) {
    stats s;
    if (e.empty()) {
        return s;
    }
    std::vector<double> v(e);
    std::sort(v.begin(), v.end());
    // numpy's default percentile (linear interpolation)
    const auto pct = [&](double q) {
        const double x = q * double(v.size() - 1);
        const size_t i = static_cast<size_t>(std::floor(x));
        const size_t j = std::min(i + 1, v.size() - 1);
        return v[i] + (v[j] - v[i]) * (x - double(i));
    };
    s.p50 = pct(0.50);
    s.p99 = pct(0.99);
    s.max = v.back();
    s.argmax = static_cast<int32_t>(std::max_element(e.begin(), e.end()) - e.begin());
    return s;
}

std::string json_stats(const stats & s) {
    char b[200];
    snprintf(b, sizeof(b), "{\"p50\": %.6e, \"p99\": %.6e, \"max\": %.6e, \"argmax\": %d}", s.p50, s.p99, s.max,
             s.argmax);
    return b;
}

// ---- FNFL's rows from the records ----

// l_last = x + inject * out per stream (the next down2's fmaf), both cards' halves assembled
void l_last_row(const layer_record * const rec[2], int32_t b, float * out) {
    for (int32_t c = 0; c < 2; ++c) {
        for (int32_t s = 0; s < 4; ++s) {
            for (int32_t i = 0; i < F; ++i) {
                out[s * D + c * F + i] =
                    std::fmaf(rec[c]->inject[b * 4 + s], rec[c]->out_ffn[b * F + i], rec[c]->x[(b * 4 + s) * F + i]);
            }
        }
    }
}

void half_row(const layer_record * const rec[2], int32_t b, bool attn, float * out) {
    for (int32_t c = 0; c < 2; ++c) {
        const float * h = attn ? rec[c]->out_attn : rec[c]->out_ffn;
        std::memcpy(out + int64_t(c) * F, h + int64_t(b) * F, int64_t(F) * 4);
    }
}

struct pass_out {
    std::vector<double> e_l[2], e_attn[2], e_ffn[2], e_mix_attn[2], e_mix_ffn[2];
    std::vector<fn::loader::digest> pos_hash;  // per position: every recorded output of both layers
    std::vector<std::vector<int32_t>> ids[2];  // per position, the router owner's 10
    std::vector<std::set<int32_t>> sel;        // layer 11's selected cells per position
    std::vector<std::vector<float>> logits[2]; // the owner's router scores per position
    uint32_t err_words = 0;
    uint32_t kernel_err = 0; // attend's and select's error words and EX1b's status, OR'd
    int64_t peer_route_mismatch = 0, peer_cells_mismatch = 0;
    fn::loader::sha256 hash;
    bool ok = true;
};

struct refs {
    ref_tensor in[2];  // l_last-9, l_last-10 (the inputs)
    ref_tensor out[2]; // l_last-10, l_last-11
    ref_tensor attn[2], ffn[2];
    ref_tensor topk[2], logits[2];
    ref_tensor sel, score;
    ref_tensor mixed[2]; // hc_mixed (debug positions only)
};

// one pass over n positions in rows of B = 1, 2, 3, 4, 1, ... (fixed_b > 0: every step fixed_b rows); chained: layer
// 11 fed by layer 10's own output. dump: FNFL's rows of the first 64 positions.
bool run_pass(pair_runner & pr, const refs & R, int32_t n_pos, bool chained, FILE * dump, FILE * gdn_dump,
              pass_out & po, std::string & err, int32_t fixed_b = 0) {
    if (!pr.reset(err)) {
        return false;
    }
    for (int32_t k = 0; k < 2; ++k) {
        po.e_l[k].clear();
        po.e_attn[k].clear();
        po.e_ffn[k].clear();
        po.ids[k].clear();
        po.logits[k].clear();
    }
    po.sel.clear();
    po.pos_hash.clear();
    std::vector<float> in0[2], in1[2];
    for (int32_t c = 0; c < 2; ++c) {
        in0[c].resize(size_t(4) * 4 * F);
        in1[c].resize(size_t(4) * 4 * F);
    }
    // pinned records, [layer][card]
    void * pinned = nullptr;
    if (cudaMallocHost(&pinned, sizeof(layer_record) * 4) != cudaSuccess) {
        err = "pinned records";
        return false;
    }
    std::unique_ptr<void, decltype(&cudaFreeHost)> hold(pinned, &cudaFreeHost);
    layer_record * recs = static_cast<layer_record *>(pinned); // plain records, written by the copies
    for (int32_t i = 0; i < 4; ++i) {
        new (&recs[i]) layer_record();
    }
    layer_record * const rec[2][2] = {{&recs[0], &recs[1]}, {&recs[2], &recs[3]}};
    std::vector<float> row(size_t(4) * D), full(D);
    int64_t step = 0;
    for (int32_t p = 0; p < n_pos; ++step) {
        const int32_t B = std::min<int32_t>(fixed_b > 0 ? fixed_b : static_cast<int32_t>(step % 4) + 1, n_pos - p);
        std::vector<fn::loader::sha256> ph(static_cast<size_t>(B));
        const auto hash = [&](int32_t b, const void * d, size_t n) {
            po.hash.update(d, n);
            ph[size_t(b)].update(d, n);
        };
        for (int32_t b = 0; b < B; ++b) {
            const float * r0 = f32(R.in[0], p + b);
            const float * r1 = f32(R.in[1], p + b);
            for (int32_t c = 0; c < 2; ++c) {
                for (int32_t s = 0; s < 4; ++s) {
                    std::memcpy(&in0[c][(size_t(b) * 4 + s) * F], r0 + int64_t(s) * D + int64_t(c) * F, int64_t(F) * 4);
                    std::memcpy(&in1[c][(size_t(b) * 4 + s) * F], r1 + int64_t(s) * D + int64_t(c) * F, int64_t(F) * 4);
                }
            }
        }
        const float * const i0[2] = {in0[0].data(), in0[1].data()};
        const float * const i1[2] = {in1[0].data(), in1[1].data()};
        if (!pr.step(B, p, i0, chained ? nullptr : i1, rec, err)) {
            err = "position " + std::to_string(p) + " (B = " + std::to_string(B) + "): " + err;
            return false;
        }
        for (int32_t k = 0; k < 2; ++k) {
            const layer_record * const lr[2] = {rec[k][0], rec[k][1]};
            const fn::model::layer_weights & w0 = pr.weights(k, 0);
            const int32_t ro = w0.router_owner;
            for (int32_t c = 0; c < 2; ++c) {
                po.err_words |= lr[c]->err;
                po.kernel_err |= lr[c]->attn_err | lr[c]->sel_err | lr[c]->ex_status;
            }
            for (int32_t b = 0; b < B; ++b) {
                const int32_t pos = p + b;
                l_last_row(lr, b, row.data());
                po.e_l[k].push_back(rel_err(row.data(), f32(R.out[k], pos), int64_t{4} * D));
                const bool dump_pos = dump != nullptr && pos < 64;
                if (dump_pos) {
                    fwrite(row.data(), 4, row.size(), dump);
                }
                hash(b, row.data(), row.size() * 4);
                half_row(lr, b, true, full.data());
                po.e_attn[k].push_back(rel_err(full.data(), f32(R.attn[k], pos), D));
                if (dump_pos) {
                    fwrite(full.data(), 4, D, dump);
                }
                hash(b, full.data(), int64_t(D) * 4);
                half_row(lr, b, false, full.data());
                po.e_ffn[k].push_back(rel_err(full.data(), f32(R.ffn[k], pos), D));
                if (dump_pos) {
                    fwrite(full.data(), 4, D, dump);
                    fwrite(lr[0]->mixed_attn + size_t(b) * D, 4, D, dump);
                    fwrite(lr[0]->mixed_ffn + size_t(b) * D, 4, D, dump);
                }
                hash(b, full.data(), int64_t(D) * 4);
                hash(b, lr[0]->mixed_attn + size_t(b) * D, int64_t(D) * 4);
                hash(b, lr[0]->mixed_ffn + size_t(b) * D, int64_t(D) * 4);
                // routing: the router owner's experts, and the other card's (its own router, XP1's V2) equal to them
                std::vector<int32_t> ids(lr[ro]->ids + int64_t(b) * 10, lr[ro]->ids + int64_t(b) * 10 + 10);
                po.ids[k].push_back(ids);
                po.logits[k].emplace_back(lr[ro]->logits + size_t(b) * 512, lr[ro]->logits + size_t(b) * 512 + 512);
                if (std::memcmp(lr[0]->ids + int64_t(b) * 10, lr[1]->ids + int64_t(b) * 10, 40) != 0 ||
                    std::memcmp(lr[0]->wts + int64_t(b) * 10, lr[1]->wts + int64_t(b) * 10, 40) != 0) {
                    po.peer_route_mismatch++;
                }
                hash(b, lr[ro]->ids + int64_t(b) * 10, 40);
                hash(b, lr[ro]->wts + int64_t(b) * 10, 40);
                if (k == 1) {
                    // row b's cells from the body: each block's 4 cells, then its tail cells
                    const int32_t io = w0.index_owner;
                    const uint32_t * row = lr[io]->sel_body + lr[io]->sel_off[b] / 4;
                    const int32_t row_words = (lr[io]->sel_off[b + 1] - lr[io]->sel_off[b]) / 4;
                    std::set<int32_t> cells;
                    for (uint32_t j = 0; j < row[0]; ++j) {
                        for (int32_t r = 0; r < 4; ++r) {
                            cells.insert(int32_t(row[2 + j]) * 4 + r);
                        }
                    }
                    for (uint32_t j = 0; j < row[1]; ++j) {
                        cells.insert(int32_t(row[2 + row[0] + j]));
                    }
                    po.sel.push_back(cells);
                    hash(b, row, size_t(row_words) * 4);
                }
            }
            // the peer's body (X-SELECT) equal to the owner's
            if (k == 1 && (lr[0]->sel_words != lr[1]->sel_words ||
                           std::memcmp(lr[0]->sel_off, lr[1]->sel_off, size_t(B + 1) * 4) != 0 ||
                           std::memcmp(lr[0]->sel_body, lr[1]->sel_body, size_t(lr[0]->sel_words) * 4) != 0)) {
                po.peer_cells_mismatch++;
            }
        }
        for (int32_t b = 0; b < B; ++b) {
            po.pos_hash.push_back(ph[size_t(b)].finish());
        }
        // debug: the GDN layer's intermediates of both cards for the first 64 positions, per step and card: conv
        // (post-SiLU) [B][5120], z [B][3072], alpha and beta [B][24] raw, gated [B][3072]
        if (gdn_dump != nullptr && p < 64) {
            for (int32_t c = 0; c < 2; ++c) {
                const layer_record & g = *rec[0][c];
                fwrite(g.gdn_conv, 4, size_t(B) * 5120, gdn_dump);
                fwrite(g.gdn_z, 4, size_t(B) * 3072, gdn_dump);
                fwrite(g.gdn_alpha, 4, size_t(B) * 24, gdn_dump);
                fwrite(g.gdn_beta, 4, size_t(B) * 24, gdn_dump);
                fwrite(g.gdn_gated, 4, size_t(B) * 3072, gdn_dump);
            }
        }
        p += B;
    }
    return true;
}

// The yardstick: the pinned engine's prompt-against-decode spread, from ref-spread.json (R2's ref_spread.py over the
// same dumps): a tensor's p50, p99 and max, and select-11's rows_differ. A minimal reader of that file's flat layout.
bool spread_number(const std::string & js, const std::string & obj, const std::string & key, double & out) {
    const size_t o = js.find("\"" + obj + "\"");
    if (o == std::string::npos) {
        return false;
    }
    const size_t k = js.find("\"" + key + "\"", o);
    const size_t end = js.find('}', o);
    if (k == std::string::npos || (end != std::string::npos && k > end && key != "rows_differ")) {
        return false;
    }
    const size_t c = js.find(':', k);
    if (c == std::string::npos) {
        return false;
    }
    out = std::strtod(js.c_str() + c + 1, nullptr);
    return true;
}

bool spread_stats(const std::string & js, const std::string & name, stats & s) {
    return spread_number(js, name, "p50", s.p50) && spread_number(js, name, "p99", s.p99) &&
           spread_number(js, name, "max", s.max);
}

} // namespace

bool fnfl_layer_report_unbuilt(const char * tool) {
    bool any = false;
    for (const int32_t layer : {L0, L1}) {
        const std::vector<std::string> pieces = fn::model::unbuilt_pieces(layer);
        any = any || !pieces.empty();
        for (const std::string & p : pieces) {
            printf("%s: not_built: layer %d: %s\n", tool, layer, p.c_str());
        }
    }
    if (any) {
        printf("%s: not_built (no reference read, no model loaded)\n", tool);
    }
    return any;
}

int fnfl_layer_test(const fnfl_layer_test_args & a) {
    if (fnfl_layer_report_unbuilt("layer-test")) {
        return fnfl_layer_not_built;
    }
    const int32_t N = a.san ? std::min(a.n_pos, 64) : a.n_pos;
    // the references (decode: the checks; prompt: the yardstick)
    refs R, P;
    const char * l_names[3] = {"l_last-9", "l_last-10", "l_last-11"};
    R.in[0] = load_ref(a.ref_decode, l_names[0], N);
    R.in[1] = load_ref(a.ref_decode, l_names[1], N);
    R.out[0] = R.in[1];
    R.out[1] = load_ref(a.ref_decode, l_names[2], N);
    R.attn[0] = load_ref(a.ref_decode, "linear_attn_out-10", N);
    R.attn[1] = load_ref(a.ref_decode, "attn_output-11", N);
    R.ffn[0] = load_ref(a.ref_decode, "ffn_out-10", N);
    R.ffn[1] = load_ref(a.ref_decode, "ffn_out-11", N);
    R.topk[0] = load_ref(a.ref_decode, "ffn_moe_topk-10", N);
    R.topk[1] = load_ref(a.ref_decode, "ffn_moe_topk-11", N);
    R.logits[0] = load_ref(a.ref_decode, "ffn_moe_logits-10", N);
    R.logits[1] = load_ref(a.ref_decode, "ffn_moe_logits-11", N);
    R.sel = load_ref(a.ref_decode, "indexer_top_k-11", N);
    R.score = load_ref(a.ref_decode, "indexer_score-11", N);
    bool refs_ok = R.in[0].ok && R.in[1].ok && R.out[1].ok && R.attn[0].ok && R.attn[1].ok && R.ffn[0].ok &&
                   R.ffn[1].ok && R.topk[0].ok && R.topk[1].ok && R.logits[0].ok && R.logits[1].ok && R.sel.ok &&
                   R.score.ok;
    // prompt mode: the router's and the indexer's scores (each row's own error); the output statistics' yardstick is
    // ref-spread.json
    std::string spread_js;
    if (!a.san) {
        P.logits[0] = load_ref(a.ref_prompt, "ffn_moe_logits-10", N);
        P.logits[1] = load_ref(a.ref_prompt, "ffn_moe_logits-11", N);
        P.score = load_ref(a.ref_prompt, "indexer_score-11", N);
        refs_ok = refs_ok && P.logits[0].ok && P.logits[1].ok && P.score.ok;
        std::vector<uint8_t> bytes;
        if (!read_file(a.spread_json, bytes)) {
            fprintf(stderr, "layer-test: cannot read %s\n", a.spread_json.c_str());
            return 1;
        }
        spread_js.assign(bytes.begin(), bytes.end());
        printf("layer-test: yardstick %s (sha256 %s)\n", a.spread_json.c_str(),
               [&] {
                   fn::loader::sha256 h;
                   h.update(bytes.data(), bytes.size());
                   return fn::loader::hex(h.finish());
               }()
                   .c_str());
    }
    if (!refs_ok) {
        return 1;
    }
    printf("layer-test: references read: %d positions, decode %s, prompt %s\n", N, a.ref_decode.c_str(),
           a.san ? "-" : a.ref_prompt.c_str());

    // the model
    fn::loader::load_params lp;
    lp.model_path = a.model_path;
    lp.draft_path = a.draft_path;
    lp.ple_table = a.ple_table;
    auto m = std::make_unique<fn::model::model>();
    fn::loader::run_load_report lr;
    std::string err;
    // the sanitizer run reserves state for 8,192 cells (its 64 positions need few; the tool needs device memory)
    const int32_t n_cells = a.san ? 8192 : a.n_cells;
    if (!fn::loader::load_for_run(lp, n_cells, *m, lr, err)) {
        fprintf(stderr, "layer-test: load: %s\n", err.c_str());
        return 1;
    }
    printf(
        "layer-test: model loaded in %.2f s (open %.2f, fill %.2f, reserve %.2f) at %d cells; free %.3f / %.3f GiB\n",
        lr.total_s, lr.open_s, lr.fill_s, lr.reserve_s, n_cells, double(lr.free_after[0]) / 1073741824.0,
        double(lr.free_after[1]) / 1073741824.0);
    for (const std::string & line : fn::loader::ple_table_lines(lr)) {
        printf("layer-test: %s\n", line.c_str());
    }
    pair_runner pr;
    if (!pr.init(*m, L0, L1, err)) {
        fprintf(stderr, "layer-test: init: %s\n", err.c_str());
        return 1;
    }

    // the passes
    pass_out A, B, C, I;
    const auto t0 = std::chrono::steady_clock::now();
    FILE * dump = fopen((a.out_dir + "/fnfl-rows-forced.bin").c_str(), "wb");
    FILE * gdn_dump = fopen((a.out_dir + "/fnfl-gdn-debug.bin").c_str(), "wb");
    if (!run_pass(pr, R, N, false, dump, gdn_dump, A, err)) {
        fprintf(stderr, "layer-test: pass 1: %s\n", err.c_str());
        if (dump != nullptr) {
            fclose(dump);
        }
        if (gdn_dump != nullptr) {
            fclose(gdn_dump);
        }
        return pr.not_built() ? fnfl_layer_not_built : 1;
    }
    if (dump != nullptr) {
        fclose(dump);
    }
    if (gdn_dump != nullptr) {
        fclose(gdn_dump);
    }
    const double pass_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const fn::loader::digest hA = A.hash.finish();
    printf("layer-test: pass 1 (teacher-forced) %.1f s, %d positions; error words 0x%x; peer route mismatches %lld, "
           "peer cell mismatches %lld\n",
           pass_s, N, A.err_words, (long long)A.peer_route_mismatch, (long long)A.peer_cells_mismatch);
    bool determinism = true;
    fn::loader::digest hB = hA;
    if (!a.san) {
        if (!run_pass(pr, R, N, false, nullptr, nullptr, B, err)) {
            fprintf(stderr, "layer-test: pass 2: %s\n", err.c_str());
            return pr.not_built() ? fnfl_layer_not_built : 1;
        }
        hB = B.hash.finish();
        determinism = hA == hB;
        if (!run_pass(pr, R, N, true, nullptr, nullptr, C, err)) {
            fprintf(stderr, "layer-test: pass 3: %s\n", err.c_str());
            return pr.not_built() ? fnfl_layer_not_built : 1;
        }
        C.hash.finish();
    }
    // batch invariance (NC1 section 2): the first 512 positions again in steps of B = 1, teacher-forced from empty
    // state; every recorded output bitwise equal to the cycled pass's at the same position
    const int32_t n_inv = std::min(N, 512);
    int32_t inv_differ = 0, inv_first = -1;
    if (!a.san) {
        if (!run_pass(pr, R, n_inv, false, nullptr, nullptr, I, err, 1)) {
            fprintf(stderr, "layer-test: batch-invariance pass: %s\n", err.c_str());
            return pr.not_built() ? fnfl_layer_not_built : 1;
        }
        I.hash.finish();
        for (int32_t p = 0; p < n_inv; ++p) {
            if (I.pos_hash[size_t(p)] != A.pos_hash[size_t(p)]) {
                inv_differ++;
                inv_first = inv_first < 0 ? p : inv_first;
            }
        }
        printf(
            "layer-test: batch invariance, positions 0..%d at B = 1 against the cycled pass: %d positions differ%s\n",
            n_inv - 1, inv_differ,
            inv_differ == 0 ? " (bitwise identical)" : (", first " + std::to_string(inv_first)).c_str());
    }
    printf("layer-test: pass 1 sha256 %s, pass 2 %s: %s\n", fn::loader::hex(hA).c_str(), fn::loader::hex(hB).c_str(),
           determinism ? "bitwise identical" : "DIFFERENT");

    // statistics
    const char * lname[2] = {"l_last-10", "l_last-11"};
    const char * aname[2] = {"linear_attn_out-10", "attn_output-11"};
    const char * fname[2] = {"ffn_out-10", "ffn_out-11"};
    bool pass = A.err_words == 0 && B.err_words == 0 && C.err_words == 0 && I.err_words == 0 && A.kernel_err == 0 &&
                B.kernel_err == 0 && C.kernel_err == 0 && I.kernel_err == 0 && A.peer_route_mismatch == 0 &&
                A.peer_cells_mismatch == 0 && determinism && inv_differ == 0;
    printf("layer-test: kernel error words (attend, select, EX1b status) 0x%x 0x%x 0x%x 0x%x\n", A.kernel_err,
           B.kernel_err, C.kernel_err, I.kernel_err);
    std::string js = "{\n \"positions\": " + std::to_string(N) + ",\n";
    FILE * pe = fopen((a.out_dir + "/errors-per-position.tsv").c_str(), "w");
    if (pe != nullptr) {
        fprintf(pe, "pos\tl10\tl11\tattn10\tattn11\tffn10\tffn11\tl11_chained\n");
        for (int32_t p = 0; p < N; ++p) {
            fprintf(pe, "%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n", p, A.e_l[0][p], A.e_l[1][p], A.e_attn[0][p],
                    A.e_attn[1][p], A.e_ffn[0][p], A.e_ffn[1][p], a.san ? 0.0 : C.e_l[1][p]);
        }
        fclose(pe);
    }
    for (int32_t k = 0; k < 2; ++k) {
        const stats sl = stats_of(A.e_l[k]), sa = stats_of(A.e_attn[k]), sf = stats_of(A.e_ffn[k]);
        printf("\nlayer %d (%s), %d positions, max |FNFL - ref| over the row / max |ref|:\n", k == 0 ? L0 : L1,
               k == 0 ? "GDN" : "QSA", N);
        printf("  %-20s FNFL p50 %.3e p99 %.3e max %.3e (position %d)\n", lname[k], sl.p50, sl.p99, sl.max, sl.argmax);
        printf("  %-20s FNFL p50 %.3e p99 %.3e max %.3e (position %d)\n", aname[k], sa.p50, sa.p99, sa.max, sa.argmax);
        printf("  %-20s FNFL p50 %.3e p99 %.3e max %.3e (position %d)\n", fname[k], sf.p50, sf.p99, sf.max, sf.argmax);
        js += " \"layer" + std::to_string(k == 0 ? L0 : L1) + "\": {\"" + lname[k] + "\": " + json_stats(sl) + ", \"" +
              aname[k] + "\": " + json_stats(sa) + ", \"" + fname[k] + "\": " + json_stats(sf);
        if (!a.san) {
            stats yl, ya, yf;
            if (!spread_stats(spread_js, lname[k], yl) || !spread_stats(spread_js, aname[k], ya) ||
                !spread_stats(spread_js, fname[k], yf)) {
                fprintf(stderr, "layer-test: the yardstick lacks layer %d's statistics\n", k == 0 ? L0 : L1);
                return 1;
            }
            const auto within = [](const stats & f, const stats & y) {
                return f.p50 <= 2 * y.p50 && f.p99 <= 2 * y.p99 && f.max <= 2 * y.max;
            };
            const bool ol = within(sl, yl), oa = within(sa, ya), of = within(sf, yf);
            pass = pass && ol && oa && of;
            printf("  yardstick (prompt against decode, the pinned engine's own spread; each FNFL statistic <= 2x):\n");
            printf("  %-20s ref   p50 %.3e p99 %.3e max %.3e -> %s\n", lname[k], yl.p50, yl.p99, yl.max,
                   ol ? "WITHIN" : "OUTSIDE");
            printf("  %-20s ref   p50 %.3e p99 %.3e max %.3e -> %s\n", aname[k], ya.p50, ya.p99, ya.max,
                   oa ? "WITHIN" : "OUTSIDE");
            printf("  %-20s ref   p50 %.3e p99 %.3e max %.3e -> %s\n", fname[k], yf.p50, yf.p99, yf.max,
                   of ? "WITHIN" : "OUTSIDE");
            js += ", \"yardstick\": {\"" + std::string(lname[k]) + "\": " + json_stats(yl) + ", \"" + aname[k] +
                  "\": " + json_stats(ya) + ", \"" + fname[k] + "\": " + json_stats(yf) + "}, \"within_2x\": [" +
                  (ol ? "true" : "false") + ", " + (oa ? "true" : "false") + ", " + (of ? "true" : "false") + "]";
        }
        js += "},\n";
    }
    if (!a.san) {
        const stats sc = stats_of(C.e_l[1]);
        printf("\nchained (information): layer 11 fed by FNFL's own layer 10: l_last-11 p50 %.3e p99 %.3e max %.3e\n",
               sc.p50, sc.p99, sc.max);
        js += " \"chained_l_last-11\": " + json_stats(sc) + ",\n";
    }

    // routing
    for (int32_t k = 0; k < 2; ++k) {
        int64_t differ = 0, explained = 0, order_differs = 0;
        std::string first_rows;
        for (int32_t p = 0; p < N; ++p) {
            const int32_t * rid = i32(R.topk[k], p);
            const std::vector<int32_t> & fid = A.ids[k][p];
            const std::set<int32_t> rs(rid, rid + 10), fs(fid.begin(), fid.end());
            if (!std::equal(fid.begin(), fid.end(), rid)) {
                order_differs++;
            }
            if (rs == fs) {
                continue;
            }
            differ++;
            if (a.san) {
                continue;
            }
            // the reference's scores of the experts that differ, against the router's own error at this row
            const float * rl = f32(R.logits[k], p);
            const float * pl = f32(P.logits[k], p);
            double own = 0.0;
            for (int32_t e = 0; e < 512; ++e) {
                own = std::max(own, std::fabs(double(rl[e]) - double(pl[e])));
            }
            double lo_in = 1e30,
                   hi_out = -1e30; // the lowest reference score kept by one side only, the highest dropped
            for (int32_t e : rs) {
                if (!fs.count(e)) {
                    lo_in = std::min(lo_in, double(rl[e]));
                }
            }
            for (int32_t e : fs) {
                if (!rs.count(e)) {
                    hi_out = std::max(hi_out, double(rl[e]));
                }
            }
            const double gap = lo_in - hi_out;
            const bool ok = gap <= 2 * own; // NC1 section 3: within 2x the router's own error at the row
            explained += ok ? 1 : 0;
            if (first_rows.size() < 4000) {
                char b[256];
                snprintf(b, sizeof(b),
                         "    position %d: reference cutoff scores %.6f (in ref only) vs %.6f (in FNFL only), "
                         "gap %.3e, router's own error at the row %.3e (2x: %.3e): %s\n",
                         p, lo_in, hi_out, gap, own, 2 * own, ok ? "within" : "NOT within");
                first_rows += b;
            }
        }
        const bool ok = a.san || differ == explained;
        pass = pass && ok;
        printf(
            "\nrouting, layer %d: %lld of %d rows differ from the reference's top 10 (%lld explained at the cutoff); "
            "order differs in %lld\n%s",
            k == 0 ? L0 : L1, (long long)differ, N, (long long)explained, (long long)order_differs, first_rows.c_str());
        js += " \"routing_layer" + std::to_string(k == 0 ? L0 : L1) +
              "\": {\"rows_differ\": " + std::to_string(differ) + ", \"explained\": " + std::to_string(explained) +
              ", \"order_differs\": " + std::to_string(order_differs) + "},\n";
    }

    // selection (layer 11)
    {
        int64_t differ = 0, explained = 0;
        std::string first_rows;
        for (int32_t p = 0; p < N; ++p) {
            const int64_t W = R.sel.len[p];
            const int32_t * rc = i32(R.sel, p);
            const int32_t * rm = i32(R.sel, p, true);
            std::set<int32_t> rs;
            for (int64_t j = 0; j < W; ++j) {
                if (rm[j] == 0) {
                    rs.insert(rc[j]);
                }
            }
            if (rs == A.sel[p]) {
                continue;
            }
            differ++;
            if (a.san) {
                continue;
            }
            const float * rsc = f32(R.score, p);
            const float * psc = f32(P.score, p);
            const int64_t nb = std::min(R.score.len[p], P.score.len[p]);
            const int64_t vis = (p + 1) / 4; // the blocks the row sees whole
            double own = 0.0;
            for (int64_t blk = 0; blk < std::min(nb, vis); ++blk) {
                own = std::max(own, std::fabs(double(rsc[blk]) - double(psc[blk])));
            }
            double lo_in = 1e30, hi_out = -1e30;
            for (int32_t c : rs) {
                if (!A.sel[p].count(c) && c / 4 < nb) {
                    lo_in = std::min(lo_in, double(rsc[c / 4]));
                }
            }
            for (int32_t c : A.sel[p]) {
                if (!rs.count(c) && c / 4 < nb) {
                    hi_out = std::max(hi_out, double(rsc[c / 4]));
                }
            }
            const double gap = lo_in - hi_out;
            const bool ok = gap <= 2 * own; // within 2x the indexer's own error at the row
            explained += ok ? 1 : 0;
            if (first_rows.size() < 4000) {
                char b[256];
                snprintf(b, sizeof(b),
                         "    position %d: reference scores at the 512th place %.6f (ref only) vs %.6f "
                         "(FNFL only), gap %.3e, indexer's own error at the row %.3e (2x: %.3e): %s\n",
                         p, lo_in, hi_out, gap, own, 2 * own, ok ? "within" : "NOT within");
                first_rows += b;
            }
        }
        // QI-11: at most 2x the pinned engine's own prompt-against-decode differing rows
        double own_rows = 0.0;
        const bool count_ok = a.san || (spread_number(spread_js, "select-11", "rows_differ", own_rows) &&
                                        double(differ) <= 2.0 * own_rows);
        const bool ok = a.san || (differ == explained && count_ok);
        pass = pass && ok;
        printf("\nselection, layer 11: %lld of %d rows differ from the reference's cells (%lld explained at the "
               "512th place); the pinned engine's own modes differ in %.0f rows (2x: %.0f): %s\n%s",
               (long long)differ, N, (long long)explained, own_rows, 2 * own_rows, count_ok ? "within" : "OUTSIDE",
               first_rows.c_str());
        js += " \"selection_layer11\": {\"rows_differ\": " + std::to_string(differ) +
              ", \"explained\": " + std::to_string(explained) +
              ", \"own_rows_differ\": " + std::to_string(static_cast<int64_t>(own_rows)) + "},\n";
    }
    js += " \"batch_invariance\": {\"positions\": " + std::to_string(a.san ? 0 : n_inv) +
          ", \"differ\": " + std::to_string(inv_differ) + ", \"first\": " + std::to_string(inv_first) +
          "},\n \"kernel_error_words\": [" + std::to_string(A.kernel_err) + ", " + std::to_string(B.kernel_err) + ", " +
          std::to_string(C.kernel_err) + ", " + std::to_string(I.kernel_err) + "],\n";
    js += " \"determinism_bitwise\": " + std::string(determinism ? "true" : "false") + ",\n \"sha256\": [\"" +
          fn::loader::hex(hA) + "\", \"" + fn::loader::hex(hB) + "\"],\n \"error_words\": [" +
          std::to_string(A.err_words) + ", " + std::to_string(B.err_words) + ", " + std::to_string(C.err_words) +
          "],\n \"peer_route_mismatch\": " + std::to_string(A.peer_route_mismatch) +
          ",\n \"peer_cells_mismatch\": " + std::to_string(A.peer_cells_mismatch) +
          ",\n \"pass_s\": " + std::to_string(pass_s) + ",\n \"load_s\": " + std::to_string(lr.total_s) +
          ",\n \"pass\": " + (pass && !a.san ? "true" : "false") + "\n}\n";
    FILE * jf = fopen((a.out_dir + (a.san ? "/layer-test-san.json" : "/layer-test.json")).c_str(), "w");
    if (jf != nullptr) {
        fputs(js.c_str(), jf);
        fclose(jf);
    }
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m->device[c]);
        cudaDeviceSynchronize();
    }
    if (a.san) {
        printf("\nlayer-test (sanitizer run, %d positions): done, error words 0x%x, kernel error words 0x%x\n", N,
               A.err_words, A.kernel_err);
        return A.err_words == 0 && A.kernel_err == 0 ? 0 : 1;
    }
    printf("\nlayer-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
