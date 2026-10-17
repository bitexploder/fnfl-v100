#include "fn/loader/run_load.h"
#include "fn/model/model.h"
#include "fn/runtime/layer_run.h"
#include "fn/runtime/layer_test.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using fn::runtime_layers::pair_runner;
using fn::runtime_layers::piece_time;

constexpr int32_t D = 2560;
constexpr int32_t F = 1280;
constexpr int32_t depth_bench = 4096;
constexpr int32_t depth_synth = 131072; // LP1: the synthetic history's cells
constexpr int32_t warmup = 3;
constexpr int32_t timed = 20;

bool read_rows(const std::string & dir, const std::string & name, int32_t n_pos, std::vector<float> & out) {
    // l_last-9 of the decode dump: one record a decode, [2560, 4, rows], in position order
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
    out.resize(size_t(n_pos) * 4 * D);
    return static_cast<bool>(
        f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size() * 4)));
}

double pct(std::vector<float> v, double q) {
    std::sort(v.begin(), v.end());
    const double x = q * double(v.size() - 1);
    const size_t i = static_cast<size_t>(x);
    const size_t j = std::min(i + 1, v.size() - 1);
    return v[i] + (v[j] - v[i]) * (x - double(i));
}

} // namespace

int fnfl_layer_bench(const fnfl_layer_test_args & a) {
    if (fnfl_layer_report_unbuilt("layer-bench")) {
        return fnfl_layer_not_built;
    }
    std::vector<float> rows;
    if (!read_rows(a.ref_decode, "l_last-9", depth_bench, rows)) {
        fprintf(stderr, "layer-bench: cannot read %s/l_last-9.bin\n", a.ref_decode.c_str());
        return 1;
    }
    fn::loader::load_params lp;
    lp.model_path = a.model_path;
    lp.draft_path = a.draft_path;
    lp.ple_table = a.ple_table;
    auto m = std::make_unique<fn::model::model>();
    fn::loader::run_load_report lr;
    std::string err;
    if (!fn::loader::load_for_run(lp, a.n_cells, *m, lr, err)) {
        fprintf(stderr, "layer-bench: load: %s\n", err.c_str());
        return 1;
    }
    printf("layer-bench: model loaded in %.2f s\n", lr.total_s);
    for (const std::string & line : fn::loader::ple_table_lines(lr)) {
        printf("layer-bench: %s\n", line.c_str());
    }
    pair_runner pr;
    if (!pr.init(*m, 10, 11, err)) {
        fprintf(stderr, "layer-bench: init: %s\n", err.c_str());
        return 1;
    }
    // the history: a teacher-forced pass over the 4,096 positions (layer 11 chained: its own history is FNFL's)
    std::vector<float> in[2];
    in[0].resize(size_t(4) * 4 * F);
    in[1].resize(size_t(4) * 4 * F);
    const auto stage = [&](int32_t p0, int32_t B) {
        for (int32_t b = 0; b < B; ++b) {
            for (int32_t c = 0; c < 2; ++c) {
                for (int32_t s = 0; s < 4; ++s) {
                    std::memcpy(&in[c][(size_t(b) * 4 + s) * F], &rows[(size_t(p0 + b) * 4 + s) * D + size_t(c) * F],
                                size_t(F) * 4);
                }
            }
        }
    };
    const float * const ip[2] = {in[0].data(), in[1].data()};
    int64_t step = 0;
    for (int32_t p = 0; p < depth_bench; ++step) {
        const int32_t B = std::min<int32_t>(static_cast<int32_t>(step % 4) + 1, depth_bench - p);
        stage(p, B);
        if (!pr.step(B, p, ip, nullptr, nullptr, err)) {
            fprintf(stderr, "layer-bench: history at %d: %s\n", p, err.c_str());
            return pr.not_built() ? fnfl_layer_not_built : 1;
        }
        p += B;
    }
    printf("layer-bench: history of %d positions built (layer 10 teacher-forced, layer 11 chained)\n", depth_bench);

    std::string js = "{\n \"depth\": " + std::to_string(depth_bench) + ", \"warmup\": " + std::to_string(warmup) +
                     ", \"timed\": " + std::to_string(timed) + ",\n \"graphs\": [\n";
    std::vector<piece_time> pieces[5];
    // one graph a card of the pair at rows B and depth: warm-up and timed replays into t[card] (us)
    const auto time_graphs = [&](int32_t B, int32_t depth, std::vector<float>(&t)[2]) {
        if (!pr.reset_exchange(err) || !pr.capture(B, depth, ip, err)) {
            fprintf(stderr, "layer-bench: capture B = %d at %d: %s\n", B, depth, err.c_str());
            return false;
        }
        for (int32_t i = 0; i < warmup + timed; ++i) {
            float ms[2] = {0, 0};
            if (!pr.replay(ms, err)) {
                fprintf(stderr, "layer-bench: replay B = %d at %d: %s\n", B, depth, err.c_str());
                return false;
            }
            if (i >= warmup) {
                t[0].push_back(ms[0] * 1000.0f);
                t[1].push_back(ms[1] * 1000.0f);
            }
        }
        pr.drop_graphs();
        return true;
    };
    for (int32_t B = 1; B <= 4; ++B) {
        // the rows' input values: the reference's last B rows (the timing does not depend on them beyond routing)
        stage(depth_bench - B, B);
        std::vector<float> t[2];
        if (!time_graphs(B, depth_bench, t)) {
            return pr.not_built() ? fnfl_layer_not_built : 1;
        }
        printf(
            "layer-bench: B = %d, the pair (layers 10 + 11) at depth %d, CUDA graph a card, %d replays: card 0 median "
            "%.2f us p90 %.2f; card 1 median %.2f us p90 %.2f\n",
            B, depth_bench, timed, pct(t[0], 0.5), pct(t[0], 0.9), pct(t[1], 0.5), pct(t[1], 0.9));
        char b[400];
        snprintf(
            b, sizeof(b),
            "  {\"B\": %d, \"us\": [{\"card\": 0, \"median\": %.3f, \"p90\": %.3f}, {\"card\": 1, \"median\": %.3f, "
            "\"p90\": %.3f}], \"samples\": [[",
            B, pct(t[0], 0.5), pct(t[0], 0.9), pct(t[1], 0.5), pct(t[1], 0.9));
        js += b;
        for (int32_t c = 0; c < 2; ++c) {
            for (size_t i = 0; i < t[c].size(); ++i) {
                snprintf(b, sizeof(b), "%s%.3f", i ? ", " : "", t[c][i]);
                js += b;
            }
            js += c == 0 ? "], [" : "]]}";
        }
        js += B < 4 ? ",\n" : "\n";
        // one eager step with an event after every piece
        if (!pr.reset_exchange(err) || !pr.step_timed(B, depth_bench, ip, pieces[B], err)) {
            fprintf(stderr, "layer-bench: timed step B = %d: %s\n", B, err.c_str());
            return pr.not_built() ? fnfl_layer_not_built : 1;
        }
    }
    // LP1: the synthetic history at depth_synth cells, B = 1 and 3
    js += " ],\n \"synthetic_depth\": " + std::to_string(depth_synth) + ",\n \"synthetic_graphs\": [\n";
    if (!pr.fill_synthetic(depth_synth, 0x4c5031u, err)) {
        fprintf(stderr, "layer-bench: synthetic history: %s\n", err.c_str());
        return 1;
    }
    for (const int32_t B : {1, 3}) {
        stage(depth_bench - B, B);
        std::vector<float> t[2];
        if (!time_graphs(B, depth_synth, t)) {
            return 1;
        }
        printf("layer-bench: B = %d, the pair at %d cells of synthetic history, CUDA graph a card, %d replays: card 0 "
               "median %.2f us p90 %.2f; card 1 median %.2f us p90 %.2f\n",
               B, depth_synth, timed, pct(t[0], 0.5), pct(t[0], 0.9), pct(t[1], 0.5), pct(t[1], 0.9));
        char b[400];
        snprintf(
            b, sizeof(b),
            "  {\"B\": %d, \"us\": [{\"card\": 0, \"median\": %.3f, \"p90\": %.3f}, {\"card\": 1, \"median\": %.3f, "
            "\"p90\": %.3f}], \"samples\": [[",
            B, pct(t[0], 0.5), pct(t[0], 0.9), pct(t[1], 0.5), pct(t[1], 0.9));
        js += b;
        for (int32_t c = 0; c < 2; ++c) {
            for (size_t i = 0; i < t[c].size(); ++i) {
                snprintf(b, sizeof(b), "%s%.3f", i ? ", " : "", t[c][i]);
                js += b;
            }
            js += c == 0 ? "], [" : "]]}";
        }
        js += B < 3 ? ",\n" : "\n";
    }
    js += " ],\n \"pieces\": [\n";
    bool first = true;
    for (int32_t B = 1; B <= 4; ++B) {
        for (const piece_time & p : pieces[B]) {
            char b[200];
            snprintf(b, sizeof(b), "%s  {\"B\": %d, \"card\": %d, \"layer\": %d, \"piece\": \"%s\", \"us\": %.3f}",
                     first ? "" : ",\n", B, p.card, p.layer, p.piece.c_str(), p.ms * 1000.0f);
            js += b;
            first = false;
        }
    }
    js += "\n ]\n}\n";
    FILE * f = fopen((a.out_dir + "/layer-bench.json").c_str(), "w");
    if (f != nullptr) {
        fputs(js.c_str(), f);
        fclose(f);
    }
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m->device[c]);
        cudaDeviceSynchronize();
    }
    printf("layer-bench: done (%s/layer-bench.json)\n", a.out_dir.c_str());
    return 0;
}
