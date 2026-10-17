#include "fn/quant/quant_test.h"

#include "fn/quant/decode.h"
#include "fn/quant/formats.h"
#include "fn/quant/synth.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using fn::quant::wfmt;

// ggml's type names, as the reference binary writes them
wfmt wfmt_of_dump_name(const std::string & n, bool & ok) {
    std::string lower = n;
    for (char & c : lower) {
        c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    wfmt f = wfmt::f32;
    ok = fn::quant::wfmt_of_name(lower.c_str(), f);
    return f;
}

bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    out.assign(s.begin(), s.end());
    return true;
}

struct fmt_tally {
    int64_t blocks = 0, mismatched = 0, first_bad = -1;
    std::set<std::string> tensors;
    size_t blocks_at = 0, f32_at = 0; // read positions in the format's files
    std::vector<uint8_t> blocks_raw, f32_raw;
    bool loaded = false, ok = true;
};

} // namespace

int fnfl_decode_check(const char * dir) {
    const std::string d = dir;
    std::ifstream man(d + "/manifest.tsv");
    if (!man) {
        fprintf(stderr, "decode-check: cannot read %s/manifest.tsv\n", dir);
        return 1;
    }
    fmt_tally t[fn::quant::n_wfmt];
    std::string line;
    std::getline(man, line); // header
    bool pass = true;
    while (std::getline(man, line)) {
        std::istringstream ls(line);
        std::string type, file, tensor;
        int32_t gguf_type = -1;
        int64_t bb = 0, bv = 0, first = 0, n = 0;
        if (!(ls >> type >> gguf_type >> bb >> bv >> file >> tensor >> first >> n)) {
            fprintf(stderr, "decode-check: bad manifest line: %s\n", line.c_str());
            return 1;
        }
        bool known = false;
        const wfmt f = wfmt_of_dump_name(type, known);
        const fn::quant::wfmt_info & in = fn::quant::info(f);
        if (!known || in.gguf_type != gguf_type || in.block_bytes != bb || in.block_values != bv) {
            fprintf(stderr, "decode-check: %s: type %d, %lld B, %lld values does not match FNFL's table\n",
                    type.c_str(), gguf_type, static_cast<long long>(bb), static_cast<long long>(bv));
            pass = false;
            continue;
        }
        fmt_tally & ft = t[static_cast<int32_t>(f)];
        if (!ft.loaded) {
            ft.loaded = true;
            if (!read_file(d + "/" + type + ".blocks", ft.blocks_raw) ||
                !read_file(d + "/" + type + ".f32", ft.f32_raw)) {
                fprintf(stderr, "decode-check: cannot read %s's files\n", type.c_str());
                return 1;
            }
        }
        if (ft.blocks_at + size_t(n) * size_t(bb) > ft.blocks_raw.size() ||
            ft.f32_at + size_t(n) * size_t(bv) * 4 > ft.f32_raw.size()) {
            fprintf(stderr, "decode-check: %s's files are shorter than the manifest\n", type.c_str());
            return 1;
        }
        std::vector<float> got(static_cast<size_t>(bv));
        for (int64_t i = 0; i < n; ++i) {
            fn::quant::decode_block(f, ft.blocks_raw.data() + ft.blocks_at, got.data());
            if (std::memcmp(got.data(), ft.f32_raw.data() + ft.f32_at, got.size() * 4) != 0) {
                if (ft.first_bad < 0) {
                    ft.first_bad = ft.blocks;
                    fprintf(stderr, "decode-check: %s %s block %lld differs from the pinned decode\n", type.c_str(),
                            tensor.c_str(), static_cast<long long>(first) + i);
                }
                ++ft.mismatched;
            }
            ft.blocks_at += size_t(bb);
            ft.f32_at += size_t(bv) * 4;
            ++ft.blocks;
        }
        ft.tensors.insert(file + ":" + tensor);
    }
    for (int32_t i = 0; i < fn::quant::n_wfmt; ++i) {
        const fmt_tally & ft = t[i];
        const wfmt f = static_cast<wfmt>(i);
        const size_t need_tensors = f == wfmt::f16 ? 1 : 2;
        const bool all_read = ft.blocks_at == ft.blocks_raw.size() && ft.f32_at == ft.f32_raw.size();
        const bool ok =
            ft.loaded && all_read && ft.mismatched == 0 && ft.blocks >= 4096 && ft.tensors.size() >= need_tensors;
        printf("decode-check %-8s %7lld blocks, %zu tensor(s), %lld differ%s\n", fn::quant::info(f).name,
               static_cast<long long>(ft.blocks), ft.tensors.size(), static_cast<long long>(ft.mismatched),
               ok ? "" : "  FAIL");
        pass = pass && ok;
    }
    printf("decode-check: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int fnfl_synth_check() {
    constexpr int64_t n = 16384;
    const float lo = fn::quant::f16_to_f32(fn::quant::f32_to_f16(1e-4f));
    const float hi = fn::quant::f16_to_f32(fn::quant::f32_to_f16(1e-1f));
    bool pass = true;
    for (int32_t i = 0; i < fn::quant::n_wfmt; ++i) {
        const wfmt f = static_cast<wfmt>(i);
        const fn::quant::wfmt_info & in = fn::quant::info(f);
        const size_t bb = static_cast<size_t>(in.block_bytes);
        std::vector<uint8_t> a(size_t(n) * bb), b(size_t(n) * bb);
        const bool made = fn::quant::synth_blocks(f, fn::quant::synth_seed(uint64_t(i)), n, a.data()) &&
                          fn::quant::synth_blocks(f, fn::quant::synth_seed(uint64_t(i)), n, b.data());
        bool same = made && a == b, finite = true, scales_ok = true, coverage = true, moments = true;
        std::vector<float> v(static_cast<size_t>(in.block_values));
        double sum = 0.0, sum2 = 0.0;
        for (int64_t k = 0; k < n; ++k) {
            fn::quant::decode_block(f, a.data() + size_t(k) * bb, v.data());
            for (float x : v) {
                finite = finite && std::isfinite(x);
                sum += x;
                sum2 += double(x) * x;
            }
        }
        // the F16 scale fields and their sign rule (synth.h)
        std::vector<size_t> scale_at;
        bool positive = true;
        switch (f) {
            case wfmt::iq2_xxs:
            case wfmt::iq2_xs:
            case wfmt::iq2_s:
            case wfmt::iq3_xxs:
            case wfmt::iq3_s:
                scale_at = {0};
                break;
            case wfmt::q4_k:
            case wfmt::q5_k:
                scale_at = {0, 2};
                break;
            case wfmt::q6_k:
                scale_at = {208};
                positive = false;
                break;
            case wfmt::q2_0:
            case wfmt::q8_0:
            case wfmt::iq4_nl:
            case wfmt::iq4_xs:
                scale_at = {0};
                positive = false;
                break;
            case wfmt::bf16:
            case wfmt::f16:
            case wfmt::f32:
                break;
        }
        int64_t n_neg = 0;
        for (size_t at : scale_at) {
            for (int64_t k = 0; k < n; ++k) {
                const uint8_t * p = a.data() + size_t(k) * bb + at;
                const uint16_t h = static_cast<uint16_t>(p[0] | (p[1] << 8));
                const float s = fn::quant::f16_to_f32(h);
                n_neg += s < 0.0f;
                scales_ok = scales_ok && ((h >> 10) & 0x1f) != 0 && std::fabs(s) >= lo && std::fabs(s) <= hi &&
                            (!positive || s > 0.0f);
            }
        }
        if (!positive) {
            scales_ok = scales_ok && n_neg > 0 && n_neg < int64_t(scale_at.size()) * n; // both signs occur
        }
        if (in.block_values > 1) {
            for (size_t at = 0; at < bb; ++at) {
                bool is_scale = false;
                for (size_t s : scale_at) {
                    is_scale = is_scale || at == s || at == s + 1;
                }
                if (is_scale) {
                    continue;
                }
                bool seen[256] = {};
                int32_t n_seen = 0;
                for (int64_t k = 0; k < n; ++k) {
                    const uint8_t x = a[size_t(k) * bb + at];
                    n_seen += seen[x] ? 0 : 1;
                    seen[x] = true;
                }
                coverage = coverage && n_seen == 256;
            }
        } else {
            const double mean = sum / double(n), sd = std::sqrt(sum2 / double(n) - mean * mean);
            moments = std::fabs(mean) < 0.05 && std::fabs(sd - 1.0) < 0.05;
        }
        const bool ok = same && finite && scales_ok && coverage && moments;
        printf("synth-check %-8s %lld blocks: deterministic %s, finite %s, scales %s, byte coverage %s, moments %s%s\n",
               in.name, static_cast<long long>(n), same ? "yes" : "NO", finite ? "yes" : "NO",
               scale_at.empty() ? "-" : (scales_ok ? "ok" : "BAD"),
               in.block_values > 1 ? (coverage ? "full" : "PARTIAL") : "-",
               in.block_values > 1 ? "-" : (moments ? "ok" : "BAD"), ok ? "" : "  FAIL");
        pass = pass && ok;
    }
    printf("synth-check: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
