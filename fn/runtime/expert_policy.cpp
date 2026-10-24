// ME2: the expert cache's host policy (expert_policy.h), adapted from Strata-V100 3d4ce4e0: src/core/expert_cache.cpp
// (read_expert_profile, rank_learned_profile, write_expert_profile), src/program/generate.cpp (adapt's choice near
// lines 9400-9440, the decay at 9544 and 9555, PcieAuto at 1894) and src/core/expert_source.cpp (the PCIe share near
// line 3640). Strata's notice:
//
// MIT License
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Edits against the source (ME2 Result): keys instead of (layer, expert) pairs; the reader also refuses a pair named
// twice; the peer, helper and remote-expert terms of adapt's candidate test are not taken (one cache); the boot
// placement (Strata fills one slot array in rank order) adds FNFL's class slots and prompt zone.

#include "fn/runtime/expert_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace fn::runtime::policy {

bool read_profile(const std::string & path, int32_t n_layers, std::vector<int32_t> & ranked, std::string & err) {
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "profile: " + path + " is too short to hold a header";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    if (std::memcmp(magic, "STRP", 4) != 0 || version != 1 || int32_t(nl) != n_layers || ne != uint32_t(n_expert) ||
        n_ranked > want || n_ranked > uint32_t(n_layers) * n_expert) {
        std::fclose(f);
        err = "profile: " + path + " is not a version 1 STRP profile of " + std::to_string(n_layers) + " x 512";
        return false;
    }
    std::vector<uint16_t> raw(size_t(n_ranked) * 2);
    const bool got = n_ranked == 0 || std::fread(raw.data(), 2, raw.size(), f) == raw.size();
    std::fclose(f);
    if (!got) {
        err = "profile: the ranked list of " + path + " is truncated";
        return false;
    }
    std::vector<uint8_t> seen(size_t(n_layers) * n_expert, 0);
    ranked.assign(n_ranked, 0);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = raw[size_t(i) * 2], e = raw[size_t(i) * 2 + 1];
        if (l >= n_layers || e >= n_expert || seen[size_t(l) * n_expert + e] != 0) {
            err = "profile: pair " + std::to_string(i) + " of " + path + " is out of range or repeated";
            return false;
        }
        seen[size_t(l) * n_expert + e] = 1;
        ranked[i] = l * n_expert + e;
    }
    return true;
}

bool write_profile(const std::string & path, int32_t n_layers, const std::vector<int32_t> & ranked, std::string & err) {
    const size_t n = size_t(n_layers) * n_expert;
    std::vector<int32_t> table(n, -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const int32_t k = ranked[r];
        if (k < 0 || size_t(k) >= n) {
            err = "profile: a ranked key is out of range";
            return false;
        }
        table[size_t(k)] = int32_t(r);
        pairs.push_back(uint16_t(k / n_expert));
        pairs.push_back(uint16_t(k % n_expert));
    }
    const uint32_t hdr[5] = {1u, uint32_t(n_layers), uint32_t(n_expert), uint32_t(ranked.size()),
                             uint32_t(ranked.size())};
    const std::string tmp = path + ".tmp";
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "profile: cannot create " + tmp;
        return false;
    }
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5 &&
              (pairs.empty() || std::fwrite(pairs.data(), 2, pairs.size(), f) == pairs.size()) &&
              std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        err = "profile: cannot write " + path;
        return false;
    }
    return true;
}

std::vector<int32_t> rank_learned(int32_t n_layers, const std::vector<uint8_t> & resident,
                                  const std::vector<double> & heat, const std::vector<int32_t> & prior) {
    const size_t n = size_t(n_layers) * n_expert;
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        if (prior[r] >= 0 && size_t(prior[r]) < n && prior_rank[size_t(prior[r])] == INT64_MAX) {
            prior_rank[size_t(prior[r])] = int64_t(r);
        }
    }
    std::vector<int32_t> order(n);
    for (size_t i = 0; i < n; ++i) {
        order[i] = int32_t(i);
    }
    const auto res = [&](int32_t i) { return size_t(i) < resident.size() && resident[size_t(i)] != 0; };
    const auto ht = [&](int32_t i) { return size_t(i) < heat.size() ? heat[size_t(i)] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        if (res(a) != res(b)) {
            return res(a);
        }
        if (ht(a) != ht(b)) {
            return ht(a) > ht(b);
        }
        if (prior_rank[size_t(a)] != prior_rank[size_t(b)]) {
            return prior_rank[size_t(a)] < prior_rank[size_t(b)];
        }
        return a < b;
    });
    return order;
}

place_out place(const place_in & in, const std::vector<int32_t> & ranked) {
    place_out o;
    const int32_t nk = int32_t(in.count.size());
    std::vector<int32_t> used(size_t(nk), 0);
    for (int32_t k = 0; k < nk; ++k) {
        o.first.push_back(o.n_main);
        o.n_main += in.count[size_t(k)];
        for (int32_t i = 0; i < in.count[size_t(k)]; ++i) {
            o.slot_cls.push_back(k);
        }
    }
    o.slot_key.assign(size_t(o.n_main), -1);
    const int32_t n_layers = int32_t(in.class_of_layer.size());
    std::vector<uint8_t> placed(size_t(n_layers) * n_expert, 0);
    int64_t zone_used = 0;
    for (const int32_t key : ranked) {
        if (key < 0 || key >= n_layers * n_expert || placed[size_t(key)] != 0) {
            continue;
        }
        const int32_t k = in.class_of_layer[size_t(key / n_expert)];
        if (used[size_t(k)] < in.count[size_t(k)]) {
            o.slot_key[size_t(o.first[size_t(k)] + used[size_t(k)]++)] = key;
        } else if (zone_used + in.entry[size_t(k)] <= in.zone_bytes) {
            o.slot_cls.push_back(k);
            o.zone_at.push_back(zone_used);
            o.slot_key.push_back(key);
            zone_used += in.entry[size_t(k)];
        } else {
            continue;
        }
        placed[size_t(key)] = 1;
    }
    return o;
}

std::vector<swap> choose_swaps(int32_t n_layers, const float * usage, const int32_t * slot_of,
                               const adapt_options & o) {
    std::vector<swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int32_t l = 0; l < n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float * u = usage + int64_t(l) * n_expert;
        const int32_t * r = slot_of + int64_t(l) * n_expert;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (r[e] < 0) {
                if (u[e] >= o.min_use) {
                    cand.emplace_back(u[e], e);
                }
            } else {
                vict.emplace_back(u[e], e);
            }
        }
        if (cand.empty() || vict.empty()) {
            continue;
        }
        std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + ptrdiff_t(nc), vict.end(),
                          [](const auto & a, const auto & b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + o.min_gain) {
                break;
            }
            swaps.push_back({cand[i].first - vict[i].first, l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const swap & a, const swap & b) { return a.gain > b.gain; });
    if (int32_t(swaps.size()) > o.max_swaps) {
        swaps.resize(size_t(o.max_swaps));
    }
    if (o.layer_order) {
        std::stable_sort(swaps.begin(), swaps.end(), [](const swap & a, const swap & b) { return a.layer < b.layer; });
    }
    return swaps;
}

void decay_usage(std::vector<float> & usage, std::vector<double> & heat, float decay) {
    for (size_t i = 0; i < heat.size() && i < usage.size(); ++i) {
        heat[i] += double(usage[i]);
    }
    for (float & v : usage) {
        v *= decay;
    }
}

double pcie_auto::window(double ms_run, int64_t jobs, double lane_ms, int64_t lane_bytes, double cur,
                         split_reading & rd) {
    rd = split_reading();
    if (!on || o.windows.empty()) {
        return -1.0;
    }
    acc_ms += ms_run - mark_ms;
    acc_jobs += jobs - mark_jobs;
    acc_lane_ms += lane_ms - mark_lane_ms;
    acc_lane_bytes += lane_bytes - mark_lane_bytes;
    mark(ms_run, jobs, lane_ms, lane_bytes);
    ++acc_win;
    // ME5 rider: the last entry repeats, so the share keeps being read for the whole run
    const int32_t u = std::min<int32_t>(updates, int32_t(o.windows.size()) - 1);
    if (acc_win < o.windows[size_t(u)]) {
        return -1.0;
    }
    rd.update = updates + 1;
    rd.samples = acc_jobs;
    rd.probe_gbps = link_gbps;
    rd.lane_ms = acc_lane_ms;
    rd.lane_bytes = acc_lane_bytes;
    rd.old_share = cur;
    rd.new_share = cur;
    const bool few_cpu = acc_jobs < o.min_cpu, no_lane = !(acc_lane_ms > 0.0) || acc_lane_bytes <= 0;
    if (few_cpu || no_lane) { // a leg too thin to read: look again
        if (acc_win >= o.off_after) {
            on = false;
            rd.reason = few_cpu ? "kept: refinement off (too few CPU experts within off-after rounds)"
                                : "kept: refinement off (no lane copy within off-after rounds)";
            return -1.0;
        }
        rd.reason = deferred ? nullptr
                    : few_cpu ? "deferred: too few CPU experts (min-cpu)"
                              : "deferred: no lane copy to read";
        deferred = true;
        return -1.0;
    }
    const double tc = acc_ms * 1000.0 / double(acc_jobs);
    const double link = double(acc_lane_bytes) / (acc_lane_ms * 1e-3) / 1e9;
    acc_ms = 0.0;
    acc_jobs = 0;
    acc_lane_ms = 0.0;
    acc_lane_bytes = 0;
    acc_win = 0;
    deferred = false;
    ++updates;
    rd.tc_us = tc;
    rd.link_gbps = link;
    if (!(tc > 1.0) || !(blob_bytes > 0.0) || !(link > 0.0)) {
        rd.reason = "kept: no reading (the time per CPU expert, the blob or the lane's rate is not positive)";
        return -1.0;
    }
    const double r = blob_bytes / (tc * 1e-6) / 1e9;
    rd.rate_gbps = r;
    const double raw = link / (link + o.k * r);
    rd.raw_share = raw;
    // ME5 rider: the deadband on the unrounded reading (a rounded comparison would hold nothing back), then the step
    if (std::fabs(raw - cur) < o.min_change) {
        rd.reason = "kept: the reading is within min-change of the share";
        return -1.0;
    }
    const double f = std::min(o.hi, std::max(o.lo, std::round(raw / o.round) * o.round));
    if (std::fabs(f - cur) < 1e-9) {
        rd.reason = "kept: the rounded and clamped share is the current one";
        return -1.0;
    }
    rd.new_share = f;
    rd.reason = "changed";
    return f;
}

int32_t share_num(double share) {
    return std::clamp(int32_t(share * 256.0 + 0.5), 0, 256);
}

} // namespace fn::runtime::policy
