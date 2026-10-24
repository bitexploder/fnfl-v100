#include "fn/runtime/split_test.h"

#include "fn/runtime/expert_policy.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace fn::runtime {

namespace {

namespace pol = fn::runtime::policy;

// The legs a window: 10 CPU experts at tc(w) us each (the blob 1 MB, so R = 1,000 / tc GB/s) and the lane's 6 MB in
// 1 ms (L = 6 GB/s), or none when lane(w) is false; the reading is L / (L + k R) = 6 / (6 + 1,000 / tc)
constexpr double blob = 1.0e6, lane_ms_win = 1.0;
constexpr int64_t lane_bytes_win = 6000000;
constexpr int32_t cpu_win = 10;

// the time per CPU expert that makes the reading r at k = 1
double tc_for(double r) {
    return 1000.0 / (6.0 * (1.0 - r) / r);
}

struct trace {
    std::vector<pol::split_reading> att;          // every reported attempt
    std::vector<std::pair<int32_t, double>> chg;  // (window, new share)
    double share = 0.0;
};

// windows 1..n from share cur0 with the start-up probe at 6.671 GB/s (reported, not used)
trace run(const pol::split_options & o, int32_t n, double cur0, const std::function<double(int32_t)> & tc,
          const std::function<bool(int32_t)> & lane, pol::pcie_auto & a, double ms0 = 0.0, int64_t jobs0 = 0) {
    a = pol::pcie_auto();
    a.on = true;
    a.o = o;
    a.link_gbps = 6.671;
    a.blob_bytes = blob;
    double ms = ms0, lms = 0.0;
    int64_t jobs = jobs0, lb = 0;
    a.mark(ms, jobs, lms, lb);
    trace t;
    t.share = cur0;
    for (int32_t w = 1; w <= n; ++w) {
        ms += tc(w) * 1e-3 * cpu_win;
        jobs += cpu_win;
        if (lane(w)) {
            lms += lane_ms_win;
            lb += lane_bytes_win;
        }
        pol::split_reading rd;
        const double f = a.window(ms, jobs, lms, lb, t.share, rd);
        if (rd.reason != nullptr) {
            t.att.push_back(rd);
        }
        if (f >= 0) {
            t.chg.emplace_back(w, f);
            t.share = f;
        }
    }
    return t;
}

// one report line's text, printf style
__attribute__((format(printf, 1, 2))) std::string text(const char * fmt, ...) {
    char b[400];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    return b;
}

bool near(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

const auto always = [](int32_t) { return true; };

} // namespace

bool split_estimator_check(std::string & report) {
    bool all = true;
    const auto line = [&](bool ok, const std::string & what) {
        report += std::string(ok ? "ok   " : "FAIL ") + what + "\n";
        all = all && ok;
    };
    const pol::split_options o;
    pol::pcie_auto a;
    // 1. constant legs read 0.20: one change at window 48 (0.30 -> 0.20), then kept, read every 96 windows for the
    //    whole run (windows 48, 144, .. 912: 10 attempts), refinement still on after them
    {
        const double tc = tc_for(0.20);
        const trace t = run(o, 1000, 0.30, [&](int32_t) { return tc; }, always, a);
        bool kept = t.att.size() == 10;
        for (size_t i = 1; i < t.att.size(); ++i) {
            kept = kept && std::string(t.att[i].reason).rfind("kept", 0) == 0;
        }
        const bool ok = t.chg.size() == 1 && t.chg[0].first == 48 && near(t.chg[0].second, 0.20) && kept && a.on &&
                        a.updates == 10 && near(t.att[0].link_gbps, 6.0) && near(t.att[0].probe_gbps, 6.671) &&
                        std::fabs(t.att[0].raw_share - 0.20) < 1e-9 && t.att[0].lane_bytes == 48 * lane_bytes_win;
        line(ok, text(
             "constant legs (reading 0.20 from 0.30): %zu change(s), the first at window %d to %.2f; %zu attempts to "
             "window 912, refinement %s after %d updates; L read %.3f GB/s from the lane (probe %.3f)",
             t.chg.size(), t.chg.empty() ? -1 : t.chg[0].first, t.chg.empty() ? -1.0 : t.chg[0].second, t.att.size(),
             a.on ? "on" : "OFF", a.updates, t.att.empty() ? 0.0 : t.att[0].link_gbps,
             t.att.empty() ? 0.0 : t.att[0].probe_gbps));
    }
    // 2. refinement continues past three readings: the legs change after window 336 (reading 0.40), and the fifth
    //    update (window 432) moves the share
    {
        const double t1 = tc_for(0.20), t2 = tc_for(0.40);
        const trace t = run(o, 600, 0.20, [&](int32_t w) { return w <= 336 ? t1 : t2; }, always, a);
        const bool ok = t.chg.size() == 1 && t.chg[0].first == 432 && near(t.chg[0].second, 0.40) && a.updates == 6;
        line(ok, text(
             "past three readings: the legs change at window 337, the share moves at window %d to %.2f (update %d "
                 "of %d)",
             t.chg.empty() ? -1 : t.chg[0].first, t.chg.empty() ? -1.0 : t.chg[0].second, 5, a.updates));
    }
    // 3. the deadband: from 0.20, a reading of 0.236 (0.036 away; rounded it would be 0.25) keeps the share; 0.24
    //    (0.04 away) changes it to 0.25; 0.164 keeps it, 0.16 changes it to 0.15
    {
        const auto one = [&](double r) {
            const double tc = tc_for(r);
            const trace t = run(o, 48, 0.20, [&](int32_t) { return tc; }, always, a);
            return t.chg.empty() ? 0.20 : t.chg[0].second;
        };
        const double s1 = one(0.236), s2 = one(0.24), s3 = one(0.164), s4 = one(0.16);
        const bool ok = near(s1, 0.20) && near(s2, 0.25) && near(s3, 0.20) && near(s4, 0.15);
        line(ok, text(
             "deadband 0.0375 on the unrounded reading from 0.20: 0.236 -> %.2f, 0.240 -> %.2f, 0.164 -> %.2f, "
                 "0.160 -> %.2f",
             s1, s2, s3, s4));
    }
    // 4. readings alternating just either side of the 0.225 rounding boundary (0.22, 0.23, ..) from 0.20 never flip
    //    the share (the old rounded test would have moved it to 0.25 at the first 0.23); likewise 0.27 and 0.28 from
    //    0.25 (the 0.275 boundary)
    {
        const double ta = tc_for(0.22), tb = tc_for(0.23);
        const auto alt = [&](int32_t w) { return ((w - 1 - 48) / 96) % 2 == 0 && w > 48 ? tb : ta; };
        const trace t = run(o, 2000, 0.20, alt, always, a);
        const double tc2 = tc_for(0.27), td = tc_for(0.28);
        const auto alt2 = [&](int32_t w) { return ((w - 1 - 48) / 96) % 2 == 0 && w > 48 ? td : tc2; };
        const trace t2 = run(o, 2000, 0.25, alt2, always, a);
        bool both = false;
        for (size_t i = 1; i < t.att.size(); ++i) {
            both = both || std::fabs(t.att[i].raw_share - 0.23) < 1e-6;
        }
        const bool ok = t.chg.empty() && t2.chg.empty() && t.att.size() == 21 && both && near(t.share, 0.20) &&
                        near(t2.share, 0.25);
        line(ok, text(
             "alternating readings 0.22/0.23 from 0.20 and 0.27/0.28 from 0.25 over %zu refinements: %zu and %zu "
                 "changes, shares %.2f and %.2f",
             t.att.size(), t.chg.size(), t2.chg.size(), t.share, t2.share));
    }
    // 5. the reset (window_service::reset_share at boot and per arm: a fresh estimator marked at the clocks now):
    //    a heavy CPU history before the mark is not read; updates and accumulators start at zero
    {
        const double tc = tc_for(0.20);
        const trace t = run(o, 48, 0.30, [&](int32_t) { return tc; }, always, a, 1.0e9, 10);
        const bool ok = t.chg.size() == 1 && near(t.chg[0].second, 0.20) && a.updates == 1 && t.att.size() == 1 &&
                        t.att[0].samples == 48 * cpu_win;
        pol::pcie_auto fresh;
        const bool zero = fresh.updates == 0 && fresh.acc_win == 0 &&
                          fresh.acc_jobs == 0 && fresh.acc_lane_bytes == 0 &&
                          !fresh.on;
        line(ok && zero, text(
             "reset: the clocks before the mark ignored (%lld CPU experts read), %d update(s); a fresh "
                         "estimator starts empty: %s",
             t.att.empty() ? -1LL : (long long)t.att[0].samples, a.updates, zero ? "yes" : "NO"));
    }
    // 6. deferrals: no lane copy before window 100 defers the first update (reported once), which then reads only
    //    once both legs have samples; too few CPU experts (1 a window) defers until 300 are read; refinement goes off
    //    after off-after rounds without a lane copy
    {
        const double tc = tc_for(0.20);
        const trace t = run(o, 200, 0.30, [&](int32_t) { return tc; }, [](int32_t w) { return w > 100; }, a);
        int32_t defers = 0;
        for (const pol::split_reading & r : t.att) {
            defers += std::string(r.reason).rfind("deferred: no lane", 0) == 0 ? 1 : 0;
        }
        int64_t changed_lane_bytes = -1; // the changing update read the one window with a lane copy
        for (const pol::split_reading & r : t.att) {
            changed_lane_bytes = std::string(r.reason) == "changed" ? r.lane_bytes : changed_lane_bytes;
        }
        const bool lane_ok = defers == 1 && t.chg.size() == 1 && t.chg[0].first == 101 &&
                             changed_lane_bytes == 1 * lane_bytes_win;
        pol::split_options few = o;
        few.min_cpu = 300;
        pol::pcie_auto a2;
        double ms = 0.0;
        int64_t jobs = 0, lb = 0;
        double lms = 0.0;
        a2.on = true;
        a2.o = few;
        a2.blob_bytes = blob;
        a2.mark(0, 0, 0, 0);
        int32_t first_update_at = -1;
        for (int32_t w = 1; w <= 400 && first_update_at < 0; ++w) {
            ms += tc * 1e-3;
            jobs += 1;
            lms += lane_ms_win;
            lb += lane_bytes_win;
            pol::split_reading rd;
            a2.window(ms, jobs, lms, lb, 0.30, rd);
            if (rd.reason != nullptr && rd.update == 1 && std::string(rd.reason).rfind("deferred", 0) != 0) {
                first_update_at = w;
            }
        }
        pol::split_options off = o;
        off.off_after = 300;
        const trace t3 = run(off, 400, 0.30, [&](int32_t) { return tc; }, [](int32_t) { return false; }, a);
        const bool off_ok = !a.on && !t3.att.empty() && std::string(t3.att.back().reason).find("no lane copy") !=
                                                            std::string::npos;
        line(lane_ok && first_update_at == 300 && off_ok, text(
             "deferrals: no lane copy until window 100 defers once and the share moves at window %d; 1 CPU expert a "
             "window first reads at window %d; no lane copy turns refinement %s after off-after",
             t.chg.empty() ? -1 : t.chg[0].first, first_update_at, a.on ? "ON" : "off"));
    }
    // 7. k = 2: 6 / (6 + 2 R) at R = 24 (0.111): from 0.30 to 0.10 (the lower clamp)
    {
        pol::split_options k2 = o;
        k2.k = 2.0;
        const double tc = tc_for(0.20);
        const trace t = run(k2, 48, 0.30, [&](int32_t) { return tc; }, always, a);
        const bool ok = t.chg.size() == 1 && near(t.chg[0].second, 0.10);
        line(ok, text("k 2 at a reading of 0.111: share %.2f", t.chg.empty() ? -1.0 : t.chg[0].second));
    }
    // 8. the clamp: a reading 0.05 from a share already at the lower bound changes nothing (kept, not "changed")
    {
        const double tc = tc_for(0.05);
        const trace t = run(o, 48, 0.10, [&](int32_t) { return tc; }, always, a);
        const bool ok = t.chg.empty() && t.att.size() == 1 && std::string(t.att[0].reason).rfind("kept", 0) == 0;
        line(ok, text("a reading of 0.05 at the lower bound 0.10: %zu changes (%s)", t.chg.size(),
             t.att.empty() ? "no attempt" : t.att[0].reason));
    }
    // 9. the share's split (unchanged)
    {
        const bool ok = pol::share_num(0.30) == 77 && pol::share_num(1.0) == 256 && pol::gpu_misses(10, 77) == 3 &&
                        pol::gpu_misses(10, 256) == 10 && pol::gpu_misses(1, 77) == 0;
        line(ok, text("the miss split: share_num and gpu_misses as before"));
    }
    return all;
}

} // namespace fn::runtime
