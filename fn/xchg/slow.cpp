#include "slow.h"

#include <atomic>
#include <cstdio>
#include <ctime>

namespace fn::xchg::slow {

namespace {

std::atomic<uint64_t>               writers{ 0 };
std::atomic<void (*)(const char *)> sink_fn{ nullptr };

}  // namespace

void set_writers(uint64_t mask) {
    writers.store(mask, std::memory_order_relaxed);
}

void writer(int32_t slot, bool on) {
    if (slot < 0 || slot >= 64) {
        return;
    }
    const uint64_t bit = uint64_t(1) << slot;
    if (on) {
        writers.fetch_or(bit, std::memory_order_relaxed);
    } else {
        writers.fetch_and(~bit, std::memory_order_relaxed);
    }
}

bool on() {
    return writers.load(std::memory_order_relaxed) != 0;
}

bool due(double s) {
    return s >= threshold_s && on();
}

void set_sink(void (*sink)(const char * line)) {
    sink_fn.store(sink, std::memory_order_release);
}

void line(const std::string & what, double s, const std::string & detail) {
    timespec wall = {}, mono = {};
    clock_gettime(CLOCK_REALTIME, &wall);
    clock_gettime(CLOCK_MONOTONIC, &mono);
    tm utc = {};
    gmtime_r(&wall.tv_sec, &utc);
    char buf[1024];
    snprintf(buf, sizeof buf, "FNFL slow: %s %.3f s%s%s (UTC %02d:%02d:%02d.%06ld, t_us %lld)", what.c_str(), s,
             detail.empty() ? "" : ", ", detail.c_str(), utc.tm_hour, utc.tm_min, utc.tm_sec, long(wall.tv_nsec / 1000),
             (long long) mono.tv_sec * 1000000LL + mono.tv_nsec / 1000);
    void (*f)(const char *) = sink_fn.load(std::memory_order_acquire);
    if (f != nullptr) {
        f(buf);
    } else {
        fprintf(stderr, "%s\n", buf);
        fflush(stderr);
    }
}

}  // namespace fn::xchg::slow
