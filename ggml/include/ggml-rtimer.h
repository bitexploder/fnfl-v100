#pragma once

// [TAG_ROUND_TIMERS] (ticket 0089) LLAMA_ROUND_TIMERS=1: per-round host phase timers, counters and GPU graph stamps,
// printed as per-round means at the end of each request. Off by default; when off every entry point returns at a
// cached flag, and no GPU work or API call is added.

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// whether LLAMA_ROUND_TIMERS is set (read once)
GGML_API bool ggml_rt_on(void);
// the id of a named phase or counter (names are string literals, the id is stable for the process)
GGML_API int  ggml_rt_id(const char * name);
// add a host duration (ns) and one call to phase id, under the current graph label
GGML_API void ggml_rt_add(int id, int64_t ns);
// add n to counter id (no duration), under the current graph label
GGML_API void ggml_rt_cnt(int id, int64_t n);
// the label of the graphs launched from now on: a context and its batch size
GGML_API void ggml_rt_label(const void * ctx, int n_tokens);
GGML_API int  ggml_rt_cur_label(void);
// the GPU stamp collector: fills up to max (timestamp ns, tag) pairs recorded since the last call, returns the count
typedef int (*ggml_rt_collect_fn)(uint64_t * t, uint32_t * tag, int max);
// (ticket P4h) a collected tag (device in bits 24 and up) saying that the device's ring lost stamps before the next entry
#define GGML_RT_TAG_LOST 0x00ffffffu
GGML_API void ggml_rt_set_collector(ggml_rt_collect_fn fn);
// a speculative round ended (after its acceptance): close the round's accumulators and GPU stamps
GGML_API void ggml_rt_round(void);
// print the per-round means since the last report under label, then reset
GGML_API void ggml_rt_report(const char * label);

#ifdef __cplusplus
}

#include <chrono>

struct ggml_rt_scope {
    int id;
    std::chrono::steady_clock::time_point t0;
    explicit ggml_rt_scope(int id_) : id(id_) {
        if (id >= 0) {
            t0 = std::chrono::steady_clock::now();
        }
    }
    ~ggml_rt_scope() {
        if (id >= 0) {
            ggml_rt_add(id, std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        }
    }
};

#define GGML_RT_CAT2(a, b) a##b
#define GGML_RT_CAT(a, b) GGML_RT_CAT2(a, b)
// time the rest of the enclosing block as phase `name` (a string literal)
#define GGML_RT_SCOPE(name) \
    static const int GGML_RT_CAT(ggml_rt_id_, __LINE__) = ggml_rt_on() ? ggml_rt_id(name) : -1; \
    ggml_rt_scope GGML_RT_CAT(ggml_rt_sc_, __LINE__)(GGML_RT_CAT(ggml_rt_id_, __LINE__))
// count n under counter `name`
#define GGML_RT_COUNT(name, n) do { \
        static const int ggml_rt_cid_ = ggml_rt_on() ? ggml_rt_id(name) : -1; \
        if (ggml_rt_cid_ >= 0) { ggml_rt_cnt(ggml_rt_cid_, (n)); } \
    } while (0)
#endif
