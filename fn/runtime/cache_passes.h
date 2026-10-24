#pragma once

// The expert cache's device orderings that ex-test checks directly (experts_test.cpp's swap and lent-region arms):
// a swap pass on one device, a layer's wait for its copies, and the device side of the lent region's borrow and return.
// ME3 (the cleanup): moved here from expert_cache.h unchanged; the residency (cache_residency.h) and the prompt loan
// (cache_loan.h) use them.

#include <cstdint>
#include <string>
#include <vector>

namespace fn::runtime {

// ME2: one pass's device work on one device: the patches (idx, val: n pairs, pinned host arrays the caller keeps
// until the pass commits) to the word table on the compute stream, ev_pub there, the swap stream's wait for it, then
// each copy in order (layer order) with ev_layer[l] recorded after layer l's last copy (has[l] set); ev_t0 / ev_t1
// (timing events) around the swap stream's copies. Used by the cache and by ex-test's swap arm.
struct pass_copy {
    int32_t layer = 0;
    void * dst = nullptr;
    const void * src = nullptr;
    int64_t bytes = 0;
};
struct pass_dev {
    uint64_t * words = nullptr;
    int32_t * idx_d = nullptr;
    uint64_t * val_d = nullptr;
    const int32_t * idx_h = nullptr;
    const uint64_t * val_h = nullptr;
    int32_t n = 0;
    void * compute = nullptr;
    void * swap = nullptr;
    void * ev_pub = nullptr;
    void * const * ev_layer = nullptr; // [layers]
    void * ev_t0 = nullptr;
    void * ev_t1 = nullptr;
    // ME3: after each layer's copies its count advanced (sw_count, host), written to sw_done by the swap stream and
    // to sw_need by the compute stream (ahead of the replay that waits for them: a replay's swap wait is a kernel)
    uint32_t * sw_count = nullptr; // [layers]
    uint32_t * sw_done = nullptr;  // [layers] device
    uint32_t * sw_need = nullptr;  // [layers] device
};
bool publish_pass(const pass_dev & p, const std::vector<pass_copy> & copies, uint8_t * has, std::string & err);
// the compute stream waits for layer l's copies, between two timing events (the exposed swap wait)
bool wait_layer(void * compute, void * ev_layer, void * ev_w0, void * ev_w1, std::string & err);
// ME2b item 6: the device side of a borrow or return of the lent region (expert_cache::prompt_hold), after the host
// has retired the region's residents and enqueued their residency rows on the compute streams (a borrow): each of the
// nd devices drained, so no reader of a former resident, no copy into the region and no row upload is outstanding when
// a prompt writer enters, or no prompt reader or copy when the region returns; a borrow with poison then fills the
// region (region, bytes: on dev[0]) with 0xFF (TEST). Synchronous.
bool lent_sync(const int32_t * dev, int32_t nd, bool borrow, uint8_t * region, int64_t bytes, bool poison,
               std::string & err);

} // namespace fn::runtime
