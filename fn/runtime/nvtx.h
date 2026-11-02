#pragma once

// TTTS7: NVTX ranges around the host phases of the split's round loop (the leader's and the follower's: packing and
// unpacking, launches, waits, the verdict, sampling and bookkeeping), so an nsys capture with --trace=nvtx attributes
// the host time between graphs. Without a tool attached a range is two calls that return at once (NVTX's injection
// check); nothing else changes.

#include <nvtx3/nvToolsExt.h>

namespace fn::runtime {

class nvtx_range {
  public:
    explicit nvtx_range(const char * name) { nvtxRangePushA(name); }

    ~nvtx_range() { nvtxRangePop(); }

    nvtx_range(const nvtx_range &)             = delete;
    nvtx_range & operator=(const nvtx_range &) = delete;
};

// one range at a time over a function's consecutive phases: next() ends the current range and starts the next
class nvtx_phase {
  public:
    explicit nvtx_phase(const char * name) { nvtxRangePushA(name); }

    ~nvtx_phase() { nvtxRangePop(); }

    void next(const char * name) {
        nvtxRangePop();
        nvtxRangePushA(name);
    }

    nvtx_phase(const nvtx_phase &)             = delete;
    nvtx_phase & operator=(const nvtx_phase &) = delete;
};

}  // namespace fn::runtime
