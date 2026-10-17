// LOAD STAND-IN (CR3, bench only, never merged): the bytes of one layer's other launches between two launches of the
// same expert code (the dense weights, HC's two sublayers, the other kernels' code), streamed with a chosen cache
// operator. No later ticket replaces it: it exists only for ex-bench's cr3seq arms. Never a kernel result, never a
// correctness reference, never in a model pass; its time is not a kernel time.
#pragma once

#include <cstdint>

namespace fn::dense {

// the cache operator of a stand-in stream
enum class stand_op : int32_t {
    normal, // ld.global.nc.L1::no_allocate: normal L2 priority, as today's dense weight loads (dense_dec.cuh ldw4)
    ef,     // ld.global.cs: evict-first in L2, as HC's and the routed experts' weight loads and CR3's fix
};

// one launch reading bytes of p (16-byte aligned, a multiple of 16) once with op; the reads feed a store to sink (144
// words of device memory) taken practically never, so they are kept
bool stand_stream(const void * p, int64_t bytes, stand_op op, uint32_t * sink, void * stream);

} // namespace fn::dense
