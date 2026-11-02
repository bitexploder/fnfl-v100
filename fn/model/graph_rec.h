#pragma once

// APF2: capture once, then replay by updating the parameters (the prompt path's unit graphs, prompt.h). The link
// wraps the CUDA runtime's stream operations (-Wl,--wrap for each, fn/CMakeLists.txt: fnfl-model's interface link
// options), and a recorder installed on the calling thread sees each one:
//   capture: the operation runs (into the stream capture) and its node is noted in order (kernels, cooperative
//            kernels, 1D / 2D / peer copies and memsets: the node the stream's capture now depends on; event records
//            and waits: no node, their place only).
//   update:  the operation does not run: the noted node of the same kind (and kernel function) at the same place
//            takes the new arguments in the executable (cudaGraphExec*SetParams, which copy them), and event records
//            and waits are already the graph's edges. Any difference (another kind, function, count, or a refused
//            update) is a mismatch; the caller then captures again.
// Without a recorder on the thread every call goes straight to the runtime.

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace fn::model {

enum class grec_kind : int32_t { kernel, coop, copy, copy_peer, copy_2d, set, set_2d, event_record, event_wait };

struct grec_op {
    grec_kind       kind;
    const void *    func = nullptr;  // kernels
    cudaGraphNode_t node = nullptr;  // nodes of the captured graph (null for events)
};

class graph_rec {
  public:
    // capture: ops is filled; update: ops is matched against, exec takes the arguments
    void begin_capture(std::vector<grec_op> * ops);
    void begin_update(const std::vector<grec_op> * ops, cudaGraphExec_t exec);
    // the recorder off the thread; true when nothing mismatched (and, updating, every op was matched)
    bool end();

    // the wrapped operations (graph_rec.cpp's __wrap_ functions)
    cudaError_t launch(grec_kind k, const void * f, dim3 g, dim3 b, void ** args, size_t smem, cudaStream_t s);
    cudaError_t copy(void * dst, const void * src, size_t n, cudaMemcpyKind kind, cudaStream_t s);
    cudaError_t copy_peer(void * dst, int dd, const void * src, int sd, size_t n, cudaStream_t s);
    cudaError_t copy_2d(void *         dst,
                        size_t         dp,
                        const void *   src,
                        size_t         sp,
                        size_t         w,
                        size_t         h,
                        cudaMemcpyKind kind,
                        cudaStream_t   s);
    cudaError_t set(void * dst, int v, size_t n, cudaStream_t s);
    cudaError_t set_2d(void * dst, size_t p, int v, size_t w, size_t h, cudaStream_t s);
    cudaError_t event_record(cudaEvent_t e, cudaStream_t s);
    cudaError_t event_wait(cudaStream_t s, cudaEvent_t e, unsigned int flags);

  private:
    bool                         capture_ = false;
    std::vector<grec_op> *       out_     = nullptr;
    const std::vector<grec_op> * in_      = nullptr;
    cudaGraphExec_t              exec_    = nullptr;
    size_t                       at_      = 0;
    bool                         bad_     = false;

    void            note(grec_kind k, const void * f, cudaStream_t s);
    const grec_op * next(grec_kind k, const void * f);
};

// the calling thread's recorder (null: none)
graph_rec * graph_rec_current();
void        graph_rec_set(graph_rec * r);

}  // namespace fn::model
