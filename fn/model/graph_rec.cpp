#include "fn/model/graph_rec.h"

// the real runtime entry points (the linker's --wrap names)
extern "C" {
cudaError_t __real_cudaLaunchKernel(const void * f, dim3 g, dim3 b, void ** args, size_t smem, cudaStream_t s);
cudaError_t __real_cudaLaunchCooperativeKernel(const void * f,
                                               dim3         g,
                                               dim3         b,
                                               void **      args,
                                               size_t       smem,
                                               cudaStream_t s);
cudaError_t __real_cudaMemcpyAsync(void * dst, const void * src, size_t n, cudaMemcpyKind kind, cudaStream_t s);
cudaError_t __real_cudaMemcpyPeerAsync(void * dst, int dd, const void * src, int sd, size_t n, cudaStream_t s);
cudaError_t __real_cudaMemcpy2DAsync(void *         dst,
                                     size_t         dp,
                                     const void *   src,
                                     size_t         sp,
                                     size_t         w,
                                     size_t         h,
                                     cudaMemcpyKind kind,
                                     cudaStream_t   s);
cudaError_t __real_cudaMemsetAsync(void * dst, int v, size_t n, cudaStream_t s);
cudaError_t __real_cudaMemset2DAsync(void * dst, size_t p, int v, size_t w, size_t h, cudaStream_t s);
cudaError_t __real_cudaEventRecord(cudaEvent_t e, cudaStream_t s);
cudaError_t __real_cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int flags);
}

namespace fn::model {
namespace {

thread_local graph_rec * tl_rec = nullptr;

}  // namespace

graph_rec * graph_rec_current() {
    return tl_rec;
}

void graph_rec_set(graph_rec * r) {
    tl_rec = r;
}

void graph_rec::begin_capture(std::vector<grec_op> * ops) {
    capture_ = true;
    out_     = ops;
    in_      = nullptr;
    exec_    = nullptr;
    at_      = 0;
    bad_     = false;
    out_->clear();
    graph_rec_set(this);
}

void graph_rec::begin_update(const std::vector<grec_op> * ops, cudaGraphExec_t exec) {
    capture_ = false;
    out_     = nullptr;
    in_      = ops;
    exec_    = exec;
    at_      = 0;
    bad_     = false;
    graph_rec_set(this);
}

bool graph_rec::end() {
    graph_rec_set(nullptr);
    return !bad_ && (capture_ || (in_ != nullptr && at_ == in_->size()));
}

// capture: the op's node (the stream's capture depends on it alone after the op) in order; events: their place only
void graph_rec::note(grec_kind k, const void * f, cudaStream_t s) {
    grec_op op{ k, f, nullptr };
    if (k != grec_kind::event_record && k != grec_kind::event_wait) {
        cudaStreamCaptureStatus st   = cudaStreamCaptureStatusNone;
        const cudaGraphNode_t * deps = nullptr;
        size_t                  nd   = 0;
        if (cudaStreamGetCaptureInfo(s, &st, nullptr, nullptr, &deps, &nd) != cudaSuccess ||
            st != cudaStreamCaptureStatusActive || nd != 1) {
            bad_ = true;  // not a node of the capture: the unit cannot be replayed by update
        } else {
            op.node = deps[0];
        }
    }
    out_->push_back(op);
}

const grec_op * graph_rec::next(grec_kind k, const void * f) {
    if (bad_ || in_ == nullptr || at_ >= in_->size() || (*in_)[at_].kind != k || (*in_)[at_].func != f) {
        bad_ = true;
        return nullptr;
    }
    return &(*in_)[at_++];
}

cudaError_t graph_rec::launch(grec_kind k, const void * f, dim3 g, dim3 b, void ** args, size_t smem, cudaStream_t s) {
    if (capture_) {
        const cudaError_t e = k == grec_kind::coop ? __real_cudaLaunchCooperativeKernel(f, g, b, args, smem, s) :
                                                     __real_cudaLaunchKernel(f, g, b, args, smem, s);
        if (e == cudaSuccess) {
            note(k, f, s);
        }
        return e;
    }
    const grec_op * op = next(k, f);
    if (op != nullptr) {
        cudaKernelNodeParams p = {};
        p.func                 = const_cast<void *>(f);
        p.gridDim              = g;
        p.blockDim             = b;
        p.sharedMemBytes       = static_cast<unsigned int>(smem);
        p.kernelParams         = args;
        if (cudaGraphExecKernelNodeSetParams(exec_, op->node, &p) != cudaSuccess) {
            (void) cudaGetLastError();
            bad_ = true;
        }
    }
    return cudaSuccess;
}

cudaError_t graph_rec::copy(void * dst, const void * src, size_t n, cudaMemcpyKind kind, cudaStream_t s) {
    if (capture_) {
        const cudaError_t e = __real_cudaMemcpyAsync(dst, src, n, kind, s);
        if (e == cudaSuccess) {
            note(grec_kind::copy, nullptr, s);
        }
        return e;
    }
    const grec_op * op = next(grec_kind::copy, nullptr);
    if (op != nullptr && cudaGraphExecMemcpyNodeSetParams1D(exec_, op->node, dst, src, n, kind) != cudaSuccess) {
        (void) cudaGetLastError();
        bad_ = true;
    }
    return cudaSuccess;
}

cudaError_t graph_rec::copy_peer(void * dst, int dd, const void * src, int sd, size_t n, cudaStream_t s) {
    if (capture_) {
        const cudaError_t e = __real_cudaMemcpyPeerAsync(dst, dd, src, sd, n, s);
        if (e == cudaSuccess) {
            note(grec_kind::copy_peer, nullptr, s);
        }
        return e;
    }
    const grec_op * op = next(grec_kind::copy_peer, nullptr);
    if (op != nullptr) {
        cudaMemcpy3DParms p = {};
        p.srcPtr            = make_cudaPitchedPtr(const_cast<void *>(src), n, n, 1);
        p.dstPtr            = make_cudaPitchedPtr(dst, n, n, 1);
        p.extent            = make_cudaExtent(n, 1, 1);
        p.kind              = cudaMemcpyDefault;  // the unified addresses name both devices
        if (cudaGraphExecMemcpyNodeSetParams(exec_, op->node, &p) != cudaSuccess) {
            (void) cudaGetLastError();
            bad_ = true;
        }
    }
    return cudaSuccess;
}

cudaError_t graph_rec::copy_2d(void *         dst,
                               size_t         dp,
                               const void *   src,
                               size_t         sp,
                               size_t         w,
                               size_t         h,
                               cudaMemcpyKind kind,
                               cudaStream_t   s) {
    if (capture_) {
        const cudaError_t e = __real_cudaMemcpy2DAsync(dst, dp, src, sp, w, h, kind, s);
        if (e == cudaSuccess) {
            note(grec_kind::copy_2d, nullptr, s);
        }
        return e;
    }
    const grec_op * op = next(grec_kind::copy_2d, nullptr);
    if (op != nullptr) {
        cudaMemcpy3DParms p = {};
        p.srcPtr            = make_cudaPitchedPtr(const_cast<void *>(src), sp, w, h);
        p.dstPtr            = make_cudaPitchedPtr(dst, dp, w, h);
        p.extent            = make_cudaExtent(w, h, 1);
        p.kind              = kind;
        if (cudaGraphExecMemcpyNodeSetParams(exec_, op->node, &p) != cudaSuccess) {
            (void) cudaGetLastError();
            bad_ = true;
        }
    }
    return cudaSuccess;
}

cudaError_t graph_rec::set(void * dst, int v, size_t n, cudaStream_t s) {
    if (capture_) {
        const cudaError_t e = __real_cudaMemsetAsync(dst, v, n, s);
        if (e == cudaSuccess) {
            note(grec_kind::set, nullptr, s);
        }
        return e;
    }
    const grec_op * op = next(grec_kind::set, nullptr);
    if (op != nullptr) {
        cudaMemsetParams p = {};
        p.dst              = dst;
        p.value            = static_cast<unsigned int>(v & 0xff);
        p.elementSize      = 1;
        p.width            = n;
        p.height           = 1;
        p.pitch            = n;
        if (cudaGraphExecMemsetNodeSetParams(exec_, op->node, &p) != cudaSuccess) {
            (void) cudaGetLastError();
            bad_ = true;
        }
    }
    return cudaSuccess;
}

cudaError_t graph_rec::set_2d(void * dst, size_t pitch, int v, size_t w, size_t h, cudaStream_t s) {
    if (capture_) {
        const cudaError_t e = __real_cudaMemset2DAsync(dst, pitch, v, w, h, s);
        if (e == cudaSuccess) {
            note(grec_kind::set_2d, nullptr, s);
        }
        return e;
    }
    const grec_op * op = next(grec_kind::set_2d, nullptr);
    if (op != nullptr) {
        cudaMemsetParams p = {};
        p.dst              = dst;
        p.value            = static_cast<unsigned int>(v & 0xff);
        p.elementSize      = 1;
        p.width            = w;
        p.height           = h;
        p.pitch            = pitch;
        if (cudaGraphExecMemsetNodeSetParams(exec_, op->node, &p) != cudaSuccess) {
            (void) cudaGetLastError();
            bad_ = true;
        }
    }
    return cudaSuccess;
}

cudaError_t graph_rec::event_record(cudaEvent_t e, cudaStream_t s) {
    if (capture_) {
        const cudaError_t r = __real_cudaEventRecord(e, s);
        if (r == cudaSuccess) {
            note(grec_kind::event_record, nullptr, s);
        }
        return r;
    }
    (void) next(grec_kind::event_record, nullptr);  // the graph's edge already
    return cudaSuccess;
}

cudaError_t graph_rec::event_wait(cudaStream_t s, cudaEvent_t e, unsigned int flags) {
    if (capture_) {
        const cudaError_t r = __real_cudaStreamWaitEvent(s, e, flags);
        if (r == cudaSuccess) {
            note(grec_kind::event_wait, nullptr, s);
        }
        return r;
    }
    (void) next(grec_kind::event_wait, nullptr);
    return cudaSuccess;
}

}  // namespace fn::model

// ---- the wrapped entry points ---------------------------------------------------------------------------------------
using fn::model::graph_rec;
using fn::model::grec_kind;

extern "C" {

cudaError_t __wrap_cudaLaunchKernel(const void * f, dim3 g, dim3 b, void ** args, size_t smem, cudaStream_t s) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaLaunchKernel(f, g, b, args, smem, s) :
                          r->launch(grec_kind::kernel, f, g, b, args, smem, s);
}

cudaError_t __wrap_cudaLaunchCooperativeKernel(const void * f,
                                               dim3         g,
                                               dim3         b,
                                               void **      args,
                                               size_t       smem,
                                               cudaStream_t s) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaLaunchCooperativeKernel(f, g, b, args, smem, s) :
                          r->launch(grec_kind::coop, f, g, b, args, smem, s);
}

cudaError_t __wrap_cudaMemcpyAsync(void * dst, const void * src, size_t n, cudaMemcpyKind kind, cudaStream_t s) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaMemcpyAsync(dst, src, n, kind, s) : r->copy(dst, src, n, kind, s);
}

cudaError_t __wrap_cudaMemcpyPeerAsync(void * dst, int dd, const void * src, int sd, size_t n, cudaStream_t s) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaMemcpyPeerAsync(dst, dd, src, sd, n, s) : r->copy_peer(dst, dd, src, sd, n, s);
}

cudaError_t __wrap_cudaMemcpy2DAsync(void *         dst,
                                     size_t         dp,
                                     const void *   src,
                                     size_t         sp,
                                     size_t         w,
                                     size_t         h,
                                     cudaMemcpyKind kind,
                                     cudaStream_t   s) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaMemcpy2DAsync(dst, dp, src, sp, w, h, kind, s) :
                          r->copy_2d(dst, dp, src, sp, w, h, kind, s);
}

cudaError_t __wrap_cudaMemsetAsync(void * dst, int v, size_t n, cudaStream_t s) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaMemsetAsync(dst, v, n, s) : r->set(dst, v, n, s);
}

cudaError_t __wrap_cudaMemset2DAsync(void * dst, size_t p, int v, size_t w, size_t h, cudaStream_t s) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaMemset2DAsync(dst, p, v, w, h, s) : r->set_2d(dst, p, v, w, h, s);
}

cudaError_t __wrap_cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaEventRecord(e, s) : r->event_record(e, s);
}

cudaError_t __wrap_cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int flags) {
    graph_rec * r = fn::model::graph_rec_current();
    return r == nullptr ? __real_cudaStreamWaitEvent(s, e, flags) : r->event_wait(s, e, flags);
}

}  // extern "C"
