#pragma once

// TFC: the checks' CUDA status test (test support, host code): each check keeps its own tag in the message

#include <cuda_runtime.h>

#include <cstdio>

namespace fn::test {

// false, with "<tag>: <what>: <the error>" on stderr, unless e is cudaSuccess
inline bool cuda_ok(cudaError_t e, const char * tag, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s: %s\n", tag, what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

}  // namespace fn::test
