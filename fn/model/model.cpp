#include "fn/model/model.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

namespace fn::model {

device_buffer::~device_buffer() {
    release();
}

device_buffer::device_buffer(device_buffer && o) noexcept : dev_(o.dev_), p_(o.p_), n_(o.n_) {
    o.p_ = nullptr;
    o.n_ = 0;
}

device_buffer & device_buffer::operator=(device_buffer && o) noexcept {
    if (this != &o) {
        release();
        dev_ = o.dev_;
        p_ = o.p_;
        n_ = o.n_;
        o.p_ = nullptr;
        o.n_ = 0;
    }
    return *this;
}

bool device_buffer::allocate(int32_t device, size_t bytes, std::string & err) {
    release();
    if (bytes == 0) {
        dev_ = device;
        return true;
    }
    int prev = -1;
    cudaGetDevice(&prev);
    cudaError_t e = cudaSetDevice(device);
    void * p = nullptr;
    if (e == cudaSuccess) {
        e = cudaMalloc(&p, bytes);
    }
    if (prev >= 0) {
        cudaSetDevice(prev);
    }
    if (e != cudaSuccess) {
        err = "cudaMalloc of " + std::to_string(bytes) + " bytes on device " + std::to_string(device) + ": " +
              cudaGetErrorString(e);
        return false;
    }
    dev_ = device;
    p_ = static_cast<uint8_t *>(p);
    n_ = bytes;
    return true;
}

void device_buffer::release() {
    if (p_ != nullptr) {
        int prev = -1;
        cudaGetDevice(&prev);
        cudaSetDevice(dev_);
        cudaFree(p_);
        if (prev >= 0) {
            cudaSetDevice(prev);
        }
    }
    p_ = nullptr;
    n_ = 0;
}

mapped_file::~mapped_file() {
    if (p_ != nullptr) {
        // reason: munmap takes the mapping's address as a mutable pointer
        munmap(const_cast<uint8_t *>(p_), n_);
    }
}

bool mapped_file::open(const std::string & path, std::string & err) {
    path_ = path;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "cannot open '" + path + "': " + std::strerror(errno);
        return false;
    }
    struct stat st = {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        err = "cannot stat '" + path + "'";
        close(fd);
        return false;
    }
    void * p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        err = "cannot map '" + path + "': " + std::strerror(errno);
        return false;
    }
    p_ = static_cast<const uint8_t *>(p);
    n_ = (uint64_t)st.st_size;
    return true;
}

uint8_t * model::region_ptr(int32_t tensor, int32_t card) const {
    if (tensor < 0 || tensor >= n_tensors || card < 0 || card > 1 || (size_t)tensor >= region_of.size()) {
        return nullptr;
    }
    const int32_t r = region_of[tensor];
    if (r < 0 || weights[card].data() == nullptr || regions[card][r].bytes == 0) {
        return nullptr;
    }
    return weights[card].data() + regions[card][r].offset;
}

int32_t model::find(const char * name) {
    for (int32_t t = 0; t < n_tensors; ++t) {
        if (std::strcmp(tensors[t].name, name) == 0) {
            return t;
        }
    }
    return -1;
}

const reserved_class * model::state_class(const std::string & name) const {
    for (const auto & c : state) {
        if (c->plan.name == name) {
            return c.get();
        }
    }
    return nullptr;
}

uint8_t * reserved_class::part_ptr(int32_t p, int32_t card) const {
    if (p < 0 || p >= (int32_t)plan.parts.size() || card < 0 || card > 1 || buf[card].data() == nullptr ||
        plan.parts[p].bytes[card] == 0) {
        return nullptr;
    }
    return buf[card].data() + plan.offset(p, card);
}

} // namespace fn::model
