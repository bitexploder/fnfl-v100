#include "fn/model/model.h"

#include "fn/model/kv_pool.h"

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

device_buffer::device_buffer(device_buffer && o) noexcept
    : dev_(o.dev_), p_(o.p_), n_(o.n_), host_(o.host_), owned_(o.owned_) {
    o.p_ = nullptr;
    o.n_ = 0;
    o.owned_ = true;
}

device_buffer & device_buffer::operator=(device_buffer && o) noexcept {
    if (this != &o) {
        release();
        dev_ = o.dev_;
        p_ = o.p_;
        n_ = o.n_;
        host_ = o.host_;
        owned_ = o.owned_;
        o.p_ = nullptr;
        o.n_ = 0;
        o.owned_ = true;
    }
    return *this;
}

void device_buffer::view(int32_t device, uint8_t * p, size_t bytes) {
    release();
    dev_ = device;
    p_ = p;
    n_ = bytes;
    owned_ = false;
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

bool device_buffer::allocate_host(int32_t device, size_t bytes, std::string & err, bool mapped) {
    release();
    dev_ = device;
    if (bytes == 0) {
        return true;
    }
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
        err = "cudaSetDevice for host allocation";
        return false;
    }
    void * p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocPortable | (mapped ? cudaHostAllocMapped : 0u));
    const cudaError_t restored = cudaSetDevice(previous);
    if (e != cudaSuccess || restored != cudaSuccess) {
        if (p != nullptr && cudaFreeHost(p) != cudaSuccess) {
            err = "cudaFreeHost after failed device restore";
            return false;
        }
        err = "host allocation/device restore of " + std::to_string(bytes) + " bytes: " +
              cudaGetErrorString(e != cudaSuccess ? e : restored);
        return false;
    }
    p_ = static_cast<uint8_t *>(p);
    n_ = bytes;
    host_ = true;
    return true;
}

void device_buffer::release() {
    if (p_ != nullptr && !owned_) {
        // a view: its owner frees the memory
    } else if (p_ != nullptr && host_) {
        cudaFreeHost(p_);
    } else if (p_ != nullptr) {
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
    host_ = false;
    owned_ = true;
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
    if (r < 0 || regions[card][r].bytes == 0 || regions[card][r].cached) {
        return nullptr;
    }
    const int32_t owner = regions[card][r].alias ? 0 : card; // ME1: card 1 reads card 0's copy on one device
    if (weights[owner].data() == nullptr) {
        return nullptr;
    }
    return weights[owner].data() + regions[owner][r].offset;
}

int32_t model::find(const char * name) {
    for (int32_t t = 0; t < n_tensors; ++t) {
        if (std::strcmp(tensors[t].name, name) == 0) {
            return t;
        }
    }
    return -1;
}

model::model() = default;
model::~model() = default;

const reserved_class * model::state_class(const std::string & name) const {
    if (bound_ > 0 && slot_class(name)) {
        if (size_t(bound_ - 1) >= slot_state.size()) {
            return nullptr;
        }
        for (const auto & c : slot_state[size_t(bound_ - 1)]) {
            if (c->plan.name == name) {
                return c.get();
            }
        }
        return nullptr;
    }
    for (const auto & c : state) {
        if (c->plan.name == name) {
            return c.get();
        }
    }
    return nullptr;
}

int64_t reserved_class::live_bytes(int32_t p, int32_t card) const {
    if (p < 0 || p >= (int32_t)plan.parts.size() || card < 0 || card > 1) {
        return 0;
    }
    const int64_t live = pool == nullptr ? -1 : pool->live_bytes(pool_slot, plan.name, p, card);
    return live < 0 ? plan.parts[p].bytes[card] : live; // a fixed part's bytes are always mapped
}

uint8_t * reserved_class::part_ptr(int32_t p, int32_t card) const {
    if (p < 0 || p >= (int32_t)plan.parts.size() || card < 0 || card > 1 || buf[card].data() == nullptr ||
        plan.parts[p].bytes[card] == 0) {
        return nullptr;
    }
    return buf[card].data() + plan.offset(p, card);
}

} // namespace fn::model
