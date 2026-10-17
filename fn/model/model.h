#pragma once

// fn::model: what FNFL holds once the model is loaded (R1). Per card: the weights in one allocation, laid out region by
// region as fn/model/placement.h and fn/loader/loader.h say; every state and workspace class in its own allocation,
// sized by fn/model/state.h; the host PLE table stays mapped, pinned in place with --ple-table full (PN1). fn::runtime
// holds one model. No CUDA type appears here.

#include "fn/model/placement.h"
#include "fn/model/state.h"
#include "fn/xchg/xchg.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fn::model {

// One device allocation; frees itself on its device. The owner must outlive every queued GPU consumer (callers drain
// both devices before a model is destroyed).
class device_buffer {
  public:
    device_buffer() = default;
    ~device_buffer();
    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
    device_buffer(device_buffer && o) noexcept;
    device_buffer & operator=(device_buffer && o) noexcept;

    // false (and err set) when cudaMalloc fails; the buffer stays empty
    bool allocate(int32_t device, size_t bytes, std::string & err);
    void release();

    uint8_t * data() const { return p_; }
    size_t size() const { return n_; }
    int32_t device() const { return dev_; }

  private:
    int32_t dev_ = -1;
    uint8_t * p_ = nullptr;
    size_t n_ = 0;
};

// A read-only mapping of a whole file (the GGUF shards and the draft file).
class mapped_file {
  public:
    mapped_file() = default;
    ~mapped_file();
    mapped_file(const mapped_file &) = delete;
    mapped_file & operator=(const mapped_file &) = delete;

    bool open(const std::string & path, std::string & err);
    const uint8_t * data() const { return p_; }
    uint64_t size() const { return n_; }
    const std::string & path() const { return path_; }

  private:
    std::string path_;
    const uint8_t * p_ = nullptr;
    uint64_t n_ = 0;
};

// A weight region on one card: a tensor alone, or the tensors that share a kernel layout (HC down with its inject,
// EX1 gate with its up).
struct region_place {
    layout lay = layout::gguf;
    int32_t members[2] = {-1, -1}; // tensors[] indices; members[1] = -1 for a single tensor
    int64_t offset = 0;            // in the card's weight allocation, 256-byte aligned
    int64_t bytes = 0;
};

// A reserved buffer class (fn/model/state.h): one allocation per card, its parts at buffer_class::offset.
struct reserved_class {
    buffer_class plan;
    device_buffer buf[2];

    uint8_t * part_ptr(int32_t p, int32_t card) const;
};

// Where one tensor's bytes come from: its file mapping and data offset (set by the loader at open).
struct tensor_source {
    const mapped_file * file = nullptr;
    uint64_t offset = 0;
    int64_t row_bytes = 0; // bytes of one GGUF row (ne0 elements)
};

constexpr int64_t region_align = 256;

class model {
  public:
    int32_t device[2] = {-1, -1};

    std::vector<std::unique_ptr<mapped_file>> files; // target shards, then the draft file
    std::vector<tensor_source> sources;              // per tensors[] entry
    std::vector<int32_t> region_of;                  // per tensors[] entry; -1 for the host PLE table

    std::vector<region_place> regions[2]; // per card, indexed like placement regions
    int64_t weight_bytes[2] = {0, 0};     // the regions' bytes, without alignment gaps
    int64_t weight_alloc[2] = {0, 0};     // the allocation, with them
    device_buffer weights[2];

    // the PLE table, left in its mapping (host)
    const uint8_t * ple_table = nullptr;
    uint64_t ple_bytes = 0;
    // PN1: its pin (fn/runtime/ple_host.h ple_pin) when the residency is full, null when lazy. Held type-erased because
    // fn/model sits below fn/runtime; declared after files, so it unregisters before the mapping is unmapped.
    std::shared_ptr<void> ple_pin;

    // GGUF keys the state sizes need
    int32_t ple_conv_kernel = 0;
    int32_t ple_ngram = 0;

    // every buffer beside the weights, in reservation order (fn/model/state.h), and the exchange rings
    std::vector<std::unique_ptr<reserved_class>> state;
    std::unique_ptr<fn::xchg::rings> rings;

    // per tensors[] entry: 1 once fn::model::pack_dense has repacked it on both cards into DP2's layout (LP1)
    std::vector<uint8_t> dense_packed;

    // device address of a tensor's region on a card (nullptr when the card holds none of it)
    uint8_t * region_ptr(int32_t tensor, int32_t card) const;
    // the tensor's index in placement.h, -1 when absent
    static int32_t find(const char * name);
    // a reserved class by name, nullptr when absent
    const reserved_class * state_class(const std::string & name) const;
};

} // namespace fn::model
