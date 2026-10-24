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
    // ME1: pinned host memory instead (cudaHostAlloc, portable): the device reads and writes it by copies (UVA,
    // cudaMemcpyDefault)
    // mapped: also mapped into the device address space (cudaHostAllocMapped; under UVA at the host address)
    bool allocate_host(int32_t device, size_t bytes, std::string & err, bool mapped = false);
    // ME2b: a non-owning view of device memory another owner allocated and frees (the expert pool's lent region);
    // release forgets it
    void view(int32_t device, uint8_t * p, size_t bytes);
    void release();

    uint8_t * data() const { return p_; }
    size_t size() const { return n_; }
    int32_t device() const { return dev_; }
    bool host() const { return host_; }

  private:
    int32_t dev_ = -1;
    uint8_t * p_ = nullptr;
    size_t n_ = 0;
    bool host_ = false;
    bool owned_ = true;
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
    // ME1: cached, a target routed-expert region the expert cache holds in host memory (no device bytes; region_ptr
    // is null); alias, on one device (model::single) card 1's copy of a tensor replicated whole on both cards, which
    // reads card 0's region (no device bytes of its own)
    bool cached = false;
    bool alias = false;
};

class kv_pool;

// A reserved buffer class (fn/model/state.h): one allocation per card, its parts at buffer_class::offset. MU1: a pooled
// class of a multi-slot load is a view of its slot's virtual range in the KV pool (pool set): only its mapped pages
// and its fixed parts hold memory (live_bytes)
struct reserved_class {
    buffer_class plan;
    device_buffer buf[2];
    const kv_pool * pool = nullptr; // MU1: the pool whose virtual range buf views, null: an allocation of its own
    int32_t pool_slot = -1;

    uint8_t * part_ptr(int32_t p, int32_t card) const;
    // the bytes of part p on card that hold memory now: the part's bytes, or (pooled) its mapped prefix
    int64_t live_bytes(int32_t p, int32_t card) const;
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
    // ME1: both logical cards on one device (device[0] == device[1]): no peer access, one copy of the replicated
    // tensors; expert_cache: the target's routed experts are not placed on the device (fn/runtime/expert_cache.h)
    bool single = false;
    bool expert_cache = false;
    // MEKV (--kv-host, one or two devices): the host-backed KV tier (fn/model/kv_tier.h): the target and draft KV in
    // host
    // memory; ME3 (--kv-window): its window in cells (kv_tier::window_cells by default)
    bool kv_host = false;
    int32_t kv_window = 32768;
    // KV8 (--fnfl-kv-format q8): the target KV classes hold the 8-bit format (state_params::kv_q8), set by
    // reserve_state
    bool kv_q8 = false;

    std::vector<std::unique_ptr<mapped_file>> files; // target shards, then the draft file
    std::vector<tensor_source> sources;              // per tensors[] entry
    std::vector<int32_t> region_of;                  // per tensors[] entry; -1 for the host PLE table

    // ME7: this load's placement entries: placement.h's tensors with the MTP draft's routed expert tensors resolved
    // from the draft file (Q8_0 as placed, or the opt-in Q2_0: fn/loader open_model), and the cards' weight bytes
    // they sum to. Empty before open_model; tp(t) is the entry every loader reader and the draft take
    std::vector<tensor_place> place;
    int64_t placed_card_bytes[2] = {weight_card_bytes[0], weight_card_bytes[1]};
    const char * draft_expert_format = "Q8_0"; // ME7: the draft's routed gate/up/down format, as resolved
    const tensor_place & tp(int32_t t) const { return place.empty() ? tensors[t] : place[size_t(t)]; }

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

    // MU1: the slots (1..4; fn/model/state.h slot_class) and, with more than one, the shared KV pool. Declared before
    // the classes, which hold views of its ranges, so it is destroyed after them
    int32_t n_slots = 1;
    std::unique_ptr<kv_pool> pool;

    // every buffer beside the weights, in reservation order (fn/model/state.h), and the exchange rings; MU1: with more
    // than one slot, slot 0's slot classes are here and slot s's (s >= 1) in slot_state[s - 1]
    std::vector<std::unique_ptr<reserved_class>> state;
    std::vector<std::vector<std::unique_ptr<reserved_class>>> slot_state;
    std::unique_ptr<fn::xchg::rings> rings;

    // per tensors[] entry: 1 once fn::model::pack_dense has repacked it on both cards into DP2's layout (LP1)
    std::vector<uint8_t> dense_packed;

    // device address of a tensor's region on a card (nullptr when the card holds none of it)
    uint8_t * region_ptr(int32_t tensor, int32_t card) const;
    // the tensor's index in placement.h, -1 when absent
    static int32_t find(const char * name);
    // a reserved class by name, nullptr when absent; MU1: a slot class is the bound slot's (slot_scope)
    const reserved_class * state_class(const std::string & name) const;
    // MU1: the slot whose slot classes state_class resolves (0 unless a slot_scope is open)
    int32_t bound_slot() const { return bound_; }

    model();
    ~model();
    model(const model &) = delete;
    model & operator=(const model &) = delete;

  private:
    friend class slot_scope;
    mutable int32_t bound_ = 0;
};

// MU1: while it lives, m.state_class resolves slot s's slot classes (a slot's frame is resolved and cleared under it);
// scopes nest, each restoring the slot it found
class slot_scope {
  public:
    slot_scope(const model & m, int32_t s) : m_(m), prev_(m.bound_) { m.bound_ = s; }
    ~slot_scope() { m_.bound_ = prev_; }
    slot_scope(const slot_scope &) = delete;
    slot_scope & operator=(const slot_scope &) = delete;

  private:
    const model & m_;
    int32_t prev_;
};

} // namespace fn::model
