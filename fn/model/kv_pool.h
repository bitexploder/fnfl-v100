#pragma once

// MU1: the shared KV cell pool of a multi-slot load (--parallel 2..4 on resident TP), in CUDA virtual memory. Every
// slot's pooled classes (fn/model/state.h pooled_class: the target and draft KV, the index keys and summaries) are a
// full-capacity virtual range on each card, so the kernels see the contiguous addresses they see today and one slot can
// hold the whole capacity. The physical memory is n_pages pages a card, created once at load and together the bytes
// a one-slot load reserves for those classes; a page holds one page_cells-cell slice of every per-cell part (one
// allocation a slice: cuMemMap maps whole allocations), and a slot's logical pages 0 .. mapped - 1 are mapped onto
// pages as its cells grow. A class's fixed parts (the draft
// index's weight copies and select depth) are mapped once a slot.
//
// Host code only. Mapping is a host call (cuMemMap and cuMemSetAccess are not stream-ordered): the caller maps before
// any consumer of the new cells is enqueued and unmaps only after every consumer of the old ones completed. A newly
// mapped page is zeroed (as a one-slot load's state starts), on the null stream, synchronously. Access is granted to
// both cards (the peer reads none of these ranges today; the grant keeps a future peer read legal).
//
// kv_pages is the page bookkeeping alone (no CUDA), so the CPU tests (ex-cpu-test) check its rounding, rollback and
// reclaim order.

#include "fn/model/state.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fn::model {

class model;

// physical pages 0 .. n_pages - 1; each slot's logical pages in order, each backed by one physical page; the free list
class kv_pages {
  public:
    void init(int32_t n_slots, int32_t n_pages);
    int32_t n_slots() const { return static_cast<int32_t>(map_.size()); }
    int32_t n_pages() const { return n_pages_; }
    int32_t free_pages() const { return static_cast<int32_t>(free_.size()); }
    int32_t mapped(int32_t slot) const;
    int32_t physical(int32_t slot, int32_t logical) const; // -1 when not mapped
    // the logical pages a prefix of cells cells needs: ceil(cells / page_cells), between 0 and n_pages
    static int32_t pages_for(int64_t cells, int32_t page_cells, int32_t n_pages);
    // the slot's next want - mapped(slot) logical pages from the free list (lowest physical page first), their physical
    // pages in taken; false with nothing taken when the free pages do not suffice (or want exceeds n_pages)
    bool take(int32_t slot, int32_t want, std::vector<int32_t> & taken);
    // the slot's logical pages keep .. mapped - 1 back to the free list (kept ascending), their physical pages in out
    void give(int32_t slot, int32_t keep, std::vector<int32_t> & out);

  private:
    int32_t n_pages_ = 0;
    std::vector<std::vector<int32_t>> map_; // [slot][logical] physical
    std::vector<int32_t> free_;             // ascending
};

// the slot an exhausted pool reclaims for slot for_slot: the least recently released (lowest released order; 0: never
// used) idle slot holding pages; an active slot (a request or a read on it) and for_slot never; -1 when none
struct kv_slot_view {
    bool active = false;
    int32_t pages = 0;
    uint64_t released = 0;
};
int32_t kv_reclaim_victim(const std::vector<kv_slot_view> & s, int32_t for_slot);

class kv_pool {
  public:
    kv_pool() = default;
    ~kv_pool();
    kv_pool(const kv_pool &) = delete;
    kv_pool & operator=(const kv_pool &) = delete;

    // full: the pooled classes at the load's capacity (n_cells a slot), half: the same classes at n_cells / 2 (a part
    // whose bytes halve is per cell, one whose bytes stay is fixed). Checks the layout (every per-cell part and its
    // page slice on the granularity, the fixed parts a tail), reserves each slot's range of each class on each card,
    // maps and zeroes the fixed tails, and creates every physical page (none mapped). Synchronous; false with nothing
    // held when a step fails. KV8: pool_cells (0: n_cells), a multiple of page_cells and at least n_cells, is the
    // physical pages' cells, shared by the slots; each slot's logical range stays n_cells
    bool init(const model & m, const std::vector<buffer_class> & full, const std::vector<buffer_class> & half,
              int32_t n_slots, int32_t n_cells, int32_t page_cells, std::string & err, int32_t pool_cells = 0);

    // slot s's range of class name on card (null for a class not pooled)
    uint8_t * base(int32_t slot, const std::string & name, int32_t card) const;
    // the slot's logical prefix of at least cells cells (capped at the capacity) mapped on both cards, the new pages
    // zeroed; one transaction: false with the slot's mapping unchanged when the free pages do not suffice (err says
    // so, capacity set) or a driver call fails
    bool ensure(int32_t slot, int64_t cells, std::string & err, bool * capacity = nullptr);
    // the slot's pages past the first pages_for(keep_cells) unmapped and freed (keep_cells = 0: all). The caller has
    // drained every consumer of them (both devices).
    bool trim(int32_t slot, int64_t keep_cells, std::string & err);
    // the bytes of a pooled class's part that hold memory for the slot: its mapped prefix (per-cell) or its bytes
    int64_t live_bytes(int32_t slot, const std::string & name, int32_t part, int32_t card) const;
    int64_t mapped_cells(int32_t slot) const { return int64_t(pages_.mapped(slot)) * page_cells_; }
    int64_t capacity_cells() const { return int64_t(pages_.n_pages()) * page_cells_; } // the physical pages' cells
    int64_t slot_cells() const { return n_cells_; }                                    // a slot's logical range
    const kv_pages & pages() const { return pages_; }
    int32_t page_cells() const { return page_cells_; }
    int64_t page_bytes(int32_t card) const { return page_bytes_[card]; }
    int64_t granularity(int32_t card) const { return gran_[card]; }
    // the startup ledger's lines: page geometry, physical and virtual bytes a card, the fixed tails
    std::vector<std::string> ledger() const;
    // TEST ONLY (state-test mu1): the next mapping fails on card at its slice `after` (a both-card transaction's
    // partial failure); -1: off
    void test_fail_map(int32_t card, int32_t after) {
        fail_card_ = card;
        fail_after_ = after;
    }
    // the slots, and the virtual range's bytes of a pooled class on a card (the class's allocation, on the granularity)
    int32_t n_slots() const { return n_slots_; }
    int64_t range_bytes(const std::string & name, int32_t card) const;

  private:
    struct slice {
        int32_t cls = 0, part = 0;
        int64_t off = 0;   // the part's offset in the class's range
        int64_t bytes = 0; // a page's slice of it
        int64_t at = 0;    // its offset in a physical page
    };
    struct pclass {
        std::string name;
        int64_t alloc[2] = {0, 0};    // the class's bytes a card (the plan's allocation)
        int64_t range[2] = {0, 0};    // its virtual range a card, on the granularity
        int64_t tail_off[2] = {0, 0}; // the fixed parts' start (range when none)
        int64_t tail[2] = {0, 0};     // the fixed tail's physical bytes (0: none)
        std::vector<int8_t> per_cell; // per part: 1 per cell
    };
    bool map_page(int32_t slot, int32_t logical, int32_t phys, std::string & err);
    bool unmap_page(int32_t slot, int32_t logical, std::string & err);
    void release_all();

    const model * m_ = nullptr;
    int32_t n_slots_ = 0, page_cells_ = 0, n_cells_ = 0;
    int64_t gran_[2] = {0, 0};
    int64_t page_bytes_[2] = {0, 0};
    std::vector<pclass> cls_;
    std::vector<slice> slices_[2];         // per card
    std::vector<uint64_t> page_handle_[2]; // per card, [physical page * slices + slice]
    std::vector<uint64_t> va_[2];          // per card, [slot * classes + class] the range's start (0: none)
    std::vector<uint64_t> tail_handle_[2]; // per card, [slot * classes + class] (0: none)
    kv_pages pages_;
    int32_t fail_card_ = -1, fail_after_ = -1; // TEST ONLY (test_fail_map)
};

} // namespace fn::model
