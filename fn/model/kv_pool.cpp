#include "fn/model/kv_pool.h"

#include "fn/model/model.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace fn::model {

namespace {

bool fail(std::string & err, const std::string & what) {
    err = what;
    return false;
}

// the driver's VMM calls through the runtime's entry points (no driver link, as target.cpp's node query)
struct vmm_api {
    CUresult (*granularity)(size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (*create)(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *,
                       unsigned long long) = nullptr;
    CUresult (*release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*reserve)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (*address_free)(CUdeviceptr, size_t) = nullptr;
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (*unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (*set_access)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t) = nullptr;
    bool ok = false;
};

template <typename F> bool entry(const char * name, F & out) {
    void * fn = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
    if (cudaGetDriverEntryPointByVersion(name, &fn, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || fn == nullptr) {
        cudaGetLastError();
        return false;
    }
    // reason: the entry point is returned untyped; its type is the driver's declaration
    out = reinterpret_cast<F>(fn);
    return true;
}

const vmm_api & vmm() {
    static const vmm_api api = [] {
        vmm_api a;
        a.ok = entry("cuMemGetAllocationGranularity", a.granularity) && entry("cuMemCreate", a.create) &&
               entry("cuMemRelease", a.release) && entry("cuMemAddressReserve", a.reserve) &&
               entry("cuMemAddressFree", a.address_free) && entry("cuMemMap", a.map) && entry("cuMemUnmap", a.unmap) &&
               entry("cuMemSetAccess", a.set_access);
        return a;
    }();
    return api;
}

bool cu_ok(CUresult r, const char * what, std::string & err) {
    if (r != CUDA_SUCCESS) {
        err = std::string("KV pool: ") + what + ": CUresult " + std::to_string(static_cast<int32_t>(r));
        return false;
    }
    return true;
}

CUmemAllocationProp prop_of(int32_t device) {
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = device;
    return p;
}

int64_t round_up(int64_t n, int64_t g) {
    return (n + g - 1) / g * g;
}

} // namespace

// ---- kv_pages ----

void kv_pages::init(int32_t n_slots, int32_t n_pages) {
    n_pages_ = std::max(0, n_pages);
    map_.assign(size_t(std::max(0, n_slots)), {});
    free_.clear();
    for (int32_t p = 0; p < n_pages_; ++p) {
        free_.push_back(p);
    }
}

int32_t kv_pages::mapped(int32_t slot) const {
    return slot < 0 || slot >= n_slots() ? 0 : static_cast<int32_t>(map_[size_t(slot)].size());
}

int32_t kv_pages::physical(int32_t slot, int32_t logical) const {
    if (slot < 0 || slot >= n_slots() || logical < 0 || logical >= mapped(slot)) {
        return -1;
    }
    return map_[size_t(slot)][size_t(logical)];
}

int32_t kv_pages::pages_for(int64_t cells, int32_t page_cells, int32_t n_pages) {
    if (cells <= 0 || page_cells <= 0) {
        return 0;
    }
    return static_cast<int32_t>(std::min<int64_t>(n_pages, (cells + page_cells - 1) / page_cells));
}

bool kv_pages::take(int32_t slot, int32_t want, std::vector<int32_t> & taken) {
    taken.clear();
    if (slot < 0 || slot >= n_slots() || want > n_pages_) {
        return false;
    }
    const int32_t need = want - mapped(slot);
    if (need <= 0) {
        return true;
    }
    if (need > free_pages()) {
        return false;
    }
    taken.assign(free_.begin(), free_.begin() + need);
    free_.erase(free_.begin(), free_.begin() + need);
    std::vector<int32_t> & m = map_[size_t(slot)];
    m.insert(m.end(), taken.begin(), taken.end());
    return true;
}

void kv_pages::give(int32_t slot, int32_t keep, std::vector<int32_t> & out) {
    out.clear();
    if (slot < 0 || slot >= n_slots()) {
        return;
    }
    std::vector<int32_t> & m = map_[size_t(slot)];
    keep = std::max(0, keep);
    while (static_cast<int32_t>(m.size()) > keep) {
        out.push_back(m.back());
        m.pop_back();
    }
    free_.insert(free_.end(), out.begin(), out.end());
    std::sort(free_.begin(), free_.end());
}

int32_t kv_reclaim_victim(const std::vector<kv_slot_view> & s, int32_t for_slot) {
    int32_t victim = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(s.size()); ++i) {
        const kv_slot_view & y = s[size_t(i)];
        if (i == for_slot || y.active || y.pages == 0) {
            continue;
        }
        if (victim < 0 || y.released < s[size_t(victim)].released) {
            victim = i;
        }
    }
    return victim;
}

// ---- kv_pool ----

kv_pool::~kv_pool() {
    release_all();
}

void kv_pool::release_all() {
    const vmm_api & a = vmm();
    if (m_ == nullptr || !a.ok) {
        return;
    }
    std::string e;
    for (int32_t s = 0; s < n_slots_; ++s) {
        for (int32_t l = pages_.mapped(s) - 1; l >= 0; --l) {
            unmap_page(s, l, e);
        }
        std::vector<int32_t> out;
        pages_.give(s, 0, out);
    }
    const size_t nc = cls_.size();
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        for (int32_t s = 0; s < n_slots_; ++s) {
            for (size_t k = 0; k < nc; ++k) {
                const size_t i = size_t(s) * nc + k;
                const pclass & pc = cls_[k];
                if (i < tail_handle_[c].size() && tail_handle_[c][i] != 0) {
                    a.unmap(CUdeviceptr(va_[c][i]) + CUdeviceptr(pc.tail_off[c]), size_t(pc.tail[c]));
                    a.release(CUmemGenericAllocationHandle(tail_handle_[c][i]));
                    tail_handle_[c][i] = 0;
                }
                if (i < va_[c].size() && va_[c][i] != 0) {
                    a.address_free(CUdeviceptr(va_[c][i]), size_t(pc.range[c]));
                    va_[c][i] = 0;
                }
            }
        }
        for (uint64_t & h : page_handle_[c]) {
            if (h != 0) {
                a.release(CUmemGenericAllocationHandle(h));
                h = 0;
            }
        }
    }
    m_ = nullptr;
}

bool kv_pool::init(const model & m, const std::vector<buffer_class> & full, const std::vector<buffer_class> & half,
                   int32_t n_slots, int32_t n_cells, int32_t page_cells, std::string & err, int32_t pool_cells) {
    const vmm_api & a = vmm();
    if (!a.ok) {
        return fail(err, "KV pool: the driver's virtual memory calls are not available");
    }
    if (pool_cells == 0) {
        pool_cells = n_cells;
    }
    if (m.single || n_slots < 2 || full.size() != half.size() || page_cells < 1 || n_cells < page_cells ||
        n_cells % page_cells != 0 || pool_cells < n_cells || pool_cells % page_cells != 0) {
        return fail(err, "KV pool: two devices, 2 .. 4 slots and a capacity that is a multiple of " +
                             std::to_string(page_cells) + " cells (" + std::to_string(n_cells) + " a slot, " +
                             std::to_string(pool_cells) + " in the pool, at least a slot's)");
    }
    m_ = &m;
    n_slots_ = n_slots;
    page_cells_ = page_cells;
    n_cells_ = n_cells;
    // n_pages: a slot's logical pages, which divide each per-cell part into page slices; n_phys: the physical pages
    const int32_t n_pages = n_cells / page_cells;
    const int32_t n_phys = pool_cells / page_cells;
    for (int32_t c = 0; c < 2; ++c) {
        const CUmemAllocationProp p = prop_of(m.device[c]);
        size_t g = 0;
        if (!cu_ok(a.granularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity", err)) {
            m_ = nullptr;
            return false;
        }
        gran_[c] = static_cast<int64_t>(g);
    }
    // the layout: per-cell parts on the granularity with page slices on it, the fixed parts after them
    cls_.clear();
    for (int32_t c = 0; c < 2; ++c) {
        slices_[c].clear();
        page_bytes_[c] = 0;
    }
    for (size_t k = 0; k < full.size(); ++k) {
        const buffer_class & f = full[k];
        const buffer_class & h = half[k];
        if (f.name != h.name || f.parts.size() != h.parts.size()) {
            m_ = nullptr;
            return fail(err, "KV pool: the half-capacity plan of " + f.name + " differs in its parts");
        }
        pclass pc;
        pc.name = f.name;
        pc.per_cell.assign(f.parts.size(), 0);
        for (int32_t c = 0; c < 2; ++c) {
            const int64_t g = gran_[c];
            pc.alloc[c] = f.allocation(c);
            pc.range[c] = round_up(pc.alloc[c], g);
            int64_t cell_end = 0, fixed_start = -1;
            for (int32_t p = 0; p < static_cast<int32_t>(f.parts.size()); ++p) {
                const int64_t b = f.parts[size_t(p)].bytes[c], bh = h.parts[size_t(p)].bytes[c];
                if (b == 0) {
                    continue;
                }
                const int64_t off = f.offset(p, c);
                if (b == 2 * bh) {
                    const int64_t sl = b / n_pages;
                    if (b % n_pages != 0 || sl % g != 0 || off % g != 0 || (fixed_start >= 0 && off >= fixed_start)) {
                        m_ = nullptr;
                        return fail(err, "KV pool: " + f.name + " / " + f.parts[size_t(p)].name + " on card " +
                                             std::to_string(c) + ": offset " + std::to_string(off) + ", " +
                                             std::to_string(b) + " bytes: not page slices on the " + std::to_string(g) +
                                             "-byte granularity, or after a fixed part");
                    }
                    pc.per_cell[size_t(p)] = 1;
                    slice s;
                    s.cls = static_cast<int32_t>(k);
                    s.part = p;
                    s.off = off;
                    s.bytes = sl;
                    s.at = page_bytes_[c];
                    page_bytes_[c] += sl;
                    slices_[c].push_back(s);
                    cell_end = std::max(cell_end, off + b);
                } else if (b == bh) {
                    fixed_start = fixed_start < 0 ? off : std::min(fixed_start, off);
                } else {
                    m_ = nullptr;
                    return fail(err, "KV pool: " + f.name + " / " + f.parts[size_t(p)].name +
                                         " is neither per cell nor fixed");
                }
            }
            if (fixed_start >= 0 && fixed_start < cell_end) {
                m_ = nullptr;
                return fail(err, "KV pool: " + f.name + "'s fixed parts are not after its per-cell parts");
            }
            pc.tail_off[c] = fixed_start < 0 ? pc.range[c] : round_up(cell_end, g);
            if (fixed_start >= 0 && fixed_start < pc.tail_off[c]) {
                pc.tail_off[c] = fixed_start / g * g;
                if (pc.tail_off[c] < cell_end) {
                    m_ = nullptr;
                    return fail(err, "KV pool: " + f.name + "'s fixed tail shares a granule with a per-cell part");
                }
            }
            pc.tail[c] = fixed_start < 0 ? 0 : pc.range[c] - pc.tail_off[c];
        }
        cls_.push_back(pc);
    }
    pages_.init(n_slots, n_phys);
    const size_t nc = cls_.size();
    for (int32_t c = 0; c < 2; ++c) {
        va_[c].assign(size_t(n_slots) * nc, 0);
        tail_handle_[c].assign(size_t(n_slots) * nc, 0);
        page_handle_[c].clear();
    }
    const auto undo = [&](bool r) {
        if (!r) {
            release_all();
        }
        return r;
    };
    for (int32_t c = 0; c < 2; ++c) {
        if (cudaSetDevice(m.device[c]) != cudaSuccess) {
            return undo(fail(err, "KV pool: cudaSetDevice"));
        }
        const CUmemAllocationProp p = prop_of(m.device[c]);
        CUmemAccessDesc acc[2] = {};
        for (int32_t d = 0; d < 2; ++d) {
            acc[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            acc[d].location.id = m.device[d];
            acc[d].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }
        for (int32_t s = 0; s < n_slots; ++s) {
            for (size_t k = 0; k < nc; ++k) {
                const pclass & pc = cls_[k];
                const size_t i = size_t(s) * nc + k;
                if (pc.range[c] == 0) {
                    continue;
                }
                CUdeviceptr v = 0;
                if (!cu_ok(a.reserve(&v, size_t(pc.range[c]), size_t(gran_[c]), 0, 0), "address reserve", err)) {
                    return undo(false);
                }
                va_[c][i] = uint64_t(v);
                if (pc.tail[c] > 0) {
                    CUmemGenericAllocationHandle th = 0;
                    if (!cu_ok(a.create(&th, size_t(pc.tail[c]), &p, 0), "fixed tail create", err)) {
                        return undo(false);
                    }
                    tail_handle_[c][i] = uint64_t(th);
                    const CUdeviceptr at = v + CUdeviceptr(pc.tail_off[c]);
                    if (!cu_ok(a.map(at, size_t(pc.tail[c]), 0, th, 0), "fixed tail map", err) ||
                        !cu_ok(a.set_access(at, size_t(pc.tail[c]), acc, 2), "fixed tail access", err) ||
                        cudaMemset(reinterpret_cast<void *>(at), 0, size_t(pc.tail[c])) != cudaSuccess) {
                        return undo(err.empty() ? fail(err, "KV pool: fixed tail zero") : false);
                    }
                }
            }
        }
        // a page is one allocation a slice (cuMemMap maps a whole allocation: its offset must be 0, CUDA 12.8)
        const size_t ns = slices_[c].size();
        page_handle_[c].assign(size_t(n_phys) * ns, 0);
        for (int32_t pg = 0; pg < n_phys; ++pg) {
            for (size_t k = 0; k < ns; ++k) {
                CUmemGenericAllocationHandle ph = 0;
                if (!cu_ok(a.create(&ph, size_t(slices_[c][k].bytes), &p, 0), "page create", err)) {
                    err += " (page " + std::to_string(pg) + " of " + std::to_string(n_phys) + ", slice " +
                           std::to_string(k) + " on card " + std::to_string(c) + ")";
                    return undo(false);
                }
                page_handle_[c][size_t(pg) * ns + k] = uint64_t(ph);
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return undo(fail(err, "KV pool: synchronize"));
        }
    }
    return true;
}

uint8_t * kv_pool::base(int32_t slot, const std::string & name, int32_t card) const {
    if (m_ == nullptr || slot < 0 || slot >= n_slots_ || card < 0 || card > 1) {
        return nullptr;
    }
    for (size_t k = 0; k < cls_.size(); ++k) {
        if (cls_[k].name == name) {
            // reason: a device virtual address is an integer; the class's views hold it as a byte pointer
            return reinterpret_cast<uint8_t *>(va_[card][size_t(slot) * cls_.size() + k]);
        }
    }
    return nullptr;
}

bool kv_pool::map_page(int32_t slot, int32_t logical, int32_t phys, std::string & err) {
    const vmm_api & a = vmm();
    const size_t nc = cls_.size();
    for (int32_t c = 0; c < 2; ++c) {
        if (cudaSetDevice(m_->device[c]) != cudaSuccess) {
            return fail(err, "KV pool: cudaSetDevice");
        }
        CUmemAccessDesc acc[2] = {};
        for (int32_t d = 0; d < 2; ++d) {
            acc[d].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            acc[d].location.id = m_->device[d];
            acc[d].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }
        const size_t ns = slices_[c].size();
        int32_t n_mapped = 0;
        for (size_t k = 0; k < ns; ++k) {
            const slice & s = slices_[c][k];
            if (c == fail_card_ && n_mapped++ == fail_after_) {
                fail_card_ = fail_after_ = -1;
                return fail(err, "KV pool: page map (TEST: an injected failure on card " + std::to_string(c) + ")");
            }
            const CUmemGenericAllocationHandle h = CUmemGenericAllocationHandle(page_handle_[c][size_t(phys) * ns + k]);
            const CUdeviceptr at = CUdeviceptr(va_[c][size_t(slot) * nc + size_t(s.cls)]) +
                                   CUdeviceptr(s.off + int64_t(logical) * s.bytes);
            if (!cu_ok(a.map(at, size_t(s.bytes), 0, h, 0), "page map", err)) {
                return false;
            }
            if (!cu_ok(a.set_access(at, size_t(s.bytes), acc, 2), "page access", err)) {
                a.unmap(at, size_t(s.bytes));
                return false;
            }
        }
    }
    return true;
}

bool kv_pool::unmap_page(int32_t slot, int32_t logical, std::string & err) {
    const vmm_api & a = vmm();
    const size_t nc = cls_.size();
    bool ok = true;
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m_->device[c]);
        for (const slice & s : slices_[c]) {
            const CUdeviceptr at = CUdeviceptr(va_[c][size_t(slot) * nc + size_t(s.cls)]) +
                                   CUdeviceptr(s.off + int64_t(logical) * s.bytes);
            // a slice of a partly mapped page (a failed map) is not mapped: its unmap fails, harmlessly
            ok = a.unmap(at, size_t(s.bytes)) == CUDA_SUCCESS && ok;
        }
    }
    if (!ok) {
        err = "KV pool: page unmap";
    }
    return ok;
}

bool kv_pool::ensure(int32_t slot, int64_t cells, std::string & err, bool * capacity) {
    if (capacity != nullptr) {
        *capacity = false;
    }
    if (m_ == nullptr || slot < 0 || slot >= n_slots_) {
        return fail(err, "KV pool: no slot " + std::to_string(slot));
    }
    const int32_t first = pages_.mapped(slot);
    const int32_t want = kv_pages::pages_for(std::min<int64_t>(cells, n_cells_), page_cells_, pages_.n_pages());
    if (want <= first) {
        return true;
    }
    std::vector<int32_t> taken;
    if (!pages_.take(slot, want, taken)) {
        if (capacity != nullptr) {
            *capacity = true;
        }
        return fail(err, "KV pool: slot " + std::to_string(slot) + " needs " + std::to_string(want - first) +
                             " more pages of " + std::to_string(page_cells_) + " cells for " + std::to_string(cells) +
                             " cells; " + std::to_string(pages_.free_pages()) + " of " +
                             std::to_string(pages_.n_pages()) + " are free");
    }
    for (int32_t j = 0; j < static_cast<int32_t>(taken.size()); ++j) {
        std::string e2;
        if (!map_page(slot, first + j, taken[size_t(j)], err)) {
            // the transaction undone: this call's pages unmapped (the failed one partly) and returned
            for (int32_t u = j; u >= 0; --u) {
                unmap_page(slot, first + u, e2);
            }
            std::vector<int32_t> back;
            pages_.give(slot, first, back);
            return false;
        }
    }
    const size_t nc = cls_.size();
    for (int32_t c = 0; c < 2; ++c) {
        if (cudaSetDevice(m_->device[c]) != cudaSuccess) {
            return fail(err, "KV pool: cudaSetDevice");
        }
        for (int32_t j = 0; j < static_cast<int32_t>(taken.size()); ++j) {
            for (const slice & s : slices_[c]) {
                const uint64_t at =
                    va_[c][size_t(slot) * nc + size_t(s.cls)] + uint64_t(s.off + int64_t(first + j) * s.bytes);
                // reason: the mapped virtual address as the runtime's pointer
                if (cudaMemset(reinterpret_cast<void *>(at), 0, size_t(s.bytes)) != cudaSuccess) {
                    return fail(err, "KV pool: zeroing a new page");
                }
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return fail(err, "KV pool: zeroing a new page");
        }
    }
    return true;
}

bool kv_pool::trim(int32_t slot, int64_t keep_cells, std::string & err) {
    if (m_ == nullptr || slot < 0 || slot >= n_slots_) {
        return fail(err, "KV pool: no slot " + std::to_string(slot));
    }
    const int32_t keep = kv_pages::pages_for(keep_cells, page_cells_, pages_.n_pages());
    bool ok = true;
    for (int32_t l = pages_.mapped(slot) - 1; l >= keep; --l) {
        ok = unmap_page(slot, l, err) && ok;
    }
    std::vector<int32_t> out;
    pages_.give(slot, keep, out);
    return ok;
}

int64_t kv_pool::live_bytes(int32_t slot, const std::string & name, int32_t part, int32_t card) const {
    for (const pclass & pc : cls_) {
        if (pc.name != name || part < 0 || part >= static_cast<int32_t>(pc.per_cell.size())) {
            continue;
        }
        if (pc.per_cell[size_t(part)] == 0) {
            return -1; // the caller's plan bytes (fixed)
        }
        for (const slice & s : slices_[card]) {
            if (cls_[size_t(s.cls)].name == name && s.part == part) {
                return int64_t(pages_.mapped(slot)) * s.bytes;
            }
        }
    }
    return -1;
}

int64_t kv_pool::range_bytes(const std::string & name, int32_t card) const {
    for (const pclass & pc : cls_) {
        if (pc.name == name && card >= 0 && card < 2) {
            return pc.range[card];
        }
    }
    return 0;
}

std::vector<std::string> kv_pool::ledger() const {
    std::vector<std::string> out;
    char b[512];
    snprintf(b, sizeof b,
             "KV pool (MU1): %d pages of %d cells (%lld cells, the whole capacity, shared by %d slots; each slot's "
             "logical range is %lld cells); page %lld / %lld bytes on cards 0 / 1 (granularity %lld / %lld); "
             "physical %.3f / %.3f GiB",
             pages_.n_pages(), page_cells_, (long long)capacity_cells(), n_slots_, (long long)n_cells_,
             (long long)page_bytes_[0], (long long)page_bytes_[1], (long long)gran_[0], (long long)gran_[1],
             double(page_bytes_[0]) * pages_.n_pages() / 1073741824.0,
             double(page_bytes_[1]) * pages_.n_pages() / 1073741824.0);
    out.push_back(b);
    int64_t va[2] = {0, 0}, tail[2] = {0, 0};
    for (const pclass & pc : cls_) {
        for (int32_t c = 0; c < 2; ++c) {
            va[c] += pc.range[c] * n_slots_;
            tail[c] += pc.tail[c] * n_slots_;
        }
    }
    snprintf(b, sizeof b,
             "KV pool (MU1): virtual %.3f / %.3f GiB reserved (%d slots x %zu classes); fixed tails mapped %.2f / %.2f "
             "MiB; slices a page %zu / %zu",
             double(va[0]) / 1073741824.0, double(va[1]) / 1073741824.0, n_slots_, cls_.size(),
             double(tail[0]) / 1048576.0, double(tail[1]) / 1048576.0, slices_[0].size(), slices_[1].size());
    out.push_back(b);
    return out;
}

} // namespace fn::model
