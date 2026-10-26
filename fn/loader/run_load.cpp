#include "fn/loader/run_load.h"

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/grids.h"
#include "fn/model/kv_pool.h"
#include "fn/model/kv_tier.h"
#include "fn/model/state.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstring>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

namespace fn::loader {

namespace {

bool cuda_ok(cudaError_t e, const char * what, std::string & err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

bool fill_ex1_grids(const model::model & m, std::string & err) {
    const model::reserved_class * ws = m.state_class("kernel workspaces and constants");
    int32_t part = -1;
    for (int32_t p = 0; ws != nullptr && p < (int32_t)ws->plan.parts.size(); ++p) {
        if (ws->plan.parts[p].name == "EX1 grid table") {
            part = p;
        }
    }
    // FNFL's constant tables (grids.h), not ggml-base's
    std::vector<uint8_t> table(experts::grid_bytes);
    if (part < 0 || !experts::grid_table(experts::grid_src, experts::grid_src_bytes, table.data())) {
        err = "the EX1 grid table part or its sources";
        return false;
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemcpy(ws->part_ptr(part, c), table.data(), table.size(), cudaMemcpyHostToDevice), "grids",
                     err)) {
            return false;
        }
    }
    return true;
}

namespace {

// one class reserved on both cards and zeroed (a host class in pinned host memory)
bool reserve_class(model::model & m, const model::buffer_class & plan, int64_t & host_bytes,
                   std::unique_ptr<model::reserved_class> & out, std::string & err) {
    auto rc = std::make_unique<model::reserved_class>();
    rc->plan = plan;
    // ME1: on one device the prefill checkpoints are held in pinned host memory (completed copies, PF1's restore by
    // cudaMemcpyDefault); with the host-backed KV tier the target and draft KV too (mapped)
    const bool host = model::host_class(plan.name, m.single, m.kv_host);
    const bool mapped = plan.name != "prefill checkpoints (PF1)";
    for (int32_t c = 0; c < 2; ++c) {
        const int64_t bytes = plan.allocation(c);
        if (!(host ? rc->buf[c].allocate_host(m.device[c], (size_t)bytes, err, mapped)
                   : rc->buf[c].allocate(m.device[c], (size_t)bytes, err))) {
            err = plan.name + ": " + err;
            return false;
        }
        if (host && bytes > 0) {
            std::memset(rc->buf[c].data(), 0, (size_t)bytes);
            host_bytes += bytes;
        } else if (bytes > 0 && (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
                                 !cuda_ok(cudaMemset(rc->buf[c].data(), 0, (size_t)bytes), "cudaMemset", err))) {
            return false;
        }
    }
    out = std::move(rc);
    return true;
}

// MU1: slot s's view of a pooled class (its virtual range in the pool on each card)
std::unique_ptr<model::reserved_class> pooled_view(const model::model & m, const model::buffer_class & plan,
                                                   int32_t s) {
    auto rc = std::make_unique<model::reserved_class>();
    rc->plan = plan;
    rc->pool = m.pool.get();
    rc->pool_slot = s;
    for (int32_t c = 0; c < 2; ++c) {
        rc->buf[c].view(m.device[c], m.pool->base(s, plan.name, c), size_t(plan.allocation(c)));
    }
    return rc;
}

// MXMU1: /proc/meminfo's MemAvailable in bytes, -1 when unreadable
int64_t host_available_bytes() {
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return -1;
    }
    char line[256];
    long long kib = -1;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        if (std::sscanf(line, "MemAvailable: %lld kB", &kib) == 1) {
            break;
        }
    }
    std::fclose(f);
    return kib < 0 ? -1 : int64_t(kib) * 1024;
}

} // namespace

bool reserve_state(model::model & m, const model::state_params & sp, int32_t n_slots, int64_t & host_bytes,
                   std::string & err, int32_t pool_cells) {
    // TTTS1: a split stage (a window) serves up to 8 slots; one box up to 4
    const int32_t max_slots = m.win.whole() ? 4 : 8;
    if (n_slots < 1 || n_slots > max_slots) {
        err = "reserve: " + std::to_string(n_slots) + " slots (1 .. " + std::to_string(max_slots) + ")";
        return false;
    }
    if (sp.kv_q8 && (m.single || m.kv_host)) {
        err = "reserve: the 8-bit KV (KV8) is resident TP on two devices only";
        return false;
    }
    m.n_slots = n_slots;
    m.kv_q8 = sp.kv_q8;
    m.n_cells = sp.n_cells;
    if (!(sp.win.lo == m.win.lo && sp.win.hi == m.win.hi && sp.win.head == m.win.head)) {
        err = "reserve: the state's window is not the model's (TTTS1)";
        return false;
    }
    const std::vector<model::buffer_class> plans = model::state_classes(sp);
    // MU1: resident TP shares the KV pool among 2..4 slots. MXMU1: one card with the expert cache and the host-backed
    // tier serves two, each slot's classes its own (its history in host memory, its window on the device), no pool
    const bool pooled = n_slots > 1 && !m.single && !m.kv_host && !m.expert_cache;
    const bool one_card = n_slots > 1 && m.single && m.kv_host && m.expert_cache;
    if (n_slots > 1 && !pooled && !(one_card && n_slots == 2)) {
        err = "reserve: " + std::to_string(n_slots) +
              " slots: more than one slot needs resident TP on two devices (MU1), or two slots on one card with the "
              "host-backed KV (MXMU1)";
        return false;
    }
    if (one_card) {
        // every slot's whole history in pinned host memory: refused with the arithmetic when the host cannot hold it
        int64_t slot_host = 0, engine_host = 0;
        for (const model::buffer_class & plan : plans) {
            if (!model::host_class(plan.name, m.single, m.kv_host)) {
                continue;
            }
            (model::slot_class(plan.name) ? slot_host : engine_host) += plan.allocation(0) + plan.allocation(1);
        }
        const int64_t need = slot_host * n_slots + engine_host;
        const int64_t avail = host_available_bytes();
        std::fprintf(stderr,
                     "FNFL: MXMU1 host ledger: %d slots x %.3f GiB pinned history (target and draft KV, index keys) + "
                     "%.3f GiB the engine's (checkpoints) = %.3f GiB; MemAvailable %.3f GiB\n",
                     n_slots, double(slot_host) / 1073741824.0, double(engine_host) / 1073741824.0,
                     double(need) / 1073741824.0, double(avail) / 1073741824.0);
        if (avail < 0 || need > avail) {
            err = "reserve: " + std::to_string(n_slots) + " slots' host histories need " + std::to_string(need) +
                  " bytes (" + std::to_string(n_slots) + " x " + std::to_string(slot_host) + " + " +
                  std::to_string(engine_host) + "), the host has " + std::to_string(avail) +
                  " available (MemAvailable" + (avail < 0 ? ", unreadable" : "") + "): launch with -np 1 or less -c";
            return false;
        }
    }
    if (pooled) {
        model::state_params hp = sp;
        hp.n_cells = sp.n_cells / 2;
        const std::vector<model::buffer_class> halves = model::state_classes(hp);
        std::vector<model::buffer_class> full, half;
        for (size_t i = 0; i < plans.size() && i < halves.size(); ++i) {
            if (model::pooled_class(plans[i].name)) {
                full.push_back(plans[i]);
                half.push_back(halves[i]);
            }
        }
        m.pool = std::make_unique<model::kv_pool>();
        if (!m.pool->init(m, full, half, n_slots, sp.n_cells, model::kv_page_cells, err, pool_cells)) {
            m.pool.reset();
            return false;
        }
        m.slot_state.clear();
        m.slot_state.resize(size_t(n_slots - 1));
    }
    if (one_card) {
        m.slot_state.clear();
        m.slot_state.resize(size_t(n_slots - 1));
    }
    for (const model::buffer_class & plan : plans) {
        if (pooled && model::pooled_class(plan.name)) {
            m.state.push_back(pooled_view(m, plan, 0));
            for (int32_t s = 1; s < n_slots; ++s) {
                m.slot_state[size_t(s - 1)].push_back(pooled_view(m, plan, s));
            }
            continue;
        }
        std::unique_ptr<model::reserved_class> rc;
        if (!reserve_class(m, plan, host_bytes, rc, err)) {
            return false;
        }
        m.state.push_back(std::move(rc));
        for (int32_t s = 1; s < n_slots && model::slot_class(plan.name); ++s) {
            std::unique_ptr<model::reserved_class> rs;
            if (!reserve_class(m, plan, host_bytes, rs, err)) {
                return false;
            }
            m.slot_state[size_t(s - 1)].push_back(std::move(rs));
        }
    }
    for (const model::buffer_class & plan : pooled ? model::lane_classes() : std::vector<model::buffer_class>()) {
        std::unique_ptr<model::reserved_class> rc;
        if (!reserve_class(m, plan, host_bytes, rc, err)) {
            return false;
        }
        m.state.push_back(std::move(rc));
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaDeviceSynchronize(), "zeroing the state", err)) {
            return false;
        }
    }
    return true;
}

void release_state(model::model & m) {
    m.state.clear();
    m.slot_state.clear();
    m.pool.reset();
    m.n_slots = 1;
    m.kv_q8 = false;
}

bool load_for_run(const load_params & p, int32_t n_cells, model::model & m, run_load_report & rep, std::string & err) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int32_t c = 0; c < (p.device[0] == p.device[1] ? 1 : 2); ++c) {
        if (!cuda_ok(cudaSetDevice(p.device[c]), "cudaSetDevice", err) || !cuda_ok(cudaFree(nullptr), "context", err)) {
            return false;
        }
    }
    load_times & lt = rep.times;
    rep.ple_table = p.ple_table;
    if (!open_model(p, m, lt, err)) {
        return false;
    }
    rep.open_s = since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    // PN1: the table's populate reads shard 2 into the page cache on its own thread while the fill runs; the
    // registration waits for the fill, so the driver pins nothing while the fill's copies are in flight
    std::shared_ptr<runtime::ple_pin> pin;
    std::string pin_err;
    bool populated = false;
    std::thread populate;
    if (p.ple_table == runtime::ple_residency::full) {
        pin = std::make_shared<runtime::ple_pin>();
        populate = std::thread([&] { populated = pin->populate(m.ple_table, m.ple_bytes, pin_err); });
    }
    const bool filled = allocate_weights(m, err) && fill_weights(p, m, lt, err);
    if (populate.joinable()) {
        populate.join();
    }
    if (!filled) {
        return false;
    }
    if (pin != nullptr) {
        if (!populated || !pin->pin(pin_err)) {
            err = pin_err;
            return false;
        }
        lt.ple_populate_s = pin->populate_s;
        lt.ple_register_s = pin->register_s;
        lt.ple_pin_s = pin->populate_s + pin->register_s;
        rep.ple_pinned_bytes = pin->pinned_bytes();
        rep.ple_foll_pin[0] = pin->foll_pin_before;
        rep.ple_foll_pin[1] = pin->foll_pin_after;
        m.ple_pin = pin;
    }
    rep.fill_s = since(t1);
    const auto t2 = std::chrono::steady_clock::now();
    model::state_params sp;
    sp.n_cells = n_cells;
    sp.ple_conv_kernel = m.ple_conv_kernel;
    sp.ple_ngram = m.ple_ngram;
    sp.kv_window = m.kv_host ? m.kv_window : 0;
    sp.single = m.single;
    sp.kv_q8 = p.kv_q8;
    sp.win             = m.win;  // TTTS1
    if (!reserve_state(m, sp, p.n_slots, rep.host_state_bytes, err, p.pool_cells)) {
        return false;
    }
    if (!fill_ex1_grids(m, err)) {
        return false;
    }
    if (!m.single && !xchg::enable_peer_access(m.device[0], m.device[1])) { // ME1: one device needs none
        err = "peer access";
        return false;
    }
    m.rings = std::make_unique<xchg::rings>(m.device[0], m.device[1]);
    if (!m.rings->ok()) {
        err = "exchange rings";
        return false;
    }
    rep.reserve_s = since(t2);
    for (int32_t c = 0; c < 2; ++c) {
        size_t f = 0, t = 0;
        if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaMemGetInfo(&f, &t), "cudaMemGetInfo", err)) {
            return false;
        }
        rep.free_after[c] = (int64_t)f;
    }
    rep.total_s = since(t0);
    return true;
}

std::vector<std::string> ple_table_lines(const run_load_report & rep) {
    if (rep.ple_table == runtime::ple_residency::lazy) {
        return {"PLE table: lazy, not pinned (the plain mapping; PL1's hints before each gather)"};
    }
    char b[2][320];
    snprintf(b[0], sizeof(b[0]),
             "PLE table: full, %llu bytes pinned (in place: madvise MADV_POPULATE_READ, then cudaHostRegister "
             "Portable | ReadOnly on the file mapping); kernel FOLL_PIN net pages %lld before the registration, %lld "
             "after (+%lld; the range is %llu pages)",
             static_cast<unsigned long long>(rep.ple_pinned_bytes), static_cast<long long>(rep.ple_foll_pin[0]),
             static_cast<long long>(rep.ple_foll_pin[1]),
             static_cast<long long>(rep.ple_foll_pin[1] - rep.ple_foll_pin[0]),
             static_cast<unsigned long long>(rep.ple_pinned_bytes / 4096));
    snprintf(b[1], sizeof(b[1]), "load_times: ple_pin_s %.2f (populate %.2f, beside the fill; register %.2f, after it)",
             rep.times.ple_pin_s, rep.times.ple_populate_s, rep.times.ple_register_s);
    return {b[0], b[1]};
}

} // namespace fn::loader
