#include "fn/loader/run_load.h"

#include "fn/kernels/experts/experts.h"
#include "fn/kernels/experts/grids.h"
#include "fn/model/state.h"
#include "fn/xchg/xchg.h"

#include <cuda_runtime_api.h>

#include <chrono>
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

bool load_for_run(const load_params & p, int32_t n_cells, model::model & m, run_load_report & rep, std::string & err) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int32_t c = 0; c < 2; ++c) {
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
    for (const model::buffer_class & plan : model::state_classes(sp)) {
        auto rc = std::make_unique<model::reserved_class>();
        rc->plan = plan;
        for (int32_t c = 0; c < 2; ++c) {
            const int64_t bytes = plan.allocation(c);
            if (!rc->buf[c].allocate(m.device[c], (size_t)bytes, err)) {
                err = plan.name + ": " + err;
                return false;
            }
            if (bytes > 0 && (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
                              !cuda_ok(cudaMemset(rc->buf[c].data(), 0, (size_t)bytes), "cudaMemset", err))) {
                return false;
            }
        }
        m.state.push_back(std::move(rc));
    }
    for (int32_t c = 0; c < 2; ++c) {
        if (!cuda_ok(cudaSetDevice(m.device[c]), "cudaSetDevice", err) ||
            !cuda_ok(cudaDeviceSynchronize(), "zeroing the state", err)) {
            return false;
        }
    }
    if (!fill_ex1_grids(m, err)) {
        return false;
    }
    if (!xchg::enable_peer_access(m.device[0], m.device[1])) {
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
