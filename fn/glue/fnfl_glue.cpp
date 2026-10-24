// fnfl.h for the server: checks the GGUF against FNFL's model, loads it on both cards (the loader fnfl-drive uses) and
// drives the runtime's engine (fn/runtime/runtime.h, SV1).

#include "fn/fnfl.h"

#include "fn/borrowed/gguf_meta.h"
#include "fn/loader/run_load.h"
#include "fn/model/kv_pool.h"
#include "fn/model/model.h"
#include "fn/model/shapes.h"
#include "fn/runtime/expert_cache.h"
#include "fn/runtime/runtime.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <memory>

static_assert(sizeof(fnfl_step_result::tokens) / sizeof(int32_t) == fn::runtime::max_step_tokens,
              "fnfl.h's step result must hold a step");

namespace {

// the draft subset when the server names none (DR1 Stage C's)
constexpr const char * default_subset = "/home/jeremy/src/flashnext/work/DR1/stageC/subset/draft-subset.ids";

} // namespace

struct fnfl_engine {
    fn::runtime::engine rt;
};

static fnfl_status to_status(fn::runtime::run_status s) {
    switch (s) {
        case fn::runtime::run_status::ok:
            return FNFL_OK;
        case fn::runtime::run_status::not_built:
            return FNFL_NOT_BUILT;
        case fn::runtime::run_status::bad_request:
            return FNFL_BAD_REQUEST;
        case fn::runtime::run_status::error:
            return FNFL_ERROR;
        case fn::runtime::run_status::cancelled:
            return FNFL_CANCELLED;
        case fn::runtime::run_status::no_context:
            return FNFL_NO_CONTEXT;
    }
    return FNFL_ERROR;
}

const char * fnfl_status_str(fnfl_status status) {
    switch (status) {
        case FNFL_OK:
            return "ok";
        case FNFL_NOT_BUILT:
            return "FNFL engine not built";
        case FNFL_BAD_REQUEST:
            return "FNFL: invalid request";
        case FNFL_ERROR:
            return "FNFL: engine error";
        case FNFL_CANCELLED:
            return "FNFL: prefill cancelled";
        case FNFL_NO_CONTEXT:
            return "FNFL: the KV pool is full";
    }
    return "FNFL: unknown status";
}

// the file must be the model shapes.h describes; returns an empty string when it is
static std::string check_shapes(const fnfl_gguf_meta & m) {
    struct key {
        const char * name;
        int64_t got;
        int64_t want;
    };
    const key keys[] = {
        {"block_count", m.n_layer, fn::n_layer},
        {"embedding_length", m.n_embd, fn::n_embd},
        {"context_length", m.n_ctx_train, fn::n_ctx_train},
        {"attention.head_count", m.n_head, fn::n_head},
        {"attention.head_count_kv", m.n_head_kv, fn::n_head_kv},
        {"expert_count", m.n_expert, fn::n_expert},
        {"hyper_connection.count", m.n_hc, fn::n_hc},
    };
    if (m.arch != fn::arch) {
        return "FNFL serves " + std::string(fn::arch) + " only; this GGUF is '" + m.arch + "'";
    }
    for (const auto & k : keys) {
        if (k.got != k.want) {
            return std::string("GGUF key ") + fn::arch + "." + k.name + " is " + std::to_string(k.got) +
                   ", FNFL is built for " + std::to_string(k.want);
        }
    }
    return std::string();
}

bool fnfl_settings(const fnfl_config & c, fn::settings::values & v, std::string & text, std::string & err) {
    v = fn::settings::values();
    for (const auto & kv : c.settings) {
        const int32_t i = fn::settings::find(kv.first.c_str());
        std::string e;
        if (i < 0 || !fn::settings::parse(i, kv.second, v, e)) {
            err = kv.first + " " + kv.second + ": " + (i < 0 ? "not a setting" : e);
            return false;
        }
    }
    const bool cached = c.single_card >= 0 || c.cached_tp;
    if ((!cached && v.any(1)) || (!c.kv_host && v.any(2))) {
        err = "an expert cache setting needs --fnfl-single-card or --fnfl-cached-tp, a K/V one --fnfl-kv-host";
        return false;
    }
    text = "ME3 settings (executor): " + fn::settings::readback(v, 0) + "\n";
    if (cached) {
        text += "ME3 settings (cache): " + fn::settings::readback(v, 1) + "; reserve " +
                std::to_string(c.expert_reserve_mib) + " MiB\n";
    }
    if (c.kv_host) {
        text += "ME3 settings (K/V): " + fn::settings::readback(v, 2) + "\n";
    }
    text += "ME3 settings (request JSON): " + fn::settings::json(v) + "\n";
    return true;
}

// MU1: the multi-slot startup ledger, per card: the pool, each slot's own classes, the engine's, the lanes, every
// slot's graphs, and the device's free memory after the load and after the engine (the graphs' capture)
static void startup_ledger(const fn::model::model & m, const fn::runtime::engine & rt,
                           const fn::loader::run_load_report & lr) {
    for (const std::string & line : m.pool->ledger()) {
        fprintf(stderr, "FNFL: %s\n", line.c_str());
    }
    int64_t slot_b[2] = {0, 0}, engine_b[2] = {0, 0}, lane_b[2] = {0, 0}, pooled_tail[2] = {0, 0};
    for (const auto & rc : m.state) {
        for (int32_t c = 0; c < 2; ++c) {
            const int64_t b = rc->plan.allocation(c);
            if (rc->pool != nullptr) {
                continue;
            }
            if (fn::model::slot_class(rc->plan.name)) {
                slot_b[c] += b;
            } else if (rc->plan.name.find("(MU1)") != std::string::npos) {
                lane_b[c] += b;
            } else {
                engine_b[c] += b;
            }
        }
    }
    (void)pooled_tail;
    fprintf(stderr,
            "FNFL: MU1 ledger: %d slots; a slot's own classes (GDN state and histories, round and pass, draft round, "
            "sampler) %.2f / %.2f MiB on cards 0 / 1, %d copies; the engine's classes %.2f / %.2f MiB; the "
            "other-reader lane (its arena for %d-row chunks in %d-row halves and its staging) %.2f / %.2f MiB\n",
            m.n_slots, double(slot_b[0]) / 1048576.0, double(slot_b[1]) / 1048576.0, m.n_slots,
            double(engine_b[0]) / 1048576.0, double(engine_b[1]) / 1048576.0, fn::model::lane_rows,
            fn::model::lane_half_rows, double(lane_b[0]) / 1048576.0, double(lane_b[1]) / 1048576.0);
    int64_t graphs[2] = {0, 0};
    for (int32_t s = 0; s < rt.n_slots(); ++s) {
        for (int32_t c = 0; c < 2; ++c) {
            graphs[c] += const_cast<fn::runtime::engine &>(rt).test_target(s).graph_bytes(c);
        }
    }
    size_t f[2] = {0, 0}, tot = 0;
    for (int32_t c = 0; c < 2; ++c) {
        cudaSetDevice(m.device[c]);
        cudaMemGetInfo(&f[c], &tot);
    }
    fprintf(stderr,
            "FNFL: MU1 ledger: graphs (every slot's, measured at capture) %.1f / %.1f MiB; free after the load %.3f / "
            "%.3f GiB, after the engine (frames, lanes, graphs) %.3f / %.3f GiB; no allocation follows at a job's "
            "begin or end (the pool maps its pages, created at load)\n",
            double(graphs[0]) / 1048576.0, double(graphs[1]) / 1048576.0, double(lr.free_after[0]) / 1073741824.0,
            double(lr.free_after[1]) / 1073741824.0, double(f[0]) / 1073741824.0, double(f[1]) / 1073741824.0);
}

bool fnfl_kv_format(const fnfl_config & c, fnfl_kv_choice & out, std::string & err) {
    out = fnfl_kv_choice();
    const std::string & f = c.kv_format;
    if (!f.empty() && f != "bf16" && f != "q8") {
        err = "--fnfl-kv-format is bf16 or q8, not '" + f + "'";
        return false;
    }
    // the configurations without q8 (one card and the expert-cached ones), named as given
    std::string other;
    for (const auto & [on, name] : {std::pair<bool, const char *>{c.single_card >= 0, "--fnfl-single-card"},
                                    {c.cached_tp, "--fnfl-cached-tp"},
                                    {c.kv_host, "--fnfl-kv-host"}}) {
        if (on) {
            other += (other.empty() ? "" : " with ") + std::string(name);
        }
    }
    if (f == "q8" && !other.empty()) {
        err = "--fnfl-kv-format q8 serves resident TP on two cards (Config A) only: not with --fnfl-single-card, "
              "--fnfl-cached-tp or --fnfl-kv-host";
        return false;
    }
    out.q8 = f == "q8" || (f.empty() && other.empty());
    out.line = !f.empty() ? f : out.q8 ? "q8 (default)" : "bf16 (default: " + other + " has no q8)";
    return true;
}

// MXMU1: the two-slot one-card startup ledger: the host's pinned histories, each slot's device classes (its frame and
// window) and graphs, the engine's classes, the expert cache's bytes and pool, and the free memory. The pool is sized
// from the free memory after every slot's classes and graphs, so a one-slot launch's pool is larger by slot 1's
// device classes and graphs (counted here, not measured)
static void startup_ledger_one_card(const fn::model::model & m, const fn::runtime::engine & rt,
                                    const fn::loader::run_load_report & lr) {
    const int32_t n = rt.n_slots();
    std::vector<int64_t> dev(size_t(n), 0), host(size_t(n), 0), graphs(size_t(n), 0);
    int64_t engine_dev = 0, engine_host = 0;
    const auto add = [&](const fn::model::reserved_class & rc, int32_t s) {
        const int64_t b = rc.plan.allocation(0) + rc.plan.allocation(1); // both logical cards (one device)
        const bool on_host = rc.buf[0].host();
        if (s < 0) {
            (on_host ? engine_host : engine_dev) += b;
        } else {
            (on_host ? host : dev)[size_t(s)] += b;
        }
    };
    for (const auto & rc : m.state) {
        add(*rc, fn::model::slot_class(rc->plan.name) ? 0 : -1);
    }
    for (size_t s = 0; s < m.slot_state.size(); ++s) {
        for (const auto & rc : m.slot_state[s]) {
            add(*rc, int32_t(s) + 1);
        }
    }
    for (int32_t s = 0; s < n; ++s) {
        const fn::runtime::target & t = const_cast<fn::runtime::engine &>(rt).test_target(s);
        graphs[size_t(s)] = t.graph_bytes(0) + t.graph_bytes(1);
    }
    for (int32_t s = 0; s < n; ++s) {
        fprintf(stderr,
                "FNFL: MXMU1 ledger: slot %d: host %.2f MiB (pinned: its whole history: target and draft KV, index "
                "keys); device %.2f MiB (its frame: GDN state and histories, round and pass, draft round, sampler, "
                "summaries, draft index; its window: the KV tier and the index key windows); graphs %.1f MiB\n",
                s, double(host[size_t(s)]) / 1048576.0, double(dev[size_t(s)]) / 1048576.0,
                double(graphs[size_t(s)]) / 1048576.0);
    }
    const auto * xc = dynamic_cast<const fn::runtime::expert_cache *>(rt.experts());
    size_t f = 0, tot = 0;
    cudaSetDevice(m.device[0]);
    cudaMemGetInfo(&f, &tot);
    const int64_t slot1 = n > 1 ? dev[1] + graphs[1] : 0;
    fprintf(stderr,
            "FNFL: MXMU1 ledger: %d slots on device %d; the engine's classes %.2f MiB device, %.2f MiB host; the "
            "expert cache %.2f MiB on the device (its pool %.2f MiB); free after the load %.3f GiB, after the engine "
            "%.3f GiB; the pool is %.2f MiB smaller than a one-slot launch's (slot 1's device classes and graphs, "
            "counted); no allocation follows at a job's begin or end\n",
            n, m.device[0], double(engine_dev) / 1048576.0, double(engine_host) / 1048576.0,
            xc != nullptr ? double(xc->device_bytes(0)) / 1048576.0 : 0.0,
            xc != nullptr ? double(xc->pool_bytes()) / 1048576.0 : 0.0, double(lr.free_after[0]) / 1073741824.0,
            double(f) / 1073741824.0, double(slot1) / 1048576.0);
}

fnfl_engine * fnfl_load(const fnfl_load_params & params, fnfl_model_meta & meta, std::string & err) {
    // refused before anything is read or loaded. MU1: 2..4 slots on resident TP (Config A); the expert-cached
    // configurations and the host KV serve one slot
    const bool cached_cfg = params.config.single_card >= 0 || params.config.cached_tp || params.config.kv_host;
    if (params.n_parallel < 1 || params.n_parallel > 4) {
        err =
            "FNFL serves 1 to 4 slots: the slot count is " + std::to_string(params.n_parallel) + " (-np / --parallel)";
        return nullptr;
    }
    // MXMU1: two slots on one card with the host-backed KV; every other cached or host-KV launch serves one slot
    if (params.n_parallel > 1 && cached_cfg) {
        const fnfl_config & cf = params.config;
        if (cf.cached_tp) {
            err =
                "FNFL serves one slot with --fnfl-cached-tp (multi-user cached TP is not built); launch it with -np 1";
        } else if (cf.single_card < 0) {
            err = "FNFL serves one slot with --fnfl-kv-host on resident TP; launch it with -np 1";
        } else if (!cf.kv_host) {
            err = "FNFL serves two slots on one card only with --fnfl-kv-host (each slot's history in host memory); "
                  "add it or launch with -np 1";
        } else if (params.n_parallel > 2) {
            err = "FNFL serves at most two slots on one card (MXMU1): the slot count is " +
                  std::to_string(params.n_parallel) + " (-np / --parallel)";
        }
        if (!err.empty()) {
            return nullptr;
        }
    }
    if (params.draft_path.empty()) {
        err = "FNFL needs the MTP draft (-md): speculation is the served path, there is no draft-less mode";
        return nullptr;
    }
    // KV8: the 8-bit KV, resident TP on two cards only (Config A), and its default there (Amendment 1); one card and
    // the expert-cached configurations keep BF16
    fnfl_kv_choice kvc;
    if (!fnfl_kv_format(params.config, kvc, err)) {
        return nullptr;
    }
    const bool kv_q8 = kvc.q8;
    fnfl_gguf_meta g;
    if (!fnfl_gguf_read_meta(params.model_path, g, err)) {
        return nullptr;
    }
    err = check_shapes(g);
    if (!err.empty()) {
        return nullptr;
    }
    meta = fnfl_model_meta();
    meta.arch = g.arch;
    meta.name = g.name;
    meta.chat_template = g.chat_template;
    meta.chat_template_tool_use = g.chat_template_tool_use;
    meta.format_name = g.format_name;
    meta.n_ctx_train = (int32_t)g.n_ctx_train;
    meta.n_embd = (int32_t)g.n_embd;
    meta.n_params = g.n_params;
    meta.size = g.size;

    // one slot: the context, at most the training context. MU1: with more slots no split: the context is the shared
    // KV pool's cells and every slot's logical context, so a lone user has all of it
    const int32_t n_ctx = params.n_ctx > 0 ? params.n_ctx : meta.n_ctx_train;
    meta.n_ctx_slot = std::min(n_ctx, meta.n_ctx_train);
    meta.n_slots = params.n_parallel;
    if (params.n_parallel > 1 && !cached_cfg && meta.n_ctx_slot % fn::model::kv_page_cells != 0) {
        err = "FNFL with more than one slot (MU1) needs a context (-c) that is a multiple of " +
              std::to_string(fn::model::kv_page_cells) + " cells (the KV pool's page); it is " +
              std::to_string(meta.n_ctx_slot);
        return nullptr;
    }
    // KV8: in q8 the shared pool is the whole -c (up to the 8-bit cells' room), each slot's context still at most
    // the training context; in bf16 the pool stays a slot's
    const int32_t pool_cells = kv_q8 && params.n_parallel > 1 ? n_ctx : 0;
    if (pool_cells % fn::model::kv_page_cells != 0) {
        err = "--fnfl-kv-format q8 with more than one slot needs a context (-c) that is a multiple of " +
              std::to_string(fn::model::kv_page_cells) + " cells (the KV pool's page); it is " + std::to_string(n_ctx);
        return nullptr;
    }

    fn::runtime::engine_config c;
    c.model_path = params.model_path;
    const int32_t one = params.config.single_card;
    const bool cached = one >= 0 || params.config.cached_tp;
    // ME3: the settings table (fn/cache_settings.h), as the driver parses it, read back at startup
    fn::settings::values sv;
    std::string sv_text;
    if (!fnfl_settings(params.config, sv, sv_text, err)) {
        return nullptr;
    }
    fprintf(stderr, "FNFL: %s", sv_text.c_str());
    c.eager = sv.executor == "eager"; // ME3: --fnfl-expert-executor (the graphs unless eager)
    c.short_read = cached ? sv.short_read : 0;
    c.prompt_chunk = sv.prompt_chunk;
    if (cached) { // ME1: one device or cached TP, the routed experts from the expert cache
        const fnfl_config cf = params.config;
        c.experts = [cf, sv](fn::model::model & mm, const fn::model::service_hook & hook,
                             std::string & e2) -> std::unique_ptr<fn::model::expert_service> {
            fn::runtime::expert_cache_options o;
            o.reserve_bytes = int64_t(cf.expert_reserve_mib) << 20; // ME3: --fnfl-expert-reserve-mib
            fn::runtime::apply_settings(sv, o);
            o.pool_bytes = cf.expert_pool_mib >= 0 ? int64_t(cf.expert_pool_mib) << 20 : -1;
            o.device_cap_bytes = cf.device_cap_mib >= 0 ? int64_t(cf.device_cap_mib) << 20 : -1;
            o.cpu_workers = cf.expert_cpu;
            o.profile = cf.expert_profile;
            o.profile_save = cf.expert_profile_save;
            o.swap_overlap = cf.expert_swap_overlap;
            o.before_pool = hook;
            auto x = std::make_unique<fn::runtime::expert_cache>();
            if (!x->init(mm, o, e2)) {
                return nullptr;
            }
            x->set_settings_json(fn::settings::json(sv));
            fprintf(stderr, "FNFL: %s: %s", cf.single_card >= 0 ? "--fnfl-single-card" : "--fnfl-cached-tp",
                    x->plan_text().c_str());
            return x;
        };
    }
    c.n_ctx_slot = meta.n_ctx_slot;
    c.draft.on = true;
    // SP2's switch: dense (DA-TC) unless the server names sparse
    c.draft.attention = params.config.draft_attention.empty() ? "dense" : params.config.draft_attention;
    c.spec_d = params.config.spec_d;
    c.draft.subset = params.config.draft_subset.empty() ? default_subset : params.config.draft_subset;
    c.eog = params.eog_ids;
    for (const int32_t id : params.config.test_eog) {
        if (std::find(c.eog.begin(), c.eog.end(), id) == c.eog.end()) {
            c.eog.push_back(id);
        }
    }
    c.test_fail_step = params.config.test_fail_step;
    c.temp = params.temperature;
    c.top_k = params.top_k;
    c.top_p = params.top_p;
    c.min_p = params.min_p;
    c.log_steps = params.config.check_state;

    fn::loader::load_params lp;
    lp.model_path = params.model_path;
    lp.draft_path = params.draft_path;
    if (one >= 0) {
        lp.device[0] = one;
        lp.device[1] = one;
    }
    lp.expert_cache = cached;
    lp.kv_host = params.config.kv_host;
    lp.n_slots = params.n_parallel; // MU1
    lp.kv_window = sv.kv_window; // ME3: --fnfl-kv-window
    lp.kv_q8 = kv_q8;            // KV8
    lp.pool_cells = pool_cells;
    fprintf(stderr, "FNFL: --fnfl-kv-format %s%s\n", kvc.line.c_str(),
            kv_q8 ? "; KV8: int8 K and V codes, an FP16 scale a group of 64; the draft KV and index keys BF16/F16"
                  : "");
    if (!params.config.ple_table.empty() && !fn::runtime::ple_residency_parse(params.config.ple_table, lp.ple_table)) {
        err = "--fnfl-ple-table is lazy or full, not '" + params.config.ple_table + "'";
        return nullptr;
    }
    fn::loader::run_load_report lr;
    auto m = std::make_unique<fn::model::model>();
    if (!fn::loader::load_for_run(lp, c.n_ctx_slot, *m, lr, err)) {
        err = "load: " + err;
        return nullptr;
    }
    fprintf(stderr,
            "FNFL: model loaded in %.2f s (open %.2f, fill %.2f, reserve %.2f) at %d cells; free %.3f / %.3f GiB\n",
            lr.total_s, lr.open_s, lr.fill_s, lr.reserve_s, c.n_ctx_slot, double(lr.free_after[0]) / 1073741824.0,
            double(lr.free_after[1]) / 1073741824.0);
    fprintf(stderr, "FNFL: --fnfl-ple-table %s%s\n", fn::runtime::ple_residency_name(lp.ple_table),
            params.config.ple_table.empty() ? " (default)" : "");
    for (const std::string & line : fn::loader::ple_table_lines(lr)) {
        fprintf(stderr, "FNFL: %s\n", line.c_str());
    }
    fprintf(stderr, "FNFL: %s\n", fn::loader::draft_expert_line(*m).c_str()); // ME7
    auto e = std::make_unique<fnfl_engine>();
    if (!e->rt.init(std::move(m), c, err)) {
        err = "engine: " + err;
        return nullptr;
    }
    fprintf(stderr, "%s; state check %s (SV1 test switch)\n", e->rt.settings().c_str(),
            params.config.check_state ? "on" : "off");
    if (params.n_parallel > 1 && e->rt.test_model().pool != nullptr) {
        startup_ledger(e->rt.test_model(), e->rt, lr);
        meta.pool_cells = e->rt.test_model().pool->capacity_cells();
        meta.pool_page_cells = e->rt.test_model().pool->page_cells();
    } else if (params.n_parallel > 1) {
        startup_ledger_one_card(e->rt.test_model(), e->rt, lr); // MXMU1: no pool (each slot's KV its own)
    }
    fflush(stderr);
    return e.release();
}

fnfl_status fnfl_set_sampling(fnfl_engine * engine, int32_t slot, const fnfl_sampling & params, std::string & err) {
    if (engine == nullptr) {
        err = "no engine";
        return FNFL_BAD_REQUEST;
    }
    fn::runtime::request_options o;
    o.sampling.temp = params.temperature;
    o.sampling.top_k = params.top_k;
    o.sampling.top_p = params.top_p;
    o.sampling.min_p = params.min_p;
    o.sampling.penalty_last_n = 0; // penalties are refused by the shell; off
    o.sampling.seed = params.seed;
    o.sampling.request = 0;
    o.max_tokens = params.max_tokens;
    o.n_ctx = params.n_ctx;
    const fnfl_status st = to_status(engine->rt.set_sampling(slot, o));
    if (st != FNFL_OK) {
        err = engine->rt.error();
    }
    return st;
}

fnfl_status fnfl_prefill(fnfl_engine * engine, int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                         int32_t * n_processed, fnfl_progress progress, void * user) {
    if (engine == nullptr) {
        return FNFL_BAD_REQUEST;
    }
    int32_t n = 0;
    fn::runtime::prefill_progress on_chunk;
    if (progress != nullptr) {
        on_chunk = [progress, user](int64_t from, int64_t done, int64_t total) {
            return progress(user, static_cast<int32_t>(from), static_cast<int32_t>(done), static_cast<int32_t>(total));
        };
    }
    const fnfl_status st = to_status(engine->rt.prefill(slot, tokens, n_tokens, n_reuse, n, on_chunk));
    if (n_processed != nullptr) {
        *n_processed = n;
    }
    return st;
}

fnfl_status fnfl_step(fnfl_engine * engine, int32_t slot, fnfl_step_result & result) {
    result = fnfl_step_result();
    if (engine == nullptr) {
        return FNFL_BAD_REQUEST;
    }
    return to_status(engine->rt.step(slot, result.tokens, result.n_tokens, result.end));
}

fnfl_status fnfl_step_joint(fnfl_engine * engine, const int32_t * slots, int32_t n, fnfl_step_result * results,
                            fnfl_status * status, std::string * errors) {
    if (engine == nullptr || slots == nullptr || results == nullptr || status == nullptr || errors == nullptr ||
        n < 1 || n > 4) {
        return FNFL_BAD_REQUEST;
    }
    fn::runtime::joint_step_out out[4];
    engine->rt.step_joint(slots, n, out);
    for (int32_t k = 0; k < n; ++k) {
        results[k] = fnfl_step_result();
        for (int32_t j = 0; j < out[k].n_out; ++j) {
            results[k].tokens[j] = out[k].out[j];
        }
        results[k].n_tokens = out[k].n_out;
        results[k].end = out[k].end;
        status[k] = to_status(out[k].st);
        errors[k] = out[k].why;
    }
    return FNFL_OK;
}

fnfl_status fnfl_truncate(fnfl_engine * engine, int32_t slot, int32_t n_keep) {
    if (engine == nullptr) {
        return FNFL_BAD_REQUEST;
    }
    return to_status(engine->rt.truncate(slot, n_keep));
}

std::string fnfl_last_error(const fnfl_engine * engine) {
    return engine == nullptr ? std::string("no engine") : engine->rt.error();
}

fnfl_status fnfl_slot_tokens(const fnfl_engine * engine, int32_t slot, std::vector<int32_t> & tokens,
                             int32_t & pending) {
    if (engine == nullptr) {
        return FNFL_BAD_REQUEST;
    }
    return to_status(engine->rt.history(slot, tokens, pending));
}

void fnfl_release(fnfl_engine * engine) {
    delete engine;
}

// ---- MU1 ----

int32_t fnfl_read_begin(fnfl_engine * engine, int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                        int32_t * n_processed) {
    if (engine == nullptr) {
        return -1;
    }
    int32_t n = 0;
    const int32_t r = engine->rt.read_begin(slot, tokens, n_tokens, n_reuse, n);
    if (n_processed != nullptr) {
        *n_processed = n;
    }
    return r;
}

fnfl_read_status fnfl_read_step(fnfl_engine * engine, int32_t slot, int32_t n_groups, fnfl_read_mode mode) {
    if (engine == nullptr) {
        return FNFL_READ_ERROR;
    }
    const fn::runtime::read_mode m = mode == FNFL_READ_LONE      ? fn::runtime::read_mode::lone
                                     : mode == FNFL_READ_WRITERS ? fn::runtime::read_mode::writers
                                                                 : fn::runtime::read_mode::shared;
    switch (engine->rt.read_step(slot, n_groups, m)) {
        case fn::runtime::read_status::yielded:
            return FNFL_READ_YIELDED;
        case fn::runtime::read_status::chunk_done:
            return FNFL_READ_CHUNK_DONE;
        case fn::runtime::read_status::done:
            return FNFL_READ_DONE;
        case fn::runtime::read_status::cancelled:
            return FNFL_READ_CANCELLED;
        case fn::runtime::read_status::error:
        default:
            return FNFL_READ_ERROR;
    }
}

void fnfl_read_cancel(fnfl_engine * engine, int32_t slot) {
    if (engine != nullptr) {
        engine->rt.read_cancel(slot);
    }
}

bool fnfl_read_info(const fnfl_engine * engine, int32_t slot, fnfl_read_report & info) {
    info = fnfl_read_report();
    fn::runtime::read_info r;
    if (engine == nullptr || !engine->rt.get_read_info(slot, r)) {
        return false;
    }
    info.live = r.live;
    info.cancel = r.cancel;
    info.total = static_cast<int32_t>(r.total);
    info.from = static_cast<int32_t>(r.from);
    info.committed = static_cast<int32_t>(r.committed);
    info.chunk = r.chunk;
    info.chunks = r.chunks;
    info.next_rows = r.next_rows;
    info.in_chunk = r.in_chunk;
    info.lane = r.lane;
    info.group = r.group;
    info.groups = r.groups;
    info.layers = r.layers;
    info.groups_done = r.groups_done;
    info.steps = r.steps;
    info.chunks_done = r.chunks_done;
    info.checkpoints = r.checkpoints;
    info.service_ms = r.service_s * 1e3;
    info.longest_step_ms = r.longest_step_s * 1e3;
    info.last_group_ms = r.last_group_s < 0 ? -1.0 : r.last_group_s * 1e3;
    return true;
}

bool fnfl_read_prefetch(fnfl_engine * engine, int32_t now_slot, int32_t next_slot) {
    return engine != nullptr && engine->rt.read_prefetch(now_slot, next_slot);
}

bool fnfl_resident_tp(const fnfl_engine * engine) {
    return engine != nullptr && engine->rt.resident_tp();
}

bool fnfl_set_read_group(fnfl_engine * engine, int32_t n_layers) {
    return engine != nullptr && engine->rt.set_read_group(n_layers);
}

int32_t fnfl_lane_rows(const fnfl_engine * engine, int32_t lane) {
    return engine == nullptr ? 0 : engine->rt.lane_rows(lane);
}

int32_t fnfl_lane_owner(const fnfl_engine * engine, int32_t lane) {
    return engine == nullptr ? -1 : engine->rt.lane_owner(lane);
}

fnfl_status fnfl_release_slot(fnfl_engine * engine, int32_t slot) {
    return engine == nullptr ? FNFL_BAD_REQUEST : to_status(engine->rt.release_slot(slot));
}

int32_t fnfl_slot_held(const fnfl_engine * engine, int32_t slot) {
    return engine == nullptr ? 0 : static_cast<int32_t>(engine->rt.held(slot));
}

bool fnfl_pool(const fnfl_engine * engine, fnfl_pool_report & out) {
    out = fnfl_pool_report();
    fn::runtime::pool_info p;
    if (engine == nullptr || !engine->rt.get_pool_info(p)) {
        return false;
    }
    out.on = p.on;
    out.pages = p.pages;
    out.page_cells = p.page_cells;
    out.free_pages = p.free_pages;
    out.reclaims = p.reclaims;
    out.capacity_cells = p.capacity_cells;
    for (int32_t i = 0; i < 4; ++i) {
        out.mapped[i] = p.mapped[i];
        out.busy[i] = p.busy[i];
    }
    return true;
}
