// fnfl.h for the server: checks the GGUF against FNFL's model, loads it on both cards (the loader fnfl-drive uses) and
// drives the runtime's engine (fn/runtime/runtime.h, SV1).

#include "fn/borrowed/gguf_meta.h"
#include "fn/fnfl.h"
#include "fn/loader/run_load.h"
#include "fn/model/kv_pool.h"
#include "fn/model/model.h"
#include "fn/model/prompt.h"
#include "fn/model/shapes.h"
#include "fn/runtime/expert_cache.h"
#include "fn/runtime/nvtx.h"
#include "fn/runtime/ple_host.h"
#include "fn/runtime/runtime.h"
#include "fn/runtime/stage.h"
#include "fn/vision/image.h"
#include "fn/xchg/link.h"
#include "fn/xchg/slow.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <memory>

static_assert(sizeof(fnfl_step_result::tokens) / sizeof(int32_t) == fn::runtime::max_step_tokens,
              "fnfl.h's step result must hold a step");

namespace {

// the draft subset when the server names none (DR1 Stage C's)
constexpr const char * default_subset = "~/src/flashnext/work/DR1/stageC/subset/draft-subset.ids";

} // namespace

// TTTS1: the split's leader side: the link, its port, the boundary buffers, the prompt crossing, the leader and its
// pipeline (fn/runtime/stage.h). Declared after rt in fnfl_engine, so it goes first (X-BYE, the link closed)
struct fnfl_split_side {
    fn::xchg::link::link                          link;
    std::unique_ptr<fn::runtime::stage::port>     port;
    fn::runtime::stage::buffers                   buf;
    fn::runtime::stage::crossing                  cross;
    fn::runtime::stage::leader                    lead;
    std::unique_ptr<fn::runtime::stage::pipeline> pipe;

    ~fnfl_split_side() {
        if (pipe != nullptr) {
            pipe->settle_all();
            lead.bye();
        }
        // TTTSBS: the link stops before the members go (every operation ended, the ranges unmapped after the
        // transport's quiescence): the crossing's and the slots' pinned buffers are freed with no operation on them
        link.stop();
    }
};

struct fnfl_engine {
    fn::runtime::engine rt;
    std::unique_ptr<fnfl_split_side> split;
    // APF1: the last paced read step's decision (fnfl_pace_last), and when it was taken (the books' clock)
    fn::runtime::pace::decision      last_pace;
    double                           last_pace_t = -1;
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
    // TTTS1: the split (--fnfl-split-layer K) serves 1 to 8 slots on resident TP on each box
    const int32_t split_k    = params.config.split_layer;
    if (split_k != 0 && (split_k < 2 || split_k > fn::n_layer - 1 || cached_cfg)) {
        err = "--fnfl-split-layer is 2 .. " + std::to_string(fn::n_layer - 1) +
              " on resident TP (not with --fnfl-single-card, --fnfl-cached-tp or --fnfl-kv-host); it is " +
              std::to_string(split_k);
        return nullptr;
    }
    const int32_t max_slots = split_k != 0 ? fn::runtime::max_split_slots : 4;
    if (params.n_parallel < 1 || params.n_parallel > max_slots) {
        err = "FNFL serves 1 to " + std::to_string(max_slots) + " slots" + (split_k != 0 ? " on the split" : "") +
              ": the slot count is " + std::to_string(params.n_parallel) + " (-np / --parallel)";
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
    c.small_read_rows = params.config.small_read_rows;  // MUP1
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
    lp.mmproj       = params.config.mmproj;  // VIS1 Stage B
    c.mmproj        = params.config.mmproj;
    auto e          = std::make_unique<fnfl_engine>();
    if (split_k != 0) {
        // TTTS1: this box the front: layers 0 .. k - 1 (with the embeddings; the head and the draft are the back's)
        lp.win.lo   = 0;
        lp.win.hi   = split_k - 1;
        lp.win.head = false;
        e->split    = std::make_unique<fnfl_split_side>();
        if (!e->split->buf.alloc(params.n_parallel, nullptr, err)) {
            err = "the split's boundary buffers: " + err;
            return nullptr;
        }
        c.stage     = 1;
        c.stage_buf = e->split->buf.slot;
    }
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
    // STALL: a lazy table is read in behind the server, so a prompt's gather does not wait on the disk
    fprintf(stderr, "FNFL: --fnfl-ple-warm %s\n", params.config.ple_warm ? "on (default)" : "off");
    if (lp.ple_table == fn::runtime::ple_residency::lazy && params.config.ple_warm && m->ple_table != nullptr &&
        m->ple_bytes > 0) {
        auto warm = std::make_shared<fn::runtime::ple_warm>();
        if (warm->start(m->ple_table, m->ple_bytes, true)) {
            m->ple_warm = warm;
        }
    }
    fprintf(stderr, "FNFL: %s\n", fn::loader::draft_expert_line(*m).c_str()); // ME7
    fprintf(stderr, "FNFL: %s\n", fn::loader::router_f16_line(*m).c_str());   // FMS5
    if (!e->rt.init(std::move(m), c, err)) {
        err = "engine: " + err;
        return nullptr;
    }
    if (split_k != 0) {
        // TTTS1: X-HELLO with the follower (connecting while it loads), the buffers mapped, the crossing on the
        // prefiller, the leader and its pipeline
        fnfl_split_side & s = *e->split;
        namespace lk        = fn::xchg::link;
        namespace st        = fn::runtime::stage;
        lk::options lo;
        lo.leader    = true;
        lo.addr      = params.config.split_addr;
        lo.port      = static_cast<uint16_t>(params.config.split_port);
        lo.timeout_s = 60.0;
        lo.connect_s = 1800.0;
        lk::set_text(lo.me.commit, sizeof lo.me.commit, params.config.split_commit);
        lk::set_text(lo.me.digest, sizeof lo.me.digest, params.config.split_digest);
        const std::string settings = st::hello_settings(split_k, params.n_parallel, c.n_ctx_slot, pool_cells, kv_q8,
                                                        c.spec_d, c.draft.attention, c.eog, !c.mmproj.empty());
        lk::set_text(lo.me.model, sizeof lo.me.model,
                     st::hello_files(e->rt.test_model()) + "/" + st::hello_hash(settings));
        lo.me.split = split_k;
        lo.me.slots = params.n_parallel;
        fprintf(stderr,
                "FNFL: TTTS1 split at k %d: this box the front (layers 0 .. %d), the follower the back at %s:%d; the "
                "shared settings: %s; UCX_TLS=%s UCX_NET_DEVICES=%s\n",
                split_k, split_k - 1, lo.addr.c_str(), int(lo.port), settings.c_str(),
                getenv("UCX_TLS") ? getenv("UCX_TLS") : "(unset)",
                getenv("UCX_NET_DEVICES") ? getenv("UCX_NET_DEVICES") : "(unset)");
        fflush(stderr);
        if (!s.link.start(lo, err)) {
            err = "the split's link: " + err;
            return nullptr;
        }
        s.port = st::link_port(s.link);
        for (auto & sl : s.buf.slot) {
            for (uint8_t * p : sl) {
                if (!s.port->map(p, size_t(fn::model::stage_bytes(fn::round::max_rows)), err)) {
                    err = "the split's buffers: " + err;
                    return nullptr;
                }
            }
        }
        if (!s.cross.init(*s.port, true, split_k, err)) {
            err = "the split's crossing: " + err;
            return nullptr;
        }
        e->rt.prefill_driver().set_stage(&s.cross);
        if (!s.lead.init(e->rt, *s.port, s.cross, err)) {
            err = "the split's leader: " + err;
            return nullptr;
        }
        s.pipe = std::make_unique<st::pipeline>(s.lead);
        fprintf(stderr,
                "FNFL: TTTS1 split: connected (X-HELLO: commit %s, device code %s); transport %s; two pipelined "
                "groups of up to four; one read at a time (no other-reader lane on the split)\n",
                params.config.split_commit.empty() ? "(none given)" : params.config.split_commit.c_str(),
                params.config.split_digest.empty() ? "(none given)" : params.config.split_digest.c_str(),
                s.link.transport().c_str());
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
    o.ignore_eos = params.ignore_eos;
    if (engine->split != nullptr) {
        const fnfl_status st = to_status(engine->split->pipe->set_sampling(slot, o));
        if (st != FNFL_OK) {
            err = engine->split->lead.error();
        }
        return st;
    }
    const fnfl_status st = to_status(engine->rt.set_sampling(slot, o));
    if (st != FNFL_OK) {
        err = engine->rt.error();
    }
    return st;
}

struct fnfl_image {
    std::shared_ptr<const fn::vision::prepared> p;
};

int32_t fnfl_image_token() {
    return fn::runtime::image_token;
}

fnfl_image * fnfl_image_prepare(const uint8_t *           data,
                                size_t                    n,
                                const fnfl_image_limits & limits,
                                fnfl_image_info &         info,
                                std::string &             err) {
    info = fnfl_image_info();
    fn::vision::limits lim;
    lim.max_bytes              = limits.max_bytes;
    lim.max_decoded            = limits.max_decoded;
    lim.max_side               = limits.max_side;
    lim.max_images             = limits.max_images;
    auto                     p = std::make_shared<fn::vision::prepared>();
    const fn::vision::status s = fn::vision::prepare(data, n, lim, *p);
    if (s != fn::vision::status::ok) {
        err = fn::vision::status_text(s);
        return nullptr;
    }
    info.nx        = p->nx;
    info.ny        = p->ny;
    info.width     = p->src.w;
    info.height    = p->src.h;
    info.resized_w = p->size.w;
    info.resized_h = p->size.h;
    auto * out     = new fnfl_image;
    out->p         = std::move(p);
    return out;
}

void fnfl_image_free(fnfl_image * image) {
    delete image;
}

bool fnfl_vision(const fnfl_engine * engine) {
    return engine != nullptr && !engine->rt.config().mmproj.empty();
}

namespace {

// the read's images as the engine takes them (null: none)
bool prompt_images(const fnfl_prompt_image * images, int32_t n_images, std::vector<fn::runtime::prompt_image> & out) {
    out.clear();
    if (n_images < 0 || (n_images > 0 && images == nullptr)) {
        return false;
    }
    for (int32_t i = 0; i < n_images; ++i) {
        if (images[i].image == nullptr) {
            return false;
        }
        fn::runtime::prompt_image pi;
        pi.span.cell = images[i].cell;
        pi.img       = images[i].image->p;
        out.push_back(std::move(pi));
    }
    return true;
}

}  // namespace

fnfl_status fnfl_prefill(fnfl_engine * engine, int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                         int32_t * n_processed, fnfl_progress progress, void * user) {
    return fnfl_prefill_images(engine, slot, tokens, n_tokens, n_reuse, nullptr, 0, n_processed, progress, user);
}

fnfl_status fnfl_prefill_images(fnfl_engine *             engine,
                                int32_t                   slot,
                                const int32_t *           tokens,
                                int32_t                   n_tokens,
                                int32_t                   n_reuse,
                                const fnfl_prompt_image * images,
                                int32_t                   n_images,
                                int32_t *                 n_processed,
                                fnfl_progress             progress,
                                void *                    user) {
    std::vector<fn::runtime::prompt_image> im;
    if (engine == nullptr || !prompt_images(images, n_images, im)) {
        return FNFL_BAD_REQUEST;
    }
    const std::vector<fn::runtime::prompt_image> * imp = im.empty() ? nullptr : &im;
    int32_t n = 0;
    fn::runtime::prefill_progress on_chunk;
    if (progress != nullptr) {
        on_chunk = [progress, user](int64_t from, int64_t done, int64_t total) {
            return progress(user, static_cast<int32_t>(from), static_cast<int32_t>(done), static_cast<int32_t>(total));
        };
    }
    // TTTS1: the split reads the prompt on both boxes (no progress calls: its chunks cross to the follower)
    const fnfl_status st = engine->split != nullptr ?
                               to_status(engine->split->pipe->prefill(slot, tokens, n_tokens, n_reuse, n, imp)) :
                               to_status(engine->rt.prefill(slot, tokens, n_tokens, n_reuse, n, on_chunk, imp));
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
    if (engine->split != nullptr) {
        // one slot's step on the split: a group of one, synchronous
        fn::runtime::joint_step_out out;
        engine->split->pipe->step_one(slot, out);
        for (int32_t j = 0; j < out.n_out; ++j) {
            result.tokens[j] = out.out[j];
        }
        result.n_tokens = out.n_out;
        result.end      = out.end;
        return to_status(out.st);
    }
    return to_status(engine->rt.step(slot, result.tokens, result.n_tokens, result.end));
}

fnfl_status fnfl_step_joint(fnfl_engine * engine, const int32_t * slots, int32_t n, fnfl_step_result * results,
                            fnfl_status * status, std::string * errors) {
    if (engine == nullptr || slots == nullptr || results == nullptr || status == nullptr || errors == nullptr ||
        n < 1 || n > 4) {
        return FNFL_BAD_REQUEST;
    }
    if (engine->split != nullptr) {
        // TTTS8: the split steps its writers through fnfl_step_pipe (and a lone one through fnfl_step); a joint call
        // here would bypass the pipeline's finished steps, so it is refused, never run
        for (int32_t k = 0; k < n; ++k) {
            results[k] = fnfl_step_result();
            status[k]  = FNFL_BAD_REQUEST;
            errors[k]  = "fnfl_step_joint on the split: its writers step through fnfl_step_pipe";
        }
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
        results[k].end      = out[k].end;
        status[k]           = to_status(out[k].st);
        errors[k]           = out[k].why;
    }
    return FNFL_OK;
}

void fnfl_range_push(const char * name) {
    nvtxRangePushA(name);
}

void fnfl_range_pop() {
    nvtxRangePop();
}

bool fnfl_is_split(const fnfl_engine * engine) {
    return engine != nullptr && engine->split != nullptr;
}

fnfl_status fnfl_step_pipe(fnfl_engine *      engine,
                           const int32_t *    slots,
                           int32_t            n,
                           fnfl_step_result * results,
                           fnfl_status *      status,
                           std::string *      errors,
                           bool *             done) {
    if (engine == nullptr || engine->split == nullptr || slots == nullptr || results == nullptr || status == nullptr ||
        errors == nullptr || done == nullptr || n < 1 || n > fn::runtime::max_split_slots) {
        return FNFL_BAD_REQUEST;
    }
    fn::runtime::nvtx_range     nr("fnfl.step_pipe");
    fn::runtime::joint_step_out out[fn::runtime::max_split_slots];
    bool                        fin[fn::runtime::max_split_slots] = {};
    engine->split->pipe->step(slots, n, out, fin);
    for (int32_t k = 0; k < n; ++k) {
        results[k] = fnfl_step_result();
        done[k]    = fin[k];
        status[k]  = FNFL_OK;
        errors[k].clear();
        if (!fin[k]) {
            continue;
        }
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
    if (engine->split != nullptr) {
        return to_status(engine->split->pipe->truncate(slot, n_keep));
    }
    return to_status(engine->rt.truncate(slot, n_keep));
}

std::string fnfl_last_error(const fnfl_engine * engine) {
    if (engine != nullptr && engine->split != nullptr) {
        const std::string & e = engine->split->lead.error();
        return engine->split->lead.lost() ? engine->split->lead.lost_why() : e.empty() ? engine->rt.error() : e;
    }
    return engine == nullptr ? std::string("no engine") : engine->rt.error();
}

fnfl_status fnfl_slot_tokens(const fnfl_engine * engine, int32_t slot, std::vector<int32_t> & tokens,
                             int32_t & pending) {
    if (engine == nullptr) {
        return FNFL_BAD_REQUEST;
    }
    if (engine->split != nullptr) {
        // the slot's group in flight ends first (the shell checks between calls)
        engine->split->pipe->settle(slot);
    }
    return to_status(engine->rt.history(slot, tokens, pending));
}

void fnfl_release(fnfl_engine * engine) {
    delete engine;
}

// ---- MU1 ----

int32_t fnfl_read_begin(fnfl_engine * engine, int32_t slot, const int32_t * tokens, int32_t n_tokens, int32_t n_reuse,
                        int32_t * n_processed) {
    return fnfl_read_begin_images(engine, slot, tokens, n_tokens, n_reuse, nullptr, 0, n_processed);
}

int32_t fnfl_read_begin_images(fnfl_engine *             engine,
                               int32_t                   slot,
                               const int32_t *           tokens,
                               int32_t                   n_tokens,
                               int32_t                   n_reuse,
                               const fnfl_prompt_image * images,
                               int32_t                   n_images,
                               int32_t *                 n_processed) {
    std::vector<fn::runtime::prompt_image> im;
    if (engine == nullptr || !prompt_images(images, n_images, im)) {
        return -1;
    }
    const std::vector<fn::runtime::prompt_image> * imp = im.empty() ? nullptr : &im;
    int32_t                                        n   = 0;
    const int32_t r = engine->split != nullptr ?
                          engine->split->pipe->read_begin(slot, tokens, n_tokens, n_reuse, n, imp) :
                          engine->rt.read_begin(slot, tokens, n_tokens, n_reuse, n, imp);
    if (n_processed != nullptr) {
        *n_processed = n;
    }
    return r;
}

namespace {

fnfl_read_status status_of(fn::runtime::read_status st) {
    switch (st) {
        case fn::runtime::read_status::yielded:
            return FNFL_READ_YIELDED;
        case fn::runtime::read_status::chunk_done:
            return FNFL_READ_CHUNK_DONE;
        case fn::runtime::read_status::done:
            return FNFL_READ_DONE;
        case fn::runtime::read_status::cancelled:
            return FNFL_READ_CANCELLED;
        case fn::runtime::read_status::front_done:
            return FNFL_READ_FRONT_DONE;
        case fn::runtime::read_status::error:
        default:
            return FNFL_READ_ERROR;
    }
}

fnfl_read_status read_step_of(fnfl_engine *                    engine,
                              int32_t                          slot,
                              int32_t                          n_groups,
                              fn::runtime::read_mode           m,
                              const fn::runtime::read_budget * b) {
    return status_of(engine->split != nullptr ? engine->split->lead.read_step(slot, n_groups, m, b) :
                                                engine->rt.read_step(slot, n_groups, m, b));
}

}  // namespace

fnfl_read_status fnfl_read_step(fnfl_engine * engine, int32_t slot, int32_t n_groups, fnfl_read_mode mode) {
    if (engine == nullptr) {
        return FNFL_READ_ERROR;
    }
    const fn::runtime::read_mode m = mode == FNFL_READ_LONE    ? fn::runtime::read_mode::lone :
                                     mode == FNFL_READ_WRITERS ? fn::runtime::read_mode::writers :
                                                                 fn::runtime::read_mode::shared;
    return read_step_of(engine, slot, n_groups, m, nullptr);
}

bool fnfl_writer_pause_valid(int64_t pause_ms) {
    return fn::runtime::pace::pause_valid(pause_ms);
}

bool fnfl_set_writer_pause(fnfl_engine * engine, int32_t pause_ms) {
    if (engine == nullptr || !fn::runtime::pace::pause_valid(pause_ms)) {
        return false;
    }
    engine->rt.pacer().set_pause(pause_ms);
    if (engine->split != nullptr) {
        engine->split->pipe->pace().set_pause(pause_ms);  // the follower takes it from the next command
    }
    return true;
}

void fnfl_set_prompt_graphs(bool on) {
    fn::model::prompt_graphs(on);
}

bool fnfl_prompt_graphs() {
    return fn::model::prompt_graphs_on();
}

int32_t fnfl_writer_pause(const fnfl_engine * engine) {
    if (engine == nullptr) {
        return fn::runtime::pace::pause_default_ms;
    }
    return engine->split != nullptr ? engine->split->pipe->pace().pause() : engine->rt.pacer().pause();
}

fnfl_read_status fnfl_read_step_paced(fnfl_engine * engine, int32_t slot) {
    namespace pace = fn::runtime::pace;
    if (engine == nullptr) {
        return FNFL_READ_ERROR;
    }
    // the owner's book: the split's front (its pipeline), or resident TP's engine
    pace::pacer &            p = engine->split != nullptr ? engine->split->pipe->pace() : engine->rt.pacer();
    fn::runtime::read_budget b;
    b.pace_before                = p.totals();
    const double         now     = pace::clock_ms();
    const pace::decision d       = p.decide(now);
    b.budget_ms                  = d.budget_ms;
    b.known                      = d.known;
    b.pause_ms                   = p.pause();
    b.paced                      = true;
    b.pace_after                 = p.totals();
    engine->last_pace            = d;
    engine->last_pace_t          = now;
    // the read work inserted here, by kind: a small read whole, the progress unit alone past the budget, or units
    // under it
    const bool             small = engine->rt.small_read(slot);
    const fnfl_read_status st    = read_step_of(engine, slot, 1, fn::runtime::read_mode::writers, &b);
    fn::runtime::read_info ri;
    const bool             minimum = engine->rt.get_read_info(slot, ri) && ri.last_minimum;
    p.read_ran(now, pace::clock_ms(),
               small   ? pace::read_kind::small :
               minimum ? pace::read_kind::minimum :
                         pace::read_kind::budgeted);
    return st;
}

bool fnfl_pace_last(const fnfl_engine * engine, fnfl_pace_report & out) {
    out = fnfl_pace_report();
    if (engine == nullptr || engine->last_pace_t < 0) {
        return false;
    }
    const fn::runtime::pace::decision & d = engine->last_pace;
    out.known                             = d.known;
    out.budget_ms                         = d.budget_ms;
    out.slack_ms                          = d.slack_ms;
    out.pause_ms                          = d.pause_ms;
    out.group                             = d.group;
    out.phase                             = fn::runtime::pace::phase_name(d.ph);
    out.projected                         = d.projected;
    out.since_c_ms                        = d.c_ms >= 0 ? engine->last_pace_t - d.c_ms : -1.0;
    out.f_ms                              = d.f_ms;
    out.b_ms                              = d.b_ms;
    out.why                               = d.why;
    return true;
}

bool fnfl_read_ready(fnfl_engine * engine, int32_t slot) {
    return engine == nullptr || engine->split == nullptr || engine->split->lead.read_ready(slot);
}

fnfl_read_status fnfl_read_poll(fnfl_engine * engine, int32_t slot) {
    if (engine == nullptr || engine->split == nullptr) {
        return FNFL_READ_ERROR;  // only the split's leader returns FRONT_DONE
    }
    return status_of(engine->split->lead.read_poll(slot));
}

void fnfl_read_cancel(fnfl_engine * engine, int32_t slot) {
    if (engine != nullptr && engine->split != nullptr) {
        engine->split->lead.read_cancel(slot);
    } else if (engine != nullptr) {
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
    // MUP2 piece 4: the units account here (the front on the split) and the back's
    const auto units     = [](const fn::runtime::read_info & x, fnfl_read_units & u) {
        u.units       = x.units_run;
        u.steps       = x.unit_steps;
        u.most        = x.most_units;
        u.floor_units = x.floor_units;
        u.piece_min   = x.piece_min;
        u.piece_max   = x.piece_max;
        u.gpu_ms      = x.unit_ms;
        u.floor_ms    = x.floor_ms;
        u.budget_ms      = x.budget_n > 0 ? x.budget_ms_sum / x.budget_n : -1.0;
        u.service_ms  = x.service_s * 1e3;
        u.below_floor = x.below_floor;
        u.budget_misses = x.budget_misses;
        // APF1: P, the zero and unknown budgets, and the box's pacing over the read
        u.pause_ms       = x.pause_ms;
        u.budget_zero    = x.budget_zero;
        u.budget_unknown = x.budget_unknown;
        u.pace_steps     = x.pace_ns.steps;
        u.pace_known     = x.pace_ns.known;
        u.pace_zero      = x.pace_ns.zero;
        u.pace_unknown   = x.pace_ns.unknown;
        u.completions    = x.pace_ns.completions;
        u.gaps           = x.pace_ns.gaps;
        u.over           = x.pace_ns.over;
        u.over_none      = x.pace_ns.over_none;
        u.over_budgeted  = x.pace_ns.over_budgeted;
        u.over_minimum   = x.pace_ns.over_minimum;
        u.over_small     = x.pace_ns.over_small;
        u.over_idle      = x.pace_ns.over_idle;
        u.longest_gap_ms = x.pace_ns.longest_ms;
        u.over_ms        = x.pace_ns.over_ms_sum;
    };
    units(r, info.here);
    if (engine->split != nullptr) {
        fn::runtime::read_info b;
        engine->split->lead.back_info(slot, b);
        units(b, info.back);
    }
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
    // TTTS1: one read at a time on the split (the back runs its chunks in order, in the main lane)
    return engine == nullptr || (engine->split != nullptr && lane != 0) ? 0 : engine->rt.lane_rows(lane);
}

int32_t fnfl_lane_owner(const fnfl_engine * engine, int32_t lane) {
    return engine == nullptr ? -1 : engine->rt.lane_owner(lane);
}

fnfl_status fnfl_release_slot(fnfl_engine * engine, int32_t slot) {
    if (engine != nullptr && engine->split != nullptr) {
        return to_status(engine->split->pipe->release_slot(slot));
    }
    return engine == nullptr ? FNFL_BAD_REQUEST : to_status(engine->rt.release_slot(slot));
}

fnfl_status fnfl_set_pos_offset(fnfl_engine * engine, int32_t slot, int32_t off) {
    if (engine != nullptr && engine->split != nullptr) {
        return to_status(engine->split->pipe->set_pos_offset(slot, off));
    }
    return engine == nullptr ? FNFL_BAD_REQUEST : to_status(engine->rt.set_pos_offset(slot, off));
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
    for (int32_t i = 0; i < fn::runtime::max_split_slots; ++i) {
        out.mapped[i] = p.mapped[i];
        out.busy[i] = p.busy[i];
    }
    return true;
}

void fnfl_trace_sink(void (*sink)(const char * line)) {
    fn::xchg::slow::set_sink(sink);
}

void fnfl_trace_writers(uint64_t mask) {
    fn::xchg::slow::set_writers(mask);
}
