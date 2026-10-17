// fnfl.h for the server: checks the GGUF against FNFL's model, loads it on both cards (the loader fnfl-drive uses) and
// drives the runtime's engine (fn/runtime/runtime.h, SV1).

#include "fn/fnfl.h"

#include "fn/borrowed/gguf_meta.h"
#include "fn/loader/run_load.h"
#include "fn/model/model.h"
#include "fn/model/shapes.h"
#include "fn/runtime/runtime.h"

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

fnfl_engine * fnfl_load(const fnfl_load_params & params, fnfl_model_meta & meta, std::string & err) {
    // refused before anything is read or loaded
    if (params.n_parallel != 1) {
        err = "FNFL serves one slot: the slot count is " + std::to_string(params.n_parallel) +
              " (-np / --parallel); launch with -np 1";
        return nullptr;
    }
    if (params.draft_path.empty()) {
        err = "FNFL needs the MTP draft (-md): speculation is the served path, there is no draft-less mode";
        return nullptr;
    }
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

    // each slot gets its share of the context, at most the training context (no unified cache: FL2); one slot
    const int32_t n_ctx = params.n_ctx > 0 ? params.n_ctx : meta.n_ctx_train;
    meta.n_ctx_slot = std::min(n_ctx / params.n_parallel, meta.n_ctx_train);

    fn::runtime::engine_config c;
    c.model_path = params.model_path;
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
    auto e = std::make_unique<fnfl_engine>();
    if (!e->rt.init(std::move(m), c, err)) {
        err = "engine: " + err;
        return nullptr;
    }
    fprintf(stderr, "%s; state check %s (SV1 test switch)\n", e->rt.settings().c_str(),
            params.config.check_state ? "on" : "off");
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
